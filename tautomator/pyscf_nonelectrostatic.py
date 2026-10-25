"""The inputs to the nonelectrostatic terms, from a converged PySCF PCM calculation.

`solute_from_mf` takes the mean field `pyscf_runner._build_mf` builds for one of
the analytical PCM models (IEF-PCM, C-PCM, COSMO), FastPCM or PySCF's own, once
its SCF has converged, and returns the backend-neutral `nonelectrostatic.Solute`:

* the PCM surface and its apparent surface charges for the converged density,
  with the sphere radii the cavity was built from (`_build_mf` sets them; an
  ECP atom's radius goes by its true nuclear charge, as in
  `_patch_pcm_ecp_cavity`);
* the solute's own electric field (nuclei and electrons, no reaction field)
  normal to each surface element;
* the electron density and its gradient on `cavity_quadrature` grids for the
  inside and the outside of the cavity.

An ECP leaves only the valence density for the dispersion and exchange
integrals, and puts the ECP-reduced nuclear charges in the field.  At
def2-QZVPPD that affects iodine and heavier elements only.

`single_point_with_descriptors` is `pyscf_runner.single_point` with the
descriptors read off its mean field before it is discarded; scoring calls it
when run_pipeline is asked for descriptors (scripts/run_mnsol_set.py).

`NonelectrostaticTerms` is the correction itself, with the solute's own field in
the surface term, as a function of the density matrix at one geometry: the
energy, its potential dG/dD (for a self-consistent SCF, or as the right-hand
side of the Z-vector equation of a post-SCF gradient) and its explicit nuclear
gradient at fixed density matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import df, dft, gto, lib

from tautomator import pyscf_runner
from tautomator.cavity_quadrature import (
    INSIDE_GRID,
    OUTSIDE_GRID,
    CavityGrid,
    Grid,
    build_cavity_grid,
    cavity_grid,
    partition_gradient,
)
from tautomator.config import (
    CAVITIES,
    DEFAULT_MEMORY_GB,
    DROP_CAVITIES,
    solvent_electron_density,
)
from tautomator.dispersion_kernel import kernel_derivatives, kernel_integral
from tautomator.nonelectrostatic import (
    DELTA,
    VV_C,
    DensityGrid,
    Descriptors,
    Energies,
    Params,
    Solute,
    Surface,
    descriptors,
    dispersion_density,
    exchange_density,
    field_power,
    whole_space_kernel_integral,
)
from tautomator.types import Geometry

Array = NDArray[np.float64]

# Grid points per block when evaluating the density, and the memory (bytes) one
# block of field integrals may take.
_GRID_BLOCK = 20_000
_FIELD_BLOCK_BYTES = 200e6


def _charge_scale(method: str, eps: float) -> float:
    """How PySCF's PCM model scales a conductor's surface charges, for a point
    charge at the centre of a sphere (see `nonelectrostatic.Surface`).

    COSMO's f(eps) has 1/2 where C-PCM's has 0; IEF-PCM keeps (eps-1)/(eps+1)
    in its K and R matrices, but its charges come out at (eps-1)/eps there.
    """
    if method.upper() == "COSMO":
        return (eps - 1.0) / (eps + 0.5)
    return (eps - 1.0) / eps


def _point_charges(mol: Any, points: Array) -> Any:
    """Unit point charges at the points, as PySCF's integral code takes them."""
    fakemol = gto.fakemol_for_charges(points)
    fakemol.cart = mol.cart  # aux_e2 refuses a spherical fakemol with a Cartesian mol
    return fakemol


def _sphere_radii(mol: Any, ws: Any) -> Array:
    """The cavity's sphere radii (bohr), by true nuclear charge also for ECP atoms."""
    from pyscf.data import radii
    from pyscf.solvent import pcm

    table = ws.radii_table
    if table is None:
        table = ws.vdw_scale * pcm.modified_Bondi + ws.r_probe / radii.BOHR
    true_z = [gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)]
    return np.asarray([table[z] for z in true_z], dtype=np.float64)


def _element_atoms(ws: Any) -> NDArray[np.intp]:
    """The atom each surface element belongs to; each atom's elements are one slice."""
    sizes = [p1 - p0 for p0, p1 in ws.surface["gslice_by_atom"]]
    return np.repeat(np.arange(len(sizes)), sizes)


def _surface(mol: Any, ws: Any) -> Surface:
    """The PCM surface and the apparent surface charges the solvent object holds."""
    eps = float(ws.eps)
    return Surface(
        coords=np.asarray(ws.surface["grid_coords"], dtype=np.float64),
        normals=np.asarray(ws.surface["norm_vec"], dtype=np.float64),
        areas=np.asarray(ws.surface["area"], dtype=np.float64),
        charges=np.asarray(ws._intermediates["q"], dtype=np.float64),
        atom_coords=np.asarray(mol.atom_coords(unit="B"), dtype=np.float64),
        atom_radii=_sphere_radii(mol, ws),
        element_atoms=_element_atoms(ws),
        eps=eps,
        charge_scale=_charge_scale(ws.method, eps),
    )


def density_on_grid(mol: Any, dm: Array, coords: Array, weights: Array) -> DensityGrid:
    """Electron density and its gradient at the given quadrature points."""
    ni = dft.numint.NumInt()
    rho = np.empty((4, len(coords)))
    for p0 in range(0, len(coords), _GRID_BLOCK):
        ao = ni.eval_ao(mol, coords[p0 : p0 + _GRID_BLOCK], deriv=1)
        rho[:, p0 : p0 + _GRID_BLOCK] = ni.eval_rho(mol, ao, dm, xctype="GGA", hermi=1)
    return DensityGrid(
        coords=coords, weights=weights, rho=rho[0], grad=np.ascontiguousarray(rho[1:].T)
    )


def electric_field(mol: Any, dm: Array, points: Array) -> Array:
    """The solute's electric field (nuclei + electrons) at each point, shape (n, 3).

    The electronic potential at s is -sum D_mn (mn|s), and by translational
    invariance its gradient with respect to s is the integral with the
    derivative moved onto the orbitals, so E_elec = 2 sum D_mn (grad m n|s).
    Nuclei enter with the charges the electrons see (ECP-reduced, if any).
    """
    sep = points[:, None, :] - mol.atom_coords(unit="B")[None, :, :]
    dist = np.linalg.norm(sep, axis=2)
    field: Array = np.einsum("a,pax->px", mol.atom_charges(), sep / dist[:, :, None] ** 3)

    intor = mol._add_suffix("int3c2e_ip1")
    block = max(1, int(_FIELD_BLOCK_BYTES / (3 * 8 * mol.nao**2)))
    for p0 in range(0, len(points), block):
        fakemol = _point_charges(mol, points[p0 : p0 + block])
        ints = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s1", comp=3)
        field[p0 : p0 + block] += 2.0 * np.einsum("xijp,ij->px", ints, dm)
    return field


def solute_from_mf(
    mf: Any, inside_grid: Grid = INSIDE_GRID, outside_grid: Grid = OUTSIDE_GRID
) -> Solute:
    """Gather surface, density and normal field from a converged PCM mean field.

    ``inside_grid`` and ``outside_grid`` (Lebedev order, radial nodes) set the cavity
    quadrature the density is
    evaluated on (see `cavity_quadrature.cavity_grid`), not the PCM surface.
    """
    if not mf.converged:
        raise ValueError("the SCF is not converged")
    mol, ws = mf.mol, mf.with_solvent
    if not ws.surface or "q" not in (ws._intermediates or {}):
        raise ValueError("the PCM solvent object holds no converged surface charges")
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    if dm.ndim == 3:  # unrestricted: total density
        dm = dm[0] + dm[1]
    # The charges for this density.  Those the SCF left behind belong to the
    # last density it evaluated, which a Newton retry need not end on.
    ws._get_vind(dm)
    surface = _surface(mol, ws)
    field = electric_field(mol, dm, surface.coords)
    centres, radii = surface.atom_coords, surface.atom_radii
    inside = cavity_grid(centres, radii, "inside", *inside_grid)
    outside = cavity_grid(centres, radii, "outside", *outside_grid)
    return Solute(
        surface=surface,
        interior=density_on_grid(mol, dm, *inside),
        exterior=density_on_grid(mol, dm, *outside),
        normal_field=np.einsum("px,px->p", field, surface.normals),
    )


def single_point_with_descriptors(
    geom: Geometry,
    charge: int,
    method: str,
    basis: str,
    cavity: str,
    solvent: str,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> tuple[float, Descriptors]:
    """An IEF-PCM single point on the cavity, as `pyscf_runner.single_point` runs it, and
    its descriptors.

    Returns (total energy in hartree, descriptors at the default cavity
    quadrature, with the solvent's electron density in the dispersion term).  On one of
    MOIST's DROP cavities the descriptors are those of
    `pyscf_nonelectrostatic_drop.drop_descriptors`, on MOIST's surface.
    """
    if cavity not in CAVITIES:
        raise ValueError(f"no cavity {cavity!r} (has: {', '.join(CAVITIES)})")
    _mol, mf = pyscf_runner._build_mf(
        geom,
        charge,
        method,
        basis,
        cavity,
        solvent,
        pcm_hydrogen_radius,
        threads,
        memory_gb,
    )
    energy = pyscf_runner._run_scf_robust(mf)
    if cavity in DROP_CAVITIES:
        from tautomator.pyscf_nonelectrostatic_drop import drop_descriptors

        return energy, drop_descriptors(mf, solvent_electron_density(solvent))
    return energy, descriptors(solute_from_mf(mf), solvent_electron_density(solvent))


# ---------------------------------------------------------------------------
# The correction as a function of the density matrix
# ---------------------------------------------------------------------------

# Memory (bytes) one block of basis-function values or field integrals may take.
_AO_BLOCK_BYTES = 200e6


def _point_blocks(n_points: int, n_values: int) -> list[slice]:
    """Slices of at most _AO_BLOCK_BYTES worth of n_values float64 per point.

    No floor above one point: the field integrals take 3 or 9 nao^2 values per
    point, so at def2-QZVPPD a block holds only a few points.
    """
    size = max(1, int(_AO_BLOCK_BYTES / (8 * max(1, n_values))))
    return [slice(p0, min(n_points, p0 + size)) for p0 in range(0, n_points, size)]


_SECOND_DERIVATIVES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))  # eval_ao comps 4-9


def _ao_atoms(mol: Any) -> NDArray[np.intp]:
    out = np.empty(mol.nao, dtype=np.intp)
    for atom, (_s0, _s1, a0, a1) in enumerate(mol.aoslice_by_atom()):
        out[a0:a1] = atom
    return out


def basis_motion_gradient(
    mol: Any,
    ao: Array,
    dm: Array,
    coeff: Array,
    ao_atom: NDArray[np.intp] | None = None,
) -> Array:
    """sum_p sum_k coeff_k(p) d u_k(p)/dR as the basis functions move with their atoms.

    u = (rho, grad rho) of the symmetric density matrix ``dm`` at fixed points,
    ``coeff`` the matching per-point coefficients, shape (1, n) for rho alone or
    (4, n) with grad rho; ``ao`` from eval_ao with deriv 1 or 2 accordingly.
    Moving atom A shifts its functions, d phi_m/dR_A = -grad phi_m, so with
    c_m = sum_n D_mn phi_n and c_m,y = sum_n D_mn d_y phi_n:

        d rho / dR_A        = -2 sum_{m on A} grad phi_m c_m
        d (d_y rho) / dR_A  = -2 sum_{m on A} (grad d_y phi_m c_m + grad phi_m c_m,y)

    Returns (natm, 3).
    """
    if ao_atom is None:
        ao_atom = _ao_atoms(mol)
    natm = mol.natm
    coeff = np.asarray(coeff).reshape(-1, ao.shape[1])
    c0 = ao[0] @ dm
    # sum over points of ao_k(p, m) w(p, m), as elementwise products (einsum with three
    # operands runs single-threaded and without BLAS)
    w = coeff[0][:, None] * c0
    if len(coeff) == 4:
        c1 = ao[1:4] @ dm
        w = w + (coeff[1:, :, None] * c1).sum(axis=0)
    T = np.stack([(ao[1 + k] * w).sum(axis=0) for k in range(3)])
    if len(coeff) == 4:
        bc0 = coeff[1:, :, None] * c0[None]
        for (k, kk), comp in zip(_SECOND_DERIVATIVES, range(4, 10), strict=True):
            T[k] += (ao[comp] * bc0[kk]).sum(axis=0)
            if k != kk:
                T[kk] += (ao[comp] * bc0[k]).sum(axis=0)
    g = np.empty((natm, 3))
    for x in range(3):
        g[:, x] = -2.0 * np.bincount(ao_atom, T[x], minlength=natm)
    return g


def _nuclear_field(points: Array, centres: Array, charges: Array) -> tuple[Array, Array]:
    """The nuclei's field at each point (n, 3), and T = I/r^3 - 3 r r^T/r^5 per
    point and nucleus (n, natm, 3, 3): dE/ds = sum_B Z_B T_B, dE/dR_B = -Z_B T_B."""
    sep = points[:, None, :] - centres[None, :, :]
    dist = np.linalg.norm(sep, axis=2)
    field = np.einsum("a,pax->px", charges, sep / dist[:, :, None] ** 3)
    tensor = (
        np.eye(3)[None, None] / dist[:, :, None, None] ** 3
        - 3.0 * np.einsum("pax,pay->paxy", sep, sep) / dist[:, :, None, None] ** 5
    )
    return np.asarray(field), np.asarray(tensor)


class NonelectrostaticTerms:
    """The nonelectrostatic correction at one geometry, as a function of the density matrix.

    The surface term uses the solute's own normal field (``source="field"``).
    Everything takes the total density matrix (alpha + beta for an unrestricted
    calculation), symmetric, in the AO basis of ``mol``:

    * `energy(dm)`: the three contributions, as `Descriptors.energy` gives them
      for the same density;
    * `energy_and_potential(dm)`: also V = dG/dD, the matrix whose trace with a
      change in the density matrix is the change in G.  It is the same for both
      spins of an unrestricted calculation;
    * `gradient(dm)`: dG/dR with the density matrix held fixed: the basis
      functions, the cavity grids, the surface elements and the nuclei move
      with their atoms.  The density's own response is not included (see
      `pyscf_nonelectrostatic_gradient` for the full post-SCF gradient).

    ``ws`` is the PCM solvent object of a mean field `pyscf_runner._build_mf`
    built: only its cavity (surface, radii) is used, not its charges.  The
    cavity grids and the dispersion kernel on them depend only on the
    geometry, so they are built once, here.
    """

    def __init__(
        self,
        mol: Any,
        ws: Any,
        params: Params,
        solvent_rho: float,
        delta: float = DELTA,
        vv_c: float = VV_C,
        inside_grid: Grid = INSIDE_GRID,
        outside_grid: Grid = OUTSIDE_GRID,
        store_field_integrals: bool = False,
    ) -> None:
        if not ws.surface:
            ws.build()
        self.mol, self.ws, self.params = mol, ws, params
        self.solvent_rho, self.delta, self.vv_c = solvent_rho, delta, vv_c
        surface = ws.surface
        self.coords = np.ascontiguousarray(surface["grid_coords"], dtype=np.float64)
        self.normals = np.ascontiguousarray(surface["norm_vec"], dtype=np.float64)
        self.areas = np.ascontiguousarray(surface["area"], dtype=np.float64)
        self.element_atoms = _element_atoms(ws)
        self.centres = np.asarray(mol.atom_coords(unit="B"), dtype=np.float64)
        self.radii = _sphere_radii(mol, ws)
        self.inside = build_cavity_grid(self.centres, self.radii, "inside", *inside_grid)
        self.outside = build_cavity_grid(self.centres, self.radii, "outside", *outside_grid)
        self.kernel = whole_space_kernel_integral(delta) - kernel_integral(
            self.inside.coords, self.coords, self.normals, self.areas, delta
        )
        self._ao_atom = _ao_atoms(mol)
        # With store_field_integrals, n_i . E(s_i) as a linear map on the
        # packed density matrix is kept after its first use, if it fits in a
        # quarter of mol.max_memory: a self-consistent SCF needs it every
        # iteration, at the geometry the terms were built for.
        self.store_field_integrals = store_field_integrals
        self._field_matrix: Array | None = None

    # -- the pieces -----------------------------------------------------------

    def _grid_integrand(
        self, grid: CavityGrid, sl: slice, rho: Array, grad_sq: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Per point of one block: (e, de/drho, de/d|grad rho|^2, kernel factor),
        each already times its coefficient; the integrand is e * kernel."""
        p = self.params
        if grid.region == "inside":
            e, e_rho, e_sigma = dispersion_density(rho, grad_sq, self.solvent_rho, self.vv_c)
            return p.dispersion * e, p.dispersion * e_rho, p.dispersion * e_sigma, self.kernel[sl]
        e, e_sigma = exchange_density(grad_sq)
        ones = np.ones_like(e)
        return p.exchange * e, np.zeros_like(e), p.exchange * e_sigma, ones

    def _field_blocks(self) -> list[slice]:
        return _point_blocks(len(self.areas), 9 * self.mol.nao**2)

    def _surface_function(self, F: Array) -> tuple[Array, Array]:
        """f(F) and f'(F) per element: c- |F|^n- on negative, c+ F^n+ on positive."""
        p = self.params
        neg, d_neg = field_power(F, p.neg_exponent, -1)
        pos, d_pos = field_power(F, p.pos_exponent, 1)
        return p.neg_coeff * neg + p.pos_coeff * pos, p.neg_coeff * d_neg + p.pos_coeff * d_pos

    # -- energy and potential --------------------------------------------------

    def energy(self, dm: Array) -> Energies:
        return self.energy_and_potential(dm, potential=False)[0]

    def energy_and_potential(
        self, dm: Array, potential: bool = True
    ) -> tuple[Energies, Array | None]:
        mol = self.mol
        dm = np.asarray(dm, dtype=np.float64)
        ni = dft.numint.NumInt()
        V = np.zeros((mol.nao, mol.nao)) if potential else None
        grid_energy = {}
        for grid in (self.inside, self.outside):
            total = 0.0
            for sl in _point_blocks(len(grid.weights), 4 * mol.nao):
                ao = ni.eval_ao(mol, grid.coords[sl], deriv=1)
                rho = ni.eval_rho(mol, ao, dm, xctype="GGA", hermi=1)
                e, e_rho, e_sigma, kernel = self._grid_integrand(
                    grid, sl, rho[0], np.sum(rho[1:] ** 2, axis=0)
                )
                w = grid.weights[sl] * kernel
                total += float(w @ e)
                if V is not None:
                    wv = np.empty((4, len(w)))
                    wv[0] = 0.5 * w * e_rho
                    wv[1:] = 2.0 * w * e_sigma * rho[1:]
                    V += ao[0].T @ (ao * wv[:, :, None]).sum(axis=0)
            grid_energy[grid.region] = total
        if V is not None:
            V = V + V.T

        nuclear, _tensor = _nuclear_field(self.coords, self.centres, mol.atom_charges())
        nuclear_normal = np.einsum("px,px->p", nuclear, self.normals)
        stored = self._stored_field_matrix()
        if stored is not None:
            weight = lib.pack_tril(2.0 - np.eye(mol.nao))
            F = nuclear_normal + stored @ (lib.pack_tril(dm) * weight)
            f, f_prime = self._surface_function(F)
            short_range = float(self.areas @ f)
            if V is not None:
                V += lib.unpack_tril((self.areas * f_prime) @ stored)
        else:
            intor = mol._add_suffix("int3c2e_ip1")
            short_range = 0.0
            M = np.zeros((mol.nao, mol.nao)) if V is not None else None
            for sl in _point_blocks(len(self.areas), 3 * mol.nao**2):
                fakemol = _point_charges(mol, self.coords[sl])
                ints = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s1", comp=3)
                field = 2.0 * np.tensordot(
                    dm.ravel(), ints.reshape(3, -1, ints.shape[-1]), ([0], [1])
                )
                F = nuclear_normal[sl] + np.einsum("xp,px->p", field, self.normals[sl])
                f, f_prime = self._surface_function(F)
                short_range += float(self.areas[sl] @ f)
                if M is not None:
                    q = (self.areas[sl] * f_prime)[:, None] * self.normals[sl]
                    for x in range(3):
                        M += ints[x] @ q[:, x]
            if V is not None and M is not None:
                V += M + M.T
        energies = Energies(
            dispersion=grid_energy["inside"],
            exchange=grid_energy["outside"],
            short_range=short_range,
        )
        return energies, V

    def _stored_field_matrix(self) -> Array | None:
        """(n_elements, n_pairs): row i is n_i . [(grad m n|s_i) + (m grad n|s_i)], packed."""
        if not self.store_field_integrals:
            return None
        if self._field_matrix is None:
            mol = self.mol
            npair = mol.nao * (mol.nao + 1) // 2
            budget_mb = 0.25 * (mol.max_memory - lib.current_memory()[0])
            if len(self.areas) * npair * 8 / 1e6 > budget_mb:
                self.store_field_integrals = False
                return None
            intor = mol._add_suffix("int3c2e_ip1")
            stored = np.empty((len(self.areas), npair))
            for sl in _point_blocks(len(self.areas), 3 * mol.nao**2):
                fakemol = _point_charges(mol, self.coords[sl])
                ints = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s1", comp=3)
                block = np.moveaxis(sum(ints[x] * self.normals[sl, x] for x in range(3)), -1, 0)
                stored[sl] = lib.pack_tril(block + block.transpose(0, 2, 1))
            self._field_matrix = stored
        return self._field_matrix

    # -- explicit nuclear gradient ---------------------------------------------

    def gradient(self, dm: Array) -> Array:
        """dG/dR (natm, 3) at fixed density matrix."""
        mol = self.mol
        dm = np.asarray(dm, dtype=np.float64)
        natm = mol.natm
        g = np.zeros((natm, 3))
        ni = dft.numint.NumInt()

        # Dispersion and exchange: the integrand at each grid point is
        # e(rho, |grad rho|^2) * kernel, summed with the grid's weights.
        dispersion_values = np.zeros(len(self.inside.weights))
        for grid in (self.inside, self.outside):
            values = np.zeros(len(grid.weights))
            for sl in _point_blocks(len(grid.weights), 14 * mol.nao):
                ao = ni.eval_ao(mol, grid.coords[sl], deriv=2)
                c0 = ao[0] @ dm
                c1 = ao[1:4] @ dm
                rho = (ao[0] * c0).sum(axis=1)
                grad = 2.0 * (ao[1:4] * c0[None]).sum(axis=2)
                # Hessian of rho: 2 (sum_m d2 phi_m c0_m + sum_m d phi_m d c_m)
                hess = np.empty((3, 3, len(rho)))
                for (k, kk), comp in zip(_SECOND_DERIVATIVES, range(4, 10), strict=True):
                    h = 2.0 * ((ao[comp] * c0).sum(axis=1) + (ao[1 + k] * c1[kk]).sum(axis=1))
                    hess[k, kk] = hess[kk, k] = h
                e, e_rho, e_sigma, kernel = self._grid_integrand(
                    grid, sl, rho, np.sum(grad**2, axis=0)
                )
                values[sl] = e * kernel
                if grid.region == "inside":
                    dispersion_values[sl] = e
                w = grid.weights[sl]
                # the points move with their owners through the integrand
                motion = (w * kernel)[:, None] * (
                    e_rho[:, None] * grad.T
                    + 2.0 * e_sigma[:, None] * np.einsum("klp,lp->pk", hess, grad)
                )
                owners = grid.owners[sl]
                for x in range(3):
                    g[:, x] += np.bincount(owners, motion[:, x], minlength=natm)
                # the basis functions move with their atoms
                coeff = np.empty((4, len(w)))
                coeff[0] = w * kernel * e_rho
                coeff[1:] = 2.0 * (w * kernel * e_sigma)[None, :] * grad
                g += basis_motion_gradient(mol, ao, dm, coeff, ao_atom=self._ao_atom)
            # the grid's weights change with the geometry
            g += partition_gradient(grid, self.centres, self.radii, values)

        # The dispersion kernel: I = const - J, J a sum over the surface that
        # moves with the points and with the elements.
        _J, dJ, dE_ds, dE_dA = kernel_derivatives(
            self.inside.coords,
            self.coords,
            self.normals,
            self.areas,
            self.delta,
            self.inside.weights * dispersion_values,
        )
        motion = -(self.inside.weights * dispersion_values)[:, None] * dJ
        for x in range(3):
            g[:, x] += np.bincount(self.inside.owners, motion[:, x], minlength=natm)
        element_force = -dE_ds
        area_force = -dE_dA

        # The surface term: sum_i A_i f(F_i), F_i = n_i . E(s_i).
        nuclear, tensor = _nuclear_field(self.coords, self.centres, mol.atom_charges())
        charges = mol.atom_charges()
        ip1 = mol._add_suffix("int3c2e_ip1")
        ip1ip2 = mol._add_suffix("int3c2e_ip1ip2")
        nao = mol.nao
        for sl in self._field_blocks():
            fakemol = _point_charges(mol, self.coords[sl])
            ints = df.incore.aux_e2(mol, fakemol, intor=ip1, aosym="s1", comp=3)
            nb = ints.shape[-1]
            field = (
                nuclear[sl] + 2.0 * np.tensordot(dm.ravel(), ints.reshape(3, -1, nb), ([0], [1])).T
            )
            F = np.einsum("px,px->p", field, self.normals[sl])
            f, f_prime = self._surface_function(F)
            area_force[sl] += f
            q = (self.areas[sl] * f_prime)[:, None] * self.normals[sl]  # dG/dE at s_i
            del ints
            # W[a, b] = (d_a m n | d_b s) = d^2 U_mn / dR_a ds_b for m on the atom at R, with
            # U_mn(s) = (m n|s).  E_x = sum D dU/ds_x, so with D symmetric moving atom C's
            # functions gives dE_x/dR_C,y = 2 sum_{m on C} D_mn W[y, x], and moving the point
            # dE_x/ds_y = -2 sum_mn D_mn W[y, x] (translational invariance).
            W = df.incore.aux_e2(mol, fakemol, intor=ip1ip2, aosym="s1", comp=9)
            W = W.reshape(3, 3, nao * nao, nb)
            field_grad = -2.0 * np.tensordot(dm.ravel(), W, ([0], [2])).transpose(2, 1, 0)
            field_grad += np.einsum("a,paxy->pxy", charges, tensor[sl])
            element_force[sl] += np.einsum("px,pxy->py", q, field_grad)
            Mx = np.zeros((3, nao * nao))
            for y in range(3):
                for x in range(3):
                    Mx[y] += W[y, x] @ q[:, x]
            T = (Mx.reshape(3, nao, nao) * dm[None]).sum(axis=2)
            for y in range(3):
                g[:, y] += 2.0 * np.bincount(self._ao_atom, T[y], minlength=natm)
            # moving the nuclei
            g -= np.einsum("a,px,paxy->ay", charges, q, tensor[sl])
        for x in range(3):
            g[:, x] += np.bincount(self.element_atoms, element_force[:, x], minlength=natm)
        # the elements' areas change through the switching function
        from pyscf.solvent.grad.pcm import get_dF_dA

        _dF, dA = get_dF_dA(self.ws.surface, self.ws.surface_discretization_method)
        g += np.einsum("i,iax->ax", area_force, dA)
        return g
