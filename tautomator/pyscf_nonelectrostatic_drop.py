"""The nonelectrostatic correction on MOIST's DROP cavities (PySCF).

The same three terms as `pyscf_nonelectrostatic` (docs/nonelectrostatic.md), on the
surface MOIST builds for its PCM (tautomator.pyscf_moist: SvdW-DROP, or rho-DROP, which
follows the density):

  dispersion   A int_inside u(rho) I(r),  I(r) = int_outside dr'/(|r-r'|^6 + delta^6)
  exchange     B int_outside |grad rho|
  short range  sum_i a_i f(n_i . E(s_i))

The kernel I is the same surface sum as on PySCF's cavity, now over DROP's points, normals
and areas.  The volume integrals are written on the surface too (`surface_quadrature`): the
outside as rays from the nuclei through every surface point, the inside as an all-space
atom grid less the outside.  Everything then depends on the cavity through its points s_i,
normals n_i and areas a_i, and MOIST takes the derivatives with respect to those to the
nuclei and, on rho-DROP, to the density (packages/moist's patch exports the two
contractions):

  Fock:      V = dG/dD at a fixed surface, plus on rho-DROP the surface's own response,
             the density weights MOIST returns against d rho/dD;
  gradient:  dG/dR at a fixed surface and density matrix (basis functions, grids, nuclei),
             plus MOIST's contraction of dG/d(s, n, a) (the anchor, switching and field
             routes), plus on rho-DROP the density weights against d rho/dR.

`DropTerms` is the correction as a function of the density matrix at one geometry, on the
cavity of a MoistSolvation (pyscf_moist), which it asks for the surface at that density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import df, dft, lib

from tautomator.dispersion_kernel import kernel_integral, kernel_surface_derivatives
from tautomator.nonelectrostatic import (
    DELTA,
    VV_C,
    Energies,
    Params,
    dispersion_density,
    exchange_density,
    field_power,
    whole_space_kernel_integral,
)
from tautomator.pyscf_nonelectrostatic import (
    _SECOND_DERIVATIVES,
    _ao_atoms,
    _nuclear_field,
    _point_blocks,
    _point_charges,
    basis_motion_gradient,
)
from tautomator.surface_quadrature import (
    RAY_RULE,
    VOLUME_RULE,
    ExteriorRays,
    RayRule,
    VolumeGrid,
    VolumeRule,
    build_exterior_rays,
    build_volume_grid,
    volume_grid_gradient,
)

Array = NDArray[np.float64]

# The share of what mol.max_memory leaves that the normal-field map of a surface may take
# (DropTerms._field_map); FastPCM's or MOIST's (k|mn) take another half.
FIELD_STORE_FRACTION = 0.25


@dataclass(frozen=True)
class DropSettings:
    """The correction's parameters, the solvent's electron density (e/bohr^3) and the grids."""

    params: Params
    solvent_rho: float
    delta: float = DELTA
    vv_c: float = VV_C
    volume_rule: VolumeRule = VOLUME_RULE
    ray_rule: RayRule = RAY_RULE


@dataclass(frozen=True)
class DropIntegrals:
    """The parameter-free integrals on one surface at one density (the descriptors' raw form)."""

    dispersion: float  # int_inside u I
    exchange: float  # int_outside |grad rho|
    electrons_outside: float
    areas: Array
    normal_field: Array  # n_i . E(s_i)
    element_atoms: NDArray[np.intp]


@dataclass
class _Surface:
    """One cavity surface and what depends on it alone."""

    key: bytes
    xyz: Array
    normals: Array
    areas: Array
    owners: NDArray[np.intp]
    rays: ExteriorRays
    kernel_volume: Array  # I at the all-space grid's points
    kernel_rays: Array  # I at the ray nodes
    field_map: Array | None = None  # DropTerms._field_map
    field_map_refused: bool = False


@dataclass
class _Pass:
    """Everything one evaluation produced."""

    energies: Energies
    integrals: DropIntegrals
    V: Array | None = None  # dG/dD at a fixed surface
    w_xyz: Array | None = None  # dG/ds
    w_n: Array | None = None  # dG/dn
    w_a: Array | None = None  # dG/da
    explicit: Array | None = None  # dG/dR at a fixed surface and density matrix


def _density(mol: Any, dm: Array, coords: Array, deriv: int) -> tuple[Array, Array, Array | None]:
    """rho (n,), grad rho (3, n) and, with deriv 2, its Hessian (3, 3, n), in blocks."""
    ni = dft.numint.NumInt()
    n = len(coords)
    rho = np.empty(n)
    grad = np.empty((3, n))
    hess = np.empty((3, 3, n)) if deriv >= 2 else None
    ncomp = 10 if deriv >= 2 else 4
    for sl in _point_blocks(n, ncomp * mol.nao):
        ao = ni.eval_ao(mol, coords[sl], deriv=deriv)
        c0 = ao[0] @ dm
        rho[sl] = (ao[0] * c0).sum(axis=1)
        grad[:, sl] = 2.0 * (ao[1:4] * c0[None]).sum(axis=2)
        if hess is not None:
            c1 = ao[1:4] @ dm
            for (k, kk), comp in zip(_SECOND_DERIVATIVES, range(4, 10), strict=True):
                h = 2.0 * ((ao[comp] * c0).sum(axis=1) + (ao[1 + k] * c1[kk]).sum(axis=1))
                hess[k, kk, sl] = h
                hess[kk, k, sl] = h
    return rho, grad, hess


def _gga_pass(
    mol: Any,
    dm: Array,
    coords: Array,
    c_rho: Array,
    c_grad: Array,
    potential: bool,
    gradient: bool,
    ao_atom: NDArray[np.intp],
) -> tuple[Array | None, Array | None]:
    """For E = sum_p (c_rho_p rho_p + c_grad_p . grad rho_p) as a linear functional at fixed
    coefficients: V = dE/dD (nao, nao) and the basis functions' motion dE/dR (natm, 3)."""
    if not (potential or gradient):
        return None, None
    ni = dft.numint.NumInt()
    nao = mol.nao
    V = np.zeros((nao, nao)) if potential else None
    g = np.zeros((mol.natm, 3)) if gradient else None
    deriv = 2 if gradient else 1
    for sl in _point_blocks(len(coords), (10 if gradient else 4) * nao):
        ao = ni.eval_ao(mol, coords[sl], deriv=deriv)
        if V is not None:
            # rho = sum D phi phi, grad rho = 2 sum D grad phi phi: dE/dD = sum_p phi (c_rho phi
            # + 2 c_grad . grad phi), symmetrized below
            wv = np.empty((4, sl.stop - sl.start))
            wv[0] = 0.5 * c_rho[sl]
            wv[1:] = c_grad[:, sl]
            V += ao[0].T @ (ao[:4] * wv[:, :, None]).sum(axis=0)
        if g is not None:
            coeff = np.empty((4, sl.stop - sl.start))
            coeff[0] = c_rho[sl]
            coeff[1:] = c_grad[:, sl]
            g += basis_motion_gradient(mol, ao, dm, coeff, ao_atom=ao_atom)
    if V is not None:
        V = V + V.T
    return V, g


class DropTerms:
    """The nonelectrostatic correction on a MOIST cavity, as a function of the density matrix.

    ``solvation`` is the MoistSolvation (pyscf_moist) whose cavity the terms live on; for
    rho-DROP the surface is the one MOIST builds at the density given.  Everything takes the
    total density matrix (alpha + beta), symmetric, in the AO basis of ``mol``:

    * `energy(dm)`, `integrals(dm)`: the terms, and the parameter-free integrals;
    * `energy_and_potential(dm)`: also V = dG/dD, the surface's response included;
    * `gradient(dm)`: dG/dR at fixed density matrix, the surface's motion included.
    """

    def __init__(self, mol: Any, solvation: Any, settings: DropSettings) -> None:
        self.mol, self.solvation, self.settings = mol, solvation, settings
        self.centres = np.asarray(mol.atom_coords(unit="B"), dtype=np.float64)
        self.volume: VolumeGrid = build_volume_grid(self.centres, settings.volume_rule)
        self._ao_atom = _ao_atoms(mol)
        self._surface_cache: _Surface | None = None

    # -- the surface ------------------------------------------------------------

    @property
    def cavity(self) -> Any:
        return self.solvation.model.cavity

    @property
    def density_dependent(self) -> bool:
        return bool(self.solvation.configuration.density_dependent)

    def _surface(self, dm: Array) -> _Surface:
        """The cavity at dm (MOIST rebuilds a rho-DROP surface for it), with its rays and the
        kernel on both point sets."""
        self.solvation.evaluate(dm)
        snap = self.cavity.snapshot()
        key = np.ascontiguousarray(snap.xyz).tobytes() + np.ascontiguousarray(snap.a).tobytes()
        cached = self._surface_cache
        if cached is not None and cached.key == key:
            return cached
        xyz = np.ascontiguousarray(snap.xyz, dtype=np.float64)
        normals = np.ascontiguousarray(snap.normal0, dtype=np.float64)
        areas = np.ascontiguousarray(snap.a, dtype=np.float64)
        rays = build_exterior_rays(xyz, normals, areas, self.centres, self.settings.ray_rule)
        delta = self.settings.delta
        whole = whole_space_kernel_integral(delta)
        surface = _Surface(
            key=key,
            xyz=xyz,
            normals=normals,
            areas=areas,
            owners=np.asarray(snap.owner, dtype=np.intp),
            rays=rays,
            kernel_volume=whole - kernel_integral(self.volume.coords, xyz, normals, areas, delta),
            kernel_rays=whole - kernel_integral(rays.coords, xyz, normals, areas, delta),
        )
        self._surface_cache = surface
        return surface

    # -- the surface term --------------------------------------------------------

    def _surface_function(self, F: Array) -> tuple[Array, Array]:
        p = self.settings.params
        neg, d_neg = field_power(F, p.neg_exponent, -1)
        pos, d_pos = field_power(F, p.pos_exponent, 1)
        return p.neg_coeff * neg + p.pos_coeff * pos, p.neg_coeff * d_neg + p.pos_coeff * d_pos

    def _field_map(self, surf: _Surface) -> Array | None:
        """(npts, npair): row i takes the packed, weighted density matrix (`_packed`) to
        n_i . E_el(s_i).  Kept with its surface while it fits in FIELD_STORE_FRACTION of what
        mol.max_memory leaves, so that an SCF on a fixed surface (SvdW-DROP) computes the
        integrals once; None where it does not fit."""
        if surf.field_map is None and not surf.field_map_refused:
            mol = self.mol
            nao, npts = mol.nao, len(surf.areas)
            npair = nao * (nao + 1) // 2
            budget = FIELD_STORE_FRACTION * (mol.max_memory - lib.current_memory()[0]) * 1e6
            if npts * npair * 8 > budget:
                surf.field_map_refused = True
                return None
            P = np.empty((npts, npair))
            for sl, ints in self._ip2_blocks(surf.xyz):
                P[sl] = -sum(ints[x] * surf.normals[sl, x] for x in range(3)).T
            surf.field_map = P
        return surf.field_map

    def _ip2_blocks(self, points: Array) -> Any:
        """(slice, (3, npair, n) d/ds (mn|s)) over the points, packed in the AO pair.

        E_el(s) = -sum_mn D_mn d/ds (mn|s): the derivative on the point, symmetric in m and n,
        so half the integrals of the bra derivative.
        """
        mol = self.mol
        npair = mol.nao * (mol.nao + 1) // 2
        intor = mol._add_suffix("int3c2e_ip2")
        for sl in _point_blocks(len(points), 3 * npair):
            fakemol = _point_charges(mol, points[sl])
            yield sl, df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", comp=3)

    def _packed(self, dm: Array) -> Array:
        """D packed with weights, so that sum_mn D_mn X_mn = packed . X_packed for symmetric X."""
        nao = self.mol.nao
        return np.asarray(lib.pack_tril(dm + dm.T) * lib.pack_tril(1.0 - 0.5 * np.eye(nao)))

    def _field_term(
        self, dm: Array, surf: _Surface, potential: bool, adjoint: bool, gradient: bool
    ) -> tuple[float, Array, Array | None, Array | None, Array | None, Array | None, Array | None]:
        """sum_i a_i f(n_i . E(s_i)): (energy, F, V, dG/ds, dG/dn, dG/da, explicit dG/dR)."""
        mol = self.mol
        nao = mol.nao
        charges = mol.atom_charges()
        nuclear, tensor = _nuclear_field(surf.xyz, self.centres, charges)
        npts = len(surf.areas)
        derivs = adjoint or gradient
        d = self._packed(dm)
        stored = None if derivs else self._field_map(surf)
        field = None
        if stored is not None:
            F = np.einsum("px,px->p", nuclear, surf.normals) + stored @ d
        else:
            field = nuclear.copy()
            for sl, ints in self._ip2_blocks(surf.xyz):
                field[sl] -= np.tensordot(d, ints, ([0], [1])).T
            F = np.einsum("px,px->p", field, surf.normals)
        f, f_prime = self._surface_function(F)
        energy = float(surf.areas @ f)
        V = None
        if potential:
            c = surf.areas * f_prime
            if stored is not None:
                packed = c @ stored
            else:
                packed = np.zeros(nao * (nao + 1) // 2)
                for sl, ints in self._ip2_blocks(surf.xyz):
                    q = c[sl, None] * surf.normals[sl]
                    packed -= sum(ints[x] @ q[:, x] for x in range(3))
            V = lib.unpack_tril(packed)
        if not derivs:
            return energy, F, V, None, None, None, None

        assert field is not None
        q = (surf.areas * f_prime)[:, None] * surf.normals  # dG/dE at s_i
        w_xyz = np.zeros((npts, 3))
        g = np.zeros((mol.natm, 3)) if gradient else None
        ip1ip2 = mol._add_suffix("int3c2e_ip1ip2")
        for sl in _point_blocks(npts, 9 * nao * nao):
            fakemol = _point_charges(mol, surf.xyz[sl])
            # W[a, b] = (d_a m n | d_b s): see NonelectrostaticTerms.gradient
            W = df.incore.aux_e2(mol, fakemol, intor=ip1ip2, aosym="s1", comp=9)
            nb = W.shape[-1]
            W = W.reshape(3, 3, nao * nao, nb)
            field_grad = -2.0 * np.tensordot(dm.ravel(), W, ([0], [2])).transpose(2, 1, 0)
            field_grad += np.einsum("a,paxy->pxy", charges, tensor[sl])
            w_xyz[sl] = np.einsum("px,pxy->py", q[sl], field_grad)
            if g is not None:
                Mx = np.zeros((3, nao * nao))
                for y in range(3):
                    for x in range(3):
                        Mx[y] += W[y, x] @ q[sl, x]
                T = (Mx.reshape(3, nao, nao) * dm[None]).sum(axis=2)
                for y in range(3):
                    g[:, y] += 2.0 * np.bincount(self._ao_atom, T[y], minlength=mol.natm)
                g -= np.einsum("a,px,paxy->ay", charges, q[sl], tensor[sl])
        w_n = (surf.areas * f_prime)[:, None] * field
        return energy, F, V, w_xyz, w_n, f, g

    # -- one evaluation ------------------------------------------------------------

    def _evaluate(
        self, dm: Array, potential: bool = False, adjoint: bool = False, gradient: bool = False
    ) -> _Pass:
        mol = self.mol
        dm = np.asarray(dm, dtype=np.float64)
        s = self.settings
        A, B = s.params.dispersion, s.params.exchange
        surf = self._surface(dm)
        rays, vol = surf.rays, self.volume
        if not (adjoint or gradient):
            return self._evaluate_once(dm, surf, potential)

        # densities on both point sets, with the Hessians the surface derivatives need
        rho_v, grad_v, hess_v = _density(mol, dm, vol.coords, 2)
        rho_r, grad_r, hess_r = _density(mol, dm, rays.coords, 2)
        u_v, ur_v, us_v = dispersion_density(
            rho_v, np.sum(grad_v**2, axis=0), s.solvent_rho, s.vv_c
        )
        u_r, ur_r, us_r = dispersion_density(
            rho_r, np.sum(grad_r**2, axis=0), s.solvent_rho, s.vv_c
        )
        x_r, xs_r = exchange_density(np.sum(grad_r**2, axis=0))

        I_v, I_r = surf.kernel_volume, surf.kernel_rays
        dispersion = float(vol.weights @ (u_v * I_v) - rays.weights @ (u_r * I_r))
        exchange = float(rays.weights @ x_r)
        electrons_outside = float(rays.weights @ rho_r)

        # V at a fixed surface, and the basis functions' motion: c_rho and c_grad per point
        cv_rho = A * vol.weights * I_v * ur_v
        cv_grad = 2.0 * (A * vol.weights * I_v * us_v)[None] * grad_v
        cr_rho = -A * rays.weights * I_r * ur_r
        cr_grad = 2.0 * (rays.weights * (-A * I_r * us_r + B * xs_r))[None] * grad_r
        V1, gb_v = _gga_pass(
            mol, dm, vol.coords, cv_rho, cv_grad, potential, gradient, self._ao_atom
        )
        V2, gb_r = _gga_pass(
            mol, dm, rays.coords, cr_rho, cr_grad, potential, gradient, self._ao_atom
        )

        sr, F, V3, f_xyz, f_n, f_a, g_field = self._field_term(
            dm, surf, potential, adjoint, gradient
        )
        energies = Energies(dispersion=A * dispersion, exchange=B * exchange, short_range=sr)
        integrals = DropIntegrals(
            dispersion=dispersion,
            exchange=exchange,
            electrons_outside=electrons_outside,
            areas=surf.areas,
            normal_field=F,
            element_atoms=surf.owners,
        )
        out = _Pass(energies=energies, integrals=integrals)
        if potential:
            assert V1 is not None and V2 is not None and V3 is not None
            out.V = V1 + V2 + V3

        assert hess_v is not None and hess_r is not None
        assert f_xyz is not None and f_n is not None and f_a is not None
        # The kernel I = const - J: J's surface sum, with dJ/dr at both point sets
        n_v = len(vol.weights)
        points = np.concatenate([vol.coords, rays.coords])
        c_kernel = np.concatenate([-A * vol.weights * u_v, A * rays.weights * u_r])
        _J, dJ, kj_xyz, kj_a, kj_n = kernel_surface_derivatives(
            points, surf.xyz, surf.normals, surf.areas, s.delta, c_kernel
        )
        dI_v, dI_r = -dJ[:n_v].T, -dJ[n_v:].T  # (3, n)

        # the integrand's spatial gradient on the rays: G = -A u I + B x
        du_r = ur_r[None] * grad_r + 2.0 * us_r[None] * np.einsum("klp,lp->kp", hess_r, grad_r)
        dx_r = 2.0 * xs_r[None] * np.einsum("klp,lp->kp", hess_r, grad_r)
        G_r = -A * u_r * I_r + B * x_r
        dG_r = -A * (du_r * I_r[None] + u_r[None] * dI_r) + B * dx_r
        r_xyz, r_n, r_a, r_R = rays.adjoint(
            surf.xyz, surf.normals, surf.areas, self.centres, G_r, dG_r.T
        )
        out.w_xyz = r_xyz + kj_xyz + f_xyz
        out.w_n = r_n + kj_n + f_n
        out.w_a = r_a + kj_a + f_a
        if not gradient:
            return out

        assert gb_v is not None and gb_r is not None and g_field is not None
        # the all-space grid moves with the atoms and its partition with all of them
        du_v = ur_v[None] * grad_v + 2.0 * us_v[None] * np.einsum("klp,lp->kp", hess_v, grad_v)
        G_v = A * u_v * I_v
        dG_v = A * (du_v * I_v[None] + u_v[None] * dI_v)
        g_vol = volume_grid_gradient(vol, self.centres, G_v, dG_v.T)
        out.explicit = g_vol + gb_v + r_R + gb_r + g_field
        return out

    def _evaluate_once(self, dm: Array, surf: _Surface, potential: bool) -> _Pass:
        """Energy and V at a fixed surface, the densities and V from one pass over each grid."""
        mol = self.mol
        s = self.settings
        A, B = s.params.dispersion, s.params.exchange
        ni = dft.numint.NumInt()
        nao = mol.nao
        V = np.zeros((nao, nao)) if potential else None
        sums = {"dispersion": 0.0, "exchange": 0.0, "electrons": 0.0}

        def run(coords: Array, weights: Array, kernel: Array, outside: bool) -> None:
            for sl in _point_blocks(len(coords), 4 * nao):
                ao = ni.eval_ao(mol, coords[sl], deriv=1)
                c0 = ao[0] @ dm
                rho = (ao[0] * c0).sum(axis=1)
                grad = 2.0 * (ao[1:4] * c0[None]).sum(axis=2)
                sigma = np.sum(grad**2, axis=0)
                u, u_r, u_s = dispersion_density(rho, sigma, s.solvent_rho, s.vv_c)
                w, k = weights[sl], kernel[sl]
                sign = -1.0 if outside else 1.0
                sums["dispersion"] += sign * float(w @ (u * k))
                c_rho = sign * A * w * k * u_r
                c_sigma = sign * A * w * k * u_s
                if outside:
                    x, x_s = exchange_density(sigma)
                    sums["exchange"] += float(w @ x)
                    sums["electrons"] += float(w @ rho)
                    c_sigma = c_sigma + B * w * x_s
                if V is not None:
                    wv = np.empty((4, len(w)))
                    wv[0] = 0.5 * c_rho
                    wv[1:] = 2.0 * c_sigma[None] * grad
                    V[:] += ao[0].T @ (ao * wv[:, :, None]).sum(axis=0)

        run(self.volume.coords, self.volume.weights, surf.kernel_volume, outside=False)
        run(surf.rays.coords, surf.rays.weights, surf.kernel_rays, outside=True)
        sr, F, V3, *_ = self._field_term(dm, surf, potential, False, False)
        energies = Energies(
            dispersion=A * sums["dispersion"], exchange=B * sums["exchange"], short_range=sr
        )
        integrals = DropIntegrals(
            dispersion=sums["dispersion"],
            exchange=sums["exchange"],
            electrons_outside=sums["electrons"],
            areas=surf.areas,
            normal_field=F,
            element_atoms=surf.owners,
        )
        out = _Pass(energies=energies, integrals=integrals)
        if V is not None:
            assert V3 is not None
            out.V = V + V.T + V3
        return out

    # -- the public interface ---------------------------------------------------------

    def energy(self, dm: Array) -> Energies:
        return self._evaluate(dm).energies

    def integrals(self, dm: Array) -> DropIntegrals:
        return self._evaluate(dm).integrals

    def _require_contractions(self) -> None:
        """The surface derivatives need MOIST's reverse contractions, which psi4-swig-pcm's
        packages/moist patch 0002 exports; energies and descriptors do not."""
        if not hasattr(self.cavity, "contract_surface_gradient"):
            import moist

            raise RuntimeError(
                f"MOIST {moist.__version__} does not export the surface contractions the "
                "nonelectrostatic correction's gradient and self-consistent Fock matrix need "
                "(moist_contract_cavity_surface_gradient and _density): they come from "
                "psi4-swig-pcm's packages/moist recipe (patches/0002).  Energies and "
                "descriptors work without them.  Install the moist package pixi.toml pins "
                "(pixi install)."
            )

    def _density_weights(self, out: _Pass) -> Any:
        """MOIST's density weights for dG/d(s, n, a), or None on a fixed surface."""
        self._require_contractions()
        return self.cavity.contract_surface_density(w_xyz=out.w_xyz, w_n=out.w_n, w_a=out.w_a)

    def energy_and_potential(self, dm: Array) -> tuple[Energies, Array]:
        """G and V = dG/dD: at a fixed surface, plus on rho-DROP the surface's response."""
        dependent = self.density_dependent
        out = self._evaluate(dm, potential=True, adjoint=dependent)
        assert out.V is not None
        V = out.V
        if dependent:
            weights = self._density_weights(out)
            if weights is not None:
                from tautomator import pyscf_moist

                V = V + pyscf_moist.density_fock(self.solvation.host, dm, self.cavity.xyz, weights)
        return out.energies, V

    def gradient(self, dm: Array) -> Array:
        """dG/dR (natm, 3) at fixed density matrix, the cavity's motion included."""
        self._require_contractions()
        out = self._evaluate(dm, gradient=True)
        assert out.explicit is not None
        g = out.explicit + self.cavity.contract_surface_gradient(
            w_xyz=out.w_xyz, w_n=out.w_n, w_a=out.w_a
        )
        if self.density_dependent:
            weights = self._density_weights(out)
            if weights is not None:
                from tautomator import pyscf_moist

                g = g + pyscf_moist.density_gradient(
                    self.solvation.host, dm, self.cavity.xyz, weights
                )
        return np.asarray(g)


# ---------------------------------------------------------------------------
# The correction inside the SCF
# ---------------------------------------------------------------------------


def _total(dm: Array) -> Array:
    dm = np.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm


def _self_consistent_state_class() -> type:
    """The MOIST solvation state with the correction added to its energy, Fock and gradient.

    Built on first use: pyscf_moist imports MOIST, which an environment may lack.
    """
    from tautomator.pyscf_moist import _MoistSolvationState

    class SelfConsistentDropState(_MoistSolvationState):
        """pyscf_moist's state for a MOIST mean field, with G and V = dG/dD in every SCF
        iteration and dG/dR in the gradient (`enable_self_consistent`)."""

        def __init__(self, mol: Any, *, nonelectrostatic: DropSettings, **kwargs: Any) -> None:
            self.nonelectrostatic = nonelectrostatic
            self.e_nonelectrostatic: Energies | None = None
            self._drop_terms: DropTerms | None = None
            super().__init__(mol, **kwargs)

        def copy(self) -> Any:
            return type(self)(
                self.mol,
                nonelectrostatic=self.nonelectrostatic,
                cavity=self.cavity,
                components=self.components,
                context=self.context,
                allow_ecp=self.allow_ecp,
            )

        def reset(self, mol: Any = None) -> Any:
            self._drop_terms = None
            return super().reset(mol)

        def terms(self) -> DropTerms:
            """The correction on the current driver's cavity (one per geometry)."""
            driver = self._driver()
            if self._drop_terms is None or self._drop_terms.solvation is not driver:
                self._drop_terms = DropTerms(self.mol, driver, self.nonelectrostatic)
            return self._drop_terms

        def kernel(self, dm: Array) -> tuple[float, Array]:
            e_pcm, v_pcm = super().kernel(dm)
            try:
                energies, V = self.terms().energy_and_potential(_total(dm))
            except Exception:
                self.reset()
                raise
            self.e_nonelectrostatic = energies
            return e_pcm + energies.total, v_pcm + V

        def gradient(self, dm: Array) -> Array:
            g = super().gradient(dm)
            try:
                return np.asarray(g + self.terms().gradient(_total(dm)))
            except Exception:
                self.reset()
                raise

    return SelfConsistentDropState


def enable_self_consistent(mf: Any, settings: DropSettings) -> Any:
    """Include the correction in the SCF of a MOIST mean field (pyscf_moist.attach); returns mf.

    The SCF must be rerun (mf.kernel()) for the energy to include it; starting from a
    converged PCM density saves most of the iterations.
    """
    from tautomator.pyscf_moist import _MoistSolvationState

    state = getattr(mf, "with_moist", None)
    if not isinstance(state, _MoistSolvationState):
        raise ValueError(
            "the mean field has no MOIST solvation from tautomator.pyscf_moist.attach"
        )
    cls = _self_consistent_state_class()
    mf.with_moist = cls(
        state.mol,
        nonelectrostatic=settings,
        cavity=state.cavity,
        components=state.components,
        context=state.context,
        allow_ecp=state.allow_ecp,
    )
    return mf


# ---------------------------------------------------------------------------
# The post-SCF gradient
# ---------------------------------------------------------------------------

# rho-DROP's response to the density is not linear (its surface follows the density), and
# MOIST has no second derivatives, so the post-SCF gradient takes it from central differences
# at D +- t x and D +- 2 t x (Richardson), with t this fraction of |D| / |x|.  MOIST
# converges the surface to 1e-10, which the differences divide by t.
RESPONSE_STEP = 1e-3


def _richardson(f: Any, D: Array, x: Array) -> Array:
    """The derivative of f at D along x, from central differences at two steps."""
    nx = float(np.linalg.norm(x))
    if nx == 0.0:
        return np.zeros_like(np.asarray(f(D)))
    t = RESPONSE_STEP * float(np.linalg.norm(D)) / nx
    r1 = (np.asarray(f(D + t * x)) - np.asarray(f(D - t * x))) / (2.0 * t)
    r2 = (np.asarray(f(D + 2.0 * t * x)) - np.asarray(f(D - 2.0 * t * x))) / (4.0 * t)
    return np.asarray((4.0 * r1 - r2) / 3.0)


def post_scf_gradient(mf: Any, terms: DropTerms) -> Array:
    """dG/dR (natm, 3) of the correction on the converged density of a MOIST mean field.

    ``mf`` a converged closed-shell mean field from pyscf_moist.attach (the SCF without the
    correction), ``terms`` the correction on its cavity.  The density's response is one
    Z-vector solve (pyscf_nonelectrostatic_gradient.z_vector_gradient) with the SCF's own
    response: PySCF's without the solvent, plus MOIST's PCM.  On SvdW-DROP the PCM's Fock
    matrix is affine in the density, so its response is exact; on rho-DROP it is
    differentiated numerically (RESPONSE_STEP).
    """
    from tautomator.pyscf_nonelectrostatic_gradient import _check_supported, z_vector_gradient

    _check_supported(mf)
    D = _total(mf.make_rdm1())
    _energies, V = terms.energy_and_potential(D)
    g = terms.gradient(D)
    driver = terms.solvation
    gas = mf.undo_moist().gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)

    def fock(dm: Array) -> Array:
        return np.array(driver.evaluate(dm).fock)

    if not terms.density_dependent:
        F0 = fock(np.zeros_like(D))

        def vresp(x: Array) -> Array:
            return np.asarray(gas(x) + fock(x) - F0)

        def bilinear(D: Array, Z: Array) -> Array:
            # quadratic in the density at fixed geometry: the polarization identity is exact
            t = float(np.linalg.norm(D) / max(np.linalg.norm(Z), 1e-300))
            return np.asarray(
                (driver.gradient(D + t * Z) - driver.gradient(D - t * Z)) / (2.0 * t)
            )

    else:

        def vresp(x: Array) -> Array:
            return np.asarray(gas(x) + _richardson(fock, D, x))

        def bilinear(D: Array, Z: Array) -> Array:
            return _richardson(driver.gradient, D, Z)

    try:
        g += z_vector_gradient(mf, D, V, vresp, bilinear)
    finally:
        driver.evaluate(D)  # leave MOIST at the SCF density
    return np.asarray(g)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

# Placeholder coefficients for evaluating the parameter-free integrals alone
_NO_PARAMS = Params(
    dispersion=0.0, exchange=0.0, neg_coeff=0.0, neg_exponent=1.0, pos_coeff=0.0, pos_exponent=1.0
)


def _charge_scale(component: Any) -> float:
    """How the PCM component scales a conductor's charges (nonelectrostatic.Surface)."""
    import moist

    eps = float(component.epsilon)
    if isinstance(component, moist.ModelComponentCOSMO):
        return (eps - 1.0) / (eps + 0.5)
    return (eps - 1.0) / eps


def apparent_charges(view: Any, component: Any) -> Array:
    """The apparent surface charges q of a MOIST PCM component's latest solve, (ngrid,).

    ``view`` is the evaluated model's component (``model.components[i]``), ``component`` its
    configuration.  MOIST (from 1.2) publishes a layer model's (IEF-PCM, SS(V)PE, ...) q as
    ``surface_charge``, and every PCM's potential adjoint w_phi as ``potential_adjoint``, which
    for the conductor models (C-PCM, COSMO) is q itself.  It lists both only while the charges
    are current: from an evaluation until the model is updated to a new structure, and not
    after a failed solve.  tautomator runs IEF-PCM only (config.DROP_COMPONENT), so the
    conductor branch is not exercised.
    """
    import moist

    published = {f.name for f in view.fields()}
    if "surface_charge" in published:
        return np.asarray(view.get("surface_charge"), dtype=np.float64)
    conductor = isinstance(component, (moist.ModelComponentCPCM, moist.ModelComponentCOSMO))
    if conductor and "potential_adjoint" in published:
        return np.asarray(view.get("potential_adjoint"), dtype=np.float64)
    raise RuntimeError(
        f"MOIST {moist.__version__} publishes no apparent surface charges for "
        f"{type(component).__name__}, which the charge-derived descriptor needs: either the "
        "model has no current solve (evaluate it after its last update), or the moist package "
        "predates 1.2 (pixi install installs the one pixi.toml pins)"
    )


def drop_descriptors(
    mf: Any,
    solvent_rho: float,
    delta: float = DELTA,
    vv_c: float = VV_C,
    volume_rule: VolumeRule = VOLUME_RULE,
    ray_rule: RayRule = RAY_RULE,
) -> Any:
    """The parameter-free integrals (nonelectrostatic.Descriptors) of a converged MOIST mean
    field, on its cavity at its density: per surface point the area, the solute's normal
    field, the PCM's apparent surface charge recast as a field, and the atom whose sphere the
    point was projected from.

    The charges are the apparent surface charges q of MOIST's solve, as PySCF's cavity gives
    its own (pyscf_nonelectrostatic._surface), read from the PCM component (apparent_charges).
    They are not the potential adjoint w_phi MOIST hands the host: for C-PCM and COSMO the two
    are the same, but for IEF-PCM w_phi is the symmetrized adjoint (the symmetric part of the
    response matrix applied to the potential), which differs from q by 0.04% RMS (0.1% at
    most) on SvdW-DROP for formate; PySCF's own q and q_sym differ by 0.3-1.3% RMS.
    """
    from tautomator.nonelectrostatic import Descriptors

    if not mf.converged:
        raise ValueError("the SCF is not converged")
    D = _total(mf.make_rdm1())
    driver = mf.with_moist._driver()
    settings = DropSettings(
        params=_NO_PARAMS,
        solvent_rho=solvent_rho,
        delta=delta,
        vv_c=vv_c,
        volume_rule=volume_rule,
        ray_rule=ray_rule,
    )
    ints = DropTerms(mf.mol, driver, settings).integrals(D)
    driver.evaluate(D)
    (component,) = mf.with_moist.components
    charges = apparent_charges(driver.model.components[0], component)
    charge_field = -4.0 * np.pi / _charge_scale(component) * charges / ints.areas
    return Descriptors(
        dispersion=ints.dispersion,
        exchange=ints.exchange,
        areas=ints.areas,
        normal_field=ints.normal_field,
        charge_field=np.asarray(charge_field, dtype=np.float64),
        electrons_outside=ints.electrons_outside,
        element_atoms=ints.element_atoms,
    )
