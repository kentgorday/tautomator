# Copyright 2026 Lukas Wittmann (MOIST, https://github.com/lukaswittmann/moist).
# Copyright 2026 Kent Gorday (modifications).
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a
# copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
#
# Unlike the rest of tautomator (Apache-2.0), this file is under MOIST's license: it adapts
# code of MOIST's python/moist/pyscf.py (PySCFSolvation.__init__, moist_for_scf and
# _MoistState, PySCFHost.structure) and follows its host routines closely where it
# re-implements them; see NOTICE.
"""MOIST's DROP cavities on a PySCF mean field, with FastPCM's integral handling.

MOIST (packages/moist, in the default pixi environment) builds the cavity, solves the PCM and
chains the cavity's motion into the nuclear gradient, in Fortran.  moist.pyscf's host answers
MOIST's requests with PySCF integrals, and almost all of the time goes there; MoistHost
answers the same requests sooner, and attach() puts MOIST on a mean field with it:

- SCF: the Coulomb integrals (k|mn) between the AO pairs and the surface's Gaussian charges
  are computed once per surface and kept packed, as FastPCM keeps them, so the potential on
  the surface and the Fock term are BLAS products.  Stock MOIST computes them twice per SCF
  iteration, unpacked and in one block.  When they don't fit in half the memory left
  (mol.max_memory), the host computes them twice per iteration too, but packed and in
  blocks of BLOCK_BYTES: the SCF then costs about what stock MOIST's does, in bounded memory.
- Gradient: one blocked pass of int3c2e_ip1 gives both the derivative of the electronic
  potential at each surface point and the AO-centre term, where stock MOIST runs
  int3c2e_ip2 and then int3c2e_ip1 over the whole surface.  The AO-centre term contracts the
  charges of the last response, known before the gradient phase asks for the point
  derivatives, so the pass runs once, up front.
- Bounded memory everywhere else: every request MOIST's host answers with one unbounded
  block of integrals over the whole surface (the point derivatives of the potential outside
  a gradient, its derivative with respect to the Gaussian widths, the Fock term of a surface
  that follows the density) is answered in blocks of BLOCK_BYTES, with BLAS products where
  MOIST's host uses einsum.  At def2-QZVPPD on acetaminophen MOIST's own blocks would be
  6.8 GB (k|mn), 41 GB (point derivatives) and 14 GB (width derivatives).
- ECPs: moist.pyscf refuses any molecule with an ECP, which rules out vDZP (an ECP on every
  atom but H); attach() allows them.  A geometric cavity (SvdW-, CFC-DROP, iSwiG) needs only
  the elements, and the potential only the ECP-reduced nuclear charges and the valence
  density, as PySCF's own PCM uses them.  An isodensity cavity is built from the valence
  pseudo-density, which can fall below the isovalue at an ECP nucleus; MOIST's core fill
  (on by default) fills it there from the nuclear charges the host reports.  Checked on HI
  with iodine's def2-SVP ECP: the outer surface is bitwise the same with the fill on and off
  (its nearest point 4.2 bohr from the iodine), and the gradient matches finite differences.

FastPCM's other gradient trick, int1e_grids_ip under rinv_zeta for the points that share a
Gaussian width, is not used: DROP gives every point its own width, so it would be one call
per point (4-6x slower than stock MOIST on drug-sized molecules).

MoistHost and attach() rely on moist.pyscf internals (REPLACED).  attach() uses them only when
the installed source of every one matches a MOIST release this module was checked against
(CHECKED); otherwise it attaches MOIST's own host, at its own speed and with its own ECP
refusal; it logs a warning saying so, and on a molecule with an ECP (every vDZP job) it
raises instead of letting every conformer fail on MOIST's refusal.  The nonelectrostatic
correction on rho-DROP calls three of them (DENSITY_TERMS, through density_fock and
density_gradient), which have no fallback: those refuse an unchecked MOIST.  To check a new MOIST:
run tests/test_pyscf_moist.py with it, and if MoistHost still
matches MOIST's host there, add the new fingerprints that test_replaced_functions_are_checked
prints to CHECKED.
"""

from __future__ import annotations

import functools
import inspect
import logging
import tokenize
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import moist.pyscf as mp
import numpy as np
from moist import CustomRadii, DROPParameters, Structure
from moist.configuration import CavityConfiguration
from moist.context import _resolve_context
from moist.interface import PotentialAdjointResponse, SolvationModel, SolvationModelComponent
from moist.pyscf import DROP
from numpy.typing import NDArray
from pyscf import df, gto, lib
from pyscf.lib import logger
from scipy.special import erf

from tautomator.config import (
    DROP_LSF,
    DROP_NLEB,
    DROP_PRUNE_LEVEL,
    PCM_STORE_FRACTION,
    pcm_radii,
)
from tautomator.pyscf_fast_pcm import fingerprint

Array = NDArray[np.float64]

log = logging.getLogger(__name__)

#: MOIST's smooth pruning of near-zero quadrature weights (config.DROP_PRUNE_LEVEL says why).
PRUNE_LEVEL = DROP_PRUNE_LEVEL

#: Lebedev points per sphere of the DROP surface (config.DROP_NLEB says why).
NLEB = DROP_NLEB

#: Size of one block of surface integrals, in bytes, wherever they are not kept: libcint fills
#: blocks this small 2-5x faster per point than one of a few hundred MB (int3c2e_ip1 on
#: acetaminophen's whole surface: 2.0 s in 16 MB blocks, 9.4 s in one; best of 8 to 256 MB
#: on acetaminophen and lidocaine at def2-SVP)
BLOCK_BYTES = 16e6


def drop_radii(hydrogen_radius: float = 1.1) -> CustomRadii:
    """PySCF's cavity's spheres (config.pcm_radii) as MOIST radii, by element: every DROP
    cavity is built on them, never on MOIST's own radius sets."""
    radii = pcm_radii(hydrogen_radius)
    return CustomRadii(radii[1:], numbers=np.arange(1, len(radii)))


def drop_cavity(name: str, *, hydrogen_radius: float = 1.1, **parameters: Any) -> Any:
    """A DROP cavity by name (config.DROP_LSF), on drop_radii(hydrogen_radius) with NLEB
    points per sphere and PRUNE_LEVEL pruning; ``parameters`` go to DROPParameters and
    override both."""
    parameters.setdefault("nleb", NLEB)
    parameters.setdefault("wleb_prune_level", PRUNE_LEVEL)
    return DROP(
        lsf=getattr(mp, DROP_LSF[name])(),
        parameters=DROPParameters(**parameters),
        radii=drop_radii(hydrogen_radius),
    )


def svdw_drop(*, hydrogen_radius: float = 1.1, **parameters: Any) -> Any:
    """The SvdW-DROP cavity on drop_radii(hydrogen_radius) at NLEB with PRUNE_LEVEL pruning;
    ``parameters`` go to DROPParameters."""
    return drop_cavity("SvdW-DROP", hydrogen_radius=hydrogen_radius, **parameters)


def rho_drop(**parameters: Any) -> Any:
    """The isodensity cavity (rho-DROP, rho_iso 1e-3 e/bohr^3 as MOIST's default) at NLEB, pruned.

    The surface follows the density, so it is rebuilt every SCF iteration, and every
    iteration also needs what a fixed surface needs only in the gradient: the derivatives of
    the potential at the moving points and with respect to their widths, and the Fock term
    of the surface's own response to the density.  Those, not (k|mn), are what a rho-DROP SCF
    costs; see the README for how much.
    """
    return drop_cavity("rho-DROP", **parameters)


def _replaced() -> dict[str, Callable[..., Any]]:
    """The moist.pyscf internals MoistHost and attach() reproduce or depend on, by name."""
    return {
        "PySCFHost.__init__": mp.PySCFHost.__init__,
        "PySCFHost.structure": mp.PySCFHost.structure,
        "PySCFHost.surface_potential": mp.PySCFHost.surface_potential,
        "PySCFHost.surface_potential_gradient": mp.PySCFHost.surface_potential_gradient,
        "PySCFHost._grad_phi_elec": mp.PySCFHost._grad_phi_elec,
        "PySCFHost.fock_potential": mp.PySCFHost.fock_potential,
        "PySCFHost._gradient_phi": mp.PySCFHost._gradient_phi,
        "PySCFHost._dphi_dxi": mp.PySCFHost._dphi_dxi,
        "PySCFHost._fock_lsf": mp.PySCFHost._fock_lsf,
        "PySCFHost._gradient_lsf": mp.PySCFHost._gradient_lsf,
        "PySCFHost._density_matrix": mp.PySCFHost._density_matrix,
        "PySCFHost._ao": mp.PySCFHost._ao,
        "_int3c1e": mp._int3c1e,
        "_component_index": mp._component_index,
        "PySCFSolvation.__init__": mp.PySCFSolvation.__init__,
        "PySCFSolvation.evaluate": mp.PySCFSolvation.evaluate,
        "PySCFSolvation._answer": mp.PySCFSolvation._answer,
        "PySCFSolvation._fock": mp.PySCFSolvation._fock,
        "PySCFSolvation.gradient_channels": mp.PySCFSolvation.gradient_channels,
        "_MoistState.__init__": mp._MoistState.__init__,
        "_MoistState.copy": mp._MoistState.copy,
        "_MoistState._driver": mp._MoistState._driver,
        "_MoistState._molecule_key": mp._MoistState._molecule_key,
        "moist_for_scf": mp.moist_for_scf,
    }


REPLACED = tuple(_replaced())

# Fingerprints of the REPLACED functions, with the MOIST releases that have them;
# tests/test_pyscf_moist.py passed on each release.
CHECKED: dict[str, dict[str, str]] = {
    "PySCFHost.__init__": {"b6c90f0c12ceaf61": "1.1.0a1"},
    "PySCFHost.structure": {"9c4b46ffe62bdfbf": "1.1.0a1"},
    "PySCFHost.surface_potential": {"677e67318603aebe": "1.1.0a1"},
    "PySCFHost.surface_potential_gradient": {"6633f10092b2f482": "1.1.0a1"},
    "PySCFHost._grad_phi_elec": {"ad591f1431834954": "1.1.0a1"},
    "PySCFHost.fock_potential": {"7fa9ed51827b8e71": "1.1.0a1"},
    "PySCFHost._gradient_phi": {"a024fca9455e4579": "1.1.0a1"},
    "PySCFHost._dphi_dxi": {"f64b07703c729dc9": "1.1.0a1"},
    "PySCFHost._fock_lsf": {"b223cd2fc52047d6": "1.1.0a1"},
    "PySCFHost._gradient_lsf": {"c5e1aa9273db56b5": "1.1.0a1"},
    "PySCFHost._density_matrix": {"ae774fa2f988738b": "1.1.0a1"},
    "PySCFHost._ao": {"9091301fa653ad09": "1.1.0a1"},
    "_int3c1e": {"f2a42d138a3e0fde": "1.1.0a1"},
    "_component_index": {"5812d4d8640965b2": "1.1.0a1"},
    "PySCFSolvation.__init__": {"1a55a55d81f6db79": "1.1.0a1"},
    "PySCFSolvation.evaluate": {"042abd787272eb96": "1.1.0a1"},
    "PySCFSolvation._answer": {"3795f6393aa2008e": "1.1.0a1"},
    "PySCFSolvation._fock": {"52d505a93d53244e": "1.1.0a1"},
    "PySCFSolvation.gradient_channels": {"10d8e946ea90216c": "1.1.0a1"},
    "_MoistState.__init__": {"fbf20468196806ea": "1.1.0a1"},
    "_MoistState.copy": {"be5a440261f32b5a": "1.1.0a1"},
    "_MoistState._driver": {"901729c736dc61b5": "1.1.0a1"},
    "_MoistState._molecule_key": {"e5572962cd09b510": "1.1.0a1"},
    "moist_for_scf": {"b7aafec165713d60": "1.1.0a1"},
}


def installed_fingerprints() -> dict[str, str | None]:
    """Fingerprint of each REPLACED function in the installed MOIST (None: source unreadable)."""
    out: dict[str, str | None] = {}
    for name, func in _replaced().items():
        try:
            out[name] = fingerprint(inspect.getsource(func))
        except (OSError, TypeError, SyntaxError, tokenize.TokenError):
            out[name] = None
    return out


@functools.cache
def unchecked_functions() -> tuple[str, ...]:
    """The REPLACED functions whose installed source this module was not checked against."""
    found = installed_fingerprints()
    unchecked = tuple(name for name in REPLACED if found[name] not in CHECKED.get(name, {}))
    if unchecked:
        import moist

        log.warning(
            "MOIST %s: %s not among the versions pyscf_moist was checked against; "
            "using MOIST's own PySCF host, which is slower and refuses ECPs (vDZP)",
            moist.__version__,
            ", ".join(unchecked),
        )
    return unchecked


# The REPLACED functions the nonelectrostatic correction calls on rho-DROP
# (pyscf_nonelectrostatic_drop), through density_fock and density_gradient: MOIST's host
# contracts the surface's density weights at the density it was given in host.dm.
DENSITY_TERMS = ("PySCFHost._density_matrix", "PySCFHost._fock_lsf", "PySCFHost._gradient_lsf")


def _check_density_terms() -> None:
    """Raise unless the installed MOIST's DENSITY_TERMS are versions this module checked."""
    unchecked = [name for name in DENSITY_TERMS if name in unchecked_functions()]
    if unchecked:
        import moist

        raise RuntimeError(
            f"MOIST {moist.__version__}: {', '.join(unchecked)} not among the versions the "
            "nonelectrostatic correction on rho-DROP was checked against (pyscf_moist.CHECKED)"
        )


def density_fock(host: Any, dm: Array, coords: Array, weights: Any) -> Array:
    """Density weights (MOIST's DensityResponse, from the cavity's contract_surface_density)
    contracted into a Fock matrix at density dm: MOIST host's _fock_lsf."""
    _check_density_terms()
    host.dm = np.asarray(dm, dtype=np.float64)
    return np.asarray(host._fock_lsf(coords, weights), dtype=np.float64)


def density_gradient(host: Any, dm: Array, coords: Array, weights: Any) -> Array:
    """Density weights contracted with d rho / dR at fixed dm, (natm, 3): MOIST host's
    _gradient_lsf."""
    _check_density_terms()
    host.dm = np.asarray(dm, dtype=np.float64)
    return np.asarray(host._gradient_lsf(coords, weights), dtype=np.float64)


def _key(coords: Array, xi: Array) -> tuple[bytes, bytes]:
    """Identity of a surface: its point positions and Gaussian widths."""
    return np.ascontiguousarray(coords).tobytes(), np.ascontiguousarray(xi).tobytes()


class MoistHost(mp.PySCFHost):  # type: ignore[misc]
    """The PySCF host that answers MOIST's requests: moist.pyscf.PySCFHost with (k|mn) kept
    per surface and a one-pass gradient."""

    def __init__(self, mol: Any, *, allow_ecp: bool = False) -> None:
        super().__init__(mol)
        self.allow_ecp = allow_ecp
        # (surface key, packed (k|mn) (npair, npts), pair weights (npair,)), or the key and
        # None where the integrals don't fit
        self._packed: tuple[tuple[bytes, bytes], Array | None, Array] | None = None
        # (surface key, charges, sum_i q_i (d mu nu|i) (3, nao, nao), (d mu nu|i).(D+D^T)
        # (3, npts)) from the up-front gradient pass, while a gradient runs
        self._fused: tuple[tuple[bytes, bytes], Array, Array, Array] | None = None

    # ------------------------------------------------------------------ geometry
    def structure(self) -> Any:
        if not self.allow_ecp:
            return super().structure()
        numbers = [gto.charge(self.mol.atom_pure_symbol(i)) for i in range(self.mol.natm)]
        return Structure(numbers, self.mol.atom_coords(), charge=self.mol.charge)

    # ------------------------------------------------------------------ memory
    def _budget_bytes(self) -> float:
        """PCM_STORE_FRACTION (half) of what mol.max_memory leaves after the process, in bytes."""
        return PCM_STORE_FRACTION * float(self.mol.max_memory - lib.current_memory()[0]) * 1e6

    def _blocksize(self, per_point: float) -> int:
        """Points per block of integrals that are not kept: BLOCK_BYTES, or the budget if less."""
        return int(max(1, min(BLOCK_BYTES, self._budget_bytes()) // per_point))

    def _integrals(self, coords: Array, xi: Array) -> tuple[Array | None, Array]:
        """(k|mn) packed (npair, npts), kept per surface (None if too big), and pair weights."""
        key = _key(coords, xi)
        if self._packed is None or self._packed[0] != key:
            mol = self.mol
            nao = mol.nao
            npair = nao * (nao + 1) // 2
            self._packed = None  # free the last surface's before sizing this one
            weights = lib.pack_tril(1.0 - 0.5 * np.eye(nao))
            if xi.size * npair * 8 > self._budget_bytes():
                logger.info(
                    mol,
                    "MoistHost: (k|mn) (%.0f MB) does not fit; computed in blocks per use",
                    xi.size * npair * 8 / 1e6,
                )
                self._packed = (key, None, weights)
            else:
                fakemol = gto.fakemol_for_charges(np.asarray(coords), expnt=np.asarray(xi) ** 2)
                fakemol.cart = mol.cart
                int3c2e = mol._add_suffix("int3c2e")
                B = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym="s2ij")
                logger.info(mol, "MoistHost: (k|mn) in core, %.0f MB", B.nbytes / 1e6)
                self._packed = (key, B, weights)
        _, B, weights = self._packed
        return B, weights

    def _coulomb_blocks(self, coords: Array, xi: Array) -> Iterator[tuple[int, int, Array]]:
        """(p0, p1, (k|mn) packed (npair, p1 - p0)) over the surface, block by block."""
        mol = self.mol
        nao = mol.nao
        int3c2e = mol._add_suffix("int3c2e")
        for p0, p1 in lib.prange(0, xi.size, self._blocksize(nao * (nao + 1) // 2 * 8)):
            fakemol = gto.fakemol_for_charges(coords[p0:p1], expnt=xi[p0:p1] ** 2)
            fakemol.cart = mol.cart
            yield p0, p1, df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym="s2ij")

    # ------------------------------------------------------------------ SCF
    def surface_potential(self, coords: Array, xi: Array | None = None) -> Array:
        """The total potential at the surface points (MOIST's), from (k|mn) kept or blocked."""
        if xi is None:
            return super().surface_potential(coords, xi)  # type: ignore[no-any-return]
        coords, xi = np.asarray(coords), np.asarray(xi)
        B, weights = self._integrals(coords, xi)
        dm = self._density_matrix()
        d = lib.pack_tril(dm + dm.T) * weights
        if B is not None:
            electronic = d @ B
        else:
            electronic = np.empty(xi.size)
            for p0, p1, block in self._coulomb_blocks(coords, xi):
                electronic[p0:p1] = d @ block
        dist = np.linalg.norm(coords[:, None, :] - self.mol.atom_coords()[None], axis=2)
        width = xi[:, None]
        # erf(xi r)/r, and its limit 2 xi/sqrt(pi) at r = 0
        kernel = np.divide(
            erf(width * dist),
            dist,
            out=np.broadcast_to(2.0 * width / np.sqrt(np.pi), dist.shape).copy(),
            where=dist != 0,
        )
        return kernel @ self.mol.atom_charges() - electronic  # type: ignore[no-any-return]

    def fock_potential(self, coords: Array, xi: Array, w_phi: Array) -> Array:
        """sum_i w_phi_i dphi_i/dP = -sum_i w_phi_i (i|mn), from (k|mn) kept or blocked."""
        coords, xi, w = np.asarray(coords), np.asarray(xi), np.asarray(w_phi)
        B, _ = self._integrals(coords, xi)
        if B is not None:
            packed = B @ w
        else:
            nao = self.mol.nao
            packed = np.zeros(nao * (nao + 1) // 2)
            for p0, p1, block in self._coulomb_blocks(coords, xi):
                packed += block @ w[p0:p1]
        return -lib.unpack_tril(packed)  # type: ignore[no-any-return]

    def _dphi_dxi(self, coords: Array, xi: Array) -> Array:
        """d(potential)/d(Gaussian width) at the points (MOIST's), the overlaps in blocks."""
        coords, xi = np.asarray(coords), np.asarray(xi)
        mol = self.mol
        delta = coords[:, None, :] - mol.atom_coords()[None, :, :]
        r2 = np.sum(delta * delta, axis=2)
        nuclear = (2.0 / np.sqrt(np.pi)) * (np.exp(-(xi[:, None] ** 2) * r2) @ mol.atom_charges())
        dm = self._density_matrix()
        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized="sp")
            dm = c2s @ dm @ c2s.T
        ncart = dm.shape[0]
        d = dm.ravel()
        electronic = np.empty(xi.size)
        for p0, p1 in lib.prange(0, xi.size, self._blocksize(ncart * ncart * 8)):
            # 3-centre overlaps with the unnormalized s probes, (ncart, ncart, p1 - p0)
            block = mp._int3c1e(mol, coords[p0:p1], xi[p0:p1] ** 2, 0)
            electronic[p0:p1] = d @ block.reshape(ncart * ncart, p1 - p0)
        # The unnormalized s probe carries 1/(2 sqrt(pi)) and the kernel derivative 2/sqrt(pi):
        # MOIST's factor of four
        return nuclear - 4.0 * electronic  # type: ignore[no-any-return]

    def _fock_lsf(self, coords: Array, density: Any) -> Array:
        """d rho/dP contracted with the density adjoints (MOIST's), by BLAS, in blocks.

        MOIST's sum of weighted outer products of AO derivative blocks, sum_i w_i L_iu R_iv,
        each as L^T (w R); the products with the AO values themselves are summed first.
        """
        coords = np.asarray(coords)
        w0 = np.asarray(density.w_rho)
        w1 = np.asarray(density.w_grad_rho).T  # (3, ngrid)
        w2 = np.asarray(density.w_hess_rho).T  # (3, 3, ngrid), indexed as MOIST indexes it
        first = [mp._component_index((a,)) for a in range(3)]
        second = [[mp._component_index((a, b)) for b in range(3)] for a in range(3)]
        nao = self.mol.nao
        fock = np.zeros((nao, nao))
        half = np.zeros((nao, nao))  # the terms MOIST symmetrises
        for p0, p1 in lib.prange(0, coords.shape[0], self._blocksize(10 * nao * 8)):
            ao = self._ao(coords[p0:p1], 2)  # (10, points, nao)
            a0 = ao[0]
            fock += a0.T @ (w0[p0:p1, None] * a0)
            with_a0 = np.zeros_like(a0)
            for a in range(3):
                with_a0 += w1[a, p0:p1, None] * ao[first[a]]
                for b in range(3):
                    with_a0 += w2[a, b, p0:p1, None] * ao[second[a][b]]
            half += with_a0.T @ a0
            for a in range(3):
                right = sum(w2[a, b, p0:p1, None] * ao[first[b]] for b in range(3))
                half += ao[first[a]].T @ right
        return fock + half + half.T

    # ------------------------------------------------------------------ gradient
    def _ip1_pass(self, coords: Array, xi: Array, q: Array) -> tuple[Array, Array]:
        """One blocked pass of int3c2e_ip1: (sum_i q_i T (3, nao, nao), T . (D + D^T) per point).

        With T = (d_x mu nu|i), those are what the AO-centre term and the point derivatives
        need: for symmetric D, sum_mn D_mn (mn|d i) = -T . (D + D^T).
        """
        mol = self.mol
        nao = mol.nao
        dm = self._density_matrix()
        dsym = (dm + dm.T).ravel()
        vq = np.zeros((3, nao * nao))
        dv = np.empty((3, xi.size))
        intor = mol._add_suffix("int3c2e_ip1")
        for p0, p1 in lib.prange(0, xi.size, self._blocksize(3 * nao * nao * 8)):
            fakemol = gto.fakemol_for_charges(coords[p0:p1], expnt=xi[p0:p1] ** 2)
            fakemol.cart = mol.cart
            t = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s1", comp=3)
            t = t.reshape(3, nao * nao, p1 - p0)
            vq += t @ q[p0:p1]
            dv[:, p0:p1] = dsym @ t
        return vq.reshape(3, nao, nao), dv

    def fused_pass(self, coords: Array, xi: Array, q: Array) -> None:
        """Run the gradient's one pass for charges q; _grad_phi_elec and _gradient_phi read it."""
        coords, xi, q = np.asarray(coords), np.asarray(xi), np.asarray(q)
        vq, dv = self._ip1_pass(coords, xi, q)
        self._fused = (_key(coords, xi), q.copy(), vq, dv)

    def clear_fused(self) -> None:
        self._fused = None

    def _grad_phi_elec(self, coords: Array, xi: Array | None = None) -> Array:
        """d(electronic potential)/d(point position), (3, npts): fused pass, or a blocked one."""
        if xi is None:
            return super()._grad_phi_elec(coords, xi)  # type: ignore[no-any-return]
        coords, xi = np.asarray(coords), np.asarray(xi)
        f = self._fused
        if f is not None and f[0] == _key(coords, xi):
            return -f[3]
        return self._point_derivatives(coords, xi)

    def _point_derivatives(self, coords: Array, xi: Array) -> Array:
        """sum_mn D_mn (mn|d i) per point (MOIST's int3c2e_ip2 contraction), packed, in blocks.

        The derivative falls on the surface Gaussian, so (mn|d i) is symmetric in m, n and is
        computed packed: half the integrals of an int3c2e_ip1 pass (0.9 s against 1.9 s on
        acetaminophen's surface at def2-SVP).  Every rho-DROP SCF iteration needs it.
        """
        mol = self.mol
        nao = mol.nao
        dm = self._density_matrix()
        d = lib.pack_tril(dm + dm.T) * lib.pack_tril(1.0 - 0.5 * np.eye(nao))
        out = np.empty((3, xi.size))
        intor = mol._add_suffix("int3c2e_ip2")
        for p0, p1 in lib.prange(0, xi.size, self._blocksize(3 * nao * (nao + 1) // 2 * 8)):
            fakemol = gto.fakemol_for_charges(coords[p0:p1], expnt=xi[p0:p1] ** 2)
            fakemol.cart = mol.cart
            out[:, p0:p1] = d @ df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", comp=3)
        return out

    def _gradient_phi(self, coords: Array, q: Array, xi: Array | None = None) -> Array:
        """sum_i q_i d(phi_elec)_i/dR_A at fixed points and density, (natm, 3)."""
        if xi is None:
            return super()._gradient_phi(coords, q, xi)  # type: ignore[no-any-return]
        coords, xi, q = np.asarray(coords), np.asarray(xi), np.asarray(q)
        f = self._fused
        if f is not None and f[0] == _key(coords, xi) and np.array_equal(q, f[1]):
            vq = f[2]
        else:
            vq, _ = self._ip1_pass(coords, xi, q)
        weighted = 2.0 * np.einsum("xuv,uv->xu", vq, self._density_matrix())
        grad = np.zeros((3, self.mol.natm))
        for atom, (lo, hi) in enumerate(self._aoslice[:, 2:4]):
            grad[:, atom] = weighted[:, lo:hi].sum(axis=1)
        return np.ascontiguousarray(grad.T)


class MoistSolvation(mp.PySCFSolvation):  # type: ignore[misc]
    """moist.pyscf.PySCFSolvation on a MoistHost, with the gradient's one pass up front."""

    def __init__(
        self,
        mol: Any,
        cavity: Any,
        components: Sequence[Any],
        *,
        context: Any,
        allow_ecp: bool = False,
    ) -> None:
        # PySCFSolvation.__init__ with MoistHost in place of PySCFHost
        if not isinstance(cavity, CavityConfiguration):
            raise TypeError("cavity must be DROP(lsf=...) or ISwiG(...)")
        items = tuple(components)
        if not items or any(not isinstance(item, SolvationModelComponent) for item in items):
            raise TypeError("components must be a nonempty sequence of MOIST components")
        self.mol = mol
        self.configuration = cavity
        self.components = items
        self.context = _resolve_context(context)
        self.host = MoistHost(mol, allow_ecp=allow_ecp)
        self.host.structure()
        live = cavity.build(basis=self.host.basis) if cavity.density_dependent else cavity.build()
        self.model = SolvationModel(self.context, live, items)
        self.coupling = None
        self.moments = None
        self._dm = None
        self._result = None
        self._response = None
        self._xi = None

    def gradient_channels(self, dm: Array) -> dict[str, Array]:
        self.evaluate(dm)
        w_phi = [
            np.asarray(item.w_phi)
            for item in self.response
            if isinstance(item, PotentialAdjointResponse)
        ]
        if self._xi is not None and len(w_phi) == 1:
            self.host.dm = self._dm
            cavity = self.model.cavity
            self.host.fused_pass(cavity.xyz, self._xi, w_phi[0])
        try:
            return super().gradient_channels(dm)  # type: ignore[no-any-return]
        finally:
            self.host.clear_fused()


class _MoistSolvationState(mp._MoistState):  # type: ignore[misc]
    """moist.pyscf's per-calculation state, driving a MoistSolvation."""

    # set by _MoistState.reset
    _fingerprint: Any
    _solvation: MoistSolvation | None

    def __init__(
        self,
        mol: Any,
        *,
        cavity: Any,
        components: Sequence[Any],
        context: Any,
        allow_ecp: bool = False,
    ) -> None:
        self.allow_ecp = allow_ecp
        super().__init__(mol, cavity=cavity, components=components, context=context)

    def copy(self) -> _MoistSolvationState:
        return type(self)(
            self.mol,
            cavity=self.cavity,
            components=self.components,
            context=self.context,
            allow_ecp=self.allow_ecp,
        )

    def _molecule_key(self) -> tuple[Any, ...]:
        """MOIST's key for "same molecule", without the slots that only say where PySCF's
        one-electron operators were last centred: a gradient moves them (its ECP and
        nuclear-attraction derivatives), and MOIST would rebuild the cavity and every kept
        integral for the first gradient after each SCF."""
        mol = self.mol
        env = np.array(mol._env, copy=True)
        env[gto.PTR_COMMON_ORIG : gto.PTR_RINV_ZETA + 1] = 0.0
        env[gto.AS_RINV_ORIG_ATOM] = 0.0
        return (
            mol._atm.tobytes(),
            mol._bas.tobytes(),
            env.tobytes(),
            mol.cart,
            mol.charge,
            mol.spin,
        )

    def _driver(self) -> MoistSolvation:
        key = self._molecule_key()
        if self._fingerprint != key:
            self.reset()
        if self._solvation is None:
            if self.mol.has_ecp() and not self.allow_ecp:
                raise ValueError("effective core potentials are not supported")
            self._solvation = MoistSolvation(
                self.mol,
                self.cavity,
                self.components,
                context=self.context,
                allow_ecp=self.allow_ecp,
            )
            self._fingerprint = key
        return self._solvation


def attach(mf: Any, *, cavity: Any, components: Sequence[Any], context: Any) -> Any:
    """MOIST on RHF/RKS/UHF/UKS ``mf``, with MoistHost; the SCF is not run.

    moist.pyscf.moist_for_scf with a MoistHost, and with ECPs allowed (see the module
    docstring).  When the installed MOIST has an unchecked version of a REPLACED function,
    this is moist_for_scf itself, which refuses ECPs.
    """
    from pyscf import scf
    from pyscf.soscf.newton_ah import _CIAH_SOSCF

    unchecked = unchecked_functions()
    if unchecked:
        if mf.mol.has_ecp():
            import moist

            raise ValueError(
                f"MOIST {moist.__version__} is not a release tautomator.pyscf_moist was "
                f"checked against ({', '.join(unchecked)} changed), so only MOIST's own PySCF "
                "code could run, and it refuses ECPs, which this molecule has (vDZP puts one on "
                "every atom but H).  Run tests/test_pyscf_moist.py with it; "
                "if it passes, add the fingerprints it prints to CHECKED in "
                "tautomator/pyscf_moist.py."
            )
        return mp.moist_for_scf(mf, cavity=cavity, components=components, context=context)
    # moist_for_scf's checks, without its ECP refusal
    if isinstance(mf, mp._MoistSCF):
        raise ValueError("MOIST is already attached; combine terms in components")
    if isinstance(mf, (scf.rohf.ROHF, _CIAH_SOSCF)) or not isinstance(
        mf, (scf.hf.RHF, scf.uhf.UHF)
    ):
        raise NotImplementedError("MOIST supports RHF, RKS, UHF and UKS calculations")
    if getattr(mf, "with_solvent", None) is not None:
        raise ValueError("Cannot attach MOIST to a calculation with an existing solvent")
    state = _MoistSolvationState(
        mf.mol, cavity=cavity, components=components, context=context, allow_ecp=True
    )
    obj = mf.copy()
    obj.with_moist = state
    obj.converged = False
    obj.scf_summary = dict(mf.scf_summary)
    return lib.set_class(obj, (mp._MoistSCF, mf.__class__), "MOIST" + mf.__class__.__name__)
