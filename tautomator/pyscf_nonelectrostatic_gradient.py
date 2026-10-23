"""Analytic nuclear gradient of the post-SCF nonelectrostatic correction (PySCF).

The correction G(D, R) is evaluated on the converged density of a PCM SCF that
does not include it, so its total derivative needs the density's response:

    dG/dR = dG/dR|_D + tr(V dD/dR),   V = dG/dD.

The response comes from one Z-vector (CPHF) solve with the SCF's own orbital
Hessian, the PCM's equilibrium response included.  For a closed-shell SCF,
with D = 2 C_o C_o^T, canonical orbitals and real coefficients:

    L z = -4 V_vo,   L(U)_ai = (e_a - e_i) U_ai + [v(2 (C_v U C_o^T + h.c.))]_ai

v is the change in the Fock matrix for a change in the density matrix (Coulomb,
exchange, the XC kernel and the PCM).  With Z = C_v z C_o^T + h.c.,

    tr(V dD/dR) = 1/2 tr(Z F^R(D)) - tr(W S^R),
    W = C_o X C_o^T + C_v z diag(e_o) C_o^T (symmetrized),
    X = 2 C_o^T V C_o + C_o^T v(Z) C_o,

where F^R(D) is the derivative of the AO Fock matrix at fixed density matrix
(core Hamiltonian with ECPs, Coulomb and exchange, XC, PCM) and S^R that of the
overlap.  1/2 tr(Z F^R(D)) is a bilinear derivative.  Its Coulomb, exchange and
PCM parts are quadratic forms in the density, so they come exactly from PySCF's
own SCF gradient code at D + tZ and D - tZ (the polarization identity, any t);
its XC part is the kernel contraction on the SCF's grid, with the grid's
response (which PySCF's SCF gradient leaves out by default).

Supported: RKS and RHF with density fitting, LDA and GGA functionals including
hybrids and range-separated hybrids, IEF-PCM, C-PCM and COSMO (FastPCM or
PySCF's own).  Meta-GGA and VV10 functionals are refused: PySCF has no response
kernel for VV10, and the meta-GGA kernel contraction is not written.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import lib
from pyscf.scf import cphf

from tautomator.pyscf_nonelectrostatic import (
    NonelectrostaticTerms,
    _ao_atoms,
    basis_motion_gradient,
)

Array = NDArray[np.float64]

# Z-vector convergence: PySCF's Krylov solver stops once its next basis vector is
# shorter than CPHF_TOL (absolute, in the units of z), and raises RuntimeError if
# that takes more than CPHF_MAX_CYCLE iterations.
CPHF_TOL = 1e-10
CPHF_MAX_CYCLE = 100


def _check_supported(mf: Any) -> None:
    from pyscf import dft, scf

    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError(
            f"post-SCF nonelectrostatic gradient for {type(mf).__name__}: closed shell only"
        )
    if not getattr(mf, "converged", False):
        raise ValueError("the SCF is not converged")
    if isinstance(mf, dft.rks.KohnShamDFT):
        xctype = mf._numint._xc_type(mf.xc)
        if xctype not in ("LDA", "GGA", "HF"):
            raise NotImplementedError(f"{mf.xc}: {xctype} functionals are not supported")
        if mf.do_nlc():
            raise NotImplementedError(f"{mf.xc}: PySCF has no response kernel for VV10")


def _rows_by_atom(mol: Any) -> list[slice]:
    return [slice(a0, a1) for _s0, _s1, a0, a1 in mol.aoslice_by_atom()]


def _two_electron_gradient(mf_grad: Any, X: Array) -> Array:
    """d/dR of 1/2 tr(X G(X)), G = J - (exchange), at fixed X: PySCF's SCF gradient code
    for the Coulomb and exchange part of an RKS (or RHF) energy, auxiliary basis included."""
    mf, mol = mf_grad.base, mf_grad.mol
    ni = getattr(mf, "_numint", None)
    if ni is None or not hasattr(mf, "xc"):
        omega, alpha, hyb = 0.0, 0.0, 1.0
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    aux_response = getattr(mf_grad, "auxbasis_response", False)
    if hyb == 0 and alpha == 0:
        vj = mf_grad.get_j(mol, X)
        vhf = vj
        aux = vj.aux.sum((0, 1)) if aux_response else None
    else:
        if omega == 0:
            vj, vk = mf_grad.get_jk(mol, X)
            vk_scaled, aux_k = vk * hyb, (vk.aux * hyb if aux_response else None)
        elif alpha == 0:
            vj = mf_grad.get_j(mol, X)
            vk = mf_grad.get_k(mol, X, omega=-omega)
            vk_scaled, aux_k = vk * hyb, (vk.aux * hyb if aux_response else None)
        elif hyb == 0:
            vj = mf_grad.get_j(mol, X)
            vk = mf_grad.get_k(mol, X, omega=omega)
            vk_scaled, aux_k = vk * alpha, (vk.aux * alpha if aux_response else None)
        else:
            vj, vk = mf_grad.get_jk(mol, X)
            vklr = mf_grad.get_k(mol, X, omega=omega)
            vk_scaled = vk * hyb + vklr * (alpha - hyb)
            aux_k = vk.aux * hyb + vklr.aux * (alpha - hyb) if aux_response else None
        vhf = vj - 0.5 * vk_scaled
        aux = (vj.aux - 0.5 * aux_k).sum((0, 1)) if aux_k is not None else None
    g = np.zeros((mol.natm, 3))
    for atom, rows in enumerate(_rows_by_atom(mol)):
        g[atom] = 2.0 * np.einsum("xij,ij->x", vhf[:, rows], X[rows])
    if aux is not None:
        g += aux
    return g


def _pcm_gradient(ws: Any, X: Array) -> Array:
    """The PCM energy's gradient at fixed density matrix X (charges solved for X)."""
    ws._get_vind(X)
    return np.asarray(ws.grad(X))


def _xc_bilinear_gradient(mf: Any, D: Array, Z: Array) -> Array:
    """d/dR tr(Z V_xc(D)) at fixed D and Z, with the grid's response.

    tr(Z V_xc(D)) = sum_p w_p Phi(r_p), Phi = v(u_D) . u_Z with u = (rho, grad rho).
    Its derivative has three parts: the basis functions moving,
    v(u_D) . du_Z/dR + (f(u_D) u_Z) . du_D/dR; the points moving with their
    atom, which by translational invariance is minus the sum of the first over
    all atoms; and the weights changing (PySCF's grids_response_cc).  The SCF's
    own gradient in PySCF leaves the last two out by default (grid_response);
    with them this is the exact derivative of tr(Z V_xc(D)) on the SCF's grid.
    """
    from pyscf import dft
    from pyscf.grad.rks import grids_response_cc

    mol = mf.mol
    g = np.zeros((mol.natm, 3))
    if not isinstance(mf, dft.rks.KohnShamDFT):
        return g
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        return g
    n_u = 1 if xctype == "LDA" else 4
    ao_deriv = 1 if xctype == "LDA" else 2
    n_comp = 4 if ao_deriv == 1 else 10
    block = max(256, int(2e8 / (8 * n_comp * mol.nao)))
    ao_atom = _ao_atoms(mol)
    for atom, (coords, weights, weights1) in enumerate(grids_response_cc(mf.grids)):
        for p0 in range(0, len(weights), block):
            sl = slice(p0, p0 + block)
            w = weights[sl]
            ao = ni.eval_ao(mol, coords[sl], deriv=ao_deriv)
            ao_u = ao[:n_u] if n_u > 1 else ao[0]
            u_D = ni.eval_rho(mol, ao_u, D, xctype=xctype, hermi=1)
            u_Z = np.asarray(ni.eval_rho(mol, ao_u, Z, xctype=xctype, hermi=1)).reshape(n_u, -1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, u_D, deriv=2, xctype=xctype)[1:3]
            vxc = np.asarray(vxc).reshape(n_u, -1)
            fxc = np.asarray(fxc).reshape(n_u, n_u, -1)
            moving = basis_motion_gradient(mol, ao, Z, w * vxc, ao_atom=ao_atom)
            moving += basis_motion_gradient(
                mol, ao, D, w * np.einsum("kln,ln->kn", fxc, u_Z), ao_atom=ao_atom
            )
            g += moving
            g[atom] -= moving.sum(axis=0)
            g += np.einsum("r,nxr->nx", np.einsum("kr,kr->r", vxc, u_Z), weights1[:, :, sl])
    return g


def post_scf_gradient(mf: Any, terms: NonelectrostaticTerms) -> Array:
    """dG/dR (natm, 3) of the correction on the converged density of ``mf``.

    ``mf`` is a converged closed-shell PCM mean field (`pyscf_runner._build_mf`),
    ``terms`` the correction at its geometry.  Only G's gradient: the SCF's own
    is ``mf.nuc_grad_method().kernel()``.
    """
    _check_supported(mf)
    D = np.asarray(mf.make_rdm1(), dtype=np.float64)
    _energies, V = terms.energy_and_potential(D)
    assert V is not None
    g = terms.gradient(D)
    ws = mf.with_solvent

    def pcm_bilinear(D: Array, Z: Array) -> Array:
        # The PCM energy's gradient at fixed density is a quadratic form in it, so the
        # polarization identity gives its bilinear part exactly, for any t.
        t = float(np.linalg.norm(D) / max(np.linalg.norm(Z), 1e-300))
        plus = _pcm_gradient(ws, D + t * Z)
        minus = _pcm_gradient(ws, D - t * Z)
        ws._get_vind(D)  # leave the PCM's charges at the SCF density
        return np.asarray((plus - minus) / (2.0 * t))

    with lib.temporary_env(ws, equilibrium_solvation=True):
        vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)
        g += z_vector_gradient(mf, D, V, vresp, pcm_bilinear)
    return g


def z_vector_gradient(
    mf: Any,
    D: Array,
    V: Array,
    vresp: Callable[[Array], Array],
    solvent_bilinear: Callable[[Array, Array], Array],
) -> Array:
    """tr(V dD/dR) (natm, 3) for a closed-shell SCF converged at D (module docstring).

    ``vresp`` is the SCF's response to a symmetric change in the density matrix, the
    solvent's equilibrium response included; ``solvent_bilinear(D, Z)`` the derivative,
    with respect to the nuclei at fixed D and Z, of tr(Z dE_solv/dD), i.e. of the solvent's
    contribution to the Fock matrix contracted with Z.
    """
    mol = mf.mol
    mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    occ = mo_occ > 0
    C_o, C_v = mo_coeff[:, occ], mo_coeff[:, ~occ]

    def fvind(x: Array) -> Array:
        x = x.reshape(C_v.shape[1], C_o.shape[1])
        dm1 = 2.0 * C_v @ x @ C_o.T
        v = vresp(dm1 + dm1.T)
        return np.asarray(C_v.T @ v @ C_o).ravel()

    rhs = 4.0 * C_v.T @ V @ C_o
    z = cphf.solve(fvind, mo_energy, mo_occ, rhs, max_cycle=CPHF_MAX_CYCLE, tol=CPHF_TOL)[
        0
    ].reshape(rhs.shape)
    Z_ao = C_v @ z @ C_o.T
    Z = Z_ao + Z_ao.T
    X = 2.0 * C_o.T @ V @ C_o + C_o.T @ vresp(Z) @ C_o
    W = C_o @ X @ C_o.T + C_v @ z @ np.diag(mo_energy[occ]) @ C_o.T
    W = 0.5 * (W + W.T)

    g = np.zeros((mol.natm, 3))
    mf_grad = mf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    for atom, rows in enumerate(_rows_by_atom(mol)):
        g[atom] += 0.5 * np.einsum("xij,ij->x", hcore_deriv(atom), Z)
        g[atom] -= 2.0 * np.einsum("xij,ij->x", s1[:, rows], W[rows])

    # The quadratic parts of 1/2 tr(Z F^R(D)): B(Z, D) = [Q(D + tZ) - Q(D - tZ)] / 2t
    # for Q the gradient of the quadratic form at fixed density.  t only balances
    # rounding; any value gives the same B.
    t = float(np.linalg.norm(D) / max(np.linalg.norm(Z), 1e-300))
    plus, minus = D + t * Z, D - t * Z
    g += (
        0.5
        * (_two_electron_gradient(mf_grad, plus) - _two_electron_gradient(mf_grad, minus))
        / (2.0 * t)
    )
    g += 0.5 * solvent_bilinear(D, Z)
    g += 0.5 * _xc_bilinear_gradient(mf, D, Z)
    return g
