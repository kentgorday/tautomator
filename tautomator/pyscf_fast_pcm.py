# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
# Copyright 2026 Kent Gorday (modifications).
# SPDX-License-Identifier: Apache-2.0
#
# FastPCM re-implements PCM._get_vind, _B_dot_x, _get_v, _get_vmat and grad from PySCF's
# pyscf/solvent/pcm.py, and grad_qv from pyscf/solvent/grad/pcm.py (Author: Xiaojie Wu
# <wxj6000@gmail.com>); PySCF's attribution notices are in NOTICE.
"""Faster PySCF PCM: the results of pyscf.solvent.pcm.PCM, several times sooner.

Stock PCM recomputes the surface integrals (k|mn) every SCF iteration, contracts them with
numpy.einsum and does two dense solves with K.  FastPCM takes the cavity, K and R from
PCM.build unchanged, then, once per geometry, LU-factors K and (when they fit in
config.PCM_STORE_FRACTION, half, of the memory left) stores the integrals packed, so an
iteration is two BLAS products and two LU solves.  When they don't fit, _get_v and
_get_vmat are PySCF's, which recompute them every iteration, so the SCF runs at stock
speed (the README has sizes by molecule).

Its gradient computes PySCF's grad_qv term from one pass of libcint's int1e_grids_ip: a
Gaussian charge of exponent xi^2 is a point charge under rinv_zeta = xi^2, so the points
sharing a xi are the grid of one one-electron operator (the formulation of gpu4pyscf's
int3c1e kernels), where PySCF uses int3c2e_ip1 and int3c2e_ip2 over one fake s shell per
point.

FastPCM replaces six PySCF functions (REPLACED).  use_fast_pcm switches a PCM to it only
when the source of every one of them matches a version FastPCM was checked against
(CHECKED); otherwise the PCM is left as it is, so any PySCF runs, at stock speed where
FastPCM has not been checked.  To check a new PySCF: run tests/test_pyscf_fast_pcm.py with
it, and if FastPCM still matches stock PCM there, add the new fingerprints that
test_replaced_functions_are_checked prints to CHECKED.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import logging
import textwrap
import tokenize
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyscf import df, gto, lib
from pyscf.lib import logger
from pyscf.solvent import pcm
from scipy.linalg import lu_factor, lu_solve

from tautomator.config import PCM_STORE_FRACTION

Array = NDArray[np.float64]

log = logging.getLogger(__name__)


def _replaced() -> dict[str, Callable[..., Any] | None]:
    """The PySCF functions FastPCM stands in for, by name (None where this PySCF has none)."""
    from pyscf.solvent.grad import pcm as pcm_grad

    return {
        "PCM._get_vind": getattr(pcm.PCM, "_get_vind", None),
        "PCM._B_dot_x": getattr(pcm.PCM, "_B_dot_x", None),
        "PCM._get_v": getattr(pcm.PCM, "_get_v", None),
        "PCM._get_vmat": getattr(pcm.PCM, "_get_vmat", None),
        "PCM.grad": getattr(pcm.PCM, "grad", None),
        "grad.pcm.grad_qv": getattr(pcm_grad, "grad_qv", None),
    }


REPLACED = tuple(_replaced())

# Fingerprints of the REPLACED functions FastPCM reproduces, with the PySCF releases that
# have them; tests/test_pyscf_fast_pcm.py passed on each release.  PySCF before 2.10 has no
# PCM.grad: its gradient driver calls grad_qv itself, so there FastPCM speeds up the SCF only.
CHECKED: dict[str, dict[str, str]] = {
    "PCM._get_vind": {"e2dcaea13299f19d": "2.7.0-2.14.0"},
    "PCM._B_dot_x": {"dd6edbf7414ec76f": "2.7.0-2.14.0"},
    "PCM._get_v": {"cd2233e59ea0c4b0": "2.7.0-2.12.1", "daa7491e1983eeef": "2.13.0-2.14.0"},
    "PCM._get_vmat": {"505ae748eb0a1346": "2.7.0-2.12.1", "98a93a29100daa58": "2.13.0-2.14.0"},
    "PCM.grad": {"absent": "2.7.0-2.9.0", "9fe5940c6431ee72": "2.10.0-2.14.0"},
    "grad.pcm.grad_qv": {
        "b2a28b60cfa6ca5a": "2.7.0-2.8.0",
        "d3b57a51e79f4981": "2.9.0-2.12.1",
        "f1831dbbcc38aeee": "2.13.0-2.13.1",
        "45ffa968d17d1036": "2.14.0",
    },
}


def fingerprint(source: str) -> str:
    """Hash of a function's tokens: comments, blank lines and indentation width don't count."""
    blank = {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.ENDMARKER}
    tokens = tokenize.generate_tokens(io.StringIO(textwrap.dedent(source)).readline)
    kept = [
        f"{tok.type}:{'' if tok.type in blank else tok.string}"
        for tok in tokens
        if tok.type not in (tokenize.COMMENT, tokenize.NL)
    ]
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()[:16]


def installed_fingerprints() -> dict[str, str | None]:
    """Fingerprint of each REPLACED function in the installed PySCF.

    "absent" where this PySCF has no such function, None where its source can't be read.
    """
    out: dict[str, str | None] = {}
    for name, func in _replaced().items():
        try:
            out[name] = "absent" if func is None else fingerprint(inspect.getsource(func))
        except (OSError, TypeError, SyntaxError, tokenize.TokenError):
            out[name] = None
    return out


@functools.cache
def unchecked_functions() -> tuple[str, ...]:
    """The REPLACED functions whose installed source FastPCM was not checked against."""
    found = installed_fingerprints()
    unchecked = tuple(name for name in REPLACED if found[name] not in CHECKED.get(name, {}))
    if unchecked:
        from pyscf import __version__

        log.info(
            "PySCF %s: %s not among the versions FastPCM was checked against; "
            "using PySCF's own PCM",
            __version__,
            ", ".join(unchecked),
        )
    return unchecked


def use_fast_pcm(mf: Any) -> bool:
    """Switch mf.with_solvent from pyscf.solvent.pcm.PCM to FastPCM, keeping its settings.

    Returns whether mf.with_solvent is a FastPCM now.  Does nothing for another solvent model
    (SMD and ddCOSMO included), for one that is a FastPCM already, or when the installed PySCF
    has an unchecked version of a REPLACED function.
    """
    ws: Any = getattr(mf, "with_solvent", None)
    if type(ws) is FastPCM:
        return True
    if type(ws) is not pcm.PCM or unchecked_functions():
        return False
    ws.__class__ = FastPCM
    ws.reset()
    return True


class FastPCM(pcm.PCM):  # type: ignore[misc]
    """pyscf.solvent.pcm.PCM with K factored and (k|mn) kept per geometry (module docstring)."""

    def build(self, ng: int | None = None) -> None:
        super().build(ng)
        inter = self._intermediates
        inter["K_lu"] = lu_factor(inter["K"])
        mol = self.mol
        nao = mol.nao
        coords, exponents = self.surface["grid_coords"], self.surface["charge_exp"]
        npair = nao * (nao + 1) // 2
        budget_mb = PCM_STORE_FRACTION * (self.max_memory - lib.current_memory()[0])
        if coords.shape[0] * npair * 8 / 1e6 <= budget_mb:
            fakemol = gto.fakemol_for_charges(coords, expnt=exponents**2)
            fakemol.cart = mol.cart
            int3c2e = mol._add_suffix("int3c2e")
            # (npair, npts); the s2ij packing is lib.pack_tril's lower triangle
            inter["B"] = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym="s2ij")
            # sum_mn D_mn (k|mn) = sum_{m>=n} w_mn (D + D^T)_mn (k|mn), for any D
            inter["B_weight"] = lib.pack_tril(1.0 - 0.5 * np.eye(nao))
            logger.info(self, "FastPCM: (k|mn) in core, %.0f MB", inter["B"].nbytes / 1e6)

    def _get_v(self, dms: Array) -> Array:
        inter = self._intermediates or {}
        if "B" not in inter:
            return super()._get_v(dms)  # type: ignore[no-any-return]
        d = lib.pack_tril(dms + dms.transpose(0, 2, 1)) * inter["B_weight"]
        return d @ inter["B"]  # type: ignore[no-any-return]

    def _get_vmat(self, q: Array) -> Array:
        inter = self._intermediates or {}
        if "B" not in inter:
            return super()._get_vmat(q)  # type: ignore[no-any-return]
        q = q.reshape(-1, inter["B"].shape[1])
        return -lib.unpack_tril(q @ inter["B"].T)  # type: ignore[no-any-return]

    def _charges(self, v_grids: Array) -> tuple[Array, Array]:
        """q = K^-1 R v and q_sym = (q + R^T K^-T v) / 2 for each row v of v_grids."""
        lu, R = self._intermediates["K_lu"], self._intermediates["R"]
        q = lu_solve(lu, R @ v_grids.T).T
        qt = (R.T @ lu_solve(lu, v_grids.T, trans=1)).T
        return q, (q + qt) / 2.0

    def _get_vind(self, dms: Array) -> tuple[float, Array]:
        if not self._intermediates:
            self.build()
        nao = dms.shape[-1]
        dms = dms.reshape(-1, nao, nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1, nao, nao)
        v_grids = self.v_grids_n - self._get_v(dms)
        q, q_sym = self._charges(v_grids)
        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * np.dot(q_sym[0], v_grids[0])
        self._intermediates.update(q=q[0], q_sym=q_sym[0], v_grids=v_grids[0], dm=dms)
        return epcm, vmat[0]

    def _B_dot_x(self, dms: Array) -> Array:
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        _, q_sym = self._charges(-self._get_v(dms.reshape(-1, nao, nao)))
        return self._get_vmat(q_sym).reshape(out_shape)

    def grad(self, dm: Array) -> Array:
        from pyscf.solvent.grad.pcm import grad_nuc, grad_solver

        de = self._grad_qv(dm)
        de += grad_nuc(self, dm)
        de += grad_solver(self, dm)
        return de

    def _grad_qv(self, dm: Array) -> Array:
        """grad_qv: (d mu nu|k) from int1e_grids_ip; (mu nu|d k) = -(d mu nu|k) - (mu d nu|k)."""
        if not self._intermediates:
            self.build()
        dm_cache = self._intermediates.get("dm")
        if dm_cache is None or np.linalg.norm(dm_cache - dm) >= 1e-10:
            self._get_vind(dm)
        mol, surf = self.mol, self.surface
        nao = mol.nao
        q_sym = self._intermediates["q_sym"]
        coords, exponents = surf["grid_coords"], surf["charge_exp"]
        dm = np.asarray(dm).reshape(-1, nao, nao).sum(0)
        dsym = (dm + dm.T).ravel()
        blksize = int(
            max((self.max_memory - lib.current_memory()[0]) * 0.9e6 / 8 / nao**2 / 3, 400)
        )
        vq = np.zeros((3, nao * nao))  # sum_k q_k (d mu nu|k)
        dq = np.empty((3, q_sym.size))  # q_k sum_mn D_mn (mu nu|d k)
        unique, group = np.unique(exponents, return_inverse=True)
        for g, xi in enumerate(unique):
            points = np.flatnonzero(group == g)
            with mol.with_rinv_zeta(xi**2):
                for p0, p1 in lib.prange(0, points.size, blksize):
                    idx = points[p0:p1]
                    v = mol.intor("int1e_grids_ip", grids=coords[idx]).reshape(3, idx.size, -1)
                    for x in range(3):
                        vq[x] += q_sym[idx] @ v[x]
                        dq[x, idx] = -(v[x] @ dsym) * q_sym[idx]
        dvj = np.einsum("xij,ij->xi", vq.reshape(3, nao, nao), dm)
        dq_atom = np.array([dq[:, p0:p1].sum(1) for p0, p1 in surf["gslice_by_atom"]])
        dvj_atom = np.array([dvj[:, p0:p1].sum(1) for p0, p1 in mol.aoslice_by_atom()[:, 2:]])
        return dq_atom + 2.0 * dvj_atom
