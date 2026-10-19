"""The nonelectrostatic correction included self-consistently in a PySCF PCM SCF.

`enable_self_consistent(mf, settings)` makes the correction part of the solvent
model of a PCM mean field `pyscf_runner._build_mf` built: every SCF iteration
adds G(D) to the energy and V = dG/dD to the Fock matrix, beside the PCM's own
reaction field, so the density relaxes in the presence of the correction.  The
energy is then variational in the density, and its nuclear gradient is PySCF's
usual one (whose energy-weighted density comes from the Fock matrix that holds
V) plus the correction's explicit gradient at fixed density matrix, which the
solvent object's ``grad`` adds.  `mf.nuc_grad_method()`, geomeTRIC
optimizations and scanners therefore work unchanged.

The post-SCF model, in which the SCF does not see the correction, is
`pyscf_nonelectrostatic_gradient`; self-consistent minus post-SCF energies
measure what the relaxation is worth.

What PySCF's response functions see is unchanged: CPHF, stability analysis and
the second-order (Newton) SCF use the PCM's response without the correction's
kernel.  Newton then works with an approximate orbital Hessian; it still
converges to the right density, as the Fock matrix and energy are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pyscf import lib

from tautomator.cavity_quadrature import INSIDE_GRID, OUTSIDE_GRID, Grid
from tautomator.nonelectrostatic import DELTA, VV_C, Energies, Params
from tautomator.pyscf_nonelectrostatic import NonelectrostaticTerms

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Settings:
    """The correction's parameters and the solvent's electron density (e/bohr^3)."""

    params: Params
    solvent_rho: float
    delta: float = DELTA
    vv_c: float = VV_C
    inside_grid: Grid = INSIDE_GRID
    outside_grid: Grid = OUTSIDE_GRID


def _total(dm: Array) -> Array:
    dm = np.asarray(dm)
    return dm[0] + dm[1] if dm.ndim == 3 else dm


class _SelfConsistentNonelectrostatic:
    """Mixin for a PySCF PCM solvent object; see the module docstring."""

    _keys: ClassVar[set[str]] = {"nonelectrostatic", "e_nonelectrostatic"}

    nonelectrostatic: Settings
    e_nonelectrostatic: Energies | None
    _nonelectrostatic_terms: NonelectrostaticTerms | None
    e: float
    v: Array

    def _terms(self) -> NonelectrostaticTerms:
        mol = self.mol  # type: ignore[attr-defined]
        coords = mol.atom_coords()
        cached = getattr(self, "_nonelectrostatic_terms", None)
        if cached is None or not np.array_equal(cached.centres, coords):
            s = self.nonelectrostatic
            cached = NonelectrostaticTerms(
                mol,
                self,
                s.params,
                s.solvent_rho,
                s.delta,
                s.vv_c,
                s.inside_grid,
                s.outside_grid,
                store_field_integrals=True,
            )
            self._nonelectrostatic_terms = cached
        return cached

    def kernel(self, dm: Array) -> tuple[float, Array]:
        e_pcm, v_pcm = super().kernel(dm)  # type: ignore[misc]
        energies, V = self._terms().energy_and_potential(_total(dm))
        assert V is not None
        self.e_nonelectrostatic = energies
        self.e = e_pcm + energies.total
        self.v = v_pcm + V
        return self.e, self.v

    def grad(self, dm: Array) -> Array:
        de = super().grad(dm)  # type: ignore[misc]
        return np.asarray(de + self._terms().gradient(_total(dm)))

    def reset(self, mol: Any = None) -> Any:
        self._nonelectrostatic_terms = None
        return super().reset(mol)  # type: ignore[misc]


def enable_self_consistent(mf: Any, settings: Settings) -> Any:
    """Include the correction in the SCF of a PCM mean field; returns ``mf``.

    The SCF must be rerun (mf.kernel()) for the energy to include it.  Starting
    from a converged PCM density saves most of the iterations with it.
    """
    ws = getattr(mf, "with_solvent", None)
    if ws is None or not hasattr(ws, "_get_vind") or not hasattr(ws, "surface"):
        raise ValueError("the mean field has no PCM solvent object")
    if not isinstance(ws, _SelfConsistentNonelectrostatic):
        lib.set_class(ws, (_SelfConsistentNonelectrostatic, type(ws)))
    ws.nonelectrostatic = settings
    ws.e_nonelectrostatic = None
    ws._nonelectrostatic_terms = None
    return mf
