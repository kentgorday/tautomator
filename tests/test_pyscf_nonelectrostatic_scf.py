"""The nonelectrostatic correction inside the SCF: energy, convergence and gradient."""

from __future__ import annotations

from typing import Any

import pytest

from tautomator.config import solvent_electron_density
from tautomator.pyscf_nonelectrostatic import NonelectrostaticTerms
from tautomator.pyscf_nonelectrostatic_scf import Settings, enable_self_consistent
from tautomator.pyscf_runner import _build_mf
from tautomator.types import Geometry
from tests.test_pyscf_nonelectrostatic_gradient import FORMATE, PARAMS, _displaced

SETTINGS = Settings(PARAMS, solvent_electron_density("water"))


def _scf(geom: Geometry, self_consistent: bool = True) -> tuple[Any, Any]:
    mol, mf = _build_mf(geom, -1, "hf", "6-31g*", "IEFPCM", "water")
    mf.verbose = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    if self_consistent:
        dm0 = mf.make_rdm1()
        enable_self_consistent(mf, SETTINGS)
        mf.kernel(dm0=dm0)
    assert mf.converged
    return mol, mf


@pytest.fixture(scope="module")
def both() -> tuple[Any, Any]:
    return _scf(FORMATE, self_consistent=False)[1], _scf(FORMATE)[1]


class TestSelfConsistent:
    def test_energy_is_below_the_post_scf_one(self, both: tuple[Any, Any]) -> None:
        """Variational: the density relaxed with G in the Fock matrix lowers E + G."""
        plain, sc = both
        terms = NonelectrostaticTerms(plain.mol, plain.with_solvent, PARAMS, SETTINGS.solvent_rho)
        post = plain.e_tot + terms.energy(plain.make_rdm1()).total
        assert sc.e_tot < post

    def test_energy_includes_the_correction_at_its_own_density(
        self, both: tuple[Any, Any]
    ) -> None:
        _plain, sc = both
        terms = NonelectrostaticTerms(sc.mol, sc.with_solvent, PARAMS, SETTINGS.solvent_rho)
        assert sc.with_solvent.e_nonelectrostatic.total == pytest.approx(
            terms.energy(sc.make_rdm1()).total, rel=1e-8
        )

    def test_gradient_matches_finite_differences(self, both: tuple[Any, Any]) -> None:
        _plain, sc = both
        grad = sc.nuc_grad_method()
        grad.verbose = 0
        analytic = grad.kernel()
        assert analytic.sum(axis=0) == pytest.approx(0.0, abs=1e-10)
        # Total energies (~190 Eh) carry ~1e-11 of rounding, which a step this small
        # turns into ~1e-7; larger steps meet the jumps in the quadrature's third derivative.
        h = 4e-5
        for atom, axis in [(0, 2), (3, 1)]:
            values = [
                _scf(_displaced(FORMATE, atom, axis, sign * h))[1].e_tot for sign in (1.0, -1.0)
            ]
            assert analytic[atom, axis] == pytest.approx(
                (values[0] - values[1]) / (2 * h), abs=3e-7
            )

    def test_follows_the_geometry_in_a_scanner(self, both: tuple[Any, Any]) -> None:
        """A gradient scanner moves the molecule; the correction is rebuilt for it."""
        _plain, sc = both
        scanner = sc.nuc_grad_method().as_scanner()
        moved = _displaced(FORMATE, 1, 0, 0.05)
        e_scan, _grad = scanner(sc.mol.set_geom_(moved.coords, unit="Angstrom", inplace=False))
        assert e_scan == pytest.approx(_scf(moved)[1].e_tot, abs=1e-8)

    def test_refuses_a_mean_field_without_pcm(self) -> None:
        _mol, mf = _build_mf(FORMATE, -1, "hf", "sto-3g")
        with pytest.raises(ValueError, match="no PCM"):
            enable_self_consistent(mf, SETTINGS)
