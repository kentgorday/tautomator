"""The nonelectrostatic correction inside the SCF: energy, convergence and gradient."""

from __future__ import annotations

from typing import Any

import pytest

from tautomator.config import solvent_electron_density
from tautomator.pyscf_nonelectrostatic import NonelectrostaticTerms
from tautomator.pyscf_nonelectrostatic_scf import Settings, enable_self_consistent
from tautomator.pyscf_runner import _build_mf
from tautomator.types import Geometry
from tests.formate_case import FORMATE, PARAMS, displaced

SETTINGS = Settings(PARAMS, solvent_electron_density("water"))


def _scf(
    geom: Geometry, self_consistent: bool = True, method: str = "hf", basis: str = "6-31g*"
) -> tuple[Any, Any]:
    mol, mf = _build_mf(geom, -1, method, basis, "ISWIG", "water")
    mf.verbose = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
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
                _scf(displaced(FORMATE, atom, axis, sign * h))[1].e_tot for sign in (1.0, -1.0)
            ]
            assert analytic[atom, axis] == pytest.approx(
                (values[0] - values[1]) / (2 * h), abs=3e-7
            )

    @pytest.mark.slow
    def test_dft_gradient_matches_finite_differences(self) -> None:
        """PBE, with the grid's response.  Its quadrature leaves an h^2 error in the
        differences (1.4e-6 at h = 1e-4, 2e-7 at 4e-5), which two steps extrapolate away."""
        sc = _scf(FORMATE, method="pbe", basis="def2-svp")[1]
        grad = sc.nuc_grad_method()
        grad.verbose = 0
        grad.grid_response = True
        analytic = grad.kernel()
        steps = (4e-5, 1e-4)
        for atom, axis in [(0, 2), (3, 1)]:
            fd = []
            for h in steps:
                e = [
                    _scf(displaced(FORMATE, atom, axis, sign * h), method="pbe", basis="def2-svp")[
                        1
                    ].e_tot
                    for sign in (1.0, -1.0)
                ]
                fd.append((e[0] - e[1]) / (2 * h))
            extrapolated = fd[0] + (fd[0] - fd[1]) * steps[0] ** 2 / (
                steps[1] ** 2 - steps[0] ** 2
            )
            assert analytic[atom, axis] == pytest.approx(extrapolated, abs=3e-7)

    def test_follows_the_geometry_in_a_scanner(self, both: tuple[Any, Any]) -> None:
        """A gradient scanner moves the molecule; the correction is rebuilt for it."""
        _plain, sc = both
        scanner = sc.nuc_grad_method().as_scanner()
        moved = displaced(FORMATE, 1, 0, 0.05)
        e_scan, _grad = scanner(sc.mol.set_geom_(moved.coords, unit="Angstrom", inplace=False))
        assert e_scan == pytest.approx(_scf(moved)[1].e_tot, abs=1e-8)

    def test_refuses_a_mean_field_without_pcm(self) -> None:
        _mol, mf = _build_mf(FORMATE, -1, "hf", "sto-3g")
        with pytest.raises(ValueError, match="no PCM"):
            enable_self_consistent(mf, SETTINGS)
