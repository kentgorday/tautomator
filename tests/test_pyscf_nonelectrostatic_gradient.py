"""The nonelectrostatic correction's potential and gradients, against finite differences."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tautomator import nonelectrostatic
from tautomator.pyscf_nonelectrostatic import NonelectrostaticTerms, solute_from_mf
from tautomator.pyscf_nonelectrostatic_gradient import post_scf_gradient
from tautomator.pyscf_runner import _build_mf
from tautomator.types import Geometry
from tests.formate_case import FORMATE, PARAMS, WATER, displaced


def _mean_field(geom: Geometry, method: str = "hf", basis: str = "6-31g*") -> tuple[Any, Any]:
    mol, mf = _build_mf(geom, -1, method, basis, "ISWIG", "water")
    mf.verbose = 0
    mf.conv_tol = 1e-13
    # G is not variational in the density, so its finite difference carries the
    # density's residual at first order, divided by the step: at 1e-9 that noise
    # reached the 5e-7 tolerance (HF and wB97X-3c, on linux-64 MKL).  PBE on its
    # grid cannot converge to 1e-11, and on osx-arm64 takes ~100 cycles to 1e-10.
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mol, mf


@pytest.fixture(scope="module")
def formate() -> tuple[Any, Any, NonelectrostaticTerms]:
    mol, mf = _mean_field(FORMATE)
    return mol, mf, NonelectrostaticTerms(mol, mf.with_solvent, PARAMS, WATER)


class TestNonelectrostaticTerms:
    def test_energy_is_the_descriptors(
        self, formate: tuple[Any, Any, NonelectrostaticTerms]
    ) -> None:
        _mol, mf, terms = formate
        energies = terms.energy(mf.make_rdm1())
        expected = nonelectrostatic.descriptors(solute_from_mf(mf), WATER).energy(PARAMS)
        assert energies.dispersion == pytest.approx(expected.dispersion, rel=1e-12)
        assert energies.exchange == pytest.approx(expected.exchange, rel=1e-12)
        assert energies.short_range == pytest.approx(expected.short_range, rel=1e-12)

    def test_potential_is_the_derivative_in_the_density(
        self, formate: tuple[Any, Any, NonelectrostaticTerms]
    ) -> None:
        _mol, mf, terms = formate
        dm = mf.make_rdm1()
        _energies, V = terms.energy_and_potential(dm)
        assert V is not None
        assert pytest.approx(V.T, abs=1e-14) == V
        rng = np.random.default_rng(1)
        P = rng.normal(size=dm.shape)
        P = 0.01 * (P + P.T)
        h = 1e-4
        numeric = (terms.energy(dm + h * P).total - terms.energy(dm - h * P).total) / (2 * h)
        assert np.sum(V * P) == pytest.approx(numeric, rel=1e-8)

    def test_explicit_gradient_matches_finite_differences(
        self, formate: tuple[Any, Any, NonelectrostaticTerms]
    ) -> None:
        """At fixed AO density matrix: basis functions, grids, surface and nuclei move."""
        _mol, mf, terms = formate
        dm = mf.make_rdm1()
        analytic = terms.gradient(dm)
        assert analytic.sum(axis=0) == pytest.approx(0.0, abs=1e-10)
        # The quadrature is C^2: its third derivative jumps where a node crosses another
        # sphere's surface, so central differences need small steps to converge.
        h = 1e-5
        for atom, axis in [(0, 2), (1, 0), (3, 1)]:
            values = []
            for sign in (1.0, -1.0):
                moved, mf_moved = _build_mf(
                    displaced(FORMATE, atom, axis, sign * h),
                    -1,
                    "hf",
                    "6-31g*",
                    "ISWIG",
                    "water",
                )
                values.append(
                    NonelectrostaticTerms(moved, mf_moved.with_solvent, PARAMS, WATER)
                    .energy(dm)
                    .total
                )
            numeric = (values[0] - values[1]) / (2 * h)
            assert analytic[atom, axis] == pytest.approx(numeric, abs=5e-8)


def _check_post_scf(method: str, basis: str, coordinates: list[tuple[int, int]]) -> None:
    mol, mf = _mean_field(FORMATE, method, basis)
    terms = NonelectrostaticTerms(mol, mf.with_solvent, PARAMS, WATER)
    analytic = post_scf_gradient(mf, terms)
    assert analytic.sum(axis=0) == pytest.approx(0.0, abs=1e-8)
    # The quadrature is C^2: its third derivative jumps where a node crosses another
    # sphere's surface, so central differences need small steps to converge.
    h = 2e-5
    for atom, axis in coordinates:
        values = []
        for sign in (1.0, -1.0):
            moved, mf_moved = _mean_field(displaced(FORMATE, atom, axis, sign * h), method, basis)
            dm = mf_moved.make_rdm1()
            values.append(
                NonelectrostaticTerms(moved, mf_moved.with_solvent, PARAMS, WATER).energy(dm).total
            )
        numeric = (values[0] - values[1]) / (2 * h)
        assert analytic[atom, axis] == pytest.approx(numeric, abs=5e-7)


class TestPostScfGradient:
    """G evaluated on the SCF density at each geometry: the density's response matters."""

    def test_hartree_fock(self) -> None:
        _check_post_scf("hf", "6-31g*", [(0, 2), (2, 0)])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("method", "basis"), [("pbe", "def2-svp"), ("wb97x", "def2-svp"), ("wB97X-3c", "vDZP")]
    )
    def test_dft(self, method: str, basis: str) -> None:
        """GGA, range-separated hybrid, and the refinement level with its ECPs.

        On the SCF's own (99, 590) grid: the XC kernel term includes the grid's
        response, so the gradient is the derivative of the energy on that grid.
        """
        _check_post_scf(method, basis, [(0, 2), (3, 1)])

    def test_response_is_not_negligible(
        self, formate: tuple[Any, Any, NonelectrostaticTerms]
    ) -> None:
        _mol, mf, terms = formate
        explicit = terms.gradient(mf.make_rdm1())
        total = post_scf_gradient(mf, terms)
        assert np.abs(total - explicit).max() > 0.5 * np.abs(total).max()

    # wB97M-V is a meta-GGA, refused before its VV10 is looked at; wB97X-V is a GGA.
    @pytest.mark.parametrize(
        ("xc", "reason"), [("tpss", "MGGA"), ("wb97m-v", "MGGA"), ("wb97x-v", "VV10")]
    )
    def test_refuses_meta_gga_and_vv10(self, xc: str, reason: str) -> None:
        mol, mf = _build_mf(FORMATE, -1, "pbe", "sto-3g", "ISWIG", "water")
        mf.verbose = 0
        mf.xc = xc
        mf.converged = True
        terms = NonelectrostaticTerms(mol, mf.with_solvent, PARAMS, WATER)
        with pytest.raises(NotImplementedError, match=reason):
            post_scf_gradient(mf, terms)
