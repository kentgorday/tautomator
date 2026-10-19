"""Tests for reading the nonelectrostatic inputs off a PySCF PCM calculation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tautomator import nonelectrostatic
from tautomator.config import solvent_electron_density
from tautomator.nonelectrostatic import Array, Solute
from tautomator.pyscf_nonelectrostatic import (
    electric_field,
    single_point_with_descriptors,
    solute_from_mf,
)
from tautomator.pyscf_runner import _build_mf, single_point
from tautomator.types import Geometry

FORMATE = Geometry(
    symbols=("C", "O", "O", "H"),
    coords=np.array([[0.0, 0.0, 0.52], [1.13, 0.0, -0.05], [-1.13, 0.0, -0.05], [0.0, 0.0, 1.66]]),
)


@pytest.fixture(scope="module")
def formate() -> tuple[Any, Solute]:
    """HF/6-31G* formate anion in IEF-PCM water, built as the pipeline builds it."""
    _mol, mf = _build_mf(FORMATE, -1, "hf", "6-31g*", "IEFPCM", "water")
    mf.verbose = 0
    mf.kernel()
    return mf, solute_from_mf(mf)


WATER = solvent_electron_density("water")


def _potential(mol: Any, dm: Array, points: Array) -> Array:
    """Electrostatic potential of the solute, by a route independent of electric_field."""
    dist = np.linalg.norm(points[:, None, :] - mol.atom_coords()[None, :, :], axis=2)
    nuclear = (mol.atom_charges()[None, :] / dist).sum(axis=1)
    electronic = np.einsum("pij,ij->p", mol.intor("int1e_grids", grids=points), dm)
    return np.asarray(nuclear - electronic)


class TestElectricField:
    def test_is_minus_the_gradient_of_the_potential(self, formate: tuple[Any, Solute]) -> None:
        mf, _solute = formate
        mol, dm = mf.mol, mf.make_rdm1()
        points = np.array([[0.3, -2.1, 3.4], [-4.0, 1.0, 0.5], [2.5, 2.5, -2.5]])
        h = 1e-3
        numeric = np.empty((3, 3))
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            numeric[:, axis] = -(
                _potential(mol, dm, points + step) - _potential(mol, dm, points - step)
            ) / (2 * h)
        assert electric_field(mol, dm, points) == pytest.approx(numeric, abs=1e-7)


class TestSoluteFromMf:
    def test_density_integrates_to_the_electron_count(self, formate: tuple[Any, Solute]) -> None:
        """The inside and outside grids together cover all of space.

        On grids finer than the defaults: 12 radial nodes inside do not resolve the
        bare density's nuclear cusps, which the dispersion integrand does not have.
        """
        mf, _solute = formate
        solute = solute_from_mf(mf, inside_grid=(41, 40), outside_grid=(41, 40))
        inside = solute.interior.weights @ solute.interior.rho
        # The two regions' rules do not share the surface, so their errors do
        # not cancel: about 5e-4 electrons here.
        assert inside + nonelectrostatic.electrons_outside(solute.exterior) == pytest.approx(
            mf.mol.nelectron, abs=1.5e-3
        )

    def test_surface_elements_lie_on_their_spheres(self, formate: tuple[Any, Solute]) -> None:
        _mf, solute = formate
        s = solute.surface
        dist = np.linalg.norm(s.coords[:, None, :] - s.atom_coords[None, :, :], axis=2)
        assert np.abs(dist - s.atom_radii[None, :]).min(axis=1) == pytest.approx(0.0, abs=1e-10)

    def test_each_element_is_assigned_its_own_sphere(self, formate: tuple[Any, Solute]) -> None:
        _mf, solute = formate
        s = solute.surface
        own = np.linalg.norm(s.coords - s.atom_coords[s.element_atoms], axis=1)
        assert own == pytest.approx(s.atom_radii[s.element_atoms], abs=1e-10)
        assert set(s.element_atoms) == set(range(len(s.atom_radii)))

    def test_charges_are_those_of_the_converged_density(self, formate: tuple[Any, Solute]) -> None:
        """The charges are recomputed from the final density, not taken from whatever
        density the SCF evaluated last."""
        mf, solute = formate
        ws = mf.with_solvent
        ws._intermediates["q"] = np.zeros_like(ws._intermediates["q"])
        assert solute_from_mf(mf).surface.charges == pytest.approx(
            solute.surface.charges, rel=1e-10, abs=1e-14
        )

    def test_gauss_law_ties_field_normals_areas_and_density_together(
        self, formate: tuple[Any, Solute]
    ) -> None:
        """The flux of the solute's field through the cavity is 4 pi times the charge inside.

        The charge inside is the molecular charge plus the electrons that lie
        outside, so this checks the sign of the field, the outward normals, the
        surface areas and the inside/outside split of the density at once.  The
        tolerance is the accuracy of the switching-function areas as a quadrature.
        """
        mf, solute = formate
        flux = solute.surface.areas @ solute.normal_field
        enclosed = mf.mol.charge + nonelectrostatic.electrons_outside(solute.exterior)
        assert flux == pytest.approx(4.0 * np.pi * enclosed, rel=2e-2)

    def test_descriptors_have_the_expected_signs(self, formate: tuple[Any, Solute]) -> None:
        _mf, solute = formate
        d = nonelectrostatic.descriptors(solute, WATER)
        assert d.dispersion > 0.0  # the attractive sign comes from A < 0
        assert d.exchange > 0.0
        assert 0.0 < d.electrons_outside < 0.5
        f_min, _f_max = d.field_extrema()
        assert f_min < -0.03  # an anion's oxygens

    @pytest.mark.slow
    def test_quadrature_is_converged(self, formate: tuple[Any, Solute]) -> None:
        mf, solute = formate
        d = nonelectrostatic.descriptors(solute, WATER)
        fine = nonelectrostatic.descriptors(
            solute_from_mf(mf, inside_grid=(59, 40), outside_grid=(59, 40)), WATER
        )
        # The default grids aim at ~0.01-0.03 kcal/mol on terms of a few kcal/mol: a few 1e-3.
        assert d.dispersion == pytest.approx(fine.dispersion, rel=3e-3)
        assert d.exchange == pytest.approx(fine.exchange, rel=5e-3)

    def test_refuses_a_mean_field_without_pcm_charges(self) -> None:
        geom = Geometry(symbols=("He",), coords=np.zeros((1, 3)))
        _mol, mf = _build_mf(geom, 0, "hf", "sto-3g", "IEFPCM", "water")
        mf.converged = True  # never run: the solvent object holds no charges
        with pytest.raises(ValueError, match="no converged surface charges"):
            solute_from_mf(mf)

    def test_cosmo_charges_give_the_same_field_as_ief_pcm(self) -> None:
        """COSMO scales its charges by (eps-1)/(eps+1/2), not (eps-1)/eps; read with the
        wrong factor its charge field comes out short by eps/(eps+1/2), 0.80 in
        hexadecane.  On a single sphere the two models must agree."""
        fluoride = Geometry(symbols=("F",), coords=np.zeros((1, 3)))
        mean_field = {}
        for model in ("IEFPCM", "COSMO"):
            _mol, mf = _build_mf(fluoride, -1, "hf", "6-31+g*", model, "hexadecane")
            mf.verbose = 0
            mf.kernel()
            surface = solute_from_mf(mf).surface
            field = nonelectrostatic.charge_field(surface)
            mean_field[model] = surface.areas @ field / surface.areas.sum()
        assert mean_field["COSMO"] == pytest.approx(mean_field["IEFPCM"], rel=2e-3)


class TestSinglePointWithDescriptors:
    def test_energy_is_that_of_the_plain_single_point(self) -> None:
        args = (FORMATE, -1, "hf", "6-31g*", "IEFPCM", "water")
        energy, d = single_point_with_descriptors(*args)
        assert energy == pytest.approx(single_point(*args), abs=1e-9)
        assert d.dispersion > 0.0
        assert len(d.areas) == len(d.normal_field) == len(d.element_atoms)

    @pytest.mark.parametrize("model", ["SMD", "ddCOSMO"])
    def test_refuses_other_solvent_models(self, model: str) -> None:
        with pytest.raises(ValueError, match="analytical PCM"):
            single_point_with_descriptors(FORMATE, -1, "hf", "sto-3g", model, "water")
