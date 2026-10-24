"""The nonelectrostatic correction on MOIST's DROP cavities (pyscf_nonelectrostatic_drop).

Formate at HF/def2-SVP on SvdW-DROP and rho-DROP, against central differences:

- the surface contractions packages/moist's patch exports, on a functional of the surface;
- the terms at a fixed density matrix: their potential (Richardson-extrapolated
  differences in the density) and their gradient (the cavity rebuilt at each geometry);
- the self-consistent correction's total gradient, against energy differences;
- the post-SCF correction's gradient, against differences of G on each geometry's density.

MOIST's PCM solves its charges iteratively to 1e-8 by default, which leaves noise of that
order in the SCF density; G is not variational, so its differences would carry it divided by
the step.  The SCFs here use MOIST's dense solver.  Runs only where MOIST is installed (the
default pixi environment).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

if importlib.util.find_spec("moist") is None:
    pytest.skip("MOIST is not installed in this environment", allow_module_level=True)

import moist
from pyscf import gto, scf

from tautomator import pyscf_moist
from tautomator.nonelectrostatic import Params
from tautomator.pyscf_nonelectrostatic_drop import (
    DropSettings,
    DropTerms,
    drop_descriptors,
    enable_self_consistent,
    post_scf_gradient,
)

ATOM = "C 0 0 0; O 1.25 0.1 0; O -0.7 1.05 0.1; H -0.6 -0.95 0.2"
CAVITIES = ("SvdW-DROP", "rho-DROP")
# Larger than any fit will give (c- and c+ put tens of kcal/mol on formate), so that every
# term's derivatives show
SETTINGS = DropSettings(
    params=Params(
        dispersion=-0.0036,
        exchange=0.0105,
        neg_coeff=-0.5,
        neg_exponent=2.5,
        pos_coeff=-0.3,
        pos_exponent=2.5,
    ),
    solvent_rho=0.0494,
)
# (atom, Cartesian direction) components checked against differences
COMPONENTS = ((0, 0), (1, 1), (2, 2), (3, 0))


def _mol(coords: np.ndarray | None = None) -> Any:
    mol = gto.M(atom=ATOM, basis="def2-svp", charge=-1, verbose=0)
    if coords is not None:
        mol.set_geom_(coords, unit="Bohr")
    return mol


def _moist_mf(mol: Any, cavity: str) -> Any:
    component = moist.ModelComponentIEFPCM(78.3553, parameters=moist.PCMParameters(solver="DENSE"))
    return pyscf_moist.attach(
        scf.RHF(mol),
        cavity=pyscf_moist.drop_cavity(cavity),
        components=[component],
        context=moist.Context(nthreads=2, verbosity=0),
    )


def _terms(mol: Any, cavity: str) -> DropTerms:
    return DropTerms(mol, _moist_mf(mol, cavity).with_moist._driver(), SETTINGS)


def _scf(mf: Any, dm0: np.ndarray | None = None) -> Any:
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 3e-11, 300
    mf.kernel(dm0=dm0)
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def density() -> np.ndarray:
    mf = scf.RHF(_mol())
    mf.kernel()
    return np.asarray(mf.make_rdm1())


def _difference(f: Any, coords: np.ndarray, atom: int, k: int, h: float) -> float:
    plus, minus = coords.copy(), coords.copy()
    plus[atom, k] += h
    minus[atom, k] -= h
    return float((f(plus) - f(minus)) / (2.0 * h))


@pytest.mark.parametrize("cavity", CAVITIES)
def test_exported_contractions(cavity: str, density: np.ndarray) -> None:
    """A functional of the surface (area, normals, positions) through MOIST's reverse pass,
    plus on rho-DROP the density route, against the cavity rebuilt at displaced nuclei."""
    u, q = np.array([0.3, -0.5, 0.8]), np.array([0.2, 0.1, -0.3])

    def snapshot(mol: Any) -> Any:
        driver = _moist_mf(mol, cavity).with_moist._driver()
        driver.evaluate(density)
        return driver

    def functional(s: Any) -> float:
        d = s.xyz - q
        return float(np.sum(s.a * (1.0 + 0.3 * (s.normal0 @ u) + 0.01 * np.sum(d * d, axis=1))))

    driver = snapshot(_mol())
    cav = driver.model.cavity
    s = cav.snapshot()
    d = s.xyz - q
    weights = {
        "w_a": 1.0 + 0.3 * (s.normal0 @ u) + 0.01 * np.sum(d * d, axis=1),
        "w_n": 0.3 * s.a[:, None] * u[None],
        "w_xyz": 0.02 * s.a[:, None] * d,
    }
    g = cav.contract_surface_gradient(**weights)
    density_weights = cav.contract_surface_density(**weights)
    assert (density_weights is not None) == (cavity == "rho-DROP")
    if density_weights is not None:
        driver.host.dm = density
        g = g + driver.host._gradient_lsf(s.xyz, density_weights)
    coords = _mol().atom_coords()

    def area_functional(c: np.ndarray) -> float:
        return functional(snapshot(_mol(c)).model.cavity.snapshot())

    for atom, k in COMPONENTS:
        # Richardson: steps of 1e-4 alone leave ~1e-6 of curvature on these surfaces
        fd = (
            4.0 * _difference(area_functional, coords, atom, k, 1e-4)
            - _difference(area_functional, coords, atom, k, 2e-4)
        ) / 3.0
        assert g[atom, k] == pytest.approx(fd, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("cavity", CAVITIES)
def test_potential_at_fixed_surface_and_density(cavity: str, density: np.ndarray) -> None:
    mol = _mol()
    _energies, V = _terms(mol, cavity).energy_and_potential(density)
    rng = np.random.default_rng(0)
    dD = rng.normal(size=density.shape)
    dD = 1e-3 * (dD + dD.T)

    def g(t: float) -> float:
        return _terms(mol, cavity).energy(density + t * dD).total

    r1, r2 = (g(5e-3) - g(-5e-3)) / 1e-2, (g(1e-2) - g(-1e-2)) / 2e-2
    assert np.sum(V * dD) == pytest.approx((4.0 * r1 - r2) / 3.0, rel=1e-7)


@pytest.mark.parametrize("cavity", CAVITIES)
def test_gradient_at_fixed_density(cavity: str, density: np.ndarray) -> None:
    g = _terms(_mol(), cavity).gradient(density)
    coords = _mol().atom_coords()
    for atom, k in COMPONENTS:
        fd = _difference(
            lambda c: _terms(_mol(c), cavity).energy(density).total, coords, atom, k, 1e-4
        )
        assert g[atom, k] == pytest.approx(fd, abs=2e-7)


# rho-DROP's takes 8 minutes on an idle machine (its surface moves with the density)
@pytest.mark.parametrize("cavity", ["SvdW-DROP", pytest.param("rho-DROP", marks=pytest.mark.slow)])
def test_self_consistent_gradient(cavity: str) -> None:
    def run(coords: np.ndarray | None = None, dm0: np.ndarray | None = None) -> Any:
        mf = _moist_mf(_mol(coords), cavity)
        enable_self_consistent(mf, SETTINGS)
        return _scf(mf, dm0)

    mf = run()
    assert mf.with_moist.e_nonelectrostatic is not None
    g = mf.nuc_grad_method().kernel()
    coords, dm0 = mf.mol.atom_coords(), mf.make_rdm1()
    for atom, k in COMPONENTS:
        fd = _difference(lambda c: run(c, dm0).e_tot, coords, atom, k, 4e-5)
        assert g[atom, k] == pytest.approx(fd, abs=1e-7)
    # a copy (as geomeTRIC and scanners make) keeps the correction
    assert mf.copy().with_moist.nonelectrostatic == SETTINGS


@pytest.mark.parametrize("cavity", CAVITIES)
def test_post_scf_gradient(cavity: str) -> None:
    def run(
        coords: np.ndarray | None = None, dm0: np.ndarray | None = None
    ) -> tuple[Any, DropTerms]:
        mol = _mol(coords)
        mf = _scf(_moist_mf(mol, cavity), dm0)
        return mf, DropTerms(mol, mf.with_moist._driver(), SETTINGS)

    mf, terms = run()
    g = post_scf_gradient(mf, terms)
    coords, dm0 = mf.mol.atom_coords(), mf.make_rdm1()

    def G(c: np.ndarray) -> float:
        m, t = run(c, dm0)
        return t.energy(m.make_rdm1()).total

    for atom, k in COMPONENTS:
        assert g[atom, k] == pytest.approx(_difference(G, coords, atom, k, 4e-5), abs=2e-7)


@pytest.mark.parametrize("cavity", CAVITIES)
def test_descriptors_reproduce_the_terms(cavity: str) -> None:
    """The saved integrals give the same energy as the terms, for any parameters; the PCM's
    charges recast as a field follow the solute's normal field."""
    mol = _mol()
    mf = _scf(_moist_mf(mol, cavity))
    d = drop_descriptors(mf, SETTINGS.solvent_rho)
    terms = DropTerms(mol, mf.with_moist._driver(), SETTINGS)
    assert d.energy(SETTINGS.params).total == pytest.approx(
        terms.energy(mf.make_rdm1()).total, rel=1e-12
    )
    assert d.element_atoms.min() == 0 and d.element_atoms.max() == mol.natm - 1
    cov = np.cov(d.normal_field, d.charge_field, aweights=d.areas)
    assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) > 0.9
    # built from IEF-PCM's apparent charges q, not the potential adjoint w_phi
    driver = mf.with_moist._driver()
    q = np.asarray(driver.model.components[0].get("q"))
    w_phi = next(
        np.asarray(item.w_phi)
        for item in driver.response
        if isinstance(item, moist.PotentialAdjointResponse)
    )
    scale = (78.3553 - 1.0) / 78.3553
    np.testing.assert_allclose(d.charge_field, -4.0 * np.pi / scale * q / d.areas, rtol=1e-12)
    assert np.abs(q - w_phi).max() > 1e-5 * np.abs(q).max()
