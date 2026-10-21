"""tautomator.pyscf_moist's MoistHost against MOIST's own PySCF host.

Each host routine MoistHost replaces is compared with MOIST's at the same input, then whole
SCF and gradient runs with MOIST's PCM models.  Molecules with ECPs, which MOIST's own host
refuses, are checked against PySCF's PCM on MOIST's surface and against finite differences.
Runs only where MOIST is installed (the moist pixi environment); to check another MOIST
release, run it there with that release and see the module docstring for CHECKED.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

moist = pytest.importorskip("moist")  # the moist pixi environment only

import moist.pyscf as mp  # noqa: E402
from moist import (  # noqa: E402
    Context,
    IsodensityParameters,
    ModelComponentCOSMO,
    ModelComponentCPCM,
    ModelComponentIEFPCM,
)
from moist.pyscf import DROP, Isodensity  # noqa: E402
from pyscf import dft, gto, scf  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402

from tautomator import pyscf_moist, pyscf_runner  # noqa: E402
from tautomator.pyscf_moist import (  # noqa: E402
    CHECKED,
    REPLACED,
    MoistHost,
    MoistSolvation,
    attach,
    installed_fingerprints,
    rho_drop,
    svdw_drop,
)
from tautomator.types import Geometry  # noqa: E402

EPS = 78.3553
CONTEXT = Context(nthreads=2, verbosity=0)

# name: (atoms in Angstrom, charge, spin, basis, cart)
MOLECULES: dict[str, tuple[str, int, int, str, bool]] = {
    "water": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 0, "6-31g*", False),
    "water_cart": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 0, "6-31g*", True),
    "formate": (
        "C 0 0 0.52; O 1.13 0 -0.05; O -1.13 0 -0.05; H 0 0 1.66",
        -1,
        0,
        "def2-svp",
        False,
    ),
    "methyl": (
        "C 0 0 0; H 1.079 0 0; H -0.5395 0.9344 0; H -0.5395 -0.9344 0",
        0,
        1,
        "def2-svp",
        False,
    ),  # UHF
}
# def2-SVP's ECP on iodine (PySCF adds it only when asked)
ECP_MOLECULE = ("H 0 0 0; I 0 0 1.609", 0, 0, "def2-svp", False)
ECP = {"I": "def2-svp"}


def _mol(spec: tuple[str, int, int, str, bool], **kwargs: Any) -> Any:
    atoms, charge, spin, basis, cart = spec
    if spec is ECP_MOLECULE:
        kwargs.setdefault("ecp", ECP)
    return gto.M(atom=atoms, charge=charge, spin=spin, basis=basis, cart=cart, verbose=0, **kwargs)


def _density(mol: Any) -> Any:
    """A gas-phase SCF density (the comparisons need not be self-consistent)."""
    mf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    mf.kernel()
    return mf.make_rdm1()


def _components(model: str) -> list[Any]:
    return [ModelComponentCPCM(EPS) if model == "C-PCM" else ModelComponentIEFPCM(EPS)]


def _same(a: Any, b: Any) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


@pytest.fixture(autouse=True)
def _checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the installed MOIST as checked, so the tests below compare MoistHost with it.

    test_replaced_functions_are_checked is the test that it really is.
    """
    monkeypatch.setattr(pyscf_moist, "unchecked_functions", lambda: ())


# ---------------------------------------------------------------------------
# The moist.pyscf internals MoistHost relies on have not changed
# ---------------------------------------------------------------------------
class TestReplacedFunctions:
    @pytest.mark.parametrize("name", REPLACED)
    def test_replaced_functions_are_checked(self, name: str) -> None:
        found = installed_fingerprints()[name]
        assert found in CHECKED.get(name, {}), (
            f"MOIST {moist.__version__}: {name} has fingerprint {found!r}, which pyscf_moist "
            f"was not checked against, so attach() uses MOIST's own host.  If the rest of "
            f"this file passes with this MOIST, add it to CHECKED: {installed_fingerprints()}"
        )

    def test_unchecked_moist_gets_its_own_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pyscf_moist, "unchecked_functions", lambda: ("PySCFHost.__init__",))
        mf = attach(
            scf.RHF(_mol(MOLECULES["water"])),
            cavity=svdw_drop(),
            components=_components("C-PCM"),
            context=CONTEXT,
        )
        assert type(mf.with_moist) is mp._MoistState
        with pytest.raises(ValueError, match="effective core potentials"):
            attach(
                scf.RHF(_mol(ECP_MOLECULE)),
                cavity=svdw_drop(),
                components=_components("C-PCM"),
                context=CONTEXT,
            )


# ---------------------------------------------------------------------------
# Each host routine against MOIST's at the same input
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluated() -> dict[str, tuple[Any, Any, Any]]:
    """(stock solvation, MoistSolvation, density) per molecule, evaluated at the density."""
    out = {}
    for name, spec in MOLECULES.items():
        mol = _mol(spec)
        dm = _density(mol)
        pair = []
        for cls in (mp.PySCFSolvation, MoistSolvation):
            s = cls(mol, svdw_drop(), _components("IEF-PCM"), context=CONTEXT)
            s.evaluate(dm)
            pair.append(s)
        out[name] = (pair[0], pair[1], dm)
    return out


class TestHost:
    @pytest.mark.parametrize("name", MOLECULES)
    def test_energy_and_fock(self, evaluated: dict[str, Any], name: str) -> None:
        stock, fast, _ = evaluated[name]
        assert type(fast.host) is MoistHost
        _same(fast.energy, stock.energy)
        _same(fast.fock, stock.fock)

    @pytest.mark.parametrize("name", MOLECULES)
    def test_potential_and_fock_term(self, evaluated: dict[str, Any], name: str) -> None:
        stock, fast, _ = evaluated[name]
        cavity = fast.model.cavity
        coords, xi = cavity.xyz, cavity.xi0
        q = np.random.default_rng(1).standard_normal(xi.size)
        _same(fast.host.surface_potential(coords, xi), stock.host.surface_potential(coords, xi))
        _same(fast.host.fock_potential(coords, xi, q), stock.host.fock_potential(coords, xi, q))

    @pytest.mark.parametrize("name", MOLECULES)
    def test_fused_gradient_pass(self, evaluated: dict[str, Any], name: str) -> None:
        stock, fast, _ = evaluated[name]
        cavity = fast.model.cavity
        coords, xi = cavity.xyz, cavity.xi0
        q = np.random.default_rng(2).standard_normal(xi.size)
        fast.host.fused_pass(coords, xi, q)
        try:
            _same(fast.host._grad_phi_elec(coords, xi), stock.host._grad_phi_elec(coords, xi))
            _same(fast.host._gradient_phi(coords, q, xi), stock.host._gradient_phi(coords, q, xi))
            # other charges than the pass's: MOIST's own route, same answer
            q2 = q[::-1].copy()
            _same(
                fast.host._gradient_phi(coords, q2, xi),
                stock.host._gradient_phi(coords, q2, xi),
            )
        finally:
            fast.host.clear_fused()

    @pytest.mark.parametrize("name", ["water", "formate"])
    def test_gradient_channels(self, evaluated: dict[str, Any], name: str) -> None:
        stock, fast, dm = evaluated[name]
        a, b = stock.gradient_channels(dm), fast.gradient_channels(dm)
        assert a.keys() == b.keys()
        for channel in a:
            _same(b[channel], a[channel])
        assert fast.host._fused is None

    def test_integrals_that_do_not_fit_fall_back(self) -> None:
        mol = _mol(MOLECULES["formate"])
        dm = _density(mol)
        mol.max_memory = 1  # MB: nothing fits
        stock = mp.PySCFSolvation(mol, svdw_drop(), _components("C-PCM"), context=CONTEXT)
        fast = MoistSolvation(mol, svdw_drop(), _components("C-PCM"), context=CONTEXT)
        _same(fast.evaluate(dm).energy, stock.evaluate(dm).energy)
        _same(fast.fock, stock.fock)
        assert fast.host._packed is not None and fast.host._packed[1] is None
        _same(fast.gradient(dm), stock.gradient(dm))


# ---------------------------------------------------------------------------
# Whole SCF and gradient runs
# ---------------------------------------------------------------------------
class TestSCF:
    @pytest.mark.parametrize("cavity", [svdw_drop, rho_drop])
    @pytest.mark.parametrize("model", ["C-PCM", "IEF-PCM"])
    @pytest.mark.parametrize("name", ["water", "formate", "methyl"])
    def test_scf_and_gradient_match_stock(self, name: str, model: str, cavity: Any) -> None:
        mol = _mol(MOLECULES[name])
        results = []
        for hook in (mp.moist_for_scf, attach):
            base = dft.UKS(mol, xc="b3lyp") if mol.spin else dft.RKS(mol, xc="b3lyp")
            mf = hook(base, cavity=cavity(), components=_components(model), context=CONTEXT)
            mf.conv_tol = 1e-11
            e = mf.kernel()
            assert mf.converged
            g = mf.nuc_grad_method()
            g.verbose = 0
            results.append((e, g.kernel(), type(mf.with_moist)))
        (e0, g0, state0), (e1, g1, state1) = results
        assert state0 is mp._MoistState and state1 is pyscf_moist._MoistSolvationState
        assert abs(e1 - e0) < 1e-9
        np.testing.assert_allclose(g1, g0, atol=1e-8)

    def test_copy_and_scanner_keep_the_fast_host(self) -> None:
        mf = attach(
            dft.RKS(_mol(MOLECULES["water"]), xc="b3lyp"),
            cavity=svdw_drop(),
            components=_components("C-PCM"),
            context=CONTEXT,
        )
        assert type(mf.copy().with_moist) is pyscf_moist._MoistSolvationState
        assert type(mf.as_scanner().with_moist) is pyscf_moist._MoistSolvationState


# ---------------------------------------------------------------------------
# ECPs, which MOIST's own host refuses, on both cavities
# ---------------------------------------------------------------------------
def _pyscf_pcm_on(cavity: Any, mol: Any, dm: Any, monkeypatch: pytest.MonkeyPatch) -> float:
    """PySCF's C-PCM energy at ``dm`` on MOIST's surface (its points, widths and switches)."""
    f = np.asarray(cavity.f, dtype=float)
    area = np.asarray(cavity.a, dtype=float)
    surface = {
        "ng": f.size,
        "gslice_by_atom": None,
        "grid_coords": np.asarray(cavity.xyz, dtype=float),
        "weights": area / np.where(f > 0, f, 1.0),
        "R_vdw": np.ones(f.size),  # so that PySCF's A is MOIST's area
        "charge_exp": np.asarray(cavity.xi0, dtype=float),
        "switch_fun": f,
        "norm_vec": np.asarray(cavity.normal0, dtype=float),
        "area": area,
    }
    monkeypatch.setattr(pcm, "gen_surface", lambda *a, **k: dict(surface))
    ws = pcm.PCM(mol)
    ws.method, ws.eps = "C-PCM", EPS
    return float(ws.kernel(dm)[0])


class TestECP:
    def test_energy_matches_pyscf_pcm_on_the_same_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mol = _mol(ECP_MOLECULE)
        assert mol.has_ecp()
        dm = _density(mol)
        s = MoistSolvation(mol, svdw_drop(), _components("C-PCM"), context=CONTEXT, allow_ecp=True)
        energy = s.evaluate(dm).energy
        assert abs(energy - _pyscf_pcm_on(s.model.cavity, mol, dm, monkeypatch)) < 1e-10

    @pytest.mark.parametrize("cavity", [svdw_drop, rho_drop])
    def test_gradient_matches_finite_differences(self, cavity: Any) -> None:
        def run(mol: Any) -> Any:
            # Hartree-Fock: a DFT grid moves with the atoms, which the analytic gradient
            # leaves out, and that alone is a few 1e-6 here
            mf = attach(
                scf.RHF(mol),
                cavity=cavity(),
                components=_components("C-PCM"),
                context=CONTEXT,
            )
            mf.conv_tol = 1e-12
            mf.max_cycle = 200
            mf.kernel()
            assert mf.converged and type(mf.with_moist) is pyscf_moist._MoistSolvationState
            return mf

        mol = _mol(ECP_MOLECULE)
        g = run(mol).nuc_grad_method()
        g.verbose = 0
        analytic = g.kernel()
        h = 1e-3  # bohr
        coords = mol.atom_coords()
        for atom, axis in [(1, 2), (0, 0)]:  # along the bond, and across it
            energies = []
            for step in (h, -h):
                moved = coords.copy()
                moved[atom, axis] += step
                energies.append(run(mol.set_geom_(moved, unit="Bohr", inplace=False)).e_tot)
            assert abs((energies[0] - energies[1]) / (2 * h) - analytic[atom, axis]) < 2e-7

    def test_isodensity_core_fill_leaves_the_outer_surface(self) -> None:
        """The fill guards against an inner sheet at the ECP nucleus; outside, no change."""
        mol = _mol(ECP_MOLECULE)
        dm = _density(mol)
        cavities = []
        for core_fill in (True, False):
            lsf = Isodensity(parameters=IsodensityParameters(core_fill=core_fill))
            s = MoistSolvation(
                mol, DROP(lsf=lsf), _components("C-PCM"), context=CONTEXT, allow_ecp=True
            )
            s.evaluate(dm)
            cavities.append(s.model.cavity)
        with_fill, without = cavities
        np.testing.assert_array_equal(with_fill.xyz, without.xyz)
        iodine = mol.atom_coords()[1]
        assert np.linalg.norm(np.asarray(with_fill.xyz) - iodine, axis=1).min() > 3.0


# ---------------------------------------------------------------------------
# The PySCF runner's DROP solvent models ("<PCM model>/<cavity>")
# ---------------------------------------------------------------------------
WATER = Geometry(
    symbols=("O", "H", "H"),
    coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
)


class TestRunner:
    @pytest.mark.parametrize(
        ("model", "component", "lsf"),
        [
            ("IEFPCM/SvdW-DROP", ModelComponentIEFPCM, "SvdW"),
            ("CPCM/rho-DROP", ModelComponentCPCM, "Isodensity"),
            ("COSMO/SvdW-DROP", ModelComponentCOSMO, "SvdW"),
        ],
    )
    def test_build_mf_attaches_moist(self, model: str, component: Any, lsf: str) -> None:
        _, mf = pyscf_runner._build_mf(WATER, 0, "b3lyp", "def2-svp", model, "water")
        assert type(mf.with_moist) is pyscf_moist._MoistSolvationState
        assert hasattr(mf, "with_df")  # density fitting underneath, as for PySCF's PCM
        (item,) = mf.with_moist.components
        assert type(item) is component
        assert type(mf.with_moist.cavity.lsf).__name__ == lsf
        assert mf.with_moist.cavity.parameters.wleb_prune_level == pyscf_moist.PRUNE_LEVEL

    @pytest.mark.parametrize("model", ["IEFPCM/SvdW-DROP", "IEFPCM/rho-DROP"])
    def test_single_point_with_vdzp(self, model: str) -> None:
        """wB97X-3c/vDZP: an ECP on the oxygen, which MOIST's own host would refuse."""
        gas = pyscf_runner.single_point(WATER, 0, "wB97X-3c", "vDZP")
        solvated = pyscf_runner.single_point(WATER, 0, "wB97X-3c", "vDZP", model, "water")
        # water's electrostatic solvation energy, in kcal/mol: -6.5 on PySCF's own cavity
        assert -8.0 < (solvated - gas) * 627.5095 < -6.0

    @pytest.mark.parametrize("model", ["IEFPCM/SvdW-DROP", "IEFPCM/rho-DROP"])
    def test_optimize(self, model: str) -> None:
        geom, energy, converged = pyscf_runner.optimize(
            WATER, 0, "b3lyp", "def2-svp", model, "water"
        )
        assert converged
        assert energy == pytest.approx(
            pyscf_runner.single_point(geom, 0, "b3lyp", "def2-svp", model, "water"), abs=1e-7
        )

    def test_unconverged_scf_retries_with_a_level_shift(self) -> None:
        _, mf = pyscf_runner._build_mf(WATER, 0, "b3lyp", "def2-svp", "IEFPCM/SvdW-DROP", "water")
        reference = pyscf_runner._run_scf_robust(mf)
        _, mf = pyscf_runner._build_mf(WATER, 0, "b3lyp", "def2-svp", "IEFPCM/SvdW-DROP", "water")
        mf.max_cycle = 2
        assert pyscf_runner._run_scf_robust(mf) == pytest.approx(reference, abs=1e-7)
        assert mf.converged and mf.level_shift == 0.0

    def test_no_hessian(self) -> None:
        with pytest.raises(NotImplementedError, match="no Hessian"):
            pyscf_runner.frequencies(WATER, 0, "b3lyp", "def2-svp", "IEFPCM/SvdW-DROP", "water")

    def test_hydrogen_radius_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pcm_hydrogen_radius"):
            pyscf_runner._build_mf(WATER, 0, "b3lyp", "def2-svp", "IEFPCM/SvdW-DROP", "water", 1.2)
