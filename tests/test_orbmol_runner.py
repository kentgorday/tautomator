"""Tests for the OrbMol-v2 refinement driver."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tautomator import orbmol_runner
from tautomator.config import RefinementConfig
from tautomator.refinement import _get_driver, refine
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate

HARTREE_TO_KCAL = 627.509474
METHOD = "OrbMol-v2"

CHLORIDE = Geometry(symbols=("Cl",), coords=np.zeros((1, 3)))

WATER = Geometry(
    symbols=("O", "H", "H"),
    coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]),
)
HYDROXIDE = Geometry(symbols=("O", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.970]]))

# wB97M-V/def2-TZVPD, the model's training level: PySCF 2.13.1, (99, 590) grid,
# (50, 194) VV10 grid, conv_tol 1e-11, at the geometries above.  Frequencies are
# at the wB97M-V/def2-TZVPD minimum of water.
REF_E_WATER = -76.433427268778
REF_E_HYDROXIDE = -75.80608220987526
REF_WATER_FREQS = (1622.67, 3823.08, 3923.39)


class TestWithoutModel:
    def test_registered_as_refinement_driver(self) -> None:
        assert _get_driver("orbmol") is orbmol_runner

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown OrbMol model"):
            orbmol_runner.single_point(WATER, 0, "wB97X-3c")

    @pytest.mark.parametrize("fn", ["single_point", "optimize", "frequencies"])
    @pytest.mark.parametrize("geom", [WATER, CHLORIDE], ids=["water", "atom"])
    def test_solvent_model_refused(self, fn: str, geom: Geometry) -> None:
        # An atom too: optimize and frequencies return early for one, and must not
        # hand back a gas-phase answer to a solvated request.
        charge = -1 if geom is CHLORIDE else 0
        with pytest.raises(ValueError, match="gas-phase potential"):
            getattr(orbmol_runner, fn)(
                geom, charge, METHOD, solvent_model="IEFPCM", solvent="water"
            )

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "weights.ckpt"
        bogus.write_bytes(b"not the model")
        with pytest.raises(RuntimeError, match="sha256"):
            orbmol_runner._verified_weights(str(bogus), orbmol_runner.MODELS["orbmol-v2"].sha256)

    def test_cached_weights_are_used_without_asking_the_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # cached_path() revalidates over the network on every call; a 429 there
        # failed every load of weights that were already on disk.
        import cached_path

        def no_network(*args: object, **kwargs: object) -> Path:
            raise AssertionError("went to the network with a cached copy on disk")

        cached = tmp_path / "weights.ckpt"
        monkeypatch.delenv("TAUTOMATOR_ORBMOL_WEIGHTS", raising=False)
        monkeypatch.setattr(cached_path, "find_latest_cached", lambda url: cached)
        monkeypatch.setattr(cached_path, "cached_path", no_network)
        assert orbmol_runner._weights_path(orbmol_runner.MODELS["orbmol-v2"]) == str(cached)

    def test_weights_are_downloaded_when_none_are_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cached_path

        fetched = tmp_path / "fetched.ckpt"
        monkeypatch.delenv("TAUTOMATOR_ORBMOL_WEIGHTS", raising=False)
        monkeypatch.setattr(cached_path, "find_latest_cached", lambda url: None)
        monkeypatch.setattr(cached_path, "cached_path", lambda url: fetched)
        assert orbmol_runner._weights_path(orbmol_runner.MODELS["orbmol-v2"]) == str(fetched)


@pytest.mark.slow  # loads a 100 MB model (downloaded on first use)
class TestAgainstTrainingLevel:
    def test_loading_prints_nothing(self) -> None:
        # warp prints an initialisation banner unless told not to before orb_models
        # initialises it; a fresh process, since this one may have loaded it already.
        code = (
            "import numpy as np\n"
            "from tautomator import orbmol_runner\n"
            "from tautomator.types import Geometry\n"
            "geom = Geometry(('O', 'H'), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]))\n"
            f"orbmol_runner.single_point(geom, -1, {METHOD!r})\n"
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert run.stdout == ""

    def test_energies(self) -> None:
        e_water = orbmol_runner.single_point(WATER, 0, METHOD)
        e_oh = orbmol_runner.single_point(HYDROXIDE, -1, METHOD)
        assert abs(e_water - REF_E_WATER) * HARTREE_TO_KCAL < 0.5
        assert abs(e_oh - REF_E_HYDROXIDE) * HARTREE_TO_KCAL < 0.5
        # The charge input matters: deprotonation energy, ~394 kcal/mol here.
        ref = (REF_E_HYDROXIDE - REF_E_WATER) * HARTREE_TO_KCAL
        assert (e_oh - e_water) * HARTREE_TO_KCAL == pytest.approx(ref, abs=0.5)

    def test_gradient_is_derivative_of_energy(self) -> None:
        # Deliberately a distorted geometry with a large step: the model runs at
        # float32-high (see _PRECISION), so a total energy of ~-2000 eV carries
        # ~1e-4 eV of noise.  Differencing that over a small step swamps the
        # result -- at the water minimum with h=1e-4 A the reference is only good
        # to ~1e-5 Hartree/Bohr against gradients of 4e-3, i.e. 0.4%.  Here the
        # gradients are ~0.1 Hartree/Bohr and h is 10x larger, so the finite
        # difference is good to 0.01% and the check is meaningful again.
        distorted = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.95, 0.75], [0.0, -0.80, 0.55]]),
        )
        _, grad = orbmol_runner._energy_gradient(distorted, 0, METHOD)
        assert np.abs(grad).max() > 0.05  # the premise above
        h = 1e-3  # Angstrom
        for atom, axis in [(0, 2), (1, 1), (2, 2)]:
            dx = np.zeros_like(distorted.coords)
            dx[atom, axis] = h
            ep = orbmol_runner.single_point(
                Geometry(distorted.symbols, distorted.coords + dx), 0, METHOD
            )
            em = orbmol_runner.single_point(
                Geometry(distorted.symbols, distorted.coords - dx), 0, METHOD
            )
            fd = (ep - em) / (2 * h / orbmol_runner.BOHR_TO_ANG)
            assert grad[atom, axis] == pytest.approx(fd, abs=5e-5)

    def test_optimize_returns_energy_of_returned_geometry(self) -> None:
        # The pyscf driver once returned the input geometry's energy; guard the
        # same contract here.
        opt, energy, converged = orbmol_runner.optimize(WATER, 0, METHOD)
        assert converged
        assert energy == pytest.approx(orbmol_runner.single_point(opt, 0, METHOD), abs=1e-10)
        assert energy < orbmol_runner.single_point(WATER, 0, METHOD)

    def test_optimize_leaves_logging_alone(self, caplog: pytest.LogCaptureFixture) -> None:
        # geomeTRIC reconfigures logging on every run by default; that used to
        # remove the pipeline's handlers and silence its log after the first
        # optimization.  Nor should geomeTRIC's per-step output leak through.
        root = logging.getLogger()
        with caplog.at_level(logging.INFO):
            handlers = list(root.handlers)
            orbmol_runner.optimize(WATER, 0, METHOD)
            assert root.handlers == handlers
            logging.getLogger("tautomator.refinement").info("still logging")
        assert "still logging" in caplog.text
        assert not [r for r in caplog.records if r.name.startswith("geometric")]

    def test_frequencies(self) -> None:
        opt, _, _ = orbmol_runner.optimize(WATER, 0, METHOD)
        freqs = orbmol_runner.frequencies(opt, 0, METHOD)
        assert len(freqs) == 3  # 3N - 6
        np.testing.assert_allclose(sorted(freqs), REF_WATER_FREQS, atol=10.0)


@pytest.mark.slow
def test_refine_with_orbmol_driver() -> None:
    """refine() runs end to end on the orbmol driver, Hessian RRHO included."""
    start = Geometry(
        WATER.symbols, WATER.coords + np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0], [0, 0, 0]])
    )
    ms = Microstate(
        tautomer_id="[H]O[H]", conformers=[Conformer(geometry=start)], smiles="[H]O[H]"
    )
    ensemble = Ensemble(input_smiles="O")
    ensemble.charge_states[0] = ChargeState(charge=0, microstates=[ms])

    config = RefinementConfig(
        method=METHOD, basis="", solvation="none", rrho="refinement", engine="orbmol"
    )
    refine(ensemble, config, driver="pyscf")

    (conf,) = ensemble.charge_states[0].microstates[0].conformers
    assert conf.refinement_converged
    assert conf.solvation_energy is None  # gas phase
    assert conf.electronic_energy == pytest.approx(
        orbmol_runner.single_point(conf.geometry, 0, METHOD), abs=1e-10
    )
    assert conf.rrho_correction is not None and conf.rrho_correction > 0
