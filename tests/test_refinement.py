"""Refinement-stage flow: what gets computed, what gets skipped, what is recorded.

The drivers are stubbed. What is under test is the order of operations and the
bookkeeping around it -- which conformers reach the solvation single point, and
what the ensemble says about the ones that did not.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from tautomator import refinement, xtb_runner
from tautomator.config import RefinementConfig
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate

FREQS = [1200.0, 1800.0, 3000.0]


def geom(stretch: float) -> Geometry:
    """A water-like geometry, stretched so that no two of them deduplicate."""
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.757 + stretch, 0.586], [0.0, -0.757, 0.586]]),
    )


def ensemble_of(gas_energies: list[float]) -> Ensemble:
    """One charge state, one microstate, one conformer per energy (Hartree)."""
    ms = Microstate(
        tautomer_id="[H]O[H]",
        conformers=[Conformer(geometry=geom(i * 0.4)) for i in range(len(gas_energies))],
        smiles="[H]O[H]",
    )
    ens = Ensemble(input_smiles="O")
    ens.charge_states[0] = ChargeState(charge=0, microstates=[ms])
    return ens


class FakeDriver:
    """Records every call; energies come from the tables passed in."""

    def __init__(
        self,
        opt: list[float] | None = None,
        gas: dict[float, float] | None = None,
        solvated: dict[float, float] | None = None,
        fail_solvation: bool = False,
        fail_preload: bool = False,
    ) -> None:
        self.opt_energies = list(opt or [])
        self.gas = gas or {}
        self.solvated = solvated or {}
        self.fail_solvation = fail_solvation
        self.fail_preload = fail_preload
        self.calls: list[tuple[str, str, bool]] = []  # (kind, method, solvated?)

    # The drivers key everything off geometry; these stubs key off the z offset
    # each conformer was built with, which survives the (no-op) optimization.
    @staticmethod
    def _key(g: Geometry) -> float:
        return round(float(g.coords[1, 1]) - 0.757, 6)

    def preload(self, method: str) -> None:
        self.calls.append(("preload", method, False))
        if self.fail_preload:
            raise OSError("429 Client Error: Too Many Requests")

    def optimize(
        self, g: Geometry, charge: int, method: str, basis: str, *a: Any, **kw: Any
    ) -> tuple[Geometry, float, bool]:
        self.calls.append(("optimize", method, a[0] is not None))
        return g, self.opt_energies.pop(0), True

    def single_point(
        self, g: Geometry, charge: int, method: str, basis: str, *a: Any, **kw: Any
    ) -> float:
        in_solvent = bool(a) and a[0] is not None
        self.calls.append(("single_point", method, in_solvent))
        if in_solvent:
            if self.fail_solvation:
                raise RuntimeError("SCF did not converge")
            return self.solvated[self._key(g)]
        return self.gas[self._key(g)]

    def frequencies(self, *a: Any, **kw: Any) -> list[float]:
        self.calls.append(("frequencies", "", False))
        return list(FREQS)

    def count(self, kind: str, solvated: bool | None = None) -> int:
        return sum(
            1 for k, _, s in self.calls if k == kind and (solvated is None or s == solvated)
        )


@pytest.fixture()
def patch_drivers(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install stub drivers; returns a setter taking (engine_driver, dft_driver)."""

    def install(engine: FakeDriver, dft: FakeDriver | None = None) -> None:
        table = {"orbmol": engine, "pyscf": dft or engine, "psi4": dft or engine}
        monkeypatch.setattr(refinement, "_get_driver", lambda name, **_: table[name])

    return install


class TestDftSolvationSinglePoint:
    def test_reuses_the_optimizers_gas_energy(self, patch_drivers: Any) -> None:
        # One solvated single point per conformer and no second gas one: the
        # optimization already produced the gas-phase energy at this level.
        driver = FakeDriver(
            opt=[-76.40, -76.39],
            solvated={0.0: -76.41, 0.4: -76.405},
        )
        patch_drivers(driver)
        ens = ensemble_of([-76.40, -76.39])
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation="single_point",
            rrho="refinement",
            ewin=100.0,
            engine="pyscf",
        )
        refinement.refine(ens, cfg, driver="pyscf", cavity="ISWIG")

        confs = ens.charge_states[0].microstates[0].conformers
        assert len(confs) == 2
        assert driver.count("single_point", solvated=True) == 2
        assert driver.count("single_point", solvated=False) == 0
        assert confs[0].electronic_energy == pytest.approx(-76.40)
        assert confs[0].solvation_energy == pytest.approx(-0.01)

    def test_failure_drops_the_conformer_with_a_record(self, patch_drivers: Any) -> None:
        # A conformer with no solvation term is not comparable to its siblings,
        # so it cannot stay in the ensemble carrying a bare gas energy.
        driver = FakeDriver(opt=[-76.40], fail_solvation=True)
        patch_drivers(driver)
        ens = ensemble_of([-76.40])
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation="single_point",
            rrho="refinement",
            engine="pyscf",
        )
        refinement.refine(ens, cfg, driver="pyscf", cavity="ISWIG")

        ms = ens.charge_states[0].microstates[0]
        assert ms.conformers == []
        (excluded,) = ms.excluded_conformers
        assert excluded.reason == "solvation_failed"
        assert excluded.stage == "refinement"
        assert excluded.electronic_energy == pytest.approx(-76.40)
        assert excluded.rrho_correction is not None  # kept for diagnosis
        assert "SCF did not converge" in excluded.detail


class TestOrbmolSolvationSinglePoint:
    def test_takes_a_dft_pair_on_the_compute_driver(self, patch_drivers: Any) -> None:
        # OrbMol energies are on a different scale from the DFT solvation level,
        # so the term needs both halves of the pair at that level -- and both
        # from the DFT driver, not from OrbMol.
        orbmol = FakeDriver(opt=[-76.40])
        dft = FakeDriver(gas={0.0: -76.30}, solvated={0.0: -76.32})
        patch_drivers(orbmol, dft)
        ens = ensemble_of([-76.40])
        cfg = RefinementConfig(
            method="OrbMol-v2",
            basis="",
            solvation="single_point",
            solvation_method="wB97X-3c",
            solvation_basis="vDZP",
            rrho="refinement",
            engine="orbmol",
        )
        refinement.refine(ens, cfg, driver="pyscf", cavity="ISWIG")

        (conf,) = ens.charge_states[0].microstates[0].conformers
        assert conf.electronic_energy == pytest.approx(-76.40)  # OrbMol's
        assert conf.solvation_energy == pytest.approx(-0.02)  # the DFT pair's difference
        assert orbmol.count("single_point") == 0
        assert [c[1] for c in dft.calls] == ["wB97X-3c", "wB97X-3c"]


class TestOrbmolModelLoad:
    def test_a_model_that_cannot_load_stops_the_run(self, patch_drivers: Any) -> None:
        # Not a conformer's failure.  Caught by the per-conformer handler, it
        # excluded every conformer as "optimization_failed" and left a run that
        # finished, exit 0, with an empty ensemble.
        orbmol = FakeDriver(opt=[-76.40], fail_preload=True)
        patch_drivers(orbmol, FakeDriver())
        ens = ensemble_of([-76.40])
        cfg = RefinementConfig(
            method="OrbMol-v2", basis="", solvation="none", rrho="refinement", engine="orbmol"
        )
        with pytest.raises(OSError, match="Too Many Requests"):
            refinement.refine(ens, cfg, driver="pyscf")

        ms = ens.charge_states[0].microstates[0]
        assert len(ms.conformers) == 1
        assert ms.excluded_conformers == []
        assert orbmol.count("optimize") == 0


class TestGasPrefilter:
    def _run(self, window: float | None, caplog: pytest.LogCaptureFixture, patch: Any) -> Any:
        # Three conformers spread over ~10 kcal/mol in gas phase.
        spread = [0.0, 5.0, 10.0]  # kcal/mol above the minimum
        opt = [-76.40 + k / HARTREE_TO_KCAL for k in spread]
        driver = FakeDriver(
            opt=list(opt),
            solvated={round(i * 0.4, 6): e - 0.01 for i, e in enumerate(opt)},
        )
        patch(driver)
        ens = ensemble_of(opt)
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation="single_point",
            solvation_prefilter_ewin=window,
            rrho="refinement",
            ewin=100.0,
            engine="pyscf",
        )
        with caplog.at_level(logging.INFO):
            refinement.refine(ens, cfg, driver="pyscf", cavity="ISWIG")
        return ens, driver

    def test_off_by_default_every_conformer_gets_a_single_point(
        self, patch_drivers: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        ens, driver = self._run(None, caplog, patch_drivers)
        assert driver.count("single_point", solvated=True) == 3
        assert ens.charge_states[0].microstates[0].excluded_conformers == []

    def test_skips_the_ones_above_the_window(
        self, patch_drivers: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        ens, driver = self._run(7.0, caplog, patch_drivers)
        assert driver.count("single_point", solvated=True) == 2  # not the 10 kcal/mol one
        ms = ens.charge_states[0].microstates[0]
        assert len(ms.conformers) == 2
        (skipped,) = ms.excluded_conformers
        assert skipped.reason == "gas_prefilter"
        # The record says how far outside it was, so a finished run can be
        # checked for near misses rather than taken on faith.
        assert "above the gas-phase minimum" in skipped.detail
        assert float(skipped.detail.split()[0]) > 7.0
        assert skipped.electronic_energy is not None
        assert skipped.rrho_correction is not None
        assert skipped.solvation_energy is None

    def test_says_so_in_the_log(
        self, patch_drivers: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._run(7.0, caplog, patch_drivers)
        assert "gas-phase pre-filter at 7.0 kcal/mol kept 2 of 3" in caplog.text


class TestSolvatedOptimization:
    def test_decomposes_the_optimized_energy(self, patch_drivers: Any) -> None:
        # PySCF minimizes in implicit solvent, so the optimizer's energy already
        # includes solvation; a gas single point at the same geometry splits it.
        driver = FakeDriver(opt=[-76.41], gas={0.0: -76.40})
        patch_drivers(driver)
        ens = ensemble_of([-76.41])
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation="optimize",
            rrho="refinement",
            engine="pyscf",
        )
        refinement.refine(ens, cfg, driver="pyscf", cavity="ISWIG")

        (conf,) = ens.charge_states[0].microstates[0].conformers
        assert conf.electronic_energy == pytest.approx(-76.40)
        assert conf.solvation_energy == pytest.approx(-0.01)
        assert driver.count("optimize", solvated=True) == 1
        assert driver.count("single_point", solvated=True) == 0


class TestGasPhaseRefinement:
    def test_leaves_solvation_unset(self, patch_drivers: Any) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        ens = ensemble_of([-76.40])
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation="none",
            rrho="refinement",
            engine="pyscf",
        )
        refinement.refine(ens, cfg, driver="pyscf")

        (conf,) = ens.charge_states[0].microstates[0].conformers
        assert conf.solvation_energy is None
        assert driver.count("single_point") == 0


class TestXtbRrhoSolvent:
    """The xtb RRHO goes in the run's solvent, and in the gas phase for a run
    scored in the gas phase: a gas-phase free energy with an ALPB RRHO term
    would carry part of the solvation free energy it is the reference for."""

    @pytest.mark.parametrize(
        ("solvation", "cavity", "expected"),
        [("optimize", "ISWIG", "woctanol"), ("none", None, None)],
    )
    def test_follows_the_scoring_phase(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        solvation: str,
        cavity: str | None,
        expected: str | None,
    ) -> None:
        # Minimizing in solvent also takes the gas-phase energy at the minimum.
        patch_drivers(FakeDriver(opt=[-76.40], gas={0.0: -76.39}))
        seen: list[str | None] = []

        def fake_frequencies(*a: Any, solvent: str | None = None, **kw: Any) -> list[float]:
            seen.append(solvent)
            return list(FREQS)

        monkeypatch.setattr(xtb_runner, "frequencies", fake_frequencies)
        cfg = RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation=solvation,  # type: ignore[arg-type]
            rrho="xtb",
            ewin=100.0,
            engine="pyscf",
        )
        refinement.refine(
            ensemble_of([-76.40]),
            cfg,
            driver="pyscf",
            solvent="woctanol",
            cavity=cavity,
        )
        assert seen == [expected]
