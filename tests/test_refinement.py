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
from tautomator.conformer_symmetry import symmetry_number
from tautomator.ensemble import HARTREE_TO_KCAL, deduplicate_charge_state
from tautomator.nudge import SETTLE, Nudge, Settled
from tautomator.thermo import quasi_rrho_free_energy
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate, Well
from tests.top_cases import methanol, ring

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
        self.starts: list[Geometry] = []  # what each optimization started from

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
        self.starts.append(g)
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


@pytest.fixture(autouse=True)
def no_nudge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nudge takes a real xtb Hessian: stubbed to "nothing to move" unless a test
    says otherwise."""
    monkeypatch.setattr(refinement, "nudge", lambda g, *a, **kw: (g, None))


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


class TestNudgeBeforeOptimizing:
    """A symmetric conformer is optimized from its nudged geometry (tautomator.nudge)."""

    @staticmethod
    def cfg(
        nudge: bool = True, solvation: str = "none", collapse: bool = True
    ) -> RefinementConfig:
        return RefinementConfig(
            method="wB97X-3c",
            basis="vDZP",
            solvation=solvation,  # type: ignore[arg-type]
            rrho="refinement",
            ewin=100.0,
            nudge=nudge,
            collapse_h_tops=collapse,
            engine="pyscf",
        )

    @staticmethod
    def fake_nudge(seen: list[str | None], later: tuple[float, ...] = ()) -> Any:
        def fake(g: Geometry, charge: int, *, solvent: str | None, **kw: Any) -> Any:
            seen.append(solvent)
            # Off the molecular plane: FakeDriver keys its energies on y.
            moved = Geometry(symbols=g.symbols, coords=g.coords + np.array([0.1, 0.0, 0.0]))
            return moved, Nudge(
                frequency=24.0,
                symmetric_part=0.0,
                operations=(((0, 2, 1), False),),
                symmetry=symmetry_number(g.coords, list(g.symbols)),
                later=later,
            )

        return fake

    def test_logs_the_later_modes_pushed(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        patch_drivers(FakeDriver(opt=[-76.40]))
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([], later=(61.4, 128.0)))
        with caplog.at_level(logging.INFO, logger="tautomator.refinement"):
            refinement.refine(ensemble_of([-76.40]), self.cfg(), driver="pyscf")
        assert "along its 24 cm-1 GFN2 mode and its 61, 128 cm-1 modes" in caplog.text

    def test_optimizes_from_the_nudged_geometry(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        seen: list[str | None] = []
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge(seen))
        ens = ensemble_of([-76.40])
        sampled = ens.charge_states[0].microstates[0].conformers[0].geometry
        with caplog.at_level(logging.INFO, logger="tautomator.refinement"):
            refinement.refine(ens, self.cfg(), driver="pyscf")
        assert np.allclose(driver.starts[0].coords - sampled.coords, [0.1, 0.0, 0.0])
        assert "along its 24 cm-1 GFN2 mode" in caplog.text

    def test_a_failed_nudged_optimization_falls_back_to_the_sampled_geometry(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([]))
        settled: list[Geometry] = []
        monkeypatch.setattr(refinement, "settle", lambda done, g, *a: settled.append(g))
        succeed = driver.optimize

        def scf_fails_once(g: Geometry, *a: Any, **kw: Any) -> tuple[Geometry, float, bool]:
            if not driver.starts:
                driver.starts.append(g)
                raise RuntimeError("SCF did not converge")
            return succeed(g, *a, **kw)

        monkeypatch.setattr(driver, "optimize", scf_fails_once)
        ens = ensemble_of([-76.40])
        sampled = ens.charge_states[0].microstates[0].conformers[0].geometry
        out = refinement.refine(ens, self.cfg(), driver="pyscf")
        ms = out.charge_states[0].microstates[0]
        assert len(ms.conformers) == 1 and ms.excluded_conformers == []
        assert driver.starts[1] is sampled
        assert settled == []  # nothing nudged survived to settle
        assert "optimizing it as sampled" in caplog.text

    def test_off_when_disabled(self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        seen: list[str | None] = []
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge(seen))
        ens = ensemble_of([-76.40])
        sampled = ens.charge_states[0].microstates[0].conformers[0].geometry
        refinement.refine(ens, self.cfg(nudge=False), driver="pyscf")
        assert seen == [] and driver.starts[0] is sampled

    @pytest.mark.parametrize(
        ("solvation", "cavity", "expected"),
        [("optimize", "ISWIG", "woctanol"), ("none", None, None)],
    )
    def test_hessian_in_the_solvent_sampling_used(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        solvation: str,
        cavity: str | None,
        expected: str | None,
    ) -> None:
        # The conformer is a GFN2 minimum only in sampling's solvent: the gas
        # phase for a run scored in the gas phase, the run's solvent otherwise.
        patch_drivers(FakeDriver(opt=[-76.40], gas={0.0: -76.39}))
        seen: list[str | None] = []
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge(seen))
        refinement.refine(
            ensemble_of([-76.40]),
            self.cfg(solvation=solvation),
            driver="pyscf",
            solvent="woctanol",
            cavity=cavity,
        )
        assert seen == [expected]

    def test_settles_at_the_level_it_optimized_at(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The symmetric shape is optimized in the solvent the nudged optimization
        # ran in, and a shape that replaces the result is what the conformer keeps.
        driver = FakeDriver(opt=[-76.40, -76.41], gas={0.0: -76.39})
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([]))
        shape = geom(0.0)

        def fake_settle(done: Nudge, g: Geometry, e: float, optimize: Any) -> Settled:
            relaxed, energy, _ = optimize(shape)
            return Settled(relaxed, energy, (energy - e) * HARTREE_TO_KCAL)

        monkeypatch.setattr(refinement, "settle", fake_settle)
        out = refinement.refine(
            ensemble_of([-76.40]),
            self.cfg(solvation="optimize"),
            driver="pyscf",
            solvent="woctanol",
            cavity="ISWIG",
        )
        conf = out.charge_states[0].microstates[0].conformers[0]
        assert conf.geometry is shape
        assert conf.electronic_energy == pytest.approx(-76.39)
        assert conf.solvation_energy == pytest.approx(-0.02)
        assert driver.count("optimize", solvated=True) == 2
        assert driver.starts[1] is shape

    def test_a_deeper_well_keeps_the_nudged_result(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver = FakeDriver(opt=[-76.40, -76.39], gas={0.0: -76.38, 0.5: -76.38})
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([]))

        def fake_settle(done: Nudge, g: Geometry, e: float, optimize: Any) -> Settled:
            relaxed, energy, _ = optimize(geom(0.5))
            return Settled(relaxed, energy, (energy - e) * HARTREE_TO_KCAL)

        monkeypatch.setattr(refinement, "settle", fake_settle)
        out = refinement.refine(ensemble_of([-76.40]), self.cfg(), driver="pyscf")
        conf = out.charge_states[0].microstates[0].conformers[0]
        assert conf.electronic_energy == pytest.approx(-76.40)
        assert conf.geometry is driver.starts[0]

    @pytest.mark.parametrize(
        ("nudge", "below", "kept"),
        [(True, 0.04, 1), (True, SETTLE + 0.05, 2), (False, 0.04, 2)],
    )
    def test_folds_a_shallow_well_into_the_symmetric_conformer(
        self, patch_drivers: Any, nudge: bool, below: float, kept: int
    ) -> None:
        # The stretched copy, `below` kcal/mol under the symmetric conformer beside
        # it, is that conformer's well when the well is shallow; only with the
        # nudge on, the switch for everything settle does.
        e = -76.40
        patch_drivers(FakeDriver(opt=[e, e - below / HARTREE_TO_KCAL]))
        out = refinement.refine(ensemble_of([e, e]), self.cfg(nudge=nudge), driver="pyscf")
        confs = out.charge_states[0].microstates[0].conformers
        assert len(confs) == kept
        assert np.allclose(confs[0].geometry.coords, geom(0.0).coords)
        # The symmetric conformer (C2v, 1/2) records the stretched well (Cs, 1) it stands for.
        folded = Well(
            depth=pytest.approx(below),  # type: ignore[arg-type]
            symmetric_multiplicity=0.5,
            broken_multiplicity=1.0,
            source="fold",
        )
        assert confs[0].well == (folded if kept == 1 else None)

    @pytest.mark.parametrize(("nudge", "multiplicity"), [(True, 0.5), (False, 1.0)])
    def test_folds_before_deduplicating(
        self, patch_drivers: Any, nudge: bool, multiplicity: float
    ) -> None:
        # A symmetric conformer and a lopsided copy near enough to deduplicate with
        # it, yet far enough from its own image to count as less symmetric.
        # Deduplication keeps the lower of a group, here the copy at twice the
        # count, so the fold has to see the pair first.
        e = -76.40
        patch_drivers(FakeDriver(opt=[e, e - 0.04 / HARTREE_TO_KCAL]))
        ens = ensemble_of([e, e])
        sampled = ens.charge_states[0].microstates[0].conformers
        symmetric = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.957, 0.586], [0.0, -0.957, 0.586]]),
        )
        sampled[0].geometry = symmetric
        out = refinement.refine(ens, self.cfg(nudge=nudge), driver="pyscf")
        (conf,) = out.charge_states[0].microstates[0].conformers
        assert (conf.geometry is symmetric) == nudge
        assert conf.multiplicity == multiplicity

    def test_an_unconverged_nudged_result_is_not_settled(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Still rolling downhill, its energy says nothing about its well's depth.
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([]))
        settled: list[Geometry] = []
        monkeypatch.setattr(refinement, "settle", lambda done, g, *a: settled.append(g))
        stalls = driver.optimize

        def unconverged(g: Geometry, *a: Any, **kw: Any) -> tuple[Geometry, float, bool]:
            relaxed, energy, _ = stalls(g, *a, **kw)
            return relaxed, energy, False

        monkeypatch.setattr(driver, "optimize", unconverged)
        out = refinement.refine(ensemble_of([-76.40]), self.cfg(), driver="pyscf")
        (conf,) = out.charge_states[0].microstates[0].conformers
        assert settled == [] and conf.refinement_converged is False

    def test_a_failing_nudge_keeps_the_conformer(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)

        def broken(*a: Any, **kw: Any) -> None:
            raise ValueError("no relabelling")

        monkeypatch.setattr(refinement, "nudge", broken)
        ens = ensemble_of([-76.40])
        sampled = ens.charge_states[0].microstates[0].conformers[0].geometry
        out = refinement.refine(ens, self.cfg(), driver="pyscf")
        ms = out.charge_states[0].microstates[0]
        assert len(ms.conformers) == 1 and ms.excluded_conformers == []
        assert driver.starts == [sampled]
        assert "The nudge failed" in caplog.text and "optimizing it as sampled" in caplog.text

    def test_a_failing_settle_keeps_the_nudged_result(
        self,
        patch_drivers: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.fake_nudge([]))

        def broken(*a: Any, **kw: Any) -> None:
            raise ValueError("no relabelling")

        monkeypatch.setattr(refinement, "settle", broken)
        out = refinement.refine(ensemble_of([-76.40]), self.cfg(), driver="pyscf")
        (conf,) = out.charge_states[0].microstates[0].conformers
        assert conf.geometry is driver.starts[0]
        assert "keeping the nudged result" in caplog.text

    def test_no_settling_without_a_nudge(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(*a: Any, **kw: Any) -> None:
            raise AssertionError("settle ran for a conformer that was not nudged")

        monkeypatch.setattr(refinement, "settle", never)
        patch_drivers(FakeDriver(opt=[-76.40]))
        refinement.refine(ensemble_of([-76.40]), self.cfg(), driver="pyscf")


class TestWellRecord:
    """A settled conformer records the comparison, whichever shape it kept."""

    @staticmethod
    def lopsided_nudge(g: Geometry, charge: int, *, solvent: str | None, **kw: Any) -> Any:
        # Longer on one side: Cs, against the sampled C2v.
        moved = g.coords.copy()
        moved[1, 1] += 0.4
        return Geometry(symbols=g.symbols, coords=moved), Nudge(
            frequency=24.0,
            symmetric_part=0.0,
            operations=(((0, 2, 1), False),),
            symmetry=symmetry_number(g.coords, list(g.symbols)),
        )

    @pytest.mark.parametrize(("above", "replaced"), [(0.03, True), (0.2, False)])
    def test_settle_records_the_depth_and_both_counts(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch, above: float, replaced: bool
    ) -> None:
        e = -76.40
        driver = FakeDriver(opt=[e, e + above / HARTREE_TO_KCAL])
        patch_drivers(driver)
        monkeypatch.setattr(refinement, "nudge", self.lopsided_nudge)
        shape = geom(0.0)

        def fake_settle(done: Nudge, g: Geometry, energy: float, optimize: Any) -> Settled:
            relaxed, at, _ = optimize(shape)
            return Settled(relaxed, at, (at - energy) * HARTREE_TO_KCAL)

        monkeypatch.setattr(refinement, "settle", fake_settle)
        out = refinement.refine(ensemble_of([e]), TestNudgeBeforeOptimizing.cfg(), driver="pyscf")
        (conf,) = out.charge_states[0].microstates[0].conformers
        assert (conf.geometry is shape) == replaced
        assert conf.well == Well(
            depth=pytest.approx(above),  # type: ignore[arg-type]
            symmetric_multiplicity=0.5,
            broken_multiplicity=1.0,
            source="settle",
        )
        assert conf.multiplicity == (0.5 if replaced else 1.0)


class TestCollapsedTops:
    """A methyl's staggered and eclipsed turns are one state once its tops are collapsed."""

    @pytest.mark.parametrize("collapse", [True, False])
    def test_a_methyls_two_turns_merge_into_the_lower(
        self, patch_drivers: Any, collapse: bool
    ) -> None:
        # The eclipsed turn is 1 kcal/mol higher, but sampling's stale RRHO
        # would make it the lower free energy; the optimized energy decides.
        e = -115.0
        ms = Microstate(
            tautomer_id="CO",
            conformers=[
                Conformer(
                    geometry=Geometry(symbols=tuple(methanol(p)[1]), coords=methanol(p)[0]),
                    rrho_correction=r,
                )
                for p, r in ((90.0, 0.010), (30.0, 0.0))
            ],
            smiles="CO",
        )
        ens = Ensemble(input_smiles="CO")
        ens.charge_states[0] = ChargeState(charge=0, microstates=[ms])
        patch_drivers(FakeDriver(opt=[e, e + 1.0 / HARTREE_TO_KCAL]))
        out = refinement.refine(
            ens, TestNudgeBeforeOptimizing.cfg(collapse=collapse), driver="pyscf"
        )
        confs = out.charge_states[0].microstates[0].conformers
        assert len(confs) == (1 if collapse else 2)
        assert confs[0].electronic_energy == pytest.approx(e)

    def test_fold_sees_the_symmetric_shape_through_a_turned_methyl(self) -> None:
        # p-Toluic acid, planar with its methyl turned 15 degrees off the mirror,
        # and its well, the carboxyl twisted 10 degrees, 0.03 kcal/mol lower.
        # Taken uncollapsed both are chiral, so fold left the pair alone and
        # deduplication kept the well, counted as a mirror pair.
        e = -459.0
        confs = [
            Conformer(geometry=Geometry(tuple(s), x), electronic_energy=e - d / HARTREE_TO_KCAL)
            for (x, s), d in (
                (ring({0: 105.0}, carboxyl=True), 0.0),
                (ring({0: 90.0}, carboxyl=True, twist=10.0), 0.03),
            )
        ]
        cs = ChargeState(charge=0, microstates=[Microstate(tautomer_id="acid", conformers=confs)])
        assert refinement._fold_shallow_wells(cs, collapse_tops=True) == 1
        deduplicate_charge_state(cs, collapse_tops=True, stale_rrho=True)
        (kept,) = cs.microstates[0].conformers
        assert kept is confs[0] and kept.multiplicity == 1.0
        assert kept.well is not None and kept.well.depth == pytest.approx(0.03)


class TestRrhoFloor:
    @pytest.mark.parametrize(("floor", "counted"), [(50.0, 50.0), (0.0, 5.0)])
    def test_a_soft_mode_counts_as_the_floor(
        self, patch_drivers: Any, monkeypatch: pytest.MonkeyPatch, floor: float, counted: float
    ) -> None:
        driver = FakeDriver(opt=[-76.40])
        patch_drivers(driver)
        monkeypatch.setattr(driver, "frequencies", lambda *a, **kw: [5.0, 1200.0])
        cfg = TestNudgeBeforeOptimizing.cfg()
        cfg.rrho_floor = floor
        out = refinement.refine(ensemble_of([-76.40]), cfg, driver="pyscf")
        (conf,) = out.charge_states[0].microstates[0].conformers
        assert conf.rrho_correction == quasi_rrho_free_energy([counted, 1200.0])
