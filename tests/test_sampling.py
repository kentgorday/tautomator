"""Tests for the sampling stage."""

from __future__ import annotations

import numpy as np
import pytest

from tautomator import sampling
from tautomator.protomer_geometry import MigrationReport
from tautomator.types import Conformer, Geometry


def test_a_failed_hessian_does_not_evict_the_microstate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conformer with no RRHO term cannot be compared: free_energy sums
    non-None components, so it is short a term worth 11-114 kcal/mol against an
    ensemble whose real spread is a few kcal/mol. Left in the window it becomes
    e_min and evicts every genuine conformer of the microstate."""
    geom = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )
    healthy = [Conformer(geometry=geom, electronic_energy=-76.0 - i * 0.0016) for i in range(5)]
    broken = Conformer(geometry=geom, electronic_energy=-76.002)

    calls = {"n": 0}

    def fake_frequencies(*args: object, **kwargs: object) -> list[float]:
        calls["n"] += 1
        if calls["n"] == 6:  # the last conformer's Hessian fails
            raise RuntimeError("xtb --hess produced no frequencies")
        return [100.0, 200.0, 300.0]

    monkeypatch.setattr(sampling, "frequencies", fake_frequencies)
    monkeypatch.setattr(sampling, "quasi_rrho_free_energy", lambda f: 0.12)
    monkeypatch.setattr(sampling, "deduplicate_conformers", lambda c, e, ethr_kcal=None: c)

    out = sampling._dedupe_add_rrho_and_filter([*healthy, broken], 0, "water", 6.0)

    # every genuine conformer survives ...
    assert all(any(c is o for o in out) for c in healthy)
    # ... and the one with no RRHO is carried to refinement, which recomputes it
    assert any(broken is o for o in out)
    assert broken.rrho_correction is None


class TestDissociationIsCaughtBeforeTheConformerSearch:
    """CREST cannot survive a dissociated input, and is slow to say so.

    Its iMTD-GC calibrates with a trial metadynamics that diverges on a loose
    fragment: it halves the time step four times, disables SHAKE, then exits
    non-zero. On molecule 4 of the pKa set this cost eight searches -- four
    at ~4 s where a proton had left outright, four at 17-33 min where a fluoride
    still weakly interacting at 2.7-4.9 A let the trial calibrate and the full
    70 ps run before failing downstream.

    The optimizer raises nothing: xtb converges onto the dissociated structure
    and reports success. `repair_migrated_conformers` catches it, but runs after
    the search, so the time is already spent.
    """

    @staticmethod
    def _intact_water() -> Geometry:
        return Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        )

    def test_an_intact_molecule_passes(self) -> None:
        assert sampling._dissociated(self._intact_water()) is None

    def test_a_detached_hydrogen_is_caught(self) -> None:
        """The fast failure: a proton 4.7 A out makes the trial MTD diverge at once."""
        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [4.7, 0.0, 0.0]]),
        )
        reason = sampling._dissociated(geom)
        assert reason is not None
        assert "further than" in reason
        assert "2" in reason, "the reason should name the distance it used"

    def test_a_broken_framework_is_caught(self) -> None:
        """The slow failure: a heavy atom leaves and takes no hydrogen with it."""
        geom = Geometry(
            symbols=("C", "F", "H", "H", "H"),
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [4.92, 0.0, 0.0],  # fluoride, as it left the CF3 in the real case
                    [-0.6, 0.9, 0.0],
                    [-0.6, -0.45, 0.78],
                    [-0.6, -0.45, -0.78],
                ]
            ),
        )
        reason = sampling._dissociated(geom)
        assert reason is not None
        assert "came apart" in reason
        assert "2 pieces" in reason

    def test_a_framework_break_is_reported_even_with_every_hydrogen_attached(self) -> None:
        """Why `is_intact` alone is not enough.

        A departing heavy atom keeps its own hydrogens, so every H stays ~1 A
        from *a* heavy atom and the hydrogen check sees nothing wrong. This is
        the case that ran for half an hour.
        """
        geom = Geometry(
            symbols=("C", "H", "H", "H", "O", "H"),
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-0.6, 0.9, 0.0],
                    [-0.6, -0.45, 0.78],
                    [-0.6, -0.45, -0.78],
                    [5.0, 0.0, 0.0],  # a water's worth of oxygen, 5 A away
                    [5.96, 0.0, 0.0],  # with its own hydrogen
                ]
            ),
        )
        from tautomator.tautomer_dedup import assign_protons

        assert assign_protons(geom).is_intact, "the hydrogen check cannot see this"
        reason = sampling._dissociated(geom)
        assert reason is not None and "came apart" in reason

    def test_the_reason_is_specific_enough_to_act_on(self) -> None:
        """It goes into a warning a human reads, so it must name what broke."""
        detached = sampling._dissociated(
            Geometry(
                symbols=("O", "H", "H"),
                coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [4.7, 0.0, 0.0]]),
            )
        )
        broken = sampling._dissociated(
            Geometry(
                symbols=("C", "F", "H", "H", "H"),
                coords=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [4.92, 0.0, 0.0],
                        [-0.6, 0.9, 0.0],
                        [-0.6, -0.45, 0.78],
                        [-0.6, -0.45, -0.78],
                    ]
                ),
            )
        )
        assert detached != broken, "the two failure modes must be distinguishable in the log"


def test_a_dissociated_geometry_skips_the_conformer_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate being right does not prove it is wired in.

    Drives `run_approach1` on water with the expensive calls stubbed, and makes
    `conformer_search` fail the test if it is ever reached. Before the check,
    every microstate went to CREST regardless of whether it was bound.
    """
    dissociated = Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [4.7, 0.0, 0.0]]),
    )
    monkeypatch.setattr(sampling, "smiles_to_3d", lambda smi, **kw: (dissociated, smi))
    monkeypatch.setattr(sampling, "optimize", lambda g, **kw: (dissociated, True))
    monkeypatch.setattr(sampling, "single_point", lambda g, **kw: -76.0)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("conformer_search must not run on a dissociated geometry")

    monkeypatch.setattr(sampling, "conformer_search", forbidden)
    # The repair pass and the Hessians are downstream of the point being tested.
    monkeypatch.setattr(
        sampling, "repair_migrated_conformers", lambda cs, stage: MigrationReport()
    )
    monkeypatch.setattr(
        sampling, "_dedupe_add_rrho_and_filter", lambda confs, *a, **kw: list(confs)
    )

    ensemble = sampling.run_approach1("O", charge_range=(0, 0), solvent="water")

    assert ensemble.charge_states, "the microstate is kept, not dropped"
    kept = [
        c for cs in ensemble.charge_states.values() for ms in cs.microstates for c in ms.conformers
    ]
    assert len(kept) == 1, "one stand-in conformer, for the repair pass to exclude"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("smiles", "charge", "has_vibration"), [("[Cl-]", -1, False), ("[OH-]", -1, True)]
)
def test_atoms_and_diatomics_get_one_conformer_without_a_search(
    monkeypatch: pytest.MonkeyPatch, smiles: str, charge: int, has_vibration: bool
) -> None:
    """With no torsion there is nothing to search, and CREST and xtb's optimizer
    stop on a single atom: a halide used to leave its charge state with no
    microstate at all.  An atom also has no vibrations, though xtb's Hessian
    reports its translations as three near-zero modes, worth -6.5 kcal/mol to
    quasi-RRHO if they were let through."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("conformer_search must not run on fewer than three atoms")

    monkeypatch.setattr(sampling, "conformer_search", forbidden)

    ensemble = sampling.run_approach1(smiles, charge_range=(charge, charge), solvent="water")

    (ms,) = ensemble.charge_states[charge].microstates
    (conf,) = ms.conformers
    assert conf.solvation_energy is not None and conf.solvation_energy < 0
    assert conf.rrho_correction is not None
    # An atom's vibrational free energy is exactly zero; OH- has its stretch.
    if has_vibration:
        assert conf.rrho_correction > 0
    else:
        assert conf.rrho_correction == 0.0


@pytest.mark.slow
def test_crest_first_takes_a_single_atom_without_a_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Approach 2's pre-screen called CREST unguarded, and CREST exits 1 on a lone
    atom, so the error escaped `run_approach2`."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("conformer_search must not run on fewer than three atoms")

    monkeypatch.setattr(sampling, "conformer_search", forbidden)

    ensemble = sampling.run_approach2("[Ne]", charge_range=(0, 0), solvent="water")

    (ms,) = ensemble.charge_states[0].microstates
    (conf,) = ms.conformers
    assert conf.rrho_correction == 0.0


class TestFixedSpecies:
    """MNSol and similar data fix the species, so a run must not add tautomers
    or charge-separated protomers whose weights would change the answer."""

    @staticmethod
    def _sampled_species(monkeypatch: pytest.MonkeyPatch, fixed: bool) -> list[str]:
        """Species run_approach1 searches for glycine, with every program stubbed."""

        def one_conformer(geom: Geometry, **kwargs: object) -> list[Conformer]:
            return [Conformer(geometry=geom, electronic_energy=-1.0)]

        monkeypatch.setattr(sampling, "optimize", lambda g, **kw: (g, True))
        monkeypatch.setattr(sampling, "single_point", lambda g, **kw: -1.0)
        monkeypatch.setattr(sampling, "conformer_search", one_conformer)
        monkeypatch.setattr(
            sampling, "repair_migrated_conformers", lambda cs, stage: MigrationReport()
        )
        monkeypatch.setattr(
            sampling, "_dedupe_add_rrho_and_filter", lambda confs, *a, **kw: list(confs)
        )
        ensemble = sampling.run_approach1(
            "NCC(=O)O", charge_range=(0, 0), solvent="water", fixed_species=fixed
        )
        return sorted(ms.tautomer_id for ms in ensemble.charge_states[0].microstates)

    def test_only_the_input_is_sampled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._sampled_species(monkeypatch, fixed=True) == ["NCC(=O)O"]

    def test_otherwise_the_zwitterion_joins_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert "[NH3+]CC(=O)[O-]" in self._sampled_species(monkeypatch, fixed=False)

    def test_another_charge_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at its own charge"):
            sampling.run_approach1("NCC(=O)O", charge_range=(-1, -1), fixed_species=True)
