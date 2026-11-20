import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from tautomator.ensemble import (
    HARTREE_TO_KCAL,
    _conformer_multiplicity,
    assign_weights,
    boltzmann_weights,
    charge_state_free_energy,
    deduplicate_charge_state,
    deduplicate_conformers,
    ensemble_free_energy,
    ensemble_to_sdf,
    filter_charge_state_by_energy,
    load_ensemble,
    serialize_ensemble,
)
from tautomator.types import (
    ChargeState,
    Conformer,
    Ensemble,
    ExcludedConformer,
    Geometry,
    Microstate,
    Well,
)


def _make_conformer(energy: float) -> Conformer:
    geom = Geometry(symbols=("H",), coords=np.zeros((1, 3)))
    return Conformer(geometry=geom, electronic_energy=energy)


class TestBoltzmannWeights:
    def test_single_energy(self) -> None:
        w = boltzmann_weights([-1.0])
        assert w == pytest.approx([1.0])

    def test_degenerate_energies(self) -> None:
        w = boltzmann_weights([-1.0, -1.0])
        assert w == pytest.approx([0.5, 0.5])

    def test_lower_energy_higher_weight(self) -> None:
        w = boltzmann_weights([-1.1, -1.0])
        assert w[0] > w[1]

    def test_sums_to_one(self) -> None:
        w = boltzmann_weights([-1.5, -1.4, -1.3, -1.2])
        assert sum(w) == pytest.approx(1.0)


class TestEnsembleFreeEnergy:
    def test_single_conformer(self) -> None:
        g = ensemble_free_energy([-1.0])
        assert g == pytest.approx(-1.0)

    def test_degenerate_lowers_free_energy(self) -> None:
        g_single = ensemble_free_energy([-1.0])
        g_double = ensemble_free_energy([-1.0, -1.0])
        # Two degenerate states -> lower free energy (by kT*ln2)
        assert g_double < g_single

    def test_high_energy_conformer_negligible(self) -> None:
        g_one = ensemble_free_energy([-1.0])
        # Adding a conformer 100 kcal/mol higher should barely change G
        g_two = ensemble_free_energy([-1.0, -1.0 + 100.0 / HARTREE_TO_KCAL])
        assert abs(g_one - g_two) < 1e-10


class TestChargeStateFreeEnergy:
    def test_basic(self) -> None:
        cs = ChargeState(
            charge=0,
            microstates=[
                Microstate(tautomer_id="a", conformers=[_make_conformer(-1.0)]),
                Microstate(tautomer_id="b", conformers=[_make_conformer(-1.01)]),
            ],
        )
        g = charge_state_free_energy(cs)
        assert g < -1.0  # Lower than the lowest individual energy

    def test_empty_raises(self) -> None:
        cs = ChargeState(charge=0, microstates=[])
        with pytest.raises(ValueError):
            charge_state_free_energy(cs)


class TestAssignWeights:
    def test_weights_across_microstates(self) -> None:
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(tautomer_id="a", conformers=[_make_conformer(-1.0)]),
                        Microstate(tautomer_id="b", conformers=[_make_conformer(-1.0)]),
                    ],
                ),
            },
        )
        assign_weights(ens)
        # Two degenerate conformers across microstates should each get 0.5
        w0 = ens.charge_states[0].microstates[0].conformers[0].weight
        w1 = ens.charge_states[0].microstates[1].conformers[0].weight
        assert w0 == pytest.approx(0.5)
        assert w1 == pytest.approx(0.5)


class TestSerialization:
    def test_round_trip(self, tmp_path: Path) -> None:
        geom = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.96, 0.0, 0.0],
                    [-0.24, 0.93, 0.0],
                ]
            ),
        )
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="abc123",
                            conformers=[
                                Conformer(geometry=geom, electronic_energy=-76.43, weight=1.0)
                            ],
                            smiles="O",
                        ),
                    ],
                ),
            },
            settings={"solvent": "water"},
        )
        json_path = serialize_ensemble(ens, tmp_path / "output")
        ens2 = load_ensemble(json_path)
        assert ens2.input_smiles == "O"
        assert 0 in ens2.charge_states
        assert len(ens2.charge_states[0].microstates) == 1
        conf = ens2.charge_states[0].microstates[0].conformers[0]
        assert conf.electronic_energy == pytest.approx(-76.43)
        np.testing.assert_allclose(conf.geometry.coords, geom.coords, atol=1e-8)


def _water_geom_smiles_order() -> Geometry:
    """Water geometry in SMILES atom order: [H]O[H] -> H, O, H."""
    return Geometry(
        symbols=("H", "O", "H"),
        coords=np.array(
            [
                [0.96, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        ),
    )


class TestEnsembleToSdf:
    def test_with_smiles(self, tmp_path: Path) -> None:
        """Approach 1: explicit-H SMILES provides bond orders."""
        geom = _water_geom_smiles_order()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="O",
                            conformers=[
                                Conformer(geometry=geom, electronic_energy=-76.4, weight=1.0)
                            ],
                            smiles="[H]O[H]",
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        assert sdf_path.exists()
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 1
        mol = mols[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        assert mol.GetNumBonds() == 2
        assert int(mol.GetProp("charge")) == 0
        assert mol.GetProp("tautomer_id") == "O"
        assert float(mol.GetDoubleProp("free_energy_hartree")) == pytest.approx(-76.4)

    def test_without_smiles(self, tmp_path: Path) -> None:
        """Approach 2: no SMILES, bonds from rdDetermineBonds."""
        geom = _water_geom_smiles_order()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="fp_abc",
                            conformers=[
                                Conformer(geometry=geom, electronic_energy=-76.4, weight=1.0)
                            ],
                            smiles=None,
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 1
        mol = mols[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 3
        assert mol.GetNumBonds() == 2

    def test_multiple_charge_states(self, tmp_path: Path) -> None:
        geom = _water_geom_smiles_order()
        ens = Ensemble(
            input_smiles="O",
            charge_states={
                0: ChargeState(
                    charge=0,
                    microstates=[
                        Microstate(
                            tautomer_id="O",
                            conformers=[
                                Conformer(geometry=geom, electronic_energy=-76.4, weight=0.6),
                                Conformer(geometry=geom, electronic_energy=-76.3, weight=0.4),
                            ],
                            smiles="[H]O[H]",
                        ),
                    ],
                ),
                -1: ChargeState(
                    charge=-1,
                    microstates=[
                        Microstate(
                            tautomer_id="[OH-]",
                            conformers=[
                                Conformer(
                                    geometry=Geometry(
                                        symbols=("H", "O"),
                                        coords=np.array([[0.96, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                                    ),
                                    electronic_energy=-75.8,
                                    weight=1.0,
                                ),
                            ],
                            smiles="[H][O-]",
                        ),
                    ],
                ),
            },
        )
        sdf_path = ensemble_to_sdf(ens, tmp_path / "test.sdf")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = list(suppl)
        assert len(mols) == 3  # 2 neutral conformers + 1 anion


# ---------------------------------------------------------------------------
# Symmetry wiring: deduplication, multiplicity, and the effective-energy window
# ---------------------------------------------------------------------------

# Staggered methylammonium (C3v, sigma = 3) and a chiral four-atom centre.
_MEAM_SYMBOLS = ("H", "C", "H", "H", "N", "H", "H", "H")
_MEAM = np.array(
    [
        [-1.15328, 1.00618, 0.21311],
        [-0.80967, -0.00001, 0.00005],
        [-1.15324, -0.68765, 0.76489],
        [-1.15320, -0.31859, -0.97790],
        [0.68275, 0.00004, 0.00010],
        [1.05716, 0.29384, 0.90191],
        [1.05720, 0.63414, -0.70522],
        [1.05724, -0.92782, -0.19637],
    ]
)
_CHIRAL_SYMBOLS = ("C", "F", "Cl", "Br")
_CHIRAL = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])


def _conf(coords: np.ndarray, symbols: tuple[str, ...], energy: float) -> Conformer:
    return Conformer(
        geometry=Geometry(symbols=symbols, coords=coords.copy()), electronic_energy=energy
    )


class TestDeduplicateChargeState:
    def test_collapses_duplicates_and_sets_multiplicity(self) -> None:
        """A rotated copy is the same state; the survivor carries sigma = 3."""
        rotated = _MEAM @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
        ms = Microstate(
            tautomer_id="x",
            conformers=[_conf(_MEAM, _MEAM_SYMBOLS, -1.0), _conf(rotated, _MEAM_SYMBOLS, -1.0)],
        )
        cs = ChargeState(charge=1, microstates=[ms])
        before, after = deduplicate_charge_state(cs)
        assert (before, after) == (2, 1)
        assert ms.conformers[0].multiplicity == pytest.approx(1 / 3)

    def test_does_not_compare_across_microstates(self) -> None:
        """Different microstates are different species and are never merged."""
        a = Microstate(tautomer_id="a", conformers=[_conf(_MEAM, _MEAM_SYMBOLS, -1.0)])
        b = Microstate(tautomer_id="b", conformers=[_conf(_MEAM, _MEAM_SYMBOLS, -1.0)])
        cs = ChargeState(charge=1, microstates=[a, b])
        assert deduplicate_charge_state(cs) == (2, 2)
        assert len(a.conformers) == 1 and len(b.conformers) == 1

    def test_mirror_pair_merges_and_doubles(self) -> None:
        mirrored = _CHIRAL * np.array([1.0, 1.0, -1.0])
        ms = Microstate(
            tautomer_id="x",
            conformers=[
                _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0),
                _conf(mirrored, _CHIRAL_SYMBOLS, -1.0),
            ],
        )
        cs = ChargeState(charge=0, microstates=[ms])
        assert deduplicate_charge_state(cs) == (2, 1)
        assert ms.conformers[0].multiplicity == pytest.approx(2.0)

    def test_prefers_a_converged_representative(self) -> None:
        rotated = _MEAM @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
        unconverged = _conf(_MEAM, _MEAM_SYMBOLS, -2.0)
        unconverged.refinement_converged = False
        converged = _conf(rotated, _MEAM_SYMBOLS, -1.0)
        converged.refinement_converged = True
        ms = Microstate(tautomer_id="x", conformers=[unconverged, converged])
        deduplicate_charge_state(ChargeState(charge=1, microstates=[ms]))
        assert ms.conformers[0] is converged

    def test_energy_criterion_is_optional(self) -> None:
        """With ethr set, structures far apart in energy are not merged."""
        rotated = _MEAM @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
        pair = [_conf(_MEAM, _MEAM_SYMBOLS, -1.0), _conf(rotated, _MEAM_SYMBOLS, -1.0 + 0.01)]
        ms = Microstate(tautomer_id="x", conformers=list(pair))
        assert deduplicate_charge_state(
            ChargeState(charge=1, microstates=[ms]), ethr_kcal=0.05
        ) == (2, 2)
        ms2 = Microstate(tautomer_id="x", conformers=list(pair))
        assert deduplicate_charge_state(ChargeState(charge=1, microstates=[ms2])) == (2, 1)


class TestMultiplicityInWeighting:
    def test_enantiomer_and_conformer_factors_multiply(self) -> None:
        conf = _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0)
        conf.multiplicity = 2.0
        ms = Microstate(tautomer_id="x", conformers=[conf], includes_enantiomer=True)
        assert _conformer_multiplicity(ms, conf) == pytest.approx(4.0)

    def test_a_doubled_conformer_outweighs_a_degenerate_single(self) -> None:
        single = _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0)
        doubled = _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0)
        doubled.multiplicity = 2.0
        ens = Ensemble(input_smiles="C")
        ens.charge_states[0] = ChargeState(
            charge=0, microstates=[Microstate(tautomer_id="x", conformers=[single, doubled])]
        )
        assign_weights(ens)
        assert single.weight is not None
        assert doubled.weight == pytest.approx(2 * single.weight)


class TestEffectiveEnergyWindow:
    def test_multiplicity_extends_the_window(self) -> None:
        """A conformer standing for two states survives 0.41 kcal/mol further out."""
        window = 1.0
        just_outside = -1.0 + (window + 0.2) / HARTREE_TO_KCAL
        base = _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0)
        edge = _conf(_CHIRAL, _CHIRAL_SYMBOLS, just_outside)

        plain = ChargeState(
            charge=0, microstates=[Microstate(tautomer_id="x", conformers=[base, edge])]
        )
        filter_charge_state_by_energy(plain, window)
        assert len(plain.microstates[0].conformers) == 1

        edge2 = _conf(_CHIRAL, _CHIRAL_SYMBOLS, just_outside)
        edge2.multiplicity = 2.0
        boosted = ChargeState(
            charge=0,
            microstates=[
                Microstate(
                    tautomer_id="x",
                    conformers=[_conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0), edge2],
                )
            ],
        )
        filter_charge_state_by_energy(boosted, window)
        assert len(boosted.microstates[0].conformers) == 2


class TestMultiplicityRoundTrip:
    def test_survives_serialization(self, tmp_path: Path) -> None:
        conf = _conf(_CHIRAL, _CHIRAL_SYMBOLS, -1.0)
        conf.multiplicity = 0.5
        ens = Ensemble(input_smiles="C")
        ens.charge_states[0] = ChargeState(
            charge=0, microstates=[Microstate(tautomer_id="x", conformers=[conf])]
        )
        loaded = load_ensemble(serialize_ensemble(ens, tmp_path))
        assert loaded.charge_states[0].microstates[0].conformers[0].multiplicity == 0.5

    def test_defaults_to_one_when_absent(self, tmp_path: Path) -> None:
        """Ensembles written before multiplicity existed still load."""
        path = tmp_path / "ensemble.json"
        path.write_text(
            json.dumps(
                {
                    "input_smiles": "C",
                    "charge_states": {
                        "0": {
                            "charge": 0,
                            "microstates": [
                                {
                                    "tautomer_id": "x",
                                    "conformers": [
                                        {
                                            "symbols": ["H"],
                                            "coords": [[0.0, 0.0, 0.0]],
                                            "electronic_energy": -1.0,
                                        }
                                    ],
                                }
                            ],
                        }
                    },
                }
            )
        )
        loaded = load_ensemble(path)
        assert loaded.charge_states[0].microstates[0].conformers[0].multiplicity == 1.0


class TestAtomOrderingGuard:
    """Conformers of one microstate must share an atom ordering.

    The invariant holds because a microstate is one labelled species, and the
    comparison machinery relies on it: a single symbol list is used for both
    sides of every pairwise comparison. A proton that migrated between heavy
    atoms during sampling or optimization would break it, so it fails loudly
    rather than producing a silently wrong grouping.
    """

    def test_mismatched_ordering_raises(self) -> None:
        a = Conformer(
            geometry=Geometry(symbols=("O", "H", "H"), coords=np.zeros((3, 3))),
            electronic_energy=-1.0,
        )
        b = Conformer(
            geometry=Geometry(symbols=("H", "O", "H"), coords=np.zeros((3, 3))),
            electronic_energy=-1.0,
        )
        with pytest.raises(ValueError, match="atom ordering"):
            deduplicate_conformers([a, b])

    def test_empty_list_is_fine(self) -> None:
        assert deduplicate_conformers([]) == []


class TestRepresentative:
    WATER = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])

    @pytest.mark.parametrize(("stale_rrho", "kept"), [(False, 1), (True, 0)])
    def test_a_stale_rrho_term_does_not_pick_the_survivor(
        self, stale_rrho: bool, kept: int
    ) -> None:
        # The second is higher in the optimized energy, lower once a stale RRHO term is added.
        confs = [
            Conformer(
                geometry=Geometry(("O", "H", "H"), self.WATER + d),
                electronic_energy=e,
                rrho_correction=r,
            )
            for d, e, r in ((0.0, -76.41, 0.02), (1e-4, -76.40, 0.0))
        ]
        (survivor,) = deduplicate_conformers(confs, stale_rrho=stale_rrho)
        assert survivor is confs[kept]


# ---------------------------------------------------------------------------
# Excluded conformers
# ---------------------------------------------------------------------------
def _geom() -> Geometry:
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )


def test_excluded_conformers_survive_a_json_round_trip(tmp_path: Path) -> None:
    """The record is the whole point: a run must be able to say what it could
    not compute, after the fact, from the output file alone."""
    ms = Microstate(
        tautomer_id="O",
        conformers=[Conformer(geometry=_geom(), electronic_energy=-1.0, rrho_correction=0.1)],
        excluded_conformers=[
            ExcludedConformer(
                geometry=_geom(),
                stage="refinement",
                reason="rrho_failed",
                detail="RuntimeError: xtb hessian blew up",
                multiplicity=3.0,
                electronic_energy=-2.0,
                solvation_energy=-0.05,
            )
        ],
    )
    ens = Ensemble(input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])})
    serialize_ensemble(ens, tmp_path)
    back = load_ensemble(tmp_path / "ensemble.json")

    exc = back.charge_states[0].microstates[0].excluded_conformers
    assert len(exc) == 1
    assert exc[0].reason == "rrho_failed"
    assert exc[0].stage == "refinement"
    assert "RuntimeError" in exc[0].detail
    # multiplicity is what makes the loss legible: deduplication runs before the
    # Hessians, so an excluded representative takes its collapsed duplicates too.
    assert exc[0].multiplicity == 3.0
    assert exc[0].electronic_energy == -2.0


def test_excluded_conformers_are_not_weighted_or_filtered() -> None:
    """Structural exclusion, not a flag: nothing that weights or filters the
    ensemble can reach them, so no call site has to remember to check."""
    good = Conformer(geometry=_geom(), electronic_energy=-1.0)
    ms = Microstate(
        tautomer_id="O",
        conformers=[good],
        excluded_conformers=[
            # An energy far below the real ensemble -- exactly the case that
            # used to capture all the weight when it was kept as a Conformer.
            ExcludedConformer(
                geometry=_geom(),
                stage="refinement",
                reason="rrho_failed",
                electronic_energy=-500.0,
            )
        ],
    )
    ens = Ensemble(input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])})
    assign_weights(ens)
    assert good.weight == pytest.approx(1.0)

    cs = ens.charge_states[0]
    filter_charge_state_by_energy(cs, ewin_kcal=10.0)
    assert cs.microstates[0].conformers == [good]
    assert len(cs.microstates[0].excluded_conformers) == 1


def test_excluded_conformer_has_no_free_energy() -> None:
    """It must be a type error to sum one into a partition function, rather
    than something that silently yields a plausible partial energy."""
    exc = ExcludedConformer(
        geometry=_geom(), stage="scoring", reason="scoring_failed", electronic_energy=-1.0
    )
    assert not hasattr(exc, "free_energy")


def test_a_file_without_excluded_conformers_still_loads(tmp_path: Path) -> None:
    """Written before the field existed. Absent is not wrong, just uninformative,
    which is why this needed no schema bump."""
    ens = Ensemble(
        input_smiles="O",
        charge_states={
            0: ChargeState(
                charge=0,
                microstates=[
                    Microstate(
                        tautomer_id="O",
                        conformers=[Conformer(geometry=_geom(), electronic_energy=-1.0)],
                    )
                ],
            )
        },
    )
    serialize_ensemble(ens, tmp_path)
    raw = json.loads((tmp_path / "ensemble.json").read_text())
    ms_raw = raw["charge_states"]["0"]["microstates"][0]
    assert "excluded_conformers" not in ms_raw  # omitted when empty
    assert (
        load_ensemble(tmp_path / "ensemble.json")
        .charge_states[0]
        .microstates[0]
        .excluded_conformers
        == []
    )


def test_pruning_keeps_a_microstate_that_only_has_excluded_conformers() -> None:
    """A microstate whose conformers all failed is the case most worth having on
    disk. Pruning on ``ms.conformers`` alone discarded the record with it."""
    g = _geom()
    dead = Microstate(
        tautomer_id="all_failed",
        conformers=[],
        excluded_conformers=[
            ExcludedConformer(
                geometry=g, stage="refinement", reason="rrho_failed", multiplicity=2.0
            )
        ],
    )
    alive = Microstate(
        tautomer_id="ok", conformers=[Conformer(geometry=g, electronic_energy=-76.0)]
    )
    cs = ChargeState(charge=0, microstates=[dead, alive])
    filter_charge_state_by_energy(cs, ewin_kcal=10.0)

    ids = {ms.tautomer_id for ms in cs.microstates}
    assert ids == {"all_failed", "ok"}
    assert sum(len(ms.excluded_conformers) for ms in cs.microstates) == 1
    # and it still contributes nothing to the energetics
    assert charge_state_free_energy(cs) == pytest.approx(-76.0)


class TestWellRecord:
    WELL = Well(depth=0.04, symmetric_multiplicity=0.5, broken_multiplicity=1.0, source="fold")

    def test_round_trips(self, tmp_path: Path) -> None:
        water = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]),
        )
        kept = Conformer(geometry=water, electronic_energy=-76.4, well=self.WELL)
        plain = Conformer(geometry=water, electronic_energy=-76.3)
        ms = Microstate(tautomer_id="O", conformers=[kept, plain], smiles="O")
        ens = Ensemble(
            input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])}
        )
        back = load_ensemble(serialize_ensemble(ens, tmp_path))
        a, b = back.charge_states[0].microstates[0].conformers
        assert a.well == self.WELL and b.well is None

    def test_round_trips_the_dropped_shape(self, tmp_path: Path) -> None:
        water = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
        shape = Geometry(("O", "H", "H"), water + np.array([0.0, 0.1, 0.0]))
        well = dataclasses.replace(self.WELL, dropped=shape)
        kept = Conformer(geometry=Geometry(("O", "H", "H"), water), electronic_energy=-76.4)
        kept.well = well
        ms = Microstate(tautomer_id="O", conformers=[kept], smiles="O")
        ens = Ensemble(
            input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])}
        )
        path = serialize_ensemble(ens, tmp_path)
        raw = json.loads(path.read_text())
        assert raw["charge_states"]["0"]["microstates"][0]["conformers"][0]["well"]["dropped"]
        (back,) = load_ensemble(path).charge_states[0].microstates[0].conformers
        assert back.well == self.WELL and back.well is not None
        assert back.well.dropped is not None
        assert back.well.dropped.symbols == ("O", "H", "H")
        assert np.allclose(back.well.dropped.coords, shape.coords)

    def test_a_record_from_before_the_dropped_shape_loads(self, tmp_path: Path) -> None:
        water = Geometry(("O", "H", "H"), np.eye(3))
        conf = Conformer(geometry=water, electronic_energy=-76.4, well=self.WELL)
        ms = Microstate(tautomer_id="O", conformers=[conf], smiles="O")
        ens = Ensemble(
            input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])}
        )
        path = serialize_ensemble(ens, tmp_path)
        raw = json.loads(path.read_text())
        del raw["charge_states"]["0"]["microstates"][0]["conformers"][0]["well"]["dropped"]
        path.write_text(json.dumps(raw))
        (back,) = load_ensemble(path).charge_states[0].microstates[0].conformers
        assert back.well == self.WELL and back.well is not None and back.well.dropped is None

    def test_an_older_file_loads_without_one(self, tmp_path: Path) -> None:
        water = Geometry(symbols=("O", "H", "H"), coords=np.eye(3))
        ms = Microstate(
            tautomer_id="O", conformers=[Conformer(geometry=water, electronic_energy=-76.4)]
        )
        ens = Ensemble(
            input_smiles="O", charge_states={0: ChargeState(charge=0, microstates=[ms])}
        )
        path = serialize_ensemble(ens, tmp_path)
        raw = json.loads(path.read_text())
        del raw["charge_states"]["0"]["microstates"][0]["conformers"][0]["well"]
        path.write_text(json.dumps(raw))
        assert load_ensemble(path).charge_states[0].microstates[0].conformers[0].well is None

    def test_deduplication_keeps_a_duplicates_record(self) -> None:
        water = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
        low = Conformer(geometry=Geometry(("O", "H", "H"), water), electronic_energy=-76.41)
        high = Conformer(
            geometry=Geometry(("O", "H", "H"), water + 1e-4),
            electronic_energy=-76.40,
            well=self.WELL,
        )
        (survivor,) = deduplicate_conformers([high, low])
        assert survivor is low and survivor.well == self.WELL

    def test_the_survivor_takes_the_lowest_members_record(self) -> None:
        water = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
        other = Well(depth=0.2, symmetric_multiplicity=0.5, broken_multiplicity=1.0, source="fold")
        confs = [
            Conformer(geometry=Geometry(("O", "H", "H"), water + d), electronic_energy=e, well=w)
            for d, e, w in ((0.0, -76.41, None), (1e-4, -76.39, other), (2e-4, -76.40, self.WELL))
        ]
        (survivor,) = deduplicate_conformers(confs)
        assert survivor is confs[0] and survivor.well == self.WELL
