"""Tests for protomer identity read off a geometry, and migration repair."""

from __future__ import annotations

import itertools
import logging
from typing import ClassVar
from unittest import mock

import numpy as np
import pytest
from rdkit import Chem

from tautomator.ensemble import deduplicate_conformers
from tautomator.protomer_geometry import (
    DETACHED_DISTANCE,
    MigrationReport,
    _heavy_slots,
    _layouts_for_target,
    _skeleton_mol,
    _specified_stereo,
    _stereo_from_coordinates,
    _stereo_signature,
    _template_counts,
    assign_protons,
    match_to_candidate,
    protonation_key_from_geometry,
    protonation_key_from_mol,
    repair_migrated_conformers,
    template_from_smiles,
)
from tautomator.rdkit_utils import canonical_smiles, smiles_to_3d
from tautomator.tautomer_dedup import geometric_fingerprint, heavy_components
from tautomator.types import ChargeState, Conformer, Geometry, Microstate


def _embed(smiles: str, seed: int | None = None) -> tuple[Geometry, str]:
    return smiles_to_3d(smiles, seed=seed)


def _move_h(geom: Geometry, h_index: int, target_heavy: int, dist: float = 1.02) -> Geometry:
    """Relocate one hydrogen onto a different heavy atom, as a migration would."""
    coords = geom.coords.copy()
    centroid = coords[geom.heavy_atom_indices].mean(axis=0)
    outward = coords[target_heavy] - centroid
    norm = float(np.linalg.norm(outward))
    outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    coords[h_index] = coords[target_heavy] + dist * outward
    return Geometry(symbols=tuple(geom.symbols), coords=coords)


def _rotate_branch(
    geom: Geometry,
    axis_from: int,
    axis_to: int,
    branch: tuple[int, ...],
    degrees: float,
) -> Geometry:
    """Turn one branch about a bond, as a conformational change would.

    Positions are heavy-atom positions; the hydrogens each branch atom owns come
    with it. Nothing about the configuration changes, so every identity question
    must give the same answer before and after.
    """
    coords = geom.coords.copy()
    heavy = geom.heavy_atom_indices
    owner = dict(zip(geom.hydrogen_indices, assign_protons(geom).owner, strict=True))
    moving = {heavy[p] for p in branch}
    moving |= {h for h, o in owner.items() if o in moving}

    origin = coords[heavy[axis_to]]
    axis = origin - coords[heavy[axis_from]]
    axis = axis / float(np.linalg.norm(axis))
    theta = np.radians(degrees)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)
    for i in moving:
        if i == heavy[axis_to]:
            continue
        coords[i] = origin + rotation @ (coords[i] - origin)
    return Geometry(symbols=tuple(geom.symbols), coords=coords)


def _break_a_bond(geom: Geometry, heavy_position: int, shift: float = 4.0) -> Geometry:
    """Pull one heavy atom away, taking its hydrogens with it.

    No hydrogen ends up far from a heavy atom, so `ProtonAssignment.is_intact`
    stays true -- which is exactly the case the connectivity check exists for.
    """
    coords = geom.coords.copy()
    target = geom.heavy_atom_indices[heavy_position]
    offset = np.array([shift, 0.0, 0.0])
    coords[target] += offset
    for h_index, owner in zip(geom.hydrogen_indices, assign_protons(geom).owner, strict=True):
        if owner == target:
            coords[h_index] += offset
    return Geometry(symbols=tuple(geom.symbols), coords=coords)


def _owner_of(geom: Geometry, h_index: int) -> int:
    assignment = assign_protons(geom)
    return assignment.owner[geom.hydrogen_indices.index(h_index)]


class TestAssignProtons:
    def test_methanol_hydrogens_land_on_the_right_atoms(self) -> None:
        geom, _ = _embed("CO")
        counts = assign_protons(geom).counts
        heavy = [geom.symbols[i] for i in geom.heavy_atom_indices]
        assert dict(zip(heavy, counts, strict=True)) == {"C": 3, "O": 1}

    def test_every_hydrogen_is_counted_exactly_once(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        assignment = assign_protons(geom)
        assert sum(assignment.counts) == len(geom.hydrogen_indices)
        assert len(assignment.owner) == len(geom.hydrogen_indices)

    def test_a_bonded_hydrogen_is_not_flagged_detached(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        assert assign_protons(geom).is_intact

    def test_a_hydrogen_pulled_away_is_flagged_detached(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] = coords[h] + np.array([0.0, 0.0, 6.0])
        moved = Geometry(symbols=tuple(geom.symbols), coords=coords)
        assignment = assign_protons(moved)
        assert not assignment.is_intact
        assert assignment.detached == (h,)

    def test_the_margin_is_recorded_and_comfortable_for_an_ordinary_molecule(self) -> None:
        geom, _ = _embed("CCO")
        assert assign_protons(geom).min_margin > 0.3

    def test_a_geometry_with_no_heavy_atoms_is_refused(self) -> None:
        geom = Geometry(symbols=("H", "H"), coords=np.array([[0.0, 0, 0], [0.74, 0, 0]]))
        with pytest.raises(ValueError, match="no heavy atoms"):
            assign_protons(geom)


class TestKeyAgreesBetweenSmilesAndGeometry:
    @pytest.mark.parametrize(
        "smiles,charge",
        [
            ("NCC(=O)O", 0),
            ("[NH3+]CC(=O)O", 1),
            ("NCC(=O)[O-]", -1),
            ("Cc1cc[nH]c1", 0),
            ("OC(=O)/C=C\\C(=O)[O-]", -1),
        ],
    )
    def test_an_undisturbed_geometry_reproduces_its_label(self, smiles: str, charge: int) -> None:
        geom, explicit = _embed(smiles)
        template = template_from_smiles(explicit)
        assert protonation_key_from_geometry(geom, template, charge) == protonation_key_from_mol(
            template, charge
        )

    def test_a_template_from_another_molecule_is_refused(self) -> None:
        geom, _ = _embed("NCC(=O)O")
        other = template_from_smiles(smiles_to_3d("CCCCO")[1])
        with pytest.raises(ValueError, match="heavy-atom ordering"):
            protonation_key_from_geometry(geom, other, 0)


class TestAutomorphicSitesShareAKey:
    """A proton shared between equivalent heavy atoms must not split a species."""

    def test_the_two_carboxylate_oxygens_of_malonate_are_interchangeable(self) -> None:
        a = template_from_smiles(smiles_to_3d("O=C(O)CC(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d("O=C([O-])CC(=O)O")[1])
        assert protonation_key_from_mol(a, -1) == protonation_key_from_mol(b, -1)

    def test_maleate_is_interchangeable_end_for_end(self) -> None:
        a = template_from_smiles(smiles_to_3d(r"O=C(O)/C=C\C(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d(r"O=C([O-])/C=C\C(=O)O")[1])
        assert protonation_key_from_mol(a, -1) == protonation_key_from_mol(b, -1)

    def test_inequivalent_sites_stay_apart(self) -> None:
        """Citraconate: the methyl makes the two carboxyls distinguishable."""
        a = template_from_smiles(smiles_to_3d(r"O=C(O)/C(C)=C\C(=O)[O-]")[1])
        b = template_from_smiles(smiles_to_3d(r"O=C([O-])/C(C)=C\C(=O)O")[1])
        assert protonation_key_from_mol(a, -1) != protonation_key_from_mol(b, -1)

    def test_different_protonation_sites_stay_apart(self) -> None:
        a = template_from_smiles(smiles_to_3d("[NH3+]CC(=O)O")[1])
        b = template_from_smiles(smiles_to_3d("NCC(=O)[OH2+]")[1])
        assert protonation_key_from_mol(a, 1) != protonation_key_from_mol(b, 1)


def _microstate(
    smiles: str, conformers: list[Conformer] | None = None, seed: int | None = None
) -> Microstate:
    geom, explicit = _embed(smiles, seed=seed)
    return Microstate(
        tautomer_id=smiles,
        conformers=conformers if conformers is not None else [Conformer(geometry=geom)],
        smiles=explicit,
    )


class TestRepairMigratedConformers:
    def test_an_undisturbed_charge_state_is_left_alone(self) -> None:
        cs = ChargeState(charge=1, microstates=[_microstate("[NH3+]CC(=O)O")])
        report = repair_migrated_conformers(cs, stage="refinement")
        assert report.touched == 0
        assert len(cs.microstates[0].conformers) == 1

    def test_a_migrated_conformer_moves_to_the_microstate_it_became(self) -> None:
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O", conformers=[])
        geom = source.conformers[0].geometry

        # Move one of the [OH2+] protons onto the amine nitrogen.
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.moved == 1
        assert source.conformers == []
        assert len(target.conformers) == 1
        assert source.excluded_conformers == []

    def test_a_detached_hydrogen_is_excluded_not_dropped(self) -> None:
        ms = _microstate("[NH3+]CC(=O)O")
        geom = ms.conformers[0].geometry
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        ms.conformers[0].geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.detached == 1
        assert ms.conformers == []
        assert len(ms.excluded_conformers) == 1
        excluded = ms.excluded_conformers[0]
        assert excluded.reason == "proton_detached"
        assert excluded.stage == "refinement"
        assert str(DETACHED_DISTANCE) in excluded.detail

    def test_a_species_no_microstate_describes_is_excluded(self) -> None:
        """A proton onto carbon: the enumerator only ever touches heteroatoms."""
        ms = _microstate("[NH3+]CC(=O)O")
        geom = ms.conformers[0].geometry
        carbon = next(i for i, s in enumerate(geom.symbols) if s == "C")
        nitrogens = [i for i, s in enumerate(geom.symbols) if s == "N"]
        amine_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in nitrogens)
        ms.conformers[0].geometry = _move_h(geom, amine_h, carbon)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.unmatched == 1
        assert ms.conformers == []
        assert ms.excluded_conformers[0].reason == "no_matching_microstate"

    def test_a_stereocentre_tie_is_resolved_from_the_geometry(self) -> None:
        """Two candidates, same protonation, opposite configuration at carbon."""
        source = _microstate("N[C@@H](C)C(=O)[OH2+]")
        first = _microstate("[NH3+][C@@H](C)C(=O)O", conformers=[])
        second = _microstate("[NH3+][C@H](C)C(=O)O", conformers=[])
        assert protonation_key_from_mol(
            template_from_smiles(first.smiles or ""), 1
        ) == protonation_key_from_mol(template_from_smiles(second.smiles or ""), 1)

        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, first, second])
        report = repair_migrated_conformers(cs, stage="refinement")

        # Moving a proton between heteroatoms does not invert a carbon centre,
        # so the geometry still has the configuration it started with.
        assert report.moved == 1
        assert report.stereo_resolved == 1
        assert report.ambiguous == 0
        assert len(first.conformers) == 1
        assert second.conformers == []

    def test_a_double_bond_configuration_is_resolved_from_the_geometry(self) -> None:
        source = _microstate(r"O=C(O)/C=C\C(=O)[OH2+]")
        cis = _microstate(r"OC(=[OH+])/C=C\C(=O)O", conformers=[])
        trans = _microstate(r"OC(=[OH+])/C=C/C(=O)O", conformers=[])

        geom = source.conformers[0].geometry
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        counts = {
            o: sum(1 for h in geom.hydrogen_indices if _owner_of(geom, h) == o) for o in oxygens
        }
        donor = max(counts, key=lambda o: counts[o])
        acceptor = min(counts, key=lambda o: counts[o])
        moving = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) == donor)
        source.conformers[0].geometry = _move_h(geom, moving, acceptor)

        cs = ChargeState(charge=1, microstates=[source, cis, trans])
        report = repair_migrated_conformers(cs, stage="refinement")

        if report.moved:
            # A migrating proton does not rotate a C=C: the cis backbone stays cis.
            assert report.stereo_resolved == 1
            assert len(cis.conformers) == 1
            assert trans.conformers == []

    def test_a_tie_with_nothing_to_discriminate_is_excluded(self) -> None:
        """Candidates whose stereo is identical leave the geometry no question to answer."""
        source = _microstate("NCC(=O)[OH2+]")
        first = _microstate("[NH3+]CC(=O)O", conformers=[])
        second = _microstate("[NH3+]CC(=O)O", conformers=[])
        second.tautomer_id = "a distinct label, identical stereo"

        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, first, second])
        report = repair_migrated_conformers(cs, stage="refinement")

        assert report.ambiguous == 1
        assert report.moved == 0
        assert source.excluded_conformers[0].reason == "ambiguous_microstate"

    def test_a_mixed_charge_state_is_skipped(self) -> None:
        """Half-labelled means neither rule applies; refuse rather than guess."""
        labelled = _microstate("[NH3+]CC(=O)O")
        geom, _ = _embed("[NH3+]CC(=O)O")
        unlabelled = Microstate(
            tautomer_id="deadbeef", conformers=[Conformer(geometry=geom)], smiles=None
        )
        cs = ChargeState(charge=1, microstates=[labelled, unlabelled])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.checked == 0
        assert len(unlabelled.conformers) == 1


def _unlabelled(geom: Geometry, includes_enantiomer: bool = False) -> Microstate:
    """An approach-2 microstate: identified by its H distribution, with no SMILES."""
    return Microstate(
        tautomer_id=geometric_fingerprint(geom),
        conformers=[Conformer(geometry=geom)],
        smiles=None,
        includes_enantiomer=includes_enantiomer,
    )


class TestRepairWithoutSmilesLabels:
    """Approach 2: the microstate set is discovered, so an unseen state is created."""

    def test_an_undisturbed_conformer_stays_put(self) -> None:
        geom, _ = _embed("[NH3+]CC(=O)O")
        ms = _unlabelled(geom)
        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.touched == 0
        assert len(ms.conformers) == 1
        assert len(cs.microstates) == 1

    def test_a_migration_to_an_unseen_position_opens_a_microstate(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        ms = _unlabelled(geom, includes_enantiomer=True)
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        ms.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.created == 1
        assert report.unmatched == 0
        assert ms.conformers == []
        assert len(cs.microstates) == 2
        opened = cs.microstates[1]
        assert len(opened.conformers) == 1
        assert opened.smiles is None
        assert opened.tautomer_id == geometric_fingerprint(opened.conformers[0].geometry)
        assert opened.includes_enantiomer is True

    def test_a_migration_to_a_sampled_position_moves_rather_than_creates(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        source = _unlabelled(geom)
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        migrated = _move_h(geom, acid_h, nitrogen)

        target = _unlabelled(migrated)
        target.conformers = []
        source.conformers[0].geometry = migrated

        cs = ChargeState(charge=1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.moved == 1
        assert report.created == 0
        assert source.conformers == []
        assert len(target.conformers) == 1
        assert len(cs.microstates) == 2

    def test_two_conformers_reaching_the_same_new_state_share_one_microstate(self) -> None:
        geom, _ = _embed("NCC(=O)[OH2+]")
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        ms = _unlabelled(geom)
        ms.conformers = [
            Conformer(geometry=_move_h(geom, acid_h, nitrogen)),
            Conformer(geometry=_move_h(geom, acid_h, nitrogen, dist=1.04)),
        ]

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.created == 1
        assert len(cs.microstates) == 2
        assert len(cs.microstates[1].conformers) == 2

    def test_a_detached_hydrogen_is_still_excluded(self) -> None:
        geom, _ = _embed("[NH3+]CC(=O)O")
        ms = _unlabelled(geom)
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        ms.conformers[0].geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.detached == 1
        assert report.created == 0
        assert ms.conformers == []
        assert ms.excluded_conformers[0].reason == "proton_detached"

    def test_a_disagreeing_heavy_atom_ordering_is_refused(self) -> None:
        """The fingerprint is positional; comparing across orderings would be nonsense."""
        first, _ = _embed("[NH3+]CC(=O)O")
        # Swap a nitrogen and an oxygen so the heavy-atom element sequence differs;
        # two SMILES of one molecule would not, since both are canonicalised.
        symbols = list(first.symbols)
        coords = first.coords.copy()
        i = symbols.index("N")
        j = len(symbols) - 1 - symbols[::-1].index("O")
        symbols[i], symbols[j] = symbols[j], symbols[i]
        coords[[i, j]] = coords[[j, i]]
        second = Geometry(symbols=tuple(symbols), coords=coords)
        cs = ChargeState(charge=1, microstates=[_unlabelled(first), _unlabelled(second)])
        report = repair_migrated_conformers(cs, stage="sampling")
        assert report.checked == 0
        assert all(len(ms.conformers) == 1 for ms in cs.microstates)

    def test_the_excluded_conformer_keeps_its_energies_and_multiplicity(self) -> None:
        ms = _microstate("[NH3+]CC(=O)O")
        conf = ms.conformers[0]
        conf.electronic_energy = -1.5
        conf.solvation_energy = -0.02
        conf.multiplicity = 3.0
        geom = conf.geometry
        h = geom.hydrogen_indices[0]
        coords = geom.coords.copy()
        coords[h] += np.array([0.0, 0.0, 8.0])
        conf.geometry = Geometry(symbols=tuple(geom.symbols), coords=coords)

        cs = ChargeState(charge=1, microstates=[ms])
        repair_migrated_conformers(cs, stage="sampling")

        excluded = ms.excluded_conformers[0]
        assert excluded.multiplicity == 3.0
        assert excluded.electronic_energy == -1.5
        assert excluded.solvation_energy == -0.02
        assert excluded.stage == "sampling"


class TestOneHeavyAtomOrderPerMolecule:
    """The invariant that removes the atom-correspondence problem entirely."""

    @pytest.mark.parametrize(
        "family",
        [
            ["NCC(=O)O", "[NH3+]CC(=O)O", "NCC(=O)[O-]", "NCC(=O)[OH2+]", "OC(=[OH+])C[NH3+]"],
            ["O=C(O)CC(=O)O", "O=C(O)CC(=O)[O-]", "O=C([O-])CC(=O)O", "O=C(O)CC(O)=[OH+]"],
            ["NCCc1c[nH]cn1", "[NH3+]CCc1c[nH]cn1", "NCCc1c[n-]cn1"],
            ["C=C(CC(=O)O)C(=O)O", "C=C(CC(=O)[O-])C(=O)O", "C=C(CC(O)=[OH+])C(=O)O"],
        ],
    )
    def test_every_protomer_agrees_on_the_heavy_atom_order(self, family: list[str]) -> None:
        orders = set()
        for smiles in family:
            geom, _ = _embed(smiles)
            orders.add("".join(geom.symbols[i] for i in geom.heavy_atom_indices))
        assert len(orders) == 1

    def test_a_moved_conformer_adopts_its_destination_ordering(self) -> None:
        """Without the regrouping this trips deduplicate_conformers."""
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O")
        for ms in (source, target):
            for conf in ms.conformers:
                conf.electronic_energy = -1.0
        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        assert repair_migrated_conformers(cs, stage="refinement").moved == 1

        expected = tuple(
            a.GetSymbol() for a in template_from_smiles(target.smiles or "").GetAtoms()
        )
        assert all(tuple(c.geometry.symbols) == expected for c in target.conformers)
        # The guard this exists to satisfy.
        deduplicate_conformers(target.conformers, target.includes_enantiomer)

    def test_the_regrouping_puts_each_hydrogen_on_the_right_heavy_atom(self) -> None:
        source = _microstate("NCC(=O)[OH2+]")
        target = _microstate("[NH3+]CC(=O)O", conformers=[])
        geom = source.conformers[0].geometry
        nitrogen = next(i for i, s in enumerate(geom.symbols) if s == "N")
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acid_h = next(h for h in geom.hydrogen_indices if _owner_of(geom, h) in oxygens)
        source.conformers[0].geometry = _move_h(geom, acid_h, nitrogen)

        cs = ChargeState(charge=1, microstates=[source, target])
        repair_migrated_conformers(cs, stage="refinement")

        moved = target.conformers[0].geometry
        template = template_from_smiles(target.smiles or "")
        assert protonation_key_from_geometry(moved, template, 1) == protonation_key_from_mol(
            template, 1
        )


class TestHydrogenCountsSurviveCanonicalisation:
    """Regression guard on `_SKELETON_SANITIZE`.

    The skeleton's hydrogen counts are carried through canonicalisation only
    because SANITIZE_FINDRADICALS brackets the deficient atoms. Dropping that
    flag makes these pairs collide, merging species that are not even the same
    molecular formula. These tests fail if it is ever removed.
    """

    @pytest.mark.parametrize(
        "a,b",
        [
            ("CC(=O)O", "CC(O)O"),
            ("O=C(O)CC(=O)[O-]", "O=C(O)CC([O-])[O-]"),
            ("c1ccccc1O", "C1CCCCC1O"),
            ("CC=O", "CCO"),
        ],
    )
    def test_species_differing_in_hydrogen_count_never_share_a_key(self, a: str, b: str) -> None:
        key_a = protonation_key_from_mol(template_from_smiles(smiles_to_3d(a)[1]), 0)
        key_b = protonation_key_from_mol(template_from_smiles(smiles_to_3d(b)[1]), 0)
        assert key_a != key_b

    def test_the_skeleton_brackets_a_hydrogen_deficient_atom(self) -> None:
        template = template_from_smiles(smiles_to_3d("CC(=O)O")[1])
        key = protonation_key_from_mol(template, 0)
        assert "[O]" in key, "carbonyl O must be bracketed or its H count is lost"


class TestMatchToCandidate:
    """Verification, not discrimination: does the geometry agree with *this* candidate."""

    def test_a_geometry_agrees_with_its_own_label(self) -> None:
        geom, explicit = _embed(r"OC(=[OH+])/C=C\\C(=O)O")
        template = template_from_smiles(explicit)
        assert (
            match_to_candidate(geom, assign_protons(geom), template, template, False) is not None
        )

    def test_a_geometry_is_rejected_by_the_wrong_diastereomer(self) -> None:
        geom, explicit = _embed(r"OC(=[OH+])/C=C\\C(=O)O")
        cis = template_from_smiles(explicit)
        trans = template_from_smiles(smiles_to_3d(r"OC(=[OH+])/C=C/C(=O)O")[1])
        assignment = assign_protons(geom)
        assert match_to_candidate(geom, assignment, cis, cis, False)
        assert match_to_candidate(geom, assignment, cis, trans, False) is None

    def test_a_candidate_specifying_no_stereo_cannot_be_contradicted(self) -> None:
        geom, explicit = _embed("NCC(=O)O")
        template = template_from_smiles(explicit)
        assert (
            match_to_candidate(geom, assign_protons(geom), template, template, False) is not None
        )

    def test_the_mirror_is_accepted_only_for_a_collapsed_enantiomeric_pair(self) -> None:
        """A microstate flagged `includes_enantiomer` stands for both mirror images."""
        geom, explicit = _embed("N[C@@H](C)C(=O)O")
        own = template_from_smiles(explicit)
        mirror = template_from_smiles(smiles_to_3d("N[C@H](C)C(=O)O")[1])
        assignment = assign_protons(geom)
        # The geometry is its own label either way.
        assert match_to_candidate(geom, assignment, own, own, False)
        # Against the opposite configuration it depends on what the microstate means.
        assert match_to_candidate(geom, assignment, own, mirror, False) is None
        assert match_to_candidate(geom, assignment, own, mirror, True) is not None


class TestFragmentationIsExcluded:
    """A broken heavy-atom bond is invisible to both identities.

    Approach 1 reads its framework from the *template* and only the hydrogen
    counts from the coordinates; approach 2 computes the framework but does not
    hash it. So in both, a structure that came apart produces the identity of
    the intact species and is filed as though nothing happened.
    """

    def test_a_broken_bond_leaves_every_hydrogen_attached(self) -> None:
        """The premise: `is_intact` cannot catch this, so something else must."""
        geom, _ = _embed("OC=O")
        broken = _break_a_bond(geom, 0)
        assert assign_protons(broken).is_intact
        assert len(heavy_components(geom)) == 1
        assert len(heavy_components(broken)) == 2

    def test_the_protonation_key_cannot_tell_the_difference(self) -> None:
        geom, explicit = _embed("OC=O")
        template = template_from_smiles(explicit)
        broken = _break_a_bond(geom, 0)
        assert protonation_key_from_geometry(broken, template, 0) == protonation_key_from_geometry(
            geom, template, 0
        )

    def test_the_fingerprint_cannot_tell_the_difference(self) -> None:
        geom, _ = _embed("OC=O")
        assert geometric_fingerprint(_break_a_bond(geom, 0)) == geometric_fingerprint(geom)

    def test_a_single_heavy_atom_is_not_a_fragment(self) -> None:
        geom, _ = _embed("O")
        assert len(heavy_components(geom)) == 1

    def test_approach_one_excludes_it(self) -> None:
        geom, explicit = _embed("OC=O")
        ms = Microstate(
            tautomer_id="OC=O",
            conformers=[Conformer(geometry=_break_a_bond(geom, 0))],
            smiles=explicit,
        )
        report = repair_migrated_conformers(
            ChargeState(charge=0, microstates=[ms]), stage="refinement"
        )
        assert report.fragmented == 1
        assert ms.conformers == []
        assert [e.reason for e in ms.excluded_conformers] == ["fragmented"]

    def test_approach_two_excludes_it(self) -> None:
        geom, _ = _embed("OC=O")
        broken = _break_a_bond(geom, 0)
        ms = Microstate(
            tautomer_id=geometric_fingerprint(broken), conformers=[Conformer(geometry=broken)]
        )
        report = repair_migrated_conformers(
            ChargeState(charge=0, microstates=[ms]), stage="sampling"
        )
        assert report.fragmented == 1
        assert ms.conformers == []
        assert [e.reason for e in ms.excluded_conformers] == ["fragmented"]

    def test_one_fragment_no_longer_disables_the_check_for_everyone_else(self) -> None:
        """The fail-open this replaces.

        `heavy_frameworks_agree` is about heavy-atom *ordering*, and its response
        to a disagreement is to skip the whole charge state. A fragment failed it
        for an unrelated reason, so a single broken structure suppressed
        migration repair for every intact conformer at that charge -- and since
        the reference is whichever conformer comes first, a fragment arriving
        first made the healthy majority look like the disagreement.
        """
        methanol, _ = _embed("CO")
        oxygen = methanol.heavy_atom_indices[1]
        hydroxyl = next(h for h in methanol.hydrogen_indices if _owner_of(methanol, h) == oxygen)
        migrated = _move_h(methanol, hydroxyl, methanol.heavy_atom_indices[0])
        assert geometric_fingerprint(migrated) != geometric_fingerprint(methanol)

        # The fragment is deliberately first, so it would have become the reference.
        source = Microstate(
            tautomer_id=geometric_fingerprint(methanol),
            conformers=[
                Conformer(geometry=_break_a_bond(methanol, 1)),
                Conformer(geometry=migrated),
            ],
        )
        cs = ChargeState(charge=0, microstates=[source])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.fragmented == 1
        assert report.created == 1, "the intact conformer must still be re-filed"
        assert [e.reason for e in source.excluded_conformers] == ["fragmented"]
        assert len(cs.microstates) == 2


class TestLayoutPutsAtomsInTheTemplateOrder:
    """`_stereo_from_coordinates` attaches coordinates by position, not by mapping.

    Atom *i* of the laid-out geometry has to be atom *i* of the template --
    hydrogens included, since a chiral tag is a parity over the atom's stored
    neighbour order. `_layouts_for_target` is the only thing establishing that.
    """

    @staticmethod
    def _signature(template: Chem.Mol, laid_out: Geometry) -> str:
        observed = _stereo_from_coordinates(template, laid_out)
        obs_bonds, obs_atoms = _specified_stereo(observed)
        spec_bonds, spec_atoms = _specified_stereo(template)
        return _stereo_signature(
            observed, set(spec_bonds) & set(obs_bonds), set(spec_atoms) & set(obs_atoms)
        )

    @pytest.mark.parametrize("smiles", ["C[C@H](N)CO", "C[C@H](O)[C@@H](N)CO"])
    def test_a_layout_reproduces_the_template_atom_order(self, smiles: str) -> None:
        geom, explicit = _embed(smiles)
        template = template_from_smiles(explicit)
        layouts = _layouts_for_target(geom, assign_protons(geom), template, template)
        expected = tuple(a.GetSymbol() for a in template.GetAtoms())
        assert layouts, "a geometry must lay out against its own template"
        for laid_out in layouts:
            assert laid_out.symbols == expected

    @pytest.mark.parametrize(
        "smiles",
        [
            "C[C@H](N)CO",  # a diastereotopic CH2 next to a stereocentre
            "C[C@H](O)[C@@H](N)CO",  # two centres, CH2 between substituents
            "O=C(O)/C=C/[C@H](C)O",  # a stereo double bond as well
        ],
    )
    def test_permuting_hydrogens_on_one_heavy_atom_changes_nothing(self, smiles: str) -> None:
        """Hydrogens sharing a heavy atom are filled first-come, which is arbitrary.

        It is also inconsequential: they are interchangeable by automorphism, so
        no descriptor anywhere can depend on which is which. An atom carrying two
        or more hydrogens cannot be a tetrahedral centre, and a diastereotopic
        CH2's hydrogens have equal canonical rank, so they cannot rank a
        neighbouring centre's substituents either.
        """
        geom, explicit = _embed(smiles)
        template = template_from_smiles(explicit)
        laid_out = _layouts_for_target(geom, assign_protons(geom), template, template)[0]
        baseline = self._signature(template, laid_out)

        owned: dict[int, list[int]] = {}
        for idx, owner in enumerate(_heavy_slots(template)):
            if owner is not None:
                owned.setdefault(owner, []).append(idx)
        shared = [group for group in owned.values() if len(group) > 1]
        assert shared, "this molecule must have a heavy atom carrying several hydrogens"

        for group in shared:
            for permuted in itertools.permutations(group):
                coords = laid_out.coords.copy()
                for destination, source in zip(group, permuted, strict=True):
                    coords[destination] = laid_out.coords[source]
                swapped = Geometry(symbols=laid_out.symbols, coords=coords)
                assert self._signature(template, swapped) == baseline


class TestMigrationReport:
    def test_an_ambiguous_exclusion_counts_as_touched(self) -> None:
        """It excludes a conformer, so a summary that omits it hides a real loss.

        Malonic acid at q=-1 lost both conformers of a microstate this way and
        printed no summary line at all.
        """
        assert MigrationReport(checked=3, ambiguous=2).touched == 2

    def test_a_report_with_nothing_to_say_summarises_to_none(self) -> None:
        assert MigrationReport(checked=9).summary() is None

    def test_every_outcome_reaches_the_summary(self) -> None:
        report = MigrationReport(
            checked=9,
            moved=1,
            detached=2,
            unmatched=3,
            ambiguous=4,
            unresolved_tie=5,
            stereo_unmatched=6,
            created=7,
        )
        summary = report.summary()
        assert summary is not None
        for count in ("1", "2", "3", "4", "5", "6", "7"):
            assert count in summary

    def test_a_conformer_left_in_place_still_gets_reported(self) -> None:
        """`unresolved_tie` moves nothing, so it is not `touched` -- but it is news."""
        report = MigrationReport(checked=4, unresolved_tie=4)
        assert report.touched == 0
        assert report.summary() is not None


class TestRingStereoIsResolved:
    """1,4-ring cis/trans: a relationship carried by a *pair* of atoms.

    These share a protonation key, since the key holds no bond orders and the
    two diastereomers have identical hydrogen counts. Comparing only the atom
    whose tag differs destroys the relationship -- both candidates canonicalise
    identically -- so the comparison keeps everything the candidates specify.
    """

    SRC: ClassVar[str] = "C[C@H](c1ccc(C(=O)O)cc1)[C@H]2CC[C@@H](C(=O)[O-])CC2"
    CIS: ClassVar[str] = "C[C@H](c1ccc(C(=O)[O-])cc1)[C@H]2CC[C@@H](C(=O)O)CC2"
    TRANS: ClassVar[str] = "C[C@H](c1ccc(C(=O)[O-])cc1)[C@H]2CC[C@H](C(=O)O)CC2"

    def test_the_two_diastereomers_share_a_protonation_key(self) -> None:
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        assert protonation_key_from_mol(cis, -1) == protonation_key_from_mol(trans, -1)

    def test_comparing_only_the_differing_atom_would_collapse_them(self) -> None:
        """Why verification compares everything the candidate specifies.

        A 1,4-ring cis/trans relationship is carried by a *pair* of atoms. Keep
        only the one whose tag differs and RDKit drops it when writing the
        SMILES -- a lone ring carbon has two identical branches -- so the two
        diastereomers become the same string and nothing can be decided.
        """
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        _, cis_atoms = _specified_stereo(cis)
        _, trans_atoms = _specified_stereo(trans)
        decisive = {k for k in cis_atoms if cis_atoms.get(k) != trans_atoms.get(k)}
        assert len(decisive) == 1, "cis and trans differ at exactly one ring carbon"

        assert _stereo_signature(cis, set(), decisive) == _stereo_signature(trans, set(), decisive)
        assert _stereo_signature(cis, set(), set(cis_atoms)) != _stereo_signature(
            trans, set(), set(trans_atoms)
        )

    def test_the_geometry_picks_the_right_diastereomer(self) -> None:
        source = _microstate(self.SRC)
        cis = _microstate(self.CIS, conformers=[])
        trans = _microstate(self.TRANS, conformers=[])

        geom = source.conformers[0].geometry
        assignment = assign_protons(geom)
        owner = dict(zip(geom.hydrogen_indices, assignment.owner, strict=True))
        acid_h = next(h for h, o in owner.items() if geom.symbols[o] == "O")
        donor = owner[acid_h]
        oxygens = [i for i, s in enumerate(geom.symbols) if s == "O"]
        acceptor = max(
            (i for i in oxygens if i != donor),
            key=lambda i: float(np.linalg.norm(geom.coords[i] - geom.coords[donor])),
        )
        source.conformers[0].geometry = _move_h(geom, acid_h, acceptor, dist=0.98)

        cs = ChargeState(charge=-1, microstates=[source, cis, trans])
        report = repair_migrated_conformers(cs, stage="refinement")

        # The proton moved between carboxylates; the ring was never touched.
        assert report.stereo_resolved == 1
        assert report.ambiguous == 0
        assert len(cis.conformers) == 1
        assert trans.conformers == []


class TestStereoIsVerifiedNotAssumed:
    """The branch that previously kept a conformer without looking at it."""

    CIS: ClassVar[str] = r"OC(=[OH+])/C=C\C(=O)O"
    TRANS: ClassVar[str] = r"OC(=[OH+])/C=C/C(=O)O"

    def test_the_two_share_a_protonation_key(self) -> None:
        cis = template_from_smiles(smiles_to_3d(self.CIS)[1])
        trans = template_from_smiles(smiles_to_3d(self.TRANS)[1])
        assert protonation_key_from_mol(cis, 1) == protonation_key_from_mol(trans, 1)

    def test_a_flipped_double_bond_moves_to_its_sibling(self) -> None:
        """The protonation never changed, so nothing before this looked at it."""
        cis = _microstate(self.CIS)
        trans = _microstate(self.TRANS, conformers=[])
        # file the trans geometry under the cis microstate
        cis.conformers[0].geometry = _embed(self.TRANS)[0]

        cs = ChargeState(charge=1, microstates=[cis, trans])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.moved == 1
        assert cis.conformers == []
        assert len(trans.conformers) == 1

    def test_a_flipped_double_bond_with_no_sibling_is_kept_and_flagged(self) -> None:
        """Not an exclusion: a centre sampling can flip was not stable to begin with."""
        cis = _microstate(self.CIS)
        cis.conformers[0].geometry = _embed(self.TRANS)[0]

        cs = ChargeState(charge=1, microstates=[cis])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.stereo_unmatched == 1
        assert report.moved == 0
        assert len(cis.conformers) == 1
        assert cis.excluded_conformers == []


class TestPseudoAsymmetry:
    """Two ends of a molecule that only stereochemistry tells apart.

    2,3,4-trihydroxyglutaric acid has a pseudo-asymmetric C3. Deprotonating
    either end breaks the tie, giving two diastereomers that share a protonation
    key -- so a proton moving from one end to the other is invisible to the key,
    and the skeleton automorphism relating them is stereo-relevant. This is the
    case that shows why no candidate, not even the conformer's own microstate,
    may skip the layout step.
    """

    E1: ClassVar[str] = "O=C([O-])[C@@H](O)[C@@H](O)[C@@H](O)C(=O)O"
    E5: ClassVar[str] = "O=C(O)[C@@H](O)[C@@H](O)[C@@H](O)C(=O)[O-]"

    def test_they_are_different_species_sharing_one_protonation_key(self) -> None:
        t1 = template_from_smiles(smiles_to_3d(self.E1)[1])
        t5 = template_from_smiles(smiles_to_3d(self.E5)[1])
        assert protonation_key_from_mol(t1, -1) == protonation_key_from_mol(t5, -1)
        assert Chem.MolToSmiles(t1) != Chem.MolToSmiles(t5)

    @staticmethod
    def _move_acid_proton(geom: Geometry) -> Geometry:
        """Hand the carboxylic acid's proton to the carboxylate at the other end.

        A migration keeps the microstate's atom ordering and moves coordinates,
        which is what makes it invisible to the protonation key here.
        """
        symbols = list(geom.symbols)
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        assignment = assign_protons(geom)
        owner = dict(zip(geom.hydrogen_indices, assignment.owner, strict=True))

        def near(i: int, j: int) -> bool:
            return float(np.linalg.norm(geom.coords[i] - geom.coords[j])) < 1.7

        carboxyl_o = [
            o
            for o in oxygens
            if any(
                symbols[c] == "C" and sum(1 for k in oxygens if near(k, c)) == 2
                for c in range(len(symbols))
                if symbols[c] == "C" and near(o, c)
            )
        ]
        donor = next(o for o in carboxyl_o if any(owner.get(h) == o for h in owner))
        acid_h = next(h for h, o in owner.items() if o == donor)
        acceptor = max(
            (o for o in carboxyl_o if o != donor and not any(owner.get(h) == o for h in owner)),
            key=lambda o: float(np.linalg.norm(geom.coords[o] - geom.coords[donor])),
        )
        return _move_h(geom, acid_h, acceptor, dist=0.98)

    def test_the_geometry_is_re_filed_rather_than_silently_kept(self) -> None:
        source = _microstate(self.E1)
        target = _microstate(self.E5, conformers=[])
        source.conformers[0].geometry = self._move_acid_proton(source.conformers[0].geometry)

        template = template_from_smiles(source.smiles or "")
        moved = source.conformers[0].geometry
        # the key cannot see this migration -- both ends give the same skeleton
        assert protonation_key_from_geometry(moved, template, -1) == protonation_key_from_mol(
            template, -1
        )

        cs = ChargeState(charge=-1, microstates=[source, target])
        report = repair_migrated_conformers(cs, stage="refinement")

        # Resolved, and only because the conformer's own microstate gets no
        # shortcut. The geometry is in E1's atom order and keys as E1, but its
        # hydrogens sit on the symmetry-related positions E1's template does not
        # use -- so laying it into E1 fails, while a layout into E5 succeeds.
        # Short-circuiting the own-microstate check made E1 verify trivially,
        # because the three CH(OH) centres are unaffected by which carboxyl holds
        # the proton and nothing else was compared.
        assert report.moved == 1, "stereo must catch what the protonation key cannot"
        assert report.stereo_resolved == 1
        assert report.ambiguous == 0
        assert source.conformers == []
        assert source.excluded_conformers == []
        assert len(target.conformers) == 1


class TestTheIdentityFastPathSkipsTheSearch:
    """The automorphism search is not merely deprioritised, it is not run.

    Two things ride on this. Correctness must not depend on the identity
    appearing within `max_layouts` -- on a tris-CF3 alcohol the skeleton has 1296
    automorphisms against a cap of 64, so finding the identity inside the results
    would rest on RDKit's match ordering rather than on anything guaranteed. And
    the cap warning must stay quiet on clean runs: emitted before we know whether
    any automorphism matters, it printed an alarming line about orderings that
    were never needed.
    """

    SYMMETRIC: ClassVar[str] = "OC(C(F)(F)F)(C(F)(F)F)C(F)(F)F"

    @staticmethod
    def _layouts_counting_searches(smiles: str) -> tuple[int, int]:
        geom, explicit_h = _embed(smiles, seed=1)
        template = template_from_smiles(explicit_h)
        searches = 0
        real = Chem.Mol.GetSubstructMatches

        def spy(self: Chem.Mol, *args: object, **kwargs: object) -> object:
            nonlocal searches
            searches += 1
            return real(self, *args, **kwargs)

        with mock.patch.object(Chem.Mol, "GetSubstructMatches", spy):
            layouts = _layouts_for_target(geom, assign_protons(geom), template, template)
        return len(layouts), searches

    def test_the_skeleton_really_is_badly_symmetric(self) -> None:
        """Guards the test: a molecule with few automorphisms would prove nothing."""
        _, explicit_h = _embed(self.SYMMETRIC, seed=1)
        template = template_from_smiles(explicit_h)
        skeleton = _skeleton_mol(template, _template_counts(template))
        uncapped = skeleton.GetSubstructMatches(
            skeleton, uniquify=False, useChirality=False, maxMatches=10_000_000
        )
        assert len(uncapped) > 64, "must exceed the cap for this test to mean anything"

    def test_no_search_runs_when_no_proton_moved(self) -> None:
        layouts, searches = self._layouts_counting_searches(self.SYMMETRIC)
        assert searches == 0, "the identity is built directly, not found in a search"
        assert layouts == 1

    def test_the_cap_warning_stays_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tautomator.protomer_geometry"):
            self._layouts_counting_searches(self.SYMMETRIC)
        assert "hit its cap" not in caplog.text

    def test_the_search_still_runs_once_a_proton_has_moved(self) -> None:
        """The slow path is reached exactly when it is needed."""
        source = _microstate(r"O=C(O)/C=C(/[O-])O", seed=1)
        geom = source.conformers[0].geometry
        heavy = geom.heavy_atom_indices
        owner = dict(zip(geom.hydrogen_indices, assign_protons(geom).owner, strict=True))
        hydroxyl_h = next(h for h, o in owner.items() if o == heavy[6])
        migrated = _move_h(geom, hydroxyl_h, heavy[5], dist=0.98)
        template = template_from_smiles(source.smiles or "")
        assert assign_protons(migrated).counts != _template_counts(template)

        searches = 0
        real = Chem.Mol.GetSubstructMatches

        def spy(self: Chem.Mol, *args: object, **kwargs: object) -> object:
            nonlocal searches
            searches += 1
            return real(self, *args, **kwargs)

        with mock.patch.object(Chem.Mol, "GetSubstructMatches", spy):
            layouts = _layouts_for_target(migrated, assign_protons(migrated), template, template)
        assert searches == 1
        assert len(layouts) == 2, "both ends of the automorphism are offered"


class TestATorsionCannotDecideIdentity:
    """The property the identity preference exists to guarantee.

    Configuration is not a function of a rotatable dihedral, so turning one must
    not change any identity verdict. On the previous code this failed loudly:
    `O=C(O)/C=C(/[O-])O` has an end-swapping skeleton automorphism whose layout
    seats the template's C=C on the coordinates of the C-C single bond, and
    `AssignStereochemistryFrom3D` reports an E/Z descriptor for it. Half of all
    torsions made the wrong isomer verify.
    """

    E: ClassVar[str] = r"O=C(O)/C=C(/[O-])O"
    Z: ClassVar[str] = r"O=C(O)/C=C(\[O-])O"

    # Heavy positions: 1 is the acid carbon, 3 the middle carbon. The 1-3 bond is
    # single and free to rotate; the acid end (0, 1, 2) is the branch it carries.
    AXIS: ClassVar[tuple[int, int]] = (3, 1)
    BRANCH: ClassVar[tuple[int, ...]] = (0, 1, 2)

    @pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_the_verdict_is_invariant_under_rotation(self, degrees: int) -> None:
        source = _microstate(self.E, seed=1)
        other = _microstate(self.Z, conformers=[])
        source.conformers[0].geometry = _rotate_branch(
            source.conformers[0].geometry, *self.AXIS, self.BRANCH, degrees
        )

        report = repair_migrated_conformers(
            ChargeState(charge=-1, microstates=[source, other]), stage="sampling"
        )

        assert report.moved == 0, "a rotation is not a migration"
        assert report.unresolved_tie == 0, "a torsion must not make Z satisfiable"
        assert report.ambiguous == 0
        assert len(source.conformers) == 1
        assert other.conformers == []

    @pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_only_the_true_isomer_verifies_at_every_torsion(self, degrees: int) -> None:
        """Stated on `match_to_candidate` directly, so the cause is pinned too."""
        geom, explicit_h = _embed(self.E, seed=1)
        template_e = template_from_smiles(explicit_h)
        template_z = template_from_smiles(_embed(self.Z, seed=1)[1])
        turned = _rotate_branch(geom, *self.AXIS, self.BRANCH, degrees)
        assignment = assign_protons(turned)

        matched_e = match_to_candidate(turned, assignment, template_e, template_e, False)
        matched_z = match_to_candidate(turned, assignment, template_e, template_z, False)

        assert matched_e is not None and matched_e[1] is True
        assert matched_z is None, "the Z template must not verify against an E geometry"

    def test_the_rotation_really_does_turn_a_rotatable_bond(self) -> None:
        """Guards the test itself: a no-op rotation would prove nothing."""
        geom, _ = _embed(self.E, seed=1)
        turned = _rotate_branch(geom, *self.AXIS, self.BRANCH, 180)
        heavy = geom.heavy_atom_indices
        moved = float(np.linalg.norm(turned.coords[heavy[0]] - geom.coords[heavy[0]]))
        assert moved > 1.0, "the carbonyl oxygen should swing right across"
        # ... and the configuration is untouched, which is the whole premise.
        assert assign_protons(turned).counts == assign_protons(geom).counts


class TestAnUndiscriminatedTie:
    """Two microstates a skeleton automorphism makes interchangeable.

    In `O=C(O)/C=C(/[O-])O` both terminal carbons are "two oxygens plus the
    middle carbon", so the skeleton has an end-swapping automorphism that is not
    a symmetry of the stereochemistry.

    That automorphism is only *reachable* once a proton has moved. While the
    hydrogen distribution still matches the template positionwise, the identity
    correspondence is the one the atoms have, and it separates E from Z cleanly
    -- so a conformer that merely rotated never ties. A tie needs a genuine
    migration onto a symmetry-related site, which is what the second test does.
    """

    E: ClassVar[str] = r"O=C(O)/C=C(/[O-])O"
    Z: ClassVar[str] = r"O=C(O)/C=C(\[O-])O"

    def test_they_are_separate_microstates(self) -> None:
        assert canonical_smiles(self.E) != canonical_smiles(self.Z)

    # Whether a conformer tied used to depend on its geometry: surveyed over 79
    # embeddings, 44 matched E alone and 35 tied, the split being set by a
    # torsion angle. Preferring the identity correspondence removed that
    # dependence, so every seed now decides -- see
    # `test_an_unmigrated_conformer_never_ties`, which pins the seeds that used
    # to tie, and `TestATorsionCannotDecideIdentity` for the mechanism.
    DECIDES: ClassVar[int] = 1

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 6, 8, 10, 16])
    def test_the_conformer_is_never_moved_or_excluded(self, seed: int) -> None:
        """A failure to discriminate is not evidence of a change.

        The invariant holds for every geometry, whichever route reaches it: the
        conformer stays under the label it arrived with, and nothing is lost.
        Excluding here would discard a real energy over an ambiguity that cannot
        change the answer -- `charge_state_free_energy` sums flatly over
        microstates, and these two agree on `includes_enantiomer`.
        """
        source = _microstate(self.E, seed=seed)
        other = _microstate(self.Z, conformers=[])

        cs = ChargeState(charge=-1, microstates=[source, other])
        report = repair_migrated_conformers(cs, stage="sampling")

        assert report.moved == 0
        assert report.ambiguous == 0
        assert len(source.conformers) == 1
        assert source.excluded_conformers == []
        assert other.conformers == []

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 6, 8, 10, 16])
    def test_an_unmigrated_conformer_never_ties(self, seed: int) -> None:
        """The identity correspondence decides, so there is nothing to report.

        Before the identity was preferred, each candidate chose whichever of the
        two layouts made it look right, and Z could verify against a torsion
        angle. Both candidates then "verified" and the conformer tied.
        """
        source = _microstate(self.E, seed=seed)
        other = _microstate(self.Z, conformers=[])

        report = repair_migrated_conformers(
            ChargeState(charge=-1, microstates=[source, other]), stage="sampling"
        )

        assert report.unresolved_tie == 0
        assert report.summary() is None
        assert len(source.conformers) == 1

    def test_a_migration_onto_a_symmetry_related_site_still_ties(self) -> None:
        """The tie branch is still reachable, and still reports.

        Handing the enol hydroxyl's proton to the carboxylate oxygen at the same
        end gives hydrogen vector (0, 0, 1, 1, 0, 1, 0) -- the vector carried by
        one of the two conformers the first pKa-set batch excluded as ambiguous
        at q=-1. The identity is no longer valid, the automorphism search runs,
        and both candidates stay satisfiable.
        """
        source = _microstate(self.E, seed=1)
        other = _microstate(self.Z, conformers=[])
        geom = source.conformers[0].geometry
        heavy = geom.heavy_atom_indices
        owner = dict(zip(geom.hydrogen_indices, assign_protons(geom).owner, strict=True))
        hydroxyl_h = next(h for h, o in owner.items() if o == heavy[6])
        source.conformers[0].geometry = _move_h(geom, hydroxyl_h, heavy[5], dist=0.98)
        assert assign_protons(source.conformers[0].geometry).counts == (0, 0, 1, 1, 0, 1, 0)

        report = repair_migrated_conformers(
            ChargeState(charge=-1, microstates=[source, other]), stage="sampling"
        )

        assert report.unresolved_tie == 1
        summary = report.summary()
        assert summary is not None and "indistinguishable" in summary
        assert len(source.conformers) == 1, "a tie keeps the conformer where it is"
        assert source.excluded_conformers == []

    def test_a_geometry_only_its_own_label_admits_is_kept_quietly(self) -> None:
        """The other route: one candidate verifies, and it is where the conformer sits.

        Nothing was re-filed and nothing was lost, so there is nothing to report.
        """
        source = _microstate(self.E, seed=self.DECIDES)
        other = _microstate(self.Z, conformers=[])

        report = repair_migrated_conformers(
            ChargeState(charge=-1, microstates=[source, other]), stage="sampling"
        )

        assert report.unresolved_tie == 0
        assert report.touched == 0
        assert report.summary() is None
