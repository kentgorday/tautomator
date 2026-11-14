"""Tests for symmetry-aware deduplication and partition-function weights."""

import logging
from typing import ClassVar

import numpy as np
import pytest

from tautomator.conformer_symmetry import (
    DEFAULT_RTHR,
    MAX_MAPPINGS,
    _candidate_mappings,
    best_max_distance,
    best_rmsd,
    conformer_multiplicity,
    deduplicate,
    effective_energy_offset,
    symmetry_number,
    symmetry_operations,
)
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.sampling import _filter_by_energy_window
from tautomator.types import Conformer, Geometry

# Staggered methylammonium, C3v, from a wB97M-V/def2-QZVPPD optimization.
# Atom order is H C H H N H H H: methyl hydrogens 0/2/3, ammonium 5/6/7.
METHYLAMMONIUM_SYMBOLS = ["H", "C", "H", "H", "N", "H", "H", "H"]
METHYLAMMONIUM = np.array(
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


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _water(twist: float = 0.0) -> tuple[np.ndarray, list[str]]:
    """A CH2 fragment stand-in: three atoms, one of which can be moved."""
    return (
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93 + twist, 0.0]]),
        ["O", "H", "H"],
    )


class TestSymmetryNumber:
    def test_methylammonium_is_c3v(self) -> None:
        """sigma = 3 from the two C3 turns plus the identity; achiral by its mirror planes."""
        sigma, achiral = symmetry_number(METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS)
        assert sigma == 3
        assert achiral is True

    def test_single_atom(self) -> None:
        sigma, achiral = symmetry_number(np.zeros((1, 3)), ["H"])
        assert sigma == 1
        assert achiral is True

    def test_sigma_is_a_property_of_the_conformer(self) -> None:
        """Twisting one hydrogen out of position destroys the two-fold axis."""
        sym, syms = _water()
        assert symmetry_number(sym, syms)[0] == 2
        skew, syms = _water(twist=0.6)
        assert symmetry_number(skew, syms)[0] == 1


class TestBestRmsd:
    def test_relabelled_and_rotated_copy_is_found(self) -> None:
        """Cycling both tripods together is a C3 turn: the same structure, relabelled."""
        rng = np.random.default_rng(0)
        perm = np.array([2, 1, 3, 0, 4, 6, 7, 5])
        moved = METHYLAMMONIUM[perm] @ _rand_rotation(rng).T + rng.normal(size=3) * 5
        assert best_rmsd(METHYLAMMONIUM, moved, METHYLAMMONIUM_SYMBOLS) < 1e-4

    def test_mirror_image_is_not_a_proper_match(self) -> None:
        """Reflection is excluded: an enantiomeric conformer is a separate state."""
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        symbols = ["C", "F", "Cl", "Br"]
        mirrored = chiral * np.array([1.0, 1.0, -1.0])
        assert best_rmsd(chiral, mirrored, symbols) > DEFAULT_RTHR
        assert best_rmsd(chiral, mirrored, symbols, mirror=True) < 1e-6

    def test_different_structures_do_not_match(self) -> None:
        a, syms = _water()
        b, _ = _water(twist=1.5)
        assert best_rmsd(a, b, syms) > DEFAULT_RTHR


class TestBestMaxDistance:
    def test_relabelled_and_rotated_copy_is_found(self) -> None:
        rng = np.random.default_rng(0)
        perm = np.array([2, 1, 3, 0, 4, 6, 7, 5])
        moved = METHYLAMMONIUM[perm] @ _rand_rotation(rng).T + rng.normal(size=3) * 5
        assert best_max_distance(METHYLAMMONIUM, moved, METHYLAMMONIUM_SYMBOLS) < 1e-4

    def test_scores_the_farthest_atom(self) -> None:
        # One ammonium hydrogen moved 0.3 A: the superposition shares the move out
        # a little, and the RMSD dilutes it over all eight atoms.
        moved = METHYLAMMONIUM.copy()
        moved[5] += [0.0, 0.0, 0.3]
        farthest = best_max_distance(METHYLAMMONIUM, moved, METHYLAMMONIUM_SYMBOLS)
        assert 0.2 < farthest <= 0.3
        assert best_rmsd(METHYLAMMONIUM, moved, METHYLAMMONIUM_SYMBOLS) < farthest / 2

    def test_mirror_image_only_on_request(self) -> None:
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        symbols = ["C", "F", "Cl", "Br"]
        mirrored = chiral * np.array([1.0, 1.0, -1.0])
        assert best_max_distance(chiral, mirrored, symbols) > DEFAULT_RTHR
        assert best_max_distance(chiral, mirrored, symbols, mirror=True) < 1e-6


class TestSymmetryOperations:
    def test_water_has_its_three(self) -> None:
        coords, syms = _water()
        assert len(symmetry_operations(coords, syms)) == 3  # C2 and two mirrors

    def test_methylammonium_has_its_five(self) -> None:
        ops = symmetry_operations(METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS)
        assert sorted(bool(np.linalg.det(r) < 0) for _, r in ops) == [False] * 2 + [True] * 3

    def test_a_chiral_centre_has_none(self) -> None:
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        assert symmetry_operations(chiral, ["C", "F", "Cl", "Br"]) == []

    def test_each_operation_maps_the_structure_onto_itself(self) -> None:
        x = METHYLAMMONIUM - METHYLAMMONIUM.mean(axis=0)
        for perm, rot in symmetry_operations(METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS):
            assert np.abs(x[perm] @ rot.T - x).max() < DEFAULT_RTHR


class TestDeduplicate:
    def test_empty_and_single(self) -> None:
        assert deduplicate(np.zeros((0, 3, 3)), ["O", "H", "H"]) == []
        assert deduplicate(np.zeros((1, 3, 3)), ["O", "H", "H"]) == [[0]]

    def test_injected_duplicate_is_collapsed(self) -> None:
        rng = np.random.default_rng(1)
        perm = np.array([2, 1, 3, 0, 4, 6, 7, 5])
        dup = METHYLAMMONIUM[perm] @ _rand_rotation(rng).T + rng.normal(size=3) * 3
        groups = deduplicate(np.stack([METHYLAMMONIUM, dup]), METHYLAMMONIUM_SYMBOLS)
        assert groups == [[0, 1]]

    def test_distinct_conformers_are_kept(self) -> None:
        a, syms = _water()
        b, _ = _water(twist=1.5)
        assert len(deduplicate(np.stack([a, b]), syms)) == 2

    def test_mirror_pair_merges_by_default_and_splits_when_asked(self) -> None:
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        symbols = ["C", "F", "Cl", "Br"]
        pair = np.stack([chiral, chiral * np.array([1.0, 1.0, -1.0])])
        assert len(deduplicate(pair, symbols, merge_mirrors=True)) == 1
        assert len(deduplicate(pair, symbols, merge_mirrors=False)) == 2

    def test_energy_criterion_blocks_a_geometric_match(self) -> None:
        """Both criteria must agree, the way CREGEN ANDs its thresholds."""
        rng = np.random.default_rng(2)
        dup = METHYLAMMONIUM @ _rand_rotation(rng).T
        pair = np.stack([METHYLAMMONIUM, dup])
        assert len(deduplicate(pair, METHYLAMMONIUM_SYMBOLS)) == 1
        far_apart = np.array([0.0, 5.0])  # kcal/mol
        assert len(deduplicate(pair, METHYLAMMONIUM_SYMBOLS, energies=far_apart, ethr=0.05)) == 2


class TestConformerMultiplicity:
    def test_achiral_conformer_stands_for_one_state(self) -> None:
        sym, syms = _water()
        m = conformer_multiplicity(np.array([sym]), syms)
        assert m == pytest.approx([1 / 2])  # achiral, sigma = 2

    def test_chiral_conformer_stands_for_two(self) -> None:
        """Its mirror image was folded away by deduplicate, so it counts twice."""
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        m = conformer_multiplicity(np.array([chiral]), ["C", "F", "Cl", "Br"])
        assert m == pytest.approx([2.0])

    def test_includes_enantiomer_suppresses_the_factor_of_two(self) -> None:
        """The microstate-level factor owns it there; applying both double-counts."""
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        m = conformer_multiplicity(
            np.array([chiral]), ["C", "F", "Cl", "Br"], includes_enantiomer=True
        )
        assert m == pytest.approx([1.0])

    def test_methylammonium(self) -> None:
        m = conformer_multiplicity(np.array([METHYLAMMONIUM]), METHYLAMMONIUM_SYMBOLS)
        assert m == pytest.approx([1 / 3])  # achiral, sigma = 3


class TestMergedAccounting:
    """A merged mirror pair must still add up to the two states it stands for."""

    SYMBOLS: ClassVar[list[str]] = ["C", "F", "Cl", "Br"]

    def _pair(self) -> np.ndarray:
        chiral = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]])
        return np.stack([chiral, chiral * np.array([1.0, 1.0, -1.0])])

    def test_merged_pair_totals_two_states(self) -> None:
        pair = self._pair()
        groups = deduplicate(pair, self.SYMBOLS, merge_mirrors=True)
        assert len(groups) == 1
        mult = conformer_multiplicity(pair[[g[0] for g in groups]], self.SYMBOLS)
        assert mult.sum() == pytest.approx(2.0)

    def test_multiplicity_assumes_mirrors_were_merged(self) -> None:
        """The contract, stated as a test so it cannot be violated silently.

        `conformer_multiplicity` gives every conformationally chiral conformer a
        factor of two on the understanding that its mirror image is not also in
        the list. A `merge_mirrors=False` grouping keeps both members of a pair
        and then doubles each, giving four states where there are two -- which
        is why the pipeline always deduplicates with mirrors merged.
        """
        pair = self._pair()
        split = deduplicate(pair, self.SYMBOLS, merge_mirrors=False)
        assert len(split) == 2
        mult = conformer_multiplicity(pair[[g[0] for g in split]], self.SYMBOLS)
        assert mult.sum() == pytest.approx(4.0)


class TestEffectiveEnergyOffset:
    def test_multiplicity_one_is_no_shift(self) -> None:
        assert effective_energy_offset(1.0) == pytest.approx(0.0)

    def test_doubling_lowers_by_rt_ln_two(self) -> None:
        kcal = effective_energy_offset(2.0) * 627.5094740631
        assert kcal == pytest.approx(-0.411, abs=1e-3)


class TestScreensDoNotHideDuplicates:
    """The screens must not reject a pair that is inside rthr.

    rthr is an RMSD, so the per-atom displacement it tolerates grows as
    rthr*sqrt(N), while the screen tolerances are fixed distances. If those are
    set too tight, a genuine duplicate is silently split past some molecule
    size -- and a missed duplicate costs RT ln 2 in the wrong direction.
    """

    def _chain(self, n_carbon: int) -> tuple[np.ndarray, list[str]]:
        """A crude alkane chain: enough atoms to make rthr*sqrt(N) exceed DIST_TOL."""
        coords, symbols = [], []
        for i in range(n_carbon):
            coords.append([1.27 * i, 0.0, 0.0])
            symbols.append("C")
            for sign in (1, -1):
                coords.append([1.27 * i, 0.63 * sign, 0.89])
                symbols.append("H")
        return np.array(coords), symbols

    def test_displaced_atom_inside_rthr_still_merges(self) -> None:
        coords, symbols = self._chain(12)  # 36 atoms
        moved = coords.copy()
        moved[1, 0] += 0.7  # one hydrogen, well beyond a fixed 0.45 A tolerance
        rmsd = np.sqrt(((coords - coords.mean(0) - (moved - moved.mean(0))) ** 2).sum(1).mean())
        assert rmsd < DEFAULT_RTHR, "test setup: the pair must be inside the threshold"
        assert deduplicate(np.stack([coords, moved]), symbols) == [[0, 1]]

    def test_a_pair_outside_rthr_still_splits(self) -> None:
        coords, symbols = self._chain(12)
        moved = coords.copy()
        moved[1, 0] += 1.6
        assert len(deduplicate(np.stack([coords, moved]), symbols)) == 2


class TestSamplingEnergyWindow:
    """Sampling's window must agree with refinement's about the same structure.

    Both measure against `G - RT ln(m)`. If sampling used the bare free energy,
    a merged mirror pair could be cut here and kept later -- and sampling.ewin
    is the window with the least headroom.
    """

    def _conf(self, energy: float, multiplicity: float = 1.0) -> Conformer:
        geom = Geometry(symbols=("H",), coords=np.zeros((1, 3)))
        conf = Conformer(geometry=geom, electronic_energy=energy)
        conf.multiplicity = multiplicity
        return conf

    def test_multiplicity_extends_the_window(self) -> None:
        window = 1.0
        just_outside = (window + 0.2) / HARTREE_TO_KCAL
        base = self._conf(0.0)

        plain = _filter_by_energy_window([base, self._conf(just_outside)], window)
        assert len(plain) == 1

        boosted = _filter_by_energy_window(
            [base, self._conf(just_outside, multiplicity=2.0)], window
        )
        assert len(boosted) == 2

    def test_empty_input(self) -> None:
        assert _filter_by_energy_window([], 10.0) == []


class TestTruncationIsReported:
    """Hitting the mapping ceiling must be audible, not absorbed.

    Truncating the enumeration can undercount sigma or lose the best mapping,
    and both failures are silent in the result: a smaller symmetry number and a
    larger RMSD look exactly like a legitimately less symmetric structure.
    """

    def test_warns_when_the_ceiling_is_reached(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tautomator.conformer_symmetry"):
            _candidate_mappings(METHYLAMMONIUM, METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS, limit=2)
        assert "truncated" in caplog.text

    def test_silent_when_it_is_not(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tautomator.conformer_symmetry"):
            maps = _candidate_mappings(METHYLAMMONIUM, METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS)
        assert len(maps) < MAX_MAPPINGS
        assert caplog.text == ""

    def test_truncation_can_undercount_sigma(self) -> None:
        """Why it is worth a warning: the wrong answer is a plausible one.

        Truncated to the identity alone, methylammonium would report sigma = 1
        instead of 3 -- indistinguishable in the output from a genuinely
        asymmetric conformer, and worth log10(3) = 0.48 pKa units.
        """
        full = _candidate_mappings(METHYLAMMONIUM, METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS)
        clipped = _candidate_mappings(
            METHYLAMMONIUM, METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS, limit=1
        )
        assert symmetry_number(METHYLAMMONIUM, METHYLAMMONIUM_SYMBOLS)[0] == 3
        assert len(clipped) == 1  # the identity, and nothing else
        assert len(full) > len(clipped)
