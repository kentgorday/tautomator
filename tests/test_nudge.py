"""The nudge off symmetric shapes before the refinement optimization (tautomator.nudge).

xtb is stubbed except in the slow tests: what is under test is which conformers
are moved, along which mode, how far, and that a failure leaves the conformer as
sampled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rdkit.Chem import GetPeriodicTable

from tautomator import xtb_runner
from tautomator.conformer_symmetry import symmetry_number, symmetry_operations
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.nudge import (
    AMPLITUDE,
    BREAKS,
    SECOND_CUTOFF,
    SETTLE,
    SETTLE_REACH,
    SOFT_CUTOFF,
    Nudge,
    Operation,
    fold,
    localize,
    nudge,
    settle,
    symmetric_part,
    symmetrize,
)
from tautomator.types import Geometry

# Water in the yz plane: C2 about z, mirror planes xz and yz.
WATER = Geometry(
    symbols=("O", "H", "H"),
    coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]]),
)
# Four different substituents on a tetrahedral carbon: no operation but the identity.
CHFCLBR = Geometry(
    symbols=("C", "H", "F", "Cl", "Br"),
    coords=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.09],
            [1.254, 0.0, -0.443],
            [-0.834, 1.444, -0.589],
            [-0.914, -1.583, -0.646],
        ]
    ),
)
# Modes of WATER: kept by every operation, and reversed by C2 and the xz mirror.
SYMMETRIC = np.array([[0.0, 0.0, -0.2], [0.0, 0.5, 0.4], [0.0, -0.5, 0.4]])
ANTISYMMETRIC = np.array([[0.0, -0.2, 0.0], [0.0, 0.5, 0.4], [0.0, 0.5, -0.4]])


def propeller() -> Geometry:
    """Three C-H arms around an N on a threefold axis, each H leaning the same way round.

    The lean leaves the threefold turn and removes every mirror: sigma 3, chiral.
    """
    coords = [[0.0, 0.0, 0.4]]
    hydrogens = []
    for k in range(3):
        a = 2 * np.pi * k / 3
        radial = np.array([np.cos(a), np.sin(a), 0.0])
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        carbon = 1.4 * radial
        coords.append(list(carbon))
        hydrogens.append(list(carbon + 0.5 * radial + 0.6 * tangent + 0.6 * np.array([0, 0, 1])))
    symbols = ("N", "C", "C", "C", "H", "H", "H")
    return Geometry(symbols=symbols, coords=np.array(coords + hydrogens))


PROPELLER = propeller()


def degenerate_mode(geom: Geometry, seed: int = 0) -> np.ndarray:
    """A displacement with no totally symmetric part: for a threefold axis alone, one of
    a degenerate pair, which the turn rotates by 120 degrees rather than reverses."""
    ops = symmetry_operations(geom.coords, geom.symbols)
    v = np.random.default_rng(seed).normal(size=geom.coords.shape)
    average = (v + sum(v[p] @ r.T for p, r in ops)) / (len(ops) + 1)
    return np.asarray(v - average)


def operations_of(geom: Geometry) -> tuple[Operation, ...]:
    return tuple(
        (tuple(int(i) for i in p), bool(np.linalg.det(r) < 0))
        for p, r in symmetry_operations(geom.coords, geom.symbols)
    )


def stub_modes(
    monkeypatch: pytest.MonkeyPatch, freqs: list[float], modes: list[np.ndarray]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(geom: Geometry, charge: int = 0, **kw: Any) -> tuple[list[float], np.ndarray]:
        calls.append({"charge": charge, **kw})
        return list(freqs), np.stack(modes)

    monkeypatch.setattr(xtb_runner, "normal_modes", fake)
    return calls


class TestSymmetricPart:
    def test_tells_the_two_kinds_of_mode_apart(self) -> None:
        ops = symmetry_operations(WATER.coords, WATER.symbols)
        assert symmetric_part(ANTISYMMETRIC, ops) == pytest.approx(0.0)
        assert symmetric_part(SYMMETRIC, ops) == pytest.approx(1.0)

    def test_a_threefold_turn_breaks_a_degenerate_mode(self) -> None:
        # The turn rotates the mode by 120 degrees, <v|gv>/<v|v> = cos 120 = -0.5
        # under both turns, which a test for reversal sits exactly on.
        ops = symmetry_operations(PROPELLER.coords, PROPELLER.symbols)
        mode = degenerate_mode(PROPELLER)
        assert len(ops) == 2
        for perm, rot in ops:
            turned = mode[perm] @ rot.T
            assert (mode * turned).sum() / (mode * mode).sum() == pytest.approx(-0.5)
        assert symmetric_part(mode, ops) == pytest.approx(0.0, abs=1e-9)


class TestNudge:
    def test_moves_along_the_softest_breaking_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The softer symmetric mode is skipped: moving along it keeps the symmetry.
        stub_modes(monkeypatch, [40.0, 80.0, 1600.0], [SYMMETRIC, ANTISYMMETRIC, SYMMETRIC])
        moved, done = nudge(WATER, 0, solvent="water")
        assert done is not None and done.frequency == 80.0
        assert done.symmetric_part == pytest.approx(0.0)
        step = moved.coords - WATER.coords
        assert np.linalg.norm(step, axis=1).max() == pytest.approx(AMPLITUDE)
        assert np.allclose(
            step / np.linalg.norm(step), ANTISYMMETRIC / np.linalg.norm(ANTISYMMETRIC)
        )

    def test_the_other_sign_moves_the_other_way(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_modes(monkeypatch, [80.0], [ANTISYMMETRIC])
        plus, _ = nudge(WATER, 0, solvent=None)
        minus, _ = nudge(WATER, 0, solvent=None, sign=-1.0)
        assert np.allclose(plus.coords - WATER.coords, WATER.coords - minus.coords)

    def test_a_stiff_mode_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_modes(monkeypatch, [SOFT_CUTOFF + 50.0], [ANTISYMMETRIC])
        moved, done = nudge(WATER, 0, solvent=None)
        assert done is None and moved is WATER

    def test_a_mostly_symmetric_mode_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mixed = SYMMETRIC + 0.3 * ANTISYMMETRIC
        ops = symmetry_operations(WATER.coords, WATER.symbols)
        assert symmetric_part(mixed, ops) > BREAKS
        stub_modes(monkeypatch, [80.0], [mixed])
        assert nudge(WATER, 0, solvent=None)[1] is None

    def test_a_degenerate_mode_of_a_threefold_axis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_modes(monkeypatch, [60.0], [degenerate_mode(PROPELLER)])
        _, done = nudge(PROPELLER, 0, solvent=None)
        assert done is not None and done.symmetric_part == pytest.approx(0.0, abs=1e-9)
        assert done.symmetry == (3, False)

    def test_no_hessian_without_a_symmetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = stub_modes(monkeypatch, [-50.0], [np.ones((5, 3))])
        moved, done = nudge(CHFCLBR, 0, solvent=None)
        assert done is None and moved is CHFCLBR and calls == []

    def test_no_hessian_for_a_linear_molecule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every turn about the axis maps it onto itself, so the operations the
        # relabelling search finds are arbitrary ones.
        hcn = Geometry(
            symbols=("H", "C", "N"),
            coords=np.array([[0.0, 0.0, -1.066], [0.0, 0.0, 0.0], [0.0, 0.003, 1.156]]),
        )
        calls = stub_modes(monkeypatch, [700.0], [np.ones((3, 3))])
        assert nudge(hcn, 0, solvent=None) == (hcn, None) and calls == []

    def test_a_failed_hessian_keeps_the_conformer_as_sampled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(*a: Any, **kw: Any) -> None:
            raise RuntimeError("xtb Hessian failed (exit 1)")

        monkeypatch.setattr(xtb_runner, "normal_modes", broken)
        moved, done = nudge(WATER, 0, solvent=None)
        assert done is None and moved is WATER
        assert "optimizing as sampled" in caplog.text

    def test_hessian_in_the_given_solvent_and_charge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = stub_modes(monkeypatch, [80.0], [ANTISYMMETRIC])
        nudge(WATER, -1, solvent="woctanol", threads=3)
        assert calls == [{"charge": -1, "solvent": "woctanol", "threads": 3}]

    def test_records_the_operations_and_the_sampled_symmetry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_modes(monkeypatch, [80.0], [ANTISYMMETRIC])
        _, done = nudge(WATER, 0, solvent=None)
        assert done is not None and done.symmetry == (2, True)
        assert len(done.operations) == 3
        assert ((0, 2, 1), False) in done.operations  # the C2 swaps the hydrogens
        assert done.later == ()


def unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v / np.linalg.norm(v, axis=1).max())


# A second symmetry-breaking displacement of WATER, not along ANTISYMMETRIC.
SECOND = degenerate_mode(WATER, seed=1)


class TestSecondPush:
    def test_a_later_breaking_mode_is_pushed_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert symmetric_part(SECOND, symmetry_operations(WATER.coords, WATER.symbols)) < 1e-9
        stub_modes(monkeypatch, [80.0, 120.0], [ANTISYMMETRIC, SECOND])
        moved, done = nudge(WATER, 0, solvent=None)
        assert done is not None and done.frequency == 80.0 and done.later == (120.0,)
        # The softest mode keeps its whole push; the later one adds its own.
        expected = AMPLITUDE * (unit(ANTISYMMETRIC) + unit(SECOND))
        assert np.allclose(moved.coords - WATER.coords, expected)

    def test_a_later_mode_above_the_second_cutoff_is_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_modes(monkeypatch, [80.0, SECOND_CUTOFF + 10.0], [ANTISYMMETRIC, SECOND])
        moved, done = nudge(WATER, 0, solvent=None)
        assert done is not None and done.later == ()
        assert np.allclose(moved.coords - WATER.coords, AMPLITUDE * unit(ANTISYMMETRIC))

    def test_a_later_symmetric_mode_is_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_modes(monkeypatch, [80.0, 100.0], [ANTISYMMETRIC, SYMMETRIC])
        moved, done = nudge(WATER, 0, solvent=None)
        assert done is not None and done.later == ()
        assert np.allclose(moved.coords - WATER.coords, AMPLITUDE * unit(ANTISYMMETRIC))

    def test_a_first_mode_above_the_second_cutoff_brings_no_later_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = SECOND_CUTOFF + 20.0
        stub_modes(monkeypatch, [first, first + 10.0], [ANTISYMMETRIC, SECOND])
        _, done = nudge(WATER, 0, solvent=None)
        assert done is not None and done.frequency == first and done.later == ()

    def test_the_other_sign_reverses_the_whole_move(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_modes(monkeypatch, [80.0, 120.0], [ANTISYMMETRIC, SECOND])
        plus, _ = nudge(WATER, 0, solvent=None)
        minus, _ = nudge(WATER, 0, solvent=None, sign=-1.0)
        assert np.allclose(plus.coords - WATER.coords, WATER.coords - minus.coords)


def two_groups(n: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Unit displacements of atoms 0-2 and of atoms 5-7 alone, of n hydrogens."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros((n, 3)), np.zeros((n, 3))
    a[:3], b[5:] = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    return a / np.linalg.norm(a), b / np.linalg.norm(b)


class TestLocalize:
    @pytest.mark.parametrize("angle", [np.pi / 4, 0.3, 1.2, 0.0])
    def test_pulls_two_groups_apart(self, angle: float) -> None:
        # An in-phase and an out-of-phase pair at pi/4, any other mix too.
        a, b = two_groups()
        mixed = [np.cos(angle) * a + np.sin(angle) * b, -np.sin(angle) * a + np.cos(angle) * b]
        found = localize(mixed, ["H"] * 8)
        overlaps = sorted([round(abs(float((u * x).sum())), 6) for x in (a, b)] for u in found)
        assert overlaps == [[0.0, 1.0], [1.0, 0.0]]

    def test_pulls_three_groups_apart(self) -> None:
        rng = np.random.default_rng(2)
        tops = []
        for k in range(3):
            t = np.zeros((9, 3))
            t[3 * k : 3 * k + 3] = rng.normal(size=(3, 3))
            tops.append(t / np.linalg.norm(t))
        mix = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        mixed = [sum(mix[i, k] * tops[k] for k in range(3)) for i in range(3)]
        found = localize(mixed, ["H"] * 9)
        overlaps = np.array([[abs(float((u * t).sum())) for t in tops] for u in found])
        assert np.allclose(np.sort(overlaps, axis=1)[:, -1], 1.0, atol=1e-6)
        assert np.allclose(np.sort(overlaps.argmax(axis=1)), [0, 1, 2])

    def test_spans_the_same_displacements(self) -> None:
        a, b = two_groups(seed=3)
        mixed = [a + b, a - 0.5 * b]
        found = localize(mixed, ["C", "H", "H", "O", "N", "H", "H", "H"])
        basis = np.linalg.qr(np.array([m.ravel() for m in mixed]).T)[0]
        for u in found:
            assert np.linalg.norm(u) == pytest.approx(1.0)
            assert np.linalg.norm(basis @ (basis.T @ u.ravel())) == pytest.approx(1.0)

    def test_one_mode_is_only_scaled(self) -> None:
        (found,) = localize([2.0 * ANTISYMMETRIC], list(WATER.symbols))
        assert np.allclose(found, ANTISYMMETRIC / np.linalg.norm(ANTISYMMETRIC))


# WATER with one O-H stretched: the C2 is gone, the molecular plane is not.
STRETCHED = Geometry(
    symbols=WATER.symbols, coords=WATER.coords + np.array([[0, 0, 0], [0, 0.25, 0.2], [0, 0, 0]])
)
C2V = Nudge(
    frequency=80.0, symmetric_part=0.0, operations=operations_of(WATER), symmetry=(2, True)
)


class TestSymmetrize:
    def test_the_average_has_the_symmetry_back(self) -> None:
        shape, gap = symmetrize(STRETCHED, C2V.operations)
        assert symmetry_number(shape.coords, list(shape.symbols)) == (2, True)
        assert symmetry_number(STRETCHED.coords, list(STRETCHED.symbols)) == (1, True)
        assert 0.0 < gap < 0.6

    def test_a_symmetric_shape_is_its_own_average(self) -> None:
        shape, gap = symmetrize(WATER, C2V.operations)
        assert gap == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(shape.coords, WATER.coords)

    def test_a_reflection_relabels_after_reflecting(self) -> None:
        # The xz mirror (y -> -y) swaps the hydrogens too; in the frame symmetrize
        # uses, that is a reflection followed by a proper rotation.
        shape, _ = symmetrize(STRETCHED, (((0, 2, 1), True),))
        assert symmetry_number(shape.coords, list(shape.symbols)) == (2, True)

    def test_a_threefold_axis_comes_back_from_far_off(self) -> None:
        # One average with a turn of order three is not symmetric; the steps are.
        mode = degenerate_mode(PROPELLER)
        far = Geometry(
            symbols=PROPELLER.symbols,
            coords=PROPELLER.coords + 0.4 * mode / np.linalg.norm(mode, axis=1).max(),
        )
        assert symmetry_number(far.coords, list(far.symbols)) == (1, False)
        shape, gap = symmetrize(far, operations_of(PROPELLER))
        assert gap > 0.4
        x = shape.coords - shape.coords.mean(axis=0)
        for perm, _ in operations_of(PROPELLER):
            z = x[list(perm)]
            u, _, vt = np.linalg.svd(z.T @ x)
            assert np.allclose(z @ (vt.T @ u.T).T, x, atol=1e-8)
        assert symmetry_number(shape.coords, list(shape.symbols)) == (3, False)


class TestSettle:
    E = -76.0

    def opt(self, depth_kcal: float, calls: list[Geometry], *, converged: bool = True) -> Any:
        """An optimizer that returns the start unchanged, depth_kcal above E."""

        def optimize(g: Geometry) -> tuple[Geometry, float, bool]:
            calls.append(g)
            return g, self.E + depth_kcal / HARTREE_TO_KCAL, converged

        return optimize

    def test_nothing_to_do_when_the_symmetry_survived(self) -> None:
        calls: list[Geometry] = []
        assert settle(C2V, WATER, self.E, self.opt(0.01, calls)) is None and calls == []

    def test_a_shallow_well_goes_back_to_the_symmetric_shape(self) -> None:
        calls: list[Geometry] = []
        settled = settle(C2V, STRETCHED, self.E, self.opt(0.02, calls))
        assert settled is not None and settled.replaces
        assert settled.depth == pytest.approx(0.02)
        assert settled.geometry is calls[0]
        assert settled.energy == pytest.approx(self.E + 0.02 / HARTREE_TO_KCAL)
        # The optimization starts from the average with its images, which has the symmetry.
        assert symmetry_number(calls[0].coords, list(calls[0].symbols)) == (2, True)

    def test_a_deeper_well_keeps_the_nudge(self) -> None:
        settled = settle(C2V, STRETCHED, self.E, self.opt(SETTLE + 0.01, []))
        assert settled is not None and not settled.replaces
        assert settled.depth == pytest.approx(SETTLE + 0.01)

    def test_a_nudged_result_above_its_symmetric_shape_goes_back(self) -> None:
        # Convergence noise on a flat valley can leave the nudged result a little high.
        settled = settle(C2V, STRETCHED, self.E, self.opt(-0.005, []))
        assert settled is not None and settled.replaces and settled.depth < 0

    def test_a_threefold_axis_settles_back(self) -> None:
        mode = degenerate_mode(PROPELLER)
        off = Geometry(
            symbols=PROPELLER.symbols,
            coords=PROPELLER.coords + 0.3 * mode / np.linalg.norm(mode, axis=1).max(),
        )
        done = Nudge(60.0, 0.0, operations_of(PROPELLER), (3, False))
        calls: list[Geometry] = []
        settled = settle(done, off, self.E, self.opt(0.02, calls))
        assert settled is not None and settled.replaces
        assert symmetry_number(calls[0].coords, list(calls[0].symbols)) == (3, False)

    def test_nothing_to_do_for_a_conformer_far_from_its_image(self) -> None:
        far = Geometry(
            symbols=WATER.symbols,
            coords=WATER.coords + np.array([[0, 0, 0], [0, 1.6, 1.0], [0, 0, 0]]),
        )
        assert symmetrize(far, C2V.operations)[1] > SETTLE_REACH
        calls: list[Geometry] = []
        assert settle(C2V, far, self.E, self.opt(0.0, calls)) is None and calls == []

    def test_a_failed_optimization_keeps_the_nudged_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(g: Geometry) -> tuple[Geometry, float, bool]:
            raise RuntimeError("SCF did not converge")

        assert settle(C2V, STRETCHED, self.E, broken) is None
        assert "keeping the nudged result" in caplog.text

    def test_an_unconverged_shape_keeps_the_nudged_result(self) -> None:
        assert settle(C2V, STRETCHED, self.E, self.opt(0.0, [], converged=False)) is None

    def test_a_shape_that_rolled_off_its_symmetry_keeps_the_nudged_result(self) -> None:
        def rolls(g: Geometry) -> tuple[Geometry, float, bool]:
            return STRETCHED, self.E, True

        assert settle(C2V, STRETCHED, self.E, rolls) is None


class TestFold:
    """A shallow well beside a symmetric conformer of the same microstate joins it."""

    @staticmethod
    def keep(other: Geometry, de: float, mult: tuple[float, float] = (0.5, 1.0)) -> list[int]:
        """fold() on [WATER, other], other lying de kcal/mol below WATER.

        WATER's multiplicity is 1/2 (sigma 2, achiral); STRETCHED's is 1.
        """
        return fold(
            [np.asarray(WATER.coords), np.asarray(other.coords)],
            [0.0, -de],
            list(mult),
            list(WATER.symbols),
        )

    def test_a_shallow_well_joins_the_symmetric_conformer(self) -> None:
        assert self.keep(STRETCHED, SETTLE - 0.03) == [0]

    def test_a_deeper_well_stays(self) -> None:
        assert self.keep(STRETCHED, SETTLE + 0.01) == [0, 1]

    def test_a_well_above_the_symmetric_conformer_joins_it(self) -> None:
        assert self.keep(STRETCHED, -0.005) == [0]

    def test_a_structure_well_above_the_symmetric_one_stays(self) -> None:
        assert self.keep(STRETCHED, -(SETTLE + 0.01)) == [0, 1]

    def test_a_relabelled_mirror_image_still_joins(self) -> None:
        # The same well reached from another start: hydrogens swapped, then reflected.
        swapped = np.asarray(STRETCHED.coords)[[0, 2, 1]] * np.array([-1.0, 1.0, 1.0])
        assert self.keep(Geometry(symbols=WATER.symbols, coords=swapped), 0.04) == [0]

    def test_a_conformer_far_from_the_symmetric_one_stays(self) -> None:
        far = Geometry(
            symbols=WATER.symbols,
            coords=WATER.coords + np.array([[0, 0, 0], [0, 1.6, 1.0], [0, 0, 0]]),
        )
        assert self.keep(far, 0.01) == [0, 1]

    def test_conformers_with_as_many_states_are_left_alone(self) -> None:
        # Nothing to fold into: neither stands for fewer states than the other.
        assert self.keep(STRETCHED, 0.01, mult=(1.0, 1.0)) == [0, 1]

    def test_a_turned_propeller_joins_its_threefold_shape(self) -> None:
        # Chiral both: the threefold shape stands for 2/3 of a state, the turned one for 2.
        mode = degenerate_mode(PROPELLER)
        turned = PROPELLER.coords + 0.1 * mode / np.linalg.norm(mode, axis=1).max()
        relabelled = turned[[0, 2, 3, 1, 5, 6, 4]]  # the arms counted from the next one
        coords = [np.asarray(PROPELLER.coords), relabelled]
        symbols = list(PROPELLER.symbols)
        assert fold(coords, [0.0, -0.04], [2 / 3, 2.0], symbols) == [0]
        assert fold(coords, [0.0, -(SETTLE + 0.01)], [2 / 3, 2.0], symbols) == [0, 1]


def spring_hessian(geom: Geometry, pairs: list[tuple[int, int]]) -> np.ndarray:
    """The Cartesian Hessian of unit springs at rest between ``pairs`` of atoms."""
    n = geom.n_atoms
    h = np.zeros((3 * n, 3 * n))
    for i, j in pairs:
        e = geom.coords[i] - geom.coords[j]
        block = np.outer(e, e) / (e @ e)
        for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            h[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += sign * block
    return h


class TestCartesianModes:
    """xtb_runner turns the Hessian file into the displacement patterns the nudge moves along."""

    MASS = np.array([GetPeriodicTable().GetAtomicWeight(s) for s in WATER.symbols])

    def test_modes_solve_the_cartesian_eigenproblem(self) -> None:
        # H d = w**2 M d: the displacements are Cartesian, not mass-weighted.
        h = spring_hessian(WATER, [(0, 1), (0, 2), (1, 2)])
        freqs, modes = xtb_runner._cartesian_modes(WATER, h)
        assert len(freqs) == 3 and min(freqs) > 0
        lam = (np.array(freqs) / xtb_runner._WAVENUMBER) ** 2
        mass = np.repeat(self.MASS, 3)
        for w2, d in zip(lam, modes, strict=True):
            assert np.allclose(h @ d.ravel(), w2 * mass * d.ravel(), atol=1e-10)

    def test_modes_carry_no_momentum(self) -> None:
        h = spring_hessian(WATER, [(0, 1), (0, 2), (1, 2)])
        _, modes = xtb_runner._cartesian_modes(WATER, h)
        r = WATER.coords - (self.MASS[:, None] * WATER.coords).sum(axis=0) / self.MASS.sum()
        for d in modes:
            assert np.allclose((self.MASS[:, None] * d).sum(axis=0), 0.0, atol=1e-12)
            assert np.allclose((self.MASS[:, None] * np.cross(r, d)).sum(axis=0), 0.0, atol=1e-12)

    def test_a_linear_molecule_has_one_mode_more(self) -> None:
        co2 = Geometry(
            symbols=("O", "C", "O"), coords=np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
        )
        freqs, _ = xtb_runner._cartesian_modes(co2, spring_hessian(co2, [(0, 1), (1, 2), (0, 2)]))
        assert len(freqs) == 4

    def test_reads_the_hessian_file(self) -> None:
        values = np.arange(81, dtype=float).reshape(9, 9)
        rows = [" ".join(f"{v:.6f}" for v in values.ravel()[k : k + 5]) for k in range(0, 81, 5)]
        text = "$hessian\n" + "\n".join(rows) + "\n$end\n"
        assert np.allclose(xtb_runner._parse_hessian(text, 3), (values + values.T) / 2)
        with pytest.raises(RuntimeError, match="holds 81 values"):
            xtb_runner._parse_hessian(text, 2)


# CH3CBr3 at its GFN2 minimum: C3v, with a degenerate pair of soft CBr3 turns.
CH3CBR3 = Geometry(
    symbols=("C", "C", "Br", "Br", "Br", "H", "H", "H"),
    coords=np.array(
        [
            [-0.84121573, -0.22087010, 0.04408964],
            [0.61334622, 0.16104069, -0.03214715],
            [1.55172520, -0.95244169, -1.34362493],
            [0.80769613, 2.03643987, -0.56644827],
            [1.48146620, -0.07553224, 1.70876433],
            [-0.94066012, -1.26485885, 0.34172522],
            [-1.35577971, 0.40274088, 0.77533644],
            [-1.31657920, -0.08651856, -0.92769627],
        ]
    ),
)


@pytest.mark.slow
class TestRealModes:
    def test_match_xtbs_frequencies_and_carry_no_momentum(self) -> None:
        freqs, modes = xtb_runner.normal_modes(CH3CBR3)
        assert np.allclose(freqs, xtb_runner.frequencies(CH3CBR3), atol=0.5)
        mass = np.array([GetPeriodicTable().GetAtomicWeight(s) for s in CH3CBR3.symbols])
        r = CH3CBR3.coords - (mass[:, None] * CH3CBR3.coords).sum(axis=0) / mass.sum()
        assert np.abs((mass[None, :, None] * modes).sum(axis=1)).max() < 1e-8
        assert np.abs((mass[None, :, None] * np.cross(r[None], modes)).sum(axis=1)).max() < 1e-8

    def test_modes_follow_the_input_frame(self) -> None:
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        turned = Geometry(CH3CBR3.symbols, CH3CBR3.coords @ q.T + np.array([1.3, -2.1, 0.7]))
        _, modes = xtb_runner.normal_modes(CH3CBR3)
        _, turned_modes = xtb_runner.normal_modes(turned)
        k = 2  # the CBr3 umbrella, nondegenerate
        assert abs((modes[k] * (turned_modes[k] @ q)).sum()) == pytest.approx(1.0, abs=1e-3)

    def test_nudges_along_a_degenerate_turn_and_its_partner(self) -> None:
        # The partner breaks the symmetry too and lies under SECOND_CUTOFF, so the
        # start moves along both members of the pair.
        moved, done = nudge(CH3CBR3, 0, solvent=None)
        assert done is not None and done.symmetry == (3, True)
        assert done.frequency < 150 and done.symmetric_part < 1e-6
        assert len(done.later) == 1 and done.later[0] == pytest.approx(done.frequency, abs=1.0)
        _, modes = xtb_runner.normal_modes(CH3CBR3)
        (partner,) = localize([modes[1]], list(CH3CBR3.symbols))
        expected = AMPLITUDE * (unit(modes[0]) + unit(partner))
        assert np.allclose(moved.coords - CH3CBR3.coords, expected, atol=1e-6)
