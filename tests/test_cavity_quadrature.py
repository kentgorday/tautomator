"""Tests for the inside/outside cavity quadrature."""

from __future__ import annotations

import numpy as np
import pytest

from tautomator import cavity_quadrature
from tautomator.cavity_quadrature import (
    Array,
    Region,
    becke_partition,
    build_cavity_grid,
    cavity_grid,
    partition_gradient,
)

# Two overlapping spheres, and a five-sphere cavity with one sphere mostly buried.
PAIR = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]]), np.array([3.4, 2.5]))
CLUSTER = (
    np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [2.4, 0.0, -1.5], [-2.4, 0.0, -1.5], [0.3, 0.2, 1.0]]
    ),
    np.array([3.85, 3.45, 3.85, 3.85, 2.0]),
)


def _inside(points: Array, centres: Array, radii: Array) -> Array:
    dist = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.asarray((dist < radii[None, :]).any(axis=1))


def _tails(points: Array, centres: Array, decay: float) -> Array:
    """Sum of exp(-decay r) on every centre: cusps inside, density-like tails outside."""
    dist = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.asarray(np.exp(-decay * dist).sum(axis=1))


class TestBeckePartition:
    def test_rows_sum_to_one_and_nuclei_own_themselves(self) -> None:
        centres, _radii = CLUSTER
        points = np.vstack([centres, np.random.default_rng(1).normal(scale=4.0, size=(50, 3))])
        cells = becke_partition(points, centres)
        assert cells.sum(axis=1) == pytest.approx(1.0, abs=1e-14)
        assert np.diag(cells[: len(centres)]) == pytest.approx(1.0, abs=1e-14)


class TestCavityGrid:
    def test_single_sphere_volume_is_exact(self) -> None:
        _coords, weights = cavity_grid(np.zeros((1, 3)), np.array([2.7]), "inside")
        assert weights.sum() == pytest.approx(4.0 / 3.0 * np.pi * 2.7**3, rel=1e-13)

    def test_volume_of_two_overlapping_spheres(self) -> None:
        centres, radii = PAIR
        d, (r1, r2) = 2.2, radii
        lens = (
            np.pi * (r1 + r2 - d) ** 2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)
        )
        expected = 4.0 / 3.0 * np.pi * (r1**3 + r2**3) - lens
        # Limited by the angular rule: the partition folds along the crease
        # where the spheres meet.
        _coords, weights = cavity_grid(centres, radii, "inside", 41, 40)
        assert weights.sum() == pytest.approx(expected, rel=1e-3)
        _coords, weights = cavity_grid(centres, radii, "inside", 77, 60)
        assert weights.sum() == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_points_lie_in_their_region(self, region: Region) -> None:
        centres, radii = CLUSTER
        coords, _weights = cavity_grid(centres, radii, region)
        assert np.all(_inside(coords, centres, radii) == (region == "inside"))

    @pytest.mark.parametrize("cavity", [PAIR, CLUSTER])
    @pytest.mark.parametrize(("order", "n_radial", "rel"), [(41, 40, 1e-3), (77, 60, 1e-4)])
    def test_inside_and_outside_fill_space(
        self, cavity: tuple[Array, Array], order: int, n_radial: int, rel: float
    ) -> None:
        centres, radii = cavity
        decay = 0.9
        total = 0.0
        region: Region
        for region in ("inside", "outside"):
            coords, weights = cavity_grid(centres, radii, region, order, n_radial)
            total += float(weights @ _tails(coords, centres, decay))
        assert total == pytest.approx(len(centres) * 8.0 * np.pi / decay**3, rel=rel)

    def test_outside_integral_converges(self) -> None:
        centres, radii = CLUSTER
        values = []
        for order, n_radial in [(41, 40), (59, 60)]:
            coords, weights = cavity_grid(centres, radii, "outside", order, n_radial)
            values.append(weights @ _tails(coords, centres, 0.9))
        assert values[0] == pytest.approx(values[1], rel=1e-3)

    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_is_smooth_in_the_geometry(self, region: Region) -> None:
        """No kinks along a displacement: second differences stay at their smooth size.

        Cutting rays at the spheres they cross gave square-root kinks whenever a
        ray became tangent to a sphere; here the points only ever move with their
        atoms.
        """
        centres, radii = CLUSTER
        h = 2e-4
        values = []
        for k in range(41):
            moved = centres.copy()
            moved[4] += (k - 20) * h * np.array([0.6, 0.0, 0.8])
            coords, weights = cavity_grid(moved, radii, region)
            values.append(weights @ _tails(coords, moved, 0.9))
        second = np.diff(values, 2) / h**2
        assert np.abs(np.diff(second)).max() < 0.2 * np.abs(second).max() + 1.0


class TestPartition:
    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_is_a_partition_of_unity(self, region: Region) -> None:
        centres, radii = CLUSTER
        rng = np.random.default_rng(2)
        points = rng.normal(scale=4.0, size=(400, 3))
        keep = _inside(points, centres, radii) == (region == "inside")
        points = points[keep]
        total = np.zeros(len(points))
        for a in range(len(centres)):
            total += cavity_quadrature._partition(
                points,
                np.full(len(points), a, dtype=np.intp),
                centres,
                radii,
                cavity_quadrature._pair_distances(centres),
                region == "inside",
                cavity_quadrature.STEP_WIDTH,
                4,
            )
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_outside_weights_vanish_inside_other_spheres(self) -> None:
        centres, radii = CLUSTER
        grid = build_cavity_grid(centres, radii, "outside")
        assert not np.any(_inside(grid.coords, centres, radii))

    @pytest.mark.parametrize("region", ["inside", "outside"])
    def test_gradient_matches_finite_differences(self, region: Region) -> None:
        """The weights' derivative, with the points moving with their atoms.

        The integrand is a field fixed in space, so its own change as the points
        move is added here: values times its gradient, on each atom's points.
        """
        centres, radii = CLUSTER
        centre = np.array([0.4, -0.3, 0.2])

        def field(points: Array) -> Array:
            return np.asarray(np.exp(-0.15 * np.sum((points - centre) ** 2, axis=1)))

        grid = build_cavity_grid(centres, radii, region)
        analytic = partition_gradient(grid, centres, radii, field(grid.coords))
        moving = (
            grid.weights[:, None] * (-0.3 * (grid.coords - centre)) * field(grid.coords)[:, None]
        )
        for a in range(len(centres)):
            analytic[a] += moving[grid.owners == a].sum(axis=0)
        h = 1e-6
        for atom, axis in [(1, 2), (4, 0), (2, 1)]:
            values = []
            for sign in (1.0, -1.0):
                moved = centres.copy()
                moved[atom, axis] += sign * h
                g = build_cavity_grid(moved, radii, region)
                values.append(g.weights @ field(g.coords))
            numeric = (values[0] - values[1]) / (2 * h)
            assert analytic[atom, axis] == pytest.approx(numeric, rel=1e-6, abs=1e-8)
        # moving every atom together moves nothing but the field under it
        assert analytic.sum(axis=0) == pytest.approx(
            (
                grid.weights[:, None]
                * (-0.3 * (grid.coords - centre))
                * field(grid.coords)[:, None]
            ).sum(axis=0),
            abs=1e-9,
        )
