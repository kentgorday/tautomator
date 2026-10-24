"""Tests for the compiled dispersion kernel against a direct numpy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from tautomator.dispersion_kernel import (
    Array,
    kernel_derivatives,
    kernel_integral,
    kernel_surface_derivatives,
)

DELTA = 7.0


def _reference(points: Array, coords: Array, normals: Array, areas: Array, delta: float) -> Array:
    """The formula written out with numpy, as nonelectrostatic had it."""
    sep = coords[None, :, :] - points[:, None, :]
    t = np.sum(sep**2, axis=2) ** 1.5 / delta**3
    r_dot_n = np.einsum("pix,ix->pi", sep, normals)
    return np.asarray((r_dot_n * np.arctan(t) / t) @ areas / (3.0 * delta**6))


@pytest.fixture()
def system() -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(3)
    normals = rng.normal(size=(40, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    coords = 3.0 * normals + rng.normal(scale=0.3, size=(40, 3))
    areas = rng.random(40)
    points = rng.normal(scale=1.5, size=(25, 3))
    return points, coords, normals, areas


class TestKernel:
    @pytest.mark.parametrize("delta", [DELTA, 2.0, 0.5])
    def test_matches_numpy(self, system: tuple[Array, Array, Array, Array], delta: float) -> None:
        points, coords, normals, areas = system
        assert kernel_integral(points, coords, normals, areas, delta) == pytest.approx(
            _reference(points, coords, normals, areas, delta), rel=1e-13, abs=1e-18
        )

    def test_series_branch_is_continuous(self) -> None:
        """Pairs on either side of the switch to the small-t series agree with arctan."""
        normals = np.array([[0.0, 0.0, 1.0]])
        areas = np.ones(1)
        for t in (0.5e-2, 0.999e-2, 1.001e-2, 2e-2, 1e-6):
            r = DELTA * t ** (1.0 / 3.0)
            coords = np.array([[0.6 * r, 0.0, 0.8 * r]])
            value = kernel_integral(np.zeros((1, 3)), coords, normals, areas, DELTA)
            exact = 0.8 * r * np.arctan(t) / t / (3.0 * DELTA**6)
            assert value[0] == pytest.approx(exact, rel=1e-14)

    def test_derivatives_match_finite_differences(
        self, system: tuple[Array, Array, Array, Array]
    ) -> None:
        points, coords, normals, areas = system
        weights = np.random.default_rng(4).normal(size=len(points))
        J, dJ, dE_ds, dE_dA = kernel_derivatives(points, coords, normals, areas, 2.0, weights)
        assert pytest.approx(kernel_integral(points, coords, normals, areas, 2.0), rel=1e-14) == J

        def energy(p: Array, c: Array, a: Array) -> float:
            return float(weights @ kernel_integral(p, c, normals, a, 2.0))

        h = 1e-5
        for x in range(3):
            step = np.zeros(3)
            step[x] = h
            # one point, one element and one area at a time
            p_plus, p_minus = points.copy(), points.copy()
            p_plus[7] += step
            p_minus[7] -= step
            numeric = (
                kernel_integral(p_plus, coords, normals, areas, 2.0)[7]
                - kernel_integral(p_minus, coords, normals, areas, 2.0)[7]
            ) / (2 * h)
            assert dJ[7, x] == pytest.approx(numeric, rel=1e-7, abs=1e-14)
            c_plus, c_minus = coords.copy(), coords.copy()
            c_plus[11] += step
            c_minus[11] -= step
            numeric = (energy(points, c_plus, areas) - energy(points, c_minus, areas)) / (2 * h)
            assert dE_ds[11, x] == pytest.approx(numeric, rel=1e-7, abs=1e-14)
        # the energy is linear in the areas, so a large step costs no truncation error
        # and keeps roundoff (eps |E| / h ~ 1e-9 at h = 1e-5) out of the comparison
        h_area = 0.1
        a_plus, a_minus = areas.copy(), areas.copy()
        a_plus[5] += h_area
        a_minus[5] -= h_area
        numeric = (energy(points, coords, a_plus) - energy(points, coords, a_minus)) / (2 * h_area)
        assert dE_dA[5] == pytest.approx(numeric, rel=1e-9)

    def test_derivatives_near_the_surface(self) -> None:
        """A point almost on an element: the kernel stays finite and smooth."""
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        coords = np.array([[0.0, 0.0, 1e-3], [2.0, 0.0, 0.0]])
        areas = np.array([0.3, 0.7])
        J, dJ, _, _ = kernel_derivatives(
            np.zeros((1, 3)), coords, normals, areas, DELTA, np.ones(1)
        )
        assert np.all(np.isfinite(J)) and np.all(np.isfinite(dJ))


def test_normals_channel() -> None:
    """On a DROP surface the normals move: dE/dn, last in kernel_surface_derivatives, as
    psi4-swig-pcm's dispersion_kernel orders it; the first four are kernel_derivatives'."""
    rng = np.random.default_rng(1)
    points = rng.normal(size=(40, 3))
    coords = 3.0 * rng.normal(size=(25, 3))
    normals = rng.normal(size=(25, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    areas = rng.uniform(0.1, 1.0, 25)
    weights = rng.normal(size=40)
    out = kernel_surface_derivatives(points, coords, normals, areas, DELTA, weights)
    for a, b in zip(
        out[:4], kernel_derivatives(points, coords, normals, areas, DELTA, weights), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    dE_dn = out[4]
    # linear in each normal, so a large step costs no truncation error
    h = 0.1
    for i, k in ((0, 0), (7, 2), (24, 1)):
        plus, minus = normals.copy(), normals.copy()
        plus[i, k] += h
        minus[i, k] -= h
        numeric = (
            weights @ kernel_integral(points, coords, plus, areas, DELTA)
            - weights @ kernel_integral(points, coords, minus, areas, DELTA)
        ) / (2 * h)
        assert dE_dn[i, k] == pytest.approx(numeric, rel=1e-10)
