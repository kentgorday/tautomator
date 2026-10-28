"""surface_quadrature: volume integrals on a closed surface, and their derivatives.

On a sphere the outside integral of a function radial about its centre is known exactly,
whichever nuclei share the rays; the derivatives of the ray sum and of the all-space grid
are checked against central differences.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import lebedev_rule, quad

from tautomator import surface_quadrature as sq

ALPHA = 0.25
H = 1e-6
# The ray sum's derivatives in the surface points, normals and areas are 1e-5 to 1e-4 of an
# integral ~2: at H the differences' rounding (~4e-10) is up to 1e-5 of them, which is
# more than their tolerance (OpenBLAS's summation order happened to stay just inside it,
# Accelerate's does not).  At this step they match to ~3e-8 with either BLAS.
H_SURFACE = 1e-4


@pytest.fixture()
def sphere() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A sphere of radius 4 bohr (Lebedev order 41) around three off-centre nuclei."""
    centres = np.random.default_rng(3).normal(scale=0.8, size=(3, 3))
    directions, w = lebedev_rule(41)
    directions = np.ascontiguousarray(directions.T)
    return 4.0 * directions, directions.copy(), 16.0 * w, centres


def _gaussian(y: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = y - centre
    v = np.exp(-ALPHA * np.sum(d * d, axis=1))
    return v, -2.0 * ALPHA * d * v[:, None]


def _outside(
    xyz: np.ndarray, normals: np.ndarray, areas: np.ndarray, centres: np.ndarray
) -> float:
    rays = sq.build_exterior_rays(xyz, normals, areas, centres)
    return float(rays.weights @ _gaussian(rays.coords, np.zeros(3))[0])


def test_partition_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(5, 3))
    points = rng.normal(scale=2.0, size=(200, 3))
    total = sum(sq.ssf(points, np.full(len(points), a), centres) for a in range(5))
    np.testing.assert_allclose(total, 1.0, atol=1e-14)


def test_outside_integral_on_a_sphere(sphere: tuple[np.ndarray, ...]) -> None:
    exact = 4.0 * np.pi * quad(lambda t: np.exp(-ALPHA * t * t) * t * t, 4.0, np.inf)[0]
    assert _outside(*sphere) == pytest.approx(exact, rel=1e-4)


def test_ray_adjoint_matches_differences(sphere: tuple[np.ndarray, ...]) -> None:
    xyz, normals, areas, centres = (np.array(a) for a in sphere)
    rays = sq.build_exterior_rays(xyz, normals, areas, centres)
    values, grads = _gaussian(rays.coords, np.zeros(3))
    ds, dn, da, dR = rays.adjoint(xyz, normals, areas, centres, values, grads)

    def diff(array: np.ndarray, index: tuple[int, ...], h: float = H) -> float:
        x0 = array[index]
        array[index] = x0 + h
        plus = _outside(xyz, normals, areas, centres)
        array[index] = x0 - h
        minus = _outside(xyz, normals, areas, centres)
        array[index] = x0
        return (plus - minus) / (2.0 * h)

    for i in (0, 17, 200):
        for k in range(3):
            fd_s, fd_n = diff(xyz, (i, k), H_SURFACE), diff(normals, (i, k), H_SURFACE)
            assert ds[i, k] == pytest.approx(fd_s, rel=1e-6, abs=1e-11)
            assert dn[i, k] == pytest.approx(fd_n, rel=1e-6, abs=1e-11)
        assert da[i] == pytest.approx(diff(areas, (i,), H_SURFACE), rel=1e-6, abs=1e-11)
    # On the sphere these vanish but for the quadrature's error (~1e-6, of an integral ~2), so
    # the differences' rounding (~1e-10 at this step, more with other thread counts) sets
    # the tolerance
    for a in range(3):
        for k in range(3):
            assert dR[a, k] == pytest.approx(diff(centres, (a, k)), rel=1e-4, abs=1e-9)


def test_volume_grid_integral_and_gradient() -> None:
    centres = np.random.default_rng(3).normal(scale=0.8, size=(3, 3))

    def integral(centres: np.ndarray) -> float:
        grid = sq.build_volume_grid(centres)
        return float(grid.weights @ _gaussian(grid.coords, centres[0])[0])

    assert integral(centres) == pytest.approx((np.pi / ALPHA) ** 1.5, rel=1e-3)
    grid = sq.build_volume_grid(centres)
    values, grads = _gaussian(grid.coords, centres[0])
    g = sq.volume_grid_gradient(grid, centres, values, grads)
    g[0] -= (grid.weights[:, None] * grads).sum(axis=0)  # the Gaussian moves with atom 0
    for a in range(3):
        for k in range(3):
            c = centres.copy()
            c[a, k] += H
            plus = integral(c)
            c[a, k] -= 2 * H
            minus = integral(c)
            assert g[a, k] == pytest.approx((plus - minus) / (2 * H), rel=1e-5, abs=1e-8)
