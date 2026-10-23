"""The cavity-surface sum behind the dispersion kernel, and its derivatives, compiled.

`nonelectrostatic.interior_kernel_integral` writes the volume integral

    J(r) = int_{r' in Omega} dr' / (|r - r'|^6 + delta^6)

as a sum over the cavity surface's elements (its docstring has the derivation):

    J(r) = sum_i A_i k(s_i - r, n_i),   k(R, n) = (R . n) phi(|R|),
    phi(R) = arctan(t) / (3 delta^3 R^3),   t = R^3 / delta^3.

The dispersion integral needs J at every point of the cavity's interior grid,
hundreds of thousands of points against a couple of thousand elements.  Written
with numpy temporaries that pair loop took longer than a vDZP SCF; here it is one
compiled pass, parallel over points.

`kernel_derivatives` returns, from the same pass, everything the nuclear gradient
needs of E = sum_p c_p J(r_p) for given point weights c: dJ/dr at each point
(the points move with their atoms), and dE/ds_i and dE/dA_i for each element
(the elements move with their atoms, and their areas change through the
switching function).  On PySCF's cavity the outward normals are those of a
sphere's Lebedev directions, fixed as the atoms move; on a DROP cavity they move
too, and `kernel_surface_derivatives` adds dE/dn_i.

Everything is in bohr.
"""

from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Below this t = R^3/delta^3, arctan(t)/t and its derivative are taken from
# their series (to t^6; the next term is below 1e-16 there), where the direct
# formulas would cancel.
_T_SERIES = 1e-2


@numba.njit(cache=True, inline="always")
def _pair(rx: float, ry: float, rz: float, nx: float, ny: float, nz: float, d3: float) -> tuple:  # type: ignore[type-arg]
    """k(R, n) * 3 delta^6, grad_R k * 3 delta^6 and arctan(t)/t, for R = (rx, ry, rz)
    (grad_n k * 3 delta^6 = R arctan(t)/t)."""
    r2 = rx * rx + ry * ry + rz * rz
    t = r2 * np.sqrt(r2) / d3
    t2 = t * t
    if t < _T_SERIES:
        atan_t = 1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0
        # 3 (1/(1+t^2) - arctan(t)/t) / R^2, with t^2 / R^2 = R^4 / delta^6
        g_over_r2 = (
            3.0 * (r2 * r2 / (d3 * d3)) * (-2.0 / 3.0 + 4.0 * t2 / 5.0 - 6.0 * t2 * t2 / 7.0)
        )
    else:
        atan_t = np.arctan(t) / t
        g_over_r2 = 3.0 * (1.0 / (1.0 + t2) - atan_t) / r2
    r_dot_n = rx * nx + ry * ny + rz * nz
    k = r_dot_n * atan_t
    s = r_dot_n * g_over_r2
    return k, nx * atan_t + s * rx, ny * atan_t + s * ry, nz * atan_t + s * rz, atan_t


@numba.njit(parallel=True, cache=True)
def _kernel_values(
    points: Array, coords: Array, normals: Array, areas: Array, delta: float
) -> Array:
    n_p, n_s = points.shape[0], coords.shape[0]
    d3 = delta**3
    scale = 1.0 / (3.0 * delta**6)
    out = np.empty(n_p)
    for p in numba.prange(n_p):
        px, py, pz = points[p, 0], points[p, 1], points[p, 2]
        acc = 0.0
        for i in range(n_s):
            k, _gx, _gy, _gz, _at = _pair(
                coords[i, 0] - px,
                coords[i, 1] - py,
                coords[i, 2] - pz,
                normals[i, 0],
                normals[i, 1],
                normals[i, 2],
                d3,
            )
            acc += areas[i] * k
        out[p] = acc * scale
    return out


@numba.njit(parallel=True, cache=True)
def _kernel_derivatives(
    points: Array,
    coords: Array,
    normals: Array,
    areas: Array,
    delta: float,
    weights: Array,
    n_chunks: int,
) -> tuple[Array, Array, Array, Array, Array]:
    n_p, n_s = points.shape[0], coords.shape[0]
    d3 = delta**3
    scale = 1.0 / (3.0 * delta**6)
    J = np.empty(n_p)
    dJ = np.empty((n_p, 3))
    # Element sums are accumulated per chunk of points, then added up, so that
    # no two threads write the same element.
    acc = np.zeros((n_chunks, n_s, 7))
    size = (n_p + n_chunks - 1) // n_chunks
    for c in numba.prange(n_chunks):
        for p in range(c * size, min(n_p, (c + 1) * size)):
            px, py, pz = points[p, 0], points[p, 1], points[p, 2]
            cp = weights[p]
            j = 0.0
            gx = 0.0
            gy = 0.0
            gz = 0.0
            for i in range(n_s):
                rx = coords[i, 0] - px
                ry = coords[i, 1] - py
                rz = coords[i, 2] - pz
                k, kx, ky, kz, at = _pair(
                    rx,
                    ry,
                    rz,
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 2],
                    d3,
                )
                a = areas[i]
                j += a * k
                gx += a * kx
                gy += a * ky
                gz += a * kz
                acc[c, i, 0] += cp * a * kx
                acc[c, i, 1] += cp * a * ky
                acc[c, i, 2] += cp * a * kz
                acc[c, i, 3] += cp * k
                acc[c, i, 4] += cp * a * rx * at
                acc[c, i, 5] += cp * a * ry * at
                acc[c, i, 6] += cp * a * rz * at
            J[p] = j * scale
            # R = s - r, so d/dr = -grad_R
            dJ[p, 0] = -gx * scale
            dJ[p, 1] = -gy * scale
            dJ[p, 2] = -gz * scale
    total = acc.sum(axis=0) * scale
    return (
        J,
        dJ,
        np.ascontiguousarray(total[:, :3]),
        np.ascontiguousarray(total[:, 3]),
        np.ascontiguousarray(total[:, 4:]),
    )


def _f64(a: Array, shape_tail: tuple[int, ...] = ()) -> Array:
    out = np.ascontiguousarray(a, dtype=np.float64)
    if out.shape[1:] != shape_tail:
        raise ValueError(f"expected shape (n, {', '.join(map(str, shape_tail))}), got {out.shape}")
    return out


def kernel_integral(
    points: Array, coords: Array, normals: Array, areas: Array, delta: float
) -> Array:
    """J(r) at each point (n_points,), for surface elements at ``coords`` with
    outward unit ``normals`` and quadrature ``areas``."""
    return np.asarray(
        _kernel_values(
            _f64(points, (3,)), _f64(coords, (3,)), _f64(normals, (3,)), _f64(areas), float(delta)
        )
    )


def kernel_derivatives(
    points: Array, coords: Array, normals: Array, areas: Array, delta: float, weights: Array
) -> tuple[Array, Array, Array, Array]:
    """J and the derivatives of E = sum_p weights_p J(r_p).

    Returns (J (n_points,), dJ/dr (n_points, 3), dE/ds (n_elements, 3),
    dE/dA (n_elements,)).
    """
    J, dJ, dE_ds, _dE_dn, dE_dA = kernel_surface_derivatives(
        points, coords, normals, areas, delta, weights
    )
    return J, dJ, dE_ds, dE_dA


def kernel_surface_derivatives(
    points: Array, coords: Array, normals: Array, areas: Array, delta: float, weights: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """`kernel_derivatives` with the normals' channel: (J (n_points,), dJ/dr (n_points, 3),
    dE/ds (n_elements, 3), dE/dn (n_elements, 3), dE/dA (n_elements,))."""
    points = _f64(points, (3,))
    n_chunks = max(1, min(len(points), 4 * numba.get_num_threads()))
    J, dJ, dE_ds, dE_dA, dE_dn = _kernel_derivatives(
        points,
        _f64(coords, (3,)),
        _f64(normals, (3,)),
        _f64(areas),
        float(delta),
        _f64(weights),
        n_chunks,
    )
    return np.asarray(J), np.asarray(dJ), np.asarray(dE_ds), np.asarray(dE_dn), np.asarray(dE_dA)
