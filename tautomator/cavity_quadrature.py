"""Quadrature grids for the inside and the outside of a union-of-spheres cavity.

The dispersion and exchange integrals of `nonelectrostatic` are cut off at the
PCM cavity's surface.  On an ordinary molecular grid that cut falls between
quadrature shells, and the integral converges only linearly in the shell
spacing: for the formate anion at HF/6-31+G* the exchange integral is still 9%
short on a (99, 590) grid and jumps as atoms move.  Here every radial rule ends
exactly on a sphere's surface.

Each atom carries a grid that moves rigidly with it: Lebedev directions, and
along each a radial rule on [0, r_a] for the inside (Gauss-Legendre) or on
[r_a, infinity) for the outside (Gauss-Legendre after the map
t = r_a + L (1 + x)/(1 - x)).  The other spheres enter only through the weights,
a smooth partition of the region among the atoms:

    inside:   P_a = g_a / sum_b g_b,   g_b = beta_b(r) S((r_b - |r - R_b|) / w)
    outside:  Q_a = g_a / sum_b g_b,   g_b = beta_b(r) prod_{c != b} S((|r - R_c| - r_c) / w)

beta_b is Becke's fuzzy cell and S a C^2 step, 0 below 0 and 1 above 1.  Inside,
an atom hands a point over to the others as the point nears its own surface;
outside, an atom gives up a point as it nears another sphere, and has no weight
inside one.  Both are partitions of unity on their region.

An earlier version cut each ray at the spheres it crossed and put a rule on each
stretch.  That is more accurate inside at the same size, but a ray becoming
tangent to a sphere adds or removes a stretch, so the integrals had square-root
kinks as atoms moved: no use for gradients.  Here the integrals are C^2 in the
geometry, and their derivatives with respect to the atoms are those of the
weights, with each atom's points moving with it (`partition_gradient`).

Everything is in bohr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import lebedev_rule

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

Region = Literal["inside", "outside"]

# (Lebedev order, radial nodes per direction) of each region's default grid, from the
# convergence study in docs/nonelectrostatic.md (42 solutes, ions included): together they
# keep the correction within 0.013 kcal/mol of converged.  The outside needs more directions
# than the inside: its error comes from the creases between spheres, and stays at a few
# 1e-2 kcal/mol below order 41 however many radial nodes.
Grid = tuple[int, int]
INSIDE_GRID: Grid = (23, 12)  # 194 directions per atom
OUTSIDE_GRID: Grid = (41, 12)  # 590 directions per atom

# Width (bohr) of the step S over which the partition hands a point from one
# atom to another near a sphere's surface.
STEP_WIDTH = 2.0

# Length scale (bohr) of the map that takes Gauss-Legendre to [r_a, infinity):
# half of the outside rule's nodes lie within this distance of the surface.
EXTERIOR_MAP = 0.5

# Points whose weight is below this are dropped.
_MIN_WEIGHT = 1e-14


@numba.njit(cache=True, inline="always")
def _becke_s(mu: float) -> tuple[float, float]:
    """Becke's cutoff s(mu) and d log s / d mu (3 iterations of p)."""
    p1 = 1.5 * mu - 0.5 * mu**3
    p2 = 1.5 * p1 - 0.5 * p1**3
    p3 = 1.5 * p2 - 0.5 * p2**3
    s = 0.5 * (1.0 - p3)
    if s <= 0.0:
        return 0.0, 0.0
    dp3 = 1.5 * (1.0 - p2 * p2) * 1.5 * (1.0 - p1 * p1) * 1.5 * (1.0 - mu * mu)
    return s, -0.5 * dp3 / s


@numba.njit(cache=True, inline="always")
def _log_step(x: float) -> tuple[float, float]:
    """log S(x) and d log S / dx for S = x^3 (10 - 15 x + 6 x^2) on [0, 1]."""
    if x >= 1.0:
        return 0.0, 0.0
    if x <= 0.0:
        return -np.inf, 0.0
    poly = 10.0 - 15.0 * x + 6.0 * x * x
    return np.log(x**3 * poly), 30.0 * (1.0 - x) ** 2 / (x * poly)


@numba.njit(cache=True, inline="always")
def _logs(
    r: Array,
    centres: Array,
    radii: Array,
    dist_cc: Array,
    inside: bool,
    width: float,
    u: Array,
    rhat: Array,
    L: Array,
) -> None:
    """log g_b at point r for every atom b, with distances and unit vectors to the centres."""
    m = centres.shape[0]
    for c in range(m):
        dx = r[0] - centres[c, 0]
        dy = r[1] - centres[c, 1]
        dz = r[2] - centres[c, 2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        u[c] = d
        if d > 0.0:
            rhat[c, 0] = dx / d
            rhat[c, 1] = dy / d
            rhat[c, 2] = dz / d
        else:
            rhat[c, 0] = 0.0
            rhat[c, 1] = 0.0
            rhat[c, 2] = 0.0
    for b in range(m):
        L[b] = 0.0
    for b in range(m):
        for c in range(m):
            if c != b:
                s, _ = _becke_s((u[b] - u[c]) / dist_cc[b, c])
                L[b] += np.log(s) if s > 0.0 else -np.inf
    if inside:
        for b in range(m):
            ls, _ = _log_step((radii[b] - u[b]) / width)
            L[b] += ls
    else:
        # sum over c != b of log S(outside-ness of c)
        total = 0.0
        n_inf = 0
        which = -1
        for c in range(m):
            ls, _ = _log_step((u[c] - radii[c]) / width)
            if ls == -np.inf:
                n_inf += 1
                which = c
            else:
                total += ls
        for b in range(m):
            if n_inf == 0:
                ls, _ = _log_step((u[b] - radii[b]) / width)
                L[b] += total - ls
            elif n_inf == 1 and b == which:
                L[b] += total
            else:
                L[b] = -np.inf


@numba.njit(cache=True, inline="always")
def _normalize(L: Array, P: Array) -> None:
    """P_b = exp(L_b) / sum_c exp(L_c), safely; all zero if every L is -inf."""
    top = -np.inf
    for b in range(L.shape[0]):
        if L[b] > top:
            top = L[b]
    if top == -np.inf:
        for b in range(L.shape[0]):
            P[b] = 0.0
        return
    total = 0.0
    for b in range(L.shape[0]):
        P[b] = np.exp(L[b] - top)
        total += P[b]
    for b in range(L.shape[0]):
        P[b] /= total


def _pair_distances(centres: Array) -> Array:
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)  # never used; avoids 0/0
    return np.asarray(d)


@numba.njit(parallel=True, cache=True)
def _partition(
    points: Array,
    owners: IntArray,
    centres: Array,
    radii: Array,
    dist_cc: Array,
    inside: bool,
    width: float,
    n_chunks: int,
) -> Array:
    n, m = points.shape[0], centres.shape[0]
    out = np.empty(n)
    size = (n + n_chunks - 1) // n_chunks
    for chunk in numba.prange(n_chunks):
        u = np.empty(m)
        rhat = np.empty((m, 3))
        L = np.empty(m)
        P = np.empty(m)
        for p in range(chunk * size, min(n, (chunk + 1) * size)):
            _logs(points[p], centres, radii, dist_cc, inside, width, u, rhat, L)
            _normalize(L, P)
            out[p] = P[owners[p]]
    return out


def _n_chunks(n: int) -> int:
    return max(1, min(n, 4 * numba.get_num_threads()))


@numba.njit(parallel=True, cache=True)
def _partition_gradient(
    points: Array,
    owners: IntArray,
    centres: Array,
    radii: Array,
    dist_cc: Array,
    inside: bool,
    width: float,
    values: Array,
    n_chunks: int,
) -> Array:
    """sum_p values_p d P_owner(p)(r_p) / d R, the points moving with their owners."""
    n, m = points.shape[0], centres.shape[0]
    acc = np.zeros((n_chunks, m, 3))
    size = (n + n_chunks - 1) // n_chunks
    for chunk in numba.prange(n_chunks):
        u = np.empty(m)
        rhat = np.empty((m, 3))
        L = np.empty(m)
        P = np.empty(m)
        g = acc[chunk]
        for p in range(chunk * size, min(n, (chunk + 1) * size)):
            a = owners[p]
            _logs(points[p], centres, radii, dist_cc, inside, width, u, rhat, L)
            _normalize(L, P)
            if P[a] == 0.0 or values[p] == 0.0:
                continue
            # dP_a = P_a (dL_a - sum_b P_b dL_b): each L_b enters with
            # omega_b = values P_a (delta_ab - P_b).
            for b in range(m):
                omega = values[p] * P[a] * ((1.0 if b == a else 0.0) - P[b])
                if omega == 0.0:
                    continue
                for c in range(m):
                    if c == b:
                        continue
                    D = dist_cc[b, c]
                    mu = (u[b] - u[c]) / D
                    s, dls = _becke_s(mu)
                    if s == 0.0:
                        continue
                    f = omega * dls / D
                    for x in range(3):
                        e = (centres[b, x] - centres[c, x]) / D
                        d_r = f * (rhat[b, x] - rhat[c, x])
                        g[a, x] += d_r  # the point moves with its owner
                        g[b, x] += f * (-rhat[b, x] - mu * e)
                        g[c, x] += f * (rhat[c, x] + mu * e)
                if inside:
                    _, dls = _log_step((radii[b] - u[b]) / width)
                    f = omega * dls / width
                    for x in range(3):
                        g[a, x] -= f * rhat[b, x]
                        g[b, x] += f * rhat[b, x]
                else:
                    for c in range(m):
                        if c == b:
                            continue
                        _, dls = _log_step((u[c] - radii[c]) / width)
                        f = omega * dls / width
                        for x in range(3):
                            g[a, x] += f * rhat[c, x]
                            g[c, x] -= f * rhat[c, x]
    return acc.sum(axis=0)


@dataclass(frozen=True)
class CavityGrid:
    """A quadrature grid over the inside or outside of the cavity.

    ``coords[p] = atom_coords[owners[p]] + radial[p] * direction``; the point
    moves rigidly with its owner, and ``weights = base_weights * partition``.
    """

    coords: Array  # (n, 3)
    weights: Array  # (n,)
    owners: IntArray  # (n,), the atom whose rays the point is on
    base_weights: Array  # (n,), radial x angular weight, before the partition
    region: Region


def _rules(region: Region, radius: float, n_radial: int) -> tuple[Array, Array]:
    """Radial nodes t and weights (with the t^2 Jacobian) on one ray."""
    x, w = np.polynomial.legendre.leggauss(n_radial)
    if region == "inside":
        t = 0.5 * radius * (x + 1.0)
        wt = 0.5 * radius * w
    else:
        t = radius + EXTERIOR_MAP * (1.0 + x) / (1.0 - x)
        wt = 2.0 * EXTERIOR_MAP * w / (1.0 - x) ** 2
    return t, wt * t**2


def build_cavity_grid(
    atom_coords: Array,
    atom_radii: Array,
    region: Region,
    lebedev_order: int | None = None,
    n_radial: int | None = None,
) -> CavityGrid:
    """The grid over the inside or outside of the union of spheres (region's default if None)."""
    default = INSIDE_GRID if region == "inside" else OUTSIDE_GRID
    lebedev_order = default[0] if lebedev_order is None else lebedev_order
    n_radial = default[1] if n_radial is None else n_radial
    centres = np.ascontiguousarray(atom_coords, dtype=np.float64)
    radii = np.ascontiguousarray(atom_radii, dtype=np.float64)
    directions, w_ang = lebedev_rule(lebedev_order)
    unit = np.ascontiguousarray(directions.T)
    coords, base, owners = [], [], []
    for a, (centre, radius) in enumerate(zip(centres, radii, strict=True)):
        t, wt = _rules(region, float(radius), n_radial)
        coords.append((centre + t[:, None, None] * unit[None, :, :]).reshape(-1, 3))
        base.append((wt[:, None] * w_ang[None, :]).ravel())
        owners.append(np.full(len(t) * len(unit), a, dtype=np.intp))
    points = np.vstack(coords)
    base_weights = np.concatenate(base)
    owner = np.concatenate(owners)
    partition = _partition(
        points,
        owner,
        centres,
        radii,
        _pair_distances(centres),
        region == "inside",
        STEP_WIDTH,
        _n_chunks(len(points)),
    )
    weights = base_weights * partition
    keep = weights > _MIN_WEIGHT
    return CavityGrid(
        coords=points[keep],
        weights=weights[keep],
        owners=owner[keep],
        base_weights=base_weights[keep],
        region=region,
    )


def cavity_grid(
    atom_coords: Array,
    atom_radii: Array,
    region: Region,
    lebedev_order: int | None = None,
    n_radial: int | None = None,
) -> tuple[Array, Array]:
    """Points (n, 3) and weights (n,) integrating over the inside or outside of the cavity."""
    grid = build_cavity_grid(atom_coords, atom_radii, region, lebedev_order, n_radial)
    return grid.coords, grid.weights


def partition_gradient(
    grid: CavityGrid, atom_coords: Array, atom_radii: Array, values: Array
) -> Array:
    """d/dR of sum_p values_p weights_p, the integrand values held fixed: (n_atoms, 3).

    This is the weights' part of the derivative of a quadrature: the points move
    with their owners, and the partition changes with every atom's position.  The
    integrand's own change (the points moving through it, and anything else it
    depends on) is the caller's.
    """
    centres = np.ascontiguousarray(atom_coords, dtype=np.float64)
    radii = np.ascontiguousarray(atom_radii, dtype=np.float64)
    return np.asarray(
        _partition_gradient(
            grid.coords,
            grid.owners,
            centres,
            radii,
            _pair_distances(centres),
            grid.region == "inside",
            STEP_WIDTH,
            np.ascontiguousarray(grid.base_weights * values, dtype=np.float64),
            _n_chunks(len(grid.coords)),
        )
    )


def becke_partition(points: Array, atom_coords: Array) -> Array:
    """Becke's fuzzy-cell weights, shape (n_points, n_atoms); each row sums to 1.

    The plain scheme (JCP 88, 2547 (1988)), with three iterations of the cutoff
    polynomial and no atomic-size adjustment.
    """
    dist = np.linalg.norm(points[:, None, :] - atom_coords[None, :, :], axis=2)
    cell = np.ones_like(dist)
    for i in range(len(atom_coords)):
        for j in range(i):
            mu = (dist[:, i] - dist[:, j]) / np.linalg.norm(atom_coords[i] - atom_coords[j])
            for _ in range(3):
                mu = 1.5 * mu - 0.5 * mu**3
            cell[:, i] *= 0.5 * (1.0 - mu)
            cell[:, j] *= 0.5 * (1.0 + mu)
    return np.asarray(cell / cell.sum(axis=1, keepdims=True))
