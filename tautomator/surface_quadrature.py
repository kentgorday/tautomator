"""Volume integrals bounded by a discretized closed surface, by the divergence theorem.

The nonelectrostatic terms integrate over the inside or the outside of the cavity.  On
MOIST's DROP cavities the surface is all there is: points s_i with outward unit normals n_i
and quadrature areas a_i, smooth in the geometry (and, for rho-DROP, in the density), with
MOIST's adjoint taking derivatives with respect to them back to the nuclei and the density.
So the volume integrals are written on that surface.

For a nucleus at c and a function q, the field

    F(x) = e / r^2  int_r^inf q(c + t e) t^2 dt,    r = |x - c|,  e = (x - c) / r

has div F = -q everywhere but at c, and vanishes at infinity, so for a cavity that holds c

    int_outside q dV = sum_i a_i n_i . F(s_i).

Each atom A takes its share q_A = p_A g of the integrand g through a partition of unity
p_A (`ssf`), with its own nucleus as c: near another nucleus p_A vanishes, so no ray sees a
nuclear cusp.  The radial integral is Gauss-Legendre after t = r + L (1 + x)/(1 - x), so the
exterior integral is a weighted point set: nodes y = s_i + tau_k e_iA with weights

    w = a_i (n_i . e_iA) ((r + tau_k)/r)^2 w_k p_A(y),

over the pairs (i, A) whose ray carries any of A's share.  The partition is
Stratmann-Scuseria-Frisch's, which is exactly zero once an atom's cell is left; along a ray
leaving its own nucleus an atom's cell factor never grows, so a pair whose surface point
already lies outside A's cell is dropped exactly (and smoothly: the factor is C^3).

Inside integrals are all-space integrals (an atom-centred grid with the same partition,
moving rigidly with the atoms: `build_volume_grid`) less the outside.

`ExteriorRays.adjoint` takes the derivatives of sum_k w_k G(y_k), for given values G and
gradients dG/dy, back to the surface (positions, normals, areas), the nuclei (each ray's
origin and the partition) and nothing else; the caller adds what G itself depends on.

Everything is in bohr.  The surface's quadrature limits the accuracy: on acetate (rho-DROP)
the outside integrals of rho and |grad rho| match a brute-force integral over rho < rho_iso
to 1e-5 relative at MOIST's nleb 1202 and 1.3e-3 at its default 194.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import lebedev_rule

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

# Stratmann-Scuseria-Frisch's a: an atom's cell factor against a neighbour falls from 1 to 0
# over -a < mu < a, mu the hyperbolic coordinate between the two.
SSF_A = 0.64

# (Lebedev order, radial nodes, map length in bohr) of the all-space grid, and (radial nodes,
# map length in bohr) of the exterior rays.
VolumeRule = tuple[int, int, float]
RayRule = tuple[int, float]
VOLUME_RULE: VolumeRule = (23, 16, 1.5)
RAY_RULE: RayRule = (12, 1.0)


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


@numba.njit(cache=True, inline="always")
def _ssf_s(mu: float, a: float) -> tuple[float, float]:
    """s(mu) = (1 - g(mu/a))/2 and ds/dmu; g the SSF polynomial, +-1 beyond +-a."""
    x = mu / a
    if x >= 1.0:
        return 0.0, 0.0
    if x <= -1.0:
        return 1.0, 0.0
    x2 = x * x
    g = (35.0 * x - 35.0 * x * x2 + 21.0 * x * x2 * x2 - 5.0 * x * x2 * x2 * x2) / 16.0
    one = 1.0 - x2
    dg = 35.0 * one * one * one / 16.0
    return 0.5 * (1.0 - g), -0.5 * dg / a


@numba.njit(cache=True, inline="always")
def _cells(
    y0: float, y1: float, y2: float, centres: Array, inv_rab: Array, a: float, d: Array, P: Array
) -> float:
    """Unnormalized cell factors P_C(y) for every atom, and their sum; d gets |y - c_C|."""
    m = centres.shape[0]
    for c in range(m):
        dx = y0 - centres[c, 0]
        dy = y1 - centres[c, 1]
        dz = y2 - centres[c, 2]
        d[c] = np.sqrt(dx * dx + dy * dy + dz * dz)
    z = 0.0
    for c in range(m):
        prod = 1.0
        for b in range(m):
            if b != c:
                s, _ = _ssf_s((d[c] - d[b]) * inv_rab[c, b], a)
                prod *= s
                if prod == 0.0:
                    break
        P[c] = prod
        z += prod
    return z


@numba.njit(parallel=True, cache=True)
def _ssf_weights(
    points: Array, owners: IntArray, centres: Array, inv_rab: Array, a: float
) -> Array:
    n, m = points.shape[0], centres.shape[0]
    out = np.zeros(n)
    for p in numba.prange(n):
        d = np.empty(m)
        P = np.empty(m)
        z = _cells(points[p, 0], points[p, 1], points[p, 2], centres, inv_rab, a, d, P)
        if z > 0.0:
            out[p] = P[owners[p]] / z
    return out


@numba.njit(cache=True)
def _cell_factor(y: Array, atom: int, centres: Array, inv_rab: Array, a: float) -> float:
    """P_atom(y), unnormalized."""
    prod = 1.0
    da = np.sqrt(
        (y[0] - centres[atom, 0]) ** 2
        + (y[1] - centres[atom, 1]) ** 2
        + (y[2] - centres[atom, 2]) ** 2
    )
    for b in range(centres.shape[0]):
        if b != atom:
            db = np.sqrt(
                (y[0] - centres[b, 0]) ** 2
                + (y[1] - centres[b, 1]) ** 2
                + (y[2] - centres[b, 2]) ** 2
            )
            s, _ = _ssf_s((da - db) * inv_rab[atom, b], a)
            prod *= s
            if prod == 0.0:
                return 0.0
    return prod


@numba.njit(parallel=True, cache=True)
def _ssf_gradient(
    points: Array,
    owners: IntArray,
    centres: Array,
    inv_rab: Array,
    a: float,
    coeff: Array,
    n_chunks: int,
) -> tuple[Array, Array]:
    """For E = sum_p coeff_p p_{owner(p)}(y_p): dE/dR (natm, 3) at fixed points, dE/dy (n, 3).

    dp_A = (dP_A - p_A dZ) / Z and dP_C = P_C sum_B (s'/s)(mu_CB) dmu_CB, zero where P_C is
    (a vanishing factor has a vanishing slope: the SSF polynomial is flat at +-a).
    """
    n, m = points.shape[0], centres.shape[0]
    g_atoms = np.zeros((n_chunks, m, 3))
    g_points = np.zeros((n, 3))
    size = (n + n_chunks - 1) // n_chunks
    for ch in numba.prange(n_chunks):
        d = np.empty(m)
        P = np.empty(m)
        e = np.empty((m, 3))
        for p in range(ch * size, min(n, (ch + 1) * size)):
            if coeff[p] == 0.0:
                continue
            y0, y1, y2 = points[p, 0], points[p, 1], points[p, 2]
            z = _cells(y0, y1, y2, centres, inv_rab, a, d, P)
            if z <= 0.0:
                continue
            own = owners[p]
            pA = P[own] / z
            for c in range(m):
                if d[c] > 0.0:
                    e[c, 0] = (y0 - centres[c, 0]) / d[c]
                    e[c, 1] = (y1 - centres[c, 1]) / d[c]
                    e[c, 2] = (y2 - centres[c, 2]) / d[c]
                else:
                    e[c, 0] = 0.0
                    e[c, 1] = 0.0
                    e[c, 2] = 0.0
            for c in range(m):
                if P[c] == 0.0:
                    continue
                # weight of dP_C in dE: coeff (delta_{C,own} - p_A) / Z
                wc = coeff[p] * ((1.0 if c == own else 0.0) - pA) / z * P[c]
                if wc == 0.0:
                    continue
                for b in range(m):
                    if b == c:
                        continue
                    mu = (d[c] - d[b]) * inv_rab[c, b]
                    s, ds = _ssf_s(mu, a)
                    if ds == 0.0:
                        continue
                    f = wc * ds / s  # times dmu_CB
                    ir = inv_rab[c, b]
                    # dmu/dy = (e_C - e_B)/R_CB;  dmu/dc_C = -e_C/R - mu (c_C - c_B)/R^2;
                    # dmu/dc_B = e_B/R + mu (c_C - c_B)/R^2
                    for k in range(3):
                        sep = (centres[c, k] - centres[b, k]) * ir * ir
                        g_points[p, k] += f * (e[c, k] - e[b, k]) * ir
                        g_atoms[ch, c, k] += f * (-e[c, k] * ir - mu * sep)
                        g_atoms[ch, b, k] += f * (e[b, k] * ir + mu * sep)
    return g_atoms.sum(axis=0), g_points


def _inverse_distances(centres: Array) -> Array:
    sep = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(sep, 1.0)
    out = 1.0 / sep
    np.fill_diagonal(out, 0.0)
    return np.asarray(out)


def _n_chunks(n: int) -> int:
    return max(1, min(n, 4 * numba.get_num_threads()))


def ssf(points: Array, owners: IntArray, centres: Array, a: float = SSF_A) -> Array:
    """p_{owner}(y) at each point: the owner atom's share of the partition of unity."""
    centres = np.ascontiguousarray(centres, dtype=np.float64)
    return np.asarray(
        _ssf_weights(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(owners, dtype=np.intp),
            centres,
            _inverse_distances(centres),
            float(a),
        )
    )


def ssf_gradient(
    points: Array, owners: IntArray, centres: Array, coeff: Array, a: float = SSF_A
) -> tuple[Array, Array]:
    """E = sum_p coeff_p p_{owner(p)}(y_p): (dE/dR at fixed points (natm, 3), dE/dy (n, 3))."""
    centres = np.ascontiguousarray(centres, dtype=np.float64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    gR, gy = _ssf_gradient(
        points,
        np.ascontiguousarray(owners, dtype=np.intp),
        centres,
        _inverse_distances(centres),
        float(a),
        np.ascontiguousarray(coeff, dtype=np.float64),
        _n_chunks(len(points)),
    )
    return np.asarray(gR), np.asarray(gy)


# ---------------------------------------------------------------------------
# Radial rules
# ---------------------------------------------------------------------------


def mapped_legendre(n: int, length: float) -> tuple[Array, Array]:
    """Gauss-Legendre on [0, infinity) after t = L (1 + x)/(1 - x): nodes and weights."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = length * (1.0 + x) / (1.0 - x)
    return np.asarray(t), np.asarray(w * 2.0 * length / (1.0 - x) ** 2)


# ---------------------------------------------------------------------------
# All-space grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolumeGrid:
    """An all-space grid: each atom's points move rigidly with it.

    weights = base_weights * p_owner(coords); points of zero weight are left out.
    """

    coords: Array  # (n, 3)
    weights: Array  # (n,)
    owners: IntArray  # (n,)
    base_weights: Array  # (n,): Lebedev x radial weights, without the partition


def build_volume_grid(centres: Array, rule: VolumeRule = VOLUME_RULE) -> VolumeGrid:
    """Lebedev directions x mapped Gauss-Legendre radii around every atom, SSF-partitioned."""
    order, n_radial, length = rule
    directions, w_ang = lebedev_rule(order)
    t, w_rad = mapped_legendre(n_radial, length)
    local = (t[:, None, None] * directions.T[None]).reshape(-1, 3)
    w_local = (w_rad[:, None] * t[:, None] ** 2 * w_ang[None]).ravel()
    centres = np.asarray(centres, dtype=np.float64)
    m = len(centres)
    coords = (centres[:, None, :] + local[None]).reshape(-1, 3)
    owners = np.repeat(np.arange(m), len(local))
    base = np.tile(w_local, m)
    p = ssf(coords, owners, centres)
    keep = p > 0.0
    return VolumeGrid(
        coords=np.ascontiguousarray(coords[keep]),
        weights=np.ascontiguousarray(base[keep] * p[keep]),
        owners=np.ascontiguousarray(owners[keep]),
        base_weights=np.ascontiguousarray(base[keep]),
    )


def volume_grid_gradient(grid: VolumeGrid, centres: Array, values: Array, dvalues: Array) -> Array:
    """d/dR of sum_p weights_p v_p, with v's spatial gradient dvalues (n, 3) carried by each
    point moving with its owner; v's other dependence on the geometry is the caller's."""
    gR, gy = ssf_gradient(grid.coords, grid.owners, centres, grid.base_weights * values)
    gy = gy + grid.weights[:, None] * dvalues
    m = len(centres)
    for x in range(3):
        gR[:, x] += np.bincount(grid.owners, gy[:, x], minlength=m)
    return gR


# ---------------------------------------------------------------------------
# Exterior rays
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _active_pairs(points: Array, centres: Array, inv_rab: Array, a: float) -> NDArray[np.bool_]:
    """Which atoms' cells hold each surface point: (npts, natm) booleans."""
    n, m = points.shape[0], centres.shape[0]
    out = np.zeros((n, m), dtype=np.bool_)
    for i in numba.prange(n):
        y = points[i]
        for A in range(m):
            out[i, A] = _cell_factor(y, A, centres, inv_rab, a) > 0.0
    return out


@dataclass(frozen=True)
class ExteriorRays:
    """The outside of a closed surface as a weighted point set (module docstring).

    Node k lies on the ray of surface point ``point[k]`` from nucleus ``atom[k]``, at
    ``tau[k]`` beyond the surface; ``weights`` include the partition.
    """

    coords: Array  # (n, 3) nodes y
    weights: Array  # (n,)
    point: IntArray  # (n,) surface point of each node
    atom: IntArray  # (n,) nucleus of each node's ray
    tau: Array  # (n,) distance beyond the surface
    w_radial: Array  # (n,) radial weight
    partition: Array  # (n,) p_atom(y)
    n_surface: int

    def adjoint(
        self,
        surface_xyz: Array,
        normals: Array,
        areas: Array,
        centres: Array,
        values: Array,
        dvalues: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Derivatives of E = sum_k weights_k G_k with G at the nodes and dG/dy (n, 3).

        Returns (dE/ds (npts, 3), dE/dn (npts, 3), dE/da (npts,), dE/dR (natm, 3)): with
        respect to the surface points, normals and areas, and the nuclei as ray origins and
        through the partition.  G's own dependence on the geometry is the caller's.
        """
        centres = np.ascontiguousarray(centres, dtype=np.float64)
        base = self.weights / np.where(self.partition > 0.0, self.partition, 1.0)
        gR, gy = ssf_gradient(self.coords, self.atom, centres, base * values)
        gy = gy + self.weights[:, None] * dvalues
        ds, dn, da, dc = _ray_adjoint(
            np.ascontiguousarray(surface_xyz, dtype=np.float64),
            np.ascontiguousarray(normals, dtype=np.float64),
            np.ascontiguousarray(areas, dtype=np.float64),
            centres,
            self.point,
            self.atom,
            self.tau,
            self.w_radial,
            self.partition,
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(gy),
            _n_chunks(len(self.point)),
        )
        return np.asarray(ds), np.asarray(dn), np.asarray(da), np.asarray(gR + dc)


@numba.njit(parallel=True, cache=True)
def _ray_nodes(
    xyz: Array,
    normals: Array,
    areas: Array,
    centres: Array,
    pairs_i: IntArray,
    pairs_A: IntArray,
    tau: Array,
    w_rad: Array,
) -> tuple[Array, Array]:
    """Nodes y and base weights a (n.e) ((r + tau)/r)^2 w_k, pair-major, radial-minor."""
    n_pairs, n_rad = pairs_i.shape[0], tau.shape[0]
    y = np.empty((n_pairs * n_rad, 3))
    base = np.empty(n_pairs * n_rad)
    for q in numba.prange(n_pairs):
        i, A = pairs_i[q], pairs_A[q]
        rx = xyz[i, 0] - centres[A, 0]
        ry = xyz[i, 1] - centres[A, 1]
        rz = xyz[i, 2] - centres[A, 2]
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        ex, ey, ez = rx / r, ry / r, rz / r
        ne = normals[i, 0] * ex + normals[i, 1] * ey + normals[i, 2] * ez
        for k in range(n_rad):
            j = q * n_rad + k
            y[j, 0] = xyz[i, 0] + tau[k] * ex
            y[j, 1] = xyz[i, 1] + tau[k] * ey
            y[j, 2] = xyz[i, 2] + tau[k] * ez
            s = (r + tau[k]) / r
            base[j] = areas[i] * ne * s * s * w_rad[k]
    return y, base


@numba.njit(parallel=True, cache=True)
def _ray_adjoint(
    xyz: Array,
    normals: Array,
    areas: Array,
    centres: Array,
    point: IntArray,
    atom: IntArray,
    tau: Array,
    w_rad: Array,
    partition: Array,
    values: Array,
    gy: Array,
    n_chunks: int,
) -> tuple[Array, Array, Array, Array]:
    """Chain dE/dy (gy, partition and integrand) and the base weight's dependence on
    (a, n, s, c) through y = s + tau e and b = a (n.e) (1 + tau/r)^2 w."""
    n_nodes = point.shape[0]
    ns, m = xyz.shape[0], centres.shape[0]
    ds = np.zeros((n_chunks, ns, 3))
    dn = np.zeros((n_chunks, ns, 3))
    da = np.zeros((n_chunks, ns))
    dc = np.zeros((n_chunks, m, 3))
    size = (n_nodes + n_chunks - 1) // n_chunks
    for ch in numba.prange(n_chunks):
        for k in range(ch * size, min(n_nodes, (ch + 1) * size)):
            i, A = point[k], atom[k]
            R0 = xyz[i, 0] - centres[A, 0]
            R1 = xyz[i, 1] - centres[A, 1]
            R2 = xyz[i, 2] - centres[A, 2]
            r = np.sqrt(R0 * R0 + R1 * R1 + R2 * R2)
            e0, e1, e2 = R0 / r, R1 / r, R2 / r
            n0, n1, n2 = normals[i, 0], normals[i, 1], normals[i, 2]
            ne = n0 * e0 + n1 * e1 + n2 * e2
            t = tau[k]
            s = 1.0 + t / r
            Gp = values[k] * partition[k]  # dE/db
            a = areas[i]
            w = w_rad[k]
            # b = a ne s^2 w
            da[ch, i] += Gp * ne * s * s * w
            dn[ch, i, 0] += Gp * a * e0 * s * s * w
            dn[ch, i, 1] += Gp * a * e1 * s * s * w
            dn[ch, i, 2] += Gp * a * e2 * s * s * w
            # db/dR (R = s_i - c): a w [s^2 (n - ne e)/r - 2 ne s t e / r^2]
            c1 = a * w * s * s / r
            c2 = a * w * (s * s * ne / r + 2.0 * ne * s * t / (r * r))
            db0 = c1 * n0 - c2 * e0
            db1 = c1 * n1 - c2 * e1
            db2 = c1 * n2 - c2 * e2
            # y = s_i + t e: dy/ds_i = I + t P/r, dy/dc = -t P/r, P = I - e e^T
            g0, g1, g2 = gy[k, 0], gy[k, 1], gy[k, 2]
            ge = g0 * e0 + g1 * e1 + g2 * e2
            f = t / r
            p0 = f * (g0 - ge * e0)
            p1 = f * (g1 - ge * e1)
            p2 = f * (g2 - ge * e2)
            ds[ch, i, 0] += g0 + p0 + Gp * db0
            ds[ch, i, 1] += g1 + p1 + Gp * db1
            ds[ch, i, 2] += g2 + p2 + Gp * db2
            dc[ch, A, 0] += -p0 - Gp * db0
            dc[ch, A, 1] += -p1 - Gp * db1
            dc[ch, A, 2] += -p2 - Gp * db2
    return ds.sum(axis=0), dn.sum(axis=0), da.sum(axis=0), dc.sum(axis=0)


def build_exterior_rays(
    surface_xyz: Array,
    normals: Array,
    areas: Array,
    centres: Array,
    rule: RayRule = RAY_RULE,
    a: float = SSF_A,
) -> ExteriorRays:
    """The outside of the surface as weighted nodes on rays from the nuclei (module docstring)."""
    surface_xyz = np.ascontiguousarray(surface_xyz, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    areas = np.ascontiguousarray(areas, dtype=np.float64)
    centres = np.ascontiguousarray(centres, dtype=np.float64)
    inv_rab = _inverse_distances(centres)
    active = _active_pairs(surface_xyz, centres, inv_rab, float(a))
    pairs_i, pairs_A = np.nonzero(active)
    pairs_i = np.ascontiguousarray(pairs_i, dtype=np.intp)
    pairs_A = np.ascontiguousarray(pairs_A, dtype=np.intp)
    n_radial, length = rule
    tau, w_rad = mapped_legendre(n_radial, length)
    y, base = _ray_nodes(surface_xyz, normals, areas, centres, pairs_i, pairs_A, tau, w_rad)
    point = np.repeat(pairs_i, n_radial)
    atom = np.repeat(pairs_A, n_radial)
    p = np.asarray(_ssf_weights(y, atom, centres, inv_rab, float(a)))
    keep = p > 0.0
    return ExteriorRays(
        coords=np.ascontiguousarray(y[keep]),
        weights=np.ascontiguousarray(base[keep] * p[keep]),
        point=np.ascontiguousarray(point[keep]),
        atom=np.ascontiguousarray(atom[keep]),
        tau=np.ascontiguousarray(np.tile(tau, len(pairs_i))[keep]),
        w_radial=np.ascontiguousarray(np.tile(w_rad, len(pairs_i))[keep]),
        partition=np.ascontiguousarray(p[keep]),
        n_surface=len(areas),
    )
