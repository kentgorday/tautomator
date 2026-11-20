"""Symmetry-aware conformer deduplication and partition-function weights.

A conformer search returns structures, but the partition function is a sum over
*states*.  The two differ whenever a molecule has identical atoms: relabelling
them produces a different coordinate array that is the same physical state, and
counting it twice inflates the ensemble.

Everything here works from coordinates and element symbols alone.  No molecular
graph is needed, and none is perceived: a permutation of like atoms that
preserves every interatomic distance is exactly a relabelling that leaves the
structure invariant, and bonds are only short distances, so connectivity is
preserved automatically.  That matters because bond orders and formal charges
cannot be perceived reliably for the charged, tautomeric species this pipeline
handles, and because a SMILES label goes stale the moment a proton migrates
during optimisation.

Two entry points:

* :func:`deduplicate` groups an ensemble into distinct states.
* :func:`conformer_multiplicity` gives each survivor its weight in the
  partition function.

Call them in that order.  Multiplicity applied to a list that still contains
duplicates counts the same state twice and then doubles one of the copies.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# CREGEN's thresholds, kept so behaviour matches the sampling stage upstream.
DEFAULT_RTHR = 0.125  # Angstrom
DEFAULT_ETHR = 0.05  # kcal/mol

KB_HARTREE = 3.1668115634556e-6  # Boltzmann constant, Hartree/K

# How far a pairwise distance may move before a candidate mapping is discarded.
# Loosening this costs time and never accuracy, because survivors are verified
# exactly. Tightening it can lose a real duplicate, which is the direction that
# matters: see "Tolerances" in docs/symmetry.md for where the guarantee ends.
DIST_TOL = 0.8  # Angstrom

# Cutoff for the descriptor screen in _screen_pairs, same reasoning.
DESC_TOL = 0.8  # Angstrom

# Ceiling on enumerated mappings. Reaching it truncates the search, which can
# undercount sigma or miss the best mapping, so it is reported rather than
# absorbed. Real molecules are orders of magnitude below it.
MAX_MAPPINGS = 20_000


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------


def _center(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    centered: NDArray[np.float64] = coords - coords.mean(axis=-2, keepdims=True)
    return centered


def _kabsch_rmsd(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """RMSD under the optimal *proper* rotation, for matched stacks of structures.

    The determinant guard forbids improper superpositions, so a structure is
    never reported as matching its own mirror image.  That distinction is the
    whole basis for treating enantiomeric conformers as separate states.
    """
    if len(a) == 0:
        return np.zeros(0)
    n = a.shape[1]
    cov = np.einsum("mai,maj->mij", a, b)
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    s = s.copy()
    s[:, -1] *= d
    g = (a * a).sum(axis=(1, 2)) + (b * b).sum(axis=(1, 2))
    rmsd: NDArray[np.float64] = np.sqrt(np.maximum(g - 2.0 * s.sum(axis=1), 0.0) / n)
    return rmsd


def _rotations(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Proper rotations ``R[k]`` that bring ``b[k] @ R[k].T`` closest to ``a[k]``.

    For matched stacks of centred structures, shape (k, n, 3); proper for the
    reason `_kabsch_rmsd` gives.
    """
    u, _, vt = np.linalg.svd(np.einsum("kni,knj->kij", a, b))
    d = np.ones((len(a), 3))
    d[:, 2] = np.sign(np.linalg.det(u @ vt))
    rot: NDArray[np.float64] = u @ (d[:, :, None] * vt)
    return rot


def superpose(mobile: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """``mobile`` turned by the proper rotation that lays it closest onto ``target``.

    Both centred, atoms matched row by row.
    """
    moved: NDArray[np.float64] = mobile @ _rotations(target[None], mobile[None])[0].T
    return moved


# ---------------------------------------------------------------------------
# The candidate search
# ---------------------------------------------------------------------------


def _distance_matrix(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    dist: NDArray[np.float64] = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return dist


def _signatures(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-atom invariant: its distances to every other atom, sorted.

    Sorting makes it blind to labelling and distances make it blind to
    orientation, so two atoms some symmetry could interchange always agree.
    """
    return np.sort(_distance_matrix(coords), axis=1)


def _candidate_classes(
    sig_a: NDArray[np.float64],
    sig_b: NDArray[np.float64],
    symbols: list[str],
    tol: float,
) -> list[NDArray[np.int64]] | None:
    """For each atom of A, the atoms of B it could stand in for."""
    out: list[NDArray[np.int64]] = []
    same_element = {s: np.array([t == s for t in symbols]) for s in set(symbols)}
    for i, s in enumerate(symbols):
        close = np.abs(sig_b - sig_a[i]).max(axis=1) <= tol
        cand = np.flatnonzero(close & same_element[s])
        if len(cand) == 0:
            return None
        out.append(cand)
    return out


def _search(
    d_a: NDArray[np.float64],
    d_b: NDArray[np.float64],
    classes: list[NDArray[np.int64]],
    tol: float,
    limit: int,
) -> list[NDArray[np.int64]]:
    """Backtracking search for distance-preserving bijections A -> B.

    Atoms are visited most-constrained-first and each trial assignment is
    checked against every atom already placed, so a branch dies on its first
    inconsistent distance rather than being explored.  That is what keeps this
    tractable when the raw permutation count is astronomical.
    """
    n = len(classes)
    order = sorted(range(n), key=lambda i: len(classes[i]))
    mapping = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    found: list[NDArray[np.int64]] = []

    def step(depth: int) -> None:
        if len(found) >= limit:
            return
        if depth == n:
            found.append(mapping.copy())
            return
        i = order[depth]
        placed = order[:depth]
        for j in classes[i]:
            if used[j]:
                continue
            if placed and np.abs(d_a[i, placed] - d_b[j, mapping[placed]]).max() > tol:
                continue
            mapping[i] = j
            used[j] = True
            step(depth + 1)
            used[j] = False
            mapping[i] = -1

    step(0)
    return found


def _candidate_mappings(
    coords_a: NDArray[np.float64],
    coords_b: NDArray[np.float64],
    symbols: list[str],
    tol: float = DIST_TOL,
    limit: int = MAX_MAPPINGS,
) -> list[NDArray[np.int64]]:
    classes = _candidate_classes(_signatures(coords_a), _signatures(coords_b), symbols, tol)
    if classes is None:
        return []
    found = _search(_distance_matrix(coords_a), _distance_matrix(coords_b), classes, tol, limit)
    if len(found) >= limit:
        log.warning(
            f"mapping search stopped at its ceiling of {limit} on a "
            f"{len(symbols)}-atom structure: the enumeration is truncated, so a "
            f"symmetry number may be undercounted or a duplicate missed"
        )
    return found


def best_rmsd(
    coords_a: NDArray[np.float64],
    coords_b: NDArray[np.float64],
    symbols: list[str],
    tol: float = DIST_TOL,
    mirror: bool = False,
) -> float:
    """Minimum RMSD over every geometrically plausible relabelling.

    Superposition is always proper.  ``mirror=True`` reflects B first, which
    tests whether the two are enantiomeric conformers rather than the same
    structure relabelled.  Returns ``inf`` when no relabelling is consistent
    with the geometry at all.
    """
    b = coords_b * np.array([1.0, 1.0, -1.0]) if mirror else coords_b
    maps = _candidate_mappings(coords_a, b, symbols, tol)
    if not maps:
        return np.inf
    xa = _center(coords_a)
    xb = _center(b)
    # mapping[i] = j means atom i of A corresponds to atom j of B, so row i of
    # the comparison stack is xb[mapping[i]], not the reverse permutation.
    return float(
        _kabsch_rmsd(np.repeat(xa[None], len(maps), axis=0), np.stack([xb[m] for m in maps])).min()
    )


def best_max_distance(
    coords_a: NDArray[np.float64],
    coords_b: NDArray[np.float64],
    symbols: list[str],
    tol: float = DIST_TOL,
    mirror: bool = False,
) -> float:
    """`best_rmsd`, scored by the atom that sits farthest from its partner instead.

    Over the same relabellings, each superposed by the proper rotation that
    minimizes its RMSD; the smallest largest-single-atom distance wins.
    """
    b = coords_b * np.array([1.0, 1.0, -1.0]) if mirror else coords_b
    maps = _candidate_mappings(coords_a, b, symbols, tol)
    if not maps:
        return np.inf
    xa = _center(coords_a)
    xb = _center(b)
    stack = np.stack([xb[m] for m in maps])
    rot = _rotations(np.repeat(xa[None], len(maps), axis=0), stack)
    moved = np.einsum("knj,kij->kni", stack, rot)
    return float(np.linalg.norm(moved - xa[None], axis=-1).max(axis=-1).min())


def symmetry_operations(
    coords: NDArray[np.float64],
    symbols: list[str] | tuple[str, ...],
    rthr: float = DEFAULT_RTHR,
    tol: float = DIST_TOL,
) -> list[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Every operation but the identity that maps the conformer onto itself.

    Each is ``(perm, R)`` with ``R @ x[perm[i]] ~ x[i]`` for every atom ``i`` of
    the centred coordinates, within ``rthr`` RMSD; ``R`` is a rotation, or a
    rotation times a reflection for an improper operation.  These are the
    relabellings `symmetry_number` counts, plus those onto the mirror image,
    kept rather than counted.
    """
    x = _center(np.asarray(coords, dtype=np.float64))
    n = len(symbols)
    ops: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []
    for reflect in (np.eye(3), np.diag([1.0, 1.0, -1.0])):
        y = x @ reflect
        maps = _candidate_mappings(x, y, list(symbols), tol)
        if not maps:
            continue
        stack = np.stack([y[m] for m in maps])
        rot = _rotations(np.repeat(x[None], len(maps), axis=0), stack)
        resid = x[None] - np.einsum("knj,kij->kni", stack, rot)
        rmsd = np.sqrt((resid**2).sum(-1).mean(-1))
        for m, r, dev in zip(maps, rot, rmsd, strict=True):
            full = r @ reflect
            if dev > rthr:
                continue
            if np.array_equal(m, np.arange(n)) and np.allclose(full, np.eye(3), atol=1e-3):
                continue
            ops.append((np.asarray(m), full))
    return ops


# ---------------------------------------------------------------------------
# Methyl-type tops
# ---------------------------------------------------------------------------

# Covalent radii (Angstrom, Cordero et al. 2008) for finding a top's bonds.  A
# pair is bonded within BOND_SCALE times the sum, as in RDKit's and Open
# Babel's distance perception.  The bonds that matter are a hydrogen's and the
# centre's, whose count must come to four; an atom two bonds from the centre
# must stay out of reach, which fluorine at OTHER_RADIUS would not (CH3-CF3's
# C...F is 2.35 Angstrom).  Elements missing here take OTHER_RADIUS, about the
# heavier main-group elements' own.
_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}
OTHER_RADIUS = 1.2
BOND_SCALE = 1.25

# A rotation of the collapsed structure that leaves a top in place turns it
# about its own axis; for the real structure it must turn it by a multiple of
# 120 degrees, within this many, so that hydrogens land on hydrogens.
TOP_TURN_TOL = 15.0


def h_tops(
    coords: NDArray[np.float64], symbols: list[str] | tuple[str, ...]
) -> list[tuple[int, tuple[int, int, int]]]:
    """Each methyl-type top of a conformer: ``(centre, its three hydrogens)``.

    A top is an atom with four neighbours, three of them hydrogens bonded to
    nothing else and the fourth not a hydrogen: CH3, NH3+, SiH3.  Bonds are
    short distances (see `_RADII`); this is the one place in the module that
    perceives any, and all it needs is which hydrogens sit on which atom.
    """
    x = np.asarray(coords, dtype=np.float64)
    radius = np.array([_RADII.get(s, OTHER_RADIUS) for s in symbols])
    bonded = _distance_matrix(x) < BOND_SCALE * (radius[:, None] + radius[None, :])
    np.fill_diagonal(bonded, False)
    is_h = np.array([s == "H" for s in symbols])
    out = []
    for c in np.flatnonzero(~is_h & (bonded.sum(axis=1) == 4)):
        nbrs = np.flatnonzero(bonded[c])
        hs = nbrs[is_h[nbrs]]
        if len(hs) == 3 and all(bonded[h].sum() == 1 for h in hs):
            out.append((int(c), (int(hs[0]), int(hs[1]), int(hs[2]))))
    return out


def _collapse(
    coords: NDArray[np.float64],
    symbols: list[str] | tuple[str, ...],
    tops: list[tuple[int, tuple[int, int, int]]],
) -> tuple[NDArray[np.float64], list[str], list[tuple[int, int]]]:
    """The structure with each top's hydrogens replaced by one pseudo-atom at their centroid.

    The centroid lies on the top's axis, so the collapsed structure no longer
    says how the top is turned.  Pseudo-atoms go after the other atoms, in the
    order of their centres, labelled by the centre's element so that a CH3
    never stands in for an NH3+.  Returns the coordinates, the symbols and, for
    each top, ``(centre, pseudo-atom)`` as indices into the collapsed structure.
    """
    x = np.asarray(coords, dtype=np.float64)
    gone = {h for _, hs in tops for h in hs}
    rest = [i for i in range(len(symbols)) if i not in gone]
    where = {a: k for k, a in enumerate(rest)}
    ordered = sorted(tops)
    xc = np.vstack([x[rest]] + [x[list(hs)].mean(axis=0)[None] for _, hs in ordered])
    sc = [symbols[i] for i in rest] + [f"{symbols[c]}H3*" for c, _ in ordered]
    fixed = [(where[c], len(rest) + k) for k, (c, _) in enumerate(ordered)]
    return xc, sc, fixed


def _top_sigma(
    xc: NDArray[np.float64],
    sc: list[str],
    fixed: list[tuple[int, int]],
    rthr: float,
    tol: float,
) -> int:
    """The symmetry number a collapsed structure stands for, whatever angles its tops sit at.

    A rotation of the collapsed structure is a symmetry of the real one only
    if every top it leaves in place it also turns by a multiple of 120 degrees,
    so that hydrogens land on hydrogens; a top it moves onto another can be
    turned to match.  The rotations that pass need not form a group (toluene's
    C2 turns the methyl by 180 degrees and fails), so this counts the largest
    subgroup made only of passing rotations.  Every finite rotation group is
    generated by two of its elements, so trying every pair finds it.  When the
    collapsed structure is a line (CH3-C#N, ethane) the search sees no turn
    about it at all; the threefold turn of the tops on it is then a symmetry
    of the real structure, so the count is three times the one found.
    """
    x = _center(xc)
    maps = _candidate_mappings(xc, xc, sc, tol)  # never empty: the identity maps
    stack = np.stack([x[m] for m in maps])
    rot = _rotations(np.repeat(x[None], len(maps), axis=0), stack)
    rmsd = np.sqrt(((x[None] - np.einsum("knj,kij->kni", stack, rot)) ** 2).sum(-1).mean(-1))
    identity = tuple(range(len(sc)))
    passing = {identity}
    for m, r, dev in zip(maps, rot, rmsd, strict=True):
        if dev > rthr:
            continue
        turn = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
        off = min(turn % 120.0, 120.0 - turn % 120.0)
        if off > TOP_TURN_TOL and any(m[c] == c and m[p] == p for c, p in fixed):
            continue
        passing.add(tuple(int(i) for i in m))

    def closure(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        """The order of the group ``a`` and ``b`` generate, or 0 if it leaves ``passing``."""
        group = {identity, a, b}
        while True:
            new = {tuple(g[i] for i in h) for g in group for h in group} - group
            if not new:
                return len(group)
            if not new <= passing:
                return 0
            group |= new

    elements = sorted(passing)
    best = max(closure(a, b) for i, a in enumerate(elements) for b in elements[i:])
    sv = np.linalg.svd(x, compute_uv=False)
    return 3 * best if sv[1] < 0.05 * sv[0] else best


def with_tops_collapsed(
    coords: NDArray[np.float64], symbols: list[str] | tuple[str, ...]
) -> tuple[tuple[tuple[int, tuple[int, int, int]], ...], NDArray[np.float64], list[str]]:
    """``(tops, coordinates, symbols)`` of a conformer with its own tops collapsed.

    The conformer as given, with ``tops`` empty, when it has none.  Two
    conformers compare this way only when their ``tops`` agree, since the
    collapsed structures otherwise hold different atoms; `deduplicate` never
    merges conformers whose tops differ.
    """
    tops = tuple(h_tops(coords, symbols))
    if not tops:
        return tops, np.asarray(coords, dtype=np.float64), list(symbols)
    xc, sc, _ = _collapse(coords, symbols, list(tops))
    return tops, xc, sc


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def _screen_pairs(
    coords: NDArray[np.float64], symbols: list[str], rthr: float
) -> NDArray[np.bool_]:
    """Which pairs are worth an exact search, as a mask over triu indices.

    Two screens, cheapest first.  The singular-value bound is rigorous: by
    Horn's inequality the sorted singular values of the centred coordinates
    bound the RMSD from below for *every* relabelling, so a pair it rejects
    cannot match under any of them.  It is also invariant to reflection, so the
    same mask serves the mirror comparison.  The element-typed sorted-distance
    descriptor that follows is a heuristic, used only to cut work further.
    """
    n = len(coords)
    iu, ju = np.triu_indices(n, k=1)

    sv = np.linalg.svd(_center(coords), compute_uv=False)
    diff = sv[:, None, :] - sv[None, :, :]
    bound = np.sqrt((diff * diff).sum(axis=-1) / coords.shape[1])
    alive: NDArray[np.bool_] = bound[iu, ju] <= rthr
    if not alive.any():
        return alive

    ia, ja = np.triu_indices(coords.shape[1], k=1)
    dist = np.linalg.norm(coords[:, ia] - coords[:, ja], axis=-1)
    pair_type = np.array(
        ["|".join(sorted((symbols[i], symbols[j]))) for i, j in zip(ia, ja, strict=True)]
    )
    blocks = [np.sort(dist[:, pair_type == t], axis=1) for t in sorted(set(pair_type.tolist()))]
    desc = np.concatenate(blocks, axis=1)
    delta = np.abs(desc[:, None, :] - desc[None, :, :]).max(axis=-1)
    survivors: NDArray[np.bool_] = alive & (delta[iu, ju] <= DESC_TOL)
    return survivors


def _connected_groups(adjacency: NDArray[np.bool_]) -> list[list[int]]:
    """Union-find over a boolean 'same state' matrix.

    Equivalence at a finite threshold is not perfectly transitive, so groups
    are connected components -- the same choice CREGEN makes.
    """
    n = adjacency.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.triu_indices(n, k=1), strict=True):
        if adjacency[int(i), int(j)]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return [sorted(v) for v in groups.values()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deduplicate(
    coords: NDArray[np.float64],
    symbols: list[str],
    rthr: float = DEFAULT_RTHR,
    energies: NDArray[np.float64] | None = None,
    ethr: float | None = None,
    merge_mirrors: bool = True,
    tol: float = DIST_TOL,
    collapse_tops: bool = False,
) -> list[list[int]]:
    """Group an ensemble into distinct physical states.

    Returns lists of indices; the caller keeps one member of each, normally the
    lowest in energy.

    ``merge_mirrors`` also folds together conformers that are reflections of
    one another.  Those are isoenergetic by symmetry, so refining and scoring
    both wastes the work twice over and lets independent Hessians disagree.
    The pair is instead represented once and given a multiplicity of two by
    :func:`conformer_multiplicity`, which assumes this merge has happened.

    ``energies`` and ``ethr`` (kcal/mol) add the energy criterion CREGEN pairs
    with its RMSD threshold.  Worth using on unrelaxed geometries, where two
    structures can sit within ``rthr`` and still differ substantially in
    energy; unnecessary once geometries are optimised, where true duplicates
    agree to well under 0.01 kcal/mol.

    ``collapse_tops`` compares the conformers with their methyl-type tops
    collapsed (`h_tops`), so two that differ only in how a methyl is turned are
    one state: its staggered minimum and eclipsed hilltop, or two stopping
    points on a nearly free turn.  Conformers whose tops differ are never
    merged.  Use it with `conformer_multiplicity`'s, so that the count
    describes the same states.
    """
    if not collapse_tops:
        return _deduplicate(coords, symbols, rthr, energies, ethr, merge_mirrors, tol)
    views = [with_tops_collapsed(x, symbols) for x in coords]
    keys = [tops for tops, _, _ in views]
    groups: list[list[int]] = []
    for key in dict.fromkeys(keys):
        members = [i for i, k in enumerate(keys) if k == key]
        xs, ss = np.stack([views[i][1] for i in members]), views[members[0]][2]
        e = None if energies is None else np.asarray(energies)[members]
        for g in _deduplicate(xs, ss, rthr, e, ethr, merge_mirrors, tol):
            groups.append([members[k] for k in g])
    return sorted(groups)


def _deduplicate(
    coords: NDArray[np.float64],
    symbols: list[str],
    rthr: float,
    energies: NDArray[np.float64] | None,
    ethr: float | None,
    merge_mirrors: bool,
    tol: float,
) -> list[list[int]]:
    """`deduplicate` on the structures as given."""
    n = len(coords)
    if n < 2:
        return [[0]] if n else []

    iu, ju = np.triu_indices(n, k=1)
    alive = _screen_pairs(coords, symbols, rthr)
    if energies is not None and ethr is not None:
        e = np.asarray(energies)
        alive &= np.abs(e[iu] - e[ju]) <= ethr

    same = np.zeros((n, n), dtype=bool)
    for i, j in zip(iu[alive], ju[alive], strict=True):
        if best_rmsd(coords[i], coords[j], symbols, tol) <= rthr or (
            merge_mirrors and best_rmsd(coords[i], coords[j], symbols, tol, True) <= rthr
        ):
            same[i, j] = same[j, i] = True
    return _connected_groups(same)


def symmetry_number(
    coords: NDArray[np.float64],
    symbols: list[str],
    rthr: float = DEFAULT_RTHR,
    tol: float = DIST_TOL,
    collapse_tops: bool = False,
) -> tuple[int, bool]:
    """``(sigma, achiral)`` for one conformer, from its coordinates alone.

    ``sigma`` counts the relabellings that map the structure back onto itself
    under a proper rotation -- the rotational symmetry number of this
    conformer, not of the molecule.  A molecule can be full of symmetric groups
    and still have sigma = 1 in every conformer it adopts, because a methyl's
    threefold turn is only a symmetry of the whole structure if everything
    attached to it lies on that axis.

    ``achiral`` is True when some relabelling maps the structure onto its own
    reflection, meaning it stands for one physical state rather than an
    enantiomeric pair.

    ``collapse_tops`` answers both for the structure with its methyl-type tops
    (`h_tops`) turned freely: achiral if some turn of them makes it so, and
    sigma by `_top_sigma`.  A methyl's turn is nearly free, so where an
    optimizer leaves it says nothing about the state; see docs/symmetry.md.
    """
    if collapse_tops and (tops := h_tops(coords, symbols)):
        xc, sc, fixed = _collapse(coords, symbols, tops)
        return (
            _top_sigma(xc, sc, fixed, rthr, tol),
            best_rmsd(xc, xc, sc, tol, mirror=True) <= rthr,
        )
    x = _center(coords)
    maps = _candidate_mappings(coords, coords, symbols, tol)
    sigma = 1
    if maps:
        stack = np.stack([x[m] for m in maps])
        hits = _kabsch_rmsd(np.repeat(x[None], len(maps), axis=0), stack) <= rthr
        sigma = max(1, int(hits.sum()))
    return sigma, best_rmsd(coords, coords, symbols, tol, mirror=True) <= rthr


def conformer_multiplicity(
    coords: NDArray[np.float64],
    symbols: list[str],
    includes_enantiomer: bool = False,
    rthr: float = DEFAULT_RTHR,
    tol: float = DIST_TOL,
    collapse_tops: bool = False,
) -> NDArray[np.float64]:
    """Weight of each conformer in ``Z = sum_i m_i exp(-G_i / RT)``.

    ``m_i = n_states_i / sigma_i``.

    ``sigma_i`` divides because symmetry removes distinguishable arrangements:
    a conformer whose relabellings map it onto itself has fewer distinct ways
    of existing, hence less entropy.

    ``n_states_i`` is 2 for a conformer that is chiral by conformation alone,
    because :func:`deduplicate` has folded its mirror image away and the
    surviving entry stands for both.  It is 1 when the conformer is achiral, or
    when ``includes_enantiomer`` is set -- there the mirror inverts a
    stereocentre, ``stereo.deduplicate_enantiomers`` has already collapsed that
    pair at the SMILES level, and the microstate-level factor owns it.  The two
    factors are disjoint by construction and applying both would double-count.

    Run this on a deduplicated ensemble.  Applied to a list that still holds
    duplicates it counts a state twice and then doubles one of the copies.
    ``collapse_tops`` as for `symmetry_number`; deduplicate with the same.
    """
    n = len(coords)
    mult = np.ones(n, dtype=float)
    for i in range(n):
        sigma, achiral = symmetry_number(coords[i], symbols, rthr, tol, collapse_tops)
        n_states = 1.0 if (achiral or includes_enantiomer) else 2.0
        mult[i] = n_states / sigma
    return mult


def effective_energy_offset(multiplicity: float, temperature: float = 298.15) -> float:
    """``-RT ln(m)`` in Hartree: the shift that turns G into an energy the
    Boltzmann factor can be read off directly.

    A conformer contributes ``m exp(-G/RT) = exp(-(G - RT ln m)/RT)``, so this
    is the quantity an energy window should be measured against.
    """
    return -KB_HARTREE * temperature * float(np.log(multiplicity))
