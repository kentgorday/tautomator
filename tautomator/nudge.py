"""Nudge symmetric conformers off their symmetry before the refinement optimization.

An optimizer cannot leave a symmetric geometry along a direction that breaks
its symmetry: the gradient there is totally symmetric, so it has no component
along any mode that is not.  CREST often hands refinement symmetric conformers
-- an NH2 lying flat, two rings coplanar, a ring held flat, a CCl3 or methyl
with one atom in a mirror plane, a threefold propeller -- and when that shape is
a saddle on the refinement surface rather than a minimum, the refinement
optimization stops on it.  The ensemble then keeps the hilltop and never sees
the minimum beside it.

So before each refinement optimization, a conformer that has a symmetry
operation is moved AMPLITUDE (its largest single-atom move) along its softest
GFN2 normal mode that is less than BREAKS totally symmetric, if that mode is
below SOFT_CUTOFF.  A symmetric shape that is a minimum at the refinement level
pulls the optimizer back; a saddle lets it roll into the minimum beside it.
One direction is enough when an operation reverses the mode, as a mirror or a
twofold axis does every mode it breaks: the operation carries the + start onto
the - start, so the two reach the same conformer or its mirror image, which
deduplication and the mirror-image bookkeeping already count as one
(docs/symmetry.md).  A threefold axis turns a degenerate mode by 120 degrees
rather than reversing it; there + and - differ only by how the valley around
the axis is warped, and the minima the warping makes are images of one another.

That push turns one soft group.  Once it has broken the symmetry, a second soft
group on a hilltop of its own feels only the coupling to the first, and the
optimizer can converge with it still there: N-methylacetamide's other methyl,
one of 2,6-lutidine's two.  So the same start also moves AMPLITUDE along each
later breaking mode below SECOND_CUTOFF.  Those modes are first rotated among
themselves (`localize`) so that each moves one group: two equivalent groups
share their modes as an in-phase and an out-of-phase pair, and the sum of the
pair would move one group twice and the other not at all.  The softest mode's
own push is left as it is; folding it into the rotation shrank its share and
sent some conformers back onto the saddle it was there to leave.
A mirror or a twofold axis reverses the whole move, later modes included, so
the argument for one direction holds for it.

A nudge can also do harm.  On a nearly flat valley -- a CBr3 or CF3 that turns
almost freely, an NH2 that barely prefers to be flat -- the optimizer stops short
of the symmetric shape it would have kept, a few hundredths of a kcal/mol below
it, and the conformer is then called chiral and counted twice: 0.41 kcal/mol for
nothing.  So `settle` follows the optimization.  When a nudged optimization
converges with less symmetry than the conformer started with, its result is
averaged over its images under the operations it was sampled with, and that
symmetric shape is optimized; started exactly symmetric, the optimizer keeps it
so, which is the trap put to use.  If the nudged result is less than SETTLE
below it, the symmetric shape replaces it.  SETTLE is where the quartic
double-well count n*(dE/kT) (docs/nudge.md, SETTLE) is sqrt(2): below it a
single symmetric well is the nearer of the two whole counts, above it the mirror
pair is, and either is off by at most RT ln sqrt(2) = 0.21 kcal/mol.  Two starts
can reach the same shallow well and settle it differently, each averaging toward
the symmetric shape its own pairing points at; `fold` applies the same rule
across the microstate before deduplication, with the symmetric conformers
already there as the shapes, which also catches such pairs that sampling handed
over unnudged.  Before, because a well this shallow can deduplicate with its
symmetric shape, and the survivor of that is picked by an energy that still
holds sampling's RRHO.

The GFN2 Hessian is the plain one sampling takes (the conformer is a GFN2
minimum from `sampling._reminimize`), in sampling's solvent, and it is taken
only for conformers with a symmetry operation.  Symmetry is judged with the
tolerance the conformer symmetry number uses, so a conformer is nudged exactly
when the counting treats it as symmetric.  Conformers with no symmetry are left
alone: no symmetry holds their optimizer on a saddle, and a push only picks a
basin by GFN2's lights.  A second soft group that GFN2 puts above SECOND_CUTOFF
is not pushed.  How the constants were chosen, what the nudge costs and where
it stops is in docs/nudge.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tautomator import xtb_runner
from tautomator.conformer_symmetry import (
    DEFAULT_RTHR,
    best_max_distance,
    superpose,
    symmetry_number,
    symmetry_operations,
)
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.types import Geometry

log = logging.getLogger(__name__)

AMPLITUDE = 0.3  # Angstrom: the largest single-atom move (docs/nudge.md)
SOFT_CUTOFF = 300.0  # cm**-1: only modes softer than this are used
SECOND_CUTOFF = 150.0  # cm**-1: later breaking modes softer than this are pushed along too
BREAKS = 0.5  # a mode breaks the symmetry when less than this fraction of it is totally symmetric
SETTLE = 0.075  # kcal/mol: a symmetric shape this close above the nudged result replaces it
SETTLE_REACH = 1.25  # Angstrom: no settling once an atom sits this far from an image

# One symmetry operation: the relabelling, and whether it reflects.
Operation = tuple[tuple[int, ...], bool]


@dataclass(frozen=True)
class Nudge:
    """What was done to one conformer."""

    frequency: float  # cm**-1: the GFN2 mode moved along
    symmetric_part: float  # the fraction of that mode the symmetry keeps; 0 is exact
    operations: tuple[Operation, ...]  # the conformer's symmetry operations, identity excluded
    symmetry: tuple[int, bool]  # (sigma, achiral) of the conformer as sampled
    later: tuple[float, ...] = ()  # cm**-1: the later breaking modes pushed along with it


def symmetric_part(
    mode: NDArray[np.float64], ops: list[tuple[NDArray[np.int64], NDArray[np.float64]]]
) -> float:
    """``|Pv|^2 / |v|^2``, with ``P`` the average over the identity and ``ops``.

    For a symmetry group, ``P`` projects onto the totally symmetric displacements,
    so this is 1 for a mode that keeps every symmetry and 0 for one that keeps
    none of it: reversed by a mirror, or turned within a degenerate pair by a
    threefold axis.  ``ops`` as `conformer_symmetry.symmetry_operations` gives them.
    """
    images = mode + sum(mode[perm] @ rot.T for perm, rot in ops)
    kept = images / (len(ops) + 1)
    return float((kept * kept).sum() / (mode * mode).sum())


def _collinear(coords: NDArray[np.float64]) -> bool:
    """Whether every turn about the line through the atoms counts as a symmetry.

    A turn by any angle moves the atoms at most twice their RMS distance from
    the line, so within the symmetry tolerance every angle is an operation, and
    the few the relabelling search reports are arbitrary.
    """
    x = np.asarray(coords, dtype=np.float64)
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    return bool(2.0 * np.sqrt((s[1] ** 2 + s[2] ** 2) / len(x)) <= DEFAULT_RTHR)


def _unit(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """``vec`` scaled so its largest single-atom move is 1."""
    return np.asarray(vec / np.linalg.norm(vec, axis=1).max())


def _oriented(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """``vec`` or ``-vec``, whichever leans along one fixed direction.

    An eigensolver gives each mode an arbitrary sign, and two BLAS builds can
    give it different ones.  For the softest mode alone that does not matter, as
    a mirror carries one direction onto the other, but the sign of each later
    push relative to it does: a mirror maps (+, +) onto (-, -), not onto (+, -).
    Fixing every sign against one direction makes the start the same for a given
    input geometry wherever it is computed.  The direction (sines of multiples of
    the golden angle, coordinate by coordinate) pairs no atom with another, so a
    mode that lies across it would be a coincidence.
    """
    x = np.asarray(vec, dtype=np.float64)
    reference = np.sin(2.399963229728653 * np.arange(1, x.size + 1)).reshape(x.shape)
    return x if float((x * reference).sum()) >= 0.0 else -x


def localize(
    modes: Sequence[NDArray[np.float64]], symbols: Sequence[str], *, sweeps: int = 200
) -> list[NDArray[np.float64]]:
    """The same displacements, rotated among themselves so that each moves as few atoms as it can.

    Two equivalent soft groups share their modes as an in-phase and an
    out-of-phase combination, in a proportion the rounding of a near-degenerate
    pair decides.  Rotating the set (Pipek-Mezey, an atom's share of a vector
    being its part of the mass-weighted norm, squared and summed over the set)
    gives one vector per group.  The vectors are taken mass-weighted
    orthonormal first, as normal modes are up to rounding; the result spans the
    same displacements, each of unit norm, in Cartesian coordinates.
    """
    from rdkit.Chem import GetPeriodicTable

    vecs = np.array([np.asarray(m, dtype=np.float64) for m in modes])
    if len(vecs) < 2:
        return [v / np.linalg.norm(v) for v in vecs]
    table = GetPeriodicTable()
    root = np.sqrt([table.GetAtomicWeight(s) for s in symbols])[None, :, None]
    flat = (vecs * root).reshape(len(vecs), -1)
    q, _ = np.linalg.qr(flat.T)
    w = (q.T * np.sign((q.T * flat).sum(axis=1))[:, None]).reshape(vecs.shape)
    for _ in range(sweeps):
        largest = 0.0
        for s in range(len(w)):
            for t in range(s + 1, len(w)):
                cross = (w[s] * w[t]).sum(axis=1)
                diff = (w[s] * w[s]).sum(axis=1) - (w[t] * w[t]).sum(axis=1)
                a = float((cross * cross - 0.25 * diff * diff).sum())
                b = float((cross * diff).sum())
                if abs(a) + abs(b) < 1e-14:
                    continue
                gamma = 0.25 * np.arctan2(b, -a)
                largest = max(largest, abs(gamma))
                ws, wt = w[s].copy(), w[t].copy()
                w[s] = np.cos(gamma) * ws + np.sin(gamma) * wt
                w[t] = -np.sin(gamma) * ws + np.cos(gamma) * wt
        if largest < 1e-9:
            break
    out = w / root
    return [v / np.linalg.norm(v) for v in out]


def nudge(
    geom: Geometry,
    charge: int,
    *,
    solvent: str | None,
    threads: int | None = None,
    amplitude: float = AMPLITUDE,
    cutoff: float = SOFT_CUTOFF,
    second_cutoff: float = SECOND_CUTOFF,
    sign: float = 1.0,
) -> tuple[Geometry, Nudge | None]:
    """The geometry to start the refinement optimization from, and what was done.

    The start is ``amplitude`` along the softest symmetry-breaking mode below
    ``cutoff``, plus ``amplitude`` along each later breaking mode below
    ``second_cutoff``, those later ones localized among themselves, each mode's
    sign fixed by `_oriented`.  Returns
    ``geom`` itself and None when the conformer is linear, has no symmetry
    operation or no symmetry-breaking mode below ``cutoff``, or its GFN2
    Hessian fails.  ``sign`` picks the direction of the whole move.
    """
    if geom.n_atoms < 3 or _collinear(geom.coords):
        return geom, None
    ops = symmetry_operations(geom.coords, geom.symbols)
    if not ops:
        return geom, None
    try:
        freqs, modes = xtb_runner.normal_modes(geom, charge, solvent=solvent, threads=threads)
    except Exception as e:
        log.warning(f"    GFN2 Hessian for the nudge failed ({e}); optimizing as sampled")
        return geom, None
    breaking: list[tuple[float, float, NDArray[np.float64]]] = []  # softest first
    for freq, mode in zip(freqs, modes, strict=True):
        if freq >= cutoff or (breaking and freq >= second_cutoff):
            break
        kept = symmetric_part(mode, ops)
        if kept < BREAKS:
            breaking.append((float(freq), kept, np.asarray(mode)))
    if not breaking:
        return geom, None
    (freq, kept, first), later = breaking[0], breaking[1:]
    step = _unit(_oriented(first))
    if later:
        localized = localize([m for _, _, m in later], geom.symbols)
        step = step + sum(_unit(_oriented(v)) for v in localized)
    coords = np.asarray(geom.coords) + sign * amplitude * step
    moved = Geometry(symbols=geom.symbols, coords=coords)
    return moved, Nudge(
        frequency=freq,
        symmetric_part=kept,
        operations=tuple(
            (tuple(int(i) for i in perm), bool(np.linalg.det(rot) < 0)) for perm, rot in ops
        ),
        symmetry=symmetry_number(geom.coords, list(geom.symbols)),
        later=tuple(f for f, _, _ in later),
    )


def _image(x: NDArray[np.float64], op: Operation) -> NDArray[np.float64]:
    """Centred ``x`` relabelled by ``op`` (reflected first if improper), laid back onto ``x``."""
    perm, improper = op
    z = x[list(perm)]
    if improper:
        z = z * np.array([1.0, 1.0, -1.0])
    return superpose(z, x)


def symmetrize(geom: Geometry, operations: tuple[Operation, ...]) -> tuple[Geometry, float]:
    """``geom`` made symmetric under ``operations``, and how far it was from its images.

    Each step replaces the geometry by the average of itself and its images
    under the operations, every image laid onto it by a proper rotation; that
    removes the asymmetric part to first order, so a few steps leave it
    symmetric to rounding.  One step is exact for a single operation of order
    two.  Relabelling by the operations found on the sampled conformer, rather
    than matching atoms afresh, is what keeps a twisted top's atoms paired with
    their own images.  The distance returned is the largest single-atom gap
    between ``geom`` and any of its images (Angstrom).
    """
    x0 = np.asarray(geom.coords, dtype=np.float64)
    centre = x0.mean(axis=0)
    x = x0 - centre
    gap = max(float(np.linalg.norm(x - _image(x, op), axis=1).max()) for op in operations)
    for _ in range(50):
        average = (x + sum(_image(x, op) for op in operations)) / (len(operations) + 1)
        change = float(np.abs(average - x).max())
        x = average
        if change < 1e-10:
            break
    return Geometry(symbols=geom.symbols, coords=x + centre), gap


@dataclass(frozen=True)
class Settled:
    """The symmetric shape beside a nudged result, optimized at the level the result was."""

    geometry: Geometry
    energy: float  # Hartree
    depth: float  # kcal/mol the nudged result lies below it; negative when it lies above

    @property
    def replaces(self) -> bool:
        """Whether the nudged result's well is too shallow to count as a mirror pair."""
        return self.depth < SETTLE


def _at_least(a: tuple[int, bool], b: tuple[int, bool]) -> bool:
    """Whether symmetry ``a`` keeps all of symmetry ``b``, as far as (sigma, achiral) tell.

    The symmetry number counts the proper rotations, and a subgroup's order
    divides the group's, so ``a``'s must be a multiple of ``b``'s; and ``a`` must
    be achiral if ``b`` is.
    """
    return a[0] % b[0] == 0 and (a[1] or not b[1])


def settle(
    done: Nudge,
    geom: Geometry,
    energy: float,
    optimize: Callable[[Geometry], tuple[Geometry, float, bool]],
) -> Settled | None:
    """The symmetric shape beside a nudged optimization's result, if there is one to compare.

    None when the result kept the symmetry the conformer was sampled with (or
    gained more), sits more than SETTLE_REACH from one of its images, or the
    symmetric shape could not be optimized: it failed, did not converge, or rolled
    off its symmetry.
    ``optimize`` takes a geometry and returns ``(geometry, energy, converged)``
    at the level of ``energy`` (Hartree).  Whether the shape replaces the result
    is `Settled.replaces`.
    """
    if _at_least(symmetry_number(geom.coords, list(geom.symbols)), done.symmetry):
        return None
    shape, gap = symmetrize(geom, done.operations)
    if gap > SETTLE_REACH:
        return None
    try:
        relaxed, e_relaxed, converged = optimize(shape)
    except Exception as e:
        log.warning(f"    Optimizing the symmetric shape failed ({e}); keeping the nudged result")
        return None
    if not converged:
        log.warning("    The symmetric shape did not converge; keeping the nudged result")
        return None
    if not _at_least(symmetry_number(relaxed.coords, list(relaxed.symbols)), done.symmetry):
        log.warning("    The symmetric shape lost its symmetry; keeping the nudged result")
        return None
    return Settled(relaxed, e_relaxed, (e_relaxed - energy) * HARTREE_TO_KCAL)


def fold(
    coords: list[NDArray[np.float64]],
    energies: list[float],
    symmetries: list[tuple[int, bool]],
    symbols: list[str],
) -> list[int]:
    """Indices of a microstate's conformers to keep once shallow wells join their symmetric shape.

    `settle` compares a nudged result with the symmetric shape its own start
    averages toward.  A second start can roll into the same shallow well while
    its pairing points at a different, deeper symmetric shape -- a CBr3 turned
    past the next symmetric position -- and the well then stays a mirror pair
    beside a symmetric conformer that already stands for it.  So each conformer
    is also compared with every more symmetric conformer of its microstate (a
    different (sigma, achiral) that keeps all of this one's, `_at_least`): it is
    dropped when it lies within SETTLE of one that sits within half
    SETTLE_REACH of it, the rule settle applies, with the shape taken from the
    ensemble.  Half, because the symmetric shape lies halfway to the result's
    image.  The distance is the largest single-atom one under the best
    relabelling, the mirror image included.  Within SETTLE either way: a little
    above the symmetric conformer is an optimizer that stopped short of it, but
    one well above it, however close, is a different structure and stays.
    Fewer states alone is not enough: a threefold propeller (sigma 3, chiral)
    stands for fewer states than a conformer with one mirror, but has none of
    its symmetry, so neither stands for the other.  ``energies`` in kcal/mol, at
    the level the geometries were optimized; ``symmetries`` as
    `conformer_symmetry.symmetry_number` gives them.
    """
    reach = SETTLE_REACH / 2

    def near(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
        return min(best_max_distance(a, b, symbols, mirror=m) for m in (False, True)) <= reach

    keep = []
    for i in range(len(coords)):
        folded = any(
            symmetries[j] != symmetries[i]
            and _at_least(symmetries[j], symmetries[i])
            and abs(energies[j] - energies[i]) < SETTLE
            and near(coords[i], coords[j])
            for j in range(len(coords))
            if j != i
        )
        if not folded:
            keep.append(i)
    return keep
