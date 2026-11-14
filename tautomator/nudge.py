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

A nudge can also do harm.  On a nearly flat valley -- a CBr3 or CF3 that turns
almost freely, an NH2 that barely prefers to be flat -- the optimizer stops short
of the symmetric shape it would have kept, a few hundredths of a kcal/mol below
it, and the conformer is then called chiral and counted twice: 0.41 kcal/mol for
nothing.  So `settle` follows the optimization.  When a nudged conformer comes
back with a different symmetry than it started with, it is averaged over its
images under the operations it was sampled with, and that symmetric shape is
optimized; started exactly symmetric, the optimizer keeps it so, which is the
trap put to use.  If the nudged result is less than SETTLE below it, the
symmetric shape replaces it.  SETTLE is where the quartic double-well count
n*(dE/kT) is sqrt(2): below it a single symmetric well is the nearer of the two
whole counts, above it the mirror pair is, and either is off by at most RT ln
sqrt(2) = 0.21 kcal/mol.  Two starts can reach the same shallow well and settle
it differently, each averaging toward the symmetric shape its own pairing points
at; `fold` applies the same rule across the microstate before deduplication,
with the symmetric conformers already there as the shapes, which also catches
such pairs that sampling handed over unnudged.  Before, because a well this
shallow can deduplicate with its symmetric shape, and the survivor of that is
picked by an energy that still holds sampling's RRHO.

The GFN2 Hessian is the plain one sampling takes (the conformer is a GFN2
minimum from `sampling._reminimize`), in sampling's solvent, and it is taken
only for conformers with a symmetry operation.  Symmetry is judged with the
tolerance the conformer symmetry number uses, so a conformer is nudged exactly
when the counting treats it as symmetric.  Conformers with no symmetry are left
alone: no symmetry holds their optimizer on a saddle, and a push only picks a
basin by GFN2's lights.  Only one mode is tried, so a second soft group can stay
on its hilltop.  How the constants were chosen, what the nudge costs and where
it stops is in docs/nudge.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def nudge(
    geom: Geometry,
    charge: int,
    *,
    solvent: str | None,
    threads: int | None = None,
    amplitude: float = AMPLITUDE,
    cutoff: float = SOFT_CUTOFF,
    sign: float = 1.0,
) -> tuple[Geometry, Nudge | None]:
    """The geometry to start the refinement optimization from, and what was done.

    Returns ``geom`` itself and None when the conformer is linear, has no
    symmetry operation or no symmetry-breaking mode below ``cutoff``, or its
    GFN2 Hessian fails.  ``sign`` picks the direction.
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
    for freq, mode in zip(freqs, modes, strict=True):
        if freq >= cutoff:
            break
        kept = symmetric_part(mode, ops)
        if kept < BREAKS:
            step = sign * amplitude * mode / np.linalg.norm(mode, axis=1).max()
            moved = Geometry(symbols=geom.symbols, coords=np.asarray(geom.coords) + step)
            return moved, Nudge(
                frequency=float(freq),
                symmetric_part=kept,
                operations=tuple(
                    (tuple(int(i) for i in perm), bool(np.linalg.det(rot) < 0))
                    for perm, rot in ops
                ),
                symmetry=symmetry_number(geom.coords, list(geom.symbols)),
            )
    return geom, None


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


def settle(
    done: Nudge,
    geom: Geometry,
    energy: float,
    optimize: Callable[[Geometry], tuple[Geometry, float, bool]],
) -> Settled | None:
    """The symmetric shape beside a nudged optimization's result, if there is one to compare.

    None when the result kept the symmetry the conformer was sampled with, sits
    more than SETTLE_REACH from one of its images, or the symmetric shape could
    not be optimized: it failed, did not converge, or rolled off its symmetry.
    ``optimize`` takes a geometry and returns ``(geometry, energy, converged)``
    at the level of ``energy`` (Hartree).  Whether the shape replaces the result
    is `Settled.replaces`.
    """
    if symmetry_number(geom.coords, list(geom.symbols)) == done.symmetry:
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
    if symmetry_number(relaxed.coords, list(relaxed.symbols)) != done.symmetry:
        log.warning("    The symmetric shape lost its symmetry; keeping the nudged result")
        return None
    return Settled(relaxed, e_relaxed, (e_relaxed - energy) * HARTREE_TO_KCAL)


def fold(
    coords: list[NDArray[np.float64]],
    energies: list[float],
    multiplicities: list[float],
    symbols: list[str],
) -> list[int]:
    """Indices of a microstate's conformers to keep once shallow wells join their symmetric shape.

    `settle` compares a nudged result with the symmetric shape its own start
    averages toward.  A second start can roll into the same shallow well while
    its pairing points at a different, deeper symmetric shape -- a CBr3 turned
    past the next symmetric position -- and the well then stays a mirror pair
    beside a symmetric conformer that already stands for it.  So each conformer
    is also compared with every conformer of fewer states (lower multiplicity:
    achiral where it is chiral, or a higher symmetry number) in its microstate:
    it is dropped when it lies within SETTLE of one that sits within half
    SETTLE_REACH of it, the rule settle applies, with the shape taken from the
    ensemble.  Half, because the symmetric shape lies halfway to the result's
    image.  The distance is the largest single-atom one under the best
    relabelling, the mirror image included.  Within SETTLE either way: a little
    above the symmetric conformer is an optimizer that stopped short of it, but
    one well above it, however close, is a different structure and stays.
    ``energies`` in kcal/mol, at the level the geometries were optimized.
    """
    reach = SETTLE_REACH / 2

    def near(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
        return min(best_max_distance(a, b, symbols, mirror=m) for m in (False, True)) <= reach

    keep = []
    for i in range(len(coords)):
        folded = any(
            multiplicities[j] < multiplicities[i]
            and abs(energies[j] - energies[i]) < SETTLE
            and near(coords[i], coords[j])
            for j in range(len(coords))
            if j != i
        )
        if not folded:
            keep.append(i)
    return keep
