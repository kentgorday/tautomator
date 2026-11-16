# The nudge off symmetric saddles

Refinement optimizes each sampled conformer at the refinement level (DFT or
OrbMol). An optimizer started on a symmetric geometry cannot leave it along a
direction that breaks the symmetry: the gradient there is totally symmetric. So
when CREST hands over a symmetric shape — a flat NH2, coplanar rings, a CCl3 or
methyl with an atom in a mirror plane, a threefold propeller — that is a saddle
at the refinement level, the optimization stops on the saddle and the ensemble
keeps a hilltop in place of the minimum beside it. `tautomator.nudge` moves
such conformers off their symmetry first, and then undoes the move where it
only manufactured a shallow mirror pair. The mechanism is described in the
module docstring; this page records how its constants were chosen, what it
costs, and where it stops.

It is on by default (`[refinement] nudge = true`).

## What runs

1. **Nudge** (before each refinement optimization). A conformer with a symmetry
   operation (found with the tolerance the symmetry number uses) gets a GFN2
   Hessian in sampling's solvent. It is moved `AMPLITUDE` (the largest
   single-atom move) along its softest normal mode that is less than `BREAKS`
   totally symmetric, if that mode is softer than `SOFT_CUTOFF`. The same start
   also moves `AMPLITUDE` along each later breaking mode softer than
   `SECOND_CUTOFF`, those modes first rotated among themselves so that each
   moves one group (`localize`). Linear molecules are skipped.
2. **Settle** (after it). If the nudged result has a different symmetry than
   the conformer was sampled with, and no atom sits more than `SETTLE_REACH`
   from its image, the result is averaged over its images under the sampled
   operations, and that symmetric shape is optimized at the refinement level.
   If the result lies less than `SETTLE` below it, the symmetric shape replaces
   it.
3. **Fold** (before deduplication). In each microstate, a conformer within
   `SETTLE` of one with a lower multiplicity, and within `SETTLE_REACH / 2` of
   it, is dropped. This catches a shallow well that two starts reached and
   settled differently, and such pairs that sampling handed over unnudged. It
   runs before deduplication because a well this shallow can lie near enough to
   its symmetric shape to deduplicate with it and still count as chiral, and
   deduplication keeps whichever of the two has the lower free energy, which at
   that point still holds sampling's RRHO. p-Toluate in water did this: one
   start settled onto the symmetric shape, a second (its carboxylate turned by
   90°, too far to settle) stopped 0.009 kcal/mol above it with the methyl
   twisted, and deduplication kept the second at twice the count.

If the nudged optimization raises, the conformer is optimized again from its
sampled geometry; a failed GFN2 Hessian skips the nudge.

## Constants

| Constant | Value | Why |
| --- | --- | --- |
| `AMPLITUDE` | 0.3 Å | as cheap as 0.2 Å, never worse, and one deeper well found |
| `SOFT_CUTOFF` | 300 cm⁻¹ | catches H2O2's trans saddle (251 cm⁻¹) |
| `SECOND_CUTOFF` | 150 cm⁻¹ | catches N-methylacetamide's second methyl (128–146 cm⁻¹) |
| `BREAKS` | 0.5 | an exact symmetry-breaking mode has a symmetric part of 0, a symmetric one 1 |
| `SETTLE` | 0.075 kcal/mol | where the quartic double-well count n*(ΔE/kT) is √2 |
| `SETTLE_REACH` | 1.25 Å | no well under `SETTLE` sat farther than 1.09 Å from its image |

**AMPLITUDE.** Too small a push stalls: a few hundredths of an ångström from
the saddle the slope is too gentle to fail the optimizer's convergence test, and
0.1 Å (old vectors) left both tert-butyl amide saddles in place. Too large a
push starts the optimizer up a wall and costs steps. The old vectors chose
0.2 Å, but the corrected 0.2 Å step goes only 0.73× as far along the soft mode,
so 0.2 and 0.3 Å were both rerun (below). They cost the same, 0.3 Å is never
worse by more than 0.05 kcal/mol, and it carries the tert-butyl amide into a well
0.23 kcal/mol deeper than 0.2 Å reaches. Along the soft mode, the corrected
0.3 Å step goes about as far as the old 0.2 Å one (1.1× at the median).

**BREAKS.** The symmetric part of a mode is |Pv|²/|v|², where P averages the
mode over the conformer's operations and the identity. For a symmetry group, P
projects onto the totally symmetric displacements, so the symmetric part is 1
for a mode that keeps every symmetry and 0 for one that keeps none of it. The
first version tested instead whether some operation reversed the mode,
⟨v|gv⟩/⟨v|v⟩ < −0.5. That misses a degenerate mode of a threefold axis, which
the turn rotates by 120° (⟨v|gv⟩/⟨v|v⟩ = cos 120° = −0.5 exactly), so a
threefold propeller such as triethylamine was never nudged.

**SECOND_CUTOFF.** A push along the softest breaking mode turns one soft
group. Once the symmetry is broken, a second group on a hilltop of its own feels
only its coupling to the first, and the optimizer can converge with it still
there. N-methylacetamide's N-methyl, a wB97X-3c saddle, sits at 128–146 cm⁻¹ in
GFN2 in its four sampled conformers, so 150 cm⁻¹ takes it. Of the 278 nudged
MNSol conformers, 77 have a later breaking mode under 150 cm⁻¹ (44 under 100,
215 under 300). The later modes are rotated (Pipek–Mezey, each atom's share of
a vector being its part of the mass-weighted norm) because two equivalent groups
share their modes as an in-phase and an out-of-phase pair: the sum of the pair
moves one group twice and the other not at all. The softest mode stays out of
the rotation, so its push is the one it had before (Measurements, second push).

**SETTLE.** On a double well whose two minima lie ΔE below the barrier between
them, the classical count of states runs from 1 (ΔE = 0, a single flat-bottomed
well) to 2 (a deep pair). The quartic count n*(ΔE/kT) crosses √2 at
ΔE = 0.075 kcal/mol. Below that, one symmetric well is the nearer of the two
whole counts, and above it the mirror pair is. Either way the count is off by at
most RT ln √2 = 0.21 kcal/mol.

## Measurements

The test sets are refinement runs (wB97X-3c/vDZP, gas and implicit solvent)
from the MNSol production set and from a set of molecules with bulky rotors
(CX3, tert-butyl, SiMe3 groups) chosen because they trap optimizers. Energies
compare the nudged result with the conformer the production run, which had no
nudge, refined from the same start.

**Mode vectors.** Until 2026-11-11 the nudge read its modes from xtb's
`g98.out`, which holds the mass-weighted eigenvectors rounded to two decimals,
not Cartesian displacements. Results marked *(old vectors)* below were measured
with those. `xtb_runner.normal_modes` now builds Cartesian modes from xtb's full
Hessian. A GFN2-only census of the 436 symmetric conformers in 279 runs compares
the two. Of the 393 conformers the old selection nudged, the corrected one picks
the same mode in 391 and a different one in 2 (one molecule in two solvents). It
also nudges a C2 conformer that the old reversal test missed. The old step put
its 0.2 Å on a heavy atom. The corrected step puts it on a hydrogen in 372 of
394 conformers and moves heavy atoms a median 0.07 Å, so along the soft mode it
goes a median 0.73× as far as the old one (10th percentile 0.52, least 0.34).
No census conformer is a pure threefold propeller or linear, so the other two
selection fixes change nothing on these sets.

**Amplitude.** Each symmetric conformer is optimized nudged at the refinement
level and compared with what production refined from the same start. The
*informative* rows are the 89 conformers where an earlier ±0.1 Å push (old
vectors) changed the structure, the symmetry or the energy by more than 0.002
kcal/mol; the *plain* rows are the other 63. Settling is scored with
`SETTLE_REACH` = 1.25 Å.

| | old vectors, 0.2 Å | 0.2 Å | 0.3 Å |
| --- | --- | --- | --- |
| informative rows more than 0.01 kcal/mol lower | 46 | 46 | 45 |
| informative rows more than 0.01 kcal/mol higher | 2 | 1 | 1 |
| informative rows settled | 24 | 22 | 23 |
| plain rows moved more than 0.01 kcal/mol | 0 | 0 | 0 |
| unconverged optimizations | 0 | 0 | 0 |
| sum of energy changes, kcal/mol | −38.5 | −38.7 | −39.2 |
| steps, informative (unnudged: 934) | 1912 | 1877 | 1815 |
| steps, plain (unnudged: 666) | 938 | 875 | 905 |
| steps, all 152, relative to unnudged | 1.78× | 1.72× | 1.70× |

Steps include settling. The one fewer lower row at 0.3 Å is a hexadecane
conformer that ends 0.016 kcal/mol lower at 0.2 Å and 0.001 lower at 0.3 Å.

On the informative rows, three conformers differ by more than 0.05 kcal/mol
between the three runs:

- The three symmetric starts of a CCl3 amide (CCl3CONHMe) all reach a well 0.22
  kcal/mol below the lowest conformer production had, with the corrected vectors
  at either amplitude. The old vectors reached it from two of them and stopped
  0.23 kcal/mol short from the third, so the microstate found the well either
  way.
- A tert-butyl amide (tBuCONHMe, listed twice under two run names) has a
  rotation saddle 0.77 kcal/mol above its lowest conformer. At 0.2 Å the push
  lands in that conformer; at 0.3 Å it crosses into a well 0.23 kcal/mol below
  it, which production never found. That molecule's other sampled conformers
  have no symmetry and are not nudged, so at 0.2 Å the deeper well stays
  missing.

The one uphill row at either amplitude is a settle doing its job: the
symmetric shape of an ammonium on a crowded naphthalene lies 0.026 kcal/mol
above the nudged result and replaces it. Wells under `SETTLE` sat at most 1.09
Å from their image at both amplitudes, as with the old vectors, so
`SETTLE_REACH` stays 1.25 Å.

**Edge cases (2026-11-13).** 56 runs (a species in one phase on one driver),
each picked to strain one assumption, were refined with the nudge on and off,
and wB97X-3c Hessians were taken of the gas-phase results the questions below
turned on. No optimization failed or stayed unconverged. With the
nudge on, the lowest conformer was never more than 0.02 kcal/mol above the
unnudged one, except where fold replaced a well 0.04 kcal/mol deep by its
symmetric shape. The 37 nudged PySCF runs took 1.28× the unnudged steps.

- *Other drivers.* 0.3 Å carries over. Unnudged, OrbMol-v2 stays on the saddle
  in all five known traps (two tert-butyl amides, 2-chloro-6-fluorotoluene,
  acetic acid, trichloroacetamide) and Psi4 in three; nudged, both reach the
  minimum in all five.
- *Threefold axes.* 2-Butyne, DABCO, tert-butylammonium, quinuclidinium and
  mesylate are nudged and return to the conformer they started as; 2-butyne is
  settled over its 11 operations.
- *A shared proton.* In hydrogen maleate, hydrogen phthalate and the enol of
  acetylacetone, no nudge moved the proton or changed the microstate.
- *Acid and base with a para methyl.* 4-Methylpyridine, p-cresol and their
  ions keep one count in both runs, each settled back from a twisted methyl.
  p-Toluate showed why fold has to run before deduplication (What runs, 3).
- *Shallow wells.* The pyramidal NH2 of 4-aminopyridine (0.42 kcal/mol below
  the planar shape in the gas phase, 0.31 in water) and of p-nitroaniline (0.32
  in the gas phase) is reached with and without the nudge. p-Nitroaniline in
  water stays planar in both.

- *Two soft sites.* 2-Methylpyridine's saddle is freed. Both conformers of
  N-methylacetamide keep a methyl on its hilltop (−26 and −55 cm⁻¹), 0.10 and
  0.01 kcal/mol above their minima, and one conformer of 2,6-lutidine keeps one,
  0.32 kcal/mol above the minimum it belongs to. For the first of these and for
  the lutidine conformer, a push along the softest breaking GFN2 mode at the
  refined geometry comes back, and one along the second-softest reaches the
  minimum. Unnudged, N-methylacetamide keeps four conformers, and the lowest
  2,6-lutidine conformer has both methyls on hilltops.
- *A top beside an asymmetric group.* N-Ethylpivalamide,
  N-ethyltrifluoroacetamide, tert-butyl N-methylcarbamate, tert-butyl acetate
  and the gauche conformers of ethyl tert-butyl ketone and ethyl trichloromethyl
  ketone: every conformer without symmetry is a minimum by its Hessian. Their
  saddles were symmetric conformers, which the nudge frees (N-ethylpivalamide's
  ends 0.79 kcal/mol lower). N-Methylpivalamide is the exception. Its nudged
  start finds a well 0.23 kcal/mol below anything the unnudged run has, but its
  other start, sampled without symmetry, optimizes onto a symmetric saddle that
  far above the well and stays.
- *MNSol recount.* The 14 MNSol gas-phase runs whose production ensembles hold
  DFT saddles were refined again with the nudge on. Production has 20 saddles
  among 61 refined conformers; nudged, there are 6 among 47. Of the six, one is
  a symmetric shape kept because the well beside it is under `SETTLE`, one is
  the 2,6-lutidine conformer above, three have no symmetry (0.98, 1.66 and 3.23
  kcal/mol above their lowest conformer), and one is a twofold-symmetric
  1,4-dioxane conformer 6.97 kcal/mol up that the nudge did not free.

What these runs left unfixed is under Limits.

**Second push (2026-11-15).** The two-site cases above led to four ways of
pushing more than one mode, each run per conformer on the 152-row set and as
full refinements of edge-case species, against the single push:

| Variant | N-methylacetamide | 2,6-lutidine | What went wrong | Steps against one push |
| --- | --- | --- | --- | --- |
| Every later breaking mode added, 0.3 Å each | freed (cutoff 150 or 300) | freed (cutoff 100) | at cutoff 300, a spurious chiral N-methylacetamide conformer: F 0.69 too low | 1.04–1.30 on those two |
| Every picked mode rotated together, cutoff 300 | freed | freed | the softest mode's share of the push shrank: 2 of 79 conformers went back onto the saddle one push leaves, 0.34 and 1.94 kcal/mol up | 1.07 |
| **One push as before, plus later modes under 150 rotated (built)** | **freed** | **freed** | **nothing found** | **1.02** |
| A second optimization from the first result when it comes back to its symmetry | freed | freed | nothing found in 42 retried conformers | 1.81 on the 58% it retries |

The retry was tested with the rule that keeps any result at least `SETTLE`
lower. A first rule that kept only results off the sampled symmetry missed
N-methylacetamide, whose minimum keeps it (its N-methyl turned 60°).

With the variant built, only one of 73 conformers moved by more than 0.01
kcal/mol per conformer: a 2,6-lutidine conformer reached the C2v minimum, 0.32
kcal/mol lower. In full refinements of ten species:

- *2,6-lutidine* (gas): the spurious conformer 0.32 kcal/mol up, a wB97X-3c
  saddle, is gone; F +0.46.
- *N-methylacetamide* (gas): the lowest conformer is 0.10 kcal/mol lower and a
  minimum (lowest frequency 43 cm⁻¹; it was a saddle, −26 cm⁻¹).
- *2-butanone* (gas): its mirror pair folds into the symmetric shape between
  them, 0.005 kcal/mol up and a saddle (−37 cm⁻¹); count 2 → 1, F +0.32. With
  one push no start reached that shape, so fold had nothing to fold the pair
  into.
- *p-Toluate* (water): same count, lowest conformer 0.016 kcal/mol higher (a
  nearly free methyl).
- CCl3COEt, N-ethylpivalamide, tBuCOEt, N-methylpivalamide, perfluorobutane and
  the acetylacetone enol (gas): unchanged.

The ten refinements took 989 optimization steps against 968 with one push.

**Earlier measurements (old vectors).** The frequencies were right (only the
vectors were wrong), so the cutoff results stand. The rest are kept for what
they decided; the amplitude results above replace their numbers.

- *Traps* (2026-10-30). The softest symmetry-breaking GFN2 mode of 17 known
  wB97X-3c traps (found with DFT Hessians) was at most +38 cm⁻¹, often a real
  mode: GFN2 calls most of these hilltops minima, but its soft mode still points
  the right way. A 0.1 Å move freed 7 of the 8 traps whose CREST conformer was
  symmetric. A 0.03 Å move often stalls, because the slope there is too small to
  fail the optimizer's convergence test.
- *One direction* (2026-11-08). On 153 symmetric conformers, each optimized
  unnudged and nudged both ways, no minimum was found by one direction alone.
  The direction choice is safe at any amplitude: the operation that breaks the
  mode carries the + start onto the − start (for a mirror; see Limits for a
  threefold axis), so the two reach one conformer or its mirror image, which
  deduplication counts once. 0.1 Å stalled in some, among them both tert-butyl
  amides' rotation saddles (0.63 and 0.77 kcal/mol) in both directions.
- *Settling* (2026-11-08). Settling took 2–12 steps and replaced 24 results,
  each within 0.03 kcal/mol of the unnudged optimization. A single point at the
  unrelaxed average is no substitute: it lies 0.02–0.07 kcal/mol above the
  relaxed shape at gaps of 0.2–0.3 Å and 2.4–8 kcal/mol at 0.6–1.0 Å.
- *Cutoff* (2026-11-08). Of 34 small conformers whose breaking mode lay at
  100–400 cm⁻¹, two were saddles the nudge freed: H2O2's trans shape (251 cm⁻¹)
  went into the gauche minimum the ensemble lacked, and hydrazine (307 cm⁻¹) into
  a conformer 3.9 kcal/mol lower that it already had. A cutoff of 300 cm⁻¹ takes
  the first, at about 4% more optimization steps than 100.
- *No symmetry* (2026-11-08). Conformers with no symmetry operation are left
  alone. Nudged along an imaginary GFN2 mode (66 conformers), they found no
  minimum the ensemble lacked. No symmetry holds their optimizer in place, so
  the push only picks a basin by GFN2's lights: the + direction lost a
  microstate's lowest conformer in 3 of 35 microstates and lowered none.

## Cost

A nudged conformer costs one GFN2 Hessian, which is cheap next to a
refinement optimization, and more refinement steps. Over all 152 rows at
0.3 Å, it took 1.70× the optimization steps it took unnudged (17.9 against 10.5
on average, settling included), and the 46 MNSol rows alone also took 1.70×
(15.4 against 9.1). The 63 plain rows, where the push changed nothing, still
took 1.36×.

In the 212 MNSol runs refined so far, 278 of 832 sampled conformers (33%) are
nudged, in 163 of the runs. That puts the extra at 12–23% of refinement's
optimization steps on MNSol. The upper end assumes the other conformers converge
as fast as these did unnudged (9.1 steps). The lower end assumes they take the
21.8 steps that conformers with no symmetry took in an earlier test. The runs
keep no step counts to check either. The second push adds about 2.6 steps to
the 9% of MNSol's sampled conformers that have a later breaking mode under
`SECOND_CUTOFF` (1.16× the one-push steps on the 16 MNSol conformers tested,
0.99× on 57 others), 1–2% more refinement steps. Most nudges change nothing: 15 of the 46 MNSol rows
ended more than 0.01 kcal/mol lower (up to 2.6), and none higher.

## Limits

- **Settling discards a symmetric shape when the nudged result is deeper.**
  Whether the symmetric shape was a saddle or a minimum of its own would take a
  refinement-level Hessian, which the nudge exists to avoid. A minimum lost this
  way is one the push crossed out of, so its barrier lies within `AMPLITUDE` of
  the symmetric shape and the deeper wells lie beside it: a dimple inside their
  basin, which a harmonic count would add as a separate state at its own risk.
  No such case turned up.
- **A threefold axis alone does not make + and − equivalent.** It turns a
  degenerate mode by 120°, so the two directions are not images of each other.
  They differ only by the warping of the valley around the axis, and every
  minimum that warping makes is an image of the others. A molecule whose
  valley has two kinds of minimum around a threefold axis (a sixfold warping
  term stronger than the threefold one) would find only one of them.
- **A second soft site that GFN2 calls stiff is not pushed.** Later breaking
  modes are pushed only below `SECOND_CUTOFF`. A second group on a hilltop
  whose GFN2 mode lies above it stays there, as it did for every site before
  the second push (N-methylacetamide's methyl, 0.10 kcal/mol too high, and
  2,6-lutidine's, F 0.46 too low). The band from 150 to 300 cm⁻¹ was tried only
  with every mode rotated together, where the one gain was 2-butyne, 0.014
  kcal/mol lower with the same count.
- **No symmetry, no nudge.** A conformer sampled without a symmetry operation
  is optimized as sampled, and can still end on a saddle. It can stop where the
  slope is gentle (3 of the 47 conformers in the MNSol recount), or optimize
  onto a symmetric saddle, which it then cannot leave (N-methylpivalamide, 0.23
  kcal/mol).
- **A saddle that GFN2 calls stiff is missed.** Planar guanidinium is a
  wB97X-3c saddle (−143 cm⁻¹) for a twist that ends 0.05 kcal/mol lower. GFN2's
  softest breaking mode there lies at 383 cm⁻¹ (322 in water), above
  `SOFT_CUTOFF`, and the twist is its third breaking mode, so a higher cutoff
  alone would not reach it. The well is under `SETTLE`, so the count is the same
  either way.
- **A nearly free methyl is counted by the shape sampling hands over.** Settle
  returns a twisted methyl to the symmetric shape it was sampled with. That is
  the whole answer when the frame has a twofold axis through the methyl
  (toluene, p-toluate, 4-methylpyridine): the symmetric shapes 60° apart are one
  conformer. On a frame without that axis they are two, and sampling hands over
  one of them. p-Cresol comes out with a count of 1; p-toluic acid, whose methyl
  is sampled twisted and so is never nudged, with 2, for the same rotor. A
  methyl sampled just outside the symmetry tolerance is not settled either, and
  is counted twice: toluene in octanol and hexadecane, with a mirror RMSD of
  0.13 Å against 0.125. Treating H-tops on their own in the symmetry analysis
  would settle all three, and is not part of the nudge.
- **A planar frame is nearly always a candidate.** A plane holding the heavy
  atoms is usually a mirror, so a conformer with one and an out-of-plane mode
  under `SOFT_CUTOFF` is nudged, and most come straight back. Of the 278 nudged
  MNSol conformers, 209 have a planar heavy-atom frame (33 are planar outright),
  so these make up most of the cost above.
