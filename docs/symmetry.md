# Symmetry and conformer counting

A conformer search returns *structures*. A partition function is a sum over
*states*. Those differ whenever a molecule contains identical atoms, and the
gap between them is worth a few tenths of a pKa unit — systematically, and in
opposite directions for acids and bases, so it does not average out.

This document describes how the pipeline closes that gap. Two mechanisms do the
work, and they are independent:

- **Deduplication** removes structures that are other structures relabelled.
- **Multiplicity** gives each survivor the weight it deserves.

Everything lives in `tautomator/conformer_symmetry.py`, driven from
`ensemble.deduplicate_charge_state()`.

---

## The problem

Take methylammonium, `CH3NH3+`. Its three methyl hydrogens can be renamed among
themselves (3! ways) and so can the three ammonium hydrogens, giving 36 valid
relabellings. Apply all 36 to an optimized structure and you get **12 distinct
coordinate arrays** — twelve different points in configuration space.

All twelve are the same physical state. The hydrogens are indistinguishable, so
renaming them does not produce a new state to sum over. A conformer search that
returned several of them, and a partition function that counted each, would
overcount.

The correction is standard statistical mechanics — the same indistinguishability
argument that puts a 1/N! in front of an ideal-gas partition function:

```
Q = (1/|Aut|) x (integral over all labelled configuration space)
  = (1/|Aut|) x sum over unique wells i of (|Aut| / sigma_i) q_i
  = sum over unique wells i of q_i / sigma_i
```

`|Aut|` cancels. What survives is `1/sigma_i` — nothing else. Two consequences
follow, and both are easy to get backwards:

- **The count of relabelled copies is not a weight.** It appears and cancels in
  the same step. Weighting by it introduces a spurious `|Aut|` that differs
  between an acid and its conjugate base, which cancels the very effect it looks
  like it should capture.
- **Symmetry lowers entropy.** A conformer with `sigma > 1` has *fewer*
  distinguishable arrangements than an asymmetric one, so it is penalised, not
  rewarded.

---

## sigma is a property of the conformer, not the molecule

`sigma_i` counts the relabellings that map **one conformer** back onto itself
under a proper rotation. It is not the molecule's point group and it is not a
product of internal-rotor symmetry numbers.

A molecule can be full of symmetric groups and still have `sigma = 1` in every
conformer it adopts. Consider 4-*tert*-butylbenzyl alcohol,
`OCc1ccc(C(C)(C)C)cc1`: a two-fold ring flip, a three-fold *tert*-butyl rotation
and three methyl rotations. Every conformer has `sigma = 1`, because none of
those operations maps the *whole structure* onto itself — flipping the ring
leaves the CH2OH and *tert*-butyl groups pointing somewhere new.

Those rotor degeneracies are real, and they are handled — by deduplication.
Rotating the *tert*-butyl by 120 degrees produces a structure that is the
original relabelled, so if the conformer search finds both, they merge.

The rule for which mechanism catches what:

| Operation | Is it a symmetry of the whole conformer? | Handled by |
|---|---|---|
| methyl 120 deg, *tert*-butyl 120 deg, ring flip with off-axis substituents | no, `sigma = 1` | deduplication |
| ring flip with linear substituents on the flip axis | yes | `sigma` |
| C3 turn of a staggered `CH3NH3+` | yes, `sigma = 3` | `sigma` |

Because `sigma` is geometric, it is measured rather than looked up: enumerate the
relabellings consistent with the structure's own interatomic distances, and count
the ones that reproduce it under a proper superposition. That degrades gracefully
where a Schoenflies label falls off a cliff — a slightly puckered ammonium still
returns 3.

---

## Mirror images

A mirror image is the one relation that is **not** a relabelling. No renaming of
atoms turns a structure into its reflection; that requires an improper operation.
So a chiral conformer and its mirror are two distinguishable states, and two
distinguishable states carry `R ln 2` of entropy.

Deduplication **merges** mirror pairs anyway, and compensates with a factor of
two. They are isoenergetic by symmetry, so optimising and scoring both spends the
work twice — and lets two independent Hessians disagree about a vibrational
spectrum that must be identical. One structure plus a factor of two is both
cheaper and more accurate.

Both accountings give the same total, which is what makes the merge safe:

```
two entries at multiplicity 1   ==   one entry at multiplicity 2
```

### Two factors of two, and they must not both fire

The pipeline collapses enantiomers twice over, at different levels, and the two
corrections are disjoint by construction:

| | Collapsed by | Compensated by |
|---|---|---|
| **Configurational** — R vs S, different SMILES | `stereo.deduplicate_enantiomers` | `Microstate.includes_enantiomer` |
| **Conformational** — gauche+ vs gauche− of one achiral SMILES | `conformer_symmetry.deduplicate` | `Conformer.multiplicity` |

Reflection either inverts a stereocentre — in which case the microstate-level
flag owns the factor and `conformer_multiplicity` returns 1 — or it does not, in
which case the conformer level owns it. Never both. `conformer_multiplicity`
takes `includes_enantiomer` precisely so it can stand down.

---

## Where it runs

`ensemble.deduplicate_charge_state(cs, ethr_kcal=None)` is the single entry
point. It deduplicates **within each microstate** — conformers of different
microstates are different species and are never compared — then recomputes every
survivor's multiplicity.

**Sampling** (`sampling._dedupe_add_rrho_and_filter`) deduplicates *before*
computing xTB Hessians, so none is spent on a structure that is another one
relabelled. It passes `ethr_kcal=0.05`, adding the energy criterion CREGEN pairs
with its RMSD threshold: CREST geometries are not converged tightly enough for
proximity alone to mean two structures share a well.

**Refinement** (`refinement.refine`) runs in two passes. Every conformer is
optimized first, then duplicates are collapsed, then only the survivors get a
Hessian. DFT optimization routinely relaxes distinct sampled conformers onto one
minimum — around 40% of the ensemble — so deduplicating between the passes avoids
paying for the same frequencies twice. No energy criterion is used here;
optimized duplicates agree to well under 0.01 kcal/mol anyway.

**Scoring** does not deduplicate. It does not move geometries, so the symmetry
analysis from refinement still holds and the multiplicities carry through.

### Ordering is load-bearing

Deduplicate first, then compute multiplicity. Applied to a list that still
contains duplicates, multiplicity counts a state twice and then doubles one of
the copies. `deduplicate_charge_state` does both in the right order; call it
rather than reaching for the pieces.

---

## How the weight is used

`Conformer.multiplicity` is `n_states / sigma`. Combined with the microstate
flag by `ensemble._conformer_multiplicity`, it enters three places:

- `charge_state_free_energy` — `Z = sum_i m_i exp(-G_i / RT)`
- `assign_weights` — Boltzmann weights within a charge state
- `filter_charge_state_by_energy` — the energy window

The window is worth a note. A conformer contributes `m exp(-G/RT)`, which is
`exp(-(G - RT ln m)/RT)`, so the window is measured against `G - RT ln m` rather
than `G`. A conformer standing for two states is effectively 0.41 kcal/mol lower
than its bare free energy suggests and should survive the window on that basis.

---

## Inspecting the result

Multiplicities are small numbers with exact expected values, so they are worth
checking by hand on a new species:

```python
from tautomator.conformer_symmetry import symmetry_number, conformer_multiplicity

sigma, achiral = symmetry_number(coords, symbols)
m = conformer_multiplicity(coords_stack, symbols, includes_enantiomer=False)
```

Expected values:

| Conformer | sigma | achiral | multiplicity |
|---|---|---|---|
| staggered `CH3NH3+` | 3 | yes | 1/3 |
| planar `ArO-` with a two-fold axis | 2 | yes | 1/2 |
| a gauche conformer of an achiral molecule | 1 | no | 2 |
| any conformer of a molecule with a fixed stereocentre | 1 | no | 1 |
| a typical asymmetric conformer | 1 | yes | 1 |

They are serialized per conformer in `ensemble.json`, so a finished run can be
audited without recomputation.

---

## Cost

The search never enumerates the permutation space. Atoms are grouped by an
invariant (element plus sorted distances to every other atom), then a
backtracking search assigns them most-constrained-first, abandoning a branch on
its first inconsistent distance. Two screens run ahead of it: a singular-value
lower bound on RMSD, which is rigorous and rejects pairs that cannot match under
any relabelling, and an element-typed sorted-distance descriptor.

In practice this is milliseconds per conformer. A 254-atom peptide macrocycle
with more than 10^300 element-preserving permutations resolves in about 50 ms;
deduplicating a 24-conformer ensemble of it takes 0.1 s.

---

## Assumptions and limits

- **No molecular graph is used or perceived.** A permutation preserving every
  interatomic distance necessarily preserves connectivity, since bonds are short
  distances. This matters because bond orders and formal charges cannot be
  perceived reliably for charged, tautomeric species, and because a stored SMILES
  goes stale the moment a proton migrates during optimization.
- **Sorted-distance descriptors are screens, never decisions.** Distinct
  structures can share a distance multiset (homometric sets), so every candidate
  that passes a screen is verified by explicit superposition. No screen can
  admit a wrong answer; see *Tolerances* below for the direction that is not
  guaranteed.
- **Enantiomeric conformers are discarded, not tracked.** Correct for pKa in an
  achiral solvent. Anything chiral — a chiral solvent, CD spectra, a chiral
  binding site — needs `deduplicate(..., merge_mirrors=False)` and a different
  multiplicity rule.
- **Conformer coverage carries what symmetry does not.** Where a protonation
  change breaks a rotor symmetry, the resulting entropy difference appears as
  *more distinct conformers* on one side, not as a symmetry number. Ethylamine
  has three distinct C–N conformers where ethylammonium has one; that factor of
  three is worth 0.48 pKa units and is captured only if the conformer search
  finds all three. Rotamer coverage is irrelevant — those are the same state —
  but conformer coverage is not.

---

## Tolerances

Two constants bound the candidate search: `DIST_TOL`, how far a pairwise
distance may move before a mapping is discarded, and `DESC_TOL`, the same cutoff
for the descriptor screen. Both are 0.8 A.

They are **not symmetric in their failure modes**. Loosening them costs only
time, because every survivor is verified by explicit superposition. Tightening
them can lose a real duplicate, and that error is silent and one-directional: a
missed duplicate contributes twice to the partition function, worth
`RT ln 2` = 0.41 kcal/mol.

The guarantee has a limit, and it is worth stating precisely. `rthr` is an
**RMSD**, so the single-atom displacement it tolerates grows as
`rthr * sqrt(N)`, while `DIST_TOL` and `DESC_TOL` are fixed distances. Past
roughly 40 atoms a pair can sit inside `rthr` with one atom displaced further
than the screens allow, and be split. Two structures related that way -- almost
all of an RMSD budget spent on a single atom -- are not what optimization
produces; CREGEN-level duplicates agree to about 0.02 A. But the possibility is
real rather than hypothetical, and the current tolerance is a wide margin rather
than a proof.

Scaling the tolerances with `rthr * sqrt(N)` would close the gap and is not
viable: the backtracking search depends on tight distance pruning, and at 254
atoms the scaled tolerance takes the search from 0.02 s to over two minutes
without changing a single grouping on real data. 0.8 A was chosen as the widest
value that stays flat in cost (2.5x over the whole pKa set, unchanged
results) while comfortably covering the displacements optimization actually
produces.

`MAX_MAPPINGS` caps the enumerated mappings at 20,000. Reaching it truncates the
search, which can undercount `sigma` or miss the best mapping, so it emits a
warning rather than being absorbed. Real molecules are far below it -- a
254-atom macrocycle enumerates one mapping, cubane 48.

---

## Further reading

- Gilson & Irikura, *Symmetry Numbers for Rigid, Flexible, and Fluxional
  Molecules: Theory and Applications*, J. Phys. Chem. B **2010**, 114,
  16304–16317 (and the 2013 correction, DOI 10.1021/jp401194k). The reference
  for symmetry numbers of molecules with thermally active internal rotors, and
  for why conformer enumeration and internal-rotor symmetry numbers must not
  both be applied.
- Fernández-Ramos, Ellingson, Meana-Pañeda, Marques & Truhlar, *Symmetry numbers
  and chemical reaction rates*, Theor. Chem. Acc. **2007**, 118, 813–826. Framed
  for transition state theory; its sections on chiral species and on multiple
  conformers map directly onto the two mechanisms here.
- Pracht & Grimme, *Calculation of absolute molecular entropies and heat
  capacities made simple*, Chem. Sci. **2021**, 12, 6551–6568. Conformer
  ensembles and conformational entropy from the authors of CREST and CREGEN.
