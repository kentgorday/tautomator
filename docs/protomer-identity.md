# Protomer identity

Two SMILES can describe the same chemical species. RDKit will canonicalise them
apart anyway, because in its model they are different molecular graphs. When
that happens to a microstate, the pipeline computes the same species twice —
two conformer ensembles, two sets of DFT energies, and a partition function
that double-counts it.

`protomer_key` in `tautomator/rdkit_utils.py` is the identity used for microstates
in the RDKit-first path. It gives one key per species, whichever Lewis
structure was written for it.

## The problem

A 4-substituted imidazolium carries its cation on a delocalised N–C–N unit.
Both nitrogens bear a hydrogen, and the `+` can be written on either one:

```
[H]c1c(R)n([H])c([H])[n+]1[H]
[H]c1c(R)[n+]([H])c([H])n1[H]
```

These are one species. Canonical SMILES makes them two. In histamine the
enumerator emitted both at q=+1 and again at q=+2; a delocalised tetronic-acid
enolate did the same thing at three charge states, once as a three-way split.

The same shape appears wherever charge is delocalised: imidazolate,
tetrazolate, amidinium, benzimidazolium, enolates of 1,3-dicarbonyls.

## The key

Resonance moves formal charge and bond order, and nothing else. It cannot break
a heavy-atom bond or relocate a hydrogen. Our enumeration is restricted the same
way — it only adds and removes H at heteroatoms. So across every microstate of
one molecule the heavy-atom framework is fixed, and

> (framework, per-atom H count, net charge, stereochemistry)

identifies a species uniquely, whatever Lewis structure it was written as.

`protomer_key` computes exactly that: erase what resonance can change, keep what
it cannot, canonicalise the result.

1. Build the **skeleton** — every bond reduced to single, every formal charge
   zeroed, aromaticity and stereo stripped, per-atom hydrogen counts pinned as
   explicit.
2. Canonicalise it. This gives both the framework string and a canonical atom
   ranking.
3. Append the **net charge** and the **stereo descriptors**, as CIP labels keyed
   by each atom's rank in the skeleton.

Hydrogen counts are what keep genuine tautomers apart: 4-methylimidazole and
5-methylimidazole have the same framework but different per-atom H, so they
never merge. The same holds for two different protonation sites.

### Why the skeleton, and not canonical SMILES

RDKit's canonical atom ranking depends on formal charge. It does *not* depend on
bond order — aromatic and Kekulé forms of one molecule rank identically — but
charge is enough. Two resonance forms of one molecule therefore canonicalise to
two *different* atom orderings.

That matters because it would make the stereo labelling depend on which Lewis
structure was handed in. Ranking on the skeleton avoids it: the skeleton is
byte-identical across every resonance form of a molecule, so the ranking is too.

### Why not a resonance enumeration

The obvious alternative is to enumerate resonance structures with
`Chem.ResonanceMolSupplier` and take a canonical minimum over the orbit. It
works, and it produces the same merges, but it rests on a much larger surface:
the enumeration is order-dependent and incomplete (3-ethylphenolate has four
resonance forms; the supplier returns three, and which three depends on atom
ordering), truncation at `maxStructs` is silent, and the flag choices interact
with obligatory charge-separated groups.

Erasing charge and bond order needs none of that. It also handles the
charge-separated groups *better*: sulfoxide `S=O` ↔ `S⁺–O⁻`, amine oxides, nitro
and phosphine oxides all merge without any special-casing, where a conservative
resonance enumeration leaves them apart.

## Where it runs

`enumerate_charge_state` collapses its BFS output by `protomer_key`, and
`run_approach1` collapses again after tautomer expansion, since both stages can
introduce the duplication. `deduplicate_protomers` chooses the survivor
deliberately rather than by sort order, because it seeds both stereoisomer
enumeration and the ETKDG geometry: fewest formally charged atoms first (the
dominant resonance contributor, which keeps a valid-but-absurd `CC([O-])=[OH+]`
from standing in for acetic acid), then the most perceivable stereo elements
(a delocalised anion drawn with its double bond in one position can expose a
stereogenic bond the other hides), then canonical SMILES for determinism.

The CREST-first path needs none of this. It identifies microstates by
`geometric_fingerprint` -- hydrogen count per heavy atom plus tetrahedral
configuration, both read off the geometry -- which contains no charges and no
bond orders, so two resonance forms produce the same fingerprint by
construction.

Configuration is there because protonation creates and destroys stereocentres.
A tertiary amine inverts freely and is not a stereocentre; protonating it
quaternises the nitrogen, which cannot invert without breaking a bond, so the
two faces become distinct species. With a second stereocentre present they are
diastereomers rather than enantiomers, and a hydrogen-count fingerprint cannot
tell them apart.

Parity is recorded at heavy atoms with four connections and **at most one
hydrogen**, with neighbours ordered `(is_hydrogen, position)`. That gate admits
the ordinary carbon stereocentre -- three heavy neighbours and one hydrogen,
which any rule counting heavy neighbours alone would miss -- and the 1,4-ring
carbon, whose two ring branches are constitutionally identical so that no
constitutional refinement can separate cis from trans: the configuration lives
in the *pair*, and a parity at each captures it. It excludes atoms carrying two
or more hydrogens, where the hydrogens are interchangeable and their ordering is
not stable across CREST outputs -- the only case that could flip for no chemical
reason. Recording a parity at an atom that is not a stereocentre is harmless: it
is a constant for that species and never splits anything.

Everything on this path is indexed by heavy-atom position, so structures must
share a heavy-atom order. `heavy_frameworks_agree` checks the connectivity, not
just the element sequence -- two different orderings can share one sequence.

## When a proton moves during minimisation

The key above identifies a species from a SMILES. `tautomator/protomer_geometry.py`
identifies one from *coordinates*, which is what catches a conformer whose proton
migrated between heavy atoms while xTB or DFT was minimising it. It is not rare:
across the first pKa-set batch about a quarter of conformers at both stages were
filed under a protomer their geometry no longer matched, and nothing detected it.
A protonated carboxylic acid handing its proton to a free amine is the typical
case, and the resulting ammonium/acid form is the obviously correct structure --
the enumerated label was the absurd one.

Ownership is decided by **nearest heavy atom, with no distance cutoff**. That is
enough because it is a deterministic function of the coordinates, and because
where a proton is genuinely shared the two candidates are chemically alike: when
they are alike by automorphism -- the two oxygens of a carboxylate, the two ends
of maleate -- canonicalising the skeleton absorbs the choice and both assignments
name the same species. Where they are inequivalent the choice is arbitrary and is
left so: `charge_state_free_energy` sums flatly over every conformer of every
microstate, so which microstate holds a conformer moves the answer only through
`includes_enantiomer`, at most kT ln 2 = 0.41 kcal/mol. The margin between first
and second nearest heavy atom is recorded for diagnosis but nothing branches on
it, so there is no threshold to tune.

Repair runs before deduplication at both stages, for the same reason
deduplication precedes the Hessians: a migrated conformer is another microstate's
structure under the wrong label, and re-filing it first lets it collapse against
that microstate's own conformers instead of buying a Hessian, and later a DFT
optimization, for a duplicate.

### Why every protomer shares a heavy-atom order

`rdkit_utils.frame_atom_order` ranks heavy atoms canonically on the *frame* --
the heavy-atom graph with hydrogen counts, charges, bond orders, aromaticity and
stereo all erased -- which is byte-identical for every protomer of a molecule.
`smiles_to_3d` embeds in that order. Without it each microstate inherits the
canonical order of its own SMILES, and those diverge, because RDKit's canonical
ranking depends on formal charge and hydrogen count; re-filing a conformer would
then require discovering an atom correspondence first. Hydrogens still interleave
differently, since SMILES writes each one attached to its heavy atom, so a moved
conformer's hydrogens are regrouped -- through the *H-pinned* skeleton's ranking,
which is what decides which automorphic site the proton sits on.

### What is not repaired

Four outcomes end in `excluded_conformers` rather than being dropped, so a run
records what it computed and could not place:

**A broken heavy-atom framework.** `heavy_components` splits the heavy-atom
connectivity into connected components; more than one means the structure came
apart, and the energy belongs to a complex rather than to the species its label
names. This is checked *first*, in both approaches, because neither identity can
see it. Approach 1 builds its key from the **template's** bonds and only the
hydrogen counts from the coordinates, so breaking a C-O and pulling the fragment
4 A away yields a byte-identical key. Approach 2 computes the framework for its
parities but does not hash it, so a fragment and the intact species share a
fingerprint — and at sampling they merge into one group, where the fragment can
become the representative a full conformer search is run on.

It is also the check a detached *hydrogen* cannot make: an oxygen that leaves
takes its hydrogens with it, so every H stays ~0.97 A from a heavy atom and
nothing looks wrong. Re-reading the first end-to-end batch, 6 of the 8
`no_matching_microstate` exclusions were fragmentations that failed to match only
by luck, plus one more that `proton_detached` had caught for the wrong reason;
every conformer that was *kept* is single-component, so nothing previously
retained becomes excluded. The criterion is the one `heavy_framework` already
used for approach 2's ordering guard, so no new threshold is introduced.

In approach 2 this runs as a pre-pass, ahead of `heavy_frameworks_agree`. That
check is about heavy-atom *ordering*, and its response to a disagreement is to
skip the whole charge state — so a fragment used to fail it for an unrelated
reason and suppress migration repair for every intact conformer at that charge.
Worse, the reference is whichever conformer is reached first, so a fragment
arriving first made the healthy majority look like the disagreement.

**A detached hydrogen.** The energy is real but belongs to a fragmented species.
Two conformers in the first batch had hydrogens 5-6 A from every heavy atom, both
carrying Boltzmann weight 1.0.

**No matching microstate.** Approach 1's microstate set is *prescribed* by the
enumerator, and a geometry outside it cannot be labelled without perceiving bond
orders from coordinates. C-protonated arenium is the observed case; most of what
used to land here was in fact fragmentation, and is now diagnosed as such. Unlike every
other exclusion in the pipeline this one discards a valid energy for a real
species; `multiplicity` records the size of the loss. Approach 2 has no such case:
its microstates are *discovered* from the geometry, so an unseen identity opens
a new microstate instead.

**An ambiguous destination.** The geometry became one of several microstates,
none of them the one it was filed under, and nothing separates them. Rare enough
that nothing in the first pKa-set batch reached it.

A tie that *includes* the microstate the conformer already sits in is not an
exclusion: it is a failure to discriminate rather than evidence of a change, so
the conformer stays. It costs nothing to leave it there -- the charge-state free
energy sums flatly over microstates, so which one holds a conformer moves the
answer only through `includes_enantiomer`. The usual cause is a skeleton
automorphism that is not a symmetry of the stereochemistry: in
`O=C(O)/C=C(/[O-])O` both terminal carbons are "two oxygens plus the middle
carbon", so the ends are interchangeable to the skeleton though not to the
molecule. Where the competing microstates also differ in hydrogen distribution
the layout step rejects the wrong correspondence and this never arises; where
they share one, as an E/Z pair does, nothing here can separate them. 31 of 6154
conformers at sampling, concentrated in one molecule.

### Stereochemistry is a tie-break, not part of the key

The skeleton cannot hold stereo. Every bond is single, so a double bond has no
configuration to annotate, and erasing charge and bond order can remove a
stereocentre outright by making two substituents identical -- which is the same
erasure that merges resonance forms, working as intended:

```
C[C@@H](C(=O)O)C(=O)[O-]     real stereo elements 1, skeleton 0
C[C@H](/C=C/C)/C=C\C         real stereo elements 3, skeleton 0
```

`protomer_key` handles this by appending CIP labels taken from the real molecule
and keyed by skeleton rank. That works because it only ever compares *resonance
forms*, which have identical hydrogen counts and therefore identical skeletons,
so the ranks mean the same thing on both sides.

The geometry path cannot do the same. It compares *different protomers*, whose
skeletons differ, so skeleton ranks are not comparable between them -- and there
is no CIP descriptor to read from bare coordinates in any case. So stereo stays
out of the protonation key and is applied only when the key leaves several
candidates, by `match_to_candidate`: impose each candidate's own bond orders on
the coordinates, ask `AssignStereochemistryFrom3D` what configuration they show,
and keep the candidates that agree with themselves. Each hypothesis is tested on
its own bond orders, so the migration having invalidated the *source's* bond
orders does not matter -- no bond is ever perceived from coordinates.

Which stereo elements enter that comparison is the whole difficulty. Each
candidate is scored on the **intersection** of two sets:

**What the candidate specifies** -- `_specified_stereo` reads the template's own
bond graph for double bonds carrying E/Z and heavy atoms carrying a chiral tag,
keyed by heavy-atom position rather than atom index so the two sides are
comparable. A bond that is *single* under this candidate's bond orders is free to
rotate and has no configuration to state, so it never appears.

**What the coordinates determine** -- the same reading taken from the mol after
`AssignStereochemistryFrom3D`. Intersecting is what keeps two failure modes out.
A migration can *create* a stereocentre -- `N/C(=C/C(F)(F)F)C([OH2+])[OH2+]` has
two identical substituents on that carbon, so it is not a centre until a proton
leaves -- and the candidate that does not constrain it must not be asked about
it. Conversely `AssignStereochemistryFrom3D` annotates whatever the coordinates
support, including bonds no label constrains: a protonated carbonyl is a
stereogenic double bond, and demanding a match there would reject every
candidate.

Everything outside the intersection is **erased** before comparing, by
`_stereo_signature`, which rewrites the mol to a canonical SMILES carrying only
the selected elements. Erasing rather than comparing is the point: the
comparison must not be able to fail on an element neither side was asserting.

An empty intersection is not a match on the merits. `match_to_candidate` returns
a `verified` flag saying whether any element was actually compared, and
`_repair_labelled` refuses to let an unverified candidate outrank a verified one
-- a candidate constraining nothing cannot be contradicted, and would otherwise
win every tie by default.

When the microstate stands for a collapsed enantiomeric pair
(`includes_enantiomer`) the mirror image is accepted too, because that is what
the microstate means. `_mirror` inverts every tetrahedral tag and leaves E/Z
alone: reflection inverts configuration at every centre but does not turn cis
into trans, and inverting bond stereo as well would make the microstate accept
the wrong diastereomer.

### Atom ordering inside the tie-break

The coordinates are attached to the candidate's template by *position*, with no
substructure mapping at attach time: atom *i* of the laid-out geometry must be
atom *i* of the template, hydrogens included. `_layouts_for_target` is what makes
that true. It walks `_heavy_slots(target_template)` -- the template's atom layout,
one entry per atom, `None` for a heavy atom and the owning heavy position for a
hydrogen -- and fills each slot from the geometry, so a migrated conformer's
hydrogens are regrouped into the destination's interleaving rather than kept in
the source's.

Hydrogens sharing one heavy atom are filled first-come, which is arbitrary among
themselves and has no consequence: they are interchangeable by graph
automorphism, so no descriptor anywhere in the molecule can depend on which is
which. An atom carrying two or more hydrogens cannot be a tetrahedral centre,
and the two hydrogens of a diastereotopic CH2 have equal canonical rank, so they
cannot rank a neighbouring centre's substituents either. Permuting them leaves
`_stereo_signature` byte-identical; there are regression tests.

More than one layout is possible when the skeleton has an automorphism that
preserves the hydrogen distribution, and then the choice is *not* free -- it
decides which of two symmetry-related sites a proton sits on, and a
pseudo-asymmetric centre reads differently under each. Matches are filtered to
those preserving hydrogen counts, because the matcher ignores them and would
otherwise map a protonated site onto an automorphic bare one.

**When the geometry's positionwise hydrogen vector equals the target's, the
identity layout is built directly and returned alone, and no search runs at all.**
Equal vectors mean the two skeletons are the same graph, so the identity is an
isomorphism; and because the geometry carries the source template's atom order
and nothing in the pipeline permutes atoms, it is the correspondence the atoms
actually have. The automorphisms are relabellings with no physical content for
that conformer. Equal vectors also say no proton changed owner, so the search is
needed only when one did -- which is the only situation in which a genuine
ambiguity can arise.

Constructing the identity rather than picking it out of the match results matters
for two reasons beyond the cost. Correctness would otherwise depend on the
identity appearing within `max_layouts`: a tris-CF3 alcohol's skeleton has 1296
automorphisms against a cap of 64, so relying on it being among them rests on
RDKit's match ordering rather than on anything guaranteed. And the cap warning is
emitted before we know whether any automorphism matters, so a clean run would
print an alarming line about orderings that were never needed. Over the whole
first pKa-set batch -- 742 stored conformers -- `GetSubstructMatches` is now
called zero times.

This is *not* the `same_microstate` shortcut removed earlier. That one keyed on
the protonation key, which is invariant under automorphism and so could not tell
whether the hydrogens sat on the positions the template actually uses; the
positionwise vector can, and is exactly what the hydrogen filter compares. The
pseudo-asymmetry case that motivated removing the shortcut is a real migration,
so its vector does not match and it still goes through the search -- there is a
regression test pinning that.

Why it matters: offering the automorphisms beside the identity let each candidate
pick *its own* correspondence, since `match_to_candidate` returns on the first
layout that agrees. "Verified" then meant only "there exists some correspondence
under which this looks right", and two candidates could each find a different
witness. For `O=C(O)/C=C(/[O-])O` the end-swapping automorphism seats the
template's C=C on the coordinates of the C-C single bond, and
`AssignStereochemistryFrom3D` duly reports an E/Z descriptor computed from a
freely-rotating torsion: half of all torsions made the wrong isomer verify.
Replaying the repair over the batch with the identity preferred takes the
q=-1 counts from 16 undiscriminated ties to 0, and re-files 7 conformers the old
code had left under the wrong label.

Worth being exact about the consequence, because it is smaller than the defect
suggests. The E and Z microstates there have identical `e_min`, so the
misfiling shifted no energy window and deleted nothing; within a charge state the
partition function sums flatly over microstates, and the two agree on
`includes_enantiomer`. What it cost was noise in the logs and the loss of a
decisive answer. The two conformers the batch *excluded* as ambiguous at q=-1 are
a separate matter: both had protons that had genuinely moved onto symmetry-related
oxygens, so the identity is invalid for them, the search still runs, and both
candidates stay satisfiable. Recovering those needs the exclusion rule to soften
where every viable candidate agrees on `includes_enantiomer`, which is not
implemented.

Because `observed` is a copy of the candidate template with new coordinates, the
two sides of every comparison share one bond graph and one atom indexing. That is
what makes the comparison safe: a chiral tag is a parity relative to the atom's
stored neighbour order, and a bond's E/Z is relative to its stereo atoms, so
neither is meaningful across differently-ordered mols. Sharing the graph also
means `_specified_stereo` can key on the bond alone without recording its stereo
atoms -- only its keys are used to choose what to compare, never its values, and
the values are resolved through `MolToSmiles` in `_stereo_signature`.

Ring cis/trans is worth calling out because it is not a classical stereocentre,
yet the two diastereomers have identical hydrogen counts and therefore share a
protonation key. Only this tie-break tells them apart. Note also that the *frame*
used for atom ordering erases hydrogen counts as well, so it cannot distinguish a
saturated ring from an aromatic one at all -- harmless, since every protomer
shares the frame and so shares its tie-break, but the H-pinned skeleton is what
keeps such branches apart for identity.

Deriving bond orders from the geometry instead, by searching for a resonance
structure that accommodates the observed hydrogen pattern, would be a much larger
problem than this one: the enumerator has already produced the candidate Lewis
structures, so the task is to choose from a known list rather than to construct
one. It would only help for **no matching microstate**, where the species is
outside the model anyway.

## Enumeration rules

Every rule in `charge_enumeration.py` pins the hydrogen count *and* the formal
charge on both sides. An RDKit product template inherits any property it does
not state, so a product written `[NH2:1]` keeps the reactant's `+1`:
`[NH3+:1]>>[NH2:1]` yields an `[NH2+]` whose net charge never reaches the
target, and the rule never fires. Every rule that neutralizes an ion has that
shape, so leaving the charge implicit disables exactly the half of the table
needed to walk a charged input back toward neutral — and, through
`[nH:1]>>[n-:1]`, aromatic N–H deprotonation for neutral inputs too.

### Pooling across stereoisomers

Approach 2 walks each input stereoisomer separately, because CREST's protonation
operations do not preserve configuration: `--deprotonate` removes the hydrogen
from a quaternised nitrogen, at which point the centre ceases to exist and the
amine inverts freely, and `--protonate` rebuilds it on whichever face the physics
picks. A run cannot be assumed to stay in the lane it started in.

Two runs therefore explore overlapping ground, and a shared `tautomer_id` used to
mean "reached twice, keep the better sampling" -- which silently deleted an
entire microstate whenever it instead meant "two diastereomers the identity could
not tell apart". Now that the identity carries configuration, a collision means
one species: the conformers are pooled and deduplication collapses whatever the
two runs found in common.

## Known limitations, not yet acted on

**A specified stereocentre that sampling flips.** Two situations differ, and only
one is handled. Where a stereo element was *introduced* by the pipeline -- by
tautomer enumeration, or created when protonation quaternises a nitrogen that
was freely inverting -- both configurations are legitimate microstates and a
conformer is filed under whichever it actually is. Where the element was
specified *by the user* in the input SMILES, a conformer that comes back with
the opposite configuration is not compliant with what was asked for, and
arguably belongs in `excluded_conformers`.

That exclusion is deliberately **not** implemented. If CREST's sampling flipped
the centre, it is not configurationally stable, and rejecting conformers for
failing to honour a claim the chemistry does not support is the wrong response
-- the more useful reading is that the input over-specified. The two cases are
distinguishable in code (a stereo element is user-specified if it appears in
`Ensemble.input_smiles`, rather than arriving from `enumerate_and_deduplicate`),
so the path stays open.

TODO: decide whether to surface this at all, and if so whether as a warning
against the input or as an exclusion. Revisit before trusting a run whose input
carries hand-assigned stereochemistry.

**E/Z isomerism in the CREST-first path.** Approach 2 identifies microstates by
a geometric fingerprint, which cannot carry double-bond configuration without
knowing that a bond is rigid, and rigidity is not recoverable from coordinates
alone. Measured on the first pKa-set batch: among planar sp2-sp2 C-C bonds,
103 of 610 visit both sides of the torsion *within one microstate* -- they are
rotatable by construction -- and no bond-length threshold separates those from
the rigid ones without either merging distinct species or splitting a third of
affected microstates several ways. Settling it needs a set where the answer is
known, not incidental sampling. Approach 2 therefore conflates E and Z isomers,
as it always has; approach 1 does not.

The descriptor itself is not the hard part: a torsion between index-picked
substituents at two planar three-connected atoms, comparable across structures
that share a heavy-atom order. Only the rigidity test was unresolved, and
**Wiberg bond order settles it** where geometry could not. Measured with
``xtb --gfn 2 --wbo`` over 158 planar sp2-sp2 C-C bonds in 55 microstates:

    rotatable   p5 0.93   p25 1.01   median 1.05   p95 1.32-1.40
    rigid       p5 1.08   p25 1.60   median 1.72   p95 1.85-1.88

The *bulk* separation is large -- medians near 1.05 against 1.72, where bond
length gave 1.465 against 1.378 with fully nested ranges. The tails do overlap
(rotatable p95 exceeds rigid p5), so this is not a clean gap. But the overlap
looks like contamination in the label rather than in the bond order: the nine
"rigid" bonds falling below a 1.4 cut have median length 1.433 A, matching the
rotatable population (1.469) rather than the rigid one (1.341). "Unimodal" only
means the sampler never flipped that bond, so the torsion label is the noisier
signal of the two.

Two further measurements. A bond's WBO moves by about 0.07 across conformers of
one species (median spread 0.068 gas, 0.058 ALPB), and 3-5% of bonds straddle a
1.4 cut within their own conformers -- so rigidity must be assigned per
*species*, from a consensus over its conformers, never per conformer. That
resolves without circularity: group by the coarse hydrogen-count fingerprint
first, which is stereo-blind and needs no bond orders, take the consensus within
each group, then refine that group's fingerprint. Gas and ALPB(water) perform
almost identically; nothing in the data chooses between them.

TODO: four things before building it. (0) Settle the threshold on a set where
the answer is known independently, not on incidental sampling: the two runs done
so far disagree on rotatable p95 (1.32 against 1.40) at n=158 and n=39, which is
too small to set a cut from. (1) Measure C-N and C-O, not just C-C;
an amide C-N is rigid but may sit near the biaryl C-C range, which would force a
per-element-pair cut. (2) Decide where the bond orders come from -- the natural
shape is one ``xtb --sp --wbo`` per *microstate representative*, since rigidity
is a property of the species rather than the conformer, with the rigid-bond set
passed into the fingerprint so it stays a pure function of its arguments.
(3) Note this makes the approach-2 identity depend on an xtb call and its
parameterisation, which the approach-1 identity does not; the gap above is wide
enough that version drift is unlikely to matter, but it is a real coupling.

## Unsupported inputs

`validate_input_smiles` rejects two classes at the pipeline entry rather than
approximating them. Both are refused because nothing downstream would treat them
correctly, and a plausible-looking number is worse than an error.

**Open-shell species (radicals).** `Geometry.multiplicity` assumes the lowest
multiplicity for the electron count, so a triplet would be evaluated as a
singlet and return an energy for the wrong state. *There are no plans to support
them.* The enumerator needs no separate radical guard: pinning the formal charge
on both sides of every rule is what removed the unpaired electrons, which arose
only on products whose charge the template had failed to adjust.

**Multi-component inputs (salts, solvates, mixtures).** Conformer search is
meaningless when fragments translate freely, and the thermodynamic cycle assumes
a single solute. Desalt and submit the component of interest. *There are no plans
to support them.*

Two further limits are **not** checked, because RDKit discards them before the
key is ever computed. Both *may* be supported in future; neither is today.

**Enhanced stereochemistry** (AND/OR stereo groups) is not modelled. Structures
relying on it are treated as having absolute stereo.

**Atropisomerism** is not perceived from SMILES — `FindPotentialStereo` reports
nothing for a tetra-*ortho*-substituted biaryl under either stereo-perception
setting, and `enumerate_stereoisomers` returns a single isomer. It survives only
in CXSMILES (`|wU:7.6|`), which round-trips correctly, but microstate labels are
written as plain canonical SMILES, which discards it. Supporting atropisomers
would require changes to the stereo enumerator and the stored label format, not
just to the key.
