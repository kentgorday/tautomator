"""Protomer identity read off a 3D geometry, and repair of migrated conformers.

`rdkit_utils.protomer_key` answers "which species is this?" from a SMILES.
This module answers it from coordinates, so that a conformer whose proton moved
during minimisation can be recognised and re-filed under the species it actually
became.  Migration is not rare: across the first pKa-set batch, 20 of 380
microstates changed protonation between xTB sampling and DFT refinement, and
nothing detected it.

Ownership is decided by *nearest heavy atom*, with no distance cutoff.  Two
properties make that sufficient:

* It is a deterministic function of the coordinates, so the same geometry always
  yields the same answer.  Where the answer changes, the geometry changed.
* Where a proton is genuinely shared, the two candidates are chemically alike,
  and when they are alike by graph automorphism -- the two oxygens of a
  carboxylate, the two ends of maleate -- `MolToSmiles` on the H-pinned skeleton
  canonicalises over the choice, so both assignments name the same species and
  the ambiguity has no consequence.

The residual case is a proton shared between *inequivalent* heavy atoms, where
the choice is arbitrary and does not cancel.  It is left arbitrary on purpose:
`charge_state_free_energy` sums flatly over every conformer of every microstate,
so which microstate holds a conformer changes the free energy only through
`Microstate.includes_enantiomer` -- at most kT ln 2 = 0.41 kcal/mol, about 0.3
pKa units, and only if the two candidates disagree on that flag.  The band is
also narrow: over 2890 refined conformers the median margin between first and
second nearest heavy atom is 0.9 A, and 0.55% sit within 0.2 A of a tie.
`ProtonAssignment.min_margin` records the margin for diagnosis; nothing branches
on it, so there is no threshold here to tune.

Stereochemistry is deliberately *not* part of the key.  Re-perceiving it would
mean reading stereo off a geometry through a template whose bond orders the
migration has just invalidated.  Microstates that share a protonation key and
differ only in configuration are therefore indistinguishable here; see
`repair_migrated_conformers` for how that case is resolved.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Geometry import Point3D

from tautomator.tautomer_dedup import (
    DETACHED_DISTANCE,
    ProtonAssignment,
    assign_protons,
    geometric_fingerprint,
    heavy_components,
    heavy_frameworks_agree,
)
from tautomator.types import (
    ChargeState,
    Conformer,
    ExcludedConformer,
    ExclusionReason,
    ExclusionStage,
    Geometry,
    Microstate,
)

log = logging.getLogger(__name__)

# Re-exported: the assignment primitive lives in `tautomer_dedup` so that the
# approach-2 fingerprint and this module's migration check cannot drift apart.
__all__ = [
    "DETACHED_DISTANCE",
    "ProtonAssignment",
    "assign_protons",
    "heavy_components",
    "match_to_candidate",
    "protonation_key_from_geometry",
    "protonation_key_from_mol",
    "repair_migrated_conformers",
    "template_from_smiles",
]


# Skip valence checking: a skeleton with every bond reduced to single and every
# charge zeroed is not a valid molecule, and does not need to be.
#
# SANITIZE_FINDRADICALS must stay enabled, counter-intuitive as that is on a
# graph whose valences are deliberately wrong. The deficiency we create is read
# as unpaired electrons, and the resulting radical count is what forces RDKit to
# bracket an atom -- `[O]` rather than `O` -- which is the only thing preserving
# its hydrogen count through canonicalisation. Without it acetic acid and
# ethane-1,1-diol both canonicalise to `CC(O)O`, and the key silently merges
# species with different molecular formulas. See the regression tests.
_SKELETON_SANITIZE = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES


def _skeleton_mol(template: Chem.Mol, counts: tuple[int, ...]) -> Chem.Mol:
    """The heavy-atom framework of ``template`` carrying ``counts`` hydrogens.

    Every bond is reduced to single and every formal charge zeroed, so the
    result depends only on the framework and the hydrogen distribution -- never
    on which Lewis structure was written.  Atom *i* of the result is heavy atom
    *i* of the template, which is what makes it usable as a mapping device as
    well as an identity.
    """
    heavy = [a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy) != len(counts):
        raise ValueError(f"template has {len(heavy)} heavy atoms, geometry has {len(counts)}")

    rw = Chem.RWMol()
    remap: dict[int, int] = {}
    for new_idx, old_idx in enumerate(heavy):
        atom = Chem.Atom(template.GetAtomWithIdx(old_idx).GetAtomicNum())
        atom.SetNumExplicitHs(int(counts[new_idx]))
        atom.SetNoImplicit(True)
        remap[old_idx] = rw.AddAtom(atom)
    for bond in template.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in remap and j in remap:
            rw.AddBond(remap[i], remap[j], Chem.BondType.SINGLE)

    skeleton = rw.GetMol()
    Chem.SanitizeMol(skeleton, _SKELETON_SANITIZE)
    return skeleton


def _key_from_counts(template: Chem.Mol, counts: tuple[int, ...], charge: int) -> str:
    """The protonation key: canonical skeleton plus net charge.

    Canonicalising the assembled skeleton is what makes a proton hop between
    automorphic sites -- the two oxygens of a carboxylate -- invisible.
    """
    return f"{Chem.MolToSmiles(_skeleton_mol(template, counts))}|q{charge}"


def template_from_smiles(smiles: str) -> Chem.Mol:
    """Parse an explicit-H SMILES without discarding its hydrogens."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def protonation_key_from_mol(mol: Chem.Mol, charge: int) -> str:
    """The protonation key of a labelled structure, taken from its bond graph."""
    return _key_from_counts(mol, _template_counts(mol), charge)


def protonation_key_from_geometry(geom: Geometry, template: Chem.Mol, charge: int) -> str:
    """The protonation key of a geometry, taken from its coordinates.

    ``template`` must share the geometry's atom ordering -- guaranteed for
    approach 1, where the microstate's stored SMILES is the one the geometry was
    embedded from.  The ordering is checked rather than assumed.
    """
    assignment = assign_protons(geom)
    heavy_syms = [
        template.GetAtomWithIdx(a.GetIdx()).GetSymbol()
        for a in template.GetAtoms()
        if a.GetAtomicNum() != 1
    ]
    geom_syms = [geom.symbols[i] for i in geom.heavy_atom_indices]
    if heavy_syms != geom_syms:
        raise ValueError(
            f"template and geometry disagree on heavy-atom ordering: "
            f"{''.join(heavy_syms)} vs {''.join(geom_syms)}"
        )
    return _key_from_counts(template, assignment.counts, charge)


_SPECIFIED_BOND_STEREO = (
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)


def _heavy_positions(template: Chem.Mol) -> dict[int, int]:
    """Atom index -> position among the heavy atoms, which every protomer shares."""
    positions: dict[int, int] = {}
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() != 1:
            positions[atom.GetIdx()] = len(positions)
    return positions


def _specified_stereo(
    template: Chem.Mol,
) -> tuple[dict[tuple[int, int], Chem.BondStereo], dict[int, Chem.ChiralType]]:
    """The stereo a template actually specifies, in heavy-atom positions."""
    positions = _heavy_positions(template)
    bonds: dict[tuple[int, int], Chem.BondStereo] = {}
    for bond in template.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in positions and j in positions and bond.GetStereo() in _SPECIFIED_BOND_STEREO:
            key = (min(positions[i], positions[j]), max(positions[i], positions[j]))
            bonds[key] = bond.GetStereo()
    atoms = {
        positions[a.GetIdx()]: a.GetChiralTag()
        for a in template.GetAtoms()
        if a.GetAtomicNum() != 1 and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }
    return bonds, atoms


def _stereo_signature(mol: Chem.Mol, bonds: set[tuple[int, int]], atoms: set[int]) -> str:
    """Canonical SMILES reduced to the given stereo elements, everything else erased.

    Erasing the rest is required, not tidiness: `AssignStereochemistryFrom3D`
    annotates whatever the coordinates support, including bonds no label
    constrains -- a protonated carbonyl is a stereogenic double bond -- and those
    would match nothing.
    """
    rw = Chem.RWMol(mol)
    positions = _heavy_positions(rw)
    for atom in rw.GetAtoms():
        if positions.get(atom.GetIdx(), -1) not in atoms:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pair = (
            (min(positions[i], positions[j]), max(positions[i], positions[j]))
            if i in positions and j in positions
            else None
        )
        if pair not in bonds:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    return str(Chem.MolToSmiles(rw.GetMol()))


_MIRROR_TAG = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


def _mirror(mol: Chem.Mol) -> Chem.Mol:
    """The enantiomer: every tetrahedral tag inverted, bond stereo untouched.

    Reflection inverts configuration at every centre and leaves E/Z alone. The
    asymmetry is deliberate -- inverting bond stereo too would turn cis into
    trans and make an enantiomer-collapsed microstate accept the wrong
    diastereomer.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        tag = atom.GetChiralTag()
        if tag in _MIRROR_TAG:
            atom.SetChiralTag(_MIRROR_TAG[tag])
    return rw.GetMol()


def _reordered(geom: Geometry, order: list[int]) -> Geometry:
    """``geom`` with its atoms permuted into ``order``."""
    return Geometry(
        symbols=tuple(geom.symbols[i] for i in order),
        coords=geom.coords[order].copy(),
    )


def _order_for_correspondence(
    geom: Geometry,
    assignment: ProtonAssignment,
    slots: list[int | None],
    source_of_target: dict[int, int],
) -> list[int] | None:
    """The geometry's atom indices in a target's slot order, under one correspondence.

    ``None`` when the geometry cannot fill the slots -- it does not have the
    target's hydrogen distribution under this correspondence.
    """
    heavy = geom.heavy_atom_indices
    pending: dict[int, list[int]] = defaultdict(list)
    for h_idx, owner in zip(geom.hydrogen_indices, assignment.owner, strict=True):
        pending[heavy.index(owner)].append(h_idx)

    order: list[int] = []
    next_heavy = 0
    for slot in slots:
        source_pos = source_of_target[next_heavy if slot is None else slot]
        if slot is None:
            order.append(heavy[source_pos])
            next_heavy += 1
        elif pending[source_pos]:
            order.append(pending[source_pos].pop(0))
        else:
            return None
    if len(order) != len(geom.symbols) or any(pending.values()):
        return None
    return order


def _assert_identity_is_meaningful(geom: Geometry, target_template: Chem.Mol) -> None:
    """Guard the frame-order assumption the identity correspondence rests on.

    Preferring the identity leans on the geometry being in the template's atom
    order harder than a search does: a search recovers the right layout from a
    permuted geometry, whereas here a permutation that happened to preserve the
    hydrogen distribution would be taken for the identity. Nothing in the
    pipeline permutes atoms -- ``smiles_to_3d`` embeds in frame order and
    neither xtb, CREST, Psi4 nor PySCF reorders them -- so this is an assumption
    worth stating rather than a failure worth handling.

    The element sequence is already implied: the identity appears among the
    skeleton matches only if source atom *i* and target atom *i* share an
    element. Checking it anyway costs nothing and keeps the invariant true if
    the matcher's atom comparator is ever loosened.
    """
    heavy_syms = [a.GetSymbol() for a in target_template.GetAtoms() if a.GetAtomicNum() != 1]
    geom_syms = [geom.symbols[i] for i in geom.heavy_atom_indices]
    if heavy_syms != geom_syms:
        raise ValueError(
            f"identity correspondence assumed but template and geometry disagree on "
            f"heavy-atom ordering: {''.join(heavy_syms)} vs {''.join(geom_syms)}"
        )


def _layouts_for_target(
    geom: Geometry,
    assignment: ProtonAssignment,
    source_template: Chem.Mol,
    target_template: Chem.Mol,
    max_layouts: int = 64,
) -> list[Geometry]:
    """Every atom ordering of ``geom`` consistent with the target's skeleton.

    Usually one. More than one when the skeleton has an automorphism that
    preserves the hydrogen distribution, and then the choice is *not* free: it
    decides which of two symmetry-related sites a proton is taken to sit on, and
    a pseudo-asymmetric centre reads differently under each. Canonical ranking
    picks one arbitrarily and gets it wrong about half the time, so every
    consistent ordering is offered and the caller keeps whichever reproduces the
    candidate's own stereo.

    Matches are filtered to those preserving hydrogen counts. The matcher itself
    ignores them, which would map a protonated site onto an automorphic bare one
    -- the same trap that made a plain substructure match unusable for the
    ordering in the first place.

    The identity correspondence, when it survives that filter, is returned
    *alone*. The geometry carries the source template's atom order and nothing
    downstream permutes atoms, so identity is the correspondence the atoms
    actually have -- the automorphisms are relabellings with no physical content
    for this conformer. Surviving the filter under the identity is exactly the
    statement that the geometry's positionwise hydrogen vector equals the
    target's, i.e. that no proton changed owner; a search is needed only when it
    did, and that is where a genuine ambiguity can arise.

    This is not the ``same_microstate`` shortcut that was removed. That one
    keyed on the protonation key, which is invariant under automorphism and so
    could not tell whether the hydrogens sat on the positions the template
    actually uses; the positionwise vector can, and is what the filter compares.
    """
    slots = _heavy_slots(target_template)
    target_counts = _template_counts(target_template)

    # Fast path, and the only one taken unless a proton changed owner: equal
    # positionwise hydrogen vectors mean the two skeletons are the same graph,
    # so the identity is an isomorphism -- and, per the docstring, the true one.
    # Built directly rather than picked out of a search, both to skip the search
    # and so that correctness does not depend on the identity surviving
    # ``max_layouts``.
    if assignment.counts == target_counts:
        _assert_identity_is_meaningful(geom, target_template)
        order = _order_for_correspondence(
            geom, assignment, slots, {pos: pos for pos in range(len(target_counts))}
        )
        if order is not None:
            return [_reordered(geom, order)]

    source_skeleton = _skeleton_mol(source_template, assignment.counts)
    target_skeleton = _skeleton_mol(target_template, target_counts)
    matches = target_skeleton.GetSubstructMatches(
        source_skeleton, uniquify=False, useChirality=False, maxMatches=max_layouts
    )
    if len(matches) == max_layouts:
        log.warning(
            f"  skeleton automorphism search hit its cap of {max_layouts}; "
            f"some atom orderings were not considered"
        )

    layouts: list[Geometry] = []
    seen: set[tuple[int, ...]] = set()
    for match in matches:
        if any(
            source_skeleton.GetAtomWithIdx(src).GetNumExplicitHs()
            != target_skeleton.GetAtomWithIdx(tgt).GetNumExplicitHs()
            for src, tgt in enumerate(match)
        ):
            continue
        order = _order_for_correspondence(
            geom, assignment, slots, {tgt: src for src, tgt in enumerate(match)}
        )
        if order is None:
            continue
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        layouts.append(_reordered(geom, order))
    return layouts


def _stereo_from_coordinates(template: Chem.Mol, laid_out: Geometry) -> Chem.Mol:
    """The template's bond graph carrying the stereo those coordinates show."""
    mol = Chem.Mol(template)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = laid_out.coords[i]
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def match_to_candidate(
    geom: Geometry,
    assignment: ProtonAssignment,
    source_template: Chem.Mol,
    candidate_template: Chem.Mol,
    includes_enantiomer: bool,
) -> tuple[Geometry, bool] | None:
    """Is this geometry the candidate? If so, in the candidate's atom order.

    Returns ``(layout, verified)``, where ``verified`` says whether any stereo
    element was actually compared. A candidate constraining nothing cannot be
    contradicted, so it always matches -- but it matched on no evidence, and the
    caller must not let that outrank a candidate whose stereo was checked.

    Stereochemistry is verified, not assumed. The protonation key carries none,
    so a microstate's label is a claim about configuration that nothing checked
    until here -- and a change confined to stereochemistry leaves the key
    untouched, which is precisely where it hides.

    Each candidate is tested on *its own* bond orders. A migration invalidates
    the bond orders of the microstate the conformer came from, never those of
    the ones it might have become, so no bond is ever perceived from
    coordinates: the question is only which hypothesis the geometry agrees with.

    Only elements the candidate constrains *and* the coordinates determine are
    compared. A candidate that specifies nothing cannot be contradicted, and a
    bond that is single under this candidate's orders is free to rotate and
    cannot testify.

    When the microstate stands for a collapsed enantiomeric pair the mirror image
    is accepted too, because that is what the microstate means.

    The conformer's own microstate is not special-cased, tempting as it is: the
    geometry is in that microstate's atom order, but its protonation key matched
    only up to automorphism, so the hydrogens can sit on symmetry-related
    positions the template does not use. Skipping the layout step there let a
    geometry verify against a template whose hydrogen distribution it did not
    actually have -- the stereocentres elsewhere in the molecule matched, and
    nothing checked the rest.
    """
    specified_bonds, specified_atoms = _specified_stereo(candidate_template)
    candidates_layouts = _layouts_for_target(geom, assignment, source_template, candidate_template)
    if not specified_bonds and not specified_atoms:
        return (candidates_layouts[0], False) if candidates_layouts else None

    for laid_out in candidates_layouts:
        observed = _stereo_from_coordinates(candidate_template, laid_out)
        observed_bonds, observed_atoms = _specified_stereo(observed)
        bonds = set(specified_bonds) & set(observed_bonds)
        atoms = set(specified_atoms) & set(observed_atoms)
        if not bonds and not atoms:
            return (laid_out, False)
        seen = _stereo_signature(observed, bonds, atoms)
        if seen == _stereo_signature(candidate_template, bonds, atoms):
            return (laid_out, True)
        if includes_enantiomer and seen == _stereo_signature(
            _mirror(candidate_template), bonds, atoms
        ):
            return (laid_out, True)
    return None


@dataclass
class MigrationReport:
    """What `repair_migrated_conformers` did to one charge state."""

    checked: int = 0
    moved: int = 0
    fragmented: int = 0
    detached: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    stereo_resolved: int = 0
    stereo_unmatched: int = 0
    unresolved_tie: int = 0
    created: int = 0
    tightest_margin: float = float("inf")

    @property
    def touched(self) -> int:
        """Conformers this moved, opened a microstate for, or refused to place.

        ``ambiguous`` belongs here: it excludes a conformer outright. Leaving it
        out made an ambiguous-only charge state print no summary at all, even as
        it discarded every conformer of a microstate -- which is what happened to
        malonic acid at q=-1, where two exclusions were visible only as
        per-conformer warnings.

        The counters that *keep* a conformer -- ``unresolved_tie``,
        ``stereo_unmatched`` -- are deliberately not here. Nothing was touched,
        so the caller's "re-filed" line would be wrong; `summary` reports them
        separately.
        """
        return (
            self.moved
            + self.fragmented
            + self.detached
            + self.unmatched
            + self.ambiguous
            + self.created
        )

    def summary(self) -> str | None:
        """One line naming every outcome that occurred, or ``None`` if none did.

        Built here rather than at each call site: the three callers had drifted
        into formatting different subsets of the same report, so an outcome was
        reported or not depending on which stage produced it.
        """
        parts: list[str] = []
        if self.moved:
            parts.append(f"{self.moved} conformer(s) re-filed after a proton moved")
        if self.created:
            parts.append(f"{self.created} new microstate(s) opened")
        if self.fragmented:
            parts.append(f"{self.fragmented} fragmented")
        if self.detached:
            parts.append(f"{self.detached} with a detached H")
        if self.unmatched:
            parts.append(f"{self.unmatched} matching no microstate")
        if self.ambiguous:
            parts.append(f"{self.ambiguous} excluded as ambiguous")
        if self.unresolved_tie:
            parts.append(f"{self.unresolved_tie} left in place, indistinguishable")
        if self.stereo_unmatched:
            parts.append(f"{self.stereo_unmatched} with an unmatched configuration")
        if not parts:
            return None
        if self.tightest_margin < 0.2:
            parts.append(f"tightest H-ownership margin {self.tightest_margin:.2f} A")
        return "; ".join(parts)


def _heavy_slots(template: Chem.Mol) -> list[int | None]:
    """Atom-slot layout of an explicit-H mol, one entry per atom in template order.

    ``None`` marks the next heavy atom; an integer marks a hydrogen bonded to
    that heavy atom's position among the heavy atoms.
    """
    position: dict[int, int] = {}
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() != 1:
            position[atom.GetIdx()] = len(position)

    slots: list[int | None] = []
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() != 1:
            slots.append(None)
            continue
        owners = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(owners) != 1:
            raise ValueError(f"hydrogen {atom.GetIdx()} does not have exactly one heavy neighbour")
        slots.append(position[owners[0]])
    return slots


def _frame_ranks(template: Chem.Mol) -> tuple[int, ...]:
    """Canonical rank of each heavy atom within the molecule's bare frame.

    The frame is the same graph for every protomer, so two templates whose rank
    lists agree position-for-position have their heavy atoms in the same order.
    """
    heavy_count = sum(1 for a in template.GetAtoms() if a.GetAtomicNum() != 1)
    frame = _skeleton_mol(template, (0,) * heavy_count)
    return tuple(Chem.CanonicalRankAtoms(frame))


def _regroup_hydrogens(
    geom: Geometry,
    assignment: ProtonAssignment,
    source_template: Chem.Mol,
    target_template: Chem.Mol,
) -> Geometry:
    """Move a migrated geometry's hydrogens into its destination's slots.

    `rdkit_utils.frame_atom_order` already guarantees that heavy atom *i* is the
    same atom in every protomer, so no atom correspondence has to be discovered.
    What still has to be resolved is *which* automorphic site holds the proton.
    A key is a canonical SMILES, so it is deliberately blind to the difference
    between a proton on one carboxylate oxygen and the other -- that is what
    keeps maleate from splitting in two. The consequence is that a geometry can
    match its destination's key while distributing its hydrogens over a
    symmetry-equivalent set of positions, and slotting them in by position would
    then put a hydrogen where the destination has none.

    Ranking the *H-pinned* skeletons resolves it: both sides are the same
    molecule with the same hydrogen distribution, so equal canonical ranks are
    corresponding atoms, automorphism included. Ranking is used rather than a
    substructure match because the matcher ignores hydrogen counts and would
    map a protonated site onto an automorphic bare one.
    """
    source_skeleton = _skeleton_mol(source_template, assignment.counts)
    target_skeleton = _skeleton_mol(target_template, _template_counts(target_template))
    source_rank = list(Chem.CanonicalRankAtoms(source_skeleton))
    target_rank = list(Chem.CanonicalRankAtoms(target_skeleton))
    if sorted(source_rank) != sorted(target_rank):
        raise ValueError("source and target skeletons do not correspond")
    source_by_rank = {rank: pos for pos, rank in enumerate(source_rank)}
    source_of_target = {pos: source_by_rank[rank] for pos, rank in enumerate(target_rank)}

    heavy = geom.heavy_atom_indices
    pending: dict[int, list[int]] = defaultdict(list)
    for h_idx, owner in zip(geom.hydrogen_indices, assignment.owner, strict=True):
        pending[heavy.index(owner)].append(h_idx)

    order: list[int] = []
    next_target_heavy = 0
    for slot in _heavy_slots(target_template):
        source_pos = source_of_target[next_target_heavy if slot is None else slot]
        if slot is None:
            order.append(heavy[source_pos])
            next_target_heavy += 1
        else:
            if not pending[source_pos]:
                raise ValueError("geometry does not have the target's hydrogen distribution")
            order.append(pending[source_pos].pop(0))
    if len(order) != len(geom.symbols) or any(pending.values()):
        raise ValueError("geometry does not have the target's hydrogen distribution")
    return Geometry(
        symbols=tuple(geom.symbols[i] for i in order),
        coords=geom.coords[order].copy(),
    )


def _template_counts(template: Chem.Mol) -> tuple[int, ...]:
    return tuple(
        a.GetTotalNumHs(includeNeighbors=True)
        for a in template.GetAtoms()
        if a.GetAtomicNum() != 1
    )


def _ordering_shape_from_geometry(geom: Geometry) -> list[int | None]:
    """The same layout, read off a geometry rather than a bond graph."""
    assignment = assign_protons(geom)
    heavy_position = {idx: pos for pos, idx in enumerate(geom.heavy_atom_indices)}
    owner_of_h = dict(zip(geom.hydrogen_indices, assignment.owner, strict=True))
    shape: list[int | None] = []
    for idx, symbol in enumerate(geom.symbols):
        shape.append(heavy_position[owner_of_h[idx]] if symbol == "H" else None)
    return shape


def _reorder_to_shape(geom: Geometry, shape: list[int | None]) -> Geometry:
    """Permute a geometry's atoms into the given layout.

    Heavy atoms keep their relative order -- every microstate of one molecule
    shares it -- so only the hydrogens move, into the slots the destination
    reserves for the heavy atom each now belongs to.  Without this a re-filed
    conformer arrives with its old microstate's hydrogen interleaving and trips
    the atom-ordering guard in `ensemble.deduplicate_conformers`.
    """
    assignment = assign_protons(geom)
    heavy = geom.heavy_atom_indices
    pending: dict[int, list[int]] = defaultdict(list)
    for h_idx, owner in zip(geom.hydrogen_indices, assignment.owner, strict=True):
        pending[heavy.index(owner)].append(h_idx)

    order: list[int] = []
    next_heavy = 0
    for slot in shape:
        if slot is None:
            order.append(heavy[next_heavy])
            next_heavy += 1
        else:
            if not pending[slot]:
                raise ValueError("geometry does not have the hydrogen distribution of the target")
            order.append(pending[slot].pop(0))
    if len(order) != len(geom.symbols) or any(pending.values()):
        raise ValueError("geometry does not have the hydrogen distribution of the target")
    return Geometry(
        symbols=tuple(geom.symbols[i] for i in order),
        coords=geom.coords[order].copy(),
    )


def _exclude(
    ms: Microstate,
    conf: Conformer,
    stage: ExclusionStage,
    reason: ExclusionReason,
    detail: str,
) -> None:
    ms.excluded_conformers.append(
        ExcludedConformer(
            geometry=conf.geometry,
            stage=stage,
            reason=reason,
            detail=detail,
            multiplicity=conf.multiplicity,
            electronic_energy=conf.electronic_energy,
            solvation_energy=conf.solvation_energy,
            rrho_correction=conf.rrho_correction,
            refinement_converged=conf.refinement_converged,
        )
    )


def _fragment_detail(components: tuple[tuple[int, ...], ...]) -> str:
    sizes = ", ".join(str(len(c)) for c in components)
    return f"heavy-atom framework in {len(components)} pieces (sizes {sizes})"


def repair_migrated_conformers(cs: ChargeState, stage: ExclusionStage) -> MigrationReport:
    """Re-file conformers whose proton moved during minimisation.

    Each conformer's protonation state is read back off its geometry and
    compared against the microstate it is filed under.  Where they differ the
    conformer is moved to the microstate it actually became.

    A conformer whose heavy-atom framework came apart is excluded first, in both
    approaches. Neither identity can see it: approach 1 takes its framework from
    the template and only the hydrogen counts from the coordinates, and approach
    2 computes the framework but does not hash it, so in both a fragment reads
    as the intact species.

    The two sampling approaches diverge on what to do when no existing
    microstate matches, because they disagree about where microstates come from.
    In approach 1 the set is *prescribed*: the enumerator decided which species
    are modelled, every microstate carries a known SMILES, and a geometry
    outside that set is an anomaly we cannot label without perceiving bond
    orders from coordinates -- so it is excluded, not dropped, and the run
    records what it could not place.  In approach 2 the set is *discovered*:
    microstates are defined by the hydrogen distribution read off the geometry
    in the first place, so an unseen distribution is not an anomaly at all and a
    new microstate is created for it.

    Runs before deduplication, so a conformer that lands beside an identical
    structure in its new microstate collapses against it instead of
    double-counting.
    """
    if not cs.microstates:
        return MigrationReport()
    labelled = [ms.smiles is not None for ms in cs.microstates]
    if all(labelled):
        return _repair_labelled(cs, stage)
    if not any(labelled):
        return _repair_unlabelled(cs, stage)
    log.warning(
        f"  q={cs.charge}: microstates disagree on whether they carry a SMILES label; "
        f"skipping the proton-migration check"
    )
    return MigrationReport()


def _repair_labelled(cs: ChargeState, stage: ExclusionStage) -> MigrationReport:
    """Approach 1: microstates carry a SMILES, so the target set is fixed."""
    report = MigrationReport()

    label_key: dict[int, str] = {}
    by_key: dict[str, list[int]] = defaultdict(list)
    templates: dict[int, Chem.Mol] = {}
    for pos, ms in enumerate(cs.microstates):
        assert ms.smiles is not None
        template = template_from_smiles(ms.smiles)
        templates[pos] = template
        key = protonation_key_from_mol(template, cs.charge)
        label_key[pos] = key
        by_key[key].append(pos)

    keep: dict[int, list[Conformer]] = {pos: [] for pos in range(len(cs.microstates))}
    moves: list[tuple[int, int, Conformer, Geometry]] = []

    for pos, ms in enumerate(cs.microstates):
        for conf in ms.conformers:
            report.checked += 1
            assignment = assign_protons(conf.geometry)
            report.tightest_margin = min(report.tightest_margin, assignment.min_margin)

            # Connectivity first, because the protonation key cannot see it: the
            # framework below comes from the *template*, and only the hydrogen
            # counts from the coordinates, so a structure whose C-O broke yields
            # a key identical to the intact species and is filed as though
            # nothing happened. Five of the eight `no_matching_microstate`
            # exclusions in the first end-to-end batch were fragmentations that
            # only failed to match by luck.
            components = heavy_components(conf.geometry)
            if len(components) > 1:
                report.fragmented += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of {ms.tautomer_id[:32]} came apart during "
                    f"minimisation ({_fragment_detail(components)}); excluding it"
                )
                _exclude(ms, conf, stage, "fragmented", _fragment_detail(components))
                continue

            if not assignment.is_intact:
                report.detached += 1
                log.warning(
                    f"  q={cs.charge}: hydrogen(s) {list(assignment.detached)} detached from "
                    f"{ms.tautomer_id[:32]}; excluding conformer"
                )
                _exclude(
                    ms,
                    conf,
                    stage,
                    "proton_detached",
                    f"H index/indices {list(assignment.detached)} further than "
                    f"{DETACHED_DISTANCE} A from every heavy atom",
                )
                continue

            geom_key = _key_from_counts(templates[pos], assignment.counts, cs.charge)
            candidates = by_key.get(geom_key, [])
            if not candidates:
                report.unmatched += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of {ms.tautomer_id[:32]} minimised to a "
                    f"species no microstate describes; excluding it"
                )
                _exclude(
                    ms,
                    conf,
                    stage,
                    "no_matching_microstate",
                    f"geometry key {geom_key} not among the microstates at this charge",
                )
                continue

            # Stereochemistry is verified here rather than assumed, including for
            # a conformer that appears to match its own label -- a change
            # confined to stereo leaves the protonation key untouched, so that
            # branch is exactly where it hides.
            viable: list[tuple[int, Geometry, bool]] = []
            for candidate in candidates:
                if candidate != pos and _frame_ranks(templates[pos]) != _frame_ranks(
                    templates[candidate]
                ):
                    # smiles_to_3d establishes this; a violation means a geometry
                    # was built some other way, and pairing up heavy atoms that
                    # may not correspond would be worse than declining.
                    log.error(
                        f"  q={cs.charge}: {ms.tautomer_id[:32]} and "
                        f"{cs.microstates[candidate].tautomer_id[:32]} disagree on heavy-atom "
                        f"order; not considering that destination"
                    )
                    continue
                matched = match_to_candidate(
                    conf.geometry,
                    assignment,
                    templates[pos],
                    templates[candidate],
                    cs.microstates[candidate].includes_enantiomer,
                )
                if matched is not None:
                    viable.append((candidate, matched[0], matched[1]))

            # A candidate constraining no stereo matches on no evidence, and must
            # not outrank one whose configuration was actually checked. The two
            # kinds of tie then differ: several candidates matching *vacuously*
            # means nothing indicates a change, so staying put is right and
            # unremarkable; several matching on real evidence is a genuine
            # conflict and has to stay visible rather than be resolved by
            # preferring where the conformer happened to start.
            verified = [entry for entry in viable if entry[2]]
            if len(verified) == 1:
                viable = verified
            elif not verified and len(viable) > 1 and any(entry[0] == pos for entry in viable):
                # Nothing to discriminate: no candidate constrains any stereo, so
                # all of them match trivially. Unremarkable, and staying put is
                # right.
                viable = [entry for entry in viable if entry[0] == pos]
            elif len(verified) > 1 and any(entry[0] == pos for entry in verified):
                # Several candidates are satisfiable and one of them is where the
                # conformer already sits. That is a failure to discriminate, not
                # evidence of a change, so it stays -- and it costs nothing to
                # leave it: the charge-state free energy sums flatly over
                # microstates, so which one holds a conformer moves the answer
                # only through `includes_enantiomer`.
                #
                # The usual cause is a skeleton automorphism that is not a
                # symmetry of the stereochemistry -- two ends of a molecule that
                # only bond order distinguishes -- which admits both atom
                # correspondences. Where the candidates also differ in hydrogen
                # distribution the layout step rejects the wrong one and this
                # never arises; where they share one, as an E/Z pair does,
                # nothing here can separate them.
                report.unresolved_tie += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of {ms.tautomer_id[:32]} is satisfiable as "
                    f"{len(viable)} microstates and cannot be told apart from where it already "
                    f"is; keeping it there"
                )
                keep[pos].append(conf)
                continue

            if len(viable) > 1:
                report.ambiguous += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of {ms.tautomer_id[:32]} became one of "
                    f"{len(viable)} microstates, none of them the one it was filed under, and "
                    f"cannot be told apart; excluding it"
                )
                _exclude(
                    ms,
                    conf,
                    stage,
                    "ambiguous_microstate",
                    f"geometry key {geom_key} matches "
                    f"{', '.join(cs.microstates[c].tautomer_id for c, _, _ in viable)}",
                )
                continue

            if not viable:
                # The protonation is right and no microstate has this
                # configuration. Deliberately kept rather than excluded: if
                # sampling inverted the centre it is not configurationally
                # stable, and the label over-claimed. See "Known limitations"
                # in docs/protomer-identity.md.
                report.stereo_unmatched += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of {ms.tautomer_id[:32]} has the right "
                    f"protonation but a configuration no microstate at this charge describes; "
                    f"keeping it where it is -- the specified stereochemistry may not be stable"
                )
                keep[pos].append(conf)
                continue

            target, laid_out, _ = viable[0]
            if target == pos:
                keep[pos].append(conf)
                continue

            if len(candidates) > 1:
                report.stereo_resolved += 1
            report.moved += 1
            log.info(
                f"  q={cs.charge}: re-filing a conformer of {ms.tautomer_id[:32]} under "
                f"{cs.microstates[target].tautomer_id[:32]}"
            )
            moves.append((target, pos, conf, laid_out))

    for pos, ms in enumerate(cs.microstates):
        ms.conformers = keep[pos]
    for target, _source, conf, laid_out in moves:
        conf.geometry = laid_out
        cs.microstates[target].conformers.append(conf)

    return report


def _repair_unlabelled(cs: ChargeState, stage: ExclusionStage) -> MigrationReport:
    """Approach 2: microstates are defined by the hydrogen distribution itself.

    An unseen distribution is a new microstate, not an error -- the same rule
    `tautomer_dedup.deduplicate_tautomers` applies when the charge state is
    first built.  The fingerprint is positional, so it only compares across
    microstates that share a heavy-atom ordering; that is checked rather than
    assumed, and the whole charge state is skipped if it does not hold.

    Structures that came apart are excluded in a pre-pass, before that check
    runs, so a fragment cannot disable it for the conformers that are fine.
    """
    report = MigrationReport()

    # Fragmented structures go first, before anything compares frameworks. The
    # agreement check below is about heavy-atom *ordering* -- the fingerprint is
    # positional, so it is meaningless across structures that order their heavy
    # atoms differently -- and a structure that came apart fails it for an
    # unrelated reason. Since the response to a disagreement is to skip the
    # whole charge state, one fragment would otherwise disable the migration
    # check for every conformer at that charge, including the intact ones; and
    # because the reference is whichever conformer is reached first, a fragment
    # arriving first would make the healthy majority look like the disagreement.
    for ms in cs.microstates:
        intact: list[Conformer] = []
        for conf in ms.conformers:
            components = heavy_components(conf.geometry)
            if len(components) > 1:
                report.checked += 1
                report.fragmented += 1
                log.warning(
                    f"  q={cs.charge}: a conformer of tautomer {ms.tautomer_id[:8]} came apart "
                    f"during minimisation ({_fragment_detail(components)}); excluding it"
                )
                _exclude(ms, conf, stage, "fragmented", _fragment_detail(components))
                continue
            intact.append(conf)
        ms.conformers = intact

    reference: Geometry | None = None
    for ms in cs.microstates:
        for conf in ms.conformers:
            if reference is None:
                reference = conf.geometry
            elif not heavy_frameworks_agree(reference, conf.geometry):
                log.warning(
                    f"  q={cs.charge}: structures do not share a heavy-atom order or "
                    f"framework; skipping the proton-migration check, which is indexed "
                    f"by heavy-atom position throughout"
                )
                return report

    by_fingerprint = {ms.tautomer_id: pos for pos, ms in enumerate(cs.microstates)}
    keep: dict[int, list[Conformer]] = {pos: [] for pos in range(len(cs.microstates))}
    moves: list[tuple[int, Conformer]] = []
    created: dict[str, Microstate] = {}

    for pos, ms in enumerate(cs.microstates):
        for conf in ms.conformers:
            report.checked += 1
            assignment = assign_protons(conf.geometry)
            report.tightest_margin = min(report.tightest_margin, assignment.min_margin)

            if not assignment.is_intact:
                report.detached += 1
                log.warning(
                    f"  q={cs.charge}: hydrogen(s) {list(assignment.detached)} detached from "
                    f"tautomer {ms.tautomer_id[:8]}; excluding conformer"
                )
                _exclude(
                    ms,
                    conf,
                    stage,
                    "proton_detached",
                    f"H index/indices {list(assignment.detached)} further than "
                    f"{DETACHED_DISTANCE} A from every heavy atom",
                )
                continue

            fingerprint = geometric_fingerprint(conf.geometry)
            if fingerprint == ms.tautomer_id:
                keep[pos].append(conf)
            elif fingerprint in by_fingerprint:
                report.moved += 1
                log.info(
                    f"  q={cs.charge}: a proton moved during minimisation; re-filing a "
                    f"conformer of tautomer {ms.tautomer_id[:8]} under {fingerprint[:8]}"
                )
                moves.append((by_fingerprint[fingerprint], conf))
            else:
                if fingerprint not in created:
                    report.created += 1
                    log.info(
                        f"  q={cs.charge}: a proton moved to a position no sampled structure "
                        f"had; opening tautomer {fingerprint[:8]} for it"
                    )
                    created[fingerprint] = Microstate(
                        tautomer_id=fingerprint,
                        conformers=[],
                        includes_enantiomer=ms.includes_enantiomer,
                    )
                created[fingerprint].conformers.append(conf)

    for pos, ms in enumerate(cs.microstates):
        ms.conformers = keep[pos]
    for target, conf in moves:
        reference = _reference_geometry(cs.microstates[target], keep[target])
        if reference is not None:
            conf.geometry = _reorder_to_shape(
                conf.geometry, _ordering_shape_from_geometry(reference)
            )
        cs.microstates[target].conformers.append(conf)
    for fp in sorted(created):
        opened = created[fp]
        shape = _ordering_shape_from_geometry(opened.conformers[0].geometry)
        for conf in opened.conformers[1:]:
            conf.geometry = _reorder_to_shape(conf.geometry, shape)
        cs.microstates.append(opened)

    return report


def _reference_geometry(ms: Microstate, kept: list[Conformer]) -> Geometry | None:
    """The atom ordering a microstate already uses, if it still has a conformer."""
    if kept:
        return kept[0].geometry
    if ms.excluded_conformers:
        return ms.excluded_conformers[0].geometry
    return None
