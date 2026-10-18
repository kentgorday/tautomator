"""Broad SMARTS-based protonation/deprotonation enumeration with BFS.

Design principle: keep SMARTS simple and general (match any heteroatom with
the right H-count/charge), not substructure-specific. Generate more variants
rather than fewer — downstream QM energetics filter unrealistic states.

The table is closed under reversal: every deprotonation has a protonation that
undoes it, so a species can be walked back to where it came from. Note that P is
covered only as phosphine <-> phosphide; there is no phosphonium path in either
direction, which is a coverage gap rather than an asymmetry.

Both sides of every rule pin the hydrogen count *and* the formal charge. An
RDKit product template inherits any property it does not state, so a product
written ``[NH2:1]`` keeps the reactant's charge: ``[NH3+:1]>>[NH2:1]`` yields
an [NH2+] whose net charge never reaches the target, and the rule silently
never fires. Every rule that neutralises an ion has that shape, so leaving the
charge implicit disables exactly the half of the table needed to walk a charged
input back toward neutral.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from tautomator.rdkit_utils import (
    canonical_smiles,
    deduplicate_protomers,
    get_formal_charge,
)

# Deprotonation reactions: remove one H from a heteroatom, decrease formal charge.
# Each pattern is intentionally broad.
_DEPROTONATION_SMARTS: list[str] = [
    # Neutral heteroatom -> anion
    "[N;H1;+0:1]>>[N;H0;-1:1]",
    "[N;H2;+0:1]>>[N;H1;-1:1]",
    "[N;H3;+0:1]>>[N;H2;-1:1]",
    "[O;H1;+0:1]>>[O;H0;-1:1]",
    "[O;H2;+0:1]>>[O;H1;-1:1]",
    "[S;H1;+0:1]>>[S;H0;-1:1]",
    "[P;H1;+0:1]>>[P;H0;-1:1]",
    # Cation -> neutral
    "[N;H4;+1:1]>>[N;H3;+0:1]",
    "[N;H3;+1:1]>>[N;H2;+0:1]",
    "[N;H2;+1:1]>>[N;H1;+0:1]",
    "[N;H1;+1:1]>>[N;H0;+0:1]",
    "[O;H2;+1:1]>>[O;H1;+0:1]",
    "[O;H1;+1:1]>>[O;H0;+0:1]",
    "[S;H2;+1:1]>>[S;H1;+0:1]",
    "[S;H1;+1:1]>>[S;H0;+0:1]",
    # Aromatic N
    "[n;H1;+0:1]>>[n;H0;-1:1]",
    "[n;H1;+1:1]>>[n;H0;+0:1]",
]

# Protonation reactions: add one H to a heteroatom, increase formal charge.
_PROTONATION_SMARTS: list[str] = [
    # Neutral heteroatom -> cation
    "[N;H2;+0:1]>>[N;H3;+1:1]",
    "[N;H1;+0:1]>>[N;H2;+1:1]",
    "[N;H0;+0;X3:1]>>[N;H1;+1:1]",
    "[O;H1;+0:1]>>[O;H2;+1:1]",
    "[O;H0;+0:1]>>[O;H1;+1:1]",
    "[S;H1;+0:1]>>[S;H2;+1:1]",
    "[S;H0;+0:1]>>[S;H1;+1:1]",
    # Anion -> neutral
    "[N;H0;-1:1]>>[N;H1;+0:1]",
    "[N;H1;-1:1]>>[N;H2;+0:1]",
    "[N;H2;-1:1]>>[N;H3;+0:1]",
    "[O;H0;-1:1]>>[O;H1;+0:1]",
    "[O;H1;-1:1]>>[O;H2;+0:1]",
    "[S;H0;-1:1]>>[S;H1;+0:1]",
    "[P;H0;-1:1]>>[P;H1;+0:1]",
    # Aromatic N
    "[n;H0;+0:1]>>[n;H1;+1:1]",
    "[n;H0;-1:1]>>[n;H1;+0:1]",
]


def _compile_reactions(smarts_list: list[str]) -> list[AllChem.ChemicalReaction]:
    reactions: list[AllChem.ChemicalReaction] = []
    for s in smarts_list:
        rxn = AllChem.ReactionFromSmarts(s)
        if rxn is None:
            raise RuntimeError(f"Failed to compile reaction SMARTS: {s}")
        reactions.append(rxn)
    return reactions


_DEPROT_RXNS: list[AllChem.ChemicalReaction] | None = None
_PROT_RXNS: list[AllChem.ChemicalReaction] | None = None


def _get_deprot_rxns() -> list[AllChem.ChemicalReaction]:
    global _DEPROT_RXNS
    if _DEPROT_RXNS is None:
        _DEPROT_RXNS = _compile_reactions(_DEPROTONATION_SMARTS)
    return _DEPROT_RXNS


def _get_prot_rxns() -> list[AllChem.ChemicalReaction]:
    global _PROT_RXNS
    if _PROT_RXNS is None:
        _PROT_RXNS = _compile_reactions(_PROTONATION_SMARTS)
    return _PROT_RXNS


def deprotonate_all_sites(smiles: str) -> list[str]:
    """Remove one proton from every possible heteroatom site.

    Returns a deduplicated list of canonical SMILES for all single-deprotonation
    products. Each product has formal charge one unit lower than the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    target_charge = get_formal_charge(smiles) - 1
    return _apply_reactions(mol, _get_deprot_rxns(), target_charge)


def protonate_all_sites(smiles: str) -> list[str]:
    """Add one proton to every possible heteroatom site.

    Returns a deduplicated list of canonical SMILES for all single-protonation
    products. Each product has formal charge one unit higher than the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    target_charge = get_formal_charge(smiles) + 1
    return _apply_reactions(mol, _get_prot_rxns(), target_charge)


def _apply_reactions(
    mol: Chem.Mol,
    reactions: list[AllChem.ChemicalReaction],
    target_charge: int,
) -> list[str]:
    """Apply all reactions to mol, return deduplicated products at target charge."""
    seen: set[str] = set()
    results: list[str] = []
    for rxn in reactions:
        products = rxn.RunReactants((mol,))
        for product_tuple in products:
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    charge = Chem.GetFormalCharge(product)
                    if charge != target_charge:
                        # Also the backstop for a template that failed to adjust
                        # the charge: such a product keeps the reactant's charge
                        # and is rejected here before anything downstream sees it.
                        continue
                    can = Chem.MolToSmiles(product)
                    if can is not None and can not in seen:
                        seen.add(can)
                        results.append(can)
                except Exception:
                    # Skip products that fail sanitization
                    continue
    return results


def charge_separated_variants(smiles: str) -> list[str]:
    """Species at this molecule's *own* net charge, reached by moving one proton.

    RDKit's ``TautomerEnumerator`` moves protons but never separates charge, and
    :func:`enumerate_charge_state` walks *between* charge states, so neither
    produces a protomer whose charge is internally separated. For ``NCC(=O)O``
    that leaves the glycine zwitterion out of the q=0 microstate set entirely,
    although it is the dominant neutral form in water.

    This closes that gap by protonating every site and then deprotonating every
    site of each product, keeping whatever comes back at the original charge.
    The input itself is excluded; it is already a microstate.

    The output is *not* filtered for plausibility, and most of it is not
    populated at any pH: a proton is as happily moved from an amine to a
    carbonyl oxygen, giving an amide anion beside a protonated carbonyl, as from
    a carboxylic acid to an amine, giving the zwitterion. Filtering would need
    the two sites' relative acidities, and that means a hand-maintained table.
    Energetic screening does not substitute for one -- measured on
    ``NC(CC(F)(F)F)C(=O)O``, two absurd protomers optimise to within 0.5 and
    2.2 kcal/mol of the genuine zwitterion's 0.7, because they are not minima at
    all: they relax straight back to the neutral form, and what gets compared is
    that form's energy under a false label.

    Relying on that relaxation is what makes leaving them in tolerable.
    ``repair_migrated_conformers`` reads the protonation key off the geometry, so
    a relaxed protomer is either re-filed onto the species it became or excluded
    as ``no_matching_microstate``; it cannot reach a result wearing the wrong
    label. The cost is the sampling spent finding that out -- across the pKa
    set this takes microstates at the reference charge from 81 to 171.
    """
    charge = get_formal_charge(smiles)
    seen: set[str] = set()
    results: list[str] = []
    for protonated in protonate_all_sites(smiles):
        for back in deprotonate_all_sites(protonated):
            if get_formal_charge(back) != charge:
                continue
            can = canonical_smiles(back)
            if can not in seen:
                seen.add(can)
                results.append(can)
    own = canonical_smiles(smiles)
    return [s for s in results if s != own]


def enumerate_charge_state(smiles: str, target_charge: int) -> list[str]:
    """BFS to enumerate the distinct species at the target charge.

    Starting from the input SMILES, iteratively applies single protonation or
    deprotonation steps until the target charge is reached.

    Returns one canonical SMILES per *species*, not per Lewis structure: two
    results that differ only in which atom was chosen to carry a delocalised
    charge are collapsed by :func:`~tautomator.rdkit_utils.deduplicate_protomers`,
    since they describe one microstate. The representative is chosen
    deterministically and does not depend on input order.

    Returns an empty list if the target charge is unreachable from this input,
    for example asking for q=-2 of a molecule with only one ionizable site.
    """
    current_charge = get_formal_charge(smiles)
    if current_charge == target_charge:
        return [canonical_smiles(smiles)]

    step_fn = deprotonate_all_sites if target_charge < current_charge else protonate_all_sites

    n_steps = abs(target_charge - current_charge)
    current_level: set[str] = {canonical_smiles(smiles)}

    for _ in range(n_steps):
        next_level: set[str] = set()
        for smi in current_level:
            products = step_fn(smi)
            next_level.update(products)
        if not next_level:
            # No further (de)protonation sites — target charge unreachable.
            # Return [] rather than the partial-walk SMILES, which would be
            # at the wrong charge and silently feed bogus species to DFT.
            return []
        current_level = next_level

    # Collapse species that differ only in which atom the enumerator chose to
    # carry a delocalised charge; they are one microstate.
    return deduplicate_protomers(sorted(current_level))
