#!/usr/bin/env python
"""Select the MNSol solutes for fitting the nonelectrostatic correction.

The fit is on absolute solvation free energies (gas to solvent), one solvent at
a time: water, 1-octanol and n-hexadecane.  Octanol-water transfer free energies
(-2.303 RT logP) exist for some solutes with no absolute value in either
solvent; those are kept out of the fit as a logP test.  MNSol's octanol values
come mostly from logP, so they belong to water-saturated octanol and are run in
tautomator's "woctanol" (docs/nonelectrostatic.md).

A solute is kept if it has at most 10 heavy atoms and at most 2 rotatable bonds,
the pKa set's limits, so that its conformer ensemble stays small; and if
the pipeline accepts it (``rdkit_utils.validate_input_smiles``): closed-shell, a
single component, and at least one heavy atom.  That drops the ion-water
clusters, the water dimer and H2.  Ionizable sites are not limited, since
MNSol fixes each solute's charge state, and carbon-free solutes stay: they carry
the strongest surface fields.

The output follows the pKa set's CSV (``unique_ID`` and ``SMILES``), with the
molecular ``charge`` to run at: each solute is one species, so a run's charge
range is ``charge`` to ``charge``.  SMILES come from the database geometries
(``tautomator.mnsol``) and carry their stereochemistry.  Solutes the database
lists separately but that share a SMILES are merged into one row (the E and Z
conformers of N-methylacetamide).

MNSol is distributed under its own license; neither it nor this script's
output is tracked in the repository (see .gitignore).

Usage:
    pixi run python scripts/select_mnsol_set.py \
        --database ~/MNSolDatabase_v2012 \
        --output mnsol_set.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from rdkit import RDLogger

from tautomator import mnsol
from tautomator.rdkit_utils import get_formal_charge, validate_input_smiles

# Output column -> (database solvent label, database type).  "abs" is gas to
# solvent; "rel" is water to solvent.
REFERENCE_COLUMNS: dict[str, tuple[str, str]] = {
    "dG_water": ("water", "abs"),
    "dG_octanol": ("octanol", "abs"),
    "dG_hexadecane": ("hexadecane", "abs"),
    "dG_octanol_water": ("octanol-water", "rel"),
}

# Experimental values (kcal/mol) of merged solutes may differ by this much.
MERGE_TOLERANCE = 0.05


def load_solutes(database: Path) -> pd.DataFrame:
    """One row per solute with a value in a fitted solvent: handle, name, formula,
    charge, and one column per entry of REFERENCE_COLUMNS."""
    data = pd.read_csv(database / mnsol.TABLE, sep="\t", encoding="latin-1")
    solutes = data.groupby("FileHandle", sort=False).first()[["SoluteName", "Formula", "Charge"]]
    for column, (solvent, kind) in REFERENCE_COLUMNS.items():
        rows = data[(data.Solvent == solvent) & (data.type == kind)]
        solutes[column] = rows.set_index("FileHandle").DeltaGsolv
    solutes = solutes[solutes[list(REFERENCE_COLUMNS)].notna().any(axis=1)]
    return solutes.rename_axis("handle").reset_index()


def describe(database: Path, row: pd.Series) -> dict[str, object]:
    """The perceived structure of one solute, or why it has none."""
    try:
        structure = mnsol.describe(mnsol.perceive(database, row.handle, row.Formula))
    except mnsol.PerceptionError as exc:
        return {"error": f"structure not established ({exc})"}
    return {"error": "", **vars(structure)}


def exclusion_reason(row: pd.Series, max_heavy_atoms: int, max_rotatable_bonds: int) -> str:
    """Why a solute is left out, or '' if it is kept."""
    if row.error:
        return str(row.error)
    if row.n_heavy > max_heavy_atoms:
        return f"more than {max_heavy_atoms} heavy atoms"
    if row.n_rotatable > max_rotatable_bonds:
        return f"more than {max_rotatable_bonds} rotatable bonds"
    try:
        validate_input_smiles(row.smiles)
    except ValueError as exc:
        return "refused by the pipeline: " + str(exc).split(":")[0].lower()
    if get_formal_charge(row.smiles) != row.Charge:
        return "SMILES charge differs from the database charge"
    return ""


def merge_duplicates(kept: pd.DataFrame) -> pd.DataFrame:
    """One row per SMILES; handles and names of merged solutes are joined with ';'."""
    rows = []
    for smiles, group in kept.groupby("smiles", sort=False):
        row = {
            "unique_ID": ";".join(group.handle),
            "name": ";".join(dict.fromkeys(group.SoluteName)),
            "SMILES": smiles,
            "charge": int(group.Charge.iloc[0]),
            "atoms": int(group.n_atoms.iloc[0]),
            "heavy_atoms": int(group.n_heavy.iloc[0]),
            "rotatable_bonds": int(group.n_rotatable.iloc[0]),
            "polar_rotors": int(group.n_polar_rotors.iloc[0]),
        }
        for column in REFERENCE_COLUMNS:
            values = group[column].dropna()
            if len(values) and values.max() - values.min() > MERGE_TOLERANCE:
                print(
                    f"  warning: {row['unique_ID']} merged with different {column} "
                    f"values {sorted(values)}; using their mean"
                )
            row[column] = values.mean() if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(selected: pd.DataFrame) -> None:
    print(f"\n{'reference':<20}{'solutes':>9}{'neutral':>9}{'anions':>8}{'cations':>9}")
    for column in REFERENCE_COLUMNS:
        have = selected[selected[column].notna()]
        print(
            f"{column:<20}{len(have):>9}{(have.charge == 0).sum():>9}"
            f"{(have.charge < 0).sum():>8}{(have.charge > 0).sum():>9}"
        )
    small = selected[selected.atoms <= 2]
    print(
        f"\n{len(small)} kept solutes have one or two atoms and no conformers to search: "
        + ", ".join(small.SMILES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help=f"the unpacked MNSol v2012 directory (holding {mnsol.TABLE})",
    )
    parser.add_argument("--output", type=Path, default=Path("mnsol_set.csv"))
    parser.add_argument("--max-heavy-atoms", type=int, default=10)
    parser.add_argument("--max-rotatable-bonds", type=int, default=2)
    args = parser.parse_args()

    if not (args.database / mnsol.TABLE).exists():
        print(f"error: {args.database / mnsol.TABLE} not found", file=sys.stderr)
        sys.exit(1)

    RDLogger.DisableLog("rdApp.*")
    solutes = load_solutes(args.database)
    print(f"{len(solutes)} solutes have a value in water, octanol or hexadecane")
    structures = pd.DataFrame([describe(args.database, row) for row in solutes.itertuples()])
    solutes = pd.concat([solutes, structures], axis=1)

    solutes["excluded"] = [
        exclusion_reason(row, args.max_heavy_atoms, args.max_rotatable_bonds)
        for row in solutes.itertuples()
    ]
    for reason, group in solutes[solutes.excluded != ""].groupby("excluded"):
        # With the handle: MNSol names an ion-water cluster (c001) after its ion's
        # parent (acetylene), which reads like the neutral molecule.
        first = group.head(4)
        examples = ", ".join(
            f"{h} {n}" for h, n in zip(first.handle, first.SoluteName, strict=True)
        )
        print(f"  dropped {len(group):>3}: {reason} (e.g. {examples})")

    kept = solutes[solutes.excluded == ""]
    selected = merge_duplicates(kept)
    for row in selected[selected.unique_ID.str.contains(";")].itertuples():
        print(f"  merged: {row.name} -> {row.SMILES}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, float_format="%.2f")
    print(f"\n{len(selected)} solutes kept ({len(kept)} database entries)")
    print_summary(selected)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
