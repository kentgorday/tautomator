#!/usr/bin/env python
"""Select a diverse pKa set from the IUPAC Dissociation-Constants dataset.

Applies a series of filters and diversity criteria to produce a compact set of
small organic molecules with experimental aqueous pKa values, for evaluating
predicted pKas.  Nothing is fitted to it, so the size limits are there only to
keep the runs affordable.

Usage:
    pixi run python scripts/select_pka_set.py \
        --input ~/Dissociation-Constants/iupac_high-confidence_v2_3.csv \
        --output pka_set.csv \
        --max-heavy-atoms 10 \
        --max-rotatable-bonds 2 \
        --max-dissociations 2 \
        --target-size 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

# Suppress RDKit warnings during bulk processing
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def base_filters(
    df: pd.DataFrame,
    *,
    max_heavy_atoms: int = 10,
    max_rotatable_bonds: int | None = 2,
    max_dissociations: int = 2,
    max_ionizable_sites: int | None = 4,
    t_min: float = 24.0,
    t_max: float = 26.0,
    assessments: tuple[str, ...] = ("Reliable",),
) -> pd.DataFrame:
    """Apply non-negotiable quality and scope filters."""
    out = df.copy()

    # Aqueous only (no cosolvent)
    out = out[out["cosolvent"].isna()]

    # Temperature: keep within 1 degree of 25 C
    out["T_num"] = pd.to_numeric(out["T"], errors="coerce")
    out = out[(out["T_num"] >= t_min) & (out["T_num"] <= t_max)]

    # pKa value: must be numeric
    out["pka_value"] = pd.to_numeric(out["pka_value"], errors="coerce")
    out = out[out["pka_value"].notna()]

    # Assessment: default to Reliable only (highest-confidence digitized data).
    # Pass --include-approximate to widen the pool.
    out = out[out["assessment"].isin(list(assessments))]

    # Only standard pKa types (no solvent mixtures, no predictions)
    valid_types = {"pKa1", "pKa2", "pKa3", "pKaH1", "pKaH2", "pKaH3", "pKb1", "pKb2"}
    out = out[out["pka_type"].isin(valid_types)]

    # Organic: must contain carbon. Ask RDKit rather than the SMILES text, where
    # aromatic carbon is a lowercase "c" and an uppercase "C" also matches Cl.
    organic = [smi for smi in out["SMILES"].dropna().unique() if has_carbon(smi)]
    out = out[out["SMILES"].isin(organic)]

    # Heavy atom count
    def _heavy_atoms(smi: str) -> int | None:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return int(mol.GetNumHeavyAtoms())

    unique_smiles = out["SMILES"].unique()
    ha_map = {smi: _heavy_atoms(smi) for smi in unique_smiles}
    out["heavy_atoms"] = out["SMILES"].map(ha_map)
    out = out[out["heavy_atoms"].notna() & (out["heavy_atoms"] <= max_heavy_atoms)]

    # Rotatable bonds (RDKit's strict definition). Limits the size of the
    # conformer ensemble each charge state carries through refinement.
    if max_rotatable_bonds is not None:
        rot_map = {smi: count_rotatable_bonds(smi) for smi in out["SMILES"].unique()}
        out["rotatable_bonds"] = out["SMILES"].map(rot_map)
        out = out[out["rotatable_bonds"].notna() & (out["rotatable_bonds"] <= max_rotatable_bonds)]

    # Max dissociations per molecule
    dissoc_counts = out.groupby("SMILES")["pka_type"].nunique()
    ok_smiles = dissoc_counts[dissoc_counts <= max_dissociations].index
    out = out[out["SMILES"].isin(ok_smiles)]

    # Max ionizable sites (SMARTS-based count of titratable centers).
    # Limits combinatorial blow-up of protonation states in the QM ensemble.
    if max_ionizable_sites is not None:
        ion_map = {smi: count_ionizable_sites(smi) for smi in out["SMILES"].unique()}
        out["ionizable_sites"] = out["SMILES"].map(ion_map)
        out = out[out["ionizable_sites"].notna() & (out["ionizable_sites"] <= max_ionizable_sites)]

    return out


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------


def classify_acid_base(row: pd.Series) -> str:
    """Classify a row as 'acid' or 'base' based on pka_type."""
    if row["pka_type"].startswith("pKaH"):
        return "base"
    return "acid"


def count_ionizable_sites(smi: str) -> int | None:
    """Count distinct heavy atoms that are titratable, using the same broad
    SMARTS reactions the QM pipeline uses to enumerate (de)protonation states.

    Atoms that can be both protonated and deprotonated (e.g. a neutral amine
    that can go to NH3+ or NH-) are counted once.

    Returns None for invalid SMILES.
    """
    from tautomator.charge_enumeration import _get_deprot_rxns, _get_prot_rxns

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    sites: set[int] = set()
    for rxn in _get_deprot_rxns() + _get_prot_rxns():
        patt = rxn.GetReactantTemplate(0)
        for match in mol.GetSubstructMatches(patt):
            # The :1-mapped atom is the first atom in the SMARTS reactant
            sites.add(match[0])
    return len(sites)


def count_rotatable_bonds(smi: str) -> int | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return int(rdMolDescriptors.CalcNumRotatableBonds(mol))


def has_carbon(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def has_aromatic_ring(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return bool(rdMolDescriptors.CalcNumAromaticRings(mol) > 0)


def pka_bin(value: float, bin_width: float = 2.0) -> int:
    """Assign pKa value to a bin for stratified sampling."""
    return int(np.floor(value / bin_width))


def morgan_fingerprint(smi: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_distance_matrix(fps: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise Tanimoto distance matrix."""
    n = len(fps)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.sum(fps[i] & fps[j])
            union = np.sum(fps[i] | fps[j])
            sim = intersection / union if union > 0 else 0.0
            dist[i, j] = dist[j, i] = 1.0 - sim
    return dist


def greedy_maxmin_pick(dist_matrix: np.ndarray, n_pick: int, seed_idx: int = 0) -> list[int]:
    """MaxMin diversity picking: iteratively select the point most distant
    from the already-selected set."""
    n = dist_matrix.shape[0]
    n_pick = min(n_pick, n)
    picked = [seed_idx]
    min_dists = dist_matrix[seed_idx].copy()

    for _ in range(n_pick - 1):
        # Among unpicked, find the one with largest minimum distance to picked set
        candidates = np.ones(n, dtype=bool)
        candidates[picked] = False
        candidate_indices = np.where(candidates)[0]
        best = candidate_indices[np.argmax(min_dists[candidate_indices])]
        picked.append(best)
        # Update min distances
        min_dists = np.minimum(min_dists, dist_matrix[best])

    return picked


def _count_pkas_by_kind(all_pkas: list[tuple[str, float]]) -> tuple[int, int]:
    """Return (n_acid_pkas, n_base_pkas) for a molecule's pKa list."""
    n_acid = sum(1 for t, _ in all_pkas if t.startswith("pKa") and not t.startswith("pKaH"))
    n_base = sum(1 for t, _ in all_pkas if t.startswith("pKaH"))
    return n_acid, n_base


def _pick_with_priority(
    full_pool: pd.DataFrame,
    priority_pool: pd.DataFrame,
    n_total: int,
    n_priority: int,
) -> pd.DataFrame:
    """Pick n_total rows from full_pool, ensuring up to n_priority come from
    priority_pool first (via MaxMin), then filling the rest from the
    remaining full_pool (also via MaxMin)."""
    if len(full_pool) <= n_total:
        return full_pool

    n_pri = min(n_priority, len(priority_pool))
    if n_pri > 0:
        fps = list(priority_pool["_fp"])
        dist = tanimoto_distance_matrix(fps)
        idx = greedy_maxmin_pick(dist, n_pri)
        priority_picked = priority_pool.iloc[idx]
    else:
        priority_picked = priority_pool.iloc[:0]

    remaining_n = n_total - len(priority_picked)
    if remaining_n <= 0:
        return priority_picked

    picked_smiles = set(priority_picked["SMILES"])
    rest = full_pool[~full_pool["SMILES"].isin(picked_smiles)].reset_index(drop=True)
    if len(rest) <= remaining_n:
        return pd.concat([priority_picked, rest], ignore_index=True)

    fps = list(rest["_fp"])
    dist = tanimoto_distance_matrix(fps)
    idx = greedy_maxmin_pick(dist, remaining_n)
    return pd.concat([priority_picked, rest.iloc[idx]], ignore_index=True)


def select_diverse_subset(
    df: pd.DataFrame,
    *,
    target_size: int = 40,
    pka_bin_width: float = 2.0,
    min_diacids: int = 5,
    min_dibases: int = 5,
) -> pd.DataFrame:
    """Select a diverse subset balancing acid/base counts and chemical diversity.

    Strategy:
      1. Deduplicate to one representative row per molecule (prefer Reliable, T=25).
         Attach all (pka_type, pka_value) tuples for that molecule as `all_pkas`.
      2. Classify acid/base (per primary pKa), flag aromatics, count pKas by kind.
      3. Balance acid/base 50/50.
      4. Within each side, prioritize "pure" multi-pKa molecules (di-acids /
         di-bases) so we get representatives that reach -2 / +2 charge.
         Amphoteric compounds (≥1 pKa AND ≥1 pKaH) don't count toward either
         priority quota — they get picked naturally via diversity.
      5. Fingerprint-based MaxMin diversity within each tier.
    """

    # Map each SMILES to its list of (pka_type, mean_value) — average across
    # independent measurements of the same pka_type.
    def _aggregate(g: pd.DataFrame) -> list[tuple[str, float]]:
        means = g.groupby("pka_type")["pka_value"].mean()
        # Sort: pKa1, pKa2, pKa3, then pKaH1, pKaH2, pKaH3, then pKb1, pKb2
        order = {
            "pKa1": 0,
            "pKa2": 1,
            "pKa3": 2,
            "pKaH1": 3,
            "pKaH2": 4,
            "pKaH3": 5,
            "pKb1": 6,
            "pKb2": 7,
        }
        return sorted(
            ((t, float(v)) for t, v in means.items()),
            key=lambda x: order.get(x[0], 99),
        )

    pka_map: dict[str, list[tuple[str, float]]] = df.groupby("SMILES").apply(_aggregate).to_dict()

    # --- Deduplicate: one best row per SMILES ---
    # Prefer Reliable > Approximate, T closest to 25
    df = df.copy()
    df["_assess_rank"] = df["assessment"].map({"Reliable": 0, "Approximate": 1}).fillna(2)
    df["_t_dist"] = (df["T_num"] - 25.0).abs()
    df = df.sort_values(["_assess_rank", "_t_dist"])
    dedup = df.drop_duplicates(subset="SMILES", keep="first").copy()

    # --- Classify ---
    dedup["acid_base"] = dedup.apply(classify_acid_base, axis=1)
    dedup["is_aromatic"] = dedup["SMILES"].apply(has_aromatic_ring)
    dedup["pka_bin"] = dedup["pka_value"].apply(lambda v: pka_bin(v, pka_bin_width))
    dedup["all_pkas"] = dedup["SMILES"].map(pka_map)
    pka_counts = dedup["all_pkas"].apply(_count_pkas_by_kind)
    dedup["n_acid_pkas"] = pka_counts.apply(lambda x: x[0])
    dedup["n_base_pkas"] = pka_counts.apply(lambda x: x[1])

    # --- Compute fingerprints ---
    fp_map = {}
    for smi in dedup["SMILES"]:
        fp_map[smi] = morgan_fingerprint(smi)
    dedup["_fp"] = dedup["SMILES"].map(fp_map)
    dedup = dedup[dedup["_fp"].notna()].reset_index(drop=True)

    # --- Balance acid/base ---
    acids = dedup[dedup["acid_base"] == "acid"].reset_index(drop=True)
    bases = dedup[dedup["acid_base"] == "base"].reset_index(drop=True)

    half = target_size // 2
    n_acids = min(half, len(acids))
    n_bases = min(target_size - n_acids, len(bases))
    if n_bases < half:
        n_acids = min(target_size - n_bases, len(acids))

    # Priority pools: pure di-acids (≥2 acid pKas, no base pKas) and
    # pure di-bases (≥2 base pKas, no acid pKas).  Amphoteric compounds
    # are excluded from priority; they'll appear naturally via diversity.
    diacids = acids[(acids["n_acid_pkas"] >= 2) & (acids["n_base_pkas"] == 0)].reset_index(
        drop=True
    )
    dibases = bases[(bases["n_base_pkas"] >= 2) & (bases["n_acid_pkas"] == 0)].reset_index(
        drop=True
    )

    selected_parts = []
    if n_acids > 0 and len(acids) > 0:
        selected_parts.append(_pick_with_priority(acids, diacids, n_acids, min_diacids))
    if n_bases > 0 and len(bases) > 0:
        selected_parts.append(_pick_with_priority(bases, dibases, n_bases, min_dibases))

    selected = pd.concat(selected_parts, ignore_index=True)

    # Clean up internal columns
    drop_cols = ["_assess_rank", "_t_dist", "_fp"]
    selected = selected.drop(columns=[c for c in drop_cols if c in selected.columns])

    return selected


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"pKa set: {len(df)} molecules")
    print(f"{'=' * 60}")
    n_acids = (df["acid_base"] == "acid").sum()
    n_bases = (df["acid_base"] == "base").sum()
    n_diacids = ((df["n_acid_pkas"] >= 2) & (df["n_base_pkas"] == 0)).sum()
    n_dibases = ((df["n_base_pkas"] >= 2) & (df["n_acid_pkas"] == 0)).sum()
    n_amphoteric = ((df["n_acid_pkas"] >= 1) & (df["n_base_pkas"] >= 1)).sum()
    print(f"  Acids: {n_acids}, Bases: {n_bases}")
    print(
        f"  Pure di-acids (-> -2): {n_diacids}, "
        f"pure di-bases (-> +2): {n_dibases}, amphoteric: {n_amphoteric}"
    )
    print(f"  Aromatic: {df['is_aromatic'].sum()}")
    print(f"  pKa range: {df['pka_value'].min():.1f} to {df['pka_value'].max():.1f}")
    print(f"  Heavy atoms: {int(df['heavy_atoms'].min())}-{int(df['heavy_atoms'].max())}")
    print("\n  pKa distribution (bin width=2):")
    bins = df["pka_bin"].value_counts().sort_index()
    for b, count in bins.items():
        lo, hi = b * 2, (b + 1) * 2
        print(f"    [{lo:5.1f}, {hi:5.1f}): {count}")
    print()


# ---------------------------------------------------------------------------
# Grid visualization
# ---------------------------------------------------------------------------


def draw_grid(df: pd.DataFrame, output_path: Path, cols_per_row: int = 5) -> None:
    """Draw selected molecules into a grid image organized by acid/base then pKa.

    Layout: acids sorted by pKa (top rows), then bases sorted by pKa (bottom rows).
    Each cell shows the structure with a multi-line label below it (drawn with
    PIL so multi-pKa legends render at full size).
    """
    import io

    from PIL import Image, ImageDraw, ImageFont
    from rdkit.Chem.Draw import rdMolDraw2D

    # Sort: acids by pKa ascending, then bases by pKa ascending
    acids = df[df["acid_base"] == "acid"].sort_values("pka_value").reset_index(drop=True)
    bases = df[df["acid_base"] == "base"].sort_values("pka_value").reset_index(drop=True)
    ordered = pd.concat([acids, bases], ignore_index=True)

    cells = []
    for _, row in ordered.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        all_pkas = row.get("all_pkas") or [(row["pka_type"], row["pka_value"])]
        sites = row.get("ionizable_sites")
        sites_str = f"  ({int(sites)} sites)" if sites is not None and not pd.isna(sites) else ""
        header = f"{row['acid_base'].upper()}{sites_str}"
        pka_lines = [f"{t} = {v:.2f}" for t, v in all_pkas]
        cells.append((mol, header, pka_lines))

    n_mols = len(cells)
    n_rows = (n_mols + cols_per_row - 1) // cols_per_row

    mol_w, mol_h = 320, 260
    label_h = 70  # fixed area for header + up to 3 pKa lines
    cell_w, cell_h = mol_w, mol_h + label_h

    total_w = cols_per_row * cell_w
    total_h = n_rows * cell_h
    canvas = Image.new("RGB", (total_w, total_h), "white")

    # Try to load a real font; fall back to PIL default if unavailable
    def _load_font(size: int):  # type: ignore[no-untyped-def]
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    header_font = _load_font(16)
    pka_font = _load_font(14)

    for idx, (mol, header, pka_lines) in enumerate(cells):
        row_i = idx // cols_per_row
        col_i = idx % cols_per_row
        x0 = col_i * cell_w
        y0 = row_i * cell_h

        # Render molecule (no legend; we draw our own)
        drawer = rdMolDraw2D.MolDraw2DCairo(mol_w, mol_h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        canvas.paste(mol_img, (x0, y0))

        # Draw label below
        draw = ImageDraw.Draw(canvas)
        text_y = y0 + mol_h + 4
        draw.text((x0 + 8, text_y), header, fill="black", font=header_font)
        text_y += 20
        for line in pka_lines:
            draw.text((x0 + 8, text_y), line, fill="black", font=pka_font)
            text_y += 16

    # Separator line + section headers
    n_acid_rows = (len(acids) + cols_per_row - 1) // cols_per_row
    sep_y = n_acid_rows * cell_h
    draw = ImageDraw.Draw(canvas)
    draw.line([(0, sep_y), (total_w, sep_y)], fill="black", width=3)
    section_font = _load_font(14)
    draw.text((8, 4), "ACIDS (sorted by pKa)", fill="black", font=section_font)
    draw.text((8, sep_y + 4), "BASES (sorted by pKa)", fill="black", font=section_font)

    canvas.save(str(output_path))
    print(f"Grid image written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the pKa set from the IUPAC dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Dissociation-Constants" / "iupac_high-confidence_v2_3.csv",
        help="Path to IUPAC CSV",
    )
    parser.add_argument("--output", type=Path, default=Path("pka_set.csv"))
    parser.add_argument("--max-heavy-atoms", type=int, default=10)
    parser.add_argument(
        "--max-rotatable-bonds",
        type=int,
        default=2,
        help="Max rotatable bonds (RDKit strict definition); negative disables the limit",
    )
    parser.add_argument("--max-dissociations", type=int, default=2)
    parser.add_argument(
        "--max-ionizable-sites",
        type=int,
        default=4,
        help="Max distinct titratable heavy atoms (via charge_enumeration SMARTS)",
    )
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--pka-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--min-diacids",
        type=int,
        default=5,
        help="Minimum number of di-acids (2 acid pKas, no base pKas) to include",
    )
    parser.add_argument(
        "--min-dibases",
        type=int,
        default=5,
        help="Minimum number of di-bases (2 base pKas, no acid pKas) to include",
    )
    parser.add_argument("--t-min", type=float, default=24.0, help="Min temperature (C)")
    parser.add_argument("--t-max", type=float, default=26.0, help="Max temperature (C)")
    parser.add_argument(
        "--include-approximate",
        action="store_true",
        help="Also include 'Approximate' assessment (default: Reliable only)",
    )
    parser.add_argument("--grid", type=Path, default=None, help="Output grid image (PNG)")
    parser.add_argument("--cols-per-row", type=int, default=5, help="Columns in grid image")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}, unique SMILES: {df['SMILES'].nunique()}")

    print("Applying base filters...")
    assessments = ("Reliable", "Approximate") if args.include_approximate else ("Reliable",)
    filtered = base_filters(
        df,
        max_heavy_atoms=args.max_heavy_atoms,
        max_rotatable_bonds=args.max_rotatable_bonds if args.max_rotatable_bonds >= 0 else None,
        max_dissociations=args.max_dissociations,
        max_ionizable_sites=args.max_ionizable_sites,
        t_min=args.t_min,
        t_max=args.t_max,
        assessments=assessments,
    )
    print(
        f"  After filtering: {len(filtered)} rows, {filtered['SMILES'].nunique()} unique molecules"
    )

    print("Selecting diverse subset...")
    selected = select_diverse_subset(
        filtered,
        target_size=args.target_size,
        pka_bin_width=args.pka_bin_width,
        min_diacids=args.min_diacids,
        min_dibases=args.min_dibases,
    )

    print_summary(selected)

    # Output columns. Format multi-pKa entries as a semicolon-joined string.
    out = selected.copy()
    out["all_pkas_str"] = out["all_pkas"].apply(
        lambda lst: ";".join(f"{t}={v:.3f}" for t, v in lst)
    )
    out_cols = [
        "unique_ID",
        "SMILES",
        "all_pkas_str",
        "T_num",
        "assessment",
        "acid_base",
        "is_aromatic",
        "heavy_atoms",
    ]
    out = out[[c for c in out_cols if c in out.columns]].rename(columns={"all_pkas_str": "pkas"})
    out.to_csv(args.output, index=False)
    print(f"Written to {args.output}")

    if args.grid:
        draw_grid(selected, args.grid, cols_per_row=args.cols_per_row)


if __name__ == "__main__":
    main()
