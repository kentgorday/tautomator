#!/usr/bin/env python
"""Run the pKa pipeline over every SMILES in the pKa set's CSV.

Builds a per-molecule TOML config, runs the three-stage pipeline, and writes
each molecule's outputs (sampling/, refinement/, ensemble.json) into its own
subdirectory under --output-root.

Per-molecule failures are caught and logged so a single bad compound does
not abort the whole batch.

Usage:
    pixi run python scripts/run_pka_set.py \
        --pka-set pka_set.csv \
        --output-root pka_set_output \
        --driver pyscf --threads 8 --protocol high_quality

Resume after interruption:
    pixi run python scripts/run_pka_set.py --resume ...

Each subdirectory is named for the molecule's unique_ID, so --resume finds a
molecule's earlier output wherever its row now sits in the CSV -- including
output from older runs, whose directories carry a row-number prefix.

--resume skips finished molecules outright and, for an unfinished one, picks up
at the first stage that has no checkpoint on disk -- a molecule killed during
scoring reuses its sampling and refinement output rather than redoing days of
DFT.  Note that only the molecule's SMILES is validated against the checkpoint;
see _find_resume_point in tautomator/pipeline.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

from tautomator.batch import compound_dir_name, find_compound_dir
from tautomator.config import DEFAULT_MEMORY_GB, PROTOCOLS, load_config
from tautomator.pipeline import run_pipeline


def make_toml(
    smi: str,
    output_dir: Path,
    driver: str,
    threads: int,
    memory_gb: float,
    protocol: str = "high_quality",
) -> str:
    """Build a TOML config string for one molecule.

    Charge range is hardcoded to [-2, +2]. Empty charge states (e.g. when
    the molecule cannot reach -2) flow through harmlessly.

    ``protocol`` picks every level of theory and energy window (see PROTOCOLS in
    tautomator/config.py); ``driver`` runs the DFT -- scoring always, and refinement
    too unless the protocol refines with OrbMol.  Solvation is the protocol's
    business rather than this script's: it knows that both DFT drivers minimize
    in implicit solvent, and that OrbMol has no implicit solvent and needs a
    separate single point.
    """
    # A TOML basic string: escape the backslashes of E/Z bonds before the quotes.
    safe_smi = smi.replace("\\", "\\\\").replace('"', '\\"')
    return f'''\
protocol = "{protocol}"

[molecule]
smiles = "{safe_smi}"
charge_min = -2
charge_max = 2
solvent = "water"

[compute]
driver = "{driver}"
threads = {threads}
memory_gb = {memory_gb}
output_dir = "{output_dir}"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pKa pipeline over the pKa set's CSV")
    parser.add_argument(
        "--pka-set",
        type=Path,
        default=Path("pka_set.csv"),
        help=(
            "CSV with at least a SMILES column. Output directories are named for "
            "the unique_ID column; without one, for the row number (mol_N)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pka_set_output"),
    )
    parser.add_argument("--driver", default="pyscf", choices=["pyscf", "psi4"])
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        default="high_quality",
        help=(
            "Level of theory and energy windows for all three stages "
            "(default: high_quality; 'fast' refines with OrbMol-v2 and scores "
            "on def2-TZVPPD, ~5x cheaper for ~0.05 pKa)"
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--memory",
        type=float,
        default=DEFAULT_MEMORY_GB,
        metavar="GB",
        help=(
            f"Per-QM-job memory ceiling in GB (default: {DEFAULT_MEMORY_GB}). "
            "Applies to both backends; a solvated job gives half of it to the PCM's "
            "surface integrals, kept between SCF iterations only if they fit. PySCF "
            "will try to allocate up to this, so keep it under real free memory."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N molecules (useful for smoke testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip molecules whose final ensemble.json already exists, and for "
            "the rest reuse any completed sampling/ or refinement/ checkpoint "
            "instead of recomputing it"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if not args.pka_set.exists():
        print(
            f"Error: {args.pka_set} not found. Run scripts/select_pka_set.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.pka_set.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SMILES"):
                rows.append(row)

    if args.limit:
        rows = rows[: args.limit]

    uids = [row.get("unique_ID") or f"mol_{i}" for i, row in enumerate(rows, start=1)]
    by_dir_name: dict[str, list[str]] = {}
    for uid in uids:
        by_dir_name.setdefault(compound_dir_name(uid), []).append(uid)
    clashes = [ids for ids in by_dir_name.values() if len(ids) > 1]
    if clashes:
        listed = "; ".join(", ".join(ids) for ids in clashes)
        print(f"Error: rows would share an output directory: {listed}", file=sys.stderr)
        sys.exit(1)

    log.info(f"Running pipeline on {len(rows)} molecules")

    n_ok = n_skip = n_fail = 0
    for i, (row, uid) in enumerate(zip(rows, uids, strict=True), start=1):
        smi = row["SMILES"]
        compound_dir = find_compound_dir(args.output_root, uid)
        ensemble_json = compound_dir / "ensemble.json"

        if args.resume and ensemble_json.exists():
            log.info(f"[{i}/{len(rows)}] {uid} ({smi}): already done, skipping")
            n_skip += 1
            continue

        log.info(f"[{i}/{len(rows)}] {uid} ({smi}): starting")
        try:
            compound_dir.mkdir(parents=True, exist_ok=True)
            toml_text = make_toml(
                smi,
                compound_dir,
                args.driver,
                args.threads,
                args.memory,
                protocol=args.protocol,
            )
            toml_path = compound_dir / "config.toml"
            toml_path.write_text(toml_text)

            config = load_config(toml_path)
            run_pipeline(config, resume=args.resume)
            n_ok += 1
            log.info(f"[{i}/{len(rows)}] {uid}: done")
        except Exception as e:
            n_fail += 1
            log.error(f"[{i}/{len(rows)}] {uid}: FAILED: {e}")
            (compound_dir / "ERROR.txt").write_text(traceback.format_exc())

    print(f"\nSummary: {n_ok} ok, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
