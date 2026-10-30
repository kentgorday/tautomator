#!/usr/bin/env python
"""Run the pipeline on the MNSol set in each medium its reference values need.

Each solute in the CSV scripts/select_mnsol_set.py writes is one fixed species
at one charge, so every run samples that species alone ([sampling]
fixed_species) at its own charge.  A solvation free energy is the difference of
two runs, G(solvent) - G(gas), each over its own conformer ensemble, so a
solute runs in:

  water       for dG_water, and for the logP test (dG_octanol_water)
  woctanol    for dG_octanol and the logP test: MNSol's octanol values come
              mostly from logP, i.e. water-saturated octanol
  hexadecane  for dG_hexadecane
  gas         for any absolute value; a gas-phase run samples, refines and
              takes its RRHO in the gas phase throughout

The logP-only solutes need no gas run: logP is G(woctanol) - G(water).

Output goes to <output-root>/<medium>/<solute>/, each a normal pipeline output
directory (sampling/, refinement/, ensemble.json, config.toml).  A solvent run
minimizes and scores in IEF-PCM on MOIST's SvdW-DROP cavity (--cavity), and saves
every conformer it scored, with its energies and nonelectrostatic descriptors, to
nonelectrostatic.npz; and the same at the same geometries on PySCF's
switching/Gaussian cavity (ISWIG) and on rho-DROP (--descriptor-cavities), to
nonelectrostatic_iswig.npz and nonelectrostatic_rhodrop.npz (read any with
nonelectrostatic.load_descriptors).  The fit reweights the conformers from
there without any quantum chemistry, on whichever cavity it chooses.  The
descriptors are not a config setting: main() asks run_pipeline for them.  Each
further cavity costs one more solvated single point per conformer; rho-DROP's
rebuilds its surface every SCF iteration and costs the most.

Solutes run in CSV order, all of one solute's media before the next solute, so
a partial run leaves complete solvation free energies behind.  Failures are
caught per run and written to ERROR.txt in its directory; --resume skips
finished runs and picks up unfinished ones at their last completed stage.  A
solvent run finished without all its descriptor files counts as unfinished,
and reruns scoring.

PySCF is the default driver here.  At high_quality on an M-series Mac (8
threads, sampling reused) the two drivers refined in the same time, but Psi4
scored at def2-QZVPPD 1.2-1.4x slower: toluene in water 215 s against PySCF's
199 s at memory_gb 4 (Psi4 at 12 GB; at 4 GB, 242 s), toluene in the gas phase
100 s against 82 s, acetate in water 46 s against 37 s (all on PySCF's cavity).
Both drivers keep the PCM's surface integrals in core when they fit in half of
--memory; at the default 12 GB and def2-QZVPPD that holds for 73% of the set on
SvdW-DROP at its nleb 434 (config.DROP_NLEB) and 81% on PySCF's cavity.  And the
nonelectrostatic descriptors can only be read off a PySCF mean field
(pyscf_nonelectrostatic), so a Psi4 run needs --no-descriptors.  Run one job at
a time on a 24 GB machine: toluene's scoring job peaked at 10.5 GB, and at 12 GB
the largest solutes that keep their surface integrals add up to 6 GB to that
(an estimate, not measured).

Usage:
    pixi run python scripts/run_mnsol_set.py \
        --mnsol-set mnsol_set.csv --output-root mnsol_output \
        --driver pyscf --threads 8 --memory 12

    # a pilot: the first 20 solutes, water and gas only
    pixi run python scripts/run_mnsol_set.py --limit 20 --media water gas ...
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from pathlib import Path

from tautomator.batch import compound_dir_name
from tautomator.config import CAVITIES, DEFAULT_CAVITY, DEFAULT_MEMORY_GB, PROTOCOLS, load_config
from tautomator.nonelectrostatic import descriptor_file
from tautomator.pipeline import run_pipeline
from tautomator.scoring import check_descriptor_cavities

# The cavities a fit can choose between (config.CAVITIES): solvent runs minimize and score
# on one (--cavity, the default cavity unless given) and by default save descriptors on the
# others at the same geometries.


def default_descriptor_cavities(cavity: str) -> list[str]:
    """CAVITIES less the run's own cavity, whose descriptors scoring saves anyway."""
    return [c for c in CAVITIES if c != cavity]


# Medium -> the reference columns that need a run in it.
MEDIA: dict[str, tuple[str, ...]] = {
    "water": ("dG_water", "dG_octanol_water"),
    "woctanol": ("dG_octanol", "dG_octanol_water"),
    "hexadecane": ("dG_hexadecane",),
    "gas": ("dG_water", "dG_octanol", "dG_hexadecane"),
}


def media_for(row: dict[str, str], wanted: list[str]) -> list[str]:
    """The media, among ``wanted``, that this solute's reference values need."""
    return [m for m in wanted if any(row.get(column) for column in MEDIA[m])]


def make_toml(
    smiles: str,
    charge: int,
    medium: str,
    output_dir: Path,
    driver: str,
    threads: int,
    memory_gb: float,
    protocol: str,
    cavity: str,
) -> str:
    """The config for one solute in one medium.

    The protocol sets every level of theory and window; a solvent run minimizes and
    scores in IEF-PCM on ``cavity``, and a gas-phase run only switches scoring to the gas
    phase, which takes refinement, sampling and the RRHO there with it.  Its
    [molecule] solvent is unused, and it has no descriptors: there is no cavity in
    the gas phase.  The descriptors are not in the config: main() asks run_pipeline
    for them.
    """
    # A TOML basic string: escape the backslashes of E/Z bonds before the quotes.
    safe_smiles = smiles.replace("\\", "\\\\").replace('"', '\\"')
    gas = medium == "gas"
    scoring = f'[scoring]\ncavity = "{"none" if gas else cavity}"\n\n'
    return f'''\
protocol = "{protocol}"

[molecule]
smiles = "{safe_smiles}"
charge_min = {charge}
charge_max = {charge}
solvent = "{"water" if gas else medium}"

[sampling]
fixed_species = true

{scoring}[compute]
driver = "{driver}"
threads = {threads}
memory_gb = {memory_gb}
output_dir = "{output_dir}"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mnsol-set", type=Path, default=Path("mnsol_set.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("mnsol_output"))
    parser.add_argument(
        "--media",
        nargs="+",
        choices=list(MEDIA),
        default=list(MEDIA),
        help="run only these media (default: all four)",
    )
    parser.add_argument("--driver", default="pyscf", choices=["pyscf", "psi4"])
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="high_quality")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--memory",
        type=float,
        default=DEFAULT_MEMORY_GB,
        metavar="GB",
        help=(
            f"per-QM-job memory in GB (default: {DEFAULT_MEMORY_GB}).  Either driver "
            "keeps the PCM's surface integrals in core only when they fit in half of it."
        ),
    )
    parser.add_argument(
        "--cavity",
        default=DEFAULT_CAVITY,
        choices=CAVITIES,
        help=f"the cavity solvent runs minimize and score on (default: {DEFAULT_CAVITY})",
    )
    parser.add_argument(
        "--descriptor-cavities",
        nargs="*",
        default=None,
        choices=CAVITIES,
        metavar="CAVITY",
        help=(
            "further cavities to save descriptors on, at the same geometries, one more "
            "solvated single point per conformer each (default: those of "
            f"{' '.join(CAVITIES)} other than --cavity; none: pass the flag alone)"
        ),
    )
    parser.add_argument(
        "--no-descriptors",
        dest="descriptors",
        action="store_false",
        help="do not save the nonelectrostatic descriptors (needed with --driver psi4)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N solutes")
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="ID", help="only these unique_IDs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip finished runs, and restart unfinished ones at their last completed stage",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if args.descriptors and args.driver != "pyscf":
        print(
            "Error: the nonelectrostatic descriptors need --driver pyscf; "
            "pass --no-descriptors to run without them.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.descriptor_cavities is None:
        args.descriptor_cavities = default_descriptor_cavities(args.cavity)
    if args.descriptors:
        # here rather than in every run, which would each record the same error
        try:
            check_descriptor_cavities(args.cavity, args.descriptor_cavities)
        except ValueError as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    if not args.mnsol_set.exists():
        print(
            f"Error: {args.mnsol_set} not found. Run scripts/select_mnsol_set.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with args.mnsol_set.open() as f:
        rows = [row for row in csv.DictReader(f) if row.get("SMILES")]
    if args.only:
        rows = [row for row in rows if row["unique_ID"] in set(args.only)]
    if args.limit:
        rows = rows[: args.limit]

    # One directory per solute; the check is case-blind because macOS file
    # systems are.
    by_dir: dict[str, list[str]] = {}
    for row in rows:
        by_dir.setdefault(compound_dir_name(row["unique_ID"]).lower(), []).append(row["unique_ID"])
    clashes = [ids for ids in by_dir.values() if len(ids) > 1]
    if clashes:
        listed = "; ".join(", ".join(ids) for ids in clashes)
        print(f"Error: solutes would share an output directory: {listed}", file=sys.stderr)
        sys.exit(1)

    runs = [(row, medium) for row in rows for medium in media_for(row, args.media)]
    log.info(f"{len(runs)} runs for {len(rows)} solutes ({', '.join(args.media)})")

    n_ok = n_skip = n_fail = 0
    for i, (row, medium) in enumerate(runs, start=1):
        uid, smiles, charge = row["unique_ID"], row["SMILES"], int(row["charge"])
        run_dir = args.output_root / medium / compound_dir_name(uid)
        label = f"[{i}/{len(runs)}] {uid} ({smiles}) in {medium}"
        descriptors = args.descriptors and medium != "gas"
        files = [descriptor_file(c) for c in (None, *args.descriptor_cavities)]
        finished = (run_dir / "ensemble.json").exists() and (
            not descriptors or all((run_dir / f).exists() for f in files)
        )
        if args.resume and finished:
            log.info(f"{label}: already done, skipping")
            n_skip += 1
            continue

        log.info(f"{label}: starting")
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "ERROR.txt").unlink(missing_ok=True)
            toml = make_toml(
                smiles,
                charge,
                medium,
                run_dir,
                args.driver,
                args.threads,
                args.memory,
                args.protocol,
                args.cavity,
            )
            (run_dir / "config.toml").write_text(toml)
            run_pipeline(
                load_config(run_dir / "config.toml"),
                resume=args.resume,
                descriptors=args.descriptor_cavities if descriptors else None,
            )
            n_ok += 1
            log.info(f"{label}: done")
        except Exception as e:
            n_fail += 1
            log.error(f"{label}: FAILED: {e}")
            (run_dir / "ERROR.txt").write_text(traceback.format_exc())

    print(f"\nSummary: {n_ok} ok, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
