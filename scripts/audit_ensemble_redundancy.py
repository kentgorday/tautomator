#!/usr/bin/env python
"""Report how many conformers in a finished run are symmetry duplicates.

The pipeline deduplicates as it goes, so a healthy run should report close to
zero here. A non-zero count means conformers reached the output that describe
the same physical state -- worth investigating rather than ignoring, since each
one double-counts in the partition function.

Usage:
    pixi run python scripts/audit_ensemble_redundancy.py --stage refinement
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from tautomator.conformer_symmetry import DEFAULT_RTHR, deduplicate
from tautomator.ensemble import load_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", default="pka_set_output")
    ap.add_argument("--stage", default="final", choices=["sampling", "refinement", "final"])
    ap.add_argument("--rthr", type=float, default=DEFAULT_RTHR)
    args = ap.parse_args()

    pattern = (
        os.path.join(args.output_root, "*", "ensemble.json")
        if args.stage == "final"
        else os.path.join(args.output_root, "*", args.stage, "ensemble.json")
    )

    grand_conf = grand_uniq = 0
    print(f"{'molecule':<24}{'conformers':>12}{'distinct':>10}{'redundant':>11}")
    for path in sorted(glob.glob(pattern)):
        name = path.split(os.sep)[-2 if args.stage == "final" else -3]
        ensemble = load_ensemble(Path(path))
        n_conf = n_uniq = 0
        for cs in ensemble.charge_states.values():
            for ms in cs.microstates:
                if not ms.conformers:
                    continue
                symbols = list(ms.conformers[0].geometry.symbols)
                coords = np.array([c.geometry.coords for c in ms.conformers])
                n_conf += len(ms.conformers)
                n_uniq += len(deduplicate(coords, symbols, rthr=args.rthr))
        grand_conf += n_conf
        grand_uniq += n_uniq
        flag = f"{n_conf - n_uniq:>11}" if n_conf != n_uniq else f"{'-':>11}"
        print(f"{name:<24}{n_conf:>12}{n_uniq:>10}{flag}")

    redundant = grand_conf - grand_uniq
    share = redundant / grand_conf if grand_conf else 0.0
    print(
        f"\nTOTAL {grand_conf} conformers, {grand_uniq} distinct "
        f"({redundant} redundant, {share:.2%})"
    )


if __name__ == "__main__":
    main()
