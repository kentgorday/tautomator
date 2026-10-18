"""Compare predicted and experimental pKas for a run over the pKa set.

The pKa set is for evaluation only: nothing here is fitted to it.  Each pKa
comes from the direct thermodynamic cycle, the Boltzmann-summed free energies of
the two charge states plus a fixed proton term.

Transitions come from `tautomator.pka_set.parse_pka_column`, which reads them
off the experimental label. Pairing values with transitions positionally
mis-assigns the two entries that mix `pKa` and `pKaH` labels.

Transitions whose ion is also in the MNSol set (--mnsol-set) are reported
apart.  MNSol's aqueous ion free energies were derived from experimental pKas
through this same cycle and the same proton solvation energy (methylammonium:
-76.5 kcal/mol from the cycle, -76.4 in MNSol), so once the nonelectrostatic
correction is fitted on MNSol, those pKas test the gas-phase acidity rather
than the correction's transfer to new molecules.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from tautomator.batch import find_compound_dir
from tautomator.ensemble import load_ensemble
from tautomator.pka_set import PkaValue, parse_pka_column

HARTREE_TO_KCAL = 627.509
RT_KCAL = 0.5924
LN10_RT = 1.364
# The proton's free energy in water at 1 M, kcal/mol: -6.28 in the gas phase at
# 1 atm, +1.89 from 1 atm to 1 M, and -265.9 for its solvation (Tissandier et
# al., 1998).  Not a fit parameter: MNSol's ion values were derived with the
# same -265.9, so changing it here would set the pKa cycle against the fit.
PROTON_TERM = -270.29


def _components(charge_state: object) -> tuple[np.ndarray, np.ndarray]:
    gas, solv = [], []
    for ms in charge_state.microstates:  # type: ignore[attr-defined]
        for c in ms.conformers:
            gas.append(c.electronic_energy + (c.rrho_correction or 0.0))
            solv.append(c.solvation_energy or 0.0)
    return np.array(gas), np.array(solv)


def _free_energy(gas: np.ndarray, solv: np.ndarray) -> float:
    """Boltzmann-summed free energy in solution, hartree."""
    g = gas + solv
    lowest = g.min()
    return float(
        lowest
        - (RT_KCAL / HARTREE_TO_KCAL)
        * np.log(np.exp(-(g - lowest) * HARTREE_TO_KCAL / RT_KCAL).sum())
    )


def _predict(g_protonated: float, g_deprotonated: float) -> float:
    """pKa implied by the free energies of the two charge states."""
    delta = (g_deprotonated - g_protonated) * HARTREE_TO_KCAL
    return (delta + PROTON_TERM) / LN10_RT


def _parent(smiles: str) -> str | None:
    """Canonical SMILES of the neutral parent, to match species across charges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return str(Chem.MolToSmiles(rdMolStandardize.Uncharger().uncharge(mol)))


def _mnsol_ions(path: Path) -> dict[str, set[int]]:
    """Neutral parent -> the nonzero charges at which MNSol has the solute."""
    ions: dict[str, set[int]] = {}
    for row in csv.DictReader(path.open()):
        charge = int(row["charge"])
        if charge and (parent := _parent(row["SMILES"])) is not None:
            ions.setdefault(parent, set()).add(charge)
    return ions


def _summary(name: str, errors: list[float]) -> str:
    err = np.array(errors)
    return (
        f"{name}: n={len(err)}  mean err {np.mean(err):+.2f}  sd {np.std(err):.2f}"
        f"  MAE {np.mean(np.abs(err)):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--pka-set", type=Path, default=Path("pka_set.csv"))
    parser.add_argument(
        "--mnsol-set",
        type=Path,
        default=Path("mnsol_set.csv"),
        help="report transitions whose ion is in this set apart (skipped if absent)",
    )
    parser.add_argument(
        "--substitute",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="use PATH's ensemble.json for molecule ID (e.g. a corrected rerun)",
    )
    args = parser.parse_args()

    meta = {r["unique_ID"]: r for r in csv.DictReader(args.pka_set.open())}
    mnsol = _mnsol_ions(args.mnsol_set) if args.mnsol_set.exists() else None
    paths = {
        uid: path
        for uid in meta
        if (path := find_compound_dir(args.output_root, uid) / "ensemble.json").exists()
    }
    for item in args.substitute:
        name, _, path = item.partition("=")
        paths[name] = Path(path)

    rows: list[tuple[str, str, PkaValue, float, bool]] = []
    for name, path in paths.items():
        if name not in meta:
            continue
        ensemble = load_ensemble(path)
        smiles = meta[name]["SMILES"]
        ion_charges = mnsol.get(_parent(smiles) or "", set()) if mnsol is not None else set()
        for value in parse_pka_column(meta[name]["pkas"]):
            hi, lo = value.higher_charge, value.lower_charge
            if hi not in ensemble.charge_states or lo not in ensemble.charge_states:
                continue
            gas_hi, solv_hi = _components(ensemble.charge_states[hi])
            gas_lo, solv_lo = _components(ensemble.charge_states[lo])
            if not len(gas_hi) or not len(gas_lo):
                continue
            pred = _predict(_free_energy(gas_hi, solv_hi), _free_energy(gas_lo, solv_lo))
            rows.append((name, smiles, value, pred, bool(ion_charges & {hi, lo})))

    print(
        f"{'molecule':<16}{'SMILES':<24}{'label':<7}{'q':>7}{'pred':>7}{'exp':>7}{'err':>7}"
        f"{'MNSol':>7}"
    )
    for name, smiles, value, pred, in_mnsol in sorted(
        rows, key=lambda r: (r[2].label, -abs(r[3] - r[2].value))
    ):
        arrow = f"{value.higher_charge:+d}/{value.lower_charge:+d}"
        print(
            f"{name:<16}{smiles:<24}{value.label:<7}{arrow:>7}"
            f"{pred:>7.2f}{value.value:>7.2f}{pred - value.value:>+7.2f}"
            f"{'ion' if in_mnsol else '':>7}"
        )

    print()
    for label in sorted({r[2].label for r in rows}):
        sub = [r for r in rows if r[2].label == label]
        if len(sub) < 2:
            continue
        line = _summary(label, [r[3] - r[2].value for r in sub])
        if len(sub) > 2:
            predicted = np.array([r[3] for r in sub])
            observed = np.array([r[2].value for r in sub])
            line += f"  r={np.corrcoef(predicted, observed)[0, 1]:.3f}"
        print(line)

    if not rows:
        return
    print()
    print(_summary("all", [r[3] - r[2].value for r in rows]))
    if mnsol is None:
        print(f"({args.mnsol_set} not found: no split by MNSol overlap)")
        return
    for name, keep in (("ion not in MNSol", False), ("ion in MNSol", True)):
        errors = [r[3] - r[2].value for r in rows if r[4] is keep]
        if errors:
            print(_summary(name, errors))


if __name__ == "__main__":
    main()
