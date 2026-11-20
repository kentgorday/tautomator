"""Boltzmann weighting, partition functions, and ensemble serialization."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from tautomator.conformer_symmetry import (
    KB_HARTREE,
    conformer_multiplicity,
    deduplicate,
    effective_energy_offset,
)
from tautomator.types import (
    ChargeState,
    Conformer,
    Ensemble,
    ExcludedConformer,
    Geometry,
    Microstate,
    Well,
)

# Constants
HARTREE_TO_KCAL = 627.5094740631

# Bumped when a change makes an older ensemble.json unsafe to resume from.
# 1 -> 2: conformers carry a `multiplicity`; without it every conformer reloads
# at 1.0 and the run silently reproduces the old unweighted answer.
# 3 -> 4: the CREST-first identity carries tetrahedral configuration as well as
# hydrogen counts, so `tautomer_id` values written by an older build mean
# something narrower and cannot be compared with ones written now.
# 2 -> 3: heavy atoms are embedded in `rdkit_utils.frame_atom_order`, so every
# microstate of a molecule shares one heavy-atom order. Older files predate it,
# and the proton-migration repair refuses to move a conformer between two
# microstates that disagree about that order -- so resuming would silently skip
# the re-filing the stage exists to do.
SCHEMA_VERSION = 4


def boltzmann_weights(
    energies: list[float],
    temperature: float = 298.15,
) -> list[float]:
    """Compute normalized Boltzmann weights from energies in Hartree."""
    e = np.array(energies)
    e_rel = e - e.min()
    log_weights = -e_rel / (KB_HARTREE * temperature)
    # Shift for numerical stability
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    result: list[float] = (weights / total).tolist()
    return result


def ensemble_free_energy(
    energies: list[float],
    temperature: float = 298.15,
) -> float:
    """Compute ensemble free energy: G = -kT * ln(Z) + E_min.

    Energies in Hartree, returns Hartree. The partition function Z is
    computed relative to the minimum energy for numerical stability.
    """
    e = np.array(energies)
    e_min = e.min()
    e_rel = e - e_min
    z = np.sum(np.exp(-e_rel / (KB_HARTREE * temperature)))
    return float(e_min - KB_HARTREE * temperature * np.log(z))


def _conformer_multiplicity(ms: Microstate, conf: Conformer) -> float:
    """How many physical states this conformer contributes, over its sigma.

    Two independent factors, disjoint by construction:

    * ``Microstate.includes_enantiomer`` -- the pipeline collapses
      *configurational* enantiomers in `stereo.deduplicate_enantiomers`, so a
      surviving microstate stands for both members of that pair.
    * ``Conformer.multiplicity`` -- set by
      `conformer_symmetry.conformer_multiplicity` after deduplication, covering
      the conformer's own rotational symmetry number and, when the conformer is
      chiral by conformation alone, the mirror image that deduplication folded
      into it.
    """
    enant = 2.0 if ms.includes_enantiomer else 1.0
    return enant * conf.multiplicity


def charge_state_free_energy(
    charge_state: ChargeState,
    temperature: float = 298.15,
) -> float:
    """Compute the free energy of a charge state.

    Boltzmann-averages over all microstates and all conformers within
    the charge state.  Each conformer is weighted by its multiplicity
    (see `_conformer_multiplicity`).
    """
    if not any(ms.conformers for ms in charge_state.microstates):
        raise ValueError(f"Charge state {charge_state.charge} has no conformers")

    kbt = KB_HARTREE * temperature
    e_min = min(conf.free_energy for ms in charge_state.microstates for conf in ms.conformers)
    z = 0.0
    for ms in charge_state.microstates:
        for conf in ms.conformers:
            z += _conformer_multiplicity(ms, conf) * np.exp(-(conf.free_energy - e_min) / kbt)
    return float(e_min - kbt * np.log(z))


def assign_weights(ensemble: Ensemble, temperature: float = 298.15) -> None:
    """Assign Boltzmann weights to all conformers within each charge state.

    Weights are normalized across all conformers in a charge state
    (spanning all microstates), since macroscopic pKa depends on the
    full partition function of each charge state. Each conformer carries
    its own multiplicity (see `_conformer_multiplicity`).
    """
    kbt = KB_HARTREE * temperature
    for cs in ensemble.charge_states.values():
        entries: list[tuple[Conformer, float]] = []
        for ms in cs.microstates:
            for conf in ms.conformers:
                entries.append((conf, _conformer_multiplicity(ms, conf)))
        if not entries:
            continue
        e_min = min(conf.free_energy for conf, _ in entries)
        raw_weights = [mult * np.exp(-(conf.free_energy - e_min) / kbt) for conf, mult in entries]
        total = sum(raw_weights)
        for (conf, _), w in zip(entries, raw_weights, strict=True):
            conf.weight = float(w / total)


def _effective_energy(ms: Microstate, conf: Conformer, temperature: float = 298.15) -> float:
    """``G - RT ln(m)``: the energy whose Boltzmann factor is the contribution.

    A conformer contributes ``m exp(-G/RT)``, which is ``exp(-(G - RT ln m)/RT)``,
    so this is what an energy window should be measured against. A conformer
    standing for two states is effectively 0.41 kcal/mol lower than its bare
    free energy suggests, and should survive the window on that basis.
    """
    return conf.free_energy + effective_energy_offset(
        _conformer_multiplicity(ms, conf), temperature
    )


def filter_charge_state_by_energy(
    cs: ChargeState, ewin_kcal: float, temperature: float = 298.15
) -> None:
    """Remove conformers outside the energy window within a charge state.

    The window is relative to the lowest *effective* energy across all
    conformers in all microstates of this charge state (see
    `_effective_energy`). Microstates with no surviving conformers are pruned.
    """
    entries = [(ms, c) for ms in cs.microstates for c in ms.conformers]
    if not entries:
        return
    e_min = min(_effective_energy(ms, c, temperature) for ms, c in entries)
    ewin_hartree = ewin_kcal / HARTREE_TO_KCAL
    for ms in cs.microstates:
        ms.conformers = [
            c
            for c in ms.conformers
            if (_effective_energy(ms, c, temperature) - e_min) <= ewin_hartree
        ]
    # Keep a microstate whose conformers all failed: its excluded_conformers are
    # the record of what could not be computed, and a total failure is exactly
    # the case most worth having on disk.  Pruning on `ms.conformers` alone
    # discarded it, defeating the point of keeping the record at all.
    cs.microstates = [ms for ms in cs.microstates if ms.conformers or ms.excluded_conformers]


def deduplicate_conformers(
    conformers: list[Conformer],
    includes_enantiomer: bool = False,
    ethr_kcal: float | None = None,
    collapse_tops: bool = False,
    stale_rrho: bool = False,
) -> list[Conformer]:
    """Collapse symmetry-duplicate conformers and set each survivor's multiplicity.

    Mirror images are merged, and the surviving conformer carries a
    multiplicity of two in their place. Enantiomeric conformers are
    isoenergetic by symmetry, so optimizing and scoring both spends the work
    twice and lets two independent Hessians disagree about a quantity that must
    be identical.

    Multiplicities are recomputed here rather than carried forward, because
    both the symmetry number and the achirality flag are properties of the
    geometry, and geometries move between stages.

    ``ethr_kcal`` adds the energy criterion CREGEN pairs with its RMSD
    threshold. Worth using on unrelaxed geometries, where two structures can
    sit within the RMSD threshold and still differ substantially in energy;
    unnecessary once geometries are optimized.

    ``collapse_tops`` treats methyl-type tops as turning freely, both in which
    conformers are one state and in what each counts (see
    `conformer_symmetry.deduplicate`).  A group can then hold a methyl's minimum
    and its hilltop, which differ by more than any duplicates do.

    Each group keeps a converged geometry if it has one, then the lowest in
    free energy.  ``stale_rrho`` says the RRHO terms were taken at other
    geometries, as at refinement before its Hessians, and the lowest is then
    taken in the energy the geometries were optimized on, electronic plus
    solvation.  A survivor that has no `Well` takes the record of the lowest
    member that has one.

    All conformers must share one atom ordering, which holds because they
    describe a single microstate and a microstate is one labelled species.
    """
    if not conformers:
        return []
    symbols = list(conformers[0].geometry.symbols)
    mismatched = next((c for c in conformers if list(c.geometry.symbols) != symbols), None)
    if mismatched is not None:
        raise ValueError(
            "conformers of one microstate must share an atom ordering; got "
            f"{''.join(symbols)} and {''.join(mismatched.geometry.symbols)}. "
            "Conformers that migrated to another microstate are re-filed by "
            "protomer_geometry.repair_migrated_conformers, which regroups their "
            "hydrogens into the destination's order; reaching here means one was "
            "moved some other way."
        )

    coords = np.array([c.geometry.coords for c in conformers])
    energies = (
        np.array([c.free_energy * HARTREE_TO_KCAL for c in conformers])
        if ethr_kcal is not None
        else None
    )
    groups = deduplicate(
        coords, symbols, energies=energies, ethr=ethr_kcal, collapse_tops=collapse_tops
    )

    def energy(c: Conformer) -> float:
        if stale_rrho and c.electronic_energy is not None:
            return c.electronic_energy + (c.solvation_energy or 0.0)
        return c.free_energy

    keys = [(0 if c.refinement_converged is not False else 1, energy(c)) for c in conformers]
    keep = [min(g, key=lambda k: keys[k]) for g in groups]
    survivors = [conformers[k] for k in keep]
    for conf, group in zip(survivors, groups, strict=True):
        if conf.well is None and (recorded := [k for k in group if conformers[k].well]):
            conf.well = conformers[min(recorded, key=lambda k: keys[k])].well
    mult = conformer_multiplicity(
        coords[keep], symbols, includes_enantiomer=includes_enantiomer, collapse_tops=collapse_tops
    )
    for conf, m in zip(survivors, mult, strict=True):
        conf.multiplicity = float(m)
    return survivors


def deduplicate_charge_state(
    cs: ChargeState,
    ethr_kcal: float | None = None,
    collapse_tops: bool = False,
    stale_rrho: bool = False,
) -> tuple[int, int]:
    """Deduplicate every microstate of a charge state; return ``(before, after)``.

    Deduplication runs *within* each microstate: conformers of different
    microstates are different species, so they are never compared.
    ``collapse_tops`` and ``stale_rrho`` as for `deduplicate_conformers`.
    """
    before = after = 0
    for ms in cs.microstates:
        before += len(ms.conformers)
        ms.conformers = deduplicate_conformers(
            ms.conformers, ms.includes_enantiomer, ethr_kcal, collapse_tops, stale_rrho
        )
        after += len(ms.conformers)
    return before, after


def _well_data(well: Well | None) -> dict[str, object] | None:
    """A `Well` as JSON: the dropped shape by its coordinates alone, in the conformer's atoms."""
    if well is None:
        return None
    data: dict[str, object] = {
        f.name: getattr(well, f.name) for f in dataclasses.fields(well) if f.name != "dropped"
    }
    data["dropped"] = None if well.dropped is None else well.dropped.coords.tolist()
    return data


def _load_well(data: dict[str, Any] | None, symbols: tuple[str, ...]) -> Well | None:
    """`_well_data` read back; a record from before ``dropped`` was kept has none."""
    if not data:
        return None
    fields = dict(data)
    coords = fields.pop("dropped", None)
    dropped = None if coords is None else Geometry(symbols=symbols, coords=np.array(coords))
    return Well(**fields, dropped=dropped)


def serialize_ensemble(ensemble: Ensemble, output_dir: Path) -> Path:
    """Write ensemble to a single JSON file with inline coordinates.

    Returns the path to the JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_smiles": ensemble.input_smiles,
        "settings": ensemble.settings,
        "charge_states": {},
    }

    cs_data: dict[str, object] = {}
    for charge, cs in sorted(ensemble.charge_states.items()):
        ms_list: list[dict[str, object]] = []
        for ms in cs.microstates:
            conf_list: list[dict[str, object]] = []
            for conf in ms.conformers:
                conf_list.append(
                    {
                        "symbols": list(conf.geometry.symbols),
                        "coords": conf.geometry.coords.tolist(),
                        "electronic_energy": conf.electronic_energy,
                        "solvation_energy": conf.solvation_energy,
                        "rrho_correction": conf.rrho_correction,
                        "free_energy": conf.free_energy,
                        "weight": conf.weight,
                        "refinement_converged": conf.refinement_converged,
                        "multiplicity": conf.multiplicity,
                        "well": _well_data(conf.well),
                    }
                )
            excluded_list: list[dict[str, object]] = []
            for exc in ms.excluded_conformers:
                excluded_list.append(
                    {
                        "symbols": list(exc.geometry.symbols),
                        "coords": exc.geometry.coords.tolist(),
                        "stage": exc.stage,
                        "reason": exc.reason,
                        "detail": exc.detail,
                        "multiplicity": exc.multiplicity,
                        "electronic_energy": exc.electronic_energy,
                        "solvation_energy": exc.solvation_energy,
                        "rrho_correction": exc.rrho_correction,
                        "refinement_converged": exc.refinement_converged,
                    }
                )
            ms_entry: dict[str, object] = {
                "tautomer_id": ms.tautomer_id,
                "smiles": ms.smiles,
                "includes_enantiomer": ms.includes_enantiomer,
                "n_conformers": len(ms.conformers),
                "conformers": conf_list,
            }
            # Omitted when empty so that runs with nothing excluded produce the
            # same file they did before, and old readers are unaffected.
            if excluded_list:
                ms_entry["n_excluded"] = len(excluded_list)
                ms_entry["excluded_conformers"] = excluded_list
            ms_list.append(ms_entry)
        cs_data[str(charge)] = {
            "charge": charge,
            "n_microstates": len(cs.microstates),
            "microstates": ms_list,
        }

    data["charge_states"] = cs_data

    json_path = output_dir / "ensemble.json"
    json_path.write_text(json.dumps(data, indent=2))
    return json_path


def schema_version(path: Path) -> int:
    """Schema version of a serialized ensemble; 1 for files written before it existed."""
    return int(json.loads(path.read_text()).get("schema_version", 1))


def load_ensemble(path: Path) -> Ensemble:
    """Load ensemble from JSON with inline coordinates.

    Fields added after a file was written take their defaults, so an older
    ensemble loads without error. Whether it is safe to *resume* from is a
    separate question -- see `schema_version` and `pipeline._find_resume_point`.
    """
    raw = json.loads(path.read_text())
    ensemble = Ensemble(
        input_smiles=raw["input_smiles"],
        settings=raw.get("settings", {}),
    )
    for charge_str, cs_data in raw.get("charge_states", {}).items():
        charge = int(charge_str)
        microstates: list[Microstate] = []
        for ms_data in cs_data.get("microstates", []):
            conformers: list[Conformer] = []
            for conf_data in ms_data.get("conformers", []):
                geom = Geometry(
                    symbols=tuple(conf_data["symbols"]),
                    coords=np.array(conf_data["coords"]),
                )
                conformers.append(
                    Conformer(
                        geometry=geom,
                        electronic_energy=conf_data.get("electronic_energy"),
                        solvation_energy=conf_data.get("solvation_energy"),
                        rrho_correction=conf_data.get("rrho_correction"),
                        weight=conf_data.get("weight"),
                        refinement_converged=conf_data.get("refinement_converged"),
                        multiplicity=conf_data.get("multiplicity", 1.0),
                        # Absent before the record existed, so no schema bump: such a
                        # conformer simply has no comparison on record.
                        well=_load_well(conf_data.get("well"), geom.symbols),
                    )
                )
            # Absent in files written before excluded_conformers existed, which
            # is why this needs no schema bump: an old checkpoint simply has
            # nothing excluded, which is a true statement about it.
            excluded: list[ExcludedConformer] = []
            for exc_data in ms_data.get("excluded_conformers", []):
                excluded.append(
                    ExcludedConformer(
                        geometry=Geometry(
                            symbols=tuple(exc_data["symbols"]),
                            coords=np.array(exc_data["coords"]),
                        ),
                        stage=exc_data["stage"],
                        reason=exc_data["reason"],
                        detail=exc_data.get("detail", ""),
                        multiplicity=exc_data.get("multiplicity", 1.0),
                        electronic_energy=exc_data.get("electronic_energy"),
                        solvation_energy=exc_data.get("solvation_energy"),
                        rrho_correction=exc_data.get("rrho_correction"),
                        refinement_converged=exc_data.get("refinement_converged"),
                    )
                )
            microstates.append(
                Microstate(
                    tautomer_id=ms_data["tautomer_id"],
                    conformers=conformers,
                    smiles=ms_data.get("smiles"),
                    includes_enantiomer=ms_data.get("includes_enantiomer", False),
                    excluded_conformers=excluded,
                )
            )
        ensemble.charge_states[charge] = ChargeState(charge=charge, microstates=microstates)
    return ensemble


def _mol_from_smiles_and_coords(smiles: str, geom: Geometry, charge: int) -> Chem.Mol:
    """Build an RDKit mol from explicit-H SMILES, setting coordinates from geometry.

    Assumes the geometry atom ordering matches the SMILES atom ordering
    (i.e. geometry was reordered at creation time via _smilesAtomOutputOrder).
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    if mol.GetNumAtoms() != geom.n_atoms:
        raise ValueError(
            f"SMILES atom count ({mol.GetNumAtoms()}) != geometry atom count ({geom.n_atoms})"
        )

    conf = Chem.Conformer(geom.n_atoms)
    for i in range(geom.n_atoms):
        conf.SetAtomPosition(
            i,
            (float(geom.coords[i, 0]), float(geom.coords[i, 1]), float(geom.coords[i, 2])),
        )
    mol.AddConformer(conf, assignId=True)
    return mol


def _mol_from_coords(geom: Geometry, charge: int) -> Chem.Mol:
    """Build an RDKit mol from coordinates only, using rdDetermineBonds."""
    mol = Chem.RWMol()
    for sym in geom.symbols:
        mol.AddAtom(Chem.Atom(sym))
    conf = Chem.Conformer(geom.n_atoms)
    for i in range(geom.n_atoms):
        conf.SetAtomPosition(
            i,
            (float(geom.coords[i, 0]), float(geom.coords[i, 1]), float(geom.coords[i, 2])),
        )
    mol.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return Chem.Mol(mol)


def ensemble_to_sdf(ensemble: Ensemble, output_path: Path) -> Path:
    """Write all conformers in an ensemble to an SDF file.

    If a microstate has an explicit-H SMILES (approach 1), bond orders
    come from the SMILES. Otherwise (approach 2), bonds are determined
    from coordinates via rdDetermineBonds.

    Each conformer is written as a separate record with properties:
    charge, tautomer_id, smiles, energy_hartree, boltzmann_weight.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))

    for charge, cs in sorted(ensemble.charge_states.items()):
        sorted_ms = sorted(cs.microstates, key=lambda m: min(c.free_energy for c in m.conformers))
        for ms in sorted_ms:
            for conf in sorted(ms.conformers, key=lambda c: c.free_energy):
                if ms.smiles is not None:
                    mol = _mol_from_smiles_and_coords(ms.smiles, conf.geometry, charge)
                else:
                    mol = _mol_from_coords(conf.geometry, charge)

                mol.SetIntProp("charge", charge)
                mol.SetProp("tautomer_id", ms.tautomer_id)
                if ms.smiles is not None:
                    mol.SetProp("smiles", ms.smiles)
                mol.SetDoubleProp("free_energy_hartree", conf.free_energy)
                if conf.electronic_energy is not None:
                    mol.SetDoubleProp("electronic_energy_hartree", conf.electronic_energy)
                if conf.solvation_energy is not None:
                    mol.SetDoubleProp("solvation_energy_hartree", conf.solvation_energy)
                if conf.rrho_correction is not None:
                    mol.SetDoubleProp("rrho_correction_hartree", conf.rrho_correction)
                if conf.weight is not None:
                    mol.SetDoubleProp("boltzmann_weight", conf.weight)

                writer.write(mol)

    writer.close()
    return output_path
