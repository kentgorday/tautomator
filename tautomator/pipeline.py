"""Three-stage pKa prediction pipeline: sampling → refinement → scoring."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from tautomator.config import PkaConfig
from tautomator.ensemble import (
    SCHEMA_VERSION,
    assign_weights,
    load_ensemble,
    schema_version,
    serialize_ensemble,
)
from tautomator.refinement import refine
from tautomator.sampling import run_approach1, run_approach2
from tautomator.scoring import check_descriptor_models, score
from tautomator.types import Ensemble

log = logging.getLogger(__name__)

# Stages that checkpoint to <output_dir>/<name>/ensemble.json, latest first.
# Scoring has no entry: it writes the final <output_dir>/ensemble.json, and a
# molecule with that file is simply done.
_RESUMABLE_STAGES = ("refinement", "sampling")


def _find_resume_point(output_dir: Path, smiles: str) -> tuple[Ensemble, str] | None:
    """Return (ensemble, stage_name) for the latest completed stage, or None.

    The schema version and ``input_smiles`` are validated.  The refinement
    checkpoint also records the refinement settings (``settings["refinement"]``),
    but they are not compared with the current config: resuming after editing
    [refinement], or after one of its defaults changed, silently mixes levels of
    theory.
    The SMILES check still catches the dangerous case: a directory holding
    another molecule's run (one from before batch directories were named by
    ``unique_ID`` alone, whose row numbers shift when the pKa set is edited, or
    an ID whose SMILES was corrected) would otherwise resume one molecule's run
    against another's data.
    """
    for stage in _RESUMABLE_STAGES:
        path = output_dir / stage / "ensemble.json"
        if not path.exists():
            continue
        found_version = schema_version(path)
        if found_version < SCHEMA_VERSION:
            log.warning(
                f"Ignoring {path}: schema v{found_version}, this build writes "
                f"v{SCHEMA_VERSION}. Resuming would reload every conformer at the "
                f"default multiplicity of 1.0 and silently reproduce the old, "
                f"unweighted answer. Rerun the stage instead."
            )
            return None
        ensemble = load_ensemble(path)
        if ensemble.input_smiles != smiles:
            log.warning(
                f"Ignoring {path}: it holds {ensemble.input_smiles!r}, "
                f"but this config asks for {smiles!r}"
            )
            return None
        return ensemble, stage
    return None


def run_pipeline(
    config: PkaConfig, resume: bool = False, *, descriptors: Sequence[str] | None = None
) -> Ensemble:
    """Run the full three-stage pKa prediction pipeline.

    Stage 1 (Sampling): CREST-based conformer/tautomer/protonation enumeration,
        with an xTB-level RRHO correction folded into the energy filter.
    Stage 2 (Refinement): geometry optimization at the refinement level (DFT, or
        the OrbMol-v2 potential), a solvation energy per [refinement] solvation,
        and an RRHO recompute on that geometry (xtb --bhess or a Hessian from
        whatever optimized it, per [refinement] rrho).
    Stage 3 (Scoring): DFT single-point energy (RRHO carried over unchanged).

    With resume=True, any stage whose ensemble.json is already on disk is
    reloaded instead of recomputed, and the run picks up at the next stage.
    Stages are days long here, so a run killed during scoring would otherwise
    redo sampling and refinement from scratch.  Resumption happens at stage
    granularity: a stage interrupted partway through restarts from its
    beginning, since neither refine() nor score() checkpoints per conformer.

    descriptors (not part of the config: for fitting the nonelectrostatic
    correction, scripts/run_mnsol_set.py): save every scored conformer's
    nonelectrostatic descriptors with its energies, on the run's own solvent model
    and on each further model listed (one more solvated single point per conformer
    and model); `scoring.score` says where.  None saves none.

    Returns Ensemble with Boltzmann weights assigned.
    """
    output_dir = Path(config.compute.output_dir)
    if descriptors is not None:
        # before any stage runs, not after days of sampling and refinement
        if config.compute.driver != "pyscf":
            raise ValueError("The nonelectrostatic descriptors need [compute] driver = 'pyscf'")
        check_descriptor_models(config.scoring.solvent_model, descriptors)

    # One solvent for the whole run: xTB's ALPB during sampling and xtb-level
    # RRHO, and the implicit-solvent model at refinement and scoring.  A run
    # that mixed them would filter on one solvent and score in another.  A run
    # scored in the gas phase uses none: it samples in the gas phase too (and
    # refine() computes its RRHO there), so its free energy is a gas-phase one
    # throughout, as the reference state of a solvation free energy must be.
    solvent = config.molecule.solvent
    sampling_solvent = solvent if config.scoring.solvent_model is not None else None
    provenance = config.provenance()

    resumed_from: str | None = None
    ensemble: Ensemble | None = None
    if resume:
        found = _find_resume_point(output_dir, config.molecule.smiles)
        if found is not None:
            ensemble, resumed_from = found
            n_conf = sum(
                len(ms.conformers)
                for cs in ensemble.charge_states.values()
                for ms in cs.microstates
            )
            log.info(
                f"=== Resuming after {resumed_from}: reusing {n_conf} conformer(s) "
                f"from {output_dir / resumed_from / 'ensemble.json'} ==="
            )

    # Stage 1: Sampling
    if ensemble is not None:
        log.info("=== Stage 1: Sampling (skipped, reusing checkpoint) ===")
    else:
        log.info("=== Stage 1: Sampling ===")
        if config.sampling.approach == "rdkit_first":
            ensemble = run_approach1(
                smiles=config.molecule.smiles,
                charge_range=config.molecule.charge_range,
                solvent=sampling_solvent,
                crest_mode=config.sampling.crest_mode,
                ewin=config.sampling.ewin,
                threads=config.compute.threads,
                max_tautomers=config.sampling.max_tautomers,
                max_transforms=config.sampling.max_transforms,
                fixed_species=config.sampling.fixed_species,
            )
        else:
            ensemble = run_approach2(
                smiles=config.molecule.smiles,
                charge_range=config.molecule.charge_range,
                solvent=sampling_solvent,
                prescreen_mode=config.sampling.prescreen_mode,
                full_mode=config.sampling.full_mode,
                prescreen_ewin=config.sampling.prescreen_ewin,
                ewin=config.sampling.ewin,
                threads=config.compute.threads,
            )
        serialize_ensemble(ensemble, output_dir / "sampling")

    # Stage 2: Refinement
    threads = config.compute.threads or 1
    memory_gb = config.compute.memory_gb
    if resumed_from == "refinement":
        log.info("=== Stage 2: Refinement (skipped, reusing checkpoint) ===")
    else:
        log.info("=== Stage 2: Refinement ===")
        refine(
            ensemble,
            config.refinement,
            driver=config.compute.driver,
            solvent=solvent,
            solvent_model=config.scoring.solvent_model,
            threads=threads,
            memory_gb=memory_gb,
        )
        # What this stage's energies actually are.  Sampling's settings say
        # nothing about the refinement level, so a checkpoint on disk used to be
        # unreadable without the config that produced it.
        ensemble.settings["protocol"] = provenance["protocol"]
        ensemble.settings["refinement"] = provenance["refinement"]
        serialize_ensemble(ensemble, output_dir / "refinement")

    # Stage 3: Scoring
    log.info("=== Stage 3: Scoring ===")
    sc = config.scoring
    score(
        ensemble,
        driver_name=config.compute.driver,
        method=sc.method,
        basis=sc.basis,
        solvent_model=sc.solvent_model,
        solvent=solvent,
        ewin=sc.ewin,
        pcm_hydrogen_radius=sc.pcm_hydrogen_radius,
        threads=threads,
        memory_gb=memory_gb,
        descriptors_dir=output_dir if descriptors is not None else None,
        descriptor_models=tuple(descriptors or ()),
    )
    ensemble.settings["scoring"] = provenance["scoring"]

    # Final: assign Boltzmann weights
    assign_weights(ensemble)
    serialize_ensemble(ensemble, output_dir)

    return ensemble
