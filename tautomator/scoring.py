"""Scoring stage: high-level DFT single-point energy evaluation.

Replaces refinement-level energies with higher-level DFT energies. The
quasi-RRHO vibrational free-energy correction is carried over unchanged from
refinement (scoring never recomputes frequencies).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

from tautomator.cavity_quadrature import INSIDE_GRID, OUTSIDE_GRID
from tautomator.config import (
    DEFAULT_MEMORY_GB,
    DROP_PRUNE_LEVEL,
    PCM_SWITCHING,
    drop_solvent_model,
    solvent_dielectric,
    solvent_electron_density,
)
from tautomator.ensemble import filter_charge_state_by_energy
from tautomator.nonelectrostatic import DELTA, VV_C, ScoredConformer, save_descriptors
from tautomator.types import Ensemble, ExcludedConformer

log = logging.getLogger(__name__)


def _quadrature_settings(solvent_model: str | None) -> dict[str, object]:
    """How the descriptors' volume integrals were done: cavity grids on PySCF's cavity, the
    surface rays and the all-space grid on a DROP cavity (with MOIST's surface settings)."""
    drop = drop_solvent_model(solvent_model)
    if drop is None:
        return {"inside_grid": list(INSIDE_GRID), "outside_grid": list(OUTSIDE_GRID)}
    from tautomator.surface_quadrature import RAY_RULE, SSF_A, VOLUME_RULE

    return {
        "cavity": drop[1],
        "drop_prune_level": DROP_PRUNE_LEVEL,
        "volume_rule": list(VOLUME_RULE),
        "ray_rule": list(RAY_RULE),
        "partition_a": SSF_A,
    }


def _get_driver(name: str, *, drop: bool = False) -> ModuleType:
    """Return the DFT driver module for the given name.

    drop: the run has a DROP solvent model, so Psi4 must be able to import MOIST.
    """
    if name == "psi4":
        from tautomator import psi4_runner

        # Outside the per-conformer handlers, which would record a missing
        # Psi4 (or MOIST) as each conformer's own failure.
        psi4_runner.check_installed(drop=drop)
        return psi4_runner
    if name == "pyscf":
        from tautomator import pyscf_runner

        return pyscf_runner
    raise ValueError(f"Unknown driver: {name!r}. Must be 'psi4' or 'pyscf'.")


def score(
    ensemble: Ensemble,
    driver_name: str,
    method: str,
    basis: str,
    solvent_model: str | None = None,
    solvent: str | None = None,
    ewin: float = 10.0,
    pcm_hydrogen_radius: float = 1.1,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
    descriptors_path: Path | None = None,
) -> Ensemble:
    """Score all conformers via high-level DFT single-point energy.

    For each conformer in each charge state:
      1. Run DFT single-point (with solvent if configured).
      2. If solvent is used, run a gas-phase single-point to decompose
         into electronic and solvation components.

    Any quasi-RRHO correction set during refinement is left untouched and
    contributes to the conformer free energy used for filtering and weighting.
    So are the conformer multiplicities: scoring does not move geometries, so
    the symmetry analysis from refinement still holds.

    Conformers that fail are dropped with a warning. After processing,
    conformers within each charge state are filtered by the energy window.

    With ``descriptors_path`` (PySCF and a PCM model only), the solvated single
    point also yields the nonelectrostatic descriptors, and every conformer
    scored there is saved with its energies (`nonelectrostatic.save_descriptors`),
    those the energy window then drops included: the correction reweights the
    conformers, and can bring one from outside the window into it.

    Modifies the ensemble in-place and returns it.
    """
    driver = _get_driver(driver_name, drop=drop_solvent_model(solvent_model) is not None)
    if descriptors_path is not None:
        if driver_name != "pyscf" or solvent_model is None or solvent is None:
            raise ValueError(
                "The nonelectrostatic descriptors come from a solvated PySCF single "
                f"point; got driver {driver_name!r}, solvent model {solvent_model!r}"
            )
        from tautomator import pyscf_nonelectrostatic
    scored: list[ScoredConformer] = []

    for cs in ensemble.charge_states.values():
        log.info(f"Scoring charge state q={cs.charge}...")
        for ms in cs.microstates:
            surviving = []
            for conf in ms.conformers:
                try:
                    if solvent_model is not None:
                        if descriptors_path is None:
                            total = driver.single_point(
                                conf.geometry,
                                cs.charge,
                                method,
                                basis,
                                solvent_model,
                                solvent,
                                pcm_hydrogen_radius=pcm_hydrogen_radius,
                                threads=threads,
                                memory_gb=memory_gb,
                            )
                        else:
                            assert solvent is not None  # checked above
                            total, desc = pyscf_nonelectrostatic.single_point_with_descriptors(
                                conf.geometry,
                                cs.charge,
                                method,
                                basis,
                                solvent_model,
                                solvent,
                                pcm_hydrogen_radius=pcm_hydrogen_radius,
                                threads=threads,
                                memory_gb=memory_gb,
                            )
                        gas = driver.single_point(
                            conf.geometry,
                            cs.charge,
                            method,
                            basis,
                            threads=threads,
                            memory_gb=memory_gb,
                        )
                        conf.electronic_energy = gas
                        conf.solvation_energy = total - gas
                        if descriptors_path is not None:
                            scored.append(
                                ScoredConformer(
                                    charge=cs.charge,
                                    microstate=ms.tautomer_id,
                                    includes_enantiomer=ms.includes_enantiomer,
                                    geometry=conf.geometry,
                                    electronic_energy=gas,
                                    solvation_energy=total - gas,
                                    rrho_correction=conf.rrho_correction,
                                    multiplicity=conf.multiplicity,
                                    descriptors=desc,
                                )
                            )
                    else:
                        conf.electronic_energy = driver.single_point(
                            conf.geometry,
                            cs.charge,
                            method,
                            basis,
                            threads=threads,
                            memory_gb=memory_gb,
                        )
                        conf.solvation_energy = None

                    surviving.append(conf)
                except Exception as e:
                    # Includes a cavity psi4-swig-pcm refuses (S not positive-
                    # definite): the solvated Psi4 job fails rather than
                    # returning a converged but meaningless energy.
                    # Scoring does not deduplicate, so this conformer may be a
                    # representative standing for several collapsed structures;
                    # multiplicity records how many states go with it.
                    log.warning(
                        f"  Scoring failed for conformer in microstate {ms.tautomer_id[:8]}: {e}"
                    )
                    ms.excluded_conformers.append(
                        ExcludedConformer(
                            geometry=conf.geometry,
                            stage="scoring",
                            reason="scoring_failed",
                            detail=f"{type(e).__name__}: {e}",
                            multiplicity=conf.multiplicity,
                            electronic_energy=conf.electronic_energy,
                            solvation_energy=conf.solvation_energy,
                            rrho_correction=conf.rrho_correction,
                            refinement_converged=conf.refinement_converged,
                        )
                    )
            ms.conformers = surviving

        filter_charge_state_by_energy(cs, ewin)
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: {len(cs.microstates)} microstate(s), "
            f"{n_conf} conformer(s) after filtering"
        )

    if descriptors_path is not None:
        assert solvent is not None
        save_descriptors(
            descriptors_path,
            scored,
            {
                "method": method,
                "basis": basis,
                "solvent_model": solvent_model,
                "solvent": solvent,
                "eps": solvent_dielectric(solvent),
                "pcm_hydrogen_radius": pcm_hydrogen_radius,
                "pcm_switching": PCM_SWITCHING,
                "solvent_rho": solvent_electron_density(solvent),
                "delta": DELTA,
                "vv_c": VV_C,
                **_quadrature_settings(solvent_model),
            },
        )
        log.info(f"  Saved descriptors of {len(scored)} conformer(s) to {descriptors_path}")

    return ensemble
