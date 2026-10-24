"""Scoring stage: high-level DFT single-point energy evaluation.

Replaces refinement-level energies with higher-level DFT energies. The
quasi-RRHO vibrational free-energy correction is carried over unchanged from
refinement (scoring never recomputes frequencies).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from tautomator.cavity_quadrature import INSIDE_GRID, OUTSIDE_GRID
from tautomator.config import (
    DEFAULT_MEMORY_GB,
    DROP_NLEB,
    DROP_PRUNE_LEVEL,
    PCM_METHODS,
    PCM_SWITCHING,
    drop_solvent_model,
    solvent_dielectric,
    solvent_electron_density,
)
from tautomator.ensemble import filter_charge_state_by_energy
from tautomator.nonelectrostatic import (
    DELTA,
    VV_C,
    Descriptors,
    ScoredConformer,
    descriptor_file,
    save_descriptors,
)
from tautomator.types import Conformer, Ensemble, ExcludedConformer, Microstate

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
        # PySCF's cavity's spheres (config.pcm_radii, hydrogen at pcm_hydrogen_radius); a
        # file without this key is from before, on MOIST's own CPCM radii
        "drop_radii": "modified Bondi x 1.2",
        "drop_nleb": DROP_NLEB,
        "drop_prune_level": DROP_PRUNE_LEVEL,
        "volume_rule": list(VOLUME_RULE),
        "ray_rule": list(RAY_RULE),
        "partition_a": SSF_A,
    }


def _scored(
    charge: int, ms: Microstate, conf: Conformer, gas: float, total: float, desc: Descriptors
) -> ScoredConformer:
    """A conformer's record for the descriptor file of the model ``total`` came from."""
    return ScoredConformer(
        charge=charge,
        microstate=ms.tautomer_id,
        includes_enantiomer=ms.includes_enantiomer,
        geometry=conf.geometry,
        electronic_energy=gas,
        solvation_energy=total - gas,
        rrho_correction=conf.rrho_correction,
        multiplicity=conf.multiplicity,
        descriptors=desc,
    )


def check_descriptor_models(solvent_model: str | None, descriptor_models: Sequence[str]) -> None:
    """Raise unless the run's solvent model and every further model can give descriptors:
    analytical PCM models (IEF-PCM, C-PCM, COSMO) on PySCF's cavity or a DROP cavity, each
    once."""

    def analytical(model: str) -> bool:
        return (
            drop_solvent_model(model) is not None or model.upper().replace("-", "") in PCM_METHODS
        )

    if solvent_model is None or not analytical(solvent_model):
        raise ValueError(
            "The nonelectrostatic descriptors need an analytical PCM solvent model (IEF-PCM, "
            f"C-PCM or COSMO, on PySCF's cavity or a DROP cavity), not {solvent_model!r}"
        )
    own = solvent_model.upper().replace("-", "")
    seen: set[str] = set()
    for model in descriptor_models:
        key = model.upper().replace("-", "")
        if not analytical(model):
            raise ValueError(
                f"descriptor model {model!r} is not an analytical PCM model (IEF-PCM, C-PCM "
                "or COSMO, on PySCF's cavity or a DROP cavity)"
            )
        if key == own:
            raise ValueError(
                f"descriptor model {model!r} is the run's own solvent model, whose "
                "descriptors are saved anyway"
            )
        if key in seen:
            raise ValueError(f"descriptor model {model!r} is listed twice")
        seen.add(key)


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
    descriptors_dir: Path | None = None,
    descriptor_models: Sequence[str] = (),
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

    With ``descriptors_dir`` (PySCF and an analytical PCM model only; for fitting
    the nonelectrostatic correction, as scripts/run_mnsol_set.py does), the
    solvated single point also yields the nonelectrostatic descriptors, and every
    conformer scored there is saved with its energies
    (`nonelectrostatic.save_descriptors`, to descriptors_dir/DESCRIPTOR_FILE),
    those the energy window then drops included: the correction reweights the
    conformers, and can bring one from outside the window into it.  Each of
    ``descriptor_models`` (further analytical PCM models) adds one more solvated
    single point per conformer, at the same geometry, whose PCM energy and
    descriptors go to descriptors_dir/descriptor_file(model); the ensemble's
    energies stay the run's own model's.  A conformer that fails on one of them is
    left out of that file only, and listed in its settings ("failed").

    Modifies the ensemble in-place and returns it.
    """
    driver = _get_driver(driver_name, drop=drop_solvent_model(solvent_model) is not None)
    if descriptors_dir is not None:
        if driver_name != "pyscf" or solvent_model is None or solvent is None:
            raise ValueError(
                "The nonelectrostatic descriptors come from a solvated PySCF single "
                f"point; got driver {driver_name!r}, solvent model {solvent_model!r}"
            )
        check_descriptor_models(solvent_model, descriptor_models)
        from tautomator import pyscf_nonelectrostatic
    elif descriptor_models:
        raise ValueError("descriptor_models needs descriptors_dir")
    # descriptors and failures by model, None the run's own
    scored: dict[str | None, list[ScoredConformer]] = {m: [] for m in (None, *descriptor_models)}
    failed: dict[str, list[dict[str, object]]] = {m: [] for m in descriptor_models}

    for cs in ensemble.charge_states.values():
        log.info(f"Scoring charge state q={cs.charge}...")
        for ms in cs.microstates:
            surviving = []
            for conf in ms.conformers:
                try:
                    if solvent_model is not None:
                        if descriptors_dir is None:
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
                        if descriptors_dir is not None:
                            assert solvent is not None  # checked above

                            scored[None].append(_scored(cs.charge, ms, conf, gas, total, desc))
                            for model in descriptor_models:
                                try:
                                    total_m, desc_m = (
                                        pyscf_nonelectrostatic.single_point_with_descriptors(
                                            conf.geometry,
                                            cs.charge,
                                            method,
                                            basis,
                                            model,
                                            solvent,
                                            pcm_hydrogen_radius=pcm_hydrogen_radius,
                                            threads=threads,
                                            memory_gb=memory_gb,
                                        )
                                    )
                                    scored[model].append(
                                        _scored(cs.charge, ms, conf, gas, total_m, desc_m)
                                    )
                                except Exception as e:
                                    log.warning(
                                        f"  Descriptors on {model} failed for a conformer in "
                                        f"microstate {ms.tautomer_id[:8]}: {e}"
                                    )
                                    failed[model].append(
                                        {
                                            "charge": cs.charge,
                                            "microstate": ms.tautomer_id,
                                            "geometry": conf.geometry.coords.tolist(),
                                            "error": f"{type(e).__name__}: {e}",
                                        }
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

    if descriptors_dir is not None:
        assert solvent is not None and solvent_model is not None
        for name in (None, *descriptor_models):
            path = descriptors_dir / descriptor_file(name)
            settings: dict[str, object] = {
                "method": method,
                "basis": basis,
                "solvent_model": name or solvent_model,
                # the model the run's geometries and weights come from
                "geometry_solvent_model": solvent_model,
                "solvent": solvent,
                "eps": solvent_dielectric(solvent),
                "pcm_hydrogen_radius": pcm_hydrogen_radius,
                "pcm_switching": PCM_SWITCHING,
                "solvent_rho": solvent_electron_density(solvent),
                "delta": DELTA,
                "vv_c": VV_C,
                # what Descriptors.charge_field is built from, on either kind of cavity
                "charge_field": "apparent surface charges q",
                **_quadrature_settings(name or solvent_model),
            }
            if name is None:
                settings["descriptor_models"] = list(descriptor_models)
            else:
                settings["failed"] = failed[name]
            save_descriptors(path, scored[name], settings)
            log.info(f"  Saved descriptors of {len(scored[name])} conformer(s) to {path}")

    return ensemble
