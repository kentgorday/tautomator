"""Refinement stage: geometry optimization for sampled conformers.

Replaces xTB-level energies with refinement-level ones -- DFT (psi4/pyscf) or
the OrbMol-v2 machine-learned potential -- adds a solvation energy unless the
stage is gas-phase, and recomputes the quasi-RRHO vibrational free energy.

Per charge state, in order: optimize every conformer, re-file the ones whose
proton moved, fold shallow off-symmetric wells into the symmetric conformer
beside them, collapse duplicates, take a Hessian for each survivor, optionally
skip the solvation single point for conformers already far above the minimum in
gas phase, compute the solvation energies, then apply the energy window.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import cast

import numpy as np

from tautomator import xtb_runner
from tautomator.config import DEFAULT_MEMORY_GB, DROP_CAVITIES, RefinementConfig
from tautomator.conformer_symmetry import conformer_multiplicity, symmetry_number
from tautomator.ensemble import (
    HARTREE_TO_KCAL,
    _effective_energy,
    deduplicate_charge_state,
    filter_charge_state_by_energy,
)
from tautomator.nudge import fold_into, nudge, settle
from tautomator.protomer_geometry import repair_migrated_conformers
from tautomator.thermo import quasi_rrho_free_energy
from tautomator.types import (
    ChargeState,
    Ensemble,
    ExcludedConformer,
    ExclusionReason,
    Geometry,
    Microstate,
    Well,
)

log = logging.getLogger(__name__)


def _get_driver(name: str, *, drop: bool = False) -> ModuleType:
    """Return the refinement driver module for the given name.

    drop: the run is on a DROP cavity, so Psi4 must be able to import MOIST.
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
    if name == "orbmol":
        from tautomator import orbmol_runner

        return orbmol_runner
    raise ValueError(f"Unknown driver: {name!r}. Must be 'psi4', 'pyscf' or 'orbmol'.")


def refine(
    ensemble: Ensemble,
    config: RefinementConfig,
    *,
    driver: str,
    solvent: str = "water",
    cavity: str | None = None,
    threads: int = 1,
    memory_gb: float = DEFAULT_MEMORY_GB,
) -> Ensemble:
    """Refine all conformers at the refinement level.

    ``config`` carries the level of theory and how solvation and RRHO enter
    (see tautomator.config).  ``driver`` is the DFT backend: it optimizes unless an
    OrbMol model does, and runs the solvation single point either way.

    Conformers whose optimizer ran but did not fully converge are kept (with
    refinement_converged=False) since the last-step geometry is usually good
    enough for conformer screening.  Conformers that raise (e.g. SCF
    non-convergence) are dropped with a warning and a record.

    Modifies the ensemble in-place and returns it.
    """
    # The run's cavity, even when this stage is gas-phase: scoring will need MOIST
    # then, and a missing one should stop the run before the refinement, not after it.
    dft = _get_driver(driver, drop=cavity in DROP_CAVITIES)
    engine = dft if config.engine == driver else _get_driver(config.engine)
    if config.engine == "orbmol":
        # Outside the per-conformer handlers below, which would record a model
        # that cannot be loaded as each conformer's own optimization failure.
        engine.preload(config.method)
    # xtb's RRHO in the run's solvent, or in the gas phase for a gas-phase run.
    xtb_solvent = solvent if config.rrho == "xtb" and cavity is not None else None

    for cs in ensemble.charge_states.values():
        log.info(f"Refining charge state q={cs.charge}...")
        _optimize_charge_state(cs, config, engine, solvent, cavity, threads, memory_gb)

        # Re-file conformers whose proton moved during the optimization before
        # deduplicating, so a migrated conformer is compared against the
        # microstate it now belongs to rather than the one it started in.
        report = repair_migrated_conformers(cs, stage="refinement")
        if (summary := report.summary()) is not None:
            log.info(f"  q={cs.charge}: {summary}")

        # Fold before deduplicating.  A shallow off-symmetric well can lie near
        # enough to its symmetric shape to deduplicate with it, and
        # deduplication would keep the well, the lower of the two, and count
        # it as chiral.
        if config.nudge and (folded := _fold_shallow_wells(cs, config.collapse_h_tops)):
            log.info(
                f"  q={cs.charge}: {folded} shallow off-symmetric well(s) folded into the "
                "symmetric conformer beside them"
            )

        # Deduplicate before the Hessians, not after. Optimization routinely
        # relaxes distinct sampled conformers onto the same minimum, and onto
        # mirror-image pairs; each of those would otherwise cost a second
        # Hessian and then contribute twice to the partition function.
        # With collapsed tops a methyl's hilltop joins its minimum here.  Each
        # group's representative is picked on the optimized energy alone, since
        # the RRHO terms are still sampling's (ensemble.deduplicate_conformers).
        before, after = deduplicate_charge_state(
            cs, collapse_tops=config.collapse_h_tops, stale_rrho=True
        )
        if after < before:
            log.info(f"  q={cs.charge}: {before} conformer(s) -> {after} distinct state(s)")

        _add_rrho(cs, config, engine, xtb_solvent, solvent, cavity, threads, memory_gb)
        if config.solvation == "single_point":
            _apply_gas_prefilter(cs, config)
            _add_solvation(cs, config, dft, driver, solvent, cavity, threads, memory_gb)

        filter_charge_state_by_energy(cs, config.ewin)
        n_conf = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: {len(cs.microstates)} microstate(s), "
            f"{n_conf} conformer(s) after filtering"
        )

    return ensemble


def _multiplicity(ms: Microstate, coords: np.ndarray, collapse_tops: bool) -> float:
    """The multiplicity of one geometry of ``ms``, as deduplication will count it."""
    return float(
        conformer_multiplicity(
            np.asarray(coords, dtype=np.float64)[None],
            list(ms.conformers[0].geometry.symbols),
            includes_enantiomer=ms.includes_enantiomer,
            collapse_tops=collapse_tops,
        )[0]
    )


def _fold_shallow_wells(cs: ChargeState, collapse_tops: bool) -> int:
    """Drop each conformer that `nudge.fold` joins to a symmetric one; return how many.

    Runs before deduplication, so the symmetries are taken from the optimized
    geometries here; the multiplicities on the conformers are still sampling's.
    They only rank the conformers, so the duplicates still in the list do no
    harm.  Compares the energies the optimizations minimized: electronic plus
    solvation where the geometry was optimized in solvent, electronic alone
    otherwise.  ``collapse_tops`` as deduplication will use it, so that a
    symmetric conformer whose methyl stopped turned off the mirror still counts
    as symmetric here.  The symmetric conformer kept gets a `Well` for the
    lowest one folded into it, with that one's geometry, unless settle already
    gave it one.
    """
    folded = 0
    for ms in cs.microstates:
        # Every optimized conformer has an energy; one without is left to the
        # code after this to report, and its microstate goes unfolded.
        if len(ms.conformers) < 2 or any(c.electronic_energy is None for c in ms.conformers):
            continue
        coords = [np.asarray(c.geometry.coords, dtype=np.float64) for c in ms.conformers]
        symbols = list(ms.conformers[0].geometry.symbols)
        energies = [
            (cast(float, c.electronic_energy) + (c.solvation_energy or 0.0)) * HARTREE_TO_KCAL
            for c in ms.conformers
        ]
        symmetries = [symmetry_number(x, symbols, collapse_tops=collapse_tops) for x in coords]
        into = fold_into(coords, energies, symmetries, symbols, collapse_tops)
        for j in sorted(set(into.values())):
            host = ms.conformers[j]
            if host.well is None:
                i = min((i for i, k in into.items() if k == j), key=lambda i: energies[i])
                host.well = Well(
                    depth=energies[j] - energies[i],
                    symmetric_multiplicity=_multiplicity(ms, coords[j], collapse_tops),
                    broken_multiplicity=_multiplicity(ms, coords[i], collapse_tops),
                    source="fold",
                    dropped=ms.conformers[i].geometry,
                )
        folded += len(into)
        ms.conformers = [c for k, c in enumerate(ms.conformers) if k not in into]
    return folded


def _optimize_charge_state(
    cs: ChargeState,
    config: RefinementConfig,
    engine: ModuleType,
    solvent: str,
    cavity: str | None,
    threads: int,
    memory_gb: float,
) -> None:
    """Optimize every conformer, setting electronic (and solvation, if minimized in solvent)."""
    in_solvent = config.solvation == "optimize"
    # The solvent sampling ran in, where the conformers are GFN2 minima.
    sampling_solvent = solvent if cavity is not None else None

    def at_level(g: Geometry) -> tuple[Geometry, float, bool]:
        """Optimize at the refinement level: (geometry, energy in Hartree, converged)."""
        relaxed, energy, ok = engine.optimize(
            g,
            cs.charge,
            config.method,
            config.basis,
            cavity if in_solvent else None,
            solvent if in_solvent else None,
            pcm_hydrogen_radius=config.pcm_hydrogen_radius,
            threads=threads,
            memory_gb=memory_gb,
        )
        return relaxed, float(energy), bool(ok)

    for ms in cs.microstates:
        surviving = []
        for conf in ms.conformers:
            # Which half of the try raised.  The gas-phase point runs only to
            # decompose a solvated energy, so a failure there means the
            # optimization succeeded and the decomposition did not -- a
            # different thing to chase, and worth separating in the record.
            fail_reason: ExclusionReason = "optimization_failed"
            try:
                start, done = conf.geometry, None
                if config.nudge:
                    try:
                        start, done = nudge(
                            conf.geometry, cs.charge, solvent=sampling_solvent, threads=threads
                        )
                    except Exception as e:
                        # As for a failed nudged optimization below: never a reason
                        # to lose the conformer.
                        log.warning(
                            f"  The nudge failed for a conformer in microstate "
                            f"{ms.tautomer_id[:8]} ({type(e).__name__}: {e}); optimizing it "
                            "as sampled"
                        )
                        start, done = conf.geometry, None
                    if done is not None:
                        more = ""
                        if done.later:
                            listed = ", ".join(f"{f:.0f}" for f in done.later)
                            plural = "s" if len(done.later) > 1 else ""
                            more = f" and its {listed} cm-1 mode{plural}"
                        log.info(
                            f"  Nudged a symmetric conformer in microstate {ms.tautomer_id[:8]} "
                            f"along its {done.frequency:.0f} cm-1 GFN2 mode{more}"
                        )
                try:
                    opt_geom, opt_energy, converged = at_level(start)
                except Exception as e:
                    if done is None:
                        raise
                    # A nudge is an improvement, never a reason to lose the conformer.
                    log.warning(
                        f"  The nudged optimization failed for a conformer in microstate "
                        f"{ms.tautomer_id[:8]} ({type(e).__name__}: {e}); optimizing it as sampled"
                    )
                    done = None
                    opt_geom, opt_energy, converged = at_level(conf.geometry)
                # An unconverged result's energy says nothing about the depth of
                # its well: it may still be rolling downhill, so its symmetric
                # shape is not compared with it.
                if done is not None and converged:
                    try:
                        settled = settle(done, opt_geom, opt_energy, at_level)
                    except Exception as e:
                        log.warning(
                            f"  Settling the nudged conformer in microstate "
                            f"{ms.tautomer_id[:8]} failed ({type(e).__name__}: {e}); "
                            "keeping the nudged result"
                        )
                        settled = None
                    if settled is not None:
                        kept = "kept the symmetric shape" if settled.replaces else "kept the nudge"
                        side = "higher" if settled.depth >= 0 else "lower"
                        log.info(
                            f"  The nudged conformer in microstate {ms.tautomer_id[:8]} lost its "
                            f"symmetry; the symmetric shape is {abs(settled.depth):.3f} kcal/mol "
                            f"{side}, {kept}"
                        )
                        collapse = config.collapse_h_tops
                        conf.well = Well(
                            depth=settled.depth,
                            symmetric_multiplicity=_multiplicity(
                                ms, settled.geometry.coords, collapse
                            ),
                            broken_multiplicity=_multiplicity(ms, opt_geom.coords, collapse),
                            source="settle",
                            dropped=opt_geom if settled.replaces else settled.geometry,
                        )
                        if settled.replaces:
                            opt_geom, opt_energy = settled.geometry, settled.energy
                            converged = True  # settle keeps only a converged shape
                conf.geometry = opt_geom
                conf.refinement_converged = converged
                if not converged:
                    log.warning(
                        f"  Geometry optimization did not fully converge for conformer "
                        f"in microstate {ms.tautomer_id[:8]}; keeping last-step geometry"
                    )

                if in_solvent:
                    # opt_energy includes solvation — decompose
                    fail_reason = "gas_phase_sp_failed"
                    gas_energy = engine.single_point(
                        opt_geom,
                        cs.charge,
                        config.method,
                        config.basis,
                        threads=threads,
                        memory_gb=memory_gb,
                    )
                    conf.electronic_energy = gas_energy
                    conf.solvation_energy = opt_energy - gas_energy
                else:
                    conf.electronic_energy = opt_energy
                    conf.solvation_energy = None  # set later if solvation == single_point

                surviving.append(conf)
            except Exception as e:
                # Geometry non-convergence does NOT arrive here: the driver
                # catches it and returns refinement_converged=False, keeping the
                # last-step geometry.  That conformer stays in the ensemble, and
                # rightly so -- an unminimized geometry sits *above* its true
                # minimum, so it is under-weighted, never dominant.  What reaches
                # this handler is SCF non-convergence and genuine faults, where
                # there is no energy at all.
                log.warning(
                    f"  Refinement failed for conformer in microstate {ms.tautomer_id[:8]}: {e}"
                )
                ms.excluded_conformers.append(
                    ExcludedConformer(
                        geometry=conf.geometry,
                        stage="refinement",
                        reason=fail_reason,
                        detail=f"{type(e).__name__}: {e}",
                        multiplicity=conf.multiplicity,
                        refinement_converged=conf.refinement_converged,
                    )
                )
        ms.conformers = surviving


def _add_rrho(
    cs: ChargeState,
    config: RefinementConfig,
    engine: ModuleType,
    xtb_solvent: str | None,
    solvent: str,
    cavity: str | None,
    threads: int,
    memory_gb: float,
) -> None:
    """Recompute the quasi-RRHO correction on the refined geometry."""
    in_solvent = config.solvation == "optimize"
    for ms in cs.microstates:
        surviving = []
        for conf in ms.conformers:
            try:
                if config.rrho == "xtb":
                    freqs = xtb_runner.frequencies(
                        conf.geometry,
                        cs.charge,
                        solvent=xtb_solvent,
                        biased=True,
                        threads=threads,
                    )
                else:  # "refinement": a Hessian from whatever optimized the geometry
                    freqs = engine.frequencies(
                        conf.geometry,
                        cs.charge,
                        config.method,
                        config.basis,
                        cavity if in_solvent else None,
                        solvent if in_solvent else None,
                        pcm_hydrogen_radius=config.pcm_hydrogen_radius,
                        threads=threads,
                        memory_gb=memory_gb,
                    )
                conf.rrho_correction = quasi_rrho_free_energy(freqs, floor=config.rrho_floor)
                surviving.append(conf)
            except Exception as e:
                # Previously the conformer was KEPT with rrho_correction unset.
                # free_energy sums non-None components, so it became electronic +
                # solvation while every sibling included RRHO -- and RRHO runs
                # 11-114 kcal/mol (median 76) in this batch, so the conformer
                # landed far below the ensemble's real ~6 kcal/mol spread, took
                # essentially all the Boltzmann weight, and the energy window
                # evicted everything else.  A missing component must never be
                # silently comparable to a complete one.
                log.warning(f"  RRHO failed for conformer in microstate {ms.tautomer_id[:8]}: {e}")
                ms.excluded_conformers.append(
                    ExcludedConformer(
                        geometry=conf.geometry,
                        stage="refinement",
                        reason="rrho_failed",
                        detail=f"{type(e).__name__}: {e}",
                        multiplicity=conf.multiplicity,
                        electronic_energy=conf.electronic_energy,
                        solvation_energy=conf.solvation_energy,
                        refinement_converged=conf.refinement_converged,
                    )
                )
        ms.conformers = surviving


def _apply_gas_prefilter(cs: ChargeState, config: RefinementConfig) -> None:
    """Drop conformers too high in gas phase to be worth a solvation single point.

    A cost control, not physics: the solvation single point is the dominant
    refinement cost for an OrbMol refinement, and most conformers cannot reach
    the window once solvation is added.  Unlike the energy-window filters, which
    drop conformers silently, these are recorded in ``excluded_conformers`` with
    reason "gas_prefilter", so a later analysis can check how close the nearest
    dropped conformer came.  Conformers inside a 6 kcal/mol *solvated* window sat
    up to 9.1 kcal/mol above the gas-phase minimum in testing, so the threshold
    needs real headroom over ``ewin`` (config validation warns below ewin + 4).
    """
    window = config.solvation_prefilter_ewin
    if window is None:
        return
    entries = [(ms, c) for ms in cs.microstates for c in ms.conformers]
    if not entries:
        return
    # Gas-phase free energy: what the window will rank on, minus solvation.
    e_min = min(_effective_energy(ms, c) for ms, c in entries)
    cutoff = window / HARTREE_TO_KCAL
    dropped = 0
    for ms in cs.microstates:
        keep = []
        for conf in ms.conformers:
            above = _effective_energy(ms, conf) - e_min
            if above <= cutoff:
                keep.append(conf)
                continue
            dropped += 1
            ms.excluded_conformers.append(
                ExcludedConformer(
                    geometry=conf.geometry,
                    stage="refinement",
                    reason="gas_prefilter",
                    detail=(
                        f"{above * HARTREE_TO_KCAL:.2f} kcal/mol above the gas-phase "
                        f"minimum (> solvation_prefilter_ewin = {window})"
                    ),
                    multiplicity=conf.multiplicity,
                    electronic_energy=conf.electronic_energy,
                    rrho_correction=conf.rrho_correction,
                    refinement_converged=conf.refinement_converged,
                )
            )
        ms.conformers = keep
    if dropped:
        kept = sum(len(ms.conformers) for ms in cs.microstates)
        log.info(
            f"  q={cs.charge}: gas-phase pre-filter at {window} kcal/mol kept "
            f"{kept} of {kept + dropped} conformer(s); the rest get no solvation "
            f"single point (recorded as gas_prefilter)"
        )


def _add_solvation(
    cs: ChargeState,
    config: RefinementConfig,
    dft: ModuleType,
    driver: str,
    solvent: str,
    cavity: str | None,
    threads: int,
    memory_gb: float,
) -> None:
    """Add a solvation energy from one solvated single point on the refined geometry.

    For DFT refinement that is one job at the refinement level: the optimizer
    already produced the gas-phase energy.  OrbMol has no implicit solvent, so
    its solvation energy comes from a cheap DFT pair (solvated and gas) at
    ``solvation_method``/``solvation_basis`` on the DFT driver.
    """
    orbmol = config.engine == "orbmol"
    method = config.solvation_method if orbmol else config.method
    basis = config.solvation_basis if orbmol else config.basis
    for ms in cs.microstates:
        surviving = []
        for conf in ms.conformers:
            try:
                solvated = dft.single_point(
                    conf.geometry,
                    cs.charge,
                    method,
                    basis,
                    cavity,
                    solvent,
                    pcm_hydrogen_radius=config.pcm_hydrogen_radius,
                    threads=threads,
                    memory_gb=memory_gb,
                )
                gas = (
                    dft.single_point(
                        conf.geometry,
                        cs.charge,
                        method,
                        basis,
                        threads=threads,
                        memory_gb=memory_gb,
                    )
                    if orbmol
                    else conf.electronic_energy
                )
                assert gas is not None
                conf.solvation_energy = solvated - gas
                surviving.append(conf)
            except Exception as e:
                # Without this term the conformer's free energy is not comparable
                # to its siblings' (see the RRHO note above), so it cannot stay.
                log.warning(
                    f"  Solvation single point failed for conformer in microstate "
                    f"{ms.tautomer_id[:8]}: {e}"
                )
                ms.excluded_conformers.append(
                    ExcludedConformer(
                        geometry=conf.geometry,
                        stage="refinement",
                        reason="solvation_failed",
                        detail=f"{type(e).__name__}: {e} (level {method}/{basis} on {driver})",
                        multiplicity=conf.multiplicity,
                        electronic_energy=conf.electronic_energy,
                        rrho_correction=conf.rrho_correction,
                        refinement_converged=conf.refinement_converged,
                    )
                )
        ms.conformers = surviving
