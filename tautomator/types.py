from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem import GetPeriodicTable

_PT = GetPeriodicTable()


@dataclass
class Geometry:
    """Molecular geometry: atom symbols and Cartesian coordinates."""

    symbols: tuple[str, ...]
    coords: NDArray[np.float64]  # shape (n_atoms, 3), Angstrom

    def __post_init__(self) -> None:
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError(
                f"symbols length ({len(self.symbols)}) != coords rows ({self.coords.shape[0]})"
            )
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n, 3), got {self.coords.shape}")
        # Normalize element symbols to canonical "Xy" case. Psi4's save_xyz_file
        # writes uppercase ("CL"), while RDKit's PeriodicTable lookup is
        # case-sensitive — without this, n_electrons/multiplicity would raise
        # cryptically on any geometry that round-tripped through Psi4.
        self.symbols = tuple(s.capitalize() for s in self.symbols)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def heavy_atom_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.symbols) if s != "H"]

    @property
    def hydrogen_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.symbols) if s == "H"]

    def n_electrons(self, charge: int) -> int:
        """Total electron count for the given molecular charge."""
        return int(sum(_PT.GetAtomicNumber(s) for s in self.symbols)) - charge

    def multiplicity(self, charge: int) -> int:
        """Spin multiplicity (2S+1). Assumes lowest multiplicity (singlet or doublet)."""
        return 1 + self.n_electrons(charge) % 2


@dataclass
class Conformer:
    """A molecular conformer with decomposed energy components.

    Energy components are populated at different stages of the pipeline:
      - electronic_energy: gas-phase electronic energy (Hartree).
        Set at sampling (xTB) and replaced at refinement/scoring (DFT).
      - solvation_energy: solvation free energy contribution (Hartree).
        At sampling, computed as (CREST ALPB total) - (gas-phase xTB SP).
        At refinement/scoring, from implicit solvent (IEF-PCM on the run's cavity).
      - rrho_correction: quasi-RRHO vibrational free energy (Hartree).
        Set at sampling (xTB --hess on the xTB geometry) and recomputed at
        refinement on the DFT geometry per the configured rrho_method.
      - multiplicity: how many physical states this conformer stands for,
        divided by its own rotational symmetry number. Recomputed after every
        deduplication pass, since both terms depend on the geometry.

    The free_energy property sums all non-None components for Boltzmann
    weighting and partition function calculations.
    """

    geometry: Geometry
    electronic_energy: float | None = None  # E_elec (Hartree)
    solvation_energy: float | None = None  # ΔG_solv (Hartree)
    rrho_correction: float | None = None  # G_RRHO (Hartree)
    weight: float | None = None  # Boltzmann weight, normalized across the charge state
    refinement_converged: bool | None = None  # None before refinement; bool after
    multiplicity: float = 1.0  # n_states / sigma; see tautomator.conformer_symmetry

    @property
    def free_energy(self) -> float:
        """Total free energy: sum of all non-None energy components."""
        components = [
            self.electronic_energy,
            self.solvation_energy,
            self.rrho_correction,
        ]
        active = [c for c in components if c is not None]
        if not active:
            raise ValueError("Conformer has no energy components set")
        return sum(active)


# Where a conformer was lost, and to what.  Spelled out as literals so that a
# typo is a type error rather than a category that silently never matches when
# someone groups a finished run by reason.
ExclusionStage = Literal["sampling", "refinement", "scoring"]
ExclusionReason = Literal[
    "optimization_failed",  # the optimizer itself raised (SCF non-convergence, faults)
    "gas_phase_sp_failed",  # optimization succeeded; the gas-phase decomposition point did not
    "rrho_failed",
    "solvation_failed",  # optimization and RRHO succeeded; the solvation single point did not
    "scoring_failed",
    # Not a failure: the refinement solvation single point was skipped because the
    # conformer was already far above the charge state's minimum in gas phase (see
    # [refinement] solvation_prefilter_ewin).  Recorded because it is a cost
    # decision rather than a physical one; the energy-window filters themselves
    # drop conformers without a record.
    "gas_prefilter",
    # The five below differ in kind from everything above: the energy computed
    # fine, and it is the *label* that is wrong.  See tautomator.protomer_geometry.
    "fragmented",  # the heavy-atom framework came apart; the energy is a complex's
    "proton_detached",  # a hydrogen left the molecule; the energy is a fragment's
    "no_matching_microstate",  # minimised to a species the enumerator never produced
    "ambiguous_microstate",  # several microstates match; they differ only in stereochemistry
    "incompatible_atom_order",  # cannot be pooled with its microstate; orders disagree
]


@dataclass
class ExcludedConformer:
    """A conformer removed from the ensemble because it has no usable energy.

    Kept, rather than dropped, so that a run records what it could not compute:
    the geometry is inspectable and the reason is explicit, instead of the
    conformer simply being absent with only a log line on someone's terminal.

    Deliberately *not* a `Conformer`: it has no ``free_energy``, so it cannot be
    summed into a partition function by accident. That matters because the
    failures collected here are ones where a partially-computed energy would
    look entirely plausible -- a conformer missing its RRHO term sits ~76
    kcal/mol below its siblings and would capture the whole charge state.

    ``multiplicity`` is the number of physical states lost, not one. Refinement
    deduplicates before computing Hessians, so a representative may already
    stand for several collapsed structures, and excluding it discards all of
    them. The non-representatives were discarded at deduplication, so no
    substitute can be promoted; recording the multiplicity at least makes the
    size of the loss visible rather than understating it. Falling back to a
    surviving duplicate would need deduplication to retain the whole group --
    deferred, since no Hessian failed across 941 conformers in the first batch.

    Whatever energy components were computed before the failure are retained
    for diagnosis.
    """

    geometry: Geometry
    stage: ExclusionStage
    reason: ExclusionReason
    detail: str = ""  # exception type and message
    multiplicity: float = 1.0  # physical states lost with this conformer
    electronic_energy: float | None = None
    solvation_energy: float | None = None
    rrho_correction: float | None = None
    refinement_converged: bool | None = None


@dataclass
class Microstate:
    """A tautomeric/protonation microstate with its conformer ensemble."""

    tautomer_id: str  # canonical SMILES (approach 1) or fingerprint hash (approach 2)
    conformers: list[Conformer]
    smiles: str | None = None  # explicit-H canonical SMILES (approach 1), None in approach 2
    includes_enantiomer: bool = False  # True if this represents a collapsed enantiomeric pair
    # Conformers with no usable energy, kept for the record.  Structurally
    # separate from `conformers` so that nothing weighting or filtering the
    # ensemble can reach them: they are excluded by construction, not by every
    # call site remembering to check a flag.
    excluded_conformers: list[ExcludedConformer] = field(default_factory=list)


@dataclass
class ChargeState:
    """All microstates at a given molecular charge."""

    charge: int
    microstates: list[Microstate]


@dataclass
class Ensemble:
    """Complete ensemble for a molecule across all charge states."""

    input_smiles: str
    charge_states: dict[int, ChargeState] = field(default_factory=dict)
    settings: dict[str, object] = field(default_factory=dict)
