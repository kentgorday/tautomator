"""Configuration for the three-stage pKa workflow.

Reads a single TOML file that configures:
  [molecule]   — input SMILES, charge range, solvent
  [sampling]   — CREST-based conformer/tautomer/protonation sampling
  [refinement] — geometry optimization + solvation + quasi-RRHO
  [scoring]    — single-point energy with expensive DFT
  [compute]    — DFT driver (psi4/pyscf), threads, output directory

The config says which physics goes into each energy term; the runners decide how
to get it (analytic vs finite-difference Hessians, fallback functionals for
second derivatives, grids, cavity construction) and record what they did.

Refinement produces three terms, each with its own setting:
  electronic — ``method``/``basis``.  An OrbMol model name here (see
      tautomator.orbmol_runner.MODELS) makes OrbMol the optimizer; anything else is
      DFT on the [compute] driver.
  solvation  — ``solvation``: "optimize" (minimize in implicit solvent),
      "single_point" (gas minimization, then one solvated single point) or
      "none" (gas phase; the filter then ranks on gas energies, which reranks
      badly against solvated scoring).
  RRHO       — ``rrho``: "xtb" (GFN2 biased Hessian in ALPB) or "refinement"
      (a Hessian from whatever optimized the geometry).

Scoring always recomputes electronic + solvation and carries RRHO over.
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Per-QM-job memory ceiling, shared by both backends so a given config means
# the same thing on either one.  Their native defaults disagree by 8x (Psi4
# 500 MiB, PySCF 4 GB), which is an invisible backend difference in a project
# that cross-checks the two.  4 GB matches PySCF's own default, so only Psi4
# moves; measured wall time on def2-QZVPPD/wB97M-V is flat from 4 GB up, so
# there is nothing to buy by going higher.
DEFAULT_MEMORY_GB = 4.0

# Coulomb/exchange fitting basis for density-fitted SCF.  Weigend's universal JK
# set is what Psi4 blends by default for the def2 family and for vDZP -- its
# output reports "SCF Algorithm Type is DF" and "Blend: DEF2-UNIVERSAL-JKFIT" --
# so PySCF is told to use it too rather than silently differ.  It was optimized
# for def2-QZVPP and is the one JK set for every def2 basis; there is no
# diffuse-augmented version for def2-QZVPPD.
#
# For a Dunning or Pople basis Psi4 picks a JKFIT set of its own (cc-pVXZ-JKFIT,
# aug-cc-pVXZ-JKFIT, heavy-aug-cc-pVDZ-JKFIT) and is left to.  With an augmented
# basis it fits small anions better with its own set than with this one: at
# wB97M-V/aug-cc-pVTZ the deprotonation energy of water is off by 0.013 kcal/mol
# against 0.115, and at 6-31+G* that of ammonia by 0.003 against 0.073.  Plain
# cc-pVTZ-JKFIT is the worse of the two (HF 0.154 against 0.001), but a basis
# without diffuse functions is no basis for an anion.  PySCF uses this set for
# every basis and does not need the augmented ones (water 0.010 at aug-cc-pVTZ),
# so for those bases the two backends fit differently.
#
# Accuracy, measured against conventional four-centre integrals at
# wB97M-V/def2-QZVPPD: a few hundredths of a kcal/mol on absolute energies
# (H2O -0.022, OH- -0.014, formic acid -0.023, formate -0.011), and the errors
# largely cancel in a difference -- formic acid's deprotonation energy moves
# 0.012 kcal/mol, which is 0.009 pKa.
AUXBASIS = "def2-universal-jkfit"

# One more s primitive for AUXBASIS on each atom of these elements (exponent).
#
# The set was optimized on neutral monohydrides against the exchange energy, and
# for these elements its s functions stop at an exponent of 0.24-0.29 where
# those of C-F, Si, Ge and Te reach 0.10-0.16.  An anion whose charge sits on
# one of them with only hydrogens around it has a tail the set cannot fit.  At
# wB97M-V/def2-QZVPPD in the gas phase PySCF is off by -0.29 kcal/mol for Cl-,
# -0.38 for Br-, -0.54 for I-, -0.25 for HS-, -0.40 for HSe- and -0.23 for
# AsH2-, about half that in water, and it cancels in nothing useful.  The acid
# is fitted well (HCl -0.01), so a pKa keeps the error; the solvent halves it,
# so a solvation energy keeps half; and a neighbour lends its own fitting
# functions, so it moves with the geometry -- Cl- with one water hydrogen-bonded
# to it is off by -0.06.  Nor do the backends agree on it: they fit long-range
# exchange differently, and Psi4's errors for those six are +0.17 to +0.35.
#
# The added exponent is the element's smallest s exponent in the set divided by
# 2.5, where a scan found a broad optimum (anything from /2 to /3 leaves under
# 0.03 kcal/mol in PySCF).  With it PySCF is within 0.012 kcal/mol of
# conventional integrals on those six and, over 20 acid/base pairs containing
# these elements, within 0.012 on the deprotonation energy in gas or water,
# against 0.53 without.  Psi4 is within 0.054 on those six: what it has left is
# on As, Se and Br, whose p functions stop short as well (0.19-0.20 against 0.09
# for P, S, Cl and I), and a p primitive there would bring it to 0.015.
#
# In PySCF no other element with a def2 diffuse basis needs one.  Sb, Bi, Te,
# Po, At, Ge, Sn and Pb already reach 0.11 or below, and their hydride anions'
# deprotonation energies are fitted to 0.004 kcal/mol.  Psi4 has the same
# short-p remainder on F, Ge and Te, whose p functions also stop at 0.18-0.20:
# it is off by +0.06 kcal/mol on F-, TeH- and GeH3- (PySCF -0.01 to -0.02) and
# by +0.09 on the deprotonation energies of HF and H2Te.  That is under 0.1
# kcal/mol, but more than Psi4 has left on the six above, and a p primitive
# would be the fix there too.  Hydrogen and lithium are left alone although a
# bare H- is off by -1.8 kcal/mol and Li- by -0.31 in the gas phase: hydridic
# anions (BH4-, AlH4-) are fine, and nothing here computes a hydride or lithide.
AUX_DIFFUSE_S: dict[str, float] = {
    "P": 0.106,
    "S": 0.112,
    "Cl": 0.116,
    "As": 0.100,
    "Se": 0.106,
    "Br": 0.108,
    "I": 0.096,
}

log = logging.getLogger(__name__)

Solvation = Literal["none", "optimize", "single_point"]
Rrho = Literal["xtb", "refinement"]

DRIVERS = ("psi4", "pyscf")

# "3c" composite methods are a functional *and* the basis they were
# parameterized with; pairing one with another basis is a different method.
COMPOSITE_BASIS: dict[str, str] = {
    "wb97x-3c": "vDZP",
    "r2scan-3c": "def2-mTZVPP",
    "b97-3c": "def2-mTZVP",
    "pbeh-3c": "def2-mSVP",
    "hf-3c": "MINIX",
}
# PySCF implements only wB97X-3c of those (pyscf.scf.dispersion: "Only wb97x-3c
# is supported for now"); Psi4 has all five.
PYSCF_COMPOSITES = {"wb97x-3c"}

# The analytical PCM models: user-facing name ([scoring] solvent_model) -> the
# name both drivers' solvers take.  Both drivers run the same switching/Gaussian
# PCM: PySCF its own pyscf.solvent.pcm, Psi4 psi4-swig-pcm, which reproduces it
# (same surface, radii and discretization; Hartree-Fock solvation energies agree
# to 1e-9 Eh and gradients to 5e-8 Eh/bohr) and adds analytical gradients to Psi4.
PCM_METHODS: dict[str, str] = {
    "PCM": "IEF-PCM",
    "IEFPCM": "IEF-PCM",
    "CPCM": "C-PCM",
    "COSMO": "COSMO",
}

# MOIST's DROP cavities (tautomator.pyscf_moist), for the same PCM models: a
# solvent model "<model>/<cavity>", e.g. "IEFPCM/SvdW-DROP".  SvdW-DROP is a smooth
# solvent-excluded-like surface over MOIST's CPCM radii, rho-DROP the electron
# isodensity surface (rho_iso = 1e-3 e/bohr^3), which follows the density through
# the SCF.  PySCF driver only, in the moist pixi environment.  Normalized name
# (upper case, no hyphens) -> MOIST's name.
DROP_CAVITIES: dict[str, str] = {
    "SVDWDROP": "SvdW-DROP",
    "RHODROP": "rho-DROP",
    "ISODENSITYDROP": "rho-DROP",
}


def drop_solvent_model(model: str) -> tuple[str, str] | None:
    """(PCM method, MOIST cavity name) of a "<model>/<cavity>" solvent model, else None."""
    pcm, sep, cavity = model.upper().replace("-", "").partition("/")
    if sep and pcm in PCM_METHODS and cavity in DROP_CAVITIES:
        return PCM_METHODS[pcm], DROP_CAVITIES[cavity]
    return None


# The switching function both drivers' PCM surfaces use (PySCF's
# surface_discretization_method, psi4-swig-pcm's PCMSettings.switching).  ISWIG
# (PySCF 2.13 and later) fades buried points out with an erf over each Gaussian's
# width instead of SWIG's polynomial band.  It keeps ~11% more points (9-14% over
# the pKa set) for the same area and moved solvation energies by ~0.02 kcal/mol;
# SCF time rose 0-3% and gradient time 4-9% (wB97X-3c/vDZP, wB97M-V/def2-TZVPPD).
# FastPCM's stored integrals grow with the points too: a molecule that just fit in
# memory_gb on SWIG can spill on ISWIG, and its SCF then runs at stock PCM speed
# (aspirin anion, def2-TZVPPD, memory_gb = 8: 3.7 GB fit, 4.2 GB did not; 1.8x).
PCM_SWITCHING = "ISWIG"

# Implicit-solvent models, by the driver that can run them.  One model is used
# for every solvation energy in a run, so the refinement filter ranks on the
# same physics scoring does.  Both drivers default to IEF-PCM.  SS(V)PE is not
# offered: it symmetrizes the system matrix, and on this surface that moved the
# solvation energy of acetate and methylammonium by 2.7 and 4.2 kcal/mol and did
# not converge with the surface grid (the ill behaviour Lange & Herbert
# describe, Chem. Phys. Lett. 509, 77).
_SOLVENT_MODELS: dict[str, set[str]] = {
    "pyscf": {
        *PCM_METHODS,
        "SMD",
        "DDPCM",
        "DDCOSMO",
        *(f"{pcm}/{cavity}" for pcm in PCM_METHODS for cavity in DROP_CAVITIES),
    },
    "psi4": set(PCM_METHODS),
}

# Dielectric constants for the PCM models, by lower-case solvent name; both
# drivers take them from here.  The name also goes to xtb's and CREST's --alpb
# in sampling and RRHO, through `alpb_name`: a solvent added here needs a name
# xtb knows, or an entry in _ALPB_NAMES.  Octanol and hexadecane carry the values of
# the Minnesota Solvent Descriptor Database (as SMD uses them), so solvation free
# energies in them compare with MNSol's.
#
# "octanol" is dry 1-octanol and "woctanol" water-saturated 1-octanol, the
# phase octanol-water partition coefficients are measured in, and so the one
# logP and most of MNSol's octanol free energies refer to.  Both use dry
# octanol's dielectric constant: no measured static permittivity of the wet
# phase was found, and xtb's ALPB "woctanol" uses 9.8629 as well (its fitted
# parameters are what differ).  Wet octanol differs in its electron density
# (_SOLVENT_LIQUID); see docs/nonelectrostatic.md.
SOLVENT_DIELECTRIC: dict[str, float] = {
    "water": 78.39,
    "methanol": 32.7,
    "ethanol": 24.55,
    "dmso": 46.7,
    "acetonitrile": 37.5,
    "thf": 7.58,
    "toluene": 2.38,
    "chloroform": 4.81,
    "dichloromethane": 8.93,
    "hexane": 1.88,
    "acetone": 20.7,
    "octanol": 9.8629,
    "woctanol": 9.8629,
    "hexadecane": 2.0402,
}


def solvent_dielectric(solvent: str) -> float:
    """Look up the dielectric constant for a solvent name."""
    key = solvent.lower()
    if key not in SOLVENT_DIELECTRIC:
        raise ValueError(
            f"Unknown solvent {solvent!r} for dielectric lookup. "
            f"Known solvents: {', '.join(sorted(SOLVENT_DIELECTRIC))}"
        )
    return SOLVENT_DIELECTRIC[key]


# xtb's ALPB parameter sets, by the names above where xtb's differ.  xtb stops
# on a name it has no parameters for ("solvent: 'dichloromethane' is not
# parametrized").
_ALPB_NAMES: dict[str, str] = {"dichloromethane": "ch2cl2"}


def alpb_name(solvent: str) -> str:
    """The name xtb and CREST know a solvent by, for --alpb."""
    key = solvent.lower()
    return _ALPB_NAMES.get(key, key)


# The liquid each solvent is, for the dispersion term of the nonelectrostatic
# correction, which needs its mean electron density: mass density at 298.15 K
# and ambient pressure (kg/m^3), and its components as (mass fraction, molar
# mass in g/mol with IUPAC standard atomic weights, electrons per molecule).
# Only the solvents the correction is fitted in so far.
#   water       IAPWS-95 at 0.1 MPa.
#   octanol     median of the pure-liquid values at 298.15 K in eight studies
#               in NIST TRC's ThermoML archive (821.5-822.3): DOIs
#               10.1021/je050200w, 10.1021/je060353z, 10.1021/je100170v,
#               10.1021/acs.jced.5b00031, 10.1016/j.jct.2008.02.005,
#               10.1016/j.jct.2013.04.021, 10.1016/j.jct.2013.10.018 and
#               10.1016/j.jct.2015.06.029.
#   woctanol    density (831.7 +- 1.7) and water mass fraction (48.91 +- 0.13
#               mg/g, Karl Fischer) of water-saturated octanol at 298.15 K, from
#               one study: J. Chem. Eng. Data 57, 2221 (2012), doi:10.1021/je3001427.
#   hexadecane  Outcalt, Laesecke & Fortin, J. Chem. Thermodyn. 42, 700 (2010),
#               doi:10.1016/j.jct.2010.01.003 (vibrating tube, 0.04%).
_WATER = (18.015, 10)
_OCTANOL = (130.231, 74)
_SOLVENT_LIQUID: dict[str, tuple[float, tuple[tuple[float, float, int], ...]]] = {
    "water": (997.047, ((1.0, *_WATER),)),
    "octanol": (821.85, ((1.0, *_OCTANOL),)),
    "woctanol": (831.7, ((1.0 - 0.04891, *_OCTANOL), (0.04891, *_WATER))),
    "hexadecane": (769.8, ((1.0, 226.448, 130),)),
}

_AVOGADRO = 6.02214076e23
_BOHR_M = 0.529177210903e-10


def solvent_electron_density(solvent: str) -> float:
    """The solvent's mean electron density as a liquid at 298.15 K, in e / bohr^3.

    Mass density times electrons per mass, every electron counted, summed over
    the liquid's components.  Water's is 0.0494; CMIRS rounds it to 0.05.
    """
    key = solvent.lower()
    if key not in _SOLVENT_LIQUID:
        raise ValueError(
            f"No liquid density for solvent {solvent!r}. "
            f"Known solvents: {', '.join(sorted(_SOLVENT_LIQUID))}"
        )
    density, components = _SOLVENT_LIQUID[key]
    mol_electrons_per_kg = sum(
        fraction * electrons / (molar_mass * 1e-3)
        for fraction, molar_mass, electrons in components
    )
    return density * mol_electrons_per_kg * _AVOGADRO * _BOHR_M**3


# The solvation single point for an OrbMol refinement: OrbMol has no implicit
# solvent, so the term comes from a cheap DFT pair (PCM and gas) on the
# [compute] driver.  wB97X-3c/vDZP is the default and the benchmarked choice.
DEFAULT_ORBMOL_SOLVATION_METHOD = "wB97X-3c"


@dataclass
class MoleculeConfig:
    smiles: str
    charge_min: int = -1
    charge_max: int = 0
    # Used by sampling (ALPB), refinement and scoring.  A run scored in the gas
    # phase ([scoring] solvent_model = "none") uses no solvent anywhere.
    solvent: str = "water"

    @property
    def charge_range(self) -> tuple[int, int]:
        return (self.charge_min, self.charge_max)


@dataclass
class SamplingConfig:
    approach: str = "rdkit_first"  # "rdkit_first" or "crest_first"
    ewin: float = 10.0  # energy window (kcal/mol)
    # Approach 1 (rdkit_first) options
    max_tautomers: int = 1000
    max_transforms: int = 1000
    crest_mode: str = "default"  # conformer search mode
    # Take the input SMILES as the only species: no tautomers, no charge-separated
    # protomers, no other charge state (charge_min = charge_max = its charge).
    # For reference data that fix the species, such as MNSol.
    fixed_species: bool = False
    # Approach 2 (crest_first) options
    prescreen_mode: str = "quick"
    prescreen_ewin: float = 6.0
    full_mode: str = "default"


@dataclass
class RefinementConfig:
    method: str = ""  # DFT functional, or an OrbMol model (see MODELS)
    basis: str = ""  # empty for OrbMol; fixed by the method for 3c composites
    solvation: Solvation = "single_point"
    # Level for the solvation single point when OrbMol optimizes (OrbMol has no
    # implicit solvent).  Refused for DFT refinement, where the single point is
    # at the refinement level so the optimizer's gas-phase energy can be reused.
    solvation_method: str = ""
    solvation_basis: str = ""
    # Optional cost control: skip the solvation single point for conformers
    # already this far above the charge state's minimum in gas phase.  Off by
    # default; skipped conformers are recorded as excluded_conformers with
    # reason "gas_prefilter".
    solvation_prefilter_ewin: float | None = None
    rrho: Rrho = "xtb"
    ewin: float = 8.0  # energy window (kcal/mol) for filtering after refinement
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen
    # Derived: "orbmol", or the [compute] driver for DFT refinement.
    engine: str = "pyscf"


@dataclass
class ScoringConfig:
    method: str = ""
    basis: str = ""
    solvent_model: str | None = None  # the run's implicit-solvent model
    ewin: float = 10.0
    pcm_hydrogen_radius: float = 1.1  # PCM cavity radius (Angstrom) for hydrogen
    # Save every solvated conformer's nonelectrostatic descriptors with its
    # energies (<output_dir>/nonelectrostatic.npz), for fitting the correction.
    nonelectrostatic_descriptors: bool = False


@dataclass
class ComputeConfig:
    driver: str = "pyscf"  # "psi4" or "pyscf": scoring, and any DFT in refinement
    threads: int | None = None
    memory_gb: float = DEFAULT_MEMORY_GB
    output_dir: str = "output"


@dataclass
class PkaConfig:
    molecule: MoleculeConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    protocol: str | None = None

    def provenance(self) -> dict[str, Any]:
        """What each stage computed, for the output file and for resume checks."""
        ref, sc = self.refinement, self.scoring
        solvation: dict[str, Any] = {"mode": ref.solvation}
        if ref.solvation != "none":
            solvation |= {
                "level": (
                    f"{ref.solvation_method}/{ref.solvation_basis}"
                    if ref.engine == "orbmol"
                    else f"{ref.method}/{ref.basis}"
                ),
                "model": sc.solvent_model,
                "driver": self.compute.driver,
            }
        return {
            "protocol": self.protocol,
            "solvent": self.molecule.solvent,
            "refinement": {
                "engine": ref.engine,
                "method": ref.method,
                "basis": ref.basis,
                "solvation": solvation,
                "rrho": ref.rrho,
                "ewin": ref.ewin,
                "solvation_prefilter_ewin": ref.solvation_prefilter_ewin,
            },
            "scoring": {
                "driver": self.compute.driver,
                "method": sc.method,
                "basis": sc.basis,
                "solvent_model": sc.solvent_model,
                "ewin": sc.ewin,
            },
        }


# ---------------------------------------------------------------------------
# Recommended protocols
# ---------------------------------------------------------------------------
# Defaults only: any key written in the file wins.  "fast" is ~5x cheaper than
# "high_quality" and, measured against it on pKa-set molecules, agrees to
# ~0.05 pKa at the scoring level and <=0.01 pKa on the refinement window.  Those
# figures, and both refinement windows, date from when "high_quality" refined
# on gas-phase geometries (Psi4, with a solvated single point); they have not
# been re-measured with solvated minimization.
PROTOCOLS: dict[str, dict[str, dict[str, Any]]] = {
    "high_quality": {
        "sampling": {"ewin": 10.0},
        # Minimized in IEF-PCM on either driver, so refinement ranks on the
        # physics scoring uses (gas phase if scoring is; see _fit_preset_solvation).
        "refinement": {"method": "wB97X-3c", "solvation": "optimize", "ewin": 8.0, "rrho": "xtb"},
        "scoring": {"method": "wB97M-V", "basis": "def2-QZVPPD", "ewin": 10.0},
    },
    "fast": {
        # A wider sampling window is affordable because OrbMol refinement costs
        # seconds per conformer, and xTB misranks against DFT in one direction.
        "sampling": {"ewin": 15.0},
        "refinement": {
            "method": "OrbMol-v2",
            "solvation": "single_point",
            "solvation_prefilter_ewin": 12.0,
            "ewin": 6.0,
            "rrho": "xtb",
        },
        "scoring": {"method": "wB97M-V", "basis": "def2-TZVPPD", "ewin": 10.0},
    },
}

_DFT_REFINEMENT_DEFAULT_METHOD = "wB97X-3c"
_SCORING_DEFAULTS = {"method": "wB97M-V", "basis": "def2-QZVPPD"}
_DEFAULT_SOLVENT_MODEL = {"pyscf": "IEFPCM", "psi4": "IEFPCM"}


# ---------------------------------------------------------------------------
# TOML loading + validation
# ---------------------------------------------------------------------------
def load_config(path: Path) -> PkaConfig:
    """Load and validate a pKa workflow config from a TOML file."""
    raw = tomllib.loads(path.read_text())
    protocol = raw.get("protocol")
    if protocol is not None and protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol!r} (must be one of {', '.join(PROTOCOLS)})")
    preset = PROTOCOLS.get(protocol or "", {})

    def section(name: str) -> dict[str, Any]:
        merged = dict(preset.get(name, {}))
        own = raw.get(name, {})
        # A preset's basis was chosen for the preset's method; a 3c composite
        # named in the file brings its own.
        if "basis" not in own and _default_basis(own.get("method", "")):
            merged.pop("basis", None)
        merged.update(own)
        return merged

    mol_raw = raw.get("molecule")
    if mol_raw is None:
        raise ValueError("Config must have a [molecule] section with 'smiles'")
    if "smiles" not in mol_raw:
        raise ValueError("[molecule] must specify 'smiles'")
    mol_raw = dict(mol_raw)

    ref_raw, score_raw, compute_raw = section("refinement"), section("scoring"), section("compute")
    _migrate(ref_raw, score_raw, compute_raw, mol_raw)
    _fit_preset_solvation(ref_raw, raw.get("refinement", {}), score_raw)

    molecule = MoleculeConfig(
        smiles=mol_raw["smiles"],
        charge_min=mol_raw.get("charge_min", -1),
        charge_max=mol_raw.get("charge_max", 0),
        solvent=mol_raw.get("solvent", "water"),
    )
    if molecule.charge_min > molecule.charge_max:
        raise ValueError(
            f"charge_min ({molecule.charge_min}) > charge_max ({molecule.charge_max})"
        )

    compute = ComputeConfig(
        driver=compute_raw.get("driver", "pyscf"),
        threads=compute_raw.get("threads"),
        memory_gb=compute_raw.get("memory_gb", DEFAULT_MEMORY_GB),
        output_dir=compute_raw.get("output_dir", "output"),
    )
    if compute.memory_gb <= 0:
        raise ValueError(f"[compute] memory_gb must be positive, got {compute.memory_gb}")
    if compute.driver == "orbmol":
        raise ValueError(
            "[compute] driver = 'orbmol' is not supported: the driver runs scoring, "
            "which needs implicit solvent, and OrbMol-v2 is gas-phase only. To refine "
            "with OrbMol set [refinement] method = 'OrbMol-v2' and keep [compute] "
            "driver = 'psi4' or 'pyscf'."
        )
    if compute.driver not in DRIVERS:
        raise ValueError(f"Unknown driver: {compute.driver!r} (must be 'psi4' or 'pyscf')")

    samp_raw = section("sampling")
    sampling = SamplingConfig(
        approach=samp_raw.get("approach", "rdkit_first"),
        ewin=samp_raw.get("ewin", 10.0),
        max_tautomers=samp_raw.get("max_tautomers", 1000),
        max_transforms=samp_raw.get("max_transforms", 1000),
        crest_mode=samp_raw.get("crest_mode", "default"),
        fixed_species=samp_raw.get("fixed_species", False),
        prescreen_mode=samp_raw.get("prescreen_mode", "quick"),
        prescreen_ewin=samp_raw.get("prescreen_ewin", 6.0),
        full_mode=samp_raw.get("full_mode", "default"),
    )
    if sampling.approach not in ("rdkit_first", "crest_first"):
        raise ValueError(
            f"Unknown sampling approach: {sampling.approach!r} "
            f"(must be 'rdkit_first' or 'crest_first')"
        )
    if sampling.fixed_species:
        if sampling.approach != "rdkit_first":
            raise ValueError(
                "[sampling] fixed_species needs approach = 'rdkit_first': crest_first "
                "finds its species with CREST and cannot be held to the input"
            )
        if molecule.charge_min != molecule.charge_max:
            raise ValueError(
                "[sampling] fixed_species runs the input's own charge state only: "
                "set charge_min = charge_max to it"
            )

    scoring = _build_scoring(score_raw, compute.driver)
    refinement = _build_refinement(ref_raw, compute.driver, scoring)
    _validate(refinement, scoring, compute)
    _warn_no_diffuse(scoring)

    return PkaConfig(
        molecule=molecule,
        sampling=sampling,
        refinement=refinement,
        scoring=scoring,
        compute=compute,
        protocol=protocol,
    )


def _build_scoring(raw: dict[str, Any], driver: str) -> ScoringConfig:
    model = raw.get("solvent_model", _DEFAULT_SOLVENT_MODEL[driver])
    if isinstance(model, str) and model.lower() == "none":
        model = None
    method = raw.get("method", _SCORING_DEFAULTS["method"])
    return ScoringConfig(
        method=method,
        basis=raw.get("basis", _default_basis(method) or _SCORING_DEFAULTS["basis"]),
        solvent_model=model,
        ewin=raw.get("ewin", 10.0),
        pcm_hydrogen_radius=raw.get("pcm_hydrogen_radius", 1.1),
        nonelectrostatic_descriptors=raw.get("nonelectrostatic_descriptors", False),
    )


def _build_refinement(
    raw: dict[str, Any], driver: str, scoring: ScoringConfig
) -> RefinementConfig:
    from tautomator.orbmol_runner import MODELS

    method = raw.get("method", _DFT_REFINEMENT_DEFAULT_METHOD)
    engine = "orbmol" if method.lower() in MODELS else driver
    basis = raw.get("basis", "" if engine == "orbmol" else _default_basis(method))
    solvation: Solvation = raw.get("solvation", _default_solvation(engine, scoring))
    solvation_method = raw.get("solvation_method", "")
    if engine == "orbmol" and solvation != "none" and not solvation_method:
        solvation_method = DEFAULT_ORBMOL_SOLVATION_METHOD
    return RefinementConfig(
        method=method,
        basis=basis,
        solvation=solvation,
        solvation_method=solvation_method,
        solvation_basis=raw.get("solvation_basis", _default_basis(solvation_method)),
        solvation_prefilter_ewin=raw.get("solvation_prefilter_ewin"),
        rrho=raw.get("rrho", "xtb"),
        ewin=raw.get("ewin", 8.0 if solvation != "none" else 10.0),
        pcm_hydrogen_radius=raw.get("pcm_hydrogen_radius", 1.1),
        engine=engine,
    )


def _default_basis(method: str) -> str:
    return COMPOSITE_BASIS.get(method.lower(), "")


def _default_solvation(engine: str, scoring: ScoringConfig) -> Solvation:
    if scoring.solvent_model is None:
        return "none"  # gas-phase scoring: the filter must rank on gas energies too
    # Both DFT drivers have analytical PCM gradients, so they minimize in solvent;
    # OrbMol has no implicit solvent at all.
    return "optimize" if engine in ("pyscf", "psi4") else "single_point"


def _validate(ref: RefinementConfig, scoring: ScoringConfig, compute: ComputeConfig) -> None:
    if ref.solvation not in ("none", "optimize", "single_point"):
        raise ValueError(
            f"Unknown [refinement] solvation: {ref.solvation!r} "
            "(must be 'none', 'optimize' or 'single_point')"
        )
    if ref.rrho not in ("xtb", "refinement"):
        raise ValueError(
            f"Unknown [refinement] rrho: {ref.rrho!r} (must be 'xtb' or 'refinement')"
        )

    # Level of theory: the 3c composites carry their basis with them.
    _check_composite("[refinement] method", ref.method, ref.basis, compute.driver)
    _check_composite(
        "[refinement] solvation_method", ref.solvation_method, ref.solvation_basis, compute.driver
    )
    _check_composite("[scoring] method", scoring.method, scoring.basis, compute.driver)

    if ref.engine == "orbmol":
        if ref.basis:
            raise ValueError(
                f"[refinement] basis = {ref.basis!r} has no meaning for an OrbMol model "
                "(the model is the level of theory); remove it."
            )
        if ref.solvation == "optimize":
            raise ValueError(
                "[refinement] solvation = 'optimize' is not supported with an OrbMol "
                "model: OrbMol-v2 is a gas-phase potential. Use 'single_point', which "
                f"adds a {DEFAULT_ORBMOL_SOLVATION_METHOD} solvation energy on the "
                "OrbMol geometry."
            )
    else:
        if ref.solvation_method or ref.solvation_basis:
            raise ValueError(
                "[refinement] solvation_method/solvation_basis apply only to an OrbMol "
                "refinement; DFT refinement takes its solvation single point at the "
                "refinement level, reusing the optimizer's gas-phase energy."
            )

    if ref.solvation != "none" and scoring.solvent_model is None:
        raise ValueError(
            "[refinement] computes a solvation energy but [scoring] is gas-phase "
            "(solvent_model = 'none'); the refinement filter would rank on different "
            "physics from scoring. Set [refinement] solvation = 'none', or give "
            "[scoring] a solvent model."
        )
    if ref.solvation == "none" and scoring.solvent_model is not None:
        log.warning(
            "[refinement] solvation = 'none' with solvated scoring: conformers are "
            "filtered on gas-phase energies, which rerank against scoring (a 4 kcal/mol "
            "window cost 3.3 pKa in testing). Keep ewin >= 10, or use "
            "solvation = 'single_point'."
        )

    if scoring.solvent_model is not None:
        allowed = _SOLVENT_MODELS[compute.driver]
        if scoring.solvent_model.upper().replace("-", "") not in {m.upper() for m in allowed}:
            raise ValueError(
                f"[scoring] solvent_model = {scoring.solvent_model!r} is not available "
                f"on the {compute.driver} driver (has: {', '.join(sorted(allowed))})."
            )
        if drop_solvent_model(scoring.solvent_model) is not None:
            _validate_drop(ref, scoring)

    if scoring.nonelectrostatic_descriptors:
        if compute.driver != "pyscf":
            raise ValueError(
                "[scoring] nonelectrostatic_descriptors needs [compute] driver = 'pyscf': "
                "the descriptors are read off the PySCF mean field of the solvated "
                "single point."
            )
        model = scoring.solvent_model
        if model is not None and drop_solvent_model(model) is not None:
            raise ValueError(
                "[scoring] nonelectrostatic_descriptors are computed on PySCF's own PCM "
                f"cavity, not yet on MOIST's DROP cavities (solvent_model = {model!r})"
            )
        if model is None or model.upper().replace("-", "") not in PCM_METHODS:
            given = "gas-phase scoring" if model is None else f"solvent_model = {model!r}"
            raise ValueError(
                "[scoring] nonelectrostatic_descriptors needs an analytical PCM solvent "
                f"model (IEF-PCM, C-PCM or COSMO), not {given}"
            )

    _validate_prefilter(ref)

    if ref.rrho == "refinement" and ref.engine == "psi4":
        log.warning(
            "[refinement] rrho = 'refinement' on the psi4 driver: Psi4 has no analytical "
            "Hessian for these functionals, so every conformer costs a finite-difference "
            "Hessian (~6N gradient calculations) -- usually far more than the geometry "
            "optimization itself. rrho = 'xtb' is the default for good reason."
        )


def _validate_drop(ref: RefinementConfig, scoring: ScoringConfig) -> None:
    """What a MOIST DROP cavity (solvent_model "<model>/<cavity>") cannot do."""
    import importlib.util

    model = scoring.solvent_model
    if importlib.util.find_spec("moist") is None:
        raise ValueError(
            f"[scoring] solvent_model = {model!r} needs MOIST, which is installed only in "
            "the moist pixi environment: `pixi install -e moist` (it compiles MOIST), then "
            "run with `pixi run -e moist ...`."
        )
    for name, radius in (
        ("[scoring]", scoring.pcm_hydrogen_radius),
        ("[refinement]", ref.pcm_hydrogen_radius),
    ):
        if radius != 1.1:
            raise ValueError(
                f"{name} pcm_hydrogen_radius = {radius} sets the hydrogen radius of PySCF's "
                f"own PCM cavity; solvent_model = {model!r} builds MOIST's cavity, with "
                "MOIST's radii. Remove it."
            )
    if ref.rrho == "refinement" and ref.solvation == "optimize" and ref.engine != "orbmol":
        raise ValueError(
            f"[refinement] rrho = 'refinement' with solvation = 'optimize' needs a Hessian "
            f"in solvent, and MOIST's PCM ({model!r}) has none. Use rrho = 'xtb'."
        )


def _validate_prefilter(ref: RefinementConfig) -> None:
    prefilter = ref.solvation_prefilter_ewin
    if prefilter is None:
        return
    if ref.solvation != "single_point":
        raise ValueError(
            "[refinement] solvation_prefilter_ewin only applies to "
            "solvation = 'single_point' (there is no solvation single point to skip "
            f"with solvation = {ref.solvation!r})."
        )
    if prefilter <= ref.ewin:
        raise ValueError(
            f"[refinement] solvation_prefilter_ewin ({prefilter}) must exceed ewin "
            f"({ref.ewin}): it would drop conformers inside the window it feeds."
        )
    if prefilter < ref.ewin + 4.0:
        log.warning(
            f"[refinement] solvation_prefilter_ewin ({prefilter}) is within 4 kcal/mol "
            f"of ewin ({ref.ewin}). Gas-phase and solvated rankings differ: conformers "
            "inside a 6 kcal/mol solvated window sat up to 9.1 kcal/mol above the "
            "gas-phase minimum in testing."
        )
    if ref.engine != "orbmol":
        log.warning(
            "[refinement] solvation_prefilter_ewin with DFT refinement saves little: "
            "the solvation single point is ~10% of one geometry optimization at the "
            "same level. It matters for OrbMol, where optimization takes seconds."
        )


def _check_composite(label: str, method: str, basis: str, driver: str) -> None:
    """A 3c composite is a functional *and* its basis; refuse other pairings."""
    if not method:
        return
    key = method.lower()
    required = COMPOSITE_BASIS.get(key)
    if required is None:
        return
    if basis and basis.lower() != required.lower():
        raise ValueError(
            f"{label} = {method!r} is parameterized with {required}, not {basis!r}; "
            f"{method} with another basis is a different method. Drop the basis to get "
            "the right one."
        )
    if driver == "pyscf" and key not in PYSCF_COMPOSITES:
        raise ValueError(
            f"{label} = {method!r} is not available on the pyscf driver (it implements "
            f"only {', '.join(sorted(PYSCF_COMPOSITES))} of the 3c composites). Use "
            "[compute] driver = 'psi4', or wB97X-3c."
        )


# Basis sets with diffuse functions on the heavy atoms, told by name.  The name
# is what the config has: each driver resolves it in its own library, and no
# library reachable from here knows all of them (PySCF's and basis_set_exchange
# lack jun-cc-pVDZ and heavy-aug-cc-pVDZ, Psi4's lacks ma-def2-SVP), so reading
# exponents would need this rule as its fallback anyway.  Nor do exponents give
# a sharper answer.  Across the 494 orbital sets in basis_set_exchange, the
# smallest sp exponent on C/N/O/F/S/Cl relative to aug-cc-pVDZ's runs from
# below 0.1 to above 6 with no gap to put a threshold in: def2-SVPD on nitrogen
# (1.7), vDZP on carbon (1.6) and cc-pV6Z (1.7) sit together.  What the name
# records is whether the set was built with an augmentation.
#
# Matched against the name with case, hyphens, underscores and spaces removed,
# which is how PySCF reads it.
_DIFFUSE_BASIS = re.compile(
    r"aug"  # aug-cc-pVXZ, d-aug-, heavy-aug-, aug-pcseg-n
    r"|^(feb|mar|apr|may|jun|jul)cc"  # the calendar truncations of aug-cc-pVXZ
    r"|^madef2"  # minimally augmented Karlsruhe
    r"|\d\+{1,2}g"  # Pople 6-31+G*, 6-311++G(2d,2p); not the tight d of cc-pV(D+d)Z
    r"|^def2(sv|tzv|qzv)pp?d$"  # Karlsruhe property-optimized: def2-SVPD ... def2-QZVPPD
)


def _normalize_basis(basis: str) -> str:
    return re.sub(r"[-_\s]", "", basis.lower())


def _has_diffuse(basis: str) -> bool:
    return bool(_DIFFUSE_BASIS.search(_normalize_basis(basis)))


def _warn_no_diffuse(scoring: ScoringConfig) -> None:
    """Warn about a scoring basis without diffuse functions.

    Scoring energies are compared across charge states, and the deprotonations
    between them need diffuse functions whether or not an anion appears.
    Deprotonation energies at wB97M-V in IEF-PCM water, def2-TZVPP minus
    def2-TZVPPD (kcal/mol):
      neutral acid -> anion: acetic acid 2.8, methanesulfonic acid 2.6,
          phenol 2.6, benzoic acid 2.0, 5-methyltetrazole 1.3
      cation -> zwitterion: glycine 1.8, beta-alanine 2.3, GABA 2.4, taurine 1.9,
          4-(aminomethyl)phenol 2.3, nicotinic acid 1.7,
          5-(aminomethyl)tetrazole 1.0; 66-91% of the same acid group without
          the cation
      ammonium -> amine (no anionic site at all): glycine 0.9, GABA 0.7
    def2-SVP minus def2-SVPD is 5-7 times larger throughout (4.1-16.1).  So the
    check does not look at the charge range: keeping anions out of the run does
    not keep the error out.

    vDZP is exempt.  It has no diffuse shells by name, but its valence s and p
    are nearly as diffuse as aug-cc-pVDZ's, and it does not have the failure
    this warns about: against def2-TZVPPD it is within 0.8 kcal/mol on the same
    deprotonations except the two tetrazoles, which it puts 1.6 and 1.9 lower,
    with 72 basis functions on acetic acid to def2-TZVPPD's 222.

    A 3c composite is left alone, its basis not being a choice.  So are the
    refinement bases: those energies only rank conformers and tautomers within
    a charge state, where def2-TZVPP and def2-TZVPPD differ by 0.7 and 0.3
    kcal/mol on a zwitterion against its neutral tautomer (glycine,
    4-aminobutanoic acid).
    """
    if scoring.method.lower() in COMPOSITE_BASIS:
        return
    if _normalize_basis(scoring.basis) == "vdzp":
        return
    if not _has_diffuse(scoring.basis):
        log.warning(
            f"[scoring] basis = {scoring.basis!r} has no diffuse functions. Scoring "
            "energies are compared across charge states, and the deprotonations "
            "between them need diffuse functions even when no anion appears: in "
            "testing, def2-TZVPP put deprotonation energies above def2-TZVPPD by "
            "1.3-2.8 kcal/mol for neutral acids, 1.0-2.4 for cations becoming "
            "zwitterions and 0.7-0.9 for ammonium ions (0.5-2.0 pKa), and def2-SVP "
            "above def2-SVPD by 4-16. Use an augmented set such as def2-TZVPPD."
        )


def _fit_preset_solvation(
    ref_raw: dict[str, Any], own: dict[str, Any], score_raw: dict[str, Any]
) -> None:
    """Drop a preset's solvation settings where they no longer apply.

    A preset's [refinement] solvation and solvation_prefilter_ewin assume solvated
    scoring and the preset's own solvation mode.  With gas-phase scoring,
    refinement is gas-phase too, as it is without a preset; and a prefilter the
    file did not set goes whenever there is no solvation single point to skip.
    TOML cannot unset a preset's key, so without this a preset could not be run
    gas-phase at all.  Anything written in the file still wins.
    """
    model = score_raw.get("solvent_model")
    gas_scoring = isinstance(model, str) and model.lower() == "none"
    if gas_scoring and "solvation" not in own:
        ref_raw.pop("solvation", None)
    single_point = ref_raw.get("solvation", "single_point") == "single_point"
    if "solvation_prefilter_ewin" not in own and (gas_scoring or not single_point):
        ref_raw.pop("solvation_prefilter_ewin", None)


def _migrate(
    ref_raw: dict[str, Any],
    score_raw: dict[str, Any],
    compute_raw: dict[str, Any],
    mol_raw: dict[str, Any],
) -> None:
    """Accept the pre-protocol spellings, with a warning."""
    if (driver := ref_raw.pop("driver", None)) is not None:
        if driver == "orbmol":
            ref_raw.setdefault("method", "OrbMol-v2")
        elif driver != compute_raw.get("driver", "pyscf"):
            raise ValueError(
                f"[refinement] driver = {driver!r} no longer selects a DFT backend: "
                "[compute] driver runs every DFT job. Set [compute] driver instead."
            )
        log.warning(
            "[refinement] driver is deprecated: an OrbMol model in [refinement] method "
            "selects OrbMol, and [compute] driver runs the DFT."
        )
    if (rrho_method := ref_raw.pop("rrho_method", None)) is not None:
        ref_raw.setdefault("rrho", "refinement" if rrho_method == "dft" else rrho_method)
        log.warning(
            f"[refinement] rrho_method = {rrho_method!r} is deprecated: use "
            f"rrho = {ref_raw['rrho']!r} ('dft' is now 'refinement', a Hessian from "
            "whatever optimized the geometry)."
        )
    if (model := ref_raw.pop("solvent_model", None)) is not None:
        ref_raw.setdefault("solvation", "optimize")
        score_raw.setdefault("solvent_model", model)
        log.warning(
            f"[refinement] solvent_model = {model!r} is deprecated: [refinement] "
            "solvation says how solvation enters refinement, and [scoring] "
            "solvent_model is the model for the whole run."
        )
    for name, sec in (("refinement", ref_raw), ("scoring", score_raw)):
        if (solvent := sec.pop("solvent", None)) is not None:
            if mol_raw.setdefault("solvent", solvent) != solvent:
                raise ValueError(
                    f"[{name}] solvent = {solvent!r} conflicts with "
                    f"[molecule] solvent = {mol_raw['solvent']!r}; a run has one solvent."
                )
            log.warning(f"[{name}] solvent is deprecated: use [molecule] solvent.")
