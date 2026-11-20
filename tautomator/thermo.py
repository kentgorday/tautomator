"""Quasi-RRHO thermochemistry from harmonic frequencies.

Implements Grimme's quasi-rigid-rotor-harmonic-oscillator (quasi-RRHO)
approach (Grimme, Chem. Eur. J. 2012, 18, 9955) for computing free
energy corrections from vibrational frequencies. Low-frequency modes
are interpolated between harmonic oscillator and free rotor treatments
using a damping function, avoiding the divergent entropy contributions
that plague standard RRHO at low frequencies.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Physical constants
KB = 1.380649e-23  # Boltzmann constant, J/K
H_PLANCK = 6.62607015e-34  # Planck constant, J·s
C_LIGHT = 2.99792458e10  # speed of light, cm/s
R_GAS = 8.314462618  # gas constant, J/(mol·K)
HARTREE_TO_JOULE = 4.3597447222071e-18
CAL_TO_JOULE = 4.184
AVOGADRO = 6.02214076e23

# Default cutoff for quasi-RRHO damping (cm⁻¹)
QRRHO_CUTOFF = 100.0

# The refinement's floor under the frequencies (cm⁻¹, by magnitude): anything
# softer counts as this.  Below the cutoff qRRHO's free-rotor branch takes its
# moment of inertia from the frequency, so G moves by RT/2 ln 2 = 0.2 kcal/mol
# each time a soft mode's frequency halves, and an xTB --bhess mode at a DFT
# geometry is that soft and that fickle.  Re-run at the same geometries jittered
# by 0.002 Angstrom (about geomeTRIC's converged step), the lowest conformers of
# 211 finished MNSol runs moved their RRHO term by more than 0.2 kcal/mol in 15,
# by up to 0.57; with a 50 cm⁻¹ floor by at most 0.04.  Between gas and
# solvent, the soft modes' share of each solute's term differed by 0.20
# kcal/mol rms, 0.08 with the floor.  30 and 40 cm⁻¹ leave up to 0.10 and 0.06
# of the jitter; 75 and 100 remove no more.
FREQUENCY_FLOOR = 50.0

# Frequency scaling factors from Kesharwani, Brauer, Martin, J. Phys. Chem. A
# 2015, 119, 1701.  These correct for the systematic overestimation of
# calculated harmonic frequencies vs experimental fundamentals.  For modern
# DFT functionals with sufficient basis sets, ZPVE scale factors are
# virtually always ~0.98 and thermal/entropy factors ~0.96.
DEFAULT_SCALE_ZPVE = 0.9856
DEFAULT_SCALE_THERMAL = 0.9627


def quasi_rrho_free_energy(
    frequencies_cm: list[float],
    temperature: float = 298.15,
    cutoff: float = QRRHO_CUTOFF,
    pressure: float = 101325.0,
    scale_zpve: float = DEFAULT_SCALE_ZPVE,
    scale_thermal: float = DEFAULT_SCALE_THERMAL,
    floor: float = 0.0,
) -> float:
    """Compute the vibrational free energy correction using quasi-RRHO.

    Uses Grimme's interpolation between harmonic oscillator and free
    rotor for low-frequency modes (below `cutoff` cm⁻¹).

    Harmonic frequencies are scaled to correct for systematic
    overestimation vs experiment (Kesharwani, Brauer, Martin, J. Phys.
    Chem. A 2015, 119, 1701).  Separate scale factors are applied to
    the ZPVE and thermal (enthalpy + entropy) contributions.

    Args:
        frequencies_cm: Vibrational frequencies in cm⁻¹.  Must contain
            only true vibrational modes (translational and rotational
            modes should already be projected out by the QM backend).
            Imaginary frequencies (negative values) are treated as real
            using their absolute value.
        temperature: Temperature in Kelvin.
        cutoff: Frequency cutoff for quasi-RRHO damping in cm⁻¹.
        pressure: Pressure in Pa (for translational entropy, unused here
            since we only compute vibrational contributions).
        scale_zpve: Scale factor for ZPVE contributions (~0.98 for
            modern functionals with sufficient basis sets).
        scale_thermal: Scale factor for thermal enthalpy and entropy
            contributions (~0.96 for modern functionals).
        floor: Frequencies softer than this (cm⁻¹, by magnitude) count as
            this; 0 for none.  See FREQUENCY_FLOOR.

    Returns:
        Vibrational free energy correction in Hartree.
        This is G_vib = ZPE + H_vib(T) - T*S_vib(T), where S_vib
        uses the quasi-RRHO treatment for low-frequency modes.
    """
    kbt = KB * temperature
    beta = 1.0 / kbt

    g_vib = 0.0
    for freq in frequencies_cm:
        # Treat imaginary frequencies as real (use absolute value).
        # Small imaginary frequencies at a nominally optimized geometry are
        # typically numerical artifacts, not true saddle points.
        if freq < 0:
            log.warning(
                f"Imaginary frequency ({freq:.1f} cm⁻¹) treated as real "
                f"({abs(freq):.1f} cm⁻¹) for free energy calculation"
            )
            freq = abs(freq)
        freq = max(freq, floor)

        # Scaled frequencies for each contribution
        freq_zpve = freq * scale_zpve
        freq_thermal = freq * scale_thermal

        # ZPE contribution (scaled)
        hv_zpve = H_PLANCK * C_LIGHT * freq_zpve
        zpe = 0.5 * hv_zpve

        # Thermal enthalpy: hv / (exp(hv/kT) - 1) (scaled)
        hv_thermal = H_PLANCK * C_LIGHT * freq_thermal
        x = hv_thermal * beta
        h_vib = hv_thermal / (np.exp(x) - 1.0)

        # Entropy: quasi-RRHO interpolation (using thermal-scaled frequency)
        s_ho = _entropy_ho(freq_thermal, temperature)
        s_fr = _entropy_free_rotor(freq_thermal, temperature)
        w = _grimme_weight(freq_thermal, cutoff)
        s_vib = w * s_ho + (1.0 - w) * s_fr

        # G = ZPE + H_thermal - T*S  (per mode, in Joules)
        g_mode = zpe + h_vib - temperature * s_vib
        g_vib += g_mode

    # Convert from Joules (per molecule) to Hartree
    return g_vib / HARTREE_TO_JOULE


def _entropy_ho(freq: float, temperature: float) -> float:
    """Harmonic oscillator entropy for a single mode (J/K per molecule)."""
    hv = H_PLANCK * C_LIGHT * freq
    x = hv / (KB * temperature)
    # S_HO = kB * [x/(e^x - 1) - ln(1 - e^{-x})]
    return float(KB * (x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))))


def _entropy_free_rotor(freq: float, temperature: float) -> float:
    """Free rotor entropy for a single mode (J/K per molecule).

    Uses the Grimme (2012) expression with an effective moment of inertia
    derived from the frequency: mu' = h/(8*pi^2*freq*c).
    """
    # Effective moment of inertia
    mu_prime = H_PLANCK / (8.0 * np.pi**2 * C_LIGHT * freq)
    # S_FR = R * (1/2 + ln(sqrt(8*pi^3*mu'*kT/h^2)))
    # Per molecule (divide R by Avogadro = kB):
    arg = 8.0 * np.pi**3 * mu_prime * KB * temperature / H_PLANCK**2
    return float(KB * (0.5 + 0.5 * np.log(arg)))


def _grimme_weight(freq: float, cutoff: float) -> float:
    """Grimme damping function: w(v) = 1 / (1 + (cutoff/v)^4).

    Returns 1 for frequencies >> cutoff (pure HO),
    0 for frequencies << cutoff (pure free rotor).
    """
    return 1.0 / (1.0 + (cutoff / freq) ** 4)
