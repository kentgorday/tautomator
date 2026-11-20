from tautomator.thermo import (
    DEFAULT_SCALE_THERMAL,
    DEFAULT_SCALE_ZPVE,
    QRRHO_CUTOFF,
    _grimme_weight,
    quasi_rrho_free_energy,
)


class TestGrimmeWeight:
    def test_high_frequency_is_ho(self) -> None:
        # Far above cutoff -> weight ~ 1 (harmonic oscillator)
        w = _grimme_weight(1000.0, QRRHO_CUTOFF)
        assert w > 0.999

    def test_low_frequency_is_free_rotor(self) -> None:
        # Far below cutoff -> weight ~ 0 (free rotor)
        w = _grimme_weight(10.0, QRRHO_CUTOFF)
        assert w < 0.01

    def test_at_cutoff_is_half(self) -> None:
        # At the cutoff frequency, w = 1/(1 + 1) = 0.5
        w = _grimme_weight(QRRHO_CUTOFF, QRRHO_CUTOFF)
        assert abs(w - 0.5) < 1e-10

    def test_monotonically_increasing(self) -> None:
        freqs = [10, 50, 100, 200, 500, 1000]
        weights = [_grimme_weight(f, QRRHO_CUTOFF) for f in freqs]
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]


class TestQuasiRRHOFreeEnergy:
    def test_single_high_frequency(self) -> None:
        # A single high-frequency mode (like a C-H stretch ~3000 cm⁻¹)
        # should give a positive free energy (dominated by ZPE)
        g = quasi_rrho_free_energy([3000.0])
        assert g > 0  # ZPE contribution is positive

    def test_imaginary_frequencies_treated_as_real(self) -> None:
        # Imaginary frequencies (negative values) are treated as real
        # using their absolute value
        g_imag = quasi_rrho_free_energy([3000.0, -100.0])
        g_real = quasi_rrho_free_energy([3000.0, 100.0])
        assert abs(g_imag - g_real) < 1e-12

    def test_water_frequencies(self) -> None:
        # Water has 3 vibrational modes: ~1595, ~3657, ~3756 cm⁻¹
        # (T/R modes already projected out by the QM backend)
        freqs = [1595.0, 3657.0, 3756.0]
        g = quasi_rrho_free_energy(freqs, temperature=298.15)
        # ZPE of water: ~0.5 * (1595 + 3657 + 3756) cm⁻¹ ≈ 4504 cm⁻¹
        # In Hartree: ~0.0205
        # Total G_vib should be in this ballpark (ZPE dominates)
        assert 0.015 < g < 0.025

    def test_temperature_dependence(self) -> None:
        freqs = [500.0, 1000.0, 1500.0]
        g_low = quasi_rrho_free_energy(freqs, temperature=200.0)
        g_high = quasi_rrho_free_energy(freqs, temperature=400.0)
        # Higher temperature -> lower free energy (larger -TS term)
        assert g_high < g_low

    def test_empty_returns_zero(self) -> None:
        g = quasi_rrho_free_energy([])
        assert g == 0.0

    def test_low_frequency_damped(self) -> None:
        # A very low frequency (50 cm⁻¹) should use mostly free-rotor
        # entropy, giving a more negative -TS contribution than pure HO
        g_low_freq = quasi_rrho_free_energy([50.0])
        g_high_freq = quasi_rrho_free_energy([500.0])
        # Low freq mode has smaller ZPE but more favorable entropy
        # -> lower free energy
        assert g_low_freq < g_high_freq


class TestFrequencyScaling:
    def test_default_scale_factors(self) -> None:
        assert DEFAULT_SCALE_ZPVE == 0.9856
        assert DEFAULT_SCALE_THERMAL == 0.9627

    def test_no_scaling_gives_higher_free_energy(self) -> None:
        # Unscaled frequencies are higher -> larger ZPE -> higher G_vib
        freqs = [1595.0, 3657.0, 3756.0]
        g_scaled = quasi_rrho_free_energy(freqs)
        g_unscaled = quasi_rrho_free_energy(freqs, scale_zpve=1.0, scale_thermal=1.0)
        assert g_unscaled > g_scaled

    def test_scaling_affects_result(self) -> None:
        # Verify scaling actually changes the result (not silently ignored)
        freqs = [1000.0, 2000.0, 3000.0]
        g_default = quasi_rrho_free_energy(freqs)
        g_custom = quasi_rrho_free_energy(freqs, scale_zpve=0.95, scale_thermal=0.90)
        assert g_default != g_custom

    def test_zpve_and_thermal_scale_independently(self) -> None:
        # Changing only one scale factor should change the result
        freqs = [1000.0, 2000.0]
        g_ref = quasi_rrho_free_energy(freqs, scale_zpve=1.0, scale_thermal=1.0)
        g_zpve_only = quasi_rrho_free_energy(freqs, scale_zpve=0.98, scale_thermal=1.0)
        g_thermal_only = quasi_rrho_free_energy(freqs, scale_zpve=1.0, scale_thermal=0.96)
        assert g_ref != g_zpve_only
        assert g_ref != g_thermal_only
        assert g_zpve_only != g_thermal_only


class TestFrequencyFloor:
    def test_a_softer_mode_counts_as_the_floor(self) -> None:
        assert quasi_rrho_free_energy([10.0, 1500.0], floor=50.0) == quasi_rrho_free_energy(
            [50.0, 1500.0]
        )

    def test_an_imaginary_mode_is_floored_by_its_magnitude(self) -> None:
        assert quasi_rrho_free_energy([-20.0], floor=50.0) == quasi_rrho_free_energy([50.0])
        assert quasi_rrho_free_energy([-80.0], floor=50.0) == quasi_rrho_free_energy([80.0])

    def test_stiffer_modes_and_no_floor_are_untouched(self) -> None:
        assert quasi_rrho_free_energy([60.0, 900.0], floor=50.0) == quasi_rrho_free_energy(
            [60.0, 900.0]
        )
        assert quasi_rrho_free_energy([10.0]) != quasi_rrho_free_energy([50.0])
