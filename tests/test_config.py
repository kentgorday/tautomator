import logging
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tautomator import xtb_runner
from tautomator.config import (
    _SETTINGS,
    PROTOCOLS,
    SOLVENT_DIELECTRIC,
    PkaConfig,
    _has_diffuse,
    alpb_name,
    load_config,
    pcm_radii,
    solvent_electron_density,
)
from tautomator.types import Geometry


@pytest.fixture()
def tmp_toml(tmp_path: Path) -> Callable[[str], Path]:
    """Helper to write a TOML string and return its path."""

    def _write(content: str) -> Path:
        p = tmp_path / "config.toml"
        p.write_text(textwrap.dedent(content))
        return p

    return _write


class TestMinimalConfig:
    def test_molecule_only(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.molecule.smiles == "CCO"
        assert cfg.molecule.charge_range == (-1, 0)
        assert cfg.molecule.solvent == "water"

    def test_missing_molecule_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must have a \\[molecule\\]"):
            load_config(tmp_toml(""))

    def test_missing_smiles_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must specify 'smiles'"):
            load_config(
                tmp_toml("""
                [molecule]
                charge_min = -1
            """)
            )


class TestChargeRange:
    def test_custom_range(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "NCC(=O)O"
            charge_min = -2
            charge_max = 2
        """)
        )
        assert cfg.molecule.charge_range == (-2, 2)

    def test_inverted_range_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"charge_min.*>.*charge_max"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                charge_min = 1
                charge_max = -1
            """)
            )


class TestDriverDefaults:
    def test_pyscf_minimizes_in_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "pyscf"
        """)
        )
        assert (cfg.refinement.method, cfg.refinement.basis) == ("wB97X-3c", "vDZP")
        assert cfg.refinement.engine == "pyscf"
        # PySCF has analytical PCM gradients, so the cheapest way to a solvated
        # refinement energy is to minimize in solvent outright.
        assert cfg.refinement.solvation == "optimize"
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-QZVPPD")
        assert cfg.scoring.solvent_model == "IEFPCM/SvdW-DROP"

    def test_psi4_minimizes_in_solvent_too(self, tmp_toml: Callable[[str], Path]) -> None:
        # psi4-swig-pcm gives Psi4 analytical PCM gradients as well.
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.solvation == "optimize"
        assert cfg.scoring.solvent_model == "IEFPCM/SvdW-DROP"

    def test_gas_phase_scoring_makes_refinement_gas_phase(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [scoring]
            solvent_model = "none"
        """)
        )
        assert cfg.scoring.solvent_model is None
        assert cfg.refinement.solvation == "none"
        assert cfg.refinement.ewin == 10.0  # wider, because gas rankings rerank

    def test_unknown_driver_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [compute]
                driver = "gaussian"
            """)
            )

    def test_solvent_model_not_on_this_driver_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="not available on the psi4 driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [scoring]
                solvent_model = "SMD"
                [compute]
                driver = "psi4"
            """)
            )

    def test_pyscf_does_not_offer_ssvpe(self, tmp_toml: Callable[[str], Path]) -> None:
        # Its symmetrized system matrix costs ions several kcal/mol on PySCF's
        # surface, so the model is refused rather than left as a trap.
        with pytest.raises(ValueError, match="not available on the pyscf driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [scoring]
                solvent_model = "SSVPE"
                [compute]
                driver = "pyscf"
            """)
            )


class TestProtocols:
    def test_fast(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.ewin == 15.0
        assert cfg.refinement.engine == "orbmol"
        assert cfg.refinement.method == "OrbMol-v2"
        assert cfg.refinement.solvation == "single_point"
        assert (cfg.refinement.solvation_method, cfg.refinement.solvation_basis) == (
            "wB97X-3c",
            "vDZP",
        )
        assert cfg.refinement.solvation_prefilter_ewin == 12.0
        assert cfg.refinement.ewin == 6.0
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-TZVPPD")

    def test_high_quality(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "high_quality"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.ewin == 10.0
        assert (cfg.refinement.method, cfg.refinement.basis) == ("wB97X-3c", "vDZP")
        assert cfg.refinement.ewin == 8.0
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97M-V", "def2-QZVPPD")

    @pytest.mark.parametrize("driver", ["pyscf", "psi4"])
    def test_high_quality_minimizes_in_solvent(
        self, driver: str, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml(f"""
            protocol = "high_quality"
            [molecule]
            smiles = "CCO"
            [compute]
            driver = "{driver}"
        """)
        )
        assert cfg.refinement.solvation == "optimize"
        assert cfg.scoring.solvent_model == "IEFPCM/SvdW-DROP"

    @pytest.mark.parametrize("protocol", ["high_quality", "fast"])
    @pytest.mark.parametrize("own_solvation", ["", 'solvation = "none"'])
    def test_gas_phase_scoring_makes_the_preset_gas_phase(
        self, protocol: str, own_solvation: str, tmp_toml: Callable[[str], Path]
    ) -> None:
        # As without a preset; fast's solvation pre-filter goes with its single point.
        cfg = load_config(
            tmp_toml(f"""
            protocol = "{protocol}"
            [molecule]
            smiles = "CCO"
            [refinement]
            {own_solvation}
            [scoring]
            solvent_model = "none"
        """)
        )
        assert cfg.scoring.solvent_model is None
        assert cfg.refinement.solvation == "none"
        assert cfg.refinement.solvation_prefilter_ewin is None

    def test_fast_without_its_single_point_drops_the_prefilter(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [refinement]
            solvation = "none"
        """)
        )
        assert cfg.refinement.solvation == "none"
        assert cfg.refinement.solvation_prefilter_ewin is None

    def test_settings_written_in_the_file_still_win(self, tmp_toml: Callable[[str], Path]) -> None:
        gas_fast = """
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [scoring]
            solvent_model = "none"
            [refinement]
        """
        with pytest.raises(ValueError, match="computes a solvation energy"):
            load_config(tmp_toml(gas_fast + '            solvation = "single_point"\n'))
        with pytest.raises(ValueError, match="only applies to solvation = 'single_point'"):
            load_config(tmp_toml(gas_fast + "            solvation_prefilter_ewin = 12.0\n"))

    def test_file_overrides_preset(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [refinement]
            ewin = 4.0
            solvation_prefilter_ewin = 14.0
        """)
        )
        assert cfg.refinement.ewin == 4.0
        assert cfg.refinement.solvation_prefilter_ewin == 14.0
        assert cfg.refinement.method == "OrbMol-v2"  # still the preset's

    def test_unknown_protocol_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown protocol"):
            load_config(
                tmp_toml("""
                protocol = "cheapest"
                [molecule]
                smiles = "CCO"
            """)
            )

    def test_provenance_records_what_each_stage_computed(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        prov = cfg.provenance()
        assert prov["protocol"] == "fast"
        assert prov["refinement"]["engine"] == "orbmol"
        # The solvation level is the DFT pair's, not OrbMol's.
        assert prov["refinement"]["solvation"] == {
            "mode": "single_point",
            "level": "wB97X-3c/vDZP",
            "model": "IEFPCM/SvdW-DROP",
            "driver": "pyscf",
        }
        assert prov["scoring"]["basis"] == "def2-TZVPPD"


class TestOrbmolRefinement:
    def test_selected_by_method(self, tmp_toml: Callable[[str], Path]) -> None:
        # The method names the engine: [compute] driver stays the DFT backend,
        # because it still runs the solvation single point and all of scoring.
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            [compute]
            driver = "pyscf"
        """)
        )
        assert cfg.refinement.engine == "orbmol"
        assert cfg.refinement.basis == ""
        assert cfg.compute.driver == "pyscf"
        assert cfg.refinement.solvation == "single_point"
        assert cfg.refinement.solvation_method == "wB97X-3c"

    def test_solvation_level_is_configurable(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            solvation_method = "r2SCAN-3c"
            [compute]
            driver = "psi4"
        """)
        )
        assert (cfg.refinement.solvation_method, cfg.refinement.solvation_basis) == (
            "r2SCAN-3c",
            "def2-mTZVPP",
        )

    def test_cannot_score(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"\[compute\] driver = 'orbmol'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [compute]
                driver = "orbmol"
            """)
            )

    def test_cannot_minimize_in_solvent(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="gas-phase potential"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "OrbMol-v2"
                solvation = "optimize"
            """)
            )

    def test_basis_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="no meaning for an OrbMol model"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "OrbMol-v2"
                basis = "def2-SVP"
            """)
            )

    def test_own_hessian_accepted(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "OrbMol-v2"
            rrho = "refinement"
        """)
        )
        assert cfg.refinement.rrho == "refinement"


class TestSolvationValidation:
    def test_solvation_method_rejected_for_dft(self, tmp_toml: Callable[[str], Path]) -> None:
        # DFT refinement takes its single point at the refinement level, reusing
        # the optimizer's gas energy; a second level would just be waste.
        with pytest.raises(ValueError, match="apply only to an OrbMol refinement"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation_method = "wB97X-3c"
            """)
            )

    def test_psi4_takes_a_single_point_on_request(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            solvation = "single_point"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.solvation == "single_point"

    @pytest.mark.parametrize("model", ["IEFPCM", "PCM", "CPCM", "COSMO"])
    def test_psi4_offers_the_pyscf_pcm_models(
        self, tmp_toml: Callable[[str], Path], model: str
    ) -> None:
        cfg = load_config(
            tmp_toml(f"""
            [molecule]
            smiles = "CCO"
            [scoring]
            solvent_model = "{model}"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.scoring.solvent_model == model

    def test_solvated_refinement_with_gas_scoring_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="different physics from scoring"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "single_point"
                [scoring]
                solvent_model = "none"
            """)
            )

    def test_gas_refinement_with_solvated_scoring_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "none"
            """)
            )
        assert cfg.refinement.solvation == "none"
        assert "rerank against scoring" in caplog.text

    def test_unknown_solvation_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"Unknown \[refinement\] solvation"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "implicit"
            """)
            )


class TestSolvationPrefilter:
    def test_accepted_below_the_window(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.refinement.solvation_prefilter_ewin == 12.0

    def test_needs_a_solvation_single_point(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="only applies to solvation = 'single_point'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation_prefilter_ewin = 12.0
                [compute]
                driver = "pyscf"
            """)
            )

    def test_inside_the_window_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="must exceed ewin"):
            load_config(
                tmp_toml("""
                protocol = "fast"
                [molecule]
                smiles = "CCO"
                [refinement]
                ewin = 6.0
                solvation_prefilter_ewin = 5.0
            """)
            )

    def test_narrow_margin_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                protocol = "fast"
                [molecule]
                smiles = "CCO"
                [refinement]
                ewin = 6.0
                solvation_prefilter_ewin = 7.0
            """)
            )
        assert "9.1 kcal/mol above the gas-phase minimum" in caplog.text

    def test_pointless_for_dft_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvation = "single_point"
                solvation_prefilter_ewin = 14.0
                [compute]
                driver = "psi4"
            """)
            )
        assert "saves little" in caplog.text


class TestCompositeMethods:
    def test_basis_is_implied(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "r2SCAN-3c"
            [compute]
            driver = "psi4"
        """)
        )
        assert cfg.refinement.basis == "def2-mTZVPP"

    def test_matching_basis_accepted(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "wB97X-3c"
            basis = "vdzp"
        """)
        )
        assert cfg.refinement.basis == "vdzp"

    def test_other_basis_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="parameterized with vDZP"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "wB97X-3c"
                basis = "def2-TZVP"
            """)
            )

    def test_pyscf_has_only_wb97x_3c(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="not available on the pyscf driver"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                method = "r2SCAN-3c"
                [compute]
                driver = "pyscf"
            """)
            )

    @pytest.mark.parametrize("header", ["", 'protocol = "high_quality"', 'protocol = "fast"'])
    def test_composite_scoring_gets_its_basis(
        self, header: str, tmp_toml: Callable[[str], Path]
    ) -> None:
        # Neither the scoring default nor a preset's basis applies to a composite.
        cfg = load_config(
            tmp_toml(f'{header}\n[molecule]\nsmiles = "CCO"\n[scoring]\nmethod = "wB97X-3c"\n')
        )
        assert (cfg.scoring.method, cfg.scoring.basis) == ("wB97X-3c", "vDZP")

    def test_preset_basis_kept_for_a_plain_functional(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        cfg = load_config(
            tmp_toml("""
            protocol = "fast"
            [molecule]
            smiles = "CCO"
            [scoring]
            method = "B3LYP"
        """)
        )
        assert cfg.scoring.basis == "def2-TZVPPD"

    def test_plain_functional_keeps_its_basis(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [refinement]
            method = "B3LYP"
            basis = "6-31G*"
        """)
        )
        assert (cfg.refinement.method, cfg.refinement.basis) == ("B3LYP", "6-31G*")


class TestDiffuseBasis:
    @pytest.mark.parametrize(
        "basis",
        [
            "def2-SVPD",
            "def2-TZVPPD",
            "DEF2-QZVPPD",
            "ma-def2-SVP",
            "aug-cc-pVDZ",
            "d-aug-cc-pVTZ",
            "heavy-aug-cc-pVDZ",
            "jun-cc-pVDZ",
            "aug-cc-pV(D+d)Z",
            "aug-pcseg-1",
            "6-31+G*",
            "6-31+G(d)",
            "6-311++G(2d,2p)",
            "augccpvdz",  # PySCF ignores case, hyphens and underscores
            "def2_svpd",
        ],
    )
    def test_names_with_diffuse_functions(self, basis: str) -> None:
        assert _has_diffuse(basis)

    @pytest.mark.parametrize(
        "basis",
        [
            "def2-SVP",
            "def2-TZVPP",
            "def2-QZVPP",
            "def2-mTZVPP",
            "cc-pVDZ",
            "cc-pV(D+d)Z",  # a tight d, not a diffuse shell
            "pcseg-1",
            "6-31G*",
            "6-31G(d)",
            "6-311G(2d,2p)",
            "vDZP",
            "STO-3G",
        ],
    )
    def test_names_without_diffuse_functions(self, basis: str) -> None:
        assert not _has_diffuse(basis)

    def test_scoring_basis_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                basis = "def2-TZVPP"
            """)
            )
        assert "[scoring] basis = 'def2-TZVPP' has no diffuse functions" in caplog.text

    def test_vdzp_scoring_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                method = "wB97M-V"
                basis = "vDZP"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_refinement_basis_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Refinement only ranks within a charge state.
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [refinement]
                method = "B3LYP"
                basis = "6-31G*"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_composite_scoring_does_not_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                [scoring]
                method = "wB97X-3c"
            """)
            )
        assert "diffuse" not in caplog.text

    def test_cation_only_run_warns(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # No anion charge state, but zwitterions and plain amine deprotonation
        # still need diffuse functions.
        with caplog.at_level(logging.WARNING):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "NCC(=O)O"
                charge_min = 0
                charge_max = 1
                [scoring]
                basis = "def2-TZVPP"
            """)
            )
        assert "[scoring] basis = 'def2-TZVPP' has no diffuse functions" in caplog.text

    @pytest.mark.parametrize("header", ["", 'protocol = "high_quality"', 'protocol = "fast"'])
    def test_defaults_are_quiet(
        self, header: str, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            load_config(tmp_toml(f'{header}\n[molecule]\nsmiles = "CC(=O)O"\n'))
        assert "diffuse" not in caplog.text


class TestSamplingConfig:
    def test_defaults(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.sampling.approach == "rdkit_first"
        assert cfg.sampling.ewin == 10.0

    def test_crest_first(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
            [sampling]
            approach = "crest_first"
            prescreen_mode = "squick"
        """)
        )
        assert cfg.sampling.approach == "crest_first"
        assert cfg.sampling.prescreen_mode == "squick"

    def test_unknown_approach_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="Unknown sampling approach"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [sampling]
                approach = "magic"
            """)
            )


class TestRrho:
    def test_default_is_xtb(self, tmp_toml: Callable[[str], Path]) -> None:
        cfg = load_config(
            tmp_toml("""
            [molecule]
            smiles = "CCO"
        """)
        )
        assert cfg.refinement.rrho == "xtb"

    def test_invalid_raises(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"Unknown \[refinement\] rrho"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho = "cheap"
            """)
            )

    def test_psi4_hessians_warn(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        # Allowed, but a finite-difference Hessian per conformer dwarfs the
        # optimization it follows.
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho = "refinement"
                [compute]
                driver = "psi4"
            """)
            )
        assert cfg.refinement.rrho == "refinement"
        assert "finite-difference" in caplog.text


class TestDeprecatedSpellings:
    def test_refinement_driver_orbmol(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                driver = "orbmol"
            """)
            )
        assert cfg.refinement.method == "OrbMol-v2"
        assert cfg.refinement.engine == "orbmol"
        assert "[refinement] driver is deprecated" in caplog.text

    def test_refinement_driver_disagreeing_with_compute_raises(
        self, tmp_toml: Callable[[str], Path]
    ) -> None:
        with pytest.raises(ValueError, match="no longer selects a DFT backend"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                driver = "psi4"
                [compute]
                driver = "pyscf"
            """)
            )

    def test_rrho_method_dft(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                rrho_method = "dft"
                [scoring]
                solvent_model = "IEFPCM"
            """)
            )
        assert cfg.refinement.rrho == "refinement"
        assert "rrho_method = 'dft' is deprecated" in caplog.text

    def test_refinement_solvent_model(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent_model = "CPCM"
                [compute]
                driver = "pyscf"
            """)
            )
        assert cfg.refinement.solvation == "optimize"
        assert cfg.scoring.solvent_model == "CPCM"
        assert "[refinement] solvent_model = 'CPCM' is deprecated" in caplog.text

    def test_per_stage_solvent_moves_to_molecule(
        self, tmp_toml: Callable[[str], Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                [refinement]
                solvent = "methanol"
                [scoring]
                solvent = "methanol"
            """)
            )
        assert cfg.molecule.solvent == "methanol"
        assert "[refinement] solvent is deprecated" in caplog.text

    def test_conflicting_solvents_raise(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="a run has one solvent"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                solvent = "water"
                [scoring]
                solvent = "methanol"
            """)
            )


class TestExampleConfigs:
    @pytest.mark.parametrize(
        "name,driver,engine,solvation",
        [
            ("glycine_pyscf", "pyscf", "pyscf", "optimize"),
            ("glycine_psi4", "psi4", "psi4", "optimize"),
            ("glycine_orbmol", "pyscf", "orbmol", "single_point"),
            ("aminobutanol_crest_pyscf", "pyscf", "pyscf", "optimize"),
            ("aminobutanol_crest_psi4", "psi4", "psi4", "optimize"),
            ("aminobutanol_rdkit_pyscf", "pyscf", "pyscf", "optimize"),
            ("aminobutanol_rdkit_psi4", "psi4", "psi4", "optimize"),
        ],
    )
    def test_loads(self, name: str, driver: str, engine: str, solvation: str) -> None:
        cfg = load_config(Path(f"examples/{name}.toml"))
        assert cfg.compute.driver == driver
        assert cfg.refinement.engine == engine
        assert cfg.refinement.solvation == solvation

    def test_examples_use_current_spellings(self, caplog: pytest.LogCaptureFixture) -> None:
        # The deprecation shims exist for configs already on disk, not for the
        # examples people copy from.
        with caplog.at_level(logging.WARNING):
            for path in sorted(Path("examples").glob("*.toml")):
                load_config(path)
        assert "deprecated" not in caplog.text


class TestSolventElectronDensity:
    def test_water_is_the_value_cmirs_rounds_to_005(self) -> None:
        assert solvent_electron_density("water") == pytest.approx(0.0494, abs=5e-5)

    def test_the_fitted_organic_solvents_are_thinner_than_water(self) -> None:
        # Electrons per gram barely differ (0.555-0.574); the lower mass density decides.
        assert solvent_electron_density("octanol") == pytest.approx(0.0417, abs=5e-5)
        assert solvent_electron_density("Hexadecane") == pytest.approx(0.0394, abs=5e-5)

    def test_water_saturated_octanol_is_one_percent_denser_in_electrons(self) -> None:
        # 4.9% water by mass and 1.2% denser than dry octanol (J. Chem. Eng. Data 57, 2221).
        wet, dry = solvent_electron_density("woctanol"), solvent_electron_density("octanol")
        assert wet == pytest.approx(0.0421, abs=5e-5)
        assert wet / dry == pytest.approx(1.011, abs=1e-3)

    def test_a_solvent_without_a_liquid_density_raises(self) -> None:
        with pytest.raises(ValueError, match="No liquid density"):
            solvent_electron_density("methanol")


class TestAlpbName:
    def test_names_xtb_spells_differently_are_translated(self) -> None:
        assert alpb_name("Dichloromethane") == "ch2cl2"

    def test_other_names_pass_through(self) -> None:
        assert alpb_name("Water") == "water"

    @pytest.mark.slow
    @pytest.mark.parametrize("solvent", sorted(SOLVENT_DIELECTRIC))
    def test_xtb_knows_every_solvent_pcm_does(self, solvent: str) -> None:
        """Sampling and RRHO run in xtb's ALPB with the same solvent name the PCM
        takes its dielectric from; xtb stops on a name it has no parameters for."""
        water = Geometry(
            symbols=("O", "H", "H"),
            coords=np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]),
        )
        assert xtb_runner.single_point(water, solvent=solvent) < 0


class TestFixedSpecies:
    def test_off_by_default(self, tmp_toml: Callable[[str], Path]) -> None:
        assert not load_config(tmp_toml('[molecule]\nsmiles = "CCO"\n')).sampling.fixed_species

    def test_needs_one_charge_state(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="charge_min = charge_max"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CC(=O)O"
                charge_min = -1
                charge_max = 0
                [sampling]
                fixed_species = true
            """)
            )

    def test_needs_rdkit_first(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match="approach = 'rdkit_first'"):
            load_config(
                tmp_toml("""
                [molecule]
                smiles = "CCO"
                charge_min = 0
                charge_max = 0
                [sampling]
                approach = "crest_first"
                fixed_species = true
            """)
            )


class TestDropSolventModels:
    """MOIST's DROP cavities as solvent models "<PCM model>/<cavity>", on either driver."""

    @pytest.fixture()
    def moist_installed(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
        """Make the config see MOIST installed or not, in whichever environment this runs."""
        import importlib.util

        find_spec = importlib.util.find_spec

        def set_installed(installed: bool) -> None:
            def fake(name: str, *args: object, **kwargs: object) -> object:
                if name == "moist":
                    return object() if installed else None
                return find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

            monkeypatch.setattr(importlib.util, "find_spec", fake)

        return set_installed

    def _load(
        self,
        tmp_toml: Callable[[str], Path],
        model: str,
        *,
        driver: str = "pyscf",
        scoring: str = "",
        refinement: str = "",
    ) -> PkaConfig:
        return load_config(
            tmp_toml(f"""
                [molecule]
                smiles = "CCO"
                [refinement]
                {refinement}
                [scoring]
                solvent_model = "{model}"
                {scoring}
                [compute]
                driver = "{driver}"
            """)
        )

    @pytest.mark.parametrize(
        ("model", "parsed"),
        [
            ("IEFPCM/SvdW-DROP", ("IEF-PCM", "SvdW-DROP")),
            ("ief-pcm/SvdW-DROP", ("IEF-PCM", "SvdW-DROP")),
            ("PCM/rho-DROP", ("IEF-PCM", "rho-DROP")),
            ("COSMO/SvdW-DROP", ("COSMO", "SvdW-DROP")),
            # one name per cavity, written exactly
            ("CPCM/isodensity-DROP", None),
            ("IEFPCM/svdw-drop", None),
            ("IEFPCM/SVDWDROP", None),
            ("IEFPCM/Rho-DROP", None),
            ("IEFPCM", None),
            ("SMD/SvdW-DROP", None),
            ("IEFPCM/ISWIG", None),
        ],
    )
    def test_names(self, model: str, parsed: tuple[str, str] | None) -> None:
        from tautomator.config import drop_solvent_model

        assert drop_solvent_model(model) == parsed

    @pytest.mark.parametrize(
        "model", ["IEFPCM/SvdW-DROP", "IEF-PCM/rho-DROP", "CPCM/SvdW-DROP", "COSMO/rho-DROP"]
    )
    def test_accepted_on_pyscf(
        self,
        tmp_toml: Callable[[str], Path],
        moist_installed: Callable[[bool], None],
        model: str,
    ) -> None:
        moist_installed(True)
        config = self._load(tmp_toml, model)
        assert config.scoring.solvent_model == model
        assert config.refinement.solvation == "optimize"  # MOIST has analytical gradients
        assert config.provenance()["scoring"]["solvent_model"] == model

    @pytest.mark.parametrize("model", ["IEFPCM/SvdW-DROP", "CPCM/rho-DROP"])
    def test_accepted_on_psi4_without_moist_here(
        self,
        tmp_toml: Callable[[str], Path],
        moist_installed: Callable[[bool], None],
        model: str,
    ) -> None:
        # Psi4 imports MOIST in the psi4 environment, not in this one
        moist_installed(False)
        config = self._load(tmp_toml, model, driver="psi4")
        assert config.scoring.solvent_model == model
        assert config.refinement.solvation == "optimize"

    def test_hessian_in_solvent_on_psi4(
        self, tmp_toml: Callable[[str], Path], moist_installed: Callable[[bool], None]
    ) -> None:
        # Psi4's frequencies are finite differences of MOIST's analytical gradients
        moist_installed(False)
        config = self._load(
            tmp_toml, "IEFPCM/SvdW-DROP", driver="psi4", refinement='rrho = "refinement"'
        )
        assert config.refinement.rrho == "refinement"
        assert config.refinement.solvation == "optimize"

    def test_hydrogen_radius_refused_on_psi4_too(
        self, tmp_toml: Callable[[str], Path], moist_installed: Callable[[bool], None]
    ) -> None:
        moist_installed(False)
        with pytest.raises(ValueError, match="pcm_hydrogen_radius"):
            self._load(
                tmp_toml,
                "IEFPCM/rho-DROP",
                driver="psi4",
                scoring="pcm_hydrogen_radius = 1.2",
            )

    def test_every_cavity_and_model_has_its_moist_classes(self) -> None:
        # both drivers build the model from these; a name missing here would fail every job
        from tautomator.config import DROP_CAVITIES, DROP_COMPONENTS, DROP_LSF, PCM_METHODS

        assert set(DROP_LSF) == set(DROP_CAVITIES)
        assert set(DROP_COMPONENTS) == set(PCM_METHODS.values())

    def test_those_classes_are_moists(self) -> None:
        from tautomator.config import DROP_COMPONENTS, DROP_LSF

        moist = pytest.importorskip("moist")
        for name in [*DROP_LSF.values(), *DROP_COMPONENTS.values()]:
            assert isinstance(getattr(moist, name), type), name

    def test_unknown_cavity(
        self, tmp_toml: Callable[[str], Path], moist_installed: Callable[[bool], None]
    ) -> None:
        moist_installed(True)
        # spelled as the README spells them, not as the config normalizes them
        with pytest.raises(ValueError, match="not available on the pyscf driver") as err:
            self._load(tmp_toml, "IEFPCM/sphere-DROP")
        assert "SvdW-DROP or rho-DROP" in str(err.value)
        assert "SVDWDROP" not in str(err.value)

    @pytest.mark.parametrize("model", ["IEFPCM/isodensity-DROP", "IEFPCM/svdw-drop"])
    def test_one_name_per_cavity(
        self,
        tmp_toml: Callable[[str], Path],
        moist_installed: Callable[[bool], None],
        model: str,
    ) -> None:
        moist_installed(True)
        with pytest.raises(ValueError, match="not available on the pyscf driver"):
            self._load(tmp_toml, model)

    def test_needs_moist(
        self, tmp_toml: Callable[[str], Path], moist_installed: Callable[[bool], None]
    ) -> None:
        moist_installed(False)
        with pytest.raises(ValueError, match="needs MOIST"):
            self._load(tmp_toml, "IEFPCM/SvdW-DROP")

    @pytest.mark.parametrize("section", ["scoring", "refinement"])
    def test_hydrogen_radius_sizes_svdw_drops_spheres(
        self,
        tmp_toml: Callable[[str], Path],
        moist_installed: Callable[[bool], None],
        section: str,
    ) -> None:
        # SvdW-DROP is built on PySCF's cavity's spheres (config.pcm_radii)
        moist_installed(True)
        config = self._load(tmp_toml, "IEFPCM/SvdW-DROP", **{section: "pcm_hydrogen_radius = 1.2"})
        assert getattr(config, section).pcm_hydrogen_radius == 1.2

    @pytest.mark.parametrize("section", ["scoring", "refinement"])
    def test_hydrogen_radius_refused_on_rho_drop(
        self,
        tmp_toml: Callable[[str], Path],
        moist_installed: Callable[[bool], None],
        section: str,
    ) -> None:
        moist_installed(True)
        with pytest.raises(ValueError, match=f"\\[{section}\\] pcm_hydrogen_radius .* nothing"):
            self._load(tmp_toml, "IEFPCM/rho-DROP", **{section: "pcm_hydrogen_radius = 1.2"})

    def test_no_hessian_in_solvent(
        self, tmp_toml: Callable[[str], Path], moist_installed: Callable[[bool], None]
    ) -> None:
        moist_installed(True)
        with pytest.raises(ValueError, match="Hessian"):
            self._load(tmp_toml, "IEFPCM/SvdW-DROP", refinement='rrho = "refinement"')
        # a gas-phase Hessian (solvation as a single point) is fine
        config = self._load(
            tmp_toml,
            "IEFPCM/SvdW-DROP",
            refinement='rrho = "refinement"\nsolvation = "single_point"',
        )
        assert config.refinement.rrho == "refinement"


class TestUnknownSettings:
    """A key the loader does not read is an error, not a silent default."""

    def _load(self, tmp_toml: Callable[[str], Path], text: str) -> PkaConfig:
        return load_config(tmp_toml(f'[molecule]\nsmiles = "CCO"\n{text}\n'))

    @pytest.mark.parametrize(
        ("section", "key", "meant"),
        [
            ("molecule", "solvnet", "solvent"),
            ("sampling", "ewni", "ewin"),
            ("refinement", "rrhoo", "rrho"),
            ("scoring", "solvent_modle", "solvent_model"),
            ("compute", "memroy_gb", "memory_gb"),
        ],
    )
    def test_misspelled_setting(
        self, tmp_toml: Callable[[str], Path], section: str, key: str, meant: str
    ) -> None:
        text = f'{key} = "x"' if section == "molecule" else f'[{section}]\n{key} = "x"'
        with pytest.raises(ValueError, match=rf"\[{section}\] {key} \(did you mean '{meant}'\?\)"):
            self._load(tmp_toml, text)

    @pytest.mark.parametrize(
        ("text", "where"),
        [
            ("[scoring]\nthreads = 4", r"\[scoring\] threads \(a setting of \[compute\]\)"),
            ("[compute]\newin = 5.0", r"a setting of \[sampling\], \[refinement\] or \[scoring\]"),
        ],
    )
    def test_setting_of_another_section(
        self, tmp_toml: Callable[[str], Path], text: str, where: str
    ) -> None:
        with pytest.raises(ValueError, match=where):
            self._load(tmp_toml, text)

    def test_top_level(self, tmp_toml: Callable[[str], Path]) -> None:
        # a section's setting outside its section, and a misspelled section
        with pytest.raises(ValueError) as err:
            load_config(
                tmp_toml('memory_gb = 4\n[molecule]\nsmiles = "CCO"\n[scorng]\newin = 5.0\n')
            )
        message = str(err.value)
        assert "'memory_gb' at the top level (a setting of [compute])" in message
        assert "'scorng' at the top level (did you mean 'scoring'?)" in message

    def test_derived_field_is_not_a_setting(self, tmp_toml: Callable[[str], Path]) -> None:
        # the method decides the refinement engine
        with pytest.raises(ValueError, match=r"\[refinement\] engine"):
            self._load(tmp_toml, '[refinement]\nengine = "orbmol"')

    def test_removed_descriptor_switch_is_refused(self, tmp_toml: Callable[[str], Path]) -> None:
        # run_pipeline(descriptors=...) asks for them now; the old key must not pass silently
        with pytest.raises(ValueError, match=r"\[scoring\] nonelectrostatic_descriptors"):
            self._load(tmp_toml, "[scoring]\nnonelectrostatic_descriptors = true")

    def test_no_hint_without_a_likely_setting(self, tmp_toml: Callable[[str], Path]) -> None:
        with pytest.raises(ValueError, match=r"\[scoring\] zzz$"):
            self._load(tmp_toml, "[scoring]\nzzz = 1")

    @pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
    def test_presets_give_only_settings(self, protocol: str) -> None:
        for section, settings in PROTOCOLS[protocol].items():
            assert set(settings) <= _SETTINGS[section], (protocol, section)


class TestPcmRadii:
    """The one table of sphere radii every cavity is built on (both drivers, all cavities)."""

    def test_is_pyscfs_own_default(self) -> None:
        import pyscf.solvent.pcm as pcm

        np.testing.assert_array_equal(pcm_radii(), 1.2 * pcm.modified_Bondi)

    def test_hydrogen_radius_moves_hydrogen_alone(self) -> None:
        from pyscf.data.radii import BOHR

        default, wider = pcm_radii(1.1), pcm_radii(1.3)
        assert wider[1] == pytest.approx(1.2 * 1.3 / BOHR)
        np.testing.assert_array_equal(np.delete(default, 1), np.delete(wider, 1))
