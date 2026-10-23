"""Tests for the Psi4 DFT runner: input decks, scratch, the executable, and the PCM."""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tautomator import psi4_runner, refinement, scoring
from tautomator.config import (
    DEFAULT_MEMORY_GB,
    DROP_PRUNE_LEVEL,
    PCM_SWITCHING,
    solvent_dielectric,
)
from tautomator.psi4_runner import (
    _df_basis_block,
    _pcm_block,
    _run_psi4,
    _scratch_dir,
    psi4_executable,
)
from tautomator.types import Geometry


def _water() -> Geometry:
    return Geometry(
        symbols=("O", "H", "H"),
        coords=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
    )


class TestPCMBlock:
    """The input lines that turn psi4-swig-pcm on, with the model pyscf_runner builds."""

    @pytest.mark.parametrize(
        ("model", "method"),
        [
            ("IEFPCM", "IEF-PCM"),
            ("PCM", "IEF-PCM"),
            ("CPCM", "C-PCM"),
            ("C-PCM", "C-PCM"),
            ("COSMO", "COSMO"),
        ],
    )
    def test_names_the_model(self, model: str, method: str) -> None:
        assert f"method={method!r}" in _pcm_block(model, "water")

    def test_dielectric_is_the_one_pyscf_gets(self) -> None:
        assert f"eps={solvent_dielectric('methanol')!r}" in _pcm_block("IEFPCM", "Methanol")

    def test_switching_is_the_one_pyscf_gets(self) -> None:
        assert f"switching={PCM_SWITCHING!r}" in _pcm_block("IEFPCM", "water")

    def test_hydrogen_radius_is_a_free_parameter(self) -> None:
        assert "modified_bondi_radii(hydrogen=1.1)" in _pcm_block("IEFPCM", "water")
        assert "modified_bondi_radii(hydrogen=1.3)" in _pcm_block("IEFPCM", "water", 1.3)

    def test_is_python(self) -> None:
        block = _pcm_block("IEFPCM", "water")
        compile(block, "<pcm block>", "exec")
        assert "psi4_swig_pcm.enable(psi4_swig_pcm.PCMSettings(" in block

    def test_unknown_model_or_solvent_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Psi4 solvent model"):
            _pcm_block("SMD", "water")
        with pytest.raises(ValueError, match="Unknown solvent"):
            _pcm_block("IEFPCM", "ammonia")

    def test_every_job_enables_it_before_computing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs: list[str] = []

        def fake_run(input_text: str, work_dir: Path, **_kwargs: object) -> str:
            inputs.append(input_text)
            (work_dir / "optimized.xyz").write_text("3\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
            (work_dir / "frequencies.dat").write_text("")
            return "=== FINAL ENERGY: -1.000000000000 ===\n=== CONVERGED: 1 ===\n"

        monkeypatch.setattr(psi4_runner, "_run_psi4", fake_run)
        solvated = ("PBE0", "def2-SVP", "IEFPCM", "water")
        psi4_runner.single_point(_water(), 0, *solvated)
        psi4_runner.optimize(_water(), 0, *solvated)
        psi4_runner.frequencies(_water(), 0, *solvated)
        psi4_runner.single_point(_water(), 0, "PBE0", "def2-SVP")
        for text, job in zip(inputs[:3], ("energy(", "optimize(", "frequency("), strict=True):
            assert text.index("psi4_swig_pcm.enable(") < text.index(job)
            assert "pcm true" not in text  # Psi4's own PCMSolver stays off
        assert "psi4_swig_pcm" not in inputs[3]


class TestMoistBlock:
    """A DROP solvent model: psi4-swig-pcm's MOIST host, with pyscf_runner's MOIST model."""

    @pytest.mark.parametrize(
        ("model", "component", "lsf"),
        [
            ("IEFPCM/SvdW-DROP", "ModelComponentIEFPCM", "SvdW"),
            ("PCM/rho-DROP", "ModelComponentIEFPCM", "Isodensity"),
            ("C-PCM/isodensity-DROP", "ModelComponentCPCM", "Isodensity"),
            ("COSMO/svdw-drop", "ModelComponentCOSMO", "SvdW"),
        ],
    )
    def test_names_the_model_and_cavity(self, model: str, component: str, lsf: str) -> None:
        block = _pcm_block(model, "water")
        assert f"components=(moist.{component}({solvent_dielectric('water')!r}),)" in block
        assert f"cavity=moist.DROP(lsf=moist.{lsf}()," in block

    def test_pruned_as_pyscf_runner_prunes(self) -> None:
        block = _pcm_block("IEFPCM/SvdW-DROP", "water")
        assert f"DROPParameters(wleb_prune_level={DROP_PRUNE_LEVEL})" in block

    def test_is_python_on_moists_own_radii(self) -> None:
        block = _pcm_block("IEFPCM/SvdW-DROP", "water")
        compile(block, "<moist block>", "exec")
        assert "psi4_swig_pcm.enable(MoistSettings(" in block
        assert "modified_bondi_radii" not in block and "PCMSettings" not in block

    def test_hydrogen_radius_raises(self) -> None:
        # as on pyscf_runner: MOIST's cavity has radii of its own
        with pytest.raises(ValueError, match="pcm_hydrogen_radius"):
            _pcm_block("IEFPCM/SvdW-DROP", "water", 1.3)

    def test_unknown_cavity_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Psi4 solvent model"):
            _pcm_block("IEFPCM/sphere-DROP", "water")

    def test_enabled_before_every_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs: list[str] = []

        def fake_run(input_text: str, work_dir: Path, **kwargs: object) -> str:
            inputs.append(input_text)
            (work_dir / "optimized.xyz").write_text("3\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
            (work_dir / "frequencies.dat").write_text("")
            return "=== FINAL ENERGY: -1.000000000000 ===\n=== CONVERGED: 1 ===\n"

        monkeypatch.setattr(psi4_runner, "_run_psi4", fake_run)
        solvated = ("PBE0", "def2-SVP", "IEFPCM/SvdW-DROP", "water")
        psi4_runner.single_point(_water(), 0, *solvated)
        psi4_runner.optimize(_water(), 0, *solvated)
        psi4_runner.frequencies(_water(), 0, *solvated)
        for text, job in zip(inputs, ("energy(", "optimize(", "frequency("), strict=True):
            assert text.index("MoistSettings(") < text.index(job)


def _thioacetate() -> Geometry:
    return Geometry(
        symbols=("C", "C", "O", "S", "H", "H", "H"),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.52, 0.0, 0.0],
                [2.13, 1.07, 0.0],
                [2.36, -1.50, 0.0],
                [-0.38, 1.02, 0.0],
                [-0.38, -0.51, 0.89],
                [-0.38, -0.51, -0.89],
            ]
        ),
    )


def _gbs_shells(block: str, elem: str) -> list[list[object]]:
    """One element's shells from a Psi4 basis block, in PySCF's [l, [exp, coef], ...] form."""
    # An element's entry opens with "Cl     0"; a shell with "S    6   1.00".
    entry = re.split(rf"(?m)^{elem}\s+0\s*$", block)[1].split("****", 1)[0]
    lines = entry.strip().splitlines()
    shells: list[list[object]] = []
    while lines:
        letter, n_prim, _scale = lines.pop(0).split()
        rows = [
            [float(x.replace("D", "E")) for x in lines.pop(0).split()] for _ in range(int(n_prim))
        ]
        shells.append(["SPDFGHI".index(letter), *rows])
    return shells


class TestFittingBasisBlock:
    """The extra s primitive of config.AUX_DIFFUSE_S, as Psi4 is told about it."""

    def test_nothing_to_add_means_no_block(self) -> None:
        assert _df_basis_block("def2-QZVPPD", _water()) == ""

    @pytest.mark.parametrize("basis", ["def2-QZVPPD", "def2-SVP", "def2-mTZVPP", "vDZP"])
    def test_bases_psi4_fits_with_that_set_get_it(self, basis: str) -> None:
        assert "def2-universal-jkfit-plus-s" in _df_basis_block(basis, _thioacetate())

    @pytest.mark.parametrize("basis", ["cc-pVTZ", "aug-cc-pVTZ", "6-31+G*", "pcseg-2"])
    def test_other_bases_keep_psi4s_own_fit(self, basis: str) -> None:
        # Psi4 fits these with a JKFIT set of its own, which for an augmented
        # basis does better by small anions than ours; see config.AUXBASIS.
        assert _df_basis_block(basis, _thioacetate()) == ""

    def test_only_those_elements_get_a_set_of_their_own(self) -> None:
        block = _df_basis_block("def2-QZVPPD", _thioacetate())
        assert 'key="df_basis_scf"' in block
        assigns = [line for line in block.splitlines() if line.startswith("assign")]
        assert assigns == ["assign def2-universal-jkfit", "assign S def2-universal-jkfit-plus-s"]

    def test_carries_the_functions_pyscf_fits_with(self) -> None:
        # Psi4's shells come from basis-set-exchange and PySCF's from its own
        # library; both backends must end up with the same fit, primitive included.
        pytest.importorskip("pyscf")
        from tautomator.config import AUX_DIFFUSE_S
        from tautomator.pyscf_runner import _auxbasis

        elements = tuple(AUX_DIFFUSE_S)
        geom = Geometry(symbols=elements, coords=np.zeros((len(elements), 3)))
        block = _df_basis_block("def2-QZVPPD", geom)
        pyscf_aux = _auxbasis(elements)
        assert isinstance(pyscf_aux, dict)
        for elem in elements:
            assert _gbs_shells(block, elem) == pyscf_aux[elem], elem

    def test_every_kind_of_job_carries_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs: list[str] = []

        def fake_run(input_text: str, work_dir: Path, **_kwargs: object) -> str:
            inputs.append(input_text)
            (work_dir / "optimized.xyz").write_text("2\n\nCl 0.0 0.0 0.0\nH 0.0 0.0 1.27\n")
            (work_dir / "frequencies.dat").write_text("")
            return "=== FINAL ENERGY: -1.000000000000 ===\n=== CONVERGED: 1 ===\n"

        monkeypatch.setattr(psi4_runner, "_run_psi4", fake_run)
        # Two atoms: optimize and frequencies return at once for a single atom.
        hcl = Geometry(symbols=("Cl", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.27]]))
        psi4_runner.single_point(hcl, 0, "PBE0", "def2-SVP")
        psi4_runner.optimize(hcl, 0, "PBE0", "def2-SVP")
        psi4_runner.frequencies(hcl, 0, "PBE0", "def2-SVP")
        assert len(inputs) == 3
        for text in inputs:
            assert _df_basis_block("def2-SVP", hcl) in text


@pytest.fixture
def isolated_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tempfile at tmp_path so scratch never lands in the real TMPDIR.

    tempfile.gettempdir() caches its answer on first call, so setting the
    TMPDIR env var after interpreter start has no effect; the module-level
    tempfile.tempdir must be patched instead.  Without this, tests that keep
    scratch on purpose leak a directory into the real temp dir on every run.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class TestScratchCleanup:
    """Psi4 scratch must not leak.

    Orphaned scratch is what filled the disk during the first pKa-set
    run: on the out-of-core DFJK path a single job writes several GB of
    three-index integrals, and nothing ever deleted the directories.
    """

    @staticmethod
    def _stub_psi4(tmp_path: Path, body: str) -> None:
        exe = tmp_path / "psi4"
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)

    def test_removed_on_success(self, isolated_tmpdir: Path) -> None:
        with _scratch_dir() as tmp:
            captured = Path(tmp)
            assert captured.exists()
        assert not captured.exists()

    def test_removed_on_exception(self, isolated_tmpdir: Path) -> None:
        captured = None
        with pytest.raises(RuntimeError, match="boom"), _scratch_dir() as tmp:
            captured = Path(tmp)
            raise RuntimeError("boom")
        assert captured is not None and not captured.exists()

    def test_kept_when_env_set(
        self, isolated_tmpdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TAUTOMATOR_KEEP_SCRATCH", "1")
        with _scratch_dir() as tmp:
            captured = Path(tmp)
        # retained on purpose -- isolated_tmpdir keeps it inside pytest's
        # tmp_path so it is still cleaned up with the rest of the test run
        assert captured.exists()
        assert captured.parent == isolated_tmpdir


class TestMemoryFlag:
    """memory_gb is passed as psi4's --memory CLI flag, not baked into input."""

    @staticmethod
    def _recording_psi4(tmp_path: Path) -> None:
        exe = tmp_path / "psi4"
        # record argv, then write a minimal parseable output to $3 (the -o path)
        exe.write_text(
            "#!/bin/sh\n"
            f'echo "$@" > {tmp_path}/argv.txt\n'
            f'cp "$1" {tmp_path}/seen_input.dat\n'
            'echo "=== FINAL ENERGY: -1.000000000000 ===" > "$3"\n'
        )
        exe.chmod(0o755)

    def test_memory_flag_present(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp), memory_gb=12.0)
        argv = (tmp_path / "argv.txt").read_text()
        assert "--memory 12.0GB" in argv
        # the input file itself stays clean
        assert "memory" not in (tmp_path / "seen_input.dat").read_text()

    def test_default_memory_always_passed(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Psi4 must never silently fall back to its own 500 MiB default.
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
        assert f"--memory {DEFAULT_MEMORY_GB}GB" in (tmp_path / "argv.txt").read_text()

    def test_scratch_pinned_to_work_dir(
        self, isolated_tmpdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # -s keeps psi4's binary scratch inside the dir we delete
        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
        self._recording_psi4(tmp_path)
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
            assert f"-s {tmp}" in (tmp_path / "argv.txt").read_text()


# ---------------------------------------------------------------------------
# Which psi4 gets launched
# ---------------------------------------------------------------------------
class TestPsi4Executable:
    """Psi4 has a pixi environment of its own, beside the one the tests run in."""

    @staticmethod
    def _psi4_in(env: Path, body: str = "") -> Path:
        exe = env / "bin" / "psi4"
        exe.parent.mkdir(parents=True)
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
        return exe

    @pytest.fixture()
    def envs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A directory of environments, with the interpreter placed in envs/default."""
        envs = tmp_path / "envs"
        monkeypatch.setattr(sys, "prefix", str(envs / "default"))
        monkeypatch.setenv("PATH", str(tmp_path / "elsewhere" / "bin"))
        return envs

    def test_prefers_its_own_environment_to_path(self, envs: Path, tmp_path: Path) -> None:
        # the locked Psi4 must not be shadowed by whatever else is on PATH
        sibling = self._psi4_in(envs / "psi4")
        self._psi4_in(tmp_path / "elsewhere")
        assert psi4_executable() == sibling

    def test_falls_back_to_path(self, envs: Path, tmp_path: Path) -> None:
        on_path = self._psi4_in(tmp_path / "elsewhere")
        assert psi4_executable() == on_path

    def test_missing_psi4_says_how_to_install_it(self, envs: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pixi install -e psi4"):
            psi4_executable()

    def test_missing_psi4_stops_the_stage(self, envs: Path) -> None:
        # refine() and score() wrap each conformer in a handler that records the
        # failure and moves on; a Psi4 that is not installed must not get that far
        with pytest.raises(FileNotFoundError):
            refinement._get_driver("psi4")
        with pytest.raises(FileNotFoundError):
            scoring._get_driver("psi4")

    def test_a_drop_model_needs_a_psi4_that_imports_moist(
        self, envs: Path, isolated_tmpdir: Path
    ) -> None:
        # a psi4 environment installed before it had MOIST, or a psi4 on PATH without it
        # (Psi4 writes the traceback to its output file, the third argument)
        self._psi4_in(
            envs / "psi4", 'echo "ModuleNotFoundError: No module named \'moist\'" > "$3"\nexit 1'
        )
        psi4_runner.check_installed()  # no DROP model: Psi4 is not launched
        with pytest.raises(
            ImportError, match=r"\(ModuleNotFoundError: No module named 'moist'\).*-e psi4"
        ):
            psi4_runner.check_installed(drop=True)
        with pytest.raises(ImportError, match="pixi install -e psi4"):
            refinement._get_driver("psi4", drop=True)
        with pytest.raises(ImportError, match="pixi install -e psi4"):
            scoring._get_driver("psi4", drop=True)

    def test_the_moist_check_imports_what_a_drop_job_does(
        self, envs: Path, tmp_path: Path, isolated_tmpdir: Path
    ) -> None:
        self._psi4_in(envs / "psi4", f'/bin/cp "$1" {tmp_path}/checked.dat')
        psi4_runner.check_installed(drop=True)
        checked = (tmp_path / "checked.dat").read_text()
        assert "import moist\n" in checked
        assert checked in _pcm_block("IEFPCM/SvdW-DROP", "water")

    def test_stages_on_a_drop_model_check_for_moist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tautomator.config import RefinementConfig, Solvation
        from tautomator.types import Ensemble

        checked: list[bool] = []
        monkeypatch.setattr(
            psi4_runner, "check_installed", lambda drop=False: checked.append(drop)
        )
        drop, pcm = "IEFPCM/SvdW-DROP", "IEFPCM"
        runs: list[tuple[str, Solvation]] = [(drop, "optimize"), (drop, "none"), (pcm, "optimize")]
        for model, solvation in runs:
            config = RefinementConfig(
                method="PBE0", basis="def2-SVP", solvation=solvation, engine="psi4"
            )
            refinement.refine(Ensemble("O"), config, driver="psi4", solvent_model=model)
        scoring.score(Ensemble("O"), "psi4", "PBE0", "def2-SVP", drop, "water")
        scoring.score(Ensemble("O"), "psi4", "PBE0", "def2-SVP")
        # Once per stage, for the Psi4 that both optimizes and adds solvation; a gas-phase
        # refinement checks too, for the scoring that will need MOIST.
        assert checked == [True, True, False, True, False]

    def test_psi4_runs_with_its_own_bin_first_on_path(
        self, envs: Path, tmp_path: Path, isolated_tmpdir: Path
    ) -> None:
        # Psi4 finds helpers such as mctc-gcp on PATH; they are installed beside it
        exe = self._psi4_in(
            envs / "psi4",
            f'echo "$PATH" > {tmp_path}/path.txt\n'
            'echo "=== FINAL ENERGY: -1.000000000000 ===" > "$3"',
        )
        with _scratch_dir() as tmp:
            _run_psi4("E = energy('scf')", Path(tmp))
        seen = (tmp_path / "path.txt").read_text().strip().split(":")
        assert seen[0] == str(exe.parent)
        assert str(tmp_path / "elsewhere" / "bin") in seen[1:]


# ---------------------------------------------------------------------------
# PCM diagnostics, and real Psi4 jobs in implicit solvent
# ---------------------------------------------------------------------------
def test_run_psi4_passes_on_a_gauss_law_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """psi4-swig-pcm only warns on a Gauss's-law violation; it must not vanish with stderr."""

    class _Result:
        returncode = 0
        stdout = ""
        stderr = "PCM nuclear surface charge violates Gauss's law by 1.20e-02 (...)\n"

    monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: tmp_path / "psi4")
    monkeypatch.setattr("tautomator.psi4_runner.subprocess.run", lambda *a, **k: _Result())
    (tmp_path / "output.dat").write_text("=== FINAL ENERGY: -1.0 ===")
    with caplog.at_level(logging.WARNING):
        psi4_runner._run_psi4("", tmp_path)
    assert "violates Gauss's law by 1.20e-02" in caplog.text


def _formate() -> Geometry:
    return Geometry(
        symbols=("C", "O", "O", "H"),
        coords=np.array(
            [[0.0, 0.0, 0.52], [1.13, 0.0, -0.05], [-1.13, 0.0, -0.05], [0.0, 0.0, 1.66]]
        ),
    )


def _require_psi4() -> None:
    try:
        psi4_executable()
    except FileNotFoundError:
        pytest.skip("no psi4 environment (pixi install -e psi4)")


@pytest.mark.slow
def test_solvation_energy_matches_pyscf(isolated_tmpdir: Path) -> None:
    """Both drivers build the same PCM: same surface, radii, dielectric and model."""
    _require_psi4()
    from tautomator import pyscf_runner

    level = ("PBE0", "def2-SVP")
    solvation = {}
    for name, driver in (("psi4", psi4_runner), ("pyscf", pyscf_runner)):
        e_pcm = driver.single_point(_formate(), -1, *level, "IEFPCM", "water", threads=4)
        e_gas = driver.single_point(_formate(), -1, *level, threads=4)
        solvation[name] = (e_pcm - e_gas) * 627.509474
    assert solvation["psi4"] < -60.0  # an anion in water
    # measured < 1e-5 kcal/mol (PCMSolver, Psi4's own PCM, was ~0.15 off)
    assert abs(solvation["psi4"] - solvation["pyscf"]) < 1e-3, solvation


@pytest.mark.slow
def test_optimizes_and_takes_frequencies_in_solvent(isolated_tmpdir: Path) -> None:
    """Analytical PCM gradients: optking minimizes in solvent, frequencies are real."""
    _require_psi4()
    solvated = ("HF", "def2-SVP", "IEFPCM", "water")
    start = psi4_runner.single_point(_water(), 0, *solvated, threads=4)
    geom, energy, converged = psi4_runner.optimize(_water(), 0, *solvated, threads=4)
    assert converged
    assert energy < start
    freqs = psi4_runner.frequencies(geom, 0, *solvated, threads=4)
    assert len(freqs) == 3
    assert all(f > 1000.0 for f in freqs)


def _require_psi4_with_moist() -> None:
    """Skip without Psi4.  Its environment has MOIST: one that does not fails here."""
    _require_psi4()
    psi4_runner.check_installed(drop=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "basis", "tolerance"),
    [
        # measured < 5e-9 kcal/mol
        ("IEFPCM/SvdW-DROP", "def2-SVP", 1e-6),
        ("CPCM/rho-DROP", "def2-SVP", 1e-6),
        # vDZP's ECPs: the two codes' ECP integrals put the gas-phase energies 2e-4 kcal/mol
        # apart, the solvation energies 3e-6 (the ECP-reduced charges and rho-DROP's core fill)
        ("IEFPCM/rho-DROP", "vDZP", 2e-5),
    ],
)
def test_drop_solvation_energy_matches_pyscf(
    isolated_tmpdir: Path, model: str, basis: str, tolerance: float
) -> None:
    """psi4-swig-pcm's MOIST host and pyscf_moist solve the same model on the same cavity.

    Runs in the moist environment (PySCF's side imports MOIST in process), with the psi4
    environment beside it.
    """
    _require_psi4_with_moist()
    pytest.importorskip("moist")
    from tautomator import pyscf_runner

    level = ("HF", basis)  # no DFT grid, which the two drivers build differently
    solvation = {}
    for name, driver in (("psi4", psi4_runner), ("pyscf", pyscf_runner)):
        e_pcm = driver.single_point(_formate(), -1, *level, model, "water", threads=4)
        e_gas = driver.single_point(_formate(), -1, *level, threads=4)
        solvation[name] = (e_pcm - e_gas) * 627.509474
    assert solvation["psi4"] < -60.0  # an anion in water
    assert abs(solvation["psi4"] - solvation["pyscf"]) < tolerance, solvation


@pytest.mark.slow
def test_optimizes_and_takes_frequencies_on_a_drop_cavity(isolated_tmpdir: Path) -> None:
    """MOIST's analytical gradients: optking minimizes on SvdW-DROP, and the frequencies
    (finite differences of those gradients; PySCF has none on MOIST) are real."""
    _require_psi4_with_moist()
    solvated = ("HF", "def2-SVP", "IEFPCM/SvdW-DROP", "water")
    start = psi4_runner.single_point(_water(), 0, *solvated, threads=4)
    geom, energy, converged = psi4_runner.optimize(_water(), 0, *solvated, threads=4)
    assert converged
    assert energy < start
    freqs = psi4_runner.frequencies(geom, 0, *solvated, threads=4)
    assert len(freqs) == 3
    assert all(f > 1000.0 for f in freqs)
