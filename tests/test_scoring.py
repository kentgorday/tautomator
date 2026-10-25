"""What scoring saves for the nonelectrostatic fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tautomator import pyscf_nonelectrostatic, pyscf_runner
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.nonelectrostatic import Descriptors, descriptor_file, load_descriptors
from tautomator.scoring import score
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


def _geometry(x: float) -> Geometry:
    return Geometry(("O", "H", "H"), np.array([[x, 0.0, 0.0], [x, 0.0, 0.96], [x, 0.93, -0.24]]))


def _descriptors(x: float) -> Descriptors:
    return Descriptors(
        dispersion=x,
        exchange=2 * x,
        areas=np.ones(4),
        normal_field=np.full(4, x),
        charge_field=np.full(4, -x),
        electrons_outside=0.1,
        element_atoms=np.array([0, 0, 1, 2]),
    )


@pytest.fixture()
def fake_pyscf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Energies set by the oxygen's x coordinate: x = 9 fails, x = 5 is far above the rest."""

    def gas(geom: Geometry, *args: Any, **kwargs: Any) -> float:
        return float(-76.0 + geom.coords[0, 0] * 20.0 / HARTREE_TO_KCAL)

    def solvated(geom: Geometry, *args: Any, **kwargs: Any) -> tuple[float, Descriptors]:
        x = geom.coords[0, 0]
        cavity = args[3]
        if x == 9.0 or (cavity == "rho-DROP" and x == 0.1):
            raise RuntimeError("SCF did not converge")
        shift = {"ISWIG": 0.01, "SvdW-DROP": 0.015, "rho-DROP": 0.02}[cavity]
        return gas(geom) - shift, _descriptors(x + shift - 0.01)

    monkeypatch.setattr(pyscf_runner, "single_point", gas)
    monkeypatch.setattr(pyscf_nonelectrostatic, "single_point_with_descriptors", solvated)


def _ensemble() -> Ensemble:
    confs = [
        Conformer(_geometry(x), rrho_correction=0.02, multiplicity=0.5) for x in (0, 0.1, 5, 9)
    ]
    ms = Microstate("O", confs, smiles="[H]O[H]", includes_enantiomer=True)
    ens = Ensemble("O")
    ens.charge_states[0] = ChargeState(0, [ms])
    return ens


@pytest.mark.usefixtures("fake_pyscf")
class TestDescriptorCapture:
    def test_saves_every_conformer_scored_in_solution(self, tmp_path: Path) -> None:
        ens = _ensemble()
        path = tmp_path / "nonelectrostatic.npz"
        score(ens, "pyscf", "hf", "sto-3g", "ISWIG", "water", ewin=10.0, descriptors_dir=tmp_path)
        # the window keeps the two low conformers; the file has the high one too
        kept = ens.charge_states[0].microstates[0].conformers
        assert [c.geometry.coords[0, 0] for c in kept] == [0.0, 0.1]
        saved, settings = load_descriptors(path)
        assert [c.geometry.coords[0, 0] for c in saved] == [0.0, 0.1, 5.0]
        for c in saved:
            x = c.geometry.coords[0, 0]
            assert c.descriptors.dispersion == x
            assert c.solvation_energy == pytest.approx(-0.01)
            assert c.electronic_energy == pytest.approx(-76.0 + x * 20.0 / HARTREE_TO_KCAL)
            assert (c.rrho_correction, c.multiplicity, c.states) == (0.02, 0.5, 1.0)
            assert (c.charge, c.microstate) == (0, "O")
        assert settings["solvent"] == "water"
        assert settings["solvent_rho"] == pytest.approx(0.0494, abs=1e-4)
        assert settings["delta"] == 7.0

    def test_energies_match_the_ensemble(self, tmp_path: Path) -> None:
        ens = _ensemble()
        path = tmp_path / "nonelectrostatic.npz"
        score(ens, "pyscf", "hf", "sto-3g", "ISWIG", "water", descriptors_dir=tmp_path)
        saved, _settings = load_descriptors(path)
        for kept, record in zip(
            ens.charge_states[0].microstates[0].conformers, saved, strict=False
        ):
            assert record.free_energy == pytest.approx(kept.free_energy, abs=1e-12)

    def test_without_a_path_nothing_is_saved(self, tmp_path: Path) -> None:
        ens = _ensemble()
        score(ens, "pyscf", "hf", "sto-3g", "ISWIG", "water")
        assert not list(tmp_path.iterdir())
        assert len(ens.charge_states[0].microstates[0].conformers) == 2

    @pytest.mark.parametrize(("driver", "cavity"), [("psi4", "ISWIG"), ("pyscf", None)])
    def test_needs_solvated_pyscf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str, cavity: str | None
    ) -> None:
        from tautomator import psi4_runner

        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: "psi4")
        with pytest.raises(ValueError, match="solvated PySCF"):
            score(
                _ensemble(),
                driver,
                "hf",
                "sto-3g",
                cavity,
                "water",
                descriptors_dir=tmp_path,
            )

    def test_further_cavities_get_files_of_their_own(self, tmp_path: Path) -> None:
        """Each further cavity's PCM energy and descriptors, at the run's geometries, in its
        own file; a conformer that fails there is left out of that file only."""
        ens = _ensemble()
        score(
            ens,
            "pyscf",
            "hf",
            "sto-3g",
            "ISWIG",
            "water",
            descriptors_dir=tmp_path,
            descriptor_cavities=("SvdW-DROP", "rho-DROP"),
        )
        # the run itself is the run's cavity's: rho-DROP's failure drops nothing
        kept = ens.charge_states[0].microstates[0].conformers
        assert [c.geometry.coords[0, 0] for c in kept] == [0.0, 0.1]
        main, main_settings = load_descriptors(tmp_path / descriptor_file())
        assert main_settings["descriptor_cavities"] == ["SvdW-DROP", "rho-DROP"]
        assert (main_settings["pcm"], main_settings["cavity"]) == ("IEF-PCM", "ISWIG")
        svdw, svdw_settings = load_descriptors(tmp_path / descriptor_file("SvdW-DROP"))
        rho, rho_settings = load_descriptors(tmp_path / descriptor_file("rho-DROP"))
        assert (svdw_settings["cavity"], svdw_settings["geometry_cavity"]) == (
            "SvdW-DROP",
            "ISWIG",
        )
        assert svdw_settings["failed"] == []
        assert [c.geometry.coords[0, 0] for c in svdw] == [0.0, 0.1, 5.0]
        assert [c.geometry.coords[0, 0] for c in rho] == [0.0, 5.0]
        assert [f["geometry"][0][0] for f in rho_settings["failed"]] == [0.1]
        assert rho_settings["drop_nleb"] == 434
        for a, b in zip(main, svdw, strict=True):
            assert b.electronic_energy == a.electronic_energy  # one gas-phase single point
            assert b.solvation_energy == pytest.approx(-0.015)
            assert b.descriptors.dispersion == pytest.approx(a.descriptors.dispersion + 0.005)

    def test_names_of_the_files(self) -> None:
        assert descriptor_file() == "nonelectrostatic.npz"
        assert descriptor_file("rho-DROP") == "nonelectrostatic_rhodrop.npz"
        assert descriptor_file("SvdW-DROP") == "nonelectrostatic_svdwdrop.npz"
        assert descriptor_file("ISWIG") == "nonelectrostatic_iswig.npz"
