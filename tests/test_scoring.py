"""What scoring saves for the nonelectrostatic fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tautomator import pyscf_nonelectrostatic, pyscf_runner
from tautomator.ensemble import HARTREE_TO_KCAL
from tautomator.nonelectrostatic import Descriptors, load_descriptors
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
        if x == 9.0:
            raise RuntimeError("SCF did not converge")
        return gas(geom) - 0.01, _descriptors(x)

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
        score(ens, "pyscf", "hf", "sto-3g", "IEFPCM", "water", ewin=10.0, descriptors_path=path)
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
        score(ens, "pyscf", "hf", "sto-3g", "IEFPCM", "water", descriptors_path=path)
        saved, _settings = load_descriptors(path)
        for kept, record in zip(
            ens.charge_states[0].microstates[0].conformers, saved, strict=False
        ):
            assert record.free_energy == pytest.approx(kept.free_energy, abs=1e-12)

    def test_without_a_path_nothing_is_saved(self, tmp_path: Path) -> None:
        ens = _ensemble()
        score(ens, "pyscf", "hf", "sto-3g", "IEFPCM", "water")
        assert not list(tmp_path.iterdir())
        assert len(ens.charge_states[0].microstates[0].conformers) == 2

    @pytest.mark.parametrize(("driver", "model"), [("psi4", "IEFPCM"), ("pyscf", None)])
    def test_needs_solvated_pyscf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str, model: str | None
    ) -> None:
        from tautomator import psi4_runner

        monkeypatch.setattr(psi4_runner, "psi4_executable", lambda: "psi4")
        with pytest.raises(ValueError, match="solvated PySCF"):
            score(
                _ensemble(),
                driver,
                "hf",
                "sto-3g",
                model,
                "water",
                descriptors_path=tmp_path / "d.npz",
            )
