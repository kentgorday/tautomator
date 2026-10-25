"""What the pipeline hands its stages: solvent, species and descriptor settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tautomator import pipeline
from tautomator.config import PkaConfig, load_config
from tautomator.ensemble import serialize_ensemble
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


class _Stop(Exception):
    """Raised by the stubbed sampling stage once it has seen its arguments."""


@pytest.mark.parametrize(("scoring", "expected"), [("", "woctanol"), ('cavity = "none"', None)])
def test_a_gas_phase_run_samples_in_the_gas_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoring: str, expected: str | None
) -> None:
    """A run scored in the gas phase samples without ALPB, so its conformers are
    gas-phase minima and its free energy a gas-phase one throughout."""
    seen: dict[str, object] = {}

    def fake_approach1(**kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pipeline, "run_approach1", fake_approach1)
    toml = tmp_path / "config.toml"
    toml.write_text(
        f"""protocol = "high_quality"
[molecule]
smiles = "CC(=O)[O-]"
charge_min = -1
charge_max = -1
solvent = "woctanol"
[sampling]
fixed_species = true
[scoring]
{scoring}
[compute]
output_dir = "{tmp_path / "out"}"
"""
    )
    with pytest.raises(_Stop):
        pipeline.run_pipeline(load_config(toml))
    assert seen["solvent"] == expected
    assert seen["fixed_species"] is True


@pytest.mark.parametrize(
    ("descriptors", "cavities"),
    [(None, ()), ([], ()), (["ISWIG", "rho-DROP"], ("ISWIG", "rho-DROP"))],
)
def test_scoring_saves_descriptors_when_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptors: list[str] | None,
    cavities: tuple[str, ...],
) -> None:
    """The descriptors (asked for by the caller, not the config) go next to the run's
    ensemble.json, on the run's cavity and each further one."""
    seen: dict[str, object] = {}

    def fake_score(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(pipeline, "score", fake_score)
    out = tmp_path / "out"
    _refinement_checkpoint(out)
    with pytest.raises(_Stop):
        pipeline.run_pipeline(_config(tmp_path, out), resume=True, descriptors=descriptors)
    assert seen["descriptors_dir"] == (None if descriptors is None else out)
    assert seen["descriptor_cavities"] == cavities


@pytest.mark.parametrize(
    ("driver", "scoring", "descriptors", "match"),
    [
        ("psi4", "", [], "driver = 'pyscf'"),
        ("pyscf", '[scoring]\ncavity = "none"', [], "need a solvated run"),
        ("pyscf", "", ["SMD"], "not a cavity"),
        ("pyscf", "", ["IEFPCM/rho-DROP"], "not a cavity"),
        ("pyscf", "", ["SvdW-DROP"], "the run's own"),
        ("pyscf", "", ["ISWIG", "ISWIG"], "listed twice"),
    ],
)
def test_descriptor_request_is_checked_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
    scoring: str,
    descriptors: list[str],
    match: str,
) -> None:
    def fail(**kwargs: object) -> None:
        raise AssertionError("a stage ran")

    monkeypatch.setattr(pipeline, "run_approach1", fail)
    with pytest.raises(ValueError, match=match):
        pipeline.run_pipeline(
            _config(tmp_path, tmp_path / "out", scoring, driver), descriptors=descriptors
        )


def _refinement_checkpoint(out: Path) -> None:
    geom = Geometry(("O", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]))
    ens = Ensemble("[OH-]")
    ens.charge_states[-1] = ChargeState(-1, [Microstate("[OH-]", [Conformer(geom, -75.0)])])
    serialize_ensemble(ens, out / "refinement")


def _config(tmp_path: Path, out: Path, extra: str = "", driver: str = "pyscf") -> PkaConfig:
    toml = tmp_path / "config.toml"
    toml.write_text(
        f"""protocol = "high_quality"
[molecule]
smiles = "[OH-]"
charge_min = -1
charge_max = -1
[sampling]
fixed_species = true
{extra}
[compute]
driver = "{driver}"
output_dir = "{out}"
"""
    )
    return load_config(toml)
