"""Tests for stage-level resumption in run_pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from tautomator.ensemble import (
    SCHEMA_VERSION,
    load_ensemble,
    schema_version,
    serialize_ensemble,
)
from tautomator.pipeline import _find_resume_point
from tautomator.types import ChargeState, Conformer, Ensemble, Geometry, Microstate


def _ensemble(smiles: str = "CCO") -> Ensemble:
    geom = Geometry(symbols=("O", "H"), coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]))
    conf = Conformer(geometry=geom, electronic_energy=-1.0)
    ms = Microstate(tautomer_id="abc123", conformers=[conf], smiles=smiles)
    ens = Ensemble(input_smiles=smiles)
    ens.charge_states[0] = ChargeState(charge=0, microstates=[ms])
    return ens


class TestFindResumePoint:
    def test_none_when_no_checkpoints(self, tmp_path: Path) -> None:
        assert _find_resume_point(tmp_path, "CCO") is None

    def test_finds_sampling(self, tmp_path: Path) -> None:
        serialize_ensemble(_ensemble(), tmp_path / "sampling")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        _ens, stage = found
        assert stage == "sampling"

    def test_refinement_supersedes_sampling(self, tmp_path: Path) -> None:
        # both present -> resume from the later stage, not the earlier one
        serialize_ensemble(_ensemble(), tmp_path / "sampling")
        serialize_ensemble(_ensemble(), tmp_path / "refinement")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        _ens, stage = found
        assert stage == "refinement"

    def test_conformers_round_trip(self, tmp_path: Path) -> None:
        serialize_ensemble(_ensemble(), tmp_path / "refinement")
        found = _find_resume_point(tmp_path, "CCO")
        assert found is not None
        ens, _stage = found
        conf = ens.charge_states[0].microstates[0].conformers[0]
        assert conf.electronic_energy == -1.0
        assert conf.geometry.symbols == ("O", "H")

    def test_smiles_mismatch_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A reused output dir (an older run's, numbered by CSV row) can hold a
        # checkpoint for a different molecule entirely.
        serialize_ensemble(_ensemble("CCO"), tmp_path / "refinement")
        assert _find_resume_point(tmp_path, "CCN") is None
        assert "but this config asks for" in caplog.text


class TestSchemaVersionGate:
    """A checkpoint predating a schema bump must not resume silently.

    `load_ensemble` fills new fields with defaults so old files still read, but
    resuming from one would reload every conformer at multiplicity 1.0 and
    reproduce the old unweighted answer with no sign that anything was lost.
    """

    def _downgrade(self, path: Path) -> None:
        """Rewrite a checkpoint as a pre-multiplicity one."""
        raw = json.loads(path.read_text())
        raw.pop("schema_version", None)
        for cs in raw["charge_states"].values():
            for ms in cs["microstates"]:
                for conf in ms["conformers"]:
                    conf.pop("multiplicity", None)
        path.write_text(json.dumps(raw))

    def test_current_schema_resumes(self, tmp_path: Path) -> None:
        path = serialize_ensemble(_ensemble(), tmp_path / "refinement")
        assert schema_version(path) == SCHEMA_VERSION
        assert _find_resume_point(tmp_path, "CCO") is not None

    def test_legacy_checkpoint_is_refused(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = serialize_ensemble(_ensemble(), tmp_path / "refinement")
        self._downgrade(path)
        assert schema_version(path) == 1
        with caplog.at_level(logging.WARNING):
            assert _find_resume_point(tmp_path, "CCO") is None
        assert "schema v1" in caplog.text

    def test_legacy_checkpoint_still_loads(self, tmp_path: Path) -> None:
        """Refusing to resume is not the same as refusing to read."""
        path = serialize_ensemble(_ensemble(), tmp_path / "refinement")
        self._downgrade(path)
        loaded = load_ensemble(path)
        assert loaded.input_smiles == "CCO"
        conf = loaded.charge_states[0].microstates[0].conformers[0]
        assert conf.multiplicity == 1.0
