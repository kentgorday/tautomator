"""scripts/run_mnsol_set.py's choice of models (the script itself needs the MNSol set)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tautomator.scoring import check_descriptor_models


def _script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_mnsol_set.py"
    spec = importlib.util.spec_from_file_location("run_mnsol_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["IEFPCM/SvdW-DROP", "IEFPCM", "IEF-PCM", "IEFPCM/rho-DROP"])
def test_default_descriptor_models_leave_out_the_runs_own(model: str) -> None:
    script = _script()
    others = script.default_descriptor_models(model)
    assert len(others) == len(script.CAVITY_MODELS) - 1
    check_descriptor_models(model, others)  # what run_pipeline would check


def test_the_default_run_saves_the_other_two_cavities() -> None:
    script = _script()
    assert script.default_descriptor_models(script.SOLVENT_MODEL) == ["IEFPCM", "IEFPCM/rho-DROP"]
