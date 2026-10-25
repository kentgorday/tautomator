"""scripts/run_mnsol_set.py's choice of cavities (the script itself needs the MNSol set)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tautomator.config import CAVITIES, DEFAULT_CAVITY
from tautomator.scoring import check_descriptor_cavities


def _script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_mnsol_set.py"
    spec = importlib.util.spec_from_file_location("run_mnsol_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cavity", CAVITIES)
def test_default_descriptor_cavities_leave_out_the_runs_own(cavity: str) -> None:
    others = _script().default_descriptor_cavities(cavity)
    assert len(others) == len(CAVITIES) - 1
    check_descriptor_cavities(cavity, others)  # what run_pipeline would check


def test_the_default_run_saves_the_other_two_cavities() -> None:
    assert _script().default_descriptor_cavities(DEFAULT_CAVITY) == ["rho-DROP", "ISWIG"]


def test_a_solvent_run_writes_its_cavity_and_a_gas_run_none() -> None:
    script = _script()
    args = ("CCO", 0)
    water = script.make_toml(*args, "water", Path("out"), "pyscf", 1, 4.0, "fast", "ISWIG")
    gas = script.make_toml(*args, "gas", Path("out"), "pyscf", 1, 4.0, "fast", "ISWIG")
    assert 'cavity = "ISWIG"' in water and "solvent_model" not in water
    assert 'cavity = "none"' in gas
