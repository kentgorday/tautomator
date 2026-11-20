"""Tests for the xtb Hessian runner."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tautomator import xtb_runner
from tautomator.types import Geometry

# 3,5-Lutidine as wB97X-3c/vDZP settled it in hexadecane, just off C2v.  xtb
# 6.7.1 finds a C2 axis and one mirror plane in it, which match no point group,
# and dies with SIGTRAP when its point-group search falls back.
LUTIDINE = Geometry(
    symbols=("C", "C", "C", "H", "N", "C", "H", "C", "C", "H", "H", "H", "C", "H", "H", "H", "H"),
    coords=np.array(
        [
            [-2.5151843786, -0.9903160500, 0.0369906588],
            [-1.2021906397, -0.2732887312, 0.0084825076],
            [-1.1360892444, 1.1121005850, -0.0211373541],
            [-2.0503918828, 1.6953829103, -0.0271177736],
            [-0.0015744237, 1.7957515101, -0.0439618500],
            [1.1343265862, 1.1140535307, -0.0383367806],
            [2.0474333418, 1.6989114821, -0.0576420460],
            [1.2032578713, -0.2712377516, -0.0103810925],
            [2.5176865527, -0.9861637333, -0.0058323104],
            [2.6262582257, -1.5963352917, 0.8899993874],
            [2.6049918334, -1.6467654201, -0.8675119360],
            [3.3472644690, -0.2831438233, -0.0359010408],
            [0.0012255035, -0.9586182053, 0.0129769728],
            [0.0023357310, -2.0428492519, 0.0346611599],
            [-2.6030532664, -1.6769875171, -0.8038891384],
            [-3.3462254164, -0.2901036868, -0.0120948506],
            [-2.6205328616, -1.5731485067, 0.9513193655],
        ]
    ),
)


def test_a_hessian_run_skips_the_point_group_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def run(cmd: list[str], cwd: Path, **_: Any) -> Any:
        seen["cmd"] = cmd
        seen["xcontrol"] = Path(cmd[cmd.index("--input") + 1]).read_text()
        (Path(cwd) / "g98.out").write_text(" Frequencies --   100.0  200.0  300.0\n")

        class Done:
            returncode = 0
            stderr = ""

        return Done()

    monkeypatch.setattr("tautomator.xtb_runner.subprocess.run", run)
    freqs = xtb_runner.frequencies(LUTIDINE, biased=True, work_dir=tmp_path)
    assert freqs == [100.0, 200.0, 300.0]
    assert "--bhess" in seen["cmd"]
    assert seen["xcontrol"] == xtb_runner.SKIP_POINT_GROUP
    assert "maxat=0" in seen["xcontrol"]


@pytest.mark.slow
def test_a_near_symmetric_shape_gets_its_frequencies() -> None:
    freqs = xtb_runner.frequencies(LUTIDINE, solvent="hexadecane", biased=True)
    assert len(freqs) == 3 * LUTIDINE.n_atoms - 6
