from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_SCRIPT = (
    PROJECT_ROOT / "experiments/c1_spill_deviation_scan_20260713/run_scan.py"
)


def _load_scan_module():
    spec = importlib.util.spec_from_file_location("spill_scan_runner", SCAN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_command_accepts_an_isolated_case_path(tmp_path) -> None:
    module = _load_scan_module()
    case_path = tmp_path / "C1_large_port_12v.yaml"

    command = module._build_command(case_path, 0.05, tmp_path / "run", None)

    run_case_index = command.index("runners.run_case")
    assert command[run_case_index + 1] == str(case_path)
    assert "deterministic.cost.spill_deviation_per_kwh=0.05" in command


def test_build_command_forwards_exact_cost_partition_keys(tmp_path) -> None:
    module = _load_scan_module()
    branch_keys = ("wind.down[1]", "wind.down[9]")

    command = module._build_command(
        "C1", 0.05, tmp_path / "run", None, branch_keys
    )

    pairs = list(zip(command, command[1:]))
    assert pairs.count(("--cost-partition-key", "wind.down[1]")) == 1
    assert pairs.count(("--cost-partition-key", "wind.down[9]")) == 1
