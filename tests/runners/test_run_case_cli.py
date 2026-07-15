from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_case_cli_exposes_repeatable_cost_partition_key() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))
    )

    completed = subprocess.run(
        [sys.executable, "-m", "runners.run_case", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--cost-partition-key COST_PARTITION_KEY" in completed.stdout

