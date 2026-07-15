"""Run C1-C4 sequentially through the same strict runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=("hourly_24", "quarter_hour_96"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    root = Path(args.output_root)
    for index in range(1, 5):
        command = [
            sys.executable,
            "-m",
            "runners.run_case",
            f"C{index}",
            "--profile",
            args.profile,
            "--output",
            str(root / f"C{index}"),
        ]
        for item in args.set:
            command.extend(["--set", item])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
