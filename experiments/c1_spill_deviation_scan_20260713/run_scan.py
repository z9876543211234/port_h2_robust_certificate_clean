"""Run the formal C1 spill-deviation cost scan without solver time limits.

The day-ahead and real-time spill prices are fixed at the migrated C1 value
of 0.1 yuan/kWh.  Only the new absolute RT-vs-DA spill-deviation price is
varied.  Every candidate uses the same wind and discrete ship-delay bundles.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = EXPERIMENT_ROOT / "runs"
DEFAULT_CANDIDATES = (0.01, 0.025, 0.05, 0.075, 0.10)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = list(rows[0]) if rows else ["spill_deviation_per_kwh"]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _candidate_label(value: float) -> str:
    return f"cdev_{value:.3f}".replace(".", "p")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _series_energy(rows: Iterable[dict[str, str]], field: str, dt_hours: float) -> float:
    return sum(float(row[field]) for row in rows if row.get(field) not in (None, "")) * dt_hours


def _series_max(rows: Iterable[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return max(values, default=0.0)


def _summarize_candidate(value: float, output: Path, execution: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "spill_deviation_per_kwh": value,
        "run_directory": str(output.relative_to(PROJECT_ROOT)),
        "exit_code": execution["exit_code"],
        "wall_time_seconds": execution["wall_time_seconds"],
        "certificate_status": "missing",
        "engineering_optimal": False,
    }
    certificate_path = output / "certificate.json"
    if not certificate_path.is_file():
        return row

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    resolved = json.loads((output / "resolved_case.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "model_stats.json").read_text(encoding="utf-8"))
    costs = json.loads((output / "cost_breakdown.json").read_text(encoding="utf-8"))
    day_ahead = _read_csv(output / "dispatch_day_ahead.csv")
    recourse = _read_csv(output / "dispatch_recourse_worst.csv")
    horizon = int(certificate["model_horizon"])
    recourse = [row for row in recourse if int(row["period"]) < horizon]
    dt_hours = float(resolved["profile"]["dt_hours"])
    reconstructed = costs["reconstructed_recourse"]

    row.update(
        {
            "certificate_status": certificate["status"],
            "engineering_optimal": certificate["engineering_optimal"],
            "lower_bound": certificate["lower_bound"],
            "upper_bound": certificate["upper_bound"],
            "gap_abs": certificate["gap_abs"],
            "gap_rel": certificate["gap_rel"],
            "max_abs_residual": certificate["max_abs_residual"],
            "ccg_iterations": stats["ccg_iterations"],
            "joint_bundle_sha256": certificate["joint_bundle_sha256"],
            "day_ahead_spill_mwh": _series_energy(day_ahead, "spill_da_mw", dt_hours),
            "worst_real_time_spill_mwh": _series_energy(recourse, "spill_rt_mw", dt_hours),
            "worst_spill_deviation_mwh": _series_energy(recourse, "spill_deviation_mw", dt_hours),
            "max_spill_deviation_mw": _series_max(recourse, "spill_deviation_mw"),
            "worst_grid_deviation_mwh": _series_energy(recourse, "grid_deviation_mw", dt_hours),
            "spill_real_time_cost": reconstructed["spill_real_time"],
            "spill_deviation_cost": reconstructed["spill_deviation"],
        }
    )
    return row


def _build_command(
    case: str | Path,
    value: float,
    output: Path,
    grid_buy_capacity_kw: float | None,
    cost_partition_keys: tuple[str, ...] = (),
) -> list[str]:
    overrides = [
        "deterministic.cost.spill_day_ahead_per_kwh=0.1",
        "deterministic.cost.spill_real_time_per_kwh=0.1",
        f"deterministic.cost.spill_deviation_per_kwh={value}",
        "ship_delay.max_delay_steps=4",
        "ship_delay.delayed_ship_budget=2",
    ]
    if grid_buy_capacity_kw is not None:
        overrides.append(
            f"deterministic.grid.buy_capacity_kw={grid_buy_capacity_kw}"
        )
    command = [
        sys.executable,
        "-m",
        "runners.run_case",
        str(case),
        "--profile",
        "quarter_hour_96",
        "--output",
        str(output),
    ]
    for override in overrides:
        command.extend(("--set", override))
    for key in cost_partition_keys:
        command.extend(("--cost-partition-key", key))
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="C1",
        help="Built-in case name or path to an isolated case specification.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=float,
        default=DEFAULT_CANDIDATES,
        help="Nonzero spill-deviation prices in yuan/kWh.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--grid-buy-capacity-kw",
        type=float,
        default=None,
        help="Optional isolated capacity override; formal input is not modified.",
    )
    parser.add_argument(
        "--output-group",
        default="runs",
        help="Single directory name under the experiment root.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Re-run candidates that already have formal certificates.",
    )
    parser.add_argument(
        "--cost-partition-key",
        action="append",
        default=[],
        help=(
            "Primary selector used by the exact cost-adversary partition; "
            "repeat for multiple keys."
        ),
    )
    args = parser.parse_args()
    candidates = tuple(float(value) for value in args.candidates)
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ValueError("all spill-deviation candidates must be strictly positive")
    if args.grid_buy_capacity_kw is not None and args.grid_buy_capacity_kw <= 0.0:
        raise ValueError("grid-buy capacity override must be positive")
    output_group = Path(args.output_group)
    if output_group.is_absolute() or len(output_group.parts) != 1 or output_group.name in {".", ".."}:
        raise ValueError("output-group must be one relative directory name")
    runs_root = EXPERIMENT_ROOT / output_group
    artifact_suffix = "" if output_group.name == "runs" else f"_{output_group.name.removeprefix('runs_')}"

    runs_root.mkdir(parents=True, exist_ok=True)
    execution_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))
    )
    _atomic_json(
        EXPERIMENT_ROOT / f"scan_settings{artifact_suffix}.json",
        {
            "case": args.case,
            "profile": "quarter_hour_96",
            "spill_day_ahead_per_kwh": 0.1,
            "spill_real_time_per_kwh": 0.1,
            "spill_deviation_candidates_per_kwh": list(candidates),
            "grid_buy_capacity_kw_override": args.grid_buy_capacity_kw,
            "ship_max_delay_steps": 4,
            "ship_delayed_ship_budget": 2,
            "ship_total_delay_step_budget": None,
            "wind_budget": 12,
            "cost_partition_branch_keys": list(args.cost_partition_key),
            "solver_time_limit_used": False,
            "output_group": output_group.name,
        },
    )

    for value in candidates:
        output = runs_root / _candidate_label(value)
        output.mkdir(parents=True, exist_ok=True)
        command = _build_command(
            args.case,
            value,
            output,
            args.grid_buy_capacity_kw,
            tuple(args.cost_partition_key),
        )
        print(f"[scan] candidate={value:.3f} yuan/kWh", flush=True)
        if args.dry_run:
            print("[scan] " + " ".join(command), flush=True)
            continue

        certificate_exists = (output / "certificate.json").is_file()
        if certificate_exists and not args.rerun_complete:
            execution = {
                "spill_deviation_per_kwh": value,
                "exit_code": 0,
                "wall_time_seconds": 0.0,
                "started_at_utc": None,
                "finished_at_utc": None,
                "skipped_existing_certificate": True,
                "command": command,
            }
            print("[scan] existing formal certificate reused", flush=True)
        else:
            started_at = datetime.now(timezone.utc)
            started = time.monotonic()
            with (output / "solver.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            finished_at = datetime.now(timezone.utc)
            execution = {
                "spill_deviation_per_kwh": value,
                "exit_code": completed.returncode,
                "wall_time_seconds": time.monotonic() - started,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "skipped_existing_certificate": False,
                "command": command,
            }
            print(
                f"[scan] exit={completed.returncode} wall={execution['wall_time_seconds']:.1f}s",
                flush=True,
            )

        execution_records.append(execution)
        summary_rows.append(_summarize_candidate(value, output, execution))
        _atomic_json(
            EXPERIMENT_ROOT / f"scan_execution{artifact_suffix}.json",
            execution_records,
        )
        _atomic_csv(
            EXPERIMENT_ROOT / f"scan_summary{artifact_suffix}.csv", summary_rows
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
