"""Reproducible fixed-AGV-group C3 scan on the completed C1 scenario pool.

This is a parameter-selection diagnostic, not a robust certificate.  Each
candidate is solved as a finite-scenario two-stage master on every scenario
already discovered by formal C1.  The best feasible candidate is the one with
the smallest known-pool robust objective, so C3 receives the strongest fair
fixed split before its own exact adversary run.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import gurobipy as gp

from experiments.agv_operation_cost_scan_20260716.metrics import (
    measure_empty_running,
)
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.residuals import evaluate_residuals
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import load_case
from port_h2_certificate.reconstruct import reconstruct_recourse
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from runners.common import PROJECT_ROOT
from runners.input_adapters import build_joint_bundle


DEFAULT_SHARES = (50, 55, 60, 65, 70, 75, 80, 85)


def portable_manifest_path(path: Path) -> str:
    """Keep scan manifests reproducible without embedding a workstation path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def integer_group_split(
    fleet_size: int, container_share_percent: int
) -> tuple[int, int]:
    """Return integer container/LOHC group capacities using half-up rounding."""
    if fleet_size <= 0:
        raise ValueError("fleet_size must be positive")
    if not 0 < container_share_percent < 100:
        raise ValueError("container_share_percent must be between 0 and 100")
    raw = Decimal(fleet_size) * Decimal(container_share_percent) / Decimal(100)
    container = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return container, fleet_size - container


def select_best_candidate(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select the lowest-objective OPTIMAL known-pool candidate."""
    feasible = [row for row in candidates if row.get("status") == "OPTIMAL"]
    if not feasible:
        raise ValueError("no OPTIMAL C3 ratio candidate")
    return min(feasible, key=lambda row: float(row["objective_scaled"]))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("scan summary cannot be empty")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_scenario_pool(path: Path, bundle) -> tuple[dict[str, tuple[float, ...]], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("C1 scenario pool must be a nonempty JSON list")
    expected = set(bundle.outputs)
    realizations: list[dict[str, tuple[float, ...]]] = []
    for index, record in enumerate(payload):
        raw = record.get("realization") if isinstance(record, dict) else None
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"scenario pool entry {index} does not match bundle outputs")
        realization = {
            key: tuple(float(value) for value in raw[key]) for key in expected
        }
        for key, values in realization.items():
            if len(values) != len(bundle.outputs[key].nominal) or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError(f"scenario pool entry {index} output {key} is invalid")
        realizations.append(realization)
    return tuple(realizations)


def _candidate_case(
    source: dict[str, Any],
    *,
    fleet_size: int,
    share_percent: int,
) -> tuple[dict[str, Any], int, int]:
    container, lohc = integer_group_split(fleet_size, share_percent)
    payload = dict(source)
    payload.update(
        {
            "calibration_name": (
                f"wind800_12v_agv{fleet_size}_agv60_tou_base20mw_"
                f"c3_ctn{container}_lohc{lohc}_20260717"
            ),
            "case_name": "C3_WorkCapacityCapsRobust",
            "container_work_capacity": float(container),
            "lohc_work_capacity": float(lohc),
        }
    )
    return payload, container, lohc


def _evaluate_candidate(
    case_path: Path,
    profile: str,
    bundle,
    realizations,
    *,
    share_percent: int,
    container_capacity: int,
    lohc_capacity: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = load_case(case_path, profile)
    two_stage = TwoStageIR(
        build_first_stage_ir(loaded.case, bundle),
        build_recourse_ir(loaded.case, bundle),
    )
    compiled = compile_master(two_stage, realizations)
    started = time.perf_counter()
    try:
        master = compiled.solve()
    except RuntimeError:
        elapsed = time.perf_counter() - started
        status = int(compiled.model.Status)
        label = {
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
        }.get(status, f"GUROBI_STATUS_{status}")
        row = {
            "share_percent": share_percent,
            "container_capacity": container_capacity,
            "lohc_capacity": lohc_capacity,
            "status": label,
            "scenario_count": len(realizations),
            "objective_scaled": None,
            "theta_scaled": None,
            "wall_time_seconds": elapsed,
            "case_semantic_sha256": loaded.case_semantic_sha256,
            "joint_bundle_sha256": bundle.bundle_sha256,
            "max_abs_recourse_residual": None,
            "maximum_terminal_backlog_teu": None,
            "minimum_completed_container_teu": None,
            "minimum_lohc_export_kg_h2eq": None,
            "maximum_container_agv_used": None,
            "maximum_lohc_agv_used": None,
            "maximum_empty_vehicle_periods": None,
        }
        return row, {"summary": row, "scenarios": []}

    elapsed = time.perf_counter() - started
    scenario_details: list[dict[str, Any]] = []
    for index, (realization, recourse) in enumerate(
        zip(realizations, master.scenario_recourse, strict=True)
    ):
        residual = evaluate_residuals(
            two_stage.recourse,
            master.first_stage,
            realization,
            recourse,
        ).max_abs_residual
        reconstructed = reconstruct_recourse(
            loaded.case,
            bundle,
            master.first_stage,
            realization,
            recourse,
        )
        empty = measure_empty_running(loaded.case, realization, recourse)
        scenario_details.append(
            {
                "scenario_index": index,
                "recourse_objective_scaled": reconstructed.recourse_objective,
                "max_abs_recourse_residual": residual,
                "terminal_backlog_teu": reconstructed.series["backlog_tasks"][-1],
                "completed_container_teu": sum(reconstructed.series["task_done"]),
                "lohc_export_kg_h2eq": sum(
                    reconstructed.series.get("lohc_outbound_kg", ())
                ),
                "maximum_container_agv_used": max(
                    recourse[("agv_container_count", period)]
                    for period in range(loaded.case.profile.periods)
                ),
                "maximum_lohc_agv_used": max(
                    recourse[("agv_lohc_count", period)]
                    for period in range(loaded.case.profile.periods)
                ),
                "empty_vehicle_periods": empty.total_empty_vehicle_periods,
            }
        )
    row = {
        "share_percent": share_percent,
        "container_capacity": container_capacity,
        "lohc_capacity": lohc_capacity,
        "status": "OPTIMAL",
        "scenario_count": len(realizations),
        "objective_scaled": master.objective,
        "theta_scaled": master.theta,
        "wall_time_seconds": elapsed,
        "case_semantic_sha256": loaded.case_semantic_sha256,
        "joint_bundle_sha256": bundle.bundle_sha256,
        "max_abs_recourse_residual": max(
            item["max_abs_recourse_residual"] for item in scenario_details
        ),
        "maximum_terminal_backlog_teu": max(
            item["terminal_backlog_teu"] for item in scenario_details
        ),
        "minimum_completed_container_teu": min(
            item["completed_container_teu"] for item in scenario_details
        ),
        "minimum_lohc_export_kg_h2eq": min(
            item["lohc_export_kg_h2eq"] for item in scenario_details
        ),
        "maximum_container_agv_used": max(
            item["maximum_container_agv_used"] for item in scenario_details
        ),
        "maximum_lohc_agv_used": max(
            item["maximum_lohc_agv_used"] for item in scenario_details
        ),
        "maximum_empty_vehicle_periods": max(
            item["empty_vehicle_periods"] for item in scenario_details
        ),
    }
    return row, {
        "summary": row,
        "first_stage": {
            f"{key[0]}[{key[1]}]": value
            for key, value in sorted(master.first_stage.items())
        },
        "scenarios": scenario_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-c1", required=True, type=Path)
    parser.add_argument("--scenario-pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fleet-size", type=int, default=34)
    parser.add_argument("--shares", type=int, nargs="+", default=DEFAULT_SHARES)
    parser.add_argument("--profile", default="quarter_hour_96")
    args = parser.parse_args()

    source_path = args.source_c1.resolve()
    scenario_pool_path = args.scenario_pool.resolve()
    output = args.output.resolve()
    if not source_path.is_file() or not scenario_pool_path.is_file():
        raise FileNotFoundError("source C1 case and C1 scenario pool must exist")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty scan directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    bundle_loaded = load_case(source_path, args.profile)
    bundle = build_joint_bundle(bundle_loaded, output / "bundle")
    realizations = _load_scenario_pool(scenario_pool_path, bundle)

    rows: list[dict[str, Any]] = []
    candidate_paths: dict[int, Path] = {}
    for share in args.shares:
        payload, container, lohc = _candidate_case(
            source,
            fleet_size=args.fleet_size,
            share_percent=share,
        )
        candidate_path = source_path.parent / (
            f"C3_base20mw_agv34_ctn{container}_lohc{lohc}.yaml"
        )
        if candidate_path.exists():
            raise FileExistsError(f"refusing to overwrite C3 candidate: {candidate_path}")
        _atomic_json(candidate_path, payload)
        candidate_paths[share] = candidate_path
        row, detail = _evaluate_candidate(
            candidate_path,
            args.profile,
            bundle,
            realizations,
            share_percent=share,
            container_capacity=container,
            lohc_capacity=lohc,
        )
        rows.append(row)
        _atomic_json(output / f"share_{share}" / "result.json", detail)

    selected = select_best_candidate(rows)
    selected_path = source_path.parent / "C3_base20mw_agv34_selected.yaml"
    if selected_path.exists():
        raise FileExistsError(f"refusing to overwrite selected C3 case: {selected_path}")
    shutil.copy2(candidate_paths[int(selected["share_percent"])], selected_path)
    _atomic_csv(output / "scan_summary.csv", rows)
    _atomic_json(
        output / "scan_manifest.json",
        {
            "proof_scope": "C1_discovered_scenario_pool_parameter_diagnostic",
            "formal_robust_certificate": False,
            "source_c1_case": portable_manifest_path(source_path),
            "source_c1_scenario_pool": portable_manifest_path(
                scenario_pool_path
            ),
            "scenario_count": len(realizations),
            "fleet_size": args.fleet_size,
            "share_candidates_percent": list(args.shares),
            "integer_rounding_rule": "ROUND_HALF_UP for container; LOHC is fleet minus container",
            "selection_rule": (
                "minimum finite-scenario two-stage robust objective among OPTIMAL "
                "candidates on the complete formal-C1 discovered scenario pool"
            ),
            "selection_rationale": (
                "This gives the fixed-group C3 ablation its strongest fair split; "
                "the selected case must still pass its own exact adversary certificate."
            ),
            "selected_candidate": selected,
            "selected_case": portable_manifest_path(selected_path),
            "joint_bundle_sha256": bundle.bundle_sha256,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
