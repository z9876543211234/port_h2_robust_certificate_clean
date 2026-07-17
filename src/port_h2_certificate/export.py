"""Atomic certificate and dispatch export for completed formal C&CG runs."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import gurobipy as gp
import numpy as np

from port_h2_certificate.compilers.residuals import evaluate_residuals
from port_h2_certificate.reconstruct import reconstruct_recourse
from port_h2_certificate.schema import CaseData
from port_h2_certificate.solver.c4_stress import C4StressResult
from port_h2_certificate.solver.ccg import (
    CcgIteration,
    CcgResult,
    ScenarioRecord,
    realization_sha256,
)
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.hashing import canonical_sha256
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


_DELAY_KEY = re.compile(r"^ship\.delay\[(\d+),(\d+),(\d+)]$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _period_rows(values, horizon: int) -> tuple[list[str], list[dict[str, Any]]]:
    families = sorted({key[0] for key in values})
    rows = []
    for period in range(horizon):
        row: dict[str, Any] = {"period": period}
        for family in families:
            key = (family, period)
            if key in values:
                row[family] = values[key]
        rows.append(row)
    return ["period", *families], rows


def _ship_assignment_rows(selector, source, horizon):
    current = source.get("current_day_arrival_count", [0] * horizon)
    rows = []
    for nominal_period, count in enumerate(current):
        for ship_index in range(int(count)):
            delay = 0
            for key, value in selector.items():
                match = _DELAY_KEY.fullmatch(key)
                if (
                    value == 1
                    and match
                    and int(match.group(1)) == nominal_period
                    and int(match.group(2)) == ship_index
                ):
                    delay = int(match.group(3))
                    break
            absolute = nominal_period + delay
            rows.append(
                {
                    "nominal_source_period": nominal_period,
                    "anonymous_ship_index": ship_index,
                    "delay_steps": delay,
                    "actual_day_offset": absolute // horizon,
                    "actual_arrival_period": absolute % horizon,
                }
            )
    return rows


def _three_day_arrival_rows(selector, source, horizon):
    previous = [int(value) for value in source.get("previous_day_arrival_count", [0] * horizon)]
    current = [int(value) for value in source.get("current_day_arrival_count", [0] * horizon)]
    following = [int(value) for value in source.get("next_day_arrival_count", [0] * horizon)]
    actual = previous + current + following
    for key, value in selector.items():
        match = _DELAY_KEY.fullmatch(key)
        if value != 1 or match is None:
            continue
        source_period = int(match.group(1))
        delay = int(match.group(3))
        origin = horizon + source_period
        destination = origin + delay
        actual[origin] -= 1
        if destination < len(actual):
            actual[destination] += 1
    return [
        {
            "three_day_index": index,
            "day_offset": index // horizon - 1,
            "period_in_day": index % horizon,
            "actual_arrival_count": value,
        }
        for index, value in enumerate(actual)
    ]


def export_ccg_run(
    output_directory: str | Path,
    *,
    case: CaseData,
    two_stage_ir: TwoStageIR,
    bundle: UncertaintyBundle,
    result: CcgResult,
    git_commit: str,
    ship_delay_source: Mapping[str, Any] | None = None,
) -> None:
    if not result.engineering_optimal or result.status != "engineering_optimal":
        raise ValueError("formal certificate export requires engineering_optimal result")
    if not result.worst_realization or not result.worst_recourse_variables:
        raise ValueError("formal certificate export requires worst-scenario replay")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    horizon = case.profile.periods
    source = ship_delay_source or {}
    semantic_hash = canonical_sha256(case.to_dict())
    residuals = evaluate_residuals(
        two_stage_ir.recourse,
        result.first_stage,
        result.worst_realization,
        result.worst_recourse_variables,
    )
    row_by_name = {row.name: row for row in two_stage_ir.recourse.constraints}
    original_unit_residuals: dict[str, float] = {}
    for name, value in residuals.violations.items():
        row = row_by_name[name]
        factor = 1.0
        if row.unit == "MW":
            factor = 1000.0
        elif row.unit == "p.u." and row.component == "hydrogen":
            assert case.hydrogen is not None
            factor = case.hydrogen.capacity_kg
        elif row.unit == "p.u." and row.component == "lohc":
            assert case.lohc is not None
            factor = case.lohc.capacity_kg
        elif row.unit == "p.u." and row.component == "logistics":
            factor = case.logistics.backlog_capacity
        original_unit_residuals[name] = value * factor
    reconstructed = reconstruct_recourse(
        case,
        bundle,
        result.first_stage,
        result.worst_realization,
        result.worst_recourse_variables,
    )

    certificate = {
        "schema_version": 1,
        "case_name": case.case_name,
        "profile_name": case.profile.profile_name,
        "status": result.status,
        "engineering_optimal": result.engineering_optimal,
        "model_horizon": horizon,
        "ship_builder_horizon": 3 * horizon,
        "ship_projection_window": [horizon, 2 * horizon],
        "master_status": "OPTIMAL",
        "phase1_adversary_status": "OPTIMAL",
        "cost_adversary_status": "OPTIMAL",
        "partition_oracle_used": result.partition_oracle_used,
        "all_partition_leaves_optimal_or_empty": (
            result.all_partition_leaves_optimal_or_empty
        ),
        "cost_partition_branch_keys": list(
            result.cost_partition_branch_keys
        ),
        "cost_partition_leaf_count": result.cost_partition_leaf_count,
        "uncertainty_coverage_complete": result.uncertainty_coverage_complete,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "gap_abs": result.gap_abs,
        "gap_rel": result.gap_rel,
        "outer_gap_tolerance": 0.01,
        "accepted_limited_solve": result.accepted_limited_solve,
        "time_limit_used_for_certificate": False,
        "primal_replay_passed": result.primal_replay_passed,
        "residual_check_passed": result.residual_check_passed,
        "max_abs_residual": residuals.max_abs_residual,
        "bounds_ordered": result.bounds_ordered,
        "joint_bundle_sha256": bundle.bundle_sha256,
        "case_semantic_sha256": semantic_hash,
        "git_commit": git_commit,
        "gurobi_version": ".".join(str(value) for value in gp.gurobi.version()),
    }
    atomic_write_json(destination / "certificate.json", certificate)
    atomic_write_json(
        destination / "run_metadata.json",
        {
            "solver": "gurobipy",
            "git_commit": git_commit,
            "accepted_limited_solve": False,
            "time_limit_used": False,
        },
    )
    atomic_write_json(destination / "resolved_case.json", case.to_dict())
    atomic_write_json(
        destination / "wind_bundle_reference.json",
        {
            "bundle_sha256": bundle.proofs.get("wind_bundle_sha256"),
            "source_hashes": {
                key: value
                for key, value in bundle.source_hashes.items()
                if key.startswith("wind.")
            },
        },
    )
    atomic_write_json(
        destination / "ship_bundle_reference.json",
        {
            "bundle_sha256": bundle.proofs.get("ship_bundle_sha256"),
            "source_hashes": {
                key: value
                for key, value in bundle.source_hashes.items()
                if key.startswith("ship_delay.")
            },
        },
    )
    atomic_write_json(
        destination / "joint_bundle_reference.json",
        {
            "bundle_sha256": bundle.bundle_sha256,
            "profile_name": bundle.profile_name,
            "model_horizon": bundle.model_horizon,
            "proofs": bundle.proofs,
        },
    )

    first_cost_by_component: dict[str, float] = {}
    for spec in two_stage_ir.first_stage.variables:
        first_cost_by_component[spec.component] = (
            first_cost_by_component.get(spec.component, 0.0)
            + spec.objective_coefficient * result.first_stage[spec.key]
        )
    atomic_write_json(
        destination / "cost_breakdown.json",
        {
            "day_ahead_by_component": first_cost_by_component,
            "recourse_by_component": result.worst_cost_breakdown,
            "reconstructed_recourse": reconstructed.cost_breakdown,
        },
    )
    atomic_write_json(
        destination / "model_stats.json",
        {
            "first_stage_variables": len(two_stage_ir.first_stage.variables),
            "first_stage_constraints": len(two_stage_ir.first_stage.constraints),
            "recourse_variables_per_scenario": len(two_stage_ir.recourse.variables),
            "recourse_constraints_per_scenario": len(two_stage_ir.recourse.constraints),
            "bundle_selectors": len(bundle.selectors),
            "bundle_constraints": len(bundle.constraints),
            "ccg_iterations": len(result.history),
        },
    )
    atomic_write_json(
        destination / "residuals.json",
        {
            "max_abs_residual": residuals.max_abs_residual,
            "scaled_space": residuals.violations,
            "original_physical_units": original_unit_residuals,
            "max_original_unit_residual": max(
                original_unit_residuals.values(), default=0.0
            ),
            "original_unit_residuals_available": True,
        },
    )
    terminal = {
        key: values[-1]
        for key, values in reconstructed.series.items()
        if key in {"hydrogen_inventory_kg", "lohc_inventory_kg", "backlog_tasks", "soc"}
    }
    atomic_write_json(destination / "terminal_states.json", terminal)

    scenario_rows = [
        {
            "realization_sha256": record.realization_sha256,
            "selector_sha256": canonical_sha256(record.selector),
        }
        for record in result.scenarios
    ]
    atomic_write_csv(
        destination / "scenario_pool.csv",
        ["realization_sha256", "selector_sha256"],
        scenario_rows,
    )
    worst_rows = []
    for period in range(horizon):
        row = {"period": period}
        for key, values in sorted(result.worst_realization.items()):
            if period < len(values):
                row[key] = values[period]
        worst_rows.append(row)
    atomic_write_csv(
        destination / "worst_scenario.csv",
        ["period", *sorted(result.worst_realization)],
        worst_rows,
    )

    assignment_rows = _ship_assignment_rows(result.worst_selector, source, horizon)
    assignment_fields = [
        "nominal_source_period",
        "anonymous_ship_index",
        "delay_steps",
        "actual_day_offset",
        "actual_arrival_period",
    ]
    atomic_write_csv(destination / "ship_delay_assignment.csv", assignment_fields, assignment_rows)
    arrival = result.worst_realization.get("ship.arrival_count", np.zeros(horizon))
    atomic_write_csv(
        destination / "actual_arrival_current_day.csv",
        ["period", "actual_arrival_count"],
        [
            {"period": period, "actual_arrival_count": arrival[period]}
            for period in range(horizon)
        ],
    )
    three_day_rows = _three_day_arrival_rows(result.worst_selector, source, horizon)
    atomic_write_csv(
        destination / "actual_arrival_three_day_audit.csv",
        ["three_day_index", "day_offset", "period_in_day", "actual_arrival_count"],
        three_day_rows,
    )

    fields, rows = _period_rows(result.first_stage, horizon)
    atomic_write_csv(destination / "dispatch_day_ahead.csv", fields, rows)
    recourse_series = dict(reconstructed.series)
    raw_families = sorted({key[0] for key in result.worst_recourse_variables})
    recourse_rows = []
    all_columns = sorted(set(raw_families) | set(recourse_series))
    for period in range(horizon + 1):
        row: dict[str, Any] = {"period": period}
        for family in raw_families:
            key = (family, period)
            if key in result.worst_recourse_variables:
                row[family] = result.worst_recourse_variables[key]
        for family, values in recourse_series.items():
            if period < len(values):
                row[family] = values[period]
        recourse_rows.append(row)
    atomic_write_csv(
        destination / "dispatch_recourse_worst.csv",
        ["period", *all_columns],
        recourse_rows,
    )
    history_rows = [asdict(item) for item in result.history]
    history_fields = list(history_rows[0]) if history_rows else ["iteration"]
    atomic_write_csv(destination / "ccg_history.csv", history_fields, history_rows)


def _export_c4_stress_infeasible(
    output_directory: str | Path,
    *,
    case: CaseData,
    two_stage_ir: TwoStageIR,
    bundle: UncertaintyBundle,
    result: C4StressResult,
    git_commit: str,
    ship_delay_source: Mapping[str, Any] | None,
) -> None:
    if (
        result.status != "stress_infeasible"
        or result.phase1_objective <= result.feasibility_tolerance
        or result.phase1_replay_objective is None
        or not result.primal_replay_passed
        or result.stress_recourse_objective is not None
        or result.stress_total_objective is not None
        or result.stress_objective_bound is not None
        or not result.worst_realization
        or abs(
            result.phase1_replay_objective - result.phase1_objective
        )
        > result.dual_replay_tolerance
    ):
        raise ValueError(
            "C4 stress-infeasible export requires an OPTIMAL phase-I "
            "adversary with a matching positive primal replay"
        )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    horizon = case.profile.periods
    source = ship_delay_source or {}
    semantic_hash = canonical_sha256(case.to_dict())
    replay_gap = abs(
        result.phase1_replay_objective - result.phase1_objective
    )

    atomic_write_json(
        destination / "certificate.json",
        {
            "schema_version": 1,
            "case_name": case.case_name,
            "profile_name": case.profile.profile_name,
            "status": "stress_infeasible",
            "engineering_optimal": False,
            "nominal_policy_optimal": True,
            "stress_feasible": False,
            "model_horizon": horizon,
            "ship_builder_horizon": 3 * horizon,
            "ship_projection_window": [horizon, 2 * horizon],
            "nominal_master_status": "OPTIMAL",
            "master_status": "OPTIMAL",
            "phase1_adversary_status": "OPTIMAL",
            "phase1_primal_replay_passed": True,
            "primal_replay_passed": True,
            "cost_adversary_status": "NOT_RUN",
            "partition_oracle_used": False,
            "all_partition_leaves_optimal_or_empty": False,
            "uncertainty_coverage_complete": True,
            "nominal_objective": result.nominal_objective,
            "phase1_objective": result.phase1_objective,
            "phase1_replay_objective": result.phase1_replay_objective,
            "phase1_primal_dual_gap_abs": replay_gap,
            "feasibility_tolerance": result.feasibility_tolerance,
            "dual_replay_tolerance": result.dual_replay_tolerance,
            "phase1_violation_margin": (
                result.phase1_objective - result.feasibility_tolerance
            ),
            "stress_recourse_objective": None,
            "stress_total_objective": None,
            "stress_objective_bound": None,
            "accepted_limited_solve": False,
            "time_limit_used_for_certificate": False,
            "joint_bundle_sha256": bundle.bundle_sha256,
            "case_semantic_sha256": semantic_hash,
            "git_commit": git_commit,
            "gurobi_version": ".".join(
                str(value) for value in gp.gurobi.version()
            ),
        },
    )
    atomic_write_json(
        destination / "run_metadata.json",
        {
            "solver": "gurobipy",
            "git_commit": git_commit,
            "accepted_limited_solve": False,
            "time_limit_used": False,
            "result_scope": "deterministic_nominal_then_exact_phase1_stress",
        },
    )
    atomic_write_json(destination / "resolved_case.json", case.to_dict())
    atomic_write_json(
        destination / "wind_bundle_reference.json",
        {
            "bundle_sha256": bundle.proofs.get("wind_bundle_sha256"),
            "source_hashes": {
                key: value
                for key, value in bundle.source_hashes.items()
                if key.startswith("wind.")
            },
        },
    )
    atomic_write_json(
        destination / "ship_bundle_reference.json",
        {
            "bundle_sha256": bundle.proofs.get("ship_bundle_sha256"),
            "source_hashes": {
                key: value
                for key, value in bundle.source_hashes.items()
                if key.startswith("ship_delay.")
            },
        },
    )
    atomic_write_json(
        destination / "joint_bundle_reference.json",
        {
            "bundle_sha256": bundle.bundle_sha256,
            "profile_name": bundle.profile_name,
            "model_horizon": bundle.model_horizon,
            "proofs": bundle.proofs,
        },
    )
    atomic_write_json(
        destination / "phase1_stress_diagnostic.json",
        {
            "status": "stress_infeasible",
            "phase1_objective": result.phase1_objective,
            "phase1_replay_objective": result.phase1_replay_objective,
            "phase1_primal_dual_gap_abs": replay_gap,
            "feasibility_tolerance": result.feasibility_tolerance,
            "dual_replay_tolerance": result.dual_replay_tolerance,
            "violation_margin": (
                result.phase1_objective - result.feasibility_tolerance
            ),
            "primal_replay_passed": result.primal_replay_passed,
        },
    )
    atomic_write_json(
        destination / "phase1_worst_selector.json", result.worst_selector
    )

    first_cost_by_component: dict[str, float] = {}
    for spec in two_stage_ir.first_stage.variables:
        first_cost_by_component[spec.component] = (
            first_cost_by_component.get(spec.component, 0.0)
            + spec.objective_coefficient * result.first_stage[spec.key]
        )
    atomic_write_json(
        destination / "cost_breakdown.json",
        {
            "day_ahead_by_component": first_cost_by_component,
            "recourse_by_component": None,
            "reason": "No feasible stress recourse exists for the nominal policy.",
        },
    )
    atomic_write_json(
        destination / "model_stats.json",
        {
            "first_stage_variables": len(two_stage_ir.first_stage.variables),
            "first_stage_constraints": len(two_stage_ir.first_stage.constraints),
            "recourse_variables_per_scenario": len(two_stage_ir.recourse.variables),
            "recourse_constraints_per_scenario": len(
                two_stage_ir.recourse.constraints
            ),
            "bundle_selectors": len(bundle.selectors),
            "bundle_constraints": len(bundle.constraints),
            "cost_adversary_run": False,
        },
    )
    atomic_write_json(
        destination / "residuals.json",
        {
            "residual_type": "phase1_primal_dual_replay",
            "phase1_primal_dual_gap_abs": replay_gap,
            "primal_replay_passed": result.primal_replay_passed,
            "physical_recourse_residuals_available": False,
            "reason": "The stress realization has no feasible physical recourse.",
        },
    )

    scenario_hash = realization_sha256(result.worst_realization)
    atomic_write_csv(
        destination / "scenario_pool.csv",
        ["realization_sha256", "selector_sha256", "scenario_role"],
        [
            {
                "realization_sha256": scenario_hash,
                "selector_sha256": canonical_sha256(result.worst_selector),
                "scenario_role": "phase1_infeasibility_witness",
            }
        ],
    )
    worst_rows = []
    for period in range(horizon):
        row = {"period": period}
        for key, values in sorted(result.worst_realization.items()):
            if period < len(values):
                row[key] = values[period]
        worst_rows.append(row)
    atomic_write_csv(
        destination / "worst_scenario.csv",
        ["period", *sorted(result.worst_realization)],
        worst_rows,
    )

    assignment_rows = _ship_assignment_rows(
        result.worst_selector, source, horizon
    )
    atomic_write_csv(
        destination / "ship_delay_assignment.csv",
        [
            "nominal_source_period",
            "anonymous_ship_index",
            "delay_steps",
            "actual_day_offset",
            "actual_arrival_period",
        ],
        assignment_rows,
    )
    arrival = result.worst_realization.get(
        "ship.arrival_count", np.zeros(horizon)
    )
    atomic_write_csv(
        destination / "actual_arrival_current_day.csv",
        ["period", "actual_arrival_count"],
        [
            {"period": period, "actual_arrival_count": arrival[period]}
            for period in range(horizon)
        ],
    )
    atomic_write_csv(
        destination / "actual_arrival_three_day_audit.csv",
        [
            "three_day_index",
            "day_offset",
            "period_in_day",
            "actual_arrival_count",
        ],
        _three_day_arrival_rows(result.worst_selector, source, horizon),
    )
    fields, rows = _period_rows(result.first_stage, horizon)
    atomic_write_csv(destination / "dispatch_day_ahead.csv", fields, rows)
    atomic_write_csv(
        destination / "ccg_history.csv",
        [
            "iteration",
            "nominal_objective",
            "phase1_objective",
            "action",
        ],
        [
            {
                "iteration": 1,
                "nominal_objective": result.nominal_objective,
                "phase1_objective": result.phase1_objective,
                "action": "c4_stress_infeasible",
            }
        ],
    )


def export_c4_run(
    output_directory: str | Path,
    *,
    case: CaseData,
    two_stage_ir: TwoStageIR,
    bundle: UncertaintyBundle,
    result: C4StressResult,
    git_commit: str,
    ship_delay_source: Mapping[str, Any] | None = None,
) -> None:
    if result.status == "stress_infeasible":
        _export_c4_stress_infeasible(
            output_directory,
            case=case,
            two_stage_ir=two_stage_ir,
            bundle=bundle,
            result=result,
            git_commit=git_commit,
            ship_delay_source=ship_delay_source,
        )
        return
    if (
        result.status != "stress_certified"
        or result.stress_total_objective is None
        or result.stress_recourse_objective is None
        or result.stress_objective_bound is None
    ):
        raise ValueError("C4 certificate export requires a certified stress replay")
    scenario_hash = realization_sha256(result.worst_realization)
    synthetic = CcgResult(
        status="engineering_optimal",
        engineering_optimal=True,
        lower_bound=result.stress_total_objective,
        upper_bound=result.stress_total_objective,
        gap_abs=0.0,
        gap_rel=0.0,
        first_stage=result.first_stage,
        theta=result.stress_recourse_objective,
        scenarios=(
            ScenarioRecord(
                scenario_hash, result.worst_selector, result.worst_realization
            ),
        ),
        history=(
            CcgIteration(
                iteration=1,
                lower_bound=result.nominal_objective,
                global_upper_bound=result.stress_total_objective,
                gap_abs=0.0,
                gap_rel=0.0,
                theta=result.stress_recourse_objective,
                phase1_objective=result.phase1_objective,
                cost_adversary_objective=result.stress_recourse_objective,
                scenario_count=1,
                action="c4_nominal_then_stress",
            ),
        ),
        primal_replay_passed=result.primal_replay_passed,
        residual_check_passed=result.max_abs_residual <= 1e-6,
        max_abs_residual=result.max_abs_residual,
        bounds_ordered=True,
        uncertainty_coverage_complete=True,
        accepted_limited_solve=False,
        worst_selector=result.worst_selector,
        worst_realization=result.worst_realization,
        worst_recourse_variables=result.worst_recourse_variables,
        worst_cost_breakdown=result.worst_cost_breakdown,
        partition_oracle_used=result.partition_oracle_used,
        all_partition_leaves_optimal_or_empty=(
            result.all_partition_leaves_optimal_or_empty
        ),
        cost_partition_branch_keys=result.cost_partition_branch_keys,
    )
    export_ccg_run(
        output_directory,
        case=case,
        two_stage_ir=two_stage_ir,
        bundle=bundle,
        result=synthetic,
        git_commit=git_commit,
        ship_delay_source=ship_delay_source,
    )
    certificate_path = Path(output_directory) / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate.update(
        {
            "status": "stress_certified",
            "engineering_optimal": True,
            "nominal_objective": result.nominal_objective,
            "stress_recourse_objective": result.stress_recourse_objective,
            "stress_total_objective": result.stress_total_objective,
            "stress_objective_bound": result.stress_objective_bound,
            "outer_gap_tolerance": None,
        }
    )
    atomic_write_json(certificate_path, certificate)
