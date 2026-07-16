"""Isolated nominal C1 scan for a positive AGV running-cost deterrent.

This experiment intentionally starts from the pre-empty-running-fix model:
container and LOHC AGV allocations remain continuous capacity variables and
their productive-flow constraints remain inequalities.  Only a common,
strictly positive operating cost in CNY/(vehicle hour) is scanned.  The zero
cost point is optionally retained as a labeled diagnostic baseline and is
never eligible as the selected parameter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from experiments.agv_operation_cost_scan_20260716.metrics import (
    measure_empty_running,
)
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.residuals import evaluate_residuals
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import load_case
from port_h2_certificate.reconstruct import reconstruct_recourse
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.schema import CaseData
from port_h2_certificate.two_stage_ir import TwoStageIR
from runners.input_adapters import build_joint_bundle


DEFAULT_POSITIVE_CANDIDATES = (
    2.0,
    4.0,
    6.5,
    8.0,
    10.0,
    12.0,
    14.5,
    18.0,
    25.0,
    35.0,
)


@dataclass(frozen=True)
class OperationCostThresholds:
    recharge_energy_kwh_per_vehicle_hour: float
    real_time_spill_only_cny_per_vehicle_hour: float
    all_spill_terms_cny_per_vehicle_hour: float
    conservative_energy_terms_cny_per_vehicle_hour: float


def candidate_costs(
    positive_candidates: Iterable[float],
    *,
    include_zero_baseline: bool,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in positive_candidates)
    if not values or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError("operation-cost candidates must be strictly positive")
    if len(set(values)) != len(values):
        raise ValueError("operation-cost candidates must be unique")
    return ((0.0,) if include_zero_baseline else ()) + values


def derive_operation_cost_thresholds(
    case: CaseData,
) -> OperationCostThresholds:
    recharge_energy = (
        case.agv.run_power_per_vehicle_kw / case.agv.charge_efficiency
    )
    real_time_spill = (
        recharge_energy * case.cost.spill_real_time_per_kwh
    )
    all_spill = recharge_energy * (
        case.cost.spill_day_ahead_per_kwh
        + case.cost.spill_real_time_per_kwh
        + case.cost.spill_deviation_per_kwh
    )
    conservative = all_spill + (
        recharge_energy * case.cost.grid_deviation_per_kwh
    )
    return OperationCostThresholds(
        recharge_energy_kwh_per_vehicle_hour=recharge_energy,
        real_time_spill_only_cny_per_vehicle_hour=real_time_spill,
        all_spill_terms_cny_per_vehicle_hour=all_spill,
        conservative_energy_terms_cny_per_vehicle_hour=conservative,
    )


def load_scenario_pool(
    path: Path, bundle
) -> tuple[dict[str, tuple[float, ...]], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("scenario pool must be a nonempty JSON list")
    expected_outputs = set(bundle.outputs)
    realizations: list[dict[str, tuple[float, ...]]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or not isinstance(
            record.get("realization"), dict
        ):
            raise ValueError(f"scenario pool entry {index} has no realization")
        raw = record["realization"]
        if set(raw) != expected_outputs:
            raise ValueError(
                f"scenario pool entry {index} outputs do not match the bundle"
            )
        realization: dict[str, tuple[float, ...]] = {}
        for key, output in bundle.outputs.items():
            values = tuple(float(value) for value in raw[key])
            if len(values) != len(output.nominal) or any(
                not math.isfinite(value) for value in values
            ):
                raise ValueError(
                    f"scenario pool entry {index} output {key} is invalid"
                )
            realization[key] = values
        realizations.append(realization)
    return tuple(realizations)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("scan summary cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _key_text(key: tuple[str, int]) -> str:
    return f"{key[0]}[{key[1]}]"


def _candidate_label(cost: float) -> str:
    if cost == 0.0:
        return "baseline_zero_not_selectable"
    return f"cost_{cost:g}_cny_per_vehicle_hour".replace(".", "p")


def _sum_family(
    solution: dict[tuple[str, int], float] | Any,
    family: str,
    horizon: int,
) -> float:
    return sum(float(solution[(family, period)]) for period in range(horizon))


def evaluate_candidate_residual(
    two_stage: TwoStageIR,
    first_stage,
    realization,
    solution,
) -> float:
    return evaluate_residuals(
        two_stage.recourse,
        first_stage,
        realization,
        solution,
    ).max_abs_residual


def scenario_extrema(
    *,
    recourse_costs: Iterable[float],
    empty_vehicle_periods: Iterable[float],
) -> tuple[int, int]:
    costs = tuple(float(value) for value in recourse_costs)
    empty = tuple(float(value) for value in empty_vehicle_periods)
    if not costs or len(costs) != len(empty):
        raise ValueError("scenario diagnostics must be nonempty and aligned")
    return (
        max(range(len(costs)), key=costs.__getitem__),
        max(range(len(empty)), key=empty.__getitem__),
    )


def select_candidate(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_cost_exclusive: float,
) -> dict[str, Any] | None:
    selectable = [
        row
        for row in rows
        if row["candidate_role"] == "positive_candidate"
        and row["agv_operation_per_vehicle_hour"]
        > minimum_cost_exclusive
        and row["total_empty_vehicle_periods"] <= 1e-7
        and row["maximum_capacity_shortfall"] <= 1e-7
    ]
    return (
        min(selectable, key=lambda row: row["agv_operation_per_vehicle_hour"])
        if selectable
        else None
    )


def _run_candidate(
    case_path: Path,
    profile: str,
    cost: float,
    bundle,
    realizations,
):
    loaded = load_case(
        case_path,
        profile,
        resolutions={
            "deterministic.cost.agv_operation_per_vehicle_hour": cost
        },
    )
    two_stage = TwoStageIR(
        build_first_stage_ir(loaded.case, bundle),
        build_recourse_ir(loaded.case, bundle),
    )
    started = time.perf_counter()
    master = compile_master(two_stage, realizations).solve()
    wall_time = time.perf_counter() - started
    metrics_by_scenario = []
    reconstructed_by_scenario = []
    residuals = []
    recourse_costs = []
    for realization, solution in zip(
        realizations, master.scenario_recourse, strict=True
    ):
        metrics_by_scenario.append(
            measure_empty_running(loaded.case, realization, solution)
        )
        residuals.append(
            evaluate_candidate_residual(
                two_stage,
                master.first_stage,
                realization,
                solution,
            )
        )
        reconstructed = reconstruct_recourse(
            loaded.case,
            bundle,
            master.first_stage,
            realization,
            solution,
        )
        reconstructed_by_scenario.append(reconstructed)
        recourse_costs.append(reconstructed.recourse_objective)
    cost_worst_index, empty_worst_index = scenario_extrema(
        recourse_costs=recourse_costs,
        empty_vehicle_periods=(
            metrics.total_empty_vehicle_periods
            for metrics in metrics_by_scenario
        ),
    )
    solution = master.scenario_recourse[cost_worst_index]
    reconstructed = reconstructed_by_scenario[cost_worst_index]
    metrics = metrics_by_scenario[empty_worst_index]
    horizon = loaded.case.profile.periods
    dt = loaded.case.profile.dt_hours
    first_stage_cost = sum(
        spec.objective_coefficient * master.first_stage[spec.key]
        for spec in two_stage.first_stage.variables
    )
    summary = {
        "agv_operation_per_vehicle_hour": cost,
        "candidate_role": (
            "diagnostic_zero_baseline" if cost == 0.0 else "positive_candidate"
        ),
        "master_status": "OPTIMAL",
        "scenario_count": len(realizations),
        "cost_worst_scenario_index": cost_worst_index,
        "empty_worst_scenario_index": empty_worst_index,
        "objective_scaled": master.objective,
        "first_stage_cost_scaled": first_stage_cost,
        "recourse_cost_scaled": reconstructed.recourse_objective,
        "wall_time_seconds": wall_time,
        "total_declared_vehicle_periods": metrics.total_declared_vehicle_periods,
        "total_productive_vehicle_periods": metrics.total_productive_vehicle_periods,
        "total_empty_vehicle_periods": metrics.total_empty_vehicle_periods,
        "cost_worst_scenario_empty_vehicle_periods": (
            metrics_by_scenario[cost_worst_index].total_empty_vehicle_periods
        ),
        "empty_fraction_of_declared_work": metrics.empty_fraction_of_declared_work,
        "container_empty_vehicle_periods": sum(
            metrics.container_empty_vehicle_periods
        ),
        "lohc_empty_vehicle_periods": sum(metrics.lohc_empty_vehicle_periods),
        "maximum_capacity_shortfall": max(
            item.maximum_capacity_shortfall for item in metrics_by_scenario
        ),
        "empty_battery_energy_kwh": metrics.empty_battery_energy_kwh,
        "recharge_energy_equivalent_kwh": metrics.recharge_energy_equivalent_kwh,
        "task_completed_total_teu": sum(reconstructed.series["task_done"]),
        "terminal_backlog_teu": reconstructed.series["backlog_tasks"][-1],
        "lohc_outbound_total_kg_h2eq": sum(
            reconstructed.series.get("lohc_outbound_kg", ())
        ),
        "agv_charge_energy_mwh": sum(
            reconstructed.series["agv_charge_power_rt_mw"]
        )
        * dt,
        "real_time_spill_mwh": _sum_family(
            solution, "spill_rt_mw", horizon
        )
        * dt,
        "agv_operation_cost_scaled": reconstructed.cost_breakdown[
            "agv_operation"
        ],
        "max_abs_recourse_residual": max(residuals),
        "case_semantic_sha256": loaded.case_semantic_sha256,
        "joint_bundle_sha256": bundle.bundle_sha256,
    }
    detail = {
        "summary": summary,
        "first_stage": {
            _key_text(key): value
            for key, value in sorted(master.first_stage.items())
        },
        "scenarios": [
            {
                "scenario_index": index,
                "recourse_cost_scaled": recourse_costs[index],
                "max_abs_recourse_residual": residuals[index],
                "metrics": asdict(metrics_by_scenario[index]),
                "cost_breakdown_scaled": dict(
                    reconstructed_by_scenario[index].cost_breakdown
                ),
                "recourse": {
                    _key_text(key): value
                    for key, value in sorted(
                        master.scenario_recourse[index].items()
                    )
                },
                "reconstructed_series": {
                    key: list(values)
                    for key, values in reconstructed_by_scenario[
                        index
                    ].series.items()
                },
            }
            for index in range(len(realizations))
        ],
    }
    return summary, detail, loaded.case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--profile",
        default="quarter_hour_96",
        choices=("hourly_24", "quarter_hour_96"),
    )
    parser.add_argument(
        "--candidates",
        type=float,
        nargs="+",
        default=DEFAULT_POSITIVE_CANDIDATES,
    )
    parser.add_argument("--without-zero-baseline", action="store_true")
    parser.add_argument(
        "--scenario-pool-json",
        type=Path,
        default=None,
        help=(
            "Optional saved C&CG checkpoint scenario_pool.json. When omitted, "
            "only the nominal realization is used."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case_path = args.case.resolve()
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty experiment directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    costs = candidate_costs(
        args.candidates,
        include_zero_baseline=not args.without_zero_baseline,
    )

    bundle_loaded = load_case(
        case_path,
        args.profile,
        resolutions={
            "deterministic.cost.agv_operation_per_vehicle_hour": costs[0]
        },
    )
    bundle = build_joint_bundle(bundle_loaded, output / "bundle")
    scenario_pool_path = (
        None
        if args.scenario_pool_json is None
        else args.scenario_pool_json.resolve()
    )
    if scenario_pool_path is not None:
        if not scenario_pool_path.is_file():
            raise FileNotFoundError(scenario_pool_path)
        realizations = load_scenario_pool(scenario_pool_path, bundle)
        proof_scope = "saved_ccg_scenario_pool_parameter_diagnostic"
    else:
        realizations = (bundle.evaluate(bundle.nominal_selector),)
        proof_scope = "nominal_single_scenario_parameter_diagnostic"
    thresholds = derive_operation_cost_thresholds(bundle_loaded.case)
    rows: list[dict[str, Any]] = []
    for cost in costs:
        summary, detail, _ = _run_candidate(
            case_path,
            args.profile,
            cost,
            bundle,
            realizations,
        )
        rows.append(summary)
        _atomic_json(output / _candidate_label(cost) / "result.json", detail)

    selected = select_candidate(
        rows,
        minimum_cost_exclusive=(
            thresholds.all_spill_terms_cny_per_vehicle_hour
        ),
    )
    _atomic_csv(output / "scan_summary.csv", rows)
    _atomic_json(
        output / "scan_manifest.json",
        {
            "experiment": "pre_fix_inequality_plus_positive_agv_operation_cost",
            "model_constraint_semantics_changed": False,
            "objective_semantics_changed": True,
            "source_case": str(case_path),
            "profile": args.profile,
            "scenario_count": len(realizations),
            "scenario_pool_source": (
                None if scenario_pool_path is None else str(scenario_pool_path)
            ),
            "positive_candidates_cny_per_vehicle_hour": list(args.candidates),
            "zero_baseline_included_for_diagnosis": (
                not args.without_zero_baseline
            ),
            "thresholds": asdict(thresholds),
            "selection_rule": (
                "smallest candidate strictly above the combined DA, RT, and "
                "spill-deviation incentive threshold, with total empty "
                "vehicle-periods and capacity shortfall both <= 1e-7"
            ),
            "selected_candidate": selected,
            "formal_robust_certificate": False,
            "proof_scope": proof_scope,
            "joint_bundle_sha256": bundle.bundle_sha256,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
