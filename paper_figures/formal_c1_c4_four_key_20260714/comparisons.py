"""Strict common-scenario and cross-policy comparisons for Fig. 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import gurobipy as gp
import numpy as np
import pandas as pd

from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import LoadedCase, load_case
from port_h2_certificate.reconstruct import reconstruct_recourse
from port_h2_certificate.recourse_ir import VariableKey, build_recourse_ir
from port_h2_certificate.solver.ccg import realization_sha256
from runners.input_adapters import build_joint_bundle

from .build_data import FORMAL_ROOT, INPUT_ROOT, PROFILE_NAME
from .checks import (
    assert_cost_reconciliation,
    assert_no_nan_or_inf,
    assert_same_bundle,
    assert_same_scenario,
)
from .data_loader import FormalRun, FormalRunError, load_formal_run
from .metrics import compute_operational_metrics, signed_cost_components


RUN_DIRECTORIES = {
    "C1": "C1_ProposedRobustMain_refined_partition",
    "C2": "C2_NoHydrogenRobust_refined_partition",
    "C3": "C3_WorkCapacityCapsRobust_refined_partition",
    "C4": "C4_DeterministicMain_refined_partition",
}
INPUT_FILES = {
    "C1": "C1_large_port_12v.yaml",
    "C2": "C2_large_port_12v.yaml",
    "C3": "C3_large_port_12v.yaml",
    "C4": "C4_large_port_12v.yaml",
}


@dataclass(frozen=True)
class ComparisonArtifacts:
    cost_components: pd.DataFrame
    architecture_metrics: pd.DataFrame
    common_scenario_timeseries: pd.DataFrame
    cross_evaluation: pd.DataFrame
    paths: Mapping[str, Path]
    normalization_exclusions: tuple[str, ...]


@dataclass(frozen=True)
class Replay:
    policy_case: str
    scenario_case: str
    scenario_sha256: str
    first_stage_cost: float
    recourse_cost: float
    total_cost: float
    max_abs_residual: float
    reconstructed: object


def _load_cases_and_runs() -> tuple[dict[str, LoadedCase], dict[str, FormalRun]]:
    cases = {
        code: load_case(INPUT_ROOT / filename, PROFILE_NAME)
        for code, filename in INPUT_FILES.items()
    }
    runs = {
        code: load_formal_run(FORMAL_ROOT / directory)
        for code, directory in RUN_DIRECTORIES.items()
    }
    assert_same_bundle(tuple(runs.values()))
    return cases, runs


def _first_stage_values(
    loaded: LoadedCase, bundle, run: FormalRun
) -> tuple[dict[VariableKey, float], float]:
    ir = build_first_stage_ir(loaded.case, bundle)
    dispatch = run.day_ahead.set_index("period")
    values: dict[VariableKey, float] = {}
    for spec in ir.variables:
        family, period = spec.key
        value = float(dispatch.at[period, family])
        if not np.isfinite(value):
            raise FormalRunError(f"nonfinite first-stage value: {spec.key}")
        values[spec.key] = value
    objective = float(
        sum(spec.objective_coefficient * values[spec.key] for spec in ir.variables)
    )
    return values, objective


def _realization(run: FormalRun) -> dict[str, np.ndarray]:
    frame = run.worst_scenario.sort_values("period")
    return {
        column: frame[column].dropna().to_numpy(dtype=float)
        for column in frame.columns
        if column != "period"
    }


def _replay(
    *,
    policy_code: str,
    scenario_code: str,
    loaded: LoadedCase,
    bundle,
    policy_run: FormalRun,
    scenario_run: FormalRun,
) -> Replay:
    first_stage, first_stage_cost = _first_stage_values(loaded, bundle, policy_run)
    realization = _realization(scenario_run)
    scenario_hash = realization_sha256(realization)
    result = compile_recourse(
        build_recourse_ir(loaded.case, bundle), first_stage, realization
    ).solve()
    if result.status != gp.GRB.OPTIMAL:
        raise FormalRunError(
            f"{policy_code}/{scenario_code} fixed recourse was not OPTIMAL"
        )
    reconstructed = reconstruct_recourse(
        loaded.case, bundle, first_stage, realization, result.variables
    )
    if abs(reconstructed.recourse_objective - result.objective) > 1e-6:
        raise FormalRunError(
            f"{policy_code}/{scenario_code} reconstructed recourse cost mismatch"
        )
    return Replay(
        policy_case=policy_code,
        scenario_case=scenario_code,
        scenario_sha256=scenario_hash,
        first_stage_cost=first_stage_cost,
        recourse_cost=float(result.objective),
        total_cost=first_stage_cost + float(result.objective),
        max_abs_residual=float(result.max_abs_residual),
        reconstructed=reconstructed,
    )


def _detailed_cost_rows(
    code: str, loaded: LoadedCase, run: FormalRun
) -> list[dict[str, object]]:
    case = loaded.case
    dt = case.profile.dt_hours
    scale = case.cost_scale
    dispatch = run.day_ahead.sort_values("period")
    buy = dispatch["grid_buy_da_mw"].to_numpy(dtype=float)
    sell = dispatch["grid_sell_da_mw"].to_numpy(dtype=float)
    spill = dispatch["spill_da_mw"].to_numpy(dtype=float)
    components = {
        "DA grid purchase": float(
            np.sum(np.asarray(case.grid.buy_price_per_kwh) * buy * 1000.0 * dt)
            / scale
        ),
        "DA grid sale": float(
            -np.sum(np.asarray(case.grid.sell_price_per_kwh) * sell * 1000.0 * dt)
            / scale
        ),
        "DA spill": float(
            case.cost.spill_day_ahead_per_kwh
            * np.sum(spill)
            * 1000.0
            * dt
            / scale
        ),
        "DA LOHC operation/ramp": 0.0,
        "RT grid deviation": 0.0,
        "RT spill": 0.0,
        "RT spill deviation": 0.0,
        "RT LOHC adjustment": 0.0,
        "RT AGV charge adjustment": 0.0,
        "Backlog penalties": 0.0,
        "LOHC outbound revenue": 0.0,
    }
    if case.has_hydrogen_chain:
        assert case.lohc is not None
        lohc_power = dispatch["lohc_power_da_mw"].to_numpy(dtype=float)
        ramp = dispatch["lohc_ramp_abs_da_mw"].fillna(0.0).to_numpy(dtype=float)
        components["DA LOHC operation/ramp"] = float(
            case.cost.lohc_operation_per_kg
            * np.sum(lohc_power)
            * 1000.0
            * dt
            / case.lohc.energy_kwh_per_kg
            / scale
            + case.cost.lohc_ramp_per_kw
            * np.sum(ramp)
            * 1000.0
            * dt
            / scale
        )
    recourse = run.cost_breakdown["reconstructed_recourse"]
    mapping = {
        "RT grid deviation": "grid_deviation",
        "RT spill": "spill_real_time",
        "RT spill deviation": "spill_deviation",
        "RT LOHC adjustment": "lohc_deviation",
        "RT AGV charge adjustment": "charge_deviation",
        "Backlog penalties": "backlog_delay",
        "LOHC outbound revenue": "lohc_export_revenue",
    }
    for label, key in mapping.items():
        components[label] = float(recourse.get(key, 0.0))

    generic = signed_cost_components(run)
    assert_cost_reconciliation(run, generic)
    target = (
        float(run.certificate["stress_total_objective"])
        if code == "C4"
        else float(run.certificate["upper_bound"])
    )
    if abs(sum(components.values()) - target) > 1e-6:
        raise FormalRunError(f"detailed signed costs do not reconcile for {code}")
    return [
        {
            "case": code,
            "component": label,
            "signed_cost": value,
            "total_objective": target,
        }
        for label, value in components.items()
    ]


def _architecture_metric_rows(
    cases: Mapping[str, LoadedCase], runs: Mapping[str, FormalRun]
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    actual: dict[str, dict[str, float]] = {}
    labels = {
        "grid_import_mwh": "Grid import",
        "grid_export_mwh": "Grid export",
        "spill_mwh": "Wind spill",
        "hydrogen_side_landing_mwh": "Hydrogen-side landing",
        "lohc_outbound_kg": "LOHC outbound",
        "backlog_area_teu_h": "Backlog area",
    }
    for code in ("C1", "C2"):
        metrics = compute_operational_metrics(
            runs[code], dt_hours=cases[code].case.profile.dt_hours
        )
        wind_nominal = np.asarray(
            cases[code].wind_source_payload["nominal_power_kw"], dtype=float
        )
        ratio = np.asarray(cases[code].case.wind_to_hydrogen_ratio, dtype=float)
        metrics["hydrogen_side_landing_mwh"] = float(
            np.sum(wind_nominal * ratio)
            * cases[code].case.profile.dt_hours
            / 1000.0
        )
        actual[code] = metrics

    excluded = tuple(
        key for key in labels if abs(actual["C1"][key]) <= 1e-12
    )
    rows: list[dict[str, object]] = []
    for key, label in labels.items():
        if key in excluded:
            continue
        for code in ("C1", "C2"):
            rows.append(
                {
                    "metric_key": key,
                    "metric_label": label,
                    "case": code,
                    "actual_value": actual[code][key],
                    "normalized_to_c1": actual[code][key] / actual["C1"][key],
                }
            )
    return rows, excluded


def _series_rows_from_replay(replay: Replay, loaded: LoadedCase, result_variables):
    """Return no-NaN long-form trajectories for the C1/C3 panel."""
    dt = loaded.case.profile.dt_hours
    rows: list[dict[str, object]] = []
    for name, values in (
        ("backlog_teu", replay.reconstructed.series["backlog_tasks"]),
        (
            "agv_container_count",
            tuple(
                result_variables[("agv_container_count", period)]
                for period in range(loaded.case.profile.periods)
            ),
        ),
        (
            "agv_lohc_count",
            tuple(
                result_variables[("agv_lohc_count", period)]
                for period in range(loaded.case.profile.periods)
            ),
        ),
    ):
        for period, value in enumerate(values):
            rows.append(
                {
                    "policy_case": replay.policy_case,
                    "scenario_case": replay.scenario_case,
                    "scenario_sha256": replay.scenario_sha256,
                    "series": name,
                    "period": period,
                    "time_hour": period * dt,
                    "value": float(value),
                }
            )
    return rows


def _replay_with_variables(
    *,
    policy_code: str,
    scenario_code: str,
    loaded: LoadedCase,
    bundle,
    policy_run: FormalRun,
    scenario_run: FormalRun,
):
    first_stage, first_stage_cost = _first_stage_values(loaded, bundle, policy_run)
    realization = _realization(scenario_run)
    result = compile_recourse(
        build_recourse_ir(loaded.case, bundle), first_stage, realization
    ).solve()
    reconstructed = reconstruct_recourse(
        loaded.case, bundle, first_stage, realization, result.variables
    )
    replay = Replay(
        policy_case=policy_code,
        scenario_case=scenario_code,
        scenario_sha256=realization_sha256(realization),
        first_stage_cost=first_stage_cost,
        recourse_cost=float(result.objective),
        total_cost=first_stage_cost + float(result.objective),
        max_abs_residual=float(result.max_abs_residual),
        reconstructed=reconstructed,
    )
    return replay, result.variables


def build_comparison_data(output_directory: str | Path) -> ComparisonArtifacts:
    """Build every numerical input for Fig. 4 with strict OPTIMAL replays."""

    output_root = Path(output_directory).resolve()
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    cases, runs = _load_cases_and_runs()
    bundle = build_joint_bundle(cases["C1"], output_root / "_comparison_bundle_cache")
    if bundle.bundle_sha256 != runs["C1"].certificate["joint_bundle_sha256"]:
        raise FormalRunError("comparison bundle does not match formal certificates")

    cost_rows = [
        row
        for code in ("C1", "C2", "C3", "C4")
        for row in _detailed_cost_rows(code, cases[code], runs[code])
    ]
    cost_components = pd.DataFrame(cost_rows)

    metric_rows, exclusions = _architecture_metric_rows(cases, runs)
    architecture_metrics = pd.DataFrame(metric_rows)

    common_replays = []
    common_rows: list[dict[str, object]] = []
    for policy in ("C1", "C3"):
        replay, variables = _replay_with_variables(
            policy_code=policy,
            scenario_code="C3",
            loaded=cases[policy],
            bundle=bundle,
            policy_run=runs[policy],
            scenario_run=runs["C3"],
        )
        common_replays.append(replay)
        common_rows.extend(_series_rows_from_replay(replay, cases[policy], variables))
    assert_same_scenario([replay.scenario_sha256 for replay in common_replays])
    common_scenario = pd.DataFrame(common_rows)

    cross_rows: list[dict[str, object]] = []
    for policy in ("C1", "C4"):
        for scenario in ("C1", "C4"):
            replay = _replay(
                policy_code=policy,
                scenario_code=scenario,
                loaded=cases[policy],
                bundle=bundle,
                policy_run=runs[policy],
                scenario_run=runs[scenario],
            )
            backlog = np.asarray(
                replay.reconstructed.series["backlog_tasks"], dtype=float
            )
            cross_rows.append(
                {
                    "policy_case": policy,
                    "scenario_case": scenario,
                    "scenario_sha256": replay.scenario_sha256,
                    "first_stage_cost": replay.first_stage_cost,
                    "recourse_cost": replay.recourse_cost,
                    "total_cost": replay.total_cost,
                    "peak_backlog_teu": float(np.max(backlog)),
                    "terminal_backlog_teu": float(backlog[-1]),
                    "max_abs_residual": replay.max_abs_residual,
                    "solver_status": "OPTIMAL",
                    "accepted_limited_solve": False,
                }
            )
    cross_evaluation = pd.DataFrame(cross_rows)

    c1_diagonal = cross_evaluation.loc[
        (cross_evaluation["policy_case"] == "C1")
        & (cross_evaluation["scenario_case"] == "C1"),
        "total_cost",
    ].item()
    c4_diagonal = cross_evaluation.loc[
        (cross_evaluation["policy_case"] == "C4")
        & (cross_evaluation["scenario_case"] == "C4"),
        "total_cost",
    ].item()
    if abs(c1_diagonal - float(runs["C1"].certificate["upper_bound"])) > 1e-6:
        raise FormalRunError("C1 diagonal cross replay does not match certificate")
    if (
        abs(
            c4_diagonal
            - float(runs["C4"].certificate["stress_total_objective"])
        )
        > 1e-6
    ):
        raise FormalRunError("C4 diagonal cross replay does not match certificate")

    for name, frame in (
        ("cost components", cost_components.select_dtypes(include=[np.number])),
        ("architecture metrics", architecture_metrics.select_dtypes(include=[np.number])),
        ("common scenario", common_scenario.select_dtypes(include=[np.number])),
        ("cross evaluation", cross_evaluation.select_dtypes(include=[np.number])),
    ):
        assert_no_nan_or_inf(frame.to_numpy(), name=name)

    paths = {
        "cost_components": data_root / "fig04a_cost_components.csv",
        "architecture_metrics": data_root / "fig04b_architecture_metrics.csv",
        "common_scenario": data_root / "fig04c_common_scenario_timeseries.csv",
        "cross_evaluation": data_root / "fig04d_cross_evaluation.csv",
    }
    cost_components.to_csv(paths["cost_components"], index=False)
    architecture_metrics.to_csv(paths["architecture_metrics"], index=False)
    common_scenario.to_csv(paths["common_scenario"], index=False)
    cross_evaluation.to_csv(paths["cross_evaluation"], index=False)
    return ComparisonArtifacts(
        cost_components=cost_components,
        architecture_metrics=architecture_metrics,
        common_scenario_timeseries=common_scenario,
        cross_evaluation=cross_evaluation,
        paths=paths,
        normalization_exclusions=exclusions,
    )
