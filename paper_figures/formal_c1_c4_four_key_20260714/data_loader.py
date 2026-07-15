"""Strict loaders for publication-eligible formal run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class FormalRunError(ValueError):
    """Raised when a run is not eligible for publication figures."""


@dataclass(frozen=True)
class FormalRun:
    path: Path
    certificate: dict[str, Any]
    resolved_case: dict[str, Any]
    cost_breakdown: dict[str, Any]
    model_stats: dict[str, Any]
    residuals: dict[str, Any]
    terminal_states: dict[str, Any]
    run_metadata: dict[str, Any]
    day_ahead: pd.DataFrame
    recourse_worst: pd.DataFrame
    worst_scenario: pd.DataFrame
    ship_delay_assignment: pd.DataFrame
    actual_arrival_current_day: pd.DataFrame
    actual_arrival_three_day_audit: pd.DataFrame
    scenario_pool: pd.DataFrame
    ccg_history: pd.DataFrame


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalRunError(f"expected JSON object: {path}")
    return payload


def _assert_publication_certificate(certificate: dict[str, Any]) -> None:
    required_true = (
        "engineering_optimal",
        "uncertainty_coverage_complete",
        "primal_replay_passed",
        "residual_check_passed",
        "all_partition_leaves_optimal_or_empty",
    )
    required_optimal = (
        "master_status",
        "phase1_adversary_status",
        "cost_adversary_status",
    )
    failures = [key for key in required_true if certificate.get(key) is not True]
    failures.extend(
        key for key in required_optimal if certificate.get(key) != "OPTIMAL"
    )
    if certificate.get("accepted_limited_solve") is not False:
        failures.append("accepted_limited_solve")
    if certificate.get("time_limit_used_for_certificate") is not False:
        failures.append("time_limit_used_for_certificate")
    if failures:
        raise FormalRunError(
            "UNRESOLVED — NOT FOR PUBLICATION: " + ", ".join(sorted(failures))
        )


def load_formal_run(path: str | Path) -> FormalRun:
    run_path = Path(path).resolve()
    certificate = _load_json(run_path / "certificate.json")
    _assert_publication_certificate(certificate)
    return FormalRun(
        path=run_path,
        certificate=certificate,
        resolved_case=_load_json(run_path / "resolved_case.json"),
        cost_breakdown=_load_json(run_path / "cost_breakdown.json"),
        model_stats=_load_json(run_path / "model_stats.json"),
        residuals=_load_json(run_path / "residuals.json"),
        terminal_states=_load_json(run_path / "terminal_states.json"),
        run_metadata=_load_json(run_path / "run_metadata.json"),
        day_ahead=pd.read_csv(run_path / "dispatch_day_ahead.csv"),
        recourse_worst=pd.read_csv(run_path / "dispatch_recourse_worst.csv"),
        worst_scenario=pd.read_csv(run_path / "worst_scenario.csv"),
        ship_delay_assignment=pd.read_csv(run_path / "ship_delay_assignment.csv"),
        actual_arrival_current_day=pd.read_csv(
            run_path / "actual_arrival_current_day.csv"
        ),
        actual_arrival_three_day_audit=pd.read_csv(
            run_path / "actual_arrival_three_day_audit.csv"
        ),
        scenario_pool=pd.read_csv(run_path / "scenario_pool.csv"),
        ccg_history=pd.read_csv(run_path / "ccg_history.csv"),
    )
