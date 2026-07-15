"""Atomic per-iteration C&CG checkpoint artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from port_h2_certificate.export import atomic_write_csv, atomic_write_json
from port_h2_certificate.solver.ccg import CcgProgress
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


def write_ccg_checkpoint(
    output_directory: str | Path,
    *,
    progress: CcgProgress,
    two_stage_ir: TwoStageIR,
    bundle: UncertaintyBundle,
    provenance: Mapping[str, Any],
) -> None:
    destination = Path(output_directory)
    latest = progress.latest
    atomic_write_json(
        destination / "checkpoint.json",
        {
            "schema_version": 1,
            "iteration": latest.iteration,
            "latest_action": latest.action,
            "joint_bundle_sha256": bundle.bundle_sha256,
            "completed_iteration": True,
            "accepted_limited_solve": False,
        },
    )
    atomic_write_json(
        destination / "scenario_pool.json",
        [
            {
                "realization_sha256": record.realization_sha256,
                "selector": record.selector,
                "realization": record.realization,
            }
            for record in progress.scenarios
        ],
    )
    atomic_write_csv(
        destination / "first_stage.csv",
        ["family", "period", "value"],
        [
            {"family": key[0], "period": key[1], "value": value}
            for key, value in sorted(progress.first_stage.items())
        ],
    )
    history_rows = [asdict(item) for item in progress.history]
    atomic_write_csv(
        destination / "ccg_history.csv",
        list(history_rows[0]),
        history_rows,
    )
    atomic_write_json(
        destination / "phase1_certificate.json",
        {
            "status": "OPTIMAL",
            "objective": latest.phase1_objective,
            "feasible_full_set": latest.phase1_objective <= 1e-8,
        },
    )
    atomic_write_json(
        destination / "cost_adversary_certificate.json",
        {
            "status": "OPTIMAL" if latest.cost_adversary_objective is not None else "not_run",
            "objective": latest.cost_adversary_objective,
            "accepted_limited_solve": False,
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
        },
    )
    atomic_write_json(
        destination / "residuals.json",
        {"max_abs_residual": progress.max_abs_residual},
    )
    atomic_write_json(destination / "provenance.json", provenance)
