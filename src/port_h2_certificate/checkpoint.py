"""Atomic per-iteration C&CG checkpoint artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from port_h2_certificate.export import atomic_write_csv, atomic_write_json
from port_h2_certificate.partition_oracle import PartitionEvent, PartitionLeaf
from port_h2_certificate.solver.ccg import CcgProgress
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.hashing import canonical_sha256
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


def _first_stage_sha256(first_stage: Mapping[tuple[str, int], float]) -> str:
    return canonical_sha256(
        [
            {"family": key[0], "period": key[1], "value": float(value)}
            for key, value in sorted(first_stage.items())
        ]
    )


def write_partition_checkpoint(
    path: str | Path,
    *,
    iteration: int,
    branch_keys: Sequence[str],
    first_stage: Mapping[tuple[str, int], float],
    bundle: UncertaintyBundle,
    events: Sequence[PartitionEvent],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "iteration": int(iteration),
            "branch_keys": list(branch_keys),
            "joint_bundle_sha256": bundle.bundle_sha256,
            "first_stage_sha256": _first_stage_sha256(first_stage),
            "events": [asdict(event) for event in events],
        },
    )


def load_partition_checkpoint(
    path: str | Path,
    *,
    iteration: int,
    branch_keys: Sequence[str],
    first_stage: Mapping[tuple[str, int], float],
    bundle: UncertaintyBundle,
) -> tuple[PartitionLeaf, ...]:
    source = Path(path)
    if not source.is_file():
        return ()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("iteration") != int(iteration)
        or tuple(payload.get("branch_keys", ())) != tuple(branch_keys)
        or payload.get("joint_bundle_sha256") != bundle.bundle_sha256
        or payload.get("first_stage_sha256") != _first_stage_sha256(first_stage)
    ):
        return ()
    leaves: list[PartitionLeaf] = []
    for event in payload.get("events", ()):
        leaf = event.get("leaf")
        if event.get("stage") not in {"COMPLETED", "RESUMED"} or leaf is None:
            continue
        restored = PartitionLeaf(**leaf)
        if restored.status not in {"OPTIMAL", "INFEASIBLE"}:
            continue
        leaves.append(restored)
    return tuple(leaves)


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
