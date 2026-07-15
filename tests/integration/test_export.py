from __future__ import annotations

import json

from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.solver.ccg import solve_robust_ccg
from port_h2_certificate.two_stage_ir import TwoStageIR


REQUIRED = {
    "certificate.json",
    "run_metadata.json",
    "resolved_case.json",
    "wind_bundle_reference.json",
    "ship_bundle_reference.json",
    "joint_bundle_reference.json",
    "cost_breakdown.json",
    "model_stats.json",
    "residuals.json",
    "terminal_states.json",
    "scenario_pool.csv",
    "worst_scenario.csv",
    "ship_delay_assignment.csv",
    "actual_arrival_current_day.csv",
    "actual_arrival_three_day_audit.csv",
    "dispatch_day_ahead.csv",
    "dispatch_recourse_worst.csv",
    "ccg_history.csv",
}


def test_engineering_optimal_export_contains_required_certificate_artifacts(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.export import export_ccg_run

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    result = solve_robust_ccg(
        two_stage, toy_joint_bundle, outer_gap_tolerance=1e-9
    )
    destination = tmp_path / "run"
    export_ccg_run(
        destination,
        case=case,
        two_stage_ir=two_stage,
        bundle=toy_joint_bundle,
        result=result,
        git_commit="test-commit",
        ship_delay_source={
            "previous_day_arrival_count": [0, 0, 0, 1],
            "current_day_arrival_count": [1, 0, 1, 0],
            "next_day_arrival_count": [0, 0, 0, 0],
        },
    )
    assert REQUIRED <= {path.name for path in destination.iterdir()}
    certificate = json.loads((destination / "certificate.json").read_text())
    assert certificate["engineering_optimal"] is True
    assert certificate["status"] == "engineering_optimal"
    assert certificate["joint_bundle_sha256"] == toy_joint_bundle.bundle_sha256
    assert certificate["primal_replay_passed"] is True
    assert certificate["residual_check_passed"] is True
    assert certificate["accepted_limited_solve"] is False


def test_certificate_records_exact_cost_partition_coverage(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.export import export_ccg_run

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]
    result = solve_robust_ccg(
        two_stage,
        toy_joint_bundle,
        outer_gap_tolerance=1e-9,
        cost_partition_branch_keys=branch_keys,
    )
    destination = tmp_path / "partitioned_run"

    export_ccg_run(
        destination,
        case=case,
        two_stage_ir=two_stage,
        bundle=toy_joint_bundle,
        result=result,
        git_commit="test-commit",
        ship_delay_source={
            "previous_day_arrival_count": [0, 0, 0, 1],
            "current_day_arrival_count": [1, 0, 1, 0],
            "next_day_arrival_count": [0, 0, 0, 0],
        },
    )

    certificate = json.loads((destination / "certificate.json").read_text())
    assert certificate["partition_oracle_used"] is True
    assert certificate["all_partition_leaves_optimal_or_empty"] is True
    assert certificate["cost_partition_branch_keys"] == list(branch_keys)
