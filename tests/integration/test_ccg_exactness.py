from __future__ import annotations

import pytest

from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from tests.duality.test_exact_adversary import _exact_realizations


def test_ccg_matches_extensive_form_on_complete_toy_set(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.solver.ccg import solve_robust_ccg

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    enumerated = _exact_realizations(toy_joint_bundle)
    extensive = compile_master(
        two_stage, [realization for _, realization in enumerated]
    ).solve()

    result = solve_robust_ccg(
        two_stage,
        toy_joint_bundle,
        outer_gap_tolerance=1e-9,
        feasibility_tolerance=1e-8,
        dual_replay_tolerance=1e-7,
        max_iterations=50,
    )
    assert result.status == "engineering_optimal"
    assert result.engineering_optimal is True
    assert result.lower_bound == pytest.approx(extensive.objective, abs=1e-6)
    assert result.upper_bound == pytest.approx(extensive.objective, abs=1e-6)
    assert result.gap_rel <= 1e-9
    assert result.primal_replay_passed is True
    assert result.residual_check_passed is True
    assert result.max_abs_residual <= 1e-6
    assert result.bounds_ordered is True
    assert len(result.scenarios) <= len(enumerated)
    assert result.history


def test_ccg_can_use_an_exact_cost_adversary_partition(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.solver.ccg import solve_robust_ccg

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    enumerated = _exact_realizations(toy_joint_bundle)
    extensive = compile_master(
        two_stage, [realization for _, realization in enumerated]
    ).solve()
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]

    result = solve_robust_ccg(
        two_stage,
        toy_joint_bundle,
        outer_gap_tolerance=1e-9,
        cost_partition_branch_keys=branch_keys,
    )

    assert result.engineering_optimal is True
    assert result.upper_bound == pytest.approx(extensive.objective, abs=1e-6)
    assert result.partition_oracle_used is True
    assert result.all_partition_leaves_optimal_or_empty is True
    assert result.cost_partition_branch_keys == branch_keys


def test_ccg_passes_replayed_master_scenarios_as_partition_lower_bounds(
    monkeypatch, toy_case, toy_joint_bundle
) -> None:
    import port_h2_certificate.solver.ccg as ccg

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    real_solve = ccg.solve_exact_partition
    recorded_batches = []

    def recording_solve(*args, **kwargs):
        recorded_batches.append(tuple(kwargs.get("known_scenario_values", ())))
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(ccg, "solve_exact_partition", recording_solve)
    result = ccg.solve_robust_ccg(
        two_stage,
        toy_joint_bundle,
        outer_gap_tolerance=1e-9,
        cost_partition_branch_keys=(
            toy_joint_bundle.primary_selector_keys[:2]
        ),
    )

    assert result.engineering_optimal is True
    assert recorded_batches
    assert all(batch for batch in recorded_batches)
    assert all(
        isinstance(value, float)
        for batch in recorded_batches
        for _, value in batch
    )
