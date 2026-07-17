from __future__ import annotations

import pytest

from port_h2_certificate.compilers.compile_adversary import compile_adversary
from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from tests.duality.test_exact_adversary import _exact_realizations


def test_exact_partition_leaves_cover_and_match_monolithic_adversary(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.partition_oracle import solve_exact_partition

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    template = compile_dual(recourse_ir, first_stage)
    monolithic = compile_adversary(template, toy_joint_bundle).solve()
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]
    partitioned = solve_exact_partition(template, toy_joint_bundle, branch_keys)
    assert partitioned.status == "OPTIMAL"
    assert partitioned.coverage_complete is True
    assert len(partitioned.leaves) == 4
    assert all(leaf.status in {"OPTIMAL", "INFEASIBLE"} for leaf in partitioned.leaves)
    assert partitioned.objective == pytest.approx(monolithic.objective, abs=1e-7)


def test_semantic_ship_partition_covers_every_feasible_ship_projection(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_adversary import compile_adversary
    from port_h2_certificate.partition_oracle import (
        build_ship_delay_partition_assignments,
        solve_exact_partition,
    )

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    template = compile_dual(recourse_ir, first_stage)
    monolithic = compile_adversary(template, toy_joint_bundle).solve()
    branch_keys, assignments = build_ship_delay_partition_assignments(
        toy_joint_bundle
    )
    expected = {
        tuple(selector[key] for key in branch_keys)
        for selector, _ in enumerated
    }

    assert {
        tuple(assignment[key] for key in branch_keys)
        for assignment in assignments
    } == expected
    partitioned = solve_exact_partition(
        template,
        toy_joint_bundle,
        branch_keys,
        partition_assignments=assignments,
    )
    assert len(partitioned.leaves) == len(assignments)
    assert partitioned.coverage_complete is True
    assert partitioned.objective == pytest.approx(monolithic.objective, abs=1e-7)


def test_partition_reuses_one_compiled_adversary_across_all_leaves(
    monkeypatch, toy_case, toy_joint_bundle
) -> None:
    import port_h2_certificate.partition_oracle as partition_oracle

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    real_compile = partition_oracle.compile_adversary
    compile_calls = []

    def recording_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(
        partition_oracle, "compile_adversary", recording_compile
    )
    result = partition_oracle.solve_exact_partition(
        compile_dual(recourse_ir, first_stage),
        toy_joint_bundle,
        toy_joint_bundle.primary_selector_keys[:2],
    )

    assert result.coverage_complete is True
    assert len(result.leaves) > 1
    assert len(compile_calls) == 1


def test_every_feasible_leaf_has_a_certified_witness_and_complete_start(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.partition_oracle import solve_exact_partition

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage

    partitioned = solve_exact_partition(
        compile_dual(recourse_ir, first_stage),
        toy_joint_bundle,
        toy_joint_bundle.primary_selector_keys[:2],
    )

    feasible = [leaf for leaf in partitioned.leaves if leaf.status == "OPTIMAL"]
    assert feasible
    assert all(leaf.witness_selector for leaf in feasible)
    assert all(leaf.known_lower_bound is not None for leaf in feasible)
    assert all(leaf.witness_primal_objective is not None for leaf in feasible)
    assert all(leaf.witness_dual_objective is not None for leaf in feasible)
    assert all(abs(leaf.witness_primal_dual_gap) <= 1e-7 for leaf in feasible)
    assert all(leaf.full_witness_start_used is True for leaf in feasible)


def test_partition_reports_solve_start_and_completion_for_every_leaf(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.partition_oracle import solve_exact_partition

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    events = []

    partitioned = solve_exact_partition(
        compile_dual(recourse_ir, first_stage),
        toy_joint_bundle,
        toy_joint_bundle.primary_selector_keys[:2],
        event_callback=events.append,
    )

    completed = [event for event in events if event.stage == "COMPLETED"]
    solve_started = [event for event in events if event.stage == "SOLVE_STARTED"]
    assert len(completed) == len(partitioned.leaves)
    assert len(solve_started) == sum(
        leaf.status == "OPTIMAL" for leaf in partitioned.leaves
    )
    assert [event.leaf_index for event in completed] == list(
        range(len(partitioned.leaves))
    )
    assert all(event.leaf is not None for event in completed)


def test_partition_can_resume_completed_leaves_without_resolving_them(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.partition_oracle import solve_exact_partition

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    template = compile_dual(recourse_ir, first_stage)
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]
    baseline = solve_exact_partition(template, toy_joint_bundle, branch_keys)
    resumed_events = []

    resumed = solve_exact_partition(
        template,
        toy_joint_bundle,
        branch_keys,
        resume_leaves=baseline.leaves[:2],
        event_callback=resumed_events.append,
    )

    assert resumed.status == "OPTIMAL"
    assert resumed.objective == pytest.approx(baseline.objective, abs=1e-7)
    assert [
        event.leaf.fixed_primary
        for event in resumed_events
        if event.stage == "RESUMED"
    ] == [leaf.fixed_primary for leaf in baseline.leaves[:2]]
    resumed_assignments = {
        tuple(sorted(event.fixed_primary.items()))
        for event in resumed_events
        if event.stage == "RESUMED"
    }
    assert not any(
        tuple(sorted(event.fixed_primary.items())) in resumed_assignments
        for event in resumed_events
        if event.stage == "SOLVE_STARTED"
    )


def test_exact_partition_preserves_a_compatible_known_scenario_lower_bound(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.partition_oracle import solve_exact_partition

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    known_selector, known_realization = enumerated[-1]
    known_value = compile_recourse(
        recourse_ir,
        first_stage,
        known_realization,
    ).solve().objective
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]

    partitioned = solve_exact_partition(
        compile_dual(recourse_ir, first_stage),
        toy_joint_bundle,
        branch_keys,
        known_scenario_values=((known_selector, known_value),),
    )
    compatible = next(
        leaf
        for leaf in partitioned.leaves
        if all(
            known_selector[key] == value
            for key, value in leaf.fixed_primary.items()
        )
    )

    assert compatible.status == "OPTIMAL"
    assert compatible.known_lower_bound == pytest.approx(known_value)
    assert compatible.lower_bound_consistent is True
    assert compatible.full_known_scenario_start_used is True
    assert compatible.objective + 1e-7 >= known_value
