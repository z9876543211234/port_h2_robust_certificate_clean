from __future__ import annotations

from dataclasses import replace
import json

import pytest

from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from tests.duality.test_exact_adversary import _exact_realizations


def test_c4_nominal_then_exact_stress_matches_enumeration(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.export import export_c4_run
    from port_h2_certificate.solver.c4_stress import solve_c4_nominal_then_stress

    case = replace(toy_case(), case_name="C4_DeterministicMain")
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    result = solve_c4_nominal_then_stress(two_stage, toy_joint_bundle)
    assert result.status == "stress_certified"
    assert result.phase1_objective <= 1e-8
    assert result.primal_replay_passed is True
    assert result.max_abs_residual <= 1e-6
    enumerated = _exact_realizations(toy_joint_bundle)
    values = [
        compile_recourse(two_stage.recourse, result.first_stage, realization)
        .solve()
        .objective
        for _, realization in enumerated
    ]
    assert result.stress_recourse_objective == pytest.approx(max(values), abs=1e-7)
    assert result.stress_objective_bound == pytest.approx(
        result.stress_recourse_objective, abs=1e-8
    )
    export_c4_run(
        tmp_path / "c4",
        case=case,
        two_stage_ir=two_stage,
        bundle=toy_joint_bundle,
        result=result,
        git_commit="test-commit",
    )
    certificate = json.loads((tmp_path / "c4" / "certificate.json").read_text())
    assert certificate["status"] == "stress_certified"
    assert certificate["nominal_objective"] == pytest.approx(result.nominal_objective)


def test_c4_partitioned_stress_records_complete_exact_coverage(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.export import export_c4_run
    from port_h2_certificate.solver.c4_stress import solve_c4_nominal_then_stress

    case = replace(toy_case(), case_name="C4_DeterministicMain")
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    branch_keys = toy_joint_bundle.primary_selector_keys[:2]

    result = solve_c4_nominal_then_stress(
        two_stage,
        toy_joint_bundle,
        cost_partition_branch_keys=branch_keys,
    )

    assert result.status == "stress_certified"
    assert result.partition_oracle_used is True
    assert result.all_partition_leaves_optimal_or_empty is True
    assert result.cost_partition_branch_keys == branch_keys
    export_c4_run(
        tmp_path / "c4_partitioned",
        case=case,
        two_stage_ir=two_stage,
        bundle=toy_joint_bundle,
        result=result,
        git_commit="test-commit",
    )
    certificate = json.loads(
        (tmp_path / "c4_partitioned" / "certificate.json").read_text()
    )
    assert certificate["partition_oracle_used"] is True
    assert certificate["all_partition_leaves_optimal_or_empty"] is True
    assert certificate["cost_partition_branch_keys"] == list(branch_keys)
