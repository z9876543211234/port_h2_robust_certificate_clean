from __future__ import annotations

import pytest

from port_h2_certificate.compilers.compile_adversary import compile_adversary
from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_master import compile_master
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
