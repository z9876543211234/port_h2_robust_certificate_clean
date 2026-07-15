from __future__ import annotations

from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR


def test_candidate_search_only_returns_replayed_noncertifying_scenarios(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.candidate_search import evaluate_candidates

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    nominal = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    master = compile_master(two_stage, [nominal]).solve()
    candidates = [
        toy_joint_bundle.nominal_selector,
        {key: 0 for key in toy_joint_bundle.primary_selector_keys},
    ]
    results = evaluate_candidates(
        two_stage.recourse,
        master.first_stage,
        toy_joint_bundle,
        candidates,
        theta=master.theta,
    )
    assert results
    assert all(result.certifying is False for result in results)
    assert all(result.objective_bound is None for result in results)
    assert all(result.replay_status == "OPTIMAL" for result in results)
