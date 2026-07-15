from __future__ import annotations

from itertools import product

import gurobipy as gp
import pytest

from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_phase1 import compile_phase1
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR


def _exact_realizations(bundle):
    unique = {}
    for values in product((0, 1), repeat=len(bundle.primary_selector_keys)):
        primary = dict(zip(bundle.primary_selector_keys, values, strict=True))
        try:
            selector = bundle.complete(primary)
        except ValueError:
            continue
        unique.setdefault(bundle.realization_hash(selector), (selector, bundle.evaluate(selector)))
    return tuple(unique.values())


def test_dual_only_adversary_matches_complete_toy_enumeration(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_adversary import compile_adversary

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    master = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve()
    replay_values = [
        compile_recourse(recourse_ir, master.first_stage, realization).solve().objective
        for _, realization in enumerated
    ]

    adversary = compile_adversary(
        compile_dual(recourse_ir, master.first_stage), toy_joint_bundle
    )
    assert adversary.model.Params.MIPGap == 0.0
    assert adversary.model.Params.MIPGapAbs == 0.0
    assert adversary.model.Params.DualReductions == 1
    assert adversary.model.NumGenConstrs == 2 * len(adversary.active_selector_keys)
    result = adversary.solve()
    assert result.status == gp.GRB.OPTIMAL
    assert result.objective_bound == pytest.approx(result.objective, abs=1e-8)
    assert result.objective == pytest.approx(max(replay_values), abs=1e-7)
    replay = compile_recourse(
        recourse_ir, master.first_stage, result.realization
    ).solve()
    assert replay.objective == pytest.approx(result.objective, abs=1e-7)


def test_phase1_adversary_matches_complete_toy_enumeration(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_phase1_adversary import (
        compile_phase1_adversary,
    )

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    nominal = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir), [nominal]
    ).solve().first_stage
    impossible = dict(first_stage)
    for period in range(4):
        impossible[("agv_charge_count_da", period)] = 100.0

    enumerated = _exact_realizations(toy_joint_bundle)
    phase1_values = [
        compile_phase1(recourse_ir, impossible, realization).solve().objective
        for _, realization in enumerated
    ]
    adversary = compile_phase1_adversary(
        recourse_ir, impossible, toy_joint_bundle, sigma=1.0
    )
    result = adversary.solve()
    assert result.status == gp.GRB.OPTIMAL
    assert result.objective > 1e-8
    assert result.objective == pytest.approx(max(phase1_values), abs=1e-7)
    replay = compile_phase1(
        recourse_ir, impossible, result.realization, sigma=1.0
    ).solve()
    assert replay.objective == pytest.approx(result.objective, abs=1e-7)
