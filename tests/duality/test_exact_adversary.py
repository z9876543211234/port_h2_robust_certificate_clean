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


def test_adversary_accepts_a_complete_explicit_dual_mip_start(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_adversary import compile_adversary

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    selector, realization = enumerated[-1]
    template = compile_dual(recourse_ir, first_stage)
    explicit_dual = template.instantiate(realization).solve()
    adversary = compile_adversary(template, toy_joint_bundle)

    started_variables = adversary.set_full_start(
        selector,
        explicit_dual.row_duals,
        explicit_dual.upper_bound_duals,
    )

    assert started_variables == adversary.model.NumVars
    assert set(adversary.product_variables) == set(adversary.active_selector_keys)
    for key, variable in adversary.selector_variables.items():
        assert variable.Start == pytest.approx(selector[key])
    for key, variable in adversary.product_variables.items():
        expected = selector[key] * sum(
            coefficient * explicit_dual.row_duals[row_name]
            for row_name, coefficient in adversary.product_row_coefficients[
                key
            ].items()
        )
        assert variable.Start == pytest.approx(expected)


def test_primal_dual_strengthening_preserves_the_exact_toy_adversary(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_adversary import (
        compile_adversary,
    )
    from port_h2_certificate.compilers.compile_recourse import compile_recourse

    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    enumerated = _exact_realizations(toy_joint_bundle)
    first_stage = compile_master(
        TwoStageIR(first_ir, recourse_ir),
        [realization for _, realization in enumerated],
    ).solve().first_stage
    template = compile_dual(recourse_ir, first_stage)
    baseline = compile_adversary(template, toy_joint_bundle).solve()
    strengthened = compile_adversary(
        template,
        toy_joint_bundle,
        strengthen_with_primal=True,
    )
    witness_primal = compile_recourse(
        recourse_ir,
        first_stage,
        toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector),
    ).solve()
    witness_dual = template.instantiate(
        toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    ).solve()

    started = strengthened.set_full_start(
        toy_joint_bundle.nominal_selector,
        witness_dual.row_duals,
        witness_dual.upper_bound_duals,
        witness_primal.variables,
    )
    result = strengthened.solve()

    assert strengthened.primal_variables
    assert started == strengthened.model.NumVars
    assert result.objective == pytest.approx(baseline.objective, abs=1e-7)
    assert result.selector == baseline.selector
    assert result.objective_bound == pytest.approx(result.objective, abs=1e-7)


def test_inferred_objective_cap_is_valid_and_preserves_exact_adversary(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_adversary import (
        compile_adversary,
        infer_optimal_recourse_objective_upper_bound,
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
    replay_values = [
        compile_recourse(recourse_ir, first_stage, realization).solve().objective
        for _, realization in enumerated
    ]

    cap = infer_optimal_recourse_objective_upper_bound(
        template, toy_joint_bundle
    )
    baseline = compile_adversary(template, toy_joint_bundle).solve()
    capped = compile_adversary(
        template,
        toy_joint_bundle,
        objective_upper_bound=cap,
    )
    capped_result = capped.solve()

    assert cap >= max(replay_values) - 1e-8
    assert capped.model.getConstrByName("valid_recourse_objective_upper_bound")
    assert capped_result.objective == pytest.approx(baseline.objective, abs=1e-7)
    assert capped_result.objective_bound == pytest.approx(
        capped_result.objective, abs=1e-7
    )


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
