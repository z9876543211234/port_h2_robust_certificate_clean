from __future__ import annotations

import gurobipy as gp
import pytest

from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR


def _nominal_master_solution(case, bundle):
    from port_h2_certificate.compilers.compile_master import compile_master

    first_ir = build_first_stage_ir(case, bundle)
    recourse_ir = build_recourse_ir(case, bundle)
    realization = bundle.evaluate(bundle.nominal_selector)
    master = compile_master(TwoStageIR(first_ir, recourse_ir), [realization])
    return recourse_ir, realization, master.solve().first_stage


def test_phase1_is_zero_for_feasible_master_solution(toy_case, toy_joint_bundle) -> None:
    from port_h2_certificate.compilers.compile_phase1 import compile_phase1

    recourse_ir, realization, first_stage = _nominal_master_solution(
        toy_case(), toy_joint_bundle
    )
    result = compile_phase1(recourse_ir, first_stage, realization).solve()
    assert result.status == gp.GRB.OPTIMAL
    assert result.objective <= 1e-8
    assert all(value <= 1e-8 for value in result.row_slacks.values())


def test_phase1_detects_infeasible_first_stage_without_silent_fallback(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_phase1 import compile_phase1

    recourse_ir, realization, first_stage = _nominal_master_solution(
        toy_case(), toy_joint_bundle
    )
    impossible = dict(first_stage)
    for period in range(4):
        impossible[("agv_charge_count_da", period)] = 100.0
    result = compile_phase1(recourse_ir, impossible, realization).solve()
    assert result.status == gp.GRB.OPTIMAL
    assert result.objective > 1e-8
    assert any(
        value > 1e-8
        for name, value in result.row_slacks.items()
        if "charge_adjust" in name
    )


def test_explicit_dual_matches_fixed_scenario_primal(toy_case, toy_joint_bundle) -> None:
    from port_h2_certificate.compilers.compile_dual import compile_dual
    from port_h2_certificate.compilers.compile_recourse import compile_recourse

    recourse_ir, realization, first_stage = _nominal_master_solution(
        toy_case(), toy_joint_bundle
    )
    primal = compile_recourse(recourse_ir, first_stage, realization).solve()
    dual = compile_dual(recourse_ir, first_stage).instantiate(realization).solve()
    assert primal.status == gp.GRB.OPTIMAL
    assert dual.status == gp.GRB.OPTIMAL
    assert dual.objective == pytest.approx(primal.objective, abs=1e-7)
    assert dual.max_constraint_violation <= 1e-8


def test_dual_template_adds_exact_upper_bound_rows(toy_case, toy_joint_bundle) -> None:
    from port_h2_certificate.compilers.compile_dual import compile_dual

    recourse_ir = build_recourse_ir(toy_case(), toy_joint_bundle)
    template = compile_dual(recourse_ir, {})
    expected = sum(spec.upper_bound is not None for spec in recourse_ir.variables)
    assert len(template.upper_bound_rows) == expected
    assert all(row.width >= 0.0 for row in template.upper_bound_rows)
