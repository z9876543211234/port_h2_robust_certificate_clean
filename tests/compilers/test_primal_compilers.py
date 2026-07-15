from __future__ import annotations

import gurobipy as gp
import math
import pytest

from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir


def _compiler(module_name: str, function_name: str):
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)


def _two_stage_ir(first_stage, recourse):
    cls = _compiler("port_h2_certificate.two_stage_ir", "TwoStageIR")
    return cls(first_stage=first_stage, recourse=recourse)


def test_single_scenario_master_matches_independent_recourse_replay(
    toy_case, toy_joint_bundle
) -> None:
    case = toy_case()
    first_ir = build_first_stage_ir(case, toy_joint_bundle)
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    two_stage = _two_stage_ir(first_ir, recourse_ir)
    realization = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)

    compile_master = _compiler(
        "port_h2_certificate.compilers.compile_master", "compile_master"
    )
    master = compile_master(two_stage, [realization])
    assert math.isinf(master.theta.LB) and master.theta.LB < 0.0
    master_result = master.solve()
    assert master_result.status == gp.GRB.OPTIMAL

    compile_recourse = _compiler(
        "port_h2_certificate.compilers.compile_recourse", "compile_recourse"
    )
    replay = compile_recourse(
        recourse_ir, master_result.first_stage, realization
    ).solve()
    assert replay.status == gp.GRB.OPTIMAL
    assert replay.objective == pytest.approx(master_result.theta, abs=1e-8)
    assert replay.max_abs_residual <= 1e-8
    assert len(replay.variable_basis) == len(recourse_ir.variables)
    assert len(replay.constraint_basis) == len(recourse_ir.constraints)


def test_recourse_rejects_missing_uncertain_output(toy_case, toy_joint_bundle) -> None:
    case = toy_case()
    recourse_ir = build_recourse_ir(case, toy_joint_bundle)
    realization = dict(
        toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    )
    realization.pop("ship.task_release")
    compile_recourse = _compiler(
        "port_h2_certificate.compilers.compile_recourse", "compile_recourse"
    )
    with pytest.raises(KeyError, match="ship.task_release"):
        compile_recourse(recourse_ir, {}, realization)


def test_master_requires_at_least_one_realization(toy_case, toy_joint_bundle) -> None:
    case = toy_case()
    two_stage = _two_stage_ir(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    compile_master = _compiler(
        "port_h2_certificate.compilers.compile_master", "compile_master"
    )
    with pytest.raises(ValueError, match="at least one"):
        compile_master(two_stage, [])
