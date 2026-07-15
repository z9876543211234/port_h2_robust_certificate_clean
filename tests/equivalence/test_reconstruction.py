from __future__ import annotations

import pytest

from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR


def test_reconstruction_restores_eliminated_physical_variables(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.reconstruct import reconstruct_recourse

    case = toy_case()
    realization = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    master = compile_master(two_stage, [realization]).solve()
    replay = compile_recourse(
        two_stage.recourse, master.first_stage, realization
    ).solve()
    reconstructed = reconstruct_recourse(
        case, toy_joint_bundle, master.first_stage, realization, replay.variables
    )
    assert reconstructed.recourse_objective == pytest.approx(replay.objective, abs=1e-8)
    assert all(
        -case.grid.sell_capacity_kw / 1000.0 - 1e-8
        <= value
        <= case.grid.buy_capacity_kw / 1000.0 + 1e-8
        for value in reconstructed.series["grid_rt_mw"]
    )
    assert all(value >= -1e-8 for value in reconstructed.series["agv_idle_count"])
    assert all(value >= -1e-8 for value in reconstructed.series["task_done"])
    assert len(reconstructed.series["hydrogen_inventory_kg"]) == 5
    assert len(reconstructed.series["lohc_inventory_kg"]) == 5
    assert len(reconstructed.series["backlog_tasks"]) == 5
