from __future__ import annotations

import csv

import pytest

from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR


def test_frozen_first_stage_and_fixed_scenario_preserve_strong_duality(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from experiments.agv_operation_cost_scan_20260716.diagnose_known_scenarios import (
        load_first_stage_csv,
        replay_known_scenario,
    )

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    realization = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    master = compile_master(two_stage, [realization]).solve()
    path = tmp_path / "first_stage.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("family", "period", "value"))
        writer.writeheader()
        for family, period in sorted(master.first_stage):
            writer.writerow(
                {
                    "family": family,
                    "period": period,
                    "value": master.first_stage[(family, period)],
                }
            )

    frozen = load_first_stage_csv(path, two_stage)
    result = replay_known_scenario(
        two_stage,
        toy_joint_bundle,
        frozen,
        toy_joint_bundle.nominal_selector,
        realization,
    )

    assert result.primal_status == "OPTIMAL"
    assert result.dual_status == "OPTIMAL"
    assert result.fixed_adversary_status == "OPTIMAL"
    assert result.primal_dual_gap == pytest.approx(0.0, abs=1e-7)
    assert result.primal_fixed_adversary_gap == pytest.approx(0.0, abs=1e-7)


def test_frozen_first_stage_rejects_missing_ir_variable(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from experiments.agv_operation_cost_scan_20260716.diagnose_known_scenarios import (
        load_first_stage_csv,
    )

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    path = tmp_path / "first_stage.csv"
    path.write_text("family,period,value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the first-stage IR"):
        load_first_stage_csv(path, two_stage)
