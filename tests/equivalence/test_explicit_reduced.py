from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.reconstruct import reconstruct_recourse
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from tests.equivalence.explicit_reference import solve_explicit


@pytest.mark.parametrize("operation_cost", [0.0, 4.6])
def test_explicit_and_reduced_recourse_are_equivalent(
    toy_case, toy_joint_bundle, operation_cost
) -> None:
    case = toy_case()
    case = replace(
        case,
        cost=replace(
            case.cost,
            agv_operation_per_vehicle_hour=operation_cost,
        ),
    )
    realization = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    master = compile_master(two_stage, [realization]).solve()
    reduced = compile_recourse(
        two_stage.recourse, master.first_stage, realization
    ).solve()
    reconstructed = reconstruct_recourse(
        case, toy_joint_bundle, master.first_stage, realization, reduced.variables
    )
    explicit_free = solve_explicit(
        case, toy_joint_bundle, master.first_stage, realization
    )
    assert explicit_free.objective == pytest.approx(reduced.objective, abs=1e-7)
    explicit_replay = solve_explicit(
        case,
        toy_joint_bundle,
        master.first_stage,
        realization,
        fixed_series=reconstructed.series,
    )
    assert explicit_replay.objective == pytest.approx(reduced.objective, abs=1e-7)
    for key, values in explicit_replay.series.items():
        assert np.asarray(values) == pytest.approx(
            np.asarray(reconstructed.series[key]), abs=1e-7
        )
