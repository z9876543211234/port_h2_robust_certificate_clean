from __future__ import annotations

import importlib
import json

import pytest


def _scan_module():
    try:
        return importlib.import_module(
            "experiments.agv_operation_cost_scan_20260716.run_scan"
        )
    except ModuleNotFoundError:
        pytest.fail("the isolated AGV operation-cost scan runner is missing")


def test_scan_costs_keep_zero_only_as_a_labeled_baseline() -> None:
    module = _scan_module()

    assert module.candidate_costs((2.0, 6.5), include_zero_baseline=True) == (
        0.0,
        2.0,
        6.5,
    )
    assert module.candidate_costs(
        (2.0, 6.5), include_zero_baseline=False
    ) == (2.0, 6.5)
    with pytest.raises(ValueError, match="strictly positive"):
        module.candidate_costs((0.0, 2.0), include_zero_baseline=True)


def test_thresholds_are_derived_from_run_energy_and_existing_penalties(
    toy_case,
) -> None:
    module = _scan_module()
    case = toy_case()

    thresholds = module.derive_operation_cost_thresholds(case)
    recharge = (
        case.agv.run_power_per_vehicle_kw / case.agv.charge_efficiency
    )

    assert thresholds.recharge_energy_kwh_per_vehicle_hour == pytest.approx(
        recharge
    )
    assert thresholds.real_time_spill_only_cny_per_vehicle_hour == pytest.approx(
        recharge * case.cost.spill_real_time_per_kwh
    )
    assert thresholds.all_spill_terms_cny_per_vehicle_hour == pytest.approx(
        recharge
        * (
            case.cost.spill_day_ahead_per_kwh
            + case.cost.spill_real_time_per_kwh
            + case.cost.spill_deviation_per_kwh
        )
    )
    assert thresholds.conservative_energy_terms_cny_per_vehicle_hour == pytest.approx(
        recharge
        * (
            case.cost.spill_day_ahead_per_kwh
            + case.cost.spill_real_time_per_kwh
            + case.cost.spill_deviation_per_kwh
            + case.cost.grid_deviation_per_kwh
        )
    )


def test_candidate_residual_is_measured_from_the_returned_solution(
    toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.compilers.compile_master import compile_master
    from port_h2_certificate.first_stage_ir import build_first_stage_ir
    from port_h2_certificate.recourse_ir import build_recourse_ir
    from port_h2_certificate.two_stage_ir import TwoStageIR

    module = _scan_module()
    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    realization = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    master = compile_master(two_stage, [realization]).solve()
    solution = master.scenario_recourse[0]

    valid = module.evaluate_candidate_residual(
        two_stage,
        master.first_stage,
        realization,
        solution,
    )
    invalid_solution = dict(solution)
    invalid_solution[("backlog", 1)] += 0.1
    invalid = module.evaluate_candidate_residual(
        two_stage,
        master.first_stage,
        realization,
        invalid_solution,
    )

    assert valid <= 1e-7
    assert invalid > valid + 1e-3


def test_saved_ccg_scenario_pool_is_loaded_as_complete_realizations(
    tmp_path, toy_joint_bundle
) -> None:
    module = _scan_module()
    nominal = toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    payload = [
        {
            "realization_sha256": "diagnostic",
            "selector": dict(toy_joint_bundle.nominal_selector),
            "realization": {
                key: list(values) for key, values in nominal.items()
            },
        }
    ]
    path = tmp_path / "scenario_pool.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = module.load_scenario_pool(path, toy_joint_bundle)

    assert len(loaded) == 1
    assert set(loaded[0]) == set(toy_joint_bundle.outputs)
    assert all(
        len(loaded[0][key]) == len(output.nominal)
        for key, output in toy_joint_bundle.outputs.items()
    )

    payload[0]["realization"].pop(next(iter(toy_joint_bundle.outputs)))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outputs"):
        module.load_scenario_pool(path, toy_joint_bundle)


def test_cost_worst_and_empty_running_worst_scenarios_remain_distinct() -> None:
    module = _scan_module()

    extrema = module.scenario_extrema(
        recourse_costs=(10.0, 20.0),
        empty_vehicle_periods=(5.0, 1.0),
    )

    assert extrema == (1, 0)


def test_selection_requires_cost_above_the_combined_spill_threshold() -> None:
    module = _scan_module()
    rows = [
        {
            "candidate_role": "positive_candidate",
            "agv_operation_per_vehicle_hour": 2.0,
            "total_empty_vehicle_periods": 0.0,
            "maximum_capacity_shortfall": 0.0,
        },
        {
            "candidate_role": "positive_candidate",
            "agv_operation_per_vehicle_hour": 6.5,
            "total_empty_vehicle_periods": 0.0,
            "maximum_capacity_shortfall": 0.0,
        },
    ]

    selected = module.select_candidate(
        rows,
        minimum_cost_exclusive=6.290322580645162,
    )

    assert selected["agv_operation_per_vehicle_hour"] == 6.5
