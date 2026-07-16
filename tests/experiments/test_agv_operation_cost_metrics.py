from __future__ import annotations

import importlib

import pytest


def _metrics_module():
    try:
        return importlib.import_module(
            "experiments.agv_operation_cost_scan_20260716.metrics"
        )
    except ModuleNotFoundError:
        pytest.fail("the isolated AGV empty-running metrics module is missing")


def test_empty_running_is_measured_against_actual_completed_work(
    toy_case, toy_joint_bundle
) -> None:
    metrics_module = _metrics_module()
    case = toy_case()
    realization = dict(
        toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    )
    horizon = case.profile.periods
    realization["ship.task_release"] = [2.0] + [0.0] * (horizon - 1)

    container_service = (
        case.agv.container_rate_per_vehicle_hour
        * case.agv.container_turn_efficiency
        * case.profile.dt_hours
    )
    lohc_service = (
        case.agv.lohc_transport_rate_kg_per_vehicle_hour
        * case.agv.lohc_turn_efficiency
        * case.profile.dt_hours
    )
    required_container = 2.0 / container_service
    required_lohc = 3.0
    outbound_normalized = required_lohc * lohc_service / case.lohc.capacity_kg

    solution = {}
    for period in range(horizon + 1):
        solution[("backlog", period)] = 0.0
    for period in range(horizon):
        solution[("agv_container_count", period)] = 0.0
        solution[("agv_lohc_count", period)] = 0.0
        solution[("lohc_outbound_normalized", period)] = 0.0
    solution[("agv_container_count", 0)] = required_container + 1.5
    solution[("agv_lohc_count", 0)] = required_lohc + 2.0
    solution[("lohc_outbound_normalized", 0)] = outbound_normalized

    metrics = metrics_module.measure_empty_running(case, realization, solution)

    assert metrics.total_empty_vehicle_periods == pytest.approx(3.5)
    assert metrics.container_empty_vehicle_periods[0] == pytest.approx(1.5)
    assert metrics.lohc_empty_vehicle_periods[0] == pytest.approx(2.0)
    assert metrics.maximum_capacity_shortfall == pytest.approx(0.0)
    assert metrics.empty_battery_energy_kwh == pytest.approx(
        3.5 * case.agv.run_power_per_vehicle_kw * case.profile.dt_hours
    )
    assert metrics.recharge_energy_equivalent_kwh == pytest.approx(
        metrics.empty_battery_energy_kwh / case.agv.charge_efficiency
    )


def test_empty_running_metric_reports_capacity_shortfall_without_hiding_it(
    toy_case, toy_joint_bundle
) -> None:
    metrics_module = _metrics_module()
    case = toy_case()
    realization = dict(
        toy_joint_bundle.evaluate(toy_joint_bundle.nominal_selector)
    )
    horizon = case.profile.periods
    realization["ship.task_release"] = [1.0] + [0.0] * (horizon - 1)
    solution = {
        ("backlog", period): 0.0 for period in range(horizon + 1)
    }
    for period in range(horizon):
        solution[("agv_container_count", period)] = 0.0
        solution[("agv_lohc_count", period)] = 0.0
        solution[("lohc_outbound_normalized", period)] = 0.0

    metrics = metrics_module.measure_empty_running(case, realization, solution)

    assert metrics.maximum_capacity_shortfall > 0.0
    assert metrics.total_empty_vehicle_periods == pytest.approx(0.0)
