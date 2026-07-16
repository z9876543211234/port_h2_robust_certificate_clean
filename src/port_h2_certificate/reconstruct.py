"""Restore eliminated physical quantities and original units after an LP replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from port_h2_certificate.recourse_ir import VariableKey
from port_h2_certificate.schema import CaseData
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class ReconstructedRecourse:
    series: Mapping[str, tuple[float, ...]]
    recourse_objective: float
    cost_breakdown: Mapping[str, float]


def _value(values: Mapping[VariableKey, float], family: str, period: int) -> float:
    return float(values[(family, period)])


def reconstruct_recourse(
    case: CaseData,
    bundle: UncertaintyBundle,
    first_stage: Mapping[VariableKey, float],
    realization: Mapping[str, Sequence[float]],
    solution: Mapping[VariableKey, float],
) -> ReconstructedRecourse:
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    scale = case.cost_scale
    fixed_hydrogen_series = tuple(
        value / 1000.0 for value in case.hydrogen_power_day_ahead_kw(bundle)
    )

    grid_rt: list[float] = []
    grid_buy: list[float] = []
    grid_sell: list[float] = []
    charge_power: list[float] = []
    idle: list[float] = []
    task_done: list[float] = []
    grid_positive: list[float] = []
    grid_negative: list[float] = []
    spill_positive: list[float] = []
    spill_negative: list[float] = []
    lohc_positive: list[float] = []
    lohc_negative: list[float] = []
    charge_positive: list[float] = []
    charge_negative: list[float] = []
    lohc_production: list[float] = []
    hydrogen_to_lohc: list[float] = []
    lohc_outbound: list[float] = []

    p_charge_mw = case.agv.charge_power_per_vehicle_kw / 1000.0
    for period in range(horizon):
        charge = p_charge_mw * _value(solution, "agv_charge_count", period)
        charge_power.append(charge)
        lohc_power = (
            _value(solution, "lohc_power_rt_mw", period)
            if case.has_hydrogen_chain
            else 0.0
        )
        grid = (
            case.base_load_kw[period] / 1000.0
            + fixed_hydrogen_series[period]
            + float(realization["ship.shore_power_kw"][period]) / 1000.0
            + float(realization["ship.quay_crane_power_kw"][period]) / 1000.0
            + lohc_power
            + charge
            + _value(solution, "spill_rt_mw", period)
            - float(realization["wind.available_power_kw"][period]) / 1000.0
        )
        grid_rt.append(grid)
        grid_buy.append(max(grid, 0.0))
        grid_sell.append(max(-grid, 0.0))
        grid_da = _value(first_stage, "grid_buy_da_mw", period) - _value(
            first_stage, "grid_sell_da_mw", period
        )
        grid_positive.append(max(grid - grid_da, 0.0))
        grid_negative.append(max(grid_da - grid, 0.0))
        spill_delta = _value(solution, "spill_rt_mw", period) - _value(
            first_stage, "spill_da_mw", period
        )
        spill_positive.append(max(spill_delta, 0.0))
        spill_negative.append(max(-spill_delta, 0.0))

        lohc_count = (
            _value(solution, "agv_lohc_count", period)
            if case.has_hydrogen_chain
            else 0.0
        )
        idle.append(
            case.agv.fleet_size
            - _value(solution, "agv_container_count", period)
            - lohc_count
            - _value(solution, "agv_charge_count", period)
        )
        backlog_now = _value(solution, "backlog", period) * case.logistics.backlog_capacity
        backlog_next = _value(solution, "backlog", period + 1) * case.logistics.backlog_capacity
        task_done.append(
            backlog_now
            + float(realization["ship.task_release"][period])
            - backlog_next
        )
        charge_delta = charge - p_charge_mw * _value(
            first_stage, "agv_charge_count_da", period
        )
        charge_positive.append(max(charge_delta, 0.0))
        charge_negative.append(max(-charge_delta, 0.0))

        if case.has_hydrogen_chain:
            assert case.hydrogen is not None and case.lohc is not None
            lohc_delta = lohc_power - _value(
                first_stage, "lohc_power_da_mw", period
            )
            lohc_positive.append(max(lohc_delta, 0.0))
            lohc_negative.append(max(-lohc_delta, 0.0))
            production = 1000.0 * lohc_power * dt / case.lohc.energy_kwh_per_kg
            lohc_production.append(production)
            hydrogen_to_lohc.append(production / case.lohc.hydrogen_yield)
            lohc_outbound.append(
                _value(solution, "lohc_outbound_normalized", period)
                * case.lohc.capacity_kg
            )

    backlog = tuple(
        _value(solution, "backlog", period) * case.logistics.backlog_capacity
        for period in range(horizon + 1)
    )
    soc = tuple(_value(solution, "soc", period) for period in range(horizon + 1))
    series: dict[str, tuple[float, ...]] = {
        "grid_rt_mw": tuple(grid_rt),
        "grid_buy_rt_mw": tuple(grid_buy),
        "grid_sell_rt_mw": tuple(grid_sell),
        "agv_charge_power_rt_mw": tuple(charge_power),
        "agv_idle_count": tuple(idle),
        "task_done": tuple(task_done),
        "grid_deviation_positive_mw": tuple(grid_positive),
        "grid_deviation_negative_mw": tuple(grid_negative),
        "spill_deviation_positive_mw": tuple(spill_positive),
        "spill_deviation_negative_mw": tuple(spill_negative),
        "charge_deviation_positive_mw": tuple(charge_positive),
        "charge_deviation_negative_mw": tuple(charge_negative),
        "backlog_tasks": backlog,
        "soc": soc,
    }
    if case.has_hydrogen_chain:
        assert case.hydrogen is not None and case.lohc is not None
        series.update(
            {
                "lohc_deviation_positive_mw": tuple(lohc_positive),
                "lohc_deviation_negative_mw": tuple(lohc_negative),
                "hydrogen_to_lohc_kg": tuple(hydrogen_to_lohc),
                "lohc_production_kg": tuple(lohc_production),
                "lohc_outbound_kg": tuple(lohc_outbound),
                "hydrogen_inventory_kg": tuple(
                    _value(solution, "hydrogen_inventory", period)
                    * case.hydrogen.capacity_kg
                    for period in range(horizon + 1)
                ),
                "lohc_inventory_kg": tuple(
                    _value(solution, "lohc_inventory", period) * case.lohc.capacity_kg
                    for period in range(horizon + 1)
                ),
            }
        )

    costs = {
        "grid_deviation": sum(
            case.cost.grid_deviation_per_kwh
            * _value(solution, "grid_deviation_mw", period)
            * 1000.0
            * dt
            / scale
            for period in range(horizon)
        ),
        "spill_real_time": sum(
            case.cost.spill_real_time_per_kwh
            * _value(solution, "spill_rt_mw", period)
            * 1000.0
            * dt
            / scale
            for period in range(horizon)
        ),
        "spill_deviation": sum(
            case.cost.spill_deviation_per_kwh
            * _value(solution, "spill_deviation_mw", period)
            * 1000.0
            * dt
            / scale
            for period in range(horizon)
        ),
        "charge_deviation": sum(
            case.cost.charge_adjustment_per_kwh
            * _value(solution, "charge_deviation_mw", period)
            * 1000.0
            * dt
            / scale
            for period in range(horizon)
        ),
        "agv_operation": sum(
            case.cost.agv_operation_per_vehicle_hour
            * (
                _value(solution, "agv_container_count", period)
                + (
                    _value(solution, "agv_lohc_count", period)
                    if case.has_hydrogen_chain
                    else 0.0
                )
            )
            * dt
            / scale
            for period in range(horizon)
        ),
        "backlog_delay": sum(
            case.cost.backlog_delay_per_task_hour * backlog[period] * dt / scale
            for period in range(horizon)
        )
        + case.cost.terminal_backlog_per_task * backlog[horizon] / scale,
    }
    if case.has_hydrogen_chain:
        costs["lohc_deviation"] = sum(
            case.cost.lohc_adjustment_per_kwh
            * _value(solution, "lohc_deviation_mw", period)
            * 1000.0
            * dt
            / scale
            for period in range(horizon)
        )
        costs["lohc_export_revenue"] = -sum(
            case.cost.lohc_export_revenue_per_kg * value / scale
            for value in lohc_outbound
        )
    objective = float(sum(costs.values()))
    return ReconstructedRecourse(series, objective, costs)
