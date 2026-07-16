"""Test-only explicit recourse model; never imported by production solvers."""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True)
class ExplicitResult:
    objective: float
    series: dict[str, tuple[float, ...]]


def solve_explicit(
    case, bundle, first_stage, realization, fixed_series=None
) -> ExplicitResult:
    assert case.has_hydrogen_chain
    assert case.hydrogen is not None and case.lohc is not None
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    scale = case.cost_scale
    agv = case.agv
    h2 = case.hydrogen
    lohc = case.lohc
    model = gp.Model("test_only_explicit_recourse")
    model.Params.OutputFlag = 0
    model.Params.Method = 1
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8

    grid = model.addVars(
        horizon,
        lb=-case.grid.sell_capacity_kw / 1000.0,
        ub=case.grid.buy_capacity_kw / 1000.0,
        name="grid_rt_mw",
    )
    spill = model.addVars(horizon, lb=0.0, name="spill_rt_mw")
    p_lohc = model.addVars(
        horizon,
        lb=lohc.reactor_min_kw / 1000.0,
        ub=lohc.reactor_max_kw / 1000.0,
        name="lohc_power_rt_mw",
    )
    outbound = model.addVars(horizon, lb=0.0, ub=lohc.capacity_kg, name="lohc_outbound_kg")
    n_container = model.addVars(horizon, lb=0.0, ub=agv.fleet_size, name="agv_container_count")
    n_lohc = model.addVars(horizon, lb=0.0, ub=agv.fleet_size, name="agv_lohc_count")
    n_charge = model.addVars(horizon, lb=0.0, ub=agv.charger_count, name="agv_charge_count")
    n_idle = model.addVars(horizon, lb=0.0, ub=agv.fleet_size, name="agv_idle_count")
    p_charge = model.addVars(
        horizon,
        lb=0.0,
        ub=agv.charge_power_per_vehicle_kw * agv.charger_count / 1000.0,
        name="agv_charge_power_rt_mw",
    )
    h2_to_lohc = model.addVars(horizon, lb=0.0, name="hydrogen_to_lohc_kg")
    lohc_prod = model.addVars(horizon, lb=0.0, name="lohc_production_kg")
    task_done = model.addVars(horizon, lb=0.0, name="task_done")
    h_inventory = model.addVars(horizon + 1, lb=0.0, ub=1.0, name="hydrogen_inventory")
    l_inventory = model.addVars(horizon + 1, lb=0.0, ub=1.0, name="lohc_inventory")
    soc = model.addVars(horizon + 1, lb=agv.min_soc, ub=agv.max_soc, name="soc")
    backlog = model.addVars(horizon + 1, lb=0.0, ub=1.0, name="backlog")
    deviation_families = {
        name: model.addVars(horizon, lb=0.0, name=name)
        for name in (
            "grid_dev_positive",
            "grid_dev_negative",
            "spill_dev_positive",
            "spill_dev_negative",
            "lohc_dev_positive",
            "lohc_dev_negative",
            "charge_dev_positive",
            "charge_dev_negative",
        )
    }

    landing = case.hydrogen_landing_kg(bundle)
    fixed_h2_mw = case.hydrogen_power_day_ahead_kw(bundle) / 1000.0
    p_charge_single = agv.charge_power_per_vehicle_kw / 1000.0
    p_run = agv.run_power_per_vehicle_kw / 1000.0
    battery = agv.battery_capacity_per_vehicle_kwh / 1000.0
    model.addConstr(h_inventory[0] == h2.initial_kg / h2.capacity_kg)
    model.addConstr(l_inventory[0] == lohc.initial_kg / lohc.capacity_kg)
    model.addConstr(soc[0] == agv.initial_soc)
    model.addConstr(backlog[0] == case.logistics.initial_backlog / case.logistics.backlog_capacity)
    for period in range(horizon):
        model.addConstr(p_charge[period] == p_charge_single * n_charge[period])
        model.addConstr(
            n_container[period] + n_lohc[period] + n_charge[period] + n_idle[period]
            == agv.fleet_size
        )
        model.addConstr(
            grid[period]
            == case.base_load_kw[period] / 1000.0
            + fixed_h2_mw[period]
            + realization["ship.shore_power_kw"][period] / 1000.0
            + realization["ship.quay_crane_power_kw"][period] / 1000.0
            + p_lohc[period]
            + p_charge[period]
            + spill[period]
            - realization["wind.available_power_kw"][period] / 1000.0
        )
        model.addConstr(
            spill[period]
            <= realization["wind.available_power_kw"][period] / 1000.0
            - fixed_h2_mw[period]
        )
        grid_da = first_stage[("grid_buy_da_mw", period)] - first_stage[("grid_sell_da_mw", period)]
        model.addConstr(
            deviation_families["grid_dev_positive"][period]
            - deviation_families["grid_dev_negative"][period]
            == grid[period] - grid_da
        )
        model.addConstr(
            deviation_families["spill_dev_positive"][period]
            - deviation_families["spill_dev_negative"][period]
            == spill[period] - first_stage[("spill_da_mw", period)]
        )
        model.addConstr(
            deviation_families["lohc_dev_positive"][period]
            - deviation_families["lohc_dev_negative"][period]
            == p_lohc[period] - first_stage[("lohc_power_da_mw", period)]
        )
        model.addConstr(
            deviation_families["charge_dev_positive"][period]
            - deviation_families["charge_dev_negative"][period]
            == p_charge[period]
            - p_charge_single * first_stage[("agv_charge_count_da", period)]
        )
        model.addConstr(
            deviation_families["lohc_dev_positive"][period]
            + deviation_families["lohc_dev_negative"][period]
            <= lohc.adjustment_limit_kw / 1000.0
        )
        model.addConstr(
            deviation_families["charge_dev_positive"][period]
            + deviation_families["charge_dev_negative"][period]
            <= agv.charge_adjustment_limit_kw / 1000.0
        )
        model.addConstr(
            h2_to_lohc[period]
            == p_lohc[period] * 1000.0 * dt / lohc.energy_kwh_per_kg / lohc.hydrogen_yield
        )
        model.addConstr(
            lohc_prod[period]
            == p_lohc[period] * 1000.0 * dt / lohc.energy_kwh_per_kg
        )
        model.addConstr(
            h_inventory[period + 1]
            == h_inventory[period]
            + landing[period] / h2.capacity_kg
            - h2_to_lohc[period] / h2.capacity_kg
        )
        model.addConstr(
            l_inventory[period + 1]
            == l_inventory[period]
            + lohc_prod[period] / lohc.capacity_kg
            - outbound[period] / lohc.capacity_kg
        )
        model.addConstr(outbound[period] <= l_inventory[period] * lohc.capacity_kg)
        model.addConstr(
            outbound[period]
            <= agv.lohc_transport_rate_kg_per_vehicle_hour
            * agv.lohc_turn_efficiency
            * dt
            * n_lohc[period]
        )
        model.addConstr(
            soc[period + 1]
            == soc[period]
            + agv.charge_efficiency
            * p_charge[period]
            * dt
            / (agv.fleet_size * battery)
            - p_run
            * (n_container[period] + n_lohc[period])
            * dt
            / (agv.fleet_size * battery)
        )
        capacity = (
            agv.container_rate_per_vehicle_hour
            * agv.container_turn_efficiency
            * dt
            * n_container[period]
        )
        model.addConstr(task_done[period] <= capacity)
        model.addConstr(
            backlog[period + 1]
            == backlog[period]
            + realization["ship.task_release"][period] / case.logistics.backlog_capacity
            - task_done[period] / case.logistics.backlog_capacity
        )
        if case.case_name == "C3_WorkCapacityCapsRobust":
            model.addConstr(n_container[period] <= case.container_work_capacity)
            model.addConstr(n_lohc[period] <= case.lohc_work_capacity)

    for period in range(1, horizon):
        model.addConstr(
            p_lohc[period] - p_lohc[period - 1]
            <= lohc.real_time_ramp_kw / 1000.0
        )
        model.addConstr(
            p_lohc[period - 1] - p_lohc[period]
            <= lohc.real_time_ramp_kw / 1000.0
        )
        if agv.charge_count_ramp is not None:
            model.addConstr(n_charge[period] - n_charge[period - 1] <= agv.charge_count_ramp)
            model.addConstr(n_charge[period - 1] - n_charge[period] <= agv.charge_count_ramp)
    model.addConstr(h_inventory[horizon] >= h2.terminal_min_kg / h2.capacity_kg)
    model.addConstr(l_inventory[horizon] >= lohc.terminal_min_kg / lohc.capacity_kg)
    model.addConstr(soc[horizon] >= agv.terminal_min_soc)

    if fixed_series is not None:
        variable_groups = {
            "grid_rt_mw": grid,
            "agv_charge_power_rt_mw": p_charge,
            "agv_idle_count": n_idle,
            "task_done": task_done,
            "hydrogen_to_lohc_kg": h2_to_lohc,
            "lohc_production_kg": lohc_prod,
        }
        for key, variables in variable_groups.items():
            for period, value in enumerate(fixed_series[key]):
                model.addConstr(variables[period] == value, name=f"fix.{key}[{period}]")
        state_groups = {
            "hydrogen_inventory_kg": (h_inventory, h2.capacity_kg),
            "lohc_inventory_kg": (l_inventory, lohc.capacity_kg),
            "backlog_tasks": (backlog, case.logistics.backlog_capacity),
            "soc": (soc, 1.0),
        }
        for key, (variables, factor) in state_groups.items():
            for period, value in enumerate(fixed_series[key]):
                model.addConstr(
                    variables[period] * factor == value,
                    name=f"fix.{key}[{period}]",
                )

    objective = gp.LinExpr()
    for period in range(horizon):
        objective += (
            case.cost.grid_deviation_per_kwh
            * 1000.0
            * dt
            / scale
            * (
                deviation_families["grid_dev_positive"][period]
                + deviation_families["grid_dev_negative"][period]
            )
        )
        objective += case.cost.spill_real_time_per_kwh * 1000.0 * dt / scale * spill[period]
        objective += (
            case.cost.spill_deviation_per_kwh
            * 1000.0
            * dt
            / scale
            * (
                deviation_families["spill_dev_positive"][period]
                + deviation_families["spill_dev_negative"][period]
            )
        )
        objective += (
            case.cost.lohc_adjustment_per_kwh
            * 1000.0
            * dt
            / scale
            * (
                deviation_families["lohc_dev_positive"][period]
                + deviation_families["lohc_dev_negative"][period]
            )
        )
        objective += (
            case.cost.charge_adjustment_per_kwh
            * 1000.0
            * dt
            / scale
            * (
                deviation_families["charge_dev_positive"][period]
                + deviation_families["charge_dev_negative"][period]
            )
        )
        objective += (
            case.cost.agv_operation_per_vehicle_hour
            * dt
            / scale
            * (n_container[period] + n_lohc[period])
        )
        objective += (
            case.cost.backlog_delay_per_task_hour
            * case.logistics.backlog_capacity
            * dt
            / scale
            * backlog[period]
        )
        objective -= case.cost.lohc_export_revenue_per_kg / scale * outbound[period]
    objective += (
        case.cost.terminal_backlog_per_task
        * case.logistics.backlog_capacity
        / scale
        * backlog[horizon]
    )
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"explicit reference model status {model.Status}")
    return ExplicitResult(
        objective=model.ObjVal,
        series={
            "grid_rt_mw": tuple(grid[t].X for t in range(horizon)),
            "agv_charge_power_rt_mw": tuple(p_charge[t].X for t in range(horizon)),
            "agv_idle_count": tuple(n_idle[t].X for t in range(horizon)),
            "task_done": tuple(task_done[t].X for t in range(horizon)),
            "hydrogen_to_lohc_kg": tuple(h2_to_lohc[t].X for t in range(horizon)),
            "lohc_production_kg": tuple(lohc_prod[t].X for t in range(horizon)),
            "hydrogen_inventory_kg": tuple(h_inventory[t].X * h2.capacity_kg for t in range(horizon + 1)),
            "lohc_inventory_kg": tuple(l_inventory[t].X * lohc.capacity_kg for t in range(horizon + 1)),
            "backlog_tasks": tuple(backlog[t].X * case.logistics.backlog_capacity for t in range(horizon + 1)),
            "soc": tuple(soc[t].X for t in range(horizon + 1)),
        },
    )
