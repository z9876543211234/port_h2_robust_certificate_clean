"""Pooled AGV allocation, charging adjustment, and SOC rows."""

from __future__ import annotations


def register(case, bundle, registry) -> None:
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    agv = case.agv
    scale = case.cost_scale
    p_charge_mw = agv.charge_power_per_vehicle_kw / 1000.0
    p_run_mw = agv.run_power_per_vehicle_kw / 1000.0
    battery_mwh = agv.battery_capacity_per_vehicle_kwh / 1000.0
    operation_coefficient = (
        case.cost.agv_operation_per_vehicle_hour * dt / scale
    )
    for period in range(horizon):
        registry.add_variable(("agv_container_count", period), 0.0, agv.fleet_size, operation_coefficient, "vehicle", "agv")
        if case.has_hydrogen_chain:
            registry.add_variable(("agv_lohc_count", period), 0.0, agv.fleet_size, operation_coefficient, "vehicle", "agv")
        registry.add_variable(("agv_charge_count", period), 0.0, agv.charger_count, 0.0, "vehicle", "agv")
        registry.add_variable(("charge_deviation_mw", period), 0.0, agv.charge_adjustment_limit_kw / 1000.0, case.cost.charge_adjustment_per_kwh * 1000.0 * dt / scale, "MW", "agv")
    for period in range(horizon + 1):
        registry.add_variable(("soc", period), agv.min_soc, agv.max_soc, 0.0, "p.u.", "agv")
    registry.add_row("agv.soc_initial", "eq", {("soc", 0): 1.0}, agv.initial_soc, None, None, "p.u.", "agv", "SOC-INIT")
    for period in range(horizon):
        allocations = {("agv_container_count", period): -1.0, ("agv_charge_count", period): -1.0}
        if case.has_hydrogen_chain:
            allocations[("agv_lohc_count", period)] = -1.0
        registry.add_row(f"agv.shared_fleet[{period}]", "ge", allocations, -agv.fleet_size, None, None, "vehicle", "agv", "AGV-FLEET")
        p_limit = agv.charge_adjustment_limit_kw / 1000.0
        n_rt = ("agv_charge_count", period)
        n_da = ("agv_charge_count_da", period)
        d_ch = ("charge_deviation_mw", period)
        registry.add_row(f"agv.charge_adjust_positive[{period}]", "ge", {n_rt: -p_charge_mw}, -p_limit, {n_da: -p_charge_mw}, None, "MW", "agv", "AGV-CH-ADJ+")
        registry.add_row(f"agv.charge_adjust_negative[{period}]", "ge", {n_rt: p_charge_mw}, -p_limit, {n_da: p_charge_mw}, None, "MW", "agv", "AGV-CH-ADJ-")
        registry.add_row(f"agv.charge_deviation_positive[{period}]", "ge", {d_ch: 1.0, n_rt: -p_charge_mw}, 0.0, {n_da: -p_charge_mw}, None, "MW", "agv", "AGV-CH-DEV+")
        registry.add_row(f"agv.charge_deviation_negative[{period}]", "ge", {d_ch: 1.0, n_rt: p_charge_mw}, 0.0, {n_da: p_charge_mw}, None, "MW", "agv", "AGV-CH-DEV-")
        soc_coefficients = {
            ("soc", period + 1): 1.0,
            ("soc", period): -1.0,
            n_rt: -agv.charge_efficiency * p_charge_mw * dt / (agv.fleet_size * battery_mwh),
            ("agv_container_count", period): p_run_mw * dt / (agv.fleet_size * battery_mwh),
        }
        if case.has_hydrogen_chain:
            soc_coefficients[("agv_lohc_count", period)] = p_run_mw * dt / (agv.fleet_size * battery_mwh)
        registry.add_row(f"agv.soc_balance[{period}]", "eq", soc_coefficients, 0.0, None, None, "p.u.", "agv", "SOC-BAL")
        if case.case_name == "C3_WorkCapacityCapsRobust":
            registry.add_row(f"agv.container_work_capacity[{period}]", "ge", {("agv_container_count", period): -1.0}, -case.container_work_capacity, None, None, "vehicle", "agv", "C3-CTN-CAP")
            registry.add_row(f"agv.lohc_work_capacity[{period}]", "ge", {("agv_lohc_count", period): -1.0}, -case.lohc_work_capacity, None, None, "vehicle", "agv", "C3-LOHC-CAP")
    if agv.charge_count_ramp is not None:
        for period in range(1, horizon):
            registry.add_row(f"agv.charge_ramp_positive[{period}]", "ge", {("agv_charge_count", period): -1.0, ("agv_charge_count", period - 1): 1.0}, -agv.charge_count_ramp, None, None, "vehicle", "agv", "AGV-CH-RAMP+")
            registry.add_row(f"agv.charge_ramp_negative[{period}]", "ge", {("agv_charge_count", period): 1.0, ("agv_charge_count", period - 1): -1.0}, -agv.charge_count_ramp, None, None, "vehicle", "agv", "AGV-CH-RAMP-")
    registry.add_row("agv.soc_terminal_min", "ge", {("soc", horizon): 1.0}, agv.terminal_min_soc, None, None, "p.u.", "agv", "SOC-TERM")
