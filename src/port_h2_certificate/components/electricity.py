"""Real-time grid expression, spill, and absolute deviation rows."""

from __future__ import annotations


def register(case, bundle, registry) -> None:
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    scale = case.cost_scale
    h_power_mw = case.hydrogen_power_day_ahead_kw(bundle) / 1000.0
    p_charge_mw = case.agv.charge_power_per_vehicle_kw / 1000.0
    for period in range(horizon):
        registry.add_variable(("spill_rt_mw", period), 0.0, None, case.cost.spill_real_time_per_kwh * 1000.0 * dt / scale, "MW", "electricity")
        registry.add_variable(("grid_deviation_mw", period), 0.0, None, case.cost.grid_deviation_per_kwh * 1000.0 * dt / scale, "MW", "electricity")
        registry.add_variable(("spill_deviation_mw", period), 0.0, None, case.cost.spill_deviation_per_kwh * 1000.0 * dt / scale, "MW", "electricity")
        y_grid = {("spill_rt_mw", period): 1.0, ("agv_charge_count", period): p_charge_mw}
        if case.has_hydrogen_chain:
            y_grid[("lohc_power_rt_mw", period)] = 1.0
        base_h = case.base_load_kw[period] / 1000.0 + h_power_mw[period]
        uncertain_g = {
            f"ship.shore_power_kw[{period}]": 0.001,
            f"ship.quay_crane_power_kw[{period}]": 0.001,
            f"wind.available_power_kw[{period}]": -0.001,
        }
        registry.add_row(f"electricity.grid_lower[{period}]", "ge", y_grid, -case.grid.sell_capacity_kw / 1000.0 - base_h, None, {key: -value for key, value in uncertain_g.items()}, "MW", "electricity", "RT-GRID-LOW")
        registry.add_row(f"electricity.grid_upper[{period}]", "ge", {key: -value for key, value in y_grid.items()}, -case.grid.buy_capacity_kw / 1000.0 + base_h, None, uncertain_g, "MW", "electricity", "RT-GRID-UP")
        registry.add_row(f"electricity.grid_deviation_positive[{period}]", "ge", {("grid_deviation_mw", period): 1.0, **{key: -value for key, value in y_grid.items()}}, base_h, {("grid_buy_da_mw", period): -1.0, ("grid_sell_da_mw", period): 1.0}, uncertain_g, "MW", "electricity", "RT-GRID-DEV+")
        registry.add_row(f"electricity.grid_deviation_negative[{period}]", "ge", {("grid_deviation_mw", period): 1.0, **y_grid}, -base_h, {("grid_buy_da_mw", period): 1.0, ("grid_sell_da_mw", period): -1.0}, {key: -value for key, value in uncertain_g.items()}, "MW", "electricity", "RT-GRID-DEV-")
        registry.add_row(f"electricity.spill_upper[{period}]", "ge", {("spill_rt_mw", period): -1.0}, h_power_mw[period], None, {f"wind.available_power_kw[{period}]": -0.001}, "MW", "electricity", "RT-SPILL-UP")
        registry.add_row(f"electricity.spill_deviation_positive[{period}]", "ge", {("spill_deviation_mw", period): 1.0, ("spill_rt_mw", period): -1.0}, 0.0, {("spill_da_mw", period): -1.0}, None, "MW", "electricity", "RT-SPILL-DEV+")
        registry.add_row(f"electricity.spill_deviation_negative[{period}]", "ge", {("spill_deviation_mw", period): 1.0, ("spill_rt_mw", period): 1.0}, 0.0, {("spill_da_mw", period): 1.0}, None, "MW", "electricity", "RT-SPILL-DEV-")

