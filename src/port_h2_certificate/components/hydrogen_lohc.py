"""H2 landing, H2/LOHC inventories, reactor, and outbound rows."""

from __future__ import annotations


def register(case, bundle, registry) -> None:
    if not case.has_hydrogen_chain:
        return
    assert case.hydrogen is not None and case.lohc is not None
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    scale = case.cost_scale
    hydrogen = case.hydrogen
    lohc = case.lohc
    landing = case.hydrogen_landing_kg(bundle)

    for period in range(horizon):
        registry.add_variable(
            ("lohc_power_rt_mw", period),
            lohc.reactor_min_kw / 1000.0,
            lohc.reactor_max_kw / 1000.0,
            0.0,
            "MW",
            "lohc",
        )
        registry.add_variable(
            ("lohc_outbound_normalized", period),
            0.0,
            1.0,
            -case.cost.lohc_export_revenue_per_kg * lohc.capacity_kg / scale,
            "p.u.",
            "lohc",
        )
        registry.add_variable(
            ("lohc_deviation_mw", period),
            0.0,
            lohc.adjustment_limit_kw / 1000.0,
            case.cost.lohc_adjustment_per_kwh * 1000.0 * dt / scale,
            "MW",
            "lohc",
        )
    for period in range(horizon + 1):
        registry.add_variable(("hydrogen_inventory", period), 0.0, 1.0, 0.0, "p.u.", "hydrogen")
        registry.add_variable(("lohc_inventory", period), 0.0, 1.0, 0.0, "p.u.", "lohc")

    registry.add_row(
        "hydrogen.initial",
        "eq",
        {("hydrogen_inventory", 0): 1.0},
        hydrogen.initial_kg / hydrogen.capacity_kg,
        None,
        None,
        "p.u.",
        "hydrogen",
        "H2-INIT",
    )
    registry.add_row(
        "lohc.initial",
        "eq",
        {("lohc_inventory", 0): 1.0},
        lohc.initial_kg / lohc.capacity_kg,
        None,
        None,
        "p.u.",
        "lohc",
        "LOHC-INIT",
    )
    h_consumption = 1000.0 * dt / lohc.energy_kwh_per_kg / lohc.hydrogen_yield / hydrogen.capacity_kg
    l_production = 1000.0 * dt / lohc.energy_kwh_per_kg / lohc.capacity_kg
    for period in range(horizon):
        p_rt = ("lohc_power_rt_mw", period)
        p_da = ("lohc_power_da_mw", period)
        deviation = ("lohc_deviation_mw", period)
        registry.add_row(
            f"hydrogen.balance[{period}]",
            "eq",
            {("hydrogen_inventory", period + 1): 1.0, ("hydrogen_inventory", period): -1.0, p_rt: h_consumption},
            landing[period] / hydrogen.capacity_kg,
            None,
            None,
            "p.u.",
            "hydrogen",
            "H2-BAL",
        )
        registry.add_row(
            f"lohc.balance[{period}]",
            "eq",
            {("lohc_inventory", period + 1): 1.0, ("lohc_inventory", period): -1.0, p_rt: -l_production, ("lohc_outbound_normalized", period): 1.0},
            0.0,
            None,
            None,
            "p.u.",
            "lohc",
            "LOHC-BAL",
        )
        registry.add_row(
            f"lohc.outbound_existing_inventory[{period}]",
            "ge",
            {("lohc_inventory", period): 1.0, ("lohc_outbound_normalized", period): -1.0},
            0.0,
            None,
            None,
            "p.u.",
            "lohc",
            "LOHC-OUT-TIMING",
        )
        limit = lohc.adjustment_limit_kw / 1000.0
        registry.add_row(f"lohc.adjustment_positive[{period}]", "ge", {p_rt: -1.0}, -limit, {p_da: -1.0}, None, "MW", "lohc", "LOHC-ADJ+")
        registry.add_row(f"lohc.adjustment_negative[{period}]", "ge", {p_rt: 1.0}, -limit, {p_da: 1.0}, None, "MW", "lohc", "LOHC-ADJ-")
        registry.add_row(f"lohc.deviation_positive[{period}]", "ge", {deviation: 1.0, p_rt: -1.0}, 0.0, {p_da: -1.0}, None, "MW", "lohc", "LOHC-DEV+")
        registry.add_row(f"lohc.deviation_negative[{period}]", "ge", {deviation: 1.0, p_rt: 1.0}, 0.0, {p_da: 1.0}, None, "MW", "lohc", "LOHC-DEV-")
        if case.outbound_mode == "disabled":
            registry.add_row(f"lohc.outbound_disabled[{period}]", "eq", {("lohc_outbound_normalized", period): 1.0}, 0.0, None, None, "p.u.", "lohc", "LOHC-OUT-OFF")
        elif case.outbound_mode == "fixed_pipeline":
            registry.add_row(f"lohc.fixed_pipeline[{period}]", "ge", {("lohc_outbound_normalized", period): -1.0}, -lohc.fixed_pipeline_rate_kg_per_hour * dt / lohc.capacity_kg, None, None, "p.u.", "lohc", "LOHC-OUT-PIPE")
        else:
            coefficient = lohc.capacity_kg
            transport = case.agv.lohc_transport_rate_kg_per_vehicle_hour * case.agv.lohc_turn_efficiency * dt
            registry.add_row(f"lohc.agv_virtual_pipeline[{period}]", "ge", {("lohc_outbound_normalized", period): -coefficient, ("agv_lohc_count", period): transport}, 0.0, None, None, "kg", "lohc", "LOHC-OUT-AGV")
    for period in range(1, horizon):
        ramp = lohc.real_time_ramp_kw / 1000.0
        registry.add_row(f"lohc.real_time_ramp_positive[{period}]", "ge", {("lohc_power_rt_mw", period): -1.0, ("lohc_power_rt_mw", period - 1): 1.0}, -ramp, None, None, "MW", "lohc", "LOHC-RT-RAMP+")
        registry.add_row(f"lohc.real_time_ramp_negative[{period}]", "ge", {("lohc_power_rt_mw", period): 1.0, ("lohc_power_rt_mw", period - 1): -1.0}, -ramp, None, None, "MW", "lohc", "LOHC-RT-RAMP-")
    registry.add_row("hydrogen.terminal_min", "ge", {("hydrogen_inventory", horizon): 1.0}, hydrogen.terminal_min_kg / hydrogen.capacity_kg, None, None, "p.u.", "hydrogen", "H2-TERM")
    registry.add_row("lohc.terminal_min", "ge", {("lohc_inventory", horizon): 1.0}, lohc.terminal_min_kg / lohc.capacity_kg, None, None, "p.u.", "lohc", "LOHC-TERM")

