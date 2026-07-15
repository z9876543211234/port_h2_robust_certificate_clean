"""Backlog state and exact task-done elimination rows."""

from __future__ import annotations


def register(case, bundle, registry) -> None:
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    capacity = case.logistics.backlog_capacity
    scale = case.cost_scale
    for period in range(horizon + 1):
        coefficient = (
            case.cost.terminal_backlog_per_task * capacity / scale
            if period == horizon
            else case.cost.backlog_delay_per_task_hour * capacity * dt / scale
        )
        registry.add_variable(("backlog", period), 0.0, 1.0, coefficient, "p.u.", "logistics")
    registry.add_row("logistics.backlog_initial", "eq", {("backlog", 0): 1.0}, case.logistics.initial_backlog / capacity, None, None, "p.u.", "logistics", "B-INIT")
    kappa = case.agv.container_rate_per_vehicle_hour * case.agv.container_turn_efficiency * dt / capacity
    for period in range(horizon):
        task_key = f"ship.task_release[{period}]"
        registry.add_row(f"logistics.backlog_upper_transition[{period}]", "ge", {("backlog", period): 1.0, ("backlog", period + 1): -1.0}, 0.0, None, {task_key: -1.0 / capacity}, "p.u.", "logistics", "B-UP")
        registry.add_row(f"logistics.backlog_lower_transition[{period}]", "ge", {("backlog", period + 1): 1.0, ("backlog", period): -1.0, ("agv_container_count", period): kappa}, 0.0, None, {task_key: 1.0 / capacity}, "p.u.", "logistics", "B-LOW")

