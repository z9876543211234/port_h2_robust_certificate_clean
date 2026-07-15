"""Anonymous unit-ship delay selectors, budgets, and symmetry rows."""

from __future__ import annotations

from dataclasses import dataclass

from port_h2_contracts.uncertainty_bundle import LinearConstraint, SelectorSpec
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource


@dataclass(frozen=True)
class DelayVariable:
    key: str
    source_period: int
    anonymous_ship_index: int
    delay_steps: int


def build_delay_variables(
    source: ShipDelaySource,
) -> tuple[list[DelayVariable], list[SelectorSpec], list[LinearConstraint]]:
    variables: list[DelayVariable] = []
    selectors: list[SelectorSpec] = []
    constraints: list[LinearConstraint] = []

    by_ship: dict[tuple[int, int], list[DelayVariable]] = {}
    for source_period, ship_count in enumerate(source.current_day_arrival_count):
        for ship_index in range(ship_count):
            entries: list[DelayVariable] = []
            for delay in range(1, source.max_delay_steps + 1):
                key = f"ship.delay[{source_period},{ship_index},{delay}]"
                variable = DelayVariable(key, source_period, ship_index, delay)
                variables.append(variable)
                entries.append(variable)
                selectors.append(SelectorSpec(key, "ship_delay", "binary", "primary"))
            by_ship[(source_period, ship_index)] = entries
            constraints.append(
                LinearConstraint(
                    name=f"ship.one_delay[{source_period},{ship_index}]",
                    coefficients={entry.key: 1.0 for entry in entries},
                    sense="le",
                    rhs=1.0,
                )
            )

    constraints.append(
        LinearConstraint(
            name="ship.delayed_ship_budget",
            coefficients={entry.key: 1.0 for entry in variables},
            sense="le",
            rhs=float(source.delayed_ship_budget),
        )
    )
    if source.total_delay_step_budget is not None:
        constraints.append(
            LinearConstraint(
                name="ship.total_delay_step_budget",
                coefficients={entry.key: float(entry.delay_steps) for entry in variables},
                sense="le",
                rhs=float(source.total_delay_step_budget),
            )
        )

    for source_period, ship_count in enumerate(source.current_day_arrival_count):
        for ship_index in range(ship_count - 1):
            current = by_ship[(source_period, ship_index)]
            following = by_ship[(source_period, ship_index + 1)]
            active_coefficients = {entry.key: 1.0 for entry in current}
            active_coefficients.update({entry.key: -1.0 for entry in following})
            delay_coefficients = {
                entry.key: float(entry.delay_steps) for entry in current
            }
            delay_coefficients.update(
                {entry.key: -float(entry.delay_steps) for entry in following}
            )
            constraints.extend(
                [
                    LinearConstraint(
                        name=f"ship.symmetry_active[{source_period},{ship_index}]",
                        coefficients=active_coefficients,
                        sense="ge",
                        rhs=0.0,
                    ),
                    LinearConstraint(
                        name=f"ship.symmetry_delay[{source_period},{ship_index}]",
                        coefficients=delay_coefficients,
                        sense="ge",
                        rhs=0.0,
                    ),
                ]
            )

    return variables, selectors, constraints

