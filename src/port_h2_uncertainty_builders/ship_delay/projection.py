"""Three-day arrival construction and current-day affine projections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from port_h2_uncertainty_builders.ship_delay.delay_variables import DelayVariable
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource


@dataclass(frozen=True)
class ShipProjection:
    extended_nominal: np.ndarray
    extended_columns: np.ndarray
    arrival_nominal: np.ndarray
    arrival_columns: np.ndarray
    in_port_nominal: np.ndarray
    in_port_columns: np.ndarray
    delayed_out_nominal: np.ndarray
    delayed_out_columns: np.ndarray


def build_ship_projection(
    source: ShipDelaySource, delay_variables: list[DelayVariable]
) -> ShipProjection:
    horizon = source.profile.periods
    extended_nominal = np.asarray(
        source.previous_day_arrival_count
        + source.current_day_arrival_count
        + source.next_day_arrival_count,
        dtype=float,
    )
    extended_columns = np.zeros((3 * horizon, len(delay_variables)), dtype=float)
    delayed_out_columns = np.zeros((1, len(delay_variables)), dtype=float)
    for column, variable in enumerate(delay_variables):
        origin = horizon + variable.source_period
        target = origin + variable.delay_steps
        extended_columns[origin, column] = -1.0
        extended_columns[target, column] = 1.0
        delayed_out_columns[0, column] = float(target >= 2 * horizon)

    arrival_nominal = extended_nominal[horizon : 2 * horizon].copy()
    arrival_columns = extended_columns[horizon : 2 * horizon, :].copy()
    in_port_nominal = np.zeros(horizon, dtype=float)
    in_port_columns = np.zeros((horizon, len(delay_variables)), dtype=float)
    for period in range(horizon):
        indices = [
            horizon + period - lag for lag in range(source.dwell_steps)
        ]
        in_port_nominal[period] = float(np.sum(extended_nominal[indices]))
        in_port_columns[period, :] = np.sum(extended_columns[indices, :], axis=0)

    return ShipProjection(
        extended_nominal=extended_nominal,
        extended_columns=extended_columns,
        arrival_nominal=arrival_nominal,
        arrival_columns=arrival_columns,
        in_port_nominal=in_port_nominal,
        in_port_columns=in_port_columns,
        delayed_out_nominal=np.zeros(1, dtype=float),
        delayed_out_columns=delayed_out_columns,
    )

