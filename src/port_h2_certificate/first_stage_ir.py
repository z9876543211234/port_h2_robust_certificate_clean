"""First-stage continuous linear IR generated from the physical case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from port_h2_contracts.uncertainty_bundle import UncertaintyBundle
from port_h2_certificate.schema import CaseData


VariableKey = tuple[str, int]


@dataclass(frozen=True)
class FirstStageVariableSpec:
    key: VariableKey
    lower_bound: float
    upper_bound: float | None
    objective_coefficient: float
    unit: str
    component: str


@dataclass(frozen=True)
class FirstStageConstraintRow:
    name: str
    sense: Literal["eq", "ge"]
    coefficients: Mapping[VariableKey, float]
    rhs_constant: float
    unit: str
    component: str
    equation_id: str


@dataclass(frozen=True)
class FirstStageIR:
    variables: tuple[FirstStageVariableSpec, ...]
    constraints: tuple[FirstStageConstraintRow, ...]

    @property
    def variable_keys(self) -> tuple[VariableKey, ...]:
        return tuple(variable.key for variable in self.variables)


def build_first_stage_ir(case: CaseData, bundle: UncertaintyBundle) -> FirstStageIR:
    case.validate(bundle)
    variables: list[FirstStageVariableSpec] = []
    rows: list[FirstStageConstraintRow] = []
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    scale = case.cost_scale
    wind_nominal = case.wind_nominal_kw(bundle)
    hydrogen_power = case.hydrogen_power_day_ahead_kw(bundle)
    direct_wind_mw = (wind_nominal - hydrogen_power) / 1000.0
    nominal = bundle.evaluate(bundle.nominal_selector)

    for period in range(horizon):
        variables.extend(
            [
                FirstStageVariableSpec(
                    ("grid_buy_da_mw", period),
                    0.0,
                    case.grid.buy_capacity_kw / 1000.0,
                    case.grid.buy_price_per_kwh[period] * 1000.0 * dt / scale,
                    "MW",
                    "electricity",
                ),
                FirstStageVariableSpec(
                    ("grid_sell_da_mw", period),
                    0.0,
                    case.grid.sell_capacity_kw / 1000.0,
                    -case.grid.sell_price_per_kwh[period] * 1000.0 * dt / scale,
                    "MW",
                    "electricity",
                ),
                FirstStageVariableSpec(
                    ("spill_da_mw", period),
                    0.0,
                    float(direct_wind_mw[period]),
                    case.cost.spill_day_ahead_per_kwh * 1000.0 * dt / scale,
                    "MW",
                    "electricity",
                ),
                FirstStageVariableSpec(
                    ("agv_charge_count_da", period),
                    0.0,
                    case.agv.charger_count,
                    0.0,
                    "vehicle",
                    "agv",
                ),
            ]
        )
        if case.has_hydrogen_chain:
            assert case.lohc is not None
            variables.append(
                FirstStageVariableSpec(
                    ("lohc_power_da_mw", period),
                    case.lohc.reactor_min_kw / 1000.0,
                    case.lohc.reactor_max_kw / 1000.0,
                    case.cost.lohc_operation_per_kg
                    * 1000.0
                    * dt
                    / case.lohc.energy_kwh_per_kg
                    / scale,
                    "MW",
                    "lohc",
                )
            )
            if period > 0:
                variables.append(
                    FirstStageVariableSpec(
                        ("lohc_ramp_abs_da_mw", period),
                        0.0,
                        case.lohc.day_ahead_ramp_kw / 1000.0,
                        case.cost.lohc_ramp_per_kw * 1000.0 * dt / scale,
                        "MW",
                        "lohc",
                    )
                )

        coefficients = {
            ("grid_buy_da_mw", period): 1.0,
            ("grid_sell_da_mw", period): -1.0,
            ("spill_da_mw", period): -1.0,
            ("agv_charge_count_da", period): (
                -case.agv.charge_power_per_vehicle_kw / 1000.0
            ),
        }
        if case.has_hydrogen_chain:
            coefficients[("lohc_power_da_mw", period)] = -1.0
        rhs = (
            case.base_load_kw[period]
            + nominal["ship.shore_power_kw"][period]
            + nominal["ship.quay_crane_power_kw"][period]
        ) / 1000.0 - direct_wind_mw[period]
        rows.append(
            FirstStageConstraintRow(
                f"electricity.day_ahead_balance[{period}]",
                "eq",
                coefficients,
                float(rhs),
                "MW",
                "electricity",
                "DA-PB",
            )
        )

    if case.has_hydrogen_chain:
        assert case.hydrogen is not None and case.lohc is not None
        for period in range(1, horizon):
            ramp = ("lohc_ramp_abs_da_mw", period)
            current = ("lohc_power_da_mw", period)
            previous = ("lohc_power_da_mw", period - 1)
            rows.extend(
                [
                    FirstStageConstraintRow(
                        f"lohc.day_ahead_ramp_positive[{period}]",
                        "ge",
                        {ramp: 1.0, current: -1.0, previous: 1.0},
                        0.0,
                        "MW",
                        "lohc",
                        "DA-LOHC-RAMP-ABS+",
                    ),
                    FirstStageConstraintRow(
                        f"lohc.day_ahead_ramp_negative[{period}]",
                        "ge",
                        {ramp: 1.0, current: 1.0, previous: -1.0},
                        0.0,
                        "MW",
                        "lohc",
                        "DA-LOHC-RAMP-ABS-",
                    ),
                ]
            )
        landing = case.hydrogen_landing_kg(bundle)
        conversion = (
            1000.0
            * dt
            / case.lohc.energy_kwh_per_kg
            / case.lohc.hydrogen_yield
        )
        for period in range(horizon):
            rows.append(
                FirstStageConstraintRow(
                    f"hydrogen.day_ahead_prefix_availability[{period}]",
                    "ge",
                    {
                        ("lohc_power_da_mw", tau): -conversion
                        for tau in range(period + 1)
                    },
                    -(
                        case.hydrogen.initial_kg
                        + float(np.sum(landing[: period + 1]))
                    ),
                    "kg",
                    "hydrogen",
                    "DA-H2-PREFIX",
                )
            )

    if case.agv.charge_count_ramp is not None:
        for period in range(1, horizon):
            current = ("agv_charge_count_da", period)
            previous = ("agv_charge_count_da", period - 1)
            rows.extend(
                [
                    FirstStageConstraintRow(
                        f"agv.day_ahead_charge_ramp_positive[{period}]",
                        "ge",
                        {current: -1.0, previous: 1.0},
                        -case.agv.charge_count_ramp,
                        "vehicle",
                        "agv",
                        "DA-AGV-RAMP+",
                    ),
                    FirstStageConstraintRow(
                        f"agv.day_ahead_charge_ramp_negative[{period}]",
                        "ge",
                        {current: 1.0, previous: -1.0},
                        -case.agv.charge_count_ramp,
                        "vehicle",
                        "agv",
                        "DA-AGV-RAMP-",
                    ),
                ]
            )
    return FirstStageIR(tuple(variables), tuple(rows))

