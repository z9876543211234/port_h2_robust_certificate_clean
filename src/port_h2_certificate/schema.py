"""Validated physical-case schema, independent of uncertainty construction rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import gurobipy as gp
import numpy as np

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


CaseName = Literal[
    "C1_ProposedRobustMain",
    "C2_NoHydrogenRobust",
    "C3_WorkCapacityCapsRobust",
    "C4_DeterministicMain",
]
OutboundMode = Literal["disabled", "fixed_pipeline", "agv_virtual_pipeline"]


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _sequence(
    name: str, values: tuple[float, ...], length: int, *, nonnegative: bool = True
) -> None:
    if len(values) != length:
        raise ValueError(f"{name} length must equal horizon")
    if not all(
        math.isfinite(value) and (not nonnegative or value >= 0.0)
        for value in values
    ):
        raise ValueError(f"{name} contains invalid values")


@dataclass(frozen=True)
class GridParameters:
    buy_capacity_kw: float
    sell_capacity_kw: float
    buy_price_per_kwh: tuple[float, ...]
    sell_price_per_kwh: tuple[float, ...]


@dataclass(frozen=True)
class HydrogenParameters:
    transport_efficiency: float
    conversion_kg_per_kwh: float
    landing_delay_steps: int
    historical_landing_kg: tuple[float, ...]
    initial_kg: float
    capacity_kg: float
    terminal_min_kg: float


@dataclass(frozen=True)
class LohcParameters:
    initial_kg: float
    capacity_kg: float
    terminal_min_kg: float
    reactor_min_kw: float
    reactor_max_kw: float
    day_ahead_ramp_kw: float
    real_time_ramp_kw: float
    adjustment_limit_kw: float
    energy_kwh_per_kg: float
    hydrogen_yield: float
    fixed_pipeline_rate_kg_per_hour: float


@dataclass(frozen=True)
class AgvParameters:
    fleet_size: float
    charger_count: float
    charge_power_per_vehicle_kw: float
    run_power_per_vehicle_kw: float
    battery_capacity_per_vehicle_kwh: float
    charge_efficiency: float
    initial_soc: float
    min_soc: float
    max_soc: float
    terminal_min_soc: float
    lohc_transport_rate_kg_per_vehicle_hour: float
    container_rate_per_vehicle_hour: float
    lohc_turn_efficiency: float
    container_turn_efficiency: float
    charge_adjustment_limit_kw: float
    charge_count_ramp: float | None


@dataclass(frozen=True)
class LogisticsParameters:
    initial_backlog: float
    backlog_capacity: float


@dataclass(frozen=True)
class CostParameters:
    grid_deviation_per_kwh: float
    spill_day_ahead_per_kwh: float
    spill_real_time_per_kwh: float
    spill_deviation_per_kwh: float
    lohc_ramp_per_kw: float
    lohc_adjustment_per_kwh: float
    lohc_operation_per_kg: float
    charge_adjustment_per_kwh: float
    backlog_delay_per_task_hour: float
    terminal_backlog_per_task: float
    lohc_export_revenue_per_kg: float
    agv_operation_per_vehicle_hour: float = 0.0


def _minimum_output(bundle: UncertaintyBundle, output_key: str) -> np.ndarray:
    matrix = bundle.output_matrix(output_key).to_scipy()
    relevant_columns = set(matrix.nonzero()[1].tolist())
    relevant_keys = {bundle.selector_keys[index] for index in relevant_columns}
    changed = True
    while changed:
        changed = False
        for row in bundle.constraints:
            if relevant_keys & set(row.coefficients):
                before = len(relevant_keys)
                relevant_keys.update(row.coefficients)
                changed |= len(relevant_keys) != before

    model = gp.Model(f"minimum_{output_key}")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    variables = {
        key: model.addVar(vtype=gp.GRB.BINARY, name=key) for key in relevant_keys
    }
    for row in bundle.constraints:
        if not set(row.coefficients) <= relevant_keys:
            continue
        expression = gp.quicksum(
            coefficient * variables[key]
            for key, coefficient in row.coefficients.items()
        )
        if row.sense == "le":
            model.addConstr(expression <= row.rhs)
        elif row.sense == "ge":
            model.addConstr(expression >= row.rhs)
        else:
            model.addConstr(expression == row.rhs)

    output = bundle.outputs[output_key]
    minimum = np.zeros(len(output.nominal), dtype=float)
    selector_index = {key: index for index, key in enumerate(bundle.selector_keys)}
    for period, nominal in enumerate(output.nominal):
        row = matrix.getrow(period)
        expression = nominal + gp.quicksum(
            value * variables[bundle.selector_keys[column]]
            for column, value in zip(row.indices, row.data, strict=True)
        )
        model.setObjective(expression, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("Bundle output minimum did not reach GRB.OPTIMAL")
        minimum[period] = model.ObjVal
    return minimum


@dataclass(frozen=True)
class CaseData:
    case_name: CaseName
    profile: HorizonProfile
    grid: GridParameters
    base_load_kw: tuple[float, ...]
    wind_to_hydrogen_ratio: tuple[float, ...]
    hydrogen: HydrogenParameters | None
    lohc: LohcParameters | None
    agv: AgvParameters
    logistics: LogisticsParameters
    cost: CostParameters
    outbound_mode: OutboundMode
    container_work_capacity: float | None
    lohc_work_capacity: float | None
    cost_scale: float

    @property
    def has_hydrogen_chain(self) -> bool:
        return self.hydrogen is not None and self.lohc is not None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def wind_nominal_kw(self, bundle: UncertaintyBundle) -> np.ndarray:
        return np.asarray(
            bundle.outputs["wind.available_power_kw"].nominal, dtype=float
        )

    def hydrogen_power_day_ahead_kw(self, bundle: UncertaintyBundle) -> np.ndarray:
        return self.wind_nominal_kw(bundle) * np.asarray(
            self.wind_to_hydrogen_ratio, dtype=float
        )

    def hydrogen_landing_kg(self, bundle: UncertaintyBundle) -> np.ndarray:
        if not self.has_hydrogen_chain:
            return np.zeros(self.profile.periods, dtype=float)
        assert self.hydrogen is not None
        hydrogen_power = self.hydrogen_power_day_ahead_kw(bundle)
        landing = np.zeros(self.profile.periods, dtype=float)
        delay = self.hydrogen.landing_delay_steps
        for period in self.profile.operating_periods:
            if period < delay:
                landing[period] = self.hydrogen.historical_landing_kg[period]
            else:
                landing[period] = (
                    self.hydrogen.transport_efficiency
                    * self.hydrogen.conversion_kg_per_kwh
                    * hydrogen_power[period - delay]
                    * self.profile.dt_hours
                )
        return landing

    def validate(self, bundle: UncertaintyBundle) -> None:
        bundle.validate()
        if self.case_name not in {
            "C1_ProposedRobustMain",
            "C2_NoHydrogenRobust",
            "C3_WorkCapacityCapsRobust",
            "C4_DeterministicMain",
        }:
            raise ValueError("unsupported case_name")
        if (
            self.profile.profile_name != bundle.profile_name
            or self.profile.periods != bundle.model_horizon
        ):
            raise ValueError("case and Bundle profile/horizon must match")
        required_outputs = {
            "wind.available_power_kw",
            "ship.shore_power_kw",
            "ship.quay_crane_power_kw",
            "ship.task_release",
        }
        missing = required_outputs - set(bundle.outputs)
        if missing:
            raise ValueError(f"Bundle is missing physical outputs: {sorted(missing)}")

        horizon = self.profile.periods
        _sequence("base_load_kw", self.base_load_kw, horizon)
        _sequence(
            "wind_to_hydrogen_ratio", self.wind_to_hydrogen_ratio, horizon
        )
        if any(value > 1.0 for value in self.wind_to_hydrogen_ratio):
            raise ValueError("wind_to_hydrogen_ratio must be within [0, 1]")
        _finite_nonnegative("grid.buy_capacity_kw", self.grid.buy_capacity_kw)
        _finite_nonnegative("grid.sell_capacity_kw", self.grid.sell_capacity_kw)
        _sequence("grid.buy_price_per_kwh", self.grid.buy_price_per_kwh, horizon)
        _sequence("grid.sell_price_per_kwh", self.grid.sell_price_per_kwh, horizon)
        if self.cost_scale <= 0.0 or not math.isfinite(self.cost_scale):
            raise ValueError("cost_scale must be positive and finite")

        if self.case_name == "C2_NoHydrogenRobust":
            if self.hydrogen is not None or self.lohc is not None:
                raise ValueError("C2 must structurally omit hydrogen and LOHC")
            if any(self.wind_to_hydrogen_ratio):
                raise ValueError("C2 wind-to-hydrogen ratio must be zero")
            if self.outbound_mode != "disabled":
                raise ValueError("C2 outbound mode must be disabled")
        elif not self.has_hydrogen_chain:
            raise ValueError("C1/C3/C4 require hydrogen and LOHC parameters")

        if self.case_name == "C3_WorkCapacityCapsRobust":
            if self.container_work_capacity is None or self.lohc_work_capacity is None:
                raise ValueError("C3 requires both work-capacity caps")
        elif self.container_work_capacity is not None or self.lohc_work_capacity is not None:
            raise ValueError("work-capacity caps are only valid for C3")

        if self.has_hydrogen_chain:
            assert self.hydrogen is not None and self.lohc is not None
            hydrogen = self.hydrogen
            lohc = self.lohc
            if not 0.0 < hydrogen.transport_efficiency <= 1.0:
                raise ValueError("invalid hydrogen transport efficiency")
            _finite_nonnegative(
                "hydrogen.conversion_kg_per_kwh", hydrogen.conversion_kg_per_kwh
            )
            if (
                not isinstance(hydrogen.landing_delay_steps, int)
                or not 0 <= hydrogen.landing_delay_steps <= horizon
                or len(hydrogen.historical_landing_kg)
                != hydrogen.landing_delay_steps
            ):
                raise ValueError("hydrogen historical landing must match integer delay")
            for name, value in (
                ("hydrogen.initial_kg", hydrogen.initial_kg),
                ("hydrogen.capacity_kg", hydrogen.capacity_kg),
                ("hydrogen.terminal_min_kg", hydrogen.terminal_min_kg),
                ("lohc.initial_kg", lohc.initial_kg),
                ("lohc.capacity_kg", lohc.capacity_kg),
                ("lohc.terminal_min_kg", lohc.terminal_min_kg),
                ("lohc.reactor_min_kw", lohc.reactor_min_kw),
                ("lohc.reactor_max_kw", lohc.reactor_max_kw),
                ("lohc.day_ahead_ramp_kw", lohc.day_ahead_ramp_kw),
                ("lohc.real_time_ramp_kw", lohc.real_time_ramp_kw),
                ("lohc.adjustment_limit_kw", lohc.adjustment_limit_kw),
                ("lohc.energy_kwh_per_kg", lohc.energy_kwh_per_kg),
                ("lohc.fixed_pipeline_rate_kg_per_hour", lohc.fixed_pipeline_rate_kg_per_hour),
            ):
                _finite_nonnegative(name, value)
            if hydrogen.capacity_kg <= 0.0 or lohc.capacity_kg <= 0.0:
                raise ValueError("hydrogen and LOHC capacities must be positive")
            if not 0.0 < lohc.hydrogen_yield <= 1.0:
                raise ValueError("invalid LOHC hydrogen yield")
            if not lohc.reactor_min_kw <= lohc.reactor_max_kw:
                raise ValueError("LOHC reactor bounds are inconsistent")
            minimum_wind = _minimum_output(bundle, "wind.available_power_kw")
            if np.any(self.hydrogen_power_day_ahead_kw(bundle) > minimum_wind + 1e-9):
                raise ValueError(
                    "fixed wind-to-hydrogen power exceeds a full-set wind minimum"
                )

        agv = self.agv
        for name, value in asdict(agv).items():
            if value is not None:
                _finite_nonnegative(f"agv.{name}", value)
        if agv.fleet_size <= 0.0 or agv.battery_capacity_per_vehicle_kwh <= 0.0:
            raise ValueError("AGV fleet and battery capacity must be positive")
        if not (0.0 <= agv.min_soc <= agv.initial_soc <= agv.max_soc <= 1.0):
            raise ValueError("AGV SOC bounds are inconsistent")
        if not agv.min_soc <= agv.terminal_min_soc <= agv.max_soc:
            raise ValueError("AGV terminal SOC is outside bounds")
        if agv.charger_count > agv.fleet_size:
            raise ValueError("charger count cannot exceed AGV fleet")
        if self.container_work_capacity is not None and (
            self.container_work_capacity > agv.fleet_size
            or self.lohc_work_capacity > agv.fleet_size
        ):
            raise ValueError("C3 work-capacity cap exceeds AGV fleet")

        _finite_nonnegative(
            "logistics.initial_backlog", self.logistics.initial_backlog
        )
        if self.logistics.backlog_capacity <= 0.0:
            raise ValueError("backlog_capacity must be positive")
        if self.logistics.initial_backlog > self.logistics.backlog_capacity:
            raise ValueError("initial backlog exceeds capacity")
        for name, value in asdict(self.cost).items():
            _finite_nonnegative(f"cost.{name}", value)
