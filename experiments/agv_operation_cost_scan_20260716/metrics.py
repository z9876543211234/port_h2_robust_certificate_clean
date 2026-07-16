"""Diagnostics for declared AGV work that exceeds completed physical work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from port_h2_certificate.recourse_ir import VariableKey
from port_h2_certificate.schema import CaseData


@dataclass(frozen=True)
class EmptyRunningMetrics:
    container_required_vehicle_periods: tuple[float, ...]
    lohc_required_vehicle_periods: tuple[float, ...]
    container_empty_vehicle_periods: tuple[float, ...]
    lohc_empty_vehicle_periods: tuple[float, ...]
    total_declared_vehicle_periods: float
    total_productive_vehicle_periods: float
    total_empty_vehicle_periods: float
    empty_fraction_of_declared_work: float
    maximum_capacity_shortfall: float
    empty_battery_energy_kwh: float
    recharge_energy_equivalent_kwh: float


def _value(
    solution: Mapping[VariableKey, float], family: str, period: int
) -> float:
    return float(solution[(family, period)])


def measure_empty_running(
    case: CaseData,
    realization: Mapping[str, Sequence[float]],
    solution: Mapping[VariableKey, float],
) -> EmptyRunningMetrics:
    """Compare declared running AGVs with work implied by physical flows.

    The pre-fix model retains capacity inequalities.  Productive container
    work is reconstructed from the backlog balance, while productive LOHC
    work is reconstructed from the actual outbound flow.  Any declared AGV
    capacity above those two quantities is reported as empty running.
    """
    horizon = case.profile.periods
    dt = case.profile.dt_hours
    container_service = (
        case.agv.container_rate_per_vehicle_hour
        * case.agv.container_turn_efficiency
        * dt
    )
    if container_service <= 0.0:
        raise ValueError("container service per AGV-period must be positive")

    lohc_service = 0.0
    if case.has_hydrogen_chain:
        assert case.lohc is not None
        lohc_service = (
            case.agv.lohc_transport_rate_kg_per_vehicle_hour
            * case.agv.lohc_turn_efficiency
            * dt
        )
        if lohc_service <= 0.0:
            raise ValueError("LOHC service per AGV-period must be positive")

    container_required: list[float] = []
    lohc_required: list[float] = []
    container_empty: list[float] = []
    lohc_empty: list[float] = []
    maximum_shortfall = 0.0
    total_declared = 0.0

    for period in range(horizon):
        backlog_now = (
            _value(solution, "backlog", period)
            * case.logistics.backlog_capacity
        )
        backlog_next = (
            _value(solution, "backlog", period + 1)
            * case.logistics.backlog_capacity
        )
        completed = (
            backlog_now
            + float(realization["ship.task_release"][period])
            - backlog_next
        )
        required_container = max(completed, 0.0) / container_service
        declared_container = _value(
            solution, "agv_container_count", period
        )
        container_required.append(required_container)
        container_empty.append(
            max(declared_container - required_container, 0.0)
        )
        maximum_shortfall = max(
            maximum_shortfall,
            required_container - declared_container,
        )

        required_lohc = 0.0
        declared_lohc = 0.0
        if case.has_hydrogen_chain:
            assert case.lohc is not None
            outbound_kg = (
                _value(solution, "lohc_outbound_normalized", period)
                * case.lohc.capacity_kg
            )
            required_lohc = outbound_kg / lohc_service
            declared_lohc = _value(solution, "agv_lohc_count", period)
            maximum_shortfall = max(
                maximum_shortfall,
                required_lohc - declared_lohc,
            )
        lohc_required.append(required_lohc)
        lohc_empty.append(max(declared_lohc - required_lohc, 0.0))
        total_declared += declared_container + declared_lohc

    total_productive = sum(container_required) + sum(lohc_required)
    total_empty = sum(container_empty) + sum(lohc_empty)
    empty_battery_energy = (
        total_empty * case.agv.run_power_per_vehicle_kw * dt
    )
    return EmptyRunningMetrics(
        container_required_vehicle_periods=tuple(container_required),
        lohc_required_vehicle_periods=tuple(lohc_required),
        container_empty_vehicle_periods=tuple(container_empty),
        lohc_empty_vehicle_periods=tuple(lohc_empty),
        total_declared_vehicle_periods=total_declared,
        total_productive_vehicle_periods=total_productive,
        total_empty_vehicle_periods=total_empty,
        empty_fraction_of_declared_work=(
            total_empty / total_declared if total_declared > 0.0 else 0.0
        ),
        maximum_capacity_shortfall=max(maximum_shortfall, 0.0),
        empty_battery_energy_kwh=empty_battery_energy,
        recharge_energy_equivalent_kwh=(
            empty_battery_energy / case.agv.charge_efficiency
        ),
    )
