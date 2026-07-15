"""Input contract for the three-day integer ship-delay builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord


@dataclass(frozen=True)
class ShipDelaySource:
    profile: HorizonProfile
    previous_day_arrival_count: tuple[int, ...]
    current_day_arrival_count: tuple[int, ...]
    next_day_arrival_count: tuple[int, ...]
    dwell_steps: int
    max_delay_steps: int
    delayed_ship_budget: int
    total_delay_step_budget: int | None
    shore_power_per_ship_kw: float
    quay_cranes_per_ship: float
    installed_quay_cranes: float
    quay_crane_power_kw: float
    quay_crane_task_rate_per_hour: float
    provenance: ProvenanceRecord

    def validate(self) -> None:
        horizon = self.profile.periods
        for name, values in (
            ("previous_day_arrival_count", self.previous_day_arrival_count),
            ("current_day_arrival_count", self.current_day_arrival_count),
            ("next_day_arrival_count", self.next_day_arrival_count),
        ):
            if len(values) != horizon:
                raise ValueError(f"{name} length must equal model horizon")
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in values
            ):
                raise ValueError(f"{name} must contain nonnegative integer counts")

        for name, value in (
            ("dwell_steps", self.dwell_steps),
            ("max_delay_steps", self.max_delay_steps),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= horizon
            ):
                raise ValueError(f"{name} must be an integer in [1, horizon]")

        if (
            not isinstance(self.delayed_ship_budget, int)
            or isinstance(self.delayed_ship_budget, bool)
            or self.delayed_ship_budget < 0
        ):
            raise ValueError("delayed_ship_budget must be a nonnegative integer")
        if self.total_delay_step_budget is not None and (
            not isinstance(self.total_delay_step_budget, int)
            or isinstance(self.total_delay_step_budget, bool)
            or self.total_delay_step_budget < 0
        ):
            raise ValueError("total_delay_step_budget must be null or nonnegative integer")

        physical_values = {
            "shore_power_per_ship_kw": self.shore_power_per_ship_kw,
            "quay_cranes_per_ship": self.quay_cranes_per_ship,
            "installed_quay_cranes": self.installed_quay_cranes,
            "quay_crane_power_kw": self.quay_crane_power_kw,
            "quay_crane_task_rate_per_hour": self.quay_crane_task_rate_per_hour,
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in physical_values.values()
        ):
            raise ValueError("ship physical coefficients must be finite and nonnegative")
        self.provenance.validate()

    def with_total_delay_step_budget(self, value: int | None) -> "ShipDelaySource":
        return replace(self, total_delay_step_budget=value)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "previous_day_arrival_count": self.previous_day_arrival_count,
            "current_day_arrival_count": self.current_day_arrival_count,
            "next_day_arrival_count": self.next_day_arrival_count,
            "dwell_steps": self.dwell_steps,
            "max_delay_steps": self.max_delay_steps,
            "delayed_ship_budget": self.delayed_ship_budget,
            "total_delay_step_budget": self.total_delay_step_budget,
            "shore_power_per_ship_kw": self.shore_power_per_ship_kw,
            "quay_cranes_per_ship": self.quay_cranes_per_ship,
            "installed_quay_cranes": self.installed_quay_cranes,
            "quay_crane_power_kw": self.quay_crane_power_kw,
            "quay_crane_task_rate_per_hour": self.quay_crane_task_rate_per_hour,
            "provenance": self.provenance,
        }

