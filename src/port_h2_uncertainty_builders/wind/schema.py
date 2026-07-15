"""Input contract for the independent wind uncertainty builder."""

from __future__ import annotations

from dataclasses import dataclass
import math

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord


@dataclass(frozen=True)
class WindUncertaintySource:
    profile: HorizonProfile
    nominal_power_kw: tuple[float, ...]
    deviation_down_kw: tuple[float, ...]
    deviation_up_kw: tuple[float, ...]
    budget: int
    provenance: ProvenanceRecord

    def validate(self) -> None:
        horizon = self.profile.periods
        arrays = {
            "nominal_power_kw": self.nominal_power_kw,
            "deviation_down_kw": self.deviation_down_kw,
            "deviation_up_kw": self.deviation_up_kw,
        }
        for name, values in arrays.items():
            if len(values) != horizon:
                raise ValueError(f"{name} length must equal model horizon")
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError(f"{name} must contain finite nonnegative values")
        if any(
            down > nominal
            for nominal, down in zip(
                self.nominal_power_kw, self.deviation_down_kw, strict=True
            )
        ):
            raise ValueError("wind lower realizations must remain nonnegative")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool):
            raise ValueError("wind budget must be an integer")
        if not 0 <= self.budget <= horizon:
            raise ValueError("wind budget must lie between zero and horizon")
        self.provenance.validate()

