"""One-day optimization-horizon contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math


_FORMAL_PROFILES = {
    "hourly_24": (24, 1.0),
    "quarter_hour_96": (96, 0.25),
}
_TOY_PERIODS = frozenset({4, 8, 12})


@dataclass(frozen=True)
class HorizonProfile:
    """Validated one-day time grid.

    Formal loaders accept only the two profiles required by the specification.
    Short grids are explicitly marked as test-only.
    """

    profile_name: str
    periods: int
    dt_hours: float
    is_formal: bool

    @classmethod
    def formal(
        cls, profile_name: str, periods: int, dt_hours: float
    ) -> "HorizonProfile":
        expected = _FORMAL_PROFILES.get(profile_name)
        if expected is None or expected != (periods, dt_hours):
            raise ValueError(
                "formal horizon must be hourly_24=(24, 1.0) or "
                "quarter_hour_96=(96, 0.25)"
            )
        return cls._validated(profile_name, periods, dt_hours, is_formal=True)

    @classmethod
    def toy(
        cls, profile_name: str, periods: int, dt_hours: float
    ) -> "HorizonProfile":
        if periods not in _TOY_PERIODS:
            raise ValueError("toy horizon periods must be one of 4, 8, or 12")
        return cls._validated(profile_name, periods, dt_hours, is_formal=False)

    @classmethod
    def _validated(
        cls, profile_name: str, periods: int, dt_hours: float, *, is_formal: bool
    ) -> "HorizonProfile":
        if not profile_name:
            raise ValueError("profile_name must be non-empty")
        if not isinstance(periods, int) or isinstance(periods, bool) or periods <= 0:
            raise ValueError("periods must be a positive integer")
        if not math.isfinite(dt_hours) or dt_hours <= 0.0:
            raise ValueError("dt_hours must be positive and finite")
        if not math.isclose(periods * dt_hours, 24.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("periods * dt_hours must equal one scheduling day")
        return cls(profile_name, periods, dt_hours, is_formal)

    @property
    def operating_periods(self) -> range:
        return range(self.periods)

    @property
    def state_periods(self) -> range:
        return range(self.periods + 1)

