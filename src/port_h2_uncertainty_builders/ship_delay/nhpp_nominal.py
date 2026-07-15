"""Freeze three days of discrete ship arrivals from an NHPP intensity.

The NHPP is an offline data generator only.  Its independent Poisson increments
produce integer vessel counts for the previous, current, and next day.  The
robust model consumes those frozen counts; it never consumes the fractional
expected-arrival profile as an equivalent logistics pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from port_h2_contracts.horizon import HorizonProfile


DAY_ROLES = ("previous_day", "current_day", "next_day")


@dataclass(frozen=True)
class NhppNominalSettings:
    """Configuration for one reproducible three-day NHPP draw."""

    profile: HorizonProfile
    target_daily_expected_arrivals: float
    random_seed: int

    def validate(self) -> None:
        if not math.isclose(
            self.profile.periods * self.profile.dt_hours,
            24.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("NHPP nominal generation requires a 24-hour profile")
        if (
            not math.isfinite(self.target_daily_expected_arrivals)
            or self.target_daily_expected_arrivals <= 0.0
        ):
            raise ValueError("target_daily_expected_arrivals must be positive")
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer")


@dataclass(frozen=True)
class ThreeDayNhppSample:
    """Frozen NHPP draw used as the integer nominal ship timetable."""

    profile: HorizonProfile
    day_roles: tuple[str, str, str]
    lambda_ship_per_hour: tuple[float, ...]
    expected_count_per_period: tuple[float, ...]
    arrival_count_by_day: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    target_daily_expected_arrivals: float
    random_seed: int
    equivalent_arrival_pressure_used: bool = False

    @property
    def previous_day_arrival_count(self) -> tuple[int, ...]:
        return self.arrival_count_by_day[0]

    @property
    def current_day_arrival_count(self) -> tuple[int, ...]:
        return self.arrival_count_by_day[1]

    @property
    def next_day_arrival_count(self) -> tuple[int, ...]:
        return self.arrival_count_by_day[2]

    @property
    def realized_daily_totals(self) -> tuple[int, int, int]:
        return tuple(sum(day) for day in self.arrival_count_by_day)  # type: ignore[return-value]

    def validate(self) -> None:
        horizon = self.profile.periods
        if self.day_roles != DAY_ROLES:
            raise ValueError("three-day roles must be previous/current/next")
        if len(self.lambda_ship_per_hour) != horizon:
            raise ValueError("lambda profile length must equal the model horizon")
        if len(self.expected_count_per_period) != horizon:
            raise ValueError("expected-count profile length must equal the model horizon")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.lambda_ship_per_hour + self.expected_count_per_period
        ):
            raise ValueError("NHPP intensity and expected counts must be nonnegative")
        if not math.isclose(
            sum(self.expected_count_per_period),
            self.target_daily_expected_arrivals,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("expected NHPP counts do not match the daily target")
        if len(self.arrival_count_by_day) != 3:
            raise ValueError("exactly three daily arrival sequences are required")
        for day in self.arrival_count_by_day:
            if len(day) != horizon:
                raise ValueError("daily arrival sequence length must equal the horizon")
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in day
            ):
                raise ValueError("NHPP realization must contain integer ship counts")
        if self.equivalent_arrival_pressure_used:
            raise ValueError("equivalent arrival pressure is prohibited")


def _gaussian(clock_hour: np.ndarray, *, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((clock_hour - center) / sigma) ** 2)


def _raw_periodic_intensity_shape(clock_hour: np.ndarray) -> np.ndarray:
    """Legacy-study intensity shape, retained only as an offline NHPP rate."""

    shape = (
        0.58
        + 0.78 * _gaussian(clock_hour, center=8.5, sigma=2.15)
        + 0.62 * _gaussian(clock_hour, center=15.2, sigma=2.65)
        + 0.24 * _gaussian(clock_hour, center=20.4, sigma=1.85)
        - 0.18 * _gaussian(clock_hour, center=3.0, sigma=2.2)
    )
    return np.maximum(shape, 0.25)


def expected_nhpp_counts(
    settings: NhppNominalSettings,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return piecewise-constant rate and exact interval means."""

    settings.validate()
    clock_hour = (
        np.arange(settings.profile.periods, dtype=float)
        * settings.profile.dt_hours
    )
    raw_shape = _raw_periodic_intensity_shape(clock_hour)
    scale = settings.target_daily_expected_arrivals / (
        float(raw_shape.sum()) * settings.profile.dt_hours
    )
    rate = raw_shape * scale
    expected = rate * settings.profile.dt_hours
    if not np.isclose(
        expected.sum(),
        settings.target_daily_expected_arrivals,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("NHPP expected-count normalization failed")
    return tuple(float(value) for value in rate), tuple(
        float(value) for value in expected
    )


def sample_three_day_arrivals(
    settings: NhppNominalSettings,
) -> ThreeDayNhppSample:
    """Draw and freeze three independent days of Poisson integer counts."""

    rate, expected = expected_nhpp_counts(settings)
    generator = np.random.default_rng(settings.random_seed)
    draws = generator.poisson(
        lam=np.asarray(expected, dtype=float),
        size=(len(DAY_ROLES), settings.profile.periods),
    )
    integer_days = tuple(
        tuple(int(value) for value in row) for row in draws
    )
    sample = ThreeDayNhppSample(
        profile=settings.profile,
        day_roles=DAY_ROLES,
        lambda_ship_per_hour=rate,
        expected_count_per_period=expected,
        arrival_count_by_day=integer_days,  # type: ignore[arg-type]
        target_daily_expected_arrivals=settings.target_daily_expected_arrivals,
        random_seed=settings.random_seed,
        equivalent_arrival_pressure_used=False,
    )
    sample.validate()
    return sample


def aggregate_three_day_sample(
    source: ThreeDayNhppSample,
    target_profile: HorizonProfile,
) -> ThreeDayNhppSample:
    """Aggregate one frozen fine-grid draw without resampling or rounding."""

    source.validate()
    if not math.isclose(
        source.profile.periods * source.profile.dt_hours,
        target_profile.periods * target_profile.dt_hours,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("source and target profiles must span the same duration")
    ratio = target_profile.dt_hours / source.profile.dt_hours
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("target step must be an integer aggregation of source steps")
    if source.profile.periods != target_profile.periods * factor:
        raise ValueError("profile periods do not match the aggregation factor")

    expected = tuple(
        sum(source.expected_count_per_period[start : start + factor])
        for start in range(0, source.profile.periods, factor)
    )
    days = tuple(
        tuple(
            sum(day[start : start + factor])
            for start in range(0, source.profile.periods, factor)
        )
        for day in source.arrival_count_by_day
    )
    sample = ThreeDayNhppSample(
        profile=target_profile,
        day_roles=source.day_roles,
        lambda_ship_per_hour=tuple(
            value / target_profile.dt_hours for value in expected
        ),
        expected_count_per_period=expected,
        arrival_count_by_day=days,  # type: ignore[arg-type]
        target_daily_expected_arrivals=source.target_daily_expected_arrivals,
        random_seed=source.random_seed,
        equivalent_arrival_pressure_used=False,
    )
    sample.validate()
    return sample
