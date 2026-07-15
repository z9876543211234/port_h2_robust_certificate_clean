from __future__ import annotations

import pytest

from port_h2_contracts.horizon import HorizonProfile
from port_h2_uncertainty_builders.ship_delay.nhpp_nominal import (
    NhppNominalSettings,
    aggregate_three_day_sample,
    sample_three_day_arrivals,
)


def _settings(*, random_seed: int = 20260713) -> NhppNominalSettings:
    return NhppNominalSettings(
        profile=HorizonProfile.formal("quarter_hour_96", 96, 0.25),
        target_daily_expected_arrivals=11.52,
        random_seed=random_seed,
    )


def test_nhpp_sampling_freezes_three_reproducible_integer_days() -> None:
    first = sample_three_day_arrivals(_settings())
    replay = sample_three_day_arrivals(_settings())

    assert first == replay
    assert first.day_roles == ("previous_day", "current_day", "next_day")
    assert len(set(first.arrival_count_by_day)) == 3
    for day in first.arrival_count_by_day:
        assert len(day) == 96
        assert all(type(value) is int and value >= 0 for value in day)

    assert sum(first.expected_count_per_period) == pytest.approx(11.52, abs=1.0e-12)


def test_nhpp_seed_changes_integer_sample_without_changing_intensity() -> None:
    first = sample_three_day_arrivals(_settings(random_seed=20260713))
    second = sample_three_day_arrivals(_settings(random_seed=20260714))

    assert first.expected_count_per_period == second.expected_count_per_period
    assert first.arrival_count_by_day != second.arrival_count_by_day


def test_hourly_profile_is_exact_integer_aggregation_of_same_three_day_draw() -> None:
    quarter_hour = sample_three_day_arrivals(_settings())
    hourly = aggregate_three_day_sample(
        quarter_hour,
        HorizonProfile.formal("hourly_24", 24, 1.0),
    )

    assert hourly.random_seed == quarter_hour.random_seed
    for source_day, hourly_day in zip(
        quarter_hour.arrival_count_by_day,
        hourly.arrival_count_by_day,
        strict=True,
    ):
        expected = tuple(
            sum(source_day[start : start + 4]) for start in range(0, 96, 4)
        )
        assert hourly_day == expected
        assert all(type(value) is int for value in hourly_day)


def test_model_nominal_counts_are_not_continuous_expected_arrival_pressure() -> None:
    sample = sample_three_day_arrivals(_settings())

    assert any(
        not float(expected).is_integer()
        for expected in sample.expected_count_per_period
    )
    assert all(
        type(count) is int
        for day in sample.arrival_count_by_day
        for count in day
    )
    assert sample.equivalent_arrival_pressure_used is False
