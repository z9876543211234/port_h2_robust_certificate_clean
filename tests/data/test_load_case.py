from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = PROJECT_ROOT / "data" / "cases"


def _resolutions():
    return {
        "deterministic.cost.spill_day_ahead_per_kwh": 0.1,
        "deterministic.cost.spill_real_time_per_kwh": 0.1,
        "deterministic.cost.spill_deviation_per_kwh": 0.05,
        "ship_delay.max_delay_steps": 4,
        "ship_delay.delayed_ship_budget": 2,
    }


def test_formal_loader_rejects_user_requested_placeholders() -> None:
    from port_h2_certificate.load_case import UnresolvedInputError, load_case

    with pytest.raises(UnresolvedInputError) as captured:
        load_case(CASE_ROOT / "C1.yaml", "quarter_hour_96")
    assert set(captured.value.paths) == {
        "deterministic.cost.spill_day_ahead_per_kwh",
        "deterministic.cost.spill_real_time_per_kwh",
        "deterministic.cost.spill_deviation_per_kwh",
        "ship_delay.max_delay_steps",
        "ship_delay.delayed_ship_budget",
    }


def test_resolved_96_input_package_uses_frozen_three_day_nhpp_integer_counts() -> None:
    from port_h2_certificate.load_case import load_case

    loaded = load_case(
        CASE_ROOT / "C1.yaml", "quarter_hour_96", resolutions=_resolutions()
    )
    assert loaded.case.profile.periods == 96
    assert loaded.case.hydrogen is not None
    assert loaded.case.hydrogen.landing_delay_steps == 1
    previous = loaded.ship_delay_payload["previous_day_arrival_count"]
    current = loaded.ship_delay_payload["current_day_arrival_count"]
    following = loaded.ship_delay_payload["next_day_arrival_count"]
    assert len({tuple(previous), tuple(current), tuple(following)}) == 3
    assert all(
        type(value) is int and value >= 0
        for day in (previous, current, following)
        for value in day
    )
    assert tuple(sum(day) for day in (previous, current, following)) == (15, 15, 9)
    assert loaded.ship_delay_payload["nominal_generation"]["process"] == (
        "nonhomogeneous_poisson_independent_increments"
    )
    assert loaded.ship_delay_payload["nominal_generation"][
        "equivalent_arrival_pressure_used"
    ] is False
    assert loaded.ship_delay_payload["provenance"]["source_type"] == "generated"
    assert loaded.ship_delay_payload["provenance"]["random_seed"] == 20260713
    assert loaded.ship_delay_payload["total_delay_step_budget"] is None


def test_hourly_hydrogen_delay_remains_an_explicit_blocker() -> None:
    from port_h2_certificate.load_case import UnresolvedInputError, load_case

    with pytest.raises(UnresolvedInputError) as captured:
        load_case(CASE_ROOT / "C1.yaml", "hourly_24", resolutions=_resolutions())
    assert captured.value.paths == (
        "deterministic.hydrogen.landing_delay_steps",
        "deterministic.hydrogen.historical_landing_kg",
    )


def test_case_semantics_are_applied_structurally() -> None:
    from port_h2_certificate.load_case import load_case

    c2 = load_case(
        CASE_ROOT / "C2.yaml", "quarter_hour_96", resolutions=_resolutions()
    ).case
    assert c2.hydrogen is None and c2.lohc is None
    assert set(c2.wind_to_hydrogen_ratio) == {0.0}
    assert c2.outbound_mode == "disabled"
    assert c2.cost.lohc_export_revenue_per_kg == 0.0

    c3 = load_case(
        CASE_ROOT / "C3.yaml", "quarter_hour_96", resolutions=_resolutions()
    ).case
    assert c3.container_work_capacity == 7.0
    assert c3.lohc_work_capacity == 6.0
