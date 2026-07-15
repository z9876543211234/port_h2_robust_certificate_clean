from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIP_ROOT = PROJECT_ROOT / "data" / "uncertainty_sources" / "ship_delay"


def _read_json(name: str) -> dict[str, object]:
    return json.loads((SHIP_ROOT / name).read_text(encoding="utf-8"))


def _aggregate(values: list[int], factor: int = 4) -> list[int]:
    return [sum(values[start : start + factor]) for start in range(0, len(values), factor)]


def test_frozen_nhpp_source_is_discrete_reproducible_and_not_equivalent_pressure() -> None:
    payload = _read_json("quarter_hour_96.json")
    metadata = payload["nominal_generation"]
    assert metadata == {
        **metadata,
        "process": "nonhomogeneous_poisson_independent_increments",
        "model_nominal": "frozen_integer_ship_count_per_period",
        "equivalent_arrival_pressure_used": False,
        "resampling_during_optimization": False,
        "random_seed": 20260713,
        "target_daily_expected_arrivals": 11.52,
    }

    days = [
        payload["previous_day_arrival_count"],
        payload["current_day_arrival_count"],
        payload["next_day_arrival_count"],
    ]
    assert len({tuple(day) for day in days}) == 3
    assert tuple(sum(day) for day in days) == (15, 15, 9)
    assert all(type(value) is int and value >= 0 for day in days for value in day)

    provenance = payload["provenance"]
    assert provenance["source_type"] == "generated"
    assert provenance["random_seed"] == 20260713
    raw_path = PROJECT_ROOT / provenance["source_reference"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == provenance["raw_file_sha256"]
    generator_path = PROJECT_ROOT / metadata["generator_reference"]
    assert hashlib.sha256(generator_path.read_bytes()).hexdigest() == provenance[
        "generation_script_sha256"
    ]

    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    for row in rows:
        for field in (
            "previous_day_arrival_count",
            "current_day_arrival_count",
            "next_day_arrival_count",
        ):
            assert str(int(row[field])) == row[field]


def test_hourly_source_is_exact_aggregation_of_same_frozen_draw() -> None:
    fine = _read_json("quarter_hour_96.json")
    hourly = _read_json("hourly_24.json")
    for field in (
        "previous_day_arrival_count",
        "current_day_arrival_count",
        "next_day_arrival_count",
    ):
        assert hourly[field] == _aggregate(fine[field])
    assert hourly["provenance"]["random_seed"] == fine["provenance"]["random_seed"]
    assert hourly["nominal_generation"]["aggregation_from"] == "quarter_hour_96"


def test_frozen_current_day_builds_exact_integer_delay_uncertainty_set(tmp_path) -> None:
    payload = _read_json("quarter_hour_96.json")
    source = ShipDelaySource(
        profile=HorizonProfile.formal("quarter_hour_96", 96, 0.25),
        previous_day_arrival_count=tuple(payload["previous_day_arrival_count"]),
        current_day_arrival_count=tuple(payload["current_day_arrival_count"]),
        next_day_arrival_count=tuple(payload["next_day_arrival_count"]),
        dwell_steps=int(payload["dwell_steps"]),
        max_delay_steps=4,
        delayed_ship_budget=2,
        total_delay_step_budget=None,
        shore_power_per_ship_kw=float(payload["shore_power_per_ship_kw"]),
        quay_cranes_per_ship=float(payload["quay_cranes_per_ship"]),
        installed_quay_cranes=float(payload["installed_quay_cranes"]),
        quay_crane_power_kw=float(payload["quay_crane_power_kw"]),
        quay_crane_task_rate_per_hour=float(payload["quay_crane_task_rate_per_hour"]),
        provenance=ProvenanceRecord.from_dict(payload["provenance"]),
    )
    bundle = build_ship_delay_bundle(source, tmp_path)
    nominal = bundle.evaluate(bundle.nominal_selector)

    np.testing.assert_array_equal(
        nominal["ship.arrival_count"],
        np.asarray(payload["current_day_arrival_count"], dtype=int),
    )
    assert len(bundle.primary_selector_keys) == 15 * 4
    assert all(key.startswith("ship.delay[") for key in bundle.primary_selector_keys)
    assert bundle.proofs["early_arrival_selectors"] == 0
    assert bundle.proofs["modulo_wraparound_used"] is False
    assert bundle.proofs["three_day_mass_conservation"] is True
