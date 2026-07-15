from __future__ import annotations

import importlib
import importlib.util
from itertools import product
import re

import numpy as np
import pytest

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord


_DELAY_KEY = re.compile(r"ship\.delay\[(\d+),(\d+),(\d+)\]")


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _source(*, installed_quay_cranes: float = 3.0, current=(2, 0, 0, 1)):
    schema = _module("port_h2_uncertainty_builders.ship_delay.schema")
    return schema.ShipDelaySource(
        profile=HorizonProfile.toy("toy_4", 4, 6.0),
        previous_day_arrival_count=(0, 0, 0, 1),
        current_day_arrival_count=tuple(current),
        next_day_arrival_count=(0, 0, 0, 0),
        dwell_steps=2,
        max_delay_steps=2,
        delayed_ship_budget=2,
        total_delay_step_budget=3,
        shore_power_per_ship_kw=100.0,
        quay_cranes_per_ship=2.0,
        installed_quay_cranes=installed_quay_cranes,
        quay_crane_power_kw=10.0,
        quay_crane_task_rate_per_hour=4.0,
        provenance=ProvenanceRecord(
            source_type="fixed_file",
            source_reference="toy_ship.yaml",
            random_seed=None,
            generation_script_sha256=None,
            raw_file_sha256="c" * 64,
            canonical_payload_sha256="d" * 64,
        ),
    )


def _direct_simulation(source, primary_assignment):
    horizon = source.profile.periods
    extended = list(
        source.previous_day_arrival_count
        + source.current_day_arrival_count
        + source.next_day_arrival_count
    )
    delayed_out = 0
    for key, value in primary_assignment.items():
        if value == 0:
            continue
        match = _DELAY_KEY.fullmatch(key)
        assert match is not None
        source_period, _, delay = (int(part) for part in match.groups())
        origin = horizon + source_period
        target = origin + delay
        extended[origin] -= 1
        extended[target] += 1
        delayed_out += int(target >= 2 * horizon)
    assert min(extended) >= 0
    assert sum(extended) == sum(
        source.previous_day_arrival_count
        + source.current_day_arrival_count
        + source.next_day_arrival_count
    )
    arrivals = np.asarray(extended[horizon : 2 * horizon], dtype=float)
    in_port = np.asarray(
        [
            sum(extended[horizon + period - lag] for lag in range(source.dwell_steps))
            for period in range(horizon)
        ],
        dtype=float,
    )
    active = np.minimum(
        source.installed_quay_cranes, source.quay_cranes_per_ship * in_port
    )
    return {
        "ship.arrival_count": arrivals,
        "ship.in_port_count": in_port,
        "ship.shore_power_kw": source.shore_power_per_ship_kw * in_port,
        "ship.active_quay_cranes": active,
        "ship.quay_crane_power_kw": source.quay_crane_power_kw * active,
        "ship.task_release": (
            source.quay_crane_task_rate_per_hour * active * source.profile.dt_hours
        ),
        "ship.delayed_out_of_current_day_count": np.asarray([delayed_out], dtype=float),
    }


def _feasible_primary_assignments(bundle):
    for values in product((0, 1), repeat=len(bundle.primary_selector_keys)):
        primary = dict(zip(bundle.primary_selector_keys, values, strict=True))
        try:
            full = bundle.complete(primary)
        except ValueError:
            continue
        yield primary, full


def test_ship_source_requires_nonnegative_integer_arrival_counts() -> None:
    schema = _module("port_h2_uncertainty_builders.ship_delay.schema")
    source = _source()
    broken = schema.ShipDelaySource(
        **{
            **source.to_dict(),
            "current_day_arrival_count": (1.5, 0, 0, 1),
            "profile": source.profile,
            "provenance": source.provenance,
        }
    )
    with pytest.raises(ValueError, match="integer"):
        broken.validate()


def test_ship_builder_uses_only_positive_current_day_delays(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    bundle = builder.build_ship_delay_bundle(_source(), tmp_path)
    assert (tmp_path / "ship_delay_uncertainty_bundle.json").is_file()
    assert all(_DELAY_KEY.fullmatch(key) for key in bundle.primary_selector_keys)
    parsed = [tuple(int(part) for part in _DELAY_KEY.fullmatch(key).groups()) for key in bundle.primary_selector_keys]
    assert {source_period for source_period, _, _ in parsed} == {0, 3}
    assert {delay for _, _, delay in parsed} == {1, 2}


def test_ship_bundle_matches_direct_unit_ship_simulation_for_every_scenario(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    source = _source()
    bundle = builder.build_ship_delay_bundle(source, tmp_path)
    scenario_count = 0
    for primary, full in _feasible_primary_assignments(bundle):
        expected = _direct_simulation(source, primary)
        actual = bundle.evaluate(full)
        for key, values in expected.items():
            np.testing.assert_allclose(actual[key], values, atol=1e-12, rtol=0.0)
        scenario_count += 1
    assert scenario_count > 1
    assert bundle.proofs["three_day_mass_conservation"] is True
    assert bundle.proofs["modulo_wraparound_used"] is False


def test_previous_day_arrivals_contribute_to_current_in_port_count(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    source = _source(current=(0, 0, 0, 0))
    bundle = builder.build_ship_delay_bundle(source, tmp_path)
    realization = bundle.evaluate(bundle.nominal_selector)
    assert realization["ship.arrival_count"][0] == 0.0
    assert realization["ship.in_port_count"][0] == 1.0


def test_ship_count_and_total_delay_budgets_are_both_enforced(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    bundle = builder.build_ship_delay_bundle(_source(), tmp_path)

    too_many = {key: 0 for key in bundle.primary_selector_keys}
    too_many["ship.delay[0,0,1]"] = 1
    too_many["ship.delay[0,1,1]"] = 1
    too_many["ship.delay[3,0,1]"] = 1
    with pytest.raises(ValueError):
        bundle.complete(too_many)

    too_long = {key: 0 for key in bundle.primary_selector_keys}
    too_long["ship.delay[0,0,2]"] = 1
    too_long["ship.delay[3,0,2]"] = 1
    with pytest.raises(ValueError):
        bundle.complete(too_long)


def test_anonymous_symmetry_preserves_the_arrival_realization_set(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    source = _source(installed_quay_cranes=100.0, current=(2, 0, 0, 0))
    source = source.with_total_delay_step_budget(None)
    bundle = builder.build_ship_delay_bundle(source, tmp_path)

    raw_realizations: set[tuple[float, ...]] = set()
    for first_delay, second_delay in product((0, 1, 2), repeat=2):
        assignment = {key: 0 for key in bundle.primary_selector_keys}
        if first_delay:
            assignment[f"ship.delay[0,0,{first_delay}]"] = 1
        if second_delay:
            assignment[f"ship.delay[0,1,{second_delay}]"] = 1
        raw_realizations.add(
            tuple(_direct_simulation(source, assignment)["ship.arrival_count"])
        )

    symmetric_realizations = {
        tuple(bundle.evaluate(full)["ship.arrival_count"])
        for _, full in _feasible_primary_assignments(bundle)
    }
    assert symmetric_realizations == raw_realizations


@pytest.mark.parametrize(
    ("capacity", "expected_classification"),
    [(100.0, "always_nonsaturated"), (0.0, "always_saturated"), (3.0, "crossing")],
)
def test_quay_crane_saturation_uses_all_exact_paths(
    tmp_path, capacity: float, expected_classification: str
) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    source = _source(installed_quay_cranes=capacity)
    bundle = builder.build_ship_delay_bundle(source, tmp_path)
    classifications = set(bundle.proofs["quay_crane_saturation_classification"])
    assert expected_classification in classifications
    assert set(bundle.proofs["quay_crane_bound_statuses"]) == {"OPTIMAL"}


def test_ship_bundle_hash_is_stable_across_output_directories(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    left = builder.build_ship_delay_bundle(_source(), tmp_path / "left")
    right = builder.build_ship_delay_bundle(_source(), tmp_path / "right")
    assert left.bundle_sha256 == right.bundle_sha256
