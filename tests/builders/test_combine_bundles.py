from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from tests.builders.test_ship_delay_bundle import _source as ship_source
from tests.builders.test_wind_bundle import _source as wind_source


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _bundles(tmp_path):
    wind_builder = _module("port_h2_uncertainty_builders.wind.builder")
    ship_builder = _module("port_h2_uncertainty_builders.ship_delay.builder")
    return (
        wind_builder.build_wind_bundle(wind_source(), tmp_path / "wind"),
        ship_builder.build_ship_delay_bundle(ship_source(), tmp_path / "ship"),
    )


def test_combiner_creates_self_contained_cartesian_product(tmp_path) -> None:
    combine = _module("port_h2_uncertainty_builders.combine")
    wind, ship = _bundles(tmp_path)
    joint = combine.combine_bundles(wind, ship, tmp_path / "joint")

    assert (tmp_path / "joint" / "joint_uncertainty_bundle.json").is_file()
    assert (tmp_path / "joint" / "joint_uncertainty_matrices.npz").is_file()
    assert joint.selector_keys == wind.selector_keys + ship.selector_keys
    assert set(joint.outputs) == set(wind.outputs) | set(ship.outputs)
    assert all(
        output.selector_columns_path.startswith("joint_uncertainty_matrices.npz::")
        for output in joint.outputs.values()
    )


def test_joint_realization_equals_independent_realizations(tmp_path) -> None:
    combine = _module("port_h2_uncertainty_builders.combine")
    wind, ship = _bundles(tmp_path)
    joint = combine.combine_bundles(wind, ship, tmp_path / "joint")

    wind_primary = {key: 0 for key in wind.primary_selector_keys}
    wind_primary["wind.down[0]"] = 1
    ship_primary = {key: 0 for key in ship.primary_selector_keys}
    ship_primary["ship.delay[3,0,1]"] = 1
    wind_full = wind.complete(wind_primary)
    ship_full = ship.complete(ship_primary)
    joint_full = joint.complete({**wind_primary, **ship_primary})

    expected = {**wind.evaluate(wind_full), **ship.evaluate(ship_full)}
    actual = joint.evaluate(joint_full)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0.0)


def test_combiner_adds_no_joint_budget_or_cross_constraint(tmp_path) -> None:
    combine = _module("port_h2_uncertainty_builders.combine")
    wind, ship = _bundles(tmp_path)
    joint = combine.combine_bundles(wind, ship, tmp_path / "joint")
    assert tuple(row.name for row in joint.constraints) == tuple(
        row.name for row in wind.constraints + ship.constraints
    )
    wind_keys = set(wind.selector_keys)
    ship_keys = set(ship.selector_keys)
    for row in joint.constraints:
        used = set(row.coefficients)
        assert not (used & wind_keys and used & ship_keys)


def test_combiner_rejects_mismatched_horizon(tmp_path) -> None:
    combine = _module("port_h2_uncertainty_builders.combine")
    wind, ship = _bundles(tmp_path)
    mismatched = wind.with_profile(profile_name="different", model_horizon=8)
    with pytest.raises(ValueError, match="profile"):
        combine.combine_bundles(mismatched, ship, tmp_path / "joint")


def test_joint_bundle_hash_is_stable(tmp_path) -> None:
    combine = _module("port_h2_uncertainty_builders.combine")
    wind, ship = _bundles(tmp_path)
    left = combine.combine_bundles(wind, ship, tmp_path / "left")
    right = combine.combine_bundles(wind, ship, tmp_path / "right")
    assert left.bundle_sha256 == right.bundle_sha256
