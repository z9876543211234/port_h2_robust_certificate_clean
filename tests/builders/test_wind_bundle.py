from __future__ import annotations

import importlib
import importlib.util
from itertools import product

import numpy as np
import pytest

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _source():
    schema = _module("port_h2_uncertainty_builders.wind.schema")
    return schema.WindUncertaintySource(
        profile=HorizonProfile.toy("toy_4", 4, 6.0),
        nominal_power_kw=(10.0, 20.0, 30.0, 40.0),
        deviation_down_kw=(1.0, 2.0, 3.0, 4.0),
        deviation_up_kw=(1.5, 2.5, 3.5, 4.5),
        budget=1,
        provenance=ProvenanceRecord(
            source_type="fixed_file",
            source_reference="toy_wind.yaml",
            random_seed=None,
            generation_script_sha256=None,
            raw_file_sha256="a" * 64,
            canonical_payload_sha256="b" * 64,
        ),
    )


def test_wind_source_rejects_negative_possible_power() -> None:
    schema = _module("port_h2_uncertainty_builders.wind.schema")
    source = _source()
    broken = schema.WindUncertaintySource(
        profile=source.profile,
        nominal_power_kw=source.nominal_power_kw,
        deviation_down_kw=(11.0, 2.0, 3.0, 4.0),
        deviation_up_kw=source.deviation_up_kw,
        budget=source.budget,
        provenance=source.provenance,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        broken.validate()


def test_wind_builder_creates_exact_affine_realization(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.wind.builder")
    bundle = builder.build_wind_bundle(_source(), tmp_path)
    assert (tmp_path / "wind_uncertainty_bundle.json").is_file()
    assignment = dict(bundle.nominal_selector)
    assignment["wind.down[0]"] = 1
    realization = bundle.evaluate(assignment)
    np.testing.assert_allclose(
        realization["wind.available_power_kw"], [9.0, 20.0, 30.0, 40.0]
    )
    assert bundle.primary_selector_keys == bundle.selector_keys


def test_wind_mutual_exclusion_and_budget_are_enforced(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.wind.builder")
    bundle = builder.build_wind_bundle(_source(), tmp_path)

    both = dict(bundle.nominal_selector)
    both["wind.down[0]"] = 1
    both["wind.up[0]"] = 1
    with pytest.raises(ValueError, match="infeasible"):
        bundle.evaluate(both)

    over_budget = dict(bundle.nominal_selector)
    over_budget["wind.down[0]"] = 1
    over_budget["wind.up[1]"] = 1
    with pytest.raises(ValueError, match="infeasible"):
        bundle.evaluate(over_budget)


def test_wind_budget_one_has_exact_short_horizon_enumeration(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.wind.builder")
    bundle = builder.build_wind_bundle(_source(), tmp_path)
    feasible = 0
    for values in product((0, 1), repeat=len(bundle.selectors)):
        assignment = dict(zip(bundle.selector_keys, values, strict=True))
        try:
            bundle.evaluate(assignment)
        except ValueError:
            continue
        feasible += 1
    assert feasible == 1 + 2 * 4


def test_wind_nominal_and_bundle_hash_are_stable(tmp_path) -> None:
    builder = _module("port_h2_uncertainty_builders.wind.builder")
    left = builder.build_wind_bundle(_source(), tmp_path / "left")
    right = builder.build_wind_bundle(_source(), tmp_path / "right")
    assert set(left.nominal_selector.values()) == {0}
    assert left.bundle_sha256 == right.bundle_sha256


def test_wind_builder_source_has_no_hydrogen_or_lohc_dependency() -> None:
    module = _module("port_h2_uncertainty_builders.wind.builder")
    text = module.__loader__.get_source(module.__name__).lower()
    assert "hydrogen" not in text
    assert "lohc" not in text
