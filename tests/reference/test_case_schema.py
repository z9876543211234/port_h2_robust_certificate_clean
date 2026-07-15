from __future__ import annotations

from dataclasses import replace

import pytest


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def test_case_validates_against_bundle_and_contains_no_uncertainty_rules(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    toy_case.validate(toy_joint_bundle)
    forbidden = ("gamma", "budget", "max_delay", "delay_selector")
    keys = [key.lower() for key in _walk_keys(toy_case.to_dict())]
    assert not any(fragment in key for key in keys for fragment in forbidden)


def test_fixed_wind_hydrogen_split_must_fit_full_wind_bundle(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    broken = replace(toy_case, wind_to_hydrogen_ratio=(0.95,) * 4)
    with pytest.raises(ValueError, match="wind-to-hydrogen"):
        broken.validate(toy_joint_bundle)


def test_no_hydrogen_case_structurally_omits_hydrogen_and_lohc(
    toy_c2_case, toy_joint_bundle
) -> None:
    toy_c2_case = toy_c2_case()
    toy_c2_case.validate(toy_joint_bundle)
    assert toy_c2_case.hydrogen is None
    assert toy_c2_case.lohc is None
    assert toy_c2_case.outbound_mode == "disabled"


def test_c3_requires_both_work_capacity_caps(toy_case, toy_joint_bundle) -> None:
    toy_case = toy_case()
    c3 = replace(
        toy_case,
        case_name="C3_WorkCapacityCapsRobust",
        container_work_capacity=6.0,
        lohc_work_capacity=None,
    )
    with pytest.raises(ValueError, match="both work-capacity"):
        c3.validate(toy_joint_bundle)
