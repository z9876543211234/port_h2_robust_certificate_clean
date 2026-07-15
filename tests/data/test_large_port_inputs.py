from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from port_h2_certificate.load_case import load_case
from runners.input_adapters import build_joint_bundle
from tools.build_large_port_inputs import build_large_port_inputs


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_large_port_builder_preserves_integer_ship_draw_and_scales_wind(tmp_path) -> None:
    paths = build_large_port_inputs(tmp_path / "inputs")
    physical = _read_json(paths["physical"])
    wind = _read_json(paths["wind"])
    ship = _read_json(paths["ship_delay"])

    assert physical["grid"]["buy_capacity_kw"] == 60_000.0
    assert physical["grid"]["sell_capacity_kw"] == 50_000.0
    assert set(physical["base_load_kw"]) == {2_500.0}
    assert physical["agv"]["fleet_size"] == 80.0
    assert physical["agv"]["charger_count"] == 24.0
    assert physical["agv"]["charge_power_per_vehicle_kw"] == 150.0
    assert physical["agv"]["battery_capacity_per_vehicle_kwh"] == 350.0
    assert physical["agv"]["run_power_per_vehicle_kw"] == 18.0
    assert physical["agv"]["charge_count_ramp"] is None
    assert physical["hydrogen"]["initial_kg"] == 1_150.0
    assert physical["hydrogen"]["capacity_kg"] == 2_050.0
    assert physical["lohc"]["initial_kg"] == 680.0
    assert physical["lohc"]["capacity_kg"] == 1_050.0
    assert physical["lohc"]["reactor_min_kw"] == 200.0
    assert physical["lohc"]["reactor_max_kw"] == 2_500.0
    assert physical["logistics"]["backlog_capacity"] == 12_000.0
    assert physical["cost"]["spill_day_ahead_per_kwh"] == 0.1
    assert physical["cost"]["spill_real_time_per_kwh"] == 0.1
    assert physical["cost"]["spill_deviation_per_kwh"] == 0.125

    assert max(wind["nominal_power_kw"]) == pytest.approx(50_000.0, abs=1e-9)
    assert wind["budget"] == 12
    nominal = np.asarray(wind["nominal_power_kw"], dtype=float)
    down = np.asarray(wind["deviation_down_kw"], dtype=float)
    up = np.asarray(wind["deviation_up_kw"], dtype=float)
    assert down == pytest.approx(0.2 * nominal, abs=1e-9)
    assert up == pytest.approx(0.2 * nominal, abs=1e-9)

    assert tuple(
        sum(ship[name])
        for name in (
            "previous_day_arrival_count",
            "current_day_arrival_count",
            "next_day_arrival_count",
        )
    ) == (15, 15, 9)
    assert all(
        isinstance(value, int)
        for name in (
            "previous_day_arrival_count",
            "current_day_arrival_count",
            "next_day_arrival_count",
        )
        for value in ship[name]
    )
    assert ship["shore_power_per_ship_kw"] == 3_000.0
    assert ship["quay_cranes_per_ship"] == 4.0
    assert ship["installed_quay_cranes"] == 16.0
    assert ship["quay_crane_power_kw"] == 500.0
    assert ship["max_delay_steps"] == 4
    assert ship["delayed_ship_budget"] == 2
    assert ship["total_delay_step_budget"] is None


def test_large_port_case_loads_and_builds_joint_bundle(tmp_path) -> None:
    paths = build_large_port_inputs(tmp_path / "inputs")
    loaded = load_case(paths["case"], "quarter_hour_96")
    bundle = build_joint_bundle(loaded, tmp_path / "bundle")

    loaded.case.validate(bundle)
    assert loaded.case.case_name == "C1_ProposedRobustMain"
    assert loaded.case.profile.periods == 96
    assert loaded.case.cost.spill_deviation_per_kwh == 0.125
    assert bundle.model_horizon == 96
    assert bundle.proofs["wind_bundle_sha256"]
    assert bundle.proofs["ship_bundle_sha256"]
    assert bundle.evaluate(bundle.nominal_selector)["ship.arrival_count"].sum() == 15
    assert min(bundle.evaluate(bundle.nominal_selector)["wind.available_power_kw"]) >= 0.0


def test_confirmed_main_budget_control_calibration_is_applied_to_c1(tmp_path) -> None:
    from port_h2_certificate.first_stage_ir import build_first_stage_ir
    from port_h2_certificate.recourse_ir import build_recourse_ir

    paths = build_large_port_inputs(
        tmp_path / "inputs",
        control_calibration="confirmed_main_budget",
    )
    physical = _read_json(paths["physical"])
    ship = _read_json(paths["ship_delay"])
    mapping = _read_json(paths["mapping"])

    assert physical["agv"]["fleet_size"] == 60.0
    assert physical["agv"]["charger_count"] == 18.0
    assert physical["agv"]["charge_adjustment_limit_kw"] == 900.0
    assert physical["agv"]["charge_count_ramp"] == 4.0
    assert physical["lohc"]["adjustment_limit_kw"] == 1_200.0
    assert physical["cost"]["charge_adjustment_per_kwh"] == 0.3
    assert physical["cost"]["lohc_adjustment_per_kwh"] == 0.3
    assert physical["cost"]["grid_deviation_per_kwh"] == 0.4
    assert physical["cost"]["spill_day_ahead_per_kwh"] == 0.1
    assert physical["cost"]["spill_real_time_per_kwh"] == 0.1
    assert physical["cost"]["spill_deviation_per_kwh"] == 0.125
    assert ship["delayed_ship_budget"] == 2
    assert mapping["control_calibration"] == "confirmed_main_budget"

    loaded = load_case(paths["case_c1"], "quarter_hour_96")
    bundle = build_joint_bundle(loaded, tmp_path / "bundle")
    first_stage = build_first_stage_ir(loaded.case, bundle)
    recourse = build_recourse_ir(loaded.case, bundle)
    assert any(
        row.name.startswith("agv.day_ahead_charge_ramp_positive")
        for row in first_stage.constraints
    )
    assert any(
        row.name.startswith("agv.charge_ramp_positive")
        for row in recourse.constraints
    )


def test_large_port_builder_writes_all_four_cases_with_scaled_c3_caps(
    tmp_path,
) -> None:
    paths = build_large_port_inputs(tmp_path / "inputs")

    cases = {
        key: _read_json(paths[key])
        for key in ("case_c1", "case_c2", "case_c3", "case_c4")
    }

    assert cases["case_c1"]["case_name"] == "C1_ProposedRobustMain"
    assert cases["case_c2"]["case_name"] == "C2_NoHydrogenRobust"
    assert cases["case_c2"]["hydrogen_chain"] is False
    assert cases["case_c2"]["outbound_mode"] == "disabled"
    assert cases["case_c3"]["case_name"] == "C3_WorkCapacityCapsRobust"
    assert cases["case_c3"]["container_work_capacity"] == 43.0
    assert cases["case_c3"]["lohc_work_capacity"] == 37.0
    assert (
        cases["case_c3"]["container_work_capacity"]
        + cases["case_c3"]["lohc_work_capacity"]
    ) == pytest.approx(80.0)
    assert cases["case_c4"]["case_name"] == "C4_DeterministicMain"


def test_large_port_four_cases_load_and_share_one_uncertainty_bundle(tmp_path) -> None:
    paths = build_large_port_inputs(tmp_path / "inputs")
    loaded_cases = [
        load_case(paths[key], "quarter_hour_96")
        for key in ("case_c1", "case_c2", "case_c3", "case_c4")
    ]
    bundles = [
        build_joint_bundle(loaded, tmp_path / f"bundle_{index}")
        for index, loaded in enumerate(loaded_cases, start=1)
    ]

    for loaded, bundle in zip(loaded_cases, bundles, strict=True):
        loaded.case.validate(bundle)

    assert len({bundle.bundle_sha256 for bundle in bundles}) == 1
    assert [loaded.case.case_name for loaded in loaded_cases] == [
        "C1_ProposedRobustMain",
        "C2_NoHydrogenRobust",
        "C3_WorkCapacityCapsRobust",
        "C4_DeterministicMain",
    ]


def test_large_port_builder_writes_reproducible_manifest(tmp_path) -> None:
    first = build_large_port_inputs(tmp_path / "first")
    second = build_large_port_inputs(tmp_path / "second")

    first_manifest = _read_json(first["manifest"])
    second_manifest = _read_json(second["manifest"])
    assert first_manifest["files"] == second_manifest["files"]
    assert first_manifest["calibration_name"] == "large_port_12v_per_day"
    assert first_manifest["formal_baseline_overwritten"] is False
    assert first_manifest["equivalent_ship_pressure_used"] is False
