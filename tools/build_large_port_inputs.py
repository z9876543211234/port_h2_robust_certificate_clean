"""Build the isolated 12-vessel/day large-port C1--C4 input package.

This offline builder adapts the previously documented large-port calibration
to the certificate-clean schema.  It never overwrites the baseline ``data/``
tree and keeps the frozen three-day integer NHPP draw unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
BASE_PHYSICAL = PROJECT_ROOT / "data/deterministic/quarter_hour_96/physical.json"
BASE_WIND = PROJECT_ROOT / "data/uncertainty_sources/wind/quarter_hour_96.json"
BASE_SHIP = PROJECT_ROOT / "data/uncertainty_sources/ship_delay/quarter_hour_96.json"
BASE_NHPP_CSV = PROJECT_ROOT / "data/uncertainty_sources/ship_delay/nhpp_three_day_96.csv"
BASE_PROFILE = PROJECT_ROOT / "data/profiles/quarter_hour_96.yaml"
REFERENCE_ROOT = WORKSPACE_ROOT / "experiments/inputs/realistic_offshore_port_20260621"
REFERENCE_CONFIG = REFERENCE_ROOT / "model_realistic_offshore_port.yaml"
REFERENCE_TIMESERIES = REFERENCE_ROOT / "timeseries.csv"
REFERENCE_H2_HISTORY = REFERENCE_ROOT / "h2_history.csv"
REFERENCE_SCALING = REFERENCE_ROOT / "scaling_summary.json"
CONFIRMED_CONTROL_SOURCE = (
    WORKSPACE_ROOT
    / "port_h2_robust_clean/experiments/active_dayahead_shift_arrival_20260710/shift_arrival_experiment/realistic_case_set.py"
)
CONFIRMED_CONTROL_BASE = (
    WORKSPACE_ROOT
    / "port_h2_robust_clean/records/parameter_scan_realistic_c1_ramp_20260622/case_comparison_generated_nhpp_current_scale_20260623_qc19_formal_rerun/configs/C1_ProposedRobustMain.yaml"
)
CONTROL_CALIBRATIONS = ("exploratory_reference", "confirmed_main_budget")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _source_bundle_sha256(paths: tuple[Path, ...]) -> str:
    return _canonical_sha256(
        {
            str(path.relative_to(WORKSPACE_ROOT)): _sha256_bytes(path.read_bytes())
            for path in paths
        }
    )


def _source_references(paths: tuple[Path, ...]) -> str:
    return ";".join(str(path.relative_to(WORKSPACE_ROOT)) for path in paths)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fixed_provenance(
    payload: dict[str, Any], sources: tuple[Path, ...], script_sha256: str
) -> dict[str, Any]:
    return {
        "source_type": "fixed_file",
        "source_reference": _source_references(sources),
        "random_seed": None,
        "generation_script_sha256": script_sha256,
        "raw_file_sha256": _source_bundle_sha256(sources),
        "canonical_payload_sha256": _canonical_sha256(payload),
    }


def _generated_provenance(
    payload: dict[str, Any], sources: tuple[Path, ...], script_sha256: str, seed: int
) -> dict[str, Any]:
    return {
        "source_type": "generated",
        "source_reference": _source_references(sources),
        "random_seed": seed,
        "generation_script_sha256": script_sha256,
        "raw_file_sha256": _source_bundle_sha256(sources),
        "canonical_payload_sha256": _canonical_sha256(payload),
    }


def _physical_payload(
    base: dict[str, Any], reference: dict[str, Any], series, history_power_kw: float
) -> dict[str, Any]:
    horizon = 96
    dt_hours = 0.25
    agv = reference["agv"]
    inventory = reference["inventory"]
    limits = reference["limits"]
    lohc = reference["lohc"]
    hydrogen = reference["hydrogen"]
    total_battery_kwh = float(agv["e_agv_max"]) / 0.9
    battery_per_vehicle_kwh = total_battery_kwh / float(agv["n_agv_total"])
    base_cost = dict(base["cost"])
    base_cost.update(
        {
            "spill_day_ahead_per_kwh": 0.1,
            "spill_real_time_per_kwh": 0.1,
            "spill_deviation_per_kwh": 0.125,
        }
    )
    historical_landing_kg = (
        float(hydrogen["eta_h2_transport"])
        * float(hydrogen["kappa_h2"])
        * history_power_kw
        * dt_hours
    )
    direct_share = float(reference["offshore_energy"]["direct_electric_share"])
    return {
        "profile_name": "quarter_hour_96",
        "grid": {
            "buy_capacity_kw": float(limits["p_grid_max"]),
            "sell_capacity_kw": 50_000.0,
            "buy_price_per_kwh": [float(row["grid_buy_price"]) for row in series],
            "sell_price_per_kwh": [float(row["grid_sell_price"]) for row in series],
        },
        "base_load_kw": [float(row["p_load_base"]) for row in series],
        "wind_to_hydrogen_ratio": [1.0 - direct_share] * horizon,
        "hydrogen": {
            "transport_efficiency": float(hydrogen["eta_h2_transport"]),
            "conversion_kg_per_kwh": float(hydrogen["kappa_h2"]),
            "landing_delay_steps": int(hydrogen["tau_h"]),
            "historical_landing_kg": [historical_landing_kg],
            "initial_kg": float(inventory["s_h2_init"]),
            "capacity_kg": float(inventory["s_h2_max"]),
            "terminal_min_kg": float(inventory["s_h2_terminal_min"]),
        },
        "lohc": {
            "initial_kg": float(inventory["s_lohc_init"]),
            "capacity_kg": float(inventory["s_lohc_max"]),
            "terminal_min_kg": float(inventory["s_lohc_terminal_min"]),
            "reactor_min_kw": float(limits["p_lohc_min"]),
            "reactor_max_kw": float(limits["p_lohc_max"]),
            "day_ahead_ramp_kw": float(limits["p_lohc_ramp"]),
            "real_time_ramp_kw": float(limits["p_lohc_ramp"]),
            "adjustment_limit_kw": float(limits["delta_p_lohc"]),
            "energy_kwh_per_kg": float(lohc["omega_lohc"]),
            "hydrogen_yield": float(lohc["mu_lohc"]),
            "fixed_pipeline_rate_kg_per_hour": 0.0,
        },
        "agv": {
            "fleet_size": float(agv["n_agv_total"]),
            "charger_count": float(agv["n_ch_max"]),
            "charge_power_per_vehicle_kw": float(agv["p_ch_single"]),
            "run_power_per_vehicle_kw": float(agv["e_h"]) / dt_hours,
            "battery_capacity_per_vehicle_kwh": battery_per_vehicle_kwh,
            "charge_efficiency": float(agv["eta_ch"]),
            "initial_soc": float(agv["e_agv_init"]) / total_battery_kwh,
            "min_soc": float(agv["e_agv_min"]) / total_battery_kwh,
            "max_soc": float(agv["e_agv_max"]) / total_battery_kwh,
            "terminal_min_soc": float(agv["e_agv_terminal_min"]) / total_battery_kwh,
            "lohc_transport_rate_kg_per_vehicle_hour": float(lohc["f_h_single"]),
            "container_rate_per_vehicle_hour": float(lohc["c_agv_task_rate"]),
            "lohc_turn_efficiency": float(lohc["eta_turn"]),
            "container_turn_efficiency": float(
                base["agv"]["container_turn_efficiency"]
            ),
            "charge_adjustment_limit_kw": float(agv["delta_p_agv_ch"]),
            "charge_count_ramp": None,
        },
        "logistics": {
            "initial_backlog": float(inventory["u_c_init"]),
            "backlog_capacity": float(inventory["u_c_max"]),
        },
        "cost": base_cost,
        "cost_scale": float(base["cost_scale"]),
    }


def _apply_control_calibration(
    physical: dict[str, Any], control_calibration: str
) -> None:
    if control_calibration == "exploratory_reference":
        return
    if control_calibration != "confirmed_main_budget":
        raise ValueError(
            f"unsupported control calibration: {control_calibration}"
        )
    physical["agv"].update(
        {
            "fleet_size": 60.0,
            "charger_count": 18.0,
            "charge_adjustment_limit_kw": 900.0,
            "charge_count_ramp": 4.0,
        }
    )
    physical["lohc"]["adjustment_limit_kw"] = 1_200.0
    physical["cost"].update(
        {
            "charge_adjustment_per_kwh": 0.3,
            "lohc_adjustment_per_kwh": 0.3,
            "grid_deviation_per_kwh": 0.4,
        }
    )


def _wind_payload(base: dict[str, Any], scaling: dict[str, Any]) -> dict[str, Any]:
    # The reference summary records a 50 MW design target with a floating-point
    # roundoff tail.  Round to the physical kW resolution before scaling so the
    # generated input has the exact documented peak rather than 50000.00000000x.
    target_peak_kw = round(float(scaling["total_wind_peak_kw"]))
    scale = target_peak_kw / max(float(value) for value in base["nominal_power_kw"])
    return {
        "profile_name": "quarter_hour_96",
        "nominal_power_kw": [float(value) * scale for value in base["nominal_power_kw"]],
        "deviation_down_kw": [
            float(value) * scale for value in base["deviation_down_kw"]
        ],
        "deviation_up_kw": [
            float(value) * scale for value in base["deviation_up_kw"]
        ],
        "budget": 12,
        "budget_source": "retained_certificate_clean_formal_spec_gamma_12",
        "calibration": {
            "target_peak_kw": target_peak_kw,
            "scale_factor": scale,
            "shape_preserved": True,
            "relative_deviation_preserved": True,
        },
    }


def _ship_payload(base: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    logistics = reference["logistics"]
    payload.update(
        {
            "dwell_steps": int(logistics["vessel_dwell_periods"]),
            "max_delay_steps": 4,
            "delayed_ship_budget": 2,
            "total_delay_step_budget": None,
            "shore_power_per_ship_kw": float(logistics["shore_power_per_ship"]),
            "quay_cranes_per_ship": float(logistics["quay_cranes_per_ship"]),
            "installed_quay_cranes": float(logistics["installed_quay_cranes"]),
            "quay_crane_power_kw": float(logistics["quay_crane_power"]),
            "quay_crane_task_rate_per_hour": float(
                logistics["quay_crane_efficiency"]
            ),
            "unresolved": {
                "total_delay_step_budget": "Optional budget disabled by null."
            },
            "large_port_calibration": {
                "equivalent_arrival_pressure_used": False,
                "integer_three_day_draw_preserved": True,
            },
        }
    )
    payload.pop("provenance", None)
    return payload


def build_large_port_inputs(
    output_directory: str | Path,
    *,
    control_calibration: str = "exploratory_reference",
) -> dict[str, Path]:
    if control_calibration not in CONTROL_CALIBRATIONS:
        raise ValueError(
            f"control_calibration must be one of {CONTROL_CALIBRATIONS}"
        )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    base_physical = _read_json(BASE_PHYSICAL)
    base_wind = _read_json(BASE_WIND)
    base_ship = _read_json(BASE_SHIP)
    reference_scaling = _read_json(REFERENCE_SCALING)
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError("large-port reference config must be a mapping")
    timeseries = _read_csv(REFERENCE_TIMESERIES)
    history = _read_csv(REFERENCE_H2_HISTORY)
    if len(timeseries) != 96 or not history:
        raise ValueError("large-port reference time series is incomplete")
    script_sha256 = _sha256_bytes(Path(__file__).read_bytes())

    physical = _physical_payload(
        base_physical, reference, timeseries, float(history[-1]["p_h2_src_history"])
    )
    _apply_control_calibration(physical, control_calibration)
    physical_sources = (
        BASE_PHYSICAL,
        REFERENCE_CONFIG,
        REFERENCE_TIMESERIES,
        REFERENCE_H2_HISTORY,
    )
    if control_calibration == "confirmed_main_budget":
        physical_sources += (CONFIRMED_CONTROL_SOURCE, CONFIRMED_CONTROL_BASE)
    physical["provenance"] = _fixed_provenance(
        physical, physical_sources, script_sha256
    )

    wind = _wind_payload(base_wind, reference_scaling)
    wind_sources = (BASE_WIND, REFERENCE_CONFIG, REFERENCE_SCALING)
    wind["provenance"] = _fixed_provenance(wind, wind_sources, script_sha256)

    ship = _ship_payload(base_ship, reference)
    ship_sources = (BASE_NHPP_CSV, REFERENCE_CONFIG)
    seed = int(base_ship["provenance"]["random_seed"])
    ship["provenance"] = _generated_provenance(
        ship, ship_sources, script_sha256, seed
    )

    profile = yaml.safe_load(BASE_PROFILE.read_text(encoding="utf-8"))
    profile_references = {
        "quarter_hour_96": {
            "profile_reference": "profile_96.yaml",
            "deterministic_reference": "physical_large_port_96.json",
            "wind_source_reference": "wind_large_port_96.json",
            "ship_delay_source_reference": "ship_delay_large_port_96.json",
        }
    }
    calibration_name = (
        "large_port_12v_confirmed_main_budget"
        if control_calibration == "confirmed_main_budget"
        else "large_port_12v_per_day"
    )
    case_c1 = {
        "schema_version": 1,
        "case_name": "C1_ProposedRobustMain",
        "profiles": profile_references,
        "outbound_mode": "agv_virtual_pipeline",
        "hydrogen_chain": True,
        "container_work_capacity": None,
        "lohc_work_capacity": None,
        "calibration_name": calibration_name,
    }
    case_c2 = {
        **case_c1,
        "case_name": "C2_NoHydrogenRobust",
        "outbound_mode": "disabled",
        "hydrogen_chain": False,
    }
    fleet_size = float(physical["agv"]["fleet_size"])
    container_work_capacity = float(round(fleet_size * 7.0 / 13.0))
    lohc_work_capacity = fleet_size - container_work_capacity
    case_c3 = {
        **case_c1,
        "case_name": "C3_WorkCapacityCapsRobust",
        "container_work_capacity": container_work_capacity,
        "lohc_work_capacity": lohc_work_capacity,
    }
    case_c4 = {
        **case_c1,
        "case_name": "C4_DeterministicMain",
    }
    mapping = {
        "calibration_name": calibration_name,
        "control_calibration": control_calibration,
        "model_semantics_changed": False,
        "formal_baseline_overwritten": False,
        "ship_nominal": {
            "target_daily_expected_arrivals": ship["nominal_generation"][
                "target_daily_expected_arrivals"
            ],
            "realized_daily_totals": ship["nominal_generation"][
                "realized_daily_totals"
            ],
            "integer_draw_preserved": True,
        },
        "wind": wind["calibration"],
        "spill_cost_selection": {
            "spill_day_ahead_per_kwh": 0.1,
            "spill_real_time_per_kwh": 0.1,
            "spill_deviation_per_kwh": 0.125,
            "selection_rule": (
                "Smallest scanned value strictly above the 0.10 yuan/kWh "
                "marginal tie with the 0.20 yuan/kWh grid-deviation cost."
            ),
            "observed_transition": (
                "Worst-case spill deviation is 0.025 MWh at 0.10 and 0 MWh "
                "from 0.125 through 0.30."
            ),
        },
        "adaptations": {
            "grid_sell_capacity_kw": (
                "Clean schema supports the intended separate 50 MW sell cap."
            ),
            "agv_run_power_per_vehicle_kw": (
                "Reference has 15 kW container and 18 kW LOHC duties; clean schema "
                "has one coefficient, so the conservative 18 kW value is used."
            ),
            "agv_charge_count_ramp": (
                "Four-vehicle ramp retained from the confirmed main-budget base."
                if control_calibration == "confirmed_main_budget"
                else "Reference has no separate count-ramp field; null avoids inventing one."
            ),
            "wind_budget": "Gamma=12 retained from the certificate-clean formal spec.",
            "ship_delay": "Four 15-minute steps and at most two delayed ships retained.",
            "c3_work_capacity_caps": (
                "The baseline 13-vehicle C3 working-state split of 7:6 is "
                f"scaled to the {fleet_size:g}-vehicle fleet and rounded to "
                f"{container_work_capacity:g}:{lohc_work_capacity:g} whole-vehicle "
                "caps; idle and charging remain pooled."
            ),
        },
        "source_files": {
            "large_port_reference": str(REFERENCE_CONFIG.relative_to(WORKSPACE_ROOT)),
            "baseline_physical": str(BASE_PHYSICAL.relative_to(PROJECT_ROOT)),
            "baseline_wind": str(BASE_WIND.relative_to(PROJECT_ROOT)),
            "baseline_integer_nhpp": str(BASE_NHPP_CSV.relative_to(PROJECT_ROOT)),
            "confirmed_control_source": (
                str(CONFIRMED_CONTROL_SOURCE.relative_to(WORKSPACE_ROOT))
                if control_calibration == "confirmed_main_budget"
                else None
            ),
            "confirmed_control_base": (
                str(CONFIRMED_CONTROL_BASE.relative_to(WORKSPACE_ROOT))
                if control_calibration == "confirmed_main_budget"
                else None
            ),
        },
    }

    paths = {
        "profile": destination / "profile_96.yaml",
        "physical": destination / "physical_large_port_96.json",
        "wind": destination / "wind_large_port_96.json",
        "ship_delay": destination / "ship_delay_large_port_96.json",
        "case": destination / "C1_large_port_12v.yaml",
        "case_c1": destination / "C1_large_port_12v.yaml",
        "case_c2": destination / "C2_large_port_12v.yaml",
        "case_c3": destination / "C3_large_port_12v.yaml",
        "case_c4": destination / "C4_large_port_12v.yaml",
        "mapping": destination / "parameter_mapping.json",
        "manifest": destination / "manifest.json",
    }
    _atomic_json(paths["profile"], profile)
    _atomic_json(paths["physical"], physical)
    _atomic_json(paths["wind"], wind)
    _atomic_json(paths["ship_delay"], ship)
    _atomic_json(paths["case_c1"], case_c1)
    _atomic_json(paths["case_c2"], case_c2)
    _atomic_json(paths["case_c3"], case_c3)
    _atomic_json(paths["case_c4"], case_c4)
    _atomic_json(paths["mapping"], mapping)
    files = {
        path.name: _sha256_bytes(path.read_bytes())
        for key, path in paths.items()
        if key != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "calibration_name": calibration_name,
        "generated_by": "tools/build_large_port_inputs.py",
        "formal_baseline_overwritten": False,
        "equivalent_ship_pressure_used": False,
        "files": files,
    }
    _atomic_json(paths["manifest"], manifest)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--control-calibration",
        choices=CONTROL_CALIBRATIONS,
        default="exploratory_reference",
    )
    args = parser.parse_args()
    paths = build_large_port_inputs(
        args.output,
        control_calibration=args.control_calibration,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
