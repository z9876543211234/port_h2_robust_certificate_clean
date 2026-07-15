"""Migrate fixed inputs and freeze three-day NHPP integer ship timetables.

The generated 24-period package is intentionally unresolved for the H2 landing
delay.  The user also requested unresolved spill costs and ship-delay budgets.
Formal loaders must reject those placeholders until they are filled.

The legacy study contributes only the documented NHPP intensity shape and its
11.52-ship daily expectation.  This script samples three independent days of
integer Poisson counts once and freezes them.  It does not migrate continuous
arrival pressure, early-arrival shifts, moved fractions, or modulo wrapping.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from port_h2_contracts.horizon import HorizonProfile
from port_h2_uncertainty_builders.ship_delay.nhpp_nominal import (
    ThreeDayNhppSample,
    NhppNominalSettings,
    aggregate_three_day_sample,
    sample_three_day_arrivals,
)


LEGACY_CONFIG = (
    WORKSPACE_ROOT
    / "port_h2_robust_clean/records/最终算例结果_修复后_20260620/configs/C1_ProposedRobustMain.yaml"
)
NHPP_INTENSITY_EVIDENCE = (
    WORKSPACE_ROOT
    / "port_h2_robust_clean/experiments/generated_nhpp_budgeted_shift_20260710/"
    "budgeted_shift_arrival.py"
)
NHPP_GENERATOR = (
    PROJECT_ROOT
    / "src/port_h2_uncertainty_builders/ship_delay/nhpp_nominal.py"
)
NHPP_RANDOM_SEED = 20260713
NHPP_TARGET_DAILY_EXPECTED_ARRIVALS = 11.52


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def provenance(source: Path, payload: Any) -> dict[str, Any]:
    return {
        "source_type": "fixed_file",
        "source_reference": str(source.relative_to(WORKSPACE_ROOT)),
        "random_seed": None,
        "generation_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "raw_file_sha256": sha256_bytes(source.read_bytes()),
        "canonical_payload_sha256": sha256_bytes(canonical_bytes(payload)),
    }


def generated_provenance(raw_source: Path, payload: Any) -> dict[str, Any]:
    return {
        "source_type": "generated",
        "source_reference": str(raw_source.relative_to(PROJECT_ROOT)),
        "random_seed": NHPP_RANDOM_SEED,
        "generation_script_sha256": sha256_bytes(NHPP_GENERATOR.read_bytes()),
        "raw_file_sha256": sha256_bytes(raw_source.read_bytes()),
        "canonical_payload_sha256": sha256_bytes(canonical_bytes(payload)),
    }


def average_four(values: list[float]) -> list[float]:
    if len(values) != 96:
        raise ValueError("expected a 96-period sequence")
    return [sum(values[start : start + 4]) / 4.0 for start in range(0, 96, 4)]


def build_nhpp_samples() -> dict[str, ThreeDayNhppSample]:
    quarter_hour = sample_three_day_arrivals(
        NhppNominalSettings(
            profile=HorizonProfile.formal("quarter_hour_96", 96, 0.25),
            target_daily_expected_arrivals=NHPP_TARGET_DAILY_EXPECTED_ARRIVALS,
            random_seed=NHPP_RANDOM_SEED,
        )
    )
    hourly = aggregate_three_day_sample(
        quarter_hour,
        HorizonProfile.formal("hourly_24", 24, 1.0),
    )
    return {"quarter_hour_96": quarter_hour, "hourly_24": hourly}


def write_nhpp_csv(path: Path, sample: ThreeDayNhppSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t",
                "hour",
                "lambda_ship_per_hour",
                "expected_ship_count_per_period",
                "previous_day_arrival_count",
                "current_day_arrival_count",
                "next_day_arrival_count",
            ]
        )
        for period in range(sample.profile.periods):
            writer.writerow(
                [
                    period,
                    f"{period * sample.profile.dt_hours:.10g}",
                    f"{sample.lambda_ship_per_hour[period]:.17g}",
                    f"{sample.expected_count_per_period[period]:.17g}",
                    sample.previous_day_arrival_count[period],
                    sample.current_day_arrival_count[period],
                    sample.next_day_arrival_count[period],
                ]
            )


def physical_payload(legacy: dict[str, Any], profile: str) -> dict[str, Any]:
    horizon = 96 if profile == "quarter_hour_96" else 24
    repeat = lambda value: [float(value)] * horizon
    h2_ratio_96 = [float(value) for value in legacy["wind_farm"]["h2_split_ratio"]]
    ratio = h2_ratio_96 if horizon == 96 else average_four(h2_ratio_96)
    hydrogen = legacy["hydrogen"]
    lohc = legacy["lohc"]
    agv = legacy["agv"]
    grid = legacy["grid"]
    logistics = legacy["logistics"]
    landing_delay = int(hydrogen["delay_steps"]) if horizon == 96 else None
    landing_history = (
        [float(value) for value in hydrogen["h2_history"]]
        if horizon == 96
        else None
    )
    return {
        "profile_name": profile,
        "grid": {
            "buy_capacity_kw": float(grid["p_buy_max"]),
            "sell_capacity_kw": float(grid["p_sell_max"]),
            "buy_price_per_kwh": repeat(grid["price_buy_da"]["repeat"]),
            "sell_price_per_kwh": repeat(grid["price_sell_da"]["repeat"]),
        },
        "base_load_kw": repeat(legacy["loads"]["p_base"]["repeat"]),
        "wind_to_hydrogen_ratio": ratio,
        "hydrogen": {
            "transport_efficiency": float(hydrogen["eta_h2_transport"]),
            "conversion_kg_per_kwh": float(hydrogen["kappa_h2"]),
            "landing_delay_steps": landing_delay,
            "historical_landing_kg": landing_history,
            "initial_kg": float(hydrogen["h2_tank_init"]),
            "capacity_kg": float(hydrogen["h2_tank_max"]),
            "terminal_min_kg": float(hydrogen["h2_terminal_min"]),
        },
        "lohc": {
            "initial_kg": float(lohc["lohc_init"]),
            "capacity_kg": float(lohc["lohc_max"]),
            "terminal_min_kg": float(lohc["lohc_terminal_min"]),
            "reactor_min_kw": float(lohc["p_reactor_min"]),
            "reactor_max_kw": float(lohc["p_reactor_max"]),
            "day_ahead_ramp_kw": float(lohc["p_reactor_ramp_da"]),
            "real_time_ramp_kw": float(lohc["p_reactor_ramp_rt"]),
            "adjustment_limit_kw": float(lohc["p_reactor_adjust_limit"]),
            "energy_kwh_per_kg": float(lohc["omega"]),
            "hydrogen_yield": float(lohc["mu"]),
            "fixed_pipeline_rate_kg_per_hour": 0.0,
        },
        "agv": {
            "fleet_size": float(agv["n_total"]),
            "charger_count": float(agv["n_charger"]),
            "charge_power_per_vehicle_kw": float(agv["p_charge_single"]),
            "run_power_per_vehicle_kw": float(agv["p_run"]),
            "battery_capacity_per_vehicle_kwh": float(agv["battery_capacity"]),
            "charge_efficiency": float(agv["charge_eff"]),
            "initial_soc": float(agv["soc_init"]),
            "min_soc": float(agv["soc_min"]),
            "max_soc": float(agv["soc_max"]),
            "terminal_min_soc": float(agv["soc_terminal_min"]),
            "lohc_transport_rate_kg_per_vehicle_hour": float(agv["lohc_transport_rate"]),
            "container_rate_per_vehicle_hour": float(agv["container_transport_rate"]),
            "lohc_turn_efficiency": float(agv["eta_lohc_turn"]),
            "container_turn_efficiency": float(agv["eta_container_turn"]),
            "charge_adjustment_limit_kw": float(agv["p_charge_adjust_limit"]),
            "charge_count_ramp": float(agv["n_ch_ramp_limit"]),
        },
        "logistics": {
            "initial_backlog": float(logistics["backlog_init"]),
            "backlog_capacity": float(logistics["backlog_max"]),
        },
        "cost": {
            "grid_deviation_per_kwh": float(grid["c_grid_dev"]),
            "spill_day_ahead_per_kwh": None,
            "spill_real_time_per_kwh": None,
            "spill_deviation_per_kwh": None,
            "lohc_ramp_per_kw": float(lohc["c_ramp_da"]),
            "lohc_adjustment_per_kwh": float(lohc["c_adjust_rt"]),
            "lohc_operation_per_kg": float(lohc["c_non_electric_op"]),
            "charge_adjustment_per_kwh": float(agv["c_charge_adjust_rt"]),
            "backlog_delay_per_task_hour": float(logistics["c_delay"]),
            "terminal_backlog_per_task": float(logistics["c_delay"]),
            "lohc_export_revenue_per_kg": float(legacy["cost"]["subsidy_lohc_export"]),
        },
        "cost_scale": 1000.0,
        "unresolved": {
            "spill_costs": "User requested explicit placeholders.",
            "hourly_h2_landing_delay": (
                "One 15-minute source step has no lossless integer-hour mapping."
                if horizon == 24
                else None
            ),
        },
    }


def wind_payload(legacy: dict[str, Any], profile: str) -> dict[str, Any]:
    source = legacy["wind_farm"]
    nominal = [float(value) for value in source["p_wf_hat"]]
    down = [float(value) for value in source["delta_down"]]
    up = [float(value) for value in source["delta_up"]]
    if profile == "hourly_24":
        nominal, down, up = average_four(nominal), average_four(down), average_four(up)
    payload = {
        "profile_name": profile,
        "nominal_power_kw": nominal,
        "deviation_down_kw": down,
        "deviation_up_kw": up,
        "budget": 3 if profile == "hourly_24" else 12,
        "budget_source": "final_implementation_spec_section_7_1",
    }
    payload["provenance"] = provenance(LEGACY_CONFIG, payload)
    return payload


def ship_payload(
    legacy: dict[str, Any],
    profile: str,
    sample: ThreeDayNhppSample,
    raw_source: Path,
) -> dict[str, Any]:
    ship = legacy["ship"]
    payload = {
        "profile_name": profile,
        "previous_day_arrival_count": list(sample.previous_day_arrival_count),
        "current_day_arrival_count": list(sample.current_day_arrival_count),
        "next_day_arrival_count": list(sample.next_day_arrival_count),
        "nominal_generation": {
            "process": "nonhomogeneous_poisson_independent_increments",
            "model_nominal": "frozen_integer_ship_count_per_period",
            "day_roles": list(sample.day_roles),
            "random_seed": sample.random_seed,
            "target_daily_expected_arrivals": (
                sample.target_daily_expected_arrivals
            ),
            "realized_daily_totals": {
                role: total
                for role, total in zip(
                    sample.day_roles, sample.realized_daily_totals, strict=True
                )
            },
            "generator_reference": str(NHPP_GENERATOR.relative_to(PROJECT_ROOT)),
            "intensity_evidence_reference": str(
                NHPP_INTENSITY_EVIDENCE.relative_to(WORKSPACE_ROOT)
            ),
            "intensity_discretization": (
                "piecewise_constant_rate; Poisson interval mean=lambda_t*dt"
            ),
            "aggregation_from": (
                None if profile == "quarter_hour_96" else "quarter_hour_96"
            ),
            "equivalent_arrival_pressure_used": False,
            "resampling_during_optimization": False,
            "continuous_shift_interface_migrated": False,
            "early_arrival_allowed": False,
            "modulo_wraparound_used": False,
        },
        "dwell_steps": int(ship["dwell_periods"]) if profile == "quarter_hour_96" else 8,
        "max_delay_steps": None,
        "delayed_ship_budget": None,
        "total_delay_step_budget": None,
        "shore_power_per_ship_kw": float(ship["shore_power_per_ship"]),
        "quay_cranes_per_ship": float(ship["quay_cranes_per_ship"]),
        "installed_quay_cranes": float(ship["installed_quay_cranes"]),
        "quay_crane_power_kw": float(ship["quay_crane_power"]),
        "quay_crane_task_rate_per_hour": float(ship["quay_crane_efficiency"]),
        "unresolved": {
            "max_delay_steps": "User requested explicit placeholder.",
            "delayed_ship_budget": "User requested explicit placeholder.",
            "total_delay_step_budget": "Optional; user requested explicit placeholder.",
        },
    }
    payload["provenance"] = generated_provenance(raw_source, payload)
    return payload


def case_payload(case_name: str) -> dict[str, Any]:
    semantics = {
        "C1_ProposedRobustMain": {
            "outbound_mode": "agv_virtual_pipeline",
            "hydrogen_chain": True,
            "container_work_capacity": None,
            "lohc_work_capacity": None,
        },
        "C2_NoHydrogenRobust": {
            "outbound_mode": "disabled",
            "hydrogen_chain": False,
            "container_work_capacity": None,
            "lohc_work_capacity": None,
        },
        "C3_WorkCapacityCapsRobust": {
            "outbound_mode": "agv_virtual_pipeline",
            "hydrogen_chain": True,
            "container_work_capacity": 7.0,
            "lohc_work_capacity": 6.0,
        },
        "C4_DeterministicMain": {
            "outbound_mode": "agv_virtual_pipeline",
            "hydrogen_chain": True,
            "container_work_capacity": None,
            "lohc_work_capacity": None,
        },
    }
    return {
        "schema_version": 1,
        "case_name": case_name,
        "profiles": {
            profile: {
                "profile_reference": f"../profiles/{profile}.yaml",
                "deterministic_reference": f"../deterministic/{profile}/physical.json",
                "wind_source_reference": f"../uncertainty_sources/wind/{profile}.json",
                "ship_delay_source_reference": f"../uncertainty_sources/ship_delay/{profile}.json",
            }
            for profile in ("hourly_24", "quarter_hour_96")
        },
        **semantics[case_name],
    }


def main() -> None:
    legacy = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    ship_samples = build_nhpp_samples()
    data_root = PROJECT_ROOT / "data"
    ship_raw_sources = {
        profile: data_root
        / "uncertainty_sources"
        / "ship_delay"
        / f"nhpp_three_day_{sample.profile.periods}.csv"
        for profile, sample in ship_samples.items()
    }
    for profile, sample in ship_samples.items():
        write_nhpp_csv(ship_raw_sources[profile], sample)
    profiles = {
        "hourly_24": {"profile_name": "hourly_24", "periods": 24, "dt_hours": 1.0, "formal": True},
        "quarter_hour_96": {"profile_name": "quarter_hour_96", "periods": 96, "dt_hours": 0.25, "formal": True},
    }
    for profile, payload in profiles.items():
        write_json(data_root / "profiles" / f"{profile}.yaml", payload)
        physical = physical_payload(legacy, profile)
        physical["provenance"] = provenance(LEGACY_CONFIG, physical)
        write_json(data_root / "deterministic" / profile / "physical.json", physical)
        write_json(
            data_root / "uncertainty_sources" / "wind" / f"{profile}.json",
            wind_payload(legacy, profile),
        )
        write_json(
            data_root / "uncertainty_sources" / "ship_delay" / f"{profile}.json",
            ship_payload(
                legacy,
                profile,
                ship_samples[profile],
                ship_raw_sources[profile],
            ),
        )
    case_names = (
        "C1_ProposedRobustMain",
        "C2_NoHydrogenRobust",
        "C3_WorkCapacityCapsRobust",
        "C4_DeterministicMain",
    )
    for index, name in enumerate(case_names, start=1):
        write_json(data_root / "cases" / f"C{index}.yaml", case_payload(name))

    generated = sorted(
        path for path in data_root.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "generated_by": "tools/migrate_authoritative_inputs.py",
        "legacy_solver_code_imported": False,
        "files": {
            str(path.relative_to(PROJECT_ROOT)): sha256_bytes(path.read_bytes())
            for path in generated
        },
    }
    write_json(data_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
