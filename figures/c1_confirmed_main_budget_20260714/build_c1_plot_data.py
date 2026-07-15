"""Build traceable plotting data from the corrected C1 certificate artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIRECTORY.parents[1]
RUN_DIRECTORY = (
    PROJECT_ROOT
    / "experiments/c1_confirmed_main_budget_20260714/C1_ProposedRobustMain_refined_partition"
)
INPUT_DIRECTORY = PROJECT_ROOT / "experiments/c1_confirmed_main_budget_20260714/inputs"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from port_h2_certificate.load_case import load_case  # noqa: E402
from runners.input_adapters import build_joint_bundle  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plot_data() -> Path:
    day_ahead = pd.read_csv(RUN_DIRECTORY / "dispatch_day_ahead.csv")
    recourse = pd.read_csv(RUN_DIRECTORY / "dispatch_recourse_worst.csv")
    worst = pd.read_csv(RUN_DIRECTORY / "worst_scenario.csv")
    physical = _read_json(INPUT_DIRECTORY / "physical_large_port_96.json")
    wind = _read_json(INPUT_DIRECTORY / "wind_large_port_96.json")

    loaded = load_case(INPUT_DIRECTORY / "C1_large_port_12v.yaml", "quarter_hour_96")
    bundle = build_joint_bundle(loaded, FIGURE_DIRECTORY / "bundle_cache")
    nominal = bundle.evaluate(bundle.nominal_selector)

    horizon = len(day_ahead)
    recourse = recourse.loc[recourse["period"] < horizon].copy()
    worst = worst.loc[worst["period"] < horizon].copy()
    if len(recourse) != horizon or len(worst) != horizon:
        raise ValueError("formal dispatch and scenario horizons do not match")

    charge_power_mw = physical["agv"]["charge_power_per_vehicle_kw"] / 1000.0
    wind_nominal_mw = np.asarray(wind["nominal_power_kw"], dtype=float) / 1000.0
    wind_to_hydrogen = np.asarray(
        physical["wind_to_hydrogen_ratio"], dtype=float
    )
    wind_hydrogen_mw = wind_nominal_mw * wind_to_hydrogen
    wind_direct_mw = wind_nominal_mw - wind_hydrogen_mw
    base_load_mw = np.asarray(physical["base_load_kw"], dtype=float) / 1000.0
    shore_nominal_mw = nominal["ship.shore_power_kw"] / 1000.0
    quay_nominal_mw = nominal["ship.quay_crane_power_kw"] / 1000.0

    data = pd.DataFrame(
        {
            "period": day_ahead["period"].astype(int),
            "time_hour": day_ahead["period"] * 0.25,
            "grid_buy_da_mw": day_ahead["grid_buy_da_mw"],
            "grid_sell_da_mw": day_ahead["grid_sell_da_mw"],
            "grid_net_da_mw": (
                day_ahead["grid_buy_da_mw"] - day_ahead["grid_sell_da_mw"]
            ),
            "wind_nominal_mw": wind_nominal_mw,
            "wind_direct_da_mw": wind_direct_mw,
            "wind_hydrogen_da_mw": wind_hydrogen_mw,
            "base_load_mw": base_load_mw,
            "shore_power_nominal_mw": shore_nominal_mw,
            "quay_crane_power_nominal_mw": quay_nominal_mw,
            "agv_charge_count_da": day_ahead["agv_charge_count_da"],
            "agv_charge_power_da_mw": (
                day_ahead["agv_charge_count_da"] * charge_power_mw
            ),
            "lohc_power_da_mw": day_ahead["lohc_power_da_mw"],
            "spill_da_mw": day_ahead["spill_da_mw"],
            "grid_net_rt_worst_mw": (
                recourse["grid_buy_rt_mw"].to_numpy()
                - recourse["grid_sell_rt_mw"].to_numpy()
            ),
            "agv_charge_power_rt_worst_mw": recourse[
                "agv_charge_power_rt_mw"
            ].to_numpy(),
            "lohc_power_rt_worst_mw": recourse["lohc_power_rt_mw"].to_numpy(),
            "spill_rt_worst_mw": recourse["spill_rt_mw"].to_numpy(),
            "wind_worst_mw": worst["wind.available_power_kw"].to_numpy() / 1000.0,
            "shore_power_worst_mw": worst["ship.shore_power_kw"].to_numpy()
            / 1000.0,
            "quay_crane_power_worst_mw": worst[
                "ship.quay_crane_power_kw"
            ].to_numpy()
            / 1000.0,
        }
    )
    data["day_ahead_demand_mw"] = (
        data["base_load_mw"]
        + data["shore_power_nominal_mw"]
        + data["quay_crane_power_nominal_mw"]
        + data["agv_charge_power_da_mw"]
        + data["lohc_power_da_mw"]
        + data["spill_da_mw"]
    )
    data["day_ahead_supply_mw"] = (
        data["grid_net_da_mw"] + data["wind_direct_da_mw"]
    )
    data["day_ahead_balance_residual_mw"] = (
        data["day_ahead_supply_mw"] - data["day_ahead_demand_mw"]
    )

    output = FIGURE_DIRECTORY / "c1_day_ahead_plot_data.csv"
    data.to_csv(output, index=False, float_format="%.12g")
    metadata = {
        "case_semantic_sha256": _read_json(RUN_DIRECTORY / "certificate.json")[
            "case_semantic_sha256"
        ],
        "joint_bundle_sha256": bundle.bundle_sha256,
        "source_run": str(RUN_DIRECTORY.relative_to(PROJECT_ROOT)),
        "maximum_day_ahead_balance_residual_mw": float(
            data["day_ahead_balance_residual_mw"].abs().max()
        ),
        "notes": {
            "day_ahead_exogenous_data": "nominal wind and nominal integer-ship trajectory",
            "real_time_data": "certificate worst-case scenario and its recourse dispatch",
            "wind_split": "40% hydrogen path and 60% direct electric path",
        },
    }
    (FIGURE_DIRECTORY / "c1_plot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(build_plot_data())
