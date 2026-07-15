"""Build traceable plotting tables from the certified C1 formal run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from port_h2_certificate.load_case import load_case
from runners.input_adapters import build_joint_bundle

from .checks import (
    assert_agv_conservation,
    assert_integer_arrivals,
    assert_no_nan_or_inf,
    assert_power_balance,
    assert_state_lengths,
)
from .data_loader import FormalRunError, load_formal_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = PROJECT_ROOT / "experiments/c1_confirmed_main_budget_20260714"
INPUT_ROOT = FORMAL_ROOT / "inputs"
C1_RUN_ROOT = FORMAL_ROOT / "C1_ProposedRobustMain_refined_partition"
PROFILE_NAME = "quarter_hour_96"


@dataclass(frozen=True)
class CorePlotArtifacts:
    """In-memory tables and their written, publication-traceable paths."""

    flows: pd.DataFrame
    states: pd.DataFrame
    flows_path: Path
    states_path: Path
    bundle_sha256: str


def _series(frame: pd.DataFrame, name: str, horizon: int) -> np.ndarray:
    values = (
        frame.loc[frame["period"] < horizon]
        .sort_values("period")[name]
        .to_numpy(dtype=float)
    )
    if len(values) != horizon:
        raise FormalRunError(f"{name} must have {horizon} operating-period values")
    return values


def build_core_plot_data(output_directory: str | Path) -> CorePlotArtifacts:
    """Create the C1 flow/state tables used by Figs. 1--3.

    The uncertainty bundle is rebuilt from the archived formal inputs and must
    hash-identically match the certified run before any plotting data are used.
    """

    output_root = Path(output_directory).resolve()
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    run = load_formal_run(C1_RUN_ROOT)
    loaded = load_case(INPUT_ROOT / "C1_large_port_12v.yaml", PROFILE_NAME)
    case = loaded.case
    bundle = build_joint_bundle(loaded, output_root / "_bundle_cache")
    if bundle.bundle_sha256 != run.certificate["joint_bundle_sha256"]:
        raise FormalRunError(
            "rebuilt uncertainty bundle does not match the certified C1 bundle"
        )

    horizon = case.profile.periods
    dt = case.profile.dt_hours
    periods = np.arange(horizon, dtype=int)
    time_hour = periods * dt
    nominal = bundle.evaluate(bundle.nominal_selector)
    worst = run.worst_scenario.sort_values("period")
    day_ahead = run.day_ahead.sort_values("period")
    recourse = run.recourse_worst.sort_values("period")

    wind_nominal_kw = np.asarray(
        loaded.wind_source_payload["nominal_power_kw"], dtype=float
    )
    wind_lower_kw = wind_nominal_kw - np.asarray(
        loaded.wind_source_payload["deviation_down_kw"], dtype=float
    )
    wind_upper_kw = wind_nominal_kw + np.asarray(
        loaded.wind_source_payload["deviation_up_kw"], dtype=float
    )
    wind_worst_kw = worst["wind.available_power_kw"].to_numpy(dtype=float)
    fixed_hydrogen_mw = case.hydrogen_power_day_ahead_kw(bundle) / 1000.0
    wind_nominal_electric_mw = wind_nominal_kw / 1000.0 - fixed_hydrogen_mw
    wind_worst_electric_mw = wind_worst_kw / 1000.0 - fixed_hydrogen_mw

    base_load_mw = np.asarray(case.base_load_kw, dtype=float) / 1000.0
    nominal_shore_mw = np.asarray(nominal["ship.shore_power_kw"]) / 1000.0
    nominal_quay_mw = np.asarray(nominal["ship.quay_crane_power_kw"]) / 1000.0
    worst_shore_mw = worst["ship.shore_power_kw"].to_numpy(dtype=float) / 1000.0
    worst_quay_mw = (
        worst["ship.quay_crane_power_kw"].to_numpy(dtype=float) / 1000.0
    )

    agv_charge_da_mw = (
        day_ahead["agv_charge_count_da"].to_numpy(dtype=float)
        * case.agv.charge_power_per_vehicle_kw
        / 1000.0
    )
    agv_charge_rt_mw = _series(
        recourse, "agv_charge_power_rt_mw", horizon
    )
    lohc_da_mw = day_ahead["lohc_power_da_mw"].to_numpy(dtype=float)
    lohc_rt_mw = _series(recourse, "lohc_power_rt_mw", horizon)
    spill_da_mw = day_ahead["spill_da_mw"].to_numpy(dtype=float)
    spill_rt_mw = _series(recourse, "spill_rt_mw", horizon)
    grid_buy_da_mw = day_ahead["grid_buy_da_mw"].to_numpy(dtype=float)
    grid_sell_da_mw = day_ahead["grid_sell_da_mw"].to_numpy(dtype=float)
    grid_buy_rt_mw = _series(recourse, "grid_buy_rt_mw", horizon)
    grid_sell_rt_mw = _series(recourse, "grid_sell_rt_mw", horizon)

    da_residual = (
        grid_buy_da_mw
        - grid_sell_da_mw
        + wind_nominal_electric_mw
        - base_load_mw
        - nominal_shore_mw
        - nominal_quay_mw
        - lohc_da_mw
        - agv_charge_da_mw
        - spill_da_mw
    )
    rt_residual = (
        grid_buy_rt_mw
        - grid_sell_rt_mw
        + wind_worst_electric_mw
        - base_load_mw
        - worst_shore_mw
        - worst_quay_mw
        - lohc_rt_mw
        - agv_charge_rt_mw
        - spill_rt_mw
    )

    h2_landing_kg = case.hydrogen_landing_kg(bundle)
    flows = pd.DataFrame(
        {
            "period": periods,
            "time_hour": time_hour,
            "wind_nominal_mw": wind_nominal_kw / 1000.0,
            "wind_lower_mw": wind_lower_kw / 1000.0,
            "wind_upper_mw": wind_upper_kw / 1000.0,
            "wind_worst_mw": wind_worst_kw / 1000.0,
            "wind_fixed_hydrogen_mw": fixed_hydrogen_mw,
            "wind_nominal_electric_mw": wind_nominal_electric_mw,
            "wind_worst_electric_mw": wind_worst_electric_mw,
            "arrival_nominal_count": np.asarray(
                nominal["ship.arrival_count"], dtype=float
            ),
            "arrival_worst_count": worst["ship.arrival_count"].to_numpy(
                dtype=float
            ),
            "in_port_nominal_count": np.asarray(
                nominal["ship.in_port_count"], dtype=float
            ),
            "in_port_worst_count": worst["ship.in_port_count"].to_numpy(
                dtype=float
            ),
            "active_quay_cranes_nominal": np.asarray(
                nominal["ship.active_quay_cranes"], dtype=float
            ),
            "active_quay_cranes_worst": worst[
                "ship.active_quay_cranes"
            ].to_numpy(dtype=float),
            "task_release_nominal_teu": np.asarray(
                nominal["ship.task_release"], dtype=float
            ),
            "task_release_worst_teu": worst["ship.task_release"].to_numpy(
                dtype=float
            ),
            "task_release_rate_teu_per_h": worst[
                "ship.task_release"
            ].to_numpy(dtype=float)
            / dt,
            "task_done_rate_teu_per_h": _series(
                recourse, "task_done", horizon
            )
            / dt,
            "base_load_mw": base_load_mw,
            "shore_power_nominal_mw": nominal_shore_mw,
            "shore_power_worst_mw": worst_shore_mw,
            "quay_crane_power_nominal_mw": nominal_quay_mw,
            "quay_crane_power_worst_mw": worst_quay_mw,
            "grid_buy_da_mw": grid_buy_da_mw,
            "grid_sell_da_mw": grid_sell_da_mw,
            "grid_buy_rt_mw": grid_buy_rt_mw,
            "grid_sell_rt_mw": grid_sell_rt_mw,
            "agv_charge_count_da": day_ahead[
                "agv_charge_count_da"
            ].to_numpy(dtype=float),
            "agv_charge_count": _series(recourse, "agv_charge_count", horizon),
            "agv_charge_power_da_mw": agv_charge_da_mw,
            "agv_charge_power_rt_mw": agv_charge_rt_mw,
            "agv_container_count": _series(
                recourse, "agv_container_count", horizon
            ),
            "agv_lohc_count": _series(recourse, "agv_lohc_count", horizon),
            "agv_idle_count": _series(recourse, "agv_idle_count", horizon),
            "lohc_power_da_mw": lohc_da_mw,
            "lohc_power_rt_mw": lohc_rt_mw,
            "spill_da_mw": spill_da_mw,
            "spill_rt_mw": spill_rt_mw,
            "h2_landing_rate_kg_per_h": h2_landing_kg / dt,
            "h2_consumption_rate_kg_per_h": _series(
                recourse, "hydrogen_to_lohc_kg", horizon
            )
            / dt,
            "lohc_production_rate_kg_per_h": _series(
                recourse, "lohc_production_kg", horizon
            )
            / dt,
            "lohc_outbound_rate_kg_per_h": _series(
                recourse, "lohc_outbound_kg", horizon
            )
            / dt,
            "da_power_balance_residual_mw": da_residual,
            "rt_power_balance_residual_mw": rt_residual,
        }
    )

    state_periods = np.arange(horizon + 1, dtype=int)
    states = pd.DataFrame(
        {
            "period": state_periods,
            "time_hour": state_periods * dt,
            "hydrogen_inventory_kg": recourse[
                "hydrogen_inventory_kg"
            ].to_numpy(dtype=float),
            "lohc_inventory_kg": recourse["lohc_inventory_kg"].to_numpy(
                dtype=float
            ),
            "soc": recourse["soc"].to_numpy(dtype=float),
            "backlog_teu": recourse["backlog_tasks"].to_numpy(dtype=float),
        }
    )

    assert_state_lengths(len(flows), len(states))
    assert_integer_arrivals(flows["arrival_nominal_count"])
    assert_integer_arrivals(flows["arrival_worst_count"])
    assert_power_balance(flows["da_power_balance_residual_mw"])
    assert_power_balance(flows["rt_power_balance_residual_mw"])
    assert_agv_conservation(
        flows[
            [
                "agv_container_count",
                "agv_lohc_count",
                "agv_charge_count",
                "agv_idle_count",
            ]
        ].sum(axis=1),
        case.agv.fleet_size,
    )
    assert_no_nan_or_inf(flows.to_numpy(), name="C1 flow table")
    assert_no_nan_or_inf(states.to_numpy(), name="C1 state table")

    flows_path = data_root / "c1_plot_flows.csv"
    states_path = data_root / "c1_plot_states.csv"
    flows.to_csv(flows_path, index=False)
    states.to_csv(states_path, index=False)
    return CorePlotArtifacts(
        flows=flows,
        states=states,
        flows_path=flows_path,
        states_path=states_path,
        bundle_sha256=bundle.bundle_sha256,
    )

