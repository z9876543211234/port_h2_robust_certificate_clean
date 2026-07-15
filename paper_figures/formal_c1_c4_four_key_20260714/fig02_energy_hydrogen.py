"""Fig. 2: C1 electricity--hydrogen coordinated dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build_data import CorePlotArtifacts
from .data_loader import FormalRun
from .style import COLORS, clean_axes, format_time_axis, new_figure, panel_label, step


FLOW_SERIES = [
    "wind_nominal_mw",
    "wind_worst_mw",
    "wind_fixed_hydrogen_mw",
    "wind_nominal_electric_mw",
    "wind_worst_electric_mw",
    "grid_buy_da_mw",
    "grid_sell_da_mw",
    "grid_buy_rt_mw",
    "grid_sell_rt_mw",
    "base_load_mw",
    "shore_power_nominal_mw",
    "shore_power_worst_mw",
    "quay_crane_power_nominal_mw",
    "quay_crane_power_worst_mw",
    "lohc_power_da_mw",
    "lohc_power_rt_mw",
    "agv_charge_power_da_mw",
    "agv_charge_power_rt_mw",
    "spill_da_mw",
    "spill_rt_mw",
    "h2_landing_rate_kg_per_h",
    "h2_consumption_rate_kg_per_h",
    "lohc_production_rate_kg_per_h",
    "lohc_outbound_rate_kg_per_h",
]
STATE_SERIES = ["hydrogen_inventory_kg", "lohc_inventory_kg"]


POWER_COLORS = {
    "Electric wind": COLORS["wind"],
    "Grid import": COLORS["grid"],
    "Base load": COLORS["base"],
    "Shore power": COLORS["shore"],
    "Quay cranes": COLORS["quay"],
    "LOHC reactor": COLORS["lohc"],
    "AGV charging": COLORS["agv_charge"],
    "Grid export": "#9A7DAD",
    "Wind spill": COLORS["spill"],
}


def _write_plot_data(artifacts: CorePlotArtifacts, data_root: Path) -> Path:
    flow_long = artifacts.flows[["period", "time_hour", *FLOW_SERIES]].melt(
        id_vars=["period", "time_hour"], var_name="series", value_name="value"
    )
    flow_long.insert(0, "trajectory_type", "flow")
    state_long = artifacts.states[["period", "time_hour", *STATE_SERIES]].melt(
        id_vars=["period", "time_hour"], var_name="series", value_name="value"
    )
    state_long.insert(0, "trajectory_type", "state")
    output = pd.concat([flow_long, state_long], ignore_index=True)
    path = data_root / "fig02_plot_data.csv"
    output.to_csv(path, index=False)
    return path


def _power_components(flows, *, stage: str):
    if stage == "DA":
        return {
            "Electric wind": flows["wind_nominal_electric_mw"].to_numpy(float),
            "Grid import": flows["grid_buy_da_mw"].to_numpy(float),
            "Base load": -flows["base_load_mw"].to_numpy(float),
            "Shore power": -flows["shore_power_nominal_mw"].to_numpy(float),
            "Quay cranes": -flows["quay_crane_power_nominal_mw"].to_numpy(float),
            "LOHC reactor": -flows["lohc_power_da_mw"].to_numpy(float),
            "AGV charging": -flows["agv_charge_power_da_mw"].to_numpy(float),
            "Grid export": -flows["grid_sell_da_mw"].to_numpy(float),
            "Wind spill": -flows["spill_da_mw"].to_numpy(float),
        }
    return {
        "Electric wind": flows["wind_worst_electric_mw"].to_numpy(float),
        "Grid import": flows["grid_buy_rt_mw"].to_numpy(float),
        "Base load": -flows["base_load_mw"].to_numpy(float),
        "Shore power": -flows["shore_power_worst_mw"].to_numpy(float),
        "Quay cranes": -flows["quay_crane_power_worst_mw"].to_numpy(float),
        "LOHC reactor": -flows["lohc_power_rt_mw"].to_numpy(float),
        "AGV charging": -flows["agv_charge_power_rt_mw"].to_numpy(float),
        "Grid export": -flows["grid_sell_rt_mw"].to_numpy(float),
        "Wind spill": -flows["spill_rt_mw"].to_numpy(float),
    }


def _signed_bars(ax, time, components):
    positive = np.zeros(len(time), dtype=float)
    negative = np.zeros(len(time), dtype=float)
    handles = []
    for label, values in components.items():
        values = np.asarray(values, dtype=float)
        bottom = np.where(values >= 0.0, positive, negative)
        bars = ax.bar(
            time,
            values,
            bottom=bottom,
            width=0.245,
            align="edge",
            color=POWER_COLORS[label],
            edgecolor="none",
            label=label,
        )
        handles.append(bars)
        positive += np.maximum(values, 0.0)
        negative += np.minimum(values, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.7)
    return handles, positive, negative


def build_figure(
    artifacts: CorePlotArtifacts, run: FormalRun, output_directory: str | Path
):
    output_root = Path(output_directory)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    plot_path = _write_plot_data(artifacts, data_root)
    flows = artifacts.flows
    states = artifacts.states
    time = flows["time_hour"].to_numpy(float)
    state_time = states["time_hour"].to_numpy(float)

    figure = new_figure(height=5.70)
    figure.subplots_adjust(
        left=0.075, right=0.925, bottom=0.075, top=0.855, wspace=0.38, hspace=0.42
    )
    outer = figure.add_gridspec(2, 2, hspace=0.16, wspace=0.14)
    ax_landing = figure.add_subplot(outer[0, 0])
    ax_da = figure.add_subplot(outer[0, 1])
    ax_rt = figure.add_subplot(outer[1, 0])
    chain = outer[1, 1].subgridspec(2, 1, hspace=0.12)
    ax_h2 = figure.add_subplot(chain[0, 0])
    ax_lohc = figure.add_subplot(chain[1, 0], sharex=ax_h2)

    step(
        ax_landing, time, flows["wind_nominal_mw"], color=COLORS["nominal"],
        linestyle="--", label="Wind forecast"
    )
    step(
        ax_landing, time, flows["wind_worst_mw"], color=COLORS["wind"],
        linewidth=1.35, label="Worst-case wind"
    )
    ax_landing.fill_between(
        time,
        0.0,
        flows["wind_fixed_hydrogen_mw"],
        step="post",
        color=COLORS["hydrogen"],
        alpha=0.22,
        label="Fixed H$_2$-side landing",
    )
    step(
        ax_landing, time, flows["wind_fixed_hydrogen_mw"],
        color=COLORS["hydrogen"], linewidth=1.15
    )
    step(
        ax_landing, time, flows["wind_worst_electric_mw"],
        color=COLORS["shore"], label="RT electricity-side landing"
    )
    ax_landing.set_ylabel("Power (MW)")
    format_time_axis(ax_landing)
    clean_axes(ax_landing)
    panel_label(ax_landing, "(a)")
    ax_landing.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.7)

    da_components = _power_components(flows, stage="DA")
    rt_components = _power_components(flows, stage="RT")
    handles, da_positive, da_negative = _signed_bars(ax_da, time, da_components)
    _, rt_positive, rt_negative = _signed_bars(ax_rt, time, rt_components)
    y_min = min(float(da_negative.min()), float(rt_negative.min())) * 1.08
    y_max = max(float(da_positive.max()), float(rt_positive.max())) * 1.08
    for ax in (ax_da, ax_rt):
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Power (MW)")
        format_time_axis(ax)
        clean_axes(ax)
    panel_label(ax_da, "(b)")
    panel_label(ax_rt, "(c)")
    ax_da.text(
        0.98, 0.94,
        rf"max $|r^{{DA}}|$={np.max(np.abs(flows['da_power_balance_residual_mw'])):.1e} MW",
        transform=ax_da.transAxes, ha="right", va="top", fontsize=7.2
    )
    ax_rt.text(
        0.98, 0.94,
        rf"max $|r^{{RT}}|$={np.max(np.abs(flows['rt_power_balance_residual_mw'])):.1e} MW",
        transform=ax_rt.transAxes, ha="right", va="top", fontsize=7.2
    )
    step(
        ax_rt, time, -flows["spill_da_mw"], color=COLORS["spill"],
        linestyle="--", linewidth=0.9, label="DA spill"
    )
    step(
        ax_rt, time, -flows["spill_rt_mw"], color="#4F8A10",
        linewidth=1.25, label="RT spill"
    )
    ax_rt.fill_between(
        time,
        -flows["spill_da_mw"],
        -flows["spill_rt_mw"],
        step="post",
        color=COLORS["spill"],
        alpha=0.15,
    )
    figure.legend(
        [item[0] for item in handles],
        list(da_components),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.3,
    )

    width = 0.12
    ax_h2.bar(
        time - width / 2,
        flows["h2_landing_rate_kg_per_h"],
        width=width,
        color=COLORS["hydrogen"],
        alpha=0.75,
        label="H$_2$ landing",
    )
    ax_h2.bar(
        time + width / 2,
        -flows["h2_consumption_rate_kg_per_h"],
        width=width,
        color=COLORS["lohc"],
        alpha=0.8,
        label="H$_2$ consumption",
    )
    ax_h2.axhline(0, color="black", linewidth=0.6)
    ax_h2_inv = ax_h2.twinx()
    step(
        ax_h2_inv, state_time, states["hydrogen_inventory_kg"],
        color="#006D6D", linewidth=1.25, label="H$_2$ inventory"
    )
    h2_terminal = float(run.resolved_case["hydrogen"]["terminal_min_kg"])
    ax_h2_inv.axhline(h2_terminal, color="#006D6D", linestyle=":", linewidth=0.9)
    ax_h2.set_ylabel("H$_2$ rate\n(kg/h)")
    ax_h2_inv.set_ylabel("Inventory (kg)", color="#006D6D")
    ax_h2_inv.tick_params(axis="y", colors="#006D6D")
    format_time_axis(ax_h2, show_xlabel=False)
    clean_axes(ax_h2)
    ax_h2_inv.spines["top"].set_visible(False)
    panel_label(ax_h2, "(d)")
    h1, l1 = ax_h2.get_legend_handles_labels()
    h2h, l2 = ax_h2_inv.get_legend_handles_labels()
    ax_h2.legend(h1 + h2h, l1 + l2, loc="upper right", frameon=False, ncol=3)

    ax_lohc.bar(
        time - width / 2,
        flows["lohc_production_rate_kg_per_h"],
        width=width,
        color=COLORS["lohc"],
        alpha=0.72,
        label="LOHC production",
    )
    ax_lohc.bar(
        time + width / 2,
        -flows["lohc_outbound_rate_kg_per_h"],
        width=width,
        color=COLORS["agv_container"],
        alpha=0.72,
        label="LOHC outbound",
    )
    ax_lohc.axhline(0, color="black", linewidth=0.6)
    ax_lohc_inv = ax_lohc.twinx()
    step(
        ax_lohc_inv, state_time, states["lohc_inventory_kg"],
        color="#8C4B00", linewidth=1.25, label="LOHC inventory"
    )
    lohc_terminal = float(run.resolved_case["lohc"]["terminal_min_kg"])
    ax_lohc_inv.axhline(lohc_terminal, color="#8C4B00", linestyle=":", linewidth=0.9)
    ax_lohc.set_ylabel("LOHC rate\n(kg/h)")
    ax_lohc_inv.set_ylabel("Inventory (kg)", color="#8C4B00")
    ax_lohc_inv.tick_params(axis="y", colors="#8C4B00")
    format_time_axis(ax_lohc)
    clean_axes(ax_lohc)
    ax_lohc_inv.spines["top"].set_visible(False)
    h1, l1 = ax_lohc.get_legend_handles_labels()
    h2h, l2 = ax_lohc_inv.get_legend_handles_labels()
    ax_lohc.legend(h1 + h2h, l1 + l2, loc="upper right", frameon=False, ncol=3)
    return figure, (plot_path,)
