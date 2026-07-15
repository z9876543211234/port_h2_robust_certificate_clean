"""Fig. 1: wind/vessel uncertainty and certified propagation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_data import CorePlotArtifacts, INPUT_ROOT
from .data_loader import FormalRun
from .style import COLORS, clean_axes, format_time_axis, new_figure, panel_label, step


PLOT_COLUMNS = [
    "wind_nominal_mw",
    "wind_lower_mw",
    "wind_upper_mw",
    "wind_worst_mw",
    "arrival_nominal_count",
    "arrival_worst_count",
    "in_port_nominal_count",
    "in_port_worst_count",
    "shore_power_nominal_mw",
    "shore_power_worst_mw",
    "quay_crane_power_nominal_mw",
    "quay_crane_power_worst_mw",
    "task_release_nominal_teu",
    "task_release_worst_teu",
]


def build_figure(
    artifacts: CorePlotArtifacts, run: FormalRun, output_directory: str | Path
):
    flows = artifacts.flows
    time = flows["time_hour"].to_numpy(dtype=float)
    output_root = Path(output_directory)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    plot_data = flows[["period", "time_hour", *PLOT_COLUMNS]].melt(
        id_vars=["period", "time_hour"], var_name="series", value_name="value"
    )
    plot_path = data_root / "fig01_plot_data.csv"
    delay_path = data_root / "fig01_delay_assignments.csv"
    plot_data.to_csv(plot_path, index=False)
    run.ship_delay_assignment.to_csv(delay_path, index=False)

    wind_source = json.loads(
        (INPUT_ROOT / "wind_large_port_96.json").read_text(encoding="utf-8")
    )
    ship_source = json.loads(
        (INPUT_ROOT / "ship_delay_large_port_96.json").read_text(encoding="utf-8")
    )
    figure = new_figure()
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.075, top=0.965, wspace=0.20, hspace=0.34
    )
    outer = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.35])
    ax_wind = figure.add_subplot(outer[0, 0])
    ax_arrival = figure.add_subplot(outer[0, 1])
    propagation = outer[1, :].subgridspec(3, 1, hspace=0.08)
    ax_ship = figure.add_subplot(propagation[0, 0])
    ax_power = figure.add_subplot(propagation[1, 0], sharex=ax_ship)
    ax_task = figure.add_subplot(propagation[2, 0], sharex=ax_ship)

    ax_wind.fill_between(
        time,
        flows["wind_lower_mw"],
        flows["wind_upper_mw"],
        step="post",
        color="#D8D8D8",
        alpha=0.72,
        label="Uncertainty interval",
    )
    step(
        ax_wind,
        time,
        flows["wind_nominal_mw"],
        color=COLORS["nominal"],
        linestyle="--",
        label="Forecast",
    )
    step(
        ax_wind,
        time,
        flows["wind_worst_mw"],
        color=COLORS["wind"],
        linewidth=1.45,
        label="Certified worst case",
    )
    down = flows["wind_worst_mw"] < flows["wind_nominal_mw"] - 1e-9
    up = flows["wind_worst_mw"] > flows["wind_nominal_mw"] + 1e-9
    ax_wind.scatter(
        time[down], flows.loc[down, "wind_worst_mw"], s=13, marker="v",
        color="#B65C5C", zorder=5, label="Downward activation"
    )
    if bool(np.any(up)):
        ax_wind.scatter(
            time[up], flows.loc[up, "wind_worst_mw"], s=13, marker="^",
            color="#2F6B9A", zorder=5, label="Upward activation"
        )
    ax_wind.text(
        0.97,
        0.95,
        rf"$\Gamma^w={int(wind_source['budget'])}$" + "\nCertified realization",
        transform=ax_wind.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
    )
    ax_wind.set_ylabel("Offshore wind power (MW)")
    format_time_axis(ax_wind)
    clean_axes(ax_wind)
    panel_label(ax_wind, "(a)")
    ax_wind.legend(loc="lower right", frameon=False, ncol=2, columnspacing=0.8)

    width = 0.18
    ax_arrival.bar(
        time - width / 2,
        flows["arrival_nominal_count"],
        width=width,
        facecolor="white",
        edgecolor=COLORS["nominal"],
        linewidth=0.8,
        label="Nominal arrivals",
    )
    ax_arrival.bar(
        time + width / 2,
        flows["arrival_worst_count"],
        width=width,
        color=COLORS["shore"],
        edgecolor=COLORS["shore"],
        label="Delayed arrivals",
    )
    delayed = run.ship_delay_assignment.loc[
        run.ship_delay_assignment["delay_steps"] > 0
    ]
    y_arrow = float(
        max(flows["arrival_nominal_count"].max(), flows["arrival_worst_count"].max())
        + 0.35
    )
    for index, row in delayed.iterrows():
        source_x = float(row["nominal_source_period"]) * 0.25
        target_x = float(row["actual_arrival_period"]) * 0.25
        ax_arrival.annotate(
            "",
            xy=(target_x, y_arrow + 0.12 * (index % 2)),
            xytext=(source_x, y_arrow + 0.12 * (index % 2)),
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["c3"],
                "linewidth": 0.8,
                "connectionstyle": "arc3,rad=-0.25",
            },
        )
    beyond = int(
        round(
            run.worst_scenario[
                "ship.delayed_out_of_current_day_count"
            ].dropna().sum()
        )
    )
    ax_arrival.text(
        0.98,
        0.95,
        f"Delayed beyond current day: {beyond}",
        transform=ax_arrival.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
    )
    ax_arrival.set_ylabel("Arriving vessels")
    ymax = max(y_arrow + 0.5, 1.5)
    ax_arrival.set_ylim(0, np.ceil(ymax))
    ax_arrival.set_yticks(np.arange(0, int(np.ceil(ymax)) + 1))
    format_time_axis(ax_arrival)
    clean_axes(ax_arrival)
    panel_label(ax_arrival, "(b)")
    ax_arrival.legend(loc="upper left", frameon=False)

    step(
        ax_ship, time, flows["in_port_nominal_count"], color=COLORS["nominal"],
        linestyle="--", label="Nominal"
    )
    step(
        ax_ship, time, flows["in_port_worst_count"], color=COLORS["shore"],
        label="Worst case"
    )
    ax_ship.set_ylabel("In-port\nvessels")
    ax_ship.set_yticks(
        np.arange(0, int(np.ceil(flows["in_port_worst_count"].max())) + 2, 4)
    )
    ax_ship.legend(loc="upper right", frameon=False, ncol=2)
    format_time_axis(ax_ship, show_xlabel=False)
    clean_axes(ax_ship)
    panel_label(ax_ship, "(c)", x=-0.06, y=1.02)

    shore = flows["shore_power_worst_mw"].to_numpy(dtype=float)
    quay = flows["quay_crane_power_worst_mw"].to_numpy(dtype=float)
    ax_power.fill_between(
        time, 0.0, shore, step="post", color=COLORS["shore"], alpha=0.75,
        label="Shore power"
    )
    ax_power.fill_between(
        time, shore, shore + quay, step="post", color=COLORS["quay"], alpha=0.78,
        label="Quay cranes"
    )
    nominal_total = (
        flows["shore_power_nominal_mw"] + flows["quay_crane_power_nominal_mw"]
    )
    step(
        ax_power, time, nominal_total, color=COLORS["nominal"], linestyle="--",
        label="Nominal total"
    )
    quay_capacity = (
        float(ship_source["installed_quay_cranes"])
        * float(ship_source["quay_crane_power_kw"])
        / 1000.0
    )
    ax_power.axhline(
        quay_capacity,
        color=COLORS["quay"],
        linestyle=":",
        linewidth=0.9,
        label="Installed QC power",
    )
    ax_power.set_ylabel("Port load\n(MW)")
    ax_power.legend(loc="upper right", frameon=False, ncol=4, columnspacing=0.8)
    format_time_axis(ax_power, show_xlabel=False)
    clean_axes(ax_power)

    step(
        ax_task, time, flows["task_release_nominal_teu"], color=COLORS["nominal"],
        linestyle="--", label="Nominal release"
    )
    step(
        ax_task, time, flows["task_release_worst_teu"], color=COLORS["agv_container"],
        label="Worst-case release"
    )
    ax_task.set_ylabel("Task release\n(TEU/period)")
    ax_task.legend(loc="upper right", frameon=False, ncol=2)
    format_time_axis(ax_task)
    clean_axes(ax_task)
    return figure, (plot_path, delay_path)
