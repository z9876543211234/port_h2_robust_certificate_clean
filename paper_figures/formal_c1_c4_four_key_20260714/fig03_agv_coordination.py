"""Fig. 3: shared-AGV energy--logistics coordination."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build_data import CorePlotArtifacts
from .data_loader import FormalRun
from .style import COLORS, clean_axes, format_time_axis, new_figure, panel_label, step


FLOW_SERIES = [
    "agv_container_count",
    "agv_lohc_count",
    "agv_charge_count",
    "agv_idle_count",
    "task_release_rate_teu_per_h",
    "task_done_rate_teu_per_h",
    "lohc_outbound_rate_kg_per_h",
    "agv_charge_power_da_mw",
    "agv_charge_power_rt_mw",
]
STATE_SERIES = ["backlog_teu", "soc"]


def _write_plot_data(artifacts: CorePlotArtifacts, data_root: Path) -> Path:
    flow_long = artifacts.flows[["period", "time_hour", *FLOW_SERIES]].melt(
        id_vars=["period", "time_hour"], var_name="series", value_name="value"
    )
    flow_long.insert(0, "trajectory_type", "flow")
    state_long = artifacts.states[["period", "time_hour", *STATE_SERIES]].melt(
        id_vars=["period", "time_hour"], var_name="series", value_name="value"
    )
    state_long.insert(0, "trajectory_type", "state")
    path = data_root / "fig03_plot_data.csv"
    pd.concat([flow_long, state_long], ignore_index=True).to_csv(path, index=False)
    return path


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
    fleet = float(run.resolved_case["agv"]["fleet_size"])

    figure = new_figure()
    figure.subplots_adjust(
        left=0.075, right=0.925, bottom=0.075, top=0.965, wspace=0.36, hspace=0.38
    )
    outer = figure.add_gridspec(2, 2, hspace=0.17, wspace=0.16)
    ax_alloc = figure.add_subplot(outer[0, 0])
    output_grid = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    ax_task = figure.add_subplot(output_grid[0, 0])
    ax_out = figure.add_subplot(output_grid[1, 0], sharex=ax_task)
    ax_backlog = figure.add_subplot(outer[1, 0])
    ax_charge = figure.add_subplot(outer[1, 1])

    values = [
        flows["agv_container_count"].to_numpy(float),
        flows["agv_lohc_count"].to_numpy(float),
        flows["agv_charge_count"].to_numpy(float),
        flows["agv_idle_count"].to_numpy(float),
    ]
    ax_alloc.stackplot(
        time,
        *values,
        step="post",
        colors=[
            COLORS["agv_container"],
            COLORS["lohc"],
            COLORS["agv_charge"],
            COLORS["agv_idle"],
        ],
        alpha=0.82,
        labels=["Container service", "LOHC transport", "Charging", "Idle"],
    )
    ax_alloc.axhline(
        fleet, color="black", linewidth=0.9, linestyle="--", label="Fleet size"
    )
    ax_alloc.set_ylim(0, fleet * 1.08)
    ax_alloc.set_ylabel("Equivalent number of AGVs")
    format_time_axis(ax_alloc)
    clean_axes(ax_alloc)
    panel_label(ax_alloc, "(a)")
    ax_alloc.legend(loc="lower center", frameon=False, ncol=3, columnspacing=0.8)

    step(
        ax_task,
        time,
        flows["task_release_rate_teu_per_h"],
        color=COLORS["shore"],
        linestyle="--",
        label="Task release",
    )
    step(
        ax_task,
        time,
        flows["task_done_rate_teu_per_h"],
        color=COLORS["agv_container"],
        label="Task completed",
    )
    ax_task.set_ylabel("Container rate\n(TEU/h)")
    format_time_axis(ax_task, show_xlabel=False)
    clean_axes(ax_task)
    panel_label(ax_task, "(b)")
    ax_task.legend(loc="upper right", frameon=False, ncol=2)

    ax_out.bar(
        time,
        flows["lohc_outbound_rate_kg_per_h"],
        width=0.245,
        align="edge",
        color=COLORS["lohc"],
        alpha=0.72,
        label="LOHC outbound",
    )
    ax_out_agv = ax_out.twinx()
    step(
        ax_out_agv,
        time,
        flows["agv_lohc_count"],
        color="#8C4B00",
        linewidth=1.1,
        label="LOHC-serving AGVs",
    )
    ax_out.set_ylabel("LOHC rate\n(kg/h)")
    ax_out_agv.set_ylabel("AGVs", color="#8C4B00")
    ax_out_agv.tick_params(axis="y", colors="#8C4B00")
    format_time_axis(ax_out)
    clean_axes(ax_out)
    ax_out_agv.spines["top"].set_visible(False)
    h1, l1 = ax_out.get_legend_handles_labels()
    h2, l2 = ax_out_agv.get_legend_handles_labels()
    ax_out.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, ncol=2)

    backlog = states["backlog_teu"].to_numpy(float)
    step(
        ax_backlog,
        state_time,
        backlog,
        color=COLORS["agv_container"],
        linewidth=1.35,
        label="Backlog",
    )
    ax_backlog.axhline(0.0, color=COLORS["nominal"], linestyle="--", linewidth=0.8)
    peak_index = int(np.argmax(backlog))
    ax_backlog.scatter(
        [state_time[peak_index]], [backlog[peak_index]], color=COLORS["c2"], s=17,
        zorder=4, label="Peak"
    )
    ax_backlog.scatter(
        [state_time[-1]], [backlog[-1]], color="black", marker="D", s=16,
        zorder=4, label="Terminal"
    )
    ax_backlog.text(
        0.03,
        0.80,
        f"Peak backlog = {backlog.max():.1f} TEU\nTerminal backlog = {backlog[-1]:.1f} TEU",
        transform=ax_backlog.transAxes,
        va="top",
        fontsize=7.3,
    )
    if backlog.max() <= 1e-10:
        ax_backlog.set_ylim(-0.05, 1.0)
    ax_backlog.set_ylabel("Backlog (TEU)")
    format_time_axis(ax_backlog)
    clean_axes(ax_backlog)
    panel_label(ax_backlog, "(c)")
    ax_backlog.legend(loc="upper right", frameon=False, ncol=3)

    ax_charge.bar(
        time,
        flows["agv_charge_power_rt_mw"],
        width=0.245,
        align="edge",
        color=COLORS["agv_charge"],
        alpha=0.75,
        label="RT charging power",
    )
    step(
        ax_charge,
        time,
        flows["agv_charge_power_da_mw"],
        color="#9A7200",
        linestyle="--",
        linewidth=1.0,
        label="DA charging power",
    )
    ax_soc = ax_charge.twinx()
    step(
        ax_soc,
        state_time,
        states["soc"],
        color=COLORS["shore"],
        linewidth=1.35,
        label="Fleet-average SOC",
    )
    min_soc = float(run.resolved_case["agv"]["min_soc"])
    terminal_min = float(run.resolved_case["agv"]["terminal_min_soc"])
    ax_soc.axhline(min_soc, color=COLORS["c2"], linestyle=":", linewidth=0.85)
    ax_soc.axhline(terminal_min, color=COLORS["shore"], linestyle="--", linewidth=0.85)
    ax_soc.scatter(
        [state_time[-1]], [states["soc"].iloc[-1]], color="black", marker="D", s=16,
        zorder=4
    )
    ax_charge.set_ylabel("Charging power (MW)")
    ax_soc.set_ylabel("SOC (p.u.)", color=COLORS["shore"])
    ax_soc.tick_params(axis="y", colors=COLORS["shore"])
    ax_soc.set_ylim(min(0.15, min_soc - 0.02), 0.93)
    format_time_axis(ax_charge)
    clean_axes(ax_charge)
    ax_soc.spines["top"].set_visible(False)
    panel_label(ax_charge, "(d)")
    h1, l1 = ax_charge.get_legend_handles_labels()
    h2, l2 = ax_soc.get_legend_handles_labels()
    ax_charge.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, ncol=2)
    return figure, (plot_path,)
