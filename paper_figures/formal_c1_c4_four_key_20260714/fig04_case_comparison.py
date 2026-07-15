"""Fig. 4: mechanism ablation and strict robustness comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .comparisons import ComparisonArtifacts
from .data_loader import FormalRun
from .style import COLORS, clean_axes, format_time_axis, new_figure, panel_label, step


COST_COLORS = {
    "DA grid purchase": "#6C5B7B",
    "DA grid sale": "#9A7DAD",
    "DA spill": "#8BC34A",
    "DA LOHC operation/ramp": "#E6A04B",
    "RT grid deviation": "#8E6C9E",
    "RT spill": "#5FAE42",
    "RT spill deviation": "#B7D88B",
    "RT LOHC adjustment": "#D97706",
    "RT AGV charge adjustment": "#E3B341",
    "Backlog penalties": "#B65C5C",
    "LOHC outbound revenue": "#00A6A6",
}

COST_LABELS = {
    "DA grid purchase": "DA grid purchase",
    "DA grid sale": "DA grid sale revenue",
    "DA spill": "DA wind spill",
    "DA LOHC operation/ramp": "DA LOHC operation/ramp",
    "RT grid deviation": "RT grid deviation",
    "RT spill": "RT wind spill",
    "RT spill deviation": "RT spill deviation",
    "RT LOHC adjustment": "RT LOHC adjustment",
    "RT AGV charge adjustment": "RT AGV adjustment",
    "Backlog penalties": "Backlog penalties",
    "LOHC outbound revenue": "LOHC outbound revenue",
}


def _plot_cost_decomposition(ax, data):
    cases = ["C1", "C2", "C3", "C4"]
    x = np.arange(len(cases), dtype=float)
    positive = np.zeros(len(cases), dtype=float)
    negative = np.zeros(len(cases), dtype=float)
    handles = []
    labels = []
    for component in COST_COLORS:
        subset = data.loc[data["component"] == component].set_index("case")
        values = np.asarray([subset.at[case, "signed_cost"] for case in cases])
        bottom = np.where(values >= 0.0, positive, negative)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=COST_COLORS[component],
            edgecolor="white",
            linewidth=0.25,
        )
        positive += np.maximum(values, 0.0)
        negative += np.minimum(values, 0.0)
        handles.append(bars[0])
        labels.append(COST_LABELS[component])
    totals = (
        data[["case", "total_objective"]]
        .drop_duplicates()
        .set_index("case")
        .loc[cases, "total_objective"]
        .to_numpy(float)
    )
    ax.scatter(x, totals, marker="D", s=24, color="black", zorder=5, label="Total")
    ax.axhline(0.0, color="black", linewidth=0.75)
    ax.set_xticks(x, cases)
    ax.set_ylabel("Scaled objective value")
    clean_axes(ax)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    panel_label(ax, "(a)")
    return handles, labels


def _plot_architecture_metrics(ax, data):
    metrics = data["metric_label"].drop_duplicates().tolist()
    display = {
        "Grid import": "Grid import",
        "Grid export": "Grid export",
        "Wind spill": "Wind spill",
        "Hydrogen-side landing": "H$_2$ landing",
        "LOHC outbound": "LOHC outbound",
    }
    y = np.arange(len(metrics), dtype=float)
    width = 0.36
    for offset, case, color in (
        (-width / 2, "C1", COLORS["c1"]),
        (width / 2, "C2", COLORS["c2"]),
    ):
        subset = data.loc[data["case"] == case].set_index("metric_label")
        values = np.asarray(
            [subset.at[metric, "normalized_to_c1"] for metric in metrics]
        )
        bars = ax.barh(
            y + offset,
            values,
            height=width,
            color=color,
            alpha=0.85,
            label=case,
        )
        for bar, value in zip(bars, values, strict=True):
            if value > 3.0:
                ax.text(
                    value,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=7.0,
                )
    ax.axvline(1.0, color=COLORS["nominal"], linestyle="--", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.7, base=10)
    ax.set_xlim(-0.03, 700.0)
    ax.set_yticks(y, [display.get(metric, metric) for metric in metrics])
    ax.set_xlabel("Normalized indicator (symlog; C1 = 1)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    clean_axes(ax)
    ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    panel_label(ax, "(b)")


def _common_series(data, policy, series):
    return data.loc[
        (data["policy_case"] == policy) & (data["series"] == series)
    ].sort_values("period")


def _plot_common_scenario(ax_backlog, ax_agv, data, c3_run):
    for policy, color, linestyle in (
        ("C1", COLORS["c1"], "-"),
        ("C3", COLORS["c3"], "--"),
    ):
        series = _common_series(data, policy, "backlog_teu")
        step(
            ax_backlog,
            series["time_hour"],
            series["value"],
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=policy,
        )
    ax_backlog.set_ylabel("Backlog (TEU)")
    ax_backlog.legend(loc="upper left", frameon=False, ncol=2)
    format_time_axis(ax_backlog, show_xlabel=False)
    clean_axes(ax_backlog)
    panel_label(ax_backlog, "(c)")

    for series_name, color, label in (
        ("agv_container_count", COLORS["agv_container"], "Container AGVs"),
        ("agv_lohc_count", COLORS["lohc"], "LOHC AGVs"),
    ):
        for policy, linestyle in (("C1", "-"), ("C3", "--")):
            series = _common_series(data, policy, series_name)
            step(
                ax_agv,
                series["time_hour"],
                series["value"],
                color=color,
                linestyle=linestyle,
                linewidth=1.1,
                label="_nolegend_",
            )
    container_cap = float(c3_run.resolved_case["container_work_capacity"])
    lohc_cap = float(c3_run.resolved_case["lohc_work_capacity"])
    ax_agv.axhline(
        container_cap,
        color=COLORS["agv_container"],
        linestyle=":",
        linewidth=0.9,
        label="_nolegend_",
    )
    ax_agv.axhline(
        lohc_cap,
        color=COLORS["lohc"],
        linestyle=":",
        linewidth=0.9,
        label="_nolegend_",
    )
    ax_agv.set_ylabel("Equivalent AGVs")
    format_time_axis(ax_agv)
    clean_axes(ax_agv)
    ax_agv.text(
        0.02,
        0.94,
        "Blue: CTN; orange: LOHC\nSolid: C1; dashed: C3; dotted: C3 cap",
        transform=ax_agv.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )


def _plot_cross_evaluation(ax, data):
    policies = ["C1", "C4"]
    scenarios = ["C1", "C4"]
    matrix = np.asarray(
        [
            [
                data.loc[
                    (data["policy_case"] == policy)
                    & (data["scenario_case"] == scenario),
                    "total_cost",
                ].item()
                for scenario in scenarios
            ]
            for policy in policies
        ],
        dtype=float,
    )
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for row, policy in enumerate(policies):
        for column, scenario in enumerate(scenarios):
            record = data.loc[
                (data["policy_case"] == policy)
                & (data["scenario_case"] == scenario)
            ].iloc[0]
            text_color = "white" if record["total_cost"] > float(np.mean(matrix)) else "black"
            ax.text(
                column,
                row,
                f"Cost {record['total_cost']:.2f}\nPeak B {record['peak_backlog_teu']:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.4,
            )
    ax.set_xticks([0, 1], ["C1 worst\nrealization", "C4 worst\nrealization"])
    ax.set_yticks([0, 1], ["C1 robust\npolicy", "C4 deterministic\npolicy"])
    ax.set_xlabel("Certified scenario")
    ax.set_ylabel("Day-ahead policy")
    panel_label(ax, "(d)")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Total cost")


def build_figure(
    artifacts: ComparisonArtifacts,
    runs: Mapping[str, FormalRun],
    output_directory: str | Path,
):
    figure = new_figure(height=5.70)
    figure.subplots_adjust(
        left=0.08, right=0.94, bottom=0.08, top=0.78, wspace=0.42, hspace=0.48
    )
    outer = figure.add_gridspec(2, 2, hspace=0.18, wspace=0.18)
    ax_cost = figure.add_subplot(outer[0, 0])
    ax_metric = figure.add_subplot(outer[0, 1])
    common = outer[1, 0].subgridspec(2, 1, hspace=0.10)
    ax_backlog = figure.add_subplot(common[0, 0])
    ax_agv = figure.add_subplot(common[1, 0], sharex=ax_backlog)
    ax_cross = figure.add_subplot(outer[1, 1])

    handles, labels = _plot_cost_decomposition(ax_cost, artifacts.cost_components)
    _plot_architecture_metrics(ax_metric, artifacts.architecture_metrics)
    _plot_common_scenario(
        ax_backlog,
        ax_agv,
        artifacts.common_scenario_timeseries,
        runs["C3"],
    )
    _plot_cross_evaluation(ax_cross, artifacts.cross_evaluation)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.25,
    )
    return figure, tuple(artifacts.paths.values())
