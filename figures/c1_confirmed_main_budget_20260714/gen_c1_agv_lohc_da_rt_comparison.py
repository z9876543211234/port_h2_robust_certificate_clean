"""Compare AGV and LOHC day-ahead schedules with worst-case intraday recourse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import (
    COLORS,
    FIGURE_DIRECTORY,
    format_time_axis,
    panel_label,
    save_figure,
)


data = pd.read_csv(FIGURE_DIRECTORY / "c1_day_ahead_plot_data.csv")
time = data["time_hour"]

figure, axes = plt.subplots(2, 1, figsize=(7.15, 4.5), sharex=True)

axis = axes[0]
agv_day_ahead = data["agv_charge_power_da_mw"].to_numpy()
adjustment_band = axis.fill_between(
    time,
    np.maximum(agv_day_ahead - 0.9, 0.0),
    np.minimum(agv_day_ahead + 0.9, 2.7),
    step="post",
    color=COLORS["sky"],
    alpha=0.18,
    linewidth=0.0,
)
day_ahead_line = axis.step(
    time,
    agv_day_ahead,
    where="post",
    color=COLORS["blue"],
    linewidth=1.65,
)[0]
intraday_line = axis.step(
    time,
    data["agv_charge_power_rt_worst_mw"],
    where="post",
    color=COLORS["red"],
    linestyle="--",
    linewidth=1.65,
)[0]
axis.set_ylabel("AGV charging power (MW)")
axis.set_ylim(-0.05, 2.9)
panel_label(axis, "(a)")

axis = axes[1]
lohc_day_ahead = data["lohc_power_da_mw"].to_numpy()
axis.fill_between(
    time,
    np.maximum(lohc_day_ahead - 1.2, 0.2),
    np.minimum(lohc_day_ahead + 1.2, 2.5),
    step="post",
    color=COLORS["sky"],
    alpha=0.18,
    linewidth=0.0,
)
axis.step(
    time,
    lohc_day_ahead,
    where="post",
    color=COLORS["blue"],
    linewidth=1.65,
)
axis.step(
    time,
    data["lohc_power_rt_worst_mw"],
    where="post",
    color=COLORS["red"],
    linestyle="--",
    linewidth=1.65,
)
axis.set_ylabel("LOHC reactor power (MW)")
axis.set_ylim(0.1, 2.65)
panel_label(axis, "(b)")

for axis in axes:
    format_time_axis(axis)
axes[0].set_xlabel("")
figure.legend(
    [day_ahead_line, intraday_line, adjustment_band],
    ["Day-ahead", "Intraday (worst case)", "Feasible adjustment band"],
    ncol=3,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    columnspacing=1.5,
)
figure.subplots_adjust(top=0.90, hspace=0.22)
save_figure(figure, "c1_agv_lohc_day_ahead_intraday_comparison")
plt.close(figure)

