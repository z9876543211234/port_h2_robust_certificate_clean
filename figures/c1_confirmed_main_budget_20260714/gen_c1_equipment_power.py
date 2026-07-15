"""Generate day-ahead versus worst-case real-time equipment power curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIGURE_DIRECTORY, format_time_axis, panel_label, save_figure


data = pd.read_csv(FIGURE_DIRECTORY / "c1_day_ahead_plot_data.csv")
time = data["time_hour"]

figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.4), sharex=True)

axis = axes[0, 0]
axis.step(time, data["grid_net_da_mw"], where="post", color=COLORS["blue"], label="Day-ahead")
axis.step(time, data["grid_net_rt_worst_mw"], where="post", color=COLORS["red"], linestyle="--", label="Worst-case RT")
axis.axhline(0.0, color="#555555", linewidth=0.65)
axis.set_ylabel("Grid exchange (MW)")
axis.legend(
    ncol=2,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.56, 1.14),
)
panel_label(axis, "(a)")

axis = axes[0, 1]
agv_da = data["agv_charge_power_da_mw"].to_numpy()
axis.fill_between(
    time,
    np.maximum(agv_da - 0.9, 0.0),
    np.minimum(agv_da + 0.9, 2.7),
    step="post",
    color=COLORS["blue"],
    alpha=0.12,
    linewidth=0.0,
    label=r"$\pm0.9$ MW band",
)
axis.step(time, agv_da, where="post", color=COLORS["blue"], label="Day-ahead")
axis.step(time, data["agv_charge_power_rt_worst_mw"], where="post", color=COLORS["red"], linestyle="--", label="Worst-case RT")
axis.set_ylabel("AGV charging (MW)")
axis.legend(
    ncol=3,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.56, 1.14),
    columnspacing=0.8,
)
panel_label(axis, "(b)")

axis = axes[1, 0]
lohc_da = data["lohc_power_da_mw"].to_numpy()
axis.fill_between(
    time,
    np.maximum(lohc_da - 1.2, 0.2),
    np.minimum(lohc_da + 1.2, 2.5),
    step="post",
    color=COLORS["blue"],
    alpha=0.12,
    linewidth=0.0,
    label=r"$\pm1.2$ MW band",
)
axis.step(time, lohc_da, where="post", color=COLORS["blue"], label="Day-ahead")
axis.step(time, data["lohc_power_rt_worst_mw"], where="post", color=COLORS["red"], linestyle="--", label="Worst-case RT")
axis.set_ylabel("LOHC reactor (MW)")
axis.legend(
    ncol=3,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.56, 1.14),
    columnspacing=0.8,
)
panel_label(axis, "(c)")

axis = axes[1, 1]
axis.step(time, data["wind_nominal_mw"], where="post", color=COLORS["green"], label="Nominal wind")
axis.step(time, data["wind_worst_mw"], where="post", color=COLORS["red"], linestyle="--", label="Worst-case wind")
axis.step(time, data["wind_hydrogen_da_mw"], where="post", color=COLORS["purple"], linestyle=":", label="DA wind-to-H$_2$")
axis.set_ylabel("Wind power (MW)")
axis.legend(
    ncol=3,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.56, 1.14),
    columnspacing=0.8,
)
panel_label(axis, "(d)")

for axis in axes.flat:
    format_time_axis(axis)
figure.subplots_adjust(top=0.88, wspace=0.24, hspace=0.62)
save_figure(figure, "c1_equipment_power_da_vs_worst_rt")
plt.close(figure)
