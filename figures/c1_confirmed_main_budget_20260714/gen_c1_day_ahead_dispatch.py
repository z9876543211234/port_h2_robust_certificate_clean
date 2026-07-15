"""Generate the corrected C1 day-ahead dispatch and power-balance curves."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, FIGURE_DIRECTORY, format_time_axis, panel_label, save_figure


data = pd.read_csv(FIGURE_DIRECTORY / "c1_day_ahead_plot_data.csv")
time = data["time_hour"]

figure, axes = plt.subplots(2, 1, figsize=(7.15, 5.0), sharex=True)

axis = axes[0]
axis.step(
    time,
    data["day_ahead_demand_mw"],
    where="post",
    color=COLORS["black"],
    linewidth=2.2,
    label="Total demand",
)
axis.step(
    time,
    data["wind_direct_da_mw"],
    where="post",
    color=COLORS["green"],
    label="Direct wind supply",
)
axis.step(
    time,
    data["grid_net_da_mw"],
    where="post",
    color=COLORS["blue"],
    label="Net grid exchange",
)
axis.step(
    time,
    data["day_ahead_supply_mw"],
    where="post",
    color=COLORS["orange"],
    linestyle="--",
    linewidth=1.45,
    label="Total supply",
)
axis.axhline(0.0, color="#555555", linewidth=0.65)
axis.set_ylabel("Power (MW)")
axis.legend(
    ncol=4,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.005),
    columnspacing=1.1,
)
panel_label(axis, "(a)")

axis = axes[1]
components = (
    ("shore_power_nominal_mw", "Shore power", COLORS["red"]),
    ("quay_crane_power_nominal_mw", "Quay cranes", COLORS["orange"]),
    ("base_load_mw", "Base load", COLORS["gray"]),
    ("agv_charge_power_da_mw", "AGV charging", COLORS["purple"]),
    ("lohc_power_da_mw", "LOHC reactor", COLORS["sky"]),
)
for column, label, color in components:
    axis.step(time, data[column], where="post", color=color, label=label)
axis.set_ylabel("Power (MW)")
axis.legend(
    ncol=5,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.005),
    columnspacing=1.05,
)
panel_label(axis, "(b)")

for axis in axes:
    format_time_axis(axis)
figure.subplots_adjust(top=0.92, hspace=0.34)
save_figure(figure, "c1_day_ahead_dispatch")
plt.close(figure)
