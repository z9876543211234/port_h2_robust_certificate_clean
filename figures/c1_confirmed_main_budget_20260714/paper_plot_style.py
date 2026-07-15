"""Shared publication style for the corrected C1 dispatch figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_DIRECTORY = Path(__file__).resolve().parent
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#7A7A7A",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.35,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def format_time_axis(axis) -> None:
    axis.set_xlim(0.0, 24.0)
    axis.set_xticks(range(0, 25, 4))
    axis.set_xlabel("Time (h)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)


def panel_label(axis, label: str) -> None:
    axis.text(
        0.012,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def save_figure(figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURE_DIRECTORY / f"{stem}.{suffix}")

