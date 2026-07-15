"""Shared IEEE-style visual language for the formal paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


FIGURE_WIDTH_IN = 7.16  # 181.9 mm
FIGURE_HEIGHT_IN = 5.55  # 141.0 mm

COLORS = {
    "wind": "#2E8B57",
    "wind_light": "#A8D5BA",
    "grid": "#6C5B7B",
    "base": "#7A7A7A",
    "shore": "#2F6B9A",
    "quay": "#56B4E9",
    "lohc": "#D97706",
    "hydrogen": "#00A6A6",
    "agv_charge": "#E3B341",
    "agv_container": "#4472C4",
    "agv_idle": "#B8B8B8",
    "spill": "#8BC34A",
    "nominal": "#4A4A4A",
    "c1": "#2E8B57",
    "c2": "#B65C5C",
    "c3": "#D97706",
    "c4": "#6C5B7B",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.15,
            "patch.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def new_figure(*, height: float = FIGURE_HEIGHT_IN):
    apply_publication_style()
    return plt.figure(
        figsize=(FIGURE_WIDTH_IN, height),
        constrained_layout=False,
        facecolor="white",
    )


def panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def format_time_axis(ax, *, show_xlabel: bool = True) -> None:
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    if show_xlabel:
        ax.set_xlabel("Time (h)")
    else:
        ax.tick_params(labelbottom=False)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def step(ax, x, y, **kwargs):
    return ax.step(x, y, where="post", **kwargs)
