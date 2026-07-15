#!/usr/bin/env python3
"""Build a three-panel day-ahead/intra-day operation figure for formal C1.

The script reads only the authoritative corrected four-key C1 artifacts.  It
does not re-solve or alter the optimization model.  Power and simultaneous AGV
counts are averaged over each hour; interval LOHC outbound quantities are
summed over each hour; inventories are sampled at hourly state boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SOURCE_RUN = (
    PROJECT_ROOT
    / "experiments"
    / "c1_confirmed_main_budget_20260714"
    / "C1_ProposedRobustMain_refined_partition"
)
DATA_DIR = OUTPUT_ROOT / "data"
FIGURE_DIR = OUTPUT_ROOT / "figures"
MPLCONFIG_DIR = OUTPUT_ROOT / ".mplconfig"

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "mint": "#B7DCCE",
    "teal": "#4D969B",
    "blue": "#2F8EA3",
    "orange": "#E79B52",
    "coral": "#E77747",
    "ink": "#4C4C4C",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.15,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    day_ahead_path = SOURCE_RUN / "dispatch_day_ahead.csv"
    recourse_path = SOURCE_RUN / "dispatch_recourse_worst.csv"
    certificate_path = SOURCE_RUN / "certificate.json"
    resolved_case_path = SOURCE_RUN / "resolved_case.json"

    for path in (
        day_ahead_path,
        recourse_path,
        certificate_path,
        resolved_case_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required formal artifact not found: {path}")

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    resolved_case = json.loads(resolved_case_path.read_text(encoding="utf-8"))

    required_statuses = {
        "master_status": "OPTIMAL",
        "phase1_adversary_status": "OPTIMAL",
        "cost_adversary_status": "OPTIMAL",
    }
    for key, expected in required_statuses.items():
        if certificate.get(key) != expected:
            raise ValueError(
                f"formal certificate rejected: {key}={certificate.get(key)!r}, "
                f"expected {expected!r}"
            )
    if certificate.get("accepted_limited_solve") is not False:
        raise ValueError("formal certificate rejected: limited solve was accepted")
    if certificate.get("time_limit_used_for_certificate") is not False:
        raise ValueError("formal certificate rejected: a solver time limit was used")
    if certificate.get("residual_check_passed") is not True:
        raise ValueError("formal certificate rejected: residual check did not pass")

    horizon = int(resolved_case["profile"]["periods"])
    dt_hours = float(resolved_case["profile"]["dt_hours"])
    if horizon != 96 or not np.isclose(dt_hours, 0.25):
        raise ValueError(
            f"this figure requires the formal 96-period, 0.25-h profile; "
            f"received horizon={horizon}, dt={dt_hours}"
        )

    day_ahead = pd.read_csv(day_ahead_path).sort_values("period").reset_index(drop=True)
    recourse_all = pd.read_csv(recourse_path).sort_values("period").reset_index(drop=True)
    recourse = recourse_all.loc[recourse_all["period"] < horizon].copy()
    states = recourse_all.loc[recourse_all["period"] <= horizon].copy()

    da_columns = [
        "period",
        "agv_charge_count_da",
        "grid_buy_da_mw",
        "grid_sell_da_mw",
        "lohc_power_da_mw",
    ]
    rt_columns = [
        "period",
        "agv_charge_count",
        "grid_buy_rt_mw",
        "grid_sell_rt_mw",
        "lohc_outbound_kg",
        "lohc_power_rt_mw",
    ]
    state_columns = ["period", "hydrogen_inventory_kg", "lohc_inventory_kg"]
    _require_columns(day_ahead, da_columns, day_ahead_path)
    _require_columns(recourse, rt_columns, recourse_path)
    _require_columns(states, state_columns, recourse_path)

    expected_flows = np.arange(horizon)
    expected_states = np.arange(horizon + 1)
    if len(day_ahead) != horizon or not np.array_equal(
        day_ahead["period"].to_numpy(int), expected_flows
    ):
        raise ValueError("day-ahead dispatch must contain periods 0,...,95 exactly once")
    if len(recourse) != horizon or not np.array_equal(
        recourse["period"].to_numpy(int), expected_flows
    ):
        raise ValueError("worst-case recourse must contain flow periods 0,...,95")
    if len(states) != horizon + 1 or not np.array_equal(
        states["period"].to_numpy(int), expected_states
    ):
        raise ValueError("worst-case recourse must contain state periods 0,...,96")

    required_numeric = da_columns[1:] + rt_columns[1:] + state_columns[1:]
    combined = pd.concat(
        [
            day_ahead[da_columns[1:]].reset_index(drop=True),
            recourse[rt_columns[1:]].reset_index(drop=True),
            states[state_columns[1:]].iloc[:horizon].reset_index(drop=True),
        ],
        axis=1,
    )
    if not np.isfinite(combined.to_numpy(float)).all():
        raise ValueError(f"non-finite values found in required plotting columns: {required_numeric}")

    fleet_size = float(resolved_case["agv"]["fleet_size"])
    charger_count = float(resolved_case["agv"]["charger_count"])
    if (day_ahead["agv_charge_count_da"] < -1e-8).any() or (
        day_ahead["agv_charge_count_da"] > charger_count + 1e-8
    ).any():
        raise ValueError("day-ahead charging count violates the archived charger limit")
    if (recourse["agv_charge_count"] < -1e-8).any() or (
        recourse["agv_charge_count"] > min(fleet_size, charger_count) + 1e-8
    ).any():
        raise ValueError("recourse charging count violates the archived fleet/charger limit")

    return day_ahead, recourse_all, certificate, resolved_case


def _build_hourly_tables(
    day_ahead: pd.DataFrame,
    recourse_all: pd.DataFrame,
    resolved_case: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = int(resolved_case["profile"]["periods"])
    dt_hours = float(resolved_case["profile"]["dt_hours"])
    periods_per_hour = int(round(1.0 / dt_hours))
    if periods_per_hour * 24 != horizon:
        raise ValueError("hourly aggregation requires exactly 24 equal-size hourly groups")

    recourse = recourse_all.loc[recourse_all["period"] < horizon].copy()
    flow = pd.DataFrame(
        {
            "period": np.arange(horizon, dtype=int),
            "hour": np.arange(horizon, dtype=int) // periods_per_hour,
            "net_grid_da_mw": day_ahead["grid_buy_da_mw"].to_numpy(float)
            - day_ahead["grid_sell_da_mw"].to_numpy(float),
            "net_grid_rt_mw": recourse["grid_buy_rt_mw"].to_numpy(float)
            - recourse["grid_sell_rt_mw"].to_numpy(float),
            "agv_charge_count_da": day_ahead["agv_charge_count_da"].to_numpy(float),
            "agv_charge_count_rt": recourse["agv_charge_count"].to_numpy(float),
            "lohc_power_da_mw": day_ahead["lohc_power_da_mw"].to_numpy(float),
            "lohc_power_rt_mw": recourse["lohc_power_rt_mw"].to_numpy(float),
            "lohc_outbound_kg": recourse["lohc_outbound_kg"].to_numpy(float),
        }
    )
    hourly_mean_columns = [
        "net_grid_da_mw",
        "net_grid_rt_mw",
        "agv_charge_count_da",
        "agv_charge_count_rt",
        "lohc_power_da_mw",
        "lohc_power_rt_mw",
    ]
    hourly = flow.groupby("hour", as_index=False)[hourly_mean_columns].mean()
    outbound = flow.groupby("hour", as_index=False)["lohc_outbound_kg"].sum()
    hourly = hourly.merge(outbound, on="hour", validate="one_to_one")
    hourly["hour_center"] = hourly["hour"] + 0.5
    hourly["grid_adjustment_mw"] = (
        hourly["net_grid_rt_mw"] - hourly["net_grid_da_mw"]
    )

    state_rows = np.arange(0, horizon + 1, periods_per_hour, dtype=int)
    states = recourse_all.set_index("period").loc[state_rows].reset_index()
    hourly_states = pd.DataFrame(
        {
            "hour_boundary": np.arange(25, dtype=int),
            "hydrogen_inventory_kg": states["hydrogen_inventory_kg"].to_numpy(float),
            "lohc_inventory_kg": states["lohc_inventory_kg"].to_numpy(float),
        }
    )
    if len(hourly) != 24 or len(hourly_states) != 25:
        raise AssertionError("unexpected hourly flow/state table length")
    return hourly, hourly_states


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="both", color="0.89", linewidth=0.5)
    axis.set_axisbelow(True)


def _plot(hourly: pd.DataFrame, states: pd.DataFrame) -> plt.Figure:
    x = hourly["hour_center"].to_numpy(float)
    state_x = states["hour_boundary"].to_numpy(float)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.16, 5.05),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.07, 1.0]},
    )

    # (a) Day-ahead versus worst-case intra-day grid exchange.
    axis = axes[0]
    axis.axhline(0.0, color="0.35", linewidth=0.65)
    axis.bar(
        x,
        hourly["grid_adjustment_mw"],
        width=0.66,
        color=COLORS["mint"],
        edgecolor=COLORS["teal"],
        linewidth=0.35,
        label="RT-DA adjustment",
        zorder=1,
    )
    axis.plot(
        x,
        hourly["net_grid_da_mw"],
        color=COLORS["ink"],
        linestyle="--",
        marker="o",
        markersize=2.4,
        label="Day-ahead grid",
        zorder=3,
    )
    axis.plot(
        x,
        hourly["net_grid_rt_mw"],
        color=COLORS["blue"],
        marker="s",
        markersize=2.4,
        linewidth=1.25,
        label="Worst-case intra-day grid",
        zorder=3,
    )
    axis.set_ylabel("Net grid exchange (MW)")
    axis.text(0.01, 0.93, "(a)", transform=axis.transAxes, va="top", fontsize=9.0)
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_labels = [
        "Day-ahead grid",
        "Worst-case intra-day grid",
        "RT-DA adjustment",
    ]
    axis.legend(
        [by_label[label] for label in legend_labels],
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        columnspacing=1.1,
    )
    panel_a_values = np.concatenate(
        [
            hourly["net_grid_da_mw"].to_numpy(float),
            hourly["net_grid_rt_mw"].to_numpy(float),
            hourly["grid_adjustment_mw"].to_numpy(float),
        ]
    )
    panel_a_min = min(0.0, float(panel_a_values.min()))
    panel_a_max = max(0.0, float(panel_a_values.max()))
    panel_a_span = max(panel_a_max - panel_a_min, 1.0)
    axis.set_ylim(panel_a_min - 0.06 * panel_a_span, panel_a_max + 0.19 * panel_a_span)
    _style_axis(axis)

    # (b) The two DA/RT pairs that are actual variables in the formal model.
    axis = axes[1]
    reactor_axis = axis.twinx()
    axis.plot(
        x,
        hourly["agv_charge_count_da"],
        color=COLORS["ink"],
        linestyle="--",
        marker="o",
        markersize=2.4,
        label="DA charging AGVs",
    )
    axis.plot(
        x,
        hourly["agv_charge_count_rt"],
        color=COLORS["blue"],
        marker="s",
        markersize=2.4,
        linewidth=1.25,
        label="RT charging AGVs",
    )
    reactor_axis.plot(
        x,
        hourly["lohc_power_da_mw"],
        color=COLORS["orange"],
        linestyle="--",
        marker="^",
        markersize=2.5,
        label="DA LOHC reactor",
    )
    reactor_axis.plot(
        x,
        hourly["lohc_power_rt_mw"],
        color=COLORS["coral"],
        marker="D",
        markersize=2.2,
        linewidth=1.2,
        label="RT LOHC reactor",
    )
    axis.set_ylabel("Charging AGVs")
    reactor_axis.set_ylabel("LOHC reactor power (MW)")
    charge_max = max(
        float(hourly["agv_charge_count_da"].max()),
        float(hourly["agv_charge_count_rt"].max()),
    )
    reactor_max = max(
        float(hourly["lohc_power_da_mw"].max()),
        float(hourly["lohc_power_rt_mw"].max()),
    )
    axis.set_ylim(0.0, max(1.0, charge_max * 1.18))
    reactor_axis.set_ylim(0.0, max(0.1, reactor_max * 1.18))
    axis.text(0.01, 0.93, "(b)", transform=axis.transAxes, va="top", fontsize=9.0)
    h1, l1 = axis.get_legend_handles_labels()
    h2, l2 = reactor_axis.get_legend_handles_labels()
    axis.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=0.85,
    )
    _style_axis(axis)
    reactor_axis.spines["top"].set_visible(False)
    reactor_axis.grid(False)

    # (c) Real-time LOHC outbound quantity and the two hydrogen-chain states.
    axis = axes[2]
    inventory_axis = axis.twinx()
    axis.bar(
        x,
        hourly["lohc_outbound_kg"],
        width=0.66,
        color="#E9B05D",
        edgecolor=COLORS["orange"],
        linewidth=0.35,
        label="Hourly LOHC outbound",
    )
    inventory_axis.plot(
        state_x,
        states["hydrogen_inventory_kg"],
        color=COLORS["blue"],
        marker="o",
        markersize=2.4,
        label="H$_2$ inventory",
    )
    inventory_axis.plot(
        state_x,
        states["lohc_inventory_kg"],
        color=COLORS["coral"],
        marker="s",
        markersize=2.4,
        label="LOHC inventory",
    )
    axis.set_ylabel("LOHC outbound\n(kg-H$_2$-eq.)")
    inventory_axis.set_ylabel("Inventory (kg-H$_2$-eq.)")
    outbound_max = float(hourly["lohc_outbound_kg"].max())
    inventory_max = max(
        float(states["hydrogen_inventory_kg"].max()),
        float(states["lohc_inventory_kg"].max()),
    )
    axis.set_ylim(0.0, max(1.0, outbound_max * 1.15))
    inventory_axis.set_ylim(0.0, max(1.0, inventory_max * 1.14))
    axis.text(0.01, 0.93, "(c)", transform=axis.transAxes, va="top", fontsize=9.0)
    h1, l1 = axis.get_legend_handles_labels()
    h2, l2 = inventory_axis.get_legend_handles_labels()
    axis.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.05),
        columnspacing=1.1,
    )
    _style_axis(axis)
    inventory_axis.spines["top"].set_visible(False)
    inventory_axis.grid(False)

    for axis in axes:
        axis.set_xlim(0.0, 24.0)
        axis.set_xticks([0, 6, 12, 18, 24])
    axes[-1].set_xlabel("Time (h)")
    return figure


def _write_manifest(certificate: dict, resolved_case: dict) -> None:
    source_files = [
        SOURCE_RUN / "dispatch_day_ahead.csv",
        SOURCE_RUN / "dispatch_recourse_worst.csv",
        SOURCE_RUN / "certificate.json",
        SOURCE_RUN / "resolved_case.json",
    ]
    manifest = {
        "figure": "fig_c1_day_intraday_operation",
        "source_run": _project_relative(SOURCE_RUN),
        "case_name": certificate["case_name"],
        "certificate_status": certificate["status"],
        "joint_bundle_sha256": certificate["joint_bundle_sha256"],
        "master_status": certificate["master_status"],
        "phase1_adversary_status": certificate["phase1_adversary_status"],
        "cost_adversary_status": certificate["cost_adversary_status"],
        "accepted_limited_solve": certificate["accepted_limited_solve"],
        "time_limit_used_for_certificate": certificate[
            "time_limit_used_for_certificate"
        ],
        "periods": resolved_case["profile"]["periods"],
        "dt_hours": resolved_case["profile"]["dt_hours"],
        "aggregation": {
            "power_mw": "arithmetic mean of four 15-minute periods",
            "simultaneous_agv_count": "arithmetic mean of four 15-minute periods",
            "lohc_outbound_kg": "sum of four 15-minute interval quantities",
            "inventory_kg": "state sampled at integer-hour boundaries",
        },
        "semantic_note": (
            "The formal model has day-ahead charging-count and LOHC-reactor-power "
            "variables, but no day-ahead container/LOHC transport AGV allocation. "
            "Panel (b) therefore compares the two DA/RT variable pairs that actually exist."
        ),
        "source_files": [
            {"path": _project_relative(path), "sha256": _sha256(path)}
            for path in source_files
        ],
        "outputs": [
            "data/hourly_operation.csv",
            "data/hourly_inventories.csv",
            "figures/fig_c1_day_intraday_operation.pdf",
            "figures/fig_c1_day_intraday_operation.svg",
            "figures/fig_c1_day_intraday_operation.png",
        ],
    }
    (OUTPUT_ROOT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    day_ahead, recourse, certificate, resolved_case = _load_and_validate()
    hourly, states = _build_hourly_tables(day_ahead, recourse, resolved_case)
    hourly.to_csv(DATA_DIR / "hourly_operation.csv", index=False, float_format="%.10g")
    states.to_csv(DATA_DIR / "hourly_inventories.csv", index=False, float_format="%.10g")

    figure = _plot(hourly, states)
    stem = FIGURE_DIR / "fig_c1_day_intraday_operation"
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(figure)

    _write_manifest(certificate, resolved_case)
    print(f"source run: {SOURCE_RUN}")
    print(f"certificate status: {certificate['status']}")
    print(f"generated: {stem.with_suffix('.pdf')}")
    print(f"generated: {stem.with_suffix('.svg')}")
    print(f"generated: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
