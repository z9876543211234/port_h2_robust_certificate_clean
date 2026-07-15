#!/usr/bin/env python3
"""Plot formal-C1 LOHC inventory dynamics and shared-AGV task allocation."""

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
    "gold": "#D9A62E",
    "orange": "#E79B52",
    "coral": "#E77747",
    "ink": "#4C4C4C",
    "gray": "#7A7A7A",
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


def _load_formal_data() -> tuple[pd.DataFrame, dict, dict]:
    recourse_path = SOURCE_RUN / "dispatch_recourse_worst.csv"
    certificate_path = SOURCE_RUN / "certificate.json"
    resolved_case_path = SOURCE_RUN / "resolved_case.json"
    for path in (recourse_path, certificate_path, resolved_case_path):
        if not path.is_file():
            raise FileNotFoundError(f"required formal artifact not found: {path}")

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    resolved_case = json.loads(resolved_case_path.read_text(encoding="utf-8"))
    for key in ("master_status", "phase1_adversary_status", "cost_adversary_status"):
        if certificate.get(key) != "OPTIMAL":
            raise ValueError(f"formal certificate rejected: {key}={certificate.get(key)!r}")
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
            f"expected the formal 96-period, 0.25-h profile; "
            f"received horizon={horizon}, dt={dt_hours}"
        )

    recourse = pd.read_csv(recourse_path).sort_values("period").reset_index(drop=True)
    required = [
        "period",
        "agv_container_count",
        "agv_lohc_count",
        "agv_charge_count",
        "agv_idle_count",
        "lohc_production_kg",
        "lohc_outbound_kg",
        "lohc_inventory_kg",
    ]
    missing = sorted(set(required) - set(recourse.columns))
    if missing:
        raise ValueError(f"{recourse_path} is missing required columns: {missing}")
    expected_states = np.arange(horizon + 1)
    if len(recourse) != horizon + 1 or not np.array_equal(
        recourse["period"].to_numpy(int), expected_states
    ):
        raise ValueError("recourse artifact must contain state periods 0,...,96")
    flows = recourse.loc[recourse["period"] < horizon]
    allocation_columns = [
        "agv_container_count",
        "agv_lohc_count",
        "agv_charge_count",
        "agv_idle_count",
    ]
    flow_columns = [
        *allocation_columns,
        "lohc_production_kg",
        "lohc_outbound_kg",
    ]
    if not np.isfinite(flows[flow_columns].to_numpy(float)).all():
        raise ValueError("non-finite values found in operating-period flow columns")
    if not np.isfinite(recourse[["lohc_inventory_kg"]].to_numpy(float)).all():
        raise ValueError("non-finite values found in the LOHC inventory state column")

    fleet_size = float(resolved_case["agv"]["fleet_size"])
    allocation_residual = flows[allocation_columns].sum(axis=1) - fleet_size
    if float(allocation_residual.abs().max()) > 1e-8:
        raise ValueError("AGV task allocation does not conserve the archived fleet size")

    inventory = recourse["lohc_inventory_kg"].to_numpy(float)
    balance_residual = (
        inventory[1:]
        - inventory[:-1]
        - flows["lohc_production_kg"].to_numpy(float)
        + flows["lohc_outbound_kg"].to_numpy(float)
    )
    if float(np.max(np.abs(balance_residual))) > 1e-8:
        raise ValueError("LOHC inventory transition does not match production and outbound flow")
    return recourse, certificate, resolved_case


def _build_hourly_tables(
    recourse: pd.DataFrame, resolved_case: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    horizon = int(resolved_case["profile"]["periods"])
    dt_hours = float(resolved_case["profile"]["dt_hours"])
    periods_per_hour = int(round(1.0 / dt_hours))
    flows = recourse.loc[recourse["period"] < horizon].copy()
    flows["hour"] = flows["period"].to_numpy(int) // periods_per_hour

    count_columns = [
        "agv_container_count",
        "agv_lohc_count",
        "agv_charge_count",
        "agv_idle_count",
    ]
    hourly = flows.groupby("hour", as_index=False)[count_columns].mean()
    lohc_flows = flows.groupby("hour", as_index=False)[
        ["lohc_production_kg", "lohc_outbound_kg"]
    ].sum()
    hourly = hourly.merge(lohc_flows, on="hour", validate="one_to_one")
    hourly["hour_center"] = hourly["hour"] + 0.5

    state_rows = np.arange(0, horizon + 1, periods_per_hour, dtype=int)
    state_source = recourse.set_index("period").loc[state_rows]
    states = pd.DataFrame(
        {
            "hour_boundary": np.arange(25, dtype=int),
            "lohc_inventory_kg": state_source["lohc_inventory_kg"].to_numpy(float),
        }
    )
    if len(hourly) != 24 or len(states) != 25:
        raise AssertionError("unexpected hourly flow/state table length")

    fleet_size = float(resolved_case["agv"]["fleet_size"])
    diagnostics = {
        "max_hourly_agv_conservation_residual": float(
            np.max(np.abs(hourly[count_columns].sum(axis=1) - fleet_size))
        ),
        "lohc_production_total_kg": float(hourly["lohc_production_kg"].sum()),
        "lohc_outbound_total_kg": float(hourly["lohc_outbound_kg"].sum()),
        "lohc_inventory_initial_kg": float(states["lohc_inventory_kg"].iloc[0]),
        "lohc_inventory_terminal_kg": float(states["lohc_inventory_kg"].iloc[-1]),
    }
    return hourly, states, diagnostics


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="both", color="0.89", linewidth=0.5)
    axis.set_axisbelow(True)


def _plot(
    hourly: pd.DataFrame, states: pd.DataFrame, resolved_case: dict
) -> plt.Figure:
    x = hourly["hour_center"].to_numpy(float)
    state_x = states["hour_boundary"].to_numpy(float)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.16, 3.9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # (a) LOHC interval flows and inventory state.
    axis = axes[0]
    inventory_axis = axis.twinx()
    bar_width = 0.34
    axis.bar(
        x - bar_width / 2,
        hourly["lohc_production_kg"],
        width=bar_width,
        color=COLORS["mint"],
        edgecolor=COLORS["teal"],
        linewidth=0.35,
        label="LOHC production",
    )
    axis.bar(
        x + bar_width / 2,
        hourly["lohc_outbound_kg"],
        width=bar_width,
        color="#E9B05D",
        edgecolor=COLORS["orange"],
        linewidth=0.35,
        label="LOHC outbound",
    )
    inventory_axis.plot(
        state_x,
        states["lohc_inventory_kg"],
        color=COLORS["coral"],
        marker="s",
        markersize=2.5,
        linewidth=1.25,
        label="LOHC inventory",
    )
    axis.set_ylabel("Hourly LOHC flow\n(kg-H$_2$-eq.)")
    inventory_axis.set_ylabel("LOHC inventory (kg-H$_2$-eq.)")
    axis.text(0.01, 0.92, "(a)", transform=axis.transAxes, va="top", fontsize=9.0)
    h1, l1 = axis.get_legend_handles_labels()
    h2, l2 = inventory_axis.get_legend_handles_labels()
    axis.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.04),
        columnspacing=1.1,
    )
    flow_max = max(
        float(hourly["lohc_production_kg"].max()),
        float(hourly["lohc_outbound_kg"].max()),
    )
    inventory_max = float(states["lohc_inventory_kg"].max())
    axis.set_ylim(0.0, max(1.0, flow_max * 1.18))
    inventory_axis.set_ylim(0.0, max(1.0, inventory_max * 1.16))
    _style_axis(axis)
    inventory_axis.spines["top"].set_visible(False)
    inventory_axis.grid(False)

    # (b) Hourly mean allocation of the shared AGV fleet.
    axis = axes[1]
    axis.plot(
        x,
        hourly["agv_container_count"],
        color=COLORS["blue"],
        marker="s",
        markersize=2.5,
        linewidth=1.25,
        label="Container service",
    )
    axis.plot(
        x,
        hourly["agv_lohc_count"],
        color=COLORS["coral"],
        marker="D",
        markersize=2.2,
        label="LOHC transport",
    )
    axis.plot(
        x,
        hourly["agv_charge_count"],
        color=COLORS["gold"],
        linestyle="--",
        marker="o",
        markersize=2.4,
        label="Charging",
    )
    axis.plot(
        x,
        hourly["agv_idle_count"],
        color=COLORS["gray"],
        linestyle=":",
        marker="^",
        markersize=2.5,
        label="Idle",
    )
    fleet_size = float(resolved_case["agv"]["fleet_size"])
    axis.axhline(
        fleet_size,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=0.85,
        label="Fleet size",
    )
    axis.set_ylabel("Equivalent number of AGVs")
    axis.set_ylim(0.0, fleet_size * 1.18)
    axis.text(0.01, 0.92, "(b)", transform=axis.transAxes, va="top", fontsize=9.0)
    axis.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.04),
        columnspacing=0.85,
    )
    _style_axis(axis)

    for axis in axes:
        axis.set_xlim(0.0, 24.0)
        axis.set_xticks([0, 6, 12, 18, 24])
    axes[-1].set_xlabel("Time (h)")
    return figure


def _write_manifest(
    certificate: dict,
    resolved_case: dict,
    diagnostics: dict[str, float],
) -> None:
    source_files = [
        SOURCE_RUN / "dispatch_recourse_worst.csv",
        SOURCE_RUN / "certificate.json",
        SOURCE_RUN / "resolved_case.json",
    ]
    manifest = {
        "figure": "fig_c1_lohc_inventory_agv_allocation",
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
            "lohc_production_and_outbound_kg": (
                "sum of four 15-minute interval quantities"
            ),
            "lohc_inventory_kg": "state sampled at integer-hour boundaries",
            "simultaneous_agv_count": "arithmetic mean of four 15-minute periods",
        },
        "diagnostics": diagnostics,
        "source_files": [
            {"path": _project_relative(path), "sha256": _sha256(path)}
            for path in source_files
        ],
        "outputs": [
            "data/hourly_lohc_agv_allocation.csv",
            "data/hourly_lohc_inventory.csv",
            "figures/fig_c1_lohc_inventory_agv_allocation.pdf",
            "figures/fig_c1_lohc_inventory_agv_allocation.svg",
            "figures/fig_c1_lohc_inventory_agv_allocation.png",
        ],
    }
    (OUTPUT_ROOT / "source_manifest_lohc_inventory_agv_allocation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    recourse, certificate, resolved_case = _load_formal_data()
    hourly, states, diagnostics = _build_hourly_tables(recourse, resolved_case)
    hourly.to_csv(
        DATA_DIR / "hourly_lohc_agv_allocation.csv",
        index=False,
        float_format="%.15g",
    )
    states.to_csv(
        DATA_DIR / "hourly_lohc_inventory.csv",
        index=False,
        float_format="%.15g",
    )

    figure = _plot(hourly, states, resolved_case)
    stem = FIGURE_DIR / "fig_c1_lohc_inventory_agv_allocation"
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(figure)
    _write_manifest(certificate, resolved_case, diagnostics)

    print(f"source run: {SOURCE_RUN}")
    print(f"certificate status: {certificate['status']}")
    print(f"generated: {stem.with_suffix('.pdf')}")
    print(f"generated: {stem.with_suffix('.svg')}")
    print(f"generated: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
