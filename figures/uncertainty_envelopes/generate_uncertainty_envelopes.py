"""Generate current ship-arrival and wind pointwise uncertainty envelopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import gurobipy as gp

MPL_CONFIG_DIRECTORY = Path(tempfile.gettempdir()) / "porth2-matplotlib"
MPL_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIRECTORY))

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource
from port_h2_uncertainty_builders.wind.builder import build_wind_bundle
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


PROFILE = HorizonProfile.formal("quarter_hour_96", 96, 0.25)
SHIP_SOURCE_PATH = (
    PROJECT_ROOT
    / "data/uncertainty_sources/ship_delay/quarter_hour_96.json"
)
WIND_SOURCE_PATH = (
    PROJECT_ROOT / "data/uncertainty_sources/wind/quarter_hour_96.json"
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "gray": "#6B6B6B",
    "light_gray": "#D9D9D9",
}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ship_source(
    payload: dict[str, object],
    *,
    max_delay_steps: int,
    delayed_ship_budget: int,
    total_delay_step_budget: int | None,
) -> ShipDelaySource:
    return ShipDelaySource(
        profile=PROFILE,
        previous_day_arrival_count=tuple(payload["previous_day_arrival_count"]),
        current_day_arrival_count=tuple(payload["current_day_arrival_count"]),
        next_day_arrival_count=tuple(payload["next_day_arrival_count"]),
        dwell_steps=int(payload["dwell_steps"]),
        max_delay_steps=max_delay_steps,
        delayed_ship_budget=delayed_ship_budget,
        total_delay_step_budget=total_delay_step_budget,
        shore_power_per_ship_kw=float(payload["shore_power_per_ship_kw"]),
        quay_cranes_per_ship=float(payload["quay_cranes_per_ship"]),
        installed_quay_cranes=float(payload["installed_quay_cranes"]),
        quay_crane_power_kw=float(payload["quay_crane_power_kw"]),
        quay_crane_task_rate_per_hour=float(
            payload["quay_crane_task_rate_per_hour"]
        ),
        provenance=ProvenanceRecord.from_dict(payload["provenance"]),
    )


def _wind_source(payload: dict[str, object]) -> WindUncertaintySource:
    return WindUncertaintySource(
        profile=PROFILE,
        nominal_power_kw=tuple(float(value) for value in payload["nominal_power_kw"]),
        deviation_down_kw=tuple(
            float(value) for value in payload["deviation_down_kw"]
        ),
        deviation_up_kw=tuple(float(value) for value in payload["deviation_up_kw"]),
        budget=int(payload["budget"]),
        provenance=ProvenanceRecord.from_dict(payload["provenance"]),
    )


def pointwise_bundle_bounds(
    bundle: UncertaintyBundle, output_key: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Optimize every output coordinate over the exact Bundle constraints."""

    output = bundle.outputs[output_key]
    matrix = bundle.output_matrix(output_key).to_scipy().toarray()
    model = gp.Model(f"pointwise_bounds_{output_key.replace('.', '_')}")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    variables = {
        key: model.addVar(vtype=gp.GRB.BINARY, name=key)
        for key in bundle.selector_keys
    }
    for row in bundle.constraints:
        expression = gp.quicksum(
            coefficient * variables[key]
            for key, coefficient in row.coefficients.items()
        )
        if row.sense == "le":
            model.addConstr(expression <= row.rhs, name=row.name)
        elif row.sense == "ge":
            model.addConstr(expression >= row.rhs, name=row.name)
        else:
            model.addConstr(expression == row.rhs, name=row.name)

    lower = np.zeros(len(output.nominal), dtype=float)
    upper = np.zeros(len(output.nominal), dtype=float)
    statuses: list[int] = []
    for period, nominal in enumerate(output.nominal):
        expression = nominal + gp.quicksum(
            matrix[period, column] * variables[key]
            for column, key in enumerate(bundle.selector_keys)
            if matrix[period, column] != 0.0
        )
        model.setObjective(expression, gp.GRB.MINIMIZE)
        model.optimize()
        statuses.append(int(model.Status))
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"lower-bound solve failed at period {period}")
        lower[period] = model.ObjVal

        model.setObjective(expression, gp.GRB.MAXIMIZE)
        model.optimize()
        statuses.append(int(model.Status))
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"upper-bound solve failed at period {period}")
        upper[period] = model.ObjVal
    return lower, upper, tuple(statuses)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _positive_markers(ax, hours: np.ndarray, values: np.ndarray, color: str) -> None:
    positive = values > 0
    ax.scatter(
        hours[positive],
        values[positive],
        s=13,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )


def _plot_ship(
    output_directory: Path,
    ship_payload: dict[str, object],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_delay_steps: int,
    delayed_ship_budget: int,
) -> None:
    hours = np.arange(PROFILE.periods, dtype=float) * PROFILE.dt_hours
    previous = np.asarray(ship_payload["previous_day_arrival_count"], dtype=float)
    current = np.asarray(ship_payload["current_day_arrival_count"], dtype=float)
    following = np.asarray(ship_payload["next_day_arrival_count"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.1), sharex=True)
    series = (
        (previous, COLORS["blue"], f"Previous day ({int(previous.sum())} ships)"),
        (current, COLORS["orange"], f"Current day ({int(current.sum())} ships)"),
        (following, COLORS["green"], f"Next day ({int(following.sum())} ships)"),
    )
    for values, color, label in series:
        axes[0].step(hours, values, where="post", color=color, linewidth=1.15, label=label)
        _positive_markers(axes[0], hours, values, color)
    axes[0].set_ylabel("Nominal arrivals\n(ship/15 min)")
    axes[0].set_ylim(-0.08, max(3.35, float(max(previous.max(), current.max(), following.max())) + 0.4))
    axes[0].set_yticks(range(0, int(axes[0].get_ylim()[1]) + 1))
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[0].grid(axis="y", color=COLORS["light_gray"], linewidth=0.5)
    axes[0].text(
        -0.055,
        0.98,
        "(a)",
        transform=axes[0].transAxes,
        va="top",
        fontweight="bold",
    )

    axes[1].fill_between(
        hours,
        lower,
        upper,
        step="post",
        color=COLORS["yellow"],
        alpha=0.24,
        linewidth=0.0,
        label="Pointwise delay envelope",
    )
    axes[1].step(
        hours,
        upper,
        where="post",
        color=COLORS["gray"],
        linewidth=0.9,
        linestyle="--",
        label="Upper/lower bounds",
    )
    axes[1].step(
        hours,
        lower,
        where="post",
        color=COLORS["gray"],
        linewidth=0.9,
        linestyle="--",
    )
    axes[1].step(
        hours,
        current,
        where="post",
        color=COLORS["orange"],
        linewidth=1.35,
        label="Current-day nominal",
    )
    _positive_markers(axes[1], hours, current, COLORS["orange"])
    axes[1].set_xlabel("Time of day (h)")
    axes[1].set_ylabel("Arrival realization\n(ship/15 min)")
    axes[1].set_xlim(0.0, 24.0)
    axes[1].set_xticks(np.arange(0.0, 25.0, 3.0))
    axes[1].set_ylim(-0.08, max(3.35, float(upper.max()) + 0.4))
    axes[1].set_yticks(range(0, int(axes[1].get_ylim()[1]) + 1))
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].grid(axis="y", color=COLORS["light_gray"], linewidth=0.5)
    axes[1].text(
        -0.055,
        0.98,
        "(b)",
        transform=axes[1].transAxes,
        va="top",
        fontweight="bold",
    )
    axes[1].text(
        0.99,
        0.91,
        rf"Validation set: $D_{{\max}}={max_delay_steps}$, $\Gamma_s={delayed_ship_budget}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["gray"],
    )
    fig.tight_layout(h_pad=0.8)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_directory / f"fig_ship_arrival_envelope.{suffix}")
    plt.close(fig)


def _plot_wind(
    output_directory: Path,
    wind_payload: dict[str, object],
    lower_kw: np.ndarray,
    upper_kw: np.ndarray,
) -> None:
    hours = np.arange(PROFILE.periods, dtype=float) * PROFILE.dt_hours
    nominal_mw = np.asarray(wind_payload["nominal_power_kw"], dtype=float) / 1000.0
    lower_mw = lower_kw / 1000.0
    upper_mw = upper_kw / 1000.0
    budget = int(wind_payload["budget"])

    fig, ax = plt.subplots(1, 1, figsize=(7.15, 3.25))
    ax.fill_between(
        hours,
        lower_mw,
        upper_mw,
        color=COLORS["blue"],
        alpha=0.16,
        linewidth=0.0,
        label=rf"Pointwise envelope ($\Gamma_w={budget}$)",
    )
    ax.plot(
        hours,
        upper_mw,
        color=COLORS["gray"],
        linewidth=0.8,
        linestyle="--",
        label="Upper/lower bounds",
    )
    ax.plot(
        hours,
        lower_mw,
        color=COLORS["gray"],
        linewidth=0.8,
        linestyle="--",
    )
    ax.plot(
        hours,
        nominal_mw,
        color=COLORS["blue"],
        linewidth=1.55,
        label="Nominal wind power",
    )
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Available wind power (MW)")
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0.0, 25.0, 3.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.5)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_directory / f"fig_wind_uncertainty_envelope.{suffix}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ship-max-delay-steps", type=int, required=True)
    parser.add_argument("--ship-delayed-budget", type=int, required=True)
    parser.add_argument("--ship-total-delay-step-budget", type=int)
    args = parser.parse_args()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    ship_payload = _read_json(SHIP_SOURCE_PATH)
    wind_payload = _read_json(WIND_SOURCE_PATH)
    ship_source = _ship_source(
        ship_payload,
        max_delay_steps=args.ship_max_delay_steps,
        delayed_ship_budget=args.ship_delayed_budget,
        total_delay_step_budget=args.ship_total_delay_step_budget,
    )
    wind_source = _wind_source(wind_payload)

    with tempfile.TemporaryDirectory(prefix="porth2-envelope-") as temporary:
        temporary_root = Path(temporary)
        ship_bundle = build_ship_delay_bundle(ship_source, temporary_root / "ship")
        wind_bundle = build_wind_bundle(wind_source, temporary_root / "wind")
        ship_lower, ship_upper, ship_statuses = pointwise_bundle_bounds(
            ship_bundle, "ship.arrival_count"
        )
        wind_lower, wind_upper, wind_statuses = pointwise_bundle_bounds(
            wind_bundle, "wind.available_power_kw"
        )

    if not np.allclose(ship_lower, np.rint(ship_lower), atol=1.0e-9):
        raise RuntimeError("ship lower envelope is not integer-valued")
    if not np.allclose(ship_upper, np.rint(ship_upper), atol=1.0e-9):
        raise RuntimeError("ship upper envelope is not integer-valued")
    ship_lower = np.rint(ship_lower).astype(int)
    ship_upper = np.rint(ship_upper).astype(int)
    if set(ship_statuses + wind_statuses) != {gp.GRB.OPTIMAL}:
        raise RuntimeError("pointwise envelope requires OPTIMAL solves only")

    hours = np.arange(PROFILE.periods, dtype=float) * PROFILE.dt_hours
    ship_rows = []
    for period in range(PROFILE.periods):
        ship_rows.append(
            {
                "t": period,
                "hour": f"{hours[period]:.2f}",
                "previous_day_nominal_ship_count": ship_source.previous_day_arrival_count[period],
                "current_day_nominal_ship_count": ship_source.current_day_arrival_count[period],
                "next_day_nominal_ship_count": ship_source.next_day_arrival_count[period],
                "current_day_pointwise_lower_ship_count": int(ship_lower[period]),
                "current_day_pointwise_upper_ship_count": int(ship_upper[period]),
            }
        )
    _write_csv(
        output_directory / "ship_arrival_envelope.csv",
        list(ship_rows[0]),
        ship_rows,
    )

    wind_rows = []
    for period in range(PROFILE.periods):
        wind_rows.append(
            {
                "t": period,
                "hour": f"{hours[period]:.2f}",
                "nominal_wind_mw": f"{wind_source.nominal_power_kw[period] / 1000.0:.10g}",
                "pointwise_lower_wind_mw": f"{wind_lower[period] / 1000.0:.10g}",
                "pointwise_upper_wind_mw": f"{wind_upper[period] / 1000.0:.10g}",
            }
        )
    _write_csv(
        output_directory / "wind_uncertainty_envelope.csv",
        list(wind_rows[0]),
        wind_rows,
    )

    _configure_style()
    _plot_ship(
        output_directory,
        ship_payload,
        ship_lower,
        ship_upper,
        max_delay_steps=args.ship_max_delay_steps,
        delayed_ship_budget=args.ship_delayed_budget,
    )
    _plot_wind(output_directory, wind_payload, wind_lower, wind_upper)

    metadata = {
        "schema_version": 1,
        "profile": PROFILE.profile_name,
        "ship_envelope_status": "provisional_validation_parameters",
        "ship_parameters": {
            "max_delay_steps": args.ship_max_delay_steps,
            "max_delay_hours": args.ship_max_delay_steps * PROFILE.dt_hours,
            "delayed_ship_budget": args.ship_delayed_budget,
            "total_delay_step_budget": args.ship_total_delay_step_budget,
        },
        "ship_nominal_daily_totals": [
            sum(ship_source.previous_day_arrival_count),
            sum(ship_source.current_day_arrival_count),
            sum(ship_source.next_day_arrival_count),
        ],
        "ship_pointwise_lower_min_max": [int(ship_lower.min()), int(ship_lower.max())],
        "ship_pointwise_upper_min_max": [int(ship_upper.min()), int(ship_upper.max())],
        "wind_budget": wind_source.budget,
        "wind_nominal_min_max_mw": [
            float(min(wind_source.nominal_power_kw) / 1000.0),
            float(max(wind_source.nominal_power_kw) / 1000.0),
        ],
        "wind_pointwise_lower_min_max_mw": [
            float(wind_lower.min() / 1000.0),
            float(wind_lower.max() / 1000.0),
        ],
        "wind_pointwise_upper_min_max_mw": [
            float(wind_upper.min() / 1000.0),
            float(wind_upper.max() / 1000.0),
        ],
        "pointwise_envelope_note": (
            "Each time-coordinate bound is exact, but all pointwise extremes cannot "
            "generally occur simultaneously under the global uncertainty budgets."
        ),
        "all_bound_solver_statuses": "OPTIMAL",
        "ship_bundle_sha256": ship_bundle.bundle_sha256,
        "wind_bundle_sha256": wind_bundle.bundle_sha256,
        "ship_source_sha256": _sha256(SHIP_SOURCE_PATH),
        "wind_source_sha256": _sha256(WIND_SOURCE_PATH),
        "generator_sha256": _sha256(Path(__file__)),
    }
    (output_directory / "envelope_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "latex_includes.tex").write_text(
        """% Ship arrival nominal profiles and provisional integer-delay envelope
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=0.98\\linewidth]{figures/uncertainty_envelopes/fig_ship_arrival_envelope.pdf}
  \\caption{Frozen three-day NHPP integer ship-arrival profiles and the exact pointwise current-day delay envelope under the stated provisional validation parameters.}
  \\label{fig:ship-arrival-envelope}
\\end{figure}

% Wind nominal profile and budgeted pointwise envelope
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=0.98\\linewidth]{figures/uncertainty_envelopes/fig_wind_uncertainty_envelope.pdf}
  \\caption{Nominal wind-power profile and exact pointwise envelope of the budgeted two-sided uncertainty set with $\\Gamma_w=12$. Pointwise extremes need not be simultaneous.}
  \\label{fig:wind-uncertainty-envelope}
\\end{figure}
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
