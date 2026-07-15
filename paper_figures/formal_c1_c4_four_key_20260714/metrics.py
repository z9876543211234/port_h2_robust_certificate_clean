"""Shared economic, energy, and logistics metrics for all figures and tables."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .data_loader import FormalRun


def signed_cost_components(run: FormalRun) -> OrderedDict[str, float]:
    components: OrderedDict[str, float] = OrderedDict()
    for name, value in run.cost_breakdown["day_ahead_by_component"].items():
        components[f"DA {name}"] = float(value)
    for name, value in run.cost_breakdown["recourse_by_component"].items():
        components[f"RT {name}"] = float(value)
    return components


def compute_operational_metrics(
    run: FormalRun, *, dt_hours: float
) -> dict[str, float]:
    recourse = run.recourse_worst.loc[run.recourse_worst["period"] < 96].copy()
    states = run.recourse_worst.sort_values("period")
    backlog = states["backlog_tasks"].dropna().to_numpy(dtype=float)
    soc = states["soc"].dropna().to_numpy(dtype=float)
    grid_import = recourse["grid_buy_rt_mw"].to_numpy(dtype=float)
    grid_export = recourse["grid_sell_rt_mw"].to_numpy(dtype=float)
    spill = recourse["spill_rt_mw"].to_numpy(dtype=float)
    lohc_outbound = (
        recourse["lohc_outbound_kg"].to_numpy(dtype=float)
        if "lohc_outbound_kg" in recourse
        else np.zeros(len(recourse), dtype=float)
    )
    return {
        "grid_import_mwh": float(np.sum(grid_import) * dt_hours),
        "grid_export_mwh": float(np.sum(grid_export) * dt_hours),
        "spill_mwh": float(np.sum(spill) * dt_hours),
        "lohc_outbound_kg": float(np.sum(lohc_outbound)),
        "backlog_area_teu_h": float(np.sum(backlog[:-1]) * dt_hours),
        "peak_backlog_teu": float(np.max(backlog)),
        "terminal_backlog_teu": float(backlog[-1]),
        "minimum_soc": float(np.min(soc)),
        "terminal_soc": float(soc[-1]),
    }

