"""Fail-closed publication checks shared by all formal figures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .data_loader import FormalRun, FormalRunError


def assert_cost_reconciliation(
    run: FormalRun,
    components: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> None:
    target = (
        float(run.certificate["stress_total_objective"])
        if run.certificate["case_name"] == "C4_DeterministicMain"
        else float(run.certificate["upper_bound"])
    )
    error = abs(sum(float(value) for value in components.values()) - target)
    if error > tolerance:
        raise FormalRunError(
            f"cost reconciliation failed for {run.certificate['case_name']}: {error}"
        )


def assert_same_bundle(runs: Sequence[FormalRun]) -> None:
    hashes = {run.certificate["joint_bundle_sha256"] for run in runs}
    if len(hashes) != 1:
        raise FormalRunError(f"formal runs use different uncertainty bundles: {hashes}")


def assert_same_scenario(scenario_hashes: Sequence[str]) -> None:
    hashes = set(scenario_hashes)
    if len(hashes) != 1:
        raise FormalRunError(f"comparison uses different scenarios: {hashes}")


def assert_integer_arrivals(values, *, tolerance: float = 1e-9) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or not np.allclose(
        array, np.rint(array), atol=tolerance, rtol=0.0
    ):
        raise FormalRunError("vessel arrivals are not finite integers")


def assert_no_nan_or_inf(values, *, name: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise FormalRunError(f"{name} contains NaN or infinite values")


def assert_power_balance(residual, *, tolerance: float = 1e-6) -> None:
    maximum = float(np.max(np.abs(np.asarray(residual, dtype=float))))
    if not math.isfinite(maximum) or maximum > tolerance:
        raise FormalRunError(f"power-balance residual exceeds tolerance: {maximum}")


def assert_agv_conservation(
    component_sum, fleet_size: float, *, tolerance: float = 1e-6
) -> None:
    maximum = float(
        np.max(np.abs(np.asarray(component_sum, dtype=float) - float(fleet_size)))
    )
    if maximum > tolerance:
        raise FormalRunError(f"AGV conservation residual exceeds tolerance: {maximum}")


def assert_state_lengths(flow_length: int, state_length: int) -> None:
    if state_length != flow_length + 1:
        raise FormalRunError(
            f"state trajectory must have T+1 points: {state_length} vs {flow_length}"
        )
