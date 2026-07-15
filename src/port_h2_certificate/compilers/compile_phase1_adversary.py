"""Exact Phase-I adversary generated from the Phase-I LP dual."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import gurobipy as gp

from port_h2_certificate.compilers.compile_adversary import (
    CompiledAdversary,
    compile_adversary,
)
from port_h2_certificate.compilers.compile_dual import UpperBoundRow
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class Phase1DualTemplate:
    ir: RecourseIR
    first_stage: Mapping[VariableKey, float]
    lower_bounds: Mapping[VariableKey, float]
    upper_bound_rows: tuple[UpperBoundRow, ...]
    objective_constant: float
    sigma: Mapping[str, float]

    @property
    def template_kind(self) -> str:
        return "phase1"

    def row_dual_bounds(self, row) -> tuple[float, float]:
        inverse_sigma = 1.0 / self.sigma[row.name]
        if row.sense == "eq":
            return -inverse_sigma, inverse_sigma
        return 0.0, inverse_sigma

    @staticmethod
    def dual_constraint_rhs(spec) -> float:
        return 0.0


def build_phase1_dual_template(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    sigma: float | Mapping[str, float] = 1.0,
) -> Phase1DualTemplate:
    if isinstance(sigma, Mapping):
        missing = {row.name for row in ir.constraints} - set(sigma)
        if missing:
            raise KeyError(f"missing Phase-I sigma values: {sorted(missing)}")
        weights = {row.name: float(sigma[row.name]) for row in ir.constraints}
    else:
        weights = {row.name: float(sigma) for row in ir.constraints}
    if any(value <= 0.0 or not math.isfinite(value) for value in weights.values()):
        raise ValueError("all Phase-I sigma values must be positive and finite")

    lower_bounds: dict[VariableKey, float] = {}
    upper_rows: list[UpperBoundRow] = []
    for spec in ir.variables:
        if not math.isfinite(spec.lower_bound):
            raise ValueError(
                f"Phase-I dual requires a finite lower bound for {spec.key}"
            )
        lower_bounds[spec.key] = spec.lower_bound
        if spec.upper_bound is not None:
            width = spec.upper_bound - spec.lower_bound
            if width < 0.0:
                raise ValueError(f"inconsistent bounds for {spec.key}")
            upper_rows.append(UpperBoundRow(spec.key, width))
    return Phase1DualTemplate(
        ir=ir,
        first_stage=dict(first_stage),
        lower_bounds=lower_bounds,
        upper_bound_rows=tuple(upper_rows),
        objective_constant=0.0,
        sigma=weights,
    )


def compile_phase1_adversary(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    joint_bundle: UncertaintyBundle,
    sigma: float | Mapping[str, float] = 1.0,
) -> CompiledAdversary:
    return compile_adversary(
        build_phase1_dual_template(ir, first_stage, sigma), joint_bundle
    )
