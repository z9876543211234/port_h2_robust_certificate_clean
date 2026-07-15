"""Residual evaluation from the same recourse IR used by every compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from port_h2_certificate.compilers.common import (
    Realization,
    evaluate_row_rhs,
    validate_realization_references,
)
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey


@dataclass(frozen=True)
class ResidualReport:
    max_abs_residual: float
    violations: Mapping[str, float]


def evaluate_residuals(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    realization: Realization,
    solution: Mapping[VariableKey, float],
) -> ResidualReport:
    validate_realization_references(ir.constraints, realization)
    violations: dict[str, float] = {}
    for row in ir.constraints:
        lhs = 0.0
        for key, coefficient in row.recourse_coefficients.items():
            if key not in solution:
                raise KeyError(f"missing recourse value: {key}")
            lhs += coefficient * float(solution[key])
        rhs = evaluate_row_rhs(row, first_stage, realization)
        violation = abs(lhs - rhs) if row.sense == "eq" else max(rhs - lhs, 0.0)
        violations[row.name] = float(violation)
    return ResidualReport(
        max(violations.values(), default=0.0),
        violations,
    )

