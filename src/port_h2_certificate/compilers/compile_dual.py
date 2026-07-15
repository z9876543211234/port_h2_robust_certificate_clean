"""Explicit LP dual generated mechanically from the reduced recourse IR."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import gurobipy as gp

from port_h2_certificate.compilers.common import (
    Realization,
    configure_formal_lp,
    evaluate_row_rhs,
    validate_realization_references,
)
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey


@dataclass(frozen=True)
class UpperBoundRow:
    variable_key: VariableKey
    width: float


@dataclass(frozen=True)
class DualResult:
    status: int
    objective: float
    row_duals: Mapping[str, float]
    upper_bound_duals: Mapping[VariableKey, float]
    max_constraint_violation: float


@dataclass
class CompiledDual:
    template: "DualTemplate"
    realization: Realization
    model: gp.Model
    row_duals: Mapping[str, gp.Var]
    upper_bound_duals: Mapping[VariableKey, gp.Var]

    def solve(self) -> DualResult:
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"fixed-scenario explicit dual did not reach GRB.OPTIMAL: {self.model.Status}"
            )
        row_values = {name: variable.X for name, variable in self.row_duals.items()}
        upper_values = {
            key: variable.X for key, variable in self.upper_bound_duals.items()
        }
        violations: list[float] = []
        for spec in self.template.ir.variables:
            lhs = 0.0
            for row in self.template.ir.constraints:
                lhs += row.recourse_coefficients.get(spec.key, 0.0) * row_values[row.name]
            if spec.key in upper_values:
                lhs -= upper_values[spec.key]
            violations.append(max(lhs - spec.objective_coefficient, 0.0))
        return DualResult(
            status=self.model.Status,
            objective=float(self.model.ObjVal),
            row_duals=row_values,
            upper_bound_duals=upper_values,
            max_constraint_violation=max(violations, default=0.0),
        )


@dataclass(frozen=True)
class DualTemplate:
    ir: RecourseIR
    first_stage: Mapping[VariableKey, float]
    lower_bounds: Mapping[VariableKey, float]
    upper_bound_rows: tuple[UpperBoundRow, ...]
    objective_constant: float

    @property
    def template_kind(self) -> str:
        return "cost"

    @staticmethod
    def row_dual_bounds(row) -> tuple[float, float]:
        if row.sense == "eq":
            return -gp.GRB.INFINITY, gp.GRB.INFINITY
        return 0.0, gp.GRB.INFINITY

    @staticmethod
    def dual_constraint_rhs(spec) -> float:
        return spec.objective_coefficient

    def instantiate(self, realization: Realization) -> CompiledDual:
        validate_realization_references(self.ir.constraints, realization)
        model = gp.Model("fixed_scenario_explicit_dual")
        configure_formal_lp(model)
        row_duals = {
            row.name: model.addVar(
                lb=self.row_dual_bounds(row)[0],
                ub=self.row_dual_bounds(row)[1],
                vtype=gp.GRB.CONTINUOUS,
                name=f"dual[{row.name}]",
            )
            for row in self.ir.constraints
        }
        upper_duals = {
            row.variable_key: model.addVar(
                lb=0.0,
                ub=gp.GRB.INFINITY,
                vtype=gp.GRB.CONTINUOUS,
                name=f"dual_upper[{row.variable_key[0]}[{row.variable_key[1]}]]",
            )
            for row in self.upper_bound_rows
        }
        for spec in self.ir.variables:
            lhs = gp.quicksum(
                row.recourse_coefficients.get(spec.key, 0.0) * row_duals[row.name]
                for row in self.ir.constraints
            )
            if spec.key in upper_duals:
                lhs -= upper_duals[spec.key]
            model.addConstr(
                lhs <= self.dual_constraint_rhs(spec),
                name=f"dual_feasibility[{spec.key[0]}[{spec.key[1]}]]",
            )

        objective = gp.LinExpr(self.objective_constant)
        for row in self.ir.constraints:
            normalized_rhs = evaluate_row_rhs(
                row, self.first_stage, realization
            ) - sum(
                coefficient * self.lower_bounds[key]
                for key, coefficient in row.recourse_coefficients.items()
            )
            objective += normalized_rhs * row_duals[row.name]
        for row in self.upper_bound_rows:
            objective -= row.width * upper_duals[row.variable_key]
        model.setObjective(objective, gp.GRB.MAXIMIZE)
        model.update()
        return CompiledDual(self, realization, model, row_duals, upper_duals)


def compile_dual(
    ir: RecourseIR, first_stage: Mapping[VariableKey, float]
) -> DualTemplate:
    lower_bounds: dict[VariableKey, float] = {}
    upper_rows: list[UpperBoundRow] = []
    objective_constant = 0.0
    for spec in ir.variables:
        if not math.isfinite(spec.lower_bound):
            raise ValueError(
                f"explicit dual requires a finite lower bound for {spec.key}"
            )
        lower_bounds[spec.key] = spec.lower_bound
        objective_constant += spec.objective_coefficient * spec.lower_bound
        if spec.upper_bound is not None:
            width = spec.upper_bound - spec.lower_bound
            if width < 0.0:
                raise ValueError(f"inconsistent bounds for {spec.key}")
            upper_rows.append(UpperBoundRow(spec.key, width))
    return DualTemplate(
        ir=ir,
        first_stage=dict(first_stage),
        lower_bounds=lower_bounds,
        upper_bound_rows=tuple(upper_rows),
        objective_constant=float(objective_constant),
    )
