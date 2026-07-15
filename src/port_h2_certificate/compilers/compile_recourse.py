"""Compile and solve one fixed-realization recourse LP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import gurobipy as gp

from port_h2_certificate.compilers.common import (
    Realization,
    configure_formal_lp,
    evaluate_row_rhs,
    validate_realization_references,
)
from port_h2_certificate.compilers.residuals import evaluate_residuals
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey


@dataclass(frozen=True)
class RecourseResult:
    status: int
    objective: float
    variables: Mapping[VariableKey, float]
    cost_by_component: Mapping[str, float]
    max_abs_residual: float
    residuals: Mapping[str, float]
    variable_basis: Mapping[VariableKey, int]
    constraint_basis: Mapping[str, int]


@dataclass
class CompiledRecourse:
    ir: RecourseIR
    first_stage: Mapping[VariableKey, float]
    realization: Realization
    model: gp.Model
    variables: Mapping[VariableKey, gp.Var]
    constraints: Mapping[str, gp.Constr]

    def solve(self) -> RecourseResult:
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"fixed-scenario recourse did not reach GRB.OPTIMAL: {self.model.Status}"
            )
        values = {key: variable.X for key, variable in self.variables.items()}
        report = evaluate_residuals(
            self.ir, self.first_stage, self.realization, values
        )
        by_component: dict[str, float] = {}
        specs = {spec.key: spec for spec in self.ir.variables}
        for key, value in values.items():
            spec = specs[key]
            by_component[spec.component] = by_component.get(spec.component, 0.0) + (
                spec.objective_coefficient * value
            )
        return RecourseResult(
            status=self.model.Status,
            objective=float(self.model.ObjVal),
            variables=values,
            cost_by_component=by_component,
            max_abs_residual=report.max_abs_residual,
            residuals=report.violations,
            variable_basis={key: variable.VBasis for key, variable in self.variables.items()},
            constraint_basis={
                name: constraint.CBasis
                for name, constraint in self.constraints.items()
            },
        )


def compile_recourse(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    realization: Realization,
) -> CompiledRecourse:
    validate_realization_references(ir.constraints, realization)
    model = gp.Model("fixed_scenario_recourse")
    configure_formal_lp(model)
    variables = {
        spec.key: model.addVar(
            lb=spec.lower_bound,
            ub=gp.GRB.INFINITY if spec.upper_bound is None else spec.upper_bound,
            vtype=gp.GRB.CONTINUOUS,
            name=f"{spec.key[0]}[{spec.key[1]}]",
        )
        for spec in ir.variables
    }
    constraints: dict[str, gp.Constr] = {}
    for row in ir.constraints:
        lhs = gp.quicksum(
            coefficient * variables[key]
            for key, coefficient in row.recourse_coefficients.items()
        )
        rhs = evaluate_row_rhs(row, first_stage, realization)
        if row.sense == "eq":
            constraint = model.addConstr(lhs == rhs, name=row.name)
        else:
            constraint = model.addConstr(lhs >= rhs, name=row.name)
        constraints[row.name] = constraint
    model.setObjective(
        gp.quicksum(
            spec.objective_coefficient * variables[spec.key]
            for spec in ir.variables
        ),
        gp.GRB.MINIMIZE,
    )
    model.update()
    return CompiledRecourse(ir, dict(first_stage), realization, model, variables, constraints)

