"""Exact fixed-realization Phase-I feasibility LP generated from recourse IR."""

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
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey


@dataclass(frozen=True)
class Phase1Result:
    status: int
    objective: float
    variables: Mapping[VariableKey, float]
    row_slacks: Mapping[str, float]


@dataclass
class CompiledPhase1:
    model: gp.Model
    variables: Mapping[VariableKey, gp.Var]
    row_slack_variables: Mapping[str, tuple[gp.Var, ...]]

    def solve(self) -> Phase1Result:
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"fixed-scenario Phase-I did not reach GRB.OPTIMAL: {self.model.Status}"
            )
        return Phase1Result(
            status=self.model.Status,
            objective=float(self.model.ObjVal),
            variables={key: variable.X for key, variable in self.variables.items()},
            row_slacks={
                name: sum(variable.X for variable in variables)
                for name, variables in self.row_slack_variables.items()
            },
        )


def compile_phase1(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    realization: Realization,
    sigma: float | Mapping[str, float] = 1.0,
) -> CompiledPhase1:
    validate_realization_references(ir.constraints, realization)
    if isinstance(sigma, Mapping):
        missing = {row.name for row in ir.constraints} - set(sigma)
        if missing:
            raise KeyError(f"missing Phase-I sigma values: {sorted(missing)}")
        weights = {row.name: float(sigma[row.name]) for row in ir.constraints}
    else:
        weights = {row.name: float(sigma) for row in ir.constraints}
    if any(value <= 0.0 for value in weights.values()):
        raise ValueError("all Phase-I sigma values must be positive")

    model = gp.Model("fixed_scenario_phase1")
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
    row_slacks: dict[str, tuple[gp.Var, ...]] = {}
    objective = gp.LinExpr()
    for row in ir.constraints:
        lhs = gp.quicksum(
            coefficient * variables[key]
            for key, coefficient in row.recourse_coefficients.items()
        )
        rhs = evaluate_row_rhs(row, first_stage, realization)
        if row.sense == "eq":
            positive = model.addVar(lb=0.0, name=f"phase1_plus[{row.name}]")
            negative = model.addVar(lb=0.0, name=f"phase1_minus[{row.name}]")
            model.addConstr(lhs + positive - negative == rhs, name=row.name)
            row_slacks[row.name] = (positive, negative)
            objective += (positive + negative) / weights[row.name]
        else:
            slack = model.addVar(lb=0.0, name=f"phase1[{row.name}]")
            model.addConstr(lhs + slack >= rhs, name=row.name)
            row_slacks[row.name] = (slack,)
            objective += slack / weights[row.name]
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.update()
    return CompiledPhase1(model, variables, row_slacks)

