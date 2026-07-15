"""Compile the finite-scenario two-stage master from the shared IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import gurobipy as gp

from port_h2_certificate.compilers.common import (
    Realization,
    configure_formal_lp,
    parse_output_reference,
    validate_realization_references,
)
from port_h2_certificate.recourse_ir import VariableKey
from port_h2_certificate.two_stage_ir import TwoStageIR


@dataclass(frozen=True)
class MasterResult:
    status: int
    objective: float
    theta: float
    first_stage: Mapping[VariableKey, float]
    scenario_recourse: tuple[Mapping[VariableKey, float], ...]


@dataclass
class CompiledMaster:
    two_stage_ir: TwoStageIR
    realization_pool: tuple[Realization, ...]
    model: gp.Model
    first_stage_variables: Mapping[VariableKey, gp.Var]
    theta: gp.Var
    scenario_variables: tuple[Mapping[VariableKey, gp.Var], ...]

    def solve(self) -> MasterResult:
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"finite-scenario master did not reach GRB.OPTIMAL: {self.model.Status}"
            )
        return MasterResult(
            status=self.model.Status,
            objective=float(self.model.ObjVal),
            theta=float(self.theta.X),
            first_stage={
                key: variable.X
                for key, variable in self.first_stage_variables.items()
            },
            scenario_recourse=tuple(
                {key: variable.X for key, variable in block.items()}
                for block in self.scenario_variables
            ),
        )


def _affine_rhs(row, first_variables, realization):
    rhs = gp.LinExpr(row.rhs_constant)
    for key, coefficient in row.first_stage_coefficients.items():
        rhs += coefficient * first_variables[key]
    for reference, coefficient in row.uncertain_output_coefficients.items():
        output_key, period = parse_output_reference(reference)
        rhs += coefficient * float(realization[output_key][period])
    return rhs


def compile_master(
    two_stage_ir: TwoStageIR, realization_pool: Sequence[Realization]
) -> CompiledMaster:
    if not realization_pool:
        raise ValueError("finite-scenario master requires at least one realization")
    two_stage_ir.validate()
    for realization in realization_pool:
        validate_realization_references(
            two_stage_ir.recourse.constraints, realization
        )

    model = gp.Model("finite_scenario_master")
    configure_formal_lp(model)
    first_variables = {
        spec.key: model.addVar(
            lb=spec.lower_bound,
            ub=gp.GRB.INFINITY if spec.upper_bound is None else spec.upper_bound,
            vtype=gp.GRB.CONTINUOUS,
            name=f"{spec.key[0]}[{spec.key[1]}]",
        )
        for spec in two_stage_ir.first_stage.variables
    }
    theta = model.addVar(
        lb=-gp.GRB.INFINITY,
        ub=gp.GRB.INFINITY,
        vtype=gp.GRB.CONTINUOUS,
        name="theta",
    )
    for row in two_stage_ir.first_stage.constraints:
        lhs = gp.quicksum(
            coefficient * first_variables[key]
            for key, coefficient in row.coefficients.items()
        )
        if row.sense == "eq":
            model.addConstr(lhs == row.rhs_constant, name=row.name)
        else:
            model.addConstr(lhs >= row.rhs_constant, name=row.name)

    scenario_blocks: list[Mapping[VariableKey, gp.Var]] = []
    for scenario_index, realization in enumerate(realization_pool):
        block = {
            spec.key: model.addVar(
                lb=spec.lower_bound,
                ub=gp.GRB.INFINITY if spec.upper_bound is None else spec.upper_bound,
                vtype=gp.GRB.CONTINUOUS,
                name=f"scenario[{scenario_index}].{spec.key[0]}[{spec.key[1]}]",
            )
            for spec in two_stage_ir.recourse.variables
        }
        scenario_blocks.append(block)
        for row in two_stage_ir.recourse.constraints:
            lhs = gp.quicksum(
                coefficient * block[key]
                for key, coefficient in row.recourse_coefficients.items()
            )
            rhs = _affine_rhs(row, first_variables, realization)
            name = f"scenario[{scenario_index}].{row.name}"
            if row.sense == "eq":
                model.addConstr(lhs == rhs, name=name)
            else:
                model.addConstr(lhs >= rhs, name=name)
        recourse_cost = gp.quicksum(
            spec.objective_coefficient * block[spec.key]
            for spec in two_stage_ir.recourse.variables
        )
        model.addConstr(
            theta >= recourse_cost,
            name=f"scenario[{scenario_index}].theta_epigraph",
        )

    first_cost = gp.quicksum(
        spec.objective_coefficient * first_variables[spec.key]
        for spec in two_stage_ir.first_stage.variables
    )
    model.setObjective(first_cost + theta, gp.GRB.MINIMIZE)
    model.update()
    return CompiledMaster(
        two_stage_ir,
        tuple(realization_pool),
        model,
        first_variables,
        theta,
        tuple(scenario_blocks),
    )
