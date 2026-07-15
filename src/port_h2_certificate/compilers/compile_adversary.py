"""Exact dual-only MILP adversary using indicators and no empirical Big-M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import gurobipy as gp
import numpy as np

from port_h2_certificate.compilers.common import parse_output_reference
from typing import Protocol

from port_h2_certificate.compilers.compile_dual import UpperBoundRow
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class AdversaryResult:
    status: int
    objective: float
    objective_bound: float
    mip_gap: float
    selector: Mapping[str, int]
    realization: Mapping[str, np.ndarray]
    row_duals: Mapping[str, float]
    upper_bound_duals: Mapping[tuple[str, int], float]


@dataclass
class CompiledAdversary:
    template: "AdversaryDualTemplate"
    bundle: UncertaintyBundle
    model: gp.Model
    selector_variables: Mapping[str, gp.Var]
    row_duals: Mapping[str, gp.Var]
    upper_bound_duals: Mapping[tuple[str, int], gp.Var]
    active_selector_keys: tuple[str, ...]

    def solve(self) -> AdversaryResult:
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"formal {self.template.template_kind} adversary did not reach "
                f"GRB.OPTIMAL: {self.model.Status}"
            )
        selector = {
            key: int(round(variable.X))
            for key, variable in self.selector_variables.items()
        }
        return AdversaryResult(
            status=self.model.Status,
            objective=float(self.model.ObjVal),
            objective_bound=float(self.model.ObjBound),
            mip_gap=float(self.model.MIPGap),
            selector=selector,
            realization=self.bundle.evaluate(selector),
            row_duals={name: variable.X for name, variable in self.row_duals.items()},
            upper_bound_duals={
                key: variable.X for key, variable in self.upper_bound_duals.items()
            },
        )


class AdversaryDualTemplate(Protocol):
    ir: RecourseIR
    first_stage: Mapping[VariableKey, float]
    lower_bounds: Mapping[VariableKey, float]
    upper_bound_rows: tuple[UpperBoundRow, ...]
    objective_constant: float
    template_kind: str

    def row_dual_bounds(self, row) -> tuple[float, float]: ...

    def dual_constraint_rhs(self, spec) -> float: ...


def _uncertain_row_columns(template: AdversaryDualTemplate, bundle: UncertaintyBundle):
    selector_count = len(bundle.selector_keys)
    columns: dict[str, np.ndarray] = {}
    bases: dict[str, float] = {}
    for row in template.ir.constraints:
        base = row.rhs_constant
        for key, coefficient in row.first_stage_coefficients.items():
            if key not in template.first_stage:
                raise KeyError(f"missing first-stage value: {key}")
            base += coefficient * float(template.first_stage[key])
        coefficients = np.zeros(selector_count, dtype=float)
        for reference, multiplier in row.uncertain_output_coefficients.items():
            output_key, period = parse_output_reference(reference)
            output = bundle.outputs[output_key]
            base += multiplier * output.nominal[period]
            matrix_row = bundle.output_matrix(output_key).to_scipy().getrow(period)
            coefficients[matrix_row.indices] += multiplier * matrix_row.data
        base -= sum(
            coefficient * template.lower_bounds[key]
            for key, coefficient in row.recourse_coefficients.items()
        )
        bases[row.name] = float(base)
        columns[row.name] = coefficients
    return bases, columns


def compile_adversary(
    dual_template: AdversaryDualTemplate, joint_bundle: UncertaintyBundle
) -> CompiledAdversary:
    joint_bundle.validate()
    model = gp.Model(f"exact_dual_only_{dual_template.template_kind}_adversary")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.FeasibilityTol = 1e-8
    model.Params.IntFeasTol = 1e-9
    model.Params.OptimalityTol = 1e-8
    model.Params.NumericFocus = 2
    model.Params.InfUnbdInfo = 1
    model.Params.MIPFocus = 3

    selectors = {
        key: model.addVar(vtype=gp.GRB.BINARY, name=key)
        for key in joint_bundle.selector_keys
    }
    for row in joint_bundle.constraints:
        expression = gp.LinExpr()
        for key, coefficient in row.coefficients.items():
            expression += coefficient * selectors[key]
        if row.sense == "le":
            model.addConstr(expression <= row.rhs, name=f"uncertainty.{row.name}")
        elif row.sense == "ge":
            model.addConstr(expression >= row.rhs, name=f"uncertainty.{row.name}")
        else:
            model.addConstr(expression == row.rhs, name=f"uncertainty.{row.name}")

    row_duals = {
        row.name: model.addVar(
            lb=dual_template.row_dual_bounds(row)[0],
            ub=dual_template.row_dual_bounds(row)[1],
            vtype=gp.GRB.CONTINUOUS,
            name=f"dual[{row.name}]",
        )
        for row in dual_template.ir.constraints
    }
    upper_duals = {
        row.variable_key: model.addVar(
            lb=0.0,
            ub=gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name=f"dual_upper[{row.variable_key[0]}[{row.variable_key[1]}]]",
        )
        for row in dual_template.upper_bound_rows
    }
    for spec in dual_template.ir.variables:
        expression = gp.quicksum(
            row.recourse_coefficients.get(spec.key, 0.0) * row_duals[row.name]
            for row in dual_template.ir.constraints
        )
        if spec.key in upper_duals:
            expression -= upper_duals[spec.key]
        model.addConstr(
            expression <= dual_template.dual_constraint_rhs(spec),
            name=f"dual_feasibility[{spec.key[0]}[{spec.key[1]}]]",
        )

    bases, columns = _uncertain_row_columns(dual_template, joint_bundle)
    objective = gp.LinExpr(dual_template.objective_constant)
    for row in dual_template.ir.constraints:
        objective += bases[row.name] * row_duals[row.name]
    for row in dual_template.upper_bound_rows:
        objective -= row.width * upper_duals[row.variable_key]

    active: list[str] = []
    for selector_index, selector_key in enumerate(joint_bundle.selector_keys):
        g_expression = gp.quicksum(
            columns[row.name][selector_index] * row_duals[row.name]
            for row in dual_template.ir.constraints
            if columns[row.name][selector_index] != 0.0
        )
        if g_expression.size() == 0:
            continue
        active.append(selector_key)
        product = model.addVar(
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name=f"selector_dual_product[{selector_key}]",
        )
        model.addGenConstrIndicator(
            selectors[selector_key],
            0,
            product == 0.0,
            name=f"product_zero[{selector_key}]",
        )
        model.addGenConstrIndicator(
            selectors[selector_key],
            1,
            product == g_expression,
            name=f"product_active[{selector_key}]",
        )
        objective += product

    model.setObjective(objective, gp.GRB.MAXIMIZE)
    model.update()
    return CompiledAdversary(
        template=dual_template,
        bundle=joint_bundle,
        model=model,
        selector_variables=selectors,
        row_duals=row_duals,
        upper_bound_duals=upper_duals,
        active_selector_keys=tuple(active),
    )
