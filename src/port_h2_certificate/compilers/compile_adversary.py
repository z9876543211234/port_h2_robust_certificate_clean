"""Exact dual-only MILP adversary using indicators and no empirical Big-M."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    primal_variables: Mapping[VariableKey, gp.Var]
    product_variables: Mapping[str, gp.Var]
    product_row_coefficients: Mapping[str, Mapping[str, float]]
    active_selector_keys: tuple[str, ...]

    def set_full_start(
        self,
        selector: Mapping[str, int],
        row_duals: Mapping[str, float],
        upper_bound_duals: Mapping[tuple[str, int], float],
        primal_values: Mapping[VariableKey, float] | None = None,
    ) -> int:
        if set(selector) != set(self.selector_variables) or any(
            int(value) not in {0, 1} for value in selector.values()
        ):
            raise ValueError("full adversary start requires every binary selector")
        if set(row_duals) != set(self.row_duals):
            raise ValueError("full adversary start requires every row dual")
        if set(upper_bound_duals) != set(self.upper_bound_duals):
            raise ValueError("full adversary start requires every upper-bound dual")
        if self.primal_variables and (
            primal_values is None
            or set(primal_values) != set(self.primal_variables)
        ):
            raise ValueError("strengthened adversary start requires every primal value")

        for key, variable in self.selector_variables.items():
            variable.Start = int(selector[key])
        for name, variable in self.row_duals.items():
            variable.Start = float(row_duals[name])
        for key, variable in self.upper_bound_duals.items():
            variable.Start = float(upper_bound_duals[key])
        for key, variable in self.primal_variables.items():
            assert primal_values is not None
            variable.Start = float(primal_values[key])
        for key, variable in self.product_variables.items():
            g_value = sum(
                coefficient * float(row_duals[row_name])
                for row_name, coefficient in self.product_row_coefficients[
                    key
                ].items()
            )
            variable.Start = int(selector[key]) * g_value
        self.model.update()
        return (
            len(self.selector_variables)
            + len(self.row_duals)
            + len(self.upper_bound_duals)
            + len(self.primal_variables)
            + len(self.product_variables)
        )

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


def _raw_row_bases(
    template: AdversaryDualTemplate, normalized_bases: Mapping[str, float]
) -> dict[str, float]:
    return {
        row.name: normalized_bases[row.name]
        + sum(
            coefficient * template.lower_bounds[key]
            for key, coefficient in row.recourse_coefficients.items()
        )
        for row in template.ir.constraints
    }


def _relaxed_rhs_range(base: float, coefficients: np.ndarray) -> tuple[float, float]:
    """Safe interval over z in [0, 1], deliberately ignoring Bundle coupling."""

    return (
        float(base + coefficients[coefficients < 0.0].sum()),
        float(base + coefficients[coefficients > 0.0].sum()),
    )


def infer_optimal_primal_upper_bounds(
    template: AdversaryDualTemplate,
    bundle: UncertaintyBundle,
    normalized_bases: Mapping[str, float],
    columns: Mapping[str, np.ndarray],
) -> dict[VariableKey, float]:
    """Infer finite bounds for positive-cost epigraph slacks.

    The added bounds preserve at least one optimum for every integer
    uncertainty realization.  They are used only in the primal block that
    strengthens the exact adversary; the source recourse IR is unchanged.
    """

    specs = {spec.key: spec for spec in template.ir.variables}
    upper_bounds = {
        spec.key: float(spec.upper_bound)
        for spec in template.ir.variables
        if spec.upper_bound is not None
    }
    raw_bases = _raw_row_bases(template, normalized_bases)

    # Direct rows of the form a*x >= rhs(z), a < 0, provide a true upper
    # bound on x.  Relaxing every selector independently gives a conservative
    # bound and therefore never removes a feasible integer realization.
    for row in template.ir.constraints:
        if row.sense != "ge" or len(row.recourse_coefficients) != 1:
            continue
        key, coefficient = next(iter(row.recourse_coefficients.items()))
        if coefficient >= 0.0:
            continue
        rhs_min, _ = _relaxed_rhs_range(
            raw_bases[row.name], columns[row.name]
        )
        candidate = rhs_min / coefficient
        candidate = max(float(specs[key].lower_bound), float(candidate))
        if key not in upper_bounds or candidate < upper_bounds[key]:
            upper_bounds[key] = candidate

    occurrences: dict[VariableKey, list] = {key: [] for key in specs}
    for row in template.ir.constraints:
        for key in row.recourse_coefficients:
            occurrences[key].append(row)

    # A positive-cost variable that appears only with a positive coefficient
    # in >= rows is an epigraph slack.  Its minimal optimal value is no larger
    # than the worst interval requirement computed from the other variables.
    for spec in template.ir.variables:
        if spec.objective_coefficient <= 0.0 or spec.key in upper_bounds:
            continue
        rows = occurrences[spec.key]
        if not rows or any(
            row.sense != "ge"
            or row.recourse_coefficients[spec.key] <= 0.0
            for row in rows
        ):
            continue
        candidates = [float(spec.lower_bound)]
        bounded = True
        for row in rows:
            coefficient = row.recourse_coefficients[spec.key]
            _, rhs_max = _relaxed_rhs_range(
                raw_bases[row.name], columns[row.name]
            )
            other_minimum = 0.0
            for key, value in row.recourse_coefficients.items():
                if key == spec.key:
                    continue
                other = specs[key]
                if value >= 0.0:
                    other_minimum += value * other.lower_bound
                elif key in upper_bounds:
                    other_minimum += value * upper_bounds[key]
                else:
                    bounded = False
                    break
            if not bounded:
                break
            candidates.append((rhs_max - other_minimum) / coefficient)
        if bounded:
            candidate = max(candidates)
            upper_bounds[spec.key] = candidate + max(
                1e-9, abs(candidate) * 1e-9
            )

    unresolved = [
        spec.key
        for spec in template.ir.variables
        if spec.objective_coefficient > 0.0 and spec.key not in upper_bounds
    ]
    if unresolved:
        raise ValueError(
            "cannot build a bounded primal adversary objective; unresolved "
            f"positive-cost variables: {unresolved[:8]}"
        )
    if any(not math.isfinite(value) for value in upper_bounds.values()):
        raise ValueError("inferred primal upper bounds must be finite")
    return upper_bounds


def infer_optimal_recourse_objective_upper_bound(
    template: AdversaryDualTemplate,
    bundle: UncertaintyBundle,
) -> float:
    """Return a finite cap that preserves every integer-scenario optimum."""

    normalized_bases, columns = _uncertain_row_columns(template, bundle)
    upper_bounds = infer_optimal_primal_upper_bounds(
        template, bundle, normalized_bases, columns
    )
    value = 0.0
    for spec in template.ir.variables:
        if spec.objective_coefficient > 0.0:
            value += spec.objective_coefficient * upper_bounds[spec.key]
        elif spec.objective_coefficient < 0.0:
            value += spec.objective_coefficient * spec.lower_bound
    if not math.isfinite(value):
        raise ValueError("inferred recourse objective upper bound must be finite")
    return float(value + max(1e-7, abs(value) * 1e-9))


def compile_adversary(
    dual_template: AdversaryDualTemplate,
    joint_bundle: UncertaintyBundle,
    *,
    strengthen_with_primal: bool = False,
    objective_upper_bound: float | None = None,
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
    product_variables: dict[str, gp.Var] = {}
    product_row_coefficients: dict[str, dict[str, float]] = {}
    for selector_index, selector_key in enumerate(joint_bundle.selector_keys):
        row_coefficients = {
            row.name: float(columns[row.name][selector_index])
            for row in dual_template.ir.constraints
            if columns[row.name][selector_index] != 0.0
        }
        g_expression = gp.quicksum(
            coefficient * row_duals[row_name]
            for row_name, coefficient in row_coefficients.items()
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
        product_variables[selector_key] = product
        product_row_coefficients[selector_key] = row_coefficients
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

    if objective_upper_bound is not None:
        if not math.isfinite(objective_upper_bound):
            raise ValueError("adversary objective upper bound must be finite")
        model.addConstr(
            objective <= float(objective_upper_bound),
            name="valid_recourse_objective_upper_bound",
        )

    primal_variables: dict[VariableKey, gp.Var] = {}
    if strengthen_with_primal:
        primal_upper_bounds = infer_optimal_primal_upper_bounds(
            dual_template, joint_bundle, bases, columns
        )
        primal_variables = {
            spec.key: model.addVar(
                lb=spec.lower_bound,
                ub=primal_upper_bounds.get(spec.key, gp.GRB.INFINITY),
                vtype=gp.GRB.CONTINUOUS,
                name=f"primal[{spec.key[0]}[{spec.key[1]}]]",
            )
            for spec in dual_template.ir.variables
        }
        raw_bases = _raw_row_bases(dual_template, bases)
        for row in dual_template.ir.constraints:
            lhs = gp.quicksum(
                coefficient * primal_variables[key]
                for key, coefficient in row.recourse_coefficients.items()
            )
            rhs = raw_bases[row.name] + gp.quicksum(
                float(columns[row.name][selector_index])
                * selectors[selector_key]
                for selector_index, selector_key in enumerate(
                    joint_bundle.selector_keys
                )
                if columns[row.name][selector_index] != 0.0
            )
            if row.sense == "eq":
                model.addConstr(
                    lhs == rhs, name=f"primal_feasibility[{row.name}]"
                )
            else:
                model.addConstr(
                    lhs >= rhs, name=f"primal_feasibility[{row.name}]"
                )
        primal_objective = gp.quicksum(
            spec.objective_coefficient * primal_variables[spec.key]
            for spec in dual_template.ir.variables
        )
        model.addConstr(
            primal_objective == objective,
            name="primal_dual_strong_duality",
        )
        model.setObjective(primal_objective, gp.GRB.MAXIMIZE)
    else:
        model.setObjective(objective, gp.GRB.MAXIMIZE)
    model.update()
    return CompiledAdversary(
        template=dual_template,
        bundle=joint_bundle,
        model=model,
        selector_variables=selectors,
        row_duals=row_duals,
        upper_bound_duals=upper_duals,
        primal_variables=primal_variables,
        product_variables=product_variables,
        product_row_coefficients=product_row_coefficients,
        active_selector_keys=tuple(active),
    )
