"""Exact mutually-exclusive partition fallback for a formal adversary."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Mapping, Sequence

import gurobipy as gp
import numpy as np

from port_h2_certificate.compilers.compile_adversary import (
    AdversaryDualTemplate,
    compile_adversary,
)
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class PartitionLeaf:
    fixed_primary: Mapping[str, int]
    status: str
    objective: float | None
    objective_bound: float | None
    selector: Mapping[str, int]
    known_lower_bound: float | None = None
    lower_bound_consistent: bool | None = None
    full_known_scenario_start_used: bool | None = None


@dataclass(frozen=True)
class PartitionResult:
    status: str
    coverage_complete: bool
    objective: float | None
    objective_bound: float | None
    selector: Mapping[str, int]
    realization: Mapping[str, np.ndarray]
    leaves: tuple[PartitionLeaf, ...]


def _partition_feasibility(
    bundle: UncertaintyBundle, assignment: Mapping[str, int]
) -> int:
    model = gp.Model("partition_bundle_feasibility")
    model.Params.OutputFlag = 0
    model.Params.DualReductions = 0
    variables = {
        key: model.addVar(vtype=gp.GRB.BINARY, name=key)
        for key in bundle.selector_keys
    }
    for key, value in assignment.items():
        model.addConstr(variables[key] == value, name=f"partition_fix[{key}]")
    for row in bundle.constraints:
        expression = gp.quicksum(
            coefficient * variables[key]
            for key, coefficient in row.coefficients.items()
        )
        if row.sense == "le":
            model.addConstr(expression <= row.rhs)
        elif row.sense == "ge":
            model.addConstr(expression >= row.rhs)
        else:
            model.addConstr(expression == row.rhs)
    model.setObjective(0.0, gp.GRB.MINIMIZE)
    model.optimize()
    return model.Status


def solve_exact_partition(
    dual_template: AdversaryDualTemplate,
    bundle: UncertaintyBundle,
    branch_keys: Sequence[str],
    *,
    known_scenario_values: Sequence[tuple[Mapping[str, int], float]] = (),
    known_scenario_tolerance: float = 1e-7,
) -> PartitionResult:
    keys = tuple(branch_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("partition branch keys must be unique")
    invalid = set(keys) - set(bundle.primary_selector_keys)
    if invalid:
        raise ValueError(
            f"partition may branch only on primary selectors: {sorted(invalid)}"
        )
    if known_scenario_tolerance < 0.0:
        raise ValueError("known-scenario tolerance must be nonnegative")
    known: list[tuple[dict[str, int], float]] = []
    expected_selector_keys = set(bundle.selector_keys)
    for selector, value in known_scenario_values:
        full_selector = {key: int(item) for key, item in selector.items()}
        if set(full_selector) != expected_selector_keys or any(
            item not in {0, 1} for item in full_selector.values()
        ):
            raise ValueError("known scenario must contain every binary selector")
        objective = float(value)
        if not math.isfinite(objective):
            raise ValueError("known scenario objective must be finite")
        bundle.evaluate(full_selector)
        known.append((full_selector, objective))

    leaves: list[PartitionLeaf] = []
    best = None
    for values in product((0, 1), repeat=len(keys)):
        assignment = dict(zip(keys, values, strict=True))
        feasibility_status = _partition_feasibility(bundle, assignment)
        if feasibility_status == gp.GRB.INFEASIBLE:
            leaves.append(PartitionLeaf(assignment, "INFEASIBLE", None, None, {}))
            continue
        if feasibility_status != gp.GRB.OPTIMAL:
            leaves.append(
                PartitionLeaf(
                    assignment,
                    f"UNRESOLVED_{feasibility_status}",
                    None,
                    None,
                    {},
                )
            )
            return PartitionResult(
                "unresolved",
                False,
                None,
                None,
                {},
                {},
                tuple(leaves),
            )

        compatible_known = [
            (selector, objective)
            for selector, objective in known
            if all(selector[key] == value for key, value in assignment.items())
        ]
        known_lower_bound = (
            max(objective for _, objective in compatible_known)
            if compatible_known
            else None
        )
        full_start_values: dict[str, float] | None = None
        if known_lower_bound is not None:
            best_known_selector = max(
                compatible_known, key=lambda item: item[1]
            )[0]
            fixed_known = compile_adversary(dual_template, bundle)
            for key, variable in fixed_known.selector_variables.items():
                fixed_known.model.addConstr(
                    variable == best_known_selector[key],
                    name=f"known_start_fix[{key}]",
                )
            fixed_known.model.update()
            try:
                fixed_known_result = fixed_known.solve()
            except RuntimeError:
                leaves.append(
                    PartitionLeaf(
                        assignment,
                        f"UNRESOLVED_KNOWN_START_{fixed_known.model.Status}",
                        None,
                        None,
                        {},
                        known_lower_bound=known_lower_bound,
                        full_known_scenario_start_used=False,
                    )
                )
                return PartitionResult(
                    "unresolved_known_scenario_start",
                    False,
                    None,
                    None,
                    {},
                    {},
                    tuple(leaves),
                )
            if (
                abs(fixed_known_result.objective - known_lower_bound)
                > known_scenario_tolerance
            ):
                leaves.append(
                    PartitionLeaf(
                        assignment,
                        "KNOWN_START_OBJECTIVE_MISMATCH",
                        fixed_known_result.objective,
                        fixed_known_result.objective_bound,
                        fixed_known_result.selector,
                        known_lower_bound=known_lower_bound,
                        lower_bound_consistent=False,
                        full_known_scenario_start_used=False,
                    )
                )
                return PartitionResult(
                    "unresolved_known_scenario_start_mismatch",
                    False,
                    None,
                    None,
                    {},
                    {},
                    tuple(leaves),
                )
            full_start_values = {
                variable.VarName: float(variable.X)
                for variable in fixed_known.model.getVars()
            }
        compiled = compile_adversary(dual_template, bundle)
        for key, value in assignment.items():
            compiled.model.addConstr(
                compiled.selector_variables[key] == value,
                name=f"partition_fix[{key}]",
            )
        if known_lower_bound is not None:
            assert full_start_values is not None
            for variable in compiled.model.getVars():
                if variable.VarName in full_start_values:
                    variable.Start = full_start_values[variable.VarName]
            compiled.model.addConstr(
                compiled.model.getObjective()
                >= known_lower_bound - known_scenario_tolerance,
                name="known_scenario_objective_floor",
            )
        compiled.model.update()
        try:
            result = compiled.solve()
        except RuntimeError:
            leaves.append(
                PartitionLeaf(
                    assignment,
                    f"UNRESOLVED_{compiled.model.Status}",
                    None,
                    None,
                    {},
                    known_lower_bound=known_lower_bound,
                    full_known_scenario_start_used=(
                        full_start_values is not None
                    ),
                )
            )
            return PartitionResult(
                "unresolved",
                False,
                None,
                None,
                {},
                {},
                tuple(leaves),
            )
        lower_bound_consistent = (
            known_lower_bound is None
            or result.objective + known_scenario_tolerance
            >= known_lower_bound
        )
        leaf = PartitionLeaf(
            assignment,
            "OPTIMAL" if lower_bound_consistent else "LOWER_BOUND_MISMATCH",
            result.objective,
            result.objective_bound,
            result.selector,
            known_lower_bound=known_lower_bound,
            lower_bound_consistent=lower_bound_consistent,
            full_known_scenario_start_used=(full_start_values is not None),
        )
        leaves.append(leaf)
        if not lower_bound_consistent:
            return PartitionResult(
                "unresolved_known_scenario_lower_bound",
                False,
                None,
                None,
                {},
                {},
                tuple(leaves),
            )
        if best is None or result.objective > best.objective:
            best = result

    if best is None:
        return PartitionResult(
            "infeasible_bundle", True, None, None, {}, {}, tuple(leaves)
        )
    return PartitionResult(
        status="OPTIMAL",
        coverage_complete=True,
        objective=best.objective,
        objective_bound=max(
            leaf.objective_bound
            for leaf in leaves
            if leaf.status == "OPTIMAL" and leaf.objective_bound is not None
        ),
        selector=best.selector,
        realization=best.realization,
        leaves=tuple(leaves),
    )
