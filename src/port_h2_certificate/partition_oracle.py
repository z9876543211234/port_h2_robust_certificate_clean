"""Exact mutually-exclusive partition fallback for a formal adversary."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
) -> PartitionResult:
    keys = tuple(branch_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("partition branch keys must be unique")
    invalid = set(keys) - set(bundle.primary_selector_keys)
    if invalid:
        raise ValueError(
            f"partition may branch only on primary selectors: {sorted(invalid)}"
        )

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

        compiled = compile_adversary(dual_template, bundle)
        for key, value in assignment.items():
            compiled.model.addConstr(
                compiled.selector_variables[key] == value,
                name=f"partition_fix[{key}]",
            )
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
        leaf = PartitionLeaf(
            assignment,
            "OPTIMAL",
            result.objective,
            result.objective_bound,
            result.selector,
        )
        leaves.append(leaf)
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
