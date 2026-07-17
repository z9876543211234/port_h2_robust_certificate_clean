"""Exact mutually-exclusive partition fallback for a formal adversary."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Callable, Mapping, Sequence

import gurobipy as gp
import numpy as np

from port_h2_certificate.compilers.compile_adversary import (
    AdversaryDualTemplate,
    compile_adversary,
    infer_optimal_recourse_objective_upper_bound,
)
from port_h2_certificate.compilers.compile_recourse import compile_recourse
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
    witness_selector: Mapping[str, int] | None = None
    witness_primal_objective: float | None = None
    witness_dual_objective: float | None = None
    witness_primal_dual_gap: float | None = None
    witness_max_abs_residual: float | None = None
    full_witness_start_used: bool | None = None
    primal_dual_strengthening_used: bool | None = None
    runtime_seconds: float | None = None
    work: float | None = None
    node_count: float | None = None


@dataclass(frozen=True)
class PartitionEvent:
    leaf_index: int
    leaf_count: int
    stage: str
    fixed_primary: Mapping[str, int]
    leaf: PartitionLeaf | None = None


@dataclass(frozen=True)
class PartitionFeasibility:
    status: int
    selector: Mapping[str, int]


@dataclass(frozen=True)
class _CertifiedWitness:
    selector: Mapping[str, int]
    primal_objective: float
    dual_objective: float
    primal_dual_gap: float
    max_abs_residual: float
    row_duals: Mapping[str, float]
    upper_bound_duals: Mapping[tuple[str, int], float]
    primal_values: Mapping[tuple[str, int], float]


@dataclass(frozen=True)
class PartitionResult:
    status: str
    coverage_complete: bool
    objective: float | None
    objective_bound: float | None
    selector: Mapping[str, int]
    realization: Mapping[str, np.ndarray]
    leaves: tuple[PartitionLeaf, ...]


def build_ship_delay_partition_assignments(
    bundle: UncertaintyBundle,
) -> tuple[tuple[str, ...], tuple[Mapping[str, int], ...]]:
    """Enumerate exact feasible projections of the ship-delay sub-Bundle."""

    ship_keys = tuple(
        key
        for key in bundle.primary_selector_keys
        if key.startswith("ship.delay[")
    )
    if not ship_keys:
        raise ValueError("Bundle has no primary ship-delay selectors")
    ship_key_set = set(ship_keys)
    mixed_rows = [
        row.name
        for row in bundle.constraints
        if set(row.coefficients) & ship_key_set
        and set(row.coefficients) - ship_key_set
    ]
    if mixed_rows:
        raise ValueError(
            "semantic ship partition requires uncertainty constraints to "
            f"separate ship and non-ship selectors: {mixed_rows}"
        )
    group_rows = [
        row
        for row in bundle.constraints
        if row.name.startswith("ship.one_delay[")
    ]
    groups: list[tuple[str, ...]] = []
    grouped: set[str] = set()
    for row in group_rows:
        keys = tuple(key for key in ship_keys if key in row.coefficients)
        if (
            row.sense != "le"
            or abs(row.rhs - 1.0) > 1e-12
            or not keys
            or any(abs(row.coefficients[key] - 1.0) > 1e-12 for key in keys)
            or grouped & set(keys)
        ):
            raise ValueError(f"invalid ship one-delay row: {row.name}")
        groups.append(keys)
        grouped.update(keys)
    if grouped != ship_key_set:
        raise ValueError("ship one-delay rows must partition every ship selector")
    budget_rows = [
        row
        for row in bundle.constraints
        if row.name == "ship.delayed_ship_budget"
    ]
    if len(budget_rows) != 1:
        raise ValueError("Bundle must contain one delayed-ship budget row")
    budget_row = budget_rows[0]
    if (
        budget_row.sense != "le"
        or set(budget_row.coefficients) != ship_key_set
        or any(
            abs(budget_row.coefficients[key] - 1.0) > 1e-12
            for key in ship_keys
        )
        or abs(budget_row.rhs - round(budget_row.rhs)) > 1e-12
    ):
        raise ValueError("delayed-ship budget must be an integer cardinality bound")
    delayed_ship_budget = min(int(round(budget_row.rhs)), len(groups))

    ship_rows = [
        row
        for row in bundle.constraints
        if set(row.coefficients) & ship_key_set
    ]

    def satisfies(assignment: Mapping[str, int]) -> bool:
        for row in ship_rows:
            lhs = sum(
                coefficient * assignment[key]
                for key, coefficient in row.coefficients.items()
            )
            if row.sense == "le" and lhs > row.rhs + 1e-9:
                return False
            if row.sense == "ge" and lhs < row.rhs - 1e-9:
                return False
            if row.sense == "eq" and abs(lhs - row.rhs) > 1e-9:
                return False
        return True

    assignments: list[Mapping[str, int]] = []
    for delayed_count in range(delayed_ship_budget + 1):
        for selected_group_indexes in combinations(
            range(len(groups)), delayed_count
        ):
            selected_groups = [groups[index] for index in selected_group_indexes]
            for active_keys in product(*selected_groups) if selected_groups else [()]:
                assignment = {key: 0 for key in ship_keys}
                for key in active_keys:
                    assignment[key] = 1
                if satisfies(assignment):
                    assignments.append(assignment)
    assignment_keys = {
        tuple(assignment[key] for key in ship_keys)
        for assignment in assignments
    }
    if len(assignment_keys) != len(assignments):
        raise RuntimeError("semantic ship partition produced duplicate leaves")
    return ship_keys, tuple(assignments)


def _partition_feasibility(
    bundle: UncertaintyBundle, assignment: Mapping[str, int]
) -> PartitionFeasibility:
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
    selector = (
        {
            key: int(round(variable.X))
            for key, variable in variables.items()
        }
        if model.Status == gp.GRB.OPTIMAL
        else {}
    )
    return PartitionFeasibility(model.Status, selector)


def _certify_witness(
    dual_template: AdversaryDualTemplate,
    bundle: UncertaintyBundle,
    selector: Mapping[str, int],
    tolerance: float,
) -> _CertifiedWitness:
    realization = bundle.evaluate(selector)
    primal = compile_recourse(
        dual_template.ir,
        dual_template.first_stage,
        realization,
    ).solve()
    dual = dual_template.instantiate(realization).solve()
    gap = float(primal.objective - dual.objective)
    if abs(gap) > tolerance:
        raise RuntimeError(
            "partition witness primal-dual mismatch: "
            f"primal={primal.objective}, dual={dual.objective}, gap={gap}"
        )
    return _CertifiedWitness(
        selector=dict(selector),
        primal_objective=float(primal.objective),
        dual_objective=float(dual.objective),
        primal_dual_gap=gap,
        max_abs_residual=float(primal.max_abs_residual),
        row_duals=dict(dual.row_duals),
        upper_bound_duals=dict(dual.upper_bound_duals),
        primal_values=dict(primal.variables),
    )


def solve_exact_partition(
    dual_template: AdversaryDualTemplate,
    bundle: UncertaintyBundle,
    branch_keys: Sequence[str],
    *,
    known_scenario_values: Sequence[tuple[Mapping[str, int], float]] = (),
    known_scenario_tolerance: float = 1e-7,
    resume_leaves: Sequence[PartitionLeaf] = (),
    event_callback: Callable[[PartitionEvent], None] | None = None,
    partition_assignments: Sequence[Mapping[str, int]] | None = None,
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

    resume_by_assignment: dict[tuple[int, ...], PartitionLeaf] = {}
    for leaf in resume_leaves:
        if set(leaf.fixed_primary) != set(keys):
            raise ValueError("resumed partition leaf does not match branch keys")
        assignment_key = tuple(int(leaf.fixed_primary[key]) for key in keys)
        if assignment_key in resume_by_assignment:
            raise ValueError("resumed partition leaves must be unique")
        if leaf.status not in {"OPTIMAL", "INFEASIBLE"}:
            raise ValueError("only resolved partition leaves may be resumed")
        if leaf.status == "OPTIMAL" and (
            leaf.objective is None
            or leaf.objective_bound is None
            or set(leaf.selector) != expected_selector_keys
        ):
            raise ValueError("resumed optimal leaf is incomplete")
        resume_by_assignment[assignment_key] = leaf

    def emit(event: PartitionEvent) -> None:
        if event_callback is not None:
            event_callback(event)

    leaves: list[PartitionLeaf] = []
    best_leaf: PartitionLeaf | None = None
    if partition_assignments is None:
        assignments = tuple(product((0, 1), repeat=len(keys)))
    else:
        assignments = tuple(
            tuple(int(assignment[key]) for key in keys)
            for assignment in partition_assignments
        )
        if any(
            set(assignment) != set(keys)
            or any(int(value) not in {0, 1} for value in assignment.values())
            for assignment in partition_assignments
        ):
            raise ValueError(
                "explicit partition assignments must contain every branch key"
            )
        if len(set(assignments)) != len(assignments):
            raise ValueError("explicit partition assignments must be unique")
    leaf_count = len(assignments)
    known_selector_keys = {
        tuple(selector[key] for key in bundle.selector_keys)
        for selector, _ in known
    }
    objective_upper_bound = infer_optimal_recourse_objective_upper_bound(
        dual_template, bundle
    )
    compiled = None
    for leaf_index, values in enumerate(assignments):
        assignment = dict(zip(keys, values, strict=True))
        resumed = resume_by_assignment.get(tuple(values))
        if resumed is not None:
            leaves.append(resumed)
            emit(
                PartitionEvent(
                    leaf_index,
                    leaf_count,
                    "RESUMED",
                    assignment,
                    resumed,
                )
            )
            if resumed.status == "OPTIMAL" and (
                best_leaf is None or resumed.objective > best_leaf.objective
            ):
                best_leaf = resumed
            continue
        emit(
            PartitionEvent(
                leaf_index,
                leaf_count,
                "FEASIBILITY_STARTED",
                assignment,
            )
        )
        feasibility = _partition_feasibility(bundle, assignment)
        if feasibility.status == gp.GRB.INFEASIBLE:
            leaf = PartitionLeaf(assignment, "INFEASIBLE", None, None, {})
            leaves.append(leaf)
            emit(
                PartitionEvent(
                    leaf_index, leaf_count, "COMPLETED", assignment, leaf
                )
            )
            continue
        if feasibility.status != gp.GRB.OPTIMAL:
            leaf = PartitionLeaf(
                assignment,
                f"UNRESOLVED_{feasibility.status}",
                None,
                None,
                {},
            )
            leaves.append(leaf)
            emit(
                PartitionEvent(
                    leaf_index, leaf_count, "COMPLETED", assignment, leaf
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
        candidate_selectors = [dict(feasibility.selector)]
        candidate_selectors.extend(selector for selector, _ in compatible_known)
        unique_candidates: dict[tuple[int, ...], Mapping[str, int]] = {}
        for selector in candidate_selectors:
            selector_key = tuple(selector[key] for key in bundle.selector_keys)
            unique_candidates.setdefault(selector_key, selector)
        witnesses = [
            _certify_witness(
                dual_template,
                bundle,
                selector,
                known_scenario_tolerance,
            )
            for selector in unique_candidates.values()
        ]
        expected_by_selector = {
            tuple(selector[key] for key in bundle.selector_keys): objective
            for selector, objective in compatible_known
        }
        for witness in witnesses:
            selector_key = tuple(
                witness.selector[key] for key in bundle.selector_keys
            )
            if selector_key in expected_by_selector and (
                abs(
                    witness.primal_objective
                    - expected_by_selector[selector_key]
                )
                > known_scenario_tolerance
            ):
                raise RuntimeError(
                    "known partition scenario objective disagrees with primal replay"
                )
        compatible_known_keys = set(expected_by_selector)
        compatible_known_witnesses = [
            witness
            for witness in witnesses
            if tuple(
                witness.selector[key] for key in bundle.selector_keys
            )
            in compatible_known_keys
        ]
        # Preserve the existing known-scenario replay contract whenever a
        # compatible C&CG scenario exists.  Other feasible leaves use the
        # automatically generated certified witness.
        best_witness = max(
            compatible_known_witnesses or witnesses,
            key=lambda item: item.primal_objective,
        )
        known_lower_bound = best_witness.primal_objective
        if compiled is None:
            compiled = compile_adversary(
                dual_template,
                bundle,
                objective_upper_bound=objective_upper_bound,
            )
        else:
            compiled.model.reset(0)
        temporary_constraints = []
        for key, value in assignment.items():
            temporary_constraints.append(
                compiled.model.addConstr(
                    compiled.selector_variables[key] == value,
                    name=f"partition_fix[{key}]",
                )
            )
        compiled.set_full_start(
            best_witness.selector,
            best_witness.row_duals,
            best_witness.upper_bound_duals,
        )
        temporary_constraints.append(
            compiled.model.addConstr(
                compiled.model.getObjective()
                >= known_lower_bound - known_scenario_tolerance,
                name="known_scenario_objective_floor",
            )
        )
        compiled.model.update()
        emit(
            PartitionEvent(
                leaf_index,
                leaf_count,
                "SOLVE_STARTED",
                assignment,
            )
        )
        try:
            result = compiled.solve()
        except RuntimeError:
            leaf = PartitionLeaf(
                assignment,
                f"UNRESOLVED_{compiled.model.Status}",
                None,
                None,
                {},
                known_lower_bound=known_lower_bound,
                full_known_scenario_start_used=False,
                witness_selector=best_witness.selector,
                witness_primal_objective=best_witness.primal_objective,
                witness_dual_objective=best_witness.dual_objective,
                witness_primal_dual_gap=best_witness.primal_dual_gap,
                witness_max_abs_residual=best_witness.max_abs_residual,
                full_witness_start_used=True,
                primal_dual_strengthening_used=False,
                runtime_seconds=float(compiled.model.Runtime),
                work=float(compiled.model.Work),
                node_count=float(compiled.model.NodeCount),
            )
            leaves.append(leaf)
            emit(
                PartitionEvent(
                    leaf_index, leaf_count, "COMPLETED", assignment, leaf
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
            full_known_scenario_start_used=(
                tuple(
                    best_witness.selector[key]
                    for key in bundle.selector_keys
                )
                in known_selector_keys
            ),
            witness_selector=best_witness.selector,
            witness_primal_objective=best_witness.primal_objective,
            witness_dual_objective=best_witness.dual_objective,
            witness_primal_dual_gap=best_witness.primal_dual_gap,
            witness_max_abs_residual=best_witness.max_abs_residual,
            full_witness_start_used=True,
            primal_dual_strengthening_used=False,
            runtime_seconds=float(compiled.model.Runtime),
            work=float(compiled.model.Work),
            node_count=float(compiled.model.NodeCount),
        )
        compiled.model.remove(temporary_constraints)
        compiled.model.update()
        leaves.append(leaf)
        emit(
            PartitionEvent(
                leaf_index, leaf_count, "COMPLETED", assignment, leaf
            )
        )
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
        if best_leaf is None or result.objective > best_leaf.objective:
            best_leaf = leaf

    if best_leaf is None:
        return PartitionResult(
            "infeasible_bundle", True, None, None, {}, {}, tuple(leaves)
        )
    return PartitionResult(
        status="OPTIMAL",
        coverage_complete=True,
        objective=best_leaf.objective,
        objective_bound=max(
            leaf.objective_bound
            for leaf in leaves
            if leaf.status == "OPTIMAL" and leaf.objective_bound is not None
        ),
        selector=best_leaf.selector,
        realization=bundle.evaluate(best_leaf.selector),
        leaves=tuple(leaves),
    )
