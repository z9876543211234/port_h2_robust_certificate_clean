"""Certification-mode column-and-constraint generation state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from port_h2_certificate.compilers.compile_adversary import compile_adversary
from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_phase1 import compile_phase1
from port_h2_certificate.compilers.compile_phase1_adversary import (
    compile_phase1_adversary,
)
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.partition_oracle import (
    PartitionEvent,
    PartitionLeaf,
    solve_exact_partition,
)
from port_h2_certificate.recourse_ir import VariableKey
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.hashing import canonical_sha256
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


Realization = Mapping[str, np.ndarray]


def realization_sha256(realization: Mapping[str, object]) -> str:
    payload = {
        key: np.asarray(value, dtype=float).tolist()
        for key, value in sorted(realization.items())
    }
    return canonical_sha256(payload)


@dataclass(frozen=True)
class ScenarioRecord:
    realization_sha256: str
    selector: Mapping[str, int]
    realization: Realization


@dataclass(frozen=True)
class CcgIteration:
    iteration: int
    lower_bound: float
    global_upper_bound: float
    gap_abs: float
    gap_rel: float
    theta: float
    phase1_objective: float
    cost_adversary_objective: float | None
    scenario_count: int
    action: str


@dataclass(frozen=True)
class CcgResult:
    status: str
    engineering_optimal: bool
    lower_bound: float
    upper_bound: float
    gap_abs: float
    gap_rel: float
    first_stage: Mapping[VariableKey, float]
    theta: float
    scenarios: tuple[ScenarioRecord, ...]
    history: tuple[CcgIteration, ...]
    primal_replay_passed: bool
    residual_check_passed: bool
    max_abs_residual: float
    bounds_ordered: bool
    uncertainty_coverage_complete: bool
    accepted_limited_solve: bool
    worst_selector: Mapping[str, int]
    worst_realization: Realization
    worst_recourse_variables: Mapping[VariableKey, float]
    worst_cost_breakdown: Mapping[str, float]
    partition_oracle_used: bool = False
    all_partition_leaves_optimal_or_empty: bool | None = None
    cost_partition_branch_keys: tuple[str, ...] = ()
    cost_partition_leaf_count: int | None = None


@dataclass(frozen=True)
class CcgProgress:
    latest: CcgIteration
    history: tuple[CcgIteration, ...]
    first_stage: Mapping[VariableKey, float]
    theta: float
    scenarios: tuple[ScenarioRecord, ...]
    max_abs_residual: float


def _first_stage_cost(two_stage_ir: TwoStageIR, values) -> float:
    return float(
        sum(
            spec.objective_coefficient * values[spec.key]
            for spec in two_stage_ir.first_stage.variables
        )
    )


def _gap(lower_bound: float, upper_bound: float) -> tuple[float, float, bool]:
    ordered = upper_bound + 1e-7 >= lower_bound
    gap_abs = max(upper_bound - lower_bound, 0.0) if ordered else math.inf
    gap_rel = gap_abs / max(1.0, abs(upper_bound), abs(lower_bound))
    return gap_abs, gap_rel, ordered


def solve_robust_ccg(
    two_stage_ir: TwoStageIR,
    joint_bundle: UncertaintyBundle,
    *,
    outer_gap_tolerance: float = 0.01,
    feasibility_tolerance: float = 1e-8,
    dual_replay_tolerance: float = 1e-7,
    residual_tolerance: float = 1e-6,
    max_iterations: int = 100,
    phase1_sigma: float = 1.0,
    cost_partition_branch_keys: tuple[str, ...] = (),
    cost_partition_assignments: Sequence[Mapping[str, int]] | None = None,
    iteration_callback: Callable[[CcgProgress], None] | None = None,
    partition_resume_provider: Callable[
        [
            int,
            Mapping[VariableKey, float],
            UncertaintyBundle,
            tuple[str, ...],
        ],
        Sequence[PartitionLeaf],
    ]
    | None = None,
    partition_event_callback: Callable[
        [int, Mapping[VariableKey, float], PartitionEvent], None
    ]
    | None = None,
) -> CcgResult:
    if outer_gap_tolerance < 0.0:
        raise ValueError("outer_gap_tolerance must be nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    two_stage_ir.validate()
    joint_bundle.validate()

    nominal = joint_bundle.evaluate(joint_bundle.nominal_selector)
    nominal_hash = realization_sha256(nominal)
    scenario_by_hash: dict[str, ScenarioRecord] = {
        nominal_hash: ScenarioRecord(
            nominal_hash, dict(joint_bundle.nominal_selector), nominal
        )
    }
    global_upper_bound = math.inf
    history: list[CcgIteration] = []
    last_lower_bound = -math.inf
    last_first_stage: Mapping[VariableKey, float] = {}
    last_theta = math.nan
    max_abs_residual = 0.0
    primal_replay_passed = True
    residual_check_passed = True
    bounds_ordered = True
    last_worst_selector: Mapping[str, int] = {}
    last_worst_realization: Realization = {}
    last_worst_recourse_variables: Mapping[VariableKey, float] = {}
    last_worst_cost_breakdown: Mapping[str, float] = {}
    branch_keys = tuple(cost_partition_branch_keys)
    explicit_partition_assignments = (
        tuple(dict(assignment) for assignment in cost_partition_assignments)
        if cost_partition_assignments is not None
        else None
    )
    if explicit_partition_assignments is not None and not branch_keys:
        raise ValueError("explicit partition assignments require branch keys")
    all_partition_leaves_optimal_or_empty: bool | None = (
        True if branch_keys else None
    )

    def _record(entry: CcgIteration) -> None:
        history.append(entry)
        if iteration_callback is not None:
            iteration_callback(
                CcgProgress(
                    latest=entry,
                    history=tuple(history),
                    first_stage=last_first_stage,
                    theta=last_theta,
                    scenarios=tuple(scenario_by_hash.values()),
                    max_abs_residual=max_abs_residual,
                )
            )

    def finish(status: str, engineering_optimal: bool) -> CcgResult:
        gap_abs, gap_rel, ordered = _gap(last_lower_bound, global_upper_bound)
        return CcgResult(
            status=status,
            engineering_optimal=engineering_optimal,
            lower_bound=last_lower_bound,
            upper_bound=global_upper_bound,
            gap_abs=gap_abs,
            gap_rel=gap_rel,
            first_stage=last_first_stage,
            theta=last_theta,
            scenarios=tuple(scenario_by_hash.values()),
            history=tuple(history),
            primal_replay_passed=primal_replay_passed,
            residual_check_passed=residual_check_passed,
            max_abs_residual=max_abs_residual,
            bounds_ordered=bounds_ordered and ordered,
            uncertainty_coverage_complete=True,
            accepted_limited_solve=False,
            worst_selector=last_worst_selector,
            worst_realization=last_worst_realization,
            worst_recourse_variables=last_worst_recourse_variables,
            worst_cost_breakdown=last_worst_cost_breakdown,
            partition_oracle_used=bool(branch_keys),
            all_partition_leaves_optimal_or_empty=(
                all_partition_leaves_optimal_or_empty
            ),
            cost_partition_branch_keys=branch_keys,
            cost_partition_leaf_count=(
                len(explicit_partition_assignments)
                if explicit_partition_assignments is not None
                else (2 ** len(branch_keys) if branch_keys else None)
            ),
        )

    for iteration in range(1, max_iterations + 1):
        master = compile_master(
            two_stage_ir,
            [record.realization for record in scenario_by_hash.values()],
        ).solve()
        last_lower_bound = master.objective
        last_first_stage = master.first_stage
        last_theta = master.theta

        phase1_adversary = compile_phase1_adversary(
            two_stage_ir.recourse,
            master.first_stage,
            joint_bundle,
            sigma=phase1_sigma,
        ).solve()
        if phase1_adversary.objective > feasibility_tolerance:
            replay = compile_phase1(
                two_stage_ir.recourse,
                master.first_stage,
                phase1_adversary.realization,
                sigma=phase1_sigma,
            ).solve()
            if abs(replay.objective - phase1_adversary.objective) > dual_replay_tolerance:
                primal_replay_passed = False
                _record(
                    CcgIteration(
                        iteration,
                        last_lower_bound,
                        global_upper_bound,
                        math.inf,
                        math.inf,
                        last_theta,
                        phase1_adversary.objective,
                        None,
                        len(scenario_by_hash),
                        "phase1_replay_mismatch",
                    )
                )
                return finish("phase1_replay_mismatch", False)
            scenario_hash = realization_sha256(phase1_adversary.realization)
            if scenario_hash in scenario_by_hash:
                _record(
                    CcgIteration(
                        iteration,
                        last_lower_bound,
                        global_upper_bound,
                        math.inf,
                        math.inf,
                        last_theta,
                        phase1_adversary.objective,
                        None,
                        len(scenario_by_hash),
                        "stalled_duplicate_infeasible_scenario",
                    )
                )
                return finish("inconsistent_or_stalled_duplicate", False)
            scenario_by_hash[scenario_hash] = ScenarioRecord(
                scenario_hash,
                phase1_adversary.selector,
                phase1_adversary.realization,
            )
            _record(
                CcgIteration(
                    iteration,
                    last_lower_bound,
                    global_upper_bound,
                    math.inf,
                    math.inf,
                    last_theta,
                    phase1_adversary.objective,
                    None,
                    len(scenario_by_hash),
                    "add_phase1_scenario",
                )
            )
            continue

        dual_template = compile_dual(two_stage_ir.recourse, master.first_stage)
        if branch_keys:
            known_scenario_values: list[tuple[Mapping[str, int], float]] = []
            for record in scenario_by_hash.values():
                known_replay = compile_recourse(
                    two_stage_ir.recourse,
                    master.first_stage,
                    record.realization,
                ).solve()
                max_abs_residual = max(
                    max_abs_residual, known_replay.max_abs_residual
                )
                residual_check_passed &= (
                    known_replay.max_abs_residual <= residual_tolerance
                )
                if known_replay.objective > master.theta + dual_replay_tolerance:
                    primal_replay_passed = False
                    _record(
                        CcgIteration(
                            iteration,
                            last_lower_bound,
                            global_upper_bound,
                            math.inf,
                            math.inf,
                            last_theta,
                            phase1_adversary.objective,
                            None,
                            len(scenario_by_hash),
                            "master_scenario_replay_mismatch",
                        )
                    )
                    return finish("master_scenario_replay_mismatch", False)
                known_scenario_values.append(
                    (record.selector, float(known_replay.objective))
                )
            resume_leaves = (
                tuple(
                    partition_resume_provider(
                        iteration,
                        master.first_stage,
                        joint_bundle,
                        branch_keys,
                    )
                )
                if partition_resume_provider is not None
                else ()
            )

            def relay_partition_event(event: PartitionEvent) -> None:
                if partition_event_callback is not None:
                    partition_event_callback(
                        iteration, master.first_stage, event
                    )

            cost_adversary = solve_exact_partition(
                dual_template,
                joint_bundle,
                branch_keys,
                known_scenario_values=tuple(known_scenario_values),
                known_scenario_tolerance=dual_replay_tolerance,
                resume_leaves=resume_leaves,
                event_callback=relay_partition_event,
                partition_assignments=explicit_partition_assignments,
            )
            leaves_are_resolved = all(
                leaf.status in {"OPTIMAL", "INFEASIBLE"}
                for leaf in cost_adversary.leaves
            )
            all_partition_leaves_optimal_or_empty &= leaves_are_resolved
            if (
                cost_adversary.status != "OPTIMAL"
                or not cost_adversary.coverage_complete
                or not leaves_are_resolved
                or cost_adversary.objective is None
                or cost_adversary.objective_bound is None
            ):
                raise RuntimeError(
                    "formal partitioned cost adversary did not provide complete "
                    "OPTIMAL/INFEASIBLE leaf coverage"
                )
        else:
            cost_adversary = compile_adversary(
                dual_template, joint_bundle
            ).solve()
        replay = compile_recourse(
            two_stage_ir.recourse,
            master.first_stage,
            cost_adversary.realization,
        ).solve()
        last_worst_selector = cost_adversary.selector
        last_worst_realization = cost_adversary.realization
        last_worst_recourse_variables = replay.variables
        last_worst_cost_breakdown = replay.cost_by_component
        replay_error = abs(cost_adversary.objective - replay.objective)
        if replay_error > dual_replay_tolerance:
            primal_replay_passed = False
            _record(
                CcgIteration(
                    iteration,
                    last_lower_bound,
                    global_upper_bound,
                    math.inf,
                    math.inf,
                    last_theta,
                    phase1_adversary.objective,
                    cost_adversary.objective,
                    len(scenario_by_hash),
                    "cost_replay_mismatch",
                )
            )
            return finish("cost_replay_mismatch", False)
        max_abs_residual = max(max_abs_residual, replay.max_abs_residual)
        residual_check_passed &= replay.max_abs_residual <= residual_tolerance

        candidate_upper_bound = _first_stage_cost(
            two_stage_ir, master.first_stage
        ) + cost_adversary.objective_bound
        global_upper_bound = min(global_upper_bound, candidate_upper_bound)
        gap_abs, gap_rel, iteration_ordered = _gap(
            last_lower_bound, global_upper_bound
        )
        bounds_ordered &= iteration_ordered
        if not iteration_ordered:
            _record(
                CcgIteration(
                    iteration,
                    last_lower_bound,
                    global_upper_bound,
                    gap_abs,
                    gap_rel,
                    last_theta,
                    phase1_adversary.objective,
                    cost_adversary.objective,
                    len(scenario_by_hash),
                    "bounds_not_ordered",
                )
            )
            return finish("bounds_not_ordered", False)
        if gap_rel <= outer_gap_tolerance and residual_check_passed:
            _record(
                CcgIteration(
                    iteration,
                    last_lower_bound,
                    global_upper_bound,
                    gap_abs,
                    gap_rel,
                    last_theta,
                    phase1_adversary.objective,
                    cost_adversary.objective,
                    len(scenario_by_hash),
                    "certified",
                )
            )
            return finish("engineering_optimal", True)

        scenario_hash = realization_sha256(cost_adversary.realization)
        if scenario_hash in scenario_by_hash:
            _record(
                CcgIteration(
                    iteration,
                    last_lower_bound,
                    global_upper_bound,
                    gap_abs,
                    gap_rel,
                    last_theta,
                    phase1_adversary.objective,
                    cost_adversary.objective,
                    len(scenario_by_hash),
                    "stalled_duplicate_cost_scenario",
                )
            )
            return finish("inconsistent_or_stalled_duplicate", False)
        scenario_by_hash[scenario_hash] = ScenarioRecord(
            scenario_hash, cost_adversary.selector, cost_adversary.realization
        )
        _record(
            CcgIteration(
                iteration,
                last_lower_bound,
                global_upper_bound,
                gap_abs,
                gap_rel,
                last_theta,
                phase1_adversary.objective,
                cost_adversary.objective,
                len(scenario_by_hash),
                "add_cost_scenario",
            )
        )

    return finish("maximum_iterations_reached", False)
