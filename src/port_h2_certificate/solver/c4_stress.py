"""C4 deterministic nominal solve followed by exact full-set stress tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from port_h2_certificate.compilers.compile_adversary import compile_adversary
from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.compilers.compile_phase1 import compile_phase1
from port_h2_certificate.compilers.compile_phase1_adversary import (
    compile_phase1_adversary,
)
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.partition_oracle import solve_exact_partition
from port_h2_certificate.recourse_ir import VariableKey
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class C4StressResult:
    status: str
    nominal_objective: float
    first_stage: Mapping[VariableKey, float]
    phase1_objective: float
    stress_recourse_objective: float | None
    stress_total_objective: float | None
    stress_objective_bound: float | None
    worst_selector: Mapping[str, int]
    worst_realization: Mapping[str, np.ndarray]
    primal_replay_passed: bool
    max_abs_residual: float
    worst_recourse_variables: Mapping[VariableKey, float]
    worst_cost_breakdown: Mapping[str, float]
    partition_oracle_used: bool
    all_partition_leaves_optimal_or_empty: bool
    cost_partition_branch_keys: tuple[str, ...]


def _first_stage_cost(two_stage_ir: TwoStageIR, values) -> float:
    return float(
        sum(
            spec.objective_coefficient * values[spec.key]
            for spec in two_stage_ir.first_stage.variables
        )
    )


def solve_c4_nominal_then_stress(
    two_stage_ir: TwoStageIR,
    joint_bundle: UncertaintyBundle,
    *,
    feasibility_tolerance: float = 1e-8,
    dual_replay_tolerance: float = 1e-7,
    phase1_sigma: float = 1.0,
    cost_partition_branch_keys: Sequence[str] = (),
) -> C4StressResult:
    branch_keys = tuple(cost_partition_branch_keys)
    nominal = joint_bundle.evaluate(joint_bundle.nominal_selector)
    master = compile_master(two_stage_ir, [nominal]).solve()
    phase1 = compile_phase1_adversary(
        two_stage_ir.recourse,
        master.first_stage,
        joint_bundle,
        sigma=phase1_sigma,
    ).solve()
    if phase1.objective > feasibility_tolerance:
        replay = compile_phase1(
            two_stage_ir.recourse,
            master.first_stage,
            phase1.realization,
            sigma=phase1_sigma,
        ).solve()
        replay_passed = (
            abs(replay.objective - phase1.objective) <= dual_replay_tolerance
        )
        return C4StressResult(
            status="stress_infeasible",
            nominal_objective=master.objective,
            first_stage=master.first_stage,
            phase1_objective=phase1.objective,
            stress_recourse_objective=None,
            stress_total_objective=None,
            stress_objective_bound=None,
            worst_selector=phase1.selector,
            worst_realization=phase1.realization,
            primal_replay_passed=replay_passed,
            max_abs_residual=0.0,
            worst_recourse_variables={},
            worst_cost_breakdown={},
            partition_oracle_used=False,
            all_partition_leaves_optimal_or_empty=False,
            cost_partition_branch_keys=branch_keys,
        )

    dual_template = compile_dual(two_stage_ir.recourse, master.first_stage)
    if branch_keys:
        adversary = solve_exact_partition(
            dual_template, joint_bundle, branch_keys
        )
        leaves_are_resolved = all(
            leaf.status in {"OPTIMAL", "INFEASIBLE"}
            for leaf in adversary.leaves
        )
        if (
            adversary.status != "OPTIMAL"
            or not adversary.coverage_complete
            or not leaves_are_resolved
            or adversary.objective is None
            or adversary.objective_bound is None
        ):
            raise RuntimeError(
                "C4 partitioned stress adversary did not provide complete "
                "OPTIMAL/INFEASIBLE leaf coverage"
            )
    else:
        adversary = compile_adversary(dual_template, joint_bundle).solve()
        leaves_are_resolved = False
    replay = compile_recourse(
        two_stage_ir.recourse, master.first_stage, adversary.realization
    ).solve()
    replay_passed = (
        abs(replay.objective - adversary.objective) <= dual_replay_tolerance
    )
    status = "stress_certified" if replay_passed else "cost_replay_mismatch"
    return C4StressResult(
        status=status,
        nominal_objective=master.objective,
        first_stage=master.first_stage,
        phase1_objective=phase1.objective,
        stress_recourse_objective=replay.objective,
        stress_total_objective=_first_stage_cost(
            two_stage_ir, master.first_stage
        )
        + replay.objective,
        stress_objective_bound=adversary.objective_bound,
        worst_selector=adversary.selector,
        worst_realization=adversary.realization,
        primal_replay_passed=replay_passed,
        max_abs_residual=replay.max_abs_residual,
        worst_recourse_variables=replay.variables,
        worst_cost_breakdown=replay.cost_by_component,
        partition_oracle_used=bool(branch_keys),
        all_partition_leaves_optimal_or_empty=leaves_are_resolved,
        cost_partition_branch_keys=branch_keys,
    )
