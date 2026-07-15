"""Noncertifying candidate replay; never supplies a formal bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from port_h2_certificate.compilers.compile_phase1 import compile_phase1
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.recourse_ir import RecourseIR, VariableKey
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class CandidateReplay:
    selector: Mapping[str, int]
    realization_sha256: str
    violation_type: str | None
    replay_objective: float
    replay_status: str
    certifying: bool
    objective_bound: None


def evaluate_candidates(
    ir: RecourseIR,
    first_stage: Mapping[VariableKey, float],
    bundle: UncertaintyBundle,
    candidates: Sequence[Mapping[str, int]],
    *,
    theta: float,
    feasibility_tolerance: float = 1e-8,
    cost_tolerance: float = 1e-8,
) -> tuple[CandidateReplay, ...]:
    replays: list[CandidateReplay] = []
    for candidate in candidates:
        if set(candidate) == set(bundle.selector_keys):
            selector = dict(candidate)
        else:
            selector = bundle.complete(candidate)
        realization = bundle.evaluate(selector)
        phase1 = compile_phase1(ir, first_stage, realization).solve()
        if phase1.objective > feasibility_tolerance:
            replays.append(
                CandidateReplay(
                    selector=selector,
                    realization_sha256=bundle.realization_hash(selector),
                    violation_type="feasibility",
                    replay_objective=phase1.objective,
                    replay_status="OPTIMAL",
                    certifying=False,
                    objective_bound=None,
                )
            )
            continue
        recourse = compile_recourse(ir, first_stage, realization).solve()
        replays.append(
            CandidateReplay(
                selector=selector,
                realization_sha256=bundle.realization_hash(selector),
                violation_type=(
                    "cost" if recourse.objective > theta + cost_tolerance else None
                ),
                replay_objective=recourse.objective,
                replay_status="OPTIMAL",
                certifying=False,
                objective_bound=None,
            )
        )
    return tuple(replays)
