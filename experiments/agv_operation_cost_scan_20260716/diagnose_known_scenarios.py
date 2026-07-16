"""Fixed-scenario primal/dual replay for the rejected formal C2 certificate.

This diagnostic does not change any physical or uncertainty-set semantics.  It
loads a frozen first-stage solution and saved C&CG scenarios, then compares the
reduced recourse LP, its explicit dual LP, and the indicator adversary with all
binary selectors fixed to the same realization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from port_h2_certificate.compilers.compile_adversary import compile_adversary
from port_h2_certificate.compilers.compile_dual import compile_dual
from port_h2_certificate.compilers.compile_recourse import compile_recourse
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import load_case
from port_h2_certificate.recourse_ir import VariableKey, build_recourse_ir
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


@dataclass(frozen=True)
class KnownScenarioReplay:
    primal_status: str
    dual_status: str
    fixed_adversary_status: str
    primal_objective: float
    dual_objective: float
    fixed_adversary_objective: float
    primal_dual_gap: float
    primal_fixed_adversary_gap: float
    primal_max_abs_residual: float
    dual_max_constraint_violation: float
    fixed_adversary_objective_bound: float


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_first_stage_csv(
    path: Path, two_stage: TwoStageIR
) -> dict[VariableKey, float]:
    values: dict[VariableKey, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (str(row["family"]), int(row["period"]))
            value = float(row["value"])
            if key in values or not math.isfinite(value):
                raise ValueError(f"invalid frozen first-stage value for {key}")
            values[key] = value
    expected = {spec.key for spec in two_stage.first_stage.variables}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(
            "frozen first-stage CSV does not match the first-stage IR: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return values


def _validate_selector_realization(
    bundle: UncertaintyBundle,
    selector: Mapping[str, int],
    realization: Mapping[str, object],
) -> None:
    if set(selector) != set(bundle.selector_keys) or any(
        int(value) not in {0, 1} for value in selector.values()
    ):
        raise ValueError("saved scenario selector is not a full binary assignment")
    evaluated = bundle.evaluate({key: int(value) for key, value in selector.items()})
    if set(realization) != set(evaluated):
        raise ValueError("saved scenario outputs do not match the uncertainty bundle")
    for key in evaluated:
        if not np.allclose(
            np.asarray(realization[key], dtype=float),
            np.asarray(evaluated[key], dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"saved scenario realization disagrees for {key}")


def replay_known_scenario(
    two_stage: TwoStageIR,
    bundle: UncertaintyBundle,
    first_stage: Mapping[VariableKey, float],
    selector: Mapping[str, int],
    realization: Mapping[str, object],
) -> KnownScenarioReplay:
    _validate_selector_realization(bundle, selector, realization)
    primal = compile_recourse(
        two_stage.recourse, first_stage, realization
    ).solve()
    dual_template = compile_dual(two_stage.recourse, first_stage)
    dual = dual_template.instantiate(realization).solve()
    fixed_adversary = compile_adversary(dual_template, bundle)
    for key, variable in fixed_adversary.selector_variables.items():
        fixed_adversary.model.addConstr(
            variable == int(selector[key]), name=f"diagnostic_fix[{key}]"
        )
    fixed_adversary.model.update()
    fixed = fixed_adversary.solve()
    return KnownScenarioReplay(
        primal_status="OPTIMAL",
        dual_status="OPTIMAL",
        fixed_adversary_status="OPTIMAL",
        primal_objective=float(primal.objective),
        dual_objective=float(dual.objective),
        fixed_adversary_objective=float(fixed.objective),
        primal_dual_gap=float(primal.objective - dual.objective),
        primal_fixed_adversary_gap=float(primal.objective - fixed.objective),
        primal_max_abs_residual=float(primal.max_abs_residual),
        dual_max_constraint_violation=float(dual.max_constraint_violation),
        fixed_adversary_objective_bound=float(fixed.objective_bound),
    )


def load_scenario_records(
    path: Path, bundle: UncertaintyBundle
) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("scenario pool must be a nonempty JSON list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"scenario pool entry {index} must be a mapping")
        selector = {key: int(value) for key, value in raw["selector"].items()}
        realization = {
            key: tuple(float(value) for value in values)
            for key, values in raw["realization"].items()
        }
        _validate_selector_realization(bundle, selector, realization)
        records.append(
            {
                "realization_sha256": str(raw["realization_sha256"]),
                "selector": selector,
                "realization": realization,
            }
        )
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--profile", default="quarter_hour_96", choices=("hourly_24", "quarter_hour_96")
    )
    parser.add_argument("--agv-operation-cost", required=True, type=float)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--first-stage-csv", required=True, type=Path)
    parser.add_argument("--scenario-pool-json", required=True, type=Path)
    parser.add_argument("--expected-theta", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty diagnostic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_case(
        args.case.resolve(),
        args.profile,
        resolutions={
            "deterministic.cost.agv_operation_per_vehicle_hour": (
                args.agv_operation_cost
            )
        },
    )
    bundle = UncertaintyBundle.load(args.bundle_manifest.resolve())
    loaded.case.validate(bundle)
    two_stage = TwoStageIR(
        build_first_stage_ir(loaded.case, bundle),
        build_recourse_ir(loaded.case, bundle),
    )
    first_stage = load_first_stage_csv(args.first_stage_csv.resolve(), two_stage)
    records = load_scenario_records(args.scenario_pool_json.resolve(), bundle)
    started = time.perf_counter()
    results = []
    for index, record in enumerate(records):
        replay = replay_known_scenario(
            two_stage,
            bundle,
            first_stage,
            record["selector"],
            record["realization"],
        )
        results.append(
            {
                "scenario_index": index,
                "realization_sha256": record["realization_sha256"],
                **asdict(replay),
            }
        )
    maximum_primal = max(item["primal_objective"] for item in results)
    payload = {
        "diagnostic": "fixed_known_scenario_primal_dual_indicator_replay",
        "changes_model_semantics": False,
        "accepted_as_formal_certificate": False,
        "case_name": loaded.case.case_name,
        "case_semantic_sha256": loaded.case_semantic_sha256,
        "joint_bundle_sha256": bundle.bundle_sha256,
        "agv_operation_per_vehicle_hour": args.agv_operation_cost,
        "scenario_count": len(results),
        "maximum_primal_objective": maximum_primal,
        "expected_theta": args.expected_theta,
        "maximum_primal_minus_expected_theta": (
            None
            if args.expected_theta is None
            else maximum_primal - args.expected_theta
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "scenarios": results,
    }
    _atomic_json(output / "fixed_scenario_duality.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
