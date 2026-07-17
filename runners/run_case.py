"""Run one C1-C4 case in strict certification mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from port_h2_certificate.checkpoint import (
    load_partition_checkpoint,
    write_ccg_checkpoint,
    write_partition_checkpoint,
)
from port_h2_certificate.export import export_c4_run, export_ccg_run
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import load_case
from port_h2_certificate.partition_oracle import (
    PartitionEvent,
    build_ship_delay_partition_assignments,
)
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.solver.c4_stress import solve_c4_nominal_then_stress
from port_h2_certificate.solver.ccg import solve_robust_ccg
from port_h2_certificate.two_stage_ir import TwoStageIR
from runners.common import case_path, current_git_commit, parse_resolutions
from runners.input_adapters import build_joint_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--profile", required=True, choices=("hourly_24", "quarter_hour_96"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--cost-partition-key",
        action="append",
        default=[],
        help=(
            "Primary uncertainty selector used by the exact mutually-exclusive "
            "cost-adversary partition; repeat for multiple keys."
        ),
    )
    parser.add_argument(
        "--resume-partition",
        action="store_true",
        help=(
            "Resume only fully resolved exact-partition leaves whose Bundle, "
            "iteration, branch keys, and first-stage hash still match."
        ),
    )
    parser.add_argument(
        "--semantic-ship-partition",
        action="store_true",
        help=(
            "Enumerate every feasible ship-delay projection and leave wind "
            "selectors inside each exact zero-gap adversary leaf."
        ),
    )
    args = parser.parse_args()
    output = Path(args.output)
    loaded = load_case(
        case_path(args.case), args.profile, resolutions=parse_resolutions(args.set)
    )
    bundle = build_joint_bundle(loaded, output / "bundles")
    if args.semantic_ship_partition and args.cost_partition_key:
        parser.error(
            "--semantic-ship-partition cannot be combined with "
            "--cost-partition-key"
        )
    if args.semantic_ship_partition:
        partition_keys, partition_assignments = (
            build_ship_delay_partition_assignments(bundle)
        )
    else:
        partition_keys = tuple(args.cost_partition_key)
        partition_assignments = None
    two_stage = TwoStageIR(
        build_first_stage_ir(loaded.case, bundle),
        build_recourse_ir(loaded.case, bundle),
    )
    git_commit = current_git_commit()
    if loaded.case.case_name == "C4_DeterministicMain":
        if args.semantic_ship_partition:
            parser.error(
                "--semantic-ship-partition is currently available for "
                "C1-C3 C&CG cases only"
            )
        result = solve_c4_nominal_then_stress(
            two_stage,
            bundle,
            cost_partition_branch_keys=partition_keys,
        )
        export_c4_run(
            output,
            case=loaded.case,
            two_stage_ir=two_stage,
            bundle=bundle,
            result=result,
            git_commit=git_commit,
            ship_delay_source=loaded.ship_delay_payload,
        )
        return 0

    def checkpoint(progress):
        write_ccg_checkpoint(
            output / "checkpoints",
            progress=progress,
            two_stage_ir=two_stage,
            bundle=bundle,
            provenance={
                "deterministic": loaded.deterministic_provenance.to_dict(),
                "source_hashes": bundle.source_hashes,
            },
        )

    partition_checkpoint_path = (
        output / "checkpoints" / "partition_progress.json"
    )
    partition_events_by_iteration = {}
    semantic_assignment_index = (
        {
            tuple(assignment[key] for key in partition_keys): index
            for index, assignment in enumerate(partition_assignments)
        }
        if partition_assignments is not None
        else {}
    )

    def resume_partition(iteration, first_stage, current_bundle, branch_keys):
        if not args.resume_partition:
            return ()
        restored = load_partition_checkpoint(
            partition_checkpoint_path,
            iteration=iteration,
            branch_keys=branch_keys,
            first_stage=first_stage,
            bundle=current_bundle,
        )
        if args.semantic_ship_partition:
            partition_events_by_iteration[iteration] = [
                PartitionEvent(
                    semantic_assignment_index[
                        tuple(leaf.fixed_primary[key] for key in partition_keys)
                    ],
                    len(partition_assignments),
                    "RESUMED",
                    leaf.fixed_primary,
                    leaf,
                )
                for leaf in restored
            ]
        return restored

    def partition_progress(iteration, first_stage, event):
        events = partition_events_by_iteration.setdefault(iteration, [])
        if args.semantic_ship_partition and event.stage != "COMPLETED":
            return
        events.append(event)
        write_partition_checkpoint(
            partition_checkpoint_path,
            iteration=iteration,
            branch_keys=partition_keys,
            first_stage=first_stage,
            bundle=bundle,
            events=events,
        )

    result = solve_robust_ccg(
        two_stage,
        bundle,
        max_iterations=args.max_iterations,
        cost_partition_branch_keys=partition_keys,
        cost_partition_assignments=partition_assignments,
        iteration_callback=checkpoint,
        partition_resume_provider=resume_partition,
        partition_event_callback=partition_progress,
    )
    export_ccg_run(
        output,
        case=loaded.case,
        two_stage_ir=two_stage,
        bundle=bundle,
        result=result,
        git_commit=git_commit,
        ship_delay_source=loaded.ship_delay_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
