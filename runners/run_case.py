"""Run one C1-C4 case in strict certification mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from port_h2_certificate.checkpoint import write_ccg_checkpoint
from port_h2_certificate.export import export_c4_run, export_ccg_run
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.load_case import load_case
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
    args = parser.parse_args()
    output = Path(args.output)
    loaded = load_case(
        case_path(args.case), args.profile, resolutions=parse_resolutions(args.set)
    )
    bundle = build_joint_bundle(loaded, output / "bundles")
    two_stage = TwoStageIR(
        build_first_stage_ir(loaded.case, bundle),
        build_recourse_ir(loaded.case, bundle),
    )
    git_commit = current_git_commit()
    if loaded.case.case_name == "C4_DeterministicMain":
        result = solve_c4_nominal_then_stress(
            two_stage,
            bundle,
            cost_partition_branch_keys=tuple(args.cost_partition_key),
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

    result = solve_robust_ccg(
        two_stage,
        bundle,
        max_iterations=args.max_iterations,
        cost_partition_branch_keys=tuple(args.cost_partition_key),
        iteration_callback=checkpoint,
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
