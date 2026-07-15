from __future__ import annotations

import json

from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.solver.ccg import solve_robust_ccg
from port_h2_certificate.two_stage_ir import TwoStageIR


CHECKPOINT_FILES = {
    "checkpoint.json",
    "scenario_pool.json",
    "first_stage.csv",
    "ccg_history.csv",
    "phase1_certificate.json",
    "cost_adversary_certificate.json",
    "model_stats.json",
    "residuals.json",
    "provenance.json",
}


def test_every_completed_ccg_iteration_can_be_atomically_checkpointed(
    tmp_path, toy_case, toy_joint_bundle
) -> None:
    from port_h2_certificate.checkpoint import write_ccg_checkpoint

    case = toy_case()
    two_stage = TwoStageIR(
        build_first_stage_ir(case, toy_joint_bundle),
        build_recourse_ir(case, toy_joint_bundle),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    callbacks = []

    def checkpoint(progress):
        callbacks.append(progress.latest.iteration)
        write_ccg_checkpoint(
            checkpoint_dir,
            progress=progress,
            two_stage_ir=two_stage,
            bundle=toy_joint_bundle,
            provenance={"case": "toy"},
        )

    result = solve_robust_ccg(
        two_stage,
        toy_joint_bundle,
        outer_gap_tolerance=1e-9,
        iteration_callback=checkpoint,
    )
    assert callbacks == [item.iteration for item in result.history]
    assert CHECKPOINT_FILES <= {path.name for path in checkpoint_dir.iterdir()}
    payload = json.loads((checkpoint_dir / "checkpoint.json").read_text())
    assert payload["iteration"] == result.history[-1].iteration
    assert payload["latest_action"] == result.history[-1].action
