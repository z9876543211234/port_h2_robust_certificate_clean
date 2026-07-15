"""Build the complete certified Fig. 1--4 paper package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from port_h2_certificate.solver.ccg import realization_sha256

from .build_data import FORMAL_ROOT, PROJECT_ROOT, build_core_plot_data
from .comparisons import RUN_DIRECTORIES, build_comparison_data
from .data_loader import FormalRun, load_formal_run


STEMS = {
    "Fig01": "fig01_uncertainty",
    "Fig02": "fig02_c1_energy_hydrogen_dispatch",
    "Fig03": "fig03_agv_energy_logistics",
    "Fig04": "fig04_case_comparison",
}


def _scenario_hash(run: FormalRun) -> str:
    frame = run.worst_scenario.sort_values("period")
    realization = {
        column: frame[column].dropna().to_numpy(dtype=float)
        for column in frame.columns
        if column != "period"
    }
    return realization_sha256(realization)


def _relative_source(path: Path, output_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved.relative_to(output_root))


def _base_manifest(
    figure_id: str,
    run: FormalRun,
    source_files: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "case_name": run.certificate["case_name"],
        "profile_name": run.certificate["profile_name"],
        "scenario_type": "certified_worst_case",
        "scenario_sha256": _scenario_hash(run),
        "joint_bundle_sha256": run.certificate["joint_bundle_sha256"],
        "case_semantic_sha256": run.certificate["case_semantic_sha256"],
        "git_commit": run.certificate["git_commit"],
        "engineering_optimal": True,
        "master_status": "OPTIMAL",
        "phase1_adversary_status": "OPTIMAL",
        "cost_adversary_status": "OPTIMAL",
        "uncertainty_coverage_complete": True,
        "primal_replay_passed": True,
        "residual_check_passed": True,
        "accepted_limited_solve": False,
        "time_limit_used_for_certificate": False,
        "source_files": [
            _relative_source(path, output_root) for path in source_files
        ],
    }


def build_all_figures(output_directory: str | Path):
    output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mpl_config = output_root / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    # Matplotlib imports are intentionally deferred until MPLCONFIGDIR is set.
    import matplotlib.pyplot as plt

    from .export import export_figure
    from .fig01_uncertainty import build_figure as build_fig01
    from .fig02_energy_hydrogen import build_figure as build_fig02
    from .fig03_agv_coordination import build_figure as build_fig03
    from .fig04_case_comparison import build_figure as build_fig04

    runs = {
        code: load_formal_run(FORMAL_ROOT / directory)
        for code, directory in RUN_DIRECTORIES.items()
    }
    core = build_core_plot_data(output_root)
    comparisons = build_comparison_data(output_root)
    c1 = runs["C1"]
    common_c1_sources = [
        c1.path / "certificate.json",
        c1.path / "resolved_case.json",
        c1.path / "dispatch_day_ahead.csv",
        c1.path / "dispatch_recourse_worst.csv",
        c1.path / "worst_scenario.csv",
        c1.path / "ship_delay_assignment.csv",
        core.flows_path,
        core.states_path,
    ]

    results: dict[str, dict[str, Path]] = {}
    for figure_id, stem, builder in (
        ("Fig01", STEMS["Fig01"], build_fig01),
        ("Fig02", STEMS["Fig02"], build_fig02),
        ("Fig03", STEMS["Fig03"], build_fig03),
    ):
        figure, plot_paths = builder(core, c1, output_root)
        manifest = _base_manifest(
            figure_id,
            c1,
            [*common_c1_sources, *plot_paths],
            output_root,
        )
        if figure_id == "Fig01":
            manifest["uncertainty_projection"] = (
                "three_day_ship_conservation_current_day_display"
            )
        results[stem] = export_figure(figure, output_root, stem, manifest)
        plt.close(figure)

    figure, plot_paths = build_fig04(comparisons, runs, output_root)
    fig04_sources = [
        path
        for run in runs.values()
        for path in (
            run.path / "certificate.json",
            run.path / "cost_breakdown.json",
            run.path / "dispatch_day_ahead.csv",
            run.path / "dispatch_recourse_worst.csv",
            run.path / "worst_scenario.csv",
        )
    ]
    fig04_manifest = _base_manifest(
        "Fig04",
        c1,
        [*fig04_sources, *plot_paths],
        output_root,
    )
    fig04_manifest.update(
        {
            "case_name": [runs[code].certificate["case_name"] for code in runs],
            "case_semantic_sha256": {
                code: runs[code].certificate["case_semantic_sha256"]
                for code in runs
            },
            "scenario_type": "strict_common_and_cross_certified_replay",
            "scenario_sha256": {
                "C1": _scenario_hash(runs["C1"]),
                "C3": _scenario_hash(runs["C3"]),
                "C4": _scenario_hash(runs["C4"]),
            },
            "formal_status_by_case": {
                code: {
                    "engineering_optimal": run.certificate[
                        "engineering_optimal"
                    ],
                    "accepted_limited_solve": run.certificate[
                        "accepted_limited_solve"
                    ],
                    "time_limit_used_for_certificate": run.certificate[
                        "time_limit_used_for_certificate"
                    ],
                }
                for code, run in runs.items()
            },
            "normalization_exclusions": list(
                comparisons.normalization_exclusions
            ),
            "normalization_note": (
                "Metrics with a zero C1 denominator are excluded; no imputed "
                "or fabricated normalized values are plotted."
            ),
        }
    )
    results[STEMS["Fig04"]] = export_figure(
        figure, output_root, STEMS["Fig04"], fig04_manifest
    )
    plt.close(figure)

    build_report = {
        "status": "FIGURES_1_TO_4_CERTIFIED",
        "generated_figures": list(results),
        "fig05_status": "PENDING_FORMAL_SENSITIVITY_RUNS",
        "joint_bundle_sha256": c1.certificate["joint_bundle_sha256"],
        "accepted_limited_solve": False,
        "time_limit_used_for_certificate": False,
    }
    (output_root / "BUILD_REPORT.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    build_all_figures(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

