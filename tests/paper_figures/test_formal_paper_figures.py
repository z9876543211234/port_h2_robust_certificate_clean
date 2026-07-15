from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = PROJECT_ROOT / "experiments/c1_confirmed_main_budget_20260714"


def test_formal_loader_accepts_authoritative_c1_certificate() -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.data_loader import (
        load_formal_run,
    )

    run = load_formal_run(FORMAL_ROOT / "C1_ProposedRobustMain_refined_partition")

    assert run.certificate["case_name"] == "C1_ProposedRobustMain"
    assert run.certificate["engineering_optimal"] is True
    assert run.certificate["accepted_limited_solve"] is False
    assert len(run.day_ahead) == 96
    assert len(run.worst_scenario) == 96


def test_formal_loader_rejects_limited_certificate(tmp_path: Path) -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.data_loader import (
        FormalRunError,
        load_formal_run,
    )

    source = FORMAL_ROOT / "C1_ProposedRobustMain_refined_partition/certificate.json"
    certificate = json.loads(source.read_text(encoding="utf-8"))
    certificate["accepted_limited_solve"] = True
    (tmp_path / "certificate.json").write_text(
        json.dumps(certificate), encoding="utf-8"
    )

    with pytest.raises(FormalRunError, match="NOT FOR PUBLICATION"):
        load_formal_run(tmp_path)


def test_c1_metrics_reconcile_to_certified_upper_bound() -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.checks import (
        assert_cost_reconciliation,
    )
    from paper_figures.formal_c1_c4_four_key_20260714.data_loader import (
        load_formal_run,
    )
    from paper_figures.formal_c1_c4_four_key_20260714.metrics import (
        compute_operational_metrics,
        signed_cost_components,
    )

    run = load_formal_run(FORMAL_ROOT / "C1_ProposedRobustMain_refined_partition")
    components = signed_cost_components(run)
    metrics = compute_operational_metrics(run, dt_hours=0.25)

    assert_cost_reconciliation(run, components)
    assert sum(components.values()) == pytest.approx(
        run.certificate["upper_bound"], abs=1e-6
    )
    assert metrics["grid_import_mwh"] > 0.0
    assert metrics["lohc_outbound_kg"] > 0.0
    assert metrics["terminal_backlog_teu"] == pytest.approx(0.0, abs=1e-9)


def test_core_plot_data_preserves_model_invariants(tmp_path: Path) -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.build_data import (
        build_core_plot_data,
    )

    artifacts = build_core_plot_data(tmp_path)
    flows = artifacts.flows
    states = artifacts.states

    assert len(flows) == 96
    assert len(states) == 97
    assert flows["time_hour"].iloc[-1] == pytest.approx(23.75)
    assert states["time_hour"].iloc[-1] == pytest.approx(24.0)
    assert np.max(np.abs(flows["da_power_balance_residual_mw"])) <= 1e-6
    assert np.max(np.abs(flows["rt_power_balance_residual_mw"])) <= 1e-6
    assert np.allclose(
        flows[
            [
                "agv_container_count",
                "agv_lohc_count",
                "agv_charge_count",
                "agv_idle_count",
            ]
        ].sum(axis=1),
        60.0,
        atol=1e-6,
        rtol=0.0,
    )
    assert np.allclose(
        flows["arrival_nominal_count"],
        np.rint(flows["arrival_nominal_count"]),
        atol=1e-9,
        rtol=0.0,
    )
    assert np.allclose(
        flows["arrival_worst_count"],
        np.rint(flows["arrival_worst_count"]),
        atol=1e-9,
        rtol=0.0,
    )


def test_formal_comparisons_use_strict_fixed_scenario_replays(
    tmp_path: Path,
) -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.comparisons import (
        build_comparison_data,
    )

    artifacts = build_comparison_data(tmp_path)
    cross = artifacts.cross_evaluation
    common = artifacts.common_scenario_timeseries

    assert len(cross) == 4
    assert set(cross["solver_status"]) == {"OPTIMAL"}
    assert np.max(cross["max_abs_residual"]) <= 1e-6
    c1_diagonal = cross.loc[
        (cross["policy_case"] == "C1") & (cross["scenario_case"] == "C1"),
        "total_cost",
    ].item()
    c4_diagonal = cross.loc[
        (cross["policy_case"] == "C4") & (cross["scenario_case"] == "C4"),
        "total_cost",
    ].item()
    assert c1_diagonal == pytest.approx(102.5165764479994, abs=1e-6)
    assert c4_diagonal == pytest.approx(104.72079643860596, abs=1e-6)
    assert set(common["policy_case"]) == {"C1", "C3"}
    assert len(common.loc[common["series"] == "backlog_teu"]) == 2 * 97
    assert common["scenario_sha256"].nunique() == 1


def test_build_all_exports_four_publication_figures_and_manifests(
    tmp_path: Path,
) -> None:
    from paper_figures.formal_c1_c4_four_key_20260714.build_all import (
        build_all_figures,
    )

    result = build_all_figures(tmp_path)
    expected = {
        "fig01_uncertainty",
        "fig02_c1_energy_hydrogen_dispatch",
        "fig03_agv_energy_logistics",
        "fig04_case_comparison",
    }
    assert set(result) == expected
    for stem in expected:
        for suffix in (".pdf", ".svg", ".png"):
            output = tmp_path / f"{stem}{suffix}"
            assert output.is_file()
            assert output.stat().st_size > 1000
        manifest = json.loads(
            (tmp_path / f"{stem}_source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["joint_bundle_sha256"] == (
            "c86a258d698635b6279717fd8f4c4f19491024e01cbe1af0ed695b2a3c8714a5"
        )
        assert manifest["accepted_limited_solve"] is False
        assert manifest["exports"]["png_dpi"] == 600
    assert not (tmp_path / "fig05_sensitivity.pdf").exists()
