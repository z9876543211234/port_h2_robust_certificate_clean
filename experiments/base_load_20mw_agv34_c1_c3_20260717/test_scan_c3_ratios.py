from __future__ import annotations

from pathlib import Path

import pytest

from experiments.base_load_20mw_agv34_c1_c3_20260717.scan_c3_ratios import (
    integer_group_split,
    portable_manifest_path,
    select_best_candidate,
)
from runners.common import PROJECT_ROOT


@pytest.mark.parametrize(
    ("share_percent", "expected"),
    [
        (50, (17, 17)),
        (55, (19, 15)),
        (60, (20, 14)),
        (65, (22, 12)),
        (70, (24, 10)),
        (75, (26, 8)),
        (80, (27, 7)),
        (85, (29, 5)),
    ],
)
def test_integer_group_split_uses_half_up_rounding_and_preserves_fleet(
    share_percent: int,
    expected: tuple[int, int],
) -> None:
    split = integer_group_split(34, share_percent)
    assert split == expected
    assert sum(split) == 34


def test_select_best_candidate_minimizes_known_pool_robust_objective() -> None:
    candidates = [
        {"status": "OPTIMAL", "objective_scaled": -120.0, "share_percent": 55},
        {"status": "INFEASIBLE", "objective_scaled": None, "share_percent": 60},
        {"status": "OPTIMAL", "objective_scaled": -125.0, "share_percent": 65},
    ]

    selected = select_best_candidate(candidates)

    assert selected["share_percent"] == 65


def test_select_best_candidate_rejects_scan_without_optimal_candidate() -> None:
    with pytest.raises(ValueError, match="no OPTIMAL C3 ratio candidate"):
        select_best_candidate(
            [{"status": "INFEASIBLE", "objective_scaled": None}]
        )


def test_portable_manifest_path_removes_workstation_prefix() -> None:
    source = PROJECT_ROOT / "experiments" / "example" / "case.yaml"

    assert portable_manifest_path(source) == "experiments/example/case.yaml"
    assert "/Users/" not in portable_manifest_path(source)


def test_portable_manifest_path_uses_name_for_external_input(tmp_path: Path) -> None:
    source = tmp_path / "case.yaml"

    assert portable_manifest_path(source) == "case.yaml"
