"""Verify the frozen confirmed input package and its manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs/recommended57"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((INPUTS / "manifest.json").read_text(encoding="utf-8"))
    mismatches = {
        name: {"expected": expected, "actual": _sha256(INPUTS / name)}
        for name, expected in manifest["sha256"].items()
        if _sha256(INPUTS / name) != expected
    }
    physical = json.loads(
        (INPUTS / "physical_large_port_96.json").read_text(encoding="utf-8")
    )
    ship = json.loads(
        (INPUTS / "ship_delay_large_port_96.json").read_text(encoding="utf-8")
    )
    assertions = {
        "manifest_hashes_match": not mismatches,
        "periods": len(physical["base_load_kw"]),
        "base_load_kw_unique": sorted(set(physical["base_load_kw"])),
        "agv_operation_per_vehicle_hour": physical["cost"][
            "agv_operation_per_vehicle_hour"
        ],
        "grid_deviation_per_kwh": physical["cost"]["grid_deviation_per_kwh"],
        "spill_costs": [
            physical["cost"]["spill_day_ahead_per_kwh"],
            physical["cost"]["spill_real_time_per_kwh"],
            physical["cost"]["spill_deviation_per_kwh"],
        ],
        "lohc_export_subsidy_per_kg_h2eq": physical["cost"][
            "lohc_export_revenue_per_kg"
        ],
        "lohc_non_electric_operation_cost": physical["cost"][
            "lohc_operation_per_kg"
        ],
        "buy_price_levels": sorted(set(physical["grid"]["buy_price_per_kwh"])),
        "sell_price_levels": sorted(set(physical["grid"]["sell_price_per_kwh"])),
        "buy_strictly_above_sell": all(
            buy > sell
            for buy, sell in zip(
                physical["grid"]["buy_price_per_kwh"],
                physical["grid"]["sell_price_per_kwh"],
                strict=True,
            )
        ),
        "ship_delay": {
            "delayed_ship_budget": ship["delayed_ship_budget"],
            "max_delay_steps": ship["max_delay_steps"],
            "total_delay_step_budget": ship["total_delay_step_budget"],
        },
    }
    expected = {
        "periods": 96,
        "base_load_kw_unique": [20000.0],
        "agv_operation_per_vehicle_hour": 60.0,
        "grid_deviation_per_kwh": 0.3,
        "spill_costs": [0.1, 0.1, 0.1],
        "lohc_export_subsidy_per_kg_h2eq": 20.0,
        "lohc_non_electric_operation_cost": 0.0,
        "buy_price_levels": [0.41, 0.63, 0.75],
        "sell_price_levels": [0.205, 0.315, 0.375],
        "buy_strictly_above_sell": True,
        "ship_delay": {
            "delayed_ship_budget": 2,
            "max_delay_steps": 4,
            "total_delay_step_budget": None,
        },
    }
    passed = assertions["manifest_hashes_match"] and all(
        assertions[key] == value for key, value in expected.items()
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "assertions": assertions,
        "hash_mismatches": mismatches,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
