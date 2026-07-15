"""Build an integer ship-delay Bundle from a frozen JSON/YAML source package."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.source).read_text(encoding="utf-8"))
    unresolved = [
        key
        for key in ("max_delay_steps", "delayed_ship_budget")
        if payload[key] is None
    ]
    if unresolved:
        raise ValueError(f"unresolved ship-delay source fields: {unresolved}")
    profile_name = payload["profile_name"]
    periods, dt = (24, 1.0) if profile_name == "hourly_24" else (96, 0.25)
    source = ShipDelaySource(
        profile=HorizonProfile.formal(profile_name, periods, dt),
        previous_day_arrival_count=tuple(payload["previous_day_arrival_count"]),
        current_day_arrival_count=tuple(payload["current_day_arrival_count"]),
        next_day_arrival_count=tuple(payload["next_day_arrival_count"]),
        dwell_steps=int(payload["dwell_steps"]),
        max_delay_steps=int(payload["max_delay_steps"]),
        delayed_ship_budget=int(payload["delayed_ship_budget"]),
        total_delay_step_budget=payload["total_delay_step_budget"],
        shore_power_per_ship_kw=float(payload["shore_power_per_ship_kw"]),
        quay_cranes_per_ship=float(payload["quay_cranes_per_ship"]),
        installed_quay_cranes=float(payload["installed_quay_cranes"]),
        quay_crane_power_kw=float(payload["quay_crane_power_kw"]),
        quay_crane_task_rate_per_hour=float(payload["quay_crane_task_rate_per_hour"]),
        provenance=ProvenanceRecord.from_dict(payload["provenance"]),
    )
    bundle = build_ship_delay_bundle(source, args.output)
    print(bundle.bundle_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
