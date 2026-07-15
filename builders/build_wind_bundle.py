"""Build a wind Bundle from a frozen JSON/YAML source package."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.wind.builder import build_wind_bundle
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.source).read_text(encoding="utf-8"))
    profile_name = payload["profile_name"]
    periods, dt = (24, 1.0) if profile_name == "hourly_24" else (96, 0.25)
    source = WindUncertaintySource(
        profile=HorizonProfile.formal(profile_name, periods, dt),
        nominal_power_kw=tuple(payload["nominal_power_kw"]),
        deviation_down_kw=tuple(payload["deviation_down_kw"]),
        deviation_up_kw=tuple(payload["deviation_up_kw"]),
        budget=int(payload["budget"]),
        provenance=ProvenanceRecord.from_dict(payload["provenance"]),
    )
    bundle = build_wind_bundle(source, args.output)
    print(bundle.bundle_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
