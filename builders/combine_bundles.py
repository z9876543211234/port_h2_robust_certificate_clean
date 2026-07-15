"""Combine independently saved wind and ship Bundles by Cartesian product."""

from __future__ import annotations

import argparse

from port_h2_contracts.uncertainty_bundle import UncertaintyBundle
from port_h2_uncertainty_builders.combine import combine_bundles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wind_manifest")
    parser.add_argument("ship_manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    joint = combine_bundles(
        UncertaintyBundle.load(args.wind_manifest),
        UncertaintyBundle.load(args.ship_manifest),
        args.output,
    )
    print(joint.bundle_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
