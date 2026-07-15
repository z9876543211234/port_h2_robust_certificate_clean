"""Validate a saved standardized Bundle manifest."""

from __future__ import annotations

import argparse

from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    bundle = UncertaintyBundle.load(args.manifest)
    bundle.validate()
    print(bundle.bundle_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
