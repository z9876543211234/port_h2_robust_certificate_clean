"""Validate one resolved input package and build its immutable joint Bundle."""

from __future__ import annotations

import argparse
import json

from port_h2_certificate.load_case import UnresolvedInputError, load_case
from runners.common import case_path, parse_resolutions
from runners.input_adapters import build_joint_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--profile", required=True, choices=("hourly_24", "quarter_hour_96"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        loaded = load_case(
            case_path(args.case), args.profile, resolutions=parse_resolutions(args.set)
        )
    except UnresolvedInputError as error:
        print(json.dumps({"status": "unresolved", "paths": error.paths}, ensure_ascii=False))
        return 2
    bundle = build_joint_bundle(loaded, args.output)
    loaded.case.validate(bundle)
    print(
        json.dumps(
            {
                "status": "valid",
                "case_name": loaded.case.case_name,
                "profile_name": args.profile,
                "case_semantic_sha256": loaded.case_semantic_sha256,
                "joint_bundle_sha256": bundle.bundle_sha256,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
