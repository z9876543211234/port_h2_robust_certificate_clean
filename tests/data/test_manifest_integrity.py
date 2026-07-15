from __future__ import annotations

import hashlib
import json
from pathlib import Path

from port_h2_contracts.hashing import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_generated_data_manifest_and_embedded_provenance_are_self_consistent() -> None:
    data_root = PROJECT_ROOT / "data"
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = PROJECT_ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    payload_paths = [
        *(data_root / "deterministic").glob("*/physical.json"),
        *(data_root / "uncertainty_sources" / "wind").glob("*.json"),
        *(data_root / "uncertainty_sources" / "ship_delay").glob("*.json"),
    ]
    script_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "tools" / "migrate_authoritative_inputs.py").read_bytes()
    ).hexdigest()
    for path in payload_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.pop("provenance")
        assert provenance["canonical_payload_sha256"] == canonical_sha256(payload)
        if provenance["source_type"] == "generated":
            generator = PROJECT_ROOT / payload["nominal_generation"][
                "generator_reference"
            ]
            expected_generator_sha256 = hashlib.sha256(
                generator.read_bytes()
            ).hexdigest()
            assert provenance["generation_script_sha256"] == expected_generator_sha256
            raw_source = PROJECT_ROOT / provenance["source_reference"]
            assert provenance["raw_file_sha256"] == hashlib.sha256(
                raw_source.read_bytes()
            ).hexdigest()
        else:
            assert provenance["generation_script_sha256"] == script_sha256
