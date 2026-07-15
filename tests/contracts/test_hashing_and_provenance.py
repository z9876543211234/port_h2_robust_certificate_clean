from __future__ import annotations

import importlib
import importlib.util

import pytest


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def test_canonical_sha256_is_mapping_order_independent() -> None:
    hashing = _module("port_h2_contracts.hashing")
    left = {"b": [2.0, 3.0], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2.0, 3.0]}
    assert hashing.canonical_sha256(left) == hashing.canonical_sha256(right)
    assert len(hashing.canonical_sha256(left)) == 64


def test_canonical_sha256_rejects_non_finite_numbers() -> None:
    hashing = _module("port_h2_contracts.hashing")
    with pytest.raises(ValueError, match="finite"):
        hashing.canonical_sha256({"bad": float("nan")})


def test_generated_provenance_requires_seed_and_generation_hash() -> None:
    provenance = _module("port_h2_contracts.provenance")
    with pytest.raises(ValueError, match="random_seed"):
        provenance.ProvenanceRecord(
            source_type="generated",
            source_reference="toy generator",
            random_seed=None,
            generation_script_sha256="a" * 64,
            raw_file_sha256="b" * 64,
            canonical_payload_sha256="c" * 64,
        ).validate()


def test_fixed_file_provenance_round_trip() -> None:
    provenance = _module("port_h2_contracts.provenance")
    record = provenance.ProvenanceRecord(
        source_type="fixed_file",
        source_reference="data/source.yaml",
        random_seed=None,
        generation_script_sha256=None,
        raw_file_sha256="b" * 64,
        canonical_payload_sha256="c" * 64,
    )
    record.validate()
    assert provenance.ProvenanceRecord.from_dict(record.to_dict()) == record

