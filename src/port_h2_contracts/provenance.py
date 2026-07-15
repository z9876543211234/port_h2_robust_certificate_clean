"""Input-data provenance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Any

from port_h2_contracts.hashing import is_sha256


SourceType = Literal["fixed_file", "empirical_record", "generated"]


@dataclass(frozen=True)
class ProvenanceRecord:
    source_type: SourceType
    source_reference: str
    random_seed: int | None
    generation_script_sha256: str | None
    raw_file_sha256: str
    canonical_payload_sha256: str

    def validate(self) -> None:
        if self.source_type not in {"fixed_file", "empirical_record", "generated"}:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if not self.source_reference.strip():
            raise ValueError("source_reference must be non-empty")
        if not is_sha256(self.raw_file_sha256):
            raise ValueError("raw_file_sha256 must be a SHA-256 hex digest")
        if not is_sha256(self.canonical_payload_sha256):
            raise ValueError("canonical_payload_sha256 must be a SHA-256 hex digest")
        if self.generation_script_sha256 is not None and not is_sha256(
            self.generation_script_sha256
        ):
            raise ValueError("generation_script_sha256 must be null or a SHA-256 digest")
        if self.source_type == "generated":
            if self.random_seed is None:
                raise ValueError("generated provenance requires random_seed")
            if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
                raise ValueError("random_seed must be an integer")
            if self.generation_script_sha256 is None:
                raise ValueError("generated provenance requires generation_script_sha256")
        elif self.random_seed is not None:
            raise ValueError("random_seed is only valid for generated data")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceRecord":
        record = cls(
            source_type=payload["source_type"],
            source_reference=str(payload["source_reference"]),
            random_seed=payload.get("random_seed"),
            generation_script_sha256=payload.get("generation_script_sha256"),
            raw_file_sha256=str(payload["raw_file_sha256"]),
            canonical_payload_sha256=str(payload["canonical_payload_sha256"]),
        )
        record.validate()
        return record

