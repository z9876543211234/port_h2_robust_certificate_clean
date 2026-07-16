"""Load resolved formal case data without importing uncertainty builders."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from port_h2_certificate.schema import (
    AgvParameters,
    CaseData,
    CostParameters,
    GridParameters,
    HydrogenParameters,
    LogisticsParameters,
    LohcParameters,
)
from port_h2_contracts.hashing import canonical_sha256
from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord


class UnresolvedInputError(ValueError):
    def __init__(self, paths: tuple[str, ...]):
        self.paths = paths
        super().__init__("formal input contains unresolved values: " + ", ".join(paths))


@dataclass(frozen=True)
class LoadedCase:
    case: CaseData
    case_semantic_sha256: str
    wind_source_payload: Mapping[str, Any]
    ship_delay_payload: Mapping[str, Any]
    deterministic_provenance: ProvenanceRecord
    references: Mapping[str, str]


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _resolve_reference(case_path: Path, reference: str) -> Path:
    path = (case_path.parent / reference).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _apply_resolution(payloads: dict[str, dict[str, Any]], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) < 2 or parts[0] not in payloads:
        raise KeyError(f"unknown resolution path: {path}")
    current: Any = payloads[parts[0]]
    for part in parts[1:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"unknown resolution path: {path}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(f"unknown resolution path: {path}")
    current[parts[-1]] = value


def _required_unresolved(
    deterministic: Mapping[str, Any],
    ship_delay: Mapping[str, Any],
    *,
    hydrogen_chain: bool,
) -> tuple[str, ...]:
    paths: list[str] = []
    if hydrogen_chain:
        hydrogen = deterministic["hydrogen"]
        for field in ("landing_delay_steps", "historical_landing_kg"):
            if hydrogen[field] is None:
                paths.append(f"deterministic.hydrogen.{field}")
    for field in (
        "spill_day_ahead_per_kwh",
        "spill_real_time_per_kwh",
        "spill_deviation_per_kwh",
    ):
        if deterministic["cost"][field] is None:
            paths.append(f"deterministic.cost.{field}")
    for field in ("max_delay_steps", "delayed_ship_budget"):
        if ship_delay[field] is None:
            paths.append(f"ship_delay.{field}")
    return tuple(paths)


def _tuple_floats(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _build_case(
    case_config: Mapping[str, Any],
    profile: HorizonProfile,
    deterministic: Mapping[str, Any],
) -> CaseData:
    grid = deterministic["grid"]
    hydrogen_payload = deterministic["hydrogen"]
    lohc_payload = deterministic["lohc"]
    agv_payload = dict(deterministic["agv"])
    cost_payload = dict(deterministic["cost"])
    hydrogen_chain = bool(case_config["hydrogen_chain"])

    hydrogen = None
    lohc = None
    ratio = _tuple_floats(deterministic["wind_to_hydrogen_ratio"])
    if hydrogen_chain:
        hydrogen = HydrogenParameters(
            transport_efficiency=float(hydrogen_payload["transport_efficiency"]),
            conversion_kg_per_kwh=float(hydrogen_payload["conversion_kg_per_kwh"]),
            landing_delay_steps=int(hydrogen_payload["landing_delay_steps"]),
            historical_landing_kg=_tuple_floats(
                hydrogen_payload["historical_landing_kg"]
            ),
            initial_kg=float(hydrogen_payload["initial_kg"]),
            capacity_kg=float(hydrogen_payload["capacity_kg"]),
            terminal_min_kg=float(hydrogen_payload["terminal_min_kg"]),
        )
        lohc = LohcParameters(**{key: float(value) for key, value in lohc_payload.items()})
    else:
        ratio = (0.0,) * profile.periods
        agv_payload["lohc_transport_rate_kg_per_vehicle_hour"] = 0.0
        cost_payload["lohc_export_revenue_per_kg"] = 0.0

    return CaseData(
        case_name=case_config["case_name"],
        profile=profile,
        grid=GridParameters(
            buy_capacity_kw=float(grid["buy_capacity_kw"]),
            sell_capacity_kw=float(grid["sell_capacity_kw"]),
            buy_price_per_kwh=_tuple_floats(grid["buy_price_per_kwh"]),
            sell_price_per_kwh=_tuple_floats(grid["sell_price_per_kwh"]),
        ),
        base_load_kw=_tuple_floats(deterministic["base_load_kw"]),
        wind_to_hydrogen_ratio=ratio,
        hydrogen=hydrogen,
        lohc=lohc,
        agv=AgvParameters(
            **{key: (None if value is None else float(value)) for key, value in agv_payload.items()}
        ),
        logistics=LogisticsParameters(
            **{key: float(value) for key, value in deterministic["logistics"].items()}
        ),
        cost=CostParameters(
            **{key: float(value) for key, value in cost_payload.items()}
        ),
        outbound_mode=case_config["outbound_mode"],
        container_work_capacity=(
            None
            if case_config["container_work_capacity"] is None
            else float(case_config["container_work_capacity"])
        ),
        lohc_work_capacity=(
            None
            if case_config["lohc_work_capacity"] is None
            else float(case_config["lohc_work_capacity"])
        ),
        cost_scale=float(deterministic["cost_scale"]),
    )


def load_case(
    case_path: str | Path,
    profile_name: str,
    *,
    resolutions: Mapping[str, Any] | None = None,
) -> LoadedCase:
    source = Path(case_path).resolve()
    case_config = _read_mapping(source)
    try:
        references = case_config["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"case does not support profile {profile_name}") from error

    profile_payload = _read_mapping(
        _resolve_reference(source, references["profile_reference"])
    )
    profile = HorizonProfile.formal(
        profile_name=str(profile_payload["profile_name"]),
        periods=int(profile_payload["periods"]),
        dt_hours=float(profile_payload["dt_hours"]),
    )
    payloads = {
        "deterministic": deepcopy(
            _read_mapping(
                _resolve_reference(source, references["deterministic_reference"])
            )
        ),
        "wind": deepcopy(
            _read_mapping(_resolve_reference(source, references["wind_source_reference"]))
        ),
        "ship_delay": deepcopy(
            _read_mapping(
                _resolve_reference(source, references["ship_delay_source_reference"])
            )
        ),
    }
    payloads["deterministic"]["cost"].setdefault(
        "agv_operation_per_vehicle_hour", 0.0
    )
    for path, value in (resolutions or {}).items():
        _apply_resolution(payloads, path, value)

    unresolved = _required_unresolved(
        payloads["deterministic"],
        payloads["ship_delay"],
        hydrogen_chain=bool(case_config["hydrogen_chain"]),
    )
    if unresolved:
        raise UnresolvedInputError(unresolved)

    case = _build_case(case_config, profile, payloads["deterministic"])
    provenance = ProvenanceRecord.from_dict(payloads["deterministic"]["provenance"])
    semantic_hash = canonical_sha256(case.to_dict())
    return LoadedCase(
        case=case,
        case_semantic_sha256=semantic_hash,
        wind_source_payload=payloads["wind"],
        ship_delay_payload=payloads["ship_delay"],
        deterministic_provenance=provenance,
        references={key: str(value) for key, value in references.items()},
    )
