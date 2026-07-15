"""Adapters from generic loaded payloads to independent builder contracts."""

from __future__ import annotations

from pathlib import Path

from port_h2_certificate.load_case import LoadedCase
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.combine import combine_bundles
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource
from port_h2_uncertainty_builders.wind.builder import build_wind_bundle
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


def build_joint_bundle(loaded: LoadedCase, output_directory: str | Path):
    destination = Path(output_directory)
    wind = loaded.wind_source_payload
    ship = loaded.ship_delay_payload
    wind_source = WindUncertaintySource(
        profile=loaded.case.profile,
        nominal_power_kw=tuple(float(value) for value in wind["nominal_power_kw"]),
        deviation_down_kw=tuple(float(value) for value in wind["deviation_down_kw"]),
        deviation_up_kw=tuple(float(value) for value in wind["deviation_up_kw"]),
        budget=int(wind["budget"]),
        provenance=ProvenanceRecord.from_dict(wind["provenance"]),
    )
    ship_source = ShipDelaySource(
        profile=loaded.case.profile,
        previous_day_arrival_count=tuple(int(value) for value in ship["previous_day_arrival_count"]),
        current_day_arrival_count=tuple(int(value) for value in ship["current_day_arrival_count"]),
        next_day_arrival_count=tuple(int(value) for value in ship["next_day_arrival_count"]),
        dwell_steps=int(ship["dwell_steps"]),
        max_delay_steps=int(ship["max_delay_steps"]),
        delayed_ship_budget=int(ship["delayed_ship_budget"]),
        total_delay_step_budget=(
            None
            if ship["total_delay_step_budget"] is None
            else int(ship["total_delay_step_budget"])
        ),
        shore_power_per_ship_kw=float(ship["shore_power_per_ship_kw"]),
        quay_cranes_per_ship=float(ship["quay_cranes_per_ship"]),
        installed_quay_cranes=float(ship["installed_quay_cranes"]),
        quay_crane_power_kw=float(ship["quay_crane_power_kw"]),
        quay_crane_task_rate_per_hour=float(ship["quay_crane_task_rate_per_hour"]),
        provenance=ProvenanceRecord.from_dict(ship["provenance"]),
    )
    wind_bundle = build_wind_bundle(wind_source, destination / "wind")
    ship_bundle = build_ship_delay_bundle(ship_source, destination / "ship_delay")
    return combine_bundles(wind_bundle, ship_bundle, destination / "joint")
