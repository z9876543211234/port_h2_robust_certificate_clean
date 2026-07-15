"""Build the exact three-day integer ship-delay UncertaintyBundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from port_h2_contracts.sparse import SparseMatrix
from port_h2_contracts.uncertainty_bundle import AffineOutput, UncertaintyBundle
from port_h2_uncertainty_builders.ship_delay.delay_variables import (
    build_delay_variables,
)
from port_h2_uncertainty_builders.ship_delay.projection import build_ship_projection
from port_h2_uncertainty_builders.ship_delay.quay_crane import (
    encode_quay_crane_saturation,
)
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource


def _expanded_columns(
    primary_columns: np.ndarray, selector_count: int
) -> np.ndarray:
    result = np.zeros((primary_columns.shape[0], selector_count), dtype=float)
    result[:, : primary_columns.shape[1]] = primary_columns
    return result


def _rows_to_matrix(
    rows: list[dict[str, float]], selector_keys: tuple[str, ...]
) -> np.ndarray:
    indices = {key: index for index, key in enumerate(selector_keys)}
    matrix = np.zeros((len(rows), len(selector_keys)), dtype=float)
    for row, coefficients in enumerate(rows):
        for key, value in coefficients.items():
            matrix[row, indices[key]] = value
    return matrix


def build_ship_delay_bundle(
    source: ShipDelaySource, output_directory: str | Path
) -> UncertaintyBundle:
    source.validate()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    delay_variables, primary_selectors, primary_constraints = build_delay_variables(
        source
    )
    projection = build_ship_projection(source, delay_variables)
    primary_keys = tuple(selector.key for selector in primary_selectors)
    unsaturated_qc_nominal = (
        source.quay_cranes_per_ship * projection.in_port_nominal
    )
    unsaturated_qc_columns = (
        source.quay_cranes_per_ship * projection.in_port_columns
    )
    quay_crane = encode_quay_crane_saturation(
        primary_keys=primary_keys,
        primary_constraints=primary_constraints,
        unsaturated_nominal=unsaturated_qc_nominal,
        unsaturated_columns=unsaturated_qc_columns,
        installed_capacity=source.installed_quay_cranes,
    )

    selectors = tuple(primary_selectors + quay_crane.selectors)
    selector_keys = tuple(selector.key for selector in selectors)
    constraints = tuple(primary_constraints + quay_crane.constraints)
    selector_count = len(selectors)
    active_qc_columns = _rows_to_matrix(
        quay_crane.coefficient_rows, selector_keys
    )

    nominal_and_primary = {
        "ship.arrival_count": (
            projection.arrival_nominal,
            projection.arrival_columns,
            "ship",
        ),
        "ship.in_port_count": (
            projection.in_port_nominal,
            projection.in_port_columns,
            "ship",
        ),
        "ship.shore_power_kw": (
            source.shore_power_per_ship_kw * projection.in_port_nominal,
            source.shore_power_per_ship_kw * projection.in_port_columns,
            "kW",
        ),
        "ship.delayed_out_of_current_day_count": (
            projection.delayed_out_nominal,
            projection.delayed_out_columns,
            "ship",
        ),
    }
    full_outputs: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        key: (nominal, _expanded_columns(columns, selector_count), unit)
        for key, (nominal, columns, unit) in nominal_and_primary.items()
    }
    full_outputs.update(
        {
            "ship.active_quay_cranes": (
                quay_crane.nominal,
                active_qc_columns,
                "quay_crane",
            ),
            "ship.quay_crane_power_kw": (
                source.quay_crane_power_kw * quay_crane.nominal,
                source.quay_crane_power_kw * active_qc_columns,
                "kW",
            ),
            "ship.task_release": (
                source.quay_crane_task_rate_per_hour
                * source.profile.dt_hours
                * quay_crane.nominal,
                source.quay_crane_task_rate_per_hour
                * source.profile.dt_hours
                * active_qc_columns,
                "task",
            ),
        }
    )

    outputs: dict[str, AffineOutput] = {}
    for key, (nominal, columns, unit) in full_outputs.items():
        file_name = key.replace(".", "_") + "_columns.npz"
        SparseMatrix.from_dense(columns).save(destination / file_name)
        outputs[key] = AffineOutput(
            key=key,
            nominal=tuple(float(value) for value in nominal),
            selector_columns_path=file_name,
            unit=unit,
        )

    nominal_selector = {key: 0 for key in selector_keys}
    nominal_selector.update(quay_crane.nominal_auxiliary)
    source_hashes = {
        "raw_file_sha256": source.provenance.raw_file_sha256,
        "canonical_payload_sha256": source.provenance.canonical_payload_sha256,
    }
    if source.provenance.generation_script_sha256 is not None:
        source_hashes["generation_script_sha256"] = (
            source.provenance.generation_script_sha256
        )

    bundle = UncertaintyBundle.create(
        schema_version=1,
        bundle_type="ship_delay",
        profile_name=source.profile.profile_name,
        model_horizon=source.profile.periods,
        selectors=selectors,
        constraints=constraints,
        outputs=outputs,
        nominal_selector=nominal_selector,
        primary_selector_keys=primary_keys,
        proofs={
            "ship_builder_horizon": 3 * source.profile.periods,
            "ship_projection_window": [
                source.profile.periods,
                2 * source.profile.periods,
            ],
            "three_day_nominal_total": int(np.sum(projection.extended_nominal)),
            "three_day_mass_conservation": bool(
                np.allclose(np.sum(projection.extended_columns, axis=0), 0.0)
            ),
            "modulo_wraparound_used": False,
            "early_arrival_selectors": 0,
            "quay_crane_minimum": quay_crane.minimum.tolist(),
            "quay_crane_maximum": quay_crane.maximum.tolist(),
            "quay_crane_bound_statuses": list(quay_crane.statuses),
            "quay_crane_saturation_classification": list(
                quay_crane.classifications
            ),
        },
        source_hashes=source_hashes,
        storage_root=destination,
    )
    bundle.save(destination / "ship_delay_uncertainty_bundle.json")
    return bundle

