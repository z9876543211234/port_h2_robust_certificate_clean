"""Build the budgeted two-sided wind UncertaintyBundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from port_h2_contracts.sparse import SparseMatrix
from port_h2_contracts.uncertainty_bundle import (
    AffineOutput,
    LinearConstraint,
    SelectorSpec,
    UncertaintyBundle,
)
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


def build_wind_bundle(
    source: WindUncertaintySource, output_directory: str | Path
) -> UncertaintyBundle:
    source.validate()
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    selectors: list[SelectorSpec] = []
    constraints: list[LinearConstraint] = []
    columns = np.zeros((source.profile.periods, 2 * source.profile.periods))

    for period in source.profile.operating_periods:
        down_key = f"wind.down[{period}]"
        up_key = f"wind.up[{period}]"
        down_column = 2 * period
        up_column = down_column + 1
        selectors.extend(
            [
                SelectorSpec(down_key, "wind", "binary", "primary"),
                SelectorSpec(up_key, "wind", "binary", "primary"),
            ]
        )
        constraints.append(
            LinearConstraint(
                name=f"wind.mutual_exclusion[{period}]",
                coefficients={down_key: 1.0, up_key: 1.0},
                sense="le",
                rhs=1.0,
            )
        )
        columns[period, down_column] = -source.deviation_down_kw[period]
        columns[period, up_column] = source.deviation_up_kw[period]

    constraints.append(
        LinearConstraint(
            name="wind.deviation_budget",
            coefficients={selector.key: 1.0 for selector in selectors},
            sense="le",
            rhs=float(source.budget),
        )
    )

    matrix_name = "wind_uncertainty_matrices.npz"
    SparseMatrix.from_dense(columns).save(destination / matrix_name)
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
        bundle_type="wind",
        profile_name=source.profile.profile_name,
        model_horizon=source.profile.periods,
        selectors=tuple(selectors),
        constraints=tuple(constraints),
        outputs={
            "wind.available_power_kw": AffineOutput(
                key="wind.available_power_kw",
                nominal=tuple(source.nominal_power_kw),
                selector_columns_path=matrix_name,
                unit="kW",
            )
        },
        nominal_selector={selector.key: 0 for selector in selectors},
        primary_selector_keys=tuple(selector.key for selector in selectors),
        proofs={
            "mutual_exclusion_rows": source.profile.periods,
            "deviation_budget": source.budget,
            "lower_realization_nonnegative": True,
        },
        source_hashes=source_hashes,
        storage_root=destination,
    )
    bundle.save(destination / "wind_uncertainty_bundle.json")
    return bundle

