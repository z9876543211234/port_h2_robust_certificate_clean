"""Cartesian-product composition of independent UncertaintyBundles."""

from __future__ import annotations

from pathlib import Path

from scipy import sparse as scipy_sparse

from port_h2_contracts.sparse import SparseMatrix, save_sparse_archive
from port_h2_contracts.uncertainty_bundle import AffineOutput, UncertaintyBundle


def combine_bundles(
    wind_bundle: UncertaintyBundle,
    ship_bundle: UncertaintyBundle,
    output_directory: str | Path,
) -> UncertaintyBundle:
    wind_bundle.validate()
    ship_bundle.validate()
    if (
        wind_bundle.profile_name != ship_bundle.profile_name
        or wind_bundle.model_horizon != ship_bundle.model_horizon
    ):
        raise ValueError("Bundle profile and model horizon must match")
    if set(wind_bundle.selector_keys) & set(ship_bundle.selector_keys):
        raise ValueError("Bundle selector keys must be disjoint")
    if set(wind_bundle.outputs) & set(ship_bundle.outputs):
        raise ValueError("Bundle output keys must be disjoint")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    wind_count = len(wind_bundle.selectors)
    ship_count = len(ship_bundle.selectors)
    archive_name = "joint_uncertainty_matrices.npz"
    matrices: dict[str, SparseMatrix] = {}
    outputs: dict[str, AffineOutput] = {}

    for key, output in wind_bundle.outputs.items():
        local = wind_bundle.output_matrix(key).to_scipy()
        zeros = scipy_sparse.csr_matrix((local.shape[0], ship_count))
        matrices[key] = SparseMatrix.from_scipy(
            scipy_sparse.hstack([local, zeros], format="csr")
        )
        outputs[key] = AffineOutput(
            key=key,
            nominal=output.nominal,
            selector_columns_path=f"{archive_name}::{key}",
            unit=output.unit,
        )

    for key, output in ship_bundle.outputs.items():
        local = ship_bundle.output_matrix(key).to_scipy()
        zeros = scipy_sparse.csr_matrix((local.shape[0], wind_count))
        matrices[key] = SparseMatrix.from_scipy(
            scipy_sparse.hstack([zeros, local], format="csr")
        )
        outputs[key] = AffineOutput(
            key=key,
            nominal=output.nominal,
            selector_columns_path=f"{archive_name}::{key}",
            unit=output.unit,
        )

    save_sparse_archive(destination / archive_name, matrices)
    source_hashes = {
        **{
            f"wind.{key}": value
            for key, value in wind_bundle.source_hashes.items()
        },
        **{
            f"ship_delay.{key}": value
            for key, value in ship_bundle.source_hashes.items()
        },
    }
    joint = UncertaintyBundle.create(
        schema_version=max(wind_bundle.schema_version, ship_bundle.schema_version),
        bundle_type="joint_wind_ship_delay",
        profile_name=wind_bundle.profile_name,
        model_horizon=wind_bundle.model_horizon,
        selectors=wind_bundle.selectors + ship_bundle.selectors,
        constraints=wind_bundle.constraints + ship_bundle.constraints,
        outputs=outputs,
        nominal_selector={
            **wind_bundle.nominal_selector,
            **ship_bundle.nominal_selector,
        },
        primary_selector_keys=(
            wind_bundle.primary_selector_keys + ship_bundle.primary_selector_keys
        ),
        proofs={
            "composition": "cartesian_product",
            "joint_budget_added": False,
            "wind_bundle_sha256": wind_bundle.bundle_sha256,
            "ship_bundle_sha256": ship_bundle.bundle_sha256,
        },
        source_hashes=source_hashes,
        storage_root=destination,
    )
    joint.save(destination / "joint_uncertainty_bundle.json")
    return joint

