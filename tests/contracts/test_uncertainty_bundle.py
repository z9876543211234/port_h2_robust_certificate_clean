from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def _build_bundle(tmp_path):
    contracts = _module("port_h2_contracts.uncertainty_bundle")
    sparse = _module("port_h2_contracts.sparse")
    matrix_path = tmp_path / "columns.npz"
    sparse.SparseMatrix.from_dense(
        np.asarray([[-2.0, 0.0], [0.0, 3.0]], dtype=float)
    ).save(matrix_path)
    selectors = (
        contracts.SelectorSpec("wind.down[0]", "wind", "binary", "primary"),
        contracts.SelectorSpec("wind.aux[0]", "wind", "binary", "auxiliary"),
    )
    constraints = (
        contracts.LinearConstraint(
            "complete_aux", {"wind.aux[0]": 1.0, "wind.down[0]": -1.0}, "eq", 0.0
        ),
    )
    outputs = {
        "wind.available_power_kw": contracts.AffineOutput(
            key="wind.available_power_kw",
            nominal=(10.0, 20.0),
            selector_columns_path=matrix_path.name,
            unit="kW",
        )
    }
    bundle = contracts.UncertaintyBundle.create(
        schema_version=1,
        bundle_type="wind",
        profile_name="toy_2",
        model_horizon=2,
        selectors=selectors,
        constraints=constraints,
        outputs=outputs,
        nominal_selector={"wind.down[0]": 0, "wind.aux[0]": 0},
        primary_selector_keys=("wind.down[0]",),
        proofs={"fixture": True},
        source_hashes={"source": "a" * 64},
        storage_root=tmp_path,
    )
    return bundle


def test_bundle_evaluates_affine_outputs(tmp_path) -> None:
    bundle = _build_bundle(tmp_path)
    bundle.validate()
    realization = bundle.evaluate({"wind.down[0]": 1, "wind.aux[0]": 1})
    np.testing.assert_allclose(realization["wind.available_power_kw"], [8.0, 23.0])


def test_bundle_complete_is_generic_feasibility_completion(tmp_path) -> None:
    bundle = _build_bundle(tmp_path)
    completed = bundle.complete({"wind.down[0]": 1})
    assert completed == {"wind.down[0]": 1, "wind.aux[0]": 1}


def test_bundle_hash_and_realization_hash_are_stable(tmp_path) -> None:
    bundle = _build_bundle(tmp_path)
    rebuilt = _build_bundle(tmp_path)
    assert bundle.bundle_sha256 == rebuilt.bundle_sha256
    assignment = {"wind.down[0]": 0, "wind.aux[0]": 0}
    assert bundle.realization_hash(assignment) == rebuilt.realization_hash(assignment)


def test_bundle_json_round_trip_preserves_hash_and_outputs(tmp_path) -> None:
    contracts = _module("port_h2_contracts.uncertainty_bundle")
    bundle = _build_bundle(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    bundle.save(manifest_path)
    restored = contracts.UncertaintyBundle.load(manifest_path)
    assert restored.bundle_sha256 == bundle.bundle_sha256
    np.testing.assert_allclose(
        restored.evaluate(restored.nominal_selector)["wind.available_power_kw"],
        [10.0, 20.0],
    )


def test_bundle_rejects_infeasible_nominal_selector(tmp_path) -> None:
    bundle = _build_bundle(tmp_path)
    broken = bundle.with_nominal_selector({"wind.down[0]": 0, "wind.aux[0]": 1})
    with pytest.raises(ValueError, match="nominal selector"):
        broken.validate()


def test_bundle_rejects_unknown_constraint_selector(tmp_path) -> None:
    contracts = _module("port_h2_contracts.uncertainty_bundle")
    bundle = _build_bundle(tmp_path)
    broken = bundle.with_constraints(
        (
            contracts.LinearConstraint("unknown", {"missing": 1.0}, "le", 1.0),
        )
    )
    with pytest.raises(ValueError, match="unknown selector"):
        broken.validate()
