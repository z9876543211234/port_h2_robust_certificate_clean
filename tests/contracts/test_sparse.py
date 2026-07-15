from __future__ import annotations

import importlib
import importlib.util

import numpy as np


def _sparse_module():
    spec = importlib.util.find_spec("port_h2_contracts.sparse")
    assert spec is not None, "sparse contract is not implemented"
    return importlib.import_module("port_h2_contracts.sparse")


def test_sparse_matrix_round_trip_and_matvec(tmp_path) -> None:
    sparse = _sparse_module()
    matrix = sparse.SparseMatrix.from_dense(
        np.asarray([[1.0, 0.0, -2.0], [0.0, 3.0, 0.0]], dtype=float)
    )
    output = tmp_path / "matrix.npz"
    matrix.save(output)
    restored = sparse.SparseMatrix.load(output)
    np.testing.assert_allclose(restored.matvec([2.0, 4.0, 5.0]), [-8.0, 12.0])
    assert restored.shape == (2, 3)
    assert restored.sha256 == matrix.sha256


def test_sparse_matrix_rejects_non_finite_values() -> None:
    sparse = _sparse_module()
    with np.testing.assert_raises(ValueError):
        sparse.SparseMatrix.from_dense(np.asarray([[float("inf")]]))

