"""Small immutable CSR matrix contract with stable content hashing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections.abc import Mapping

import numpy as np
from scipy import sparse as scipy_sparse

from port_h2_contracts.hashing import canonical_sha256


@dataclass(frozen=True)
class SparseMatrix:
    shape: tuple[int, int]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    def __post_init__(self) -> None:
        rows, columns = self.shape
        if rows < 0 or columns < 0:
            raise ValueError("sparse matrix dimensions must be non-negative")
        if self.indptr.ndim != 1 or self.indices.ndim != 1 or self.data.ndim != 1:
            raise ValueError("CSR arrays must be one-dimensional")
        if len(self.indptr) != rows + 1:
            raise ValueError("CSR indptr length does not match row count")
        if len(self.indices) != len(self.data):
            raise ValueError("CSR indices and data lengths differ")
        if len(self.indptr) and int(self.indptr[-1]) != len(self.data):
            raise ValueError("CSR indptr terminal value does not match nonzeros")
        if np.any(self.indices < 0) or np.any(self.indices >= columns):
            raise ValueError("CSR column index is outside matrix shape")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("sparse matrix data must be finite")

    @classmethod
    def from_dense(cls, matrix: np.ndarray | Iterable[Iterable[float]]) -> "SparseMatrix":
        dense = np.asarray(matrix, dtype=float)
        if dense.ndim != 2:
            raise ValueError("dense input must be two-dimensional")
        if not np.all(np.isfinite(dense)):
            raise ValueError("sparse matrix values must be finite")
        csr = scipy_sparse.csr_matrix(dense)
        csr.sort_indices()
        return cls(
            shape=(int(csr.shape[0]), int(csr.shape[1])),
            indptr=np.asarray(csr.indptr, dtype=np.int64),
            indices=np.asarray(csr.indices, dtype=np.int64),
            data=np.asarray(csr.data, dtype=np.float64),
        )

    @classmethod
    def from_scipy(cls, matrix: scipy_sparse.spmatrix) -> "SparseMatrix":
        csr = scipy_sparse.csr_matrix(matrix, dtype=np.float64)
        csr.sort_indices()
        if not np.all(np.isfinite(csr.data)):
            raise ValueError("sparse matrix values must be finite")
        return cls(
            shape=(int(csr.shape[0]), int(csr.shape[1])),
            indptr=np.asarray(csr.indptr, dtype=np.int64),
            indices=np.asarray(csr.indices, dtype=np.int64),
            data=np.asarray(csr.data, dtype=np.float64),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "shape": self.shape,
                "indptr": self.indptr.tolist(),
                "indices": self.indices.tolist(),
                "data": self.data.tolist(),
            }
        )

    def matvec(self, vector: Iterable[float]) -> np.ndarray:
        values = np.asarray(tuple(vector), dtype=float)
        if values.shape != (self.shape[1],):
            raise ValueError("vector length does not match sparse matrix columns")
        return self.to_scipy().dot(values)

    def to_scipy(self) -> scipy_sparse.csr_matrix:
        return scipy_sparse.csr_matrix(
            (self.data, self.indices, self.indptr), shape=self.shape, copy=True
        )

    def to_dense(self) -> np.ndarray:
        return self.to_scipy().toarray()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            shape=np.asarray(self.shape, dtype=np.int64),
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SparseMatrix":
        with np.load(Path(path), allow_pickle=False) as payload:
            shape_values = np.asarray(payload["shape"], dtype=np.int64)
            if shape_values.shape != (2,):
                raise ValueError("stored sparse matrix shape is invalid")
            return cls(
                shape=(int(shape_values[0]), int(shape_values[1])),
                indptr=np.asarray(payload["indptr"], dtype=np.int64),
                indices=np.asarray(payload["indices"], dtype=np.int64),
                data=np.asarray(payload["data"], dtype=np.float64),
            )


def save_sparse_archive(
    path: str | Path, matrices: Mapping[str, SparseMatrix]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(matrices)
    payload: dict[str, np.ndarray] = {"names": np.asarray(names, dtype=np.str_)}
    for index, name in enumerate(names):
        matrix = matrices[name]
        payload[f"matrix_{index}_shape"] = np.asarray(matrix.shape, dtype=np.int64)
        payload[f"matrix_{index}_indptr"] = matrix.indptr
        payload[f"matrix_{index}_indices"] = matrix.indices
        payload[f"matrix_{index}_data"] = matrix.data
    np.savez_compressed(destination, **payload)


def load_sparse_archive(path: str | Path, name: str) -> SparseMatrix:
    with np.load(Path(path), allow_pickle=False) as payload:
        names = [str(value) for value in payload["names"].tolist()]
        try:
            index = names.index(name)
        except ValueError as error:
            raise KeyError(f"sparse archive has no matrix named {name!r}") from error
        shape = np.asarray(payload[f"matrix_{index}_shape"], dtype=np.int64)
        return SparseMatrix(
            shape=(int(shape[0]), int(shape[1])),
            indptr=np.asarray(payload[f"matrix_{index}_indptr"], dtype=np.int64),
            indices=np.asarray(payload[f"matrix_{index}_indices"], dtype=np.int64),
            data=np.asarray(payload[f"matrix_{index}_data"], dtype=np.float64),
        )
