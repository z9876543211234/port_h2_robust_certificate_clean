"""Immutable interface shared by independent uncertainty builders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Literal, Mapping

import gurobipy as gp
import numpy as np

from port_h2_contracts.hashing import canonical_sha256, is_sha256
from port_h2_contracts.sparse import SparseMatrix, load_sparse_archive


@dataclass(frozen=True)
class SelectorSpec:
    key: str
    namespace: str
    domain: Literal["binary"]
    role: Literal["primary", "auxiliary"]


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    coefficients: Mapping[str, float]
    sense: Literal["le", "eq", "ge"]
    rhs: float


@dataclass(frozen=True)
class AffineOutput:
    key: str
    nominal: tuple[float, ...]
    selector_columns_path: str
    unit: str


@dataclass(frozen=True)
class UncertaintyBundle:
    schema_version: int
    bundle_type: str
    profile_name: str
    model_horizon: int
    selectors: tuple[SelectorSpec, ...]
    constraints: tuple[LinearConstraint, ...]
    outputs: Mapping[str, AffineOutput]
    nominal_selector: Mapping[str, int]
    primary_selector_keys: tuple[str, ...]
    proofs: Mapping[str, object]
    source_hashes: Mapping[str, str]
    bundle_sha256: str
    storage_root: Path = field(default=Path("."), repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        schema_version: int,
        bundle_type: str,
        profile_name: str,
        model_horizon: int,
        selectors: tuple[SelectorSpec, ...],
        constraints: tuple[LinearConstraint, ...],
        outputs: Mapping[str, AffineOutput],
        nominal_selector: Mapping[str, int],
        primary_selector_keys: tuple[str, ...],
        proofs: Mapping[str, object],
        source_hashes: Mapping[str, str],
        storage_root: str | Path,
    ) -> "UncertaintyBundle":
        provisional = cls(
            schema_version=schema_version,
            bundle_type=bundle_type,
            profile_name=profile_name,
            model_horizon=model_horizon,
            selectors=tuple(selectors),
            constraints=tuple(constraints),
            outputs=dict(outputs),
            nominal_selector=dict(nominal_selector),
            primary_selector_keys=tuple(primary_selector_keys),
            proofs=dict(proofs),
            source_hashes=dict(source_hashes),
            bundle_sha256="",
            storage_root=Path(storage_root),
        )
        bundle = provisional._rehash()
        bundle.validate()
        return bundle

    @property
    def selector_keys(self) -> tuple[str, ...]:
        return tuple(selector.key for selector in self.selectors)

    def _matrix_reference(self, output: AffineOutput) -> tuple[Path, str | None]:
        path_text, separator, archive_name = output.selector_columns_path.partition("::")
        path = Path(path_text)
        resolved = path if path.is_absolute() else self.storage_root / path
        return resolved, archive_name if separator else None

    def _load_output_matrix(self, output: AffineOutput) -> SparseMatrix:
        path, archive_name = self._matrix_reference(output)
        if archive_name is None:
            return SparseMatrix.load(path)
        return load_sparse_archive(path, archive_name)

    def output_matrix(self, output_key: str) -> SparseMatrix:
        try:
            output = self.outputs[output_key]
        except KeyError as error:
            raise KeyError(f"unknown Bundle output: {output_key}") from error
        return self._load_output_matrix(output)

    def _hash_payload(self) -> dict[str, object]:
        output_payload: dict[str, object] = {}
        for key, output in sorted(self.outputs.items()):
            matrix = self._load_output_matrix(output)
            output_payload[key] = {
                "key": output.key,
                "nominal": output.nominal,
                "selector_columns_path": output.selector_columns_path,
                "selector_columns_sha256": matrix.sha256,
                "unit": output.unit,
            }
        return {
            "schema_version": self.schema_version,
            "bundle_type": self.bundle_type,
            "profile_name": self.profile_name,
            "model_horizon": self.model_horizon,
            "selectors": [
                {
                    "key": item.key,
                    "namespace": item.namespace,
                    "domain": item.domain,
                    "role": item.role,
                }
                for item in self.selectors
            ],
            "constraints": [
                {
                    "name": row.name,
                    "coefficients": dict(row.coefficients),
                    "sense": row.sense,
                    "rhs": row.rhs,
                }
                for row in self.constraints
            ],
            "outputs": output_payload,
            "nominal_selector": dict(self.nominal_selector),
            "primary_selector_keys": self.primary_selector_keys,
            "proofs": dict(self.proofs),
            "source_hashes": dict(self.source_hashes),
        }

    def _rehash(self) -> "UncertaintyBundle":
        return replace(self, bundle_sha256=canonical_sha256(self._hash_payload()))

    def with_nominal_selector(self, assignment: Mapping[str, int]) -> "UncertaintyBundle":
        return replace(self, nominal_selector=dict(assignment))._rehash()

    def with_constraints(
        self, constraints: tuple[LinearConstraint, ...]
    ) -> "UncertaintyBundle":
        return replace(self, constraints=tuple(constraints))._rehash()

    def with_profile(
        self, *, profile_name: str, model_horizon: int
    ) -> "UncertaintyBundle":
        return replace(
            self, profile_name=profile_name, model_horizon=model_horizon
        )._rehash()

    def validate(self) -> None:
        if not isinstance(self.schema_version, int) or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")
        if not self.bundle_type or not self.profile_name:
            raise ValueError("bundle_type and profile_name must be non-empty")
        if not isinstance(self.model_horizon, int) or self.model_horizon <= 0:
            raise ValueError("model_horizon must be a positive integer")

        keys = self.selector_keys
        if len(set(keys)) != len(keys):
            raise ValueError("selector keys must be unique")
        for selector in self.selectors:
            if not selector.key or not selector.namespace:
                raise ValueError("selector key and namespace must be non-empty")
            if selector.domain != "binary":
                raise ValueError("all bundle selectors must be binary")
            if selector.role not in {"primary", "auxiliary"}:
                raise ValueError("selector role must be primary or auxiliary")

        primary = tuple(item.key for item in self.selectors if item.role == "primary")
        if self.primary_selector_keys != primary:
            raise ValueError("primary_selector_keys must match primary selector order")

        key_set = set(keys)
        for row in self.constraints:
            if not row.name or row.sense not in {"le", "eq", "ge"}:
                raise ValueError("invalid linear uncertainty constraint")
            unknown = set(row.coefficients) - key_set
            if unknown:
                raise ValueError(f"constraint references unknown selector: {sorted(unknown)}")
            if not math.isfinite(row.rhs) or not all(
                math.isfinite(value) for value in row.coefficients.values()
            ):
                raise ValueError("constraint coefficients and rhs must be finite")

        if set(self.nominal_selector) != key_set:
            raise ValueError("nominal selector must assign every selector")
        self._validate_assignment(self.nominal_selector)
        if not self._is_feasible(self.nominal_selector):
            raise ValueError("nominal selector violates bundle constraints")

        for key, output in self.outputs.items():
            if key != output.key:
                raise ValueError("output mapping key must equal AffineOutput.key")
            if not output.unit:
                raise ValueError("output unit must be non-empty")
            if not all(math.isfinite(value) for value in output.nominal):
                raise ValueError("output nominal values must be finite")
            matrix = self._load_output_matrix(output)
            if matrix.shape != (len(output.nominal), len(self.selectors)):
                raise ValueError("output matrix shape does not match output/selectors")

        if not all(is_sha256(value) for value in self.source_hashes.values()):
            raise ValueError("source_hashes values must be SHA-256 digests")
        expected_hash = canonical_sha256(self._hash_payload())
        if self.bundle_sha256 != expected_hash:
            raise ValueError("bundle_sha256 does not match canonical bundle payload")

    def _validate_assignment(self, assignment: Mapping[str, int]) -> None:
        if set(assignment) != set(self.selector_keys):
            raise ValueError("full selector assignment must assign every selector")
        if any(value not in {0, 1} for value in assignment.values()):
            raise ValueError("selector assignments must be binary")

    def _is_feasible(self, assignment: Mapping[str, int], tolerance: float = 1e-9) -> bool:
        for row in self.constraints:
            lhs = sum(
                coefficient * assignment[key]
                for key, coefficient in row.coefficients.items()
            )
            if row.sense == "le" and lhs > row.rhs + tolerance:
                return False
            if row.sense == "ge" and lhs < row.rhs - tolerance:
                return False
            if row.sense == "eq" and abs(lhs - row.rhs) > tolerance:
                return False
        return True

    def evaluate(self, full_selector_assignment: Mapping[str, int]) -> dict[str, np.ndarray]:
        self._validate_assignment(full_selector_assignment)
        if not self._is_feasible(full_selector_assignment):
            raise ValueError("selector assignment is infeasible for this bundle")
        selector_vector = [full_selector_assignment[key] for key in self.selector_keys]
        return {
            key: np.asarray(output.nominal, dtype=float)
            + self._load_output_matrix(output).matvec(selector_vector)
            for key, output in self.outputs.items()
        }

    def complete(self, primary_selector_assignment: Mapping[str, int]) -> dict[str, int]:
        unknown = set(primary_selector_assignment) - set(self.primary_selector_keys)
        if unknown:
            raise ValueError(f"completion received non-primary selector keys: {sorted(unknown)}")
        if any(value not in {0, 1} for value in primary_selector_assignment.values()):
            raise ValueError("primary selector assignments must be binary")

        model = gp.Model("bundle_completion")
        model.Params.OutputFlag = 0
        variables = {
            key: model.addVar(vtype=gp.GRB.BINARY, name=key) for key in self.selector_keys
        }
        for key, value in primary_selector_assignment.items():
            model.addConstr(variables[key] == value, name=f"fix[{key}]")
        for row in self.constraints:
            expression = gp.quicksum(
                coefficient * variables[key]
                for key, coefficient in row.coefficients.items()
            )
            if row.sense == "le":
                model.addConstr(expression <= row.rhs, name=row.name)
            elif row.sense == "ge":
                model.addConstr(expression >= row.rhs, name=row.name)
            else:
                model.addConstr(expression == row.rhs, name=row.name)
        model.setObjective(0.0, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise ValueError("primary selector assignment has no feasible completion")
        return {key: int(round(variable.X)) for key, variable in variables.items()}

    def realization_hash(self, full_selector_assignment: Mapping[str, int]) -> str:
        realization = self.evaluate(full_selector_assignment)
        return canonical_sha256(
            {key: values.tolist() for key, values in sorted(realization.items())}
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "bundle_type": self.bundle_type,
            "profile_name": self.profile_name,
            "model_horizon": self.model_horizon,
            "selectors": [
                {
                    "key": item.key,
                    "namespace": item.namespace,
                    "domain": item.domain,
                    "role": item.role,
                }
                for item in self.selectors
            ],
            "constraints": [
                {
                    "name": row.name,
                    "coefficients": dict(row.coefficients),
                    "sense": row.sense,
                    "rhs": row.rhs,
                }
                for row in self.constraints
            ],
            "outputs": {
                key: {
                    "key": output.key,
                    "nominal": list(output.nominal),
                    "selector_columns_path": output.selector_columns_path,
                    "unit": output.unit,
                }
                for key, output in self.outputs.items()
            },
            "nominal_selector": dict(self.nominal_selector),
            "primary_selector_keys": list(self.primary_selector_keys),
            "proofs": dict(self.proofs),
            "source_hashes": dict(self.source_hashes),
            "bundle_sha256": self.bundle_sha256,
        }

    def save(self, manifest_path: str | Path) -> None:
        destination = Path(manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, manifest_path: str | Path) -> "UncertaintyBundle":
        source = Path(manifest_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        bundle = cls(
            schema_version=int(payload["schema_version"]),
            bundle_type=str(payload["bundle_type"]),
            profile_name=str(payload["profile_name"]),
            model_horizon=int(payload["model_horizon"]),
            selectors=tuple(SelectorSpec(**item) for item in payload["selectors"]),
            constraints=tuple(
                LinearConstraint(
                    name=item["name"],
                    coefficients=dict(item["coefficients"]),
                    sense=item["sense"],
                    rhs=float(item["rhs"]),
                )
                for item in payload["constraints"]
            ),
            outputs={
                key: AffineOutput(
                    key=item["key"],
                    nominal=tuple(float(value) for value in item["nominal"]),
                    selector_columns_path=item["selector_columns_path"],
                    unit=item["unit"],
                )
                for key, item in payload["outputs"].items()
            },
            nominal_selector={
                key: int(value) for key, value in payload["nominal_selector"].items()
            },
            primary_selector_keys=tuple(payload["primary_selector_keys"]),
            proofs=dict(payload["proofs"]),
            source_hashes=dict(payload["source_hashes"]),
            bundle_sha256=str(payload["bundle_sha256"]),
            storage_root=source.parent,
        )
        bundle.validate()
        return bundle
