"""Single-source reduced recourse linear IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from port_h2_contracts.uncertainty_bundle import UncertaintyBundle
from port_h2_certificate.schema import CaseData


VariableKey = tuple[str, int]


@dataclass(frozen=True)
class VariableSpec:
    key: VariableKey
    lower_bound: float
    upper_bound: float | None
    objective_coefficient: float
    unit: str
    component: str


@dataclass(frozen=True)
class ConstraintRow:
    name: str
    sense: Literal["eq", "ge"]
    recourse_coefficients: Mapping[VariableKey, float]
    rhs_constant: float
    first_stage_coefficients: Mapping[VariableKey, float]
    uncertain_output_coefficients: Mapping[str, float]
    unit: str
    component: str
    equation_id: str


@dataclass(frozen=True)
class RecourseIR:
    variables: tuple[VariableSpec, ...]
    constraints: tuple[ConstraintRow, ...]

    @property
    def variable_keys(self) -> tuple[VariableKey, ...]:
        return tuple(variable.key for variable in self.variables)


class RecourseRegistry:
    def __init__(self) -> None:
        self.variables: list[VariableSpec] = []
        self.constraints: list[ConstraintRow] = []

    def add_variable(
        self,
        key: VariableKey,
        lower_bound: float,
        upper_bound: float | None,
        objective_coefficient: float,
        unit: str,
        component: str,
    ) -> None:
        self.variables.append(
            VariableSpec(
                key,
                lower_bound,
                upper_bound,
                objective_coefficient,
                unit,
                component,
            )
        )

    def add_row(
        self,
        name: str,
        sense: Literal["eq", "ge"],
        recourse: Mapping[VariableKey, float],
        rhs: float,
        first_stage: Mapping[VariableKey, float] | None,
        uncertain: Mapping[str, float] | None,
        unit: str,
        component: str,
        equation_id: str,
    ) -> None:
        self.constraints.append(
            ConstraintRow(
                name,
                sense,
                dict(recourse),
                rhs,
                dict(first_stage or {}),
                dict(uncertain or {}),
                unit,
                component,
                equation_id,
            )
        )

    def build(self) -> RecourseIR:
        keys = [variable.key for variable in self.variables]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate recourse variable key")
        key_set = set(keys)
        names = [row.name for row in self.constraints]
        if len(set(names)) != len(names):
            raise ValueError("duplicate recourse constraint name")
        for row in self.constraints:
            unknown = set(row.recourse_coefficients) - key_set
            if unknown:
                raise ValueError(f"row {row.name} references unknown variables {unknown}")
        return RecourseIR(tuple(self.variables), tuple(self.constraints))


def build_recourse_ir(case: CaseData, bundle: UncertaintyBundle) -> RecourseIR:
    case.validate(bundle)
    registry = RecourseRegistry()
    from port_h2_certificate.components import agv, electricity, hydrogen_lohc, logistics

    hydrogen_lohc.register(case, bundle, registry)
    agv.register(case, bundle, registry)
    electricity.register(case, bundle, registry)
    logistics.register(case, bundle, registry)
    return registry.build()

