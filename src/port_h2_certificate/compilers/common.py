"""Shared deterministic IR evaluation helpers."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from port_h2_certificate.recourse_ir import ConstraintRow, VariableKey


Realization = Mapping[str, Sequence[float]]
Values = Mapping[VariableKey, float]
_INDEXED_OUTPUT = re.compile(r"^(?P<key>.+)\[(?P<period>\d+)]$")


def parse_output_reference(reference: str) -> tuple[str, int]:
    match = _INDEXED_OUTPUT.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid indexed uncertain output reference: {reference}")
    return match.group("key"), int(match.group("period"))


def validate_realization_references(
    rows: Sequence[ConstraintRow], realization: Realization
) -> None:
    for row in rows:
        for reference in row.uncertain_output_coefficients:
            output_key, period = parse_output_reference(reference)
            if output_key not in realization:
                raise KeyError(f"missing uncertain output: {output_key}")
            if period >= len(realization[output_key]):
                raise IndexError(
                    f"uncertain output {output_key} has no period {period}"
                )


def evaluate_row_rhs(
    row: ConstraintRow, first_stage: Values, realization: Realization
) -> float:
    value = row.rhs_constant
    for key, coefficient in row.first_stage_coefficients.items():
        if key not in first_stage:
            raise KeyError(f"missing first-stage value: {key}")
        value += coefficient * float(first_stage[key])
    for reference, coefficient in row.uncertain_output_coefficients.items():
        output_key, period = parse_output_reference(reference)
        value += coefficient * float(realization[output_key][period])
    return float(value)


def configure_formal_lp(model) -> None:
    model.Params.OutputFlag = 0
    model.Params.Method = 1
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    model.Params.NumericFocus = 1
