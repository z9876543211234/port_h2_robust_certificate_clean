"""Container binding the two linear-program stages without duplicating physics."""

from __future__ import annotations

from dataclasses import dataclass

from port_h2_certificate.first_stage_ir import FirstStageIR
from port_h2_certificate.recourse_ir import RecourseIR


@dataclass(frozen=True)
class TwoStageIR:
    first_stage: FirstStageIR
    recourse: RecourseIR

    def validate(self) -> None:
        first_keys = set(self.first_stage.variable_keys)
        referenced = {
            key
            for row in self.recourse.constraints
            for key in row.first_stage_coefficients
        }
        unknown = referenced - first_keys
        if unknown:
            raise ValueError(
                f"recourse rows reference unknown first-stage variables: {sorted(unknown)}"
            )

