"""Exact affine encoding of saturated quay-crane activity."""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
import numpy as np

from port_h2_contracts.uncertainty_bundle import LinearConstraint, SelectorSpec


@dataclass(frozen=True)
class QuayCraneEncoding:
    selectors: list[SelectorSpec]
    constraints: list[LinearConstraint]
    nominal: np.ndarray
    coefficient_rows: list[dict[str, float]]
    nominal_auxiliary: dict[str, int]
    minimum: np.ndarray
    maximum: np.ndarray
    classifications: tuple[str, ...]
    statuses: tuple[str, ...]


def _add_rows_to_model(model, variables, constraints) -> None:
    for row in constraints:
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


def _exact_bounds(
    primary_keys: tuple[str, ...],
    primary_constraints: list[LinearConstraint],
    nominal: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    model = gp.Model("ship_quay_crane_bounds")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.IntFeasTol = 1e-9
    variables = {
        key: model.addVar(vtype=gp.GRB.BINARY, name=key) for key in primary_keys
    }
    _add_rows_to_model(model, variables, primary_constraints)
    minimum = np.zeros(len(nominal), dtype=float)
    maximum = np.zeros(len(nominal), dtype=float)
    statuses: list[str] = []
    for period in range(len(nominal)):
        expression = nominal[period] + gp.quicksum(
            columns[period, column] * variables[key]
            for column, key in enumerate(primary_keys)
            if columns[period, column] != 0.0
        )
        model.setObjective(expression, gp.GRB.MINIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("quay-crane lower-bound MILP did not reach GRB.OPTIMAL")
        minimum[period] = model.ObjVal
        statuses.append("OPTIMAL")
        model.setObjective(expression, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError("quay-crane upper-bound MILP did not reach GRB.OPTIMAL")
        maximum[period] = model.ObjVal
        statuses.append("OPTIMAL")
    return minimum, maximum, tuple(statuses)


def encode_quay_crane_saturation(
    *,
    primary_keys: tuple[str, ...],
    primary_constraints: list[LinearConstraint],
    unsaturated_nominal: np.ndarray,
    unsaturated_columns: np.ndarray,
    installed_capacity: float,
) -> QuayCraneEncoding:
    minimum, maximum, statuses = _exact_bounds(
        primary_keys,
        primary_constraints,
        unsaturated_nominal,
        unsaturated_columns,
    )
    auxiliary_selectors: list[SelectorSpec] = []
    auxiliary_constraints: list[LinearConstraint] = []
    output_nominal = np.zeros(len(unsaturated_nominal), dtype=float)
    coefficient_rows: list[dict[str, float]] = []
    nominal_auxiliary: dict[str, int] = {}
    classifications: list[str] = []
    tolerance = 1e-9

    for period in range(len(unsaturated_nominal)):
        base = float(unsaturated_nominal[period])
        primary_coefficients = {
            key: float(unsaturated_columns[period, column])
            for column, key in enumerate(primary_keys)
            if abs(unsaturated_columns[period, column]) > tolerance
        }
        if maximum[period] <= installed_capacity + tolerance:
            classifications.append("always_nonsaturated")
            output_nominal[period] = base
            coefficient_rows.append(primary_coefficients)
            continue
        if minimum[period] >= installed_capacity - tolerance:
            classifications.append("always_saturated")
            output_nominal[period] = installed_capacity
            coefficient_rows.append({})
            continue

        classifications.append("crossing")
        saturation_key = f"ship.qc_saturated[{period}]"
        auxiliary_selectors.append(
            SelectorSpec(saturation_key, "ship_delay", "binary", "auxiliary")
        )
        nominal_auxiliary[saturation_key] = int(base > installed_capacity)
        output_nominal[period] = base
        output_coefficients = dict(primary_coefficients)
        output_coefficients[saturation_key] = installed_capacity - base

        threshold_high = dict(primary_coefficients)
        threshold_high[saturation_key] = -(
            maximum[period] - installed_capacity
        )
        auxiliary_constraints.append(
            LinearConstraint(
                name=f"ship.qc_threshold_high[{period}]",
                coefficients=threshold_high,
                sense="le",
                rhs=installed_capacity - base,
            )
        )
        threshold_low = {key: -value for key, value in primary_coefficients.items()}
        threshold_low[saturation_key] = installed_capacity - minimum[period]
        auxiliary_constraints.append(
            LinearConstraint(
                name=f"ship.qc_threshold_low[{period}]",
                coefficients=threshold_low,
                sense="le",
                rhs=base - minimum[period],
            )
        )

        for primary_key, coefficient in primary_coefficients.items():
            product_key = f"ship.qc_product[{period},{primary_key}]"
            auxiliary_selectors.append(
                SelectorSpec(product_key, "ship_delay", "binary", "auxiliary")
            )
            nominal_auxiliary[product_key] = 0
            output_coefficients[product_key] = -coefficient
            auxiliary_constraints.extend(
                [
                    LinearConstraint(
                        name=f"{product_key}.le_s",
                        coefficients={product_key: 1.0, saturation_key: -1.0},
                        sense="le",
                        rhs=0.0,
                    ),
                    LinearConstraint(
                        name=f"{product_key}.le_q",
                        coefficients={product_key: 1.0, primary_key: -1.0},
                        sense="le",
                        rhs=0.0,
                    ),
                    LinearConstraint(
                        name=f"{product_key}.ge_sq",
                        coefficients={
                            product_key: 1.0,
                            saturation_key: -1.0,
                            primary_key: -1.0,
                        },
                        sense="ge",
                        rhs=-1.0,
                    ),
                ]
            )
        coefficient_rows.append(output_coefficients)

    return QuayCraneEncoding(
        selectors=auxiliary_selectors,
        constraints=auxiliary_constraints,
        nominal=output_nominal,
        coefficient_rows=coefficient_rows,
        nominal_auxiliary=nominal_auxiliary,
        minimum=minimum,
        maximum=maximum,
        classifications=tuple(classifications),
        statuses=statuses,
    )

