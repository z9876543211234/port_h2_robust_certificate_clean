from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util


def _builder(module_name: str, function_name: str):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"{module_name} is not implemented"
    return getattr(importlib.import_module(module_name), function_name)


def build_first_stage_ir(*args, **kwargs):
    return _builder(
        "port_h2_certificate.first_stage_ir", "build_first_stage_ir"
    )(*args, **kwargs)


def build_recourse_ir(*args, **kwargs):
    return _builder("port_h2_certificate.recourse_ir", "build_recourse_ir")(
        *args, **kwargs
    )


def _families(keys):
    return {key[0] for key in keys}


def test_first_stage_contains_declared_primary_variables_and_derived_charge_power(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    ir = build_first_stage_ir(toy_case, toy_joint_bundle)
    families = _families(ir.variable_keys)
    assert {
        "grid_buy_da_mw",
        "grid_sell_da_mw",
        "spill_da_mw",
        "lohc_power_da_mw",
        "agv_charge_count_da",
    } <= families
    assert "agv_charge_power_da_mw" not in families


def test_recourse_uses_only_reduced_variables_and_state_index_ranges(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    ir = build_recourse_ir(toy_case, toy_joint_bundle)
    families = _families(ir.variable_keys)
    forbidden = {
        "grid_power_rt",
        "agv_charge_power_rt",
        "h2_to_lohc",
        "lohc_production",
        "agv_idle",
        "task_done",
        "positive_deviation",
        "negative_deviation",
    }
    assert not families & forbidden
    assert {key[1] for key in ir.variable_keys if key[0] == "soc"} == set(range(5))
    assert {key[1] for key in ir.variable_keys if key[0] == "backlog"} == set(range(5))
    assert "spill_rt_mw" in families
    assert "spill_deviation_mw" in families


def test_positive_agv_operation_cost_prices_each_running_allocation(
    toy_case, toy_joint_bundle
) -> None:
    case = toy_case()
    operation_cost = 4.6
    case = replace(
        case,
        cost=replace(
            case.cost,
            agv_operation_per_vehicle_hour=operation_cost,
        ),
    )

    ir = build_recourse_ir(case, toy_joint_bundle)
    variables = {spec.key: spec for spec in ir.variables}
    expected = operation_cost * case.profile.dt_hours / case.cost_scale

    for period in range(case.profile.periods):
        assert variables[("agv_container_count", period)].objective_coefficient == expected
        assert variables[("agv_lohc_count", period)].objective_coefficient == expected
        assert variables[("agv_charge_count", period)].objective_coefficient == 0.0


def test_ir_uncertainty_rows_reference_only_allowed_bundle_outputs(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    ir = build_recourse_ir(toy_case, toy_joint_bundle)
    allowed_prefixes = {
        "wind.available_power_kw",
        "ship.shore_power_kw",
        "ship.quay_crane_power_kw",
        "ship.task_release",
    }
    used = {
        key.rsplit("[", 1)[0]
        for row in ir.constraints
        for key in row.uncertain_output_coefficients
    }
    assert used <= allowed_prefixes


def test_c2_ir_has_no_hydrogen_or_lohc_variables_or_rows(
    toy_c2_case, toy_joint_bundle
) -> None:
    toy_c2_case = toy_c2_case()
    first_stage = build_first_stage_ir(toy_c2_case, toy_joint_bundle)
    recourse = build_recourse_ir(toy_c2_case, toy_joint_bundle)
    assert not any("hydrogen" in key[0] or "lohc" in key[0] for key in first_stage.variable_keys)
    assert not any("hydrogen" in key[0] or "lohc" in key[0] for key in recourse.variable_keys)
    assert not any(row.component in {"hydrogen", "lohc"} for row in recourse.constraints)


def test_c3_differs_from_c1_only_by_two_work_capacity_rows(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    c1 = build_recourse_ir(toy_case, toy_joint_bundle)
    c3_case = replace(
        toy_case,
        case_name="C3_WorkCapacityCapsRobust",
        container_work_capacity=6.0,
        lohc_work_capacity=4.0,
    )
    c3 = build_recourse_ir(c3_case, toy_joint_bundle)
    assert c3.variables == c1.variables
    c1_names = {row.name for row in c1.constraints}
    added = [row for row in c3.constraints if row.name not in c1_names]
    assert {row.name for row in added} == {
        "agv.container_work_capacity[0]",
        "agv.container_work_capacity[1]",
        "agv.container_work_capacity[2]",
        "agv.container_work_capacity[3]",
        "agv.lohc_work_capacity[0]",
        "agv.lohc_work_capacity[1]",
        "agv.lohc_work_capacity[2]",
        "agv.lohc_work_capacity[3]",
    }


def test_c4_uses_identical_physical_ir_to_c1(toy_case, toy_joint_bundle) -> None:
    toy_case = toy_case()
    c1 = build_recourse_ir(toy_case, toy_joint_bundle)
    c4 = build_recourse_ir(
        replace(toy_case, case_name="C4_DeterministicMain"), toy_joint_bundle
    )
    assert c4 == c1


def test_backlog_has_penalized_terminal_state_without_hard_zero(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    ir = build_recourse_ir(toy_case, toy_joint_bundle)
    terminal = next(variable for variable in ir.variables if variable.key == ("backlog", 4))
    assert terminal.objective_coefficient > 0.0
    assert not any("terminal_zero" in row.name for row in ir.constraints)


def test_lohc_export_revenue_is_negative_objective_coefficient(
    toy_case, toy_joint_bundle
) -> None:
    toy_case = toy_case()
    ir = build_recourse_ir(toy_case, toy_joint_bundle)
    outbound = [
        variable.objective_coefficient
        for variable in ir.variables
        if variable.key[0] == "lohc_outbound_normalized"
    ]
    assert outbound and all(value < 0.0 for value in outbound)
