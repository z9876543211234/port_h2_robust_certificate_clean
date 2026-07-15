from __future__ import annotations

from itertools import chain

import pytest

from port_h2_certificate import schema
from port_h2_certificate.compilers.compile_master import compile_master
from port_h2_certificate.first_stage_ir import build_first_stage_ir
from port_h2_certificate.recourse_ir import build_recourse_ir
from port_h2_certificate.solver.ccg import solve_robust_ccg
from port_h2_certificate.two_stage_ir import TwoStageIR
from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.combine import combine_bundles
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource
from port_h2_uncertainty_builders.wind.builder import build_wind_bundle
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


PROVENANCE = ProvenanceRecord(
    source_type="fixed_file",
    source_reference="profile-integration-test",
    random_seed=None,
    generation_script_sha256=None,
    raw_file_sha256="a" * 64,
    canonical_payload_sha256="b" * 64,
)


def _joint_bundle(profile, root, *, wind_budget):
    periods = profile.periods
    wind = build_wind_bundle(
        WindUncertaintySource(
            profile=profile,
            nominal_power_kw=(1000.0,) * periods,
            deviation_down_kw=(100.0,) * periods,
            deviation_up_kw=(100.0,) * periods,
            budget=wind_budget,
            provenance=PROVENANCE,
        ),
        root / "wind",
    )
    previous = [0] * periods
    previous[-1] = 1
    current = [0] * periods
    current[1] = 1
    ship = build_ship_delay_bundle(
        ShipDelaySource(
            profile=profile,
            previous_day_arrival_count=tuple(previous),
            current_day_arrival_count=tuple(current),
            next_day_arrival_count=(0,) * periods,
            dwell_steps=2,
            max_delay_steps=1,
            delayed_ship_budget=1,
            total_delay_step_budget=None,
            shore_power_per_ship_kw=100.0,
            quay_cranes_per_ship=1.0,
            installed_quay_cranes=2.0,
            quay_crane_power_kw=20.0,
            quay_crane_task_rate_per_hour=2.0,
            provenance=PROVENANCE,
        ),
        root / "ship",
    )
    return combine_bundles(wind, ship, root / "joint")


def _c2_case(profile):
    periods = profile.periods
    return schema.CaseData(
        case_name="C2_NoHydrogenRobust",
        profile=profile,
        grid=schema.GridParameters(
            buy_capacity_kw=5000.0,
            sell_capacity_kw=4000.0,
            buy_price_per_kwh=(0.6,) * periods,
            sell_price_per_kwh=(0.3,) * periods,
        ),
        base_load_kw=(300.0,) * periods,
        wind_to_hydrogen_ratio=(0.0,) * periods,
        hydrogen=None,
        lohc=None,
        agv=schema.AgvParameters(
            fleet_size=10.0,
            charger_count=5.0,
            charge_power_per_vehicle_kw=50.0,
            run_power_per_vehicle_kw=8.0,
            battery_capacity_per_vehicle_kwh=200.0,
            charge_efficiency=0.9,
            initial_soc=0.7,
            min_soc=0.2,
            max_soc=0.9,
            terminal_min_soc=0.4,
            lohc_transport_rate_kg_per_vehicle_hour=0.0,
            container_rate_per_vehicle_hour=3.0,
            lohc_turn_efficiency=0.8,
            container_turn_efficiency=0.9,
            charge_adjustment_limit_kw=200.0,
            charge_count_ramp=3.0,
        ),
        logistics=schema.LogisticsParameters(
            initial_backlog=0.0, backlog_capacity=500.0
        ),
        cost=schema.CostParameters(
            grid_deviation_per_kwh=0.2,
            spill_day_ahead_per_kwh=0.1,
            spill_real_time_per_kwh=0.1,
            spill_deviation_per_kwh=0.05,
            lohc_ramp_per_kw=0.0,
            lohc_adjustment_per_kwh=0.0,
            lohc_operation_per_kg=0.0,
            charge_adjustment_per_kwh=0.02,
            backlog_delay_per_task_hour=5.0,
            terminal_backlog_per_task=5.0,
            lohc_export_revenue_per_kg=0.0,
        ),
        outbound_mode="disabled",
        container_work_capacity=None,
        lohc_work_capacity=None,
        cost_scale=1000.0,
    )


def _structured_exact_realizations(bundle, wind_budget):
    wind_keys = [key for key in bundle.primary_selector_keys if key.startswith("wind.")]
    ship_keys = [key for key in bundle.primary_selector_keys if key.startswith("ship.")]
    assert len(ship_keys) == 1
    wind_choices = [None] if wind_budget == 0 else [None, *wind_keys]
    unique = {}
    for wind_key in wind_choices:
        for ship_value in (0, 1):
            primary = {key: 0 for key in bundle.primary_selector_keys}
            if wind_key is not None:
                primary[wind_key] = 1
            primary[ship_keys[0]] = ship_value
            full = bundle.complete(primary)
            unique.setdefault(bundle.realization_hash(full), bundle.evaluate(full))
    return tuple(unique.values())


@pytest.mark.parametrize(
    ("periods", "dt_hours"),
    ((8, 3.0), (12, 2.0)),
)
def test_complete_exact_ccg_for_8_and_12_periods(
    tmp_path, periods, dt_hours
) -> None:
    profile = HorizonProfile.toy(f"toy_{periods}", periods, dt_hours)
    bundle = _joint_bundle(profile, tmp_path / str(periods), wind_budget=1)
    case = _c2_case(profile)
    two_stage = TwoStageIR(
        build_first_stage_ir(case, bundle), build_recourse_ir(case, bundle)
    )
    realizations = _structured_exact_realizations(bundle, 1)
    extensive = compile_master(two_stage, realizations).solve()
    ccg = solve_robust_ccg(
        two_stage, bundle, outer_gap_tolerance=1e-9, max_iterations=100
    )
    assert ccg.engineering_optimal
    assert ccg.lower_bound == pytest.approx(extensive.objective, abs=1e-6)
    assert ccg.upper_bound == pytest.approx(extensive.objective, abs=1e-6)


@pytest.mark.parametrize(
    ("profile_name", "periods", "dt_hours"),
    (("hourly_24", 24, 1.0), ("quarter_hour_96", 96, 0.25)),
)
def test_formal_profiles_use_identical_builders_and_model_code(
    tmp_path, profile_name, periods, dt_hours
) -> None:
    profile = HorizonProfile.formal(profile_name, periods, dt_hours)
    bundle = _joint_bundle(profile, tmp_path / profile_name, wind_budget=0)
    case = _c2_case(profile)
    two_stage = TwoStageIR(
        build_first_stage_ir(case, bundle), build_recourse_ir(case, bundle)
    )
    realizations = _structured_exact_realizations(bundle, 0)
    extensive = compile_master(two_stage, realizations).solve()
    ccg = solve_robust_ccg(
        two_stage, bundle, outer_gap_tolerance=1e-9, max_iterations=20
    )
    assert ccg.engineering_optimal
    assert ccg.upper_bound == pytest.approx(extensive.objective, abs=1e-6)
