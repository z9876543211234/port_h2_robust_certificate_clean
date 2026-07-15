from __future__ import annotations

from dataclasses import replace

import pytest

from port_h2_contracts.horizon import HorizonProfile
from port_h2_contracts.provenance import ProvenanceRecord
from port_h2_uncertainty_builders.combine import combine_bundles
from port_h2_uncertainty_builders.ship_delay.builder import build_ship_delay_bundle
from port_h2_uncertainty_builders.ship_delay.schema import ShipDelaySource
from port_h2_uncertainty_builders.wind.builder import build_wind_bundle
from port_h2_uncertainty_builders.wind.schema import WindUncertaintySource


@pytest.fixture
def toy_profile():
    return HorizonProfile.toy("toy_4", 4, 6.0)


@pytest.fixture
def toy_joint_bundle(tmp_path, toy_profile):
    fixed = ProvenanceRecord(
        source_type="fixed_file",
        source_reference="toy",
        random_seed=None,
        generation_script_sha256=None,
        raw_file_sha256="e" * 64,
        canonical_payload_sha256="f" * 64,
    )
    wind = build_wind_bundle(
        WindUncertaintySource(
            profile=toy_profile,
            nominal_power_kw=(1000.0, 1200.0, 800.0, 900.0),
            deviation_down_kw=(100.0, 100.0, 100.0, 100.0),
            deviation_up_kw=(100.0, 100.0, 100.0, 100.0),
            budget=1,
            provenance=fixed,
        ),
        tmp_path / "wind",
    )
    ship = build_ship_delay_bundle(
        ShipDelaySource(
            profile=toy_profile,
            previous_day_arrival_count=(0, 0, 0, 1),
            current_day_arrival_count=(1, 0, 1, 0),
            next_day_arrival_count=(0, 0, 0, 0),
            dwell_steps=2,
            max_delay_steps=1,
            delayed_ship_budget=1,
            total_delay_step_budget=None,
            shore_power_per_ship_kw=100.0,
            quay_cranes_per_ship=1.0,
            installed_quay_cranes=2.0,
            quay_crane_power_kw=20.0,
            quay_crane_task_rate_per_hour=2.0,
            provenance=fixed,
        ),
        tmp_path / "ship",
    )
    return combine_bundles(wind, ship, tmp_path / "joint")


def _make_toy_case(toy_profile):
    from port_h2_certificate import schema

    return schema.CaseData(
        case_name="C1_ProposedRobustMain",
        profile=toy_profile,
        grid=schema.GridParameters(
            buy_capacity_kw=5000.0,
            sell_capacity_kw=4000.0,
            buy_price_per_kwh=(0.6, 0.6, 0.8, 0.8),
            sell_price_per_kwh=(0.3, 0.3, 0.4, 0.4),
        ),
        base_load_kw=(300.0, 300.0, 300.0, 300.0),
        wind_to_hydrogen_ratio=(0.2, 0.2, 0.2, 0.2),
        hydrogen=schema.HydrogenParameters(
            transport_efficiency=0.95,
            conversion_kg_per_kwh=0.0180018,
            landing_delay_steps=1,
            historical_landing_kg=(2.0,),
            initial_kg=50.0,
            capacity_kg=200.0,
            terminal_min_kg=20.0,
        ),
        lohc=schema.LohcParameters(
            initial_kg=30.0,
            capacity_kg=120.0,
            terminal_min_kg=10.0,
            reactor_min_kw=0.0,
            reactor_max_kw=100.0,
            day_ahead_ramp_kw=80.0,
            real_time_ramp_kw=90.0,
            adjustment_limit_kw=50.0,
            energy_kwh_per_kg=2.5,
            hydrogen_yield=0.98,
            fixed_pipeline_rate_kg_per_hour=0.0,
        ),
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
            lohc_transport_rate_kg_per_vehicle_hour=10.0,
            container_rate_per_vehicle_hour=3.0,
            lohc_turn_efficiency=0.8,
            container_turn_efficiency=0.9,
            charge_adjustment_limit_kw=200.0,
            charge_count_ramp=3.0,
        ),
        logistics=schema.LogisticsParameters(
            initial_backlog=0.0,
            backlog_capacity=500.0,
        ),
        cost=schema.CostParameters(
            grid_deviation_per_kwh=0.2,
            spill_day_ahead_per_kwh=0.1,
            spill_real_time_per_kwh=0.1,
            spill_deviation_per_kwh=0.05,
            lohc_ramp_per_kw=0.01,
            lohc_adjustment_per_kwh=0.1,
            lohc_operation_per_kg=0.0,
            charge_adjustment_per_kwh=0.02,
            backlog_delay_per_task_hour=5.0,
            terminal_backlog_per_task=5.0,
            lohc_export_revenue_per_kg=20.0,
        ),
        outbound_mode="agv_virtual_pipeline",
        container_work_capacity=None,
        lohc_work_capacity=None,
        cost_scale=1000.0,
    )


@pytest.fixture
def toy_case(toy_profile):
    return lambda: _make_toy_case(toy_profile)


@pytest.fixture
def toy_c2_case(toy_case):
    def factory():
        base = toy_case()
        return replace(
            base,
            case_name="C2_NoHydrogenRobust",
            wind_to_hydrogen_ratio=(0.0,) * base.profile.periods,
            hydrogen=None,
            lohc=None,
            outbound_mode="disabled",
            cost=replace(base.cost, lohc_export_revenue_per_kg=0.0),
        )

    return factory
