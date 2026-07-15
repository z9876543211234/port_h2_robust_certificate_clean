from __future__ import annotations

import importlib
import importlib.util

import pytest


def _horizon_module():
    spec = importlib.util.find_spec("port_h2_contracts.horizon")
    assert spec is not None, "horizon contract is not implemented"
    return importlib.import_module("port_h2_contracts.horizon")


@pytest.mark.parametrize(
    ("profile_name", "periods", "dt_hours"),
    [("hourly_24", 24, 1.0), ("quarter_hour_96", 96, 0.25)],
)
def test_formal_horizon_accepts_only_specified_profiles(
    profile_name: str, periods: int, dt_hours: float
) -> None:
    module = _horizon_module()
    horizon = module.HorizonProfile.formal(profile_name, periods, dt_hours)
    assert tuple(horizon.operating_periods) == tuple(range(periods))
    assert tuple(horizon.state_periods) == tuple(range(periods + 1))


@pytest.mark.parametrize("periods", [4, 8, 12])
def test_toy_horizon_is_explicitly_non_formal(periods: int) -> None:
    module = _horizon_module()
    horizon = module.HorizonProfile.toy(f"toy_{periods}", periods, 24.0 / periods)
    assert horizon.is_formal is False
    assert horizon.periods == periods


def test_formal_horizon_rejects_test_length() -> None:
    module = _horizon_module()
    with pytest.raises(ValueError, match="formal horizon"):
        module.HorizonProfile.formal("toy_12", 12, 2.0)


@pytest.mark.parametrize("periods,dt_hours", [(24, 0.25), (96, 1.0), (0, 1.0)])
def test_horizon_rejects_inconsistent_period_and_step(
    periods: int, dt_hours: float
) -> None:
    module = _horizon_module()
    with pytest.raises(ValueError):
        module.HorizonProfile.formal("bad", periods, dt_hours)

