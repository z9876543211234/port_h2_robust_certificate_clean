"""Post-build validation for a wind Bundle and its source."""

from __future__ import annotations

import numpy as np


def validate_wind_bundle(source, bundle) -> None:
    source.validate()
    bundle.validate()
    if bundle.profile_name != source.profile.profile_name:
        raise ValueError("wind source and Bundle profile differ")
    nominal = bundle.evaluate(bundle.nominal_selector)["wind.available_power_kw"]
    if not np.allclose(nominal, source.nominal_power_kw, atol=1e-10, rtol=0.0):
        raise ValueError("wind nominal realization differs from source")
