# Current nominal profiles and pointwise uncertainty envelopes

The figures are regenerated from the current 96-period source JSON files, not
from historical result plots.

## Ship arrivals

- Panel (a) shows the frozen integer NHPP nominal counts for the previous,
  current, and next day. Their realized totals are 15, 15, and 9 ships.
- Panel (b) is an exact pointwise envelope of `ship.arrival_count[t]` obtained by
  optimizing each coordinate over an integer ShipDelayBundle.
- The formal source still leaves `max_delay_steps` and `delayed_ship_budget`
  unresolved. Therefore the plotted ship envelope is explicitly provisional and
  uses the tested validation values `max_delay_steps=4` (1 h),
  `delayed_ship_budget=2`, and `total_delay_step_budget=null`.
- The lower envelope is zero at every period because every positive nominal
  period contains at most two ships, so the validation budget can delay all ships
  originally assigned to that period. The pointwise upper envelope reaches four
  ships per 15-minute period.

## Wind power

- The wind plot uses the current formal nominal profile and two-sided deviations.
- The exact pointwise envelope is computed from the WindBundle with
  `Gamma_w=12`. The nominal profile spans 0.1760--5.8608 MW; the pointwise band
  spans 0.1408--7.0330 MW across the day.

Pointwise envelopes show the exact minimum and maximum of each time coordinate.
They are not a single jointly feasible trajectory: global budgets generally
prevent all coordinate-wise extrema from occurring simultaneously.

Regenerate with:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  figures/uncertainty_envelopes/generate_uncertainty_envelopes.py \
  --ship-max-delay-steps 4 \
  --ship-delayed-budget 2
```

All 384 coordinate-bound solves (lower and upper bounds for two 96-period
outputs) finished with Gurobi status `OPTIMAL`. Exact source, generator, and
Bundle hashes are recorded in `envelope_metadata.json`.
