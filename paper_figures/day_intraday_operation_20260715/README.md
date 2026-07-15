# C1 day-ahead / intra-day operation figure

This isolated package redraws the requested three-panel operation figure from
the authoritative corrected four-key C1 result:

`experiments/c1_confirmed_main_budget_20260714/C1_ProposedRobustMain_refined_partition`

The package does not re-solve or modify the optimization model.

## Data mapping

- Panel (a): hourly mean day-ahead and worst-case real-time net grid exchange;
  bars show the real-time minus day-ahead adjustment.
- Panel (b): hourly mean day-ahead/real-time AGV charging counts and LOHC
  reactor powers. These are the two DA/RT variable pairs present in the formal
  model.
- Panel (c): hourly sum of LOHC outbound mass and H2/LOHC inventories sampled
  at hourly state boundaries.

The current formal model has no day-ahead container-transport or LOHC-transport
AGV allocation variables. Consequently, the corresponding curves in the old
Case D figure are not reproduced or inferred.

## Rebuild

From the project root:

```bash
python3 paper_figures/day_intraday_operation_20260715/scripts/generate_day_intraday_operation.py
```

Scripts and generated figures are stored in separate `scripts/` and `figures/`
directories. Exact hourly plotting tables are stored in `data/`; hashes and
certificate fields are recorded in `source_manifest.json`.

## LOHC inventory and AGV allocation figure

`scripts/generate_lohc_inventory_agv_allocation.py` produces a second,
style-matched figure. Its first panel shows hourly LOHC production/outbound
quantities and the inventory state; its second panel shows the hourly mean AGV
allocation among container service, LOHC transport, charging, and idle states.
The corresponding traceability file is
`source_manifest_lohc_inventory_agv_allocation.json`.
