# Model Specification

## 1. Robust program

The implementation represents

\[
\min_x\left\{C^{DA}(x)+\max_{u\in\mathcal U}\min_y C^{RT}(x,y,u)\right\}.
\]

All operating variables use `t = 0, ..., T-1`; state variables use
`t = 0, ..., T`. Formal horizons are `(T, dt) = (24, 1 h)` and
`(96, 0.25 h)`. No closure tail or next-day physical operation is permitted.

## 2. Uncertainty boundary

Wind selectors and integer ship-delay selectors are constructed independently and
combined only by Cartesian product. The physical model may consume only:

- `wind.available_power_kw[t]`;
- `ship.arrival_count[t]`;
- `ship.in_port_count[t]`;
- `ship.shore_power_kw[t]`;
- `ship.quay_crane_power_kw[t]`;
- `ship.task_release[t]`.

It may not inspect uncertainty budgets, delay variables, saturation auxiliaries, or
builder code.

The three nominal ship sequences are frozen offline from independent-increment
NHPP Poisson draws for the previous, current, and next day.  The expected rate is
provenance-only and is not a physical-model input.  Each realized period count is
a nonnegative integer.  Only individual ships in the frozen current-day sequence
may receive a positive integer delay; previous/next-day counts are fixed boundary
data.  Early arrivals, continuous equivalent arrival pressure, modulo wrapping,
and optimization-time resampling are prohibited.

## 3. First stage

Continuous variables are day-ahead grid purchase/sale/spill,
`P_LOHC_DA[t]`, and equivalent charging-fleet count `N_ch_DA[t]`. Charging power
is the derived expression `p_ch_single * N_ch_DA[t]`. The day-ahead power balance
uses the nominal Bundle realization. Constraints include equipment bounds, LOHC
ramp limits, optional charging-count ramp limits, and every-prefix hydrogen
availability. No grid buy/sell exclusivity binary is introduced.

Day-ahead cost contains purchase, sale revenue, spill cost, LOHC ramp cost, and
LOHC non-electric operation cost on day-ahead production only.

## 4. Second stage

The production recourse is a continuous reduced LP with variables for real-time
spill, LOHC power/outbound, pooled AGV allocations, H2/LOHC/SOC/backlog states,
and absolute-deviation epigraphs. Grid exchange, charging power, H2 conversion,
LOHC production, idle fleet, and completed tasks are reconstructed expressions.

The model preserves:

- grid capacity and deviation from the day-ahead net schedule;
- real-time spill and day-ahead/real-time spill deviation;
- LOHC bounds, adjustment, ramping, inventories, delayed material use, and export
  revenue;
- pooled AGV capacity, charging adjustment/ramp, and SOC dynamics with `dt`;
- backlog upper/lower transition inequalities and no hard terminal clearing.

The objective contains grid, spill, LOHC, and charging deviations, backlog delay
penalties including the terminal node, and negative LOHC export revenue.

## 5. Cases

- `C1_ProposedRobustMain`: full H2/LOHC and AGV virtual pipeline.
- `C2_NoHydrogenRobust`: omits all H2/LOHC variables and rows structurally.
- `C3_WorkCapacityCapsRobust`: C1 plus container and LOHC work-capacity bounds
  only; charging remains pooled.
- `C4_DeterministicMain`: nominal first-stage optimization followed by full-set
  Phase-I and cost stress adversaries at fixed first stage.

AGV counts are continuous equivalent fleet capacities because both the fixed
recourse and finite-scenario master are required to be LPs.

## 6. Scaling and acceptance

Power is represented internally in MW, H2/LOHC/backlog states are normalized by
their capacities, SOC remains per-unit, and objective is divided by a fixed
`cost_scale`. Phase-I uses `sigma_i = 1` after this scaling and
`epsilon_f = 1e-8`. Original-unit and scaled residuals are both reported.
