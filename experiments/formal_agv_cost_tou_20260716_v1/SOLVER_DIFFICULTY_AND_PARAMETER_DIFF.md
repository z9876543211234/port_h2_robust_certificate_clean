# C1 solving difficulty and parameter-difference audit

Date: 2026-07-16

## 1. Executive conclusion

The current C1 attempt did not fail a feasibility or primal/dual replay test.
It was explicitly interrupted while the first exact partitioned cost adversary
was still proving complete leaf coverage.  At interruption, no C&CG iteration
had completed, no checkpoint had been written, and no formal certificate had
been exported.  The fail-closed runtime error is therefore the correct result.

The previous C1 cases were fast under the same mathematical dimensions, so the
current delay cannot be attributed to model size alone.  The evidence supports
two direct changes to the search geometry:

1. the economic coefficients now create different and more time-dependent
   trade-offs; and
2. the partition oracle now enforces a fixed known-scenario lower bound and
   primal/dual consistency in each compatible leaf.

The statements about which coefficient causes the largest slowdown remain
inferences until a controlled one-factor and leaf-level diagnostic is run.

## 2. Model semantics retained in this branch

The branch starts from the pre-exact-transport formulation:

\[
B_{t+1}\ge B_t+q_t-\kappa^{ctn}n_t^{ctn},
\]

and retains continuous equivalent working-vehicle allocations.  It does not
replace the inequality with an equality and does not eliminate the AGV work
variables.  Instead, the recourse objective adds

\[
C^{AGV}\Delta t\sum_t(n_t^{ctn}+n_t^{LOHC}),
\qquad C^{AGV}=60\;\text{CNY/(vehicle h)}.
\]

Only working allocations are charged.  Charging and idle vehicles have no AGV
operating-cost coefficient.  All working vehicles still consume 18 kW through
the SOC transition.  This is the user-confirmed attempt to make empty running
more expensive than spilling wind without changing the original capacity
constraint.

The exact partition oracle separately fixes every known compatible selector,
checks its dual objective against the primal replay, adds that value as a lower
floor to the leaf maximization, supplies the full solution as a MIP start, and
accepts the leaf only if the final optimum remains consistent with the floor.

## 3. Confirmed parameter differences

| Parameter | Previous formal C1 | Intermediate AGV-cost C1 | Current frozen C1 |
|---|---:|---:|---:|
| AGV operating cost, CNY/(working vehicle h) | 0 | 6.5 | **60** |
| Port base load | 40 MW | 40 MW | **20 MW** |
| Grid purchase price | 0.6 flat | 0.6 flat | **0.41/0.63/0.75 TOU** |
| Grid sale price | 0.3 flat | 0.3 flat | **0.205/0.315/0.375 TOU** |
| Sale-price rule | fixed | fixed | **50% of purchase price** |
| Day-ahead spill cost, CNY/kWh | 0.1 | 0.1 | **0.1** |
| Real-time spill cost, CNY/kWh | 0.1 | 0.1 | **0.1** |
| Spill-deviation cost, CNY/kWh | 0.125 | 0.125 | **0.1** |
| Grid-deviation cost, CNY/kWh | 0.4 | 0.4 | **0.3** |
| LOHC export term, CNY/kg-H2-eq | 46.8 revenue | 46.8 revenue | **20 subsidy** |
| LOHC non-electric operation cost, CNY/kg-H2-eq | 0.1 | 0.1 | **0, explicitly confirmed** |
| Known-scenario objective floor | absent in original snapshot | added after mismatch diagnosis | **enabled** |

The purchase tariff is 0.41 CNY/kWh at 00:00--07:00, 08:00--09:00,
11:00--12:00, 20:00--21:00, and 22:00--24:00; 0.63 at 07:00--08:00,
12:00--17:00, and 21:00--22:00; and 0.75 at 09:00--11:00 and
17:00--20:00.  Each hourly value is repeated four times in the 96-period model.

Unchanged confirmed values include 57 AGVs, 18 chargers, 350 kWh/vehicle,
150 kW/charger, 18 kW/working vehicle, 900 kW charging adjustment, charging
ramp 4 vehicles/period, H2 initial/capacity 5/25 t-H2, LOHC initial/capacity
2/12 TEU, 12.5 CNY/(TEU h) backlog cost, 300 CNY/TEU terminal
backlog cost, 100/300 MW grid buy/sell capacities, 3 MW/ship shore power,
0.5 MW/quay crane, 32 quay cranes, 12 vessels/day, 200 TEU/vessel, and the
3:7 direct-wind/hydrogen split.

The ship uncertainty budget remains at most two delayed vessels, each delayed
by at most four 15-minute periods (one hour), with no separate total-delay-step
budget.  The wind deviation budget remains 12.

## 4. Previous and current solve evidence

The previous zero-AGV-cost formal C1 and the intermediate 6.5-CNY case both
completed in two C&CG iterations using the same 240-selector, 114-constraint
joint Bundle and the same five partition keys.  The 6.5-CNY certificate passed
all partition, replay, residual, and no-time-limit gates.

Current model dimensions are unchanged:

| Component | Size |
|---|---:|
| First-stage variables | 575 |
| First-stage constraints | 572 |
| Recourse variables per scenario | 1,348 |
| Recourse constraints per scenario | 2,595 |
| Joint uncertainty selectors | 240 binary |
| Joint uncertainty constraints | 114 |
| Fixed partition keys | 5 |
| Mutually exclusive partition leaves | 32 |

The current formal attempt started at approximately 22:09:53 +08:00 and was
interrupted around 22:48:30 +08:00, approximately 2,317 seconds later.  Read-only
process observations showed roughly 1007%--1125% CPU and 1.5% memory, so the
solver was actively using multiple cores rather than sleeping or silently
exiting.  The first partitioned adversary call had not returned, hence the C&CG
iteration callback had not written a checkpoint.

The interruption produced:

```text
RuntimeError: formal partitioned cost adversary did not provide complete
OPTIMAL/INFEASIBLE leaf coverage
```

This is not a solver claim of infeasibility and not a limited certificate.  It
means the user-requested interruption prevented the full exact leaf cover from
being proved.

## 5. Likely difficulty mechanisms

### 5.1 Evidence-backed mechanisms

- Five inherited partition keys fix only five of 240 binary selectors.  Each of
  the 32 leaves can still contain up to 235 unfixed selectors.
- The indicator adversary contains products between uncertain selectors and
  free/nonnegative LP-dual expressions.  Fixed-scenario primal and dual LPs are
  fast; proving the mixed-integer maximum over all selectors is the hard step.
- Every leaf must reach exact `OPTIMAL` or `INFEASIBLE`; `MIPGap` and
  `MIPGapAbs` are both zero.
- The current code already uses `MIPFocus=3`, `NumericFocus=2`, tight
  feasibility/integrality tolerances, and automatic multithreading.
- The known-scenario floor prevents a leaf from returning a value below a
  scenario that is explicitly feasible in that leaf.  This closes the prior
  adversary-underestimation failure mode but can make a narrow high-objective
  region harder to prove.

### 5.2 Plausible but not yet isolated parameter effects

- Reducing the base load from 40 to 20 MW increases excess-wind pressure under
  an 800 MW peak and a 300 MW export cap.
- Lowering the LOHC export term from 46.8 to 20 while raising working-AGV cost
  from 0/6.5 to 60 changes the active bases associated with container service,
  LOHC outbound, backlog, and storage.
- Time-of-use prices make the placement of wind deviations and ship delays
  period-specific, while flat prices previously produced a more monotone
  economic direction.
- Setting all three spill terms to 0.1 and grid deviation to 0.3 changes the
  marginal ordering among spilling, export, charging, and hydrogen conversion.

No single item above should be reported as the proved root cause until the same
leaf is solved under one-factor parameter restorations.

## 6. Exact acceleration path

The following changes preserve the model feasible region, uncertainty set, and
formal acceptance policy:

1. Add atomic leaf-level checkpoints, elapsed time, Gurobi logs, and resume so
   completed `OPTIMAL/INFEASIBLE` leaves are never recomputed.
2. Select partition keys from the current parameter set rather than reusing the
   old five keys.  Compare complete 4--7-key partitions on representative
   adversaries; every candidate must still cover all mutually exclusive leaves.
3. Export the first slow leaf and use controlled Gurobi parameter tuning on the
   diagnostic copy.  A diagnostic may use a time budget, but the selected
   formal run must remove all time limits and still reach zero gap and OPTIMAL.
4. Supply complete known-selector and dual MIP starts to compatible leaves.
5. As a larger mathematical-equivalence project, derive and test an exact
   support-function reformulation of the cardinality-budget wind selectors to
   reduce indicator products.  This requires an enumerable toy proof and
   fixed-scenario/worst-scenario equivalence tests before formal use.

Changing costs, removing scenarios, reducing uncertainty budgets, accepting a
positive MIP gap, or imposing a formal time limit are not classified as solver
acceleration and must not be used to publish a certificate.

## 7. Remaining work

- Resume C1 only after choosing whether to keep the inherited partition keys or
  implement the isolated exact acceleration path.
- Run the agreed C3 integer group-capacity scan over 50%--85% container share in
  5-point steps; write C3 input only after the predeclared rule selects a split.
- Complete formal C3, C4, and C2 in the requested order after C1.
- Do not reuse the interrupted directory as a formal result directory; use a
  new versioned output or a verified leaf-resume implementation.
