# C1--C4 confirmed-parameter results (four-key partition)

## Common model and input scope

- AGV fleet / chargers: `60 / 18`
- AGV charging / LOHC adjustment limits: `900 / 1,200 kW`
- Charging / LOHC / grid deviation costs: `0.3 / 0.3 / 0.4 yuan/kWh`
- Charging-count ramp: `4 vehicles/period`
- Spill costs (DA / RT / deviation): `0.1 / 0.1 / 0.125 yuan/kWh`
- Wind / integer-ship budgets: `Gamma_W=12`, `Gamma_N=2 ships`
- Maximum ship delay: `4` periods; total delay-step budget: none
- Joint uncertainty bundle SHA-256:
  `c86a258d698635b6279717fd8f4c4f19491024e01cbe1af0ed695b2a3c8714a5`
- Cost-adversary partition keys: `wind.down[1]`, `wind.down[9]`,
  `ship.delay[51,0,2]`, and `ship.delay[48,0,4]`
- No solver time limit; every formal master/adversary solve requires
  `GRB.OPTIMAL`; every partition leaf is `OPTIMAL` or strictly infeasible.

## Results

| Case | Meaning | LB | UB / stress total | Relative gap | Status |
|---|---|---:|---:|---:|---|
| C1 | Proposed robust main | 102.436207 | 102.516576 | 0.0784% | engineering optimal |
| C2 | No-hydrogen robust | 187.719184 | 187.812934 | 0.0499% | engineering optimal |
| C3 | 32/28 work-capacity robust | 119.855973 | 120.673738 | 0.6777% | engineering optimal |
| C4 | Deterministic nominal, then stress | 104.720796 | 104.720796 | 0 | stress certified |

C4's nominal objective is `92.644367`; `104.720796` is the worst-case stress
total for that fixed nominal first-stage schedule. It is not a robustly optimized
objective and should be reported separately from C1--C3.

## Certificate and constraint audit

| Case | Original-unit max residual | Delayed ships | Max charge adjustment | Max LOHC adjustment | Key case check |
|---|---:|---:|---:|---:|---|
| C1 | 1.0242e-11 | 2 | 0.9 MW | 1.2 MW | proposed robust schedule |
| C2 | 5.3291e-12 | 2 | 0.9 MW | n/a | hydrogen chain disabled |
| C3 | 8.1601e-12 | 2 | 0.9 MW | 1.2 MW | container/LOHC AGVs hit 32/28 |
| C4 | 8.1601e-12 | 2 | 0.6400 MW | 1.2 MW | nominal schedule stress replay |

All cases satisfy the 18-charger limit and four-vehicle inter-period charging
ramp (up to floating-point tolerance). All accepted certificates record
`accepted_limited_solve=false` and `time_limit_used_for_certificate=false`.
The complete project regression suite passed: `104 passed in 18.45s`.

## Why the fourth partition key is required

The first C2 three-key attempt was rejected at iteration 2 because it produced
`LB=187.034943` and `UB=186.709680`. Direct primal replay showed that the master
scenario pool already contained a feasible recourse cost of `10.545509`, while
the three-key adversary incorrectly reported a full-set maximum of `10.220246`.
Fixing that complete selector restored primal--dual equality to `5.33e-15`.
Adding `ship.delay[48,0,4]` split the unstable ship-delay leaf and recovered a
strict leaf optimum of `11.389570` with zero MIP gap. This is a mutually
exclusive partition refinement only: it changes neither model semantics nor
the uncertainty set.

The rejected C2 checkpoint remains at `C2_NoHydrogenRobust`; the authoritative
C2 output is `C2_NoHydrogenRobust_refined_partition`. Likewise, the
authoritative C1 output is `C1_ProposedRobustMain_refined_partition`; the earlier
three-key C1 result is superseded.
