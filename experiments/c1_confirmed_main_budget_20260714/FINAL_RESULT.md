# C1 corrected main-budget result (2026-07-14)

## Scope

This run replaces only the eight controls explicitly confirmed by the user. It
does not overwrite the earlier exploratory large-port result or the baseline
`data/` directory.

| Control | Confirmed value |
|---|---:|
| AGV fleet | 60 vehicles |
| Chargers | 18 |
| AGV charging adjustment limit | 900 kW |
| LOHC adjustment limit | 1,200 kW |
| AGV charging adjustment cost | 0.3 yuan/kWh |
| LOHC adjustment cost | 0.3 yuan/kWh |
| Grid deviation cost | 0.4 yuan/kWh |
| Charging-count ramp | 4 vehicles/period |

The retained uncertainty settings are wind budget `Gamma_W = 12`, integer ship
budget `Gamma_N = 2 ships`, maximum ship delay `4` periods, and no total
delay-step budget. Spill costs are `0.1/0.1/0.125 yuan/kWh` for day-ahead,
real-time, and deviation terms, respectively.

## Input identity

- Calibration: `large_port_12v_confirmed_main_budget`
- Case semantic SHA-256: `1a7f9bd02d921ac514b7e4d39ce71fb8b3f7127b755e320a9c9f50da80157fa2`
- Joint uncertainty bundle SHA-256: `c86a258d698635b6279717fd8f4c4f19491024e01cbe1af0ed695b2a3c8714a5`
- Three-day integer NHPP arrival totals: `15/15/9` ships
- Equivalent ship pressure: not used

## Strict C1 result

The no-time-limit four-key refined-partition run completed in two C&CG
iterations. The earlier three-key run is superseded because the C2 audit proved
that partition numerics could miss a known worse ship-delay selector.

| Certificate item | Result |
|---|---:|
| Status | `engineering_optimal` |
| Master status | `OPTIMAL` |
| Phase-I adversary status | `OPTIMAL` |
| Cost adversary status | `OPTIMAL` |
| Lower bound | 102.4362065216 |
| Global upper bound | 102.5165764480 |
| Absolute gap | 0.0803699264 |
| Relative gap | 0.0783970058% |
| Configured outer gap tolerance | 1% |
| Maximum scaled residual | 2.3093e-14 |
| Maximum original-unit residual | 1.0242e-11 |
| Accepted limited solve | false |
| Certificate time limit used | false |

With `cost_scale = 1000`, the reported objective bounds use the model's scaled
cost convention (approximately thousand-yuan units). The certified worst-case
upper-bound reconstruction is:

- Day-ahead cost: `258.1033486963`
- Recourse cost: `-155.5867722483`
- Total: `102.5165764480`

The nonzero deviation costs in that worst-case recourse are `0.5776344086` for
AGV charging, `0.72` for LOHC, `10.2982272595` for grid deviation,
`0.5392454955` for spill deviation, and `0.4313963964` for real-time spill.
Therefore none of the three confirmed adjustment/deviation penalties was
silently removed.

## Constraint audit

| Check | Observed | Limit | Verdict |
|---|---:|---:|---|
| Day-ahead charger use | 18 | 18 | pass |
| Real-time charger use | 18 | 18 | pass |
| Day-ahead charging-count jump | 4 | 4 | pass |
| Real-time charging-count jump | 4 | 4 | pass |
| AGV charging adjustment | 0.9 MW | 0.9 MW | pass |
| LOHC adjustment | 1.2 MW | 1.2 MW | pass |
| AGV allocation balance error | 1.4211e-14 vehicles | numerical zero | pass |
| Delayed ships in worst scenario | 2 | 2 | pass |

AGV recourse allocations remain continuous working-state equivalents, as
required by the current LP-recourse/dual-adversary formulation. The ship arrival
draw and ship delay selectors are discrete integers; no equivalent-pressure ship
representation is used.

## Regression evidence

- Independent case/bundle validation: `status=valid`, with both hashes matching
  the formal run
- Focused input/IR tests: `14 passed`
- Full project suite after C1--C4 completion: `104 passed in 18.45s`

The earlier result at
`experiments/c1_large_port_12v_20260713/final_selected_cdev_0p125` remains a
valid certificate only for its different exploratory input bundle. Its upper
bound `91.7545243692` must not be used as the corrected C1 result.

The intermediate three-key result at
`experiments/c1_confirmed_main_budget_20260714/C1_ProposedRobustMain` is also
superseded. Its upper bound `101.4229132122` is not used in the final C1--C4
comparison; the four-key result above is authoritative.
