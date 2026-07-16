# Formal AGV-cost and time-of-use-price case (2026-07-16)

This directory freezes the user-confirmed inputs for the 96-period large-port
case.  It is an isolated experiment and does not overwrite the authoritative
2026-07-14 formal results.

## Model path

This branch deliberately retains the original continuous equivalent-AGV
capacity inequalities.  It does **not** use the later exact-transport variable
elimination.  Empty AGV allocations are discouraged by a positive operating
cost of 60 CNY/(working vehicle h).  The cost applies only to container-working
and LOHC-working equivalent vehicles; charging and idle vehicles are excluded.
Every working vehicle still consumes 18 kW through the SOC equation.

The exact partition oracle also checks every compatible known scenario as a
valid lower bound on a leaf's maximum recourse cost.  A leaf is accepted only
after the fixed known-scenario dual, the leaf objective floor, and the final
primal replay are mutually consistent.

## Frozen status

- Input package: `inputs/recommended57/`
- Input validation: passed
- Case semantic SHA-256: `6321a8ec1eb9ee274e1d0fccef6b0ed64586c3c9ba9bc99ad41405e515947f49`
- Joint Bundle SHA-256: `f1af6d2a2d5c6d4ab362891dfd84a122e0f0ff802e2946b11a30ae4633749e53`
- Full repository tests before the formal attempt: `126 passed in 44.59s`
- C1 formal attempt: interrupted by explicit user request after approximately
  38 min 37 s, before the first complete C&CG iteration
- Formal certificate from this directory: **none**

The interruption returned a fail-closed error because all partition leaves had
not reached `OPTIMAL/INFEASIBLE`.  No incumbent or partial leaf result is
presented as a certificate.

## Intended solve order

1. C1 robust main case;
2. C3 nominal screen over integer container-group shares 50%--85% in 5-point
   increments, followed by one formal C3 run using the predeclared selection
   rule;
3. C4 nominal schedule plus exact stress certificate;
4. C2 no-hydrogen robust case.

C3 is intentionally absent from the frozen input directory because its integer
working-state capacities have not yet been selected by the agreed scan.

## Reproduce validation

From the repository root:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m runners.validate_case \
  experiments/formal_agv_cost_tou_20260716_v1/inputs/recommended57/C1_large_port_12v.yaml \
  --profile quarter_hour_96 \
  --output experiments/formal_agv_cost_tou_20260716_v1/validation/C1

/opt/anaconda3/bin/python \
  experiments/formal_agv_cost_tou_20260716_v1/scripts/verify_inputs.py
```

## Formal C1 command

The interrupted command was:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -m runners.run_case \
  experiments/formal_agv_cost_tou_20260716_v1/inputs/recommended57/C1_large_port_12v.yaml \
  --profile quarter_hour_96 \
  --output experiments/formal_agv_cost_tou_20260716_v1/runs/recommended57/C1_ProposedRobustMain \
  --max-iterations 100 \
  --cost-partition-key 'wind.down[1]' \
  --cost-partition-key 'wind.down[9]' \
  --cost-partition-key 'ship.delay[50,0,2]' \
  --cost-partition-key 'ship.delay[50,0,4]' \
  --cost-partition-key 'ship.delay[35,0,4]'
```

No solver time limit is set.  A resumed or accelerated formal run must still
reach zero MIP gap and `GRB.OPTIMAL`; tuning or diagnostic limits may never be
reported as a formal result.

See `SOLVER_DIFFICULTY_AND_PARAMETER_DIFF.md` for the complete parameter delta,
observed failure status, evidence, and exact acceleration options.
