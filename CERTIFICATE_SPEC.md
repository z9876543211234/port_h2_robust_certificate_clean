# Certificate Specification

## Fail-closed rule

A formal result uses only models with `Status == GRB.OPTIMAL`. `TIME_LIMIT`,
`NODE_LIMIT`, `ITERATION_LIMIT`, `SOLUTION_LIMIT`, `MEM_LIMIT`, `INTERRUPTED`,
`NUMERIC`, `SUBOPTIMAL`, and `INF_OR_UNBD` are rejected. Candidate searches may
use limits but may only propose selectors for exact replay; they never update a
formal upper bound or certify absence of violations.

## C1/C2/C3 acceptance

The certificate is `engineering_optimal=true` only if all of the following hold:

- master, Phase-I adversary, cost adversary, and every recourse replay are
  `OPTIMAL`;
- uncertainty coverage is complete;
- adversary/primal replay passes within `1e-7`;
- scaled and original-unit residual checks pass, with maximum absolute residual at
  most `1e-6`;
- bounds are ordered and
  `(UB-LB)/max(1, abs(UB), abs(LB)) <= 0.01`;
- no limited solve contributes to the certificate.

The upper bound is formed from the formal adversary `ObjBound`, not a candidate
incumbent. A duplicate worst realization with an open gap fails as
`inconsistent_or_stalled_duplicate`.

## C4 acceptance

C4 requires an `OPTIMAL` nominal master, fixed-first-stage Phase-I adversary,
`OPTIMAL` stress adversary, `OPTIMAL` worst-scenario replay, and passing residuals.
It reports nominal and stress objectives and the worst realization.

## Provenance and recovery

Certificates include case semantic hash, Bundle hashes, source hashes, Git commit,
Gurobi version, solver parameters, statuses, bounds, gaps, and residual gates.
Checkpoints are atomically written after every completed outer iteration. Only
completed `OPTIMAL` solves and proved-empty partition leaves are recoverable as
formal evidence.

