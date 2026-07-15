# Model Changelog

## 2026-07-13 — clean-room baseline

- Established the final specification as the sole mathematical implementation
  source.
- Separated WindBundle and integer ShipDelayBundle construction from the physical
  model.
- Rejected historical continuous arrival intensity as a model nominal, fixed moved
  fractions, early-arrival selectors, modulo wraparound, joint wind/ship budgets,
  and legacy adversary/C&CG code.
- Added a reproducible offline NHPP generator that freezes three independent days
  of Poisson integer vessel counts (seed `20260713`, daily expectation `11.52`) as
  boundary/current-day nominal data.  Fractional NHPP expectations remain audit
  metadata only and never become equivalent logistics pressure.
- Preserved the declared clean-model boundary: no fuel cell, no day-ahead LOHC
  export, no day-ahead AGV work allocation, and exogenous wind-to-hydrogen split.
- Defined formal fail-closed `GRB.OPTIMAL` certificate gates and test-only short
  horizons.
- Implemented shared first-stage/recourse IR compilers, fixed-scenario replay,
  Phase-I, explicit duals, indicator-only exact adversaries, exact partition
  fallback, C&CG, C4 stress testing, reconstruction, checkpoints, and exports.
- Migrated fixed legacy data with hashes only; no legacy solver module is imported
  or called. User-deferred parameters remain explicit formal-run blockers.
- Kept Gurobi dual reductions enabled for indicator adversaries after
  `DualReductions=0` produced false zero-objective `OPTIMAL` reports in Gurobi
  13.0.1; complete 8/12/24/96 enumeration/integration tests guard this exception.
