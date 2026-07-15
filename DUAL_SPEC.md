# Dual and Adversary Specification

## Standard form

Every recourse relation is registered once in the linear IR. Variable lower bounds
are shifted so the compiled recourse has

\[
\min_{\bar y\ge0} c^T\bar y+c_0,
\quad A_{eq}\bar y=b_{eq}(x,u),
\quad A_{ge}\bar y\ge b_{ge}(x,u).
\]

Its generated dual is

\[
\max_{\pi,\mu}\ b_{eq}(x,u)^T\pi+b_{ge}(x,u)^T\mu+c_0
\]

subject to

\[
A_{eq}^T\pi+A_{ge}^T\mu\le c,
\qquad \pi\text{ free},\quad \mu\ge0.
\]

No physical row may be handwritten separately in primal, master, dual, adversary,
or residual code.

## Phase-I

Equalities receive positive and negative slacks; `ge` rows receive a nonnegative
slack. The scaled slack objective uses `sigma_i = 1`. A formal exact Phase-I
adversary proves robust feasibility only when its status is `GRB.OPTIMAL` and its
objective is at most `1e-8`.

## Dual-only adversary

All Bundle selectors, including exact quay-crane saturation auxiliaries, are
binary. For selector `z_j` and dual affine term `g_j(lambda)`, only one product
variable `w_j = z_j g_j(lambda)` is created. Gurobi indicator constraints impose
`z_j=0 => w_j=0` and `z_j=1 => w_j=g_j(lambda)`. Rowwise
`z_j * lambda_i` products and empirical Big-M constants are prohibited.

Exact fixed-scenario recourse values may add first-stage-hash-local weak-duality
cuts. A cut is never reused after the first-stage hash changes.

### Gurobi 13 parameter exception

The indicator adversary deliberately leaves `DualReductions` at Gurobi's default.
With `DualReductions=0`, Gurobi 13.0.1 reported an `OPTIMAL` zero objective after
an unbounded root relaxation on the 8/12/24/96 validation models, while complete
enumeration proved a strictly positive worst-case value. Keeping dual reductions
enabled restores exact agreement on all enumerated sets. Bundle-feasibility LPs
used to prove empty partition leaves still set `DualReductions=0` because they do
not contain indicator products.

## Replay and partitions

The adversary selector is replayed through the original recourse LP and must match
within `1e-7`. If a partition fallback is used, leaves are mutually exclusive and
cover the Bundle. Every nonempty leaf must be `OPTIMAL`, every empty leaf must be
proved `INFEASIBLE`, and any unresolved leaf makes the global adversary unresolved.
