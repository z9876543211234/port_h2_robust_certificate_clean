# 20 MW base load / 34-AGV C1 and C3 experiment

This isolated experiment is derived from the preserved 40 MW / 57-AGV input
package.  The only common physical changes authorized for C1 and C3 are:

- port base load: 40 MW -> 20 MW in all 96 periods;
- AGV fleet size: 57 -> 34 vehicles.

All other deterministic parameters and all exogenous wind and ship-delay
inputs are copied unchanged.  C3 additionally applies an integer container /
LOHC work-capacity split selected from the previously confirmed 50%--85%
container-share scan at 5-percentage-point increments.

The prior 40 MW / 57-AGV directory remains untouched and is additionally
preserved as `experiments/preserved_runs_20260717/
base_load_40mw_agv57_complete_20260717.tar.gz`.

Formal runs use the semantic ship-delay partition, require every nonempty leaf
to reach `OPTIMAL`, and do not accept time-limited results.
