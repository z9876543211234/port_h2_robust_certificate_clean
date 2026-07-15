# Certified C1--C4 paper figures

This independent package produces Figs. 1--4 only from the corrected four-key
formal runs in `experiments/c1_confirmed_main_budget_20260714`. All formal run
loaders fail closed unless the publication certificate is complete and every
required solve is `OPTIMAL` without a time limit.

Build from the repository root:

```bash
PYTHONPATH=src:. python -m paper_figures.formal_c1_c4_four_key_20260714.build_all
```

Each figure is exported as PDF, SVG, and a white-background 600-dpi PNG. The
`data/` folder contains exact plot tables; each figure has a source-manifest
JSON. Fig. 4(c) and Fig. 4(d) are newly solved fixed-scenario LP replays and
must also reach strict `OPTIMAL` status. Fig. 5 remains pending until its scan
grid is confirmed and every sensitivity point has a formal certificate.
