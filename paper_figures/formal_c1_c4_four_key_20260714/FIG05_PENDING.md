# Fig. 5 pending formal runs

Fig. 5 is intentionally not generated because the independent, formally
certified sensitivity points do not yet exist. The plotting specification
forbids interpolation, smoothing, or reuse of non-certified/limited solves.

Required one-factor formal scans:

1. wind uncertainty budget, with robust LB/UB and worst-case wind spill;
2. vessel-delay budget, with robust LB/UB and backlog area;
3. electricity--hydrogen split ratio, with robust LB/UB and LOHC outbound mass;
4. exactly one AGV resource parameter, with robust LB/UB and backlog area.

Every point must satisfy the same publication gate as C1--C4: all relevant
solver statuses are `OPTIMAL`, uncertainty coverage and primal replay pass,
no time limit is used, and the point has a complete certificate. Scan grids
remain pending user confirmation because the supplied plotting requirements
do not prescribe numerical points.

