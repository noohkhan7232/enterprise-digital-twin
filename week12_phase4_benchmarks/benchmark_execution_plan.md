# Benchmark Execution Plan

A step-by-step plan for executing the benchmark suite (`benchmark_suite.md`) using the procedures in
`../validation/benchmark_methodology.md`, and recording results in `benchmark_results_template.md`.
The plan is designed so that two engineers running it independently obtain comparable results.

---

## 1. Preconditions

- [ ] Clean checkout at a known commit/tag; record the commit hash.
- [ ] Dedicated, otherwise-idle machine (or node) to reduce interference.
- [ ] Environment captured per methodology §2 (CPU, memory, OS, Python, dependency versions, commit).
- [ ] Power and thermal settings fixed (disable CPU frequency scaling where possible, or note it).
- [ ] Background processes minimised; network quiesced for non-network scenarios.

## 2. Execution Order

Run fast, local scenarios first; run cluster scenarios last to amortise setup.

1. BM-02 Import Cost
2. BM-01 Cold Start
3. BM-03 Memory Footprint
4. BM-04 CPU Time
5. BM-12 Capacity Forecast Computation
6. BM-09 Health Check Time
7. BM-05 Inference Workflow Latency
8. BM-06 Workflow Engine Latency
9. BM-11 Repository Build / Test Time
10. BM-10 CI/CD Stage Duration
11. BM-07 Deployment Time (Kubernetes)
12. BM-08 Recovery / Rollback Time

## 3. Per-Scenario Loop

For each scenario:

1. **Warm up** (if measuring warm performance): run once and discard.
2. **Repeat** the measurement *N* times as specified in the scenario.
3. **Record** each raw sample; do not average prematurely.
4. **Compute** median and P95 from the raw samples.
5. **Capture** any anomalies (outliers, failures) in the notes column.
6. **Write** the row to `benchmark_results_template.md` with units and environment reference.

## 4. Statistical Handling

- Report **median** as the central tendency (robust to outliers) and **P95** as the tail.
- Keep **N ≥ 5** for slow scenarios and **N ≥ 10** for fast ones; increase N if variance is high.
- Note the **coefficient of variation**; if it is large, investigate environmental interference
  before trusting the result.
- Never report a single run as a benchmark result.

## 5. Cluster Scenario Notes (BM-07, BM-08)

- Pre-build images so deployment time excludes build time (state this explicitly in results).
- Verify the `kubectl` context twice before each run.
- Use a fixed replica count; record it.
- For recovery, ensure a known-good previous revision exists before measuring rollback.

## 6. Validity Rules

A result is valid only if it is accompanied by: the environment capture, the repetition count, the
statistic definitions, and any anomaly notes. Results lacking these are marked *indicative* and must
not be presented as benchmarks.

## 7. Reproducibility Check

After completing a run, have a second engineer repeat at least three scenarios (one fast, one medium,
one cluster) on the same environment. If medians differ by more than a pre-agreed tolerance,
investigate environmental differences before publishing.

## 8. Reporting

Compile the populated `benchmark_results_template.md` with a short narrative: environment summary, any
anomalies, and explicit statements of what was and was not measured. Do not interpolate or extrapolate
beyond measured points, and do not compare against unmeasured external systems.

## 9. Safety and Honesty

- If a scenario cannot be executed (for example, no cluster available), leave its rows empty and mark
  it *not executed*; do not estimate.
- Keep the distinction between measured results and the configured SLO targets explicit in any
  summary.
