# Benchmark Suite

A set of benchmark scenarios for evaluating the platform's runtime characteristics. Each scenario
defines what is measured, the workload, the procedure reference, and the metrics to record. The suite
contains **no measured values**; results are recorded in `benchmark_results_template.md` following
`../validation/benchmark_methodology.md`.

---

## Conventions

- Every scenario records the environment (per the methodology) and reports median and P95 over the
  specified repetitions.
- Warm and cold measurements are reported separately where relevant.
- Inputs are fixed (the deterministic demo dataset) so runs are comparable.

---

## BM-01 — Cold Start (subsystem entry)

- **Measures:** time from a fresh process to a subsystem entry point producing output.
- **Workload:** first invocation of an observability CLI demonstration in a new process.
- **Procedure:** §3.1 of the methodology. Repetitions: 10 fresh processes.
- **Record:** total wall time; optionally interpreter-start vs. subsystem-work split.

## BM-02 — Import Cost

- **Measures:** time and memory to import a subsystem package.
- **Workload:** import the observability package only (no work performed).
- **Procedure:** §3.1/§3.2. Repetitions: 10.
- **Record:** import wall time; peak RSS after import.

## BM-03 — Memory Footprint (single demo)

- **Measures:** peak resident memory for one representative operation.
- **Workload:** a single observability demonstration on the demo dataset.
- **Procedure:** §3.2. Repetitions: 10.
- **Record:** peak RSS (median, P95).

## BM-04 — CPU Time (full demo)

- **Measures:** CPU time for the full demonstration run.
- **Workload:** the combined observability demonstration.
- **Procedure:** §3.3. Repetitions: 5.
- **Record:** user+system CPU time; top functions by cumulative time (from a profile).

## BM-05 — Inference Workflow Latency

- **Measures:** end-to-end latency of a representative predictive workflow.
- **Workload:** twin read → prediction → result on the demo dataset.
- **Procedure:** §3.4. Repetitions: 10.
- **Record:** total latency (median, P95); dataset size.

## BM-06 — Workflow Engine Latency

- **Measures:** per-step and total latency for a multi-step process.
- **Workload:** breach → twin query → prediction → retrieval → work-order draft.
- **Procedure:** §3.5. Repetitions: 10.
- **Record:** per-step latency and total (median, P95).

## BM-07 — Deployment Time (Kubernetes)

- **Measures:** time from deployment initiation to all replicas ready.
- **Workload:** apply manifests and await rollout at a fixed replica count (image pre-built).
- **Procedure:** §3.6. Repetitions: 5.
- **Record:** rollout-to-ready time (median, P95); replica count.

## BM-08 — Recovery / Rollback Time

- **Measures:** time from rollback initiation to healthy service.
- **Workload:** roll back one revision and await the in-pod health check.
- **Procedure:** §3.7. Repetitions: 5.
- **Record:** rollback-to-healthy time (median, P95).

## BM-09 — Health Check Time

- **Measures:** wall time of one health-check evaluation.
- **Workload:** local checks with a fixed probe outcome; and with a no-op probe to isolate local cost.
- **Procedure:** §3.8. Repetitions: 10.
- **Record:** evaluation time local-only and with-probe (median, P95).

## BM-10 — CI/CD Stage Duration

- **Measures:** duration of each validation stage and the total.
- **Workload:** repository validation, quality gates, release validation, deployment readiness.
- **Procedure:** §3.9. Repetitions: 5 (local) and as available (CI).
- **Record:** per-stage and total duration (median, P95); runner specification.

## BM-11 — Repository Build / Test Time

- **Measures:** clean dependency install time and full-suite run time.
- **Workload:** fresh checkout → install → `pytest tests/ -q` (1,503 tests).
- **Procedure:** §3.10. Repetitions: 5.
- **Record:** install time; full-suite time (median, P95); pytest-reported duration.

## BM-12 — Capacity Forecast Computation

- **Measures:** time to compute capacity forecasts across resource classes.
- **Workload:** forecast CPU, memory, storage, request volume, model and data growth on fixed
  histories.
- **Procedure:** §3.3/§3.4 (CPU/latency). Repetitions: 10.
- **Record:** computation time (median, P95).

---

## Coverage Map

| Concern (from methodology) | Scenario(s) |
|----------------------------|-------------|
| Cold start | BM-01, BM-02 |
| Memory usage | BM-02, BM-03 |
| CPU usage | BM-04, BM-12 |
| Inference workflow | BM-05 |
| Workflow latency | BM-06 |
| Deployment time | BM-07 |
| Recovery time | BM-08 |
| Health check time | BM-09 |
| CI/CD duration | BM-10 |
| Repository build time | BM-11 |
