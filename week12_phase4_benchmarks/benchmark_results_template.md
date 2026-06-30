# Benchmark Results Template

This template records measured results. **All measurement fields are intentionally empty** and are to
be filled after executing `benchmark_execution_plan.md`. Do not populate any field with an estimated
or fabricated value; leave it blank until measured. The empty tables here are the only permitted
placeholders in this validation package.

---

## Run Metadata

| Field | Value |
|-------|-------|
| Date (UTC) | |
| Operator | |
| Commit hash | |
| Tag / release | |

## Environment

| Field | Value |
|-------|-------|
| CPU model / cores | |
| Memory | |
| OS / kernel | |
| Python version | |
| numpy version | |
| pyyaml version | |
| pytest version | |
| Container runtime | |
| Cluster version | |
| CPU frequency scaling | |

---

## Results

> For each scenario: record median and P95 over N repetitions, with units. Note warm vs. cold where
> applicable. Leave blank if not executed and mark status accordingly.

### Local / process scenarios

| ID | Scenario | Metric | Unit | N | Median | P95 | Warm/Cold | Status | Notes |
|----|----------|--------|------|---|--------|-----|-----------|--------|-------|
| BM-01 | Cold start (subsystem entry) | wall time | ms | | | | Cold | | |
| BM-02 | Import cost | wall time | ms | | | | Cold | | |
| BM-02 | Import cost | peak RSS | MiB | | | | Cold | | |
| BM-03 | Memory footprint (single demo) | peak RSS | MiB | | | | Warm | | |
| BM-04 | CPU time (full demo) | CPU time | ms | | | | Warm | | |
| BM-05 | Inference workflow latency | latency | ms | | | | Warm | | |
| BM-06 | Workflow engine latency (total) | latency | ms | | | | Warm | | |
| BM-09 | Health check (local-only) | wall time | ms | | | | Warm | | |
| BM-09 | Health check (with probe) | wall time | ms | | | | Warm | | |
| BM-12 | Capacity forecast computation | wall time | ms | | | | Warm | | |

### Workflow per-step detail (BM-06)

| Step | Unit | N | Median | P95 | Notes |
|------|------|---|--------|-----|-------|
| Twin query | ms | | | | |
| Prediction | ms | | | | |
| Retrieval | ms | | | | |
| Work-order draft | ms | | | | |

### Build / pipeline scenarios

| ID | Scenario | Metric | Unit | N | Median | P95 | Status | Notes |
|----|----------|--------|------|---|--------|-----|--------|-------|
| BM-10 | CI/CD: repository validation | duration | s | | | | | |
| BM-10 | CI/CD: quality gates | duration | s | | | | | |
| BM-10 | CI/CD: release validation | duration | s | | | | | |
| BM-10 | CI/CD: deployment readiness | duration | s | | | | | |
| BM-10 | CI/CD: total | duration | s | | | | | |
| BM-11 | Dependency install | duration | s | | | | | |
| BM-11 | Full test suite (1,503 tests) | duration | s | | | | | |

### Cluster scenarios

| ID | Scenario | Metric | Unit | Replicas | N | Median | P95 | Status | Notes |
|----|----------|--------|------|----------|---|--------|-----|--------|-------|
| BM-07 | Deployment time (rollout-to-ready) | duration | s | | | | | | image pre-built |
| BM-08 | Recovery / rollback time | duration | s | | | | | | |

---

## Narrative Summary (to complete after measurement)

- **Environment summary:** _(fill in)_
- **Anomalies observed:** _(fill in)_
- **Scenarios not executed:** _(list with reason)_
- **Explicit scope statement:** These are measured results for the stated environment only. They are
  distinct from the configured SLO targets (availability ≥ 0.99; P95 latency ≤ 250 ms; error rate ≤
  0.01; freshness ≤ 300 s), which are runtime objectives, not benchmarks. No comparison to external
  systems is implied.

## Sign-off

| Role | Name | Date |
|------|------|------|
| Operator | | |
| Reviewer (reproducibility check) | | |
