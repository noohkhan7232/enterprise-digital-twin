# Benchmark Methodology

This document defines repeatable procedures for measuring the platform's runtime characteristics. It
contains **no measured values** — those are recorded in `../benchmarks/benchmark_results_template.md`
after execution. The aim is that any engineer can reproduce a measurement and obtain comparable
results.

---

## 1. Principles

- **Reproducibility first.** Every measurement specifies the environment, the exact command, the
  number of repetitions, and the statistic reported.
- **Report distributions, not single runs.** Record median and a high percentile (P95) over a fixed
  number of repetitions; never report a single sample as the result.
- **Control the environment.** Pin hardware, OS, Python version, dependency versions and
  configuration; record them with every result.
- **Warm vs. cold.** Distinguish cold-start measurements (first invocation) from warm measurements
  (steady state); report them separately.
- **No fabrication.** If a measurement has not been taken, the corresponding field remains empty in
  the results template.

## 2. Environment Capture (record with every run)

| Field | How to capture |
|-------|----------------|
| CPU model / cores | `lscpu` (Linux) |
| Memory | `free -h` |
| OS / kernel | `uname -a` |
| Python version | `python3 --version` |
| Dependency versions | `pip freeze` (record numpy, pyyaml, pytest) |
| Container runtime | `docker version` (if applicable) |
| Cluster version | `kubectl version --short` (if applicable) |
| Commit | `git rev-parse HEAD` |

## 3. Procedures

For each procedure: define repetitions *N* (default 10 for fast operations, 5 for slow ones), discard
the first run when measuring warm performance, and report median and P95.

### 3.1 Cold Start
- **Definition:** time from process invocation to readiness of a subsystem entry point.
- **Procedure:** measure end-to-end wall time of the first invocation of a CLI demonstration in a
  fresh process, e.g. `time PYTHONPATH=src python3 -c "from observability import main; main(['metrics','--quiet'])"`.
- **Report:** median and P95 over *N* fresh-process runs; record interpreter start separately from
  subsystem work if isolating import cost.

### 3.2 Memory Usage
- **Definition:** peak resident memory during a representative operation.
- **Procedure:** run the operation under `/usr/bin/time -v <command>` and record "Maximum resident set
  size"; for finer profiles use `tracemalloc` around the operation.
- **Report:** peak RSS (median, P95) per operation class (import-only, single demo, full demo).

### 3.3 CPU Usage
- **Definition:** CPU time consumed by a representative operation.
- **Procedure:** `/usr/bin/time -v` "User time" + "System time"; for sampling profiles use `cProfile`
  to attribute time to functions.
- **Report:** total CPU time (median, P95) and, where relevant, the top time-consuming functions.

### 3.4 Inference Workflow
- **Definition:** time to execute a representative predictive workflow end to end (twin read →
  prediction → result).
- **Procedure:** drive the workflow on the deterministic demo dataset; measure wall time around the
  workflow invocation; hold inputs fixed.
- **Report:** median and P95 latency, with the dataset size and configuration recorded.

### 3.5 Workflow Latency
- **Definition:** time for the workflow engine to advance a multi-step process to completion.
- **Procedure:** execute a fixed workflow (e.g., breach → twin query → prediction → retrieval → work
  order) and measure per-step and total latency.
- **Report:** per-step and total latency (median, P95).

### 3.6 Deployment Time
- **Definition:** time from initiating deployment to all replicas ready.
- **Procedure (Kubernetes):** time `deployment/scripts/deploy_kubernetes.sh`, or measure
  `kubectl rollout status deployment/edt-app` duration after apply; exclude image build by
  pre-building.
- **Report:** rollout-to-ready time (median, P95) for a fixed replica count.

### 3.7 Recovery Time
- **Definition:** time to restore healthy service after a rollback.
- **Procedure:** time `deployment/scripts/rollback.sh` from invocation to the in-pod health check
  passing.
- **Report:** rollback-to-healthy time (median, P95).

### 3.8 Health Check Time
- **Definition:** wall time of one deterministic health-check evaluation.
- **Procedure:** `time python3 deployment/scripts/health_check.py --root . --quiet` with a fixed probe
  outcome; separate the local-checks cost from the endpoint-probe cost by injecting a no-op probe.
- **Report:** evaluation time (median, P95), local-only and with-probe.

### 3.9 CI/CD Duration
- **Definition:** wall time of the validation pipeline stages.
- **Procedure:** time each stage (repository validation, quality gates, release validation,
  deployment readiness) locally and, separately, in the CI environment; record runner specification.
- **Report:** per-stage and total duration (median, P95), local and CI.

### 3.10 Repository Build Time
- **Definition:** time to install dependencies and run the full test suite.
- **Procedure:** time a clean dependency install (`pip install ...`) and `time pytest tests/ -q`
  from a clean checkout; pytest reports its own total duration, which should also be recorded.
- **Report:** install time and full-suite time (median, P95), with the test count (1,503) noted.

## 4. Reporting

Record every result in `../benchmarks/benchmark_results_template.md`, one row per procedure, with the
environment captured per §2. A result is only valid when accompanied by its environment and
repetition count. Results without these are considered indicative only and must be labelled as such.

## 5. What This Methodology Does Not Do

It does not provide expected values, targets or comparisons to other systems' measured performance.
The default SLO targets in the reliability policy (availability ≥ 0.99; P95 latency ≤ 250 ms; error
rate ≤ 0.01; freshness ≤ 300 s) are *objectives evaluated at runtime*, not benchmark predictions, and
must not be confused with measured results.
