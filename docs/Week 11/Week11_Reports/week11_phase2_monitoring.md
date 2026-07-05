# Week 11 — Phase 2: Enterprise Production Monitoring & Data Drift Intelligence

> Subsystem: `src/monitoring/`
> Status: production-ready, deterministic, pure Python + NumPy.
> Integration model: **composition only** — adds new files; modifies no prior week.

---

## 1. Business Motivation

A deployed model is a depreciating asset. The world it was trained on moves —
customer behaviour shifts, upstream pipelines change schema, sensors drift — and
accuracy silently erodes long before anyone files a ticket. The cost of that
silence is paid in bad decisions: declined-but-creditworthy customers, missed
fraud, mis-priced risk. Production monitoring exists to convert silent
degradation into an explicit, auditable signal that triggers action.

This subsystem provides that signal for the platform's deployed models. It
answers, continuously and deterministically: has the input distribution moved
(data drift)? has the relationship between inputs and outcomes moved (concept
drift)? are predictions, latency and error rates within bounds? is the data
feeding the model clean? and, rolled up: is this model healthy enough to keep
serving traffic? It deliberately reimplements the statistics from first
principles rather than depending on Evidently, WhyLabs, Arize, Prometheus or
Grafana, because regulated and safety-critical deployments require every number
to be owned, explainable and reproducible.

## 2. Architecture

The subsystem is layered on the immutable value objects in
`monitoring_models.py`, with five independent detectors and a dashboard/alert
layer composed on top.

```
                          monitoring_models.py
        (16 frozen dataclasses · enums · Clock · IdGenerator · validation)
                                    |
   +-----------------+-------------+-------------+------------------+-----------+
   |                 |             |             |                  |           |
 data_drift     concept_drift  prediction    model_health     data_quality     |
 _detector.py   _detector.py   _monitor.py   _monitor.py      _monitor.py      |
 (PSI/KL/JS/    (rolling/      (latency/     (composite       (missing/dupes/  |
  KS/hist)       sliding/       throughput/   weighted          ranges/outliers|
                 residual)      error rate)   health score)     /schema)       |
   +-----------------+-------------+-------------+------------------+-----------+
                                    |
                          monitoring_dashboard.py
        (AlertEngine — Observer pattern · MonitoringDashboard · executive summary)
```

Cross-cutting patterns:

- **Dependency Injection** — every detector and the dashboard accept a
  `MonitoringConfiguration`, a `Clock` and an `IdGenerator`; factories assemble
  deterministic or system-backed variants.
- **Strategy** — drift method (PSI / KL / JS / KS / histogram), alert policies
  and health weights are all swappable behaviours.
- **Observer** — the `AlertEngine` fans out every emitted alert to subscribed
  observers (callables or objects with `on_alert`).
- **Immutability** — all results are `frozen=True, slots=True` dataclasses with
  symmetric `to_dict` / `from_dict`.

Integration with the MLOps platform, model registry, experiment tracker,
workflow engine, scheduler, event bus, integration layer and executive copilot
is **by composition**: collaborators are injected as plain values or duck-typed
objects. The subsystem imports none of them, so it remains independently
testable and the earlier weeks remain untouched.

## 3. Drift Detection Algorithms

All algorithms operate on probability vectors obtained by binning the reference
sample on its own quantile edges, with the outer edges extended to ±∞ so that
production values outside the reference range are captured. Proportions are
floored at 1e-9 and renormalised to avoid division by zero.

- **Population Stability Index** — `Σ (q − p)·ln(q/p)`. Zero for identical
  distributions; the conventional bands (<0.1, 0.1–0.25, >0.25) map onto the
  severity classifier relative to the configured threshold.
- **Kullback–Leibler divergence** — `KL(current ‖ reference) = Σ q·ln(q/p)`,
  the expected surprise of production data under the reference model.
- **Jensen–Shannon distance** — `√JSD` with base-2 logs, bounded in [0, 1] and
  far more stable than raw KL. Because binning is reference-relative, it is
  symmetric only approximately.
- **Kolmogorov–Smirnov** — the two-sample statistic `D = max|F_ref − F_cur|`
  computed over the merged order statistics, with an asymptotic p-value from the
  Kolmogorov series.
- **Histogram distance** — total-variation distance `½ Σ|p − q|`, bounded [0, 1].
- **Categorical drift** — PSI over the union of category frequencies.

The detector produces an **overall drift score** (mean of feature scores), a
**feature drift ranking** (descending, name-tie-broken for determinism), a
**severity classification** (NONE / LOW / MODERATE / HIGH / CRITICAL relative to
threshold) and a **confidence score** that rises with the smaller of the two
sample sizes.

## 4. Concept Drift

Concept drift is inferred from changes in model behaviour over time:
**rolling performance** (earliest vs latest window plus a least-squares trend
slope), **sliding window** (early vs late accuracy), **error-distribution drift**
(PSI on errors), **residual drift** (JS distance on regression residuals),
**target drift** (PSI on a continuous target), **label-distribution drift** (PSI
on class frequencies) and **trend analysis** (slope sign → increasing /
decreasing / flat). Near-zero slopes are snapped to exactly zero so that a flat
series is never misclassified as trending. Each method returns a
`ConceptDriftResult` with a severity derived from the same threshold-relative
classifier as data drift.

## 5. Prediction Monitoring

`PredictionMonitor` records batches of predictions, confidences, latencies and
inference times under a re-entrant lock and aggregates them into a
`PredictionStatistics` value object: mean, standard deviation, variance, min,
max and the 5th/50th/95th percentiles of the prediction distribution; positive
rate; confidence mean and standard deviation; latency p50/p95/p99; throughput
(records per second over an elapsed window); error and success rates; and mean
inference time. A stateless `compute_statistics` variant supports one-shot
analysis, and `prediction_drift` compares reference and production prediction
distributions via JS distance or PSI.

## 6. Health Scoring

`ModelHealthMonitor` reduces eight normalised components — accuracy trend,
latency trend, prediction stability, drift, availability, resource usage,
reliability and freshness — to a single score in [0, 1]. Each component is
oriented so that 1.0 is perfectly healthy: drift and resource usage are
inverted, latency is scored against its budget with a penalty for a rising
trend, and freshness decays linearly against a configurable budget. The weighted
sum (weights normalised to 1.0) is mapped to EXCELLENT, HEALTHY, WARNING or
CRITICAL using the configured thresholds. Weights are injectable, so an
organisation can re-prioritise (for example, weighting latency more heavily for
a real-time service) without code changes.

## 7. Data Quality

`DataQualityMonitor` evaluates a tabular dataset and returns `QualityMetrics`
plus a deterministic, sorted list of `QualityIssue` records. It detects missing
values and the overall null rate (and its complement, completeness), duplicate
records (and the complementary consistency score), values outside declared valid
ranges (and the resulting validity rate), outliers by z-score against a
configurable threshold, schema violations against an expected column set, and
data staleness against a freshness budget. Each issue carries a severity graded
by its rate, so a 0.2% null rate and a 40% null rate are not treated alike.

## 8. Alert Engine

The `AlertEngine` turns metrics into alerts deterministically. Declarative
`AlertPolicy` rules (metric, comparison, threshold, level, type) drive
**threshold** alerts; **trend** policies evaluate injected slopes; **composite**
alerts summarise co-occurring conditions (the dashboard raises a CRITICAL
`model_degradation` alert when drift and health degrade together); and
**repeated** alerts are recognised by a stable fingerprint
(`sha256(metric|level|entity)`) whose recurrence count is tracked, so the same
condition is deduplicated rather than spamming downstream consumers. Only the
most severe firing policy per metric is retained. Every emitted alert is fanned
out to subscribed observers, decoupling detection from delivery (event bus,
paging, dashboards) entirely.

## 9. Dashboard

`MonitoringDashboard` assembles two artifacts. `build_report` produces a
`MonitoringReport` that bundles drift results, concept drift, prediction drift,
health, quality and generated alerts with the overall drift score.
`build_snapshot` produces a `DashboardSnapshot` containing the health score, a
drift summary, the top drifted features, prediction and latency trend maps, data
quality, active alerts and a model-status string. `executive_summary` renders a
deterministic, human-readable briefing with a recommended action graded by the
most severe active alert.

## 10. Engineering Decisions

- **Determinism end to end.** No wall-clock or ambient randomness; time and
  identity are injected, demo data is generated from a seeded
  `numpy.random.default_rng`, and rankings break ties by name. `run_demo()` is
  byte-identical across runs.
- **Reference-relative binning.** Binning on reference quantiles with ±∞ outer
  edges makes PSI/KL/JS robust to production values outside the training range,
  at the cost of exact JS symmetry — an acceptable, well-understood trade.
- **Threshold-relative severity.** A single classifier maps any score to a
  severity band relative to its threshold, giving consistent semantics across
  heterogeneous methods.
- **Self-contained infrastructure.** Monitoring defines its own clocks and id
  generators rather than importing earlier weeks, preserving the "never modify
  prior modules" rule and keeping the subsystem independently testable.
- **Thread safety.** Stateful components (prediction monitor, alert engine,
  dashboard counters) guard mutation with a re-entrant lock while exposing only
  immutable value objects.

## 11. Performance

Drift statistics are O(n log n) in the sample size (dominated by the sort inside
binning and the KS merge); health, quality and alert evaluation are linear in
their inputs. The suite exercises drift detection on 200k-row samples within
interactive latency, and the prediction monitor sustains concurrent recording
from multiple threads.

## 12. Complexity

| Operation | Complexity |
|-----------|------------|
| PSI / KL / JS / histogram distance | O(n log n) binning + O(b) over bins |
| Kolmogorov–Smirnov statistic | O(n log n) sort + merge |
| `detect_dataset` | O(F · n log n) over F features |
| feature drift ranking | O(F log F) |
| concept-drift trend slope | O(w) over the window |
| prediction `statistics` | O(n) + O(n log n) for percentiles |
| health `evaluate` | O(1) over fixed components |
| data-quality `evaluate` | O(F · n) + O(n) duplicate hashing |
| alert `evaluate_metrics` | O(P) over policies |

## 13. Enterprise Applications

- **Continuous assurance** of deployed models with auditable drift and health
  records suitable for model-risk-management review.
- **Automated retraining triggers** — a scheduler can run `run_demo`-style
  pipelines and act on health level or composite alerts.
- **Incident response** — deterministic snapshots reproduce exactly what the
  dashboard showed at the time of an incident.
- **Executive reporting** — the executive summary translates statistics into a
  decision and a recommendation.

## 14. Integration with Enterprise MLOps

The subsystem composes with the Phase 1 MLOps platform without importing it. A
model promoted through the registry carries an id and version; the monitor keys
its reports and snapshots on that id. Drift scores and health levels are designed
to feed promotion gates (block promotion or trigger rollback on CRITICAL), the
event bus (publish alerts to subscribers), the scheduler (run periodic checks),
the workflow engine (orchestrate retraining) and the executive copilot (consume
the executive summary). Because every output is a JSON-serialisable value object
and every collaborator is injected, wiring monitoring into the wider platform is
configuration, not modification.