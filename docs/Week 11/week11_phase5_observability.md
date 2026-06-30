# Week 11 — Phase 5: Enterprise Observability, Reliability & Production Operations

> Subsystem: `src/observability/` (v11.5.0) with `configs/observability.yaml`,
> `configs/logging.yaml`, `configs/reliability_policy.yaml`.
> Status: production-ready, deterministic, thread-safe, pure-Python (+ NumPy).
> Integration model: **additive and compositional** — adds the final operations
> layer and modifies no prior week.

---

## 1. Business Motivation

The preceding phases produced a platform that can train, validate, monitor,
release and deploy models. What remained was the discipline that keeps such a
platform trustworthy once real traffic depends on it: knowing what is happening
right now, how reliable the system has been, where it will run out of capacity,
and whether it is fit to carry production load. This phase supplies that
discipline as code. It turns operational questions — "are we meeting our service
objectives?", "how much error budget is left?", "which span dominated that slow
request?", "are we ready to ship?" — into deterministic, testable computations
rather than dashboards bolted on after the fact. The result is an operations
layer an on-call engineer can reason about, a release manager can gate on, and
an auditor can replay, because every number it produces is reproducible from its
inputs.

## 2. Architecture

The subsystem is a set of cooperating engines over a shared library of immutable
value objects. Nothing here reaches into earlier weeks; integration is by
composition, so the operations dashboard is handed the outputs of the other
engines rather than importing their internals.

```
                         observability_models.py
        (Clock, IdGenerator, enums, 16 frozen+slots value objects)
                                   |
   ┌───────────┬───────────┬───────┴───────┬───────────┬─────────────┐
 metrics    tracing    structured        reliability  sli_slo     incident
 _engine    _engine     _logger           _engine     _engine     _manager
   │           │            │                 │           │            │
   └───────────┴────────────┴────────┬────────┴───────────┴────────────┘
                                      │
                            operations_dashboard
                                      │
                            production_readiness
```

Every engine takes an injected `Clock` (and, where identifiers are needed, an
injected `IdGenerator`), which is the single mechanism that makes the whole
subsystem deterministic. The default `Clock` advances by a fixed step on each
read; a `ManualClock` is available for tests that need to pin time precisely.

## 3. Metrics Design

The metrics engine collects points into named series across the eight required
categories — application, inference, workflow, scheduler, deployment, CI/CD,
monitoring and business KPIs. It supports aggregation (count, sum, mean, min,
max, standard deviation), percentiles (P50/P95/P99 by deterministic
linear-interpolation), rolling and time-bounded windows, least-squares trend
analysis with a flat-slope dead zone, and JSON export. Percentiles use a single
shared implementation so the metrics engine, the latency SLI and any ad-hoc
analysis all agree to the last decimal. NumPy is used for the heavy reductions;
everything that affects ordering or rounding is pinned for reproducibility.

## 4. Tracing Design

The tracing engine is a lightweight, dependency-free tracer. It records spans
with trace and span identifiers, parent links, start and end times, status and
attributes; it reconstructs a trace's timeline in deterministic order and
computes the critical path as the maximum-duration root-to-leaf walk of the span
tree. A context-manager API measures span duration via the injected clock and
propagates a `TraceContext` from parent to child, so nested work is correctly
attributed without any global state. Orphaned spans (whose declared parent was
never recorded) are treated as roots, which keeps timeline and critical-path
analysis robust against partial traces.

## 5. Reliability Model

The reliability engine derives availability, MTBF, MTTR, success and failure
rates, a composite reliability score and an operational-risk estimate from
request outcomes and outage windows. Availability is period-based when an
observation window is set and falls back to request-based availability
otherwise. The composite score is a weighted blend of availability and success
rate, discounted by operational risk; the weights are injectable so a team can
emphasise availability or correctness as their domain demands. Operational risk
combines the availability gap, the failure rate and a recovery-friction term
that rises when recovery time is large relative to time between failures.
Resolved incidents can be folded in directly as failure windows, linking the
incident manager to the reliability picture.

## 6. SLI/SLO Strategy

The SLI/SLO engine builds availability, latency, error-rate and freshness SLIs,
evaluates them against directional SLOs (higher-is-better or lower-is-better),
and computes error budgets and burn rates. For availability-style objectives the
budget is the allowed shortfall below target; for error-rate-style objectives it
is the allowed ceiling. Burn rate expresses the fraction of budget consumed
relative to the fraction of the window elapsed, so a burn rate above one means
the budget is being spent faster than the window can sustain — the signal the
reliability policy uses to freeze releases. The compliance report rolls every
registered objective into a single, JSON-serialisable view.

## 7. Incident Management

The incident manager models the full lifecycle — open, investigating,
identified, monitoring, resolved, closed — and enforces legal transitions, so an
incident cannot, for example, jump from resolved back to open. It records
severity, root cause, affected services and accumulating corrective actions,
maintains an ordered timeline, computes recovery time, and generates a
postmortem once an incident is resolved. It implements the Observer pattern:
paging systems, dashboards or audit sinks subscribe and are notified of every
transition with both the new state and the previous one. Identifiers and
timestamps come from injected generators and clocks, so an entire incident
history replays identically.

## 8. Capacity Planning

The capacity planner forecasts CPU, memory, storage, request volume, model
growth and data growth using two deterministic strategies: least-squares linear
projection for steady trends and compound (geometric) growth for exponential
ones. Each forecast reports the projected series, the growth rate, a recommended
capacity that adds configurable headroom over the projected peak, and — when a
capacity limit is supplied — the first step at which the resource is projected to
reach it. The Strategy pattern keeps the two algorithms interchangeable behind a
single `forecast` call, and forecasts for many resources can be produced in one
pass.

## 9. Production Operations

The operations dashboard composes the outputs of every engine into a single
`OperationsSnapshot` — metrics summary, reliability score, active incidents, SLO
compliance, capacity forecasts and readiness score — and renders a deterministic
executive summary with a graded recommendation. The recommendation is computed,
not editorialised: an active SEV1 demands incident response and a release freeze;
elevated operational risk demands investigation before the next release; a low
readiness score demands closing gaps before promotion; otherwise the system is
reported as operating within targets.

## 10. Operational Procedures

Day-to-day use follows a consistent shape. Engines are constructed with injected
clocks, fed observations (metric points, spans, request outcomes, SLIs,
incidents, resource histories), and queried for derived views. The production
readiness assessment validates ten areas — architecture, security, reliability,
monitoring, deployment, CI/CD, testing, documentation, MLOps and observability —
against the repository and any injected live metrics, and produces a weighted
score in the range nought to one hundred with a categorical level from not-ready
through conditional and ready to exemplary. The CLI demos (`metrics`, `tracing`,
`reliability`, `capacity`, `readiness`, and `all`) run each subsystem on fixed
inputs and emit sorted-key JSON, so their output is byte-identical on every run.
The companion `production_runbook.md` documents incident response, and
`architecture_decision_records.md` records why the subsystem is built the way it
is.

## 11. Design Decisions

Three decisions shape everything. First, determinism by construction: time and
identifiers are injected, collections are sorted before serialisation, and
floating-point results are rounded consistently, so outputs are reproducible and
testable offline. Second, immutability: all sixteen value objects are frozen and
slotted, carry `to_dict`/`from_dict`, and round-trip through JSON without loss,
which makes them safe to share across threads and to persist or transmit. Third,
self-containment: the subsystem deliberately reimplements its small slice of
clock, identifier and percentile logic rather than depending on Prometheus,
Grafana, Jaeger, Zipkin, ELK or the OpenTelemetry SDK, so it has no external
moving parts and integrates with earlier weeks purely by composition.

## 12. Enterprise Applications

In an enterprise setting this layer is the connective tissue of production
operations. The metrics and tracing engines answer "what is happening and why is
it slow"; the reliability and SLI/SLO engines answer "are we keeping our
promises and how much room is left"; the incident manager answers "what broke,
why, and what did we do about it"; the capacity planner answers "when do we run
out"; and the production-readiness assessment answers "should this ship". Because
every answer is deterministic and JSON-serialisable, the same computations feed
live dashboards, release gates, audit trails and postmortems without divergence.

## 13. Integration with CI/CD, Monitoring and MLOps

Integration is compositional and one-directional. The production-readiness
assessment credits the CI/CD workflows and quality gates from Phase 3 and the
container, Kubernetes and health-check assets from Phase 4, so the same artifact
that gates a release also counts toward the readiness score. The monitoring
subsystem from Phase 2 and the MLOps subsystem from Phase 1 are validated as
present-and-importable readiness areas, and their runtime outputs (model health,
drift, registry state) are exactly the kind of signal the metrics engine ingests
and the dashboard surfaces. Nothing in this phase imports those subsystems'
internals; it consumes their outputs and reports on their presence, which keeps
the locked weeks untouched while completing the operational picture.