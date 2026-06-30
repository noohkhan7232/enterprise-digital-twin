# Architecture Walkthrough

A layer-by-layer walkthrough of the platform, written for a presenter to narrate while a diagram is
on screen. It complements `presentation/technical_presentation.md` and the documentation's
`architecture_overview.md`, and follows the dependency direction from the physical world up to
operations. No code is included.

---

## Orientation

The platform is ten layers with one dominant dependency direction. Read it bottom-to-top by
*dependency*: the production-engineering subsystems operate the capability layers, and the capability
layers build on the physical-asset representation. The single invariant to keep in mind throughout:
**every boundary is crossed by an immutable, serialisable value object** — never shared mutable
state, never a call into another layer's internals. That is what makes the walkthrough a clean,
acyclic story rather than a tangle.

## Layer 1 — Digital Twin

Start at the physical world. The digital twin couples a *static* asset model — identity,
configuration, relationships — with a *dynamic* state stream synchronised from telemetry. The layer
exposes asset and state as immutable contracts. Note what it deliberately does *not* do: it does not
hand a live mutable object to every consumer. Consumers read immutable snapshots, which is why many
layers can depend on the twin without coupling to it.

## Layer 2 — Predictive Intelligence

On top of twin state and telemetry sits forecasting and prognostics — remaining-useful-life,
demand, and similar. The key engineering move is that the forecasting *strategy* is injected, so the
predictive *interface* is stable while the algorithm behind it varies per asset class. Predictions
are emitted as immutable values and bound to their provenance through MLOps, so a forecast is never
an orphan number.

## Layer 3 — Agentic AI

Autonomous, tool-using reasoning follows the classical perception–deliberation–action loop, grounded
in the platform's contracts. Two constraints make autonomy safe to operate: actions are restricted
to *typed tool interfaces*, so an agent's possible effects are enumerable; and reasoning trajectories
are *recorded*, so behaviour is reconstructable. The agent is observable like any other component.

## Layer 4 — Knowledge Intelligence (RAG)

Question answering over enterprise corpora retrieves evidence and conditions generation on it. Two
properties matter for an industrial setting: retrieved evidence travels with the answer, so
responses are *attributable*; and the corpus index is a *versioned artefact*, so the evidence
available at a moment in time is reproducible. Asset identifiers from Layer 1 let retrieval be scoped
to specific equipment.

## Layer 5 — Enterprise Workflow Engine

The workflow engine composes steps across the capability layers — a twin query, a prediction, a
retrieval, an agent action — into a governed process with explicit, deterministic state. It mirrors
the incident-lifecycle design in observability: state transitions are explicit and recorded, so a
process can be audited and replayed. Orchestration lives here, in one place, rather than being
smeared across application code.

## Layer 6 — Enterprise MLOps

This is the provenance backbone. A versioned model registry (semantic versioning, stage promotion),
a content-addressed artifact store, a reproducibility engine binding runs to source revision and
environment, and a lineage graph linking runs, datasets, artifacts and models. The payoff is the
question production AI usually cannot answer — *exactly how was this result produced?* — answered
structurally rather than by archaeology.

## Layer 7 — Production Monitoring

Monitoring watches data and model behaviour after deployment: data drift (input distribution
change), concept drift (input-to-output relationship change), prediction anomalies, data quality,
and a composite model-health score. The alert engine uses observer-style fan-out, so new reactions
attach by subscription. Critically, monitoring signals are themselves contracts consumed upward by
reliability and observability — degradation becomes an operational input, not a dashboard footnote.

## Layer 8 — CI/CD

Delivery is gated at the boundary where change enters the system: a shared validation library,
twenty quality gates, release validation against policy, and a deployment-readiness check, wired into
three workflows. The design choice worth narrating: gates report *honest* failures. When the
repository legitimately fails a gate, that is surfaced, not suppressed — the gate keeps its
diagnostic value.

## Layer 9 — Enterprise Deployment

Packaging and orchestration: a multi-stage, non-root container image and ten Kubernetes manifests
describing rolling updates with no unavailable replicas, startup/readiness/liveness probes, a
hardened security context, autoscaling, a network policy, a pod disruption budget and durable
storage. The keystone is a *single deterministic health check* used by the container, the probes and
the deployment and rollback scripts — so "healthy" is defined once. Rollback reverts and re-verifies
health automatically; high availability follows from redundancy, spread and durable state.

## Layer 10 — Enterprise Observability

The top layer turns operation into measurement: metrics (percentiles, windows, trends), distributed
tracing (timeline and critical-path analysis), structured logging, a reliability engine
(availability, MTBF, MTTR, composite score), an SLI/SLO engine (error budgets, burn rate), an
incident manager (validated lifecycle, postmortems), a capacity planner (deterministic forecasting),
an operations dashboard and a production-readiness assessment. It integrates purely by composition —
consuming the other layers' outputs and validating the presence of deployment and CI/CD assets
without importing their internals.

## Closing the Loop

Narrate the loop to finish: monitoring detects change, observability quantifies its reliability and
SLO impact, the incident manager governs response, the capacity planner anticipates resource limits,
and the readiness assessment judges fitness to ship — feeding decisions that flow back through CI/CD
and deployment. The architecture is not a stack of independent boxes; it is a closed operational loop
held together by immutable contracts and a single dependency direction.