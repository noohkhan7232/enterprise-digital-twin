# Architecture Decision Records — Observability & Production Operations

> ADRs for Week 11 Phase 5 (`src/observability/`, v11.5.0). Each record states a
> decision, its context, and its consequences. These decisions are binding for
> the subsystem and explain why it is built as it is.

---

## ADR-001: Compose with prior weeks; never import their internals

**Context.** The platform spans many locked subsystems (MLOps, monitoring,
workflow, scheduler, integration, CI/CD, deployment). The observability layer
must reflect their state without destabilising them.

**Decision.** Integrate purely by composition. The observability engines accept
the *outputs* of other subsystems (summaries, scores, incidents, histories) and
the production-readiness assessment validates other subsystems' *presence* on
disk, but no observability module imports another week's internals.

**Consequences.** The locked weeks remain untouched and cannot be broken by this
phase. The coupling is one-directional and explicit, which keeps the dependency
graph acyclic and the subsystem independently testable. The cost is that the
dashboard must be *handed* data rather than reaching for it, which is a
deliberate, healthy constraint.

## ADR-002: Determinism by construction via injected clocks and identifiers

**Context.** Observability output must be reproducible for testing, auditing and
offline demos, yet real systems depend on wall-clock time and unique ids.

**Decision.** Inject a `Clock` (and an `IdGenerator` where needed) into every
engine. The default clock advances by a fixed step per read; a `ManualClock`
allows precise control. Collections are sorted before serialisation and
floating-point results are rounded consistently.

**Consequences.** Every computation replays identically from its inputs, so tests
need no mocking of time and the CLI demos are byte-stable. Production deployments
can supply a wall-clock-backed clock without changing any engine. The cost is the
discipline of never reading time directly, enforced by code review.

## ADR-003: Immutable, slotted, JSON-serialisable value objects

**Context.** Observability data crosses threads, is persisted, and is emitted as
JSON. Mutable shared state is a frequent source of subtle bugs.

**Decision.** Model all sixteen domain types as `frozen=True, slots=True`
dataclasses with `to_dict`/`from_dict` and lossless JSON round-trips. Mutable
inputs (mappings) are frozen into sorted tuples on construction.

**Consequences.** Values are safe to share across threads and to cache, equality
and hashing are well-defined, and serialisation is uniform. The round-trip
guarantee is covered by tests for every type. The cost is that "updates" produce
new objects, which suits an event/observation model well.

## ADR-004: No external observability dependencies

**Context.** Prometheus, Grafana, Jaeger, Zipkin, ELK and the OpenTelemetry SDK
are the industry defaults, but each adds heavy runtime dependencies, network
services and version surface.

**Decision.** Implement the needed slice — metric series and percentiles,
spans and critical-path analysis, structured JSON logs, SLO and error-budget
math — in pure Python plus NumPy, with original implementations inspired by those
systems.

**Consequences.** The subsystem has no external moving parts, installs trivially,
and runs deterministically offline, which is exactly what the build environment
and the test strategy require. The trade-off is that it is not wire-compatible
with those ecosystems; bridging (e.g. exporting to a real backend) is left as a
clean, additive extension point rather than a built-in dependency.

## ADR-005: Inject live operational metrics into production readiness

**Context.** Some readiness areas (reliability, testing) depend on runtime
results that cannot be derived from the filesystem alone, yet the assessment must
remain deterministic and runnable offline.

**Decision.** Let `ProductionReadiness` accept injected signals — reliability
score, tests passed/failed, coverage — and fall back to presence-based scoring
when they are absent. Area weights and thresholds are also injectable.

**Consequences.** The assessment is honest in both modes: it reports a genuine
score from live data when supplied, and a conservative presence-based score
otherwise. It stays deterministic and offline-friendly, and organisations can
re-weight areas to match their risk posture without code changes.

## ADR-006: Observer pattern for incidents and logging

**Context.** Incident transitions and log records must fan out to multiple
downstream consumers (paging, dashboards, audit) without those consumers being
hard-wired into the producers.

**Decision.** Use the Observer pattern. The incident manager notifies subscribed
observers of every transition with the new and previous states; the structured
logger fans records out to subscribed sinks (callables or objects with
`on_log`). Subscription is idempotent and unsubscription is supported.

**Consequences.** Producers stay decoupled from consumers, new reactions are
added by subscribing rather than by editing the producer, and the fan-out is
deterministic in subscription order. The cost is that observers must be
well-behaved; the contract is documented and exercised by tests.

## ADR-007: Strategy pattern for capacity forecasting

**Context.** Different resources grow differently — some linearly, some
compounding — and the planner must support both without branching logic leaking
through its interface.

**Decision.** Expose forecasting as a single `forecast` call parameterised by a
strategy (`linear` or `compound`), each a self-contained deterministic algorithm
selected at call time.

**Consequences.** Callers pick the model that fits the resource without the
planner's surface changing, and new strategies can be added behind the same
interface. Both strategies are deterministic and covered by tests, including flat
and decreasing histories and exhaustion detection.