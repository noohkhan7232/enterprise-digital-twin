# Week 10 · Phase 5 — Enterprise Integration Layer

**Component:** `src/integration/enterprise_integration_layer.py`
**Status:** New package and the platform's single integration gateway. Composes every module by adapter injection; imports and modifies none. No prior file is touched and no public API is changed.
**Dependencies:** Pure Python (standard library) + NumPy. No FastAPI, gRPC, Kafka, RabbitMQ, Redis, or Celery.

---

## 1. Business motivation

By the end of Phase 4 the platform has eight substantial subsystems — Prediction, Simulation, Knowledge, Executive Copilot, Workflow Engine, Business Process Orchestrator, Event Bus, and Scheduler. Left to themselves they would wire directly into one another, and every new capability would mean editing the modules it touches. On a frozen, additively-built platform that is untenable: it breaks encapsulation, multiplies coupling, and makes the system impossible to reason about as it grows.

The Integration Layer solves this by becoming the **single gateway** through which modules coordinate. It is deliberately *not* a business-logic layer, *not* an API, and *not* a service locator handing out references. It holds no domain logic; it routes, dispatches, and observes. Each module is wrapped once as an adapter and registered; thereafter callers ask the layer to dispatch a request by name, by capability, by priority, by condition, by fallback, by broadcast, or as a pipeline — and the layer resolves the target, invokes it with resilience (retry, timeout, circuit breaker, fallback), records an immutable audit entry, and updates health and observability statistics. Modules stay loosely coupled and independently testable; the platform gains one place to see and govern all cross-module traffic.

---

## 2. Architecture

```
                         ┌──────────────────────────────────────────────┐
   callers ──request──▶  │           EnterpriseIntegrationLayer          │  ◀── snapshot/stats
                         │                   (Facade)                    │
                         └──────────────────────────────────────────────┘
                            │            │             │            │
              ┌─────────────┘            │             │            └──────────────┐
              ▼                          ▼             ▼                           ▼
        Module registry          Routing engine   Dispatch engine            Observability
        (descriptors,            (Strategy:        (retry / timeout /         (audit log,
         adapters, freeze)        direct/cap/      circuit breaker /          statistics,
                                  priority/cond/    fallback)                  health monitor)
                                  fallback/
                                  broadcast/
                                  pipeline)
              │                                         │
              ▼                                         ▼
        ModuleAdapter callables  ───────────────▶  real modules (Prediction, Workflow,
        (Adapter pattern; injected)                 Scheduler, Event Bus, ... unmodified)
```

The layer is a **Facade** over four collaborators: a module **Registry** of **Adapters**, a **Strategy**-based routing engine, a deterministic dispatch engine, and an observability subsystem (audit, statistics, health). Everything is injected (clock, circuit thresholds, audit cap). The domain model is ten `frozen=True, slots=True` dataclasses with symmetric `to_dict`/`from_dict`. All mutable state lives behind a single re-entrant lock; adapters are invoked deterministically and the layer returns only immutable objects.

---

## 3. Registry design

Modules are registered as a `ModuleDescriptor` (id, type, semantic version, capabilities, priority, metadata) plus a `ModuleAdapter` callable. The descriptor is frozen and serialisable; the adapter is held separately so the registration record stays immutable. The registry enforces duplicate detection (re-registering an id requires explicit `overwrite`), version validation (dotted numeric components), and provides `register_module`, `unregister_module`, `list_modules`, `find_module` (by id, capability, or type — returned in deterministic priority order), `module_exists`, `freeze_registry` (seals the registry so no further mutation is possible — the production posture once wiring is complete), and `registry_snapshot`. Per-module runtime state (health counters, circuit state) is created on registration and isolated per module.

---

## 4. Routing engine

Routing is the Strategy pattern; `resolve_route` maps a request to one or more target modules:

- **Direct** — an explicit target id.
- **Capability** — the highest-priority module advertising a capability (deterministic id tie-break).
- **Priority** — the highest-priority module, optionally filtered by type.
- **Conditional** — the first matching `RoutingRule` (ordered by rule priority), matching on capability, type, and/or a context-metadata condition.
- **Fallback** — a primary target with a secondary used on failure (handled in dispatch).
- **Broadcast** — every module matching a capability or type, fanned out.
- **Pipeline** — a sequenced multi-stage execution (via `dispatch_pipeline`).

`validate_route` resolves without dispatching, so callers can verify a request is routable before committing. All candidate ordering is `(priority descending, module id ascending)`, making every routing decision deterministic.

---

## 5. Pipeline engine

`dispatch_pipeline` executes an ordered list of stages — given as module ids or, with `by_capability`, as capabilities resolved per stage. Each stage runs through the full dispatch engine; on success its output is threaded into the next stage's context (exposed as both `{module_id}_output` and `last_output`), so a stage reads its predecessor's result. By default the pipeline stops at the first failure (configurable). This expresses the flagship flow deterministically: prediction → risk assessment → knowledge retrieval → executive recommendation → workflow creation → business process → scheduler → event publication. Because context is immutable and threading is explicit, a pipeline replays identically every run, which the test suite verifies.

---

## 6. Dispatch engine

`dispatch` resolves the route then invokes the target with four layers of resilience: **retry** up to `max_retries` (with optional logical backoff), **timeout** (an adapter reports a duration; exceeding the request timeout yields `TIMEOUT`), **circuit breaker** (per module: consecutive failures at or above the threshold open the circuit; subsequent calls short-circuit to `CIRCUIT_OPEN` until a cooldown elapses, then a half-open trial closes the circuit on success or re-opens on failure), and **fallback** (on failure or an open circuit, an optional fallback target is tried once and, if it succeeds, the response is marked `FALLBACK`). `dispatch_batch` dispatches a sequence; broadcast dispatch aggregates per-target outputs into a single response whose status is `SUCCESS`, `PARTIAL`, or `FAILURE`. Every dispatch produces an immutable `IntegrationResponse` and an `IntegrationAudit` record.

---

## 7. Health monitoring

Each module accrues success, failure, and timeout counts, a duration sum, a last-seen timestamp, a heartbeat counter, and circuit state. `health(module_id)` computes an `IntegrationHealth`: availability (from circuit state), mean response time, failure and success counts, success rate, a health score (`success_rate × availability`), last seen, heartbeat, and a derived `HealthState` — `UNKNOWN` before any traffic, `UNHEALTHY` when the circuit is open or the success rate is at or below one-half, `DEGRADED` below ninety percent, otherwise `HEALTHY`. `heartbeat(module_id)` lets a module signal liveness independent of dispatch traffic, and `health_all` returns the fleet view.

---

## 8. Audit

Every dispatch appends an immutable `IntegrationAudit` capturing sequence, request id, correlation id, module id, timestamp, duration, status, and route. The log is an append-only tuple with an optional ring-buffer cap for bounded memory. Because the layer is deterministic, the audit trail is reproducible — the same inputs yield the same sequence of records — which makes it suitable as a tamper-evident record of cross-module coordination and as the substrate for the observability statistics.

---

## 9. Observability

`statistics()` returns an immutable `IntegrationStatistics`: total requests, successful/failed counts, timeouts, fallbacks, circuit-open short-circuits, pipeline runs, average dispatch time, throughput (requests per unit of elapsed logical time), success and failure rates, per-module usage, and per-route counts. `snapshot()` combines the registry, statistics, the full health fleet, and the audit count into a single `IntegrationSnapshot` — a point-in-time, serialisable view of the entire integration surface, ideal for dashboards and periodic export. NumPy backs the numeric aggregation.

---

## 10. Performance and complexity

Registration and lookup are dictionary operations; `find_module` and routing are `O(M)` over registered modules plus an `O(K log K)` ordering of the `K` candidates. A direct dispatch is `O(1)` resolution plus the adapter cost and `O(1)` audit/health updates. A pipeline of `N` stages is `N` dispatches with `O(1)` context threading each. Broadcast is `O(T)` over `T` matched targets. There is no I/O, no network, no thread spawned per request, and no real sleep on the hot path. The suite exercises a 200-module registry, a 100-stage pipeline, a 500-request batch, and ten threads dispatching concurrently.

---

## 11. Design decisions

| Decision | Rationale |
| --- | --- |
| Facade over registry + routing + dispatch + observability | One coordination point; modules never reference each other. |
| Adapters injected per module | Composition with every subsystem without importing or modifying it. |
| Strategy-based routing (seven modes) | One engine covers direct/capability/priority/conditional/fallback/broadcast/pipeline. |
| Frozen + slotted dataclasses; immutable responses/snapshots | Safe sharing across threads, lossless JSON serialisation, reproducibility. |
| Injectable logical clock | Byte-identical audits, durations, and pipeline replays under test. |
| Per-module circuit breaker | Failures are isolated; a sick module cannot drag down the gateway. |
| Single re-entrant lock; adapters invoked under controlled state | Thread safety with deterministic ordering and tie-breaks. |
| Freeze-able registry | A production posture that seals wiring once the platform is assembled. |
| No web/RPC framework | The gateway is in-process and deterministic; transport is a future concern, not a core one. |

---

## 12. Enterprise use cases

- **Decision pipeline.** An asset anomaly flows prediction → risk → knowledge → copilot → workflow → process → scheduler → event publication through one `dispatch_pipeline` call.
- **Capability routing.** A caller asks for `recommend` without knowing which module currently provides it; the layer selects the highest-priority provider.
- **Resilient fan-out.** A status change is broadcast to every `publish`-capable module; partial failures are reported without aborting the rest.
- **Graceful degradation.** A flaky module's circuit opens and traffic is transparently served by a fallback, while health and audit make the degradation visible.
- **Governance.** The immutable audit and snapshot give compliance and operations a single, reproducible view of all cross-module traffic.

---

## 13. Future integration with FastAPI

Because dispatch is a pure function of request to response, a FastAPI service can sit in front of the layer with near-zero glue: each HTTP endpoint constructs an `IntegrationRequest` from the request body, calls `dispatch` (or `dispatch_pipeline`), and serialises the resulting `IntegrationResponse.to_dict()` as JSON. Module registration happens once at application startup. The layer's determinism and immutability make it safe to share a single instance across worker requests behind its lock, and the audit/statistics endpoints map directly onto `audit_log`, `statistics`, and `snapshot`. No change to the layer is required — the web framework is strictly additive.

---

## 14. Future integration with a dashboard

`snapshot()` already returns everything a dashboard needs in one serialisable object: the module registry with versions and capabilities, the full health fleet (state, score, response time, circuit state, heartbeat), aggregate statistics (throughput, success/failure rates, route and module usage), and the audit count. A dashboard polls `snapshot()` (or streams the audit log) and renders it; because every field is JSON-serialisable and deterministic, historical snapshots can be stored and replayed. Live module health, per-route traffic, and pipeline run counts become first-class panels without any modification to the integration layer or the modules behind it. All existing modules remain untouched, preserving full compatibility with Weeks 1–10 Phase 4.

---

## 15. Verification

The suite `tests/test_enterprise_integration_layer.py` is standard pytest and contains **335 collected test cases, all passing**, covering: every enum; serialization round-trips for all ten dataclasses (and a parametrized sweep over module types, response statuses, and strategies); the full registry API including duplicate detection, version validation, find-by-id/capability/type, and freeze semantics; all seven routing strategies plus route validation and conditional-rule precedence; direct/batch/broadcast/pipeline dispatch; retry (attempt counts and eventual success), timeout, fallback, and the circuit breaker (open after threshold, short-circuit, half-open recovery, fallback-on-open); per-module health and heartbeat with threshold-derived states; the immutable audit trail (sequence, correlation, route, capping); observability statistics and snapshots; determinism (identical pipeline replays and audit logs across runs, deterministic tie-breaks); thread safety (concurrent registration and dispatch); large registries, pipelines, and batches; frozen/slots and JSON guarantees; the CLI; and non-invasiveness checks confirming no FastAPI/gRPC/broker imports and no upstream-module imports. Run the demonstration with `python src/integration/enterprise_integration_layer.py --demo`.