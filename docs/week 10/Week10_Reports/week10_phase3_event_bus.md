# Week 10 · Phase 3 — Enterprise Event Bus

**Component:** `src/events/enterprise_event_bus.py`
**Status:** New, self-contained package. Imports nothing from previous modules; existing modules consume it by composition. No prior file is modified and no public API is changed.
**Dependencies:** Pure Python (standard library) + NumPy. No asyncio, Kafka, RabbitMQ, Redis, Celery, or any external broker.

---

## 1. Business motivation

By the end of Phase 2 the platform has several layers that need to talk to each other: the Workflow Engine, the Business Process Orchestrator, the Executive RAG Copilot, and the Knowledge Agent — with dashboards, REST APIs, monitoring, and MLOps still to come. Wiring each producer directly to each consumer is the classic `O(N²)` integration trap: every new component forces edits to the ones it touches, which is exactly what a frozen, incrementally-built platform cannot tolerate.

An event bus inverts that. Producers publish facts ("workflow completed", "risk threshold exceeded") without knowing who listens; consumers subscribe to the facts they care about without knowing who produced them. Integration becomes `O(N)` — each component connects once, to the bus. This Phase 3 deliverable is that backbone: an **in-process**, deterministic, thread-safe publish/subscribe core. It is intentionally not Kafka or RabbitMQ — the platform is a single deterministic research system, so an external broker would add nondeterminism, deployment weight, and a network dependency for no benefit. The bus gives the decoupling and the audit trail of a message broker while remaining pure, embedded, and reproducible.

---

## 2. Architecture

```
   Producers                         EnterpriseEventBus                     Consumers
 ┌───────────────┐   publish()   ┌──────────────────────────────┐  deliver  ┌──────────────┐
 │ Workflow Eng. │ ────────────▶ │  Routing  →  Subscription     │ ────────▶ │ Knowledge    │
 │ BP Orchestr.  │               │  engine      registry         │           │ Dashboard    │
 │ Exec Copilot  │               │  (Observer)  (Registry)       │           │ Monitoring   │
 └───────────────┘               │                               │           │ MLOps        │
                                  │  ┌─────────────────────────┐  │           └──────────────┘
                                  │  │ immutable EventHistory  │  │
                                  │  │ DeadLetterQueue         │  │  replay()
                                  │  │ Replay engine           │ ◀┼──────────── (by topic/time/
                                  │  │ Statistics / analytics  │  │              correlation/last-N)
                                  │  └─────────────────────────┘  │
                                  └──────────────────────────────┘
```

Every collaborator inside the bus is small and single-purpose: a subscription **registry**, a **routing** matcher, an immutable **history**, a **dead-letter queue**, a **replay** engine, and an **analytics** aggregator. State is injected (clock, retry policy, history/DLQ caps, logger) so the bus is configurable and testable. The domain model is ten `frozen=True, slots=True` dataclasses with symmetric `to_dict`/`from_dict`.

---

## 3. Observer pattern

The bus is a textbook Observer with enterprise hardening. Subscribers register an `EventHandler` callable plus an `EventSubscription` describing what they want (topic pattern, delivery mode, priority, minimum event priority, optional filter, persistence). On `publish`, the bus computes the matching observers and notifies them. Two refinements matter:

- **Deterministic notification order.** Matching subscribers are sorted by `(subscriber priority descending, registration order ascending)`. Given the same subscriptions and events, delivery order is always identical — essential for a reproducible research platform and verified directly in the suite.
- **Isolation.** A handler that raises does not abort the publish or other handlers; the failure is captured (after the configured retries) into the dead-letter queue. Handlers are invoked outside the lock, so a handler may itself publish — re-entrant publishing runs depth-first and deterministically.

---

## 4. Registry pattern

Subscriptions live in a registry keyed by a generated `subscription_id`; handlers live in a parallel map so the `EventSubscription` dataclass stays frozen and JSON-serialisable (callables are not serialisable). The registry supports `subscribe`, `once` (one-time, auto-removed after its first delivery), `unsubscribe`, `clear`, and ordered enumeration. Registration order is recorded (`created_seq`) to make tie-broken delivery deterministic.

---

## 5. Routing engine

A single matcher implements all six required routing styles:

- **Topic** — exact topic equality.
- **Wildcard** — `*` matches exactly one segment, `#` matches zero or more trailing segments (`#` alone matches everything), evaluated segment-by-segment.
- **Broadcast** — the subscription matches every event regardless of topic.
- **Direct** — the event is published with a `target`; only the subscriber whose id equals the target receives it.
- **Filtered** — an attached `EventFilter` must also pass.
- **Priority** — a subscription's `min_priority` gates events below a threshold.

Filtering supports topic, priority, source, correlation id, workflow id, process id, event type, time range, and arbitrary metadata key/value — reading workflow/process ids from either the payload or the metadata tags.

---

## 6. Event lifecycle

1. **Create.** A producer builds an `EnterpriseEvent` (via `EnterpriseEvent.create`), which canonicalises the payload and computes a stable `payload_hash`.
2. **Publish / stamp.** Under the lock, the bus assigns a monotonic `sequence`, a `timestamp` from the injected clock, a default `trace_id`/`correlation_id` if absent, and a content-addressed `event_id`. The stamped event is appended to the immutable history.
3. **Route.** Matching subscribers are computed and ordered deterministically.
4. **Deliver.** Each handler is invoked (with retries on failure); successes update delivery/latency counters, failures go to the dead-letter queue.
5. **Finalise.** One-time subscriptions are removed; counters, DLQ, and history are committed.

Every event therefore carries a timestamp, sequence number, source, correlation id, trace id, and payload hash — the full provenance required for audit and replay.

---

## 7. Replay engine

Because history is a complete, immutable, ordered log, replay is a selection plus a re-delivery. The engine supports replay **by topic**, **by time range**, **by correlation id**, **last N**, and **by custom filter**. Replayed events can be sent to a dedicated sink handler (for projection rebuilds or backfills) or re-dispatched to current subscribers. Replay never mutates history and never re-increments the publish counters; it increments a separate `replayed` counter and returns an `EventReplay` record (mode, criteria, event ids, count).

---

## 8. Dead-letter queue

A delivery that still fails after the configured `max_retries` becomes a `DeadLetterEvent` capturing the original event, the subscriber id, the retry count, the failure reason, a recovery status, and timestamps. `recover_dead_letters(handler)` attempts redelivery: successes are removed from the queue, failures are marked `FAILED` and retained with an incremented retry count. The queue can be capped (`max_dead_letters`) for memory safety and cleared explicitly. This gives at-least-once semantics with explicit, inspectable failure handling rather than silent loss.

---

## 9. Analytics

`statistics()` returns an immutable `EventStatistics` snapshot: published, delivered, dropped, unrouted, and replayed counts; current subscriber and dead-letter counts; average delivery latency; failure rate and delivery-success rate (over attempted deliveries); and per-topic counts. "Dropped" counts failed `(event, subscriber)` deliveries, while "unrouted" counts events that matched no subscriber — kept distinct so an undelivered-because-nobody-listened event is never conflated with a handler failure. NumPy backs the numeric aggregation.

---

## 10. Performance and complexity

Publishing is `O(S)` in the number of subscriptions (each is tested for a match), plus `O(M log M)` to order the `M` matched subscribers, plus the cost of the handlers themselves. History append is amortised `O(1)` (with optional ring-buffer truncation when `max_history` is set). Topic matching is `O(segments)`. There is no background thread, no network, and no serialization on the hot path. The suite exercises a 1,000-event stream, a 50-way fan-out, and 8–10 threads publishing concurrently; throughput is bounded only by the injected handlers.

---

## 11. Design decisions

| Decision | Rationale |
| --- | --- |
| In-process, no external broker | Determinism, zero deployment weight, no network dependency for a single-system research platform. |
| Synchronous, ordered delivery | Reproducibility; deterministic `(priority, registration)` ordering is testable and auditable. |
| Handlers invoked outside the lock | Allows safe re-entrant publishing (depth-first) and prevents a slow handler from blocking all mutation. |
| Frozen + slotted dataclasses | Immutability, low memory, lossless JSON serialisation. |
| Injectable clock (logical by default) | Byte-identical histories and latencies under test; wall-clock available for production. |
| Separate `dropped` vs `unrouted` | A handler failure and a no-listener event are different operational signals. |
| Dead-letter queue with explicit recovery | At-least-once semantics without silent loss; failures are inspectable and replayable. |
| Self-contained (no platform imports) | The bus is the lowest common layer; nothing above it may be a dependency of it. |

---

## 12. Enterprise use cases

- **Cross-layer choreography.** The Workflow Engine publishes `workflow.completed`; the Orchestrator advances the parent process; the Knowledge Agent indexes the result — none imports another.
- **Executive alerting.** A `risk.threshold.exceeded` event at `CRITICAL` priority fans out to the Executive Copilot and a monitoring sink via priority-gated subscriptions.
- **Audit & compliance.** A single broadcast subscriber persists every event; the immutable history with payload hashes is the tamper-evident audit log.
- **Backfill & recovery.** A new dashboard replays `last N` or a correlation id to rebuild its projection without touching producers.
- **Resilient integration.** A flaky downstream consumer's failures land in the dead-letter queue and are recovered later, without affecting other subscribers.

---

## 13. Future integration

Consumers attach to the bus through the `EventHandler` callable contract, so future Dashboard, REST, Monitoring, and MLOps components integrate without any change to the bus or to existing producers. The immutable, serialisable history is a natural seam for durable persistence and event sourcing; the replay engine already supports projection rebuilds. Should a future phase require cross-process distribution, the bus can be fronted by an adapter that bridges to an external transport while preserving this exact API — the in-process core remains the deterministic reference implementation. Producers in Weeks 1–10 adopt the bus purely additively: each calls `publish` at its existing lifecycle points, with no modification to its own public API.

---

## 14. Verification

The suite `tests/test_enterprise_event_bus.py` is standard pytest and contains **359 collected test cases, all passing**, covering: every enum; serialization round-trips for all ten dataclasses; topic/wildcard matching; the full filter matrix; all six routing modes; deterministic priority and registration ordering; one-time subscriptions; failure handling, retries, dead-lettering, and recovery; batch publishing; all five replay modes (including replay-to-subscribers and the no-history-growth guarantee); statistics and rates; determinism (identical histories and delivery order across runs); thread safety (concurrent publish and subscribe, unique sequence guarantee); large streams and memory-bounding via `max_history`/`max_dead_letters`; re-entrant publishing; frozen/slots guarantees; JSON compatibility; the CLI; and non-invasiveness checks asserting the module imports nothing from the platform or any forbidden broker. Run the demonstration with `python src/events/enterprise_event_bus.py --demo`.