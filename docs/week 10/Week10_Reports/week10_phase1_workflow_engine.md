# Week 10 · Phase 1 — Enterprise Workflow Engine

**Component:** `src/workflow/workflow_engine.py`
**Status:** Additive. No Weeks 1–9 file is modified, and no existing public API is referenced, imported, or duplicated.
**Dependencies:** Pure Python (standard library) and NumPy. No external workflow engines (Airflow, Temporal, Prefect, Celery), no cloud services, no network access.

---

## 1. Purpose and scope

The Enterprise Workflow Engine turns executive recommendations into executable, auditable operational workflows. It is the operational-execution layer that sits beneath the platform's decision-making components: where the Executive RAG Copilot, Executive Intelligence Agent, Decision Copilot, Scenario Planning, and Knowledge Agent decide *what* should happen, this engine governs *how* it happens — sequencing the work, gating it with deterministic rules and approval checkpoints, dispatching side-effecting actions, and recording an immutable audit trail of every transition.

The engine is deliberately small in surface area and large in guarantees. It provides ten workflow archetypes, a validated seven-state lifecycle, conditional and parallel execution, retry and timeout handling, approval pausing/resumption, cancellation, a configurable rule engine, a pluggable action engine, a definition registry, and aggregate statistics — all of it deterministic and reproducible.

---

## 2. Integration model — seams, not couplings

A central design constraint is that Weeks 1–9 are frozen. The engine therefore integrates through **structural contracts** rather than imports.

- **`RecommendationSource` protocol.** Any upstream object that can describe itself as a JSON-like mapping (`as_recommendation() -> Mapping`) is a valid source. The Executive RAG Copilot, Decision Copilot, Scenario Planning, and the various agents all emit recommendation-shaped data; each can be adapted to this protocol with a thin shim in the wiring layer, with no change to their code.
- **`RecommendationCompiler`.** A deterministic translator from a recommendation mapping to a validated `WorkflowDefinition`. The `kind → WorkflowType` map and the per-kind step builders are configurable via `register_builder`, so new recommendation shapes are added without editing the compiler.
- **Dependency injection everywhere.** `WorkflowEngine` receives its registry, rule engine, action engine, state machine, compiler, clock factory, and default actor by constructor injection. Real side effects (creating a maintenance ticket, reserving inventory in the ERP, paging an executive) are introduced by registering bespoke `ActionHandler`s — the engine itself never reaches into another module.

The result is zero import coupling to frozen code, which makes it structurally impossible for this component to break Weeks 1–9. The seams are the *only* places real systems are wired in, and they are explicit and documented.

```
Executive layer (frozen)                 Workflow engine (this module)
─────────────────────────                ─────────────────────────────
RAG Copilot / Agents  ──recommendation──▶ RecommendationCompiler
                                          │
                                          ▼
                                   WorkflowDefinition ──▶ WorkflowEngine.run()
                                          │                     │
                          rules (RuleEngine)         actions (ActionEngine)
                                          │                     │
                                          ▼                     ▼
                              immutable WorkflowExecution + WorkflowHistory
```

---

## 3. Architecture

The module is composed of small, independently testable collaborators that the engine orchestrates.

| Collaborator | Responsibility |
| --- | --- |
| `WorkflowStateMachine` | Validates the seven-state lifecycle; rejects illegal transitions. |
| `RuleEngine` | Registers and evaluates configurable, typed `Rule`s against a context. |
| `ActionEngine` | Dispatches `WorkflowAction`s to deterministic, injectable handlers. |
| `WorkflowRegistry` | Stores and retrieves `WorkflowDefinition`s by id. |
| `RecommendationCompiler` | Compiles recommendation mappings into definitions. |
| `WorkflowEngine` | Stateless orchestrator producing immutable execution snapshots. |

The domain model is a set of frozen, JSON-serialisable dataclasses: `WorkflowCondition`, `WorkflowAction`, `RetryPolicy`, `WorkflowStep`, `WorkflowDefinition`, `ActionOutcome`, `WorkflowResult`, `AuditEvent`, `WorkflowHistory`, `WorkflowExecution`, and `WorkflowStatistics`. Each implements symmetric `to_dict()` / `from_dict()`, so any object can be persisted to JSON and reconstructed losslessly.

### 3.1 The immutability decision

Every dataclass is `frozen=True`, and the engine never mutates an execution in place. Each public operation (`submit`, `run`, `approve`, `reject`, `pause`, `resume`, `cancel`) accepts a `WorkflowExecution` and returns a **new** one via `dataclasses.replace`. Execution progress is thus a sequence of immutable snapshots rather than a mutable object graph.

This choice pays for itself repeatedly: snapshots are trivially serialisable and cacheable; concurrent readers can never observe a half-updated state; "what changed" is always a diff between two values; and reproducibility is structural rather than aspirational. The minor cost — allocating new tuples on each transition — is negligible at the scale of operational workflows (tens to low hundreds of steps).

Free-form mappings (action parameters, step/definition metadata, evaluation context, action output, audit detail) are stored internally as **canonical JSON strings** (`json.dumps(..., sort_keys=True)`). This keeps every dataclass hashable and order-insensitive while remaining directly serialisable: two actions with the same parameters in different key order compare equal and hash identically.

---

## 4. Workflow lifecycle

States and the only legal transitions:

```
        ┌─────────────────────────────── cancel ───────────────────────────────┐
        │                                                                        ▼
  DRAFT ──▶ PENDING ──▶ RUNNING ──▶ COMPLETED                               CANCELLED
                            │  ▲                                                 ▲
                            │  └──── resume ────┐                                │
                            ▼                   │                                │
                          PAUSED ───────────────┘────── cancel / fail ──────────┘
                            │
                            └──▶ FAILED
```

`WorkflowStateMachine` encodes the allowed set for each state and raises `InvalidStateTransitionError` on anything else. `COMPLETED`, `FAILED`, and `CANCELLED` are terminal. The full 7×7 transition matrix is exhaustively tested.

A typical run: `create_execution` (→ `DRAFT`) → `run` auto-submits (→ `PENDING`) and starts (→ `RUNNING`) → the driver executes steps until it completes (→ `COMPLETED`), hits an approval checkpoint (→ `PAUSED`), or fails (→ `FAILED`). A paused execution resumes through `approve`/`resume` and continues from exactly where it stopped.

### 4.1 Execution semantics

The driver walks an ordered list of steps with an explicit cursor and an id→index jump table, which yields four control-flow behaviours from one simple loop:

- **Sequential** — the default; the cursor advances to the next step.
- **Conditional branching** — a step's `next_steps` redirect the cursor (a goto), and per-step guard `conditions` decide whether a step runs or is recorded as `SKIPPED`.
- **Failure compensation** — a failed step with `on_failure` targets redirects to a compensation path; an `optional` failed step is tolerated and execution continues; otherwise the workflow fails.
- **Parallel groups** — a maximal contiguous run of steps sharing a `parallel_group` is executed as a logical batch.

Successful step output is merged back into the evaluation context as `{step_id}_output`, so later guards and rules can react to earlier results. This makes data-dependent branching deterministic and inspectable.

### 4.2 Logical parallelism

Parallelism here is **logical, not threaded** — a deliberate choice given the "deterministic, no external engines" mandate. Steps in a group are independent (no ordering dependency between them), executed in a fixed order for reproducibility, and aggregated under the definition's `parallel_policy` (`ALL` — every child must succeed; `ANY` — at least one). To model concurrency in the timeline, the group advances logical time by the **maximum** child duration rather than the sum, so a group of a 3-second and a 5-second task costs 5 seconds, exactly as true concurrency would. Each child's `WorkflowResult` still reports its own duration.

This gives the analytical benefits of parallelism (correct wall-clock modelling, all-or-any policies) without nondeterministic thread scheduling.

---

## 5. Rule engine

Rules are deterministic, typed, and configurable. A `Rule` pairs a `RuleType` — `RISK_THRESHOLD`, `ASSET_CRITICALITY`, `BUDGET_THRESHOLD`, `SCENARIO_SCORE`, `KNOWLEDGE_CONFIDENCE`, `EXECUTIVE_CONFIDENCE` — with a `WorkflowCondition` over a named context field. `WorkflowCondition` supports `GT`, `GTE`, `LT`, `LTE`, `EQ`, `NEQ`, `IN`, `NOT_IN`, `BETWEEN`, and `EXISTS`.

Evaluation is intentionally lenient and crash-free: a missing field evaluates to `False` (except `EXISTS`, which reports presence), and incompatible operand types are treated as a non-match rather than raising. This keeps a single rule misconfiguration from aborting an execution while remaining fully deterministic.

`RuleEngine` offers `register`/`get`/`remove`/`list_rules` plus `evaluate`, `evaluate_all`, `matching_rules`, and `matching_by_type`. Listing is sorted for stable, reproducible output.

---

## 6. Action engine

`ActionEngine` maps each `ActionType` to an injected `ActionHandler` with the signature `(action, context, clock) -> ActionOutcome`. The seven required action types — Create Maintenance Task, Schedule Inspection, Generate Executive Alert, Reserve Inventory, Escalate Approval, Create Knowledge Review, Generate Audit Record — plus an explicit `NO_OP` are provided with **default, side-effect-free handlers**.

The default handlers are pure record-producers: they synthesise a deterministic output record (with content-addressed ids derived via SHA-256 over canonical inputs) describing the operation that a downstream system *would* perform. They never touch another module. Real side effects are introduced by registering production handlers through `register_handler`, which is also how tests inject failing, slow, or timing-controlled handlers. Determinism is guaranteed: identical inputs always yield identical output records.

---

## 7. Audit architecture

Every operation appends an immutable `AuditEvent` to the execution's `WorkflowHistory`. Each event carries the required fields — `timestamp`, `workflow_id`, `step`, `event`, `actor`, `result` — plus an `execution_id`, a structured `detail`, and a monotonically increasing `sequence`.

The `sequence` field is the ordering authority. Because logical time may not advance between events (see §8), timestamps can tie; `sequence` guarantees a total, gap-free order regardless. `WorkflowHistory` is append-only — `append` returns a new history — and offers `filter_by_event`, `filter_by_step`, `timeline`, and `next_sequence`. The lifecycle emits a complete event vocabulary: `CREATED`, `SUBMITTED`, `STARTED`, `STEP_STARTED`, `STEP_COMPLETED`/`STEP_FAILED`/`STEP_TIMEOUT`/`STEP_SKIPPED`, `RETRY_BACKOFF`, `STEP_ATTEMPT_FAILED`, `APPROVAL_REQUIRED`/`APPROVAL_GRANTED`/`APPROVAL_REJECTED`, `PAUSED`, `RESUMED`, `COMPLETED`, `FAILED`, and `CANCELLED`.

---

## 8. Determinism and the injectable clock

The engine never calls `time.time()` in its core logic. All timestamps and backoff calculations flow through an injected `Clock`:

- **`LogicalClock`** (default) — starts at zero and advances *only* when the engine explicitly advances it, for a step's reported duration or a retry's backoff. This makes executions byte-identical across runs: the demo verifies that two independent runs produce an identical `to_dict()`, and the suite asserts the same property across all ten workflow archetypes.
- **`FixedClock`** — never advances; useful for boundary tests.
- **`SystemClock`** — wall-clock time for production telemetry, where reproducibility is not required.

Retry backoff is *logical*: `delay_for_attempt` advances the logical clock (or, inside a parallel group, accrues to the step's own elapsed time) but never calls `sleep`. Tests run instantly and deterministically, and production behaviour is configurable by choosing the clock.

---

## 9. Statistics

`WorkflowStatistics.from_executions` aggregates a collection of executions into workflow count, completed/failed/cancelled/running/paused counts, average duration (computed with NumPy), total retries, approval count, and success rate (completed ÷ finished). The result is a frozen, serialisable dataclass. NumPy is used where it is the natural tool — vectorised mean over a duration array — keeping the dependency meaningful rather than decorative.

---

## 10. Design decisions (summary)

| Decision | Rationale |
| --- | --- |
| Integration via protocols + DI, no imports of frozen code | Structurally guarantees Weeks 1–9 cannot be broken. |
| Frozen dataclasses + snapshot execution model | Reproducibility, safe sharing, trivial serialisation, diffable state. |
| Canonical-JSON storage for free-form mappings | Frozen + hashable + order-insensitive + directly serialisable. |
| Injectable clock; logical time by default | Byte-identical executions; instant, deterministic tests. |
| Logical (ordered) parallelism, max-duration cost | Concurrency semantics without nondeterministic scheduling. |
| Side-effect-free default action handlers | Safe by default; real effects are opt-in via injection. |
| Monotonic audit `sequence` independent of timestamp | Total ordering even when logical time ties. |
| Lenient condition evaluation | One misconfigured rule cannot abort an execution. |

---

## 11. Performance characteristics

Execution cost is linear in the number of steps; each transition allocates new tuples for results and history (O(n) in the count so far). For operational workflows — typically tens of steps, occasionally low hundreds — this is immaterial; the suite exercises a 100-step workflow and a 200-definition registry without issue. There is no I/O, no locking, and no background thread in the core, so latency is dominated by the injected action handlers. When production handlers perform real I/O, that work is isolated behind the `ActionHandler` boundary and can be made asynchronous in the wiring layer without changing the engine.

---

## 12. Enterprise use cases

- **Predictive maintenance.** A risk-scored recommendation compiles to reserve-parts → create-task → audit, gated by risk and criticality rules.
- **Emergency maintenance.** Alert → executive approval checkpoint → emergency task with a multi-attempt retry policy; pauses for sign-off and resumes on approval.
- **Shutdown planning & executive approval.** Approval-first workflows that hold at a checkpoint until a named actor (recorded in the audit trail) approves.
- **Inventory procurement.** Reserve → budget approval gated by a `BUDGET_THRESHOLD` condition, so low-value requests skip escalation automatically.
- **Inspection, risk mitigation, safety response, knowledge review, scenario evaluation.** Each maps to a deterministic step template, extensible via `register_builder`.

---

## 13. Future integration with BPM systems

The engine is intentionally aligned with established Business Process Management concepts, which makes future interoperability a mapping exercise rather than a rewrite:

- **BPMN export/import.** `WorkflowDefinition` maps cleanly to BPMN tasks (steps), gateways (guard conditions and `next_steps`), parallel gateways (`parallel_group` with `ALL`/`ANY`), and user tasks (`requires_approval`). A serializer to BPMN 2.0 XML can be added as a separate module against the existing `to_dict()` shape.
- **External orchestrators.** Because the engine is a pure function over immutable snapshots, it can run *inside* an external scheduler (one snapshot per scheduler tick) or *delegate* long-running actions to one via injected handlers — without the engine itself depending on that system.
- **Event streaming.** `WorkflowHistory` is already an ordered event log; a handler can mirror each `AuditEvent` to Kafka or an event store for downstream analytics and compliance.
- **Durable persistence.** Snapshots serialise to JSON today; a persistence adapter can checkpoint executions to a database for crash recovery, again without engine changes.

---

## 14. Verification

The accompanying suite (`tests/test_workflow_engine.py`) is standard pytest and contains **499 collected test cases**, all passing, covering: workflow lifecycle and the full state-transition matrix; the rule engine across every rule type and operator; the action engine across every action type; execution history and audit ordering; serialization round-trips for every dataclass; the registry; retry logic; conditional and parallel workflows; timeouts; cancellation; statistics; determinism (identical replay across all ten archetypes); the CLI; immutability/frozen guarantees; large-scale workflows and registries; and backward-compatibility guards that assert the module imports no forbidden dependency and no frozen package.

Run the demonstration:

```bash
python src/workflow/workflow_engine.py --demo
```

It compiles an emergency-maintenance recommendation, runs it to an approval checkpoint, approves it as `cfo`, prints the step results, audit timeline, and aggregate statistics, and verifies that a second independent run is byte-for-byte identical.