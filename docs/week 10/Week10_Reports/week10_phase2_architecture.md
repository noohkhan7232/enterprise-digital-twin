# Week 10 · Phase 2 — Enterprise Business Process Orchestrator

**Component:** `src/orchestration/business_process_orchestrator.py`
**Status:** New package. Composes the Week 10 Phase 1 Workflow Engine through its public API; no previous file is modified, no public API is changed, no workflow logic is duplicated.
**Dependencies:** Pure Python (standard library) + NumPy. No Pandas, NetworkX, SciPy, or any external workflow engine.

---

## 1. Business motivation

Enterprises rarely run a single workflow in isolation. A real operational event — a turbine fault, a plant shutdown, a procurement cycle — triggers a *business process*: a coordinated set of workflows with ordering constraints, human approval gates, service-level commitments, working-hour rules, and a defined behaviour when something fails. Phase 1 gave the platform a deterministic engine that executes one workflow well. Phase 2 supplies the layer that enterprises actually buy tools like SAP BPM, Camunda, IBM Business Automation, or the Power Automate backend to provide: composition, dependency management, planning, approvals, SLAs, calendars, rollback, simulation, and analytics.

The orchestrator is a coordination layer. It decides *what runs, in what order, under which approvals, and what happens on failure*; the Workflow Engine remains the sole executor of individual workflows. This separation keeps each layer small, testable, and independently evolvable.

---

## 2. Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │        BusinessProcessOrchestrator (facade)    │
                    │  register · plan · execute · simulate ·        │
                    │  analytics · kpis · rollback/compensation      │
                    └───────────────────────────────────────────────┘
                       │        │        │         │          │
        ┌──────────────┘        │        │         │          └───────────────┐
        ▼                       ▼        ▼         ▼                          ▼
 ProcessRegistry        DependencyGraph  ExecutionPlanner  ApprovalEngine   SLAEngine
 (thread-safe,          (DAG, cycles,    (stages, critical (chains, escalate (status,
  versioned,            topo, critical    path, makespan,   delegate,        penalty,
  freezable)            path, blocking)   risk, calendar)   timeout, metrics) compliance)
        │                                          │              │              │
        │                                          ▼              ▼              ▼
        │                                   RollbackEngine   ProcessAnalytics / BusinessKPIs
        │                                   (undo, recovery,  (dashboard / API / records)
        │                                    isolation, comp.)
        ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  Week 10 Phase 1 Workflow Engine  (consumed by composition; never modified) │
 │  WorkflowEngine.create_execution / run / approve · WorkflowDefinition       │
 └───────────────────────────────────────────────────────────────────────────┘
```

Every collaborator is injected into the facade, so each can be unit-tested in isolation or replaced (Open/Closed). The domain model is a set of `frozen=True, slots=True` dataclasses with symmetric `to_dict` / `from_dict`, giving immutability, low memory footprint, and lossless JSON serialisation.

---

## 3. Integration with Week 10 Phase 1

The orchestrator imports the engine module and uses only its public surface: `WorkflowEngine`, `WorkflowDefinition`, `WorkflowExecution`, `WorkflowState`, `create_default_engine`, and `LogicalClock`. A `ProcessStep` carries a `workflow_ref`; a `WorkflowProvider` (default: the engine's own registry) resolves that ref to a `WorkflowDefinition`, which the orchestrator runs by calling `create_execution` → `run`, auto-clearing any internal approval pause via `approve`. The engine is never imported privately, monkey-patched, or duplicated; the test suite asserts the Phase 1 source bytes are unchanged across an execution. Because the dependency is one-directional and public-API-only, Phase 1 remains frozen and safe.

---

## 4. Execution lifecycle

A process moves through `ProcessState`: `DRAFT → PLANNED → RUNNING →` (`COMPLETED` | `FAILED` | `ROLLED_BACK` | `COMPENSATED` | `CANCELLED`), with `PAUSED` and `SIMULATED` as auxiliary states. `execute()` performs:

1. Resolve and validate the process; build the dependency graph; produce an `ExecutionPlan`.
2. Walk the plan's stages in order. `WAITING` stages record a wait; `APPROVAL` stages drive the relevant approval chains; `SEQUENTIAL`/`PARALLEL` stages run each step's workflow through the Phase 1 engine. Contingency (`ROLLBACK`/`COMPENSATION`) stages are skipped on the happy path.
3. Advance a logical clock by each work stage's modelled cost (the maximum member duration for a parallel stage — concurrency is modelled as max, not sum).
4. On a rejected approval or a failed workflow, stop and, if rollback is enabled, synthesise a `RollbackPlan` (and `CompensationPlan` when compensations exist), marking the result `ROLLED_BACK`.
5. Evaluate the SLA and return an immutable `ProcessExecutionResult` with per-step records, approval records, an audit `ProcessHistory`, and run-summary metrics.

The orchestrator never mutates its inputs; it returns new immutable snapshots, exactly as Phase 1 does.

---

## 5. Dependency graph

`DependencyGraph` is implemented from scratch — no graph library — and is deterministic because adjacency and ready-sets are ordered by the original step declaration index. It provides:

- **Topological sort & cycle detection** via Kahn's algorithm; a residual node count signals a cycle (`CycleError`).
- **Layered ordering** (layered Kahn) — each layer is a set of mutually independent steps, which is exactly the unit of safe parallelism the planner needs.
- **Critical path** — the longest weighted path, computed with a DP pass over the topological order with predecessor tracking for reconstruction.
- **Roots / leaves / ancestors / descendants**, **blocking nodes** (ranked by descendant count — the steps whose delay threatens the most downstream work), and **connectivity** (undirected components for disconnected-graph detection).

Complexity is `O(V + E)` for sort, layering, and critical path; reachability and components are `O(V·(V + E))` in the worst case, which is comfortable for enterprise processes (tens to low hundreds of steps) and is exercised at 120–150 nodes in the test suite.

---

## 6. Execution planner

`ExecutionPlanner.plan` converts a process into an `ExecutionPlan` of ordered `ExecutionGroup`s. For each topological layer it emits a `WAITING` + `APPROVAL` stage when any member needs sign-off, then a `PARALLEL` stage (when the layer has more than one independent step) or a `SEQUENTIAL` stage. It appends contingency `ROLLBACK` and `COMPENSATION` stages. The plan reports the topological order, the critical path and its duration, the **makespan** (sum of per-layer maxima — a concrete schedule length), the calendar-aware estimated completion time, the resource estimate (Σ step cost), the risk estimate (`1 − Π(1 − riskᵢ)`, the probability that at least one step is risky), and the parallelism ratio (`1 − work_stages / steps`). Planning is pure and deterministic; the suite asserts byte-identical plans across runs.

---

## 7. Approval architecture

Enterprise approvals follow an ascending authority chain — Engineer → Lead → Manager → Director → Vice President → Executive — enforced at construction (`ApprovalChain` rejects out-of-order roles). `ApprovalEngine` is stateless and returns immutable `ApprovalChainState` snapshots, mirroring Phase 1's snapshot model, so it is inherently thread-safe. It supports `approve`, `reject`, `delegate` (reassign approver, same rung), `escalate` (advance to higher authority), `timeout` (record then escalate), `history`, and aggregate `metrics` (counts, escalation percentage, mean approval latency). Latency is measured against an injected clock, keeping it deterministic.

---

## 8. Business calendar

`BusinessCalendar` is timezone-aware (fixed offset, no DST for determinism) and models business hours, working days, holidays, maintenance windows, blackout periods, and an emergency mode that bypasses hour/day restrictions (but never an active blackout). `is_business_time`, `next_business_time`, and `advance_business_seconds` let the planner translate a modelled duration into a realistic completion timestamp that only consumes working time. A permissive 24/7 `always_on()` calendar is the default so that timing is well-defined even when no calendar is supplied.

---

## 9. SLA design

`SLAConfig` declares the expected duration, a warning threshold, an escalation threshold, a per-second penalty, and a recovery cap. `SLAEngine.evaluate` returns an immutable `SLAReport` with status (`ON_TRACK` / `WARNING` / `VIOLATED` / `RECOVERED` / `NOT_APPLICABLE`), delay, remaining time, penalty (charged only on violation), a compliance ratio, an escalation flag, and a bounded recovery time. `compliance_pct` aggregates many reports for fleet-level reporting.

---

## 10. Rollback & compensation design

On failure the `RollbackEngine` produces a deterministic `RollbackPlan`: an **undo chain** (completed steps in reverse topological order), **isolated steps** (the failed step and everything downstream — failure isolation), a **recovery chain** (the failed step and descendants in forward order, for re-execution), and **checkpoints** (completed blocking nodes for checkpoint-restore). A `CompensationPlan` lists `(step, compensation_step)` pairs in reverse-topological order. Strategies are explicit (`SEQUENTIAL_UNDO`, `CHECKPOINT_RESTORE`, `COMPENSATE`, `NONE`). Plans are pure data, so they can be inspected, serialised, audited, or handed to an executor.

---

## 11. Simulation model

`simulate()` supports `DRY_RUN`, `SIMULATION`, `WHAT_IF`, `REPLAY`, and `RECOVERY` by projecting from estimates without touching real workflows (`LIVE` is routed to `execute`). It returns a `SimulationResult` with the plan, projected duration (makespan), a projected success probability (`Π(1 − riskᵢ)`), the projected SLA status, and per-step projections. `WHAT_IF` accepts per-step overrides (duration, risk, cost) applied to a derived process via `dataclasses.replace`, leaving the original untouched. Simulation is deterministic and serialisable.

---

## 12. Analytics & KPIs

`BusinessKPIs.from_results` computes the full enterprise scorecard: process success rate, average duration, failure rate, automation vs manual percentage, mean approval latency, escalation percentage, rollback percentage, SLA percentage, critical-path percentage, parallelism percentage, process utilisation, and a composite business-efficiency score. `ProcessAnalytics.from_results` adds state and SLA breakdowns and duration percentiles (NumPy), and exposes the same data through `as_dashboard()`, `as_api()`, `as_dataframe_records()`, and `to_dict()` — dashboard-, API-, and DataFrame-ready, with no plotting and no Pandas.

---

## 13. Performance, complexity, and trade-offs

Planning and graph algorithms are linear or low-polynomial in the process size; execution cost is dominated by the injected workflow runs, not the orchestration. The deliberate trade-offs are: (1) **logical, ordered parallelism** rather than threads — reproducibility over raw speed, with concurrency modelled as max-duration; (2) **immutable snapshots** — slightly more allocation in exchange for thread-safety, diffability, and trivial serialisation; (3) **layered scheduling** — simple and deterministic, yielding a valid makespan that is an upper bound on the critical-path lower bound (both are reported); (4) **modelled timing from estimates** — deterministic SLA evaluation independent of wall-clock noise. The registry is the only mutable component and is guarded by an `RLock`; all engines are stateless and therefore thread-safe by construction.

---

## 14. Extension points

New step templates, approval policies, SLA rules, rollback strategies, calendars, and workflow providers are all injected or subclassed without editing the core (Open/Closed). `WorkflowProvider` and `DecisionProvider` are `Protocol`s, so any conforming callable participates. Analytics gain new surfaces by adding methods to `ProcessAnalytics`. Because the domain model is fully serialisable, persistence and event-streaming adapters can be layered on without touching the engines.

---

## 15. Future compatibility with Week 10 Phase 3

Phase 2 is shaped to receive a Phase 3 runtime/governance layer. The immutable `ProcessExecutionResult` and `ProcessHistory` are a ready event log for durable persistence, replay, and distributed scheduling. `ExecutionPlan` is an explicit, serialisable schedule that an external runtime could execute stage-by-stage. The `Protocol`-based seams (`WorkflowProvider`, `DecisionProvider`) are where Phase 3 can inject live human-task inboxes, real ERP side effects, or a distributed clock — without modifying Phase 1 or Phase 2. The dependency graph, critical path, and KPI surfaces give a Phase 3 control tower the analytics it needs for SLA-driven prioritisation and adaptive rescheduling.

---

## 16. Verification

The suite `tests/test_business_process_orchestrator.py` is standard pytest and contains **316 collected test cases, all passing**, covering: the registry (including freeze/version and 50-thread concurrent registration); the dependency graph (topological sort, cycle detection, layering, critical path, blocking nodes, connectivity, 120–150 node scale); the execution planner; the approval engine (approve/reject/delegate/escalate/timeout, latency, metrics); the business calendar (hours, weekends, holidays, blackouts, emergency mode, timezone); the SLA engine; rollback and compensation; simulation (all modes, what-if overrides, determinism); analytics and all KPI surfaces; serialization round-trips for every dataclass; frozen/slots guarantees; JSON compatibility; determinism; invalid configurations; large and wide processes; and a non-invasiveness check asserting the Phase 1 source is byte-identical before and after execution.