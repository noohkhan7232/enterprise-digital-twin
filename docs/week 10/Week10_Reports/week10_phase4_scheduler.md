# Week 10 · Phase 4 — Enterprise Scheduler & Automation Engine

**Component:** `src/scheduler/enterprise_scheduler.py`
**Status:** New package. Composes the platform by injection and emits to the Phase 3 Event Bus; imports no upstream module and modifies none. No prior file is touched and no public API is changed.
**Dependencies:** Pure Python (standard library) + NumPy. No APScheduler, Celery Beat, cron libraries, or asyncio.

---

## 1. Business motivation

Most enterprise value from a digital twin comes not from one-off requests but from *standing automation*: refresh the predictive models every night, run a risk assessment when an asset health signal degrades, generate the executive brief on the first business day of the month, and re-plan maintenance whenever a workflow completes. Those are time-based, condition-based, and event-based triggers respectively, and they need to compose with retries, timeouts, priorities, dependencies, and a business calendar.

A general-purpose scheduler like cron or APScheduler cannot serve a *deterministic research platform*: real timers and background threads make runs non-reproducible and untestable, and they pull in process/broker dependencies the platform forbids. This Phase 4 deliverable is a purpose-built, **deterministic, tick-driven** scheduler. It holds no background thread; time is supplied by an injectable clock and advanced explicitly, so a year of automation can be simulated in microseconds and asserted exactly. It automates jobs across the platform by composition — executing them through injected executors and announcing every lifecycle change on the Enterprise Event Bus.

---

## 2. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            EnterpriseScheduler              │
                    │  register/remove/pause/resume · run_now     │
                    │  tick / advance_to · fire_event · evaluate  │
                    └─────────────────────────────────────────────┘
                       │          │            │           │
        ┌──────────────┘          │            │           └──────────────┐
        ▼                         ▼            ▼                          ▼
  Job registry            Trigger engine   Automation engine        Executor registry
  (jobs, next_exec,       (ScheduleTrigger (event / condition /     (JobType → JobExecutor;
   pause set, queues)      next_after)      chained rules)           default = record exec.)
        │                         │            │                          │
        └─────────────┬──────────┴────────────┴────────────┬─────────────┘
                      ▼                                      ▼
              Dependency DAG                          immutable JobHistory
              (cycle detect, order)                   + ScheduleStatistics
                      │
                      ▼
              _Emitter ──▶ (optional) Enterprise Event Bus  [Phase 3, frozen]
                                 publishes scheduler.* topics
```

Every collaborator is injected (clock, calendar, event bus, default executor, history cap). The domain model is ten `frozen=True, slots=True` dataclasses with symmetric `to_dict`/`from_dict`. The Registry pattern holds jobs, executors, and automation rules; the Observer pattern delivers lifecycle events to the bus; the Strategy pattern selects per-job execution behaviour via `ExecutionPolicy`.

---

## 3. Scheduling engine

`ScheduleTrigger` answers one deterministic question — *given a reference time, when does this job fire next?* — for ten trigger types: one-time, fixed interval, daily, weekly, monthly, a minimal five-field cron, plus event, condition, manual, and emergency. Time-based computations are closed-form (interval arithmetic, day/week/month boundary maths in a fixed-offset timezone) except cron, which scans minute-by-minute from the rounded reference. The scheduler keeps a `next_exec` time per job and advances state only when the host calls `tick(now)` (fire everything due at or before `now`) or `advance_to(target)` (replay the timeline, firing due jobs in order and rescheduling recurring ones). This explicit, tick-driven model is what makes the engine fully reproducible.

---

## 4. Automation engine

Three automation styles sit on top of the trigger engine. **Time-based** automation is the trigger engine itself. **Event-driven** automation runs jobs whose `EVENT` trigger names a fired event, plus any `EVENT_DRIVEN` `AutomationRule` whose `trigger_ref` matches — this is the bridge from the Event Bus into the scheduler. **Condition-based** automation runs `CONDITION` jobs and `CONDITION_BASED` rules when an evaluated context satisfies them. **Chained** automation runs a job's declared `children` (and any `CHAINED` rule keyed on its id) after it succeeds, enabling pipelines like *predict → assess risk → brief executive*. Retries, timeouts, priorities, and maintenance windows are all honoured during execution.

---

## 5. Calendar rules

`CalendarRule` is a timezone-aware (fixed-offset, no DST) gate over business hours, working days, holidays, maintenance windows, blackout periods, and an emergency mode that bypasses hour/day limits but never an active blackout. `ExecutionWindow` constrains a job to a daily local time window on selected weekdays. Both expose `is_allowed`/`is_within` and a `next_allowed` advance so the scheduler can defer a firing to the next permissible moment. A permissive 24/7 calendar is the default, so scheduling is well-defined without configuration.

---

## 6. Execution policies

`ExecutionPolicy` is the Strategy that decides how a firing behaves: `RETRY` re-attempts up to `max_retries` with logical backoff; `RUN_ONCE` and `CANCEL` stop recurrence after the firing; `SKIP`, `QUEUE`, `REPLACE`, and `IGNORE` execute a single attempt without retrying. Timeouts are deterministic — an executor reports a duration, and exceeding the policy timeout yields a `TIMEOUT` status. Every firing produces an immutable `JobExecution` (attempts, status, latency, duration, output) appended to the history.

---

## 7. Dependency model

Jobs may declare `depends_on` predecessors and `children` successors. A due job runs only when every dependency has a recorded success; otherwise it is reported `BLOCKED` and retried on a later tick. Within a tick, due jobs are ordered by `(next firing time, priority descending, job id)`. Dependencies form a DAG validated at registration by a from-scratch Kahn topological pass that raises `DependencyCycleError` on a cycle — no graph library. `blocking_jobs()` ranks the most-depended-upon jobs, the ones whose delay most threatens downstream automation.

---

## 8. Integration with the Event Bus

The scheduler announces its lifecycle on the Phase 3 Enterprise Event Bus by composition. An injected bus is wrapped in a small `_Emitter` that records every event locally (so emissions are inspectable even without a bus) and, when a bus is present, publishes an `EnterpriseEvent` on a `scheduler.<name>` topic for: job registered, job started, job completed, job failed, job cancelled, schedule updated, and automation triggered. The events module is imported defensively and optionally — if it is absent, the scheduler still runs and records events internally. The bus is never modified; the scheduler simply calls its public `publish`.

Execution itself is also composition: a `JobExecutor` is registered per `JobType`. The default executor is a pure record producer, so the scheduler is self-contained and testable; real integrations inject executors that call `WorkflowEngine.run`, `BusinessProcessOrchestrator.execute`, or the Executive Copilot — none of which the scheduler imports.

---

## 9. Performance and complexity

A `tick` is `O(J)` to find due jobs plus `O(D log D)` to order the `D` that are due; `advance_to` fires due jobs one at a time in timeline order with a guarded iteration bound. Trigger `next_after` is closed-form `O(1)` for all types except cron, which is bounded minute-scanning that resolves within a day for typical expressions. History append is amortised `O(1)` with optional ring-buffer capping for memory safety. There is no I/O, no thread, and no real sleep on the hot path. The suite exercises 100 concurrent registrations, 20 concurrent `run_now` calls, a 100-job fan-out, and a 201-occurrence recurring horizon.

---

## 10. Design decisions

| Decision | Rationale |
| --- | --- |
| Tick-driven, no background thread or real timer | Determinism and testability; a year of schedule replays instantly and identically. |
| Injectable clock (logical default) | Byte-identical histories under test; wall-clock available for production. |
| Executors injected per job type | Composition with Workflow/Orchestrator/Copilot without importing or modifying them. |
| Optional, defensive event-bus import | Lifecycle observability via Phase 3 without a hard dependency. |
| Frozen + slotted dataclasses | Immutability, low memory, lossless JSON serialisation. |
| Dependency DAG with Kahn cycle detection | Correct ordering and early rejection of cyclic automation; no graph library. |
| Strategy-based execution policies | Clear, testable per-job behaviour for retry/once/cancel/skip semantics. |

---

## 11. Enterprise use cases

- **Nightly model refresh.** A `daily` `PREDICTION_REFRESH` job, gated to off-hours by a calendar, retried on transient failure.
- **Reactive risk response.** A `risk_spike` event fires an `EXECUTIVE_REPORT` job and, via a chained rule, a `RISK_ASSESSMENT` — straight off the Event Bus.
- **Condition-driven maintenance.** When monitoring publishes `degraded=true`, a `MAINTENANCE_PLANNING` job runs and chains a workflow execution.
- **Monthly governance.** A `monthly` `EXECUTIVE_REPORT` on the first business day, with an SLA-style priority over routine jobs.
- **Pipelines.** Parent → children chaining expresses multi-stage automation deterministically, with dependency gating ensuring correct order.

---

## 12. Future integration

Because executors and the event bus are injected, future Dashboard, REST, Monitoring, and MLOps components attach without changing the scheduler: a REST layer calls `register_job`/`run_now`; a monitoring layer subscribes to `scheduler.*` on the bus; an MLOps layer registers `PREDICTION_REFRESH` executors that invoke training pipelines. The immutable, serialisable `JobHistory` and `ScheduleStatistics` are ready for durable persistence and dashboards. Should distribution ever be required, the deterministic tick model can be driven by an external coordinator while the core remains the reference implementation. Every existing module adopts the scheduler purely additively — by registering jobs and executors — with no change to its own public API, preserving full compatibility with Weeks 1–10 Phase 3.

---

## 13. Verification

The suite `tests/test_enterprise_scheduler.py` is standard pytest and contains **258 collected test cases, all passing**, covering: every enum; `next_after` for all ten trigger types (interval, daily, weekly, monthly, cron, one-time, emergency, and the non-time triggers); cron field parsing; calendar and execution-window gating; serialization round-trips for all dataclasses; policy and job validation; the full scheduler API (register/remove/pause/resume/run_now/next_execution/tick/advance_to/statistics); dependency gating, ordering, cycle detection, and blocking-job ranking; event, condition, and chained automation plus automation rules; retry, timeout, backoff, and policy semantics; statistics; live integration with the Enterprise Event Bus (asserting `scheduler.*` events are published); determinism (identical histories, emitted events, and priority ordering across runs); thread safety (concurrent registration and execution); large job sets and long horizons; frozen/slots and JSON guarantees; the CLI; and non-invasiveness checks confirming no APScheduler/Celery/cron/asyncio imports and no upstream-module imports. Run the demonstration with `python src/scheduler/enterprise_scheduler.py --demo`.