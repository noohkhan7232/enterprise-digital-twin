# Glossary
## Enterprise Digital Twin & Decision Intelligence Platform

Authoritative definitions of the domain and engineering terms used in the Week 12 research
paper (`research_paper.md`) and its appendices. Terms are listed alphabetically.

---

**Agentic AI.** An approach in which an autonomous system perceives its environment,
deliberates over possible actions, and acts through defined interfaces — often using external
tools — to accomplish multi-step tasks. In this platform, agent actions are typed and
traceable so that autonomous behaviour remains observable and governable.

**Alert engine.** A component that consumes monitoring signals and emits alerts according to
configurable severity and routing, decoupled from signal producers so that downstream
reactions can be added without modifying detectors.

**Artifact store.** A storage component that holds build artifacts against content-addressed
identifiers, so that an artifact reference is stable and identical content is not duplicated.

**Burn rate.** The rate at which an error budget is consumed, expressed relative to the
fraction of the objective's window that has elapsed; a value above one indicates the budget
is being spent faster than the window can sustain.

**Capacity planning.** The forecasting of future resource demand (e.g., CPU, memory,
storage, request volume, data growth) to recommend provisioned capacity and to detect
projected exhaustion.

**Composition over inheritance.** A design preference for assembling behaviour by combining
independent components rather than by deriving subclasses, yielding flatter, more explicit
and more loosely coupled designs.

**Concept drift.** A change over time in the relationship between model inputs and outcomes,
as distinct from a change in the inputs alone; it degrades model validity even when input
distributions appear stable.

**Cyber-physical system (CPS).** A system in which computational elements monitor and
control physical processes, forming the basis for digital twins and Industry 4.0 manufacturing.

**Data drift.** A change over time in the distribution of model inputs relative to a reference
baseline, which can degrade model performance.

**Decision intelligence.** A discipline that frames analytics in terms of the decisions and
outcomes they inform, connecting data, models, actions and consequences rather than treating
predictions as endpoints.

**Dependency injection (DI).** A technique in which a component receives its dependencies
(such as clocks, identifier sources or strategies) from outside rather than constructing them
internally, enabling testability and deterministic behaviour.

**Deterministic computation.** Computation whose output is a function of its explicit inputs
alone, with all sources of time and randomness injected, so that repeated execution yields
identical results.

**Digital twin.** A virtual representation of a physical asset, synchronised with the asset's
state through telemetry, used for monitoring, simulation and decision support. Categorical
treatments distinguish digital models, digital shadows and digital twins by their degree of
automated data flow.

**Error budget.** The permissible amount of unreliability for a service-level objective over
its window (for example, the allowed shortfall below an availability target); its consumption
governs release and operational decisions.

**Immutable domain model.** A data type whose instances cannot be modified after creation;
in this platform such types are frozen, slotted and self-serialising, making them safe to share
across threads and giving well-defined equality.

**Incident lifecycle.** The sequence of states an incident passes through (for example, open,
investigating, identified, monitoring, resolved, closed), with validated transitions and a
recorded timeline supporting postmortems and recovery-time computation.

**Lineage graph.** A record of provenance linking runs, datasets, artifacts and models, so
that any result can be traced to the inputs and process that produced it.

**Machine Learning Operations (MLOps).** The set of practices and tooling for managing the
machine-learning lifecycle in production, including experiment tracking, model registry,
artifact management, reproducibility and governance.

**Mean time between failures (MTBF).** The average operating time between successive
failures over an observation period; a reliability indicator.

**Mean time to recovery (MTTR).** The average time taken to recover from failures; a
reliability indicator derived from recorded outage windows.

**Model card.** Structured documentation describing a model's intent, characteristics and
risk profile, supporting governance and responsible use.

**Model registry.** A versioned catalog of models that enforces semantic versioning and
manages promotion between lifecycle stages, providing an authoritative record of which model
version exists and in what state.

**Observability.** The capability to understand a system's internal state from its external
outputs, realised here through metrics, tracing, structured logging, reliability metrics,
service-level objectives, incident management and capacity planning.

**Predictive maintenance.** The use of data-driven prognostics to anticipate equipment
failure and schedule maintenance before failure occurs, often expressed via remaining useful
life estimation.

**Production readiness.** The degree to which a system is fit to operate in production,
assessed in this platform across ten areas (architecture, security, reliability, monitoring,
deployment, CI/CD, testing, documentation, MLOps and observability) with a weighted score.

**Prognostics and health management (PHM).** The engineering discipline concerned with
assessing equipment health, predicting failures and managing maintenance decisions.

**Reproducibility.** The property that a result can be reconstructed from first principles, here
achieved by binding runs to source revision and environment snapshot and supporting
deterministic re-execution.

**Retrieval-augmented generation (RAG).** A technique that grounds language-model
generation in evidence retrieved from an external corpus, improving factuality and provenance.

**Service-level indicator (SLI).** A quantitative measure of a service aspect such as
availability, latency, error rate or freshness.

**Service-level objective (SLO).** A target value or range for a service-level indicator over a
defined window, against which compliance and error budgets are evaluated.

**Site reliability engineering (SRE).** A discipline that applies software-engineering practice
to operations, emphasising service-level objectives, error budgets and systematic reliability.

**SOLID.** Five object-oriented design principles — single responsibility, open/closed, Liskov
substitution, interface segregation and dependency inversion — that promote maintainable,
extensible software.

**Thread safety.** The property that a component behaves correctly under concurrent access,
achieved here by guarding mutable state with re-entrant locks and exposing immutable
snapshots.

**Trace and span.** A trace records the path of a unit of work across a system; it is composed
of spans, each representing a timed operation with parent/child relationships, enabling timeline
reconstruction and critical-path analysis.

**Workflow engine.** A component that composes individual steps into governed, multi-step
processes with explicit, recoverable state.