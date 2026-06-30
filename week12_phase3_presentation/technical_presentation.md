# Technical Presentation

**Enterprise Digital Twin & Decision Intelligence Platform**
20 slides for an engineering audience. Each slide lists its title, on-slide content, and a speaker
cue. Pair with `presentation_notes.md`.

---

### Slide 1 — Title
- **Content:** Platform name; "An integrated architecture for industrial AI, engineered for
  production."
- **Cue:** Set expectation: depth on architecture and the production-engineering substrate.

### Slide 2 — Thesis
- **Content:** "The dominant difficulty is systemic, not algorithmic."
- **Cue:** We optimise the system, not a single metric.

### Slide 3 — Architecture Overview
- **Content:** Architecture figure; ten layers; single dependency direction.
- **Cue:** Five capability layers over five production-engineering subsystems.

### Slide 4 — Core Invariant: Integration by Composition
- **Content:** "Immutable value objects cross boundaries; no shared mutable state; acyclic graph."
- **Cue:** This is what keeps a ten-layer system maintainable.

### Slide 5 — Engineering Decisions
- **Content:** SOLID · Dependency Injection · Immutable models · Determinism · Thread safety ·
  Composition over inheritance.
- **Cue:** Each decision earns its place; explain the *why* on the next slides.

### Slide 6 — Determinism by Construction
- **Content:** "Inject time and identity; order and round outputs; assert byte-identical reports."
- **Cue:** This is why the suite needs no time/randomness mocking.

### Slide 7 — Immutable Domain Models
- **Content:** "Frozen, slotted, self-serialising value objects; lossless JSON round-trip."
- **Cue:** Thread-safe sharing, well-defined equality, fewer state bugs.

### Slide 8 — Digital Twin & Predictive Layers
- **Content:** Asset + synchronised state contracts; injectable forecasting strategies.
- **Cue:** Pluggable behind contracts; provenance bound via MLOps.

### Slide 9 — Agentic & Knowledge Layers
- **Content:** Typed tool actions + recorded trajectories; evidence-grounded retrieval over a
  versioned index.
- **Cue:** Autonomy stays auditable; retrieval stays reproducible.

### Slide 10 — Workflow Engine
- **Content:** Composed steps; explicit, deterministic process state.
- **Cue:** Orchestration in one auditable place; mirrors the incident-lifecycle design.

### Slide 11 — MLOps & Provenance
- **Content:** Versioned registry · artifact store · reproducibility · lineage graph.
- **Cue:** "How was this produced?" is a graph traversal, not a log search.

### Slide 12 — Monitoring
- **Content:** Data drift vs concept drift · prediction anomalies · data quality · composite health ·
  observer-based alerts.
- **Cue:** Signals are contracts consumed by reliability/observability.

### Slide 13 — CI/CD
- **Content:** Shared validation · 20 quality gates · release validation · deployment readiness · 3
  workflows.
- **Cue:** Quality enforced at the boundary; honest gate failures reported.

### Slide 14 — Deployment
- **Content:** Multi-stage non-root image · 10 K8s manifests · rolling updates · autoscaling ·
  network policy · PDB · PVC.
- **Cue:** One deterministic health check across container, probes and scripts.

### Slide 15 — Rollback & High Availability
- **Content:** Health-gated rollback; replicas, topology spread, disruption budget, durable storage.
- **Cue:** Zero-downtime; recovery is declarative.

### Slide 16 — Observability
- **Content:** Metrics (P50/P95/P99, windows, trends) · tracing (timeline, critical path) ·
  structured logging · reliability · SLO/error budgets · incidents · capacity.
- **Cue:** Operational questions become numbers.

### Slide 17 — SLOs & Reliability
- **Content:** Default targets: availability ≥ 0.99; P95 ≤ 250 ms; error rate ≤ 0.01; freshness
  ≤ 300 s. Error budgets and burn rate.
- **Cue:** "Configured targets evaluated at runtime — not benchmark results."

### Slide 18 — Testing & Verification
- **Content:** "1,503 deterministic, framework-agnostic tests; value-object / engine / edge-case /
  determinism levels."
- **Cue:** Failures localise; full run re-verifies all subsystems.

### Slide 19 — Lessons Learned
- **Content:** Composition keeps large systems sane; determinism makes testing cheap; immutable
  contracts decouple subsystems; honest signals beat masked ones; additive discipline preserves
  backward compatibility.
- **Cue:** Speak candidly; senior engineers value the reflection.

### Slide 20 — Close & Q&A
- **Content:** "Integration done with discipline." Repository placeholder.
- **Cue:** Invite hard questions; reference the research paper for depth.

---

## Delivery Notes
- Target 20–25 minutes; interleave the live observability demo around slides 16–18.
- Keep code off the slides; demonstrate it live or via recordings.
- Use slide 17's honesty note proactively.