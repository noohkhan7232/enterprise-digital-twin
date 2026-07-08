# Interview Cheat Sheet

A quick-reference for discussing the Enterprise Digital Twin & Decision Intelligence Platform in
technical interviews. Each item pairs a likely question with a concise, defensible answer.

---

## The 30-Second Pitch

"It's an integrated platform that hosts several industrial-AI capabilities — digital twins,
predictive maintenance, retrieval-augmented knowledge, agents and workflows — behind one
governance, deployment and observability fabric. The interesting problem wasn't any single
algorithm; it was coherence at scale: operating many capabilities together without integration
debt, lost provenance or fragmented monitoring. I solved it with a layered, composition-based
architecture built additively and verified by 1,503 deterministic tests."

## Architecture

- **Ten layers, one dependency direction.** Capability layers (twin, predictive, agentic, RAG,
  workflow) on top of production-engineering subsystems (MLOps, monitoring, CI/CD, deployment,
  observability).
- **Integration by composition.** Subsystems exchange immutable value objects; no shared mutable
  state, no reaching into internals. Acyclic dependency graph.
- **Why it matters.** Cohesive *and* loosely coupled — each subsystem is internally complete, the
  only cross-boundary contract is a serialisable value.

## Design Principles (and why each was chosen)

| Principle | Why |
|-----------|-----|
| Dependency injection | Makes the system deterministic and testable offline (inject clocks/IDs) |
| Immutable domain models | Thread-safe sharing, well-defined equality, fewer state bugs |
| Determinism | Reproducible outputs; tests need no time/randomness mocking |
| Thread safety | Re-entrant locks + immutable snapshots; no partial reads |
| Composition over inheritance | Flat, explicit, independently evolvable subsystems |
| SOLID | Narrow responsibilities; extend via new types/strategies, not modification |

## Testing

- **1,503 tests, deterministic, framework-agnostic** — only assertions and parameterisation, no
  fixtures, no network.
- **Four levels:** value-object, engine/behavioural, edge-case, determinism.
- **Talking point:** "Because all non-determinism is injected, a determinism test can assert that
  repeated runs produce byte-identical serialised reports — that's how reproducibility is guarded
  against regression."

## Per-Subsystem One-Liners

- **MLOps:** versioned registry + artifact store + reproducibility + lineage = provenance by
  construction; every prediction traces to model, experiment, data, code and environment.
- **Monitoring:** data drift vs concept drift (inputs vs input–output relationship), prediction
  anomalies, data quality, composite health, observer-based alert fan-out.
- **CI/CD:** 20 quality gates + release validation + deployment readiness; honest gate failures are
  reported, not masked.
- **Deployment:** multi-stage non-root image, 10 K8s manifests, autoscaling 3–12, zero-downtime
  rolling updates, health-gated rollback; one deterministic health check shared everywhere.
- **Observability:** metrics (P50/P95/P99, windows, trends), tracing (timeline + critical path),
  structured logging, reliability (availability/MTBF/MTTR), SLI/SLO + error budgets + burn rate,
  incident lifecycle + postmortems, capacity forecasting, ten-area readiness score.

## Likely Hard Questions

- **"How do you avoid a big ball of mud across ten layers?"** Strict dependency direction,
  composition only, immutable value-object contracts, and additive construction enforced by CI and
  the test suite.
- **"How is an agent's behaviour kept safe?"** Typed tool interfaces bound the effects; reasoning
  trajectories are recorded; actions are traceable through observability like any service request.
- **"What about reproducibility for RAG?"** The corpus index is a versioned artefact and retrieved
  evidence is carried with answers, so the evidence set at a point in time is reconstructable.
- **"Did you benchmark it?"** I report verified engineering properties and configured SLO targets,
  not fabricated latency/throughput. Live benchmarking under sustained load is identified as future
  work — I was deliberate about not inventing numbers.
- **"What would you change?"** Bridge the self-contained observability to external ecosystems; study
  composed behaviour under real workloads; formalise the cross-capability data contracts.

## Numbers to Remember

10 layers · 30 modules · ~10,620 LOC · 1,503 tests · 10 K8s manifests · 3 workflows · SLOs:
availability ≥ 0.99, P95 ≤ 250 ms, error rate ≤ 0.01, freshness ≤ 300 s.

## Honesty Notes (say these proactively — they read as senior)

- Capability-layer internals are pluggable; I evaluated architecture and the production-engineering
  substrate, not model accuracy.
- The readiness score is a transparent self-assessment, not an external audit.
- All statistics are measured from the repository; none are estimated.