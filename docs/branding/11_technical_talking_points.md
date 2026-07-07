# Top 30 Technical Talking Points

Discussion topics the candidate can drive in interviews, each anchored to repository evidence.

## Architecture & System Design (1–7)

1. **Immutable value objects as the only boundary currency** — why this one rule produces
   determinism, testability, and governance simultaneously.
2. **Single dominant dependency direction across ten layers** — what it prevents, what it costs,
   and where it was tempting to cheat.
3. **Additive construction** — twelve weekly increments where no layer modifies its predecessors;
   what that discipline does to design quality.
4. **Composition over inheritance at platform scale** — 23 packages that compose rather than
   entangle.
5. **Explicit coordination** — event bus, scheduler, workflow engine, orchestration: why implicit
   coordination through shared state was banned.
6. **Thread safety strategy** — 27 modules use threading; how immutability shrinks the locking
   surface.
7. **Fleet-scale trade-offs** — business-unit sharding and summarisation-on-the-fly (memory
   proportional to assets, not samples), from the week-6 fleet analysis.

## AI: ML, DL, RAG, Agents (8–13)

8. **Torch-optional deep-learning contracts** — checkpoint validation (config + state-dict)
   importable without a GPU stack, so registry/CI tooling runs anywhere.
9. **Model architecture portfolio** — CNN-BiLSTM with attention, acoustic transformer, anomaly
   autoencoder — and what each was for in the acoustic-monitoring lineage.
10. **Evidence-grounded RAG** — hybrid retrieval over a versioned corpus with fully serialisable
    retrieval state; why versioning the corpus matters more than the vector store choice.
11. **Agents with typed actions and recorded trajectories** — making an agent's action as
    traceable as a service request.
12. **Injectable prediction strategies** — forecasting/prognostics as swappable strategies with
    provenance binding, not hardcoded models.
13. **Audio-ML research foundations** — denoising benchmarks, feature engineering, separability
    heatmaps; how research rigour translated into platform rigour.

## Testing & Validation (14–18)

14. **51K lines of tests for 61K lines of source** — where the near-1:1 investment pays and where
    it's overkill.
15. **Byte-reproducible outputs → exact assertions** — determinism as a testing strategy.
16. **Testing the testers** — the CI quality gates, release validator, and readiness validator are
    themselves unit-tested (247 passing).
17. **Collected vs passing** — why the repo reports 8,361 tests *collected* and refuses to claim
    execution it hasn't run; verification honesty as an engineering value.
18. **The self-validator** — repository-level checks (syntax 174/174, type hints 97.2%, packages
    23/23) and its known findings (complexity max 36, 20 hotspots) left visible rather than hidden.

## Docker, Deployment & CI/CD (19–24)

19. **Multi-stage non-root image** — read-only rootfs, dropped capabilities, resource limits; each
    hardening choice mapped to a failure it prevents.
20. **One deterministic health check, four consumers** — Docker HEALTHCHECK, K8s liveness,
    readiness, and rollback gate; why a single definition of "healthy" matters.
21. **The ten-manifest Kubernetes surface** — what HPA, PodDisruptionBudget, and NetworkPolicy
    each protect against.
22. **Rollback as automation, not heroics** — health-gated rollback scripting.
23. **Policy-as-YAML governance** — a 20-gate quality engine where changing a threshold never
    means changing engine code.
24. **Three-workflow CI design** — CI, release, and dependency scanning; what gates what.

## Monitoring, Observability & Analytics (25–28)

25. **Data drift vs concept drift vs prediction quality** — three failure modes, three detectors,
    one composite health score with routed alerting.
26. **SLOs with error budgets and burn rates** — configured targets today, and the concrete path
    to measured SLIs in production.
27. **Provenance as a lineage graph** — content-addressed artifacts making "where did this result
    come from?" a traversal.
28. **Incident management and capacity planning as code** — operational logic that usually lives
    in wikis, implemented and tested.

## Business & Decisions (29–30)

29. **The seams are the product** — why integration coherence, not model accuracy, is the binding
    constraint on enterprise AI value.
30. **Bounded claims as a business asset** — measured vs configured labelling, no-fabrication
    policy, and why trustworthy engineering reporting lowers organisational risk.
