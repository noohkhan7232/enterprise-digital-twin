# Frequently Asked Questions

Answers are grouped by topic. They describe the platform as built and avoid claims that are not
supported by the repository.

---

## Architecture

**1. What is the platform, in one sentence?**
An integrated, production-engineered system that hosts digital-twin, predictive, agentic,
retrieval-augmented and workflow capabilities behind one governance, deployment and observability
fabric.

**2. How many layers does the architecture have?**
Ten: digital twin, predictive intelligence, agentic AI, knowledge intelligence (RAG), workflow
engine, MLOps, monitoring, CI/CD, deployment and observability.

**3. What is the central architectural idea?**
Integration by composition with a single dependency direction. Layers exchange immutable,
serialisable value objects instead of sharing mutable state or calling into each other's internals.

**4. Why composition over inheritance?**
Composition keeps the design flat and explicit, lets subsystems evolve independently, and avoids
fragile inheritance hierarchies. Inheritance is reserved for genuine type specialisation.

**5. How do subsystems stay decoupled?**
The only contract between subsystems is a set of immutable value objects. A consumer depends on a
stable data shape, not on another subsystem's behaviour, which keeps the dependency graph acyclic.

**6. Was the system built additively?**
Yes. Each layer was added without modifying its predecessors, and the full test suite re-verifies
every prior layer whenever it runs.

**7. What language and dependencies does it use?**
Python 3.12 with NumPy for numerical computing and PyYAML for configuration. No external
observability, orchestration or ML platforms are required at runtime.

**8. Is the architecture documented?**
Yes — see `architecture_overview.md` and the companion research paper `research_paper.md`.

## Deployment

**9. How is the platform deployed locally?**
Via Docker Compose, using either a development image with bind-mounted source or a hardened
production image.

**10. How is it deployed to Kubernetes?**
With `deployment/scripts/deploy_kubernetes.sh`, which applies ten manifests in dependency order,
waits for rollout and verifies health in a live pod.

**11. What makes the production container secure?**
It is multi-stage and non-root, runs with a read-only root filesystem, drops all Linux
capabilities, forbids privilege escalation and declares a container health check.

**12. How is high availability achieved?**
Multiple replicas, autoscaling (3–12), node-level topology spread, a pod disruption budget and
durable state in a persistent volume.

**13. How does rollback work?**
`rollback.sh` reverts the deployment to a previous revision, waits for stabilisation and
re-verifies health, failing if the rolled-back state is unhealthy. Rollback is zero-downtime.

**14. What is the single health check used for?**
It backs the container `HEALTHCHECK`, the Kubernetes probes' intent and every deployment and
rollback script, so "healthy" is defined once and consistently.

**15. Can I deploy without Kubernetes?**
Yes. The hardened Docker Compose profile runs the production image locally with health checks,
logging, resource limits and restart policy.

## Testing

**16. How many tests are there?**
1,503 automated tests, all passing.

**17. What kind of tests are they?**
Deterministic and framework-agnostic: standard assertions and parameterisation only, with no
fixtures, network access or external services.

**18. How is determinism tested?**
Dedicated tests assert that repeated execution with identical inputs yields identical outputs,
including fully serialised reports.

**19. How is the suite structured?**
Into value-object tests, engine (behavioural) tests, edge-case tests and determinism tests, so
that a failure localises to a single component.

**20. How do I run the tests?**
`PYTHONPATH=src:scripts pytest tests/ -q` for the full suite, or a single test file for one
subsystem.

**21. Are benchmark results reported?**
No. The repository reports verified engineering properties and configured targets, not fabricated
latency or throughput numbers.

## Artificial Intelligence

**22. What AI capabilities does the platform host?**
Predictive intelligence (forecasting and prognostics), agentic reasoning, and retrieval-augmented
knowledge, coordinated by a workflow engine and governed by MLOps.

**23. Does the platform train models itself?**
The platform manages the lifecycle of models and provides the substrate to operate them; specific
training algorithms are pluggable behind the predictive interface.

**24. How are AI outputs made trustworthy?**
Through provenance (every prediction traces to its model version, experiment, dataset and code),
monitoring (drift and health), and observability (traceable, auditable behaviour).

**25. What is the agentic layer?**
A tool-using autonomous reasoning layer following the perception–deliberation–action abstraction,
with typed actions and recorded reasoning trajectories so behaviour is governable and auditable.

## Digital Twin

**26. What is a digital twin here?**
A virtual representation of a physical asset coupling a static asset model with a dynamic state
stream synchronised from telemetry.

**27. How does the twin integrate with other layers?**
It exposes asset and state as immutable value objects. Predictive intelligence reads state to
forecast; the workflow engine conditions processes on state; the knowledge layer links documents
to assets by identifier.

**28. Does the platform distinguish models, shadows and twins?**
Conceptually yes — the degree of automated synchronisation provided for an asset class reflects
the standard model/shadow/twin distinction.

## Retrieval-Augmented Generation (RAG)

**29. What does the knowledge layer do?**
It answers questions over enterprise corpora by retrieving relevant evidence and conditioning
generation on it.

**30. How is provenance handled in RAG?**
Retrieved evidence is carried alongside generated answers, so responses are attributable to their
sources.

**31. Is retrieval reproducible?**
The corpus index is treated as a versioned artefact, so the evidence available at a given time can
be reconstructed.

**32. How does RAG connect to the digital twin?**
Asset identifiers link documents to physical assets, allowing retrieval to be scoped to the
equipment a query concerns.

## MLOps

**33. What does the MLOps subsystem provide?**
Experiment tracking, a versioned model registry with semantic versioning and stage promotion, a
content-addressed artifact store, a reproducibility engine and governance via documentation and a
lineage graph.

**34. How is reproducibility achieved?**
Runs are bound to their source revision and an environment snapshot, and support deterministic
re-execution.

**35. What is the lineage graph?**
A record of provenance linking runs, datasets, artifacts and models, so any result can be traced
to the inputs and process that produced it.

**36. How does governance work?**
Model cards document intent and risk; the lineage graph records provenance; policy files express
release and promotion rules. Governance is data-driven and queryable.

## Monitoring

**37. What does production monitoring detect?**
Data drift, concept drift, prediction anomalies, data-quality regressions and composite model
health.

**38. What is the difference between data drift and concept drift?**
Data drift is a change in input distributions; concept drift is a change in the relationship
between inputs and outcomes. The platform detects both.

**39. How are alerts handled?**
An alert engine consumes monitoring signals and emits routed alerts using observer-style fan-out,
so new reactions can be added without modifying detectors.

**40. Where do monitoring signals go?**
They are data contracts consumed by the reliability and observability layers, so degradation feeds
operational decisions rather than terminating at a dashboard.

## CI/CD

**41. What does the CI/CD subsystem enforce?**
Repository validation, twenty quality gates, release-policy validation and deployment-readiness
checks, wired into three GitHub Actions workflows.

**42. What happens when a quality gate fails legitimately?**
The engine reports the failure honestly rather than masking it, preserving the gate's diagnostic
value.

**43. How does CI/CD connect to deployment?**
The deployment-readiness validator confirms that container, orchestration, health and rollback
assets are present and coherent, linking delivery forward to deployment.

## Observability

**44. What does the observability subsystem cover?**
Metrics, distributed tracing, structured logging, reliability engineering, SLI/SLO with error
budgets, incident management, capacity planning and a production-readiness assessment.

**45. What are the default service-level objectives?**
Availability ≥ 0.99 (30-day window), P95 latency ≤ 250 ms, error rate ≤ 0.01 and data freshness
≤ 300 s. These are configured targets evaluated at runtime.

**46. What is the production-readiness assessment?**
A weighted evaluation across ten areas — architecture, security, reliability, monitoring,
deployment, CI/CD, testing, documentation, MLOps and observability — producing a score and a
categorical level. It is a transparent self-assessment, not an external certification.

**47. How does distributed tracing work without an external backend?**
The tracing engine records spans with parent/child relationships and reconstructs trace timelines
and critical paths in pure Python, with no dependency on external tracing systems.

**48. Can observability data be exported to external tools?**
The subsystem is self-contained and emits structured JSON; bridging to external monitoring
ecosystems is an integration task left open by design.