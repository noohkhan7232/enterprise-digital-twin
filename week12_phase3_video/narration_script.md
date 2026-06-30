# Video Narration Script

**Title:** Enterprise Digital Twin & Decision Intelligence Platform — Technical Walkthrough
**Target length:** 12–15 minutes
**Style:** Calm, precise, engineering-credible. No hype. Spoken in first person.

The script is divided into timed segments. Each segment gives the **narration** (spoken verbatim or
near-verbatim) and the **on-screen** action. Bracketed cues are not spoken. Pacing assumes roughly
130–140 words per minute.

---

### Segment 1 — Cold Open (0:00–0:45)
**On-screen:** Title card, then the architecture figure fading in.
**Narration:**
"Most industrial AI doesn't fail because the models are weak. It fails because the systems around
the models are. In this walkthrough I'll show a platform built to address that — one that hosts
several AI capabilities, but operates them as a single, coherent, governable system. I'll keep this
honest: I'll show verified engineering properties and reproducible behaviour, not marketing
benchmarks."

### Segment 2 — The Problem (0:45–2:00)
**On-screen:** Before-state diagram: twin, prediction, knowledge, agents — disconnected.
**Narration:**
"Here's the situation in many plants today. There's a digital twin of the assets. There are
predictive models. There's a search system over technical documents. And increasingly there are
agents that act on results. Each was built separately. So they disagree about what an asset even is,
they ship on different schedules, they're monitored by different tools, and when something goes
wrong, nobody can quickly say which model produced which result from which data. That lost
provenance is where audit pain, undetected failures, and risky deployments come from."

### Segment 3 — The Thesis and Architecture (2:00–3:30)
**On-screen:** The ten-layer architecture figure; highlight capability vs. operational layers.
**Narration:**
"The platform integrates these capabilities into ten layers with a single dependency direction. Five
capability layers — digital twin, predictive intelligence, agentic AI, retrieval-augmented knowledge,
and a workflow engine — sit on top of five production-engineering subsystems: MLOps, monitoring,
CI/CD, deployment, and observability. The invariant that makes this maintainable is that layers
integrate by composition. The only thing that crosses a boundary is an immutable, serialisable value
object — never shared mutable state. That keeps the dependency graph acyclic, and it means every
layer can be tested in isolation."

### Segment 4 — Capability Layers (3:30–5:15)
**On-screen:** Zoom each capability layer in turn.
**Narration:**
"Briefly, the capability layers. The digital twin couples a static asset model with a synchronised
state stream, exposed as immutable contracts. Predictive intelligence forecasts over that state, with
the forecasting strategy injected so the interface stays stable while the algorithm varies by asset
class. The agentic layer reasons and acts, but only through typed tool interfaces, and it records its
reasoning trajectory, so autonomy stays auditable. The knowledge layer answers questions by
retrieving evidence and carrying that evidence with the answer, over a versioned index so retrieval
is reproducible. And the workflow engine composes these into governed, deterministic processes."

### Segment 5 — MLOps and Provenance (5:15–6:45)
**On-screen:** MLOps layer; lineage sketch.
**Narration:**
"Underneath, MLOps is the provenance backbone. A versioned model registry, a content-addressed
artifact store, a reproducibility engine that binds each run to its source revision and environment,
and a lineage graph linking runs, datasets, artifacts and models. The question production AI usually
can't answer — exactly how was this result produced — becomes a graph traversal, because provenance
is built into the data model rather than bolted on as logging."

### Segment 6 — Monitoring (6:45–8:00)
**On-screen:** Monitoring concept: drift, quality, health, alerts.
**Narration:**
"Monitoring watches behaviour after deployment. It distinguishes data drift — a change in the inputs
— from concept drift — a change in the relationship between inputs and outcomes. It tracks prediction
anomalies, data quality, and a composite model-health score, and routes alerts through an
observer-based engine, so new reactions are added by subscribing rather than by editing detectors.
And those signals feed upward into reliability and observability — they don't just light up a
dashboard."

### Segment 7 — CI/CD and Deployment (8:00–9:30)
**On-screen:** Terminal — health check; then deployment topology figure.
**Narration:**
"Delivery is gated: a shared validation library, twenty quality gates, release validation, and a
deployment-readiness check, wired into three workflows. When a gate legitimately fails, it's reported
honestly, not masked. Deployment is a multi-stage, non-root container and ten Kubernetes manifests —
rolling updates with no unavailable replicas, autoscaling, a network policy, a disruption budget, and
durable storage. One deterministic health check backs the container, the probes, and the scripts, so
'healthy' means one thing everywhere. Rollback reverts and re-verifies health automatically."

### Segment 8 — Observability, Live (9:30–11:30)
**On-screen:** Terminal — run metrics, reliability, readiness demos; show JSON; run one twice.
**Narration:**
"Observability is where operation becomes measurement. Watch — I'll run the metrics demo, the
reliability demo, and the readiness assessment. Each emits structured JSON. Now I'll run one a second
time — and the output is byte-identical, because the whole system is deterministic. One note in the
interest of honesty: the service-level numbers here are configured targets the engine evaluates at
runtime — they are not benchmark latencies. I'm not going to show you invented performance figures."

### Segment 9 — Verification (11:30–12:45)
**On-screen:** Terminal — `pytest tests/ -q` summary.
**Narration:**
"All of this is verified by 1,503 deterministic, framework-agnostic tests — just assertions and
parameterisation, no fixtures, no network. They're structured so a failure points to one component,
and running the full suite re-verifies every subsystem's contracts. That's about 10,620 lines of
source across 30 production-engineering modules."

### Segment 10 — Close (12:45–14:00)
**On-screen:** Architecture figure; repository placeholder.
**Narration:**
"So, the engineering story here is integration done with discipline. Established techniques, assembled
so the whole stays coherent, governable, and operable as it grows. I've framed the contributions as
engineering — architecture, integration and implementation — not as algorithmic novelty, and I've
been deliberate about not inventing numbers. If you want the depth, there's a research paper and full
documentation in the repository. Thanks for watching."

---

## Word-Count Guide
Total spoken content is approximately 1,700–1,900 words, which fits 12–15 minutes at a measured pace
with pauses for the live terminal segments.