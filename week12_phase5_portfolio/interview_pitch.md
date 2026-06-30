# Interview Pitch (60–90 seconds)

A spoken-length pitch for the platform, for use when an interviewer says "walk me through a project."
Timed at roughly 150 words per minute. A primary version (~90 s) and a compressed version (~60 s) are
provided, followed by likely follow-ups.

---

## Primary (~90 seconds)

"The project I'm proudest of is an Enterprise Digital Twin and Decision Intelligence Platform. The
problem it addresses is that industrial AI usually fails for systemic reasons, not algorithmic ones —
organisations run a digital twin, prediction models, document search and agents, but each is built
separately, so they disagree about data, ship on different schedules, and lose provenance. Nobody can
say how a given result was produced.

So I built one platform that hosts all of those capabilities behind a single governance, deployment and
observability fabric — ten layers with one dependency direction. The key decision was integration by
composition: subsystems exchange immutable data contracts instead of sharing state, which keeps the
dependency graph acyclic and lets me test every layer in isolation.

Underneath the capability layers, I built the production-engineering substrate: MLOps with a lineage
graph so every prediction traces to its model, data and code; monitoring that distinguishes data drift
from concept drift; quality-gated CI/CD; Kubernetes deployment with zero-downtime rollback; and full
observability with SLOs and incident management. The whole substrate is verified by a deterministic,
framework-agnostic test suite.

One thing I'm deliberate about: I don't claim benchmark numbers I haven't measured. SLO values are
configured targets, the readiness score is a self-assessment, and I documented a benchmark methodology
for the measurements that remain. I think being honest about scope is part of senior engineering."

## Compressed (~60 seconds)

"I built an Enterprise Digital Twin and Decision Intelligence Platform. The insight is that industrial
AI fails systemically — integration debt, lost provenance, fragmented monitoring — not because models
are weak. So I integrated digital twin, predictive, agentic, retrieval-augmented and workflow
capabilities into one ten-layer platform with a single dependency direction, integrated by composition
through immutable data contracts.

Beneath them I built the production substrate: MLOps with full lineage, data- and concept-drift
monitoring, quality-gated CI/CD, Kubernetes deployment with health-gated rollback, and end-to-end
observability — all verified by a deterministic test suite. And I was careful not to fabricate metrics:
SLO targets and the readiness score are labelled as configured targets and self-assessment, with a
benchmark methodology defined for what's left to measure."

## Likely Follow-ups (have these ready)

- *"What was the hardest part?"* — Keeping a ten-layer system from becoming a big ball of mud; solved by
  the composition-only contract and additive construction enforced by the test suite.
- *"What does 'integration by composition' buy you?"* — Acyclic dependencies, isolated testability, and
  the freedom to change a subsystem's internals as long as its value-object contract holds.
- *"Why no benchmarks?"* — I report verified properties and configured targets; sustained-load
  benchmarking is documented as future work. I won't present numbers I haven't measured.
- *"What would you do next?"* — Execute the benchmark methodology, mature capability-layer verification,
  and complete pre-production security hardening.
