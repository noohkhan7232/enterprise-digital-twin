# Demonstration Storyboard

A scene-by-scene storyboard for the platform demonstration, structured as beginning, middle and end.
Each scene lists the speaker, what is on screen, the talking points, and the expected outcome. The
storyboard suits both live delivery and the recorded video walkthrough.

---

## Beginning — Frame the Problem (Scenes 1–3)

### Scene 1 — Cold open
- **Speaker:** Presenter
- **Screen:** Title card — platform name and one-line description.
- **Talking points:** "Most industrial AI fails not because the models are weak, but because the
  *systems* around them are. Today I'll show a platform built to fix that."
- **Expected outcome:** Audience oriented; problem framed as systemic.

### Scene 2 — The integration problem
- **Speaker:** Presenter
- **Screen:** Simple before-state diagram: four disconnected capabilities (twin, prediction,
  knowledge, agents) each with its own data, release and monitoring.
- **Talking points:** Incompatible data models, separate releases, fragmented monitoring, lost
  provenance. The cost is audit pain, undetected failures and risky deployments.
- **Expected outcome:** Audience feels the pain concretely.

### Scene 3 — The thesis
- **Speaker:** Presenter
- **Screen:** Architecture figure (the integrated ten-layer view).
- **Talking points:** One platform, ten layers, integrated by composition — capabilities stay
  independent but share governance, deployment and observability.
- **Expected outcome:** Audience sees the proposed solution at a glance.

## Middle — Show the Platform (Scenes 4–9)

### Scene 4 — Capability layers
- **Speaker:** Presenter
- **Screen:** Architecture figure with capability layers highlighted.
- **Talking points:** Digital twin (asset + state), predictive (forecasts, injectable strategies),
  agentic (typed, auditable actions), knowledge (evidence-grounded retrieval), workflow
  (deterministic multi-step processes).
- **Expected outcome:** Audience understands what the platform *does*.

### Scene 5 — MLOps and provenance
- **Speaker:** Presenter
- **Screen:** MLOps layer highlighted; lineage concept sketch.
- **Talking points:** Versioned registry, artifact store, reproducibility, lineage graph. "How was
  this result produced?" becomes a query.
- **Expected outcome:** Audience grasps provenance-by-construction.

### Scene 6 — Monitoring
- **Speaker:** Presenter
- **Screen:** Monitoring concept: drift, quality, health, routed alerts.
- **Talking points:** Data drift vs concept drift; composite health; observer-based alerting;
  signals feed operational decisions.
- **Expected outcome:** Audience sees early-detection value.

### Scene 7 — CI/CD and deployment
- **Speaker:** Presenter
- **Screen:** Terminal — health check; deployment topology figure.
- **Talking points:** Twenty quality gates, release validation, readiness checks; zero-downtime
  rollout and health-gated rollback; one health check everywhere.
- **Expected outcome:** Audience sees safe, fast change.

### Scene 8 — Observability, live
- **Speaker:** Presenter
- **Screen:** Terminal — run metrics, reliability and readiness demos; show JSON.
- **Talking points:** Operational questions become numbers; run twice to show determinism; SLO
  values are configured targets, readiness is a transparent self-assessment.
- **Expected outcome:** Audience sees measurable operations and reproducibility.

### Scene 9 — Verification
- **Speaker:** Presenter
- **Screen:** Terminal — `pytest tests/ -q` summary (1,503 passing).
- **Talking points:** Deterministic, framework-agnostic suite; structured so failures localise; full
  run re-verifies every subsystem.
- **Expected outcome:** Audience trusts the engineering rigour.

## End — Land the Value (Scenes 10–12)

### Scene 10 — Business outcomes
- **Speaker:** Presenter
- **Screen:** Three-point outcome card: trust/auditability, early detection, safe change.
- **Talking points:** Tie each technical capability to a business outcome.
- **Expected outcome:** Technical depth reframed as value.

### Scene 11 — Honest framing and future
- **Speaker:** Presenter
- **Screen:** "What we claim / what we don't" card; future-work bullets.
- **Talking points:** Verified engineering properties and configured targets, not fabricated
  benchmarks; future work is sustained-load study, contract formalisation, agent observability,
  decision-quality evaluation.
- **Expected outcome:** Audience trusts the presenter's integrity.

### Scene 12 — Close
- **Speaker:** Presenter
- **Screen:** Architecture figure; contact/repository placeholder.
- **Talking points:** "The hard part of industrial AI is operating many capabilities together,
  coherently and safely — that is what this platform is built to do."
- **Expected outcome:** Clear, memorable close; transition to Q&A.

---

## Timing Guides

| Cut | Scenes | Target duration |
|-----|--------|-----------------|
| Executive | 1–3, 5–7, 10–12 | 10–12 min |
| Technical | 1, 3–9, 11–12 | 20–25 min |
| Video walkthrough | 1–12 | 12–15 min |