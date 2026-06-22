# Week 8 — Phase 4: Executive Intelligence Agent

## Architecture Document

**Component:** `src/agent/executive_intelligence_agent.py`
**Role:** Top-level executive orchestration layer — the platform's final synthesis
**Inputs:** `FleetSnapshot`; optional `AssetEvidence`, budget, scenario flag
**Outputs:** `ExecutiveIntelligenceReport`
**Status:** Implemented and validated — 182 tests passing with zero skips; all Week 5–8 Phase 3 suites unchanged and coexisting (16/16 modules import cleanly).

---

## 1. Purpose

This is the capstone of the platform. Each prior module answers one slice of the decision problem; the Executive Intelligence Agent orchestrates them into a single report that answers, in order, the questions leadership actually asks:

| Question | Answered by |
|----------|-------------|
| What is happening right now? | Fleet Digital Twin snapshot + Copilot overview |
| What is most likely to fail? | Risk assessment + executive priority score |
| Why? | Root Cause Analysis Agent |
| What should we do? | Executive Decision Engine |
| What happens if we change strategy? | Scenario Planning Agent |
| What is the best executive decision? | Synthesised narrative and summary |

The defining architectural constraint is that this agent contains **no business logic of its own** beyond the executive priority score and the report assembly. It is pure orchestration: it composes the frozen Fleet, Executive, Copilot, Root-Cause, and Scenario modules and never re-implements them. This keeps the platform's single source of truth intact — every health, risk, cause, decision, and projection is computed exactly once, in the module that owns it.

---

## 2. Architecture

```
   FleetSnapshot  (+ optional AssetEvidence, budget, scenario flag)
        ▼
   ExecutiveIntelligenceAgent
        ├── fleet_assessment()      → Fleet snapshot + Copilot.explain_fleet
        ├── risk_assessment()       → executive priority score over assets
        ├── root_cause_assessment() → RootCauseAnalysisAgent.analyze_fleet
        ├── decision_assessment()   → ExecutiveDecisionEngine.recommend
        ├── scenario_assessment()   → ScenarioPlanningAgent.plan
        ├── narrative_generation()  → DecisionCopilotAgent.explain_portfolio
        └── generate_report()       → synthesis of all of the above
        ▼
   ExecutiveIntelligenceReport
        (frozen · JSON-serialisable · deterministic)
```

The agent default-creates its collaborators (Copilot, Root-Cause, Scenario) so it works out of the box, while accepting injected instances for testing or customisation. The Monte Carlo engine is accepted as an optional interoperability hook for callers that hold simulator configurations.

---

## 3. Executive Priority Score

The one piece of new quantitative logic the agent introduces is the executive priority score, which ranks assets for leadership attention by combining four normalised signals:

```
priority = ( w_risk·risk
           + w_crit·criticality
           + w_cost·cost_exposure
           + w_fail·failure_probability ) / Σ weights
```

where `risk` is the asset's risk score, `criticality` is its severity normalised by the severity cap, `cost_exposure` is its expected savings normalised by the fleet maximum, and `failure_probability` is its failure probability — each clamped to `[0, 1]`. The default weights are 0.40 / 0.20 / 0.20 / 0.20, placing risk first while still rewarding high-severity, high-stakes, near-failure assets. Dividing by the weight sum guarantees the result lands in `[0, 1]` for any non-negative weights, so the formula is both explainable and robust to re-weighting. The score is monotone in each input, which the test suite verifies directly.

---

## 4. Core Workflow

`generate_report` runs the seven assessments and synthesises them:

1. **Fleet assessment** reads the snapshot's average health, mean asset risk, and average RUL, and asks the Copilot for a fleet overview.
2. **Risk assessment** scores every asset with the priority formula and returns the top-N as `ExecutiveRisk` records.
3. **Root-cause assessment** runs the Root-Cause agent over the supplied evidence (and is empty when no evidence is given — a causal claim requires subsystem signals).
4. **Decision assessment** runs the Executive Decision Engine at the chosen budget (defaulting to the fleet's expected maintenance cost).
5. **Scenario assessment** runs the Scenario Planning agent and extracts the recommended posture and its confidence — skipped when scenarios are disabled.
6. **Narrative generation** composes the situation, diagnosis, action, and outlook, drawing the action sentence from the Copilot's portfolio explanation.
7. **Synthesis** assembles the `ExecutiveIntelligenceReport`, including a budget recommendation, a scenario recommendation, a ranked set of `ExecutiveRecommendation`s spanning maintenance / budget / investigation / scenario, a strategic narrative, an executive summary, and an overall confidence blended from the decision and scenario confidences.

Each assessment is also a public method, so callers can run any single capability in isolation.

---

## 5. Dataclasses

Five frozen, JSON-serialisable dataclasses carry the output, each with a `to_dict`:

- **ExecutiveRisk** — a top-risk asset with its risk score, failure probability, cost exposure, composite priority score, and qualitative risk tier.
- **ExecutiveFinding** — a root-cause finding (type, subject, statement, confidence).
- **ExecutiveRecommendation** — a recommendation (category, title, rationale, priority).
- **ExecutiveNarrative** — the composed narrative (overview, situation, diagnosis, action, outlook, executive summary).
- **ExecutiveIntelligenceReport** — the top-level report carrying all eleven required sections.

Three enums (`RiskTier`, `RecommendationCategory`, `FindingType`) keep the categorical fields type-safe and serialisation-friendly.

---

## 6. Engineering Standards

The agent meets the platform standard in full: six frozen, validated dataclasses (five output types plus the config) with `to_dict` on every serialised output; the `register` / `build` / `list` registry triad under `EXECUTIVE_INTELLIGENCE_REGISTRY`; failure-safe tracker and failure-safe orchestration (a Copilot, Root-Cause, or Scenario fault is caught and logged, never crashing a report); pure Python and NumPy with no LLM, PyTorch, SciPy, pandas, or external APIs; JSON-serialisable outputs with non-finite floats rendered as `null`; and full determinism — every assessment is a closed-form function of the snapshot with stable tie-breaking by priority then asset id. The snapshot is validated (empty fleets and missing fields are rejected), the config is validated (non-negative weights summing to a positive total, ordered risk thresholds), and a CLI `--demo` produces a complete report end-to-end over a synthetic fleet.

---

## 7. Validation Summary

182 tests pass with zero skips, covering the three enums, config validation and the risk-tier map, the registry, all five dataclasses (construction, frozen-ness, and serialization), all seven assessment capabilities, the executive priority score (explicit-formula, weight-isolation, monotonicity, and clamping checks), the full `generate_report` orchestration (all sections populated, consistency between standalone assessments and the assembled report), serialization, determinism, failure-safe tracker behaviour, and the full matrix of optional-input edge cases (single-asset and large fleets, no evidence, no scenario, no budget, zero budget, custom collaborators, and invalid inputs). The six Week 7 / Week 8 prior suites continue to pass unchanged, and all sixteen platform modules import and coexist cleanly.

---

## 8. Closing the Platform

With this agent, the Enterprise Digital Twin & Decision Intelligence Platform is functionally complete end-to-end: simulation and prediction produce the raw diagnosis; the fleet twin aggregates it; the risk, executive, and strategic engines turn it into decisions; and the Week-8 agent layer explains those decisions (Copilot), attributes their causes (Root-Cause), projects their futures (Scenario), and — here — fuses everything into one executive report. Because the orchestration layer adds no duplicated logic, the platform retains a single, auditable source of truth for every quantity it reports.