# Week 8 — Phase 3: Scenario Planning Agent

## Architecture Document

**Component:** `src/agent/scenario_planning_agent.py`
**Role:** Forward-looking strategic-planning layer — predicts outcomes under alternative future decisions
**Inputs:** `FleetSnapshot` (Phase 2), baseline budget; optional per-asset `AssetEvidence`
**Outputs:** `ScenarioPlanningReport`, `ScenarioComparison`, `ScenarioRanking`, `ExecutivePlanningSummary`, `ScenarioPlan`
**Status:** Implemented and validated — 202 tests passing with zero skips; all Week 5–8 Phase 2 suites unchanged and coexisting (15/15 modules import cleanly).

---

## 1. Purpose

The platform can explain *what happened* (predictive stack, fleet twin) and *why* (root-cause agent). The Scenario Planning Agent adds the missing forward-looking dimension: *what may happen next, under alternative future decisions.* It is the strategic-planning layer, projecting four families of scenarios and condensing them into a ranked, executive-ready plan:

- **Budget** — decrease, freeze, increase → predict risk, coverage, ROI, expected savings.
- **Delay** — defer maintenance 7 / 14 / 30 / 60 days → predict failure probability, downtime, loss.
- **Load** — operating load increase / decrease → predict health, RUL, risk.
- **Growth** — fleet expands +10% / +25% / +50% / +100% → predict maintenance demand, risk exposure, budget impact.

It then compares scenarios, ranks them, and assembles an executive planning summary with a recommended path and confidence.

A guiding principle: every projection is a **deterministic, analytic function** of the snapshot's real aggregates — no LLM, no randomness — so each scenario is auditable and reproducible. Where a scenario depends on real decision economics (budget), the agent composes the actual Executive Decision Engine rather than approximating it.

---

## 2. Architecture and Integration

```
   FleetSnapshot (Phase 2)
        ▼
   ScenarioPlanningAgent
        ├── budget_scenarios()  ── runs ExecutiveDecisionEngine at each budget
        ├── delay_scenarios()   ── compounding-hazard failure escalation
        ├── load_scenarios()    ── load-coupled degradation model
        ├── growth_scenarios()  ── extensive-quantity scaling
        ├── compare()           ── delta metrics on shared predictions
        ├── rank_scenarios()    ── by risk / savings / ROI / coverage
        ├── executive_summary() ── best / worst / recommended / confidence
        │                          └── DecisionCopilotAgent narrative
        └── plan()              ── full ScenarioPlan
                                   └── RootCauseAnalysisAgent cause context
        ▼
   ScenarioPlan  (frozen · JSON-serialisable · deterministic)
```

The agent integrates with the platform exactly as the brief requires, composing frozen modules without modifying them:

- **Fleet Digital Twin** — its `FleetSnapshot` is the world-state input every scenario reads.
- **Executive Decision Engine** — budget scenarios *run it* at the decrease, freeze, and increase budgets and read off real portfolio economics (residual risk, coverage, ROI, savings, confidence). This is the strongest integration: budget projections are not approximated, they are the engine's actual output.
- **Decision Copilot Agent** — the recommended budget scenario's portfolio is passed through the Copilot's `explain_portfolio` to weave a grounded risk-reduction statement into the strategic commentary.
- **Root Cause Analysis Agent** — optional; when per-asset evidence is supplied to `plan`, the fleet's dominant cause is folded into the commentary so planning can weight interventions accordingly.
- **Monte Carlo Risk Engine** — its uncertainty methodology informs the confidence model (confidence decays with extrapolation distance); an instance may be supplied for callers that hold simulator configurations.

---

## 3. Scenario Models

### 3.1 Budget scenarios

For each budget level — `decrease_factor·B`, `B`, `increase_factor·B` — the agent constructs an `ExecutiveDecisionConfig`, runs the Executive Decision Engine, and records residual risk (`average_risk_after`), coverage (`selected / asset_count`), ROI, and expected savings, with the engine's own confidence score. Because the engine is composed directly, the monotonic relationships hold by construction: a larger budget never raises residual risk or lowers coverage.

### 3.2 Delay scenarios

Deferring maintenance compounds the failure hazard. For a delay of `d` days against a baseline fleet failure probability `p₀`:

```
k          = 1 + d / delay_period
p_delayed  = 1 − (1 − p₀)^k
ratio      = p_delayed / p₀
downtime   = fleet_expected_downtime · ratio
loss       = fleet_expected_failure_cost · ratio
```

The survival term `(1 − p₀)^k` guarantees `p_delayed` is monotone increasing in `d` and bounded below 1 — a principled compounding-hazard model rather than a linear extrapolation that could exceed unity. Confidence decays linearly with the horizon (floored at 0.4), reflecting that longer-horizon projections are less certain.

### 3.3 Load scenarios

Operating load scales the degradation rate. For a load multiplier `m` (e.g. 1.2 for +20%):

```
health = clip( avg_health · (1 − health_sensitivity·(m−1)), 0, 100 )
rul    = max( avg_rul · (1 − rul_sensitivity·(m−1)), 0 )
risk   = clip( mean_risk · (1 + risk_sensitivity·(m−1)), 0, 1 )
```

A load increase (`m > 1`) lowers health and RUL and raises risk; a decrease does the inverse. All outputs are clamped to their valid ranges, so extreme load excursions saturate rather than producing impossible values.

### 3.4 Growth scenarios

Fleet growth scales the *extensive* quantities by the growth factor `g`:

```
maintenance_demand = (assets requiring action) · g
risk_exposure      = fleet_expected_failure_cost · g
budget_impact      = fleet_expected_cost · g
```

Confidence decays with extrapolation distance (`1 − (g−1)·0.4`, floored at 0.4), since a 100% expansion is a far bolder assumption than a 10% one.

---

## 4. Comparison, Ranking, and Summary

**Comparison** computes deltas on the metrics two scenarios share. Convenience fields (`risk_delta`, `cost_delta`, `roi_delta`, `coverage_delta`) are populated when both scenarios carry the metric and are `None` otherwise — so comparing a budget scenario with a delay scenario yields no spurious deltas.

**Ranking** orders scenarios by a criterion, with the correct sense per metric: lower residual risk is better; higher savings, ROI, and coverage are better. Scenarios lacking the ranking metric are skipped, so ranking a set of delay scenarios by coverage simply returns an empty ranking rather than an error.

**Executive summary** ranks the scenarios, names the best, worst, and recommended (the best by the chosen criterion), reports mean confidence, and composes a strategic commentary. When a recommended portfolio is available, the commentary is enriched with the Copilot's risk-reduction narrative; when root-cause evidence is available, the dominant cause is appended.

---

## 5. Engineering Standards

The agent meets the platform standard in full: six frozen, validated dataclasses each with a `to_dict`; the `register` / `build` / `list` registry triad under `SCENARIO_PLANNING_REGISTRY`; failure-safe tracker and failure-safe optional-integration calls (a Copilot or RCA fault is caught and logged, never interrupting a plan); pure Python and NumPy with no LLM, PyTorch, SciPy, or pandas; JSON-serialisable outputs with non-finite floats rendered as `null`; and full determinism — every projection is a closed-form function of the snapshot with stable tie-breaking. Inputs are validated (empty snapshots and negative budgets are rejected, unknown ranking criteria raise), and a CLI `--demo` runs all four families plus comparison, ranking, and the executive summary end-to-end.

---

## 6. Validation Summary

202 tests pass with zero skips, covering the scenario and criterion enums, config validation, the registry, all four scenario families (with monotonicity and boundedness checks — delay failure probability monotone and capped at 1, load increase degrading health/RUL while raising risk, growth scaling extensive quantities), comparison delta arithmetic, ranking in the correct sense for every criterion, the executive summary, the top-level `plan` orchestration, the Executive / Copilot / Root-Cause integrations, serialization, determinism, failure-safe tracker behaviour, and edge cases (single-asset and large fleets, zero and very large budgets, custom factors, zero sensitivities). The five Week 7 / Week 8 prior suites continue to pass unchanged, and all fifteen platform modules import and coexist cleanly.