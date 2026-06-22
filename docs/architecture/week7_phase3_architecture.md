# Week 7 — Phase 3: Executive Decision Intelligence Layer

## Architecture Document

**Component:** `src/executive/executive_decision_engine.py`
**Role:** Transform fleet analytics into budget-constrained business decisions
**Input:** `FleetSnapshot` (from `src/fleet/fleet_digital_twin.py`, reused as-is)
**Output:** `ExecutiveDecisionPortfolio`
**Status:** Implemented, validated — 182 tests passing; all Week 1–7 Phase 2 modules unchanged and their suites still pass.

---

## 1. Purpose

The platform's predictive stack (Weeks 1–5) diagnoses individual assets; the fleet engine (Week 7 Phase 2) rolls those diagnoses into a fleet-wide health-and-risk snapshot. This engine takes the final step: it converts that snapshot into a **decision** an executive can approve. It answers six questions leadership actually asks:

1. Which assets should we repair first?
2. Which maintenance actions maximise ROI?
3. Given a limited budget, what is the optimal maintenance portfolio?
4. What risk reduction should we expect?
5. What is the expected financial impact?
6. What should we approve this month?

It consumes a `FleetSnapshot` exactly as delivered — composing, not modifying, the frozen fleet engine — and emits an `ExecutiveDecisionPortfolio` carrying the selected assets, costs, savings, ROI, risk-reduction analytics, a prioritised ranking, natural-language recommendations, and a confidence score.

---

## 2. Architecture Diagram

```
                 ┌──────────────────────────────────────┐
                 │   FleetSnapshot  (frozen, Phase 2)    │
                 │   .assets: tuple[FleetAsset, ...]     │
                 │   .risk_concentration, .currency, ... │
                 └───────────────────┬──────────────────┘
                                     │
                 ┌───────────────────▼──────────────────┐
                 │       ExecutiveDecisionEngine         │
                 │                                       │
                 │  1. prioritize_assets()               │
                 │       priority_score per asset        │
                 │                                       │
                 │  2. optimize_portfolio()              │
                 │       budget-constrained selection    │
                 │       ├── greedy ROI                   │
                 │       ├── greedy savings               │
                 │       └── hybrid (priority)            │
                 │                                       │
                 │  3. risk-reduction analytics          │
                 │       before / after / reduction %    │
                 │                                       │
                 │  4. ROI analytics                     │
                 │       ROI / payback / cost efficiency │
                 │                                       │
                 │  5. executive recommendation          │
                 │       top risks · opportunities ·     │
                 │       immediate actions · narrative   │
                 │                                       │
                 │  6. confidence score                  │
                 │       data quality · coverage ·       │
                 │       risk dispersion                 │
                 └───────────────────┬──────────────────┘
                                     │
                 ┌───────────────────▼──────────────────┐
                 │   ExecutiveDecisionPortfolio          │
                 │   (frozen, JSON-serialisable)         │
                 └──────────────────────────────────────┘
```

The engine adds a decision layer **on top of** the frozen fleet engine. It imports `FleetSnapshot` and `FleetAsset` and reads them; it writes nothing back. The five frozen prior test suites (185 + 121 + 154 + 202 + 157) continue to pass unchanged.

---

## 3. Core Capabilities

### 3.1 Asset Prioritization

Each asset receives a composite **priority score** blending four normalised signals, exactly as specified:

> priority_score = (0.40·risk + 0.30·savings_norm + 0.20·criticality + 0.10·urgency) / Σweights

where
- **risk** = the asset's fleet risk score, already in `[0, 1]`;
- **savings_norm** = the asset's expected savings divided by the fleet maximum (so the most valuable repair scores 1);
- **criticality** = the maintenance decision's severity score normalised by a cap of 20;
- **urgency** = the asset's failure probability, in `[0, 1]`.

Dividing by the weight sum keeps the score in `[0, 1]` even when the weights are reconfigured. Higher score = repair sooner. The weights are config-exposed, so an operator who prioritises pure risk (or pure savings) can re-weight without touching code.

### 3.2 Budget-Constrained Portfolio Optimization

Given a budget, the engine selects assets to maximise realised value subject to `Σ maintenance_cost ≤ budget`. Three greedy strategies are provided, differing only in the **ordering** they greedily fill the budget with:

| Strategy | Orders assets by | Picks |
|----------|------------------|-------|
| `greedy_roi` | ROI = savings / cost, descending | cheap, high-return repairs first |
| `greedy_savings` | absolute expected savings, descending | the biggest-ticket repairs first |
| `hybrid` | composite priority score, descending | the balanced risk-and-value picks |

The strategies genuinely diverge when costs are heterogeneous. Worked example (budget = 6,000):

| Asset | Cost | Savings | ROI |
|-------|-----:|--------:|----:|
| H | 10,000 | 30,000 | 3.0 |
| L1 | 2,000 | 12,000 | 6.0 |
| L2 | 2,000 | 11,000 | 5.5 |
| L3 | 2,000 | 10,000 | 5.0 |

`greedy_roi` selects {L1, L2, L3} for 6,000 cost and **33,000** savings; `greedy_savings` cannot fit H (10,000 > 6,000) and so under a tight budget the ROI-first ordering captures more value. This is precisely why ROI-first matters under capital rationing, and the test suite asserts the divergence.

Greedy selection is used deliberately rather than exact knapsack: it is `O(N log N)`, transparent (an executive can see *why* each asset was chosen), and within a few percent of optimal for the near-uniform cost structures typical of a turbine fleet. Tie-breaking is by `asset_id`, so the selection is fully deterministic.

### 3.3 Risk-Reduction Analytics

The engine estimates the fleet risk before and after the proposed maintenance:

> risk_before = mean(risk_score over all assets)
> risk_after  = mean(risk_score·(1 − recovery_factor) if selected else risk_score)
> risk_reduction = risk_before − risk_after
> portfolio_risk_reduction_pct = 100 · risk_reduction / risk_before

The `recovery_factor` (default 0.70) is the fraction of an asset's risk removed by repairing it — a tunable that encodes how effective maintenance is. A repaired asset's residual risk is `risk·(1 − recovery)`; an unrepaired asset's risk is unchanged. Setting `recovery_factor = 1.0` models a perfect repair (residual risk 0); `0.0` models a no-op (no reduction).

### 3.4 ROI Analytics

> ROI = (expected_savings − total_cost) / total_cost
> payback_ratio = expected_savings / total_cost
> cost_efficiency = expected_savings / total_cost

ROI is the net return per unit of capital deployed; payback ratio and cost efficiency express savings per unit cost (a payback ratio of 9 means every dollar spent returns nine in avoided failure cost). All three are zero when no spend occurs, avoiding division-by-zero on an empty or zero-budget portfolio.

### 3.5 Executive Recommendation Generator

The engine surfaces the decision-relevant lists and a narrative:

- **top_risks** — the highest-risk asset ids (the residual exposure to watch);
- **top_opportunities** — the highest-savings asset ids (where value concentrates);
- **immediate_action_assets** — selected assets whose maintenance action is `immediate_maintenance` or `shutdown`;
- **budget_utilization** — the fraction of the budget consumed;
- **recommendations** — structured approval lines;
- **executive_summary** — a plain-English paragraph, e.g.:

> "Of 10 fleet assets, 4 are recommended for maintenance this period. The proposed portfolio costs USD 20,000 and is projected to return USD 180,000 in savings (ROI 800%), cutting fleet risk by 36%. Highest residual exposure: WTG-042. This consumes 100% of the approved budget."

### 3.6 Confidence Score

A single `[0, 1]` confidence score combines three components:

> confidence = 0.40·data_quality + 0.40·portfolio_coverage + 0.20·risk_dispersion

- **data_quality** — saturating in fleet size (`min(N/10, 1)`); larger fleets give more statistically stable aggregates;
- **portfolio_coverage** — the fraction of at-risk assets (risk ≥ 0.5) that the selection addresses; a plan that leaves high-risk assets untouched is less trustworthy;
- **risk_dispersion** — `1 − risk_concentration`; risk spread across the fleet supports a more robust decision than risk hidden in a single asset.

The score lets leadership weight the recommendation by how much evidence and coverage stand behind it.

---

## 4. KPI Definitions

| KPI | Definition |
|-----|------------|
| `total_maintenance_cost` | Σ cost of selected assets |
| `expected_savings` | Σ expected savings of selected assets |
| `total_roi` | (savings − cost) / cost |
| `payback_ratio` | savings / cost |
| `cost_efficiency` | savings per unit cost (= payback_ratio) |
| `budget_utilization` | total_cost / budget (0 when unconstrained) |
| `average_risk_before` | mean asset risk score pre-maintenance |
| `average_risk_after` | mean asset risk score post-maintenance |
| `expected_risk_reduction` | risk_before − risk_after |
| `portfolio_risk_reduction_pct` | 100 · reduction / risk_before |
| `confidence_score` | weighted blend of data quality, coverage, dispersion |

---

## 5. Executive Workflows

**Monthly maintenance approval.** An operations director sets the month's budget, runs the engine with the `hybrid` strategy, and reads the executive summary and recommendations. The selected portfolio, its cost, projected savings, risk reduction, and confidence score are the approval package — with the prioritised ranking available to justify each inclusion.

**Capital-rationing what-if.** A finance partner compares the three strategies at several budget levels to see how realised savings and risk reduction respond to budget — identifying the point of diminishing returns where additional budget no longer buys meaningful risk reduction.

**Board-level risk reporting.** The `confidence_score`, `portfolio_risk_reduction_pct`, and `top_risks` give a board a defensible, quantified statement of how much fleet risk the approved spend removes and what residual exposure remains.

---

## 6. Engineering Standards

The engine adheres to the platform standard in full: frozen, validated dataclasses (`ExecutiveDecisionConfig`, `PrioritizedAsset`, `ExecutiveDecisionPortfolio`); the `register` / `build` / `list` registry triad under `EXECUTIVE_DECISION_REGISTRY`; failure-safe tracker integration (every tracker call wrapped so a tracker fault cannot interrupt a decision); pure-NumPy numerics with no SciPy or PyTorch; JSON-serialisable outputs in which non-finite floats (an infinite budget, for instance) render as `null`; and full determinism — the same snapshot, budget, and strategy always yield the same portfolio, with stable `asset_id` tie-breaking. There is no global mutable state. A CLI `--demo` exercises all three strategies on a synthetic ten-turbine fleet.

---

## 7. Validation Summary

182 tests pass with zero skips, covering config validation, prioritization (including per-weight isolation tests), all three optimization strategies and their divergence, ROI and risk-reduction formulas checked against independent computation, confidence scoring, the executive narrative, registry, serialization, determinism, failure-safe tracker integration, and edge cases (empty, single, and 50-asset fleets; zero, fractional, and infinite budgets; zero-savings and unaffordable assets). The five frozen prior suites (185 + 121 + 154 + 202 + 157) continue to pass unchanged, and all eleven platform modules import and coexist cleanly.