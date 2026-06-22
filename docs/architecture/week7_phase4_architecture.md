# Week 7 — Phase 4: Strategic Portfolio Optimization Engine

## Architecture Document

**Component:** `src/strategy/portfolio_strategy_engine.py`
**Role:** Enterprise capital-allocation intelligence — the decision-optimization layer above the Executive Decision Engine
**Input:** `FleetSnapshot` (Phase 2); composes `ExecutiveDecisionPortfolio` (Phase 3) outputs internally
**Output:** `StrategicPortfolioPlan`
**Status:** Implemented and validated — 210 tests passing; all Week 1–7 Phase 3 modules unchanged and their suites still pass (185 + 121 + 154 + 202 + 157 + 182).

---

## 1. Purpose

This is **not** a dashboard, a visualization, or a reporting layer. It is an enterprise decision-optimization engine that sits *above* the Executive Decision Engine. Where Phase 3 answers "given a fixed budget, which assets do we repair?", Phase 4 answers the questions a capital owner asks *before* the budget is set:

- How do different strategic postures — risk-first, ROI-first, criticality-first, balanced — compare for this fleet?
- How do risk reduction, savings, ROI, and coverage respond as the budget sweeps from low to high?
- Which budget levels are Pareto-efficient trade-offs of savings against residual risk?
- Where is the capital-efficiency knee, beyond which extra budget buys little additional risk reduction?
- What should we strategically recommend: increase the budget, hold, or stop because ROI has flattened?

The engine consumes `ExecutiveDecisionPortfolio` outputs — it runs the frozen Phase-3 engine across postures and budget levels and analyses the resulting portfolios. It modifies no prior code; it composes only.

---

## 2. Architecture Diagram

```
                ┌──────────────────────────────────────────┐
                │     FleetSnapshot   (frozen, Phase 2)      │
                └─────────────────────┬────────────────────┘
                                      │  composes the frozen Phase-3 engine
                                      │  once per (posture × budget)
                ┌─────────────────────▼────────────────────┐
                │   ExecutiveDecisionEngine  (Phase 3)      │
                │      → ExecutiveDecisionPortfolio          │
                └─────────────────────┬────────────────────┘
                                      │
                ┌─────────────────────▼────────────────────┐
                │       StrategicPortfolioEngine            │
                │                                           │
                │  1. compare_strategies()                  │
                │       4 postures → StrategyComparison     │
                │  2. budget_sweep()                        │
                │       risk / savings / ROI / coverage     │
                │       curves + knee + ROI-diminish        │
                │  3. pareto_frontier()                     │
                │       non-dominated (savings↑, risk↓)     │
                │  4. capital_efficiency()                  │
                │       per-dollar metrics + index          │
                │  5. recommend()                           │
                │       rule-based strategic messages       │
                │  6. confidence_assessment()               │
                │       stability · agreement · coverage    │
                └─────────────────────┬────────────────────┘
                                      │
                ┌─────────────────────▼────────────────────┐
                │   StrategicPortfolioPlan  (frozen, JSON)  │
                └──────────────────────────────────────────┘
```

The four strategic postures are realised as configurations of the Phase-3 engine: risk-first weights risk at 0.70 in the priority score, criticality-first weights criticality at 0.70, ROI-first uses the greedy-ROI optimization strategy, and balanced uses the default hybrid weights. The budget is supplied per call.

---

## 3. Core Capabilities

### 3.1 Strategy Comparison Engine

`compare_strategies(snapshot, budget)` runs all four postures at a fixed budget and returns a `StrategyComparison`. Each posture yields a `StrategyResult` carrying the selected assets, total cost, expected savings, risk reduction (absolute and percentage), ROI, and coverage. The comparison also reports which posture is best by each criterion and an **agreement** score — the mean pairwise Jaccard overlap of the four selected sets, in `[0, 1]`.

When risk, ROI, and criticality are correlated (the worst asset is worst on every axis), the postures agree and agreement approaches 1 — itself a confidence signal. When the signals are anti-correlated (a high-risk asset with low savings versus a low-risk asset with high savings), the postures diverge and select different assets; the test suite verifies both regimes explicitly.

### 3.2 Budget Sweep Analysis

`budget_sweep(snapshot, posture=)` evaluates the configured posture across the budget grid `budget_levels()` — for example 100k, 200k, 300k, 400k, 500k — and returns a `BudgetSweep` with four chart-ready response curves: **risk reduction**, **savings**, **ROI**, and **coverage**, plus the per-budget `BudgetSweepPoint` detail. The sweep also computes the capital-efficiency **knee** and the **ROI-diminish budget** (below).

A representative sweep on a graded fleet shows the classic diminishing-returns shape: risk reduction climbing 10% → 70%, savings rising monotonically, ROI declining 7.9 → 5.5, and coverage filling 10% → 100% as the budget grows.

### 3.3 Pareto Frontier Engine

`pareto_frontier(sweep)` identifies the **non-dominated** portfolios. Per the spec, a portfolio dominates another when it has **higher savings AND lower residual risk** (with at least one strict inequality); residual risk is tracked as the negative of risk reduction, so "lower risk" is equivalent to "more risk removed". Each candidate becomes a `ParetoPortfolio` storing `portfolio_id`, `budget`, `risk`, `savings`, and `roi`.

A subtle but important property: a *single-strategy monotone sweep* — where each higher budget yields both more savings and less risk — collapses under this rule to a single non-dominated point (the top budget). This is mathematically honest, not a defect: with one strategy, every cheaper portfolio is genuinely dominated. The frontier becomes rich when candidates come from *multiple strategies* at varying budgets, because different postures trade savings against risk differently at the same spend. The knee of the frontier is found by the same Kneedle heuristic used on the sweep curve.

### 3.4 Capital Efficiency Analytics

`capital_efficiency(sweep, budget)` computes, for the portfolio nearest a given budget, three per-dollar metrics — **risk_reduction_per_dollar**, **savings_per_dollar**, **coverage_per_dollar** — and a composite **maintenance_efficiency_index**. The index scales each per-dollar metric by the sweep's best-in-class value and averages them, yielding a `[0, 1]` score where 1 means best-in-sweep on all three dimensions. This lets leadership compare the marginal productivity of capital across budget levels on a single normalised scale.

### 3.5 Strategic Recommendation Engine

`recommend(sweep, current_budget=)` generates rule-based, fully deterministic recommendations. Three rules fire:

1. **Budget vs knee** — comparing the current budget to the capital-efficiency knee yields one of: "Current budget … lies before the Pareto knee point", "… lies beyond the knee", or "… is at the knee".
2. **Unlock opportunity** — when a budget increase to the knee unlocks proportionally more risk reduction than the cost increase (governed by `unlock_ratio`), it emits "Increase budget by X% to unlock Y% additional risk reduction."
3. **ROI diminishing** — when ROI falls below `roi_diminish_fraction` of its peak, it emits "ROI diminishes beyond $X."

These map directly onto the spec's example phrasings. The engine also composes a natural-language `executive_summary`.

### 3.6 Confidence Assessment

`confidence_assessment(comparison, sweep, reference_budget)` combines three signals into a `[0, 1]` score:

> confidence = 0.40·portfolio_stability + 0.40·strategy_agreement + 0.20·coverage

- **portfolio_stability** — the mean Jaccard overlap of consecutive sweep selections; a selection that stays consistent as the budget grows is more trustworthy than one that churns;
- **strategy_agreement** — the four postures' mean pairwise overlap at the reference budget;
- **coverage** — fleet coverage at the reference budget.

---

## 4. KPI Definitions

| KPI | Definition |
|-----|------------|
| `risk_reduction_curve` | absolute mean-risk reduction at each budget |
| `savings_curve` | total expected savings at each budget |
| `roi_curve` | portfolio ROI at each budget |
| `coverage_curve` | fraction of fleet selected at each budget |
| `knee_budget` | budget at the max-curvature point of the risk-reduction curve |
| `roi_diminish_budget` | first budget where ROI < fraction × peak ROI |
| `risk_reduction_per_dollar` | risk reduction / cost |
| `savings_per_dollar` | savings / cost |
| `coverage_per_dollar` | coverage / cost |
| `maintenance_efficiency_index` | mean of the three per-dollar metrics, scaled to sweep-best, in `[0, 1]` |
| `agreement` | mean pairwise Jaccard of the four postures' selections |
| `confidence.score` | 0.40·stability + 0.40·agreement + 0.20·coverage |

---

## 5. Executive Workflows

**Setting the annual maintenance budget.** A capital owner runs `optimize()` over a budget range, reads the knee budget as the recommended spend, and uses the recommendation messages to justify it — increasing toward the knee where marginal risk reduction is strong, stopping where ROI flattens.

**Posture selection.** An operations director compares the four postures at a candidate budget; high agreement means the choice of posture is low-stakes, while divergence flags a genuine strategic decision between chasing risk, return, or criticality.

**Board capital justification.** The Pareto frontier, capital-efficiency index, and confidence score together form a defensible, quantified case for a budget request — showing the efficient trade-offs available and how much confidence the recommendation carries.

---

## 6. Engineering Standards

The engine meets the platform standard in full: eleven frozen, validated dataclasses, each with a `to_dict`; the `register` / `build` / `list` registry triad under `STRATEGY_ENGINE_REGISTRY`; failure-safe tracker integration; and pure-NumPy numerics with **no pandas, SciPy, OR-tools, or ML libraries** — the knee detection, Pareto dominance, Jaccard overlaps, and capital-efficiency math are all hand-implemented in NumPy. Outputs are JSON-serialisable with non-finite floats rendered as `null`. Every analysis is deterministic in the snapshot and parameters, with stable tie-breaking. A CLI `--demo` exercises the full pipeline on a synthetic forty-turbine fleet.

---

## 7. Validation Summary

210 tests pass with zero skips, covering the posture enum and config validation, `budget_levels` grid construction, strategy comparison (including posture divergence under anti-correlated signals), the budget sweep and its four curves, Pareto dominance and the non-dominated set, capital-efficiency formulas and the bounded index, all three recommendation rules, the confidence components, the top-level `optimize` orchestration, serialization, determinism, failure-safe tracker integration, and edge cases (empty, single, and 50-asset fleets; zero and very large budgets; single-budget sweeps; flat, linear, and concave curves for the knee detector). The six frozen prior suites continue to pass unchanged, and all twelve platform modules import and coexist cleanly.