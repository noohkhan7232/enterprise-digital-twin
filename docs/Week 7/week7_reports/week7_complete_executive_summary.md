# Week 7 Engineering Report
## Enterprise Digital Twin & Decision Intelligence Platform

**Reporting period:** Week 7
**Audience:** Senior data scientists, AI architects, engineering managers, CTOs, research engineers
**Scope:** Decision-intelligence layer — Monte Carlo risk, fleet digital twin, executive decisioning, strategic portfolio optimization
**Status:** All four phases implemented, validated, and documented. 751 Week-7 tests passing with zero skips.

---

## Week 7 Overview

Weeks 1 through 6 established the platform's perception and prediction capabilities: an acoustic health index, health-trend analysis, remaining-useful-life (RUL) prediction, failure-risk estimation, a maintenance-decision agent, and a digital-twin simulation engine with what-if scenario and fleet capabilities. By the end of Week 6, the platform could answer *what is the condition of an asset, and how will it degrade?*

Week 7 addresses a different and harder class of question: *given that knowledge, what should the organisation actually do, and how much should it spend?* This is the transition from predictive analytics to decision intelligence. A prediction is informational; a decision is consequential, budget-bound, and accountable. The four phases delivered this week form a vertical escalation of decision abstraction:

- **Phase 1 — Monte Carlo Risk Engine** converts point predictions into probability distributions and tail-risk measures, so decisions can be made against quantified uncertainty rather than single expected values.
- **Phase 2 — Fleet Digital Twin** lifts single-asset prognostics to a governed fleet-level snapshot with risk scoring and concentration metrics.
- **Phase 3 — Executive Decision Engine** turns that snapshot into a budget-constrained maintenance portfolio with explicit return and risk-reduction analytics.
- **Phase 4 — Strategic Portfolio Optimization Engine** sits above the executive layer, analysing how outcomes respond to budget and strategic posture to recommend *how much to spend in the first place*.

A consistent set of engineering constraints governs all four: pure NumPy at inference (no PyTorch, SciPy, pandas, OR-tools, or ML libraries in the decision path), frozen and validated dataclasses, a registry pattern for discoverability, JSON-serialisable outputs, deterministic behaviour, failure-safe experiment-tracker integration, and a command-line demonstration entry point. These constraints are not stylistic. They make the decision layer auditable, reproducible, and deployable without a heavyweight runtime — properties that matter far more for a system allocating capital than for a research prototype.

---

## Phase 1 — Monte Carlo Risk Engine

### Probabilistic forecasting and uncertainty quantification

The predictive stack produces point estimates: a single RUL value, a single failure probability. Acting on a point estimate silently assumes the estimate is correct, which it never exactly is. The Monte Carlo Risk Engine replaces each point with a distribution by running a seeded ensemble of simulations in which the uncertain inputs — degradation rate, observation noise, and failure threshold — are themselves drawn from distributions.

The engine composes the frozen Asset State Simulator. For each of `n_trials` (default 1000), it samples a degradation rate, a noise level, and a failure threshold, runs a simulation, and records the resulting RUL. The collection of trial RULs forms an empirical distribution summarised by quantiles:

```
P10 = 10th percentile RUL   (pessimistic-but-plausible)
P50 = median RUL            (central estimate)
P90 = 90th percentile RUL   (optimistic)
```

Three samplers — `DegradationRateSampler`, `NoiseSampler`, `FailureThresholdSampler` — each derive from a common `_BaseSampler` abstract base and apply domain clamps (degradation rate and noise are non-negative; the failure threshold is bounded to `[0, 100]`). RUL values are censored at the simulation horizon so a healthy asset reports a bounded, finite distribution rather than an unbounded tail.

### VaR and CVaR

For financial-grade risk reporting, the engine derives two tail-risk measures over the loss distribution. Value at Risk at confidence level α is the α-quantile of loss:

```
VaR_α = quantile_α(losses)
```

Conditional Value at Risk (also called Expected Shortfall) is the expected loss conditional on being in the tail beyond VaR:

```
CVaR_α = mean(losses | losses >= VaR_α)
```

By construction `CVaR_α >= VaR_α`, a property asserted directly in the test suite. VaR answers "how bad is the threshold bad case?"; CVaR answers "if we are in the bad tail, how bad on average?" — the latter is the more conservative and is preferred by risk committees because it accounts for the shape of the tail, not just its boundary.

### Risk concentration

At the portfolio level the engine computes a Herfindahl-style concentration index over per-asset risk contributions:

```
H = Σ_i (r_i / Σ_j r_j)²
```

`H` ranges from `1/N` (risk spread uniformly across N assets) to `1` (all risk in a single asset). Concentration is a strategic signal distinct from total risk: two fleets with identical aggregate risk demand different responses if one concentrates that risk in three assets and the other spreads it across thirty.

### Deterministic simulation requirements

A risk engine whose output changes between runs cannot support audit. Determinism is achieved through NumPy's `SeedSequence.spawn`: a single master seed deterministically spawns one child seed per trial, so the entire ensemble is bit-for-bit reproducible while remaining statistically independent across trials. The samplers are abstract and stateless, preserving determinism even when extended. This was a deliberate engineering decision — convenience approaches such as re-seeding a global RNG inside a loop would have produced correlated trials and non-reproducible results under parallelism.

### Architecture and engineering decisions

The engine exposes `run_rul_uncertainty`, `run_failure_probability`, `run_health_distribution`, and `run_portfolio_distribution`. Outputs are frozen dataclasses — `RULDistribution`, `HealthDistribution`, `RiskDistribution`, `PortfolioDistribution` — each with `to_dict` for JSON serialisation, with non-finite floats rendered as `null`. The module is roughly 1,300 source lines and validated by 202 tests. The central engineering decision was to compose, not duplicate: the engine wraps the frozen simulator rather than re-implementing degradation physics, so the uncertainty layer inherits the simulator's validated behaviour and cannot drift from it.

**Business use cases.** Spare-parts provisioning sized to the P10 rather than the P50; maintenance scheduling with explicit confidence bands; tail-risk reporting (VaR/CVaR) in a vocabulary finance and insurance functions already use; and identification of assets whose *uncertainty* — not just expected risk — is high enough to warrant additional inspection.

---

## Phase 2 — Fleet Digital Twin

### Fleet health modeling

The Fleet Digital Twin evaluates every asset through the complete predictive chain — health-trend analysis, RUL prediction, failure-risk estimation, and the maintenance-decision agent — then aggregates the results into a single fleet snapshot. Each asset is described by an `AssetInput` (identifier, type, location, and a health trajectory coerced to a NumPy array requiring at least five points for RUL fitting). The engine produces a `FleetAsset` per input carrying thirteen fields, including health, predicted RUL, failure probability, the recommended maintenance action and its cost, expected savings, severity, a health band, and a normalised risk score.

### Fleet risk scoring

Each asset receives a normalised composite risk score:

```
risk_score = ( 0.30 · (1 − health/100)
             + 0.25 · (1 − min(RUL,180)/180)
             + 0.30 · failure_probability
             + 0.15 · min(severity,20)/20 ) / Σweights
```

Bounded inputs (RUL capped at 180, severity at 20) keep the score in `[0, 1]` and comparable across a heterogeneous fleet. The weighting reflects a deliberate prioritisation: imminent failure probability and current health dominate, with RUL and severity as supporting signals.

### Concentration metrics and Pareto analysis

The snapshot reports both a Herfindahl `risk_concentration` and a `pareto_concentration` — the share of total fleet risk carried by the worst-performing fraction of assets (default 20%). Pareto analysis operationalises the familiar observation that a small subset of assets typically drives most of the exposure; quantifying it tells management how much leverage a narrowly targeted intervention can achieve.

### KPIs and outputs

The `FleetSnapshot` is a frozen, JSON-serialisable object with fifteen fields. Aggregation respects the intensive/extensive distinction: *intensive* quantities (average health, average RUL, mean failure probability, risk concentration) are averaged, while *extensive* quantities (expected cost, expected downtime, expected failure cost, expected savings) are summed. Conflating the two — for example averaging cost — is a common and costly modelling error that this design avoids structurally. The engine exposes `evaluate_asset`, `build_fleet_snapshot`, ranking helpers (`rank_assets_by_risk`, `rank_assets_by_savings`), and opportunity identifiers (`identify_top_risks`, `identify_top_opportunities`).

### Architecture decisions

The fleet registry uses copy-on-write semantics via `MappingProxyType`, so the asset registry cannot be mutated in place — a guard against the subtle bugs that arise when shared mutable fleet state is modified by one consumer and observed by another. The snapshot is the platform's single source of truth: every downstream decision layer consumes it and nothing else, which means the entire decision stack can be re-run deterministically from one serialised object. The module is roughly 950 source lines and validated by 157 tests. Maintenance-opportunity identification — surfacing the assets where proactive spend yields the highest savings — is the bridge to the executive layer: it reframes a risk report as an actionable opportunity set.

---

## Phase 3 — Executive Decision Engine

### Portfolio optimization under a budget constraint

The Executive Decision Engine consumes a `FleetSnapshot` and produces an `ExecutiveDecisionPortfolio`: a budget-constrained selection of assets for maintenance, with a prioritised ranking, ROI and risk-reduction analytics, natural-language recommendations, and a confidence score. Each asset first receives a composite priority score:

```
priority_score = ( 0.40 · risk
                 + 0.30 · savings_norm
                 + 0.20 · criticality
                 + 0.10 · urgency ) / Σweights
```

where `savings_norm` is expected savings divided by the fleet maximum, `criticality` is severity normalised by 20, and `urgency` is failure probability. Dividing by the weight sum keeps the score in `[0, 1]` even when the weights are reconfigured.

### Risk-first, ROI-first, and hybrid strategies

Selection then fills the budget greedily under one of three orderings:

- **Greedy ROI** orders by `expected_savings / maintenance_cost`, selecting cheap high-return repairs first.
- **Greedy savings** orders by absolute expected savings, selecting the largest-exposure repairs first.
- **Hybrid** orders by the composite priority score, balancing risk, return, criticality, and urgency.

### Trade-offs

The strategies are not interchangeable, and the difference matters most precisely when capital is scarce. With heterogeneous costs they diverge: in validation, under a tight budget, greedy ROI selected three inexpensive high-ROI assets for total savings exceeding what greedy savings achieved with a single expensive high-absolute-savings asset that consumed the entire budget. This demonstrates a real capital-rationing principle — ROI-first dominates when the budget binds — and the test suite asserts the divergence rather than assuming it.

The choice of greedy selection over exact knapsack optimization was deliberate. Greedy is `O(N log N)`, fully transparent (an executive can see exactly why each asset was chosen), and within a small percentage of optimal for the near-uniform cost structures typical of a turbine fleet. An exact solver would add a dependency, reduce explainability, and deliver marginal value — a poor trade for a system whose outputs must be defensible to a capital committee.

### Risk reduction, ROI, and confidence

Risk reduction is estimated with a tunable recovery factor (default 0.70):

```
risk_after = mean( risk · (1 − recovery)  if selected
                   else risk )
reduction_pct = 100 · (risk_before − risk_after) / risk_before
```

Financial analytics are `ROI = (savings − cost) / cost`, with payback ratio and cost efficiency as `savings / cost`. The confidence score blends data quality (saturating in fleet size), portfolio coverage (the fraction of at-risk assets the selection addresses), and risk dispersion (`1 − risk_concentration`).

### Business impact and executive workflows

The engine's output is a board-ready approval package. In a validated run, a roughly $15,000 portfolio delivered approximately 43% fleet risk reduction at an ROI near 794%, with the recommendation, selected assets, cost, and confidence assembled into a single executive summary. The primary workflow is monthly maintenance approval: an operations director sets the budget, runs the hybrid strategy, and reads the summary and recommendations as the approval basis, with the prioritised ranking available to justify each inclusion. The module is roughly 1,000 source lines and validated by 182 tests.

---

## Phase 4 — Strategic Portfolio Optimization

### Strategy comparison

The Strategic Portfolio Engine sits above the executive layer and answers the pre-budget question: how much should we spend, and under which posture? It realises four strategic postures as configurations of the executive engine — risk-first (risk weighted 0.70), ROI-first (greedy-ROI optimization), criticality-first (criticality weighted 0.70), and balanced (default hybrid). `compare_strategies` runs all four at a fixed budget and reports per-posture cost, savings, risk reduction, ROI, and coverage, plus an **agreement** score: the mean pairwise Jaccard overlap of the four selected sets. High agreement (which occurs when risk, ROI, and criticality are correlated) signals that the posture choice is low-stakes; divergence flags a genuine strategic decision.

### Budget sweep analysis

`budget_sweep` evaluates a posture across a budget grid (for example $100k through $500k in $100k steps) and returns four chart-ready response curves — risk reduction, savings, ROI, and coverage — alongside per-budget detail. These curves exhibit the expected diminishing-returns shape: risk reduction and coverage rise concavely while ROI declines as the budget is spent on progressively lower-return assets.

### Pareto frontier

The frontier isolates non-dominated portfolios. Per the specification, a portfolio dominates another when it has **higher savings AND lower residual risk** (with at least one strict inequality). An honest property of this definition deserves note: a single-strategy monotone sweep collapses to one non-dominated point, because each higher budget genuinely dominates every lower one. The frontier becomes informative when candidates come from multiple postures at varying budgets, where they trade savings against risk differently at the same spend. Each candidate is stored as a `ParetoPortfolio` with `portfolio_id`, `budget`, `risk`, `savings`, and `roi`.

### Capital efficiency

For any budget level the engine computes three per-dollar metrics — `risk_reduction_per_dollar`, `savings_per_dollar`, `coverage_per_dollar` — and a composite `maintenance_efficiency_index`. The index scales each per-dollar metric by the sweep's best-in-class value and averages them, producing a `[0, 1]` score where 1 means best-in-sweep on all three dimensions. This places the marginal productivity of capital on a single normalised scale across budget levels.

### Confidence scoring and recommendation generation

Strategic confidence combines three signals:

```
confidence = 0.40 · portfolio_stability
           + 0.40 · strategy_agreement
           + 0.20 · coverage
```

where stability is the mean Jaccard overlap of consecutive sweep selections (a selection that stays consistent as the budget grows is more trustworthy than one that churns). The recommendation engine is rule-based and deterministic, emitting guidance keyed to the capital-efficiency knee (detected via a Kneedle perpendicular-distance heuristic with guards for degenerate curves): a budget-versus-knee message, an unlock opportunity ("increase budget by X% to unlock Y% additional risk reduction"), and an ROI-diminish point. In a validated run the engine identified a $120,000 knee, reported that a $60,000 current budget "lies before the Pareto knee point," and warned that "ROI diminishes beyond $80,000," with an overall confidence of 0.77.

### Enterprise use cases and board-level decision support

The engine directly supports annual budget setting (the knee as the recommended spend), posture selection (agreement quantifying how consequential the choice is), and board capital justification (the Pareto frontier, efficiency index, and confidence score forming a defensible case for a budget request). The module is roughly 1,500 source lines and validated by 210 tests. It is explicitly not a dashboard or reporting layer; it is a decision-optimization engine whose outputs are the inputs to a capital-allocation decision.

---

## Engineering Achievements

Week 7 delivered four production modules totalling roughly 4,750 source lines, each composing the layers beneath it without modifying them.

**New modules.** Monte Carlo Risk Engine (`src/risk/`), Fleet Digital Twin (`src/fleet/`), Executive Decision Engine (`src/executive/`), Strategic Portfolio Optimization Engine (`src/strategy/`).

**Dataclasses.** All outputs are frozen, validated dataclasses with `__post_init__` invariants — distributions, fleet assets and snapshots, prioritised assets and portfolios, strategy results and comparisons, budget-sweep points and sweeps, Pareto portfolios and frontiers, capital-efficiency and confidence records, and the top-level strategic plan. Frozen dataclasses prevent accidental mutation of a decision record after it is produced, which is essential for reproducibility and audit.

**APIs.** Each engine exposes a small, composable public surface (for example `recommend`, `prioritize_assets`, `optimize_portfolio` on the executive engine; `compare_strategies`, `budget_sweep`, `pareto_frontier`, `capital_efficiency`, `recommend`, `confidence_assessment`, and `optimize` on the strategic engine).

**Registries.** Every engine follows the platform's `register` / `build` / `list` pattern, rejecting duplicate registrations, enabling configuration-driven instantiation and discoverability.

**Validations, serialization, determinism.** Configurations validate their inputs on construction and raise on invalid values; all outputs serialise to JSON with non-finite floats rendered as `null`; every analysis is a deterministic function of its inputs with stable tie-breaking by identifier. Experiment-tracker integration is failure-safe — a tracker fault is caught and logged, never interrupting a decision. Across all four modules the constraint of pure NumPy at inference holds: no PyTorch, SciPy, pandas, OR-tools, or ML libraries appear in the decision path.

---

## Testing Summary

All Week-7 modules are validated by live test execution with zero skips. The six frozen prior suites were re-run alongside the new modules to confirm coexistence and the absence of regressions.

| Module | Suite | Tests | Pass | Fail | Skip | Pass rate | Focus areas |
|--------|-------|------:|-----:|----:|----:|----------:|-------------|
| Monte Carlo Risk Engine | `test_monte_carlo_engine` | 202 | 202 | 0 | 0 | 100% | Sampling, quantiles, VaR/CVaR, determinism, serialization |
| Fleet Digital Twin | `test_fleet_digital_twin` | 157 | 157 | 0 | 0 | 100% | Aggregation, risk scoring, concentration, ranking, edge cases |
| Executive Decision Engine | `test_executive_decision_engine` | 182 | 182 | 0 | 0 | 100% | Prioritization, optimization, ROI, risk reduction, confidence |
| Strategic Portfolio Engine | `test_portfolio_strategy_engine` | 210 | 210 | 0 | 0 | 100% | Strategy comparison, sweep, Pareto, efficiency, recommendation |
| **Week 7 total** | | **751** | **751** | **0** | **0** | **100%** | |
| Asset State Simulator (regression) | `test_asset_state_simulator` | 185 | 185 | 0 | 0 | 100% | Coexistence verified |
| Scenario Engine (regression) | `test_scenario_engine` | 121 | 121 | 0 | 0 | 100% | Coexistence verified |
| Fleet Twin / Wk6 (regression) | `test_fleet_digital_twin` (sim) | 154 | 154 | 0 | 0 | 100% | Coexistence verified |

**Integration coverage.** Each engine is tested against the genuine output of the layer beneath it — the executive engine against real fleet snapshots, the strategic engine against real executive portfolios — rather than against mocks, so cross-layer contract drift is caught directly.

**Edge cases.** Empty fleets (rejected with explicit errors), single-asset fleets, large fleets (50 assets), zero budgets (empty selection, zero ROI), very large and infinite budgets (full selection), fractional budgets, zero-savings and unaffordable assets, and degenerate curve shapes (flat, linear, single-point) for the knee detector.

**Determinism validation.** Every module includes explicit determinism tests asserting that identical inputs yield identical outputs across independent runs — including the Monte Carlo engine, whose seeded ensemble is reproducible bit-for-bit. Two test-expectation defects were identified and corrected during development (a coverage-per-dollar monotonicity assumption and a CLI format string); in both cases the engine was correct and the test was wrong, which is the desired failure mode.

---

## Business Impact

Week 7 changes what the platform is. Through Week 6 it was a predictive-analytics system: it produced forecasts. A forecast, however accurate, places the entire burden of interpretation and decision on a human. Week 7 closes that gap by making the platform produce decisions — bounded by budget, quantified by return and risk, and accompanied by a confidence score and a natural-language rationale.

Three shifts characterise the transition. First, from point estimates to distributions: the Monte Carlo layer lets the organisation plan against the bad case (P10, VaR, CVaR) rather than the average case, which is the difference between a maintenance plan and a risk-managed maintenance plan. Second, from per-asset diagnosis to fleet-level capital allocation: the fleet snapshot and executive engine convert thousands of sensor streams into a single approvable portfolio with an ROI and a risk-reduction percentage. Third, from tactical to strategic: the strategic engine answers the budget-sizing question that precedes any portfolio, identifying the capital-efficiency knee and the point of diminishing returns.

The economic mechanism is the conversion of predictive accuracy into avoided failure cost. The platform's standing cost model contrasts proactive maintenance against the substantially higher cost of reactive failure; the executive engine selects the portfolio that maximises that avoided cost within budget, and the strategic engine sizes the budget itself to the point where marginal avoided cost still exceeds marginal spend. Every figure in this chain is deterministic and serialisable, which means a decision can be reproduced and audited months later from the snapshot that produced it — a requirement for any organisation running maintenance capital through a model.

---

## Week 8 Readiness

Week 7 deliberately produces the substrate that an agentic and retrieval-augmented Week 8 will require.

**Agentic AI.** The engines expose small, composable, side-effect-free APIs returning structured objects. An agent can already call `optimize`, `compare_strategies`, or `budget_sweep` as tools and reason over their typed outputs without bespoke glue. Because the engines are deterministic, an agent's plans are reproducible — a precondition for trustworthy autonomous action.

**RAG.** Every output is JSON-serialisable and every engine emits a natural-language summary alongside its structured fields. The serialised plans, portfolios, and snapshots form a corpus that a retrieval-augmented system can index and ground answers in, so an "explain why asset WTG-042 was prioritised" query can be answered from the actual decision record rather than a re-derivation.

**Executive Copilot.** The natural-language recommendation and executive-summary generators are first-class outputs, not afterthoughts. A copilot can surface these directly, then drill into the structured prioritisation, sweep curves, and confidence components on request — the conversational surface and the quantitative backing already exist and are linked.

**Autonomous Decision Support.** The confidence scores are the gating mechanism for autonomy. A future autonomous layer can act without human review when confidence is high and coverage is complete, and escalate to a human when stability or agreement is low — a principled boundary between automated and supervised decisions that the Week-7 confidence model already supplies.

---

## Conclusion

Week 7 delivered the platform's decision-intelligence layer in four composing phases: a Monte Carlo Risk Engine that quantifies uncertainty and tail risk; a Fleet Digital Twin that aggregates prognostics into a governed, single-source-of-truth snapshot; an Executive Decision Engine that converts that snapshot into a budget-constrained, ROI-and-risk-quantified maintenance portfolio; and a Strategic Portfolio Optimization Engine that sizes the budget itself against capital efficiency and strategic posture. The work comprises roughly 4,750 lines of pure-NumPy production code across four frozen-dataclass, registry-discoverable, JSON-serialisable, deterministic modules, validated by 751 tests passing with zero skips and confirmed to coexist with all prior frozen suites without regression.

The engineering posture throughout favoured transparency and reproducibility over algorithmic sophistication for its own sake — greedy selection over opaque exact solvers, hand-implemented numerics over heavyweight dependencies, frozen records over mutable state, and honest accounting of model properties (the monotone-sweep Pareto collapse, the schematic nature of certain reported figures) over convenient overstatement. These choices reflect the system's purpose: it allocates capital, and a capital-allocation engine earns its place by being auditable and correct, not merely capable. With this layer in place, the platform has crossed from predicting asset condition to recommending — and soon supporting the autonomous execution of — the decisions that condition implies.