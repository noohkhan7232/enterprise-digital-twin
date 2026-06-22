# Week 7 — Phase 1: Monte Carlo Risk Intelligence Engine

## Architecture Notes, Engineering Notes & Completeness Report

**Component:** `src/risk/monte_carlo_engine.py`
**Role:** Uncertainty quantification over the platform's deterministic prognostics
**Status:** Implemented, validated — 202 tests passing; the frozen Week-1–6 modules are unchanged and their suites (185 + 121 + 154) still pass.

---

## 1. Architecture Notes

### 1.1 Purpose

The Week-1–6 platform produces *point estimates*: a single RUL, a single failure probability, a single fleet risk. Real assets are uncertain — degradation rates vary with duty and environment, the health signal carries observation noise, and the failure threshold itself is an engineering judgement, not a constant. This engine propagates that input uncertainty through the frozen simulator and prognostic chain by Monte Carlo sampling, converting `RUL = 42 days` into a distribution (`P10 = 28, P50 = 43, P90 = 67`) with full risk metrics, portfolio analytics, and chart-ready arrays.

### 1.2 Composition over modification

The engine is a strict additive composition of the frozen lower layers, the same discipline that governed Weeks 6.2 and 6.3. Each trial constructs a frozen `AssetSimulatorConfig` (via `dataclasses.replace` on the asset baseline, substituting the sampled rate, noise, and threshold), runs the frozen `AssetStateSimulator`, and records the outcome. The engine modifies no prior code; the simulator's 185 tests, the scenario engine's 121, and the fleet engine's 154 all continue to pass unchanged. The Monte Carlo capability is layered *over* the validated stack, not *into* it.

### 1.3 Layer structure

```
MonteCarloConfig (frozen, validated)
    +  three uncertainty samplers (normal / uniform / triangular)
        ▼
MonteCarloEngine
    ├── _trial_seeds()    SeedSequence.spawn → deterministic per-trial seeds
    ├── _sample_params()  per-trial rate / noise / threshold (degenerate default)
    ├── _run_trials()     frozen AssetStateSimulator per trial → RUL/health/fail
    ├── run_rul_uncertainty()        → RULDistribution
    ├── run_failure_probability()    → RiskDistribution (VaR / CVaR / loss)
    ├── run_health_distribution()    → HealthDistribution
    └── run_portfolio_distribution() → PortfolioDistribution
        ▼
compute_statistics() · value_at_risk() · conditional_value_at_risk()
build_visualization()  → histogram / cdf / survival / risk curve
```

### 1.4 Uncertainty samplers

Three samplers isolate the sources of input uncertainty, each supporting `normal`, `uniform`, and `triangular` distributions with construction-time validation. `DegradationRateSampler` and `NoiseSampler` clamp draws to non-negative (a negative rate or standard deviation is unphysical); `FailureThresholdSampler` clamps to `[0, 100]` (the health range). When a sampler is omitted, the engine substitutes the asset's configured constant, so an engine with no samplers degenerates to a deterministic ensemble around the baseline — useful as a control and for isolating the effect of a single uncertainty source.

### 1.5 Determinism

A single master seed deterministically derives per-trial seeds through `numpy.random.SeedSequence.spawn`, so an entire ensemble is reproducible bit-for-bit. Parameter sampling uses a separate generator seeded from the master seed, keeping parameter draws independent of the per-trial simulation seeds. The result is that two engines with the same configuration produce identical distributions — validated explicitly across all four API methods.

---

## 2. Design Decisions

**Loss-based VaR and CVaR.** Value-at-Risk and Conditional VaR are computed over a *loss* distribution: each trial contributes `failure_cost` if it fails within the horizon and `0` otherwise. VaR at confidence *c* is the *c*-th percentile of that loss; CVaR is the mean of the losses at or beyond VaR — the expected loss in the worst `(100−c)%` of outcomes. CVaR is therefore always ≥ VaR, a property the tests assert. This loss-based formulation is the financial-risk convention and is the correct one for an enterprise that thinks in monetary exposure.

**Censoring at the horizon.** A trial that does not reach the failure threshold within the simulation horizon has no observed failure time; its RUL is *censored* at the horizon rather than recorded as infinite or discarded. This keeps the RUL distribution well-defined and bounded, and it is the statistically honest treatment of a survivor — we know its life exceeds the horizon, not that it equals any particular value.

**Risk concentration as a Herfindahl index.** Portfolio risk concentration is the sum of squared normalised per-asset failure probabilities — the Herfindahl–Hirschman index applied to failure risk. It ranges from `1/N` (risk spread uniformly across the fleet) to `1` (all risk in a single asset), giving leadership a single scalar for whether fleet risk is diversified or dangerously concentrated. This is a deliberate reuse of a well-understood economic concentration measure rather than an ad-hoc metric.

**Intensive vs extensive portfolio aggregation.** Consistent with the Week-6 fleet engine, portfolio risk is the *mean* of per-asset failure probabilities (intensive), while the expected fleet-failure count is their *sum* (extensive). The engine additionally reports the full distribution of the per-trial failure *count*, so leadership sees not just the expected number of fleet failures but its spread.

**Chart-ready arrays, subsampled.** The visualization builder emits histogram bins, an empirical CDF, a survival curve, and a risk curve. The CDF/survival curves are subsampled to a bounded number of points so the JSON payload stays small for large ensembles without distorting the curve shape — a pragmatic memory/fidelity trade-off for dashboard consumption.

---

## 3. Engineering Notes

The engine adheres to the platform standard in full: a frozen, validated `MonteCarloConfig`; the `register` / `build` / `list` registry triad under `MONTE_CARLO_ENGINE_REGISTRY`; failure-safe tracker integration (every tracker call wrapped so a tracker fault can never interrupt an ensemble); pure-NumPy numerics with no SciPy or PyTorch; deterministic RNG via `SeedSequence`; full Google-style docstrings; and JSON-serialisable outputs in which every non-finite float (an undefined expected-failure-time when no trial fails, for instance) renders as `null`. There is no global mutable state — the registry is the only module-level dictionary and it holds class registrations, not run state.

A note on the computational model: the per-trial loop is serial, and the `parallel_ready` config flag documents that the workload is embarrassingly parallel (each trial is independent and the simulator is pure) without yet implementing a parallel backend. Because each trial's seed derives independently from the master seed, a future process-pool backend would preserve determinism exactly — parallelism is a backend substitution, not a re-architecture. For very large ensembles the `store_trials=False` default keeps only the aggregated statistics, bounding memory; raw trials are retained only when explicitly requested.

---

## 4. Completeness Report

| # | Requirement | Status | Evidence |
|--:|-------------|--------|----------|
| 1 | `MonteCarloConfig` (frozen, 5 fields, validation) | Done | `n_trials`, `random_seed`, `confidence_levels`, `parallel_ready`, `store_trials` + validation in `__post_init__` |
| 2 | Three samplers, normal/uniform/triangular | Done | `DegradationRateSampler`, `NoiseSampler`, `FailureThresholdSampler` with all three distributions + clamping |
| 3 | Engine API (4 methods) | Done | `run_rul_uncertainty`, `run_failure_probability`, `run_health_distribution`, `run_portfolio_distribution` |
| 4 | Four distribution outputs | Done | `RULDistribution`, `HealthDistribution`, `RiskDistribution`, `PortfolioDistribution` |
| 5 | Statistics (mean…P95) | Done | `DistributionStatistics` with mean, median, std, variance, P5–P95 |
| 6 | Risk metrics | Done | P(fail), expected failure time, expected downtime, expected loss, VaR, CVaR |
| 7 | Portfolio analytics | Done | portfolio risk, worst-asset probability, expected fleet failures, risk concentration |
| 8 | Visualization data | Done | histogram, CDF, survival curve, risk curve (`build_visualization`) |
| 9 | Registry | Done | `MONTE_CARLO_ENGINE_REGISTRY`, `register_monte_carlo_engine`, `build_monte_carlo_engine` |
| 10 | 200+ tests | Done | **202 tests, 100% pass, 0 skipped** |

**Architecture standards:** frozen dataclasses ✓ · immutable configs ✓ · registry pattern ✓ · failure-safe tracker ✓ · pure NumPy ✓ · deterministic RNG ✓ · full docstrings ✓ · JSON serializable ✓ · no global mutable state ✓

**Frozen-module guarantee:** `asset_state_simulator.py`, `scenario_engine.py`, and `fleet_digital_twin.py` unchanged; their 185 + 121 + 154 tests still pass. 9/9 platform modules import and coexist cleanly.

**Deliverable summary:** `monte_carlo_engine.py` (1,283 lines, pure NumPy) · `test_monte_carlo_engine.py` (202 tests) · this notes document.

---

## 5. Worked Example (from the CLI demo)

For a turbine degrading at ~0.8 health/day over a 120-day horizon, with degradation-rate uncertainty (normal, σ=0.2), noise uncertainty (uniform 0–1.5), and threshold uncertainty (triangular 25/30/35), 2,000 trials yield:

- **RUL:** P10 ≈ 62, P50 ≈ 83, P90 ≈ 119 days; P(failure within horizon) ≈ 0.90.
- **Risk:** expected loss ≈ \$45,000; VaR(95%) = \$50,000; CVaR(95%) = \$50,000.
- **Fleet (3 assets):** portfolio risk ≈ 0.90; expected failures ≈ 2.7; concentration ≈ 0.33 (risk spread across the fleet).

The deterministic platform would have reported a single RUL number; the Monte Carlo engine reports the distribution leadership needs for risk-adjusted decisions.

---

## 6. Week 7 Forward Look

This Phase-1 engine quantifies uncertainty for single assets and aggregates per-asset failure probabilities into portfolio risk. Subsequent Week-7 phases can build distributional fleet answers directly atop these outputs — P10/P50/P90 of *portfolio* cost and downtime, fleet-level VaR/CVaR on the aggregated loss distribution, and the probability that fleet failures exceed a budgeted threshold — by composing this engine's per-asset distributions with the Week-6 fleet aggregation, again without modifying any frozen layer.