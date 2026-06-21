# Principal Engineer Design Review — Scenario Engine

## Scaling Assessment: Single-Asset What-If → Enterprise Fleet Digital Twin

**Component under review:** `src/simulation/scenario_engine.py` (1,020 lines, 121 tests passing)
**Review type:** Architecture review only — no implementation changes
**Reviewer role:** Principal Digital Twin Architect / Enterprise Software Architect
**Verdict in one line:** A correct, well-factored *single-asset* component with clean extension seams, but missing every *fleet-scale* abstraction; it is a strong foundation, not a fleet platform.

---

## A. Architecture Review

### What the component does well (and must be preserved)

The engine is genuinely well-built *for its current scope*. The baseline-vs-counterfactual structure is clean, the `scenario − baseline` delta convention is consistent, the `failure_avoided`/`failure_induced` flags correctly handle the one-sided-failure case, non-finite values are serialised safely, and the two transform mechanisms (config-mutation vs trajectory-overlay) keep the validated Step-1 simulator untouched. The `compare()` method is stateless and pure, and `ScenarioResult` is an immutable value object. These four properties — statelessness, immutability, pure deltas, and graceful prognostics degradation — are exactly the seams a fleet layer needs, so the refactor ahead is *additive*, not corrective.

### The fundamental scaling boundary

The engine's entire mental model is **one asset, one baseline, one scenario at a time**. Every signature encodes this assumption:

- `ScenarioEngine.__init__(baseline_config)` binds the engine to a *single* `AssetSimulatorConfig`.
- `run(scenario)` takes *one* `ScenarioConfig` and returns *one* `ScenarioResult`.
- `ScenarioResult` has no asset identity — there is no `asset_id`, no fleet membership, no grouping key.
- `generate_report(results)` accepts a list, but it is a list of scenarios *for the same single asset*, not a fleet.

A fleet digital twin inverts this. It is **N assets × M scenarios × K Monte Carlo trials**, aggregated into *portfolio* quantities. None of those three multiplicities exist in the current type system. The word "portfolio" appears in the code, but only inside a docstring and a report header — there is no `PortfolioResult`, no aggregation function, no asset collection. That gap between the vocabulary and the abstractions is the central finding of this review.

### Computational model

The engine is **serial and re-simulating**. Each `run()` executes at least two full simulations, and for overlay scenarios (maintenance, sudden fault) the baseline is simulated **twice** — once in `run()` and again inside `_simulate_scenario` before the overlay is applied. For `NORMAL`, the baseline is simulated a second time only to be compared against itself. At single-asset scale this waste is invisible. At a fleet of 5,000 turbines × 8 scenarios × 1,000 Monte Carlo trials = 40M `run()` calls, it is 80M+ redundant simulations and a multi-hour serial job. There is no parallelism, no batching, no vectorisation, and no caching of the shared baseline.

---

## B. Missing Components

The following abstractions do not exist and are prerequisites for fleet scale. Each is additive.

| # | Missing component | Why it blocks fleet scale |
|--:|-------------------|---------------------------|
| 1 | **Asset identity** (`asset_id`, asset metadata on config and result) | Without it, results cannot be grouped, joined to a fleet registry, or attributed back to a physical machine |
| 2 | **`FleetConfig` / asset collection** | No type represents "the set of assets under management"; the engine binds to exactly one config |
| 3 | **`ScenarioBatch`** | No way to express "apply this scenario set across these assets" as a single schedulable unit |
| 4 | **`PortfolioResult` aggregator** | No type holds fleet-level rollups; `generate_report` formats single-asset rows only |
| 5 | **Portfolio risk aggregation** | Per-asset `risk_delta` exists, but no fleet expected-failure-count, no correlated-risk handling, no worst-N identification |
| 6 | **Portfolio cost aggregation** | `cost_impact` is per-asset and per-event; no fleet maintenance-budget rollup, no discounting, no multi-event streams |
| 7 | **Portfolio downtime / availability** | `downtime_impact` is per-asset hours; no fleet availability %, no concurrent-outage or crew/crane-capacity constraint |
| 8 | **Scenario ranking engine** | No cross-scenario or cross-asset ranking by a configurable objective (cost, availability, risk-adjusted value) |
| 9 | **Decision-support layer** | No "recommended action per asset" rollup that chains the Week-5 maintenance agent across the fleet |
| 10 | **Monte Carlo ensemble runner** | `ScenarioResult` is single-trial; no `EnsembleScenarioResult`, no distributional summaries (P10/P50/P90, probability of `failure_avoided`) |
| 11 | **Parallel execution backend** | No `multiprocessing`/`concurrent.futures`/joblib; the workload is embarrassingly parallel but run serially |
| 12 | **Baseline caching / dedup** | The shared baseline is re-simulated every `run()`; no memoisation keyed on config + seed |
| 13 | **Streaming / out-of-core results** | `ScenarioResult` holds full `(N,)` health arrays in memory; a fleet ensemble would hold N×M×K arrays simultaneously with no streaming or summarisation-on-the-fly |
| 14 | **Persistence / result store** | Results live only in memory and `to_dict`; no columnar/Parquet sink for downstream BI or audit |

---

## C. Technical Debt

These are existing decisions that are *correct at single-asset scale* but become debt at fleet scale.

1. **Redundant baseline simulation (concrete, measurable).** `run()` simulates the baseline, then `_simulate_scenario` simulates it again for overlay and normal scenarios. This doubles the simulation count for those scenario types and will dominate fleet runtime. Severity: **high** at scale, zero at present.

2. **Engine bound to one baseline at construction.** `self.baseline_config` is a single config. A fleet needs the engine (or a fleet wrapper) to iterate configs; today that means constructing one `ScenarioEngine` per asset, which re-pays object-construction and logging overhead N times.

3. **Full trajectory retained in every result.** `ScenarioResult` carries `times`, `baseline_health`, `scenario_health`, and `health_delta` — four `(N,)` arrays. For fleet ensembles this is the memory bottleneck; most consumers need only the scalar deltas and a summary, not every per-sample value.

4. **Per-event point economics.** `_economics` charges one `failure_cost` per failed trajectory and one `maintenance_cost` per intervention. There is no discounting, no multi-event stream, and no shared-resource (crew, crane, vessel) contention — all of which are first-order at fleet scale. Documented as a known boundary in the Step-2 notes, but it remains debt for portfolio costing.

5. **Prognostics re-imported and re-instantiated per call.** `_prognostics` imports and constructs a fresh `RULPredictor` and `FailureRiskEngine` on every invocation (twice per `run()`). Cheap once; wasteful across millions of calls. The engines could be constructed once and reused.

6. **No asset identity anywhere in the type system.** This is debt because retrofitting an identity field into a frozen result later is a breaking change to `to_dict` consumers; it would have been cheaper to include from the start.

7. **`generate_report` returns a formatted string, not structured data.** Fine for a CLI demo, but a fleet decision layer needs machine-readable rankings, not a pre-rendered text table. The presentation and the aggregation are entangled.

---

## D. Refactoring Recommendations

All recommendations are **additive and backward-compatible**; none requires modifying the single-asset path that 121 tests currently guard.

**R1 — Introduce an asset-identity layer.** Add an optional `asset_id` (and free-form metadata) to `ScenarioConfig`/`ScenarioResult` via new optional fields with defaults. Frozen-dataclass-safe, JSON-safe, and non-breaking. This is the keystone; everything else groups on it.

**R2 — Extract a `FleetScenarioEngine` wrapper, do not modify `ScenarioEngine`.** A new class owns a collection of `(asset_id, baseline_config)` pairs and a parallel backend, and *delegates* per-asset work to the existing `ScenarioEngine.compare()`. This preserves the validated core exactly as the Step-1 → Step-2 relationship did.

**R3 — Eliminate the redundant baseline simulation.** Have `run()` pass the already-computed `baseline_result` into `_simulate_scenario` so overlay scenarios transform it instead of re-simulating, and short-circuit `NORMAL` to reuse the baseline directly. This is a localised, test-covered change with immediate fleet payoff.

**R4 — Add a `PortfolioResult` aggregator.** A new frozen value object holding fleet rollups: expected failure count, total maintenance budget, fleet availability, worst-N assets, and a ranked scenario table. Built by *composing* per-asset `ScenarioResult`s — no change to `compare()`.

**R5 — Separate aggregation from presentation.** Make ranking and rollup return structured objects; relegate `generate_report`'s string formatting to a thin view over them. A fleet dashboard or BI sink then consumes the structured form.

**R6 — Add a parallel execution backend behind an interface.** Mirror the Step-1 sampler-hook pattern: an `ExecutionBackend` strategy (`SerialBackend` default, `ProcessPoolBackend` for fleet) so the workload parallelises without the engine's logic knowing how. Determinism is preserved by per-trial seed derivation.

**R7 — Add a summarisation mode to results.** An option to retain only scalar deltas + a compact health summary (min/mean/final, failure step) instead of full arrays, switching the memory profile from O(N×M×K×samples) to O(N×M×K).

**R8 — Construct prognostics engines once.** Lift the `RULPredictor`/`FailureRiskEngine` construction out of `_prognostics` into engine state (or inject them), reused across calls.

---

## E. Week-6 Phase-3 Readiness Assessment

Phase 3 is the fleet digital twin. Readiness by required capability:

| Capability | Status | Gap to close |
|------------|--------|--------------|
| 1. Fleet-scale simulation | **Not ready** | No fleet entity, no asset collection, serial only |
| 2. Scenario batching | **Not ready** | No `ScenarioBatch`; scenarios run one at a time |
| 3. Multi-asset aggregation | **Not ready** | No asset identity, no aggregator |
| 4. Portfolio-level risk | **Partial (per-asset only)** | Per-asset `risk_delta` exists; no fleet rollup or correlation |
| 5. Portfolio-level cost | **Partial (per-asset only)** | Per-asset `cost_impact` exists; no budget rollup, no discounting |
| 6. Portfolio-level downtime | **Partial (per-asset only)** | Per-asset hours exist; no availability % or capacity constraint |
| 7. Scenario ranking | **Not ready** | No ranking engine; report is unranked text |
| 8. Enterprise decision support | **Not ready** | No fleet-wide recommended-action rollup |
| 9. Monte Carlo integration | **Seam ready, not implemented** | Step-1 sampler hooks + stateless `compare()` are in place; no ensemble runner yet |
| 10. Computational scalability | **Not ready** | Serial, re-simulating, full-array retention, no caching |

**Summary:** The single-asset *primitives* are production-grade and the *extension seams* (stateless `compare()`, immutable results, sampler hooks, graceful degradation) are deliberately in place — which is why Phase 3 is a wrapper-and-aggregate effort rather than a rewrite. But **zero fleet-level abstractions are implemented today**. Three of ten capabilities are partially served (the per-asset versions of risk/cost/downtime), one is seam-ready (Monte Carlo), and six are absent.

---

## F. Production Readiness Score

Scored against the **fleet digital twin** target (not single-asset, where it would score far higher).

| Dimension | Weight | Score (0–100) | Weighted |
|-----------|-------:|--------------:|---------:|
| Single-asset correctness & quality | 20% | 95 | 19.0 |
| Fleet-scale abstractions present | 20% | 10 | 2.0 |
| Aggregation (risk / cost / downtime) | 15% | 30 | 4.5 |
| Scenario batching & ranking | 10% | 10 | 1.0 |
| Computational scalability | 15% | 15 | 2.25 |
| Monte Carlo readiness | 10% | 55 | 5.5 |
| Decision support (fleet) | 5% | 10 | 0.5 |
| Extensibility / clean seams | 5% | 90 | 4.5 |
| **Total** | **100%** | | **≈ 39 / 100** |

### Interpretation

**39 / 100 against the fleet target.** This is not a criticism of the delivered work — as a *single-asset* component the same code would score in the low 90s, and its high marks on correctness and extensibility (19.0 and 4.5 weighted) are what make the path forward cheap. The low total reflects that the fleet target is a genuinely larger system: the single-asset engine implements roughly one of the three multiplicities (assets × scenarios × trials) a fleet twin requires.

**The score is dominated by absence, not defect.** Every low dimension is "missing component," not "broken component." There is no rework debt and no incorrect behaviour to unwind — the refactor is purely additive (R1–R8), and the validated core is reused unchanged, exactly as Step 2 reused Step 1. A focused Phase-3 effort delivering the asset-identity layer (R1), the fleet wrapper (R2), the portfolio aggregator (R4), and the parallel backend (R6) would move this from ~39 to the production range without touching the 121 passing single-asset tests.

**Recommended Phase-3 entry criteria:** land R1 + R3 first (asset identity and the redundant-simulation fix) as they are small, test-covered, and unblock everything downstream; then R2 + R4 + R6 for the fleet wrapper, aggregator, and parallelism; finish with R5 + R7 + R8 for decision-support ergonomics and memory/throughput.