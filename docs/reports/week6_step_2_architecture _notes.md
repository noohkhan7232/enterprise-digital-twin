# Week-6 Step-2 — Scenario & What-If Simulation Engine

## Architecture Notes, Design Decisions, Risk Analysis & Week-7 Extension Points

**Component:** `src/simulation/scenario_engine.py`
**Role:** Digital-twin what-if capability layered on the Step-1 Asset State Simulator
**Status:** Implemented, validated, production-candidate — 121 tests, all passing; 48/48 completeness checks; the Step-1 simulator's 185 tests remain green (zero modification).

---

## A. Architecture Notes

### Position in the platform

The scenario engine is the layer that converts a *forward simulator* into a *digital twin*. Step 1 answers "given these conditions, how does health evolve?"; Step 2 answers the operationally decisive question, "what changes if we alter future operating conditions?" It does so by running a **baseline** simulation and a **counterfactual (scenario)** simulation, then computing a structured, explainable set of deltas — health, failure time, remaining useful life, failure risk, cost, and downtime.

Crucially, the RUL and risk deltas are not re-derived; they are computed by chaining the **same Week-5 prognostic engines** (`RULPredictor`, `FailureRiskEngine`) that score the real asset in production. A what-if answer is therefore produced by exactly the machinery that scores reality, which is what makes the comparison trustworthy rather than illustrative.

### Layered data flow

```
ScenarioEngine.run(scenario)
  ├── AssetStateSimulator(baseline_config).simulate()      → baseline SimulationResult
  ├── _simulate_scenario(scenario)                         → scenario SimulationResult
  │     ├── config-mutation path  (accelerated / load / custom)  → re-simulate
  │     └── trajectory-overlay path (maintenance / fault)        → transform health
  └── compare(baseline, scenario, scenario_cfg)
        ├── health deltas         (vectorised difference)
        ├── failure-time delta    (+ avoided / induced flags)
        ├── RUL & risk deltas      (Week-5 chain, graceful if absent)
        ├── cost & downtime impact (transparent economic model)
        └── _summarise(...)        (human-readable explainability)
                → ScenarioResult (frozen, JSON-serialisable)
```

### Two transform mechanisms

The five scenario archetypes are realised by two mechanisms, chosen so the validated Step-1 simulator is never edited:

| Mechanism | Scenarios | How |
|-----------|-----------|-----|
| **Config mutation** | accelerated degradation, load reduction, custom | Derive a new frozen `AssetSimulatorConfig` via `dataclasses.replace` and re-simulate; the degradation maths is reused verbatim |
| **Trajectory overlay** | maintenance intervention, sudden fault | Simulate the baseline, then apply a health gain (+) or loss (−) from the event time onward, re-clip to `[0, 100]`, and recompute failure bookkeeping |

The overlay mechanism is the correct choice for repair and injection because the base simulator has no native "repair" or "inject" concept — and overlaying preserves the baseline degradation *shape* between events, so the counterfactual differs from the baseline only by the intervention itself.

---

## B. Design Decisions

**1. Delta convention: scenario minus baseline.** Every delta is defined as `scenario − baseline`, so the sign is always interpretable the same way: positive `health_delta`/`rul_delta` means the scenario is better, negative `risk_delta` means the scenario is safer. This single convention removes a whole class of sign-confusion bugs from downstream consumers.

**2. Failure-time delta is `NaN` when only one trajectory fails.** A naive `scenario_failure_time − baseline_failure_time` is meaningless when one side never fails (its failure time is the `-1` sentinel). Rather than emit a misleading number, the engine sets the delta to `NaN` and exposes two explicit booleans, `failure_avoided` and `failure_induced`, which carry the actual decision-relevant information. This is the single most important correctness decision in the module.

**3. Infinite-RUL deltas collapse to `NaN` (or `0` when both infinite).** A non-degrading trajectory has infinite RUL. `finite − inf` and `inf − inf` are undefined for ranking, so `_safe_delta` returns `NaN` when exactly one side is infinite and `0.0` when both are (no difference). The `to_dict` serialiser renders every non-finite float as JSON `null`.

**4. Prognostics degrade gracefully.** The Week-5 chain is imported lazily inside a `try/except`; if it is unavailable (a minimal deployment without the predictive package), the engine still produces health, failure-time, cost, and downtime deltas, with the RUL/risk fields set to `NaN`/`0`. The what-if capability never hard-fails on a missing optional dependency.

**5. Economics mirror the Week-5 decision agent.** The default maintenance cost (5,000), failure cost (50,000), and downtime bands are the same constants the Week-5 maintenance decision agent uses, so scenario economics are consistent with production decisioning. The model is intentionally transparent — each in-service failure carries the failure cost/downtime; a maintenance intervention adds its planned cost/downtime — so a `cost_impact` figure is auditable, not a black box.

**6. The simulator is never modified.** Backward compatibility was a hard constraint. The engine consumes the Step-1 public API only; the simulator's 185 tests pass unchanged, confirming zero behavioural drift.

---

## C. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Misinterpreted failure-time delta when one side never fails | Medium | High (wrong maintenance decision) | `NaN` delta + explicit `failure_avoided`/`failure_induced` flags; tested |
| Infinite/NaN values breaking JSON export or ranking | Medium | Medium | `_safe_delta` + `_jsonsafe` render non-finite as `null`; tested |
| Overlay applied at a time off the sample grid | Low | Medium | `np.searchsorted` snaps to the first sample at/after the event; boundary tested |
| Restoration/fault pushing health out of `[0, 100]` | Medium | Low | Re-clip after every overlay; tested with extreme magnitudes |
| Week-5 package absent in a minimal deployment | Low | Medium | Lazy import + graceful degradation to health/cost deltas; tested |
| Tracker fault interrupting a scenario | Low | Medium | All tracker calls wrapped failure-safe (`# noqa: BLE001`); tested |
| Scenario config silently ignoring a typo'd field | Medium | Medium | `CUSTOM` overrides validated against the real config field set; unknown fields rejected |
| Baseline/scenario grid mismatch in `compare()` | Low | High | Length check raises `ValueError`; tested |
| Non-determinism under noise | Low | High | Seeds flow through unchanged; determinism tested on baseline, scenario, and deltas |

**Residual risk.** The economic model is deliberately simple (point costs per failure/intervention, not a discounted multi-event stream). This is appropriate for single-horizon what-if comparison but should be replaced by the Week-5 agent's full cost-benefit engine when scenarios are chained into multi-event maintenance schedules. This is a known, documented boundary, not a defect.

---

## D. Extension Points for Week-7 Monte Carlo

The scenario engine is architected to be the deterministic inner loop of the Week-7 Monte Carlo engine. Three extension points are already in place:

1. **Sampler hooks on the simulator (Step-1).** The Monte Carlo engine supplies stochastic `DegradationRateSampler`, `FaultSampler`, and `NoiseSampler` instances to the `AssetStateSimulator` that the scenario engine constructs. Because the scenario engine builds its simulators from a baseline config, a future overload that accepts pre-built simulators (or samplers) lets each Monte Carlo trial randomise degradation, faults, and noise without touching scenario logic.

2. **`ScenarioResult` is a pure value object.** Every field is a scalar or array with no hidden state, so an ensemble of `ScenarioResult`s can be aggregated directly — percentiles of `rul_delta`, the probability of `failure_avoided` across trials, the distribution of `cost_impact` — to produce a *probabilistic* what-if answer ("the de-rating avoids failure in 86% of sampled futures, with a P50 cost saving of 41,000").

3. **`compare()` is stateless and reusable.** It takes two `SimulationResult`s and returns a `ScenarioResult` with no engine-internal mutation, so the Monte Carlo engine can call it across thousands of sampled baseline/scenario pairs in a tight loop and stream the results into an aggregator.

**Proposed Week-7 surface (non-binding sketch):**

```
class MonteCarloScenarioEngine:
    def run_ensemble(self, scenario: ScenarioConfig, n_trials: int,
                     rate_sampler, fault_sampler, noise_sampler
                     ) -> EnsembleScenarioResult: ...
```

`EnsembleScenarioResult` would hold the per-trial `ScenarioResult` list plus pre-computed distributional summaries — built entirely by *composing* the Step-1 samplers and the Step-2 `compare()`, with no modification to either.

---

## E. Deliverable summary

| Item | Value |
|------|-------|
| Implementation | `scenario_engine.py`, 1,020 lines, pure NumPy |
| Tests | `test_scenario_engine.py`, 121 tests (target 120+), 100% passing, 0 skipped |
| Completeness checks | 48 / 48 |
| Scenario archetypes | 6 (normal, accelerated, maintenance, sudden fault, load reduction, custom) |
| Result fields | baseline/scenario health, health delta, failure-time delta, RUL delta, risk delta, cost impact, downtime impact, summary, plus avoided/induced flags |
| Engine methods | `run()`, `compare()`, `generate_report()` |
| Architectural standards | frozen validated configs, registry, failure-safe tracker, immutable JSON-serialisable results, full type hints, graceful degradation |
| Backward compatibility | Step-1 simulator unmodified; its 185 tests still pass |
| Stack coexistence | 7 / 7 modules (Week-5 ×5 + simulator + scenario engine) import cleanly |