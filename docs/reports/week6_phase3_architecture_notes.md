# Week-6 Phase-3 — Fleet Digital Twin Engine

## Architecture Notes, Design Decisions, Scaling Strategy & Monte Carlo Integration Points

**Component:** `src/simulation/fleet_digital_twin.py`
**Scope:** Scale single-asset what-if (Phase 2) to enterprise fleet (N assets × M scenarios)
**Status:** Implemented, validated — 154 tests passing; 46/46 completeness checks. The two frozen lower modules are byte-for-byte unchanged and their 185 + 121 tests still pass.

This module is the direct execution of the Phase-2 architecture review. That review scored the single-asset engine ~39/100 against the fleet target and prescribed specific additive recommendations — R1 (asset-identity layer), R2 (fleet wrapper that delegates to the validated core), R4 (portfolio aggregator). This phase implements exactly those, plus ranking and decision intelligence, without modifying a single line of the frozen simulator or scenario engine.

---

## A. Architecture Notes

### Composition, not modification

The fleet engine is a strict **composition** of the frozen lower layers. It imports `AssetSimulatorConfig`, `ScenarioConfig`, and `ScenarioEngine` and uses them through their public APIs only. For each asset it constructs a `ScenarioEngine` from that asset's baseline config, calls `run()`, and tags the returned `ScenarioResult` with fleet identity. The validated single-asset logic is reused verbatim — confirmed by the frozen files remaining byte-identical and their full test suites continuing to pass.

### Layer stack

```
AssetRecord (frozen)              identity + baseline_config per asset
    ▼
FleetRegistry (immutable)         register / remove / get / list  → returns NEW registry
    ▼
FleetDigitalTwinEngine            run_scenario / _batch / _fleet / _all_assets
    │   └── per asset → ScenarioEngine(record.baseline_config).run(scenario)   [FROZEN]
    ▼
AssetScenarioOutcome (frozen)     ScenarioResult tagged with asset identity
    ▼
PortfolioResult (frozen)          fleet rollups: risk, cost, downtime, health, best/worst
    ▼
ScenarioRanker / RankedItem       rank scenarios or assets by a business objective
    ▼
DecisionIntelligence (frozen)     executive summary: risks, opportunities, actions,
                                  savings, downtime reduction, narrative
```

### The four runner methods

| Method | Shape | Returns |
|--------|-------|---------|
| `run_scenario(asset_id, scenario)` | 1 asset × 1 scenario | `AssetScenarioOutcome` |
| `run_scenario_batch(asset_ids, scenario)` | k assets × 1 scenario | `list[AssetScenarioOutcome]` |
| `run_fleet_scenario(scenario)` | N assets × 1 scenario | `PortfolioResult` |
| `run_all_assets(scenarios)` | N assets × M scenarios | `dict[label, PortfolioResult]` |

`run_all_assets` is the full N×M sweep and the natural input to `rank_scenarios`, which selects the best scenario across the fleet under a chosen objective.

---

## B. Design Decisions

**1. Immutable registry via copy-on-write.** Every `FleetRegistry` mutator (`register_asset`, `remove_asset`) returns a *new* registry; the internal mapping is wrapped in `MappingProxyType` so it cannot be mutated in place even by accident. This delivers the "no global state" requirement structurally rather than by convention — there is no in-place fleet mutation anywhere, so two engines sharing a registry can never interfere. Tested explicitly (running on one engine does not change another's view).

**2. Asset identity is a separate layer from the result.** `AssetRecord` carries identity + baseline config; `AssetScenarioOutcome` tags a `ScenarioResult` with that identity. Keeping identity out of the frozen `ScenarioResult` is what let Phase 3 add fleet semantics without touching Phase 2 — exactly the "retrofitting identity later is a breaking change" debt the review flagged, resolved by introducing it in a new wrapper type.

**3. Aggregation sign conventions are inherited, not reinvented.** Portfolio metrics preserve the Phase-2 `scenario − baseline` convention: cost and downtime are summed (negative = fleet saving / reduction), risk and health deltas are averaged (negative risk = safer, positive health = better). Cost/downtime sum because they are extensive (fleet totals); risk/health average because they are intensive (per-asset rates). This distinction is deliberate and tested.

**4. Ranking normalises to "smaller is better".** The `ScenarioRanker` maps every objective to a single ascending sort: the three minimisation objectives use the raw delta, and `HEALTH_MAXIMIZATION` negates the health delta. `NaN` scores sort last deterministically (ties broken by label), so ranking is total and reproducible. One sort, four objectives, no branching in the comparison.

**5. Decision intelligence separates savings from costs.** `expected_savings` sums only the *negative* cost impacts (the genuine savings), not the net — a fleet where some assets save and others cost should report the realisable saving, not a netted figure that hides it. Same logic for downtime reduction. This matches how an operations budget owner actually reads the number.

**6. Graceful degradation inherited for free.** Because RUL/risk come through the Phase-2 engine, the Phase-2 graceful-degradation behaviour (works without the Week-5 predictive package) is inherited automatically. The fleet layer added no new hard dependency.

---

## C. Scaling Strategy

The current implementation is **correct and serial**. The scaling path — deliberately left as clean seams, not premature optimisation — is:

1. **Embarrassingly parallel by asset.** `run_fleet_scenario` loops assets independently; each `run_scenario` is pure and shares no state. A `ProcessPoolExecutor` over `self.registry.list_assets()` parallelises the fleet with no logic change, because `AssetScenarioOutcome` is a picklable frozen value object. This is the single highest-leverage scaling step and requires only swapping the loop for a pool map.

2. **Business-unit sharding.** `list_by_business_unit` already partitions the fleet; large fleets can be processed unit-by-unit (or node-by-node) and the per-unit `PortfolioResult`s combined by a higher-level aggregator that reuses `aggregate`.

3. **Summarisation-on-the-fly.** For very large N×M×K sweeps, the per-asset full health arrays dominate memory. A future `summarise_only` mode would retain just the scalar deltas in `AssetScenarioOutcome`, switching the memory profile from O(N·M·K·samples) to O(N·M·K) — the review's R7.

4. **Result persistence.** `to_dict` already yields a flat, columnar-friendly structure; a Parquet/BI sink is a thin adapter over it, with no change to the engine.

None of these require modifying the frozen core or the fleet engine's public API — they are additive backends and modes, mirroring how this phase itself was additive over Phase 2.

---

## D. Future Monte Carlo Integration Points

Week 7's Monte Carlo engine plugs in at three already-present seams:

1. **Simulator sampler hooks (Phase 1).** Each asset's `ScenarioEngine` builds an `AssetStateSimulator`; the Monte Carlo engine supplies stochastic `DegradationRateSampler` / `FaultSampler` / `NoiseSampler` so every trial randomises that asset's future. The fleet engine's per-asset delegation means this composes per asset with no fleet-level change.

2. **`PortfolioResult` is a pure value object.** An ensemble of K portfolios (one per Monte Carlo trial) aggregates directly into distributional fleet answers: P10/P50/P90 of `portfolio_cost_impact`, the probability that `assets_failure_avoided ≥ threshold`, the distribution of `best_asset` across trials. No new engine internals — just aggregation over a list of `PortfolioResult`s.

3. **Stateless aggregation and ranking.** `aggregate`, `rank_scenarios`, and `decision_intelligence` are pure functions of their inputs with no engine-internal mutation, so a Monte Carlo driver can call them across thousands of sampled fleets in a tight loop and stream results into a distributional summariser.

**Proposed Week-7 surface (non-binding sketch):**

```
class MonteCarloFleetEngine:
    def run_ensemble(self, scenario: ScenarioConfig, n_trials: int,
                     samplers: SamplerBundle) -> EnsembleFleetResult: ...
```

`EnsembleFleetResult` would hold the per-trial `PortfolioResult` list plus pre-computed percentiles and exceedance probabilities — built entirely by *composing* the Phase-1 samplers, the Phase-2 `compare()`, and the Phase-3 `aggregate()`, with zero modification to any of the three frozen layers.

---

## E. Deliverable Summary

| Item | Value |
|------|-------|
| Implementation | `fleet_digital_twin.py`, 1,242 lines, pure NumPy |
| Tests | `test_fleet_digital_twin.py`, 154 tests (target 150+), 100% passing, 0 skipped |
| Completeness checks | 46 / 46 |
| Asset-identity layer | `AssetRecord` (id, name, type, business unit, baseline config) |
| Fleet registry | `FleetRegistry` — immutable, copy-on-write, read-only mapping |
| Runner methods | `run_scenario`, `run_scenario_batch`, `run_fleet_scenario`, `run_all_assets` |
| Portfolio aggregator | `PortfolioResult` — 10 required metrics + best/worst + summary |
| Ranking | 4 objectives (risk / cost / downtime / health), scenario- and asset-level |
| Decision intelligence | top risks, top opportunities, per-asset actions, savings, downtime reduction, narrative |
| Architectural standards | frozen dataclasses, registry, failure-safe tracker, JSON-serialisable, deterministic, no global state |
| Frozen-module guarantee | `asset_state_simulator.py` and `scenario_engine.py` byte-identical; their 185 + 121 tests still pass |
| Stack coexistence | 8 / 8 modules import cleanly (Week-5 ×5 + simulator + scenario + fleet) |