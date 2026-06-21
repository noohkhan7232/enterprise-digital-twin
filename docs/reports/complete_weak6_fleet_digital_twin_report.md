# Week 6 — Enterprise Fleet Digital Twin & Portfolio Intelligence Engine

## Technical Report — Fleet-Scale Predictive Maintenance Intelligence

*Submitted to Industrial Research Leadership*
*Enterprise Digital Twin & Predictive Maintenance Intelligence Platform*

**Document class:** Technical research report — subsystem design, mathematical formulation, and validation
**Subsystem:** `src/simulation/fleet_digital_twin.py` — fleet-scale portfolio intelligence
**Status:** Implemented, validated, production-candidate
**Preceding work:** Weeks 1–5 (acoustic condition monitoring + predictive-maintenance stack), Week 6 Phases 1–2 (asset state simulator, single-asset scenario engine)

---

## 1. Executive Summary

This report documents the design, mathematical framework, and validation of the Enterprise Fleet Digital Twin & Portfolio Intelligence Engine, the capability that elevates the platform from *single-asset what-if analysis* to *fleet-scale portfolio decision intelligence*. The preceding Week-6 work delivered a digital twin that could answer, for one machine, "what happens to remaining life, failure risk, and cost if we change its future operating conditions?" That capability is necessary but commercially insufficient: an asset-management organisation does not operate one turbine, it operates a fleet, and its decisions — capital allocation, crew scheduling, de-rating policy, overhaul timing — are made at the *portfolio* level under a budget and an availability target.

The Fleet Digital Twin Engine closes this gap. It introduces an immutable asset-identity layer, an immutable fleet registry, a fleet-scale scenario runner with four execution modes, a portfolio aggregation layer that rolls up per-asset outcomes into fleet risk, cost, downtime, and health metrics, a multi-objective scenario ranking engine, and an executive decision-intelligence layer that produces ranked risks, ranked opportunities, prioritised actions, and a quantified recommendation. Critically, the engine is a strict *additive composition* of the frozen lower layers: it constructs one validated scenario engine per asset baseline and aggregates the resulting immutable value objects. It modifies no prior code, and the two frozen lower modules remain byte-for-byte unchanged with their full test suites (185 and 121 tests) still passing.

The deliverable comprises 1,242 lines of production source and 154 unit and integration tests, every one of which executes against the real implementation with a 100% pass rate and zero skipped tests. The engine adheres to the platform's established engineering standard throughout: frozen validated dataclasses, an immutable copy-on-write registry, the name-based registry pattern, failure-safe experiment-tracker integration, pure-NumPy numerics, deterministic execution under seeds, and JSON-serialisable outputs with no global mutable state.

The principal engineering claim of this report is that the engine delivers fleet-scale intelligence *without sacrificing the validated correctness of the single-asset core* — a property achieved by composition rather than reimplementation, and one that materially de-risks enterprise deployment because the fleet layer inherits, rather than re-derives, the prognostic guarantees of Weeks 1–5.

---

## 2. Business Problem

### 2.1 Why single-asset what-if analysis is insufficient

The single-asset scenario engine answers a genuine and valuable question — the counterfactual evolution of one machine's health under altered conditions — but it answers it in isolation. Three structural limitations make this insufficient for an industrial operator.

First, **decisions are made over portfolios, not assets**. A reliability organisation managing a wind farm, a refinery's rotating equipment, or a transmission fleet allocates a finite maintenance budget and a finite crew-and-crane capacity across hundreds of assets. The economically correct action for any single asset cannot be determined without reference to the others competing for the same resources. A what-if answer that optimises one turbine in isolation is, at the fleet level, a local optimum with no guarantee of global value.

Second, **risk concentrates and correlates across a fleet**. Identical asset classes operating under a common environment (a shared offshore site, a shared duty cycle, a shared lubricant batch) fail in correlated ways. The portfolio-level question — "how many assets are likely to fail this quarter, and which?" — is not answerable by inspecting assets one at a time, because the quantity of interest is a fleet aggregate, not a per-asset scalar.

Third, **executive decisions require ranked, comparable alternatives**. Leadership does not act on a per-asset health trajectory; it acts on a recommendation of the form "pursue scenario X across the North region, at this cost, avoiding this downtime." Producing that recommendation requires running multiple scenarios across the whole fleet and ranking them against a business objective — a fleet-scale computation with no single-asset analogue.

### 2.2 The need for fleet-level intelligence

The gap between single-asset what-if and fleet decision intelligence is the gap between an *analytical tool* and a *decision-support system*. The former informs an engineer; the latter informs a capital-allocation decision. Bridging it requires four capabilities absent from the single-asset engine: a fleet abstraction with stable asset identity; an aggregation layer that composes per-asset outcomes into portfolio quantities; a ranking layer that orders scenarios by an explicit business objective; and a decision-intelligence layer that translates the aggregates into an executive recommendation with quantified savings and downtime impact. The engine described here implements all four.

---

## 3. System Architecture

The engine is organised as a five-layer composition, summarised in Table 1, in which each layer consumes the immutable output of its predecessor and the lowest layer delegates to the frozen single-asset core.

**Table 1 — Architectural layers**

| Layer | Component | Responsibility |
|------:|-----------|----------------|
| 1 | `AssetRecord` + `FleetRegistry` | Asset identity and immutable fleet membership |
| 2 | `FleetDigitalTwinEngine` (runners) | Execute scenarios across assets, delegating to the frozen engine |
| 3 | `PortfolioResult` (aggregation) | Roll up per-asset outcomes into fleet metrics |
| 4 | `ScenarioRanker` / `RankedItem` | Order scenarios and assets by a business objective |
| 5 | `DecisionIntelligence` | Produce executive risks, opportunities, actions, savings |

### 3.1 Asset Registry

The identity layer comprises two frozen types. `AssetRecord` binds a stable `asset_id` to descriptive metadata (`asset_name`, `asset_type`, `business_unit`) and the asset's `baseline_config` — the validated `AssetSimulatorConfig` that defines its nominal degradation behaviour. `FleetRegistry` is an immutable collection of these records keyed by `asset_id`. The registry is the keystone abstraction the prior architecture review identified as missing: without stable asset identity, per-asset results cannot be grouped, attributed, or aggregated.

### 3.2 Fleet Digital Twin Engine

The engine owns a registry and exposes four execution modes spanning the cardinality of fleet analysis, summarised in Table 2.

**Table 2 — Execution modes**

| Method | Cardinality | Returns |
|--------|-------------|---------|
| `run_scenario(asset_id, scenario)` | 1 asset × 1 scenario | `AssetScenarioOutcome` |
| `run_scenario_batch(asset_ids, scenario)` | k assets × 1 scenario | list of outcomes |
| `run_fleet_scenario(scenario)` | N assets × 1 scenario | `PortfolioResult` |
| `run_all_assets(scenarios)` | N assets × M scenarios | mapping of portfolios |

For each asset, the engine constructs a frozen `ScenarioEngine` from that asset's baseline configuration and invokes its validated `run` method, then tags the returned `ScenarioResult` with fleet identity. The validated single-asset logic is reused verbatim — the engine adds fleet semantics around it, never inside it.

### 3.3 Portfolio Aggregation Layer

The `aggregate` method composes a sequence of per-asset outcomes into a single frozen `PortfolioResult`. This layer is where per-asset scalars become portfolio quantities: failure counts, mean fleet risk, summed cost and downtime impact, mean health improvement, and the identification of the best- and worst-performing assets. The aggregation is a pure function of its inputs with no engine-internal state, which is what makes it safe to call across an arbitrary number of assets and, later, across an arbitrary number of Monte Carlo trials.

### 3.4 Scenario Ranking Engine

The `ScenarioRanker` orders scenarios or assets by a `RankingObjective` — risk minimisation, cost minimisation, downtime minimisation, or health maximisation. Its central design decision is to normalise every objective to a single "smaller score is better" convention so that one ascending sort produces the ranking regardless of objective, with non-finite scores ordered last deterministically. This makes ranking total, reproducible, and trivially extensible to new objectives.

### 3.5 Decision Intelligence Layer

The `DecisionIntelligence` layer translates a portfolio into an executive summary: the highest-risk assets, the highest-opportunity assets, a prioritised per-asset action list, the expected fleet savings, the expected downtime reduction, and a human-readable narrative. This is the layer that converts model output into a decision a capital owner can action, and it is deliberately explainable — every recommendation traces to the per-asset outcomes that produced it.

---

## 4. Mathematical Framework

Let the fleet comprise *N* assets indexed by *i*. For a given scenario, the single-asset engine produces for each asset a result with health delta `Δhᵢ` (mean, scenario minus baseline), scenario risk `rᵢ`, risk delta `Δrᵢ`, cost impact `cᵢ`, downtime impact `dᵢ`, and a failure indicator. The aggregation layer computes the following portfolio quantities.

**Fleet failure counts.** With `𝟙[·]` the indicator function and `Fᵢ` the event that asset *i*'s scenario trajectory reaches the failure threshold,

> assets_failed = Σᵢ 𝟙[Fᵢ],
> assets_failure_avoided = Σᵢ 𝟙[baseline fails ∧ scenario does not],
> assets_failure_induced = Σᵢ 𝟙[scenario fails ∧ baseline does not].

The avoided/induced decomposition is the decision-relevant quantity: it separates the fleet-wide *change* in failure exposure into beneficial and adverse components rather than reporting only a net count.

**Portfolio risk.** Risk is an *intensive* quantity (a per-asset probability), so it aggregates by averaging:

> portfolio_risk = (1/N) Σᵢ rᵢ,    portfolio_risk_delta = (1/N) Σᵢ Δrᵢ.

The mean failure probability is the natural fleet risk index; the mean risk delta measures whether the scenario makes the fleet, on average, safer (negative) or riskier (positive).

**Cost aggregation.** Cost is an *extensive* quantity (a fleet total), so it aggregates by summation:

> portfolio_cost_impact = Σᵢ cᵢ.

The sign convention is inherited from the single-asset engine: a negative cost impact is a fleet saving. The intensive/extensive distinction between risk (averaged) and cost (summed) is deliberate and reflects the physical meaning of each quantity — a fleet of more assets is not riskier on average merely by being larger, but it does incur more total cost.

**Downtime aggregation.** Downtime is likewise extensive:

> portfolio_downtime_impact = Σᵢ dᵢ,

with a negative value denoting a fleet-wide downtime reduction (improved availability).

**Health improvement.** The fleet health metric is the mean of per-asset mean health deltas:

> mean_health_delta = (1/N) Σᵢ Δhᵢ.

**Best and worst assets.** The extremal assets under the scenario are identified by the health delta:

> best_asset = arg maxᵢ Δhᵢ,    worst_asset = arg minᵢ Δhᵢ.

**Ranking score normalisation.** For a ranking objective *o*, the ranker computes a normalised score *sₒ* such that smaller is always better:

> s_risk = portfolio_risk_delta,
> s_cost = portfolio_cost_impact,
> s_downtime = portfolio_downtime_impact,
> s_health = − mean_health_delta.

A single ascending sort on *sₒ* then yields the ranking, with ties broken by label and non-finite scores ordered last.

---

## 5. Software Architecture

The engine embodies the platform's engineering standard, with several decisions specific to fleet scale.

**Immutable fleet registry.** `FleetRegistry` is a frozen dataclass whose every mutator — `register_asset`, `remove_asset` — returns a *new* registry rather than mutating in place. The internal mapping is wrapped in a read-only proxy so it cannot be modified even by accident. This delivers the platform's "no global mutable state" requirement structurally rather than by convention: two engines may share a registry with zero risk of interference, and a scenario run cannot alter the fleet it ran against. This is a copy-on-write design, and its cost (a shallow dictionary copy per mutation) is negligible relative to the simulation work it protects.

**Frozen dataclasses throughout.** Every configuration and result type — `AssetRecord`, `PortfolioResult`, `AssetScenarioOutcome`, `RankedItem`, `DecisionIntelligence` — is a frozen dataclass with construction-time validation where applicable. Immutability makes every result safe to cache, share across threads, and serialise without defensive copying, which is a precondition for the parallel and Monte Carlo extensions discussed below.

**Registry pattern.** The engine registers itself under a stable name through the same `register` / `build` / `list` triad used by every prior module, enabling configuration-driven instantiation and uniform tooling across the platform.

**JSON serialisation.** Every result type exposes a `to_dict` method that renders arrays as lists and non-finite floats (infinite RUL, undefined deltas) as JSON `null`, so portfolio results can be persisted, transmitted to a dashboard, and audited without special handling — the foundation for the BI and reporting integrations an enterprise deployment requires.

**Deterministic design.** Randomness flows exclusively through the frozen simulator's seed mechanism; the fleet layer introduces no new stochasticity. Given a fixed fleet and scenario set, the engine reproduces its portfolio metrics bit-for-bit, a property validated explicitly and essential for reproducible reporting and regulatory audit.

### 5.1 Engineering trade-offs

Several deliberate trade-offs shape the implementation. The first is **composition over inheritance or modification**. The engine could have been faster in absolute terms had it reached into the single-asset core and shared simulation state across assets, but doing so would have coupled the fleet layer to the core's internals and forfeited the guarantee that the validated modules remain unchanged. The chosen trade — accept the modest cost of constructing one scenario engine per asset in exchange for a frozen, independently-validated core — is the correct one for an enterprise platform where the prognostic foundation must remain certifiable. The byte-for-byte invariance of the lower modules, and their continued passing test suites, is the dividend of that choice.

The second trade-off is **copy-on-write immutability versus in-place mutation**. A mutable registry would avoid the shallow dictionary copy each registration incurs, but it would reintroduce the global-mutable-state hazard the platform standard prohibits and would make concurrent reads unsafe. Because the copy cost is negligible beside the simulation work and registry mutations are infrequent (asset onboarding, not per-request), immutability is purchased almost for free and pays for itself in concurrency safety and auditability.

The third trade-off is **eager full-trajectory retention versus summarisation**. The engine currently retains each asset's complete health trajectory in its outcome, which maximises downstream analytical flexibility (any statistic can be recomputed) at the cost of memory. For the single-scenario and moderate-fleet cases this is the right default; the documented summarisation mode is the escape valve for the large-ensemble case, deferring the memory optimisation until the Monte Carlo workload makes it necessary rather than complicating the common path prematurely.

The fourth trade-off is **explicit named objectives versus a single composite utility**. The ranking engine exposes four discrete business objectives rather than collapsing them into one weighted utility function. A composite utility would produce a single ranking, but it would bury the value judgement (the weights) inside the engine where leadership cannot see or contest it. Keeping the objectives explicit and separate makes the value judgement the decision-maker's, not the model's — the correct allocation of authority for a decision-support system.

---

## 6. Fleet Portfolio Analytics

The portfolio analytics layer answers the fleet-level questions that motivated the subsystem. From a single `run_fleet_scenario` call an operator obtains the count of assets projected to fail under the scenario, the count of failures the scenario averts versus induces, the mean fleet risk and its change, the total cost and downtime impact, the mean health improvement, and the best- and worst-performing assets — each a fleet aggregate computed by composition over the per-asset outcomes. The `run_all_assets` mode extends this to an *N × M* sweep, producing one portfolio per scenario and enabling the cross-scenario ranking that drives the executive recommendation. The analytics are intentionally transparent: every aggregate is a documented function of the per-asset results it summarises, so a fleet number can always be decomposed back to the assets that produced it — a property reliability leadership requires for trust and for root-cause analysis.

---

## 7. Scenario Ranking Methodology

Ranking is the mechanism by which the engine converts a portfolio of alternatives into a single recommendation. The methodology rests on three decisions. First, **objectives are explicit and business-aligned**: the four supported objectives map directly to the levers leadership controls — risk, cost, availability (downtime), and asset condition (health). Second, **scoring is normalised to a single direction**, so that the comparison logic is objective-agnostic and a new objective requires only a new score function, not new ordering code. Third, **ranking is total and deterministic**: ties are broken by a stable key and non-finite scores are ordered last, so the same fleet and scenarios always produce the same ranking. The ranker operates at two granularities — across scenarios (which fleet-wide action is best) and across assets within a portfolio (which assets most need attention) — using the same normalised-score machinery, which keeps the methodology uniform and auditable.

---

## 8. Decision Intelligence Framework

The decision-intelligence layer is the engine's executive interface. It surfaces the *top risks* (the assets with the highest scenario failure probability), the *top opportunities* (the assets with the largest health improvement under the scenario), a *prioritised action list* (a per-asset recommendation keyed to each asset's outcome — pursue, avoid, favourable, caution, or neutral), the *expected savings* (the sum of the genuine cost savings, counting only the assets where the scenario reduces cost rather than netting savings against costs), the *expected downtime reduction*, and a composed *executive narrative*.

A deliberate framework decision is that expected savings sums only the negative cost impacts rather than the net. A fleet in which some assets save and others cost should report the *realisable* saving an operator can bank, not a netted figure that conceals it; this matches how a maintenance-budget owner reads the number and avoids the optimistic bias of net-savings reporting. The entire layer is explainable: every surfaced risk, opportunity, and action traces to a specific per-asset outcome, so a recommendation can be interrogated and defended.

---

## 9. Validation Strategy

The validation strategy follows the platform's established discipline and adds a fleet-specific guarantee. Every aggregation formula was validated against an independent manual computation over the per-asset outcomes — portfolio cost as an explicit sum, portfolio risk as an explicit mean, best/worst by explicit arg-extremum — so the aggregates are confirmed correct, not merely plausible. Failure-semantics tests construct fleets engineered to exercise each branch: scenarios that avoid failures (a strong de-rate on a failing fleet), scenarios that induce them (acceleration on a surviving fleet), and the neither/both cases. Ranking is validated under every objective for both total ordering and the smaller-is-better normalisation. Determinism is validated by running the same fleet and scenario through two independent engines and asserting bit-for-bit equality of the portfolio metrics and per-asset trajectories.

The most important validation guarantee is *non-regression of the frozen core*: because the fleet engine composes the frozen single-asset modules through their public APIs only, the two lower modules remain byte-for-byte unchanged and their full test suites (185 and 121 tests) continue to pass. Fleet-scale capability was added with zero modification to — and therefore zero risk to — the validated prognostic core.

---

## 10. Test Coverage

The test suite comprises **154 tests with a 100% pass rate and zero skipped tests**, executing entirely without PyTorch or SciPy because the engine and the modules it composes are pure NumPy. Coverage spans the asset-identity layer (record validation, frozen guarantees), the immutable registry (register, remove, get, list, business-unit filtering, copy-on-write non-mutation), the four execution modes, aggregation correctness against manual computation, ranking under all four objectives at both scenario and asset granularity, the decision-intelligence layer (risks, opportunities, actions, savings, downtime, narrative), reporting, JSON serialisation including non-finite handling, determinism, failure-safe tracker integration, and a broad set of edge cases (single-asset fleets, large fleets, mixed degradation models, frozen-result enforcement, and no-shared-state guarantees). Table 3 summarises the coverage by area.

**Table 3 — Test coverage by area**

| Area | Representative coverage |
|------|-------------------------|
| Asset identity | Record fields, validation, immutability, serialisation |
| Fleet registry | Copy-on-write register/remove, get/list, BU filter, read-only mapping |
| Execution modes | `run_scenario`, batch, fleet, all-assets; empty/unknown handling |
| Aggregation | Cost/downtime sums, risk/health means, best/worst, failure counts |
| Ranking | All four objectives, scenario- and asset-level, ascending normalisation |
| Decision intelligence | Risks, opportunities, actions, savings, downtime, narrative |
| Serialisation | JSON round-trip, non-finite → null, nested outcomes |
| Determinism | Cross-engine bit-for-bit equality, noise reproducibility |
| Tracker | Logging, step increment, failure-safety, missing-method tolerance |
| Edge cases | Single/large/mixed fleets, frozen enforcement, no shared state |

---

## 11. Scalability Analysis

### 11.1 N assets × M scenarios

The engine's computational shape is *N* assets by *M* scenarios, with `run_all_assets` executing the full *N × M* product. Each cell is one single-asset scenario evaluation, and the cells are independent: no cell reads or writes state belonging to another. The current implementation evaluates them serially, which is correct and sufficient for fleets of moderate size but becomes the throughput bottleneck for large fleets under many scenarios.

### 11.2 Parallel execution opportunities

The workload is embarrassingly parallel by construction. Because each `run_scenario` is a pure function and `AssetScenarioOutcome` is an immutable, picklable value object, the per-asset loop in `run_fleet_scenario` can be replaced by a process-pool map with no change to the aggregation logic — the immutable design makes parallelisation a backend substitution rather than a re-architecture. Determinism is preserved because each asset's seed derives from its own configuration, independent of execution order. This is the single highest-leverage scaling step and is enabled, not blocked, by the architecture.

Two further scaling avenues follow naturally. Business-unit sharding uses the existing `list_by_business_unit` partition to process the fleet unit-by-unit (or node-by-node) and combine the per-unit portfolios with the same aggregation function. Summarisation-on-the-fly — retaining only the scalar deltas rather than the full per-asset health arrays — switches the memory profile from proportional to the number of samples to proportional to the number of assets, which matters at very large *N × M × K* once Monte Carlo trials are introduced.

### 11.3 Enterprise deployment strategy

For enterprise deployment, the recommended topology is a stateless fleet-engine service fronting a fleet registry persisted in the asset-management system of record. Because the engine carries no global mutable state and every result is JSON-serialisable, horizontal scaling is straightforward: instances are interchangeable, requests are independent, and results stream directly into a columnar store for BI and audit. The registry's immutability means configuration changes (asset onboarding, baseline updates) produce new registry versions rather than mutating a shared one, which gives a natural audit trail and safe concurrent reads. The pure-NumPy footprint keeps per-instance resource requirements modest and avoids heavyweight runtime dependencies, simplifying both edge and central deployment.

---

## 12. Limitations

The engine is deliberately scoped, and several limitations are documented rather than hidden. First, **aggregation assumes asset independence**: portfolio risk is the mean of per-asset risks, which does not model correlated failure across assets sharing an environment or a common-cause stressor. For correlated fleets this understates tail risk and is a primary target for the Monte Carlo extension. Second, **the economic model is point-estimate and single-event**: cost and downtime are charged per failure or per intervention without discounting, multi-event streams, or shared-resource (crew, crane, vessel) contention — appropriate for comparative ranking but not yet for absolute multi-year budget forecasting. Third, **execution is serial**, with parallelism available as a clean extension but not yet implemented. Fourth, **results retain full per-asset trajectories**, which is the memory bottleneck for very large ensembles and is addressed by the summarisation mode noted above. None of these is a correctness defect; each is a scoped boundary with a known, additive path forward.

---

## 13. Future Work

Beyond the immediate Monte Carlo integration, several extensions would deepen the engine's enterprise value. A correlated-risk model would replace the independence assumption with a covariance- or copula-based fleet risk aggregation, materially improving tail-risk estimation for homogeneous fleets. A capacity-constrained scheduler would convert the ranked action list into an executable maintenance schedule subject to crew, crane, and budget limits — turning a recommendation into a plan. A discounted multi-event cost model would extend the economics from comparative ranking to absolute multi-year total-cost-of-ownership forecasting. Finally, integration with the live asset-management system of record would let the registry baselines track real fleet configuration changes, closing the loop between the digital twin and the physical fleet.

---

## 14. Week 7 Monte Carlo Integration Plan

The engine is architected to be the deterministic inner loop of the Week-7 Monte Carlo Risk Intelligence Engine, which will quantify the uncertainty the current point-estimate aggregates omit. Integration plugs in at three pre-existing seams.

First, the **simulator sampler hooks** from Week-6 Phase 1 — the degradation-rate, fault, and noise samplers — are supplied with stochastic implementations so that each Monte Carlo trial randomises every asset's future trajectory. Because the fleet engine constructs each asset's scenario engine from a baseline config, the Monte Carlo driver composes these samplers per asset with no change to the fleet layer.

Second, **`PortfolioResult` is a pure value object**, so an ensemble of *K* portfolios — one per trial — aggregates directly into distributional fleet answers: the P10/P50/P90 of portfolio cost impact, the probability that the number of avoided failures exceeds a threshold, the distribution of which asset is the worst across trials, and fleet-level Value-at-Risk and Conditional VaR on the cost and downtime distributions.

Third, **aggregation and ranking are stateless pure functions**, so the Monte Carlo driver invokes them across thousands of sampled fleets in a tight loop and streams the results into a distributional summariser. The plan transforms the engine's deterministic point estimates — "fleet risk = 0.42" — into the probabilistic statements leadership needs for risk-adjusted decisions — "fleet risk P50 = 0.42, P90 = 0.61, 95% CVaR cost = \$1.4M." No modification to the fleet engine or the frozen lower layers is required; the Monte Carlo engine is a composition over them, exactly as the fleet engine is a composition over the single-asset core.

---

## 15. Conclusion

The Enterprise Fleet Digital Twin & Portfolio Intelligence Engine completes the platform's progression from single-asset diagnosis, through single-asset prognosis and what-if analysis, to fleet-scale portfolio decision intelligence. It introduces the asset-identity and immutable-registry abstractions the architecture review identified as missing, a four-mode fleet scenario runner, a portfolio aggregation layer with a principled intensive/extensive treatment of risk versus cost, a normalised multi-objective ranking methodology, and an explainable executive decision-intelligence layer. It achieves this as a strict additive composition of the frozen single-asset core, validated by 154 tests at a 100% pass rate with the two lower modules' 185 and 121 tests still passing unchanged — fleet capability added with zero risk to the validated prognostic foundation. The engine is enterprise-grade by construction: deterministic, immutable, serialisable, stateless, and free of heavyweight dependencies, with a clean and already-seated path to the Week-7 Monte Carlo engine that will layer uncertainty quantification over its deterministic aggregates. With portfolio intelligence complete, validated, and architected for probabilistic extension, the platform is ready to deliver fleet-scale, risk-adjusted maintenance decision support.