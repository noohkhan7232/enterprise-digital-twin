#!/usr/bin/env python3
"""Comprehensive test suite for ``src/simulation/fleet_digital_twin.py``.

Pure NumPy throughout (the engine composes the pure-NumPy scenario engine and
simulator), so the **entire** suite runs without PyTorch or SciPy.  Coverage
(150+ tests):

- AssetRecord identity + validation
- FleetRegistry immutability (register / remove / get / list)
- RankingObjective enum
- Engine registry (register / build / list)
- run_scenario / run_scenario_batch / run_fleet_scenario / run_all_assets
- PortfolioResult aggregation correctness
- Scenario & asset ranking under every objective
- DecisionIntelligence executive summary
- Reporting
- Serialization (JSON, non-finite handling)
- Determinism
- Tracker integration (failure-safe)
- Edge cases & error handling

Run::

    pytest tests/test_fleet_digital_twin.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.asset_state_simulator import AssetSimulatorConfig
from src.simulation.fleet_digital_twin import (
    ENGINE_NAME,
    FLEET_ENGINE_REGISTRY,
    AssetRecord,
    AssetScenarioOutcome,
    DecisionIntelligence,
    FleetDigitalTwinEngine,
    FleetRegistry,
    PortfolioResult,
    RankedItem,
    RankingObjective,
    ScenarioRanker,
    build_fleet_engine,
    list_fleet_engines,
    register_fleet_engine,
)
from src.simulation.scenario_engine import ScenarioConfig, ScenarioType


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _cfg(rate: float = 1.0, **kw) -> AssetSimulatorConfig:
    d = dict(horizon=80, timestep=1.0, model="linear", degradation_rate=rate,
             initial_health=95.0, failure_threshold=30.0, noise_std=0.0)
    d.update(kw)
    return AssetSimulatorConfig(**d)


def _rec(aid: str, rate: float = 1.0, bu: str = "North") -> AssetRecord:
    return AssetRecord(asset_id=aid, asset_name=f"Asset {aid}",
                       asset_type="wind_turbine", business_unit=bu,
                       baseline_config=_cfg(rate))


def _fleet(*specs) -> FleetRegistry:
    """Build a registry from (asset_id, rate, business_unit) specs."""
    reg = FleetRegistry()
    for spec in specs:
        if len(spec) == 3:
            aid, rate, bu = spec
        else:
            aid, rate = spec
            bu = "North"
        reg = reg.register_asset(_rec(aid, rate, bu))
    return reg


def _default_fleet() -> FleetRegistry:
    return _fleet(("A", 1.0, "North"), ("B", 0.6, "North"), ("C", 1.3, "South"))


def _engine() -> FleetDigitalTwinEngine:
    return FleetDigitalTwinEngine(_default_fleet())


def _sc(stype, **kw) -> ScenarioConfig:
    return ScenarioConfig(scenario_type=stype, **kw)


def _derate(name="derate") -> ScenarioConfig:
    return _sc("load_reduction", degradation_factor=0.5, name=name)


# ---------------------------------------------------------------------------
# AssetRecord
# ---------------------------------------------------------------------------


class TestAssetRecord:
    """Tests for the asset-identity record."""

    def test_fields(self) -> None:
        r = _rec("A", 1.0, "North")
        assert r.asset_id == "A"
        assert r.asset_name == "Asset A"
        assert r.asset_type == "wind_turbine"
        assert r.business_unit == "North"

    def test_baseline_config_attached(self) -> None:
        r = _rec("A")
        assert isinstance(r.baseline_config, AssetSimulatorConfig)

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetRecord(asset_id="", asset_name="x", asset_type="t",
                        business_unit="b", baseline_config=_cfg())

    def test_whitespace_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetRecord(asset_id="   ", asset_name="x", asset_type="t",
                        business_unit="b", baseline_config=_cfg())

    def test_bad_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="baseline_config"):
            AssetRecord(asset_id="A", asset_name="x", asset_type="t",
                        business_unit="b", baseline_config="not a config")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        r = _rec("A")
        with pytest.raises((AttributeError, TypeError)):
            r.asset_id = "B"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        d = _rec("A").to_dict()
        assert d["asset_id"] == "A"
        assert "baseline_model" in d
        assert "metadata" in d

    def test_metadata_default(self) -> None:
        assert _rec("A").metadata == {}

    def test_metadata_custom(self) -> None:
        r = AssetRecord(asset_id="A", asset_name="x", asset_type="t",
                        business_unit="b", baseline_config=_cfg(),
                        metadata={"site": "NS1"})
        assert r.metadata["site"] == "NS1"


# ---------------------------------------------------------------------------
# FleetRegistry
# ---------------------------------------------------------------------------


class TestFleetRegistry:
    """Tests for the immutable fleet registry."""

    def test_empty_registry(self) -> None:
        assert len(FleetRegistry()) == 0

    def test_register_adds(self) -> None:
        reg = FleetRegistry().register_asset(_rec("A"))
        assert len(reg) == 1
        assert "A" in reg

    def test_register_returns_new(self) -> None:
        reg0 = FleetRegistry()
        reg1 = reg0.register_asset(_rec("A"))
        assert len(reg0) == 0 and len(reg1) == 1

    def test_register_duplicate_raises(self) -> None:
        reg = FleetRegistry().register_asset(_rec("A"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register_asset(_rec("A"))

    def test_remove_returns_new(self) -> None:
        reg = _fleet(("A", 1.0), ("B", 1.0))
        reg2 = reg.remove_asset("A")
        assert "A" in reg and "A" not in reg2

    def test_remove_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            FleetRegistry().remove_asset("Z")

    def test_get_asset(self) -> None:
        reg = _fleet(("A", 1.0))
        assert reg.get_asset("A").asset_id == "A"

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            FleetRegistry().get_asset("Z")

    def test_list_assets_sorted(self) -> None:
        reg = _fleet(("C", 1.0), ("A", 1.0), ("B", 1.0))
        assert reg.list_assets() == ["A", "B", "C"]

    def test_list_by_business_unit(self) -> None:
        reg = _default_fleet()
        assert reg.list_by_business_unit("North") == ["A", "B"]
        assert reg.list_by_business_unit("South") == ["C"]

    def test_list_by_unknown_bu_empty(self) -> None:
        assert _default_fleet().list_by_business_unit("Mars") == []

    def test_contains(self) -> None:
        reg = _fleet(("A", 1.0))
        assert "A" in reg and "Z" not in reg

    def test_len(self) -> None:
        assert len(_default_fleet()) == 3

    def test_assets_mapping_readonly(self) -> None:
        reg = _fleet(("A", 1.0))
        with pytest.raises((TypeError, AttributeError)):
            reg.assets["B"] = _rec("B")  # type: ignore[index]

    def test_frozen(self) -> None:
        reg = FleetRegistry()
        with pytest.raises((AttributeError, TypeError)):
            reg.assets = {}  # type: ignore[misc]

    def test_register_does_not_mutate_source_mapping(self) -> None:
        reg0 = FleetRegistry()
        reg0.register_asset(_rec("A"))
        # Original remains empty (immutability)
        assert len(reg0) == 0

    def test_chained_registration(self) -> None:
        reg = (FleetRegistry()
               .register_asset(_rec("A"))
               .register_asset(_rec("B"))
               .register_asset(_rec("C")))
        assert reg.list_assets() == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# RankingObjective enum
# ---------------------------------------------------------------------------


class TestRankingObjective:
    """Tests for the ranking-objective enum."""

    def test_values(self) -> None:
        assert RankingObjective.RISK_MINIMIZATION.value == "risk_minimization"
        assert RankingObjective.COST_MINIMIZATION.value == "cost_minimization"
        assert RankingObjective.DOWNTIME_MINIMIZATION.value == "downtime_minimization"
        assert RankingObjective.HEALTH_MAXIMIZATION.value == "health_maximization"

    def test_is_str(self) -> None:
        assert RankingObjective.RISK_MINIMIZATION == "risk_minimization"

    def test_four_objectives(self) -> None:
        assert len(list(RankingObjective)) == 4


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    """Tests for the fleet-engine registry."""

    def test_registered(self) -> None:
        assert ENGINE_NAME in FLEET_ENGINE_REGISTRY
        assert ENGINE_NAME in list_fleet_engines()

    def test_build_by_name(self) -> None:
        assert isinstance(build_fleet_engine(ENGINE_NAME), FleetDigitalTwinEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown fleet engine"):
            build_fleet_engine("nope")

    def test_registry_name_stamped(self) -> None:
        assert FleetDigitalTwinEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_fleet_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_registry(self) -> None:
        eng = build_fleet_engine(ENGINE_NAME, registry=_default_fleet())
        assert len(eng.registry) == 3


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


class TestRunScenario:
    """Tests for single-asset scenario execution."""

    def test_returns_outcome(self) -> None:
        o = _engine().run_scenario("A", _derate())
        assert isinstance(o, AssetScenarioOutcome)

    def test_outcome_tagged(self) -> None:
        o = _engine().run_scenario("A", _derate())
        assert o.asset_id == "A" and o.business_unit == "North"

    def test_unknown_asset_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            _engine().run_scenario("Z", _derate())

    def test_result_attached(self) -> None:
        o = _engine().run_scenario("A", _derate())
        assert o.result.scenario_type == "load_reduction"

    def test_load_reduction_improves(self) -> None:
        o = _engine().run_scenario("A", _derate())
        assert o.result.mean_health_delta > 0

    def test_normal_zero_delta(self) -> None:
        o = _engine().run_scenario("A", _sc("normal"))
        assert o.result.mean_health_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_scenario_batch
# ---------------------------------------------------------------------------


class TestRunScenarioBatch:
    """Tests for batch scenario execution."""

    def test_batch_count(self) -> None:
        batch = _engine().run_scenario_batch(["A", "B"], _derate())
        assert len(batch) == 2

    def test_batch_order(self) -> None:
        batch = _engine().run_scenario_batch(["B", "A"], _derate())
        assert [o.asset_id for o in batch] == ["B", "A"]

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _engine().run_scenario_batch([], _derate())

    def test_batch_unknown_asset_raises(self) -> None:
        with pytest.raises(KeyError):
            _engine().run_scenario_batch(["A", "Z"], _derate())

    def test_batch_all_assets(self) -> None:
        batch = _engine().run_scenario_batch(["A", "B", "C"], _derate())
        assert {o.asset_id for o in batch} == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# run_fleet_scenario
# ---------------------------------------------------------------------------


class TestRunFleetScenario:
    """Tests for fleet-wide scenario execution."""

    def test_returns_portfolio(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert isinstance(p, PortfolioResult)

    def test_total_assets(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert p.total_assets == 3

    def test_empty_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty fleet"):
            FleetDigitalTwinEngine(FleetRegistry()).run_fleet_scenario(_derate())

    def test_outcomes_count(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert len(p.outcomes) == 3

    def test_label_from_scenario(self) -> None:
        p = _engine().run_fleet_scenario(_derate("my label"))
        assert p.scenario_label == "my label"

    def test_explicit_label(self) -> None:
        p = _engine().run_fleet_scenario(_derate(), scenario_label="override")
        assert p.scenario_label == "override"


# ---------------------------------------------------------------------------
# run_all_assets
# ---------------------------------------------------------------------------


class TestRunAllAssets:
    """Tests for multi-scenario fleet execution."""

    def test_returns_mapping(self) -> None:
        ports = _engine().run_all_assets([_sc("normal", name="n"), _derate()])
        assert isinstance(ports, dict) and len(ports) == 2

    def test_keys_are_labels(self) -> None:
        ports = _engine().run_all_assets([_derate("derate")])
        assert "derate" in ports

    def test_empty_scenarios_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one scenario"):
            _engine().run_all_assets([])

    def test_empty_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty fleet"):
            FleetDigitalTwinEngine(FleetRegistry()).run_all_assets([_derate()])

    def test_each_is_portfolio(self) -> None:
        ports = _engine().run_all_assets([_derate()])
        assert all(isinstance(p, PortfolioResult) for p in ports.values())

    def test_auto_label_when_unnamed(self) -> None:
        ports = _engine().run_all_assets([_sc("normal")])
        assert any("normal" in k for k in ports)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for portfolio aggregation correctness."""

    def test_total_assets(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert p.total_assets == 3

    def test_cost_is_sum(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        manual = sum(o.result.cost_impact for o in p.outcomes)
        assert p.portfolio_cost_impact == pytest.approx(manual)

    def test_downtime_is_sum(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        manual = sum(o.result.downtime_impact for o in p.outcomes)
        assert p.portfolio_downtime_impact == pytest.approx(manual)

    def test_risk_is_mean(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        manual = np.mean([o.result.scenario_risk for o in p.outcomes])
        assert p.portfolio_risk == pytest.approx(manual)

    def test_mean_health_delta_is_mean(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        manual = np.mean([o.result.mean_health_delta for o in p.outcomes])
        assert p.mean_health_delta == pytest.approx(manual)

    def test_best_worst_distinct(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert p.best_asset != p.worst_asset

    def test_best_has_max_delta(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        deltas = {o.asset_id: o.result.mean_health_delta for o in p.outcomes}
        assert deltas[p.best_asset] == max(deltas.values())

    def test_worst_has_min_delta(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        deltas = {o.asset_id: o.result.mean_health_delta for o in p.outcomes}
        assert deltas[p.worst_asset] == min(deltas.values())

    def test_failure_avoided_counted(self) -> None:
        # Aggressive de-rate on a failing fleet avoids failures
        reg = _fleet(("A", 1.0), ("B", 1.0))
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(
            _sc("load_reduction", degradation_factor=0.2))
        assert p.assets_failure_avoided >= 1

    def test_failure_induced_counted(self) -> None:
        reg = _fleet(("A", 1.0))
        eng = FleetDigitalTwinEngine(FleetRegistry().register_asset(
            AssetRecord(asset_id="A", asset_name="A", asset_type="t",
                        business_unit="b",
                        baseline_config=_cfg(1.0, horizon=50))))
        p = eng.run_fleet_scenario(_sc("accelerated_degradation",
                                       degradation_factor=2.0))
        assert p.assets_failure_induced >= 1

    def test_aggregate_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one outcome"):
            _engine().aggregate([], "label")

    def test_assets_failed_count(self) -> None:
        reg = _fleet(("A", 2.0), ("B", 0.1))  # A fails, B doesn't
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_sc("normal"))
        assert p.assets_failed == 1


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    """Tests for scenario and asset ranking."""

    def _ports(self):
        eng = _engine()
        return eng, eng.run_all_assets([
            _sc("normal", name="nominal"),
            _derate("derate"),
            _sc("accelerated_degradation", degradation_factor=1.5, name="harsh"),
        ])

    def test_rank_scenarios_cost(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        assert ranked[0].rank == 1 and len(ranked) == 3

    def test_rank_ascending_score(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        scores = [r.score for r in ranked]
        assert scores == sorted(scores)

    def test_rank_risk(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "risk_minimization")
        assert ranked[0].rank == 1

    def test_rank_downtime(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "downtime_minimization")
        assert ranked[0].rank == 1

    def test_rank_health(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "health_maximization")
        # health max: best is the one with the highest mean_health_delta
        assert ranked[0].detail["mean_health_delta"] >= ranked[-1].detail["mean_health_delta"]

    def test_rank_unknown_objective_raises(self) -> None:
        eng, ports = self._ports()
        with pytest.raises(ValueError, match="Unknown ranking objective"):
            eng.rank_scenarios(ports, "maximize_chaos")

    def test_rank_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _engine().rank_scenarios({}, "cost_minimization")

    def test_rank_assets(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        ranked = _engine().rank_assets(p, "health_maximization")
        assert len(ranked) == 3 and ranked[0].rank == 1

    def test_rank_assets_order(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        ranked = eng.rank_assets(p, "health_maximization")
        # best asset (rank 1) should match the portfolio's best_asset
        assert ranked[0].label == p.best_asset

    def test_ranked_item_to_dict(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        d = ranked[0].to_dict()
        assert "rank" in d and "score" in d and "detail" in d

    def test_all_objectives_rank(self) -> None:
        eng, ports = self._ports()
        for obj in RankingObjective:
            ranked = eng.rank_scenarios(ports, obj.value)
            assert ranked[0].rank == 1
            assert [r.rank for r in ranked] == list(range(1, len(ranked) + 1))


# ---------------------------------------------------------------------------
# ScenarioRanker (unit)
# ---------------------------------------------------------------------------


class TestScenarioRanker:
    """Direct unit tests for the ranker."""

    def test_score_risk(self) -> None:
        o = _engine().run_scenario("A", _derate())
        s = ScenarioRanker._score(o.result, "risk_minimization")
        assert s == pytest.approx(o.result.risk_delta)

    def test_score_health_negated(self) -> None:
        o = _engine().run_scenario("A", _derate())
        s = ScenarioRanker._score(o.result, "health_maximization")
        assert s == pytest.approx(-o.result.mean_health_delta)

    def test_score_unknown_raises(self) -> None:
        o = _engine().run_scenario("A", _derate())
        with pytest.raises(ValueError, match="Unknown ranking objective"):
            ScenarioRanker._score(o.result, "bogus")

    def test_rank_scenarios_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ScenarioRanker().rank_scenarios([], "cost_minimization")

    def test_rank_assigns_sequential_ranks(self) -> None:
        eng = _engine()
        labelled = [(o.asset_id, o.result)
                    for o in eng.run_fleet_scenario(_derate()).outcomes]
        ranked = ScenarioRanker().rank_scenarios(labelled, "health_maximization")
        assert [r.rank for r in ranked] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Decision intelligence
# ---------------------------------------------------------------------------


class TestDecisionIntelligence:
    """Tests for the executive decision-intelligence layer."""

    def test_returns_intelligence(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p)
        assert isinstance(di, DecisionIntelligence)

    def test_top_risks_present(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        assert len(di.top_risks) >= 1

    def test_top_opportunities_present(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        assert len(di.top_opportunities) >= 1

    def test_recommended_actions_per_asset(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p)
        assert len(di.recommended_actions) == p.total_assets

    def test_top_n_limit(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p, top_n=2)
        assert len(di.top_risks) <= 2

    def test_top_n_invalid_raises(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        with pytest.raises(ValueError, match="top_n"):
            eng.decision_intelligence(p, top_n=0)

    def test_savings_non_negative(self) -> None:
        eng = _engine()
        # de-rate on a failing fleet -> savings
        reg = _fleet(("A", 1.0), ("B", 1.0))
        e2 = FleetDigitalTwinEngine(reg)
        p = e2.run_fleet_scenario(_sc("load_reduction", degradation_factor=0.2))
        di = e2.decision_intelligence(p)
        assert di.expected_savings >= 0

    def test_downtime_reduction_non_negative(self) -> None:
        reg = _fleet(("A", 1.0), ("B", 1.0))
        e2 = FleetDigitalTwinEngine(reg)
        p = e2.run_fleet_scenario(_sc("load_reduction", degradation_factor=0.2))
        di = e2.decision_intelligence(p)
        assert di.expected_downtime_reduction >= 0

    def test_narrative_non_empty(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        assert len(di.narrative) > 20

    def test_actions_reference_assets(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p)
        joined = " ".join(di.recommended_actions)
        assert "A" in joined and "B" in joined and "C" in joined

    def test_to_dict(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        d = di.to_dict()
        assert "top_risks" in d and "expected_savings" in d and "narrative" in d


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Tests for the fleet report."""

    def _ports(self):
        eng = _engine()
        return eng, eng.run_all_assets([
            _sc("normal", name="nominal"), _derate("derate")])

    def test_report_header(self) -> None:
        eng, ports = self._ports()
        rep = eng.generate_report(ports, "cost_minimization")
        assert "FLEET DIGITAL TWIN" in rep

    def test_report_lists_scenarios(self) -> None:
        eng, ports = self._ports()
        rep = eng.generate_report(ports, "cost_minimization")
        assert "derate" in rep and "nominal" in rep

    def test_report_recommends(self) -> None:
        eng, ports = self._ports()
        rep = eng.generate_report(ports, "cost_minimization")
        assert "Recommended scenario" in rep

    def test_report_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _engine().generate_report({}, "cost_minimization")

    def test_report_shows_objective(self) -> None:
        eng, ports = self._ports()
        rep = eng.generate_report(ports, "risk_minimization")
        assert "risk_minimization" in rep


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for JSON-serialisable results."""

    def test_portfolio_to_dict(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        d = p.to_dict()
        for k in ("total_assets", "portfolio_cost_impact", "best_asset",
                  "worst_asset", "outcomes", "summary"):
            assert k in d

    def test_portfolio_json(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert isinstance(json.dumps(p.to_dict()), str)

    def test_outcome_to_dict(self) -> None:
        o = _engine().run_scenario("A", _derate())
        d = o.to_dict()
        assert d["asset_id"] == "A" and "result" in d

    def test_outcome_json(self) -> None:
        o = _engine().run_scenario("A", _derate())
        assert isinstance(json.dumps(o.to_dict()), str)

    def test_decision_to_dict_json(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        assert isinstance(json.dumps(di.to_dict()), str)

    def test_asset_record_json(self) -> None:
        assert isinstance(json.dumps(_rec("A").to_dict()), str)

    def test_ranked_item_json(self) -> None:
        eng = _engine()
        ports = eng.run_all_assets([_derate(), _sc("normal", name="n")])
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        assert isinstance(json.dumps([r.to_dict() for r in ranked]), str)

    def test_outcomes_nested_in_portfolio_dict(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        d = p.to_dict()
        assert isinstance(d["outcomes"], list) and len(d["outcomes"]) == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for deterministic behaviour."""

    def test_portfolio_deterministic(self) -> None:
        e1, e2 = _engine(), _engine()
        p1 = e1.run_fleet_scenario(_derate())
        p2 = e2.run_fleet_scenario(_derate())
        assert p1.portfolio_cost_impact == p2.portfolio_cost_impact
        assert p1.mean_health_delta == p2.mean_health_delta

    def test_outcomes_deterministic(self) -> None:
        e1, e2 = _engine(), _engine()
        p1 = e1.run_fleet_scenario(_derate())
        p2 = e2.run_fleet_scenario(_derate())
        for o1, o2 in zip(p1.outcomes, p2.outcomes):
            assert np.array_equal(o1.result.scenario_health,
                                  o2.result.scenario_health)

    def test_deterministic_with_noise(self) -> None:
        reg = FleetRegistry().register_asset(
            AssetRecord(asset_id="A", asset_name="A", asset_type="t",
                        business_unit="b",
                        baseline_config=_cfg(1.0, noise_std=2.0, random_seed=7)))
        p1 = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        p2 = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p1.portfolio_cost_impact == p2.portfolio_cost_impact

    def test_ranking_deterministic(self) -> None:
        e1, e2 = _engine(), _engine()
        ports1 = e1.run_all_assets([_derate(), _sc("normal", name="n")])
        ports2 = e2.run_all_assets([_derate(), _sc("normal", name="n")])
        r1 = e1.rank_scenarios(ports1, "cost_minimization")
        r2 = e2.rank_scenarios(ports2, "cost_minimization")
        assert [x.label for x in r1] == [x.label for x in r2]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_portfolio(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        FleetDigitalTwinEngine(_default_fleet(),
                               experiment_tracker=FakeTracker()).run_fleet_scenario(
            _derate())
        assert len(logged) == 1
        assert "fleet_total_assets" in logged[0]

    def test_logs_scenario_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        FleetDigitalTwinEngine(_default_fleet(),
                               experiment_tracker=FakeTracker()).run_fleet_scenario(
            _derate("xyz"))
        assert params and params[0]["fleet_scenario"] == "xyz"

    def test_step_increments(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                steps.append(step)

        eng = FleetDigitalTwinEngine(_default_fleet(),
                                     experiment_tracker=FakeTracker())
        eng.run_fleet_scenario(_derate())
        eng.run_fleet_scenario(_derate())
        assert steps == [0, 1]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        p = FleetDigitalTwinEngine(_default_fleet(),
                                   experiment_tracker=BrokenTracker()).run_fleet_scenario(
            _derate())
        assert p.total_assets == 3

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, metrics, step=None):
                pass

        p = FleetDigitalTwinEngine(_default_fleet(),
                                   experiment_tracker=MetricsOnly()).run_fleet_scenario(
            _derate())
        assert p is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and miscellaneous behaviour."""

    def test_single_asset_fleet(self) -> None:
        reg = _fleet(("solo", 1.0))
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p.total_assets == 1
        assert p.best_asset == p.worst_asset == "solo"

    def test_default_engine_empty(self) -> None:
        assert len(FleetDigitalTwinEngine().registry) == 0

    def test_n_runs_increments(self) -> None:
        eng = _engine()
        eng.run_fleet_scenario(_derate())
        eng.run_fleet_scenario(_derate())
        assert eng._n_runs == 2

    def test_large_fleet(self) -> None:
        reg = FleetRegistry()
        for i in range(25):
            reg = reg.register_asset(_rec(f"A{i:02d}", 0.5 + (i % 5) * 0.2))
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p.total_assets == 25

    def test_mixed_models_fleet(self) -> None:
        reg = (FleetRegistry()
               .register_asset(AssetRecord(
                   asset_id="lin", asset_name="lin", asset_type="t",
                   business_unit="b", baseline_config=_cfg(1.0)))
               .register_asset(AssetRecord(
                   asset_id="exp", asset_name="exp", asset_type="t",
                   business_unit="b",
                   baseline_config=_cfg(0.04, model="exponential"))))
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p.total_assets == 2

    def test_portfolio_frozen(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        with pytest.raises((AttributeError, TypeError)):
            p.total_assets = 99  # type: ignore[misc]

    def test_outcome_frozen(self) -> None:
        o = _engine().run_scenario("A", _derate())
        with pytest.raises((AttributeError, TypeError)):
            o.asset_id = "Z"  # type: ignore[misc]

    def test_decision_frozen(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate()))
        with pytest.raises((AttributeError, TypeError)):
            di.expected_savings = 1.0  # type: ignore[misc]

    def test_no_shared_state_between_engines(self) -> None:
        e1 = _engine()
        e2 = _engine()
        e1.run_fleet_scenario(_derate())
        assert e2._n_runs == 0

    def test_maintenance_scenario_fleet(self) -> None:
        p = _engine().run_fleet_scenario(
            _sc("maintenance_intervention", intervention_time=40,
                restoration_amount=30))
        assert p.mean_health_delta > 0

    def test_custom_scenario_fleet(self) -> None:
        p = _engine().run_fleet_scenario(
            _sc("custom", overrides={"degradation_rate": 0.2}))
        assert p.total_assets == 3

    def test_sudden_fault_scenario_fleet(self) -> None:
        p = _engine().run_fleet_scenario(
            _sc("sudden_fault", fault_time=30, fault_magnitude=25))
        assert p.mean_health_delta < 0

    def test_business_unit_subset_run(self) -> None:
        eng = _engine()
        north = eng.registry.list_by_business_unit("North")
        batch = eng.run_scenario_batch(north, _derate())
        assert {o.asset_id for o in batch} == {"A", "B"}


# ---------------------------------------------------------------------------
# Additional coverage to reach the 150+ target
# ---------------------------------------------------------------------------


class TestAdditionalAggregation:
    """Further aggregation-correctness tests."""

    def test_risk_delta_mean(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        manual = np.mean([o.result.risk_delta for o in p.outcomes])
        assert p.portfolio_risk_delta == pytest.approx(manual)

    def test_load_reduction_reduces_portfolio_risk(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert p.portfolio_risk_delta <= 0

    def test_accelerated_increases_portfolio_risk(self) -> None:
        eng = FleetDigitalTwinEngine(_fleet(("A", 0.4), ("B", 0.5)))
        p = eng.run_fleet_scenario(_sc("accelerated_degradation",
                                       degradation_factor=2.0))
        assert p.portfolio_risk_delta >= 0

    def test_normal_zero_portfolio_cost(self) -> None:
        eng = FleetDigitalTwinEngine(_fleet(("A", 0.3), ("B", 0.3)))
        p = eng.run_fleet_scenario(_sc("normal"))
        assert p.portfolio_cost_impact == pytest.approx(0.0)

    def test_summary_non_empty(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert len(p.summary) > 20

    def test_summary_names_scenario(self) -> None:
        p = _engine().run_fleet_scenario(_derate("special"))
        assert "special" in p.summary

    def test_outcomes_preserve_order(self) -> None:
        p = _engine().run_fleet_scenario(_derate())
        assert [o.asset_id for o in p.outcomes] == ["A", "B", "C"]

    def test_aggregate_directly(self) -> None:
        eng = _engine()
        outs = eng.run_scenario_batch(["A", "B"], _derate())
        p = eng.aggregate(outs, "manual")
        assert p.total_assets == 2 and p.scenario_label == "manual"


class TestAdditionalRanking:
    """Further ranking tests."""

    def _ports(self):
        eng = _engine()
        return eng, eng.run_all_assets([
            _sc("normal", name="nominal"),
            _derate("derate"),
            _sc("maintenance_intervention", intervention_time=40,
                restoration_amount=40, name="overhaul"),
        ])

    def test_derate_or_overhaul_beats_nominal_on_health(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "health_maximization")
        # nominal (zero delta) should not be rank 1 when improvements exist
        assert ranked[0].label != "nominal"

    def test_rank_detail_keys(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        for key in ("portfolio_risk_delta", "portfolio_cost_impact",
                    "portfolio_downtime_impact", "mean_health_delta"):
            assert key in ranked[0].detail

    def test_rank_single_portfolio(self) -> None:
        eng = _engine()
        ports = eng.run_all_assets([_derate("only")])
        ranked = eng.rank_scenarios(ports, "cost_minimization")
        assert len(ranked) == 1 and ranked[0].rank == 1

    def test_rank_assets_all_objectives(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        for obj in RankingObjective:
            ranked = eng.rank_assets(p, obj.value)
            assert len(ranked) == 3

    def test_rank_objective_is_recorded(self) -> None:
        eng, ports = self._ports()
        ranked = eng.rank_scenarios(ports, "risk_minimization")
        assert all(r.objective == "risk_minimization" for r in ranked)


class TestAdditionalDecisionIntelligence:
    """Further decision-intelligence tests."""

    def test_top_risks_ordered(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_sc("normal"))
        di = eng.decision_intelligence(p)
        risks = [next(o.result.scenario_risk for o in p.outcomes
                      if o.asset_id == aid) for aid in di.top_risks]
        assert risks == sorted(risks, reverse=True)

    def test_top_opportunities_ordered(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p)
        opps = [next(o.result.mean_health_delta for o in p.outcomes
                     if o.asset_id == aid) for aid in di.top_opportunities]
        assert opps == sorted(opps, reverse=True)

    def test_actions_for_maintenance(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(
            _sc("maintenance_intervention", intervention_time=40,
                restoration_amount=40))
        di = eng.decision_intelligence(p)
        assert len(di.recommended_actions) == 3

    def test_narrative_mentions_scenario(self) -> None:
        eng = _engine()
        di = eng.decision_intelligence(eng.run_fleet_scenario(_derate("zzz")))
        assert "zzz" in di.narrative

    def test_top_n_larger_than_fleet(self) -> None:
        eng = _engine()
        p = eng.run_fleet_scenario(_derate())
        di = eng.decision_intelligence(p, top_n=10)
        assert len(di.top_risks) == 3  # capped at fleet size


class TestImmutabilityGuarantees:
    """Tests reinforcing the no-global-state / immutability guarantees."""

    def test_registry_reuse_across_engines(self) -> None:
        reg = _default_fleet()
        e1 = FleetDigitalTwinEngine(reg)
        e2 = FleetDigitalTwinEngine(reg)
        e1.run_fleet_scenario(_derate())
        # Shared registry is immutable; e2 sees the same assets, no leakage
        assert e2.registry.list_assets() == ["A", "B", "C"]

    def test_running_does_not_mutate_registry(self) -> None:
        eng = _engine()
        before = eng.registry.list_assets()
        eng.run_fleet_scenario(_derate())
        assert eng.registry.list_assets() == before

    def test_remove_then_run(self) -> None:
        reg = _default_fleet().remove_asset("C")
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p.total_assets == 2

    def test_register_then_run(self) -> None:
        reg = _default_fleet().register_asset(_rec("D", 0.8))
        p = FleetDigitalTwinEngine(reg).run_fleet_scenario(_derate())
        assert p.total_assets == 4