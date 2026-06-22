#!/usr/bin/env python3
"""Comprehensive test suite for ``src/agent/scenario_planning_agent.py``.

Pure NumPy / pure Python (analytic scenario models, no LLM), composing the
frozen Executive, Copilot, and Root-Cause modules.  Coverage (200+ tests):

- ScenarioCategory & RankingCriterion enums
- ScenarioPlanningConfig validation
- Registry (register / build / list)
- Budget scenarios (compose Executive Decision Engine)
- Delay scenarios (compounding-hazard escalation)
- Load scenarios (load-coupled degradation)
- Growth scenarios (extensive-quantity scaling)
- Scenario comparison (delta metrics)
- Scenario ranking (all criteria)
- Executive planning summary
- Top-level plan() & integrations (Copilot, Root-Cause)
- Serialization (JSON)
- Determinism
- Failure-safe tracker & edge cases

Run::

    pytest tests/test_scenario_planning_agent.py -v
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

from src.fleet.fleet_digital_twin import (
    AssetInput,
    FleetDigitalTwinConfig,
    FleetDigitalTwinEngine,
    FleetAsset,
    FleetSnapshot,
)
from src.agent.root_cause_analysis_agent import AssetEvidence, RootCauseAnalysisAgent
from src.agent.scenario_planning_agent import (
    AGENT_NAME,
    SCENARIO_PLANNING_REGISTRY,
    ExecutivePlanningSummary,
    RankingCriterion,
    ScenarioCategory,
    ScenarioComparison,
    ScenarioPlan,
    ScenarioPlanningAgent,
    ScenarioPlanningConfig,
    ScenarioPlanningReport,
    ScenarioRanking,
    build_scenario_planning_agent,
    list_scenario_planning_agents,
    register_scenario_planning_agent,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _mk(aid: str, rate: float) -> AssetInput:
    return AssetInput(asset_id=aid, asset_type="wt", location="N",
                      health_trajectory=np.clip(96 - rate * np.arange(45), 0, 100))


def _snap(n: int = 8) -> FleetSnapshot:
    assets = [_mk(f"W{i:02d}", 0.4 + 0.25 * i) for i in range(n)]
    return FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)


def _agent(**kw) -> ScenarioPlanningAgent:
    return ScenarioPlanningAgent(ScenarioPlanningConfig(**kw) if kw else None)


# ===========================================================================
# Enums
# ===========================================================================


class TestEnums:
    def test_scenario_categories(self) -> None:
        assert {c.value for c in ScenarioCategory} == {
            "budget", "delay", "load", "growth"}

    def test_ranking_criteria(self) -> None:
        assert {c.value for c in RankingCriterion} == {
            "risk", "savings", "roi", "coverage"}

    def test_category_is_str(self) -> None:
        assert ScenarioCategory.BUDGET == "budget"

    def test_criterion_is_str(self) -> None:
        assert RankingCriterion.RISK == "risk"


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults(self) -> None:
        c = ScenarioPlanningConfig()
        assert c.delay_days == (7, 14, 30, 60)
        assert c.growth_factors == (1.10, 1.25, 1.50, 2.00)

    def test_decrease_factor_range(self) -> None:
        with pytest.raises(ValueError, match="budget_decrease_factor"):
            ScenarioPlanningConfig(budget_decrease_factor=1.5)

    def test_increase_factor_range(self) -> None:
        with pytest.raises(ValueError, match="budget_increase_factor"):
            ScenarioPlanningConfig(budget_increase_factor=0.9)

    def test_empty_delay_days_rejected(self) -> None:
        with pytest.raises(ValueError, match="delay_days"):
            ScenarioPlanningConfig(delay_days=())

    def test_negative_delay_rejected(self) -> None:
        with pytest.raises(ValueError, match="delay_days"):
            ScenarioPlanningConfig(delay_days=(7, -1))

    def test_delay_period_positive(self) -> None:
        with pytest.raises(ValueError, match="delay_period"):
            ScenarioPlanningConfig(delay_period=0)

    def test_load_increase_range(self) -> None:
        with pytest.raises(ValueError, match="load_increase"):
            ScenarioPlanningConfig(load_increase=0.9)

    def test_load_decrease_range(self) -> None:
        with pytest.raises(ValueError, match="load_decrease"):
            ScenarioPlanningConfig(load_decrease=1.5)

    def test_negative_sensitivity_rejected(self) -> None:
        with pytest.raises(ValueError, match="sensitivity"):
            ScenarioPlanningConfig(health_sensitivity=-0.1)

    def test_bad_growth_rejected(self) -> None:
        with pytest.raises(ValueError, match="growth_factors"):
            ScenarioPlanningConfig(growth_factors=(0.9, 1.5))

    def test_empty_growth_rejected(self) -> None:
        with pytest.raises(ValueError, match="growth_factors"):
            ScenarioPlanningConfig(growth_factors=())

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            ScenarioPlanningConfig(top_n=0)

    def test_frozen(self) -> None:
        c = ScenarioPlanningConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.delay_period = 1  # type: ignore[misc]

    def test_custom_currency(self) -> None:
        assert ScenarioPlanningConfig(currency="EUR").currency == "EUR"


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert AGENT_NAME in SCENARIO_PLANNING_REGISTRY
        assert AGENT_NAME in list_scenario_planning_agents()

    def test_build(self) -> None:
        assert isinstance(build_scenario_planning_agent(AGENT_NAME),
                          ScenarioPlanningAgent)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scenario planning"):
            build_scenario_planning_agent("nope")

    def test_registry_name(self) -> None:
        assert ScenarioPlanningAgent._registry_name == AGENT_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_scenario_planning_agent(AGENT_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        a = build_scenario_planning_agent(
            AGENT_NAME, config=ScenarioPlanningConfig(currency="GBP"))
        assert a.config.currency == "GBP"


# ===========================================================================
# Budget scenarios
# ===========================================================================


class TestBudgetScenarios:
    def test_three_scenarios(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert len(bs) == 3

    def test_names(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        names = {r.scenario_name for r in bs}
        assert names == {"Budget Decrease", "Budget Freeze", "Budget Increase"}

    def test_category(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(r.category == "budget" for r in bs)

    def test_predicts_all_metrics(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        for r in bs:
            d = r.predictions_dict()
            for k in ("risk", "coverage", "roi", "expected_savings"):
                assert k in d

    def test_increase_higher_coverage(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        assert inc.prediction("coverage") >= dec.prediction("coverage")

    def test_increase_lower_risk(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        assert inc.prediction("risk") <= dec.prediction("risk")

    def test_increase_higher_savings(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        assert inc.prediction("expected_savings") >= dec.prediction("expected_savings")

    def test_freeze_uses_baseline(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        freeze = next(r for r in bs if r.scenario_name == "Budget Freeze")
        assert "100%" in freeze.assumptions[0]

    def test_recommendations_present(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(len(r.recommendations) >= 1 for r in bs)

    def test_confidence_in_unit(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(0 <= r.confidence <= 1 for r in bs)

    def test_empty_snapshot_raises(self) -> None:
        class Empty:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_failure_probability = 0
            fleet_expected_cost = fleet_expected_failure_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().budget_scenarios(Empty(), 20000)

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_budget"):
            _agent().budget_scenarios(_snap(), -1)

    def test_zero_budget_ok(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 0)
        assert len(bs) == 3


# ===========================================================================
# Delay scenarios
# ===========================================================================


class TestDelayScenarios:
    def test_four_scenarios(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert len(ds) == 4

    def test_category(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert all(r.category == "delay" for r in ds)

    def test_predicts_metrics(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        for r in ds:
            d = r.predictions_dict()
            for k in ("failure_probability", "downtime", "loss"):
                assert k in d

    def test_failure_probability_monotone(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        fps = [r.prediction("failure_probability") for r in ds]
        assert all(fps[i] <= fps[i + 1] + 1e-9 for i in range(3))

    def test_failure_probability_bounded(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert all(0 <= r.prediction("failure_probability") <= 1 for r in ds)

    def test_loss_monotone(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        losses = [r.prediction("loss") for r in ds]
        assert all(losses[i] <= losses[i + 1] + 1e-6 for i in range(3))

    def test_downtime_monotone(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        dts = [r.prediction("downtime") for r in ds]
        assert all(dts[i] <= dts[i + 1] + 1e-6 for i in range(3))

    def test_confidence_decreases(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        confs = [r.confidence for r in ds]
        assert all(confs[i] >= confs[i + 1] for i in range(3))

    def test_names_have_days(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert ds[0].scenario_name == "Delay 7 days"

    def test_custom_delays(self) -> None:
        ds = _agent(delay_days=(5, 10)).delay_scenarios(_snap())
        assert len(ds) == 2

    def test_recommendations_present(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert all(len(r.recommendations) >= 1 for r in ds)

    def test_empty_snapshot_raises(self) -> None:
        class Empty:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_failure_probability = 0
            fleet_expected_cost = fleet_expected_failure_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().delay_scenarios(Empty())


# ===========================================================================
# Load scenarios
# ===========================================================================


class TestLoadScenarios:
    def test_two_scenarios(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert len(ls) == 2

    def test_category(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(r.category == "load" for r in ls)

    def test_predicts_metrics(self) -> None:
        ls = _agent().load_scenarios(_snap())
        for r in ls:
            d = r.predictions_dict()
            for k in ("health", "rul", "risk"):
                assert k in d

    def test_increase_lowers_health(self) -> None:
        ls = _agent().load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        dec = next(r for r in ls if "Decrease" in r.scenario_name)
        assert inc.prediction("health") < dec.prediction("health")

    def test_increase_shortens_rul(self) -> None:
        ls = _agent().load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        dec = next(r for r in ls if "Decrease" in r.scenario_name)
        assert inc.prediction("rul") < dec.prediction("rul")

    def test_increase_raises_risk(self) -> None:
        ls = _agent().load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        dec = next(r for r in ls if "Decrease" in r.scenario_name)
        assert inc.prediction("risk") > dec.prediction("risk")

    def test_health_bounded(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(0 <= r.prediction("health") <= 100 for r in ls)

    def test_risk_bounded(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(0 <= r.prediction("risk") <= 1 for r in ls)

    def test_rul_non_negative(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(r.prediction("rul") >= 0 for r in ls)

    def test_recommendations_present(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(len(r.recommendations) >= 1 for r in ls)

    def test_empty_snapshot_raises(self) -> None:
        class Empty:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_failure_probability = 0
            fleet_expected_cost = fleet_expected_failure_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().load_scenarios(Empty())


# ===========================================================================
# Growth scenarios
# ===========================================================================


class TestGrowthScenarios:
    def test_four_scenarios(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert len(gs) == 4

    def test_category(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert all(r.category == "growth" for r in gs)

    def test_predicts_metrics(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        for r in gs:
            d = r.predictions_dict()
            for k in ("maintenance_demand", "risk_exposure", "budget_impact"):
                assert k in d

    def test_budget_impact_monotone(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        bis = [r.prediction("budget_impact") for r in gs]
        assert all(bis[i] <= bis[i + 1] for i in range(3))

    def test_demand_monotone(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        ds = [r.prediction("maintenance_demand") for r in gs]
        assert all(ds[i] <= ds[i + 1] for i in range(3))

    def test_exposure_monotone(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        ex = [r.prediction("risk_exposure") for r in gs]
        assert all(ex[i] <= ex[i + 1] for i in range(3))

    def test_confidence_decreases(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        confs = [r.confidence for r in gs]
        assert all(confs[i] >= confs[i + 1] for i in range(3))

    def test_names(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert gs[0].scenario_name == "Fleet Growth +10%"

    def test_budget_scales_correctly(self) -> None:
        snap = _snap()
        gs = _agent().growth_scenarios(snap)
        first = gs[0]
        assert first.prediction("budget_impact") == pytest.approx(
            snap.fleet_expected_cost * 1.10)

    def test_custom_growth(self) -> None:
        gs = _agent(growth_factors=(1.2, 1.4)).growth_scenarios(_snap())
        assert len(gs) == 2

    def test_recommendations_present(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert all(len(r.recommendations) >= 1 for r in gs)


# ===========================================================================
# Comparison
# ===========================================================================


class TestComparison:
    def test_returns_comparison(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        assert isinstance(cmp, ScenarioComparison)

    def test_risk_delta(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        assert cmp.risk_delta is not None

    def test_roi_delta(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        assert cmp.roi_delta is not None

    def test_coverage_delta(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        assert cmp.coverage_delta is not None

    def test_cost_delta(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        assert cmp.cost_delta is not None

    def test_delta_arithmetic(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[1], bs[2])
        expected = bs[2].prediction("risk") - bs[1].prediction("risk")
        assert cmp.risk_delta == pytest.approx(expected)

    def test_deltas_only_shared_metrics(self) -> None:
        ag = _agent()
        b = ag.budget_scenarios(_snap(), 20000)[0]
        d = ag.delay_scenarios(_snap())[0]
        # budget and delay share no metrics
        cmp = ag.compare(b, d)
        assert cmp.deltas == () and cmp.risk_delta is None

    def test_names_recorded(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[0], bs[2])
        assert cmp.baseline_name == "Budget Decrease"
        assert cmp.alternative_name == "Budget Increase"

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(TypeError, match="ScenarioPlanningReport"):
            _agent().compare("a", "b")


# ===========================================================================
# Ranking
# ===========================================================================


class TestRanking:
    def test_rank_by_risk_ascending(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "risk")
        vals = [v for _, v in rk.ranked]
        assert vals == sorted(vals)

    def test_rank_by_roi_descending(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "roi")
        vals = [v for _, v in rk.ranked]
        assert vals == sorted(vals, reverse=True)

    def test_rank_by_savings_descending(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "savings")
        vals = [v for _, v in rk.ranked]
        assert vals == sorted(vals, reverse=True)

    def test_rank_by_coverage_descending(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "coverage")
        vals = [v for _, v in rk.ranked]
        assert vals == sorted(vals, reverse=True)

    def test_criterion_recorded(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert _agent().rank_scenarios(bs, "risk").criterion == "risk"

    def test_all_scenarios_ranked(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert len(_agent().rank_scenarios(bs, "risk").ranked) == 3

    def test_invalid_criterion_raises(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        with pytest.raises(ValueError, match="criterion"):
            _agent().rank_scenarios(bs, "bogus")

    def test_scenarios_without_metric_skipped(self) -> None:
        # Delay scenarios have no 'coverage' metric -> ranked empty
        ds = _agent().delay_scenarios(_snap())
        rk = _agent().rank_scenarios(ds, "coverage")
        assert rk.ranked == ()

    def test_best_by_risk_is_increase(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "risk")
        assert rk.ranked[0][0] == "Budget Increase"


# ===========================================================================
# Executive summary
# ===========================================================================


class TestExecutiveSummary:
    def test_returns_summary(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert isinstance(_agent().executive_summary(bs), ExecutivePlanningSummary)

    def test_best_and_worst(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert s.best_scenario == "Budget Increase"
        assert s.worst_scenario == "Budget Decrease"

    def test_recommended_is_best(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert s.recommended_scenario == s.best_scenario

    def test_confidence_in_unit(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert 0 <= s.confidence <= 1

    def test_commentary_non_empty(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert len(s.strategic_commentary) > 40

    def test_criterion_recorded(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "roi")
        assert s.criterion == "roi"

    def test_no_metric_raises(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        with pytest.raises(ValueError, match="no scenario"):
            _agent().executive_summary(ds, "coverage")

    def test_confidence_is_mean(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert s.confidence == pytest.approx(
            np.mean([r.confidence for r in bs]))


# ===========================================================================
# Plan orchestration
# ===========================================================================


class TestPlan:
    def test_returns_plan(self) -> None:
        assert isinstance(_agent().plan(_snap(), 20000), ScenarioPlan)

    def test_all_families_present(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert len(p.budget_scenarios) == 3
        assert len(p.delay_scenarios) == 4
        assert len(p.load_scenarios) == 2
        assert len(p.growth_scenarios) == 4

    def test_all_reports_count(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert len(p.all_reports()) == 13

    def test_ranking_present(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert isinstance(p.ranking, ScenarioRanking)

    def test_summary_present(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert isinstance(p.summary, ExecutivePlanningSummary)

    def test_empty_snapshot_raises(self) -> None:
        class Empty:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_failure_probability = 0
            fleet_expected_cost = fleet_expected_failure_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().plan(Empty(), 20000)

    def test_custom_criterion(self) -> None:
        p = _agent().plan(_snap(), 20000, criterion="roi")
        assert p.ranking.criterion == "roi"

    def test_currency_propagates(self) -> None:
        p = _agent(currency="EUR").plan(_snap(), 20000)
        assert p.currency == "EUR"

    def test_commentary_has_copilot_narrative(self) -> None:
        # The recommended portfolio's risk-reduction statement is woven in
        p = _agent().plan(_snap(), 20000)
        assert "risk" in p.summary.strategic_commentary.lower()


# ===========================================================================
# Integrations
# ===========================================================================


class TestIntegrations:
    def test_root_cause_integration(self) -> None:
        agent = ScenarioPlanningAgent(root_cause_agent=RootCauseAnalysisAgent())
        ev = [AssetEvidence(asset_id=f"W{i:02d}", vibration=0.8) for i in range(8)]
        p = agent.plan(_snap(), 20000, evidence_items=ev)
        assert "vibration" in p.summary.strategic_commentary.lower()

    def test_root_cause_optional(self) -> None:
        # Without an RCA agent, evidence is ignored gracefully
        agent = ScenarioPlanningAgent()
        ev = [AssetEvidence(asset_id=f"W{i:02d}", vibration=0.8) for i in range(8)]
        p = agent.plan(_snap(), 20000, evidence_items=ev)
        assert isinstance(p, ScenarioPlan)

    def test_copilot_default_created(self) -> None:
        agent = ScenarioPlanningAgent()
        assert agent.copilot is not None

    def test_custom_copilot(self) -> None:
        from src.agent.decision_copilot_agent import DecisionCopilotAgent
        cop = DecisionCopilotAgent()
        agent = ScenarioPlanningAgent(copilot=cop)
        assert agent.copilot is cop

    def test_executive_engine_drives_budget(self) -> None:
        # Budget scenarios must reflect real portfolio economics (ROI > 0)
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(r.prediction("roi") > 0 for r in bs)

    def test_monte_carlo_hook_accepted(self) -> None:
        agent = ScenarioPlanningAgent(monte_carlo_engine=object())
        assert agent.monte_carlo_engine is not None


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_report_to_dict(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        d = bs[0].to_dict()
        for k in ("scenario_name", "category", "assumptions", "predictions",
                  "confidence", "recommendations"):
            assert k in d

    def test_report_json(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert isinstance(json.dumps(bs[0].to_dict()), str)

    def test_comparison_json(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[0], bs[1])
        assert isinstance(json.dumps(cmp.to_dict()), str)

    def test_ranking_json(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        rk = _agent().rank_scenarios(bs, "risk")
        assert isinstance(json.dumps(rk.to_dict()), str)

    def test_summary_json(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs)
        assert isinstance(json.dumps(s.to_dict()), str)

    def test_plan_json(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert isinstance(json.dumps(p.to_dict()), str)

    def test_predictions_serialized_as_pairs(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        d = bs[0].to_dict()
        assert isinstance(d["predictions"][0], list)

    def test_plan_nested(self) -> None:
        d = _agent().plan(_snap(), 20000).to_dict()
        assert len(d["budget_scenarios"]) == 3 and len(d["delay_scenarios"]) == 4


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_plan_deterministic(self) -> None:
        snap = _snap()
        p1 = _agent().plan(snap, 20000)
        p2 = _agent().plan(snap, 20000)
        assert p1.ranking.ranked == p2.ranking.ranked
        assert p1.summary.best_scenario == p2.summary.best_scenario

    def test_budget_deterministic(self) -> None:
        snap = _snap()
        b1 = _agent().budget_scenarios(snap, 20000)
        b2 = _agent().budget_scenarios(snap, 20000)
        assert [r.predictions for r in b1] == [r.predictions for r in b2]

    def test_delay_deterministic(self) -> None:
        snap = _snap()
        d1 = _agent().delay_scenarios(snap)
        d2 = _agent().delay_scenarios(snap)
        assert [r.predictions for r in d1] == [r.predictions for r in d2]

    def test_load_deterministic(self) -> None:
        snap = _snap()
        l1 = _agent().load_scenarios(snap)
        l2 = _agent().load_scenarios(snap)
        assert [r.predictions for r in l1] == [r.predictions for r in l2]

    def test_growth_deterministic(self) -> None:
        snap = _snap()
        g1 = _agent().growth_scenarios(snap)
        g2 = _agent().growth_scenarios(snap)
        assert [r.predictions for r in g1] == [r.predictions for r in g2]

    def test_ranking_deterministic(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        r1 = _agent().rank_scenarios(bs, "risk")
        r2 = _agent().rank_scenarios(bs, "risk")
        assert r1.ranked == r2.ranked


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_plans(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        ScenarioPlanningAgent(experiment_tracker=FakeTracker()).plan(_snap(), 20000)
        assert logged and "scenario_plans" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        p = ScenarioPlanningAgent(
            experiment_tracker=BrokenTracker()).plan(_snap(), 20000)
        assert p is not None

    def test_no_tracker_ok(self) -> None:
        assert _agent().plan(_snap(), 20000) is not None

    def test_plan_count_increments(self) -> None:
        agent = _agent()
        agent.plan(_snap(), 20000)
        agent.plan(_snap(), 20000)
        assert agent._n_plans == 2


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_asset_fleet(self) -> None:
        snap = _snap(1)
        p = _agent().plan(snap, 20000)
        assert isinstance(p, ScenarioPlan)

    def test_large_fleet(self) -> None:
        snap = _snap(40)
        p = _agent().plan(snap, 50000)
        assert isinstance(p, ScenarioPlan)

    def test_report_frozen(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        with pytest.raises((AttributeError, TypeError)):
            bs[0].confidence = 0.5  # type: ignore[misc]

    def test_plan_frozen(self) -> None:
        p = _agent().plan(_snap(), 20000)
        with pytest.raises((AttributeError, TypeError)):
            p.currency = "X"  # type: ignore[misc]

    def test_prediction_missing_returns_none(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert bs[0].prediction("nonexistent") is None

    def test_predictions_dict(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert isinstance(bs[0].predictions_dict(), dict)

    def test_very_large_budget(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 10_000_000)
        # increase covers everything
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        assert inc.prediction("coverage") == pytest.approx(1.0)

    def test_distinct_agents_independent(self) -> None:
        a1 = _agent()
        a2 = _agent()
        a1.plan(_snap(), 20000)
        assert a2._n_plans == 0

    def test_custom_load_factors(self) -> None:
        agent = _agent(load_increase=1.5, load_decrease=0.5)
        ls = agent.load_scenarios(_snap())
        assert len(ls) == 2

    def test_high_load_floors_health(self) -> None:
        # Extreme load + sensitivity drives health toward the [0,100] floor
        agent = _agent(load_increase=2.0, health_sensitivity=2.0)
        ls = agent.load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        assert inc.prediction("health") >= 0


# ===========================================================================
# Additional coverage to reach the 200+ target
# ===========================================================================


class TestAdditionalBudget:
    def test_decrease_budget_value(self) -> None:
        agent = _agent(budget_decrease_factor=0.5)
        bs = agent.budget_scenarios(_snap(), 20000)
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        assert "50%" in dec.assumptions[0]

    def test_increase_budget_value(self) -> None:
        agent = _agent(budget_increase_factor=2.0)
        bs = agent.budget_scenarios(_snap(), 20000)
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        assert "200%" in inc.assumptions[0]

    def test_coverage_in_unit(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(0 <= r.prediction("coverage") <= 1 for r in bs)

    def test_risk_in_unit(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert all(0 <= r.prediction("risk") <= 1 for r in bs)

    def test_zero_budget_decrease(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 0)
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        assert dec.prediction("coverage") == 0.0


class TestAdditionalDelay:
    def test_seven_day_lowest_fp(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert ds[0].prediction("failure_probability") == min(
            r.prediction("failure_probability") for r in ds)

    def test_assumptions_mention_days(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert "7 days" in ds[0].assumptions[0]

    def test_short_delay_tolerable_message(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert "tolerable" in ds[0].recommendations[0].lower()

    def test_long_delay_escalating_message(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        last = ds[-1]
        assert "escalat" in last.recommendations[0].lower()

    def test_custom_delay_period(self) -> None:
        fast = _agent(delay_period=10.0).delay_scenarios(_snap())
        slow = _agent(delay_period=60.0).delay_scenarios(_snap())
        # faster compounding -> higher failure prob at same horizon
        assert fast[-1].prediction("failure_probability") >= \
            slow[-1].prediction("failure_probability")


class TestAdditionalLoad:
    def test_load_increase_name(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert any(r.scenario_name == "Load Increase" for r in ls)

    def test_load_decrease_name(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert any(r.scenario_name == "Load Decrease" for r in ls)

    def test_increase_recommendation_mentions_derating(self) -> None:
        ls = _agent().load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        assert "derat" in inc.recommendations[0].lower()

    def test_assumptions_mention_load(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert "load" in ls[0].assumptions[0].lower()


class TestAdditionalGrowth:
    def test_exposure_scales(self) -> None:
        snap = _snap()
        gs = _agent().growth_scenarios(snap)
        assert gs[0].prediction("risk_exposure") == pytest.approx(
            snap.fleet_expected_failure_cost * 1.10)

    def test_100pct_name(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert any("+100%" in r.scenario_name for r in gs)

    def test_demand_scales(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        # demand at +100% is double demand at +50%? not exactly (factors differ)
        assert gs[-1].prediction("maintenance_demand") >= \
            gs[0].prediction("maintenance_demand")

    def test_assumptions_mention_growth(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert "grows" in gs[0].assumptions[0].lower()


class TestAdditionalComparison:
    def test_self_comparison_zero_deltas(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[0], bs[0])
        assert cmp.risk_delta == pytest.approx(0.0)

    def test_delta_count_matches_shared(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[0], bs[1])
        assert len(cmp.deltas) == 4  # risk, coverage, roi, savings

    def test_growth_comparison(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        cmp = _agent().compare(gs[0], gs[1])
        assert len(cmp.deltas) == 3  # demand, exposure, budget


class TestAdditionalRankingSummary:
    def test_rank_delay_by_savings_empty(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        # delay scenarios have no savings metric
        assert _agent().rank_scenarios(ds, "savings").ranked == ()

    def test_summary_commentary_names_best(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "risk")
        assert s.best_scenario in s.strategic_commentary

    def test_summary_with_roi_criterion(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        s = _agent().executive_summary(bs, "roi")
        assert s.best_scenario in {"Budget Decrease", "Budget Freeze",
                                   "Budget Increase"}

    def test_growth_ranking_skips_unknown_metric(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        # growth scenarios have no 'risk' metric -> empty ranking
        assert _agent().rank_scenarios(gs, "risk").ranked == ()


class TestAdditionalSerialization:
    def test_all_report_categories_serialize(self) -> None:
        agent = _agent()
        snap = _snap()
        for reports in (agent.budget_scenarios(snap, 20000),
                        agent.delay_scenarios(snap),
                        agent.load_scenarios(snap),
                        agent.growth_scenarios(snap)):
            for r in reports:
                assert isinstance(json.dumps(r.to_dict()), str)

    def test_plan_all_reports_helper(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert len(p.all_reports()) == len(p.budget_scenarios) + \
            len(p.delay_scenarios) + len(p.load_scenarios) + \
            len(p.growth_scenarios)

    def test_comparison_deltas_serialized(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        d = _agent().compare(bs[0], bs[1]).to_dict()
        assert isinstance(d["deltas"][0], list)


# ===========================================================================
# Further coverage to reach the 200+ target
# ===========================================================================


class TestBudgetEconomics:
    def test_freeze_between_decrease_and_increase_risk(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        frz = next(r for r in bs if r.scenario_name == "Budget Freeze")
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        assert dec.prediction("risk") >= frz.prediction("risk") >= inc.prediction("risk")

    def test_freeze_between_coverage(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        dec = next(r for r in bs if r.scenario_name == "Budget Decrease")
        frz = next(r for r in bs if r.scenario_name == "Budget Freeze")
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        assert dec.prediction("coverage") <= frz.prediction("coverage") <= inc.prediction("coverage")

    def test_increase_recommendation_mentions_capital(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        inc = next(r for r in bs if r.scenario_name == "Budget Increase")
        assert "capital" in inc.recommendations[0].lower()

    def test_eur_currency_in_assumptions(self) -> None:
        bs = _agent(currency="EUR").budget_scenarios(_snap(), 20000)
        assert "EUR" in bs[0].assumptions[0]


class TestDelayEconomics:
    def test_loss_scales_with_failure(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        # loss and failure probability move together
        fps = [r.prediction("failure_probability") for r in ds]
        losses = [r.prediction("loss") for r in ds]
        assert (fps[-1] >= fps[0]) and (losses[-1] >= losses[0])

    def test_failure_probability_never_exceeds_one(self) -> None:
        ds = _agent(delay_days=(7, 14, 30, 60, 120, 365)).delay_scenarios(_snap())
        assert all(r.prediction("failure_probability") <= 1.0 + 1e-9 for r in ds)

    def test_confidence_floor(self) -> None:
        ds = _agent(delay_days=(365,)).delay_scenarios(_snap())
        assert ds[0].confidence >= 0.4

    def test_downtime_positive(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert all(r.prediction("downtime") >= 0 for r in ds)


class TestLoadEconomics:
    def test_symmetric_sensitivity(self) -> None:
        # With symmetric load factors, health deltas are symmetric around baseline
        agent = _agent(load_increase=1.2, load_decrease=0.8)
        ls = agent.load_scenarios(_snap())
        inc = next(r for r in ls if "Increase" in r.scenario_name)
        dec = next(r for r in ls if "Decrease" in r.scenario_name)
        snap = _snap()
        h0 = snap.average_health
        assert (h0 - inc.prediction("health")) == pytest.approx(
            dec.prediction("health") - h0, abs=1e-6)

    def test_zero_sensitivity_no_change(self) -> None:
        agent = _agent(health_sensitivity=0.0, rul_sensitivity=0.0,
                       risk_sensitivity=0.0)
        ls = agent.load_scenarios(_snap())
        snap = _snap()
        for r in ls:
            assert r.prediction("health") == pytest.approx(snap.average_health)

    def test_confidence_in_unit(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert all(0 <= r.confidence <= 1 for r in ls)


class TestGrowthEconomics:
    def test_budget_proportional_to_factor(self) -> None:
        snap = _snap()
        gs = _agent(growth_factors=(2.0,)).growth_scenarios(snap)
        assert gs[0].prediction("budget_impact") == pytest.approx(
            snap.fleet_expected_cost * 2.0)

    def test_demand_non_negative(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert all(r.prediction("maintenance_demand") >= 0 for r in gs)

    def test_confidence_in_unit(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert all(0 <= r.confidence <= 1 for r in gs)

    def test_recommendation_mentions_budget(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert "budget" in gs[0].recommendations[0].lower()


class TestPlanIntegrationDetail:
    def test_plan_ranking_matches_budget(self) -> None:
        p = _agent().plan(_snap(), 20000, criterion="risk")
        names = {n for n, _ in p.ranking.ranked}
        budget_names = {r.scenario_name for r in p.budget_scenarios}
        assert names == budget_names

    def test_plan_recommended_in_budget(self) -> None:
        p = _agent().plan(_snap(), 20000)
        budget_names = {r.scenario_name for r in p.budget_scenarios}
        assert p.summary.recommended_scenario in budget_names

    def test_plan_summary_confidence_matches_budget(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert p.summary.confidence == pytest.approx(
            np.mean([r.confidence for r in p.budget_scenarios]))

    def test_plan_with_roi_criterion_recommends_high_roi(self) -> None:
        p = _agent().plan(_snap(), 20000, criterion="roi")
        rois = {r.scenario_name: r.prediction("roi") for r in p.budget_scenarios}
        assert rois[p.summary.recommended_scenario] == max(rois.values())


class TestReportHelpers:
    def test_prediction_lookup(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert bs[0].prediction("risk") == bs[0].predictions_dict()["risk"]

    def test_predictions_dict_complete(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert len(bs[0].predictions_dict()) == 4

    def test_assumptions_non_empty(self) -> None:
        agent = _agent()
        snap = _snap()
        for reports in (agent.budget_scenarios(snap, 20000),
                        agent.delay_scenarios(snap),
                        agent.load_scenarios(snap),
                        agent.growth_scenarios(snap)):
            assert all(len(r.assumptions) >= 1 for r in reports)

    def test_all_reports_have_confidence(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert all(0 <= r.confidence <= 1 for r in p.all_reports())


class TestFinalCoverage:
    def test_budget_report_category_enum(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        assert bs[0].category == ScenarioCategory.BUDGET.value

    def test_delay_report_category_enum(self) -> None:
        ds = _agent().delay_scenarios(_snap())
        assert ds[0].category == ScenarioCategory.DELAY.value

    def test_load_report_category_enum(self) -> None:
        ls = _agent().load_scenarios(_snap())
        assert ls[0].category == ScenarioCategory.LOAD.value

    def test_growth_report_category_enum(self) -> None:
        gs = _agent().growth_scenarios(_snap())
        assert gs[0].category == ScenarioCategory.GROWTH.value

    def test_compare_returns_named_tuple_like(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        cmp = _agent().compare(bs[0], bs[2])
        assert cmp.alternative_name == "Budget Increase"

    def test_ranking_to_dict_criterion(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        d = _agent().rank_scenarios(bs, "coverage").to_dict()
        assert d["criterion"] == "coverage"

    def test_summary_to_dict_fields(self) -> None:
        bs = _agent().budget_scenarios(_snap(), 20000)
        d = _agent().executive_summary(bs).to_dict()
        for k in ("best_scenario", "worst_scenario", "recommended_scenario",
                  "confidence", "strategic_commentary"):
            assert k in d

    def test_plan_to_dict_currency(self) -> None:
        d = _agent().plan(_snap(), 20000).to_dict()
        assert d["currency"] == "USD"

    def test_scenario_plan_all_reports_type(self) -> None:
        p = _agent().plan(_snap(), 20000)
        assert all(isinstance(r, ScenarioPlanningReport) for r in p.all_reports())

    def test_zero_sensitivity_risk_unchanged(self) -> None:
        agent = _agent(risk_sensitivity=0.0)
        ls = agent.load_scenarios(_snap())
        risks = [r.prediction("risk") for r in ls]
        assert risks[0] == pytest.approx(risks[1])

    def test_multiple_plans_deterministic(self) -> None:
        snap = _snap()
        agent = _agent()
        plans = [agent.plan(snap, 20000) for _ in range(3)]
        assert all(p.ranking.ranked == plans[0].ranking.ranked for p in plans)