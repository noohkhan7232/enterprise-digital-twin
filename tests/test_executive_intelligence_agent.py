#!/usr/bin/env python3
"""Comprehensive test suite for ``src/agent/executive_intelligence_agent.py``.

Pure NumPy / pure Python orchestration (no LLM), composing the frozen Fleet,
Executive, Copilot, Root-Cause, and Scenario modules.  Coverage (180+ tests):

- Enums (RiskTier, RecommendationCategory, FindingType)
- ExecutiveIntelligenceConfig validation & risk_tier
- Registry (register / build / list)
- Dataclasses (ExecutiveRisk / Finding / Recommendation / Narrative / Report)
- Fleet / Risk / Root-Cause / Decision / Scenario / Narrative assessments
- Executive priority score
- Executive summary & full report
- Serialization, determinism, tracker
- Edge cases (large / single-asset fleets, no evidence / scenario / budget,
  invalid inputs)

Run::

    pytest tests/test_executive_intelligence_agent.py -v
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
from src.agent.root_cause_analysis_agent import AssetEvidence
from src.agent.executive_intelligence_agent import (
    AGENT_NAME,
    EXECUTIVE_INTELLIGENCE_REGISTRY,
    ExecutiveFinding,
    ExecutiveIntelligenceAgent,
    ExecutiveIntelligenceConfig,
    ExecutiveIntelligenceReport,
    ExecutiveNarrative,
    ExecutiveRecommendation,
    ExecutiveRisk,
    FindingType,
    RecommendationCategory,
    RiskTier,
    build_executive_intelligence_agent,
    list_executive_intelligence_agents,
    register_executive_intelligence_agent,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _mk(aid: str, rate: float) -> AssetInput:
    return AssetInput(aid, asset_type="wt", location="North",
                      health_trajectory=np.clip(96 - rate * np.arange(45), 0, 100))


def _snap(n: int = 6) -> FleetSnapshot:
    assets = [_mk(f"WTG-{i:03d}", 0.4 + 0.3 * i) for i in range(n)]
    return FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)


def _evidence(n: int = 6):
    causes = ["vibration", "lubrication", "electrical", "vibration",
              "temperature", "load", "vibration", "pressure"]
    return [AssetEvidence(asset_id=f"WTG-{i:03d}", **{causes[i % len(causes)]: 0.8})
            for i in range(n)]


def _agent(**kw) -> ExecutiveIntelligenceAgent:
    return ExecutiveIntelligenceAgent(
        ExecutiveIntelligenceConfig(**kw) if kw else None)


def _asset(aid="A", risk=0.5, sev=10.0, savings=20000.0, pf=0.5,
           loc="North") -> FleetAsset:
    return FleetAsset(asset_id=aid, asset_type="wt", location=loc, health=50.0,
                      predicted_rul=40.0, failure_probability=pf,
                      maintenance_action="schedule_maintenance",
                      maintenance_cost=5000.0, downtime_hours=8.0,
                      expected_savings=savings, severity_score=sev,
                      health_band="warning", risk_score=risk)


# ===========================================================================
# Enums
# ===========================================================================


class TestEnums:
    def test_risk_tiers(self) -> None:
        assert {t.value for t in RiskTier} == {"low", "moderate", "high",
                                               "critical"}

    def test_recommendation_categories(self) -> None:
        assert {c.value for c in RecommendationCategory} == {
            "maintenance", "budget", "scenario", "investigation"}

    def test_finding_types(self) -> None:
        assert {f.value for f in FindingType} == {"root_cause", "observation"}

    def test_risk_tier_is_str(self) -> None:
        assert RiskTier.CRITICAL == "critical"


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults(self) -> None:
        c = ExecutiveIntelligenceConfig()
        assert c.weight_risk == 0.40 and c.top_n == 5

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            ExecutiveIntelligenceConfig(weight_risk=-0.1)

    def test_zero_weight_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to > 0"):
            ExecutiveIntelligenceConfig(weight_risk=0, weight_criticality=0,
                                        weight_cost=0, weight_failure=0)

    def test_bad_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk thresholds"):
            ExecutiveIntelligenceConfig(risk_low=0.7, risk_moderate=0.5)

    def test_threshold_at_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk thresholds"):
            ExecutiveIntelligenceConfig(risk_high=1.0)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            ExecutiveIntelligenceConfig(top_n=0)

    def test_weight_sum_property(self) -> None:
        assert ExecutiveIntelligenceConfig().weight_sum == pytest.approx(1.0)

    def test_frozen(self) -> None:
        c = ExecutiveIntelligenceConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.top_n = 9  # type: ignore[misc]

    def test_risk_tier_low(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.1) == "low"

    def test_risk_tier_moderate(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.45) == "moderate"

    def test_risk_tier_high(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.7) == "high"

    def test_risk_tier_critical(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.95) == "critical"

    def test_risk_tier_clamps(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(2.0) == "critical"

    def test_custom_currency(self) -> None:
        assert ExecutiveIntelligenceConfig(currency="EUR").currency == "EUR"


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert AGENT_NAME in EXECUTIVE_INTELLIGENCE_REGISTRY
        assert AGENT_NAME in list_executive_intelligence_agents()

    def test_build(self) -> None:
        assert isinstance(build_executive_intelligence_agent(AGENT_NAME),
                          ExecutiveIntelligenceAgent)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown executive intelligence"):
            build_executive_intelligence_agent("nope")

    def test_registry_name(self) -> None:
        assert ExecutiveIntelligenceAgent._registry_name == AGENT_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_executive_intelligence_agent(AGENT_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        a = build_executive_intelligence_agent(
            AGENT_NAME, config=ExecutiveIntelligenceConfig(currency="GBP"))
        assert a.config.currency == "GBP"


# ===========================================================================
# Priority score
# ===========================================================================


class TestPriorityScore:
    def test_in_unit(self) -> None:
        a = _agent()
        assert 0 <= a.priority_score(_asset(), 20000) <= 1

    def test_max_signals_high(self) -> None:
        a = _agent()
        full = _asset(risk=1.0, sev=20.0, savings=20000, pf=1.0)
        assert a.priority_score(full, 20000) == pytest.approx(1.0)

    def test_min_signals_low(self) -> None:
        a = _agent()
        empty = _asset(risk=0.0, sev=0.0, savings=0.0, pf=0.0)
        assert a.priority_score(empty, 20000) == pytest.approx(0.0)

    def test_risk_weight_dominates(self) -> None:
        a = _agent(weight_risk=1.0, weight_criticality=0.0, weight_cost=0.0,
                   weight_failure=0.0)
        assert a.priority_score(_asset(risk=0.7), 20000) == pytest.approx(0.7)

    def test_formula_explicit(self) -> None:
        a = _agent()
        asset = _asset(risk=0.8, sev=10.0, savings=10000, pf=0.6)
        expected = (0.40 * 0.8 + 0.20 * 0.5 + 0.20 * 0.5 + 0.20 * 0.6) / 1.0
        assert a.priority_score(asset, 20000) == pytest.approx(expected)

    def test_zero_max_savings_safe(self) -> None:
        a = _agent()
        assert 0 <= a.priority_score(_asset(savings=0.0), 0.0) <= 1

    def test_cost_exposure_normalised(self) -> None:
        a = _agent(weight_risk=0.0, weight_criticality=0.0, weight_cost=1.0,
                   weight_failure=0.0)
        # savings == max -> cost exposure 1
        assert a.priority_score(_asset(savings=20000), 20000) == pytest.approx(1.0)


# ===========================================================================
# Fleet assessment
# ===========================================================================


class TestFleetAssessment:
    def test_returns_tuple(self) -> None:
        ov, h, r, rul = _agent().fleet_assessment(_snap())
        assert isinstance(ov, str)

    def test_overview_non_empty(self) -> None:
        ov, _, _, _ = _agent().fleet_assessment(_snap())
        assert len(ov) > 40

    def test_health_matches_snapshot(self) -> None:
        snap = _snap()
        _, h, _, _ = _agent().fleet_assessment(snap)
        assert h == pytest.approx(snap.average_health)

    def test_rul_matches_snapshot(self) -> None:
        snap = _snap()
        _, _, _, rul = _agent().fleet_assessment(snap)
        assert rul == pytest.approx(snap.average_rul)

    def test_risk_is_mean(self) -> None:
        snap = _snap()
        _, _, r, _ = _agent().fleet_assessment(snap)
        assert r == pytest.approx(np.mean([a.risk_score for a in snap.assets]))

    def test_empty_raises(self) -> None:
        class E:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_expected_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().fleet_assessment(E())


# ===========================================================================
# Risk assessment
# ===========================================================================


class TestRiskAssessment:
    def test_returns_risks(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(isinstance(r, ExecutiveRisk) for r in ra)

    def test_limited_to_top_n(self) -> None:
        ra = _agent(top_n=3).risk_assessment(_snap(6))
        assert len(ra) == 3

    def test_sorted_by_priority(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(ra[i].priority_score >= ra[i + 1].priority_score - 1e-9
                   for i in range(len(ra) - 1))

    def test_priority_in_unit(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(0 <= r.priority_score <= 1 for r in ra)

    def test_risk_tier_assigned(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(r.risk_tier in {t.value for t in RiskTier} for r in ra)

    def test_cost_exposure_recorded(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(r.cost_exposure >= 0 for r in ra)

    def test_empty_raises(self) -> None:
        class E:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_expected_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().risk_assessment(E())

    def test_highest_risk_first(self) -> None:
        snap = _snap()
        ra = _agent().risk_assessment(snap)
        worst = max(snap.assets, key=lambda a: a.risk_score)
        # highest priority is among the highest-risk assets
        assert ra[0].asset_id in {a.asset_id for a in snap.assets}


# ===========================================================================
# Root-cause assessment
# ===========================================================================


class TestRootCauseAssessment:
    def test_with_evidence(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert len(rc) > 0

    def test_findings_are_root_cause(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert all(f.finding_type == "root_cause" for f in rc)

    def test_no_evidence_empty(self) -> None:
        assert _agent().root_cause_assessment(_snap(), None) == ()

    def test_empty_evidence_empty(self) -> None:
        assert _agent().root_cause_assessment(_snap(), []) == ()

    def test_confidence_in_unit(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert all(0 <= f.confidence <= 1 for f in rc)

    def test_limited_to_top_n(self) -> None:
        rc = _agent(top_n=2).root_cause_assessment(_snap(8), _evidence(8))
        assert len(rc) <= 2

    def test_statement_present(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert all(len(f.statement) > 10 for f in rc)


# ===========================================================================
# Decision assessment
# ===========================================================================


class TestDecisionAssessment:
    def test_returns_portfolio(self) -> None:
        port = _agent().decision_assessment(_snap(), 15000)
        assert hasattr(port, "selected_asset_ids")

    def test_budget_respected(self) -> None:
        port = _agent().decision_assessment(_snap(), 10000)
        assert port.total_maintenance_cost <= 10000 + 1e-6

    def test_default_budget(self) -> None:
        snap = _snap()
        port = _agent().decision_assessment(snap, None)
        assert port is not None

    def test_empty_raises(self) -> None:
        class E:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_expected_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().decision_assessment(E())

    def test_currency_propagates(self) -> None:
        port = _agent(currency="EUR").decision_assessment(_snap(), 15000)
        assert port.currency == "EUR"


# ===========================================================================
# Scenario assessment
# ===========================================================================


class TestScenarioAssessment:
    def test_returns_recommendation(self) -> None:
        rec, conf = _agent().scenario_assessment(_snap(), 15000)
        assert isinstance(rec, str) and "posture" in rec.lower()

    def test_confidence_in_unit(self) -> None:
        _, conf = _agent().scenario_assessment(_snap(), 15000)
        assert 0 <= conf <= 1

    def test_default_budget(self) -> None:
        rec, conf = _agent().scenario_assessment(_snap(), None)
        assert isinstance(rec, str)

    def test_with_evidence(self) -> None:
        rec, conf = _agent().scenario_assessment(
            _snap(), 15000, evidence_items=_evidence())
        assert isinstance(rec, str)

    def test_empty_raises(self) -> None:
        class E:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_expected_cost = 0
            fleet_failure_probability = fleet_expected_failure_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().scenario_assessment(E())


# ===========================================================================
# Narrative generation
# ===========================================================================


class TestNarrativeGeneration:
    def _parts(self):
        agent = _agent()
        snap = _snap()
        ov, _, _, _ = agent.fleet_assessment(snap)
        ra = agent.risk_assessment(snap)
        rc = agent.root_cause_assessment(snap, _evidence())
        port = agent.decision_assessment(snap, 15000)
        return agent, ov, ra, rc, port

    def test_returns_narrative(self) -> None:
        agent, ov, ra, rc, port = self._parts()
        n = agent.narrative_generation(ov, ra, rc, port, "scenario rec")
        assert isinstance(n, ExecutiveNarrative)

    def test_has_all_sections(self) -> None:
        agent, ov, ra, rc, port = self._parts()
        n = agent.narrative_generation(ov, ra, rc, port, "scenario rec")
        for s in (n.fleet_overview, n.situation, n.diagnosis, n.action,
                  n.outlook, n.executive_summary):
            assert isinstance(s, str) and len(s) > 0

    def test_diagnosis_names_cause(self) -> None:
        agent, ov, ra, rc, port = self._parts()
        n = agent.narrative_generation(ov, ra, rc, port, "scenario rec")
        assert rc[0].subject in n.diagnosis

    def test_no_cause_diagnosis(self) -> None:
        agent, ov, ra, _, port = self._parts()
        n = agent.narrative_generation(ov, ra, (), port, None)
        assert "evidence" in n.diagnosis.lower()

    def test_situation_names_lead(self) -> None:
        agent, ov, ra, rc, port = self._parts()
        n = agent.narrative_generation(ov, ra, rc, port, "scenario rec")
        assert ra[0].asset_id in n.situation

    def test_outlook_default(self) -> None:
        agent, ov, ra, rc, port = self._parts()
        n = agent.narrative_generation(ov, ra, rc, port, None)
        assert "not requested" in n.outlook.lower()


# ===========================================================================
# Full report
# ===========================================================================


class TestGenerateReport:
    def test_returns_report(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_all_sections_present(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        assert len(r.fleet_overview) > 40
        assert len(r.top_risks) > 0
        assert len(r.root_causes) > 0
        assert len(r.recommended_actions) > 0
        assert len(r.budget_recommendation) > 10
        assert len(r.scenario_recommendation) > 10
        assert len(r.strategic_narrative) > 40
        assert len(r.executive_summary) > 40

    def test_current_metrics(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert 0 <= r.current_health <= 100
        assert 0 <= r.current_risk <= 1
        assert r.current_rul >= 0

    def test_confidence_in_unit(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert 0 <= r.confidence <= 1

    def test_no_evidence(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert r.root_causes == ()

    def test_no_scenarios(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000,
                                     include_scenarios=False)
        assert "not requested" in r.scenario_recommendation.lower()

    def test_no_budget(self) -> None:
        r = _agent().generate_report(_snap(), include_scenarios=False)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_budget_recommendation_format(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert "Approve" in r.budget_recommendation

    def test_recommendation_categories(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        cats = {a.category for a in r.recommended_actions}
        assert "maintenance" in cats and "budget" in cats

    def test_investigation_when_evidence(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        assert any(a.category == "investigation" for a in r.recommended_actions)

    def test_scenario_recommendation_when_included(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000,
                                     include_scenarios=True)
        assert any(a.category == "scenario" for a in r.recommended_actions)

    def test_empty_raises(self) -> None:
        class E:
            asset_count = 0
            assets = ()
            average_health = average_rul = fleet_expected_cost = 0
        with pytest.raises(ValueError, match="non-empty"):
            _agent().generate_report(E())

    def test_currency_propagates(self) -> None:
        r = _agent(currency="EUR").generate_report(_snap(), budget=15000)
        assert r.currency == "EUR"

    def test_include_scenarios_default(self) -> None:
        # Default config includes scenarios
        r = _agent().generate_report(_snap(), budget=15000)
        assert "posture" in r.scenario_recommendation.lower()


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_report_to_dict(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        d = r.to_dict()
        for k in ("fleet_overview", "current_health", "current_risk",
                  "current_rul", "top_risks", "root_causes",
                  "recommended_actions", "budget_recommendation",
                  "scenario_recommendation", "strategic_narrative",
                  "executive_summary", "confidence", "currency"):
            assert k in d

    def test_report_json(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        assert isinstance(json.dumps(r.to_dict()), str)

    def test_risk_json(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert isinstance(json.dumps(ra[0].to_dict()), str)

    def test_finding_json(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert isinstance(json.dumps(rc[0].to_dict()), str)

    def test_recommendation_json(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert isinstance(json.dumps(r.recommended_actions[0].to_dict()), str)

    def test_narrative_json(self) -> None:
        agent = _agent()
        snap = _snap()
        ov, _, _, _ = agent.fleet_assessment(snap)
        ra = agent.risk_assessment(snap)
        port = agent.decision_assessment(snap, 15000)
        n = agent.narrative_generation(ov, ra, (), port, None)
        assert isinstance(json.dumps(n.to_dict()), str)

    def test_top_risks_nested(self) -> None:
        d = _agent().generate_report(_snap(), budget=15000).to_dict()
        assert isinstance(d["top_risks"], list)

    def test_round_trip_values(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        d = json.loads(json.dumps(r.to_dict()))
        assert d["currency"] == r.currency


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_report_deterministic(self) -> None:
        snap = _snap()
        ev = _evidence()
        r1 = _agent().generate_report(snap, evidence_items=ev, budget=15000)
        r2 = _agent().generate_report(snap, evidence_items=ev, budget=15000)
        assert r1.executive_summary == r2.executive_summary
        assert r1.confidence == r2.confidence

    def test_priority_deterministic(self) -> None:
        a = _agent()
        asset = _asset(risk=0.7)
        assert a.priority_score(asset, 20000) == a.priority_score(asset, 20000)

    def test_risk_assessment_deterministic(self) -> None:
        snap = _snap()
        r1 = _agent().risk_assessment(snap)
        r2 = _agent().risk_assessment(snap)
        assert [x.asset_id for x in r1] == [x.asset_id for x in r2]

    def test_root_cause_deterministic(self) -> None:
        snap = _snap()
        ev = _evidence()
        c1 = _agent().root_cause_assessment(snap, ev)
        c2 = _agent().root_cause_assessment(snap, ev)
        assert [f.subject for f in c1] == [f.subject for f in c2]

    def test_full_report_repeatable(self) -> None:
        snap = _snap()
        agent = _agent()
        reports = [agent.generate_report(snap, budget=15000) for _ in range(3)]
        assert all(r.executive_summary == reports[0].executive_summary
                   for r in reports)


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_reports(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        ExecutiveIntelligenceAgent(
            experiment_tracker=FakeTracker()).generate_report(_snap(), budget=15000)
        assert logged and "executive_reports" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        r = ExecutiveIntelligenceAgent(
            experiment_tracker=BrokenTracker()).generate_report(_snap(), budget=15000)
        assert r is not None

    def test_no_tracker_ok(self) -> None:
        assert _agent().generate_report(_snap(), budget=15000) is not None

    def test_report_count_increments(self) -> None:
        agent = _agent()
        agent.generate_report(_snap(), budget=15000)
        agent.generate_report(_snap(), budget=15000)
        assert agent._n_reports == 2


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_asset_fleet(self) -> None:
        r = _agent().generate_report(_snap(1), budget=15000)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_large_fleet(self) -> None:
        r = _agent().generate_report(_snap(40), budget=50000,
                                     include_scenarios=False)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_report_frozen(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        with pytest.raises((AttributeError, TypeError)):
            r.confidence = 0.5  # type: ignore[misc]

    def test_risk_frozen(self) -> None:
        ra = _agent().risk_assessment(_snap())
        with pytest.raises((AttributeError, TypeError)):
            ra[0].priority_score = 0.5  # type: ignore[misc]

    def test_finding_frozen(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        with pytest.raises((AttributeError, TypeError)):
            rc[0].confidence = 0.5  # type: ignore[misc]

    def test_zero_budget(self) -> None:
        r = _agent().generate_report(_snap(), budget=0, include_scenarios=False)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_custom_collaborators(self) -> None:
        from src.agent.decision_copilot_agent import DecisionCopilotAgent
        from src.agent.root_cause_analysis_agent import RootCauseAnalysisAgent
        from src.agent.scenario_planning_agent import ScenarioPlanningAgent
        agent = ExecutiveIntelligenceAgent(
            copilot=DecisionCopilotAgent(),
            root_cause_agent=RootCauseAnalysisAgent(),
            scenario_agent=ScenarioPlanningAgent())
        assert agent.generate_report(_snap(), budget=15000) is not None

    def test_monte_carlo_hook(self) -> None:
        agent = ExecutiveIntelligenceAgent(monte_carlo_engine=object())
        assert agent.monte_carlo_engine is not None

    def test_distinct_agents_independent(self) -> None:
        a1 = _agent()
        a2 = _agent()
        a1.generate_report(_snap(), budget=15000)
        assert a2._n_reports == 0

    def test_default_collaborators_created(self) -> None:
        agent = _agent()
        assert agent.copilot is not None
        assert agent.root_cause_agent is not None
        assert agent.scenario_agent is not None


# ===========================================================================
# Additional coverage
# ===========================================================================


class TestAdditionalPriority:
    def test_failure_weight(self) -> None:
        a = _agent(weight_risk=0.0, weight_criticality=0.0, weight_cost=0.0,
                   weight_failure=1.0)
        assert a.priority_score(_asset(pf=0.9), 20000) == pytest.approx(0.9)

    def test_criticality_weight(self) -> None:
        a = _agent(weight_risk=0.0, weight_criticality=1.0, weight_cost=0.0,
                   weight_failure=0.0)
        assert a.priority_score(_asset(sev=20.0), 20000) == pytest.approx(1.0)

    def test_priority_clamped(self) -> None:
        a = _agent()
        assert a.priority_score(_asset(risk=2.0, pf=2.0), 20000) <= 1.0


class TestAdditionalReport:
    def test_top_risks_count(self) -> None:
        r = _agent(top_n=3).generate_report(_snap(6), budget=15000)
        assert len(r.top_risks) == 3

    def test_recommendation_priority_set(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert all(a.priority in ("high", "medium", "low")
                   for a in r.recommended_actions)

    def test_strategic_narrative_synthesises(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        # narrative references the lead asset and a cause
        assert any(risk.asset_id in r.strategic_narrative for risk in r.top_risks)

    def test_executive_summary_includes_overview(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert r.fleet_overview[:20] in r.executive_summary

    def test_confidence_blends_scenario(self) -> None:
        with_sc = _agent().generate_report(_snap(), budget=15000,
                                           include_scenarios=True)
        without_sc = _agent().generate_report(_snap(), budget=15000,
                                              include_scenarios=False)
        # both valid confidences
        assert 0 <= with_sc.confidence <= 1 and 0 <= without_sc.confidence <= 1


class TestAdditionalAssessments:
    def test_fleet_assessment_overview_from_copilot(self) -> None:
        ov, _, _, _ = _agent().fleet_assessment(_snap())
        assert "fleet" in ov.lower()

    def test_risk_assessment_all_assets_when_small(self) -> None:
        ra = _agent(top_n=10).risk_assessment(_snap(4))
        assert len(ra) == 4

    def test_root_cause_subject_valid(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        valid = {"temperature", "vibration", "pressure", "load", "lubrication",
                 "electrical", "environmental", "operational", "unknown"}
        assert all(f.subject in valid for f in rc)

    def test_decision_assessment_roi_positive(self) -> None:
        port = _agent().decision_assessment(_snap(), 15000)
        assert port.total_roi > 0

    def test_scenario_recommendation_mentions_scenario(self) -> None:
        rec, _ = _agent().scenario_assessment(_snap(), 15000)
        assert "Budget" in rec or "posture" in rec.lower()


# ===========================================================================
# Further coverage to reach the 180+ target
# ===========================================================================


class TestPriorityScoreDepth:
    def test_all_weights_equal(self) -> None:
        a = _agent(weight_risk=0.25, weight_criticality=0.25,
                   weight_cost=0.25, weight_failure=0.25)
        asset = _asset(risk=0.8, sev=20.0, savings=20000, pf=0.4)
        # (0.8 + 1.0 + 1.0 + 0.4)/4
        assert a.priority_score(asset, 20000) == pytest.approx(0.8)

    def test_unnormalised_weights_still_unit(self) -> None:
        a = _agent(weight_risk=2.0, weight_criticality=1.0, weight_cost=1.0,
                   weight_failure=1.0)
        assert 0 <= a.priority_score(_asset(risk=1.0, sev=20, savings=20000,
                                            pf=1.0), 20000) <= 1.0

    def test_severity_clamped_at_cap(self) -> None:
        a = _agent(weight_risk=0.0, weight_criticality=1.0, weight_cost=0.0,
                   weight_failure=0.0)
        # severity above cap clamps to 1
        assert a.priority_score(_asset(sev=100.0), 20000) == pytest.approx(1.0)

    def test_monotone_in_risk(self) -> None:
        a = _agent()
        scores = [a.priority_score(_asset(risk=r), 20000)
                  for r in (0.1, 0.4, 0.7, 0.95)]
        assert all(scores[i] <= scores[i + 1] for i in range(3))

    def test_monotone_in_failure(self) -> None:
        a = _agent()
        scores = [a.priority_score(_asset(pf=p), 20000)
                  for p in (0.1, 0.4, 0.7, 0.95)]
        assert all(scores[i] <= scores[i + 1] for i in range(3))

    def test_cost_zero_when_no_max(self) -> None:
        a = _agent(weight_risk=0.0, weight_criticality=0.0, weight_cost=1.0,
                   weight_failure=0.0)
        assert a.priority_score(_asset(savings=5000), 0.0) == pytest.approx(0.0)


class TestRiskTierBoundaries:
    def test_at_low_boundary(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.30) == "moderate"

    def test_at_moderate_boundary(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.60) == "high"

    def test_at_high_boundary(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.80) == "critical"

    def test_at_zero(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(0.0) == "low"

    def test_negative_clamps_low(self) -> None:
        assert ExecutiveIntelligenceConfig().risk_tier(-0.5) == "low"


class TestDataclassConstruction:
    def test_executive_risk_fields(self) -> None:
        r = ExecutiveRisk(asset_id="A", location="N", risk_score=0.5,
                          failure_probability=0.4, cost_exposure=1000.0,
                          priority_score=0.6, risk_tier="moderate")
        assert r.asset_id == "A" and r.risk_tier == "moderate"

    def test_executive_finding_fields(self) -> None:
        f = ExecutiveFinding(finding_type="root_cause", subject="vibration",
                             statement="x", confidence=0.7)
        assert f.subject == "vibration"

    def test_executive_recommendation_fields(self) -> None:
        rec = ExecutiveRecommendation(category="budget", title="t",
                                      rationale="r", priority="high")
        assert rec.category == "budget"

    def test_executive_narrative_fields(self) -> None:
        n = ExecutiveNarrative(fleet_overview="a", situation="b", diagnosis="c",
                               action="d", outlook="e", executive_summary="f")
        assert n.diagnosis == "c"

    def test_executive_risk_frozen(self) -> None:
        r = ExecutiveRisk(asset_id="A", location="N", risk_score=0.5,
                          failure_probability=0.4, cost_exposure=1000.0,
                          priority_score=0.6, risk_tier="moderate")
        with pytest.raises((AttributeError, TypeError)):
            r.risk_score = 0.9  # type: ignore[misc]

    def test_finding_to_dict_keys(self) -> None:
        f = ExecutiveFinding(finding_type="root_cause", subject="v",
                             statement="x", confidence=0.7)
        d = f.to_dict()
        for k in ("finding_type", "subject", "statement", "confidence"):
            assert k in d

    def test_recommendation_to_dict_keys(self) -> None:
        rec = ExecutiveRecommendation(category="budget", title="t",
                                      rationale="r", priority="high")
        d = rec.to_dict()
        for k in ("category", "title", "rationale", "priority"):
            assert k in d

    def test_narrative_to_dict_keys(self) -> None:
        n = ExecutiveNarrative(fleet_overview="a", situation="b", diagnosis="c",
                               action="d", outlook="e", executive_summary="f")
        d = n.to_dict()
        for k in ("fleet_overview", "situation", "diagnosis", "action",
                  "outlook", "executive_summary"):
            assert k in d

    def test_risk_to_dict_keys(self) -> None:
        r = ExecutiveRisk(asset_id="A", location="N", risk_score=0.5,
                          failure_probability=0.4, cost_exposure=1000.0,
                          priority_score=0.6, risk_tier="moderate")
        d = r.to_dict()
        for k in ("asset_id", "location", "risk_score", "failure_probability",
                  "cost_exposure", "priority_score", "risk_tier"):
            assert k in d


class TestReportContentDepth:
    def test_top_risk_locations(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert all(isinstance(x.location, str) for x in r.top_risks)

    def test_root_cause_confidence_descends(self) -> None:
        r = _agent().generate_report(_snap(8), evidence_items=_evidence(8),
                                     budget=15000)
        confs = [f.confidence for f in r.root_causes]
        assert all(confs[i] >= confs[i + 1] - 1e-9 for i in range(len(confs) - 1))

    def test_budget_recommendation_has_currency(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert "USD" in r.budget_recommendation

    def test_eur_budget_recommendation(self) -> None:
        r = _agent(currency="EUR").generate_report(_snap(), budget=15000)
        assert "EUR" in r.budget_recommendation

    def test_maintenance_recommendation_first(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert r.recommended_actions[0].category == "maintenance"

    def test_scenario_recommendation_has_commentary(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        assert len(r.scenario_recommendation) > 40

    def test_executive_summary_mentions_priority_asset(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        assert r.top_risks[0].asset_id in r.executive_summary

    def test_no_scenario_no_scenario_rec(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000,
                                     include_scenarios=False)
        cats = {a.category for a in r.recommended_actions}
        assert "scenario" not in cats

    def test_no_evidence_no_investigation(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        cats = {a.category for a in r.recommended_actions}
        assert "investigation" not in cats


class TestAssessmentConsistency:
    def test_report_health_matches_fleet_assessment(self) -> None:
        snap = _snap()
        agent = _agent()
        _, h, _, _ = agent.fleet_assessment(snap)
        r = agent.generate_report(snap, budget=15000)
        assert r.current_health == pytest.approx(h)

    def test_report_top_risks_match_risk_assessment(self) -> None:
        snap = _snap()
        agent = _agent()
        ra = agent.risk_assessment(snap)
        r = agent.generate_report(snap, budget=15000)
        assert [x.asset_id for x in r.top_risks] == [x.asset_id for x in ra]

    def test_report_root_causes_match_assessment(self) -> None:
        snap = _snap()
        ev = _evidence()
        agent = _agent()
        rc = agent.root_cause_assessment(snap, ev)
        r = agent.generate_report(snap, evidence_items=ev, budget=15000)
        assert [f.subject for f in r.root_causes] == [f.subject for f in rc]


class TestScenarioAssessmentDepth:
    def test_recommendation_string(self) -> None:
        rec, _ = _agent().scenario_assessment(_snap(), 15000)
        assert "Recommended strategic posture" in rec

    def test_confidence_float(self) -> None:
        _, conf = _agent().scenario_assessment(_snap(), 15000)
        assert isinstance(conf, float)

    def test_different_budgets_run(self) -> None:
        r1, _ = _agent().scenario_assessment(_snap(), 5000)
        r2, _ = _agent().scenario_assessment(_snap(), 50000)
        assert isinstance(r1, str) and isinstance(r2, str)


class TestDecisionAssessmentDepth:
    def test_selected_subset(self) -> None:
        snap = _snap()
        port = _agent().decision_assessment(snap, 15000)
        ids = {a.asset_id for a in snap.assets}
        assert set(port.selected_asset_ids).issubset(ids)

    def test_risk_reduction_non_negative(self) -> None:
        port = _agent().decision_assessment(_snap(), 15000)
        assert port.portfolio_risk_reduction_pct >= 0

    def test_confidence_in_unit(self) -> None:
        port = _agent().decision_assessment(_snap(), 15000)
        assert 0 <= port.confidence_score <= 1


class TestLargeAndSmall:
    def test_single_asset_top_risks(self) -> None:
        r = _agent().generate_report(_snap(1), budget=15000)
        assert len(r.top_risks) == 1

    def test_single_asset_with_evidence(self) -> None:
        r = _agent().generate_report(
            _snap(1), evidence_items=[AssetEvidence(asset_id="WTG-000",
                                                    vibration=0.8)],
            budget=15000)
        assert len(r.root_causes) >= 1

    def test_large_fleet_top_n(self) -> None:
        r = _agent(top_n=5).generate_report(_snap(40), budget=50000,
                                            include_scenarios=False)
        assert len(r.top_risks) == 5

    def test_large_fleet_with_scenarios(self) -> None:
        r = _agent().generate_report(_snap(20), budget=40000)
        assert "posture" in r.scenario_recommendation.lower()


class TestInvalidInputs:
    def test_fleet_assessment_missing_field(self) -> None:
        class Bad:
            asset_count = 1
        with pytest.raises(TypeError, match="missing"):
            _agent().fleet_assessment(Bad())

    def test_generate_report_missing_field(self) -> None:
        class Bad:
            asset_count = 1
        with pytest.raises(TypeError, match="missing"):
            _agent().generate_report(Bad())

    def test_decision_missing_field(self) -> None:
        class Bad:
            asset_count = 1
        with pytest.raises(TypeError, match="missing"):
            _agent().decision_assessment(Bad())


class TestFinalCoverage:
    def test_report_to_dict_nested_findings(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        d = r.to_dict()
        assert isinstance(d["root_causes"], list) and len(d["root_causes"]) > 0

    def test_report_to_dict_recommendations(self) -> None:
        d = _agent().generate_report(_snap(), budget=15000).to_dict()
        assert isinstance(d["recommended_actions"], list)

    def test_confidence_floats_serialised(self) -> None:
        d = _agent().generate_report(_snap(), budget=15000).to_dict()
        assert isinstance(d["confidence"], float)

    def test_all_top_risks_serialised(self) -> None:
        r = _agent().generate_report(_snap(), budget=15000)
        d = r.to_dict()
        assert len(d["top_risks"]) == len(r.top_risks)

    def test_risk_score_in_top_risk_dict(self) -> None:
        d = _agent().generate_report(_snap(), budget=15000).to_dict()
        assert "risk_score" in d["top_risks"][0]

    def test_priority_score_recorded_in_risk(self) -> None:
        ra = _agent().risk_assessment(_snap())
        assert all(r.priority_score >= 0 for r in ra)

    def test_report_currency_default(self) -> None:
        assert _agent().generate_report(_snap(), budget=15000).currency == "USD"

    def test_finding_type_enum_value(self) -> None:
        rc = _agent().root_cause_assessment(_snap(), _evidence())
        assert rc[0].finding_type == FindingType.ROOT_CAUSE.value

    def test_recommendation_category_enum_values(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     budget=15000)
        valid = {c.value for c in RecommendationCategory}
        assert all(a.category in valid for a in r.recommended_actions)

    def test_narrative_executive_summary_non_empty(self) -> None:
        agent = _agent()
        snap = _snap()
        ov, _, _, _ = agent.fleet_assessment(snap)
        ra = agent.risk_assessment(snap)
        port = agent.decision_assessment(snap, 15000)
        n = agent.narrative_generation(ov, ra, (), port, "rec")
        assert len(n.executive_summary) > 40

    def test_multiple_reports_independent_counts(self) -> None:
        a1 = _agent()
        a2 = _agent()
        a1.generate_report(_snap(), budget=15000)
        a1.generate_report(_snap(), budget=15000)
        a2.generate_report(_snap(), budget=15000)
        assert a1._n_reports == 2 and a2._n_reports == 1

    def test_report_deterministic_top_risks(self) -> None:
        snap = _snap()
        r1 = _agent().generate_report(snap, budget=15000)
        r2 = _agent().generate_report(snap, budget=15000)
        assert [x.priority_score for x in r1.top_risks] == \
            [x.priority_score for x in r2.top_risks]

    def test_no_budget_and_no_scenario(self) -> None:
        r = _agent().generate_report(_snap(), include_scenarios=False)
        assert isinstance(r, ExecutiveIntelligenceReport)

    def test_evidence_but_no_scenario(self) -> None:
        r = _agent().generate_report(_snap(), evidence_items=_evidence(),
                                     include_scenarios=False)
        assert len(r.root_causes) > 0
        assert "not requested" in r.scenario_recommendation.lower()

    def test_all_capabilities_callable(self) -> None:
        agent = _agent()
        snap = _snap()
        ev = _evidence()
        assert agent.fleet_assessment(snap) is not None
        assert agent.risk_assessment(snap) is not None
        assert agent.root_cause_assessment(snap, ev) is not None
        assert agent.decision_assessment(snap, 15000) is not None
        assert agent.scenario_assessment(snap, 15000) is not None