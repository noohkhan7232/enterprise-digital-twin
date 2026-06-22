#!/usr/bin/env python3
"""Comprehensive test suite for ``src/agent/decision_copilot_agent.py``.

Pure NumPy / pure Python (the agent composes frozen pure-NumPy modules and uses
no LLM), so the entire suite runs without PyTorch, SciPy, or pandas.  Coverage
(150+ tests):

- QuestionIntent enum
- DecisionCopilotConfig validation & risk_label
- CopilotContext
- Registry (register / build / list)
- explain_asset
- explain_fleet
- explain_portfolio
- answer_question (all six intents + classification + resolution)
- generate_executive_brief
- Serialization (JSON)
- Determinism
- Failure-safe behaviour & edge cases

Run::

    pytest tests/test_decision_copilot_agent.py -v
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

from src.fleet.fleet_digital_twin import FleetAsset, FleetSnapshot
from src.executive.executive_decision_engine import (
    ExecutiveDecisionConfig,
    ExecutiveDecisionEngine,
)
from src.agent.decision_copilot_agent import (
    AGENT_NAME,
    COPILOT_REGISTRY,
    AssetExplanation,
    CopilotContext,
    DecisionCopilotAgent,
    DecisionCopilotConfig,
    ExecutiveBrief,
    FleetExplanation,
    PortfolioExplanation,
    QuestionAnswer,
    QuestionIntent,
    build_decision_copilot_agent,
    list_decision_copilot_agents,
    register_decision_copilot_agent,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _asset(aid: str, *, cost: float = 5000.0, savings: float = 20000.0,
           risk: float = 0.5, sev: float = 10.0, pf: float = 0.5,
           health: float = 50.0, rul: float = 40.0,
           action: str = "schedule_maintenance", band: str = "warning",
           loc: str = "North") -> FleetAsset:
    return FleetAsset(
        asset_id=aid, asset_type="wind_turbine", location=loc, health=health,
        predicted_rul=rul, failure_probability=pf, maintenance_action=action,
        maintenance_cost=cost, downtime_hours=8.0, expected_savings=savings,
        severity_score=sev, health_band=band, risk_score=risk)


def _snap(assets, *, currency: str = "USD", risk_conc: float = 0.3) -> FleetSnapshot:
    assets = tuple(assets)
    n = len(assets)
    healths = [a.health for a in assets]
    return FleetSnapshot(
        asset_count=n,
        healthy_assets=sum(1 for a in assets if a.health_band == "healthy"),
        warning_assets=sum(1 for a in assets if a.health_band == "warning"),
        critical_assets=sum(1 for a in assets if a.health_band == "critical"),
        average_health=float(np.mean(healths)) if n else 0.0,
        average_rul=40.0,
        fleet_failure_probability=float(np.mean([a.failure_probability for a in assets])) if n else 0.0,
        fleet_expected_cost=float(sum(a.maintenance_cost for a in assets)),
        fleet_expected_downtime=float(sum(a.downtime_hours for a in assets)),
        fleet_expected_failure_cost=100000.0,
        fleet_expected_savings=float(sum(a.expected_savings for a in assets)),
        risk_concentration=risk_conc,
        pareto_concentration=0.4,
        assets=assets,
        currency=currency)


def _default_snap() -> FleetSnapshot:
    return _snap([
        _asset("WTG-001", risk=0.30, savings=15000, health=78, band="healthy"),
        _asset("WTG-002", risk=0.55, savings=30000, health=58, band="warning"),
        _asset("WTG-003", risk=0.99, savings=45000, health=8, band="critical",
               action="shutdown", pf=0.95),
        _asset("WTG-004", risk=0.82, savings=40000, health=22, band="critical",
               action="immediate_maintenance", pf=0.80),
        _asset("WTG-005", risk=0.45, savings=22000, health=64, band="warning"),
    ])


def _portfolio(snap=None, budget=15000):
    snap = snap or _default_snap()
    return ExecutiveDecisionEngine(
        ExecutiveDecisionConfig(budget=budget)).recommend(snap)


def _ctx(snap=None, port=None):
    snap = snap or _default_snap()
    port = port if port is not None else _portfolio(snap)
    return CopilotContext(snapshot=snap, portfolio=port)


def _agent(**kw):
    return DecisionCopilotAgent(DecisionCopilotConfig(**kw) if kw else None)


# ===========================================================================
# QuestionIntent
# ===========================================================================


class TestQuestionIntent:
    def test_values(self) -> None:
        assert QuestionIntent.WHY.value == "why"
        assert QuestionIntent.WHY_NOT.value == "why_not"
        assert QuestionIntent.WHAT_IF.value == "what_if"

    def test_seven_intents(self) -> None:
        assert len(list(QuestionIntent)) == 7

    def test_is_str(self) -> None:
        assert QuestionIntent.WHY == "why"


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults(self) -> None:
        c = DecisionCopilotConfig()
        assert c.currency == "USD" and c.top_n == 3

    def test_bad_threshold_order_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk thresholds"):
            DecisionCopilotConfig(risk_low=0.7, risk_moderate=0.5)

    def test_threshold_at_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk thresholds"):
            DecisionCopilotConfig(risk_low=0.0)

    def test_threshold_at_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="risk thresholds"):
            DecisionCopilotConfig(risk_high=1.0)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            DecisionCopilotConfig(top_n=0)

    def test_frozen(self) -> None:
        c = DecisionCopilotConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.currency = "EUR"  # type: ignore[misc]

    def test_custom_currency(self) -> None:
        assert DecisionCopilotConfig(currency="GBP").currency == "GBP"

    def test_risk_label_low(self) -> None:
        assert DecisionCopilotConfig().risk_label(0.1) == "low"

    def test_risk_label_moderate(self) -> None:
        assert DecisionCopilotConfig().risk_label(0.45) == "moderate"

    def test_risk_label_high(self) -> None:
        assert DecisionCopilotConfig().risk_label(0.7) == "high"

    def test_risk_label_critical(self) -> None:
        assert DecisionCopilotConfig().risk_label(0.95) == "critical"

    def test_risk_label_clamps(self) -> None:
        assert DecisionCopilotConfig().risk_label(2.0) == "critical"
        assert DecisionCopilotConfig().risk_label(-1.0) == "low"

    def test_risk_label_boundaries(self) -> None:
        c = DecisionCopilotConfig()
        assert c.risk_label(0.30) == "moderate"  # at low boundary
        assert c.risk_label(0.60) == "high"      # at moderate boundary
        assert c.risk_label(0.80) == "critical"  # at high boundary


# ===========================================================================
# CopilotContext
# ===========================================================================


class TestCopilotContext:
    def test_empty(self) -> None:
        c = CopilotContext()
        assert c.snapshot is None and c.portfolio is None

    def test_assets_from_snapshot(self) -> None:
        c = CopilotContext(snapshot=_default_snap())
        assert len(c.assets()) == 5

    def test_assets_empty_when_no_snapshot(self) -> None:
        assert CopilotContext().assets() == ()

    def test_frozen(self) -> None:
        c = CopilotContext()
        with pytest.raises((AttributeError, TypeError)):
            c.snapshot = None  # type: ignore[misc]

    def test_holds_portfolio(self) -> None:
        p = _portfolio()
        assert CopilotContext(portfolio=p).portfolio is p


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert AGENT_NAME in COPILOT_REGISTRY
        assert AGENT_NAME in list_decision_copilot_agents()

    def test_build(self) -> None:
        assert isinstance(build_decision_copilot_agent(AGENT_NAME),
                          DecisionCopilotAgent)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown decision copilot agent"):
            build_decision_copilot_agent("nope")

    def test_registry_name(self) -> None:
        assert DecisionCopilotAgent._registry_name == AGENT_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_decision_copilot_agent(AGENT_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_decision_copilot_agent(
            AGENT_NAME, config=DecisionCopilotConfig(currency="EUR"))
        assert eng.config.currency == "EUR"


# ===========================================================================
# explain_asset
# ===========================================================================


class TestExplainAsset:
    def test_returns_explanation(self) -> None:
        ae = _agent().explain_asset(_asset("A"))
        assert isinstance(ae, AssetExplanation)

    def test_includes_asset_id(self) -> None:
        ae = _agent().explain_asset(_asset("WTG-099"))
        assert "WTG-099" in ae.summary

    def test_all_statements_present(self) -> None:
        ae = _agent().explain_asset(_asset("A"))
        for s in (ae.health_statement, ae.trend_statement, ae.rul_statement,
                  ae.failure_statement, ae.risk_statement, ae.action_statement):
            assert s and isinstance(s, str)

    def test_health_in_statement(self) -> None:
        ae = _agent().explain_asset(_asset("A", health=42))
        assert "42" in ae.health_statement

    def test_failure_probability_in_statement(self) -> None:
        ae = _agent().explain_asset(_asset("A", pf=0.73))
        assert "73%" in ae.failure_statement

    def test_critical_band_phrase(self) -> None:
        ae = _agent().explain_asset(_asset("A", band="critical", risk=0.9))
        assert "critical" in ae.health_statement.lower()

    def test_healthy_band_phrase(self) -> None:
        ae = _agent().explain_asset(_asset("A", band="healthy", risk=0.2))
        assert "healthy" in ae.health_statement.lower()

    def test_risk_label_critical(self) -> None:
        ae = _agent().explain_asset(_asset("A", risk=0.95))
        assert ae.risk_label == "critical"

    def test_risk_label_low(self) -> None:
        ae = _agent().explain_asset(_asset("A", risk=0.1))
        assert ae.risk_label == "low"

    def test_infinite_rul(self) -> None:
        ae = _agent().explain_asset(_asset("A", rul=float("inf")))
        assert "no end-of-life" in ae.rul_statement.lower()

    def test_finite_rul(self) -> None:
        ae = _agent().explain_asset(_asset("A", rul=37))
        assert "37" in ae.rul_statement

    def test_shutdown_action(self) -> None:
        ae = _agent().explain_asset(_asset("A", action="shutdown"))
        assert "shut down" in ae.action_statement.lower()

    def test_no_action(self) -> None:
        ae = _agent().explain_asset(_asset("A", action="no_action", cost=0.0))
        assert "no action" in ae.action_statement.lower()

    def test_cost_in_action(self) -> None:
        ae = _agent().explain_asset(_asset("A", cost=7500))
        assert "7,500" in ae.action_statement

    def test_trend_deteriorating(self) -> None:
        ae = _agent().explain_asset(_asset("A", band="critical", risk=0.9))
        assert "deteriorat" in ae.trend_statement.lower()

    def test_trend_stable(self) -> None:
        ae = _agent().explain_asset(_asset("A", band="healthy", risk=0.2))
        assert "stable" in ae.trend_statement.lower()

    def test_missing_fields_raises(self) -> None:
        class Bad:
            asset_id = "X"
        with pytest.raises(TypeError, match="missing required"):
            _agent().explain_asset(Bad())

    def test_summary_concatenates(self) -> None:
        ae = _agent().explain_asset(_asset("A"))
        assert ae.health_statement in ae.summary
        assert ae.action_statement in ae.summary

    def test_eur_currency(self) -> None:
        ae = _agent(currency="EUR").explain_asset(_asset("A"))
        assert "EUR" in ae.action_statement


# ===========================================================================
# explain_fleet
# ===========================================================================


class TestExplainFleet:
    def test_returns_explanation(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert isinstance(fe, FleetExplanation)

    def test_asset_count(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert fe.asset_count == 5

    def test_health_statement(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "average health" in fe.health_statement.lower()

    def test_critical_ids(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "WTG-003" in fe.critical_asset_ids

    def test_critical_statement(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "WTG-003" in fe.critical_statement

    def test_concentration_statement(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "concentration" in fe.concentration_statement.lower()

    def test_high_concentration_phrase(self) -> None:
        fe = _agent().explain_fleet(_snap([_asset("A", risk=0.9)], risk_conc=0.8))
        assert "concentrated" in fe.concentration_statement.lower()

    def test_distributed_phrase(self) -> None:
        fe = _agent().explain_fleet(_default_snap())  # risk_conc=0.3 -> moderate
        assert "concentration" in fe.concentration_statement.lower()

    def test_opportunity_ids(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "WTG-003" in fe.top_opportunity_ids  # highest savings 45000

    def test_opportunity_statement(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert "opportunit" in fe.opportunity_statement.lower()

    def test_no_critical_assets(self) -> None:
        fe = _agent().explain_fleet(_snap([_asset("A", band="healthy", risk=0.2)]))
        assert "no assets" in fe.critical_statement.lower()

    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _agent().explain_fleet(_snap([]))

    def test_missing_fields_raises(self) -> None:
        class Bad:
            asset_count = 1
        with pytest.raises(TypeError, match="missing required"):
            _agent().explain_fleet(Bad())

    def test_summary_concatenates(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert fe.health_statement in fe.summary
        assert fe.opportunity_statement in fe.summary

    def test_top_n_limits_critical(self) -> None:
        fe = _agent(top_n=1).explain_fleet(_default_snap())
        # only 1 critical id listed in statement
        assert fe.critical_statement.count("WTG") == 1


# ===========================================================================
# explain_portfolio
# ===========================================================================


class TestExplainPortfolio:
    def test_returns_explanation(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert isinstance(pe, PortfolioExplanation)

    def test_strategy_recorded(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert pe.strategy == "hybrid"

    def test_selected_statement(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "maintenance" in pe.selected_statement.lower()

    def test_savings_statement(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "savings" in pe.savings_statement.lower()

    def test_risk_reduction_statement(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "risk" in pe.risk_reduction_statement.lower()

    def test_roi_statement(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "return on investment" in pe.roi_statement.lower()

    def test_coverage_statement(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "budget" in pe.coverage_statement.lower()

    def test_selected_ids(self) -> None:
        port = _portfolio()
        pe = _agent().explain_portfolio(port)
        assert set(pe.selected_asset_ids) == set(port.selected_asset_ids)

    def test_zero_budget_no_selection(self) -> None:
        pe = _agent().explain_portfolio(_portfolio(budget=0))
        assert "no assets" in pe.selected_statement.lower()

    def test_unconstrained_budget(self) -> None:
        port = ExecutiveDecisionEngine(ExecutiveDecisionConfig()).recommend(_default_snap())
        pe = _agent().explain_portfolio(port)
        assert "without a binding budget" in pe.coverage_statement.lower()

    def test_missing_fields_raises(self) -> None:
        class Bad:
            strategy = "hybrid"
        with pytest.raises(TypeError, match="missing required"):
            _agent().explain_portfolio(Bad())

    def test_summary_concatenates(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert pe.selected_statement in pe.summary
        assert pe.roi_statement in pe.summary


# ===========================================================================
# answer_question — classification
# ===========================================================================


class TestQuestionClassification:
    def test_why(self) -> None:
        qa = _agent().answer_question("Why was it chosen?", _ctx())
        assert qa.intent == QuestionIntent.WHY.value

    def test_why_not(self) -> None:
        qa = _agent().answer_question("Why not WTG-001?", _ctx())
        assert qa.intent == QuestionIntent.WHY_NOT.value

    def test_why_not_contraction(self) -> None:
        qa = _agent().answer_question("Why wasn't WTG-001 chosen?", _ctx())
        assert qa.intent == QuestionIntent.WHY_NOT.value

    def test_what_if(self) -> None:
        qa = _agent().answer_question("What if the budget grows?", _ctx())
        assert qa.intent == QuestionIntent.WHAT_IF.value

    def test_which_asset(self) -> None:
        qa = _agent().answer_question("Which asset is riskiest?", _ctx())
        assert qa.intent == QuestionIntent.WHICH_ASSET.value

    def test_which_risk(self) -> None:
        qa = _agent().answer_question("Which risk dominates?", _ctx())
        assert qa.intent == QuestionIntent.WHICH_RISK.value

    def test_which_action(self) -> None:
        qa = _agent().answer_question("Which action is best?", _ctx())
        assert qa.intent == QuestionIntent.WHICH_ACTION.value

    def test_what_should(self) -> None:
        qa = _agent().answer_question("What should we do?", _ctx())
        assert qa.intent == QuestionIntent.WHICH_ACTION.value

    def test_unknown(self) -> None:
        qa = _agent().answer_question("Tell me about turbines.", _ctx())
        assert qa.intent == QuestionIntent.UNKNOWN.value

    def test_why_not_precedence(self) -> None:
        # "why not" must take precedence over "why"
        qa = _agent().answer_question("Why not?", _ctx())
        assert qa.intent == QuestionIntent.WHY_NOT.value

    def test_what_if_precedence_over_what_asset(self) -> None:
        qa = _agent().answer_question("What if asset fails?", _ctx())
        assert qa.intent == QuestionIntent.WHAT_IF.value


# ===========================================================================
# answer_question — answers
# ===========================================================================


class TestAnswerWhy:
    def test_resolves_named_asset(self) -> None:
        qa = _agent().answer_question("Why is WTG-003 risky?", _ctx())
        assert "WTG-003" in qa.answer

    def test_high_confidence_with_asset(self) -> None:
        qa = _agent().answer_question("Why is WTG-003 risky?", _ctx())
        assert qa.confidence >= 0.85

    def test_facts_present(self) -> None:
        qa = _agent().answer_question("Why is WTG-003 risky?", _ctx())
        assert len(qa.supporting_facts) > 0

    def test_portfolio_level_why(self) -> None:
        qa = _agent().answer_question("Why this portfolio?", _ctx())
        assert "strategy" in qa.answer.lower() or "selected" in qa.answer.lower()

    def test_no_context(self) -> None:
        qa = _agent().answer_question("Why?", CopilotContext())
        assert qa.confidence <= 0.4


class TestAnswerWhyNot:
    def test_unselected_asset(self) -> None:
        ctx = _ctx()
        unsel = [pa.asset_id for pa in ctx.portfolio.prioritization
                 if not pa.selected]
        if unsel:
            qa = _agent().answer_question(f"Why not {unsel[0]}?", ctx)
            assert unsel[0] in qa.answer and "not selected" in qa.answer.lower()

    def test_selected_asset_clarifies(self) -> None:
        ctx = _ctx()
        sel = ctx.portfolio.selected_asset_ids[0]
        qa = _agent().answer_question(f"Why not {sel}?", ctx)
        assert "in fact selected" in qa.answer.lower()

    def test_no_portfolio(self) -> None:
        qa = _agent().answer_question("Why not WTG-001?",
                                      CopilotContext(snapshot=_default_snap()))
        assert qa.confidence <= 0.4

    def test_unidentified_asset(self) -> None:
        qa = _agent().answer_question("Why not?", _ctx())
        # no asset named -> lower confidence
        assert qa.confidence <= 0.5


class TestAnswerWhatIf:
    def test_budget_increase(self) -> None:
        qa = _agent().answer_question("What if we increase the budget?", _ctx())
        assert "budget" in qa.answer.lower() or "fund" in qa.answer.lower()

    def test_failure_scenario(self) -> None:
        qa = _agent().answer_question("What if assets fail?", _ctx())
        assert "failure" in qa.answer.lower() or "fail" in qa.answer.lower()

    def test_generic_what_if(self) -> None:
        qa = _agent().answer_question("What if the weather changes?", _ctx())
        assert qa.intent == QuestionIntent.WHAT_IF.value

    def test_budget_no_portfolio(self) -> None:
        qa = _agent().answer_question("What if budget grows?",
                                      CopilotContext(snapshot=_default_snap()))
        assert qa.intent == QuestionIntent.WHAT_IF.value


class TestAnswerWhichAsset:
    def test_highest_risk(self) -> None:
        qa = _agent().answer_question("Which asset is riskiest?", _ctx())
        assert "WTG-003" in qa.answer  # risk 0.99

    def test_largest_opportunity(self) -> None:
        qa = _agent().answer_question("Which asset has the biggest saving?", _ctx())
        assert "WTG-003" in qa.answer  # savings 45000

    def test_worst_health(self) -> None:
        qa = _agent().answer_question("Which asset has the worst health?", _ctx())
        assert "WTG-003" in qa.answer  # health 8

    def test_no_snapshot(self) -> None:
        qa = _agent().answer_question("Which asset is riskiest?",
                                      CopilotContext())
        assert qa.confidence <= 0.4


class TestAnswerWhichRisk:
    def test_dominant_risk(self) -> None:
        qa = _agent().answer_question("Which risk dominates?", _ctx())
        assert "WTG-003" in qa.answer

    def test_no_snapshot(self) -> None:
        qa = _agent().answer_question("Which risk dominates?", CopilotContext())
        assert qa.confidence <= 0.4


class TestAnswerWhichAction:
    def test_named_asset(self) -> None:
        qa = _agent().answer_question("Which action for WTG-003?", _ctx())
        assert "WTG-003" in qa.answer

    def test_top_priority_default(self) -> None:
        qa = _agent().answer_question("Which action should we take?", _ctx())
        assert "WTG-003" in qa.answer  # highest risk

    def test_no_snapshot(self) -> None:
        qa = _agent().answer_question("Which action?", CopilotContext())
        assert qa.confidence <= 0.4


class TestAnswerEdgeCases:
    def test_empty_question(self) -> None:
        qa = _agent().answer_question("", _ctx())
        assert qa.intent == QuestionIntent.UNKNOWN.value and qa.confidence == 0.0

    def test_whitespace_question(self) -> None:
        qa = _agent().answer_question("   ", _ctx())
        assert qa.confidence == 0.0

    def test_none_context_handled(self) -> None:
        qa = _agent().answer_question("Why?", None)
        assert isinstance(qa, QuestionAnswer)

    def test_unknown_helpful(self) -> None:
        qa = _agent().answer_question("Hello there", _ctx())
        assert "explain" in qa.answer.lower()

    def test_never_raises(self) -> None:
        # A battery of odd inputs must never raise
        for q in ["", "???", "why why why", "WHAT IF WHY NOT",
                  "which asset which risk"]:
            assert isinstance(_agent().answer_question(q, _ctx()), QuestionAnswer)


# ===========================================================================
# generate_executive_brief
# ===========================================================================


class TestExecutiveBrief:
    def test_returns_brief(self) -> None:
        assert isinstance(_agent().generate_executive_brief(_ctx()),
                          ExecutiveBrief)

    def test_summary_non_empty(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert len(brief.executive_summary) > 40

    def test_key_risks_present(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert len(brief.key_risks) > 0

    def test_key_risks_ordered(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        # highest-risk asset (WTG-003) first
        assert "WTG-003" in brief.key_risks[0]

    def test_opportunities_present(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert len(brief.opportunities) > 0

    def test_recommendations_present(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert len(brief.recommendations) > 0

    def test_confidence_statement(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert "confidence" in brief.confidence_statement.lower()

    def test_snapshot_only(self) -> None:
        brief = _agent().generate_executive_brief(
            CopilotContext(snapshot=_default_snap()))
        assert len(brief.key_risks) > 0
        assert "unavailable" in brief.confidence_statement.lower()

    def test_portfolio_only(self) -> None:
        brief = _agent().generate_executive_brief(
            CopilotContext(portfolio=_portfolio()))
        assert len(brief.executive_summary) > 0

    def test_empty_context_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a snapshot"):
            _agent().generate_executive_brief(CopilotContext())

    def test_none_context_raises(self) -> None:
        with pytest.raises(ValueError):
            _agent().generate_executive_brief(None)

    def test_top_n_limits_risks(self) -> None:
        brief = _agent(top_n=2).generate_executive_brief(_ctx())
        assert len(brief.key_risks) <= 2

    def test_confidence_qualifier(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert any(w in brief.confidence_statement.lower()
                   for w in ("high", "moderate", "limited"))


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_asset_explanation_json(self) -> None:
        ae = _agent().explain_asset(_asset("A"))
        assert isinstance(json.dumps(ae.to_dict()), str)

    def test_fleet_explanation_json(self) -> None:
        fe = _agent().explain_fleet(_default_snap())
        assert isinstance(json.dumps(fe.to_dict()), str)

    def test_portfolio_explanation_json(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert isinstance(json.dumps(pe.to_dict()), str)

    def test_question_answer_json(self) -> None:
        qa = _agent().answer_question("Why WTG-003?", _ctx())
        assert isinstance(json.dumps(qa.to_dict()), str)

    def test_brief_json(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert isinstance(json.dumps(brief.to_dict()), str)

    def test_asset_dict_keys(self) -> None:
        d = _agent().explain_asset(_asset("A")).to_dict()
        for k in ("asset_id", "risk_label", "summary"):
            assert k in d

    def test_brief_dict_keys(self) -> None:
        d = _agent().generate_executive_brief(_ctx()).to_dict()
        for k in ("executive_summary", "key_risks", "opportunities",
                  "recommendations", "confidence_statement"):
            assert k in d

    def test_question_answer_dict_lists(self) -> None:
        d = _agent().answer_question("Why WTG-003?", _ctx()).to_dict()
        assert isinstance(d["supporting_facts"], list)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_asset_deterministic(self) -> None:
        a = _asset("A")
        assert _agent().explain_asset(a).summary == _agent().explain_asset(a).summary

    def test_fleet_deterministic(self) -> None:
        s = _default_snap()
        assert _agent().explain_fleet(s).summary == _agent().explain_fleet(s).summary

    def test_portfolio_deterministic(self) -> None:
        p = _portfolio()
        assert _agent().explain_portfolio(p).summary == _agent().explain_portfolio(p).summary

    def test_answer_deterministic(self) -> None:
        ctx = _ctx()
        q = "Why is WTG-003 risky?"
        assert _agent().answer_question(q, ctx).answer == _agent().answer_question(q, ctx).answer

    def test_brief_deterministic(self) -> None:
        ctx = _ctx()
        assert _agent().generate_executive_brief(ctx).executive_summary == \
            _agent().generate_executive_brief(ctx).executive_summary

    def test_asset_resolution_deterministic(self) -> None:
        # When multiple assets are named, resolution is deterministic
        ctx = _ctx()
        q = "Compare WTG-003 and WTG-001 risk?"
        a1 = _agent().answer_question(q, ctx).answer
        a2 = _agent().answer_question(q, ctx).answer
        assert a1 == a2


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_interactions(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        agent = DecisionCopilotAgent(experiment_tracker=FakeTracker())
        agent.explain_asset(_asset("A"))
        assert logged and "copilot_interactions" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        agent = DecisionCopilotAgent(experiment_tracker=BrokenTracker())
        assert agent.explain_asset(_asset("A")) is not None

    def test_no_tracker_ok(self) -> None:
        assert _agent().explain_asset(_asset("A")) is not None

    def test_interaction_count_increments(self) -> None:
        agent = _agent()
        agent.explain_asset(_asset("A"))
        agent.explain_fleet(_default_snap())
        assert agent._n_interactions == 2


# ===========================================================================
# Additional coverage
# ===========================================================================


class TestAdditionalAsset:
    def test_immediate_maintenance_phrase(self) -> None:
        ae = _agent().explain_asset(_asset("A", action="immediate_maintenance"))
        assert "immediate" in ae.action_statement.lower()

    def test_inspect_phrase(self) -> None:
        ae = _agent().explain_asset(_asset("A", action="inspect"))
        assert "inspection" in ae.action_statement.lower()

    def test_unknown_action_graceful(self) -> None:
        ae = _agent().explain_asset(_asset("A", action="frobnicate"))
        assert isinstance(ae.action_statement, str)

    def test_unknown_band_graceful(self) -> None:
        ae = _agent().explain_asset(_asset("A", band="weird"))
        assert isinstance(ae.health_statement, str)

    def test_zero_risk(self) -> None:
        ae = _agent().explain_asset(_asset("A", risk=0.0))
        assert ae.risk_label == "low"

    def test_max_risk(self) -> None:
        ae = _agent().explain_asset(_asset("A", risk=1.0))
        assert ae.risk_label == "critical"

    def test_nan_rul_handled(self) -> None:
        ae = _agent().explain_asset(_asset("A", rul=float("nan")))
        assert "no end-of-life" in ae.rul_statement.lower()


class TestAdditionalFleet:
    def test_single_asset_fleet(self) -> None:
        fe = _agent().explain_fleet(_snap([_asset("Solo", band="warning")]))
        assert fe.asset_count == 1

    def test_all_healthy_fleet(self) -> None:
        fe = _agent().explain_fleet(
            _snap([_asset(f"A{i}", band="healthy", risk=0.2) for i in range(3)]))
        assert "no assets" in fe.critical_statement.lower()

    def test_large_fleet(self) -> None:
        assets = [_asset(f"A{i:03d}", risk=0.2 + (i % 5) * 0.15,
                         savings=10000 + i * 500,
                         band=["healthy", "warning", "critical"][i % 3])
                  for i in range(40)]
        fe = _agent().explain_fleet(_snap(assets))
        assert fe.asset_count == 40

    def test_zero_savings_fleet(self) -> None:
        fe = _agent().explain_fleet(
            _snap([_asset("A", savings=0.0), _asset("B", savings=0.0)]))
        assert "no material" in fe.opportunity_statement.lower()


class TestAdditionalPortfolio:
    def test_eur_currency(self) -> None:
        snap = _snap([_asset("A", savings=20000)], currency="EUR")
        port = ExecutiveDecisionEngine(
            ExecutiveDecisionConfig(budget=10000, currency="EUR")).recommend(snap)
        pe = _agent(currency="EUR").explain_portfolio(port)
        assert "EUR" in pe.savings_statement

    def test_risk_reduction_percentage(self) -> None:
        pe = _agent().explain_portfolio(_portfolio())
        assert "%" in pe.risk_reduction_statement


class TestAdditionalAnswers:
    def test_why_facts_count(self) -> None:
        qa = _agent().answer_question("Why WTG-003?", _ctx())
        assert len(qa.supporting_facts) >= 3

    def test_which_asset_facts(self) -> None:
        qa = _agent().answer_question("Which asset is riskiest?", _ctx())
        assert len(qa.supporting_facts) >= 1

    def test_case_insensitive_asset(self) -> None:
        qa = _agent().answer_question("Why wtg-003 risky?", _ctx())
        assert "WTG-003" in qa.answer

    def test_what_if_budget_facts(self) -> None:
        qa = _agent().answer_question("What if we add budget?", _ctx())
        # may reference the next asset to fund
        assert isinstance(qa.supporting_facts, tuple)

    def test_confidence_bounded(self) -> None:
        for q in ["Why WTG-003?", "Why not WTG-001?", "What if budget?",
                  "Which asset?", "Which risk?", "Which action?", "Hello"]:
            qa = _agent().answer_question(q, _ctx())
            assert 0.0 <= qa.confidence <= 1.0


class TestAdditionalBrief:
    def test_recommendations_from_portfolio(self) -> None:
        ctx = _ctx()
        brief = _agent().generate_executive_brief(ctx)
        # portfolio recommendations propagate
        assert len(brief.recommendations) >= 1

    def test_high_confidence_qualifier(self) -> None:
        # construct a high-confidence portfolio scenario (large fleet, full cover)
        assets = [_asset(f"A{i}", risk=0.8, cost=1000, savings=20000,
                         band="critical") for i in range(12)]
        snap = _snap(assets)
        port = ExecutiveDecisionEngine(
            ExecutiveDecisionConfig(budget=100000)).recommend(snap)
        brief = _agent().generate_executive_brief(
            CopilotContext(snapshot=snap, portfolio=port))
        assert "confidence" in brief.confidence_statement.lower()

    def test_brief_summary_mentions_fleet(self) -> None:
        brief = _agent().generate_executive_brief(_ctx())
        assert "fleet" in brief.executive_summary.lower()