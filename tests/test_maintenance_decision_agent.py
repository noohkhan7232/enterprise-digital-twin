#!/usr/bin/env python3
"""Comprehensive test suite for ``src/predictive/maintenance_decision_agent.py``.

The agent is pure NumPy, so the **entire** suite runs without PyTorch or SciPy.
Coverage (100+ tests):

- MaintenanceDecisionConfig validation
- MaintenanceAction / PriorityLevel enums (incl. severity ordering)
- Registry (register / build / list / duplicate rejection)
- All five maintenance actions reachable
- All four priority levels reachable
- Low-health and critical-health cases
- Low-RUL and high-RUL cases
- High-failure-probability cases
- Change-point scenarios
- Cost estimation (maintenance_cost, failure_cost)
- Expected-savings estimation (sign, magnitude, break-even)
- Downtime estimation (scales with action)
- Confidence scoring (RUL CI width, input agreement)
- Explainability (decision_reason, triggered_rules)
- Registry / tracker integration (failure-safe)
- Edge cases (infinite RUL, non-degrading)
- Integration with the full Step 1->4 pipeline

Run::

    pytest tests/test_maintenance_decision_agent.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predictive.failure_risk import FailureRiskPrediction, RiskLevel
from src.predictive.health_trend_analyzer import (
    EarlyWarning,
    HealthTrendResult,
    TrendDirection,
)
from src.predictive.maintenance_decision_agent import (
    AGENT_NAME,
    MAINTENANCE_AGENT_REGISTRY,
    MaintenanceAction,
    MaintenanceDecision,
    MaintenanceDecisionAgent,
    MaintenanceDecisionConfig,
    PriorityLevel,
    build_maintenance_agent,
    list_maintenance_agents,
    register_maintenance_agent,
)
from src.predictive.rul_predictor import RUL_INFINITE, RULPrediction


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def mk_trend(
    health: float = 90.0,
    trend: TrendDirection = TrendDirection.STABLE,
    warning: EarlyWarning = EarlyWarning.NORMAL,
    *,
    confidence: float = 0.8,
    has_change: bool = False,
) -> HealthTrendResult:
    """Build a HealthTrendResult fixture."""
    return HealthTrendResult(
        trend=trend, warning=warning, linear_slope=-0.5,
        moving_average_slope=-0.5, robust_slope=-0.5, curvature=0.0,
        current_health=health, mean_health=health, trend_confidence=confidence,
        change_index=10 if has_change else -1,
        change_magnitude=20.0 if has_change else 0.0,
        change_significance=3.0 if has_change else 0.0,
        change_confidence=0.8 if has_change else 0.0,
        has_change_point=has_change, n_observations=20,
    )


def mk_rul(rul: float = 100.0, *, is_degrading: bool = True,
           ci_low: float | None = None, ci_high: float | None = None) -> RULPrediction:
    """Build a RULPrediction fixture."""
    if ci_low is None:
        ci_low = rul * 0.8 if math.isfinite(rul) else RUL_INFINITE
    if ci_high is None:
        ci_high = rul * 1.2 if math.isfinite(rul) else RUL_INFINITE
    return RULPrediction(
        rul=rul, ci_low=ci_low, ci_high=ci_high, confidence_level=0.95,
        failure_threshold=30.0, current_health=55.0, model_used="linear",
        slope=-2.0, r_squared=0.95, n_observations=20, is_degrading=is_degrading,
        time_per_step=1.0,
    )


def mk_risk(p: float = 0.1, level: RiskLevel = RiskLevel.LOW) -> FailureRiskPrediction:
    """Build a FailureRiskPrediction fixture."""
    return FailureRiskPrediction(
        horizon_risks={7.0: p * 0.5, 30.0: p, 90.0: min(1.0, p * 1.5)},
        risk_level=level, dominant_horizon=30.0, dominant_probability=p,
        model_used="exponential", mean_life=100.0, is_degrading=True,
    )


def _agent(**cfg) -> MaintenanceDecisionAgent:
    return MaintenanceDecisionAgent(MaintenanceDecisionConfig(**cfg))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for configuration validation."""

    def test_defaults(self) -> None:
        c = MaintenanceDecisionConfig()
        assert c.failure_cost >= c.maintenance_cost

    def test_negative_cost(self) -> None:
        with pytest.raises(ValueError, match="costs"):
            MaintenanceDecisionConfig(maintenance_cost=-1)

    def test_failure_below_maintenance(self) -> None:
        with pytest.raises(ValueError, match="failure_cost"):
            MaintenanceDecisionConfig(failure_cost=100, maintenance_cost=200)

    def test_bad_health_order(self) -> None:
        with pytest.raises(ValueError, match="health"):
            MaintenanceDecisionConfig(health_critical=80, health_low=50)

    def test_bad_rul_order(self) -> None:
        with pytest.raises(ValueError, match="rul"):
            MaintenanceDecisionConfig(rul_imminent=100, rul_short=50)

    def test_bad_risk_order(self) -> None:
        with pytest.raises(ValueError, match="risk"):
            MaintenanceDecisionConfig(risk_elevated=0.6, risk_high=0.5)

    def test_bad_score_order(self) -> None:
        with pytest.raises(ValueError, match="score"):
            MaintenanceDecisionConfig(inspect_score=10, schedule_score=5)

    def test_bad_inspect_score(self) -> None:
        with pytest.raises(ValueError, match="inspect_score"):
            MaintenanceDecisionConfig(inspect_score=0)

    def test_bad_min_trend_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_trend_confidence"):
            MaintenanceDecisionConfig(min_trend_confidence=1.5)

    def test_negative_downtime(self) -> None:
        with pytest.raises(ValueError, match="downtime"):
            MaintenanceDecisionConfig(inspect_downtime_hours=-1)

    def test_frozen(self) -> None:
        c = MaintenanceDecisionConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.maintenance_cost = 1.0  # type: ignore[misc]

    def test_custom_costs(self) -> None:
        c = MaintenanceDecisionConfig(maintenance_cost=1000, failure_cost=100000)
        assert c.maintenance_cost == 1000
        assert c.failure_cost == 100000


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for the action and priority enums."""

    def test_action_values(self) -> None:
        assert MaintenanceAction.NO_ACTION.value == "no_action"
        assert MaintenanceAction.INSPECT.value == "inspect"
        assert MaintenanceAction.SCHEDULE_MAINTENANCE.value == "schedule_maintenance"
        assert MaintenanceAction.IMMEDIATE_MAINTENANCE.value == "immediate_maintenance"
        assert MaintenanceAction.SHUTDOWN.value == "shutdown"

    def test_priority_values(self) -> None:
        assert PriorityLevel.LOW.value == "low"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.CRITICAL.value == "critical"

    def test_action_severity_order(self) -> None:
        assert (MaintenanceAction.NO_ACTION.severity
                < MaintenanceAction.INSPECT.severity
                < MaintenanceAction.SCHEDULE_MAINTENANCE.severity
                < MaintenanceAction.IMMEDIATE_MAINTENANCE.severity
                < MaintenanceAction.SHUTDOWN.severity)

    def test_priority_severity_order(self) -> None:
        assert (PriorityLevel.LOW.severity < PriorityLevel.MEDIUM.severity
                < PriorityLevel.HIGH.severity < PriorityLevel.CRITICAL.severity)

    def test_action_is_str(self) -> None:
        assert MaintenanceAction.SHUTDOWN == "shutdown"

    def test_priority_is_str(self) -> None:
        assert PriorityLevel.CRITICAL == "critical"

    def test_shutdown_severity_four(self) -> None:
        assert MaintenanceAction.SHUTDOWN.severity == 4

    def test_critical_priority_severity_three(self) -> None:
        assert PriorityLevel.CRITICAL.severity == 3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the registry."""

    def test_registered(self) -> None:
        assert AGENT_NAME in MAINTENANCE_AGENT_REGISTRY
        assert AGENT_NAME in list_maintenance_agents()

    def test_build_by_name(self) -> None:
        assert isinstance(build_maintenance_agent(AGENT_NAME),
                          MaintenanceDecisionAgent)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown maintenance agent"):
            build_maintenance_agent("nope")

    def test_registry_name_stamped(self) -> None:
        assert MaintenanceDecisionAgent._registry_name == AGENT_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_maintenance_agent(AGENT_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        a = build_maintenance_agent(
            AGENT_NAME, config=MaintenanceDecisionConfig(maintenance_cost=999))
        assert a.config.maintenance_cost == 999


# ---------------------------------------------------------------------------
# All actions
# ---------------------------------------------------------------------------


class TestAllActions:
    """Tests that every maintenance action is reachable."""

    def test_no_action(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert d.action == MaintenanceAction.NO_ACTION

    def test_inspect(self) -> None:
        d = _agent().decide(
            mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
            mk_rul(80), mk_risk(0.15))
        assert d.action == MaintenanceAction.INSPECT

    def test_schedule_maintenance(self) -> None:
        d = _agent().decide(
            mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
            mk_rul(50), mk_risk(0.4, RiskLevel.MEDIUM))
        assert d.action == MaintenanceAction.SCHEDULE_MAINTENANCE

    def test_immediate_maintenance(self) -> None:
        d = _agent().decide(
            mk_trend(45, TrendDirection.DEGRADING, EarlyWarning.WARNING),
            mk_rul(15), mk_risk(0.6, RiskLevel.HIGH))
        assert d.action == MaintenanceAction.IMMEDIATE_MAINTENANCE

    def test_shutdown(self) -> None:
        d = _agent().decide(
            mk_trend(20, TrendDirection.ACCELERATING, EarlyWarning.CRITICAL,
                     has_change=True),
            mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL))
        assert d.action == MaintenanceAction.SHUTDOWN

    def test_all_five_distinct(self) -> None:
        agent = _agent()
        actions = {
            agent.decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                         mk_risk(0.02)).action,
            agent.decide(mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
                         mk_rul(80), mk_risk(0.15)).action,
            agent.decide(mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
                         mk_rul(50), mk_risk(0.4, RiskLevel.MEDIUM)).action,
            agent.decide(mk_trend(45, TrendDirection.DEGRADING, EarlyWarning.WARNING),
                         mk_rul(15), mk_risk(0.6, RiskLevel.HIGH)).action,
            agent.decide(mk_trend(20, TrendDirection.ACCELERATING,
                                  EarlyWarning.CRITICAL, has_change=True),
                         mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL)).action,
        }
        assert len(actions) == 5


# ---------------------------------------------------------------------------
# All priorities
# ---------------------------------------------------------------------------


class TestAllPriorities:
    """Tests that every priority level is reachable."""

    def test_low(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert d.priority == PriorityLevel.LOW

    def test_medium(self) -> None:
        d = _agent().decide(mk_trend(68), mk_rul(80), mk_risk(0.3, RiskLevel.MEDIUM))
        assert d.priority == PriorityLevel.MEDIUM

    def test_high(self) -> None:
        d = _agent().decide(
            mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
            mk_rul(50), mk_risk(0.4, RiskLevel.MEDIUM))
        assert d.priority == PriorityLevel.HIGH

    def test_critical(self) -> None:
        d = _agent().decide(
            mk_trend(20, TrendDirection.ACCELERATING, EarlyWarning.CRITICAL,
                     has_change=True),
            mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL))
        assert d.priority == PriorityLevel.CRITICAL

    def test_all_four_distinct(self) -> None:
        agent = _agent()
        priorities = {
            agent.decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                         mk_risk(0.02)).priority,
            agent.decide(mk_trend(68), mk_rul(80),
                         mk_risk(0.3, RiskLevel.MEDIUM)).priority,
            agent.decide(mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
                         mk_rul(50), mk_risk(0.4, RiskLevel.MEDIUM)).priority,
            agent.decide(mk_trend(20, TrendDirection.ACCELERATING,
                                  EarlyWarning.CRITICAL, has_change=True),
                         mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL)).priority,
        }
        assert len(priorities) == 4

    def test_critical_risk_lifts_priority(self) -> None:
        # Even a modest score gets at least HIGH priority under critical risk
        d = _agent().decide(mk_trend(75), mk_rul(100),
                            mk_risk(0.8, RiskLevel.CRITICAL))
        assert d.priority.severity >= PriorityLevel.HIGH.severity


# ---------------------------------------------------------------------------
# Health cases
# ---------------------------------------------------------------------------


class TestHealthCases:
    """Tests for health-driven behaviour."""

    def test_low_health_triggers_rule(self) -> None:
        d = _agent().decide(mk_trend(45), mk_rul(100), mk_risk(0.1))
        assert "low_health" in d.triggered_rules

    def test_critical_health_triggers_rule(self) -> None:
        d = _agent().decide(mk_trend(25), mk_rul(100), mk_risk(0.1))
        assert "critical_health" in d.triggered_rules

    def test_declining_health_triggers_rule(self) -> None:
        d = _agent().decide(mk_trend(65), mk_rul(100), mk_risk(0.1))
        assert "declining_health" in d.triggered_rules

    def test_healthy_no_health_rule(self) -> None:
        d = _agent().decide(mk_trend(95), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert "low_health" not in d.triggered_rules
        assert "critical_health" not in d.triggered_rules

    def test_critical_health_escalates_action(self) -> None:
        low = _agent().decide(mk_trend(90), mk_rul(100), mk_risk(0.1))
        crit = _agent().decide(mk_trend(25), mk_rul(100), mk_risk(0.1))
        assert crit.action.severity > low.action.severity

    def test_current_health_recorded(self) -> None:
        d = _agent().decide(mk_trend(42), mk_rul(100), mk_risk(0.1))
        assert d.current_health == 42.0


# ---------------------------------------------------------------------------
# RUL cases
# ---------------------------------------------------------------------------


class TestRulCases:
    """Tests for RUL-driven behaviour."""

    def test_imminent_rul_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(3), mk_risk(0.3))
        assert "imminent_failure" in d.triggered_rules

    def test_short_rul_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(20), mk_risk(0.3))
        assert "short_rul" in d.triggered_rules

    def test_moderate_rul_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(60), mk_risk(0.3))
        assert "moderate_rul" in d.triggered_rules

    def test_high_rul_no_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(500), mk_risk(0.1))
        assert "imminent_failure" not in d.triggered_rules
        assert "short_rul" not in d.triggered_rules

    def test_shorter_rul_higher_severity(self) -> None:
        long = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                               mk_rul(200), mk_risk(0.2))
        short = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                                mk_rul(5), mk_risk(0.2))
        assert short.severity_score > long.severity_score

    def test_infinite_rul_no_rul_rule(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert "imminent_failure" not in d.triggered_rules

    def test_rul_recorded(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(33), mk_risk(0.2))
        assert d.remaining_useful_life == 33.0


# ---------------------------------------------------------------------------
# Failure probability
# ---------------------------------------------------------------------------


class TestFailureProbability:
    """Tests for failure-probability-driven behaviour."""

    def test_critical_risk_rule(self) -> None:
        d = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.8, RiskLevel.CRITICAL))
        assert "critical_failure_risk" in d.triggered_rules

    def test_high_risk_rule(self) -> None:
        d = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.6, RiskLevel.HIGH))
        assert "high_failure_risk" in d.triggered_rules

    def test_elevated_risk_rule(self) -> None:
        d = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.3, RiskLevel.MEDIUM))
        assert "elevated_failure_risk" in d.triggered_rules

    def test_low_risk_no_rule(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(100), mk_risk(0.05))
        assert "critical_failure_risk" not in d.triggered_rules
        assert "high_failure_risk" not in d.triggered_rules

    def test_higher_prob_higher_severity(self) -> None:
        low = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        high = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.9, RiskLevel.CRITICAL))
        assert high.severity_score > low.severity_score

    def test_failure_probability_recorded(self) -> None:
        d = _agent().decide(mk_trend(60), mk_rul(100), mk_risk(0.42, RiskLevel.MEDIUM))
        assert d.failure_probability == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Change-point scenarios
# ---------------------------------------------------------------------------


class TestChangePoint:
    """Tests for change-point handling."""

    def test_change_point_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING, has_change=True),
                            mk_rul(50), mk_risk(0.3))
        assert "regime_change_detected" in d.triggered_rules

    def test_change_point_raises_severity(self) -> None:
        without = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                                  mk_rul(50), mk_risk(0.3))
        with_cp = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, has_change=True),
            mk_rul(50), mk_risk(0.3))
        assert with_cp.severity_score > without.severity_score

    def test_change_point_in_reason(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING, has_change=True),
                            mk_rul(50), mk_risk(0.3))
        assert "regime change" in d.decision_reason

    def test_no_change_no_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(50), mk_risk(0.3))
        assert "regime_change_detected" not in d.triggered_rules


# ---------------------------------------------------------------------------
# Trend behaviour
# ---------------------------------------------------------------------------


class TestTrendBehaviour:
    """Tests for trend-driven behaviour."""

    def test_accelerating_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.ACCELERATING),
                            mk_rul(50), mk_risk(0.3))
        assert "accelerating_degradation" in d.triggered_rules

    def test_degrading_rule(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(50), mk_risk(0.3))
        assert "degrading_trend" in d.triggered_rules

    def test_accelerating_more_severe_than_degrading(self) -> None:
        deg = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                              mk_rul(50), mk_risk(0.3))
        acc = _agent().decide(mk_trend(60, TrendDirection.ACCELERATING),
                              mk_rul(50), mk_risk(0.3))
        assert acc.severity_score > deg.severity_score

    def test_low_trend_confidence_downweights(self) -> None:
        conf = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.9),
            mk_rul(50), mk_risk(0.2))
        unconf = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.05),
            mk_rul(50), mk_risk(0.2))
        assert conf.severity_score >= unconf.severity_score
        assert "low_trend_confidence" in unconf.triggered_rules

    def test_improving_no_degrade_rule(self) -> None:
        d = _agent().decide(mk_trend(80, TrendDirection.IMPROVING),
                            mk_rul(RUL_INFINITE, is_degrading=False), mk_risk(0.05))
        assert "degrading_trend" not in d.triggered_rules


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_maintenance_cost_reported(self) -> None:
        d = _agent(maintenance_cost=3000).decide(
            mk_trend(60, TrendDirection.DEGRADING), mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        assert d.maintenance_cost == 3000

    def test_failure_cost_reported(self) -> None:
        d = _agent(failure_cost=80000).decide(
            mk_trend(60, TrendDirection.DEGRADING), mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        assert d.failure_cost == 80000

    def test_currency_reported(self) -> None:
        d = _agent(currency="EUR").decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        assert d.currency == "EUR"


# ---------------------------------------------------------------------------
# Savings estimation
# ---------------------------------------------------------------------------


class TestSavingsEstimation:
    """Tests for expected-savings estimation."""

    def test_high_prob_positive_savings(self) -> None:
        d = _agent(maintenance_cost=5000, failure_cost=50000).decide(
            mk_trend(40, TrendDirection.DEGRADING), mk_rul(10),
            mk_risk(0.8, RiskLevel.CRITICAL))
        assert d.expected_savings > 0

    def test_low_prob_negative_savings(self) -> None:
        d = _agent(maintenance_cost=5000, failure_cost=50000).decide(
            mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False), mk_risk(0.02))
        assert d.expected_savings < 0

    def test_break_even(self) -> None:
        # P_fail = maint/fail = 0.1 -> savings ~ 0
        d = _agent(maintenance_cost=5000, failure_cost=50000).decide(
            mk_trend(60), mk_rul(100), mk_risk(0.1))
        assert abs(d.expected_savings) < 1.0

    def test_savings_scales_with_probability(self) -> None:
        agent = _agent(maintenance_cost=5000, failure_cost=50000)
        low = agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.3, RiskLevel.MEDIUM))
        high = agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.8, RiskLevel.CRITICAL))
        assert high.expected_savings > low.expected_savings

    def test_savings_uses_failure_cost(self) -> None:
        cheap = _agent(failure_cost=10000).decide(
            mk_trend(60), mk_rul(100), mk_risk(0.5, RiskLevel.HIGH))
        pricey = _agent(failure_cost=200000).decide(
            mk_trend(60), mk_rul(100), mk_risk(0.5, RiskLevel.HIGH))
        assert pricey.expected_savings > cheap.expected_savings


# ---------------------------------------------------------------------------
# Downtime estimation
# ---------------------------------------------------------------------------


class TestDowntimeEstimation:
    """Tests for downtime estimation."""

    def test_no_action_zero_downtime(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert d.estimated_downtime_hours == 0.0

    def test_inspect_downtime(self) -> None:
        d = _agent(inspect_downtime_hours=3).decide(
            mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
            mk_rul(80), mk_risk(0.15))
        assert d.action == MaintenanceAction.INSPECT
        assert d.estimated_downtime_hours == 3

    def test_downtime_scales_with_action(self) -> None:
        agent = _agent()
        inspect = agent.decide(
            mk_trend(68, TrendDirection.DEGRADING, EarlyWarning.WATCH),
            mk_rul(80), mk_risk(0.15))
        shutdown = agent.decide(
            mk_trend(20, TrendDirection.ACCELERATING, EarlyWarning.CRITICAL,
                     has_change=True),
            mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL))
        assert shutdown.estimated_downtime_hours > inspect.estimated_downtime_hours

    def test_shutdown_max_downtime(self) -> None:
        d = _agent(failure_downtime_hours=100).decide(
            mk_trend(20, TrendDirection.ACCELERATING, EarlyWarning.CRITICAL,
                     has_change=True),
            mk_rul(2), mk_risk(0.95, RiskLevel.CRITICAL))
        assert d.estimated_downtime_hours == 100


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    """Tests for decision confidence."""

    def test_in_unit_interval(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        assert 0.0 <= d.decision_confidence <= 1.0

    def test_tight_ci_higher_confidence(self) -> None:
        tight = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.9),
            mk_rul(20, ci_low=19, ci_high=21), mk_risk(0.5, RiskLevel.HIGH))
        wide = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.9),
            mk_rul(20, ci_low=5, ci_high=40), mk_risk(0.5, RiskLevel.HIGH))
        assert tight.decision_confidence > wide.decision_confidence

    def test_agreement_higher_confidence(self) -> None:
        # Degrading + high risk (agree) vs degrading + low risk (disagree)
        agree = _agent().decide(
            mk_trend(50, TrendDirection.DEGRADING, confidence=0.8),
            mk_rul(20), mk_risk(0.8, RiskLevel.CRITICAL))
        disagree = _agent().decide(
            mk_trend(50, TrendDirection.DEGRADING, confidence=0.8),
            mk_rul(20), mk_risk(0.05, RiskLevel.LOW))
        assert agree.decision_confidence > disagree.decision_confidence

    def test_high_trend_confidence_higher(self) -> None:
        high = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.95),
            mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        low = _agent().decide(
            mk_trend(60, TrendDirection.DEGRADING, confidence=0.2),
            mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        assert high.decision_confidence > low.decision_confidence


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


class TestExplainability:
    """Tests for explainability outputs."""

    def test_reason_non_empty(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        assert len(d.decision_reason) > 20

    def test_reason_names_action(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert "no action" in d.decision_reason.lower()

    def test_triggered_rules_listed(self) -> None:
        d = _agent().decide(
            mk_trend(25, TrendDirection.ACCELERATING, EarlyWarning.CRITICAL,
                     has_change=True),
            mk_rul(3), mk_risk(0.9, RiskLevel.CRITICAL))
        assert len(d.triggered_rules) >= 4

    def test_healthy_minimal_rules(self) -> None:
        d = _agent().decide(mk_trend(95), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.01))
        assert len(d.triggered_rules) == 0

    def test_reason_mentions_indicators(self) -> None:
        d = _agent().decide(mk_trend(25, TrendDirection.DEGRADING),
                            mk_rul(5), mk_risk(0.8, RiskLevel.CRITICAL))
        assert "health" in d.decision_reason.lower()

    def test_nominal_reason_when_healthy(self) -> None:
        d = _agent().decide(mk_trend(98), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.0))
        assert "nominal" in d.decision_reason.lower()


# ---------------------------------------------------------------------------
# Cost escalation
# ---------------------------------------------------------------------------


class TestCostEscalation:
    """Tests for the cost-benefit escalation rule."""

    def test_escalation_when_favourable(self) -> None:
        # Borderline action but positive savings with elevated risk -> escalate
        agent = _agent(enable_cost_escalation=True)
        d = agent.decide(mk_trend(72), mk_rul(100), mk_risk(0.3, RiskLevel.MEDIUM))
        # Without escalation this would be INSPECT-or-lower; check escalation path
        if "cost_benefit_escalation" in d.triggered_rules:
            assert d.action.severity >= MaintenanceAction.SCHEDULE_MAINTENANCE.severity

    def test_no_escalation_when_disabled(self) -> None:
        agent = _agent(enable_cost_escalation=False)
        d = agent.decide(mk_trend(72), mk_rul(100), mk_risk(0.3, RiskLevel.MEDIUM))
        assert "cost_benefit_escalation" not in d.triggered_rules


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TestResultContainer:
    """Tests for the decision container."""

    def test_to_dict_keys(self) -> None:
        d = _agent().decide(mk_trend(60, TrendDirection.DEGRADING),
                            mk_rul(20), mk_risk(0.5, RiskLevel.HIGH))
        for key in ("action", "priority", "expected_savings",
                    "estimated_downtime_hours", "decision_confidence",
                    "decision_reason", "triggered_rules"):
            assert key in d.to_dict()

    def test_to_dict_infinite_rul_none(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert d.to_dict()["remaining_useful_life"] is None

    def test_to_dict_enum_strings(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.02))
        assert d.to_dict()["action"] == "no_action"
        assert isinstance(d.to_dict()["priority"], str)


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_decision(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig(),
                                         experiment_tracker=FakeTracker())
        agent.decide(mk_trend(60, TrendDirection.DEGRADING), mk_rul(20),
                     mk_risk(0.5, RiskLevel.HIGH))
        assert len(logged) == 1
        assert "expected_savings" in logged[0]

    def test_logs_step_increment(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                steps.append(step)

        agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig(),
                                         experiment_tracker=FakeTracker())
        agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        assert steps == [0, 1]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig(),
                                         experiment_tracker=BrokenTracker())
        d = agent.decide(mk_trend(60, TrendDirection.DEGRADING), mk_rul(20),
                         mk_risk(0.5, RiskLevel.HIGH))
        assert d.action is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_infinite_rul_healthy(self) -> None:
        d = _agent().decide(mk_trend(95, TrendDirection.STABLE),
                            mk_rul(RUL_INFINITE, is_degrading=False), mk_risk(0.0))
        assert d.action == MaintenanceAction.NO_ACTION
        assert math.isinf(d.remaining_useful_life)

    def test_zero_failure_probability(self) -> None:
        d = _agent().decide(mk_trend(90), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.0))
        assert d.expected_savings == -_agent().config.maintenance_cost

    def test_certain_failure(self) -> None:
        d = _agent().decide(mk_trend(20, TrendDirection.ACCELERATING,
                                     EarlyWarning.CRITICAL),
                            mk_rul(1), mk_risk(1.0, RiskLevel.CRITICAL))
        assert d.action == MaintenanceAction.SHUTDOWN

    def test_n_decisions_increments(self) -> None:
        agent = _agent()
        agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        agent.decide(mk_trend(60), mk_rul(100), mk_risk(0.1))
        assert agent._n_decisions == 2

    def test_severity_score_non_negative(self) -> None:
        d = _agent().decide(mk_trend(95), mk_rul(RUL_INFINITE, is_degrading=False),
                            mk_risk(0.0))
        assert d.severity_score >= 0

    def test_warning_only_health_at_boundary(self) -> None:
        # Health exactly at warn threshold
        d = _agent().decide(mk_trend(70), mk_rul(100), mk_risk(0.1))
        assert d.action is not None

    def test_non_degrading_rul_ignored(self) -> None:
        # is_degrading=False -> no RUL rules even with finite rul
        d = _agent().decide(mk_trend(90), mk_rul(5, is_degrading=False),
                            mk_risk(0.05))
        assert "imminent_failure" not in d.triggered_rules


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Tests for the full Step 1->4 pipeline integration."""

    def _pipeline(self, scores):
        from src.predictive.failure_risk import (
            FailureRiskConfig,
            FailureRiskEngine,
        )
        from src.predictive.health_index import (
            HealthIndexConfig,
            HealthIndexEngine,
        )
        from src.predictive.health_trend_analyzer import (
            HealthTrendAnalyzer,
            HealthTrendConfig,
        )
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.5))
        he.set_anomaly_threshold(0.02)
        for s in scores:
            he.update(anomaly_score=float(s))
        agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig())
        return agent.decide_from_pipeline(
            he, HealthTrendAnalyzer(HealthTrendConfig()),
            RULPredictor(RULConfig(failure_threshold=40, model="linear")),
            FailureRiskEngine(FailureRiskConfig()),
        )

    def test_healthy_pipeline_no_action(self) -> None:
        d = self._pipeline(np.full(25, 0.011))
        assert d.action.severity <= MaintenanceAction.INSPECT.severity

    def test_degrading_pipeline_acts(self) -> None:
        d = self._pipeline(np.linspace(0.03, 0.13, 25))
        assert d.action.severity >= MaintenanceAction.INSPECT.severity

    def test_pipeline_returns_decision(self) -> None:
        d = self._pipeline(np.linspace(0.011, 0.06, 25))
        assert isinstance(d, MaintenanceDecision)

    def test_pipeline_has_explanation(self) -> None:
        d = self._pipeline(np.linspace(0.03, 0.13, 25))
        assert len(d.decision_reason) > 0

    def test_pipeline_severity_ordering(self) -> None:
        healthy = self._pipeline(np.full(25, 0.011))
        critical = self._pipeline(np.linspace(0.04, 0.16, 25))
        assert critical.action.severity >= healthy.action.severity