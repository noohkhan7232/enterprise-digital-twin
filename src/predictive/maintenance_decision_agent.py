#!/usr/bin/env python3
"""Maintenance decision agent for wind turbine acoustic predictive maintenance.

This module is the prescriptive capstone of the Week-5 predictive-maintenance
framework.  The earlier stages are *descriptive* — they tell an operator how
healthy a machine is (Step 1), how its health is trending and whether a regime
change occurred (the trend analyzer), how much useful life remains (Step 2), and
the probability of failure over each horizon (Step 3).  This agent closes the
loop by converting those signals into a single **actionable maintenance
recommendation**: what to do, how urgently, what it will cost, what it will
save, and — crucially for an industrial system that humans must trust — *why*.

It fuses three upstream results:

* :class:`~src.predictive.health_trend_analyzer.HealthTrendResult`
* :class:`~src.predictive.rul_predictor.RULPrediction`
* :class:`~src.predictive.failure_risk.FailureRiskPrediction`

into a :class:`MaintenanceDecision` carrying an action, a priority, a full cost
/ benefit / downtime estimate, a confidence score, and an explainable list of
the named rules that fired.

Decision model
--------------
A transparent, auditable **rule engine** scores severity from every input
dimension the requirements demand — current health, trend direction, trend
confidence, remaining useful life, failure probability, risk level, and
change-point events.  Each rule that fires contributes weighted evidence and
records its name, so the final recommendation is fully explainable rather than a
black-box score.  The aggregate severity maps to one of five actions
(``NO_ACTION`` → ``INSPECT`` → ``SCHEDULE_MAINTENANCE`` →
``IMMEDIATE_MAINTENANCE`` → ``SHUTDOWN``) and one of four priorities.

Cost-aware recommendation
-------------------------
Maintenance is an economic decision, so the agent computes it as one.  Acting
now costs a (cheaper, planned) ``maintenance_cost``; *not* acting risks a
(far more expensive, unplanned) ``failure_cost`` weighted by the failure
probability.  The ``expected_savings`` of acting is the avoided expected failure
cost minus the maintenance spend; the economic break-even failure probability is
``maintenance_cost / failure_cost``.  The agent can let this economic test
escalate an otherwise-borderline recommendation, so the action is justified on
the balance sheet, not just the severity score.

Downtime estimation
------------------
Planned maintenance is faster than emergency repair after a failure, so the
estimated downtime depends on the recommended action: an inspection costs hours,
a scheduled intervention more, an emergency repair the most.

Confidence
----------
The decision confidence reflects how much the inputs agree and how trustworthy
they are: a degrading verdict backed by a high-R² trend, a tight RUL interval,
and a consistent risk level scores high; conflicting or low-confidence inputs
score low, signalling that a human should look closer.

Architecture
------------
* :class:`MaintenanceAction` / :class:`PriorityLevel` — decision enums.
* :class:`MaintenanceDecisionConfig` — frozen, validated configuration.
* :class:`MaintenanceDecision` — the immutable, explainable result.
* :class:`MaintenanceDecisionAgent` — the engine; registry- and tracker-
  integrated, with helpers to run the full Step 1->4 pipeline end to end.

Usage::

    from src.predictive.maintenance_decision_agent import (
        MaintenanceDecisionAgent, MaintenanceDecisionConfig,
    )

    agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig())
    decision = agent.decide(trend_result, rul_prediction, risk_prediction)

    print(decision.action.value, decision.priority.value)
    print(decision.decision_reason)
    print(decision.expected_savings, decision.estimated_downtime_hours)
    for rule in decision.triggered_rules:
        print(" -", rule)

CLI::

    python src/predictive/maintenance_decision_agent.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predictive.failure_risk import FailureRiskPrediction, RiskLevel
from src.predictive.health_trend_analyzer import (
    EarlyWarning,
    HealthTrendResult,
    TrendDirection,
)
from src.predictive.rul_predictor import RULPrediction

logger = logging.getLogger("maintenance_decision_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named maintenance-decision agents.
MAINTENANCE_AGENT_REGISTRY: dict[str, type] = {}

AGENT_NAME: Final[str] = "acoustic_maintenance_agent"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MaintenanceAction(str, Enum):
    """Recommended maintenance action, ordered from least to most disruptive."""

    NO_ACTION = "no_action"
    INSPECT = "inspect"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    IMMEDIATE_MAINTENANCE = "immediate_maintenance"
    SHUTDOWN = "shutdown"

    @property
    def severity(self) -> int:
        """Integer severity rank (``NO_ACTION`` = 0 … ``SHUTDOWN`` = 4).

        Returns:
            The ordinal severity of the action.
        """
        return _ACTION_ORDER[self]


class PriorityLevel(str, Enum):
    """Priority of the recommendation, ordered from least to most urgent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        """Integer severity rank (``LOW`` = 0 … ``CRITICAL`` = 3).

        Returns:
            The ordinal severity of the priority.
        """
        return _PRIORITY_ORDER[self]


_ACTION_ORDER: Final[dict[MaintenanceAction, int]] = {
    MaintenanceAction.NO_ACTION: 0,
    MaintenanceAction.INSPECT: 1,
    MaintenanceAction.SCHEDULE_MAINTENANCE: 2,
    MaintenanceAction.IMMEDIATE_MAINTENANCE: 3,
    MaintenanceAction.SHUTDOWN: 4,
}

_PRIORITY_ORDER: Final[dict[PriorityLevel, int]] = {
    PriorityLevel.LOW: 0,
    PriorityLevel.MEDIUM: 1,
    PriorityLevel.HIGH: 2,
    PriorityLevel.CRITICAL: 3,
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_maintenance_agent(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a maintenance-decision agent by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = MAINTENANCE_AGENT_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Maintenance agent '{name}' already registered to "
                f"{existing.__name__}"
            )
        MAINTENANCE_AGENT_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered maintenance agent '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_maintenance_agent(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered maintenance-decision agent by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the agent constructor.

    Returns:
        An instantiated agent.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in MAINTENANCE_AGENT_REGISTRY:
        available = ", ".join(sorted(MAINTENANCE_AGENT_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown maintenance agent '{name}'. Available: {available}"
        )
    return MAINTENANCE_AGENT_REGISTRY[name](**kwargs)


def list_maintenance_agents() -> list[str]:
    """Return the sorted names of registered maintenance-decision agents.

    Returns:
        Sorted registry keys.
    """
    return sorted(MAINTENANCE_AGENT_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceDecisionConfig:
    """Configuration for the :class:`MaintenanceDecisionAgent`.

    Attributes:
        maintenance_cost: Cost of a planned maintenance intervention.
        failure_cost: Cost of an unplanned in-service failure (typically much
            larger — lost production, secondary damage, emergency labour).
        currency: Currency label for reporting (informational).
        inspect_downtime_hours: Downtime for an inspection.
        scheduled_downtime_hours: Downtime for scheduled maintenance.
        immediate_downtime_hours: Downtime for immediate maintenance.
        failure_downtime_hours: Downtime for an in-service failure / shutdown.
        health_warn: Health at/below which "declining health" fires.
        health_low: Health at/below which "low health" fires.
        health_critical: Health at/below which "critical health" fires.
        rul_moderate: RUL at/below which "moderate RUL" fires.
        rul_short: RUL at/below which "short RUL" fires.
        rul_imminent: RUL at/below which "imminent failure" fires.
        risk_elevated: Failure probability at/above which risk is "elevated".
        risk_high: Failure probability at/above which risk is "high".
        risk_critical: Failure probability at/above which risk is "critical".
        inspect_score: Severity at/above which the action is at least INSPECT.
        schedule_score: Severity at/above which the action is SCHEDULE.
        immediate_score: Severity at/above which the action is IMMEDIATE.
        shutdown_score: Severity at/above which the action is SHUTDOWN.
        enable_cost_escalation: Allow a positive expected-savings test to bump a
            borderline action one level up.
        min_trend_confidence: Below this trend confidence, severity from the
            trend dimension is down-weighted (uncertain trends act cautiously).
    """

    maintenance_cost:           float = 5_000.0
    failure_cost:               float = 50_000.0
    currency:                   str = "USD"
    inspect_downtime_hours:     float = 2.0
    scheduled_downtime_hours:   float = 8.0
    immediate_downtime_hours:   float = 24.0
    failure_downtime_hours:     float = 72.0
    health_warn:                float = 70.0
    health_low:                 float = 50.0
    health_critical:            float = 30.0
    rul_moderate:               float = 90.0
    rul_short:                  float = 30.0
    rul_imminent:               float = 7.0
    risk_elevated:              float = 0.25
    risk_high:                  float = 0.50
    risk_critical:              float = 0.75
    inspect_score:              int = 2
    schedule_score:             int = 5
    immediate_score:            int = 8
    shutdown_score:             int = 12
    enable_cost_escalation:     bool = True
    min_trend_confidence:       float = 0.25

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.maintenance_cost < 0 or self.failure_cost < 0:
            raise ValueError("costs must be >= 0")
        if self.failure_cost < self.maintenance_cost:
            raise ValueError("failure_cost should be >= maintenance_cost")
        for name, lo, mid, hi in (
            ("health", self.health_critical, self.health_low, self.health_warn),
        ):
            if not (0.0 <= lo <= mid <= hi <= 100.0):
                raise ValueError(f"{name} thresholds must satisfy 0<=crit<=low<=warn<=100")
        if not (0.0 < self.rul_imminent <= self.rul_short <= self.rul_moderate):
            raise ValueError("require 0 < rul_imminent <= rul_short <= rul_moderate")
        if not (0.0 < self.risk_elevated <= self.risk_high
                <= self.risk_critical < 1.0):
            raise ValueError(
                "require 0 < risk_elevated <= risk_high <= risk_critical < 1"
            )
        if not (self.inspect_score <= self.schedule_score
                <= self.immediate_score <= self.shutdown_score):
            raise ValueError("action score thresholds must be non-decreasing")
        if self.inspect_score < 1:
            raise ValueError("inspect_score must be >= 1")
        if not (0.0 <= self.min_trend_confidence <= 1.0):
            raise ValueError("min_trend_confidence must be in [0, 1]")
        for field_name in (
            "inspect_downtime_hours", "scheduled_downtime_hours",
            "immediate_downtime_hours", "failure_downtime_hours",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")


# ---------------------------------------------------------------------------
# Decision container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceDecision:
    """The result of one maintenance decision.

    Attributes:
        action: The recommended :class:`MaintenanceAction`.
        priority: The :class:`PriorityLevel` of the recommendation.
        severity_score: The aggregate rule-engine severity.
        maintenance_cost: Estimated cost of acting now (planned).
        failure_cost: Estimated cost of an unplanned failure.
        expected_savings: Expected economic benefit of acting vs. not acting.
        estimated_downtime_hours: Downtime implied by the recommended action.
        decision_confidence: Confidence in the recommendation ``[0, 1]``.
        decision_reason: A human-readable justification.
        triggered_rules: Names of every rule that fired.
        failure_probability: The failure probability used in the cost model.
        current_health: The machine's current health.
        remaining_useful_life: The RUL used (``inf`` when non-degrading).
        currency: Currency label for the cost figures.
        warnings: Non-fatal diagnostics.
    """

    action:                 MaintenanceAction
    priority:               PriorityLevel
    severity_score:         float
    maintenance_cost:       float
    failure_cost:           float
    expected_savings:       float
    estimated_downtime_hours: float
    decision_confidence:    float
    decision_reason:        str
    triggered_rules:        list[str] = field(default_factory=list)
    failure_probability:    float = 0.0
    current_health:         float = 100.0
    remaining_useful_life:  float = float("inf")
    currency:               str = "USD"
    warnings:               list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; infinite RUL renders as ``None``.
        """
        return {
            "action": self.action.value,
            "priority": self.priority.value,
            "severity_score": self.severity_score,
            "maintenance_cost": self.maintenance_cost,
            "failure_cost": self.failure_cost,
            "expected_savings": self.expected_savings,
            "estimated_downtime_hours": self.estimated_downtime_hours,
            "decision_confidence": self.decision_confidence,
            "decision_reason": self.decision_reason,
            "triggered_rules": list(self.triggered_rules),
            "failure_probability": self.failure_probability,
            "current_health": self.current_health,
            "remaining_useful_life": (
                None if math.isinf(self.remaining_useful_life)
                else self.remaining_useful_life
            ),
            "currency": self.currency,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# MaintenanceDecisionAgent
# ---------------------------------------------------------------------------


@register_maintenance_agent(AGENT_NAME)
class MaintenanceDecisionAgent:
    """Fuses health-trend, RUL, and failure-risk outputs into a decision.

    Runs a transparent, named-rule severity engine over every input dimension,
    maps the aggregate severity to an action and priority, computes the
    cost / savings / downtime, scores its own confidence, and explains itself.
    Registry- and tracker-integrated.

    Args:
        config: The agent configuration.
        experiment_tracker: Optional tracker for logging decisions.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: MaintenanceDecisionConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or MaintenanceDecisionConfig()
        self.tracker = experiment_tracker
        self._n_decisions = 0
        logger.info(
            "MaintenanceDecisionAgent ready | maint=%.0f %s | fail=%.0f %s",
            self.config.maintenance_cost, self.config.currency,
            self.config.failure_cost, self.config.currency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        trend: HealthTrendResult,
        rul: RULPrediction,
        risk: FailureRiskPrediction,
    ) -> MaintenanceDecision:
        """Produce a maintenance decision from the three upstream results.

        Args:
            trend: The :class:`HealthTrendResult` from the trend analyzer.
            rul: The :class:`RULPrediction` from the RUL predictor.
            risk: The :class:`FailureRiskPrediction` from the risk engine.

        Returns:
            A :class:`MaintenanceDecision`.
        """
        warnings: list[str] = []
        cfg = self.config

        # Resolve the key scalars from the inputs.
        health = float(trend.current_health)
        rul_value = float(rul.rul)
        p_fail = float(risk.dominant_probability)

        # --- Run the rule engine ---
        score, rules, reasons = self._score(trend, rul, risk)

        # --- Map severity -> action / priority ---
        action = self._action_from_score(score)
        priority = self._priority_from(score, risk.risk_level, trend.warning)

        # --- Cost / savings ---
        maint_cost = cfg.maintenance_cost
        fail_cost = cfg.failure_cost
        expected_savings = self._expected_savings(p_fail, maint_cost, fail_cost)

        # --- Cost-aware escalation (optional) ---
        if (cfg.enable_cost_escalation and expected_savings > 0
                and action.severity < MaintenanceAction.SCHEDULE_MAINTENANCE.severity
                and p_fail >= cfg.risk_elevated):
            action = MaintenanceAction.SCHEDULE_MAINTENANCE
            rules.append("cost_benefit_escalation")
            reasons.append(
                f"acting is economically favourable "
                f"(expected savings {expected_savings:.0f} {cfg.currency})"
            )

        # --- Downtime ---
        downtime = self._downtime_for(action)

        # --- Confidence ---
        confidence = self._decision_confidence(trend, rul, risk)

        # --- Reason text ---
        reason = self._compose_reason(action, priority, reasons)

        decision = MaintenanceDecision(
            action=action,
            priority=priority,
            severity_score=float(score),
            maintenance_cost=maint_cost,
            failure_cost=fail_cost,
            expected_savings=expected_savings,
            estimated_downtime_hours=downtime,
            decision_confidence=confidence,
            decision_reason=reason,
            triggered_rules=rules,
            failure_probability=p_fail,
            current_health=health,
            remaining_useful_life=rul_value,
            currency=cfg.currency,
            warnings=warnings,
        )
        self._log_decision(decision)
        self._n_decisions += 1
        return decision

    def decide_from_pipeline(
        self,
        health_engine: Any,
        trend_analyzer: Any,
        rul_predictor: Any,
        risk_engine: Any,
    ) -> MaintenanceDecision:
        """Run the full Step 1->4 pipeline and return a decision.

        Chains: trend analysis of the health engine's history, RUL prediction
        from that history, failure-risk estimation from the RUL, then the
        decision fusing all three.

        Args:
            health_engine: A Step-1 health engine (``history(smoothed=True)``).
            trend_analyzer: A trend analyzer (``analyze_engine``).
            rul_predictor: A RUL predictor (``predict_from_engine``).
            risk_engine: A failure-risk engine (``predict_from_rul``).

        Returns:
            A :class:`MaintenanceDecision`.
        """
        trend = trend_analyzer.analyze_engine(health_engine)
        rul = rul_predictor.predict_from_engine(health_engine)
        risk = risk_engine.predict_from_rul(rul)
        return self.decide(trend, rul, risk)

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _score(
        self,
        trend: HealthTrendResult,
        rul: RULPrediction,
        risk: FailureRiskPrediction,
    ) -> tuple[int, list[str], list[str]]:
        """Run the named-rule severity engine.

        Args:
            trend: The trend result.
            rul: The RUL prediction.
            risk: The failure-risk prediction.

        Returns:
            Tuple ``(score, triggered_rules, reasons)``.
        """
        cfg = self.config
        score = 0
        rules: list[str] = []
        reasons: list[str] = []

        # --- Current health ---
        health = trend.current_health
        if health <= cfg.health_critical:
            score += 4
            rules.append("critical_health")
            reasons.append(f"health critically low ({health:.0f})")
        elif health <= cfg.health_low:
            score += 3
            rules.append("low_health")
            reasons.append(f"health low ({health:.0f})")
        elif health <= cfg.health_warn:
            score += 1
            rules.append("declining_health")
            reasons.append(f"health declining ({health:.0f})")

        # --- Trend direction (down-weighted when trend confidence is low) ---
        trust = trend.trend_confidence >= cfg.min_trend_confidence
        if trend.trend == TrendDirection.ACCELERATING:
            inc = 3 if trust else 2
            score += inc
            rules.append("accelerating_degradation")
            reasons.append("degradation is accelerating")
        elif trend.trend == TrendDirection.DEGRADING:
            inc = 2 if trust else 1
            score += inc
            rules.append("degrading_trend")
            reasons.append("health is degrading")
        if not trust and trend.trend in (
            TrendDirection.DEGRADING, TrendDirection.ACCELERATING,
        ):
            rules.append("low_trend_confidence")
            reasons.append(
                f"trend confidence low ({trend.trend_confidence:.2f}); "
                "acting cautiously"
            )

        # --- Remaining useful life ---
        if rul.is_degrading and math.isfinite(rul.rul):
            if rul.rul <= cfg.rul_imminent:
                score += 4
                rules.append("imminent_failure")
                reasons.append(f"failure imminent (RUL {rul.rul:.1f})")
            elif rul.rul <= cfg.rul_short:
                score += 2
                rules.append("short_rul")
                reasons.append(f"short remaining life (RUL {rul.rul:.1f})")
            elif rul.rul <= cfg.rul_moderate:
                score += 1
                rules.append("moderate_rul")
                reasons.append(f"moderate remaining life (RUL {rul.rul:.1f})")

        # --- Failure probability ---
        p = risk.dominant_probability
        if p >= cfg.risk_critical:
            score += 4
            rules.append("critical_failure_risk")
            reasons.append(f"failure probability critical ({p:.2f})")
        elif p >= cfg.risk_high:
            score += 2
            rules.append("high_failure_risk")
            reasons.append(f"failure probability high ({p:.2f})")
        elif p >= cfg.risk_elevated:
            score += 1
            rules.append("elevated_failure_risk")
            reasons.append(f"failure probability elevated ({p:.2f})")

        # --- Risk level (categorical reinforcement) ---
        if risk.risk_level == RiskLevel.CRITICAL:
            score += 2
            rules.append("critical_risk_level")
        elif risk.risk_level == RiskLevel.HIGH:
            score += 1
            rules.append("high_risk_level")

        # --- Change-point event ---
        if trend.has_change_point:
            score += 1
            rules.append("regime_change_detected")
            reasons.append(
                f"regime change at step {trend.change_index} "
                f"(magnitude {trend.change_magnitude:.1f})"
            )

        # --- Early-warning reinforcement ---
        if trend.warning == EarlyWarning.CRITICAL:
            score += 2
            rules.append("critical_early_warning")
        elif trend.warning == EarlyWarning.WARNING:
            score += 1
            rules.append("early_warning")

        return score, rules, reasons

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _action_from_score(self, score: int) -> MaintenanceAction:
        """Map an aggregate severity score to an action.

        Args:
            score: The aggregate severity.

        Returns:
            The :class:`MaintenanceAction`.
        """
        cfg = self.config
        if score >= cfg.shutdown_score:
            return MaintenanceAction.SHUTDOWN
        if score >= cfg.immediate_score:
            return MaintenanceAction.IMMEDIATE_MAINTENANCE
        if score >= cfg.schedule_score:
            return MaintenanceAction.SCHEDULE_MAINTENANCE
        if score >= cfg.inspect_score:
            return MaintenanceAction.INSPECT
        return MaintenanceAction.NO_ACTION

    def _priority_from(
        self, score: int, risk_level: RiskLevel, warning: EarlyWarning
    ) -> PriorityLevel:
        """Map severity, risk level, and warning to a priority.

        Args:
            score: The aggregate severity.
            risk_level: The failure-risk level.
            warning: The early-warning level.

        Returns:
            The :class:`PriorityLevel`.
        """
        cfg = self.config
        # Base priority from the score relative to the action bands.
        if score >= cfg.immediate_score:
            base = PriorityLevel.CRITICAL
        elif score >= cfg.schedule_score:
            base = PriorityLevel.HIGH
        elif score >= cfg.inspect_score:
            base = PriorityLevel.MEDIUM
        else:
            base = PriorityLevel.LOW

        # A CRITICAL risk level or early warning lifts the priority floor.
        floor = PriorityLevel.LOW
        if risk_level == RiskLevel.CRITICAL or warning == EarlyWarning.CRITICAL:
            floor = PriorityLevel.HIGH
        elif risk_level == RiskLevel.HIGH or warning == EarlyWarning.WARNING:
            floor = PriorityLevel.MEDIUM

        return base if base.severity >= floor.severity else floor

    def _downtime_for(self, action: MaintenanceAction) -> float:
        """Return the estimated downtime for an action.

        Args:
            action: The recommended action.

        Returns:
            Estimated downtime in hours.
        """
        cfg = self.config
        return {
            MaintenanceAction.NO_ACTION: 0.0,
            MaintenanceAction.INSPECT: cfg.inspect_downtime_hours,
            MaintenanceAction.SCHEDULE_MAINTENANCE: cfg.scheduled_downtime_hours,
            MaintenanceAction.IMMEDIATE_MAINTENANCE: cfg.immediate_downtime_hours,
            MaintenanceAction.SHUTDOWN: cfg.failure_downtime_hours,
        }[action]

    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    def _expected_savings(
        self, p_fail: float, maint_cost: float, fail_cost: float
    ) -> float:
        """Compute the expected savings of acting now versus waiting.

        Expected cost of *not* acting is ``p_fail * fail_cost``; the cost of
        acting is ``maint_cost``.  The savings is the difference — positive when
        the avoided expected failure cost exceeds the maintenance spend.

        Args:
            p_fail: Failure probability over the dominant horizon.
            maint_cost: Planned maintenance cost.
            fail_cost: Unplanned failure cost.

        Returns:
            The expected savings (may be negative).
        """
        p = min(1.0, max(0.0, p_fail))
        return p * fail_cost - maint_cost

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _decision_confidence(
        self,
        trend: HealthTrendResult,
        rul: RULPrediction,
        risk: FailureRiskPrediction,
    ) -> float:
        """Score confidence in the decision from input agreement and quality.

        Combines the trend confidence, the relative tightness of the RUL
        confidence interval, and whether the trend and risk signals agree on
        the direction of concern.

        Args:
            trend: The trend result.
            rul: The RUL prediction.
            risk: The failure-risk prediction.

        Returns:
            Confidence in ``[0, 1]``.
        """
        # Trend quality.
        c_trend = float(trend.trend_confidence)

        # RUL interval tightness: narrow interval (relative to RUL) -> confident.
        if rul.is_degrading and math.isfinite(rul.rul) and rul.rul > 0:
            if math.isfinite(rul.ci_low) and math.isfinite(rul.ci_high):
                width = max(0.0, rul.ci_high - rul.ci_low)
                rel = width / max(1e-9, rul.rul)
                c_rul = 1.0 / (1.0 + rel)  # rel=0 -> 1, large rel -> 0
            else:
                c_rul = 0.5
        else:
            # Non-degrading: a confident "healthy" reading.
            c_rul = 0.7

        # Agreement: do the trend and the risk tell the same story?
        degrading = trend.trend in (
            TrendDirection.DEGRADING, TrendDirection.ACCELERATING,
        )
        risky = risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        agree = (degrading == risky)
        c_agree = 1.0 if agree else 0.5

        # Weighted blend.
        confidence = 0.4 * c_trend + 0.35 * c_rul + 0.25 * c_agree
        return float(max(0.0, min(1.0, confidence)))

    # ------------------------------------------------------------------
    # Reason text
    # ------------------------------------------------------------------

    def _compose_reason(
        self, action: MaintenanceAction, priority: PriorityLevel,
        reasons: list[str],
    ) -> str:
        """Compose a human-readable decision justification.

        Args:
            action: The recommended action.
            priority: The recommendation priority.
            reasons: The list of contributing reason fragments.

        Returns:
            A justification string.
        """
        head = (
            f"Recommended action: {action.value.replace('_', ' ')} "
            f"(priority {priority.value})"
        )
        if not reasons:
            return head + " — all indicators nominal."
        return head + " — because " + "; ".join(reasons) + "."

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_decision(self, decision: MaintenanceDecision) -> None:
        """Log a decision to the experiment tracker (failure-safe).

        Args:
            decision: The decision to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "decision_severity": decision.severity_score,
                    "decision_action": float(decision.action.severity),
                    "decision_priority": float(decision.priority.severity),
                    "expected_savings": decision.expected_savings,
                    "estimated_downtime_hours": decision.estimated_downtime_hours,
                    "decision_confidence": decision.decision_confidence,
                },
                step=self._n_decisions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic end-to-end decision demo.

    Returns:
        Exit code 0.
    """
    import numpy as np

    from src.predictive.failure_risk import FailureRiskConfig, FailureRiskEngine
    from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine
    from src.predictive.health_trend_analyzer import (
        HealthTrendAnalyzer,
        HealthTrendConfig,
    )
    from src.predictive.rul_predictor import RULConfig, RULPredictor

    scenarios = {
        "healthy": np.full(25, 0.011),
        "degrading": np.linspace(0.011, 0.06, 25),
        "critical": np.linspace(0.03, 0.13, 25),
    }
    agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig())
    analyzer = HealthTrendAnalyzer(HealthTrendConfig())
    predictor = RULPredictor(RULConfig(failure_threshold=40.0, model="linear"))
    risk_engine = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90)))

    for name, scores in scenarios.items():
        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.5))
        he.set_anomaly_threshold(0.02)
        for s in scores:
            he.update(anomaly_score=float(s))
        decision = agent.decide_from_pipeline(he, analyzer, predictor, risk_engine)
        logger.info(
            "[%-9s] %-22s priority=%-8s savings=%7.0f down=%4.0fh conf=%.2f",
            name, decision.action.value, decision.priority.value,
            decision.expected_savings, decision.estimated_downtime_hours,
            decision.decision_confidence,
        )
        logger.info("           reason: %s", decision.decision_reason)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Maintenance decision agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic end-to-end decision demo.")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("Registered maintenance agents:", list_maintenance_agents())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())