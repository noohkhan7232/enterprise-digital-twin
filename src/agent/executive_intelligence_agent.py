#!/usr/bin/env python3
"""Executive Intelligence Agent — the platform's top-level orchestration layer.

Week-8 Phase-4 is the final executive layer.  Every prior module answers one
slice of the decision problem; this agent orchestrates them into a single
Executive Intelligence Report that answers the questions leadership actually
asks, in order:

    * "What is happening right now?"          → Fleet Digital Twin
    * "What is most likely to fail?"          → risk + executive priority score
    * "Why?"                                   → Root Cause Analysis Agent
    * "What should we do?"                     → Executive Decision Engine
    * "What happens if we change strategy?"   → Scenario Planning Agent
    * "What is the best executive decision?"  → synthesised narrative + summary

This agent contains **no business logic of its own** beyond the executive
priority score and report assembly.  It is pure orchestration: it composes the
frozen Fleet, Executive, Copilot, Root-Cause, and Scenario modules and never
re-implements them.  It is deterministic, LLM-free, and uses only Python and
NumPy.

============================================================================
Architecture
============================================================================
::

    FleetSnapshot  (+ optional AssetEvidence, budget, scenario flag)
        ▼
    ExecutiveIntelligenceAgent
        ├── fleet_assessment()     → Fleet Digital Twin snapshot + Copilot
        ├── risk_assessment()      → executive priority score over assets
        ├── root_cause_assessment()→ Root Cause Analysis Agent
        ├── decision_assessment()  → Executive Decision Engine
        ├── scenario_assessment()  → Scenario Planning Agent
        ├── narrative_generation() → Decision Copilot Agent
        └── executive_summary()    → synthesis of all of the above
        ▼
    ExecutiveIntelligenceReport
        (frozen · JSON-serialisable · deterministic)

CLI::

    python src/agent/executive_intelligence_agent.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.executive.executive_decision_engine import (  # noqa: E402
    ExecutiveDecisionConfig,
    ExecutiveDecisionEngine,
    ExecutiveDecisionPortfolio,
)
from src.fleet.fleet_digital_twin import (  # noqa: E402
    FleetAsset,
    FleetSnapshot,
)
from src.agent.decision_copilot_agent import DecisionCopilotAgent  # noqa: E402
from src.agent.root_cause_analysis_agent import (  # noqa: E402
    AssetEvidence,
    RootCauseAnalysisAgent,
)
from src.agent.scenario_planning_agent import (  # noqa: E402
    ScenarioPlanningAgent,
    ScenarioPlanningConfig,
)

logger = logging.getLogger("executive_intelligence_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named executive intelligence agents.
EXECUTIVE_INTELLIGENCE_REGISTRY: dict[str, type] = {}

AGENT_NAME: Final[str] = "executive_intelligence_agent"

_SEVERITY_CAP: Final[float] = 20.0
_EPS: Final[float] = 1e-12


def _jsonsafe(x: float) -> float | None:
    """Render a non-finite float as ``None`` for JSON.

    Args:
        x: A float that may be ``inf`` or ``NaN``.

    Returns:
        ``None`` when non-finite, else the float.
    """
    return None if (math.isinf(x) or math.isnan(x)) else float(x)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskTier(str, Enum):
    """Qualitative risk tiers for executive reporting."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationCategory(str, Enum):
    """Categories of executive recommendation."""

    MAINTENANCE = "maintenance"
    BUDGET = "budget"
    SCENARIO = "scenario"
    INVESTIGATION = "investigation"


class FindingType(str, Enum):
    """Types of executive finding."""

    ROOT_CAUSE = "root_cause"
    OBSERVATION = "observation"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_executive_intelligence_agent(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering an executive intelligence agent by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = EXECUTIVE_INTELLIGENCE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Executive intelligence agent '{name}' already registered to "
                f"{existing.__name__}"
            )
        EXECUTIVE_INTELLIGENCE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered executive intelligence agent '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_executive_intelligence_agent(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered executive intelligence agent by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the constructor.

    Returns:
        An instantiated agent.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in EXECUTIVE_INTELLIGENCE_REGISTRY:
        available = ", ".join(sorted(EXECUTIVE_INTELLIGENCE_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown executive intelligence agent '{name}'. Available: {available}"
        )
    return EXECUTIVE_INTELLIGENCE_REGISTRY[name](**kwargs)


def list_executive_intelligence_agents() -> list[str]:
    """Return the sorted names of registered executive intelligence agents.

    Returns:
        Sorted registry keys.
    """
    return sorted(EXECUTIVE_INTELLIGENCE_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutiveIntelligenceConfig:
    """Configuration for the :class:`ExecutiveIntelligenceAgent`.

    Attributes:
        weight_risk: Priority-score weight on risk.
        weight_criticality: Priority-score weight on criticality (severity).
        weight_cost: Priority-score weight on cost exposure (normalised savings).
        weight_failure: Priority-score weight on failure probability.
        risk_low: Upper bound of the "low" risk tier (exclusive).
        risk_moderate: Upper bound of the "moderate" risk tier.
        risk_high: Upper bound of the "high" risk tier; at or above is critical.
        top_n: Number of items in top-risk and top-cause lists.
        include_scenarios_default: Whether ``generate_report`` runs scenarios
            when the caller does not specify.
        currency: Currency label.
    """

    weight_risk:                float = 0.40
    weight_criticality:         float = 0.20
    weight_cost:                float = 0.20
    weight_failure:             float = 0.20
    risk_low:                   float = 0.30
    risk_moderate:              float = 0.60
    risk_high:                  float = 0.80
    top_n:                      int = 5
    include_scenarios_default:  bool = True
    currency:                   str = "USD"

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid weights, thresholds, or parameters.
        """
        weights = (self.weight_risk, self.weight_criticality,
                   self.weight_cost, self.weight_failure)
        if any(w < 0 for w in weights):
            raise ValueError("priority weights must be >= 0")
        if sum(weights) <= 0:
            raise ValueError("priority weights must sum to > 0")
        if not (0.0 < self.risk_low < self.risk_moderate < self.risk_high < 1.0):
            raise ValueError(
                "risk thresholds must satisfy 0 < low < moderate < high < 1")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

    @property
    def weight_sum(self) -> float:
        """Return the sum of the four priority weights."""
        return (self.weight_risk + self.weight_criticality
                + self.weight_cost + self.weight_failure)

    def risk_tier(self, risk_score: float) -> str:
        """Map a risk score to a qualitative tier.

        Args:
            risk_score: The risk score in ``[0, 1]``.

        Returns:
            One of the :class:`RiskTier` values.
        """
        r = float(np.clip(risk_score, 0.0, 1.0))
        if r < self.risk_low:
            return RiskTier.LOW.value
        if r < self.risk_moderate:
            return RiskTier.MODERATE.value
        if r < self.risk_high:
            return RiskTier.HIGH.value
        return RiskTier.CRITICAL.value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutiveRisk:
    """A top-risk asset with its executive priority score.

    Attributes:
        asset_id: The asset identifier.
        location: The asset's location.
        risk_score: The fleet risk score in ``[0, 1]``.
        failure_probability: The failure probability in ``[0, 1]``.
        cost_exposure: The financial exposure (expected avoidable cost).
        priority_score: The composite executive priority score in ``[0, 1]``.
        risk_tier: The qualitative risk tier.
    """

    asset_id:            str
    location:            str
    risk_score:          float
    failure_probability: float
    cost_exposure:       float
    priority_score:      float
    risk_tier:           str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "asset_id": self.asset_id,
            "location": self.location,
            "risk_score": _jsonsafe(self.risk_score),
            "failure_probability": _jsonsafe(self.failure_probability),
            "cost_exposure": _jsonsafe(self.cost_exposure),
            "priority_score": _jsonsafe(self.priority_score),
            "risk_tier": self.risk_tier,
        }


@dataclass(frozen=True)
class ExecutiveFinding:
    """An executive finding (e.g. a root cause or a key observation).

    Attributes:
        finding_type: The finding type.
        subject: The subject of the finding (e.g. a cause category).
        statement: The natural-language finding statement.
        confidence: Confidence in the finding in ``[0, 1]``.
    """

    finding_type: str
    subject:      str
    statement:    str
    confidence:   float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "finding_type": self.finding_type,
            "subject": self.subject,
            "statement": self.statement,
            "confidence": _jsonsafe(self.confidence),
        }


@dataclass(frozen=True)
class ExecutiveRecommendation:
    """An executive recommendation.

    Attributes:
        category: The recommendation category.
        title: A short title.
        rationale: The supporting rationale.
        priority: A qualitative priority (high / medium / low).
    """

    category:  str
    title:     str
    rationale: str
    priority:  str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "category": self.category,
            "title": self.title,
            "rationale": self.rationale,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class ExecutiveNarrative:
    """The composed strategic narrative.

    Attributes:
        fleet_overview: A one-paragraph fleet overview.
        situation: What is happening now.
        diagnosis: Why it is happening (root cause).
        action: What to do.
        outlook: The forward-looking scenario outlook.
        executive_summary: The headline executive summary.
    """

    fleet_overview:    str
    situation:         str
    diagnosis:         str
    action:            str
    outlook:           str
    executive_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "fleet_overview": self.fleet_overview,
            "situation": self.situation,
            "diagnosis": self.diagnosis,
            "action": self.action,
            "outlook": self.outlook,
            "executive_summary": self.executive_summary,
        }


@dataclass(frozen=True)
class ExecutiveIntelligenceReport:
    """The top-level executive intelligence report — the agent's output.

    Attributes:
        fleet_overview: A one-paragraph fleet overview.
        current_health: The fleet's average health.
        current_risk: The fleet's mean asset risk score.
        current_rul: The fleet's average remaining useful life.
        top_risks: The highest-priority assets.
        root_causes: The dominant root-cause findings.
        recommended_actions: The executive recommendations.
        budget_recommendation: The budget recommendation statement.
        scenario_recommendation: The scenario recommendation statement.
        strategic_narrative: The composed strategic narrative.
        executive_summary: The headline executive summary.
        confidence: Overall confidence in ``[0, 1]``.
        currency: Currency label.
    """

    fleet_overview:          str
    current_health:          float
    current_risk:            float
    current_rul:             float
    top_risks:               tuple[ExecutiveRisk, ...]
    root_causes:             tuple[ExecutiveFinding, ...]
    recommended_actions:     tuple[ExecutiveRecommendation, ...]
    budget_recommendation:   str
    scenario_recommendation: str
    strategic_narrative:     str
    executive_summary:       str
    confidence:              float
    currency:                str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "fleet_overview": self.fleet_overview,
            "current_health": _jsonsafe(self.current_health),
            "current_risk": _jsonsafe(self.current_risk),
            "current_rul": _jsonsafe(self.current_rul),
            "top_risks": [r.to_dict() for r in self.top_risks],
            "root_causes": [f.to_dict() for f in self.root_causes],
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "budget_recommendation": self.budget_recommendation,
            "scenario_recommendation": self.scenario_recommendation,
            "strategic_narrative": self.strategic_narrative,
            "executive_summary": self.executive_summary,
            "confidence": _jsonsafe(self.confidence),
            "currency": self.currency,
        }


# ---------------------------------------------------------------------------
# Executive Intelligence Agent
# ---------------------------------------------------------------------------


@register_executive_intelligence_agent(AGENT_NAME)
class ExecutiveIntelligenceAgent:
    """Top-level orchestration agent producing executive intelligence reports.

    The agent composes the frozen Fleet, Executive, Copilot, Root-Cause, and
    Scenario modules.  It contributes only the executive priority score and the
    report synthesis; all domain logic lives in the composed modules.  It holds
    no per-call mutable state.

    Args:
        config: The agent configuration.
        copilot: Optional Decision Copilot Agent (created if None).
        root_cause_agent: Optional Root Cause Analysis Agent (created if None).
        scenario_agent: Optional Scenario Planning Agent (created if None).
        monte_carlo_engine: Optional Monte Carlo engine (interoperability hook).
        experiment_tracker: Optional tracker for logging report counts.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: ExecutiveIntelligenceConfig | None = None,
        copilot: DecisionCopilotAgent | None = None,
        root_cause_agent: RootCauseAnalysisAgent | None = None,
        scenario_agent: ScenarioPlanningAgent | None = None,
        monte_carlo_engine: Any = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or ExecutiveIntelligenceConfig()
        self.copilot = copilot or DecisionCopilotAgent()
        self.root_cause_agent = root_cause_agent or RootCauseAnalysisAgent()
        self.scenario_agent = scenario_agent or ScenarioPlanningAgent()
        self.monte_carlo_engine = monte_carlo_engine
        self.tracker = experiment_tracker
        self._n_reports = 0
        logger.info("ExecutiveIntelligenceAgent ready | currency=%s",
                    self.config.currency)

    # ------------------------------------------------------------------
    # Executive priority score
    # ------------------------------------------------------------------

    def priority_score(self, asset: FleetAsset, max_savings: float) -> float:
        """Compute the executive priority score for an asset.

        The score combines four normalised signals — risk, criticality (severity),
        cost exposure (expected savings relative to the fleet maximum), and
        failure probability — weighted and normalised to ``[0, 1]``::

            priority = ( w_risk·risk + w_crit·criticality
                       + w_cost·cost_exposure + w_fail·failure_probability )
                       / Σ weights

        Args:
            asset: The fleet asset.
            max_savings: The fleet's maximum expected savings (for normalising
                cost exposure).

        Returns:
            The priority score in ``[0, 1]``.
        """
        cfg = self.config
        risk = float(np.clip(asset.risk_score, 0.0, 1.0))
        criticality = float(np.clip(asset.severity_score / _SEVERITY_CAP,
                                    0.0, 1.0))
        cost_exposure = (asset.expected_savings / max_savings
                         if max_savings > _EPS else 0.0)
        cost_exposure = float(np.clip(cost_exposure, 0.0, 1.0))
        failure = float(np.clip(asset.failure_probability, 0.0, 1.0))
        blended = (cfg.weight_risk * risk
                   + cfg.weight_criticality * criticality
                   + cfg.weight_cost * cost_exposure
                   + cfg.weight_failure * failure)
        return float(blended / cfg.weight_sum)

    # ------------------------------------------------------------------
    # 1. Fleet assessment
    # ------------------------------------------------------------------

    def fleet_assessment(
        self, snapshot: FleetSnapshot
    ) -> tuple[str, float, float, float]:
        """Assess the current fleet state (composes the Copilot).

        Args:
            snapshot: The fleet snapshot.

        Returns:
            Tuple ``(overview, current_health, current_risk, current_rul)``.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        overview = self.copilot.explain_fleet(snapshot).summary
        health = float(snapshot.average_health)
        risk = float(np.mean([a.risk_score for a in snapshot.assets]))
        rul = float(snapshot.average_rul)
        return overview, health, risk, rul

    # ------------------------------------------------------------------
    # 2. Risk assessment
    # ------------------------------------------------------------------

    def risk_assessment(
        self, snapshot: FleetSnapshot
    ) -> tuple[ExecutiveRisk, ...]:
        """Rank assets by the executive priority score.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            The top-N assets as :class:`ExecutiveRisk`, highest priority first.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        assets = list(snapshot.assets)
        max_savings = max((a.expected_savings for a in assets), default=0.0)
        scored = []
        for a in assets:
            score = self.priority_score(a, max_savings)
            scored.append(ExecutiveRisk(
                asset_id=a.asset_id,
                location=a.location,
                risk_score=float(a.risk_score),
                failure_probability=float(a.failure_probability),
                cost_exposure=float(a.expected_savings),
                priority_score=score,
                risk_tier=cfg.risk_tier(a.risk_score),
            ))
        scored.sort(key=lambda r: (-r.priority_score, r.asset_id))
        return tuple(scored[:cfg.top_n])

    # ------------------------------------------------------------------
    # 3. Root-cause assessment
    # ------------------------------------------------------------------

    def root_cause_assessment(
        self, snapshot: FleetSnapshot,
        evidence_items: Sequence[AssetEvidence] | None = None,
    ) -> tuple[ExecutiveFinding, ...]:
        """Assess the dominant root causes (composes the Root-Cause agent).

        Args:
            snapshot: The fleet snapshot.
            evidence_items: Optional per-asset evidence; when absent, root-cause
                attribution is unavailable.

        Returns:
            The dominant root-cause findings (empty when no evidence is supplied).
        """
        if not evidence_items:
            return ()
        try:
            fleet_rca = self.root_cause_agent.analyze_fleet(
                list(evidence_items), snapshot=snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Root-cause assessment failed: %s", exc)
            return ()
        findings = []
        for freq in fleet_rca.top_causes[:self.config.top_n]:
            findings.append(ExecutiveFinding(
                finding_type=FindingType.ROOT_CAUSE.value,
                subject=freq.cause,
                statement=(f"{freq.cause} is the attributed cause for "
                           f"{freq.count} asset(s) ({freq.percentage:.0f}% of "
                           "the fleet)."),
                confidence=float(np.clip(freq.percentage / 100.0, 0.0, 1.0)),
            ))
        return tuple(findings)

    # ------------------------------------------------------------------
    # 4. Decision assessment
    # ------------------------------------------------------------------

    def decision_assessment(
        self, snapshot: FleetSnapshot, budget: float | None = None
    ) -> ExecutiveDecisionPortfolio:
        """Produce the maintenance decision (composes the Executive engine).

        Args:
            snapshot: The fleet snapshot.
            budget: The maintenance budget; defaults to the fleet's expected
                maintenance cost when ``None``.

        Returns:
            The :class:`ExecutiveDecisionPortfolio`.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        effective = (budget if budget is not None
                     else float(snapshot.fleet_expected_cost))
        engine = ExecutiveDecisionEngine(
            ExecutiveDecisionConfig(budget=effective,
                                    currency=self.config.currency))
        return engine.recommend(snapshot)

    # ------------------------------------------------------------------
    # 5. Scenario assessment
    # ------------------------------------------------------------------

    def scenario_assessment(
        self, snapshot: FleetSnapshot, budget: float | None = None, *,
        evidence_items: Sequence[AssetEvidence] | None = None,
    ) -> tuple[str, float]:
        """Assess forward-looking scenarios (composes the Scenario agent).

        Args:
            snapshot: The fleet snapshot.
            budget: The baseline budget; defaults to the fleet's expected cost.
            evidence_items: Optional evidence forwarded to scenario planning.

        Returns:
            Tuple ``(scenario_recommendation, confidence)``.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        effective = (budget if budget is not None
                     else float(snapshot.fleet_expected_cost))
        plan = self.scenario_agent.plan(
            snapshot, effective, evidence_items=evidence_items)
        rec = (f"Recommended strategic posture: {plan.summary.recommended_scenario}. "
               + plan.summary.strategic_commentary)
        return rec, float(plan.summary.confidence)

    # ------------------------------------------------------------------
    # 6. Narrative generation
    # ------------------------------------------------------------------

    def narrative_generation(
        self, overview: str, top_risks: Sequence[ExecutiveRisk],
        root_causes: Sequence[ExecutiveFinding],
        portfolio: ExecutiveDecisionPortfolio,
        scenario_recommendation: str | None,
    ) -> ExecutiveNarrative:
        """Compose the strategic narrative (composes the Copilot).

        Args:
            overview: The fleet overview.
            top_risks: The top-risk assets.
            root_causes: The root-cause findings.
            portfolio: The decision portfolio.
            scenario_recommendation: The scenario recommendation, if any.

        Returns:
            An :class:`ExecutiveNarrative`.
        """
        lead = top_risks[0] if top_risks else None
        situation = (
            f"The highest-priority asset is {lead.asset_id} "
            f"({lead.risk_tier} risk, priority {lead.priority_score:.2f})."
            if lead else "No assets exceed the attention threshold.")
        if root_causes:
            diagnosis = ("Degradation is driven primarily by "
                         f"{root_causes[0].subject}.")
        else:
            diagnosis = ("Root-cause attribution requires subsystem evidence, "
                         "which was not supplied.")
        try:
            action = self.copilot.explain_portfolio(portfolio).risk_reduction_statement
        except Exception as exc:  # noqa: BLE001
            logger.debug("Copilot portfolio narrative failed: %s", exc)
            action = (f"The recommended portfolio funds "
                      f"{len(portfolio.selected_asset_ids)} asset(s).")
        outlook = scenario_recommendation or "Scenario analysis was not requested."
        executive_summary = " ".join([
            overview, situation, diagnosis, action,
            (outlook if scenario_recommendation else "")]).strip()
        return ExecutiveNarrative(
            fleet_overview=overview,
            situation=situation,
            diagnosis=diagnosis,
            action=action,
            outlook=outlook,
            executive_summary=executive_summary,
        )

    # ------------------------------------------------------------------
    # 7. Executive report (full orchestration)
    # ------------------------------------------------------------------

    def generate_report(
        self, snapshot: FleetSnapshot, *,
        evidence_items: Sequence[AssetEvidence] | None = None,
        budget: float | None = None,
        include_scenarios: bool | None = None,
    ) -> ExecutiveIntelligenceReport:
        """Orchestrate all assessments into an executive intelligence report.

        Args:
            snapshot: The fleet snapshot (required).
            evidence_items: Optional per-asset evidence for root-cause analysis.
            budget: Optional maintenance budget (defaults to expected cost).
            include_scenarios: Whether to run scenario analysis (defaults to the
                configured value).

        Returns:
            A populated :class:`ExecutiveIntelligenceReport`.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        run_scenarios = (cfg.include_scenarios_default
                         if include_scenarios is None else include_scenarios)

        overview, health, risk, rul = self.fleet_assessment(snapshot)
        top_risks = self.risk_assessment(snapshot)
        root_causes = self.root_cause_assessment(snapshot, evidence_items)
        portfolio = self.decision_assessment(snapshot, budget)

        scenario_rec = "Scenario analysis was not requested."
        scenario_conf: float | None = None
        if run_scenarios:
            scenario_rec, scenario_conf = self.scenario_assessment(
                snapshot, budget, evidence_items=evidence_items)

        narrative = self.narrative_generation(
            overview, top_risks, root_causes, portfolio,
            scenario_rec if run_scenarios else None)

        budget_rec = self._budget_recommendation(portfolio)
        recommendations = self._recommendations(
            portfolio, root_causes, scenario_rec if run_scenarios else None,
            budget_rec)

        confidences = [portfolio.confidence_score]
        if scenario_conf is not None:
            confidences.append(scenario_conf)
        confidence = float(np.clip(np.mean(confidences), 0.0, 1.0))

        strategic_narrative = " ".join([
            narrative.situation, narrative.diagnosis, narrative.action,
            narrative.outlook]).strip()

        report = ExecutiveIntelligenceReport(
            fleet_overview=overview,
            current_health=health,
            current_risk=risk,
            current_rul=rul,
            top_risks=top_risks,
            root_causes=root_causes,
            recommended_actions=recommendations,
            budget_recommendation=budget_rec,
            scenario_recommendation=scenario_rec,
            strategic_narrative=strategic_narrative,
            executive_summary=narrative.executive_summary,
            confidence=confidence,
            currency=cfg.currency,
        )
        self._n_reports += 1
        self._log()
        return report

    # ------------------------------------------------------------------
    # Recommendation assembly
    # ------------------------------------------------------------------

    def _budget_recommendation(
        self, portfolio: ExecutiveDecisionPortfolio
    ) -> str:
        """Compose the budget recommendation statement.

        Args:
            portfolio: The decision portfolio.

        Returns:
            The budget recommendation string.
        """
        cur = self.config.currency
        return (
            f"Approve {cur} {portfolio.total_maintenance_cost:,.0f} to maintain "
            f"{len(portfolio.selected_asset_ids)} asset(s), cutting fleet risk "
            f"by {portfolio.portfolio_risk_reduction_pct:.0f}% at an ROI of "
            f"{portfolio.total_roi:.0%}.")

    def _recommendations(
        self, portfolio: ExecutiveDecisionPortfolio,
        root_causes: Sequence[ExecutiveFinding],
        scenario_rec: str | None, budget_rec: str,
    ) -> tuple[ExecutiveRecommendation, ...]:
        """Assemble the executive recommendations.

        Args:
            portfolio: The decision portfolio.
            root_causes: The root-cause findings.
            scenario_rec: The scenario recommendation, if any.
            budget_rec: The budget recommendation statement.

        Returns:
            A tuple of :class:`ExecutiveRecommendation`.
        """
        recs = []
        if portfolio.selected_asset_ids:
            recs.append(ExecutiveRecommendation(
                category=RecommendationCategory.MAINTENANCE.value,
                title=f"Maintain {len(portfolio.selected_asset_ids)} asset(s)",
                rationale=(f"Selected by the {portfolio.strategy} strategy for "
                           "the strongest risk-adjusted return."),
                priority="high"))
        recs.append(ExecutiveRecommendation(
            category=RecommendationCategory.BUDGET.value,
            title="Budget approval",
            rationale=budget_rec,
            priority="high" if portfolio.selected_asset_ids else "medium"))
        if root_causes:
            recs.append(ExecutiveRecommendation(
                category=RecommendationCategory.INVESTIGATION.value,
                title=f"Investigate {root_causes[0].subject}",
                rationale=root_causes[0].statement,
                priority="medium"))
        if scenario_rec:
            recs.append(ExecutiveRecommendation(
                category=RecommendationCategory.SCENARIO.value,
                title="Strategic posture",
                rationale=scenario_rec,
                priority="medium"))
        return tuple(recs)

    # ------------------------------------------------------------------
    # Validation & tracker
    # ------------------------------------------------------------------

    @staticmethod
    def _require_snapshot(snapshot: Any) -> None:
        """Validate that *snapshot* is a non-empty fleet snapshot.

        Args:
            snapshot: The candidate snapshot.

        Raises:
            TypeError: When required fields are missing.
            ValueError: When the snapshot has no assets.
        """
        for f in ("asset_count", "assets", "average_health", "average_rul",
                  "fleet_expected_cost"):
            if not hasattr(snapshot, f):
                raise TypeError(f"snapshot is missing required field: {f}")
        if snapshot.asset_count == 0 or not snapshot.assets:
            raise ValueError("executive intelligence requires a non-empty snapshot")

    def _log(self) -> None:
        """Log the report count to the tracker (failure-safe)."""
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics({"executive_reports": float(self._n_reports)},
                                     step=self._n_reports)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short executive-intelligence demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    from src.fleet.fleet_digital_twin import (
        AssetInput,
        FleetDigitalTwinConfig,
        FleetDigitalTwinEngine,
    )

    rng = np.random.default_rng(84)
    specs = [("WTG-001", 0.4), ("WTG-014", 0.9), ("WTG-027", 1.6),
             ("WTG-042", 2.6), ("WTG-051", 1.1), ("WTG-068", 0.6)]
    assets, evidence = [], []
    causes = ["vibration", "lubrication", "electrical", "vibration",
              "temperature", "load"]
    for (aid, rate), dom in zip(specs, causes):
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.3, 45), 0, 100)
        assets.append(AssetInput(asset_id=aid, asset_type="wind_turbine",
                                 location="North Sea", health_trajectory=traj))
        evidence.append(AssetEvidence(asset_id=aid, **{dom: 0.8}))
    snap = FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)

    agent = ExecutiveIntelligenceAgent()
    report = agent.generate_report(snap, evidence_items=evidence, budget=15000)

    print("=== EXECUTIVE INTELLIGENCE REPORT ===")
    print(f"Fleet overview: {report.fleet_overview}")
    print(f"Health={report.current_health:.0f}/100 | "
          f"Risk={report.current_risk:.2f} | RUL={report.current_rul:.0f}")
    print()
    print("Top risks (by executive priority):")
    for r in report.top_risks:
        print(f"  {r.asset_id:10s} priority={r.priority_score:.2f} "
              f"tier={r.risk_tier} risk={r.risk_score:.2f}")
    print()
    print("Root causes:")
    for f in report.root_causes:
        print(f"  {f.subject:12s} conf={f.confidence:.2f} — {f.statement}")
    print()
    print("Recommended actions:")
    for a in report.recommended_actions:
        print(f"  [{a.category}] {a.title} ({a.priority})")
    print()
    print(f"Budget: {report.budget_recommendation}")
    print(f"Scenario: {report.scenario_recommendation}")
    print()
    print(f"Confidence: {report.confidence:.2f}")
    print()
    print("Executive summary:")
    print(f"  {report.executive_summary}")
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
        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Executive intelligence agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run an executive-intelligence demo.")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("Registered executive intelligence agents:",
              list_executive_intelligence_agents())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())