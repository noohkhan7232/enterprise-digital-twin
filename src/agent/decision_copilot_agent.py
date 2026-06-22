#!/usr/bin/env python3
"""Decision Copilot Agent — rule-based explanation layer over all engines.

Week-8 Phase-1 introduces the platform's explanation and reasoning layer.  Every
prior engine produces a *decision* or a *diagnosis*; none of them explains itself
in language a non-specialist can act on.  The Decision Copilot Agent closes that
gap.  It consumes the frozen outputs of the predictive, fleet, and executive
layers and renders them into executive-friendly explanations, answers natural
questions about them with rule-based reasoning, and assembles an executive brief.

Crucially, the agent has **no LLM dependency**.  All language is produced by
deterministic templates over the structured fields of the engine outputs, and all
reasoning is rule-based.  This keeps the agent auditable (every sentence traces to
a field and a rule), reproducible (identical inputs yield identical text), and
deployable without a model runtime.

============================================================================
Capabilities
============================================================================
* ``explain_asset(asset)``       — health, trend, RUL, failure probability, risk
                                    level, and recommended action for one asset.
* ``explain_fleet(snapshot)``    — fleet health, critical assets, risk
                                    concentration, and top opportunities.
* ``explain_portfolio(portfolio)``— selected assets, savings, risk reduction,
                                    ROI, and coverage for a decision portfolio.
* ``answer_question(question, context)`` — rule-based answers to Why / Why-not /
                                    What-if / Which-asset / Which-risk /
                                    Which-action questions.
* ``generate_executive_brief(context)`` — summary, key risks, opportunities,
                                    recommendations, and a confidence statement.

============================================================================
Architecture
============================================================================
::

    FleetAsset ─────────────┐
    FleetSnapshot ──────────┼──► DecisionCopilotAgent ──► *Explanation / Brief
    ExecutiveDecisionPortfolio ┘        (deterministic templates + rule reasoning)

The agent composes the frozen Phase-2/Phase-3 contracts and modifies no prior
module.  Outputs are frozen, JSON-serialisable dataclasses.

Usage::

    from src.agent.decision_copilot_agent import (
        DecisionCopilotAgent, CopilotContext,
    )
    agent = DecisionCopilotAgent()
    print(agent.explain_asset(asset).summary)
    print(agent.answer_question("Why was WTG-042 selected?",
                                CopilotContext(snapshot=snap, portfolio=port)).answer)

CLI::

    python src/agent/decision_copilot_agent.py --demo
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
    ExecutiveDecisionPortfolio,
)
from src.fleet.fleet_digital_twin import (  # noqa: E402
    FleetAsset,
    FleetSnapshot,
)

logger = logging.getLogger("decision_copilot_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named decision copilot agents.
COPILOT_REGISTRY: dict[str, type] = {}

AGENT_NAME: Final[str] = "decision_copilot_agent"

_EPS: Final[float] = 1e-12

_ACTION_PHRASES: Final[dict[str, str]] = {
    "no_action": "no action is required",
    "inspect": "a manual inspection is advised",
    "schedule_maintenance": "maintenance should be scheduled",
    "immediate_maintenance": "immediate maintenance is required",
    "shutdown": "the asset should be shut down",
}

_BAND_PHRASES: Final[dict[str, str]] = {
    "healthy": "healthy",
    "warning": "showing early warning signs",
    "critical": "in critical condition",
}


def _jsonsafe(x: float) -> float | None:
    """Render a non-finite float as ``None`` for JSON.

    Args:
        x: A float that may be ``inf`` or ``NaN``.

    Returns:
        ``None`` when non-finite, else the float.
    """
    return None if (math.isinf(x) or math.isnan(x)) else float(x)


# ---------------------------------------------------------------------------
# Question-intent enum
# ---------------------------------------------------------------------------


class QuestionIntent(str, Enum):
    """The rule-based question intents the agent recognises."""

    WHY = "why"
    WHY_NOT = "why_not"
    WHAT_IF = "what_if"
    WHICH_ASSET = "which_asset"
    WHICH_RISK = "which_risk"
    WHICH_ACTION = "which_action"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_decision_copilot_agent(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a decision copilot agent by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = COPILOT_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Decision copilot agent '{name}' already registered to "
                f"{existing.__name__}"
            )
        COPILOT_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered decision copilot agent '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_decision_copilot_agent(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered decision copilot agent by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the constructor.

    Returns:
        An instantiated agent.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in COPILOT_REGISTRY:
        available = ", ".join(sorted(COPILOT_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown decision copilot agent '{name}'. Available: {available}"
        )
    return COPILOT_REGISTRY[name](**kwargs)


def list_decision_copilot_agents() -> list[str]:
    """Return the sorted names of registered decision copilot agents.

    Returns:
        Sorted registry keys.
    """
    return sorted(COPILOT_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionCopilotConfig:
    """Configuration for the :class:`DecisionCopilotAgent`.

    Attributes:
        currency: Currency label for monetary figures.
        risk_low: Upper bound of the "low" risk band (exclusive).
        risk_moderate: Upper bound of the "moderate" risk band.
        risk_high: Upper bound of the "high" risk band; at or above this is
            "critical".
        top_n: Number of items in top-risk / top-opportunity lists.
    """

    currency:      str = "USD"
    risk_low:      float = 0.30
    risk_moderate: float = 0.60
    risk_high:     float = 0.80
    top_n:         int = 3

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid thresholds or parameters.
        """
        if not (0.0 < self.risk_low < self.risk_moderate < self.risk_high < 1.0):
            raise ValueError(
                "risk thresholds must satisfy 0 < low < moderate < high < 1")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

    def risk_label(self, risk_score: float) -> str:
        """Map a risk score to a qualitative risk label.

        Args:
            risk_score: The risk score in ``[0, 1]``.

        Returns:
            One of ``"low"``, ``"moderate"``, ``"high"``, ``"critical"``.
        """
        r = float(np.clip(risk_score, 0.0, 1.0))
        if r < self.risk_low:
            return "low"
        if r < self.risk_moderate:
            return "moderate"
        if r < self.risk_high:
            return "high"
        return "critical"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopilotContext:
    """The decision context the agent reasons over.

    Attributes:
        snapshot: The fleet snapshot, if available.
        portfolio: The executive decision portfolio, if available.
    """

    snapshot:  FleetSnapshot | None = None
    portfolio: ExecutiveDecisionPortfolio | None = None

    def assets(self) -> tuple[FleetAsset, ...]:
        """Return the fleet assets known to the context.

        Returns:
            The snapshot's assets, or an empty tuple.
        """
        if self.snapshot is not None and self.snapshot.assets:
            return tuple(self.snapshot.assets)
        return ()


# ---------------------------------------------------------------------------
# Explanation dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetExplanation:
    """An executive-friendly explanation of a single asset.

    Attributes:
        asset_id: The asset identifier.
        risk_label: The qualitative risk label.
        health_statement: Health sentence.
        trend_statement: Degradation-trend sentence.
        rul_statement: Remaining-useful-life sentence.
        failure_statement: Failure-probability sentence.
        risk_statement: Risk-level sentence.
        action_statement: Recommended-action sentence.
        summary: The composed multi-sentence explanation.
    """

    asset_id:          str
    risk_label:        str
    health_statement:  str
    trend_statement:   str
    rul_statement:     str
    failure_statement: str
    risk_statement:    str
    action_statement:  str
    summary:           str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "asset_id": self.asset_id,
            "risk_label": self.risk_label,
            "health_statement": self.health_statement,
            "trend_statement": self.trend_statement,
            "rul_statement": self.rul_statement,
            "failure_statement": self.failure_statement,
            "risk_statement": self.risk_statement,
            "action_statement": self.action_statement,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class FleetExplanation:
    """An executive-friendly explanation of a fleet snapshot.

    Attributes:
        asset_count: Number of assets.
        health_statement: Fleet-health sentence.
        critical_statement: Critical-assets sentence.
        concentration_statement: Risk-concentration sentence.
        opportunity_statement: Top-opportunity sentence.
        critical_asset_ids: Ids of critical assets.
        top_opportunity_ids: Ids of the highest-savings assets.
        summary: The composed multi-sentence explanation.
    """

    asset_count:              int
    health_statement:         str
    critical_statement:       str
    concentration_statement:  str
    opportunity_statement:    str
    critical_asset_ids:       tuple[str, ...]
    top_opportunity_ids:      tuple[str, ...]
    summary:                  str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "asset_count": self.asset_count,
            "health_statement": self.health_statement,
            "critical_statement": self.critical_statement,
            "concentration_statement": self.concentration_statement,
            "opportunity_statement": self.opportunity_statement,
            "critical_asset_ids": list(self.critical_asset_ids),
            "top_opportunity_ids": list(self.top_opportunity_ids),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class PortfolioExplanation:
    """An executive-friendly explanation of a decision portfolio.

    Attributes:
        strategy: The optimization strategy used.
        selected_statement: Selected-assets sentence.
        savings_statement: Expected-savings sentence.
        risk_reduction_statement: Risk-reduction sentence.
        roi_statement: ROI sentence.
        coverage_statement: Coverage sentence.
        selected_asset_ids: The selected asset ids.
        summary: The composed multi-sentence explanation.
    """

    strategy:                  str
    selected_statement:        str
    savings_statement:         str
    risk_reduction_statement:  str
    roi_statement:             str
    coverage_statement:        str
    selected_asset_ids:        tuple[str, ...]
    summary:                   str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "strategy": self.strategy,
            "selected_statement": self.selected_statement,
            "savings_statement": self.savings_statement,
            "risk_reduction_statement": self.risk_reduction_statement,
            "roi_statement": self.roi_statement,
            "coverage_statement": self.coverage_statement,
            "selected_asset_ids": list(self.selected_asset_ids),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class QuestionAnswer:
    """The agent's answer to a natural-language question.

    Attributes:
        question: The original question.
        intent: The classified intent.
        answer: The natural-language answer.
        supporting_facts: The structured facts the answer rests on.
        confidence: A deterministic confidence in ``[0, 1]``.
    """

    question:         str
    intent:           str
    answer:           str
    supporting_facts: tuple[str, ...]
    confidence:       float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "question": self.question,
            "intent": self.intent,
            "answer": self.answer,
            "supporting_facts": list(self.supporting_facts),
            "confidence": _jsonsafe(self.confidence),
        }


@dataclass(frozen=True)
class ExecutiveBrief:
    """A composed executive brief.

    Attributes:
        executive_summary: The headline summary.
        key_risks: Key-risk statements.
        opportunities: Opportunity statements.
        recommendations: Recommendation statements.
        confidence_statement: The confidence statement.
    """

    executive_summary:    str
    key_risks:            tuple[str, ...]
    opportunities:        tuple[str, ...]
    recommendations:      tuple[str, ...]
    confidence_statement: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "executive_summary": self.executive_summary,
            "key_risks": list(self.key_risks),
            "opportunities": list(self.opportunities),
            "recommendations": list(self.recommendations),
            "confidence_statement": self.confidence_statement,
        }


# ---------------------------------------------------------------------------
# Decision Copilot Agent
# ---------------------------------------------------------------------------


@register_decision_copilot_agent(AGENT_NAME)
class DecisionCopilotAgent:
    """Rule-based, LLM-free explanation and reasoning layer over the platform.

    The agent renders the structured outputs of the predictive, fleet, and
    executive engines into executive-friendly language, answers natural questions
    with rule-based reasoning, and assembles an executive brief.  It holds no
    per-call mutable state and composes the frozen engine contracts only.

    Args:
        config: The copilot configuration.
        experiment_tracker: Optional tracker for logging interaction counts.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: DecisionCopilotConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or DecisionCopilotConfig()
        self.tracker = experiment_tracker
        self._n_interactions = 0
        logger.info("DecisionCopilotAgent ready | currency=%s",
                    self.config.currency)

    # ------------------------------------------------------------------
    # 1. explain_asset
    # ------------------------------------------------------------------

    def explain_asset(self, asset: FleetAsset) -> AssetExplanation:
        """Explain a single asset in executive-friendly language.

        Args:
            asset: The fleet asset to explain.

        Returns:
            An :class:`AssetExplanation`.

        Raises:
            TypeError: When *asset* lacks the required fleet-asset fields.
        """
        self._require_asset(asset)
        cfg = self.config
        cur = cfg.currency
        band = getattr(asset, "health_band", "")
        band_phrase = _BAND_PHRASES.get(band, "of indeterminate condition")
        risk_label = cfg.risk_label(asset.risk_score)

        health_stmt = (f"{asset.asset_id} is {band_phrase} "
                       f"(health {asset.health:.0f}/100).")
        trend_stmt = self._trend_statement(asset)
        rul_stmt = self._rul_statement(asset)
        failure_stmt = (f"The estimated probability of failure within the "
                        f"modelling horizon is {asset.failure_probability:.0%}.")
        risk_stmt = (f"This places it at {risk_label} risk "
                     f"(score {asset.risk_score:.2f}).")
        action_phrase = _ACTION_PHRASES.get(asset.maintenance_action,
                                            "a maintenance review is advised")
        if asset.maintenance_cost > _EPS:
            action_stmt = (f"Recommended action: {action_phrase}, at an "
                           f"estimated {cur} {asset.maintenance_cost:,.0f}, "
                           f"avoiding about {cur} {asset.expected_savings:,.0f} "
                           "in expected failure costs.")
        else:
            action_stmt = f"Recommended action: {action_phrase}."

        summary = " ".join([health_stmt, trend_stmt, rul_stmt, failure_stmt,
                            risk_stmt, action_stmt])
        self._record()
        return AssetExplanation(
            asset_id=asset.asset_id,
            risk_label=risk_label,
            health_statement=health_stmt,
            trend_statement=trend_stmt,
            rul_statement=rul_stmt,
            failure_statement=failure_stmt,
            risk_statement=risk_stmt,
            action_statement=action_stmt,
            summary=summary,
        )

    def _trend_statement(self, asset: FleetAsset) -> str:
        """Compose a qualitative degradation-trend sentence.

        The trend direction is inferred from the asset's diagnostic state
        (health band and risk score), since a frozen :class:`FleetAsset` records
        condition rather than an explicit slope.

        Args:
            asset: The asset.

        Returns:
            The trend sentence.
        """
        band = getattr(asset, "health_band", "")
        if band == "critical" or asset.risk_score >= self.config.risk_high:
            return "Its condition is deteriorating and warrants prompt attention."
        if band == "warning" or asset.risk_score >= self.config.risk_moderate:
            return "Its condition is trending downward and should be watched."
        return "Its condition is currently stable."

    def _rul_statement(self, asset: FleetAsset) -> str:
        """Compose a remaining-useful-life sentence (handles infinite RUL).

        Args:
            asset: The asset.

        Returns:
            The RUL sentence.
        """
        rul = asset.predicted_rul
        if rul is None or math.isinf(rul) or math.isnan(rul):
            return ("No end-of-life is projected within the modelling "
                    "horizon.")
        return (f"It has an estimated {rul:,.0f} units of remaining useful "
                "life.")

    # ------------------------------------------------------------------
    # 2. explain_fleet
    # ------------------------------------------------------------------

    def explain_fleet(self, snapshot: FleetSnapshot) -> FleetExplanation:
        """Explain a fleet snapshot in executive-friendly language.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            A :class:`FleetExplanation`.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        cur = cfg.currency
        assets = list(snapshot.assets)

        health_stmt = (
            f"The fleet of {snapshot.asset_count} assets has an average health "
            f"of {snapshot.average_health:.0f}/100, with "
            f"{snapshot.critical_assets} critical, {snapshot.warning_assets} in "
            f"warning, and {snapshot.healthy_assets} healthy.")

        critical = sorted([a for a in assets if getattr(a, "health_band", "") == "critical"],
                          key=lambda a: (-a.risk_score, a.asset_id))
        critical_ids = tuple(a.asset_id for a in critical)
        if critical_ids:
            critical_stmt = ("Critical assets requiring attention: "
                             + ", ".join(critical_ids[:cfg.top_n]) + ".")
        else:
            critical_stmt = "No assets are currently in critical condition."

        conc = snapshot.risk_concentration
        if conc >= 0.5:
            conc_desc = "highly concentrated in a few assets"
        elif conc >= 0.25:
            conc_desc = "moderately concentrated"
        else:
            conc_desc = "well distributed across the fleet"
        concentration_stmt = (f"Fleet risk is {conc_desc} "
                              f"(concentration index {conc:.2f}).")

        opps = sorted(assets, key=lambda a: (-a.expected_savings, a.asset_id))
        opp_ids = tuple(a.asset_id for a in opps[:cfg.top_n])
        if opp_ids and opps[0].expected_savings > _EPS:
            opportunity_stmt = (
                "The largest maintenance opportunities are "
                + ", ".join(opp_ids)
                + f", with a combined expected saving of {cur} "
                + f"{sum(a.expected_savings for a in opps[:cfg.top_n]):,.0f}.")
        else:
            opportunity_stmt = "No material maintenance opportunities were identified."

        summary = " ".join([health_stmt, critical_stmt, concentration_stmt,
                            opportunity_stmt])
        self._record()
        return FleetExplanation(
            asset_count=snapshot.asset_count,
            health_statement=health_stmt,
            critical_statement=critical_stmt,
            concentration_statement=concentration_stmt,
            opportunity_statement=opportunity_stmt,
            critical_asset_ids=critical_ids,
            top_opportunity_ids=opp_ids,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 3. explain_portfolio
    # ------------------------------------------------------------------

    def explain_portfolio(
        self, portfolio: ExecutiveDecisionPortfolio
    ) -> PortfolioExplanation:
        """Explain a decision portfolio in executive-friendly language.

        Args:
            portfolio: The executive decision portfolio.

        Returns:
            A :class:`PortfolioExplanation`.

        Raises:
            TypeError: When *portfolio* lacks the required fields.
        """
        self._require_portfolio(portfolio)
        cur = self.config.currency
        n_sel = len(portfolio.selected_asset_ids)

        if n_sel:
            selected_stmt = (
                f"Using the {portfolio.strategy} strategy, {n_sel} asset(s) "
                f"are recommended for maintenance: "
                + ", ".join(portfolio.selected_asset_ids) + ".")
        else:
            selected_stmt = ("No assets are recommended for maintenance under "
                             f"the {portfolio.strategy} strategy and current "
                             "budget.")
        savings_stmt = (
            f"The portfolio costs {cur} {portfolio.total_maintenance_cost:,.0f} "
            f"and is expected to return {cur} {portfolio.expected_savings:,.0f} "
            "in savings.")
        risk_reduction_stmt = (
            f"It reduces mean fleet risk by "
            f"{portfolio.portfolio_risk_reduction_pct:.0f}% "
            f"(from {portfolio.average_risk_before:.2f} to "
            f"{portfolio.average_risk_after:.2f}).")
        roi_stmt = f"The expected return on investment is {portfolio.total_roi:.0%}."
        if not math.isinf(portfolio.budget):
            coverage_stmt = (f"This consumes {portfolio.budget_utilization:.0%} "
                             "of the approved budget.")
        else:
            coverage_stmt = "The selection was made without a binding budget."

        summary = " ".join([selected_stmt, savings_stmt, risk_reduction_stmt,
                            roi_stmt, coverage_stmt])
        self._record()
        return PortfolioExplanation(
            strategy=portfolio.strategy,
            selected_statement=selected_stmt,
            savings_statement=savings_stmt,
            risk_reduction_statement=risk_reduction_stmt,
            roi_statement=roi_stmt,
            coverage_statement=coverage_stmt,
            selected_asset_ids=tuple(portfolio.selected_asset_ids),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 4. answer_question
    # ------------------------------------------------------------------

    def answer_question(
        self, question: str, context: CopilotContext
    ) -> QuestionAnswer:
        """Answer a natural-language question with rule-based reasoning.

        The question is classified into an intent (why / why-not / what-if /
        which-asset / which-risk / which-action) and routed to the matching
        rule.  The agent never raises on missing context — it returns a
        low-confidence fallback instead.

        Args:
            question: The natural-language question.
            context: The decision context.

        Returns:
            A :class:`QuestionAnswer`.
        """
        if not isinstance(question, str) or not question.strip():
            self._record()
            return QuestionAnswer(
                question=str(question), intent=QuestionIntent.UNKNOWN.value,
                answer="No question was provided.", supporting_facts=(),
                confidence=0.0)
        if context is None:
            context = CopilotContext()

        intent = self._classify(question)
        handler = {
            QuestionIntent.WHY: self._answer_why,
            QuestionIntent.WHY_NOT: self._answer_why_not,
            QuestionIntent.WHAT_IF: self._answer_what_if,
            QuestionIntent.WHICH_ASSET: self._answer_which_asset,
            QuestionIntent.WHICH_RISK: self._answer_which_risk,
            QuestionIntent.WHICH_ACTION: self._answer_which_action,
        }.get(intent, self._answer_unknown)
        answer, facts, confidence = handler(question, context)
        self._record()
        return QuestionAnswer(
            question=question, intent=intent.value, answer=answer,
            supporting_facts=tuple(facts), confidence=float(confidence))

    @staticmethod
    def _classify(question: str) -> QuestionIntent:
        """Classify a question into an intent via ordered keyword rules.

        Args:
            question: The question.

        Returns:
            The :class:`QuestionIntent`.
        """
        q = question.lower()
        negators = ("why not", "why isn", "why wasn", "why didn", "why aren",
                    "why won", "why doesn")
        if any(n in q for n in negators):
            return QuestionIntent.WHY_NOT
        if "what if" in q or "what would happen" in q:
            return QuestionIntent.WHAT_IF
        if "why" in q:
            return QuestionIntent.WHY
        if "which asset" in q or "what asset" in q:
            return QuestionIntent.WHICH_ASSET
        if "which risk" in q or "what risk" in q or "biggest risk" in q:
            return QuestionIntent.WHICH_RISK
        if ("which action" in q or "what action" in q or "what should" in q
                or "what do" in q):
            return QuestionIntent.WHICH_ACTION
        return QuestionIntent.UNKNOWN

    def _resolve_asset(
        self, question: str, context: CopilotContext
    ) -> FleetAsset | None:
        """Resolve a fleet asset referenced in the question, if any.

        Scans the question for any known asset id (case-insensitive), choosing
        the earliest occurrence for determinism.

        Args:
            question: The question.
            context: The decision context.

        Returns:
            The resolved asset or ``None``.
        """
        q = question.lower()
        best: tuple[int, str, FleetAsset] | None = None
        for a in context.assets():
            pos = q.find(a.asset_id.lower())
            if pos >= 0:
                cand = (pos, a.asset_id, a)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        return best[2] if best else None

    def _answer_why(self, question, context):
        """Answer a 'why' question (rationale for a decision/state)."""
        asset = self._resolve_asset(question, context)
        if asset is not None:
            label = self.config.risk_label(asset.risk_score)
            facts = [
                f"health={asset.health:.0f}",
                f"failure_probability={asset.failure_probability:.0%}",
                f"risk_score={asset.risk_score:.2f}",
                f"action={asset.maintenance_action}",
            ]
            selected = (context.portfolio is not None
                        and asset.asset_id in context.portfolio.selected_asset_ids)
            sel_clause = ((" It was selected for maintenance because its "
                           "priority outweighed its cost within budget.")
                          if selected else "")
            answer = (
                f"{asset.asset_id} is rated {label} risk (score "
                f"{asset.risk_score:.2f}) because its health is "
                f"{asset.health:.0f}/100 and its failure probability is "
                f"{asset.failure_probability:.0%}.{sel_clause}")
            return answer, facts, 0.9
        if context.portfolio is not None:
            p = context.portfolio
            facts = [f"strategy={p.strategy}",
                     f"selected={len(p.selected_asset_ids)}",
                     f"roi={p.total_roi:.0%}"]
            answer = (
                f"The {p.strategy} strategy selected "
                f"{len(p.selected_asset_ids)} asset(s) because they maximised "
                f"risk reduction and return within the budget, achieving an ROI "
                f"of {p.total_roi:.0%}.")
            return answer, facts, 0.75
        return ("I need a fleet snapshot or decision portfolio to explain a "
                "decision.", [], 0.3)

    def _answer_why_not(self, question, context):
        """Answer a 'why not' question (why an asset was not selected)."""
        asset = self._resolve_asset(question, context)
        if asset is not None and context.portfolio is not None:
            p = context.portfolio
            if asset.asset_id in p.selected_asset_ids:
                return (f"{asset.asset_id} was in fact selected for maintenance.",
                        [f"selected={asset.asset_id}"], 0.85)
            facts = [f"risk_score={asset.risk_score:.2f}",
                     f"cost={asset.maintenance_cost:,.0f}",
                     f"budget_utilization={p.budget_utilization:.0%}"]
            reason = ("its priority score was lower than the selected assets'"
                      if p.budget_utilization < 0.999
                      else "the budget was exhausted by higher-priority assets")
            answer = (
                f"{asset.asset_id} was not selected because {reason}. Its risk "
                f"score is {asset.risk_score:.2f} and its maintenance cost is "
                f"{self.config.currency} {asset.maintenance_cost:,.0f}.")
            return answer, facts, 0.85
        if context.portfolio is None:
            return ("I need a decision portfolio to explain why an asset was "
                    "not selected.", [], 0.3)
        return ("I could not identify which asset you are asking about; please "
                "name it.", [], 0.4)

    def _answer_what_if(self, question, context):
        """Answer a 'what if' question with rule-based, qualitative reasoning."""
        q = question.lower()
        if "budget" in q and context.portfolio is not None:
            p = context.portfolio
            unselected = [pa for pa in p.prioritization if not pa.selected]
            if unselected:
                nxt = min(unselected, key=lambda pa: pa.rank)
                facts = [f"next_asset={nxt.asset_id}",
                         f"next_cost={nxt.maintenance_cost:,.0f}",
                         f"next_savings={nxt.expected_savings:,.0f}"]
                answer = (
                    f"Increasing the budget would next fund {nxt.asset_id} "
                    f"(cost {self.config.currency} {nxt.maintenance_cost:,.0f}, "
                    f"expected saving {self.config.currency} "
                    f"{nxt.expected_savings:,.0f}), the highest-priority asset "
                    "not currently covered.")
                return answer, facts, 0.7
            return ("The current budget already funds every recommended asset; "
                    "more budget would add no further maintenance.", [], 0.7)
        if ("fail" in q or "failure" in q) and context.snapshot is not None:
            s = context.snapshot
            facts = [f"fleet_expected_failure_cost={s.fleet_expected_failure_cost:,.0f}"]
            answer = (
                f"If the at-risk assets were left to fail, the fleet's expected "
                f"failure cost is about {self.config.currency} "
                f"{s.fleet_expected_failure_cost:,.0f}, which proactive "
                "maintenance is designed to avoid.")
            return answer, facts, 0.7
        return ("I can reason about what-if scenarios for budget changes or "
                "asset failures when given a portfolio or snapshot.", [], 0.4)

    def _answer_which_asset(self, question, context):
        """Answer a 'which asset' question by superlative."""
        assets = context.assets()
        if not assets:
            return ("I need a fleet snapshot to identify an asset.", [], 0.3)
        q = question.lower()
        if "save" in q or "saving" in q or "opportunit" in q:
            target = max(assets, key=lambda a: (a.expected_savings, a.asset_id))
            answer = (f"{target.asset_id} is the largest opportunity, with an "
                      f"expected saving of {self.config.currency} "
                      f"{target.expected_savings:,.0f}.")
            facts = [f"asset={target.asset_id}",
                     f"savings={target.expected_savings:,.0f}"]
        elif "health" in q or "unhealth" in q or "worst" in q:
            target = min(assets, key=lambda a: (a.health, a.asset_id))
            answer = (f"{target.asset_id} is in the worst health at "
                      f"{target.health:.0f}/100.")
            facts = [f"asset={target.asset_id}", f"health={target.health:.0f}"]
        else:
            target = max(assets, key=lambda a: (a.risk_score, a.asset_id))
            answer = (f"{target.asset_id} carries the highest risk "
                      f"(score {target.risk_score:.2f}).")
            facts = [f"asset={target.asset_id}",
                     f"risk_score={target.risk_score:.2f}"]
        return answer, facts, 0.85

    def _answer_which_risk(self, question, context):
        """Answer a 'which risk' question by identifying the dominant exposure."""
        assets = context.assets()
        if not assets:
            return ("I need a fleet snapshot to identify the dominant risk.",
                    [], 0.3)
        target = max(assets, key=lambda a: (a.risk_score, a.asset_id))
        facts = [f"asset={target.asset_id}",
                 f"risk_score={target.risk_score:.2f}",
                 f"failure_probability={target.failure_probability:.0%}"]
        answer = (
            f"The dominant risk is {target.asset_id} with a risk score of "
            f"{target.risk_score:.2f}, driven by a {target.failure_probability:.0%} "
            "failure probability.")
        return answer, facts, 0.85

    def _answer_which_action(self, question, context):
        """Answer a 'which action' question for a named or top-priority asset."""
        asset = self._resolve_asset(question, context)
        if asset is None:
            assets = context.assets()
            if not assets:
                return ("I need a fleet snapshot to recommend an action.",
                        [], 0.3)
            asset = max(assets, key=lambda a: (a.risk_score, a.asset_id))
        phrase = _ACTION_PHRASES.get(asset.maintenance_action,
                                     "a maintenance review")
        facts = [f"asset={asset.asset_id}",
                 f"action={asset.maintenance_action}"]
        answer = (f"For {asset.asset_id}, {phrase} "
                  f"(risk score {asset.risk_score:.2f}).")
        return answer, facts, 0.85

    def _answer_unknown(self, question, context):
        """Provide a helpful fallback for unrecognised questions."""
        return ("I can explain assets, fleets, and portfolios, and answer "
                "why, why-not, what-if, and which-asset/risk/action questions.",
                [], 0.3)

    # ------------------------------------------------------------------
    # 5. generate_executive_brief
    # ------------------------------------------------------------------

    def generate_executive_brief(
        self, context: CopilotContext
    ) -> ExecutiveBrief:
        """Assemble an executive brief from the available context.

        Args:
            context: The decision context (needs a snapshot and/or portfolio).

        Returns:
            An :class:`ExecutiveBrief`.

        Raises:
            ValueError: When the context has neither a snapshot nor a portfolio.
        """
        if context is None or (context.snapshot is None
                               and context.portfolio is None):
            raise ValueError(
                "generate_executive_brief requires a snapshot or a portfolio")
        cur = self.config.currency
        cfg = self.config
        snap = context.snapshot
        port = context.portfolio

        # Executive summary.
        parts = []
        if snap is not None:
            parts.append(
                f"The fleet of {snap.asset_count} assets averages "
                f"{snap.average_health:.0f}/100 health with "
                f"{snap.critical_assets} critical asset(s).")
        if port is not None:
            parts.append(
                f"The recommended portfolio funds "
                f"{len(port.selected_asset_ids)} asset(s) for {cur} "
                f"{port.total_maintenance_cost:,.0f}, cutting fleet risk by "
                f"{port.portfolio_risk_reduction_pct:.0f}% at an ROI of "
                f"{port.total_roi:.0%}.")
        executive_summary = " ".join(parts)

        # Key risks.
        key_risks: list[str] = []
        if snap is not None and snap.assets:
            ranked = sorted(snap.assets,
                            key=lambda a: (-a.risk_score, a.asset_id))
            for a in ranked[:cfg.top_n]:
                key_risks.append(
                    f"{a.asset_id}: {cfg.risk_label(a.risk_score)} risk "
                    f"(score {a.risk_score:.2f}, "
                    f"{a.failure_probability:.0%} failure probability).")

        # Opportunities.
        opportunities: list[str] = []
        if snap is not None and snap.assets:
            by_sav = sorted(snap.assets,
                            key=lambda a: (-a.expected_savings, a.asset_id))
            for a in by_sav[:cfg.top_n]:
                if a.expected_savings > _EPS:
                    opportunities.append(
                        f"{a.asset_id}: up to {cur} {a.expected_savings:,.0f} "
                        "in avoidable cost.")

        # Recommendations.
        recommendations: list[str] = []
        if port is not None:
            recommendations.extend(port.recommendations)
        if not recommendations and snap is not None:
            recommendations.append(
                "Review the critical assets and authorise inspection.")

        # Confidence statement.
        if port is not None:
            conf = port.confidence_score
            qualifier = ("high" if conf >= 0.75 else
                         "moderate" if conf >= 0.5 else "limited")
            confidence_statement = (
                f"Overall decision confidence is {qualifier} "
                f"({conf:.2f}), reflecting data quality, coverage of at-risk "
                "assets, and risk dispersion.")
        else:
            confidence_statement = (
                "No decision portfolio was provided, so a portfolio confidence "
                "score is unavailable.")

        self._record()
        return ExecutiveBrief(
            executive_summary=executive_summary,
            key_risks=tuple(key_risks),
            opportunities=tuple(opportunities),
            recommendations=tuple(recommendations),
            confidence_statement=confidence_statement,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_asset(asset: Any) -> None:
        """Validate that *asset* has the required fleet-asset fields.

        Args:
            asset: The candidate asset.

        Raises:
            TypeError: When required fields are missing.
        """
        required = ("asset_id", "health", "predicted_rul", "failure_probability",
                    "risk_score", "maintenance_action", "maintenance_cost",
                    "expected_savings")
        missing = [f for f in required if not hasattr(asset, f)]
        if missing:
            raise TypeError(f"asset is missing required fields: {missing}")

    @staticmethod
    def _require_snapshot(snapshot: Any) -> None:
        """Validate that *snapshot* is a non-empty fleet snapshot.

        Args:
            snapshot: The candidate snapshot.

        Raises:
            TypeError: When required fields are missing.
            ValueError: When the snapshot has no assets.
        """
        for f in ("asset_count", "assets", "average_health", "risk_concentration"):
            if not hasattr(snapshot, f):
                raise TypeError(f"snapshot is missing required field: {f}")
        if snapshot.asset_count == 0 or not snapshot.assets:
            raise ValueError("explain_fleet requires a non-empty snapshot")

    @staticmethod
    def _require_portfolio(portfolio: Any) -> None:
        """Validate that *portfolio* has the required portfolio fields.

        Args:
            portfolio: The candidate portfolio.

        Raises:
            TypeError: When required fields are missing.
        """
        required = ("strategy", "selected_asset_ids", "total_maintenance_cost",
                    "expected_savings", "total_roi",
                    "portfolio_risk_reduction_pct", "average_risk_before",
                    "average_risk_after", "budget", "budget_utilization")
        missing = [f for f in required if not hasattr(portfolio, f)]
        if missing:
            raise TypeError(f"portfolio is missing required fields: {missing}")

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _record(self) -> None:
        """Increment the interaction counter and log it (failure-safe)."""
        self._n_interactions += 1
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {"copilot_interactions": float(self._n_interactions)},
                step=self._n_interactions)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short copilot demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    from src.executive.executive_decision_engine import (
        ExecutiveDecisionConfig,
        ExecutiveDecisionEngine,
    )
    from src.fleet.fleet_digital_twin import (
        AssetInput,
        FleetDigitalTwinConfig,
        FleetDigitalTwinEngine,
    )

    rng = np.random.default_rng(8)
    specs = [("WTG-001", 0.4), ("WTG-014", 0.9), ("WTG-027", 1.6),
             ("WTG-042", 2.6), ("WTG-051", 1.1), ("WTG-068", 0.6)]
    assets = []
    for aid, rate in specs:
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.3, 45), 0, 100)
        assets.append(AssetInput(asset_id=aid, asset_type="wind_turbine",
                                 location="North Sea", health_trajectory=traj))
    snap = FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)
    port = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=15000)).recommend(snap)
    ctx = CopilotContext(snapshot=snap, portfolio=port)
    agent = DecisionCopilotAgent()

    print("=== explain_asset (highest risk) ===")
    worst = max(snap.assets, key=lambda a: a.risk_score)
    print(agent.explain_asset(worst).summary)
    print()
    print("=== explain_fleet ===")
    print(agent.explain_fleet(snap).summary)
    print()
    print("=== explain_portfolio ===")
    print(agent.explain_portfolio(port).summary)
    print()
    print("=== answer_question ===")
    for q in [f"Why was {port.selected_asset_ids[0] if port.selected_asset_ids else worst.asset_id} selected?",
              f"Why not {worst.asset_id}?",
              "What if we increase the budget?",
              "Which asset has the highest risk?",
              "Which action should we take?"]:
        ans = agent.answer_question(q, ctx)
        print(f"Q: {q}")
        print(f"   [{ans.intent}] {ans.answer} (confidence {ans.confidence:.2f})")
    print()
    print("=== generate_executive_brief ===")
    brief = agent.generate_executive_brief(ctx)
    print(brief.executive_summary)
    print("Key risks:")
    for r in brief.key_risks:
        print(f"  - {r}")
    print(brief.confidence_statement)
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
    parser = argparse.ArgumentParser(description="Decision copilot agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run a copilot demo.")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("Registered decision copilot agents:",
              list_decision_copilot_agents())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())