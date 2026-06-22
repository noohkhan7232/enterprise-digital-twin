#!/usr/bin/env python3
"""Scenario Planning Agent — forward-looking strategic planning layer.

Week-8 Phase-3 adds the platform's strategic-planning layer.  The prior agents
explain *what happened* (predictive stack, fleet twin) and *why* (root-cause
agent); this agent predicts *what may happen next under alternative future
decisions*.  It answers the questions a planner asks before committing:

    * What happens to risk, coverage, ROI, and savings if we cut, hold, or
      raise the maintenance budget?
    * What failure probability, downtime, and loss do we incur if maintenance
      is delayed by 7, 14, 30, or 60 days?
    * How do health, RUL, and risk move if operating load rises or falls?
    * How do maintenance demand, risk exposure, and budget scale as the fleet
      grows by 10%, 25%, 50%, or 100%?

It is a strict, additive composition of frozen modules:

* **Fleet Digital Twin** — its :class:`FleetSnapshot` is the world-state input.
* **Executive Decision Engine** — run across budgets to evaluate budget
  scenarios with real portfolio economics.
* **Decision Copilot Agent** — renders the recommended scenario into executive
  narrative.
* **Root Cause Analysis Agent** — optional; when asset evidence is supplied, the
  fleet's dominant cause is woven into the strategic commentary.
* **Monte Carlo Risk Engine** — its uncertainty methodology informs the
  confidence model; an instance may be supplied for callers that have simulator
  configurations.

All scenario predictions are deterministic, analytic functions of the snapshot's
real aggregates (no LLM, no randomness), so every projection is auditable and
reproducible.

============================================================================
Architecture
============================================================================
::

    FleetSnapshot (Phase 2)
        ▼
    ScenarioPlanningAgent
        ├── budget_scenarios()  → ExecutiveDecisionEngine per budget
        ├── delay_scenarios()   → compounding-hazard escalation
        ├── load_scenarios()    → load-coupled degradation model
        ├── growth_scenarios()  → extensive-quantity scaling
        ├── compare()           → delta metrics
        ├── rank_scenarios()    → by risk / savings / ROI / coverage
        ├── executive_summary() → best / worst / recommended / confidence
        └── plan()              → full ScenarioPlan  (+ Copilot / RCA narrative)
        ▼
    ScenarioPlanningReport · ScenarioComparison · ScenarioRanking ·
    ExecutivePlanningSummary · ScenarioPlan   (frozen · JSON-serialisable)

CLI::

    python src/agent/scenario_planning_agent.py --demo
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
from src.fleet.fleet_digital_twin import FleetSnapshot  # noqa: E402
from src.agent.decision_copilot_agent import DecisionCopilotAgent  # noqa: E402
from src.agent.root_cause_analysis_agent import (  # noqa: E402
    AssetEvidence,
    RootCauseAnalysisAgent,
)

logger = logging.getLogger("scenario_planning_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named scenario planning agents.
SCENARIO_PLANNING_REGISTRY: dict[str, type] = {}

AGENT_NAME: Final[str] = "scenario_planning_agent"

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


class ScenarioCategory(str, Enum):
    """The families of forward-looking scenarios."""

    BUDGET = "budget"
    DELAY = "delay"
    LOAD = "load"
    GROWTH = "growth"


class RankingCriterion(str, Enum):
    """Criteria by which scenarios may be ranked."""

    RISK = "risk"
    SAVINGS = "savings"
    ROI = "roi"
    COVERAGE = "coverage"


#: For each ranking criterion, whether a higher value is better.
_HIGHER_IS_BETTER: Final[dict[str, bool]] = {
    RankingCriterion.RISK.value: False,      # lower residual risk is better
    RankingCriterion.SAVINGS.value: True,
    RankingCriterion.ROI.value: True,
    RankingCriterion.COVERAGE.value: True,
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_scenario_planning_agent(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a scenario planning agent by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = SCENARIO_PLANNING_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Scenario planning agent '{name}' already registered to "
                f"{existing.__name__}"
            )
        SCENARIO_PLANNING_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered scenario planning agent '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_scenario_planning_agent(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered scenario planning agent by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the constructor.

    Returns:
        An instantiated agent.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in SCENARIO_PLANNING_REGISTRY:
        available = ", ".join(sorted(SCENARIO_PLANNING_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown scenario planning agent '{name}'. Available: {available}"
        )
    return SCENARIO_PLANNING_REGISTRY[name](**kwargs)


def list_scenario_planning_agents() -> list[str]:
    """Return the sorted names of registered scenario planning agents.

    Returns:
        Sorted registry keys.
    """
    return sorted(SCENARIO_PLANNING_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioPlanningConfig:
    """Configuration for the :class:`ScenarioPlanningAgent`.

    Attributes:
        budget_decrease_factor: Multiplier applied to the baseline budget for the
            decrease scenario (in ``(0, 1)``).
        budget_increase_factor: Multiplier for the increase scenario (``> 1``).
        delay_days: Maintenance-delay horizons to evaluate (days).
        delay_period: Hazard time constant (days) governing how fast failure
            probability compounds with delay.
        load_increase: Load multiplier for the increase scenario (``> 1``).
        load_decrease: Load multiplier for the decrease scenario (in ``(0, 1)``).
        health_sensitivity: Sensitivity of health to a unit load change.
        rul_sensitivity: Sensitivity of RUL to a unit load change.
        risk_sensitivity: Sensitivity of risk to a unit load change.
        growth_factors: Fleet-growth multipliers (e.g. 1.1 = +10%).
        currency: Currency label.
        top_n: Number of items in ranked / summary lists.
    """

    budget_decrease_factor: float = 0.50
    budget_increase_factor: float = 1.50
    delay_days:             tuple[int, ...] = (7, 14, 30, 60)
    delay_period:           float = 30.0
    load_increase:          float = 1.20
    load_decrease:          float = 0.80
    health_sensitivity:     float = 0.50
    rul_sensitivity:        float = 0.80
    risk_sensitivity:       float = 0.60
    growth_factors:         tuple[float, ...] = (1.10, 1.25, 1.50, 2.00)
    currency:               str = "USD"
    top_n:                  int = 5

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On any invalid factor, horizon, or sensitivity.
        """
        if not (0.0 < self.budget_decrease_factor < 1.0):
            raise ValueError("budget_decrease_factor must be in (0, 1)")
        if self.budget_increase_factor <= 1.0:
            raise ValueError("budget_increase_factor must be > 1")
        if not self.delay_days or any(d <= 0 for d in self.delay_days):
            raise ValueError("delay_days must be positive and non-empty")
        if self.delay_period <= 0:
            raise ValueError("delay_period must be > 0")
        if self.load_increase <= 1.0:
            raise ValueError("load_increase must be > 1")
        if not (0.0 < self.load_decrease < 1.0):
            raise ValueError("load_decrease must be in (0, 1)")
        for name in ("health_sensitivity", "rul_sensitivity", "risk_sensitivity"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not self.growth_factors or any(g <= 1.0 for g in self.growth_factors):
            raise ValueError("growth_factors must each be > 1 and non-empty")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioPlanningReport:
    """A forward-looking scenario and its predicted outcomes.

    Attributes:
        scenario_name: Human-readable scenario name.
        category: The scenario family.
        assumptions: The assumptions the scenario rests on.
        predictions: ``(metric, value)`` pairs predicted for this scenario.
        confidence: Confidence in the predictions in ``[0, 1]``.
        recommendations: Scenario-specific recommendations.
    """

    scenario_name:   str
    category:        str
    assumptions:     tuple[str, ...]
    predictions:     tuple[tuple[str, float], ...]
    confidence:      float
    recommendations: tuple[str, ...]

    def prediction(self, metric: str) -> float | None:
        """Return a predicted metric value by name, or ``None`` if absent.

        Args:
            metric: The metric key.

        Returns:
            The value, or ``None``.
        """
        for k, v in self.predictions:
            if k == metric:
                return v
        return None

    def predictions_dict(self) -> dict[str, float]:
        """Return the predictions as a dictionary."""
        return {k: v for k, v in self.predictions}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "category": self.category,
            "assumptions": list(self.assumptions),
            "predictions": [[k, _jsonsafe(v)] for k, v in self.predictions],
            "confidence": _jsonsafe(self.confidence),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class ScenarioComparison:
    """Delta metrics between a baseline and an alternative scenario.

    Attributes:
        baseline_name: The baseline scenario name.
        alternative_name: The alternative scenario name.
        deltas: ``(metric, baseline, alternative, delta)`` tuples for shared
            metrics.
        risk_delta: Change in residual risk (alternative − baseline), or ``None``.
        cost_delta: Change in expected savings, or ``None``.
        roi_delta: Change in ROI, or ``None``.
        coverage_delta: Change in coverage, or ``None``.
    """

    baseline_name:    str
    alternative_name: str
    deltas:           tuple[tuple[str, float, float, float], ...]
    risk_delta:       float | None
    cost_delta:       float | None
    roi_delta:        float | None
    coverage_delta:   float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "baseline_name": self.baseline_name,
            "alternative_name": self.alternative_name,
            "deltas": [[m, _jsonsafe(b), _jsonsafe(a), _jsonsafe(d)]
                       for m, b, a, d in self.deltas],
            "risk_delta": (None if self.risk_delta is None
                           else _jsonsafe(self.risk_delta)),
            "cost_delta": (None if self.cost_delta is None
                           else _jsonsafe(self.cost_delta)),
            "roi_delta": (None if self.roi_delta is None
                          else _jsonsafe(self.roi_delta)),
            "coverage_delta": (None if self.coverage_delta is None
                               else _jsonsafe(self.coverage_delta)),
        }


@dataclass(frozen=True)
class ScenarioRanking:
    """A ranking of scenarios by a criterion.

    Attributes:
        criterion: The ranking criterion.
        ranked: ``(scenario_name, value)`` pairs, best first.
    """

    criterion: str
    ranked:    tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "criterion": self.criterion,
            "ranked": [[n, _jsonsafe(v)] for n, v in self.ranked],
        }


@dataclass(frozen=True)
class ExecutivePlanningSummary:
    """An executive summary across a set of scenarios.

    Attributes:
        criterion: The criterion used to rank.
        best_scenario: The best scenario name.
        worst_scenario: The worst scenario name.
        recommended_scenario: The recommended scenario name.
        confidence: Mean confidence across the scenarios.
        strategic_commentary: A composed natural-language commentary.
    """

    criterion:             str
    best_scenario:         str
    worst_scenario:        str
    recommended_scenario:  str
    confidence:            float
    strategic_commentary:  str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "criterion": self.criterion,
            "best_scenario": self.best_scenario,
            "worst_scenario": self.worst_scenario,
            "recommended_scenario": self.recommended_scenario,
            "confidence": _jsonsafe(self.confidence),
            "strategic_commentary": self.strategic_commentary,
        }


@dataclass(frozen=True)
class ScenarioPlan:
    """The full forward-looking plan — the agent's top-level output.

    Attributes:
        budget_scenarios: Budget decrease / freeze / increase reports.
        delay_scenarios: Maintenance-delay reports.
        load_scenarios: Load increase / decrease reports.
        growth_scenarios: Fleet-growth reports.
        ranking: Ranking of the budget scenarios by the default criterion.
        summary: The executive planning summary.
        currency: Currency label.
    """

    budget_scenarios: tuple[ScenarioPlanningReport, ...]
    delay_scenarios:  tuple[ScenarioPlanningReport, ...]
    load_scenarios:   tuple[ScenarioPlanningReport, ...]
    growth_scenarios: tuple[ScenarioPlanningReport, ...]
    ranking:          ScenarioRanking
    summary:          ExecutivePlanningSummary
    currency:         str

    def all_reports(self) -> tuple[ScenarioPlanningReport, ...]:
        """Return every scenario report across all families."""
        return (self.budget_scenarios + self.delay_scenarios
                + self.load_scenarios + self.growth_scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "budget_scenarios": [r.to_dict() for r in self.budget_scenarios],
            "delay_scenarios": [r.to_dict() for r in self.delay_scenarios],
            "load_scenarios": [r.to_dict() for r in self.load_scenarios],
            "growth_scenarios": [r.to_dict() for r in self.growth_scenarios],
            "ranking": self.ranking.to_dict(),
            "summary": self.summary.to_dict(),
            "currency": self.currency,
        }


# ---------------------------------------------------------------------------
# Scenario Planning Agent
# ---------------------------------------------------------------------------


@register_scenario_planning_agent(AGENT_NAME)
class ScenarioPlanningAgent:
    """Forward-looking strategic planning over alternative future decisions.

    The agent projects budget, delay, load, and growth scenarios from a fleet
    snapshot, comparing and ranking them and assembling an executive summary.
    Budget scenarios compose the Executive Decision Engine for real portfolio
    economics; delay, load, and growth scenarios use deterministic analytic
    models grounded in the snapshot's aggregates.  It holds no per-call mutable
    state.

    Args:
        config: The planning configuration.
        copilot: Optional Decision Copilot Agent for narrative (created if None).
        root_cause_agent: Optional Root Cause Analysis Agent for cause context.
        monte_carlo_engine: Optional Monte Carlo engine for callers with
            simulator configurations (interoperability hook).
        experiment_tracker: Optional tracker for logging plan counts.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: ScenarioPlanningConfig | None = None,
        copilot: DecisionCopilotAgent | None = None,
        root_cause_agent: RootCauseAnalysisAgent | None = None,
        monte_carlo_engine: Any = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or ScenarioPlanningConfig()
        self.copilot = copilot or DecisionCopilotAgent()
        self.root_cause_agent = root_cause_agent
        self.monte_carlo_engine = monte_carlo_engine
        self.tracker = experiment_tracker
        self._n_plans = 0
        logger.info("ScenarioPlanningAgent ready | currency=%s",
                    self.config.currency)

    # ------------------------------------------------------------------
    # 1. Budget scenarios (compose the Executive Decision Engine)
    # ------------------------------------------------------------------

    def budget_scenarios(
        self, snapshot: FleetSnapshot, baseline_budget: float
    ) -> tuple[ScenarioPlanningReport, ...]:
        """Project budget decrease / freeze / increase scenarios.

        Args:
            snapshot: The fleet snapshot.
            baseline_budget: The current budget.

        Returns:
            Reports for the decrease, freeze, and increase scenarios.

        Raises:
            ValueError: When the snapshot is empty or the budget is negative.
        """
        return tuple(r for r, _ in
                     self._budget_with_portfolios(snapshot, baseline_budget))

    def _budget_with_portfolios(
        self, snapshot: FleetSnapshot, baseline_budget: float
    ) -> list[tuple[ScenarioPlanningReport, ExecutiveDecisionPortfolio]]:
        """Build budget scenarios, retaining each portfolio for narrative use.

        Args:
            snapshot: The fleet snapshot.
            baseline_budget: The current budget.

        Returns:
            A list of ``(report, portfolio)`` pairs.

        Raises:
            ValueError: When the snapshot is empty or the budget is negative.
        """
        self._require_snapshot(snapshot)
        if baseline_budget < 0:
            raise ValueError("baseline_budget must be >= 0")
        cfg = self.config
        levels = (
            ("Budget Decrease", baseline_budget * cfg.budget_decrease_factor),
            ("Budget Freeze", baseline_budget),
            ("Budget Increase", baseline_budget * cfg.budget_increase_factor),
        )
        out = []
        for name, budget in levels:
            engine = ExecutiveDecisionEngine(
                ExecutiveDecisionConfig(budget=budget, currency=cfg.currency))
            p = engine.recommend(snapshot)
            coverage = (len(p.selected_asset_ids) / snapshot.asset_count
                        if snapshot.asset_count else 0.0)
            predictions = (
                ("risk", float(p.average_risk_after)),
                ("coverage", float(coverage)),
                ("roi", float(p.total_roi)),
                ("expected_savings", float(p.expected_savings)),
            )
            assumptions = (
                f"Maintenance budget set to {cfg.currency} {budget:,.0f} "
                f"({budget / baseline_budget * 100:.0f}% of baseline)."
                if baseline_budget > _EPS else
                f"Maintenance budget set to {cfg.currency} {budget:,.0f}.",
            )
            recs = self._budget_recommendations(name, p, coverage)
            out.append((ScenarioPlanningReport(
                scenario_name=name,
                category=ScenarioCategory.BUDGET.value,
                assumptions=assumptions,
                predictions=predictions,
                confidence=float(p.confidence_score),
                recommendations=recs,
            ), p))
        return out

    def _budget_recommendations(
        self, name: str, portfolio: ExecutiveDecisionPortfolio, coverage: float,
    ) -> tuple[str, ...]:
        """Build recommendations for a budget scenario.

        Args:
            name: The scenario name.
            portfolio: The portfolio produced at this budget.
            coverage: The fraction of the fleet covered.

        Returns:
            Recommendation strings.
        """
        recs = []
        if name == "Budget Increase":
            recs.append(
                f"A larger budget lifts coverage to {coverage:.0%} and cuts "
                f"residual risk to {portfolio.average_risk_after:.2f}; justify "
                "if the marginal risk reduction clears the cost of capital.")
        elif name == "Budget Decrease":
            recs.append(
                f"A reduced budget leaves residual risk at "
                f"{portfolio.average_risk_after:.2f} and coverage at "
                f"{coverage:.0%}; acceptable only if the unaddressed assets are "
                "low-consequence.")
        else:
            recs.append(
                f"Holding the budget keeps coverage at {coverage:.0%} and ROI "
                f"at {portfolio.total_roi:.0%}.")
        return tuple(recs)

    # ------------------------------------------------------------------
    # 2. Delay scenarios (compounding-hazard escalation)
    # ------------------------------------------------------------------

    def delay_scenarios(
        self, snapshot: FleetSnapshot
    ) -> tuple[ScenarioPlanningReport, ...]:
        """Project maintenance-delay scenarios.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            One report per configured delay horizon.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        p0 = float(np.clip(snapshot.fleet_failure_probability, 0.0, 1.0))
        base_downtime = float(snapshot.fleet_expected_downtime)
        base_loss = float(snapshot.fleet_expected_failure_cost)

        reports = []
        for d in cfg.delay_days:
            k = 1.0 + d / cfg.delay_period
            p_delayed = float(1.0 - (1.0 - p0) ** k) if p0 < 1.0 else 1.0
            ratio = (p_delayed / p0) if p0 > _EPS else 1.0
            downtime = base_downtime * ratio
            loss = base_loss * ratio
            confidence = float(np.clip(1.0 - d / 120.0, 0.4, 0.95))
            predictions = (
                ("failure_probability", p_delayed),
                ("downtime", downtime),
                ("loss", loss),
            )
            assumptions = (
                f"Maintenance deferred by {d} days.",
                f"Failure hazard compounds with a {cfg.delay_period:.0f}-day "
                "time constant.",
            )
            recs = (
                f"Deferring {d} days raises fleet failure probability to "
                f"{p_delayed:.0%} and expected loss to {cfg.currency} "
                f"{loss:,.0f}; "
                + ("tolerable for short horizons." if d <= 14 else
                   "escalating — schedule intervention."),
            )
            reports.append(ScenarioPlanningReport(
                scenario_name=f"Delay {d} days",
                category=ScenarioCategory.DELAY.value,
                assumptions=assumptions,
                predictions=predictions,
                confidence=confidence,
                recommendations=recs,
            ))
        return tuple(reports)

    # ------------------------------------------------------------------
    # 3. Load change scenarios (load-coupled degradation)
    # ------------------------------------------------------------------

    def load_scenarios(
        self, snapshot: FleetSnapshot
    ) -> tuple[ScenarioPlanningReport, ...]:
        """Project load increase / decrease scenarios.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            Reports for the load-increase and load-decrease scenarios.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        h0 = float(snapshot.average_health)
        r0 = float(snapshot.average_rul)
        risk0 = float(np.mean([a.risk_score for a in snapshot.assets]))

        reports = []
        for label, m in (("Load Increase", cfg.load_increase),
                         ("Load Decrease", cfg.load_decrease)):
            health = float(np.clip(h0 * (1.0 - cfg.health_sensitivity * (m - 1.0)),
                                   0.0, 100.0))
            rul = float(max(r0 * (1.0 - cfg.rul_sensitivity * (m - 1.0)), 0.0))
            risk = float(np.clip(risk0 * (1.0 + cfg.risk_sensitivity * (m - 1.0)),
                                 0.0, 1.0))
            # Confidence decays with the magnitude of the load excursion.
            confidence = float(np.clip(1.0 - abs(m - 1.0) * 1.0, 0.4, 0.9))
            predictions = (
                ("health", health),
                ("rul", rul),
                ("risk", risk),
            )
            assumptions = (
                f"Operating load scaled by {m:.2f}x.",
                "Degradation rate is assumed proportional to load.",
            )
            if m > 1.0:
                rec = (f"Higher load drives health to {health:.0f}/100 and risk "
                       f"to {risk:.2f}; consider derating or accelerated "
                       "inspection.")
            else:
                rec = (f"Lower load improves health to {health:.0f}/100 and "
                       f"extends RUL to {rul:.0f}; a viable risk-mitigation "
                       "lever.")
            reports.append(ScenarioPlanningReport(
                scenario_name=label,
                category=ScenarioCategory.LOAD.value,
                assumptions=assumptions,
                predictions=predictions,
                confidence=confidence,
                recommendations=(rec,),
            ))
        return tuple(reports)

    # ------------------------------------------------------------------
    # 4. Fleet growth scenarios (extensive-quantity scaling)
    # ------------------------------------------------------------------

    def growth_scenarios(
        self, snapshot: FleetSnapshot
    ) -> tuple[ScenarioPlanningReport, ...]:
        """Project fleet-growth scenarios.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            One report per configured growth factor.

        Raises:
            ValueError: When the snapshot is empty.
        """
        self._require_snapshot(snapshot)
        cfg = self.config
        base_demand = float(sum(1 for a in snapshot.assets
                                if a.maintenance_action not in ("no_action",)))
        base_exposure = float(snapshot.fleet_expected_failure_cost)
        base_budget = float(snapshot.fleet_expected_cost)

        reports = []
        for g in cfg.growth_factors:
            demand = base_demand * g
            exposure = base_exposure * g
            budget_impact = base_budget * g
            confidence = float(np.clip(1.0 - (g - 1.0) * 0.4, 0.4, 0.95))
            predictions = (
                ("maintenance_demand", demand),
                ("risk_exposure", exposure),
                ("budget_impact", budget_impact),
            )
            assumptions = (
                f"Fleet grows by {(g - 1.0) * 100:.0f}% "
                f"(scale factor {g:.2f}x).",
                "New assets are assumed to mirror the current fleet's profile.",
            )
            rec = (
                f"A {(g - 1.0) * 100:.0f}% larger fleet requires roughly "
                f"{cfg.currency} {budget_impact:,.0f} in maintenance budget and "
                f"raises risk exposure to {cfg.currency} {exposure:,.0f}.",
            )
            reports.append(ScenarioPlanningReport(
                scenario_name=f"Fleet Growth +{(g - 1.0) * 100:.0f}%",
                category=ScenarioCategory.GROWTH.value,
                assumptions=assumptions,
                predictions=predictions,
                confidence=confidence,
                recommendations=rec,
            ))
        return tuple(reports)

    # ------------------------------------------------------------------
    # 5. Scenario comparison
    # ------------------------------------------------------------------

    def compare(
        self, baseline: ScenarioPlanningReport,
        alternative: ScenarioPlanningReport,
    ) -> ScenarioComparison:
        """Compute delta metrics between a baseline and alternative scenario.

        Args:
            baseline: The baseline scenario report.
            alternative: The alternative scenario report.

        Returns:
            A :class:`ScenarioComparison`.

        Raises:
            TypeError: When either argument is not a report.
        """
        for r in (baseline, alternative):
            if not isinstance(r, ScenarioPlanningReport):
                raise TypeError("compare requires ScenarioPlanningReport inputs")
        b = baseline.predictions_dict()
        a = alternative.predictions_dict()
        shared = [m for m in b if m in a]
        deltas = tuple((m, b[m], a[m], a[m] - b[m]) for m in shared)

        def _d(metric: str) -> float | None:
            if metric in b and metric in a:
                return a[metric] - b[metric]
            return None

        return ScenarioComparison(
            baseline_name=baseline.scenario_name,
            alternative_name=alternative.scenario_name,
            deltas=deltas,
            risk_delta=_d("risk"),
            cost_delta=_d("expected_savings"),
            roi_delta=_d("roi"),
            coverage_delta=_d("coverage"),
        )

    # ------------------------------------------------------------------
    # 6. Scenario ranking
    # ------------------------------------------------------------------

    def rank_scenarios(
        self, reports: Sequence[ScenarioPlanningReport],
        criterion: str = RankingCriterion.RISK.value,
    ) -> ScenarioRanking:
        """Rank scenarios by a criterion.

        Only scenarios carrying the relevant metric are ranked; for ``risk`` a
        lower value is better, for savings / ROI / coverage a higher value is.

        Args:
            reports: The scenario reports.
            criterion: One of ``risk``, ``savings``, ``roi``, ``coverage``.

        Returns:
            A :class:`ScenarioRanking`.

        Raises:
            ValueError: When *criterion* is not recognised.
        """
        valid = {c.value for c in RankingCriterion}
        if criterion not in valid:
            raise ValueError(f"criterion must be one of {sorted(valid)}")
        metric = ("expected_savings" if criterion == RankingCriterion.SAVINGS.value
                  else criterion)
        higher_better = _HIGHER_IS_BETTER[criterion]

        scored = []
        for r in reports:
            v = r.prediction(metric)
            if v is not None:
                scored.append((r.scenario_name, float(v)))
        scored.sort(key=lambda t: ((-t[1] if higher_better else t[1]), t[0]))
        return ScenarioRanking(criterion=criterion, ranked=tuple(scored))

    # ------------------------------------------------------------------
    # 7. Executive planning summary
    # ------------------------------------------------------------------

    def executive_summary(
        self, reports: Sequence[ScenarioPlanningReport],
        criterion: str = RankingCriterion.RISK.value,
        *, recommended_portfolio: ExecutiveDecisionPortfolio | None = None,
        cause_context: str | None = None,
    ) -> ExecutivePlanningSummary:
        """Summarise a set of scenarios for executives.

        Args:
            reports: The scenario reports to summarise.
            criterion: The ranking criterion.
            recommended_portfolio: Optional portfolio of the recommended
                scenario, used (via the Copilot) to enrich the commentary.
            cause_context: Optional root-cause context string to weave in.

        Returns:
            An :class:`ExecutivePlanningSummary`.

        Raises:
            ValueError: When no scenario carries the ranking metric.
        """
        ranking = self.rank_scenarios(reports, criterion)
        if not ranking.ranked:
            raise ValueError(
                "no scenario carries the ranking metric for the given criterion")
        best = ranking.ranked[0][0]
        worst = ranking.ranked[-1][0]
        recommended = best
        confidences = [r.confidence for r in reports]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        commentary = self._commentary(
            criterion, best, worst, recommended, recommended_portfolio,
            cause_context)
        return ExecutivePlanningSummary(
            criterion=criterion,
            best_scenario=best,
            worst_scenario=worst,
            recommended_scenario=recommended,
            confidence=confidence,
            strategic_commentary=commentary,
        )

    def _commentary(
        self, criterion: str, best: str, worst: str, recommended: str,
        portfolio: ExecutiveDecisionPortfolio | None, cause_context: str | None,
    ) -> str:
        """Compose the strategic commentary (optionally via the Copilot).

        Args:
            criterion: The ranking criterion.
            best: The best scenario name.
            worst: The worst scenario name.
            recommended: The recommended scenario name.
            portfolio: Optional portfolio for Copilot narrative.
            cause_context: Optional root-cause context.

        Returns:
            The commentary string.
        """
        parts = [
            f"Ranked by {criterion}, the strongest scenario is '{best}' and the "
            f"weakest is '{worst}'. The recommended path is '{recommended}'."
        ]
        if portfolio is not None:
            try:
                explanation = self.copilot.explain_portfolio(portfolio)
                parts.append(explanation.risk_reduction_statement)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Copilot narrative failed: %s", exc)
        if cause_context:
            parts.append(cause_context)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def plan(
        self, snapshot: FleetSnapshot, baseline_budget: float, *,
        criterion: str = RankingCriterion.RISK.value,
        evidence_items: Sequence[AssetEvidence] | None = None,
    ) -> ScenarioPlan:
        """Run the full forward-looking analysis and assemble the plan.

        Args:
            snapshot: The fleet snapshot.
            baseline_budget: The current maintenance budget.
            criterion: The ranking criterion for budget scenarios.
            evidence_items: Optional per-asset evidence; when supplied with a
                root-cause agent, the fleet's dominant cause enriches the
                commentary.

        Returns:
            A :class:`ScenarioPlan`.

        Raises:
            ValueError: When the snapshot is empty or the budget is negative.
        """
        self._require_snapshot(snapshot)
        budget_pairs = self._budget_with_portfolios(snapshot, baseline_budget)
        budget_reports = tuple(r for r, _ in budget_pairs)
        delay = self.delay_scenarios(snapshot)
        load = self.load_scenarios(snapshot)
        growth = self.growth_scenarios(snapshot)

        ranking = self.rank_scenarios(budget_reports, criterion)
        # Recommended budget scenario's portfolio for Copilot narrative.
        rec_name = ranking.ranked[0][0] if ranking.ranked else None
        rec_portfolio = next((p for r, p in budget_pairs
                              if r.scenario_name == rec_name), None)

        cause_context = self._cause_context(evidence_items)
        summary = self.executive_summary(
            budget_reports, criterion,
            recommended_portfolio=rec_portfolio, cause_context=cause_context)

        self._n_plans += 1
        self._log()
        return ScenarioPlan(
            budget_scenarios=budget_reports,
            delay_scenarios=delay,
            load_scenarios=load,
            growth_scenarios=growth,
            ranking=ranking,
            summary=summary,
            currency=self.config.currency,
        )

    def _cause_context(
        self, evidence_items: Sequence[AssetEvidence] | None
    ) -> str | None:
        """Derive a dominant-cause commentary fragment via the RCA agent.

        Args:
            evidence_items: Optional per-asset evidence.

        Returns:
            A commentary fragment, or ``None`` when unavailable.
        """
        if not evidence_items or self.root_cause_agent is None:
            return None
        try:
            fleet_rca = self.root_cause_agent.analyze_fleet(list(evidence_items))
            return (f"Root-cause analysis attributes most degradation to "
                    f"{fleet_rca.most_common_cause}; scenario planning should "
                    "weight interventions accordingly.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Root-cause context failed: %s", exc)
            return None

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
                  "fleet_failure_probability", "fleet_expected_cost",
                  "fleet_expected_failure_cost"):
            if not hasattr(snapshot, f):
                raise TypeError(f"snapshot is missing required field: {f}")
        if snapshot.asset_count == 0 or not snapshot.assets:
            raise ValueError("scenario planning requires a non-empty snapshot")

    def _log(self) -> None:
        """Log the plan count to the tracker (failure-safe)."""
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics({"scenario_plans": float(self._n_plans)},
                                     step=self._n_plans)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short scenario-planning demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    from src.fleet.fleet_digital_twin import (
        AssetInput,
        FleetDigitalTwinConfig,
        FleetDigitalTwinEngine,
    )

    rng = np.random.default_rng(9)
    assets = []
    for i in range(8):
        rate = 0.4 + 0.25 * i
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.3, 45), 0, 100)
        assets.append(AssetInput(asset_id=f"WTG-{i:03d}", asset_type="wind_turbine",
                                 location="North Sea", health_trajectory=traj))
    snap = FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)

    agent = ScenarioPlanningAgent()
    plan = agent.plan(snap, baseline_budget=20_000)

    def show(title, reports):
        print(f"=== {title} ===")
        for r in reports:
            preds = ", ".join(f"{k}={v:,.2f}" for k, v in r.predictions)
            print(f"  {r.scenario_name:22s} [{preds}] conf={r.confidence:.2f}")
        print()

    show("Budget scenarios", plan.budget_scenarios)
    show("Delay scenarios", plan.delay_scenarios)
    show("Load scenarios", plan.load_scenarios)
    show("Growth scenarios", plan.growth_scenarios)

    print("=== Scenario comparison (Freeze vs Increase) ===")
    cmp = agent.compare(plan.budget_scenarios[1], plan.budget_scenarios[2])
    print(f"  risk_delta={cmp.risk_delta:+.3f} roi_delta={cmp.roi_delta:+.2f} "
          f"coverage_delta={cmp.coverage_delta:+.2f} "
          f"savings_delta={cmp.cost_delta:+,.0f}")
    print()
    print("=== Ranking (by risk) ===")
    for n, v in plan.ranking.ranked:
        print(f"  {n:22s} risk={v:.3f}")
    print()
    print("=== Executive planning summary ===")
    s = plan.summary
    print(f"  best={s.best_scenario} worst={s.worst_scenario} "
          f"recommended={s.recommended_scenario} confidence={s.confidence:.2f}")
    print(f"  {s.strategic_commentary}")
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
    parser = argparse.ArgumentParser(description="Scenario planning agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run a scenario-planning demo.")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("Registered scenario planning agents:",
              list_scenario_planning_agents())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())