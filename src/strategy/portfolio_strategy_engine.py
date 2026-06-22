#!/usr/bin/env python3
"""Strategic portfolio optimization engine — capital-allocation intelligence.

Week-7 Phase-4 is the platform's strategic decision-optimization layer.  It sits
*above* the Executive Decision Engine: where that engine answers "given a budget,
which assets do we repair?", this engine answers the questions a capital owner
asks *before* fixing the budget:

    * How do different strategic postures (risk-first, ROI-first,
      criticality-first, balanced) compare for the same fleet?
    * How do risk reduction, savings, ROI, and coverage respond as we sweep the
      budget from low to high?
    * Which budget levels are Pareto-efficient — non-dominated trade-offs of
      savings against residual risk?
    * Where is the capital-efficiency knee, beyond which extra budget buys
      little additional risk reduction?
    * What should we strategically recommend — increase the budget, hold, or
      stop because ROI has flattened?

It is a strict, additive **composition** of the frozen Executive Decision Engine
(Phase 3) and, through it, the whole prognostic stack.  It runs that engine
across strategic postures and budget levels, then analyses the resulting
:class:`~src.executive.executive_decision_engine.ExecutiveDecisionPortfolio`
objects.  It modifies no prior code.

============================================================================
Architecture
============================================================================
::

    FleetSnapshot (frozen, Phase 2)
        ▼  (composes the frozen Phase-3 engine, one run per posture / budget)
    ExecutiveDecisionEngine → ExecutiveDecisionPortfolio
        ▼
    StrategicPortfolioEngine
        ├── compare_strategies()      4 postures → StrategyComparison
        ├── budget_sweep()            curves over a budget range
        ├── pareto_frontier()         non-dominated (savings ↑, risk ↓)
        ├── capital_efficiency()      per-dollar analytics + efficiency index
        ├── recommend()               rule-based strategic recommendation
        └── confidence_assessment()   stability · agreement · coverage
        ▼
    StrategicPortfolioPlan (frozen, JSON-serialisable)

============================================================================
Architecture properties
============================================================================
* **Pure NumPy** — no pandas, SciPy, OR-tools, or ML libraries.
* **Frozen, validated dataclasses** for every config and result.
* **Registry-compatible**, mirroring every prior module.
* **JSON-serialisable** outputs (non-finite floats render as ``null``).
* **Deterministic** — every analysis is a deterministic function of the
  snapshot and parameters, with stable tie-breaking.
* **Failure-safe tracker integration**; no global mutable state.

Usage::

    from src.fleet.fleet_digital_twin import FleetDigitalTwinEngine, AssetInput
    from src.strategy.portfolio_strategy_engine import (
        StrategicPortfolioEngine, StrategicPortfolioConfig,
    )

    snapshot = FleetDigitalTwinEngine().build_fleet_snapshot(assets)
    engine = StrategicPortfolioEngine(StrategicPortfolioConfig(
        budget_start=100_000, budget_end=500_000, budget_step=100_000))
    plan = engine.optimize(snapshot, current_budget=200_000)
    print(plan.recommendation.executive_summary)

CLI::

    python src/strategy/portfolio_strategy_engine.py --demo
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
    OptimizationStrategy,
)
from src.fleet.fleet_digital_twin import FleetSnapshot  # noqa: E402

logger = logging.getLogger("portfolio_strategy_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named strategic portfolio engines.
STRATEGY_ENGINE_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "strategic_portfolio_engine"

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
# Strategic-posture enum
# ---------------------------------------------------------------------------


class StrategicPosture(str, Enum):
    """The four strategic maintenance-investment postures."""

    RISK_FIRST = "risk_first"
    ROI_FIRST = "roi_first"
    CRITICALITY_FIRST = "criticality_first"
    BALANCED = "balanced"


#: Maps each posture to the Executive Decision Engine configuration that
#: realises it (budget is supplied per call, so it is omitted here).
_POSTURE_CONFIG: Final[dict[str, dict[str, Any]]] = {
    StrategicPosture.RISK_FIRST.value: dict(
        strategy=OptimizationStrategy.HYBRID.value,
        weight_risk=0.70, weight_savings=0.10,
        weight_criticality=0.10, weight_urgency=0.10),
    StrategicPosture.ROI_FIRST.value: dict(
        strategy=OptimizationStrategy.GREEDY_ROI.value),
    StrategicPosture.CRITICALITY_FIRST.value: dict(
        strategy=OptimizationStrategy.HYBRID.value,
        weight_risk=0.10, weight_savings=0.10,
        weight_criticality=0.70, weight_urgency=0.10),
    StrategicPosture.BALANCED.value: dict(
        strategy=OptimizationStrategy.HYBRID.value),
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_strategic_portfolio_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a strategic portfolio engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = STRATEGY_ENGINE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Strategic portfolio engine '{name}' already registered to "
                f"{existing.__name__}"
            )
        STRATEGY_ENGINE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered strategic portfolio engine '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_strategic_portfolio_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered strategic portfolio engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in STRATEGY_ENGINE_REGISTRY:
        available = ", ".join(sorted(STRATEGY_ENGINE_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown strategic portfolio engine '{name}'. Available: {available}"
        )
    return STRATEGY_ENGINE_REGISTRY[name](**kwargs)


def list_strategic_portfolio_engines() -> list[str]:
    """Return the sorted names of registered strategic portfolio engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(STRATEGY_ENGINE_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategicPortfolioConfig:
    """Configuration for the :class:`StrategicPortfolioEngine`.

    Attributes:
        budget_start: First budget level of the sweep (inclusive, ``>= 0``).
        budget_end: Last budget level of the sweep (inclusive, ``>= start``).
        budget_step: Budget increment (``> 0``).
        sweep_posture: The strategic posture used for the budget sweep.
        roi_diminish_fraction: Fraction of peak ROI below which ROI is deemed to
            have "diminished" (in ``(0, 1]``).
        unlock_ratio: The risk-reduction-to-budget gain ratio above which a
            budget increase is recommended (a value of 1.5 means "recommend if a
            given % budget increase unlocks >= 1.5× that % in risk reduction").
        recovery_factor: Recovery factor forwarded to the executive engine.
        currency: Currency label for monetary figures.
    """

    budget_start:          float = 100_000.0
    budget_end:            float = 500_000.0
    budget_step:           float = 100_000.0
    sweep_posture:         str = StrategicPosture.BALANCED.value
    roi_diminish_fraction: float = 0.95
    unlock_ratio:          float = 1.5
    recovery_factor:       float = 0.70
    currency:              str = "USD"

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On an invalid budget range, posture, or parameter.
        """
        if self.budget_start < 0:
            raise ValueError("budget_start must be >= 0")
        if self.budget_end < self.budget_start:
            raise ValueError("budget_end must be >= budget_start")
        if self.budget_step <= 0:
            raise ValueError("budget_step must be > 0")
        valid = {p.value for p in StrategicPosture}
        if self.sweep_posture not in valid:
            raise ValueError(f"sweep_posture must be one of {sorted(valid)}")
        if not (0.0 < self.roi_diminish_fraction <= 1.0):
            raise ValueError("roi_diminish_fraction must be in (0, 1]")
        if self.unlock_ratio <= 0:
            raise ValueError("unlock_ratio must be > 0")
        if not (0.0 <= self.recovery_factor <= 1.0):
            raise ValueError("recovery_factor must be in [0, 1]")

    def budget_levels(self) -> tuple[float, ...]:
        """Return the budget levels of the sweep.

        Returns:
            The inclusive budget grid from start to end in steps of ``budget_step``.
        """
        levels = []
        b = self.budget_start
        # Use a count to avoid floating-point drift accumulating.
        n = int(math.floor((self.budget_end - self.budget_start)
                           / self.budget_step + _EPS)) + 1
        for i in range(n):
            levels.append(self.budget_start + i * self.budget_step)
        return tuple(levels)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    """The outcome of one strategic posture at a fixed budget.

    Attributes:
        posture: The strategic posture name.
        selected_asset_ids: The assets selected under this posture.
        total_cost: Total maintenance cost of the selection.
        expected_savings: Total expected savings of the selection.
        risk_reduction: Absolute reduction in mean fleet risk score.
        risk_reduction_pct: Percentage reduction in mean fleet risk.
        roi: Portfolio ROI.
        coverage: Fraction of the fleet selected (``selected / asset_count``).
    """

    posture:            str
    selected_asset_ids: tuple[str, ...]
    total_cost:         float
    expected_savings:   float
    risk_reduction:     float
    risk_reduction_pct: float
    roi:                float
    coverage:           float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "posture": self.posture,
            "selected_asset_ids": list(self.selected_asset_ids),
            "total_cost": _jsonsafe(self.total_cost),
            "expected_savings": _jsonsafe(self.expected_savings),
            "risk_reduction": _jsonsafe(self.risk_reduction),
            "risk_reduction_pct": _jsonsafe(self.risk_reduction_pct),
            "roi": _jsonsafe(self.roi),
            "coverage": _jsonsafe(self.coverage),
        }


@dataclass(frozen=True)
class StrategyComparison:
    """Comparison of the four strategic postures at a fixed budget.

    Attributes:
        budget: The budget at which the comparison was run.
        results: The per-posture results.
        best_by_savings: Posture with the highest expected savings.
        best_by_risk_reduction: Posture with the highest risk reduction.
        best_by_roi: Posture with the highest ROI.
        best_by_coverage: Posture with the highest coverage.
        agreement: Mean pairwise Jaccard overlap of the selected sets in
            ``[0, 1]`` (1 = all postures select the same assets).
        currency: Currency label.
    """

    budget:                 float
    results:                tuple[StrategyResult, ...]
    best_by_savings:        str
    best_by_risk_reduction: str
    best_by_roi:            str
    best_by_coverage:       str
    agreement:              float
    currency:               str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "budget": _jsonsafe(self.budget),
            "results": [r.to_dict() for r in self.results],
            "best_by_savings": self.best_by_savings,
            "best_by_risk_reduction": self.best_by_risk_reduction,
            "best_by_roi": self.best_by_roi,
            "best_by_coverage": self.best_by_coverage,
            "agreement": _jsonsafe(self.agreement),
            "currency": self.currency,
        }


@dataclass(frozen=True)
class BudgetSweepPoint:
    """A single budget level's portfolio outcome.

    Attributes:
        budget: The budget level.
        total_cost: Cost actually spent (``<= budget``).
        risk_reduction: Absolute reduction in mean fleet risk score.
        risk_reduction_pct: Percentage reduction in mean fleet risk.
        savings: Total expected savings.
        roi: Portfolio ROI.
        coverage: Fraction of the fleet selected.
        n_selected: Number of assets selected.
    """

    budget:             float
    total_cost:         float
    risk_reduction:     float
    risk_reduction_pct: float
    savings:            float
    roi:                float
    coverage:           float
    n_selected:         int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "budget": _jsonsafe(self.budget),
            "total_cost": _jsonsafe(self.total_cost),
            "risk_reduction": _jsonsafe(self.risk_reduction),
            "risk_reduction_pct": _jsonsafe(self.risk_reduction_pct),
            "savings": _jsonsafe(self.savings),
            "roi": _jsonsafe(self.roi),
            "coverage": _jsonsafe(self.coverage),
            "n_selected": self.n_selected,
        }


@dataclass(frozen=True)
class BudgetSweep:
    """The full budget-sweep analysis.

    Attributes:
        posture: The posture used for the sweep.
        points: The per-budget points.
        budgets: The budget grid (chart-ready).
        risk_reduction_curve: Risk reduction at each budget (chart-ready).
        savings_curve: Savings at each budget (chart-ready).
        roi_curve: ROI at each budget (chart-ready).
        coverage_curve: Coverage at each budget (chart-ready).
        knee_budget: The capital-efficiency knee budget (max curvature of the
            risk-reduction curve).
        roi_diminish_budget: The budget beyond which ROI falls below the
            configured fraction of its peak (``None`` if it never does).
        currency: Currency label.
    """

    posture:              str
    points:               tuple[BudgetSweepPoint, ...]
    budgets:              tuple[float, ...]
    risk_reduction_curve: tuple[float, ...]
    savings_curve:        tuple[float, ...]
    roi_curve:            tuple[float, ...]
    coverage_curve:       tuple[float, ...]
    knee_budget:          float
    roi_diminish_budget:  float | None
    currency:             str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "posture": self.posture,
            "points": [p.to_dict() for p in self.points],
            "budgets": [float(b) for b in self.budgets],
            "risk_reduction_curve": [float(v) for v in self.risk_reduction_curve],
            "savings_curve": [float(v) for v in self.savings_curve],
            "roi_curve": [float(v) for v in self.roi_curve],
            "coverage_curve": [float(v) for v in self.coverage_curve],
            "knee_budget": _jsonsafe(self.knee_budget),
            "roi_diminish_budget": (None if self.roi_diminish_budget is None
                                    else _jsonsafe(self.roi_diminish_budget)),
            "currency": self.currency,
        }


@dataclass(frozen=True)
class ParetoPortfolio:
    """A portfolio on (or considered for) the Pareto frontier.

    Attributes:
        portfolio_id: A stable identifier (e.g. ``"budget_300000"``).
        budget: The budget that produced this portfolio.
        risk: The residual (after-maintenance) mean fleet risk score.
        savings: The expected savings.
        roi: The portfolio ROI.
    """

    portfolio_id: str
    budget:       float
    risk:         float
    savings:      float
    roi:          float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "portfolio_id": self.portfolio_id,
            "budget": _jsonsafe(self.budget),
            "risk": _jsonsafe(self.risk),
            "savings": _jsonsafe(self.savings),
            "roi": _jsonsafe(self.roi),
        }


@dataclass(frozen=True)
class ParetoFrontier:
    """The non-dominated set of candidate portfolios.

    Attributes:
        portfolios: The non-dominated portfolios, ordered by budget.
        knee_portfolio_id: The id of the knee portfolio (best trade-off), or
            ``None`` when the frontier is empty.
        n_candidates: The number of candidate portfolios considered.
    """

    portfolios:        tuple[ParetoPortfolio, ...]
    knee_portfolio_id: str | None
    n_candidates:      int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "portfolios": [p.to_dict() for p in self.portfolios],
            "knee_portfolio_id": self.knee_portfolio_id,
            "n_candidates": self.n_candidates,
        }


@dataclass(frozen=True)
class CapitalEfficiency:
    """Per-dollar capital-efficiency analytics for one portfolio.

    Attributes:
        budget: The budget evaluated.
        total_cost: The cost actually spent.
        risk_reduction_per_dollar: Risk reduction per unit cost.
        savings_per_dollar: Savings per unit cost.
        coverage_per_dollar: Coverage per unit cost.
        maintenance_efficiency_index: A composite ``[0, 1]`` efficiency index
            blending the three per-dollar metrics against the sweep's best.
    """

    budget:                        float
    total_cost:                    float
    risk_reduction_per_dollar:     float
    savings_per_dollar:            float
    coverage_per_dollar:           float
    maintenance_efficiency_index:  float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "budget": _jsonsafe(self.budget),
            "total_cost": _jsonsafe(self.total_cost),
            "risk_reduction_per_dollar":
                _jsonsafe(self.risk_reduction_per_dollar),
            "savings_per_dollar": _jsonsafe(self.savings_per_dollar),
            "coverage_per_dollar": _jsonsafe(self.coverage_per_dollar),
            "maintenance_efficiency_index":
                _jsonsafe(self.maintenance_efficiency_index),
        }


@dataclass(frozen=True)
class StrategicRecommendation:
    """The rule-based strategic recommendation.

    Attributes:
        current_budget: The budget the recommendation is relative to (``None``
            if not supplied).
        recommended_budget: The recommended budget (typically the knee).
        knee_budget: The capital-efficiency knee budget.
        roi_diminish_budget: The budget beyond which ROI diminishes (``None`` if
            it never does within the sweep).
        before_knee: Whether the current budget lies before the knee.
        messages: The ordered, rule-based recommendation messages.
        executive_summary: A composed natural-language summary.
        currency: Currency label.
    """

    current_budget:      float | None
    recommended_budget:  float
    knee_budget:         float
    roi_diminish_budget: float | None
    before_knee:         bool
    messages:            tuple[str, ...]
    executive_summary:   str
    currency:            str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "current_budget": (None if self.current_budget is None
                               else _jsonsafe(self.current_budget)),
            "recommended_budget": _jsonsafe(self.recommended_budget),
            "knee_budget": _jsonsafe(self.knee_budget),
            "roi_diminish_budget": (None if self.roi_diminish_budget is None
                                    else _jsonsafe(self.roi_diminish_budget)),
            "before_knee": self.before_knee,
            "messages": list(self.messages),
            "executive_summary": self.executive_summary,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class StrategicConfidence:
    """The strategic confidence assessment.

    Attributes:
        portfolio_stability: Stability of the selection across adjacent budget
            levels (mean Jaccard overlap of consecutive sweep selections).
        strategy_agreement: Agreement across the four postures at the reference
            budget (mean pairwise Jaccard).
        coverage: Fleet coverage at the reference budget.
        score: The combined confidence in ``[0, 1]``.
    """

    portfolio_stability: float
    strategy_agreement:  float
    coverage:            float
    score:               float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "portfolio_stability": _jsonsafe(self.portfolio_stability),
            "strategy_agreement": _jsonsafe(self.strategy_agreement),
            "coverage": _jsonsafe(self.coverage),
            "score": _jsonsafe(self.score),
        }


@dataclass(frozen=True)
class StrategicPortfolioPlan:
    """The top-level strategic portfolio plan — the engine's output.

    Attributes:
        strategy_comparison: The four-posture comparison at the reference budget.
        budget_sweep: The budget-sweep analysis.
        pareto_frontier: The Pareto frontier over the sweep portfolios.
        capital_efficiency: Capital efficiency at the recommended budget.
        recommendation: The rule-based strategic recommendation.
        confidence: The confidence assessment.
        currency: Currency label.
    """

    strategy_comparison: StrategyComparison
    budget_sweep:        BudgetSweep
    pareto_frontier:     ParetoFrontier
    capital_efficiency:  CapitalEfficiency
    recommendation:      StrategicRecommendation
    confidence:          StrategicConfidence
    currency:            str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "strategy_comparison": self.strategy_comparison.to_dict(),
            "budget_sweep": self.budget_sweep.to_dict(),
            "pareto_frontier": self.pareto_frontier.to_dict(),
            "capital_efficiency": self.capital_efficiency.to_dict(),
            "recommendation": self.recommendation.to_dict(),
            "confidence": self.confidence.to_dict(),
            "currency": self.currency,
        }


# ---------------------------------------------------------------------------
# Strategic portfolio engine
# ---------------------------------------------------------------------------


@register_strategic_portfolio_engine(ENGINE_NAME)
class StrategicPortfolioEngine:
    """Strategic capital-allocation layer above the Executive Decision Engine.

    The engine composes the frozen Executive Decision Engine, running it across
    strategic postures and budget levels, then analyses the resulting portfolios
    into a strategy comparison, budget-sweep curves, a Pareto frontier, capital
    efficiency, a rule-based recommendation, and a confidence assessment.  It
    holds no per-call mutable state.

    Args:
        config: The strategic configuration.
        experiment_tracker: Optional tracker for logging plan summaries.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: StrategicPortfolioConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or StrategicPortfolioConfig()
        self.tracker = experiment_tracker
        self._n_runs = 0
        logger.info("StrategicPortfolioEngine ready | sweep=%s..%s step=%s",
                    self.config.budget_start, self.config.budget_end,
                    self.config.budget_step)

    # ------------------------------------------------------------------
    # Executive-engine composition
    # ------------------------------------------------------------------

    def _run_posture(
        self, snapshot: FleetSnapshot, posture: str, budget: float
    ) -> ExecutiveDecisionPortfolio:
        """Run the executive engine for one posture at one budget.

        Args:
            snapshot: The fleet snapshot.
            posture: The strategic posture.
            budget: The budget constraint.

        Returns:
            The resulting executive decision portfolio.
        """
        cfg_kwargs = dict(_POSTURE_CONFIG[posture])
        cfg_kwargs["budget"] = budget
        cfg_kwargs["recovery_factor"] = self.config.recovery_factor
        cfg_kwargs["currency"] = self.config.currency
        engine = ExecutiveDecisionEngine(ExecutiveDecisionConfig(**cfg_kwargs))
        return engine.recommend(snapshot)

    @staticmethod
    def _coverage(portfolio: ExecutiveDecisionPortfolio,
                  asset_count: int) -> float:
        """Return the fleet coverage of a portfolio.

        Args:
            portfolio: The executive decision portfolio.
            asset_count: The fleet size.

        Returns:
            ``selected / asset_count`` in ``[0, 1]``.
        """
        if asset_count <= 0:
            return 0.0
        return len(portfolio.selected_asset_ids) / asset_count

    # ------------------------------------------------------------------
    # 1. Strategy comparison
    # ------------------------------------------------------------------

    def compare_strategies(
        self, snapshot: FleetSnapshot, budget: float
    ) -> StrategyComparison:
        """Compare the four strategic postures at a fixed budget.

        Args:
            snapshot: The fleet snapshot.
            budget: The budget constraint.

        Returns:
            A :class:`StrategyComparison`.

        Raises:
            ValueError: When the snapshot has no assets or the budget is
                negative.
        """
        self._require_assets(snapshot)
        if budget < 0:
            raise ValueError("budget must be >= 0")

        results = []
        selected_sets: dict[str, set[str]] = {}
        for posture in (p.value for p in StrategicPosture):
            p = self._run_posture(snapshot, posture, budget)
            cov = self._coverage(p, snapshot.asset_count)
            results.append(StrategyResult(
                posture=posture,
                selected_asset_ids=tuple(p.selected_asset_ids),
                total_cost=float(p.total_maintenance_cost),
                expected_savings=float(p.expected_savings),
                risk_reduction=float(p.expected_risk_reduction),
                risk_reduction_pct=float(p.portfolio_risk_reduction_pct),
                roi=float(p.total_roi),
                coverage=float(cov),
            ))
            selected_sets[posture] = set(p.selected_asset_ids)

        results_t = tuple(results)
        best_savings = max(results_t, key=lambda r: (r.expected_savings, r.posture)).posture
        best_risk = max(results_t, key=lambda r: (r.risk_reduction, r.posture)).posture
        best_roi = max(results_t, key=lambda r: (r.roi, r.posture)).posture
        best_cov = max(results_t, key=lambda r: (r.coverage, r.posture)).posture
        agreement = self._mean_jaccard(list(selected_sets.values()))

        return StrategyComparison(
            budget=float(budget),
            results=results_t,
            best_by_savings=best_savings,
            best_by_risk_reduction=best_risk,
            best_by_roi=best_roi,
            best_by_coverage=best_cov,
            agreement=agreement,
            currency=self.config.currency,
        )

    # ------------------------------------------------------------------
    # 2. Budget sweep
    # ------------------------------------------------------------------

    def budget_sweep(
        self, snapshot: FleetSnapshot, *, posture: str | None = None
    ) -> BudgetSweep:
        """Sweep the budget range and build the response curves.

        Args:
            snapshot: The fleet snapshot.
            posture: The posture to sweep (defaults to the configured posture).

        Returns:
            A :class:`BudgetSweep`.

        Raises:
            ValueError: When the snapshot has no assets or the posture is invalid.
        """
        self._require_assets(snapshot)
        posture = posture or self.config.sweep_posture
        if posture not in {p.value for p in StrategicPosture}:
            raise ValueError(f"invalid posture '{posture}'")

        levels = self.config.budget_levels()
        points = []
        selected_sets = []
        for budget in levels:
            p = self._run_posture(snapshot, posture, budget)
            cov = self._coverage(p, snapshot.asset_count)
            points.append(BudgetSweepPoint(
                budget=float(budget),
                total_cost=float(p.total_maintenance_cost),
                risk_reduction=float(p.expected_risk_reduction),
                risk_reduction_pct=float(p.portfolio_risk_reduction_pct),
                savings=float(p.expected_savings),
                roi=float(p.total_roi),
                coverage=float(cov),
                n_selected=len(p.selected_asset_ids),
            ))
            selected_sets.append(set(p.selected_asset_ids))

        budgets = np.array(levels, dtype=float)
        rr = np.array([pt.risk_reduction for pt in points], dtype=float)
        sv = np.array([pt.savings for pt in points], dtype=float)
        roi = np.array([pt.roi for pt in points], dtype=float)
        cov = np.array([pt.coverage for pt in points], dtype=float)

        knee_idx = self._knee_index(budgets, rr)
        knee_budget = float(budgets[knee_idx])
        roi_diminish = self._roi_diminish_budget(budgets, roi)

        # Stash the consecutive-overlap stability for the confidence step.
        self._last_sweep_stability = self._consecutive_stability(selected_sets)

        return BudgetSweep(
            posture=posture,
            points=tuple(points),
            budgets=tuple(float(b) for b in budgets),
            risk_reduction_curve=tuple(float(v) for v in rr),
            savings_curve=tuple(float(v) for v in sv),
            roi_curve=tuple(float(v) for v in roi),
            coverage_curve=tuple(float(v) for v in cov),
            knee_budget=knee_budget,
            roi_diminish_budget=roi_diminish,
            currency=self.config.currency,
        )

    # ------------------------------------------------------------------
    # 3. Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(self, sweep: BudgetSweep) -> ParetoFrontier:
        """Compute the non-dominated frontier over the sweep's portfolios.

        A portfolio dominates another when it has **higher savings AND lower
        residual risk** (with at least one strict).  Among portfolios that tie
        on (savings, risk), the lower-budget one is kept.

        Args:
            sweep: The budget sweep.

        Returns:
            A :class:`ParetoFrontier`.
        """
        candidates = [
            ParetoPortfolio(
                portfolio_id=f"budget_{int(pt.budget)}",
                budget=pt.budget,
                # residual risk = risk_before - risk_reduction; rank uses
                # negative risk reduction so "lower risk" == "more reduction".
                risk=-pt.risk_reduction,
                savings=pt.savings,
                roi=pt.roi,
            )
            for pt in sweep.points
        ]
        non_dominated = self._non_dominated(candidates)
        # Order by budget for stable presentation.
        non_dominated = tuple(sorted(non_dominated, key=lambda p: p.budget))

        knee_id = None
        if non_dominated:
            budgets = np.array([p.budget for p in non_dominated], dtype=float)
            savings = np.array([p.savings for p in non_dominated], dtype=float)
            knee_idx = self._knee_index(budgets, savings)
            knee_id = non_dominated[knee_idx].portfolio_id

        return ParetoFrontier(
            portfolios=non_dominated,
            knee_portfolio_id=knee_id,
            n_candidates=len(candidates),
        )

    @staticmethod
    def _non_dominated(
        candidates: Sequence[ParetoPortfolio]
    ) -> list[ParetoPortfolio]:
        """Return the non-dominated subset (higher savings AND lower risk).

        Args:
            candidates: Candidate portfolios.

        Returns:
            The non-dominated portfolios.
        """
        def dominates(a: ParetoPortfolio, b: ParetoPortfolio) -> bool:
            # a dominates b: savings >= and risk <= with at least one strict.
            return (a.savings >= b.savings and a.risk <= b.risk
                    and (a.savings > b.savings or a.risk < b.risk))

        out: list[ParetoPortfolio] = []
        for c in candidates:
            if any(dominates(o, c) for o in candidates if o is not c):
                continue
            # Drop later duplicates that tie another already-kept point.
            if any(o.savings == c.savings and o.risk == c.risk
                   and o.budget <= c.budget and o is not c for o in out):
                continue
            out.append(c)
        return out

    # ------------------------------------------------------------------
    # 4. Capital efficiency
    # ------------------------------------------------------------------

    def capital_efficiency(
        self, sweep: BudgetSweep, budget: float
    ) -> CapitalEfficiency:
        """Compute per-dollar capital efficiency at a budget level.

        Args:
            sweep: The budget sweep (used to normalise the efficiency index).
            budget: The budget level to evaluate (matched to the nearest sweep
                point).

        Returns:
            A :class:`CapitalEfficiency`.

        Raises:
            ValueError: When the sweep has no points.
        """
        if not sweep.points:
            raise ValueError("capital_efficiency requires a non-empty sweep")
        # Match to the nearest sweep point by budget.
        pt = min(sweep.points, key=lambda p: abs(p.budget - budget))
        cost = pt.total_cost
        if cost <= _EPS:
            return CapitalEfficiency(
                budget=float(pt.budget), total_cost=0.0,
                risk_reduction_per_dollar=0.0, savings_per_dollar=0.0,
                coverage_per_dollar=0.0, maintenance_efficiency_index=0.0)

        rr_pd = pt.risk_reduction / cost
        sv_pd = pt.savings / cost
        cov_pd = pt.coverage / cost

        # Efficiency index: each per-dollar metric scaled by the sweep's best,
        # then averaged -> [0, 1] with 1 = best-in-sweep on all three.
        best_rr = max((p.risk_reduction / p.total_cost
                       for p in sweep.points if p.total_cost > _EPS),
                      default=0.0)
        best_sv = max((p.savings / p.total_cost
                       for p in sweep.points if p.total_cost > _EPS),
                      default=0.0)
        best_cov = max((p.coverage / p.total_cost
                        for p in sweep.points if p.total_cost > _EPS),
                       default=0.0)
        idx_parts = []
        if best_rr > _EPS:
            idx_parts.append(rr_pd / best_rr)
        if best_sv > _EPS:
            idx_parts.append(sv_pd / best_sv)
        if best_cov > _EPS:
            idx_parts.append(cov_pd / best_cov)
        mei = float(np.mean(idx_parts)) if idx_parts else 0.0

        return CapitalEfficiency(
            budget=float(pt.budget),
            total_cost=float(cost),
            risk_reduction_per_dollar=float(rr_pd),
            savings_per_dollar=float(sv_pd),
            coverage_per_dollar=float(cov_pd),
            maintenance_efficiency_index=float(np.clip(mei, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # 5. Strategic recommendation
    # ------------------------------------------------------------------

    def recommend(
        self, sweep: BudgetSweep, *, current_budget: float | None = None
    ) -> StrategicRecommendation:
        """Generate the rule-based strategic recommendation.

        Args:
            sweep: The budget sweep.
            current_budget: The operator's current budget (optional).

        Returns:
            A :class:`StrategicRecommendation`.
        """
        cur = self.config.currency
        knee = sweep.knee_budget
        roi_dim = sweep.roi_diminish_budget
        recommended = knee
        messages: list[str] = []

        before_knee = (current_budget is not None
                       and current_budget < knee - _EPS)

        # Rule 1: budget relative to the knee.
        if current_budget is not None:
            if before_knee:
                messages.append(
                    f"Current budget ({cur} {current_budget:,.0f}) lies before "
                    f"the Pareto knee point ({cur} {knee:,.0f}); increasing it "
                    "yields strong marginal risk reduction.")
            elif current_budget > knee + _EPS:
                messages.append(
                    f"Current budget ({cur} {current_budget:,.0f}) lies beyond "
                    f"the knee ({cur} {knee:,.0f}); marginal returns are "
                    "diminishing.")
            else:
                messages.append(
                    f"Current budget is at the capital-efficiency knee "
                    f"({cur} {knee:,.0f}).")

        # Rule 2: unlock opportunity — % budget increase vs % risk-reduction gain.
        unlock = self._unlock_opportunity(sweep, current_budget)
        if unlock is not None:
            budget_pct, rr_pct = unlock
            if rr_pct >= self.config.unlock_ratio * budget_pct:
                messages.append(
                    f"Increase budget by {budget_pct:.0f}% to unlock "
                    f"{rr_pct:.0f}% additional risk reduction.")

        # Rule 3: ROI diminishing.
        if roi_dim is not None:
            messages.append(
                f"ROI diminishes beyond {cur} {roi_dim:,.0f}; spend past this "
                "point returns less per dollar.")

        if not messages:
            messages.append(
                f"The recommended budget is the capital-efficiency knee at "
                f"{cur} {knee:,.0f}.")

        summary = self._recommendation_summary(
            knee, recommended, current_budget, before_knee, roi_dim)

        return StrategicRecommendation(
            current_budget=(None if current_budget is None
                            else float(current_budget)),
            recommended_budget=float(recommended),
            knee_budget=float(knee),
            roi_diminish_budget=roi_dim,
            before_knee=bool(before_knee),
            messages=tuple(messages),
            executive_summary=summary,
            currency=cur,
        )

    def _unlock_opportunity(
        self, sweep: BudgetSweep, current_budget: float | None
    ) -> tuple[float, float] | None:
        """Compute the budget-increase / risk-reduction-gain opportunity.

        Compares the current budget level to the knee budget: the percentage
        budget increase to reach the knee, and the percentage of additional
        risk reduction that increase unlocks.

        Args:
            sweep: The budget sweep.
            current_budget: The current budget.

        Returns:
            ``(budget_increase_pct, risk_reduction_gain_pct)`` or ``None``.
        """
        if current_budget is None or current_budget <= _EPS:
            return None
        budgets = np.array([p.budget for p in sweep.points], dtype=float)
        rr = np.array([p.risk_reduction for p in sweep.points], dtype=float)
        # Nearest points for current and knee budgets.
        cur_idx = int(np.argmin(np.abs(budgets - current_budget)))
        knee_idx = int(np.argmin(np.abs(budgets - sweep.knee_budget)))
        if knee_idx <= cur_idx:
            return None
        cur_b = budgets[cur_idx]
        knee_b = budgets[knee_idx]
        cur_rr = rr[cur_idx]
        knee_rr = rr[knee_idx]
        if cur_b <= _EPS or cur_rr <= _EPS:
            return None
        budget_pct = 100.0 * (knee_b - cur_b) / cur_b
        rr_pct = 100.0 * (knee_rr - cur_rr) / cur_rr
        if budget_pct <= 0 or rr_pct <= 0:
            return None
        return float(budget_pct), float(rr_pct)

    def _recommendation_summary(
        self, knee: float, recommended: float, current_budget: float | None,
        before_knee: bool, roi_dim: float | None,
    ) -> str:
        """Compose the recommendation's natural-language summary.

        Args:
            knee: The knee budget.
            recommended: The recommended budget.
            current_budget: The current budget.
            before_knee: Whether the current budget is before the knee.
            roi_dim: The ROI-diminish budget.

        Returns:
            The summary string.
        """
        cur = self.config.currency
        parts = [
            f"The capital-efficiency knee is at {cur} {knee:,.0f}, the "
            "recommended maintenance budget."
        ]
        if current_budget is not None:
            if before_knee:
                parts.append(
                    f"The current budget of {cur} {current_budget:,.0f} is "
                    "below this; an increase is advised.")
            else:
                parts.append(
                    f"The current budget of {cur} {current_budget:,.0f} meets "
                    "or exceeds the knee.")
        if roi_dim is not None:
            parts.append(f"Returns flatten beyond {cur} {roi_dim:,.0f}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # 6. Confidence assessment
    # ------------------------------------------------------------------

    def confidence_assessment(
        self, comparison: StrategyComparison, sweep: BudgetSweep,
        reference_budget: float,
    ) -> StrategicConfidence:
        """Combine stability, strategy agreement, and coverage into confidence.

        Args:
            comparison: The four-posture comparison at the reference budget.
            sweep: The budget sweep (for selection stability).
            reference_budget: The budget at which coverage is read.

        Returns:
            A :class:`StrategicConfidence`.
        """
        stability = getattr(self, "_last_sweep_stability", None)
        if stability is None:
            stability = 1.0
        agreement = comparison.agreement
        # Coverage at the reference budget (nearest sweep point).
        if sweep.points:
            pt = min(sweep.points, key=lambda p: abs(p.budget - reference_budget))
            coverage = pt.coverage
        else:
            coverage = 0.0
        score = 0.40 * stability + 0.40 * agreement + 0.20 * coverage
        return StrategicConfidence(
            portfolio_stability=float(stability),
            strategy_agreement=float(agreement),
            coverage=float(coverage),
            score=float(np.clip(score, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def optimize(
        self, snapshot: FleetSnapshot, *, current_budget: float | None = None
    ) -> StrategicPortfolioPlan:
        """Run the full strategic analysis and assemble the plan.

        Args:
            snapshot: The fleet snapshot.
            current_budget: The operator's current budget (optional). Defaults to
                the sweep midpoint for the comparison reference.

        Returns:
            A :class:`StrategicPortfolioPlan`.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        self._require_assets(snapshot)
        sweep = self.budget_sweep(snapshot)
        reference_budget = (current_budget if current_budget is not None
                            else sweep.knee_budget)
        comparison = self.compare_strategies(snapshot, reference_budget)
        frontier = self.pareto_frontier(sweep)
        efficiency = self.capital_efficiency(sweep, sweep.knee_budget)
        recommendation = self.recommend(sweep, current_budget=current_budget)
        confidence = self.confidence_assessment(comparison, sweep,
                                                reference_budget)

        plan = StrategicPortfolioPlan(
            strategy_comparison=comparison,
            budget_sweep=sweep,
            pareto_frontier=frontier,
            capital_efficiency=efficiency,
            recommendation=recommendation,
            confidence=confidence,
            currency=self.config.currency,
        )
        self._log_plan(plan)
        self._n_runs += 1
        return plan

    # ------------------------------------------------------------------
    # Numerical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _knee_index(x: "np.ndarray", y: "np.ndarray") -> int:
        """Return the index of the knee (max curvature) of a curve.

        Uses the Kneedle perpendicular-distance heuristic on the normalised
        curve.  Degenerate inputs (fewer than three points, or zero range)
        return the last index.

        Args:
            x: The x values (e.g. budgets).
            y: The y values (e.g. risk reduction).

        Returns:
            The knee index.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 3:
            return int(x.size - 1)
        xr = x.max() - x.min()
        yr = y.max() - y.min()
        if xr <= _EPS or yr <= _EPS:
            return int(x.size - 1)
        xn = (x - x.min()) / xr
        yn = (y - y.min()) / yr
        x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
        num = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0)
        den = math.hypot(y1 - y0, x1 - x0)
        if den <= _EPS:
            return int(x.size - 1)
        return int(np.argmax(num / den))

    def _roi_diminish_budget(
        self, budgets: "np.ndarray", roi: "np.ndarray"
    ) -> float | None:
        """Return the budget beyond which ROI falls below the configured peak fraction.

        Args:
            budgets: The budget grid.
            roi: The ROI at each budget.

        Returns:
            The first budget at which ROI drops below ``roi_diminish_fraction``
            of its peak, or ``None`` if it never does.
        """
        if roi.size == 0:
            return None
        peak = float(np.max(roi))
        if peak <= _EPS:
            return None
        threshold = self.config.roi_diminish_fraction * peak
        below = np.where(roi < threshold - _EPS)[0]
        if below.size == 0:
            return None
        return float(budgets[below[0]])

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Return the Jaccard overlap of two sets.

        Args:
            a: First set.
            b: Second set.

        Returns:
            ``|a ∩ b| / |a ∪ b|``; 1.0 when both are empty.
        """
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 1.0
        return len(a & b) / len(union)

    def _mean_jaccard(self, sets: Sequence[set[str]]) -> float:
        """Return the mean pairwise Jaccard overlap of a collection of sets.

        Args:
            sets: The sets.

        Returns:
            The mean pairwise overlap; 1.0 for fewer than two sets.
        """
        if len(sets) < 2:
            return 1.0
        vals = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                vals.append(self._jaccard(sets[i], sets[j]))
        return float(np.mean(vals)) if vals else 1.0

    def _consecutive_stability(self, sets: Sequence[set[str]]) -> float:
        """Return the mean Jaccard overlap of consecutive selections.

        Args:
            sets: The per-budget selected sets, in budget order.

        Returns:
            Mean overlap of consecutive pairs; 1.0 for fewer than two.
        """
        if len(sets) < 2:
            return 1.0
        vals = [self._jaccard(sets[i], sets[i + 1])
                for i in range(len(sets) - 1)]
        return float(np.mean(vals)) if vals else 1.0

    @staticmethod
    def _require_assets(snapshot: FleetSnapshot) -> None:
        """Validate that the snapshot has at least one asset.

        Args:
            snapshot: The fleet snapshot.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        if snapshot.asset_count == 0 or not snapshot.assets:
            raise ValueError("strategic analysis requires a non-empty snapshot")

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_plan(self, plan: StrategicPortfolioPlan) -> None:
        """Log a plan summary to the experiment tracker (failure-safe).

        Args:
            plan: The plan to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "strat_knee_budget": float(plan.budget_sweep.knee_budget),
                    "strat_recommended_budget":
                        float(plan.recommendation.recommended_budget),
                    "strat_confidence": float(plan.confidence.score),
                    "strat_agreement": float(plan.strategy_comparison.agreement),
                    "strat_efficiency_index":
                        float(plan.capital_efficiency.maintenance_efficiency_index),
                    "strat_pareto_size": float(len(plan.pareto_frontier.portfolios)),
                },
                step=self._n_runs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"strat_sweep_posture": plan.budget_sweep.posture})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short strategic-portfolio demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    from src.fleet.fleet_digital_twin import (
        AssetInput,
        FleetDigitalTwinConfig,
        FleetDigitalTwinEngine,
    )

    rng = np.random.default_rng(13)
    assets = []
    for i in range(40):
        rate = 0.3 + 0.07 * (i % 12)
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.4, 45), 0, 100)
        assets.append(AssetInput(asset_id=f"WTG-{i:03d}", asset_type="wind_turbine",
                                 location=["North", "Baltic", "Celtic"][i % 3],
                                 health_trajectory=traj))
    snapshot = FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)

    engine = StrategicPortfolioEngine(StrategicPortfolioConfig(
        budget_start=20_000, budget_end=200_000, budget_step=20_000))
    plan = engine.optimize(snapshot, current_budget=60_000)

    print("STRATEGY COMPARISON @ knee budget:")
    for r in plan.strategy_comparison.results:
        print(f"  {r.posture:18s}: cost={plan.currency} {r.total_cost:>9,.0f} "
              f"savings={plan.currency} {r.expected_savings:>10,.0f} "
              f"risk_cut={r.risk_reduction_pct:5.1f}% coverage={r.coverage:.0%}")
    print(f"  strategy agreement: {plan.strategy_comparison.agreement:.2f}")
    print()
    print("BUDGET SWEEP:")
    for pt in plan.budget_sweep.points:
        print(f"  {plan.currency} {pt.budget:>9,.0f}: risk_cut={pt.risk_reduction_pct:5.1f}% "
              f"savings={plan.currency} {pt.savings:>11,.0f} roi={pt.roi:5.1f} "
              f"coverage={pt.coverage:.0%}")
    print(f"  knee budget: {plan.currency} {plan.budget_sweep.knee_budget:,.0f}")
    print()
    print("PARETO FRONTIER:", [p.portfolio_id for p in plan.pareto_frontier.portfolios])
    print(f"  knee portfolio: {plan.pareto_frontier.knee_portfolio_id}")
    print()
    print(f"CAPITAL EFFICIENCY @ knee: index={plan.capital_efficiency.maintenance_efficiency_index:.2f} "
          f"savings/$={plan.capital_efficiency.savings_per_dollar:.2f}")
    print()
    print("RECOMMENDATION:")
    print(" ", plan.recommendation.executive_summary)
    for m in plan.recommendation.messages:
        print(f"  - {m}")
    print()
    print(f"CONFIDENCE: {plan.confidence.score:.2f} "
          f"(stability={plan.confidence.portfolio_stability:.2f} "
          f"agreement={plan.confidence.strategy_agreement:.2f} "
          f"coverage={plan.confidence.coverage:.2f})")
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
    parser = argparse.ArgumentParser(description="Strategic portfolio engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a strategic-portfolio demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered strategic portfolio engines:",
              list_strategic_portfolio_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())