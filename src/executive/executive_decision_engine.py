#!/usr/bin/env python3
"""Executive decision engine — fleet analytics to budget-constrained decisions.

Week-7 Phase-3 is the platform's executive decision-intelligence layer.  The
Phase-2 fleet digital twin answers "what is the state and risk of our fleet?";
this engine answers the questions leadership actually acts on:

    1. Which assets should we repair first?
    2. Which maintenance actions maximise ROI?
    3. Given a limited budget, what is the optimal maintenance portfolio?
    4. What risk reduction and financial impact should we expect?
    5. What should executives approve this month?

It consumes a :class:`~src.fleet.fleet_digital_twin.FleetSnapshot` exactly as
delivered by Phase 2 — reusing, not modifying, the frozen fleet engine — and
produces an :class:`ExecutiveDecisionPortfolio`: a budget-constrained selection
of assets with a prioritised ranking, ROI and risk-reduction analytics, a
natural-language executive recommendation, and a confidence score.

============================================================================
Architecture
============================================================================
::

    FleetSnapshot (frozen, from Phase 2)
        ▼
    ExecutiveDecisionEngine
        ├── prioritise_assets()        priority_score over each asset
        ├── optimize_portfolio()       budget-constrained selection
        │     ├── greedy ROI strategy
        │     ├── greedy savings strategy
        │     └── hybrid (priority)  strategy
        ├── risk-reduction analytics   before / after / reduction %
        ├── ROI analytics              ROI / payback ratio / cost efficiency
        ├── executive recommendation   natural-language summary
        └── confidence score           data quality · coverage · concentration
        ▼
    ExecutiveDecisionPortfolio (frozen, JSON-serialisable)

============================================================================
Architecture properties
============================================================================
* **Pure NumPy** — no SciPy, no PyTorch.
* **Frozen, validated dataclasses** for the config, prioritised asset, and
  portfolio.
* **Registry-compatible**, mirroring every prior module.
* **JSON-serialisable** outputs (non-finite floats render as ``null``).
* **Deterministic** — the same snapshot, budget, and strategy always yield the
  same portfolio, with stable tie-breaking by ``asset_id``.
* **Failure-safe tracker integration**; no global mutable state.

Usage::

    from src.fleet.fleet_digital_twin import FleetDigitalTwinEngine, AssetInput
    from src.executive.executive_decision_engine import (
        ExecutiveDecisionEngine, ExecutiveDecisionConfig, OptimizationStrategy,
    )

    snapshot = FleetDigitalTwinEngine().build_fleet_snapshot(assets)
    engine = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=25_000))
    portfolio = engine.recommend(snapshot)
    print(portfolio.executive_summary)

CLI::

    python src/executive/executive_decision_engine.py --demo
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

from src.fleet.fleet_digital_twin import (  # noqa: E402
    FleetAsset,
    FleetSnapshot,
)

logger = logging.getLogger("executive_decision_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named executive decision engines.
EXECUTIVE_DECISION_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "executive_decision_engine"

#: Severity normalisation cap (matches the fleet engine's severity scale).
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
# Optimization-strategy enum
# ---------------------------------------------------------------------------


class OptimizationStrategy(str, Enum):
    """Budget-constrained portfolio-selection strategies."""

    GREEDY_ROI = "greedy_roi"
    GREEDY_SAVINGS = "greedy_savings"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_executive_decision_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering an executive decision engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = EXECUTIVE_DECISION_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Executive decision engine '{name}' already registered to "
                f"{existing.__name__}"
            )
        EXECUTIVE_DECISION_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered executive decision engine '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_executive_decision_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered executive decision engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in EXECUTIVE_DECISION_REGISTRY:
        available = ", ".join(sorted(EXECUTIVE_DECISION_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown executive decision engine '{name}'. Available: {available}"
        )
    return EXECUTIVE_DECISION_REGISTRY[name](**kwargs)


def list_executive_decision_engines() -> list[str]:
    """Return the sorted names of registered executive decision engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(EXECUTIVE_DECISION_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutiveDecisionConfig:
    """Configuration for the :class:`ExecutiveDecisionEngine`.

    Attributes:
        budget: Maintenance budget constraint; ``inf`` means unconstrained.
        strategy: The portfolio-optimization strategy.
        weight_risk: Priority-score weight on risk.
        weight_savings: Priority-score weight on normalised expected savings.
        weight_criticality: Priority-score weight on criticality (severity).
        weight_urgency: Priority-score weight on urgency (failure probability).
        recovery_factor: Fraction of an asset's risk removed by repairing it
            (used in the risk-after estimate); in ``[0, 1]``.
        top_n: Number of assets in the top-risk / top-opportunity lists.
        currency: Currency label for monetary figures.
    """

    budget:             float = float("inf")
    strategy:           str = OptimizationStrategy.HYBRID.value
    weight_risk:        float = 0.40
    weight_savings:     float = 0.30
    weight_criticality: float = 0.20
    weight_urgency:     float = 0.10
    recovery_factor:    float = 0.70
    top_n:              int = 5
    currency:           str = "USD"

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid budget, strategy, weights, or parameters.
        """
        if self.budget < 0:
            raise ValueError("budget must be >= 0 (use inf for unconstrained)")
        valid = {s.value for s in OptimizationStrategy}
        if self.strategy not in valid:
            raise ValueError(f"strategy must be one of {sorted(valid)}")
        weights = (self.weight_risk, self.weight_savings,
                   self.weight_criticality, self.weight_urgency)
        if any(w < 0 for w in weights):
            raise ValueError("priority weights must be >= 0")
        if sum(weights) <= 0:
            raise ValueError("priority weights must sum to > 0")
        if not (0.0 <= self.recovery_factor <= 1.0):
            raise ValueError("recovery_factor must be in [0, 1]")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

    @property
    def weight_sum(self) -> float:
        """Return the sum of the four priority weights."""
        return (self.weight_risk + self.weight_savings
                + self.weight_criticality + self.weight_urgency)


# ---------------------------------------------------------------------------
# Prioritised asset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrioritizedAsset:
    """A fleet asset annotated with its decision-layer priority and economics.

    Attributes:
        asset_id: Unique asset identifier.
        location: Physical location / site.
        risk_score: The fleet risk score in ``[0, 1]``.
        maintenance_cost: The recommended maintenance cost.
        expected_savings: The expected savings of proactive maintenance.
        roi: Return on investment (``expected_savings / maintenance_cost``).
        priority_score: The composite priority score (higher = repair sooner).
        rank: 1-based priority rank (1 = highest priority).
        selected: Whether this asset is in the budget-constrained portfolio.
    """

    asset_id:         str
    location:         str
    risk_score:       float
    maintenance_cost: float
    expected_savings: float
    roi:              float
    priority_score:   float
    rank:             int
    selected:         bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "asset_id": self.asset_id,
            "location": self.location,
            "risk_score": _jsonsafe(self.risk_score),
            "maintenance_cost": _jsonsafe(self.maintenance_cost),
            "expected_savings": _jsonsafe(self.expected_savings),
            "roi": _jsonsafe(self.roi),
            "priority_score": _jsonsafe(self.priority_score),
            "rank": self.rank,
            "selected": self.selected,
        }


# ---------------------------------------------------------------------------
# Executive decision portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutiveDecisionPortfolio:
    """Immutable executive decision portfolio — the engine's output.

    Attributes:
        strategy: The optimization strategy used.
        budget: The budget constraint applied.
        selected_asset_ids: The ids of the assets selected for maintenance.
        total_maintenance_cost: Total cost of the selected portfolio.
        expected_savings: Total expected savings of the selected portfolio.
        total_roi: Portfolio ROI (savings / cost).
        payback_ratio: Savings-to-cost ratio (alias of ROI + 1 interpretation).
        cost_efficiency: Savings per unit cost.
        budget_utilization: Fraction of the budget consumed (``0`` when
            unconstrained).
        expected_risk_reduction: Absolute reduction in mean fleet risk score.
        portfolio_risk_reduction_pct: Percentage reduction in mean fleet risk.
        average_risk_before: Mean fleet risk score before maintenance.
        average_risk_after: Mean fleet risk score after the selected maintenance.
        prioritization: The full prioritised asset ranking.
        top_risks: Ids of the highest-risk assets.
        top_opportunities: Ids of the highest-savings assets.
        immediate_action_assets: Ids of selected assets requiring urgent action.
        recommendations: Structured recommendation lines.
        executive_summary: Natural-language management summary.
        confidence_score: Overall decision confidence in ``[0, 1]``.
        currency: Currency label.
    """

    strategy:                      str
    budget:                        float
    selected_asset_ids:            tuple[str, ...]
    total_maintenance_cost:        float
    expected_savings:              float
    total_roi:                     float
    payback_ratio:                 float
    cost_efficiency:               float
    budget_utilization:            float
    expected_risk_reduction:       float
    portfolio_risk_reduction_pct:  float
    average_risk_before:           float
    average_risk_after:            float
    prioritization:                tuple[PrioritizedAsset, ...]
    top_risks:                     tuple[str, ...]
    top_opportunities:             tuple[str, ...]
    immediate_action_assets:       tuple[str, ...]
    recommendations:               tuple[str, ...]
    executive_summary:             str
    confidence_score:              float
    currency:                      str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; the prioritisation is nested.
        """
        return {
            "strategy": self.strategy,
            "budget": _jsonsafe(self.budget),
            "selected_asset_ids": list(self.selected_asset_ids),
            "total_maintenance_cost": _jsonsafe(self.total_maintenance_cost),
            "expected_savings": _jsonsafe(self.expected_savings),
            "total_roi": _jsonsafe(self.total_roi),
            "payback_ratio": _jsonsafe(self.payback_ratio),
            "cost_efficiency": _jsonsafe(self.cost_efficiency),
            "budget_utilization": _jsonsafe(self.budget_utilization),
            "expected_risk_reduction": _jsonsafe(self.expected_risk_reduction),
            "portfolio_risk_reduction_pct":
                _jsonsafe(self.portfolio_risk_reduction_pct),
            "average_risk_before": _jsonsafe(self.average_risk_before),
            "average_risk_after": _jsonsafe(self.average_risk_after),
            "prioritization": [p.to_dict() for p in self.prioritization],
            "top_risks": list(self.top_risks),
            "top_opportunities": list(self.top_opportunities),
            "immediate_action_assets": list(self.immediate_action_assets),
            "recommendations": list(self.recommendations),
            "executive_summary": self.executive_summary,
            "confidence_score": _jsonsafe(self.confidence_score),
            "currency": self.currency,
        }


# ---------------------------------------------------------------------------
# Executive decision engine
# ---------------------------------------------------------------------------


@register_executive_decision_engine(ENGINE_NAME)
class ExecutiveDecisionEngine:
    """Transforms a fleet snapshot into a budget-constrained decision portfolio.

    The engine prioritises assets by a composite score, selects a maintenance
    portfolio under a budget constraint using the configured strategy, computes
    ROI and risk-reduction analytics, and produces a natural-language executive
    recommendation with a confidence score.  It holds no per-call mutable state.

    Args:
        config: The decision configuration.
        experiment_tracker: Optional tracker for logging portfolio summaries.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: ExecutiveDecisionConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or ExecutiveDecisionConfig()
        self.tracker = experiment_tracker
        self._n_runs = 0
        logger.info("ExecutiveDecisionEngine ready | strategy=%s | budget=%s",
                    self.config.strategy, self.config.budget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, snapshot: FleetSnapshot) -> ExecutiveDecisionPortfolio:
        """Produce an executive decision portfolio for a fleet snapshot.

        Args:
            snapshot: The fleet snapshot to act on.

        Returns:
            A populated :class:`ExecutiveDecisionPortfolio`.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        if snapshot.asset_count == 0 or not snapshot.assets:
            raise ValueError("recommend requires a non-empty fleet snapshot")

        assets = list(snapshot.assets)
        scored = self._priority_table(assets)
        selected = self._optimize(scored)
        selected_ids = {a.asset_id for a in selected}

        prioritization = self._build_prioritization(scored, selected_ids)

        total_cost = float(sum(a.maintenance_cost for a in selected))
        total_savings = float(sum(a.expected_savings for a in selected))
        roi, payback, cost_eff = self._roi_analytics(total_cost, total_savings)

        risk_before, risk_after, risk_red, risk_red_pct = self._risk_reduction(
            assets, selected_ids)

        budget_util = (0.0 if math.isinf(self.config.budget)
                       else (total_cost / self.config.budget
                             if self.config.budget > 0 else 0.0))

        top_risks = self._top_by(assets, key=lambda a: a.risk_score)
        top_opps = self._top_by(assets, key=lambda a: a.expected_savings)
        immediate = tuple(
            a.asset_id for a in selected
            if a.maintenance_action in ("immediate_maintenance", "shutdown")
        )

        confidence = self._confidence(snapshot, selected, total_cost)
        recommendations = self._recommendations(
            selected, total_cost, total_savings, risk_red_pct, budget_util)
        summary = self._executive_summary(
            snapshot, selected, total_cost, total_savings, roi, risk_red_pct,
            budget_util, top_risks)

        portfolio = ExecutiveDecisionPortfolio(
            strategy=self.config.strategy,
            budget=float(self.config.budget),
            selected_asset_ids=tuple(a.asset_id for a in selected),
            total_maintenance_cost=total_cost,
            expected_savings=total_savings,
            total_roi=roi,
            payback_ratio=payback,
            cost_efficiency=cost_eff,
            budget_utilization=float(budget_util),
            expected_risk_reduction=risk_red,
            portfolio_risk_reduction_pct=risk_red_pct,
            average_risk_before=risk_before,
            average_risk_after=risk_after,
            prioritization=prioritization,
            top_risks=top_risks,
            top_opportunities=top_opps,
            immediate_action_assets=immediate,
            recommendations=recommendations,
            executive_summary=summary,
            confidence_score=confidence,
            currency=self.config.currency,
        )
        self._log_portfolio(portfolio)
        self._n_runs += 1
        return portfolio

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    def _priority_table(
        self, assets: Sequence[FleetAsset]
    ) -> list[tuple[FleetAsset, float]]:
        """Compute the priority score for every asset.

        Args:
            assets: The fleet assets.

        Returns:
            A list of ``(asset, priority_score)`` pairs.
        """
        max_sav = max((a.expected_savings for a in assets), default=0.0)
        out = []
        for a in assets:
            out.append((a, self._priority_score(a, max_sav)))
        return out

    def _priority_score(self, asset: FleetAsset, max_savings: float) -> float:
        """Compute the composite priority score for one asset.

        ``priority = w_risk·risk + w_sav·savings_norm + w_crit·criticality
        + w_urg·urgency``, normalised by the weight sum.

        Args:
            asset: The asset.
            max_savings: The fleet's maximum expected savings (for normalising).

        Returns:
            The priority score in ``[0, 1]``.
        """
        cfg = self.config
        risk = float(np.clip(asset.risk_score, 0.0, 1.0))
        sav_norm = (asset.expected_savings / max_savings
                    if max_savings > _EPS else 0.0)
        sav_norm = float(np.clip(sav_norm, 0.0, 1.0))
        criticality = float(np.clip(asset.severity_score / _SEVERITY_CAP,
                                    0.0, 1.0))
        urgency = float(np.clip(asset.failure_probability, 0.0, 1.0))
        blended = (cfg.weight_risk * risk + cfg.weight_savings * sav_norm
                   + cfg.weight_criticality * criticality
                   + cfg.weight_urgency * urgency)
        return float(blended / cfg.weight_sum)

    def prioritize_assets(
        self, snapshot: FleetSnapshot
    ) -> list[PrioritizedAsset]:
        """Return the fleet assets ranked by priority score (public helper).

        Args:
            snapshot: The fleet snapshot.

        Returns:
            The prioritised assets, highest priority first.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        if not snapshot.assets:
            raise ValueError("prioritize_assets requires a non-empty snapshot")
        scored = self._priority_table(list(snapshot.assets))
        return self._build_prioritization(scored, selected_ids=set())

    def _build_prioritization(
        self,
        scored: Sequence[tuple[FleetAsset, float]],
        selected_ids: set[str],
    ) -> tuple[PrioritizedAsset, ...]:
        """Rank scored assets and mark the selected ones.

        Args:
            scored: ``(asset, priority_score)`` pairs.
            selected_ids: The ids selected by the optimizer.

        Returns:
            The prioritised assets, highest priority first (ties by id).
        """
        ordered = sorted(scored, key=lambda t: (-t[1], t[0].asset_id))
        out = []
        for rank, (asset, score) in enumerate(ordered, start=1):
            roi = (asset.expected_savings / asset.maintenance_cost
                   if asset.maintenance_cost > _EPS else 0.0)
            out.append(PrioritizedAsset(
                asset_id=asset.asset_id,
                location=asset.location,
                risk_score=float(asset.risk_score),
                maintenance_cost=float(asset.maintenance_cost),
                expected_savings=float(asset.expected_savings),
                roi=float(roi),
                priority_score=float(score),
                rank=rank,
                selected=asset.asset_id in selected_ids,
            ))
        return tuple(out)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize_portfolio(
        self, snapshot: FleetSnapshot
    ) -> list[FleetAsset]:
        """Return the budget-constrained selected assets (public helper).

        Args:
            snapshot: The fleet snapshot.

        Returns:
            The selected assets under the configured strategy and budget.

        Raises:
            ValueError: When the snapshot has no assets.
        """
        if not snapshot.assets:
            raise ValueError("optimize_portfolio requires a non-empty snapshot")
        scored = self._priority_table(list(snapshot.assets))
        return self._optimize(scored)

    def _optimize(
        self, scored: Sequence[tuple[FleetAsset, float]]
    ) -> list[FleetAsset]:
        """Select assets under the budget using the configured strategy.

        Args:
            scored: ``(asset, priority_score)`` pairs.

        Returns:
            The selected assets (greedy under the chosen ordering).
        """
        strategy = self.config.strategy
        if strategy == OptimizationStrategy.GREEDY_ROI.value:
            order = sorted(
                scored,
                key=lambda t: (-self._roi_of(t[0]), t[0].asset_id))
        elif strategy == OptimizationStrategy.GREEDY_SAVINGS.value:
            order = sorted(
                scored,
                key=lambda t: (-t[0].expected_savings, t[0].asset_id))
        else:  # HYBRID — order by priority score
            order = sorted(scored, key=lambda t: (-t[1], t[0].asset_id))

        budget = self.config.budget
        selected: list[FleetAsset] = []
        spent = 0.0
        for asset, _ in order:
            cost = asset.maintenance_cost
            if math.isinf(budget) or spent + cost <= budget + _EPS:
                selected.append(asset)
                spent += cost
        return selected

    @staticmethod
    def _roi_of(asset: FleetAsset) -> float:
        """Return an asset's ROI (savings / cost), 0 when cost is zero.

        Args:
            asset: The asset.

        Returns:
            The ROI ratio.
        """
        return (asset.expected_savings / asset.maintenance_cost
                if asset.maintenance_cost > _EPS else 0.0)

    # ------------------------------------------------------------------
    # ROI analytics
    # ------------------------------------------------------------------

    @staticmethod
    def _roi_analytics(
        total_cost: float, total_savings: float
    ) -> tuple[float, float, float]:
        """Compute ROI, payback ratio, and cost efficiency.

        Args:
            total_cost: Total portfolio maintenance cost.
            total_savings: Total portfolio expected savings.

        Returns:
            Tuple ``(roi, payback_ratio, cost_efficiency)``.  ROI is the net
            return per unit cost ``(savings − cost) / cost``; payback ratio is
            ``savings / cost``; cost efficiency is savings per unit cost (alias
            of payback).  All are ``0`` when there is no spend.
        """
        if total_cost <= _EPS:
            return 0.0, 0.0, 0.0
        roi = (total_savings - total_cost) / total_cost
        payback = total_savings / total_cost
        return float(roi), float(payback), float(payback)

    # ------------------------------------------------------------------
    # Risk-reduction analytics
    # ------------------------------------------------------------------

    def _risk_reduction(
        self, assets: Sequence[FleetAsset], selected_ids: set[str]
    ) -> tuple[float, float, float, float]:
        """Compute mean fleet risk before and after the selected maintenance.

        A repaired asset's risk is reduced by ``recovery_factor``; an unrepaired
        asset's risk is unchanged.

        Args:
            assets: All fleet assets.
            selected_ids: The ids selected for maintenance.

        Returns:
            Tuple ``(risk_before, risk_after, reduction, reduction_pct)``.
        """
        scores = np.array([a.risk_score for a in assets], dtype=float)
        risk_before = float(np.mean(scores))
        rec = self.config.recovery_factor
        after = np.array([
            a.risk_score * (1.0 - rec) if a.asset_id in selected_ids
            else a.risk_score
            for a in assets
        ], dtype=float)
        risk_after = float(np.mean(after))
        reduction = risk_before - risk_after
        reduction_pct = (100.0 * reduction / risk_before
                         if risk_before > _EPS else 0.0)
        return risk_before, risk_after, float(reduction), float(reduction_pct)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _confidence(
        self, snapshot: FleetSnapshot, selected: Sequence[FleetAsset],
        total_cost: float,
    ) -> float:
        """Compute the decision confidence score in ``[0, 1]``.

        Combines three components: *data quality* (a heuristic from fleet size —
        larger fleets give more stable aggregates), *portfolio coverage* (the
        fraction of high-risk assets the selection addresses), and *risk
        dispersion* (``1 − risk_concentration`` — diversified risk gives a more
        robust decision than risk hidden in one asset).

        Args:
            snapshot: The fleet snapshot.
            selected: The selected assets.
            total_cost: The portfolio cost.

        Returns:
            The confidence score in ``[0, 1]``.
        """
        n = snapshot.asset_count
        # Data quality: saturating in fleet size (10+ assets -> ~1.0).
        data_quality = float(np.clip(n / 10.0, 0.0, 1.0))

        # Portfolio coverage: fraction of at-risk assets that are selected.
        at_risk = [a for a in snapshot.assets if a.risk_score >= 0.5]
        if at_risk:
            sel_ids = {a.asset_id for a in selected}
            covered = sum(1 for a in at_risk if a.asset_id in sel_ids)
            coverage = covered / len(at_risk)
        else:
            coverage = 1.0  # nothing at risk -> the decision is trivially sound

        # Risk dispersion: 1 - concentration (diversified -> more confident).
        dispersion = float(np.clip(1.0 - snapshot.risk_concentration, 0.0, 1.0))

        confidence = 0.40 * data_quality + 0.40 * coverage + 0.20 * dispersion
        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Recommendations & narrative
    # ------------------------------------------------------------------

    def _top_by(self, assets: Sequence[FleetAsset], *, key) -> tuple[str, ...]:
        """Return the ids of the top-N assets by *key*, descending.

        Args:
            assets: The fleet assets.
            key: A function mapping an asset to a sortable score.

        Returns:
            The top-N asset ids (ties broken by id).
        """
        ordered = sorted(assets, key=lambda a: (-key(a), a.asset_id))
        return tuple(a.asset_id for a in ordered[:self.config.top_n])

    def _recommendations(
        self, selected: Sequence[FleetAsset], total_cost: float,
        total_savings: float, risk_red_pct: float, budget_util: float,
    ) -> tuple[str, ...]:
        """Build structured recommendation lines.

        Args:
            selected: The selected assets.
            total_cost: The portfolio cost.
            total_savings: The portfolio savings.
            risk_red_pct: The percentage risk reduction.
            budget_util: The budget utilisation fraction.

        Returns:
            A tuple of recommendation strings.
        """
        cur = self.config.currency
        recs = []
        if selected:
            recs.append(
                f"Approve maintenance on {len(selected)} asset(s) for "
                f"{cur} {total_cost:,.0f}, returning {cur} {total_savings:,.0f} "
                f"in expected savings.")
            recs.append(
                f"Projected fleet risk reduction: {risk_red_pct:.1f}%.")
        else:
            recs.append("No maintenance is affordable within the budget; "
                        "consider increasing the budget or deferring.")
        if not math.isinf(self.config.budget):
            recs.append(f"Budget utilisation: {budget_util:.0%}.")
        urgent = [a.asset_id for a in selected
                  if a.maintenance_action in ("immediate_maintenance",
                                              "shutdown")]
        if urgent:
            recs.append("Immediate action required on: "
                        + ", ".join(urgent) + ".")
        return tuple(recs)

    def _executive_summary(
        self, snapshot: FleetSnapshot, selected: Sequence[FleetAsset],
        total_cost: float, total_savings: float, roi: float,
        risk_red_pct: float, budget_util: float, top_risks: tuple[str, ...],
    ) -> str:
        """Compose the natural-language executive summary.

        Args:
            snapshot: The fleet snapshot.
            selected: The selected assets.
            total_cost: The portfolio cost.
            total_savings: The portfolio savings.
            roi: The portfolio ROI.
            risk_red_pct: The percentage risk reduction.
            budget_util: The budget utilisation fraction.
            top_risks: The highest-risk asset ids.

        Returns:
            The executive summary string.
        """
        cur = self.config.currency
        parts = [
            f"Of {snapshot.asset_count} fleet assets, "
            f"{len(selected)} are recommended for maintenance this period."
        ]
        if selected:
            parts.append(
                f"The proposed portfolio costs {cur} {total_cost:,.0f} and is "
                f"projected to return {cur} {total_savings:,.0f} in savings "
                f"(ROI {roi:.0%}), cutting fleet risk by {risk_red_pct:.0f}%.")
        else:
            parts.append("No maintenance is affordable within the current "
                         "budget.")
        if top_risks:
            parts.append(f"Highest residual exposure: {top_risks[0]}.")
        if not math.isinf(self.config.budget):
            parts.append(f"This consumes {budget_util:.0%} of the approved "
                         "budget.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_portfolio(self, portfolio: ExecutiveDecisionPortfolio) -> None:
        """Log a portfolio summary to the experiment tracker (failure-safe).

        Args:
            portfolio: The portfolio to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "exec_selected_count": float(len(portfolio.selected_asset_ids)),
                    "exec_total_cost": float(portfolio.total_maintenance_cost),
                    "exec_expected_savings": float(portfolio.expected_savings),
                    "exec_roi": float(portfolio.total_roi),
                    "exec_risk_reduction_pct":
                        float(portfolio.portfolio_risk_reduction_pct),
                    "exec_budget_utilization":
                        float(portfolio.budget_utilization),
                    "exec_confidence": float(portfolio.confidence_score),
                },
                step=self._n_runs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"exec_strategy": portfolio.strategy})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short executive-decision demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    from src.fleet.fleet_digital_twin import (
        AssetInput,
        FleetDigitalTwinConfig,
        FleetDigitalTwinEngine,
    )

    rng = np.random.default_rng(11)
    specs = [
        ("WTG-001", "North Sea", 0.4), ("WTG-014", "North Sea", 0.9),
        ("WTG-027", "Baltic", 1.4), ("WTG-033", "Baltic", 0.6),
        ("WTG-042", "North Sea", 2.6), ("WTG-051", "Celtic", 1.1),
        ("WTG-068", "Celtic", 1.9), ("WTG-074", "Baltic", 0.5),
        ("WTG-080", "North Sea", 2.2), ("WTG-091", "Celtic", 0.7),
    ]
    assets = []
    for aid, loc, rate in specs:
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.4, 45),
                       0, 100)
        assets.append(AssetInput(asset_id=aid, asset_type="wind_turbine",
                                 location=loc, health_trajectory=traj))
    snapshot = FleetDigitalTwinEngine(FleetDigitalTwinConfig()).build_fleet_snapshot(assets)

    for strat in (OptimizationStrategy.GREEDY_ROI.value,
                  OptimizationStrategy.GREEDY_SAVINGS.value,
                  OptimizationStrategy.HYBRID.value):
        engine = ExecutiveDecisionEngine(
            ExecutiveDecisionConfig(budget=20_000, strategy=strat))
        p = engine.recommend(snapshot)
        logger.info(
            "[%-14s] selected=%d cost=%s %s savings=%s %s ROI=%d%% "
            "risk_cut=%d%% conf=%.2f",
            strat, len(p.selected_asset_ids), p.currency,
            f"{p.total_maintenance_cost:,.0f}", p.currency,
            f"{p.expected_savings:,.0f}", round(p.total_roi * 100),
            round(p.portfolio_risk_reduction_pct), p.confidence_score)

    engine = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=20_000))
    p = engine.recommend(snapshot)
    print()
    print(p.executive_summary)
    print()
    print("Recommendations:")
    for r in p.recommendations:
        print(f"  - {r}")
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
    parser = argparse.ArgumentParser(description="Executive decision engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run an executive-decision demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered executive decision engines:",
              list_executive_decision_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())