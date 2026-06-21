#!/usr/bin/env python3
"""Fleet digital-twin engine — N assets x M scenarios at portfolio scale.

Week-6 Phase-3 scales the single-asset what-if capability of the (frozen) Step-2
:class:`~src.simulation.scenario_engine.ScenarioEngine` to an **enterprise fleet
digital twin**.  Where Phase 2 answers "what happens to *this asset* if we change
its future conditions?", this engine answers the question an asset-management
organisation actually asks:

    "Across our entire fleet, which scenario should we pursue, what does it cost,
     how much downtime does it avoid, and which assets are most at risk?"

The engine is a strict, additive **composition** of the frozen lower layers.  It
never modifies — and never needs to modify — ``asset_state_simulator.py`` or
``scenario_engine.py``; it constructs one :class:`ScenarioEngine` per asset
baseline and aggregates the resulting :class:`ScenarioResult` value objects into
portfolio-level intelligence.  This is exactly the wrapper-and-aggregate design
prescribed by the Phase-2 architecture review (recommendations R1, R2, R4): an
asset-identity layer, a fleet wrapper that delegates to the validated core, and a
portfolio aggregator built purely by composing per-asset results.

============================================================================
Layered design
============================================================================
::

    AssetRecord (frozen)         identity + baseline_config per asset
        ▼
    FleetRegistry (immutable)    register / remove / get / list assets
        ▼
    FleetDigitalTwinEngine       run_scenario / batch / fleet / all-assets
        │   └── delegates each asset to ScenarioEngine (frozen Phase-2 core)
        ▼
    PortfolioResult (frozen)     fleet rollups (risk, cost, downtime, health)
        ▼
    ScenarioRanker               rank scenarios by a business objective
        ▼
    DecisionIntelligence         executive summary: risks, opportunities,
                                 recommended actions, savings, downtime cut

============================================================================
Architecture properties
============================================================================
* **Pure NumPy** — no SciPy, no PyTorch.
* **Frozen, validated dataclasses** for every config and result.
* **Immutable fleet registry** — mutations return a *new* registry; no in-place
  state change, so the engine has no hidden global state.
* **JSON-serialisable** results throughout (non-finite floats render as ``null``).
* **Deterministic** — seeds flow through the frozen simulator unchanged.
* **Registry-compatible**, mirroring every Week-1–6 module.

Usage::

    from src.simulation.asset_state_simulator import AssetSimulatorConfig
    from src.simulation.scenario_engine import ScenarioConfig, ScenarioType
    from src.simulation.fleet_digital_twin import (
        AssetRecord, FleetRegistry, FleetDigitalTwinEngine, RankingObjective,
    )

    registry = (
        FleetRegistry()
        .register_asset(AssetRecord(
            asset_id="WTG-001", asset_name="Turbine 1", asset_type="wind_turbine",
            business_unit="North Sea",
            baseline_config=AssetSimulatorConfig(horizon=120, degradation_rate=0.8)))
        .register_asset(AssetRecord(
            asset_id="WTG-002", asset_name="Turbine 2", asset_type="wind_turbine",
            business_unit="North Sea",
            baseline_config=AssetSimulatorConfig(horizon=120, degradation_rate=1.1)))
    )

    engine = FleetDigitalTwinEngine(registry)
    portfolio = engine.run_fleet_scenario(
        ScenarioConfig(scenario_type=ScenarioType.LOAD_REDUCTION.value,
                       degradation_factor=0.6))
    print(portfolio.summary)

CLI::

    python src/simulation/fleet_digital_twin.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.asset_state_simulator import (  # noqa: E402
    AssetSimulatorConfig,
)
from src.simulation.scenario_engine import (  # noqa: E402
    ScenarioConfig,
    ScenarioEngine,
    ScenarioResult,
)

logger = logging.getLogger("fleet_digital_twin")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named fleet engines.
FLEET_ENGINE_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "enterprise_fleet_digital_twin"

_EPS: Final[float] = 1e-9


def _jsonsafe(x: float) -> float | None:
    """Render a non-finite float as ``None`` for JSON.

    Args:
        x: A float that may be ``inf`` or ``NaN``.

    Returns:
        ``None`` when non-finite, else the float.
    """
    return None if (math.isinf(x) or math.isnan(x)) else float(x)


# ---------------------------------------------------------------------------
# Ranking objective
# ---------------------------------------------------------------------------


class RankingObjective(str, Enum):
    """Business objective by which scenarios / assets are ranked."""

    RISK_MINIMIZATION = "risk_minimization"
    COST_MINIMIZATION = "cost_minimization"
    DOWNTIME_MINIMIZATION = "downtime_minimization"
    HEALTH_MAXIMIZATION = "health_maximization"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_fleet_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a fleet engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = FLEET_ENGINE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Fleet engine '{name}' already registered to {existing.__name__}"
            )
        FLEET_ENGINE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered fleet engine '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_fleet_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered fleet engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated fleet engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in FLEET_ENGINE_REGISTRY:
        available = ", ".join(sorted(FLEET_ENGINE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown fleet engine '{name}'. Available: {available}")
    return FLEET_ENGINE_REGISTRY[name](**kwargs)


def list_fleet_engines() -> list[str]:
    """Return the sorted names of registered fleet engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(FLEET_ENGINE_REGISTRY)


# ---------------------------------------------------------------------------
# Asset identity layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetRecord:
    """Immutable identity + baseline configuration for one fleet asset.

    Attributes:
        asset_id: Unique, stable asset identifier (the fleet key).
        asset_name: Human-readable asset name.
        asset_type: Asset class (e.g. ``"wind_turbine"``, ``"gearbox"``).
        business_unit: Owning business unit / site, for portfolio grouping.
        baseline_config: The asset's baseline :class:`AssetSimulatorConfig`.
        metadata: Optional free-form metadata (never used for control flow).
    """

    asset_id:        str
    asset_name:      str
    asset_type:      str
    business_unit:   str
    baseline_config: AssetSimulatorConfig
    metadata:        Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the asset record.

        Raises:
            ValueError: When ``asset_id`` is empty or the baseline config is not
                an :class:`AssetSimulatorConfig`.
        """
        if not self.asset_id or not str(self.asset_id).strip():
            raise ValueError("asset_id must be a non-empty string")
        if not isinstance(self.baseline_config, AssetSimulatorConfig):
            raise ValueError(
                "baseline_config must be an AssetSimulatorConfig instance"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary (config rendered shallowly).

        Returns:
            Dictionary representation of the asset identity.
        """
        return {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "asset_type": self.asset_type,
            "business_unit": self.business_unit,
            "baseline_model": self.baseline_config.model,
            "baseline_horizon": self.baseline_config.horizon,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Fleet registry (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetRegistry:
    """Immutable collection of :class:`AssetRecord` keyed by ``asset_id``.

    Every mutator returns a *new* registry; the instance is never modified in
    place, so the fleet engine carries no hidden mutable state.  The internal
    mapping is wrapped in a read-only view to prevent accidental mutation.

    Attributes:
        assets: A read-only mapping ``asset_id -> AssetRecord``.
    """

    assets: Mapping[str, AssetRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the mapping behind a read-only proxy."""
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))

    def register_asset(self, record: AssetRecord) -> "FleetRegistry":
        """Return a new registry with *record* added.

        Args:
            record: The asset to register.

        Returns:
            A new :class:`FleetRegistry`.

        Raises:
            ValueError: When the ``asset_id`` is already registered.
        """
        if record.asset_id in self.assets:
            raise ValueError(f"asset_id '{record.asset_id}' already registered")
        new_map = dict(self.assets)
        new_map[record.asset_id] = record
        return FleetRegistry(new_map)

    def remove_asset(self, asset_id: str) -> "FleetRegistry":
        """Return a new registry with *asset_id* removed.

        Args:
            asset_id: The asset to remove.

        Returns:
            A new :class:`FleetRegistry`.

        Raises:
            KeyError: When the ``asset_id`` is not registered.
        """
        if asset_id not in self.assets:
            raise KeyError(f"asset_id '{asset_id}' not registered")
        new_map = dict(self.assets)
        del new_map[asset_id]
        return FleetRegistry(new_map)

    def get_asset(self, asset_id: str) -> AssetRecord:
        """Return the record for *asset_id*.

        Args:
            asset_id: The asset to fetch.

        Returns:
            The :class:`AssetRecord`.

        Raises:
            KeyError: When the ``asset_id`` is not registered.
        """
        if asset_id not in self.assets:
            raise KeyError(f"asset_id '{asset_id}' not registered")
        return self.assets[asset_id]

    def list_assets(self) -> list[str]:
        """Return the sorted registered asset ids.

        Returns:
            Sorted asset ids.
        """
        return sorted(self.assets)

    def list_by_business_unit(self, business_unit: str) -> list[str]:
        """Return sorted asset ids belonging to *business_unit*.

        Args:
            business_unit: The owning unit to filter by.

        Returns:
            Sorted matching asset ids.
        """
        return sorted(
            aid for aid, rec in self.assets.items()
            if rec.business_unit == business_unit
        )

    def __len__(self) -> int:
        """Return the number of registered assets."""
        return len(self.assets)

    def __contains__(self, asset_id: object) -> bool:
        """Return whether *asset_id* is registered."""
        return asset_id in self.assets


# ---------------------------------------------------------------------------
# Per-asset scenario outcome (lightweight wrapper around a ScenarioResult)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetScenarioOutcome:
    """A single asset's scenario result tagged with its fleet identity.

    Attributes:
        asset_id: The asset the result belongs to.
        asset_name: Human-readable asset name.
        business_unit: The asset's owning unit.
        result: The underlying :class:`ScenarioResult` from the frozen engine.
    """

    asset_id:      str
    asset_name:    str
    business_unit: str
    result:        ScenarioResult

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary with identity plus the nested scenario result.
        """
        return {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "business_unit": self.business_unit,
            "result": self.result.to_dict(),
        }


# ---------------------------------------------------------------------------
# Portfolio result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioResult:
    """Immutable fleet-level aggregation of per-asset scenario outcomes.

    Deltas follow the same *scenario minus baseline* convention as the Phase-2
    engine: a negative ``portfolio_cost_impact`` means the scenario saves money
    across the fleet; a negative ``portfolio_risk_delta`` means the fleet is
    safer.

    Attributes:
        scenario_label: A label describing the scenario(s) aggregated.
        total_assets: Number of assets in the aggregation.
        assets_failed: Count whose scenario trajectory reaches failure.
        assets_failure_avoided: Count where the scenario averts a baseline failure.
        assets_failure_induced: Count where the scenario causes a new failure.
        portfolio_risk: Mean scenario dominant-horizon failure probability.
        portfolio_risk_delta: Mean (scenario − baseline) risk across the fleet.
        portfolio_cost_impact: Summed cost impact across the fleet.
        portfolio_downtime_impact: Summed downtime impact (hours) across the fleet.
        mean_health_delta: Mean per-asset mean-health delta.
        best_asset: ``asset_id`` with the largest mean-health delta (or ``""``).
        worst_asset: ``asset_id`` with the smallest mean-health delta (or ``""``).
        outcomes: The per-asset outcomes that were aggregated.
        summary: Human-readable portfolio summary.
    """

    scenario_label:            str
    total_assets:              int
    assets_failed:             int
    assets_failure_avoided:    int
    assets_failure_induced:    int
    portfolio_risk:            float
    portfolio_risk_delta:      float
    portfolio_cost_impact:     float
    portfolio_downtime_impact: float
    mean_health_delta:         float
    best_asset:                str
    worst_asset:               str
    outcomes:                  tuple[AssetScenarioOutcome, ...]
    summary:                   str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; per-asset outcomes are nested.
        """
        return {
            "scenario_label": self.scenario_label,
            "total_assets": self.total_assets,
            "assets_failed": self.assets_failed,
            "assets_failure_avoided": self.assets_failure_avoided,
            "assets_failure_induced": self.assets_failure_induced,
            "portfolio_risk": _jsonsafe(self.portfolio_risk),
            "portfolio_risk_delta": _jsonsafe(self.portfolio_risk_delta),
            "portfolio_cost_impact": float(self.portfolio_cost_impact),
            "portfolio_downtime_impact": float(self.portfolio_downtime_impact),
            "mean_health_delta": float(self.mean_health_delta),
            "best_asset": self.best_asset,
            "worst_asset": self.worst_asset,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedItem:
    """A single ranked entry (scenario or asset) under an objective.

    Attributes:
        rank: 1-based rank (1 is best under the objective).
        label: The ranked item's label (scenario name or asset id).
        objective: The objective the ranking was computed under.
        score: The objective value used for ordering (lower is better for
            minimisation objectives; the sign is normalised so a *smaller*
            score is always better).
        detail: A small mapping of the contributing metrics.
    """

    rank:      int
    label:     str
    objective: str
    score:     float
    detail:    Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the ranked item.
        """
        return {
            "rank": self.rank,
            "label": self.label,
            "objective": self.objective,
            "score": _jsonsafe(self.score),
            "detail": {k: _jsonsafe(float(v)) for k, v in self.detail.items()},
        }


class ScenarioRanker:
    """Ranks scenarios or portfolios by a :class:`RankingObjective`.

    The ranker normalises every objective to "smaller score is better", so a
    single ascending sort produces the ranking regardless of objective.  For the
    minimisation objectives the score is the raw delta (more negative is better);
    for ``HEALTH_MAXIMIZATION`` the score is the negated health delta.
    """

    @staticmethod
    def _score(result: ScenarioResult, objective: str) -> float:
        """Return the normalised (smaller-is-better) score for *result*.

        Args:
            result: The scenario result to score.
            objective: The :class:`RankingObjective` value.

        Returns:
            The normalised score.

        Raises:
            ValueError: When *objective* is unknown.
        """
        if objective == RankingObjective.RISK_MINIMIZATION.value:
            return float(result.risk_delta)
        if objective == RankingObjective.COST_MINIMIZATION.value:
            return float(result.cost_impact)
        if objective == RankingObjective.DOWNTIME_MINIMIZATION.value:
            return float(result.downtime_impact)
        if objective == RankingObjective.HEALTH_MAXIMIZATION.value:
            return -float(result.mean_health_delta)
        raise ValueError(f"Unknown ranking objective: {objective}")

    @staticmethod
    def _detail(result: ScenarioResult) -> dict[str, float]:
        """Return the contributing metric mapping for a ranked item.

        Args:
            result: The scenario result.

        Returns:
            A mapping of the headline deltas.
        """
        return {
            "risk_delta": float(result.risk_delta),
            "cost_impact": float(result.cost_impact),
            "downtime_impact": float(result.downtime_impact),
            "mean_health_delta": float(result.mean_health_delta),
        }

    def rank_scenarios(
        self,
        labelled_results: Sequence[tuple[str, ScenarioResult]],
        objective: str,
    ) -> list[RankedItem]:
        """Rank labelled scenario results by *objective*.

        Args:
            labelled_results: Pairs of ``(label, ScenarioResult)``.
            objective: The :class:`RankingObjective` value.

        Returns:
            Ranked items, best (rank 1) first.

        Raises:
            ValueError: When *labelled_results* is empty or *objective* unknown.
        """
        if not labelled_results:
            raise ValueError("rank_scenarios requires at least one result")
        scored = [
            (label, self._score(res, objective), self._detail(res))
            for label, res in labelled_results
        ]
        # NaN scores sort last (treated as worst) deterministically.
        scored.sort(key=lambda t: (math.isnan(t[1]), t[1], t[0]))
        return [
            RankedItem(rank=i + 1, label=label, objective=objective,
                       score=score, detail=detail)
            for i, (label, score, detail) in enumerate(scored)
        ]


# ---------------------------------------------------------------------------
# Decision intelligence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionIntelligence:
    """Executive-level decision summary derived from a portfolio.

    Attributes:
        top_risks: Asset ids with the highest scenario risk, worst first.
        top_opportunities: Asset ids with the largest health improvement.
        recommended_actions: Per-asset textual recommendations.
        expected_savings: Total expected cost saved (positive = saving).
        expected_downtime_reduction: Total downtime hours avoided (positive =
            reduction).
        narrative: A human-readable executive narrative.
    """

    top_risks:                   tuple[str, ...]
    top_opportunities:           tuple[str, ...]
    recommended_actions:         tuple[str, ...]
    expected_savings:            float
    expected_downtime_reduction: float
    narrative:                   str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the decision summary.
        """
        return {
            "top_risks": list(self.top_risks),
            "top_opportunities": list(self.top_opportunities),
            "recommended_actions": list(self.recommended_actions),
            "expected_savings": float(self.expected_savings),
            "expected_downtime_reduction": float(self.expected_downtime_reduction),
            "narrative": self.narrative,
        }


# ---------------------------------------------------------------------------
# Fleet engine
# ---------------------------------------------------------------------------


@register_fleet_engine(ENGINE_NAME)
class FleetDigitalTwinEngine:
    """Runs what-if scenarios across a fleet and aggregates portfolio outcomes.

    The engine holds an immutable :class:`FleetRegistry` and delegates every
    per-asset computation to a frozen :class:`ScenarioEngine` constructed from
    that asset's baseline config.  It owns no per-asset mutable state: each call
    builds the per-asset engines it needs and returns immutable results.

    Args:
        registry: The fleet registry of assets.
        experiment_tracker: Optional tracker for logging portfolio summaries.
        failure_threshold: Health threshold used to count failed assets in
            aggregation (defaults to each asset's own configured threshold when
            ``None``).
    """

    _registry_name: str | None = None

    def __init__(
        self,
        registry: FleetRegistry | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.registry = registry or FleetRegistry()
        self.tracker = experiment_tracker
        self._ranker = ScenarioRanker()
        self._n_runs = 0
        logger.info("FleetDigitalTwinEngine ready | assets=%d", len(self.registry))

    # ------------------------------------------------------------------
    # Per-asset execution
    # ------------------------------------------------------------------

    def run_scenario(
        self, asset_id: str, scenario: ScenarioConfig
    ) -> AssetScenarioOutcome:
        """Run a single scenario against a single asset.

        Args:
            asset_id: The asset to evaluate.
            scenario: The scenario to apply.

        Returns:
            The tagged :class:`AssetScenarioOutcome`.

        Raises:
            KeyError: When the asset is not registered.
        """
        record = self.registry.get_asset(asset_id)
        engine = ScenarioEngine(record.baseline_config)
        result = engine.run(scenario)
        return AssetScenarioOutcome(
            asset_id=record.asset_id, asset_name=record.asset_name,
            business_unit=record.business_unit, result=result,
        )

    def run_scenario_batch(
        self,
        asset_ids: Sequence[str],
        scenario: ScenarioConfig,
    ) -> list[AssetScenarioOutcome]:
        """Run one scenario against a batch of assets.

        Args:
            asset_ids: The assets to evaluate.
            scenario: The scenario to apply to each.

        Returns:
            Per-asset outcomes, in the order of *asset_ids*.

        Raises:
            ValueError: When *asset_ids* is empty.
            KeyError: When any asset is not registered.
        """
        if not asset_ids:
            raise ValueError("run_scenario_batch requires at least one asset id")
        return [self.run_scenario(aid, scenario) for aid in asset_ids]

    def run_fleet_scenario(
        self, scenario: ScenarioConfig, *, scenario_label: str | None = None
    ) -> PortfolioResult:
        """Run one scenario across the entire fleet and aggregate.

        Args:
            scenario: The scenario to apply to every asset.
            scenario_label: Optional label for the portfolio result.

        Returns:
            The aggregated :class:`PortfolioResult`.

        Raises:
            ValueError: When the fleet is empty.
        """
        if len(self.registry) == 0:
            raise ValueError("run_fleet_scenario requires a non-empty fleet")
        outcomes = [
            self.run_scenario(aid, scenario)
            for aid in self.registry.list_assets()
        ]
        label = scenario_label or scenario.name or scenario.scenario_type
        portfolio = self.aggregate(outcomes, label)
        self._log_portfolio(portfolio)
        self._n_runs += 1
        return portfolio

    def run_all_assets(
        self, scenarios: Sequence[ScenarioConfig]
    ) -> dict[str, PortfolioResult]:
        """Run several scenarios across the fleet, one portfolio per scenario.

        Args:
            scenarios: The scenarios to evaluate.

        Returns:
            A mapping ``scenario_label -> PortfolioResult``.

        Raises:
            ValueError: When *scenarios* is empty or the fleet is empty.
        """
        if not scenarios:
            raise ValueError("run_all_assets requires at least one scenario")
        if len(self.registry) == 0:
            raise ValueError("run_all_assets requires a non-empty fleet")
        out: dict[str, PortfolioResult] = {}
        for i, sc in enumerate(scenarios):
            label = sc.name or f"{sc.scenario_type}_{i}"
            out[label] = self.run_fleet_scenario(sc, scenario_label=label)
        return out

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        outcomes: Sequence[AssetScenarioOutcome],
        scenario_label: str,
    ) -> PortfolioResult:
        """Aggregate per-asset outcomes into a :class:`PortfolioResult`.

        Args:
            outcomes: The per-asset outcomes to aggregate.
            scenario_label: A label describing the aggregated scenario.

        Returns:
            The portfolio aggregation.

        Raises:
            ValueError: When *outcomes* is empty.
        """
        if not outcomes:
            raise ValueError("aggregate requires at least one outcome")
        n = len(outcomes)
        results = [o.result for o in outcomes]

        assets_failed = sum(1 for r in results if r.scenario_failure_time >= 0)
        avoided = sum(1 for r in results if r.failure_avoided)
        induced = sum(1 for r in results if r.failure_induced)

        portfolio_risk = float(np.mean([r.scenario_risk for r in results]))
        portfolio_risk_delta = float(np.mean([r.risk_delta for r in results]))
        portfolio_cost = float(np.sum([r.cost_impact for r in results]))
        portfolio_down = float(np.sum([r.downtime_impact for r in results]))
        mean_hd = float(np.mean([r.mean_health_delta for r in results]))

        best_idx = int(np.argmax([r.mean_health_delta for r in results]))
        worst_idx = int(np.argmin([r.mean_health_delta for r in results]))
        best_asset = outcomes[best_idx].asset_id
        worst_asset = outcomes[worst_idx].asset_id

        summary = self._summarise_portfolio(
            scenario_label, n, assets_failed, avoided, induced,
            portfolio_cost, portfolio_down, mean_hd, best_asset, worst_asset,
        )

        return PortfolioResult(
            scenario_label=scenario_label,
            total_assets=n,
            assets_failed=assets_failed,
            assets_failure_avoided=avoided,
            assets_failure_induced=induced,
            portfolio_risk=portfolio_risk,
            portfolio_risk_delta=portfolio_risk_delta,
            portfolio_cost_impact=portfolio_cost,
            portfolio_downtime_impact=portfolio_down,
            mean_health_delta=mean_hd,
            best_asset=best_asset,
            worst_asset=worst_asset,
            outcomes=tuple(outcomes),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_scenarios(
        self,
        portfolios: Mapping[str, PortfolioResult],
        objective: str,
    ) -> list[RankedItem]:
        """Rank multiple portfolios (one per scenario) by *objective*.

        The portfolio's aggregate metrics are wrapped into a synthetic
        single-result view so the same :class:`ScenarioRanker` ordering applies.

        Args:
            portfolios: A mapping ``label -> PortfolioResult``.
            objective: The :class:`RankingObjective` value.

        Returns:
            Ranked items, best (rank 1) first.

        Raises:
            ValueError: When *portfolios* is empty or *objective* unknown.
        """
        if not portfolios:
            raise ValueError("rank_scenarios requires at least one portfolio")
        valid = {o.value for o in RankingObjective}
        if objective not in valid:
            raise ValueError(f"Unknown ranking objective: {objective}")

        scored: list[tuple[str, float, dict[str, float]]] = []
        for label, p in portfolios.items():
            if objective == RankingObjective.RISK_MINIMIZATION.value:
                score = p.portfolio_risk_delta
            elif objective == RankingObjective.COST_MINIMIZATION.value:
                score = p.portfolio_cost_impact
            elif objective == RankingObjective.DOWNTIME_MINIMIZATION.value:
                score = p.portfolio_downtime_impact
            else:  # HEALTH_MAXIMIZATION
                score = -p.mean_health_delta
            detail = {
                "portfolio_risk_delta": p.portfolio_risk_delta,
                "portfolio_cost_impact": p.portfolio_cost_impact,
                "portfolio_downtime_impact": p.portfolio_downtime_impact,
                "mean_health_delta": p.mean_health_delta,
            }
            scored.append((label, float(score), detail))
        scored.sort(key=lambda t: (math.isnan(t[1]), t[1], t[0]))
        return [
            RankedItem(rank=i + 1, label=label, objective=objective,
                       score=score, detail=detail)
            for i, (label, score, detail) in enumerate(scored)
        ]

    def rank_assets(
        self, portfolio: PortfolioResult, objective: str
    ) -> list[RankedItem]:
        """Rank the assets *within* a portfolio by *objective*.

        Args:
            portfolio: The portfolio whose assets to rank.
            objective: The :class:`RankingObjective` value.

        Returns:
            Ranked asset items, best (rank 1) first.

        Raises:
            ValueError: When the objective is unknown.
        """
        labelled = [(o.asset_id, o.result) for o in portfolio.outcomes]
        return self._ranker.rank_scenarios(labelled, objective)

    # ------------------------------------------------------------------
    # Decision intelligence
    # ------------------------------------------------------------------

    def decision_intelligence(
        self, portfolio: PortfolioResult, *, top_n: int = 3
    ) -> DecisionIntelligence:
        """Derive an executive decision summary from a portfolio.

        Args:
            portfolio: The aggregated portfolio.
            top_n: How many assets to surface in the risk/opportunity lists.

        Returns:
            The :class:`DecisionIntelligence` summary.

        Raises:
            ValueError: When ``top_n`` is not positive.
        """
        if top_n <= 0:
            raise ValueError("top_n must be >= 1")
        outcomes = list(portfolio.outcomes)

        # Top risks: highest scenario risk first.
        by_risk = sorted(outcomes, key=lambda o: -o.result.scenario_risk)
        top_risks = tuple(o.asset_id for o in by_risk[:top_n])

        # Top opportunities: largest health improvement first.
        by_opp = sorted(outcomes, key=lambda o: -o.result.mean_health_delta)
        top_opps = tuple(o.asset_id for o in by_opp[:top_n])

        # Recommended actions per asset (frozen, explainable).
        actions = tuple(self._recommend_action(o) for o in outcomes)

        # Savings: negative cost impact is a saving; sum the savings only.
        expected_savings = float(
            sum(-o.result.cost_impact for o in outcomes if o.result.cost_impact < 0)
        )
        # Downtime reduction: negative downtime impact is a reduction.
        expected_dt_reduction = float(
            sum(-o.result.downtime_impact for o in outcomes
                if o.result.downtime_impact < 0)
        )

        narrative = self._executive_narrative(
            portfolio, top_risks, top_opps, expected_savings,
            expected_dt_reduction,
        )
        return DecisionIntelligence(
            top_risks=top_risks,
            top_opportunities=top_opps,
            recommended_actions=actions,
            expected_savings=expected_savings,
            expected_downtime_reduction=expected_dt_reduction,
            narrative=narrative,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self, portfolios: Mapping[str, PortfolioResult], objective: str
    ) -> str:
        """Render a fleet decision report ranking *portfolios* by *objective*.

        Args:
            portfolios: A mapping ``label -> PortfolioResult``.
            objective: The :class:`RankingObjective` value.

        Returns:
            A formatted, human-readable report.

        Raises:
            ValueError: When *portfolios* is empty.
        """
        if not portfolios:
            raise ValueError("generate_report requires at least one portfolio")
        ranked = self.rank_scenarios(portfolios, objective)
        lines: list[str] = []
        lines.append("=" * 76)
        lines.append("FLEET DIGITAL TWIN — PORTFOLIO DECISION REPORT")
        lines.append("=" * 76)
        lines.append(f"Fleet size: {len(self.registry)} assets   "
                     f"Objective: {objective}")
        lines.append("-" * 76)
        header = (f"{'Rank':<5}{'Scenario':<26}{'ΔRisk':>9}{'ΔCost':>12}"
                  f"{'ΔDown(h)':>11}{'ΔHealth':>9}")
        lines.append(header)
        lines.append("-" * 76)
        for item in ranked:
            d = item.detail
            lines.append(
                f"{item.rank:<5}{item.label[:25]:<26}"
                f"{d['portfolio_risk_delta']:>+9.3f}"
                f"{d['portfolio_cost_impact']:>+12.0f}"
                f"{d['portfolio_downtime_impact']:>+11.1f}"
                f"{d['mean_health_delta']:>+9.1f}"
            )
        lines.append("-" * 76)
        best = ranked[0]
        lines.append(f"\nRecommended scenario: '{best.label}' "
                     f"(rank 1 under {objective}).")
        lines.append("=" * 76)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend_action(outcome: AssetScenarioOutcome) -> str:
        """Compose a per-asset recommended action string.

        Args:
            outcome: The per-asset outcome.

        Returns:
            A short recommendation referencing the asset and its outcome.
        """
        r = outcome.result
        if r.failure_avoided:
            return (f"{outcome.asset_id}: PURSUE — scenario averts projected "
                    f"failure (saves {abs(r.cost_impact):,.0f}).")
        if r.failure_induced:
            return (f"{outcome.asset_id}: AVOID — scenario induces a new failure "
                    f"(adds {abs(r.cost_impact):,.0f}).")
        if r.mean_health_delta > 0.05:
            return (f"{outcome.asset_id}: FAVOURABLE — health improves by "
                    f"{r.mean_health_delta:.1f}.")
        if r.mean_health_delta < -0.05:
            return (f"{outcome.asset_id}: CAUTION — health worsens by "
                    f"{abs(r.mean_health_delta):.1f}.")
        return f"{outcome.asset_id}: NEUTRAL — negligible change."

    @staticmethod
    def _summarise_portfolio(
        label: str, n: int, failed: int, avoided: int, induced: int,
        cost: float, downtime: float, mean_hd: float,
        best: str, worst: str,
    ) -> str:
        """Compose the portfolio summary string.

        Args:
            label: The scenario label.
            n: Total assets.
            failed: Count of failed assets.
            avoided: Count of avoided failures.
            induced: Count of induced failures.
            cost: Portfolio cost impact.
            downtime: Portfolio downtime impact.
            mean_hd: Mean health delta.
            best: Best asset id.
            worst: Worst asset id.

        Returns:
            A single-paragraph portfolio summary.
        """
        parts = [f"Scenario '{label}' across {n} assets:"]
        parts.append(f"{failed} project failure")
        if avoided:
            parts.append(f"{avoided} failure(s) avoided")
        if induced:
            parts.append(f"{induced} failure(s) induced")
        if cost < 0:
            parts.append(f"net expected saving {abs(cost):,.0f}")
        elif cost > 0:
            parts.append(f"net expected cost {cost:,.0f}")
        if abs(downtime) >= 0.1:
            verb = "reduction" if downtime < 0 else "increase"
            parts.append(f"downtime {verb} {abs(downtime):.1f}h")
        parts.append(f"mean health delta {mean_hd:+.1f}")
        parts.append(f"(best {best}, worst {worst})")
        return "; ".join(parts) + "."

    @staticmethod
    def _executive_narrative(
        portfolio: PortfolioResult,
        top_risks: tuple[str, ...],
        top_opps: tuple[str, ...],
        savings: float,
        dt_reduction: float,
    ) -> str:
        """Compose the executive narrative.

        Args:
            portfolio: The portfolio.
            top_risks: Highest-risk asset ids.
            top_opps: Highest-opportunity asset ids.
            savings: Expected savings.
            dt_reduction: Expected downtime reduction.

        Returns:
            A human-readable executive narrative.
        """
        parts = [
            f"Across {portfolio.total_assets} assets under scenario "
            f"'{portfolio.scenario_label}',",
            f"{portfolio.assets_failure_avoided} projected failure(s) are averted",
            f"and {portfolio.assets_failure_induced} induced.",
        ]
        if savings > 0:
            parts.append(f"Expected fleet saving: {savings:,.0f}.")
        if dt_reduction > 0:
            parts.append(f"Expected downtime reduction: {dt_reduction:.1f}h.")
        if top_risks:
            parts.append(f"Highest residual risk: {', '.join(top_risks)}.")
        if top_opps:
            parts.append(f"Greatest opportunity: {', '.join(top_opps)}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_portfolio(self, portfolio: PortfolioResult) -> None:
        """Log a portfolio summary to the experiment tracker (failure-safe).

        Args:
            portfolio: The portfolio result to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "fleet_total_assets": float(portfolio.total_assets),
                    "fleet_assets_failed": float(portfolio.assets_failed),
                    "fleet_failure_avoided": float(portfolio.assets_failure_avoided),
                    "fleet_failure_induced": float(portfolio.assets_failure_induced),
                    "fleet_risk": float(portfolio.portfolio_risk),
                    "fleet_cost_impact": float(portfolio.portfolio_cost_impact),
                    "fleet_downtime_impact": float(portfolio.portfolio_downtime_impact),
                    "fleet_mean_health_delta": float(portfolio.mean_health_delta),
                },
                step=self._n_runs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"fleet_scenario": portfolio.scenario_label})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short fleet what-if demo across several scenarios.

    Returns:
        Exit code 0.
    """
    from src.simulation.scenario_engine import ScenarioType

    registry = FleetRegistry()
    fleet_specs = [
        ("WTG-001", "North Sea Alpha", "North Sea", 1.1),
        ("WTG-002", "North Sea Bravo", "North Sea", 0.7),
        ("WTG-003", "Baltic Charlie", "Baltic", 0.9),
        ("WTG-004", "Baltic Delta", "Baltic", 1.4),
    ]
    for aid, name, bu, rate in fleet_specs:
        registry = registry.register_asset(AssetRecord(
            asset_id=aid, asset_name=name, asset_type="wind_turbine",
            business_unit=bu,
            baseline_config=AssetSimulatorConfig(
                horizon=120, model="linear", degradation_rate=rate,
                initial_health=95, failure_threshold=30, noise_std=0.0),
        ))

    engine = FleetDigitalTwinEngine(registry)
    scenarios = [
        ScenarioConfig(scenario_type=ScenarioType.NORMAL.value, name="Do nothing"),
        ScenarioConfig(scenario_type=ScenarioType.LOAD_REDUCTION.value,
                       name="Fleet de-rate", degradation_factor=0.6),
        ScenarioConfig(scenario_type=ScenarioType.MAINTENANCE_INTERVENTION.value,
                       name="Scheduled overhaul", intervention_time=60.0,
                       restoration_amount=40.0),
    ]
    portfolios = engine.run_all_assets(scenarios)
    print(engine.generate_report(portfolios,
                                 RankingObjective.COST_MINIMIZATION.value))
    best_label = engine.rank_scenarios(
        portfolios, RankingObjective.COST_MINIMIZATION.value)[0].label
    di = engine.decision_intelligence(portfolios[best_label])
    print("\nEXECUTIVE SUMMARY")
    print(di.narrative)
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
    parser = argparse.ArgumentParser(description="Fleet digital-twin engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a fleet what-if demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered fleet engines:", list_fleet_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())