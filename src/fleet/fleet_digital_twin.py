#!/usr/bin/env python3
"""Fleet digital-twin engine — fleet-level health, risk, and decision intelligence.

Week-7 Phase-2 moves the platform from *asset-level* intelligence to *fleet-level*
intelligence.  Where the Week-5 predictive stack answers, for one machine, "how
healthy is it, how long will it last, how likely is it to fail, and what should
we do?", this engine answers the questions an asset-management organisation asks
of its **whole fleet**:

    "How many of our assets are healthy, in warning, or critical?  What is the
     fleet's aggregate failure exposure and expected cost?  Which assets carry
     the most risk, which offer the most maintenance savings, and is our risk
     concentrated in a few machines or spread across the fleet?"

The engine is a strict, additive **composition** of the frozen prognostic
engines delivered in Weeks 1–5.  For each managed asset it runs the full
predictive chain — health-trend analysis, RUL prediction, failure-risk scoring,
and the maintenance decision agent — to produce a :class:`FleetAsset` record,
then aggregates those records into a :class:`FleetSnapshot` with portfolio KPIs,
risk and opportunity rankings, concentration analytics, and an executive
narrative.  It modifies no prior code.

============================================================================
Architecture
============================================================================
::

    health trajectory per asset
        ▼  (frozen Week-5 chain, reused exactly)
    HealthTrendAnalyzer → RULPredictor → FailureRiskEngine → MaintenanceDecisionAgent
        ▼
    FleetAsset (frozen)        one managed asset's condition + decision
        ▼
    FleetDigitalTwinEngine     build_fleet_snapshot / rank / identify / summarise
        ▼
    FleetSnapshot (frozen)     fleet KPIs, rankings, analytics, narrative

============================================================================
Architecture properties
============================================================================
* **Pure NumPy** — no SciPy, no PyTorch.
* **Frozen, validated dataclasses** for the config, asset record, and snapshot.
* **Registry-compatible**, mirroring every prior module.
* **JSON-serialisable** outputs (non-finite floats render as ``null``).
* **Deterministic** — the predictive chain is deterministic, and so is this.
* **Failure-safe tracker integration**; no global mutable state.

Usage::

    import numpy as np
    from src.fleet.fleet_digital_twin import (
        FleetDigitalTwinEngine, FleetDigitalTwinConfig, AssetInput,
    )

    engine = FleetDigitalTwinEngine(FleetDigitalTwinConfig())
    assets = [
        AssetInput(asset_id="WTG-001", asset_type="wind_turbine",
                   location="North Sea",
                   health_trajectory=np.clip(95 - 0.6*np.arange(40), 0, 100)),
        AssetInput(asset_id="WTG-002", asset_type="wind_turbine",
                   location="Baltic",
                   health_trajectory=np.clip(95 - 2.0*np.arange(40), 0, 100)),
    ]
    snapshot = engine.build_fleet_snapshot(assets)
    print(snapshot.generate_fleet_summary())

CLI::

    python src/fleet/fleet_digital_twin.py --demo
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

from src.predictive.failure_risk import (  # noqa: E402
    FailureRiskConfig,
    FailureRiskEngine,
)
from src.predictive.health_trend_analyzer import (  # noqa: E402
    HealthTrendAnalyzer,
    HealthTrendConfig,
)
from src.predictive.maintenance_decision_agent import (  # noqa: E402
    MaintenanceDecisionAgent,
    MaintenanceDecisionConfig,
)
from src.predictive.rul_predictor import RULConfig, RULPredictor  # noqa: E402

logger = logging.getLogger("fleet_digital_twin")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named fleet digital-twin engines.
FLEET_DIGITAL_TWIN_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "fleet_digital_twin_engine"

#: Health-band thresholds (health index in [0, 100]).
HEALTHY_THRESHOLD: Final[float] = 70.0
WARNING_THRESHOLD: Final[float] = 50.0

#: Normalisation caps for the fleet risk score.
_RUL_CAP: Final[float] = 180.0
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
# Health-band enum
# ---------------------------------------------------------------------------


class HealthBand(str, Enum):
    """Coarse health classification for fleet roll-up."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_fleet_digital_twin(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a fleet digital-twin engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = FLEET_DIGITAL_TWIN_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Fleet digital twin '{name}' already registered to "
                f"{existing.__name__}"
            )
        FLEET_DIGITAL_TWIN_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered fleet digital twin '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_fleet_digital_twin(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered fleet digital-twin engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in FLEET_DIGITAL_TWIN_REGISTRY:
        available = ", ".join(sorted(FLEET_DIGITAL_TWIN_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown fleet digital twin '{name}'. Available: {available}"
        )
    return FLEET_DIGITAL_TWIN_REGISTRY[name](**kwargs)


def list_fleet_digital_twins() -> list[str]:
    """Return the sorted names of registered fleet digital-twin engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(FLEET_DIGITAL_TWIN_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetDigitalTwinConfig:
    """Configuration for the :class:`FleetDigitalTwinEngine`.

    Attributes:
        failure_threshold: Health at/below which an asset is deemed failed; passed
            to the RUL predictor and risk engine.
        rul_model: The RUL extrapolation model ("linear" or "exponential").
        healthy_threshold: Health at/above which an asset is "healthy".
        warning_threshold: Health at/above which an asset is "warning" (below is
            "critical").
        weight_health: Risk-score weight on (inverse) health.
        weight_rul: Risk-score weight on (inverse) RUL.
        weight_failure_prob: Risk-score weight on failure probability.
        weight_severity: Risk-score weight on decision severity.
        top_n: Default number of assets in top-risk / top-opportunity lists.
        pareto_fraction: Fraction of the fleet treated as the "top" cohort for the
            Pareto concentration metric (e.g. 0.2 = top 20%).
        currency: Currency label for monetary KPIs.
    """

    failure_threshold:   float = 30.0
    rul_model:           str = "linear"
    healthy_threshold:   float = HEALTHY_THRESHOLD
    warning_threshold:   float = WARNING_THRESHOLD
    weight_health:       float = 0.30
    weight_rul:          float = 0.25
    weight_failure_prob: float = 0.30
    weight_severity:     float = 0.15
    top_n:               int = 5
    pareto_fraction:     float = 0.20
    currency:            str = "USD"

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid thresholds, weights, or parameters.
        """
        if not (0.0 <= self.warning_threshold <= self.healthy_threshold <= 100.0):
            raise ValueError(
                "require 0 <= warning_threshold <= healthy_threshold <= 100"
            )
        if not (0.0 <= self.failure_threshold <= 100.0):
            raise ValueError("failure_threshold must be in [0, 100]")
        weights = (self.weight_health, self.weight_rul,
                   self.weight_failure_prob, self.weight_severity)
        if any(w < 0 for w in weights):
            raise ValueError("risk-score weights must be >= 0")
        if sum(weights) <= 0:
            raise ValueError("risk-score weights must sum to > 0")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not (0.0 < self.pareto_fraction <= 1.0):
            raise ValueError("pareto_fraction must be in (0, 1]")

    @property
    def weight_sum(self) -> float:
        """Return the sum of the four risk-score weights."""
        return (self.weight_health + self.weight_rul
                + self.weight_failure_prob + self.weight_severity)


# ---------------------------------------------------------------------------
# Asset input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetInput:
    """Input describing one managed asset to evaluate.

    Attributes:
        asset_id: Unique, stable asset identifier.
        asset_type: Asset class (e.g. ``"wind_turbine"``).
        location: Physical location / site / business unit.
        health_trajectory: The observed health-index history ``(T,)`` on a
            ``[0, 100]`` scale (most recent last).
    """

    asset_id:          str
    asset_type:        str
    location:          str
    health_trajectory: "np.ndarray"

    def __post_init__(self) -> None:
        """Validate the asset input.

        Raises:
            ValueError: On an empty id or an empty trajectory.
        """
        if not self.asset_id or not str(self.asset_id).strip():
            raise ValueError("asset_id must be a non-empty string")
        traj = np.asarray(self.health_trajectory, dtype=float)
        if traj.size == 0:
            raise ValueError("health_trajectory must be non-empty")
        object.__setattr__(self, "health_trajectory", traj)


# ---------------------------------------------------------------------------
# Fleet asset record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetAsset:
    """One managed asset's condition and maintenance decision.

    Attributes:
        asset_id: Unique asset identifier.
        asset_type: Asset class.
        location: Physical location / site.
        health: Current health index in ``[0, 100]``.
        predicted_rul: Predicted remaining useful life (may be ``inf``).
        failure_probability: Dominant-horizon failure probability in ``[0, 1]``.
        maintenance_action: The recommended maintenance action.
        maintenance_cost: The cost of the recommended action.
        downtime_hours: The downtime associated with the recommended action.
        expected_savings: Expected savings of proactive maintenance vs failure.
        severity_score: The maintenance decision's severity score.
        health_band: Coarse health classification.
        risk_score: Normalised fleet risk score in ``[0, 1]`` (higher = riskier).
    """

    asset_id:            str
    asset_type:          str
    location:            str
    health:              float
    predicted_rul:       float
    failure_probability: float
    maintenance_action:  str
    maintenance_cost:    float
    downtime_hours:      float
    expected_savings:    float
    severity_score:      float
    health_band:         str
    risk_score:          float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the asset record.
        """
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "location": self.location,
            "health": _jsonsafe(self.health),
            "predicted_rul": _jsonsafe(self.predicted_rul),
            "failure_probability": _jsonsafe(self.failure_probability),
            "maintenance_action": self.maintenance_action,
            "maintenance_cost": _jsonsafe(self.maintenance_cost),
            "downtime_hours": _jsonsafe(self.downtime_hours),
            "expected_savings": _jsonsafe(self.expected_savings),
            "severity_score": _jsonsafe(self.severity_score),
            "health_band": self.health_band,
            "risk_score": _jsonsafe(self.risk_score),
        }


# ---------------------------------------------------------------------------
# Fleet snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetSnapshot:
    """Immutable snapshot of an entire fleet's condition and economics.

    Attributes:
        asset_count: Number of assets in the fleet.
        healthy_assets: Count with health >= healthy threshold.
        warning_assets: Count in the warning band.
        critical_assets: Count below the warning threshold.
        average_health: Mean asset health.
        average_rul: Mean finite predicted RUL (censored values excluded).
        fleet_failure_probability: Mean per-asset failure probability.
        fleet_expected_cost: Total recommended maintenance cost across the fleet.
        fleet_expected_downtime: Total recommended downtime across the fleet.
        fleet_expected_failure_cost: Expected failure cost (sum of per-asset
            ``failure_probability × failure_cost``).
        fleet_expected_savings: Total expected savings from proactive maintenance.
        risk_concentration: Herfindahl index of the fleet risk scores.
        pareto_concentration: Fraction of total risk held by the top cohort.
        assets: The per-asset records (sorted by ``asset_id``).
        currency: Currency label for the monetary KPIs.
    """

    asset_count:                  int
    healthy_assets:               int
    warning_assets:               int
    critical_assets:              int
    average_health:               float
    average_rul:                  float
    fleet_failure_probability:    float
    fleet_expected_cost:          float
    fleet_expected_downtime:      float
    fleet_expected_failure_cost:  float
    fleet_expected_savings:       float
    risk_concentration:           float
    pareto_concentration:         float
    assets:                       tuple[FleetAsset, ...]
    currency:                     str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; per-asset records are nested.
        """
        return {
            "asset_count": self.asset_count,
            "healthy_assets": self.healthy_assets,
            "warning_assets": self.warning_assets,
            "critical_assets": self.critical_assets,
            "average_health": _jsonsafe(self.average_health),
            "average_rul": _jsonsafe(self.average_rul),
            "fleet_failure_probability": _jsonsafe(self.fleet_failure_probability),
            "fleet_expected_cost": _jsonsafe(self.fleet_expected_cost),
            "fleet_expected_downtime": _jsonsafe(self.fleet_expected_downtime),
            "fleet_expected_failure_cost":
                _jsonsafe(self.fleet_expected_failure_cost),
            "fleet_expected_savings": _jsonsafe(self.fleet_expected_savings),
            "risk_concentration": _jsonsafe(self.risk_concentration),
            "pareto_concentration": _jsonsafe(self.pareto_concentration),
            "assets": [a.to_dict() for a in self.assets],
            "currency": self.currency,
        }

    def generate_fleet_summary(self) -> str:
        """Compose a plain-English management summary.

        Returns:
            A human-readable executive narrative of the fleet state.
        """
        parts: list[str] = [
            f"Fleet contains {self.asset_count} "
            f"{'asset' if self.asset_count == 1 else 'assets'}."
        ]
        if self.warning_assets:
            parts.append(f"{self.warning_assets} "
                         f"{'is' if self.warning_assets == 1 else 'are'} "
                         "in warning state.")
        if self.critical_assets:
            parts.append(f"{self.critical_assets} "
                         f"{'is' if self.critical_assets == 1 else 'are'} "
                         "critical.")
        if self.assets:
            worst = max(self.assets, key=lambda a: a.risk_score)
            parts.append(
                f"Asset {worst.asset_id} shows the highest failure exposure "
                f"(health {worst.health:.0f}, "
                f"failure probability {worst.failure_probability:.0%})."
            )
        parts.append(
            f"Average fleet health is {self.average_health:.0f} with mean "
            f"remaining life of {self.average_rul:.0f}."
        )
        if self.fleet_expected_savings > 0:
            parts.append(
                f"Estimated savings from proactive maintenance are "
                f"{self.currency} {self.fleet_expected_savings:,.0f}."
            )
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Fleet digital-twin engine
# ---------------------------------------------------------------------------


@register_fleet_digital_twin(ENGINE_NAME)
class FleetDigitalTwinEngine:
    """Builds fleet-level intelligence by composing the frozen predictive chain.

    The engine constructs the frozen Week-5 prognostic engines once and reuses
    them across every asset.  For each asset it runs trend → RUL → risk →
    maintenance decision, assembles a :class:`FleetAsset`, and aggregates the
    fleet into a :class:`FleetSnapshot`.  It holds no per-asset mutable state.

    Args:
        config: The fleet configuration.
        experiment_tracker: Optional tracker for logging fleet summaries.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: FleetDigitalTwinConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or FleetDigitalTwinConfig()
        self.tracker = experiment_tracker
        # Frozen Week-5 engines, constructed once and reused (read-only use).
        self._trend = HealthTrendAnalyzer(HealthTrendConfig())
        self._rul = RULPredictor(RULConfig(
            failure_threshold=self.config.failure_threshold,
            model=self.config.rul_model))
        self._risk = FailureRiskEngine(FailureRiskConfig())
        self._agent = MaintenanceDecisionAgent(MaintenanceDecisionConfig(
            currency=self.config.currency))
        self._n_runs = 0
        logger.info("FleetDigitalTwinEngine ready | currency=%s",
                    self.config.currency)

    # ------------------------------------------------------------------
    # Per-asset evaluation
    # ------------------------------------------------------------------

    def evaluate_asset(self, asset: AssetInput) -> FleetAsset:
        """Run the full predictive chain for one asset.

        Args:
            asset: The asset input (identity + health trajectory).

        Returns:
            A populated :class:`FleetAsset`.
        """
        traj = np.asarray(asset.health_trajectory, dtype=float)
        trend = self._trend.analyze(traj)
        rul = self._rul.predict(traj)
        risk = self._risk.predict_from_rul(rul)
        decision = self._agent.decide(trend, rul, risk)

        health = float(decision.current_health)
        rul_val = float(decision.remaining_useful_life)
        p_fail = float(decision.failure_probability)
        severity = float(decision.severity_score)
        band = self._health_band(health)
        score = self._risk_score(health, rul_val, p_fail, severity)

        return FleetAsset(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            location=asset.location,
            health=health,
            predicted_rul=rul_val,
            failure_probability=p_fail,
            maintenance_action=decision.action.value,
            maintenance_cost=float(decision.maintenance_cost),
            downtime_hours=float(decision.estimated_downtime_hours),
            expected_savings=float(decision.expected_savings),
            severity_score=severity,
            health_band=band,
            risk_score=score,
        )

    # ------------------------------------------------------------------
    # Fleet snapshot
    # ------------------------------------------------------------------

    def build_fleet_snapshot(
        self, assets: Sequence[AssetInput]
    ) -> FleetSnapshot:
        """Evaluate every asset and aggregate the fleet snapshot.

        Args:
            assets: The fleet's asset inputs.

        Returns:
            The aggregated :class:`FleetSnapshot`.

        Raises:
            ValueError: When *assets* is empty or contains duplicate ids.
        """
        if not assets:
            raise ValueError("build_fleet_snapshot requires >= 1 asset")
        ids = [a.asset_id for a in assets]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate asset_id in fleet input")

        records = tuple(
            sorted((self.evaluate_asset(a) for a in assets),
                   key=lambda r: r.asset_id)
        )
        snapshot = self._aggregate(records)
        self._log_snapshot(snapshot)
        self._n_runs += 1
        return snapshot

    def _aggregate(self, records: tuple[FleetAsset, ...]) -> FleetSnapshot:
        """Aggregate per-asset records into a fleet snapshot.

        Args:
            records: The per-asset records.

        Returns:
            The :class:`FleetSnapshot`.
        """
        cfg = self.config
        n = len(records)
        healths = np.array([r.health for r in records], dtype=float)
        ruls = np.array([r.predicted_rul for r in records], dtype=float)
        pfails = np.array([r.failure_probability for r in records], dtype=float)
        costs = np.array([r.maintenance_cost for r in records], dtype=float)
        downtimes = np.array([r.downtime_hours for r in records], dtype=float)
        savings = np.array([r.expected_savings for r in records], dtype=float)
        scores = np.array([r.risk_score for r in records], dtype=float)

        healthy = int(np.sum(healths >= cfg.healthy_threshold))
        warning = int(np.sum((healths < cfg.healthy_threshold)
                             & (healths >= cfg.warning_threshold)))
        critical = int(np.sum(healths < cfg.warning_threshold))

        # Average RUL over finite values only (censored/inf excluded).
        finite_rul = ruls[np.isfinite(ruls)]
        avg_rul = float(np.mean(finite_rul)) if finite_rul.size else float("inf")

        # Expected failure cost: per-asset p_fail × the agent's failure cost.
        failure_cost = self._agent.config.failure_cost
        expected_failure_cost = float(np.sum(pfails * failure_cost))

        risk_conc = self._risk_concentration(scores)
        pareto_conc = self._pareto_concentration(scores)

        return FleetSnapshot(
            asset_count=n,
            healthy_assets=healthy,
            warning_assets=warning,
            critical_assets=critical,
            average_health=float(np.mean(healths)),
            average_rul=avg_rul,
            fleet_failure_probability=float(np.mean(pfails)),
            fleet_expected_cost=float(np.sum(costs)),
            fleet_expected_downtime=float(np.sum(downtimes)),
            fleet_expected_failure_cost=expected_failure_cost,
            fleet_expected_savings=float(np.sum(savings)),
            risk_concentration=risk_conc,
            pareto_concentration=pareto_conc,
            assets=records,
            currency=cfg.currency,
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_assets_by_risk(
        self, snapshot: FleetSnapshot
    ) -> list[FleetAsset]:
        """Return the fleet assets sorted by risk score, highest first.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            Assets ordered most-risky first (ties broken by ``asset_id``).
        """
        return sorted(snapshot.assets,
                      key=lambda a: (-a.risk_score, a.asset_id))

    def rank_assets_by_savings(
        self, snapshot: FleetSnapshot
    ) -> list[FleetAsset]:
        """Return the fleet assets sorted by expected savings, highest first.

        Expected savings is the avoided failure cost minus the maintenance cost,
        as computed by the frozen maintenance decision agent.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            Assets ordered highest-savings first (ties broken by ``asset_id``).
        """
        return sorted(snapshot.assets,
                      key=lambda a: (-a.expected_savings, a.asset_id))

    def identify_top_risks(
        self, snapshot: FleetSnapshot, *, top_n: int | None = None
    ) -> list[FleetAsset]:
        """Return the top-*N* highest-risk assets.

        Args:
            snapshot: The fleet snapshot.
            top_n: Number of assets (defaults to the configured ``top_n``).

        Returns:
            The top-*N* assets by risk score.

        Raises:
            ValueError: When ``top_n`` is not positive.
        """
        k = self.config.top_n if top_n is None else int(top_n)
        if k < 1:
            raise ValueError("top_n must be >= 1")
        return self.rank_assets_by_risk(snapshot)[:k]

    def identify_top_opportunities(
        self, snapshot: FleetSnapshot, *, top_n: int | None = None
    ) -> list[FleetAsset]:
        """Return the top-*N* highest-savings assets.

        Args:
            snapshot: The fleet snapshot.
            top_n: Number of assets (defaults to the configured ``top_n``).

        Returns:
            The top-*N* assets by expected savings.

        Raises:
            ValueError: When ``top_n`` is not positive.
        """
        k = self.config.top_n if top_n is None else int(top_n)
        if k < 1:
            raise ValueError("top_n must be >= 1")
        return self.rank_assets_by_savings(snapshot)[:k]

    def generate_fleet_summary(self, snapshot: FleetSnapshot) -> str:
        """Return the plain-English management summary for a snapshot.

        Args:
            snapshot: The fleet snapshot.

        Returns:
            The executive narrative.
        """
        return snapshot.generate_fleet_summary()

    # ------------------------------------------------------------------
    # Scoring & analytics helpers
    # ------------------------------------------------------------------

    def _health_band(self, health: float) -> str:
        """Classify a health value into a coarse band.

        Args:
            health: The health index.

        Returns:
            The :class:`HealthBand` value.
        """
        if health >= self.config.healthy_threshold:
            return HealthBand.HEALTHY.value
        if health >= self.config.warning_threshold:
            return HealthBand.WARNING.value
        return HealthBand.CRITICAL.value

    def _risk_score(
        self, health: float, rul: float, p_fail: float, severity: float
    ) -> float:
        """Compute the normalised fleet risk score in ``[0, 1]``.

        Combines inverse health, inverse RUL, failure probability, and decision
        severity, each normalised to ``[0, 1]`` and blended by the configured
        weights (higher score = higher risk).

        Args:
            health: Current health.
            rul: Predicted RUL (may be ``inf``).
            p_fail: Failure probability.
            severity: Decision severity score.

        Returns:
            The risk score in ``[0, 1]``.
        """
        cfg = self.config
        h = 1.0 - float(np.clip(health / 100.0, 0.0, 1.0))
        if math.isinf(rul):
            r = 0.0  # infinite life -> no RUL-driven risk
        else:
            r = 1.0 - float(np.clip(rul / _RUL_CAP, 0.0, 1.0))
        p = float(np.clip(p_fail, 0.0, 1.0))
        s = float(np.clip(severity / _SEVERITY_CAP, 0.0, 1.0))
        blend = (cfg.weight_health * h + cfg.weight_rul * r
                 + cfg.weight_failure_prob * p + cfg.weight_severity * s)
        return float(blend / cfg.weight_sum)

    @staticmethod
    def _risk_concentration(scores: "np.ndarray") -> float:
        """Herfindahl concentration of the fleet risk scores.

        Ranges from ``1/N`` (risk spread uniformly) to ``1`` (all risk in one
        asset); ``0`` when there is no risk at all.

        Args:
            scores: The per-asset risk scores.

        Returns:
            The concentration index.
        """
        total = float(np.sum(scores))
        if total <= _EPS:
            return 0.0
        shares = scores / total
        return float(np.sum(shares ** 2))

    def _pareto_concentration(self, scores: "np.ndarray") -> float:
        """Fraction of total fleet risk held by the top cohort.

        The cohort size is ``ceil(pareto_fraction × N)``.

        Args:
            scores: The per-asset risk scores.

        Returns:
            The Pareto concentration ratio in ``[0, 1]``.
        """
        total = float(np.sum(scores))
        if total <= _EPS:
            return 0.0
        n = scores.size
        k = max(1, int(math.ceil(self.config.pareto_fraction * n)))
        top = np.sort(scores)[::-1][:k]
        return float(np.sum(top) / total)

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_snapshot(self, snapshot: FleetSnapshot) -> None:
        """Log a fleet summary to the experiment tracker (failure-safe).

        Args:
            snapshot: The snapshot to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "fleet_asset_count": float(snapshot.asset_count),
                    "fleet_healthy": float(snapshot.healthy_assets),
                    "fleet_warning": float(snapshot.warning_assets),
                    "fleet_critical": float(snapshot.critical_assets),
                    "fleet_avg_health": float(snapshot.average_health),
                    "fleet_failure_probability":
                        float(snapshot.fleet_failure_probability),
                    "fleet_expected_cost": float(snapshot.fleet_expected_cost),
                    "fleet_expected_savings":
                        float(snapshot.fleet_expected_savings),
                    "fleet_risk_concentration":
                        float(snapshot.risk_concentration),
                },
                step=self._n_runs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"fleet_currency": snapshot.currency})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short fleet digital-twin demo.

    Returns:
        Exit code 0.
    """
    rng = np.random.default_rng(7)
    specs = [
        ("WTG-001", "North Sea", 0.4), ("WTG-014", "North Sea", 0.7),
        ("WTG-027", "Baltic", 1.1), ("WTG-033", "Baltic", 0.5),
        ("WTG-042", "North Sea", 2.4), ("WTG-051", "Celtic", 0.9),
        ("WTG-068", "Celtic", 1.6), ("WTG-074", "Baltic", 0.6),
    ]
    assets = []
    for aid, loc, rate in specs:
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.4, 45),
                       0, 100)
        assets.append(AssetInput(asset_id=aid, asset_type="wind_turbine",
                                 location=loc, health_trajectory=traj))

    engine = FleetDigitalTwinEngine(FleetDigitalTwinConfig())
    snap = engine.build_fleet_snapshot(assets)
    print(engine.generate_fleet_summary(snap))
    print()
    print("Top risks:")
    for a in engine.identify_top_risks(snap, top_n=3):
        print(f"  {a.asset_id} ({a.location}): risk={a.risk_score:.3f} "
              f"health={a.health:.0f} action={a.maintenance_action}")
    print("Top opportunities:")
    for a in engine.identify_top_opportunities(snap, top_n=3):
        print(f"  {a.asset_id} ({a.location}): savings="
              f"{snap.currency} {a.expected_savings:,.0f}")
    print()
    print(f"Fleet KPIs: avg_health={snap.average_health:.1f} "
          f"avg_rul={snap.average_rul:.1f} "
          f"failure_exposure={snap.fleet_failure_probability:.3f}")
    print(f"  total_cost={snap.currency} {snap.fleet_expected_cost:,.0f} "
          f"expected_failure_cost={snap.currency} "
          f"{snap.fleet_expected_failure_cost:,.0f} "
          f"expected_savings={snap.currency} {snap.fleet_expected_savings:,.0f}")
    print(f"  risk_concentration={snap.risk_concentration:.3f} "
          f"pareto_concentration={snap.pareto_concentration:.3f}")
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
                        help="Run a fleet digital-twin demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered fleet digital twins:", list_fleet_digital_twins())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())