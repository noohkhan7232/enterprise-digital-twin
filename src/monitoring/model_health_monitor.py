"""Deterministic model-health scoring.

Combines accuracy trend, latency trend, prediction stability, drift score,
availability, resource usage, reliability and freshness into a single weighted
health score in ``[0, 1]`` and classifies it as EXCELLENT, HEALTHY, WARNING or
CRITICAL. All components are normalised so that 1.0 is perfectly healthy.
"""

from __future__ import annotations

import threading
from typing import Mapping, Optional, Sequence

import numpy as np

from monitoring.monitoring_models import (
    Clock,
    DeterministicIdGenerator,
    HealthLevel,
    HealthScore,
    IdGenerator,
    LogicalClock,
    ModelHealthStatus,
    MonitoringConfiguration,
    MonitoringError,
    ValidationError,
)

__all__ = ["ModelHealthMonitor", "create_model_health_monitor", "DEFAULT_HEALTH_WEIGHTS"]

DEFAULT_HEALTH_WEIGHTS = {
    "accuracy_trend": 0.20,
    "drift": 0.20,
    "prediction_stability": 0.15,
    "latency_trend": 0.10,
    "availability": 0.10,
    "reliability": 0.10,
    "resource_usage": 0.075,
    "freshness": 0.075,
}


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _slope(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    s = float(np.polyfit(np.arange(arr.size, dtype=float), arr, 1)[0])
    return 0.0 if abs(s) < 1e-12 else s


class ModelHealthMonitor:
    """Computes deterministic composite health scores for a model."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        weights: Optional[Mapping[str, float]] = None,
        freshness_budget_seconds: float = 3600.0,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="health")
        weights = dict(weights or DEFAULT_HEALTH_WEIGHTS)
        total = sum(weights.values())
        if total <= 0:
            raise ValidationError("health weights must sum to a positive value")
        self._weights = {k: v / total for k, v in weights.items()}
        self._freshness_budget = float(freshness_budget_seconds)
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._weights)

    def classify(self, score: float) -> HealthLevel:
        """Map a composite score in ``[0, 1]`` to a health level."""
        c = self._config
        if score >= c.health_excellent:
            return HealthLevel.EXCELLENT
        if score >= c.health_healthy:
            return HealthLevel.HEALTHY
        if score >= c.health_warning:
            return HealthLevel.WARNING
        return HealthLevel.CRITICAL

    def compute_components(
        self,
        *,
        accuracy_trend: Optional[Sequence[float]] = None,
        latency_trend: Optional[Sequence[float]] = None,
        drift_score: float = 0.0,
        prediction_stability: float = 1.0,
        availability: float = 1.0,
        resource_usage: float = 0.0,
        reliability: float = 1.0,
        freshness_seconds: Optional[float] = None,
    ) -> dict:
        """Return the normalised health components (each in ``[0, 1]``)."""
        # Accuracy: latest level adjusted by trend direction.
        if accuracy_trend:
            acc_latest = float(accuracy_trend[-1])
            acc_component = _clip01(acc_latest + 10.0 * _slope(accuracy_trend))
        else:
            acc_component = 1.0

        # Latency: lower is better; rising trend penalised.
        budget = max(self._config.latency_p95_threshold_ms, 1e-9)
        if latency_trend:
            lat_latest = float(latency_trend[-1])
            rising = max(0.0, _slope(latency_trend)) / budget
            lat_component = _clip01(1.0 - lat_latest / (2.0 * budget) - rising)
        else:
            lat_component = 1.0

        components = {
            "accuracy_trend": acc_component,
            "latency_trend": lat_component,
            "prediction_stability": _clip01(prediction_stability),
            "drift": _clip01(1.0 - drift_score),
            "availability": _clip01(availability),
            "resource_usage": _clip01(1.0 - resource_usage),
            "reliability": _clip01(reliability),
            "freshness": (
                1.0 if freshness_seconds is None
                else _clip01(1.0 - float(freshness_seconds) / self._freshness_budget)
            ),
        }
        return components

    def evaluate(
        self,
        *,
        accuracy_trend: Optional[Sequence[float]] = None,
        latency_trend: Optional[Sequence[float]] = None,
        drift_score: float = 0.0,
        prediction_stability: float = 1.0,
        availability: float = 1.0,
        resource_usage: float = 0.0,
        reliability: float = 1.0,
        freshness_seconds: Optional[float] = None,
    ) -> HealthScore:
        """Compute a composite :class:`HealthScore`."""
        components = self.compute_components(
            accuracy_trend=accuracy_trend,
            latency_trend=latency_trend,
            drift_score=drift_score,
            prediction_stability=prediction_stability,
            availability=availability,
            resource_usage=resource_usage,
            reliability=reliability,
            freshness_seconds=freshness_seconds,
        )
        overall = float(sum(self._weights[k] * components[k] for k in self._weights))
        overall = _clip01(overall)
        return HealthScore(
            overall=overall,
            level=self.classify(overall),
            components=components,
            weights=dict(self._weights),
            created_at=self._clock.now(),
        )

    def status(
        self, model_id: str, health: HealthScore, summary: str = ""
    ) -> ModelHealthStatus:
        """Wrap a health score into a :class:`ModelHealthStatus`."""
        return ModelHealthStatus(
            model_id=model_id,
            level=health.level,
            score=health.overall,
            created_at=self._clock.now(),
            summary=summary or f"Model {model_id} is {health.level.value}",
            health=health,
        )


def create_model_health_monitor(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> ModelHealthMonitor:
    """Factory for a configured :class:`ModelHealthMonitor`."""
    if deterministic:
        return ModelHealthMonitor(config=config, clock=LogicalClock(),
                                  id_generator=DeterministicIdGenerator(seed="health"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return ModelHealthMonitor(config=config, clock=SystemClock(),
                              id_generator=SequentialIdGenerator())