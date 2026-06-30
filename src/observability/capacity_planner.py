"""Deterministic capacity forecasting.

Forecasts CPU, memory, storage, request volume, model growth and data growth
using deterministic algorithms (least-squares linear projection and compound
growth), and recommends provisioned capacity with optional exhaustion
detection. Pure Python + NumPy, deterministic, thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .observability_models import CapacityForecast, ResourceKind, ValidationError

__all__ = ["ForecastStrategy", "CapacityPlanner", "create_capacity_planner"]

# Forecasting strategies (Strategy pattern).
ForecastStrategy = str  # "linear" | "compound"


def _linear_forecast(history: np.ndarray, horizon: int) -> Tuple[np.ndarray, float]:
    n = history.size
    xs = np.arange(n, dtype=float)
    denom = float(((xs - xs.mean()) ** 2).sum())
    if denom == 0:
        slope = 0.0
        intercept = float(history.mean())
    else:
        slope = float(((xs - xs.mean()) * (history - history.mean())).sum() / denom)
        intercept = float(history.mean() - slope * xs.mean())
    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = intercept + slope * future_x
    return np.maximum(forecast, 0.0), slope


def _compound_forecast(history: np.ndarray, horizon: int) -> Tuple[np.ndarray, float]:
    n = history.size
    first = float(history[0])
    last = float(history[-1])
    if n < 2 or first <= 0:
        rate = 0.0
    else:
        # Geometric mean growth rate per step.
        ratio = last / first if first > 0 else 1.0
        rate = ratio ** (1.0 / (n - 1)) - 1.0 if ratio > 0 else 0.0
    forecast = np.array([last * ((1.0 + rate) ** (i + 1)) for i in range(horizon)], dtype=float)
    return np.maximum(forecast, 0.0), rate


class CapacityPlanner:
    """Produces deterministic capacity forecasts per resource."""

    def __init__(self, *, headroom: float = 0.25) -> None:
        if headroom < 0:
            raise ValidationError("headroom must be non-negative")
        self._headroom = float(headroom)
        self._lock = threading.RLock()

    def forecast(self, resource: ResourceKind, history: Sequence[float], *,
                 horizon: int = 12, capacity_limit: Optional[float] = None,
                 strategy: ForecastStrategy = "linear", unit: str = "") -> CapacityForecast:
        if horizon <= 0:
            raise ValidationError("horizon must be positive")
        if len(history) < 1:
            raise ValidationError("history must be non-empty")
        if not isinstance(resource, ResourceKind):
            resource = ResourceKind(resource)
        arr = np.asarray([float(v) for v in history], dtype=float)
        with self._lock:
            if strategy == "linear":
                values, rate = _linear_forecast(arr, horizon)
            elif strategy == "compound":
                values, rate = _compound_forecast(arr, horizon)
            else:
                raise ValidationError(f"unknown strategy: {strategy}")
        forecast_vals = tuple(float(v) for v in values)
        peak = max(forecast_vals) if forecast_vals else float(arr[-1])
        recommended = round(peak * (1.0 + self._headroom), 6)
        exhaustion_step: Optional[int] = None
        if capacity_limit is not None:
            for i, v in enumerate(forecast_vals):
                if v >= capacity_limit:
                    exhaustion_step = i
                    break
        return CapacityForecast(
            resource=resource, current=float(arr[-1]), horizon=horizon,
            forecast=forecast_vals, growth_rate=round(float(rate), 6),
            recommended_capacity=recommended, exhaustion_step=exhaustion_step, unit=unit,
        )

    def forecast_all(self, histories: Mapping[ResourceKind, Sequence[float]],
                     **kwargs: Any) -> Dict[str, CapacityForecast]:
        out: Dict[str, CapacityForecast] = {}
        for resource, history in histories.items():
            kind = resource if isinstance(resource, ResourceKind) else ResourceKind(resource)
            out[kind.value] = self.forecast(kind, history, **kwargs)
        return out

    def export(self, forecasts: Mapping[str, CapacityForecast]) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in sorted(forecasts.items())}


def create_capacity_planner(*, headroom: float = 0.25) -> CapacityPlanner:
    return CapacityPlanner(headroom=headroom)