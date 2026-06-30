"""Deterministic metrics collection, aggregation and analysis engine.

Collects metrics across application, inference, workflow, scheduler, deployment,
CI/CD, monitoring and business-KPI categories, and supports aggregation, rolling
windows, percentiles, trend analysis and JSON export. Pure Python + NumPy,
thread-safe, deterministic (time is injected).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .observability_models import (
    Clock, MetricPoint, MetricSeries, MetricType, ValidationError, percentile,
)

__all__ = ["MetricCategory", "MetricsEngine", "create_metrics_engine"]

# The eight required metric categories.
MetricCategory = (
    "application", "inference", "workflow", "scheduler",
    "deployment", "cicd", "monitoring", "business",
)


class MetricsEngine:
    """Thread-safe, deterministic collector and analyser of metric series."""

    def __init__(self, *, clock: Optional[Clock] = None,
                 categories: Sequence[str] = MetricCategory) -> None:
        self._clock = clock or Clock()
        self._categories = tuple(categories)
        self._series: Dict[str, List[MetricPoint]] = {}
        self._types: Dict[str, MetricType] = {}
        self._units: Dict[str, str] = {}
        self._category_of: Dict[str, str] = {}
        self._lock = threading.RLock()

    # -- recording ---------------------------------------------------------- #
    def record(self, name: str, value: float, *, category: str = "application",
               metric_type: MetricType = MetricType.GAUGE, unit: str = "",
               timestamp: Optional[float] = None,
               labels: Optional[Mapping[str, Any]] = None) -> MetricPoint:
        if not name:
            raise ValidationError("metric name must be non-empty")
        if category not in self._categories:
            raise ValidationError(f"unknown category: {category}")
        ts = self._clock.now() if timestamp is None else float(timestamp)
        point = MetricPoint(ts, value, labels or {})
        with self._lock:
            self._series.setdefault(name, []).append(point)
            self._types.setdefault(name, metric_type)
            self._units.setdefault(name, unit)
            self._category_of.setdefault(name, category)
        return point

    def record_many(self, name: str, values: Sequence[float], **kwargs: Any) -> int:
        for value in values:
            self.record(name, value, **kwargs)
        return len(values)

    # -- retrieval ---------------------------------------------------------- #
    def series(self, name: str) -> MetricSeries:
        with self._lock:
            if name not in self._series:
                raise ValidationError(f"unknown metric: {name}")
            return MetricSeries(name, self._types[name], tuple(self._series[name]), self._units[name])

    def names(self, *, category: Optional[str] = None) -> Tuple[str, ...]:
        with self._lock:
            if category is None:
                return tuple(sorted(self._series))
            return tuple(sorted(n for n in self._series if self._category_of.get(n) == category))

    def window(self, name: str, *, start: float = float("-inf"),
               end: float = float("inf")) -> MetricSeries:
        """Return a series restricted to ``start <= timestamp <= end``."""
        full = self.series(name)
        pts = tuple(p for p in full.points if start <= p.timestamp <= end)
        return MetricSeries(name, full.metric_type, pts, full.unit)

    def rolling_window(self, name: str, size: int) -> MetricSeries:
        """Return the most recent *size* points as a series."""
        if size <= 0:
            raise ValidationError("window size must be positive")
        full = self.series(name)
        return MetricSeries(name, full.metric_type, full.points[-size:], full.unit)

    # -- aggregation -------------------------------------------------------- #
    def aggregate(self, name: str) -> Dict[str, float]:
        series = self.series(name)
        values = series.values
        if not values:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
                    "p50": 0.0, "p95": 0.0, "p99": 0.0, "stddev": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "sum": round(float(arr.sum()), 6),
            "mean": round(float(arr.mean()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "stddev": round(float(arr.std()), 6),
        }

    def percentiles(self, name: str,
                    pcts: Sequence[float] = (50.0, 95.0, 99.0)) -> Dict[str, float]:
        values = self.series(name).values
        return {f"p{int(p)}": percentile(values, p) for p in pcts}

    def trend(self, name: str) -> Dict[str, Any]:
        """Least-squares slope of value against time; near-zero snaps to flat."""
        series = self.series(name)
        if series.count < 2:
            return {"slope": 0.0, "direction": "flat", "delta": 0.0}
        xs = np.asarray([p.timestamp for p in series.points], dtype=float)
        ys = np.asarray(series.values, dtype=float)
        xs = xs - xs[0]
        denom = float(((xs - xs.mean()) ** 2).sum())
        slope = 0.0 if denom == 0 else float(((xs - xs.mean()) * (ys - ys.mean())).sum() / denom)
        if abs(slope) < 1e-9:
            slope = 0.0
        direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
        return {"slope": round(slope, 6), "direction": direction,
                "delta": round(float(ys[-1] - ys[0]), 6)}

    # -- export ------------------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            names = sorted(self._series)
        return {name: self.aggregate(name) for name in names}

    def category_summary(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {c: {} for c in self._categories}
        for name in self.names():
            cat = self._category_of.get(name, "application")
            out.setdefault(cat, {})[name] = self.aggregate(name)
        return out

    def export(self) -> Dict[str, Any]:
        return {name: self.series(name).to_dict() for name in self.names()}


def create_metrics_engine(*, clock: Optional[Clock] = None) -> MetricsEngine:
    return MetricsEngine(clock=clock)