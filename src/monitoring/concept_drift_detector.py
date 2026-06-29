"""Deterministic concept-drift detection (pure NumPy).

Concept drift is detected from changes in model behaviour over time: rolling and
sliding-window performance, error-distribution drift, target / label-distribution
drift, residual drift and trend analysis. All methods are deterministic and
return :class:`ConceptDriftResult` value objects.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

import numpy as np

from monitoring.data_drift_detector import (
    DataDriftDetector,
    js_distance,
    population_stability_index,
)
from monitoring.monitoring_models import (
    Clock,
    ConceptDriftResult,
    DeterministicIdGenerator,
    DriftSeverity,
    IdGenerator,
    LogicalClock,
    MonitoringConfiguration,
    MonitoringError,
    ValidationError,
)

__all__ = ["ConceptDriftDetector", "create_concept_drift_detector"]


def _as_array(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty")
    return arr


class ConceptDriftDetector:
    """Detects concept drift from performance, error and label dynamics."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        performance_threshold: float = 0.05,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="concept")
        self._perf_threshold = float(performance_threshold)
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    # -- performance-based -------------------------------------------------- #
    def detect_performance_drift(
        self,
        reference_metric: float,
        current_metric: float,
        *,
        higher_is_better: bool = True,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect degradation between a reference and current scalar metric."""
        threshold = self._perf_threshold if threshold is None else float(threshold)
        delta = float(current_metric - reference_metric)
        degradation = -delta if higher_is_better else delta
        score = max(0.0, degradation)
        detected = score >= threshold
        return ConceptDriftResult(
            detected=detected,
            method="performance_drift",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            reference_metric=float(reference_metric),
            current_metric=float(current_metric),
            delta=delta,
            threshold=threshold,
            details={"higher_is_better": higher_is_better, "degradation": degradation},
        )

    def detect_rolling_performance(
        self,
        scores: Sequence[float],
        *,
        window: Optional[int] = None,
        higher_is_better: bool = True,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Compare the earliest and latest performance windows of a series."""
        arr = _as_array(scores, "scores")
        window = window or self._config.window_size
        window = max(1, min(window, arr.size))
        ref_window = arr[:window]
        cur_window = arr[-window:]
        ref_mean = float(np.mean(ref_window))
        cur_mean = float(np.mean(cur_window))
        slope = self.trend_slope(arr)
        threshold = self._perf_threshold if threshold is None else float(threshold)
        delta = cur_mean - ref_mean
        degradation = -delta if higher_is_better else delta
        score = max(0.0, degradation)
        return ConceptDriftResult(
            detected=score >= threshold,
            method="rolling_performance",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            reference_metric=ref_mean,
            current_metric=cur_mean,
            delta=delta,
            window_size=window,
            trend_slope=slope,
            threshold=threshold,
            details={"series_length": int(arr.size)},
        )

    def detect_sliding_window(
        self,
        correctness: Sequence[float],
        *,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect accuracy drift between early and late sliding windows."""
        arr = _as_array(correctness, "correctness")
        window = window or self._config.window_size
        window = max(1, min(window, arr.size))
        early = float(np.mean(arr[:window]))
        late = float(np.mean(arr[-window:]))
        threshold = self._perf_threshold if threshold is None else float(threshold)
        score = max(0.0, early - late)
        return ConceptDriftResult(
            detected=score >= threshold,
            method="sliding_window",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            reference_metric=early,
            current_metric=late,
            delta=late - early,
            window_size=window,
            trend_slope=self.trend_slope(arr),
            threshold=threshold,
        )

    # -- distribution-based ------------------------------------------------- #
    def detect_error_distribution(
        self,
        reference_errors: Sequence[float],
        current_errors: Sequence[float],
        *,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect drift in the distribution of model errors."""
        threshold = self._config.psi_threshold if threshold is None else float(threshold)
        score = population_stability_index(reference_errors, current_errors, self._config.num_bins)
        return ConceptDriftResult(
            detected=score >= threshold,
            method="error_distribution",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            reference_metric=float(np.mean(np.abs(np.asarray(reference_errors, dtype=float)))),
            current_metric=float(np.mean(np.abs(np.asarray(current_errors, dtype=float)))),
            threshold=threshold,
        )

    def detect_residual_drift(
        self,
        reference_residuals: Sequence[float],
        current_residuals: Sequence[float],
        *,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect drift in regression residuals via Jensen-Shannon distance."""
        threshold = self._config.js_threshold if threshold is None else float(threshold)
        score = js_distance(reference_residuals, current_residuals, self._config.num_bins)
        return ConceptDriftResult(
            detected=score >= threshold,
            method="residual_drift",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            threshold=threshold,
        )

    def detect_label_distribution(
        self,
        reference_labels: Sequence[Any],
        current_labels: Sequence[Any],
        *,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect drift in the label / target distribution."""
        threshold = self._config.psi_threshold if threshold is None else float(threshold)
        detector = DataDriftDetector(config=self._config)
        result = detector.detect_categorical("__label__", reference_labels, current_labels)
        return ConceptDriftResult(
            detected=result.drift_score >= threshold,
            method="label_distribution",
            drift_score=result.drift_score,
            severity=DataDriftDetector.classify_severity(result.drift_score, threshold),
            threshold=threshold,
            details={"categories": dict(result.details).get("categories", 0)},
        )

    def detect_target_drift(
        self,
        reference_targets: Sequence[float],
        current_targets: Sequence[float],
        *,
        threshold: Optional[float] = None,
    ) -> ConceptDriftResult:
        """Detect drift in a continuous target variable."""
        threshold = self._config.psi_threshold if threshold is None else float(threshold)
        score = population_stability_index(reference_targets, current_targets, self._config.num_bins)
        return ConceptDriftResult(
            detected=score >= threshold,
            method="target_drift",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, threshold),
            reference_metric=float(np.mean(np.asarray(reference_targets, dtype=float))),
            current_metric=float(np.mean(np.asarray(current_targets, dtype=float))),
            threshold=threshold,
        )

    # -- trend -------------------------------------------------------------- #
    @staticmethod
    def trend_slope(values: Sequence[float]) -> float:
        """Return the least-squares linear slope of a series (0 if too short)."""
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size, dtype=float)
        slope = np.polyfit(x, arr, 1)[0]
        if abs(slope) < 1e-12:
            return 0.0
        return float(slope)

    def analyze_trend(
        self, values: Sequence[float], *, threshold: float = 0.0
    ) -> ConceptDriftResult:
        """Classify the trend of a metric series."""
        arr = _as_array(values, "values")
        slope = self.trend_slope(arr)
        score = abs(slope)
        direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
        return ConceptDriftResult(
            detected=score > threshold,
            method="trend_analysis",
            drift_score=score,
            severity=DataDriftDetector.classify_severity(score, max(threshold, 1e-9)),
            trend_slope=slope,
            threshold=threshold,
            details={"direction": direction, "length": int(arr.size)},
        )


def create_concept_drift_detector(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> ConceptDriftDetector:
    """Factory for a configured :class:`ConceptDriftDetector`."""
    if deterministic:
        return ConceptDriftDetector(config=config, clock=LogicalClock(),
                                    id_generator=DeterministicIdGenerator(seed="concept"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return ConceptDriftDetector(config=config, clock=SystemClock(),
                                id_generator=SequentialIdGenerator())