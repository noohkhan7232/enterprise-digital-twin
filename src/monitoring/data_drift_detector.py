"""Deterministic data-drift detection algorithms (pure NumPy).

Implements Population Stability Index, Kullback-Leibler divergence,
Jensen-Shannon distance, the two-sample Kolmogorov-Smirnov test, histogram
distance, and numerical / categorical feature-distribution drift. Produces an
overall drift score, a feature drift ranking, severity classification and a
confidence score. All computations are deterministic and side-effect free.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from monitoring.monitoring_models import (
    Clock,
    DataDriftResult,
    DeterministicIdGenerator,
    DriftMethod,
    DriftSeverity,
    FeatureStatistics,
    FeatureType,
    IdGenerator,
    LogicalClock,
    MonitoringConfiguration,
    MonitoringError,
    ValidationError,
)

__all__ = [
    "DriftComputationError",
    "DataDriftDetector",
    "create_data_drift_detector",
    "population_stability_index",
    "kl_divergence",
    "js_distance",
    "ks_statistic",
    "histogram_distance",
]

_EPS = 1e-9


class DriftComputationError(MonitoringError):
    """Raised when a drift computation receives invalid inputs."""


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #
def _as_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValidationError(f"{name} must contain at least one non-NaN value")
    return arr


def _reference_edges(reference: np.ndarray, num_bins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(reference, probs))
    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        edges = np.array([center - 0.5, center + 0.5])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _binned_proportions(
    reference: np.ndarray, current: np.ndarray, num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    edges = _reference_edges(reference, num_bins)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    p = ref_hist.astype(float)
    q = cur_hist.astype(float)
    p = np.clip(p / max(p.sum(), _EPS), _EPS, None)
    q = np.clip(q / max(q.sum(), _EPS), _EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return p, q


def _categorical_proportions(
    reference: Sequence[Any], current: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ref = [str(v) for v in reference]
    cur = [str(v) for v in current]
    if not ref or not cur:
        raise ValidationError("categorical drift requires non-empty inputs")
    categories = sorted(set(ref) | set(cur))
    ref_counts = np.array([ref.count(c) for c in categories], dtype=float)
    cur_counts = np.array([cur.count(c) for c in categories], dtype=float)
    p = np.clip(ref_counts / max(ref_counts.sum(), _EPS), _EPS, None)
    q = np.clip(cur_counts / max(cur_counts.sum(), _EPS), _EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return p, q, categories


# --------------------------------------------------------------------------- #
# Core algorithms (operate on probability vectors or raw samples)
# --------------------------------------------------------------------------- #
def _psi_from_proportions(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((q - p) * np.log(q / p)))


def _kl_from_proportions(p: np.ndarray, q: np.ndarray) -> float:
    # KL(current || reference): how surprising current is given reference.
    return float(np.sum(q * np.log(q / p)))


def _js_distance_from_proportions(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)))
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(math.sqrt(max(jsd, 0.0)))


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], num_bins: int = 10
) -> float:
    """Population Stability Index between two numeric samples."""
    ref = _as_float_array(reference, "reference")
    cur = _as_float_array(current, "current")
    p, q = _binned_proportions(ref, cur, num_bins)
    return _psi_from_proportions(p, q)


def kl_divergence(
    reference: Sequence[float], current: Sequence[float], num_bins: int = 10
) -> float:
    """Kullback-Leibler divergence KL(current || reference)."""
    ref = _as_float_array(reference, "reference")
    cur = _as_float_array(current, "current")
    p, q = _binned_proportions(ref, cur, num_bins)
    return _kl_from_proportions(p, q)


def js_distance(
    reference: Sequence[float], current: Sequence[float], num_bins: int = 10
) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence, base 2) in [0, 1]."""
    ref = _as_float_array(reference, "reference")
    cur = _as_float_array(current, "current")
    p, q = _binned_proportions(ref, cur, num_bins)
    return _js_distance_from_proportions(p, q)


def _ks_pvalue(lam: float) -> float:
    if lam <= 0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        total += (-1.0) ** (k - 1) * math.exp(-2.0 * (k ** 2) * (lam ** 2))
    return float(min(1.0, max(0.0, 2.0 * total)))


def ks_statistic(
    reference: Sequence[float], current: Sequence[float]
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    ref = np.sort(_as_float_array(reference, "reference"))
    cur = np.sort(_as_float_array(current, "current"))
    grid = np.sort(np.concatenate([ref, cur]))
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, grid, side="right") / cur.size
    d = float(np.max(np.abs(cdf_ref - cdf_cur)))
    n1, n2 = ref.size, cur.size
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p_value = _ks_pvalue((en + 0.12 + 0.11 / en) * d)
    return d, p_value


def histogram_distance(
    reference: Sequence[float], current: Sequence[float], num_bins: int = 10
) -> float:
    """Total-variation distance between two binned numeric samples in [0, 1]."""
    ref = _as_float_array(reference, "reference")
    cur = _as_float_array(current, "current")
    p, q = _binned_proportions(ref, cur, num_bins)
    return float(0.5 * np.sum(np.abs(p - q)))


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class DataDriftDetector:
    """Computes feature- and dataset-level data drift deterministically."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="drift")
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    # -- thresholds & classification --------------------------------------- #
    def _threshold_for(self, method: DriftMethod) -> float:
        return {
            DriftMethod.PSI: self._config.psi_threshold,
            DriftMethod.KL_DIVERGENCE: self._config.kl_threshold,
            DriftMethod.JS_DISTANCE: self._config.js_threshold,
            DriftMethod.KS_TEST: self._config.ks_threshold,
            DriftMethod.HISTOGRAM_DISTANCE: self._config.js_threshold,
            DriftMethod.CATEGORICAL: self._config.psi_threshold,
        }.get(method, self._config.drift_threshold)

    @staticmethod
    def classify_severity(score: float, threshold: float) -> DriftSeverity:
        """Map a drift score to a severity band relative to its threshold."""
        if threshold <= 0:
            threshold = _EPS
        if score < 0.5 * threshold:
            return DriftSeverity.NONE
        if score < threshold:
            return DriftSeverity.LOW
        if score < 2.0 * threshold:
            return DriftSeverity.MODERATE
        if score < 4.0 * threshold:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

    @staticmethod
    def confidence_score(reference_size: int, current_size: int) -> float:
        """Confidence in a drift verdict, increasing with sample size."""
        n = max(0, min(reference_size, current_size))
        return float(1.0 - 1.0 / (1.0 + math.sqrt(n))) if n > 0 else 0.0

    # -- feature statistics ------------------------------------------------- #
    def compute_feature_statistics(
        self,
        feature_name: str,
        values: Sequence[Any],
        feature_type: FeatureType = FeatureType.NUMERICAL,
        num_bins: Optional[int] = None,
    ) -> FeatureStatistics:
        """Compute summary statistics for a single feature."""
        num_bins = num_bins or self._config.num_bins
        if feature_type is FeatureType.CATEGORICAL:
            raw = [str(v) for v in values]
            categories = sorted(set(raw))
            freq = {c: raw.count(c) / len(raw) for c in categories} if raw else {}
            return FeatureStatistics(
                feature_name=feature_name,
                feature_type=FeatureType.CATEGORICAL,
                count=len(raw),
                unique_count=len(categories),
                categories=freq,
            )
        arr = np.asarray(values, dtype=float)
        missing = int(np.isnan(arr).sum())
        clean = arr[~np.isnan(arr)]
        if clean.size == 0:
            return FeatureStatistics(
                feature_name=feature_name, count=int(arr.size),
                missing_count=missing, missing_rate=1.0 if arr.size else 0.0,
            )
        hist, edges = np.histogram(clean, bins=num_bins)
        return FeatureStatistics(
            feature_name=feature_name,
            feature_type=FeatureType.NUMERICAL,
            count=int(arr.size),
            mean=float(np.mean(clean)),
            std=float(np.std(clean)),
            minimum=float(np.min(clean)),
            maximum=float(np.max(clean)),
            median=float(np.median(clean)),
            q25=float(np.quantile(clean, 0.25)),
            q75=float(np.quantile(clean, 0.75)),
            missing_count=missing,
            missing_rate=float(missing / arr.size) if arr.size else 0.0,
            unique_count=int(np.unique(clean).size),
            histogram=tuple(float(h) for h in hist),
            bin_edges=tuple(float(e) for e in edges),
        )

    # -- single-feature drift ---------------------------------------------- #
    def detect_numerical(
        self,
        feature_name: str,
        reference: Sequence[float],
        current: Sequence[float],
        method: DriftMethod = DriftMethod.PSI,
    ) -> DataDriftResult:
        ref = _as_float_array(reference, "reference")
        cur = _as_float_array(current, "current")
        bins = self._config.num_bins
        p_value: Optional[float] = None
        if method is DriftMethod.PSI:
            score = population_stability_index(ref, cur, bins)
        elif method is DriftMethod.KL_DIVERGENCE:
            score = kl_divergence(ref, cur, bins)
        elif method is DriftMethod.JS_DISTANCE:
            score = js_distance(ref, cur, bins)
        elif method is DriftMethod.HISTOGRAM_DISTANCE:
            score = histogram_distance(ref, cur, bins)
        elif method is DriftMethod.KS_TEST:
            score, p_value = ks_statistic(ref, cur)
        else:
            raise ValidationError(f"Unsupported numerical drift method: {method}")
        threshold = self._threshold_for(method)
        drifted = score >= threshold
        severity = self.classify_severity(score, threshold)
        return DataDriftResult(
            feature_name=feature_name,
            method=method,
            drift_score=score,
            threshold=threshold,
            drifted=drifted,
            severity=severity,
            reference_size=int(ref.size),
            current_size=int(cur.size),
            p_value=p_value,
            details={"confidence": self.confidence_score(ref.size, cur.size)},
        )

    def detect_categorical(
        self,
        feature_name: str,
        reference: Sequence[Any],
        current: Sequence[Any],
    ) -> DataDriftResult:
        p, q, categories = _categorical_proportions(reference, current)
        score = _psi_from_proportions(p, q)
        threshold = self._threshold_for(DriftMethod.CATEGORICAL)
        return DataDriftResult(
            feature_name=feature_name,
            method=DriftMethod.CATEGORICAL,
            drift_score=score,
            threshold=threshold,
            drifted=score >= threshold,
            severity=self.classify_severity(score, threshold),
            reference_size=len(reference),
            current_size=len(current),
            details={"categories": len(categories),
                     "confidence": self.confidence_score(len(reference), len(current))},
        )

    def detect_feature(
        self,
        feature_name: str,
        reference: Sequence[Any],
        current: Sequence[Any],
        feature_type: FeatureType = FeatureType.NUMERICAL,
        method: DriftMethod = DriftMethod.PSI,
    ) -> DataDriftResult:
        if feature_type is FeatureType.CATEGORICAL:
            return self.detect_categorical(feature_name, reference, current)
        return self.detect_numerical(feature_name, reference, current, method)

    # -- dataset-level drift ----------------------------------------------- #
    def detect_dataset(
        self,
        reference: Mapping[str, Sequence[Any]],
        current: Mapping[str, Sequence[Any]],
        feature_types: Optional[Mapping[str, FeatureType]] = None,
        method: DriftMethod = DriftMethod.PSI,
    ) -> List[DataDriftResult]:
        """Run drift detection on every shared feature, ordered by name."""
        if not reference:
            raise ValidationError("reference dataset is empty")
        feature_types = feature_types or {}
        shared = sorted(set(reference) & set(current))
        if not shared:
            raise ValidationError("reference and current share no features")
        results: List[DataDriftResult] = []
        for name in shared:
            ftype = feature_types.get(name, FeatureType.NUMERICAL)
            results.append(self.detect_feature(name, reference[name], current[name], ftype, method))
        return results

    # -- aggregation -------------------------------------------------------- #
    @staticmethod
    def overall_drift_score(results: Sequence[DataDriftResult]) -> float:
        if not results:
            return 0.0
        return float(np.mean([r.drift_score for r in results]))

    @staticmethod
    def feature_drift_ranking(
        results: Sequence[DataDriftResult],
    ) -> List[Tuple[str, float]]:
        """Return ``(feature, score)`` pairs ordered by descending drift."""
        ranked = sorted(results, key=lambda r: (-r.drift_score, r.feature_name))
        return [(r.feature_name, r.drift_score) for r in ranked]

    @staticmethod
    def drifted_features(results: Sequence[DataDriftResult]) -> List[str]:
        return sorted(r.feature_name for r in results if r.drifted)

    def summarize(self, results: Sequence[DataDriftResult]) -> Dict[str, Any]:
        """Produce a deterministic drift summary for a set of results."""
        ranking = self.feature_drift_ranking(results)
        return {
            "overall_drift_score": self.overall_drift_score(results),
            "feature_count": len(results),
            "drifted_count": len(self.drifted_features(results)),
            "ranking": ranking,
            "top_feature": ranking[0][0] if ranking else None,
        }


def create_data_drift_detector(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> DataDriftDetector:
    """Factory for a configured :class:`DataDriftDetector`."""
    if deterministic:
        return DataDriftDetector(config=config, clock=LogicalClock(),
                                 id_generator=DeterministicIdGenerator(seed="drift"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return DataDriftDetector(config=config, clock=SystemClock(),
                             id_generator=SequentialIdGenerator())