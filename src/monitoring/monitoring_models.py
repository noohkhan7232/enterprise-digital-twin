"""Immutable domain models for the Enterprise Production Monitoring subsystem.

This module defines the deterministic, JSON-serializable value objects shared
across data-drift, concept-drift, prediction, health, data-quality and dashboard
components, together with a small deterministic infrastructure (clocks and id
generators).

Design rules:

* Every aggregate is ``@dataclass(frozen=True, slots=True)`` — immutable,
  hashable and JSON round-trippable through ``to_dict`` / ``from_dict``.
* No wall-clock or randomness is consumed here; time and identity are injected.
* Mapping-like fields are stored as deterministic, sorted tuples of pairs.
* The subsystem is self-contained (pure Python + NumPy) and integrates with the
  MLOps platform, registry, schedulers and event bus *by composition* — it
  accepts injected collaborators and plain values, importing none of them.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "MonitoringError",
    "ValidationError",
    "SerializationError",
    "FeatureType",
    "DriftMethod",
    "DriftSeverity",
    "AlertLevel",
    "AlertType",
    "HealthLevel",
    "QualityIssueType",
    "Comparison",
    "FeatureStatistics",
    "DistributionSnapshot",
    "DataDriftResult",
    "ConceptDriftResult",
    "PredictionStatistics",
    "PredictionDriftResult",
    "ModelHealthStatus",
    "HealthScore",
    "MonitoringAlert",
    "MonitoringReport",
    "QualityMetrics",
    "QualityIssue",
    "DashboardSnapshot",
    "MonitoringStatistics",
    "MonitoringConfiguration",
    "AlertPolicy",
    "Clock",
    "SystemClock",
    "FixedClock",
    "LogicalClock",
    "IdGenerator",
    "SequentialIdGenerator",
    "DeterministicIdGenerator",
    "freeze_mapping",
    "thaw_mapping",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class MonitoringError(Exception):
    """Base exception for the monitoring subsystem."""


class ValidationError(MonitoringError):
    """Raised when a domain invariant is violated."""


class SerializationError(MonitoringError):
    """Raised when a payload cannot be deserialised."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class FeatureType(str, Enum):
    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"


class DriftMethod(str, Enum):
    PSI = "PSI"
    JS_DISTANCE = "JS_DISTANCE"
    KL_DIVERGENCE = "KL_DIVERGENCE"
    KS_TEST = "KS_TEST"
    HISTOGRAM_DISTANCE = "HISTOGRAM_DISTANCE"
    CATEGORICAL = "CATEGORICAL"


class DriftSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    THRESHOLD = "THRESHOLD"
    TREND = "TREND"
    COMPOSITE = "COMPOSITE"
    REPEATED = "REPEATED"


class HealthLevel(str, Enum):
    EXCELLENT = "EXCELLENT"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class QualityIssueType(str, Enum):
    MISSING_VALUES = "MISSING_VALUES"
    DUPLICATE_RECORDS = "DUPLICATE_RECORDS"
    INVALID_RANGE = "INVALID_RANGE"
    OUTLIER = "OUTLIER"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    NULL_PERCENTAGE = "NULL_PERCENTAGE"
    FRESHNESS = "FRESHNESS"
    CONSISTENCY = "CONSISTENCY"
    COMPLETENESS = "COMPLETENESS"


class Comparison(str, Enum):
    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"


# --------------------------------------------------------------------------- #
# Collection / scalar helpers
# --------------------------------------------------------------------------- #
_SCALAR_TYPES = (str, int, float, bool, type(None))
FrozenMap = Tuple[Tuple[str, Any], ...]


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise ValidationError(f"Unsupported non-scalar value of type {type(value).__name__!r}")


def freeze_mapping(data: Optional[Mapping[str, Any]]) -> FrozenMap:
    if data is None:
        return ()
    if isinstance(data, tuple):
        items = [(str(k), _coerce_scalar(v)) for k, v in data]
    else:
        items = [(str(k), _coerce_scalar(v)) for k, v in dict(data).items()]
    return tuple(sorted(items, key=lambda kv: kv[0]))


def thaw_mapping(frozen: FrozenMap) -> Dict[str, Any]:
    return {k: v for k, v in frozen}


def _float_tuple(values: Optional[Iterable[Any]]) -> Tuple[float, ...]:
    if values is None:
        return ()
    return tuple(float(v) for v in values)


def _str_tuple(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(v) for v in values)


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name!r} must be a non-empty string")


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


# --------------------------------------------------------------------------- #
# Deterministic infrastructure
# --------------------------------------------------------------------------- #
_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@runtime_checkable
class Clock(Protocol):
    def now(self) -> str: ...


class SystemClock:
    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class FixedClock:
    def __init__(self, instant: str = "2024-01-01T00:00:00+00:00") -> None:
        self._instant = instant

    def now(self) -> str:
        return self._instant


class LogicalClock:
    def __init__(self, start: int = 0, step_seconds: int = 1) -> None:
        if step_seconds <= 0:
            raise ValidationError("LogicalClock.step_seconds must be positive")
        self._counter = itertools.count(start)
        self._step = step_seconds
        self._lock = threading.Lock()

    def now(self) -> str:
        with self._lock:
            tick = next(self._counter)
        return (_EPOCH + timedelta(seconds=tick * self._step)).isoformat()


@runtime_checkable
class IdGenerator(Protocol):
    def generate(self, prefix: str) -> str: ...


class SequentialIdGenerator:
    def __init__(self, width: int = 6, start: int = 1) -> None:
        if width <= 0:
            raise ValidationError("SequentialIdGenerator.width must be positive")
        self._counters: Dict[str, "itertools.count[int]"] = {}
        self._width = width
        self._start = start
        self._lock = threading.Lock()

    def generate(self, prefix: str) -> str:
        with self._lock:
            counter = self._counters.get(prefix)
            if counter is None:
                counter = itertools.count(self._start)
                self._counters[prefix] = counter
            value = next(counter)
        return f"{prefix}-{value:0{self._width}d}"


class DeterministicIdGenerator:
    def __init__(self, seed: str = "monitoring", length: int = 12) -> None:
        if length <= 0 or length > 64:
            raise ValidationError("DeterministicIdGenerator.length must be in 1..64")
        self._seed = seed
        self._length = length
        self._counters: Dict[str, "itertools.count[int]"] = {}
        self._lock = threading.Lock()

    def generate(self, prefix: str) -> str:
        with self._lock:
            counter = self._counters.get(prefix)
            if counter is None:
                counter = itertools.count(0)
                self._counters[prefix] = counter
            value = next(counter)
        digest = hashlib.sha256(f"{self._seed}:{prefix}:{value}".encode("utf-8")).hexdigest()
        return f"{prefix}-{digest[: self._length]}"


# --------------------------------------------------------------------------- #
# Feature & distribution models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    """Summary statistics for a single feature."""

    feature_name: str
    feature_type: FeatureType = FeatureType.NUMERICAL
    count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    missing_count: int = 0
    missing_rate: float = 0.0
    unique_count: int = 0
    histogram: Tuple[float, ...] = ()
    bin_edges: Tuple[float, ...] = ()
    categories: FrozenMap = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.feature_name, "feature_name")
        if not isinstance(self.feature_type, FeatureType):
            object.__setattr__(self, "feature_type", FeatureType(self.feature_type))
        object.__setattr__(self, "histogram", _float_tuple(self.histogram))
        object.__setattr__(self, "bin_edges", _float_tuple(self.bin_edges))
        object.__setattr__(self, "categories", freeze_mapping(self.categories))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type.value,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "histogram": list(self.histogram),
            "bin_edges": list(self.bin_edges),
            "categories": thaw_mapping(self.categories),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureStatistics":
        return cls(
            feature_name=data["feature_name"],
            feature_type=FeatureType(data.get("feature_type", "NUMERICAL")),
            count=int(data.get("count", 0)),
            mean=_opt_float(data.get("mean")),
            std=_opt_float(data.get("std")),
            minimum=_opt_float(data.get("minimum")),
            maximum=_opt_float(data.get("maximum")),
            median=_opt_float(data.get("median")),
            q25=_opt_float(data.get("q25")),
            q75=_opt_float(data.get("q75")),
            missing_count=int(data.get("missing_count", 0)),
            missing_rate=float(data.get("missing_rate", 0.0)),
            unique_count=int(data.get("unique_count", 0)),
            histogram=_float_tuple(data.get("histogram", ())),
            bin_edges=_float_tuple(data.get("bin_edges", ())),
            categories=freeze_mapping(data.get("categories", {})),
        )


@dataclass(frozen=True, slots=True)
class DistributionSnapshot:
    """A snapshot of a dataset's per-feature distributions."""

    snapshot_id: str
    feature_stats: Tuple[FeatureStatistics, ...] = ()
    sample_size: int = 0
    created_at: str = ""
    label_distribution: FrozenMap = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_id, "snapshot_id")
        object.__setattr__(self, "feature_stats", tuple(self.feature_stats))
        object.__setattr__(self, "label_distribution", freeze_mapping(self.label_distribution))

    def feature(self, name: str) -> Optional[FeatureStatistics]:
        for stat in self.feature_stats:
            if stat.feature_name == name:
                return stat
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "feature_stats": [s.to_dict() for s in self.feature_stats],
            "sample_size": self.sample_size,
            "created_at": self.created_at,
            "label_distribution": thaw_mapping(self.label_distribution),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistributionSnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            feature_stats=tuple(FeatureStatistics.from_dict(s) for s in data.get("feature_stats", [])),
            sample_size=int(data.get("sample_size", 0)),
            created_at=data.get("created_at", ""),
            label_distribution=freeze_mapping(data.get("label_distribution", {})),
        )


# --------------------------------------------------------------------------- #
# Drift results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DataDriftResult:
    """The result of a single feature drift test."""

    feature_name: str
    method: DriftMethod
    drift_score: float
    threshold: float
    drifted: bool
    severity: DriftSeverity
    reference_size: int = 0
    current_size: int = 0
    p_value: Optional[float] = None
    details: FrozenMap = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.feature_name, "feature_name")
        if not isinstance(self.method, DriftMethod):
            object.__setattr__(self, "method", DriftMethod(self.method))
        if not isinstance(self.severity, DriftSeverity):
            object.__setattr__(self, "severity", DriftSeverity(self.severity))
        object.__setattr__(self, "drift_score", float(self.drift_score))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "drifted", bool(self.drifted))
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "method": self.method.value,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "severity": self.severity.value,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "p_value": self.p_value,
            "details": thaw_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataDriftResult":
        return cls(
            feature_name=data["feature_name"],
            method=DriftMethod(data["method"]),
            drift_score=float(data["drift_score"]),
            threshold=float(data["threshold"]),
            drifted=bool(data["drifted"]),
            severity=DriftSeverity(data["severity"]),
            reference_size=int(data.get("reference_size", 0)),
            current_size=int(data.get("current_size", 0)),
            p_value=_opt_float(data.get("p_value")),
            details=freeze_mapping(data.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class ConceptDriftResult:
    """The result of a concept-drift evaluation."""

    detected: bool
    method: str
    drift_score: float
    severity: DriftSeverity
    reference_metric: Optional[float] = None
    current_metric: Optional[float] = None
    delta: float = 0.0
    window_size: int = 0
    trend_slope: float = 0.0
    threshold: float = 0.0
    details: FrozenMap = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, DriftSeverity):
            object.__setattr__(self, "severity", DriftSeverity(self.severity))
        object.__setattr__(self, "detected", bool(self.detected))
        object.__setattr__(self, "drift_score", float(self.drift_score))
        object.__setattr__(self, "delta", float(self.delta))
        object.__setattr__(self, "trend_slope", float(self.trend_slope))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "method": self.method,
            "drift_score": self.drift_score,
            "severity": self.severity.value,
            "reference_metric": self.reference_metric,
            "current_metric": self.current_metric,
            "delta": self.delta,
            "window_size": self.window_size,
            "trend_slope": self.trend_slope,
            "threshold": self.threshold,
            "details": thaw_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConceptDriftResult":
        return cls(
            detected=bool(data["detected"]),
            method=data["method"],
            drift_score=float(data["drift_score"]),
            severity=DriftSeverity(data["severity"]),
            reference_metric=_opt_float(data.get("reference_metric")),
            current_metric=_opt_float(data.get("current_metric")),
            delta=float(data.get("delta", 0.0)),
            window_size=int(data.get("window_size", 0)),
            trend_slope=float(data.get("trend_slope", 0.0)),
            threshold=float(data.get("threshold", 0.0)),
            details=freeze_mapping(data.get("details", {})),
        )


# --------------------------------------------------------------------------- #
# Prediction models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PredictionStatistics:
    """Aggregate statistics over a batch of predictions."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    variance: float = 0.0
    q05: float = 0.0
    q50: float = 0.0
    q95: float = 0.0
    positive_rate: Optional[float] = None
    confidence_mean: Optional[float] = None
    confidence_std: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_per_s: Optional[float] = None
    error_rate: float = 0.0
    success_rate: float = 1.0
    inference_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "variance": self.variance,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
            "positive_rate": self.positive_rate,
            "confidence_mean": self.confidence_mean,
            "confidence_std": self.confidence_std,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_per_s": self.throughput_per_s,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "inference_time_ms": self.inference_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionStatistics":
        return cls(
            count=int(data.get("count", 0)),
            mean=float(data.get("mean", 0.0)),
            std=float(data.get("std", 0.0)),
            minimum=float(data.get("minimum", 0.0)),
            maximum=float(data.get("maximum", 0.0)),
            variance=float(data.get("variance", 0.0)),
            q05=float(data.get("q05", 0.0)),
            q50=float(data.get("q50", 0.0)),
            q95=float(data.get("q95", 0.0)),
            positive_rate=_opt_float(data.get("positive_rate")),
            confidence_mean=_opt_float(data.get("confidence_mean")),
            confidence_std=_opt_float(data.get("confidence_std")),
            latency_p50_ms=_opt_float(data.get("latency_p50_ms")),
            latency_p95_ms=_opt_float(data.get("latency_p95_ms")),
            latency_p99_ms=_opt_float(data.get("latency_p99_ms")),
            throughput_per_s=_opt_float(data.get("throughput_per_s")),
            error_rate=float(data.get("error_rate", 0.0)),
            success_rate=float(data.get("success_rate", 1.0)),
            inference_time_ms=_opt_float(data.get("inference_time_ms")),
        )


@dataclass(frozen=True, slots=True)
class PredictionDriftResult:
    """Drift between a reference and current prediction distribution."""

    drift_score: float
    method: DriftMethod
    severity: DriftSeverity
    drifted: bool
    threshold: float = 0.0
    reference_size: int = 0
    current_size: int = 0
    details: FrozenMap = ()

    def __post_init__(self) -> None:
        if not isinstance(self.method, DriftMethod):
            object.__setattr__(self, "method", DriftMethod(self.method))
        if not isinstance(self.severity, DriftSeverity):
            object.__setattr__(self, "severity", DriftSeverity(self.severity))
        object.__setattr__(self, "drift_score", float(self.drift_score))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "drifted", bool(self.drifted))
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "method": self.method.value,
            "severity": self.severity.value,
            "drifted": self.drifted,
            "threshold": self.threshold,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "details": thaw_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionDriftResult":
        return cls(
            drift_score=float(data["drift_score"]),
            method=DriftMethod(data["method"]),
            severity=DriftSeverity(data["severity"]),
            drifted=bool(data["drifted"]),
            threshold=float(data.get("threshold", 0.0)),
            reference_size=int(data.get("reference_size", 0)),
            current_size=int(data.get("current_size", 0)),
            details=freeze_mapping(data.get("details", {})),
        )


# --------------------------------------------------------------------------- #
# Health models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HealthScore:
    """A composite model-health score with component breakdown."""

    overall: float
    level: HealthLevel
    components: FrozenMap = ()
    weights: FrozenMap = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.level, HealthLevel):
            object.__setattr__(self, "level", HealthLevel(self.level))
        object.__setattr__(self, "overall", float(self.overall))
        object.__setattr__(self, "components", freeze_mapping(self.components))
        object.__setattr__(self, "weights", freeze_mapping(self.weights))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "level": self.level.value,
            "components": thaw_mapping(self.components),
            "weights": thaw_mapping(self.weights),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HealthScore":
        return cls(
            overall=float(data["overall"]),
            level=HealthLevel(data["level"]),
            components=freeze_mapping(data.get("components", {})),
            weights=freeze_mapping(data.get("weights", {})),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True, slots=True)
class ModelHealthStatus:
    """The health status of a monitored model at a point in time."""

    model_id: str
    level: HealthLevel
    score: float
    created_at: str = ""
    summary: str = ""
    health: Optional[HealthScore] = None

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        if not isinstance(self.level, HealthLevel):
            object.__setattr__(self, "level", HealthLevel(self.level))
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "level": self.level.value,
            "score": self.score,
            "created_at": self.created_at,
            "summary": self.summary,
            "health": self.health.to_dict() if self.health is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelHealthStatus":
        health = data.get("health")
        return cls(
            model_id=data["model_id"],
            level=HealthLevel(data["level"]),
            score=float(data["score"]),
            created_at=data.get("created_at", ""),
            summary=data.get("summary", ""),
            health=HealthScore.from_dict(health) if health else None,
        )


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MonitoringAlert:
    """A deterministic monitoring alert."""

    alert_id: str
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str = ""
    metric: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None
    entity: str = ""
    fingerprint: str = ""
    count: int = 1
    created_at: str = ""
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.alert_id, "alert_id")
        _require_non_empty(self.title, "title")
        if not isinstance(self.level, AlertLevel):
            object.__setattr__(self, "level", AlertLevel(self.level))
        if not isinstance(self.alert_type, AlertType):
            object.__setattr__(self, "alert_type", AlertType(self.alert_type))
        object.__setattr__(self, "tags", _str_tuple(self.tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "alert_type": self.alert_type.value,
            "title": self.title,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "entity": self.entity,
            "fingerprint": self.fingerprint,
            "count": self.count,
            "created_at": self.created_at,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitoringAlert":
        return cls(
            alert_id=data["alert_id"],
            level=AlertLevel(data["level"]),
            alert_type=AlertType(data["alert_type"]),
            title=data["title"],
            message=data.get("message", ""),
            metric=data.get("metric", ""),
            value=_opt_float(data.get("value")),
            threshold=_opt_float(data.get("threshold")),
            entity=data.get("entity", ""),
            fingerprint=data.get("fingerprint", ""),
            count=int(data.get("count", 1)),
            created_at=data.get("created_at", ""),
            tags=_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    """A declarative alerting rule."""

    name: str
    metric: str
    level: AlertLevel
    comparison: Comparison
    threshold: float
    alert_type: AlertType = AlertType.THRESHOLD
    enabled: bool = True
    dedup_window: int = 1

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        _require_non_empty(self.metric, "metric")
        if not isinstance(self.level, AlertLevel):
            object.__setattr__(self, "level", AlertLevel(self.level))
        if not isinstance(self.comparison, Comparison):
            object.__setattr__(self, "comparison", Comparison(self.comparison))
        if not isinstance(self.alert_type, AlertType):
            object.__setattr__(self, "alert_type", AlertType(self.alert_type))
        object.__setattr__(self, "threshold", float(self.threshold))

    def evaluate(self, value: float) -> bool:
        """Return ``True`` when *value* breaches the policy threshold."""
        if self.comparison is Comparison.GT:
            return value > self.threshold
        if self.comparison is Comparison.GE:
            return value >= self.threshold
        if self.comparison is Comparison.LT:
            return value < self.threshold
        return value <= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "level": self.level.value,
            "comparison": self.comparison.value,
            "threshold": self.threshold,
            "alert_type": self.alert_type.value,
            "enabled": self.enabled,
            "dedup_window": self.dedup_window,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlertPolicy":
        return cls(
            name=data["name"],
            metric=data["metric"],
            level=AlertLevel(data["level"]),
            comparison=Comparison(data["comparison"]),
            threshold=float(data["threshold"]),
            alert_type=AlertType(data.get("alert_type", "THRESHOLD")),
            enabled=bool(data.get("enabled", True)),
            dedup_window=int(data.get("dedup_window", 1)),
        )


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class QualityIssue:
    """A single detected data-quality issue."""

    issue_type: QualityIssueType
    feature_name: str = ""
    severity: DriftSeverity = DriftSeverity.LOW
    count: int = 0
    rate: float = 0.0
    message: str = ""
    details: FrozenMap = ()

    def __post_init__(self) -> None:
        if not isinstance(self.issue_type, QualityIssueType):
            object.__setattr__(self, "issue_type", QualityIssueType(self.issue_type))
        if not isinstance(self.severity, DriftSeverity):
            object.__setattr__(self, "severity", DriftSeverity(self.severity))
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "details", freeze_mapping(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type.value,
            "feature_name": self.feature_name,
            "severity": self.severity.value,
            "count": self.count,
            "rate": self.rate,
            "message": self.message,
            "details": thaw_mapping(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualityIssue":
        return cls(
            issue_type=QualityIssueType(data["issue_type"]),
            feature_name=data.get("feature_name", ""),
            severity=DriftSeverity(data.get("severity", "LOW")),
            count=int(data.get("count", 0)),
            rate=float(data.get("rate", 0.0)),
            message=data.get("message", ""),
            details=freeze_mapping(data.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Aggregate data-quality metrics for a dataset."""

    total_records: int = 0
    completeness: float = 1.0
    null_rate: float = 0.0
    duplicate_rate: float = 0.0
    outlier_rate: float = 0.0
    validity_rate: float = 1.0
    consistency_score: float = 1.0
    schema_violation_count: int = 0
    freshness_seconds: Optional[float] = None
    issue_count: int = 0
    issues: Tuple[QualityIssue, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "completeness": self.completeness,
            "null_rate": self.null_rate,
            "duplicate_rate": self.duplicate_rate,
            "outlier_rate": self.outlier_rate,
            "validity_rate": self.validity_rate,
            "consistency_score": self.consistency_score,
            "schema_violation_count": self.schema_violation_count,
            "freshness_seconds": self.freshness_seconds,
            "issue_count": self.issue_count,
            "issues": [i.to_dict() for i in self.issues],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualityMetrics":
        return cls(
            total_records=int(data.get("total_records", 0)),
            completeness=float(data.get("completeness", 1.0)),
            null_rate=float(data.get("null_rate", 0.0)),
            duplicate_rate=float(data.get("duplicate_rate", 0.0)),
            outlier_rate=float(data.get("outlier_rate", 0.0)),
            validity_rate=float(data.get("validity_rate", 1.0)),
            consistency_score=float(data.get("consistency_score", 1.0)),
            schema_violation_count=int(data.get("schema_violation_count", 0)),
            freshness_seconds=_opt_float(data.get("freshness_seconds")),
            issue_count=int(data.get("issue_count", 0)),
            issues=tuple(QualityIssue.from_dict(i) for i in data.get("issues", [])),
            created_at=data.get("created_at", ""),
        )


# --------------------------------------------------------------------------- #
# Reports, dashboard, statistics, configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MonitoringReport:
    """A consolidated monitoring report."""

    report_id: str
    created_at: str = ""
    model_id: str = ""
    drift_results: Tuple[DataDriftResult, ...] = ()
    concept_drift: Optional[ConceptDriftResult] = None
    prediction_drift: Optional[PredictionDriftResult] = None
    health: Optional[HealthScore] = None
    quality: Optional[QualityMetrics] = None
    alerts: Tuple[MonitoringAlert, ...] = ()
    overall_drift_score: float = 0.0
    summary: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.report_id, "report_id")
        object.__setattr__(self, "drift_results", tuple(self.drift_results))
        object.__setattr__(self, "alerts", tuple(self.alerts))
        object.__setattr__(self, "overall_drift_score", float(self.overall_drift_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "drift_results": [d.to_dict() for d in self.drift_results],
            "concept_drift": self.concept_drift.to_dict() if self.concept_drift else None,
            "prediction_drift": self.prediction_drift.to_dict() if self.prediction_drift else None,
            "health": self.health.to_dict() if self.health else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "alerts": [a.to_dict() for a in self.alerts],
            "overall_drift_score": self.overall_drift_score,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitoringReport":
        return cls(
            report_id=data["report_id"],
            created_at=data.get("created_at", ""),
            model_id=data.get("model_id", ""),
            drift_results=tuple(DataDriftResult.from_dict(d) for d in data.get("drift_results", [])),
            concept_drift=ConceptDriftResult.from_dict(data["concept_drift"]) if data.get("concept_drift") else None,
            prediction_drift=PredictionDriftResult.from_dict(data["prediction_drift"]) if data.get("prediction_drift") else None,
            health=HealthScore.from_dict(data["health"]) if data.get("health") else None,
            quality=QualityMetrics.from_dict(data["quality"]) if data.get("quality") else None,
            alerts=tuple(MonitoringAlert.from_dict(a) for a in data.get("alerts", [])),
            overall_drift_score=float(data.get("overall_drift_score", 0.0)),
            summary=data.get("summary", ""),
        )


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """A point-in-time dashboard snapshot."""

    snapshot_id: str
    model_id: str = ""
    created_at: str = ""
    health: Optional[HealthScore] = None
    drift_summary: FrozenMap = ()
    top_drifted_features: Tuple[Tuple[str, float], ...] = ()
    prediction_trends: FrozenMap = ()
    latency_trends: FrozenMap = ()
    data_quality: Optional[QualityMetrics] = None
    active_alerts: Tuple[MonitoringAlert, ...] = ()
    model_status: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_id, "snapshot_id")
        object.__setattr__(
            self, "top_drifted_features",
            tuple((str(n), float(s)) for n, s in self.top_drifted_features),
        )
        object.__setattr__(self, "drift_summary", freeze_mapping(self.drift_summary))
        object.__setattr__(self, "prediction_trends", freeze_mapping(self.prediction_trends))
        object.__setattr__(self, "latency_trends", freeze_mapping(self.latency_trends))
        object.__setattr__(self, "active_alerts", tuple(self.active_alerts))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "health": self.health.to_dict() if self.health else None,
            "drift_summary": thaw_mapping(self.drift_summary),
            "top_drifted_features": [list(t) for t in self.top_drifted_features],
            "prediction_trends": thaw_mapping(self.prediction_trends),
            "latency_trends": thaw_mapping(self.latency_trends),
            "data_quality": self.data_quality.to_dict() if self.data_quality else None,
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "model_status": self.model_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DashboardSnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            model_id=data.get("model_id", ""),
            created_at=data.get("created_at", ""),
            health=HealthScore.from_dict(data["health"]) if data.get("health") else None,
            drift_summary=freeze_mapping(data.get("drift_summary", {})),
            top_drifted_features=tuple((t[0], float(t[1])) for t in data.get("top_drifted_features", [])),
            prediction_trends=freeze_mapping(data.get("prediction_trends", {})),
            latency_trends=freeze_mapping(data.get("latency_trends", {})),
            data_quality=QualityMetrics.from_dict(data["data_quality"]) if data.get("data_quality") else None,
            active_alerts=tuple(MonitoringAlert.from_dict(a) for a in data.get("active_alerts", [])),
            model_status=data.get("model_status", ""),
        )


@dataclass(frozen=True, slots=True)
class MonitoringStatistics:
    """Aggregate statistics across monitoring activity."""

    total_reports: int = 0
    total_alerts: int = 0
    alerts_by_level: FrozenMap = ()
    drift_checks: int = 0
    drifted_features: int = 0
    average_health_score: float = 0.0
    generated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "alerts_by_level", freeze_mapping(self.alerts_by_level))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reports": self.total_reports,
            "total_alerts": self.total_alerts,
            "alerts_by_level": thaw_mapping(self.alerts_by_level),
            "drift_checks": self.drift_checks,
            "drifted_features": self.drifted_features,
            "average_health_score": self.average_health_score,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitoringStatistics":
        return cls(
            total_reports=int(data.get("total_reports", 0)),
            total_alerts=int(data.get("total_alerts", 0)),
            alerts_by_level=freeze_mapping(data.get("alerts_by_level", {})),
            drift_checks=int(data.get("drift_checks", 0)),
            drifted_features=int(data.get("drifted_features", 0)),
            average_health_score=float(data.get("average_health_score", 0.0)),
            generated_at=data.get("generated_at", ""),
        )


@dataclass(frozen=True, slots=True)
class MonitoringConfiguration:
    """Tunable thresholds and parameters for the monitoring subsystem."""

    psi_threshold: float = 0.2
    js_threshold: float = 0.1
    kl_threshold: float = 0.1
    ks_threshold: float = 0.1
    drift_threshold: float = 0.2
    latency_p95_threshold_ms: float = 250.0
    error_rate_threshold: float = 0.05
    null_rate_threshold: float = 0.1
    duplicate_rate_threshold: float = 0.05
    outlier_z_threshold: float = 3.0
    health_excellent: float = 0.9
    health_healthy: float = 0.75
    health_warning: float = 0.5
    num_bins: int = 10
    window_size: int = 50
    seed: int = 7

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise ValidationError("num_bins must be positive")
        if self.window_size <= 0:
            raise ValidationError("window_size must be positive")
        if not (0.0 <= self.health_warning <= self.health_healthy <= self.health_excellent <= 1.0):
            raise ValidationError("health thresholds must be ordered within [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_threshold": self.psi_threshold,
            "js_threshold": self.js_threshold,
            "kl_threshold": self.kl_threshold,
            "ks_threshold": self.ks_threshold,
            "drift_threshold": self.drift_threshold,
            "latency_p95_threshold_ms": self.latency_p95_threshold_ms,
            "error_rate_threshold": self.error_rate_threshold,
            "null_rate_threshold": self.null_rate_threshold,
            "duplicate_rate_threshold": self.duplicate_rate_threshold,
            "outlier_z_threshold": self.outlier_z_threshold,
            "health_excellent": self.health_excellent,
            "health_healthy": self.health_healthy,
            "health_warning": self.health_warning,
            "num_bins": self.num_bins,
            "window_size": self.window_size,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonitoringConfiguration":
        return cls(
            psi_threshold=float(data.get("psi_threshold", 0.2)),
            js_threshold=float(data.get("js_threshold", 0.1)),
            kl_threshold=float(data.get("kl_threshold", 0.1)),
            ks_threshold=float(data.get("ks_threshold", 0.1)),
            drift_threshold=float(data.get("drift_threshold", 0.2)),
            latency_p95_threshold_ms=float(data.get("latency_p95_threshold_ms", 250.0)),
            error_rate_threshold=float(data.get("error_rate_threshold", 0.05)),
            null_rate_threshold=float(data.get("null_rate_threshold", 0.1)),
            duplicate_rate_threshold=float(data.get("duplicate_rate_threshold", 0.05)),
            outlier_z_threshold=float(data.get("outlier_z_threshold", 3.0)),
            health_excellent=float(data.get("health_excellent", 0.9)),
            health_healthy=float(data.get("health_healthy", 0.75)),
            health_warning=float(data.get("health_warning", 0.5)),
            num_bins=int(data.get("num_bins", 10)),
            window_size=int(data.get("window_size", 50)),
            seed=int(data.get("seed", 7)),
        )