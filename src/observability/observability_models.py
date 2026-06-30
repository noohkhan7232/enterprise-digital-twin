"""Immutable value objects and shared infrastructure for the observability platform.

This module defines the frozen, slotted, JSON-serialisable dataclasses that the
Week 11 Phase 5 observability and production-operations subsystem is built on,
together with deterministic infrastructure (``Clock``, ``IdGenerator``), enums
and validation helpers. It imports nothing from earlier weeks; integration is by
composition.
"""

from __future__ import annotations

import itertools
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ValidationError", "Clock", "ManualClock", "IdGenerator",
    "Severity", "MetricType", "SpanStatus", "IncidentSeverity", "IncidentStatus",
    "SLIType", "ResourceKind", "ReadinessLevel",
    "freeze_mapping", "thaw_mapping", "clamp", "percentile",
    "MetricPoint", "MetricSeries", "TraceSpan", "TraceContext", "LogRecord",
    "TimelineEvent", "Incident", "IncidentTimeline", "ReliabilityScore",
    "SLI", "SLO", "ErrorBudget", "CapacityForecast", "ReadinessCheck",
    "OperationsSnapshot", "ProductionReport", "RunbookEntry",
]


class ValidationError(ValueError):
    """Raised when a value object receives invalid inputs."""


# --------------------------------------------------------------------------- #
# Deterministic infrastructure
# --------------------------------------------------------------------------- #
class Clock:
    """A monotonic, injectable clock. Default starts at a fixed epoch."""

    def __init__(self, start: float = 0.0, step: float = 1.0) -> None:
        self._value = float(start)
        self._step = float(step)
        self._lock = threading.RLock()

    def now(self) -> float:
        with self._lock:
            current = self._value
            self._value += self._step
            return current

    def peek(self) -> float:
        with self._lock:
            return self._value


class ManualClock(Clock):
    """A clock whose time only advances when explicitly told to."""

    def __init__(self, start: float = 0.0) -> None:
        super().__init__(start=start, step=0.0)

    def now(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._value += float(seconds)
            return self._value


class IdGenerator:
    """A deterministic, thread-safe identifier generator."""

    def __init__(self, prefix: str = "id", seed: int = 0) -> None:
        self._prefix = prefix
        self._counter = itertools.count(seed + 1)
        self._lock = threading.RLock()

    def next_id(self) -> str:
        with self._lock:
            return f"{self._prefix}-{next(self._counter):08d}"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(str, Enum):
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"


class SpanStatus(str, Enum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


class IncidentSeverity(str, Enum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    IDENTIFIED = "IDENTIFIED"
    MONITORING = "MONITORING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class SLIType(str, Enum):
    AVAILABILITY = "AVAILABILITY"
    LATENCY = "LATENCY"
    ERROR_RATE = "ERROR_RATE"
    FRESHNESS = "FRESHNESS"


class ResourceKind(str, Enum):
    CPU = "CPU"
    MEMORY = "MEMORY"
    STORAGE = "STORAGE"
    REQUEST_VOLUME = "REQUEST_VOLUME"
    MODEL_GROWTH = "MODEL_GROWTH"
    DATA_GROWTH = "DATA_GROWTH"


class ReadinessLevel(str, Enum):
    NOT_READY = "NOT_READY"
    CONDITIONAL = "CONDITIONAL"
    READY = "READY"
    EXEMPLARY = "EXEMPLARY"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def freeze_mapping(data: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    if not data:
        return ()
    return tuple(sorted((str(k), v) for k, v in dict(data).items()))


def thaw_mapping(items: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in items}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def percentile(values: Sequence[float], pct: float) -> float:
    """Deterministic linear-interpolation percentile (pct in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * frac, 6)


# --------------------------------------------------------------------------- #
# Metric value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MetricPoint:
    timestamp: float
    value: float
    labels: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "labels", freeze_mapping(dict(self.labels)))

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "value": self.value, "labels": thaw_mapping(self.labels)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricPoint":
        return cls(timestamp=data["timestamp"], value=data["value"],
                   labels=freeze_mapping(data.get("labels", {})))


@dataclass(frozen=True, slots=True)
class MetricSeries:
    name: str
    metric_type: MetricType
    points: Tuple[MetricPoint, ...] = ()
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("MetricSeries.name must be non-empty")
        if not isinstance(self.metric_type, MetricType):
            object.__setattr__(self, "metric_type", MetricType(self.metric_type))
        object.__setattr__(self, "points", tuple(self.points))

    @property
    def values(self) -> Tuple[float, ...]:
        return tuple(p.value for p in self.points)

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def latest(self) -> Optional[float]:
        return self.points[-1].value if self.points else None

    @property
    def mean(self) -> float:
        return round(sum(self.values) / len(self.points), 6) if self.points else 0.0

    @property
    def total(self) -> float:
        return round(sum(self.values), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "unit": self.unit,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSeries":
        return cls(
            name=data["name"],
            metric_type=MetricType(data["metric_type"]),
            points=tuple(MetricPoint.from_dict(p) for p in data.get("points", [])),
            unit=data.get("unit", ""),
        )


# --------------------------------------------------------------------------- #
# Tracing value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TraceSpan:
    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: float
    status: SpanStatus = SpanStatus.OK
    parent_id: Optional[str] = None
    attributes: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.span_id or not self.trace_id:
            raise ValidationError("TraceSpan requires span_id and trace_id")
        object.__setattr__(self, "start_time", float(self.start_time))
        object.__setattr__(self, "end_time", float(self.end_time))
        if self.end_time < self.start_time:
            raise ValidationError("TraceSpan.end_time must be >= start_time")
        if not isinstance(self.status, SpanStatus):
            object.__setattr__(self, "status", SpanStatus(self.status))
        object.__setattr__(self, "attributes", freeze_mapping(dict(self.attributes)))

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 6)

    @property
    def is_error(self) -> bool:
        return self.status is SpanStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "attributes": thaw_mapping(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceSpan":
        return cls(
            span_id=data["span_id"], trace_id=data["trace_id"], name=data["name"],
            start_time=data["start_time"], end_time=data["end_time"],
            status=SpanStatus(data.get("status", "OK")), parent_id=data.get("parent_id"),
            attributes=freeze_mapping(data.get("attributes", {})),
        )


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    baggage: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.trace_id or not self.span_id:
            raise ValidationError("TraceContext requires trace_id and span_id")
        object.__setattr__(self, "baggage", freeze_mapping(dict(self.baggage)))

    def child(self, span_id: str) -> "TraceContext":
        return TraceContext(self.trace_id, span_id, self.span_id, self.baggage)

    def to_dict(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "span_id": self.span_id,
                "parent_id": self.parent_id, "baggage": thaw_mapping(self.baggage)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceContext":
        return cls(trace_id=data["trace_id"], span_id=data["span_id"],
                   parent_id=data.get("parent_id"), baggage=freeze_mapping(data.get("baggage", {})))


# --------------------------------------------------------------------------- #
# Logging value object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LogRecord:
    timestamp: float
    severity: Severity
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    workflow_id: Optional[str] = None
    context: Tuple[Tuple[str, Any], ...] = ()
    exception: Optional[Tuple[Tuple[str, Any], ...]] = None
    audit_ref: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.severity, Severity):
            object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "context", freeze_mapping(dict(self.context)))
        if self.exception is not None:
            object.__setattr__(self, "exception", freeze_mapping(dict(self.exception)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "context": thaw_mapping(self.context),
            "exception": thaw_mapping(self.exception) if self.exception is not None else None,
            "audit_ref": self.audit_ref,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogRecord":
        exc = data.get("exception")
        return cls(
            timestamp=data["timestamp"], severity=Severity(data["severity"]),
            message=data["message"], correlation_id=data.get("correlation_id"),
            request_id=data.get("request_id"), workflow_id=data.get("workflow_id"),
            context=freeze_mapping(data.get("context", {})),
            exception=freeze_mapping(exc) if exc else None,
            audit_ref=data.get("audit_ref"),
        )


# --------------------------------------------------------------------------- #
# Incident value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TimelineEvent:
    timestamp: float
    status: IncidentStatus
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.status, IncidentStatus):
            object.__setattr__(self, "status", IncidentStatus(self.status))

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "status": self.status.value, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimelineEvent":
        return cls(timestamp=data["timestamp"], status=IncidentStatus(data["status"]),
                   message=data.get("message", ""))


@dataclass(frozen=True, slots=True)
class IncidentTimeline:
    incident_id: str
    events: Tuple[TimelineEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(sorted(self.events, key=lambda e: e.timestamp)))

    @property
    def ordered(self) -> Tuple[TimelineEvent, ...]:
        return self.events

    def to_dict(self) -> Dict[str, Any]:
        return {"incident_id": self.incident_id, "events": [e.to_dict() for e in self.ordered]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IncidentTimeline":
        return cls(incident_id=data["incident_id"],
                   events=tuple(TimelineEvent.from_dict(e) for e in data.get("events", [])))


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: float
    resolved_at: Optional[float] = None
    root_cause: str = ""
    services: Tuple[str, ...] = ()
    corrective_actions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.incident_id:
            raise ValidationError("Incident.incident_id must be non-empty")
        if not isinstance(self.severity, IncidentSeverity):
            object.__setattr__(self, "severity", IncidentSeverity(self.severity))
        if not isinstance(self.status, IncidentStatus):
            object.__setattr__(self, "status", IncidentStatus(self.status))
        object.__setattr__(self, "created_at", float(self.created_at))
        if self.resolved_at is not None:
            object.__setattr__(self, "resolved_at", float(self.resolved_at))
        object.__setattr__(self, "services", tuple(self.services))
        object.__setattr__(self, "corrective_actions", tuple(self.corrective_actions))

    @property
    def is_resolved(self) -> bool:
        return self.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)

    @property
    def duration(self) -> Optional[float]:
        if self.resolved_at is None:
            return None
        return round(self.resolved_at - self.created_at, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id, "title": self.title,
            "severity": self.severity.value, "status": self.status.value,
            "created_at": self.created_at, "resolved_at": self.resolved_at,
            "duration": self.duration, "root_cause": self.root_cause,
            "services": list(self.services), "corrective_actions": list(self.corrective_actions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Incident":
        return cls(
            incident_id=data["incident_id"], title=data["title"],
            severity=IncidentSeverity(data["severity"]), status=IncidentStatus(data["status"]),
            created_at=data["created_at"], resolved_at=data.get("resolved_at"),
            root_cause=data.get("root_cause", ""), services=tuple(data.get("services", [])),
            corrective_actions=tuple(data.get("corrective_actions", [])),
        )


# --------------------------------------------------------------------------- #
# Reliability / SLO value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ReliabilityScore:
    availability: float
    mtbf: float
    mttr: float
    success_rate: float
    failure_rate: float
    score: float
    operational_risk: float

    def __post_init__(self) -> None:
        for f in ("availability", "success_rate", "failure_rate", "score", "operational_risk"):
            object.__setattr__(self, f, round(float(getattr(self, f)), 6))
        object.__setattr__(self, "mtbf", round(float(self.mtbf), 6))
        object.__setattr__(self, "mttr", round(float(self.mttr), 6))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "availability": self.availability, "mtbf": self.mtbf, "mttr": self.mttr,
            "success_rate": self.success_rate, "failure_rate": self.failure_rate,
            "score": self.score, "operational_risk": self.operational_risk,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReliabilityScore":
        return cls(
            availability=data["availability"], mtbf=data["mtbf"], mttr=data["mttr"],
            success_rate=data["success_rate"], failure_rate=data["failure_rate"],
            score=data["score"], operational_risk=data["operational_risk"],
        )


@dataclass(frozen=True, slots=True)
class SLI:
    name: str
    sli_type: SLIType
    value: float
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("SLI.name must be non-empty")
        if not isinstance(self.sli_type, SLIType):
            object.__setattr__(self, "sli_type", SLIType(self.sli_type))
        object.__setattr__(self, "value", round(float(self.value), 6))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "sli_type": self.sli_type.value,
                "value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SLI":
        return cls(name=data["name"], sli_type=SLIType(data["sli_type"]),
                   value=data["value"], unit=data.get("unit", ""))


@dataclass(frozen=True, slots=True)
class SLO:
    name: str
    sli_type: SLIType
    target: float
    window_seconds: float
    description: str = ""
    comparison: str = "gte"  # gte: higher is better; lte: lower is better

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("SLO.name must be non-empty")
        if not isinstance(self.sli_type, SLIType):
            object.__setattr__(self, "sli_type", SLIType(self.sli_type))
        if self.comparison not in ("gte", "lte"):
            raise ValidationError("SLO.comparison must be 'gte' or 'lte'")
        object.__setattr__(self, "target", round(float(self.target), 6))
        object.__setattr__(self, "window_seconds", float(self.window_seconds))

    def is_met(self, sli_value: float) -> bool:
        if self.comparison == "gte":
            return sli_value >= self.target
        return sli_value <= self.target

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "sli_type": self.sli_type.value, "target": self.target,
                "window_seconds": self.window_seconds, "description": self.description,
                "comparison": self.comparison}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SLO":
        return cls(name=data["name"], sli_type=SLIType(data["sli_type"]), target=data["target"],
                   window_seconds=data["window_seconds"], description=data.get("description", ""),
                   comparison=data.get("comparison", "gte"))


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    slo_name: str
    target: float
    actual: float
    window_seconds: float
    budget_total: float
    budget_consumed: float
    budget_remaining: float
    burn_rate: float

    def __post_init__(self) -> None:
        for f in ("target", "actual", "budget_total", "budget_consumed",
                  "budget_remaining", "burn_rate"):
            object.__setattr__(self, f, round(float(getattr(self, f)), 6))
        object.__setattr__(self, "window_seconds", float(self.window_seconds))

    @property
    def is_exhausted(self) -> bool:
        return self.budget_remaining <= 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo_name": self.slo_name, "target": self.target, "actual": self.actual,
            "window_seconds": self.window_seconds, "budget_total": self.budget_total,
            "budget_consumed": self.budget_consumed, "budget_remaining": self.budget_remaining,
            "burn_rate": self.burn_rate, "is_exhausted": self.is_exhausted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorBudget":
        return cls(
            slo_name=data["slo_name"], target=data["target"], actual=data["actual"],
            window_seconds=data["window_seconds"], budget_total=data["budget_total"],
            budget_consumed=data["budget_consumed"], budget_remaining=data["budget_remaining"],
            burn_rate=data["burn_rate"],
        )


# --------------------------------------------------------------------------- #
# Capacity / operations / readiness value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CapacityForecast:
    resource: ResourceKind
    current: float
    horizon: int
    forecast: Tuple[float, ...]
    growth_rate: float
    recommended_capacity: float
    exhaustion_step: Optional[int] = None
    unit: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.resource, ResourceKind):
            object.__setattr__(self, "resource", ResourceKind(self.resource))
        object.__setattr__(self, "current", round(float(self.current), 6))
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "forecast", tuple(round(float(v), 6) for v in self.forecast))
        object.__setattr__(self, "growth_rate", round(float(self.growth_rate), 6))
        object.__setattr__(self, "recommended_capacity", round(float(self.recommended_capacity), 6))

    @property
    def peak(self) -> float:
        return max(self.forecast) if self.forecast else self.current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource.value, "current": self.current, "horizon": self.horizon,
            "forecast": list(self.forecast), "growth_rate": self.growth_rate,
            "recommended_capacity": self.recommended_capacity,
            "exhaustion_step": self.exhaustion_step, "peak": self.peak, "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapacityForecast":
        return cls(
            resource=ResourceKind(data["resource"]), current=data["current"],
            horizon=data["horizon"], forecast=tuple(data.get("forecast", [])),
            growth_rate=data["growth_rate"], recommended_capacity=data["recommended_capacity"],
            exhaustion_step=data.get("exhaustion_step"), unit=data.get("unit", ""),
        )


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    score: float
    message: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("ReadinessCheck.name must be non-empty")
        object.__setattr__(self, "score", clamp(self.score))
        object.__setattr__(self, "weight", float(self.weight))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "score": self.score,
                "message": self.message, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReadinessCheck":
        return cls(name=data["name"], passed=data["passed"], score=data["score"],
                   message=data.get("message", ""), weight=data.get("weight", 1.0))


@dataclass(frozen=True, slots=True)
class OperationsSnapshot:
    generated_at: float
    metrics_summary: Tuple[Tuple[str, Any], ...] = ()
    reliability: Optional[ReliabilityScore] = None
    active_incidents: Tuple[Incident, ...] = ()
    slo_compliance: Tuple[Tuple[str, Any], ...] = ()
    capacity: Tuple[CapacityForecast, ...] = ()
    readiness_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", float(self.generated_at))
        object.__setattr__(self, "metrics_summary", freeze_mapping(dict(self.metrics_summary)))
        object.__setattr__(self, "slo_compliance", freeze_mapping(dict(self.slo_compliance)))
        object.__setattr__(self, "active_incidents", tuple(self.active_incidents))
        object.__setattr__(self, "capacity", tuple(self.capacity))
        object.__setattr__(self, "readiness_score", round(float(self.readiness_score), 6))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "metrics_summary": thaw_mapping(self.metrics_summary),
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "active_incidents": [i.to_dict() for i in self.active_incidents],
            "slo_compliance": thaw_mapping(self.slo_compliance),
            "capacity": [c.to_dict() for c in self.capacity],
            "readiness_score": self.readiness_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperationsSnapshot":
        rel = data.get("reliability")
        return cls(
            generated_at=data["generated_at"],
            metrics_summary=freeze_mapping(data.get("metrics_summary", {})),
            reliability=ReliabilityScore.from_dict(rel) if rel else None,
            active_incidents=tuple(Incident.from_dict(i) for i in data.get("active_incidents", [])),
            slo_compliance=freeze_mapping(data.get("slo_compliance", {})),
            capacity=tuple(CapacityForecast.from_dict(c) for c in data.get("capacity", [])),
            readiness_score=data.get("readiness_score", 0.0),
        )


@dataclass(frozen=True, slots=True)
class ProductionReport:
    generated_at: float
    checks: Tuple[ReadinessCheck, ...] = ()
    score: float = 0.0
    level: ReadinessLevel = ReadinessLevel.NOT_READY

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", float(self.generated_at))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "score", round(float(self.score), 6))
        if not isinstance(self.level, ReadinessLevel):
            object.__setattr__(self, "level", ReadinessLevel(self.level))

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def check(self, name: str) -> Optional[ReadinessCheck]:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at, "score": self.score, "level": self.level.value,
            "summary": {"passed": self.passed, "failed": self.failed},
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProductionReport":
        return cls(
            generated_at=data["generated_at"],
            checks=tuple(ReadinessCheck.from_dict(c) for c in data.get("checks", [])),
            score=data.get("score", 0.0), level=ReadinessLevel(data.get("level", "NOT_READY")),
        )


@dataclass(frozen=True, slots=True)
class RunbookEntry:
    entry_id: str
    title: str
    severity: IncidentSeverity
    symptoms: Tuple[str, ...] = ()
    steps: Tuple[str, ...] = ()
    related_services: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValidationError("RunbookEntry.entry_id must be non-empty")
        if not isinstance(self.severity, IncidentSeverity):
            object.__setattr__(self, "severity", IncidentSeverity(self.severity))
        object.__setattr__(self, "symptoms", tuple(self.symptoms))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "related_services", tuple(self.related_services))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id, "title": self.title, "severity": self.severity.value,
            "symptoms": list(self.symptoms), "steps": list(self.steps),
            "related_services": list(self.related_services),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunbookEntry":
        return cls(
            entry_id=data["entry_id"], title=data["title"],
            severity=IncidentSeverity(data["severity"]), symptoms=tuple(data.get("symptoms", [])),
            steps=tuple(data.get("steps", [])), related_services=tuple(data.get("related_services", [])),
        )