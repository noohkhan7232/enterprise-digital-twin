"""Enterprise Event Bus (Week 10, Phase 3).

A deterministic, thread-safe, in-process event backbone that connects every
component of the Enterprise Digital Twin & Decision Intelligence Platform. It
is **not** Kafka, RabbitMQ, Redis, or any external broker; it is a pure-Python
publish/subscribe core implementing the Observer and Registry patterns with
immutable event history, deterministic delivery ordering, topic/wildcard/
priority/direct/filtered routing, a replay engine, a dead-letter queue, and
rich analytics.

Design tenets: pure Python + NumPy, no asyncio and no external brokers,
dependency injection, thread-safe mutation behind a re-entrant lock,
synchronous and deterministic delivery, frozen + slotted dataclasses with full
JSON serialisation, custom exceptions, and rich logging.

Integration is by composition only. The bus imports nothing from the Workflow
Engine, Business Process Orchestrator, Executive Copilot, or Knowledge Agent;
those modules (and future ones) consume the bus, never the reverse.

Command-line demonstration::

    python src/events/enterprise_event_bus.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

__all__ = [
    # Exceptions
    "EventBusError",
    "EventValidationError",
    "SubscriptionError",
    "PublishError",
    "ReplayError",
    "DeadLetterError",
    "RoutingError",
    # Enums
    "EventType",
    "EventPriority",
    "DeliveryMode",
    "RecoveryStatus",
    # Clocks
    "Clock",
    "LogicalClock",
    "SystemClock",
    # Domain model
    "EventMetadata",
    "EnterpriseEvent",
    "EventEnvelope",
    "EventResult",
    "EventFilter",
    "EventSubscription",
    "EventStatistics",
    "EventHistory",
    "DeadLetterEvent",
    "EventReplay",
    "EventBatch",
    # Bus
    "EnterpriseEventBus",
    "create_default_event_bus",
]

logger = logging.getLogger("events.bus")
if not logger.handlers:  # pragma: no cover - environmental
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class EventBusError(Exception):
    """Base class for all event bus errors."""


class EventValidationError(EventBusError):
    """Raised when an event or domain object fails validation."""


class SubscriptionError(EventBusError):
    """Raised for invalid subscription operations."""


class PublishError(EventBusError):
    """Raised when an event cannot be published."""


class ReplayError(EventBusError):
    """Raised for invalid replay requests."""


class DeadLetterError(EventBusError):
    """Raised for invalid dead-letter operations."""


class RoutingError(EventBusError):
    """Raised for invalid routing configuration."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise EventValidationError(f"Expected a mapping, received {type(payload)!r}")
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _parse_json(payload: str) -> Dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)


def _payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class _ValueEnum(str, Enum):
    @classmethod
    def coerce(cls, value: Any) -> "_ValueEnum":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise EventValidationError(f"{value!r} is not a valid {cls.__name__}") from exc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class EventType(_ValueEnum):
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    BUSINESS_PROCESS_STARTED = "business_process_started"
    BUSINESS_PROCESS_COMPLETED = "business_process_completed"
    BUSINESS_PROCESS_FAILED = "business_process_failed"
    KNOWLEDGE_RETRIEVED = "knowledge_retrieved"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    EXECUTIVE_BRIEF_GENERATED = "executive_brief_generated"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_COMPLETED = "simulation_completed"
    PREDICTION_GENERATED = "prediction_generated"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    ASSET_HEALTH_CHANGED = "asset_health_changed"
    MAINTENANCE_SCHEDULED = "maintenance_scheduled"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    AUDIT_EVENT = "audit_event"
    SYSTEM_EVENT = "system_event"
    CUSTOM_EVENT = "custom_event"

    @property
    def default_topic(self) -> str:
        return self.value.replace("_", ".")


class EventPriority(_ValueEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        return _PRIORITY_ORDER.index(self)


class DeliveryMode(_ValueEnum):
    BROADCAST = "broadcast"
    DIRECT = "direct"
    FILTERED = "filtered"
    TOPIC = "topic"
    WILDCARD = "wildcard"
    PRIORITY = "priority"


class RecoveryStatus(_ValueEnum):
    PENDING = "pending"
    RETRYING = "retrying"
    RECOVERED = "recovered"
    FAILED = "failed"
    ABANDONED = "abandoned"


_PRIORITY_ORDER: List[EventPriority] = [
    EventPriority.LOW, EventPriority.NORMAL, EventPriority.HIGH, EventPriority.CRITICAL,
]


# ---------------------------------------------------------------------------
# Clocks (injectable for determinism)
# ---------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...
    def advance(self, seconds: float) -> None: ...


class LogicalClock:
    """Deterministic monotonic clock; advances only when told to."""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise EventBusError("cannot advance clock by a negative amount")
        self._t += float(seconds)


class SystemClock:
    """Wall-clock time source (non-deterministic)."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def advance(self, seconds: float) -> None:  # pragma: no cover - timing
        if seconds > 0:
            time.sleep(seconds)


# ---------------------------------------------------------------------------
# Domain model - metadata & events
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventMetadata:
    """Immutable provenance attached to every event."""

    source: str = "system"
    correlation_id: str = ""
    trace_id: str = ""
    causation_id: str = ""
    schema_version: str = "1.0"
    tags_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags_json", _canonical_json(_parse_json(self.tags_json)))

    @property
    def tags(self) -> Dict[str, Any]:
        return _parse_json(self.tags_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "causation_id": self.causation_id,
            "schema_version": self.schema_version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventMetadata":
        return cls(
            source=data.get("source", "system"),
            correlation_id=data.get("correlation_id", ""),
            trace_id=data.get("trace_id", ""),
            causation_id=data.get("causation_id", ""),
            schema_version=data.get("schema_version", "1.0"),
            tags_json=_canonical_json(data.get("tags") or {}),
        )


@dataclass(frozen=True, slots=True)
class EnterpriseEvent:
    """An immutable enterprise event. ``payload_hash`` is always consistent."""

    event_type: EventType
    topic: str
    priority: EventPriority = EventPriority.NORMAL
    payload_json: str = "{}"
    metadata: EventMetadata = field(default_factory=EventMetadata)
    event_id: str = ""
    sequence: int = -1
    timestamp: float = 0.0
    payload_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType.coerce(self.event_type))
        object.__setattr__(self, "priority", EventPriority.coerce(self.priority))
        if not self.topic:
            raise EventValidationError("event topic must be non-empty")
        object.__setattr__(self, "payload_json", _canonical_json(_parse_json(self.payload_json)))
        object.__setattr__(self, "payload_hash", _payload_hash(self.payload_json))
        if not isinstance(self.metadata, EventMetadata):
            raise EventValidationError("metadata must be an EventMetadata")

    @classmethod
    def create(
        cls,
        event_type: EventType,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        topic: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "system",
        correlation_id: str = "",
        trace_id: str = "",
        causation_id: str = "",
        tags: Optional[Mapping[str, Any]] = None,
        event_id: str = "",
    ) -> "EnterpriseEvent":
        etype = EventType.coerce(event_type)
        metadata = EventMetadata(
            source=source, correlation_id=correlation_id, trace_id=trace_id,
            causation_id=causation_id, tags_json=_canonical_json(tags or {}))
        return cls(
            event_type=etype,
            topic=topic or etype.default_topic,
            priority=EventPriority.coerce(priority),
            payload_json=_canonical_json(payload or {}),
            metadata=metadata,
            event_id=event_id,
        )

    @property
    def payload(self) -> Dict[str, Any]:
        return _parse_json(self.payload_json)

    @property
    def correlation_id(self) -> str:
        return self.metadata.correlation_id

    @property
    def is_stamped(self) -> bool:
        return self.sequence >= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "topic": self.topic,
            "priority": self.priority.value,
            "payload": self.payload,
            "metadata": self.metadata.to_dict(),
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnterpriseEvent":
        return cls(
            event_type=EventType.coerce(data["event_type"]),
            topic=data["topic"],
            priority=EventPriority.coerce(data.get("priority", "normal")),
            payload_json=_canonical_json(data.get("payload") or {}),
            metadata=EventMetadata.from_dict(data.get("metadata") or {}),
            event_id=data.get("event_id", ""),
            sequence=int(data.get("sequence", -1)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """A routed event ready for delivery."""

    event: EnterpriseEvent
    delivery_mode: DeliveryMode = DeliveryMode.TOPIC
    target: Optional[str] = None
    attempt: int = 1
    enqueued_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delivery_mode", DeliveryMode.coerce(self.delivery_mode))
        if not isinstance(self.event, EnterpriseEvent):
            raise EventValidationError("envelope must wrap an EnterpriseEvent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "delivery_mode": self.delivery_mode.value,
            "target": self.target,
            "attempt": self.attempt,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventEnvelope":
        return cls(
            event=EnterpriseEvent.from_dict(data["event"]),
            delivery_mode=DeliveryMode.coerce(data.get("delivery_mode", "topic")),
            target=data.get("target"),
            attempt=int(data.get("attempt", 1)),
            enqueued_at=float(data.get("enqueued_at", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EventResult:
    """The outcome of delivering one event to one subscriber."""

    event_id: str
    subscriber_id: str
    delivered: bool
    error: str = ""
    latency: float = 0.0
    attempts: int = 1
    dead_lettered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "subscriber_id": self.subscriber_id,
            "delivered": self.delivered,
            "error": self.error,
            "latency": self.latency,
            "attempts": self.attempts,
            "dead_lettered": self.dead_lettered,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventResult":
        return cls(
            event_id=data["event_id"],
            subscriber_id=data["subscriber_id"],
            delivered=bool(data["delivered"]),
            error=data.get("error", ""),
            latency=float(data.get("latency", 0.0)),
            attempts=int(data.get("attempts", 1)),
            dead_lettered=bool(data.get("dead_lettered", False)),
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventFilter:
    """A deterministic, declarative event filter."""

    topic_pattern: Optional[str] = None
    min_priority: Optional[EventPriority] = None
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    workflow_id: Optional[str] = None
    process_id: Optional[str] = None
    event_type: Optional[EventType] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    metadata_key: Optional[str] = None
    metadata_value: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.min_priority is not None:
            object.__setattr__(self, "min_priority", EventPriority.coerce(self.min_priority))
        if self.event_type is not None:
            object.__setattr__(self, "event_type", EventType.coerce(self.event_type))

    def matches(self, event: EnterpriseEvent) -> bool:
        if self.topic_pattern is not None and not _topic_matches(self.topic_pattern, event.topic):
            return False
        if self.min_priority is not None and event.priority.level < self.min_priority.level:
            return False
        if self.source is not None and event.metadata.source != self.source:
            return False
        if self.correlation_id is not None and event.metadata.correlation_id != self.correlation_id:
            return False
        if self.event_type is not None and event.event_type is not self.event_type:
            return False
        if self.time_start is not None and event.timestamp < self.time_start:
            return False
        if self.time_end is not None and event.timestamp > self.time_end:
            return False
        if self.workflow_id is not None and self._field(event, "workflow_id") != self.workflow_id:
            return False
        if self.process_id is not None and self._field(event, "process_id") != self.process_id:
            return False
        if self.metadata_key is not None:
            tags = event.metadata.tags
            if self.metadata_key not in tags:
                return False
            if self.metadata_value is not None and tags[self.metadata_key] != self.metadata_value:
                return False
        return True

    @staticmethod
    def _field(event: EnterpriseEvent, key: str) -> Any:
        payload = event.payload
        if key in payload:
            return payload[key]
        return event.metadata.tags.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_pattern": self.topic_pattern,
            "min_priority": self.min_priority.value if self.min_priority else None,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "workflow_id": self.workflow_id,
            "process_id": self.process_id,
            "event_type": self.event_type.value if self.event_type else None,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "metadata_key": self.metadata_key,
            "metadata_value": self.metadata_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventFilter":
        return cls(
            topic_pattern=data.get("topic_pattern"),
            min_priority=EventPriority.coerce(data["min_priority"]) if data.get("min_priority") else None,
            source=data.get("source"),
            correlation_id=data.get("correlation_id"),
            workflow_id=data.get("workflow_id"),
            process_id=data.get("process_id"),
            event_type=EventType.coerce(data["event_type"]) if data.get("event_type") else None,
            time_start=data.get("time_start"),
            time_end=data.get("time_end"),
            metadata_key=data.get("metadata_key"),
            metadata_value=data.get("metadata_value"),
        )


def _topic_matches(pattern: str, topic: str) -> bool:
    """Match a topic against a pattern.

    ``*`` matches exactly one segment, ``#`` matches zero or more trailing
    segments, and an exact string matches itself. ``#`` alone matches all.
    """

    if pattern == "#":
        return True
    p = pattern.split(".")
    t = topic.split(".")
    i = j = 0
    while i < len(p):
        seg = p[i]
        if seg == "#":
            return True
        if j >= len(t):
            return False
        if seg == "*" or seg == t[j]:
            i += 1
            j += 1
        else:
            return False
    return j == len(t)


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventSubscription:
    """Immutable subscription metadata (the handler is held by the bus)."""

    subscription_id: str
    subscriber_id: str
    topic_pattern: str = "#"
    delivery_mode: DeliveryMode = DeliveryMode.TOPIC
    priority: int = 0
    min_priority: EventPriority = EventPriority.LOW
    persistent: bool = True
    event_filter: Optional[EventFilter] = None
    created_seq: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delivery_mode", DeliveryMode.coerce(self.delivery_mode))
        object.__setattr__(self, "min_priority", EventPriority.coerce(self.min_priority))
        if not self.subscription_id:
            raise SubscriptionError("subscription_id must be non-empty")
        if not self.subscriber_id:
            raise SubscriptionError("subscriber_id must be non-empty")
        if self.event_filter is not None and not isinstance(self.event_filter, EventFilter):
            raise SubscriptionError("event_filter must be an EventFilter")

    def matches(self, event: EnterpriseEvent, target: Optional[str]) -> bool:
        mode = self.delivery_mode
        if mode is DeliveryMode.BROADCAST:
            base = True
        elif mode is DeliveryMode.DIRECT:
            base = target is not None and target == self.subscriber_id
        else:
            base = _topic_matches(self.topic_pattern, event.topic)
        if not base:
            return False
        if event.priority.level < self.min_priority.level:
            return False
        if self.event_filter is not None and not self.event_filter.matches(event):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "subscriber_id": self.subscriber_id,
            "topic_pattern": self.topic_pattern,
            "delivery_mode": self.delivery_mode.value,
            "priority": self.priority,
            "min_priority": self.min_priority.value,
            "persistent": self.persistent,
            "event_filter": self.event_filter.to_dict() if self.event_filter else None,
            "created_seq": self.created_seq,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventSubscription":
        return cls(
            subscription_id=data["subscription_id"],
            subscriber_id=data["subscriber_id"],
            topic_pattern=data.get("topic_pattern", "#"),
            delivery_mode=DeliveryMode.coerce(data.get("delivery_mode", "topic")),
            priority=int(data.get("priority", 0)),
            min_priority=EventPriority.coerce(data.get("min_priority", "low")),
            persistent=bool(data.get("persistent", True)),
            event_filter=EventFilter.from_dict(data["event_filter"]) if data.get("event_filter") else None,
            created_seq=int(data.get("created_seq", 0)),
        )


# ---------------------------------------------------------------------------
# History, dead letters, replay, batches, statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventHistory:
    """An append-only, immutable history of stamped events."""

    events: Tuple[EnterpriseEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @property
    def next_sequence(self) -> int:
        return self.events[-1].sequence + 1 if self.events else 0

    def append(self, event: EnterpriseEvent, *, max_history: int = 0) -> "EventHistory":
        events = self.events + (event,)
        if max_history and len(events) > max_history:
            events = events[len(events) - max_history:]
        return EventHistory(events)

    def by_topic(self, pattern: str) -> Tuple[EnterpriseEvent, ...]:
        return tuple(e for e in self.events if _topic_matches(pattern, e.topic))

    def by_correlation(self, correlation_id: str) -> Tuple[EnterpriseEvent, ...]:
        return tuple(e for e in self.events if e.metadata.correlation_id == correlation_id)

    def by_time(self, start: float, end: float) -> Tuple[EnterpriseEvent, ...]:
        return tuple(e for e in self.events if start <= e.timestamp <= end)

    def by_type(self, event_type: EventType) -> Tuple[EnterpriseEvent, ...]:
        event_type = EventType.coerce(event_type)
        return tuple(e for e in self.events if e.event_type is event_type)

    def last(self, n: int) -> Tuple[EnterpriseEvent, ...]:
        if n < 0:
            raise EventValidationError("n must be >= 0")
        return self.events[-n:] if n else ()

    def filter(self, event_filter: EventFilter) -> Tuple[EnterpriseEvent, ...]:
        return tuple(e for e in self.events if event_filter.matches(e))

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventHistory":
        return cls(tuple(EnterpriseEvent.from_dict(e) for e in data.get("events", ())))


@dataclass(frozen=True, slots=True)
class DeadLetterEvent:
    """A failed delivery captured for inspection and recovery."""

    event: EnterpriseEvent
    subscriber_id: str
    retry_count: int
    failure_reason: str
    recovery_status: RecoveryStatus = RecoveryStatus.PENDING
    first_failed_at: float = 0.0
    last_attempt_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery_status", RecoveryStatus.coerce(self.recovery_status))
        if not isinstance(self.event, EnterpriseEvent):
            raise DeadLetterError("dead letter must wrap an EnterpriseEvent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "subscriber_id": self.subscriber_id,
            "retry_count": self.retry_count,
            "failure_reason": self.failure_reason,
            "recovery_status": self.recovery_status.value,
            "first_failed_at": self.first_failed_at,
            "last_attempt_at": self.last_attempt_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeadLetterEvent":
        return cls(
            event=EnterpriseEvent.from_dict(data["event"]),
            subscriber_id=data["subscriber_id"],
            retry_count=int(data["retry_count"]),
            failure_reason=data["failure_reason"],
            recovery_status=RecoveryStatus.coerce(data.get("recovery_status", "pending")),
            first_failed_at=float(data.get("first_failed_at", 0.0)),
            last_attempt_at=float(data.get("last_attempt_at", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EventReplay:
    """The immutable record of a replay operation."""

    replay_id: str
    mode: str
    criteria: str
    event_ids: Tuple[str, ...]
    count: int
    replayed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_ids", tuple(self.event_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "mode": self.mode,
            "criteria": self.criteria,
            "event_ids": list(self.event_ids),
            "count": self.count,
            "replayed_at": self.replayed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventReplay":
        return cls(
            replay_id=data["replay_id"],
            mode=data["mode"],
            criteria=data.get("criteria", ""),
            event_ids=tuple(data.get("event_ids", ())),
            count=int(data["count"]),
            replayed_at=float(data.get("replayed_at", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EventBatch:
    """An immutable batch of events to publish atomically (in order)."""

    batch_id: str
    events: Tuple[EnterpriseEvent, ...]
    created_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if not self.batch_id:
            raise EventValidationError("batch_id must be non-empty")
        for e in self.events:
            if not isinstance(e, EnterpriseEvent):
                raise EventValidationError("batch entries must be EnterpriseEvent")

    def __len__(self) -> int:
        return len(self.events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "events": [e.to_dict() for e in self.events],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventBatch":
        return cls(
            batch_id=data["batch_id"],
            events=tuple(EnterpriseEvent.from_dict(e) for e in data.get("events", ())),
            created_at=float(data.get("created_at", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class EventStatistics:
    """Aggregate analytics for the event bus."""

    published: int = 0
    delivered: int = 0
    dropped: int = 0
    unrouted: int = 0
    replayed: int = 0
    subscribers: int = 0
    dead_letter_count: int = 0
    average_latency: float = 0.0
    failure_rate: float = 0.0
    delivery_success_rate: float = 0.0
    topic_counts_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "topic_counts_json", _canonical_json(_parse_json(self.topic_counts_json)))

    @property
    def topic_counts(self) -> Dict[str, int]:
        return _parse_json(self.topic_counts_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "unrouted": self.unrouted,
            "replayed": self.replayed,
            "subscribers": self.subscribers,
            "dead_letter_count": self.dead_letter_count,
            "average_latency": self.average_latency,
            "failure_rate": self.failure_rate,
            "delivery_success_rate": self.delivery_success_rate,
            "topic_counts": self.topic_counts,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventStatistics":
        return cls(
            published=int(data.get("published", 0)),
            delivered=int(data.get("delivered", 0)),
            dropped=int(data.get("dropped", 0)),
            unrouted=int(data.get("unrouted", 0)),
            replayed=int(data.get("replayed", 0)),
            subscribers=int(data.get("subscribers", 0)),
            dead_letter_count=int(data.get("dead_letter_count", 0)),
            average_latency=float(data.get("average_latency", 0.0)),
            failure_rate=float(data.get("failure_rate", 0.0)),
            delivery_success_rate=float(data.get("delivery_success_rate", 0.0)),
            topic_counts_json=_canonical_json(data.get("topic_counts") or {}),
        )


EventHandler = Callable[[EnterpriseEvent], None]


# ---------------------------------------------------------------------------
# The event bus
# ---------------------------------------------------------------------------
class EnterpriseEventBus:
    """A deterministic, thread-safe, in-process publish/subscribe bus.

    Delivery is synchronous and ordered by ``(subscriber priority desc,
    registration order asc)``. All mutable state is guarded by a re-entrant
    lock; handlers are invoked outside the lock so that a handler may safely
    publish further events (depth-first, deterministic).
    """

    def __init__(
        self,
        *,
        clock: Optional[Clock] = None,
        max_retries: int = 1,
        dead_letter_enabled: bool = True,
        max_history: int = 0,
        max_dead_letters: int = 0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._max_retries = max(1, int(max_retries))
        self._dead_letter_enabled = dead_letter_enabled
        self._max_history = max(0, int(max_history))
        self._max_dead_letters = max(0, int(max_dead_letters))
        self._log = logger_ or logger
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._handlers: Dict[str, EventHandler] = {}
        self._history = EventHistory()
        self._dlq: Tuple[DeadLetterEvent, ...] = ()
        self._seq = 0
        self._sub_seq = 0
        self._published = 0
        self._delivered = 0
        self._dropped = 0
        self._unrouted = 0
        self._replayed = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._topic_counts: Dict[str, int] = {}

    # -- subscription management --------------------------------------------
    def subscribe(
        self,
        handler: EventHandler,
        *,
        topic_pattern: str = "#",
        subscriber_id: Optional[str] = None,
        delivery_mode: DeliveryMode = DeliveryMode.TOPIC,
        priority: int = 0,
        min_priority: EventPriority = EventPriority.LOW,
        persistent: bool = True,
        event_filter: Optional[EventFilter] = None,
    ) -> EventSubscription:
        if not callable(handler):
            raise SubscriptionError("handler must be callable")
        with self._lock:
            seq = self._sub_seq
            self._sub_seq += 1
            sub_id = f"sub-{seq:06d}"
            subscription = EventSubscription(
                subscription_id=sub_id,
                subscriber_id=subscriber_id or sub_id,
                topic_pattern=topic_pattern,
                delivery_mode=delivery_mode,
                priority=priority,
                min_priority=min_priority,
                persistent=persistent,
                event_filter=event_filter,
                created_seq=seq,
            )
            self._subscriptions[sub_id] = subscription
            self._handlers[sub_id] = handler
            self._log.debug("subscribed %s -> %s", sub_id, topic_pattern)
            return subscription

    def once(self, handler: EventHandler, **kwargs: Any) -> EventSubscription:
        """Register a one-time subscription removed after its first delivery."""

        kwargs["persistent"] = False
        return self.subscribe(handler, **kwargs)

    def unsubscribe(self, subscription_id: str) -> None:
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise SubscriptionError(f"unknown subscription: {subscription_id}")
            del self._subscriptions[subscription_id]
            self._handlers.pop(subscription_id, None)

    def clear(self) -> None:
        """Remove every subscription (history and DLQ are retained)."""

        with self._lock:
            self._subscriptions.clear()
            self._handlers.clear()

    def subscriptions(self) -> Tuple[EventSubscription, ...]:
        with self._lock:
            return tuple(self._subscriptions[k] for k in sorted(self._subscriptions))

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    # -- publishing ----------------------------------------------------------
    def publish(
        self, event: EnterpriseEvent, *, target: Optional[str] = None,
        delivery_mode: Optional[DeliveryMode] = None,
    ) -> Tuple[EventResult, ...]:
        if not isinstance(event, EnterpriseEvent):
            raise PublishError("publish requires an EnterpriseEvent")

        # Phase 1 (locked): stamp, record history, snapshot matching subscribers.
        with self._lock:
            stamped = self._stamp(event)
            self._history = self._history.append(stamped, max_history=self._max_history)
            self._published += 1
            self._topic_counts[stamped.topic] = self._topic_counts.get(stamped.topic, 0) + 1
            matches = [
                (sub, self._handlers[sub.subscription_id])
                for sub in self._subscriptions.values()
                if sub.matches(stamped, target)
            ]
            matches.sort(key=lambda pair: (-pair[0].priority, pair[0].created_seq))
            if not matches:
                self._unrouted += 1

        # Phase 2 (unlocked): deliver. Handlers may re-enter publish() safely.
        results: List[EventResult] = []
        dead_letters: List[DeadLetterEvent] = []
        delivered = dropped = 0
        latency_sum = 0.0
        latency_count = 0
        fired_once: List[str] = []
        for sub, handler in matches:
            result, dead = self._deliver(stamped, sub, handler)
            results.append(result)
            if result.delivered:
                delivered += 1
                latency_sum += result.latency
                latency_count += 1
            else:
                dropped += 1
                if dead is not None:
                    dead_letters.append(dead)
            if not sub.persistent:
                fired_once.append(sub.subscription_id)

        # Phase 3 (locked): commit counters / DLQ / one-time removals.
        with self._lock:
            self._delivered += delivered
            self._dropped += dropped
            self._latency_sum += latency_sum
            self._latency_count += latency_count
            if dead_letters:
                self._dlq = self._dlq + tuple(dead_letters)
                if self._max_dead_letters and len(self._dlq) > self._max_dead_letters:
                    self._dlq = self._dlq[len(self._dlq) - self._max_dead_letters:]
            for sub_id in fired_once:
                self._subscriptions.pop(sub_id, None)
                self._handlers.pop(sub_id, None)
        return tuple(results)

    def publish_batch(self, batch: EventBatch) -> Tuple[EventResult, ...]:
        if not isinstance(batch, EventBatch):
            raise PublishError("publish_batch requires an EventBatch")
        results: List[EventResult] = []
        for event in batch.events:
            results.extend(self.publish(event))
        return tuple(results)

    def _deliver(
        self, event: EnterpriseEvent, sub: EventSubscription, handler: EventHandler,
    ) -> Tuple[EventResult, Optional[DeadLetterEvent]]:
        attempts = 0
        last_error = ""
        start = self._clock.now()
        for attempt in range(1, self._max_retries + 1):
            attempts = attempt
            try:
                handler(event)
                latency = max(0.0, self._clock.now() - start)
                return (
                    EventResult(event.event_id, sub.subscriber_id, True, "", latency, attempts),
                    None,
                )
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                last_error = f"{type(exc).__name__}: {exc}"
                self._log.warning("delivery failed %s -> %s: %s",
                                  event.event_id, sub.subscriber_id, last_error)
        dead: Optional[DeadLetterEvent] = None
        if self._dead_letter_enabled:
            now = self._clock.now()
            dead = DeadLetterEvent(
                event=event, subscriber_id=sub.subscriber_id, retry_count=attempts,
                failure_reason=last_error, recovery_status=RecoveryStatus.PENDING,
                first_failed_at=start, last_attempt_at=now)
        return (
            EventResult(event.event_id, sub.subscriber_id, False, last_error, 0.0, attempts,
                        dead_lettered=dead is not None),
            dead,
        )

    def _stamp(self, event: EnterpriseEvent) -> EnterpriseEvent:
        seq = self._seq
        self._seq += 1
        ts = self._clock.now()
        self._clock.advance(1.0)
        meta = event.metadata
        if not meta.trace_id or not meta.correlation_id:
            meta = replace(
                meta,
                trace_id=meta.trace_id or f"trace-{seq}",
                correlation_id=meta.correlation_id or f"corr-{seq}",
            )
        event_id = event.event_id or f"evt-{seq:08d}-{event.payload_hash[:8]}"
        return replace(event, sequence=seq, timestamp=ts, metadata=meta, event_id=event_id)

    # -- history -------------------------------------------------------------
    def history(self, event_filter: Optional[EventFilter] = None) -> EventHistory:
        with self._lock:
            history = self._history
        if event_filter is None:
            return history
        return EventHistory(history.filter(event_filter))

    # -- replay --------------------------------------------------------------
    def replay(
        self, events: Sequence[EnterpriseEvent], *, handler: Optional[EventHandler] = None,
        mode: str = "custom", criteria: str = "",
    ) -> EventReplay:
        if handler is not None and not callable(handler):
            raise ReplayError("replay handler must be callable")
        delivered_ids: List[str] = []
        for event in events:
            if handler is not None:
                handler(event)
            else:
                self._replay_to_subscribers(event)
            delivered_ids.append(event.event_id)
        with self._lock:
            self._replayed += len(delivered_ids)
            now = self._clock.now()
            replay_id = f"replay-{self._replayed:06d}"
        return EventReplay(replay_id, mode, criteria, tuple(delivered_ids), len(delivered_ids), now)

    def _replay_to_subscribers(self, event: EnterpriseEvent) -> None:
        with self._lock:
            matches = [
                (sub, self._handlers[sub.subscription_id])
                for sub in self._subscriptions.values()
                if sub.matches(event, None)
            ]
            matches.sort(key=lambda pair: (-pair[0].priority, pair[0].created_seq))
        for _, handler in matches:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - replay is best-effort
                self._log.warning("replay delivery failed for %s: %s", event.event_id, exc)

    def replay_by_topic(self, pattern: str, *, handler: Optional[EventHandler] = None) -> EventReplay:
        return self.replay(self.history().by_topic(pattern), handler=handler,
                           mode="topic", criteria=pattern)

    def replay_by_time(self, start: float, end: float, *,
                       handler: Optional[EventHandler] = None) -> EventReplay:
        return self.replay(self.history().by_time(start, end), handler=handler,
                           mode="time", criteria=f"{start}:{end}")

    def replay_by_correlation(self, correlation_id: str, *,
                              handler: Optional[EventHandler] = None) -> EventReplay:
        return self.replay(self.history().by_correlation(correlation_id), handler=handler,
                           mode="correlation", criteria=correlation_id)

    def replay_last_n(self, n: int, *, handler: Optional[EventHandler] = None) -> EventReplay:
        return self.replay(self.history().last(n), handler=handler,
                           mode="last_n", criteria=str(n))

    def replay_custom(self, event_filter: EventFilter, *,
                      handler: Optional[EventHandler] = None) -> EventReplay:
        return self.replay(self.history().filter(event_filter), handler=handler,
                           mode="custom", criteria=json.dumps(event_filter.to_dict(), sort_keys=True))

    # -- dead-letter queue ---------------------------------------------------
    def dead_letter_queue(self) -> Tuple[DeadLetterEvent, ...]:
        with self._lock:
            return self._dlq

    def recover_dead_letters(self, handler: EventHandler) -> int:
        """Attempt to redeliver pending dead letters with ``handler``.

        Recovered entries are removed from the queue; failures are marked
        ``FAILED`` and retained. Returns the number recovered.
        """

        if not callable(handler):
            raise DeadLetterError("recovery handler must be callable")
        with self._lock:
            current = self._dlq
        recovered = 0
        survivors: List[DeadLetterEvent] = []
        for dead in current:
            if dead.recovery_status in (RecoveryStatus.RECOVERED, RecoveryStatus.ABANDONED):
                survivors.append(dead)
                continue
            try:
                handler(dead.event)
                recovered += 1
            except Exception as exc:  # noqa: BLE001
                survivors.append(replace(
                    dead, recovery_status=RecoveryStatus.FAILED,
                    retry_count=dead.retry_count + 1,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    last_attempt_at=self._clock.now()))
        with self._lock:
            self._dlq = tuple(survivors)
        return recovered

    def clear_dead_letters(self) -> None:
        with self._lock:
            self._dlq = ()

    # -- statistics ----------------------------------------------------------
    def statistics(self) -> EventStatistics:
        with self._lock:
            attempted = self._delivered + self._dropped
            success_rate = self._delivered / attempted if attempted else 0.0
            failure_rate = self._dropped / attempted if attempted else 0.0
            avg_latency = self._latency_sum / self._latency_count if self._latency_count else 0.0
            return EventStatistics(
                published=self._published,
                delivered=self._delivered,
                dropped=self._dropped,
                unrouted=self._unrouted,
                replayed=self._replayed,
                subscribers=len(self._subscriptions),
                dead_letter_count=len(self._dlq),
                average_latency=avg_latency,
                failure_rate=failure_rate,
                delivery_success_rate=success_rate,
                topic_counts_json=_canonical_json(self._topic_counts),
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_default_event_bus() -> EnterpriseEventBus:
    """Factory wiring a deterministic event bus (logical clock)."""

    return EnterpriseEventBus(clock=LogicalClock(), max_retries=1, dead_letter_enabled=True)


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def _demo() -> int:
    print("=" * 70)
    print("Enterprise Event Bus - Week 10 Phase 3 Demonstration")
    print("=" * 70)

    bus = create_default_event_bus()
    received: List[str] = []

    def audit_logger(event: EnterpriseEvent) -> None:
        received.append(f"audit:{event.event_type.value}")

    def workflow_watcher(event: EnterpriseEvent) -> None:
        received.append(f"wf:{event.topic}")

    def flaky(event: EnterpriseEvent) -> None:
        raise RuntimeError("downstream unavailable")

    bus.subscribe(audit_logger, topic_pattern="#", subscriber_id="audit", priority=10)
    bus.subscribe(workflow_watcher, topic_pattern="workflow.*", subscriber_id="wf-watch",
                  priority=5)
    bus.subscribe(flaky, topic_pattern="risk.#", subscriber_id="risk-sink")
    bus.once(lambda e: received.append("once-fired"), topic_pattern="#")

    print(f"\nSubscribers: {bus.subscriber_count()}")

    bus.publish(EnterpriseEvent.create(
        EventType.WORKFLOW_STARTED, {"workflow_id": "WF-1"}, source="engine",
        correlation_id="run-1"))
    bus.publish(EnterpriseEvent.create(
        EventType.WORKFLOW_COMPLETED, {"workflow_id": "WF-1"}, source="engine",
        correlation_id="run-1"))
    bus.publish(EnterpriseEvent.create(
        EventType.RISK_THRESHOLD_EXCEEDED, {"asset_id": "A-7", "score": 0.95},
        priority=EventPriority.CRITICAL, source="risk", correlation_id="run-1"))

    print("\nDelivery log (deterministic order):")
    for entry in received:
        print(f"  - {entry}")

    print(f"\nHistory length      : {len(bus.history())}")
    print(f"Dead-letter entries : {len(bus.dead_letter_queue())}")

    replay = bus.replay_by_correlation("run-1", handler=lambda e: None)
    print(f"Replayed (corr=run-1): {replay.count}")

    stats = bus.statistics()
    print("\nStatistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key:<22}: {value}")

    # Determinism check: a fresh, identically-driven bus yields identical history.
    bus2 = create_default_event_bus()
    for sub_args in (("audit", "#", 10), ("wf-watch", "workflow.*", 5)):
        sid, pat, prio = sub_args
        bus2.subscribe(lambda e: None, topic_pattern=pat, subscriber_id=sid, priority=prio)
    for etype, payload, prio, corr in (
        (EventType.WORKFLOW_STARTED, {"workflow_id": "WF-1"}, EventPriority.NORMAL, "run-1"),
        (EventType.WORKFLOW_COMPLETED, {"workflow_id": "WF-1"}, EventPriority.NORMAL, "run-1"),
    ):
        bus2.publish(EnterpriseEvent.create(etype, payload, priority=prio,
                                            source="engine", correlation_id=corr))
    print(f"\nDeterministic history sequences: "
          f"{[e.sequence for e in bus2.history()]}")

    print("\nDemonstration complete.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Event Bus")
    parser.add_argument("--demo", action="store_true", help="Run the demonstration")
    args = parser.parse_args(argv)
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())