"""Enterprise Integration Layer (Week 10, Phase 5).

The single integration gateway for the Enterprise Digital Twin & Decision
Intelligence Platform. It coordinates communication between independent
modules - Prediction, Simulation, Knowledge, Executive Copilot, Workflow
Engine, Business Process Orchestrator, Event Bus, and Scheduler - while keeping
every module loosely coupled.

This layer holds no business logic. It is a **Facade** over a module
**Registry** of **Adapters**, with a **Strategy**-based routing engine, a
deterministic pipeline/dispatch engine (retry, timeout, fallback, circuit
breaker), per-module health monitoring, an immutable audit trail, and
observability statistics.

Integration is by composition only: real modules are wrapped as ``ModuleAdapter``
callables and registered; the layer imports and modifies nothing. Time is
supplied by an injectable clock and the layer is fully deterministic and
thread-safe.

Command-line demonstration::

    python src/integration/enterprise_integration_layer.py --demo
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
    FrozenSet,
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
    "IntegrationError",
    "ModuleRegistrationError",
    "RouteValidationError",
    "DispatchError",
    "ModuleNotFoundError",
    "CircuitOpenError",
    "PipelineError",
    "RegistryFrozenError",
    # Enums
    "ModuleType",
    "RouteStrategy",
    "ResponseStatus",
    "HealthState",
    "CircuitState",
    # Clocks
    "Clock",
    "LogicalClock",
    "SystemClock",
    # Domain model
    "ModuleDescriptor",
    "ModuleRegistration",
    "IntegrationContext",
    "IntegrationRequest",
    "IntegrationResponse",
    "RoutingRule",
    "IntegrationAudit",
    "IntegrationHealth",
    "IntegrationStatistics",
    "IntegrationSnapshot",
    # Engine
    "ModuleAdapter",
    "EnterpriseIntegrationLayer",
    "create_default_integration_layer",
]

logger = logging.getLogger("integration.layer")
if not logger.handlers:  # pragma: no cover - environmental
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class IntegrationError(Exception):
    """Base class for integration-layer errors."""


class ModuleRegistrationError(IntegrationError):
    """Raised for invalid or duplicate module registrations."""


class RouteValidationError(IntegrationError):
    """Raised when a request cannot be routed."""


class DispatchError(IntegrationError):
    """Raised when a request cannot be dispatched."""


class ModuleNotFoundError(IntegrationError):
    """Raised when a module id is unknown."""


class CircuitOpenError(IntegrationError):
    """Raised when a module's circuit breaker is open (used internally)."""


class PipelineError(IntegrationError):
    """Raised for invalid pipeline configuration."""


class RegistryFrozenError(IntegrationError):
    """Raised when mutating a frozen registry."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise IntegrationError(f"Expected a mapping, received {type(payload)!r}")
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def _parse_json(payload: str) -> Dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)


def _short_hash(*parts: Any, length: int = 12) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:length]


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
            raise IntegrationError(f"{value!r} is not a valid {cls.__name__}") from exc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class ModuleType(_ValueEnum):
    PREDICTION = "prediction"
    SIMULATION = "simulation"
    KNOWLEDGE = "knowledge"
    EXECUTIVE_COPILOT = "executive_copilot"
    WORKFLOW = "workflow"
    BUSINESS_PROCESS = "business_process"
    EVENT_BUS = "event_bus"
    SCHEDULER = "scheduler"
    CUSTOM = "custom"


class RouteStrategy(_ValueEnum):
    DIRECT = "direct"
    CAPABILITY = "capability"
    PRIORITY = "priority"
    CONDITIONAL = "conditional"
    FALLBACK = "fallback"
    BROADCAST = "broadcast"
    PIPELINE = "pipeline"


class ResponseStatus(_ValueEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    CIRCUIT_OPEN = "circuit_open"
    FALLBACK = "fallback"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class HealthState(_ValueEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(_ValueEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Clocks
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
            raise IntegrationError("cannot advance clock by a negative amount")
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
# Domain model - descriptors & registration
# ---------------------------------------------------------------------------
def _validate_version(version: str) -> None:
    if not version:
        raise ModuleRegistrationError("version must be non-empty")
    parts = version.split(".")
    if not all(p.isdigit() for p in parts):
        raise ModuleRegistrationError(f"invalid version: {version!r}")


@dataclass(frozen=True, slots=True)
class ModuleDescriptor:
    """Immutable description of an integrable module."""

    module_id: str
    module_type: ModuleType
    version: str = "1.0.0"
    capabilities: Tuple[str, ...] = ()
    priority: int = 0
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.module_id:
            raise ModuleRegistrationError("module_id must be non-empty")
        object.__setattr__(self, "module_type", ModuleType.coerce(self.module_type))
        _validate_version(self.version)
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "metadata_json", _canonical_json(_parse_json(self.metadata_json)))

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_type": self.module_type.value,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleDescriptor":
        return cls(
            module_id=data["module_id"],
            module_type=ModuleType.coerce(data["module_type"]),
            version=data.get("version", "1.0.0"),
            capabilities=tuple(data.get("capabilities", ())),
            priority=int(data.get("priority", 0)),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ModuleRegistration:
    """Immutable registration record (the adapter is held by the layer)."""

    descriptor: ModuleDescriptor
    registered_at: float = 0.0
    enabled: bool = True
    frozen: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ModuleDescriptor):
            raise ModuleRegistrationError("descriptor must be a ModuleDescriptor")

    @property
    def module_id(self) -> str:
        return self.descriptor.module_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "registered_at": self.registered_at,
            "enabled": self.enabled,
            "frozen": self.frozen,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModuleRegistration":
        return cls(
            descriptor=ModuleDescriptor.from_dict(data["descriptor"]),
            registered_at=float(data.get("registered_at", 0.0)),
            enabled=bool(data.get("enabled", True)),
            frozen=bool(data.get("frozen", False)),
        )


# ---------------------------------------------------------------------------
# Domain model - context & request/response
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntegrationContext:
    """Immutable data carried through a dispatch or pipeline."""

    correlation_id: str = ""
    trace_id: str = ""
    source: str = "integration"
    payload_json: str = "{}"
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_json", _canonical_json(_parse_json(self.payload_json)))
        object.__setattr__(self, "metadata_json", _canonical_json(_parse_json(self.metadata_json)))

    @classmethod
    def create(cls, payload: Optional[Mapping[str, Any]] = None, *, correlation_id: str = "",
               trace_id: str = "", source: str = "integration",
               metadata: Optional[Mapping[str, Any]] = None) -> "IntegrationContext":
        return cls(correlation_id=correlation_id, trace_id=trace_id, source=source,
                   payload_json=_canonical_json(payload or {}),
                   metadata_json=_canonical_json(metadata or {}))

    @property
    def payload(self) -> Dict[str, Any]:
        return _parse_json(self.payload_json)

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    def with_output(self, module_id: str, output: Mapping[str, Any]) -> "IntegrationContext":
        payload = self.payload
        payload[f"{module_id}_output"] = dict(output)
        payload["last_output"] = dict(output)
        return replace(self, payload_json=_canonical_json(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "source": self.source,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationContext":
        return cls(
            correlation_id=data.get("correlation_id", ""),
            trace_id=data.get("trace_id", ""),
            source=data.get("source", "integration"),
            payload_json=_canonical_json(data.get("payload") or {}),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class IntegrationRequest:
    """An immutable request to route and dispatch."""

    request_id: str
    route_strategy: RouteStrategy
    context: IntegrationContext
    target: str = ""
    capability: str = ""
    module_type: Optional[ModuleType] = None
    fallback_target: str = ""
    timeout: float = 0.0
    max_retries: int = 1
    retry_backoff: float = 0.0
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_strategy", RouteStrategy.coerce(self.route_strategy))
        if self.module_type is not None:
            object.__setattr__(self, "module_type", ModuleType.coerce(self.module_type))
        if not isinstance(self.context, IntegrationContext):
            raise DispatchError("context must be an IntegrationContext")
        if self.max_retries < 1:
            raise DispatchError("max_retries must be >= 1")
        if self.timeout < 0 or self.retry_backoff < 0:
            raise DispatchError("timeout and retry_backoff must be >= 0")

    @classmethod
    def create(cls, route_strategy: RouteStrategy, *, target: str = "", capability: str = "",
               module_type: Optional[ModuleType] = None, context: Optional[IntegrationContext] = None,
               fallback_target: str = "", timeout: float = 0.0, max_retries: int = 1,
               retry_backoff: float = 0.0, priority: int = 0,
               request_id: str = "") -> "IntegrationRequest":
        ctx = context or IntegrationContext.create()
        rid = request_id or f"req-{_short_hash(route_strategy, target, capability, ctx.payload_json)}"
        return cls(
            request_id=rid, route_strategy=RouteStrategy.coerce(route_strategy), context=ctx,
            target=target, capability=capability, module_type=module_type,
            fallback_target=fallback_target, timeout=timeout, max_retries=max_retries,
            retry_backoff=retry_backoff, priority=priority)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "route_strategy": self.route_strategy.value,
            "context": self.context.to_dict(),
            "target": self.target,
            "capability": self.capability,
            "module_type": self.module_type.value if self.module_type else None,
            "fallback_target": self.fallback_target,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationRequest":
        return cls(
            request_id=data["request_id"],
            route_strategy=RouteStrategy.coerce(data["route_strategy"]),
            context=IntegrationContext.from_dict(data["context"]),
            target=data.get("target", ""),
            capability=data.get("capability", ""),
            module_type=ModuleType.coerce(data["module_type"]) if data.get("module_type") else None,
            fallback_target=data.get("fallback_target", ""),
            timeout=float(data.get("timeout", 0.0)),
            max_retries=int(data.get("max_retries", 1)),
            retry_backoff=float(data.get("retry_backoff", 0.0)),
            priority=int(data.get("priority", 0)),
        )


@dataclass(frozen=True, slots=True)
class IntegrationResponse:
    """The immutable outcome of dispatching a request."""

    request_id: str
    module_id: str
    status: ResponseStatus
    route: str
    output_json: str = "{}"
    error: str = ""
    duration: float = 0.0
    attempts: int = 1
    fallback_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResponseStatus.coerce(self.status))
        object.__setattr__(self, "output_json", _canonical_json(_parse_json(self.output_json)))

    @property
    def succeeded(self) -> bool:
        return self.status in (ResponseStatus.SUCCESS, ResponseStatus.FALLBACK)

    @property
    def output(self) -> Dict[str, Any]:
        return _parse_json(self.output_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "module_id": self.module_id,
            "status": self.status.value,
            "route": self.route,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationResponse":
        return cls(
            request_id=data["request_id"],
            module_id=data["module_id"],
            status=ResponseStatus.coerce(data["status"]),
            route=data["route"],
            output_json=_canonical_json(data.get("output") or {}),
            error=data.get("error", ""),
            duration=float(data.get("duration", 0.0)),
            attempts=int(data.get("attempts", 1)),
            fallback_used=bool(data.get("fallback_used", False)),
        )


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RoutingRule:
    """A declarative conditional routing rule (Strategy configuration)."""

    rule_id: str
    target_module_id: str
    match_capability: Optional[str] = None
    match_module_type: Optional[ModuleType] = None
    condition_key: Optional[str] = None
    condition_value: Any = None
    priority: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise RouteValidationError("rule_id must be non-empty")
        if not self.target_module_id:
            raise RouteValidationError("target_module_id must be non-empty")
        if self.match_module_type is not None:
            object.__setattr__(self, "match_module_type", ModuleType.coerce(self.match_module_type))

    def matches(self, request: IntegrationRequest) -> bool:
        if not self.enabled:
            return False
        if self.match_capability is not None and request.capability != self.match_capability:
            return False
        if self.match_module_type is not None and request.module_type is not self.match_module_type:
            return False
        if self.condition_key is not None:
            meta = request.context.metadata
            if self.condition_key not in meta:
                return False
            if self.condition_value is not None and meta[self.condition_key] != self.condition_value:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "target_module_id": self.target_module_id,
            "match_capability": self.match_capability,
            "match_module_type": self.match_module_type.value if self.match_module_type else None,
            "condition_key": self.condition_key,
            "condition_value": self.condition_value,
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoutingRule":
        return cls(
            rule_id=data["rule_id"],
            target_module_id=data["target_module_id"],
            match_capability=data.get("match_capability"),
            match_module_type=ModuleType.coerce(data["match_module_type"]) if data.get("match_module_type") else None,
            condition_key=data.get("condition_key"),
            condition_value=data.get("condition_value"),
            priority=int(data.get("priority", 0)),
            enabled=bool(data.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# Audit / health / statistics / snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntegrationAudit:
    """An immutable audit record for one dispatch."""

    sequence: int
    request_id: str
    correlation_id: str
    module_id: str
    timestamp: float
    duration: float
    status: ResponseStatus
    route: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResponseStatus.coerce(self.status))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "module_id": self.module_id,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "status": self.status.value,
            "route": self.route,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationAudit":
        return cls(
            sequence=int(data["sequence"]),
            request_id=data["request_id"],
            correlation_id=data.get("correlation_id", ""),
            module_id=data["module_id"],
            timestamp=float(data["timestamp"]),
            duration=float(data["duration"]),
            status=ResponseStatus.coerce(data["status"]),
            route=data["route"],
        )


@dataclass(frozen=True, slots=True)
class IntegrationHealth:
    """An immutable health snapshot for one module."""

    module_id: str
    state: HealthState
    availability: float
    response_time: float
    failure_count: int
    success_count: int
    success_rate: float
    health_score: float
    last_seen: float
    heartbeat: int
    circuit_state: CircuitState = CircuitState.CLOSED

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", HealthState.coerce(self.state))
        object.__setattr__(self, "circuit_state", CircuitState.coerce(self.circuit_state))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "state": self.state.value,
            "availability": self.availability,
            "response_time": self.response_time,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "health_score": self.health_score,
            "last_seen": self.last_seen,
            "heartbeat": self.heartbeat,
            "circuit_state": self.circuit_state.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationHealth":
        return cls(
            module_id=data["module_id"],
            state=HealthState.coerce(data["state"]),
            availability=float(data["availability"]),
            response_time=float(data["response_time"]),
            failure_count=int(data["failure_count"]),
            success_count=int(data["success_count"]),
            success_rate=float(data["success_rate"]),
            health_score=float(data["health_score"]),
            last_seen=float(data["last_seen"]),
            heartbeat=int(data["heartbeat"]),
            circuit_state=CircuitState.coerce(data.get("circuit_state", "closed")),
        )


@dataclass(frozen=True, slots=True)
class IntegrationStatistics:
    """Aggregate observability statistics for the layer."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timeouts: int = 0
    fallbacks: int = 0
    circuit_open: int = 0
    pipeline_runs: int = 0
    average_dispatch_time: float = 0.0
    throughput: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    module_usage_json: str = "{}"
    route_counts_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_usage_json", _canonical_json(_parse_json(self.module_usage_json)))
        object.__setattr__(self, "route_counts_json", _canonical_json(_parse_json(self.route_counts_json)))

    @property
    def module_usage(self) -> Dict[str, int]:
        return _parse_json(self.module_usage_json)

    @property
    def route_counts(self) -> Dict[str, int]:
        return _parse_json(self.route_counts_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "fallbacks": self.fallbacks,
            "circuit_open": self.circuit_open,
            "pipeline_runs": self.pipeline_runs,
            "average_dispatch_time": self.average_dispatch_time,
            "throughput": self.throughput,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "module_usage": self.module_usage,
            "route_counts": self.route_counts,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationStatistics":
        return cls(
            total_requests=int(data.get("total_requests", 0)),
            successful=int(data.get("successful", 0)),
            failed=int(data.get("failed", 0)),
            timeouts=int(data.get("timeouts", 0)),
            fallbacks=int(data.get("fallbacks", 0)),
            circuit_open=int(data.get("circuit_open", 0)),
            pipeline_runs=int(data.get("pipeline_runs", 0)),
            average_dispatch_time=float(data.get("average_dispatch_time", 0.0)),
            throughput=float(data.get("throughput", 0.0)),
            success_rate=float(data.get("success_rate", 0.0)),
            failure_rate=float(data.get("failure_rate", 0.0)),
            module_usage_json=_canonical_json(data.get("module_usage") or {}),
            route_counts_json=_canonical_json(data.get("route_counts") or {}),
        )


@dataclass(frozen=True, slots=True)
class IntegrationSnapshot:
    """An immutable point-in-time snapshot of the whole layer."""

    timestamp: float
    modules: Tuple[ModuleRegistration, ...]
    statistics: IntegrationStatistics
    health: Tuple[IntegrationHealth, ...]
    audit_count: int
    registry_frozen: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "health", tuple(self.health))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "modules": [m.to_dict() for m in self.modules],
            "statistics": self.statistics.to_dict(),
            "health": [h.to_dict() for h in self.health],
            "audit_count": self.audit_count,
            "registry_frozen": self.registry_frozen,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntegrationSnapshot":
        return cls(
            timestamp=float(data["timestamp"]),
            modules=tuple(ModuleRegistration.from_dict(m) for m in data.get("modules", ())),
            statistics=IntegrationStatistics.from_dict(data.get("statistics") or {}),
            health=tuple(IntegrationHealth.from_dict(h) for h in data.get("health", ())),
            audit_count=int(data.get("audit_count", 0)),
            registry_frozen=bool(data.get("registry_frozen", False)),
        )


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------
@runtime_checkable
class ModuleAdapter(Protocol):
    """Wraps a real module; returns a JSON-like output or raises on failure."""

    def __call__(self, context: IntegrationContext, clock: Clock) -> Mapping[str, Any]:
        ...


def _echo_adapter(context: IntegrationContext, clock: Clock) -> Mapping[str, Any]:
    """A pure default adapter echoing the context payload (test-friendly)."""

    return {"echo": context.payload}


# ---------------------------------------------------------------------------
# Per-module mutable runtime state (guarded by the layer lock)
# ---------------------------------------------------------------------------
class _ModuleRuntime:
    __slots__ = ("success", "failure", "timeout", "duration_sum", "last_seen",
                 "heartbeat", "consecutive_failures", "circuit_state", "opened_at", "usage")

    def __init__(self) -> None:
        self.success = 0
        self.failure = 0
        self.timeout = 0
        self.duration_sum = 0.0
        self.last_seen = 0.0
        self.heartbeat = 0
        self.consecutive_failures = 0
        self.circuit_state = CircuitState.CLOSED
        self.opened_at = 0.0
        self.usage = 0


# ---------------------------------------------------------------------------
# The integration layer (Facade)
# ---------------------------------------------------------------------------
class EnterpriseIntegrationLayer:
    """A deterministic, thread-safe integration gateway.

    Modules are registered as adapters; requests are routed by strategy and
    dispatched with retry, timeout, circuit-breaking, and fallback. The layer
    is otherwise stateless per request and returns immutable response and
    snapshot objects.
    """

    def __init__(
        self,
        *,
        clock: Optional[Clock] = None,
        circuit_threshold: int = 3,
        circuit_cooldown: float = 5.0,
        max_audit: int = 0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._circuit_threshold = max(1, int(circuit_threshold))
        self._circuit_cooldown = float(circuit_cooldown)
        self._max_audit = max(0, int(max_audit))
        self._log = logger_ or logger
        self._lock = threading.RLock()
        self._modules: Dict[str, ModuleRegistration] = {}
        self._adapters: Dict[str, ModuleAdapter] = {}
        self._runtime: Dict[str, _ModuleRuntime] = {}
        self._rules: Dict[str, RoutingRule] = {}
        self._audit: Tuple[IntegrationAudit, ...] = ()
        self._audit_seq = 0
        self._registry_frozen = False
        # observability counters
        self._total = 0
        self._success = 0
        self._failure = 0
        self._timeouts = 0
        self._fallbacks = 0
        self._circuit_open = 0
        self._pipeline_runs = 0
        self._dispatch_time_sum = 0.0
        self._route_counts: Dict[str, int] = {}
        self._first_dispatch: Optional[float] = None

    # -- module registry -----------------------------------------------------
    def register_module(self, descriptor: ModuleDescriptor, adapter: ModuleAdapter, *,
                        overwrite: bool = False) -> ModuleRegistration:
        if not isinstance(descriptor, ModuleDescriptor):
            raise ModuleRegistrationError("descriptor must be a ModuleDescriptor")
        if not callable(adapter):
            raise ModuleRegistrationError("adapter must be callable")
        with self._lock:
            if self._registry_frozen:
                raise RegistryFrozenError("registry is frozen")
            if descriptor.module_id in self._modules and not overwrite:
                raise ModuleRegistrationError(f"module already registered: {descriptor.module_id}")
            registration = ModuleRegistration(descriptor, registered_at=self._clock.now())
            self._modules[descriptor.module_id] = registration
            self._adapters[descriptor.module_id] = adapter
            self._runtime.setdefault(descriptor.module_id, _ModuleRuntime())
            return registration

    def unregister_module(self, module_id: str) -> None:
        with self._lock:
            if self._registry_frozen:
                raise RegistryFrozenError("registry is frozen")
            if module_id not in self._modules:
                raise ModuleNotFoundError(module_id)
            del self._modules[module_id]
            self._adapters.pop(module_id, None)
            self._runtime.pop(module_id, None)

    def module_exists(self, module_id: str) -> bool:
        with self._lock:
            return module_id in self._modules

    def list_modules(self) -> Tuple[ModuleRegistration, ...]:
        with self._lock:
            return tuple(self._modules[k] for k in sorted(self._modules))

    def find_module(self, *, module_id: Optional[str] = None, capability: Optional[str] = None,
                    module_type: Optional[ModuleType] = None) -> Tuple[ModuleRegistration, ...]:
        with self._lock:
            regs = [self._modules[k] for k in sorted(self._modules)]
        mtype = ModuleType.coerce(module_type) if module_type is not None else None
        result = []
        for reg in regs:
            if module_id is not None and reg.module_id != module_id:
                continue
            if capability is not None and not reg.descriptor.has_capability(capability):
                continue
            if mtype is not None and reg.descriptor.module_type is not mtype:
                continue
            result.append(reg)
        result.sort(key=lambda r: (-r.descriptor.priority, r.module_id))
        return tuple(result)

    def freeze_registry(self) -> None:
        with self._lock:
            self._registry_frozen = True
            self._modules = {k: replace(v, frozen=True) for k, v in self._modules.items()}

    def registry_snapshot(self) -> IntegrationSnapshot:
        return self.snapshot()

    def register_route(self, rule: RoutingRule) -> None:
        with self._lock:
            if rule.rule_id in self._rules:
                raise RouteValidationError(f"route rule already exists: {rule.rule_id}")
            self._rules[rule.rule_id] = rule

    # -- routing -------------------------------------------------------------
    def resolve_route(self, request: IntegrationRequest) -> Tuple[List[str], str]:
        strat = request.route_strategy
        if strat is RouteStrategy.PIPELINE:
            raise RouteValidationError("use dispatch_pipeline for pipeline routing")
        with self._lock:
            modules = {k: v for k, v in self._modules.items() if v.enabled}
        if strat is RouteStrategy.DIRECT:
            if request.target not in modules:
                raise RouteValidationError(f"unknown target module: {request.target!r}")
            return [request.target], "direct"
        if strat is RouteStrategy.FALLBACK:
            if request.target not in modules:
                raise RouteValidationError(f"unknown target module: {request.target!r}")
            return [request.target], "fallback"
        if strat is RouteStrategy.CAPABILITY:
            candidates = self._by_capability(modules, request.capability)
            if not candidates:
                raise RouteValidationError(f"no module provides capability {request.capability!r}")
            return [candidates[0]], "capability"
        if strat is RouteStrategy.PRIORITY:
            candidates = self._by_priority(modules, request.module_type)
            if not candidates:
                raise RouteValidationError("no candidate modules for priority routing")
            return [candidates[0]], "priority"
        if strat is RouteStrategy.CONDITIONAL:
            target = self._conditional_target(request, modules)
            if target is None:
                raise RouteValidationError("no routing rule matched the request")
            return [target], "conditional"
        if strat is RouteStrategy.BROADCAST:
            candidates = self._broadcast_targets(modules, request)
            if not candidates:
                raise RouteValidationError("no modules matched for broadcast")
            return candidates, "broadcast"
        raise RouteValidationError(f"unsupported strategy: {strat}")  # pragma: no cover

    def validate_route(self, request: IntegrationRequest) -> bool:
        self.resolve_route(request)
        return True

    def _by_capability(self, modules: Mapping[str, ModuleRegistration], capability: str) -> List[str]:
        cands = [r for r in modules.values() if r.descriptor.has_capability(capability)]
        cands.sort(key=lambda r: (-r.descriptor.priority, r.module_id))
        return [r.module_id for r in cands]

    def _by_priority(self, modules: Mapping[str, ModuleRegistration],
                     module_type: Optional[ModuleType]) -> List[str]:
        cands = list(modules.values())
        if module_type is not None:
            cands = [r for r in cands if r.descriptor.module_type is module_type]
        cands.sort(key=lambda r: (-r.descriptor.priority, r.module_id))
        return [r.module_id for r in cands]

    def _conditional_target(self, request: IntegrationRequest,
                            modules: Mapping[str, ModuleRegistration]) -> Optional[str]:
        rules = sorted(self._rules.values(), key=lambda r: (-r.priority, r.rule_id))
        for rule in rules:
            if rule.matches(request) and rule.target_module_id in modules:
                return rule.target_module_id
        return None

    def _broadcast_targets(self, modules: Mapping[str, ModuleRegistration],
                           request: IntegrationRequest) -> List[str]:
        cands = list(modules.values())
        if request.capability:
            cands = [r for r in cands if r.descriptor.has_capability(request.capability)]
        elif request.module_type is not None:
            cands = [r for r in cands if r.descriptor.module_type is request.module_type]
        cands.sort(key=lambda r: (-r.descriptor.priority, r.module_id))
        return [r.module_id for r in cands]

    # -- dispatch ------------------------------------------------------------
    def dispatch(self, request: IntegrationRequest) -> IntegrationResponse:
        targets, route = self.resolve_route(request)
        if request.route_strategy is RouteStrategy.BROADCAST:
            return self._dispatch_broadcast(request, targets)
        return self._dispatch_one(request, targets[0], route)

    def dispatch_batch(self, requests: Sequence[IntegrationRequest]) -> Tuple[IntegrationResponse, ...]:
        return tuple(self.dispatch(r) for r in requests)

    def dispatch_pipeline(self, stages: Sequence[str], *,
                          context: Optional[IntegrationContext] = None,
                          by_capability: bool = False,
                          stop_on_failure: bool = True) -> Tuple[IntegrationResponse, ...]:
        if not stages:
            raise PipelineError("pipeline requires at least one stage")
        ctx = context or IntegrationContext.create()
        responses: List[IntegrationResponse] = []
        for stage in stages:
            if by_capability:
                req = IntegrationRequest.create(RouteStrategy.CAPABILITY, capability=stage, context=ctx)
                targets, route = self.resolve_route(req)
                module_id = targets[0]
            else:
                module_id = stage
                if not self.module_exists(module_id):
                    raise RouteValidationError(f"unknown pipeline stage module: {module_id!r}")
                req = IntegrationRequest.create(RouteStrategy.DIRECT, target=module_id, context=ctx)
                route = "pipeline"
            response = self._dispatch_one(req, module_id, "pipeline")
            responses.append(response)
            if response.succeeded:
                ctx = ctx.with_output(module_id, response.output)
            elif stop_on_failure:
                break
        with self._lock:
            self._pipeline_runs += 1
        return tuple(responses)

    def _dispatch_broadcast(self, request: IntegrationRequest,
                            targets: Sequence[str]) -> IntegrationResponse:
        outputs: Dict[str, Any] = {}
        statuses: List[bool] = []
        total_duration = 0.0
        for module_id in targets:
            resp = self._dispatch_one(request, module_id, "broadcast")
            outputs[module_id] = resp.output
            statuses.append(resp.succeeded)
            total_duration += resp.duration
        if all(statuses):
            status = ResponseStatus.SUCCESS
        elif any(statuses):
            status = ResponseStatus.PARTIAL
        else:
            status = ResponseStatus.FAILURE
        return IntegrationResponse(
            request_id=request.request_id, module_id="*broadcast*", status=status,
            route="broadcast", output_json=_canonical_json(outputs), duration=total_duration,
            attempts=1)

    def _dispatch_one(self, request: IntegrationRequest, module_id: str,
                      route: str) -> IntegrationResponse:
        status, output, error, duration, attempts = self._invoke(module_id, request)
        fallback_used = False
        if status is not ResponseStatus.SUCCESS and request.fallback_target:
            with self._lock:
                has_fb = request.fallback_target in self._modules
            if has_fb:
                fb_status, fb_out, fb_err, fb_dur, fb_att = self._invoke(
                    request.fallback_target, request, is_fallback=True)
                if fb_status is ResponseStatus.SUCCESS:
                    self._audit_and_count(request, request.fallback_target,
                                          ResponseStatus.FALLBACK, fb_dur, route + "->fallback")
                    return IntegrationResponse(
                        request_id=request.request_id, module_id=request.fallback_target,
                        status=ResponseStatus.FALLBACK, route=route + "->fallback",
                        output_json=_canonical_json(fb_out), duration=fb_dur, attempts=fb_att,
                        fallback_used=True)
                fallback_used = True
        self._audit_and_count(request, module_id, status, duration, route)
        return IntegrationResponse(
            request_id=request.request_id, module_id=module_id, status=status, route=route,
            output_json=_canonical_json(output), error=error, duration=duration,
            attempts=attempts, fallback_used=fallback_used)

    def _invoke(self, module_id: str, request: IntegrationRequest, *,
                is_fallback: bool = False) -> Tuple[ResponseStatus, Mapping[str, Any], str, float, int]:
        with self._lock:
            registration = self._modules.get(module_id)
            adapter = self._adapters.get(module_id)
            runtime = self._runtime.get(module_id)
            if registration is None or adapter is None or runtime is None:
                raise ModuleNotFoundError(module_id)
            if not registration.enabled:
                return ResponseStatus.REJECTED, {}, "module disabled", 0.0, 0
            # Circuit breaker gate.
            now = self._clock.now()
            if runtime.circuit_state is CircuitState.OPEN:
                if now - runtime.opened_at >= self._circuit_cooldown:
                    runtime.circuit_state = CircuitState.HALF_OPEN
                else:
                    return ResponseStatus.CIRCUIT_OPEN, {}, "circuit open", 0.0, 0
            clock = self._clock

        status = ResponseStatus.FAILURE
        output: Mapping[str, Any] = {}
        error = ""
        attempts = 0
        start = clock.now()
        for attempt in range(1, request.max_retries + 1):
            attempts = attempt
            backoff = request.retry_backoff * (attempt - 1)
            if backoff > 0:
                with self._lock:
                    clock.advance(backoff)
            call_start = clock.now()
            try:
                result = adapter(request.context, clock)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                error = f"{type(exc).__name__}: {exc}"
                status = ResponseStatus.FAILURE
                continue
            if not isinstance(result, Mapping):
                error = "adapter did not return a mapping"
                status = ResponseStatus.FAILURE
                continue
            call_duration = clock.now() - call_start
            if request.timeout and call_duration > request.timeout:
                status = ResponseStatus.TIMEOUT
                error = f"timeout ({call_duration} > {request.timeout})"
                continue
            status = ResponseStatus.SUCCESS
            output = result
            error = ""
            break
        finish = clock.now()
        duration = max(0.0, finish - start)
        with self._lock:
            clock.advance(1.0)  # deterministic separation between dispatches
            self._update_runtime(module_id, status, duration)
        return status, output, error, duration, attempts

    def _update_runtime(self, module_id: str, status: ResponseStatus, duration: float) -> None:
        runtime = self._runtime[module_id]
        runtime.usage += 1
        runtime.last_seen = self._clock.now()
        runtime.duration_sum += duration
        if status is ResponseStatus.SUCCESS:
            runtime.success += 1
            runtime.consecutive_failures = 0
            if runtime.circuit_state is CircuitState.HALF_OPEN:
                runtime.circuit_state = CircuitState.CLOSED
        elif status in (ResponseStatus.FAILURE, ResponseStatus.TIMEOUT):
            if status is ResponseStatus.TIMEOUT:
                runtime.timeout += 1
            else:
                runtime.failure += 1
            runtime.consecutive_failures += 1
            if (runtime.circuit_state is CircuitState.HALF_OPEN
                    or runtime.consecutive_failures >= self._circuit_threshold):
                runtime.circuit_state = CircuitState.OPEN
                runtime.opened_at = self._clock.now()

    def _audit_and_count(self, request: IntegrationRequest, module_id: str,
                         status: ResponseStatus, duration: float, route: str) -> None:
        with self._lock:
            if self._first_dispatch is None:
                self._first_dispatch = self._clock.now()
            self._total += 1
            self._dispatch_time_sum += duration
            self._route_counts[route] = self._route_counts.get(route, 0) + 1
            if status in (ResponseStatus.SUCCESS, ResponseStatus.FALLBACK):
                self._success += 1
            else:
                self._failure += 1
            if status is ResponseStatus.TIMEOUT:
                self._timeouts += 1
            if status is ResponseStatus.FALLBACK:
                self._fallbacks += 1
            if status is ResponseStatus.CIRCUIT_OPEN:
                self._circuit_open += 1
            audit = IntegrationAudit(
                sequence=self._audit_seq, request_id=request.request_id,
                correlation_id=request.context.correlation_id, module_id=module_id,
                timestamp=self._clock.now(), duration=duration, status=status, route=route)
            self._audit_seq += 1
            self._audit = self._audit + (audit,)
            if self._max_audit and len(self._audit) > self._max_audit:
                self._audit = self._audit[len(self._audit) - self._max_audit:]

    # -- health --------------------------------------------------------------
    def heartbeat(self, module_id: str) -> IntegrationHealth:
        with self._lock:
            if module_id not in self._runtime:
                raise ModuleNotFoundError(module_id)
            runtime = self._runtime[module_id]
            runtime.heartbeat += 1
            runtime.last_seen = self._clock.now()
        return self.health(module_id)

    def health(self, module_id: str) -> IntegrationHealth:
        with self._lock:
            if module_id not in self._runtime:
                raise ModuleNotFoundError(module_id)
            return self._compute_health(module_id)

    def health_all(self) -> Tuple[IntegrationHealth, ...]:
        with self._lock:
            return tuple(self._compute_health(mid) for mid in sorted(self._runtime))

    def _compute_health(self, module_id: str) -> IntegrationHealth:
        rt = self._runtime[module_id]
        total = rt.success + rt.failure + rt.timeout
        success_rate = rt.success / total if total else 0.0
        response_time = rt.duration_sum / total if total else 0.0
        if rt.circuit_state is CircuitState.OPEN:
            availability = 0.0
        elif rt.circuit_state is CircuitState.HALF_OPEN:
            availability = 0.5
        else:
            availability = 1.0
        if total == 0:
            state = HealthState.UNKNOWN
        elif rt.circuit_state is CircuitState.OPEN or success_rate <= 0.5:
            state = HealthState.UNHEALTHY
        elif success_rate < 0.9:
            state = HealthState.DEGRADED
        else:
            state = HealthState.HEALTHY
        health_score = round(success_rate * availability, 6)
        return IntegrationHealth(
            module_id=module_id, state=state, availability=availability,
            response_time=response_time, failure_count=rt.failure + rt.timeout,
            success_count=rt.success, success_rate=success_rate, health_score=health_score,
            last_seen=rt.last_seen, heartbeat=rt.heartbeat, circuit_state=rt.circuit_state)

    # -- observability -------------------------------------------------------
    def statistics(self) -> IntegrationStatistics:
        with self._lock:
            total = self._total
            avg = self._dispatch_time_sum / total if total else 0.0
            elapsed = (self._clock.now() - self._first_dispatch) if self._first_dispatch is not None else 0.0
            throughput = (total / elapsed) if elapsed > 0 else float(total)
            success_rate = self._success / total if total else 0.0
            failure_rate = self._failure / total if total else 0.0
            module_usage = {mid: rt.usage for mid, rt in self._runtime.items() if rt.usage}
            return IntegrationStatistics(
                total_requests=total, successful=self._success, failed=self._failure,
                timeouts=self._timeouts, fallbacks=self._fallbacks, circuit_open=self._circuit_open,
                pipeline_runs=self._pipeline_runs, average_dispatch_time=avg, throughput=throughput,
                success_rate=success_rate, failure_rate=failure_rate,
                module_usage_json=_canonical_json(module_usage),
                route_counts_json=_canonical_json(self._route_counts))

    def audit_log(self) -> Tuple[IntegrationAudit, ...]:
        with self._lock:
            return self._audit

    def snapshot(self) -> IntegrationSnapshot:
        with self._lock:
            modules = tuple(self._modules[k] for k in sorted(self._modules))
            audit_count = len(self._audit)
            frozen = self._registry_frozen
            now = self._clock.now()
        return IntegrationSnapshot(
            timestamp=now, modules=modules, statistics=self.statistics(),
            health=self.health_all(), audit_count=audit_count, registry_frozen=frozen)

    @property
    def registry_is_frozen(self) -> bool:
        with self._lock:
            return self._registry_frozen


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_default_integration_layer() -> EnterpriseIntegrationLayer:
    """Factory wiring a deterministic integration layer (logical clock)."""

    return EnterpriseIntegrationLayer(clock=LogicalClock(), circuit_threshold=3,
                                      circuit_cooldown=5.0)


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def _demo() -> int:
    print("=" * 70)
    print("Enterprise Integration Layer - Week 10 Phase 5 Demonstration")
    print("=" * 70)

    layer = create_default_integration_layer()

    def prediction(ctx, clock):
        return {"risk_score": 0.87, "asset_id": ctx.payload.get("asset_id", "A-1")}

    def risk(ctx, clock):
        score = ctx.payload.get("last_output", {}).get("risk_score", 0.0)
        return {"severity": "high" if score >= 0.8 else "low"}

    def knowledge(ctx, clock):
        return {"docs": ["maint-guide", "fault-history"]}

    def copilot(ctx, clock):
        return {"recommendation": "schedule_emergency_maintenance"}

    def workflow(ctx, clock):
        return {"workflow_id": "WF-EM-1"}

    def process(ctx, clock):
        return {"process_id": "BP-EM-1"}

    def scheduler(ctx, clock):
        return {"job_id": "JOB-EM-1"}

    def event_bus(ctx, clock):
        return {"published": True}

    specs = [
        ("prediction", ModuleType.PREDICTION, ("predict",), prediction, 10),
        ("risk", ModuleType.SIMULATION, ("assess_risk",), risk, 9),
        ("knowledge", ModuleType.KNOWLEDGE, ("retrieve",), knowledge, 8),
        ("copilot", ModuleType.EXECUTIVE_COPILOT, ("recommend",), copilot, 7),
        ("workflow", ModuleType.WORKFLOW, ("create_workflow",), workflow, 6),
        ("process", ModuleType.BUSINESS_PROCESS, ("orchestrate",), process, 5),
        ("scheduler", ModuleType.SCHEDULER, ("schedule",), scheduler, 4),
        ("eventbus", ModuleType.EVENT_BUS, ("publish",), event_bus, 3),
    ]
    for mid, mtype, caps, adapter, prio in specs:
        layer.register_module(ModuleDescriptor(mid, mtype, "1.0.0", caps, prio), adapter)

    print(f"\nRegistered modules: {[m.module_id for m in layer.list_modules()]}")

    # Direct dispatch.
    resp = layer.dispatch(IntegrationRequest.create(
        RouteStrategy.DIRECT, target="prediction",
        context=IntegrationContext.create({"asset_id": "TURBINE-7"}, correlation_id="run-1")))
    print(f"\nDirect dispatch -> prediction: {resp.status.value} {resp.output}")

    # Capability routing.
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="recommend"))
    print(f"Capability route 'recommend' -> {resp.module_id}: {resp.output}")

    # The flagship pipeline.
    print("\nRunning integration pipeline ...")
    pipeline = ["prediction", "risk", "knowledge", "copilot", "workflow", "process",
                "scheduler", "eventbus"]
    responses = layer.dispatch_pipeline(
        pipeline, context=IntegrationContext.create({"asset_id": "TURBINE-7"}, correlation_id="run-1"))
    for resp in responses:
        print(f"  {resp.module_id:<12} {resp.status.value:<8} {resp.output}")

    # Broadcast.
    layer.register_module(ModuleDescriptor("monitor", ModuleType.CUSTOM, "1.0.0", ("publish",), 1),
                          lambda ctx, clock: {"ack": True})
    bresp = layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="publish"))
    print(f"\nBroadcast 'publish' -> {bresp.status.value}: {list(bresp.output)}")

    stats = layer.statistics()
    print("\nObservability statistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key:<22}: {value}")

    print("\nModule health:")
    for h in layer.health_all():
        print(f"  {h.module_id:<12} {h.state.value:<9} score={h.health_score} circuit={h.circuit_state.value}")

    # Determinism check.
    layer2 = create_default_integration_layer()
    for mid, mtype, caps, adapter, prio in specs:
        layer2.register_module(ModuleDescriptor(mid, mtype, "1.0.0", caps, prio), adapter)
    r2 = layer2.dispatch_pipeline(pipeline, context=IntegrationContext.create(
        {"asset_id": "TURBINE-7"}, correlation_id="run-1"))
    identical = [r.to_dict() for r in responses] == [r.to_dict() for r in r2]
    print(f"\nDeterministic pipeline replay identical: {identical}")

    print("\nDemonstration complete.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Integration Layer")
    parser.add_argument("--demo", action="store_true", help="Run the demonstration")
    args = parser.parse_args(argv)
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())