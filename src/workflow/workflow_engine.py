"""Enterprise Workflow Engine (Week 10, Phase 1).

This module adds a deterministic, enterprise-grade workflow orchestration
engine on top of the existing Enterprise Digital Twin & Decision Intelligence
Platform. It converts executive recommendations into executable operational
workflows and drives them through a validated state machine while emitting an
immutable audit trail.

Design constraints honoured by this module:

* Pure Python plus NumPy only. No external workflow engines (Airflow, Temporal,
  Prefect, Celery), no cloud services, no network access.
* Fully deterministic. All time is supplied by an injectable :class:`Clock`;
  identifiers derived inside the engine are content-addressed hashes; iteration
  order is stable.
* Additive and non-invasive. This module imports **nothing** from Weeks 1-9.
  Integration with the Executive RAG Copilot, Executive Intelligence Agent,
  Decision Copilot, Scenario Planning, and Knowledge Agent is performed through
  the :class:`RecommendationSource` protocol and dependency-injected providers,
  so no existing public API is referenced, modified, or duplicated.
* Immutable domain model. Every dataclass is ``frozen`` and JSON serialisable
  through symmetric ``to_dict`` / ``from_dict`` methods. Execution progress is
  represented as a sequence of immutable :class:`WorkflowExecution` snapshots
  rather than in-place mutation.

The public surface is intentionally composed from small, independently testable
collaborators (state machine, rule engine, action engine, registry, audit
history) that the :class:`WorkflowEngine` orchestrates via dependency injection.

Run the command line demonstration with::

    python src/workflow/workflow_engine.py --demo
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
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
    "WorkflowError",
    "WorkflowValidationError",
    "InvalidStateTransitionError",
    "WorkflowExecutionError",
    "RuleEvaluationError",
    "ActionDispatchError",
    "RegistryError",
    # Enums
    "WorkflowType",
    "WorkflowState",
    "StepStatus",
    "ActionType",
    "ConditionOperator",
    "RuleType",
    "ExecutionMode",
    # Clocks
    "Clock",
    "LogicalClock",
    "SystemClock",
    "FixedClock",
    # Domain model
    "WorkflowCondition",
    "WorkflowAction",
    "RetryPolicy",
    "WorkflowStep",
    "WorkflowDefinition",
    "ActionOutcome",
    "WorkflowResult",
    "AuditEvent",
    "WorkflowHistory",
    "WorkflowExecution",
    "WorkflowStatistics",
    # Engine collaborators
    "WorkflowStateMachine",
    "Rule",
    "RuleEngine",
    "ActionHandler",
    "ActionEngine",
    "WorkflowRegistry",
    "RecommendationSource",
    "RecommendationCompiler",
    "WorkflowEngine",
    # Factories
    "create_default_action_engine",
    "create_default_engine",
]

logger = logging.getLogger("workflow.engine")
if not logger.handlers:  # pragma: no cover - logging wiring is environmental
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class WorkflowError(Exception):
    """Base class for every error raised by the workflow engine."""


class WorkflowValidationError(WorkflowError):
    """Raised when a definition or domain object fails validation."""


class InvalidStateTransitionError(WorkflowError):
    """Raised when an illegal workflow state transition is attempted."""


class WorkflowExecutionError(WorkflowError):
    """Raised when execution cannot proceed for structural reasons."""


class RuleEvaluationError(WorkflowError):
    """Raised when a rule cannot be evaluated."""


class ActionDispatchError(WorkflowError):
    """Raised when no handler is registered for an action type."""


class RegistryError(WorkflowError):
    """Raised for registry lookup / duplication problems."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Optional[Mapping[str, Any]]) -> str:
    """Return a stable, key-sorted JSON encoding for a mapping.

    Storing free-form mappings as canonical JSON strings keeps every dataclass
    hashable and frozen while remaining trivially serialisable and
    deterministic (key order is fixed).
    """

    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise WorkflowValidationError(
            f"Expected a mapping for canonical JSON, received {type(payload)!r}"
        )
    try:
        return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=_json_default)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise WorkflowValidationError(f"Mapping is not JSON serialisable: {exc}") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _parse_json(payload: str) -> Dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)


def _short_hash(*parts: Any, length: int = 12) -> str:
    """Deterministic short hash used for content-addressed identifiers."""

    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:length]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class _ValueEnum(str, Enum):
    """String-valued enum with a forgiving ``coerce`` constructor."""

    @classmethod
    def coerce(cls, value: Any) -> "_ValueEnum":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise WorkflowValidationError(
                f"{value!r} is not a valid {cls.__name__}"
            ) from exc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class WorkflowType(_ValueEnum):
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    EMERGENCY_MAINTENANCE = "emergency_maintenance"
    INSPECTION = "inspection"
    SHUTDOWN_PLANNING = "shutdown_planning"
    INVENTORY_PROCUREMENT = "inventory_procurement"
    EXECUTIVE_APPROVAL = "executive_approval"
    RISK_MITIGATION = "risk_mitigation"
    SAFETY_RESPONSE = "safety_response"
    KNOWLEDGE_REVIEW = "knowledge_review"
    SCENARIO_EVALUATION = "scenario_evaluation"


class WorkflowState(_ValueEnum):
    DRAFT = "draft"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


class StepStatus(_ValueEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class ActionType(_ValueEnum):
    CREATE_MAINTENANCE_TASK = "create_maintenance_task"
    SCHEDULE_INSPECTION = "schedule_inspection"
    GENERATE_EXECUTIVE_ALERT = "generate_executive_alert"
    RESERVE_INVENTORY = "reserve_inventory"
    ESCALATE_APPROVAL = "escalate_approval"
    CREATE_KNOWLEDGE_REVIEW = "create_knowledge_review"
    GENERATE_AUDIT_RECORD = "generate_audit_record"
    NO_OP = "no_op"


class ConditionOperator(_ValueEnum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    EXISTS = "exists"


class RuleType(_ValueEnum):
    RISK_THRESHOLD = "risk_threshold"
    ASSET_CRITICALITY = "asset_criticality"
    BUDGET_THRESHOLD = "budget_threshold"
    SCENARIO_SCORE = "scenario_score"
    KNOWLEDGE_CONFIDENCE = "knowledge_confidence"
    EXECUTIVE_CONFIDENCE = "executive_confidence"


class ExecutionMode(_ValueEnum):
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"


_TERMINAL_STATES = frozenset(
    {WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED}
)


# ---------------------------------------------------------------------------
# Clocks (dependency-injected time source for determinism)
# ---------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    """Time source abstraction.

    The engine never calls :func:`time.time` directly; all timestamps and
    backoff calculations flow through a clock, which makes executions fully
    reproducible under test.
    """

    def now(self) -> float:
        ...

    def advance(self, seconds: float) -> None:
        ...


class LogicalClock:
    """Deterministic monotonic clock starting at ``start`` seconds.

    Time only advances when the engine explicitly advances it (for step
    durations or retry backoff), which guarantees byte-identical executions.
    """

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise WorkflowExecutionError("Cannot advance clock by a negative amount")
        self._t += float(seconds)


class FixedClock:
    """Clock that never advances; useful for boundary tests."""

    __slots__ = ("_t",)

    def __init__(self, fixed: float = 0.0) -> None:
        self._t = float(fixed)

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:  # noqa: D401 - intentional no-op
        """Ignore advancement; the clock is fixed."""


class SystemClock:
    """Wall-clock time source for production use (non-deterministic)."""

    __slots__ = ("_t",)

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return time.time()

    def advance(self, seconds: float) -> None:  # pragma: no cover - timing
        if seconds > 0:
            time.sleep(seconds)


# ---------------------------------------------------------------------------
# Domain model - conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowCondition:
    """A single deterministic predicate evaluated against a context mapping."""

    field: str
    operator: ConditionOperator
    value: Any = None
    value2: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise WorkflowValidationError("Condition field must be a non-empty string")
        object.__setattr__(self, "operator", ConditionOperator.coerce(self.operator))
        if self.operator is ConditionOperator.BETWEEN and self.value2 is None:
            raise WorkflowValidationError("BETWEEN requires both value and value2")

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        present = self.field in context
        if self.operator is ConditionOperator.EXISTS:
            return present
        if not present:
            return False
        actual = context[self.field]
        op = self.operator
        try:
            if op is ConditionOperator.EQ:
                return actual == self.value
            if op is ConditionOperator.NEQ:
                return actual != self.value
            if op is ConditionOperator.IN:
                return actual in self.value  # type: ignore[operator]
            if op is ConditionOperator.NOT_IN:
                return actual not in self.value  # type: ignore[operator]
            if op is ConditionOperator.GT:
                return actual > self.value
            if op is ConditionOperator.GTE:
                return actual >= self.value
            if op is ConditionOperator.LT:
                return actual < self.value
            if op is ConditionOperator.LTE:
                return actual <= self.value
            if op is ConditionOperator.BETWEEN:
                return self.value <= actual <= self.value2
        except TypeError:
            # Incompatible operand types are treated as a non-match rather than
            # crashing an execution; this keeps the engine robust against
            # heterogeneous contexts while remaining deterministic.
            return False
        raise RuleEvaluationError(f"Unsupported operator {op!r}")  # pragma: no cover

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "value2": self.value2,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowCondition":
        return cls(
            field=data["field"],
            operator=ConditionOperator.coerce(data["operator"]),
            value=data.get("value"),
            value2=data.get("value2"),
        )


# ---------------------------------------------------------------------------
# Domain model - actions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowAction:
    """Declarative description of a deterministic side-effect-free action."""

    action_type: ActionType
    name: str
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", ActionType.coerce(self.action_type))
        if not isinstance(self.name, str) or not self.name:
            raise WorkflowValidationError("Action name must be a non-empty string")
        # Normalise parameters to canonical JSON for determinism / hashability.
        object.__setattr__(
            self, "parameters_json", _canonical_json(_parse_json(self.parameters_json))
        )

    @classmethod
    def create(
        cls,
        action_type: ActionType,
        name: str,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> "WorkflowAction":
        return cls(action_type, name, _canonical_json(parameters or {}))

    @property
    def parameters(self) -> Dict[str, Any]:
        return _parse_json(self.parameters_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "name": self.name,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowAction":
        return cls.create(
            ActionType.coerce(data["action_type"]),
            data["name"],
            data.get("parameters") or {},
        )


# ---------------------------------------------------------------------------
# Domain model - retry policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    """Deterministic retry configuration.

    Backoff never sleeps under a :class:`LogicalClock`; it only advances logical
    time, which keeps tests fast and reproducible. The delay before attempt *n*
    (1-indexed, n>=2) is ``backoff_base * backoff_factor ** (n - 2)``.
    """

    max_attempts: int = 1
    backoff_base: float = 0.0
    backoff_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise WorkflowValidationError("max_attempts must be >= 1")
        if self.backoff_base < 0:
            raise WorkflowValidationError("backoff_base must be >= 0")
        if self.backoff_factor <= 0:
            raise WorkflowValidationError("backoff_factor must be > 0")

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay *before* the given 1-indexed attempt."""

        if attempt <= 1:
            return 0.0
        return float(self.backoff_base) * float(self.backoff_factor) ** (attempt - 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "backoff_base": self.backoff_base,
            "backoff_factor": self.backoff_factor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetryPolicy":
        return cls(
            max_attempts=int(data.get("max_attempts", 1)),
            backoff_base=float(data.get("backoff_base", 0.0)),
            backoff_factor=float(data.get("backoff_factor", 1.0)),
        )


_DEFAULT_RETRY = RetryPolicy()


# ---------------------------------------------------------------------------
# Domain model - steps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowStep:
    """A single unit of work within a workflow definition."""

    step_id: str
    name: str
    action: WorkflowAction
    conditions: Tuple[WorkflowCondition, ...] = ()
    condition_logic: str = "AND"
    next_steps: Tuple[str, ...] = ()
    on_failure: Tuple[str, ...] = ()
    parallel_group: Optional[str] = None
    requires_approval: bool = False
    retry_policy: RetryPolicy = _DEFAULT_RETRY
    timeout: Optional[float] = None
    optional: bool = False
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id:
            raise WorkflowValidationError("step_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise WorkflowValidationError("step name must be a non-empty string")
        if not isinstance(self.action, WorkflowAction):
            raise WorkflowValidationError("step action must be a WorkflowAction")
        logic = str(self.condition_logic).upper()
        if logic not in ("AND", "OR"):
            raise WorkflowValidationError("condition_logic must be 'AND' or 'OR'")
        object.__setattr__(self, "condition_logic", logic)
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "next_steps", tuple(self.next_steps))
        object.__setattr__(self, "on_failure", tuple(self.on_failure))
        if self.timeout is not None and self.timeout < 0:
            raise WorkflowValidationError("timeout must be >= 0")
        object.__setattr__(
            self, "metadata_json", _canonical_json(_parse_json(self.metadata_json))
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    def guard_passes(self, context: Mapping[str, Any]) -> bool:
        """Evaluate the step guard conditions against ``context``."""

        if not self.conditions:
            return True
        results = [c.evaluate(context) for c in self.conditions]
        if self.condition_logic == "OR":
            return any(results)
        return all(results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "action": self.action.to_dict(),
            "conditions": [c.to_dict() for c in self.conditions],
            "condition_logic": self.condition_logic,
            "next_steps": list(self.next_steps),
            "on_failure": list(self.on_failure),
            "parallel_group": self.parallel_group,
            "requires_approval": self.requires_approval,
            "retry_policy": self.retry_policy.to_dict(),
            "timeout": self.timeout,
            "optional": self.optional,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowStep":
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            action=WorkflowAction.from_dict(data["action"]),
            conditions=tuple(
                WorkflowCondition.from_dict(c) for c in data.get("conditions", ())
            ),
            condition_logic=data.get("condition_logic", "AND"),
            next_steps=tuple(data.get("next_steps", ())),
            on_failure=tuple(data.get("on_failure", ())),
            parallel_group=data.get("parallel_group"),
            requires_approval=bool(data.get("requires_approval", False)),
            retry_policy=RetryPolicy.from_dict(data.get("retry_policy", {})),
            timeout=data.get("timeout"),
            optional=bool(data.get("optional", False)),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Domain model - definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowDefinition:
    """An immutable, validated description of a workflow."""

    workflow_id: str
    name: str
    workflow_type: WorkflowType
    steps: Tuple[WorkflowStep, ...]
    version: str = "1.0.0"
    description: str = ""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    parallel_policy: str = "ALL"
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.workflow_id, str) or not self.workflow_id:
            raise WorkflowValidationError("workflow_id must be a non-empty string")
        object.__setattr__(self, "workflow_type", WorkflowType.coerce(self.workflow_type))
        object.__setattr__(self, "execution_mode", ExecutionMode.coerce(self.execution_mode))
        object.__setattr__(self, "steps", tuple(self.steps))
        policy = str(self.parallel_policy).upper()
        if policy not in ("ALL", "ANY"):
            raise WorkflowValidationError("parallel_policy must be 'ALL' or 'ANY'")
        object.__setattr__(self, "parallel_policy", policy)
        object.__setattr__(
            self, "metadata_json", _canonical_json(_parse_json(self.metadata_json))
        )
        self.validate()

    def validate(self) -> None:
        if not self.steps:
            raise WorkflowValidationError("A workflow must contain at least one step")
        seen: set = set()
        for step in self.steps:
            if not isinstance(step, WorkflowStep):
                raise WorkflowValidationError("All steps must be WorkflowStep instances")
            if step.step_id in seen:
                raise WorkflowValidationError(f"Duplicate step_id: {step.step_id}")
            seen.add(step.step_id)
        known = set(seen)
        for step in self.steps:
            for ref in tuple(step.next_steps) + tuple(step.on_failure):
                if ref not in known:
                    raise WorkflowValidationError(
                        f"Step {step.step_id} references unknown step {ref!r}"
                    )

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    @property
    def step_ids(self) -> Tuple[str, ...]:
        return tuple(s.step_id for s in self.steps)

    def get_step(self, step_id: str) -> WorkflowStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise WorkflowValidationError(f"Unknown step_id: {step_id}")

    def fingerprint(self) -> str:
        """Deterministic content hash of the definition."""

        return _short_hash(json.dumps(self.to_dict(), sort_keys=True), length=16)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "workflow_type": self.workflow_type.value,
            "steps": [s.to_dict() for s in self.steps],
            "version": self.version,
            "description": self.description,
            "execution_mode": self.execution_mode.value,
            "parallel_policy": self.parallel_policy,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowDefinition":
        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            workflow_type=WorkflowType.coerce(data["workflow_type"]),
            steps=tuple(WorkflowStep.from_dict(s) for s in data["steps"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            execution_mode=ExecutionMode.coerce(data.get("execution_mode", "sequential")),
            parallel_policy=data.get("parallel_policy", "ALL"),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Domain model - action outcome & step result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionOutcome:
    """The deterministic result returned by an action handler."""

    success: bool
    output_json: str = "{}"
    error: str = ""
    duration: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_json", _canonical_json(_parse_json(self.output_json))
        )
        if self.duration < 0:
            raise WorkflowValidationError("duration must be >= 0")

    @classmethod
    def succeeded(
        cls, output: Optional[Mapping[str, Any]] = None, duration: float = 0.0
    ) -> "ActionOutcome":
        return cls(True, _canonical_json(output or {}), "", duration)

    @classmethod
    def failed(
        cls, error: str, output: Optional[Mapping[str, Any]] = None, duration: float = 0.0
    ) -> "ActionOutcome":
        return cls(False, _canonical_json(output or {}), error, duration)

    @property
    def output(self) -> Dict[str, Any]:
        return _parse_json(self.output_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionOutcome":
        return cls(
            success=bool(data["success"]),
            output_json=_canonical_json(data.get("output") or {}),
            error=data.get("error", ""),
            duration=float(data.get("duration", 0.0)),
        )


@dataclass(frozen=True)
class WorkflowResult:
    """The recorded outcome of a single executed (or skipped) step."""

    step_id: str
    status: StepStatus
    output_json: str = "{}"
    error: str = ""
    attempts: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", StepStatus.coerce(self.status))
        object.__setattr__(
            self, "output_json", _canonical_json(_parse_json(self.output_json))
        )

    @property
    def output(self) -> Dict[str, Any]:
        return _parse_json(self.output_json)

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def succeeded(self) -> bool:
        return self.status is StepStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowResult":
        return cls(
            step_id=data["step_id"],
            status=StepStatus.coerce(data["status"]),
            output_json=_canonical_json(data.get("output") or {}),
            error=data.get("error", ""),
            attempts=int(data.get("attempts", 0)),
            started_at=float(data.get("started_at", 0.0)),
            finished_at=float(data.get("finished_at", 0.0)),
        )


# ---------------------------------------------------------------------------
# Audit model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditEvent:
    """An immutable audit record. Ordering is guaranteed by ``sequence``."""

    sequence: int
    timestamp: float
    execution_id: str
    workflow_id: str
    step: str
    event: str
    actor: str
    result: str
    detail_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "detail_json", _canonical_json(_parse_json(self.detail_json))
        )

    @property
    def detail(self) -> Dict[str, Any]:
        return _parse_json(self.detail_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "step": self.step,
            "event": self.event,
            "actor": self.actor,
            "result": self.result,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditEvent":
        return cls(
            sequence=int(data["sequence"]),
            timestamp=float(data["timestamp"]),
            execution_id=data["execution_id"],
            workflow_id=data["workflow_id"],
            step=data.get("step", ""),
            event=data["event"],
            actor=data.get("actor", "system"),
            result=data.get("result", ""),
            detail_json=_canonical_json(data.get("detail") or {}),
        )


@dataclass(frozen=True)
class WorkflowHistory:
    """An append-only, immutable timeline of audit events."""

    events: Tuple[AuditEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self.events)

    @property
    def next_sequence(self) -> int:
        return self.events[-1].sequence + 1 if self.events else 0

    def append(self, event: AuditEvent) -> "WorkflowHistory":
        """Return a *new* history with ``event`` appended (immutability)."""

        return WorkflowHistory(self.events + (event,))

    def filter_by_event(self, event: str) -> Tuple[AuditEvent, ...]:
        return tuple(e for e in self.events if e.event == event)

    def filter_by_step(self, step: str) -> Tuple[AuditEvent, ...]:
        return tuple(e for e in self.events if e.step == step)

    def timeline(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowHistory":
        return cls(tuple(AuditEvent.from_dict(e) for e in data.get("events", ())))


# ---------------------------------------------------------------------------
# Execution snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowExecution:
    """An immutable snapshot of a workflow execution at a point in time."""

    execution_id: str
    workflow_id: str
    workflow_type: WorkflowType
    state: WorkflowState
    results: Tuple[WorkflowResult, ...] = ()
    history: WorkflowHistory = field(default_factory=WorkflowHistory)
    context_json: str = "{}"
    current_step: Optional[str] = None
    pending_approval_step: Optional[str] = None
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_type", WorkflowType.coerce(self.workflow_type))
        object.__setattr__(self, "state", WorkflowState.coerce(self.state))
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self, "context_json", _canonical_json(_parse_json(self.context_json))
        )

    @property
    def context(self) -> Dict[str, Any]:
        return _parse_json(self.context_json)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def completed_steps(self) -> Tuple[WorkflowResult, ...]:
        return tuple(r for r in self.results if r.status is StepStatus.COMPLETED)

    @property
    def failed_steps(self) -> Tuple[WorkflowResult, ...]:
        return tuple(
            r for r in self.results if r.status in (StepStatus.FAILED, StepStatus.TIMEOUT)
        )

    @property
    def total_attempts(self) -> int:
        return int(sum(r.attempts for r in self.results))

    @property
    def total_retries(self) -> int:
        return int(sum(max(0, r.attempts - 1) for r in self.results))

    @property
    def duration(self) -> float:
        return max(0.0, self.updated_at - self.created_at)

    def result_for(self, step_id: str) -> Optional[WorkflowResult]:
        for r in self.results:
            if r.step_id == step_id:
                return r
        return None

    def metrics(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "state": self.state.value,
            "step_count": len(self.results),
            "completed": len(self.completed_steps),
            "failed": len(self.failed_steps),
            "skipped": sum(1 for r in self.results if r.status is StepStatus.SKIPPED),
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "duration": self.duration,
            "audit_events": len(self.history),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
            "state": self.state.value,
            "results": [r.to_dict() for r in self.results],
            "history": self.history.to_dict(),
            "context": self.context,
            "current_step": self.current_step,
            "pending_approval_step": self.pending_approval_step,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowExecution":
        return cls(
            execution_id=data["execution_id"],
            workflow_id=data["workflow_id"],
            workflow_type=WorkflowType.coerce(data["workflow_type"]),
            state=WorkflowState.coerce(data["state"]),
            results=tuple(WorkflowResult.from_dict(r) for r in data.get("results", ())),
            history=WorkflowHistory.from_dict(data.get("history", {})),
            context_json=_canonical_json(data.get("context") or {}),
            current_step=data.get("current_step"),
            pending_approval_step=data.get("pending_approval_step"),
            error=data.get("error", ""),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkflowStatistics:
    """Aggregate statistics computed over a collection of executions."""

    workflow_count: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    running: int = 0
    paused: int = 0
    average_duration: float = 0.0
    total_retries: int = 0
    approval_count: int = 0
    success_rate: float = 0.0

    @classmethod
    def from_executions(
        cls, executions: Sequence[WorkflowExecution]
    ) -> "WorkflowStatistics":
        count = len(executions)
        if count == 0:
            return cls()
        states = [e.state for e in executions]
        completed = states.count(WorkflowState.COMPLETED)
        failed = states.count(WorkflowState.FAILED)
        cancelled = states.count(WorkflowState.CANCELLED)
        running = states.count(WorkflowState.RUNNING)
        paused = states.count(WorkflowState.PAUSED)
        durations = np.array([e.duration for e in executions], dtype=float)
        avg_duration = float(np.mean(durations)) if durations.size else 0.0
        total_retries = int(sum(e.total_retries for e in executions))
        approvals = 0
        for e in executions:
            approvals += len(e.history.filter_by_event("APPROVAL_GRANTED"))
        finished = completed + failed + cancelled
        success_rate = float(completed / finished) if finished else 0.0
        return cls(
            workflow_count=count,
            completed=completed,
            failed=failed,
            cancelled=cancelled,
            running=running,
            paused=paused,
            average_duration=avg_duration,
            total_retries=total_retries,
            approval_count=approvals,
            success_rate=success_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_count": self.workflow_count,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "running": self.running,
            "paused": self.paused,
            "average_duration": self.average_duration,
            "total_retries": self.total_retries,
            "approval_count": self.approval_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowStatistics":
        return cls(
            workflow_count=int(data.get("workflow_count", 0)),
            completed=int(data.get("completed", 0)),
            failed=int(data.get("failed", 0)),
            cancelled=int(data.get("cancelled", 0)),
            running=int(data.get("running", 0)),
            paused=int(data.get("paused", 0)),
            average_duration=float(data.get("average_duration", 0.0)),
            total_retries=int(data.get("total_retries", 0)),
            approval_count=int(data.get("approval_count", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
        )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class WorkflowStateMachine:
    """Validated deterministic transitions between workflow states."""

    _ALLOWED: Dict[WorkflowState, frozenset] = {
        WorkflowState.DRAFT: frozenset({WorkflowState.PENDING, WorkflowState.CANCELLED}),
        WorkflowState.PENDING: frozenset(
            {WorkflowState.RUNNING, WorkflowState.CANCELLED}
        ),
        WorkflowState.RUNNING: frozenset(
            {
                WorkflowState.PAUSED,
                WorkflowState.COMPLETED,
                WorkflowState.FAILED,
                WorkflowState.CANCELLED,
            }
        ),
        WorkflowState.PAUSED: frozenset(
            {WorkflowState.RUNNING, WorkflowState.CANCELLED, WorkflowState.FAILED}
        ),
        WorkflowState.COMPLETED: frozenset(),
        WorkflowState.FAILED: frozenset(),
        WorkflowState.CANCELLED: frozenset(),
    }

    def allowed_transitions(self, state: WorkflowState) -> frozenset:
        return self._ALLOWED[WorkflowState.coerce(state)]

    def can_transition(self, src: WorkflowState, dst: WorkflowState) -> bool:
        return WorkflowState.coerce(dst) in self._ALLOWED[WorkflowState.coerce(src)]

    def validate(self, src: WorkflowState, dst: WorkflowState) -> None:
        if not self.can_transition(src, dst):
            raise InvalidStateTransitionError(
                f"Illegal transition {WorkflowState.coerce(src).value} -> "
                f"{WorkflowState.coerce(dst).value}"
            )

    @staticmethod
    def is_terminal(state: WorkflowState) -> bool:
        return WorkflowState.coerce(state).is_terminal


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """A named, typed, configurable rule wrapping a condition."""

    rule_id: str
    rule_type: RuleType
    condition: WorkflowCondition
    description: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise WorkflowValidationError("rule_id must be a non-empty string")
        object.__setattr__(self, "rule_type", RuleType.coerce(self.rule_type))
        if not isinstance(self.condition, WorkflowCondition):
            raise WorkflowValidationError("Rule condition must be a WorkflowCondition")

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False
        return self.condition.evaluate(context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "condition": self.condition.to_dict(),
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rule":
        return cls(
            rule_id=data["rule_id"],
            rule_type=RuleType.coerce(data["rule_type"]),
            condition=WorkflowCondition.from_dict(data["condition"]),
            description=data.get("description", ""),
            enabled=bool(data.get("enabled", True)),
        )


class RuleEngine:
    """A configurable, deterministic registry of rules."""

    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules: Dict[str, Rule] = {}
        for rule in rules or ():
            self.register(rule)

    def register(self, rule: Rule, *, overwrite: bool = False) -> None:
        if not isinstance(rule, Rule):
            raise WorkflowValidationError("Only Rule instances may be registered")
        if rule.rule_id in self._rules and not overwrite:
            raise RegistryError(f"Rule already registered: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> Rule:
        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise RegistryError(f"Unknown rule: {rule_id}") from exc

    def remove(self, rule_id: str) -> None:
        if rule_id not in self._rules:
            raise RegistryError(f"Unknown rule: {rule_id}")
        del self._rules[rule_id]

    def list_rules(self) -> Tuple[Rule, ...]:
        return tuple(self._rules[k] for k in sorted(self._rules))

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def evaluate(self, rule_id: str, context: Mapping[str, Any]) -> bool:
        return self.get(rule_id).evaluate(context)

    def evaluate_all(self, context: Mapping[str, Any]) -> Dict[str, bool]:
        return {rid: self._rules[rid].evaluate(context) for rid in sorted(self._rules)}

    def matching_rules(self, context: Mapping[str, Any]) -> Tuple[Rule, ...]:
        return tuple(r for r in self.list_rules() if r.evaluate(context))

    def matching_by_type(
        self, rule_type: RuleType, context: Mapping[str, Any]
    ) -> Tuple[Rule, ...]:
        rule_type = RuleType.coerce(rule_type)
        return tuple(
            r for r in self.list_rules() if r.rule_type is rule_type and r.evaluate(context)
        )


# ---------------------------------------------------------------------------
# Action engine
# ---------------------------------------------------------------------------
@runtime_checkable
class ActionHandler(Protocol):
    """Callable contract for deterministic action handlers."""

    def __call__(
        self, action: WorkflowAction, context: Mapping[str, Any], clock: Clock
    ) -> ActionOutcome:
        ...


def _deterministic_id(prefix: str, action: WorkflowAction, context: Mapping[str, Any]) -> str:
    keys = ("asset_id", "workflow_id", "recommendation_id")
    salt = {k: context.get(k) for k in keys if k in context}
    return f"{prefix}-{_short_hash(action.name, action.parameters_json, salt)}"


def _make_record_handler(prefix: str) -> ActionHandler:
    """Build a default handler that records intent without external effects.

    Default handlers are pure: they synthesise a deterministic output record
    describing the operation that *would* be performed in a downstream system.
    They never touch Weeks 1-9 modules; real side effects are wired in by
    injecting bespoke handlers through :meth:`ActionEngine.register_handler`.
    """

    def handler(
        action: WorkflowAction, context: Mapping[str, Any], clock: Clock
    ) -> ActionOutcome:
        record = {
            "record_id": _deterministic_id(prefix, action, context),
            "action": action.action_type.value,
            "name": action.name,
            "parameters": action.parameters,
        }
        return ActionOutcome.succeeded(record, duration=0.0)

    return handler


def _audit_record_handler(
    action: WorkflowAction, context: Mapping[str, Any], clock: Clock
) -> ActionOutcome:
    record = {
        "record_id": _deterministic_id("audit", action, context),
        "logged_at": clock.now(),
        "name": action.name,
    }
    return ActionOutcome.succeeded(record, duration=0.0)


def _no_op_handler(
    action: WorkflowAction, context: Mapping[str, Any], clock: Clock
) -> ActionOutcome:
    return ActionOutcome.succeeded({"noop": True}, duration=0.0)


class ActionEngine:
    """Dispatches actions to deterministic, dependency-injected handlers."""

    def __init__(self, handlers: Optional[Mapping[ActionType, ActionHandler]] = None) -> None:
        self._handlers: Dict[ActionType, ActionHandler] = {}
        if handlers:
            for action_type, handler in handlers.items():
                self.register_handler(action_type, handler)

    def register_handler(self, action_type: ActionType, handler: ActionHandler) -> None:
        action_type = ActionType.coerce(action_type)
        if not callable(handler):
            raise WorkflowValidationError("handler must be callable")
        self._handlers[action_type] = handler

    def has_handler(self, action_type: ActionType) -> bool:
        return ActionType.coerce(action_type) in self._handlers

    def registered_types(self) -> Tuple[ActionType, ...]:
        return tuple(sorted(self._handlers, key=lambda a: a.value))

    def dispatch(
        self, action: WorkflowAction, context: Mapping[str, Any], clock: Clock
    ) -> ActionOutcome:
        action_type = ActionType.coerce(action.action_type)
        handler = self._handlers.get(action_type)
        if handler is None:
            raise ActionDispatchError(f"No handler registered for {action_type.value}")
        outcome = handler(action, context, clock)
        if not isinstance(outcome, ActionOutcome):
            raise ActionDispatchError(
                f"Handler for {action_type.value} returned {type(outcome)!r}, "
                "expected ActionOutcome"
            )
        return outcome


def create_default_action_engine() -> ActionEngine:
    """Factory producing an action engine with deterministic default handlers."""

    engine = ActionEngine()
    engine.register_handler(
        ActionType.CREATE_MAINTENANCE_TASK, _make_record_handler("maint-task")
    )
    engine.register_handler(
        ActionType.SCHEDULE_INSPECTION, _make_record_handler("inspection")
    )
    engine.register_handler(
        ActionType.GENERATE_EXECUTIVE_ALERT, _make_record_handler("exec-alert")
    )
    engine.register_handler(
        ActionType.RESERVE_INVENTORY, _make_record_handler("inventory")
    )
    engine.register_handler(
        ActionType.ESCALATE_APPROVAL, _make_record_handler("approval")
    )
    engine.register_handler(
        ActionType.CREATE_KNOWLEDGE_REVIEW, _make_record_handler("knowledge-review")
    )
    engine.register_handler(ActionType.GENERATE_AUDIT_RECORD, _audit_record_handler)
    engine.register_handler(ActionType.NO_OP, _no_op_handler)
    return engine


# ---------------------------------------------------------------------------
# Definition registry
# ---------------------------------------------------------------------------
class WorkflowRegistry:
    """A repository of workflow definitions keyed by ``workflow_id``."""

    def __init__(self) -> None:
        self._defs: Dict[str, WorkflowDefinition] = {}

    def register_workflow(
        self, definition: WorkflowDefinition, *, overwrite: bool = False
    ) -> None:
        if not isinstance(definition, WorkflowDefinition):
            raise WorkflowValidationError("Only WorkflowDefinition instances may register")
        if definition.workflow_id in self._defs and not overwrite:
            raise RegistryError(f"Workflow already registered: {definition.workflow_id}")
        self._defs[definition.workflow_id] = definition

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._defs[workflow_id]
        except KeyError as exc:
            raise RegistryError(f"Unknown workflow: {workflow_id}") from exc

    def list_workflows(self) -> Tuple[WorkflowDefinition, ...]:
        return tuple(self._defs[k] for k in sorted(self._defs))

    def remove_workflow(self, workflow_id: str) -> None:
        if workflow_id not in self._defs:
            raise RegistryError(f"Unknown workflow: {workflow_id}")
        del self._defs[workflow_id]

    def __len__(self) -> int:
        return len(self._defs)

    def __contains__(self, workflow_id: object) -> bool:
        return workflow_id in self._defs


# ---------------------------------------------------------------------------
# Recommendation ingestion (integration seam, no Weeks 1-9 imports)
# ---------------------------------------------------------------------------
@runtime_checkable
class RecommendationSource(Protocol):
    """Structural contract for upstream recommendation producers.

    The Executive RAG Copilot, Executive Intelligence Agent, Decision Copilot,
    Scenario Planning and Knowledge Agent all emit recommendation-shaped data.
    Any object exposing ``as_recommendation`` returning a JSON-like mapping can
    be compiled to a workflow without this module importing those packages.
    """

    def as_recommendation(self) -> Mapping[str, Any]:
        ...


_RecommendationBuilder = Callable[[Mapping[str, Any]], Tuple[WorkflowStep, ...]]


class RecommendationCompiler:
    """Deterministically compiles recommendation mappings into definitions.

    Mapping from recommendation ``kind`` to workflow type and default step
    templates is configurable: register custom builders to extend the catalogue
    without modifying this class.
    """

    _KIND_TO_TYPE: Dict[str, WorkflowType] = {
        "predictive_maintenance": WorkflowType.PREDICTIVE_MAINTENANCE,
        "emergency_maintenance": WorkflowType.EMERGENCY_MAINTENANCE,
        "inspection": WorkflowType.INSPECTION,
        "shutdown_planning": WorkflowType.SHUTDOWN_PLANNING,
        "inventory_procurement": WorkflowType.INVENTORY_PROCUREMENT,
        "executive_approval": WorkflowType.EXECUTIVE_APPROVAL,
        "risk_mitigation": WorkflowType.RISK_MITIGATION,
        "safety_response": WorkflowType.SAFETY_RESPONSE,
        "knowledge_review": WorkflowType.KNOWLEDGE_REVIEW,
        "scenario_evaluation": WorkflowType.SCENARIO_EVALUATION,
    }

    def __init__(self) -> None:
        self._builders: Dict[str, _RecommendationBuilder] = {}
        self._register_default_builders()

    # -- builder registration ------------------------------------------------
    def register_builder(self, kind: str, builder: _RecommendationBuilder) -> None:
        self._builders[kind] = builder

    def supported_kinds(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders))

    # -- compilation ---------------------------------------------------------
    def compile(
        self, recommendation: Mapping[str, Any], *, workflow_id: Optional[str] = None
    ) -> WorkflowDefinition:
        if isinstance(recommendation, RecommendationSource):
            recommendation = recommendation.as_recommendation()
        if not isinstance(recommendation, Mapping):
            raise WorkflowValidationError("recommendation must be a mapping")
        kind = str(recommendation.get("kind", "predictive_maintenance"))
        workflow_type = self._KIND_TO_TYPE.get(kind, WorkflowType.PREDICTIVE_MAINTENANCE)
        builder = self._builders.get(kind, self._builders["predictive_maintenance"])
        steps = builder(recommendation)
        rec_id = str(recommendation.get("recommendation_id", "rec"))
        wid = workflow_id or f"wf-{kind}-{_short_hash(rec_id, kind)}"
        name = str(recommendation.get("title", f"{kind.replace('_', ' ').title()} Workflow"))
        metadata = {
            "recommendation_id": rec_id,
            "kind": kind,
            "source": recommendation.get("source", "executive_copilot"),
        }
        return WorkflowDefinition(
            workflow_id=wid,
            name=name,
            workflow_type=workflow_type,
            steps=steps,
            description=str(recommendation.get("summary", "")),
            metadata_json=_canonical_json(metadata),
        )

    # -- default templates ---------------------------------------------------
    def _register_default_builders(self) -> None:
        self._builders["predictive_maintenance"] = self._build_predictive
        self._builders["emergency_maintenance"] = self._build_emergency
        self._builders["inspection"] = self._build_inspection
        self._builders["shutdown_planning"] = self._build_shutdown
        self._builders["inventory_procurement"] = self._build_procurement
        self._builders["executive_approval"] = self._build_executive_approval
        self._builders["risk_mitigation"] = self._build_risk_mitigation
        self._builders["safety_response"] = self._build_safety
        self._builders["knowledge_review"] = self._build_knowledge_review
        self._builders["scenario_evaluation"] = self._build_scenario

    @staticmethod
    def _asset(recommendation: Mapping[str, Any]) -> Dict[str, Any]:
        return {"asset_id": recommendation.get("asset_id", "unknown")}

    def _build_predictive(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="reserve_parts",
                name="Reserve maintenance inventory",
                action=WorkflowAction.create(
                    ActionType.RESERVE_INVENTORY, "Reserve parts", params
                ),
            ),
            WorkflowStep(
                step_id="create_task",
                name="Create maintenance task",
                action=WorkflowAction.create(
                    ActionType.CREATE_MAINTENANCE_TASK, "Create task", params
                ),
            ),
            WorkflowStep(
                step_id="audit",
                name="Record audit entry",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_emergency(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="alert",
                name="Generate executive alert",
                action=WorkflowAction.create(
                    ActionType.GENERATE_EXECUTIVE_ALERT, "Alert", params
                ),
            ),
            WorkflowStep(
                step_id="approval",
                name="Escalate emergency approval",
                action=WorkflowAction.create(
                    ActionType.ESCALATE_APPROVAL, "Escalate", params
                ),
                requires_approval=True,
            ),
            WorkflowStep(
                step_id="create_task",
                name="Create emergency maintenance task",
                action=WorkflowAction.create(
                    ActionType.CREATE_MAINTENANCE_TASK, "Emergency task", params
                ),
                retry_policy=RetryPolicy(max_attempts=3, backoff_base=1.0, backoff_factor=2.0),
            ),
        )

    def _build_inspection(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="schedule",
                name="Schedule inspection",
                action=WorkflowAction.create(
                    ActionType.SCHEDULE_INSPECTION, "Schedule", params
                ),
            ),
            WorkflowStep(
                step_id="audit",
                name="Record audit entry",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_shutdown(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="approval",
                name="Executive shutdown approval",
                action=WorkflowAction.create(
                    ActionType.ESCALATE_APPROVAL, "Approve shutdown", params
                ),
                requires_approval=True,
            ),
            WorkflowStep(
                step_id="schedule",
                name="Schedule shutdown inspection",
                action=WorkflowAction.create(
                    ActionType.SCHEDULE_INSPECTION, "Pre-shutdown inspection", params
                ),
            ),
            WorkflowStep(
                step_id="audit",
                name="Record audit entry",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_procurement(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="reserve",
                name="Reserve inventory",
                action=WorkflowAction.create(
                    ActionType.RESERVE_INVENTORY, "Reserve", params
                ),
            ),
            WorkflowStep(
                step_id="approval",
                name="Budget approval",
                action=WorkflowAction.create(
                    ActionType.ESCALATE_APPROVAL, "Budget approval", params
                ),
                conditions=(
                    WorkflowCondition("budget", ConditionOperator.GT, 10000.0),
                ),
            ),
        )

    def _build_executive_approval(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="approval",
                name="Executive approval checkpoint",
                action=WorkflowAction.create(
                    ActionType.ESCALATE_APPROVAL, "Executive approval", params
                ),
                requires_approval=True,
            ),
            WorkflowStep(
                step_id="audit",
                name="Record approval audit",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_risk_mitigation(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="alert",
                name="Generate risk alert",
                action=WorkflowAction.create(
                    ActionType.GENERATE_EXECUTIVE_ALERT, "Risk alert", params
                ),
            ),
            WorkflowStep(
                step_id="task",
                name="Create mitigation task",
                action=WorkflowAction.create(
                    ActionType.CREATE_MAINTENANCE_TASK, "Mitigation task", params
                ),
            ),
        )

    def _build_safety(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="alert",
                name="Generate safety alert",
                action=WorkflowAction.create(
                    ActionType.GENERATE_EXECUTIVE_ALERT, "Safety alert", params
                ),
            ),
            WorkflowStep(
                step_id="inspection",
                name="Schedule safety inspection",
                action=WorkflowAction.create(
                    ActionType.SCHEDULE_INSPECTION, "Safety inspection", params
                ),
            ),
            WorkflowStep(
                step_id="audit",
                name="Record safety audit",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_knowledge_review(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="review",
                name="Create knowledge review",
                action=WorkflowAction.create(
                    ActionType.CREATE_KNOWLEDGE_REVIEW, "Knowledge review", params
                ),
            ),
            WorkflowStep(
                step_id="audit",
                name="Record review audit",
                action=WorkflowAction.create(
                    ActionType.GENERATE_AUDIT_RECORD, "Audit", params
                ),
            ),
        )

    def _build_scenario(self, rec: Mapping[str, Any]) -> Tuple[WorkflowStep, ...]:
        params = self._asset(rec)
        return (
            WorkflowStep(
                step_id="review",
                name="Create scenario knowledge review",
                action=WorkflowAction.create(
                    ActionType.CREATE_KNOWLEDGE_REVIEW, "Scenario review", params
                ),
            ),
            WorkflowStep(
                step_id="alert",
                name="Generate scenario alert",
                action=WorkflowAction.create(
                    ActionType.GENERATE_EXECUTIVE_ALERT, "Scenario alert", params
                ),
                conditions=(
                    WorkflowCondition("scenario_score", ConditionOperator.GTE, 0.7),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Workflow engine (orchestrator)
# ---------------------------------------------------------------------------
class WorkflowEngine:
    """Stateless orchestrator producing immutable execution snapshots.

    Every public operation accepts an immutable :class:`WorkflowExecution` and
    returns a new one, never mutating its argument. Collaborators are injected
    so the engine can be exercised in isolation.
    """

    def __init__(
        self,
        registry: Optional[WorkflowRegistry] = None,
        rule_engine: Optional[RuleEngine] = None,
        action_engine: Optional[ActionEngine] = None,
        *,
        clock_factory: Callable[[], Clock] = LogicalClock,
        state_machine: Optional[WorkflowStateMachine] = None,
        compiler: Optional[RecommendationCompiler] = None,
        default_actor: str = "system",
    ) -> None:
        self.registry = registry or WorkflowRegistry()
        self.rule_engine = rule_engine or RuleEngine()
        self.action_engine = action_engine or create_default_action_engine()
        self.state_machine = state_machine or WorkflowStateMachine()
        self.compiler = compiler or RecommendationCompiler()
        self._clock_factory = clock_factory
        self.default_actor = default_actor

    # -- construction helpers ------------------------------------------------
    def compile_recommendation(
        self, recommendation: Mapping[str, Any], *, register: bool = False
    ) -> WorkflowDefinition:
        definition = self.compiler.compile(recommendation)
        if register and definition.workflow_id not in self.registry:
            self.registry.register_workflow(definition)
        return definition

    def create_execution(
        self,
        definition: WorkflowDefinition,
        *,
        context: Optional[Mapping[str, Any]] = None,
        execution_id: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> WorkflowExecution:
        if not isinstance(definition, WorkflowDefinition):
            raise WorkflowValidationError("definition must be a WorkflowDefinition")
        clk = clock or self._clock_factory()
        ctx = dict(context or {})
        ctx.setdefault("workflow_id", definition.workflow_id)
        exec_id = execution_id or f"exec-{definition.fingerprint()}-{_short_hash(ctx)}"
        now = clk.now()
        execution = WorkflowExecution(
            execution_id=exec_id,
            workflow_id=definition.workflow_id,
            workflow_type=definition.workflow_type,
            state=WorkflowState.DRAFT,
            context_json=_canonical_json(ctx),
            created_at=now,
            updated_at=now,
        )
        event = self._event(
            execution, clk, step="", event="CREATED", actor=self.default_actor,
            result="draft", detail={"definition": definition.workflow_id},
        )
        return replace(execution, history=execution.history.append(event))

    # -- lifecycle operations ------------------------------------------------
    def submit(
        self, execution: WorkflowExecution, *, clock: Optional[Clock] = None
    ) -> WorkflowExecution:
        clk = self._resume_clock(execution, clock)
        return self._transition(
            execution, WorkflowState.PENDING, clk, event="SUBMITTED", result="pending"
        )

    def run(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        *,
        clock: Optional[Clock] = None,
        actor: Optional[str] = None,
    ) -> WorkflowExecution:
        """Drive the workflow forward to a terminal or paused state."""

        self._assert_matches(execution, definition)
        clk = self._resume_clock(execution, clock)
        actor = actor or self.default_actor
        if execution.state is WorkflowState.DRAFT:
            execution = self.submit(execution, clock=clk)
        if execution.state is WorkflowState.PENDING:
            execution = self._transition(
                execution, WorkflowState.RUNNING, clk, event="STARTED", result="running"
            )
        elif execution.state is WorkflowState.PAUSED:
            execution = self._transition(
                execution, WorkflowState.RUNNING, clk, event="RESUMED", result="running"
            )
        elif execution.state is not WorkflowState.RUNNING:
            raise InvalidStateTransitionError(
                f"Cannot run an execution in state {execution.state.value}"
            )
        return self._drive(execution, definition, clk, actor)

    def approve(
        self,
        execution: WorkflowExecution,
        step_id: str,
        definition: WorkflowDefinition,
        *,
        actor: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> WorkflowExecution:
        if execution.state is not WorkflowState.PAUSED:
            raise InvalidStateTransitionError("Approvals require a paused execution")
        if execution.pending_approval_step != step_id:
            raise WorkflowExecutionError(
                f"Step {step_id!r} is not awaiting approval"
            )
        clk = self._resume_clock(execution, clock)
        actor = actor or self.default_actor
        event = self._event(
            execution, clk, step=step_id, event="APPROVAL_GRANTED", actor=actor,
            result="approved",
        )
        execution = replace(
            execution,
            history=execution.history.append(event),
            pending_approval_step=None,
            updated_at=clk.now(),
        )
        return self.run(execution, definition, clock=clk, actor=actor)

    def reject(
        self,
        execution: WorkflowExecution,
        step_id: str,
        *,
        actor: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> WorkflowExecution:
        if execution.state is not WorkflowState.PAUSED:
            raise InvalidStateTransitionError("Rejections require a paused execution")
        clk = self._resume_clock(execution, clock)
        actor = actor or self.default_actor
        event = self._event(
            execution, clk, step=step_id, event="APPROVAL_REJECTED", actor=actor,
            result="rejected",
        )
        execution = replace(execution, history=execution.history.append(event))
        result = WorkflowResult(
            step_id=step_id, status=StepStatus.REJECTED, error="Approval rejected",
            attempts=0, started_at=clk.now(), finished_at=clk.now(),
        )
        execution = replace(execution, results=execution.results + (result,))
        return self._transition(
            execution, WorkflowState.FAILED, clk, event="FAILED",
            result="rejected", error="Approval rejected", actor=actor,
        )

    def pause(
        self, execution: WorkflowExecution, *, clock: Optional[Clock] = None,
        actor: Optional[str] = None,
    ) -> WorkflowExecution:
        clk = self._resume_clock(execution, clock)
        return self._transition(
            execution, WorkflowState.PAUSED, clk, event="PAUSED", result="paused",
            actor=actor or self.default_actor,
        )

    def resume(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        *,
        clock: Optional[Clock] = None,
        actor: Optional[str] = None,
    ) -> WorkflowExecution:
        if execution.state is not WorkflowState.PAUSED:
            raise InvalidStateTransitionError("Only paused executions can resume")
        if execution.pending_approval_step is not None:
            raise WorkflowExecutionError(
                "Execution is awaiting approval; call approve() or reject()"
            )
        return self.run(execution, definition, clock=clock, actor=actor)

    def cancel(
        self, execution: WorkflowExecution, *, clock: Optional[Clock] = None,
        actor: Optional[str] = None,
    ) -> WorkflowExecution:
        clk = self._resume_clock(execution, clock)
        return self._transition(
            execution, WorkflowState.CANCELLED, clk, event="CANCELLED",
            result="cancelled", actor=actor or self.default_actor,
        )

    # -- statistics ----------------------------------------------------------
    @staticmethod
    def statistics(executions: Sequence[WorkflowExecution]) -> WorkflowStatistics:
        return WorkflowStatistics.from_executions(executions)

    # -- internal: driving ---------------------------------------------------
    def _drive(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        clk: Clock,
        actor: str,
    ) -> WorkflowExecution:
        steps = definition.steps
        index = {s.step_id: i for i, s in enumerate(steps)}
        completed_ids = {r.step_id for r in execution.results if r.status is not StepStatus.SKIPPED}
        granted = {e.step for e in execution.history.filter_by_event("APPROVAL_GRANTED")}

        # Determine the starting cursor: resume after the last recorded step.
        cursor = 0
        if execution.results:
            last = execution.results[-1]
            if last.step_id in index:
                cursor = index[last.step_id] + 1
        if execution.current_step in index and execution.pending_approval_step is None:
            # A previously paused-for-approval step must be re-attempted.
            if execution.current_step not in completed_ids:
                cursor = index[execution.current_step]

        results = list(execution.results)
        history = execution.history
        context = execution.context

        while cursor < len(steps):
            step = steps[cursor]

            # Parallel group: gather the maximal contiguous run sharing a group.
            group = step.parallel_group
            if group is not None:
                end = cursor
                while end < len(steps) and steps[end].parallel_group == group:
                    end += 1
                batch = steps[cursor:end]
                (
                    execution, results, history, context, group_ok, paused,
                ) = self._run_parallel_group(
                    execution, definition, batch, clk, actor, results, history,
                    context, granted, completed_ids,
                )
                if paused:
                    return self._pause_for_approval(
                        execution, results, history, context, clk, actor
                    )
                if not group_ok:
                    return self._fail(
                        execution, results, history, context, clk, actor,
                        step=group, error=f"Parallel group {group!r} failed",
                    )
                cursor = end
                continue

            # Approval checkpoint.
            if step.requires_approval and step.step_id not in granted:
                history = history.append(
                    self._event(
                        execution, clk, step=step.step_id, event="APPROVAL_REQUIRED",
                        actor=actor, result="paused", history=history,
                    )
                )
                execution = replace(
                    execution, results=tuple(results), history=history,
                    current_step=step.step_id, pending_approval_step=step.step_id,
                    context_json=_canonical_json(context),
                )
                return self._transition(
                    execution, WorkflowState.PAUSED, clk, event="PAUSED",
                    result="awaiting_approval", actor=actor,
                )

            # Guard evaluation.
            if not step.guard_passes(context):
                result, history = self._record_skip(execution, step, clk, actor, history)
                results.append(result)
                cursor = self._advance_cursor(step, index, cursor, success=True)
                continue

            # Execute (with retry / timeout).
            result, history = self._execute_step(
                execution, step, clk, actor, history, context
            )
            results.append(result)
            context = self._merge_output(context, step, result)

            if result.status is StepStatus.COMPLETED:
                cursor = self._advance_cursor(step, index, cursor, success=True)
            else:
                if step.optional:
                    cursor = self._advance_cursor(step, index, cursor, success=True)
                elif step.on_failure:
                    cursor = index[step.on_failure[0]]
                else:
                    return self._fail(
                        execution, results, history, context, clk, actor,
                        step=step.step_id, error=result.error or "step failed",
                    )

        # All steps processed -> completed.
        execution = replace(
            execution, results=tuple(results), history=history,
            context_json=_canonical_json(context), current_step=None,
        )
        return self._transition(
            execution, WorkflowState.COMPLETED, clk, event="COMPLETED",
            result="completed", actor=actor,
        )

    def _run_parallel_group(
        self, execution, definition, batch, clk, actor, results, history, context,
        granted, completed_ids,
    ):
        """Execute a logical parallel group deterministically (ordered)."""

        policy = definition.parallel_policy
        outcomes: List[bool] = []
        max_duration = 0.0
        start = clk.now()
        for step in batch:
            if step.requires_approval and step.step_id not in granted:
                history = history.append(
                    self._event(
                        execution, clk, step=step.step_id, event="APPROVAL_REQUIRED",
                        actor=actor, result="paused", history=history,
                    )
                )
                execution = replace(
                    execution, history=history, current_step=step.step_id,
                    pending_approval_step=step.step_id,
                )
                return execution, results, history, context, False, True
            if not step.guard_passes(context):
                result, history = self._record_skip(execution, step, clk, actor, history)
                results.append(result)
                continue
            result, history = self._execute_step(
                execution, step, clk, actor, history, context, advance_clock=False
            )
            results.append(result)
            context = self._merge_output(context, step, result)
            outcomes.append(result.status is StepStatus.COMPLETED)
            max_duration = max(max_duration, result.duration)
        # Model concurrency: the group costs the longest child, not the sum.
        clk.advance(max_duration)
        if not outcomes:
            group_ok = True
        elif policy == "ANY":
            group_ok = any(outcomes)
        else:
            group_ok = all(outcomes)
        return execution, results, history, context, group_ok, False

    def _execute_step(
        self, execution, step, clk, actor, history, context, *, advance_clock=True,
    ) -> Tuple[WorkflowResult, WorkflowHistory]:
        history = history.append(
            self._event(
                execution, clk, step=step.step_id, event="STEP_STARTED",
                actor=actor, result="running", history=history,
            )
        )
        attempts = 0
        last_error = ""
        last_output: Mapping[str, Any] = {}
        started = clk.now()
        elapsed = 0.0
        status = StepStatus.FAILED
        for attempt in range(1, step.retry_policy.max_attempts + 1):
            delay = step.retry_policy.delay_for_attempt(attempt)
            if delay > 0:
                if advance_clock:
                    clk.advance(delay)
                else:
                    elapsed += delay
                history = history.append(
                    self._event(
                        execution, clk, step=step.step_id, event="RETRY_BACKOFF",
                        actor=actor, result="retry",
                        detail={"attempt": attempt, "delay": delay}, history=history,
                    )
                )
            attempts = attempt
            try:
                outcome = self.action_engine.dispatch(step.action, context, clk)
            except ActionDispatchError as exc:
                last_error = str(exc)
                status = StepStatus.FAILED
                break
            if outcome.duration > 0:
                if advance_clock:
                    clk.advance(outcome.duration)
                else:
                    elapsed += outcome.duration
            if step.timeout is not None and outcome.duration > step.timeout:
                last_error = (
                    f"step exceeded timeout ({outcome.duration} > {step.timeout})"
                )
                last_output = outcome.output
                status = StepStatus.TIMEOUT
                history = history.append(
                    self._event(
                        execution, clk, step=step.step_id, event="STEP_TIMEOUT",
                        actor=actor, result="timeout",
                        detail={"attempt": attempt, "duration": outcome.duration},
                        history=history,
                    )
                )
                continue
            if outcome.success:
                status = StepStatus.COMPLETED
                last_output = outcome.output
                last_error = ""
                break
            last_error = outcome.error
            last_output = outcome.output
            status = StepStatus.FAILED
            history = history.append(
                self._event(
                    execution, clk, step=step.step_id, event="STEP_ATTEMPT_FAILED",
                    actor=actor, result="failed",
                    detail={"attempt": attempt, "error": outcome.error},
                    history=history,
                )
            )
        finished = clk.now() if advance_clock else started + elapsed
        result = WorkflowResult(
            step_id=step.step_id, status=status,
            output_json=_canonical_json(dict(last_output)), error=last_error,
            attempts=attempts, started_at=started, finished_at=finished,
        )
        history = history.append(
            self._event(
                execution, clk, step=step.step_id,
                event="STEP_COMPLETED" if status is StepStatus.COMPLETED else "STEP_FAILED",
                actor=actor, result=status.value,
                detail={"attempts": attempts}, history=history,
            )
        )
        return result, history

    def _record_skip(self, execution, step, clk, actor, history):
        now = clk.now()
        result = WorkflowResult(
            step_id=step.step_id, status=StepStatus.SKIPPED,
            error="guard condition not satisfied", attempts=0,
            started_at=now, finished_at=now,
        )
        history = history.append(
            self._event(
                execution, clk, step=step.step_id, event="STEP_SKIPPED",
                actor=actor, result="skipped", history=history,
            )
        )
        return result, history

    @staticmethod
    def _advance_cursor(step, index, cursor, *, success):
        if success and step.next_steps:
            return index[step.next_steps[0]]
        return cursor + 1

    @staticmethod
    def _merge_output(context, step, result):
        if result.status is not StepStatus.COMPLETED:
            return context
        merged = dict(context)
        merged[f"{step.step_id}_output"] = result.output
        return merged

    # -- internal: state transitions & terminal helpers ----------------------
    def _pause_for_approval(self, execution, results, history, context, clk, actor):
        execution = replace(
            execution, results=tuple(results), history=history,
            context_json=_canonical_json(context),
        )
        return self._transition(
            execution, WorkflowState.PAUSED, clk, event="PAUSED",
            result="awaiting_approval", actor=actor,
        )

    def _fail(self, execution, results, history, context, clk, actor, *, step, error):
        execution = replace(
            execution, results=tuple(results), history=history,
            context_json=_canonical_json(context), current_step=step,
        )
        return self._transition(
            execution, WorkflowState.FAILED, clk, event="FAILED",
            result="failed", error=error, actor=actor,
        )

    def _transition(
        self, execution, new_state, clk, *, event, result, error="", actor=None,
    ) -> WorkflowExecution:
        actor = actor or self.default_actor
        self.state_machine.validate(execution.state, new_state)
        history = execution.history.append(
            self._event(
                execution, clk, step=execution.current_step or "", event=event,
                actor=actor, result=result,
                detail={"from": execution.state.value, "to": WorkflowState.coerce(new_state).value},
            )
        )
        pending = execution.pending_approval_step
        current = execution.current_step
        if new_state is WorkflowState.COMPLETED:
            current = None
        if new_state is not WorkflowState.PAUSED:
            if event not in ("PAUSED",):
                pending = pending  # retained only while paused-for-approval
        return replace(
            execution, state=WorkflowState.coerce(new_state), history=history,
            error=error or execution.error, updated_at=clk.now(),
            current_step=current, pending_approval_step=pending,
        )

    def _event(
        self, execution, clk, *, step, event, actor, result, detail=None, history=None,
    ) -> AuditEvent:
        seq_source = history if history is not None else execution.history
        return AuditEvent(
            sequence=seq_source.next_sequence,
            timestamp=clk.now(),
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id,
            step=step or "",
            event=event,
            actor=actor,
            result=result,
            detail_json=_canonical_json(detail or {}),
        )

    def _resume_clock(self, execution: WorkflowExecution, clock: Optional[Clock]) -> Clock:
        if clock is not None:
            return clock
        clk = self._clock_factory()
        # Re-seat a fresh logical clock at the execution's last known time so
        # multi-call lifecycles keep timestamps monotonic and deterministic.
        if isinstance(clk, (LogicalClock,)):
            clk.advance(max(0.0, execution.updated_at - clk.now()))
        return clk

    @staticmethod
    def _assert_matches(execution: WorkflowExecution, definition: WorkflowDefinition) -> None:
        if execution.workflow_id != definition.workflow_id:
            raise WorkflowExecutionError(
                "Execution / definition workflow_id mismatch: "
                f"{execution.workflow_id!r} != {definition.workflow_id!r}"
            )


def create_default_engine() -> WorkflowEngine:
    """Factory wiring an engine with default deterministic collaborators."""

    return WorkflowEngine(
        registry=WorkflowRegistry(),
        rule_engine=RuleEngine(),
        action_engine=create_default_action_engine(),
        clock_factory=LogicalClock,
    )


# ---------------------------------------------------------------------------
# Command line demonstration
# ---------------------------------------------------------------------------
def _demo() -> int:
    """Deterministic end-to-end demonstration used by ``--demo``."""

    print("=" * 70)
    print("Enterprise Workflow Engine - Week 10 Phase 1 Demonstration")
    print("=" * 70)

    engine = create_default_engine()

    recommendation = {
        "kind": "emergency_maintenance",
        "recommendation_id": "REC-2031",
        "asset_id": "TURBINE-07",
        "title": "Emergency turbine intervention",
        "summary": "Bearing temperature exceeded critical threshold.",
        "source": "executive_rag_copilot",
        "risk_score": 0.92,
    }

    definition = engine.compile_recommendation(recommendation, register=True)
    print(f"\nCompiled workflow : {definition.workflow_id}")
    print(f"Workflow type     : {definition.workflow_type.value}")
    print(f"Steps             : {', '.join(definition.step_ids)}")

    clock = LogicalClock()
    context = {
        "asset_id": "TURBINE-07",
        "risk_score": 0.92,
        "asset_criticality": 0.95,
        "executive_confidence": 0.88,
    }
    execution = engine.create_execution(definition, context=context, clock=clock)
    execution = engine.run(execution, definition, clock=clock)
    print(f"\nState after run   : {execution.state.value}")
    print(f"Awaiting approval : {execution.pending_approval_step}")

    # The emergency workflow pauses for an approval checkpoint.
    execution = engine.approve(
        execution, execution.pending_approval_step, definition,
        actor="cfo", clock=clock,
    )
    print(f"State after approve: {execution.state.value}")

    print("\nStep results:")
    for result in execution.results:
        print(
            f"  - {result.step_id:<14} {result.status.value:<10} "
            f"attempts={result.attempts} duration={result.duration:.1f}"
        )

    print("\nAudit timeline (event @ t):")
    for event in execution.history:
        print(
            f"  [{event.sequence:02d}] t={event.timestamp:6.1f} "
            f"{event.event:<18} step={event.step or '-':<14} actor={event.actor}"
        )

    stats = engine.statistics([execution])
    print("\nAggregate statistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key:<18}: {value}")

    # Demonstrate determinism: a second identical run yields identical results.
    clock2 = LogicalClock()
    exec2 = engine.create_execution(definition, context=context, clock=clock2)
    exec2 = engine.run(exec2, definition, clock=clock2)
    exec2 = engine.approve(exec2, exec2.pending_approval_step, definition,
                           actor="cfo", clock=clock2)
    identical = exec2.to_dict() == execution.to_dict()
    print(f"\nDeterministic replay identical: {identical}")

    print("\nRule engine demonstration:")
    rule_engine = RuleEngine(
        [
            Rule("r-risk", RuleType.RISK_THRESHOLD,
                 WorkflowCondition("risk_score", ConditionOperator.GTE, 0.8),
                 "Risk score is critical"),
            Rule("r-conf", RuleType.EXECUTIVE_CONFIDENCE,
                 WorkflowCondition("executive_confidence", ConditionOperator.GTE, 0.9),
                 "Executive confidence is high"),
        ]
    )
    for rule in rule_engine.matching_rules(context):
        print(f"  matched: {rule.rule_id} ({rule.rule_type.value})")

    print("\nDemonstration complete.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Workflow Engine")
    parser.add_argument("--demo", action="store_true", help="Run the demonstration")
    args = parser.parse_args(argv)
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())