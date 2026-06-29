"""Enterprise Business Process Orchestrator (Week 10, Phase 2).

This module sits *above* the Week 10 Phase 1 Workflow Engine. Where the
Workflow Engine executes individual workflows, the Business Process
Orchestrator coordinates complete enterprise business processes composed of
many workflows, governed by a dependency graph, approval chains, SLAs, a
business calendar, and rollback / compensation strategies.

The orchestrator owns coordination concerns only:

* Business process registration and versioning.
* Dependency-graph management (DAG, cycle detection, topological sort,
  critical path, blocking nodes, connectivity) - implemented from scratch with
  no graph libraries.
* Deterministic execution planning (sequential, parallel, waiting, approval,
  rollback and compensation stages).
* Enterprise approval chains (Engineer -> Lead -> Manager -> Director ->
  Vice President -> Executive) with delegation, escalation and timeout.
* SLA tracking, a timezone-aware business calendar, rollback / compensation
  planning, dry-run / what-if simulation, analytics and enterprise KPIs.

Integration is by **composition only**. The Workflow Engine is consumed
through its public API; it is never modified, monkey-patched, or duplicated.

Design tenets: pure Python + NumPy, frozen+slotted dataclasses, dependency
injection, protocols, custom exceptions, thread-safe registry, full JSON
serialisation, and 100% deterministic planning and execution.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import threading
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

# --- Phase 1 Workflow Engine (composition only, never modified) ------------
try:  # package-relative import (repository layout)
    from ..workflow import workflow_engine as wfe
except (ImportError, ValueError):  # pragma: no cover - import resolution
    try:
        from src.workflow import workflow_engine as wfe
    except ImportError:
        import workflow_engine as wfe  # isolated execution (path-injected)


__all__ = [
    # Exceptions
    "OrchestratorError",
    "ProcessValidationError",
    "DependencyError",
    "CycleError",
    "PlanningError",
    "ApprovalError",
    "SLAError",
    "RollbackError",
    "SimulationError",
    "ProcessRegistryError",
    # Enums
    "ProcessState",
    "StageType",
    "ApprovalRole",
    "ApprovalState",
    "SLAStatus",
    "SimulationMode",
    "RollbackStrategy",
    "DependencyType",
    # Domain model
    "ProcessStep",
    "Dependency",
    "ApprovalStep",
    "ApprovalChain",
    "SLAConfig",
    "BusinessCalendar",
    "BusinessProcess",
    "ExecutionGroup",
    "ExecutionPlan",
    "StepExecutionRecord",
    "ApprovalRecord",
    "ApprovalChainState",
    "SLAReport",
    "ProcessEvent",
    "ProcessHistory",
    "ProcessExecutionResult",
    "ProcessAnalytics",
    "BusinessKPIs",
    "RollbackPlan",
    "CompensationPlan",
    "SimulationResult",
    # Engines / collaborators
    "DependencyGraph",
    "ProcessRegistry",
    "ExecutionPlanner",
    "ApprovalEngine",
    "SLAEngine",
    "RollbackEngine",
    "WorkflowProvider",
    "DecisionProvider",
    "BusinessProcessOrchestrator",
    # Factories
    "create_default_orchestrator",
    "default_approval_chain",
]

logger = logging.getLogger("orchestration.bpo")
if not logger.handlers:  # pragma: no cover - environmental
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class OrchestratorError(Exception):
    """Base class for all orchestrator errors."""


class ProcessValidationError(OrchestratorError):
    """Raised when a business process fails validation."""


class DependencyError(OrchestratorError):
    """Raised for invalid dependency references."""


class CycleError(DependencyError):
    """Raised when the dependency graph contains a cycle."""


class PlanningError(OrchestratorError):
    """Raised when an execution plan cannot be produced."""


class ApprovalError(OrchestratorError):
    """Raised for invalid approval-chain operations."""


class SLAError(OrchestratorError):
    """Raised for invalid SLA configuration."""


class RollbackError(OrchestratorError):
    """Raised when a rollback / compensation plan cannot be produced."""


class SimulationError(OrchestratorError):
    """Raised for invalid simulation requests."""


class ProcessRegistryError(OrchestratorError):
    """Raised for registry lookup / duplication / freeze violations."""


# ---------------------------------------------------------------------------
# Serialisation helpers (local; no dependency on Phase 1 internals)
# ---------------------------------------------------------------------------
def _canonical_json(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise ProcessValidationError(f"Expected mapping, got {type(payload)!r}")
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
            raise ProcessValidationError(f"{value!r} is not a valid {cls.__name__}") from exc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class ProcessState(_ValueEnum):
    DRAFT = "draft"
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    COMPENSATED = "compensated"
    SIMULATED = "simulated"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_PROCESS_STATES


class StageType(_ValueEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    WAITING = "waiting"
    APPROVAL = "approval"
    ROLLBACK = "rollback"
    COMPENSATION = "compensation"


class ApprovalRole(_ValueEnum):
    ENGINEER = "engineer"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    VICE_PRESIDENT = "vice_president"
    EXECUTIVE = "executive"

    @property
    def level(self) -> int:
        return _ROLE_ORDER.index(self)


class ApprovalState(_ValueEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELEGATED = "delegated"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


class SLAStatus(_ValueEnum):
    ON_TRACK = "on_track"
    WARNING = "warning"
    VIOLATED = "violated"
    RECOVERED = "recovered"
    NOT_APPLICABLE = "not_applicable"


class SimulationMode(_ValueEnum):
    DRY_RUN = "dry_run"
    SIMULATION = "simulation"
    REPLAY = "replay"
    RECOVERY = "recovery"
    LIVE = "live"
    WHAT_IF = "what_if"


class RollbackStrategy(_ValueEnum):
    SEQUENTIAL_UNDO = "sequential_undo"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    COMPENSATE = "compensate"
    NONE = "none"


class DependencyType(_ValueEnum):
    FINISH_TO_START = "finish_to_start"
    HARD = "hard"
    SOFT = "soft"


_ROLE_ORDER: List[ApprovalRole] = [
    ApprovalRole.ENGINEER,
    ApprovalRole.LEAD,
    ApprovalRole.MANAGER,
    ApprovalRole.DIRECTOR,
    ApprovalRole.VICE_PRESIDENT,
    ApprovalRole.EXECUTIVE,
]

_TERMINAL_PROCESS_STATES: FrozenSet[ProcessState] = frozenset({
    ProcessState.COMPLETED,
    ProcessState.FAILED,
    ProcessState.CANCELLED,
    ProcessState.ROLLED_BACK,
    ProcessState.COMPENSATED,
    ProcessState.SIMULATED,
})


# ---------------------------------------------------------------------------
# Domain model - process composition
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProcessStep:
    """A single step of a business process bound to one Phase 1 workflow."""

    step_id: str
    name: str
    workflow_ref: str
    estimated_duration: float = 0.0
    resource_cost: float = 0.0
    risk_weight: float = 0.0
    requires_approval: bool = False
    approval_chain_id: Optional[str] = None
    compensation_step_id: Optional[str] = None
    retryable: bool = True
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ProcessValidationError("step_id must be non-empty")
        if not self.name:
            raise ProcessValidationError("step name must be non-empty")
        if not self.workflow_ref:
            raise ProcessValidationError("workflow_ref must be non-empty")
        if self.estimated_duration < 0:
            raise ProcessValidationError("estimated_duration must be >= 0")
        if self.resource_cost < 0:
            raise ProcessValidationError("resource_cost must be >= 0")
        if not 0.0 <= self.risk_weight <= 1.0:
            raise ProcessValidationError("risk_weight must be in [0, 1]")
        if self.requires_approval and not self.approval_chain_id:
            raise ProcessValidationError(
                "steps requiring approval must reference an approval_chain_id"
            )
        object.__setattr__(self, "metadata_json", _canonical_json(_parse_json(self.metadata_json)))

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    @property
    def automated(self) -> bool:
        return not self.requires_approval

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "workflow_ref": self.workflow_ref,
            "estimated_duration": self.estimated_duration,
            "resource_cost": self.resource_cost,
            "risk_weight": self.risk_weight,
            "requires_approval": self.requires_approval,
            "approval_chain_id": self.approval_chain_id,
            "compensation_step_id": self.compensation_step_id,
            "retryable": self.retryable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessStep":
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            workflow_ref=data["workflow_ref"],
            estimated_duration=float(data.get("estimated_duration", 0.0)),
            resource_cost=float(data.get("resource_cost", 0.0)),
            risk_weight=float(data.get("risk_weight", 0.0)),
            requires_approval=bool(data.get("requires_approval", False)),
            approval_chain_id=data.get("approval_chain_id"),
            compensation_step_id=data.get("compensation_step_id"),
            retryable=bool(data.get("retryable", True)),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class Dependency:
    """``successor`` depends on ``predecessor`` (predecessor finishes first)."""

    predecessor: str
    successor: str
    dependency_type: DependencyType = DependencyType.FINISH_TO_START

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_type", DependencyType.coerce(self.dependency_type))
        if self.predecessor == self.successor:
            raise DependencyError("A step cannot depend on itself")
        if not self.predecessor or not self.successor:
            raise DependencyError("Dependency endpoints must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predecessor": self.predecessor,
            "successor": self.successor,
            "dependency_type": self.dependency_type.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dependency":
        return cls(
            predecessor=data["predecessor"],
            successor=data["successor"],
            dependency_type=DependencyType.coerce(data.get("dependency_type", "finish_to_start")),
        )


@dataclass(frozen=True, slots=True)
class ApprovalStep:
    """A single rung of an approval chain."""

    role: ApprovalRole
    approver_id: str
    sla_hours: float = 24.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ApprovalRole.coerce(self.role))
        if not self.approver_id:
            raise ApprovalError("approver_id must be non-empty")
        if self.sla_hours < 0:
            raise ApprovalError("sla_hours must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role.value, "approver_id": self.approver_id, "sla_hours": self.sla_hours}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApprovalStep":
        return cls(ApprovalRole.coerce(data["role"]), data["approver_id"], float(data.get("sla_hours", 24.0)))


@dataclass(frozen=True, slots=True)
class ApprovalChain:
    """An ordered enterprise approval chain."""

    chain_id: str
    steps: Tuple[ApprovalStep, ...]

    def __post_init__(self) -> None:
        if not self.chain_id:
            raise ApprovalError("chain_id must be non-empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ApprovalError("approval chain must have at least one step")
        levels = [s.role.level for s in self.steps]
        if levels != sorted(levels):
            raise ApprovalError("approval chain roles must be in ascending authority order")

    @property
    def modeled_latency_seconds(self) -> float:
        return float(sum(s.sla_hours for s in self.steps)) * 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {"chain_id": self.chain_id, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApprovalChain":
        return cls(data["chain_id"], tuple(ApprovalStep.from_dict(s) for s in data["steps"]))


@dataclass(frozen=True, slots=True)
class SLAConfig:
    """Service-level configuration for a business process."""

    expected_duration: float
    warning_threshold: float = 0.8
    escalation_threshold: float = 1.0
    penalty_per_second: float = 0.0
    max_recovery_time: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_duration <= 0:
            raise SLAError("expected_duration must be > 0")
        if not 0.0 < self.warning_threshold <= 1.0:
            raise SLAError("warning_threshold must be in (0, 1]")
        if self.escalation_threshold < self.warning_threshold:
            raise SLAError("escalation_threshold must be >= warning_threshold")
        if self.penalty_per_second < 0:
            raise SLAError("penalty_per_second must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_duration": self.expected_duration,
            "warning_threshold": self.warning_threshold,
            "escalation_threshold": self.escalation_threshold,
            "penalty_per_second": self.penalty_per_second,
            "max_recovery_time": self.max_recovery_time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SLAConfig":
        return cls(
            expected_duration=float(data["expected_duration"]),
            warning_threshold=float(data.get("warning_threshold", 0.8)),
            escalation_threshold=float(data.get("escalation_threshold", 1.0)),
            penalty_per_second=float(data.get("penalty_per_second", 0.0)),
            max_recovery_time=float(data.get("max_recovery_time", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class BusinessCalendar:
    """A timezone-aware business calendar (no DST; fully deterministic).

    Timestamps are POSIX epoch seconds (UTC). ``timezone_offset_minutes`` is
    added to UTC to obtain local civil time for weekday / hour / holiday tests.
    """

    timezone_offset_minutes: int = 0
    business_start_hour: int = 9
    business_end_hour: int = 17
    working_days: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon..Fri
    holidays: Tuple[str, ...] = ()                    # 'YYYY-MM-DD' local
    maintenance_windows: Tuple[Tuple[float, float], ...] = ()
    blackout_periods: Tuple[Tuple[float, float], ...] = ()
    emergency_mode: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.business_start_hour < self.business_end_hour <= 24:
            raise OrchestratorError("invalid business hours")
        object.__setattr__(self, "working_days", tuple(sorted(set(int(d) for d in self.working_days))))
        object.__setattr__(self, "holidays", tuple(self.holidays))
        object.__setattr__(self, "maintenance_windows", tuple(tuple(w) for w in self.maintenance_windows))
        object.__setattr__(self, "blackout_periods", tuple(tuple(b) for b in self.blackout_periods))
        for day in self.working_days:
            if not 0 <= day <= 6:
                raise OrchestratorError("working_days entries must be 0..6 (Mon..Sun)")

    @classmethod
    def always_on(cls) -> "BusinessCalendar":
        """A 24/7 calendar that never blocks (used as the permissive default)."""

        return cls(business_start_hour=0, business_end_hour=24,
                   working_days=(0, 1, 2, 3, 4, 5, 6), emergency_mode=True)

    def _local(self, ts: float) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc) + _dt.timedelta(
            minutes=self.timezone_offset_minutes
        )

    def in_blackout(self, ts: float) -> bool:
        return any(start <= ts < end for start, end in self.blackout_periods)

    def in_maintenance_window(self, ts: float) -> bool:
        return any(start <= ts < end for start, end in self.maintenance_windows)

    def is_business_time(self, ts: float) -> bool:
        if self.emergency_mode:
            return not self.in_blackout(ts)
        if self.in_blackout(ts):
            return False
        local = self._local(ts)
        if local.weekday() not in self.working_days:
            return False
        if local.strftime("%Y-%m-%d") in self.holidays:
            return False
        return self.business_start_hour <= local.hour < self.business_end_hour

    def next_business_time(self, ts: float, *, step_seconds: float = 900.0,
                           max_steps: int = 20000) -> float:
        if self.is_business_time(ts):
            return ts
        cursor = ts
        for _ in range(max_steps):
            cursor += step_seconds
            if self.is_business_time(cursor):
                return cursor
        raise OrchestratorError("could not find a business time within the search horizon")

    def advance_business_seconds(self, start: float, seconds: float, *,
                                 step_seconds: float = 900.0, max_steps: int = 200000) -> float:
        """Return the timestamp reached after ``seconds`` of *business* time."""

        if seconds <= 0:
            return start
        if self.emergency_mode and not self.blackout_periods:
            return start + seconds
        cursor = self.next_business_time(start)
        remaining = seconds
        for _ in range(max_steps):
            if self.is_business_time(cursor):
                chunk = min(step_seconds, remaining)
                remaining -= chunk
                cursor += chunk
                if remaining <= 1e-9:
                    return cursor
            else:
                cursor = self.next_business_time(cursor)
        raise OrchestratorError("advance_business_seconds exceeded its horizon")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timezone_offset_minutes": self.timezone_offset_minutes,
            "business_start_hour": self.business_start_hour,
            "business_end_hour": self.business_end_hour,
            "working_days": list(self.working_days),
            "holidays": list(self.holidays),
            "maintenance_windows": [list(w) for w in self.maintenance_windows],
            "blackout_periods": [list(b) for b in self.blackout_periods],
            "emergency_mode": self.emergency_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BusinessCalendar":
        return cls(
            timezone_offset_minutes=int(data.get("timezone_offset_minutes", 0)),
            business_start_hour=int(data.get("business_start_hour", 9)),
            business_end_hour=int(data.get("business_end_hour", 17)),
            working_days=tuple(data.get("working_days", (0, 1, 2, 3, 4))),
            holidays=tuple(data.get("holidays", ())),
            maintenance_windows=tuple(tuple(w) for w in data.get("maintenance_windows", ())),
            blackout_periods=tuple(tuple(b) for b in data.get("blackout_periods", ())),
            emergency_mode=bool(data.get("emergency_mode", False)),
        )


@dataclass(frozen=True, slots=True)
class BusinessProcess:
    """An immutable, validated enterprise business process definition."""

    process_id: str
    name: str
    steps: Tuple[ProcessStep, ...]
    dependencies: Tuple[Dependency, ...] = ()
    approval_chains: Tuple[ApprovalChain, ...] = ()
    sla: Optional[SLAConfig] = None
    calendar: Optional[BusinessCalendar] = None
    version: int = 1
    frozen: bool = False
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.process_id:
            raise ProcessValidationError("process_id must be non-empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "approval_chains", tuple(self.approval_chains))
        object.__setattr__(self, "metadata_json", _canonical_json(_parse_json(self.metadata_json)))
        self.validate()

    def validate(self) -> None:
        if not self.steps:
            raise ProcessValidationError("a business process must contain at least one step")
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ProcessValidationError("duplicate step_id detected")
        known = set(ids)
        for dep in self.dependencies:
            if dep.predecessor not in known or dep.successor not in known:
                raise DependencyError(f"dependency references unknown step: {dep.to_dict()}")
        chain_ids = {c.chain_id for c in self.approval_chains}
        if len(chain_ids) != len(self.approval_chains):
            raise ProcessValidationError("duplicate approval chain id")
        for step in self.steps:
            if step.requires_approval and step.approval_chain_id not in chain_ids:
                raise ProcessValidationError(
                    f"step {step.step_id} references unknown approval chain {step.approval_chain_id!r}"
                )
            if step.compensation_step_id and step.compensation_step_id not in known:
                raise ProcessValidationError(
                    f"step {step.step_id} references unknown compensation step"
                )
        # Acyclicity is part of structural validity.
        DependencyGraph.from_process(self).assert_acyclic()

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    @property
    def step_ids(self) -> Tuple[str, ...]:
        return tuple(s.step_id for s in self.steps)

    def get_step(self, step_id: str) -> ProcessStep:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise ProcessValidationError(f"unknown step_id: {step_id}")

    def get_chain(self, chain_id: str) -> ApprovalChain:
        for c in self.approval_chains:
            if c.chain_id == chain_id:
                return c
        raise ApprovalError(f"unknown approval chain: {chain_id}")

    def fingerprint(self) -> str:
        return _short_hash(json.dumps(self.to_dict(), sort_keys=True), length=16)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "name": self.name,
            "version": self.version,
            "frozen": self.frozen,
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "approval_chains": [c.to_dict() for c in self.approval_chains],
            "sla": self.sla.to_dict() if self.sla else None,
            "calendar": self.calendar.to_dict() if self.calendar else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BusinessProcess":
        return cls(
            process_id=data["process_id"],
            name=data["name"],
            steps=tuple(ProcessStep.from_dict(s) for s in data["steps"]),
            dependencies=tuple(Dependency.from_dict(d) for d in data.get("dependencies", ())),
            approval_chains=tuple(ApprovalChain.from_dict(c) for c in data.get("approval_chains", ())),
            sla=SLAConfig.from_dict(data["sla"]) if data.get("sla") else None,
            calendar=BusinessCalendar.from_dict(data["calendar"]) if data.get("calendar") else None,
            version=int(data.get("version", 1)),
            frozen=bool(data.get("frozen", False)),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Dependency graph (implemented from scratch; no graph libraries)
# ---------------------------------------------------------------------------
class DependencyGraph:
    """A directed acyclic graph with deterministic algorithms.

    Node iteration order follows the original step declaration order, so every
    derived ordering (topological sort, layers, critical path) is reproducible.
    """

    def __init__(self, nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> None:
        self._nodes: Tuple[str, ...] = tuple(nodes)
        self._index: Dict[str, int] = {n: i for i, n in enumerate(self._nodes)}
        if len(self._index) != len(self._nodes):
            raise DependencyError("duplicate node id in dependency graph")
        self._succ: Dict[str, List[str]] = {n: [] for n in self._nodes}
        self._pred: Dict[str, List[str]] = {n: [] for n in self._nodes}
        for pre, suc in edges:
            if pre not in self._index or suc not in self._index:
                raise DependencyError(f"edge references unknown node: {pre} -> {suc}")
            self._succ[pre].append(suc)
            self._pred[suc].append(pre)
        for n in self._nodes:  # deterministic adjacency
            self._succ[n].sort(key=lambda x: self._index[x])
            self._pred[n].sort(key=lambda x: self._index[x])

    # -- construction --------------------------------------------------------
    @classmethod
    def from_process(cls, process: "BusinessProcess") -> "DependencyGraph":
        nodes = [s.step_id for s in process.steps]
        edges = [(d.predecessor, d.successor) for d in process.dependencies]
        return cls(nodes, edges)

    # -- accessors -----------------------------------------------------------
    @property
    def nodes(self) -> Tuple[str, ...]:
        return self._nodes

    def successors(self, node: str) -> Tuple[str, ...]:
        return tuple(self._succ[node])

    def predecessors(self, node: str) -> Tuple[str, ...]:
        return tuple(self._pred[node])

    def in_degree(self, node: str) -> int:
        return len(self._pred[node])

    def out_degree(self, node: str) -> int:
        return len(self._succ[node])

    def roots(self) -> Tuple[str, ...]:
        return tuple(n for n in self._nodes if not self._pred[n])

    def leaves(self) -> Tuple[str, ...]:
        return tuple(n for n in self._nodes if not self._succ[n])

    # -- cycle detection / topological sort (Kahn's algorithm) ---------------
    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except CycleError:
            return True

    def assert_acyclic(self) -> None:
        self.topological_sort()  # raises CycleError on a cycle

    def topological_sort(self) -> Tuple[str, ...]:
        indeg = {n: len(self._pred[n]) for n in self._nodes}
        ready = [n for n in self._nodes if indeg[n] == 0]
        ready.sort(key=lambda x: self._index[x])
        order: List[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for suc in self._succ[node]:
                indeg[suc] -= 1
                if indeg[suc] == 0:
                    ready.append(suc)
            ready.sort(key=lambda x: self._index[x])
        if len(order) != len(self._nodes):
            raise CycleError("dependency graph contains a cycle")
        return tuple(order)

    def layered_order(self) -> Tuple[Tuple[str, ...], ...]:
        """Layered topological sort; each layer is a set of independent nodes."""

        indeg = {n: len(self._pred[n]) for n in self._nodes}
        current = sorted((n for n in self._nodes if indeg[n] == 0), key=lambda x: self._index[x])
        layers: List[Tuple[str, ...]] = []
        seen = 0
        while current:
            layers.append(tuple(current))
            seen += len(current)
            nxt: List[str] = []
            for node in current:
                for suc in self._succ[node]:
                    indeg[suc] -= 1
                    if indeg[suc] == 0:
                        nxt.append(suc)
            current = sorted(nxt, key=lambda x: self._index[x])
        if seen != len(self._nodes):
            raise CycleError("dependency graph contains a cycle")
        return tuple(layers)

    # -- reachability --------------------------------------------------------
    def descendants(self, node: str) -> FrozenSet[str]:
        seen: set = set()
        stack = list(self._succ[node])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self._succ[cur])
        return frozenset(seen)

    def ancestors(self, node: str) -> FrozenSet[str]:
        seen: set = set()
        stack = list(self._pred[node])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self._pred[cur])
        return frozenset(seen)

    def blocking_nodes(self) -> Tuple[str, ...]:
        """Nodes that gate downstream work, most-blocking first.

        A node blocks every one of its descendants; ranking by descendant count
        identifies the steps whose delay most threatens overall completion.
        """

        scored = [(len(self.descendants(n)), self._index[n], n) for n in self._nodes]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return tuple(n for _, _, n in scored)

    # -- connectivity --------------------------------------------------------
    def connected_components(self) -> Tuple[FrozenSet[str], ...]:
        adj: Dict[str, set] = {n: set() for n in self._nodes}
        for n in self._nodes:
            for s in self._succ[n]:
                adj[n].add(s)
                adj[s].add(n)
        seen: set = set()
        components: List[FrozenSet[str]] = []
        for start in self._nodes:
            if start in seen:
                continue
            comp: set = set()
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur not in comp:
                    comp.add(cur)
                    stack.extend(adj[cur] - comp)
            seen |= comp
            components.append(frozenset(comp))
        return tuple(components)

    def is_connected(self) -> bool:
        return len(self.connected_components()) <= 1

    def is_disconnected(self) -> bool:
        return len(self.connected_components()) > 1

    # -- critical path (longest weighted path in the DAG) --------------------
    def critical_path(self, durations: Mapping[str, float]) -> Tuple[Tuple[str, ...], float]:
        order = self.topological_sort()
        dist: Dict[str, float] = {n: float(durations.get(n, 0.0)) for n in self._nodes}
        prev: Dict[str, Optional[str]] = {n: None for n in self._nodes}
        for node in order:
            for suc in self._succ[node]:
                candidate = dist[node] + float(durations.get(suc, 0.0))
                if candidate > dist[suc]:
                    dist[suc] = candidate
                    prev[suc] = node
        if not self._nodes:
            return (), 0.0
        end = max(self._nodes, key=lambda n: (dist[n], -self._index[n]))
        path: List[str] = []
        cursor: Optional[str] = end
        while cursor is not None:
            path.append(cursor)
            cursor = prev[cursor]
        path.reverse()
        return tuple(path), dist[end]

    def validate(self) -> None:
        self.assert_acyclic()


# ---------------------------------------------------------------------------
# Execution plan model
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionGroup:
    """One stage of an execution plan."""

    group_id: str
    stage_type: StageType
    step_ids: Tuple[str, ...]
    estimated_duration: float = 0.0
    order: int = 0
    contingency: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_type", StageType.coerce(self.stage_type))
        object.__setattr__(self, "step_ids", tuple(self.step_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "stage_type": self.stage_type.value,
            "step_ids": list(self.step_ids),
            "estimated_duration": self.estimated_duration,
            "order": self.order,
            "contingency": self.contingency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionGroup":
        return cls(
            group_id=data["group_id"],
            stage_type=StageType.coerce(data["stage_type"]),
            step_ids=tuple(data.get("step_ids", ())),
            estimated_duration=float(data.get("estimated_duration", 0.0)),
            order=int(data.get("order", 0)),
            contingency=bool(data.get("contingency", False)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A deterministic plan for executing a business process."""

    process_id: str
    groups: Tuple[ExecutionGroup, ...]
    topological_order: Tuple[str, ...]
    critical_path: Tuple[str, ...]
    critical_path_duration: float
    makespan: float
    estimated_completion_time: float
    resource_estimate: float
    risk_estimate: float
    parallelism_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))

    @property
    def work_groups(self) -> Tuple[ExecutionGroup, ...]:
        return tuple(g for g in self.groups if g.stage_type in (StageType.SEQUENTIAL, StageType.PARALLEL))

    @property
    def approval_groups(self) -> Tuple[ExecutionGroup, ...]:
        return tuple(g for g in self.groups if g.stage_type is StageType.APPROVAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "groups": [g.to_dict() for g in self.groups],
            "topological_order": list(self.topological_order),
            "critical_path": list(self.critical_path),
            "critical_path_duration": self.critical_path_duration,
            "makespan": self.makespan,
            "estimated_completion_time": self.estimated_completion_time,
            "resource_estimate": self.resource_estimate,
            "risk_estimate": self.risk_estimate,
            "parallelism_ratio": self.parallelism_ratio,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        return cls(
            process_id=data["process_id"],
            groups=tuple(ExecutionGroup.from_dict(g) for g in data["groups"]),
            topological_order=tuple(data.get("topological_order", ())),
            critical_path=tuple(data.get("critical_path", ())),
            critical_path_duration=float(data["critical_path_duration"]),
            makespan=float(data["makespan"]),
            estimated_completion_time=float(data["estimated_completion_time"]),
            resource_estimate=float(data["resource_estimate"]),
            risk_estimate=float(data["risk_estimate"]),
            parallelism_ratio=float(data["parallelism_ratio"]),
        )


# ---------------------------------------------------------------------------
# Execution / approval / SLA records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StepExecutionRecord:
    """The recorded outcome of one process step (one workflow execution)."""

    step_id: str
    workflow_ref: str
    workflow_execution_id: str
    status: str
    started_at: float
    finished_at: float
    succeeded: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "workflow_ref": self.workflow_ref,
            "workflow_execution_id": self.workflow_execution_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "succeeded": self.succeeded,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepExecutionRecord":
        return cls(
            step_id=data["step_id"],
            workflow_ref=data["workflow_ref"],
            workflow_execution_id=data["workflow_execution_id"],
            status=data["status"],
            started_at=float(data["started_at"]),
            finished_at=float(data["finished_at"]),
            succeeded=bool(data["succeeded"]),
        )


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """An immutable record of a single approval-chain decision."""

    chain_id: str
    role: ApprovalRole
    approver_id: str
    state: ApprovalState
    timestamp: float
    latency: float
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ApprovalRole.coerce(self.role))
        object.__setattr__(self, "state", ApprovalState.coerce(self.state))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "role": self.role.value,
            "approver_id": self.approver_id,
            "state": self.state.value,
            "timestamp": self.timestamp,
            "latency": self.latency,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApprovalRecord":
        return cls(
            chain_id=data["chain_id"],
            role=ApprovalRole.coerce(data["role"]),
            approver_id=data["approver_id"],
            state=ApprovalState.coerce(data["state"]),
            timestamp=float(data["timestamp"]),
            latency=float(data["latency"]),
            note=data.get("note", ""),
        )


@dataclass(frozen=True, slots=True)
class ApprovalChainState:
    """An immutable snapshot of an approval chain's progress."""

    chain_id: str
    index: int
    state: ApprovalState
    records: Tuple[ApprovalRecord, ...]
    started_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ApprovalState.coerce(self.state))
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def last_timestamp(self) -> float:
        return self.records[-1].timestamp if self.records else self.started_at

    @property
    def is_complete(self) -> bool:
        return self.state in (ApprovalState.APPROVED, ApprovalState.REJECTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "index": self.index,
            "state": self.state.value,
            "records": [r.to_dict() for r in self.records],
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ApprovalChainState":
        return cls(
            chain_id=data["chain_id"],
            index=int(data["index"]),
            state=ApprovalState.coerce(data["state"]),
            records=tuple(ApprovalRecord.from_dict(r) for r in data.get("records", ())),
            started_at=float(data["started_at"]),
        )


@dataclass(frozen=True, slots=True)
class SLAReport:
    """The result of evaluating an SLA for one process execution."""

    status: SLAStatus
    expected_duration: float
    actual_duration: float
    delay: float
    remaining: float
    penalty: float
    compliance: float
    escalate: bool
    recovery_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SLAStatus.coerce(self.status))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "expected_duration": self.expected_duration,
            "actual_duration": self.actual_duration,
            "delay": self.delay,
            "remaining": self.remaining,
            "penalty": self.penalty,
            "compliance": self.compliance,
            "escalate": self.escalate,
            "recovery_time": self.recovery_time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SLAReport":
        return cls(
            status=SLAStatus.coerce(data["status"]),
            expected_duration=float(data["expected_duration"]),
            actual_duration=float(data["actual_duration"]),
            delay=float(data["delay"]),
            remaining=float(data["remaining"]),
            penalty=float(data["penalty"]),
            compliance=float(data["compliance"]),
            escalate=bool(data["escalate"]),
            recovery_time=float(data["recovery_time"]),
        )


@dataclass(frozen=True, slots=True)
class ProcessEvent:
    """An immutable orchestration audit event."""

    sequence: int
    timestamp: float
    process_id: str
    stage: str
    event: str
    actor: str
    detail_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail_json", _canonical_json(_parse_json(self.detail_json)))

    @property
    def detail(self) -> Dict[str, Any]:
        return _parse_json(self.detail_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "process_id": self.process_id,
            "stage": self.stage,
            "event": self.event,
            "actor": self.actor,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessEvent":
        return cls(
            sequence=int(data["sequence"]),
            timestamp=float(data["timestamp"]),
            process_id=data["process_id"],
            stage=data.get("stage", ""),
            event=data["event"],
            actor=data.get("actor", "orchestrator"),
            detail_json=_canonical_json(data.get("detail") or {}),
        )


@dataclass(frozen=True, slots=True)
class ProcessHistory:
    """An append-only, immutable orchestration timeline."""

    events: Tuple[ProcessEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @property
    def next_sequence(self) -> int:
        return self.events[-1].sequence + 1 if self.events else 0

    def append(self, event: ProcessEvent) -> "ProcessHistory":
        return ProcessHistory(self.events + (event,))

    def filter_by_event(self, event: str) -> Tuple[ProcessEvent, ...]:
        return tuple(e for e in self.events if e.event == event)

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessHistory":
        return cls(tuple(ProcessEvent.from_dict(e) for e in data.get("events", ())))


@dataclass(frozen=True, slots=True)
class ProcessExecutionResult:
    """An immutable snapshot of a business-process execution."""

    process_id: str
    execution_id: str
    state: ProcessState
    mode: SimulationMode
    step_results: Tuple[StepExecutionRecord, ...] = ()
    approval_records: Tuple[ApprovalRecord, ...] = ()
    history: ProcessHistory = field(default_factory=ProcessHistory)
    started_at: float = 0.0
    finished_at: float = 0.0
    actual_duration: float = 0.0
    expected_duration: float = 0.0
    critical_path_duration: float = 0.0
    makespan: float = 0.0
    parallelism_ratio: float = 0.0
    automated_steps: int = 0
    manual_steps: int = 0
    escalations: int = 0
    sla_status: SLAStatus = SLAStatus.NOT_APPLICABLE
    penalty: float = 0.0
    rolled_back: bool = False
    compensated: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ProcessState.coerce(self.state))
        object.__setattr__(self, "mode", SimulationMode.coerce(self.mode))
        object.__setattr__(self, "sla_status", SLAStatus.coerce(self.sla_status))
        object.__setattr__(self, "step_results", tuple(self.step_results))
        object.__setattr__(self, "approval_records", tuple(self.approval_records))

    @property
    def succeeded(self) -> bool:
        return self.state is ProcessState.COMPLETED

    @property
    def completed_step_ids(self) -> Tuple[str, ...]:
        return tuple(r.step_id for r in self.step_results if r.succeeded)

    def metrics(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "steps": len(self.step_results),
            "completed": len(self.completed_step_ids),
            "actual_duration": self.actual_duration,
            "expected_duration": self.expected_duration,
            "sla_status": self.sla_status.value,
            "rolled_back": self.rolled_back,
            "escalations": self.escalations,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "execution_id": self.execution_id,
            "state": self.state.value,
            "mode": self.mode.value,
            "step_results": [r.to_dict() for r in self.step_results],
            "approval_records": [r.to_dict() for r in self.approval_records],
            "history": self.history.to_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "actual_duration": self.actual_duration,
            "expected_duration": self.expected_duration,
            "critical_path_duration": self.critical_path_duration,
            "makespan": self.makespan,
            "parallelism_ratio": self.parallelism_ratio,
            "automated_steps": self.automated_steps,
            "manual_steps": self.manual_steps,
            "escalations": self.escalations,
            "sla_status": self.sla_status.value,
            "penalty": self.penalty,
            "rolled_back": self.rolled_back,
            "compensated": self.compensated,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessExecutionResult":
        return cls(
            process_id=data["process_id"],
            execution_id=data["execution_id"],
            state=ProcessState.coerce(data["state"]),
            mode=SimulationMode.coerce(data["mode"]),
            step_results=tuple(StepExecutionRecord.from_dict(r) for r in data.get("step_results", ())),
            approval_records=tuple(ApprovalRecord.from_dict(r) for r in data.get("approval_records", ())),
            history=ProcessHistory.from_dict(data.get("history", {})),
            started_at=float(data.get("started_at", 0.0)),
            finished_at=float(data.get("finished_at", 0.0)),
            actual_duration=float(data.get("actual_duration", 0.0)),
            expected_duration=float(data.get("expected_duration", 0.0)),
            critical_path_duration=float(data.get("critical_path_duration", 0.0)),
            makespan=float(data.get("makespan", 0.0)),
            parallelism_ratio=float(data.get("parallelism_ratio", 0.0)),
            automated_steps=int(data.get("automated_steps", 0)),
            manual_steps=int(data.get("manual_steps", 0)),
            escalations=int(data.get("escalations", 0)),
            sla_status=SLAStatus.coerce(data.get("sla_status", "not_applicable")),
            penalty=float(data.get("penalty", 0.0)),
            rolled_back=bool(data.get("rolled_back", False)),
            compensated=bool(data.get("compensated", False)),
            error=data.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Rollback / compensation / simulation / analytics / KPIs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """A deterministic plan to undo a partially executed process."""

    process_id: str
    strategy: RollbackStrategy
    failed_step: str
    undo_chain: Tuple[str, ...]
    recovery_chain: Tuple[str, ...]
    isolated_steps: Tuple[str, ...]
    checkpoints: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", RollbackStrategy.coerce(self.strategy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "strategy": self.strategy.value,
            "failed_step": self.failed_step,
            "undo_chain": list(self.undo_chain),
            "recovery_chain": list(self.recovery_chain),
            "isolated_steps": list(self.isolated_steps),
            "checkpoints": list(self.checkpoints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RollbackPlan":
        return cls(
            process_id=data["process_id"],
            strategy=RollbackStrategy.coerce(data["strategy"]),
            failed_step=data["failed_step"],
            undo_chain=tuple(data.get("undo_chain", ())),
            recovery_chain=tuple(data.get("recovery_chain", ())),
            isolated_steps=tuple(data.get("isolated_steps", ())),
            checkpoints=tuple(data.get("checkpoints", ())),
        )


@dataclass(frozen=True, slots=True)
class CompensationPlan:
    """A deterministic plan of compensating actions."""

    process_id: str
    compensations: Tuple[Tuple[str, str], ...]  # (failed_step, compensation_step)
    order: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "compensations", tuple(tuple(c) for c in self.compensations))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "compensations": [list(c) for c in self.compensations],
            "order": list(self.order),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompensationPlan":
        return cls(
            process_id=data["process_id"],
            compensations=tuple(tuple(c) for c in data.get("compensations", ())),
            order=tuple(data.get("order", ())),
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """The deterministic projection of a what-if / dry-run simulation."""

    process_id: str
    mode: SimulationMode
    plan: ExecutionPlan
    projected_duration: float
    projected_success_probability: float
    projected_sla_status: SLAStatus
    step_projections: Tuple[Tuple[str, float, float], ...]  # (step, duration, success_prob)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SimulationMode.coerce(self.mode))
        object.__setattr__(self, "projected_sla_status", SLAStatus.coerce(self.projected_sla_status))
        object.__setattr__(self, "step_projections", tuple(tuple(p) for p in self.step_projections))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "mode": self.mode.value,
            "plan": self.plan.to_dict(),
            "projected_duration": self.projected_duration,
            "projected_success_probability": self.projected_success_probability,
            "projected_sla_status": self.projected_sla_status.value,
            "step_projections": [list(p) for p in self.step_projections],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationResult":
        return cls(
            process_id=data["process_id"],
            mode=SimulationMode.coerce(data["mode"]),
            plan=ExecutionPlan.from_dict(data["plan"]),
            projected_duration=float(data["projected_duration"]),
            projected_success_probability=float(data["projected_success_probability"]),
            projected_sla_status=SLAStatus.coerce(data["projected_sla_status"]),
            step_projections=tuple(tuple(p) for p in data.get("step_projections", ())),
        )


@dataclass(frozen=True, slots=True)
class BusinessKPIs:
    """Enterprise KPIs aggregated across many process executions."""

    process_success_rate: float = 0.0
    average_duration: float = 0.0
    failure_rate: float = 0.0
    automation_pct: float = 0.0
    manual_pct: float = 0.0
    approval_latency: float = 0.0
    escalation_pct: float = 0.0
    rollback_pct: float = 0.0
    sla_pct: float = 0.0
    critical_path_pct: float = 0.0
    parallelism_pct: float = 0.0
    process_utilization: float = 0.0
    business_efficiency: float = 0.0

    @classmethod
    def from_results(cls, results: Sequence["ProcessExecutionResult"]) -> "BusinessKPIs":
        n = len(results)
        if n == 0:
            return cls()
        completed = sum(1 for r in results if r.state is ProcessState.COMPLETED)
        failed = sum(1 for r in results if r.state in (ProcessState.FAILED, ProcessState.ROLLED_BACK))
        rolled = sum(1 for r in results if r.rolled_back)
        total_auto = sum(r.automated_steps for r in results)
        total_manual = sum(r.manual_steps for r in results)
        total_steps = total_auto + total_manual
        latencies = [rec.latency for r in results for rec in r.approval_records
                     if rec.state is ApprovalState.APPROVED]
        approvals = sum(len(r.approval_records) for r in results)
        escalations = sum(r.escalations for r in results)
        on_track = sum(1 for r in results if r.sla_status in
                       (SLAStatus.ON_TRACK, SLAStatus.RECOVERED, SLAStatus.NOT_APPLICABLE))
        cp_ratios = [r.critical_path_duration / r.makespan for r in results if r.makespan > 0]
        util = [(_safe_div(r.critical_path_duration, r.actual_duration)) for r in results if r.actual_duration > 0]
        success_rate = completed / n
        sla_pct = on_track / n
        rollback_pct = rolled / n
        return cls(
            process_success_rate=success_rate,
            average_duration=_mean([r.actual_duration for r in results]),
            failure_rate=failed / n,
            automation_pct=_safe_div(total_auto, total_steps),
            manual_pct=_safe_div(total_manual, total_steps),
            approval_latency=_mean(latencies),
            escalation_pct=_safe_div(escalations, approvals),
            rollback_pct=rollback_pct,
            sla_pct=sla_pct,
            critical_path_pct=_mean(cp_ratios),
            parallelism_pct=_mean([r.parallelism_ratio for r in results]),
            process_utilization=_mean(util),
            business_efficiency=success_rate * sla_pct * (1.0 - rollback_pct),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_success_rate": self.process_success_rate,
            "average_duration": self.average_duration,
            "failure_rate": self.failure_rate,
            "automation_pct": self.automation_pct,
            "manual_pct": self.manual_pct,
            "approval_latency": self.approval_latency,
            "escalation_pct": self.escalation_pct,
            "rollback_pct": self.rollback_pct,
            "sla_pct": self.sla_pct,
            "critical_path_pct": self.critical_path_pct,
            "parallelism_pct": self.parallelism_pct,
            "process_utilization": self.process_utilization,
            "business_efficiency": self.business_efficiency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BusinessKPIs":
        return cls(**{k: float(data.get(k, 0.0)) for k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ProcessAnalytics:
    """Dashboard- / API- / DataFrame-ready analytics over executions."""

    total_processes: int
    state_breakdown: str           # canonical JSON {state: count}
    sla_breakdown: str             # canonical JSON {sla_status: count}
    kpis: BusinessKPIs
    duration_p50: float
    duration_p90: float
    duration_max: float

    @classmethod
    def from_results(cls, results: Sequence["ProcessExecutionResult"]) -> "ProcessAnalytics":
        states: Dict[str, int] = {}
        slas: Dict[str, int] = {}
        for r in results:
            states[r.state.value] = states.get(r.state.value, 0) + 1
            slas[r.sla_status.value] = slas.get(r.sla_status.value, 0) + 1
        durations = np.asarray([r.actual_duration for r in results], dtype=float)
        if durations.size:
            p50 = float(np.percentile(durations, 50))
            p90 = float(np.percentile(durations, 90))
            dmax = float(np.max(durations))
        else:
            p50 = p90 = dmax = 0.0
        return cls(
            total_processes=len(results),
            state_breakdown=_canonical_json(states),
            sla_breakdown=_canonical_json(slas),
            kpis=BusinessKPIs.from_results(results),
            duration_p50=p50,
            duration_p90=p90,
            duration_max=dmax,
        )

    # -- multiple presentation surfaces --------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def as_dashboard(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_processes": self.total_processes,
                "success_rate": self.kpis.process_success_rate,
                "sla_pct": self.kpis.sla_pct,
                "business_efficiency": self.kpis.business_efficiency,
            },
            "states": _parse_json(self.state_breakdown),
            "sla": _parse_json(self.sla_breakdown),
            "durations": {"p50": self.duration_p50, "p90": self.duration_p90, "max": self.duration_max},
        }

    def as_api(self) -> Dict[str, Any]:
        return {"analytics": self.to_dict(), "kpis": self.kpis.to_dict()}

    def as_dataframe_records(self) -> List[Dict[str, Any]]:
        """A flat list-of-dicts suitable for ``DataFrame(records)`` (no Pandas here)."""

        records = [{"metric": k, "value": v} for k, v in self.kpis.to_dict().items()]
        records.append({"metric": "duration_p50", "value": self.duration_p50})
        records.append({"metric": "duration_p90", "value": self.duration_p90})
        records.append({"metric": "duration_max", "value": self.duration_max})
        return records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_processes": self.total_processes,
            "state_breakdown": _parse_json(self.state_breakdown),
            "sla_breakdown": _parse_json(self.sla_breakdown),
            "kpis": self.kpis.to_dict(),
            "duration_p50": self.duration_p50,
            "duration_p90": self.duration_p90,
            "duration_max": self.duration_max,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessAnalytics":
        return cls(
            total_processes=int(data["total_processes"]),
            state_breakdown=_canonical_json(data.get("state_breakdown") or {}),
            sla_breakdown=_canonical_json(data.get("sla_breakdown") or {}),
            kpis=BusinessKPIs.from_dict(data.get("kpis") or {}),
            duration_p50=float(data.get("duration_p50", 0.0)),
            duration_p90=float(data.get("duration_p90", 0.0)),
            duration_max=float(data.get("duration_max", 0.0)),
        )


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


# ---------------------------------------------------------------------------
# Protocols (dependency-injection seams)
# ---------------------------------------------------------------------------
@runtime_checkable
class WorkflowProvider(Protocol):
    """Resolves a ProcessStep ``workflow_ref`` to a Phase 1 WorkflowDefinition."""

    def __call__(self, workflow_ref: str) -> "wfe.WorkflowDefinition":
        ...


@runtime_checkable
class DecisionProvider(Protocol):
    """Decides each rung of a process-level approval chain (deterministic)."""

    def __call__(self, chain: ApprovalChain, step: ApprovalStep) -> str:
        ...


def _approve_all(chain: ApprovalChain, step: ApprovalStep) -> str:
    return "approve"


# ---------------------------------------------------------------------------
# Process registry (thread-safe, versioned, freezable)
# ---------------------------------------------------------------------------
class ProcessRegistry:
    """A thread-safe repository of versioned business-process snapshots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, BusinessProcess] = {}

    def register(self, process: BusinessProcess, *, overwrite: bool = False) -> None:
        if not isinstance(process, BusinessProcess):
            raise ProcessValidationError("only BusinessProcess instances may be registered")
        with self._lock:
            if process.process_id in self._store and not overwrite:
                raise ProcessRegistryError(f"process already registered: {process.process_id}")
            self._store[process.process_id] = process

    def remove(self, process_id: str) -> None:
        with self._lock:
            existing = self._store.get(process_id)
            if existing is None:
                raise ProcessRegistryError(f"unknown process: {process_id}")
            if existing.frozen:
                raise ProcessRegistryError(f"cannot remove a frozen process: {process_id}")
            del self._store[process_id]

    def exists(self, process_id: str) -> bool:
        with self._lock:
            return process_id in self._store

    def lookup(self, process_id: str) -> BusinessProcess:
        with self._lock:
            try:
                return self._store[process_id]
            except KeyError as exc:
                raise ProcessRegistryError(f"unknown process: {process_id}") from exc

    def find(self, predicate: Callable[[BusinessProcess], bool]) -> Tuple[BusinessProcess, ...]:
        with self._lock:
            return tuple(p for _, p in sorted(self._store.items()) if predicate(p))

    def list(self) -> Tuple[BusinessProcess, ...]:
        with self._lock:
            return tuple(self._store[k] for k in sorted(self._store))

    def freeze(self, process_id: str) -> BusinessProcess:
        with self._lock:
            process = self.lookup(process_id)
            frozen = replace(process, frozen=True)
            self._store[process_id] = frozen
            return frozen

    def update_version(self, process: BusinessProcess) -> BusinessProcess:
        with self._lock:
            existing = self._store.get(process.process_id)
            if existing is None:
                raise ProcessRegistryError(f"unknown process: {process.process_id}")
            if existing.frozen:
                raise ProcessRegistryError(f"cannot update a frozen process: {process.process_id}")
            updated = replace(process, version=existing.version + 1)
            self._store[process.process_id] = updated
            return updated

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, process_id: object) -> bool:
        with self._lock:
            return process_id in self._store


# ---------------------------------------------------------------------------
# Execution planner
# ---------------------------------------------------------------------------
class ExecutionPlanner:
    """Produces deterministic execution plans from a business process."""

    def plan(self, process: BusinessProcess, *, start_time: float = 0.0) -> ExecutionPlan:
        graph = DependencyGraph.from_process(process)
        graph.assert_acyclic()
        step_map = {s.step_id: s for s in process.steps}
        durations = {s.step_id: s.estimated_duration for s in process.steps}
        layers = graph.layered_order()
        calendar = process.calendar or BusinessCalendar.always_on()

        groups: List[ExecutionGroup] = []
        order = 0
        makespan = 0.0
        for layer in layers:
            approval_ids = tuple(sid for sid in layer if step_map[sid].requires_approval)
            if approval_ids:
                groups.append(ExecutionGroup(
                    group_id=f"wait-{order}", stage_type=StageType.WAITING,
                    step_ids=approval_ids, estimated_duration=0.0, order=order))
                order += 1
                approval_duration = max(
                    self._approval_duration(process, sid) for sid in approval_ids
                )
                groups.append(ExecutionGroup(
                    group_id=f"approval-{order}", stage_type=StageType.APPROVAL,
                    step_ids=approval_ids, estimated_duration=approval_duration, order=order))
                order += 1
            stage_type = StageType.PARALLEL if len(layer) > 1 else StageType.SEQUENTIAL
            stage_duration = max(durations[sid] for sid in layer)
            makespan += stage_duration
            groups.append(ExecutionGroup(
                group_id=f"work-{order}", stage_type=stage_type,
                step_ids=tuple(layer), estimated_duration=stage_duration, order=order))
            order += 1

        # Contingency stages (planned, executed only on failure).
        undo_chain = tuple(reversed(graph.topological_sort()))
        groups.append(ExecutionGroup(
            group_id=f"rollback-{order}", stage_type=StageType.ROLLBACK,
            step_ids=undo_chain, estimated_duration=0.0, order=order, contingency=True))
        order += 1
        comp_ids = tuple(s.step_id for s in process.steps if s.compensation_step_id)
        if comp_ids:
            groups.append(ExecutionGroup(
                group_id=f"compensation-{order}", stage_type=StageType.COMPENSATION,
                step_ids=comp_ids, estimated_duration=0.0, order=order, contingency=True))
            order += 1

        critical_path, cp_duration = graph.critical_path(durations)
        completion = calendar.advance_business_seconds(start_time, makespan)
        resource_estimate = float(sum(s.resource_cost for s in process.steps))
        risk_estimate = self._risk_estimate(process.steps)
        num_steps = len(process.steps)
        num_work_stages = len([g for g in groups if g.stage_type in
                               (StageType.SEQUENTIAL, StageType.PARALLEL)])
        parallelism_ratio = 1.0 - _safe_div(num_work_stages, num_steps) if num_steps else 0.0

        return ExecutionPlan(
            process_id=process.process_id,
            groups=tuple(groups),
            topological_order=graph.topological_sort(),
            critical_path=critical_path,
            critical_path_duration=cp_duration,
            makespan=makespan,
            estimated_completion_time=completion,
            resource_estimate=resource_estimate,
            risk_estimate=risk_estimate,
            parallelism_ratio=parallelism_ratio,
        )

    @staticmethod
    def _approval_duration(process: BusinessProcess, step_id: str) -> float:
        step = process.get_step(step_id)
        if not step.approval_chain_id:
            return 0.0
        try:
            return process.get_chain(step.approval_chain_id).modeled_latency_seconds
        except ApprovalError:
            return 0.0

    @staticmethod
    def _risk_estimate(steps: Sequence[ProcessStep]) -> float:
        if not steps:
            return 0.0
        survival = np.prod([1.0 - s.risk_weight for s in steps])
        return float(np.clip(1.0 - survival, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Approval engine (stateless; immutable snapshots)
# ---------------------------------------------------------------------------
class ApprovalEngine:
    """Drives enterprise approval chains via immutable state snapshots."""

    def start(self, chain: ApprovalChain, *, clock: Optional["wfe.Clock"] = None) -> ApprovalChainState:
        now = (clock or wfe.LogicalClock()).now()
        return ApprovalChainState(chain.chain_id, 0, ApprovalState.PENDING, (), now)

    def _record(self, chain: ApprovalChain, state: ApprovalChainState,
                approver_id: str, decision: ApprovalState, clock: "wfe.Clock", note: str) -> ApprovalRecord:
        step = chain.steps[state.index]
        now = clock.now()
        latency = max(0.0, now - state.last_timestamp)
        return ApprovalRecord(chain.chain_id, step.role, approver_id or step.approver_id,
                              decision, now, latency, note)

    def approve(self, chain: ApprovalChain, state: ApprovalChainState, *,
                approver_id: str = "", clock: Optional["wfe.Clock"] = None,
                note: str = "") -> ApprovalChainState:
        self._require_open(state)
        clk = clock or wfe.LogicalClock()
        record = self._record(chain, state, approver_id, ApprovalState.APPROVED, clk, note)
        index = state.index + 1
        new_state = ApprovalState.APPROVED if index >= len(chain.steps) else ApprovalState.PENDING
        return replace(state, index=index, state=new_state, records=state.records + (record,))

    def reject(self, chain: ApprovalChain, state: ApprovalChainState, *,
               approver_id: str = "", clock: Optional["wfe.Clock"] = None,
               note: str = "") -> ApprovalChainState:
        self._require_open(state)
        clk = clock or wfe.LogicalClock()
        record = self._record(chain, state, approver_id, ApprovalState.REJECTED, clk, note)
        return replace(state, state=ApprovalState.REJECTED, records=state.records + (record,))

    def delegate(self, chain: ApprovalChain, state: ApprovalChainState, *,
                 to_approver_id: str, clock: Optional["wfe.Clock"] = None,
                 note: str = "") -> ApprovalChainState:
        self._require_open(state)
        if not to_approver_id:
            raise ApprovalError("delegation requires a target approver")
        clk = clock or wfe.LogicalClock()
        record = self._record(chain, state, to_approver_id, ApprovalState.DELEGATED, clk, note)
        # Delegation reassigns the approver but keeps the same rung pending.
        return replace(state, records=state.records + (record,))

    def escalate(self, chain: ApprovalChain, state: ApprovalChainState, *,
                 clock: Optional["wfe.Clock"] = None, note: str = "") -> ApprovalChainState:
        self._require_open(state)
        clk = clock or wfe.LogicalClock()
        record = self._record(chain, state, "", ApprovalState.ESCALATED, clk, note)
        index = state.index + 1
        new_state = ApprovalState.APPROVED if index >= len(chain.steps) else ApprovalState.PENDING
        return replace(state, index=index, state=new_state, records=state.records + (record,))

    def timeout(self, chain: ApprovalChain, state: ApprovalChainState, *,
                clock: Optional["wfe.Clock"] = None) -> ApprovalChainState:
        self._require_open(state)
        clk = clock or wfe.LogicalClock()
        record = self._record(chain, state, "", ApprovalState.TIMED_OUT, clk, "timeout")
        timed = replace(state, records=state.records + (record,))
        return self.escalate(chain, timed, clock=clk, note="escalated after timeout")

    def history(self, state: ApprovalChainState) -> Tuple[ApprovalRecord, ...]:
        return state.records

    def run_full(self, chain: ApprovalChain, *, decision_provider: DecisionProvider = _approve_all,
                 clock: Optional["wfe.Clock"] = None) -> ApprovalChainState:
        clk = clock or wfe.LogicalClock()
        state = self.start(chain, clock=clk)
        guard = 0
        while not state.is_complete and guard < 1000:
            guard += 1
            step = chain.steps[state.index]
            decision = decision_provider(chain, step)
            if decision == "approve":
                state = self.approve(chain, state, approver_id=step.approver_id, clock=clk)
            elif decision == "reject":
                state = self.reject(chain, state, approver_id=step.approver_id, clock=clk)
            elif decision == "escalate":
                state = self.escalate(chain, state, clock=clk)
            elif decision == "timeout":
                state = self.timeout(chain, state, clock=clk)
            else:
                raise ApprovalError(f"unknown decision: {decision!r}")
        return state

    @staticmethod
    def metrics(states: Sequence[ApprovalChainState]) -> Dict[str, Any]:
        records = [r for s in states for r in s.records]
        approved = [r for r in records if r.state is ApprovalState.APPROVED]
        escalations = sum(1 for r in records if r.state is ApprovalState.ESCALATED)
        delegations = sum(1 for r in records if r.state is ApprovalState.DELEGATED)
        timeouts = sum(1 for r in records if r.state is ApprovalState.TIMED_OUT)
        return {
            "chains": len(states),
            "approved_chains": sum(1 for s in states if s.state is ApprovalState.APPROVED),
            "rejected_chains": sum(1 for s in states if s.state is ApprovalState.REJECTED),
            "records": len(records),
            "escalations": escalations,
            "delegations": delegations,
            "timeouts": timeouts,
            "average_latency": _mean([r.latency for r in approved]),
            "escalation_pct": _safe_div(escalations, len(records)),
        }

    @staticmethod
    def _require_open(state: ApprovalChainState) -> None:
        if state.is_complete:
            raise ApprovalError(f"approval chain already {state.state.value}")


# ---------------------------------------------------------------------------
# SLA engine
# ---------------------------------------------------------------------------
class SLAEngine:
    """Evaluates SLA compliance deterministically."""

    def evaluate(self, config: Optional[SLAConfig], actual_duration: float, *,
                 recovered: bool = False) -> SLAReport:
        if config is None:
            return SLAReport(SLAStatus.NOT_APPLICABLE, 0.0, actual_duration, 0.0, 0.0,
                             0.0, 1.0, False, 0.0)
        expected = config.expected_duration
        delay = max(0.0, actual_duration - expected)
        remaining = expected - actual_duration
        warning_at = expected * config.warning_threshold
        escalate_at = expected * config.escalation_threshold
        violated = actual_duration > escalate_at
        if violated and recovered:
            status = SLAStatus.RECOVERED
        elif violated:
            status = SLAStatus.VIOLATED
        elif actual_duration >= warning_at:
            status = SLAStatus.WARNING
        else:
            status = SLAStatus.ON_TRACK
        penalty = delay * config.penalty_per_second if violated else 0.0
        compliance = float(np.clip(_safe_div(expected, actual_duration) if actual_duration > 0 else 1.0, 0.0, 1.0))
        recovery_time = min(delay, config.max_recovery_time) if config.max_recovery_time else delay
        escalate = violated and not recovered
        return SLAReport(status, expected, actual_duration, delay, remaining, penalty,
                         compliance, escalate, recovery_time)

    @staticmethod
    def compliance_pct(reports: Sequence[SLAReport]) -> float:
        if not reports:
            return 0.0
        ok = sum(1 for r in reports if r.status in
                 (SLAStatus.ON_TRACK, SLAStatus.RECOVERED, SLAStatus.NOT_APPLICABLE))
        return ok / len(reports)


# ---------------------------------------------------------------------------
# Rollback engine
# ---------------------------------------------------------------------------
class RollbackEngine:
    """Builds rollback and compensation plans deterministically."""

    def rollback_plan(self, process: BusinessProcess, failed_step: str,
                      completed_steps: Sequence[str], *,
                      strategy: RollbackStrategy = RollbackStrategy.SEQUENTIAL_UNDO) -> RollbackPlan:
        if failed_step not in process.step_ids:
            raise RollbackError(f"unknown failed step: {failed_step}")
        graph = DependencyGraph.from_process(process)
        topo = graph.topological_sort()
        completed_set = set(completed_steps)
        # Undo completed steps in reverse topological order.
        undo_chain = tuple(s for s in reversed(topo) if s in completed_set)
        # Isolate the failed step and everything downstream of it.
        isolated = (failed_step,) + tuple(
            s for s in topo if s in graph.descendants(failed_step)
        )
        # Recovery re-runs the failed step and its descendants (forward order).
        recovery_chain = tuple(s for s in topo if s == failed_step or s in graph.descendants(failed_step))
        # Checkpoints are completed steps that gate downstream work.
        blocking = set(graph.blocking_nodes())
        checkpoints = tuple(s for s in topo if s in completed_set and s in blocking)
        return RollbackPlan(
            process_id=process.process_id, strategy=strategy, failed_step=failed_step,
            undo_chain=undo_chain, recovery_chain=recovery_chain,
            isolated_steps=isolated, checkpoints=checkpoints)

    def compensation_plan(self, process: BusinessProcess) -> CompensationPlan:
        graph = DependencyGraph.from_process(process)
        topo = graph.topological_sort()
        comps = tuple(
            (s.step_id, s.compensation_step_id)
            for s in process.steps if s.compensation_step_id
        )
        comp_steps = {c[0] for c in comps}
        order = tuple(s for s in reversed(topo) if s in comp_steps)
        return CompensationPlan(process.process_id, comps, order)


# ---------------------------------------------------------------------------
# Orchestrator (facade)
# ---------------------------------------------------------------------------
class BusinessProcessOrchestrator:
    """Coordinates business processes built from Phase 1 workflows.

    Collaborators are injected; the orchestrator is otherwise stateless and
    produces immutable result snapshots, which makes planning and execution
    fully deterministic and thread-safe.
    """

    def __init__(
        self,
        workflow_engine: Optional["wfe.WorkflowEngine"] = None,
        registry: Optional[ProcessRegistry] = None,
        planner: Optional[ExecutionPlanner] = None,
        approval_engine: Optional[ApprovalEngine] = None,
        sla_engine: Optional[SLAEngine] = None,
        rollback_engine: Optional[RollbackEngine] = None,
        *,
        workflow_provider: Optional[WorkflowProvider] = None,
        clock_factory: Callable[[], "wfe.Clock"] = wfe.LogicalClock,
        enable_rollback: bool = True,
        default_actor: str = "orchestrator",
    ) -> None:
        self.workflow_engine = workflow_engine or wfe.create_default_engine()
        self.registry = registry or ProcessRegistry()
        self.planner = planner or ExecutionPlanner()
        self.approval_engine = approval_engine or ApprovalEngine()
        self.sla_engine = sla_engine or SLAEngine()
        self.rollback_engine = rollback_engine or RollbackEngine()
        self._clock_factory = clock_factory
        self.enable_rollback = enable_rollback
        self.default_actor = default_actor
        self._workflow_provider = workflow_provider or self._default_provider

    # -- registration --------------------------------------------------------
    def register_process(self, process: BusinessProcess, *, overwrite: bool = False) -> None:
        self.registry.register(process, overwrite=overwrite)

    def _resolve(self, process: Any) -> BusinessProcess:
        if isinstance(process, BusinessProcess):
            return process
        return self.registry.lookup(str(process))

    def _default_provider(self, workflow_ref: str) -> "wfe.WorkflowDefinition":
        try:
            return self.workflow_engine.registry.get_workflow(workflow_ref)
        except Exception as exc:  # noqa: BLE001 - normalise to planning error
            raise PlanningError(
                f"no workflow definition registered for ref {workflow_ref!r}"
            ) from exc

    # -- planning ------------------------------------------------------------
    def plan(self, process: Any, *, start_time: float = 0.0) -> ExecutionPlan:
        return self.planner.plan(self._resolve(process), start_time=start_time)

    # -- dependency graph passthrough ---------------------------------------
    def dependency_graph(self, process: Any) -> DependencyGraph:
        return DependencyGraph.from_process(self._resolve(process))

    # -- execution -----------------------------------------------------------
    def execute(
        self,
        process: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
        decision_provider: DecisionProvider = _approve_all,
        mode: SimulationMode = SimulationMode.LIVE,
        start_time: float = 0.0,
    ) -> ProcessExecutionResult:
        proc = self._resolve(process)
        proc.validate()
        plan = self.planner.plan(proc, start_time=start_time)
        step_map = {s.step_id: s for s in proc.steps}
        clock = self._clock_factory()
        # Seat the orchestration clock at the start time for deterministic timing.
        if isinstance(clock, wfe.LogicalClock):
            clock.advance(max(0.0, start_time - clock.now()))
        ctx = dict(context or {})

        execution_id = f"pexec-{proc.fingerprint()}-{_short_hash(ctx, mode.value)}"
        history = ProcessHistory()
        history = self._emit(history, clock, proc.process_id, "", "PROCESS_STARTED",
                             self.default_actor, {"mode": mode.value})

        step_results: List[StepExecutionRecord] = []
        approval_records: List[ApprovalRecord] = []
        escalations = 0
        completed: List[str] = []
        failed_step: Optional[str] = None
        rejected = False

        for group in plan.groups:
            if group.contingency:
                continue  # rollback / compensation are executed only on failure
            if group.stage_type is StageType.WAITING:
                history = self._emit(history, clock, proc.process_id, group.group_id,
                                     "WAITING", self.default_actor, {})
                continue
            if group.stage_type is StageType.APPROVAL:
                for sid in group.step_ids:
                    step = step_map[sid]
                    chain = proc.get_chain(step.approval_chain_id)
                    chain_state = self.approval_engine.run_full(
                        chain, decision_provider=decision_provider, clock=clock)
                    approval_records.extend(chain_state.records)
                    escalations += sum(1 for r in chain_state.records
                                       if r.state is ApprovalState.ESCALATED)
                    history = self._emit(history, clock, proc.process_id, group.group_id,
                                         "APPROVAL", self.default_actor,
                                         {"chain": chain.chain_id, "state": chain_state.state.value})
                    if chain_state.state is ApprovalState.REJECTED:
                        rejected = True
                        failed_step = sid
                        break
                if rejected:
                    break
                continue

            # SEQUENTIAL / PARALLEL work stage.
            stage_failed = False
            for sid in group.step_ids:
                step = step_map[sid]
                record = self._run_step(step, ctx)
                step_results.append(record)
                history = self._emit(history, clock, proc.process_id, group.group_id,
                                     "STEP_COMPLETED" if record.succeeded else "STEP_FAILED",
                                     self.default_actor, {"step": sid, "status": record.status})
                if record.succeeded:
                    completed.append(sid)
                else:
                    failed_step = sid
                    stage_failed = True
            # Model concurrency: the stage costs its longest member.
            clock.advance(group.estimated_duration)
            if stage_failed:
                break

        finished_at = clock.now()
        actual_duration = max(0.0, finished_at - start_time)
        expected = proc.sla.expected_duration if proc.sla else plan.critical_path_duration
        rolled_back = False
        compensated = False

        if rejected:
            state = ProcessState.FAILED
            error = f"approval rejected at step {failed_step}"
        elif failed_step is not None:
            error = f"step {failed_step} failed"
            if self.enable_rollback:
                self.rollback_engine.rollback_plan(proc, failed_step, completed)
                rolled_back = True
                comp = self.rollback_engine.compensation_plan(proc)
                compensated = bool(comp.compensations)
                state = ProcessState.ROLLED_BACK
            else:
                state = ProcessState.FAILED
        else:
            state = ProcessState.COMPLETED
            error = ""

        sla_report = self.sla_engine.evaluate(proc.sla, actual_duration)
        history = self._emit(history, clock, proc.process_id, "",
                             "PROCESS_" + state.value.upper(), self.default_actor,
                             {"duration": actual_duration})

        automated = sum(1 for s in proc.steps if s.automated)
        manual = len(proc.steps) - automated

        return ProcessExecutionResult(
            process_id=proc.process_id,
            execution_id=execution_id,
            state=state,
            mode=mode,
            step_results=tuple(step_results),
            approval_records=tuple(approval_records),
            history=history,
            started_at=start_time,
            finished_at=finished_at,
            actual_duration=actual_duration,
            expected_duration=expected,
            critical_path_duration=plan.critical_path_duration,
            makespan=plan.makespan,
            parallelism_ratio=plan.parallelism_ratio,
            automated_steps=automated,
            manual_steps=manual,
            escalations=escalations,
            sla_status=sla_report.status,
            penalty=sla_report.penalty,
            rolled_back=rolled_back,
            compensated=compensated,
            error=error,
        )

    def _run_step(self, step: ProcessStep, context: Mapping[str, Any]) -> StepExecutionRecord:
        definition = self._workflow_provider(step.workflow_ref)
        wf_clock = wfe.LogicalClock()
        execution = self.workflow_engine.create_execution(
            definition, context=dict(context), clock=wf_clock)
        execution = self.workflow_engine.run(execution, definition, clock=wf_clock)
        guard = 0
        while (execution.state is wfe.WorkflowState.PAUSED
               and execution.pending_approval_step and guard < 100):
            guard += 1
            execution = self.workflow_engine.approve(
                execution, execution.pending_approval_step, definition,
                actor=self.default_actor, clock=wf_clock)
        succeeded = execution.state is wfe.WorkflowState.COMPLETED
        return StepExecutionRecord(
            step_id=step.step_id,
            workflow_ref=step.workflow_ref,
            workflow_execution_id=execution.execution_id,
            status=execution.state.value,
            started_at=0.0,
            finished_at=step.estimated_duration,
            succeeded=succeeded,
        )

    # -- simulation ----------------------------------------------------------
    def simulate(
        self,
        process: Any,
        *,
        mode: SimulationMode = SimulationMode.DRY_RUN,
        start_time: float = 0.0,
        overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
        decision_provider: DecisionProvider = _approve_all,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SimulationResult:
        proc = self._resolve(process)
        if mode is SimulationMode.LIVE:
            raise SimulationError("use execute() for LIVE mode")
        if overrides:
            proc = self._apply_overrides(proc, overrides)
        plan = self.planner.plan(proc, start_time=start_time)
        step_projections: List[Tuple[str, float, float]] = []
        survival = 1.0
        for s in proc.steps:
            success_prob = float(np.clip(1.0 - s.risk_weight, 0.0, 1.0))
            survival *= success_prob
            step_projections.append((s.step_id, s.estimated_duration, success_prob))
        projected_duration = plan.makespan
        sla_report = self.sla_engine.evaluate(proc.sla, projected_duration)
        return SimulationResult(
            process_id=proc.process_id,
            mode=mode,
            plan=plan,
            projected_duration=projected_duration,
            projected_success_probability=float(np.clip(survival, 0.0, 1.0)),
            projected_sla_status=sla_report.status,
            step_projections=tuple(step_projections),
        )

    @staticmethod
    def _apply_overrides(process: BusinessProcess,
                         overrides: Mapping[str, Mapping[str, float]]) -> BusinessProcess:
        new_steps = []
        for s in process.steps:
            ov = overrides.get(s.step_id)
            if ov:
                s = replace(
                    s,
                    estimated_duration=float(ov.get("estimated_duration", s.estimated_duration)),
                    risk_weight=float(ov.get("risk_weight", s.risk_weight)),
                    resource_cost=float(ov.get("resource_cost", s.resource_cost)),
                )
            new_steps.append(s)
        return replace(process, steps=tuple(new_steps))

    # -- rollback / compensation passthrough --------------------------------
    def rollback_plan(self, process: Any, failed_step: str,
                      completed_steps: Sequence[str] = ()) -> RollbackPlan:
        return self.rollback_engine.rollback_plan(self._resolve(process), failed_step, completed_steps)

    def compensation_plan(self, process: Any) -> CompensationPlan:
        return self.rollback_engine.compensation_plan(self._resolve(process))

    # -- analytics / KPIs ----------------------------------------------------
    @staticmethod
    def analytics(results: Sequence[ProcessExecutionResult]) -> ProcessAnalytics:
        return ProcessAnalytics.from_results(results)

    @staticmethod
    def kpis(results: Sequence[ProcessExecutionResult]) -> BusinessKPIs:
        return BusinessKPIs.from_results(results)

    # -- internals -----------------------------------------------------------
    def _emit(self, history: ProcessHistory, clock: "wfe.Clock", process_id: str,
              stage: str, event: str, actor: str, detail: Mapping[str, Any]) -> ProcessHistory:
        return history.append(ProcessEvent(
            sequence=history.next_sequence, timestamp=clock.now(), process_id=process_id,
            stage=stage, event=event, actor=actor, detail_json=_canonical_json(detail)))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def default_approval_chain(chain_id: str = "standard",
                           roles: Sequence[ApprovalRole] = tuple(_ROLE_ORDER),
                           sla_hours: float = 24.0) -> ApprovalChain:
    """Build a canonical Engineer -> ... -> Executive approval chain."""

    steps = tuple(
        ApprovalStep(role, f"{role.value}@enterprise", sla_hours)
        for role in roles
    )
    return ApprovalChain(chain_id, steps)


def create_default_orchestrator() -> BusinessProcessOrchestrator:
    """Wire an orchestrator with default deterministic collaborators."""

    return BusinessProcessOrchestrator(
        workflow_engine=wfe.create_default_engine(),
        registry=ProcessRegistry(),
        planner=ExecutionPlanner(),
        approval_engine=ApprovalEngine(),
        sla_engine=SLAEngine(),
        rollback_engine=RollbackEngine(),
        clock_factory=wfe.LogicalClock,
    )