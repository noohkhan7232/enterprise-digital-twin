"""Enterprise Scheduler & Automation Engine (Week 10, Phase 4).

A deterministic scheduling and automation engine for time-based, event-based,
and condition-based automation across the Enterprise Digital Twin & Decision
Intelligence Platform. It is **not** cron, APScheduler, or Celery Beat: there
is no background thread and no real timer. Time is supplied by an injectable
clock and the scheduler is advanced explicitly (``tick`` / ``advance_to``),
which makes every schedule reproducible and unit-testable.

Integration is by composition only. Jobs are executed through injected
``JobExecutor`` callables (so the Workflow Engine, Business Process
Orchestrator, and Executive Copilot are wired in without being imported or
modified), and lifecycle events are emitted to an optional Enterprise Event
Bus. The scheduler imports no platform module except, optionally and
defensively, the frozen Phase 3 event bus for event construction.

Design tenets: pure Python + NumPy, no asyncio, dependency injection,
immutable dataclasses (``frozen=True, slots=True``), thread-safe mutation
behind a re-entrant lock, the Registry / Observer / Strategy patterns, custom
exceptions, rich logging, and full JSON serialisation.

Command-line demonstration::

    python src/scheduler/enterprise_scheduler.py --demo
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
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

# --- optional composition with the frozen Phase 3 event bus ----------------
try:  # package-relative
    from ..events import enterprise_event_bus as _eeb
except (ImportError, ValueError):  # pragma: no cover - import resolution
    try:
        from src.events import enterprise_event_bus as _eeb
    except ImportError:
        try:
            import enterprise_event_bus as _eeb  # isolated execution
        except ImportError:
            _eeb = None  # the scheduler still works without the bus


__all__ = [
    # Exceptions
    "SchedulerError",
    "JobValidationError",
    "JobNotFoundError",
    "DependencyCycleError",
    "TriggerError",
    "ExecutionError",
    "ScheduleConflictError",
    # Enums
    "TriggerType",
    "JobType",
    "ExecutionPolicy",
    "JobState",
    "JobStatus",
    "AutomationType",
    # Clocks
    "Clock",
    "LogicalClock",
    "SystemClock",
    # Domain model
    "ScheduleTrigger",
    "ExecutionWindow",
    "CalendarRule",
    "SchedulePolicy",
    "ScheduledJob",
    "JobOutcome",
    "JobExecution",
    "JobHistory",
    "ScheduleResult",
    "ScheduleStatistics",
    "AutomationRule",
    # Engine
    "JobExecutor",
    "EnterpriseScheduler",
    "create_default_scheduler",
]

logger = logging.getLogger("scheduler.engine")
if not logger.handlers:  # pragma: no cover - environmental
    logger.addHandler(logging.NullHandler())

_SECONDS_PER_DAY = 86400
_EPOCH_WEEKDAY = 3  # 1970-01-01 was a Thursday (Mon=0 .. Sun=6)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SchedulerError(Exception):
    """Base class for scheduler errors."""


class JobValidationError(SchedulerError):
    """Raised when a job or domain object fails validation."""


class JobNotFoundError(SchedulerError):
    """Raised when a job id is unknown."""


class DependencyCycleError(SchedulerError):
    """Raised when job dependencies form a cycle."""


class TriggerError(SchedulerError):
    """Raised for invalid trigger configuration."""


class ExecutionError(SchedulerError):
    """Raised when a job cannot be executed."""


class ScheduleConflictError(SchedulerError):
    """Raised for invalid scheduling conflicts."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "{}"
    if not isinstance(payload, Mapping):
        raise JobValidationError(f"Expected a mapping, received {type(payload)!r}")
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
            raise JobValidationError(f"{value!r} is not a valid {cls.__name__}") from exc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class TriggerType(_ValueEnum):
    ONE_TIME = "one_time"
    FIXED_INTERVAL = "fixed_interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"
    EVENT = "event"
    CONDITION = "condition"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class JobType(_ValueEnum):
    WORKFLOW_EXECUTION = "workflow_execution"
    BUSINESS_PROCESS = "business_process"
    EXECUTIVE_REPORT = "executive_report"
    KNOWLEDGE_REFRESH = "knowledge_refresh"
    RISK_ASSESSMENT = "risk_assessment"
    SIMULATION = "simulation"
    MAINTENANCE_PLANNING = "maintenance_planning"
    PREDICTION_REFRESH = "prediction_refresh"
    HEALTH_MONITORING = "health_monitoring"
    CUSTOM = "custom"


class ExecutionPolicy(_ValueEnum):
    RUN_ONCE = "run_once"
    RETRY = "retry"
    SKIP = "skip"
    QUEUE = "queue"
    REPLACE = "replace"
    CANCEL = "cancel"
    IGNORE = "ignore"


class JobState(_ValueEnum):
    SCHEDULED = "scheduled"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class JobStatus(_ValueEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    RETRIED = "retried"


class AutomationType(_ValueEnum):
    TIME_BASED = "time_based"
    EVENT_DRIVEN = "event_driven"
    CONDITION_BASED = "condition_based"
    CHAINED = "chained"


_TIME_TRIGGERS: FrozenSet[TriggerType] = frozenset({
    TriggerType.ONE_TIME, TriggerType.FIXED_INTERVAL, TriggerType.DAILY,
    TriggerType.WEEKLY, TriggerType.MONTHLY, TriggerType.CRON, TriggerType.EMERGENCY,
})


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
            raise SchedulerError("cannot advance clock by a negative amount")
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
# Calendar / window helpers
# ---------------------------------------------------------------------------
def _weekday(epoch_day: int) -> int:
    return (epoch_day + _EPOCH_WEEKDAY) % 7


@dataclass(frozen=True, slots=True)
class CalendarRule:
    """A timezone-aware calendar gating when jobs may run."""

    timezone_offset_minutes: int = 0
    business_start_hour: int = 0
    business_end_hour: int = 24
    working_days: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    holidays: Tuple[str, ...] = ()
    maintenance_windows: Tuple[Tuple[float, float], ...] = ()
    blackout_periods: Tuple[Tuple[float, float], ...] = ()
    emergency_mode: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.business_start_hour < self.business_end_hour <= 24:
            raise JobValidationError("invalid business hours")
        object.__setattr__(self, "working_days", tuple(sorted(set(int(d) for d in self.working_days))))
        object.__setattr__(self, "holidays", tuple(self.holidays))
        object.__setattr__(self, "maintenance_windows", tuple(tuple(w) for w in self.maintenance_windows))
        object.__setattr__(self, "blackout_periods", tuple(tuple(b) for b in self.blackout_periods))
        for d in self.working_days:
            if not 0 <= d <= 6:
                raise JobValidationError("working_days entries must be 0..6")

    @classmethod
    def always_on(cls) -> "CalendarRule":
        return cls(emergency_mode=True)

    def _local(self, ts: float) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc) + _dt.timedelta(
            minutes=self.timezone_offset_minutes)

    def in_blackout(self, ts: float) -> bool:
        return any(s <= ts < e for s, e in self.blackout_periods)

    def in_maintenance_window(self, ts: float) -> bool:
        return any(s <= ts < e for s, e in self.maintenance_windows)

    def is_allowed(self, ts: float) -> bool:
        if self.in_blackout(ts):
            return False
        if self.emergency_mode:
            return True
        local = self._local(ts)
        if local.weekday() not in self.working_days:
            return False
        if local.strftime("%Y-%m-%d") in self.holidays:
            return False
        return self.business_start_hour <= local.hour < self.business_end_hour

    def next_allowed(self, ts: float, *, step: float = 900.0, max_steps: int = 20000) -> float:
        if self.is_allowed(ts):
            return ts
        cursor = ts
        for _ in range(max_steps):
            cursor += step
            if self.is_allowed(cursor):
                return cursor
        raise SchedulerError("no allowed calendar time within horizon")

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
    def from_dict(cls, data: Mapping[str, Any]) -> "CalendarRule":
        return cls(
            timezone_offset_minutes=int(data.get("timezone_offset_minutes", 0)),
            business_start_hour=int(data.get("business_start_hour", 0)),
            business_end_hour=int(data.get("business_end_hour", 24)),
            working_days=tuple(data.get("working_days", (0, 1, 2, 3, 4, 5, 6))),
            holidays=tuple(data.get("holidays", ())),
            maintenance_windows=tuple(tuple(w) for w in data.get("maintenance_windows", ())),
            blackout_periods=tuple(tuple(b) for b in data.get("blackout_periods", ())),
            emergency_mode=bool(data.get("emergency_mode", False)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionWindow:
    """An allowed daily window (local seconds-of-day) on selected weekdays."""

    start_second: int = 0
    end_second: int = _SECONDS_PER_DAY
    working_days: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    timezone_offset_minutes: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.start_second < self.end_second <= _SECONDS_PER_DAY:
            raise JobValidationError("invalid execution window bounds")
        object.__setattr__(self, "working_days", tuple(sorted(set(int(d) for d in self.working_days))))

    def is_within(self, ts: float) -> bool:
        local = ts + self.timezone_offset_minutes * 60
        day = math.floor(local / _SECONDS_PER_DAY)
        sod = local - day * _SECONDS_PER_DAY
        if _weekday(int(day)) not in self.working_days:
            return False
        return self.start_second <= sod < self.end_second

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_second": self.start_second,
            "end_second": self.end_second,
            "working_days": list(self.working_days),
            "timezone_offset_minutes": self.timezone_offset_minutes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionWindow":
        return cls(
            start_second=int(data.get("start_second", 0)),
            end_second=int(data.get("end_second", _SECONDS_PER_DAY)),
            working_days=tuple(data.get("working_days", (0, 1, 2, 3, 4, 5, 6))),
            timezone_offset_minutes=int(data.get("timezone_offset_minutes", 0)),
        )


# ---------------------------------------------------------------------------
# Cron parsing (minimal, deterministic 5-field implementation)
# ---------------------------------------------------------------------------
def _parse_cron_field(field_str: str, lo: int, hi: int) -> FrozenSet[int]:
    values: set = set()
    for part in field_str.split(","):
        step = 1
        body = part
        if "/" in part:
            body, step_s = part.split("/", 1)
            step = int(step_s)
            if step <= 0:
                raise TriggerError("cron step must be positive")
        if body == "*":
            start, end = lo, hi
        elif "-" in body:
            a, b = body.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(body)
        if start < lo or end > hi or start > end:
            raise TriggerError(f"cron field out of range: {part}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


class _CronExpr:
    __slots__ = ("minute", "hour", "dom", "month", "dow")

    def __init__(self, expr: str) -> None:
        fields = expr.split()
        if len(fields) != 5:
            raise TriggerError("cron expression must have 5 fields")
        self.minute = _parse_cron_field(fields[0], 0, 59)
        self.hour = _parse_cron_field(fields[1], 0, 23)
        self.dom = _parse_cron_field(fields[2], 1, 31)
        self.month = _parse_cron_field(fields[3], 1, 12)
        self.dow = _parse_cron_field(fields[4], 0, 6)

    def matches(self, dtobj: _dt.datetime) -> bool:
        return (
            dtobj.minute in self.minute
            and dtobj.hour in self.hour
            and dtobj.day in self.dom
            and dtobj.month in self.month
            and (dtobj.weekday() in self.dow)
        )


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScheduleTrigger:
    """A deterministic description of *when* a job fires."""

    trigger_type: TriggerType
    interval_seconds: float = 0.0
    at_second: int = 0
    weekday: int = 0
    day_of_month: int = 1
    cron_expression: str = ""
    event_name: str = ""
    condition_key: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_occurrences: int = 0
    timezone_offset_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger_type", TriggerType.coerce(self.trigger_type))
        tt = self.trigger_type
        if tt is TriggerType.FIXED_INTERVAL and self.interval_seconds <= 0:
            raise TriggerError("fixed interval requires interval_seconds > 0")
        if tt in (TriggerType.DAILY, TriggerType.WEEKLY, TriggerType.MONTHLY):
            if not 0 <= self.at_second < _SECONDS_PER_DAY:
                raise TriggerError("at_second must be within a day")
        if tt is TriggerType.WEEKLY and not 0 <= self.weekday <= 6:
            raise TriggerError("weekday must be 0..6")
        if tt is TriggerType.MONTHLY and not 1 <= self.day_of_month <= 31:
            raise TriggerError("day_of_month must be 1..31")
        if tt is TriggerType.CRON:
            _CronExpr(self.cron_expression)  # validate
        if tt is TriggerType.EVENT and not self.event_name:
            raise TriggerError("event trigger requires event_name")
        if tt is TriggerType.CONDITION and not self.condition_key:
            raise TriggerError("condition trigger requires condition_key")

    @property
    def is_time_based(self) -> bool:
        return self.trigger_type in _TIME_TRIGGERS

    def next_after(self, reference: float) -> Optional[float]:
        """Return the next firing strictly determined by ``reference``."""

        tt = self.trigger_type
        base = self.start_time if self.start_time is not None else 0.0
        candidate: Optional[float]
        if tt is TriggerType.ONE_TIME:
            target = self.start_time if self.start_time is not None else reference
            candidate = target if reference <= target else None
        elif tt is TriggerType.EMERGENCY:
            target = self.start_time if self.start_time is not None else reference
            candidate = target if reference <= target else None
        elif tt is TriggerType.FIXED_INTERVAL:
            if reference <= base:
                candidate = base
            else:
                steps = math.ceil((reference - base) / self.interval_seconds)
                candidate = base + steps * self.interval_seconds
        elif tt is TriggerType.DAILY:
            candidate = self._next_daily(reference)
        elif tt is TriggerType.WEEKLY:
            candidate = self._next_weekly(reference)
        elif tt is TriggerType.MONTHLY:
            candidate = self._next_monthly(reference)
        elif tt is TriggerType.CRON:
            candidate = self._next_cron(reference)
        else:  # EVENT / CONDITION / MANUAL
            return None
        if candidate is not None and self.end_time is not None and candidate > self.end_time:
            return None
        return candidate

    # -- per-type computation ------------------------------------------------
    def _off(self) -> float:
        return self.timezone_offset_minutes * 60.0

    def _next_daily(self, reference: float) -> float:
        off = self._off()
        local = reference + off
        day = math.floor(local / _SECONDS_PER_DAY)
        cand = day * _SECONDS_PER_DAY + self.at_second
        if cand < local:
            cand += _SECONDS_PER_DAY
        return cand - off

    def _next_weekly(self, reference: float) -> float:
        off = self._off()
        local = reference + off
        day0 = math.floor(local / _SECONDS_PER_DAY)
        for add in range(0, 14):
            day = day0 + add
            if _weekday(int(day)) == self.weekday:
                cand = day * _SECONDS_PER_DAY + self.at_second
                if cand >= local:
                    return cand - off
        return (day0 + 14) * _SECONDS_PER_DAY + self.at_second - off  # pragma: no cover

    def _next_monthly(self, reference: float) -> float:
        off = self._off()
        local_dt = _dt.datetime.fromtimestamp(reference, tz=_dt.timezone.utc) + _dt.timedelta(
            minutes=self.timezone_offset_minutes)
        year, month = local_dt.year, local_dt.month
        for _ in range(0, 60):
            dim = _days_in_month(year, month)
            day = min(self.day_of_month, dim)
            cand_local = _dt.datetime(year, month, day, tzinfo=_dt.timezone.utc) + _dt.timedelta(
                seconds=self.at_second)
            cand = cand_local.timestamp() - off
            if cand >= reference:
                return cand
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise TriggerError("could not compute next monthly occurrence")  # pragma: no cover

    def _next_cron(self, reference: float) -> Optional[float]:
        cron = _CronExpr(self.cron_expression)
        off_min = self.timezone_offset_minutes
        start_local = _dt.datetime.fromtimestamp(reference, tz=_dt.timezone.utc) + _dt.timedelta(
            minutes=off_min)
        # Round up to the next whole minute.
        minute_dt = start_local.replace(second=0, microsecond=0)
        if minute_dt.timestamp() < start_local.timestamp():
            minute_dt += _dt.timedelta(minutes=1)
        for _ in range(0, 1_500_000):
            if cron.matches(minute_dt):
                return (minute_dt - _dt.timedelta(minutes=off_min)).timestamp()
            minute_dt += _dt.timedelta(minutes=1)
        return None  # pragma: no cover

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "interval_seconds": self.interval_seconds,
            "at_second": self.at_second,
            "weekday": self.weekday,
            "day_of_month": self.day_of_month,
            "cron_expression": self.cron_expression,
            "event_name": self.event_name,
            "condition_key": self.condition_key,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "max_occurrences": self.max_occurrences,
            "timezone_offset_minutes": self.timezone_offset_minutes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleTrigger":
        return cls(
            trigger_type=TriggerType.coerce(data["trigger_type"]),
            interval_seconds=float(data.get("interval_seconds", 0.0)),
            at_second=int(data.get("at_second", 0)),
            weekday=int(data.get("weekday", 0)),
            day_of_month=int(data.get("day_of_month", 1)),
            cron_expression=data.get("cron_expression", ""),
            event_name=data.get("event_name", ""),
            condition_key=data.get("condition_key", ""),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            max_occurrences=int(data.get("max_occurrences", 0)),
            timezone_offset_minutes=int(data.get("timezone_offset_minutes", 0)),
        )


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.timezone.utc)
    else:
        nxt = _dt.datetime(year, month + 1, 1, tzinfo=_dt.timezone.utc)
    last = nxt - _dt.timedelta(days=1)
    return last.day


# ---------------------------------------------------------------------------
# Policy & job
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SchedulePolicy:
    """Execution strategy applied when a job fires (Strategy pattern)."""

    execution_policy: ExecutionPolicy = ExecutionPolicy.RETRY
    max_retries: int = 1
    retry_backoff: float = 0.0
    timeout: float = 0.0
    priority: int = 0
    max_queue: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_policy", ExecutionPolicy.coerce(self.execution_policy))
        if self.max_retries < 1:
            raise JobValidationError("max_retries must be >= 1")
        if self.retry_backoff < 0:
            raise JobValidationError("retry_backoff must be >= 0")
        if self.timeout < 0:
            raise JobValidationError("timeout must be >= 0")
        if self.max_queue < 1:
            raise JobValidationError("max_queue must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_policy": self.execution_policy.value,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "timeout": self.timeout,
            "priority": self.priority,
            "max_queue": self.max_queue,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchedulePolicy":
        return cls(
            execution_policy=ExecutionPolicy.coerce(data.get("execution_policy", "retry")),
            max_retries=int(data.get("max_retries", 1)),
            retry_backoff=float(data.get("retry_backoff", 0.0)),
            timeout=float(data.get("timeout", 0.0)),
            priority=int(data.get("priority", 0)),
            max_queue=int(data.get("max_queue", 100)),
        )


_DEFAULT_POLICY = SchedulePolicy()


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """An immutable scheduled job definition."""

    job_id: str
    name: str
    job_type: JobType
    trigger: ScheduleTrigger
    policy: SchedulePolicy = _DEFAULT_POLICY
    depends_on: Tuple[str, ...] = ()
    children: Tuple[str, ...] = ()
    enabled: bool = True
    respect_calendar: bool = False
    window: Optional[ExecutionWindow] = None
    payload_json: str = "{}"
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.job_id:
            raise JobValidationError("job_id must be non-empty")
        if not self.name:
            raise JobValidationError("name must be non-empty")
        object.__setattr__(self, "job_type", JobType.coerce(self.job_type))
        if not isinstance(self.trigger, ScheduleTrigger):
            raise JobValidationError("trigger must be a ScheduleTrigger")
        if not isinstance(self.policy, SchedulePolicy):
            raise JobValidationError("policy must be a SchedulePolicy")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "children", tuple(self.children))
        if self.job_id in self.depends_on:
            raise JobValidationError("a job cannot depend on itself")
        object.__setattr__(self, "payload_json", _canonical_json(_parse_json(self.payload_json)))
        object.__setattr__(self, "metadata_json", _canonical_json(_parse_json(self.metadata_json)))

    @property
    def payload(self) -> Dict[str, Any]:
        return _parse_json(self.payload_json)

    @property
    def metadata(self) -> Dict[str, Any]:
        return _parse_json(self.metadata_json)

    @property
    def priority(self) -> int:
        return self.policy.priority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "job_type": self.job_type.value,
            "trigger": self.trigger.to_dict(),
            "policy": self.policy.to_dict(),
            "depends_on": list(self.depends_on),
            "children": list(self.children),
            "enabled": self.enabled,
            "respect_calendar": self.respect_calendar,
            "window": self.window.to_dict() if self.window else None,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduledJob":
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            job_type=JobType.coerce(data["job_type"]),
            trigger=ScheduleTrigger.from_dict(data["trigger"]),
            policy=SchedulePolicy.from_dict(data.get("policy", {})),
            depends_on=tuple(data.get("depends_on", ())),
            children=tuple(data.get("children", ())),
            enabled=bool(data.get("enabled", True)),
            respect_calendar=bool(data.get("respect_calendar", False)),
            window=ExecutionWindow.from_dict(data["window"]) if data.get("window") else None,
            payload_json=_canonical_json(data.get("payload") or {}),
            metadata_json=_canonical_json(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class JobOutcome:
    """The deterministic result returned by a job executor."""

    success: bool
    output_json: str = "{}"
    error: str = ""
    duration: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_json", _canonical_json(_parse_json(self.output_json)))
        if self.duration < 0:
            raise JobValidationError("duration must be >= 0")

    @classmethod
    def succeeded(cls, output: Optional[Mapping[str, Any]] = None, duration: float = 0.0) -> "JobOutcome":
        return cls(True, _canonical_json(output or {}), "", duration)

    @classmethod
    def failed(cls, error: str, duration: float = 0.0) -> "JobOutcome":
        return cls(False, "{}", error, duration)

    @property
    def output(self) -> Dict[str, Any]:
        return _parse_json(self.output_json)

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "output": self.output, "error": self.error,
                "duration": self.duration}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobOutcome":
        return cls(bool(data["success"]), _canonical_json(data.get("output") or {}),
                   data.get("error", ""), float(data.get("duration", 0.0)))


@dataclass(frozen=True, slots=True)
class JobExecution:
    """An immutable record of one job execution."""

    execution_id: str
    job_id: str
    scheduled_time: float
    started_at: float
    finished_at: float
    status: JobStatus
    attempts: int
    triggered_by: str
    error: str = ""
    output_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", JobStatus.coerce(self.status))
        object.__setattr__(self, "output_json", _canonical_json(_parse_json(self.output_json)))

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def latency(self) -> float:
        return max(0.0, self.started_at - self.scheduled_time)

    @property
    def succeeded(self) -> bool:
        return self.status is JobStatus.SUCCESS

    @property
    def output(self) -> Dict[str, Any]:
        return _parse_json(self.output_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "job_id": self.job_id,
            "scheduled_time": self.scheduled_time,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "attempts": self.attempts,
            "triggered_by": self.triggered_by,
            "error": self.error,
            "output": self.output,
            "duration": self.duration,
            "latency": self.latency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobExecution":
        return cls(
            execution_id=data["execution_id"],
            job_id=data["job_id"],
            scheduled_time=float(data["scheduled_time"]),
            started_at=float(data["started_at"]),
            finished_at=float(data["finished_at"]),
            status=JobStatus.coerce(data["status"]),
            attempts=int(data["attempts"]),
            triggered_by=data.get("triggered_by", ""),
            error=data.get("error", ""),
            output_json=_canonical_json(data.get("output") or {}),
        )


@dataclass(frozen=True, slots=True)
class JobHistory:
    """An append-only, immutable execution history."""

    executions: Tuple[JobExecution, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "executions", tuple(self.executions))

    def __len__(self) -> int:
        return len(self.executions)

    def __iter__(self):
        return iter(self.executions)

    def append(self, execution: JobExecution, *, max_history: int = 0) -> "JobHistory":
        execs = self.executions + (execution,)
        if max_history and len(execs) > max_history:
            execs = execs[len(execs) - max_history:]
        return JobHistory(execs)

    def for_job(self, job_id: str) -> Tuple[JobExecution, ...]:
        return tuple(e for e in self.executions if e.job_id == job_id)

    def by_status(self, status: JobStatus) -> Tuple[JobExecution, ...]:
        status = JobStatus.coerce(status)
        return tuple(e for e in self.executions if e.status is status)

    def last(self, n: int) -> Tuple[JobExecution, ...]:
        if n < 0:
            raise JobValidationError("n must be >= 0")
        return self.executions[-n:] if n else ()

    def to_dict(self) -> Dict[str, Any]:
        return {"executions": [e.to_dict() for e in self.executions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobHistory":
        return cls(tuple(JobExecution.from_dict(e) for e in data.get("executions", ())))


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """The immutable outcome of running a job once."""

    job_id: str
    execution: JobExecution
    state: JobState
    rescheduled_for: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", JobState.coerce(self.state))

    @property
    def succeeded(self) -> bool:
        return self.execution.status is JobStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "execution": self.execution.to_dict(),
            "state": self.state.value,
            "rescheduled_for": self.rescheduled_for,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleResult":
        return cls(
            job_id=data["job_id"],
            execution=JobExecution.from_dict(data["execution"]),
            state=JobState.coerce(data["state"]),
            rescheduled_for=data.get("rescheduled_for"),
        )


@dataclass(frozen=True, slots=True)
class ScheduleStatistics:
    """Aggregate scheduler / job analytics."""

    scheduled_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    retry_count: int = 0
    average_runtime: float = 0.0
    automation_success_rate: float = 0.0
    upcoming_jobs: int = 0
    execution_latency: float = 0.0
    queue_size: int = 0
    total_executions: int = 0
    timeout_count: int = 0
    skipped_count: int = 0

    @classmethod
    def from_history(
        cls, history: JobHistory, *, scheduled_jobs: int = 0, upcoming_jobs: int = 0,
        queue_size: int = 0,
    ) -> "ScheduleStatistics":
        execs = history.executions
        success = sum(1 for e in execs if e.status is JobStatus.SUCCESS)
        failure = sum(1 for e in execs if e.status is JobStatus.FAILURE)
        timeout = sum(1 for e in execs if e.status is JobStatus.TIMEOUT)
        skipped = sum(1 for e in execs if e.status is JobStatus.SKIPPED)
        retries = sum(max(0, e.attempts - 1) for e in execs)
        terminal = success + failure + timeout
        runtimes = [e.duration for e in execs if e.status is JobStatus.SUCCESS]
        latencies = [e.latency for e in execs if e.status is JobStatus.SUCCESS]
        return cls(
            scheduled_jobs=scheduled_jobs,
            completed_jobs=success,
            failed_jobs=failure + timeout,
            retry_count=retries,
            average_runtime=_mean(runtimes),
            automation_success_rate=(success / terminal) if terminal else 0.0,
            upcoming_jobs=upcoming_jobs,
            execution_latency=_mean(latencies),
            queue_size=queue_size,
            total_executions=len(execs),
            timeout_count=timeout,
            skipped_count=skipped,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduled_jobs": self.scheduled_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "retry_count": self.retry_count,
            "average_runtime": self.average_runtime,
            "automation_success_rate": self.automation_success_rate,
            "upcoming_jobs": self.upcoming_jobs,
            "execution_latency": self.execution_latency,
            "queue_size": self.queue_size,
            "total_executions": self.total_executions,
            "timeout_count": self.timeout_count,
            "skipped_count": self.skipped_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleStatistics":
        return cls(**{k: (int(data.get(k, 0)) if k not in ("average_runtime",
                     "automation_success_rate", "execution_latency")
                     else float(data.get(k, 0.0))) for k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class AutomationRule:
    """A declarative automation rule (event / condition / chained)."""

    rule_id: str
    automation_type: AutomationType
    trigger_ref: str
    target_job_ids: Tuple[str, ...]
    condition_value: Any = None
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "automation_type", AutomationType.coerce(self.automation_type))
        object.__setattr__(self, "target_job_ids", tuple(self.target_job_ids))
        if not self.rule_id:
            raise JobValidationError("rule_id must be non-empty")
        if not self.target_job_ids:
            raise JobValidationError("automation rule needs at least one target")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "automation_type": self.automation_type.value,
            "trigger_ref": self.trigger_ref,
            "target_job_ids": list(self.target_job_ids),
            "condition_value": self.condition_value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AutomationRule":
        return cls(
            rule_id=data["rule_id"],
            automation_type=AutomationType.coerce(data["automation_type"]),
            trigger_ref=data.get("trigger_ref", ""),
            target_job_ids=tuple(data.get("target_job_ids", ())),
            condition_value=data.get("condition_value"),
            enabled=bool(data.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# Executors & event emission
# ---------------------------------------------------------------------------
@runtime_checkable
class JobExecutor(Protocol):
    """Executes a job, returning a deterministic :class:`JobOutcome`."""

    def __call__(self, job: ScheduledJob, context: Mapping[str, Any], clock: Clock) -> JobOutcome:
        ...


def _default_executor(job: ScheduledJob, context: Mapping[str, Any], clock: Clock) -> JobOutcome:
    """A pure, side-effect-free default executor producing a record.

    Real integrations (Workflow Engine, Orchestrator, Copilot) are injected via
    :meth:`EnterpriseScheduler.register_executor`; this default never touches
    another module.
    """

    record = {
        "record_id": _short_hash(job.job_id, job.job_type.value, job.payload_json),
        "job_type": job.job_type.value,
        "job_id": job.job_id,
    }
    return JobOutcome.succeeded(record, duration=0.0)


class _Emitter:
    """Records lifecycle events and, if a bus is supplied, publishes them."""

    __slots__ = ("_bus", "records")

    def __init__(self, bus: Any = None) -> None:
        self._bus = bus
        self.records: List[Tuple[str, Dict[str, Any]]] = []

    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        data = dict(payload)
        self.records.append((name, data))
        if self._bus is not None and _eeb is not None:
            event = _eeb.EnterpriseEvent.create(
                _eeb.EventType.SYSTEM_EVENT, data, topic=f"scheduler.{name}", source="scheduler")
            self._bus.publish(event)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class EnterpriseScheduler:
    """A deterministic, tick-driven scheduler and automation engine.

    The scheduler holds no background thread. Time is advanced explicitly via
    :meth:`tick` or :meth:`advance_to`; given the same jobs and the same time
    advancement, the execution sequence is identical every run.
    """

    def __init__(
        self,
        *,
        clock: Optional[Clock] = None,
        calendar: Optional[CalendarRule] = None,
        event_bus: Any = None,
        default_executor: JobExecutor = _default_executor,
        max_history: int = 0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._calendar = calendar or CalendarRule.always_on()
        self._emitter = _Emitter(event_bus)
        self._default_executor = default_executor
        self._max_history = max(0, int(max_history))
        self._log = logger_ or logger
        self._lock = threading.RLock()
        self._jobs: Dict[str, ScheduledJob] = {}
        self._paused: set = set()
        self._next_exec: Dict[str, Optional[float]] = {}
        self._occurrences: Dict[str, int] = {}
        self._queues: Dict[str, int] = {}
        self._executors: Dict[JobType, JobExecutor] = {}
        self._automation: Dict[str, AutomationRule] = {}
        self._history = JobHistory()
        self._exec_seq = 0

    # -- executor / automation registration ---------------------------------
    def register_executor(self, job_type: JobType, executor: JobExecutor) -> None:
        if not callable(executor):
            raise ExecutionError("executor must be callable")
        with self._lock:
            self._executors[JobType.coerce(job_type)] = executor

    def register_automation(self, rule: AutomationRule) -> None:
        with self._lock:
            if rule.rule_id in self._automation:
                raise SchedulerError(f"automation rule already exists: {rule.rule_id}")
            self._automation[rule.rule_id] = rule

    # -- job registry --------------------------------------------------------
    def register_job(self, job: ScheduledJob) -> ScheduledJob:
        if not isinstance(job, ScheduledJob):
            raise JobValidationError("register_job requires a ScheduledJob")
        with self._lock:
            if job.job_id in self._jobs:
                raise SchedulerError(f"job already registered: {job.job_id}")
            self._jobs[job.job_id] = job
            self._assert_acyclic()
            self._occurrences[job.job_id] = 0
            self._queues[job.job_id] = 0
            self._next_exec[job.job_id] = (
                job.trigger.next_after(self._clock.now()) if job.enabled and job.trigger.is_time_based
                else None
            )
            self._emit_job(job, "job_registered")
            return job

    def remove_job(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            del self._jobs[job_id]
            self._paused.discard(job_id)
            self._next_exec.pop(job_id, None)
            self._occurrences.pop(job_id, None)
            self._queues.pop(job_id, None)
            self._emit("schedule_updated", {"job_id": job_id, "change": "removed"})

    def pause_job(self, job_id: str) -> None:
        with self._lock:
            self._require(job_id)
            self._paused.add(job_id)
            self._emit("schedule_updated", {"job_id": job_id, "change": "paused"})

    def resume_job(self, job_id: str) -> None:
        with self._lock:
            self._require(job_id)
            self._paused.discard(job_id)
            job = self._jobs[job_id]
            if job.trigger.is_time_based and self._next_exec.get(job_id) is None:
                self._next_exec[job_id] = job.trigger.next_after(self._clock.now())
            self._emit("schedule_updated", {"job_id": job_id, "change": "resumed"})

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def list_jobs(self) -> Tuple[ScheduledJob, ...]:
        with self._lock:
            return tuple(self._jobs[k] for k in sorted(self._jobs))

    def next_execution(self, job_id: str) -> Optional[float]:
        with self._lock:
            self._require(job_id)
            return self._next_exec.get(job_id)

    def job_state(self, job_id: str) -> JobState:
        with self._lock:
            self._require(job_id)
            if job_id in self._paused:
                return JobState.PAUSED
            return JobState.SCHEDULED if self._next_exec.get(job_id) is not None else JobState.COMPLETED

    # -- driving the schedule ------------------------------------------------
    def run_now(self, job_id: str, *, context: Optional[Mapping[str, Any]] = None) -> ScheduleResult:
        with self._lock:
            self._require(job_id)
            job = self._jobs[job_id]
            now = self._clock.now()
        return self._execute(job, now, "manual", dict(context or {}))

    def tick(self, now: Optional[float] = None, *,
             context: Optional[Mapping[str, Any]] = None) -> Tuple[ScheduleResult, ...]:
        ctx = dict(context or {})
        with self._lock:
            if now is not None and now > self._clock.now():
                self._clock.advance(now - self._clock.now())
            current = self._clock.now()
            due = self._due_jobs(current)
        results: List[ScheduleResult] = []
        for job in due:
            results.append(self._fire_time_job(job, current, ctx))
        return tuple(results)

    def advance_to(self, target: float, *,
                   context: Optional[Mapping[str, Any]] = None) -> Tuple[ScheduleResult, ...]:
        ctx = dict(context or {})
        results: List[ScheduleResult] = []
        for _ in range(0, 2_000_000):
            with self._lock:
                nxt = self._earliest_due(target)
                if nxt is None:
                    if self._clock.now() < target:
                        self._clock.advance(target - self._clock.now())
                    break
                job, when = nxt
                if when > self._clock.now():
                    self._clock.advance(when - self._clock.now())
                current = self._clock.now()
            results.append(self._fire_time_job(job, current, ctx))
        return tuple(results)

    # -- event / condition automation ---------------------------------------
    def fire_event(self, event_name: str, *,
                   context: Optional[Mapping[str, Any]] = None) -> Tuple[ScheduleResult, ...]:
        ctx = dict(context or {})
        with self._lock:
            now = self._clock.now()
            targets = [j for j in self._runnable_jobs()
                       if j.trigger.trigger_type is TriggerType.EVENT
                       and j.trigger.event_name == event_name]
            rule_targets = self._automation_targets(AutomationType.EVENT_DRIVEN, event_name)
            self._emit("automation_triggered", {"trigger": event_name, "type": "event"})
        results = [self._execute(j, now, "event", ctx) for j in targets]
        results.extend(self._run_targets(rule_targets, now, "event", ctx))
        return tuple(results)

    def evaluate_conditions(self, context: Mapping[str, Any]) -> Tuple[ScheduleResult, ...]:
        ctx = dict(context)
        with self._lock:
            now = self._clock.now()
            targets = [j for j in self._runnable_jobs()
                       if j.trigger.trigger_type is TriggerType.CONDITION
                       and self._condition_met(j.trigger.condition_key, ctx)]
            rule_targets: List[str] = []
            for rule in self._automation.values():
                if (rule.enabled and rule.automation_type is AutomationType.CONDITION_BASED
                        and self._rule_condition_met(rule, ctx)):
                    rule_targets.extend(rule.target_job_ids)
            if targets or rule_targets:
                self._emit("automation_triggered", {"type": "condition"})
        results = [self._execute(j, now, "condition", ctx) for j in targets]
        results.extend(self._run_targets(rule_targets, now, "condition", ctx))
        return tuple(results)

    # -- statistics ----------------------------------------------------------
    def execution_history(self, job_id: Optional[str] = None) -> JobHistory:
        with self._lock:
            if job_id is None:
                return self._history
            return JobHistory(self._history.for_job(job_id))

    def job_statistics(self, job_id: str) -> ScheduleStatistics:
        with self._lock:
            self._require(job_id)
            history = JobHistory(self._history.for_job(job_id))
            upcoming = 1 if self._next_exec.get(job_id) is not None else 0
            queue = self._queues.get(job_id, 0)
        return ScheduleStatistics.from_history(history, scheduled_jobs=1,
                                               upcoming_jobs=upcoming, queue_size=queue)

    def scheduler_statistics(self) -> ScheduleStatistics:
        with self._lock:
            history = self._history
            now = self._clock.now()
            scheduled = len(self._jobs)
            upcoming = sum(1 for v in self._next_exec.values() if v is not None and v >= now)
            queue = sum(self._queues.values())
        return ScheduleStatistics.from_history(history, scheduled_jobs=scheduled,
                                               upcoming_jobs=upcoming, queue_size=queue)

    @property
    def emitted_events(self) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
        return tuple(self._emitter.records)

    # -- internals: scheduling ----------------------------------------------
    def _runnable_jobs(self) -> List[ScheduledJob]:
        return [j for j in self._jobs.values() if j.enabled and j.job_id not in self._paused]

    def _due_jobs(self, now: float) -> List[ScheduledJob]:
        due = []
        for job in self._runnable_jobs():
            nxt = self._next_exec.get(job.job_id)
            if nxt is not None and nxt <= now:
                due.append(job)
        return self._order(due)

    def _earliest_due(self, target: float) -> Optional[Tuple[ScheduledJob, float]]:
        best: Optional[Tuple[ScheduledJob, float]] = None
        for job in self._runnable_jobs():
            nxt = self._next_exec.get(job.job_id)
            if nxt is not None and nxt <= target:
                if best is None or nxt < best[1] or (nxt == best[1] and self._rank(job) < self._rank(best[0])):
                    best = (job, nxt)
        return best

    def _order(self, jobs: List[ScheduledJob]) -> List[ScheduledJob]:
        return sorted(jobs, key=lambda j: (self._next_exec.get(j.job_id) or 0.0, -j.priority, j.job_id))

    @staticmethod
    def _rank(job: ScheduledJob) -> Tuple[int, str]:
        return (-job.priority, job.job_id)

    def _fire_time_job(self, job: ScheduledJob, now: float,
                       context: Mapping[str, Any]) -> ScheduleResult:
        # Dependency gate: all dependencies must have a recorded success.
        if not self._dependencies_met(job):
            execution = self._make_execution(job, now, now, now, JobStatus.SKIPPED, 0,
                                             job.trigger.trigger_type.value, "dependencies not met")
            with self._lock:
                self._history = self._history.append(execution, max_history=self._max_history)
                # Advance the schedule so a permanently-blocked job cannot be
                # re-selected forever within a single advance_to() horizon.
                self._reschedule(job, now)
                rescheduled = self._next_exec.get(job.job_id)
            return ScheduleResult(job.job_id, execution, JobState.BLOCKED,
                                  rescheduled_for=rescheduled)
        result = self._execute(job, now, job.trigger.trigger_type.value, context)
        with self._lock:
            self._reschedule(job, now)
            rescheduled = self._next_exec.get(job.job_id)
        return replace(result, rescheduled_for=rescheduled)

    def _reschedule(self, job: ScheduledJob, now: float) -> None:
        self._occurrences[job.job_id] = self._occurrences.get(job.job_id, 0) + 1
        trigger = job.trigger
        if trigger.trigger_type in (TriggerType.ONE_TIME, TriggerType.EMERGENCY):
            self._next_exec[job.job_id] = None
            return
        if trigger.max_occurrences and self._occurrences[job.job_id] >= trigger.max_occurrences:
            self._next_exec[job.job_id] = None
            return
        policy = job.policy.execution_policy
        if policy is ExecutionPolicy.RUN_ONCE:
            self._next_exec[job.job_id] = None
            return
        if policy is ExecutionPolicy.CANCEL:
            self._next_exec[job.job_id] = None
            return
        nxt = trigger.next_after(now + 1e-9)
        self._next_exec[job.job_id] = nxt

    def _dependencies_met(self, job: ScheduledJob) -> bool:
        if not job.depends_on:
            return True
        succeeded = {e.job_id for e in self._history.executions if e.status is JobStatus.SUCCESS}
        return all(dep in succeeded for dep in job.depends_on)

    # -- internals: execution -----------------------------------------------
    def _execute(self, job: ScheduledJob, scheduled_time: float, triggered_by: str,
                 context: Mapping[str, Any]) -> ScheduleResult:
        with self._lock:
            executor = self._executors.get(job.job_type, self._default_executor)
            clock = self._clock
            self._emit_job(job, "job_started")
            started = clock.now()
        status = JobStatus.FAILURE
        error = ""
        output: Mapping[str, Any] = {}
        attempts = 0
        elapsed = 0.0
        for attempt in range(1, job.policy.max_retries + 1):
            attempts = attempt
            backoff = job.policy.retry_backoff * (attempt - 1)
            if backoff > 0:
                with self._lock:
                    clock.advance(backoff)
                elapsed += backoff
            try:
                outcome = executor(job, context, clock)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                error = f"{type(exc).__name__}: {exc}"
                status = JobStatus.FAILURE
                if job.policy.execution_policy is not ExecutionPolicy.RETRY:
                    break
                continue
            if not isinstance(outcome, JobOutcome):
                raise ExecutionError("executor must return a JobOutcome")
            with self._lock:
                if outcome.duration > 0:
                    clock.advance(outcome.duration)
            elapsed += outcome.duration
            if job.policy.timeout and outcome.duration > job.policy.timeout:
                status = JobStatus.TIMEOUT
                error = f"timeout ({outcome.duration} > {job.policy.timeout})"
                if job.policy.execution_policy is not ExecutionPolicy.RETRY:
                    break
                continue
            if outcome.success:
                status = JobStatus.SUCCESS
                output = outcome.output
                error = ""
                break
            error = outcome.error
            if job.policy.execution_policy is not ExecutionPolicy.RETRY:
                break
        with self._lock:
            finished = clock.now()
            execution = self._make_execution(job, scheduled_time, started, finished, status,
                                             attempts, triggered_by, error, output)
            self._history = self._history.append(execution, max_history=self._max_history)
            state = self._emit_completion(job, execution)
        result = ScheduleResult(job.job_id, execution, state,
                                rescheduled_for=self._next_exec.get(job.job_id))
        if status is JobStatus.SUCCESS and job.children:
            self._run_children(job, finished, context)
        # Chained automation rules keyed on this job id.
        chained = self._automation_targets(AutomationType.CHAINED, job.job_id)
        if status is JobStatus.SUCCESS and chained:
            self._run_targets(chained, finished, "chained", context)
        return result

    def _emit_completion(self, job: ScheduledJob, execution: JobExecution) -> JobState:
        if execution.status is JobStatus.SUCCESS:
            self._emit_job(job, "job_completed")
            return JobState.COMPLETED
        if execution.status is JobStatus.TIMEOUT:
            self._emit_job(job, "job_failed", {"reason": "timeout"})
            return JobState.FAILED
        self._emit_job(job, "job_failed", {"reason": execution.error})
        return JobState.FAILED

    def _run_children(self, parent: ScheduledJob, now: float, context: Mapping[str, Any]) -> None:
        for child_id in parent.children:
            with self._lock:
                child = self._jobs.get(child_id)
            if child is not None and child.enabled:
                self._execute(child, now, "chained", context)

    def _run_targets(self, target_ids: Sequence[str], now: float, triggered_by: str,
                     context: Mapping[str, Any]) -> List[ScheduleResult]:
        results: List[ScheduleResult] = []
        for tid in target_ids:
            with self._lock:
                job = self._jobs.get(tid)
            if job is not None and job.enabled and job.job_id not in self._paused:
                results.append(self._execute(job, now, triggered_by, context))
        return results

    def _automation_targets(self, automation_type: AutomationType, ref: str) -> List[str]:
        targets: List[str] = []
        for rule in self._automation.values():
            if rule.enabled and rule.automation_type is automation_type and rule.trigger_ref == ref:
                targets.extend(rule.target_job_ids)
        return targets

    def _make_execution(self, job: ScheduledJob, scheduled_time: float, started: float,
                        finished: float, status: JobStatus, attempts: int, triggered_by: str,
                        error: str = "", output: Optional[Mapping[str, Any]] = None) -> JobExecution:
        seq = self._exec_seq
        self._exec_seq += 1
        return JobExecution(
            execution_id=f"exec-{seq:08d}-{_short_hash(job.job_id, scheduled_time)}",
            job_id=job.job_id, scheduled_time=scheduled_time, started_at=started,
            finished_at=finished, status=status, attempts=attempts, triggered_by=triggered_by,
            error=error, output_json=_canonical_json(output or {}))

    # -- internals: conditions / dependencies / events ----------------------
    @staticmethod
    def _condition_met(key: str, context: Mapping[str, Any]) -> bool:
        return bool(context.get(key, False))

    @staticmethod
    def _rule_condition_met(rule: AutomationRule, context: Mapping[str, Any]) -> bool:
        actual = context.get(rule.trigger_ref)
        if rule.condition_value is None:
            return bool(actual)
        return actual == rule.condition_value

    def _assert_acyclic(self) -> None:
        graph = {jid: set(job.depends_on) for jid, job in self._jobs.items()}
        indeg = {jid: 0 for jid in graph}
        for jid, deps in graph.items():
            for dep in deps:
                if dep in indeg:
                    indeg[jid] += 1
        ready = [jid for jid, d in indeg.items() if d == 0]
        seen = 0
        dependents: Dict[str, List[str]] = {jid: [] for jid in graph}
        for jid, deps in graph.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(jid)
        while ready:
            node = ready.pop()
            seen += 1
            for child in dependents[node]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)
        if seen != len(graph):
            raise DependencyCycleError("job dependencies contain a cycle")

    def _emit_job(self, job: ScheduledJob, name: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        payload = {"job_id": job.job_id, "job_type": job.job_type.value, "name": job.name}
        if extra:
            payload.update(extra)
        self._emit(name, payload)

    def _emit(self, name: str, payload: Mapping[str, Any]) -> None:
        self._emitter.emit(name, payload)

    def _require(self, job_id: str) -> None:
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)

    def blocking_jobs(self) -> Tuple[str, ...]:
        """Jobs that other jobs depend upon (most-depended-upon first)."""

        with self._lock:
            counts: Dict[str, int] = {jid: 0 for jid in self._jobs}
            for job in self._jobs.values():
                for dep in job.depends_on:
                    if dep in counts:
                        counts[dep] += 1
        ranked = sorted((c for c in counts.items() if c[1] > 0), key=lambda kv: (-kv[1], kv[0]))
        return tuple(jid for jid, _ in ranked)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_default_scheduler(*, event_bus: Any = None) -> EnterpriseScheduler:
    """Factory wiring a deterministic scheduler (logical clock)."""

    return EnterpriseScheduler(clock=LogicalClock(), calendar=CalendarRule.always_on(),
                               event_bus=event_bus)


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def _demo() -> int:
    print("=" * 70)
    print("Enterprise Scheduler & Automation Engine - Week 10 Phase 4 Demo")
    print("=" * 70)

    scheduler = create_default_scheduler()
    log: List[str] = []

    def record_executor(job, context, clock):
        log.append(f"ran:{job.job_id}@{clock.now():.0f}")
        return JobOutcome.succeeded({"job": job.job_id})

    scheduler.register_executor(JobType.HEALTH_MONITORING, record_executor)
    scheduler.register_executor(JobType.PREDICTION_REFRESH, record_executor)
    scheduler.register_executor(JobType.EXECUTIVE_REPORT, record_executor)
    scheduler.register_executor(JobType.RISK_ASSESSMENT, record_executor)

    # A recurring health monitor every 10 ticks of logical time.
    scheduler.register_job(ScheduledJob(
        "health", "Health monitor", JobType.HEALTH_MONITORING,
        ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0, start_time=0.0),
        SchedulePolicy(priority=5)))

    # A one-time prediction refresh at t=25.
    scheduler.register_job(ScheduledJob(
        "predict", "Prediction refresh", JobType.PREDICTION_REFRESH,
        ScheduleTrigger(TriggerType.ONE_TIME, start_time=25.0)))

    # An event-triggered executive report, fired by a risk event.
    scheduler.register_job(ScheduledJob(
        "report", "Executive report", JobType.EXECUTIVE_REPORT,
        ScheduleTrigger(TriggerType.EVENT, event_name="risk_spike")))

    # A condition-triggered risk assessment.
    scheduler.register_job(ScheduledJob(
        "risk", "Risk assessment", JobType.RISK_ASSESSMENT,
        ScheduleTrigger(TriggerType.CONDITION, condition_key="degraded")))

    print(f"\nRegistered jobs: {[j.job_id for j in scheduler.list_jobs()]}")
    print(f"Next health run: {scheduler.next_execution('health')}")

    print("\nAdvancing logical time to t=30 ...")
    scheduler.advance_to(30.0)

    print("Firing event 'risk_spike' ...")
    scheduler.fire_event("risk_spike")

    print("Evaluating condition {'degraded': True} ...")
    scheduler.evaluate_conditions({"degraded": True})

    print("\nExecution log (deterministic):")
    for entry in log:
        print(f"  - {entry}")

    stats = scheduler.scheduler_statistics()
    print("\nScheduler statistics:")
    for key, value in stats.to_dict().items():
        print(f"  {key:<24}: {value}")

    print(f"\nEmitted lifecycle events: {len(scheduler.emitted_events)}")

    # Determinism check.
    s2 = create_default_scheduler()
    s2.register_executor(JobType.HEALTH_MONITORING, lambda j, c, k: JobOutcome.succeeded())
    s2.register_job(ScheduledJob(
        "health", "Health monitor", JobType.HEALTH_MONITORING,
        ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0, start_time=0.0)))
    s2.advance_to(30.0)
    seqs = [e.scheduled_time for e in s2.execution_history("health")]
    print(f"Deterministic health fire times: {seqs}")

    print("\nDemonstration complete.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Enterprise Scheduler & Automation Engine")
    parser.add_argument("--demo", action="store_true", help="Run the demonstration")
    args = parser.parse_args(argv)
    if args.demo:
        return _demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())