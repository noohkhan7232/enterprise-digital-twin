"""Deterministic structured (JSON) logger with correlation and audit linkage.

Emits immutable :class:`LogRecord` objects carrying correlation, request and
workflow identifiers, severity, structured context, exception metadata and audit
references. Supports the Observer pattern (sinks), severity filtering and
retrieval/filtering of buffered records. Pure Python, thread-safe, deterministic.
No ELK or external logging backend.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .observability_models import Clock, LogRecord, Severity, ValidationError

__all__ = ["LogSink", "ListSink", "StructuredLogger", "create_logger"]

# A sink is either a callable taking a LogRecord or an object with on_log.
LogSink = Union[Callable[[LogRecord], None], Any]

_SEVERITY_ORDER = {
    Severity.DEBUG: 10, Severity.INFO: 20, Severity.WARNING: 30,
    Severity.ERROR: 40, Severity.CRITICAL: 50,
}


class ListSink:
    """A simple observer that accumulates records (useful for tests/dashboards)."""

    def __init__(self) -> None:
        self.records: List[LogRecord] = []
        self._lock = threading.RLock()

    def on_log(self, record: LogRecord) -> None:
        with self._lock:
            self.records.append(record)


class StructuredLogger:
    """A deterministic structured logger that buffers and fans out records."""

    def __init__(self, *, clock: Optional[Clock] = None,
                 min_severity: Severity = Severity.DEBUG,
                 default_context: Optional[Mapping[str, Any]] = None) -> None:
        self._clock = clock or Clock()
        self._min = min_severity if isinstance(min_severity, Severity) else Severity(min_severity)
        self._default_context = dict(default_context or {})
        self._records: List[LogRecord] = []
        self._sinks: List[LogSink] = []
        self._lock = threading.RLock()

    # -- observer management ------------------------------------------------ #
    def subscribe(self, sink: LogSink) -> None:
        with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def unsubscribe(self, sink: LogSink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def _emit(self, record: LogRecord) -> None:
        with self._lock:
            self._records.append(record)
            sinks = tuple(self._sinks)
        for sink in sinks:
            if callable(sink):
                sink(record)
            else:
                sink.on_log(record)

    # -- logging ------------------------------------------------------------ #
    def log(self, severity: Severity, message: str, *,
            correlation_id: Optional[str] = None, request_id: Optional[str] = None,
            workflow_id: Optional[str] = None, audit_ref: Optional[str] = None,
            context: Optional[Mapping[str, Any]] = None,
            exception: Optional[BaseException] = None,
            timestamp: Optional[float] = None) -> Optional[LogRecord]:
        severity = severity if isinstance(severity, Severity) else Severity(severity)
        if _SEVERITY_ORDER[severity] < _SEVERITY_ORDER[self._min]:
            return None
        merged = dict(self._default_context)
        merged.update(context or {})
        exc_meta: Optional[Dict[str, Any]] = None
        if exception is not None:
            exc_meta = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback_present": bool(getattr(exception, "__traceback__", None)),
            }
        ts = self._clock.now() if timestamp is None else float(timestamp)
        record = LogRecord(
            timestamp=ts, severity=severity, message=message,
            correlation_id=correlation_id, request_id=request_id, workflow_id=workflow_id,
            context=merged, exception=exc_meta, audit_ref=audit_ref,
        )
        self._emit(record)
        return record

    def debug(self, message: str, **kw: Any) -> Optional[LogRecord]:
        return self.log(Severity.DEBUG, message, **kw)

    def info(self, message: str, **kw: Any) -> Optional[LogRecord]:
        return self.log(Severity.INFO, message, **kw)

    def warning(self, message: str, **kw: Any) -> Optional[LogRecord]:
        return self.log(Severity.WARNING, message, **kw)

    def error(self, message: str, **kw: Any) -> Optional[LogRecord]:
        return self.log(Severity.ERROR, message, **kw)

    def critical(self, message: str, **kw: Any) -> Optional[LogRecord]:
        return self.log(Severity.CRITICAL, message, **kw)

    # -- retrieval / filtering ---------------------------------------------- #
    def records(self) -> Tuple[LogRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def filter(self, *, severity: Optional[Severity] = None,
               min_severity: Optional[Severity] = None,
               correlation_id: Optional[str] = None, request_id: Optional[str] = None,
               workflow_id: Optional[str] = None) -> Tuple[LogRecord, ...]:
        def keep(r: LogRecord) -> bool:
            if severity is not None and r.severity is not severity:
                return False
            if min_severity is not None and _SEVERITY_ORDER[r.severity] < _SEVERITY_ORDER[min_severity]:
                return False
            if correlation_id is not None and r.correlation_id != correlation_id:
                return False
            if request_id is not None and r.request_id != request_id:
                return False
            if workflow_id is not None and r.workflow_id != workflow_id:
                return False
            return True

        return tuple(r for r in self.records() if keep(r))

    def count_by_severity(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for r in self.records():
            counts[r.severity.value] += 1
        return counts

    def export(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records()]


def create_logger(*, clock: Optional[Clock] = None,
                  min_severity: Severity = Severity.DEBUG) -> StructuredLogger:
    return StructuredLogger(clock=clock, min_severity=min_severity)