"""Lightweight, deterministic distributed tracing engine.

Supports trace and span identifiers, parent/child relationships, durations,
status, attributes, timeline reconstruction and critical-path analysis. Pure
Python, thread-safe, deterministic (time and identifiers are injected). No
OpenTelemetry, Jaeger or Zipkin dependency.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from .observability_models import (
    Clock, IdGenerator, SpanStatus, TraceContext, TraceSpan, ValidationError,
)

__all__ = ["Tracer", "create_tracer"]


class Tracer:
    """A deterministic span recorder with timeline and critical-path analysis."""

    def __init__(self, *, clock: Optional[Clock] = None,
                 trace_ids: Optional[IdGenerator] = None,
                 span_ids: Optional[IdGenerator] = None) -> None:
        self._clock = clock or Clock()
        self._trace_ids = trace_ids or IdGenerator("trace")
        self._span_ids = span_ids or IdGenerator("span")
        self._spans: List[TraceSpan] = []
        self._lock = threading.RLock()

    # -- context creation --------------------------------------------------- #
    def new_trace(self) -> TraceContext:
        return TraceContext(self._trace_ids.next_id(), self._span_ids.next_id())

    def record_span(self, name: str, *, trace_id: str, start_time: float,
                    end_time: float, span_id: Optional[str] = None,
                    parent_id: Optional[str] = None,
                    status: SpanStatus = SpanStatus.OK,
                    attributes: Optional[Mapping[str, Any]] = None) -> TraceSpan:
        span = TraceSpan(
            span_id=span_id or self._span_ids.next_id(), trace_id=trace_id, name=name,
            start_time=start_time, end_time=end_time, status=status,
            parent_id=parent_id, attributes=attributes or {},
        )
        with self._lock:
            self._spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, *, context: Optional[TraceContext] = None,
             status: SpanStatus = SpanStatus.OK,
             attributes: Optional[Mapping[str, Any]] = None) -> Iterator[TraceContext]:
        """Open a span; duration is measured by the injected clock."""
        if context is None:
            ctx = self.new_trace()
            parent_id: Optional[str] = None
            span_id = ctx.span_id
            trace_id = ctx.trace_id
        else:
            span_id = self._span_ids.next_id()
            parent_id = context.span_id
            trace_id = context.trace_id
            ctx = TraceContext(trace_id, span_id, parent_id, context.baggage)
        start = self._clock.now()
        try:
            yield ctx
        finally:
            end = self._clock.now()
            self.record_span(name, trace_id=trace_id, start_time=start, end_time=end,
                             span_id=span_id, parent_id=parent_id, status=status,
                             attributes=attributes or {})

    # -- retrieval ---------------------------------------------------------- #
    def spans(self, trace_id: Optional[str] = None) -> Tuple[TraceSpan, ...]:
        with self._lock:
            snapshot = tuple(self._spans)
        if trace_id is None:
            return snapshot
        return tuple(s for s in snapshot if s.trace_id == trace_id)

    def trace_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted({s.trace_id for s in self._spans}))

    # -- analysis ----------------------------------------------------------- #
    def timeline(self, trace_id: str) -> Tuple[TraceSpan, ...]:
        """Spans ordered by start time then span id (deterministic)."""
        spans = self.spans(trace_id)
        if not spans:
            raise ValidationError(f"unknown trace: {trace_id}")
        return tuple(sorted(spans, key=lambda s: (s.start_time, s.span_id)))

    def children(self, trace_id: str, span_id: str) -> Tuple[TraceSpan, ...]:
        return tuple(sorted((s for s in self.spans(trace_id) if s.parent_id == span_id),
                            key=lambda s: (s.start_time, s.span_id)))

    def roots(self, trace_id: str) -> Tuple[TraceSpan, ...]:
        spans = self.spans(trace_id)
        ids = {s.span_id for s in spans}
        return tuple(sorted((s for s in spans if s.parent_id is None or s.parent_id not in ids),
                            key=lambda s: (s.start_time, s.span_id)))

    def trace_duration(self, trace_id: str) -> float:
        spans = self.timeline(trace_id)
        start = min(s.start_time for s in spans)
        end = max(s.end_time for s in spans)
        return round(end - start, 6)

    def critical_path(self, trace_id: str) -> Tuple[TraceSpan, ...]:
        """The maximum-duration root-to-leaf path through the span tree."""
        spans = self.spans(trace_id)
        if not spans:
            raise ValidationError(f"unknown trace: {trace_id}")
        by_id = {s.span_id: s for s in spans}
        children: Dict[Optional[str], List[TraceSpan]] = {}
        for s in spans:
            parent = s.parent_id if s.parent_id in by_id else None
            children.setdefault(parent, []).append(s)

        best_path: List[TraceSpan] = []
        best_duration = -1.0

        def walk(node: TraceSpan, path: List[TraceSpan], total: float) -> None:
            nonlocal best_path, best_duration
            path = path + [node]
            total += node.duration
            kids = sorted(children.get(node.span_id, []), key=lambda s: (s.start_time, s.span_id))
            if not kids:
                if total > best_duration or (total == best_duration and not best_path):
                    best_duration, best_path = total, path
                return
            for kid in kids:
                walk(kid, path, total)

        for root in sorted(children.get(None, []), key=lambda s: (s.start_time, s.span_id)):
            walk(root, [], 0.0)
        return tuple(best_path)

    def error_spans(self, trace_id: Optional[str] = None) -> Tuple[TraceSpan, ...]:
        return tuple(s for s in self.spans(trace_id) if s.is_error)

    def export(self, trace_id: str) -> Dict[str, Any]:
        spans = self.timeline(trace_id)
        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "duration": self.trace_duration(trace_id),
            "error_count": len(self.error_spans(trace_id)),
            "critical_path": [s.span_id for s in self.critical_path(trace_id)],
            "spans": [s.to_dict() for s in spans],
        }


def create_tracer(*, clock: Optional[Clock] = None) -> Tracer:
    return Tracer(clock=clock)