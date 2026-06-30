"""Tests for the tracing engine."""

from __future__ import annotations

import json
import threading

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.tracing_engine import Tracer, create_tracer  # noqa: E402
from observability.observability_models import Clock, SpanStatus, ValidationError  # noqa: E402


def build_trace(t, trace_id="t1"):
    # root(0-10) -> a(1-8) -> a1(2-7); root -> b(1-3)
    t.record_span("root", trace_id=trace_id, start_time=0.0, end_time=10.0, span_id="root")
    t.record_span("a", trace_id=trace_id, start_time=1.0, end_time=8.0, span_id="a", parent_id="root")
    t.record_span("b", trace_id=trace_id, start_time=1.0, end_time=3.0, span_id="b", parent_id="root")
    t.record_span("a1", trace_id=trace_id, start_time=2.0, end_time=7.0, span_id="a1", parent_id="a")
    return t


def tracer():
    return Tracer(clock=Clock())


def test_new_trace_ids():
    ctx = tracer().new_trace()
    assert ctx.trace_id.startswith("trace-") and ctx.span_id.startswith("span-")


def test_record_span_returns_span():
    t = tracer()
    s = t.record_span("op", trace_id="t1", start_time=0.0, end_time=1.0)
    assert s.duration == 1.0


def test_spans_all():
    t = build_trace(tracer())
    assert len(t.spans()) == 4


def test_spans_by_trace():
    t = build_trace(tracer())
    assert len(t.spans("t1")) == 4 and len(t.spans("other")) == 0


def test_trace_ids():
    t = build_trace(tracer())
    assert t.trace_ids() == ("t1",)


def test_timeline_order():
    t = build_trace(tracer())
    assert [s.span_id for s in t.timeline("t1")] == ["root", "a", "b", "a1"]


def test_timeline_unknown_raises():
    with pytest.raises(ValidationError):
        tracer().timeline("nope")


def test_children():
    t = build_trace(tracer())
    assert [s.span_id for s in t.children("t1", "root")] == ["a", "b"]


def test_children_leaf():
    t = build_trace(tracer())
    assert t.children("t1", "a1") == ()


def test_roots():
    t = build_trace(tracer())
    assert [s.span_id for s in t.roots("t1")] == ["root"]


def test_trace_duration():
    t = build_trace(tracer())
    assert t.trace_duration("t1") == 10.0


def test_critical_path():
    t = build_trace(tracer())
    assert [s.span_id for s in t.critical_path("t1")] == ["root", "a", "a1"]


def test_critical_path_unknown_raises():
    with pytest.raises(ValidationError):
        tracer().critical_path("nope")


def test_critical_path_single_span():
    t = tracer()
    t.record_span("solo", trace_id="t1", start_time=0.0, end_time=5.0, span_id="solo")
    assert [s.span_id for s in t.critical_path("t1")] == ["solo"]


def test_error_spans():
    t = tracer()
    t.record_span("ok", trace_id="t1", start_time=0.0, end_time=1.0, span_id="ok")
    t.record_span("bad", trace_id="t1", start_time=1.0, end_time=2.0, span_id="bad",
                  status=SpanStatus.ERROR)
    assert len(t.error_spans("t1")) == 1


def test_error_spans_global():
    t = tracer()
    t.record_span("bad", trace_id="t1", start_time=0.0, end_time=1.0, status=SpanStatus.ERROR)
    assert len(t.error_spans()) == 1


def test_export_structure():
    t = build_trace(tracer())
    exp = t.export("t1")
    assert exp["span_count"] == 4 and exp["duration"] == 10.0
    assert exp["critical_path"] == ["root", "a", "a1"]


def test_export_json_serializable():
    t = build_trace(tracer())
    assert json.dumps(t.export("t1"))


def test_context_manager_records_span():
    t = tracer()
    with t.span("op"):
        pass
    assert len(t.spans()) == 1


def test_context_manager_nested():
    t = tracer()
    with t.span("outer") as ctx:
        with t.span("inner", context=ctx):
            pass
    spans = t.spans()
    assert len(spans) == 2
    inner = [s for s in spans if s.name == "inner"][0]
    assert inner.parent_id is not None


def test_context_manager_same_trace():
    t = tracer()
    with t.span("outer") as ctx:
        with t.span("inner", context=ctx):
            pass
    assert len(t.trace_ids()) == 1


def test_context_manager_duration():
    t = tracer()
    with t.span("op"):
        pass
    # Clock advances start then end -> duration 1.0
    assert t.spans()[0].duration == 1.0


def test_context_manager_error_status():
    t = tracer()
    with t.span("op", status=SpanStatus.ERROR):
        pass
    assert t.spans()[0].is_error


def test_context_manager_attributes():
    t = tracer()
    with t.span("op", attributes={"k": "v"}):
        pass
    assert dict(t.spans()[0].attributes)["k"] == "v"


def test_record_span_custom_status():
    t = tracer()
    s = t.record_span("op", trace_id="t1", start_time=0.0, end_time=1.0, status=SpanStatus.ERROR)
    assert s.status is SpanStatus.ERROR


def test_record_span_attributes():
    t = tracer()
    s = t.record_span("op", trace_id="t1", start_time=0.0, end_time=1.0, attributes={"a": 1})
    assert dict(s.attributes)["a"] == 1


def test_multiple_traces():
    t = tracer()
    t.record_span("a", trace_id="t1", start_time=0.0, end_time=1.0)
    t.record_span("b", trace_id="t2", start_time=0.0, end_time=1.0)
    assert len(t.trace_ids()) == 2


def test_orphan_span_treated_as_root():
    t = tracer()
    t.record_span("orphan", trace_id="t1", start_time=0.0, end_time=5.0, span_id="o", parent_id="ghost")
    assert [s.span_id for s in t.roots("t1")] == ["o"]


def test_critical_path_picks_longest():
    t = tracer()
    t.record_span("root", trace_id="t1", start_time=0.0, end_time=10.0, span_id="root")
    t.record_span("short", trace_id="t1", start_time=0.0, end_time=1.0, span_id="short", parent_id="root")
    t.record_span("long", trace_id="t1", start_time=0.0, end_time=9.0, span_id="long", parent_id="root")
    assert [s.span_id for s in t.critical_path("t1")] == ["root", "long"]


def test_factory():
    assert isinstance(create_tracer(), Tracer)


def test_thread_safety():
    t = tracer()

    def worker(i):
        t.record_span(f"s{i}", trace_id="t1", start_time=0.0, end_time=1.0, span_id=f"s{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(t.spans("t1")) == 50


def test_export_error_count():
    t = tracer()
    t.record_span("bad", trace_id="t1", start_time=0.0, end_time=1.0, status=SpanStatus.ERROR)
    assert t.export("t1")["error_count"] == 1


def test_zero_duration_span():
    t = tracer()
    s = t.record_span("instant", trace_id="t1", start_time=5.0, end_time=5.0)
    assert s.duration == 0.0


def test_deterministic_ids():
    a = Tracer(clock=Clock())
    b = Tracer(clock=Clock())
    assert a.new_trace().trace_id == b.new_trace().trace_id