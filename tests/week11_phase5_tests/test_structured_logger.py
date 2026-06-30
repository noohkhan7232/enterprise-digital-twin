"""Tests for the structured logger."""

from __future__ import annotations

import json
import threading

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.structured_logger import StructuredLogger, ListSink, create_logger  # noqa: E402
from observability.observability_models import Clock, Severity  # noqa: E402


def logger(**kw):
    kw.setdefault("clock", Clock())
    return StructuredLogger(**kw)


def test_info_records():
    log = logger()
    log.info("hi")
    assert len(log.records()) == 1


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error", "critical"])
def test_severity_methods(level):
    log = logger()
    getattr(log, level)("msg")
    assert log.records()[0].severity.value == level.upper()


def test_list_sink_receives():
    log = logger()
    sink = ListSink()
    log.subscribe(sink)
    log.info("hi")
    assert len(sink.records) == 1


def test_callable_sink_receives():
    log = logger()
    captured = []
    log.subscribe(lambda r: captured.append(r))
    log.info("hi")
    assert len(captured) == 1


def test_unsubscribe():
    log = logger()
    sink = ListSink()
    log.subscribe(sink)
    log.unsubscribe(sink)
    log.info("hi")
    assert len(sink.records) == 0


def test_multiple_sinks():
    log = logger()
    s1, s2 = ListSink(), ListSink()
    log.subscribe(s1)
    log.subscribe(s2)
    log.info("hi")
    assert len(s1.records) == 1 and len(s2.records) == 1


def test_double_subscribe_idempotent():
    log = logger()
    sink = ListSink()
    log.subscribe(sink)
    log.subscribe(sink)
    log.info("hi")
    assert len(sink.records) == 1


def test_default_context_merged():
    log = logger(default_context={"service": "edt"})
    log.info("hi", context={"extra": 1})
    ctx = dict(log.records()[0].context)
    assert ctx["service"] == "edt" and ctx["extra"] == 1


def test_context_override():
    log = logger(default_context={"k": "default"})
    log.info("hi", context={"k": "override"})
    assert dict(log.records()[0].context)["k"] == "override"


def test_exception_metadata():
    log = logger()
    log.error("boom", exception=ValueError("bad"))
    exc = dict(log.records()[0].exception)
    assert exc["type"] == "ValueError" and exc["message"] == "bad"


def test_exception_traceback_flag():
    log = logger()
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        log.error("caught", exception=e)
    assert dict(log.records()[0].exception)["traceback_present"] is True


def test_min_severity_filters():
    log = logger(min_severity=Severity.WARNING)
    assert log.info("ignored") is None
    assert log.error("kept") is not None
    assert len(log.records()) == 1


def test_min_severity_boundary():
    log = logger(min_severity=Severity.WARNING)
    assert log.warning("kept") is not None


def test_correlation_id_set():
    log = logger()
    log.info("hi", correlation_id="c1")
    assert log.records()[0].correlation_id == "c1"


def test_request_workflow_audit_ids():
    log = logger()
    log.info("hi", request_id="r1", workflow_id="w1", audit_ref="a1")
    r = log.records()[0]
    assert r.request_id == "r1" and r.workflow_id == "w1" and r.audit_ref == "a1"


def test_filter_by_severity():
    log = logger()
    log.info("a")
    log.error("b")
    assert len(log.filter(severity=Severity.ERROR)) == 1


def test_filter_by_min_severity():
    log = logger()
    log.debug("a")
    log.warning("b")
    log.error("c")
    assert len(log.filter(min_severity=Severity.WARNING)) == 2


def test_filter_by_correlation():
    log = logger()
    log.info("a", correlation_id="c1")
    log.info("b", correlation_id="c2")
    assert len(log.filter(correlation_id="c1")) == 1


def test_filter_by_request():
    log = logger()
    log.info("a", request_id="r1")
    log.info("b")
    assert len(log.filter(request_id="r1")) == 1


def test_filter_by_workflow():
    log = logger()
    log.info("a", workflow_id="w1")
    log.info("b", workflow_id="w2")
    assert len(log.filter(workflow_id="w1")) == 1


def test_filter_combined():
    log = logger()
    log.error("a", correlation_id="c1")
    log.info("b", correlation_id="c1")
    assert len(log.filter(severity=Severity.ERROR, correlation_id="c1")) == 1


def test_count_by_severity():
    log = logger()
    log.info("a")
    log.error("b")
    log.error("c")
    counts = log.count_by_severity()
    assert counts["ERROR"] == 2 and counts["INFO"] == 1


def test_export_json_serializable():
    log = logger()
    log.info("a", correlation_id="c1")
    assert json.dumps(log.export())


def test_explicit_timestamp():
    log = logger()
    log.info("hi", timestamp=42.0)
    assert log.records()[0].timestamp == 42.0


def test_clock_timestamps():
    log = logger()
    log.info("a")
    log.info("b")
    assert [r.timestamp for r in log.records()] == [0.0, 1.0]


def test_severity_string_accepted():
    log = logger()
    log.log("ERROR", "boom")
    assert log.records()[0].severity is Severity.ERROR


def test_factory():
    assert isinstance(create_logger(), StructuredLogger)


def test_log_returns_record():
    log = logger()
    r = log.info("hi")
    assert r is not None and r.message == "hi"


def test_records_immutable_snapshot():
    log = logger()
    log.info("a")
    snapshot = log.records()
    log.info("b")
    assert len(snapshot) == 1


def test_thread_safety():
    log = logger()

    def worker():
        for _ in range(100):
            log.info("x")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.records()) == 800


def test_sink_after_unsubscribe_one_remains():
    log = logger()
    s1, s2 = ListSink(), ListSink()
    log.subscribe(s1)
    log.subscribe(s2)
    log.unsubscribe(s1)
    log.info("hi")
    assert len(s1.records) == 0 and len(s2.records) == 1


def test_count_by_severity_all_keys():
    log = logger()
    counts = log.count_by_severity()
    assert set(counts) == {s.value for s in Severity}


def test_json_record_has_all_fields():
    log = logger()
    log.info("hi", correlation_id="c1")
    d = json.loads(log.records()[0].to_json())
    assert "severity" in d and "correlation_id" in d and "context" in d