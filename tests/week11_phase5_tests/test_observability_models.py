"""Tests for observability value objects and shared infrastructure."""

from __future__ import annotations

import json

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability import observability_models as m  # noqa: E402


# --------------------------------------------------------------------------- #
# Infrastructure: Clock / ManualClock / IdGenerator
# --------------------------------------------------------------------------- #
def test_clock_advances():
    c = m.Clock()
    assert (c.now(), c.now(), c.now()) == (0.0, 1.0, 2.0)


def test_clock_custom_step():
    c = m.Clock(start=10.0, step=5.0)
    assert (c.now(), c.now()) == (10.0, 15.0)


def test_clock_peek_does_not_advance():
    c = m.Clock()
    c.now()
    assert c.peek() == 1.0 and c.peek() == 1.0


def test_manual_clock_static():
    mc = m.ManualClock(start=3.0)
    assert mc.now() == 3.0 and mc.now() == 3.0


def test_manual_clock_advance():
    mc = m.ManualClock()
    mc.advance(7.5)
    assert mc.now() == 7.5


def test_id_generator_sequence():
    g = m.IdGenerator("x")
    assert g.next_id() == "x-00000001" and g.next_id() == "x-00000002"


def test_id_generator_seed():
    g = m.IdGenerator("y", seed=100)
    assert g.next_id() == "y-00000101"


def test_id_generator_unique():
    g = m.IdGenerator("z")
    ids = {g.next_id() for _ in range(1000)}
    assert len(ids) == 1000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_freeze_mapping_sorted():
    assert m.freeze_mapping({"b": 1, "a": 2}) == (("a", 2), ("b", 1))


def test_freeze_mapping_empty():
    assert m.freeze_mapping(None) == () and m.freeze_mapping({}) == ()


def test_thaw_mapping_roundtrip():
    frozen = m.freeze_mapping({"a": 1, "b": 2})
    assert m.thaw_mapping(frozen) == {"a": 1, "b": 2}


@pytest.mark.parametrize("value,expected", [(-1, 0.0), (0.5, 0.5), (2.0, 1.0), (0.0, 0.0), (1.0, 1.0)])
def test_clamp(value, expected):
    assert m.clamp(value) == expected


def test_clamp_custom_bounds():
    assert m.clamp(50, 0, 100) == 50 and m.clamp(150, 0, 100) == 100


@pytest.mark.parametrize("pct,expected", [(0, 1.0), (50, 5.5), (100, 10.0)])
def test_percentile(pct, expected):
    assert m.percentile(list(range(1, 11)), pct) == expected


def test_percentile_empty():
    assert m.percentile([], 50) == 0.0


def test_percentile_single():
    assert m.percentile([42.0], 95) == 42.0


# --------------------------------------------------------------------------- #
# MetricPoint / MetricSeries
# --------------------------------------------------------------------------- #
def test_metric_point_roundtrip():
    p = m.MetricPoint(1.0, 2.5, {"region": "eu"})
    assert m.MetricPoint.from_dict(p.to_dict()) == p


def test_metric_point_coerces_floats():
    p = m.MetricPoint(1, 2)
    assert isinstance(p.timestamp, float) and isinstance(p.value, float)


def test_metric_point_immutable():
    p = m.MetricPoint(1.0, 2.0)
    with pytest.raises(Exception):
        p.value = 3.0  # type: ignore


def test_metric_series_stats():
    s = m.MetricSeries("x", m.MetricType.GAUGE,
                       (m.MetricPoint(0.0, 2.0), m.MetricPoint(1.0, 4.0)))
    assert s.count == 2 and s.mean == 3.0 and s.total == 6.0 and s.latest == 4.0


def test_metric_series_empty():
    s = m.MetricSeries("x", m.MetricType.GAUGE, ())
    assert s.count == 0 and s.mean == 0.0 and s.latest is None


def test_metric_series_roundtrip():
    s = m.MetricSeries("x", m.MetricType.COUNTER, (m.MetricPoint(0.0, 1.0),), "req")
    assert m.MetricSeries.from_dict(s.to_dict()) == s


def test_metric_series_requires_name():
    with pytest.raises(m.ValidationError):
        m.MetricSeries("", m.MetricType.GAUGE)


def test_metric_series_coerces_type():
    s = m.MetricSeries("x", "GAUGE")
    assert s.metric_type is m.MetricType.GAUGE


def test_metric_series_values():
    s = m.MetricSeries("x", m.MetricType.GAUGE, (m.MetricPoint(0.0, 1.0), m.MetricPoint(1.0, 2.0)))
    assert s.values == (1.0, 2.0)


# --------------------------------------------------------------------------- #
# TraceSpan / TraceContext
# --------------------------------------------------------------------------- #
def test_trace_span_duration():
    s = m.TraceSpan("s", "t", "op", 1.0, 4.0)
    assert s.duration == 3.0


def test_trace_span_roundtrip():
    s = m.TraceSpan("s", "t", "op", 1.0, 4.0, m.SpanStatus.ERROR, "p", {"k": "v"})
    assert m.TraceSpan.from_dict(s.to_dict()) == s


def test_trace_span_is_error():
    assert m.TraceSpan("s", "t", "op", 0.0, 1.0, m.SpanStatus.ERROR).is_error


def test_trace_span_requires_ids():
    with pytest.raises(m.ValidationError):
        m.TraceSpan("", "t", "op", 0.0, 1.0)


def test_trace_span_end_before_start_raises():
    with pytest.raises(m.ValidationError):
        m.TraceSpan("s", "t", "op", 5.0, 1.0)


def test_trace_context_child():
    ctx = m.TraceContext("t", "s1")
    child = ctx.child("s2")
    assert child.parent_id == "s1" and child.trace_id == "t"


def test_trace_context_roundtrip():
    ctx = m.TraceContext("t", "s", "p", {"b": 1})
    assert m.TraceContext.from_dict(ctx.to_dict()) == ctx


def test_trace_context_requires_ids():
    with pytest.raises(m.ValidationError):
        m.TraceContext("", "s")


# --------------------------------------------------------------------------- #
# LogRecord
# --------------------------------------------------------------------------- #
def test_log_record_roundtrip():
    r = m.LogRecord(1.0, m.Severity.ERROR, "boom", "c", "r", "w", {"k": "v"}, {"type": "X"}, "a1")
    assert m.LogRecord.from_dict(r.to_dict()) == r


def test_log_record_json():
    r = m.LogRecord(1.0, m.Severity.INFO, "ok")
    assert json.loads(r.to_json())["severity"] == "INFO"


def test_log_record_coerces_severity():
    assert m.LogRecord(1.0, "WARNING", "x").severity is m.Severity.WARNING


def test_log_record_no_exception():
    r = m.LogRecord(1.0, m.Severity.INFO, "ok")
    assert r.to_dict()["exception"] is None


# --------------------------------------------------------------------------- #
# Incident / TimelineEvent / IncidentTimeline
# --------------------------------------------------------------------------- #
def test_timeline_event_roundtrip():
    e = m.TimelineEvent(1.0, m.IncidentStatus.OPEN, "msg")
    assert m.TimelineEvent.from_dict(e.to_dict()) == e


def test_incident_resolved_duration():
    inc = m.Incident("i", "t", m.IncidentSeverity.SEV1, m.IncidentStatus.RESOLVED, 0.0, 100.0)
    assert inc.is_resolved and inc.duration == 100.0


def test_incident_unresolved_duration_none():
    inc = m.Incident("i", "t", m.IncidentSeverity.SEV2, m.IncidentStatus.OPEN, 0.0)
    assert not inc.is_resolved and inc.duration is None


def test_incident_roundtrip():
    inc = m.Incident("i", "t", m.IncidentSeverity.SEV1, m.IncidentStatus.CLOSED, 0.0, 10.0,
                     "rc", ("db",), ("fix",))
    assert m.Incident.from_dict(inc.to_dict()) == inc


def test_incident_requires_id():
    with pytest.raises(m.ValidationError):
        m.Incident("", "t", m.IncidentSeverity.SEV1, m.IncidentStatus.OPEN, 0.0)


def test_incident_timeline_sorted():
    tl = m.IncidentTimeline("i", (m.TimelineEvent(5.0, m.IncidentStatus.OPEN),
                                  m.TimelineEvent(1.0, m.IncidentStatus.INVESTIGATING)))
    assert tl.events[0].timestamp == 1.0


def test_incident_timeline_roundtrip():
    tl = m.IncidentTimeline("i", (m.TimelineEvent(1.0, m.IncidentStatus.OPEN),))
    assert m.IncidentTimeline.from_dict(tl.to_dict()) == tl


# --------------------------------------------------------------------------- #
# ReliabilityScore / SLI / SLO / ErrorBudget
# --------------------------------------------------------------------------- #
def test_reliability_score_roundtrip():
    rs = m.ReliabilityScore(0.99, 1000.0, 50.0, 0.98, 0.02, 0.95, 0.05)
    assert m.ReliabilityScore.from_dict(rs.to_dict()) == rs


def test_sli_roundtrip():
    sli = m.SLI("a", m.SLIType.AVAILABILITY, 0.999, "ratio")
    assert m.SLI.from_dict(sli.to_dict()) == sli


def test_sli_requires_name():
    with pytest.raises(m.ValidationError):
        m.SLI("", m.SLIType.LATENCY, 1.0)


def test_slo_gte_met():
    slo = m.SLO("a", m.SLIType.AVAILABILITY, 0.99, 3600.0)
    assert slo.is_met(0.999) and not slo.is_met(0.98)


def test_slo_lte_met():
    slo = m.SLO("l", m.SLIType.LATENCY, 200.0, 3600.0, comparison="lte")
    assert slo.is_met(150.0) and not slo.is_met(250.0)


def test_slo_invalid_comparison():
    with pytest.raises(m.ValidationError):
        m.SLO("a", m.SLIType.AVAILABILITY, 0.99, 3600.0, comparison="eq")


def test_slo_roundtrip():
    slo = m.SLO("a", m.SLIType.ERROR_RATE, 0.01, 3600.0, "desc", "lte")
    assert m.SLO.from_dict(slo.to_dict()) == slo


def test_error_budget_exhausted():
    eb = m.ErrorBudget("a", 0.99, 0.0, 3600.0, 0.01, 0.01, 0.0, 1.0)
    assert eb.is_exhausted


def test_error_budget_not_exhausted():
    eb = m.ErrorBudget("a", 0.99, 0.995, 3600.0, 0.01, 0.0, 0.01, 0.0)
    assert not eb.is_exhausted


def test_error_budget_roundtrip():
    eb = m.ErrorBudget("a", 0.99, 0.995, 3600.0, 0.01, 0.005, 0.005, 0.5)
    assert m.ErrorBudget.from_dict(eb.to_dict()) == eb


# --------------------------------------------------------------------------- #
# CapacityForecast / ReadinessCheck
# --------------------------------------------------------------------------- #
def test_capacity_forecast_peak():
    cf = m.CapacityForecast(m.ResourceKind.CPU, 0.4, 3, (0.5, 0.6, 0.55), 0.05, 0.7)
    assert cf.peak == 0.6


def test_capacity_forecast_roundtrip():
    cf = m.CapacityForecast(m.ResourceKind.MEMORY, 0.4, 2, (0.5, 0.6), 0.05, 0.7, 1, "ratio")
    assert m.CapacityForecast.from_dict(cf.to_dict()) == cf


def test_capacity_forecast_coerces_resource():
    cf = m.CapacityForecast("CPU", 0.4, 1, (0.5,), 0.0, 0.6)
    assert cf.resource is m.ResourceKind.CPU


def test_readiness_check_clamps_score():
    assert m.ReadinessCheck("c", True, 5.0).score == 1.0


def test_readiness_check_roundtrip():
    rc = m.ReadinessCheck("c", True, 0.9, "ok", 2.0)
    assert m.ReadinessCheck.from_dict(rc.to_dict()) == rc


def test_readiness_check_requires_name():
    with pytest.raises(m.ValidationError):
        m.ReadinessCheck("", True, 1.0)


# --------------------------------------------------------------------------- #
# OperationsSnapshot / ProductionReport / RunbookEntry
# --------------------------------------------------------------------------- #
def test_operations_snapshot_roundtrip():
    rs = m.ReliabilityScore(0.99, 1000.0, 50.0, 0.98, 0.02, 0.95, 0.05)
    inc = m.Incident("i", "t", m.IncidentSeverity.SEV1, m.IncidentStatus.OPEN, 0.0)
    cf = m.CapacityForecast(m.ResourceKind.CPU, 0.4, 1, (0.5,), 0.05, 0.7)
    snap = m.OperationsSnapshot(0.0, {"r": 1}, rs, (inc,), {"c": True}, (cf,), 90.0)
    assert m.OperationsSnapshot.from_dict(snap.to_dict()) == snap


def test_operations_snapshot_minimal():
    snap = m.OperationsSnapshot(0.0)
    assert m.OperationsSnapshot.from_dict(snap.to_dict()) == snap


def test_production_report_counts():
    pr = m.ProductionReport(0.0, (m.ReadinessCheck("a", True, 1.0),
                                  m.ReadinessCheck("b", False, 0.0)), 50.0, m.ReadinessLevel.CONDITIONAL)
    assert pr.passed == 1 and pr.failed == 1


def test_production_report_check_lookup():
    pr = m.ProductionReport(0.0, (m.ReadinessCheck("a", True, 1.0),), 100.0, m.ReadinessLevel.READY)
    assert pr.check("a").passed and pr.check("missing") is None


def test_production_report_roundtrip():
    pr = m.ProductionReport(0.0, (m.ReadinessCheck("a", True, 1.0),), 100.0, m.ReadinessLevel.EXEMPLARY)
    assert m.ProductionReport.from_dict(json.loads(pr.to_json())).to_dict() == pr.to_dict()


def test_production_report_coerces_level():
    pr = m.ProductionReport(0.0, (), 0.0, "NOT_READY")
    assert pr.level is m.ReadinessLevel.NOT_READY


def test_runbook_entry_roundtrip():
    rb = m.RunbookEntry("rb", "title", m.IncidentSeverity.SEV1, ("s",), ("step",), ("svc",))
    assert m.RunbookEntry.from_dict(rb.to_dict()) == rb


def test_runbook_entry_requires_id():
    with pytest.raises(m.ValidationError):
        m.RunbookEntry("", "t", m.IncidentSeverity.SEV1)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
def test_severity_values():
    assert {s.value for s in m.Severity} == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_incident_severity_values():
    assert {s.value for s in m.IncidentSeverity} == {"SEV1", "SEV2", "SEV3", "SEV4"}


def test_sli_type_values():
    assert {s.value for s in m.SLIType} == {"AVAILABILITY", "LATENCY", "ERROR_RATE", "FRESHNESS"}


def test_readiness_level_values():
    assert {s.value for s in m.ReadinessLevel} == {"NOT_READY", "CONDITIONAL", "READY", "EXEMPLARY"}


def test_resource_kind_values():
    assert "MODEL_GROWTH" in {r.value for r in m.ResourceKind}