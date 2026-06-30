"""Tests for the operations dashboard."""

from __future__ import annotations

import json

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.operations_dashboard import OperationsDashboard, create_operations_dashboard  # noqa: E402
from observability.observability_models import (  # noqa: E402
    Clock, ReliabilityScore, Incident, IncidentSeverity, IncidentStatus,
    CapacityForecast, ResourceKind,
)


def dashboard():
    return OperationsDashboard(clock=Clock())


def rel(score=0.97, risk=0.05):
    return ReliabilityScore(0.999, 1000.0, 50.0, 0.99, 0.01, score, risk)


def test_build_snapshot_basic():
    snap = dashboard().build_snapshot(readiness_score=90.0)
    assert snap.readiness_score == 90.0


def test_build_snapshot_with_reliability():
    snap = dashboard().build_snapshot(reliability=rel())
    assert snap.reliability.score == 0.97


def test_build_snapshot_metrics_summary():
    snap = dashboard().build_snapshot(metrics_summary={"rps": 120})
    assert dict(snap.metrics_summary)["rps"] == 120


def test_build_snapshot_incidents():
    inc = Incident("i", "x", IncidentSeverity.SEV1, IncidentStatus.OPEN, 0.0)
    snap = dashboard().build_snapshot(active_incidents=[inc])
    assert len(snap.active_incidents) == 1


def test_build_snapshot_capacity():
    cap = CapacityForecast(ResourceKind.CPU, 0.5, 3, (0.55, 0.6, 0.65), 0.05, 0.78)
    snap = dashboard().build_snapshot(capacity=[cap])
    assert len(snap.capacity) == 1


def test_executive_summary_contains_readiness():
    snap = dashboard().build_snapshot(readiness_score=92.0)
    assert "92.0/100" in dashboard().executive_summary(snap)


def test_executive_summary_reliability_line():
    d = dashboard()
    snap = d.build_snapshot(reliability=rel())
    assert "Reliability score" in d.executive_summary(snap)


def test_executive_summary_no_incidents():
    d = dashboard()
    snap = d.build_snapshot(readiness_score=98.0)
    assert "Active incidents: none" in d.executive_summary(snap)


def test_executive_summary_with_incidents():
    d = dashboard()
    inc = Incident("i", "x", IncidentSeverity.SEV1, IncidentStatus.OPEN, 0.0)
    snap = d.build_snapshot(active_incidents=[inc])
    assert "most severe SEV1" in d.executive_summary(snap)


def test_recommendation_sev1():
    d = dashboard()
    inc = Incident("i", "x", IncidentSeverity.SEV1, IncidentStatus.OPEN, 0.0)
    snap = d.build_snapshot(active_incidents=[inc], readiness_score=99.0)
    assert "incident response" in d.executive_summary(snap)


def test_recommendation_high_risk():
    d = dashboard()
    snap = d.build_snapshot(reliability=rel(risk=0.4), readiness_score=99.0)
    assert "operational risk" in d.executive_summary(snap)


def test_recommendation_low_readiness():
    d = dashboard()
    snap = d.build_snapshot(reliability=rel(risk=0.01), readiness_score=70.0)
    assert "readiness gaps" in d.executive_summary(snap)


def test_recommendation_healthy():
    d = dashboard()
    snap = d.build_snapshot(reliability=rel(risk=0.01), readiness_score=98.0)
    assert "no action required" in d.executive_summary(snap)


def test_recommendation_priority_sev1_over_risk():
    d = dashboard()
    inc = Incident("i", "x", IncidentSeverity.SEV1, IncidentStatus.OPEN, 0.0)
    snap = d.build_snapshot(active_incidents=[inc], reliability=rel(risk=0.4), readiness_score=50.0)
    assert "incident response" in d.executive_summary(snap)


def test_summary_slo_compliance():
    d = dashboard()
    snap = d.build_snapshot(slo_compliance={"compliance_rate": 0.98}, readiness_score=99.0)
    assert "98.0%" in d.executive_summary(snap)


def test_summary_capacity_warning():
    d = dashboard()
    cap = CapacityForecast(ResourceKind.CPU, 0.5, 3, (0.6, 0.7, 0.8), 0.1, 0.96, 2)
    snap = d.build_snapshot(capacity=[cap], readiness_score=99.0)
    assert "Capacity warnings" in d.executive_summary(snap)


def test_summary_capacity_ok():
    d = dashboard()
    cap = CapacityForecast(ResourceKind.CPU, 0.5, 3, (0.55, 0.6, 0.65), 0.05, 0.78)
    snap = d.build_snapshot(capacity=[cap], readiness_score=99.0)
    assert "within projected limits" in d.executive_summary(snap)


def test_render_includes_summary():
    d = dashboard()
    snap = d.build_snapshot(readiness_score=90.0)
    assert "executive_summary" in d.render(snap)


def test_render_json_serializable():
    d = dashboard()
    snap = d.build_snapshot(reliability=rel(), readiness_score=90.0)
    assert json.dumps(d.render(snap))


def test_statistics_count():
    d = dashboard()
    d.build_snapshot(readiness_score=90.0)
    d.build_snapshot(readiness_score=91.0)
    assert d.statistics()["snapshots"] == 2


def test_snapshot_roundtrip():
    from observability.observability_models import OperationsSnapshot
    snap = dashboard().build_snapshot(reliability=rel(), readiness_score=90.0)
    assert OperationsSnapshot.from_dict(snap.to_dict()) == snap


def test_determinism():
    a = OperationsDashboard(clock=Clock()).build_snapshot(reliability=rel(), readiness_score=90.0)
    b = OperationsDashboard(clock=Clock()).build_snapshot(reliability=rel(), readiness_score=90.0)
    assert a.to_dict() == b.to_dict()


def test_factory():
    assert isinstance(create_operations_dashboard(), OperationsDashboard)


def test_empty_snapshot_summary():
    d = dashboard()
    snap = d.build_snapshot()
    summary = d.executive_summary(snap)
    assert "Executive Operations Summary" in summary


def test_multiple_incidents_most_severe():
    d = dashboard()
    incs = [
        Incident("i1", "x", IncidentSeverity.SEV3, IncidentStatus.OPEN, 0.0),
        Incident("i2", "y", IncidentSeverity.SEV1, IncidentStatus.OPEN, 0.0),
    ]
    snap = d.build_snapshot(active_incidents=incs)
    assert "most severe SEV1" in d.executive_summary(snap)