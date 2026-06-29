"""Tests for the alert engine (Observer pattern) and monitoring dashboard."""

from __future__ import annotations

import json

import pytest

from monitoring.monitoring_dashboard import (
    AlertEngine, MonitoringDashboard, create_alert_engine, create_monitoring_dashboard,
    default_alert_policies,
)
from monitoring.monitoring_models import (
    AlertLevel, AlertPolicy, AlertType, Comparison, ConceptDriftResult, DataDriftResult,
    DriftMethod, DriftSeverity, HealthLevel, HealthScore, MonitoringConfiguration,
    PredictionDriftResult, PredictionStatistics, QualityMetrics, ValidationError,
)


def engine():
    return create_alert_engine()


def dashboard():
    return create_monitoring_dashboard()


def drift_results():
    return [
        DataDriftResult(feature_name="amount", method=DriftMethod.PSI, drift_score=0.8,
                        threshold=0.2, drifted=True, severity=DriftSeverity.CRITICAL),
        DataDriftResult(feature_name="region", method=DriftMethod.CATEGORICAL, drift_score=0.4,
                        threshold=0.2, drifted=True, severity=DriftSeverity.HIGH),
        DataDriftResult(feature_name="age", method=DriftMethod.PSI, drift_score=0.02,
                        threshold=0.2, drifted=False, severity=DriftSeverity.NONE),
    ]


def health(level=HealthLevel.WARNING, overall=0.45):
    return HealthScore(overall=overall, level=level)


def prediction_stats():
    return PredictionStatistics(count=1000, mean=0.6, variance=0.05, positive_rate=0.6,
                                error_rate=0.08, latency_p50_ms=100.0, latency_p95_ms=300.0,
                                latency_p99_ms=400.0)


# --------------------------------------------------------------------------- #
# Alert engine: threshold
# --------------------------------------------------------------------------- #
def test_threshold_alert_fires():
    alerts = engine().evaluate_metrics({"overall_drift_score": 0.5}, entity="mdl")
    assert any(a.metric == "overall_drift_score" for a in alerts)


def test_no_alert_below_threshold():
    alerts = engine().evaluate_metrics({"overall_drift_score": 0.01}, entity="mdl")
    assert not any(a.metric == "overall_drift_score" for a in alerts)


def test_health_critical_beats_warning():
    alerts = engine().evaluate_metrics({"health_score": 0.3}, entity="mdl")
    health_alerts = [a for a in alerts if a.metric == "health_score"]
    assert len(health_alerts) == 1
    assert health_alerts[0].level is AlertLevel.CRITICAL


def test_health_warning_band():
    alerts = engine().evaluate_metrics({"health_score": 0.6}, entity="mdl")
    health_alerts = [a for a in alerts if a.metric == "health_score"]
    assert health_alerts[0].level is AlertLevel.WARNING


def test_null_rate_alert():
    alerts = engine().evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    assert any(a.metric == "null_rate" for a in alerts)


def test_error_rate_alert():
    alerts = engine().evaluate_metrics({"error_rate": 0.5}, entity="mdl")
    assert any(a.metric == "error_rate" for a in alerts)


def test_latency_alert():
    alerts = engine().evaluate_metrics({"latency_p95_ms": 999.0}, entity="mdl")
    assert any(a.metric == "latency_p95_ms" for a in alerts)


def test_alerts_sorted_by_metric():
    alerts = engine().evaluate_metrics(
        {"overall_drift_score": 0.5, "null_rate": 0.5, "error_rate": 0.5}, entity="mdl")
    metrics = [a.metric for a in alerts]
    assert metrics == sorted(metrics)


def test_no_metrics_no_alerts():
    assert engine().evaluate_metrics({}, entity="mdl") == []


# --------------------------------------------------------------------------- #
# Deduplication & repeated alerts
# --------------------------------------------------------------------------- #
def test_repeated_alert_marked():
    eng = engine()
    eng.evaluate_metrics({"overall_drift_score": 0.5}, entity="mdl")
    second = eng.evaluate_metrics({"overall_drift_score": 0.5}, entity="mdl")
    assert all(a.alert_type is AlertType.REPEATED for a in second)


def test_repeated_count_increments():
    eng = engine()
    eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    third = eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    assert third[0].count == 3


def test_fingerprint_stable_for_same_metric():
    eng = engine()
    a = eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")[0]
    b = eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")[0]
    assert a.fingerprint == b.fingerprint


def test_fingerprint_differs_by_entity():
    eng = engine()
    a = eng.evaluate_metrics({"null_rate": 0.5}, entity="m1")[0]
    b = eng.evaluate_metrics({"null_rate": 0.5}, entity="m2")[0]
    assert a.fingerprint != b.fingerprint


def test_reset_clears_repeat_state():
    eng = engine()
    eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    eng.reset()
    again = eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    assert again[0].alert_type is AlertType.THRESHOLD


# --------------------------------------------------------------------------- #
# Observer pattern
# --------------------------------------------------------------------------- #
def test_observer_callable_notified():
    eng = engine()
    received = []
    eng.subscribe(lambda a: received.append(a))
    eng.evaluate_metrics({"overall_drift_score": 0.5}, entity="mdl")
    assert len(received) == 1


def test_observer_object_notified():
    class Collector:
        def __init__(self):
            self.alerts = []

        def on_alert(self, alert):
            self.alerts.append(alert)

    eng = engine()
    collector = Collector()
    eng.subscribe(collector)
    eng.evaluate_metrics({"null_rate": 0.5}, entity="mdl")
    assert len(collector.alerts) == 1


def test_multiple_observers():
    eng = engine()
    a, b = [], []
    eng.subscribe(lambda x: a.append(x))
    eng.subscribe(lambda x: b.append(x))
    eng.evaluate_metrics({"error_rate": 0.5}, entity="mdl")
    assert len(a) == 1 and len(b) == 1


def test_unsubscribe_stops_notifications():
    eng = engine()
    received = []
    observer = lambda x: received.append(x)
    eng.subscribe(observer)
    eng.unsubscribe(observer)
    eng.evaluate_metrics({"error_rate": 0.5}, entity="mdl")
    assert received == []


def test_subscribe_idempotent():
    eng = engine()
    received = []
    observer = lambda x: received.append(x)
    eng.subscribe(observer)
    eng.subscribe(observer)
    eng.evaluate_metrics({"error_rate": 0.5}, entity="mdl")
    assert len(received) == 1


# --------------------------------------------------------------------------- #
# Composite & trend
# --------------------------------------------------------------------------- #
def test_composite_alert_type():
    alert = engine().composite_alert("deg", AlertLevel.CRITICAL, "msg", components=["a", "b"])
    assert alert.alert_type is AlertType.COMPOSITE


def test_trend_policy_uses_slopes():
    eng = create_alert_engine(policies=[
        AlertPolicy(name="lat_trend", metric="latency_p95_ms", level=AlertLevel.WARNING,
                    comparison=Comparison.GT, threshold=1.0, alert_type=AlertType.TREND),
    ])
    alerts = eng.evaluate_metrics({}, entity="mdl", slopes={"latency_p95_ms": 5.0})
    assert len(alerts) == 1


def test_disabled_policy_skipped():
    eng = create_alert_engine(policies=[
        AlertPolicy(name="drift", metric="overall_drift_score", level=AlertLevel.HIGH,
                    comparison=Comparison.GT, threshold=0.2, enabled=False),
    ])
    assert eng.evaluate_metrics({"overall_drift_score": 0.9}, entity="mdl") == []


def test_default_policies_count():
    assert len(default_alert_policies(MonitoringConfiguration())) >= 8


def test_add_policy():
    eng = create_alert_engine(policies=[])
    eng.add_policy(AlertPolicy(name="x", metric="m", level=AlertLevel.INFO,
                               comparison=Comparison.GT, threshold=0.0))
    assert len(eng.policies) == 1


# --------------------------------------------------------------------------- #
# Dashboard: reports
# --------------------------------------------------------------------------- #
def test_build_report_overall_score():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results())
    assert report.overall_drift_score > 0.0


def test_build_report_generates_alerts():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results(), health=health())
    assert len(report.alerts) >= 1


def test_build_report_composite_on_drift_and_health():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results(), health=health())
    assert any(a.alert_type is AlertType.COMPOSITE for a in report.alerts)


def test_build_report_no_composite_when_healthy():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results(),
                                      health=health(HealthLevel.EXCELLENT, 0.95))
    assert not any(a.alert_type is AlertType.COMPOSITE for a in report.alerts)


def test_build_report_requires_model_id():
    with pytest.raises(ValidationError):
        dashboard().build_report(model_id="", drift_results=drift_results())


def test_build_report_alerts_sorted_by_severity():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results(),
                                      health=health(), quality=QualityMetrics(null_rate=0.5))
    levels = [a.level for a in report.alerts]
    order = {AlertLevel.CRITICAL: 0, AlertLevel.HIGH: 1, AlertLevel.WARNING: 2, AlertLevel.INFO: 3}
    assert levels == sorted(levels, key=lambda l: order[l])


def test_build_report_serializable():
    report = dashboard().build_report(model_id="mdl", drift_results=drift_results(), health=health())
    assert json.dumps(report.to_dict())


# --------------------------------------------------------------------------- #
# Dashboard: snapshots
# --------------------------------------------------------------------------- #
def test_snapshot_top_drifted_feature():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results())
    assert snap.top_drifted_features[0][0] == "amount"


def test_snapshot_model_status_from_health():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results(),
                                      health=health(HealthLevel.HEALTHY, 0.8))
    assert snap.model_status == "HEALTHY"


def test_snapshot_status_unknown_without_health():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results())
    assert snap.model_status == "UNKNOWN"


def test_snapshot_prediction_trends():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results(),
                                      prediction_stats=prediction_stats())
    assert dict(snap.prediction_trends)["mean"] == 0.6


def test_snapshot_latency_trends():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results(),
                                      prediction_stats=prediction_stats())
    assert dict(snap.latency_trends)["p95"] == 300.0


def test_snapshot_drift_summary():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results())
    summary = dict(snap.drift_summary)
    assert summary["drifted_count"] == 2
    assert summary["feature_count"] == 3


def test_snapshot_requires_model_id():
    with pytest.raises(ValidationError):
        dashboard().build_snapshot(model_id="", drift_results=drift_results())


def test_snapshot_carries_alerts():
    dash = dashboard()
    report = dash.build_report(model_id="mdl", drift_results=drift_results(), health=health())
    snap = dash.build_snapshot(model_id="mdl", drift_results=drift_results(),
                               health=health(), alerts=report.alerts)
    assert len(snap.active_alerts) == len(report.alerts)


def test_snapshot_serializable():
    snap = dashboard().build_snapshot(model_id="mdl", drift_results=drift_results(),
                                      health=health(), prediction_stats=prediction_stats())
    assert json.dumps(snap.to_dict())


# --------------------------------------------------------------------------- #
# Executive summary & statistics
# --------------------------------------------------------------------------- #
def test_executive_summary_contains_model():
    dash = dashboard()
    report = dash.build_report(model_id="mdl", drift_results=drift_results(), health=health())
    snap = dash.build_snapshot(model_id="mdl", drift_results=drift_results(), health=health())
    summary = dash.executive_summary(report, snap)
    assert "mdl" in summary
    assert "Executive Monitoring Summary" in summary


def test_executive_summary_critical_recommendation():
    dash = dashboard()
    report = dash.build_report(model_id="mdl", drift_results=drift_results(), health=health())
    snap = dash.build_snapshot(model_id="mdl", drift_results=drift_results(), health=health())
    summary = dash.executive_summary(report, snap)
    assert "rollback" in summary.lower() or "investigation" in summary.lower()


def test_executive_summary_healthy_recommendation():
    dash = dashboard()
    healthy = health(HealthLevel.EXCELLENT, 0.95)
    clean = [DataDriftResult(feature_name="x", method=DriftMethod.PSI, drift_score=0.01,
                             threshold=0.2, drifted=False, severity=DriftSeverity.NONE)]
    report = dash.build_report(model_id="mdl", drift_results=clean, health=healthy)
    snap = dash.build_snapshot(model_id="mdl", drift_results=clean, health=healthy)
    summary = dash.executive_summary(report, snap)
    assert "no action" in summary.lower()


def test_statistics_counts_reports():
    dash = dashboard()
    dash.build_report(model_id="mdl", drift_results=drift_results(), health=health())
    stats = dash.statistics()
    assert stats.total_reports == 1


def test_dashboard_deterministic():
    r1 = create_monitoring_dashboard().build_report(model_id="mdl", drift_results=drift_results(), health=health())
    r2 = create_monitoring_dashboard().build_report(model_id="mdl", drift_results=drift_results(), health=health())
    assert r1.to_dict() == r2.to_dict()


def test_factory_with_config():
    cfg = MonitoringConfiguration(drift_threshold=0.4)
    dash = create_monitoring_dashboard(config=cfg)
    assert dash.config.drift_threshold == 0.4