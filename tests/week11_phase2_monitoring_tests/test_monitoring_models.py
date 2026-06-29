"""Tests for monitoring domain models, enums and deterministic infrastructure."""

from __future__ import annotations

import json

import pytest

from monitoring.monitoring_models import (
    AlertLevel, AlertPolicy, AlertType, Comparison, ConceptDriftResult,
    DashboardSnapshot, DataDriftResult, DeterministicIdGenerator, DistributionSnapshot,
    DriftMethod, DriftSeverity, FeatureStatistics, FeatureType, FixedClock, HealthLevel,
    HealthScore, LogicalClock, ModelHealthStatus, MonitoringAlert, MonitoringConfiguration,
    MonitoringReport, MonitoringStatistics, PredictionDriftResult, PredictionStatistics,
    QualityIssue, QualityIssueType, QualityMetrics, SequentialIdGenerator, ValidationError,
    freeze_mapping, thaw_mapping,
)


# --------------------------------------------------------------------------- #
# Instance factories
# --------------------------------------------------------------------------- #
def make_feature_statistics():
    return FeatureStatistics(feature_name="x", mean=1.0, std=0.5, minimum=0.0, maximum=2.0,
                             median=1.0, q25=0.5, q75=1.5, count=100, histogram=[1, 2, 3],
                             bin_edges=[0, 1, 2, 3], unique_count=50)


def make_categorical_statistics():
    return FeatureStatistics(feature_name="c", feature_type=FeatureType.CATEGORICAL,
                             count=10, unique_count=2, categories={"a": 0.5, "b": 0.5})


def make_distribution_snapshot():
    return DistributionSnapshot(snapshot_id="snap-1", feature_stats=(make_feature_statistics(),),
                                sample_size=100, created_at="2024-01-01T00:00:00+00:00",
                                label_distribution={"pos": 0.6, "neg": 0.4})


def make_data_drift_result():
    return DataDriftResult(feature_name="x", method=DriftMethod.PSI, drift_score=0.35,
                           threshold=0.2, drifted=True, severity=DriftSeverity.HIGH,
                           reference_size=1000, current_size=1000, p_value=0.01,
                           details={"confidence": 0.9})


def make_concept_drift_result():
    return ConceptDriftResult(detected=True, method="rolling_performance", drift_score=0.12,
                              severity=DriftSeverity.MODERATE, reference_metric=0.92,
                              current_metric=0.80, delta=-0.12, window_size=50,
                              trend_slope=-0.01, threshold=0.05, details={"series_length": 120})


def make_prediction_statistics():
    return PredictionStatistics(count=1000, mean=0.5, std=0.28, minimum=0.0, maximum=1.0,
                                variance=0.08, q05=0.05, q50=0.5, q95=0.95, positive_rate=0.5,
                                confidence_mean=0.7, confidence_std=0.1, latency_p50_ms=100.0,
                                latency_p95_ms=180.0, latency_p99_ms=220.0, throughput_per_s=50.0,
                                error_rate=0.02, success_rate=0.98, inference_time_ms=12.0)


def make_prediction_drift_result():
    return PredictionDriftResult(drift_score=0.2, method=DriftMethod.JS_DISTANCE,
                                 severity=DriftSeverity.MODERATE, drifted=True, threshold=0.1,
                                 reference_size=1000, current_size=1000, details={"bins": 10})


def make_health_score():
    return HealthScore(overall=0.82, level=HealthLevel.HEALTHY,
                       components={"drift": 0.8, "availability": 0.99},
                       weights={"drift": 0.5, "availability": 0.5},
                       created_at="2024-01-01T00:00:00+00:00")


def make_model_health_status():
    return ModelHealthStatus(model_id="mdl", level=HealthLevel.HEALTHY, score=0.82,
                             created_at="2024-01-01T00:00:00+00:00", summary="ok",
                             health=make_health_score())


def make_monitoring_alert():
    return MonitoringAlert(alert_id="alert-1", level=AlertLevel.HIGH, alert_type=AlertType.THRESHOLD,
                           title="High drift", message="psi=0.4", metric="overall_drift_score",
                           value=0.4, threshold=0.2, entity="mdl", fingerprint="abc123",
                           count=1, created_at="2024-01-01T00:00:00+00:00", tags=["drift"])


def make_quality_issue():
    return QualityIssue(issue_type=QualityIssueType.OUTLIER, feature_name="x",
                        severity=DriftSeverity.MODERATE, count=5, rate=0.05, message="outliers",
                        details={"z": 3.0})


def make_quality_metrics():
    return QualityMetrics(total_records=1000, completeness=0.97, null_rate=0.03,
                          duplicate_rate=0.01, outlier_rate=0.02, validity_rate=0.99,
                          consistency_score=0.99, schema_violation_count=1, freshness_seconds=120.0,
                          issue_count=1, issues=(make_quality_issue(),),
                          created_at="2024-01-01T00:00:00+00:00")


def make_monitoring_report():
    return MonitoringReport(report_id="report-1", created_at="2024-01-01T00:00:00+00:00",
                            model_id="mdl", drift_results=(make_data_drift_result(),),
                            concept_drift=make_concept_drift_result(),
                            prediction_drift=make_prediction_drift_result(),
                            health=make_health_score(), quality=make_quality_metrics(),
                            alerts=(make_monitoring_alert(),), overall_drift_score=0.35,
                            summary="summary")


def make_dashboard_snapshot():
    return DashboardSnapshot(snapshot_id="snap-1", model_id="mdl",
                             created_at="2024-01-01T00:00:00+00:00", health=make_health_score(),
                             drift_summary={"overall_drift_score": 0.35, "drifted_count": 1},
                             top_drifted_features=(("x", 0.35), ("y", 0.1)),
                             prediction_trends={"mean": 0.5}, latency_trends={"p95": 180.0},
                             data_quality=make_quality_metrics(),
                             active_alerts=(make_monitoring_alert(),), model_status="HEALTHY")


def make_monitoring_statistics():
    return MonitoringStatistics(total_reports=3, total_alerts=5, alerts_by_level={"HIGH": 2},
                                drift_checks=12, drifted_features=4, average_health_score=0.8,
                                generated_at="2024-01-01T00:00:00+00:00")


def make_monitoring_configuration():
    return MonitoringConfiguration()


def make_alert_policy():
    return AlertPolicy(name="psi", metric="overall_drift_score", level=AlertLevel.HIGH,
                       comparison=Comparison.GT, threshold=0.2, alert_type=AlertType.THRESHOLD)


ALL_FACTORIES = [
    make_feature_statistics, make_categorical_statistics, make_distribution_snapshot,
    make_data_drift_result, make_concept_drift_result, make_prediction_statistics,
    make_prediction_drift_result, make_health_score, make_model_health_status,
    make_monitoring_alert, make_quality_issue, make_quality_metrics, make_monitoring_report,
    make_dashboard_snapshot, make_monitoring_statistics, make_monitoring_configuration,
    make_alert_policy,
]


# --------------------------------------------------------------------------- #
# Round-trip, JSON, hashability, immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_to_from_dict_roundtrip(factory):
    obj = factory()
    restored = type(obj).from_dict(obj.to_dict())
    assert restored == obj


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_json_serializable(factory):
    obj = factory()
    text = json.dumps(obj.to_dict(), sort_keys=True)
    restored = type(obj).from_dict(json.loads(text))
    assert restored == obj


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_hashable(factory):
    obj = factory()
    assert isinstance(hash(obj), int)
    assert len({obj, factory()}) == 1


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_json_dumps_stable(factory):
    obj = factory()
    assert json.dumps(obj.to_dict(), sort_keys=True) == json.dumps(obj.to_dict(), sort_keys=True)


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_immutable(factory):
    obj = factory()
    with pytest.raises(Exception):
        object.__setattr__  # ensure name resolves
        setattr(obj, "nonexistent_attr_xyz", 1)


# --------------------------------------------------------------------------- #
# Enum coverage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("member", list(DriftSeverity))
def test_drift_severity_roundtrip(member):
    assert DriftSeverity(member.value) is member


@pytest.mark.parametrize("member", list(AlertLevel))
def test_alert_level_roundtrip(member):
    assert AlertLevel(member.value) is member


@pytest.mark.parametrize("member", list(HealthLevel))
def test_health_level_roundtrip(member):
    assert HealthLevel(member.value) is member


@pytest.mark.parametrize("member", list(DriftMethod))
def test_drift_method_roundtrip(member):
    assert DriftMethod(member.value) is member


@pytest.mark.parametrize("member", list(AlertType))
def test_alert_type_roundtrip(member):
    assert AlertType(member.value) is member


@pytest.mark.parametrize("member", list(QualityIssueType))
def test_quality_issue_type_roundtrip(member):
    assert QualityIssueType(member.value) is member


@pytest.mark.parametrize("member", list(FeatureType))
def test_feature_type_roundtrip(member):
    assert FeatureType(member.value) is member


@pytest.mark.parametrize("member", list(Comparison))
def test_comparison_roundtrip(member):
    assert Comparison(member.value) is member


# --------------------------------------------------------------------------- #
# AlertPolicy.evaluate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("comparison,threshold,value,expected", [
    (Comparison.GT, 0.2, 0.3, True),
    (Comparison.GT, 0.2, 0.1, False),
    (Comparison.GT, 0.2, 0.2, False),
    (Comparison.GE, 0.2, 0.2, True),
    (Comparison.GE, 0.2, 0.19, False),
    (Comparison.LT, 0.5, 0.4, True),
    (Comparison.LT, 0.5, 0.5, False),
    (Comparison.LE, 0.5, 0.5, True),
    (Comparison.LE, 0.5, 0.51, False),
])
def test_alert_policy_evaluate(comparison, threshold, value, expected):
    policy = AlertPolicy(name="p", metric="m", level=AlertLevel.WARNING,
                         comparison=comparison, threshold=threshold)
    assert policy.evaluate(value) is expected


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_feature_statistics_requires_name():
    with pytest.raises(ValidationError):
        FeatureStatistics(feature_name="")


def test_distribution_snapshot_requires_id():
    with pytest.raises(ValidationError):
        DistributionSnapshot(snapshot_id="")


def test_alert_requires_title():
    with pytest.raises(ValidationError):
        MonitoringAlert(alert_id="a", level=AlertLevel.INFO, alert_type=AlertType.THRESHOLD, title="")


def test_alert_requires_id():
    with pytest.raises(ValidationError):
        MonitoringAlert(alert_id="", level=AlertLevel.INFO, alert_type=AlertType.THRESHOLD, title="t")


def test_config_rejects_unordered_health_thresholds():
    with pytest.raises(ValidationError):
        MonitoringConfiguration(health_warning=0.9, health_healthy=0.5, health_excellent=0.8)


def test_config_rejects_nonpositive_bins():
    with pytest.raises(ValidationError):
        MonitoringConfiguration(num_bins=0)


def test_config_rejects_nonpositive_window():
    with pytest.raises(ValidationError):
        MonitoringConfiguration(window_size=0)


def test_config_defaults():
    c = MonitoringConfiguration()
    assert c.psi_threshold == 0.2
    assert c.num_bins == 10
    assert c.seed == 7


def test_report_requires_id():
    with pytest.raises(ValidationError):
        MonitoringReport(report_id="")


def test_alert_policy_requires_metric():
    with pytest.raises(ValidationError):
        AlertPolicy(name="p", metric="", level=AlertLevel.INFO,
                    comparison=Comparison.GT, threshold=0.1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_freeze_mapping_sorted():
    frozen = freeze_mapping({"b": 1, "a": 2})
    assert frozen == (("a", 2), ("b", 1))


def test_freeze_mapping_none():
    assert freeze_mapping(None) == ()


def test_thaw_mapping_roundtrip():
    data = {"a": 1.0, "b": 2.0}
    assert thaw_mapping(freeze_mapping(data)) == data


def test_freeze_mapping_idempotent_on_tuple():
    frozen = freeze_mapping({"x": 1})
    assert freeze_mapping(frozen) == frozen


def test_feature_statistics_categories_frozen():
    stat = make_categorical_statistics()
    assert isinstance(stat.categories, tuple)
    assert dict(stat.categories)["a"] == 0.5


def test_feature_statistics_histogram_is_tuple():
    stat = make_feature_statistics()
    assert isinstance(stat.histogram, tuple)
    assert isinstance(stat.bin_edges, tuple)


def test_distribution_snapshot_feature_lookup():
    snap = make_distribution_snapshot()
    assert snap.feature("x") is not None
    assert snap.feature("missing") is None


def test_health_score_components_accessible():
    hs = make_health_score()
    assert dict(hs.components)["drift"] == 0.8


def test_dashboard_top_features_typed():
    snap = make_dashboard_snapshot()
    assert snap.top_drifted_features[0] == ("x", 0.35)


# --------------------------------------------------------------------------- #
# Deterministic infrastructure
# --------------------------------------------------------------------------- #
def test_sequential_id_generator_increments():
    gen = SequentialIdGenerator()
    assert gen.generate("alert") == "alert-000001"
    assert gen.generate("alert") == "alert-000002"


def test_sequential_id_generator_per_prefix():
    gen = SequentialIdGenerator()
    assert gen.generate("a") == "a-000001"
    assert gen.generate("b") == "b-000001"


def test_deterministic_id_generator_reproducible():
    g1 = DeterministicIdGenerator(seed="s")
    g2 = DeterministicIdGenerator(seed="s")
    assert g1.generate("x") == g2.generate("x")


def test_deterministic_id_generator_seed_sensitive():
    g1 = DeterministicIdGenerator(seed="a")
    g2 = DeterministicIdGenerator(seed="b")
    assert g1.generate("x") != g2.generate("x")


def test_fixed_clock_constant():
    clock = FixedClock("2024-05-01T00:00:00+00:00")
    assert clock.now() == clock.now() == "2024-05-01T00:00:00+00:00"


def test_logical_clock_monotonic():
    clock = LogicalClock()
    first, second = clock.now(), clock.now()
    assert first < second


def test_logical_clock_rejects_bad_step():
    with pytest.raises(ValidationError):
        LogicalClock(step_seconds=0)


def test_sequential_id_generator_rejects_bad_width():
    with pytest.raises(ValidationError):
        SequentialIdGenerator(width=0)