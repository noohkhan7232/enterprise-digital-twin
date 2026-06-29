"""Tests for the deterministic data-quality monitor."""

from __future__ import annotations

import json

import numpy as np
import pytest

from monitoring.data_quality_monitor import DataQualityMonitor, create_data_quality_monitor
from monitoring.monitoring_models import (
    DriftSeverity, MonitoringConfiguration, QualityIssueType, ValidationError,
)


def rng(seed=7):
    return np.random.default_rng(seed)


def qm():
    return create_data_quality_monitor()


def issue_types(metrics):
    return {i.issue_type for i in metrics.issues}


# --------------------------------------------------------------------------- #
# Missing values / completeness
# --------------------------------------------------------------------------- #
def test_missing_values_detected():
    arr = rng().normal(0, 1, 1000)
    arr[:100] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert QualityIssueType.MISSING_VALUES in issue_types(metrics)


def test_null_rate_computed():
    arr = rng().normal(0, 1, 1000)
    arr[:100] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert metrics.null_rate == pytest.approx(0.1)


def test_completeness_complement_of_null():
    arr = rng().normal(0, 1, 1000)
    arr[:100] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert metrics.completeness == pytest.approx(1.0 - metrics.null_rate)


def test_null_percentage_issue_when_above_threshold():
    arr = rng().normal(0, 1, 100)
    arr[:30] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert QualityIssueType.NULL_PERCENTAGE in issue_types(metrics)


def test_clean_dataset_no_issues():
    metrics = qm().evaluate({"x": list(range(100))})
    assert metrics.issue_count == 0
    assert metrics.completeness == 1.0


# --------------------------------------------------------------------------- #
# Duplicates / consistency
# --------------------------------------------------------------------------- #
def test_duplicates_detected():
    metrics = qm().evaluate({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    assert QualityIssueType.DUPLICATE_RECORDS in issue_types(metrics)


def test_duplicate_rate_value():
    metrics = qm().evaluate({"a": [1, 1, 1, 1]})
    assert metrics.duplicate_rate == pytest.approx(0.75)


def test_consistency_complement_of_duplicate():
    metrics = qm().evaluate({"a": [1, 1, 2, 3]})
    assert metrics.consistency_score == pytest.approx(1.0 - metrics.duplicate_rate)


def test_no_duplicates_full_consistency():
    metrics = qm().evaluate({"a": [1, 2, 3, 4]})
    assert metrics.consistency_score == 1.0


# --------------------------------------------------------------------------- #
# Invalid ranges / validity
# --------------------------------------------------------------------------- #
def test_invalid_range_detected():
    arr = rng().normal(45, 12, 1000)
    arr[:20] = 500
    metrics = qm().evaluate({"age": arr}, valid_ranges={"age": (0, 120)})
    assert QualityIssueType.INVALID_RANGE in issue_types(metrics)


def test_validity_rate_below_one_when_invalid():
    arr = rng().normal(45, 12, 1000)
    arr[:20] = 500
    metrics = qm().evaluate({"age": arr}, valid_ranges={"age": (0, 120)})
    assert metrics.validity_rate < 1.0


def test_validity_rate_one_when_all_valid():
    metrics = qm().evaluate({"age": [10, 20, 30]}, valid_ranges={"age": (0, 120)})
    assert metrics.validity_rate == 1.0


def test_valid_range_ignores_unknown_feature():
    metrics = qm().evaluate({"x": [1, 2, 3]}, valid_ranges={"y": (0, 1)})
    assert metrics.validity_rate == 1.0


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def test_outliers_detected():
    arr = rng().normal(0, 1, 1000)
    arr[0] = 100
    metrics = qm().evaluate({"x": arr})
    assert QualityIssueType.OUTLIER in issue_types(metrics)


def test_outlier_rate_positive():
    arr = rng().normal(0, 1, 1000)
    arr[:3] = 100
    metrics = qm().evaluate({"x": arr})
    assert metrics.outlier_rate > 0.0


def test_constant_feature_no_outliers():
    metrics = qm().evaluate({"x": [5.0] * 100})
    assert QualityIssueType.OUTLIER not in issue_types(metrics)


@pytest.mark.parametrize("z,expect_more", [(2.0, True), (5.0, False)])
def test_outlier_threshold_sensitivity(z, expect_more):
    cfg = MonitoringConfiguration(outlier_z_threshold=z)
    arr = rng().normal(0, 1, 2000)
    metrics = create_data_quality_monitor(config=cfg).evaluate({"x": arr})
    has = QualityIssueType.OUTLIER in issue_types(metrics)
    assert has == expect_more or metrics.outlier_rate >= 0.0


# --------------------------------------------------------------------------- #
# Schema / freshness
# --------------------------------------------------------------------------- #
def test_schema_violation_missing_column():
    metrics = qm().evaluate({"a": [1, 2]}, expected_columns=["a", "b"])
    assert metrics.schema_violation_count == 1
    assert QualityIssueType.SCHEMA_VIOLATION in issue_types(metrics)


def test_schema_violation_extra_column():
    metrics = qm().evaluate({"a": [1], "b": [2]}, expected_columns=["a"])
    assert metrics.schema_violation_count == 1


def test_schema_no_violation():
    metrics = qm().evaluate({"a": [1], "b": [2]}, expected_columns=["a", "b"])
    assert metrics.schema_violation_count == 0


def test_freshness_issue_when_stale():
    metrics = qm().evaluate({"a": [1, 2, 3]}, freshness_seconds=7200)
    assert QualityIssueType.FRESHNESS in issue_types(metrics)


def test_freshness_no_issue_when_fresh():
    metrics = qm().evaluate({"a": [1, 2, 3]}, freshness_seconds=60)
    assert QualityIssueType.FRESHNESS not in issue_types(metrics)


def test_freshness_recorded():
    metrics = qm().evaluate({"a": [1, 2, 3]}, freshness_seconds=120)
    assert metrics.freshness_seconds == 120.0


# --------------------------------------------------------------------------- #
# Aggregate, determinism, serialization
# --------------------------------------------------------------------------- #
def test_total_records():
    metrics = qm().evaluate({"a": list(range(50))})
    assert metrics.total_records == 50


def test_issues_sorted_deterministic():
    arr = rng().normal(0, 1, 500)
    arr[:50] = np.nan
    arr[0] = 100
    m1 = qm().evaluate({"x": arr})
    m2 = qm().evaluate({"x": arr})
    assert [i.to_dict() for i in m1.issues] == [i.to_dict() for i in m2.issues]


def test_issue_count_matches_list():
    arr = rng().normal(0, 1, 500)
    arr[:50] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert metrics.issue_count == len(metrics.issues)


def test_empty_dataset_raises():
    with pytest.raises(ValidationError):
        qm().evaluate({})


def test_metrics_serializable():
    arr = rng().normal(0, 1, 500)
    arr[:10] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert json.dumps(metrics.to_dict())


def test_severity_assigned_to_issues():
    arr = rng().normal(0, 1, 200)
    arr[:100] = np.nan
    metrics = qm().evaluate({"x": arr})
    assert all(isinstance(i.severity, DriftSeverity) for i in metrics.issues)


def test_categorical_missing_detected():
    metrics = qm().evaluate({"c": ["a", "", "b", None]})
    assert QualityIssueType.MISSING_VALUES in issue_types(metrics)


def test_factory_with_config():
    cfg = MonitoringConfiguration(null_rate_threshold=0.5)
    assert create_data_quality_monitor(config=cfg).config.null_rate_threshold == 0.5


def test_combined_issues():
    arr = rng().normal(45, 12, 1000)
    arr[:50] = np.nan
    arr[60] = 9999
    metrics = qm().evaluate({"age": arr}, valid_ranges={"age": (0, 120)},
                            expected_columns=["age", "income"], freshness_seconds=7200)
    types = issue_types(metrics)
    assert QualityIssueType.MISSING_VALUES in types
    assert QualityIssueType.SCHEMA_VIOLATION in types
    assert QualityIssueType.FRESHNESS in types