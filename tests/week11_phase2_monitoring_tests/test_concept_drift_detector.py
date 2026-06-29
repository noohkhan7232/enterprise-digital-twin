"""Tests for deterministic concept-drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from monitoring.concept_drift_detector import ConceptDriftDetector, create_concept_drift_detector
from monitoring.monitoring_models import DriftSeverity, MonitoringConfiguration, ValidationError


def rng(seed=7):
    return np.random.default_rng(seed)


def det():
    return ConceptDriftDetector()


# --------------------------------------------------------------------------- #
# Performance drift
# --------------------------------------------------------------------------- #
def test_performance_drift_detected_on_degradation():
    r = det().detect_performance_drift(0.92, 0.80)
    assert r.detected is True
    assert r.delta < 0


def test_performance_drift_not_detected_on_improvement():
    r = det().detect_performance_drift(0.92, 0.95)
    assert r.detected is False


def test_performance_drift_lower_is_better():
    r = det().detect_performance_drift(0.10, 0.30, higher_is_better=False)
    assert r.detected is True


def test_performance_drift_method_name():
    assert det().detect_performance_drift(0.9, 0.8).method == "performance_drift"


def test_performance_drift_custom_threshold():
    r = det().detect_performance_drift(0.92, 0.90, threshold=0.05)
    assert r.detected is False


@pytest.mark.parametrize("ref,cur,detected", [
    (0.95, 0.10, True),
    (0.95, 0.94, False),
    (0.80, 0.50, True),
    (0.70, 0.69, False),
])
def test_performance_drift_cases(ref, cur, detected):
    assert det().detect_performance_drift(ref, cur).detected is detected


# --------------------------------------------------------------------------- #
# Rolling performance
# --------------------------------------------------------------------------- #
def test_rolling_performance_detects_decline():
    r = det().detect_rolling_performance([0.9] * 60 + [0.7] * 60, window=40)
    assert r.detected is True
    assert r.trend_slope < 0


def test_rolling_performance_stable():
    r = det().detect_rolling_performance([0.9] * 120, window=40)
    assert r.detected is False


def test_rolling_performance_records_window():
    r = det().detect_rolling_performance([0.9] * 100, window=30)
    assert r.window_size == 30


def test_rolling_performance_empty_raises():
    with pytest.raises(ValidationError):
        det().detect_rolling_performance([])


def test_rolling_performance_window_capped():
    r = det().detect_rolling_performance([0.8, 0.7], window=100)
    assert r.window_size <= 2


# --------------------------------------------------------------------------- #
# Sliding window
# --------------------------------------------------------------------------- #
def test_sliding_window_detects_accuracy_drop():
    r = det().detect_sliding_window([1] * 60 + [0, 1, 0, 0] * 15, window=30)
    assert r.detected is True


def test_sliding_window_stable_no_drift():
    r = det().detect_sliding_window([1, 0] * 60, window=30)
    assert r.detected is False


def test_sliding_window_empty_raises():
    with pytest.raises(ValidationError):
        det().detect_sliding_window([])


# --------------------------------------------------------------------------- #
# Distribution-based
# --------------------------------------------------------------------------- #
def test_error_distribution_drift():
    r = det().detect_error_distribution(np.abs(rng(1).normal(0, 1, 3000)),
                                        np.abs(rng(2).normal(0, 2.5, 3000)))
    assert r.detected is True
    assert r.method == "error_distribution"


def test_error_distribution_no_drift():
    r = det().detect_error_distribution(np.abs(rng(1).normal(0, 1, 3000)),
                                        np.abs(rng(2).normal(0, 1, 3000)))
    assert r.detected is False


def test_residual_drift_detected():
    r = det().detect_residual_drift(rng(1).normal(0, 1, 3000), rng(2).normal(0, 3, 3000))
    assert r.detected is True
    assert r.method == "residual_drift"


def test_residual_drift_stable():
    r = det().detect_residual_drift(rng(1).normal(0, 1, 3000), rng(2).normal(0, 1, 3000))
    assert r.detected is False


def test_label_distribution_drift():
    r = det().detect_label_distribution(["a"] * 90 + ["b"] * 10, ["a"] * 40 + ["b"] * 60)
    assert r.detected is True
    assert r.method == "label_distribution"


def test_label_distribution_stable():
    r = det().detect_label_distribution(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50)
    assert r.detected is False


def test_target_drift_detected():
    r = det().detect_target_drift(rng(1).normal(0, 1, 3000), rng(2).normal(2, 1, 3000))
    assert r.detected is True
    assert r.method == "target_drift"


def test_target_drift_stable():
    r = det().detect_target_drift(rng(1).normal(0, 1, 3000), rng(2).normal(0, 1, 3000))
    assert r.detected is False


# --------------------------------------------------------------------------- #
# Trend analysis
# --------------------------------------------------------------------------- #
def test_trend_slope_increasing():
    assert ConceptDriftDetector.trend_slope([1, 2, 3, 4, 5]) > 0


def test_trend_slope_decreasing():
    assert ConceptDriftDetector.trend_slope([5, 4, 3, 2, 1]) < 0


def test_trend_slope_flat_is_zero():
    assert ConceptDriftDetector.trend_slope([5, 5, 5, 5]) == 0.0


def test_trend_slope_single_point():
    assert ConceptDriftDetector.trend_slope([1.0]) == 0.0


def test_analyze_trend_increasing():
    r = det().analyze_trend([1, 2, 3, 4, 5, 6])
    assert dict(r.details)["direction"] == "increasing"


def test_analyze_trend_decreasing():
    r = det().analyze_trend([6, 5, 4, 3, 2, 1])
    assert dict(r.details)["direction"] == "decreasing"


def test_analyze_trend_flat():
    r = det().analyze_trend([3, 3, 3, 3])
    assert dict(r.details)["direction"] == "flat"
    assert r.trend_slope == 0.0


def test_analyze_trend_empty_raises():
    with pytest.raises(ValidationError):
        det().analyze_trend([])


# --------------------------------------------------------------------------- #
# Determinism, severity, factory
# --------------------------------------------------------------------------- #
def test_error_distribution_deterministic():
    a = np.abs(rng(1).normal(0, 1, 2000))
    b = np.abs(rng(2).normal(0, 2, 2000))
    assert det().detect_error_distribution(a, b).drift_score == det().detect_error_distribution(a, b).drift_score


def test_severity_is_drift_severity():
    r = det().detect_performance_drift(0.95, 0.30)
    assert isinstance(r.severity, DriftSeverity)


def test_factory_deterministic():
    assert isinstance(create_concept_drift_detector(), ConceptDriftDetector)


def test_factory_with_config():
    cfg = MonitoringConfiguration(psi_threshold=0.4)
    assert create_concept_drift_detector(config=cfg).config.psi_threshold == 0.4


def test_custom_performance_threshold_constructor():
    d = ConceptDriftDetector(performance_threshold=0.2)
    assert d.detect_performance_drift(0.9, 0.75).detected is False


def test_result_serializable():
    import json
    r = det().detect_performance_drift(0.9, 0.7)
    assert json.dumps(r.to_dict())