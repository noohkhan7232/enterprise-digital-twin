"""Tests for deterministic data-drift detection algorithms and detector."""

from __future__ import annotations

import numpy as np
import pytest

from monitoring.data_drift_detector import (
    DataDriftDetector, DriftComputationError, create_data_drift_detector,
    histogram_distance, js_distance, kl_divergence, ks_statistic,
    population_stability_index,
)
from monitoring.monitoring_models import (
    DriftMethod, DriftSeverity, FeatureType, MonitoringConfiguration, ValidationError,
)


def rng(seed=7):
    return np.random.default_rng(seed)


def ref_sample(seed=7, n=4000):
    return rng(seed).normal(0.0, 1.0, n)


def shifted_sample(seed=11, n=4000):
    return rng(seed).normal(1.6, 1.3, n)


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def test_psi_zero_for_identical():
    a = ref_sample()
    assert population_stability_index(a, a) == pytest.approx(0.0, abs=1e-6)


def test_psi_small_for_same_distribution():
    assert population_stability_index(ref_sample(1), rng(2).normal(0, 1, 4000)) < 0.1


def test_psi_large_for_shift():
    assert population_stability_index(ref_sample(), shifted_sample()) > 0.25


def test_psi_nonnegative():
    assert population_stability_index(ref_sample(), shifted_sample()) >= 0.0


def test_psi_deterministic():
    a, b = ref_sample(), shifted_sample()
    assert population_stability_index(a, b) == population_stability_index(a, b)


@pytest.mark.parametrize("bins", [5, 10, 20, 50])
def test_psi_varies_with_bins(bins):
    val = population_stability_index(ref_sample(), shifted_sample(), num_bins=bins)
    assert val > 0.0


def test_psi_empty_raises():
    with pytest.raises(ValidationError):
        population_stability_index([], [1, 2, 3])


def test_psi_all_nan_raises():
    with pytest.raises(ValidationError):
        population_stability_index([float("nan")], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------------- #
def test_kl_zero_for_identical():
    a = ref_sample()
    assert kl_divergence(a, a) == pytest.approx(0.0, abs=1e-6)


def test_kl_larger_for_shift():
    assert kl_divergence(ref_sample(), shifted_sample()) > kl_divergence(ref_sample(), rng(3).normal(0, 1, 4000))


def test_kl_nonnegative():
    assert kl_divergence(ref_sample(), shifted_sample()) >= 0.0


def test_kl_deterministic():
    a, b = ref_sample(), shifted_sample()
    assert kl_divergence(a, b) == kl_divergence(a, b)


# --------------------------------------------------------------------------- #
# JS distance
# --------------------------------------------------------------------------- #
def test_js_zero_for_identical():
    a = ref_sample()
    assert js_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_js_bounded():
    assert 0.0 <= js_distance(ref_sample(), shifted_sample()) <= 1.0


def test_js_larger_for_shift():
    assert js_distance(ref_sample(), shifted_sample()) > js_distance(ref_sample(), rng(3).normal(0, 1, 4000))


def test_js_approximately_symmetric():
    a, b = ref_sample(), shifted_sample()
    # Binning uses the reference's quantiles, so symmetry holds only approximately.
    assert abs(js_distance(a, b) - js_distance(b, a)) < 0.05


def test_js_deterministic():
    a, b = ref_sample(), shifted_sample()
    assert js_distance(a, b) == js_distance(a, b)


# --------------------------------------------------------------------------- #
# KS test
# --------------------------------------------------------------------------- #
def test_ks_statistic_in_range():
    d, p = ks_statistic(ref_sample(), shifted_sample())
    assert 0.0 <= d <= 1.0
    assert 0.0 <= p <= 1.0


def test_ks_larger_d_for_shift():
    d_same, _ = ks_statistic(ref_sample(1), rng(2).normal(0, 1, 4000))
    d_shift, _ = ks_statistic(ref_sample(), shifted_sample())
    assert d_shift > d_same


def test_ks_smaller_p_for_shift():
    _, p_same = ks_statistic(ref_sample(1), rng(2).normal(0, 1, 4000))
    _, p_shift = ks_statistic(ref_sample(), shifted_sample())
    assert p_shift < p_same


def test_ks_identical_zero():
    a = ref_sample()
    d, p = ks_statistic(a, a)
    assert d == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_ks_deterministic():
    a, b = ref_sample(), shifted_sample()
    assert ks_statistic(a, b) == ks_statistic(a, b)


# --------------------------------------------------------------------------- #
# Histogram distance
# --------------------------------------------------------------------------- #
def test_histogram_distance_bounded():
    assert 0.0 <= histogram_distance(ref_sample(), shifted_sample()) <= 1.0


def test_histogram_distance_zero_identical():
    a = ref_sample()
    assert histogram_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_histogram_distance_larger_for_shift():
    assert histogram_distance(ref_sample(), shifted_sample()) > histogram_distance(ref_sample(), rng(3).normal(0, 1, 4000))


# --------------------------------------------------------------------------- #
# Severity & confidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("score,expected", [
    (0.05, DriftSeverity.NONE),
    (0.15, DriftSeverity.LOW),
    (0.30, DriftSeverity.MODERATE),
    (0.60, DriftSeverity.HIGH),
    (0.90, DriftSeverity.CRITICAL),
])
def test_classify_severity_bands(score, expected):
    assert DataDriftDetector.classify_severity(score, 0.2) is expected


def test_classify_severity_zero_threshold():
    assert DataDriftDetector.classify_severity(0.0, 0.0) is DriftSeverity.NONE


def test_confidence_increases_with_size():
    small = DataDriftDetector.confidence_score(10, 10)
    large = DataDriftDetector.confidence_score(10000, 10000)
    assert 0.0 <= small < large <= 1.0


def test_confidence_zero_for_empty():
    assert DataDriftDetector.confidence_score(0, 0) == 0.0


def test_confidence_uses_min_size():
    assert DataDriftDetector.confidence_score(10, 100000) == DataDriftDetector.confidence_score(10, 10)


# --------------------------------------------------------------------------- #
# Numerical / categorical detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", [
    DriftMethod.PSI, DriftMethod.KL_DIVERGENCE, DriftMethod.JS_DISTANCE,
    DriftMethod.HISTOGRAM_DISTANCE, DriftMethod.KS_TEST,
])
def test_detect_numerical_methods(method):
    det = DataDriftDetector()
    result = det.detect_numerical("x", ref_sample(), shifted_sample(), method)
    assert result.method is method
    assert result.drifted is True
    assert result.reference_size == 4000


def test_detect_numerical_no_drift():
    det = DataDriftDetector()
    result = det.detect_numerical("x", ref_sample(1), rng(2).normal(0, 1, 4000), DriftMethod.PSI)
    assert result.drifted is False


def test_detect_numerical_ks_has_pvalue():
    det = DataDriftDetector()
    result = det.detect_numerical("x", ref_sample(), shifted_sample(), DriftMethod.KS_TEST)
    assert result.p_value is not None


def test_detect_numerical_unsupported_method():
    det = DataDriftDetector()
    with pytest.raises(ValidationError):
        det.detect_numerical("x", ref_sample(), shifted_sample(), DriftMethod.CATEGORICAL)


def test_detect_categorical_drift():
    det = DataDriftDetector()
    result = det.detect_categorical("c", ["a"] * 80 + ["b"] * 20, ["a"] * 20 + ["b"] * 80)
    assert result.drifted is True
    assert result.method is DriftMethod.CATEGORICAL


def test_detect_categorical_no_drift():
    det = DataDriftDetector()
    result = det.detect_categorical("c", ["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50)
    assert result.drifted is False


def test_detect_categorical_empty_raises():
    det = DataDriftDetector()
    with pytest.raises(ValidationError):
        det.detect_categorical("c", [], ["a"])


def test_detect_feature_dispatch_categorical():
    det = DataDriftDetector()
    result = det.detect_feature("c", ["a", "b"], ["a", "a"], FeatureType.CATEGORICAL)
    assert result.method is DriftMethod.CATEGORICAL


def test_detect_feature_dispatch_numerical():
    det = DataDriftDetector()
    result = det.detect_feature("x", ref_sample(), shifted_sample(), FeatureType.NUMERICAL)
    assert result.method is DriftMethod.PSI


# --------------------------------------------------------------------------- #
# Dataset-level
# --------------------------------------------------------------------------- #
def test_detect_dataset_returns_sorted():
    det = DataDriftDetector()
    ref = {"b": ref_sample(), "a": ref_sample(2)}
    cur = {"b": shifted_sample(), "a": rng(2).normal(0, 1, 4000)}
    results = det.detect_dataset(ref, cur)
    assert [r.feature_name for r in results] == ["a", "b"]


def test_detect_dataset_empty_reference():
    det = DataDriftDetector()
    with pytest.raises(ValidationError):
        det.detect_dataset({}, {"a": [1, 2]})


def test_detect_dataset_no_shared_features():
    det = DataDriftDetector()
    with pytest.raises(ValidationError):
        det.detect_dataset({"a": ref_sample()}, {"b": shifted_sample()})


def test_overall_drift_score_mean():
    det = DataDriftDetector()
    results = det.detect_dataset({"x": ref_sample(), "y": ref_sample(2)},
                                 {"x": shifted_sample(), "y": rng(2).normal(0, 1, 4000)})
    overall = det.overall_drift_score(results)
    assert overall == pytest.approx(np.mean([r.drift_score for r in results]))


def test_overall_drift_score_empty():
    assert DataDriftDetector.overall_drift_score([]) == 0.0


def test_feature_drift_ranking_descending():
    det = DataDriftDetector()
    results = det.detect_dataset({"x": ref_sample(), "y": ref_sample(2)},
                                 {"x": shifted_sample(), "y": rng(2).normal(0, 1, 4000)})
    ranking = det.feature_drift_ranking(results)
    assert ranking[0][1] >= ranking[1][1]
    assert ranking[0][0] == "x"


def test_drifted_features_list():
    det = DataDriftDetector()
    results = det.detect_dataset({"x": ref_sample(), "y": ref_sample(2)},
                                 {"x": shifted_sample(), "y": rng(2).normal(0, 1, 4000)})
    assert "x" in det.drifted_features(results)


def test_summarize_structure():
    det = DataDriftDetector()
    results = det.detect_dataset({"x": ref_sample()}, {"x": shifted_sample()})
    summary = det.summarize(results)
    assert summary["top_feature"] == "x"
    assert summary["feature_count"] == 1
    assert "ranking" in summary


# --------------------------------------------------------------------------- #
# Statistics & misc
# --------------------------------------------------------------------------- #
def test_compute_feature_statistics_numerical():
    det = DataDriftDetector()
    stat = det.compute_feature_statistics("x", ref_sample())
    assert stat.mean is not None
    assert len(stat.histogram) == det.config.num_bins


def test_compute_feature_statistics_categorical():
    det = DataDriftDetector()
    stat = det.compute_feature_statistics("c", ["a", "a", "b"], FeatureType.CATEGORICAL)
    assert stat.unique_count == 2
    assert dict(stat.categories)["a"] == pytest.approx(2 / 3)


def test_compute_feature_statistics_missing():
    det = DataDriftDetector()
    stat = det.compute_feature_statistics("x", [1.0, float("nan"), 3.0])
    assert stat.missing_count == 1


def test_factory_deterministic_default():
    det = create_data_drift_detector()
    assert isinstance(det, DataDriftDetector)


def test_factory_with_config():
    cfg = MonitoringConfiguration(psi_threshold=0.5)
    det = create_data_drift_detector(config=cfg)
    assert det.config.psi_threshold == 0.5


def test_detector_result_threshold_matches_config():
    cfg = MonitoringConfiguration(psi_threshold=0.123)
    det = DataDriftDetector(config=cfg)
    result = det.detect_numerical("x", ref_sample(), shifted_sample(), DriftMethod.PSI)
    assert result.threshold == 0.123


def test_large_dataset_performance():
    det = DataDriftDetector()
    big_ref = rng(1).normal(0, 1, 200000)
    big_cur = rng(2).normal(0.5, 1, 200000)
    result = det.detect_numerical("x", big_ref, big_cur, DriftMethod.PSI)
    assert result.drift_score > 0.0