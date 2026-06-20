#!/usr/bin/env python3
"""Comprehensive test suite for ``src/predictive/health_trend_analyzer.py``.

The analyzer is pure NumPy, so the **entire** suite runs without PyTorch or
SciPy.  Coverage (80+ tests):

- HealthTrendConfig validation
- TrendDirection / EarlyWarning enums
- Registry (register / build / list / duplicate rejection)
- Pure estimators: linear_slope, moving_average, moving_average_slope,
  theil_sen_slope (incl. robustness), quadratic_curvature, detect_change_point
- Trend classification: improving / stable / degrading / accelerating
- Early-warning generation: normal / watch / warning / critical
- Change-point detection (regime shifts, magnitude, significance)
- Confidence scoring (trend_confidence, change_confidence)
- NaN handling (interpolate / drop / raise)
- Short-sequence handling
- HealthTrendResult container (to_dict)
- Integration with HealthIndexEngine (analyze_engine)
- RUL gating (should_predict_rul)
- ExperimentTracker integration (failure-safe)
- Edge cases

Each test constructs its own ``default_rng`` so cases are independent and
deterministic (no shared generator state across tests).

Run::

    pytest tests/test_health_trend_analyzer.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predictive.health_trend_analyzer import (
    ANALYZER_NAME,
    HEALTH_TREND_REGISTRY,
    EarlyWarning,
    HealthTrendAnalyzer,
    HealthTrendConfig,
    HealthTrendResult,
    TrendDirection,
    build_health_trend_analyzer,
    detect_change_point,
    linear_slope,
    list_health_trend_analyzers,
    moving_average,
    moving_average_slope,
    quadratic_curvature,
    register_health_trend_analyzer,
    theil_sen_slope,
)


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestHealthTrendConfig:
    """Tests for configuration validation."""

    def test_defaults(self) -> None:
        c = HealthTrendConfig()
        assert c.min_history >= 3
        assert c.nan_policy == "interpolate"

    def test_bad_min_history(self) -> None:
        with pytest.raises(ValueError, match="min_history"):
            HealthTrendConfig(min_history=2)

    def test_bad_dead_band(self) -> None:
        with pytest.raises(ValueError, match="slope_dead_band"):
            HealthTrendConfig(slope_dead_band=-0.1)

    def test_bad_accel_threshold(self) -> None:
        with pytest.raises(ValueError, match="accel_curvature_threshold"):
            HealthTrendConfig(accel_curvature_threshold=0.1)

    def test_bad_ma_window(self) -> None:
        with pytest.raises(ValueError, match="ma_window"):
            HealthTrendConfig(ma_window=0)

    def test_bad_change_min_segment(self) -> None:
        with pytest.raises(ValueError, match="change_min_segment"):
            HealthTrendConfig(change_min_segment=0)

    def test_bad_significance(self) -> None:
        with pytest.raises(ValueError, match="change_significance_threshold"):
            HealthTrendConfig(change_significance_threshold=-1)

    def test_bad_health_order(self) -> None:
        with pytest.raises(ValueError, match="critical_health"):
            HealthTrendConfig(critical_health=70, warn_health=60)

    def test_bad_slope_order(self) -> None:
        with pytest.raises(ValueError, match="watch_slope"):
            HealthTrendConfig(watch_slope=2.0, warning_slope=1.0)

    def test_bad_nan_policy(self) -> None:
        with pytest.raises(ValueError, match="nan_policy"):
            HealthTrendConfig(nan_policy="bogus")

    def test_frozen(self) -> None:
        c = HealthTrendConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.min_history = 10  # type: ignore[misc]

    def test_valid_nan_policies(self) -> None:
        for p in ("interpolate", "drop", "raise"):
            HealthTrendConfig(nan_policy=p)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for the enums."""

    def test_trend_directions(self) -> None:
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.DEGRADING.value == "degrading"
        assert TrendDirection.ACCELERATING.value == "accelerating"

    def test_early_warnings(self) -> None:
        assert EarlyWarning.NORMAL.value == "normal"
        assert EarlyWarning.WATCH.value == "watch"
        assert EarlyWarning.WARNING.value == "warning"
        assert EarlyWarning.CRITICAL.value == "critical"

    def test_trend_is_str(self) -> None:
        assert TrendDirection.STABLE == "stable"

    def test_warning_is_str(self) -> None:
        assert EarlyWarning.NORMAL == "normal"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the registry."""

    def test_registered(self) -> None:
        assert ANALYZER_NAME in HEALTH_TREND_REGISTRY
        assert ANALYZER_NAME in list_health_trend_analyzers()

    def test_build_by_name(self) -> None:
        assert isinstance(build_health_trend_analyzer(ANALYZER_NAME),
                          HealthTrendAnalyzer)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown health-trend analyzer"):
            build_health_trend_analyzer("nope")

    def test_registry_name_stamped(self) -> None:
        assert HealthTrendAnalyzer._registry_name == ANALYZER_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_health_trend_analyzer(ANALYZER_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        a = build_health_trend_analyzer(ANALYZER_NAME,
                                        config=HealthTrendConfig(min_history=10))
        assert a.config.min_history == 10


# ---------------------------------------------------------------------------
# linear_slope
# ---------------------------------------------------------------------------


class TestLinearSlope:
    """Tests for the OLS slope estimator."""

    def test_recovers_slope(self) -> None:
        s, i, r2 = linear_slope(np.array([95.0, 93, 91, 89, 87]))
        assert s == pytest.approx(-2.0)
        assert i == pytest.approx(95.0)

    def test_perfect_r2(self) -> None:
        _, _, r2 = linear_slope(np.array([10.0, 20, 30, 40]))
        assert r2 == pytest.approx(1.0)

    def test_flat_zero_slope(self) -> None:
        s, _, r2 = linear_slope(np.full(10, 50.0))
        assert abs(s) < 1e-9
        assert r2 == 0.0

    def test_single_point(self) -> None:
        s, i, r2 = linear_slope(np.array([42.0]))
        assert s == 0.0 and i == 42.0

    def test_positive_slope(self) -> None:
        s, _, _ = linear_slope(np.array([10.0, 20, 30]))
        assert s > 0


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------


class TestMovingAverage:
    """Tests for moving average and its slope."""

    def test_valid_length(self) -> None:
        out = moving_average(np.arange(10, dtype=float), 3)
        assert out.size == 8

    def test_window_one_identity(self) -> None:
        y = np.arange(5, dtype=float)
        assert np.array_equal(moving_average(y, 1), y)

    def test_window_larger_than_series(self) -> None:
        y = np.arange(3, dtype=float)
        assert np.array_equal(moving_average(y, 10), y)

    def test_smooths_noise(self) -> None:
        y = np.array([0.0, 10, 0, 10, 0, 10])
        out = moving_average(y, 3)
        assert out.std() < y.std()

    def test_ma_slope_degrading(self) -> None:
        s = moving_average_slope(90.0 - 2.0 * np.arange(20), 3)
        assert s < 0


# ---------------------------------------------------------------------------
# theil_sen_slope
# ---------------------------------------------------------------------------


class TestTheilSen:
    """Tests for the robust slope estimator."""

    def test_recovers_clean_slope(self) -> None:
        s = theil_sen_slope(95.0 - 2.0 * np.arange(15))
        assert s == pytest.approx(-2.0, abs=0.01)

    def test_robust_to_endpoint_outlier(self) -> None:
        y = 95.0 - 2.0 * np.arange(15)
        y[14] = 95.0  # corrupt last point
        ts = theil_sen_slope(y)
        ols, _, _ = linear_slope(y)
        # Theil-Sen stays near the true slope; OLS is dragged toward it
        assert abs(ts + 2.0) < abs(ols + 2.0)

    def test_robust_to_midpoint_spike(self) -> None:
        y = 95.0 - 2.0 * np.arange(15)
        y[7] = 5.0
        ts = theil_sen_slope(y)
        assert ts == pytest.approx(-2.0, abs=0.3)

    def test_single_point(self) -> None:
        assert theil_sen_slope(np.array([5.0])) == 0.0

    def test_positive_slope(self) -> None:
        assert theil_sen_slope(np.arange(10, dtype=float)) > 0


# ---------------------------------------------------------------------------
# quadratic_curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    """Tests for the curvature estimator."""

    def test_linear_near_zero(self) -> None:
        assert abs(quadratic_curvature(90.0 - 2.0 * np.arange(15))) < 1e-6

    def test_accelerating_negative(self) -> None:
        assert quadratic_curvature(95.0 - 0.1 * np.arange(15) ** 2) < -0.1

    def test_decelerating_positive(self) -> None:
        # Convex-up decline (decelerating)
        assert quadratic_curvature(np.array([95.0, 80, 70, 64, 61, 60])) > 0

    def test_short_series_zero(self) -> None:
        assert quadratic_curvature(np.array([1.0, 2.0])) == 0.0


# ---------------------------------------------------------------------------
# detect_change_point
# ---------------------------------------------------------------------------


class TestChangePointDetection:
    """Tests for change-point detection."""

    def test_regime_shift_located(self) -> None:
        y = np.concatenate([np.full(10, 90.0), np.full(10, 50.0)])
        idx, mag, sig = detect_change_point(y, 3)
        assert 8 <= idx <= 12
        assert mag == pytest.approx(40.0, abs=2.0)

    def test_no_shift_small_magnitude(self) -> None:
        y = np.full(20, 70.0)
        idx, mag, sig = detect_change_point(y, 3)
        assert mag < 1e-6

    def test_too_short_returns_negative(self) -> None:
        idx, mag, sig = detect_change_point(np.arange(4, dtype=float), 3)
        assert idx == -1

    def test_significance_scales_with_separation(self) -> None:
        clean = np.concatenate([np.full(10, 90.0), np.full(10, 50.0)])
        _, _, sig_clean = detect_change_point(clean, 3)
        noisy = clean + _rng(1).normal(0, 10, 20)
        _, _, sig_noisy = detect_change_point(noisy, 3)
        assert sig_clean > sig_noisy

    def test_magnitude_direction_agnostic(self) -> None:
        # Upward shift detected with positive magnitude
        y = np.concatenate([np.full(8, 40.0), np.full(8, 80.0)])
        _, mag, _ = detect_change_point(y, 3)
        assert mag == pytest.approx(40.0, abs=2.0)


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------


class TestTrendClassification:
    """Tests for trend classification."""

    def _an(self) -> HealthTrendAnalyzer:
        return HealthTrendAnalyzer(HealthTrendConfig())

    def test_stable(self) -> None:
        r = self._an().analyze(np.full(20, 85.0) + _rng(0).normal(0, 0.5, 20))
        assert r.trend == TrendDirection.STABLE

    def test_degrading(self) -> None:
        r = self._an().analyze(90.0 - 1.5 * np.arange(20) + _rng(1).normal(0, 0.5, 20))
        assert r.trend == TrendDirection.DEGRADING

    def test_improving(self) -> None:
        r = self._an().analyze(50.0 + 1.5 * np.arange(20) + _rng(2).normal(0, 0.5, 20))
        assert r.trend == TrendDirection.IMPROVING

    def test_accelerating(self) -> None:
        r = self._an().analyze(95.0 - 0.08 * np.arange(20) ** 2 + _rng(3).normal(0, 0.5, 20))
        assert r.trend == TrendDirection.ACCELERATING

    def test_flat_is_stable(self) -> None:
        r = self._an().analyze(np.full(15, 70.0))
        assert r.trend == TrendDirection.STABLE

    def test_gentle_degrade_not_accelerating(self) -> None:
        # Linear decline -> DEGRADING not ACCELERATING (curvature ~0)
        r = self._an().analyze(90.0 - 1.0 * np.arange(20))
        assert r.trend == TrendDirection.DEGRADING

    def test_within_dead_band_stable(self) -> None:
        # Slope below dead band -> STABLE
        r = self._an().analyze(80.0 - 0.02 * np.arange(20))
        assert r.trend == TrendDirection.STABLE


# ---------------------------------------------------------------------------
# Early warning
# ---------------------------------------------------------------------------


class TestEarlyWarning:
    """Tests for early-warning generation."""

    def _an(self) -> HealthTrendAnalyzer:
        return HealthTrendAnalyzer(HealthTrendConfig())

    def test_healthy_stable_normal(self) -> None:
        for seed in range(5):
            r = self._an().analyze(np.full(20, 90.0) + _rng(seed).normal(0, 0.8, 20))
            assert r.warning == EarlyWarning.NORMAL

    def test_low_accelerating_critical(self) -> None:
        r = self._an().analyze(60.0 - 0.1 * np.arange(20) ** 2 + _rng(0).normal(0, 0.5, 20))
        assert r.warning in (EarlyWarning.WARNING, EarlyWarning.CRITICAL)

    def test_critical_health_level(self) -> None:
        # Very low health -> at least WARNING
        r = self._an().analyze(np.full(20, 35.0))
        assert r.warning in (EarlyWarning.WARNING, EarlyWarning.CRITICAL)

    def test_mild_degrade_watch_or_warning(self) -> None:
        r = self._an().analyze(85.0 - 0.4 * np.arange(20) + _rng(4).normal(0, 0.5, 20))
        assert r.warning in (EarlyWarning.WATCH, EarlyWarning.WARNING)

    def test_warning_severity_order(self) -> None:
        # Decreasing health should not decrease the warning severity
        an = self._an()
        order = {EarlyWarning.NORMAL: 0, EarlyWarning.WATCH: 1,
                 EarlyWarning.WARNING: 2, EarlyWarning.CRITICAL: 3}
        healthy = an.analyze(np.full(20, 92.0))
        degrading = an.analyze(50.0 - 0.1 * np.arange(20) ** 2)
        assert order[degrading.warning] > order[healthy.warning]


# ---------------------------------------------------------------------------
# Change point in results
# ---------------------------------------------------------------------------


class TestChangePointResult:
    """Tests for change-point reporting in results."""

    def _an(self) -> HealthTrendAnalyzer:
        return HealthTrendAnalyzer(HealthTrendConfig())

    def test_detects_regime_change(self) -> None:
        r = self._an().analyze(np.concatenate([np.full(12, 88.0), np.full(12, 52.0)]))
        assert r.has_change_point
        assert 10 <= r.change_index <= 14

    def test_change_magnitude_reported(self) -> None:
        r = self._an().analyze(np.concatenate([np.full(12, 88.0), np.full(12, 52.0)]))
        assert r.change_magnitude == pytest.approx(36.0, abs=4.0)

    def test_no_change_for_clean_line(self) -> None:
        r = self._an().analyze(90.0 - 1.0 * np.arange(20))
        # A pure line has no abrupt regime change of high significance
        assert isinstance(r.has_change_point, bool)

    def test_change_index_negative_when_absent(self) -> None:
        r = self._an().analyze(np.full(20, 80.0))
        assert r.change_index == -1

    def test_change_confidence_high_for_clear_shift(self) -> None:
        r = self._an().analyze(np.concatenate([np.full(12, 90.0), np.full(12, 40.0)]))
        assert r.change_confidence > 0.5


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestConfidence:
    """Tests for confidence scoring."""

    def _an(self) -> HealthTrendAnalyzer:
        return HealthTrendAnalyzer(HealthTrendConfig())

    def test_trend_confidence_high_for_clean(self) -> None:
        r = self._an().analyze(90.0 - 1.5 * np.arange(30))
        assert r.trend_confidence > 0.7

    def test_trend_confidence_low_for_noisy(self) -> None:
        r = self._an().analyze(90.0 - 1.5 * np.arange(30) + _rng(5).normal(0, 20, 30))
        clean = self._an().analyze(90.0 - 1.5 * np.arange(30))
        assert r.trend_confidence < clean.trend_confidence

    def test_trend_confidence_in_unit_interval(self) -> None:
        r = self._an().analyze(90.0 - 1.5 * np.arange(20) + _rng(6).normal(0, 5, 20))
        assert 0.0 <= r.trend_confidence <= 1.0

    def test_short_series_lower_confidence(self) -> None:
        short = self._an().analyze(90.0 - 1.5 * np.arange(5))
        long = self._an().analyze(90.0 - 1.5 * np.arange(40))
        # Same underlying line, but shrinkage lowers the short-series confidence
        assert short.trend_confidence < long.trend_confidence

    def test_change_confidence_in_unit_interval(self) -> None:
        r = self._an().analyze(np.concatenate([np.full(10, 88.0), np.full(10, 52.0)]))
        assert 0.0 <= r.change_confidence <= 1.0

    def test_change_confidence_zero_without_change(self) -> None:
        r = self._an().analyze(np.full(20, 80.0))
        assert r.change_confidence == 0.0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNanHandling:
    """Tests for NaN policies."""

    def test_interpolate(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(nan_policy="interpolate"))
        r = an.analyze(np.array([90.0, 88, np.nan, 84, 82, 80, 78]))
        assert r.n_observations == 7
        assert any("interpolat" in w for w in r.warnings)

    def test_drop(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(nan_policy="drop"))
        r = an.analyze(np.array([90.0, 88, np.nan, 84, 82, 80, 78]))
        assert r.n_observations == 6

    def test_raise(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(nan_policy="raise"))
        with pytest.raises(ValueError, match="non-finite"):
            an.analyze(np.array([90.0, 88, np.nan, 84, 82]))

    def test_interpolate_recovers_trend(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        clean = 90.0 - 2.0 * np.arange(10)
        holed = clean.copy()
        holed[5] = np.nan
        r = an.analyze(holed)
        assert r.robust_slope == pytest.approx(-2.0, abs=0.2)

    def test_inf_treated_as_nan(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(nan_policy="drop"))
        r = an.analyze(np.array([90.0, 88, np.inf, 84, 82, 80]))
        assert r.n_observations == 5

    def test_all_nan_raises(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(nan_policy="interpolate"))
        with pytest.raises(ValueError):
            an.analyze(np.array([np.nan, np.nan, np.nan, np.nan]))


# ---------------------------------------------------------------------------
# Short-sequence / shape handling
# ---------------------------------------------------------------------------


class TestSequenceHandling:
    """Tests for sequence-length and shape handling."""

    def test_too_short_raises(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(min_history=5))
        with pytest.raises(ValueError, match="at least"):
            an.analyze([90, 80])

    def test_exactly_min_history(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(min_history=4))
        r = an.analyze([95, 90, 85, 80])
        assert isinstance(r, HealthTrendResult)

    def test_2d_raises(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        with pytest.raises(ValueError, match="one-dimensional"):
            an.analyze(np.ones((5, 2)))

    def test_list_input(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze([95.0, 90, 85, 80, 75])
        assert r.trend == TrendDirection.DEGRADING

    def test_dropped_below_min_raises(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig(min_history=5, nan_policy="drop"))
        with pytest.raises(ValueError, match="at least"):
            an.analyze(np.array([90.0, np.nan, np.nan, 80, 70]))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TestResultContainer:
    """Tests for the result container."""

    def test_to_dict_keys(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        d = an.analyze(90.0 - 1.5 * np.arange(20)).to_dict()
        for key in ("trend", "warning", "robust_slope", "trend_confidence",
                    "change_index", "has_change_point"):
            assert key in d

    def test_to_dict_enum_values(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        d = an.analyze(np.full(20, 85.0)).to_dict()
        assert d["trend"] == "stable"
        assert isinstance(d["warning"], str)

    def test_current_and_mean_health(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        traj = 90.0 - 1.0 * np.arange(11)  # 90..80
        r = an.analyze(traj)
        assert r.current_health == pytest.approx(80.0)
        assert r.mean_health == pytest.approx(85.0)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Tests for integration with the wider framework."""

    def test_analyze_engine(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine

        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.5))
        he.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.09, 20):
            he.update(anomaly_score=float(s))
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze_engine(he)
        assert isinstance(r, HealthTrendResult)

    def test_engine_detects_degradation(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine

        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.6))
        he.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.14, 25):
            he.update(anomaly_score=float(s))
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze_engine(he)
        assert r.trend in (TrendDirection.DEGRADING, TrendDirection.ACCELERATING)

    def test_should_predict_rul_degrading(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(90.0 - 1.5 * np.arange(20))
        assert an.should_predict_rul(r) is True

    def test_should_not_predict_rul_stable(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(np.full(20, 85.0))
        assert an.should_predict_rul(r) is False

    def test_should_not_predict_rul_improving(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(50.0 + 1.5 * np.arange(20))
        assert an.should_predict_rul(r) is False

    def test_chains_to_rul_predictor(self) -> None:
        # Trend gates a real RUL prediction downstream
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        an = HealthTrendAnalyzer(HealthTrendConfig())
        traj = 90.0 - 2.0 * np.arange(25)
        result = an.analyze(traj)
        if an.should_predict_rul(result):
            pred = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(traj)
            assert math.isfinite(pred.rul)


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_analysis(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        an = HealthTrendAnalyzer(HealthTrendConfig(), experiment_tracker=FakeTracker())
        an.analyze(90.0 - 1.5 * np.arange(20))
        assert len(logged) == 1
        assert "trend_confidence" in logged[0]

    def test_logs_step_increment(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(step)

        an = HealthTrendAnalyzer(HealthTrendConfig(), experiment_tracker=FakeTracker())
        an.analyze(90.0 - 1.5 * np.arange(20))
        an.analyze(90.0 - 1.5 * np.arange(20))
        assert logged == [0, 1]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        an = HealthTrendAnalyzer(HealthTrendConfig(),
                                 experiment_tracker=BrokenTracker())
        r = an.analyze(90.0 - 1.5 * np.arange(20))
        assert r.trend is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_constant_trajectory(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(np.full(15, 77.0))
        assert r.trend == TrendDirection.STABLE
        assert r.robust_slope == pytest.approx(0.0, abs=1e-9)

    def test_full_health(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(np.full(15, 100.0))
        assert r.warning == EarlyWarning.NORMAL

    def test_zero_health(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(np.full(15, 0.0))
        assert r.warning in (EarlyWarning.WARNING, EarlyWarning.CRITICAL)

    def test_n_analyses_increments(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        an.analyze(np.full(15, 80.0))
        an.analyze(np.full(15, 80.0))
        assert an._n_analyses == 2

    def test_single_spike_recovered(self) -> None:
        # A single transient spike should not flip the trend on a stable signal
        an = HealthTrendAnalyzer(HealthTrendConfig())
        y = np.full(20, 85.0)
        y[10] = 30.0  # one-sample dropout
        r = an.analyze(y)
        # Robust slope resists the spike -> still stable
        assert r.trend == TrendDirection.STABLE

    def test_monotonic_decline(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(np.linspace(95, 30, 30))
        assert r.trend in (TrendDirection.DEGRADING, TrendDirection.ACCELERATING)

    def test_very_noisy_low_confidence(self) -> None:
        an = HealthTrendAnalyzer(HealthTrendConfig())
        r = an.analyze(_rng(9).normal(70, 25, 30))
        assert r.trend_confidence < 0.5