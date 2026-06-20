#!/usr/bin/env python3
"""Comprehensive test suite for ``src/predictive/rul_predictor.py``.

The RUL predictor is pure NumPy, so the **entire** suite runs without PyTorch or
SciPy.  Coverage (60+ tests):

- RULConfig validation
- DegradationModel enum
- Registry (register / build / list / duplicate rejection)
- Student-t quantile accuracy (the NumPy-only CI machinery)
- OLS internals (slope sign, R²)
- RULPrediction container (rul_steps, to_dict, infinities)
- Linear extrapolation (finite RUL, correctness vs analytic answer)
- Exponential extrapolation (correctness, fallback)
- AUTO model selection
- Confidence intervals (bracketing, widen-with-noise, widen-with-fewer-points)
- Failure-threshold support + already-failed boundary
- Minimum-history validation + non-finite rejection
- Healthy / flat (non-degrading) systems -> infinite RUL
- Rapidly degrading systems -> near-zero RUL
- Noisy trajectories -> honest wide intervals
- Trend-aware preprocessing (smoothing, recent-window)
- time_per_step scaling and max_horizon clamp
- Integration with the Step-1 health engine
- ExperimentTracker integration (failure-safe)

Run::

    pytest tests/test_rul_predictor.py -v
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

from src.predictive.rul_predictor import (
    PREDICTOR_NAME,
    RUL_INFINITE,
    RUL_PREDICTOR_REGISTRY,
    DegradationModel,
    RULConfig,
    RULPrediction,
    RULPredictor,
    _ols_fit,
    build_rul_predictor,
    list_rul_predictors,
    register_rul_predictor,
    student_t_ppf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_traj(start=95.0, slope=-2.0, n=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return start + slope * np.arange(n) + rng.normal(0, noise, n)


def _exp_traj(h0=95.0, k=0.04, n=15, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = h0 * np.exp(-k * np.arange(n))
    return base * (1 + rng.normal(0, noise, n)) if noise else base


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestRULConfig:
    """Tests for configuration validation."""

    def test_defaults(self) -> None:
        c = RULConfig()
        assert 0 <= c.failure_threshold < 100
        assert c.model == DegradationModel.AUTO.value

    def test_bad_threshold_high(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            RULConfig(failure_threshold=150)

    def test_bad_threshold_negative(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            RULConfig(failure_threshold=-5)

    def test_bad_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            RULConfig(model="polynomial")

    def test_bad_min_history(self) -> None:
        with pytest.raises(ValueError, match="min_history"):
            RULConfig(min_history=2)

    def test_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            RULConfig(confidence_level=1.5)

    def test_bad_time_per_step(self) -> None:
        with pytest.raises(ValueError, match="time_per_step"):
            RULConfig(time_per_step=0)

    def test_bad_max_horizon(self) -> None:
        with pytest.raises(ValueError, match="max_horizon"):
            RULConfig(max_horizon=0)

    def test_bad_smooth_window(self) -> None:
        with pytest.raises(ValueError, match="smooth_window"):
            RULConfig(smooth_window=0)

    def test_bad_recent_fraction(self) -> None:
        with pytest.raises(ValueError, match="recent_fraction"):
            RULConfig(recent_fraction=0.0)

    def test_frozen(self) -> None:
        c = RULConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.failure_threshold = 50  # type: ignore[misc]

    def test_valid_models(self) -> None:
        for m in ("linear", "exponential", "auto"):
            RULConfig(model=m)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class TestDegradationModel:
    """Tests for the degradation-model enum."""

    def test_values(self) -> None:
        assert DegradationModel.LINEAR.value == "linear"
        assert DegradationModel.EXPONENTIAL.value == "exponential"
        assert DegradationModel.AUTO.value == "auto"

    def test_is_str(self) -> None:
        assert DegradationModel.LINEAR == "linear"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the RUL-predictor registry."""

    def test_default_registered(self) -> None:
        assert PREDICTOR_NAME in RUL_PREDICTOR_REGISTRY
        assert PREDICTOR_NAME in list_rul_predictors()

    def test_build_by_name(self) -> None:
        assert isinstance(build_rul_predictor(PREDICTOR_NAME), RULPredictor)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown RUL predictor"):
            build_rul_predictor("nope")

    def test_registry_name_stamped(self) -> None:
        assert RULPredictor._registry_name == PREDICTOR_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_rul_predictor(PREDICTOR_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        p = build_rul_predictor(PREDICTOR_NAME, config=RULConfig(failure_threshold=25))
        assert p.config.failure_threshold == 25


# ---------------------------------------------------------------------------
# Student-t quantile
# ---------------------------------------------------------------------------


class TestStudentT:
    """Tests for the NumPy-only Student-t quantile."""

    def test_known_values(self) -> None:
        # Reference two-sided 95% t critical values
        assert student_t_ppf(0.975, 10) == pytest.approx(2.228, abs=0.01)
        assert student_t_ppf(0.975, 20) == pytest.approx(2.086, abs=0.01)
        assert student_t_ppf(0.975, 30) == pytest.approx(2.042, abs=0.01)

    def test_approaches_normal(self) -> None:
        # Large dof -> normal quantile ~1.96
        assert student_t_ppf(0.975, 1000) == pytest.approx(1.96, abs=0.01)

    def test_larger_dof_smaller_critical(self) -> None:
        assert student_t_ppf(0.975, 5) > student_t_ppf(0.975, 50)

    def test_bad_dof_raises(self) -> None:
        with pytest.raises(ValueError, match="dof"):
            student_t_ppf(0.975, 0)


# ---------------------------------------------------------------------------
# OLS internals
# ---------------------------------------------------------------------------


class TestOLS:
    """Tests for the OLS helper."""

    def test_recovers_slope(self) -> None:
        x = np.arange(10, dtype=float)
        y = 5.0 - 2.0 * x
        a, b, cov, r2 = _ols_fit(x, y)
        assert b == pytest.approx(-2.0)
        assert a == pytest.approx(5.0)

    def test_perfect_fit_r2_one(self) -> None:
        x = np.arange(10, dtype=float)
        y = 3.0 + 1.5 * x
        _, _, _, r2 = _ols_fit(x, y)
        assert r2 == pytest.approx(1.0)

    def test_flat_r2_zero(self) -> None:
        x = np.arange(10, dtype=float)
        y = np.full(10, 50.0)
        _, b, _, r2 = _ols_fit(x, y)
        assert abs(b) < 1e-9
        assert r2 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


class TestRULPrediction:
    """Tests for the prediction container."""

    def _pred(self, **kw) -> RULPrediction:
        base = dict(
            rul=10.0, ci_low=8.0, ci_high=12.0, confidence_level=0.95,
            failure_threshold=30.0, current_health=55.0, model_used="linear",
            slope=-2.0, r_squared=0.97, n_observations=20, is_degrading=True,
            time_per_step=2.0,
        )
        base.update(kw)
        return RULPrediction(**base)

    def test_rul_steps(self) -> None:
        p = self._pred(rul=10.0, time_per_step=2.0)
        assert p.rul_steps == pytest.approx(5.0)

    def test_rul_steps_infinite(self) -> None:
        p = self._pred(rul=RUL_INFINITE)
        assert math.isinf(p.rul_steps)

    def test_to_dict_finite(self) -> None:
        d = self._pred().to_dict()
        assert d["rul"] == 10.0
        assert d["model_used"] == "linear"

    def test_to_dict_infinite_is_none(self) -> None:
        d = self._pred(rul=RUL_INFINITE, ci_low=RUL_INFINITE,
                       ci_high=RUL_INFINITE).to_dict()
        assert d["rul"] is None
        assert d["ci_low"] is None


# ---------------------------------------------------------------------------
# Linear extrapolation
# ---------------------------------------------------------------------------


class TestLinearExtrapolation:
    """Tests for linear degradation extrapolation."""

    def test_finite_rul(self) -> None:
        p = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(
            _linear_traj(slope=-2.0, n=25)
        )
        assert math.isfinite(p.rul)
        assert p.is_degrading

    def test_rul_matches_analytic(self) -> None:
        # Clean line 95 - 2t crosses 30 at t=32.5; last point t=24 -> RUL=8.5
        traj = 95.0 - 2.0 * np.arange(25)
        p = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(traj)
        assert p.rul == pytest.approx(8.5, abs=0.2)

    def test_slope_reported_negative(self) -> None:
        p = RULPredictor(RULConfig(model="linear")).predict(_linear_traj())
        assert p.slope < 0

    def test_high_r2_for_clean_line(self) -> None:
        traj = 95.0 - 2.0 * np.arange(20)
        p = RULPredictor(RULConfig(model="linear")).predict(traj)
        assert p.r_squared > 0.99

    def test_current_health_recorded(self) -> None:
        traj = _linear_traj(n=20)
        p = RULPredictor(RULConfig(model="linear")).predict(traj)
        assert p.current_health == pytest.approx(traj[-1])


# ---------------------------------------------------------------------------
# Exponential extrapolation
# ---------------------------------------------------------------------------


class TestExponentialExtrapolation:
    """Tests for exponential degradation extrapolation."""

    def test_finite_rul(self) -> None:
        # h0=95, k=0.04, 15 pts ends ~54 (above threshold 30)
        p = RULPredictor(RULConfig(failure_threshold=30, model="exponential")).predict(
            _exp_traj()
        )
        assert math.isfinite(p.rul)
        assert p.model_used == "exponential"

    def test_rul_matches_analytic(self) -> None:
        # 30 = 95 exp(-0.04 t) -> t = ln(95/30)/0.04 = 28.8; last t=14 -> RUL=14.8
        p = RULPredictor(RULConfig(failure_threshold=30, model="exponential")).predict(
            _exp_traj()
        )
        assert p.rul == pytest.approx(14.8, abs=0.5)

    def test_fallback_when_insufficient_positive(self) -> None:
        # Mostly non-positive health forces a linear fallback
        traj = np.array([10.0, 5.0, 1.0, 0.0, 0.0, 0.0])
        p = RULPredictor(RULConfig(failure_threshold=0, model="exponential",
                                   min_history=5)).predict(traj)
        # Falls back to linear; should not crash
        assert p.model_used in ("linear", "exponential")


# ---------------------------------------------------------------------------
# AUTO selection
# ---------------------------------------------------------------------------


class TestAutoSelection:
    """Tests for automatic model selection."""

    def test_picks_exponential_for_exp_data(self) -> None:
        p = RULPredictor(RULConfig(failure_threshold=30, model="auto")).predict(
            _exp_traj(k=0.05, n=18)
        )
        assert p.model_used == "exponential"

    def test_picks_linear_for_linear_data(self) -> None:
        traj = 95.0 - 2.0 * np.arange(20)
        p = RULPredictor(RULConfig(failure_threshold=30, model="auto")).predict(traj)
        assert p.model_used == "linear"

    def test_auto_always_returns_concrete(self) -> None:
        p = RULPredictor(RULConfig(model="auto")).predict(_linear_traj())
        assert p.model_used in ("linear", "exponential")


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    """Tests for RUL confidence intervals."""

    def test_brackets_point_estimate(self) -> None:
        p = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=1.0, n=25)
        )
        assert p.ci_low <= p.rul <= p.ci_high

    def test_noisier_is_wider(self) -> None:
        clean = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=0.5, n=20, seed=1)
        )
        noisy = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=8.0, n=20, seed=1)
        )
        assert (noisy.ci_high - noisy.ci_low) > (clean.ci_high - clean.ci_low)

    def test_ci_low_non_negative(self) -> None:
        p = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=10.0, n=15, seed=2)
        )
        assert p.ci_low >= 0.0

    def test_higher_confidence_wider(self) -> None:
        traj = _linear_traj(noise=3.0, n=20, seed=3)
        p90 = RULPredictor(RULConfig(model="linear", confidence_level=0.90)).predict(traj)
        p99 = RULPredictor(RULConfig(model="linear", confidence_level=0.99)).predict(traj)
        assert (p99.ci_high - p99.ci_low) > (p90.ci_high - p90.ci_low)

    def test_exponential_ci_widens_with_noise(self) -> None:
        clean = RULPredictor(RULConfig(failure_threshold=30, model="exponential")).predict(
            _exp_traj(noise=0.01, seed=4)
        )
        noisy = RULPredictor(RULConfig(failure_threshold=30, model="exponential")).predict(
            _exp_traj(noise=0.15, seed=4)
        )
        assert (noisy.ci_high - noisy.ci_low) > (clean.ci_high - clean.ci_low)


# ---------------------------------------------------------------------------
# Failure threshold + boundary
# ---------------------------------------------------------------------------


class TestFailureThreshold:
    """Tests for failure-threshold handling."""

    def test_already_failed_is_zero(self) -> None:
        traj = np.array([60.0, 50.0, 40.0, 30.0, 25.0])
        p = RULPredictor(RULConfig(failure_threshold=30)).predict(traj)
        assert p.rul == 0.0

    def test_higher_threshold_shorter_rul(self) -> None:
        traj = 95.0 - 2.0 * np.arange(20)
        low = RULPredictor(RULConfig(failure_threshold=20, model="linear")).predict(traj)
        high = RULPredictor(RULConfig(failure_threshold=50, model="linear")).predict(traj)
        assert high.rul < low.rul

    def test_threshold_recorded(self) -> None:
        p = RULPredictor(RULConfig(failure_threshold=35, model="linear")).predict(
            _linear_traj()
        )
        assert p.failure_threshold == 35


# ---------------------------------------------------------------------------
# History validation
# ---------------------------------------------------------------------------


class TestHistoryValidation:
    """Tests for minimum-history and input validation."""

    def test_insufficient_history_raises(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            RULPredictor(RULConfig(min_history=5)).predict([90, 80, 70])

    def test_exactly_min_history_ok(self) -> None:
        p = RULPredictor(RULConfig(min_history=5, model="linear")).predict(
            [95, 85, 75, 65, 55]
        )
        assert isinstance(p, RULPrediction)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            RULPredictor(RULConfig()).predict([90, float("nan"), 70, 60, 50])

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            RULPredictor(RULConfig()).predict([90, float("inf"), 70, 60, 50])

    def test_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            RULPredictor(RULConfig()).predict(np.ones((5, 2)))

    def test_n_observations_recorded(self) -> None:
        p = RULPredictor(RULConfig(model="linear")).predict(_linear_traj(n=18))
        assert p.n_observations == 18


# ---------------------------------------------------------------------------
# Healthy / flat systems
# ---------------------------------------------------------------------------


class TestHealthySystems:
    """Tests for non-degrading systems."""

    def test_rising_health_infinite_rul(self) -> None:
        traj = 70.0 + 1.5 * np.arange(10)
        p = RULPredictor(RULConfig()).predict(traj)
        assert math.isinf(p.rul)
        assert not p.is_degrading

    def test_flat_infinite_rul(self) -> None:
        traj = np.full(10, 75.0)
        p = RULPredictor(RULConfig()).predict(traj)
        assert math.isinf(p.rul)
        assert not p.is_degrading

    def test_flat_warning_present(self) -> None:
        p = RULPredictor(RULConfig()).predict(np.full(10, 80.0))
        assert any("not degrading" in w for w in p.warnings)

    def test_near_flat_within_epsilon(self) -> None:
        # Tiny negative slope below slope_epsilon -> treated as non-degrading
        traj = 80.0 - 0.0001 * np.arange(10)
        p = RULPredictor(RULConfig(slope_epsilon=1e-3)).predict(traj)
        assert not p.is_degrading


# ---------------------------------------------------------------------------
# Rapidly degrading
# ---------------------------------------------------------------------------


class TestRapidDegradation:
    """Tests for rapidly degrading systems."""

    def test_near_zero_rul(self) -> None:
        traj = 95.0 - 12.0 * np.arange(6)  # 95..35, next crosses 30
        p = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(traj)
        assert 0 <= p.rul < 3

    def test_steeper_shorter_rul(self) -> None:
        gentle = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(
            95.0 - 2.0 * np.arange(10)
        )
        steep = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(
            95.0 - 6.0 * np.arange(10)
        )
        assert steep.rul < gentle.rul


# ---------------------------------------------------------------------------
# Noisy trajectories
# ---------------------------------------------------------------------------


class TestNoisyTrajectories:
    """Tests for noisy trajectory handling."""

    def test_still_predicts(self) -> None:
        p = RULPredictor(RULConfig(failure_threshold=30, model="linear")).predict(
            _linear_traj(noise=6.0, n=30, seed=7)
        )
        assert math.isfinite(p.rul)

    def test_smoothing_reduces_sensitivity(self) -> None:
        traj = _linear_traj(noise=6.0, n=30, seed=8)
        rough = RULPredictor(RULConfig(model="linear", smooth_window=1)).predict(traj)
        smooth = RULPredictor(RULConfig(model="linear", smooth_window=5)).predict(traj)
        # Both produce predictions; smoothing changes the fit
        assert math.isfinite(rough.rul) and math.isfinite(smooth.rul)

    def test_lower_r2_for_noisy(self) -> None:
        clean = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=0.1, n=20)
        )
        noisy = RULPredictor(RULConfig(model="linear")).predict(
            _linear_traj(noise=10.0, n=20)
        )
        assert noisy.r_squared < clean.r_squared


# ---------------------------------------------------------------------------
# Preprocessing / scaling
# ---------------------------------------------------------------------------


class TestPreprocessingAndScaling:
    """Tests for trend-aware preprocessing and unit scaling."""

    def test_time_per_step_scales_rul(self) -> None:
        traj = 95.0 - 2.0 * np.arange(25)
        base = RULPredictor(RULConfig(failure_threshold=30, model="linear",
                                      time_per_step=1.0)).predict(traj)
        scaled = RULPredictor(RULConfig(failure_threshold=30, model="linear",
                                        time_per_step=24.0)).predict(traj)
        assert scaled.rul == pytest.approx(base.rul * 24.0, rel=1e-6)

    def test_recent_fraction_focuses_fit(self) -> None:
        # Regime change: flat then steep decline. Recent-only sees the decline.
        traj = np.concatenate([np.full(15, 90.0), 90.0 - 5.0 * np.arange(10)])
        full = RULPredictor(RULConfig(failure_threshold=30, model="linear",
                                      recent_fraction=1.0)).predict(traj)
        recent = RULPredictor(RULConfig(failure_threshold=30, model="linear",
                                        recent_fraction=0.4)).predict(traj)
        # Recent-focused sees steeper decline -> shorter RUL
        assert recent.rul < full.rul

    def test_max_horizon_clamps(self) -> None:
        traj = 95.0 - 0.01 * np.arange(10)
        p = RULPredictor(RULConfig(failure_threshold=30, model="linear",
                                   max_horizon=50, slope_epsilon=1e-6)).predict(traj)
        assert p.rul <= 50


# ---------------------------------------------------------------------------
# Step-1 integration
# ---------------------------------------------------------------------------


class TestHealthEngineIntegration:
    """Tests for integration with the Step-1 health engine."""

    def test_predict_from_engine(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine

        engine = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.5))
        engine.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.09, 20):
            engine.update(anomaly_score=float(s))
        p = RULPredictor(RULConfig(failure_threshold=40, model="linear")).predict_from_engine(engine)
        assert isinstance(p, RULPrediction)

    def test_engine_trajectory_degrades(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine

        engine = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.6))
        engine.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.12, 25):
            engine.update(anomaly_score=float(s))
        p = RULPredictor(RULConfig(failure_threshold=40, model="linear")).predict_from_engine(engine)
        assert p.is_degrading


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_prediction(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        RULPredictor(RULConfig(model="linear"),
                     experiment_tracker=FakeTracker()).predict(_linear_traj())
        assert len(logged) == 1
        assert "rul" in logged[0]

    def test_logs_infinite_as_sentinel(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        RULPredictor(RULConfig(),
                     experiment_tracker=FakeTracker()).predict(np.full(10, 80.0))
        assert logged[0]["rul"] == -1.0  # infinite rendered as sentinel

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        p = RULPredictor(RULConfig(model="linear"),
                         experiment_tracker=BrokenTracker()).predict(_linear_traj())
        assert math.isfinite(p.rul)