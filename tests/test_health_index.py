#!/usr/bin/env python3
"""Comprehensive test suite for ``src/predictive/health_index.py``.

The Health Index Engine is pure NumPy, so the **entire** suite runs without
PyTorch.  Coverage:

- HealthIndexConfig validation (weights, alpha, gamma, thresholds, window)
- Registry (register / build / list / duplicate rejection)
- HealthStatus banding
- Pure signal mappings (anomaly / fault / deviation), incl. NaN-safety
- Signal fusion (full, subset, weighting, empty)
- HealthState container (to_dict)
- Streaming update (status, contributions, step, timestamp)
- EWMA trend-smoothing behaviour
- Monotonic (degradation-aware) mode
- Degradation-rate estimation + trend classification
- Batch / stateless scoring
- History management (access, bounding, reset, current_index)
- Threshold management
- ExperimentTracker integration (failure-safe)
- Vectorised mappings
- Edge cases (clipping, single update, all-degraded)

Run::

    pytest tests/test_health_index.py -v
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

from src.predictive.health_index import (
    ENGINE_NAME,
    HEALTH_ENGINE_REGISTRY,
    HEALTH_MAX,
    HEALTH_MIN,
    SIGNAL_NAMES,
    HealthIndexConfig,
    HealthIndexEngine,
    HealthState,
    HealthStatus,
    _clip_health,
    anomaly_health,
    build_health_engine,
    deviation_health,
    fault_health,
    list_health_engines,
    raw_health_from_signals,
    register_health_engine,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestHealthIndexConfig:
    """Tests for configuration validation."""

    def test_defaults(self) -> None:
        c = HealthIndexConfig()
        assert 0 < c.smoothing_alpha <= 1
        assert c.critical_threshold <= c.warning_threshold <= c.healthy_threshold

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HealthIndexConfig(weight_anomaly=-0.1)

    def test_all_zero_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HealthIndexConfig(weight_anomaly=0, weight_fault=0, weight_deviation=0)

    def test_bad_alpha_low(self) -> None:
        with pytest.raises(ValueError, match="smoothing_alpha"):
            HealthIndexConfig(smoothing_alpha=0.0)

    def test_bad_alpha_high(self) -> None:
        with pytest.raises(ValueError, match="smoothing_alpha"):
            HealthIndexConfig(smoothing_alpha=1.5)

    def test_bad_gamma(self) -> None:
        with pytest.raises(ValueError, match="fault_gamma"):
            HealthIndexConfig(fault_gamma=0.0)

    def test_bad_window(self) -> None:
        with pytest.raises(ValueError, match="degradation_window"):
            HealthIndexConfig(degradation_window=1)

    def test_bad_history_size(self) -> None:
        with pytest.raises(ValueError, match="history_size"):
            HealthIndexConfig(history_size=0)

    def test_threshold_order_enforced(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            HealthIndexConfig(healthy_threshold=50, warning_threshold=60)

    def test_bad_anomaly_scale(self) -> None:
        with pytest.raises(ValueError, match="anomaly_scale"):
            HealthIndexConfig(anomaly_scale=0.0)

    def test_frozen(self) -> None:
        c = HealthIndexConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.smoothing_alpha = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the health-engine registry."""

    def test_default_engine_registered(self) -> None:
        assert ENGINE_NAME in HEALTH_ENGINE_REGISTRY
        assert ENGINE_NAME in list_health_engines()

    def test_build_by_name(self) -> None:
        engine = build_health_engine(ENGINE_NAME)
        assert isinstance(engine, HealthIndexEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown health engine"):
            build_health_engine("does_not_exist")

    def test_registry_name_stamped(self) -> None:
        assert HealthIndexEngine._registry_name == ENGINE_NAME

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_health_engine(ENGINE_NAME)
            class _Other:
                pass


# ---------------------------------------------------------------------------
# Status banding
# ---------------------------------------------------------------------------


class TestHealthStatus:
    """Tests for status banding."""

    def test_healthy(self) -> None:
        assert HealthStatus.from_index(90, 80, 60, 40) == HealthStatus.HEALTHY

    def test_degrading(self) -> None:
        assert HealthStatus.from_index(70, 80, 60, 40) == HealthStatus.DEGRADING

    def test_warning(self) -> None:
        assert HealthStatus.from_index(50, 80, 60, 40) == HealthStatus.WARNING

    def test_critical(self) -> None:
        assert HealthStatus.from_index(20, 80, 60, 40) == HealthStatus.CRITICAL

    def test_boundary_inclusive(self) -> None:
        # exactly at healthy threshold -> HEALTHY
        assert HealthStatus.from_index(80, 80, 60, 40) == HealthStatus.HEALTHY

    def test_status_is_str(self) -> None:
        assert HealthStatus.HEALTHY.value == "healthy"


# ---------------------------------------------------------------------------
# Pure mappings
# ---------------------------------------------------------------------------


class TestAnomalyHealth:
    """Tests for the anomaly-score mapping."""

    def test_below_threshold_is_full(self) -> None:
        assert anomaly_health(0.01, 0.02) == 100.0

    def test_at_threshold_is_full(self) -> None:
        assert anomaly_health(0.02, 0.02) == 100.0

    def test_above_threshold_decays(self) -> None:
        assert anomaly_health(0.1, 0.02) < 10.0

    def test_monotonic_in_score(self) -> None:
        vals = [anomaly_health(s, 0.02) for s in (0.02, 0.03, 0.05, 0.1)]
        assert vals == sorted(vals, reverse=True)

    def test_nan_safe(self) -> None:
        assert anomaly_health(float("nan"), 0.02) == HEALTH_MIN

    def test_vectorised(self) -> None:
        out = anomaly_health(np.array([0.01, 0.05, 0.1]), 0.02)
        assert isinstance(out, np.ndarray)
        assert out[0] == 100.0

    def test_custom_scale(self) -> None:
        # Larger scale -> slower decay -> higher health at the same score
        slow = anomaly_health(0.05, 0.02, scale=0.1)
        fast = anomaly_health(0.05, 0.02, scale=0.01)
        assert slow > fast


class TestFaultHealth:
    """Tests for the fault-probability mapping."""

    def test_zero_fault_is_full(self) -> None:
        assert fault_health(0.0) == pytest.approx(100.0)

    def test_full_fault_is_zero(self) -> None:
        assert fault_health(1.0) == pytest.approx(0.0)

    def test_linear_default(self) -> None:
        assert fault_health(0.3) == pytest.approx(70.0)

    def test_gamma_strictness(self) -> None:
        strict = fault_health(0.3, gamma=2.0)
        linear = fault_health(0.3, gamma=1.0)
        assert strict < linear

    def test_clips_out_of_range(self) -> None:
        assert fault_health(1.5) == pytest.approx(0.0)
        assert fault_health(-0.5) == pytest.approx(100.0)

    def test_nan_safe(self) -> None:
        assert fault_health(float("nan")) == HEALTH_MIN

    def test_vectorised(self) -> None:
        out = fault_health(np.array([0.0, 0.5, 1.0]))
        assert isinstance(out, np.ndarray)
        assert out[0] == pytest.approx(100.0)


class TestDeviationHealth:
    """Tests for the deviation mapping."""

    def test_zero_deviation_full(self) -> None:
        assert deviation_health(0.0) == 100.0

    def test_full_deviation_zero(self) -> None:
        assert deviation_health(1.0) == 0.0

    def test_linear(self) -> None:
        assert deviation_health(0.25) == 75.0

    def test_nan_safe(self) -> None:
        assert deviation_health(float("nan")) == HEALTH_MIN


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestSignalFusion:
    """Tests for multi-signal fusion."""

    def test_all_three_signals(self) -> None:
        raw, contrib = raw_health_from_signals(
            anomaly_score=0.02, fault_probability=0.1, deviation=0.0,
            anomaly_threshold=0.02,
        )
        assert 0 <= raw <= 100
        assert set(contrib) == {"anomaly_score", "fault_probability", "deviation"}

    def test_subset_fault_only(self) -> None:
        raw, contrib = raw_health_from_signals(fault_probability=0.2)
        assert raw == pytest.approx(80.0)
        assert list(contrib) == ["fault_probability"]

    def test_subset_anomaly_only(self) -> None:
        raw, contrib = raw_health_from_signals(
            anomaly_score=0.01, anomaly_threshold=0.02
        )
        assert raw == pytest.approx(100.0)

    def test_no_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            raw_health_from_signals()

    def test_weighting_matters(self) -> None:
        # When fault health is worse, weighting it higher lowers fused health
        low_fault_w, _ = raw_health_from_signals(
            anomaly_score=0.01, fault_probability=0.8, anomaly_threshold=0.02,
            weight_anomaly=0.9, weight_fault=0.1,
        )
        high_fault_w, _ = raw_health_from_signals(
            anomaly_score=0.01, fault_probability=0.8, anomaly_threshold=0.02,
            weight_anomaly=0.1, weight_fault=0.9,
        )
        assert high_fault_w < low_fault_w

    def test_result_clipped(self) -> None:
        raw, _ = raw_health_from_signals(fault_probability=0.0)
        assert raw <= HEALTH_MAX


# ---------------------------------------------------------------------------
# HealthState
# ---------------------------------------------------------------------------


class TestHealthState:
    """Tests for the state container."""

    def test_to_dict(self) -> None:
        s = HealthState(
            index=90.0, raw_index=88.0, status=HealthStatus.HEALTHY,
            trend="stable", degradation_rate=-0.1,
            contributions={"fault_probability": 90.0}, step=3,
            timestamp="2026-01-01T00:00:00Z",
        )
        d = s.to_dict()
        assert d["index"] == 90.0
        assert d["status"] == "healthy"
        assert d["step"] == 3


# ---------------------------------------------------------------------------
# Streaming update
# ---------------------------------------------------------------------------


class TestStreamingUpdate:
    """Tests for the streaming update path."""

    def _engine(self, **cfg) -> HealthIndexEngine:
        e = HealthIndexEngine(HealthIndexConfig(**cfg))
        e.set_anomaly_threshold(0.02)
        return e

    def test_update_returns_state(self) -> None:
        e = self._engine()
        s = e.update(anomaly_score=0.01)
        assert isinstance(s, HealthState)

    def test_healthy_clip_full_health(self) -> None:
        e = self._engine()
        s = e.update(anomaly_score=0.005)
        assert s.index == 100.0
        assert s.status == HealthStatus.HEALTHY

    def test_critical_clip_low_health(self) -> None:
        e = self._engine()
        s = e.update(fault_probability=0.85)
        assert s.status == HealthStatus.CRITICAL

    def test_step_increments(self) -> None:
        e = self._engine()
        s0 = e.update(fault_probability=0.1)
        s1 = e.update(fault_probability=0.1)
        assert s0.step == 0 and s1.step == 1

    def test_timestamp_present(self) -> None:
        e = self._engine()
        s = e.update(fault_probability=0.1)
        assert isinstance(s.timestamp, str) and len(s.timestamp) > 0

    def test_custom_timestamp(self) -> None:
        e = self._engine()
        s = e.update(fault_probability=0.1, timestamp="2026-06-20T00:00:00Z")
        assert s.timestamp == "2026-06-20T00:00:00Z"

    def test_contributions_recorded(self) -> None:
        e = self._engine()
        s = e.update(anomaly_score=0.03, fault_probability=0.2)
        assert "anomaly_score" in s.contributions
        assert "fault_probability" in s.contributions


# ---------------------------------------------------------------------------
# Smoothing / trend
# ---------------------------------------------------------------------------


class TestSmoothing:
    """Tests for EWMA trend-smoothing."""

    def test_first_value_unsmoothed(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.3))
        s = e.update(fault_probability=0.1)
        assert s.index == pytest.approx(s.raw_index)

    def test_smoothed_lags_raw_on_jump(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.3))
        e.set_anomaly_threshold(0.02)
        e.update(anomaly_score=0.01)   # health ~100
        e.update(anomaly_score=0.01)
        s = e.update(anomaly_score=0.1)  # raw crashes
        # Smoothed should stay above the crashed raw value
        assert s.index > s.raw_index

    def test_lower_alpha_smooths_more(self) -> None:
        def final_index(alpha: float) -> float:
            e = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=alpha))
            e.set_anomaly_threshold(0.02)
            for s in (0.01, 0.01, 0.1):
                state = e.update(anomaly_score=s)
            return state.index
        # Heavier smoothing (lower alpha) retains more of the old high health
        assert final_index(0.1) > final_index(0.9)


# ---------------------------------------------------------------------------
# Monotonic mode
# ---------------------------------------------------------------------------


class TestMonotonicMode:
    """Tests for degradation-aware monotonic mode."""

    def test_never_increases(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(monotonic=True))
        e.set_anomaly_threshold(0.02)
        # Signal recovers mid-stream, but health must not
        idxs = [e.update(anomaly_score=s).index
                for s in (0.01, 0.06, 0.01, 0.01)]
        assert all(idxs[i] >= idxs[i + 1] - 1e-9 for i in range(len(idxs) - 1))

    def test_non_monotonic_can_recover(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(monotonic=False,
                                                smoothing_alpha=1.0))
        e.set_anomaly_threshold(0.02)
        e.update(anomaly_score=0.06)        # degraded
        recovered = e.update(anomaly_score=0.01)  # signal recovers
        # Without monotonic constraint, health recovers
        assert recovered.index > 50

    def test_reset_clears_monotonic_floor(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(monotonic=True))
        e.set_anomaly_threshold(0.02)
        e.update(anomaly_score=0.1)  # drives floor down
        e.reset()
        s = e.update(anomaly_score=0.01)
        assert s.index == 100.0  # floor reset


# ---------------------------------------------------------------------------
# Degradation rate / trend
# ---------------------------------------------------------------------------


class TestDegradation:
    """Tests for degradation-rate estimation and trend."""

    def test_degrading_trend_detected(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.08, 10):
            state = e.update(anomaly_score=float(s))
        assert state.degradation_rate < 0
        assert state.trend == "degrading"

    def test_stable_trend(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        for _ in range(8):
            state = e.update(fault_probability=0.1)
        assert state.trend == "stable"
        assert abs(state.degradation_rate) < 0.5

    def test_improving_trend(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=1.0))
        e.set_anomaly_threshold(0.02)
        # Health rising over time (anomaly score falling)
        for s in np.linspace(0.08, 0.01, 8):
            state = e.update(anomaly_score=float(s))
        assert state.trend == "improving"

    def test_rate_zero_with_single_point(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        state = e.update(fault_probability=0.1)
        assert state.degradation_rate == 0.0


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


class TestBatchScoring:
    """Tests for stateless batch scoring."""

    def test_returns_array(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.02)
        out = e.score_batch(anomaly_scores=[0.01, 0.05, 0.1])
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_ordering(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.02)
        out = e.score_batch(anomaly_scores=[0.01, 0.05, 0.1])
        assert out[0] > out[1] > out[2]

    def test_does_not_change_state(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.02)
        e.score_batch(anomaly_scores=[0.01, 0.05])
        assert len(e.history()) == 0

    def test_multi_signal_batch(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.02)
        out = e.score_batch(
            anomaly_scores=[0.01, 0.05],
            fault_probabilities=[0.1, 0.5],
        )
        assert out.shape == (2,)

    def test_no_signal_raises(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        with pytest.raises(ValueError, match="At least one"):
            e.score_batch()

    def test_length_mismatch_raises(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        with pytest.raises(ValueError, match="length"):
            e.score_batch(anomaly_scores=[0.1, 0.2], fault_probabilities=[0.1])


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    """Tests for history management."""

    def test_history_grows(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        for s in (0.1, 0.2, 0.3):
            e.update(fault_probability=s)
        assert len(e.history()) == 3

    def test_current_index(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.update(fault_probability=0.1)
        assert e.current_index == e.history()[-1]

    def test_current_index_before_update(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        assert e.current_index == HEALTH_MAX

    def test_history_bounded(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(history_size=5))
        for _ in range(20):
            e.update(fault_probability=0.1)
        assert len(e.history()) == 5

    def test_raw_vs_smoothed_history(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.3))
        e.set_anomaly_threshold(0.02)
        for s in (0.01, 0.1, 0.1):
            e.update(anomaly_score=s)
        assert e.history(smoothed=True) != e.history(smoothed=False)

    def test_reset_clears_history(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.update(fault_probability=0.1)
        e.reset()
        assert len(e.history()) == 0
        assert e.current_index == HEALTH_MAX


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------


class TestThreshold:
    """Tests for anomaly-threshold management."""

    def test_set_and_get(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        e.set_anomaly_threshold(0.05)
        assert e.anomaly_threshold == 0.05

    def test_negative_raises(self) -> None:
        e = HealthIndexEngine(HealthIndexConfig())
        with pytest.raises(ValueError, match="threshold"):
            e.set_anomaly_threshold(-0.1)

    def test_threshold_affects_health(self) -> None:
        e_low = HealthIndexEngine(HealthIndexConfig())
        e_low.set_anomaly_threshold(0.01)
        e_high = HealthIndexEngine(HealthIndexConfig())
        e_high.set_anomaly_threshold(0.1)
        # Same score is healthier under the higher threshold
        s_low = e_low.update(anomaly_score=0.05)
        s_high = e_high.update(anomaly_score=0.05)
        assert s_high.index > s_low.index


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_each_update(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append((step, metrics))

        e = HealthIndexEngine(HealthIndexConfig(), experiment_tracker=FakeTracker())
        e.update(fault_probability=0.1)
        e.update(fault_probability=0.2)
        assert len(logged) == 2
        assert "health_index" in logged[0][1]

    def test_broken_tracker_does_not_crash(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        e = HealthIndexEngine(HealthIndexConfig(),
                              experiment_tracker=BrokenTracker())
        state = e.update(fault_probability=0.1)
        assert state.index == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Helpers / edge cases
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper functions and edge cases."""

    def test_clip_health_bounds(self) -> None:
        assert _clip_health(150) == HEALTH_MAX
        assert _clip_health(-5) == HEALTH_MIN
        assert _clip_health(50) == 50.0

    def test_clip_health_nan(self) -> None:
        assert _clip_health(float("nan")) == HEALTH_MIN

    def test_signal_names(self) -> None:
        assert set(SIGNAL_NAMES) == {"anomaly_score", "fault_probability", "deviation"}

    def test_health_bounds_constants(self) -> None:
        assert HEALTH_MAX == 100.0 and HEALTH_MIN == 0.0

    def test_full_degradation_lifecycle(self) -> None:
        # End-to-end: healthy -> degrading -> critical
        e = HealthIndexEngine(HealthIndexConfig(monotonic=True, smoothing_alpha=0.5))
        e.set_anomaly_threshold(0.02)
        statuses = []
        for s in np.concatenate([
            np.full(5, 0.01),                # healthy
            np.linspace(0.02, 0.06, 8),      # degrading
            np.full(5, 0.12),                # critical
        ]):
            statuses.append(e.update(anomaly_score=float(s)).status)
        assert statuses[0] == HealthStatus.HEALTHY
        assert statuses[-1] == HealthStatus.CRITICAL