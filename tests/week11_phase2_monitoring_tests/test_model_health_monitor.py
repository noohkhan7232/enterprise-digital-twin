"""Tests for the deterministic model health monitor."""

from __future__ import annotations

import json

import pytest

from monitoring.model_health_monitor import (
    DEFAULT_HEALTH_WEIGHTS, ModelHealthMonitor, create_model_health_monitor,
)
from monitoring.monitoring_models import HealthLevel, MonitoringConfiguration, ValidationError


def hm():
    return create_model_health_monitor()


def healthy_kwargs():
    return dict(accuracy_trend=[0.97, 0.97, 0.98], latency_trend=[40, 40, 42], drift_score=0.02,
                prediction_stability=0.98, availability=0.9999, resource_usage=0.2,
                reliability=0.99, freshness_seconds=60)


def critical_kwargs():
    return dict(accuracy_trend=[0.7, 0.6, 0.5], latency_trend=[300, 400, 500], drift_score=0.9,
                prediction_stability=0.3, availability=0.8, resource_usage=0.95,
                reliability=0.6, freshness_seconds=7200)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("score,level", [
    (0.95, HealthLevel.EXCELLENT),
    (0.90, HealthLevel.EXCELLENT),
    (0.80, HealthLevel.HEALTHY),
    (0.75, HealthLevel.HEALTHY),
    (0.60, HealthLevel.WARNING),
    (0.50, HealthLevel.WARNING),
    (0.40, HealthLevel.CRITICAL),
    (0.0, HealthLevel.CRITICAL),
])
def test_classify_levels(score, level):
    assert hm().classify(score) is level


def test_evaluate_excellent():
    assert hm().evaluate(**healthy_kwargs()).level is HealthLevel.EXCELLENT


def test_evaluate_critical():
    assert hm().evaluate(**critical_kwargs()).level is HealthLevel.CRITICAL


def test_excellent_outscores_critical():
    assert hm().evaluate(**healthy_kwargs()).overall > hm().evaluate(**critical_kwargs()).overall


def test_overall_in_unit_interval():
    score = hm().evaluate(**healthy_kwargs())
    assert 0.0 <= score.overall <= 1.0


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def test_eight_components():
    components = hm().compute_components(**healthy_kwargs())
    assert len(components) == 8


@pytest.mark.parametrize("name", [
    "accuracy_trend", "latency_trend", "prediction_stability", "drift",
    "availability", "resource_usage", "reliability", "freshness",
])
def test_component_present_and_bounded(name):
    components = hm().compute_components(**healthy_kwargs())
    assert name in components
    assert 0.0 <= components[name] <= 1.0


def test_drift_component_inverts_score():
    high = hm().compute_components(drift_score=0.0)["drift"]
    low = hm().compute_components(drift_score=0.9)["drift"]
    assert high > low


def test_resource_component_inverts_usage():
    low_use = hm().compute_components(resource_usage=0.1)["resource_usage"]
    high_use = hm().compute_components(resource_usage=0.9)["resource_usage"]
    assert low_use > high_use


def test_freshness_component_decays():
    fresh = hm().compute_components(freshness_seconds=0)["freshness"]
    stale = hm().compute_components(freshness_seconds=3000)["freshness"]
    assert fresh > stale


def test_declining_accuracy_lowers_component():
    rising = hm().compute_components(accuracy_trend=[0.8, 0.85, 0.9])["accuracy_trend"]
    falling = hm().compute_components(accuracy_trend=[0.9, 0.85, 0.8])["accuracy_trend"]
    assert rising > falling


def test_rising_latency_lowers_component():
    stable = hm().compute_components(latency_trend=[50, 50, 50])["latency_trend"]
    rising = hm().compute_components(latency_trend=[50, 150, 300])["latency_trend"]
    assert stable > rising


def test_missing_inputs_default_healthy():
    components = hm().compute_components()
    assert components["accuracy_trend"] == 1.0
    assert components["latency_trend"] == 1.0
    assert components["freshness"] == 1.0


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one():
    assert sum(hm().weights.values()) == pytest.approx(1.0)


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_HEALTH_WEIGHTS.values()) == pytest.approx(1.0)


def test_custom_weights_normalized():
    monitor = ModelHealthMonitor(weights={"drift": 2.0, "availability": 2.0})
    assert sum(monitor.weights.values()) == pytest.approx(1.0)


def test_zero_weights_rejected():
    with pytest.raises(ValidationError):
        ModelHealthMonitor(weights={"drift": 0.0})


def test_health_score_carries_weights():
    score = hm().evaluate(**healthy_kwargs())
    assert sum(dict(score.weights).values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Status, determinism, serialization
# --------------------------------------------------------------------------- #
def test_status_wraps_score():
    monitor = hm()
    score = monitor.evaluate(**healthy_kwargs())
    status = monitor.status("mdl", score)
    assert status.model_id == "mdl"
    assert status.level is score.level
    assert status.health == score


def test_evaluate_deterministic():
    a = hm().evaluate(**healthy_kwargs())
    b = hm().evaluate(**healthy_kwargs())
    assert a.overall == b.overall


def test_health_score_serializable():
    score = hm().evaluate(**healthy_kwargs())
    assert json.dumps(score.to_dict())


def test_custom_config_thresholds():
    cfg = MonitoringConfiguration(health_excellent=0.95, health_healthy=0.85, health_warning=0.6)
    monitor = create_model_health_monitor(config=cfg)
    assert monitor.classify(0.9) is HealthLevel.HEALTHY


def test_warning_band():
    score = hm().evaluate(accuracy_trend=[0.7, 0.7, 0.7], latency_trend=[200, 210, 220],
                          drift_score=0.4, prediction_stability=0.6, availability=0.95,
                          resource_usage=0.6, reliability=0.85, freshness_seconds=1800)
    assert score.level in (HealthLevel.WARNING, HealthLevel.HEALTHY)