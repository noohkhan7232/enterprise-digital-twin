"""Tests for the capacity planner."""

from __future__ import annotations

import json

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.capacity_planner import CapacityPlanner, create_capacity_planner  # noqa: E402
from observability.observability_models import ResourceKind, ValidationError  # noqa: E402


def planner(headroom=0.25):
    return CapacityPlanner(headroom=headroom)


def test_linear_forecast_length():
    f = planner().forecast(ResourceKind.CPU, [0.1, 0.2, 0.3], horizon=5)
    assert len(f.forecast) == 5


def test_linear_growth_rate():
    f = planner().forecast(ResourceKind.CPU, [0.40, 0.42, 0.44, 0.46], strategy="linear")
    assert abs(f.growth_rate - 0.02) < 1e-6


def test_linear_increasing():
    f = planner().forecast(ResourceKind.CPU, [1.0, 2.0, 3.0], horizon=3, strategy="linear")
    assert f.forecast[0] < f.forecast[1] < f.forecast[2]


def test_compound_growth_rate():
    f = planner().forecast(ResourceKind.DATA_GROWTH, [100.0, 110.0, 121.0], strategy="compound")
    assert abs(f.growth_rate - 0.1) < 1e-6


def test_compound_forecast():
    f = planner().forecast(ResourceKind.DATA_GROWTH, [100.0, 110.0, 121.0], horizon=2, strategy="compound")
    assert abs(f.forecast[0] - 121.0 * 1.1) < 1e-3


def test_flat_history_zero_growth():
    f = planner().forecast(ResourceKind.MEMORY, [0.5, 0.5, 0.5], horizon=3)
    assert f.growth_rate == 0.0


def test_current_is_last():
    f = planner().forecast(ResourceKind.CPU, [0.1, 0.2, 0.35], horizon=2)
    assert f.current == 0.35


def test_recommended_capacity_headroom():
    f = planner(headroom=0.5).forecast(ResourceKind.CPU, [1.0, 2.0], horizon=2)
    assert abs(f.recommended_capacity - f.peak * 1.5) < 1e-6


def test_exhaustion_detection():
    f = planner().forecast(ResourceKind.CPU, [0.50, 0.52, 0.54], horizon=5,
                           capacity_limit=0.60, strategy="linear")
    assert f.exhaustion_step is not None


def test_no_exhaustion_when_below_limit():
    f = planner().forecast(ResourceKind.CPU, [0.1, 0.11, 0.12], horizon=3, capacity_limit=10.0)
    assert f.exhaustion_step is None


def test_horizon_must_be_positive():
    with pytest.raises(ValidationError):
        planner().forecast(ResourceKind.CPU, [0.1], horizon=0)


def test_empty_history_raises():
    with pytest.raises(ValidationError):
        planner().forecast(ResourceKind.CPU, [], horizon=3)


def test_unknown_strategy_raises():
    with pytest.raises(ValidationError):
        planner().forecast(ResourceKind.CPU, [0.1, 0.2], strategy="magic")


def test_negative_headroom_raises():
    with pytest.raises(ValidationError):
        CapacityPlanner(headroom=-0.1)


def test_forecast_non_negative():
    f = planner().forecast(ResourceKind.CPU, [10.0, 5.0, 1.0], horizon=10, strategy="linear")
    assert all(v >= 0.0 for v in f.forecast)


@pytest.mark.parametrize("resource", list(ResourceKind))
def test_all_resource_kinds(resource):
    f = planner().forecast(resource, [1.0, 2.0, 3.0], horizon=2)
    assert f.resource is resource


def test_forecast_all():
    p = planner()
    out = p.forecast_all({ResourceKind.CPU: [1.0, 2.0], ResourceKind.MEMORY: [3.0, 4.0]}, horizon=3)
    assert "CPU" in out and "MEMORY" in out


def test_export_json_serializable():
    p = planner()
    out = p.forecast_all({ResourceKind.CPU: [1.0, 2.0]}, horizon=2)
    assert json.dumps(p.export(out))


def test_determinism():
    hist = [0.4, 0.42, 0.44, 0.46]
    a = planner().forecast(ResourceKind.CPU, hist, horizon=5, strategy="linear")
    b = planner().forecast(ResourceKind.CPU, hist, horizon=5, strategy="linear")
    assert a.forecast == b.forecast


def test_single_point_history():
    f = planner().forecast(ResourceKind.CPU, [0.5], horizon=3, strategy="linear")
    assert len(f.forecast) == 3


def test_compound_single_point():
    f = planner().forecast(ResourceKind.CPU, [0.5], horizon=2, strategy="compound")
    assert f.growth_rate == 0.0


def test_resource_string_coercion():
    f = planner().forecast("CPU", [1.0, 2.0], horizon=2)
    assert f.resource is ResourceKind.CPU


def test_unit_preserved():
    f = planner().forecast(ResourceKind.STORAGE, [1.0, 2.0], horizon=2, unit="GB")
    assert f.unit == "GB"


def test_peak_property():
    f = planner().forecast(ResourceKind.CPU, [1.0, 2.0, 3.0], horizon=3, strategy="linear")
    assert f.peak == max(f.forecast)


def test_factory():
    assert isinstance(create_capacity_planner(), CapacityPlanner)


def test_decreasing_trend_negative_rate():
    f = planner().forecast(ResourceKind.REQUEST_VOLUME, [10.0, 8.0, 6.0, 4.0], strategy="linear")
    assert f.growth_rate < 0.0


def test_exhaustion_step_zero():
    f = planner().forecast(ResourceKind.CPU, [0.9, 0.95], horizon=3, capacity_limit=0.5,
                           strategy="linear")
    assert f.exhaustion_step == 0


def test_large_horizon():
    f = planner().forecast(ResourceKind.CPU, [0.1, 0.2], horizon=100, strategy="linear")
    assert len(f.forecast) == 100


def test_forecast_all_with_limit():
    p = planner()
    out = p.forecast_all({ResourceKind.CPU: [0.5, 0.6, 0.7]}, horizon=5, capacity_limit=0.8,
                         strategy="linear")
    assert out["CPU"].exhaustion_step is not None