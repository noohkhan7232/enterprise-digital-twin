"""Tests for the metrics engine."""

from __future__ import annotations

import json
import threading

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.metrics_engine import MetricsEngine, create_metrics_engine, MetricCategory  # noqa: E402
from observability.observability_models import Clock, MetricType, ValidationError  # noqa: E402


def engine():
    return MetricsEngine(clock=Clock())


def test_record_returns_point():
    e = engine()
    p = e.record("cpu", 0.5)
    assert p.value == 0.5 and p.timestamp == 0.0


def test_record_assigns_timestamps():
    e = engine()
    e.record("cpu", 1.0)
    e.record("cpu", 2.0)
    assert [p.timestamp for p in e.series("cpu").points] == [0.0, 1.0]


def test_record_explicit_timestamp():
    e = engine()
    e.record("cpu", 1.0, timestamp=99.0)
    assert e.series("cpu").points[0].timestamp == 99.0


def test_record_empty_name_raises():
    with pytest.raises(ValidationError):
        engine().record("", 1.0)


def test_record_unknown_category_raises():
    with pytest.raises(ValidationError):
        engine().record("x", 1.0, category="nope")


@pytest.mark.parametrize("category", list(MetricCategory))
def test_all_categories_accepted(category):
    e = engine()
    e.record("m", 1.0, category=category)
    assert "m" in e.names(category=category)


def test_record_many():
    e = engine()
    assert e.record_many("x", [1.0, 2.0, 3.0]) == 3
    assert e.series("x").count == 3


def test_series_unknown_raises():
    with pytest.raises(ValidationError):
        engine().series("missing")


def test_names_sorted():
    e = engine()
    e.record("b", 1.0)
    e.record("a", 1.0)
    assert e.names() == ("a", "b")


def test_names_by_category():
    e = engine()
    e.record("inf", 1.0, category="inference")
    e.record("app", 1.0, category="application")
    assert e.names(category="inference") == ("inf",)


def test_aggregate_basic():
    e = engine()
    e.record_many("x", [float(i) for i in range(1, 101)])
    agg = e.aggregate("x")
    assert agg["count"] == 100 and agg["min"] == 1.0 and agg["max"] == 100.0


def test_aggregate_percentiles():
    e = engine()
    e.record_many("x", [float(i) for i in range(1, 101)])
    agg = e.aggregate("x")
    assert agg["p50"] == 50.5 and abs(agg["p95"] - 95.05) < 1e-6


def test_aggregate_mean_sum():
    e = engine()
    e.record_many("x", [2.0, 4.0, 6.0])
    agg = e.aggregate("x")
    assert agg["mean"] == 4.0 and agg["sum"] == 12.0


def test_aggregate_stddev():
    e = engine()
    e.record_many("x", [1.0, 1.0, 1.0])
    assert e.aggregate("x")["stddev"] == 0.0


def test_aggregate_empty_series():
    e = engine()
    e.record("x", 1.0)
    w = e.window("x", start=1000.0, end=2000.0)  # excludes the point
    assert w.count == 0


def test_percentiles_method():
    e = engine()
    e.record_many("x", [float(i) for i in range(1, 101)])
    pct = e.percentiles("x", (50.0, 99.0))
    assert "p50" in pct and "p99" in pct


def test_trend_increasing():
    e = engine()
    e.record_many("x", [1.0, 2.0, 3.0, 4.0])
    assert e.trend("x")["direction"] == "increasing"


def test_trend_decreasing():
    e = engine()
    e.record_many("x", [4.0, 3.0, 2.0, 1.0])
    assert e.trend("x")["direction"] == "decreasing"


def test_trend_flat():
    e = engine()
    e.record_many("x", [5.0, 5.0, 5.0])
    assert e.trend("x")["direction"] == "flat"


def test_trend_single_point():
    e = engine()
    e.record("x", 1.0)
    assert e.trend("x")["direction"] == "flat"


def test_trend_delta():
    e = engine()
    e.record_many("x", [1.0, 5.0])
    assert e.trend("x")["delta"] == 4.0


def test_window_filters():
    e = engine()
    for i in range(10):
        e.record("x", float(i), timestamp=float(i))
    w = e.window("x", start=3.0, end=6.0)
    assert w.count == 4


def test_rolling_window():
    e = engine()
    e.record_many("x", [float(i) for i in range(100)])
    rw = e.rolling_window("x", 10)
    assert rw.count == 10 and rw.values[-1] == 99.0


def test_rolling_window_invalid_size():
    e = engine()
    e.record("x", 1.0)
    with pytest.raises(ValidationError):
        e.rolling_window("x", 0)


def test_summary_all_metrics():
    e = engine()
    e.record("a", 1.0)
    e.record("b", 2.0)
    summary = e.summary()
    assert "a" in summary and "b" in summary


def test_category_summary():
    e = engine()
    e.record("inf", 1.0, category="inference")
    cs = e.category_summary()
    assert "inf" in cs["inference"]


def test_export_json_serializable():
    e = engine()
    e.record_many("x", [1.0, 2.0])
    assert json.dumps(e.export())


def test_determinism():
    a = MetricsEngine(clock=Clock())
    b = MetricsEngine(clock=Clock())
    a.record_many("x", [1.0, 2.0, 3.0])
    b.record_many("x", [1.0, 2.0, 3.0])
    assert a.summary() == b.summary()


def test_factory():
    assert isinstance(create_metrics_engine(), MetricsEngine)


def test_large_dataset():
    e = engine()
    e.record_many("x", [float(i) for i in range(10000)])
    agg = e.aggregate("x")
    assert agg["count"] == 10000 and agg["max"] == 9999.0


def test_metric_type_preserved():
    e = engine()
    e.record("x", 1.0, metric_type=MetricType.COUNTER)
    assert e.series("x").metric_type is MetricType.COUNTER


def test_unit_preserved():
    e = engine()
    e.record("x", 1.0, unit="ms")
    assert e.series("x").unit == "ms"


def test_labels_recorded():
    e = engine()
    e.record("x", 1.0, labels={"region": "us"})
    assert dict(e.series("x").points[0].labels)["region"] == "us"


def test_thread_safety():
    e = engine()

    def worker():
        for _ in range(100):
            e.record("x", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert e.series("x").count == 800


def test_p99_of_constant():
    e = engine()
    e.record_many("x", [7.0] * 50)
    assert e.aggregate("x")["p99"] == 7.0


def test_window_open_ended():
    e = engine()
    for i in range(5):
        e.record("x", float(i), timestamp=float(i))
    assert e.window("x", start=2.0).count == 3


def test_rolling_window_larger_than_series():
    e = engine()
    e.record_many("x", [1.0, 2.0])
    assert e.rolling_window("x", 100).count == 2


def test_aggregate_negative_values():
    e = engine()
    e.record_many("x", [-5.0, 0.0, 5.0])
    agg = e.aggregate("x")
    assert agg["min"] == -5.0 and agg["mean"] == 0.0


def test_category_summary_has_all_categories():
    e = engine()
    cs = e.category_summary()
    for c in MetricCategory:
        assert c in cs