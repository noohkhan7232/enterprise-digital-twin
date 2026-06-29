"""Tests for the prediction monitor, prediction drift and the CLI demo."""

from __future__ import annotations

import json
import os
import tempfile
import threading

import numpy as np
import pytest

from monitoring.monitoring_models import DriftMethod, MonitoringConfiguration, ValidationError
from monitoring.prediction_monitor import (
    PredictionMonitor, create_prediction_monitor, main, run_demo,
)


def rng(seed=7):
    return np.random.default_rng(seed)


def monitor():
    return create_prediction_monitor()


# --------------------------------------------------------------------------- #
# Recording & statistics
# --------------------------------------------------------------------------- #
def test_record_accumulates_count():
    m = monitor()
    m.record(rng().uniform(0, 1, 500))
    assert m.statistics().count == 500


def test_record_multiple_batches():
    m = monitor()
    m.record([0.1, 0.2])
    m.record([0.3, 0.4])
    assert m.statistics().count == 4


def test_statistics_mean_range():
    m = monitor()
    m.record(rng().uniform(0, 1, 1000))
    assert 0.0 <= m.statistics().mean <= 1.0


def test_statistics_variance_nonnegative():
    m = monitor()
    m.record(rng().uniform(0, 1, 1000))
    assert m.statistics().variance >= 0.0


def test_error_and_success_rate():
    m = monitor()
    m.record(rng().uniform(0, 1, 1000), errors=50)
    stats = m.statistics()
    assert stats.error_rate == pytest.approx(0.05)
    assert stats.success_rate == pytest.approx(0.95)


def test_latency_percentiles_ordered():
    m = monitor()
    m.record(rng().uniform(0, 1, 1000), latencies_ms=rng(2).normal(100, 20, 1000))
    stats = m.statistics()
    assert stats.latency_p50_ms <= stats.latency_p95_ms <= stats.latency_p99_ms


def test_throughput_computed():
    m = monitor()
    m.record(rng().uniform(0, 1, 600))
    assert m.statistics(elapsed_seconds=60.0).throughput_per_s == pytest.approx(10.0)


def test_throughput_none_without_elapsed():
    m = monitor()
    m.record(rng().uniform(0, 1, 10))
    assert m.statistics().throughput_per_s is None


def test_confidence_stats_present():
    m = monitor()
    preds = rng().uniform(0, 1, 500)
    m.record(preds, confidences=preds)
    stats = m.statistics()
    assert stats.confidence_mean is not None
    assert stats.confidence_std is not None


def test_inference_time_recorded():
    m = monitor()
    m.record([0.5] * 100, inference_times_ms=[12.0] * 100)
    assert m.statistics().inference_time_ms == pytest.approx(12.0)


def test_positive_rate():
    m = monitor()
    m.record([0.6, 0.7, 0.2, 0.1])
    assert m.statistics().positive_rate == pytest.approx(0.5)


def test_quantiles_present():
    m = monitor()
    m.record(rng().uniform(0, 1, 1000))
    stats = m.statistics()
    assert stats.q05 <= stats.q50 <= stats.q95


def test_reset_clears_state():
    m = monitor()
    m.record([0.5, 0.5])
    m.reset()
    assert m.statistics().count == 0


def test_record_negative_errors_raises():
    m = monitor()
    with pytest.raises(ValidationError):
        m.record([0.5], errors=-1)


def test_empty_statistics():
    assert monitor().statistics().count == 0


def test_compute_statistics_stateless():
    m = monitor()
    stats = m.compute_statistics([0.1, 0.2, 0.3], errors=0)
    assert stats.count == 3
    assert m.statistics().count == 0


def test_compute_statistics_with_total():
    m = monitor()
    stats = m.compute_statistics([0.5] * 90, errors=10, total=100)
    assert stats.error_rate == pytest.approx(0.1)


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000])
def test_statistics_various_sizes(n):
    m = monitor()
    m.record(rng().uniform(0, 1, n))
    assert m.statistics().count == n


# --------------------------------------------------------------------------- #
# Prediction drift
# --------------------------------------------------------------------------- #
def test_prediction_drift_js():
    m = monitor()
    r = m.prediction_drift(rng(1).uniform(0, 1, 2000),
                           np.clip(rng(2).uniform(0, 1, 2000) + 0.3, 0, 1))
    assert r.method is DriftMethod.JS_DISTANCE
    assert r.drift_score >= 0.0


def test_prediction_drift_psi():
    m = monitor()
    r = m.prediction_drift(rng(1).uniform(0, 1, 2000),
                           np.clip(rng(2).uniform(0, 1, 2000) + 0.3, 0, 1), DriftMethod.PSI)
    assert r.method is DriftMethod.PSI


def test_prediction_drift_no_drift():
    m = monitor()
    r = m.prediction_drift(rng(1).uniform(0, 1, 2000), rng(2).uniform(0, 1, 2000))
    assert r.drifted is False


def test_prediction_drift_empty_raises():
    with pytest.raises(ValidationError):
        monitor().prediction_drift([], [0.1])


def test_prediction_drift_unsupported_method():
    with pytest.raises(ValidationError):
        monitor().prediction_drift([0.1, 0.2], [0.3, 0.4], DriftMethod.KS_TEST)


def test_prediction_drift_deterministic():
    m = monitor()
    a, b = rng(1).uniform(0, 1, 1000), rng(2).uniform(0, 1, 1000)
    assert m.prediction_drift(a, b).drift_score == m.prediction_drift(a, b).drift_score


# --------------------------------------------------------------------------- #
# Determinism & threading
# --------------------------------------------------------------------------- #
def test_statistics_deterministic():
    preds = rng().uniform(0, 1, 1000)
    a = monitor(); a.record(preds)
    b = monitor(); b.record(preds)
    assert a.statistics().to_dict() == b.statistics().to_dict()


def test_thread_safe_recording():
    m = monitor()

    def worker():
        for _ in range(50):
            m.record([0.5, 0.6])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.statistics().count == 8 * 50 * 2


# --------------------------------------------------------------------------- #
# Demo & CLI
# --------------------------------------------------------------------------- #
def test_run_demo_structure():
    result = run_demo()
    for key in ("drift_summary", "concept_drift", "prediction_statistics",
                "prediction_drift", "quality", "health", "report", "dashboard",
                "executive_summary"):
        assert key in result


def test_run_demo_deterministic():
    assert json.dumps(run_demo(), sort_keys=True) == json.dumps(run_demo(), sort_keys=True)


def test_run_demo_detects_drift():
    result = run_demo()
    assert result["drift_summary"]["overall_drift_score"] > 0.2


def test_run_demo_top_feature_is_amount():
    result = run_demo()
    assert result["dashboard"]["top_drifted_features"][0][0] == "amount"


def test_run_demo_generates_alerts():
    result = run_demo()
    assert len(result["report"]["alerts"]) >= 1


def test_run_demo_executive_summary_text():
    result = run_demo()
    assert "Executive Monitoring Summary" in result["executive_summary"]


def test_cli_main_demo_returns_zero():
    assert main(["--demo", "--quiet"]) == 0


def test_cli_main_demo_writes_files():
    with tempfile.TemporaryDirectory() as d:
        code = main(["--demo", "--quiet", "--output-dir", d])
        assert code == 0
        for name in ("monitoring_report.json", "dashboard.json", "drift.json", "health.json"):
            assert os.path.exists(os.path.join(d, name))


def test_cli_main_no_args_returns_zero():
    assert main(["--quiet"]) == 0


def test_cli_written_json_valid():
    with tempfile.TemporaryDirectory() as d:
        main(["--demo", "--quiet", "--output-dir", d])
        with open(os.path.join(d, "dashboard.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["model_id"] == "fraud-detector"


def test_factory_with_config():
    cfg = MonitoringConfiguration(error_rate_threshold=0.2)
    m = create_prediction_monitor(config=cfg)
    assert m.config.error_rate_threshold == 0.2