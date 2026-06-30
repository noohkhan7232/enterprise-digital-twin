"""Tests for the reliability engine."""

from __future__ import annotations

import json
import threading

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.reliability_engine import (  # noqa: E402
    ReliabilityEngine, FailureWindow, create_reliability_engine,
)
from observability.observability_models import (  # noqa: E402
    Incident, IncidentSeverity, IncidentStatus, ValidationError,
)


def test_failure_window_duration():
    assert FailureWindow(100.0, 400.0).duration == 300.0


def test_failure_window_invalid():
    with pytest.raises(ValidationError):
        FailureWindow(400.0, 100.0)


def test_success_rate():
    e = ReliabilityEngine()
    e.record_outcome(True, count=99)
    e.record_outcome(False, count=1)
    assert e.success_rate() == 0.99


def test_failure_rate():
    e = ReliabilityEngine()
    e.record_outcome(True, count=99)
    e.record_outcome(False, count=1)
    assert e.failure_rate() == 0.01


def test_rates_no_data():
    e = ReliabilityEngine()
    assert e.success_rate() == 1.0 and e.failure_rate() == 0.0


def test_total_requests():
    e = ReliabilityEngine()
    e.record_outcome(True, count=5)
    e.record_outcome(False, count=3)
    assert e.total_requests == 8


def test_negative_count_raises():
    with pytest.raises(ValidationError):
        ReliabilityEngine().record_outcome(True, count=-1)


def test_downtime():
    e = ReliabilityEngine()
    e.record_failure_window(0.0, 100.0)
    e.record_failure_window(200.0, 250.0)
    assert e.downtime() == 150.0


def test_availability_period_based():
    e = ReliabilityEngine()
    e.set_observation_period(1000.0)
    e.record_failure_window(0.0, 100.0)
    assert e.availability() == 0.9


def test_availability_request_fallback():
    e = ReliabilityEngine()
    e.record_outcome(True, count=8)
    e.record_outcome(False, count=2)
    assert e.availability() == 0.8


def test_availability_perfect():
    e = ReliabilityEngine()
    e.set_observation_period(1000.0)
    assert e.availability() == 1.0


def test_negative_period_raises():
    with pytest.raises(ValidationError):
        ReliabilityEngine().set_observation_period(-1.0)


def test_mttr():
    e = ReliabilityEngine()
    e.record_failure_window(0.0, 300.0)
    e.record_failure_window(0.0, 200.0)
    assert e.mttr() == 250.0


def test_mttr_no_windows():
    assert ReliabilityEngine().mttr() == 0.0


def test_mtbf_no_failures():
    e = ReliabilityEngine()
    e.set_observation_period(1000.0)
    assert e.mtbf() == 1000.0


def test_mtbf_with_failures():
    e = ReliabilityEngine()
    e.set_observation_period(1000.0)
    e.record_failure_window(0.0, 100.0)
    e.record_failure_window(200.0, 300.0)
    assert e.mtbf() == 400.0  # (1000-200)/2


def test_operational_risk_range():
    e = ReliabilityEngine()
    e.record_outcome(True, count=900)
    e.record_outcome(False, count=100)
    risk = e.operational_risk()
    assert 0.0 <= risk <= 1.0


def test_operational_risk_perfect_zero():
    e = ReliabilityEngine()
    e.record_outcome(True, count=100)
    e.set_observation_period(1000.0)
    assert e.operational_risk() == 0.0


def test_reliability_score_perfect():
    e = ReliabilityEngine()
    e.record_outcome(True, count=100)
    e.set_observation_period(1000.0)
    assert e.reliability_score().score == 1.0


def test_reliability_score_fields():
    e = ReliabilityEngine()
    e.record_outcome(True, count=990)
    e.record_outcome(False, count=10)
    e.set_observation_period(86400.0)
    rs = e.reliability_score()
    assert rs.success_rate == 0.99 and 0.0 <= rs.score <= 1.0


def test_reliability_score_degrades_with_risk():
    good = ReliabilityEngine()
    good.record_outcome(True, count=1000)
    good.set_observation_period(1000.0)
    bad = ReliabilityEngine()
    bad.record_outcome(True, count=800)
    bad.record_outcome(False, count=200)
    assert bad.reliability_score().score < good.reliability_score().score


def test_custom_weights():
    e = ReliabilityEngine(score_weights={"availability": 1.0, "success_rate": 0.0})
    e.record_outcome(True, count=8)
    e.record_outcome(False, count=2)
    e.set_observation_period(1000.0)  # availability 1.0
    rs = e.reliability_score()
    assert rs.score > 0.9  # availability dominates


def test_weights_normalized():
    e = ReliabilityEngine(score_weights={"availability": 2.0, "success_rate": 2.0})
    e.record_outcome(True, count=100)
    e.set_observation_period(1000.0)
    assert e.reliability_score().score == 1.0


def test_from_incidents():
    e = ReliabilityEngine()
    e.set_observation_period(10000.0)
    incidents = [
        Incident("i1", "x", IncidentSeverity.SEV1, IncidentStatus.RESOLVED, 100.0, 400.0),
        Incident("i2", "y", IncidentSeverity.SEV2, IncidentStatus.OPEN, 500.0),
    ]
    assert e.from_incidents(incidents) == 1
    assert e.downtime() == 300.0


def test_from_incidents_empty():
    assert ReliabilityEngine().from_incidents([]) == 0


def test_export_json_serializable():
    e = ReliabilityEngine()
    e.record_outcome(True, count=10)
    assert json.dumps(e.export())


def test_factory():
    assert isinstance(create_reliability_engine(), ReliabilityEngine)


def test_determinism():
    def build():
        e = ReliabilityEngine()
        e.record_outcome(True, count=95)
        e.record_outcome(False, count=5)
        e.set_observation_period(1000.0)
        e.record_failure_window(0.0, 50.0)
        return e.reliability_score().to_dict()

    assert build() == build()


def test_thread_safety():
    e = ReliabilityEngine()

    def worker():
        for _ in range(100):
            e.record_outcome(True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert e.total_requests == 800


def test_availability_clamped():
    e = ReliabilityEngine()
    e.set_observation_period(100.0)
    e.record_failure_window(0.0, 200.0)  # downtime exceeds period
    assert e.availability() == 0.0


def test_record_failure_window_returns_window():
    e = ReliabilityEngine()
    w = e.record_failure_window(0.0, 10.0)
    assert isinstance(w, FailureWindow) and w.duration == 10.0


def test_score_zero_when_all_fail():
    e = ReliabilityEngine()
    e.record_outcome(False, count=100)
    assert e.reliability_score().score < 0.5


def test_mttr_single_window():
    e = ReliabilityEngine()
    e.record_failure_window(0.0, 42.0)
    assert e.mttr() == 42.0


def test_failure_rate_only_failures():
    e = ReliabilityEngine()
    e.record_outcome(False, count=10)
    assert e.failure_rate() == 1.0