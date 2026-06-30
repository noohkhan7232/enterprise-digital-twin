"""Tests for the SLI/SLO engine."""

from __future__ import annotations

import json

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.sli_slo_engine import SLISLOEngine, create_sli_slo_engine  # noqa: E402
from observability.observability_models import SLO, SLI, SLIType, ValidationError  # noqa: E402


def engine_with_slos():
    e = SLISLOEngine()
    e.register_slo(SLO("availability", SLIType.AVAILABILITY, 0.99, 2592000.0))
    e.register_slo(SLO("latency", SLIType.LATENCY, 250.0, 3600.0, comparison="lte"))
    e.register_slo(SLO("error_rate", SLIType.ERROR_RATE, 0.01, 3600.0, comparison="lte"))
    e.register_slo(SLO("freshness", SLIType.FRESHNESS, 300.0, 3600.0, comparison="lte"))
    return e


def test_availability_sli():
    assert SLISLOEngine.availability_sli("a", 9990, 10000).value == 0.999


def test_availability_sli_zero_total():
    assert SLISLOEngine.availability_sli("a", 0, 0).value == 1.0


def test_availability_sli_invalid():
    with pytest.raises(ValidationError):
        SLISLOEngine.availability_sli("a", 10, 5)


def test_latency_sli():
    sli = SLISLOEngine.latency_sli("l", [float(i) for i in range(1, 101)], 95)
    assert abs(sli.value - 95.05) < 1e-6


def test_latency_sli_type():
    assert SLISLOEngine.latency_sli("l", [1.0, 2.0]).sli_type is SLIType.LATENCY


def test_error_rate_sli():
    assert SLISLOEngine.error_rate_sli("e", 5, 1000).value == 0.005


def test_error_rate_sli_zero():
    assert SLISLOEngine.error_rate_sli("e", 0, 0).value == 0.0


def test_error_rate_sli_invalid():
    with pytest.raises(ValidationError):
        SLISLOEngine.error_rate_sli("e", 10, 5)


def test_freshness_sli():
    assert SLISLOEngine.freshness_sli("f", 120.0).value == 120.0


def test_register_and_get_slo():
    e = engine_with_slos()
    assert e.slo("availability").target == 0.99


def test_get_unknown_slo():
    with pytest.raises(ValidationError):
        SLISLOEngine().slo("missing")


def test_is_compliant_gte():
    e = engine_with_slos()
    assert e.is_compliant("availability", 0.999)
    assert not e.is_compliant("availability", 0.98)


def test_is_compliant_lte():
    e = engine_with_slos()
    assert e.is_compliant("latency", 200.0)
    assert not e.is_compliant("latency", 300.0)


def test_error_budget_gte_healthy():
    e = engine_with_slos()
    eb = e.error_budget("availability", 0.995)
    assert eb.budget_total == 0.01 and eb.budget_consumed == 0.0
    assert eb.budget_remaining == 0.01 and not eb.is_exhausted


def test_error_budget_gte_breach():
    e = engine_with_slos()
    eb = e.error_budget("availability", 0.985)
    assert eb.budget_consumed == 0.005 and eb.budget_remaining == 0.005


def test_error_budget_gte_exhausted():
    e = engine_with_slos()
    eb = e.error_budget("availability", 0.97)
    assert eb.is_exhausted


def test_error_budget_lte_healthy():
    e = engine_with_slos()
    eb = e.error_budget("error_rate", 0.005)
    assert eb.budget_total == 0.01 and eb.budget_consumed == 0.005


def test_error_budget_lte_exhausted():
    e = engine_with_slos()
    eb = e.error_budget("error_rate", 0.02)
    assert eb.budget_consumed == 0.01 and eb.is_exhausted


def test_burn_rate_full_window():
    e = engine_with_slos()
    eb = e.error_budget("availability", 0.985)
    assert eb.burn_rate == 0.5  # consumed half of budget over full window


def test_burn_rate_scales_with_elapsed():
    e = engine_with_slos()
    slow = e.error_budget("availability", 0.985, elapsed_fraction=1.0)
    fast = e.error_budget("availability", 0.985, elapsed_fraction=0.5)
    assert fast.burn_rate > slow.burn_rate


def test_burn_rate_zero_when_healthy():
    e = engine_with_slos()
    assert e.error_budget("availability", 0.999).burn_rate == 0.0


def test_compliance_report_all_compliant():
    e = engine_with_slos()
    rep = e.compliance_report({"availability": 0.999, "latency": 100.0,
                               "error_rate": 0.005, "freshness": 60.0})
    assert rep["compliant"] == 4 and rep["compliance_rate"] == 1.0


def test_compliance_report_partial():
    e = engine_with_slos()
    rep = e.compliance_report({"availability": 0.98, "latency": 100.0,
                               "error_rate": 0.005, "freshness": 60.0})
    assert rep["compliant"] == 3


def test_compliance_report_unevaluated():
    e = engine_with_slos()
    rep = e.compliance_report({"availability": 0.999})
    assert rep["results"]["latency"]["evaluated"] is False


def test_compliance_report_uses_recorded_sli():
    e = engine_with_slos()
    e.record_sli(SLI("availability", SLIType.AVAILABILITY, 0.999))
    rep = e.compliance_report()
    assert rep["results"]["availability"]["evaluated"] is True


def test_compliance_report_override_recorded():
    e = engine_with_slos()
    e.record_sli(SLI("availability", SLIType.AVAILABILITY, 0.999))
    rep = e.compliance_report({"availability": 0.98})
    assert rep["results"]["availability"]["compliant"] is False


def test_compliance_report_slo_count():
    e = engine_with_slos()
    assert e.compliance_report()["slo_count"] == 4


def test_compliance_report_empty():
    rep = SLISLOEngine().compliance_report()
    assert rep["compliance_rate"] == 1.0 and rep["slo_count"] == 0


def test_compliance_report_json_serializable():
    e = engine_with_slos()
    rep = e.compliance_report({"availability": 0.999})
    assert json.dumps(rep)


def test_factory():
    assert isinstance(create_sli_slo_engine(), SLISLOEngine)


def test_error_budget_unknown_slo():
    with pytest.raises(ValidationError):
        SLISLOEngine().error_budget("missing", 0.99)


def test_determinism():
    def build():
        e = engine_with_slos()
        return e.compliance_report({"availability": 0.999, "latency": 100.0,
                                    "error_rate": 0.005, "freshness": 60.0})

    assert build() == build()


def test_compliance_report_includes_budget():
    e = engine_with_slos()
    rep = e.compliance_report({"availability": 0.985})
    assert "budget_remaining" in rep["results"]["availability"]


def test_latency_sli_custom_percentile():
    sli = SLISLOEngine.latency_sli("l", [float(i) for i in range(1, 101)], 50)
    assert sli.value == 50.5