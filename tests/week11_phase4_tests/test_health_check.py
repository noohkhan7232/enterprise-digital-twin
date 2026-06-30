"""Tests for the deployment health check."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "deployment", "scripts"))

from health_check import (  # noqa: E402
    CheckResult, HealthChecker, HealthReport, HealthStatus, aggregate, default_probe, main,
)


# --------------------------------------------------------------------------- #
# Builders / probes
# --------------------------------------------------------------------------- #
def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build(root, *, mlops=True, monitoring=True, configs=True, empty_config=False):
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    if mlops:
        write(os.path.join(root, "src", "mlops", "__init__.py"), "x = 1\n")
    if monitoring:
        write(os.path.join(root, "src", "monitoring", "__init__.py"), "y = 1\n")
    if configs:
        write(os.path.join(root, "configs", "quality_gate.yaml"),
              "" if empty_config else "version: 1\n")
        write(os.path.join(root, "configs", "release_policy.yaml"),
              "" if empty_config else "version: 1\n")
    return root


def healthy_probe(url):
    return True, "HTTP 200"


def failing_probe(url):
    return False, "unreachable"


def checker(root, **kw):
    kw.setdefault("probe", healthy_probe)
    return HealthChecker(root, **kw)


# --------------------------------------------------------------------------- #
# CheckResult / HealthReport value objects
# --------------------------------------------------------------------------- #
def test_check_result_healthy_property():
    assert CheckResult("c", HealthStatus.HEALTHY).healthy is True
    assert CheckResult("c", HealthStatus.DEGRADED).healthy is False


def test_check_result_roundtrip():
    cr = CheckResult("c", HealthStatus.DEGRADED, "msg", {"k": "v"})
    assert CheckResult.from_dict(cr.to_dict()) == cr


def test_check_result_coerces_status():
    assert CheckResult("c", "HEALTHY").status is HealthStatus.HEALTHY


def test_check_result_requires_name():
    with pytest.raises(ValueError):
        CheckResult("", HealthStatus.HEALTHY)


def test_check_result_details_sorted():
    cr = CheckResult("c", HealthStatus.HEALTHY, details={"b": 1, "a": 2})
    assert cr.details == (("a", 2), ("b", 1))


def test_check_result_immutable():
    cr = CheckResult("c", HealthStatus.HEALTHY)
    with pytest.raises(Exception):
        cr.name = "z"  # type: ignore


@pytest.mark.parametrize("statuses,expected", [
    ([], HealthStatus.HEALTHY),
    ([HealthStatus.HEALTHY, HealthStatus.HEALTHY], HealthStatus.HEALTHY),
    ([HealthStatus.HEALTHY, HealthStatus.DEGRADED], HealthStatus.DEGRADED),
    ([HealthStatus.DEGRADED, HealthStatus.UNHEALTHY], HealthStatus.UNHEALTHY),
    ([HealthStatus.HEALTHY, HealthStatus.UNHEALTHY, HealthStatus.DEGRADED], HealthStatus.UNHEALTHY),
])
def test_aggregate(statuses, expected):
    assert aggregate(statuses) is expected


def test_report_overall_and_ready():
    report = HealthReport((CheckResult("a", HealthStatus.HEALTHY),
                           CheckResult("b", HealthStatus.DEGRADED)))
    assert report.overall_status is HealthStatus.DEGRADED
    assert report.ready is True
    assert report.healthy is False


def test_report_unhealthy_not_ready():
    report = HealthReport((CheckResult("a", HealthStatus.UNHEALTHY),))
    assert report.ready is False


def test_report_result_lookup():
    report = HealthReport((CheckResult("a", HealthStatus.HEALTHY),))
    assert report.result("a").status is HealthStatus.HEALTHY
    assert report.result("missing") is None


def test_report_summary_counts():
    report = HealthReport((CheckResult("a", HealthStatus.HEALTHY),
                           CheckResult("b", HealthStatus.DEGRADED),
                           CheckResult("c", HealthStatus.UNHEALTHY)))
    assert report.to_dict()["summary"] == {"healthy": 1, "degraded": 1, "unhealthy": 1}


def test_report_json_roundtrip():
    report = HealthReport((CheckResult("a", HealthStatus.HEALTHY, "ok"),))
    assert HealthReport.from_dict(json.loads(report.to_json())).to_dict() == report.to_dict()


# --------------------------------------------------------------------------- #
# Healthy deployment
# --------------------------------------------------------------------------- #
def test_healthy_deployment_overall():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        report = checker(d, endpoint="http://svc/health").evaluate()
        assert report.healthy and report.ready


def test_healthy_deployment_all_checks():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        report = checker(d, endpoint="http://svc/health").evaluate()
        assert all(c.healthy for c in report.results)


def test_application_check_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d).check_application().status is HealthStatus.HEALTHY


def test_mlops_check_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d).check_mlops().status is HealthStatus.HEALTHY


def test_monitoring_check_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d).check_monitoring().status is HealthStatus.HEALTHY


def test_endpoint_check_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d, endpoint="http://svc/health").check_health_endpoint().status is HealthStatus.HEALTHY


def test_configuration_check_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d).check_configuration().status is HealthStatus.HEALTHY


def test_five_checks_present():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        names = {c.name for c in checker(d).evaluate().results}
        assert names == {"application", "mlops", "monitoring", "health_endpoint", "configuration"}


# --------------------------------------------------------------------------- #
# Unhealthy deployment
# --------------------------------------------------------------------------- #
def test_failing_probe_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        report = HealthChecker(d, endpoint="http://svc/health", probe=failing_probe).evaluate()
        assert report.overall_status is HealthStatus.UNHEALTHY
        assert not report.ready


def test_missing_mlops_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d, mlops=False)
        assert checker(d).check_mlops().status is HealthStatus.UNHEALTHY


def test_missing_monitoring_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d, monitoring=False)
        assert checker(d).check_monitoring().status is HealthStatus.UNHEALTHY


def test_no_src_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        assert checker(d).check_application().status is HealthStatus.UNHEALTHY


def test_empty_src_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        assert checker(d).check_application().status is HealthStatus.UNHEALTHY


def test_package_without_init_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d, mlops=False)
        os.makedirs(os.path.join(d, "src", "mlops"))  # dir without __init__.py
        assert checker(d).check_mlops().status is HealthStatus.UNHEALTHY


# --------------------------------------------------------------------------- #
# Missing / degraded configuration
# --------------------------------------------------------------------------- #
def test_missing_configuration_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d, configs=False)
        assert checker(d).check_configuration().status is HealthStatus.UNHEALTHY


def test_partial_configuration_unhealthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        os.remove(os.path.join(d, "configs", "release_policy.yaml"))
        result = checker(d).check_configuration()
        assert result.status is HealthStatus.UNHEALTHY
        assert "release_policy.yaml" in dict(result.details)["missing"]


def test_empty_configuration_degraded():
    with tempfile.TemporaryDirectory() as d:
        build(d, empty_config=True)
        assert checker(d).check_configuration().status is HealthStatus.DEGRADED


def test_no_required_configs_healthy():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        hc = HealthChecker(d, required_configs=(), probe=healthy_probe)
        assert hc.check_configuration().status is HealthStatus.HEALTHY


# --------------------------------------------------------------------------- #
# Endpoint behaviour / degraded
# --------------------------------------------------------------------------- #
def test_no_endpoint_degraded():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        report = checker(d).evaluate()
        assert report.result("health_endpoint").status is HealthStatus.DEGRADED
        assert report.ready and not report.healthy


def test_probe_receives_endpoint():
    seen = {}

    def spy(url):
        seen["url"] = url
        return True, "ok"

    with tempfile.TemporaryDirectory() as d:
        build(d)
        HealthChecker(d, endpoint="http://svc:8080/health", probe=spy).check_health_endpoint()
    assert seen["url"] == "http://svc:8080/health"


def test_default_probe_unreachable_fails_closed():
    ok, detail = default_probe("http://127.0.0.1:0/health", timeout=0.05)
    assert ok is False
    assert "unreachable" in detail


def test_custom_required_packages():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        write(os.path.join(d, "src", "extra", "__init__.py"), "z = 1\n")
        hc = HealthChecker(d, required_packages=("extra",), probe=healthy_probe)
        assert hc.evaluate().result("application").status is HealthStatus.HEALTHY


# --------------------------------------------------------------------------- #
# Determinism / JSON output
# --------------------------------------------------------------------------- #
def test_evaluate_deterministic():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        a = checker(d, endpoint="http://svc").evaluate()
        b = checker(d, endpoint="http://svc").evaluate()
        assert a.to_dict() == b.to_dict()


def test_report_json_serializable():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert json.dumps(checker(d, endpoint="http://svc").evaluate().to_dict())


def test_report_to_json_sorted_keys():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        payload = checker(d).evaluate().to_json()
        assert payload == json.dumps(json.loads(payload), indent=2, sort_keys=True)


def test_report_contains_ready_and_healthy_keys():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        d_ = checker(d).evaluate().to_dict()
        assert "ready" in d_ and "healthy" in d_ and "overall_status" in d_


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_healthy_returns_zero():
    # No endpoint -> DEGRADED -> still ready -> exit 0.
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert main(["--root", d, "--quiet"]) == 0


def test_cli_unhealthy_returns_one():
    with tempfile.TemporaryDirectory() as d:
        build(d, mlops=False)
        assert main(["--root", d, "--quiet"]) == 1


def test_cli_writes_output():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        out = os.path.join(d, "health.json")
        main(["--root", d, "--output", out, "--quiet"])
        with open(out) as fh:
            payload = json.load(fh)
        assert "overall_status" in payload and "checks" in payload


def test_cli_missing_config_returns_one():
    with tempfile.TemporaryDirectory() as d:
        build(d, configs=False)
        assert main(["--root", d, "--quiet"]) == 1


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_probe_never_raises_in_check():
    def raising(url):
        raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as d:
        build(d)
        # The checker calls the probe; a raising probe would surface here.
        with pytest.raises(RuntimeError):
            HealthChecker(d, endpoint="http://x", probe=raising).check_health_endpoint()


def test_default_probe_is_resilient():
    # default_probe must swallow its own errors and report failure.
    ok, _ = default_probe("not-a-valid-url", timeout=0.05)
    assert ok is False


def test_application_lists_packages():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        details = dict(checker(d).check_application().details)
        assert "mlops" in details["packages"] and "monitoring" in details["packages"]


def test_degraded_endpoint_does_not_block_readiness():
    with tempfile.TemporaryDirectory() as d:
        build(d)
        assert checker(d).evaluate().ready is True


def test_unhealthy_beats_degraded_overall():
    with tempfile.TemporaryDirectory() as d:
        build(d, configs=False)  # config UNHEALTHY, endpoint DEGRADED
        report = checker(d).evaluate()
        assert report.overall_status is HealthStatus.UNHEALTHY