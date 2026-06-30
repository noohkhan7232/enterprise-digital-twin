"""Tests for the production-readiness assessment."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.production_readiness import ProductionReadiness, create_production_readiness  # noqa: E402
from observability.observability_models import Clock, ReadinessLevel  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def full_repo(root):
    for pkg in ("mlops", "monitoring", "observability"):
        write(os.path.join(root, "src", pkg, "__init__.py"), "x = 1\n")
    write(os.path.join(root, "deployment", "kubernetes", "networkpolicy.yaml"), "kind: NetworkPolicy\n")
    write(os.path.join(root, "deployment", "kubernetes", "secret.example.yaml"), "kind: Secret\n")
    write(os.path.join(root, "deployment", "kubernetes", "deployment.yaml"), "runAsNonRoot: true\n")
    write(os.path.join(root, "deployment", "docker", "Dockerfile"), "USER app\n")
    write(os.path.join(root, "deployment", "scripts", "health_check.py"), "x = 1\n")
    for i in range(3):
        write(os.path.join(root, ".github", "workflows", f"wf{i}.yml"), "on: push\n")
    for i in range(4):
        write(os.path.join(root, "docs", f"doc{i}.md"), "# doc\n")
    for i in range(3):
        write(os.path.join(root, "tests", f"test_{i}.py"), "def test_x(): pass\n")
    return root


def pr(root, **kw):
    kw.setdefault("clock", Clock())
    return ProductionReadiness(root, **kw)


# -- real repo ------------------------------------------------------------- #
def test_real_repo_evaluates():
    rep = pr(REPO_ROOT, reliability_score=0.99, tests_passed=1000, tests_failed=0).evaluate()
    assert 0.0 <= rep.score <= 100.0 and len(rep.checks) == 10


def test_real_repo_has_subsystems():
    rep = pr(REPO_ROOT, reliability_score=0.99).evaluate()
    assert rep.check("observability").passed and rep.check("mlops").passed
    assert rep.check("monitoring").passed


def test_real_repo_high_score():
    rep = pr(REPO_ROOT, reliability_score=0.99, tests_passed=1000, tests_failed=0).evaluate()
    assert rep.score >= 80.0


# -- synthetic full repo --------------------------------------------------- #
def test_synthetic_full_repo_exemplary():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        rep = pr(d, reliability_score=0.99, tests_passed=400, tests_failed=0).evaluate()
        assert rep.level in (ReadinessLevel.READY, ReadinessLevel.EXEMPLARY)


def test_synthetic_all_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        rep = pr(d, reliability_score=0.99, tests_passed=400, tests_failed=0).evaluate()
        assert rep.passed == 10


# -- individual area checks ------------------------------------------------ #
def test_architecture_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_architecture().passed


def test_architecture_fail_missing_subsystem():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "mlops", "__init__.py"))
        assert not pr(d).check_architecture().passed


def test_security_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_security().passed


def test_security_fail_no_controls():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_security().passed


def test_reliability_pass_with_score():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d, reliability_score=0.99).check_reliability().passed


def test_reliability_fail_low_score():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert not pr(d, reliability_score=0.5).check_reliability().passed


def test_reliability_no_score_uses_presence():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "observability", "reliability_engine.py"), "x=1\n")
        assert pr(d).check_reliability().passed


def test_monitoring_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_monitoring().passed


def test_monitoring_fail():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_monitoring().passed


def test_deployment_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_deployment().passed


def test_deployment_fail():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_deployment().passed


def test_cicd_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_cicd().passed


def test_cicd_fail():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_cicd().passed


def test_testing_pass_with_results():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d, tests_passed=400, tests_failed=0).check_testing().passed


def test_testing_fail_with_failures():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert not pr(d, tests_passed=390, tests_failed=10).check_testing().passed


def test_testing_presence_only():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_testing().passed


def test_documentation_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_documentation().passed


def test_documentation_fail_too_few():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "docs", "only.md"), "# x\n")
        assert not pr(d).check_documentation().passed


def test_mlops_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_mlops().passed


def test_mlops_fail():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_mlops().passed


def test_observability_pass():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d).check_observability().passed


def test_observability_fail():
    with tempfile.TemporaryDirectory() as d:
        assert not pr(d).check_observability().passed


# -- scoring / levels ------------------------------------------------------ #
def test_empty_repo_not_ready():
    with tempfile.TemporaryDirectory() as d:
        rep = pr(d, reliability_score=0.0).evaluate()
        assert rep.level is ReadinessLevel.NOT_READY


def test_level_thresholds_monotonic():
    p = ProductionReadiness(".", clock=Clock())
    assert p._level(99.0) is ReadinessLevel.EXEMPLARY
    assert p._level(85.0) is ReadinessLevel.READY
    assert p._level(70.0) is ReadinessLevel.CONDITIONAL
    assert p._level(10.0) is ReadinessLevel.NOT_READY


def test_readiness_score_method():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert pr(d, reliability_score=0.99).readiness_score() >= 80.0


def test_summary_json_serializable():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        assert json.dumps(pr(d, reliability_score=0.99).summary())


def test_custom_weights():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        rep = pr(d, reliability_score=0.99, weights={"architecture": 10.0}).evaluate()
        assert rep.check("architecture").weight == 10.0


def test_custom_thresholds():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        # Raise doc threshold above what the synthetic repo provides (4 docs).
        assert not pr(d, thresholds={"documentation": 5}).check_documentation().passed


def test_determinism():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        a = pr(d, reliability_score=0.99).evaluate().score
        b = pr(d, reliability_score=0.99).evaluate().score
        assert a == b


def test_factory():
    assert isinstance(create_production_readiness("."), ProductionReadiness)


def test_report_summary_structure():
    with tempfile.TemporaryDirectory() as d:
        full_repo(d)
        summary = pr(d, reliability_score=0.99).summary()
        assert "score" in summary and "level" in summary and "checks" in summary