"""Tests for the enterprise deployment-readiness validator."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from deployment_readiness import DeploymentReadiness, main  # noqa: E402
from validate_repository import Status  # noqa: E402


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build_full_repo(root):
    """A repository that should be fully deployment-ready."""
    write(os.path.join(root, "src", "monitoring", "__init__.py"), '"""Monitoring."""\n')
    write(os.path.join(root, "Dockerfile"),
          "FROM python:3.12-slim\nENV APP_ENV=production\nHEALTHCHECK CMD curl -f http://localhost:8000/health\nCMD [\"python\", \"-m\", \"app\"]\n")
    write(os.path.join(root, "k8s", "deployment.yaml"),
          "kind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n        - name: app\n          livenessProbe:\n            httpGet:\n              path: /health\n")
    write(os.path.join(root, "configs", "release_policy.yaml"),
          "rollback_policy:\n  enabled: true\n  automatic_on_failure: true\n  retain_previous_versions: 3\n"
          "approval_policy:\n  required_approvals: 2\n")
    write(os.path.join(root, "configs", "app.yaml"), "service: app\n")
    write(os.path.join(root, "SECURITY.md"), "# Security Policy\n")
    write(os.path.join(root, ".env.example"), "APP_ENV=production\n")
    write(os.path.join(root, "CHANGELOG.md"), "# Changelog\n")
    return root


def build_empty_repo(root):
    write(os.path.join(root, "README.md"), "# Empty\n")
    return root


# --------------------------------------------------------------------------- #
# Docker
# --------------------------------------------------------------------------- #
def test_docker_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_docker().status is Status.PASS


def test_docker_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_docker().status is Status.FAIL


def test_docker_partial_warning():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "Dockerfile"), "FROM python:3.12\n")
        assert DeploymentReadiness(d).check_docker().status is Status.WARNING


# --------------------------------------------------------------------------- #
# Kubernetes
# --------------------------------------------------------------------------- #
def test_kubernetes_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_kubernetes().status is Status.PASS


def test_kubernetes_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_kubernetes().status is Status.FAIL


def test_kubernetes_kind_marker():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "manifests", "svc.yaml"), "kind: Service\n")
        assert DeploymentReadiness(d).check_kubernetes().status is Status.PASS


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
def test_monitoring_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_monitoring().status is Status.PASS


def test_monitoring_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_monitoring().status is Status.FAIL


# --------------------------------------------------------------------------- #
# Health endpoints
# --------------------------------------------------------------------------- #
def test_health_pass_via_dockerfile():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_health_endpoints().status is Status.PASS


def test_health_pass_via_config():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "configs", "health.yaml"), "path: /health\n")
        assert DeploymentReadiness(d).check_health_endpoints().status is Status.PASS


def test_health_missing_warning():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_health_endpoints().status is Status.WARNING


# --------------------------------------------------------------------------- #
# Configuration files
# --------------------------------------------------------------------------- #
def test_configuration_files_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_configuration_files().status is Status.PASS


def test_configuration_files_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_configuration_files().status is Status.FAIL


# --------------------------------------------------------------------------- #
# Environment variables
# --------------------------------------------------------------------------- #
def test_environment_pass_via_example():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_environment_variables().status is Status.PASS


def test_environment_pass_via_dockerfile_env():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "Dockerfile"), "FROM x\nENV A=1\nCMD [\"run\"]\n")
        assert DeploymentReadiness(d).check_environment_variables().status is Status.PASS


def test_environment_missing_warning():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_environment_variables().status is Status.WARNING


# --------------------------------------------------------------------------- #
# Release manifests
# --------------------------------------------------------------------------- #
def test_release_manifests_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_release_manifests().status is Status.PASS


def test_release_manifests_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_release_manifests().status is Status.FAIL


# --------------------------------------------------------------------------- #
# Security configuration
# --------------------------------------------------------------------------- #
def test_security_pass_via_file():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_security_configuration().status is Status.PASS


def test_security_pass_via_policy():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "configs", "release_policy.yaml"),
              "approval_policy:\n  required_approvals: 2\n")
        assert DeploymentReadiness(d).check_security_configuration().status is Status.PASS


def test_security_missing_warning():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_security_configuration().status is Status.WARNING


# --------------------------------------------------------------------------- #
# Rollback / recovery
# --------------------------------------------------------------------------- #
def test_rollback_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_rollback_support().status is Status.PASS


def test_rollback_disabled_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "configs", "release_policy.yaml"),
              "rollback_policy:\n  enabled: false\n")
        assert DeploymentReadiness(d).check_rollback_support().status is Status.FAIL


def test_rollback_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_rollback_support().status is Status.FAIL


def test_recovery_pass():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).check_recovery_readiness().status is Status.PASS


def test_recovery_partial_warning():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "configs", "release_policy.yaml"),
              "rollback_policy:\n  retain_previous_versions: 3\n  automatic_on_failure: false\n")
        assert DeploymentReadiness(d).check_recovery_readiness().status is Status.WARNING


def test_recovery_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).check_recovery_readiness().status is Status.FAIL


# --------------------------------------------------------------------------- #
# Aggregation / percentage
# --------------------------------------------------------------------------- #
def test_full_repo_high_readiness():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).readiness_fraction() >= 0.9


def test_empty_repo_low_readiness():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).readiness_fraction() <= 0.3


def test_percentage_is_fraction_times_100():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        dr = DeploymentReadiness(d)
        assert dr.readiness_percentage() == pytest.approx(dr.readiness_fraction() * 100.0, abs=0.01)


def test_evaluate_has_ten_checks():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert len(DeploymentReadiness(d).evaluate().results) == 10


def test_percentage_bounds():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert 0.0 <= DeploymentReadiness(d).readiness_percentage() <= 100.0


# --------------------------------------------------------------------------- #
# Summary / determinism / json / CLI
# --------------------------------------------------------------------------- #
def test_summary_structure():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        summary = DeploymentReadiness(d).summary()
        assert set(summary) >= {"readiness_percentage", "overall_status", "passed", "warnings", "failed", "report"}


def test_summary_overall_status_pass_on_full_repo():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).summary()["overall_status"] in ("PASS", "WARNING")


def test_deterministic():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert DeploymentReadiness(d).evaluate().to_dict() == DeploymentReadiness(d).evaluate().to_dict()


def test_report_json_serializable():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert json.dumps(DeploymentReadiness(d).summary())


def test_cli_runs():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        assert main(["--root", d, "--quiet"]) == 0


def test_cli_writes_output():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        out = os.path.join(d, "dep.json")
        main(["--root", d, "--output", out, "--quiet"])
        with open(out) as fh:
            assert "readiness_percentage" in json.load(fh)


def test_cli_empty_repo_returns_one():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert main(["--root", d, "--quiet"]) == 1


# --------------------------------------------------------------------------- #
# Additional coverage: config injection, variants, names
# --------------------------------------------------------------------------- #
def test_config_injection_overrides_file():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        cfg = {"rollback_policy": {"enabled": True}}
        assert DeploymentReadiness(d, config=cfg).check_rollback_support().status is Status.PASS


def test_health_via_k8s_liveness():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "k8s", "deploy.yaml"), "kind: Deployment\nspec:\n  livenessProbe: {}\n")
        assert DeploymentReadiness(d).check_health_endpoints().status is Status.PASS


def test_security_via_config_file():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "configs", "security.yaml"), "policy: strict\n")
        assert DeploymentReadiness(d).check_security_configuration().status is Status.PASS


def test_environment_via_policy_section():
    with tempfile.TemporaryDirectory() as d:
        cfg = {"environment": {"APP_ENV": "prod"}}
        assert DeploymentReadiness(d, config=cfg).check_environment_variables().status is Status.PASS


def test_release_manifest_via_changelog():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "CHANGELOG.md"), "# Changelog\n")
        assert DeploymentReadiness(d).check_release_manifests().status is Status.PASS


def test_evaluate_check_names():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        names = {r.name for r in DeploymentReadiness(d).evaluate().results}
        assert "docker_readiness" in names and "recovery_readiness" in names


def test_overall_status_fail_on_empty():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert DeploymentReadiness(d).evaluate().overall_status is Status.FAIL


def test_readiness_fraction_bounds_empty():
    with tempfile.TemporaryDirectory() as d:
        build_empty_repo(d)
        assert 0.0 <= DeploymentReadiness(d).readiness_fraction() <= 1.0


def test_docker_env_check_independent_of_health():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "Dockerfile"), "FROM x\nENV A=1\nCMD [\"r\"]\n")
        dr = DeploymentReadiness(d)
        assert dr.check_environment_variables().status is Status.PASS


def test_summary_percentage_matches():
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        dr = DeploymentReadiness(d)
        assert dr.summary()["readiness_percentage"] == dr.readiness_percentage()


def test_recovery_retain_only_warning():
    with tempfile.TemporaryDirectory() as d:
        cfg = {"rollback_policy": {"retain_previous_versions": 2, "automatic_on_failure": False}}
        assert DeploymentReadiness(d, config=cfg).check_recovery_readiness().status is Status.WARNING


def test_report_roundtrip_via_dict():
    from validate_repository import ValidationReport
    with tempfile.TemporaryDirectory() as d:
        build_full_repo(d)
        report = DeploymentReadiness(d).evaluate()
        assert ValidationReport.from_dict(report.to_dict()).to_dict() == report.to_dict()