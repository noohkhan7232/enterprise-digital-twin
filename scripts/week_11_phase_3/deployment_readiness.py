"""Enterprise deployment-readiness validator.

Assesses whether the repository is ready to deploy: Docker and Kubernetes
manifests, monitoring wiring, health endpoints, configuration files, environment
variables, release manifests, security configuration, rollback support and
recovery readiness. Produces per-check results and an overall readiness
percentage. Runnable as a CLI::

    python scripts/deployment_readiness.py --root .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_repository import (  # noqa: E402
    CheckResult, Status, ValidationReport, load_yaml, DEFAULT_TIMESTAMP,
)

__all__ = ["DeploymentReadiness", "main"]


def _exists(root: str, *parts: str) -> bool:
    return os.path.exists(os.path.join(root, *parts))


def _read_if(root: str, *parts: str) -> str:
    path = os.path.join(root, *parts)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def _read_dockerfile(root: str) -> str:
    """Read the repository Dockerfile from the root or the deployment layout."""
    return _read_if(root, "Dockerfile") or _read_if(root, "deployment", "docker", "Dockerfile")


def _find_files(root: str, predicate) -> List[str]:
    matches: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__", "_validation"})
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if predicate(name, full):
                matches.append(full)
    return matches


class DeploymentReadiness:
    """Computes deterministic deployment-readiness checks for a repository."""

    def __init__(self, root: str, *, config: Optional[Dict[str, Any]] = None,
                 timestamp: str = DEFAULT_TIMESTAMP) -> None:
        self.root = root
        self.timestamp = timestamp
        policy_path = os.path.join(root, "configs", "release_policy.yaml")
        if config is not None:
            self.policy = dict(config)
        elif os.path.exists(policy_path):
            self.policy = load_yaml(policy_path)
        else:
            self.policy = {}

    # -- checks ------------------------------------------------------------- #
    def check_docker(self) -> CheckResult:
        content = _read_dockerfile(self.root)
        if not content:
            return CheckResult("docker_readiness", Status.FAIL, 0.0, "no Dockerfile")
        has_from = "FROM " in content
        has_entry = "CMD " in content or "ENTRYPOINT " in content
        score = (0.5 if has_from else 0.0) + (0.5 if has_entry else 0.0)
        status = Status.PASS if score == 1.0 else (Status.WARNING if score > 0 else Status.FAIL)
        return CheckResult("docker_readiness", status, score, "Dockerfile present")

    def check_kubernetes(self) -> CheckResult:
        manifests = _find_files(
            self.root,
            lambda n, f: n.endswith((".yaml", ".yml")) and (
                os.sep + "k8s" + os.sep in f or os.sep + "deploy" in f or
                "kind:" in _read_if(os.path.dirname(f), os.path.basename(f))))
        status = Status.PASS if manifests else Status.FAIL
        return CheckResult("kubernetes_readiness", status, 1.0 if manifests else 0.0,
                           f"{len(manifests)} kubernetes manifest(s)")

    def check_monitoring(self) -> CheckResult:
        present = _exists(self.root, "src", "monitoring")
        return CheckResult("monitoring_readiness", Status.PASS if present else Status.FAIL,
                           1.0 if present else 0.0,
                           "monitoring subsystem present" if present else "monitoring missing")

    def check_health_endpoints(self) -> CheckResult:
        signals = (
            "HEALTHCHECK" in _read_dockerfile(self.root) or
            _exists(self.root, "configs", "health.yaml") or
            any("livenessProbe" in _read_if(os.path.dirname(m), os.path.basename(m))
                for m in _find_files(self.root, lambda n, f: n.endswith((".yaml", ".yml"))))
        )
        return CheckResult("health_endpoints", Status.PASS if signals else Status.WARNING,
                           1.0 if signals else 0.0,
                           "health checks configured" if signals else "no health checks")

    def check_configuration_files(self) -> CheckResult:
        config_dir = os.path.join(self.root, "configs")
        configs = ([f for f in os.listdir(config_dir) if f.endswith((".yaml", ".yml"))]
                   if os.path.isdir(config_dir) else [])
        status = Status.PASS if configs else Status.FAIL
        return CheckResult("configuration_files", status, 1.0 if configs else 0.0,
                           f"{len(configs)} configuration file(s)")

    def check_environment_variables(self) -> CheckResult:
        signals = (
            _exists(self.root, ".env.example") or _exists(self.root, ".env.template") or
            "environment" in self.policy or "ENV " in _read_dockerfile(self.root))
        return CheckResult("environment_variables", Status.PASS if signals else Status.WARNING,
                           1.0 if signals else 0.0,
                           "environment documented" if signals else "no environment template")

    def check_release_manifests(self) -> CheckResult:
        signals = (_exists(self.root, "configs", "release_policy.yaml") or
                   _exists(self.root, "CHANGELOG.md") or _exists(self.root, "RELEASE.md"))
        return CheckResult("release_manifests", Status.PASS if signals else Status.FAIL,
                           1.0 if signals else 0.0,
                           "release manifest present" if signals else "no release manifest")

    def check_security_configuration(self) -> CheckResult:
        signals = (_exists(self.root, "SECURITY.md") or _exists(self.root, "configs", "security.yaml")
                   or bool(self.policy.get("approval_policy")))
        return CheckResult("security_configuration", Status.PASS if signals else Status.WARNING,
                           1.0 if signals else 0.0,
                           "security policy present" if signals else "no security policy")

    def check_rollback_support(self) -> CheckResult:
        rollback = self.policy.get("rollback_policy", {})
        enabled = bool(rollback.get("enabled", False))
        return CheckResult("rollback_support", Status.PASS if enabled else Status.FAIL,
                           1.0 if enabled else 0.0,
                           "rollback enabled" if enabled else "rollback not configured")

    def check_recovery_readiness(self) -> CheckResult:
        rollback = self.policy.get("rollback_policy", {})
        retain = int(rollback.get("retain_previous_versions", 0) or 0)
        auto = bool(rollback.get("automatic_on_failure", False))
        score = (0.5 if retain > 0 else 0.0) + (0.5 if auto else 0.0)
        status = Status.PASS if score == 1.0 else (Status.WARNING if score > 0 else Status.FAIL)
        return CheckResult("recovery_readiness", status, score,
                           f"retain={retain}, automatic={auto}")

    # -- aggregation -------------------------------------------------------- #
    def evaluate(self) -> ValidationReport:
        checks = [
            self.check_docker(), self.check_kubernetes(), self.check_monitoring(),
            self.check_health_endpoints(), self.check_configuration_files(),
            self.check_environment_variables(), self.check_release_manifests(),
            self.check_security_configuration(), self.check_rollback_support(),
            self.check_recovery_readiness(),
        ]
        return ValidationReport("deployment_readiness", tuple(checks), self.timestamp)

    def readiness_fraction(self) -> float:
        report = self.evaluate()
        if not report.results:
            return 0.0
        return round(sum(r.score for r in report.results) / len(report.results), 4)

    def readiness_percentage(self) -> float:
        return round(self.readiness_fraction() * 100.0, 2)

    def summary(self) -> Dict[str, Any]:
        report = self.evaluate()
        return {
            "readiness_percentage": self.readiness_percentage(),
            "overall_status": report.overall_status.value,
            "passed": report.passed,
            "warnings": report.warnings,
            "failed": report.failed,
            "report": report.to_dict(),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assess deployment readiness")
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    from validate_repository import _resolve_root  # noqa: E402

    summary = DeploymentReadiness(_resolve_root(args.root)).summary()
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    if not args.quiet:
        print(payload)
    return 0 if summary["overall_status"] != Status.FAIL.value else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())