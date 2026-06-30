"""Final enterprise production-readiness assessment.

Validates ten readiness areas — architecture, security, reliability, monitoring,
deployment, CI/CD, testing, documentation, MLOps and observability — against the
repository and injected operational metrics, and produces a weighted readiness
score in ``[0, 100]`` with a categorical level. Pure Python, deterministic.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .observability_models import (
    Clock, ProductionReport, ReadinessCheck, ReadinessLevel, clamp,
)

__all__ = ["ProductionReadiness", "create_production_readiness"]


def _exists(root: str, *parts: str) -> bool:
    return os.path.exists(os.path.join(root, *parts))


def _isdir(root: str, *parts: str) -> bool:
    return os.path.isdir(os.path.join(root, *parts))


def _list(root: str, *parts: str) -> List[str]:
    path = os.path.join(root, *parts)
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def _read(root: str, *parts: str) -> str:
    path = os.path.join(root, *parts)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


class ProductionReadiness:
    """Assesses production readiness across ten weighted areas."""

    def __init__(self, root: str, *, clock: Optional[Clock] = None,
                 reliability_score: Optional[float] = None,
                 tests_passed: Optional[int] = None, tests_failed: Optional[int] = None,
                 coverage: Optional[float] = None,
                 weights: Optional[Mapping[str, float]] = None,
                 thresholds: Optional[Mapping[str, float]] = None) -> None:
        self.root = root
        self._clock = clock or Clock()
        self.reliability_score = reliability_score
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.coverage = coverage
        self.weights = dict(weights or {})
        self.thresholds = {"reliability": 0.95, "coverage": 0.85, "documentation": 3}
        self.thresholds.update(thresholds or {})
        self._lock = threading.RLock()

    # -- area checks -------------------------------------------------------- #
    def check_architecture(self) -> ReadinessCheck:
        packages = _list(self.root, "src")
        expected = {"mlops", "monitoring", "observability"}
        present = expected & set(packages)
        score = len(present) / len(expected)
        return ReadinessCheck("architecture", score == 1.0, score,
                              f"{len(present)}/{len(expected)} core subsystems present", 1.5)

    def check_security(self) -> ReadinessCheck:
        signals = [
            _exists(self.root, "deployment", "kubernetes", "networkpolicy.yaml"),
            _exists(self.root, "deployment", "kubernetes", "secret.example.yaml"),
            "runAsNonRoot" in _read(self.root, "deployment", "kubernetes", "deployment.yaml"),
            "USER app" in _read(self.root, "deployment", "docker", "Dockerfile"),
        ]
        score = sum(signals) / len(signals)
        return ReadinessCheck("security", score >= 0.75, score,
                              f"{sum(signals)}/{len(signals)} security controls present", 1.5)

    def check_reliability(self) -> ReadinessCheck:
        if self.reliability_score is None:
            present = _exists(self.root, "src", "observability", "reliability_engine.py")
            return ReadinessCheck("reliability", present, 0.7 if present else 0.0,
                                  "reliability engine present; no live score supplied")
        target = float(self.thresholds["reliability"])
        passed = self.reliability_score >= target
        score = clamp(self.reliability_score)
        return ReadinessCheck("reliability", passed, score,
                              f"reliability {self.reliability_score:.3f} vs {target:.3f}", 1.5)

    def check_monitoring(self) -> ReadinessCheck:
        present = _isdir(self.root, "src", "monitoring")
        return ReadinessCheck("monitoring", present, 1.0 if present else 0.0,
                              "monitoring subsystem present" if present else "monitoring missing", 1.0)

    def check_deployment(self) -> ReadinessCheck:
        signals = [_isdir(self.root, "deployment", "docker"),
                   _isdir(self.root, "deployment", "kubernetes"),
                   _exists(self.root, "deployment", "scripts", "health_check.py")]
        score = sum(signals) / len(signals)
        return ReadinessCheck("deployment", score == 1.0, score,
                              f"{sum(signals)}/{len(signals)} deployment assets present", 1.0)

    def check_cicd(self) -> ReadinessCheck:
        workflows = [f for f in _list(self.root, ".github", "workflows") if f.endswith((".yml", ".yaml"))]
        score = clamp(len(workflows) / 3.0)
        return ReadinessCheck("cicd", len(workflows) >= 1, score,
                              f"{len(workflows)} CI/CD workflow(s)", 1.0)

    def check_testing(self) -> ReadinessCheck:
        test_files = [f for f in _list(self.root, "tests") if f.startswith("test_")]
        if self.tests_failed is not None or self.tests_passed is not None:
            failed = self.tests_failed or 0
            passed = self.tests_passed or 0
            if failed == 0 and passed > 0:
                return ReadinessCheck("testing", True, 1.0, f"{passed} tests passed", 1.5)
            score = passed / (passed + failed) if (passed + failed) else 0.0
            return ReadinessCheck("testing", False, score, f"{failed} failing tests", 1.5)
        present = len(test_files) > 0
        return ReadinessCheck("testing", present, 0.8 if present else 0.0,
                              f"{len(test_files)} test file(s); no live results", 1.5)

    def check_documentation(self) -> ReadinessCheck:
        docs = [f for f in _list(self.root, "docs") if f.endswith(".md")]
        threshold = int(self.thresholds["documentation"])
        score = clamp(len(docs) / max(threshold, 1))
        return ReadinessCheck("documentation", len(docs) >= threshold, score,
                              f"{len(docs)} documentation file(s)", 1.0)

    def check_mlops(self) -> ReadinessCheck:
        present = _isdir(self.root, "src", "mlops")
        return ReadinessCheck("mlops", present, 1.0 if present else 0.0,
                              "mlops subsystem present" if present else "mlops missing", 1.0)

    def check_observability(self) -> ReadinessCheck:
        present = _isdir(self.root, "src", "observability")
        return ReadinessCheck("observability", present, 1.0 if present else 0.0,
                              "observability subsystem present" if present else "missing", 1.0)

    # -- aggregation -------------------------------------------------------- #
    def _checks(self) -> List[ReadinessCheck]:
        return [
            self.check_architecture(), self.check_security(), self.check_reliability(),
            self.check_monitoring(), self.check_deployment(), self.check_cicd(),
            self.check_testing(), self.check_documentation(), self.check_mlops(),
            self.check_observability(),
        ]

    def _level(self, score: float) -> ReadinessLevel:
        if score >= 95.0:
            return ReadinessLevel.EXEMPLARY
        if score >= 80.0:
            return ReadinessLevel.READY
        if score >= 60.0:
            return ReadinessLevel.CONDITIONAL
        return ReadinessLevel.NOT_READY

    def evaluate(self) -> ProductionReport:
        with self._lock:
            checks = []
            for c in self._checks():
                weight = float(self.weights.get(c.name, c.weight))
                checks.append(ReadinessCheck(c.name, c.passed, c.score, c.message, weight))
        total_weight = sum(c.weight for c in checks) or 1.0
        score = round(100.0 * sum(c.score * c.weight for c in checks) / total_weight, 4)
        return ProductionReport(self._clock.now(), tuple(checks), score, self._level(score))

    def readiness_score(self) -> float:
        return self.evaluate().score

    def summary(self) -> Dict[str, Any]:
        return self.evaluate().to_dict()


def create_production_readiness(root: str, **kwargs: Any) -> ProductionReadiness:
    return ProductionReadiness(root, **kwargs)