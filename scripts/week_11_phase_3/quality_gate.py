"""Enterprise quality-gate engine.

Runs a battery of deterministic quality gates over the repository and produces
a quality score in ``[0, 100]`` together with a PASS / WARNING / FAIL verdict.
Each gate is a strategy (a named method with a configured weight); measured
inputs that cannot be derived statically offline (test results, coverage) are
injected through :class:`QualityContext` so the engine stays deterministic and
testable. Runnable as a CLI::

    python scripts/quality_gate.py --root . --coverage 0.92 --tests-passed 809
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_repository import (  # noqa: E402
    CheckResult, RepositoryValidator, Status, ValidationReport, _parse, _read,
    iter_python_files, load_yaml, DEFAULT_TIMESTAMP,
)

__all__ = ["QualityContext", "QualityGate", "load_quality_config", "main"]

_DEFAULT_CONFIG = {
    "thresholds": {
        "coverage": 0.85, "complexity": 15, "documentation": 0.60, "type_hints": 0.70,
        "naming": 0.90, "quality_score": 80.0, "monitoring": 1.0, "mlops": 1.0, "deployment": 0.70,
    },
    "weights": {},
    "required_packages": ["mlops", "monitoring"],
    "required_configs": ["configs/quality_gate.yaml", "configs/release_policy.yaml"],
}


def load_quality_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a quality-gate config, falling back to built-in defaults."""
    config = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
              for k, v in _DEFAULT_CONFIG.items()}
    if path and os.path.exists(path):
        loaded = load_yaml(path)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


class QualityContext:
    """Inputs for a quality-gate evaluation."""

    def __init__(
        self,
        root: str,
        *,
        config: Optional[Mapping[str, Any]] = None,
        coverage: Optional[float] = None,
        tests_passed: Optional[int] = None,
        tests_failed: Optional[int] = None,
        timestamp: str = DEFAULT_TIMESTAMP,
    ) -> None:
        self.root = root
        self.config = dict(config or load_quality_config(
            os.path.join(root, "configs", "quality_gate.yaml")))
        self.thresholds = dict(self.config.get("thresholds", {}))
        self.weights = dict(self.config.get("weights", {}))
        self.coverage = coverage
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.timestamp = timestamp
        self.validator = RepositoryValidator(
            root,
            type_hint_threshold=float(self.thresholds.get("type_hints", 0.7)),
            documentation_threshold=float(self.thresholds.get("documentation", 0.6)),
            complexity_threshold=int(self.thresholds.get("complexity", 15)),
            naming_threshold=float(self.thresholds.get("naming", 0.9)),
            timestamp=timestamp,
        )


class QualityGate:
    """Runs the configured quality gates and computes a quality score."""

    def __init__(self, context: QualityContext) -> None:
        self.ctx = context
        self._gates: List[Tuple[str, Callable[[], CheckResult]]] = [
            ("repository_structure", self.ctx.validator.validate_structure),
            ("python_syntax", self.ctx.validator.validate_python_syntax),
            ("import_validation", self.ctx.validator.validate_imports),
            ("type_hint_coverage", self.ctx.validator.validate_type_hints),
            ("documentation_coverage", self.ctx.validator.validate_documentation),
            ("test_discovery", self.ctx.validator.discover_tests),
            ("pytest_execution", self.gate_pytest_execution),
            ("coverage_threshold", self.gate_coverage_threshold),
            ("code_complexity", self.ctx.validator.validate_complexity),
            ("dependency_validation", self.gate_dependency_validation),
            ("configuration_validation", self.gate_configuration_validation),
            ("package_integrity", self.ctx.validator.validate_package_integrity),
            ("architecture_consistency", self.gate_architecture_consistency),
            ("naming_convention", self.ctx.validator.validate_naming),
            ("json_serialization", self.gate_json_serialization),
            ("thread_safety", self.gate_thread_safety),
            ("deterministic_behaviour", self.gate_deterministic_behaviour),
            ("mlops_integration", self.gate_mlops_integration),
            ("monitoring_integration", self.gate_monitoring_integration),
            ("deployment_readiness", self.gate_deployment_readiness),
        ]

    # -- metric-driven gates ----------------------------------------------- #
    def gate_pytest_execution(self) -> CheckResult:
        if self.ctx.tests_failed is None and self.ctx.tests_passed is None:
            return CheckResult("pytest_execution", Status.WARNING, 0.5,
                               "no test results supplied; run pytest in CI")
        failed = self.ctx.tests_failed or 0
        passed = self.ctx.tests_passed or 0
        if failed == 0 and passed > 0:
            return CheckResult("pytest_execution", Status.PASS, 1.0, f"{passed} passed, 0 failed")
        score = passed / (passed + failed) if (passed + failed) else 0.0
        return CheckResult("pytest_execution", Status.FAIL, score, f"{passed} passed, {failed} failed")

    def gate_coverage_threshold(self) -> CheckResult:
        threshold = float(self.ctx.thresholds.get("coverage", 0.85))
        if self.ctx.coverage is None:
            return CheckResult("coverage_threshold", Status.WARNING, 0.5,
                               "no coverage supplied; measured in CI",
                               details={"threshold": str(threshold)})
        cov = float(self.ctx.coverage)
        status = Status.PASS if cov >= threshold else (
            Status.WARNING if cov >= threshold * 0.9 else Status.FAIL)
        return CheckResult("coverage_threshold", status, min(1.0, cov),
                           f"coverage {cov:.2%} vs threshold {threshold:.2%}")

    # -- static gates ------------------------------------------------------- #
    def gate_dependency_validation(self) -> CheckResult:
        candidates = ["requirements.txt", "pyproject.toml", "setup.cfg"]
        found = [c for c in candidates if os.path.exists(os.path.join(self.ctx.root, c))]
        if not found:
            return CheckResult("dependency_validation", Status.WARNING, 0.5,
                               "no dependency manifest found")
        req = os.path.join(self.ctx.root, "requirements.txt")
        duplicates = 0
        conflicts = 0
        if os.path.exists(req):
            names: Dict[str, str] = {}
            for line in _read(req).splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                name = line.replace(">=", "==").replace("~=", "==").split("==")[0].strip().lower()
                version = line[len(name):]
                if name in names:
                    duplicates += 1
                    if names[name] != version:
                        conflicts += 1
                names[name] = version
        status = Status.PASS if duplicates == 0 and conflicts == 0 else (
            Status.WARNING if conflicts == 0 else Status.FAIL)
        score = 1.0 if status is Status.PASS else (0.6 if status is Status.WARNING else 0.2)
        return CheckResult("dependency_validation", status, score,
                           f"{len(found)} manifest(s), {duplicates} duplicates, {conflicts} conflicts")

    def gate_configuration_validation(self) -> CheckResult:
        required = list(self.ctx.config.get("required_configs", []))
        if not required:
            return CheckResult("configuration_validation", Status.PASS, 1.0, "no required configs")
        ok = 0
        for rel in required:
            path = os.path.join(self.ctx.root, rel)
            if os.path.exists(path):
                try:
                    if load_yaml(path):
                        ok += 1
                except Exception:  # noqa: BLE001
                    pass
        ratio = ok / len(required)
        status = Status.PASS if ratio == 1.0 else (Status.WARNING if ratio >= 0.5 else Status.FAIL)
        return CheckResult("configuration_validation", status, ratio,
                           f"{ok}/{len(required)} configs valid")

    def gate_architecture_consistency(self) -> CheckResult:
        src = os.path.join(self.ctx.root, "src")
        if not os.path.isdir(src):
            return CheckResult("architecture_consistency", Status.WARNING, 0.0, "no src/")
        packages = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
        if not packages:
            return CheckResult("architecture_consistency", Status.WARNING, 0.0, "no packages")
        consistent = 0
        for pkg in packages:
            pkg_dir = os.path.join(src, pkg)
            has_init = os.path.exists(os.path.join(pkg_dir, "__init__.py"))
            modules = [f for f in os.listdir(pkg_dir) if f.endswith(".py") and f != "__init__.py"]
            if has_init and modules:
                consistent += 1
        ratio = consistent / len(packages)
        status = Status.PASS if ratio == 1.0 else Status.WARNING
        return CheckResult("architecture_consistency", status, ratio,
                           f"{consistent}/{len(packages)} packages consistent")

    def gate_json_serialization(self) -> CheckResult:
        dataclasses_total, serializable = 0, 0
        for path in iter_python_files(os.path.join(self.ctx.root, "src")):
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                is_dc = any(
                    (isinstance(d, ast.Name) and d.id == "dataclass") or
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                    for d in node.decorator_list)
                if not is_dc:
                    continue
                dataclasses_total += 1
                methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                if {"to_dict", "from_dict"} <= methods:
                    serializable += 1
        if dataclasses_total == 0:
            return CheckResult("json_serialization", Status.WARNING, 0.5, "no dataclasses found")
        ratio = serializable / dataclasses_total
        status = Status.PASS if ratio >= 0.9 else (Status.WARNING if ratio >= 0.6 else Status.FAIL)
        return CheckResult("json_serialization", status, ratio,
                           f"{serializable}/{dataclasses_total} dataclasses serialisable")

    def gate_thread_safety(self) -> CheckResult:
        needing, guarded = 0, 0
        for path in iter_python_files(os.path.join(self.ctx.root, "src")):
            text = _read(path)
            tree = _parse(path)
            if tree is None:
                continue
            has_class = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            mutates = ".append(" in text or ".extend(" in text or ".pop(" in text or ".clear(" in text
            if has_class and mutates:
                needing += 1
                if "Lock" in text or "RLock" in text:
                    guarded += 1
        if needing == 0:
            return CheckResult("thread_safety", Status.PASS, 1.0, "no shared mutable state detected")
        ratio = guarded / needing
        status = Status.PASS if ratio >= 0.8 else (Status.WARNING if ratio >= 0.5 else Status.FAIL)
        return CheckResult("thread_safety", status, ratio,
                           f"{guarded}/{needing} stateful modules use locks")

    def gate_deterministic_behaviour(self) -> CheckResult:
        relevant, flagged = 0, 0
        for path in iter_python_files(os.path.join(self.ctx.root, "src")):
            text = _read(path)
            relevant += 1
            uses_now = "datetime.now(" in text or "time.time(" in text
            injected_clock = "Clock" in text or "default_rng" in text or "seed" in text
            if uses_now and not injected_clock:
                flagged += 1
        if relevant == 0:
            return CheckResult("deterministic_behaviour", Status.WARNING, 0.5, "no source files")
        ratio = 1.0 - flagged / relevant
        status = Status.PASS if flagged == 0 else (Status.WARNING if flagged <= 2 else Status.FAIL)
        return CheckResult("deterministic_behaviour", status, ratio,
                           f"{flagged} modules use unseeded nondeterminism")

    def _package_present(self, name: str, gate_name: str) -> CheckResult:
        present = os.path.isdir(os.path.join(self.ctx.root, "src", name))
        status = Status.PASS if present else Status.FAIL
        return CheckResult(gate_name, status, 1.0 if present else 0.0,
                           f"package '{name}' {'present' if present else 'missing'}")

    def gate_mlops_integration(self) -> CheckResult:
        return self._package_present("mlops", "mlops_integration")

    def gate_monitoring_integration(self) -> CheckResult:
        return self._package_present("monitoring", "monitoring_integration")

    def gate_deployment_readiness(self) -> CheckResult:
        try:
            from deployment_readiness import DeploymentReadiness  # noqa: E402
        except ImportError:
            return CheckResult("deployment_readiness", Status.WARNING, 0.5, "readiness module unavailable")
        fraction = DeploymentReadiness(self.ctx.root).readiness_fraction()
        threshold = float(self.ctx.thresholds.get("deployment", 0.7))
        status = Status.PASS if fraction >= threshold else (
            Status.WARNING if fraction >= threshold * 0.7 else Status.FAIL)
        return CheckResult("deployment_readiness", status, fraction,
                           f"deployment readiness {fraction:.2%}")

    # -- aggregation -------------------------------------------------------- #
    def evaluate(self) -> ValidationReport:
        results: List[CheckResult] = []
        for name, gate in self._gates:
            result = gate()
            weight = float(self.ctx.weights.get(name, result.weight))
            results.append(CheckResult(result.name, result.status, result.score,
                                       result.message, weight, dict(result.details)))
        return ValidationReport("quality_gate", tuple(results), self.ctx.timestamp)

    def quality_score(self) -> float:
        return self.evaluate().score

    def summary(self) -> Dict[str, Any]:
        report = self.evaluate()
        threshold = float(self.ctx.thresholds.get("quality_score", 80.0))
        score = report.score
        if report.failed > 0 or score < threshold:
            verdict = Status.FAIL.value if report.failed > 0 else Status.WARNING.value
        else:
            verdict = Status.PASS.value
        return {
            "quality_score": score,
            "verdict": verdict,
            "threshold": threshold,
            "passed": report.passed,
            "warnings": report.warnings,
            "failed": report.failed,
            "report": report.to_dict(),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the enterprise quality gate")
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--tests-failed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    from validate_repository import _resolve_root  # noqa: E402

    root = _resolve_root(args.root)
    config = load_quality_config(args.config or os.path.join(root, "configs", "quality_gate.yaml"))
    ctx = QualityContext(root, config=config, coverage=args.coverage,
                         tests_passed=args.tests_passed, tests_failed=args.tests_failed)
    summary = QualityGate(ctx).summary()
    import json

    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    if not args.quiet:
        print(payload)
    return 0 if summary["verdict"] != Status.FAIL.value else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())