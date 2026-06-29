"""Enterprise release validator.

Validates a release candidate against the release policy: semantic version,
release notes, documentation, test success, coverage, repository cleanliness,
configuration integrity, required files and directories, and artifacts. Produces
per-check results and an overall PASS / WARNING / FAIL verdict. Runnable as a
CLI::

    python scripts/release_validator.py --root . --version 11.3.0 --coverage 0.92
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_repository import (  # noqa: E402
    CheckResult, Status, ValidationReport, load_yaml, DEFAULT_TIMESTAMP,
)

__all__ = ["SemanticVersion", "ReleaseContext", "ReleaseValidator", "load_release_policy", "main"]

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z\-.]+))?(?:\+(?P<build>[0-9A-Za-z\-.]+))?$")

_DEFAULT_POLICY = {
    "version_policy": {"scheme": "semver", "allow_prerelease": False},
    "validation_policy": {
        "require_tests_pass": True, "min_coverage": 0.85, "require_documentation": True,
        "require_clean_working_tree": True, "require_changelog": True, "min_quality_score": 80.0,
        "required_files": ["configs/quality_gate.yaml", "configs/release_policy.yaml"],
        "required_directories": ["src", "tests", "docs"],
    },
}


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    """A parsed, comparable semantic version."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, text: str) -> "SemanticVersion":
        match = _SEMVER_RE.match(text.strip())
        if not match:
            raise ValueError(f"invalid semantic version: {text!r}")
        return cls(int(match["major"]), int(match["minor"]), int(match["patch"]),
                   match["prerelease"], match["build"])

    @classmethod
    def is_valid(cls, text: str) -> bool:
        return bool(_SEMVER_RE.match(text.strip()))

    @property
    def is_prerelease(self) -> bool:
        return self.prerelease is not None

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        if self.build:
            base += f"+{self.build}"
        return base


def load_release_policy(path: Optional[str]) -> Dict[str, Any]:
    """Load a release policy, merging over built-in defaults."""
    policy = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULT_POLICY.items()}
    if path and os.path.exists(path):
        loaded = load_yaml(path)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(policy.get(key), dict):
                policy[key].update(value)
            else:
                policy[key] = value
    return policy


@dataclass(frozen=True, slots=True)
class ReleaseContext:
    """Inputs for a release validation."""

    version: str
    release_notes: str = ""
    coverage: Optional[float] = None
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    quality_score: Optional[float] = None
    clean_working_tree: bool = True
    artifacts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


class ReleaseValidator:
    """Validates a release candidate against the release policy."""

    def __init__(self, root: str, *, policy: Optional[Dict[str, Any]] = None,
                 timestamp: str = DEFAULT_TIMESTAMP) -> None:
        self.root = root
        self.timestamp = timestamp
        self.policy = policy or load_release_policy(
            os.path.join(root, "configs", "release_policy.yaml"))
        self.version_policy = dict(self.policy.get("version_policy", {}))
        self.validation_policy = dict(self.policy.get("validation_policy", {}))

    # -- checks ------------------------------------------------------------- #
    def check_version(self, context: ReleaseContext) -> CheckResult:
        if not SemanticVersion.is_valid(context.version):
            return CheckResult("semantic_version", Status.FAIL, 0.0,
                               f"invalid version {context.version!r}")
        version = SemanticVersion.parse(context.version)
        allow_pre = bool(self.version_policy.get("allow_prerelease", False))
        if version.is_prerelease and not allow_pre:
            return CheckResult("semantic_version", Status.WARNING, 0.6,
                               "prerelease not permitted for production by policy")
        return CheckResult("semantic_version", Status.PASS, 1.0, f"version {version} valid")

    def check_release_notes(self, context: ReleaseContext) -> CheckResult:
        notes = context.release_notes.strip()
        if not notes and self._has_changelog():
            notes = "changelog file present in repository"
        if len(notes) >= 20:
            return CheckResult("release_notes", Status.PASS, 1.0, "release notes present")
        if notes:
            return CheckResult("release_notes", Status.WARNING, 0.5, "release notes too short")
        return CheckResult("release_notes", Status.FAIL, 0.0, "no release notes")

    def _has_changelog(self) -> bool:
        return any(os.path.exists(os.path.join(self.root, n))
                   for n in ("CHANGELOG.md", "CHANGES.md", "RELEASE.md"))

    def check_documentation(self, context: ReleaseContext) -> CheckResult:
        docs = os.path.join(self.root, "docs")
        files = ([f for f in os.listdir(docs) if f.endswith(".md")]
                 if os.path.isdir(docs) else [])
        if not bool(self.validation_policy.get("require_documentation", True)):
            return CheckResult("documentation", Status.PASS, 1.0, "documentation not required")
        status = Status.PASS if files else Status.FAIL
        return CheckResult("documentation", status, 1.0 if files else 0.0,
                           f"{len(files)} documentation file(s)")

    def check_tests(self, context: ReleaseContext) -> CheckResult:
        if not bool(self.validation_policy.get("require_tests_pass", True)):
            return CheckResult("test_success", Status.PASS, 1.0, "tests not required")
        if context.tests_failed is None and context.tests_passed is None:
            return CheckResult("test_success", Status.WARNING, 0.5, "no test results supplied")
        failed = context.tests_failed or 0
        passed = context.tests_passed or 0
        if failed == 0 and passed > 0:
            return CheckResult("test_success", Status.PASS, 1.0, f"{passed} passed")
        score = passed / (passed + failed) if (passed + failed) else 0.0
        return CheckResult("test_success", Status.FAIL, score, f"{failed} failing tests")

    def check_coverage(self, context: ReleaseContext) -> CheckResult:
        threshold = float(self.validation_policy.get("min_coverage", 0.85))
        if context.coverage is None:
            return CheckResult("coverage", Status.WARNING, 0.5, "no coverage supplied")
        cov = float(context.coverage)
        status = Status.PASS if cov >= threshold else (
            Status.WARNING if cov >= threshold * 0.9 else Status.FAIL)
        return CheckResult("coverage", status, min(1.0, cov),
                           f"coverage {cov:.2%} vs {threshold:.2%}")

    def check_cleanliness(self, context: ReleaseContext) -> CheckResult:
        if not bool(self.validation_policy.get("require_clean_working_tree", True)):
            return CheckResult("repository_cleanliness", Status.PASS, 1.0, "cleanliness not required")
        status = Status.PASS if context.clean_working_tree else Status.FAIL
        return CheckResult("repository_cleanliness", status, 1.0 if context.clean_working_tree else 0.0,
                           "clean working tree" if context.clean_working_tree else "uncommitted changes")

    def check_configuration_integrity(self, context: ReleaseContext) -> CheckResult:
        required = list(self.validation_policy.get("required_files", []))
        configs = [r for r in required if r.endswith((".yaml", ".yml"))]
        if not configs:
            return CheckResult("configuration_integrity", Status.PASS, 1.0, "no configs required")
        ok = 0
        for rel in configs:
            path = os.path.join(self.root, rel)
            if os.path.exists(path):
                try:
                    if load_yaml(path):
                        ok += 1
                except Exception:  # noqa: BLE001
                    pass
        ratio = ok / len(configs)
        status = Status.PASS if ratio == 1.0 else Status.FAIL
        return CheckResult("configuration_integrity", status, ratio,
                           f"{ok}/{len(configs)} configs valid")

    def check_required_files(self, context: ReleaseContext) -> CheckResult:
        required = list(self.validation_policy.get("required_files", []))
        if not required:
            return CheckResult("required_files", Status.PASS, 1.0, "no required files")
        present = [r for r in required if os.path.exists(os.path.join(self.root, r))]
        ratio = len(present) / len(required)
        status = Status.PASS if ratio == 1.0 else (Status.WARNING if ratio >= 0.5 else Status.FAIL)
        return CheckResult("required_files", status, ratio,
                           f"{len(present)}/{len(required)} required files present")

    def check_required_directories(self, context: ReleaseContext) -> CheckResult:
        required = list(self.validation_policy.get("required_directories", []))
        if not required:
            return CheckResult("required_directories", Status.PASS, 1.0, "no required dirs")
        present = [r for r in required if os.path.isdir(os.path.join(self.root, r))]
        ratio = len(present) / len(required)
        status = Status.PASS if ratio == 1.0 else (Status.WARNING if ratio >= 0.5 else Status.FAIL)
        return CheckResult("required_directories", status, ratio,
                           f"{len(present)}/{len(required)} required directories present")

    def check_artifacts(self, context: ReleaseContext) -> CheckResult:
        if not context.artifacts:
            return CheckResult("artifacts", Status.WARNING, 0.5, "no artifacts declared")
        missing = [a for a in context.artifacts if not os.path.exists(os.path.join(self.root, a))]
        ratio = 1.0 - len(missing) / len(context.artifacts)
        status = Status.PASS if not missing else Status.FAIL
        return CheckResult("artifacts", status, ratio,
                           f"{len(context.artifacts) - len(missing)}/{len(context.artifacts)} artifacts present")

    # -- aggregation -------------------------------------------------------- #
    def validate(self, context: ReleaseContext) -> ValidationReport:
        checks = [
            self.check_version(context), self.check_release_notes(context),
            self.check_documentation(context), self.check_tests(context),
            self.check_coverage(context), self.check_cleanliness(context),
            self.check_configuration_integrity(context), self.check_required_files(context),
            self.check_required_directories(context), self.check_artifacts(context),
        ]
        return ValidationReport("release_validation", tuple(checks), self.timestamp)

    def summary(self, context: ReleaseContext) -> Dict[str, Any]:
        report = self.validate(context)
        return {
            "version": context.version,
            "verdict": report.overall_status.value,
            "score": report.score,
            "passed": report.passed,
            "warnings": report.warnings,
            "failed": report.failed,
            "report": report.to_dict(),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a release candidate")
    parser.add_argument("--root", default=None)
    parser.add_argument("--version", default="0.0.0")
    parser.add_argument("--notes", default="")
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--tests-failed", type=int, default=None)
    parser.add_argument("--quality-score", type=float, default=None)
    parser.add_argument("--dirty", action="store_true", help="working tree has uncommitted changes")
    parser.add_argument("--artifact", action="append", default=[], help="declared artifact path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    from validate_repository import _resolve_root  # noqa: E402

    root = _resolve_root(args.root)
    context = ReleaseContext(
        version=args.version, release_notes=args.notes, coverage=args.coverage,
        tests_passed=args.tests_passed, tests_failed=args.tests_failed,
        quality_score=args.quality_score, clean_working_tree=not args.dirty,
        artifacts=tuple(args.artifact))
    summary = ReleaseValidator(root).summary(context)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    if not args.quiet:
        print(payload)
    return 0 if summary["verdict"] != Status.FAIL.value else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())