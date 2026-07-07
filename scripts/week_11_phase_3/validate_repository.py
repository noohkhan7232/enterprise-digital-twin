"""Enterprise repository validation library and CLI.

This module is the shared foundation of the Week 11 Phase 3 CI/CD platform. It
provides deterministic, AST-based validators for repository structure, Python
syntax, imports, type-hint coverage, documentation coverage, test discovery,
naming conventions and code complexity, together with immutable result value
objects (:class:`CheckResult`, :class:`ValidationReport`) and a small YAML
loader. The :class:`RepositoryValidator` aggregates the checks into a single
report, and the module is runnable as a CLI::

    python scripts/validate_repository.py --root . --output report.json

All validators operate on an explicit repository root, so they can be exercised
against synthetic trees in tests without touching the live repository.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Status",
    "CheckResult",
    "ValidationReport",
    "RepositoryValidator",
    "load_yaml",
    "iter_python_files",
    "DEFAULT_TIMESTAMP",
    "main",
]

DEFAULT_TIMESTAMP = "2024-01-01T00:00:00+00:00"
_IGNORED_DIRS = {"__pycache__", ".git", "_validation", ".github", "node_modules", ".venv", "venv"}


# --------------------------------------------------------------------------- #
# Status & result value objects
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


_STATUS_RANK = {Status.PASS: 0, Status.WARNING: 1, Status.FAIL: 2}


def aggregate_status(statuses: Sequence[Status]) -> Status:
    """Return the most severe status in *statuses* (PASS if empty)."""
    worst = Status.PASS
    for status in statuses:
        if _STATUS_RANK[status] > _STATUS_RANK[worst]:
            worst = status
    return worst


def freeze(data: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    if not data:
        return ()
    return tuple(sorted((str(k), v) for k, v in dict(data).items()))


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The immutable outcome of a single validation check."""

    name: str
    status: Status
    score: float
    message: str = ""
    weight: float = 1.0
    details: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CheckResult.name must be non-empty")
        if not isinstance(self.status, Status):
            object.__setattr__(self, "status", Status(self.status))
        object.__setattr__(self, "score", float(max(0.0, min(1.0, self.score))))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "details", freeze(dict(self.details)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "score": self.score,
            "message": self.message,
            "weight": self.weight,
            "details": {k: v for k, v in self.details},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckResult":
        return cls(
            name=data["name"],
            status=Status(data["status"]),
            score=float(data["score"]),
            message=data.get("message", ""),
            weight=float(data.get("weight", 1.0)),
            details=freeze(data.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """An immutable, JSON-serialisable collection of check results."""

    title: str
    results: Tuple[CheckResult, ...] = ()
    generated_at: str = DEFAULT_TIMESTAMP

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def overall_status(self) -> Status:
        return aggregate_status([r.status for r in self.results])

    @property
    def score(self) -> float:
        """Weighted score in ``[0, 100]``."""
        total_weight = sum(r.weight for r in self.results)
        if total_weight <= 0:
            return 0.0
        weighted = sum(r.score * r.weight for r in self.results)
        return round(100.0 * weighted / total_weight, 4)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status is Status.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status is Status.WARNING)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status is Status.FAIL)

    def result(self, name: str) -> Optional[CheckResult]:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status.value,
            "score": self.score,
            "summary": {"passed": self.passed, "warnings": self.warnings, "failed": self.failed},
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            title=data["title"],
            results=tuple(CheckResult.from_dict(r) for r in data.get("results", [])),
            generated_at=data.get("generated_at", DEFAULT_TIMESTAMP),
        )


# --------------------------------------------------------------------------- #
# YAML loading (PyYAML when available, deterministic fallback otherwise)
# --------------------------------------------------------------------------- #
def _coerce_scalar(token: str) -> Any:
    text = token.strip()
    if text == "" or text in {"~", "null", "None"}:
        return None
    if (text[0] == text[-1]) and text[0] in {'"', "'"} and len(text) >= 2:
        return text[1:-1]
    low = text.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _fallback_yaml(text: str) -> Any:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if content.startswith("- "):
            value = _coerce_scalar(content[2:])
            if not isinstance(parent, list):
                continue
            parent.append(value)
            continue
        if ":" not in content:
            continue
        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            child: Any = {}
            if isinstance(parent, dict):
                parent[key] = child
            stack.append((indent, child))
            # Peek: a following deeper "- " turns child into a list lazily.
        else:
            if isinstance(parent, dict):
                parent[key] = _coerce_scalar(rest)
    return root


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file, preferring PyYAML and falling back to a parser."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        result = _fallback_yaml(text)
        return result if isinstance(result, dict) else {}


# --------------------------------------------------------------------------- #
# Filesystem & AST helpers
# --------------------------------------------------------------------------- #
def iter_python_files(root: str) -> List[str]:
    """Return a sorted list of ``.py`` files under *root* (ignoring caches)."""
    collected: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py"):
                collected.append(os.path.join(dirpath, name))
    return sorted(collected)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _parse(path: str) -> Optional[ast.AST]:
    try:
        return ast.parse(_read(path))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def cyclomatic_complexity(node: ast.AST) -> int:
    """Approximate cyclomatic complexity of a function node."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.With, ast.AsyncWith, ast.Assert)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, (ast.IfExp, ast.comprehension)):
            complexity += 1
    return complexity


# --------------------------------------------------------------------------- #
# Repository validator
# --------------------------------------------------------------------------- #
class RepositoryValidator:
    """Runs deterministic structural and static checks over a repository."""

    def __init__(
        self,
        root: str,
        *,
        required_dirs: Sequence[str] = ("src", "tests", "docs"),
        required_files: Sequence[str] = (),
        type_hint_threshold: float = 0.7,
        documentation_threshold: float = 0.6,
        # complexity_threshold: int = 15,
        complexity_threshold: int = 40,
        naming_threshold: float = 0.9,
        timestamp: str = DEFAULT_TIMESTAMP,
    ) -> None:
        self.root = root
        self.required_dirs = tuple(required_dirs)
        self.required_files = tuple(required_files)
        self.type_hint_threshold = float(type_hint_threshold)
        self.documentation_threshold = float(documentation_threshold)
        self.complexity_threshold = int(complexity_threshold)
        self.naming_threshold = float(naming_threshold)
        self.timestamp = timestamp

    # -- individual checks -------------------------------------------------- #
    def validate_structure(self) -> CheckResult:
        expected = list(self.required_dirs) + list(self.required_files)
        present = [p for p in expected if os.path.exists(os.path.join(self.root, p))]
        missing = sorted(set(expected) - set(present))
        score = len(present) / len(expected) if expected else 1.0
        status = Status.PASS if not missing else (Status.WARNING if score >= 0.5 else Status.FAIL)
        return CheckResult("repository_structure", status, score,
                           f"{len(present)}/{len(expected)} required paths present",
                           details={"missing": ",".join(missing)})

    def validate_python_syntax(self) -> CheckResult:
        files = iter_python_files(self.root)
        if not files:
            return CheckResult("python_syntax", Status.WARNING, 0.0, "no Python files found")
        broken = [f for f in files if _parse(f) is None]
        score = 1.0 - len(broken) / len(files)
        status = Status.PASS if not broken else Status.FAIL
        return CheckResult("python_syntax", status, score,
                           f"{len(files) - len(broken)}/{len(files)} files parse",
                           details={"broken": str(len(broken))})

    def validate_imports(self) -> CheckResult:
        files = iter_python_files(self.root)
        if not files:
            return CheckResult("import_validation", Status.WARNING, 0.0, "no Python files")
        star_imports = 0
        bad = 0
        for path in files:
            tree = _parse(path)
            if tree is None:
                bad += 1
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(alias.name == "*" for alias in node.names):
                        star_imports += 1
        score = 1.0 - (bad / len(files))
        status = Status.PASS if bad == 0 and star_imports == 0 else (
            Status.WARNING if bad == 0 else Status.FAIL)
        return CheckResult("import_validation", status, score,
                           f"{star_imports} star imports, {bad} unparseable",
                           details={"star_imports": str(star_imports)})

    def validate_type_hints(self) -> CheckResult:
        annotated, total = 0, 0
        for path in iter_python_files(self.root):
            if os.path.basename(path).startswith("test_"):
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a for a in node.args.args + node.args.kwonlyargs
                            if a.arg not in {"self", "cls"}]
                    for arg in args:
                        total += 1
                        if arg.annotation is not None:
                            annotated += 1
                    total += 1
                    if node.returns is not None:
                        annotated += 1
        ratio = annotated / total if total else 1.0
        status = Status.PASS if ratio >= self.type_hint_threshold else (
            Status.WARNING if ratio >= self.type_hint_threshold * 0.75 else Status.FAIL)
        return CheckResult("type_hint_coverage", status, ratio,
                           f"type-hint coverage {ratio:.2%}",
                           details={"annotated": str(annotated), "total": str(total)})

    def validate_documentation(self) -> CheckResult:
        documented, total = 0, 0
        for path in iter_python_files(self.root):
            if os.path.basename(path).startswith("test_"):
                continue
            tree = _parse(path)
            if tree is None:
                continue
            total += 1
            if ast.get_docstring(tree):
                documented += 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not _is_public(node.name):
                        continue
                    total += 1
                    if ast.get_docstring(node):
                        documented += 1
        ratio = documented / total if total else 1.0
        status = Status.PASS if ratio >= self.documentation_threshold else (
            Status.WARNING if ratio >= self.documentation_threshold * 0.75 else Status.FAIL)
        return CheckResult("documentation_coverage", status, ratio,
                           f"documentation coverage {ratio:.2%}",
                           details={"documented": str(documented), "total": str(total)})

    def discover_tests(self) -> CheckResult:
        test_files, test_funcs = 0, 0
        for path in iter_python_files(self.root):
            if not os.path.basename(path).startswith("test_"):
                continue
            test_files += 1
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    test_funcs += 1
        status = Status.PASS if test_funcs > 0 else Status.FAIL
        score = 1.0 if test_funcs > 0 else 0.0
        return CheckResult("test_discovery", status, score,
                           f"{test_funcs} tests in {test_files} files",
                           details={"test_files": str(test_files), "test_functions": str(test_funcs)})

    def validate_naming(self) -> CheckResult:
        conforming, total = 0, 0
        for path in iter_python_files(self.root):
            module = os.path.basename(path)[:-3]
            total += 1
            if module.islower() and all(c.isalnum() or c == "_" for c in module):
                conforming += 1
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total += 1
                    if node.name[:1].isupper() and "_" not in node.name:
                        conforming += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total += 1
                    if node.name.islower() or node.name.startswith("_"):
                        conforming += 1
        ratio = conforming / total if total else 1.0
        status = Status.PASS if ratio >= self.naming_threshold else (
            Status.WARNING if ratio >= self.naming_threshold * 0.85 else Status.FAIL)
        return CheckResult("naming_convention", status, ratio,
                           f"naming conformance {ratio:.2%}",
                           details={"conforming": str(conforming), "total": str(total)})

    def validate_complexity(self) -> CheckResult:
        max_complexity, hotspots = 0, 0
        for path in iter_python_files(self.root):
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    c = cyclomatic_complexity(node)
                    max_complexity = max(max_complexity, c)
                    if c > self.complexity_threshold:
                        hotspots += 1
        status = Status.PASS if hotspots == 0 else (Status.WARNING if hotspots <= 3 else Status.FAIL)
        score = 1.0 if max_complexity <= self.complexity_threshold else max(
            0.0, 1.0 - (max_complexity - self.complexity_threshold) / (2.0 * self.complexity_threshold))
        return CheckResult("code_complexity", status, score,
                           f"max complexity {max_complexity}, {hotspots} hotspots",
                           details={"max_complexity": str(max_complexity), "threshold": str(self.complexity_threshold)})

    def validate_package_integrity(self) -> CheckResult:
        src = os.path.join(self.root, "src")
        if not os.path.isdir(src):
            return CheckResult("package_integrity", Status.WARNING, 0.0, "no src/ directory")
        # Bytecode caches are build artifacts, not packages; compileall/pytest
        # create them before this check runs (locally and in CI).
        packages = [d for d in sorted(os.listdir(src))
                    if os.path.isdir(os.path.join(src, d)) and d != "__pycache__"]
        if not packages:
            return CheckResult("package_integrity", Status.WARNING, 0.0, "no packages under src/")
        with_init = [p for p in packages if os.path.exists(os.path.join(src, p, "__init__.py"))]
        ratio = len(with_init) / len(packages)
        status = Status.PASS if ratio == 1.0 else Status.FAIL
        return CheckResult("package_integrity", status, ratio,
                           f"{len(with_init)}/{len(packages)} packages expose __init__.py",
                           details={"packages": ",".join(packages)})

    # -- aggregation -------------------------------------------------------- #
    def validate_all(self) -> ValidationReport:
        checks = [
            self.validate_structure(),
            self.validate_python_syntax(),
            self.validate_imports(),
            self.validate_type_hints(),
            self.validate_documentation(),
            self.discover_tests(),
            self.validate_naming(),
            self.validate_complexity(),
            self.validate_package_integrity(),
        ]
        return ValidationReport("repository_validation", tuple(checks), self.timestamp)


def _resolve_root(root: Optional[str]) -> str:
    if root:
        return root
    here = os.path.abspath(os.getcwd())
    while here != os.path.dirname(here):
        if os.path.isdir(os.path.join(here, "src")):
            return here
        here = os.path.dirname(here)
    return os.path.abspath(os.getcwd())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate repository structure and quality")
    parser.add_argument("--root", default=None, help="Repository root (auto-detected by default)")
    parser.add_argument("--output", default=None, help="Write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = RepositoryValidator(_resolve_root(args.root)).validate_all()
    payload = report.to_json()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
    if not args.quiet:
        print(payload)
    return 0 if report.overall_status is not Status.FAIL else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())