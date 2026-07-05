"""Tests for the enterprise quality-gate engine."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "week_11_phase_3"))

from quality_gate import QualityContext, QualityGate, load_quality_config, main  # noqa: E402
from validate_repository import Status  # noqa: E402


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build_repo(root, *, with_mlops=True, with_monitoring=True):
    write(os.path.join(root, "src", "pkg", "__init__.py"), '"""Pkg."""\n')
    write(os.path.join(root, "src", "pkg", "core.py"),
          '"""Core."""\nfrom dataclasses import dataclass\nimport threading\n\n\n'
          '@dataclass(frozen=True)\nclass Model:\n    """Model."""\n    x: int\n\n'
          '    def to_dict(self) -> dict:\n        """D."""\n        return {"x": self.x}\n\n'
          '    @classmethod\n    def from_dict(cls, d: dict) -> "Model":\n        """F."""\n        return cls(d["x"])\n\n\n'
          'class Store:\n    """Store."""\n\n    def __init__(self) -> None:\n        self._lock = threading.RLock()\n        self._items: list = []\n\n'
          '    def add(self, item: int) -> None:\n        """Add."""\n        self._items.append(item)\n')
    if with_mlops:
        write(os.path.join(root, "src", "mlops", "__init__.py"), '"""MLOps."""\n')
        write(os.path.join(root, "src", "mlops", "m.py"), '"""m."""\nx = 1\n')
    if with_monitoring:
        write(os.path.join(root, "src", "monitoring", "__init__.py"), '"""Monitoring."""\n')
        write(os.path.join(root, "src", "monitoring", "m.py"), '"""m."""\ny = 2\n')
    write(os.path.join(root, "tests", "test_core.py"), "def test_x():\n    assert True\n")
    write(os.path.join(root, "docs", "guide.md"), "# Guide\n")
    write(os.path.join(root, "configs", "quality_gate.yaml"), 'version: "1.0"\nthresholds:\n  coverage: 0.85\n')
    write(os.path.join(root, "configs", "release_policy.yaml"), 'version: "1.0"\nrollback_policy:\n  enabled: true\n')
    return root


def ctx(root, **kw):
    return QualityContext(root, **kw)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_load_config_defaults():
    cfg = load_quality_config(None)
    assert cfg["thresholds"]["coverage"] == 0.85


def test_load_config_merges_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "q.yaml")
        write(p, "thresholds:\n  coverage: 0.99\n")
        cfg = load_quality_config(p)
    assert cfg["thresholds"]["coverage"] == 0.99
    assert "complexity" in cfg["thresholds"]


def test_load_config_missing_file_defaults():
    assert load_quality_config("/nonexistent/x.yaml")["thresholds"]["quality_score"] == 80.0


# --------------------------------------------------------------------------- #
# Gate count & evaluation
# --------------------------------------------------------------------------- #
def test_twenty_gates():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert len(QualityGate(ctx(d)).evaluate().results) == 20


def test_quality_score_range():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        score = QualityGate(ctx(d, coverage=0.95, tests_passed=10, tests_failed=0)).quality_score()
        assert 0.0 <= score <= 100.0


def test_all_gate_names_present():
    expected = {
        "repository_structure", "python_syntax", "import_validation", "type_hint_coverage",
        "documentation_coverage", "test_discovery", "pytest_execution", "coverage_threshold",
        "code_complexity", "dependency_validation", "configuration_validation", "package_integrity",
        "architecture_consistency", "naming_convention", "json_serialization", "thread_safety",
        "deterministic_behaviour", "mlops_integration", "monitoring_integration", "deployment_readiness",
    }
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        names = {r.name for r in QualityGate(ctx(d)).evaluate().results}
    assert names == expected


# --------------------------------------------------------------------------- #
# pytest_execution gate
# --------------------------------------------------------------------------- #
def test_pytest_gate_no_metrics_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_pytest_execution().status is Status.WARNING


def test_pytest_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d, tests_passed=100, tests_failed=0)).gate_pytest_execution().status is Status.PASS


def test_pytest_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d, tests_passed=90, tests_failed=10)).gate_pytest_execution().status is Status.FAIL


# --------------------------------------------------------------------------- #
# coverage_threshold gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coverage,expected", [
    (0.95, Status.PASS),
    (0.85, Status.PASS),
    (0.80, Status.WARNING),
    (0.50, Status.FAIL),
])
def test_coverage_gate(coverage, expected):
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d, coverage=coverage)).gate_coverage_threshold().status is expected


def test_coverage_gate_no_metric_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_coverage_threshold().status is Status.WARNING


# --------------------------------------------------------------------------- #
# integration gates
# --------------------------------------------------------------------------- #
def test_mlops_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d, with_mlops=True)
        assert QualityGate(ctx(d)).gate_mlops_integration().status is Status.PASS


def test_mlops_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d, with_mlops=False)
        assert QualityGate(ctx(d)).gate_mlops_integration().status is Status.FAIL


def test_monitoring_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d, with_monitoring=True)
        assert QualityGate(ctx(d)).gate_monitoring_integration().status is Status.PASS


def test_monitoring_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d, with_monitoring=False)
        assert QualityGate(ctx(d)).gate_monitoring_integration().status is Status.FAIL


# --------------------------------------------------------------------------- #
# static gates
# --------------------------------------------------------------------------- #
def test_json_serialization_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_json_serialization().status is Status.PASS


def test_json_serialization_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              "from dataclasses import dataclass\n\n\n@dataclass\nclass A:\n    x: int\n")
        assert QualityGate(ctx(d)).gate_json_serialization().status is Status.FAIL


def test_thread_safety_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_thread_safety().status is Status.PASS


def test_thread_safety_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              "class S:\n    def __init__(self):\n        self.items = []\n    def add(self, x):\n        self.items.append(x)\n")
        assert QualityGate(ctx(d)).gate_thread_safety().status is Status.FAIL


def test_deterministic_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_deterministic_behaviour().status is Status.PASS


def test_deterministic_gate_flags_now():
    with tempfile.TemporaryDirectory() as d:
        for i in range(4):
            write(os.path.join(d, "src", f"m{i}.py"), "import datetime\nx = datetime.now()\n")
        assert QualityGate(ctx(d)).gate_deterministic_behaviour().status is Status.FAIL


def test_dependency_gate_no_manifest_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_dependency_validation().status is Status.WARNING


def test_dependency_gate_clean_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        write(os.path.join(d, "requirements.txt"), "numpy==2.0\npyyaml==6.0\n")
        assert QualityGate(ctx(d)).gate_dependency_validation().status is Status.PASS


def test_dependency_gate_conflict_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        write(os.path.join(d, "requirements.txt"), "numpy==2.0\nnumpy==1.0\n")
        assert QualityGate(ctx(d)).gate_dependency_validation().status is Status.FAIL


def test_configuration_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        result = QualityGate(ctx(d)).gate_configuration_validation()
        assert result.status is Status.PASS


def test_configuration_gate_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        os.remove(os.path.join(d, "configs", "quality_gate.yaml"))
        os.remove(os.path.join(d, "configs", "release_policy.yaml"))
        assert QualityGate(ctx(d)).gate_configuration_validation().status is Status.FAIL


def test_architecture_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).gate_architecture_consistency().status is Status.PASS


def test_deployment_gate_runs():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        result = QualityGate(ctx(d)).gate_deployment_readiness()
        assert result.name == "deployment_readiness"
        assert 0.0 <= result.score <= 1.0


# --------------------------------------------------------------------------- #
# summary / weights / determinism / CLI
# --------------------------------------------------------------------------- #
def test_summary_structure():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        summary = QualityGate(ctx(d, coverage=0.95, tests_passed=10, tests_failed=0)).summary()
        assert set(summary) >= {"quality_score", "verdict", "threshold", "passed", "warnings", "failed", "report"}


def test_summary_fail_on_failing_gate():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        summary = QualityGate(ctx(d, tests_passed=1, tests_failed=99)).summary()
        assert summary["verdict"] == Status.FAIL.value


def test_weights_from_config_applied():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        config = load_quality_config(None)
        config["weights"] = {"python_syntax": 10.0}
        report = QualityGate(ctx(d, config=config)).evaluate()
        assert report.result("python_syntax").weight == 10.0


def test_quality_gate_deterministic():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        a = QualityGate(ctx(d, coverage=0.9, tests_passed=5, tests_failed=0)).evaluate()
        b = QualityGate(ctx(d, coverage=0.9, tests_passed=5, tests_failed=0)).evaluate()
        assert a.to_dict() == b.to_dict()


def test_summary_json_serializable():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert json.dumps(QualityGate(ctx(d, coverage=0.9, tests_passed=5, tests_failed=0)).summary())


def test_cli_runs():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        code = main(["--root", d, "--coverage", "0.95", "--tests-passed", "10",
                     "--tests-failed", "0", "--quiet"])
        assert code in (0, 1)


def test_cli_writes_output():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        out = os.path.join(d, "q.json")
        main(["--root", d, "--coverage", "0.95", "--tests-passed", "10",
              "--tests-failed", "0", "--output", out, "--quiet"])
        with open(out) as fh:
            assert "quality_score" in json.load(fh)


def test_cli_fail_returns_one():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert main(["--root", d, "--tests-passed", "1", "--tests-failed", "50", "--quiet"]) == 1


# --------------------------------------------------------------------------- #
# Additional coverage: delegated gates, thresholds, edges
# --------------------------------------------------------------------------- #
def test_evaluate_structure_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).evaluate().result("repository_structure").status is Status.PASS


def test_evaluate_syntax_gate_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).evaluate().result("python_syntax").status is Status.PASS


def test_evaluate_naming_gate_present():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d)).evaluate().result("naming_convention") is not None


def test_json_serialization_no_dataclasses_warning():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "x = 1\n")
        assert QualityGate(ctx(d)).gate_json_serialization().status is Status.WARNING


def test_thread_safety_no_state_pass():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "def f(x):\n    return x + 1\n")
        assert QualityGate(ctx(d)).gate_thread_safety().status is Status.PASS


def test_deterministic_no_src_warning():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        assert QualityGate(ctx(d)).gate_deterministic_behaviour().status is Status.WARNING


def test_architecture_no_src_warning():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        assert QualityGate(ctx(d)).gate_architecture_consistency().status is Status.WARNING


def test_dependency_duplicate_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        write(os.path.join(d, "requirements.txt"), "numpy==2.0\nnumpy==2.0\n")
        assert QualityGate(ctx(d)).gate_dependency_validation().status is Status.WARNING


def test_coverage_gate_near_threshold_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert QualityGate(ctx(d, coverage=0.78)).gate_coverage_threshold().status is Status.WARNING


def test_custom_coverage_threshold_from_config():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        config = load_quality_config(None)
        config["thresholds"]["coverage"] = 0.5
        assert QualityGate(ctx(d, config=config, coverage=0.6)).gate_coverage_threshold().status is Status.PASS


def test_summary_verdict_pass_path():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        config = load_quality_config(None)
        config["thresholds"]["quality_score"] = 1.0
        config["thresholds"]["deployment"] = 0.01
        summary = QualityGate(ctx(d, config=config, coverage=0.99, tests_passed=50, tests_failed=0)).summary()
        assert summary["verdict"] in (Status.PASS.value, Status.WARNING.value)


def test_quality_score_changes_with_metrics():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        low = QualityGate(ctx(d, coverage=0.2, tests_passed=1, tests_failed=20)).quality_score()
        high = QualityGate(ctx(d, coverage=0.99, tests_passed=50, tests_failed=0)).quality_score()
        assert high > low


def test_context_default_config_loaded():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert "coverage" in ctx(d).thresholds


def test_report_json_export():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        report = QualityGate(ctx(d, coverage=0.9, tests_passed=5, tests_failed=0)).evaluate()
        assert json.loads(report.to_json())["title"] == "quality_gate"


def test_deployment_gate_threshold_from_config():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        config = load_quality_config(None)
        config["thresholds"]["deployment"] = 0.01
        assert QualityGate(ctx(d, config=config)).gate_deployment_readiness().status is Status.PASS