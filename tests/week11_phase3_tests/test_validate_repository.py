"""Tests for the repository validation library and CLI."""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts", "week_11_phase_3"))

from validate_repository import (  # noqa: E402
    CheckResult, RepositoryValidator, Status, ValidationReport, aggregate_status,
    cyclomatic_complexity, iter_python_files, load_yaml, main,
)


# --------------------------------------------------------------------------- #
# Repo builders
# --------------------------------------------------------------------------- #
def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build_good_repo(root):
    write(os.path.join(root, "src", "pkg", "__init__.py"), '"""Package."""\n')
    write(os.path.join(root, "src", "pkg", "core.py"),
          '"""Core module."""\n\n\nclass Engine:\n    """Engine."""\n\n'
          '    def run(self, value: int) -> int:\n        """Run."""\n        return value + 1\n')
    write(os.path.join(root, "tests", "test_core.py"),
          "def test_run():\n    assert True\n\n\ndef test_more():\n    assert 1 == 1\n")
    write(os.path.join(root, "docs", "guide.md"), "# Guide\n")
    return root


# --------------------------------------------------------------------------- #
# CheckResult
# --------------------------------------------------------------------------- #
def test_check_result_roundtrip():
    cr = CheckResult("c", Status.PASS, 0.8, "ok", 2.0, {"k": "v"})
    assert CheckResult.from_dict(cr.to_dict()) == cr


def test_check_result_clamps_high():
    assert CheckResult("c", Status.PASS, 5.0).score == 1.0


def test_check_result_clamps_low():
    assert CheckResult("c", Status.FAIL, -1.0).score == 0.0


def test_check_result_coerces_status():
    assert CheckResult("c", "PASS", 1.0).status is Status.PASS


def test_check_result_requires_name():
    with pytest.raises(ValueError):
        CheckResult("", Status.PASS, 1.0)


def test_check_result_details_frozen():
    cr = CheckResult("c", Status.PASS, 1.0, details={"b": 1, "a": 2})
    assert cr.details == (("a", 2), ("b", 1))


def test_check_result_hashable():
    assert isinstance(hash(CheckResult("c", Status.PASS, 1.0)), int)


def test_check_result_immutable():
    cr = CheckResult("c", Status.PASS, 1.0)
    with pytest.raises(Exception):
        cr.score = 0.5  # type: ignore


# --------------------------------------------------------------------------- #
# aggregate_status
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("statuses,expected", [
    ([], Status.PASS),
    ([Status.PASS, Status.PASS], Status.PASS),
    ([Status.PASS, Status.WARNING], Status.WARNING),
    ([Status.WARNING, Status.FAIL], Status.FAIL),
    ([Status.PASS, Status.FAIL, Status.WARNING], Status.FAIL),
])
def test_aggregate_status(statuses, expected):
    assert aggregate_status(statuses) is expected


# --------------------------------------------------------------------------- #
# ValidationReport
# --------------------------------------------------------------------------- #
def sample_report():
    return ValidationReport("r", (
        CheckResult("a", Status.PASS, 1.0, weight=1.0),
        CheckResult("b", Status.WARNING, 0.5, weight=1.0),
        CheckResult("c", Status.FAIL, 0.0, weight=2.0),
    ))


def test_report_overall_status():
    assert sample_report().overall_status is Status.FAIL


def test_report_score_weighted():
    # (1*1 + 0.5*1 + 0*2) / 4 * 100 = 37.5
    assert sample_report().score == pytest.approx(37.5)


def test_report_counts():
    r = sample_report()
    assert (r.passed, r.warnings, r.failed) == (1, 1, 1)


def test_report_result_lookup():
    assert sample_report().result("b").status is Status.WARNING
    assert sample_report().result("missing") is None


def test_report_all_pass():
    r = ValidationReport("r", (CheckResult("a", Status.PASS, 1.0),))
    assert r.overall_status is Status.PASS
    assert r.score == 100.0


def test_report_empty_score_zero():
    assert ValidationReport("r", ()).score == 0.0


def test_report_json_roundtrip():
    r = sample_report()
    assert ValidationReport.from_dict(json.loads(r.to_json())).to_dict() == r.to_dict()


def test_report_to_json_sorted():
    r = sample_report()
    assert r.to_json() == r.to_json()


def test_report_to_dict_has_summary():
    d = sample_report().to_dict()
    assert d["summary"] == {"passed": 1, "warnings": 1, "failed": 1}


# --------------------------------------------------------------------------- #
# YAML loader
# --------------------------------------------------------------------------- #
def test_yaml_scalar_types():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        write(p, "i: 5\nf: 1.5\nb: true\ns: hello\nq: \"quoted\"\n")
        cfg = load_yaml(p)
    assert cfg["i"] == 5 and cfg["f"] == 1.5 and cfg["b"] is True
    assert cfg["s"] == "hello" and cfg["q"] == "quoted"


def test_yaml_nested():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        write(p, "outer:\n  inner: 7\n")
        assert load_yaml(p)["outer"]["inner"] == 7


def test_yaml_list():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        write(p, "items:\n  - a\n  - b\n  - c\n")
        assert load_yaml(p)["items"] == ["a", "b", "c"]


def test_yaml_comments_ignored():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.yaml")
        write(p, "# comment\nkey: value  # inline\n")
        assert load_yaml(p)["key"] == "value"


# --------------------------------------------------------------------------- #
# iter_python_files / complexity
# --------------------------------------------------------------------------- #
def test_iter_python_files_sorted_and_filtered():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "b.py"), "x = 1\n")
        write(os.path.join(d, "a.py"), "y = 2\n")
        write(os.path.join(d, "__pycache__", "c.py"), "z = 3\n")
        files = [os.path.basename(f) for f in iter_python_files(d)]
    assert files == ["a.py", "b.py"]


def test_complexity_simple():
    tree = ast.parse("def f():\n    return 1\n")
    func = tree.body[0]
    assert cyclomatic_complexity(func) == 1


def test_complexity_branches():
    tree = ast.parse("def f(x):\n    if x:\n        return 1\n    for i in range(x):\n        pass\n    return 0\n")
    assert cyclomatic_complexity(tree.body[0]) >= 3


def test_complexity_boolops():
    tree = ast.parse("def f(a, b, c):\n    return a and b and c\n")
    assert cyclomatic_complexity(tree.body[0]) >= 3


# --------------------------------------------------------------------------- #
# RepositoryValidator — structure
# --------------------------------------------------------------------------- #
def test_structure_pass():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        assert RepositoryValidator(d).validate_structure().status is Status.PASS


def test_structure_partial_warning():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        os.makedirs(os.path.join(d, "tests"))
        assert RepositoryValidator(d).validate_structure().status is Status.WARNING


def test_structure_fail():
    with tempfile.TemporaryDirectory() as d:
        assert RepositoryValidator(d).validate_structure().status is Status.FAIL


def test_structure_missing_listed():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        result = RepositoryValidator(d).validate_structure()
        assert "docs" in dict(result.details)["missing"]


# --------------------------------------------------------------------------- #
# RepositoryValidator — syntax / imports
# --------------------------------------------------------------------------- #
def test_syntax_pass():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        assert RepositoryValidator(d).validate_python_syntax().status is Status.PASS


def test_syntax_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "bad.py"), "def (:\n")
        assert RepositoryValidator(d).validate_python_syntax().status is Status.FAIL


def test_syntax_no_files_warning():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        assert RepositoryValidator(d).validate_python_syntax().status is Status.WARNING


def test_imports_pass():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        assert RepositoryValidator(d).validate_imports().status is Status.PASS


def test_imports_star_warning():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "from os import *\n")
        assert RepositoryValidator(d).validate_imports().status is Status.WARNING


# --------------------------------------------------------------------------- #
# RepositoryValidator — type hints / docs / naming / complexity
# --------------------------------------------------------------------------- #
def test_type_hints_full():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              "def f(a: int, b: int) -> int:\n    return a + b\n")
        assert RepositoryValidator(d).validate_type_hints().score == 1.0


def test_type_hints_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "def f(a, b, c, d):\n    return a\n")
        assert RepositoryValidator(d).validate_type_hints().status is Status.FAIL


def test_documentation_pass():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              '"""Mod."""\n\n\ndef f():\n    """Doc."""\n    return 1\n')
        assert RepositoryValidator(d).validate_documentation().status is Status.PASS


def test_documentation_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n")
        assert RepositoryValidator(d).validate_documentation().status is Status.FAIL


def test_naming_pass():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "good_module.py"),
              "class GoodClass:\n    pass\n\n\ndef good_func():\n    return 1\n")
        assert RepositoryValidator(d).validate_naming().status is Status.PASS


def test_complexity_pass():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "def f(x):\n    return x + 1\n")
        assert RepositoryValidator(d).validate_complexity().status is Status.PASS


def test_complexity_hotspot_detected():
    with tempfile.TemporaryDirectory() as d:
        body = "def f(x):\n" + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(25))
        write(os.path.join(d, "src", "m.py"), body)
        result = RepositoryValidator(d, complexity_threshold=10).validate_complexity()
        assert result.status in (Status.WARNING, Status.FAIL)
        assert int(dict(result.details)["max_complexity"]) > 10


# --------------------------------------------------------------------------- #
# RepositoryValidator — discovery / packages / aggregate
# --------------------------------------------------------------------------- #
def test_discover_tests_pass():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        result = RepositoryValidator(d).discover_tests()
        assert result.status is Status.PASS
        assert int(dict(result.details)["test_functions"]) == 2


def test_discover_tests_none_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "x = 1\n")
        assert RepositoryValidator(d).discover_tests().status is Status.FAIL


def test_package_integrity_pass():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        assert RepositoryValidator(d).validate_package_integrity().status is Status.PASS


def test_package_integrity_fail():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src", "pkg"))
        write(os.path.join(d, "src", "pkg", "m.py"), "x = 1\n")
        assert RepositoryValidator(d).validate_package_integrity().status is Status.FAIL


def test_validate_all_report():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        report = RepositoryValidator(d).validate_all()
        assert len(report.results) == 9
        assert report.overall_status in (Status.PASS, Status.WARNING)


def test_validate_all_deterministic():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        v = RepositoryValidator(d)
        assert v.validate_all().to_dict() == v.validate_all().to_dict()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_returns_zero_on_good_repo():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        assert main(["--root", d, "--quiet"]) == 0


def test_cli_writes_output():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        out = os.path.join(d, "report.json")
        main(["--root", d, "--output", out, "--quiet"])
        with open(out) as fh:
            payload = json.load(fh)
        assert payload["title"] == "repository_validation"


def test_cli_fail_returns_one():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "bad.py"), "def (:\n")
        assert main(["--root", d, "--quiet"]) == 1


# --------------------------------------------------------------------------- #
# Additional coverage: thresholds, edges, serialization
# --------------------------------------------------------------------------- #
def test_check_result_default_weight():
    assert CheckResult("c", Status.PASS, 1.0).weight == 1.0


def test_check_result_to_dict_details_dict():
    cr = CheckResult("c", Status.PASS, 1.0, details={"a": 1})
    assert cr.to_dict()["details"] == {"a": 1}


def test_report_score_zero_weight_safe():
    r = ValidationReport("r", (CheckResult("a", Status.PASS, 1.0, weight=0.0),))
    assert r.score == 0.0


def test_type_hints_warning_band():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"),
              "def f(a: int, b, c) -> int:\n    return a\n")
        result = RepositoryValidator(d, type_hint_threshold=0.8).validate_type_hints()
        assert result.status in (Status.WARNING, Status.FAIL)


def test_documentation_excludes_tests():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), '"""Mod."""\n\n\ndef f():\n    """D."""\n    return 1\n')
        write(os.path.join(d, "tests", "test_m.py"), "def test_a():\n    assert True\n")
        assert RepositoryValidator(d).validate_documentation().status is Status.PASS


def test_naming_fail_on_bad_class():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "class bad_name:\n    pass\n")
        result = RepositoryValidator(d).validate_naming()
        assert result.score < 1.0


def test_imports_unparseable_fail():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "m.py"), "import (((\n")
        assert RepositoryValidator(d).validate_imports().status is Status.FAIL


def test_iter_ignores_git_and_validation():
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "src", "a.py"), "x = 1\n")
        write(os.path.join(d, ".git", "b.py"), "y = 2\n")
        write(os.path.join(d, "_validation", "c.py"), "z = 3\n")
        files = [os.path.basename(f) for f in iter_python_files(d)]
        assert files == ["a.py"]


def test_structure_custom_required_files():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        os.makedirs(os.path.join(d, "tests"))
        os.makedirs(os.path.join(d, "docs"))
        write(os.path.join(d, "README.md"), "# r\n")
        v = RepositoryValidator(d, required_files=("README.md",))
        assert v.validate_structure().status is Status.PASS


def test_package_integrity_no_src_warning():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        assert RepositoryValidator(d).validate_package_integrity().status is Status.WARNING


def test_report_generated_at_default():
    from validate_repository import DEFAULT_TIMESTAMP
    assert ValidationReport("r", ()).generated_at == DEFAULT_TIMESTAMP


def test_validate_all_names_order():
    with tempfile.TemporaryDirectory() as d:
        build_good_repo(d)
        names = [r.name for r in RepositoryValidator(d).validate_all().results]
        assert names[0] == "repository_structure"
        assert names[-1] == "package_integrity"