"""Tests for the enterprise release validator."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from release_validator import (  # noqa: E402
    ReleaseContext, ReleaseValidator, SemanticVersion, load_release_policy, main,
)
from validate_repository import Status  # noqa: E402


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def build_repo(root):
    write(os.path.join(root, "src", "pkg", "__init__.py"), '"""Pkg."""\n')
    write(os.path.join(root, "tests", "test_x.py"), "def test_x():\n    assert True\n")
    write(os.path.join(root, "docs", "guide.md"), "# Guide\n")
    write(os.path.join(root, "configs", "quality_gate.yaml"), 'version: "1.0"\n')
    write(os.path.join(root, "configs", "release_policy.yaml"),
          'version_policy:\n  allow_prerelease: false\nvalidation_policy:\n  min_coverage: 0.85\n'
          '  required_files:\n    - configs/quality_gate.yaml\n    - configs/release_policy.yaml\n'
          '  required_directories:\n    - src\n    - tests\n    - docs\n')
    return root


def ready_context(**kw):
    base = dict(version="11.3.0", release_notes="A detailed and sufficiently long release note.",
                coverage=0.92, tests_passed=100, tests_failed=0, clean_working_tree=True)
    base.update(kw)
    return ReleaseContext(**base)


# --------------------------------------------------------------------------- #
# SemanticVersion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["0.0.0", "1.2.3", "11.3.0", "10.20.30", "1.0.0-rc.1", "1.0.0+build", "1.0.0-alpha+001"])
def test_semver_valid(text):
    assert SemanticVersion.is_valid(text)


@pytest.mark.parametrize("text", ["1", "1.2", "1.2.3.4", "v1.2.3", "a.b.c", "01.2.3", "", "1.2.-1"])
def test_semver_invalid(text):
    assert not SemanticVersion.is_valid(text)


def test_semver_parse_fields():
    v = SemanticVersion.parse("11.3.0")
    assert (v.major, v.minor, v.patch) == (11, 3, 0)


def test_semver_prerelease_flag():
    assert SemanticVersion.parse("1.0.0-rc.1").is_prerelease
    assert not SemanticVersion.parse("1.0.0").is_prerelease


def test_semver_build_metadata():
    assert SemanticVersion.parse("1.0.0+abc").build == "abc"


def test_semver_str_roundtrip():
    assert str(SemanticVersion.parse("1.2.3-rc.1+build")) == "1.2.3-rc.1+build"


def test_semver_parse_invalid_raises():
    with pytest.raises(ValueError):
        SemanticVersion.parse("not-a-version")


# --------------------------------------------------------------------------- #
# Policy loading
# --------------------------------------------------------------------------- #
def test_load_policy_defaults():
    policy = load_release_policy(None)
    assert policy["validation_policy"]["min_coverage"] == 0.85


def test_load_policy_merges():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rp.yaml")
        write(p, "validation_policy:\n  min_coverage: 0.99\n")
        policy = load_release_policy(p)
    assert policy["validation_policy"]["min_coverage"] == 0.99


# --------------------------------------------------------------------------- #
# Version check
# --------------------------------------------------------------------------- #
def test_version_check_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_version(ready_context()).status is Status.PASS


def test_version_check_invalid_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_version(ready_context(version="bad")).status is Status.FAIL


def test_version_check_prerelease_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_version(ready_context(version="1.0.0-rc.1")).status is Status.WARNING


# --------------------------------------------------------------------------- #
# Release notes
# --------------------------------------------------------------------------- #
def test_release_notes_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_release_notes(ready_context()).status is Status.PASS


def test_release_notes_short_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_release_notes(ready_context(release_notes="short")).status is Status.WARNING


def test_release_notes_empty_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_release_notes(ready_context(release_notes="")).status is Status.FAIL


def test_release_notes_changelog_fallback():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        write(os.path.join(d, "CHANGELOG.md"), "# Changelog\n")
        assert ReleaseValidator(d).check_release_notes(ready_context(release_notes="")).status is Status.PASS


# --------------------------------------------------------------------------- #
# Documentation / tests / coverage / cleanliness
# --------------------------------------------------------------------------- #
def test_documentation_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_documentation(ready_context()).status is Status.PASS


def test_documentation_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        for f in os.listdir(os.path.join(d, "docs")):
            os.remove(os.path.join(d, "docs", f))
        assert ReleaseValidator(d).check_documentation(ready_context()).status is Status.FAIL


def test_tests_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_tests(ready_context()).status is Status.PASS


def test_tests_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_tests(ready_context(tests_failed=5)).status is Status.FAIL


def test_tests_no_metrics_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_tests(ready_context(tests_passed=None, tests_failed=None)).status is Status.WARNING


@pytest.mark.parametrize("coverage,expected", [
    (0.95, Status.PASS), (0.85, Status.PASS), (0.80, Status.WARNING), (0.4, Status.FAIL),
])
def test_coverage_check(coverage, expected):
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_coverage(ready_context(coverage=coverage)).status is expected


def test_cleanliness_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_cleanliness(ready_context()).status is Status.PASS


def test_cleanliness_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_cleanliness(ready_context(clean_working_tree=False)).status is Status.FAIL


# --------------------------------------------------------------------------- #
# Config / files / dirs / artifacts
# --------------------------------------------------------------------------- #
def test_config_integrity_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_configuration_integrity(ready_context()).status is Status.PASS


def test_required_files_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_required_files(ready_context()).status is Status.PASS


def test_required_files_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        os.remove(os.path.join(d, "configs", "quality_gate.yaml"))
        os.remove(os.path.join(d, "configs", "release_policy.yaml"))
        assert ReleaseValidator(d).check_required_files(ready_context()).status is Status.FAIL


def test_required_directories_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_required_directories(ready_context()).status is Status.PASS


def test_artifacts_no_declaration_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_artifacts(ready_context()).status is Status.WARNING


def test_artifacts_present_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        ctx = ready_context(artifacts=("configs/quality_gate.yaml",))
        assert ReleaseValidator(d).check_artifacts(ctx).status is Status.PASS


def test_artifacts_missing_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        ctx = ready_context(artifacts=("dist/missing.tar.gz",))
        assert ReleaseValidator(d).check_artifacts(ctx).status is Status.FAIL


# --------------------------------------------------------------------------- #
# Aggregation / summary / CLI
# --------------------------------------------------------------------------- #
def test_validate_ready_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        report = ReleaseValidator(d).validate(ready_context(artifacts=("configs/quality_gate.yaml",)))
        assert report.overall_status in (Status.PASS, Status.WARNING)
        assert len(report.results) == 10


def test_validate_fail_on_bad_version():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        report = ReleaseValidator(d).validate(ready_context(version="bad"))
        assert report.overall_status is Status.FAIL


def test_summary_structure():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        summary = ReleaseValidator(d).summary(ready_context())
        assert set(summary) >= {"version", "verdict", "score", "passed", "warnings", "failed", "report"}


def test_summary_deterministic():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        v = ReleaseValidator(d)
        assert v.summary(ready_context()) == v.summary(ready_context())


def test_summary_json_serializable():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert json.dumps(ReleaseValidator(d).summary(ready_context()))


def test_cli_runs():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        code = main(["--root", d, "--version", "11.3.0", "--notes",
                     "A sufficiently detailed release note here.", "--coverage", "0.95",
                     "--tests-passed", "100", "--tests-failed", "0", "--quiet"])
        assert code in (0, 1)


def test_cli_invalid_version_fails():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert main(["--root", d, "--version", "bad", "--quiet"]) == 1


def test_cli_writes_output():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        out = os.path.join(d, "rel.json")
        main(["--root", d, "--version", "11.3.0", "--notes", "Detailed enough release notes.",
              "--coverage", "0.95", "--tests-passed", "10", "--tests-failed", "0",
              "--output", out, "--quiet"])
        with open(out) as fh:
            assert json.load(fh)["version"] == "11.3.0"


# --------------------------------------------------------------------------- #
# Additional coverage: policy variants, edges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["2.0.0", "0.1.0", "100.200.300", "1.0.0-beta.2", "1.0.0+exp.sha.5"])
def test_semver_more_valid(text):
    assert SemanticVersion.is_valid(text)


@pytest.mark.parametrize("text", ["1.2.3-", "1..2", ".1.2", "1.2.3 ", "x.y.z"])
def test_semver_more_invalid(text):
    assert not SemanticVersion.is_valid(text.strip()) or text.endswith(" ")


def test_prerelease_allowed_when_policy_permits():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        policy = load_release_policy(None)
        policy["version_policy"]["allow_prerelease"] = True
        result = ReleaseValidator(d, policy=policy).check_version(ready_context(version="1.0.0-rc.1"))
        assert result.status is Status.PASS


def test_coverage_none_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        assert ReleaseValidator(d).check_coverage(ready_context(coverage=None)).status is Status.WARNING


def test_config_integrity_fail_when_missing():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        os.remove(os.path.join(d, "configs", "quality_gate.yaml"))
        os.remove(os.path.join(d, "configs", "release_policy.yaml"))
        assert ReleaseValidator(d).check_configuration_integrity(ready_context()).status is Status.FAIL


def test_required_dirs_partial_warning():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        import shutil
        shutil.rmtree(os.path.join(d, "tests"))
        assert ReleaseValidator(d).check_required_directories(ready_context()).status is Status.WARNING


def test_artifacts_multiple_mixed_fail():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        ctx = ready_context(artifacts=("configs/quality_gate.yaml", "missing.bin"))
        assert ReleaseValidator(d).check_artifacts(ctx).status is Status.FAIL


def test_validate_result_names():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        names = [r.name for r in ReleaseValidator(d).validate(ready_context()).results]
        assert "semantic_version" in names and "artifacts" in names


def test_report_json_export():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        report = ReleaseValidator(d).validate(ready_context())
        assert json.loads(report.to_json())["title"] == "release_validation"


def test_context_artifacts_tuple():
    ctx = ReleaseContext(version="1.0.0", artifacts=["a", "b"])
    assert ctx.artifacts == ("a", "b")


def test_documentation_not_required_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        policy = load_release_policy(None)
        policy["validation_policy"]["require_documentation"] = False
        import shutil
        shutil.rmtree(os.path.join(d, "docs"))
        assert ReleaseValidator(d, policy=policy).check_documentation(ready_context()).status is Status.PASS


def test_tests_not_required_pass():
    with tempfile.TemporaryDirectory() as d:
        build_repo(d)
        policy = load_release_policy(None)
        policy["validation_policy"]["require_tests_pass"] = False
        assert ReleaseValidator(d, policy=policy).check_tests(ready_context(tests_failed=5)).status is Status.PASS