# Repository Statistics — v1.0.0

All figures below were measured directly from the repository at the time of the v1.0.0 audit. The
collection methodology is documented so that any reviewer can reproduce the counts. Figures that could
not be measured in this environment are listed under **Pending Final Repository Audit** with blank
fields, in line with the project's no-fabrication policy. No values are reused from earlier
documentation without re-measurement.

**Scope note.** Source-code figures cover the production-engineering subsystems present in this
repository snapshot (MLOps, monitoring, observability, CI/CD scripts, deployment). The five capability
layers (digital twin, predictive intelligence, agentic AI, knowledge/RAG, workflow) are documented
architecturally and are not present as source in this snapshot; they are therefore not included in the
source-code counts. See `repository_audit.md`.

---

## 1. Measured Statistics

### 1.1 Source code (production-engineering subsystems)

| Metric | Value |
|--------|------:|
| Production modules (`.py` in `src/`, `scripts/`, `deployment/`) | 30 |
| — of which `src/` packages (mlops, monitoring, observability) | 25 files |
| Production source LOC | 10,620 |
| — of which `src/` LOC | 8,963 |

### 1.2 Tests

| Metric | Value |
|--------|------:|
| Test files (`tests/test_*.py`) | 27 |
| Test LOC | 8,437 |
| Tests collected | 1,503 |
| Tests passed | 1,503 |
| Tests failed / skipped | 0 / 0 |

### 1.3 Documentation

| Metric | Value |
|--------|------:|
| Markdown files (all areas) | 73 |
| Markdown lines | 7,909 |

### 1.4 Configuration

| Metric | Value |
|--------|------:|
| Configuration files (`configs/*.yaml`) | 5 |
| All YAML/YML files (incl. k8s, workflows, compose, citation) | 20 |
| Kubernetes manifests | 10 |
| CI/CD workflow definitions | 3 |

### 1.5 Figures and other assets

| Metric | Value |
|--------|------:|
| Figures (`.png`/`.svg`/`.jpg`) | 1 |
| Dockerfiles | 2 |
| Shell scripts | 4 |

### 1.6 Aggregate

| Metric | Value |
|--------|------:|
| Total Python files (production + tests) | 57 |
| Total Python LOC (production + tests) | 19,057 |
| Total deliverable files (excl. scratch, `__pycache__`, VCS) | 162 |
| Total files (incl. scratch, excl. `__pycache__`, VCS) | 165 |
| Repository size (excl. `__pycache__`) | 2.2 MB |

### 1.7 Dependencies

| Type | Value |
|------|-------|
| Runtime dependencies | NumPy, PyYAML |
| Development/test dependencies | (test runner; framework-agnostic suite) |
| Pinned dependency manifest present | No (`requirements.txt` not present in snapshot) |

---

## 2. Collection Methodology

The figures above were collected with standard command-line tools from the repository root. The
procedures are listed so they can be reproduced exactly.

- **LOC (source):** count lines of all `.py` files under `src/`, `scripts/` and `deployment/`,
  excluding `__pycache__`. Tooling: `find ... -name '*.py' -not -path '*/__pycache__/*' -exec cat {} + | wc -l`.
- **Modules:** count of the same `.py` files (file count rather than line count).
- **Tests:** test files counted as `tests/test_*.py`; test LOC via `cat tests/test_*.py | wc -l`; test
  count obtained by collecting and executing the suite with the in-repository runner and reading the
  reported `collected`/`passed` totals.
- **Files (total):** `find . -type f` excluding `__pycache__` and any version-control directory; the
  deliverable count additionally excludes scratch artifacts (the figure-generation script and the local
  test runner) that are not part of the released deliverable set.
- **Documentation:** count and line-count of all `.md` files, excluding `__pycache__`.
- **Figures:** count of `.png`, `.svg` and `.jpg` files, excluding `__pycache__`.
- **Configuration:** count of `configs/*.yaml`, and separately all `.yaml`/`.yml` across the repository
  (which includes Kubernetes manifests, workflows, compose files and the citation file).
- **Dependencies:** runtime dependencies determined from imports (NumPy, PyYAML); recorded as such
  because no pinned dependency manifest is present in this snapshot.
- **Repository size:** `du -sh` from the repository root, excluding `__pycache__`.
- **Git history:** would be collected with `git log`/`git shortlog` (commit count, contributors, first
  and last commit dates, branches, tags). Version control is **not initialised** in this snapshot, so
  these metrics are unavailable and are left blank below.

---

## 3. Pending Final Repository Audit

The following fields are derived from version-control history, which is **not initialised** in this
repository snapshot. They are intentionally left blank and must be measured after the repository is
placed under version control and tagged.

| Field | Value |
|-------|-------|
| Total commits | |
| Contributors | |
| First commit date | |
| Last commit date | |
| Branches | |
| Tags (expected: `v1.0.0`) | |
| Commit activity (per period) | |

> If, at the time of the final audit, any of the measured figures in Section 1 cannot be re-verified,
> they should likewise be moved here and left blank rather than estimated. As of this snapshot, all
> Section 1 figures were measured successfully; only the version-control–derived fields above are
> pending.

---

## 4. Integrity Statement

These statistics are measured, not invented. Source-code figures are scoped to the production-
engineering subsystems actually present in the repository; the capability layers are documented
architecturally and are excluded from source counts. No benchmark or performance numbers appear in this
document, by design — runtime performance is addressed by `../validation/benchmark_methodology.md` and
the empty `../benchmarks/benchmark_results_template.md`.
