# Week 11 — Phase 3: Enterprise CI/CD Pipeline & Automated Validation Platform

> Subsystems: `scripts/` (validation library + tools), `configs/` (policy), `.github/workflows/` (orchestration).
> Status: production-ready, deterministic, pure Python (+ PyYAML, with a built-in fallback parser).
> Integration model: **additive only** — adds new files; modifies no prior week.

---

## 1. Business Motivation

Shipping a model platform to production is not a single act of courage; it is a
repeatable, auditable process. Every release of the Enterprise Digital Twin
platform touches MLOps tracking, drift monitoring and downstream decision
systems, and a regression in any of them is expensive and slow to detect. The
purpose of this subsystem is to make "is this repository fit to release?" a
deterministic, machine-checkable question rather than a matter of reviewer
fatigue.

It does so the way mature platform-engineering organisations do — Google Cloud
Build, Azure DevOps, GitHub Enterprise Actions — but reimplemented in owned,
explainable Python so that every gate, score and verdict can be reproduced
byte-for-byte and defended in an audit. The platform answers three questions on
every change: does the repository meet quality standards (a 0–100 quality
score across twenty gates)? is this specific version safe to release
(PASS/WARNING/FAIL)? and is the artifact ready to deploy (a readiness
percentage)? Together they form a release gate that is fast enough to run on
every pull request and strict enough to block a bad release.

## 2. Architecture

The subsystem is layered. A single validation library
(`scripts/validate_repository.py`) defines the immutable result value objects
and the AST-based checks; the three higher-level tools compose it.

```
                      validate_repository.py
        (Status · CheckResult · ValidationReport · load_yaml ·
         AST helpers · RepositoryValidator · CLI)
                              |
        +---------------------+----------------------+
        |                     |                      |
   quality_gate.py     release_validator.py   deployment_readiness.py
   (20 gates ->         (10 checks ->          (10 checks ->
    score 0-100)         PASS/WARN/FAIL)        readiness %)
        |                     |                      |
        +---------------------+----------------------+
                              |
                   .github/workflows/*.yml
        (enterprise_ci · enterprise_release · dependency_scan)
```

Cross-cutting design choices:

- **Dependency Injection.** Every validator takes an explicit repository root
  and an injected configuration, so the same code validates the live repository
  and synthetic trees in tests. Quality and release tools accept injected
  measured metrics (coverage, test counts) rather than reaching out to the
  environment.
- **Strategy pattern.** Each quality gate is a named, weighted strategy in an
  ordered registry; gates can be reweighted or replaced through configuration
  without touching the engine.
- **Immutability.** `CheckResult` and `ValidationReport` are
  `frozen=True, slots=True` dataclasses with symmetric `to_dict`/`from_dict`
  and sorted JSON export.
- **Single source of severity.** A three-level `Status` (PASS / WARNING / FAIL)
  and a most-severe aggregation rule are shared by all four tools.

## 3. Quality Gate Design

`QualityGate` runs twenty deterministic gates and reduces them to a weighted
score in `[0, 100]`. Six gates delegate to the repository validator
(structure, syntax, imports, type hints, documentation, test discovery); the
remainder are purpose-built: pytest execution and coverage threshold (which
consume CI-measured metrics), code complexity, dependency validation,
configuration validation, package integrity, architecture consistency, naming
convention, JSON serialization, thread safety, deterministic behaviour, MLOps
integration, monitoring integration and deployment readiness.

The score is the weight-normalised mean of each gate's `[0, 1]` score times one
hundred, with weights drawn from `quality_gate.yaml`. The verdict is FAIL if any
gate fails or the score falls below the configured minimum, WARNING if warnings
remain, and PASS otherwise. Two design points deserve emphasis. First, gates
that require runtime measurement (test results, coverage) are **policy checks
over injected values**: the CI workflow runs pytest and coverage and feeds the
numbers in, so the gate is deterministic and unit-testable while still reflecting
real measurements. Second, static gates such as thread safety and deterministic
behaviour are deliberately conservative heuristics — they flag modules that
mutate shared state without a lock, or that call `datetime.now`/`time.time`
without an injected clock or seed — surfacing risk for human review rather than
asserting proof.

## 4. Release Validation

`ReleaseValidator` evaluates a `ReleaseContext` (version, release notes,
coverage, test results, working-tree cleanliness, declared artifacts) against
the release policy and emits ten checks: semantic version, release notes,
documentation, test success, coverage, repository cleanliness, configuration
integrity, required files, required directories and artifacts. Versions are
parsed by a strict SemVer implementation; a prerelease is downgraded to a
warning unless policy explicitly permits prereleases in production. The overall
verdict is the most severe check result, so a single invalid version or a failing
test set blocks the release while soft issues (missing artifacts, short notes)
surface as warnings.

## 5. Deployment Readiness

`DeploymentReadiness` scores ten operational concerns: Docker readiness
(a Dockerfile with a base image and an entrypoint), Kubernetes readiness
(manifests or `kind:` markers), monitoring readiness (the Phase 2 subsystem),
health endpoints, configuration files, environment variables, release manifests,
security configuration, rollback support and recovery readiness. Each check
returns a `[0, 1]` score; the readiness percentage is their mean times one
hundred. Operational signals that are advisory rather than mandatory (health
checks, environment templates, security policy) degrade to warnings rather than
failures, so the percentage communicates how close to deployable the repository
is rather than collapsing to a single pass/fail.

## 6. GitHub Actions

Three workflows orchestrate the platform. `enterprise_ci.yml` runs on every
push, pull request and manual dispatch: checkout, Python setup, dependency
install, static compile, repository validation, pytest with coverage, coverage
extraction, quality gate, monitoring and MLOps import/demo validation, build
(deployment-readiness) validation, a step summary and report upload.
`enterprise_release.yml` runs on `v*.*.*` tags (or a dispatched version): it
resolves and validates the version, runs tests and coverage, runs the release
validator, validates documentation and deployment readiness, packages a
tarball artifact and publishes a release summary. `dependency_scan.yml` runs on
manifest changes and weekly: it scans `requirements.txt` for duplicates,
version conflicts, unsafe packages and dependencies declared but never imported,
failing the build on conflicts or unsafe packages. The Python tools are the
source of truth; the YAML is thin orchestration that calls them.

## 7. Configuration

`configs/quality_gate.yaml` holds thresholds (coverage, complexity,
documentation, type hints, naming, quality score, monitoring, MLOps,
deployment) and per-gate weights, plus the required packages and required config
files. `configs/release_policy.yaml` holds the release strategy (progressive
rollout with a canary percentage and bake time), the version policy (SemVer,
prerelease and build-metadata rules, tag matching), the rollback policy
(automatic rollback, window, retained versions), the approval policy (required
approvals, code-owner and release-manager gates, incident blocking) and the
validation policy (test, coverage, documentation, cleanliness, changelog and
quality-score requirements plus required files and directories). Both loaders
merge file values over built-in defaults, so the tools run even when a config is
absent and can be tuned without code changes.

## 8. Engineering Decisions

- **Validate by root, not by environment.** Every validator takes a repository
  root, so the live tree and synthetic temporary trees are validated by the same
  code path — the foundation of the 247-test suite.
- **Inject measured metrics.** Coverage and test results are produced by the CI
  workflow and injected into the quality and release tools, keeping the tools
  deterministic and offline-testable while still policy-accurate.
- **One shared library.** The three tools import the value objects, YAML loader
  and AST helpers from `validate_repository.py` rather than duplicating them,
  giving a single definition of `Status`, scoring and JSON export.
- **Self-contained YAML.** PyYAML is used when present; a deterministic
  fallback parser covers the simple subset these configs use, so config loading
  never hard-depends on a third-party package.
- **Honest signals over green-washing.** Heuristic gates report risk for review
  instead of asserting proof, and the quality gate will legitimately fail a
  genuinely complex or not-yet-deployable repository rather than masking it.

## 9. Performance

Every check is at worst linear in the number of source files times the size of
each file, with per-file AST parsing dominating. A full repository validation,
the twenty-gate quality evaluation, release validation and deployment readiness
each complete in well under interactive latency on this repository, and all are
single-pass over the file tree. The CI workflow's cost is dominated by
dependency installation and the pytest run, not by the validators.

## 10. Complexity

| Operation | Complexity |
|-----------|------------|
| `iter_python_files` | O(n) over tree entries, sorted O(n log n) |
| per-file AST parse + walk | O(s) in file size |
| `RepositoryValidator.validate_all` | O(F · s) over F files |
| `QualityGate.evaluate` (20 gates) | O(F · s) dominated by static scans |
| `ReleaseValidator.validate` (10 checks) | O(R) over required files/dirs |
| `DeploymentReadiness.evaluate` (10 checks) | O(F) over tree for manifest scans |
| report `to_json` | O(C log C) over C checks (sorted keys) |

## 11. Enterprise Applications

- **Merge gating.** `enterprise_ci.yml` blocks pull requests that drop below the
  quality threshold or break a hard gate.
- **Release governance.** `enterprise_release.yml` enforces the release policy on
  tagged releases and produces an auditable release report and artifact.
- **Supply-chain hygiene.** `dependency_scan.yml` catches conflicting,
  duplicated, unused or unsafe dependencies on a schedule and on manifest
  changes.
- **Deployment sign-off.** The readiness percentage gives release managers a
  single, reproducible number for go/no-go decisions.

## 12. Integration with Enterprise MLOps

The quality gate's `mlops_integration` gate verifies that the Phase 1 MLOps
package is present, and the CI workflow imports `mlops` and runs its tracker
demo as a smoke test on every change. Because the platform validates by root and
never imports week-specific internals, it exercises the MLOps subsystem as a
black box — exactly as a release pipeline should — without coupling to it.

## 13. Integration with Monitoring

Symmetrically, the `monitoring_integration` gate and the CI workflow's
monitoring step import the Phase 2 monitoring package and run its prediction
monitor demo, ensuring the drift, health and quality subsystem builds and runs
before any release. Deployment readiness also credits the presence of the
monitoring subsystem as a precondition for safe production operation.

## 14. Integration with Deployment

Deployment readiness is consumed in two places: directly, as the build-validation
step of the CI workflow and the readiness step of the release workflow, and
indirectly, as the twentieth quality gate. This makes deployment concerns —
containerisation, orchestration, health, rollback and recovery — first-class
inputs to both the quality score and the release verdict, closing the loop from
source quality through to operational readiness.