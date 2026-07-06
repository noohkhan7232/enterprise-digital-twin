# Verified Repository Metrics

**Measured:** 2026-07-04/05, at v1.0.1 (working tree). Only values in this file may be quoted in
the rest of the package. Anything not listed here is not a claim this project makes.

## Source Code

| Metric | Value | How measured |
|---|---:|---|
| Python modules in `src/` | 94 | `find src -name "*.py" -not -path "*__pycache__*" \| wc -l` |
| Source LOC in `src/` | 61,330 | `find src -name "*.py" ... -exec cat {} + \| wc -l` |
| Packages in `src/` | 23 | directory count; each with `__init__.py` (validator: 23/23) |
| Syntax-clean Python files | 174/174 | repository validator (`scripts/week_11_phase_3/validate_repository.py`) |
| Type-hint coverage | 97.17% | repository validator |
| Naming conformance | 98.98% | repository validator |
| Repository validator score | 87.45 | same script; known findings: max cyclomatic complexity 36 (20 hotspots), 5 star imports |

## Tests

| Metric | Value | How measured |
|---|---:|---|
| Test files | 68 | `find tests -name "test_*.py" \| wc -l` |
| Test LOC | 51,292 | `find tests -name "*.py" ... \| wc -l` |
| Tests collected (full suite) | **8,361** | `python -m pytest tests --collect-only -q` (22.8s) |
| Verified passing — CI/CD validation suite | 247/247 | `python -m pytest tests/week11_phase3_tests -q` (run 2026-07-04, 3.64s) |
| Verified passing — production-engineering suite at v1.0.0 | 1,503/1,503 | recorded in `week12_phase5_release/repository_statistics.md` |

> **Scope note:** "8,361 collected" is a collection count, not an execution result. A full-suite
> execution was not performed during package preparation and is therefore not claimed. The
> historical `repository_statistics.md` figures (30 modules / 10,620 LOC / 1,503 tests) are an
> explicitly scoped v1.0.0 snapshot covering the production-engineering subsystems only; the
> current full-tree figures above supersede them for whole-repository claims.

## Deployment & CI/CD

| Metric | Value |
|---|---:|
| Dockerfiles (prod multi-stage non-root + dev) | 2 |
| Docker Compose files (dev, prod) | 2 |
| Kubernetes manifests | 10 |
| Deployment scripts (deploy ×2, rollback, health check) | 4 |
| GitHub Actions workflows | 3 |
| YAML policy/config files under `configs/` | 6 (`config.yaml`, `quality_gate.yaml`, `release_policy.yaml`, `week11_phase5/` ×3) |

## Documentation & Assets

| Metric | Value |
|---|---:|
| Markdown files (repo-wide) | 105 |
| Broken relative links at release | 0 (all 105 files scanned) |
| Image assets | 59 |
| Research paper | IEEE-style, `docs/week 12/research_paper.md` + `references.bib`, appendices, glossary |
| Community files | LICENSE (MIT), SECURITY.md, CITATION.cff, CONTRIBUTING, CODE_OF_CONDUCT, issue + PR templates |

## Versioning

| Reference | Value |
|---|---|
| `week12_phase5_version/VERSION` | 1.0.1 |
| README release badge | v1.0.1 |
| `CHANGELOG.md` head entry | [1.0.1] — 2026-07-04 |
| `CITATION.cff` | 1.0.1 |
| Git tag | `v1.0.1` exists (see caveat below) |

> **Release caveat (disclosed for integrity):** at measurement time, tag `v1.0.1` pointed at commit
> `fcdf49c` (2026-07-03), which predates the stabilization changes present in the working tree
> (config consolidation, SECURITY.md, README link repairs, version sync). Until the staged work is
> committed and the tag updated (or a v1.0.2 cut), the published tag does not contain everything
> this package describes. Working-tree measurements are what this package reports.

## Technology Stack (verified by import analysis)

**Imported in `src/`:** numpy, scipy, pandas, scikit-learn, librosa, soundfile, pywavelets,
matplotlib, seaborn, psutil — plus standard library (typing, dataclasses, logging, threading,
pathlib, json, hashlib, …).
**Pinned in `requirements.txt` (research lineage):** the above plus torch, torchaudio, mlflow,
fastapi, uvicorn, captum, shap, audiomentations, noisereduce, pytest.
**Not imported at module level in `src/`:** torch (deep-learning contracts are torch-optional by
design).

## Reproduction

Every measurement above can be reproduced from a fresh clone with the commands shown, plus:

```bash
python scripts/week_11_phase_3/validate_repository.py --root .
python -m pytest tests --collect-only -q
python -m pytest tests/week11_phase3_tests -q
```
