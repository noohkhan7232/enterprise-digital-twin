# Release Notes — v1.0.1 "Production Stabilized"

**Enterprise Digital Twin & Decision Intelligence Platform**
**Release:** 1.0.1 · Production stabilization release
**Date:** 2026-07-04
**License:** MIT

These notes cover the v1.0.1 stabilization release only. For the platform's capabilities,
architecture and verification record, see the first-release notes in
[`release_notes_v1.0.md`](release_notes_v1.0.md); v1.0.1 changes none of that. This release contains
**no functional, architectural, or behavioural changes** — it is a repository-hygiene and
release-preparation pass performed against the findings of a full repository audit.

---

## Changes

### Repository hygiene

- Consolidated all configuration under a single `configs/` root (formerly split across `config/`
  and `configs/`). No code, container, manifest or CI path referenced the old root; no runtime
  impact.
- Removed dead files identified by the audit: an obsolete README copy, a stale structure dump,
  editor temp files, and a duplicate figure.
- Fixed documentation asset naming typos ("Weak" → "Week", accidental spaces in filenames) with
  history-preserving renames.
- Extended `.gitignore` coverage (editor temp files, tool caches, Windows system files).

### Package integrity

- Added previously untracked package `__init__.py` files (`src/evaluation`, `src/executive`,
  `src/inference`, `src/predictive`, `src/training`, `src/workflow`) so fresh clones import
  correctly.

### Release presentation

- Synchronized all version references (`VERSION`, README badge, `CITATION.cff`, `CHANGELOG.md`)
  to 1.0.1.
- Repaired all broken relative links and placeholder URLs in the README and public documentation.
- Added a `SECURITY.md` security policy.

## Verification

- Repository validation: structure, syntax (174/174 files), package integrity (23/23 packages)
  passing; 6,887 tests discovered.
- The config-path-dependent CI/CD validation suite (`tests/week11_phase3_tests`, 247 tests) was
  re-run after the configuration consolidation: all passing.

## Upgrade Notes

- If you referenced YAML under `config/` (singular), update paths to `configs/`. No other action
  is required; all interfaces, images, and manifests are unchanged from v1.0.0.
