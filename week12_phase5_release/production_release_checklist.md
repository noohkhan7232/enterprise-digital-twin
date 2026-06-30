# Production Release Checklist — v1.0.0

A checklist for the v1.0.0 public release and for the subsequent step of deploying into an untrusted
production environment. Items are grouped by whether they gate **public release of the repository** or
**production deployment**. This checklist records intent and status; it does not assert completion of
items that depend on the deploying organisation.

---

## A. Public Release of the Repository

### A.1 Content completeness
- [x] Architecture, capability and production-engineering subsystems documented.
- [x] Complete documentation set (architecture overview, quick start, developer/deployment guides, FAQ).
- [x] Research paper with bibliography, appendices and glossary.
- [x] Demonstration and presentation assets.
- [x] Engineering validation package.
- [x] Release assets (notes, summary, audit, statistics, this checklist, engineering summary, overview).

### A.2 Metadata and licensing
- [x] `LICENSE` present (MIT).
- [x] `CITATION.cff` present and valid.
- [x] `CHANGELOG.md` present with the v1.0.0 entry.
- [x] `VERSION` file set to `1.0.0`.
- [x] `RELEASE_MANIFEST.md` lists all major directories and deliverables.

### A.3 Community files
- [x] `CONTRIBUTING.md`.
- [x] `CODE_OF_CONDUCT.md`.
- [x] Issue templates (bug report, feature request).
- [x] Pull-request template.

### A.4 Statistics integrity
- [x] Repository statistics measured directly from the repository.
- [x] Unmeasured fields (e.g., git history) left blank and marked pending.
- [x] No fabricated benchmark or repository figures.

### A.5 Cosmetic items (recommended before publishing)
- [ ] Replace placeholder owner/org references (`<org>`) in README, citation and changelog.
- [ ] Replace placeholder author names with real identities.
- [ ] Set the MIT copyright holder line appropriately.
- [ ] Insert real images where figure/screenshot placeholders are marked.
- [ ] Add a top-level `README.md` (or pointer) if the canonical README lives under `docs/`.
- [ ] Optionally add `SECURITY.md` describing private vulnerability reporting.
- [ ] Initialise version control and create the `v1.0.0` tag.

## B. Production Deployment in an Untrusted Environment (additional)

### B.1 Security hardening
- [ ] Source secrets from an external secrets manager; enable encryption at rest.
- [ ] Enable repository secret scanning.
- [ ] Add container image vulnerability scanning to the pipeline.
- [ ] Hash-pin dependencies; generate a software bill of materials.
- [ ] Add an authentication/authorisation layer at the ingress; document trust boundaries.
- [ ] Add admission-policy enforcement and least-privilege RBAC.
- [ ] Commission an independent security audit (the included review is architectural only).

### B.2 Measurement and tuning
- [ ] Execute the benchmark methodology; populate the results template.
- [ ] Measure reliability under sustained operation; add concurrency and fault-injection testing.
- [ ] Tune autoscaling and resource limits from measured data.
- [ ] Agree real SLO targets with stakeholders; rehearse incident response.

### B.3 Operational readiness
- [ ] Configure log retention, access control and field scrubbing.
- [ ] Set registry and artifact retention policy at production scale.
- [ ] Establish on-call and escalation procedures.

## C. Sign-off

| Gate | Decision | Owner | Date |
|------|----------|-------|------|
| Public release (Section A) | Approve pending A.5 | | |
| Production deployment (Section B) | Conditional on B.1–B.3 | | |

## Notes

Section A items marked complete reflect the repository's content as built and verified. Section A.5 and
all of Section B depend on the publishing or deploying organisation and are intentionally left
unchecked.
