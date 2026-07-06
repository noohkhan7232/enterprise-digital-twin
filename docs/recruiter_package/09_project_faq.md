# Project FAQ

## Recruiter FAQ

**Q: What is this project in one sentence?**
An open-source enterprise platform that runs digital twins, predictive models, AI agents, and
retrieval-augmented knowledge as one governed, observable, deployable system — 61K lines of Python
with a 51K-line test suite, Docker/Kubernetes deployment, and full MLOps.

**Q: Is it real code or a tutorial project?**
Real: 94 modules across 23 packages, 8,361 collected tests, a hardened container image, 10
Kubernetes manifests, and CI quality gates that are themselves unit-tested. Every number has a
published measurement command.

**Q: What roles does it support?**
Senior/staff platform engineer, MLOps engineer, production ML engineer, backend/systems engineer
with AI focus.

**Q: How long did it take?**
Twelve documented weekly increments (per-week engineering reports in `docs/`), released as v1.0.0
then stabilized as v1.0.1.

**Q: Why is the repo called `wind-turbine-acoustics`?**
The project began as wind-turbine acoustic-monitoring ML research (weeks 1–6, preserved in
`docs/reports/` and `notebooks/`) and grew into the general platform. The lineage is kept honestly
rather than rewritten.

## Hiring Manager FAQ

**Q: Is anything here running in production?**
No, and the project never claims so. It is a reference implementation: the deployment chain is
real and exercisable, but SLO values are configured targets and no traffic/incident history exists.

**Q: What is verified vs claimed?**
Verified: module/LOC/test counts, 247-test CI/CD suite passing, 1,503-test v1.0.0 suite passing,
validator results, zero broken doc links. Collected-but-not-executed: the full 8,361-test suite.
The repo's own statistics document models this discipline.

**Q: Where are the known weaknesses?**
Self-reported: 20 cyclomatic-complexity hotspots (max 36), 5 star imports, unmeasured SLOs, no
benchmark executions, and a social-preview image with an outdated URL. See
`08_hiring_manager_review.md` — nothing is hidden.

**Q: Was AI assistance used?**
The repository contains AI-generated diagram assets (one figure retains its generation prompt in
the source image), and the documentation practices are consistent with AI-assisted drafting.
Interviewers should probe understanding directly — the candidate's fluency across the 23 packages
is the real test, and the interview guide (`04_interview_guide.md`) invites exactly that probing.

**Q: How does it handle the classic solo-project gaps (review, collaboration, on-call)?**
It doesn't pretend to. Contribution/community machinery exists (templates, CONTRIBUTING, CoC), but
collaboration evidence must come from the candidate's work history, not this repo.

## Technical FAQ

**Q: Why composition with immutable value objects instead of shared services?**
Boundary crossings become serialisable, diffable, and testable in isolation; no layer can corrupt
another's state; determinism (byte-identical outputs) becomes achievable, which the test suite
exploits with exact assertions.

**Q: Why doesn't `src/` import torch if the project claims deep learning?**
`src/models/` defines architectures and a checkpoint contract (config + state-dict validation)
written to be importable without a GPU stack, so registry/CI tooling runs anywhere; the full
research stack (torch, torchaudio, librosa, captum, shap) is pinned in `requirements.txt`. Model
code paths are unit-tested (`tests/test_acoustic_transformer.py`, `tests/test_cnn_bilstm*.py`,
`tests/test_anomaly_autoencoder.py`).

**Q: What makes the health check "deterministic," and why one health check?**
`deployment/scripts/health_check.py` produces reproducible results from explicit inputs and backs
the Docker HEALTHCHECK, both Kubernetes probes, and the rollback gate — so "healthy" means the same
thing on every surface and cannot drift.

**Q: How is quality gated in CI?**
`scripts/week_11_phase_3/quality_gate.py` evaluates twenty gates configured from
`configs/quality_gate.yaml`; a release validator enforces semantic-version and policy rules from
`configs/release_policy.yaml`; a deployment-readiness validator checks the deploy surface. All
three are unit-tested (247 tests, passing).

**Q: RAG without an external vector database?**
`src/knowledge/` implements retrieval intelligence with hybrid retrieval configuration over a
versioned corpus index, with evidence grounding and serialisable state — the architectural
contracts matter more than any specific store, and no external service is required to run or test
it.

**Q: How would this scale?**
Fleet-level scaling paths are analysed in
`docs/reports/complete_week6_fleet_digital_twin_report.md` (business-unit sharding via the existing
partition, summarisation-on-the-fly to make memory proportional to assets rather than samples);
horizontal scaling is provided by the HPA manifest; the honest answer to "proven at what N?" is
that no load benchmarks have been executed.

**Q: What are the SLOs?**
Configured targets in `configs/week11_phase5/reliability_policy.yaml`, evaluated by the
SLI/SLO/error-budget engine in `src/observability/` — explicitly labelled as targets, not
measurements, throughout the docs.
