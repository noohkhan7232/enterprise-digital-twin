# Interview Guide

## Recruiter Talking Points

Each point is backed by a repository artifact a reviewer can open.

1. **Scale with discipline** — 94 modules / 61,330 LOC in 23 packages, with 51,292 lines of tests
   (8,361 collected): near 1:1 test-to-source investment.
2. **Full production chain, not just models** — Docker → Kubernetes (10 manifests) → health-gated
   rollback → CI quality gates, all in-repo and wired together.
3. **One health definition everywhere** — the same `deployment/scripts/health_check.py` backs the
   container HEALTHCHECK, the K8s probes, and the rollback gate.
4. **Five AI capability families in one system** — digital twin, predictive, agentic, RAG,
   workflow — integrated by immutable contracts, not glue code.
5. **Policy-driven quality** — a twenty-gate CI quality engine configured from YAML; the gates are
   themselves unit-tested (247 tests, verified passing).
6. **Honesty as an engineering feature** — every published number is measured with reproduction
   commands; configured targets are never presented as benchmarks.
7. **Release management done properly** — audited cleanup, semantic versioning, changelog,
   security policy, citation metadata, zero broken README links at release.
8. **Deterministic by construction** — byte-identical demo and observability outputs across runs;
   tests assert exact values.

## 30 Technical Interview Questions (with evidence pointers)

### Architecture & System Design (1–10)

1. Why composition over inheritance for cross-layer integration? *(immutable value objects at
   boundaries — `src/` package structure)*
2. What does "single dominant dependency direction" buy you, and what does it cost?
3. How do immutable contracts affect memory and performance at fleet scale? *(see scaling notes in
   `docs/reports/complete_week6_fleet_digital_twin_report.md`)*
4. Walk through the event bus (`src/events/`) — how is ordering handled? Delivery guarantees?
5. How does the workflow engine keep multi-step state explicit and deterministic (`src/workflow/`)?
6. Where would you put a new capability layer, and what must it not do?
7. How is thread safety achieved (27 modules import `threading`)? What's the locking strategy?
8. Why an in-repo scheduler (`src/scheduler/`) instead of cron/Airflow?
9. What breaks first if two layers start sharing mutable state?
10. Defend the ten-layer decomposition against "this is over-engineering."

### MLOps & Production ML (11–18)

11. Explain the content-addressed artifact store (`src/mlops/`) — why hashes as identity?
12. How does the lineage graph answer "where did this prediction come from?"
13. Data drift vs concept drift in `src/monitoring/` — detection methods and thresholds?
14. What is "composite model health," and how are alerts routed?
15. How does the model registry work without requiring torch at import time
    (`src/models/base_model.py` checkpoint contract)?
16. How would you promote a model through the twenty-gate quality engine?
17. What's in `configs/release_policy.yaml`, and why is policy separated from engine code?
18. How is reproducibility enforced — what makes outputs byte-identical?

### Testing & Quality (19–24)

19. 51K lines of tests for 61K lines of source — how do you keep that suite fast and honest?
20. Why deterministic, framework-agnostic tests? What do you give up?
21. How do the CI/CD validators test themselves (`tests/week11_phase3_tests/`)?
22. The validator reports max cyclomatic complexity 36 across 20 hotspots — how would you pay that
    down without behaviour change?
23. What's your strategy for testing thread-safe components?
24. Type-hint coverage is 97.2% — where are the gaps and why?

### Deployment & Operations (25–30)

25. Walk through the multi-stage Dockerfile — what lives in the builder and why non-root +
    read-only rootfs?
26. Why one deterministic health check across Docker, K8s, and CI? What did that prevent?
27. Explain the rollback flow (`deployment/scripts/rollback.sh`) and its health gating.
28. What do the HPA + PodDisruptionBudget + NetworkPolicy manifests each protect against?
29. SLOs here are configured targets, not measurements — how would you turn them into measured
    SLIs in production?
30. How does the error-budget/burn-rate engine (`src/observability/`) decide when to halt releases?

## STAR Stories

**1. The clone that wouldn't import.**
*S:* Pre-release audit of a 94-module platform. *T:* Certify the repo release-ready. *A:* Found six
packages (`evaluation`, `executive`, `inference`, `predictive`, `training`, `workflow`) whose
`__init__.py` existed only locally and was never committed — every fresh clone would have broken
imports despite all local tests passing. Tracked, staged, and re-verified package integrity 23/23.
*R:* A clone-breaking defect invisible to local testing was caught before a single external user hit
it.

**2. 49 broken links on the front page.**
*S:* Final release verification. *T:* Verify README integrity. *A:* Wrote a repo-wide link checker;
found 49 broken relative links — the README described an idealized folder layout that had never
existed. Rather than restructure frozen folders, corrected every link to the real paths and
re-scanned all 105 markdown files to zero. *R:* Release approved; every documentation link a
reviewer clicks now resolves.

**3. Two config roots, zero behaviour changes.**
*S:* Audit found configuration split across `config/` and `configs/`. *T:* Consolidate without
touching runtime behaviour. *A:* Grepped every loader path across src, scripts, tests, Docker, K8s
and CI before moving anything; proved no code referenced the old root; migrated via
history-preserving `git mv`; re-ran the 247-test config-dependent suite. *R:* Single configuration
root, all tests green, no functional diff.

**4. Saying "collected," not "passing."**
*S:* Portfolio claims needed test numbers. *T:* Publish impressive but true metrics. *A:* Measured
8,361 collected tests but declined to claim them "passing" without a full verified run; published
the distinction plus reproduction commands, keeping the verified-passing claims scoped to the 247
and 1,503-test suites that were actually executed. *R:* Every number in the package survives
adversarial checking — which is the point.

**5. Additive architecture over twelve weeks.**
*S:* Twelve weekly increments from audio-ML research to enterprise platform. *T:* Grow capability
without destabilising predecessors. *A:* Enforced additive construction — each week's layer consumes
earlier layers through immutable contracts only; weekly reports and ADRs document each decision.
*R:* 23 packages that compose rather than entangle; the git history itself demonstrates the
discipline.

## Interview FAQ (rapid answers)

- **"Is this production-deployed?"** No — it's a reference platform. SLO values are configured
  targets; the deployment chain is real and exercisable, but no production traffic is claimed.
- **"Why doesn't `src/` import torch?"** The platform core is deliberately dependency-light;
  deep-learning modules define checkpoint/config contracts importable without a GPU stack, and the
  full research stack is pinned separately in `requirements.txt`.
- **"What would you build next?"** Measured SLIs from a running deployment, complexity paydown of
  the 20 identified hotspots, and benchmark execution against the published methodology
  (`week12_phase4_benchmarks/`).
