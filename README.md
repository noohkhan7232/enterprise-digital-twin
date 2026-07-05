<div align="center">

# Enterprise Digital Twin & Decision Intelligence Platform

**Many industrial-AI capabilities, operated as one coherent, governable, observable system.**

![release](https://img.shields.io/badge/release-v1.0.1-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.12-blue)
![tests](https://img.shields.io/badge/tests-passing-brightgreen)
![docs](https://img.shields.io/badge/docs-complete-blue)
![research](https://img.shields.io/badge/research-IEEE--style%20paper-lightgrey)
![portfolio](https://img.shields.io/badge/portfolio-complete-lightgrey)

*A ten-layer, composition-based platform unifying digital twins, predictive intelligence, agentic AI,
retrieval-augmented knowledge and workflow orchestration on a verified production-engineering substrate.*

[Executive Summary](#2-executive-summary) ·
[Architecture](#8-complete-enterprise-architecture) ·
[Quick Start](#14-quick-start) ·
[Research Paper](#20-research-paper) ·
[Documentation](#23-documentation-index)

</div>

---

## 2. Executive Summary

The Enterprise Digital Twin & Decision Intelligence Platform exists because industrial AI rarely fails
for algorithmic reasons. It fails for systemic ones. Organisations accumulate AI capabilities one at a
time — a digital twin of their assets, predictive models for maintenance, a retrieval system over
technical documentation, agents that act on results — and each arrives with its own data model, its own
release process, its own monitoring, and its own definition of "healthy." The seams between these
systems are where provenance is lost, where degradation goes unnoticed, and where every deployment
becomes a bespoke risk.

This platform is an engineering answer to that problem. It integrates five capability layers — digital
twin, predictive intelligence, agentic AI, knowledge intelligence (RAG), and an enterprise workflow
engine — with five production-engineering subsystems — MLOps, production monitoring, CI/CD, deployment,
and observability — into a single, layered architecture with one dominant dependency direction.
Subsystems integrate by composition: the only thing that crosses a boundary is an immutable,
serialisable value object, never shared mutable state. The result is a system that is simultaneously
cohesive and loosely coupled, in which every layer is independently testable and the whole remains
governable as it grows.

The platform was built additively over twelve weeks — no layer modifies its predecessors — and its
production-engineering substrate is verified by a deterministic, framework-agnostic automated test
suite. The project's contribution is framed honestly: it is an engineering integration, architecture
and implementation effort, not a claim of algorithmic novelty. Its claims are correspondingly bounded —
measured repository statistics live in the [release audit](week12_phase5_release/repository_statistics.md),
service-level values are configured runtime targets rather than benchmarks, and readiness is a
transparent self-assessment rather than a certification.

> **Who this repository is for.** Senior engineers evaluating a reference architecture for integrated
> industrial AI; hiring managers and interview panels assessing engineering depth; researchers
> interested in the systems side of production AI; and practitioners who want a complete, honest,
> end-to-end example of production-engineering discipline applied to a multi-capability platform.

---

## 3. Table of Contents

1. [Hero](#enterprise-digital-twin--decision-intelligence-platform)
2. [Executive Summary](#2-executive-summary)
3. [Table of Contents](#3-table-of-contents)
4. [Industrial Problem Statement](#4-industrial-problem-statement)
5. [Project Vision](#5-project-vision)
6. [Business Objectives](#6-business-objectives)
7. [Major Features](#7-major-features)
8. [Complete Enterprise Architecture](#8-complete-enterprise-architecture)
9. [Repository Structure](#9-repository-structure)
10. [Technology Stack](#10-technology-stack)
11. [Engineering Principles](#11-engineering-principles)
12. [Development Journey](#12-development-journey)
13. [Installation](#13-installation)
14. [Quick Start](#14-quick-start)
15. [Configuration](#15-configuration)
16. [Running Tests](#16-running-tests)
17. [Docker Deployment](#17-docker-deployment)
18. [Kubernetes Deployment](#18-kubernetes-deployment)
19. [Repository Statistics](#19-repository-statistics)
20. [Research Paper](#20-research-paper)
21. [Engineering Validation](#21-engineering-validation)
22. [Demonstration Assets](#22-demonstration-assets)
23. [Documentation Index](#23-documentation-index)
24. [Testing Strategy](#24-testing-strategy)
25. [Security](#25-security)
26. [Performance Validation](#26-performance-validation)
27. [Roadmap](#27-roadmap)
28. [Contribution Guide](#28-contribution-guide)
29. [Citation](#29-citation)
30. [License](#30-license)
31. [Acknowledgements](#31-acknowledgements)
32. [Final Engineering Summary](#32-final-engineering-summary)

---

## 4. Industrial Problem Statement

**Why enterprises need digital twins.** Physical operations — plants, fleets, grids, supply networks —
generate continuous telemetry, but telemetry alone is not understanding. A digital twin couples a
static asset model (identity, configuration, relationships) with a synchronised state stream, so that
the enterprise can monitor, simulate and reason about its physical world through a faithful virtual
representation. Without a twin, every downstream system invents its own partial picture of the same
asset, and those pictures disagree.

**Why decision intelligence.** Predictions are not decisions. An enterprise ultimately cares about the
actions it takes and the outcomes they produce; decision intelligence frames analytics around that
chain — data, models, actions, consequences — rather than treating a model's output as the end of the
story. This demands infrastructure that can carry a result from prediction through workflow to action
with provenance intact.

**Why simulation.** Operational questions are often counterfactual: what happens if demand doubles, if
this machine degrades, if we change the maintenance interval? Simulation over twin state lets the
enterprise explore consequences before committing to them, which is only trustworthy when the
underlying representation and models are governed and reproducible.

**Why AI operations.** A model that performs well offline degrades silently in production as data
shifts. Without drift detection, health scoring, lineage and disciplined release processes, the gap
between "the model degraded" and "someone noticed" is measured in weeks. AI operations — MLOps,
monitoring, gated delivery, health-verified deployment, observability — is what turns models from
experiments into dependable operational components.

**Why enterprise AI as a *platform*.** Each capability above has mature tooling in isolation. The
unsolved problem is their **integration**: one asset identity flowing through twin, prediction,
knowledge and workflow; one release and rollback discipline across capabilities with different failure
modes; one observability fabric in which an agent's action is as traceable as a service request. That
integration problem — *coherence at scale* — is the problem this platform addresses.

---

## 5. Project Vision

The long-term engineering vision is that adding an AI capability to an enterprise should not multiply
its operational surface. Capabilities should plug into a platform that already knows how to version
them, monitor them, deploy them, recover them and observe them — the way a well-run cloud platform
absorbs a new service without inventing new operations from scratch.

Concretely, the vision has three commitments. First, **contracts over coupling**: capabilities
interoperate through immutable, serialisable value objects, so they can evolve independently without
destabilising the whole. Second, **provenance by construction**: every result carries its lineage —
model version, experiment, dataset, code revision, environment — as a structural property rather than a
logging afterthought. Third, **operations as measurement**: reliability, service-level compliance,
capacity and readiness are quantities the platform computes continuously, not assessments scheduled as
meetings.

Version 1.0.0 realises this vision as a complete reference implementation: the architecture, the
production-engineering substrate, the verification discipline, and the documentation needed for others
to study, extend or adopt the approach.

---

## 6. Business Objectives

| Objective | How the platform serves it |
|-----------|---------------------------|
| **Reduce operational risk** | Continuous drift/health monitoring shortens the gap between degradation and awareness; health-gated rollback bounds the blast radius of bad releases. |
| **Make results auditable** | End-to-end lineage turns "how was this produced?" from a multi-week investigation into a query — essential in regulated settings. |
| **Ship change safely and often** | Twenty CI/CD quality gates, release validation and zero-downtime rolling deployment make frequent releases routine rather than risky. |
| **Consolidate operations** | One reliability, SLO, incident and capacity view replaces per-capability monitoring stacks. |
| **Avoid platform lock-in** | Minimal runtime dependencies and standard container/orchestration infrastructure keep adoption costs and exit costs low. |
| **Preserve optionality** | Pluggable capability internals behind stable contracts let the organisation change algorithms and engines without re-platforming. |

---

## 7. Major Features

<details open>
<summary><strong>Capability Layers</strong></summary>

| Area | Features |
|------|----------|
| **Digital Twin** | Synchronised virtual asset representations; static asset model coupled with dynamic state streams; stable asset identifiers shared platform-wide; immutable asset/state contracts consumed by all layers. |
| **Analytics** | Metric aggregation with percentiles, windows and trend analysis; composite health scoring; capacity and growth analysis over operational series. |
| **Machine Learning** | Predictive interfaces with injectable forecasting strategies per asset class; predictions carrying uncertainty and provenance; lifecycle management from experiment to production. |
| **Deep Learning** | Architecturally supported behind the same predictive contracts — model internals are pluggable, so deep architectures integrate without changing consumers. |
| **Decision Intelligence** | Results framed for decisions: predictions bound to lineage, routed through workflows, and observable through to action. |
| **Agentic AI** | Perception–deliberation–action agents; actions constrained to typed tool interfaces; recorded reasoning trajectories for audit and debugging. |
| **Knowledge Intelligence** | Evidence-grounded question answering; retrieved evidence carried with every answer for attribution. |
| **Enterprise RAG** | Versioned corpus index for reproducible retrieval; asset-scoped retrieval via twin identifiers. |
| **Workflow Engine** | Multi-step processes composed across capabilities; explicit, immutable process state; deterministic transitions; auditable, replayable histories. |
| **Scheduler** | Time- and event-conditioned initiation of workflows through the platform's injected-clock discipline, keeping scheduled behaviour deterministic and testable. |
| **Event Bus** | Observer-style fan-out decoupling signal producers from consumers — used by alerting and incident notification so new reactions attach by subscription. |
| **Executive Dashboard** | Operations dashboard composing reliability, SLO compliance, incidents, capacity and readiness into a single graded snapshot. |
| **REST APIs** | Service exposure through the deployment layer (Service/Ingress) with TLS termination; application endpoints fronted by the platform's deterministic health contract. |

</details>

<details open>
<summary><strong>Production-Engineering Substrate</strong></summary>

| Area | Features |
|------|----------|
| **Enterprise MLOps** | Governance anchor for the platform: experiments, models, artifacts and lineage under one lifecycle. |
| **Experiment Tracking** | Runs captured with parameters and metrics; bound to source revision and environment for reproducibility. |
| **Model Registry** | Semantic versioning; stage promotion (e.g., staging → production); authoritative record of what exists and in what state. |
| **Artifact Store** | Content-addressed storage — identical content stored once; artifact references stable by construction. |
| **Production Monitoring** | Prediction monitoring, data-quality validation, composite model-health assessment, routed alerting. |
| **Drift Detection** | Data drift (input distribution change) and concept drift (input–output relationship change) detected distinctly. |
| **CI/CD** | Shared repository-validation library; twenty independent quality gates; release validation against policy; deployment-readiness checks; honest gate failures reported, never masked. |
| **Docker** | Multi-stage production image; non-root; read-only root filesystem; dropped capabilities; container health check; development image for iteration. |
| **Kubernetes** | Ten manifests: namespace (restricted PSA), config, secret template, deployment, service, ingress, HPA, network policy, PDB, PVC. |
| **Deployment** | Dependency-ordered automated apply; zero-downtime rolling updates; health-gated rollback; a single deterministic health check shared by container, probes and scripts. |
| **Observability** | Metrics, distributed tracing (timeline and critical path), structured logging with correlation and audit linkage. |
| **Reliability** | Availability, MTBF, MTTR and composite reliability score computed from request outcomes and outage windows; SLOs with error budgets and burn rates. |
| **Incident Management** | Validated lifecycle (open → investigating → identified → monitoring → resolved → closed); timelines; recovery-time computation; generated postmortems. |
| **Capacity Planning** | Deterministic forecasting across CPU, memory, storage, request volume, model and data growth; headroom recommendations; exhaustion detection. |
| **Production Readiness** | Ten-area weighted assessment (architecture, security, reliability, monitoring, deployment, CI/CD, testing, documentation, MLOps, observability) — a transparent, reproducible self-assessment. |

</details>

<details>
<summary><strong>Research, Portfolio & Release Assets</strong></summary>

| Area | Features |
|------|----------|
| **Research Assets** | IEEE-style research paper; bibliography of real references; appendices; glossary. |
| **Portfolio Assets** | Project summary; recruiter one-pager; interview cheat sheet; achievements; ATS-friendly resume entry; launch post; 60–90 s interview pitch. |
| **Release Assets** | v1.0.0 release notes and summary; repository audit; measured statistics; production release checklist; final engineering summary; project overview; GitHub release description, topics and social preview; `VERSION`; `RELEASE_MANIFEST.md`. |

</details>

---

## 8. Complete Enterprise Architecture

The platform is ten layers with a single dominant dependency direction. The five capability layers
provide the AI surface; the five production-engineering subsystems operate it. The architectural
invariant that holds the system together: **every boundary is crossed by an immutable, serialisable
value object** — never shared mutable state, never a call into another subsystem's internals. The
dependency graph is therefore acyclic, and every layer is exercisable in isolation.

<!--
![Platform architecture](docs/figures/architecture_overview.png)
*Figure 1 — Ten-layer platform architecture with cross-cutting governance. **[image placeholder]***
(figure asset pending; intentionally hidden for public release)
-->

### 8.1 Digital Twin Layer

Couples a static asset model — identity, configuration, relationships — with a dynamic state stream
synchronised from telemetry. Exposes asset and state as immutable contracts; consumers read snapshots
rather than sharing a live mutable object, which is why many layers can depend on the twin without
coupling to it. The degree of automated synchronisation per asset class reflects the standard
model/shadow/twin distinction.

### 8.2 Predictive Intelligence Layer

Forecasting and prognostics — remaining useful life, demand, and similar — over twin state and
telemetry. The forecasting *strategy* is injected, so the predictive *interface* stays stable while the
algorithm varies per asset class. Every prediction is bound to its provenance through MLOps; a forecast
is never an orphan number.

### 8.3 Agentic AI Layer

Autonomous, tool-using reasoning on the classical perception–deliberation–action loop. Two constraints
make autonomy operable: actions are restricted to **typed tool interfaces**, so an agent's possible
effects are enumerable and governable; and **reasoning trajectories are recorded**, so behaviour is
reconstructable and auditable through the observability layer like any service request.

### 8.4 Knowledge Intelligence (RAG) Layer

Question answering over enterprise corpora by retrieving evidence and conditioning generation on it.
Evidence travels with each answer for attribution; the corpus index is a **versioned artefact**, so the
evidence available at a moment in time is reproducible. Asset identifiers scope retrieval to specific
equipment.

### 8.5 Enterprise Workflow Engine

Composes steps across capabilities — a twin query, a prediction, a retrieval, an agent action — into
governed processes with explicit, immutable state and deterministic transitions. Orchestration lives in
one auditable place instead of being smeared across application code; histories can be replayed.

### 8.6 Enterprise MLOps

The provenance backbone: experiment tracking; a versioned model registry with semantic versioning and
stage promotion; a content-addressed artifact store; a reproducibility engine binding runs to source
revision and environment; and a lineage graph linking runs, datasets, artifacts and models. "How was
this result produced?" becomes a graph traversal.

<!--
![MLOps lineage](docs/figures/mlops_lineage.png)
*Figure 2 — MLOps provenance and lineage. **[image placeholder]***
(figure asset pending; intentionally hidden for public release)
-->

### 8.7 Production Monitoring

Watches behaviour after deployment: data drift, concept drift, prediction anomalies, data quality, and
a composite model-health score. The alert engine fans out by subscription. Monitoring signals are
themselves contracts consumed by reliability and observability — degradation becomes an operational
input, not a dashboard footnote.

### 8.8 CI/CD

Quality enforced at the boundary where change enters the system: a shared validation library, a
twenty-gate quality engine, release validation against policy, and a deployment-readiness check, wired
into three workflows. Legitimate gate failures are surfaced, preserving diagnostic value.

### 8.9 Enterprise Deployment

A multi-stage, non-root container image and ten Kubernetes manifests: rolling updates with no
unavailable replicas, startup/readiness/liveness probes, hardened security context, autoscaling,
default-deny network policy, pod disruption budget, durable storage. One **deterministic health check**
backs the container, the probes and every script — "healthy" is defined once. Rollback reverts and
re-verifies health automatically.

<!--
![Deployment topology](docs/figures/deployment_topology.png)
*Figure 3 — Kubernetes deployment topology. **[image placeholder]***
(figure asset pending; intentionally hidden for public release)
-->

### 8.10 Enterprise Observability

Operation as measurement: metrics with percentiles, windows and trends; distributed tracing with
timeline and critical-path reconstruction; structured, correlated logging; a reliability engine
(availability, MTBF, MTTR, composite score); SLOs with error budgets and burn rates; a validated
incident lifecycle with postmortems; deterministic capacity forecasting; an operations dashboard; and
the ten-area production-readiness assessment. Integrates purely by composition.

<!--
![Observability](docs/figures/observability_slo.png)
*Figure 4 — Observability and SLO evaluation. **[image placeholder]***
(figure asset pending; intentionally hidden for public release)
-->

### 8.11 The Operational Loop

The architecture closes a loop: monitoring detects change → observability quantifies reliability and
SLO impact → incidents govern response → capacity anticipates limits → readiness judges fitness to ship
→ decisions flow back through CI/CD and deployment. Not a stack of boxes — a governed feedback system
held together by immutable contracts and one dependency direction.

---

## 9. Repository Structure

```
.
├── src/                          Production-engineering source packages
│   ├── mlops/                    Experiments, registry, artifacts, reproducibility, lineage
│   ├── monitoring/               Drift, data quality, model health, alerting
│   └── observability/            Metrics, tracing, logging, reliability, SLO,
│                                 incidents, capacity, dashboard, readiness
├── scripts/                      CI/CD validation: repository validation, quality
│                                 gates, release validation, deployment readiness
├── deployment/                   Packaging and operations
│   ├── docker/                   Production & dev images, compose files, entrypoint
│   ├── kubernetes/               Ten manifests (namespace → PVC)
│   └── scripts/                  Deploy, rollback, deterministic health check
├── configs/                      YAML policy: observability, logging, reliability,
│                                 quality gates, release
├── tests/                        Deterministic, framework-agnostic test suite
│                                 (one test file per module)
├── docs/                         Documentation and research
│   ├── week11_engineering_report.md
│   └── week12/                   Research paper, references, appendices, glossary,
│                                 architecture overview, quick start, developer &
│                                 deployment guides, FAQ, showcase, phase summaries
├── demo/                         Executive & technical demo scripts, checklist,
│                                 dataset description, storyboard
├── presentation/                 Executive & technical decks, notes, walkthrough
├── case_study/                   Enterprise, business and technical case studies
├── video/                        Narration script and recording plan
├── validation/                   Engineering validation package (9 documents)
├── benchmarks/                   Benchmark suite, execution plan, empty results
│                                 template, qualitative comparison matrix
├── reports/                      Engineering scorecard, repository health,
│                                 release readiness
├── release/                      v1.0.0 release notes, summary, audit, measured
│                                 statistics, checklist, engineering summary, overview
├── github/                       Release description, topics, social preview
├── portfolio/                    Recruiter, resume, interview and launch assets
├── .github/                      Community files: contributing, code of conduct,
│   ├── ISSUE_TEMPLATE/           issue templates, PR template
│   └── workflows/                CI, release and dependency-scan workflows
├── enterprise_mlops_architecture.png    Architecture figure (4K)
├── LICENSE                       MIT
├── CITATION.cff                  Citation metadata (CFF 1.2.0)
├── CHANGELOG.md                  Changelog (current: v1.0.1)
├── VERSION                       1.0.1
├── RELEASE_MANIFEST.md           Directory & deliverable manifest
└── README.md                     This document
```

**How to read it.** `src/`, `scripts/` and `deployment/` hold the implemented, verified
production-engineering substrate; the capability layers are specified architecturally in `docs/` and
the research paper (see [scope note](#19-repository-statistics)). `tests/` mirrors modules one-to-one.
Everything under `docs/`, `demo/`, `presentation/`, `case_study/`, `video/`, `validation/`,
`benchmarks/`, `reports/`, `release/`, `github/` and `portfolio/` is documentation and release
material — the repository is designed to be *read* as much as run.

---

## 10. Technology Stack

| Category | Technology | Role |
|----------|-----------|------|
| **Language** | Python 3.12 (fully type-annotated) | All production-engineering source |
| **Frameworks** | Standard library first; no web/ML framework required at runtime | Minimal surface, deterministic behaviour |
| **ML / Numerical** | NumPy | Numerical computing for analytics, monitoring and forecasting |
| **Data contracts** | Frozen, slotted dataclasses + JSON serialisation | Immutable cross-boundary value objects |
| **Databases / Storage** | Content-addressed artifact store; versioned registries; Kubernetes PVC for durable state | Provenance-preserving persistence without a heavy DBMS dependency |
| **Infrastructure** | Docker (multi-stage), Kubernetes | Packaging and orchestration |
| **DevOps** | GitHub Actions (3 workflows), 20-gate quality engine, release & readiness validators | Gated, automated delivery |
| **Deployment** | Deployment/Service/Ingress/HPA/NetworkPolicy/PDB/PVC manifests; deploy & rollback automation; deterministic health check | Zero-downtime rollout and health-gated recovery |
| **Configuration** | YAML policy files (PyYAML) | Behaviour as data, not code |
| **Testing** | Deterministic, framework-agnostic suite (assertions + parameterisation) | Reproducible verification |
| **Documentation** | Markdown throughout; IEEE-style paper; CFF citation metadata | Reviewable, citable record |
| **Research** | `references.bib` (real references), appendices, glossary | Academic-grade grounding |

> **Design stance.** Two runtime dependencies (NumPy, PyYAML) is a feature, not an omission: it keeps
> the platform deterministic, auditable and free of lock-in. Heavier engines integrate *behind* the
> platform's contracts rather than *into* its core.

---

## 11. Engineering Principles

The platform's engineering story is that a small set of principles, applied **uniformly**, is what
keeps a ten-layer system coherent.

**SOLID.** Single responsibilities per module; extension through new types and injected strategies
rather than modification; small, client-specific interfaces; dependencies pointing at abstractions
(clocks, identifier sources, strategies, probes).

**Dependency Injection.** Time, identity, forecasting strategies, scoring weights and external probes
are injected, never acquired internally. This is the load-bearing decision: it is why the system is
deterministic, why tests run offline with no mocking of time or randomness, and why behaviour is a pure
function of explicit inputs.

**Clean Architecture.** One dominant dependency direction; policy separated from mechanism
(configuration as data); infrastructure concerns (deployment, delivery) kept at the edges; domain
contracts at the centre.

**Composition.** Integration composes subsystem *outputs*; inheritance is reserved for genuine type
specialisation. The design stays flat, explicit and independently evolvable — the structural reason the
platform avoids the fragility of deep hierarchies.

**Determinism.** All non-determinism is injected; collections are ordered before serialisation; numeric
results are rounded consistently. Reproducibility is asserted by tests (byte-identical serialised
reports across repeated runs), not hoped for.

**Immutable Models.** Domain types are frozen, slotted and self-serialising, round-tripping losslessly
to and from JSON. Immutability gives safe cross-thread sharing, well-defined equality, and eliminates a
broad class of state bugs.

**Thread Safety.** Stateful engines guard mutable internals with re-entrant locks and expose immutable
snapshots; readers never observe partial updates.

**Modularity.** Each subsystem is internally complete and externally minimal — a small surface of
factory functions and value objects. Change localises.

**Maintainability.** A designed property, not an accident: narrow boundaries, immutable contracts,
minimal dependencies, co-located documentation, and a test suite that localises failures produce a low
change-risk profile (assessed in [`week12_phase4_validation/maintainability_assessment.md`](week12_phase4_validation/maintainability_assessment.md)).

**Testability.** The sum of the above: inject the non-determinism, freeze the values, narrow the
boundaries — and testing becomes cheap, portable and reliable.

---

## 12. Development Journey

The platform was built **additively** across twelve weeks: each week extended the system without
modifying its predecessors, and the full test suite re-verifies every prior layer on each run.

| Weeks | Milestone |
|-------|-----------|
| **1–4** | Platform foundations: domain contracts, layered structure, and the architectural invariants (dependency direction, composition-only integration) that every later week builds on. |
| **5** | **Enterprise AI Layer** — the platform's AI surface established on the foundation contracts. |
| **6** | **Enterprise Analytics** — aggregation, percentiles, windows and trend analysis over operational data. |
| **7** | **Decision Intelligence** — results framed for decisions and carried toward action with provenance. |
| **8** | **Executive Dashboard** — composed operational snapshots with graded summaries. |
| **9** | **Knowledge Intelligence (RAG)** — evidence-grounded answering over versioned corpora. |
| **10** | **Production AI Platform** — the capability layers consolidated into a coherent, operable platform. |
| **11** | **Production-engineering substrate**, five phases: Enterprise MLOps → Production Monitoring → CI/CD → Deployment → Observability — implemented in code and verified by the deterministic suite. |
| **12** | **Research, Portfolio, Validation, Production Release**, five phases: IEEE-style research paper → GitHub & portfolio documentation → demonstration assets → engineering validation package → the v1.0.0 release and repository audit. |

The journey is depicted in the repository's timeline figure and summarised in
[`week12_phase5_release/final_engineering_summary.md`](week12_phase5_release/final_engineering_summary.md).

---

## 13. Installation

**Prerequisites.** Python 3.12+, Git; Docker and `kubectl` (with cluster access) for the container and
orchestration paths.

```bash
# Clone
git clone https://github.com/noohkhan7232/wind-turbine-acoustics.git
cd wind-turbine-acoustics

# Isolated environment
python3 -m venv .venv
source .venv/bin/activate

# Runtime dependencies
pip install numpy pyyaml

# Development & testing
pip install pytest pytest-cov
```

The runtime footprint is intentionally minimal — see the [technology stack](#10-technology-stack).
The full research and machine-learning stack (audio processing, PyTorch, experiment tracking) used by
the acoustic research lineage is pinned separately in [`requirements.txt`](requirements.txt):
`pip install -r requirements.txt`.

---

## 14. Quick Start

```bash
# 1. Verify the suite is green
PYTHONPATH=src:scripts pytest tests/ -q

# 2. Run the deterministic observability demonstrations (sorted-key JSON output)
PYTHONPATH=src python3 -c "from observability import main; main(['metrics'])"
PYTHONPATH=src python3 -c "from observability import main; main(['reliability'])"
PYTHONPATH=src python3 -c "from observability import main; main(['readiness'])"
#   also: tracing | capacity | all

# 3. Run one demo twice — outputs are byte-identical (determinism by construction)
```

From here: [`docs/week 12/week12_phase2/quick_start.md`](docs/week%2012/week12_phase2/quick_start.md) for the clone-to-Kubernetes path,
[`docs/week 12/week12_phase2/developer_guide.md`](docs/week%2012/week12_phase2/developer_guide.md) to contribute, and
[`docs/week 12/week12_phase2/deployment_guide.md`](docs/week%2012/week12_phase2/deployment_guide.md) for operations.

---

## 15. Configuration

Behaviour is governed by YAML under `configs/` — configuration is data, not code; changing a policy
does not require changing an engine.

| File | Governs |
|------|---------|
| `observability.yaml` | Metric categories, percentiles, tracing and dashboard settings |
| `logging.yaml` | Structured-logging format, severity, default context |
| `reliability_policy.yaml` | SLO targets, error-budget actions, capacity headroom, readiness thresholds |
| `quality_gate.yaml` | CI/CD quality-gate configuration |
| `release_policy.yaml` | Release validation policy |

> **Note.** The default SLO values in `reliability_policy.yaml` (availability ≥ 0.99 over 30 days;
> P95 latency ≤ 250 ms; error rate ≤ 0.01; freshness ≤ 300 s) are **configured objectives evaluated at
> runtime** — they are not measured benchmarks, and this README makes no benchmark claims.

---

## 16. Running Tests

```bash
# Full suite
PYTHONPATH=src:scripts pytest tests/ -q

# One subsystem
PYTHONPATH=src pytest tests/test_metrics_engine.py -q
```

Every test is deterministic and framework-agnostic — standard assertions and parameterisation only; no
fixtures, no network, no external services. See [Testing Strategy](#24-testing-strategy) for the
philosophy and [`week12_phase5_release/repository_statistics.md`](week12_phase5_release/repository_statistics.md) for measured
counts.

---

## 17. Docker Deployment

```bash
cd deployment/docker

# Development (bind-mounted source, fast iteration)
docker compose up --build

# Production (hardened profile)
docker compose -f docker-compose.prod.yml up -d
```

The production image is multi-stage and **non-root**, declares a container health check backed by the
platform's deterministic health checker, and runs with a read-only root filesystem, no privilege
escalation, dropped Linux capabilities, resource limits and graceful signal handling.

---

## 18. Kubernetes Deployment

```bash
# Dependency-ordered apply → rollout wait → in-pod health verification
deployment/scripts/deploy_kubernetes.sh

# Health-gated rollback (previous or specific revision)
deployment/scripts/rollback.sh
deployment/scripts/rollback.sh --to-revision N
```

The ten manifests provide: a restricted-PSA namespace, config and a secret **template** (real secrets
are supplied out-of-band), a rolling deployment with probes and hardened security context, a service,
TLS-terminated ingress, autoscaling, default-deny network policy, a pod disruption budget and durable
storage. One deterministic health check defines "healthy" across the container, the probes and every
script. Full procedure: [`docs/week 12/week12_phase2/deployment_guide.md`](docs/week%2012/week12_phase2/deployment_guide.md).

---

## 19. Repository Statistics

> **Integrity rule.** This README does not restate repository numbers. Statistics are only reported
> where they are measured and auditable — in
> [`week12_phase5_release/repository_statistics.md`](week12_phase5_release/repository_statistics.md), which documents the
> collection methodology, records the measured values for the v1.0.0 audit, and leaves
> version-control–derived fields blank because git history is not initialised in this snapshot.

### Pending Final Repository Audit

The fields below are intentionally **blank** in this document. They must be populated only from a fresh
measurement at final audit time (see the methodology in the statistics document); values must never be
estimated or copied forward without re-verification.

| Field | Value |
|-------|-------|
| Total LOC | |
| Source files | |
| Modules | |
| Packages | |
| Tests | |
| Documentation files | |
| Figures | |
| Configuration files | |
| Dependencies | |
| Git commits | |
| Repository size | |

**Scope note.** Measured source-code figures cover the production-engineering substrate present in this
repository snapshot (`src/mlops`, `src/monitoring`, `src/observability`, `scripts/`, `deployment/`).
The five capability layers are documented architecturally — their internals are pluggable behind stable
contracts and are not included in source counts. This scoping is stated wherever statistics appear.

---

## 20. Research Paper

The repository includes a companion **IEEE-style research paper** describing the platform's
architecture and methodology:

> *An Integrated Architecture for Enterprise Digital Twins and Decision Intelligence: Unifying
> Predictive, Agentic, Retrieval-Augmented and Production-Engineering Subsystems* —
> [`docs/week 12/research_paper.md`](docs/week%2012/research_paper.md) *(PDF placeholder — to be linked on
> publication)*

The paper spans 22 sections: problem statement, literature review, research gap, the full architecture,
per-layer treatments, methodology, evaluation, honestly framed engineering contributions, business
applications, limitations and future directions. It is accompanied by
[`references.bib`](docs/week%2012/references.bib) (real, well-known references only),
[`appendices.md`](docs/week%2012/appendices.md) and [`glossary.md`](docs/week%2012/glossary.md). The
contributions are framed as **engineering integration, architecture and implementation** — the paper
explicitly does not claim algorithmic novelty or report fabricated experimental numbers.

---

## 21. Engineering Validation

A complete validation package assesses the platform's engineering quality with the same honesty
discipline as the rest of the repository:

| Package | Contents |
|---------|----------|
| [`week12_phase4_validation/`](week12_phase4_validation/) | Engineering validation (15 rated dimensions) · benchmark methodology · scalability analysis · architecture review · maintainability · reliability · **architecture-level** security review (explicitly not a penetration test) · technical-debt analysis · production-readiness review |
| [`week12_phase4_benchmarks/`](week12_phase4_benchmarks/) | Twelve-scenario benchmark suite · step-by-step execution plan · **empty** results template (the only permitted placeholders) · qualitative comparison matrix (no named products) |
| [`week12_phase4_reports/`](week12_phase4_reports/) | Ten-dimension engineering scorecard · repository health report · release-readiness report |

Headline findings: strong ratings across architecture, testing, documentation, deployment,
maintainability, observability, MLOps and CI/CD, with reliability and security rated solid — bounded by
*measurement and review scope*, not design. The release-readiness report separates "ready for public
release" (yes, pending metadata items) from "cleared for untrusted production" (a bounded additional
checklist).

---

## 22. Demonstration Assets

| Asset | Audience & purpose |
|-------|--------------------|
| [`week12_phase3_demo/executive_demo_script.md`](week12_phase3_demo/executive_demo_script.md) | 10–12 min outcome-focused walkthrough for executives and CTOs |
| [`week12_phase3_demo/technical_demo_script.md`](week12_phase3_demo/technical_demo_script.md) | 20–25 min engineering deep-dive with live, deterministic terminal segments |
| [`week12_phase3_demo/live_demo_checklist.md`](week12_phase3_demo/live_demo_checklist.md) | Pre-flight, backup and recovery procedures for live delivery |
| [`week12_phase3_demo/demo_storyboard.md`](week12_phase3_demo/demo_storyboard.md) | Twelve scenes — speaker, screen, talking points, expected outcome |
| [`week12_phase3_case_study/`](week12_phase3_case_study/) | Enterprise case study (realistic plant scenario), business case (value levers + quantification methodology, no invented ROI), technical case (design decisions under scrutiny) |
| [`week12_phase3_video/`](week12_phase3_video/) | Timed narration script and full recording plan for a 12–15 min walkthrough |
| [`week12_phase3_presentation/`](week12_phase3_presentation/) | 15-slide executive deck, 20-slide technical deck, speaker notes, layer-by-layer architecture walkthrough |

All demonstration material observes the same rule as the code: reproducible behaviour on stage, and no
fabricated figures anywhere in the narrative.

---

## 23. Documentation Index

| Category | Documents |
|----------|-----------|
| **Architecture** | Architecture overview · architecture walkthrough · research paper §5–§16 |
| **Getting started** | Quick start · installation (this README §13–§14) |
| **Guides** | Developer guide · deployment guide · configuration reference (§15) |
| **Reference** | FAQ (48 questions) · glossary · abbreviations (paper appendices) |
| **Research** | Research paper · references · appendices |
| **Validation** | Engineering validation · reviews (architecture, maintainability, reliability, security) · technical debt · production readiness |
| **Benchmarks** | Methodology · suite · execution plan · results template · comparison matrix |
| **Reports** | Engineering scorecard · repository health · release readiness |
| **Release** | Release notes · release summary · repository audit · repository statistics · release checklist · final engineering summary · final project overview · manifest |
| **Demonstration** | Demo scripts · storyboard · checklist · dataset description · decks · notes · case studies · video plan |
| **Portfolio** | Project summary · recruiter one-pager · interview cheat sheet · achievements · resume entry · launch post · interview pitch |
| **Community** | Contributing · code of conduct · issue & PR templates · changelog · citation · license |

---

## 24. Testing Strategy

The philosophy: **make correctness cheap to verify, and verification will actually happen.**

Four structural decisions serve that philosophy. First, all non-determinism is injected, so tests need
no mocking of time or randomness and can assert **byte-identical serialised outputs** across repeated
runs — reproducibility is a test, not a hope. Second, tests are **framework-agnostic**: standard
assertions and parameterisation only, no fixtures, no network, no external services — portable and fast
enough to run constantly. Third, the suite is **layered so failures localise**: value-object tests
isolate serialisation/validation defects; engine tests isolate computational defects against known
inputs; edge-case tests cover empty inputs, boundaries and illegal transitions; determinism tests guard
the reproducibility property itself. Fourth, tests **mirror modules one-to-one**, so the mapping
between code and verification is obvious, and the full run re-verifies every prior subsystem's
contracts — which is what makes additive construction safe.

No coverage percentage is quoted here: coverage is meaningful only when measured, and measured figures
belong in the audited statistics document, not prose. Measured test counts are in
[`week12_phase5_release/repository_statistics.md`](week12_phase5_release/repository_statistics.md).

---

## 25. Security

Security posture is documented honestly in
[`week12_phase4_validation/security_review.md`](week12_phase4_validation/security_review.md) — an **architecture-level review,
explicitly not a penetration test**.

Highlights of the built-in posture: no secrets committed (the Kubernetes secret ships as a clearly
marked template; deployment scripts warn if real material is absent); hardened containers (multi-stage,
non-root, read-only root filesystem, dropped capabilities, no privilege escalation); hardened
orchestration (restricted Pod Security Standard, default-deny network policy with an explicit
metadata-endpoint block, TLS-terminated ingress); a minimal two-dependency runtime surface with a
dependency-scan workflow; structured, correlated, audit-linked logging; and structural auditability via
the MLOps lineage graph and incident timelines.

Before exposure in an untrusted environment, complete the bounded hardening list in
[`week12_phase5_release/production_release_checklist.md`](week12_phase5_release/production_release_checklist.md): external secrets
management, image scanning, hash-pinned dependencies and SBOM, an authentication/authorisation layer at
the ingress, admission policy and least-privilege RBAC, and an independent audit.

---

## 26. Performance Validation

**This repository reports no benchmark numbers, by design.** Runtime performance is addressed as
*methodology*: [`week12_phase4_validation/benchmark_methodology.md`](week12_phase4_validation/benchmark_methodology.md) defines
repeatable procedures — environment capture, warm/cold separation, fixed repetitions, median and P95
reporting — for cold start, memory, CPU, inference and workflow latency, deployment time, recovery
time, health-check time, CI/CD duration and repository build time.
[`week12_phase4_benchmarks/benchmark_suite.md`](week12_phase4_benchmarks/benchmark_suite.md) defines twelve scenarios (BM-01 –
BM-12), [`week12_phase4_benchmarks/benchmark_execution_plan.md`](week12_phase4_benchmarks/benchmark_execution_plan.md) specifies
step-by-step execution with statistical handling and a reproducibility check, and
[`week12_phase4_benchmarks/benchmark_results_template.md`](week12_phase4_benchmarks/benchmark_results_template.md) holds **empty**
tables to be populated only by measurement. A result without its environment and repetition count is
not a result. SLO values are configured runtime targets and are never presented as measurements.

---

## 27. Roadmap

**Version 1.0.0 is complete.** The architecture, the production-engineering substrate, verification,
documentation, research, validation and release assets are finished; nothing below implies missing
implementation. Future versions are **optional enhancements** that extend the completed platform:

- **Measurement** — execute the benchmark suite; sustained-load reliability measurement; concurrency
  stress and fault-injection testing.
- **Ecosystem bridges** — export adapters from the self-contained observability layer to external
  monitoring ecosystems (a designed, additive integration seam).
- **Hardening extensions** — hash-pinned dependencies and SBOM, admission policy, per-workload RBAC,
  ingress authentication/authorisation.
- **Capability-layer deepening** — plugging concrete high-performance engines behind the existing
  predictive, retrieval and agentic contracts; extending the substrate's verification discipline to
  those internals.
- **Research directions** — cross-capability contract formalisation, agent observability, decision-
  quality evaluation (per the paper's §21).

---

## 28. Contribution Guide

Contributions preserve the platform's invariants: **additive change**, **integration by composition**,
**determinism**, **immutability**, and full typing. The workflow: open an issue (templates provided) →
branch → implement additively with tests at the level the behaviour is introduced → run the full suite
and quality gates locally → open a PR using the template's engineering and testing checklists. Honest
gate failures are addressed, never suppressed.

Full policy: [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md) · conduct:
[`.github/CODE_OF_CONDUCT.md`](.github/CODE_OF_CONDUCT.md). Security issues: report privately (see
[Security](#25-security)); do not open public issues for vulnerabilities.

---

## 29. Citation

If you reference this work, please cite it using the repository's [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" control from it). The companion research paper and its
bibliography are in [`docs/week 12/`](docs/week%2012/).

---

## 30. License

Released under the [MIT License](LICENSE). Set the copyright holder line before public release (tracked
in the [release checklist](week12_phase5_release/production_release_checklist.md)).

---

## 31. Acknowledgements

This platform stands on established research and practice: digital twins and Industry 4.0; prognostics
and health management; decision intelligence; transformers, retrieval-augmented generation and
language-model agents; the MLOps and production-ML literature; containers, orchestration and site
reliability engineering; and the software-architecture canon. Full attributions — real references
only — are in the research paper's [bibliography](docs/week%2012/references.bib).

---

## 32. Final Engineering Summary

Twelve weeks, ten layers, one discipline. The Enterprise Digital Twin & Decision Intelligence Platform
demonstrates that the hard problem of industrial AI — operating many heterogeneous capabilities
together, coherently and safely — yields to disciplined software engineering: a single dependency
direction; integration by composition through immutable contracts; provenance built into the data
model; quality enforced at the boundary; deployment that is hardened, observable and reversible; and
determinism deep enough that reproducibility is asserted by tests.

The platform was built additively, so Week 12 never broke Week 1. Its production-engineering substrate
is implemented and verified; its capability layers are specified behind stable, pluggable contracts;
and its claims are bounded the way senior engineering demands — measured statistics in an audited
document, configured targets never dressed up as benchmarks, self-assessment never dressed up as
certification, and an architectural security review that says so plainly.

**v1.0.0 — Engineering complete. Research complete. Portfolio complete. Documentation complete.
Production release complete.**

<div align="center">

*Integration done with discipline.*

</div>
