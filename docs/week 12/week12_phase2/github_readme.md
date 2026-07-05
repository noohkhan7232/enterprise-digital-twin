<!-- Badges are placeholders; replace the badge URLs and links on public release. -->
![build](https://img.shields.io/badge/build-placeholder-lightgrey)
![coverage](https://img.shields.io/badge/coverage-placeholder-lightgrey)
![tests](https://img.shields.io/badge/tests-1503%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-v1.0-blue)

# Enterprise Digital Twin & Decision Intelligence Platform

> An integrated, production-engineered platform that unifies digital twins, predictive
> intelligence, agentic AI, retrieval-augmented knowledge and workflow orchestration behind a
> single governance, deployment and observability fabric.

---

## Project Overview

The Enterprise Digital Twin & Decision Intelligence Platform is a layered system that hosts
many industrial-AI capabilities as one coherent, governable whole. Rather than treating digital
twins, prognostics, retrieval-augmented reasoning and autonomous agents as isolated point
solutions, the platform integrates them behind shared, immutable data contracts and a uniform
production-engineering substrate covering machine-learning operations, monitoring, continuous
delivery, deployment and observability.

The repository is the result of a multi-week engineering effort built additively: each layer
extends the system without modifying its predecessors. The production-engineering subsystems
are implemented in pure Python with NumPy as the only numerical dependency and are verified by
**1,503 deterministic automated tests**.

## Project Vision

Industrial AI most often fails for systemic rather than algorithmic reasons: integration debt,
lost provenance, fragmented monitoring and inconsistent governance. The platform's vision is to
make coherence at scale the default — to provide many AI capabilities to an enterprise while
preserving a single source of provenance, one deployment-and-recovery story, and one
observability fabric, without sacrificing the independence that lets each capability evolve.

## Key Features

- **Ten-layer integrated architecture** spanning digital twin, predictive intelligence, agentic
  AI, knowledge intelligence (RAG), workflow orchestration, MLOps, monitoring, CI/CD, deployment
  and observability.
- **Provenance by construction** — every prediction is traceable to its model version,
  experiment, dataset, code revision and environment through the MLOps lineage graph.
- **Production monitoring** for data drift, concept drift, prediction anomalies, data quality and
  composite model health, with a routed alert engine.
- **Enforced quality** through twenty CI/CD quality gates, release validation and
  deployment-readiness checks.
- **Zero-downtime deployment** with multi-stage containers, Kubernetes orchestration, autoscaling,
  network policy, pod disruption budget and health-gated rollback.
- **Full observability** — metrics, distributed tracing, structured logging, reliability
  engineering, SLI/SLO with error budgets, incident management and capacity planning.
- **Deterministic and reproducible** behaviour throughout, via injected clocks and identifiers.

## Architecture Overview

The platform is organised into ten layers with a single dominant direction of dependency. Each
layer consumes the outputs of others as immutable, serialisable value objects rather than
reaching into their internals, which keeps the dependency graph acyclic and every layer
independently testable. See [`architecture_overview.md`](architecture_overview.md) for the full
treatment.

<!-- ![Figure 1 — Platform architecture](docs/figures/architecture_overview.png) (figure asset pending) -->
*Figure 1. Enterprise platform architecture. **[image placeholder]***

## Technology Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.12 (typed, standard library) |
| Numerical computing | NumPy |
| Data contracts | Immutable, slotted dataclasses with JSON serialisation |
| Configuration | YAML policy and configuration files |
| Containerisation | Multi-stage Docker images |
| Orchestration | Kubernetes (Deployment, Service, Ingress, HPA, NetworkPolicy, PDB, PVC) |
| Continuous integration | GitHub Actions workflows |
| Testing | Deterministic, framework-agnostic test suite |

## Repository Structure

```
.
├── src/
│   ├── digital_twin/        # Layer 1 — asset models and synchronised state
│   ├── predictive/          # Layer 2 — forecasting and prognostics
│   ├── agentic/             # Layer 3 — tool-using autonomous reasoning
│   ├── knowledge/           # Layer 4 — retrieval-augmented knowledge (RAG)
│   ├── workflow/            # Layer 5 — process orchestration
│   ├── mlops/               # Layer 6 — experiments, registry, artifacts, lineage
│   ├── monitoring/          # Layer 7 — drift, quality, health, alerts
│   └── observability/       # Layer 10 — metrics, tracing, logging, SLO, incidents
├── scripts/                 # Layer 8 — CI/CD validation (quality gates, release, readiness)
├── deployment/              # Layer 9 — Docker, Kubernetes, automation, health check
│   ├── docker/
│   ├── kubernetes/
│   └── scripts/
├── configs/                 # YAML policy and configuration
├── tests/                   # Deterministic automated test suite
├── docs/                    # Documentation, research paper, guides
│   └── week12/
├── .github/                 # Issue/PR templates, contribution policy, workflows
├── LICENSE
├── CITATION.cff
└── CHANGELOG.md
```

> The five capability layers are described at the architectural level; the five
> production-engineering subsystems (`mlops`, `monitoring`, `observability`, `scripts`,
> `deployment`) are implemented and verified in this repository.

## Installation

```bash
# Clone
git clone https://github.com/noohkhan7232/wind-turbine-acoustics.git
cd wind-turbine-acoustics

# Create an isolated environment
python3 -m venv .venv && source .venv/bin/activate

# Install runtime dependencies (numerical computing + configuration parsing)
pip install numpy pyyaml

# For development and testing
pip install pytest pytest-cov
```

## Quick Start

```bash
# Run the observability CLI demonstrations (deterministic JSON output)
PYTHONPATH=src python3 -c "from observability import main; main(['metrics'])"
#   demos: metrics | tracing | reliability | capacity | readiness | all
```

See [`quick_start.md`](quick_start.md) for clone-to-Kubernetes instructions.

## Configuration

Runtime behaviour is governed by YAML files under `configs/`, including observability,
logging and reliability policy (SLO targets, error-budget actions, capacity headroom and
production-readiness thresholds). Configuration is data, not code: changing a policy does not
require changing an engine.

## Running Tests

```bash
# Full suite (1,503 tests)
PYTHONPATH=src:scripts pytest tests/ -q

# A single subsystem
PYTHONPATH=src pytest tests/test_observability_models.py -q
```

The tests are deterministic and framework-agnostic: they use only standard assertions and
parameterisation, with no fixtures, network access or external services.

## Deployment

```bash
# Local (Docker Compose, hardened production profile)
cd deployment/docker && docker compose -f docker-compose.prod.yml up -d

# Kubernetes (dependency-ordered apply, rollout wait, in-pod health verification)
deployment/scripts/deploy_kubernetes.sh

# Health-gated rollback
deployment/scripts/rollback.sh
```

See [`deployment_guide.md`](deployment_guide.md) for the full procedure.

## Screenshots

<!-- ![Operations dashboard](docs/figures/operations_dashboard.png) (figure asset pending) -->
*Operations dashboard executive summary. **[image placeholder]***

<!-- ![Readiness report](docs/figures/readiness_report.png) (figure asset pending) -->
*Production-readiness assessment. **[image placeholder]***

## Architecture Figures

Architecture figures (overall platform and per-layer) are referenced throughout the
documentation. Figure files are placeholders pending export.

- `docs/figures/architecture_overview.png` — overall platform *(placeholder)*
- `docs/figures/mlops_lineage.png` — MLOps provenance *(placeholder)*
- `docs/figures/deployment_topology.png` — Kubernetes topology *(placeholder)*
- `docs/figures/observability_slo.png` — observability and SLO *(placeholder)*

## Research Paper

A companion IEEE-style research paper describes the architecture and methodology:
[`research_paper.md`](../research_paper.md). See also [`references.bib`](../references.bib),
[`appendices.md`](../appendices.md) and [`glossary.md`](../glossary.md).

## Performance Highlights

This repository reports engineering characteristics, not fabricated benchmark figures. No
synthetic latency or throughput numbers are claimed; the highlights below describe verified
design properties and configurable targets.

- **Deterministic computation** — identical inputs yield identical outputs, including fully
  serialised reports, verified by dedicated determinism tests.
- **Zero-downtime deployment** — rolling updates with no unavailable replicas and health-gated
  rollback.
- **Elastic scaling** — horizontal autoscaling configured from 3 to 12 replicas on CPU and
  memory utilisation.
- **Default SLO targets** — availability ≥ 0.99 (30-day window); P95 latency ≤ 250 ms; error
  rate ≤ 0.01; data freshness ≤ 300 s. These are configured objectives evaluated at runtime,
  not measured results.

## Engineering Highlights

- Architecture-first, additive construction across all layers; no layer modifies its predecessors.
- SOLID design, dependency injection, immutable domain models, thread safety and composition
  over inheritance applied uniformly.
- Pure Python with a single numerical dependency; no external observability, orchestration or
  ML platforms required at runtime.
- Provenance, quality gating, deployment, recovery and reliability provided as platform
  properties rather than per-capability reinventions.

## Repository Statistics

| Metric | Value |
|--------|------:|
| Production-engineering modules | 30 |
| Source lines of code (measured) | 10,620 |
| Test files | 27 |
| Automated tests | 1,503 (all passing) |
| Kubernetes manifests | 10 |
| CI/CD workflows | 3 |
| Configuration files | 5 |

*Statistics are measured for the verified production-engineering subsystems; capability-layer
modules are described architecturally.*

## Future Research

Opportunities include empirical study of integrated platforms under sustained workloads,
formalisation of cross-capability data contracts, observability for agentic systems,
reproducibility for unstructured knowledge corpora, and decision-quality (rather than
prediction-accuracy) evaluation. See the research paper for detail.

## License

Released under the [MIT License](../../../LICENSE).

## Citation

If you reference this work, please cite it using [`CITATION.cff`](../../../CITATION.cff).

## Acknowledgements

This platform builds on established research and practice in digital twins, MLOps, retrieval-
augmented generation, cloud-native infrastructure, site reliability engineering and software
architecture. Full references are provided in the research paper's bibliography.