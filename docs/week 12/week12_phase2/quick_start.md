# Quick Start

This guide takes you from a fresh clone to a running, tested and deployed platform. It assumes a
Unix-like environment with Python 3.12, Git, and (for the container and orchestration steps) Docker
and `kubectl` with access to a cluster.

---

## 1. Clone

```bash
git clone https://github.com/<org>/enterprise-digital-twin.git
cd enterprise-digital-twin
```

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyyaml          # runtime dependencies
pip install pytest pytest-cov     # development and testing
```

The platform deliberately keeps runtime dependencies minimal: NumPy for numerical computing and
PyYAML for configuration parsing.

## 3. Configuration

Runtime behaviour is governed by YAML files under `configs/`:

| File | Purpose |
|------|---------|
| `observability.yaml` | Metric categories, percentiles, tracing and dashboard settings |
| `logging.yaml` | Structured-logging format, severity and default context |
| `reliability_policy.yaml` | SLO targets, error-budget actions, capacity headroom, readiness thresholds |
| `quality_gate.yaml` | CI/CD quality-gate configuration |
| `release_policy.yaml` | Release validation policy |

Configuration is data, not code. Adjusting a policy (for example, an SLO target) does not require
changing an engine.

## 4. Run

```bash
# Observability CLI demonstrations (deterministic JSON output)
PYTHONPATH=src python3 -c "from observability import main; main(['all'])"
#   individual demos: metrics | tracing | reliability | capacity | readiness
```

Each demonstration runs a subsystem on fixed inputs and emits sorted-key JSON, so output is
byte-identical on every run.

## 5. Tests

```bash
# Full suite (1,503 tests)
PYTHONPATH=src:scripts pytest tests/ -q

# A single subsystem
PYTHONPATH=src pytest tests/test_metrics_engine.py -q
```

All tests are deterministic and require no network access or external services.

## 6. Docker

```bash
cd deployment/docker

# Development image (bind-mounted source for fast iteration)
docker compose up --build

# Hardened production image (health check, read-only root, resource limits)
docker compose -f docker-compose.prod.yml up -d
```

The production image is multi-stage and non-root, declares a container health check, and runs with
a read-only root filesystem and dropped Linux capabilities.

## 7. Kubernetes

```bash
# Apply manifests in dependency order, wait for rollout, verify health in a live pod
deployment/scripts/deploy_kubernetes.sh

# Roll back to the previous revision and re-verify health
deployment/scripts/rollback.sh
```

The deployment script applies resources in the correct order (namespace, configuration and
secrets, storage, workload, networking, then resilience policies), waits for the rollout to
complete, and runs the deterministic health check inside a running pod before reporting success.

## 8. Verify Health

```bash
# Run the deterministic health check against a deployment root
python3 deployment/scripts/health_check.py --root . --endpoint http://localhost:8080/health
```

The same health check backs the container `HEALTHCHECK`, the Kubernetes probes and the deployment
scripts, so "healthy" means the same thing in every context.

## Next Steps

- Read [`architecture_overview.md`](architecture_overview.md) to understand the ten layers.
- Read [`developer_guide.md`](developer_guide.md) to contribute.
- Read [`deployment_guide.md`](deployment_guide.md) for production operations.