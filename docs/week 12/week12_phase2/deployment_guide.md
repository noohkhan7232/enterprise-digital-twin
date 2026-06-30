# Deployment Guide

This guide describes how to deploy and operate the platform across local, container and Kubernetes
environments, and how rollback, health checks and monitoring fit together. All paths refer to the
`deployment/` subsystem.

---

## 1. Local Deployment

For development and evaluation, the platform runs locally via Docker Compose.

```bash
cd deployment/docker
docker compose up --build          # development image, bind-mounted source
```

The development image includes test tooling and mounts the source tree so that edits are reflected
immediately. It runs as a non-root user.

## 2. Docker

The production image is built and run from the hardened compose profile.

```bash
cd deployment/docker
docker compose -f docker-compose.prod.yml up -d
```

Production-image characteristics:

- **Multi-stage build** separating build tooling from the runtime surface.
- **Non-root execution** under a fixed unprivileged user.
- **Container health check** wired to the deterministic health checker.
- **Hardening**: read-only root filesystem with a writable `tmpfs`, no privilege escalation, all
  Linux capabilities dropped.
- **Resource bounds**: CPU and memory reservations and limits.
- **Graceful shutdown**: the entrypoint forwards termination signals to the child process.

## 3. Kubernetes

The orchestrated deployment is described by ten manifests and applied by an automation script.

```bash
deployment/scripts/deploy_kubernetes.sh
```

The script applies resources in dependency order: namespace, configuration and secrets, storage,
workload, networking, then resilience policies. It then waits for the rollout to complete and runs
the health check inside a live pod before reporting success.

| Manifest | Purpose |
|----------|---------|
| Namespace | Restricted pod-security admission |
| ConfigMap | Non-secret configuration |
| Secret (template) | Placeholder for secret material supplied out-of-band |
| Deployment | Zero-downtime rolling deployment with probes and hardened security context |
| Service | Cluster-internal access |
| Ingress | TLS-terminated external routing |
| HorizontalPodAutoscaler | Autoscaling on CPU and memory utilisation |
| NetworkPolicy | Default-deny with explicit ingress/egress |
| PodDisruptionBudget | Availability during voluntary disruption |
| PersistentVolumeClaim | Durable storage |

## 4. Rollback

```bash
deployment/scripts/rollback.sh                 # previous revision
deployment/scripts/rollback.sh --to-revision N # specific revision
```

Rollback reverts the deployment, waits for the rollout to stabilise, and re-verifies health in a
live pod, failing if the rolled-back state is unhealthy. Because the deployment retains revision
history and rolls with no unavailable replicas, rollback is itself zero-downtime.

## 5. Health Checks

A single deterministic health check is the keystone of deployment and recovery.

```bash
python3 deployment/scripts/health_check.py --root <path> --endpoint <url>
```

It validates the application packages, the MLOps and monitoring subsystems, the HTTP health
endpoint (through an injected, fail-closed probe) and the required configuration, returning a
deterministic ready/healthy verdict. The same check backs the container `HEALTHCHECK`, the
Kubernetes startup/readiness/liveness probes' intent, and every deployment and rollback script.

## 6. Monitoring and Operation

Once deployed, the platform is operated through the observability subsystem:

- **Metrics** across application, inference, workflow, scheduler, deployment, CI/CD, monitoring and
  business categories, with percentiles, windows and trends.
- **Reliability** — availability, MTBF, MTTR and a composite reliability score from live outcomes
  and outage windows.
- **SLI/SLO** — evaluation against the default objectives (availability ≥ 0.99; P95 latency ≤
  250 ms; error rate ≤ 0.01; freshness ≤ 300 s), with error budgets and burn rates; budget
  exhaustion triggers the policy's release-freeze action.
- **Incidents** — a validated lifecycle with timeline, recovery time and postmortem generation.
- **Capacity** — deterministic forecasts of resource exhaustion with headroom recommendations.

The operations dashboard composes these into a single snapshot with a graded executive summary, and
the production-readiness assessment scores the platform across ten areas. After any mitigation,
rollback or deployment, re-run the health check and review the readiness assessment to confirm the
platform remains fit to carry load.

## 7. High Availability

High availability follows from multiple replicas, autoscaling (configured 3–12 replicas),
node-level topology spread, the pod disruption budget and durable state in the persistent volume.
The service therefore survives node failures and voluntary disruptions without data loss, and
recovery is declarative: the entire stack can be recreated from the manifests.