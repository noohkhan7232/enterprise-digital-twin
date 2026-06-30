# Week 11 — Phase 4: Enterprise Containerisation & Kubernetes Deployment Platform

> Subsystem: `deployment/` (`docker/`, `kubernetes/`, `scripts/`) plus `tests/test_health_check.py`.
> Status: production-ready, deterministic, non-root, hardened by default.
> Integration model: **additive only** — adds new deployment assets; modifies no prior week.

---

## 1. Business Motivation

A decision-intelligence platform is only as valuable as its uptime. The MLOps,
monitoring and CI/CD work of the previous phases produces a validated artifact;
this phase turns that artifact into a service that runs reliably, scales with
demand, fails safely and recovers quickly. The goal is operational
trustworthiness: any engineer can build the image, stand the stack up locally,
deploy it to Kubernetes, and — critically — roll it back, with the same
deterministic health check gating every step. The assets here encode the
operational defaults a platform team would otherwise rediscover through
incidents: non-root containers, read-only root filesystems, resource bounds,
liveness/readiness/startup probes, autoscaling, disruption budgets and network
isolation.

## 2. Deployment Architecture

The deployment layer is three cooperating asset groups bound together by one
deterministic health check.

```
        deployment/docker/                deployment/kubernetes/
   Dockerfile (multi-stage, prod)    namespace · configmap · secret(template)
   Dockerfile.dev (bind-mount)       deployment · service · ingress
   docker-compose.yml (dev)          hpa · pdb · networkpolicy · pvc
   docker-compose.prod.yml (prod)
   entrypoint.sh (signals)
                    \                         /
                     \                       /
                  deployment/scripts/   +  tests/test_health_check.py
        deploy_local.sh · deploy_kubernetes.sh · rollback.sh
                     health_check.py  <-- single source of health truth
```

`health_check.py` is the keystone: the container `HEALTHCHECK`, the Kubernetes
probes' intent, the Compose healthcheck and every deployment script all converge
on the same checker, so "healthy" means the same thing everywhere.

## 3. Docker Design

The production image is a multi-stage build. The builder stage installs runtime
dependencies into an isolated prefix on a `python:3.12-slim` base; the runtime
stage copies only that prefix and the application code, so build toolchains
never reach production. The image runs as an unprivileged user (uid/gid 10001),
sets `PYTHONDONTWRITEBYTECODE` and `PYTHONUNBUFFERED` for clean, deterministic
behaviour, exposes port 8080, and declares a `HEALTHCHECK` that runs the health
checker. Dependency versions are constrained for reproducibility, and the notes
below describe hash-pinning for fully deterministic builds.

The development image is single-stage, adds `pytest`/`pytest-cov` and is paired
with a bind mount so source edits are reflected immediately. `entrypoint.sh` is
the production entrypoint: it traps `SIGTERM`/`SIGINT`, forwards them to the
child process and waits, giving graceful shutdown within the termination grace
period; it runs a startup self-check before serving and supports `serve`,
`health` and `shell` subcommands. `.dockerignore` keeps caches, tests, docs and
internal tooling out of the build context for a small, deterministic image.

## 4. Docker Compose

Two compose files express the two environments. The development file
(`docker-compose.yml`) builds the dev image, bind-mounts `src`, `scripts`,
`configs`, `tests` and `deployment`, injects environment variables and uses an
`unless-stopped` restart policy for fast iteration. The production file
(`docker-compose.prod.yml`) builds the runtime image and adds the operational
hardening expected in production: a container healthcheck wired to the health
checker, JSON file logging with rotation, a read-only root filesystem with a
`tmpfs` for `/tmp`, `no-new-privileges` and all Linux capabilities dropped, an
`on-failure` restart policy with a cap, a 30-second stop grace period, and CPU
and memory reservations and limits.

## 5. Kubernetes Design

Ten manifests describe a production deployment, applied in dependency order.
The **Namespace** carries restricted Pod Security Admission labels. The
**ConfigMap** holds non-secret configuration; the **Secret** is a clearly
marked template with placeholder values only. The **Deployment** runs three
replicas with a zero-downtime rolling-update strategy (`maxUnavailable: 0`,
`maxSurge: 1`), a hardened pod and container security context (non-root,
read-only root filesystem, no privilege escalation, dropped capabilities,
`RuntimeDefault` seccomp), resource requests and limits, topology spread across
nodes, and three probes — startup, readiness and liveness — all targeting
`/health`. The **Service** is a ClusterIP fronting the pods; the **Ingress**
terminates TLS and routes by host. The **HorizontalPodAutoscaler** scales from
3 to 12 replicas on CPU and memory utilisation with tuned scale-up/scale-down
behaviour. The **NetworkPolicy** restricts ingress to the ingress controller and
egress to DNS and HTTPS while blocking the cloud metadata endpoint. The
**PodDisruptionBudget** keeps at least two pods available during voluntary
disruptions, and the **PersistentVolumeClaim** provides durable storage.

## 6. Security

Security is layered and on by default: images run as a fixed unprivileged user;
the root filesystem is read-only in both Compose production and Kubernetes, with
writable scratch confined to `tmpfs`/`emptyDir`; privilege escalation is
disabled and all capabilities are dropped; the namespace enforces the restricted
Pod Security Standard; the network policy denies all traffic except the
explicitly allowed ingress and egress, including an explicit exception that
blocks the instance metadata service. Secrets are never committed — only a
clearly labelled template is shipped, and the Kubernetes deploy script prefers a
real `secret.yaml` (supplied out-of-band or by an external secrets manager) and
warns loudly if only the template is present.

## 7. Scaling

Horizontal scaling is delegated to the HPA, which targets 70% CPU and 75% memory
utilisation between three and twelve replicas. Scale-up is aggressive (up to
100% more pods every 30 seconds after a short stabilisation window) to absorb
load spikes; scale-down is deliberately conservative (at most 50% every minute
after a five-minute window) to avoid thrashing. Resource requests make the
scheduler's decisions predictable, and topology spread constraints distribute
replicas across nodes so a single node failure cannot take down a majority.

## 8. Health Checks

`health_check.py` validates five concerns and returns a deterministic JSON
report: the application (source packages present), the MLOps subsystem, the
monitoring subsystem, the HTTP health endpoint (probed through an injected,
fail-closed probe), and the required configuration (present and non-empty). Each
check yields HEALTHY, DEGRADED or UNHEALTHY; the overall status is the most
severe, and the report exposes both `healthy` and `ready` flags — readiness is
true unless something is UNHEALTHY, so a missing endpoint configuration degrades
without failing readiness. The probe is injected, which is what makes the
checker fully deterministic and unit-testable offline; the suite in
`tests/test_health_check.py` exercises healthy, unhealthy, missing-configuration,
degraded, JSON-output, CLI and edge-case paths. The Kubernetes startup probe
gates liveness until the process is up, the readiness probe controls traffic and
the liveness probe restarts a wedged container.

## 9. Rollback Strategy

`rollback.sh` performs `kubectl rollout undo` against the deployment — to the
previous revision by default or to a specified revision — waits for the rollout
to stabilise, and then verifies health in a live pod, failing non-zero if the
rolled-back state is not healthy. Because the deployment keeps a generous
revision history and uses `maxUnavailable: 0`, rollbacks are themselves
zero-downtime. This complements the release policy from Phase 3, whose rollback
section declares automatic rollback on failure and the number of previous
versions to retain.

## 10. Disaster Recovery

Recovery rests on three properties: durable state lives in the
PersistentVolumeClaim rather than in pods, so pod loss is not data loss;
deployments are fully declarative, so the entire stack can be recreated from the
manifests with `deploy_kubernetes.sh`; and the PodDisruptionBudget plus
multi-replica, multi-node spread ensure the service survives node drains and
partial failures. Combined with retained deployment revisions and health-gated
rollback, the platform can be restored to a known-good state quickly and
verifiably.

## 11. Deployment Workflow

Locally, `deploy_local.sh` builds the image, starts the production compose
stack, polls Docker's health status and falls back to an in-container health
verification. For Kubernetes, `deploy_kubernetes.sh` applies the manifests in
safe order (namespace, configuration and secrets, storage, workload, networking,
then resilience policies), waits for the rollout and runs a health verification
inside a pod. If a release misbehaves, `rollback.sh` reverts and re-verifies.
Every path ends in the same health check, so success criteria are identical
across environments.

## 12. Integration with CI/CD

The Phase 3 pipeline and this phase share the same health and readiness
vocabulary. The CI workflow's build-validation step runs the deployment-readiness
validator, which credits exactly the assets produced here (Dockerfile,
Kubernetes manifests, health endpoint signals, rollback and recovery policy).
The release workflow's deployment-readiness step gates tagged releases on the
same assets, and the release policy's rollback settings are realised by
`rollback.sh`. In effect, Phase 3 decides whether a release *may* ship and this
phase determines how it *runs* and *recovers*.

## 13. Integration with Monitoring

`health_check.py` treats the Phase 2 monitoring subsystem as a first-class
health dependency: if `src/monitoring` is absent or not importable, the
deployment is UNHEALTHY. Operationally, the monitoring subsystem consumes the
running service's predictions and emits drift, health and alert signals; this
phase ensures monitoring is present and wired before traffic is admitted, and
the readiness probe keeps a pod out of rotation until its dependencies are
satisfied.

## 14. Integration with MLOps

Symmetrically, the MLOps subsystem from Phase 1 is a required health dependency:
the model registry, experiment tracker and artifact store ship inside the image
and must be importable for the deployment to be considered healthy. The
PersistentVolumeClaim provides a durable location for MLOps artifacts in
environments that persist them, and the configuration check ensures the policy
files that govern promotion and release are present at runtime. Together the
four phases form a single line from experiment to monitored, recoverable
production service.