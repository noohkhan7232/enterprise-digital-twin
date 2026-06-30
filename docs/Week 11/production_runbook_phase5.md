# Production Runbook — Enterprise Digital Twin & Decision Intelligence Platform

> Operational reference for on-call engineers. Pairs with the observability
> subsystem (`src/observability/`, v11.5.0) and the deployment assets from Week
> 11 Phase 4 (`deployment/`). Procedures are deterministic and replayable.

---

## 1. Purpose and Scope

This runbook describes how to detect, triage, mitigate and recover from
production incidents on the platform, and how to use the observability subsystem
to do so. It assumes the platform is deployed per `deployment/` (Docker Compose
locally, Kubernetes in production) and that the observability engines are
available for computing reliability, SLO compliance, capacity and readiness. The
runbook is structured so that each procedure maps to a concrete capability in the
codebase rather than to an external tool.

## 2. Severity Definitions

Severities follow the incident manager's four levels. SEV1 is a full or
near-full outage or data-integrity risk affecting most users; it demands
immediate response and a release freeze. SEV2 is a major degradation affecting a
significant subset of users or a critical workflow. SEV3 is a contained problem
with a workaround and limited impact. SEV4 is a minor or cosmetic issue tracked
but not paged. The dashboard's executive summary surfaces the most severe active
incident and, for any active SEV1, recommends engaging incident response and
holding all releases.

## 3. Roles During an Incident

The incident commander owns coordination and the decision to escalate or stand
down. The operations engineer drives mitigation and recovery. The communications
lead keeps stakeholders informed. The scribe records the timeline. In a small
on-call rotation one person may hold several roles, but the timeline must always
be recorded, because the incident manager's postmortem is generated from it.

## 4. Detection

Incidents are detected from the observability signals. Reliability degradation
shows up as a falling reliability score or availability from the reliability
engine. Objective breaches show up as failing SLOs and a rising burn rate from
the SLI/SLO engine; a burn rate above the policy's alert threshold means the
error budget is being consumed faster than the window can sustain. Capacity
pressure shows up as a near-term exhaustion step in a capacity forecast. Request
-level symptoms show up as error spans and inflated critical-path durations in
the tracing engine, and as anomalous percentiles and trends in the metrics
engine.

## 5. Incident Lifecycle Procedure

Open an incident as soon as impact is suspected, with a title, severity and the
affected services. Move it to investigating while gathering signal, to
identified once the root cause is understood (recording that root cause), and to
monitoring once a mitigation is in place and being observed. Resolve it when
impact has ended; the resolution timestamp is recorded automatically and
recovery time becomes available. Close it after the postmortem is complete.
Every transition is validated, so illegal jumps are rejected, and every
transition notifies subscribed observers (paging, chat, audit). Record
corrective actions as you go — they accumulate on the incident and appear in the
postmortem.

## 6. Common Incidents and Recovery Steps

For an availability drop, confirm the breach via the reliability engine and the
availability SLO, identify the failing component from the tracing critical path
and error spans, mitigate (fail over, scale out, or roll back), and verify
recovery before resolving. For latency regression, locate the dominant span on
the critical path, correlate with a metrics trend on the relevant latency
series, and address the hot path or provision capacity. For error-budget burn,
freeze releases per policy when the budget is exhausted, stabilise, and resume
only once burn rate returns below threshold. For capacity exhaustion, use the
forecast's recommended capacity (projected peak plus headroom) to provision
ahead of the exhaustion step. For a bad release, follow the rollback procedure.

## 7. Rollback and Recovery

Rollback uses the Phase 4 tooling. In Kubernetes, run `deployment/scripts/
rollback.sh` (optionally `--to-revision N`); it reverts the deployment, waits for
the rollout to stabilise and verifies health in a live pod via
`deployment/scripts/health_check.py`, failing if the rolled-back state is not
healthy. Locally, redeploy a known-good image with `deployment/scripts/
deploy_local.sh`. Because the production deployment retains revision history and
rolls with zero unavailable replicas, rollbacks are themselves zero-downtime.
Durable state lives in the persistent volume rather than in pods, so pod
replacement during recovery does not lose data.

## 8. Escalation

Escalate from SEV3 to SEV2, or SEV2 to SEV1, whenever impact widens, a
mitigation fails, or recovery exceeds the expected window. Escalation is a state
the incident commander declares; raising severity should trigger the same
response posture as opening at that severity — for SEV1 that means engaging the
full response and freezing releases. De-escalate only when the dashboard's active
-incident view and the reliability signal both confirm reduced impact.

## 9. Postmortem

Once an incident is resolved, generate the postmortem from the incident manager.
It contains the title, severity, root cause, time to recovery, affected
services, the accumulated corrective actions and the full ordered timeline. The
reliability policy expects a root cause and corrective actions to be recorded;
fill these during the incident so the postmortem is complete at resolution. Feed
durable corrective actions back into the platform — for example as new readiness
checks, capacity headroom changes or additional SLOs.

## 10. Runbook Entries

Recurring scenarios are captured as `RunbookEntry` records: an identifier, a
title, the relevant severity, observable symptoms, ordered recovery steps and the
related services. Maintaining a library of these turns hard-won incident
knowledge into a structured, queryable form that maps directly onto the incident
manager's lifecycle and the dashboard's signals.

## 11. Health Verification

After any mitigation, rollback or deployment, verify health with
`deployment/scripts/health_check.py --root <path>`, which checks the application
packages, the MLOps and monitoring subsystems, the HTTP health endpoint and the
required configuration, and returns a deterministic ready/healthy verdict. The
same check backs the container `HEALTHCHECK` and the Kubernetes probes, so
"healthy" means the same thing during recovery as it does in steady state.

## 12. Post-Incident Readiness Review

After significant incidents, re-run the production-readiness assessment to
confirm the platform still scores within its expected band across all ten areas,
and review whether the incident exposed a gap — missing capacity headroom, an
absent SLO, or a security or deployment control — that should be closed. The
assessment's weighted score and categorical level give a single, reproducible
answer to whether the platform remains fit to carry production load.