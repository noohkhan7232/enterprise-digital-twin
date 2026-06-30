# Live Demo Checklist

A pre-flight and in-flight checklist for delivering the platform demonstration reliably. Work
top-to-bottom before the session; keep the recovery section visible during it.

---

## 1. Environment Verification (T-30 minutes)

- [ ] Correct machine, charged, charger present; power outlet located at the venue.
- [ ] Display adapter and cable tested with the venue projector/screen.
- [ ] Screen resolution set; font sizes increased in terminal and editor for legibility.
- [ ] Notifications silenced (system, chat, email); do-not-disturb enabled.
- [ ] Network checked; offline fallback confirmed (recordings available — see §7).
- [ ] Python 3.12 available; virtualenv created and activated.
- [ ] Dependencies installed: `numpy`, `pyyaml`, `pytest`, `pytest-cov`.

## 2. Repository

- [ ] Repository cloned at the intended commit/tag (v1.0.0).
- [ ] `PYTHONPATH` exported: `export PYTHONPATH=src:scripts`.
- [ ] Working tree clean (`git status`), correct branch checked out.
- [ ] Full test suite green before the session:
  ```bash
  pytest tests/ -q          # expect 1,503 passing
  ```
- [ ] Observability CLI demos run and produce JSON:
  ```bash
  PYTHONPATH=src python3 -c "from observability import main; main(['all'])"
  ```

## 3. Docker (if shown)

- [ ] Docker daemon running; sufficient disk and memory.
- [ ] Images pre-built to avoid live build waits:
  ```bash
  cd deployment/docker && docker compose -f docker-compose.prod.yml build
  ```
- [ ] Production stack starts and reports healthy:
  ```bash
  docker compose -f docker-compose.prod.yml up -d
  docker inspect --format '{{ .State.Health.Status }}' edt-app
  ```
- [ ] Stack stops cleanly; rehearsed start time noted.

## 4. Kubernetes (if shown)

- [ ] Cluster reachable; `kubectl` context correct (verify twice — wrong-context risk is real).
- [ ] Namespace and manifests applied in a rehearsal run:
  ```bash
  deployment/scripts/deploy_kubernetes.sh
  ```
- [ ] Rollout reaches ready; pods healthy.
- [ ] Rollback rehearsed:
  ```bash
  deployment/scripts/rollback.sh
  ```

## 5. Health Checks

- [ ] Local health check passes:
  ```bash
  python3 deployment/scripts/health_check.py --root . --endpoint http://localhost:8080/health
  ```
- [ ] Health output interpretation rehearsed (ready vs healthy vs degraded vs unhealthy).
- [ ] Container `HEALTHCHECK` and Kubernetes probe behaviour understood for Q&A.

## 6. Demo Flow

- [ ] Script chosen (executive or technical) and timed in rehearsal.
- [ ] Commands copied into a notes file for paste-in (avoid live typos).
- [ ] Architecture figure open and ready to reference.
- [ ] Determinism point rehearsed: run a demo twice, show identical JSON.
- [ ] Honesty notes ready: SLO values are configured targets; readiness is a self-assessment; no
      fabricated benchmarks.
- [ ] Timer or visible clock arranged; segment time budgets memorised.

## 7. Backup Plan

- [ ] Pre-recorded terminal captures of every command's output saved locally (asciinema or video).
- [ ] Screenshots of: test suite summary, each CLI demo's JSON, the readiness report, the operations
      dashboard summary, and a healthy deployment.
- [ ] Static copy of the architecture figure and slides available offline (PDF).
- [ ] If network/cluster is unavailable, switch to recordings without breaking narrative flow.

## 8. Recovery Plan (during the session)

- [ ] **Command fails or hangs:** stop it, narrate the expected result, switch to the recording or
      screenshot, continue. Do not debug live.
- [ ] **Cluster/Docker unavailable:** drop the deployment segment; describe it from the deployment
      guide and show the recorded rollout/rollback.
- [ ] **Time overrun:** skip the tracing and capacity demos; keep metrics, readiness and the test
      summary.
- [ ] **Tough question mid-flow:** acknowledge, park it for Q&A, continue the flow.
- [ ] **Total AV failure:** present from the slides and the printed one-pager; the narrative stands
      without the terminal.

## 9. Post-Demo

- [ ] Stop running containers / clean up cluster resources.
- [ ] Note questions asked for follow-up.
- [ ] Reset environment for the next session.