#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Production entrypoint: deterministic startup, signal forwarding, graceful
# shutdown. Forwards SIGTERM/SIGINT to the child so Kubernetes and Compose can
# stop the container cleanly within the termination grace period.
# ----------------------------------------------------------------------------
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
APP_PORT="${APP_PORT:-8080}"

log() { printf '%s [entrypoint] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

child_pid=0
shutdown() {
  log "received shutdown signal, forwarding to child pid ${child_pid}"
  if [ "${child_pid}" -ne 0 ]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  log "shutdown complete"
  exit 0
}
trap shutdown SIGTERM SIGINT

run_health() {
  python "${APP_ROOT}/deployment/scripts/health_check.py" --root "${APP_ROOT}" "$@"
}

cmd="${1:-serve}"
case "${cmd}" in
  serve)
    log "starting application (env=${APP_ENV:-production}, port=${APP_PORT})"
    # Run a deterministic readiness self-check before serving.
    run_health --quiet || log "warning: startup health check reported degraded/unhealthy state"
    # The platform exposes its subsystems as importable packages; the long-running
    # process below keeps the container alive and responsive to signals. Replace
    # the module path here with the service runner for your environment.
    python -m http.server "${APP_PORT}" --directory "${APP_ROOT}" &
    child_pid=$!
    wait "${child_pid}"
    ;;
  health)
    run_health
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    log "executing custom command: $*"
    exec "$@"
    ;;
esac