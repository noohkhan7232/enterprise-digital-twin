#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Rollback the Kubernetes deployment to the previous (or a specified) revision,
# wait for the rollout, then verify health. Usage:
#   ./rollback.sh                # roll back one revision
#   ./rollback.sh --to-revision 7
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="enterprise-digital-twin"
DEPLOYMENT="edt-app"
KUBECTL="${KUBECTL:-kubectl}"
REVISION=""

log() { printf '%s [rollback] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
require() { command -v "$1" >/dev/null 2>&1 || { log "ERROR: '$1' is required but not installed"; exit 1; }; }
require "${KUBECTL%% *}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --to-revision) REVISION="${2:-}"; shift 2 ;;
    *) log "unknown argument: $1"; exit 2 ;;
  esac
done

if [ -n "${REVISION}" ]; then
  log "rolling back ${DEPLOYMENT} to revision ${REVISION}"
  ${KUBECTL} -n "${NAMESPACE}" rollout undo "deployment/${DEPLOYMENT}" --to-revision="${REVISION}"
else
  log "rolling back ${DEPLOYMENT} to the previous revision"
  ${KUBECTL} -n "${NAMESPACE}" rollout undo "deployment/${DEPLOYMENT}"
fi

log "waiting for rollout to stabilise"
${KUBECTL} -n "${NAMESPACE}" rollout status "deployment/${DEPLOYMENT}" --timeout=180s

pod="$(${KUBECTL} -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')"
log "verifying health in pod ${pod}"
if ! ${KUBECTL} -n "${NAMESPACE}" exec "${pod}" -- python /app/deployment/scripts/health_check.py --root /app --quiet; then
  log "ERROR: health verification failed after rollback"
  exit 1
fi

log "rollback complete and healthy"