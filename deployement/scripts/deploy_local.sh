#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Local deployment: build the image, start the production compose stack, and
# verify health. Idempotent and safe to re-run.
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/deployment/docker/docker-compose.prod.yml"
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"

log() { printf '%s [deploy-local] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

require() { command -v "$1" >/dev/null 2>&1 || { log "ERROR: '$1' is required but not installed"; exit 1; }; }
require docker

log "building image and starting stack from ${COMPOSE_FILE}"
${COMPOSE_BIN} -f "${COMPOSE_FILE}" up --build -d

log "waiting for container health"
container="edt-app"
deadline=$(( $(date +%s) + 120 ))
status="starting"
while [ "$(date +%s)" -lt "${deadline}" ]; do
  status="$(docker inspect --format '{{ if .State.Health }}{{ .State.Health.Status }}{{ else }}none{{ end }}' "${container}" 2>/dev/null || echo "absent")"
  case "${status}" in
    healthy) log "container is healthy"; break ;;
    unhealthy) log "ERROR: container reported unhealthy"; docker logs --tail 50 "${container}" || true; exit 1 ;;
    *) sleep 5 ;;
  esac
done

if [ "${status}" != "healthy" ]; then
  log "running an in-container health verification as a fallback"
  docker exec "${container}" python /app/deployment/scripts/health_check.py --root /app
fi

log "local deployment complete"