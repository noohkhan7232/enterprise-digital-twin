#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Kubernetes deployment: apply namespace, config, secrets and workloads in a
# safe order, wait for rollout, then verify health. Apply order matters:
# namespace -> config/secret -> storage -> workload -> networking -> policies.
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(cd "${SCRIPT_DIR}/../kubernetes" && pwd)"
NAMESPACE="enterprise-digital-twin"
DEPLOYMENT="edt-app"
KUBECTL="${KUBECTL:-kubectl}"

log() { printf '%s [deploy-k8s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
require() { command -v "$1" >/dev/null 2>&1 || { log "ERROR: '$1' is required but not installed"; exit 1; }; }
require "${KUBECTL%% *}"

apply() { log "applying $1"; ${KUBECTL} apply -f "${K8S_DIR}/$1"; }

# 1. Namespace.
apply namespace.yaml

# 2. Configuration and secrets. Use the real secret if present, else the template.
apply configmap.yaml
if [ -f "${K8S_DIR}/secret.yaml" ]; then
  apply secret.yaml
else
  log "WARNING: secret.yaml not found; applying template secret.example.yaml (placeholders only)"
  apply secret.example.yaml
fi

# 3. Storage.
apply persistentvolumeclaim.yaml

# 4. Workload.
apply deployment.yaml

# 5. Networking.
apply service.yaml
apply ingress.yaml

# 6. Resilience policies.
apply hpa.yaml
apply poddisruptionbudget.yaml
apply networkpolicy.yaml

# 7. Wait for the rollout to complete.
log "waiting for rollout of ${DEPLOYMENT}"
${KUBECTL} -n "${NAMESPACE}" rollout status "deployment/${DEPLOYMENT}" --timeout=180s

# 8. Health verification from inside a running pod.
pod="$(${KUBECTL} -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')"
log "running health verification in pod ${pod}"
${KUBECTL} -n "${NAMESPACE}" exec "${pod}" -- python /app/deployment/scripts/health_check.py --root /app

log "kubernetes deployment complete"