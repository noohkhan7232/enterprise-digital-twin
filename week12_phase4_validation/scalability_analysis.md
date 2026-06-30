# Scalability Analysis

A qualitative analysis of how the platform scales along several axes, the mechanisms that support
scaling, and the trade-offs involved. It describes design properties and the methodology for
measuring scaling behaviour; it does not present fabricated scaling numbers.

---

## 1. Horizontal Scaling

The deployment runs multiple identical replicas behind a service, with state externalised to durable
storage rather than held in pods. Because replicas are stateless with respect to request handling,
horizontal scaling is the primary scaling mode: adding replicas increases capacity for concurrent
work. The Kubernetes `HorizontalPodAutoscaler` is configured to scale between three and twelve
replicas on CPU and memory utilisation, with deliberately aggressive scale-up and conservative
scale-down to absorb spikes without thrashing.

**Measurement methodology:** drive a fixed workload at increasing concurrency and record replica
count, utilisation and rollout behaviour over time; report the relationship qualitatively until
measured under representative load.

## 2. Vertical Scaling

Per-replica resource requests and limits are declared, so a replica can be given more CPU and memory
where a workload benefits from it. Vertical scaling is bounded by node capacity and by the diminishing
returns of single-process Python for CPU-bound work; the platform favours horizontal scaling for
throughput and reserves vertical scaling for memory-pressured or latency-sensitive components.

**Trade-off:** vertical scaling is simpler operationally but less elastic and less fault-tolerant than
horizontal scaling; the design prefers the latter.

## 3. Kubernetes Scaling

Orchestration provides the scaling substrate: the autoscaler adjusts replica count; topology spread
constraints distribute replicas across nodes; the pod disruption budget preserves availability during
voluntary disruption; and rolling updates change versions without downtime. These mechanisms make
scaling and change operationally routine rather than exceptional.

## 4. Data Growth

The platform treats datasets as versioned, immutable snapshots and artifacts as content-addressed,
which bounds duplication and keeps growth auditable. The capacity planner forecasts data growth using
deterministic strategies and can flag projected exhaustion against a configured limit.

**Trade-off:** immutable versioning increases storage footprint relative to in-place mutation, in
exchange for reproducibility and provenance.

## 5. Model Growth

Models accumulate in the versioned registry with semantic versioning and stage promotion. Growth is
managed by retention policy rather than by unbounded accumulation. The capacity planner includes a
model-growth resource class for forecasting.

**Trade-off:** retaining historical versions supports reproducibility and rollback at the cost of
registry size; retention policy is the control.

## 6. Workflow Growth

Workflows compose steps with explicit state. As the number and complexity of workflows grow, the
engine's explicit-state design keeps execution observable and recoverable, but coordination cost
rises with step count and fan-out. The architecture localises orchestration in the workflow engine so
that growth is contained rather than spread across application code.

## 7. Storage Growth

Durable storage is provisioned via a persistent volume claim. Growth is driven primarily by dataset
and model versioning and by retained artifacts and logs. The capacity planner's storage resource
class supports forecasting and headroom recommendations.

**Methodology:** track storage consumption over time per data class; feed the series to the capacity
planner; provision against the recommended capacity (projected peak plus headroom).

## 8. Future Capacity

The capacity planner is the platform's forward-looking mechanism. It forecasts CPU, memory, storage,
request volume, model growth and data growth using linear and compound strategies, recommends
provisioned capacity with configurable headroom (default 25%), and detects the step at which a
resource is projected to reach a limit. Future-capacity planning is therefore a built-in, repeatable
computation rather than an ad hoc exercise.

## 9. Scaling Trade-offs (Summary)

| Axis | Mechanism | Primary trade-off |
|------|-----------|-------------------|
| Horizontal | Replicas + autoscaling | Requires stateless request handling and externalised state |
| Vertical | Per-replica resources | Bounded by node size and single-process limits |
| Data | Versioned, immutable snapshots | Storage footprint vs. reproducibility |
| Model | Versioned registry + retention | Registry size vs. rollback/provenance |
| Workflow | Explicit-state engine | Coordination cost grows with complexity |
| Storage | PVC + capacity forecasting | Provisioning vs. cost |

## 10. Limits and Measurement Gaps

The analysis above describes scaling *mechanisms and trade-offs*. Actual scaling limits — maximum
sustainable throughput, the utilisation at which latency degrades, and the point of diminishing
returns for vertical scaling — must be measured under representative load using the methodology in
`benchmark_methodology.md`. No such limits are claimed here.
