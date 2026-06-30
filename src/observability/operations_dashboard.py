"""Operations dashboard: composition of all observability subsystems.

Aggregates metrics, tracing, reliability, incidents, capacity and production
readiness into a single :class:`OperationsSnapshot` and renders a deterministic
executive summary. Pure composition — it imports the other engines' value
objects and is handed their outputs; it owns no global state. Deterministic.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .observability_models import (
    CapacityForecast, Clock, Incident, OperationsSnapshot, ReliabilityScore,
)

__all__ = ["OperationsDashboard", "create_operations_dashboard"]


class OperationsDashboard:
    """Builds operations snapshots and executive summaries by composition."""

    def __init__(self, *, clock: Optional[Clock] = None) -> None:
        self._clock = clock or Clock()
        self._snapshots = 0
        self._lock = threading.RLock()

    def build_snapshot(
        self, *,
        metrics_summary: Optional[Mapping[str, Any]] = None,
        reliability: Optional[ReliabilityScore] = None,
        active_incidents: Sequence[Incident] = (),
        slo_compliance: Optional[Mapping[str, Any]] = None,
        capacity: Sequence[CapacityForecast] = (),
        readiness_score: float = 0.0,
    ) -> OperationsSnapshot:
        with self._lock:
            self._snapshots += 1
        return OperationsSnapshot(
            generated_at=self._clock.now(),
            metrics_summary=tuple(sorted((str(k), v) for k, v in dict(metrics_summary or {}).items())),
            reliability=reliability,
            active_incidents=tuple(active_incidents),
            slo_compliance=tuple(sorted((str(k), v) for k, v in dict(slo_compliance or {}).items())),
            capacity=tuple(capacity),
            readiness_score=readiness_score,
        )

    def executive_summary(self, snapshot: OperationsSnapshot) -> str:
        lines: List[str] = ["Executive Operations Summary", "=" * 28]
        if snapshot.reliability is not None:
            r = snapshot.reliability
            lines.append(f"Reliability score: {r.score:.3f} "
                         f"(availability {r.availability:.3f}, risk {r.operational_risk:.3f})")
        lines.append(f"Production readiness: {snapshot.readiness_score:.1f}/100")
        active = snapshot.active_incidents
        if active:
            worst = min(active, key=lambda i: i.severity.value)
            lines.append(f"Active incidents: {len(active)} (most severe {worst.severity.value})")
        else:
            lines.append("Active incidents: none")
        compliance = dict(snapshot.slo_compliance)
        if "compliance_rate" in compliance:
            lines.append(f"SLO compliance: {float(compliance['compliance_rate']):.1%}")
        if snapshot.capacity:
            exhausting = [c for c in snapshot.capacity if c.exhaustion_step is not None]
            if exhausting:
                names = ", ".join(c.resource.value for c in exhausting)
                lines.append(f"Capacity warnings: {names} projected to reach limit within horizon")
            else:
                lines.append("Capacity: within projected limits")
        # Recommendation, graded deterministically.
        recommendation = self._recommendation(snapshot)
        lines.append(f"Recommendation: {recommendation}")
        return "\n".join(lines)

    def _recommendation(self, snapshot: OperationsSnapshot) -> str:
        sev1 = any(i.severity.value == "SEV1" for i in snapshot.active_incidents)
        if sev1:
            return "engage incident response; hold all releases"
        if snapshot.reliability is not None and snapshot.reliability.operational_risk > 0.25:
            return "investigate elevated operational risk before next release"
        if snapshot.readiness_score < 80.0:
            return "address readiness gaps before promoting to production"
        return "operating within targets; no action required"

    def render(self, snapshot: OperationsSnapshot) -> Dict[str, Any]:
        data = snapshot.to_dict()
        data["executive_summary"] = self.executive_summary(snapshot)
        return data

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {"snapshots": self._snapshots}


def create_operations_dashboard(*, clock: Optional[Clock] = None) -> OperationsDashboard:
    return OperationsDashboard(clock=clock)