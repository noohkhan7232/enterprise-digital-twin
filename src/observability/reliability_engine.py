"""Deterministic reliability computation engine.

Computes availability, MTBF, MTTR, success and failure rates, a composite
reliability score and an operational-risk estimate from request outcomes and
incident windows. Pure Python + NumPy, deterministic, thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .observability_models import Incident, ReliabilityScore, ValidationError, clamp

__all__ = ["FailureWindow", "ReliabilityEngine", "create_reliability_engine"]


class FailureWindow:
    """A single failure/outage interval with a start and an end time."""

    __slots__ = ("start", "end")

    def __init__(self, start: float, end: float) -> None:
        if end < start:
            raise ValidationError("FailureWindow.end must be >= start")
        self.start = float(start)
        self.end = float(end)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)


class ReliabilityEngine:
    """Computes reliability metrics from outcomes and outage windows."""

    def __init__(self, *, score_weights: Optional[Mapping[str, float]] = None) -> None:
        # Composite score = weighted blend of availability and success rate,
        # penalised by operational risk. Weights are injectable.
        default = {"availability": 0.5, "success_rate": 0.5}
        self._weights = dict(score_weights or default)
        total = sum(self._weights.values()) or 1.0
        self._weights = {k: v / total for k, v in self._weights.items()}
        self._successes = 0
        self._failures = 0
        self._windows: List[FailureWindow] = []
        self._observation_period = 0.0
        self._lock = threading.RLock()

    # -- recording ---------------------------------------------------------- #
    def record_outcome(self, success: bool, *, count: int = 1) -> None:
        if count < 0:
            raise ValidationError("count must be non-negative")
        with self._lock:
            if success:
                self._successes += count
            else:
                self._failures += count

    def record_failure_window(self, start: float, end: float) -> FailureWindow:
        window = FailureWindow(start, end)
        with self._lock:
            self._windows.append(window)
        return window

    def set_observation_period(self, seconds: float) -> None:
        if seconds < 0:
            raise ValidationError("observation period must be non-negative")
        with self._lock:
            self._observation_period = float(seconds)

    # -- derived metrics ---------------------------------------------------- #
    @property
    def total_requests(self) -> int:
        return self._successes + self._failures

    def success_rate(self) -> float:
        total = self.total_requests
        return round(self._successes / total, 6) if total else 1.0

    def failure_rate(self) -> float:
        total = self.total_requests
        return round(self._failures / total, 6) if total else 0.0

    def downtime(self) -> float:
        return round(sum(w.duration for w in self._windows), 6)

    def availability(self) -> float:
        period = self._observation_period
        if period <= 0:
            # Fall back to request-based availability if no period is set.
            return self.success_rate()
        uptime = max(0.0, period - self.downtime())
        return round(clamp(uptime / period), 6)

    def mtbf(self) -> float:
        """Mean time between failures over the observation period."""
        failures = len(self._windows)
        if failures == 0:
            return round(self._observation_period, 6) if self._observation_period > 0 else 0.0
        uptime = max(0.0, self._observation_period - self.downtime())
        return round(uptime / failures, 6)

    def mttr(self) -> float:
        """Mean time to recovery across recorded outage windows."""
        if not self._windows:
            return 0.0
        return round(sum(w.duration for w in self._windows) / len(self._windows), 6)

    def operational_risk(self) -> float:
        """Risk in [0, 1]: higher means more operationally risky."""
        avail_gap = 1.0 - self.availability()
        failure = self.failure_rate()
        # Recovery friction: long MTTR relative to MTBF raises risk.
        mtbf = self.mtbf()
        recovery_friction = 0.0 if mtbf <= 0 else clamp(self.mttr() / (self.mttr() + mtbf))
        risk = 0.4 * avail_gap + 0.4 * failure + 0.2 * recovery_friction
        return round(clamp(risk), 6)

    def reliability_score(self) -> ReliabilityScore:
        availability = self.availability()
        success = self.success_rate()
        blend = (self._weights.get("availability", 0.5) * availability +
                 self._weights.get("success_rate", 0.5) * success)
        risk = self.operational_risk()
        score = clamp(blend * (1.0 - 0.5 * risk))
        return ReliabilityScore(
            availability=availability, mtbf=self.mtbf(), mttr=self.mttr(),
            success_rate=success, failure_rate=self.failure_rate(),
            score=round(score, 6), operational_risk=risk,
        )

    # -- helpers ------------------------------------------------------------ #
    def from_incidents(self, incidents: Sequence[Incident]) -> int:
        """Register resolved incidents as failure windows."""
        added = 0
        for inc in incidents:
            if inc.resolved_at is not None:
                self.record_failure_window(inc.created_at, inc.resolved_at)
                added += 1
        return added

    def export(self) -> Dict[str, Any]:
        return self.reliability_score().to_dict()


def create_reliability_engine(
        *, score_weights: Optional[Mapping[str, float]] = None) -> ReliabilityEngine:
    return ReliabilityEngine(score_weights=score_weights)