"""Deterministic incident lifecycle manager.

Manages the incident lifecycle (open -> investigating -> identified ->
monitoring -> resolved -> closed), records severity, root cause, timeline,
recovery, corrective actions and generates postmortems. Pure Python,
deterministic (time and ids injected), thread-safe. Supports the Observer
pattern so downstream systems (paging, dashboards) can react to transitions.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .observability_models import (
    Clock, IdGenerator, Incident, IncidentSeverity, IncidentStatus, IncidentTimeline,
    TimelineEvent, ValidationError,
)

__all__ = ["IncidentObserver", "IncidentManager", "create_incident_manager"]

IncidentObserver = Union[Callable[[Incident, IncidentStatus], None], Any]

# Allowed forward transitions in the lifecycle.
_TRANSITIONS = {
    IncidentStatus.OPEN: {IncidentStatus.INVESTIGATING, IncidentStatus.IDENTIFIED,
                          IncidentStatus.RESOLVED},
    IncidentStatus.INVESTIGATING: {IncidentStatus.IDENTIFIED, IncidentStatus.MONITORING,
                                   IncidentStatus.RESOLVED},
    IncidentStatus.IDENTIFIED: {IncidentStatus.MONITORING, IncidentStatus.RESOLVED},
    IncidentStatus.MONITORING: {IncidentStatus.RESOLVED, IncidentStatus.INVESTIGATING},
    IncidentStatus.RESOLVED: {IncidentStatus.CLOSED, IncidentStatus.INVESTIGATING},
    IncidentStatus.CLOSED: set(),
}


class IncidentManager:
    """Tracks incidents and their timelines deterministically."""

    def __init__(self, *, clock: Optional[Clock] = None,
                 ids: Optional[IdGenerator] = None) -> None:
        self._clock = clock or Clock()
        self._ids = ids or IdGenerator("inc")
        self._incidents: Dict[str, Incident] = {}
        self._timelines: Dict[str, List[TimelineEvent]] = {}
        self._observers: List[IncidentObserver] = []
        self._lock = threading.RLock()

    # -- observer management ------------------------------------------------ #
    def subscribe(self, observer: IncidentObserver) -> None:
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def unsubscribe(self, observer: IncidentObserver) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def _notify(self, incident: Incident, previous: IncidentStatus) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for observer in observers:
            if callable(observer):
                observer(incident, previous)
            else:
                observer.on_incident(incident, previous)

    # -- lifecycle ---------------------------------------------------------- #
    def open(self, title: str, severity: IncidentSeverity, *,
             services: Sequence[str] = (), timestamp: Optional[float] = None) -> Incident:
        ts = self._clock.now() if timestamp is None else float(timestamp)
        incident = Incident(
            incident_id=self._ids.next_id(), title=title, severity=severity,
            status=IncidentStatus.OPEN, created_at=ts, services=tuple(services),
        )
        with self._lock:
            self._incidents[incident.incident_id] = incident
            self._timelines[incident.incident_id] = [
                TimelineEvent(ts, IncidentStatus.OPEN, f"opened: {title}")]
        self._notify(incident, IncidentStatus.OPEN)
        return incident

    def get(self, incident_id: str) -> Incident:
        with self._lock:
            if incident_id not in self._incidents:
                raise ValidationError(f"unknown incident: {incident_id}")
            return self._incidents[incident_id]

    def transition(self, incident_id: str, status: IncidentStatus, *,
                   message: str = "", root_cause: Optional[str] = None,
                   corrective_action: Optional[str] = None,
                   timestamp: Optional[float] = None) -> Incident:
        status = status if isinstance(status, IncidentStatus) else IncidentStatus(status)
        with self._lock:
            current = self.get(incident_id)
            if status not in _TRANSITIONS[current.status]:
                raise ValidationError(
                    f"illegal transition {current.status.value} -> {status.value}")
            ts = self._clock.now() if timestamp is None else float(timestamp)
            resolved_at = current.resolved_at
            if status is IncidentStatus.RESOLVED and resolved_at is None:
                resolved_at = ts
            actions = current.corrective_actions
            if corrective_action:
                actions = actions + (corrective_action,)
            updated = Incident(
                incident_id=current.incident_id, title=current.title, severity=current.severity,
                status=status, created_at=current.created_at, resolved_at=resolved_at,
                root_cause=root_cause if root_cause is not None else current.root_cause,
                services=current.services, corrective_actions=actions,
            )
            self._incidents[incident_id] = updated
            self._timelines[incident_id].append(
                TimelineEvent(ts, status, message or f"-> {status.value}"))
            previous = current.status
        self._notify(updated, previous)
        return updated

    # Convenience transitions.
    def investigate(self, incident_id: str, **kw: Any) -> Incident:
        return self.transition(incident_id, IncidentStatus.INVESTIGATING, **kw)

    def identify(self, incident_id: str, root_cause: str, **kw: Any) -> Incident:
        return self.transition(incident_id, IncidentStatus.IDENTIFIED, root_cause=root_cause, **kw)

    def monitor(self, incident_id: str, **kw: Any) -> Incident:
        return self.transition(incident_id, IncidentStatus.MONITORING, **kw)

    def resolve(self, incident_id: str, **kw: Any) -> Incident:
        return self.transition(incident_id, IncidentStatus.RESOLVED, **kw)

    def close(self, incident_id: str, **kw: Any) -> Incident:
        return self.transition(incident_id, IncidentStatus.CLOSED, **kw)

    # -- queries ------------------------------------------------------------ #
    def timeline(self, incident_id: str) -> IncidentTimeline:
        with self._lock:
            if incident_id not in self._timelines:
                raise ValidationError(f"unknown incident: {incident_id}")
            return IncidentTimeline(incident_id, tuple(self._timelines[incident_id]))

    def all_incidents(self) -> Tuple[Incident, ...]:
        with self._lock:
            return tuple(self._incidents[k] for k in sorted(self._incidents))

    def active(self) -> Tuple[Incident, ...]:
        return tuple(i for i in self.all_incidents() if not i.is_resolved)

    def by_severity(self, severity: IncidentSeverity) -> Tuple[Incident, ...]:
        return tuple(i for i in self.all_incidents() if i.severity is severity)

    def recovery_time(self, incident_id: str) -> Optional[float]:
        return self.get(incident_id).duration

    def postmortem(self, incident_id: str) -> Dict[str, Any]:
        incident = self.get(incident_id)
        timeline = self.timeline(incident_id)
        if not incident.is_resolved:
            raise ValidationError("postmortem requires a resolved incident")
        return {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "root_cause": incident.root_cause,
            "time_to_recovery": incident.duration,
            "services": list(incident.services),
            "corrective_actions": list(incident.corrective_actions),
            "timeline": timeline.to_dict()["events"],
        }

    def statistics(self) -> Dict[str, Any]:
        incidents = self.all_incidents()
        resolved = [i for i in incidents if i.is_resolved and i.duration is not None]
        durations = [i.duration for i in resolved]
        by_sev = {s.value: len(self.by_severity(s)) for s in IncidentSeverity}
        return {
            "total": len(incidents),
            "active": len(self.active()),
            "resolved": len(resolved),
            "by_severity": by_sev,
            "mean_time_to_recovery": round(sum(durations) / len(durations), 6) if durations else 0.0,
        }


def create_incident_manager(*, clock: Optional[Clock] = None) -> IncidentManager:
    return IncidentManager(clock=clock)