"""Tests for the incident manager."""

from __future__ import annotations

import json
import threading

import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from observability.incident_manager import IncidentManager, create_incident_manager  # noqa: E402
from observability.observability_models import (  # noqa: E402
    Clock, IncidentSeverity, IncidentStatus, ValidationError,
)


def manager():
    return IncidentManager(clock=Clock())


def open_incident(m, sev=IncidentSeverity.SEV1):
    return m.open("outage", sev, services=["db"])


def test_open_creates_incident():
    m = manager()
    inc = open_incident(m)
    assert inc.status is IncidentStatus.OPEN and inc.incident_id == "inc-00000001"


def test_open_records_services():
    m = manager()
    inc = open_incident(m)
    assert inc.services == ("db",)


def test_get_incident():
    m = manager()
    inc = open_incident(m)
    assert m.get(inc.incident_id).title == "outage"


def test_get_unknown_raises():
    with pytest.raises(ValidationError):
        manager().get("nope")


def test_investigate():
    m = manager()
    inc = open_incident(m)
    updated = m.investigate(inc.incident_id)
    assert updated.status is IncidentStatus.INVESTIGATING


def test_identify_sets_root_cause():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    updated = m.identify(inc.incident_id, root_cause="bad deploy")
    assert updated.root_cause == "bad deploy"


def test_monitor():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    updated = m.monitor(inc.incident_id)
    assert updated.status is IncidentStatus.MONITORING


def test_resolve_sets_resolved_at():
    m = manager()
    inc = open_incident(m)
    resolved = m.resolve(inc.incident_id)
    assert resolved.is_resolved and resolved.resolved_at is not None


def test_resolve_duration_positive():
    m = manager()
    inc = open_incident(m)
    resolved = m.resolve(inc.incident_id)
    assert resolved.duration > 0


def test_close():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    closed = m.close(inc.incident_id)
    assert closed.status is IncidentStatus.CLOSED


def test_corrective_action_accumulates():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id, corrective_action="action1")
    resolved = m.resolve(inc.incident_id, corrective_action="action2")
    assert "action1" in resolved.corrective_actions and "action2" in resolved.corrective_actions


def test_illegal_transition_raises():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    with pytest.raises(ValidationError):
        m.transition(inc.incident_id, IncidentStatus.OPEN)


def test_illegal_open_to_monitoring():
    m = manager()
    inc = open_incident(m)
    with pytest.raises(ValidationError):
        m.transition(inc.incident_id, IncidentStatus.MONITORING)


def test_open_directly_to_resolved():
    m = manager()
    inc = open_incident(m)
    resolved = m.resolve(inc.incident_id)
    assert resolved.is_resolved


def test_reopen_from_monitoring():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    m.monitor(inc.incident_id)
    reopened = m.investigate(inc.incident_id)
    assert reopened.status is IncidentStatus.INVESTIGATING


def test_timeline_has_events():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    m.resolve(inc.incident_id)
    assert len(m.timeline(inc.incident_id).events) == 3


def test_timeline_ordered():
    m = manager()
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    events = m.timeline(inc.incident_id).events
    assert events[0].timestamp <= events[1].timestamp


def test_timeline_unknown_raises():
    with pytest.raises(ValidationError):
        manager().timeline("nope")


def test_all_incidents_sorted():
    m = manager()
    open_incident(m)
    open_incident(m)
    ids = [i.incident_id for i in m.all_incidents()]
    assert ids == sorted(ids)


def test_active_excludes_resolved():
    m = manager()
    inc1 = open_incident(m)
    open_incident(m)
    m.resolve(inc1.incident_id)
    assert len(m.active()) == 1


def test_by_severity():
    m = manager()
    open_incident(m, IncidentSeverity.SEV1)
    open_incident(m, IncidentSeverity.SEV3)
    assert len(m.by_severity(IncidentSeverity.SEV1)) == 1


def test_recovery_time():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    assert m.recovery_time(inc.incident_id) is not None


def test_recovery_time_unresolved():
    m = manager()
    inc = open_incident(m)
    assert m.recovery_time(inc.incident_id) is None


def test_postmortem():
    m = manager()
    inc = open_incident(m)
    m.identify(inc.incident_id, root_cause="rc")
    m.resolve(inc.incident_id, corrective_action="fix")
    pm = m.postmortem(inc.incident_id)
    assert pm["root_cause"] == "rc" and "fix" in pm["corrective_actions"]


def test_postmortem_requires_resolved():
    m = manager()
    inc = open_incident(m)
    with pytest.raises(ValidationError):
        m.postmortem(inc.incident_id)


def test_postmortem_has_timeline():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    assert len(m.postmortem(inc.incident_id)["timeline"]) == 2


def test_observer_fires():
    m = manager()
    events = []
    m.subscribe(lambda inc, prev: events.append(inc.status.value))
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    assert events == ["OPEN", "INVESTIGATING"]


def test_observer_unsubscribe():
    m = manager()
    events = []
    obs = lambda inc, prev: events.append(1)
    m.subscribe(obs)
    m.unsubscribe(obs)
    open_incident(m)
    assert events == []


def test_observer_receives_previous():
    m = manager()
    transitions = []
    m.subscribe(lambda inc, prev: transitions.append((prev.value, inc.status.value)))
    inc = open_incident(m)
    m.investigate(inc.incident_id)
    assert ("OPEN", "INVESTIGATING") in transitions


def test_statistics():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    open_incident(m)
    stats = m.statistics()
    assert stats["total"] == 2 and stats["active"] == 1 and stats["resolved"] == 1


def test_statistics_mttr():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    assert m.statistics()["mean_time_to_recovery"] > 0


def test_statistics_by_severity():
    m = manager()
    open_incident(m, IncidentSeverity.SEV2)
    assert m.statistics()["by_severity"]["SEV2"] == 1


def test_factory():
    assert isinstance(create_incident_manager(), IncidentManager)


def test_postmortem_json_serializable():
    m = manager()
    inc = open_incident(m)
    m.resolve(inc.incident_id)
    assert json.dumps(m.postmortem(inc.incident_id))


def test_explicit_timestamp_open():
    m = manager()
    inc = m.open("x", IncidentSeverity.SEV1, timestamp=500.0)
    assert inc.created_at == 500.0


def test_thread_safety():
    m = manager()

    def worker():
        for _ in range(20):
            m.open("x", IncidentSeverity.SEV3)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(m.all_incidents()) == 100


def test_object_observer():
    m = manager()

    class Obs:
        def __init__(self):
            self.count = 0

        def on_incident(self, inc, prev):
            self.count += 1

    obs = Obs()
    m.subscribe(obs)
    open_incident(m)
    assert obs.count == 1