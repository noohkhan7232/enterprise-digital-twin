"""Enterprise test suite for the Week 10 Phase 3 Enterprise Event Bus.

Standard pytest (parametrize / raises / approx only - no fixtures), with a
bootstrap that resolves the module in the repository layout and in isolation.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import sys
import threading
from dataclasses import FrozenInstanceError

import pytest

# --- import bootstrap -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "events"))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:
    eb = importlib.import_module("src.events.enterprise_event_bus")
except ModuleNotFoundError:
    eb = importlib.import_module("enterprise_event_bus")

globals().update({name: getattr(eb, name) for name in eb.__all__})


# ---------------------------------------------------------------------------
# Helpers (no fixtures)
# ---------------------------------------------------------------------------
def ev(etype=None, topic=None, priority=None, payload=None, source="system",
       corr="", tags=None, ts=0.0, seq=0, event_id="e"):
    e = EnterpriseEvent.create(
        etype or EventType.SYSTEM_EVENT, payload, topic=topic,
        priority=priority or EventPriority.NORMAL, source=source,
        correlation_id=corr, tags=tags, event_id=event_id)
    return dataclasses.replace(e, sequence=seq, timestamp=ts)


def collector():
    received = []

    def handler(event):
        received.append(event)

    return received, handler


def failing_handler(event):
    raise RuntimeError("boom")


ALL_EVENT_TYPES = list(EventType)
ALL_PRIORITIES = list(EventPriority)
ALL_DELIVERY_MODES = list(DeliveryMode)
ALL_RECOVERY = list(RecoveryStatus)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enum_cls,value", (
    [(EventType, e.value) for e in EventType]
    + [(EventPriority, e.value) for e in EventPriority]
    + [(DeliveryMode, e.value) for e in DeliveryMode]
    + [(RecoveryStatus, e.value) for e in RecoveryStatus]
))
def test_enum_coerce(enum_cls, value):
    assert enum_cls.coerce(value).value == value


@pytest.mark.parametrize("enum_cls", [EventType, EventPriority, DeliveryMode, RecoveryStatus])
def test_enum_coerce_invalid(enum_cls):
    with pytest.raises(EventValidationError):
        enum_cls.coerce("__nope__")


@pytest.mark.parametrize("p,level", [
    (EventPriority.LOW, 0), (EventPriority.NORMAL, 1),
    (EventPriority.HIGH, 2), (EventPriority.CRITICAL, 3),
])
def test_priority_levels(p, level):
    assert p.level == level


@pytest.mark.parametrize("etype", ALL_EVENT_TYPES)
def test_event_type_default_topic(etype):
    assert etype.default_topic == etype.value.replace("_", ".")


# ---------------------------------------------------------------------------
# EventMetadata
# ---------------------------------------------------------------------------
def test_metadata_roundtrip():
    m = EventMetadata(source="s", correlation_id="c", trace_id="t",
                      causation_id="x", tags_json='{"k":1}')
    assert EventMetadata.from_dict(m.to_dict()).to_dict() == m.to_dict()


def test_metadata_tags():
    m = EventMetadata(tags_json='{"a":1}')
    assert m.tags == {"a": 1}


# ---------------------------------------------------------------------------
# EnterpriseEvent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("etype", ALL_EVENT_TYPES)
def test_event_create_default_topic(etype):
    e = EnterpriseEvent.create(etype, {"x": 1})
    assert e.topic == etype.default_topic
    assert e.event_type is etype


@pytest.mark.parametrize("etype", ALL_EVENT_TYPES)
@pytest.mark.parametrize("priority", ALL_PRIORITIES)
def test_event_roundtrip(etype, priority):
    e = ev(etype, priority=priority, payload={"a": 1}, corr="c", seq=3, ts=2.0)
    assert EnterpriseEvent.from_dict(e.to_dict()).to_dict() == e.to_dict()


def test_event_payload_hash_consistent():
    a = EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"a": 1, "b": 2})
    b = EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"b": 2, "a": 1})
    assert a.payload_hash == b.payload_hash


def test_event_empty_topic_rejected():
    with pytest.raises(EventValidationError):
        EnterpriseEvent(EventType.SYSTEM_EVENT, "")


def test_event_payload_property():
    e = EnterpriseEvent.create(EventType.CUSTOM_EVENT, {"k": "v"})
    assert e.payload == {"k": "v"}


def test_event_is_stamped():
    assert EnterpriseEvent.create(EventType.SYSTEM_EVENT).is_stamped is False
    assert ev(seq=0).is_stamped is True


def test_event_correlation_property():
    e = EnterpriseEvent.create(EventType.SYSTEM_EVENT, correlation_id="abc")
    assert e.correlation_id == "abc"


# ---------------------------------------------------------------------------
# EventEnvelope / EventResult
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ALL_DELIVERY_MODES)
def test_envelope_roundtrip(mode):
    env = EventEnvelope(ev(), delivery_mode=mode, target="t", attempt=2, enqueued_at=1.0)
    assert EventEnvelope.from_dict(env.to_dict()).to_dict() == env.to_dict()


def test_envelope_requires_event():
    with pytest.raises(EventValidationError):
        EventEnvelope("not-an-event")


@pytest.mark.parametrize("delivered", [True, False])
def test_event_result_roundtrip(delivered):
    r = EventResult("e", "s", delivered, "err" if not delivered else "", 0.5, 2, not delivered)
    assert EventResult.from_dict(r.to_dict()).to_dict() == r.to_dict()


# ---------------------------------------------------------------------------
# Topic matching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pattern,topic,expected", [
    ("workflow.started", "workflow.started", True),
    ("workflow.started", "workflow.completed", False),
    ("workflow.*", "workflow.started", True),
    ("workflow.*", "workflow.started.x", False),
    ("workflow.#", "workflow.started.x", True),
    ("workflow.#", "workflow.started", True),
    ("#", "anything.at.all", True),
    ("*", "single", True),
    ("*", "two.segments", False),
    ("risk.#", "risk.threshold.exceeded", True),
    ("a.*.c", "a.b.c", True),
    ("a.*.c", "a.b.d", False),
])
def test_topic_matches(pattern, topic, expected):
    assert eb._topic_matches(pattern, topic) is expected


# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------
def test_filter_topic():
    f = EventFilter(topic_pattern="workflow.*")
    assert f.matches(ev(EventType.WORKFLOW_STARTED)) is True
    assert f.matches(ev(EventType.AUDIT_EVENT)) is False


@pytest.mark.parametrize("min_p,evt_p,expected", [
    (EventPriority.LOW, EventPriority.LOW, True),
    (EventPriority.HIGH, EventPriority.NORMAL, False),
    (EventPriority.HIGH, EventPriority.CRITICAL, True),
])
def test_filter_priority(min_p, evt_p, expected):
    f = EventFilter(min_priority=min_p)
    assert f.matches(ev(priority=evt_p)) is expected


def test_filter_source():
    f = EventFilter(source="engine")
    assert f.matches(ev(source="engine")) is True
    assert f.matches(ev(source="other")) is False


def test_filter_correlation():
    f = EventFilter(correlation_id="c1")
    assert f.matches(ev(corr="c1")) is True
    assert f.matches(ev(corr="c2")) is False


def test_filter_event_type():
    f = EventFilter(event_type=EventType.AUDIT_EVENT)
    assert f.matches(ev(EventType.AUDIT_EVENT)) is True
    assert f.matches(ev(EventType.SYSTEM_EVENT)) is False


def test_filter_time_range():
    f = EventFilter(time_start=5.0, time_end=10.0)
    assert f.matches(ev(ts=7.0)) is True
    assert f.matches(ev(ts=2.0)) is False
    assert f.matches(ev(ts=12.0)) is False


def test_filter_workflow_id_from_payload():
    f = EventFilter(workflow_id="WF-1")
    assert f.matches(ev(payload={"workflow_id": "WF-1"})) is True
    assert f.matches(ev(payload={"workflow_id": "WF-2"})) is False


def test_filter_process_id_from_payload():
    f = EventFilter(process_id="P-1")
    assert f.matches(ev(payload={"process_id": "P-1"})) is True


def test_filter_metadata_key_value():
    f = EventFilter(metadata_key="region", metadata_value="emea")
    assert f.matches(ev(tags={"region": "emea"})) is True
    assert f.matches(ev(tags={"region": "us"})) is False
    assert f.matches(ev(tags={})) is False


def test_filter_combined():
    f = EventFilter(topic_pattern="workflow.*", min_priority=EventPriority.HIGH)
    assert f.matches(ev(EventType.WORKFLOW_FAILED, priority=EventPriority.CRITICAL)) is True
    assert f.matches(ev(EventType.WORKFLOW_FAILED, priority=EventPriority.LOW)) is False


def test_filter_roundtrip():
    f = EventFilter(topic_pattern="a.*", min_priority=EventPriority.HIGH, source="s",
                    correlation_id="c", workflow_id="w", process_id="p",
                    event_type=EventType.AUDIT_EVENT, time_start=1.0, time_end=2.0,
                    metadata_key="k", metadata_value="v")
    assert EventFilter.from_dict(f.to_dict()).to_dict() == f.to_dict()


# ---------------------------------------------------------------------------
# EventSubscription
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ALL_DELIVERY_MODES)
def test_subscription_roundtrip(mode):
    sub = EventSubscription("sub-1", "subscriber", "topic.*", mode, 5,
                            EventPriority.NORMAL, True, EventFilter(source="s"), 0)
    assert EventSubscription.from_dict(sub.to_dict()).to_dict() == sub.to_dict()


def test_subscription_validation():
    with pytest.raises(SubscriptionError):
        EventSubscription("", "s")
    with pytest.raises(SubscriptionError):
        EventSubscription("id", "")


def test_subscription_matches_broadcast():
    sub = EventSubscription("s1", "sub", delivery_mode=DeliveryMode.BROADCAST)
    assert sub.matches(ev(EventType.AUDIT_EVENT), None) is True


def test_subscription_matches_direct():
    sub = EventSubscription("s1", "target-a", delivery_mode=DeliveryMode.DIRECT)
    assert sub.matches(ev(), "target-a") is True
    assert sub.matches(ev(), "target-b") is False
    assert sub.matches(ev(), None) is False


def test_subscription_matches_topic():
    sub = EventSubscription("s1", "sub", "workflow.*", DeliveryMode.TOPIC)
    assert sub.matches(ev(EventType.WORKFLOW_STARTED), None) is True
    assert sub.matches(ev(EventType.AUDIT_EVENT), None) is False


def test_subscription_min_priority_gate():
    sub = EventSubscription("s1", "sub", "#", DeliveryMode.TOPIC,
                            min_priority=EventPriority.HIGH)
    assert sub.matches(ev(priority=EventPriority.CRITICAL), None) is True
    assert sub.matches(ev(priority=EventPriority.LOW), None) is False


def test_subscription_filtered():
    sub = EventSubscription("s1", "sub", "#", DeliveryMode.FILTERED,
                            event_filter=EventFilter(source="engine"))
    assert sub.matches(ev(source="engine"), None) is True
    assert sub.matches(ev(source="other"), None) is False


# ---------------------------------------------------------------------------
# EventHistory
# ---------------------------------------------------------------------------
def test_history_append_immutable():
    h0 = EventHistory()
    h1 = h0.append(ev(seq=0))
    assert len(h0) == 0 and len(h1) == 1


def test_history_max_history_truncates():
    h = EventHistory()
    for i in range(10):
        h = h.append(ev(seq=i), max_history=3)
    assert len(h) == 3
    assert [e.sequence for e in h] == [7, 8, 9]


def test_history_next_sequence():
    h = EventHistory().append(ev(seq=4))
    assert h.next_sequence == 5


def test_history_by_topic():
    h = EventHistory((ev(EventType.WORKFLOW_STARTED), ev(EventType.AUDIT_EVENT)))
    assert len(h.by_topic("workflow.*")) == 1


def test_history_by_correlation():
    h = EventHistory((ev(corr="c1"), ev(corr="c2")))
    assert len(h.by_correlation("c1")) == 1


def test_history_by_time():
    h = EventHistory((ev(ts=1.0), ev(ts=5.0), ev(ts=9.0)))
    assert len(h.by_time(2.0, 8.0)) == 1


def test_history_by_type():
    h = EventHistory((ev(EventType.AUDIT_EVENT), ev(EventType.SYSTEM_EVENT)))
    assert len(h.by_type(EventType.AUDIT_EVENT)) == 1


def test_history_last():
    h = EventHistory(tuple(ev(seq=i) for i in range(5)))
    assert [e.sequence for e in h.last(2)] == [3, 4]
    assert h.last(0) == ()


def test_history_last_negative():
    with pytest.raises(EventValidationError):
        EventHistory().last(-1)


def test_history_filter():
    h = EventHistory((ev(source="a"), ev(source="b")))
    assert len(h.filter(EventFilter(source="a"))) == 1


def test_history_roundtrip():
    h = EventHistory((ev(seq=0), ev(seq=1)))
    assert EventHistory.from_dict(h.to_dict()).to_dict() == h.to_dict()


# ---------------------------------------------------------------------------
# DeadLetterEvent / EventReplay / EventBatch / EventStatistics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", ALL_RECOVERY)
def test_dead_letter_roundtrip(status):
    dle = DeadLetterEvent(ev(), "sub", 2, "reason", status, 1.0, 2.0)
    assert DeadLetterEvent.from_dict(dle.to_dict()).to_dict() == dle.to_dict()


def test_dead_letter_requires_event():
    with pytest.raises(DeadLetterError):
        DeadLetterEvent("not-an-event", "s", 1, "r")


def test_replay_roundtrip():
    r = EventReplay("r1", "topic", "workflow.*", ("e1", "e2"), 2, 5.0)
    assert EventReplay.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_batch_roundtrip():
    b = EventBatch("b1", (ev(seq=0), ev(seq=1)), 3.0)
    assert EventBatch.from_dict(b.to_dict()).to_dict() == b.to_dict()
    assert len(b) == 2


def test_batch_validation():
    with pytest.raises(EventValidationError):
        EventBatch("", (ev(),))
    with pytest.raises(EventValidationError):
        EventBatch("b", ("not-an-event",))


def test_statistics_roundtrip():
    s = EventStatistics(10, 8, 2, 1, 3, 2, 1, 0.5, 0.2, 0.8, '{"t":3}')
    assert EventStatistics.from_dict(s.to_dict()).to_dict() == s.to_dict()
    assert s.topic_counts == {"t": 3}


# ---------------------------------------------------------------------------
# Bus: subscribe / publish basics
# ---------------------------------------------------------------------------
def test_publish_requires_event():
    with pytest.raises(PublishError):
        eb.create_default_event_bus().publish("not-an-event")


def test_publish_delivers_to_subscriber():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    results = bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"x": 1}))
    assert len(received) == 1
    assert results[0].delivered is True


def test_publish_stamps_event():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    e = received[0]
    assert e.sequence == 0
    assert e.event_id.startswith("evt-")
    assert e.metadata.trace_id and e.metadata.correlation_id
    assert e.payload_hash


def test_publish_history_grows():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))
    assert len(bus.history()) == 2
    assert [e.sequence for e in bus.history()] == [0, 1]


def test_publish_no_subscribers_unrouted():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert bus.statistics().unrouted == 1


def test_subscribe_returns_subscription():
    bus = eb.create_default_event_bus()
    sub = bus.subscribe(lambda e: None, topic_pattern="#", subscriber_id="x")
    assert sub.subscriber_id == "x"
    assert bus.subscriber_count() == 1


def test_subscribe_non_callable():
    with pytest.raises(SubscriptionError):
        eb.create_default_event_bus().subscribe("not-callable")


def test_unsubscribe():
    bus = eb.create_default_event_bus()
    sub = bus.subscribe(lambda e: None)
    bus.unsubscribe(sub.subscription_id)
    assert bus.subscriber_count() == 0


def test_unsubscribe_unknown():
    with pytest.raises(SubscriptionError):
        eb.create_default_event_bus().unsubscribe("ghost")


def test_clear_subscriptions():
    bus = eb.create_default_event_bus()
    bus.subscribe(lambda e: None)
    bus.subscribe(lambda e: None)
    bus.clear()
    assert bus.subscriber_count() == 0


def test_subscriptions_sorted():
    bus = eb.create_default_event_bus()
    bus.subscribe(lambda e: None)
    bus.subscribe(lambda e: None)
    subs = bus.subscriptions()
    assert [s.subscription_id for s in subs] == sorted(s.subscription_id for s in subs)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_routing_topic_exact():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="workflow.started")
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_COMPLETED))
    assert len(received) == 1


def test_routing_wildcard():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="workflow.*")
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))
    assert len(received) == 1


def test_routing_broadcast():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, delivery_mode=DeliveryMode.BROADCAST)
    bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    assert len(received) == 2


def test_routing_direct():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, subscriber_id="target-a", delivery_mode=DeliveryMode.DIRECT)
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT), target="target-a")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT), target="target-b")
    assert len(received) == 1


def test_routing_filtered():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, delivery_mode=DeliveryMode.FILTERED,
                  event_filter=EventFilter(source="engine"))
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, source="engine"))
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, source="other"))
    assert len(received) == 1


def test_routing_priority_gate():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#", min_priority=EventPriority.HIGH)
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, priority=EventPriority.LOW))
    bus.publish(EnterpriseEvent.create(EventType.RISK_THRESHOLD_EXCEEDED,
                                       priority=EventPriority.CRITICAL))
    assert len(received) == 1


def test_routing_deterministic_priority_order():
    bus = eb.create_default_event_bus()
    order = []
    bus.subscribe(lambda e: order.append("low"), topic_pattern="#", priority=1)
    bus.subscribe(lambda e: order.append("high"), topic_pattern="#", priority=10)
    bus.subscribe(lambda e: order.append("mid"), topic_pattern="#", priority=5)
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert order == ["high", "mid", "low"]


def test_routing_same_priority_registration_order():
    bus = eb.create_default_event_bus()
    order = []
    bus.subscribe(lambda e: order.append("first"), topic_pattern="#", priority=0)
    bus.subscribe(lambda e: order.append("second"), topic_pattern="#", priority=0)
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# once / one-time subscriptions
# ---------------------------------------------------------------------------
def test_once_fires_then_removed():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.once(handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert len(received) == 1
    assert bus.subscriber_count() == 0


# ---------------------------------------------------------------------------
# Failure handling & DLQ
# ---------------------------------------------------------------------------
def test_failing_handler_dead_letters():
    bus = eb.create_default_event_bus()
    bus.subscribe(failing_handler, topic_pattern="#")
    results = bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert results[0].delivered is False
    assert results[0].dead_lettered is True
    assert len(bus.dead_letter_queue()) == 1


def test_dlq_records_reason():
    bus = eb.create_default_event_bus()
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    dead = bus.dead_letter_queue()[0]
    assert "boom" in dead.failure_reason
    assert dead.recovery_status is RecoveryStatus.PENDING


def test_retries_recorded_in_attempts():
    bus = eb.EnterpriseEventBus(clock=eb.LogicalClock(), max_retries=3)
    bus.subscribe(failing_handler, topic_pattern="#")
    results = bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert results[0].attempts == 3
    assert bus.dead_letter_queue()[0].retry_count == 3


def test_dead_letter_disabled():
    bus = eb.EnterpriseEventBus(dead_letter_enabled=False)
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert len(bus.dead_letter_queue()) == 0


def test_recover_dead_letters_success():
    bus = eb.create_default_event_bus()
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    recovered = bus.recover_dead_letters(lambda e: None)
    assert recovered == 1
    assert len(bus.dead_letter_queue()) == 0


def test_recover_dead_letters_failure_marks_failed():
    bus = eb.create_default_event_bus()
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    recovered = bus.recover_dead_letters(failing_handler)
    assert recovered == 0
    assert bus.dead_letter_queue()[0].recovery_status is RecoveryStatus.FAILED


def test_recover_non_callable():
    with pytest.raises(DeadLetterError):
        eb.create_default_event_bus().recover_dead_letters("nope")


def test_clear_dead_letters():
    bus = eb.create_default_event_bus()
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    bus.clear_dead_letters()
    assert len(bus.dead_letter_queue()) == 0


def test_max_dead_letters_caps():
    bus = eb.EnterpriseEventBus(max_dead_letters=2)
    bus.subscribe(failing_handler, topic_pattern="#")
    for _ in range(5):
        bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert len(bus.dead_letter_queue()) == 2


# ---------------------------------------------------------------------------
# Batch publishing
# ---------------------------------------------------------------------------
def test_publish_batch():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    batch = EventBatch("b1", (
        EnterpriseEvent.create(EventType.SYSTEM_EVENT),
        EnterpriseEvent.create(EventType.AUDIT_EVENT),
        EnterpriseEvent.create(EventType.WORKFLOW_STARTED),
    ))
    results = bus.publish_batch(batch)
    assert len(received) == 3
    assert len(results) == 3
    assert bus.statistics().published == 3


def test_publish_batch_requires_batch():
    with pytest.raises(PublishError):
        eb.create_default_event_bus().publish_batch("not-a-batch")


def test_publish_batch_order():
    bus = eb.create_default_event_bus()
    seqs = []
    bus.subscribe(lambda e: seqs.append(e.sequence), topic_pattern="#")
    batch = EventBatch("b", tuple(
        EnterpriseEvent.create(EventType.SYSTEM_EVENT) for _ in range(4)))
    bus.publish_batch(batch)
    assert seqs == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _seed(bus, n=5, etype=EventType.SYSTEM_EVENT, corr="c"):
    for i in range(n):
        bus.publish(EnterpriseEvent.create(etype, {"i": i}, correlation_id=corr))


def test_replay_last_n():
    bus = eb.create_default_event_bus()
    _seed(bus, 5)
    sink, handler = collector()
    replay = bus.replay_last_n(2, handler=handler)
    assert replay.count == 2
    assert len(sink) == 2
    assert bus.statistics().replayed == 2


def test_replay_by_topic():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))
    sink, handler = collector()
    replay = bus.replay_by_topic("workflow.*", handler=handler)
    assert replay.count == 1
    assert replay.mode == "topic"


def test_replay_by_time():
    bus = eb.create_default_event_bus()
    _seed(bus, 5)  # timestamps 0..4
    sink, handler = collector()
    replay = bus.replay_by_time(1.0, 3.0, handler=handler)
    assert replay.count == 3


def test_replay_by_correlation():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, correlation_id="x"))
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, correlation_id="y"))
    sink, handler = collector()
    replay = bus.replay_by_correlation("x", handler=handler)
    assert replay.count == 1


def test_replay_custom_filter():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, source="a"))
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, source="b"))
    sink, handler = collector()
    replay = bus.replay_custom(EventFilter(source="a"), handler=handler)
    assert replay.count == 1


def test_replay_to_subscribers():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    replay = bus.replay_last_n(1)  # no handler -> re-dispatch to subscribers
    assert replay.count == 1
    assert len(received) == 1


def test_replay_does_not_grow_history():
    bus = eb.create_default_event_bus()
    _seed(bus, 3)
    before = len(bus.history())
    bus.replay_last_n(3, handler=lambda e: None)
    assert len(bus.history()) == before


def test_replay_invalid_handler():
    bus = eb.create_default_event_bus()
    with pytest.raises(ReplayError):
        bus.replay([], handler="not-callable")


def test_replay_roundtrip_record():
    bus = eb.create_default_event_bus()
    _seed(bus, 2)
    replay = bus.replay_last_n(2, handler=lambda e: None)
    assert EventReplay.from_dict(replay.to_dict()).to_dict() == replay.to_dict()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_statistics_counts():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    stats = bus.statistics()
    assert stats.published == 1
    assert stats.delivered == 1
    assert stats.dropped == 1
    assert stats.subscribers == 2
    assert stats.dead_letter_count == 1


def test_statistics_success_failure_rates():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    bus.subscribe(failing_handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    stats = bus.statistics()
    assert stats.delivery_success_rate == pytest.approx(0.5)
    assert stats.failure_rate == pytest.approx(0.5)


def test_statistics_topic_counts():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))
    counts = bus.statistics().topic_counts
    assert counts["workflow.started"] == 2
    assert counts["audit.event"] == 1


def test_statistics_average_latency_is_float():
    bus = eb.create_default_event_bus()
    bus.subscribe(lambda e: None, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert isinstance(bus.statistics().average_latency, float)


def test_statistics_empty_bus():
    stats = eb.create_default_event_bus().statistics()
    assert stats.published == 0
    assert stats.delivery_success_rate == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_determinism_identical_history():
    def build():
        bus = eb.create_default_event_bus()
        bus.subscribe(lambda e: None, topic_pattern="#", subscriber_id="a", priority=5)
        for i in range(5):
            bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"i": i},
                                               correlation_id="c"))
        return bus.history().to_dict()

    assert build() == build()


def test_determinism_delivery_order_stable():
    def order():
        bus = eb.create_default_event_bus()
        seq = []
        bus.subscribe(lambda e: seq.append("a"), topic_pattern="#", priority=1)
        bus.subscribe(lambda e: seq.append("b"), topic_pattern="#", priority=2)
        bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
        return seq

    assert order() == order() == ["b", "a"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
def test_thread_safe_publishing():
    bus = eb.create_default_event_bus()
    counter = {"n": 0}
    lock = threading.Lock()

    def handler(event):
        with lock:
            counter["n"] += 1

    bus.subscribe(handler, topic_pattern="#")

    def worker():
        for _ in range(20):
            bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = bus.statistics()
    assert stats.published == 200
    assert stats.delivered == 200
    assert counter["n"] == 200
    assert len(bus.history()) == 200


def test_thread_safe_subscribe():
    bus = eb.create_default_event_bus()

    def worker():
        bus.subscribe(lambda e: None, topic_pattern="#")

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bus.subscriber_count() == 40


def test_thread_safe_sequences_unique():
    bus = eb.create_default_event_bus()

    def worker():
        for _ in range(25):
            bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [e.sequence for e in bus.history()]
    assert len(seqs) == 200
    assert len(set(seqs)) == 200  # all sequences unique


# ---------------------------------------------------------------------------
# Scale / memory safety
# ---------------------------------------------------------------------------
def test_large_event_stream():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    for i in range(1000):
        bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"i": i}))
    assert len(received) == 1000
    assert bus.statistics().published == 1000


def test_max_history_bounds_memory():
    bus = eb.EnterpriseEventBus(max_history=100)
    for i in range(1000):
        bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"i": i}))
    assert len(bus.history()) == 100
    # The retained window is the most recent.
    assert bus.history().events[-1].sequence == 999


def test_many_subscribers_fan_out():
    bus = eb.create_default_event_bus()
    counts = []
    for _ in range(50):
        received, handler = collector()
        counts.append(received)
        bus.subscribe(handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    assert all(len(c) == 1 for c in counts)
    assert bus.statistics().delivered == 50


# ---------------------------------------------------------------------------
# Frozen / slots / JSON
# ---------------------------------------------------------------------------
_FROZEN = [
    EventMetadata(),
    EnterpriseEvent.create(EventType.SYSTEM_EVENT),
    EventEnvelope(ev()),
    EventResult("e", "s", True),
    EventFilter(source="s"),
    EventSubscription("s1", "sub"),
    EventHistory(),
    DeadLetterEvent(ev(), "s", 1, "r"),
    EventReplay("r", "topic", "c", ("e",), 1, 0.0),
    EventBatch("b", (ev(),)),
    EventStatistics(),
]


@pytest.mark.parametrize("instance", _FROZEN)
def test_frozen(instance):
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, getattr(instance, field_name))


@pytest.mark.parametrize("instance", _FROZEN)
def test_slots(instance):
    assert not hasattr(instance, "__dict__")


@pytest.mark.parametrize("factory", [
    lambda: EnterpriseEvent.create(EventType.SYSTEM_EVENT, {"a": 1}).to_dict(),
    lambda: EventMetadata(source="s").to_dict(),
    lambda: EventFilter(source="s").to_dict(),
    lambda: EventSubscription("s1", "sub").to_dict(),
    lambda: DeadLetterEvent(ev(), "s", 1, "r").to_dict(),
    lambda: EventStatistics(1, 1).to_dict(),
    lambda: EventBatch("b", (ev(),)).to_dict(),
])
def test_json_serializable(factory):
    payload = factory()
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
def test_logical_clock():
    clk = eb.LogicalClock()
    assert clk.now() == 0.0
    clk.advance(3.0)
    assert clk.now() == 3.0


def test_logical_clock_negative():
    with pytest.raises(EventBusError):
        eb.LogicalClock().advance(-1.0)


def test_system_clock_now_is_float():
    assert isinstance(eb.SystemClock().now(), float)


# ---------------------------------------------------------------------------
# Backward compatibility / non-invasiveness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "import kafka", "import pika", "import redis", "import celery",
    "import asyncio", "from kafka", "import aio_pika",
])
def test_no_forbidden_imports(forbidden):
    with open(eb.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert forbidden not in source


def test_module_self_contained():
    with open(eb.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "from src" not in source
    assert "workflow_engine" not in source
    assert "business_process_orchestrator" not in source


def test_factory_builds_bus():
    assert isinstance(eb.create_default_event_bus(), eb.EnterpriseEventBus)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_demo():
    assert eb.main(["--demo"]) == 0


def test_cli_no_args():
    assert eb.main([]) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_publish_event_with_explicit_id_preserved():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, event_id="my-id"))
    assert received[0].event_id == "my-id"


def test_publish_preserves_explicit_correlation():
    bus = eb.create_default_event_bus()
    received, handler = collector()
    bus.subscribe(handler, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT, correlation_id="given"))
    assert received[0].correlation_id == "given"


def test_handler_can_publish_reentrantly():
    bus = eb.create_default_event_bus()
    seen = []

    def cascading(event):
        seen.append(event.event_type)
        if event.event_type is EventType.WORKFLOW_STARTED:
            bus.publish(EnterpriseEvent.create(EventType.AUDIT_EVENT))

    bus.subscribe(cascading, topic_pattern="#")
    bus.publish(EnterpriseEvent.create(EventType.WORKFLOW_STARTED))
    assert EventType.WORKFLOW_STARTED in seen
    assert EventType.AUDIT_EVENT in seen
    assert len(bus.history()) == 2


def test_unrouted_does_not_count_as_dropped():
    bus = eb.create_default_event_bus()
    bus.publish(EnterpriseEvent.create(EventType.SYSTEM_EVENT))
    stats = bus.statistics()
    assert stats.dropped == 0
    assert stats.unrouted == 1