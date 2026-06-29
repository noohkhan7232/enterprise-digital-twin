"""Enterprise test suite for the Week 10 Phase 5 Enterprise Integration Layer.

Standard pytest (parametrize / raises / approx only - no fixtures), with a
bootstrap resolving the module in the repo layout and standalone.
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
sys.path.insert(0, os.path.join(_HERE, "..", "src", "integration"))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:
    il = importlib.import_module("src.integration.enterprise_integration_layer")
except ModuleNotFoundError:
    il = importlib.import_module("enterprise_integration_layer")

globals().update({name: getattr(il, name) for name in il.__all__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ok_adapter(output=None):
    def adapter(ctx, clock):
        return output if output is not None else {"ok": True}
    return adapter


def fail_adapter(ctx, clock):
    raise RuntimeError("boom")


def slow_adapter(seconds):
    def adapter(ctx, clock):
        clock.advance(seconds)
        return {"slow": True}
    return adapter


def bad_return_adapter(ctx, clock):
    return "not-a-mapping"


def desc(module_id="m", module_type=None, version="1.0.0", capabilities=(), priority=0):
    return ModuleDescriptor(module_id, module_type or ModuleType.CUSTOM, version,
                            tuple(capabilities), priority)


def layer_with(*modules):
    """modules: tuples of (module_id, module_type, capabilities, adapter, priority)."""
    layer = create_default_integration_layer()
    for spec in modules:
        mid, mtype, caps, adapter, prio = spec
        layer.register_module(desc(mid, mtype, "1.0.0", caps, prio), adapter)
    return layer


def ctx(payload=None, **kw):
    return IntegrationContext.create(payload or {}, **kw)


def direct_req(target, payload=None, **kw):
    return IntegrationRequest.create(RouteStrategy.DIRECT, target=target,
                                     context=ctx(payload), **kw)


ALL_MODULE_TYPES = list(ModuleType)
ALL_ROUTE_STRATEGIES = list(RouteStrategy)
ALL_RESPONSE_STATUSES = list(ResponseStatus)
ALL_HEALTH_STATES = list(HealthState)
ALL_CIRCUIT_STATES = list(CircuitState)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enum_cls,value", (
    [(ModuleType, e.value) for e in ModuleType]
    + [(RouteStrategy, e.value) for e in RouteStrategy]
    + [(ResponseStatus, e.value) for e in ResponseStatus]
    + [(HealthState, e.value) for e in HealthState]
    + [(CircuitState, e.value) for e in CircuitState]
))
def test_enum_coerce(enum_cls, value):
    assert enum_cls.coerce(value).value == value


@pytest.mark.parametrize("enum_cls", [
    ModuleType, RouteStrategy, ResponseStatus, HealthState, CircuitState])
def test_enum_coerce_invalid(enum_cls):
    with pytest.raises(IntegrationError):
        enum_cls.coerce("__nope__")


# ---------------------------------------------------------------------------
# ModuleDescriptor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mtype", ALL_MODULE_TYPES)
def test_descriptor_roundtrip(mtype):
    d = desc("m1", mtype, "2.1.0", ("a", "b"), 5)
    assert ModuleDescriptor.from_dict(d.to_dict()).to_dict() == d.to_dict()


def test_descriptor_empty_id():
    with pytest.raises(ModuleRegistrationError):
        ModuleDescriptor("", ModuleType.CUSTOM)


@pytest.mark.parametrize("version", ["", "abc", "1.x", "1..0", "v1.0"])
def test_descriptor_invalid_version(version):
    with pytest.raises(ModuleRegistrationError):
        ModuleDescriptor("m", ModuleType.CUSTOM, version)


@pytest.mark.parametrize("version", ["1", "1.0", "1.0.0", "10.20.30"])
def test_descriptor_valid_version(version):
    assert ModuleDescriptor("m", ModuleType.CUSTOM, version).version == version


def test_descriptor_has_capability():
    d = desc("m", capabilities=("predict", "score"))
    assert d.has_capability("predict")
    assert not d.has_capability("missing")


# ---------------------------------------------------------------------------
# ModuleRegistration
# ---------------------------------------------------------------------------
def test_registration_roundtrip():
    r = ModuleRegistration(desc("m1"), registered_at=3.0, enabled=True, frozen=False)
    assert ModuleRegistration.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert r.module_id == "m1"


def test_registration_requires_descriptor():
    with pytest.raises(ModuleRegistrationError):
        ModuleRegistration("not-a-descriptor")


# ---------------------------------------------------------------------------
# IntegrationContext
# ---------------------------------------------------------------------------
def test_context_roundtrip():
    c = IntegrationContext.create({"k": 1}, correlation_id="c", trace_id="t",
                                  metadata={"m": 2})
    assert IntegrationContext.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_context_payload_metadata():
    c = IntegrationContext.create({"a": 1}, metadata={"b": 2})
    assert c.payload == {"a": 1}
    assert c.metadata == {"b": 2}


def test_context_with_output():
    c = IntegrationContext.create({"x": 1})
    c2 = c.with_output("mod", {"r": 9})
    assert c.payload == {"x": 1}  # original unchanged
    assert c2.payload["mod_output"] == {"r": 9}
    assert c2.payload["last_output"] == {"r": 9}


# ---------------------------------------------------------------------------
# IntegrationRequest / IntegrationResponse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ALL_ROUTE_STRATEGIES)
def test_request_roundtrip(strategy):
    r = IntegrationRequest.create(strategy, target="t", capability="c",
                                  module_type=ModuleType.WORKFLOW, context=ctx({"k": 1}),
                                  fallback_target="fb", timeout=2.0, max_retries=3,
                                  retry_backoff=1.0, priority=4)
    assert IntegrationRequest.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_request_validation_max_retries():
    with pytest.raises(DispatchError):
        IntegrationRequest("r", RouteStrategy.DIRECT, ctx(), max_retries=0)


def test_request_validation_negative_timeout():
    with pytest.raises(DispatchError):
        IntegrationRequest("r", RouteStrategy.DIRECT, ctx(), timeout=-1.0)


def test_request_requires_context():
    with pytest.raises(DispatchError):
        IntegrationRequest("r", RouteStrategy.DIRECT, "not-a-context")


def test_request_auto_id():
    r = IntegrationRequest.create(RouteStrategy.DIRECT, target="t")
    assert r.request_id.startswith("req-")


@pytest.mark.parametrize("status", ALL_RESPONSE_STATUSES)
def test_response_roundtrip(status):
    r = IntegrationResponse("r", "m", status, "direct", '{"o":1}', "err", 1.5, 2, True)
    assert IntegrationResponse.from_dict(r.to_dict()).to_dict() == r.to_dict()


@pytest.mark.parametrize("status,succeeded", [
    (ResponseStatus.SUCCESS, True), (ResponseStatus.FALLBACK, True),
    (ResponseStatus.FAILURE, False), (ResponseStatus.TIMEOUT, False),
    (ResponseStatus.CIRCUIT_OPEN, False), (ResponseStatus.REJECTED, False),
])
def test_response_succeeded(status, succeeded):
    assert IntegrationResponse("r", "m", status, "direct").succeeded is succeeded


# ---------------------------------------------------------------------------
# RoutingRule
# ---------------------------------------------------------------------------
def test_routing_rule_roundtrip():
    rule = RoutingRule("r1", "target", match_capability="cap",
                       match_module_type=ModuleType.WORKFLOW, condition_key="k",
                       condition_value="v", priority=3)
    assert RoutingRule.from_dict(rule.to_dict()).to_dict() == rule.to_dict()


def test_routing_rule_validation():
    with pytest.raises(RouteValidationError):
        RoutingRule("", "t")
    with pytest.raises(RouteValidationError):
        RoutingRule("r", "")


def test_routing_rule_matches_capability():
    rule = RoutingRule("r", "t", match_capability="cap")
    assert rule.matches(IntegrationRequest.create(RouteStrategy.CONDITIONAL, capability="cap"))
    assert not rule.matches(IntegrationRequest.create(RouteStrategy.CONDITIONAL, capability="x"))


def test_routing_rule_matches_condition():
    rule = RoutingRule("r", "t", condition_key="env", condition_value="prod")
    match = IntegrationRequest.create(RouteStrategy.CONDITIONAL, context=ctx(metadata={"env": "prod"}))
    nomatch = IntegrationRequest.create(RouteStrategy.CONDITIONAL, context=ctx(metadata={"env": "dev"}))
    assert rule.matches(match)
    assert not rule.matches(nomatch)


def test_routing_rule_disabled():
    rule = RoutingRule("r", "t", enabled=False)
    assert not rule.matches(IntegrationRequest.create(RouteStrategy.CONDITIONAL))


# ---------------------------------------------------------------------------
# Audit / Health / Statistics / Snapshot dataclasses
# ---------------------------------------------------------------------------
def test_audit_roundtrip():
    a = IntegrationAudit(0, "r", "c", "m", 1.0, 0.5, ResponseStatus.SUCCESS, "direct")
    assert IntegrationAudit.from_dict(a.to_dict()).to_dict() == a.to_dict()


@pytest.mark.parametrize("state", ALL_HEALTH_STATES)
@pytest.mark.parametrize("circuit", ALL_CIRCUIT_STATES)
def test_health_roundtrip(state, circuit):
    h = IntegrationHealth("m", state, 1.0, 0.2, 1, 9, 0.9, 0.9, 5.0, 3, circuit)
    assert IntegrationHealth.from_dict(h.to_dict()).to_dict() == h.to_dict()


def test_statistics_roundtrip():
    s = IntegrationStatistics(10, 8, 2, 1, 1, 0, 2, 0.5, 3.0, 0.8, 0.2,
                              '{"m":3}', '{"direct":5}')
    assert IntegrationStatistics.from_dict(s.to_dict()).to_dict() == s.to_dict()
    assert s.module_usage == {"m": 3}
    assert s.route_counts == {"direct": 5}


def test_snapshot_roundtrip():
    snap = IntegrationSnapshot(
        1.0, (ModuleRegistration(desc("m")),), IntegrationStatistics(),
        (IntegrationHealth("m", HealthState.UNKNOWN, 1.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0),),
        0, False)
    assert IntegrationSnapshot.from_dict(snap.to_dict()).to_dict() == snap.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_register_and_exists():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    assert layer.module_exists("m1")


def test_register_duplicate():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    with pytest.raises(ModuleRegistrationError):
        layer.register_module(desc("m1"), ok_adapter())


def test_register_overwrite():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter({"v": 1}))
    layer.register_module(desc("m1"), ok_adapter({"v": 2}), overwrite=True)
    resp = layer.dispatch(direct_req("m1"))
    assert resp.output == {"v": 2}


def test_register_non_callable_adapter():
    layer = create_default_integration_layer()
    with pytest.raises(ModuleRegistrationError):
        layer.register_module(desc("m1"), "not-callable")


def test_register_non_descriptor():
    layer = create_default_integration_layer()
    with pytest.raises(ModuleRegistrationError):
        layer.register_module("not-a-descriptor", ok_adapter())


def test_unregister():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    layer.unregister_module("m1")
    assert not layer.module_exists("m1")


def test_unregister_unknown():
    with pytest.raises(ModuleNotFoundError):
        create_default_integration_layer().unregister_module("ghost")


def test_list_modules_sorted():
    layer = create_default_integration_layer()
    layer.register_module(desc("b"), ok_adapter())
    layer.register_module(desc("a"), ok_adapter())
    assert [m.module_id for m in layer.list_modules()] == ["a", "b"]


def test_find_module_by_id():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    found = layer.find_module(module_id="a")
    assert len(found) == 1 and found[0].module_id == "a"


def test_find_module_by_capability():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, ("predict",), ok_adapter(), 1),
        ("b", ModuleType.PREDICTION, ("other",), ok_adapter(), 2))
    found = layer.find_module(capability="predict")
    assert [m.module_id for m in found] == ["a"]


def test_find_module_by_type():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), ok_adapter(), 0))
    found = layer.find_module(module_type=ModuleType.WORKFLOW)
    assert [m.module_id for m in found] == ["b"]


def test_find_module_priority_order():
    layer = layer_with(
        ("low", ModuleType.PREDICTION, ("c",), ok_adapter(), 1),
        ("high", ModuleType.PREDICTION, ("c",), ok_adapter(), 10))
    found = layer.find_module(capability="c")
    assert [m.module_id for m in found] == ["high", "low"]


def test_freeze_registry_blocks_register():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    layer.freeze_registry()
    assert layer.registry_is_frozen
    with pytest.raises(RegistryFrozenError):
        layer.register_module(desc("m2"), ok_adapter())


def test_freeze_registry_blocks_unregister():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    layer.freeze_registry()
    with pytest.raises(RegistryFrozenError):
        layer.unregister_module("m1")


def test_freeze_marks_registrations():
    layer = create_default_integration_layer()
    layer.register_module(desc("m1"), ok_adapter())
    layer.freeze_registry()
    assert all(m.frozen for m in layer.list_modules())


def test_registry_snapshot():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    snap = layer.registry_snapshot()
    assert isinstance(snap, IntegrationSnapshot)
    assert len(snap.modules) == 1


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_route_direct():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    targets, route = layer.resolve_route(direct_req("a"))
    assert targets == ["a"] and route == "direct"


def test_route_direct_unknown():
    layer = create_default_integration_layer()
    with pytest.raises(RouteValidationError):
        layer.resolve_route(direct_req("ghost"))


def test_route_capability():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, ("predict",), ok_adapter(), 1),
        ("b", ModuleType.PREDICTION, ("predict",), ok_adapter(), 5))
    targets, route = layer.resolve_route(
        IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="predict"))
    assert targets == ["b"] and route == "capability"  # highest priority


def test_route_capability_none():
    layer = create_default_integration_layer()
    with pytest.raises(RouteValidationError):
        layer.resolve_route(IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="x"))


def test_route_priority():
    layer = layer_with(
        ("a", ModuleType.WORKFLOW, (), ok_adapter(), 2),
        ("b", ModuleType.WORKFLOW, (), ok_adapter(), 9))
    targets, route = layer.resolve_route(
        IntegrationRequest.create(RouteStrategy.PRIORITY, module_type=ModuleType.WORKFLOW))
    assert targets == ["b"] and route == "priority"


def test_route_conditional():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.register_route(RoutingRule("r", "a", condition_key="env", condition_value="prod"))
    req = IntegrationRequest.create(RouteStrategy.CONDITIONAL, context=ctx(metadata={"env": "prod"}))
    targets, route = layer.resolve_route(req)
    assert targets == ["a"] and route == "conditional"


def test_route_conditional_no_match():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.register_route(RoutingRule("r", "a", condition_key="env", condition_value="prod"))
    req = IntegrationRequest.create(RouteStrategy.CONDITIONAL, context=ctx(metadata={"env": "dev"}))
    with pytest.raises(RouteValidationError):
        layer.resolve_route(req)


def test_route_conditional_priority():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.register_route(RoutingRule("low", "a", priority=1))
    layer.register_route(RoutingRule("high", "b", priority=10))
    targets, _ = layer.resolve_route(IntegrationRequest.create(RouteStrategy.CONDITIONAL))
    assert targets == ["b"]


def test_route_broadcast():
    layer = layer_with(
        ("a", ModuleType.EVENT_BUS, ("publish",), ok_adapter(), 1),
        ("b", ModuleType.CUSTOM, ("publish",), ok_adapter(), 2))
    targets, route = layer.resolve_route(
        IntegrationRequest.create(RouteStrategy.BROADCAST, capability="publish"))
    assert set(targets) == {"a", "b"} and route == "broadcast"


def test_route_pipeline_rejected():
    layer = create_default_integration_layer()
    with pytest.raises(RouteValidationError):
        layer.resolve_route(IntegrationRequest.create(RouteStrategy.PIPELINE))


def test_validate_route():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    assert layer.validate_route(direct_req("a")) is True


def test_register_duplicate_route():
    layer = create_default_integration_layer()
    layer.register_route(RoutingRule("r", "a"))
    with pytest.raises(RouteValidationError):
        layer.register_route(RoutingRule("r", "b"))


def test_route_skips_disabled_rule():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.register_route(RoutingRule("r", "a", enabled=False))
    with pytest.raises(RouteValidationError):
        layer.resolve_route(IntegrationRequest.create(RouteStrategy.CONDITIONAL))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def test_dispatch_direct_success():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter({"r": 1}), 0))
    resp = layer.dispatch(direct_req("a"))
    assert resp.succeeded and resp.output == {"r": 1}
    assert resp.module_id == "a"


def test_dispatch_failure():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    resp = layer.dispatch(direct_req("a"))
    assert resp.status is ResponseStatus.FAILURE
    assert "boom" in resp.error


def test_dispatch_bad_return():
    layer = layer_with(("a", ModuleType.PREDICTION, (), bad_return_adapter, 0))
    resp = layer.dispatch(direct_req("a"))
    assert resp.status is ResponseStatus.FAILURE


def test_dispatch_capability():
    layer = layer_with(("a", ModuleType.PREDICTION, ("predict",), ok_adapter({"p": 1}), 0))
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="predict"))
    assert resp.succeeded and resp.module_id == "a"


def test_dispatch_batch():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter({"x": 1}), 0),
        ("b", ModuleType.WORKFLOW, (), ok_adapter({"y": 2}), 0))
    responses = layer.dispatch_batch([direct_req("a"), direct_req("b")])
    assert len(responses) == 2
    assert all(r.succeeded for r in responses)


def test_dispatch_broadcast_all_success():
    layer = layer_with(
        ("a", ModuleType.EVENT_BUS, ("pub",), ok_adapter({"a": 1}), 0),
        ("b", ModuleType.CUSTOM, ("pub",), ok_adapter({"b": 2}), 0))
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="pub"))
    assert resp.status is ResponseStatus.SUCCESS
    assert set(resp.output) == {"a", "b"}


def test_dispatch_broadcast_partial():
    layer = layer_with(
        ("a", ModuleType.EVENT_BUS, ("pub",), ok_adapter({"a": 1}), 0),
        ("b", ModuleType.CUSTOM, ("pub",), fail_adapter, 0))
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="pub"))
    assert resp.status is ResponseStatus.PARTIAL


def test_dispatch_broadcast_all_fail():
    layer = layer_with(
        ("a", ModuleType.EVENT_BUS, ("pub",), fail_adapter, 0),
        ("b", ModuleType.CUSTOM, ("pub",), fail_adapter, 0))
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="pub"))
    assert resp.status is ResponseStatus.FAILURE


def test_dispatch_unknown_module():
    layer = create_default_integration_layer()
    with pytest.raises(RouteValidationError):
        layer.dispatch(direct_req("ghost"))


# ---------------------------------------------------------------------------
# Retry / timeout / fallback / circuit breaker
# ---------------------------------------------------------------------------
def test_retry_to_max():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    resp = layer.dispatch(direct_req("a", max_retries=3))
    assert resp.attempts == 3
    assert resp.status is ResponseStatus.FAILURE


def test_retry_eventual_success():
    state = {"n": 0}

    def flaky(ctx, clock):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("transient")
        return {"ok": True}

    layer = layer_with(("a", ModuleType.PREDICTION, (), flaky, 0))
    resp = layer.dispatch(direct_req("a", max_retries=3))
    assert resp.succeeded and resp.attempts == 2


def test_timeout():
    layer = layer_with(("a", ModuleType.PREDICTION, (), slow_adapter(5.0), 0))
    resp = layer.dispatch(direct_req("a", timeout=1.0))
    assert resp.status is ResponseStatus.TIMEOUT


def test_fallback_used_on_failure():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), fail_adapter, 0),
        ("fb", ModuleType.PREDICTION, (), ok_adapter({"recovered": True}), 0))
    req = IntegrationRequest.create(RouteStrategy.DIRECT, target="a", fallback_target="fb",
                                    context=ctx())
    resp = layer.dispatch(req)
    assert resp.status is ResponseStatus.FALLBACK
    assert resp.fallback_used and resp.module_id == "fb"
    assert resp.output == {"recovered": True}


def test_fallback_not_triggered_on_success():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter({"primary": True}), 0),
        ("fb", ModuleType.PREDICTION, (), ok_adapter({"fb": True}), 0))
    req = IntegrationRequest.create(RouteStrategy.DIRECT, target="a", fallback_target="fb",
                                    context=ctx())
    resp = layer.dispatch(req)
    assert resp.status is ResponseStatus.SUCCESS
    assert not resp.fallback_used


def test_circuit_opens_after_threshold():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    for _ in range(3):  # default threshold = 3
        layer.dispatch(direct_req("a"))
    assert layer.health("a").circuit_state is CircuitState.OPEN


def test_circuit_open_short_circuits():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    for _ in range(3):
        layer.dispatch(direct_req("a"))
    resp = layer.dispatch(direct_req("a"))
    assert resp.status is ResponseStatus.CIRCUIT_OPEN


def test_circuit_half_open_then_closed():
    calls = {"n": 0}

    def recover(ctx, clock):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("down")
        return {"ok": True}

    layer = il.EnterpriseIntegrationLayer(clock=il.LogicalClock(), circuit_threshold=3,
                                          circuit_cooldown=2.0)
    layer.register_module(desc("a", ModuleType.PREDICTION), recover)
    for _ in range(3):
        layer.dispatch(direct_req("a"))
    assert layer.health("a").circuit_state is CircuitState.OPEN
    # advance the clock past cooldown via successful dispatches on another module
    layer.register_module(desc("b", ModuleType.CUSTOM), ok_adapter())
    for _ in range(3):
        layer.dispatch(direct_req("b"))
    resp = layer.dispatch(direct_req("a"))  # half-open trial -> success -> closed
    assert resp.succeeded
    assert layer.health("a").circuit_state is CircuitState.CLOSED


def test_circuit_open_uses_fallback():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), fail_adapter, 0),
        ("fb", ModuleType.PREDICTION, (), ok_adapter({"fb": True}), 0))
    for _ in range(3):
        layer.dispatch(direct_req("a"))
    req = IntegrationRequest.create(RouteStrategy.DIRECT, target="a", fallback_target="fb",
                                    context=ctx())
    resp = layer.dispatch(req)
    assert resp.status is ResponseStatus.FALLBACK


def test_retry_backoff_advances_clock():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    resp = layer.dispatch(direct_req("a", max_retries=3, retry_backoff=2.0))
    assert resp.duration == pytest.approx(6.0)  # backoff before attempts 2 and 3: 2+4


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def test_pipeline_runs_in_order():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter({"step": "a"}), 0),
        ("b", ModuleType.WORKFLOW, (), ok_adapter({"step": "b"}), 0),
        ("c", ModuleType.SCHEDULER, (), ok_adapter({"step": "c"}), 0))
    responses = layer.dispatch_pipeline(["a", "b", "c"])
    assert [r.module_id for r in responses] == ["a", "b", "c"]
    assert all(r.succeeded for r in responses)


def test_pipeline_threads_output():
    seen = {}

    def stage_a(ctx, clock):
        return {"value": 10}

    def stage_b(ctx, clock):
        seen["b_input"] = ctx.payload.get("last_output")
        return {"value": 20}

    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), stage_a, 0),
        ("b", ModuleType.WORKFLOW, (), stage_b, 0))
    layer.dispatch_pipeline(["a", "b"])
    assert seen["b_input"] == {"value": 10}


def test_pipeline_stops_on_failure():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), fail_adapter, 0),
        ("c", ModuleType.SCHEDULER, (), ok_adapter(), 0))
    responses = layer.dispatch_pipeline(["a", "b", "c"])
    assert len(responses) == 2  # stopped at b
    assert responses[-1].status is ResponseStatus.FAILURE


def test_pipeline_continue_on_failure():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), fail_adapter, 0),
        ("c", ModuleType.SCHEDULER, (), ok_adapter(), 0))
    responses = layer.dispatch_pipeline(["a", "b", "c"], stop_on_failure=False)
    assert len(responses) == 3


def test_pipeline_by_capability():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, ("predict",), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, ("create",), ok_adapter(), 0))
    responses = layer.dispatch_pipeline(["predict", "create"], by_capability=True)
    assert [r.module_id for r in responses] == ["a", "b"]


def test_pipeline_empty():
    layer = create_default_integration_layer()
    with pytest.raises(PipelineError):
        layer.dispatch_pipeline([])


def test_pipeline_unknown_stage():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    with pytest.raises(RouteValidationError):
        layer.dispatch_pipeline(["a", "ghost"])


def test_pipeline_increments_run_count():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch_pipeline(["a"])
    layer.dispatch_pipeline(["a"])
    assert layer.statistics().pipeline_runs == 2


def test_flagship_pipeline_eight_stages():
    specs = [
        ("prediction", ModuleType.PREDICTION),
        ("risk", ModuleType.SIMULATION),
        ("knowledge", ModuleType.KNOWLEDGE),
        ("copilot", ModuleType.EXECUTIVE_COPILOT),
        ("workflow", ModuleType.WORKFLOW),
        ("process", ModuleType.BUSINESS_PROCESS),
        ("scheduler", ModuleType.SCHEDULER),
        ("eventbus", ModuleType.EVENT_BUS),
    ]
    layer = create_default_integration_layer()
    for mid, mtype in specs:
        layer.register_module(desc(mid, mtype), ok_adapter({"m": mid}))
    responses = layer.dispatch_pipeline([s[0] for s in specs])
    assert len(responses) == 8
    assert all(r.succeeded for r in responses)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_unknown_initially():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    assert layer.health("a").state is HealthState.UNKNOWN


def test_health_healthy_after_success():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    h = layer.health("a")
    assert h.state is HealthState.HEALTHY
    assert h.success_rate == pytest.approx(1.0)
    assert h.health_score == pytest.approx(1.0)


def test_health_unhealthy_after_failures():
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    for _ in range(2):
        layer.dispatch(direct_req("a"))
    assert layer.health("a").state is HealthState.UNHEALTHY


def test_health_records_counts():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(direct_req("a"))
    h = layer.health("a")
    assert h.success_count == 2
    assert h.failure_count == 0


def test_heartbeat():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    h = layer.heartbeat("a")
    assert h.heartbeat == 1
    assert layer.heartbeat("a").heartbeat == 2


def test_heartbeat_unknown():
    with pytest.raises(ModuleNotFoundError):
        create_default_integration_layer().heartbeat("ghost")


def test_health_unknown_module():
    with pytest.raises(ModuleNotFoundError):
        create_default_integration_layer().health("ghost")


def test_health_all():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), ok_adapter(), 0))
    healths = layer.health_all()
    assert {h.module_id for h in healths} == {"a", "b"}


def test_health_response_time():
    layer = layer_with(("a", ModuleType.PREDICTION, (), slow_adapter(3.0), 0))
    layer.dispatch(direct_req("a"))
    assert layer.health("a").response_time == pytest.approx(3.0)


def test_health_last_seen_advances():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    first = layer.health("a").last_seen
    layer.dispatch(direct_req("a"))
    assert layer.health("a").last_seen > first


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def test_audit_records_dispatch():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a", payload={}))
    audit = layer.audit_log()
    assert len(audit) == 1
    assert audit[0].module_id == "a"
    assert audit[0].status is ResponseStatus.SUCCESS


def test_audit_sequence_increments():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(direct_req("a"))
    seqs = [a.sequence for a in layer.audit_log()]
    assert seqs == [0, 1]


def test_audit_records_correlation():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(IntegrationRequest.create(RouteStrategy.DIRECT, target="a",
                                             context=ctx(correlation_id="corr-9")))
    assert layer.audit_log()[0].correlation_id == "corr-9"


def test_audit_records_route():
    layer = layer_with(("a", ModuleType.PREDICTION, ("c",), ok_adapter(), 0))
    layer.dispatch(IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="c"))
    assert layer.audit_log()[0].route == "capability"


def test_audit_max_cap():
    layer = il.EnterpriseIntegrationLayer(clock=il.LogicalClock(), max_audit=5)
    layer.register_module(desc("a", ModuleType.PREDICTION), ok_adapter())
    for _ in range(20):
        layer.dispatch(direct_req("a"))
    assert len(layer.audit_log()) == 5


def test_audit_roundtrip_from_log():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    a = layer.audit_log()[0]
    assert IntegrationAudit.from_dict(a.to_dict()).to_dict() == a.to_dict()


# ---------------------------------------------------------------------------
# Statistics / observability
# ---------------------------------------------------------------------------
def test_statistics_counts():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), fail_adapter, 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(direct_req("b"))
    stats = layer.statistics()
    assert stats.total_requests == 2
    assert stats.successful == 1
    assert stats.failed == 1


def test_statistics_rates():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), ok_adapter(), 0),
        ("b", ModuleType.WORKFLOW, (), fail_adapter, 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(direct_req("b"))
    stats = layer.statistics()
    assert stats.success_rate == pytest.approx(0.5)
    assert stats.failure_rate == pytest.approx(0.5)


def test_statistics_module_usage():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(direct_req("a"))
    assert layer.statistics().module_usage["a"] == 2


def test_statistics_route_counts():
    layer = layer_with(("a", ModuleType.PREDICTION, ("c",), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    layer.dispatch(IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="c"))
    counts = layer.statistics().route_counts
    assert counts["direct"] == 1
    assert counts["capability"] == 1


def test_statistics_timeouts_and_fallbacks():
    layer = layer_with(
        ("a", ModuleType.PREDICTION, (), slow_adapter(5.0), 0),
        ("fb", ModuleType.PREDICTION, (), ok_adapter(), 0))
    req = IntegrationRequest.create(RouteStrategy.DIRECT, target="a", fallback_target="fb",
                                    timeout=1.0, context=ctx())
    layer.dispatch(req)
    stats = layer.statistics()
    assert stats.fallbacks == 1


def test_statistics_empty():
    stats = create_default_integration_layer().statistics()
    assert stats.total_requests == 0
    assert stats.success_rate == 0.0


def test_snapshot_contains_everything():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    snap = layer.snapshot()
    assert len(snap.modules) == 1
    assert len(snap.health) == 1
    assert snap.audit_count == 1
    assert snap.statistics.total_requests == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_determinism_pipeline_replay():
    def build():
        layer = create_default_integration_layer()
        for mid in ["a", "b", "c"]:
            layer.register_module(desc(mid, ModuleType.CUSTOM), ok_adapter({"m": mid}))
        responses = layer.dispatch_pipeline(["a", "b", "c"],
                                            context=ctx({"x": 1}, correlation_id="c"))
        return [r.to_dict() for r in responses]

    assert build() == build()


def test_determinism_audit():
    def build():
        layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
        for _ in range(5):
            layer.dispatch(direct_req("a"))
        return [a.to_dict() for a in layer.audit_log()]

    assert build() == build()


def test_determinism_capability_tiebreak():
    # Equal priority -> deterministic tie-break by module_id.
    layer = layer_with(
        ("z", ModuleType.PREDICTION, ("c",), ok_adapter(), 5),
        ("a", ModuleType.PREDICTION, ("c",), ok_adapter(), 5))
    targets, _ = layer.resolve_route(
        IntegrationRequest.create(RouteStrategy.CAPABILITY, capability="c"))
    assert targets == ["a"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
def test_thread_safe_registration():
    layer = create_default_integration_layer()

    def worker(i):
        layer.register_module(desc(f"m{i}", ModuleType.CUSTOM), ok_adapter())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(layer.list_modules()) == 50


def test_thread_safe_dispatch():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    errors = []

    def worker():
        for _ in range(20):
            try:
                layer.dispatch(direct_req("a"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert layer.statistics().total_requests == 200


# ---------------------------------------------------------------------------
# Scale / performance
# ---------------------------------------------------------------------------
def test_large_registry():
    layer = create_default_integration_layer()
    for i in range(200):
        layer.register_module(desc(f"m{i}", ModuleType.CUSTOM, capabilities=("c",)), ok_adapter())
    assert len(layer.list_modules()) == 200
    assert len(layer.find_module(capability="c")) == 200


def test_large_pipeline():
    layer = create_default_integration_layer()
    stages = []
    for i in range(100):
        mid = f"m{i:03d}"
        layer.register_module(desc(mid, ModuleType.CUSTOM), ok_adapter({"i": i}))
        stages.append(mid)
    responses = layer.dispatch_pipeline(stages)
    assert len(responses) == 100
    assert all(r.succeeded for r in responses)


def test_large_batch():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    requests = [direct_req("a") for _ in range(500)]
    responses = layer.dispatch_batch(requests)
    assert len(responses) == 500
    assert layer.statistics().total_requests == 500


# ---------------------------------------------------------------------------
# Frozen / slots / JSON
# ---------------------------------------------------------------------------
_FROZEN = [
    ModuleDescriptor("m", ModuleType.CUSTOM),
    ModuleRegistration(ModuleDescriptor("m", ModuleType.CUSTOM)),
    IntegrationContext.create({"a": 1}),
    IntegrationRequest.create(RouteStrategy.DIRECT, target="t"),
    IntegrationResponse("r", "m", ResponseStatus.SUCCESS, "direct"),
    RoutingRule("r", "t"),
    IntegrationAudit(0, "r", "c", "m", 0.0, 0.0, ResponseStatus.SUCCESS, "direct"),
    IntegrationHealth("m", HealthState.UNKNOWN, 1.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0),
    IntegrationStatistics(),
    IntegrationSnapshot(0.0, (), IntegrationStatistics(), (), 0),
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
    lambda: ModuleDescriptor("m", ModuleType.PREDICTION, "1.0.0", ("c",)).to_dict(),
    lambda: IntegrationContext.create({"a": 1}).to_dict(),
    lambda: IntegrationRequest.create(RouteStrategy.DIRECT, target="t").to_dict(),
    lambda: IntegrationResponse("r", "m", ResponseStatus.SUCCESS, "direct").to_dict(),
    lambda: RoutingRule("r", "t").to_dict(),
    lambda: IntegrationAudit(0, "r", "c", "m", 0.0, 0.0, ResponseStatus.SUCCESS, "d").to_dict(),
    lambda: IntegrationStatistics(1, 1).to_dict(),
])
def test_json_serializable(factory):
    payload = factory()
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
def test_logical_clock():
    clk = il.LogicalClock()
    clk.advance(4.0)
    assert clk.now() == 4.0


def test_logical_clock_negative():
    with pytest.raises(IntegrationError):
        il.LogicalClock().advance(-1.0)


def test_system_clock_now():
    assert isinstance(il.SystemClock().now(), float)


# ---------------------------------------------------------------------------
# Backward compatibility / non-invasiveness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "import fastapi", "from fastapi", "import grpc", "import kafka", "import pika",
    "import redis", "import celery", "import flask", "import django",
])
def test_no_forbidden_imports(forbidden):
    with open(il.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert forbidden not in source


def test_does_not_import_upstream_modules():
    with open(il.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "workflow_engine" not in source
    assert "business_process_orchestrator" not in source
    assert "enterprise_event_bus" not in source
    assert "enterprise_scheduler" not in source


def test_factory_builds_layer():
    assert isinstance(create_default_integration_layer(), EnterpriseIntegrationLayer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_demo():
    assert il.main(["--demo"]) == 0


def test_cli_no_args():
    assert il.main([]) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_disabled_module_rejected():
    layer = create_default_integration_layer()
    layer.register_module(desc("a", ModuleType.PREDICTION), ok_adapter())
    # Disable by replacing registration via re-register with overwrite & enabled False is
    # not exposed; instead verify a removed module cannot be dispatched.
    layer.unregister_module("a")
    with pytest.raises(RouteValidationError):
        layer.dispatch(direct_req("a"))


def test_response_output_isolation():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter({"k": "v"}), 0))
    resp = layer.dispatch(direct_req("a"))
    out = resp.output
    out["mutated"] = True
    assert "mutated" not in resp.output  # fresh parse each time


def test_pipeline_single_stage():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    responses = layer.dispatch_pipeline(["a"])
    assert len(responses) == 1 and responses[0].succeeded


def test_find_module_empty():
    layer = create_default_integration_layer()
    assert layer.find_module(module_id="ghost") == ()


def test_broadcast_no_targets():
    layer = create_default_integration_layer()
    with pytest.raises(RouteValidationError):
        layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="none"))


def test_snapshot_roundtrip_live():
    layer = layer_with(("a", ModuleType.PREDICTION, (), ok_adapter(), 0))
    layer.dispatch(direct_req("a"))
    snap = layer.snapshot()
    assert IntegrationSnapshot.from_dict(snap.to_dict()).to_dict() == snap.to_dict()


# ===========================================================================
# Extended parametrized coverage
# ===========================================================================

# -- full serialization round-trips for every domain dataclass --------------
@pytest.mark.parametrize("instance", _FROZEN)
def test_serialize_roundtrip_all(instance):
    restored = type(instance).from_dict(instance.to_dict())
    assert restored.to_dict() == instance.to_dict()


# -- registration round-trip across all module types -----------------------
@pytest.mark.parametrize("mtype", ALL_MODULE_TYPES)
def test_registration_roundtrip_types(mtype):
    r = ModuleRegistration(desc("m", mtype, "1.2.3", ("x",), 4), registered_at=1.0)
    assert ModuleRegistration.from_dict(r.to_dict()).to_dict() == r.to_dict()


# -- dispatch succeeds for a module of each type ----------------------------
@pytest.mark.parametrize("mtype", ALL_MODULE_TYPES)
def test_dispatch_each_module_type(mtype):
    layer = layer_with(("a", mtype, (), ok_adapter({"t": mtype.value}), 0))
    resp = layer.dispatch(direct_req("a"))
    assert resp.succeeded
    assert resp.output == {"t": mtype.value}


# -- audit round-trip across every response status --------------------------
@pytest.mark.parametrize("status", ALL_RESPONSE_STATUSES)
def test_audit_roundtrip_status(status):
    a = IntegrationAudit(1, "r", "c", "m", 2.0, 0.3, status, "direct")
    assert IntegrationAudit.from_dict(a.to_dict()).to_dict() == a.to_dict()


# -- response round-trip across status x fallback ---------------------------
@pytest.mark.parametrize("status", ALL_RESPONSE_STATUSES)
@pytest.mark.parametrize("fallback", [True, False])
def test_response_roundtrip_matrix(status, fallback):
    r = IntegrationResponse("r", "m", status, "route", '{"k":1}', "e", 1.0, 2, fallback)
    assert IntegrationResponse.from_dict(r.to_dict()).to_dict() == r.to_dict()


# -- retry attempts equal max_retries on persistent failure -----------------
@pytest.mark.parametrize("retries", [1, 2, 3, 4, 5])
def test_retry_attempts_count(retries):
    layer = layer_with(("a", ModuleType.PREDICTION, (), fail_adapter, 0))
    resp = layer.dispatch(direct_req("a", max_retries=retries))
    assert resp.attempts == retries
    assert resp.status is ResponseStatus.FAILURE


# -- pipeline of varying lengths -------------------------------------------
@pytest.mark.parametrize("length", [1, 2, 5, 10, 25])
def test_pipeline_lengths(length):
    layer = create_default_integration_layer()
    stages = []
    for i in range(length):
        mid = f"s{i:03d}"
        layer.register_module(desc(mid, ModuleType.CUSTOM), ok_adapter({"i": i}))
        stages.append(mid)
    responses = layer.dispatch_pipeline(stages)
    assert len(responses) == length
    assert all(r.succeeded for r in responses)


# -- broadcast across varying fan-out sizes ---------------------------------
@pytest.mark.parametrize("size", [1, 2, 5, 12])
def test_broadcast_sizes(size):
    layer = create_default_integration_layer()
    for i in range(size):
        layer.register_module(desc(f"b{i}", ModuleType.CUSTOM, capabilities=("pub",)),
                              ok_adapter({"i": i}))
    resp = layer.dispatch(IntegrationRequest.create(RouteStrategy.BROADCAST, capability="pub"))
    assert resp.status is ResponseStatus.SUCCESS
    assert len(resp.output) == size


# -- health-state derivation from success ratios ----------------------------
@pytest.mark.parametrize("successes,failures,expected", [
    (10, 0, HealthState.HEALTHY),
    (9, 1, HealthState.HEALTHY),
    (8, 2, HealthState.DEGRADED),
    (1, 1, HealthState.UNHEALTHY),
    (0, 5, HealthState.UNHEALTHY),
])
def test_health_state_thresholds(successes, failures, expected):
    # Use a high circuit threshold so failures don't trip the breaker.
    layer = il.EnterpriseIntegrationLayer(clock=il.LogicalClock(), circuit_threshold=10_000)
    state = {"calls": 0}

    def adapter(c, k):
        state["calls"] += 1
        if state["calls"] <= successes:
            return {"ok": True}
        raise RuntimeError("fail")

    layer.register_module(desc("a", ModuleType.PREDICTION), adapter)
    for _ in range(successes + failures):
        layer.dispatch(direct_req("a"))
    assert layer.health("a").state is expected


# -- minimal request round-trip across strategies ---------------------------
@pytest.mark.parametrize("strategy", ALL_ROUTE_STRATEGIES)
def test_request_minimal_roundtrip(strategy):
    r = IntegrationRequest.create(strategy, target="t", capability="c")
    assert IntegrationRequest.from_dict(r.to_dict()).to_dict() == r.to_dict()


# -- conditional routing across condition values ----------------------------
@pytest.mark.parametrize("env,expected", [
    ("prod", "prod-mod"), ("staging", "staging-mod"), ("dev", "dev-mod"),
])
def test_conditional_routing_values(env, expected):
    layer = layer_with(
        ("prod-mod", ModuleType.WORKFLOW, (), ok_adapter(), 0),
        ("staging-mod", ModuleType.WORKFLOW, (), ok_adapter(), 0),
        ("dev-mod", ModuleType.WORKFLOW, (), ok_adapter(), 0))
    layer.register_route(RoutingRule("r-prod", "prod-mod", condition_key="env", condition_value="prod"))
    layer.register_route(RoutingRule("r-stg", "staging-mod", condition_key="env", condition_value="staging"))
    layer.register_route(RoutingRule("r-dev", "dev-mod", condition_key="env", condition_value="dev"))
    req = IntegrationRequest.create(RouteStrategy.CONDITIONAL, context=ctx(metadata={"env": env}))
    targets, _ = layer.resolve_route(req)
    assert targets == [expected]


# -- descriptor capability round-trips --------------------------------------
@pytest.mark.parametrize("caps", [(), ("a",), ("a", "b"), ("a", "b", "c", "d")])
def test_descriptor_capability_roundtrip(caps):
    d = desc("m", ModuleType.CUSTOM, "1.0.0", caps, 0)
    assert ModuleDescriptor.from_dict(d.to_dict()).capabilities == caps


# -- statistics round-trip across usage shapes ------------------------------
@pytest.mark.parametrize("usage", ['{}', '{"a":1}', '{"a":3,"b":7}'])
def test_statistics_usage_roundtrip(usage):
    s = IntegrationStatistics(total_requests=5, module_usage_json=usage)
    assert IntegrationStatistics.from_dict(s.to_dict()).to_dict() == s.to_dict()


# -- circuit threshold parametrized -----------------------------------------
@pytest.mark.parametrize("threshold", [1, 2, 3, 5])
def test_circuit_threshold_param(threshold):
    layer = il.EnterpriseIntegrationLayer(clock=il.LogicalClock(), circuit_threshold=threshold)
    layer.register_module(desc("a", ModuleType.PREDICTION), fail_adapter)
    for _ in range(threshold):
        layer.dispatch(direct_req("a"))
    assert layer.health("a").circuit_state is CircuitState.OPEN