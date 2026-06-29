"""Enterprise test suite for the Week 10 Phase 2 Business Process Orchestrator.

Standard pytest (parametrize / raises / approx only - no fixtures). A small
bootstrap resolves both the orchestrator and the composed Phase 1 engine in the
repository layout and in isolated execution.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from dataclasses import FrozenInstanceError

import pytest

# --- import bootstrap -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "workflow"))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "orchestration"))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:
    wfe = importlib.import_module("src.workflow.workflow_engine")
except ModuleNotFoundError:
    wfe = importlib.import_module("workflow_engine")

try:
    bpo = importlib.import_module("src.orchestration.business_process_orchestrator")
except ModuleNotFoundError:
    bpo = importlib.import_module("business_process_orchestrator")

globals().update({name: getattr(bpo, name) for name in bpo.__all__})


# ---------------------------------------------------------------------------
# Helpers (no fixtures)
# ---------------------------------------------------------------------------
def simple_workflow(workflow_id, action_type=None):
    action = wfe.WorkflowAction.create(
        action_type or wfe.ActionType.GENERATE_AUDIT_RECORD, "a", {})
    wstep = wfe.WorkflowStep("s", "s", action)
    return wfe.WorkflowDefinition(
        workflow_id, "wf", wfe.WorkflowType.INSPECTION, (wstep,))


def make_orchestrator(workflow_ids=("wf-a", "wf-b", "wf-c", "wf-d"), failing=()):
    orch = bpo.create_default_orchestrator()
    ids = list(dict.fromkeys(list(workflow_ids) + list(failing)))
    for wid in ids:
        if wid in failing:
            orch.workflow_engine.action_engine.register_handler(
                wfe.ActionType.NO_OP, lambda a, c, k: wfe.ActionOutcome.failed("forced"))
            orch.workflow_engine.registry.register_workflow(
                simple_workflow(wid, wfe.ActionType.NO_OP))
        else:
            orch.workflow_engine.registry.register_workflow(simple_workflow(wid))
    return orch


def step(step_id, ref="wf-a", **kw):
    return bpo.ProcessStep(step_id, step_id, ref, **kw)


def linear_process(process_id="P", n=3, sla=None):
    steps = tuple(step(f"s{i}", "wf-a", estimated_duration=10.0, resource_cost=1.0,
                        risk_weight=0.1) for i in range(n))
    deps = tuple(bpo.Dependency(f"s{i}", f"s{i+1}") for i in range(n - 1))
    return bpo.BusinessProcess(process_id, "Linear", steps, deps, sla=sla)


def parallel_process(process_id="PP"):
    steps = (
        step("root", "wf-a", estimated_duration=5.0),
        step("a", "wf-b", estimated_duration=10.0),
        step("b", "wf-c", estimated_duration=20.0),
        step("join", "wf-d", estimated_duration=5.0),
    )
    deps = (
        bpo.Dependency("root", "a"), bpo.Dependency("root", "b"),
        bpo.Dependency("a", "join"), bpo.Dependency("b", "join"),
    )
    return bpo.BusinessProcess(process_id, "Parallel", steps, deps)


def approval_process(process_id="AP"):
    chain = bpo.default_approval_chain(
        "chain", roles=(bpo.ApprovalRole.MANAGER, bpo.ApprovalRole.DIRECTOR), sla_hours=2.0)
    steps = (
        step("prep", "wf-a", estimated_duration=10.0),
        step("gate", "wf-b", estimated_duration=5.0, requires_approval=True,
             approval_chain_id="chain"),
        step("done", "wf-c", estimated_duration=10.0, compensation_step_id="prep"),
    )
    deps = (bpo.Dependency("prep", "gate"), bpo.Dependency("gate", "done"))
    return bpo.BusinessProcess(process_id, "Approval", steps, deps,
                               approval_chains=(chain,))


ALL_PROCESS_STATES = list(ProcessState)
ALL_STAGE_TYPES = list(StageType)
ALL_ROLES = list(ApprovalRole)
ALL_APPROVAL_STATES = list(ApprovalState)
ALL_SLA_STATUSES = list(SLAStatus)
ALL_SIM_MODES = list(SimulationMode)
ALL_ROLLBACK = list(RollbackStrategy)
ALL_DEP_TYPES = list(DependencyType)


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enum_cls,value", (
    [(ProcessState, e.value) for e in ProcessState]
    + [(StageType, e.value) for e in StageType]
    + [(ApprovalRole, e.value) for e in ApprovalRole]
    + [(ApprovalState, e.value) for e in ApprovalState]
    + [(SLAStatus, e.value) for e in SLAStatus]
    + [(SimulationMode, e.value) for e in SimulationMode]
    + [(RollbackStrategy, e.value) for e in RollbackStrategy]
    + [(DependencyType, e.value) for e in DependencyType]
))
def test_enum_coerce(enum_cls, value):
    assert enum_cls.coerce(value).value == value


@pytest.mark.parametrize("enum_cls", [
    ProcessState, StageType, ApprovalRole, ApprovalState, SLAStatus,
    SimulationMode, RollbackStrategy, DependencyType,
])
def test_enum_coerce_invalid(enum_cls):
    with pytest.raises(ProcessValidationError):
        enum_cls.coerce("__nope__")


@pytest.mark.parametrize("role,level", [
    (ApprovalRole.ENGINEER, 0), (ApprovalRole.LEAD, 1), (ApprovalRole.MANAGER, 2),
    (ApprovalRole.DIRECTOR, 3), (ApprovalRole.VICE_PRESIDENT, 4), (ApprovalRole.EXECUTIVE, 5),
])
def test_role_levels(role, level):
    assert role.level == level


@pytest.mark.parametrize("state,terminal", [
    (ProcessState.COMPLETED, True), (ProcessState.FAILED, True),
    (ProcessState.CANCELLED, True), (ProcessState.ROLLED_BACK, True),
    (ProcessState.COMPENSATED, True), (ProcessState.SIMULATED, True),
    (ProcessState.DRAFT, False), (ProcessState.RUNNING, False),
    (ProcessState.PAUSED, False), (ProcessState.PLANNED, False),
])
def test_process_state_terminal(state, terminal):
    assert state.is_terminal is terminal


# ---------------------------------------------------------------------------
# ProcessStep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {}, {"estimated_duration": 5.0}, {"resource_cost": 3.0}, {"risk_weight": 0.5},
    {"retryable": False}, {"compensation_step_id": "x"}, {"metadata_json": '{"k":1}'},
])
def test_process_step_roundtrip(kw):
    s = step("s1", "wf-a", **kw)
    assert ProcessStep.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_process_step_requires_chain_when_approval():
    with pytest.raises(ProcessValidationError):
        step("s1", "wf-a", requires_approval=True)


@pytest.mark.parametrize("kw", [
    {"step_id": ""}, {"workflow_ref": ""}, {"estimated_duration": -1.0},
    {"resource_cost": -1.0}, {"risk_weight": 1.5}, {"risk_weight": -0.1},
])
def test_process_step_validation(kw):
    base = {"step_id": "s", "name": "s", "workflow_ref": "wf-a"}
    base.update(kw)
    with pytest.raises(ProcessValidationError):
        ProcessStep(**base)


def test_process_step_automated_flag():
    assert step("s", "wf-a").automated is True
    s = step("s", "wf-a", requires_approval=True, approval_chain_id="c")
    assert s.automated is False


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ALL_DEP_TYPES)
def test_dependency_roundtrip(dtype):
    d = bpo.Dependency("a", "b", dtype)
    assert Dependency.from_dict(d.to_dict()).to_dict() == d.to_dict()


def test_dependency_self_loop():
    with pytest.raises(DependencyError):
        bpo.Dependency("a", "a")


def test_dependency_empty_endpoint():
    with pytest.raises(DependencyError):
        bpo.Dependency("", "b")


# ---------------------------------------------------------------------------
# ApprovalStep / ApprovalChain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ALL_ROLES)
def test_approval_step_roundtrip(role):
    s = bpo.ApprovalStep(role, "id", 12.0)
    assert ApprovalStep.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_approval_step_validation():
    with pytest.raises(ApprovalError):
        bpo.ApprovalStep(ApprovalRole.LEAD, "", 1.0)
    with pytest.raises(ApprovalError):
        bpo.ApprovalStep(ApprovalRole.LEAD, "id", -1.0)


def test_approval_chain_roundtrip():
    chain = bpo.default_approval_chain()
    assert ApprovalChain.from_dict(chain.to_dict()).to_dict() == chain.to_dict()


def test_approval_chain_requires_ascending_roles():
    with pytest.raises(ApprovalError):
        bpo.ApprovalChain("c", (
            bpo.ApprovalStep(bpo.ApprovalRole.DIRECTOR, "d"),
            bpo.ApprovalStep(bpo.ApprovalRole.MANAGER, "m"),
        ))


def test_approval_chain_empty():
    with pytest.raises(ApprovalError):
        bpo.ApprovalChain("c", ())


def test_approval_chain_modeled_latency():
    chain = bpo.default_approval_chain(roles=(bpo.ApprovalRole.LEAD, bpo.ApprovalRole.MANAGER),
                                       sla_hours=2.0)
    assert chain.modeled_latency_seconds == pytest.approx(4.0 * 3600.0)


def test_default_approval_chain_full_hierarchy():
    chain = bpo.default_approval_chain()
    assert [s.role for s in chain.steps] == ALL_ROLES


# ---------------------------------------------------------------------------
# SLAConfig
# ---------------------------------------------------------------------------
def test_sla_config_roundtrip():
    c = bpo.SLAConfig(100.0, 0.7, 1.1, 0.5, 20.0)
    assert SLAConfig.from_dict(c.to_dict()).to_dict() == c.to_dict()


@pytest.mark.parametrize("kw", [
    {"expected_duration": 0.0}, {"expected_duration": -5.0},
    {"warning_threshold": 0.0}, {"warning_threshold": 1.5},
    {"escalation_threshold": 0.5}, {"penalty_per_second": -1.0},
])
def test_sla_config_validation(kw):
    base = {"expected_duration": 100.0}
    base.update(kw)
    with pytest.raises(SLAError):
        bpo.SLAConfig(**base)


# ---------------------------------------------------------------------------
# BusinessCalendar
# ---------------------------------------------------------------------------
def test_calendar_roundtrip():
    cal = bpo.BusinessCalendar(timezone_offset_minutes=120, holidays=("2026-01-01",),
                               blackout_periods=((0.0, 100.0),))
    assert BusinessCalendar.from_dict(cal.to_dict()).to_dict() == cal.to_dict()


def test_calendar_always_on_is_business_time():
    cal = bpo.BusinessCalendar.always_on()
    assert cal.is_business_time(0.0) is True
    assert cal.is_business_time(1_000_000.0) is True


def test_calendar_weekend_blocked():
    cal = bpo.BusinessCalendar(business_start_hour=0, business_end_hour=24)
    saturday = 2 * 86400.0  # 1970-01-03 is a Saturday
    assert cal.is_business_time(saturday) is False


def test_calendar_business_hours():
    cal = bpo.BusinessCalendar(business_start_hour=9, business_end_hour=17)
    assert cal.is_business_time(3 * 3600.0) is False
    assert cal.is_business_time(10 * 3600.0) is True


def test_calendar_holiday_blocked():
    cal = bpo.BusinessCalendar(business_start_hour=0, business_end_hour=24,
                               working_days=(0, 1, 2, 3, 4, 5, 6), holidays=("1970-01-01",))
    assert cal.is_business_time(10 * 3600.0) is False


def test_calendar_blackout_blocked():
    cal = bpo.BusinessCalendar(business_start_hour=0, business_end_hour=24,
                               working_days=tuple(range(7)), blackout_periods=((0.0, 7200.0),))
    assert cal.is_business_time(3600.0) is False
    assert cal.is_business_time(8000.0) is True


def test_calendar_emergency_mode_ignores_hours():
    cal = bpo.BusinessCalendar(emergency_mode=True)
    assert cal.is_business_time(3 * 3600.0) is True


def test_calendar_advance_always_on():
    cal = bpo.BusinessCalendar.always_on()
    assert cal.advance_business_seconds(0.0, 50.0) == pytest.approx(50.0)


def test_calendar_invalid_hours():
    with pytest.raises(OrchestratorError):
        bpo.BusinessCalendar(business_start_hour=18, business_end_hour=9)


def test_calendar_next_business_time_skips_weekend():
    cal = bpo.BusinessCalendar(business_start_hour=0, business_end_hour=24)
    saturday = 2 * 86400.0
    nxt = cal.next_business_time(saturday)
    assert cal.is_business_time(nxt)


def test_calendar_maintenance_window():
    cal = bpo.BusinessCalendar(maintenance_windows=((100.0, 200.0),))
    assert cal.in_maintenance_window(150.0) is True
    assert cal.in_maintenance_window(250.0) is False


# ---------------------------------------------------------------------------
# BusinessProcess
# ---------------------------------------------------------------------------
def test_business_process_roundtrip():
    proc = approval_process()
    assert BusinessProcess.from_dict(proc.to_dict()).to_dict() == proc.to_dict()


def test_business_process_roundtrip_with_sla_calendar():
    proc = bpo.BusinessProcess(
        "P", "n", (step("a", "wf-a"),), sla=bpo.SLAConfig(100.0),
        calendar=bpo.BusinessCalendar.always_on())
    assert BusinessProcess.from_dict(proc.to_dict()).to_dict() == proc.to_dict()


def test_business_process_requires_steps():
    with pytest.raises(ProcessValidationError):
        bpo.BusinessProcess("P", "n", ())


def test_business_process_duplicate_step():
    with pytest.raises(ProcessValidationError):
        bpo.BusinessProcess("P", "n", (step("a"), step("a")))


def test_business_process_unknown_dependency():
    with pytest.raises(DependencyError):
        bpo.BusinessProcess("P", "n", (step("a"),), (bpo.Dependency("a", "ghost"),))


def test_business_process_unknown_chain():
    with pytest.raises(ProcessValidationError):
        bpo.BusinessProcess("P", "n",
                            (step("a", requires_approval=True, approval_chain_id="missing"),))


def test_business_process_cycle_rejected():
    with pytest.raises(CycleError):
        bpo.BusinessProcess("P", "n", (step("a"), step("b")),
                            (bpo.Dependency("a", "b"), bpo.Dependency("b", "a")))


def test_business_process_unknown_compensation():
    with pytest.raises(ProcessValidationError):
        bpo.BusinessProcess("P", "n", (step("a", compensation_step_id="ghost"),))


def test_business_process_get_step_and_chain():
    proc = approval_process()
    assert proc.get_step("gate").step_id == "gate"
    assert proc.get_chain("chain").chain_id == "chain"


def test_business_process_get_step_unknown():
    with pytest.raises(ProcessValidationError):
        linear_process().get_step("nope")


def test_business_process_fingerprint_stable():
    assert linear_process().fingerprint() == linear_process().fingerprint()


def test_business_process_step_ids():
    assert linear_process(n=3).step_ids == ("s0", "s1", "s2")


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------
def graph_from(nodes, edges):
    return bpo.DependencyGraph(nodes, edges)


def test_graph_topological_sort_linear():
    g = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert g.topological_sort() == ("a", "b", "c")


def test_graph_topological_sort_deterministic():
    g = graph_from(["a", "b", "c", "d"], [("a", "c"), ("b", "c"), ("c", "d")])
    assert g.topological_sort() == g.topological_sort()


def test_graph_cycle_detection():
    g = graph_from(["a", "b"], [("a", "b"), ("b", "a")])
    assert g.has_cycle() is True
    with pytest.raises(CycleError):
        g.topological_sort()


def test_graph_no_cycle():
    g = graph_from(["a", "b", "c"], [("a", "b"), ("a", "c")])
    assert g.has_cycle() is False


def test_graph_self_cycle():
    g = graph_from(["a"], [("a", "a")])
    assert g.has_cycle() is True


def test_graph_roots_and_leaves():
    g = graph_from(["a", "b", "c"], [("a", "b"), ("a", "c")])
    assert g.roots() == ("a",)
    assert set(g.leaves()) == {"b", "c"}


def test_graph_layered_order():
    g = graph_from(["root", "a", "b", "join"],
                   [("root", "a"), ("root", "b"), ("a", "join"), ("b", "join")])
    layers = g.layered_order()
    assert layers[0] == ("root",)
    assert set(layers[1]) == {"a", "b"}
    assert layers[2] == ("join",)


def test_graph_layered_cycle_raises():
    g = graph_from(["a", "b"], [("a", "b"), ("b", "a")])
    with pytest.raises(CycleError):
        g.layered_order()


def test_graph_descendants_ancestors():
    g = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert g.descendants("a") == frozenset({"b", "c"})
    assert g.ancestors("c") == frozenset({"a", "b"})


def test_graph_blocking_nodes():
    g = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    blocking = g.blocking_nodes()
    assert blocking[0] == "a"  # blocks the most descendants


def test_graph_connected_components():
    g = graph_from(["a", "b", "c", "d"], [("a", "b")])
    comps = g.connected_components()
    assert len(comps) == 3  # {a,b}, {c}, {d}


def test_graph_is_connected():
    g = graph_from(["a", "b"], [("a", "b")])
    assert g.is_connected() is True


def test_graph_is_disconnected():
    g = graph_from(["a", "b", "c"], [("a", "b")])
    assert g.is_disconnected() is True


@pytest.mark.parametrize("durations,expected_len,expected_dur", [
    ({"a": 1.0, "b": 2.0, "c": 3.0}, 3, 6.0),
    ({"a": 5.0, "b": 1.0, "c": 1.0}, 3, 7.0),
])
def test_graph_critical_path_linear(durations, expected_len, expected_dur):
    g = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    path, dur = g.critical_path(durations)
    assert len(path) == expected_len
    assert dur == pytest.approx(expected_dur)


def test_graph_critical_path_diamond():
    g = graph_from(["root", "a", "b", "join"],
                   [("root", "a"), ("root", "b"), ("a", "join"), ("b", "join")])
    durations = {"root": 1.0, "a": 2.0, "b": 10.0, "join": 1.0}
    path, dur = g.critical_path(durations)
    assert dur == pytest.approx(12.0)
    assert "b" in path


def test_graph_duplicate_node_rejected():
    with pytest.raises(DependencyError):
        graph_from(["a", "a"], [])


def test_graph_edge_unknown_node():
    with pytest.raises(DependencyError):
        graph_from(["a"], [("a", "ghost")])


def test_graph_large_chain_topological_sort():
    n = 150
    nodes = [f"n{i}" for i in range(n)]
    edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
    g = graph_from(nodes, edges)
    order = g.topological_sort()
    assert len(order) == n
    assert order[0] == "n0" and order[-1] == f"n{n-1}"


def test_graph_large_critical_path():
    n = 120
    nodes = [f"n{i}" for i in range(n)]
    edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
    g = graph_from(nodes, edges)
    durations = {node: 1.0 for node in nodes}
    path, dur = g.critical_path(durations)
    assert dur == pytest.approx(float(n))


# ---------------------------------------------------------------------------
# ProcessRegistry
# ---------------------------------------------------------------------------
def test_registry_register_lookup_exists():
    reg = bpo.ProcessRegistry()
    proc = linear_process("R1")
    reg.register(proc)
    assert reg.exists("R1")
    assert reg.lookup("R1").process_id == "R1"
    assert "R1" in reg and len(reg) == 1


def test_registry_duplicate():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    with pytest.raises(ProcessRegistryError):
        reg.register(linear_process("R1"))


def test_registry_overwrite():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    reg.register(linear_process("R1", n=2), overwrite=True)
    assert len(reg.lookup("R1").steps) == 2


def test_registry_remove():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    reg.remove("R1")
    assert not reg.exists("R1")


def test_registry_remove_unknown():
    with pytest.raises(ProcessRegistryError):
        bpo.ProcessRegistry().remove("nope")


def test_registry_lookup_unknown():
    with pytest.raises(ProcessRegistryError):
        bpo.ProcessRegistry().lookup("nope")


def test_registry_list_sorted():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("B"))
    reg.register(linear_process("A"))
    assert [p.process_id for p in reg.list()] == ["A", "B"]


def test_registry_find():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("A", n=2))
    reg.register(linear_process("B", n=5))
    found = reg.find(lambda p: len(p.steps) >= 5)
    assert [p.process_id for p in found] == ["B"]


def test_registry_freeze():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    frozen = reg.freeze("R1")
    assert frozen.frozen is True
    assert reg.lookup("R1").frozen is True


def test_registry_remove_frozen_blocked():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    reg.freeze("R1")
    with pytest.raises(ProcessRegistryError):
        reg.remove("R1")


def test_registry_update_version():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    updated = reg.update_version(linear_process("R1", n=4))
    assert updated.version == 2
    assert len(reg.lookup("R1").steps) == 4


def test_registry_update_frozen_blocked():
    reg = bpo.ProcessRegistry()
    reg.register(linear_process("R1"))
    reg.freeze("R1")
    with pytest.raises(ProcessRegistryError):
        reg.update_version(linear_process("R1"))


def test_registry_update_unknown():
    with pytest.raises(ProcessRegistryError):
        bpo.ProcessRegistry().update_version(linear_process("ghost"))


def test_registry_register_non_process():
    with pytest.raises(ProcessValidationError):
        bpo.ProcessRegistry().register("not-a-process")


def test_registry_thread_safety():
    reg = bpo.ProcessRegistry()
    errors = []

    def worker(i):
        try:
            reg.register(linear_process(f"P{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(reg) == 50


# ---------------------------------------------------------------------------
# ExecutionPlanner
# ---------------------------------------------------------------------------
def test_plan_sequential_groups():
    plan = bpo.ExecutionPlanner().plan(linear_process(n=3))
    work = plan.work_groups
    assert len(work) == 3
    assert all(g.stage_type is StageType.SEQUENTIAL for g in work)
    assert plan.makespan == pytest.approx(30.0)


def test_plan_parallel_group():
    plan = bpo.ExecutionPlanner().plan(parallel_process())
    parallel = [g for g in plan.groups if g.stage_type is StageType.PARALLEL]
    assert len(parallel) == 1
    assert set(parallel[0].step_ids) == {"a", "b"}
    # makespan = root(5) + max(a=10,b=20) + join(5) = 30
    assert plan.makespan == pytest.approx(30.0)


def test_plan_critical_path():
    plan = bpo.ExecutionPlanner().plan(parallel_process())
    assert plan.critical_path_duration == pytest.approx(30.0)
    assert "b" in plan.critical_path


def test_plan_inserts_waiting_and_approval():
    plan = bpo.ExecutionPlanner().plan(approval_process())
    types = [g.stage_type for g in plan.groups]
    assert StageType.WAITING in types
    assert StageType.APPROVAL in types


def test_plan_includes_rollback_contingency():
    plan = bpo.ExecutionPlanner().plan(linear_process())
    rollback = [g for g in plan.groups if g.stage_type is StageType.ROLLBACK]
    assert len(rollback) == 1
    assert rollback[0].contingency is True


def test_plan_includes_compensation_when_present():
    plan = bpo.ExecutionPlanner().plan(approval_process())
    comp = [g for g in plan.groups if g.stage_type is StageType.COMPENSATION]
    assert len(comp) == 1


def test_plan_no_compensation_when_absent():
    plan = bpo.ExecutionPlanner().plan(linear_process())
    comp = [g for g in plan.groups if g.stage_type is StageType.COMPENSATION]
    assert comp == []


def test_plan_resource_estimate():
    plan = bpo.ExecutionPlanner().plan(linear_process(n=3))
    assert plan.resource_estimate == pytest.approx(3.0)


def test_plan_risk_estimate_range():
    plan = bpo.ExecutionPlanner().plan(linear_process(n=3))
    assert 0.0 <= plan.risk_estimate <= 1.0


def test_plan_parallelism_ratio():
    seq = bpo.ExecutionPlanner().plan(linear_process(n=4))
    par = bpo.ExecutionPlanner().plan(parallel_process())
    assert seq.parallelism_ratio == pytest.approx(0.0)
    assert par.parallelism_ratio > 0.0


def test_plan_deterministic():
    p1 = bpo.ExecutionPlanner().plan(parallel_process())
    p2 = bpo.ExecutionPlanner().plan(parallel_process())
    assert p1.to_dict() == p2.to_dict()


def test_plan_roundtrip():
    plan = bpo.ExecutionPlanner().plan(parallel_process())
    assert ExecutionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_plan_topological_order():
    plan = bpo.ExecutionPlanner().plan(linear_process(n=3))
    assert plan.topological_order == ("s0", "s1", "s2")


def test_plan_completion_with_calendar():
    proc = bpo.BusinessProcess("P", "n", (step("a", "wf-a", estimated_duration=100.0),),
                               calendar=bpo.BusinessCalendar.always_on())
    plan = bpo.ExecutionPlanner().plan(proc, start_time=0.0)
    assert plan.estimated_completion_time == pytest.approx(100.0)


def test_plan_large_process():
    proc = linear_process(n=120)
    plan = bpo.ExecutionPlanner().plan(proc)
    assert len(plan.work_groups) == 120
    assert plan.makespan == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# ExecutionGroup serialization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ALL_STAGE_TYPES)
def test_execution_group_roundtrip(stage):
    g = bpo.ExecutionGroup("g1", stage, ("a", "b"), 5.0, 0)
    assert ExecutionGroup.from_dict(g.to_dict()).to_dict() == g.to_dict()


# ---------------------------------------------------------------------------
# ApprovalEngine
# ---------------------------------------------------------------------------
def two_step_chain():
    return bpo.default_approval_chain(
        "c", roles=(bpo.ApprovalRole.MANAGER, bpo.ApprovalRole.DIRECTOR), sla_hours=1.0)


def test_approval_start():
    eng = bpo.ApprovalEngine()
    state = eng.start(two_step_chain())
    assert state.state is ApprovalState.PENDING and state.index == 0


def test_approval_full_approve():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.start(chain)
    state = eng.approve(chain, state, approver_id="m")
    assert state.state is ApprovalState.PENDING
    state = eng.approve(chain, state, approver_id="d")
    assert state.state is ApprovalState.APPROVED
    assert len(state.records) == 2


def test_approval_reject():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.reject(chain, eng.start(chain), approver_id="m")
    assert state.state is ApprovalState.REJECTED


def test_approval_delegate_keeps_index():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.start(chain)
    state = eng.delegate(chain, state, to_approver_id="other")
    assert state.index == 0
    assert state.records[-1].state is ApprovalState.DELEGATED


def test_approval_delegate_requires_target():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    with pytest.raises(ApprovalError):
        eng.delegate(chain, eng.start(chain), to_approver_id="")


def test_approval_escalate_advances():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.escalate(chain, eng.start(chain))
    assert state.index == 1
    assert state.records[-1].state is ApprovalState.ESCALATED


def test_approval_timeout_escalates():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.timeout(chain, eng.start(chain))
    assert any(r.state is ApprovalState.TIMED_OUT for r in state.records)
    assert any(r.state is ApprovalState.ESCALATED for r in state.records)


def test_approval_run_full_approve():
    eng = bpo.ApprovalEngine()
    state = eng.run_full(two_step_chain())
    assert state.state is ApprovalState.APPROVED


def test_approval_run_full_reject():
    eng = bpo.ApprovalEngine()
    state = eng.run_full(two_step_chain(), decision_provider=lambda c, s: "reject")
    assert state.state is ApprovalState.REJECTED


def test_approval_run_full_unknown_decision():
    eng = bpo.ApprovalEngine()
    with pytest.raises(ApprovalError):
        eng.run_full(two_step_chain(), decision_provider=lambda c, s: "maybe")


def test_approval_operate_after_complete_raises():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    state = eng.run_full(chain)
    with pytest.raises(ApprovalError):
        eng.approve(chain, state, approver_id="x")


def test_approval_latency_uses_clock():
    eng = bpo.ApprovalEngine()
    chain = two_step_chain()
    clock = wfe.LogicalClock()
    state = eng.start(chain, clock=clock)
    clock.advance(10.0)
    state = eng.approve(chain, state, approver_id="m", clock=clock)
    assert state.records[-1].latency == pytest.approx(10.0)


def test_approval_metrics():
    eng = bpo.ApprovalEngine()
    states = [eng.run_full(two_step_chain()) for _ in range(3)]
    metrics = eng.metrics(states)
    assert metrics["chains"] == 3
    assert metrics["approved_chains"] == 3
    assert metrics["records"] == 6


@pytest.mark.parametrize("astate", ALL_APPROVAL_STATES)
def test_approval_record_roundtrip(astate):
    rec = bpo.ApprovalRecord("c", bpo.ApprovalRole.MANAGER, "m", astate, 1.0, 0.5, "n")
    assert ApprovalRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()


def test_approval_chain_state_roundtrip():
    eng = bpo.ApprovalEngine()
    state = eng.run_full(two_step_chain())
    assert ApprovalChainState.from_dict(state.to_dict()).to_dict() == state.to_dict()


# ---------------------------------------------------------------------------
# SLAEngine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("actual,expected_status", [
    (50.0, SLAStatus.ON_TRACK),
    (85.0, SLAStatus.WARNING),
    (120.0, SLAStatus.VIOLATED),
])
def test_sla_evaluate_status(actual, expected_status):
    cfg = bpo.SLAConfig(expected_duration=100.0, warning_threshold=0.8, escalation_threshold=1.0)
    report = bpo.SLAEngine().evaluate(cfg, actual)
    assert report.status is expected_status


def test_sla_not_applicable():
    report = bpo.SLAEngine().evaluate(None, 100.0)
    assert report.status is SLAStatus.NOT_APPLICABLE


def test_sla_recovered():
    cfg = bpo.SLAConfig(expected_duration=100.0)
    report = bpo.SLAEngine().evaluate(cfg, 150.0, recovered=True)
    assert report.status is SLAStatus.RECOVERED


def test_sla_penalty():
    cfg = bpo.SLAConfig(expected_duration=100.0, penalty_per_second=2.0)
    report = bpo.SLAEngine().evaluate(cfg, 130.0)
    assert report.penalty == pytest.approx(60.0)


def test_sla_delay_and_remaining():
    cfg = bpo.SLAConfig(expected_duration=100.0)
    report = bpo.SLAEngine().evaluate(cfg, 130.0)
    assert report.delay == pytest.approx(30.0)
    assert report.remaining == pytest.approx(-30.0)


def test_sla_report_roundtrip():
    cfg = bpo.SLAConfig(expected_duration=100.0)
    report = bpo.SLAEngine().evaluate(cfg, 90.0)
    assert SLAReport.from_dict(report.to_dict()).to_dict() == report.to_dict()


def test_sla_compliance_pct():
    cfg = bpo.SLAConfig(expected_duration=100.0)
    eng = bpo.SLAEngine()
    reports = [eng.evaluate(cfg, 50.0), eng.evaluate(cfg, 200.0)]
    assert eng.compliance_pct(reports) == pytest.approx(0.5)


def test_sla_compliance_pct_empty():
    assert bpo.SLAEngine().compliance_pct([]) == 0.0


# ---------------------------------------------------------------------------
# RollbackEngine
# ---------------------------------------------------------------------------
def test_rollback_plan_undo_order():
    proc = linear_process(n=3)
    plan = bpo.RollbackEngine().rollback_plan(proc, "s2", ["s0", "s1"])
    assert plan.undo_chain == ("s1", "s0")
    assert plan.failed_step == "s2"


def test_rollback_plan_isolated_steps():
    proc = linear_process(n=4)
    plan = bpo.RollbackEngine().rollback_plan(proc, "s1", ["s0"])
    assert plan.isolated_steps[0] == "s1"
    assert "s2" in plan.isolated_steps and "s3" in plan.isolated_steps


def test_rollback_plan_recovery_chain():
    proc = linear_process(n=3)
    plan = bpo.RollbackEngine().rollback_plan(proc, "s0", [])
    assert plan.recovery_chain == ("s0", "s1", "s2")


def test_rollback_plan_unknown_step():
    with pytest.raises(RollbackError):
        bpo.RollbackEngine().rollback_plan(linear_process(), "ghost", [])


@pytest.mark.parametrize("strategy", ALL_ROLLBACK)
def test_rollback_plan_roundtrip(strategy):
    proc = linear_process(n=3)
    plan = bpo.RollbackEngine().rollback_plan(proc, "s1", ["s0"], strategy=strategy)
    assert RollbackPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_compensation_plan():
    proc = approval_process()
    plan = bpo.RollbackEngine().compensation_plan(proc)
    assert plan.compensations == (("done", "prep"),)


def test_compensation_plan_roundtrip():
    proc = approval_process()
    plan = bpo.RollbackEngine().compensation_plan(proc)
    assert CompensationPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_compensation_plan_empty():
    plan = bpo.RollbackEngine().compensation_plan(linear_process())
    assert plan.compensations == ()


# ---------------------------------------------------------------------------
# Orchestrator: execution
# ---------------------------------------------------------------------------
def test_orchestrator_register_and_plan():
    orch = make_orchestrator()
    proc = linear_process()
    orch.register_process(proc)
    plan = orch.plan("P")
    assert plan.process_id == "P"


def test_orchestrator_execute_success():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    assert res.state is ProcessState.COMPLETED
    assert len(res.step_results) == 3
    assert res.sla_status is SLAStatus.ON_TRACK


def test_orchestrator_execute_with_approval():
    orch = make_orchestrator()
    res = orch.execute(approval_process())
    assert res.state is ProcessState.COMPLETED
    assert len(res.approval_records) == 2


def test_orchestrator_execute_rejection_fails():
    orch = make_orchestrator()
    res = orch.execute(approval_process(), decision_provider=lambda c, s: "reject")
    assert res.state is ProcessState.FAILED
    assert "approval rejected" in res.error


def test_orchestrator_execute_workflow_failure_rolls_back():
    orch = make_orchestrator(failing=("wf-b",))
    proc = bpo.BusinessProcess("FP", "n", (
        step("ok", "wf-a", estimated_duration=5.0),
        step("bad", "wf-b", estimated_duration=5.0, compensation_step_id="ok"),
    ), (bpo.Dependency("ok", "bad"),))
    res = orch.execute(proc)
    assert res.state is ProcessState.ROLLED_BACK
    assert res.rolled_back is True
    assert res.compensated is True


def test_orchestrator_execute_failure_no_rollback():
    orch = make_orchestrator(failing=("wf-b",))
    orch.enable_rollback = False
    proc = bpo.BusinessProcess("FP", "n", (step("bad", "wf-b", estimated_duration=5.0),))
    res = orch.execute(proc)
    assert res.state is ProcessState.FAILED
    assert res.rolled_back is False


def test_orchestrator_execute_deterministic():
    orch = make_orchestrator()
    proc = parallel_process()
    r1 = orch.execute(proc)
    r2 = orch.execute(proc)
    assert r1.to_dict() == r2.to_dict()


def test_orchestrator_execute_result_roundtrip():
    orch = make_orchestrator()
    res = orch.execute(linear_process())
    assert ProcessExecutionResult.from_dict(res.to_dict()).to_dict() == res.to_dict()


def test_orchestrator_execute_unknown_workflow_ref():
    orch = bpo.create_default_orchestrator()  # no workflows registered
    with pytest.raises(PlanningError):
        orch.execute(linear_process())


def test_orchestrator_execute_actual_duration():
    orch = make_orchestrator()
    res = orch.execute(parallel_process())
    # makespan = 5 + max(10,20) + 5 = 30
    assert res.actual_duration == pytest.approx(30.0)


def test_orchestrator_execute_via_registry_id():
    orch = make_orchestrator()
    orch.register_process(linear_process("RID"))
    res = orch.execute("RID")
    assert res.process_id == "RID"


def test_orchestrator_automated_manual_counts():
    orch = make_orchestrator()
    res = orch.execute(approval_process())
    assert res.manual_steps == 1
    assert res.automated_steps == 2


# ---------------------------------------------------------------------------
# Orchestrator: simulation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [
    SimulationMode.DRY_RUN, SimulationMode.SIMULATION, SimulationMode.WHAT_IF,
    SimulationMode.REPLAY, SimulationMode.RECOVERY,
])
def test_orchestrator_simulate_modes(mode):
    orch = make_orchestrator()
    sim = orch.simulate(parallel_process(), mode=mode)
    assert sim.mode is mode
    assert sim.projected_duration == pytest.approx(30.0)


def test_orchestrator_simulate_live_raises():
    orch = make_orchestrator()
    with pytest.raises(SimulationError):
        orch.simulate(linear_process(), mode=SimulationMode.LIVE)


def test_orchestrator_simulate_success_probability():
    orch = make_orchestrator()
    proc = linear_process(n=3)  # each risk 0.1 -> 0.9^3
    sim = orch.simulate(proc)
    assert sim.projected_success_probability == pytest.approx(0.9 ** 3)


def test_orchestrator_simulate_whatif_overrides():
    orch = make_orchestrator()
    proc = linear_process(n=2)
    sim = orch.simulate(proc, mode=SimulationMode.WHAT_IF,
                        overrides={"s0": {"estimated_duration": 100.0}})
    assert sim.projected_duration == pytest.approx(110.0)


def test_orchestrator_simulate_roundtrip():
    orch = make_orchestrator()
    sim = orch.simulate(parallel_process())
    assert SimulationResult.from_dict(sim.to_dict()).to_dict() == sim.to_dict()


def test_orchestrator_simulate_deterministic():
    orch = make_orchestrator()
    s1 = orch.simulate(parallel_process())
    s2 = orch.simulate(parallel_process())
    assert s1.to_dict() == s2.to_dict()


# ---------------------------------------------------------------------------
# Orchestrator: rollback / compensation passthrough
# ---------------------------------------------------------------------------
def test_orchestrator_rollback_plan():
    orch = make_orchestrator()
    plan = orch.rollback_plan(linear_process(n=3), "s2", ["s0", "s1"])
    assert plan.undo_chain == ("s1", "s0")


def test_orchestrator_compensation_plan():
    orch = make_orchestrator()
    plan = orch.compensation_plan(approval_process())
    assert plan.compensations == (("done", "prep"),)


def test_orchestrator_dependency_graph():
    orch = make_orchestrator()
    g = orch.dependency_graph(linear_process(n=3))
    assert g.topological_sort() == ("s0", "s1", "s2")


# ---------------------------------------------------------------------------
# KPIs and analytics
# ---------------------------------------------------------------------------
def test_kpis_empty():
    kpis = bpo.BusinessKPIs.from_results([])
    assert kpis.process_success_rate == 0.0


def test_kpis_all_success():
    orch = make_orchestrator()
    results = [orch.execute(linear_process(f"P{i}", sla=bpo.SLAConfig(expected_duration=1000.0)))
               for i in range(3)]
    kpis = orch.kpis(results)
    assert kpis.process_success_rate == pytest.approx(1.0)
    assert kpis.failure_rate == pytest.approx(0.0)
    assert kpis.sla_pct == pytest.approx(1.0)


def test_kpis_mixed():
    orch = make_orchestrator(failing=("wf-x",))
    good = orch.execute(linear_process("G", sla=bpo.SLAConfig(expected_duration=1000.0)))
    bad_proc = bpo.BusinessProcess("B", "n", (step("bad", "wf-x", estimated_duration=5.0),))
    bad = orch.execute(bad_proc)
    kpis = orch.kpis([good, bad])
    assert kpis.process_success_rate == pytest.approx(0.5)
    assert kpis.rollback_pct == pytest.approx(0.5)


def test_kpis_automation_pct():
    orch = make_orchestrator()
    res = orch.execute(approval_process())
    kpis = orch.kpis([res])
    assert kpis.automation_pct == pytest.approx(2.0 / 3.0)
    assert kpis.manual_pct == pytest.approx(1.0 / 3.0)


def test_kpis_roundtrip():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    kpis = orch.kpis([res])
    assert BusinessKPIs.from_dict(kpis.to_dict()).to_dict() == kpis.to_dict()


def test_analytics_from_results():
    orch = make_orchestrator()
    results = [orch.execute(linear_process(f"P{i}", sla=bpo.SLAConfig(expected_duration=1000.0)))
               for i in range(4)]
    analytics = orch.analytics(results)
    assert analytics.total_processes == 4
    assert "completed" in analytics.to_dict()["state_breakdown"]


def test_analytics_dashboard_surface():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    dash = orch.analytics([res]).as_dashboard()
    assert "summary" in dash and "states" in dash and "durations" in dash


def test_analytics_api_surface():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    api = orch.analytics([res]).as_api()
    assert "analytics" in api and "kpis" in api


def test_analytics_dataframe_records():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    records = orch.analytics([res]).as_dataframe_records()
    assert isinstance(records, list) and all("metric" in r for r in records)


def test_analytics_roundtrip():
    orch = make_orchestrator()
    res = orch.execute(linear_process(sla=bpo.SLAConfig(expected_duration=1000.0)))
    analytics = orch.analytics([res])
    assert ProcessAnalytics.from_dict(analytics.to_dict()).to_dict() == analytics.to_dict()


def test_analytics_percentiles():
    orch = make_orchestrator()
    results = [orch.execute(linear_process(f"P{i}", n=i + 1,
                                            sla=bpo.SLAConfig(expected_duration=1000.0)))
               for i in range(3)]
    analytics = orch.analytics(results)
    assert analytics.duration_max >= analytics.duration_p50


# ---------------------------------------------------------------------------
# JSON compatibility
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("obj_factory", [
    lambda: step("s", "wf-a").to_dict(),
    lambda: bpo.Dependency("a", "b").to_dict(),
    lambda: bpo.default_approval_chain().to_dict(),
    lambda: bpo.SLAConfig(100.0).to_dict(),
    lambda: bpo.BusinessCalendar.always_on().to_dict(),
    lambda: linear_process().to_dict(),
    lambda: bpo.ExecutionPlanner().plan(parallel_process()).to_dict(),
])
def test_json_serializable(obj_factory):
    payload = obj_factory()
    text = json.dumps(payload)
    assert json.loads(text) == payload


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------
_FROZEN = [
    step("s", "wf-a"),
    bpo.Dependency("a", "b"),
    bpo.ApprovalStep(bpo.ApprovalRole.LEAD, "id"),
    bpo.default_approval_chain(),
    bpo.SLAConfig(100.0),
    bpo.BusinessCalendar.always_on(),
    linear_process(),
    bpo.ExecutionGroup("g", StageType.SEQUENTIAL, ("a",)),
]


@pytest.mark.parametrize("instance", _FROZEN)
def test_instances_are_frozen(instance):
    import dataclasses
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, getattr(instance, field_name))


@pytest.mark.parametrize("instance", _FROZEN)
def test_instances_have_slots(instance):
    assert not hasattr(instance, "__dict__")


# ---------------------------------------------------------------------------
# Integration / non-invasiveness
# ---------------------------------------------------------------------------
def test_orchestrator_reuses_phase1_engine():
    orch = make_orchestrator()
    assert isinstance(orch.workflow_engine, wfe.WorkflowEngine)


def test_module_does_not_modify_phase1_source():
    with open(wfe.__file__, "r", encoding="utf-8") as fh:
        before = fh.read()
    make_orchestrator().execute(linear_process())
    with open(wfe.__file__, "r", encoding="utf-8") as fh:
        after = fh.read()
    assert before == after


@pytest.mark.parametrize("forbidden", [
    "import pandas", "import networkx", "import scipy", "import airflow",
    "import temporal", "import prefect", "import celery",
])
def test_no_forbidden_imports(forbidden):
    with open(bpo.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert forbidden not in source


def test_factory_builds_orchestrator():
    assert isinstance(bpo.create_default_orchestrator(), bpo.BusinessProcessOrchestrator)


# ---------------------------------------------------------------------------
# Edge cases and scale
# ---------------------------------------------------------------------------
def test_single_step_process():
    orch = make_orchestrator()
    res = orch.execute(bpo.BusinessProcess("S", "n", (step("only", "wf-a", estimated_duration=3.0),)))
    assert res.state is ProcessState.COMPLETED


def test_disconnected_process_plan():
    proc = bpo.BusinessProcess("D", "n", (
        step("a", "wf-a", estimated_duration=1.0),
        step("b", "wf-b", estimated_duration=1.0),
    ))  # no dependencies -> disconnected
    graph = bpo.DependencyGraph.from_process(proc)
    assert graph.is_disconnected() is True
    plan = bpo.ExecutionPlanner().plan(proc)
    # both independent -> single parallel layer
    parallel = [g for g in plan.groups if g.stage_type is StageType.PARALLEL]
    assert len(parallel) == 1


def test_large_process_execute():
    orch = make_orchestrator()
    res = orch.execute(linear_process("BIG", n=100,
                                      sla=bpo.SLAConfig(expected_duration=100000.0)))
    assert res.state is ProcessState.COMPLETED
    assert len(res.step_results) == 100


def test_wide_parallel_process_plan():
    steps = tuple(step(f"p{i}", "wf-a", estimated_duration=float(i + 1)) for i in range(50))
    proc = bpo.BusinessProcess("WIDE", "n", steps)  # all independent
    plan = bpo.ExecutionPlanner().plan(proc)
    parallel = [g for g in plan.groups if g.stage_type is StageType.PARALLEL]
    assert len(parallel) == 1
    assert plan.makespan == pytest.approx(50.0)  # max single duration


def test_process_event_roundtrip():
    evt = bpo.ProcessEvent(0, 1.0, "P", "stage", "EVENT", "actor", '{"k":1}')
    assert ProcessEvent.from_dict(evt.to_dict()).to_dict() == evt.to_dict()


def test_process_history_append_immutable():
    h0 = bpo.ProcessHistory()
    evt = bpo.ProcessEvent(0, 0.0, "P", "", "E", "a")
    h1 = h0.append(evt)
    assert len(h0) == 0 and len(h1) == 1


def test_step_execution_record_roundtrip():
    rec = bpo.StepExecutionRecord("s", "wf-a", "ex", "completed", 0.0, 5.0, True)
    assert StepExecutionRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()
    assert rec.duration == pytest.approx(5.0)


@pytest.mark.parametrize("pstate", ALL_PROCESS_STATES)
def test_process_execution_result_roundtrip(pstate):
    res = bpo.ProcessExecutionResult("P", "ex", pstate, SimulationMode.LIVE)
    assert ProcessExecutionResult.from_dict(res.to_dict()).to_dict() == res.to_dict()


def test_process_execution_result_metrics():
    orch = make_orchestrator()
    res = orch.execute(linear_process())
    metrics = res.metrics()
    for key in ("state", "steps", "completed", "actual_duration", "sla_status"):
        assert key in metrics