"""Enterprise-grade test suite for the Week 10 Phase 1 Workflow Engine.

The suite is written as standard pytest (using only ``parametrize``,
``raises`` and ``approx`` - no fixtures), so it runs unmodified inside the
platform repository. A small import bootstrap lets it resolve the engine both
as ``src.workflow.workflow_engine`` (repository layout) and as a bare
``workflow_engine`` module (isolated execution).
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import FrozenInstanceError

import pytest

# --- import bootstrap -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "workflow"))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:  # repository layout
    _mod = importlib.import_module("src.workflow.workflow_engine")
except ModuleNotFoundError:  # isolated execution
    _mod = importlib.import_module("workflow_engine")

globals().update({name: getattr(_mod, name) for name in _mod.__all__})


# ---------------------------------------------------------------------------
# Shared helpers (no fixtures by design)
# ---------------------------------------------------------------------------
def make_action(action_type=None, name="action", params=None):
    return WorkflowAction.create(
        action_type or ActionType.GENERATE_AUDIT_RECORD, name, params or {}
    )


def make_step(step_id="s1", name=None, action=None, **kw):
    return WorkflowStep(
        step_id=step_id, name=name or step_id, action=action or make_action(), **kw
    )


def make_definition(workflow_id="wf", workflow_type=None, steps=None, **kw):
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name="Definition",
        workflow_type=workflow_type or WorkflowType.PREDICTIVE_MAINTENANCE,
        steps=steps or (make_step(),),
        **kw,
    )


def always_success(action, context, clock):
    return ActionOutcome.succeeded({"ok": True}, duration=0.0)


def always_fail(action, context, clock):
    return ActionOutcome.failed("boom", duration=0.0)


def duration_handler(seconds):
    def handler(action, context, clock):
        return ActionOutcome.succeeded({"slow": True}, duration=seconds)

    return handler


def fail_n_times(n):
    state = {"count": 0}

    def handler(action, context, clock):
        if state["count"] < n:
            state["count"] += 1
            return ActionOutcome.failed(f"attempt {state['count']}")
        return ActionOutcome.succeeded({"recovered": True})

    return handler


def drive_all(engine, definition, context=None, approver="approver"):
    clock = LogicalClock()
    ex = engine.create_execution(definition, context=context or {}, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    guard = 0
    while ex.state is WorkflowState.PAUSED and ex.pending_approval_step:
        ex = engine.approve(
            ex, ex.pending_approval_step, definition, actor=approver, clock=clock
        )
        guard += 1
        if guard > 20:
            break
    return ex


def sample_recommendation(kind):
    return {
        "kind": kind,
        "recommendation_id": f"REC-{kind}",
        "asset_id": "ASSET-1",
        "title": f"{kind} title",
        "summary": "summary",
        "budget": 25000.0,
        "scenario_score": 0.9,
    }


ALL_WORKFLOW_TYPES = list(WorkflowType)
ALL_STATES = list(WorkflowState)
ALL_STEP_STATUSES = list(StepStatus)
ALL_ACTION_TYPES = list(ActionType)
ALL_OPERATORS = list(ConditionOperator)
ALL_RULE_TYPES = list(RuleType)
ALL_EXEC_MODES = list(ExecutionMode)
ALL_KINDS = [
    "predictive_maintenance", "emergency_maintenance", "inspection",
    "shutdown_planning", "inventory_procurement", "executive_approval",
    "risk_mitigation", "safety_response", "knowledge_review", "scenario_evaluation",
]


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enum_cls,value", (
    [(WorkflowType, e.value) for e in WorkflowType]
    + [(WorkflowState, e.value) for e in WorkflowState]
    + [(StepStatus, e.value) for e in StepStatus]
    + [(ActionType, e.value) for e in ActionType]
    + [(ConditionOperator, e.value) for e in ConditionOperator]
    + [(RuleType, e.value) for e in RuleType]
    + [(ExecutionMode, e.value) for e in ExecutionMode]
))
def test_enum_coerce_from_value(enum_cls, value):
    coerced = enum_cls.coerce(value)
    assert coerced.value == value
    assert enum_cls.coerce(coerced) is coerced


@pytest.mark.parametrize("enum_cls", [
    WorkflowType, WorkflowState, StepStatus, ActionType, ConditionOperator,
    RuleType, ExecutionMode,
])
def test_enum_coerce_invalid_raises(enum_cls):
    with pytest.raises(WorkflowValidationError):
        enum_cls.coerce("__not_a_member__")


@pytest.mark.parametrize("state,terminal", [
    (WorkflowState.DRAFT, False),
    (WorkflowState.PENDING, False),
    (WorkflowState.RUNNING, False),
    (WorkflowState.PAUSED, False),
    (WorkflowState.COMPLETED, True),
    (WorkflowState.FAILED, True),
    (WorkflowState.CANCELLED, True),
])
def test_state_is_terminal(state, terminal):
    assert state.is_terminal is terminal


# ---------------------------------------------------------------------------
# WorkflowCondition
# ---------------------------------------------------------------------------
_COND_CASES = [
    ("x", ConditionOperator.EQ, 5, None, {"x": 5}, True),
    ("x", ConditionOperator.EQ, 5, None, {"x": 6}, False),
    ("x", ConditionOperator.NEQ, 5, None, {"x": 6}, True),
    ("x", ConditionOperator.NEQ, 5, None, {"x": 5}, False),
    ("x", ConditionOperator.GT, 5, None, {"x": 6}, True),
    ("x", ConditionOperator.GT, 5, None, {"x": 5}, False),
    ("x", ConditionOperator.GTE, 5, None, {"x": 5}, True),
    ("x", ConditionOperator.GTE, 5, None, {"x": 4}, False),
    ("x", ConditionOperator.LT, 5, None, {"x": 4}, True),
    ("x", ConditionOperator.LT, 5, None, {"x": 5}, False),
    ("x", ConditionOperator.LTE, 5, None, {"x": 5}, True),
    ("x", ConditionOperator.LTE, 5, None, {"x": 6}, False),
    ("x", ConditionOperator.IN, [1, 2, 3], None, {"x": 2}, True),
    ("x", ConditionOperator.IN, [1, 2, 3], None, {"x": 9}, False),
    ("x", ConditionOperator.NOT_IN, [1, 2, 3], None, {"x": 9}, True),
    ("x", ConditionOperator.NOT_IN, [1, 2, 3], None, {"x": 2}, False),
    ("x", ConditionOperator.BETWEEN, 1, 10, {"x": 5}, True),
    ("x", ConditionOperator.BETWEEN, 1, 10, {"x": 11}, False),
    ("x", ConditionOperator.BETWEEN, 1, 10, {"x": 1}, True),
    ("x", ConditionOperator.BETWEEN, 1, 10, {"x": 10}, True),
    ("x", ConditionOperator.EXISTS, None, None, {"x": 0}, True),
    ("x", ConditionOperator.EXISTS, None, None, {"y": 0}, False),
    ("x", ConditionOperator.GT, 5, None, {"y": 1}, False),       # missing field
    ("x", ConditionOperator.EQ, 5, None, {}, False),             # missing field
    ("x", ConditionOperator.GT, 5, None, {"x": "str"}, False),   # type mismatch
    ("x", ConditionOperator.GTE, 0.8, None, {"x": 0.92}, True),
    ("x", ConditionOperator.LTE, 0.8, None, {"x": 0.92}, False),
]


@pytest.mark.parametrize("field,op,value,value2,ctx,expected", _COND_CASES)
def test_condition_evaluate(field, op, value, value2, ctx, expected):
    cond = WorkflowCondition(field, op, value, value2)
    assert cond.evaluate(ctx) is expected


@pytest.mark.parametrize("op", ALL_OPERATORS)
def test_condition_serialization_roundtrip(op):
    v2 = 10 if op is ConditionOperator.BETWEEN else None
    cond = WorkflowCondition("f", op, 1, v2)
    restored = WorkflowCondition.from_dict(cond.to_dict())
    assert restored.to_dict() == cond.to_dict()
    assert restored.operator is op


def test_condition_between_requires_value2():
    with pytest.raises(WorkflowValidationError):
        WorkflowCondition("f", ConditionOperator.BETWEEN, 1)


@pytest.mark.parametrize("bad_field", ["", None, 5])
def test_condition_bad_field_raises(bad_field):
    with pytest.raises(WorkflowValidationError):
        WorkflowCondition(bad_field, ConditionOperator.EQ, 1)


# ---------------------------------------------------------------------------
# WorkflowAction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action_type", ALL_ACTION_TYPES)
@pytest.mark.parametrize("params", [{}, {"asset_id": "A1", "n": 3}])
def test_action_serialization_roundtrip(action_type, params):
    action = WorkflowAction.create(action_type, "name", params)
    restored = WorkflowAction.from_dict(action.to_dict())
    assert restored.to_dict() == action.to_dict()
    assert restored.action_type is action_type
    assert restored.parameters == params


def test_action_create_exposes_parameters():
    action = WorkflowAction.create(ActionType.RESERVE_INVENTORY, "r", {"qty": 5})
    assert action.parameters == {"qty": 5}


def test_action_parameters_canonicalized():
    a = WorkflowAction.create(ActionType.NO_OP, "n", {"b": 1, "a": 2})
    b = WorkflowAction.create(ActionType.NO_OP, "n", {"a": 2, "b": 1})
    assert a.parameters_json == b.parameters_json
    assert a == b


@pytest.mark.parametrize("bad_name", ["", None])
def test_action_bad_name_raises(bad_name):
    with pytest.raises(WorkflowValidationError):
        WorkflowAction(ActionType.NO_OP, bad_name)


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attempts,base,factor,n,expected", [
    (1, 0.0, 1.0, 1, 0.0),
    (3, 1.0, 2.0, 1, 0.0),
    (3, 1.0, 2.0, 2, 1.0),
    (3, 1.0, 2.0, 3, 2.0),
    (5, 2.0, 2.0, 4, 8.0),
    (5, 1.5, 1.0, 3, 1.5),
    (2, 0.0, 3.0, 2, 0.0),
])
def test_retry_delay_for_attempt(attempts, base, factor, n, expected):
    policy = RetryPolicy(attempts, base, factor)
    assert policy.delay_for_attempt(n) == pytest.approx(expected)


def test_retry_policy_serialization():
    p = RetryPolicy(4, 2.0, 1.5)
    assert RetryPolicy.from_dict(p.to_dict()).to_dict() == p.to_dict()


@pytest.mark.parametrize("attempts,base,factor", [
    (0, 0.0, 1.0), (-1, 0.0, 1.0), (1, -1.0, 1.0), (1, 0.0, 0.0), (1, 0.0, -2.0),
])
def test_retry_policy_validation(attempts, base, factor):
    with pytest.raises(WorkflowValidationError):
        RetryPolicy(attempts, base, factor)


# ---------------------------------------------------------------------------
# WorkflowStep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {},
    {"requires_approval": True},
    {"optional": True},
    {"timeout": 5.0},
    {"parallel_group": "g1"},
    {"next_steps": ("s1",)},
    {"on_failure": ("s1",)},
    {"condition_logic": "OR"},
    {"conditions": (WorkflowCondition("x", ConditionOperator.GT, 1),)},
    {"retry_policy": RetryPolicy(3, 1.0, 2.0)},
    {"metadata_json": '{"k":"v"}'},
])
def test_step_serialization_roundtrip(kwargs):
    step = make_step(**kwargs)
    restored = WorkflowStep.from_dict(step.to_dict())
    assert restored.to_dict() == step.to_dict()


@pytest.mark.parametrize("logic,conds,ctx,expected", [
    ("AND", (("x", ConditionOperator.GT, 1), ("y", ConditionOperator.LT, 9)),
     {"x": 5, "y": 5}, True),
    ("AND", (("x", ConditionOperator.GT, 1), ("y", ConditionOperator.LT, 9)),
     {"x": 5, "y": 50}, False),
    ("OR", (("x", ConditionOperator.GT, 100), ("y", ConditionOperator.LT, 9)),
     {"x": 5, "y": 5}, True),
    ("OR", (("x", ConditionOperator.GT, 100), ("y", ConditionOperator.GT, 100)),
     {"x": 5, "y": 5}, False),
])
def test_step_guard_passes(logic, conds, ctx, expected):
    conditions = tuple(WorkflowCondition(f, o, v) for f, o, v in conds)
    step = make_step(conditions=conditions, condition_logic=logic)
    assert step.guard_passes(ctx) is expected


def test_step_guard_passes_no_conditions():
    assert make_step().guard_passes({}) is True


@pytest.mark.parametrize("bad", [
    {"step_id": ""}, {"name": ""}, {"condition_logic": "XOR"}, {"timeout": -1.0},
])
def test_step_validation_errors(bad):
    kwargs = {"step_id": "s1", "name": "s1", "action": make_action()}
    kwargs.update(bad)
    with pytest.raises(WorkflowValidationError):
        WorkflowStep(**kwargs)


def test_step_action_must_be_action():
    with pytest.raises(WorkflowValidationError):
        WorkflowStep("s", "s", action="not-an-action")


# ---------------------------------------------------------------------------
# WorkflowDefinition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wtype", ALL_WORKFLOW_TYPES)
def test_definition_serialization_roundtrip(wtype):
    d = make_definition(workflow_type=wtype, steps=(
        make_step("a", action=make_action(ActionType.CREATE_MAINTENANCE_TASK)),
        make_step("b", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    restored = WorkflowDefinition.from_dict(d.to_dict())
    assert restored.to_dict() == d.to_dict()
    assert restored.workflow_type is wtype


def test_definition_requires_steps():
    with pytest.raises(WorkflowValidationError):
        WorkflowDefinition(
            workflow_id="wf", name="d",
            workflow_type=WorkflowType.PREDICTIVE_MAINTENANCE, steps=(),
        )


def test_definition_duplicate_step_ids():
    with pytest.raises(WorkflowValidationError):
        make_definition(steps=(make_step("dup"), make_step("dup")))


def test_definition_unknown_reference():
    with pytest.raises(WorkflowValidationError):
        make_definition(steps=(make_step("a", next_steps=("missing",)),))


def test_definition_get_step():
    d = make_definition(steps=(make_step("a"), make_step("b")))
    assert d.get_step("b").step_id == "b"


def test_definition_get_step_unknown():
    with pytest.raises(WorkflowValidationError):
        make_definition().get_step("nope")


def test_definition_step_ids():
    d = make_definition(steps=(make_step("a"), make_step("b")))
    assert d.step_ids == ("a", "b")


def test_definition_fingerprint_stable():
    d1 = make_definition()
    d2 = make_definition()
    assert d1.fingerprint() == d2.fingerprint()


def test_definition_fingerprint_changes():
    d1 = make_definition(workflow_id="x")
    d2 = make_definition(workflow_id="y")
    assert d1.fingerprint() != d2.fingerprint()


@pytest.mark.parametrize("policy", ["ALL", "ANY", "all", "any"])
def test_definition_parallel_policy_ok(policy):
    assert make_definition(parallel_policy=policy).parallel_policy in ("ALL", "ANY")


def test_definition_parallel_policy_invalid():
    with pytest.raises(WorkflowValidationError):
        make_definition(parallel_policy="SOME")


# ---------------------------------------------------------------------------
# ActionOutcome
# ---------------------------------------------------------------------------
def test_action_outcome_succeeded():
    o = ActionOutcome.succeeded({"a": 1}, duration=2.0)
    assert o.success and o.output == {"a": 1} and o.duration == 2.0


def test_action_outcome_failed():
    o = ActionOutcome.failed("err", {"a": 1}, duration=1.0)
    assert not o.success and o.error == "err"


@pytest.mark.parametrize("success,err", [(True, ""), (False, "boom")])
def test_action_outcome_serialization(success, err):
    o = ActionOutcome(success, '{"k":1}', err, 3.0)
    assert ActionOutcome.from_dict(o.to_dict()).to_dict() == o.to_dict()


def test_action_outcome_negative_duration():
    with pytest.raises(WorkflowValidationError):
        ActionOutcome(True, "{}", "", -1.0)


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", ALL_STEP_STATUSES)
def test_result_serialization_roundtrip(status):
    r = WorkflowResult("s", status, '{"o":1}', "", 2, 1.0, 4.0)
    assert WorkflowResult.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert r.duration == pytest.approx(3.0)


def test_result_succeeded_flag():
    assert WorkflowResult("s", StepStatus.COMPLETED).succeeded is True
    assert WorkflowResult("s", StepStatus.FAILED).succeeded is False


def test_result_duration_never_negative():
    assert WorkflowResult("s", StepStatus.COMPLETED, started_at=5.0, finished_at=1.0).duration == 0.0


# ---------------------------------------------------------------------------
# AuditEvent / WorkflowHistory
# ---------------------------------------------------------------------------
def make_event(seq=0, step="s", event="EVT", actor="system", result="ok"):
    return AuditEvent(seq, float(seq), "exec", "wf", step, event, actor, result, '{"d":1}')


@pytest.mark.parametrize("seq", [0, 1, 5, 42])
def test_audit_event_serialization(seq):
    e = make_event(seq)
    assert AuditEvent.from_dict(e.to_dict()).to_dict() == e.to_dict()


def test_audit_event_detail():
    assert make_event().detail == {"d": 1}


def test_history_append_is_immutable():
    h0 = WorkflowHistory()
    h1 = h0.append(make_event(0))
    assert len(h0) == 0 and len(h1) == 1


def test_history_next_sequence():
    h = WorkflowHistory().append(make_event(0)).append(make_event(1))
    assert h.next_sequence == 2


def test_history_next_sequence_empty():
    assert WorkflowHistory().next_sequence == 0


def test_history_filter_by_event():
    h = WorkflowHistory().append(make_event(0, event="A")).append(make_event(1, event="B"))
    assert len(h.filter_by_event("A")) == 1


def test_history_filter_by_step():
    h = WorkflowHistory().append(make_event(0, step="x")).append(make_event(1, step="y"))
    assert len(h.filter_by_step("y")) == 1


def test_history_timeline_and_iter():
    h = WorkflowHistory().append(make_event(0)).append(make_event(1))
    assert len(h.timeline()) == 2
    assert [e.sequence for e in h] == [0, 1]


def test_history_serialization():
    h = WorkflowHistory().append(make_event(0)).append(make_event(1))
    assert WorkflowHistory.from_dict(h.to_dict()).to_dict() == h.to_dict()


# ---------------------------------------------------------------------------
# WorkflowExecution
# ---------------------------------------------------------------------------
def make_execution(state=WorkflowState.DRAFT, results=(), **kw):
    return WorkflowExecution(
        execution_id="exec", workflow_id="wf",
        workflow_type=WorkflowType.PREDICTIVE_MAINTENANCE, state=state,
        results=tuple(results), **kw,
    )


@pytest.mark.parametrize("state", ALL_STATES)
def test_execution_serialization_roundtrip(state):
    ex = make_execution(state=state, created_at=1.0, updated_at=4.0)
    assert WorkflowExecution.from_dict(ex.to_dict()).to_dict() == ex.to_dict()


def test_execution_context_property():
    ex = make_execution(context_json='{"a":1}')
    assert ex.context == {"a": 1}


def test_execution_completed_failed_steps():
    results = (
        WorkflowResult("a", StepStatus.COMPLETED),
        WorkflowResult("b", StepStatus.FAILED),
        WorkflowResult("c", StepStatus.TIMEOUT),
        WorkflowResult("d", StepStatus.SKIPPED),
    )
    ex = make_execution(results=results)
    assert len(ex.completed_steps) == 1
    assert len(ex.failed_steps) == 2


def test_execution_total_attempts_and_retries():
    results = (
        WorkflowResult("a", StepStatus.COMPLETED, attempts=3),
        WorkflowResult("b", StepStatus.COMPLETED, attempts=1),
    )
    ex = make_execution(results=results)
    assert ex.total_attempts == 4
    assert ex.total_retries == 2


def test_execution_duration():
    ex = make_execution(created_at=2.0, updated_at=7.0)
    assert ex.duration == pytest.approx(5.0)


def test_execution_result_for():
    results = (WorkflowResult("a", StepStatus.COMPLETED),)
    ex = make_execution(results=results)
    assert ex.result_for("a").step_id == "a"
    assert ex.result_for("z") is None


def test_execution_metrics_keys():
    ex = make_execution(results=(WorkflowResult("a", StepStatus.COMPLETED),))
    metrics = ex.metrics()
    for key in ("state", "step_count", "completed", "failed", "skipped",
                "total_attempts", "total_retries", "duration", "audit_events"):
        assert key in metrics


@pytest.mark.parametrize("state,terminal", [
    (WorkflowState.COMPLETED, True), (WorkflowState.FAILED, True),
    (WorkflowState.CANCELLED, True), (WorkflowState.RUNNING, False),
])
def test_execution_is_terminal(state, terminal):
    assert make_execution(state=state).is_terminal is terminal


# ---------------------------------------------------------------------------
# WorkflowStatistics
# ---------------------------------------------------------------------------
def test_statistics_empty():
    stats = WorkflowStatistics.from_executions([])
    assert stats.workflow_count == 0 and stats.success_rate == 0.0


def test_statistics_counts():
    execs = [
        make_execution(state=WorkflowState.COMPLETED, created_at=0.0, updated_at=2.0),
        make_execution(state=WorkflowState.COMPLETED, created_at=0.0, updated_at=4.0),
        make_execution(state=WorkflowState.FAILED),
        make_execution(state=WorkflowState.CANCELLED),
        make_execution(state=WorkflowState.RUNNING),
        make_execution(state=WorkflowState.PAUSED),
    ]
    stats = WorkflowStatistics.from_executions(execs)
    assert stats.workflow_count == 6
    assert stats.completed == 2
    assert stats.failed == 1
    assert stats.cancelled == 1
    assert stats.running == 1
    assert stats.paused == 1


def test_statistics_average_duration():
    execs = [
        make_execution(state=WorkflowState.COMPLETED, created_at=0.0, updated_at=2.0),
        make_execution(state=WorkflowState.COMPLETED, created_at=0.0, updated_at=6.0),
    ]
    assert WorkflowStatistics.from_executions(execs).average_duration == pytest.approx(4.0)


@pytest.mark.parametrize("completed,failed,cancelled,expected", [
    (2, 0, 0, 1.0),
    (1, 1, 0, 0.5),
    (0, 2, 0, 0.0),
    (3, 1, 0, 0.75),
])
def test_statistics_success_rate(completed, failed, cancelled, expected):
    execs = (
        [make_execution(state=WorkflowState.COMPLETED) for _ in range(completed)]
        + [make_execution(state=WorkflowState.FAILED) for _ in range(failed)]
        + [make_execution(state=WorkflowState.CANCELLED) for _ in range(cancelled)]
    )
    assert WorkflowStatistics.from_executions(execs).success_rate == pytest.approx(expected)


def test_statistics_serialization():
    stats = WorkflowStatistics(
        workflow_count=5, completed=3, failed=1, cancelled=1, running=0, paused=0,
        average_duration=2.5, total_retries=4, approval_count=2, success_rate=0.6,
    )
    assert WorkflowStatistics.from_dict(stats.to_dict()).to_dict() == stats.to_dict()


def test_statistics_total_retries():
    execs = [make_execution(results=(WorkflowResult("a", StepStatus.COMPLETED, attempts=3),))]
    assert WorkflowStatistics.from_executions(execs).total_retries == 2


# ---------------------------------------------------------------------------
# WorkflowStateMachine
# ---------------------------------------------------------------------------
_EXPECTED = {
    WorkflowState.DRAFT: {WorkflowState.PENDING, WorkflowState.CANCELLED},
    WorkflowState.PENDING: {WorkflowState.RUNNING, WorkflowState.CANCELLED},
    WorkflowState.RUNNING: {
        WorkflowState.PAUSED, WorkflowState.COMPLETED, WorkflowState.FAILED,
        WorkflowState.CANCELLED,
    },
    WorkflowState.PAUSED: {
        WorkflowState.RUNNING, WorkflowState.CANCELLED, WorkflowState.FAILED,
    },
    WorkflowState.COMPLETED: set(),
    WorkflowState.FAILED: set(),
    WorkflowState.CANCELLED: set(),
}


@pytest.mark.parametrize("src", ALL_STATES)
@pytest.mark.parametrize("dst", ALL_STATES)
def test_state_machine_transition_matrix(src, dst):
    sm = WorkflowStateMachine()
    expected = dst in _EXPECTED[src]
    assert sm.can_transition(src, dst) is expected
    if expected:
        sm.validate(src, dst)
    else:
        with pytest.raises(InvalidStateTransitionError):
            sm.validate(src, dst)


@pytest.mark.parametrize("state", ALL_STATES)
def test_state_machine_allowed_transitions(state):
    sm = WorkflowStateMachine()
    assert set(sm.allowed_transitions(state)) == _EXPECTED[state]


@pytest.mark.parametrize("state,terminal", [
    (WorkflowState.COMPLETED, True), (WorkflowState.FAILED, True),
    (WorkflowState.CANCELLED, True), (WorkflowState.DRAFT, False),
    (WorkflowState.PENDING, False), (WorkflowState.RUNNING, False),
    (WorkflowState.PAUSED, False),
])
def test_state_machine_is_terminal(state, terminal):
    assert WorkflowStateMachine.is_terminal(state) is terminal


# ---------------------------------------------------------------------------
# Rule / RuleEngine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rtype", ALL_RULE_TYPES)
def test_rule_serialization_roundtrip(rtype):
    rule = Rule("r", rtype, WorkflowCondition("x", ConditionOperator.GT, 1), "d")
    assert Rule.from_dict(rule.to_dict()).to_dict() == rule.to_dict()
    assert rule.rule_type is rtype


@pytest.mark.parametrize("rtype,field,threshold,ctx_val,expected", [
    (RuleType.RISK_THRESHOLD, "risk_score", 0.8, 0.9, True),
    (RuleType.RISK_THRESHOLD, "risk_score", 0.8, 0.5, False),
    (RuleType.ASSET_CRITICALITY, "asset_criticality", 0.7, 0.95, True),
    (RuleType.BUDGET_THRESHOLD, "budget", 10000, 25000, True),
    (RuleType.BUDGET_THRESHOLD, "budget", 10000, 5000, False),
    (RuleType.SCENARIO_SCORE, "scenario_score", 0.6, 0.7, True),
    (RuleType.KNOWLEDGE_CONFIDENCE, "knowledge_confidence", 0.5, 0.4, False),
    (RuleType.EXECUTIVE_CONFIDENCE, "executive_confidence", 0.9, 0.95, True),
])
def test_rule_evaluate(rtype, field, threshold, ctx_val, expected):
    rule = Rule("r", rtype, WorkflowCondition(field, ConditionOperator.GTE, threshold))
    assert rule.evaluate({field: ctx_val}) is expected


def test_rule_disabled_never_matches():
    rule = Rule("r", RuleType.RISK_THRESHOLD,
                WorkflowCondition("x", ConditionOperator.GT, 0), enabled=False)
    assert rule.evaluate({"x": 100}) is False


def test_rule_condition_type_validated():
    with pytest.raises(WorkflowValidationError):
        Rule("r", RuleType.RISK_THRESHOLD, "not-a-condition")


def test_rule_engine_register_get():
    eng = RuleEngine()
    rule = Rule("r1", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1))
    eng.register(rule)
    assert eng.get("r1") is rule
    assert "r1" in eng and len(eng) == 1


def test_rule_engine_duplicate():
    eng = RuleEngine()
    rule = Rule("r1", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1))
    eng.register(rule)
    with pytest.raises(RegistryError):
        eng.register(rule)


def test_rule_engine_overwrite():
    eng = RuleEngine()
    cond = WorkflowCondition("x", ConditionOperator.GT, 1)
    eng.register(Rule("r1", RuleType.RISK_THRESHOLD, cond))
    eng.register(Rule("r1", RuleType.BUDGET_THRESHOLD, cond), overwrite=True)
    assert eng.get("r1").rule_type is RuleType.BUDGET_THRESHOLD


def test_rule_engine_remove():
    eng = RuleEngine()
    eng.register(Rule("r1", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1)))
    eng.remove("r1")
    assert "r1" not in eng


def test_rule_engine_remove_unknown():
    with pytest.raises(RegistryError):
        RuleEngine().remove("nope")


def test_rule_engine_get_unknown():
    with pytest.raises(RegistryError):
        RuleEngine().get("nope")


def test_rule_engine_register_non_rule():
    with pytest.raises(WorkflowValidationError):
        RuleEngine().register("not-a-rule")


def test_rule_engine_list_sorted():
    eng = RuleEngine()
    cond = WorkflowCondition("x", ConditionOperator.GT, 1)
    eng.register(Rule("b", RuleType.RISK_THRESHOLD, cond))
    eng.register(Rule("a", RuleType.RISK_THRESHOLD, cond))
    assert [r.rule_id for r in eng.list_rules()] == ["a", "b"]


def test_rule_engine_evaluate_all_and_matching():
    eng = RuleEngine([
        Rule("hi", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1)),
        Rule("lo", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 100)),
    ])
    results = eng.evaluate_all({"x": 5})
    assert results == {"hi": True, "lo": False}
    assert eng.evaluate("hi", {"x": 5}) is True
    matching = eng.matching_rules({"x": 5})
    assert [r.rule_id for r in matching] == ["hi"]


def test_rule_engine_matching_by_type():
    eng = RuleEngine([
        Rule("risk", RuleType.RISK_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1)),
        Rule("budget", RuleType.BUDGET_THRESHOLD, WorkflowCondition("x", ConditionOperator.GT, 1)),
    ])
    matched = eng.matching_by_type(RuleType.BUDGET_THRESHOLD, {"x": 5})
    assert [r.rule_id for r in matched] == ["budget"]


# ---------------------------------------------------------------------------
# ActionEngine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action_type", [
    ActionType.CREATE_MAINTENANCE_TASK, ActionType.SCHEDULE_INSPECTION,
    ActionType.GENERATE_EXECUTIVE_ALERT, ActionType.RESERVE_INVENTORY,
    ActionType.ESCALATE_APPROVAL, ActionType.CREATE_KNOWLEDGE_REVIEW,
    ActionType.GENERATE_AUDIT_RECORD, ActionType.NO_OP,
])
def test_default_action_dispatch(action_type):
    eng = create_default_action_engine()
    action = WorkflowAction.create(action_type, "n", {"asset_id": "A"})
    outcome = eng.dispatch(action, {"asset_id": "A"}, LogicalClock())
    assert outcome.success is True


def test_default_action_deterministic_ids():
    eng = create_default_action_engine()
    action = WorkflowAction.create(ActionType.CREATE_MAINTENANCE_TASK, "n", {"asset_id": "A"})
    o1 = eng.dispatch(action, {"asset_id": "A"}, LogicalClock())
    o2 = eng.dispatch(action, {"asset_id": "A"}, LogicalClock())
    assert o1.output == o2.output


def test_action_engine_unknown_handler():
    eng = ActionEngine()
    with pytest.raises(ActionDispatchError):
        eng.dispatch(WorkflowAction.create(ActionType.NO_OP, "n"), {}, LogicalClock())


def test_action_engine_custom_handler():
    eng = ActionEngine()
    eng.register_handler(ActionType.NO_OP, always_success)
    assert eng.has_handler(ActionType.NO_OP)
    assert eng.dispatch(WorkflowAction.create(ActionType.NO_OP, "n"), {}, LogicalClock()).success


def test_action_engine_bad_handler_return():
    eng = ActionEngine()
    eng.register_handler(ActionType.NO_OP, lambda a, c, k: "not-an-outcome")
    with pytest.raises(ActionDispatchError):
        eng.dispatch(WorkflowAction.create(ActionType.NO_OP, "n"), {}, LogicalClock())


def test_action_engine_non_callable_handler():
    with pytest.raises(WorkflowValidationError):
        ActionEngine().register_handler(ActionType.NO_OP, 123)


def test_action_engine_registered_types():
    eng = create_default_action_engine()
    assert ActionType.NO_OP in eng.registered_types()
    assert len(eng.registered_types()) == 8


# ---------------------------------------------------------------------------
# WorkflowRegistry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wtype", ALL_WORKFLOW_TYPES)
def test_registry_register_get(wtype):
    reg = WorkflowRegistry()
    d = make_definition(workflow_id=f"wf-{wtype.value}", workflow_type=wtype)
    reg.register_workflow(d)
    assert reg.get_workflow(d.workflow_id) is d
    assert d.workflow_id in reg


def test_registry_duplicate():
    reg = WorkflowRegistry()
    d = make_definition()
    reg.register_workflow(d)
    with pytest.raises(RegistryError):
        reg.register_workflow(d)


def test_registry_overwrite():
    reg = WorkflowRegistry()
    reg.register_workflow(make_definition(workflow_id="wf"))
    reg.register_workflow(make_definition(workflow_id="wf", description="v2"), overwrite=True)
    assert reg.get_workflow("wf").description == "v2"


def test_registry_remove():
    reg = WorkflowRegistry()
    reg.register_workflow(make_definition(workflow_id="wf"))
    reg.remove_workflow("wf")
    assert "wf" not in reg and len(reg) == 0


def test_registry_remove_unknown():
    with pytest.raises(RegistryError):
        WorkflowRegistry().remove_workflow("nope")


def test_registry_get_unknown():
    with pytest.raises(RegistryError):
        WorkflowRegistry().get_workflow("nope")


def test_registry_list_sorted():
    reg = WorkflowRegistry()
    reg.register_workflow(make_definition(workflow_id="b"))
    reg.register_workflow(make_definition(workflow_id="a"))
    assert [d.workflow_id for d in reg.list_workflows()] == ["a", "b"]


def test_registry_register_non_definition():
    with pytest.raises(WorkflowValidationError):
        WorkflowRegistry().register_workflow("not-a-def")


# ---------------------------------------------------------------------------
# RecommendationCompiler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_compiler_compiles_kind(kind):
    compiler = RecommendationCompiler()
    definition = compiler.compile(sample_recommendation(kind))
    definition.validate()
    assert len(definition.steps) >= 1


@pytest.mark.parametrize("kind,expected_type", [
    ("predictive_maintenance", WorkflowType.PREDICTIVE_MAINTENANCE),
    ("emergency_maintenance", WorkflowType.EMERGENCY_MAINTENANCE),
    ("inspection", WorkflowType.INSPECTION),
    ("shutdown_planning", WorkflowType.SHUTDOWN_PLANNING),
    ("inventory_procurement", WorkflowType.INVENTORY_PROCUREMENT),
    ("executive_approval", WorkflowType.EXECUTIVE_APPROVAL),
    ("risk_mitigation", WorkflowType.RISK_MITIGATION),
    ("safety_response", WorkflowType.SAFETY_RESPONSE),
    ("knowledge_review", WorkflowType.KNOWLEDGE_REVIEW),
    ("scenario_evaluation", WorkflowType.SCENARIO_EVALUATION),
])
def test_compiler_maps_type(kind, expected_type):
    definition = RecommendationCompiler().compile(sample_recommendation(kind))
    assert definition.workflow_type is expected_type


def test_compiler_supported_kinds():
    assert len(RecommendationCompiler().supported_kinds()) == 10


def test_compiler_unknown_kind_falls_back():
    definition = RecommendationCompiler().compile({"kind": "totally_unknown"})
    assert definition.workflow_type is WorkflowType.PREDICTIVE_MAINTENANCE


def test_compiler_custom_builder():
    compiler = RecommendationCompiler()
    compiler.register_builder(
        "custom", lambda rec: (make_step("only", action=make_action(ActionType.NO_OP)),)
    )
    compiler._KIND_TO_TYPE["custom"] = WorkflowType.INSPECTION
    definition = compiler.compile({"kind": "custom"})
    assert definition.step_ids == ("only",)


def test_compiler_metadata_carries_recommendation():
    definition = RecommendationCompiler().compile(sample_recommendation("inspection"))
    assert definition.metadata["recommendation_id"] == "REC-inspection"


def test_compiler_accepts_recommendation_source():
    class Source:
        def as_recommendation(self):
            return sample_recommendation("inspection")

    assert isinstance(Source(), RecommendationSource)
    definition = RecommendationCompiler().compile(Source())
    assert definition.workflow_type is WorkflowType.INSPECTION


def test_compiler_rejects_non_mapping():
    with pytest.raises(WorkflowValidationError):
        RecommendationCompiler().compile(12345)


def test_compiler_explicit_workflow_id():
    definition = RecommendationCompiler().compile(
        sample_recommendation("inspection"), workflow_id="custom-id"
    )
    assert definition.workflow_id == "custom-id"


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------
def test_create_execution_is_draft():
    engine = create_default_engine()
    ex = engine.create_execution(make_definition())
    assert ex.state is WorkflowState.DRAFT
    assert len(ex.history.filter_by_event("CREATED")) == 1


def test_create_execution_requires_definition():
    with pytest.raises(WorkflowValidationError):
        create_default_engine().create_execution("not-a-def")


def test_run_sequential_completes():
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.RESERVE_INVENTORY)),
        make_step("b", action=make_action(ActionType.CREATE_MAINTENANCE_TASK)),
        make_step("c", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert len(ex.completed_steps) == 3


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_engine_drives_recommendation_to_terminal(kind):
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation(kind))
    ex = drive_all(engine, definition)
    assert ex.is_terminal
    assert ex.state is WorkflowState.COMPLETED


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_engine_determinism(kind):
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation(kind))
    ctx = {"asset_id": "ASSET-1", "budget": 25000.0, "scenario_score": 0.9}
    ex1 = drive_all(engine, definition, ctx)
    ex2 = drive_all(engine, definition, ctx)
    assert ex1.to_dict() == ex2.to_dict()


def test_emergency_pauses_for_approval():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    assert ex.state is WorkflowState.PAUSED
    assert ex.pending_approval_step == "approval"


def test_approve_completes_workflow():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    ex = engine.approve(ex, "approval", definition, actor="cfo", clock=clock)
    assert ex.state is WorkflowState.COMPLETED
    assert len(ex.history.filter_by_event("APPROVAL_GRANTED")) == 1


def test_reject_fails_workflow():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    ex = engine.reject(ex, "approval", actor="cfo", clock=clock)
    assert ex.state is WorkflowState.FAILED
    assert ex.result_for("approval").status is StepStatus.REJECTED


def test_reject_requires_paused():
    engine = create_default_engine()
    definition = make_definition()
    ex = engine.create_execution(definition)
    with pytest.raises(InvalidStateTransitionError):
        engine.reject(ex, "s1")


def test_approve_wrong_step_raises():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    with pytest.raises(WorkflowExecutionError):
        engine.approve(ex, "not-the-step", definition, clock=clock)


def test_approve_requires_paused():
    engine = create_default_engine()
    definition = make_definition()
    ex = engine.create_execution(definition)
    with pytest.raises(InvalidStateTransitionError):
        engine.approve(ex, "s1", definition)


@pytest.mark.parametrize("setup_state", ["draft", "pending", "paused"])
def test_cancel_from_various_states(setup_state):
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    if setup_state == "pending":
        ex = engine.submit(ex, clock=clock)
    elif setup_state == "paused":
        ex = engine.run(ex, definition, clock=clock)
    ex = engine.cancel(ex, actor="ops", clock=clock)
    assert ex.state is WorkflowState.CANCELLED


def test_cancel_terminal_raises():
    engine = create_default_engine()
    definition = make_definition()
    ex = drive_all(engine, definition)
    with pytest.raises(InvalidStateTransitionError):
        engine.cancel(ex)


def test_pause_running_snapshot():
    engine = create_default_engine()
    running = make_execution(state=WorkflowState.RUNNING)
    paused = engine.pause(running)
    assert paused.state is WorkflowState.PAUSED


@pytest.mark.parametrize("state", [WorkflowState.DRAFT, WorkflowState.PENDING])
def test_pause_invalid_state(state):
    engine = create_default_engine()
    with pytest.raises(InvalidStateTransitionError):
        engine.pause(make_execution(state=state))


def test_resume_runs_to_completion():
    engine = create_default_engine()
    definition = make_definition(workflow_id="rwf", steps=(
        make_step("a", action=make_action(ActionType.RESERVE_INVENTORY)),
        make_step("b", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    clock = LogicalClock()
    running = WorkflowExecution(
        execution_id="e", workflow_id="rwf",
        workflow_type=WorkflowType.PREDICTIVE_MAINTENANCE,
        state=WorkflowState.RUNNING,
    )
    paused = engine.pause(running, clock=clock)
    resumed = engine.resume(paused, definition, clock=clock)
    assert resumed.state is WorkflowState.COMPLETED


def test_resume_pending_approval_raises():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    with pytest.raises(WorkflowExecutionError):
        engine.resume(ex, definition, clock=clock)


def test_resume_non_paused_raises():
    engine = create_default_engine()
    definition = make_definition()
    ex = engine.create_execution(definition)
    with pytest.raises(InvalidStateTransitionError):
        engine.resume(ex, definition)


def test_run_invalid_state_raises():
    engine = create_default_engine()
    definition = make_definition()
    ex = drive_all(engine, definition)  # completed
    with pytest.raises(InvalidStateTransitionError):
        engine.run(ex, definition)


def test_run_definition_mismatch_raises():
    engine = create_default_engine()
    ex = engine.create_execution(make_definition(workflow_id="a"))
    with pytest.raises(WorkflowExecutionError):
        engine.run(ex, make_definition(workflow_id="b"))


def test_guard_skip_records_skipped():
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP),
                  conditions=(WorkflowCondition("flag", ConditionOperator.EQ, True),)),
        make_step("b", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition, context={"flag": False})
    assert ex.result_for("a").status is StepStatus.SKIPPED
    assert ex.state is WorkflowState.COMPLETED


@pytest.mark.parametrize("fail_count,expected_attempts", [(0, 1), (1, 2), (2, 3)])
def test_retry_succeeds_after_failures(fail_count, expected_attempts):
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.NO_OP, fail_n_times(fail_count))
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP),
                  retry_policy=RetryPolicy(max_attempts=3, backoff_base=1.0, backoff_factor=2.0)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert ex.result_for("a").attempts == expected_attempts


def test_retry_exhausted_fails_workflow():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.NO_OP, always_fail)
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP),
                  retry_policy=RetryPolicy(max_attempts=3)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.FAILED
    assert ex.result_for("a").attempts == 3


def test_optional_step_failure_continues():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.NO_OP, always_fail)
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP), optional=True),
        make_step("b", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert ex.result_for("a").status is StepStatus.FAILED


def test_on_failure_branch_taken():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.CREATE_MAINTENANCE_TASK, always_fail)
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.CREATE_MAINTENANCE_TASK),
                  on_failure=("c",)),
        make_step("b", action=make_action(ActionType.SCHEDULE_INSPECTION)),
        make_step("c", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert ex.result_for("a").status is StepStatus.FAILED
    assert ex.result_for("c").status is StepStatus.COMPLETED
    assert ex.result_for("b") is None  # branch skipped b


def test_conditional_next_steps_branch():
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP), next_steps=("c",)),
        make_step("b", action=make_action(ActionType.SCHEDULE_INSPECTION)),
        make_step("c", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert ex.result_for("a").status is StepStatus.COMPLETED
    assert ex.result_for("c").status is StepStatus.COMPLETED
    assert ex.result_for("b") is None


@pytest.mark.parametrize("duration,timeout,expected", [
    (5.0, 1.0, StepStatus.TIMEOUT),
    (0.5, 1.0, StepStatus.COMPLETED),
    (1.0, 1.0, StepStatus.COMPLETED),
    (2.0, None, StepStatus.COMPLETED),
])
def test_timeout_handling(duration, timeout, expected):
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.NO_OP, duration_handler(duration))
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP), timeout=timeout),
    ))
    ex = drive_all(engine, definition)
    assert ex.result_for("a").status is expected


def test_parallel_group_all_success():
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("p1", action=make_action(ActionType.RESERVE_INVENTORY), parallel_group="g"),
        make_step("p2", action=make_action(ActionType.SCHEDULE_INSPECTION), parallel_group="g"),
        make_step("done", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert len(ex.completed_steps) == 3


def test_parallel_group_all_policy_fails():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.SCHEDULE_INSPECTION, always_fail)
    definition = make_definition(parallel_policy="ALL", steps=(
        make_step("p1", action=make_action(ActionType.RESERVE_INVENTORY), parallel_group="g"),
        make_step("p2", action=make_action(ActionType.SCHEDULE_INSPECTION), parallel_group="g"),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.FAILED


def test_parallel_group_any_policy_survives_one_failure():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.SCHEDULE_INSPECTION, always_fail)
    definition = make_definition(parallel_policy="ANY", steps=(
        make_step("p1", action=make_action(ActionType.RESERVE_INVENTORY), parallel_group="g"),
        make_step("p2", action=make_action(ActionType.SCHEDULE_INSPECTION), parallel_group="g"),
    ))
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED


def test_parallel_group_duration_is_max_not_sum():
    engine = create_default_engine()
    engine.action_engine.register_handler(ActionType.RESERVE_INVENTORY, duration_handler(3.0))
    engine.action_engine.register_handler(ActionType.SCHEDULE_INSPECTION, duration_handler(5.0))
    definition = make_definition(steps=(
        make_step("p1", action=make_action(ActionType.RESERVE_INVENTORY), parallel_group="g"),
        make_step("p2", action=make_action(ActionType.SCHEDULE_INSPECTION), parallel_group="g"),
    ))
    ex = drive_all(engine, definition)
    assert ex.duration == pytest.approx(5.0)


def test_merge_output_feeds_context():
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.NO_OP)),
        make_step("b", action=make_action(ActionType.GENERATE_AUDIT_RECORD),
                  conditions=(WorkflowCondition("a_output", ConditionOperator.EXISTS, None),)),
    ))
    ex = drive_all(engine, definition)
    assert ex.result_for("b").status is StepStatus.COMPLETED


def test_execution_id_deterministic():
    engine = create_default_engine()
    definition = make_definition()
    ex1 = engine.create_execution(definition, context={"k": 1})
    ex2 = engine.create_execution(definition, context={"k": 1})
    assert ex1.execution_id == ex2.execution_id


def test_compile_recommendation_registers():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("inspection"), register=True)
    assert definition.workflow_id in engine.registry


# ---------------------------------------------------------------------------
# Audit completeness
# ---------------------------------------------------------------------------
def test_audit_sequence_monotonic():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("emergency_maintenance"))
    ex = drive_all(engine, definition)
    seqs = [e.sequence for e in ex.history]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))  # unique


@pytest.mark.parametrize("event_name", [
    "CREATED", "SUBMITTED", "STARTED", "STEP_STARTED", "STEP_COMPLETED", "COMPLETED",
])
def test_audit_contains_lifecycle_events(event_name):
    engine = create_default_engine()
    definition = make_definition(steps=(
        make_step("a", action=make_action(ActionType.GENERATE_AUDIT_RECORD)),
    ))
    ex = drive_all(engine, definition)
    assert len(ex.history.filter_by_event(event_name)) >= 1


def test_audit_events_carry_actor_and_workflow_id():
    engine = create_default_engine()
    definition = make_definition()
    ex = drive_all(engine, definition)
    for event in ex.history:
        assert event.workflow_id == definition.workflow_id
        assert event.actor


def test_audit_approval_records_actor():
    engine = create_default_engine()
    definition = engine.compile_recommendation(sample_recommendation("executive_approval"))
    clock = LogicalClock()
    ex = engine.create_execution(definition, clock=clock)
    ex = engine.run(ex, definition, clock=clock)
    ex = engine.approve(ex, ex.pending_approval_step, definition, actor="ceo", clock=clock)
    granted = ex.history.filter_by_event("APPROVAL_GRANTED")
    assert granted[0].actor == "ceo"


# ---------------------------------------------------------------------------
# Statistics integration
# ---------------------------------------------------------------------------
def test_statistics_over_real_runs():
    engine = create_default_engine()
    executions = []
    for kind in ("inspection", "knowledge_review", "risk_mitigation"):
        definition = engine.compile_recommendation(sample_recommendation(kind))
        executions.append(drive_all(engine, definition))
    stats = engine.statistics(executions)
    assert stats.workflow_count == 3
    assert stats.completed == 3
    assert stats.success_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Backward compatibility / non-invasiveness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "import airflow", "import temporal", "import prefect", "import celery",
    "import torch", "import tensorflow", "from src.", "import openai",
])
def test_module_has_no_forbidden_imports(forbidden):
    with open(_mod.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert forbidden not in source


def test_module_only_imports_numpy_and_stdlib():
    # The engine must not import any Weeks 1-9 package.
    with open(_mod.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "src.workflow" not in source.replace("src.workflow.workflow_engine", "")


def test_factory_returns_engine():
    assert isinstance(create_default_engine(), WorkflowEngine)


def test_action_factory_has_all_defaults():
    eng = create_default_action_engine()
    for action_type in ActionType:
        assert eng.has_handler(action_type)


# ---------------------------------------------------------------------------
# Frozen / immutability guarantees
# ---------------------------------------------------------------------------
_FROZEN_INSTANCES = [
    WorkflowCondition("f", ConditionOperator.EQ, 1),
    WorkflowAction.create(ActionType.NO_OP, "n"),
    RetryPolicy(),
    make_step(),
    make_definition(),
    ActionOutcome.succeeded({}),
    WorkflowResult("s", StepStatus.COMPLETED),
    AuditEvent(0, 0.0, "e", "w", "s", "EVT", "a", "ok"),
    WorkflowHistory(),
    make_execution(),
    WorkflowStatistics(),
    Rule("r", RuleType.RISK_THRESHOLD, WorkflowCondition("f", ConditionOperator.EQ, 1)),
]


@pytest.mark.parametrize("instance", _FROZEN_INSTANCES)
def test_instances_are_frozen(instance):
    with pytest.raises(FrozenInstanceError):
        instance.__dict__  # access first to ensure it's a dataclass
        object.__setattr__  # noqa
        setattr(instance, "nonexistent_attr_xyz", 1)


@pytest.mark.parametrize("instance", _FROZEN_INSTANCES)
def test_instances_serialization_idempotent(instance):
    d1 = instance.to_dict()
    restored = type(instance).from_dict(d1)
    assert restored.to_dict() == d1


# ---------------------------------------------------------------------------
# Edge cases and scale
# ---------------------------------------------------------------------------
def test_large_sequential_workflow_executes():
    engine = create_default_engine()
    steps = tuple(
        make_step(f"s{i}", action=make_action(ActionType.GENERATE_AUDIT_RECORD))
        for i in range(100)
    )
    definition = make_definition(workflow_id="big", steps=steps)
    ex = drive_all(engine, definition)
    assert ex.state is WorkflowState.COMPLETED
    assert len(ex.completed_steps) == 100


def test_large_registry():
    reg = WorkflowRegistry()
    for i in range(200):
        reg.register_workflow(make_definition(workflow_id=f"wf{i}"))
    assert len(reg) == 200
    assert len(reg.list_workflows()) == 200


def test_clock_logical_advances_only_explicitly():
    clock = LogicalClock()
    assert clock.now() == 0.0
    clock.advance(5.0)
    assert clock.now() == 5.0


def test_clock_fixed_never_advances():
    clock = FixedClock(3.0)
    clock.advance(10.0)
    assert clock.now() == 3.0


def test_clock_negative_advance_raises():
    with pytest.raises(WorkflowExecutionError):
        LogicalClock().advance(-1.0)


@pytest.mark.parametrize("payload", [
    {"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 1, "b": 2, "c": [1, 2, 3]},
])
def test_canonical_json_is_stable(payload):
    a = WorkflowAction.create(ActionType.NO_OP, "n", payload)
    b = WorkflowAction.create(ActionType.NO_OP, "n", dict(reversed(list(payload.items()))))
    assert a.parameters_json == b.parameters_json


def test_empty_context_run():
    engine = create_default_engine()
    ex = drive_all(engine, make_definition())
    assert ex.state is WorkflowState.COMPLETED


def test_submit_transitions_to_pending():
    engine = create_default_engine()
    ex = engine.create_execution(make_definition())
    ex = engine.submit(ex)
    assert ex.state is WorkflowState.PENDING


def test_cli_demo_runs():
    assert _mod.main(["--demo"]) == 0


def test_cli_no_args_prints_help():
    assert _mod.main([]) == 0