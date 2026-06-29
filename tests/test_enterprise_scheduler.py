"""Enterprise test suite for the Week 10 Phase 4 Enterprise Scheduler.

Standard pytest (parametrize / raises / approx only - no fixtures), with a
bootstrap resolving the scheduler and the composed Phase 3 event bus.
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
sys.path.insert(0, os.path.join(_HERE, "..", "src", "events"))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "scheduler"))
sys.path.insert(0, os.path.join(_HERE, ".."))

try:
    sch = importlib.import_module("src.scheduler.enterprise_scheduler")
except ModuleNotFoundError:
    sch = importlib.import_module("enterprise_scheduler")

try:
    eeb = importlib.import_module("src.events.enterprise_event_bus")
except ModuleNotFoundError:
    try:
        eeb = importlib.import_module("enterprise_event_bus")
    except ModuleNotFoundError:
        eeb = None

globals().update({name: getattr(sch, name) for name in sch.__all__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trig(ttype=None, **kw):
    return ScheduleTrigger(ttype or TriggerType.MANUAL, **kw)


def job(job_id="j", job_type=None, trigger=None, **kw):
    return ScheduledJob(job_id, job_id, job_type or JobType.CUSTOM,
                        trigger or trig(TriggerType.MANUAL), **kw)


def interval_job(job_id="j", interval=10.0, start=0.0, **kw):
    return job(job_id, JobType.HEALTH_MONITORING,
               ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=interval,
                               start_time=start), **kw)


def ok_executor(j, c, k):
    return JobOutcome.succeeded({"j": j.job_id})


def fail_executor(j, c, k):
    return JobOutcome.failed("boom")


def slow_executor(seconds):
    def ex(j, c, k):
        return JobOutcome.succeeded({}, duration=seconds)
    return ex


def make_scheduler(executor=ok_executor):
    s = create_default_scheduler()
    for jt in JobType:
        s.register_executor(jt, executor)
    return s


def ex_record(job_id="j", status=None, sched=0.0, started=0.0, finished=1.0, attempts=1):
    return JobExecution("e", job_id, sched, started, finished,
                        status or JobStatus.SUCCESS, attempts, "manual")


ALL_TRIGGER_TYPES = list(TriggerType)
ALL_JOB_TYPES = list(JobType)
ALL_POLICIES = list(ExecutionPolicy)
ALL_JOB_STATES = list(JobState)
ALL_JOB_STATUSES = list(JobStatus)
ALL_AUTOMATION = list(AutomationType)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enum_cls,value", (
    [(TriggerType, e.value) for e in TriggerType]
    + [(JobType, e.value) for e in JobType]
    + [(ExecutionPolicy, e.value) for e in ExecutionPolicy]
    + [(JobState, e.value) for e in JobState]
    + [(JobStatus, e.value) for e in JobStatus]
    + [(AutomationType, e.value) for e in AutomationType]
))
def test_enum_coerce(enum_cls, value):
    assert enum_cls.coerce(value).value == value


@pytest.mark.parametrize("enum_cls", [
    TriggerType, JobType, ExecutionPolicy, JobState, JobStatus, AutomationType])
def test_enum_coerce_invalid(enum_cls):
    with pytest.raises(JobValidationError):
        enum_cls.coerce("__nope__")


# ---------------------------------------------------------------------------
# ScheduleTrigger - next_after
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ref,expected", [
    (0.0, 0.0), (5.0, 10.0), (10.0, 10.0), (11.0, 20.0), (25.0, 30.0),
])
def test_trigger_fixed_interval(ref, expected):
    t = ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0, start_time=0.0)
    assert t.next_after(ref) == pytest.approx(expected)


@pytest.mark.parametrize("ref,expected", [
    (0.0, 25.0), (25.0, 25.0), (26.0, None),
])
def test_trigger_one_time(ref, expected):
    t = ScheduleTrigger(TriggerType.ONE_TIME, start_time=25.0)
    assert t.next_after(ref) == expected


def test_trigger_emergency_fires_now():
    t = ScheduleTrigger(TriggerType.EMERGENCY)
    assert t.next_after(5.0) == pytest.approx(5.0)


@pytest.mark.parametrize("ref,expected", [
    (0.0, 3600.0), (3600.0, 3600.0), (3601.0, 3600.0 + 86400.0),
])
def test_trigger_daily(ref, expected):
    t = ScheduleTrigger(TriggerType.DAILY, at_second=3600)
    assert t.next_after(ref) == pytest.approx(expected)


def test_trigger_weekly_same_day():
    # epoch day 0 (1970-01-01) is Thursday (weekday 3)
    t = ScheduleTrigger(TriggerType.WEEKLY, weekday=3, at_second=0)
    assert t.next_after(0.0) == pytest.approx(0.0)
    assert t.next_after(1.0) == pytest.approx(7 * 86400.0)


def test_trigger_monthly():
    t = ScheduleTrigger(TriggerType.MONTHLY, day_of_month=1, at_second=0)
    assert t.next_after(0.0) == pytest.approx(0.0)
    assert t.next_after(1.0) == pytest.approx(31 * 86400.0)  # 1970-02-01


@pytest.mark.parametrize("ref,expected", [(0.0, 0.0), (1.0, 900.0), (900.0, 900.0)])
def test_trigger_cron_every_15(ref, expected):
    t = ScheduleTrigger(TriggerType.CRON, cron_expression="*/15 * * * *")
    assert t.next_after(ref) == pytest.approx(expected)


def test_trigger_cron_specific_hour():
    t = ScheduleTrigger(TriggerType.CRON, cron_expression="0 1 * * *")
    assert t.next_after(0.0) == pytest.approx(3600.0)


@pytest.mark.parametrize("ttype", [TriggerType.EVENT, TriggerType.CONDITION, TriggerType.MANUAL])
def test_trigger_non_time_returns_none(ttype):
    kw = {}
    if ttype is TriggerType.EVENT:
        kw["event_name"] = "e"
    if ttype is TriggerType.CONDITION:
        kw["condition_key"] = "k"
    t = ScheduleTrigger(ttype, **kw)
    assert t.next_after(100.0) is None


def test_trigger_end_time_bounds():
    t = ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0,
                        start_time=0.0, end_time=15.0)
    assert t.next_after(11.0) is None  # next would be 20 > 15


@pytest.mark.parametrize("ttype", ALL_TRIGGER_TYPES)
def test_trigger_roundtrip(ttype):
    kw = {}
    if ttype is TriggerType.FIXED_INTERVAL:
        kw["interval_seconds"] = 5.0
    if ttype is TriggerType.EVENT:
        kw["event_name"] = "e"
    if ttype is TriggerType.CONDITION:
        kw["condition_key"] = "k"
    if ttype is TriggerType.CRON:
        kw["cron_expression"] = "* * * * *"
    t = ScheduleTrigger(ttype, **kw)
    assert ScheduleTrigger.from_dict(t.to_dict()).to_dict() == t.to_dict()


def test_trigger_validation_interval():
    with pytest.raises(TriggerError):
        ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=0.0)


def test_trigger_validation_event_name():
    with pytest.raises(TriggerError):
        ScheduleTrigger(TriggerType.EVENT)


def test_trigger_validation_condition_key():
    with pytest.raises(TriggerError):
        ScheduleTrigger(TriggerType.CONDITION)


def test_trigger_validation_cron():
    with pytest.raises(TriggerError):
        ScheduleTrigger(TriggerType.CRON, cron_expression="bad")


def test_trigger_validation_weekday():
    with pytest.raises(TriggerError):
        ScheduleTrigger(TriggerType.WEEKLY, weekday=9)


# ---------------------------------------------------------------------------
# Cron field parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expr,minute,hour", [
    ("0 0 * * *", {0}, {0}),
    ("*/30 * * * *", {0, 30}, set(range(24))),
    ("0 9-17 * * *", {0}, set(range(9, 18))),
    ("0,15,30,45 * * * *", {0, 15, 30, 45}, set(range(24))),
])
def test_cron_parse(expr, minute, hour):
    cron = sch._CronExpr(expr)
    assert cron.minute == minute
    assert cron.hour == hour


def test_cron_field_out_of_range():
    with pytest.raises(TriggerError):
        sch._parse_cron_field("99", 0, 59)


# ---------------------------------------------------------------------------
# CalendarRule / ExecutionWindow
# ---------------------------------------------------------------------------
def test_calendar_always_on():
    cal = CalendarRule.always_on()
    assert cal.is_allowed(0.0) and cal.is_allowed(1e6)


def test_calendar_business_hours():
    cal = CalendarRule(business_start_hour=9, business_end_hour=17,
                       working_days=(0, 1, 2, 3, 4))
    assert cal.is_allowed(10 * 3600.0) is True
    assert cal.is_allowed(3 * 3600.0) is False


def test_calendar_weekend_blocked():
    cal = CalendarRule(working_days=(0, 1, 2, 3, 4))
    assert cal.is_allowed(2 * 86400.0) is False  # Saturday


def test_calendar_blackout():
    cal = CalendarRule(blackout_periods=((0.0, 3600.0),))
    assert cal.is_allowed(1800.0) is False


def test_calendar_next_allowed():
    cal = CalendarRule(business_start_hour=9, business_end_hour=17)
    nxt = cal.next_allowed(3 * 3600.0)
    assert cal.is_allowed(nxt)


def test_calendar_roundtrip():
    cal = CalendarRule(timezone_offset_minutes=60, holidays=("2026-01-01",),
                       blackout_periods=((0.0, 10.0),))
    assert CalendarRule.from_dict(cal.to_dict()).to_dict() == cal.to_dict()


def test_calendar_invalid_hours():
    with pytest.raises(JobValidationError):
        CalendarRule(business_start_hour=20, business_end_hour=9)


def test_window_within():
    w = ExecutionWindow(start_second=9 * 3600, end_second=17 * 3600,
                        working_days=(0, 1, 2, 3, 4))
    assert w.is_within(10 * 3600.0) is True
    assert w.is_within(3 * 3600.0) is False


def test_window_roundtrip():
    w = ExecutionWindow(start_second=100, end_second=200, working_days=(0, 1))
    assert ExecutionWindow.from_dict(w.to_dict()).to_dict() == w.to_dict()


def test_window_invalid():
    with pytest.raises(JobValidationError):
        ExecutionWindow(start_second=200, end_second=100)


# ---------------------------------------------------------------------------
# SchedulePolicy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_policy_roundtrip(policy):
    p = SchedulePolicy(execution_policy=policy, max_retries=2, timeout=5.0, priority=3)
    assert SchedulePolicy.from_dict(p.to_dict()).to_dict() == p.to_dict()


@pytest.mark.parametrize("kw", [
    {"max_retries": 0}, {"retry_backoff": -1.0}, {"timeout": -1.0}, {"max_queue": 0},
])
def test_policy_validation(kw):
    base = {"execution_policy": ExecutionPolicy.RETRY}
    base.update(kw)
    with pytest.raises(JobValidationError):
        SchedulePolicy(**base)


# ---------------------------------------------------------------------------
# ScheduledJob
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("jtype", ALL_JOB_TYPES)
def test_job_roundtrip(jtype):
    j = ScheduledJob("j1", "Job", jtype, trig(TriggerType.MANUAL),
                     depends_on=("d",), children=("c",), payload_json='{"k":1}')
    assert ScheduledJob.from_dict(j.to_dict()).to_dict() == j.to_dict()


def test_job_roundtrip_with_window():
    j = job("j1", trigger=trig(TriggerType.MANUAL), window=ExecutionWindow(0, 3600))
    assert ScheduledJob.from_dict(j.to_dict()).to_dict() == j.to_dict()


def test_job_self_dependency():
    with pytest.raises(JobValidationError):
        job("j1", depends_on=("j1",))


@pytest.mark.parametrize("bad", [{"job_id": ""}, {"name": ""}])
def test_job_validation(bad):
    base = {"job_id": "j", "name": "j", "job_type": JobType.CUSTOM, "trigger": trig()}
    base.update(bad)
    with pytest.raises(JobValidationError):
        ScheduledJob(**base)


def test_job_priority_from_policy():
    j = job("j", trigger=trig(), policy=SchedulePolicy(priority=7))
    assert j.priority == 7


# ---------------------------------------------------------------------------
# JobOutcome / JobExecution / JobHistory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("success", [True, False])
def test_job_outcome_roundtrip(success):
    o = JobOutcome(success, '{"k":1}', "" if success else "err", 2.0)
    assert JobOutcome.from_dict(o.to_dict()).to_dict() == o.to_dict()


@pytest.mark.parametrize("status", ALL_JOB_STATUSES)
def test_execution_roundtrip(status):
    e = JobExecution("e", "j", 1.0, 2.0, 5.0, status, 2, "manual", "", '{"o":1}')
    assert JobExecution.from_dict(e.to_dict()).to_dict() == e.to_dict()
    assert e.duration == pytest.approx(3.0)
    assert e.latency == pytest.approx(1.0)


def test_history_append_immutable():
    h0 = JobHistory()
    h1 = h0.append(ex_record())
    assert len(h0) == 0 and len(h1) == 1


def test_history_for_job_and_status():
    h = JobHistory((ex_record("a", JobStatus.SUCCESS), ex_record("b", JobStatus.FAILURE)))
    assert len(h.for_job("a")) == 1
    assert len(h.by_status(JobStatus.FAILURE)) == 1


def test_history_last():
    h = JobHistory(tuple(ex_record(f"j{i}") for i in range(5)))
    assert len(h.last(2)) == 2
    assert h.last(0) == ()


def test_history_max_history():
    h = JobHistory()
    for i in range(10):
        h = h.append(ex_record(f"j{i}"), max_history=3)
    assert len(h) == 3


def test_history_roundtrip():
    h = JobHistory((ex_record("a"), ex_record("b")))
    assert JobHistory.from_dict(h.to_dict()).to_dict() == h.to_dict()


# ---------------------------------------------------------------------------
# ScheduleResult / ScheduleStatistics / AutomationRule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_JOB_STATES)
def test_schedule_result_roundtrip(state):
    r = ScheduleResult("j", ex_record(), state, rescheduled_for=5.0)
    assert ScheduleResult.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_statistics_from_history():
    h = JobHistory((
        ex_record("a", JobStatus.SUCCESS),
        ex_record("a", JobStatus.FAILURE),
        ex_record("b", JobStatus.SUCCESS, attempts=3),
    ))
    stats = ScheduleStatistics.from_history(h, scheduled_jobs=2)
    assert stats.completed_jobs == 2
    assert stats.failed_jobs == 1
    assert stats.retry_count == 2
    assert stats.total_executions == 3


def test_statistics_success_rate():
    h = JobHistory((ex_record("a", JobStatus.SUCCESS), ex_record("b", JobStatus.FAILURE)))
    stats = ScheduleStatistics.from_history(h)
    assert stats.automation_success_rate == pytest.approx(0.5)


def test_statistics_roundtrip():
    s = ScheduleStatistics(5, 3, 1, 2, 1.5, 0.75, 4, 0.2, 1, 6, 1, 0)
    assert ScheduleStatistics.from_dict(s.to_dict()).to_dict() == s.to_dict()


@pytest.mark.parametrize("atype", ALL_AUTOMATION)
def test_automation_rule_roundtrip(atype):
    r = AutomationRule("r1", atype, "ref", ("t1", "t2"), condition_value=True)
    assert AutomationRule.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_automation_rule_requires_target():
    with pytest.raises(JobValidationError):
        AutomationRule("r", AutomationType.EVENT_DRIVEN, "e", ())


# ---------------------------------------------------------------------------
# Scheduler: registry
# ---------------------------------------------------------------------------
def test_register_and_list():
    s = make_scheduler()
    s.register_job(interval_job("a"))
    assert s.exists("a")
    assert [j.job_id for j in s.list_jobs()] == ["a"]


def test_register_duplicate():
    s = make_scheduler()
    s.register_job(interval_job("a"))
    with pytest.raises(SchedulerError):
        s.register_job(interval_job("a"))


def test_register_non_job():
    with pytest.raises(JobValidationError):
        make_scheduler().register_job("not-a-job")


def test_remove_job():
    s = make_scheduler()
    s.register_job(interval_job("a"))
    s.remove_job("a")
    assert not s.exists("a")


def test_remove_unknown():
    with pytest.raises(JobNotFoundError):
        make_scheduler().remove_job("ghost")


def test_pause_resume():
    s = make_scheduler()
    s.register_job(interval_job("a"))
    s.pause_job("a")
    assert s.job_state("a") is JobState.PAUSED
    s.resume_job("a")
    assert s.job_state("a") is JobState.SCHEDULED


def test_paused_job_not_fired():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.pause_job("a")
    results = s.advance_to(30.0)
    assert results == ()


def test_next_execution():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    assert s.next_execution("a") == pytest.approx(0.0)


def test_next_execution_unknown():
    with pytest.raises(JobNotFoundError):
        make_scheduler().next_execution("ghost")


# ---------------------------------------------------------------------------
# Scheduler: time-based firing
# ---------------------------------------------------------------------------
def test_tick_fires_due_job():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    results = s.tick(0.0)
    assert len(results) == 1
    assert results[0].succeeded


def test_advance_to_catch_up():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    results = s.advance_to(30.0)
    times = [r.execution.scheduled_time for r in results]
    assert times == [0.0, 10.0, 20.0, 30.0]


def test_one_time_fires_once():
    s = make_scheduler()
    s.register_job(job("a", JobType.PREDICTION_REFRESH,
                       ScheduleTrigger(TriggerType.ONE_TIME, start_time=5.0)))
    s.advance_to(100.0)
    assert len(s.execution_history("a")) == 1
    assert s.next_execution("a") is None


def test_max_occurrences():
    s = make_scheduler()
    s.register_job(job("a", JobType.HEALTH_MONITORING,
                       ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0,
                                       start_time=0.0, max_occurrences=2)))
    s.advance_to(100.0)
    assert len(s.execution_history("a")) == 2


def test_run_now():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    result = s.run_now("a")
    assert result.succeeded


def test_run_now_unknown():
    with pytest.raises(JobNotFoundError):
        make_scheduler().run_now("ghost")


def test_advance_no_jobs():
    s = make_scheduler()
    assert s.advance_to(100.0) == ()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def test_dependency_runs_after_completion():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0)))
    s.register_job(job("b", JobType.CUSTOM, ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0),
                       depends_on=("a",)))
    results = s.advance_to(0.0)
    states = {r.job_id: r.state for r in results}
    assert states["a"] is JobState.COMPLETED
    assert states["b"] is JobState.COMPLETED


def test_dependency_blocks_when_unmet():
    s = make_scheduler()
    # 'a' is an event job that never fires here; 'b' depends on it.
    s.register_job(job("a", JobType.CUSTOM, ScheduleTrigger(TriggerType.EVENT, event_name="x")))
    s.register_job(job("b", JobType.CUSTOM, ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0),
                       depends_on=("a",)))
    results = s.advance_to(0.0)
    assert any(r.job_id == "b" and r.state is JobState.BLOCKED for r in results)


def test_dependency_cycle_detected():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL), depends_on=("b",)))
    with pytest.raises(DependencyCycleError):
        s.register_job(job("b", JobType.CUSTOM, trig(TriggerType.MANUAL), depends_on=("a",)))


def test_blocking_jobs():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_job(job("b", JobType.CUSTOM, trig(TriggerType.MANUAL), depends_on=("a",)))
    assert s.blocking_jobs() == ("a",)


# ---------------------------------------------------------------------------
# Automation: events, conditions, chained
# ---------------------------------------------------------------------------
def test_event_triggered_job():
    s = make_scheduler()
    s.register_job(job("a", JobType.EXECUTIVE_REPORT,
                       ScheduleTrigger(TriggerType.EVENT, event_name="risk")))
    results = s.fire_event("risk")
    assert len(results) == 1 and results[0].succeeded


def test_event_no_match():
    s = make_scheduler()
    s.register_job(job("a", JobType.EXECUTIVE_REPORT,
                       ScheduleTrigger(TriggerType.EVENT, event_name="risk")))
    assert s.fire_event("other") == ()


def test_condition_triggered_job():
    s = make_scheduler()
    s.register_job(job("a", JobType.RISK_ASSESSMENT,
                       ScheduleTrigger(TriggerType.CONDITION, condition_key="degraded")))
    results = s.evaluate_conditions({"degraded": True})
    assert len(results) == 1
    assert s.evaluate_conditions({"degraded": False}) == ()


def test_chained_children():
    s = make_scheduler()
    s.register_job(job("child", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_job(job("parent", JobType.CUSTOM, ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0),
                       children=("child",)))
    s.advance_to(0.0)
    assert len(s.execution_history("child")) == 1


def test_automation_rule_event():
    s = make_scheduler()
    s.register_job(job("t", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_automation(AutomationRule("r", AutomationType.EVENT_DRIVEN, "evt", ("t",)))
    s.fire_event("evt")
    assert len(s.execution_history("t")) == 1


def test_automation_rule_condition():
    s = make_scheduler()
    s.register_job(job("t", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_automation(AutomationRule("r", AutomationType.CONDITION_BASED, "flag",
                                         ("t",), condition_value=True))
    s.evaluate_conditions({"flag": True})
    assert len(s.execution_history("t")) == 1


def test_automation_rule_chained():
    s = make_scheduler()
    s.register_job(job("t", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_job(job("p", JobType.CUSTOM, ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0)))
    s.register_automation(AutomationRule("r", AutomationType.CHAINED, "p", ("t",)))
    s.advance_to(0.0)
    assert len(s.execution_history("t")) == 1


def test_automation_duplicate_rule():
    s = make_scheduler()
    s.register_automation(AutomationRule("r", AutomationType.EVENT_DRIVEN, "e", ("t",)))
    with pytest.raises(SchedulerError):
        s.register_automation(AutomationRule("r", AutomationType.EVENT_DRIVEN, "e", ("t",)))


# ---------------------------------------------------------------------------
# Retry / timeout / policies
# ---------------------------------------------------------------------------
def test_retry_to_max():
    s = make_scheduler(executor=fail_executor)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.RETRY, max_retries=3)))
    result = s.run_now("a")
    assert result.execution.attempts == 3
    assert result.execution.status is JobStatus.FAILURE


def test_retry_success_first_try():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.RETRY, max_retries=3)))
    result = s.run_now("a")
    assert result.execution.attempts == 1
    assert result.succeeded


def test_timeout():
    s = make_scheduler(executor=slow_executor(5.0))
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.RETRY, max_retries=2, timeout=1.0)))
    result = s.run_now("a")
    assert result.execution.status is JobStatus.TIMEOUT


def test_skip_policy_no_retry():
    s = make_scheduler(executor=fail_executor)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.SKIP, max_retries=3)))
    result = s.run_now("a")
    assert result.execution.attempts == 1  # SKIP does not retry


def test_run_once_policy_stops_recurrence():
    s = make_scheduler()
    s.register_job(job("a", JobType.HEALTH_MONITORING,
                       ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0,
                                       start_time=0.0),
                       policy=SchedulePolicy(ExecutionPolicy.RUN_ONCE)))
    s.advance_to(100.0)
    assert len(s.execution_history("a")) == 1


def test_cancel_policy_stops_recurrence():
    s = make_scheduler()
    s.register_job(job("a", JobType.HEALTH_MONITORING,
                       ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0,
                                       start_time=0.0),
                       policy=SchedulePolicy(ExecutionPolicy.CANCEL)))
    s.advance_to(100.0)
    assert len(s.execution_history("a")) == 1


def test_executor_exception_recorded():
    def boom(j, c, k):
        raise RuntimeError("crash")
    s = make_scheduler(executor=boom)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.RETRY, max_retries=2)))
    result = s.run_now("a")
    assert result.execution.status is JobStatus.FAILURE
    assert "crash" in result.execution.error


def test_retry_backoff_advances_clock():
    s = make_scheduler(executor=fail_executor)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL),
                       policy=SchedulePolicy(ExecutionPolicy.RETRY, max_retries=3,
                                             retry_backoff=2.0)))
    result = s.run_now("a")
    # backoff applied before attempts 2 and 3 -> 2 + 4 = 6 logical seconds
    assert result.execution.duration == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Statistics from a running scheduler
# ---------------------------------------------------------------------------
def test_scheduler_statistics():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.advance_to(30.0)
    stats = s.scheduler_statistics()
    assert stats.scheduled_jobs == 1
    assert stats.completed_jobs == 4
    assert stats.total_executions == 4


def test_job_statistics():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.advance_to(20.0)
    stats = s.job_statistics("a")
    assert stats.completed_jobs == 3


# ---------------------------------------------------------------------------
# Event bus integration (composition with Phase 3)
# ---------------------------------------------------------------------------
def test_emits_lifecycle_events_internally():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.run_now("a")
    names = [n for n, _ in s.emitted_events]
    assert "job_registered" in names
    assert "job_started" in names
    assert "job_completed" in names


@pytest.mark.skipif(eeb is None, reason="event bus module unavailable")
def test_publishes_to_event_bus():
    bus = eeb.create_default_event_bus()
    received = []
    bus.subscribe(lambda e: received.append(e), topic_pattern="scheduler.#")
    s = create_default_scheduler(event_bus=bus)
    s.register_executor(JobType.CUSTOM, ok_executor)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.run_now("a")
    topics = [e.topic for e in received]
    assert "scheduler.job_registered" in topics
    assert "scheduler.job_completed" in topics


@pytest.mark.skipif(eeb is None, reason="event bus module unavailable")
def test_event_bus_failure_event():
    bus = eeb.create_default_event_bus()
    received = []
    bus.subscribe(lambda e: received.append(e.topic), topic_pattern="scheduler.job_failed")
    s = create_default_scheduler(event_bus=bus)
    s.register_executor(JobType.CUSTOM, fail_executor)
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.run_now("a")
    assert "scheduler.job_failed" in received


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_determinism_identical_history():
    def build():
        s = make_scheduler()
        s.register_job(interval_job("a", interval=10.0, start=0.0))
        s.register_job(interval_job("b", interval=15.0, start=0.0))
        s.advance_to(60.0)
        return s.execution_history().to_dict()

    assert build() == build()


def test_determinism_emitted_events():
    def emit():
        s = make_scheduler()
        s.register_job(interval_job("a", interval=10.0, start=0.0))
        s.advance_to(30.0)
        return [n for n, _ in s.emitted_events]

    assert emit() == emit()


def test_determinism_priority_order():
    s = make_scheduler()
    order = []
    s.register_executor(JobType.CUSTOM, lambda j, c, k: (order.append(j.job_id) or JobOutcome.succeeded()))
    s.register_job(job("low", JobType.CUSTOM,
                       ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0),
                       policy=SchedulePolicy(priority=1)))
    s.register_job(job("high", JobType.CUSTOM,
                       ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0),
                       policy=SchedulePolicy(priority=10)))
    s.tick(0.0)
    assert order == ["high", "low"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
def test_thread_safe_registration():
    s = make_scheduler()

    def worker(i):
        s.register_job(interval_job(f"j{i}", interval=10.0, start=0.0))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(s.list_jobs()) == 40


def test_thread_safe_run_now():
    s = make_scheduler()
    for i in range(20):
        s.register_job(job(f"j{i}", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    errors = []

    def worker(i):
        try:
            s.run_now(f"j{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(s.execution_history()) == 20


# ---------------------------------------------------------------------------
# Scale / performance
# ---------------------------------------------------------------------------
def test_large_job_set():
    s = make_scheduler()
    for i in range(100):
        s.register_job(job(f"j{i}", JobType.CUSTOM,
                           ScheduleTrigger(TriggerType.ONE_TIME, start_time=0.0)))
    results = s.advance_to(0.0)
    assert len(results) == 100
    assert all(r.succeeded for r in results)


def test_recurring_large_horizon():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=1.0, start=0.0))
    s.advance_to(200.0)
    assert len(s.execution_history("a")) == 201  # t=0..200 inclusive


# ---------------------------------------------------------------------------
# Frozen / slots / JSON
# ---------------------------------------------------------------------------
_FROZEN = [
    ScheduleTrigger(TriggerType.MANUAL),
    ExecutionWindow(),
    CalendarRule(),
    SchedulePolicy(),
    job("j", trigger=trig(TriggerType.MANUAL)),
    JobOutcome.succeeded(),
    ex_record(),
    JobHistory(),
    ScheduleResult("j", ex_record(), JobState.COMPLETED),
    ScheduleStatistics(),
    AutomationRule("r", AutomationType.EVENT_DRIVEN, "e", ("t",)),
]


@pytest.mark.parametrize("instance", _FROZEN)
def test_frozen(instance):
    import dataclasses
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, getattr(instance, field_name))


@pytest.mark.parametrize("instance", _FROZEN)
def test_slots(instance):
    assert not hasattr(instance, "__dict__")


@pytest.mark.parametrize("factory", [
    lambda: ScheduleTrigger(TriggerType.FIXED_INTERVAL, interval_seconds=10.0).to_dict(),
    lambda: job("j", trigger=trig(TriggerType.MANUAL)).to_dict(),
    lambda: SchedulePolicy().to_dict(),
    lambda: CalendarRule().to_dict(),
    lambda: ScheduleStatistics(1, 1).to_dict(),
    lambda: AutomationRule("r", AutomationType.CHAINED, "p", ("t",)).to_dict(),
    lambda: ex_record().to_dict(),
])
def test_json_serializable(factory):
    payload = factory()
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
def test_logical_clock():
    clk = sch.LogicalClock()
    clk.advance(5.0)
    assert clk.now() == 5.0


def test_logical_clock_negative():
    with pytest.raises(SchedulerError):
        sch.LogicalClock().advance(-1.0)


def test_system_clock_now():
    assert isinstance(sch.SystemClock().now(), float)


# ---------------------------------------------------------------------------
# Backward compatibility / non-invasiveness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "import apscheduler", "from apscheduler", "import celery", "import croniter",
    "import asyncio", "import schedule",
])
def test_no_forbidden_imports(forbidden):
    with open(sch.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert forbidden not in source


def test_does_not_import_upstream_modules():
    with open(sch.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "workflow_engine" not in source
    assert "business_process_orchestrator" not in source


def test_factory_builds_scheduler():
    assert isinstance(create_default_scheduler(), EnterpriseScheduler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_demo():
    assert sch.main(["--demo"]) == 0


def test_cli_no_args():
    assert sch.main([]) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_resume_recomputes_next_execution():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.advance_to(100.0)  # recurring, still scheduled
    s.pause_job("a")
    s.resume_job("a")
    assert s.next_execution("a") is not None


def test_emergency_job_fires_once_on_advance():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, ScheduleTrigger(TriggerType.EMERGENCY, start_time=0.0)))
    s.advance_to(50.0)
    assert len(s.execution_history("a")) == 1


def test_remove_clears_schedule():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.remove_job("a")
    assert s.advance_to(100.0) == ()


def test_history_filter_by_job_isolation():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_job(job("b", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.run_now("a")
    s.run_now("b")
    assert len(s.execution_history("a")) == 1
    assert len(s.execution_history()) == 2


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------
def test_job_outcome_failed_factory():
    o = JobOutcome.failed("err", duration=2.0)
    assert o.success is False and o.error == "err" and o.duration == 2.0


def test_job_outcome_negative_duration():
    with pytest.raises(JobValidationError):
        JobOutcome(True, "{}", "", -1.0)


def test_history_last_negative():
    with pytest.raises(JobValidationError):
        JobHistory().last(-1)


def test_calendar_maintenance_window():
    cal = CalendarRule(maintenance_windows=((100.0, 200.0),))
    assert cal.in_maintenance_window(150.0) is True
    assert cal.in_maintenance_window(250.0) is False


def test_scheduler_statistics_empty():
    stats = make_scheduler().scheduler_statistics()
    assert stats.scheduled_jobs == 0
    assert stats.total_executions == 0


def test_automation_rule_disabled_does_not_fire():
    s = make_scheduler()
    s.register_job(job("t", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.register_automation(AutomationRule("r", AutomationType.EVENT_DRIVEN, "evt", ("t",),
                                         enabled=False))
    s.fire_event("evt")
    assert len(s.execution_history("t")) == 0


def test_daily_trigger_timezone_offset():
    # 01:00 local with a +60 minute offset == 00:00 UTC
    t = ScheduleTrigger(TriggerType.DAILY, at_second=3600, timezone_offset_minutes=60)
    assert t.next_after(0.0) == pytest.approx(0.0)


def test_run_now_on_paused_job():
    s = make_scheduler()
    s.register_job(job("a", JobType.CUSTOM, trig(TriggerType.MANUAL)))
    s.pause_job("a")
    result = s.run_now("a")  # explicit run bypasses pause
    assert result.succeeded


def test_job_state_unknown_raises():
    with pytest.raises(JobNotFoundError):
        make_scheduler().job_state("ghost")


def test_tick_advances_clock():
    s = make_scheduler()
    s.register_job(interval_job("a", interval=10.0, start=0.0))
    s.tick(50.0)
    # next_execution should now be in the future relative to advanced clock
    assert s.next_execution("a") >= 50.0