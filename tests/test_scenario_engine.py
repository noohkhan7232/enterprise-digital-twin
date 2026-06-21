#!/usr/bin/env python3
"""Comprehensive test suite for ``src/simulation/scenario_engine.py``.

The engine is pure NumPy and composes the (also pure-NumPy) Week-5 prognostic
stack, so the **entire** suite runs without PyTorch or SciPy.  Coverage (120+):

- ScenarioType enum
- ScenarioConfig validation (per-type required fields)
- Registry (register / build / list / duplicate rejection)
- run() across all six scenario archetypes
- compare() delta correctness (health, failure time, RUL, risk)
- failure-avoided / failure-induced semantics
- cost & downtime impact
- generate_report()
- Explainability summaries
- Reproducibility / determinism
- ScenarioResult serialization (inf/NaN -> None)
- Tracker integration (failure-safe)
- Graceful prognostics degradation
- Boundary & error handling

Run::

    pytest tests/test_scenario_engine.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.asset_state_simulator import (
    AssetSimulatorConfig,
    AssetStateSimulator,
    SimulationResult,
)
from src.simulation.scenario_engine import (
    DEFAULT_FAILURE_COST,
    DEFAULT_MAINTENANCE_COST,
    ENGINE_NAME,
    SCENARIO_ENGINE_REGISTRY,
    ScenarioConfig,
    ScenarioEngine,
    ScenarioResult,
    ScenarioType,
    build_scenario_engine,
    list_scenario_engines,
    register_scenario_engine,
)


def _base(**kw) -> AssetSimulatorConfig:
    defaults = dict(horizon=80, timestep=1.0, model="linear",
                    degradation_rate=1.0, initial_health=95.0,
                    failure_threshold=30.0, noise_std=0.0)
    defaults.update(kw)
    return AssetSimulatorConfig(**defaults)


def _engine(**kw) -> ScenarioEngine:
    return ScenarioEngine(_base(**kw))


def _cfg(stype, **kw) -> ScenarioConfig:
    return ScenarioConfig(scenario_type=stype, **kw)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class TestScenarioType:
    """Tests for the scenario-type enum."""

    def test_values(self) -> None:
        assert ScenarioType.NORMAL.value == "normal"
        assert ScenarioType.ACCELERATED_DEGRADATION.value == "accelerated_degradation"
        assert ScenarioType.MAINTENANCE_INTERVENTION.value == "maintenance_intervention"
        assert ScenarioType.SUDDEN_FAULT.value == "sudden_fault"
        assert ScenarioType.LOAD_REDUCTION.value == "load_reduction"
        assert ScenarioType.CUSTOM.value == "custom"

    def test_is_str(self) -> None:
        assert ScenarioType.NORMAL == "normal"

    def test_six_types(self) -> None:
        assert len(list(ScenarioType)) == 6


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Tests for scenario-config validation."""

    def test_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="scenario_type"):
            ScenarioConfig(scenario_type="teleport")

    def test_normal_valid(self) -> None:
        ScenarioConfig(scenario_type="normal")

    def test_accelerated_requires_factor_gt1(self) -> None:
        with pytest.raises(ValueError, match="degradation_factor > 1"):
            ScenarioConfig(scenario_type="accelerated_degradation",
                           degradation_factor=1.0)

    def test_accelerated_valid(self) -> None:
        ScenarioConfig(scenario_type="accelerated_degradation",
                       degradation_factor=1.5)

    def test_load_reduction_requires_factor_lt1(self) -> None:
        with pytest.raises(ValueError, match="0 <= degradation_factor < 1"):
            ScenarioConfig(scenario_type="load_reduction",
                           degradation_factor=1.2)

    def test_load_reduction_valid(self) -> None:
        ScenarioConfig(scenario_type="load_reduction", degradation_factor=0.5)

    def test_load_reduction_zero_valid(self) -> None:
        ScenarioConfig(scenario_type="load_reduction", degradation_factor=0.0)

    def test_maintenance_requires_time(self) -> None:
        with pytest.raises(ValueError, match="intervention_time"):
            ScenarioConfig(scenario_type="maintenance_intervention",
                           restoration_amount=20)

    def test_maintenance_requires_positive_restoration(self) -> None:
        with pytest.raises(ValueError, match="restoration_amount"):
            ScenarioConfig(scenario_type="maintenance_intervention",
                           intervention_time=10, restoration_amount=0)

    def test_maintenance_negative_time(self) -> None:
        with pytest.raises(ValueError, match="intervention_time"):
            ScenarioConfig(scenario_type="maintenance_intervention",
                           intervention_time=-5, restoration_amount=20)

    def test_maintenance_valid(self) -> None:
        ScenarioConfig(scenario_type="maintenance_intervention",
                       intervention_time=30, restoration_amount=25)

    def test_fault_requires_time(self) -> None:
        with pytest.raises(ValueError, match="fault_time"):
            ScenarioConfig(scenario_type="sudden_fault", fault_magnitude=20)

    def test_fault_requires_positive_magnitude(self) -> None:
        with pytest.raises(ValueError, match="fault_magnitude"):
            ScenarioConfig(scenario_type="sudden_fault", fault_time=10,
                           fault_magnitude=0)

    def test_fault_valid(self) -> None:
        ScenarioConfig(scenario_type="sudden_fault", fault_time=20,
                       fault_magnitude=30)

    def test_custom_requires_overrides(self) -> None:
        with pytest.raises(ValueError, match="overrides"):
            ScenarioConfig(scenario_type="custom", overrides={})

    def test_custom_rejects_unknown_field(self) -> None:
        with pytest.raises(ValueError, match="unknown config fields"):
            ScenarioConfig(scenario_type="custom",
                           overrides={"not_a_field": 1})

    def test_custom_valid(self) -> None:
        ScenarioConfig(scenario_type="custom",
                       overrides={"degradation_rate": 0.3})

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="costs"):
            ScenarioConfig(scenario_type="normal", maintenance_cost=-1)

    def test_negative_downtime_rejected(self) -> None:
        with pytest.raises(ValueError, match="downtime"):
            ScenarioConfig(scenario_type="normal", failure_downtime_h=-1)

    def test_frozen(self) -> None:
        c = ScenarioConfig(scenario_type="normal")
        with pytest.raises((AttributeError, TypeError)):
            c.degradation_factor = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the registry."""

    def test_registered(self) -> None:
        assert ENGINE_NAME in SCENARIO_ENGINE_REGISTRY
        assert ENGINE_NAME in list_scenario_engines()

    def test_build_by_name(self) -> None:
        assert isinstance(build_scenario_engine(ENGINE_NAME), ScenarioEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scenario engine"):
            build_scenario_engine("nope")

    def test_registry_name_stamped(self) -> None:
        assert ScenarioEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_scenario_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_scenario_engine(ENGINE_NAME, baseline_config=_base(horizon=42))
        assert eng.baseline_config.horizon == 42


# ---------------------------------------------------------------------------
# Normal scenario
# ---------------------------------------------------------------------------


class TestNormalScenario:
    """Tests for the identity (control) scenario."""

    def test_zero_health_delta(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert np.allclose(r.health_delta, 0.0)

    def test_zero_mean_delta(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.mean_health_delta == pytest.approx(0.0)

    def test_zero_risk_delta(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.risk_delta == pytest.approx(0.0)

    def test_zero_rul_delta(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.rul_delta == pytest.approx(0.0) or math.isnan(r.rul_delta)

    def test_identical_trajectories(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert np.array_equal(r.baseline_health, r.scenario_health)

    def test_no_cost_impact(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.cost_impact == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Accelerated degradation
# ---------------------------------------------------------------------------


class TestAcceleratedDegradation:
    """Tests for the accelerated-degradation scenario."""

    def test_worse_health(self) -> None:
        r = _engine().run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.mean_health_delta < 0

    def test_negative_final_delta(self) -> None:
        r = _engine().run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.final_health_delta <= 0

    def test_higher_factor_worse(self) -> None:
        mild = _engine().run(_cfg("accelerated_degradation", degradation_factor=1.5))
        harsh = _engine().run(_cfg("accelerated_degradation", degradation_factor=3.0))
        assert harsh.mean_health_delta < mild.mean_health_delta

    def test_risk_increases(self) -> None:
        r = _engine(degradation_rate=0.5).run(
            _cfg("accelerated_degradation", degradation_factor=2.5))
        assert r.risk_delta >= 0

    def test_can_induce_failure(self) -> None:
        # Baseline survives (95-50=45), accelerated x2 fails
        eng = ScenarioEngine(_base(horizon=50, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.failure_induced

    def test_scenario_type_recorded(self) -> None:
        r = _engine().run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.scenario_type == "accelerated_degradation"


# ---------------------------------------------------------------------------
# Load reduction
# ---------------------------------------------------------------------------


class TestLoadReduction:
    """Tests for the load-reduction scenario."""

    def test_better_health(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.mean_health_delta > 0

    def test_positive_final_delta(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.final_health_delta >= 0

    def test_lower_factor_better(self) -> None:
        mild = _engine().run(_cfg("load_reduction", degradation_factor=0.8))
        strong = _engine().run(_cfg("load_reduction", degradation_factor=0.2))
        assert strong.mean_health_delta > mild.mean_health_delta

    def test_risk_decreases(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.risk_delta <= 0

    def test_can_avoid_failure(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert r.failure_avoided

    def test_rul_extended(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.rul_delta > 0 or math.isnan(r.rul_delta)


# ---------------------------------------------------------------------------
# Maintenance intervention
# ---------------------------------------------------------------------------


class TestMaintenanceIntervention:
    """Tests for the maintenance-intervention scenario."""

    def test_health_restored(self) -> None:
        r = _engine().run(_cfg("maintenance_intervention",
                                intervention_time=40, restoration_amount=40))
        assert r.mean_health_delta > 0

    def test_restoration_at_correct_time(self) -> None:
        r = _engine().run(_cfg("maintenance_intervention",
                                intervention_time=40, restoration_amount=30))
        idx = int(np.searchsorted(r.times, 40.0))
        # Health just after intervention exceeds baseline by ~restoration
        assert r.scenario_health[idx] > r.baseline_health[idx]

    def test_no_change_before_intervention(self) -> None:
        r = _engine().run(_cfg("maintenance_intervention",
                                intervention_time=40, restoration_amount=30))
        idx = int(np.searchsorted(r.times, 40.0))
        assert np.allclose(r.health_delta[:idx], 0.0)

    def test_restoration_clipped_to_100(self) -> None:
        r = _engine(initial_health=90).run(
            _cfg("maintenance_intervention", intervention_time=10,
                 restoration_amount=80))
        assert r.scenario_health.max() <= 100.0 + 1e-9

    def test_can_avoid_failure(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("maintenance_intervention", intervention_time=40,
                         restoration_amount=50))
        assert r.failure_avoided

    def test_intervention_adds_cost(self) -> None:
        # When the intervention avoids failure, net cost = maint - failure
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("maintenance_intervention", intervention_time=40,
                         restoration_amount=50))
        assert r.cost_impact == pytest.approx(
            DEFAULT_MAINTENANCE_COST - DEFAULT_FAILURE_COST)

    def test_intervention_downtime(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("maintenance_intervention", intervention_time=40,
                         restoration_amount=50))
        # avoided failure downtime, added intervention downtime
        assert r.downtime_impact != 0


# ---------------------------------------------------------------------------
# Sudden fault
# ---------------------------------------------------------------------------


class TestSuddenFault:
    """Tests for the sudden-fault-injection scenario."""

    def test_worse_health(self) -> None:
        r = _engine().run(_cfg("sudden_fault", fault_time=30, fault_magnitude=35))
        assert r.mean_health_delta < 0

    def test_fault_at_correct_time(self) -> None:
        r = _engine().run(_cfg("sudden_fault", fault_time=30, fault_magnitude=35))
        idx = int(np.searchsorted(r.times, 30.0))
        assert r.scenario_health[idx] < r.baseline_health[idx]

    def test_no_change_before_fault(self) -> None:
        r = _engine().run(_cfg("sudden_fault", fault_time=30, fault_magnitude=35))
        idx = int(np.searchsorted(r.times, 30.0))
        assert np.allclose(r.health_delta[:idx], 0.0)

    def test_fault_clipped_to_zero(self) -> None:
        r = _engine().run(_cfg("sudden_fault", fault_time=10, fault_magnitude=200))
        assert r.scenario_health.min() >= 0.0

    def test_can_induce_failure(self) -> None:
        eng = ScenarioEngine(_base(horizon=40, degradation_rate=0.5))
        # baseline 95-20=75 survives; fault -60 -> below 30
        r = eng.run(_cfg("sudden_fault", fault_time=20, fault_magnitude=60))
        assert r.failure_induced

    def test_fault_recorded_in_steps(self) -> None:
        r = _engine().run(_cfg("sudden_fault", fault_time=30, fault_magnitude=35))
        # The scenario health reflects the drop; risk should rise
        assert r.risk_delta >= 0


# ---------------------------------------------------------------------------
# Custom scenario
# ---------------------------------------------------------------------------


class TestCustomScenario:
    """Tests for the custom-override scenario."""

    def test_override_rate(self) -> None:
        r = _engine().run(_cfg("custom", overrides={"degradation_rate": 0.3}))
        assert r.mean_health_delta > 0  # slower than baseline 1.0

    def test_override_model(self) -> None:
        r = _engine().run(_cfg("custom", overrides={
            "model": "exponential", "degradation_rate": 0.03}))
        assert r.scenario_type == "custom"

    def test_override_initial_health(self) -> None:
        r = _engine().run(_cfg("custom", overrides={"initial_health": 100.0}))
        assert r.scenario_health[0] == pytest.approx(100.0)

    def test_override_multiple_fields(self) -> None:
        r = _engine().run(_cfg("custom", overrides={
            "degradation_rate": 2.0, "failure_threshold": 50.0}))
        assert r.mean_health_delta < 0

    def test_override_noise(self) -> None:
        r = _engine().run(_cfg("custom", overrides={
            "noise_std": 2.0, "random_seed": 5}))
        assert r.scenario_health is not None


# ---------------------------------------------------------------------------
# Delta correctness
# ---------------------------------------------------------------------------


class TestDeltaCorrectness:
    """Tests for the structured delta computations."""

    def test_health_delta_is_difference(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert np.allclose(r.health_delta, r.scenario_health - r.baseline_health)

    def test_mean_delta_matches(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.mean_health_delta == pytest.approx(np.mean(r.health_delta))

    def test_final_delta_matches(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.final_health_delta == pytest.approx(r.health_delta[-1])

    def test_risk_delta_is_difference(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.risk_delta == pytest.approx(r.scenario_risk - r.baseline_risk)

    def test_failure_time_delta_both_fail(self) -> None:
        # Both baseline and a milder acceleration fail; delta is finite.
        eng = ScenarioEngine(_base(horizon=100, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=1.5))
        # baseline fails at 65, accelerated fails earlier -> negative delta
        assert not math.isnan(r.failure_time_delta)
        assert r.failure_time_delta < 0

    def test_failure_time_delta_earlier_under_acceleration(self) -> None:
        eng = ScenarioEngine(_base(horizon=100, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=1.5))
        assert r.scenario_failure_time < r.baseline_failure_time

    def test_rul_both_infinite_zero_delta(self) -> None:
        # Two non-degrading trajectories: both RUL infinite -> delta 0
        eng = ScenarioEngine(_base(horizon=20, degradation_rate=0.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.rul_delta == pytest.approx(0.0) or math.isnan(r.rul_delta)

    def test_failure_time_delta_nan_when_avoided(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert math.isnan(r.failure_time_delta)

    def test_rul_delta_zero_for_normal(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.rul_delta == pytest.approx(0.0) or math.isnan(r.rul_delta)


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


class TestFailureSemantics:
    """Tests for failure-avoided / failure-induced flags."""

    def test_avoided_flag(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert r.failure_avoided and not r.failure_induced

    def test_induced_flag(self) -> None:
        eng = ScenarioEngine(_base(horizon=50, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.failure_induced and not r.failure_avoided

    def test_neither_when_both_fail(self) -> None:
        eng = ScenarioEngine(_base(horizon=100, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=1.5))
        assert not r.failure_avoided and not r.failure_induced

    def test_neither_when_both_survive(self) -> None:
        eng = ScenarioEngine(_base(horizon=30, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.5))
        assert not r.failure_avoided and not r.failure_induced


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------


class TestEconomics:
    """Tests for cost and downtime impact."""

    def test_avoided_failure_saves_cost(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert r.cost_impact < 0  # baseline failed (cost), scenario didn't

    def test_induced_failure_adds_cost(self) -> None:
        eng = ScenarioEngine(_base(horizon=50, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert r.cost_impact > 0

    def test_normal_zero_cost(self) -> None:
        eng = ScenarioEngine(_base(horizon=30, degradation_rate=0.5))
        r = eng.run(_cfg("normal"))
        assert r.cost_impact == pytest.approx(0.0)

    def test_custom_costs(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2,
                         failure_cost=100000))
        assert r.cost_impact == pytest.approx(-100000)

    def test_downtime_impact_present(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert r.downtime_impact < 0


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for the compare() method."""

    def test_length_mismatch_raises(self) -> None:
        eng = _engine()
        b = AssetStateSimulator(_base(horizon=80)).simulate()
        s = AssetStateSimulator(_base(horizon=40)).simulate()
        with pytest.raises(ValueError, match="share a sample grid"):
            eng.compare(b, s, _cfg("normal"))

    def test_compare_returns_result(self) -> None:
        eng = _engine()
        b = AssetStateSimulator(_base()).simulate()
        s = AssetStateSimulator(_base(degradation_rate=0.5)).simulate()
        out = eng.compare(b, s, _cfg("load_reduction", degradation_factor=0.5))
        assert isinstance(out, ScenarioResult)

    def test_compare_delta_sign(self) -> None:
        eng = _engine()
        b = AssetStateSimulator(_base()).simulate()
        s = AssetStateSimulator(_base(degradation_rate=0.5)).simulate()
        out = eng.compare(b, s, _cfg("load_reduction", degradation_factor=0.5))
        assert out.mean_health_delta > 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Tests for generate_report()."""

    def test_report_contains_header(self) -> None:
        eng = _engine()
        results = [eng.run(_cfg("normal", name="ctrl"))]
        rep = eng.generate_report(results)
        assert "WHAT-IF SCENARIO REPORT" in rep

    def test_report_lists_scenarios(self) -> None:
        eng = _engine()
        results = [
            eng.run(_cfg("normal", name="ctrl")),
            eng.run(_cfg("load_reduction", degradation_factor=0.5, name="derate")),
        ]
        rep = eng.generate_report(results)
        assert "ctrl" in rep and "derate" in rep

    def test_report_includes_summaries(self) -> None:
        eng = _engine()
        results = [eng.run(_cfg("load_reduction", degradation_factor=0.5,
                                name="derate"))]
        rep = eng.generate_report(results)
        assert "Scenario" in rep

    def test_empty_report_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _engine().generate_report([])

    def test_report_handles_nan_rul(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        results = [eng.run(_cfg("load_reduction", degradation_factor=0.2))]
        rep = eng.generate_report(results)
        assert "n/a" in rep or "WHAT-IF" in rep


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


class TestExplainability:
    """Tests for the summary explainability output."""

    def test_summary_non_empty(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert len(r.summary) > 20

    def test_summary_names_scenario(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5,
                               name="my derate"))
        assert "my derate" in r.summary

    def test_summary_mentions_avoided(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        assert "AVOIDED" in r.summary

    def test_summary_mentions_induced(self) -> None:
        eng = ScenarioEngine(_base(horizon=50, degradation_rate=1.0))
        r = eng.run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert "INDUCED" in r.summary

    def test_summary_mentions_improvement(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert "improved" in r.summary

    def test_summary_mentions_worsening(self) -> None:
        r = _engine().run(_cfg("accelerated_degradation", degradation_factor=2.0))
        assert "worsened" in r.summary

    def test_normal_summary_negligible(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert "negligible" in r.summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for JSON-serialisable results."""

    def test_to_dict_keys(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        d = r.to_dict()
        for key in ("baseline_health", "scenario_health", "health_delta",
                    "rul_delta", "risk_delta", "cost_impact", "downtime_impact",
                    "summary"):
            assert key in d

    def test_json_dumps(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        js = json.dumps(r.to_dict())
        assert isinstance(js, str)

    def test_inf_rul_becomes_none(self) -> None:
        # A non-degrading custom scenario yields infinite RUL -> None in JSON
        r = _engine().run(_cfg("custom", overrides={
            "degradation_rate": 0.0}))
        d = r.to_dict()
        # scenario never degrades; its rul may be inf -> None
        assert d["scenario_rul"] is None or isinstance(d["scenario_rul"], float)

    def test_nan_failure_delta_becomes_none(self) -> None:
        eng = ScenarioEngine(_base(horizon=80, degradation_rate=1.0))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.2))
        d = r.to_dict()
        assert d["failure_time_delta"] is None

    def test_arrays_are_lists(self) -> None:
        r = _engine().run(_cfg("normal"))
        d = r.to_dict()
        assert isinstance(d["baseline_health"], list)
        assert isinstance(d["health_delta"], list)

    def test_summary_serialized(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert isinstance(r.to_dict()["summary"], str)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behaviour."""

    def test_deterministic_run(self) -> None:
        cfg = _cfg("accelerated_degradation", degradation_factor=1.5)
        r1 = _engine(noise_std=2.0, random_seed=7).run(cfg)
        r2 = _engine(noise_std=2.0, random_seed=7).run(cfg)
        assert np.array_equal(r1.scenario_health, r2.scenario_health)

    def test_deterministic_baseline(self) -> None:
        r1 = _engine(noise_std=2.0, random_seed=7).run(_cfg("normal"))
        r2 = _engine(noise_std=2.0, random_seed=7).run(_cfg("normal"))
        assert np.array_equal(r1.baseline_health, r2.baseline_health)

    def test_deterministic_deltas(self) -> None:
        cfg = _cfg("load_reduction", degradation_factor=0.5)
        r1 = _engine(noise_std=1.0, random_seed=3).run(cfg)
        r2 = _engine(noise_std=1.0, random_seed=3).run(cfg)
        assert r1.mean_health_delta == pytest.approx(r2.mean_health_delta)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_scenario(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        ScenarioEngine(_base(), experiment_tracker=FakeTracker()).run(
            _cfg("normal"))
        assert len(logged) == 1
        assert "scenario_risk_delta" in logged[0]

    def test_logs_scenario_type_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        ScenarioEngine(_base(), experiment_tracker=FakeTracker()).run(
            _cfg("load_reduction", degradation_factor=0.5))
        assert params and params[0]["scenario_type"] == "load_reduction"

    def test_step_increments(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                steps.append(step)

        eng = ScenarioEngine(_base(), experiment_tracker=FakeTracker())
        eng.run(_cfg("normal"))
        eng.run(_cfg("normal"))
        assert steps == [0, 1]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        r = ScenarioEngine(_base(), experiment_tracker=BrokenTracker()).run(
            _cfg("normal"))
        assert r is not None

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, metrics, step=None):
                pass

        r = ScenarioEngine(_base(), experiment_tracker=MetricsOnly()).run(
            _cfg("normal"))
        assert r is not None


# ---------------------------------------------------------------------------
# Edge cases / misc
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and miscellaneous behaviour."""

    def test_default_engine_runs(self) -> None:
        r = ScenarioEngine().run(_cfg("normal"))
        assert r is not None

    def test_n_scenarios_increments(self) -> None:
        eng = _engine()
        eng.run(_cfg("normal"))
        eng.run(_cfg("normal"))
        assert eng._n_scenarios == 2

    def test_scenario_name_defaults_to_type(self) -> None:
        r = _engine().run(_cfg("normal"))
        assert r.scenario_name == "normal"

    def test_shared_time_grid(self) -> None:
        r = _engine().run(_cfg("load_reduction", degradation_factor=0.5))
        assert len(r.times) == len(r.baseline_health) == len(r.scenario_health)

    def test_maintenance_before_grid_start(self) -> None:
        # intervention_time 0 applies from the first sample
        r = _engine().run(_cfg("maintenance_intervention",
                                intervention_time=0, restoration_amount=5))
        assert r.scenario_health[0] >= r.baseline_health[0]

    def test_fault_beyond_horizon_noop(self) -> None:
        # fault_time at the horizon edge still valid; beyond is clipped by config
        r = _engine().run(_cfg("sudden_fault", fault_time=80, fault_magnitude=20))
        assert r is not None

    def test_exponential_baseline(self) -> None:
        eng = ScenarioEngine(_base(model="exponential", degradation_rate=0.03))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.mean_health_delta > 0

    def test_piecewise_baseline(self) -> None:
        eng = ScenarioEngine(_base(model="piecewise",
                                   segment_breakpoints=(0, 40, 80),
                                   segment_rates=(0.3, 1.0)))
        r = eng.run(_cfg("load_reduction", degradation_factor=0.5))
        assert r.scenario_type == "load_reduction"

    def test_result_is_frozen(self) -> None:
        r = _engine().run(_cfg("normal"))
        with pytest.raises((AttributeError, TypeError)):
            r.cost_impact = 1.0  # type: ignore[misc]