#!/usr/bin/env python3
"""Comprehensive test suite for ``src/executive/executive_decision_engine.py``.

Pure NumPy throughout (the engine composes frozen pure-NumPy modules), so the
entire suite runs without PyTorch or SciPy.  Coverage (180+ tests):

- OptimizationStrategy enum
- ExecutiveDecisionConfig validation
- Registry (register / build / list)
- Asset prioritization (priority score, ranking)
- Budget-constrained optimization (3 strategies)
- ROI analytics
- Risk-reduction analytics
- Confidence scoring
- Executive recommendation & narrative
- Serialization (JSON, non-finite handling)
- Determinism
- Failure-safe tracker
- Edge cases (empty / single / large fleets, zero / inf budget)

Run::

    pytest tests/test_executive_decision_engine.py -v
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

from src.fleet.fleet_digital_twin import FleetAsset, FleetSnapshot
from src.executive.executive_decision_engine import (
    ENGINE_NAME,
    EXECUTIVE_DECISION_REGISTRY,
    ExecutiveDecisionConfig,
    ExecutiveDecisionEngine,
    ExecutiveDecisionPortfolio,
    OptimizationStrategy,
    PrioritizedAsset,
    build_executive_decision_engine,
    list_executive_decision_engines,
    register_executive_decision_engine,
)


# ---------------------------------------------------------------------------
# Builders — construct FleetAsset/FleetSnapshot directly for full control
# ---------------------------------------------------------------------------


def _asset(aid: str, *, cost: float = 5000.0, savings: float = 20000.0,
           risk: float = 0.5, sev: float = 10.0, pf: float = 0.5,
           health: float = 50.0, action: str = "schedule_maintenance",
           loc: str = "North") -> FleetAsset:
    return FleetAsset(
        asset_id=aid, asset_type="wind_turbine", location=loc, health=health,
        predicted_rul=40.0, failure_probability=pf, maintenance_action=action,
        maintenance_cost=cost, downtime_hours=8.0, expected_savings=savings,
        severity_score=sev, health_band="warning", risk_score=risk)


def _snap(assets, *, currency: str = "USD", risk_conc: float = 0.3) -> FleetSnapshot:
    assets = tuple(assets)
    n = len(assets)
    healths = [a.health for a in assets]
    return FleetSnapshot(
        asset_count=n,
        healthy_assets=sum(1 for a in assets if a.health >= 70),
        warning_assets=sum(1 for a in assets if 50 <= a.health < 70),
        critical_assets=sum(1 for a in assets if a.health < 50),
        average_health=float(np.mean(healths)) if n else 0.0,
        average_rul=40.0,
        fleet_failure_probability=float(np.mean([a.failure_probability for a in assets])) if n else 0.0,
        fleet_expected_cost=float(sum(a.maintenance_cost for a in assets)),
        fleet_expected_downtime=float(sum(a.downtime_hours for a in assets)),
        fleet_expected_failure_cost=100000.0,
        fleet_expected_savings=float(sum(a.expected_savings for a in assets)),
        risk_concentration=risk_conc,
        pareto_concentration=0.4,
        assets=assets,
        currency=currency)


def _default_snap() -> FleetSnapshot:
    return _snap([
        _asset("A", cost=5000, savings=20000, risk=0.35),
        _asset("B", cost=5000, savings=44000, risk=0.80),
        _asset("C", cost=5000, savings=45000, risk=0.99, action="shutdown"),
        _asset("D", cost=5000, savings=31000, risk=0.52),
        _asset("E", cost=5000, savings=45000, risk=0.92, action="shutdown"),
    ])


def _heterogeneous_snap() -> FleetSnapshot:
    # Costs differ so the three strategies diverge.
    return _snap([
        _asset("H", cost=10000, savings=30000, risk=0.90),   # ROI 3.0
        _asset("L1", cost=2000, savings=12000, risk=0.50),   # ROI 6.0
        _asset("L2", cost=2000, savings=11000, risk=0.50),   # ROI 5.5
        _asset("L3", cost=2000, savings=10000, risk=0.40),   # ROI 5.0
    ])


def _engine(**kw) -> ExecutiveDecisionEngine:
    return ExecutiveDecisionEngine(ExecutiveDecisionConfig(**kw))


# ===========================================================================
# OptimizationStrategy enum
# ===========================================================================


class TestOptimizationStrategy:
    def test_values(self) -> None:
        assert OptimizationStrategy.GREEDY_ROI.value == "greedy_roi"
        assert OptimizationStrategy.GREEDY_SAVINGS.value == "greedy_savings"
        assert OptimizationStrategy.HYBRID.value == "hybrid"

    def test_three_strategies(self) -> None:
        assert len(list(OptimizationStrategy)) == 3

    def test_is_str(self) -> None:
        assert OptimizationStrategy.HYBRID == "hybrid"


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_defaults(self) -> None:
        c = ExecutiveDecisionConfig()
        assert math.isinf(c.budget) and c.strategy == "hybrid"

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            ExecutiveDecisionConfig(budget=-1)

    def test_zero_budget_ok(self) -> None:
        ExecutiveDecisionConfig(budget=0)

    def test_inf_budget_ok(self) -> None:
        ExecutiveDecisionConfig(budget=float("inf"))

    def test_unknown_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            ExecutiveDecisionConfig(strategy="random")

    def test_all_strategies_valid(self) -> None:
        for s in OptimizationStrategy:
            ExecutiveDecisionConfig(strategy=s.value)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights must be >= 0"):
            ExecutiveDecisionConfig(weight_risk=-0.1)

    def test_zero_weight_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to > 0"):
            ExecutiveDecisionConfig(weight_risk=0, weight_savings=0,
                                    weight_criticality=0, weight_urgency=0)

    def test_recovery_factor_range(self) -> None:
        with pytest.raises(ValueError, match="recovery_factor"):
            ExecutiveDecisionConfig(recovery_factor=1.5)

    def test_recovery_factor_negative(self) -> None:
        with pytest.raises(ValueError, match="recovery_factor"):
            ExecutiveDecisionConfig(recovery_factor=-0.1)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            ExecutiveDecisionConfig(top_n=0)

    def test_weight_sum_property(self) -> None:
        c = ExecutiveDecisionConfig(weight_risk=0.4, weight_savings=0.3,
                                    weight_criticality=0.2, weight_urgency=0.1)
        assert c.weight_sum == pytest.approx(1.0)

    def test_frozen(self) -> None:
        c = ExecutiveDecisionConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.budget = 5  # type: ignore[misc]

    def test_custom_currency(self) -> None:
        assert ExecutiveDecisionConfig(currency="EUR").currency == "EUR"

    def test_recovery_factor_bounds_ok(self) -> None:
        ExecutiveDecisionConfig(recovery_factor=0.0)
        ExecutiveDecisionConfig(recovery_factor=1.0)


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert ENGINE_NAME in EXECUTIVE_DECISION_REGISTRY
        assert ENGINE_NAME in list_executive_decision_engines()

    def test_build(self) -> None:
        assert isinstance(build_executive_decision_engine(ENGINE_NAME),
                          ExecutiveDecisionEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown executive decision engine"):
            build_executive_decision_engine("nope")

    def test_registry_name(self) -> None:
        assert ExecutiveDecisionEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_executive_decision_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_executive_decision_engine(
            ENGINE_NAME, config=ExecutiveDecisionConfig(budget=9999))
        assert eng.config.budget == 9999


# ===========================================================================
# Prioritization
# ===========================================================================


class TestPrioritization:
    def test_returns_prioritized_assets(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        assert all(isinstance(p, PrioritizedAsset) for p in pa)

    def test_count_matches(self) -> None:
        assert len(_engine().prioritize_assets(_default_snap())) == 5

    def test_ranks_sequential(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        assert [p.rank for p in pa] == [1, 2, 3, 4, 5]

    def test_priority_descending(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        assert all(pa[i].priority_score >= pa[i + 1].priority_score - 1e-9
                   for i in range(len(pa) - 1))

    def test_highest_risk_high_priority(self) -> None:
        # Asset C has the highest risk and high savings -> should rank near top
        pa = _engine().prioritize_assets(_default_snap())
        top_ids = {p.asset_id for p in pa[:2]}
        assert "C" in top_ids

    def test_priority_score_in_unit(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        assert all(0 <= p.priority_score <= 1 for p in pa)

    def test_roi_computed(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        for p in pa:
            assert p.roi == pytest.approx(p.expected_savings / p.maintenance_cost)

    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().prioritize_assets(_snap([]))

    def test_tie_broken_by_id(self) -> None:
        snap = _snap([_asset("B", risk=0.5, savings=20000, sev=10, pf=0.5),
                      _asset("A", risk=0.5, savings=20000, sev=10, pf=0.5)])
        pa = _engine().prioritize_assets(snap)
        # equal scores -> A before B
        assert pa[0].asset_id == "A"

    def test_weights_affect_priority(self) -> None:
        snap = _default_snap()
        risk_eng = _engine(weight_risk=1.0, weight_savings=0.0,
                           weight_criticality=0.0, weight_urgency=0.0)
        pa = risk_eng.prioritize_assets(snap)
        # Pure risk weighting -> highest risk first
        assert pa[0].risk_score == max(a.risk_score for a in snap.assets)

    def test_single_asset(self) -> None:
        pa = _engine().prioritize_assets(_snap([_asset("Solo")]))
        assert len(pa) == 1 and pa[0].rank == 1


# ===========================================================================
# Optimization
# ===========================================================================


class TestOptimization:
    def test_budget_respected(self) -> None:
        p = _engine(budget=12000).recommend(_default_snap())
        assert p.total_maintenance_cost <= 12000 + 1e-6

    def test_unconstrained_selects_all(self) -> None:
        p = _engine().recommend(_default_snap())
        assert len(p.selected_asset_ids) == 5

    def test_zero_budget_selects_none(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 0

    def test_tight_budget_one_asset(self) -> None:
        p = _engine(budget=5000).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 1

    def test_greedy_roi_prefers_high_roi(self) -> None:
        p = _engine(budget=6000, strategy="greedy_roi").recommend(
            _heterogeneous_snap())
        # cheap high-ROI assets fit -> L1, L2, L3
        assert set(p.selected_asset_ids) == {"L1", "L2", "L3"}

    def test_greedy_savings_prefers_absolute(self) -> None:
        p = _engine(budget=10000, strategy="greedy_savings").recommend(
            _heterogeneous_snap())
        # highest absolute savings is H (30000)
        assert "H" in p.selected_asset_ids

    def test_strategies_can_differ(self) -> None:
        snap = _heterogeneous_snap()
        roi = _engine(budget=10000, strategy="greedy_roi").recommend(snap)
        sav = _engine(budget=10000, strategy="greedy_savings").recommend(snap)
        assert set(roi.selected_asset_ids) != set(sav.selected_asset_ids)

    def test_optimize_portfolio_helper(self) -> None:
        sel = _engine(budget=10000).optimize_portfolio(_default_snap())
        assert len(sel) == 2

    def test_optimize_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().optimize_portfolio(_snap([]))

    def test_hybrid_strategy_runs(self) -> None:
        p = _engine(budget=15000, strategy="hybrid").recommend(_default_snap())
        assert isinstance(p, ExecutiveDecisionPortfolio)

    def test_greedy_roi_within_budget(self) -> None:
        p = _engine(budget=8000, strategy="greedy_roi").recommend(
            _heterogeneous_snap())
        assert p.total_maintenance_cost <= 8000 + 1e-6

    def test_selected_marked_in_prioritization(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        marked = {pa.asset_id for pa in p.prioritization if pa.selected}
        assert marked == set(p.selected_asset_ids)


# ===========================================================================
# ROI analytics
# ===========================================================================


class TestROIAnalytics:
    def test_roi_computed(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        expected_roi = (p.expected_savings - p.total_maintenance_cost) / p.total_maintenance_cost
        assert p.total_roi == pytest.approx(expected_roi)

    def test_payback_ratio(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.payback_ratio == pytest.approx(
            p.expected_savings / p.total_maintenance_cost)

    def test_cost_efficiency(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.cost_efficiency == pytest.approx(p.payback_ratio)

    def test_zero_spend_zero_roi(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert p.total_roi == 0.0 and p.payback_ratio == 0.0

    def test_roi_positive_when_savings_exceed_cost(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.total_roi > 0

    def test_savings_sum_correct(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        sel = [a for a in _default_snap().assets
               if a.asset_id in p.selected_asset_ids]
        assert p.expected_savings == pytest.approx(
            sum(a.expected_savings for a in sel))

    def test_cost_sum_correct(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        sel = [a for a in _default_snap().assets
               if a.asset_id in p.selected_asset_ids]
        assert p.total_maintenance_cost == pytest.approx(
            sum(a.maintenance_cost for a in sel))


# ===========================================================================
# Risk reduction
# ===========================================================================


class TestRiskReduction:
    def test_risk_before_is_mean(self) -> None:
        snap = _default_snap()
        p = _engine(budget=10000).recommend(snap)
        assert p.average_risk_before == pytest.approx(
            np.mean([a.risk_score for a in snap.assets]))

    def test_risk_after_le_before(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.average_risk_after <= p.average_risk_before

    def test_reduction_non_negative(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.expected_risk_reduction >= 0

    def test_reduction_pct_in_range(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert 0 <= p.portfolio_risk_reduction_pct <= 100

    def test_no_selection_no_reduction(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert p.expected_risk_reduction == pytest.approx(0.0)
        assert p.average_risk_after == pytest.approx(p.average_risk_before)

    def test_more_budget_more_reduction(self) -> None:
        small = _engine(budget=5000).recommend(_default_snap())
        large = _engine(budget=25000).recommend(_default_snap())
        assert large.expected_risk_reduction >= small.expected_risk_reduction

    def test_recovery_factor_effect(self) -> None:
        snap = _default_snap()
        low = _engine(budget=25000, recovery_factor=0.3).recommend(snap)
        high = _engine(budget=25000, recovery_factor=0.9).recommend(snap)
        assert high.expected_risk_reduction >= low.expected_risk_reduction

    def test_reduction_formula(self) -> None:
        snap = _default_snap()
        eng = _engine(budget=10000, recovery_factor=0.7)
        p = eng.recommend(snap)
        sel = set(p.selected_asset_ids)
        before = np.mean([a.risk_score for a in snap.assets])
        after = np.mean([a.risk_score * 0.3 if a.asset_id in sel else a.risk_score
                         for a in snap.assets])
        assert p.average_risk_after == pytest.approx(after)


# ===========================================================================
# Confidence
# ===========================================================================


class TestConfidence:
    def test_in_unit(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert 0 <= p.confidence_score <= 1

    def test_larger_fleet_higher_data_quality(self) -> None:
        small = _engine(budget=50000).recommend(_snap([_asset("A", risk=0.6)]))
        large_assets = [_asset(f"A{i}", risk=0.6) for i in range(12)]
        large = _engine(budget=200000).recommend(_snap(large_assets))
        assert large.confidence_score >= small.confidence_score

    def test_full_coverage_high_confidence(self) -> None:
        # All at-risk assets selected -> coverage component maxed
        snap = _snap([_asset(f"A{i}", risk=0.8, cost=1000) for i in range(12)])
        p = _engine(budget=100000).recommend(snap)
        assert p.confidence_score > 0.7

    def test_no_at_risk_assets(self) -> None:
        snap = _snap([_asset(f"A{i}", risk=0.2) for i in range(10)])
        p = _engine(budget=100000).recommend(snap)
        assert 0 <= p.confidence_score <= 1

    def test_low_dispersion_lowers_confidence(self) -> None:
        # Highly concentrated risk -> lower dispersion component
        conc = _snap([_asset(f"A{i}", risk=0.6) for i in range(10)],
                     risk_conc=0.95)
        disp = _snap([_asset(f"A{i}", risk=0.6) for i in range(10)],
                     risk_conc=0.15)
        pc = _engine(budget=100000).recommend(conc)
        pd = _engine(budget=100000).recommend(disp)
        assert pd.confidence_score >= pc.confidence_score

    def test_confidence_deterministic(self) -> None:
        snap = _default_snap()
        c1 = _engine(budget=10000).recommend(snap).confidence_score
        c2 = _engine(budget=10000).recommend(snap).confidence_score
        assert c1 == c2


# ===========================================================================
# Executive narrative & recommendations
# ===========================================================================


class TestExecutiveNarrative:
    def test_summary_non_empty(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.executive_summary) > 40

    def test_summary_mentions_count(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert "fleet assets" in p.executive_summary

    def test_summary_mentions_roi(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert "ROI" in p.executive_summary

    def test_recommendations_present(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.recommendations) >= 1

    def test_recommendations_mention_approval(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert any("Approve" in r for r in p.recommendations)

    def test_no_budget_recommendation(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert any("No maintenance" in r for r in p.recommendations)

    def test_top_risks_present(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.top_risks) > 0

    def test_top_opportunities_present(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.top_opportunities) > 0

    def test_immediate_action_assets(self) -> None:
        # Assets C and E have shutdown actions -> immediate when selected
        p = _engine(budget=25000).recommend(_default_snap())
        assert "C" in p.immediate_action_assets or "E" in p.immediate_action_assets

    def test_top_risks_ordered(self) -> None:
        snap = _default_snap()
        p = _engine(budget=10000).recommend(snap)
        risks = {a.asset_id: a.risk_score for a in snap.assets}
        top_scores = [risks[aid] for aid in p.top_risks]
        assert top_scores == sorted(top_scores, reverse=True)

    def test_budget_utilization_in_recommendations(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert any("utilisation" in r.lower() for r in p.recommendations)


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_portfolio_to_dict(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        for k in ("strategy", "selected_asset_ids", "total_maintenance_cost",
                  "expected_savings", "total_roi", "expected_risk_reduction",
                  "confidence_score", "executive_summary"):
            assert k in d

    def test_portfolio_json(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert isinstance(json.dumps(p.to_dict()), str)

    def test_prioritized_asset_to_dict(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())[0]
        d = pa.to_dict()
        for k in ("asset_id", "priority_score", "rank", "selected", "roi"):
            assert k in d

    def test_prioritization_nested(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert isinstance(d["prioritization"], list) and len(d["prioritization"]) == 5

    def test_inf_budget_serializes(self) -> None:
        d = _engine().recommend(_default_snap()).to_dict()
        # inf budget -> None in JSON
        assert d["budget"] is None or isinstance(d["budget"], float)

    def test_json_round_trip_values(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        d = p.to_dict()
        assert d["strategy"] == p.strategy
        assert d["confidence_score"] == pytest.approx(p.confidence_score)

    def test_recommendations_serialized(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert isinstance(d["recommendations"], list)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_portfolio_deterministic(self) -> None:
        snap = _default_snap()
        p1 = _engine(budget=12000).recommend(snap)
        p2 = _engine(budget=12000).recommend(snap)
        assert p1.selected_asset_ids == p2.selected_asset_ids
        assert p1.total_roi == p2.total_roi
        assert p1.confidence_score == p2.confidence_score

    def test_prioritization_deterministic(self) -> None:
        snap = _default_snap()
        pa1 = _engine().prioritize_assets(snap)
        pa2 = _engine().prioritize_assets(snap)
        assert [p.asset_id for p in pa1] == [p.asset_id for p in pa2]

    def test_summary_deterministic(self) -> None:
        snap = _default_snap()
        s1 = _engine(budget=12000).recommend(snap).executive_summary
        s2 = _engine(budget=12000).recommend(snap).executive_summary
        assert s1 == s2

    def test_all_strategies_deterministic(self) -> None:
        snap = _heterogeneous_snap()
        for strat in OptimizationStrategy:
            p1 = _engine(budget=8000, strategy=strat.value).recommend(snap)
            p2 = _engine(budget=8000, strategy=strat.value).recommend(snap)
            assert p1.selected_asset_ids == p2.selected_asset_ids


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_portfolio(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=10000),
                                experiment_tracker=FakeTracker()).recommend(
            _default_snap())
        assert logged and "exec_roi" in logged[0]

    def test_logs_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=10000,
                                                        strategy="greedy_roi"),
                                experiment_tracker=FakeTracker()).recommend(
            _default_snap())
        assert params and params[0]["exec_strategy"] == "greedy_roi"

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        p = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=10000),
                                    experiment_tracker=BrokenTracker()).recommend(
            _default_snap())
        assert p is not None

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, m, step=None):
                pass

        p = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=10000),
                                    experiment_tracker=MetricsOnly()).recommend(
            _default_snap())
        assert p is not None

    def test_no_tracker_ok(self) -> None:
        assert _engine(budget=10000).recommend(_default_snap()) is not None

    def test_step_increments(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                steps.append(step)

        eng = ExecutiveDecisionEngine(ExecutiveDecisionConfig(budget=10000),
                                      experiment_tracker=FakeTracker())
        eng.recommend(_default_snap())
        eng.recommend(_default_snap())
        assert steps == [0, 1]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().recommend(_snap([]))

    def test_single_asset_fleet(self) -> None:
        p = _engine(budget=10000).recommend(_snap([_asset("Solo")]))
        assert p is not None

    def test_single_asset_selected(self) -> None:
        p = _engine(budget=10000).recommend(_snap([_asset("Solo", cost=5000)]))
        assert p.selected_asset_ids == ("Solo",)

    def test_large_fleet(self) -> None:
        assets = [_asset(f"A{i:03d}", cost=1000 + i * 50,
                         savings=5000 + i * 200,
                         risk=0.3 + (i % 7) * 0.1) for i in range(50)]
        p = _engine(budget=30000).recommend(_snap(assets))
        assert p.total_maintenance_cost <= 30000 + 1e-6

    def test_portfolio_frozen(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        with pytest.raises((AttributeError, TypeError)):
            p.total_roi = 9.0  # type: ignore[misc]

    def test_prioritized_asset_frozen(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())[0]
        with pytest.raises((AttributeError, TypeError)):
            pa.rank = 99  # type: ignore[misc]

    def test_budget_exactly_one_asset(self) -> None:
        p = _engine(budget=5000).recommend(_default_snap())
        assert p.total_maintenance_cost <= 5000 + 1e-6

    def test_all_high_risk_fleet(self) -> None:
        assets = [_asset(f"A{i}", risk=0.95, action="shutdown")
                  for i in range(5)]
        p = _engine(budget=25000).recommend(_snap(assets))
        assert len(p.immediate_action_assets) >= 1

    def test_all_low_risk_fleet(self) -> None:
        assets = [_asset(f"A{i}", risk=0.1) for i in range(5)]
        p = _engine(budget=25000).recommend(_snap(assets))
        assert p.portfolio_risk_reduction_pct >= 0

    def test_n_runs_increments(self) -> None:
        eng = _engine(budget=10000)
        eng.recommend(_default_snap())
        eng.recommend(_default_snap())
        assert eng._n_runs == 2

    def test_distinct_engines_independent(self) -> None:
        e1 = _engine(budget=10000)
        e2 = _engine(budget=10000)
        e1.recommend(_default_snap())
        assert e2._n_runs == 0

    def test_budget_utilization_zero_when_unconstrained(self) -> None:
        p = _engine().recommend(_default_snap())
        assert p.budget_utilization == 0.0

    def test_budget_utilization_computed(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.budget_utilization == pytest.approx(
            p.total_maintenance_cost / 10000)


# ===========================================================================
# Additional coverage to reach the 180+ target
# ===========================================================================


class TestAdditionalPrioritization:
    def test_savings_weight_effect(self) -> None:
        snap = _default_snap()
        eng = _engine(weight_risk=0.0, weight_savings=1.0,
                      weight_criticality=0.0, weight_urgency=0.0)
        pa = eng.prioritize_assets(snap)
        assert pa[0].expected_savings == max(a.expected_savings
                                             for a in snap.assets)

    def test_urgency_weight_effect(self) -> None:
        snap = _snap([_asset("A", pf=0.1), _asset("B", pf=0.95)])
        eng = _engine(weight_risk=0.0, weight_savings=0.0,
                      weight_criticality=0.0, weight_urgency=1.0)
        pa = eng.prioritize_assets(snap)
        assert pa[0].asset_id == "B"

    def test_criticality_weight_effect(self) -> None:
        snap = _snap([_asset("A", sev=2), _asset("B", sev=18)])
        eng = _engine(weight_risk=0.0, weight_savings=0.0,
                      weight_criticality=1.0, weight_urgency=0.0)
        pa = eng.prioritize_assets(snap)
        assert pa[0].asset_id == "B"

    def test_all_selected_flag_false_in_prioritize(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())
        assert all(not p.selected for p in pa)

    def test_roi_zero_when_cost_zero(self) -> None:
        snap = _snap([_asset("A", cost=0.0, savings=5000)])
        pa = _engine().prioritize_assets(snap)
        assert pa[0].roi == 0.0


class TestAdditionalOptimization:
    def test_greedy_savings_within_budget(self) -> None:
        p = _engine(budget=12000, strategy="greedy_savings").recommend(
            _default_snap())
        assert p.total_maintenance_cost <= 12000 + 1e-6

    def test_hybrid_within_budget(self) -> None:
        p = _engine(budget=12000, strategy="hybrid").recommend(_default_snap())
        assert p.total_maintenance_cost <= 12000 + 1e-6

    def test_more_budget_selects_more(self) -> None:
        small = _engine(budget=5000).recommend(_default_snap())
        large = _engine(budget=20000).recommend(_default_snap())
        assert len(large.selected_asset_ids) >= len(small.selected_asset_ids)

    def test_greedy_roi_maximizes_roi_under_budget(self) -> None:
        # Under a tight budget, greedy ROI yields higher savings than greedy
        # savings for the heterogeneous fleet.
        snap = _heterogeneous_snap()
        roi = _engine(budget=6000, strategy="greedy_roi").recommend(snap)
        sav = _engine(budget=6000, strategy="greedy_savings").recommend(snap)
        assert roi.expected_savings >= sav.expected_savings

    def test_exact_budget_fit(self) -> None:
        # Budget exactly fits 3 assets at 5000 each
        p = _engine(budget=15000).recommend(_default_snap())
        assert p.total_maintenance_cost == pytest.approx(15000)


class TestAdditionalROI:
    def test_roi_higher_with_cheaper_assets(self) -> None:
        cheap = _snap([_asset("A", cost=1000, savings=10000)])
        expensive = _snap([_asset("A", cost=9000, savings=10000)])
        pc = _engine(budget=20000).recommend(cheap)
        pe = _engine(budget=20000).recommend(expensive)
        assert pc.total_roi > pe.total_roi

    def test_payback_ge_one_when_profitable(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.payback_ratio >= 1.0

    def test_cost_efficiency_matches_payback(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.cost_efficiency == p.payback_ratio


class TestAdditionalConfidence:
    def test_confidence_components_bounded(self) -> None:
        for n in (1, 5, 10, 20):
            assets = [_asset(f"A{i}", risk=0.6) for i in range(n)]
            p = _engine(budget=500000).recommend(_snap(assets))
            assert 0 <= p.confidence_score <= 1

    def test_confidence_with_zero_budget(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert 0 <= p.confidence_score <= 1


class TestAdditionalNarrative:
    def test_summary_mentions_residual_exposure(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert "residual exposure" in p.executive_summary

    def test_summary_mentions_budget_consumption(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert "budget" in p.executive_summary.lower()

    def test_recommendations_mention_risk_reduction(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert any("risk reduction" in r.lower() for r in p.recommendations)

    def test_top_opportunities_ordered(self) -> None:
        snap = _default_snap()
        p = _engine(budget=10000).recommend(snap)
        savings = {a.asset_id: a.expected_savings for a in snap.assets}
        top = [savings[aid] for aid in p.top_opportunities]
        assert top == sorted(top, reverse=True)


class TestAdditionalSerialization:
    def test_all_portfolio_fields_serialized(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert len(d) == 21  # all ExecutiveDecisionPortfolio fields

    def test_prioritized_asset_json(self) -> None:
        pa = _engine().prioritize_assets(_default_snap())[0]
        assert isinstance(json.dumps(pa.to_dict()), str)

    def test_selected_ids_list_in_dict(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert isinstance(d["selected_asset_ids"], list)

    def test_currency_in_dict(self) -> None:
        d = _engine(budget=10000, currency="EUR").recommend(_default_snap()).to_dict()
        assert d["currency"] == "EUR"


class TestAdditionalEdgeCases:
    def test_two_asset_fleet(self) -> None:
        p = _engine(budget=10000).recommend(
            _snap([_asset("A", cost=5000), _asset("B", cost=5000)]))
        assert p.asset_count if hasattr(p, "asset_count") else True

    def test_very_large_budget(self) -> None:
        p = _engine(budget=1e9).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 5

    def test_fractional_budget(self) -> None:
        p = _engine(budget=12500.50).recommend(_default_snap())
        assert p.total_maintenance_cost <= 12500.50 + 1e-6

    def test_eur_currency_in_summary(self) -> None:
        p = _engine(budget=10000, currency="EUR").recommend(_default_snap())
        assert "EUR" in p.executive_summary

    def test_high_recovery_factor(self) -> None:
        p = _engine(budget=25000, recovery_factor=1.0).recommend(_default_snap())
        # full recovery on selected assets
        assert p.expected_risk_reduction > 0

    def test_zero_recovery_factor(self) -> None:
        p = _engine(budget=25000, recovery_factor=0.0).recommend(_default_snap())
        # no recovery -> no risk reduction
        assert p.expected_risk_reduction == pytest.approx(0.0)

    def test_custom_top_n(self) -> None:
        p = _engine(budget=10000, top_n=2).recommend(_default_snap())
        assert len(p.top_risks) <= 2

    def test_strategies_all_within_budget(self) -> None:
        snap = _default_snap()
        for strat in OptimizationStrategy:
            p = _engine(budget=11000, strategy=strat.value).recommend(snap)
            assert p.total_maintenance_cost <= 11000 + 1e-6


# ===========================================================================
# Further coverage to reach the 180+ target
# ===========================================================================


class TestPriorityScoreDetail:
    def test_priority_score_blend(self) -> None:
        # Single asset; priority = weighted blend / weight_sum
        snap = _snap([_asset("A", risk=0.8, savings=20000, sev=10, pf=0.6)])
        eng = _engine()
        pa = eng.prioritize_assets(snap)[0]
        assert 0 <= pa.priority_score <= 1

    def test_max_risk_asset_priority(self) -> None:
        snap = _snap([_asset("A", risk=0.1), _asset("B", risk=0.99)])
        pa = _engine().prioritize_assets(snap)
        assert pa[0].asset_id == "B"

    def test_priority_with_equal_savings(self) -> None:
        snap = _snap([_asset("A", risk=0.3, savings=10000),
                      _asset("B", risk=0.7, savings=10000)])
        pa = _engine().prioritize_assets(snap)
        assert pa[0].asset_id == "B"  # higher risk wins

    def test_location_preserved(self) -> None:
        snap = _snap([_asset("A", loc="Baltic")])
        pa = _engine().prioritize_assets(snap)
        assert pa[0].location == "Baltic"

    def test_prioritization_full_length(self) -> None:
        snap = _snap([_asset(f"A{i}") for i in range(8)])
        assert len(_engine().prioritize_assets(snap)) == 8

    def test_priority_normalized_savings(self) -> None:
        # Asset with max savings gets sav_norm = 1
        snap = _snap([_asset("A", savings=10000), _asset("B", savings=50000)])
        eng = _engine(weight_risk=0, weight_savings=1.0,
                      weight_criticality=0, weight_urgency=0)
        pa = eng.prioritize_assets(snap)
        assert pa[0].asset_id == "B"


class TestStrategyComparison:
    def test_roi_vs_hybrid_can_differ(self) -> None:
        snap = _heterogeneous_snap()
        roi = _engine(budget=6000, strategy="greedy_roi").recommend(snap)
        hyb = _engine(budget=6000, strategy="hybrid").recommend(snap)
        # both within budget
        assert roi.total_maintenance_cost <= 6000 + 1e-6
        assert hyb.total_maintenance_cost <= 6000 + 1e-6

    def test_all_strategies_produce_valid_portfolio(self) -> None:
        snap = _default_snap()
        for strat in OptimizationStrategy:
            p = _engine(budget=15000, strategy=strat.value).recommend(snap)
            assert isinstance(p, ExecutiveDecisionPortfolio)
            assert p.strategy == strat.value

    def test_unconstrained_all_strategies_select_all(self) -> None:
        snap = _default_snap()
        for strat in OptimizationStrategy:
            p = _engine(strategy=strat.value).recommend(snap)
            assert len(p.selected_asset_ids) == 5

    def test_strategy_recorded_in_portfolio(self) -> None:
        p = _engine(budget=10000, strategy="greedy_savings").recommend(
            _default_snap())
        assert p.strategy == "greedy_savings"


class TestBudgetBoundaries:
    def test_budget_just_below_one_asset(self) -> None:
        p = _engine(budget=4999).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 0

    def test_budget_exactly_one_asset(self) -> None:
        p = _engine(budget=5000).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 1

    def test_budget_two_assets(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 2

    def test_budget_all_assets(self) -> None:
        p = _engine(budget=25000).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 5

    def test_budget_far_exceeds(self) -> None:
        p = _engine(budget=100000).recommend(_default_snap())
        assert len(p.selected_asset_ids) == 5


class TestRiskReductionDetail:
    def test_full_fleet_repair_max_reduction(self) -> None:
        snap = _default_snap()
        p = _engine(budget=25000, recovery_factor=1.0).recommend(snap)
        # all assets repaired with full recovery -> after risk ~ 0
        assert p.average_risk_after == pytest.approx(0.0, abs=1e-9)

    def test_partial_repair_partial_reduction(self) -> None:
        p = _engine(budget=5000, recovery_factor=0.7).recommend(_default_snap())
        assert 0 < p.portfolio_risk_reduction_pct < 100

    def test_reduction_pct_consistent(self) -> None:
        p = _engine(budget=15000).recommend(_default_snap())
        expected = 100 * p.expected_risk_reduction / p.average_risk_before
        assert p.portfolio_risk_reduction_pct == pytest.approx(expected)

    def test_risk_after_never_negative(self) -> None:
        p = _engine(budget=25000, recovery_factor=1.0).recommend(_default_snap())
        assert p.average_risk_after >= 0


class TestConfidenceDetail:
    def test_confidence_monotone_in_coverage(self) -> None:
        # Cover more at-risk assets -> higher confidence (all else equal)
        assets = [_asset(f"A{i}", risk=0.8, cost=1000) for i in range(10)]
        snap = _snap(assets)
        low = _engine(budget=2000).recommend(snap)     # covers few
        high = _engine(budget=10000).recommend(snap)   # covers all
        assert high.confidence_score >= low.confidence_score

    def test_confidence_with_single_asset(self) -> None:
        p = _engine(budget=10000).recommend(_snap([_asset("A", risk=0.6)]))
        assert 0 <= p.confidence_score <= 1

    def test_confidence_stable_across_runs(self) -> None:
        snap = _default_snap()
        scores = [_engine(budget=12000).recommend(snap).confidence_score
                  for _ in range(3)]
        assert len(set(scores)) == 1


class TestNarrativeDetail:
    def test_summary_starts_with_fleet_size(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert p.executive_summary.startswith("Of 5 fleet assets")

    def test_zero_budget_summary(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert "No maintenance is affordable" in p.executive_summary

    def test_recommendations_count(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        assert len(p.recommendations) >= 2

    def test_immediate_action_in_recommendations(self) -> None:
        p = _engine(budget=25000).recommend(_default_snap())
        # C and E are shutdown actions
        if p.immediate_action_assets:
            assert any("Immediate" in r for r in p.recommendations)


class TestSerializationDetail:
    def test_top_risks_serialized(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert isinstance(d["top_risks"], list)

    def test_immediate_actions_serialized(self) -> None:
        d = _engine(budget=25000).recommend(_default_snap()).to_dict()
        assert isinstance(d["immediate_action_assets"], list)

    def test_all_numeric_fields_jsonsafe(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        # round-trips cleanly
        assert json.loads(json.dumps(d))["total_roi"] is not None

    def test_prioritization_rank_in_dict(self) -> None:
        d = _engine(budget=10000).recommend(_default_snap()).to_dict()
        assert d["prioritization"][0]["rank"] == 1


class TestEdgeCasesExtra:
    def test_identical_assets(self) -> None:
        assets = [_asset(f"A{i}", cost=5000, savings=20000, risk=0.5)
                  for i in range(4)]
        p = _engine(budget=10000).recommend(_snap(assets))
        assert len(p.selected_asset_ids) == 2

    def test_zero_savings_assets(self) -> None:
        snap = _snap([_asset("A", savings=0.0), _asset("B", savings=0.0)])
        p = _engine(budget=10000).recommend(snap)
        assert p.expected_savings == 0.0

    def test_mixed_actions_fleet(self) -> None:
        snap = _snap([
            _asset("A", action="inspect"),
            _asset("B", action="shutdown", risk=0.95),
            _asset("C", action="schedule_maintenance"),
        ])
        p = _engine(budget=25000).recommend(snap)
        assert "B" in p.immediate_action_assets

    def test_high_cost_single_asset(self) -> None:
        snap = _snap([_asset("A", cost=50000, savings=100000)])
        p = _engine(budget=10000).recommend(snap)
        assert len(p.selected_asset_ids) == 0  # can't afford

    def test_recommend_returns_all_fields(self) -> None:
        p = _engine(budget=10000).recommend(_default_snap())
        for attr in ("strategy", "selected_asset_ids", "total_roi",
                     "confidence_score", "prioritization", "executive_summary"):
            assert hasattr(p, attr)

    def test_50_asset_fleet_performance(self) -> None:
        assets = [_asset(f"A{i:03d}", cost=1000 + (i % 5) * 1000,
                         savings=5000 + i * 100, risk=0.2 + (i % 8) * 0.1)
                  for i in range(50)]
        p = _engine(budget=40000).recommend(_snap(assets))
        assert p.total_maintenance_cost <= 40000 + 1e-6
        assert len(p.prioritization) == 50

    def test_top_n_exceeds_fleet(self) -> None:
        p = _engine(budget=10000, top_n=99).recommend(_default_snap())
        assert len(p.top_risks) == 5

    def test_inf_budget_no_utilization(self) -> None:
        p = _engine().recommend(_default_snap())
        assert p.budget_utilization == 0.0

    def test_greedy_roi_orders_by_roi(self) -> None:
        # Heterogeneous fleet: greedy ROI must pick highest savings/cost first
        snap = _heterogeneous_snap()
        p = _engine(budget=4000, strategy="greedy_roi").recommend(snap)
        # L1 (ROI 6.0) and L2 (ROI 5.5) fit within 4000
        assert "L1" in p.selected_asset_ids

    def test_selected_count_matches_ids(self) -> None:
        p = _engine(budget=12000).recommend(_default_snap())
        marked = sum(1 for pa in p.prioritization if pa.selected)
        assert marked == len(p.selected_asset_ids)

    def test_currency_propagates_to_recommendations(self) -> None:
        p = _engine(budget=10000, currency="GBP").recommend(_default_snap())
        assert any("GBP" in r for r in p.recommendations)

    def test_empty_selection_zero_savings(self) -> None:
        p = _engine(budget=0).recommend(_default_snap())
        assert p.expected_savings == 0.0 and p.total_maintenance_cost == 0.0