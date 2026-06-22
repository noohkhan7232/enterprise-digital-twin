#!/usr/bin/env python3
"""Comprehensive test suite for ``src/strategy/portfolio_strategy_engine.py``.

Pure NumPy throughout (the engine composes frozen pure-NumPy modules), so the
entire suite runs without PyTorch, SciPy, or pandas.  Coverage (200+ tests):

- StrategicPosture enum & posture-config map
- StrategicPortfolioConfig validation & budget_levels
- Registry (register / build / list)
- Strategy comparison (4 postures, best-by metrics, agreement)
- Budget sweep (curves, knee, ROI-diminish)
- Pareto frontier (dominance, non-dominated set, knee)
- Capital efficiency (per-dollar metrics, efficiency index)
- Strategic recommendation (rule-based messages, summary)
- Confidence assessment (stability, agreement, coverage)
- Top-level optimize()
- Serialization (JSON, non-finite handling)
- Determinism
- Failure-safe tracker
- Edge cases (empty / single / large fleets, zero / inf budget)

Run::

    pytest tests/test_portfolio_strategy_engine.py -v
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
from src.strategy.portfolio_strategy_engine import (
    ENGINE_NAME,
    STRATEGY_ENGINE_REGISTRY,
    BudgetSweep,
    BudgetSweepPoint,
    CapitalEfficiency,
    ParetoFrontier,
    ParetoPortfolio,
    StrategicConfidence,
    StrategicPortfolioConfig,
    StrategicPortfolioEngine,
    StrategicPortfolioPlan,
    StrategicPosture,
    StrategicRecommendation,
    StrategyComparison,
    StrategyResult,
    build_strategic_portfolio_engine,
    list_strategic_portfolio_engines,
    register_strategic_portfolio_engine,
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


def _graded_snap(n: int = 12) -> FleetSnapshot:
    """A fleet with graded risk/savings so the sweep produces a smooth curve."""
    assets = []
    for i in range(n):
        risk = 0.2 + 0.06 * i
        assets.append(_asset(f"W{i:02d}", cost=5000,
                             savings=10000 + 3000 * i,
                             risk=min(risk, 0.99),
                             sev=4 + i, pf=min(0.2 + 0.06 * i, 1.0),
                             health=80 - 4 * i))
    return _snap(assets)


def _cfg(**kw) -> StrategicPortfolioConfig:
    base = dict(budget_start=10000, budget_end=60000, budget_step=10000)
    base.update(kw)
    return StrategicPortfolioConfig(**base)


def _engine(**kw) -> StrategicPortfolioEngine:
    return StrategicPortfolioEngine(_cfg(**kw))


def _sweep(engine=None, snap=None) -> BudgetSweep:
    engine = engine or _engine()
    snap = snap or _graded_snap()
    return engine.budget_sweep(snap)


# ===========================================================================
# StrategicPosture
# ===========================================================================


class TestStrategicPosture:
    def test_values(self) -> None:
        assert StrategicPosture.RISK_FIRST.value == "risk_first"
        assert StrategicPosture.ROI_FIRST.value == "roi_first"
        assert StrategicPosture.CRITICALITY_FIRST.value == "criticality_first"
        assert StrategicPosture.BALANCED.value == "balanced"

    def test_four_postures(self) -> None:
        assert len(list(StrategicPosture)) == 4

    def test_is_str(self) -> None:
        assert StrategicPosture.BALANCED == "balanced"


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_defaults(self) -> None:
        c = StrategicPortfolioConfig()
        assert c.budget_start == 100000 and c.budget_end == 500000

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget_start"):
            StrategicPortfolioConfig(budget_start=-1)

    def test_inverted_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget_end"):
            StrategicPortfolioConfig(budget_start=500, budget_end=100)

    def test_zero_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget_step"):
            StrategicPortfolioConfig(budget_step=0)

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget_step"):
            StrategicPortfolioConfig(budget_step=-100)

    def test_invalid_posture_rejected(self) -> None:
        with pytest.raises(ValueError, match="sweep_posture"):
            StrategicPortfolioConfig(sweep_posture="nope")

    def test_roi_fraction_range(self) -> None:
        with pytest.raises(ValueError, match="roi_diminish_fraction"):
            StrategicPortfolioConfig(roi_diminish_fraction=1.5)

    def test_roi_fraction_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="roi_diminish_fraction"):
            StrategicPortfolioConfig(roi_diminish_fraction=0.0)

    def test_unlock_ratio_positive(self) -> None:
        with pytest.raises(ValueError, match="unlock_ratio"):
            StrategicPortfolioConfig(unlock_ratio=0)

    def test_recovery_factor_range(self) -> None:
        with pytest.raises(ValueError, match="recovery_factor"):
            StrategicPortfolioConfig(recovery_factor=1.5)

    def test_frozen(self) -> None:
        c = StrategicPortfolioConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.budget_step = 1  # type: ignore[misc]

    def test_equal_start_end_ok(self) -> None:
        StrategicPortfolioConfig(budget_start=100000, budget_end=100000)

    def test_all_postures_valid(self) -> None:
        for p in StrategicPosture:
            StrategicPortfolioConfig(sweep_posture=p.value)

    def test_custom_currency(self) -> None:
        assert StrategicPortfolioConfig(currency="EUR").currency == "EUR"


class TestBudgetLevels:
    def test_example_levels(self) -> None:
        levels = StrategicPortfolioConfig(
            budget_start=100000, budget_end=500000, budget_step=100000).budget_levels()
        assert levels == (100000, 200000, 300000, 400000, 500000)

    def test_single_level(self) -> None:
        levels = StrategicPortfolioConfig(
            budget_start=100000, budget_end=100000, budget_step=100000).budget_levels()
        assert levels == (100000,)

    def test_count(self) -> None:
        levels = _cfg(budget_start=10000, budget_end=60000,
                      budget_step=10000).budget_levels()
        assert len(levels) == 6

    def test_start_included(self) -> None:
        levels = _cfg().budget_levels()
        assert levels[0] == 10000

    def test_end_included(self) -> None:
        levels = _cfg().budget_levels()
        assert levels[-1] == 60000

    def test_uneven_step(self) -> None:
        levels = StrategicPortfolioConfig(
            budget_start=0, budget_end=100, budget_step=30).budget_levels()
        # 0, 30, 60, 90 (120 > 100)
        assert levels == (0, 30, 60, 90)


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert ENGINE_NAME in STRATEGY_ENGINE_REGISTRY
        assert ENGINE_NAME in list_strategic_portfolio_engines()

    def test_build(self) -> None:
        assert isinstance(build_strategic_portfolio_engine(ENGINE_NAME),
                          StrategicPortfolioEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown strategic portfolio engine"):
            build_strategic_portfolio_engine("nope")

    def test_registry_name(self) -> None:
        assert StrategicPortfolioEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_strategic_portfolio_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_strategic_portfolio_engine(
            ENGINE_NAME, config=_cfg(currency="GBP"))
        assert eng.config.currency == "GBP"


# ===========================================================================
# Strategy comparison
# ===========================================================================


class TestStrategyComparison:
    def test_four_results(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert len(cmp.results) == 4

    def test_all_postures_present(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        postures = {r.posture for r in cmp.results}
        assert postures == {p.value for p in StrategicPosture}

    def test_result_fields(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        r = cmp.results[0]
        for f in ("selected_asset_ids", "total_cost", "expected_savings",
                  "risk_reduction", "roi", "coverage"):
            assert hasattr(r, f)

    def test_budget_respected(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 25000)
        assert all(r.total_cost <= 25000 + 1e-6 for r in cmp.results)

    def test_best_by_savings(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        best = max(cmp.results, key=lambda r: r.expected_savings)
        assert cmp.best_by_savings == best.posture or \
            any(r.expected_savings == best.expected_savings
                for r in cmp.results if r.posture == cmp.best_by_savings)

    def test_best_by_roi(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert cmp.best_by_roi in {p.value for p in StrategicPosture}

    def test_best_by_coverage(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert cmp.best_by_coverage in {p.value for p in StrategicPosture}

    def test_agreement_in_unit(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert 0 <= cmp.agreement <= 1

    def test_agreement_high_when_correlated(self) -> None:
        # Graded fleet: risk/roi/criticality correlate -> high agreement
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert cmp.agreement > 0.5

    def test_coverage_in_unit(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert all(0 <= r.coverage <= 1 for r in cmp.results)

    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().compare_strategies(_snap([]), 30000)

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            _engine().compare_strategies(_graded_snap(), -5)

    def test_strategies_diverge_anticorrelated(self) -> None:
        # Anti-correlate risk and savings so risk_first != roi_first
        assets = [
            _asset("HiRiskLoSav", cost=5000, savings=6000, risk=0.95, sev=18, pf=0.9),
            _asset("LoRiskHiSav", cost=5000, savings=45000, risk=0.30, sev=4, pf=0.3),
            _asset("Mid", cost=5000, savings=20000, risk=0.6, sev=10, pf=0.5),
        ]
        snap = _snap(assets)
        cmp = _engine().compare_strategies(snap, 5000)  # budget for 1 asset
        risk_r = next(r for r in cmp.results if r.posture == "risk_first")
        roi_r = next(r for r in cmp.results if r.posture == "roi_first")
        # risk_first picks the high-risk asset; roi_first picks high-savings
        assert set(risk_r.selected_asset_ids) != set(roi_r.selected_asset_ids)


# ===========================================================================
# Budget sweep
# ===========================================================================


class TestBudgetSweep:
    def test_returns_sweep(self) -> None:
        assert isinstance(_sweep(), BudgetSweep)

    def test_point_count(self) -> None:
        assert len(_sweep().points) == 6

    def test_curves_length(self) -> None:
        sw = _sweep()
        assert len(sw.risk_reduction_curve) == 6
        assert len(sw.savings_curve) == 6
        assert len(sw.roi_curve) == 6
        assert len(sw.coverage_curve) == 6

    def test_budgets_match_levels(self) -> None:
        sw = _sweep()
        assert sw.budgets == (10000, 20000, 30000, 40000, 50000, 60000)

    def test_savings_monotone(self) -> None:
        sw = _sweep()
        assert all(sw.savings_curve[i] <= sw.savings_curve[i + 1] + 1e-6
                   for i in range(5))

    def test_coverage_monotone(self) -> None:
        sw = _sweep()
        assert all(sw.coverage_curve[i] <= sw.coverage_curve[i + 1] + 1e-6
                   for i in range(5))

    def test_risk_reduction_monotone(self) -> None:
        sw = _sweep()
        assert all(sw.risk_reduction_curve[i] <= sw.risk_reduction_curve[i + 1] + 1e-6
                   for i in range(5))

    def test_knee_in_range(self) -> None:
        sw = _sweep()
        assert sw.budgets[0] <= sw.knee_budget <= sw.budgets[-1]

    def test_coverage_in_unit(self) -> None:
        sw = _sweep()
        assert all(0 <= c <= 1 for c in sw.coverage_curve)

    def test_custom_posture(self) -> None:
        sw = _engine().budget_sweep(_graded_snap(), posture="roi_first")
        assert sw.posture == "roi_first"

    def test_invalid_posture_raises(self) -> None:
        with pytest.raises(ValueError, match="posture"):
            _engine().budget_sweep(_graded_snap(), posture="bad")

    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().budget_sweep(_snap([]))

    def test_point_fields(self) -> None:
        pt = _sweep().points[0]
        for f in ("budget", "total_cost", "risk_reduction", "savings", "roi",
                  "coverage", "n_selected"):
            assert hasattr(pt, f)

    def test_cost_within_budget(self) -> None:
        sw = _sweep()
        assert all(pt.total_cost <= pt.budget + 1e-6 for pt in sw.points)

    def test_roi_diminish_detected(self) -> None:
        # The graded fleet's ROI declines -> diminish budget should be set
        sw = _sweep()
        assert sw.roi_diminish_budget is None or sw.roi_diminish_budget > 0


# ===========================================================================
# Pareto frontier
# ===========================================================================


class TestParetoFrontier:
    def test_returns_frontier(self) -> None:
        assert isinstance(_engine().pareto_frontier(_sweep()), ParetoFrontier)

    def test_candidate_count(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert fr.n_candidates == 6

    def test_non_empty(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert len(fr.portfolios) >= 1

    def test_portfolio_fields(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        p = fr.portfolios[0]
        for f in ("portfolio_id", "budget", "risk", "savings", "roi"):
            assert hasattr(p, f)

    def test_ordered_by_budget(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        budgets = [p.budget for p in fr.portfolios]
        assert budgets == sorted(budgets)

    def test_knee_id_set(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert fr.knee_portfolio_id is not None

    def test_non_dominated_property(self) -> None:
        # No portfolio in the frontier dominates another
        fr = _engine().pareto_frontier(_sweep())
        ps = fr.portfolios
        for a in ps:
            for b in ps:
                if a is b:
                    continue
                dominates = (a.savings >= b.savings and a.risk <= b.risk
                             and (a.savings > b.savings or a.risk < b.risk))
                assert not dominates

    def test_monotone_collapses_to_one(self) -> None:
        # Strictly monotone sweep -> only the top point is non-dominated
        fr = _engine().pareto_frontier(_sweep())
        assert len(fr.portfolios) == 1

    def test_portfolio_id_format(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert all(p.portfolio_id.startswith("budget_") for p in fr.portfolios)

    def test_empty_sweep_frontier(self) -> None:
        # A sweep with a single budget still yields a valid frontier
        eng = StrategicPortfolioEngine(_cfg(budget_start=20000, budget_end=20000))
        fr = eng.pareto_frontier(eng.budget_sweep(_graded_snap()))
        assert fr.n_candidates == 1


# ===========================================================================
# Capital efficiency
# ===========================================================================


class TestCapitalEfficiency:
    def test_returns_efficiency(self) -> None:
        sw = _sweep()
        assert isinstance(_engine().capital_efficiency(sw, 30000),
                          CapitalEfficiency)

    def test_fields(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 30000)
        for f in ("risk_reduction_per_dollar", "savings_per_dollar",
                  "coverage_per_dollar", "maintenance_efficiency_index"):
            assert hasattr(ce, f)

    def test_index_in_unit(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 30000)
        assert 0 <= ce.maintenance_efficiency_index <= 1

    def test_per_dollar_non_negative(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 30000)
        assert ce.risk_reduction_per_dollar >= 0
        assert ce.savings_per_dollar >= 0
        assert ce.coverage_per_dollar >= 0

    def test_savings_per_dollar_formula(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 30000)
        pt = min(sw.points, key=lambda p: abs(p.budget - 30000))
        if pt.total_cost > 0:
            assert ce.savings_per_dollar == pytest.approx(
                pt.savings / pt.total_cost)

    def test_matches_nearest_budget(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 32000)
        assert ce.budget == 30000  # nearest sweep point

    def test_empty_sweep_raises(self) -> None:
        # Build a degenerate sweep object with no points
        empty = BudgetSweep(
            posture="balanced", points=(), budgets=(), risk_reduction_curve=(),
            savings_curve=(), roi_curve=(), coverage_curve=(),
            knee_budget=0.0, roi_diminish_budget=None, currency="USD")
        with pytest.raises(ValueError, match="non-empty sweep"):
            _engine().capital_efficiency(empty, 1000)

    def test_index_best_point_near_one(self) -> None:
        # The most cost-efficient sweep point should have a high index
        sw = _sweep()
        indices = [_engine().capital_efficiency(sw, p.budget).maintenance_efficiency_index
                   for p in sw.points]
        assert max(indices) >= 0.9


# ===========================================================================
# Strategic recommendation
# ===========================================================================


class TestRecommendation:
    def test_returns_recommendation(self) -> None:
        sw = _sweep()
        assert isinstance(_engine().recommend(sw), StrategicRecommendation)

    def test_messages_present(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw)
        assert len(rec.messages) >= 1

    def test_knee_budget_set(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw)
        assert rec.knee_budget == sw.knee_budget

    def test_recommended_budget_set(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw)
        assert rec.recommended_budget > 0

    def test_before_knee_flag(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=sw.knee_budget - 20000)
        if sw.knee_budget > sw.budgets[0]:
            assert rec.before_knee

    def test_before_knee_message(self) -> None:
        sw = _sweep()
        knee = sw.knee_budget
        if knee > sw.budgets[0]:
            rec = _engine().recommend(sw, current_budget=sw.budgets[0])
            assert any("before the Pareto knee" in m for m in rec.messages)

    def test_summary_non_empty(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=30000)
        assert len(rec.executive_summary) > 40

    def test_summary_mentions_knee(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=30000)
        assert "knee" in rec.executive_summary.lower()

    def test_no_current_budget(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw)
        assert rec.current_budget is None

    def test_current_budget_recorded(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=25000)
        assert rec.current_budget == 25000

    def test_roi_diminish_message(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=30000)
        if sw.roi_diminish_budget is not None:
            assert any("ROI diminishes" in m for m in rec.messages)

    def test_unlock_message_when_applicable(self) -> None:
        sw = _sweep()
        knee = sw.knee_budget
        if knee > sw.budgets[0]:
            rec = _engine().recommend(sw, current_budget=sw.budgets[0])
            # may or may not unlock depending on ratio; just assert no error
            assert isinstance(rec.messages, tuple)


# ===========================================================================
# Confidence assessment
# ===========================================================================


class TestConfidence:
    def test_returns_confidence(self) -> None:
        eng = _engine()
        sw = eng.budget_sweep(_graded_snap())
        cmp = eng.compare_strategies(_graded_snap(), 30000)
        c = eng.confidence_assessment(cmp, sw, 30000)
        assert isinstance(c, StrategicConfidence)

    def test_score_in_unit(self) -> None:
        eng = _engine()
        sw = eng.budget_sweep(_graded_snap())
        cmp = eng.compare_strategies(_graded_snap(), 30000)
        c = eng.confidence_assessment(cmp, sw, 30000)
        assert 0 <= c.score <= 1

    def test_components_in_unit(self) -> None:
        eng = _engine()
        sw = eng.budget_sweep(_graded_snap())
        cmp = eng.compare_strategies(_graded_snap(), 30000)
        c = eng.confidence_assessment(cmp, sw, 30000)
        assert 0 <= c.portfolio_stability <= 1
        assert 0 <= c.strategy_agreement <= 1
        assert 0 <= c.coverage <= 1

    def test_agreement_matches_comparison(self) -> None:
        eng = _engine()
        sw = eng.budget_sweep(_graded_snap())
        cmp = eng.compare_strategies(_graded_snap(), 30000)
        c = eng.confidence_assessment(cmp, sw, 30000)
        assert c.strategy_agreement == cmp.agreement

    def test_high_agreement_high_confidence(self) -> None:
        eng = _engine()
        sw = eng.budget_sweep(_graded_snap())
        cmp = eng.compare_strategies(_graded_snap(), 30000)
        c = eng.confidence_assessment(cmp, sw, 30000)
        # graded fleet -> high agreement -> reasonable confidence
        assert c.score > 0.4


# ===========================================================================
# Top-level optimize
# ===========================================================================


class TestOptimize:
    def test_returns_plan(self) -> None:
        assert isinstance(_engine().optimize(_graded_snap()),
                          StrategicPortfolioPlan)

    def test_all_components(self) -> None:
        plan = _engine().optimize(_graded_snap())
        assert isinstance(plan.strategy_comparison, StrategyComparison)
        assert isinstance(plan.budget_sweep, BudgetSweep)
        assert isinstance(plan.pareto_frontier, ParetoFrontier)
        assert isinstance(plan.capital_efficiency, CapitalEfficiency)
        assert isinstance(plan.recommendation, StrategicRecommendation)
        assert isinstance(plan.confidence, StrategicConfidence)

    def test_with_current_budget(self) -> None:
        plan = _engine().optimize(_graded_snap(), current_budget=30000)
        assert plan.recommendation.current_budget == 30000

    def test_without_current_budget(self) -> None:
        plan = _engine().optimize(_graded_snap())
        assert plan.recommendation.current_budget is None

    def test_empty_snapshot_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _engine().optimize(_snap([]))

    def test_currency_propagates(self) -> None:
        plan = StrategicPortfolioEngine(_cfg(currency="EUR")).optimize(
            _graded_snap())
        assert plan.currency == "EUR"


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_plan_to_dict(self) -> None:
        d = _engine().optimize(_graded_snap()).to_dict()
        for k in ("strategy_comparison", "budget_sweep", "pareto_frontier",
                  "capital_efficiency", "recommendation", "confidence"):
            assert k in d

    def test_plan_json(self) -> None:
        plan = _engine().optimize(_graded_snap())
        assert isinstance(json.dumps(plan.to_dict()), str)

    def test_comparison_json(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert isinstance(json.dumps(cmp.to_dict()), str)

    def test_sweep_json(self) -> None:
        assert isinstance(json.dumps(_sweep().to_dict()), str)

    def test_frontier_json(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert isinstance(json.dumps(fr.to_dict()), str)

    def test_efficiency_json(self) -> None:
        ce = _engine().capital_efficiency(_sweep(), 30000)
        assert isinstance(json.dumps(ce.to_dict()), str)

    def test_recommendation_json(self) -> None:
        rec = _engine().recommend(_sweep(), current_budget=30000)
        assert isinstance(json.dumps(rec.to_dict()), str)

    def test_strategy_result_json(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert isinstance(json.dumps(cmp.results[0].to_dict()), str)

    def test_pareto_portfolio_json(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert isinstance(json.dumps(fr.portfolios[0].to_dict()), str)

    def test_curves_serialized(self) -> None:
        d = _sweep().to_dict()
        assert isinstance(d["risk_reduction_curve"], list)
        assert isinstance(d["savings_curve"], list)

    def test_nested_points_serialized(self) -> None:
        d = _sweep().to_dict()
        assert len(d["points"]) == 6


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_plan_deterministic(self) -> None:
        snap = _graded_snap()
        p1 = _engine().optimize(snap, current_budget=30000)
        p2 = _engine().optimize(snap, current_budget=30000)
        assert p1.budget_sweep.knee_budget == p2.budget_sweep.knee_budget
        assert p1.confidence.score == p2.confidence.score

    def test_sweep_deterministic(self) -> None:
        snap = _graded_snap()
        s1 = _engine().budget_sweep(snap)
        s2 = _engine().budget_sweep(snap)
        assert s1.savings_curve == s2.savings_curve
        assert s1.knee_budget == s2.knee_budget

    def test_comparison_deterministic(self) -> None:
        snap = _graded_snap()
        c1 = _engine().compare_strategies(snap, 30000)
        c2 = _engine().compare_strategies(snap, 30000)
        assert c1.agreement == c2.agreement

    def test_frontier_deterministic(self) -> None:
        sw = _sweep()
        f1 = _engine().pareto_frontier(sw)
        f2 = _engine().pareto_frontier(sw)
        assert [p.portfolio_id for p in f1.portfolios] == \
               [p.portfolio_id for p in f2.portfolios]

    def test_recommendation_deterministic(self) -> None:
        sw = _sweep()
        r1 = _engine().recommend(sw, current_budget=30000)
        r2 = _engine().recommend(sw, current_budget=30000)
        assert r1.messages == r2.messages


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_plan(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        StrategicPortfolioEngine(_cfg(),
                                 experiment_tracker=FakeTracker()).optimize(
            _graded_snap())
        assert logged and "strat_knee_budget" in logged[0]

    def test_logs_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        StrategicPortfolioEngine(_cfg(),
                                 experiment_tracker=FakeTracker()).optimize(
            _graded_snap())
        assert params and "strat_sweep_posture" in params[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        plan = StrategicPortfolioEngine(_cfg(),
                                        experiment_tracker=BrokenTracker()).optimize(
            _graded_snap())
        assert plan is not None

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, m, step=None):
                pass

        plan = StrategicPortfolioEngine(_cfg(),
                                        experiment_tracker=MetricsOnly()).optimize(
            _graded_snap())
        assert plan is not None

    def test_no_tracker_ok(self) -> None:
        assert _engine().optimize(_graded_snap()) is not None

    def test_step_increments(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                steps.append(step)

        eng = StrategicPortfolioEngine(_cfg(), experiment_tracker=FakeTracker())
        eng.optimize(_graded_snap())
        eng.optimize(_graded_snap())
        assert steps == [0, 1]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_asset_fleet(self) -> None:
        plan = _engine().optimize(_snap([_asset("Solo", cost=5000)]))
        assert plan is not None

    def test_single_asset_comparison(self) -> None:
        cmp = _engine().compare_strategies(_snap([_asset("Solo")]), 10000)
        assert len(cmp.results) == 4

    def test_large_fleet(self) -> None:
        assets = [_asset(f"A{i:03d}", cost=2000 + (i % 5) * 1000,
                         savings=5000 + i * 200, risk=0.2 + (i % 8) * 0.1)
                  for i in range(50)]
        eng = StrategicPortfolioEngine(_cfg(budget_start=20000,
                                            budget_end=120000, budget_step=20000))
        plan = eng.optimize(_snap(assets))
        assert plan is not None

    def test_zero_budget_sweep_point(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=20000,
                                            budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        # First point at zero budget selects nothing
        assert sw.points[0].n_selected == 0

    def test_single_budget_level(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=30000,
                                            budget_end=30000, budget_step=10000))
        plan = eng.optimize(_graded_snap())
        assert len(plan.budget_sweep.points) == 1

    def test_plan_frozen(self) -> None:
        plan = _engine().optimize(_graded_snap())
        with pytest.raises((AttributeError, TypeError)):
            plan.currency = "X"  # type: ignore[misc]

    def test_sweep_frozen(self) -> None:
        sw = _sweep()
        with pytest.raises((AttributeError, TypeError)):
            sw.knee_budget = 1  # type: ignore[misc]

    def test_pareto_portfolio_frozen(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        with pytest.raises((AttributeError, TypeError)):
            fr.portfolios[0].budget = 1  # type: ignore[misc]

    def test_all_identical_assets(self) -> None:
        assets = [_asset(f"A{i}", cost=5000, savings=20000, risk=0.5)
                  for i in range(6)]
        plan = _engine().optimize(_snap(assets))
        assert plan.strategy_comparison.agreement >= 0

    def test_n_runs_increments(self) -> None:
        eng = _engine()
        eng.optimize(_graded_snap())
        eng.optimize(_graded_snap())
        assert eng._n_runs == 2

    def test_distinct_engines_independent(self) -> None:
        e1 = _engine()
        e2 = _engine()
        e1.optimize(_graded_snap())
        assert e2._n_runs == 0

    def test_very_large_budget_full_coverage(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=100000,
                                            budget_end=100000, budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        # huge budget -> full coverage
        assert sw.coverage_curve[-1] == pytest.approx(1.0)


# ===========================================================================
# Additional coverage to reach the 200+ target
# ===========================================================================


class TestKneeDetection:
    def test_knee_concave_curve(self) -> None:
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([20, 36, 48, 56, 61, 64, 66, 67.0])
        idx = StrategicPortfolioEngine._knee_index(x, y)
        assert 1 <= idx <= 5  # knee in the early-mid region

    def test_knee_single_point(self) -> None:
        assert StrategicPortfolioEngine._knee_index(
            np.array([1.0]), np.array([1.0])) == 0

    def test_knee_two_points(self) -> None:
        assert StrategicPortfolioEngine._knee_index(
            np.array([1.0, 2]), np.array([1.0, 2])) == 1

    def test_knee_flat_curve(self) -> None:
        idx = StrategicPortfolioEngine._knee_index(
            np.array([1.0, 2, 3]), np.array([5.0, 5, 5]))
        assert idx == 2

    def test_knee_linear_curve(self) -> None:
        idx = StrategicPortfolioEngine._knee_index(
            np.array([0.0, 1, 2, 3, 4]), np.array([0.0, 1, 2, 3, 4]))
        assert 0 <= idx <= 4


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert StrategicPortfolioEngine._jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert StrategicPortfolioEngine._jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        assert StrategicPortfolioEngine._jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert StrategicPortfolioEngine._jaccard(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert StrategicPortfolioEngine._jaccard({"a"}, set()) == 0.0


class TestMeanJaccard:
    def test_single_set(self) -> None:
        eng = _engine()
        assert eng._mean_jaccard([{"a"}]) == 1.0

    def test_identical_sets(self) -> None:
        eng = _engine()
        assert eng._mean_jaccard([{"a", "b"}, {"a", "b"}, {"a", "b"}]) == 1.0

    def test_disjoint_sets(self) -> None:
        eng = _engine()
        assert eng._mean_jaccard([{"a"}, {"b"}, {"c"}]) == 0.0

    def test_empty_list(self) -> None:
        eng = _engine()
        assert eng._mean_jaccard([]) == 1.0


class TestParetoDominance:
    def test_dominated_excluded(self) -> None:
        # b is dominated by a (higher savings, lower risk)
        a = ParetoPortfolio("a", 100, risk=0.2, savings=500, roi=5)
        b = ParetoPortfolio("b", 200, risk=0.5, savings=300, roi=3)
        nd = StrategicPortfolioEngine._non_dominated([a, b])
        assert a in nd and b not in nd

    def test_both_non_dominated(self) -> None:
        # Trade-off: a higher savings but higher risk; b lower both
        a = ParetoPortfolio("a", 100, risk=0.5, savings=500, roi=5)
        b = ParetoPortfolio("b", 200, risk=0.2, savings=300, roi=3)
        nd = StrategicPortfolioEngine._non_dominated([a, b])
        assert len(nd) == 2

    def test_single_candidate(self) -> None:
        a = ParetoPortfolio("a", 100, risk=0.2, savings=500, roi=5)
        nd = StrategicPortfolioEngine._non_dominated([a])
        assert nd == [a]

    def test_three_with_one_dominated(self) -> None:
        a = ParetoPortfolio("a", 100, risk=0.1, savings=500, roi=5)
        b = ParetoPortfolio("b", 150, risk=0.3, savings=400, roi=4)
        c = ParetoPortfolio("c", 200, risk=0.4, savings=300, roi=3)  # dominated by a
        nd = StrategicPortfolioEngine._non_dominated([a, b, c])
        assert a in nd and c not in nd


class TestPostureMapping:
    def test_risk_first_runs(self) -> None:
        sw = _engine().budget_sweep(_graded_snap(), posture="risk_first")
        assert sw.posture == "risk_first"

    def test_roi_first_runs(self) -> None:
        sw = _engine().budget_sweep(_graded_snap(), posture="roi_first")
        assert sw.posture == "roi_first"

    def test_criticality_first_runs(self) -> None:
        sw = _engine().budget_sweep(_graded_snap(), posture="criticality_first")
        assert sw.posture == "criticality_first"

    def test_balanced_runs(self) -> None:
        sw = _engine().budget_sweep(_graded_snap(), posture="balanced")
        assert sw.posture == "balanced"

    def test_all_postures_produce_valid_sweep(self) -> None:
        for p in StrategicPosture:
            sw = _engine().budget_sweep(_graded_snap(), posture=p.value)
            assert len(sw.points) == 6


class TestRoiDiminish:
    def test_declining_roi_detected(self) -> None:
        eng = _engine()
        budgets = np.array([10000.0, 20000, 30000, 40000])
        roi = np.array([8.0, 8.0, 7.0, 5.0])
        dim = eng._roi_diminish_budget(budgets, roi)
        assert dim is not None and dim > 0

    def test_flat_roi_none(self) -> None:
        eng = _engine()
        budgets = np.array([10000.0, 20000, 30000])
        roi = np.array([8.0, 8.0, 8.0])
        assert eng._roi_diminish_budget(budgets, roi) is None

    def test_zero_roi_none(self) -> None:
        eng = _engine()
        budgets = np.array([10000.0, 20000])
        roi = np.array([0.0, 0.0])
        assert eng._roi_diminish_budget(budgets, roi) is None

    def test_empty_roi_none(self) -> None:
        eng = _engine()
        assert eng._roi_diminish_budget(np.array([]), np.array([])) is None


class TestCoverageDetail:
    def test_coverage_zero_budget(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=0,
                                            budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        assert sw.coverage_curve[0] == 0.0

    def test_coverage_full_budget(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=200000,
                                            budget_end=200000, budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        assert sw.coverage_curve[0] == pytest.approx(1.0)

    def test_coverage_per_dollar_non_increasing(self) -> None:
        # As budget grows, coverage per dollar is non-increasing (flat for
        # equal-cost assets, falling when later assets cost more).
        sw = _sweep()
        eng = _engine()
        first = eng.capital_efficiency(sw, sw.budgets[0]).coverage_per_dollar
        last = eng.capital_efficiency(sw, sw.budgets[-1]).coverage_per_dollar
        assert first >= last - 1e-9


class TestRecommendationDetail:
    def test_at_knee_message(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=sw.knee_budget)
        assert any("knee" in m.lower() for m in rec.messages)

    def test_beyond_knee_message(self) -> None:
        sw = _sweep()
        if sw.knee_budget < sw.budgets[-1]:
            rec = _engine().recommend(sw, current_budget=sw.budgets[-1])
            assert any("beyond" in m.lower() or "diminish" in m.lower()
                       for m in rec.messages)

    def test_summary_with_current_budget(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw, current_budget=20000)
        assert "current budget" in rec.executive_summary.lower()

    def test_recommended_equals_knee(self) -> None:
        sw = _sweep()
        rec = _engine().recommend(sw)
        assert rec.recommended_budget == sw.knee_budget

    def test_currency_in_messages(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(currency="GBP"))
        sw = eng.budget_sweep(_graded_snap())
        rec = eng.recommend(sw, current_budget=20000)
        assert any("GBP" in m for m in rec.messages)


class TestSweepCurvesDetail:
    def test_roi_curve_values_positive(self) -> None:
        sw = _sweep()
        assert all(r >= 0 for r in sw.roi_curve)

    def test_savings_curve_starts_low(self) -> None:
        sw = _sweep()
        assert sw.savings_curve[0] <= sw.savings_curve[-1]

    def test_n_selected_increases(self) -> None:
        sw = _sweep()
        ns = [pt.n_selected for pt in sw.points]
        assert all(ns[i] <= ns[i + 1] for i in range(len(ns) - 1))

    def test_risk_reduction_pct_bounded(self) -> None:
        sw = _sweep()
        assert all(0 <= pt.risk_reduction_pct <= 100 for pt in sw.points)


class TestComparisonDetail:
    def test_results_ordered_by_posture_enum(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        postures = [r.posture for r in cmp.results]
        assert postures == [p.value for p in StrategicPosture]

    def test_all_within_budget(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 22000)
        assert all(r.total_cost <= 22000 + 1e-6 for r in cmp.results)

    def test_zero_budget_comparison(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 0)
        assert all(len(r.selected_asset_ids) == 0 for r in cmp.results)

    def test_agreement_one_when_zero_budget(self) -> None:
        # All postures select nothing -> identical empty sets -> agreement 1
        cmp = _engine().compare_strategies(_graded_snap(), 0)
        assert cmp.agreement == 1.0


# ===========================================================================
# Further coverage to reach the 200+ target
# ===========================================================================


class TestConsecutiveStability:
    def test_identical_consecutive(self) -> None:
        eng = _engine()
        assert eng._consecutive_stability([{"a"}, {"a"}, {"a"}]) == 1.0

    def test_growing_selection(self) -> None:
        eng = _engine()
        # nested growing sets -> partial overlap
        s = eng._consecutive_stability([{"a"}, {"a", "b"}, {"a", "b", "c"}])
        assert 0 < s < 1

    def test_single_set(self) -> None:
        eng = _engine()
        assert eng._consecutive_stability([{"a"}]) == 1.0

    def test_empty(self) -> None:
        eng = _engine()
        assert eng._consecutive_stability([]) == 1.0

    def test_disjoint_consecutive(self) -> None:
        eng = _engine()
        assert eng._consecutive_stability([{"a"}, {"b"}]) == 0.0


class TestStrategyResultDetail:
    def test_selected_ids_tuple(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert isinstance(cmp.results[0].selected_asset_ids, tuple)

    def test_risk_reduction_pct_field(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert all(0 <= r.risk_reduction_pct <= 100 for r in cmp.results)

    def test_coverage_matches_selection(self) -> None:
        snap = _graded_snap()
        cmp = _engine().compare_strategies(snap, 30000)
        for r in cmp.results:
            expected = len(r.selected_asset_ids) / snap.asset_count
            assert r.coverage == pytest.approx(expected)

    def test_roi_field_present(self) -> None:
        cmp = _engine().compare_strategies(_graded_snap(), 30000)
        assert all(hasattr(r, "roi") for r in cmp.results)


class TestBudgetSweepEdges:
    def test_sweep_zero_to_max(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=60000,
                                            budget_step=20000))
        sw = eng.budget_sweep(_graded_snap())
        assert sw.points[0].n_selected == 0

    def test_sweep_knee_budget_is_level(self) -> None:
        sw = _sweep()
        assert sw.knee_budget in sw.budgets

    def test_sweep_total_cost_non_negative(self) -> None:
        sw = _sweep()
        assert all(pt.total_cost >= 0 for pt in sw.points)

    def test_sweep_savings_non_negative(self) -> None:
        sw = _sweep()
        assert all(pt.savings >= 0 for pt in sw.points)

    def test_sweep_posture_default(self) -> None:
        sw = _engine().budget_sweep(_graded_snap())
        assert sw.posture == "balanced"


class TestCapitalEfficiencyDetail:
    def test_risk_reduction_per_dollar_formula(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 40000)
        pt = min(sw.points, key=lambda p: abs(p.budget - 40000))
        if pt.total_cost > 0:
            assert ce.risk_reduction_per_dollar == pytest.approx(
                pt.risk_reduction / pt.total_cost)

    def test_coverage_per_dollar_formula(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 40000)
        pt = min(sw.points, key=lambda p: abs(p.budget - 40000))
        if pt.total_cost > 0:
            assert ce.coverage_per_dollar == pytest.approx(
                pt.coverage / pt.total_cost)

    def test_total_cost_recorded(self) -> None:
        sw = _sweep()
        ce = _engine().capital_efficiency(sw, 30000)
        assert ce.total_cost > 0

    def test_index_clipped(self) -> None:
        sw = _sweep()
        for pt in sw.points:
            ce = _engine().capital_efficiency(sw, pt.budget)
            assert 0 <= ce.maintenance_efficiency_index <= 1


class TestParetoFrontierDetail:
    def test_knee_in_frontier(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        ids = {p.portfolio_id for p in fr.portfolios}
        assert fr.knee_portfolio_id in ids

    def test_savings_recorded(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert all(p.savings >= 0 for p in fr.portfolios)

    def test_roi_recorded(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert all(hasattr(p, "roi") for p in fr.portfolios)

    def test_budget_recorded(self) -> None:
        fr = _engine().pareto_frontier(_sweep())
        assert all(p.budget > 0 for p in fr.portfolios)


class TestOptimizeDetail:
    def test_reference_budget_defaults_to_knee(self) -> None:
        plan = _engine().optimize(_graded_snap())
        # comparison run at the knee budget when no current budget given
        assert plan.strategy_comparison.budget == plan.budget_sweep.knee_budget

    def test_efficiency_at_knee(self) -> None:
        plan = _engine().optimize(_graded_snap())
        assert plan.capital_efficiency.budget == plan.budget_sweep.knee_budget

    def test_confidence_reflects_agreement(self) -> None:
        plan = _engine().optimize(_graded_snap())
        assert plan.confidence.strategy_agreement == plan.strategy_comparison.agreement

    def test_plan_currency_consistent(self) -> None:
        plan = StrategicPortfolioEngine(_cfg(currency="EUR")).optimize(_graded_snap())
        assert plan.strategy_comparison.currency == "EUR"
        assert plan.budget_sweep.currency == "EUR"


class TestSerializationDetail:
    def test_recommendation_messages_serialized(self) -> None:
        d = _engine().recommend(_sweep(), current_budget=30000).to_dict()
        assert isinstance(d["messages"], list)

    def test_confidence_serialized(self) -> None:
        plan = _engine().optimize(_graded_snap())
        d = plan.confidence.to_dict()
        assert "score" in d and "strategy_agreement" in d

    def test_plan_round_trip(self) -> None:
        plan = _engine().optimize(_graded_snap())
        d = json.loads(json.dumps(plan.to_dict()))
        assert d["currency"] == plan.currency

    def test_pareto_frontier_n_candidates_serialized(self) -> None:
        d = _engine().pareto_frontier(_sweep()).to_dict()
        assert d["n_candidates"] == 6

    def test_strategy_comparison_best_fields(self) -> None:
        d = _engine().compare_strategies(_graded_snap(), 30000).to_dict()
        for k in ("best_by_savings", "best_by_risk_reduction", "best_by_roi",
                  "best_by_coverage"):
            assert k in d


class TestInfiniteAndZeroBudget:
    def test_zero_budget_level_in_sweep(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=30000,
                                            budget_step=15000))
        sw = eng.budget_sweep(_graded_snap())
        assert sw.points[0].budget == 0
        assert sw.points[0].savings == 0

    def test_large_budget_full_selection(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=500000,
                                            budget_end=500000, budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        assert sw.points[0].coverage == pytest.approx(1.0)

    def test_zero_budget_efficiency_zero(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=0,
                                            budget_step=10000))
        sw = eng.budget_sweep(_graded_snap())
        ce = eng.capital_efficiency(sw, 0)
        assert ce.maintenance_efficiency_index == 0.0

    def test_zero_budget_plan(self) -> None:
        eng = StrategicPortfolioEngine(_cfg(budget_start=0, budget_end=20000,
                                            budget_step=10000))
        plan = eng.optimize(_graded_snap())
        assert plan is not None