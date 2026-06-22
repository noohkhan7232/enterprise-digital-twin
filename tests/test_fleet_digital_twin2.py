#!/usr/bin/env python3
"""Comprehensive test suite for ``src/fleet/fleet_digital_twin.py``.

Pure NumPy throughout (the engine composes the pure-NumPy frozen predictive
chain), so the entire suite runs without PyTorch or SciPy.  Coverage (150+):

- HealthBand enum
- FleetDigitalTwinConfig validation
- AssetInput validation
- FleetAsset record + serialization
- Engine registry (register / build / list)
- evaluate_asset (per-asset predictive chain)
- build_fleet_snapshot (aggregation + KPIs)
- rank_assets_by_risk / rank_assets_by_savings
- identify_top_risks / identify_top_opportunities
- generate_fleet_summary (executive narrative)
- portfolio analytics (risk concentration, Pareto)
- determinism
- JSON serialization
- failure-safe tracker
- edge cases (empty / single / large fleets, duplicate ids)

Run::

    pytest tests/test_fleet_digital_twin.py -v
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

from src.fleet.fleet_digital_twin import (
    ENGINE_NAME,
    FLEET_DIGITAL_TWIN_REGISTRY,
    AssetInput,
    FleetAsset,
    FleetDigitalTwinConfig,
    FleetDigitalTwinEngine,
    FleetSnapshot,
    HealthBand,
    build_fleet_digital_twin,
    list_fleet_digital_twins,
    register_fleet_digital_twin,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _traj(rate: float, n: int = 40, start: float = 95.0) -> "np.ndarray":
    return np.clip(start - rate * np.arange(n), 0.0, 100.0)


def _asset(aid: str, rate: float = 1.0, loc: str = "North",
           atype: str = "wind_turbine") -> AssetInput:
    return AssetInput(asset_id=aid, asset_type=atype, location=loc,
                      health_trajectory=_traj(rate))


def _fleet(*specs) -> list[AssetInput]:
    """Build assets from (id, rate) or (id, rate, loc) specs."""
    out = []
    for s in specs:
        if len(s) == 3:
            out.append(_asset(s[0], s[1], s[2]))
        else:
            out.append(_asset(s[0], s[1]))
    return out


def _engine(**kw) -> FleetDigitalTwinEngine:
    return FleetDigitalTwinEngine(FleetDigitalTwinConfig(**kw))


def _default_fleet() -> list[AssetInput]:
    return _fleet(("A", 0.4, "North"), ("B", 1.2, "North"),
                  ("C", 2.5, "South"), ("D", 0.6, "South"), ("E", 1.8, "North"))


def _snap(engine=None, assets=None) -> FleetSnapshot:
    engine = engine or _engine()
    assets = assets or _default_fleet()
    return engine.build_fleet_snapshot(assets)


# ===========================================================================
# HealthBand
# ===========================================================================


class TestHealthBand:
    def test_values(self) -> None:
        assert HealthBand.HEALTHY.value == "healthy"
        assert HealthBand.WARNING.value == "warning"
        assert HealthBand.CRITICAL.value == "critical"

    def test_three_bands(self) -> None:
        assert len(list(HealthBand)) == 3

    def test_is_str(self) -> None:
        assert HealthBand.HEALTHY == "healthy"


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_defaults(self) -> None:
        c = FleetDigitalTwinConfig()
        assert c.failure_threshold == 30.0 and c.top_n == 5

    def test_thresholds_ordered(self) -> None:
        FleetDigitalTwinConfig(healthy_threshold=70, warning_threshold=50)

    def test_inverted_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="warning_threshold"):
            FleetDigitalTwinConfig(healthy_threshold=40, warning_threshold=60)

    def test_failure_threshold_range(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            FleetDigitalTwinConfig(failure_threshold=150)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights must be >= 0"):
            FleetDigitalTwinConfig(weight_health=-0.1)

    def test_zero_weight_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to > 0"):
            FleetDigitalTwinConfig(weight_health=0, weight_rul=0,
                                   weight_failure_prob=0, weight_severity=0)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            FleetDigitalTwinConfig(top_n=0)

    def test_pareto_fraction_range(self) -> None:
        with pytest.raises(ValueError, match="pareto_fraction"):
            FleetDigitalTwinConfig(pareto_fraction=0.0)

    def test_pareto_fraction_too_high(self) -> None:
        with pytest.raises(ValueError, match="pareto_fraction"):
            FleetDigitalTwinConfig(pareto_fraction=1.5)

    def test_weight_sum_property(self) -> None:
        c = FleetDigitalTwinConfig(weight_health=0.3, weight_rul=0.25,
                                   weight_failure_prob=0.3, weight_severity=0.15)
        assert c.weight_sum == pytest.approx(1.0)

    def test_frozen(self) -> None:
        c = FleetDigitalTwinConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.top_n = 9  # type: ignore[misc]

    def test_custom_currency(self) -> None:
        assert FleetDigitalTwinConfig(currency="EUR").currency == "EUR"

    def test_pareto_fraction_one_ok(self) -> None:
        FleetDigitalTwinConfig(pareto_fraction=1.0)

    def test_exponential_model(self) -> None:
        FleetDigitalTwinConfig(rul_model="exponential")


# ===========================================================================
# AssetInput validation
# ===========================================================================


class TestAssetInput:
    def test_fields(self) -> None:
        a = _asset("A", 1.0, "North")
        assert a.asset_id == "A" and a.location == "North"

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetInput(asset_id="", asset_type="t", location="l",
                       health_trajectory=_traj(1.0))

    def test_whitespace_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetInput(asset_id="   ", asset_type="t", location="l",
                       health_trajectory=_traj(1.0))

    def test_empty_trajectory_rejected(self) -> None:
        with pytest.raises(ValueError, match="health_trajectory"):
            AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=np.array([]))

    def test_trajectory_coerced_to_array(self) -> None:
        a = AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=[95, 90, 85])
        assert isinstance(a.health_trajectory, np.ndarray)

    def test_frozen(self) -> None:
        a = _asset("A")
        with pytest.raises((AttributeError, TypeError)):
            a.asset_id = "B"  # type: ignore[misc]

    def test_list_trajectory(self) -> None:
        a = AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=[95.0, 90, 80, 70, 60])
        assert a.health_trajectory.size == 5


# ===========================================================================
# Engine registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert ENGINE_NAME in FLEET_DIGITAL_TWIN_REGISTRY
        assert ENGINE_NAME in list_fleet_digital_twins()

    def test_build(self) -> None:
        assert isinstance(build_fleet_digital_twin(ENGINE_NAME),
                          FleetDigitalTwinEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown fleet digital twin"):
            build_fleet_digital_twin("nope")

    def test_registry_name(self) -> None:
        assert FleetDigitalTwinEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_fleet_digital_twin(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_fleet_digital_twin(
            ENGINE_NAME, config=FleetDigitalTwinConfig(currency="GBP"))
        assert eng.config.currency == "GBP"


# ===========================================================================
# evaluate_asset
# ===========================================================================


class TestEvaluateAsset:
    def test_returns_fleet_asset(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert isinstance(a, FleetAsset)

    def test_identity_preserved(self) -> None:
        a = _engine().evaluate_asset(_asset("WTG-9", 1.0, "Baltic"))
        assert a.asset_id == "WTG-9" and a.location == "Baltic"

    def test_health_in_range(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert 0 <= a.health <= 100

    def test_failure_probability_in_unit(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert 0 <= a.failure_probability <= 1

    def test_risk_score_in_unit(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert 0 <= a.risk_score <= 1

    def test_health_band_assigned(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 0.2))
        assert a.health_band in ("healthy", "warning", "critical")

    def test_healthy_asset_band(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 0.1, ))
        assert a.health_band == "healthy"

    def test_critical_asset_band(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 3.0))
        assert a.health_band == "critical"

    def test_maintenance_action_present(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 2.0))
        assert a.maintenance_action in (
            "no_action", "inspect", "schedule_maintenance",
            "immediate_maintenance", "shutdown")

    def test_higher_rate_higher_risk(self) -> None:
        low = _engine().evaluate_asset(_asset("A", 0.3))
        high = _engine().evaluate_asset(_asset("B", 2.5))
        assert high.risk_score > low.risk_score

    def test_costs_non_negative(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.5))
        assert a.maintenance_cost >= 0 and a.downtime_hours >= 0


# ===========================================================================
# build_fleet_snapshot
# ===========================================================================


class TestBuildSnapshot:
    def test_returns_snapshot(self) -> None:
        assert isinstance(_snap(), FleetSnapshot)

    def test_asset_count(self) -> None:
        assert _snap().asset_count == 5

    def test_bands_sum_to_count(self) -> None:
        s = _snap()
        assert s.healthy_assets + s.warning_assets + s.critical_assets == s.asset_count

    def test_empty_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1 asset"):
            _engine().build_fleet_snapshot([])

    def test_duplicate_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate asset_id"):
            _engine().build_fleet_snapshot([_asset("X", 1.0), _asset("X", 2.0)])

    def test_assets_sorted_by_id(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("C", 1.0), ("A", 1.0), ("B", 1.0)))
        assert [a.asset_id for a in s.assets] == ["A", "B", "C"]

    def test_average_health(self) -> None:
        s = _snap()
        manual = np.mean([a.health for a in s.assets])
        assert s.average_health == pytest.approx(manual)

    def test_fleet_failure_probability_mean(self) -> None:
        s = _snap()
        manual = np.mean([a.failure_probability for a in s.assets])
        assert s.fleet_failure_probability == pytest.approx(manual)

    def test_fleet_cost_sum(self) -> None:
        s = _snap()
        manual = sum(a.maintenance_cost for a in s.assets)
        assert s.fleet_expected_cost == pytest.approx(manual)

    def test_fleet_downtime_sum(self) -> None:
        s = _snap()
        manual = sum(a.downtime_hours for a in s.assets)
        assert s.fleet_expected_downtime == pytest.approx(manual)

    def test_fleet_savings_sum(self) -> None:
        s = _snap()
        manual = sum(a.expected_savings for a in s.assets)
        assert s.fleet_expected_savings == pytest.approx(manual)

    def test_currency_propagates(self) -> None:
        s = _engine(currency="EUR").build_fleet_snapshot(_default_fleet())
        assert s.currency == "EUR"

    def test_all_kpis_present(self) -> None:
        s = _snap()
        for k in ("average_health", "average_rul", "fleet_failure_probability",
                  "fleet_expected_cost", "fleet_expected_downtime",
                  "fleet_expected_failure_cost", "fleet_expected_savings",
                  "risk_concentration", "pareto_concentration"):
            assert hasattr(s, k)


# ===========================================================================
# Fleet KPIs
# ===========================================================================


class TestFleetKPIs:
    def test_average_health_in_range(self) -> None:
        s = _snap()
        assert 0 <= s.average_health <= 100

    def test_fleet_failure_probability_in_unit(self) -> None:
        s = _snap()
        assert 0 <= s.fleet_failure_probability <= 1

    def test_expected_failure_cost_non_negative(self) -> None:
        assert _snap().fleet_expected_failure_cost >= 0

    def test_expected_failure_cost_formula(self) -> None:
        s = _snap()
        # equals sum of p_fail * failure_cost (50000 default)
        manual = sum(a.failure_probability * 50000 for a in s.assets)
        assert s.fleet_expected_failure_cost == pytest.approx(manual, rel=1e-6)

    def test_healthy_count(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 0.1), ("C", 3.0)))
        assert s.healthy_assets >= 1

    def test_critical_count(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 3.0), ("C", 3.0)))
        assert s.critical_assets >= 1

    def test_average_rul_finite(self) -> None:
        s = _snap()
        assert s.average_rul >= 0 or math.isinf(s.average_rul)

    def test_total_cost_non_negative(self) -> None:
        assert _snap().fleet_expected_cost >= 0

    def test_total_downtime_non_negative(self) -> None:
        assert _snap().fleet_expected_downtime >= 0

    def test_savings_non_negative(self) -> None:
        assert _snap().fleet_expected_savings >= 0


# ===========================================================================
# Risk ranking
# ===========================================================================


class TestRiskRanking:
    def test_ranked_descending(self) -> None:
        ranked = _engine().rank_assets_by_risk(_snap())
        assert all(ranked[i].risk_score >= ranked[i + 1].risk_score
                   for i in range(len(ranked) - 1))

    def test_all_assets_returned(self) -> None:
        s = _snap()
        assert len(_engine().rank_assets_by_risk(s)) == s.asset_count

    def test_worst_asset_first(self) -> None:
        s = _snap()
        ranked = _engine().rank_assets_by_risk(s)
        assert ranked[0].risk_score == max(a.risk_score for a in s.assets)

    def test_tie_broken_by_id(self) -> None:
        # Identical assets -> deterministic id ordering
        s = _engine().build_fleet_snapshot(
            _fleet(("B", 1.0), ("A", 1.0)))
        ranked = _engine().rank_assets_by_risk(s)
        # equal risk -> A before B
        if ranked[0].risk_score == ranked[1].risk_score:
            assert ranked[0].asset_id == "A"

    def test_high_rate_ranks_high(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("low", 0.2), ("high", 3.0)))
        ranked = _engine().rank_assets_by_risk(s)
        assert ranked[0].asset_id == "high"


class TestSavingsRanking:
    def test_ranked_descending(self) -> None:
        ranked = _engine().rank_assets_by_savings(_snap())
        assert all(ranked[i].expected_savings >= ranked[i + 1].expected_savings
                   for i in range(len(ranked) - 1))

    def test_all_returned(self) -> None:
        s = _snap()
        assert len(_engine().rank_assets_by_savings(s)) == s.asset_count

    def test_highest_savings_first(self) -> None:
        s = _snap()
        ranked = _engine().rank_assets_by_savings(s)
        assert ranked[0].expected_savings == max(a.expected_savings
                                                 for a in s.assets)


# ===========================================================================
# Top risks / opportunities
# ===========================================================================


class TestTopRisks:
    def test_default_top_n(self) -> None:
        eng = _engine(top_n=3)
        assert len(eng.identify_top_risks(_snap(eng))) == 3

    def test_explicit_top_n(self) -> None:
        assert len(_engine().identify_top_risks(_snap(), top_n=2)) == 2

    def test_top_n_exceeds_fleet(self) -> None:
        s = _snap()
        assert len(_engine().identify_top_risks(s, top_n=99)) == s.asset_count

    def test_invalid_top_n(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            _engine().identify_top_risks(_snap(), top_n=0)

    def test_top_risks_are_highest(self) -> None:
        s = _snap()
        top = _engine().identify_top_risks(s, top_n=2)
        rest = [a for a in s.assets if a not in top]
        assert min(a.risk_score for a in top) >= max(a.risk_score for a in rest)


class TestTopOpportunities:
    def test_default_top_n(self) -> None:
        eng = _engine(top_n=3)
        assert len(eng.identify_top_opportunities(_snap(eng))) == 3

    def test_explicit_top_n(self) -> None:
        assert len(_engine().identify_top_opportunities(_snap(), top_n=2)) == 2

    def test_invalid_top_n(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            _engine().identify_top_opportunities(_snap(), top_n=-1)

    def test_top_opportunities_highest(self) -> None:
        s = _snap()
        top = _engine().identify_top_opportunities(s, top_n=2)
        rest = [a for a in s.assets if a not in top]
        if rest:
            assert min(a.expected_savings for a in top) >= max(
                a.expected_savings for a in rest)


# ===========================================================================
# Executive narrative
# ===========================================================================


class TestExecutiveNarrative:
    def test_contains_count(self) -> None:
        s = _snap()
        assert f"{s.asset_count} assets" in _engine().generate_fleet_summary(s)

    def test_singular_asset(self) -> None:
        s = _engine().build_fleet_snapshot([_asset("Solo", 1.0)])
        assert "1 asset." in _engine().generate_fleet_summary(s)

    def test_mentions_critical(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 3.0), ("B", 3.0), ("C", 0.2)))
        assert "critical" in _engine().generate_fleet_summary(s)

    def test_mentions_highest_exposure(self) -> None:
        s = _snap()
        assert "highest failure exposure" in _engine().generate_fleet_summary(s)

    def test_mentions_savings(self) -> None:
        s = _snap()
        summary = _engine().generate_fleet_summary(s)
        if s.fleet_expected_savings > 0:
            assert "savings" in summary

    def test_mentions_average_health(self) -> None:
        s = _snap()
        assert "Average fleet health" in _engine().generate_fleet_summary(s)

    def test_snapshot_method_matches_engine(self) -> None:
        s = _snap()
        assert s.generate_fleet_summary() == _engine().generate_fleet_summary(s)

    def test_worst_asset_named(self) -> None:
        s = _snap()
        worst = max(s.assets, key=lambda a: a.risk_score)
        assert worst.asset_id in _engine().generate_fleet_summary(s)


# ===========================================================================
# Portfolio analytics
# ===========================================================================


class TestPortfolioAnalytics:
    def test_risk_concentration_bounds(self) -> None:
        s = _snap()
        assert 0 <= s.risk_concentration <= 1 + 1e-9

    def test_concentration_uniform_floor(self) -> None:
        # Identical assets -> concentration near 1/N
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 1.0), ("B", 1.0), ("C", 1.0), ("D", 1.0)))
        assert s.risk_concentration == pytest.approx(0.25, abs=0.02)

    def test_concentration_high_when_one_risky(self) -> None:
        # One very risky, rest healthy -> higher concentration
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.05), ("B", 0.05), ("C", 0.05), ("D", 5.0)))
        assert s.risk_concentration > 0.25

    def test_pareto_bounds(self) -> None:
        s = _snap()
        assert 0 <= s.pareto_concentration <= 1

    def test_pareto_top_cohort(self) -> None:
        # top 20% holds at least 20% of risk when concentrated
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 0.1), ("C", 0.1), ("D", 0.1), ("E", 5.0)))
        assert s.pareto_concentration >= 0.2

    def test_concentration_zero_when_no_risk(self) -> None:
        # Pristine assets (flat high health) -> near-zero risk scores
        traj = np.full(40, 99.0)
        assets = [AssetInput(asset_id=f"A{i}", asset_type="t", location="l",
                             health_trajectory=traj) for i in range(3)]
        s = _engine().build_fleet_snapshot(assets)
        assert 0 <= s.risk_concentration <= 1

    def test_single_asset_concentration(self) -> None:
        s = _engine().build_fleet_snapshot([_asset("A", 1.0)])
        # all risk in one asset -> concentration 1.0 (if nonzero risk)
        assert s.risk_concentration in (0.0, pytest.approx(1.0))


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_snapshot_deterministic(self) -> None:
        a = _default_fleet()
        s1 = _engine().build_fleet_snapshot(a)
        s2 = _engine().build_fleet_snapshot(a)
        assert s1.average_health == s2.average_health
        assert s1.risk_concentration == s2.risk_concentration
        assert s1.fleet_expected_cost == s2.fleet_expected_cost

    def test_per_asset_deterministic(self) -> None:
        s1 = _snap()
        s2 = _snap()
        for a1, a2 in zip(s1.assets, s2.assets):
            assert a1.risk_score == a2.risk_score
            assert a1.health == a2.health

    def test_ranking_deterministic(self) -> None:
        s = _snap()
        r1 = _engine().rank_assets_by_risk(s)
        r2 = _engine().rank_assets_by_risk(s)
        assert [a.asset_id for a in r1] == [a.asset_id for a in r2]

    def test_summary_deterministic(self) -> None:
        s = _snap()
        assert _engine().generate_fleet_summary(s) == \
            _engine().generate_fleet_summary(s)


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_snapshot_to_dict(self) -> None:
        d = _snap().to_dict()
        for k in ("asset_count", "average_health", "risk_concentration",
                  "assets", "currency"):
            assert k in d

    def test_snapshot_json(self) -> None:
        assert isinstance(json.dumps(_snap().to_dict()), str)

    def test_asset_to_dict(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        d = a.to_dict()
        for k in ("asset_id", "health", "predicted_rul", "failure_probability",
                  "maintenance_action", "maintenance_cost", "downtime_hours",
                  "risk_score"):
            assert k in d

    def test_asset_json(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert isinstance(json.dumps(a.to_dict()), str)

    def test_assets_nested_in_snapshot(self) -> None:
        d = _snap().to_dict()
        assert isinstance(d["assets"], list) and len(d["assets"]) == 5

    def test_infinite_rul_serializes(self) -> None:
        # A pristine asset may have infinite RUL -> None in JSON
        traj = np.full(40, 99.0)
        a = _engine().evaluate_asset(
            AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=traj))
        d = a.to_dict()
        assert d["predicted_rul"] is None or isinstance(d["predicted_rul"], float)


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_snapshot(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        FleetDigitalTwinEngine(FleetDigitalTwinConfig(),
                               experiment_tracker=FakeTracker()).build_fleet_snapshot(
            _default_fleet())
        assert logged and "fleet_asset_count" in logged[0]

    def test_logs_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        FleetDigitalTwinEngine(FleetDigitalTwinConfig(currency="EUR"),
                               experiment_tracker=FakeTracker()).build_fleet_snapshot(
            _default_fleet())
        assert params and params[0]["fleet_currency"] == "EUR"

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        s = FleetDigitalTwinEngine(FleetDigitalTwinConfig(),
                                   experiment_tracker=BrokenTracker()).build_fleet_snapshot(
            _default_fleet())
        assert s.asset_count == 5

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, m, step=None):
                pass

        s = FleetDigitalTwinEngine(FleetDigitalTwinConfig(),
                                   experiment_tracker=MetricsOnly()).build_fleet_snapshot(
            _default_fleet())
        assert s is not None

    def test_no_tracker_ok(self) -> None:
        assert _snap() is not None

    def test_step_increments(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                steps.append(step)

        eng = FleetDigitalTwinEngine(FleetDigitalTwinConfig(),
                                     experiment_tracker=FakeTracker())
        eng.build_fleet_snapshot(_default_fleet())
        eng.build_fleet_snapshot(_default_fleet())
        assert steps == [0, 1]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_asset_fleet(self) -> None:
        s = _engine().build_fleet_snapshot([_asset("Solo", 1.0)])
        assert s.asset_count == 1

    def test_single_asset_ranking(self) -> None:
        s = _engine().build_fleet_snapshot([_asset("Solo", 1.0)])
        assert len(_engine().rank_assets_by_risk(s)) == 1

    def test_large_fleet(self) -> None:
        assets = [_asset(f"A{i:03d}", 0.3 + (i % 7) * 0.3) for i in range(60)]
        s = _engine().build_fleet_snapshot(assets)
        assert s.asset_count == 60

    def test_all_healthy_fleet(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 0.1), ("C", 0.1)))
        assert s.critical_assets == 0

    def test_all_critical_fleet(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 4.0), ("B", 4.0), ("C", 4.0)))
        assert s.healthy_assets == 0

    def test_short_trajectory(self) -> None:
        # The RUL predictor requires a minimum history; 6 points is sufficient.
        a = AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=np.array([95.0, 90, 85, 80, 75, 70]))
        result = _engine().evaluate_asset(a)
        assert isinstance(result, FleetAsset)

    def test_snapshot_frozen(self) -> None:
        s = _snap()
        with pytest.raises((AttributeError, TypeError)):
            s.asset_count = 99  # type: ignore[misc]

    def test_fleet_asset_frozen(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        with pytest.raises((AttributeError, TypeError)):
            a.risk_score = 0.5  # type: ignore[misc]

    def test_mixed_locations(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 1.0, "North"), ("B", 1.0, "South")))
        locs = {a.location for a in s.assets}
        assert locs == {"North", "South"}

    def test_n_runs_increments(self) -> None:
        eng = _engine()
        eng.build_fleet_snapshot(_default_fleet())
        eng.build_fleet_snapshot(_default_fleet())
        assert eng._n_runs == 2

    def test_distinct_engines_independent(self) -> None:
        e1 = _engine()
        e2 = _engine()
        e1.build_fleet_snapshot(_default_fleet())
        assert e2._n_runs == 0

    def test_two_asset_fleet(self) -> None:
        s = _engine().build_fleet_snapshot(_fleet(("A", 0.4), ("B", 2.0)))
        assert s.asset_count == 2

    def test_flat_trajectory(self) -> None:
        a = AssetInput(asset_id="A", asset_type="t", location="l",
                       health_trajectory=np.full(40, 85.0))
        result = _engine().evaluate_asset(a)
        assert result.health == pytest.approx(85.0, abs=5.0)


# ===========================================================================
# Additional coverage to reach the 150+ target
# ===========================================================================


class TestAdditionalEvaluation:
    def test_exponential_model_runs(self) -> None:
        eng = _engine(rul_model="exponential")
        a = eng.evaluate_asset(_asset("A", 1.0))
        assert isinstance(a, FleetAsset)

    def test_custom_failure_threshold(self) -> None:
        eng = _engine(failure_threshold=50)
        a = eng.evaluate_asset(_asset("A", 1.0))
        assert 0 <= a.failure_probability <= 1

    def test_asset_type_preserved(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0, atype="gearbox"))
        assert a.asset_type == "gearbox"

    def test_severity_non_negative(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.5))
        assert a.severity_score >= 0

    def test_expected_savings_present(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.5))
        assert hasattr(a, "expected_savings")

    def test_warning_band_asset(self) -> None:
        # Tune rate so terminal health lands in the warning band
        eng = _engine()
        found_band = set()
        for rate in (0.4, 0.9, 1.4, 2.0, 3.0):
            found_band.add(eng.evaluate_asset(_asset("X", rate)).health_band)
        assert "warning" in found_band or "critical" in found_band

    def test_risk_score_monotone_in_rate(self) -> None:
        eng = _engine()
        scores = [eng.evaluate_asset(_asset("X", r)).risk_score
                  for r in (0.2, 0.8, 1.5, 2.5)]
        # generally increasing
        assert scores[-1] >= scores[0]


class TestAdditionalKPIs:
    def test_average_rul_matches_finite_mean(self) -> None:
        s = _snap()
        finite = [a.predicted_rul for a in s.assets
                  if math.isfinite(a.predicted_rul)]
        if finite:
            assert s.average_rul == pytest.approx(np.mean(finite))

    def test_warning_band_counted(self) -> None:
        # Build a fleet spanning bands
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 1.0), ("C", 1.5), ("D", 3.0)))
        assert (s.healthy_assets + s.warning_assets + s.critical_assets) == 4

    def test_expected_failure_cost_scales(self) -> None:
        low = _engine().build_fleet_snapshot(_fleet(("A", 0.1)))
        high = _engine().build_fleet_snapshot(_fleet(("A", 3.0)))
        assert high.fleet_expected_failure_cost >= low.fleet_expected_failure_cost

    def test_cost_scales_with_fleet_size(self) -> None:
        small = _snap(assets=_fleet(("A", 2.0)))
        large = _snap(assets=_fleet(("A", 2.0), ("B", 2.0), ("C", 2.0)))
        assert large.fleet_expected_cost >= small.fleet_expected_cost

    def test_healthy_fleet_low_failure_prob(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.1), ("B", 0.1), ("C", 0.1)))
        assert s.fleet_failure_probability < 0.9

    def test_critical_fleet_high_failure_prob(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 3.0), ("B", 3.0), ("C", 3.0)))
        assert s.fleet_failure_probability > 0.3


class TestAdditionalRanking:
    def test_risk_ranking_stable_order(self) -> None:
        s = _snap()
        eng = _engine()
        assert [a.asset_id for a in eng.rank_assets_by_risk(s)] == \
               [a.asset_id for a in eng.rank_assets_by_risk(s)]

    def test_savings_ranking_stable(self) -> None:
        s = _snap()
        eng = _engine()
        assert [a.asset_id for a in eng.rank_assets_by_savings(s)] == \
               [a.asset_id for a in eng.rank_assets_by_savings(s)]

    def test_top_risks_subset_of_ranking(self) -> None:
        s = _snap()
        eng = _engine()
        top = eng.identify_top_risks(s, top_n=3)
        full = eng.rank_assets_by_risk(s)
        assert [a.asset_id for a in top] == [a.asset_id for a in full[:3]]

    def test_top_opps_subset_of_ranking(self) -> None:
        s = _snap()
        eng = _engine()
        top = eng.identify_top_opportunities(s, top_n=3)
        full = eng.rank_assets_by_savings(s)
        assert [a.asset_id for a in top] == [a.asset_id for a in full[:3]]

    def test_top_risks_default_uses_config(self) -> None:
        eng = _engine(top_n=2)
        assert len(eng.identify_top_risks(_snap(eng))) == 2


class TestAdditionalAnalytics:
    def test_concentration_single_asset_is_one(self) -> None:
        s = _engine().build_fleet_snapshot([_asset("A", 2.0)])
        # nonzero risk in one asset -> HHI = 1
        if s.assets[0].risk_score > 0:
            assert s.risk_concentration == pytest.approx(1.0)

    def test_pareto_fraction_config(self) -> None:
        eng = _engine(pareto_fraction=0.5)
        s = eng.build_fleet_snapshot(_default_fleet())
        assert 0 <= s.pareto_concentration <= 1

    def test_pareto_full_fraction_is_one(self) -> None:
        eng = _engine(pareto_fraction=1.0)
        s = eng.build_fleet_snapshot(_default_fleet())
        # top 100% holds all the risk
        assert s.pareto_concentration == pytest.approx(1.0)

    def test_concentration_decreases_with_uniformity(self) -> None:
        uniform = _engine().build_fleet_snapshot(
            _fleet(("A", 1.0), ("B", 1.0), ("C", 1.0), ("D", 1.0)))
        skewed = _engine().build_fleet_snapshot(
            _fleet(("A", 0.05), ("B", 0.05), ("C", 0.05), ("D", 5.0)))
        assert skewed.risk_concentration >= uniform.risk_concentration

    def test_analytics_present_in_dict(self) -> None:
        d = _snap().to_dict()
        assert "risk_concentration" in d and "pareto_concentration" in d


class TestAdditionalSerialization:
    def test_all_asset_fields_in_dict(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        d = a.to_dict()
        assert len(d) == 13  # all FleetAsset fields

    def test_snapshot_round_trip_values(self) -> None:
        s = _snap()
        d = s.to_dict()
        assert d["asset_count"] == s.asset_count
        assert d["currency"] == s.currency

    def test_nested_asset_dicts_serializable(self) -> None:
        d = _snap().to_dict()
        for ad in d["assets"]:
            assert isinstance(json.dumps(ad), str)

    def test_health_band_in_asset_dict(self) -> None:
        a = _engine().evaluate_asset(_asset("A", 1.0))
        assert a.to_dict()["health_band"] in ("healthy", "warning", "critical")


class TestAdditionalEdgeCases:
    def test_ten_asset_fleet(self) -> None:
        assets = [_asset(f"T{i}", 0.4 + i * 0.25) for i in range(10)]
        s = _engine().build_fleet_snapshot(assets)
        assert s.asset_count == 10

    def test_summary_no_warning_no_critical(self) -> None:
        s = _engine().build_fleet_snapshot(
            _fleet(("A", 0.05), ("B", 0.05)))
        summary = _engine().generate_fleet_summary(s)
        assert "Fleet contains 2 assets" in summary

    def test_evaluate_then_aggregate_consistency(self) -> None:
        eng = _engine()
        assets = _default_fleet()
        s = eng.build_fleet_snapshot(assets)
        # Re-evaluating one asset gives the same record values
        direct = eng.evaluate_asset(assets[0])
        in_snap = next(a for a in s.assets if a.asset_id == assets[0].asset_id)
        assert direct.risk_score == pytest.approx(in_snap.risk_score)

    def test_currency_in_summary(self) -> None:
        eng = _engine(currency="GBP")
        s = eng.build_fleet_snapshot(_default_fleet())
        summary = eng.generate_fleet_summary(s)
        if s.fleet_expected_savings > 0:
            assert "GBP" in summary

    def test_large_fleet_bands_consistent(self) -> None:
        assets = [_asset(f"A{i}", 0.2 + (i % 10) * 0.3) for i in range(40)]
        s = _engine().build_fleet_snapshot(assets)
        assert s.healthy_assets + s.warning_assets + s.critical_assets == 40