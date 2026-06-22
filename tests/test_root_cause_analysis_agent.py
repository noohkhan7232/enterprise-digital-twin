#!/usr/bin/env python3
"""Comprehensive test suite for ``src/agent/root_cause_analysis_agent.py``.

Pure NumPy / pure Python (rule-based, no LLM), so the entire suite runs without
PyTorch, SciPy, or pandas.  Coverage (180+ tests):

- CauseCategory enum & investigation map
- RCAConfig validation
- AssetEvidence validation & indicators
- Registry (register / build / list)
- Cause scoring (scores, contributions, confidence, ordering)
- Single-asset analysis (single / multi / conflicting / unknown)
- Investigation recommendations
- Fleet RCA (distribution, most-common, highest-risk, concentration)
- Executive summary
- Serialization (JSON)
- Determinism
- Failure-safe behaviour & edge cases

Run::

    pytest tests/test_root_cause_analysis_agent.py -v
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
from src.agent.root_cause_analysis_agent import (
    AGENT_NAME,
    RCA_AGENT_REGISTRY,
    AssetEvidence,
    CauseCategory,
    CauseFrequency,
    CauseScore,
    ExecutiveRCASummary,
    FleetRCAReport,
    RCAConfig,
    RootCauseAnalysisAgent,
    RootCauseReport,
    build_root_cause_analysis_agent,
    list_root_cause_analysis_agents,
    register_root_cause_analysis_agent,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _ev(aid: str = "A", **kw) -> AssetEvidence:
    return AssetEvidence(asset_id=aid, **kw)


def _agent(**kw) -> RootCauseAnalysisAgent:
    return RootCauseAnalysisAgent(RCAConfig(**kw) if kw else None)


def _fleet_asset(aid: str, risk: float) -> FleetAsset:
    return FleetAsset(
        asset_id=aid, asset_type="wt", location="N", health=50.0,
        predicted_rul=40.0, failure_probability=0.5,
        maintenance_action="schedule_maintenance", maintenance_cost=5000.0,
        downtime_hours=8.0, expected_savings=20000.0, severity_score=10.0,
        health_band="warning", risk_score=risk)


def _snap(assets) -> FleetSnapshot:
    assets = tuple(assets)
    n = len(assets)
    return FleetSnapshot(
        asset_count=n, healthy_assets=0, warning_assets=n, critical_assets=0,
        average_health=50.0, average_rul=40.0, fleet_failure_probability=0.5,
        fleet_expected_cost=5000.0 * n, fleet_expected_downtime=8.0 * n,
        fleet_expected_failure_cost=100000.0, fleet_expected_savings=20000.0 * n,
        risk_concentration=0.3, pareto_concentration=0.4, assets=assets,
        currency="USD")


# ===========================================================================
# CauseCategory
# ===========================================================================


class TestCauseCategory:
    def test_nine_categories(self) -> None:
        assert len(list(CauseCategory)) == 9

    def test_values(self) -> None:
        assert CauseCategory.TEMPERATURE.value == "temperature"
        assert CauseCategory.VIBRATION.value == "vibration"
        assert CauseCategory.UNKNOWN.value == "unknown"

    def test_all_required_present(self) -> None:
        names = {c.value for c in CauseCategory}
        for required in ("temperature", "vibration", "pressure", "load",
                         "lubrication", "electrical", "environmental",
                         "operational", "unknown"):
            assert required in names

    def test_is_str(self) -> None:
        assert CauseCategory.VIBRATION == "vibration"


# ===========================================================================
# RCAConfig
# ===========================================================================


class TestRCAConfig:
    def test_defaults(self) -> None:
        c = RCAConfig()
        assert c.evidence_floor == 0.15 and c.top_n == 5

    def test_floor_range(self) -> None:
        with pytest.raises(ValueError, match="evidence_floor"):
            RCAConfig(evidence_floor=1.5)

    def test_floor_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence_floor"):
            RCAConfig(evidence_floor=0.0)

    def test_contributing_fraction_range(self) -> None:
        with pytest.raises(ValueError, match="contributing_fraction"):
            RCAConfig(contributing_fraction=0.0)

    def test_contributing_fraction_above_one(self) -> None:
        with pytest.raises(ValueError, match="contributing_fraction"):
            RCAConfig(contributing_fraction=1.5)

    def test_conflict_margin_range(self) -> None:
        with pytest.raises(ValueError, match="conflict_margin"):
            RCAConfig(conflict_margin=1.0)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            RCAConfig(evidence_weight=-0.1)

    def test_zero_weight_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to > 0"):
            RCAConfig(evidence_weight=0.0, separation_weight=0.0)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            RCAConfig(top_n=0)

    def test_weight_sum_property(self) -> None:
        assert RCAConfig(evidence_weight=0.6,
                         separation_weight=0.4).weight_sum == pytest.approx(1.0)

    def test_frozen(self) -> None:
        c = RCAConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.evidence_floor = 0.5  # type: ignore[misc]

    def test_custom_floor_ok(self) -> None:
        assert RCAConfig(evidence_floor=0.25).evidence_floor == 0.25


# ===========================================================================
# AssetEvidence
# ===========================================================================


class TestAssetEvidence:
    def test_defaults_zero(self) -> None:
        e = _ev("A")
        assert e.vibration == 0.0 and e.temperature == 0.0

    def test_empty_asset_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetEvidence(asset_id="")

    def test_whitespace_asset_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="asset_id"):
            AssetEvidence(asset_id="   ")

    def test_indicator_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="vibration"):
            AssetEvidence(asset_id="A", vibration=1.5)

    def test_indicator_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            AssetEvidence(asset_id="A", temperature=-0.1)

    def test_indicator_boundaries_ok(self) -> None:
        AssetEvidence(asset_id="A", vibration=0.0, load=1.0)

    def test_indicators_dict(self) -> None:
        e = _ev("A", vibration=0.5)
        ind = e.indicators()
        assert ind[CauseCategory.VIBRATION] == 0.5
        assert len(ind) == 8

    def test_indicators_excludes_unknown(self) -> None:
        e = _ev("A")
        assert CauseCategory.UNKNOWN not in e.indicators()

    def test_to_dict(self) -> None:
        d = _ev("A", vibration=0.5).to_dict()
        assert d["asset_id"] == "A" and d["vibration"] == 0.5

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(_ev("A", load=0.3).to_dict()), str)

    def test_frozen(self) -> None:
        e = _ev("A")
        with pytest.raises((AttributeError, TypeError)):
            e.vibration = 0.9  # type: ignore[misc]

    def test_all_eight_indicators(self) -> None:
        e = AssetEvidence(asset_id="A", temperature=0.1, vibration=0.2,
                          pressure=0.3, load=0.4, lubrication=0.5,
                          electrical=0.6, environmental=0.7, operational=0.8)
        assert e.indicators()[CauseCategory.OPERATIONAL] == 0.8


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert AGENT_NAME in RCA_AGENT_REGISTRY
        assert AGENT_NAME in list_root_cause_analysis_agents()

    def test_build(self) -> None:
        assert isinstance(build_root_cause_analysis_agent(AGENT_NAME),
                          RootCauseAnalysisAgent)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown root cause"):
            build_root_cause_analysis_agent("nope")

    def test_registry_name(self) -> None:
        assert RootCauseAnalysisAgent._registry_name == AGENT_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_root_cause_analysis_agent(AGENT_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        a = build_root_cause_analysis_agent(
            AGENT_NAME, config=RCAConfig(top_n=3))
        assert a.config.top_n == 3


# ===========================================================================
# Cause scoring
# ===========================================================================


class TestCauseScoring:
    def test_eight_scores(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5))
        assert len(sc) == 8

    def test_all_categories_scored(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5))
        causes = {s.cause for s in sc}
        assert causes == {c.value for c in CauseCategory
                          if c != CauseCategory.UNKNOWN}

    def test_score_equals_indicator(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.7))
        vib = next(s for s in sc if s.cause == "vibration")
        assert vib.cause_score == pytest.approx(0.7)

    def test_contribution_sums_100(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5, load=0.3))
        assert sum(s.contribution_percentage for s in sc) == pytest.approx(100.0)

    def test_contribution_zero_when_no_evidence(self) -> None:
        sc = _agent().score_causes(_ev("A"))
        assert all(s.contribution_percentage == 0.0 for s in sc)

    def test_scores_sorted_descending(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.8, load=0.4,
                                       temperature=0.6))
        assert all(sc[i].cause_score >= sc[i + 1].cause_score
                   for i in range(len(sc) - 1))

    def test_confidence_equals_indicator(self) -> None:
        sc = _agent().score_causes(_ev("A", load=0.6))
        ld = next(s for s in sc if s.cause == "load")
        assert ld.confidence == pytest.approx(0.6)

    def test_tie_broken_by_category_order(self) -> None:
        # temperature and vibration equal -> temperature (earlier) first
        sc = _agent().score_causes(_ev("A", temperature=0.5, vibration=0.5))
        assert sc[0].cause == "temperature"

    def test_score_to_dict(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5))
        d = sc[0].to_dict()
        for k in ("cause", "cause_score", "contribution_percentage",
                  "confidence"):
            assert k in d

    def test_invalid_evidence_raises(self) -> None:
        with pytest.raises(TypeError, match="indicators"):
            _agent().score_causes(object())


# ===========================================================================
# Single-asset analysis
# ===========================================================================


class TestSingleCause:
    def test_dominant_primary(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85, temperature=0.1))
        assert r.primary_cause == "vibration"

    def test_no_contributing_when_dominant(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85, temperature=0.05))
        assert r.contributing_causes == ()

    def test_high_confidence(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.9))
        assert r.confidence >= 0.7

    def test_not_unknown(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85))
        assert not r.is_unknown

    def test_not_conflicting(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85, temperature=0.05))
        assert not r.is_conflicting

    def test_evidence_present(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85))
        assert len(r.evidence) >= 1 and "vibration" in r.evidence[0].lower()

    def test_asset_id_preserved(self) -> None:
        r = _agent().analyze(_ev("WTG-099", vibration=0.85))
        assert r.asset_id == "WTG-099"


class TestMultiCause:
    def test_contributing_present(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6, load=0.5))
        assert len(r.contributing_causes) >= 1

    def test_primary_is_strongest(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert r.primary_cause == "vibration"

    def test_contributing_above_fraction(self) -> None:
        # lubrication 0.6 >= 0.5*0.8=0.4 -> contributing
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert "lubrication" in r.contributing_causes

    def test_weak_not_contributing(self) -> None:
        # temperature 0.2 < 0.5*0.8=0.4 -> not contributing
        r = _agent().analyze(_ev("A", vibration=0.8, temperature=0.2))
        assert "temperature" not in r.contributing_causes

    def test_multiple_contributing(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.7, load=0.6))
        assert len(r.contributing_causes) == 2


class TestConflictingCause:
    def test_flagged_conflicting(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.82, load=0.78))
        assert r.is_conflicting

    def test_lower_confidence(self) -> None:
        clear = _agent().analyze(_ev("A", vibration=0.85, temperature=0.05))
        confl = _agent().analyze(_ev("B", vibration=0.82, load=0.80))
        assert confl.confidence < clear.confidence

    def test_conflict_note_in_evidence(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.82, load=0.80))
        assert any("not clear-cut" in e for e in r.evidence)

    def test_primary_still_assigned(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.82, load=0.80))
        assert r.primary_cause in ("vibration", "load")

    def test_not_conflicting_when_separated(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.85, load=0.30))
        assert not r.is_conflicting


class TestUnknownCause:
    def test_all_below_floor(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.05, temperature=0.08))
        assert r.primary_cause == "unknown" and r.is_unknown

    def test_zero_evidence(self) -> None:
        r = _agent().analyze(_ev("A"))
        assert r.is_unknown

    def test_low_confidence(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.05))
        assert r.confidence <= 0.5

    def test_no_contributing(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.05))
        assert r.contributing_causes == ()

    def test_broad_investigation(self) -> None:
        r = _agent().analyze(_ev("A"))
        assert "broad" in r.investigation_actions[0].lower()

    def test_evidence_explains_unknown(self) -> None:
        r = _agent().analyze(_ev("A"))
        assert "below" in r.evidence[0].lower() or "insufficient" in r.evidence[0].lower()

    def test_just_below_floor(self) -> None:
        r = _agent(evidence_floor=0.2).analyze(_ev("A", vibration=0.19))
        assert r.is_unknown

    def test_just_above_floor(self) -> None:
        r = _agent(evidence_floor=0.2).analyze(_ev("A", vibration=0.21))
        assert not r.is_unknown


# ===========================================================================
# Investigation recommendations
# ===========================================================================


class TestInvestigations:
    def test_vibration_bearings(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        assert "bearings" in r.investigation_actions[0].lower()

    def test_lubrication_action(self) -> None:
        r = _agent().analyze(_ev("A", lubrication=0.8))
        assert "lubrication" in r.investigation_actions[0].lower()

    def test_load_action(self) -> None:
        r = _agent().analyze(_ev("A", load=0.8))
        assert "load" in r.investigation_actions[0].lower()

    def test_electrical_action(self) -> None:
        r = _agent().analyze(_ev("A", electrical=0.8))
        assert "electrical" in r.investigation_actions[0].lower()

    def test_operational_action(self) -> None:
        r = _agent().analyze(_ev("A", operational=0.8))
        assert ("operating procedures" in r.investigation_actions[0].lower()
                or "operating" in r.investigation_actions[0].lower())

    def test_temperature_action(self) -> None:
        r = _agent().analyze(_ev("A", temperature=0.8))
        assert "cooling" in r.investigation_actions[0].lower()

    def test_pressure_action(self) -> None:
        r = _agent().analyze(_ev("A", pressure=0.8))
        assert "pressure" in r.investigation_actions[0].lower()

    def test_environmental_action(self) -> None:
        r = _agent().analyze(_ev("A", environmental=0.8))
        assert "environmental" in r.investigation_actions[0].lower()

    def test_primary_action_first(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert "bearings" in r.investigation_actions[0].lower()

    def test_actions_for_contributing(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert len(r.investigation_actions) >= 2

    def test_actions_deduped(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6, load=0.5))
        assert len(r.investigation_actions) == len(set(r.investigation_actions))


# ===========================================================================
# Fleet RCA
# ===========================================================================


class TestFleetRCA:
    def _items(self):
        causes = ["vibration", "vibration", "vibration", "lubrication",
                  "electrical", "load", "temperature", "operational"]
        return [_ev(f"W{i:02d}", **{c: 0.8}) for i, c in enumerate(causes)]

    def test_returns_report(self) -> None:
        assert isinstance(_agent().analyze_fleet(self._items()), FleetRCAReport)

    def test_asset_count(self) -> None:
        assert _agent().analyze_fleet(self._items()).asset_count == 8

    def test_most_common(self) -> None:
        assert _agent().analyze_fleet(self._items()).most_common_cause == "vibration"

    def test_top_causes_ordered(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert all(fr.top_causes[i].count >= fr.top_causes[i + 1].count
                   for i in range(len(fr.top_causes) - 1))

    def test_top_cause_is_vibration(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert fr.top_causes[0].cause == "vibration"
        assert fr.top_causes[0].count == 3

    def test_percentage(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert fr.top_causes[0].percentage == pytest.approx(37.5)

    def test_concentration_in_unit(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert 0 < fr.cause_concentration <= 1

    def test_concentration_value(self) -> None:
        # 3/8 vibration + five singletons: 0.375^2 + 5*0.125^2
        fr = _agent().analyze_fleet(self._items())
        expected = (3 / 8) ** 2 + 5 * (1 / 8) ** 2
        assert fr.cause_concentration == pytest.approx(expected)

    def test_all_same_cause_concentration_one(self) -> None:
        items = [_ev(f"W{i}", vibration=0.8) for i in range(5)]
        fr = _agent().analyze_fleet(items)
        assert fr.cause_concentration == pytest.approx(1.0)

    def test_reports_present(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert len(fr.reports) == 8

    def test_highest_risk_with_snapshot(self) -> None:
        items = self._items()
        # Make the electrical asset by far the riskiest
        assets = [_fleet_asset(e.asset_id,
                               risk=0.95 if e.electrical > 0.5 else 0.2)
                  for e in items]
        fr = _agent().analyze_fleet(items, snapshot=_snap(assets))
        assert fr.highest_risk_cause == "electrical"

    def test_highest_risk_without_snapshot(self) -> None:
        fr = _agent().analyze_fleet(self._items())
        assert fr.highest_risk_cause in {c.value for c in CauseCategory}

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _agent().analyze_fleet([])

    def test_single_asset_fleet(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        assert fr.asset_count == 1 and fr.most_common_cause == "vibration"

    def test_unknown_assets_in_fleet(self) -> None:
        items = [_ev("A", vibration=0.8), _ev("B"), _ev("C")]
        fr = _agent().analyze_fleet(items)
        causes = {f.cause for f in fr.top_causes}
        assert "unknown" in causes


# ===========================================================================
# Executive summary
# ===========================================================================


class TestExecutiveSummary:
    def _fleet(self):
        causes = ["vibration", "vibration", "lubrication", "electrical",
                  "load", "temperature"]
        return _agent().analyze_fleet([_ev(f"W{i}", **{c: 0.8})
                                       for i, c in enumerate(causes)])

    def test_returns_summary(self) -> None:
        assert isinstance(_agent().executive_summary(self._fleet()),
                          ExecutiveRCASummary)

    def test_top_causes_limited(self) -> None:
        s = _agent(top_n=3).executive_summary(self._fleet())
        assert len(s.top_causes) <= 3

    def test_recommended_actions(self) -> None:
        s = _agent().executive_summary(self._fleet())
        assert len(s.recommended_actions) >= 1

    def test_actions_deduped(self) -> None:
        s = _agent().executive_summary(self._fleet())
        assert len(s.recommended_actions) == len(set(s.recommended_actions))

    def test_distribution_present(self) -> None:
        s = _agent().executive_summary(self._fleet())
        assert len(s.cause_distribution) >= 1

    def test_distribution_percentages(self) -> None:
        s = _agent().executive_summary(self._fleet())
        total = sum(p for _, p in s.cause_distribution)
        assert total == pytest.approx(100.0)

    def test_summary_mentions_most_common(self) -> None:
        s = _agent().executive_summary(self._fleet())
        assert "vibration" in s.summary

    def test_summary_non_empty(self) -> None:
        s = _agent().executive_summary(self._fleet())
        assert len(s.summary) > 40

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(TypeError, match="FleetRCAReport"):
            _agent().executive_summary("not a report")

    def test_top_5_default(self) -> None:
        causes = ["vibration", "lubrication", "electrical", "load",
                  "temperature", "pressure", "operational", "environmental"]
        fr = _agent().analyze_fleet([_ev(f"W{i}", **{c: 0.8})
                                     for i, c in enumerate(causes)])
        s = _agent().executive_summary(fr)
        assert len(s.top_causes) == 5


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_report_to_dict(self) -> None:
        d = _agent().analyze(_ev("A", vibration=0.8)).to_dict()
        for k in ("asset_id", "primary_cause", "contributing_causes",
                  "confidence", "evidence", "investigation_actions",
                  "cause_scores", "is_conflicting", "is_unknown"):
            assert k in d

    def test_report_json(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        assert isinstance(json.dumps(r.to_dict()), str)

    def test_cause_score_json(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5))
        assert isinstance(json.dumps(sc[0].to_dict()), str)

    def test_fleet_report_json(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        assert isinstance(json.dumps(fr.to_dict()), str)

    def test_executive_summary_json(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        s = _agent().executive_summary(fr)
        assert isinstance(json.dumps(s.to_dict()), str)

    def test_cause_frequency_json(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        assert isinstance(json.dumps(fr.top_causes[0].to_dict()), str)

    def test_nested_cause_scores_serialized(self) -> None:
        d = _agent().analyze(_ev("A", vibration=0.8)).to_dict()
        assert len(d["cause_scores"]) == 8

    def test_distribution_serialized_as_pairs(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        d = _agent().executive_summary(fr).to_dict()
        assert isinstance(d["cause_distribution"][0], list)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_analyze_deterministic(self) -> None:
        e = _ev("A", vibration=0.8, lubrication=0.5)
        r1 = _agent().analyze(e)
        r2 = _agent().analyze(e)
        assert r1.primary_cause == r2.primary_cause
        assert r1.contributing_causes == r2.contributing_causes
        assert r1.confidence == r2.confidence

    def test_scoring_deterministic(self) -> None:
        e = _ev("A", vibration=0.8, load=0.4)
        s1 = _agent().score_causes(e)
        s2 = _agent().score_causes(e)
        assert [s.cause for s in s1] == [s.cause for s in s2]

    def test_fleet_deterministic(self) -> None:
        items = [_ev(f"W{i}", vibration=0.8) for i in range(4)]
        f1 = _agent().analyze_fleet(items)
        f2 = _agent().analyze_fleet(items)
        assert f1.most_common_cause == f2.most_common_cause
        assert f1.cause_concentration == f2.cause_concentration

    def test_tie_resolution_deterministic(self) -> None:
        # Equal indicators always resolve to the same primary
        e = _ev("A", temperature=0.6, vibration=0.6, load=0.6)
        assert _agent().analyze(e).primary_cause == _agent().analyze(e).primary_cause

    def test_summary_deterministic(self) -> None:
        fr = _agent().analyze_fleet([_ev(f"W{i}", vibration=0.8) for i in range(3)])
        s1 = _agent().executive_summary(fr)
        s2 = _agent().executive_summary(fr)
        assert s1.summary == s2.summary


# ===========================================================================
# Tracker
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_analyses(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        agent = RootCauseAnalysisAgent(experiment_tracker=FakeTracker())
        agent.analyze(_ev("A", vibration=0.8))
        assert logged and "rca_analyses" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        agent = RootCauseAnalysisAgent(experiment_tracker=BrokenTracker())
        assert agent.analyze(_ev("A", vibration=0.8)) is not None

    def test_no_tracker_ok(self) -> None:
        assert _agent().analyze(_ev("A", vibration=0.8)) is not None

    def test_analysis_count_increments(self) -> None:
        agent = _agent()
        agent.analyze(_ev("A", vibration=0.8))
        agent.analyze(_ev("B", load=0.8))
        assert agent._n_analyses == 2


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_analyze_non_evidence_raises(self) -> None:
        with pytest.raises(TypeError, match="AssetEvidence"):
            _agent().analyze({"asset_id": "A"})

    def test_report_frozen(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        with pytest.raises((AttributeError, TypeError)):
            r.primary_cause = "load"  # type: ignore[misc]

    def test_cause_score_frozen(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.5))
        with pytest.raises((AttributeError, TypeError)):
            sc[0].cause_score = 0.9  # type: ignore[misc]

    def test_all_equal_evidence(self) -> None:
        e = AssetEvidence(asset_id="A", temperature=0.5, vibration=0.5,
                          pressure=0.5, load=0.5, lubrication=0.5,
                          electrical=0.5, environmental=0.5, operational=0.5)
        r = _agent().analyze(e)
        # All equal -> conflicting, primary is first by category order
        assert r.primary_cause == "temperature" and r.is_conflicting

    def test_single_max_indicator(self) -> None:
        r = _agent().analyze(_ev("A", operational=1.0))
        assert r.primary_cause == "operational" and r.confidence >= 0.7

    def test_large_fleet(self) -> None:
        items = [_ev(f"W{i:03d}",
                     **{["vibration", "load", "electrical"][i % 3]: 0.8})
                 for i in range(60)]
        fr = _agent().analyze_fleet(items)
        assert fr.asset_count == 60

    def test_fleet_all_unknown(self) -> None:
        items = [_ev(f"W{i}") for i in range(4)]
        fr = _agent().analyze_fleet(items)
        assert fr.most_common_cause == "unknown"
        assert fr.cause_concentration == pytest.approx(1.0)

    def test_n_analyses_across_fleet(self) -> None:
        agent = _agent()
        agent.analyze_fleet([_ev(f"W{i}", vibration=0.8) for i in range(5)])
        assert agent._n_analyses == 5

    def test_distinct_agents_independent(self) -> None:
        a1 = _agent()
        a2 = _agent()
        a1.analyze(_ev("A", vibration=0.8))
        assert a2._n_analyses == 0

    def test_custom_contributing_fraction(self) -> None:
        # With a low fraction, more causes qualify as contributing
        strict = _agent(contributing_fraction=0.9).analyze(
            _ev("A", vibration=0.8, load=0.5))
        loose = _agent(contributing_fraction=0.3).analyze(
            _ev("A", vibration=0.8, load=0.5))
        assert len(loose.contributing_causes) >= len(strict.contributing_causes)

    def test_custom_conflict_margin(self) -> None:
        # A wide margin flags more cases as conflicting
        wide = _agent(conflict_margin=0.5).analyze(
            _ev("A", vibration=0.8, load=0.5))
        assert wide.is_conflicting


# ===========================================================================
# Additional coverage
# ===========================================================================


class TestAdditionalScoring:
    def test_contribution_proportional(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=0.6, load=0.2))
        vib = next(s for s in sc if s.cause == "vibration")
        ld = next(s for s in sc if s.cause == "load")
        assert vib.contribution_percentage == pytest.approx(75.0)
        assert ld.contribution_percentage == pytest.approx(25.0)

    def test_confidence_clamped(self) -> None:
        sc = _agent().score_causes(_ev("A", vibration=1.0))
        assert all(0 <= s.confidence <= 1 for s in sc)

    def test_zero_evidence_all_zero_scores(self) -> None:
        sc = _agent().score_causes(_ev("A"))
        assert all(s.cause_score == 0.0 for s in sc)


class TestAdditionalAnalysis:
    def test_confidence_in_unit(self) -> None:
        for kw in ({"vibration": 0.9}, {"vibration": 0.5, "load": 0.5},
                   {"vibration": 0.05}, {}):
            r = _agent().analyze(_ev("A", **kw))
            assert 0.0 <= r.confidence <= 1.0

    def test_primary_always_valid_category(self) -> None:
        r = _agent().analyze(_ev("A", electrical=0.7))
        assert r.primary_cause in {c.value for c in CauseCategory}

    def test_contributing_excludes_primary(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert r.primary_cause not in r.contributing_causes

    def test_evidence_mentions_contribution(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        assert "%" in r.evidence[0]

    def test_higher_evidence_higher_confidence(self) -> None:
        low = _agent().analyze(_ev("A", vibration=0.3, temperature=0.05))
        high = _agent().analyze(_ev("B", vibration=0.9, temperature=0.05))
        assert high.confidence >= low.confidence


class TestAdditionalFleet:
    def test_top_causes_percentages_sum(self) -> None:
        items = [_ev(f"W{i}", **{["vibration", "load"][i % 2]: 0.8})
                 for i in range(6)]
        fr = _agent().analyze_fleet(items)
        assert sum(f.percentage for f in fr.top_causes) == pytest.approx(100.0)

    def test_total_risk_accumulates(self) -> None:
        items = [_ev("A", vibration=0.8), _ev("B", vibration=0.8)]
        assets = [_fleet_asset("A", 0.5), _fleet_asset("B", 0.7)]
        fr = _agent().analyze_fleet(items, snapshot=_snap(assets))
        vib = next(f for f in fr.top_causes if f.cause == "vibration")
        assert vib.total_risk == pytest.approx(1.2)

    def test_concentration_two_equal_causes(self) -> None:
        items = [_ev(f"W{i}", **{["vibration", "load"][i % 2]: 0.8})
                 for i in range(4)]
        fr = _agent().analyze_fleet(items)
        # two causes at 0.5 each -> 0.5
        assert fr.cause_concentration == pytest.approx(0.5)


# ===========================================================================
# Further coverage to reach the 180+ target
# ===========================================================================


class TestInvestigationMapCompleteness:
    def test_every_category_has_action(self) -> None:
        for cat in ("temperature", "vibration", "pressure", "load",
                    "lubrication", "electrical", "environmental",
                    "operational"):
            r = _agent().analyze(_ev("A", **{cat: 0.8}))
            assert len(r.investigation_actions) >= 1
            assert r.investigation_actions[0].endswith(".")

    def test_pressure_specific(self) -> None:
        r = _agent().analyze(_ev("A", pressure=0.8))
        assert "hydraulic" in r.investigation_actions[0].lower()

    def test_environmental_specific(self) -> None:
        r = _agent().analyze(_ev("A", environmental=0.8))
        assert "exposure" in r.investigation_actions[0].lower()

    def test_temperature_specific(self) -> None:
        r = _agent().analyze(_ev("A", temperature=0.8))
        assert "thermal" in r.investigation_actions[0].lower() or \
            "cooling" in r.investigation_actions[0].lower()


class TestPrimaryCauseSelection:
    def test_each_subsystem_can_be_primary(self) -> None:
        for cat in ("temperature", "vibration", "pressure", "load",
                    "lubrication", "electrical", "environmental",
                    "operational"):
            r = _agent().analyze(_ev("A", **{cat: 0.9}))
            assert r.primary_cause == cat

    def test_strongest_wins_among_many(self) -> None:
        e = AssetEvidence(asset_id="A", temperature=0.3, vibration=0.4,
                          pressure=0.5, load=0.9, lubrication=0.2)
        assert _agent().analyze(e).primary_cause == "load"

    def test_second_strongest_not_primary(self) -> None:
        e = _ev("A", vibration=0.9, load=0.6)
        assert _agent().analyze(e).primary_cause == "vibration"


class TestConfidenceBehaviour:
    def test_isolated_max_high_confidence(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.95, temperature=0.02))
        assert r.confidence > 0.8

    def test_two_equal_lower_confidence(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, load=0.8))
        assert r.confidence < 0.7

    def test_confidence_monotone_in_evidence(self) -> None:
        confs = [_agent().analyze(_ev("A", vibration=v, temperature=0.02)).confidence
                 for v in (0.3, 0.5, 0.7, 0.9)]
        assert all(confs[i] <= confs[i + 1] + 1e-9 for i in range(len(confs) - 1))

    def test_unknown_confidence_capped(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.1))
        assert r.confidence <= 0.5


class TestFleetConcentrationDetail:
    def test_three_causes_even(self) -> None:
        items = [_ev(f"W{i}", **{["vibration", "load", "electrical"][i % 3]: 0.8})
                 for i in range(6)]
        fr = _agent().analyze_fleet(items)
        # three causes at 1/3 each -> 3*(1/3)^2 = 1/3
        assert fr.cause_concentration == pytest.approx(1 / 3)

    def test_concentration_higher_when_focused(self) -> None:
        focused = _agent().analyze_fleet(
            [_ev(f"W{i}", vibration=0.8) for i in range(6)])
        spread = _agent().analyze_fleet(
            [_ev(f"W{i}", **{["vibration", "load", "electrical",
                              "temperature", "lubrication", "pressure"][i]: 0.8})
             for i in range(6)])
        assert focused.cause_concentration > spread.cause_concentration

    def test_most_common_tie_break(self) -> None:
        # vibration and load both 2 -> vibration (earlier enum order) wins
        items = [_ev("A", vibration=0.8), _ev("B", vibration=0.8),
                 _ev("C", load=0.8), _ev("D", load=0.8)]
        fr = _agent().analyze_fleet(items)
        assert fr.most_common_cause == "vibration"


class TestFleetRiskWeighting:
    def test_highest_risk_differs_from_most_common(self) -> None:
        # vibration most common, but a single electrical asset is riskiest
        items = [_ev("A", vibration=0.8), _ev("B", vibration=0.8),
                 _ev("C", electrical=0.8)]
        assets = [_fleet_asset("A", 0.1), _fleet_asset("B", 0.1),
                  _fleet_asset("C", 0.99)]
        fr = _agent().analyze_fleet(items, snapshot=_snap(assets))
        assert fr.most_common_cause == "vibration"
        assert fr.highest_risk_cause == "electrical"

    def test_risk_falls_back_to_confidence(self) -> None:
        # No snapshot -> risk proxied by confidence; should still pick a cause
        items = [_ev("A", vibration=0.9), _ev("B", load=0.4)]
        fr = _agent().analyze_fleet(items)
        assert fr.highest_risk_cause in {"vibration", "load"}


class TestExecutiveSummaryDetail:
    def test_summary_mentions_concentration(self) -> None:
        fr = _agent().analyze_fleet([_ev(f"W{i}", vibration=0.8) for i in range(4)])
        s = _agent().executive_summary(fr)
        assert "concentration" in s.summary.lower()

    def test_focused_phrase(self) -> None:
        fr = _agent().analyze_fleet([_ev(f"W{i}", vibration=0.8) for i in range(4)])
        s = _agent().executive_summary(fr)
        assert "focused" in s.summary.lower()

    def test_spread_phrase(self) -> None:
        items = [_ev(f"W{i}", **{["vibration", "load", "electrical",
                                  "temperature"][i]: 0.8}) for i in range(4)]
        fr = _agent().analyze_fleet(items)
        s = _agent().executive_summary(fr)
        assert "spread" in s.summary.lower()

    def test_actions_match_top_causes(self) -> None:
        fr = _agent().analyze_fleet(
            [_ev("A", vibration=0.8), _ev("B", lubrication=0.8)])
        s = _agent().executive_summary(fr)
        # vibration action present
        assert any("bearings" in a.lower() for a in s.recommended_actions)


class TestEvidenceStatementDetail:
    def test_primary_statement_has_percent(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        assert "%" in r.evidence[0]

    def test_contributing_statement_present(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8, lubrication=0.6))
        assert any("contributing" in e.lower() for e in r.evidence)

    def test_conflict_statement_only_when_conflicting(self) -> None:
        clear = _agent().analyze(_ev("A", vibration=0.9, temperature=0.05))
        assert not any("clear-cut" in e for e in clear.evidence)


class TestExtraEdgeCases:
    def test_one_indicator_at_floor_exactly(self) -> None:
        # At exactly the floor it counts as evidence (not < floor)
        r = _agent(evidence_floor=0.2).analyze(_ev("A", vibration=0.2))
        assert not r.is_unknown

    def test_max_single_indicator(self) -> None:
        r = _agent().analyze(_ev("A", vibration=1.0))
        assert r.primary_cause == "vibration"

    def test_report_has_eight_scores(self) -> None:
        r = _agent().analyze(_ev("A", vibration=0.8))
        assert len(r.cause_scores) == 8

    def test_fleet_reports_match_count(self) -> None:
        items = [_ev(f"W{i}", vibration=0.8) for i in range(7)]
        fr = _agent().analyze_fleet(items)
        assert len(fr.reports) == 7

    def test_executive_summary_empty_fleet_safe(self) -> None:
        # A fleet of one unknown still summarises without error
        fr = _agent().analyze_fleet([_ev("A")])
        s = _agent().executive_summary(fr)
        assert isinstance(s, ExecutiveRCASummary)

    def test_custom_top_n_fleet(self) -> None:
        causes = ["vibration", "load", "electrical", "temperature",
                  "lubrication", "pressure"]
        fr = _agent(top_n=2).analyze_fleet(
            [_ev(f"W{i}", **{c: 0.8}) for i, c in enumerate(causes)])
        s = _agent(top_n=2).executive_summary(fr)
        assert len(s.top_causes) == 2

    def test_evidence_floor_affects_unknown_boundary(self) -> None:
        # Higher floor -> more cases unknown
        low_floor = _agent(evidence_floor=0.1).analyze(_ev("A", vibration=0.15))
        high_floor = _agent(evidence_floor=0.3).analyze(_ev("A", vibration=0.15))
        assert not low_floor.is_unknown and high_floor.is_unknown

    def test_score_count_constant(self) -> None:
        for kw in ({}, {"vibration": 0.5}, {"load": 0.9, "electrical": 0.3}):
            assert len(_agent().score_causes(_ev("A", **kw))) == 8

    def test_contributing_capped_by_floor(self) -> None:
        # A weak contributing candidate below floor is excluded even if above fraction
        r = _agent(evidence_floor=0.3).analyze(_ev("A", vibration=0.5, load=0.26))
        assert "load" not in r.contributing_causes

    def test_fleet_single_unknown_concentration(self) -> None:
        fr = _agent().analyze_fleet([_ev("A")])
        assert fr.cause_concentration == pytest.approx(1.0)

    def test_analyze_returns_report_type(self) -> None:
        assert isinstance(_agent().analyze(_ev("A", vibration=0.8)),
                          RootCauseReport)

    def test_score_returns_tuple(self) -> None:
        assert isinstance(_agent().score_causes(_ev("A", vibration=0.5)), tuple)

    def test_cause_frequency_fields(self) -> None:
        fr = _agent().analyze_fleet([_ev("A", vibration=0.8)])
        f = fr.top_causes[0]
        assert hasattr(f, "cause") and hasattr(f, "count") and \
            hasattr(f, "percentage") and hasattr(f, "total_risk")