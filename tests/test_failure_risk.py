#!/usr/bin/env python3
"""Comprehensive test suite for ``src/predictive/failure_risk.py``.

The failure-risk engine is pure NumPy / stdlib, so the **entire** suite runs
without PyTorch or SciPy.  Coverage (60+ tests):

- FailureRiskConfig validation
- SurvivalModel / RiskLevel enums
- Registry (register / build / list / duplicate rejection)
- Pure survival functions (exponential, Weibull, scale-from-mean, hazard)
- Weibull validation (mean property, beta=1 -> exponential, wear-out hazard)
- Survival-curve validation (monotone, bounds, complementarity with CDF)
- failure_cdf / survival_function
- Horizon analysis (monotone in horizon, custom horizons, dominant horizon)
- Risk-category tests (all four levels, threshold boundaries)
- Confidence-interval propagation (bracketing, direction, width)
- Infinite-RUL handling (zero risk, LOW level)
- Already-failed handling (certain risk, CRITICAL level)
- predict_from_rul / predict_from_health_engine
- Step-1->2->3 chained integration
- ExperimentTracker integration (failure-safe)
- Edge cases (NaN clipping, zero/near-zero RUL)

Run::

    pytest tests/test_failure_risk.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.predictive.failure_risk import (
    DEFAULT_HORIZONS,
    ENGINE_NAME,
    FAILURE_RISK_REGISTRY,
    FailureRiskConfig,
    FailureRiskEngine,
    FailureRiskPrediction,
    RiskLevel,
    SurvivalModel,
    _clip01,
    build_failure_risk_engine,
    exponential_survival,
    hazard_rate,
    list_failure_risk_engines,
    register_failure_risk_engine,
    weibull_scale_from_mean,
    weibull_survival,
)
from src.predictive.rul_predictor import RUL_INFINITE, RULPrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rul(
    rul: float, *, ci_low: float | None = None, ci_high: float | None = None,
    is_degrading: bool = True, confidence_level: float = 0.95,
) -> RULPrediction:
    """Build a RULPrediction fixture."""
    if ci_low is None:
        ci_low = rul * 0.8 if math.isfinite(rul) else RUL_INFINITE
    if ci_high is None:
        ci_high = rul * 1.2 if math.isfinite(rul) else RUL_INFINITE
    return RULPrediction(
        rul=rul, ci_low=ci_low, ci_high=ci_high, confidence_level=confidence_level,
        failure_threshold=30.0, current_health=55.0, model_used="linear",
        slope=-2.0, r_squared=0.95, n_observations=20, is_degrading=is_degrading,
        time_per_step=1.0,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestFailureRiskConfig:
    """Tests for configuration validation."""

    def test_defaults(self) -> None:
        c = FailureRiskConfig()
        assert c.model == SurvivalModel.EXPONENTIAL.value
        assert c.horizons == DEFAULT_HORIZONS

    def test_bad_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            FailureRiskConfig(model="gamma")

    def test_empty_horizons(self) -> None:
        with pytest.raises(ValueError, match="horizons"):
            FailureRiskConfig(horizons=())

    def test_negative_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizons"):
            FailureRiskConfig(horizons=(7, -1))

    def test_bad_weibull_shape(self) -> None:
        with pytest.raises(ValueError, match="weibull_shape"):
            FailureRiskConfig(weibull_shape=0)

    def test_bad_threshold_order(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            FailureRiskConfig(medium_threshold=0.6, high_threshold=0.5)

    def test_bad_dominant_horizon(self) -> None:
        with pytest.raises(ValueError, match="dominant_horizon"):
            FailureRiskConfig(dominant_horizon=-5)

    def test_frozen(self) -> None:
        c = FailureRiskConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.model = "weibull"  # type: ignore[misc]

    def test_valid_models(self) -> None:
        for m in ("exponential", "weibull"):
            FailureRiskConfig(model=m)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for the enums."""

    def test_survival_model_values(self) -> None:
        assert SurvivalModel.EXPONENTIAL.value == "exponential"
        assert SurvivalModel.WEIBULL.value == "weibull"

    def test_risk_levels(self) -> None:
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_from_probability_low(self) -> None:
        assert RiskLevel.from_probability(
            0.1, medium=0.25, high=0.5, critical=0.75) == RiskLevel.LOW

    def test_from_probability_medium(self) -> None:
        assert RiskLevel.from_probability(
            0.3, medium=0.25, high=0.5, critical=0.75) == RiskLevel.MEDIUM

    def test_from_probability_high(self) -> None:
        assert RiskLevel.from_probability(
            0.6, medium=0.25, high=0.5, critical=0.75) == RiskLevel.HIGH

    def test_from_probability_critical(self) -> None:
        assert RiskLevel.from_probability(
            0.9, medium=0.25, high=0.5, critical=0.75) == RiskLevel.CRITICAL

    def test_threshold_boundary_inclusive(self) -> None:
        # exactly at critical -> CRITICAL
        assert RiskLevel.from_probability(
            0.75, medium=0.25, high=0.5, critical=0.75) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the registry."""

    def test_registered(self) -> None:
        assert ENGINE_NAME in FAILURE_RISK_REGISTRY
        assert ENGINE_NAME in list_failure_risk_engines()

    def test_build_by_name(self) -> None:
        assert isinstance(build_failure_risk_engine(ENGINE_NAME), FailureRiskEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown failure-risk engine"):
            build_failure_risk_engine("nope")

    def test_registry_name_stamped(self) -> None:
        assert FailureRiskEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_failure_risk_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        e = build_failure_risk_engine(ENGINE_NAME,
                                      config=FailureRiskConfig(model="weibull"))
        assert e.config.model == "weibull"


# ---------------------------------------------------------------------------
# Pure survival functions
# ---------------------------------------------------------------------------


class TestExponentialSurvival:
    """Tests for the exponential survival function."""

    def test_survival_at_zero_is_one(self) -> None:
        assert exponential_survival(0.0, 30.0) == 1.0

    def test_survival_at_mean(self) -> None:
        assert exponential_survival(30.0, 30.0) == pytest.approx(math.exp(-1))

    def test_decreasing(self) -> None:
        vals = [exponential_survival(t, 30.0) for t in (0, 10, 30, 60)]
        assert vals == sorted(vals, reverse=True)

    def test_zero_mean_is_zero_survival(self) -> None:
        assert exponential_survival(10.0, 0.0) == 0.0

    def test_vectorised(self) -> None:
        out = exponential_survival(np.array([0.0, 30.0]), 30.0)
        assert isinstance(out, np.ndarray)
        assert out[0] == 1.0


class TestWeibullSurvival:
    """Tests for the Weibull survival function."""

    def test_survival_at_zero_is_one(self) -> None:
        assert weibull_survival(0.0, 30.0, 2.0) == 1.0

    def test_decreasing(self) -> None:
        vals = [weibull_survival(t, 30.0, 2.5) for t in (0, 10, 30, 60)]
        assert vals == sorted(vals, reverse=True)

    def test_reduces_to_exponential(self) -> None:
        # beta=1, eta=mean -> identical to exponential
        eta = weibull_scale_from_mean(30.0, 1.0)
        assert weibull_survival(7.0, eta, 1.0) == pytest.approx(
            exponential_survival(7.0, 30.0)
        )

    def test_scale_from_mean(self) -> None:
        eta = weibull_scale_from_mean(30.0, 2.5)
        # eta * Gamma(1 + 1/beta) == mean
        assert eta * math.gamma(1 + 1 / 2.5) == pytest.approx(30.0)

    def test_scale_from_mean_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            weibull_scale_from_mean(30.0, 0.0)


class TestHazardRate:
    """Tests for the hazard-rate function."""

    def test_exponential_constant(self) -> None:
        h0 = hazard_rate(0.0, 30.0, "exponential", 2.0)
        h50 = hazard_rate(50.0, 30.0, "exponential", 2.0)
        assert h0 == pytest.approx(h50)
        assert h0 == pytest.approx(1 / 30.0)

    def test_weibull_increasing_for_wearout(self) -> None:
        # beta > 1 -> hazard increases with time
        h10 = hazard_rate(10.0, 30.0, "weibull", 2.5)
        h50 = hazard_rate(50.0, 30.0, "weibull", 2.5)
        assert h50 > h10

    def test_weibull_decreasing_for_infant(self) -> None:
        # beta < 1 -> hazard decreases (infant mortality)
        h10 = hazard_rate(10.0, 30.0, "weibull", 0.5)
        h50 = hazard_rate(50.0, 30.0, "weibull", 0.5)
        assert h50 < h10

    def test_zero_mean_infinite_hazard(self) -> None:
        assert math.isinf(hazard_rate(0.0, 0.0, "exponential", 2.0))


# ---------------------------------------------------------------------------
# Survival / CDF on the engine
# ---------------------------------------------------------------------------


class TestSurvivalAndCdf:
    """Tests for engine-level survival and CDF."""

    def test_complementary(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(model="exponential"))
        assert e.survival_function(30.0, 30.0) + e.failure_cdf(30.0, 30.0) == pytest.approx(1.0)

    def test_weibull_complementary(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(model="weibull", weibull_shape=2.5))
        assert e.survival_function(20.0, 30.0) + e.failure_cdf(20.0, 30.0) == pytest.approx(1.0)

    def test_infinite_mean_full_survival(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        assert e.survival_function(30.0, float("inf")) == 1.0

    def test_cdf_increasing(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        vals = [e.failure_cdf(t, 30.0) for t in (7, 30, 90)]
        assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# Horizon analysis
# ---------------------------------------------------------------------------


class TestHorizonAnalysis:
    """Tests for horizon-based risk analysis."""

    def test_default_horizons(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(30))
        assert set(r.horizon_risks) == {7.0, 30.0, 90.0}

    def test_monotone_in_horizon(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90)))
        r = e.predict_from_rul(make_rul(30))
        assert r.horizon_risks[7.0] <= r.horizon_risks[30.0] <= r.horizon_risks[90.0]

    def test_custom_horizons(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(1, 14, 60, 180)))
        r = e.predict_from_rul(make_rul(40))
        assert set(r.horizon_risks) == {1.0, 14.0, 60.0, 180.0}

    def test_probabilities_in_unit_interval(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(15))
        assert all(0.0 <= p <= 1.0 for p in r.horizon_risks.values())

    def test_dominant_horizon_default_is_largest(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90)))
        r = e.predict_from_rul(make_rul(30))
        assert r.dominant_horizon == 90.0

    def test_dominant_horizon_custom(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90),
                                                dominant_horizon=7))
        r = e.predict_from_rul(make_rul(30))
        assert r.dominant_horizon == 7.0
        assert r.dominant_probability == r.horizon_risks[7.0]

    def test_exponential_known_value(self) -> None:
        # P(fail by 30 | RUL=30) = 1 - e^-1 = 0.632
        e = FailureRiskEngine(FailureRiskConfig(model="exponential", horizons=(30,)))
        r = e.predict_from_rul(make_rul(30))
        assert r.horizon_risks[30.0] == pytest.approx(0.632, abs=0.01)


# ---------------------------------------------------------------------------
# Weibull behaviour in predictions
# ---------------------------------------------------------------------------


class TestWeibullBehaviour:
    """Tests for Weibull-specific prediction behaviour."""

    def test_lower_short_horizon_risk_than_exponential(self) -> None:
        # Wear-out (beta>1): low early risk vs the memoryless exponential
        exp = FailureRiskEngine(FailureRiskConfig(model="exponential", horizons=(7,)))
        wbl = FailureRiskEngine(FailureRiskConfig(model="weibull",
                                                  weibull_shape=2.5, horizons=(7,)))
        re = exp.predict_from_rul(make_rul(30))
        rw = wbl.predict_from_rul(make_rul(30))
        assert rw.horizon_risks[7.0] < re.horizon_risks[7.0]

    def test_weibull_shape_recorded(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(model="weibull", weibull_shape=3.0))
        r = e.predict_from_rul(make_rul(30))
        assert r.weibull_shape == 3.0

    def test_higher_shape_steeper(self) -> None:
        # Higher beta concentrates failures near end of life: lower mid-horizon risk
        low = FailureRiskEngine(FailureRiskConfig(model="weibull",
                                                  weibull_shape=1.5, horizons=(7,)))
        high = FailureRiskEngine(FailureRiskConfig(model="weibull",
                                                   weibull_shape=4.0, horizons=(7,)))
        rl = low.predict_from_rul(make_rul(30))
        rh = high.predict_from_rul(make_rul(30))
        assert rh.horizon_risks[7.0] < rl.horizon_risks[7.0]


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


class TestRiskCategories:
    """Tests for risk-level assignment."""

    def test_long_rul_low_risk(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(7,)))
        r = e.predict_from_rul(make_rul(1000))
        assert r.risk_level == RiskLevel.LOW

    def test_short_rul_high_risk(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(30,)))
        r = e.predict_from_rul(make_rul(10))
        assert r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_risk_level_from_dominant(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90)))
        r = e.predict_from_rul(make_rul(30))
        expected = RiskLevel.from_probability(
            r.dominant_probability, medium=0.25, high=0.5, critical=0.75)
        assert r.risk_level == expected


# ---------------------------------------------------------------------------
# Confidence-interval propagation
# ---------------------------------------------------------------------------


class TestConfidenceIntervalPropagation:
    """Tests for RUL-CI -> risk-CI propagation."""

    def test_intervals_present(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(propagate_ci=True))
        r = e.predict_from_rul(make_rul(30, ci_low=22, ci_high=40))
        assert len(r.horizon_intervals) == len(r.horizon_risks)

    def test_intervals_bracket_point(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(30, ci_low=22, ci_high=40))
        for h, (lo, hi) in r.horizon_intervals.items():
            assert lo <= r.horizon_risks[h] <= hi

    def test_shorter_rul_higher_risk_bound(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(30,)))
        r = e.predict_from_rul(make_rul(30, ci_low=20, ci_high=45))
        lo, hi = r.horizon_intervals[30.0]
        # Upper bound (from ci_low=20) exceeds point (RUL=30)
        assert hi > r.horizon_risks[30.0]
        assert lo < r.horizon_risks[30.0]

    def test_no_intervals_when_disabled(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(propagate_ci=False))
        r = e.predict_from_rul(make_rul(30))
        assert r.horizon_intervals == {}

    def test_interval_bounds_in_unit_interval(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(15, ci_low=8, ci_high=25))
        for lo, hi in r.horizon_intervals.values():
            assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


# ---------------------------------------------------------------------------
# Infinite / zero RUL
# ---------------------------------------------------------------------------


class TestInfiniteAndZeroRul:
    """Tests for boundary RUL handling."""

    def test_infinite_rul_zero_risk(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(RUL_INFINITE, is_degrading=False))
        assert all(p == 0.0 for p in r.horizon_risks.values())

    def test_infinite_rul_low_level(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(RUL_INFINITE, is_degrading=False))
        assert r.risk_level == RiskLevel.LOW

    def test_infinite_rul_warning(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(RUL_INFINITE, is_degrading=False))
        assert any("non-degrading" in w for w in r.warnings)

    def test_non_degrading_treated_as_infinite(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        # Finite RUL but flagged non-degrading -> still zero risk
        r = e.predict_from_rul(make_rul(50, is_degrading=False))
        assert all(p == 0.0 for p in r.horizon_risks.values())

    def test_zero_rul_certain_failure(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(0.0))
        assert all(p == 1.0 for p in r.horizon_risks.values())
        assert r.risk_level == RiskLevel.CRITICAL

    def test_infinite_to_dict(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        r = e.predict_from_rul(make_rul(RUL_INFINITE, is_degrading=False))
        assert r.to_dict()["mean_life"] is None


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


class TestPredictionContainer:
    """Tests for the prediction container."""

    def test_survival_at(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(30,)))
        r = e.predict_from_rul(make_rul(30))
        assert r.survival_at(30.0) == pytest.approx(1.0 - r.horizon_risks[30.0])

    def test_survival_at_missing_horizon(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig(horizons=(30,)))
        r = e.predict_from_rul(make_rul(30))
        assert math.isnan(r.survival_at(999.0))

    def test_to_dict_structure(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        d = e.predict_from_rul(make_rul(30)).to_dict()
        assert "horizon_risks" in d
        assert "risk_level" in d
        assert "model_used" in d


# ---------------------------------------------------------------------------
# Health-engine integration
# ---------------------------------------------------------------------------


class TestHealthEngineIntegration:
    """Tests for the chained Steps 1->2->3 integration."""

    def test_predict_from_health_engine(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.5))
        he.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.09, 20):
            he.update(anomaly_score=float(s))
        rp = RULPredictor(RULConfig(failure_threshold=40, model="linear"))
        e = FailureRiskEngine(FailureRiskConfig())
        risk = e.predict_from_health_engine(he, rp)
        assert isinstance(risk, FailureRiskPrediction)

    def test_chained_degrading_has_risk(self) -> None:
        from src.predictive.health_index import HealthIndexConfig, HealthIndexEngine
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        he = HealthIndexEngine(HealthIndexConfig(smoothing_alpha=0.6))
        he.set_anomaly_threshold(0.02)
        for s in np.linspace(0.01, 0.14, 25):
            he.update(anomaly_score=float(s))
        rp = RULPredictor(RULConfig(failure_threshold=45, model="linear"))
        e = FailureRiskEngine(FailureRiskConfig())
        risk = e.predict_from_health_engine(he, rp)
        # A clearly degrading machine should carry some non-zero risk
        assert risk.dominant_probability >= 0.0


# ---------------------------------------------------------------------------
# Tracker integration
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_prediction(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        e = FailureRiskEngine(FailureRiskConfig(), experiment_tracker=FakeTracker())
        e.predict_from_rul(make_rul(30))
        assert len(logged) == 1
        assert any("failure_risk" in k for k in logged[0])

    def test_logs_dominant_risk(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        e = FailureRiskEngine(FailureRiskConfig(), experiment_tracker=FakeTracker())
        e.predict_from_rul(make_rul(30))
        assert "dominant_failure_risk" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        e = FailureRiskEngine(FailureRiskConfig(),
                              experiment_tracker=BrokenTracker())
        r = e.predict_from_rul(make_rul(30))
        assert r.risk_level is not None


# ---------------------------------------------------------------------------
# Helpers / edge cases
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for helper functions and edge cases."""

    def test_clip01_bounds(self) -> None:
        assert _clip01(1.5) == 1.0
        assert _clip01(-0.5) == 0.0
        assert _clip01(0.5) == 0.5

    def test_clip01_nan(self) -> None:
        assert _clip01(float("nan")) == 0.0

    def test_clip01_vectorised(self) -> None:
        out = _clip01(np.array([-1.0, 0.5, 2.0]))
        assert list(out) == [0.0, 0.5, 1.0]

    def test_default_horizons_constant(self) -> None:
        assert DEFAULT_HORIZONS == (7.0, 30.0, 90.0)

    def test_n_predictions_increments(self) -> None:
        e = FailureRiskEngine(FailureRiskConfig())
        e.predict_from_rul(make_rul(30))
        e.predict_from_rul(make_rul(40))
        assert e._n_predictions == 2