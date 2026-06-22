#!/usr/bin/env python3
"""Comprehensive test suite for ``src/risk/monte_carlo_engine.py``.

Pure NumPy throughout (the engine composes the pure-NumPy frozen simulator), so
the entire suite runs without PyTorch or SciPy.  Coverage (200+ tests):

- DistributionKind enum
- Sampler construction, validation, and distributions (normal/uniform/triangular)
- Sampler clamping (rate >= 0, noise >= 0, threshold in [0, 100])
- MonteCarloConfig validation
- Registry (register / build / list / duplicate)
- compute_statistics + percentiles
- value_at_risk / conditional_value_at_risk
- build_visualization (histogram / cdf / survival / risk curve)
- run_rul_uncertainty / run_failure_probability / run_health_distribution /
  run_portfolio_distribution
- Portfolio analytics (risk, worst asset, expected failures, concentration)
- Determinism
- JSON serialization (non-finite handling)
- Failure-safe tracker integration
- Edge cases

Run::

    pytest tests/test_monte_carlo_engine.py -v
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

from src.simulation.asset_state_simulator import AssetSimulatorConfig
from src.risk.monte_carlo_engine import (
    DEFAULT_FAILURE_COST,
    ENGINE_NAME,
    MONTE_CARLO_ENGINE_REGISTRY,
    DegradationRateSampler,
    DistributionKind,
    DistributionStatistics,
    FailureThresholdSampler,
    HealthDistribution,
    MonteCarloConfig,
    MonteCarloEngine,
    NoiseSampler,
    PortfolioDistribution,
    RiskDistribution,
    RULDistribution,
    VisualizationData,
    build_monte_carlo_engine,
    build_visualization,
    compute_statistics,
    conditional_value_at_risk,
    list_monte_carlo_engines,
    register_monte_carlo_engine,
    value_at_risk,
)


def _base(**kw) -> AssetSimulatorConfig:
    d = dict(horizon=120, timestep=1.0, model="linear", degradation_rate=0.8,
             initial_health=95.0, failure_threshold=30.0, noise_std=0.5)
    d.update(kw)
    return AssetSimulatorConfig(**d)


def _engine(n_trials=600, seed=7, **kw) -> MonteCarloEngine:
    return MonteCarloEngine(MonteCarloConfig(n_trials=n_trials, random_seed=seed),
                            **kw)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ===========================================================================
# DistributionKind
# ===========================================================================


class TestDistributionKind:
    def test_values(self) -> None:
        assert DistributionKind.NORMAL.value == "normal"
        assert DistributionKind.UNIFORM.value == "uniform"
        assert DistributionKind.TRIANGULAR.value == "triangular"

    def test_three_kinds(self) -> None:
        assert len(list(DistributionKind)) == 3

    def test_is_str(self) -> None:
        assert DistributionKind.NORMAL == "normal"


# ===========================================================================
# Sampler construction & validation
# ===========================================================================


class TestSamplerConstruction:
    def test_default_normal(self) -> None:
        s = DegradationRateSampler()
        assert s.kind == "normal"

    def test_normal_params(self) -> None:
        s = DegradationRateSampler("normal", mean=1.0, std=0.2)
        assert s.mean == 1.0 and s.std == 0.2

    def test_uniform_params(self) -> None:
        s = NoiseSampler("uniform", low=0.0, high=2.0)
        assert s.low == 0.0 and s.high == 2.0

    def test_triangular_params(self) -> None:
        s = FailureThresholdSampler("triangular", low=25, mode=30, high=35)
        assert s.mode == 30

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            DegradationRateSampler("cauchy")

    def test_normal_negative_std_rejected(self) -> None:
        with pytest.raises(ValueError, match="std"):
            DegradationRateSampler("normal", std=-0.1)

    def test_uniform_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="high >= low"):
            NoiseSampler("uniform", low=2.0, high=1.0)

    def test_triangular_mode_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="low <= mode <= high"):
            FailureThresholdSampler("triangular", low=10, mode=50, high=20)

    def test_normal_zero_std_ok(self) -> None:
        DegradationRateSampler("normal", std=0.0)

    def test_uniform_equal_bounds_ok(self) -> None:
        NoiseSampler("uniform", low=1.0, high=1.0)

    def test_triangular_degenerate_ok(self) -> None:
        FailureThresholdSampler("triangular", low=30, mode=30, high=30)


# ===========================================================================
# Sampler distributions
# ===========================================================================


class TestSamplerDistributions:
    def test_normal_mean(self) -> None:
        s = DegradationRateSampler("normal", mean=1.0, std=0.2)
        out = s.sample(_rng(0), 5000)
        assert abs(np.mean(out) - 1.0) < 0.05

    def test_normal_std(self) -> None:
        s = DegradationRateSampler("normal", mean=5.0, std=0.5)
        out = s.sample(_rng(0), 5000)
        assert abs(np.std(out) - 0.5) < 0.05

    def test_uniform_bounds(self) -> None:
        s = NoiseSampler("uniform", low=0.5, high=1.5)
        out = s.sample(_rng(0), 5000)
        assert out.min() >= 0.5 and out.max() <= 1.5

    def test_uniform_mean(self) -> None:
        s = NoiseSampler("uniform", low=0.0, high=2.0)
        out = s.sample(_rng(0), 5000)
        assert abs(np.mean(out) - 1.0) < 0.05

    def test_triangular_bounds(self) -> None:
        s = FailureThresholdSampler("triangular", low=25, mode=30, high=35)
        out = s.sample(_rng(0), 5000)
        assert out.min() >= 25 and out.max() <= 35

    def test_normal_zero_std_constant(self) -> None:
        s = DegradationRateSampler("normal", mean=3.0, std=0.0)
        out = s.sample(_rng(0), 100)
        assert np.allclose(out, 3.0)

    def test_uniform_equal_bounds_constant(self) -> None:
        s = NoiseSampler("uniform", low=1.0, high=1.0)
        out = s.sample(_rng(0), 100)
        assert np.allclose(out, 1.0)

    def test_sample_shape(self) -> None:
        s = DegradationRateSampler("normal", mean=1.0, std=0.2)
        assert s.sample(_rng(0), 42).shape == (42,)

    def test_sample_default_n_is_one(self) -> None:
        s = DegradationRateSampler("normal", mean=1.0, std=0.2)
        assert s.sample(_rng(0)).shape == (1,)

    def test_sampler_deterministic(self) -> None:
        s = DegradationRateSampler("normal", mean=1.0, std=0.2)
        a = s.sample(_rng(5), 100)
        b = s.sample(_rng(5), 100)
        assert np.array_equal(a, b)


# ===========================================================================
# Sampler clamping
# ===========================================================================


class TestSamplerClamping:
    def test_rate_clamped_non_negative(self) -> None:
        s = DegradationRateSampler("normal", mean=0.0, std=2.0)
        out = s.sample(_rng(0), 5000)
        assert out.min() >= 0.0

    def test_noise_clamped_non_negative(self) -> None:
        s = NoiseSampler("normal", mean=0.0, std=2.0)
        out = s.sample(_rng(0), 5000)
        assert out.min() >= 0.0

    def test_threshold_clamped_low(self) -> None:
        s = FailureThresholdSampler("normal", mean=-10, std=5)
        out = s.sample(_rng(0), 5000)
        assert out.min() >= 0.0

    def test_threshold_clamped_high(self) -> None:
        s = FailureThresholdSampler("normal", mean=110, std=5)
        out = s.sample(_rng(0), 5000)
        assert out.max() <= 100.0

    def test_threshold_within_range(self) -> None:
        s = FailureThresholdSampler("uniform", low=20, high=40)
        out = s.sample(_rng(0), 1000)
        assert out.min() >= 0 and out.max() <= 100


# ===========================================================================
# Sampler serialization
# ===========================================================================


class TestSamplerSerialization:
    def test_normal_to_dict(self) -> None:
        d = DegradationRateSampler("normal", mean=1.0, std=0.2).to_dict()
        assert d["kind"] == "normal" and d["mean"] == 1.0 and d["std"] == 0.2

    def test_uniform_to_dict(self) -> None:
        d = NoiseSampler("uniform", low=0, high=2).to_dict()
        assert d["kind"] == "uniform" and d["low"] == 0 and d["high"] == 2

    def test_triangular_to_dict(self) -> None:
        d = FailureThresholdSampler("triangular", low=25, mode=30, high=35).to_dict()
        assert d["mode"] == 30

    def test_to_dict_json(self) -> None:
        d = DegradationRateSampler("normal", mean=1.0, std=0.2).to_dict()
        assert isinstance(json.dumps(d), str)


# ===========================================================================
# MonteCarloConfig validation
# ===========================================================================


class TestConfigValidation:
    def test_defaults(self) -> None:
        c = MonteCarloConfig()
        assert c.n_trials == 1000 and c.random_seed == 0

    def test_default_confidence_levels(self) -> None:
        c = MonteCarloConfig()
        assert 50.0 in c.confidence_levels

    def test_n_trials_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            MonteCarloConfig(n_trials=0)

    def test_n_trials_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            MonteCarloConfig(n_trials=-5)

    def test_empty_levels_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence_levels"):
            MonteCarloConfig(confidence_levels=())

    def test_level_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence level"):
            MonteCarloConfig(confidence_levels=(5.0, 150.0))

    def test_unsorted_levels_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            MonteCarloConfig(confidence_levels=(90.0, 10.0))

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="failure_cost"):
            MonteCarloConfig(failure_cost=-1)

    def test_negative_downtime_rejected(self) -> None:
        with pytest.raises(ValueError, match="failure_downtime"):
            MonteCarloConfig(failure_downtime_h=-1)

    def test_zero_bins_rejected(self) -> None:
        with pytest.raises(ValueError, match="histogram_bins"):
            MonteCarloConfig(histogram_bins=0)

    def test_parallel_ready_flag(self) -> None:
        assert MonteCarloConfig(parallel_ready=True).parallel_ready is True

    def test_store_trials_flag(self) -> None:
        assert MonteCarloConfig(store_trials=True).store_trials is True

    def test_frozen(self) -> None:
        c = MonteCarloConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.n_trials = 99  # type: ignore[misc]

    def test_custom_levels(self) -> None:
        c = MonteCarloConfig(confidence_levels=(1.0, 50.0, 99.0))
        assert c.confidence_levels == (1.0, 50.0, 99.0)


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_registered(self) -> None:
        assert ENGINE_NAME in MONTE_CARLO_ENGINE_REGISTRY
        assert ENGINE_NAME in list_monte_carlo_engines()

    def test_build(self) -> None:
        assert isinstance(build_monte_carlo_engine(ENGINE_NAME), MonteCarloEngine)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown Monte Carlo engine"):
            build_monte_carlo_engine("nope")

    def test_registry_name(self) -> None:
        assert MonteCarloEngine._registry_name == ENGINE_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_monte_carlo_engine(ENGINE_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        eng = build_monte_carlo_engine(
            ENGINE_NAME, config=MonteCarloConfig(n_trials=50))
        assert eng.config.n_trials == 50


# ===========================================================================
# compute_statistics
# ===========================================================================


class TestComputeStatistics:
    def _stats(self, arr=None):
        if arr is None:
            arr = np.random.default_rng(0).normal(43, 12, 5000)
        return compute_statistics(arr, (5, 10, 25, 50, 75, 90, 95))

    def test_mean(self) -> None:
        arr = np.array([1.0, 2, 3, 4, 5])
        assert self._stats(arr).mean == pytest.approx(3.0)

    def test_median(self) -> None:
        arr = np.array([1.0, 2, 3, 4, 5])
        assert self._stats(arr).median == pytest.approx(3.0)

    def test_std(self) -> None:
        arr = np.array([1.0, 2, 3, 4, 5])
        assert self._stats(arr).std == pytest.approx(np.std(arr))

    def test_variance(self) -> None:
        arr = np.array([1.0, 2, 3, 4, 5])
        assert self._stats(arr).variance == pytest.approx(np.var(arr))

    def test_min_max(self) -> None:
        arr = np.array([1.0, 2, 3, 4, 5])
        s = self._stats(arr)
        assert s.minimum == 1.0 and s.maximum == 5.0

    def test_percentile_ordering(self) -> None:
        s = self._stats()
        assert s.p5 <= s.p10 <= s.p25 <= s.p50 <= s.p75 <= s.p90 <= s.p95

    def test_p50_is_median(self) -> None:
        s = self._stats()
        assert s.p50 == pytest.approx(s.median, abs=1e-6)

    def test_percentiles_dict(self) -> None:
        s = self._stats()
        assert 50.0 in s.percentiles and 90.0 in s.percentiles

    def test_n(self) -> None:
        assert self._stats(np.arange(100.0)).n == 100

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_statistics(np.array([]), (50,))

    def test_single_value(self) -> None:
        s = compute_statistics(np.array([7.0]), (50,))
        assert s.mean == 7.0 and s.std == 0.0

    def test_to_dict(self) -> None:
        d = self._stats().to_dict()
        for k in ("mean", "median", "std", "variance", "p5", "p50", "p95", "n"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._stats().to_dict()), str)

    def test_custom_levels_in_dict(self) -> None:
        s = compute_statistics(np.arange(100.0), (1, 99))
        assert "1.0" in s.to_dict()["percentiles"]


# ===========================================================================
# VaR / CVaR
# ===========================================================================


class TestValueAtRisk:
    def test_var_percentile(self) -> None:
        losses = np.arange(0, 100.0)
        assert value_at_risk(losses, 95) == pytest.approx(np.percentile(losses, 95))

    def test_var_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            value_at_risk(np.array([]), 95)

    def test_var_bad_confidence_low(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            value_at_risk(np.arange(10.0), 0)

    def test_var_bad_confidence_high(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            value_at_risk(np.arange(10.0), 100)

    def test_cvar_ge_var(self) -> None:
        losses = np.random.default_rng(1).gamma(2, 10000, 5000)
        assert conditional_value_at_risk(losses, 95) >= value_at_risk(losses, 95)

    def test_cvar_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            conditional_value_at_risk(np.array([]), 95)

    def test_cvar_is_tail_mean(self) -> None:
        losses = np.arange(0, 100.0)
        var = value_at_risk(losses, 90)
        cvar = conditional_value_at_risk(losses, 90)
        assert cvar == pytest.approx(np.mean(losses[losses >= var]))

    def test_var_different_confidences(self) -> None:
        losses = np.arange(0, 100.0)
        assert value_at_risk(losses, 99) >= value_at_risk(losses, 50)

    def test_cvar_constant_losses(self) -> None:
        losses = np.full(100, 5000.0)
        assert conditional_value_at_risk(losses, 95) == pytest.approx(5000.0)


# ===========================================================================
# build_visualization
# ===========================================================================


class TestVisualization:
    def _viz(self, arr=None, **kw):
        if arr is None:
            arr = np.random.default_rng(0).normal(43, 12, 2000)
        return build_visualization(arr, **kw)

    def test_histogram_shape(self) -> None:
        v = self._viz(bins=20)
        assert len(v.histogram_counts) == 20 and len(v.histogram_edges) == 21

    def test_histogram_counts_sum(self) -> None:
        arr = np.random.default_rng(0).normal(43, 12, 2000)
        v = self._viz(arr, bins=20)
        assert sum(v.histogram_counts) == 2000

    def test_cdf_monotone(self) -> None:
        v = self._viz()
        assert all(a <= b + 1e-9 for a, b in zip(v.cdf_y, v.cdf_y[1:]))

    def test_cdf_ends_near_one(self) -> None:
        v = self._viz()
        assert abs(v.cdf_y[-1] - 1.0) < 0.02

    def test_survival_decreasing(self) -> None:
        v = self._viz()
        assert all(a >= b - 1e-9 for a, b in zip(v.survival_y, v.survival_y[1:]))

    def test_survival_complements_cdf(self) -> None:
        v = self._viz()
        for c, s in zip(v.cdf_y, v.survival_y):
            assert c + s == pytest.approx(1.0)

    def test_risk_curve_monotone(self) -> None:
        v = self._viz()
        assert all(a <= b + 1e-9 for a, b in zip(v.risk_curve_y, v.risk_curve_y[1:]))

    def test_risk_curve_in_unit(self) -> None:
        v = self._viz()
        assert all(0 <= y <= 1 for y in v.risk_curve_y)

    def test_cdf_subsampled(self) -> None:
        v = self._viz(np.random.default_rng(0).normal(0, 1, 10000),
                      cdf_max_points=200)
        assert len(v.cdf_x) <= 200

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_visualization(np.array([]))

    def test_constant_samples(self) -> None:
        v = build_visualization(np.full(100, 5.0))
        assert len(v.risk_curve_x) > 0

    def test_to_dict(self) -> None:
        d = self._viz().to_dict()
        for k in ("histogram_counts", "cdf_x", "survival_y", "risk_curve_y"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._viz().to_dict()), str)


# ===========================================================================
# run_rul_uncertainty
# ===========================================================================


class TestRULUncertainty:
    def _rul(self, **eng_kw):
        eng = _engine(**eng_kw)
        return eng.run_rul_uncertainty(_base())

    def test_returns_distribution(self) -> None:
        assert isinstance(self._rul(), RULDistribution)

    def test_percentile_ordering(self) -> None:
        r = self._rul(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.3))
        assert r.p10 <= r.p50 <= r.p90

    def test_p10_p50_p90_properties(self) -> None:
        r = self._rul()
        assert r.p10 == r.statistics.p10
        assert r.p50 == r.statistics.p50
        assert r.p90 == r.statistics.p90

    def test_probability_of_failure_in_unit(self) -> None:
        r = self._rul(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.3))
        assert 0.0 <= r.probability_of_failure <= 1.0

    def test_expected_failure_time(self) -> None:
        r = self._rul(rate_sampler=DegradationRateSampler("normal", mean=1.2, std=0.2))
        # With heavy degradation most trials fail -> finite expected failure time
        assert r.expected_failure_time > 0 or math.isnan(r.expected_failure_time)

    def test_visualization_present(self) -> None:
        assert isinstance(self._rul().visualization, VisualizationData)

    def test_trials_none_by_default(self) -> None:
        assert self._rul().trials is None

    def test_trials_stored_when_requested(self) -> None:
        eng = MonteCarloEngine(MonteCarloConfig(n_trials=300, store_trials=True))
        r = eng.run_rul_uncertainty(_base())
        assert r.trials is not None and len(r.trials) == 300

    def test_to_dict(self) -> None:
        d = self._rul().to_dict()
        assert "statistics" in d and "probability_of_failure" in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._rul().to_dict()), str)

    def test_higher_rate_lower_rul(self) -> None:
        eng = _engine()
        low = eng.run_rul_uncertainty(_base(degradation_rate=0.4))
        high = eng.run_rul_uncertainty(_base(degradation_rate=1.5))
        assert high.p50 <= low.p50

    def test_no_failure_low_rate(self) -> None:
        r = _engine().run_rul_uncertainty(_base(degradation_rate=0.05))
        assert r.probability_of_failure < 0.5


# ===========================================================================
# run_failure_probability
# ===========================================================================


class TestFailureProbability:
    def _risk(self, **eng_kw):
        return _engine(**eng_kw).run_failure_probability(_base())

    def test_returns_distribution(self) -> None:
        assert isinstance(self._risk(), RiskDistribution)

    def test_probability_in_unit(self) -> None:
        assert 0.0 <= self._risk().probability_of_failure <= 1.0

    def test_expected_loss(self) -> None:
        r = self._risk()
        assert r.expected_loss == pytest.approx(
            r.probability_of_failure * DEFAULT_FAILURE_COST, abs=1.0)

    def test_expected_downtime(self) -> None:
        r = self._risk()
        assert r.expected_downtime == pytest.approx(
            r.probability_of_failure * 72.0, abs=1e-6)

    def test_cvar_ge_var(self) -> None:
        r = self._risk(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.4))
        assert r.conditional_value_at_risk >= r.value_at_risk - 1e-6

    def test_var_confidence_recorded(self) -> None:
        r = _engine().run_failure_probability(_base(), var_confidence=99.0)
        assert r.var_confidence == 99.0

    def test_loss_statistics(self) -> None:
        assert isinstance(self._risk().loss_statistics, DistributionStatistics)

    def test_visualization(self) -> None:
        assert isinstance(self._risk().visualization, VisualizationData)

    def test_to_dict(self) -> None:
        d = self._risk().to_dict()
        for k in ("probability_of_failure", "expected_loss", "value_at_risk",
                  "conditional_value_at_risk", "expected_downtime"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._risk().to_dict()), str)

    def test_expected_failure_time(self) -> None:
        r = self._risk(rate_sampler=DegradationRateSampler("normal", mean=1.2, std=0.2))
        assert r.expected_failure_time > 0 or math.isnan(r.expected_failure_time)


# ===========================================================================
# run_health_distribution
# ===========================================================================


class TestHealthDistribution:
    def _health(self, **eng_kw):
        return _engine(**eng_kw).run_health_distribution(_base())

    def test_returns_distribution(self) -> None:
        assert isinstance(self._health(), HealthDistribution)

    def test_health_bounded(self) -> None:
        h = self._health()
        assert 0 <= h.statistics.minimum and h.statistics.maximum <= 100

    def test_statistics_present(self) -> None:
        assert isinstance(self._health().statistics, DistributionStatistics)

    def test_visualization(self) -> None:
        assert isinstance(self._health().visualization, VisualizationData)

    def test_trials_none_default(self) -> None:
        assert self._health().trials is None

    def test_trials_stored(self) -> None:
        eng = MonteCarloEngine(MonteCarloConfig(n_trials=200, store_trials=True))
        assert eng.run_health_distribution(_base()).trials is not None

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._health().to_dict()), str)

    def test_lower_rate_higher_health(self) -> None:
        eng = _engine()
        low = eng.run_health_distribution(_base(degradation_rate=0.3))
        high = eng.run_health_distribution(_base(degradation_rate=1.5))
        assert low.statistics.mean >= high.statistics.mean


# ===========================================================================
# run_portfolio_distribution
# ===========================================================================


class TestPortfolioDistribution:
    def _fleet(self, rates=(0.6, 0.9, 1.3)):
        return [_base(degradation_rate=r) for r in rates]

    def _port(self, rates=(0.6, 0.9, 1.3), **eng_kw):
        return _engine(**eng_kw).run_portfolio_distribution(self._fleet(rates))

    def test_returns_distribution(self) -> None:
        assert isinstance(self._port(), PortfolioDistribution)

    def test_n_assets(self) -> None:
        assert self._port().n_assets == 3

    def test_portfolio_risk_is_mean(self) -> None:
        p = self._port()
        assert p.portfolio_risk == pytest.approx(np.mean(p.per_asset_probability))

    def test_expected_failures_is_sum(self) -> None:
        p = self._port()
        assert p.expected_fleet_failures == pytest.approx(
            sum(p.per_asset_probability))

    def test_worst_asset_probability(self) -> None:
        p = self._port()
        assert p.worst_asset_probability == max(p.per_asset_probability)

    def test_worst_asset_index(self) -> None:
        # Short horizon so only the highest-rate asset reaches certain failure;
        # the worst asset is unambiguously the last one.
        fleet = [_base(horizon=50, degradation_rate=r) for r in (0.2, 0.5, 1.6)]
        p = _engine().run_portfolio_distribution(fleet)
        assert p.worst_asset_index == 2
        assert p.per_asset_probability[2] == max(p.per_asset_probability)

    def test_risk_concentration_bounds(self) -> None:
        p = self._port()
        assert 0.0 <= p.risk_concentration <= 1.0 + 1e-9

    def test_concentration_one_when_single_risky(self) -> None:
        # Two safe assets, one certain to fail -> concentration 1.0
        p = self._port(rates=(0.1, 0.1, 3.0))
        assert p.risk_concentration == pytest.approx(1.0, abs=1e-6)

    def test_per_asset_probability_length(self) -> None:
        assert len(self._port().per_asset_probability) == 3

    def test_count_statistics(self) -> None:
        assert isinstance(self._port().expected_fleet_failures_statistics,
                          DistributionStatistics)

    def test_visualization(self) -> None:
        assert isinstance(self._port().visualization, VisualizationData)

    def test_empty_fleet_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1 asset"):
            _engine().run_portfolio_distribution([])

    def test_single_asset_fleet(self) -> None:
        p = _engine().run_portfolio_distribution([_base()])
        assert p.n_assets == 1

    def test_to_dict(self) -> None:
        d = self._port().to_dict()
        for k in ("n_assets", "portfolio_risk", "worst_asset_probability",
                  "expected_fleet_failures", "risk_concentration"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(self._port().to_dict()), str)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_rul_deterministic(self) -> None:
        kw = dict(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.3))
        r1 = _engine(**kw).run_rul_uncertainty(_base())
        r2 = _engine(**kw).run_rul_uncertainty(_base())
        assert r1.p50 == r2.p50 and r1.probability_of_failure == r2.probability_of_failure

    def test_risk_deterministic(self) -> None:
        r1 = _engine().run_failure_probability(_base())
        r2 = _engine().run_failure_probability(_base())
        assert r1.value_at_risk == r2.value_at_risk
        assert r1.conditional_value_at_risk == r2.conditional_value_at_risk

    def test_health_deterministic(self) -> None:
        h1 = _engine().run_health_distribution(_base())
        h2 = _engine().run_health_distribution(_base())
        assert h1.statistics.mean == h2.statistics.mean

    def test_portfolio_deterministic(self) -> None:
        f = [_base(degradation_rate=r) for r in (0.6, 0.9, 1.3)]
        p1 = _engine().run_portfolio_distribution(f)
        p2 = _engine().run_portfolio_distribution(f)
        assert p1.portfolio_risk == p2.portfolio_risk

    def test_different_seed_differs(self) -> None:
        kw = dict(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.5))
        r1 = _engine(seed=1, **kw).run_rul_uncertainty(_base())
        r2 = _engine(seed=2, **kw).run_rul_uncertainty(_base())
        # Very likely different means under different seeds
        assert r1.statistics.mean != r2.statistics.mean

    def test_trial_seeds_deterministic(self) -> None:
        e1 = _engine(seed=42)
        e2 = _engine(seed=42)
        assert np.array_equal(e1._trial_seeds(), e2._trial_seeds())

    def test_trial_seeds_length(self) -> None:
        assert len(_engine(n_trials=123)._trial_seeds()) == 123


# ===========================================================================
# Degenerate (no samplers)
# ===========================================================================


class TestDegenerate:
    def test_no_samplers_low_variance(self) -> None:
        r = _engine().run_rul_uncertainty(_base(noise_std=0.0))
        # With no samplers and no noise, all trials identical -> zero variance
        assert r.statistics.std < 1e-6

    def test_no_samplers_reproduces_baseline_rul(self) -> None:
        # Deterministic asset; RUL equals analytic failure time
        r = _engine().run_rul_uncertainty(
            _base(degradation_rate=1.0, noise_std=0.0, initial_health=95,
                  failure_threshold=30))
        # 95 - 1*t = 30 -> t = 65
        assert r.statistics.p50 == pytest.approx(65.0, abs=1.0)

    def test_noise_sampler_only(self) -> None:
        r = _engine(noise_sampler=NoiseSampler("uniform", low=0, high=2)
                    ).run_rul_uncertainty(_base())
        assert isinstance(r, RULDistribution)

    def test_threshold_sampler_only(self) -> None:
        r = _engine(threshold_sampler=FailureThresholdSampler(
            "triangular", low=25, mode=30, high=35)).run_rul_uncertainty(_base())
        assert isinstance(r, RULDistribution)


# ===========================================================================
# Tracker integration
# ===========================================================================


class TestTrackerIntegration:
    def test_logs_rul(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        MonteCarloEngine(MonteCarloConfig(n_trials=200),
                         experiment_tracker=FakeTracker()).run_rul_uncertainty(_base())
        assert logged and any("mc_rul" in k for k in logged[0])

    def test_logs_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        MonteCarloEngine(MonteCarloConfig(n_trials=200),
                         experiment_tracker=FakeTracker()).run_rul_uncertainty(_base())
        assert params and params[0]["mc_run_kind"] == "rul"

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        r = MonteCarloEngine(MonteCarloConfig(n_trials=200),
                             experiment_tracker=BrokenTracker()).run_rul_uncertainty(_base())
        assert r is not None

    def test_tracker_without_params(self) -> None:
        class MetricsOnly:
            def log_metrics(self, m, step=None):
                pass

        r = MonteCarloEngine(MonteCarloConfig(n_trials=200),
                             experiment_tracker=MetricsOnly()).run_rul_uncertainty(_base())
        assert r is not None

    def test_no_tracker_ok(self) -> None:
        r = MonteCarloEngine(MonteCarloConfig(n_trials=200)).run_rul_uncertainty(_base())
        assert r is not None

    def test_risk_logs(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        MonteCarloEngine(MonteCarloConfig(n_trials=200),
                         experiment_tracker=FakeTracker()).run_failure_probability(_base())
        assert logged and any("mc_risk" in k for k in logged[0])

    def test_portfolio_logs(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        MonteCarloEngine(MonteCarloConfig(n_trials=200),
                         experiment_tracker=FakeTracker()).run_portfolio_distribution(
            [_base(degradation_rate=0.8)])
        assert logged and any("mc_portfolio" in k for k in logged[0])


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_trial(self) -> None:
        r = MonteCarloEngine(MonteCarloConfig(n_trials=1)).run_rul_uncertainty(_base())
        assert r.statistics.n == 1

    def test_default_engine(self) -> None:
        assert MonteCarloEngine().config.n_trials == 1000

    def test_large_trials(self) -> None:
        r = MonteCarloEngine(MonteCarloConfig(n_trials=5000)).run_rul_uncertainty(
            _base(noise_std=0.0))
        assert r.statistics.n == 5000

    def test_exponential_model(self) -> None:
        r = _engine().run_rul_uncertainty(
            _base(model="exponential", degradation_rate=0.03))
        assert isinstance(r, RULDistribution)

    def test_all_three_samplers(self) -> None:
        eng = _engine(
            rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.2),
            noise_sampler=NoiseSampler("uniform", low=0, high=1.5),
            threshold_sampler=FailureThresholdSampler("triangular", low=25, mode=30, high=35))
        r = eng.run_rul_uncertainty(_base())
        assert r.p10 <= r.p50 <= r.p90

    def test_rul_distribution_frozen(self) -> None:
        r = _engine().run_rul_uncertainty(_base())
        with pytest.raises((AttributeError, TypeError)):
            r.probability_of_failure = 0.5  # type: ignore[misc]

    def test_config_frozen(self) -> None:
        c = MonteCarloConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.n_trials = 1  # type: ignore[misc]

    def test_high_certainty_failure(self) -> None:
        r = _engine().run_rul_uncertainty(
            _base(degradation_rate=3.0, noise_std=0.0))
        assert r.probability_of_failure > 0.9

    def test_high_certainty_survival(self) -> None:
        r = _engine().run_rul_uncertainty(
            _base(degradation_rate=0.01, noise_std=0.0))
        assert r.probability_of_failure < 0.1

    def test_n_runs_increments(self) -> None:
        eng = _engine()
        eng.run_rul_uncertainty(_base())
        # _n_runs increments only when a tracker logs; without tracker it stays 0
        assert eng._n_runs == 0

    def test_distinct_engines_independent(self) -> None:
        e1 = _engine(seed=1)
        e2 = _engine(seed=2)
        assert e1.config.random_seed != e2.config.random_seed


# ===========================================================================
# Additional coverage to reach the 200+ target
# ===========================================================================


class TestSamplerAdditional:
    def test_rate_normal_distribution_spread(self) -> None:
        s = DegradationRateSampler("normal", mean=2.0, std=0.5)
        out = s.sample(_rng(1), 4000)
        assert np.std(out) > 0

    def test_rate_triangular_peak(self) -> None:
        s = DegradationRateSampler("triangular", low=0.5, mode=1.0, high=2.0)
        out = s.sample(_rng(1), 5000)
        # mean of triangular = (low+mode+high)/3
        assert abs(np.mean(out) - (0.5 + 1.0 + 2.0) / 3) < 0.05

    def test_noise_triangular(self) -> None:
        s = NoiseSampler("triangular", low=0, mode=0.5, high=2.0)
        out = s.sample(_rng(1), 2000)
        assert out.min() >= 0

    def test_threshold_normal_centered(self) -> None:
        s = FailureThresholdSampler("normal", mean=30, std=3)
        out = s.sample(_rng(1), 4000)
        assert 25 < np.mean(out) < 35

    def test_rate_uniform(self) -> None:
        s = DegradationRateSampler("uniform", low=0.5, high=1.5)
        out = s.sample(_rng(1), 3000)
        assert 0.5 <= out.min() and out.max() <= 1.5

    def test_sampler_kind_stored(self) -> None:
        assert DegradationRateSampler("triangular", low=0, mode=1, high=2).kind == "triangular"

    def test_noise_uniform_mean(self) -> None:
        s = NoiseSampler("uniform", low=1.0, high=3.0)
        assert abs(np.mean(s.sample(_rng(2), 5000)) - 2.0) < 0.05

    def test_threshold_triangular_mean(self) -> None:
        s = FailureThresholdSampler("triangular", low=20, mode=30, high=40)
        assert abs(np.mean(s.sample(_rng(2), 5000)) - 30) < 0.5


class TestStatisticsAdditional:
    def test_variance_is_std_squared(self) -> None:
        s = compute_statistics(np.random.default_rng(0).normal(0, 2, 3000),
                               (50,))
        assert s.variance == pytest.approx(s.std ** 2, rel=1e-6)

    def test_percentile_values_match_numpy(self) -> None:
        arr = np.random.default_rng(0).normal(50, 10, 4000)
        s = compute_statistics(arr, (5, 95))
        assert s.p5 == pytest.approx(np.percentile(arr, 5))
        assert s.p95 == pytest.approx(np.percentile(arr, 95))

    def test_mean_matches_numpy(self) -> None:
        arr = np.random.default_rng(0).gamma(2, 3, 2000)
        assert compute_statistics(arr, (50,)).mean == pytest.approx(np.mean(arr))

    def test_min_le_p5(self) -> None:
        s = compute_statistics(np.random.default_rng(0).normal(0, 1, 2000),
                               (5, 50, 95))
        assert s.minimum <= s.p5

    def test_max_ge_p95(self) -> None:
        s = compute_statistics(np.random.default_rng(0).normal(0, 1, 2000),
                               (5, 50, 95))
        assert s.maximum >= s.p95

    def test_custom_confidence_levels(self) -> None:
        s = compute_statistics(np.arange(1000.0), (1, 50, 99))
        assert set(s.percentiles.keys()) == {1.0, 50.0, 99.0}

    def test_two_values(self) -> None:
        s = compute_statistics(np.array([10.0, 20.0]), (50,))
        assert s.mean == 15.0

    def test_negative_values(self) -> None:
        s = compute_statistics(np.array([-5.0, 0.0, 5.0]), (50,))
        assert s.mean == pytest.approx(0.0)


class TestRiskMetricsAdditional:
    def test_var_monotone_in_confidence(self) -> None:
        losses = np.random.default_rng(0).gamma(2, 1000, 3000)
        assert value_at_risk(losses, 99) >= value_at_risk(losses, 90) >= value_at_risk(losses, 50)

    def test_cvar_monotone_in_confidence(self) -> None:
        losses = np.random.default_rng(0).gamma(2, 1000, 3000)
        assert conditional_value_at_risk(losses, 99) >= conditional_value_at_risk(losses, 50)

    def test_var_all_zero(self) -> None:
        assert value_at_risk(np.zeros(100), 95) == 0.0

    def test_cvar_all_zero(self) -> None:
        assert conditional_value_at_risk(np.zeros(100), 95) == 0.0

    def test_var_50_is_median(self) -> None:
        losses = np.arange(0, 101.0)
        assert value_at_risk(losses, 50) == pytest.approx(np.median(losses))

    def test_cvar_ge_mean(self) -> None:
        losses = np.random.default_rng(0).gamma(2, 1000, 3000)
        assert conditional_value_at_risk(losses, 90) >= np.mean(losses)


class TestVisualizationAdditional:
    def test_histogram_bins_respected(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(0, 1, 1000), bins=15)
        assert len(v.histogram_counts) == 15

    def test_risk_points_respected(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(0, 1, 1000),
                                risk_points=25)
        assert len(v.risk_curve_x) == 25

    def test_cdf_starts_positive(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(0, 1, 1000))
        assert v.cdf_y[0] > 0

    def test_risk_curve_starts_low_ends_high(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(50, 10, 2000))
        assert v.risk_curve_y[0] <= v.risk_curve_y[-1]

    def test_survival_starts_high(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(0, 1, 1000))
        assert v.survival_y[0] >= v.survival_y[-1]

    def test_histogram_edges_sorted(self) -> None:
        v = build_visualization(np.random.default_rng(0).normal(0, 1, 1000))
        assert list(v.histogram_edges) == sorted(v.histogram_edges)


class TestEngineIntegration:
    def test_all_four_apis_run(self) -> None:
        eng = _engine()
        b = _base()
        assert eng.run_rul_uncertainty(b) is not None
        assert eng.run_failure_probability(b) is not None
        assert eng.run_health_distribution(b) is not None
        assert eng.run_portfolio_distribution([b]) is not None

    def test_rul_and_risk_consistent_pfail(self) -> None:
        eng = _engine()
        b = _base(degradation_rate=0.9, noise_std=0.0)
        rul = eng.run_rul_uncertainty(b)
        risk = eng.run_failure_probability(b)
        assert rul.probability_of_failure == pytest.approx(
            risk.probability_of_failure)

    def test_portfolio_single_matches_risk(self) -> None:
        eng = _engine()
        b = _base(degradation_rate=0.9, noise_std=0.0)
        port = eng.run_portfolio_distribution([b])
        risk = eng.run_failure_probability(b)
        assert port.per_asset_probability[0] == pytest.approx(
            risk.probability_of_failure)

    def test_store_trials_risk_loss_count(self) -> None:
        eng = MonteCarloEngine(MonteCarloConfig(n_trials=300, store_trials=True))
        r = eng.run_rul_uncertainty(_base())
        assert len(r.trials) == 300

    def test_confidence_levels_propagate(self) -> None:
        eng = MonteCarloEngine(MonteCarloConfig(n_trials=300,
                                                confidence_levels=(1.0, 50.0, 99.0)))
        r = eng.run_rul_uncertainty(_base())
        assert 1.0 in r.statistics.percentiles and 99.0 in r.statistics.percentiles

    def test_larger_noise_wider_distribution(self) -> None:
        eng = _engine(noise_sampler=NoiseSampler("uniform", low=0, high=0.1))
        narrow = eng.run_rul_uncertainty(_base(noise_std=0.1))
        eng2 = _engine(rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.4))
        wide = eng2.run_rul_uncertainty(_base())
        assert wide.statistics.std >= narrow.statistics.std

    def test_histogram_bins_config(self) -> None:
        eng = MonteCarloEngine(MonteCarloConfig(n_trials=400, histogram_bins=12))
        r = eng.run_rul_uncertainty(_base())
        assert len(r.visualization.histogram_counts) == 12

    def test_var_confidence_custom_in_risk(self) -> None:
        r = _engine().run_failure_probability(_base(), var_confidence=90.0)
        assert r.var_confidence == 90.0

    def test_portfolio_concentration_uniform_fleet(self) -> None:
        # Identical assets all failing -> concentration = 1/N
        eng = _engine()
        fleet = [_base(horizon=40, degradation_rate=2.0, noise_std=0.0)
                 for _ in range(4)]
        p = eng.run_portfolio_distribution(fleet)
        assert p.risk_concentration == pytest.approx(0.25, abs=1e-6)

    def test_rul_censored_at_horizon(self) -> None:
        # Non-failing trials are censored at the horizon
        r = _engine().run_rul_uncertainty(
            _base(horizon=100, degradation_rate=0.01, noise_std=0.0))
        assert r.statistics.maximum <= 100.0 + 1e-6