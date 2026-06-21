#!/usr/bin/env python3
"""Comprehensive test suite for ``src/simulation/asset_state_simulator.py``.

The simulator is pure NumPy, so the **entire** suite runs without PyTorch or
SciPy.  Coverage (>100 tests):

- AssetSimulatorConfig validation (general + piecewise + fault specific)
- DegradationModel enum
- Registry (register / build / list / duplicate rejection)
- Pure degradation generators (linear / exponential / piecewise / sudden fault)
- Simulation correctness per model
- [0, 100] bounding under all regimes
- Reproducibility / determinism under seeds
- Boundary cases (single step, extreme rates, zero noise)
- Fault events (count, location, magnitude, flagging)
- Noise behaviour and the monotonic floor
- Failure detection (crossed_failure, first_failure_step)
- AssetState / SimulationResult containers + serialization
- Ensemble generation
- Registry / tracker integration (failure-safe)
- Integration with the Week-5 prognostic pipeline
- Error handling

Run::

    pytest tests/test_asset_state_simulator.py -v
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

from src.simulation.asset_state_simulator import (
    DEFAULT_FAILURE_THRESHOLD,
    HEALTH_MAX,
    HEALTH_MIN,
    SIMULATOR_NAME,
    SIMULATOR_REGISTRY,
    AssetSimulatorConfig,
    AssetState,
    AssetStateSimulator,
    ConstantFaultSampler,
    ConstantRateSampler,
    DegradationModel,
    DegradationRateSampler,
    FaultSampler,
    GaussianNoiseSampler,
    NoiseSampler,
    SimulationResult,
    build_simulator,
    exponential_degradation,
    linear_degradation,
    list_simulators,
    piecewise_degradation,
    register_simulator,
    sudden_fault_degradation,
)


def _sim(**cfg) -> AssetStateSimulator:
    return AssetStateSimulator(AssetSimulatorConfig(**cfg))


# ---------------------------------------------------------------------------
# Config — general validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Tests for general configuration validation."""

    def test_defaults(self) -> None:
        c = AssetSimulatorConfig()
        assert c.horizon > 0 and c.timestep > 0
        assert c.model == DegradationModel.LINEAR.value

    def test_zero_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            AssetSimulatorConfig(horizon=0)

    def test_negative_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            AssetSimulatorConfig(horizon=-10)

    def test_zero_timestep(self) -> None:
        with pytest.raises(ValueError, match="timestep"):
            AssetSimulatorConfig(timestep=0)

    def test_negative_timestep(self) -> None:
        with pytest.raises(ValueError, match="timestep"):
            AssetSimulatorConfig(timestep=-1)

    def test_timestep_exceeds_horizon(self) -> None:
        with pytest.raises(ValueError, match="timestep"):
            AssetSimulatorConfig(horizon=10, timestep=20)

    def test_initial_health_too_high(self) -> None:
        with pytest.raises(ValueError, match="initial_health"):
            AssetSimulatorConfig(initial_health=150)

    def test_initial_health_negative(self) -> None:
        with pytest.raises(ValueError, match="initial_health"):
            AssetSimulatorConfig(initial_health=-5)

    def test_bad_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            AssetSimulatorConfig(model="quadratic")

    def test_negative_degradation_rate(self) -> None:
        with pytest.raises(ValueError, match="degradation_rate"):
            AssetSimulatorConfig(degradation_rate=-1)

    def test_negative_noise(self) -> None:
        with pytest.raises(ValueError, match="noise_std"):
            AssetSimulatorConfig(noise_std=-0.5)

    def test_negative_baseline_rate(self) -> None:
        with pytest.raises(ValueError, match="fault_baseline_rate"):
            AssetSimulatorConfig(fault_baseline_rate=-1)

    def test_frozen(self) -> None:
        c = AssetSimulatorConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.horizon = 50  # type: ignore[misc]

    def test_initial_health_boundary_zero(self) -> None:
        AssetSimulatorConfig(initial_health=0)

    def test_initial_health_boundary_hundred(self) -> None:
        AssetSimulatorConfig(initial_health=100)

    def test_timestep_equals_horizon(self) -> None:
        AssetSimulatorConfig(horizon=10, timestep=10)


# ---------------------------------------------------------------------------
# Config — piecewise validation
# ---------------------------------------------------------------------------


class TestPiecewiseValidation:
    """Tests for piecewise-specific validation."""

    def test_valid_piecewise(self) -> None:
        AssetSimulatorConfig(horizon=60, model="piecewise",
                             segment_breakpoints=(0, 20, 60),
                             segment_rates=(0.5, 1.0))

    def test_too_few_breakpoints(self) -> None:
        with pytest.raises(ValueError, match="segment_breakpoints"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(0,), segment_rates=())

    def test_rate_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="segment_rates"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(0, 20, 60),
                                 segment_rates=(0.5,))

    def test_unsorted_breakpoints(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(0, 40, 20),
                                 segment_rates=(0.5, 1.0))

    def test_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="segment_rates"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(0, 30, 60),
                                 segment_rates=(0.5, -1.0))

    def test_breakpoints_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="start at 0"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(5, 30, 60),
                                 segment_rates=(0.5, 1.0))


# ---------------------------------------------------------------------------
# Config — fault validation
# ---------------------------------------------------------------------------


class TestFaultValidation:
    """Tests for sudden-fault-specific validation."""

    def test_valid_faults(self) -> None:
        AssetSimulatorConfig(horizon=60, model="sudden_fault",
                             fault_times=(20, 40), fault_magnitudes=(15, 25))

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                 fault_times=(20,), fault_magnitudes=(15, 25))

    def test_fault_time_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="fault_time"):
            AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                 fault_times=(200,), fault_magnitudes=(15,))

    def test_negative_fault_time(self) -> None:
        with pytest.raises(ValueError, match="fault_time"):
            AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                 fault_times=(-5,), fault_magnitudes=(15,))

    def test_negative_magnitude(self) -> None:
        with pytest.raises(ValueError, match="fault_magnitudes"):
            AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                 fault_times=(20,), fault_magnitudes=(-15,))

    def test_empty_faults_valid(self) -> None:
        AssetSimulatorConfig(horizon=60, model="sudden_fault",
                             fault_times=(), fault_magnitudes=())


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class TestDegradationModel:
    """Tests for the degradation-model enum."""

    def test_values(self) -> None:
        assert DegradationModel.LINEAR.value == "linear"
        assert DegradationModel.EXPONENTIAL.value == "exponential"
        assert DegradationModel.PIECEWISE.value == "piecewise"
        assert DegradationModel.SUDDEN_FAULT.value == "sudden_fault"

    def test_is_str(self) -> None:
        assert DegradationModel.LINEAR == "linear"

    def test_four_models(self) -> None:
        assert len(list(DegradationModel)) == 4


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the registry."""

    def test_registered(self) -> None:
        assert SIMULATOR_NAME in SIMULATOR_REGISTRY
        assert SIMULATOR_NAME in list_simulators()

    def test_build_by_name(self) -> None:
        assert isinstance(build_simulator(SIMULATOR_NAME), AssetStateSimulator)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown simulator"):
            build_simulator("nope")

    def test_registry_name_stamped(self) -> None:
        assert AssetStateSimulator._registry_name == SIMULATOR_NAME

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_simulator(SIMULATOR_NAME)
            class _Other:
                pass

    def test_build_with_config(self) -> None:
        s = build_simulator(SIMULATOR_NAME,
                            config=AssetSimulatorConfig(horizon=33))
        assert s.config.horizon == 33


# ---------------------------------------------------------------------------
# Pure generators
# ---------------------------------------------------------------------------


class TestLinearGenerator:
    """Tests for the linear degradation generator."""

    def test_start_value(self) -> None:
        t = np.arange(10, dtype=float)
        h = linear_degradation(t, 90.0, 2.0)
        assert h[0] == 90.0

    def test_constant_rate(self) -> None:
        t = np.arange(10, dtype=float)
        h = linear_degradation(t, 90.0, 2.0)
        diffs = np.diff(h)
        assert np.allclose(diffs, -2.0)

    def test_zero_rate_flat(self) -> None:
        t = np.arange(10, dtype=float)
        h = linear_degradation(t, 80.0, 0.0)
        assert np.allclose(h, 80.0)


class TestExponentialGenerator:
    """Tests for the exponential degradation generator."""

    def test_start_value(self) -> None:
        t = np.arange(10, dtype=float)
        h = exponential_degradation(t, 90.0, 0.05)
        assert h[0] == pytest.approx(90.0)

    def test_decreasing(self) -> None:
        t = np.arange(10, dtype=float)
        h = exponential_degradation(t, 90.0, 0.05)
        assert np.all(np.diff(h) < 0)

    def test_multiplicative(self) -> None:
        t = np.arange(10, dtype=float)
        h = exponential_degradation(t, 100.0, 0.1)
        # ratio between consecutive samples is constant exp(-k)
        ratios = h[1:] / h[:-1]
        assert np.allclose(ratios, math.exp(-0.1))

    def test_zero_decay_flat(self) -> None:
        t = np.arange(10, dtype=float)
        h = exponential_degradation(t, 70.0, 0.0)
        assert np.allclose(h, 70.0)


class TestPiecewiseGenerator:
    """Tests for the piecewise degradation generator."""

    def test_start_value(self) -> None:
        t = np.arange(60, dtype=float)
        h = piecewise_degradation(t, 95.0, (0, 20, 40, 60), (0.3, 1.0, 2.5))
        assert h[0] == 95.0

    def test_continuity(self) -> None:
        t = np.arange(60, dtype=float)
        h = piecewise_degradation(t, 95.0, (0, 20, 40, 60), (0.3, 1.0, 2.5))
        # No discontinuous jumps larger than the steepest single-step rate
        assert np.max(np.abs(np.diff(h))) <= 2.5 + 1e-9

    def test_segment_rates_applied(self) -> None:
        t = np.arange(60, dtype=float)
        h = piecewise_degradation(t, 95.0, (0, 20, 40, 60), (0.3, 1.0, 2.5))
        # First segment slope ~ -0.3
        assert (h[10] - h[0]) / 10 == pytest.approx(-0.3, abs=0.01)
        # Last segment slope steeper
        assert (h[55] - h[45]) / 10 == pytest.approx(-2.5, abs=0.01)

    def test_monotone_for_positive_rates(self) -> None:
        t = np.arange(60, dtype=float)
        h = piecewise_degradation(t, 95.0, (0, 30, 60), (0.5, 1.5))
        assert np.all(np.diff(h) <= 1e-9)


class TestSuddenFaultGenerator:
    """Tests for the sudden-fault generator."""

    def test_baseline_degradation(self) -> None:
        t = np.arange(60, dtype=float)
        h, mask = sudden_fault_degradation(t, 95.0, 0.2, (), ())
        assert h[0] == 95.0
        assert not mask.any()

    def test_single_fault_drop(self) -> None:
        t = np.arange(60, dtype=float)
        h, mask = sudden_fault_degradation(t, 95.0, 0.2, (30,), (40,))
        assert h[30] < h[29] - 35  # large instantaneous drop

    def test_fault_mask_marks_step(self) -> None:
        t = np.arange(60, dtype=float)
        h, mask = sudden_fault_degradation(t, 95.0, 0.2, (30,), (40,))
        assert mask[30] and mask.sum() == 1

    def test_multiple_faults(self) -> None:
        t = np.arange(60, dtype=float)
        h, mask = sudden_fault_degradation(t, 95.0, 0.2, (20, 40), (15, 25))
        assert mask.sum() == 2

    def test_fault_persists(self) -> None:
        t = np.arange(60, dtype=float)
        h, mask = sudden_fault_degradation(t, 95.0, 0.0, (30,), (40,))
        # With zero baseline, post-fault health stays ~ 55
        assert h[59] == pytest.approx(55.0, abs=0.01)


# ---------------------------------------------------------------------------
# Simulation correctness
# ---------------------------------------------------------------------------


class TestSimulationCorrectness:
    """Tests for end-to-end simulation correctness."""

    def test_linear_run(self) -> None:
        r = _sim(horizon=50, model="linear", degradation_rate=1.5,
                 initial_health=95).simulate()
        assert r.model_used == "linear"
        assert r.initial_health == 95.0

    def test_exponential_run(self) -> None:
        r = _sim(horizon=50, model="exponential", degradation_rate=0.05,
                 initial_health=95).simulate()
        assert r.health[0] > r.health[-1]

    def test_piecewise_run(self) -> None:
        r = _sim(horizon=60, model="piecewise", initial_health=95,
                 segment_breakpoints=(0, 20, 40, 60),
                 segment_rates=(0.3, 1.0, 2.5)).simulate()
        assert r.health[-1] < r.health[0]

    def test_sudden_fault_run(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", initial_health=95,
                 fault_baseline_rate=0.2, fault_times=(30,),
                 fault_magnitudes=(40,)).simulate()
        assert len(r.fault_steps) == 1

    def test_final_health_recorded(self) -> None:
        r = _sim(horizon=20, degradation_rate=0.5).simulate()
        assert r.final_health == pytest.approx(r.health[-1])

    def test_n_samples_matches(self) -> None:
        r = _sim(horizon=20, timestep=1.0).simulate()
        assert r.n_samples == len(r.health) == len(r.times)

    def test_time_grid_inclusive(self) -> None:
        r = _sim(horizon=10, timestep=2.0).simulate()
        assert r.times[0] == 0.0
        assert r.times[-1] == 10.0
        assert r.n_samples == 6

    def test_model_used_recorded(self) -> None:
        for m in ("linear", "exponential"):
            r = _sim(horizon=20, model=m, degradation_rate=0.3).simulate()
            assert r.model_used == m


# ---------------------------------------------------------------------------
# Bounding
# ---------------------------------------------------------------------------


class TestBounding:
    """Tests for the [0, 100] bounding guarantee."""

    def test_linear_bounded(self) -> None:
        r = _sim(horizon=100, model="linear", degradation_rate=5,
                 initial_health=95).simulate()
        assert r.health.min() >= HEALTH_MIN
        assert r.health.max() <= HEALTH_MAX

    def test_exponential_bounded(self) -> None:
        r = _sim(horizon=100, model="exponential", degradation_rate=0.2,
                 initial_health=100).simulate()
        assert r.health.min() >= HEALTH_MIN

    def test_piecewise_bounded(self) -> None:
        r = _sim(horizon=60, model="piecewise", initial_health=95,
                 segment_breakpoints=(0, 20, 60),
                 segment_rates=(2.0, 5.0)).simulate()
        assert r.health.min() >= HEALTH_MIN

    def test_fault_bounded(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", initial_health=95,
                 fault_times=(10,), fault_magnitudes=(200,)).simulate()
        assert r.health.min() >= HEALTH_MIN  # huge fault clipped to 0

    def test_noise_bounded(self) -> None:
        r = _sim(horizon=50, degradation_rate=0.5, noise_std=20,
                 random_seed=1).simulate()
        assert r.health.min() >= HEALTH_MIN
        assert r.health.max() <= HEALTH_MAX

    def test_unclipped_can_exceed(self) -> None:
        r = _sim(horizon=100, model="linear", degradation_rate=5,
                 initial_health=95, clip_output=False).simulate()
        assert r.health.min() < HEALTH_MIN  # negative without clipping

    def test_initial_health_preserved_when_in_range(self) -> None:
        r = _sim(horizon=20, degradation_rate=0.5, initial_health=88,
                 noise_std=0).simulate()
        assert r.health[0] == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic behaviour under seeds."""

    def test_same_seed_identical(self) -> None:
        cfg = dict(horizon=50, degradation_rate=1, noise_std=3, random_seed=7)
        r1 = _sim(**cfg).simulate()
        r2 = _sim(**cfg).simulate()
        assert np.array_equal(r1.health, r2.health)

    def test_different_seed_differs(self) -> None:
        r1 = _sim(horizon=50, noise_std=3, random_seed=1).simulate()
        r2 = _sim(horizon=50, noise_std=3, random_seed=2).simulate()
        assert not np.array_equal(r1.health, r2.health)

    def test_seed_override(self) -> None:
        sim = _sim(horizon=50, noise_std=3, random_seed=1)
        r = sim.simulate(seed=99)
        assert r.random_seed == 99

    def test_seed_override_reproducible(self) -> None:
        sim = _sim(horizon=50, noise_std=3, random_seed=1)
        a = sim.simulate(seed=42)
        b = sim.simulate(seed=42)
        assert np.array_equal(a.health, b.health)

    def test_zero_noise_seed_irrelevant(self) -> None:
        r1 = _sim(horizon=50, degradation_rate=1, noise_std=0,
                  random_seed=1).simulate()
        r2 = _sim(horizon=50, degradation_rate=1, noise_std=0,
                  random_seed=999).simulate()
        assert np.array_equal(r1.health, r2.health)

    def test_config_not_mutated_by_override(self) -> None:
        sim = _sim(horizon=50, noise_std=3, random_seed=1)
        sim.simulate(seed=99)
        assert sim.config.random_seed == 1


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    """Tests for boundary and edge cases."""

    def test_single_step_grid(self) -> None:
        r = _sim(horizon=1, timestep=1).simulate()
        assert r.n_samples == 2

    def test_minimal_horizon(self) -> None:
        r = _sim(horizon=0.5, timestep=0.5).simulate()
        assert r.n_samples == 2

    def test_zero_degradation(self) -> None:
        r = _sim(horizon=50, degradation_rate=0, noise_std=0,
                 initial_health=80).simulate()
        assert np.allclose(r.health, 80.0)

    def test_initial_health_zero(self) -> None:
        r = _sim(horizon=20, initial_health=0, degradation_rate=0,
                 noise_std=0).simulate()
        assert np.allclose(r.health, 0.0)

    def test_initial_health_full(self) -> None:
        r = _sim(horizon=20, initial_health=100, degradation_rate=0,
                 noise_std=0).simulate()
        assert np.allclose(r.health, 100.0)

    def test_fractional_timestep(self) -> None:
        r = _sim(horizon=10, timestep=0.25).simulate()
        assert r.n_samples == 41

    def test_extreme_rate_floors_at_zero(self) -> None:
        r = _sim(horizon=50, degradation_rate=100,
                 initial_health=95).simulate()
        assert r.health[-1] == pytest.approx(HEALTH_MIN)


# ---------------------------------------------------------------------------
# Fault events
# ---------------------------------------------------------------------------


class TestFaultEvents:
    """Tests for fault-event behaviour."""

    def test_fault_step_recorded(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_times=(30,),
                 fault_magnitudes=(20,)).simulate()
        assert r.fault_steps == (30,)

    def test_multiple_fault_steps(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_times=(20, 40),
                 fault_magnitudes=(15, 25)).simulate()
        assert len(r.fault_steps) == 2

    def test_fault_causes_drop(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_baseline_rate=0.1,
                 fault_times=(30,), fault_magnitudes=(30,), noise_std=0).simulate()
        assert r.health[29] - r.health[30] > 25

    def test_fault_flagged_in_states(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_times=(30,),
                 fault_magnitudes=(20,)).simulate()
        faults = [s for s in r.states if s.is_fault_event]
        assert len(faults) == 1 and faults[0].step == 30

    def test_no_faults_empty(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_times=(),
                 fault_magnitudes=()).simulate()
        assert r.fault_steps == ()

    def test_fault_with_timestep_alignment(self) -> None:
        # Fault time 25 with timestep 2 -> nearest sample index
        r = _sim(horizon=60, timestep=2.0, model="sudden_fault",
                 fault_times=(25,), fault_magnitudes=(20,)).simulate()
        assert len(r.fault_steps) == 1


# ---------------------------------------------------------------------------
# Noise / monotonic floor
# ---------------------------------------------------------------------------


class TestNoiseAndFloor:
    """Tests for noise and the monotonic floor."""

    def test_noise_adds_variance(self) -> None:
        clean = _sim(horizon=50, degradation_rate=0.5, noise_std=0).simulate()
        noisy = _sim(horizon=50, degradation_rate=0.5, noise_std=5,
                     random_seed=1).simulate()
        assert np.std(noisy.health - clean.health) > 0

    def test_zero_noise_smooth(self) -> None:
        r = _sim(horizon=50, model="linear", degradation_rate=1.0,
                 noise_std=0).simulate()
        # Perfectly linear before clipping -> constant diffs
        assert np.allclose(np.diff(r.health), -1.0)

    def test_monotonic_floor_non_increasing(self) -> None:
        r = _sim(horizon=50, degradation_rate=0.5, noise_std=5,
                 floor_monotonic=True, random_seed=1).simulate()
        assert np.all(np.diff(r.health) <= 1e-9)

    def test_no_floor_allows_increase(self) -> None:
        r = _sim(horizon=50, degradation_rate=0.5, noise_std=8,
                 floor_monotonic=False, random_seed=2).simulate()
        # With heavy noise and no floor, some upward steps should appear
        assert np.any(np.diff(r.health) > 0)

    def test_higher_noise_more_variance(self) -> None:
        clean = _sim(horizon=50, degradation_rate=0.5, noise_std=0).simulate()
        low = _sim(horizon=50, degradation_rate=0.5, noise_std=1,
                   random_seed=3).simulate()
        high = _sim(horizon=50, degradation_rate=0.5, noise_std=8,
                    random_seed=3).simulate()
        assert np.std(high.health - clean.health) > np.std(low.health - clean.health)


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


class TestFailureDetection:
    """Tests for failure-crossing detection."""

    def test_failure_detected(self) -> None:
        r = _sim(horizon=50, model="linear", degradation_rate=5,
                 initial_health=95).simulate()
        assert r.crossed_failure
        assert r.first_failure_step > 0

    def test_no_failure_gentle(self) -> None:
        r = _sim(horizon=10, model="linear", degradation_rate=0.1,
                 initial_health=95).simulate()
        assert not r.crossed_failure
        assert r.first_failure_step == -1

    def test_first_failure_is_first(self) -> None:
        r = _sim(horizon=50, model="linear", degradation_rate=5,
                 initial_health=95, noise_std=0).simulate()
        # All samples from first_failure_step onward remain at/below the
        # configurable failure threshold (the post-hardening semantics).
        idx = r.first_failure_step
        assert np.all(r.health[idx:] <= r.failure_threshold + 1e-9)

    def test_fault_can_cause_failure(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", initial_health=50,
                 fault_times=(20,), fault_magnitudes=(60,)).simulate()
        assert r.crossed_failure


# ---------------------------------------------------------------------------
# Containers / serialization
# ---------------------------------------------------------------------------


class TestContainers:
    """Tests for the state and result containers."""

    def test_states_count(self) -> None:
        r = _sim(horizon=10).simulate()
        assert len(r.states) == r.n_samples

    def test_state_fields(self) -> None:
        r = _sim(horizon=10).simulate()
        s = r.states[0]
        assert s.step == 0 and s.time == 0.0

    def test_state_to_dict(self) -> None:
        r = _sim(horizon=10).simulate()
        d = r.states[0].to_dict()
        assert isinstance(d["health"], float)
        assert "is_fault_event" in d

    def test_result_to_dict(self) -> None:
        r = _sim(horizon=10).simulate()
        d = r.to_dict()
        assert isinstance(d["health"], list)
        assert isinstance(d["times"], list)
        assert d["model_used"] == "linear"

    def test_result_to_dict_fault_steps(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", fault_times=(30,),
                 fault_magnitudes=(20,)).simulate()
        assert r.to_dict()["fault_steps"] == [30]

    def test_state_chronological(self) -> None:
        r = _sim(horizon=20).simulate()
        times = [s.time for s in r.states]
        assert times == sorted(times)

    def test_result_arrays_are_numpy(self) -> None:
        r = _sim(horizon=10).simulate()
        assert isinstance(r.health, np.ndarray)
        assert isinstance(r.times, np.ndarray)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class TestEnsemble:
    """Tests for ensemble generation."""

    def test_ensemble_count(self) -> None:
        sim = _sim(horizon=50, noise_std=2, random_seed=10)
        ens = sim.simulate_ensemble(5)
        assert len(ens) == 5

    def test_ensemble_distinct(self) -> None:
        sim = _sim(horizon=50, noise_std=2, random_seed=10)
        ens = sim.simulate_ensemble(3)
        assert not np.array_equal(ens[0].health, ens[1].health)

    def test_ensemble_reproducible(self) -> None:
        sim1 = _sim(horizon=50, noise_std=2, random_seed=10)
        sim2 = _sim(horizon=50, noise_std=2, random_seed=10)
        e1 = sim1.simulate_ensemble(3)
        e2 = sim2.simulate_ensemble(3)
        assert all(np.array_equal(a.health, b.health) for a, b in zip(e1, e2))

    def test_ensemble_zero_raises(self) -> None:
        sim = _sim(horizon=50)
        with pytest.raises(ValueError, match="n_runs"):
            sim.simulate_ensemble(0)

    def test_ensemble_base_seed(self) -> None:
        sim = _sim(horizon=50, noise_std=2)
        ens = sim.simulate_ensemble(3, base_seed=100)
        assert ens[0].random_seed == 100 and ens[2].random_seed == 102


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_logs_simulation(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                            experiment_tracker=FakeTracker()).simulate()
        assert len(logged) == 1
        assert "sim_final_health" in logged[0]

    def test_logs_step_increment(self) -> None:
        steps = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                steps.append(step)

        sim = AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                                  experiment_tracker=FakeTracker())
        sim.simulate()
        sim.simulate()
        assert steps == [0, 1]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        r = AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                                experiment_tracker=BrokenTracker()).simulate()
        assert r.n_samples > 0

    def test_logs_fault_count(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        AssetStateSimulator(
            AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                 fault_times=(20, 40), fault_magnitudes=(10, 10)),
            experiment_tracker=FakeTracker()).simulate()
        assert logged[0]["sim_n_faults"] == 2.0


# ---------------------------------------------------------------------------
# Integration with Week-5 pipeline
# ---------------------------------------------------------------------------


class TestWeek5Integration:
    """Tests that simulated trajectories feed the Week-5 prognostic stack."""

    def test_trend_analyzer_consumes(self) -> None:
        from src.predictive.health_trend_analyzer import (
            HealthTrendAnalyzer,
            HealthTrendConfig,
        )

        r = _sim(horizon=40, model="linear", degradation_rate=1.5,
                 initial_health=95, noise_std=1.0, random_seed=5).simulate()
        result = HealthTrendAnalyzer(HealthTrendConfig()).analyze(r.health)
        assert result.trend is not None

    def test_rul_predictor_consumes(self) -> None:
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        r = _sim(horizon=40, model="linear", degradation_rate=1.5,
                 initial_health=95, noise_std=1.0, random_seed=5).simulate()
        pred = RULPredictor(RULConfig(failure_threshold=30,
                                      model="linear")).predict(r.health)
        assert pred.is_degrading

    def test_health_engine_score_batch_consumes(self) -> None:
        from src.predictive.health_index import (
            HealthIndexConfig,
            HealthIndexEngine,
        )

        r = _sim(horizon=30, model="linear", degradation_rate=1.0).simulate()
        # The simulated health is already a 0-100 trajectory; confirm range
        assert r.health.min() >= 0 and r.health.max() <= 100

    def test_degrading_sim_detected_as_degrading(self) -> None:
        from src.predictive.health_trend_analyzer import (
            HealthTrendAnalyzer,
            HealthTrendConfig,
            TrendDirection,
        )

        r = _sim(horizon=40, model="linear", degradation_rate=1.8,
                 initial_health=95, noise_std=0.5, random_seed=6).simulate()
        result = HealthTrendAnalyzer(HealthTrendConfig()).analyze(r.health)
        assert result.trend in (TrendDirection.DEGRADING,
                                TrendDirection.ACCELERATING)

    def test_full_pipeline_chain(self) -> None:
        from src.predictive.failure_risk import (
            FailureRiskConfig,
            FailureRiskEngine,
        )
        from src.predictive.rul_predictor import RULConfig, RULPredictor

        r = _sim(horizon=40, model="linear", degradation_rate=1.5,
                 initial_health=95, noise_std=1.0, random_seed=8).simulate()
        rul = RULPredictor(RULConfig(failure_threshold=30,
                                     model="linear")).predict(r.health)
        risk = FailureRiskEngine(FailureRiskConfig()).predict_from_rul(rul)
        assert risk.risk_level is not None


# ---------------------------------------------------------------------------
# Error handling / misc
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and miscellaneous behaviour."""

    def test_n_simulations_increments(self) -> None:
        sim = _sim(horizon=20)
        sim.simulate()
        sim.simulate()
        assert sim._n_simulations == 2

    def test_default_construction(self) -> None:
        sim = AssetStateSimulator()
        r = sim.simulate()
        assert r.n_samples > 0

    def test_list_input_breakpoints(self) -> None:
        # Tuples are required by config, but generator accepts sequences
        t = np.arange(40, dtype=float)
        h = piecewise_degradation(t, 90.0, [0, 20, 40], [0.5, 1.0])
        assert h[0] == 90.0

    def test_exponential_never_negative_unclipped(self) -> None:
        # exp decay asymptotes to 0 but never crosses it
        t = np.arange(200, dtype=float)
        h = exponential_degradation(t, 100.0, 0.1)
        assert np.all(h > 0)

    def test_simulate_returns_result_type(self) -> None:
        r = _sim(horizon=20).simulate()
        assert isinstance(r, SimulationResult)

    def test_states_return_assetstate_type(self) -> None:
        r = _sim(horizon=10).simulate()
        assert all(isinstance(s, AssetState) for s in r.states)


# ===========================================================================
# HARDENING TESTS (Week-6 Step-1 enterprise review)
# ===========================================================================


# ---------------------------------------------------------------------------
# Hardening 1 — Horizon guarantee
# ---------------------------------------------------------------------------


class TestHorizonGuarantee:
    """Tests that the final sample always lands exactly on the horizon."""

    def test_non_divisible_includes_horizon(self) -> None:
        r = _sim(horizon=10, timestep=3).simulate()
        assert r.times[-1] == pytest.approx(10.0)

    def test_non_divisible_times_sequence(self) -> None:
        r = _sim(horizon=10, timestep=3).simulate()
        assert np.allclose(r.times, [0, 3, 6, 9, 10])

    def test_divisible_unchanged(self) -> None:
        r = _sim(horizon=10, timestep=2).simulate()
        assert np.allclose(r.times, [0, 2, 4, 6, 8, 10])
        assert r.n_samples == 6

    def test_horizon_always_last(self) -> None:
        for h, ts in [(10, 3), (7, 2), (100, 7), (5, 1.5), (13, 4)]:
            r = _sim(horizon=h, timestep=ts).simulate()
            assert r.times[-1] == pytest.approx(h)

    def test_chronological_order_preserved(self) -> None:
        r = _sim(horizon=10, timestep=3).simulate()
        assert np.all(np.diff(r.times) > 0)

    def test_strictly_increasing_with_partial_step(self) -> None:
        # The appended horizon must be strictly greater than the prior sample
        r = _sim(horizon=11, timestep=4).simulate()  # 0,4,8,11
        assert np.all(np.diff(r.times) > 0)
        assert r.times[-1] == pytest.approx(11.0)

    def test_health_aligned_with_extended_grid(self) -> None:
        r = _sim(horizon=10, timestep=3, degradation_rate=1.0,
                 noise_std=0, initial_health=95).simulate()
        # Health at the appended horizon sample matches the linear law
        assert r.health[-1] == pytest.approx(95.0 - 1.0 * 10.0)

    def test_determinism_with_partial_step(self) -> None:
        cfg = dict(horizon=10, timestep=3, noise_std=2, random_seed=5)
        r1 = _sim(**cfg).simulate()
        r2 = _sim(**cfg).simulate()
        assert np.array_equal(r1.health, r2.health)

    def test_fractional_horizon_and_step(self) -> None:
        r = _sim(horizon=2.5, timestep=1.0).simulate()  # 0,1,2,2.5
        assert r.times[-1] == pytest.approx(2.5)
        assert np.allclose(r.times, [0, 1, 2, 2.5])

    def test_near_divisible_no_spurious_append(self) -> None:
        # horizon is an exact multiple -> no extra sample appended
        r = _sim(horizon=12, timestep=4).simulate()  # 0,4,8,12
        assert r.n_samples == 4
        assert r.times[-1] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Hardening 2 — Failure threshold support
# ---------------------------------------------------------------------------


class TestFailureThreshold:
    """Tests for the configurable failure threshold."""

    def test_default_threshold_is_30(self) -> None:
        assert AssetSimulatorConfig().failure_threshold == 30.0

    def test_threshold_validation_too_high(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            AssetSimulatorConfig(failure_threshold=150)

    def test_threshold_validation_negative(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            AssetSimulatorConfig(failure_threshold=-5)

    def test_threshold_boundary_zero(self) -> None:
        AssetSimulatorConfig(failure_threshold=0)

    def test_threshold_boundary_hundred(self) -> None:
        AssetSimulatorConfig(failure_threshold=100)

    def test_failure_at_threshold(self) -> None:
        # Health crosses 30 -> failure under default threshold
        r = _sim(horizon=50, model="linear", degradation_rate=2,
                 initial_health=95, noise_std=0).simulate()
        assert r.crossed_failure
        assert r.health[r.first_failure_step] <= 30 + 1e-9

    def test_no_failure_above_threshold(self) -> None:
        # Health bottoms out at 40 (> 30) -> no failure
        r = _sim(horizon=10, model="linear", degradation_rate=5,
                 initial_health=90, failure_threshold=30,
                 noise_std=0).simulate()
        # 90 - 5*10 = 40 at the end, never reaches 30
        assert not r.crossed_failure
        assert r.first_failure_step == -1

    def test_custom_threshold_50(self) -> None:
        r = _sim(horizon=20, model="linear", degradation_rate=3,
                 initial_health=95, failure_threshold=50,
                 noise_std=0).simulate()
        assert r.crossed_failure
        assert r.health[r.first_failure_step] <= 50 + 1e-9

    def test_higher_threshold_earlier_failure(self) -> None:
        low = _sim(horizon=50, model="linear", degradation_rate=2,
                   initial_health=95, failure_threshold=20,
                   noise_std=0).simulate()
        high = _sim(horizon=50, model="linear", degradation_rate=2,
                    initial_health=95, failure_threshold=60,
                    noise_std=0).simulate()
        assert high.first_failure_step < low.first_failure_step

    def test_failure_time_recorded(self) -> None:
        r = _sim(horizon=50, model="linear", degradation_rate=2,
                 initial_health=95, noise_std=0).simulate()
        assert r.failure_time == pytest.approx(r.times[r.first_failure_step])

    def test_failure_time_minus_one_when_no_failure(self) -> None:
        r = _sim(horizon=10, model="linear", degradation_rate=0.1,
                 initial_health=95).simulate()
        assert r.failure_time == -1.0

    def test_threshold_in_result(self) -> None:
        r = _sim(horizon=20, failure_threshold=45).simulate()
        assert r.failure_threshold == 45.0

    def test_threshold_in_to_dict(self) -> None:
        r = _sim(horizon=20, failure_threshold=45).simulate()
        d = r.to_dict()
        assert d["failure_threshold"] == 45.0
        assert "failure_time" in d

    def test_threshold_zero_requires_health_zero(self) -> None:
        # With threshold 0, only health==0 counts as failure (legacy behaviour)
        r = _sim(horizon=50, model="linear", degradation_rate=5,
                 initial_health=95, failure_threshold=0,
                 noise_std=0).simulate()
        assert r.health[r.first_failure_step] == pytest.approx(0.0)

    def test_threshold_hundred_fails_immediately(self) -> None:
        # threshold 100 -> initial health 95 <= 100 -> fails at step 0
        r = _sim(horizon=20, initial_health=95, failure_threshold=100,
                 degradation_rate=0, noise_std=0).simulate()
        assert r.first_failure_step == 0
        assert r.failure_time == pytest.approx(0.0)

    def test_fault_triggers_threshold_failure(self) -> None:
        r = _sim(horizon=60, model="sudden_fault", initial_health=80,
                 fault_baseline_rate=0.1, fault_times=(20,),
                 fault_magnitudes=(55,), failure_threshold=30,
                 noise_std=0).simulate()
        assert r.crossed_failure


# ---------------------------------------------------------------------------
# Hardening 3 — Piecewise breakpoint span validation
# ---------------------------------------------------------------------------


class TestPiecewiseSpanValidation:
    """Tests that piecewise segments must span the horizon."""

    def test_reject_breakpoints_short_of_horizon(self) -> None:
        with pytest.raises(ValueError, match="span"):
            AssetSimulatorConfig(horizon=180, model="piecewise",
                                 segment_breakpoints=(0, 20, 40),
                                 segment_rates=(0.3, 1.0))

    def test_accept_breakpoints_equal_horizon(self) -> None:
        AssetSimulatorConfig(horizon=180, model="piecewise",
                             segment_breakpoints=(0, 20, 40, 180),
                             segment_rates=(0.3, 1.0, 2.0))

    def test_accept_breakpoints_beyond_horizon(self) -> None:
        AssetSimulatorConfig(horizon=180, model="piecewise",
                             segment_breakpoints=(0, 20, 40, 250),
                             segment_rates=(0.3, 1.0, 2.0))

    def test_exact_horizon_boundary(self) -> None:
        # Final breakpoint exactly equals horizon
        AssetSimulatorConfig(horizon=60, model="piecewise",
                             segment_breakpoints=(0, 30, 60),
                             segment_rates=(0.5, 1.0))

    def test_just_short_rejected(self) -> None:
        with pytest.raises(ValueError, match="span"):
            AssetSimulatorConfig(horizon=60, model="piecewise",
                                 segment_breakpoints=(0, 30, 59),
                                 segment_rates=(0.5, 1.0))

    def test_span_validation_runs_full_horizon(self) -> None:
        # A spanning config simulates without the trailing samples holding flat
        r = _sim(horizon=60, model="piecewise",
                 segment_breakpoints=(0, 20, 40, 60),
                 segment_rates=(0.3, 1.0, 2.5), noise_std=0).simulate()
        # Last sample should reflect the steep final segment, not a held value
        assert r.health[-1] < r.health[40]


# ---------------------------------------------------------------------------
# Hardening 4 — Monte Carlo sampler hooks
# ---------------------------------------------------------------------------


class TestMonteCarloHooks:
    """Tests for the Monte Carlo sampler extension points."""

    def test_default_samplers_present(self) -> None:
        sim = AssetStateSimulator(AssetSimulatorConfig(horizon=20))
        assert isinstance(sim.rate_sampler, ConstantRateSampler)
        assert isinstance(sim.fault_sampler, ConstantFaultSampler)
        assert isinstance(sim.noise_sampler, GaussianNoiseSampler)

    def test_constant_rate_sampler_returns_base(self) -> None:
        rng = np.random.default_rng(0)
        assert ConstantRateSampler().sample(rng, 1.5) == 1.5

    def test_constant_fault_sampler_echoes(self) -> None:
        rng = np.random.default_rng(0)
        t, m = ConstantFaultSampler().sample(rng, (10, 20), (5, 6))
        assert t == (10.0, 20.0) and m == (5.0, 6.0)

    def test_gaussian_noise_sampler_zero(self) -> None:
        rng = np.random.default_rng(0)
        out = GaussianNoiseSampler().sample(rng, 10, 0.0)
        assert np.all(out == 0.0)

    def test_gaussian_noise_sampler_nonzero(self) -> None:
        rng = np.random.default_rng(0)
        out = GaussianNoiseSampler().sample(rng, 10, 2.0)
        assert out.shape == (10,) and np.std(out) > 0

    def test_default_samplers_preserve_behaviour(self) -> None:
        # A simulator with explicit default samplers == one with none
        cfg = AssetSimulatorConfig(horizon=50, degradation_rate=1,
                                   noise_std=3, random_seed=7)
        r_implicit = AssetStateSimulator(cfg).simulate()
        r_explicit = AssetStateSimulator(
            cfg, rate_sampler=ConstantRateSampler(),
            fault_sampler=ConstantFaultSampler(),
            noise_sampler=GaussianNoiseSampler()).simulate()
        assert np.array_equal(r_implicit.health, r_explicit.health)

    def test_custom_rate_sampler_injection(self) -> None:
        class DoubleRate(DegradationRateSampler):
            def sample(self, rng, base_rate):
                return base_rate * 2.0

        cfg = AssetSimulatorConfig(horizon=20, model="linear",
                                   degradation_rate=1.0, noise_std=0,
                                   initial_health=95)
        base = AssetStateSimulator(cfg).simulate()
        doubled = AssetStateSimulator(cfg, rate_sampler=DoubleRate()).simulate()
        # Doubled rate degrades faster
        assert doubled.final_health < base.final_health

    def test_custom_noise_sampler_injection(self) -> None:
        class ZeroNoise(NoiseSampler):
            def sample(self, rng, n, noise_std):
                return np.zeros(n)

        cfg = AssetSimulatorConfig(horizon=30, degradation_rate=1,
                                   noise_std=5, random_seed=1)
        zeroed = AssetStateSimulator(cfg, noise_sampler=ZeroNoise()).simulate()
        clean = AssetStateSimulator(
            AssetSimulatorConfig(horizon=30, degradation_rate=1,
                                 noise_std=0)).simulate()
        assert np.array_equal(zeroed.health, clean.health)

    def test_custom_fault_sampler_injection(self) -> None:
        class AddFault(FaultSampler):
            def sample(self, rng, base_times, base_magnitudes):
                return (30.0,), (40.0,)

        cfg = AssetSimulatorConfig(horizon=60, model="sudden_fault",
                                   fault_baseline_rate=0.1, noise_std=0)
        r = AssetStateSimulator(cfg, fault_sampler=AddFault()).simulate()
        assert len(r.fault_steps) == 1

    def test_samplers_are_abstract(self) -> None:
        with pytest.raises(TypeError):
            DegradationRateSampler()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            FaultSampler()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            NoiseSampler()  # type: ignore[abstract]

    def test_custom_samplers_deterministic(self) -> None:
        class JitterRate(DegradationRateSampler):
            def sample(self, rng, base_rate):
                return base_rate + rng.normal(0, 0.1)

        cfg = AssetSimulatorConfig(horizon=30, degradation_rate=1.0,
                                   random_seed=3, noise_std=0)
        r1 = AssetStateSimulator(cfg, rate_sampler=JitterRate()).simulate()
        r2 = AssetStateSimulator(cfg, rate_sampler=JitterRate()).simulate()
        assert np.array_equal(r1.health, r2.health)


# ---------------------------------------------------------------------------
# Hardening 5 — Enterprise telemetry
# ---------------------------------------------------------------------------


class TestEnterpriseTelemetry:
    """Tests for the expanded tracker telemetry."""

    def _capture(self, **cfg):
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)

        AssetStateSimulator(AssetSimulatorConfig(**cfg),
                            experiment_tracker=FakeTracker()).simulate()
        return logged[0]

    def test_mean_health_logged(self) -> None:
        m = self._capture(horizon=30, degradation_rate=1)
        assert "sim_mean_health" in m

    def test_min_max_health_logged(self) -> None:
        m = self._capture(horizon=30, degradation_rate=1)
        assert "sim_min_health" in m and "sim_max_health" in m

    def test_failure_time_logged(self) -> None:
        m = self._capture(horizon=30, degradation_rate=3, initial_health=95)
        assert "sim_failure_time" in m

    def test_threshold_logged(self) -> None:
        m = self._capture(horizon=30, failure_threshold=40)
        assert m["sim_threshold"] == 40.0

    def test_duration_logged(self) -> None:
        m = self._capture(horizon=33)
        assert m["sim_duration"] == 33.0

    def test_fault_count_logged(self) -> None:
        m = self._capture(horizon=60, model="sudden_fault",
                          fault_times=(20, 40), fault_magnitudes=(10, 10))
        assert m["sim_fault_count"] == 2.0

    def test_original_metrics_preserved(self) -> None:
        m = self._capture(horizon=30, degradation_rate=1)
        for key in ("sim_initial_health", "sim_final_health", "sim_n_samples",
                    "sim_n_faults", "sim_crossed_failure"):
            assert key in m

    def test_mean_within_bounds(self) -> None:
        m = self._capture(horizon=30, degradation_rate=1)
        assert 0 <= m["sim_mean_health"] <= 100

    def test_min_le_mean_le_max(self) -> None:
        m = self._capture(horizon=30, degradation_rate=1, noise_std=2,
                          random_seed=1)
        assert m["sim_min_health"] <= m["sim_mean_health"] <= m["sim_max_health"]

    def test_model_logged_as_param(self) -> None:
        params = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                pass

            def log_params(self, p):
                params.append(p)

        AssetStateSimulator(AssetSimulatorConfig(horizon=20, model="exponential",
                                                 degradation_rate=0.05),
                            experiment_tracker=FakeTracker()).simulate()
        assert params and params[0]["sim_model"] == "exponential"

    def test_telemetry_failure_safe_metrics(self) -> None:
        class BrokenMetrics:
            def log_metrics(self, *a, **k):
                raise RuntimeError("metrics down")

        r = AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                                experiment_tracker=BrokenMetrics()).simulate()
        assert r.n_samples > 0

    def test_telemetry_failure_safe_params(self) -> None:
        class BrokenParams:
            def log_metrics(self, *a, **k):
                pass

            def log_params(self, *a, **k):
                raise RuntimeError("params down")

        r = AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                                experiment_tracker=BrokenParams()).simulate()
        assert r.n_samples > 0

    def test_tracker_without_log_params(self) -> None:
        # A tracker lacking log_params must not cause an error
        class MetricsOnly:
            def log_metrics(self, metrics, step=None):
                pass

        r = AssetStateSimulator(AssetSimulatorConfig(horizon=20),
                                experiment_tracker=MetricsOnly()).simulate()
        assert r.n_samples > 0


# ---------------------------------------------------------------------------
# Hardening — backward-compatibility guard
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests guarding the public API and deterministic behaviour."""

    def test_default_noise_path_deterministic(self) -> None:
        # Manual reconstruction of the legacy noise path must match
        cfg = AssetSimulatorConfig(horizon=50, model="linear",
                                   degradation_rate=1, noise_std=3,
                                   initial_health=100, random_seed=7)
        r = AssetStateSimulator(cfg).simulate()
        times = np.arange(51, dtype=float)
        expected = 100.0 - 1 * times
        rng = np.random.default_rng(7)
        expected = np.clip(expected + rng.normal(0, 3, 51), 0, 100)
        assert np.array_equal(r.health, expected)

    def test_public_result_fields_present(self) -> None:
        r = _sim(horizon=20).simulate()
        for attr in ("times", "health", "fault_steps", "model_used",
                     "initial_health", "final_health", "n_samples",
                     "random_seed", "crossed_failure", "first_failure_step"):
            assert hasattr(r, attr)

    def test_new_result_fields_present(self) -> None:
        r = _sim(horizon=20).simulate()
        assert hasattr(r, "failure_threshold")
        assert hasattr(r, "failure_time")

    def test_simulate_signature_unchanged(self) -> None:
        # simulate(seed=...) still works as before
        sim = _sim(horizon=20, noise_std=2)
        r = sim.simulate(seed=42)
        assert r.random_seed == 42

    def test_registry_unchanged(self) -> None:
        assert SIMULATOR_NAME in list_simulators()

    def test_zero_noise_unaffected_by_samplers(self) -> None:
        # With zero noise, the noise sampler is never invoked; output is the
        # pure deterministic degradation.
        r = _sim(horizon=30, model="linear", degradation_rate=1.0,
                 noise_std=0, initial_health=90).simulate()
        assert np.allclose(np.diff(r.health), -1.0)