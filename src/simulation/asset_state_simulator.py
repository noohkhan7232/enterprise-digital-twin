#!/usr/bin/env python3
"""Asset state simulator for the digital-twin simulation engine.

This module is the first stage of the Week-6 simulation engine, the synthetic
counterpart to the Week-5 prognostic stack.  Where the prognostic engines *infer*
the condition of a real machine from acoustic evidence, the simulator *generates*
plausible future health trajectories under controlled degradation regimes and
fault scenarios.  Those synthetic trajectories are the substrate for closed-loop
validation of the prognostic stack (against known ground-truth failure times),
for what-if analysis (propagating a hypothetical future through the Week-5
pipeline), and for policy optimisation of the maintenance decision agent.

The simulator emits health-index trajectories on the same ``[0, 100]`` scale that
the Week-5 :class:`~src.predictive.health_index.HealthIndexEngine` produces and
the :class:`~src.predictive.health_trend_analyzer.HealthTrendAnalyzer` and
:class:`~src.predictive.rul_predictor.RULPredictor` consume, so a simulated
trajectory can be fed directly into the prognostic pipeline without adaptation.

============================================================================
Simulation assumptions
============================================================================
The simulator is a *behavioural* digital twin: it reproduces the macroscopic
shape of health-index degradation, not the underlying micro-mechanics.  Each
degradation regime encodes an explicit assumption:

* **Linear wear** (``H(t) = H0 - rate·t``) — assumes a constant-rate loss of
  health, appropriate for steady abrasive or adhesive wear where the damage
  accumulation rate is approximately time-invariant.
* **Exponential wear** (``H(t) = H0·exp(-k·t)``) — assumes a damage rate
  proportional to remaining health, producing accelerating multiplicative
  decay; appropriate for self-reinforcing mechanisms such as fatigue-crack
  growth or progressive bearing spallation.
* **Regime transitions** (piecewise-linear) — assumes the asset passes through
  discrete operating regimes (e.g. benign run-in, normal service, accelerated
  end-of-life), each with its own constant wear rate, joined continuously so no
  artificial discontinuity is introduced at a transition.
* **Sudden fault behaviour** — assumes discrete, instantaneous step losses of
  health superimposed on a baseline wear trend, modelling abrupt events such as
  a blade strike, a lubrication loss, or a bearing seizure; each step persists
  for the remainder of the trajectory.

Gaussian observation noise (optional) models measurement and estimation jitter
in the health index itself, not a physical process.  An optional monotonic floor
models the irreversibility of mechanical wear at the trajectory level.

============================================================================
Limitations
============================================================================
This component is deliberately *not* a high-fidelity physics engine.  In
particular it is:

* **Not a finite-element (FEA) stress/strain solver** — it does not model
  material mechanics, geometry, or load paths.
* **Not a computational-fluid-dynamics (CFD) model** — it does not model
  aerodynamic or thermal fields.
* **Not a vibration-spectrum synthesiser** — it produces a scalar health index,
  not time-domain acoustic or vibration signals with spectral content.

It is a trajectory-level behavioural model whose purpose is to exercise and
validate the downstream prognostic and decision logic, not to substitute for a
physics-based twin.

============================================================================
Intended usage
============================================================================
* **Closed-loop validation** — generate trajectories with a known ground-truth
  failure time (via the configurable failure threshold) and measure the
  accuracy and interval calibration of the Week-5 RUL predictor against them.
* **RUL benchmarking** — sweep degradation regimes and rates to characterise
  predictor behaviour across the operating envelope.
* **Maintenance policy evaluation** — drive the maintenance decision agent with
  controlled scenarios to assess action/priority/cost outcomes.
* **Scenario simulation** — construct specific what-if futures (a sudden fault
  at a chosen time, an accelerating regime) for tabletop analysis.
* **Monte Carlo forecasting** — the sampler hooks (see below) allow a Week-7
  Monte Carlo engine to randomise degradation rate, fault occurrence, and noise
  across an ensemble without modifying simulator internals.

============================================================================
Monte Carlo extension architecture (Week-7 readiness)
============================================================================
The simulator exposes three *sampler* extension points, each a small strategy
object that supplies a parameter to the deterministic core.  The default
samplers are **degenerate** — they return the configured constant exactly — so
behaviour is unchanged from a non-sampled simulator and full backward
compatibility and determinism are preserved.  A future Monte Carlo engine
supplies stochastic samplers instead, without touching the simulation maths.

Extension points::

    DegradationRateSampler.sample(rng, base_rate)            -> float
    FaultSampler.sample(rng, base_times, base_magnitudes)    -> (times, mags)
    NoiseSampler.sample(rng, n, noise_std)                   -> np.ndarray

Design rationale: the deterministic degradation maths is the *invariant* of the
simulator and must never be duplicated or forked for the stochastic case.  By
isolating every source of randomness behind a sampler interface, the Monte Carlo
engine becomes a *composition* (supply stochastic samplers) rather than a
*modification* (edit the simulator), which keeps the validated core untouched
and the two engines independently testable.  The default samplers consume the
random generator in exactly the same order and quantity as the pre-hardening
implementation, so seeded reproducibility is bit-for-bit preserved.

Class diagram (text)::

    AssetStateSimulator
      ├── config: AssetSimulatorConfig            (frozen, validated)
      ├── rate_sampler:  DegradationRateSampler ──┐
      ├── fault_sampler: FaultSampler            ├─ strategy hooks
      └── noise_sampler: NoiseSampler ───────────┘   (default = degenerate)

    DegradationRateSampler (ABC)
      └── ConstantRateSampler            (default; returns config rate)
    FaultSampler (ABC)
      └── ConstantFaultSampler           (default; returns config faults)
    NoiseSampler (ABC)
      └── GaussianNoiseSampler           (default; reproduces config noise)

Usage::

    from src.simulation.asset_state_simulator import (
        AssetStateSimulator, AssetSimulatorConfig, DegradationModel,
    )

    cfg = AssetSimulatorConfig(
        horizon=180, timestep=1.0, initial_health=98.0,
        model=DegradationModel.PIECEWISE,
        segment_breakpoints=(0, 60, 120, 180),
        segment_rates=(0.10, 0.35, 0.90),
        noise_std=1.5, failure_threshold=30.0, random_seed=7,
    )
    result = AssetStateSimulator(cfg).simulate()

    trajectory = result.health      # np.ndarray bounded to [0, 100]
    print(result.crossed_failure, result.failure_time)

CLI::

    python src/simulation/asset_state_simulator.py --demo
"""

from __future__ import annotations

import abc
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("asset_state_simulator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Health-index bounds (matching the Week-5 health engine).
HEALTH_MIN: Final[float] = 0.0
HEALTH_MAX: Final[float] = 100.0

#: Default failure threshold (health at/below which the asset is "failed").
DEFAULT_FAILURE_THRESHOLD: Final[float] = 30.0

#: Tolerance for floating-point grid/threshold comparisons.
_EPS: Final[float] = 1e-9

#: Registry of named asset simulators.
SIMULATOR_REGISTRY: dict[str, type] = {}

SIMULATOR_NAME: Final[str] = "acoustic_asset_simulator"


# ---------------------------------------------------------------------------
# Degradation-model enum
# ---------------------------------------------------------------------------


class DegradationModel(str, Enum):
    """Supported degradation regimes."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    PIECEWISE = "piecewise"
    SUDDEN_FAULT = "sudden_fault"


# ---------------------------------------------------------------------------
# Monte Carlo sampler hooks (Week-7 readiness)
# ---------------------------------------------------------------------------
#
# These strategy objects isolate every source of randomness behind a stable
# interface.  The DEFAULT implementations are degenerate: they reproduce the
# configured constant behaviour exactly, so a simulator constructed without
# explicit samplers behaves identically to the pre-hardening implementation and
# remains fully deterministic.  A Week-7 Monte Carlo engine supplies stochastic
# subclasses without modifying any simulator internals.


class DegradationRateSampler(abc.ABC):
    """Strategy supplying the scalar degradation rate (or decay constant).

    Subclasses randomise the rate for Monte Carlo ensembles; the default
    :class:`ConstantRateSampler` returns the configured rate unchanged.
    """

    @abc.abstractmethod
    def sample(self, rng: "np.random.Generator", base_rate: float) -> float:
        """Return a degradation rate.

        Args:
            rng: A seeded NumPy generator (for reproducibility).
            base_rate: The configured rate to draw around.

        Returns:
            The degradation rate to use for this run.
        """
        raise NotImplementedError


class ConstantRateSampler(DegradationRateSampler):
    """Degenerate rate sampler that returns the configured rate exactly.

    Consumes no random numbers, so it does not perturb the generator stream.
    """

    def sample(self, rng: "np.random.Generator", base_rate: float) -> float:
        """Return *base_rate* unchanged.

        Args:
            rng: Unused (present for interface compatibility).
            base_rate: The configured rate.

        Returns:
            ``base_rate``.
        """
        return float(base_rate)


class FaultSampler(abc.ABC):
    """Strategy supplying the fault times and magnitudes for a run.

    Subclasses randomise fault occurrence for Monte Carlo ensembles; the default
    :class:`ConstantFaultSampler` returns the configured faults unchanged.
    """

    @abc.abstractmethod
    def sample(
        self, rng: "np.random.Generator",
        base_times: Sequence[float], base_magnitudes: Sequence[float],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return fault times and magnitudes.

        Args:
            rng: A seeded NumPy generator.
            base_times: The configured fault times.
            base_magnitudes: The configured fault magnitudes.

        Returns:
            Tuple ``(times, magnitudes)``.
        """
        raise NotImplementedError


class ConstantFaultSampler(FaultSampler):
    """Degenerate fault sampler that returns the configured faults exactly.

    Consumes no random numbers, so it does not perturb the generator stream.
    """

    def sample(
        self, rng: "np.random.Generator",
        base_times: Sequence[float], base_magnitudes: Sequence[float],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the configured faults unchanged.

        Args:
            rng: Unused (present for interface compatibility).
            base_times: The configured fault times.
            base_magnitudes: The configured fault magnitudes.

        Returns:
            Tuple ``(times, magnitudes)`` echoing the configuration.
        """
        return (
            tuple(float(t) for t in base_times),
            tuple(float(m) for m in base_magnitudes),
        )


class NoiseSampler(abc.ABC):
    """Strategy supplying the additive observation-noise vector for a run.

    Subclasses can implement heteroscedastic or correlated noise for Monte Carlo
    ensembles; the default :class:`GaussianNoiseSampler` reproduces the
    configured i.i.d. Gaussian noise exactly.
    """

    @abc.abstractmethod
    def sample(
        self, rng: "np.random.Generator", n: int, noise_std: float
    ) -> "np.ndarray":
        """Return an additive-noise vector of length *n*.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.
            noise_std: The configured noise standard deviation.

        Returns:
            Array ``(n,)`` of additive noise (zeros when ``noise_std == 0``).
        """
        raise NotImplementedError


class GaussianNoiseSampler(NoiseSampler):
    """Degenerate noise sampler reproducing the configured i.i.d. Gaussian noise.

    This draws exactly as the pre-hardening implementation did
    (``rng.normal(0, noise_std, n)``), preserving determinism bit-for-bit.
    """

    def sample(
        self, rng: "np.random.Generator", n: int, noise_std: float
    ) -> "np.ndarray":
        """Return an i.i.d. Gaussian noise vector (or zeros).

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.
            noise_std: The configured noise standard deviation.

        Returns:
            Array ``(n,)`` of noise; all zeros when ``noise_std == 0``.
        """
        if noise_std <= 0:
            return np.zeros(n, dtype=float)
        return rng.normal(0.0, noise_std, size=n)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_simulator(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering an asset simulator by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = SIMULATOR_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Simulator '{name}' already registered to {existing.__name__}"
            )
        SIMULATOR_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered simulator '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_simulator(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered asset simulator by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the simulator constructor.

    Returns:
        An instantiated simulator.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in SIMULATOR_REGISTRY:
        available = ", ".join(sorted(SIMULATOR_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown simulator '{name}'. Available: {available}")
    return SIMULATOR_REGISTRY[name](**kwargs)


def list_simulators() -> list[str]:
    """Return the sorted names of registered asset simulators.

    Returns:
        Sorted registry keys.
    """
    return sorted(SIMULATOR_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetSimulatorConfig:
    """Configuration for the :class:`AssetStateSimulator`.

    Attributes:
        horizon: Total simulated duration in physical time units (e.g. days);
            the trajectory spans ``[0, horizon]`` and the final sample is always
            exactly ``horizon`` (see the horizon guarantee in
            :meth:`AssetStateSimulator._build_time_grid`).
        timestep: Nominal duration of one simulation step; the grid is uniform at
            this step except for a final partial step to land on ``horizon``.
        initial_health: Starting health index in ``[0, 100]``.
        model: The degradation regime (:class:`DegradationModel`).
        degradation_rate: Health lost per unit time for the linear model, and the
            decay constant ``k`` for the exponential model.
        noise_std: Standard deviation of the additive Gaussian observation noise
            (in health units); ``0`` disables noise.
        random_seed: Seed for the noise and stochastic-fault generators; fixing
            it makes a trajectory reproducible bit-for-bit.
        failure_threshold: Health at/below which the asset is considered failed;
            drives ``crossed_failure``, ``first_failure_step``, and
            ``failure_time`` in the result.  Defaults to 30.
        segment_breakpoints: For the piecewise model, the ordered time boundaries
            (length ``S + 1`` for ``S`` segments), starting at 0; the final
            breakpoint must be ``>= horizon`` so the segments span the run.
        segment_rates: For the piecewise model, the per-segment linear rates
            (length ``S``).
        fault_times: For the sudden-fault model, the times at which instantaneous
            health drops occur.
        fault_magnitudes: For the sudden-fault model, the magnitude of each drop
            (aligned with ``fault_times``).
        fault_baseline_rate: For the sudden-fault model, the linear degradation
            rate applied between fault events.
        clip_output: Clip the final trajectory to ``[0, 100]`` (always advisable;
            exposed for testing the unclipped signal).
        floor_monotonic: When ``True``, enforce a non-increasing trajectory after
            noise, modelling irreversible wear (faults still apply).
    """

    horizon:             float = 100.0
    timestep:            float = 1.0
    initial_health:      float = 100.0
    model:               str = DegradationModel.LINEAR.value
    degradation_rate:    float = 0.5
    noise_std:           float = 0.0
    random_seed:         int = 0
    failure_threshold:   float = DEFAULT_FAILURE_THRESHOLD
    segment_breakpoints: tuple[float, ...] = ()
    segment_rates:       tuple[float, ...] = ()
    fault_times:         tuple[float, ...] = ()
    fault_magnitudes:    tuple[float, ...] = ()
    fault_baseline_rate: float = 0.2
    clip_output:         bool = True
    floor_monotonic:     bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.timestep <= 0:
            raise ValueError("timestep must be > 0")
        if self.timestep > self.horizon:
            raise ValueError("timestep must be <= horizon")
        if not (HEALTH_MIN <= self.initial_health <= HEALTH_MAX):
            raise ValueError("initial_health must be in [0, 100]")
        if self.model not in {m.value for m in DegradationModel}:
            raise ValueError(
                f"model must be one of {[m.value for m in DegradationModel]}"
            )
        if self.degradation_rate < 0:
            raise ValueError("degradation_rate must be >= 0")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if self.fault_baseline_rate < 0:
            raise ValueError("fault_baseline_rate must be >= 0")
        if not (HEALTH_MIN <= self.failure_threshold <= HEALTH_MAX):
            raise ValueError("failure_threshold must be in [0, 100]")

        if self.model == DegradationModel.PIECEWISE.value:
            self._validate_piecewise()
        if self.model == DegradationModel.SUDDEN_FAULT.value:
            self._validate_faults()

    def _validate_piecewise(self) -> None:
        """Validate the piecewise-specific configuration.

        Raises:
            ValueError: On malformed breakpoints or rates, or when the segments
                do not span the simulation horizon.
        """
        bps = self.segment_breakpoints
        rates = self.segment_rates
        if len(bps) < 2:
            raise ValueError(
                "piecewise model requires segment_breakpoints with >= 2 entries"
            )
        if len(rates) != len(bps) - 1:
            raise ValueError(
                "segment_rates must have exactly len(segment_breakpoints) - 1 entries"
            )
        if list(bps) != sorted(bps):
            raise ValueError("segment_breakpoints must be non-decreasing")
        if any(r < 0 for r in rates):
            raise ValueError("segment_rates must be >= 0")
        if bps[0] != 0:
            raise ValueError("segment_breakpoints must start at 0")
        if bps[-1] < self.horizon - _EPS:
            raise ValueError(
                "segment_breakpoints[-1] must be >= horizon so the segments span "
                f"the simulation (got {bps[-1]} < horizon {self.horizon})"
            )

    def _validate_faults(self) -> None:
        """Validate the sudden-fault-specific configuration.

        Raises:
            ValueError: On malformed fault specification.
        """
        if len(self.fault_times) != len(self.fault_magnitudes):
            raise ValueError(
                "fault_times and fault_magnitudes must have equal length"
            )
        if any(t < 0 or t > self.horizon for t in self.fault_times):
            raise ValueError("every fault_time must be in [0, horizon]")
        if any(m < 0 for m in self.fault_magnitudes):
            raise ValueError("fault_magnitudes must be >= 0")


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetState:
    """A single sampled asset state.

    Attributes:
        step: The 0-based sample index.
        time: The physical time of the sample.
        health: The health index in ``[0, 100]``.
        is_fault_event: Whether a discrete fault event occurred at this step.
    """

    step:           int
    time:           float
    health:         float
    is_fault_event: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the state.
        """
        return {
            "step": self.step,
            "time": self.time,
            "health": self.health,
            "is_fault_event": self.is_fault_event,
        }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """The result of one simulation run.

    Attributes:
        times: Array ``(N,)`` of sample times (final entry is exactly horizon).
        health: Array ``(N,)`` of health values in ``[0, 100]``.
        fault_steps: Sorted indices at which fault events occurred.
        model_used: The degradation model applied.
        initial_health: The starting health.
        final_health: The health at the last sample.
        n_samples: Number of samples.
        random_seed: The seed used.
        failure_threshold: The failure threshold applied for the bookkeeping.
        crossed_failure: Whether the trajectory reached/passed the threshold.
        first_failure_step: The first index at which ``health <= threshold``
            (``-1`` if never).
        failure_time: The physical time of the first threshold crossing
            (``-1.0`` if never).
    """

    times:              "np.ndarray"
    health:             "np.ndarray"
    fault_steps:        tuple[int, ...]
    model_used:         str
    initial_health:     float
    final_health:       float
    n_samples:          int
    random_seed:        int
    failure_threshold:  float
    crossed_failure:    bool
    first_failure_step: int
    failure_time:       float

    @property
    def states(self) -> list[AssetState]:
        """Materialise the per-sample :class:`AssetState` objects.

        Returns:
            A list of ``n_samples`` states in chronological order.
        """
        fault_set = set(self.fault_steps)
        return [
            AssetState(
                step=i, time=float(self.times[i]), health=float(self.health[i]),
                is_fault_event=(i in fault_set),
            )
            for i in range(self.n_samples)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; arrays are rendered as lists.
        """
        return {
            "times": [float(t) for t in self.times],
            "health": [float(h) for h in self.health],
            "fault_steps": list(self.fault_steps),
            "model_used": self.model_used,
            "initial_health": self.initial_health,
            "final_health": self.final_health,
            "n_samples": self.n_samples,
            "random_seed": self.random_seed,
            "failure_threshold": self.failure_threshold,
            "crossed_failure": self.crossed_failure,
            "first_failure_step": self.first_failure_step,
            "failure_time": self.failure_time,
        }


# ---------------------------------------------------------------------------
# Pure degradation generators (NumPy only)
# ---------------------------------------------------------------------------


def linear_degradation(
    times: "np.ndarray", initial_health: float, rate: float
) -> "np.ndarray":
    """Generate a linear degradation trajectory ``H0 - rate·t``.

    Args:
        times: Sample times ``(N,)``.
        initial_health: Starting health.
        rate: Health lost per unit time.

    Returns:
        The (unclipped) health trajectory ``(N,)``.
    """
    return initial_health - rate * times


def exponential_degradation(
    times: "np.ndarray", initial_health: float, decay: float
) -> "np.ndarray":
    """Generate an exponential degradation trajectory ``H0·exp(-k·t)``.

    Args:
        times: Sample times ``(N,)``.
        initial_health: Starting health.
        decay: Decay constant ``k`` (``>= 0``).

    Returns:
        The (unclipped) health trajectory ``(N,)``.
    """
    return initial_health * np.exp(-decay * times)


def piecewise_degradation(
    times: "np.ndarray", initial_health: float,
    breakpoints: Sequence[float], rates: Sequence[float],
) -> "np.ndarray":
    """Generate a piecewise-linear degradation trajectory.

    Each segment degrades at its own constant rate; the trajectory is continuous
    across segment boundaries (each segment starts where the previous ended).

    Args:
        times: Sample times ``(N,)``.
        initial_health: Starting health.
        breakpoints: Ordered segment boundaries (length ``S + 1``).
        rates: Per-segment rates (length ``S``).

    Returns:
        The (unclipped) health trajectory ``(N,)``.
    """
    health = np.empty_like(times, dtype=float)
    bp = np.asarray(breakpoints, dtype=float)
    seg_start_health = float(initial_health)
    for s in range(len(rates)):
        lo, hi = bp[s], bp[s + 1]
        mask = (times >= lo) & (times < hi) if s < len(rates) - 1 else (times >= lo)
        local = times[mask] - lo
        health[mask] = seg_start_health - rates[s] * local
        # Health at the end of this segment becomes the next segment's start.
        seg_start_health = seg_start_health - rates[s] * (hi - lo)
    # Any samples before the first breakpoint hold the initial health.
    pre = times < bp[0]
    if pre.any():
        health[pre] = initial_health
    return health


def sudden_fault_degradation(
    times: "np.ndarray", initial_health: float, baseline_rate: float,
    fault_times: Sequence[float], fault_magnitudes: Sequence[float],
) -> tuple["np.ndarray", "np.ndarray"]:
    """Generate a baseline trajectory with superimposed instantaneous faults.

    Between faults the asset degrades linearly at ``baseline_rate``; at each fault
    time the health drops instantaneously by the corresponding magnitude, and the
    drop persists for the remainder of the trajectory.

    Args:
        times: Sample times ``(N,)``.
        initial_health: Starting health.
        baseline_rate: Linear degradation rate between faults.
        fault_times: Times at which faults occur.
        fault_magnitudes: Health drop at each fault.

    Returns:
        Tuple ``(health, fault_mask)`` where *fault_mask* is a boolean array
        marking the first sample at or after each fault time.
    """
    health = initial_health - baseline_rate * times
    fault_mask = np.zeros_like(times, dtype=bool)
    for ft, fm in zip(fault_times, fault_magnitudes):
        # Index of the first sample at or after the fault time.
        idx = int(np.searchsorted(times, ft, side="left"))
        if idx < times.size:
            health[idx:] -= fm
            fault_mask[idx] = True
    return health, fault_mask


# ---------------------------------------------------------------------------
# AssetStateSimulator
# ---------------------------------------------------------------------------


@register_simulator(SIMULATOR_NAME)
class AssetStateSimulator:
    """Generates synthetic asset health trajectories.

    Produces a bounded ``[0, 100]`` health trajectory under the configured
    degradation regime, with optional Gaussian observation noise and an optional
    irreversible-wear floor.  Deterministic under a fixed seed.  Registry- and
    tracker-integrated.

    Randomness is isolated behind three sampler hooks
    (:class:`DegradationRateSampler`, :class:`FaultSampler`,
    :class:`NoiseSampler`).  The default samplers are degenerate and reproduce
    the configured constant behaviour exactly, so a simulator built without
    explicit samplers is bit-for-bit identical to the pre-hardening version.  A
    Week-7 Monte Carlo engine supplies stochastic samplers without modifying any
    simulator internals.

    Args:
        config: The simulator configuration.
        experiment_tracker: Optional tracker for logging simulation summaries.
        rate_sampler: Optional degradation-rate sampler (default constant).
        fault_sampler: Optional fault sampler (default constant).
        noise_sampler: Optional noise sampler (default Gaussian, config-driven).
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: AssetSimulatorConfig | None = None,
        experiment_tracker: Any = None,
        rate_sampler: DegradationRateSampler | None = None,
        fault_sampler: FaultSampler | None = None,
        noise_sampler: NoiseSampler | None = None,
    ) -> None:
        self.config = config or AssetSimulatorConfig()
        self.tracker = experiment_tracker
        self.rate_sampler = rate_sampler or ConstantRateSampler()
        self.fault_sampler = fault_sampler or ConstantFaultSampler()
        self.noise_sampler = noise_sampler or GaussianNoiseSampler()
        self._n_simulations = 0
        logger.info(
            "AssetStateSimulator ready | model=%s | horizon=%.1f | dt=%.3f | "
            "threshold=%.1f",
            self.config.model, self.config.horizon, self.config.timestep,
            self.config.failure_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, *, seed: int | None = None) -> SimulationResult:
        """Run one simulation and return the trajectory.

        Args:
            seed: Optional override of the configured random seed (does not
                mutate the frozen config); enables generating an ensemble of
                trajectories from one simulator.

        Returns:
            A :class:`SimulationResult`.
        """
        cfg = self.config
        times = self._build_time_grid()
        effective_seed = cfg.random_seed if seed is None else int(seed)
        rng = np.random.default_rng(effective_seed)

        # --- Resolve sampled parameters (degenerate by default) ---
        # The default samplers consume no random numbers, so the generator
        # stream reaching the noise draw is identical to the pre-hardening code.
        rate = self.rate_sampler.sample(rng, cfg.degradation_rate)
        if cfg.model == DegradationModel.SUDDEN_FAULT.value:
            baseline_rate = self.rate_sampler.sample(rng, cfg.fault_baseline_rate)
        else:
            baseline_rate = cfg.fault_baseline_rate
        fault_times, fault_mags = self.fault_sampler.sample(
            rng, cfg.fault_times, cfg.fault_magnitudes,
        )

        # --- Generate the deterministic degradation baseline ---
        fault_mask = np.zeros_like(times, dtype=bool)
        if cfg.model == DegradationModel.LINEAR.value:
            health = linear_degradation(times, cfg.initial_health, rate)
        elif cfg.model == DegradationModel.EXPONENTIAL.value:
            health = exponential_degradation(times, cfg.initial_health, rate)
        elif cfg.model == DegradationModel.PIECEWISE.value:
            health = piecewise_degradation(times, cfg.initial_health,
                                           cfg.segment_breakpoints,
                                           cfg.segment_rates)
        else:  # SUDDEN_FAULT
            health, fault_mask = sudden_fault_degradation(
                times, cfg.initial_health, baseline_rate,
                fault_times, fault_mags,
            )

        # --- Additive observation noise (deterministic under the seed) ---
        # The default GaussianNoiseSampler draws exactly rng.normal(0, std, n),
        # preserving bit-for-bit compatibility with the pre-hardening version.
        if cfg.noise_std > 0:
            health = health + self.noise_sampler.sample(
                rng, int(health.size), cfg.noise_std,
            )

        # --- Optional irreversible-wear floor (post-noise) ---
        if cfg.floor_monotonic:
            health = np.minimum.accumulate(health)

        # --- Bound to [0, 100] ---
        if cfg.clip_output:
            health = np.clip(health, HEALTH_MIN, HEALTH_MAX)

        # --- Failure bookkeeping against the configurable threshold ---
        crossed, first_fail, fail_time = self._failure_bookkeeping(health, times)

        fault_steps = tuple(int(i) for i in np.flatnonzero(fault_mask))

        result = SimulationResult(
            times=times,
            health=health,
            fault_steps=fault_steps,
            model_used=cfg.model,
            initial_health=float(cfg.initial_health),
            final_health=float(health[-1]),
            n_samples=int(health.size),
            random_seed=effective_seed,
            failure_threshold=float(cfg.failure_threshold),
            crossed_failure=crossed,
            first_failure_step=first_fail,
            failure_time=fail_time,
        )
        self._log_result(result)
        self._n_simulations += 1
        return result

    def simulate_ensemble(
        self, n_runs: int, *, base_seed: int | None = None
    ) -> list[SimulationResult]:
        """Run an ensemble of simulations with distinct seeds.

        Args:
            n_runs: Number of trajectories to generate.
            base_seed: Starting seed; run ``i`` uses ``base_seed + i``.  Defaults
                to the configured seed.

        Returns:
            A list of ``n_runs`` results.

        Raises:
            ValueError: When ``n_runs`` is not positive.
        """
        if n_runs <= 0:
            raise ValueError("n_runs must be >= 1")
        start = self.config.random_seed if base_seed is None else int(base_seed)
        return [self.simulate(seed=start + i) for i in range(n_runs)]

    # ------------------------------------------------------------------
    # Failure bookkeeping
    # ------------------------------------------------------------------

    def _failure_bookkeeping(
        self, health: "np.ndarray", times: "np.ndarray"
    ) -> tuple[bool, int, float]:
        """Locate the first failure-threshold crossing.

        Args:
            health: The bounded health trajectory.
            times: The matching sample times.

        Returns:
            Tuple ``(crossed_failure, first_failure_step, failure_time)``;
            the step is ``-1`` and the time ``-1.0`` when the threshold is never
            reached.
        """
        at_or_below = health <= (self.config.failure_threshold + _EPS)
        crossed = bool(at_or_below.any())
        if not crossed:
            return False, -1, -1.0
        first = int(np.argmax(at_or_below))
        return True, first, float(times[first])

    # ------------------------------------------------------------------
    # Time grid (with horizon guarantee)
    # ------------------------------------------------------------------

    def _build_time_grid(self) -> "np.ndarray":
        """Construct the sample-time grid ``[0, horizon]`` at the configured step.

        The grid is uniform at ``timestep``.  When ``horizon`` is not an exact
        integer multiple of ``timestep`` the final uniform sample falls short of
        ``horizon``; in that case the exact ``horizon`` is appended as a final
        (shorter) step, guaranteeing the trajectory always ends precisely at the
        horizon while preserving strict chronological ordering.

        Returns:
            Array ``(N,)`` of sample times with ``times[0] == 0`` and
            ``times[-1] == horizon``.
        """
        cfg = self.config
        n = int(math.floor(cfg.horizon / cfg.timestep)) + 1
        grid = np.arange(n, dtype=float) * cfg.timestep
        # Horizon guarantee: append the exact horizon if the grid stops short.
        if cfg.horizon - grid[-1] > _EPS:
            grid = np.append(grid, float(cfg.horizon))
        return grid

    # ------------------------------------------------------------------
    # Tracker integration (enterprise telemetry)
    # ------------------------------------------------------------------

    def _log_result(self, result: SimulationResult) -> None:
        """Log a simulation summary to the experiment tracker (failure-safe).

        Emits the original five metrics plus the enterprise-telemetry set
        (health distribution, failure time, threshold, duration, fault count).
        The categorical ``sim_model`` is logged via ``log_params`` when the
        tracker supports it.  Every tracker call is guarded so that a tracker
        fault can never interrupt a simulation.

        Args:
            result: The result to log.
        """
        if self.tracker is None:
            return
        cfg = self.config
        try:
            health = result.health
            self.tracker.log_metrics(
                {
                    # --- original metrics (backward compatible) ---
                    "sim_initial_health": result.initial_health,
                    "sim_final_health": result.final_health,
                    "sim_n_samples": float(result.n_samples),
                    "sim_n_faults": float(len(result.fault_steps)),
                    "sim_crossed_failure": float(result.crossed_failure),
                    # --- enterprise telemetry ---
                    "sim_mean_health": float(np.mean(health)),
                    "sim_min_health": float(np.min(health)),
                    "sim_max_health": float(np.max(health)),
                    "sim_failure_time": float(result.failure_time),
                    "sim_threshold": float(result.failure_threshold),
                    "sim_duration": float(cfg.horizon),
                    "sim_fault_count": float(len(result.fault_steps)),
                },
                step=self._n_simulations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        # The model name is categorical; log it as a param when supported.
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"sim_model": result.model_used})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic simulation demo across the four models.

    Returns:
        Exit code 0.
    """
    demos = {
        "linear": AssetSimulatorConfig(
            horizon=60, timestep=1.0, initial_health=95.0,
            model=DegradationModel.LINEAR.value, degradation_rate=1.2,
            noise_std=1.0, random_seed=1,
        ),
        "exponential": AssetSimulatorConfig(
            horizon=60, timestep=1.0, initial_health=95.0,
            model=DegradationModel.EXPONENTIAL.value, degradation_rate=0.05,
            noise_std=1.0, random_seed=2,
        ),
        "piecewise": AssetSimulatorConfig(
            horizon=60, timestep=1.0, initial_health=95.0,
            model=DegradationModel.PIECEWISE.value,
            segment_breakpoints=(0, 20, 40, 60),
            segment_rates=(0.3, 1.0, 2.5), noise_std=1.0, random_seed=3,
        ),
        "sudden_fault": AssetSimulatorConfig(
            horizon=60, timestep=1.0, initial_health=95.0,
            model=DegradationModel.SUDDEN_FAULT.value, fault_baseline_rate=0.3,
            fault_times=(25, 45), fault_magnitudes=(30, 25),
            noise_std=1.0, random_seed=4,
        ),
    }
    for name, cfg in demos.items():
        result = AssetStateSimulator(cfg).simulate()
        logger.info(
            "[%-13s] start=%.1f end=%.1f faults=%d failed=%s fail_time=%.1f",
            name, result.initial_health, result.final_health,
            len(result.fault_steps), result.crossed_failure, result.failure_time,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Asset state simulator")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic simulation demo.")
    parser.add_argument("--list-simulators", action="store_true")
    args = parser.parse_args(argv)

    if args.list_simulators:
        print("Registered simulators:", list_simulators())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())