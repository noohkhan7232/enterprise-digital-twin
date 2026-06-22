#!/usr/bin/env python3
"""Monte Carlo risk-intelligence engine — uncertainty quantification.

Week-7 Phase-1 layers *uncertainty quantification* over the platform's
deterministic prognostics.  The Week-5/6 stack produces point estimates — a
single remaining-useful-life number, a single failure probability, a single
fleet risk.  Real assets are uncertain: degradation rates vary, observation
noise perturbs the health signal, and the failure threshold itself is not known
exactly.  This engine propagates that input uncertainty through the (frozen)
simulator and prognostic chain by Monte Carlo sampling, turning::

    RUL = 42 days

into a *distribution*::

    P10 = 28    P50 = 43    P90 = 67

together with full risk metrics (probability of failure, expected loss,
Value-at-Risk, Conditional VaR), portfolio analytics, and chart-ready arrays.

============================================================================
Design
============================================================================
The engine is a strict, additive **composition** of the frozen lower layers.
It never modifies the simulator, scenario engine, or fleet engine; it drives the
frozen :class:`~src.simulation.asset_state_simulator.AssetStateSimulator` across
many trials, each with sampled parameters, and aggregates the resulting
trajectories into distributions.

Randomness is isolated behind three uncertainty samplers
(:class:`DegradationRateSampler`, :class:`NoiseSampler`,
:class:`FailureThresholdSampler`), each supporting ``normal`` / ``uniform`` /
``triangular`` distributions.  A single master seed deterministically derives
per-trial seeds, so an entire ensemble is reproducible bit-for-bit.

============================================================================
Architecture properties
============================================================================
* **Pure NumPy** — no SciPy, no PyTorch.
* **Frozen, validated dataclasses** for every config and distribution.
* **Registry-compatible**, mirroring every Week-1–6 module.
* **JSON-serialisable** outputs (non-finite floats render as ``null``).
* **Deterministic** under a master seed; per-trial seeds derived reproducibly.
* **Failure-safe tracker integration**; no global mutable state.

Usage::

    from src.simulation.asset_state_simulator import AssetSimulatorConfig
    from src.risk.monte_carlo_engine import (
        MonteCarloEngine, MonteCarloConfig,
        DegradationRateSampler, NoiseSampler, FailureThresholdSampler,
    )

    base = AssetSimulatorConfig(horizon=120, model="linear",
                                degradation_rate=0.8, initial_health=95,
                                failure_threshold=30)
    engine = MonteCarloEngine(
        MonteCarloConfig(n_trials=2000, random_seed=7),
        rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.2),
        noise_sampler=NoiseSampler("uniform", low=0.0, high=1.5),
        threshold_sampler=FailureThresholdSampler("triangular",
                                                  low=25, mode=30, high=35),
    )
    rul = engine.run_rul_uncertainty(base)
    print(rul.p10, rul.p50, rul.p90)

CLI::

    python src/risk/monte_carlo_engine.py --demo
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

from src.simulation.asset_state_simulator import (  # noqa: E402
    AssetSimulatorConfig,
    AssetStateSimulator,
)

logger = logging.getLogger("monte_carlo_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named Monte Carlo engines.
MONTE_CARLO_ENGINE_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "risk_monte_carlo_engine"

#: Default per-event economics (matching the Week-5 maintenance agent).
DEFAULT_FAILURE_COST: Final[float] = 50_000.0
DEFAULT_FAILURE_DOWNTIME_H: Final[float] = 72.0

#: A finite sentinel used when a trial does not fail within the horizon.
_NO_FAILURE: Final[float] = -1.0

_EPS: Final[float] = 1e-12


def _jsonsafe(x: float) -> float | None:
    """Render a non-finite float as ``None`` for JSON.

    Args:
        x: A float that may be ``inf`` or ``NaN``.

    Returns:
        ``None`` when non-finite, else the float.
    """
    return None if (math.isinf(x) or math.isnan(x)) else float(x)


# ---------------------------------------------------------------------------
# Distribution-kind enum
# ---------------------------------------------------------------------------


class DistributionKind(str, Enum):
    """Supported sampling distributions."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# ---------------------------------------------------------------------------
# Uncertainty samplers
# ---------------------------------------------------------------------------


class _BaseSampler(abc.ABC):
    """Base class for parameterised uncertainty samplers.

    A sampler draws a scalar (or per-sample vector, for noise) from one of the
    supported distributions.  Construction validates the distribution-specific
    parameters so an ill-formed sampler fails immediately.

    Args:
        kind: The distribution kind (``normal`` / ``uniform`` / ``triangular``).
        mean: Mean (normal).
        std: Standard deviation (normal).
        low: Lower bound (uniform / triangular).
        high: Upper bound (uniform / triangular).
        mode: Mode (triangular).
    """

    def __init__(
        self,
        kind: str = DistributionKind.NORMAL.value,
        *,
        mean: float = 0.0,
        std: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
        mode: float = 0.5,
    ) -> None:
        valid = {d.value for d in DistributionKind}
        if kind not in valid:
            raise ValueError(f"kind must be one of {sorted(valid)}, got '{kind}'")
        self.kind = kind
        self.mean = float(mean)
        self.std = float(std)
        self.low = float(low)
        self.high = float(high)
        self.mode = float(mode)
        self._validate()

    def _validate(self) -> None:
        """Validate the distribution parameters.

        Raises:
            ValueError: On invalid parameters for the chosen kind.
        """
        if self.kind == DistributionKind.NORMAL.value:
            if self.std < 0:
                raise ValueError("normal sampler requires std >= 0")
        elif self.kind == DistributionKind.UNIFORM.value:
            if self.high < self.low:
                raise ValueError("uniform sampler requires high >= low")
        elif self.kind == DistributionKind.TRIANGULAR.value:
            if not (self.low <= self.mode <= self.high):
                raise ValueError(
                    "triangular sampler requires low <= mode <= high"
                )
            if self.high < self.low:
                raise ValueError("triangular sampler requires high >= low")

    def _draw(self, rng: "np.random.Generator", n: int) -> "np.ndarray":
        """Draw *n* samples from the configured distribution.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.

        Returns:
            Array ``(n,)`` of samples.
        """
        if self.kind == DistributionKind.NORMAL.value:
            if self.std == 0:
                return np.full(n, self.mean, dtype=float)
            return rng.normal(self.mean, self.std, size=n)
        if self.kind == DistributionKind.UNIFORM.value:
            if self.high == self.low:
                return np.full(n, self.low, dtype=float)
            return rng.uniform(self.low, self.high, size=n)
        # triangular
        if self.high == self.low:
            return np.full(n, self.low, dtype=float)
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def sample(self, rng: "np.random.Generator", n: int = 1) -> "np.ndarray":
        """Public sampling entry point.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples (default 1).

        Returns:
            Array ``(n,)`` of samples.
        """
        return self._draw(rng, int(n))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the sampler.

        Returns:
            Dictionary with the kind and its active parameters.
        """
        base = {"kind": self.kind}
        if self.kind == DistributionKind.NORMAL.value:
            base.update({"mean": self.mean, "std": self.std})
        elif self.kind == DistributionKind.UNIFORM.value:
            base.update({"low": self.low, "high": self.high})
        else:
            base.update({"low": self.low, "mode": self.mode, "high": self.high})
        return base


class DegradationRateSampler(_BaseSampler):
    """Samples the asset degradation rate (or decay constant) per trial.

    Negative draws are clamped to zero, since a negative degradation rate is
    unphysical (it would model healing).
    """

    def sample(self, rng: "np.random.Generator", n: int = 1) -> "np.ndarray":
        """Sample non-negative degradation rates.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.

        Returns:
            Array ``(n,)`` of non-negative rates.
        """
        return np.clip(self._draw(rng, int(n)), 0.0, None)


class NoiseSampler(_BaseSampler):
    """Samples the observation-noise standard deviation per trial.

    Negative draws are clamped to zero (a standard deviation cannot be negative).
    """

    def sample(self, rng: "np.random.Generator", n: int = 1) -> "np.ndarray":
        """Sample non-negative noise standard deviations.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.

        Returns:
            Array ``(n,)`` of non-negative noise levels.
        """
        return np.clip(self._draw(rng, int(n)), 0.0, None)


class FailureThresholdSampler(_BaseSampler):
    """Samples the failure threshold per trial, clamped to ``[0, 100]``."""

    def sample(self, rng: "np.random.Generator", n: int = 1) -> "np.ndarray":
        """Sample failure thresholds bounded to the health range.

        Args:
            rng: A seeded NumPy generator.
            n: Number of samples.

        Returns:
            Array ``(n,)`` of thresholds in ``[0, 100]``.
        """
        return np.clip(self._draw(rng, int(n)), 0.0, 100.0)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_monte_carlo_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a Monte Carlo engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = MONTE_CARLO_ENGINE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Monte Carlo engine '{name}' already registered to "
                f"{existing.__name__}"
            )
        MONTE_CARLO_ENGINE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered Monte Carlo engine '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_monte_carlo_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered Monte Carlo engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated Monte Carlo engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in MONTE_CARLO_ENGINE_REGISTRY:
        available = ", ".join(sorted(MONTE_CARLO_ENGINE_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown Monte Carlo engine '{name}'. Available: {available}"
        )
    return MONTE_CARLO_ENGINE_REGISTRY[name](**kwargs)


def list_monte_carlo_engines() -> list[str]:
    """Return the sorted names of registered Monte Carlo engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(MONTE_CARLO_ENGINE_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for the :class:`MonteCarloEngine`.

    Attributes:
        n_trials: Number of Monte Carlo trials (``>= 1``).
        random_seed: Master seed; per-trial seeds derive from it deterministically.
        confidence_levels: Percentile levels to report (each in ``[0, 100]``).
        parallel_ready: Architectural flag signalling the workload may be
            parallelised (the engine is pure and per-trial independent); does not
            change results, only documents intent for a future parallel backend.
        store_trials: Whether to retain the raw per-trial samples in outputs
            (memory-heavy for large ``n_trials``; statistics are always retained).
        failure_cost: Cost attributed to an in-horizon failure (for expected loss
            and VaR/CVaR).
        failure_downtime_h: Downtime hours attributed to an in-horizon failure.
        histogram_bins: Number of bins for chart-ready histograms.
    """

    n_trials:           int = 1000
    random_seed:        int = 0
    confidence_levels:  tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)
    parallel_ready:     bool = True
    store_trials:       bool = False
    failure_cost:       float = DEFAULT_FAILURE_COST
    failure_downtime_h: float = DEFAULT_FAILURE_DOWNTIME_H
    histogram_bins:     int = 30

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid trial count, seed, levels, or economics.
        """
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not self.confidence_levels:
            raise ValueError("confidence_levels must be non-empty")
        for lvl in self.confidence_levels:
            if not (0.0 <= lvl <= 100.0):
                raise ValueError("each confidence level must be in [0, 100]")
        if list(self.confidence_levels) != sorted(self.confidence_levels):
            raise ValueError("confidence_levels must be sorted ascending")
        if self.failure_cost < 0:
            raise ValueError("failure_cost must be >= 0")
        if self.failure_downtime_h < 0:
            raise ValueError("failure_downtime_h must be >= 0")
        if self.histogram_bins < 1:
            raise ValueError("histogram_bins must be >= 1")


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionStatistics:
    """Summary statistics of a sampled quantity.

    Attributes:
        mean: Arithmetic mean.
        median: Median (P50).
        std: Standard deviation.
        variance: Variance.
        minimum: Smallest sample.
        maximum: Largest sample.
        p5, p10, p25, p50, p75, p90, p95: Standard percentiles.
        percentiles: Mapping of every configured confidence level to its value.
        n: Number of samples.
    """

    mean:        float
    median:      float
    std:         float
    variance:    float
    minimum:     float
    maximum:     float
    p5:          float
    p10:         float
    p25:         float
    p50:         float
    p75:         float
    p90:         float
    p95:         float
    percentiles: dict[float, float]
    n:           int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; percentile keys are stringified floats.
        """
        return {
            "mean": _jsonsafe(self.mean),
            "median": _jsonsafe(self.median),
            "std": _jsonsafe(self.std),
            "variance": _jsonsafe(self.variance),
            "minimum": _jsonsafe(self.minimum),
            "maximum": _jsonsafe(self.maximum),
            "p5": _jsonsafe(self.p5),
            "p10": _jsonsafe(self.p10),
            "p25": _jsonsafe(self.p25),
            "p50": _jsonsafe(self.p50),
            "p75": _jsonsafe(self.p75),
            "p90": _jsonsafe(self.p90),
            "p95": _jsonsafe(self.p95),
            "percentiles": {str(k): _jsonsafe(v)
                            for k, v in self.percentiles.items()},
            "n": self.n,
        }


def compute_statistics(
    samples: "np.ndarray", confidence_levels: Sequence[float]
) -> DistributionStatistics:
    """Compute summary statistics for a 1-D sample array.

    Args:
        samples: The samples ``(n,)``.
        confidence_levels: Percentile levels to report.

    Returns:
        A :class:`DistributionStatistics`.

    Raises:
        ValueError: When *samples* is empty.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("compute_statistics requires a non-empty sample array")
    pct = {float(lvl): float(np.percentile(arr, lvl))
           for lvl in confidence_levels}

    def p(level: float) -> float:
        return float(np.percentile(arr, level))

    return DistributionStatistics(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        variance=float(np.var(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        p5=p(5), p10=p(10), p25=p(25), p50=p(50),
        p75=p(75), p90=p(90), p95=p(95),
        percentiles=pct,
        n=int(arr.size),
    )


# ---------------------------------------------------------------------------
# Chart-ready visualization data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualizationData:
    """Chart-ready arrays derived from a sample set.

    Attributes:
        histogram_counts: Histogram bin counts.
        histogram_edges: Histogram bin edges (length = counts + 1).
        cdf_x: Sorted sample values for the empirical CDF.
        cdf_y: Cumulative probabilities (0, 1].
        survival_x: Sorted sample values for the survival curve.
        survival_y: Survival probabilities ``1 - CDF``.
        risk_curve_x: Horizon grid for the risk curve.
        risk_curve_y: ``P(value <= horizon)`` over the grid.
    """

    histogram_counts: tuple[int, ...]
    histogram_edges:  tuple[float, ...]
    cdf_x:            tuple[float, ...]
    cdf_y:            tuple[float, ...]
    survival_x:      tuple[float, ...]
    survival_y:      tuple[float, ...]
    risk_curve_x:    tuple[float, ...]
    risk_curve_y:    tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary of chart-ready arrays as lists.
        """
        return {
            "histogram_counts": list(self.histogram_counts),
            "histogram_edges": [float(e) for e in self.histogram_edges],
            "cdf_x": [float(v) for v in self.cdf_x],
            "cdf_y": [float(v) for v in self.cdf_y],
            "survival_x": [float(v) for v in self.survival_x],
            "survival_y": [float(v) for v in self.survival_y],
            "risk_curve_x": [float(v) for v in self.risk_curve_x],
            "risk_curve_y": [float(v) for v in self.risk_curve_y],
        }


def build_visualization(
    samples: "np.ndarray", *, bins: int = 30, risk_points: int = 40,
    cdf_max_points: int = 200,
) -> VisualizationData:
    """Build chart-ready arrays (histogram, CDF, survival, risk curve).

    Args:
        samples: The samples ``(n,)``.
        bins: Number of histogram bins.
        risk_points: Number of points on the risk-curve horizon grid.
        cdf_max_points: Maximum points retained for the CDF/survival curves
            (the empirical curve is subsampled to bound output size).

    Returns:
        A :class:`VisualizationData`.

    Raises:
        ValueError: When *samples* is empty.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("build_visualization requires a non-empty sample array")

    counts, edges = np.histogram(arr, bins=bins)

    xs = np.sort(arr)
    cdf = np.arange(1, xs.size + 1, dtype=float) / xs.size
    # Subsample the CDF to bound payload size while preserving the shape.
    if xs.size > cdf_max_points:
        idx = np.linspace(0, xs.size - 1, cdf_max_points).astype(int)
        cdf_x = xs[idx]
        cdf_y = cdf[idx]
    else:
        cdf_x, cdf_y = xs, cdf
    surv_y = 1.0 - cdf_y

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, risk_points)
    risk_y = np.array([float(np.mean(arr <= g)) for g in grid])

    return VisualizationData(
        histogram_counts=tuple(int(c) for c in counts),
        histogram_edges=tuple(float(e) for e in edges),
        cdf_x=tuple(float(v) for v in cdf_x),
        cdf_y=tuple(float(v) for v in cdf_y),
        survival_x=tuple(float(v) for v in cdf_x),
        survival_y=tuple(float(v) for v in surv_y),
        risk_curve_x=tuple(float(v) for v in grid),
        risk_curve_y=tuple(float(v) for v in risk_y),
    )


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def value_at_risk(losses: "np.ndarray", confidence: float = 95.0) -> float:
    """Return the Value-at-Risk of a loss distribution.

    VaR at confidence *c* is the loss not exceeded with probability ``c/100`` —
    the ``c``-th percentile of the loss distribution.

    Args:
        losses: Loss samples ``(n,)`` (larger = worse).
        confidence: Confidence level in ``(0, 100)``.

    Returns:
        The VaR.

    Raises:
        ValueError: When *losses* is empty or *confidence* is out of range.
    """
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        raise ValueError("value_at_risk requires a non-empty array")
    if not (0.0 < confidence < 100.0):
        raise ValueError("confidence must be in (0, 100)")
    return float(np.percentile(arr, confidence))


def conditional_value_at_risk(
    losses: "np.ndarray", confidence: float = 95.0
) -> float:
    """Return the Conditional VaR (expected shortfall) of a loss distribution.

    CVaR is the mean of the losses at or beyond the VaR — the expected loss in
    the worst ``(100 - confidence)%`` of outcomes.  It is always ``>= VaR``.

    Args:
        losses: Loss samples ``(n,)`` (larger = worse).
        confidence: Confidence level in ``(0, 100)``.

    Returns:
        The CVaR.

    Raises:
        ValueError: When *losses* is empty or *confidence* is out of range.
    """
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        raise ValueError("conditional_value_at_risk requires a non-empty array")
    if not (0.0 < confidence < 100.0):
        raise ValueError("confidence must be in (0, 100)")
    var = float(np.percentile(arr, confidence))
    tail = arr[arr >= var]
    if tail.size == 0:
        return var
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# Distribution outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RULDistribution:
    """Monte Carlo distribution of remaining useful life.

    Attributes:
        statistics: Summary statistics of the RUL samples.
        probability_of_failure: Fraction of trials that fail within the horizon.
        expected_failure_time: Mean failure time over trials that fail
            (``NaN`` when no trial fails).
        visualization: Chart-ready arrays for the RUL distribution.
        trials: Raw per-trial RUL samples (only when ``store_trials``).
    """

    statistics:             DistributionStatistics
    probability_of_failure: float
    expected_failure_time:  float
    visualization:          VisualizationData
    trials:                 tuple[float, ...] | None

    # Convenience percentile accessors (the headline P10/P50/P90).
    @property
    def p10(self) -> float:
        """The 10th-percentile RUL."""
        return self.statistics.p10

    @property
    def p50(self) -> float:
        """The median RUL."""
        return self.statistics.p50

    @property
    def p90(self) -> float:
        """The 90th-percentile RUL."""
        return self.statistics.p90

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "statistics": self.statistics.to_dict(),
            "probability_of_failure": _jsonsafe(self.probability_of_failure),
            "expected_failure_time": _jsonsafe(self.expected_failure_time),
            "visualization": self.visualization.to_dict(),
            "trials": (None if self.trials is None
                       else [float(v) for v in self.trials]),
        }


@dataclass(frozen=True)
class HealthDistribution:
    """Monte Carlo distribution of terminal (end-of-horizon) health.

    Attributes:
        statistics: Summary statistics of the terminal-health samples.
        visualization: Chart-ready arrays.
        trials: Raw per-trial terminal-health samples (only when ``store_trials``).
    """

    statistics:    DistributionStatistics
    visualization: VisualizationData
    trials:        tuple[float, ...] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "statistics": self.statistics.to_dict(),
            "visualization": self.visualization.to_dict(),
            "trials": (None if self.trials is None
                       else [float(v) for v in self.trials]),
        }


@dataclass(frozen=True)
class RiskDistribution:
    """Monte Carlo failure-risk and loss distribution.

    Attributes:
        probability_of_failure: Fraction of trials failing within the horizon.
        expected_failure_time: Mean failure time over failing trials (``NaN`` if
            none fail).
        expected_downtime: Expected downtime hours (P(fail) × downtime/failure).
        expected_loss: Expected monetary loss (P(fail) × cost/failure).
        value_at_risk: VaR of the per-trial loss distribution.
        conditional_value_at_risk: CVaR (expected shortfall) of the losses.
        var_confidence: The confidence level used for VaR/CVaR.
        loss_statistics: Summary statistics of the per-trial loss distribution.
        visualization: Chart-ready arrays for the loss distribution.
    """

    probability_of_failure:     float
    expected_failure_time:      float
    expected_downtime:          float
    expected_loss:              float
    value_at_risk:              float
    conditional_value_at_risk:  float
    var_confidence:             float
    loss_statistics:            DistributionStatistics
    visualization:              VisualizationData

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "probability_of_failure": _jsonsafe(self.probability_of_failure),
            "expected_failure_time": _jsonsafe(self.expected_failure_time),
            "expected_downtime": _jsonsafe(self.expected_downtime),
            "expected_loss": _jsonsafe(self.expected_loss),
            "value_at_risk": _jsonsafe(self.value_at_risk),
            "conditional_value_at_risk": _jsonsafe(self.conditional_value_at_risk),
            "var_confidence": _jsonsafe(self.var_confidence),
            "loss_statistics": self.loss_statistics.to_dict(),
            "visualization": self.visualization.to_dict(),
        }


@dataclass(frozen=True)
class PortfolioDistribution:
    """Monte Carlo distribution of fleet-level risk.

    Attributes:
        n_assets: Number of assets in the fleet.
        portfolio_risk: Mean per-asset failure probability across the fleet.
        worst_asset_probability: The highest per-asset failure probability.
        worst_asset_index: The index of the worst asset.
        expected_fleet_failures: Expected number of fleet failures
            (sum of per-asset failure probabilities).
        expected_fleet_failures_statistics: Distribution of the fleet failure
            *count* across trials (each trial sums the per-asset failure
            indicators).
        risk_concentration: Herfindahl-style concentration of failure
            probability across assets (1 = all risk in one asset, 1/N = uniform).
        per_asset_probability: Per-asset failure probabilities.
        visualization: Chart-ready arrays of the fleet-failure-count distribution.
    """

    n_assets:                            int
    portfolio_risk:                      float
    worst_asset_probability:             float
    worst_asset_index:                   int
    expected_fleet_failures:             float
    expected_fleet_failures_statistics:  DistributionStatistics
    risk_concentration:                  float
    per_asset_probability:               tuple[float, ...]
    visualization:                       VisualizationData

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "n_assets": self.n_assets,
            "portfolio_risk": _jsonsafe(self.portfolio_risk),
            "worst_asset_probability": _jsonsafe(self.worst_asset_probability),
            "worst_asset_index": self.worst_asset_index,
            "expected_fleet_failures": _jsonsafe(self.expected_fleet_failures),
            "expected_fleet_failures_statistics":
                self.expected_fleet_failures_statistics.to_dict(),
            "risk_concentration": _jsonsafe(self.risk_concentration),
            "per_asset_probability": [float(p) for p in self.per_asset_probability],
            "visualization": self.visualization.to_dict(),
        }


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------


@register_monte_carlo_engine(ENGINE_NAME)
class MonteCarloEngine:
    """Quantifies prognostic uncertainty by Monte Carlo over the frozen simulator.

    Each trial samples a degradation rate, a noise level, and a failure threshold
    from the configured uncertainty samplers, runs the frozen
    :class:`AssetStateSimulator` with those parameters, and records the resulting
    RUL, terminal health, and failure outcome.  The ensemble is aggregated into
    distributions, risk metrics, and chart-ready arrays.

    Per-trial seeds derive deterministically from the master seed via a
    :class:`numpy.random.SeedSequence`, so an entire ensemble is reproducible.

    Args:
        config: The Monte Carlo configuration.
        rate_sampler: Degradation-rate sampler (defaults to a degenerate sampler
            centred on the asset's configured rate).
        noise_sampler: Noise sampler (defaults to the asset's configured noise).
        threshold_sampler: Failure-threshold sampler (defaults to the asset's
            configured threshold).
        experiment_tracker: Optional tracker for logging ensemble summaries.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: MonteCarloConfig | None = None,
        rate_sampler: DegradationRateSampler | None = None,
        noise_sampler: NoiseSampler | None = None,
        threshold_sampler: FailureThresholdSampler | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or MonteCarloConfig()
        self.rate_sampler = rate_sampler
        self.noise_sampler = noise_sampler
        self.threshold_sampler = threshold_sampler
        self.tracker = experiment_tracker
        self._n_runs = 0
        logger.info(
            "MonteCarloEngine ready | trials=%d | seed=%d",
            self.config.n_trials, self.config.random_seed,
        )

    # ------------------------------------------------------------------
    # Trial sampling
    # ------------------------------------------------------------------

    def _trial_seeds(self) -> "np.ndarray":
        """Derive deterministic per-trial seeds from the master seed.

        Returns:
            Array ``(n_trials,)`` of uint32 seeds.
        """
        ss = np.random.SeedSequence(self.config.random_seed)
        children = ss.spawn(self.config.n_trials)
        return np.array([c.generate_state(1)[0] for c in children])

    def _sample_params(
        self, base: AssetSimulatorConfig
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Sample per-trial degradation rate, noise, and threshold.

        Defaults (degenerate) reproduce the asset's configured values when a
        sampler is not supplied, so an engine with no samplers degenerates to a
        deterministic ensemble around the baseline.

        Args:
            base: The asset's baseline configuration.

        Returns:
            Tuple of arrays ``(rates, noises, thresholds)`` each ``(n_trials,)``.
        """
        n = self.config.n_trials
        param_rng = np.random.default_rng(self.config.random_seed)
        if self.rate_sampler is not None:
            rates = self.rate_sampler.sample(param_rng, n)
        else:
            rates = np.full(n, base.degradation_rate, dtype=float)
        if self.noise_sampler is not None:
            noises = self.noise_sampler.sample(param_rng, n)
        else:
            noises = np.full(n, base.noise_std, dtype=float)
        if self.threshold_sampler is not None:
            thresholds = self.threshold_sampler.sample(param_rng, n)
        else:
            thresholds = np.full(n, base.failure_threshold, dtype=float)
        return rates, noises, thresholds

    def _run_trials(
        self, base: AssetSimulatorConfig
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Run all trials and collect RUL, terminal health, and failure flags.

        Args:
            base: The asset's baseline configuration.

        Returns:
            Tuple ``(rul, terminal_health, failure_time)`` each ``(n_trials,)``;
            ``rul`` is the failure time, or the horizon when the trial survives;
            ``failure_time`` is the crossing time or ``-1`` when no failure.
        """
        import dataclasses

        n = self.config.n_trials
        rates, noises, thresholds = self._sample_params(base)
        seeds = self._trial_seeds()

        rul = np.empty(n, dtype=float)
        terminal = np.empty(n, dtype=float)
        fail_time = np.empty(n, dtype=float)

        for i in range(n):
            cfg = dataclasses.replace(
                base,
                degradation_rate=float(rates[i]),
                noise_std=float(noises[i]),
                failure_threshold=float(thresholds[i]),
                random_seed=int(seeds[i]),
            )
            result = AssetStateSimulator(cfg).simulate()
            terminal[i] = result.final_health
            if result.crossed_failure:
                fail_time[i] = result.failure_time
                rul[i] = result.failure_time
            else:
                fail_time[i] = _NO_FAILURE
                rul[i] = base.horizon  # censored at the horizon
        return rul, terminal, fail_time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_rul_uncertainty(self, base: AssetSimulatorConfig) -> RULDistribution:
        """Quantify remaining-useful-life uncertainty for one asset.

        Args:
            base: The asset's baseline configuration.

        Returns:
            A :class:`RULDistribution` with percentiles, failure probability,
            expected failure time, and chart-ready arrays.
        """
        rul, _, fail_time = self._run_trials(base)
        stats = compute_statistics(rul, self.config.confidence_levels)
        failed = fail_time >= 0
        p_fail = float(np.mean(failed))
        exp_fail_time = (float(np.mean(fail_time[failed]))
                         if failed.any() else float("nan"))
        viz = build_visualization(rul, bins=self.config.histogram_bins)
        out = RULDistribution(
            statistics=stats,
            probability_of_failure=p_fail,
            expected_failure_time=exp_fail_time,
            visualization=viz,
            trials=(tuple(float(v) for v in rul)
                    if self.config.store_trials else None),
        )
        self._log_summary("rul", {"p50": stats.p50, "p10": stats.p10,
                                  "p90": stats.p90, "p_fail": p_fail})
        return out

    def run_failure_probability(
        self, base: AssetSimulatorConfig, *, var_confidence: float = 95.0
    ) -> RiskDistribution:
        """Quantify failure-risk and loss uncertainty for one asset.

        Each trial contributes a loss of ``failure_cost`` if it fails within the
        horizon and ``0`` otherwise; VaR and CVaR are computed over that loss
        distribution.

        Args:
            base: The asset's baseline configuration.
            var_confidence: Confidence level for VaR/CVaR (in ``(0, 100)``).

        Returns:
            A :class:`RiskDistribution`.
        """
        _, _, fail_time = self._run_trials(base)
        failed = fail_time >= 0
        p_fail = float(np.mean(failed))
        exp_fail_time = (float(np.mean(fail_time[failed]))
                         if failed.any() else float("nan"))
        losses = np.where(failed, self.config.failure_cost, 0.0)
        exp_loss = float(np.mean(losses))
        exp_downtime = p_fail * self.config.failure_downtime_h
        var = value_at_risk(losses, var_confidence)
        cvar = conditional_value_at_risk(losses, var_confidence)
        stats = compute_statistics(losses, self.config.confidence_levels)
        viz = build_visualization(losses, bins=self.config.histogram_bins)
        out = RiskDistribution(
            probability_of_failure=p_fail,
            expected_failure_time=exp_fail_time,
            expected_downtime=exp_downtime,
            expected_loss=exp_loss,
            value_at_risk=var,
            conditional_value_at_risk=cvar,
            var_confidence=float(var_confidence),
            loss_statistics=stats,
            visualization=viz,
        )
        self._log_summary("risk", {"p_fail": p_fail, "expected_loss": exp_loss,
                                   "var": var, "cvar": cvar})
        return out

    def run_health_distribution(
        self, base: AssetSimulatorConfig
    ) -> HealthDistribution:
        """Quantify terminal-health uncertainty for one asset.

        Args:
            base: The asset's baseline configuration.

        Returns:
            A :class:`HealthDistribution`.
        """
        _, terminal, _ = self._run_trials(base)
        stats = compute_statistics(terminal, self.config.confidence_levels)
        viz = build_visualization(terminal, bins=self.config.histogram_bins)
        out = HealthDistribution(
            statistics=stats,
            visualization=viz,
            trials=(tuple(float(v) for v in terminal)
                    if self.config.store_trials else None),
        )
        self._log_summary("health", {"p50": stats.p50, "mean": stats.mean})
        return out

    def run_portfolio_distribution(
        self, bases: Sequence[AssetSimulatorConfig]
    ) -> PortfolioDistribution:
        """Quantify fleet-level risk uncertainty across multiple assets.

        Each asset is evaluated with its own Monte Carlo ensemble; the per-asset
        failure probabilities are aggregated into fleet risk, the worst-asset
        probability, the expected fleet-failure count and its distribution, and a
        Herfindahl-style risk-concentration index.

        Args:
            bases: The baseline configurations of the fleet assets.

        Returns:
            A :class:`PortfolioDistribution`.

        Raises:
            ValueError: When *bases* is empty.
        """
        if not bases:
            raise ValueError("run_portfolio_distribution requires >= 1 asset")
        n_assets = len(bases)
        n = self.config.n_trials

        # Per-asset failure indicator matrix (n_trials x n_assets).
        fail_matrix = np.empty((n, n_assets), dtype=float)
        per_asset_p = np.empty(n_assets, dtype=float)
        for j, base in enumerate(bases):
            _, _, fail_time = self._run_trials(base)
            failed = (fail_time >= 0).astype(float)
            fail_matrix[:, j] = failed
            per_asset_p[j] = float(np.mean(failed))

        portfolio_risk = float(np.mean(per_asset_p))
        worst_idx = int(np.argmax(per_asset_p))
        worst_p = float(per_asset_p[worst_idx])
        expected_failures = float(np.sum(per_asset_p))

        # Distribution of the fleet failure count across trials.
        per_trial_counts = fail_matrix.sum(axis=1)
        count_stats = compute_statistics(per_trial_counts,
                                         self.config.confidence_levels)

        # Risk concentration (Herfindahl index of normalised failure prob).
        total_p = per_asset_p.sum()
        if total_p > _EPS:
            shares = per_asset_p / total_p
            concentration = float(np.sum(shares ** 2))
        else:
            concentration = 0.0

        viz = build_visualization(per_trial_counts,
                                  bins=min(self.config.histogram_bins,
                                           max(1, n_assets + 1)))
        out = PortfolioDistribution(
            n_assets=n_assets,
            portfolio_risk=portfolio_risk,
            worst_asset_probability=worst_p,
            worst_asset_index=worst_idx,
            expected_fleet_failures=expected_failures,
            expected_fleet_failures_statistics=count_stats,
            risk_concentration=concentration,
            per_asset_probability=tuple(float(p) for p in per_asset_p),
            visualization=viz,
        )
        self._log_summary("portfolio",
                          {"portfolio_risk": portfolio_risk,
                           "expected_failures": expected_failures,
                           "worst_p": worst_p})
        return out

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_summary(self, kind: str, metrics: dict[str, float]) -> None:
        """Log an ensemble summary to the experiment tracker (failure-safe).

        Args:
            kind: A short label for the run kind.
            metrics: The metrics to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {f"mc_{kind}_{k}": float(v) for k, v in metrics.items()},
                step=self._n_runs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"mc_run_kind": kind})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)
        self._n_runs += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short Monte Carlo demo over a single asset and a small fleet.

    Returns:
        Exit code 0.
    """
    base = AssetSimulatorConfig(
        horizon=120, timestep=1.0, model="linear", degradation_rate=0.8,
        initial_health=95.0, failure_threshold=30.0, noise_std=0.5,
    )
    engine = MonteCarloEngine(
        MonteCarloConfig(n_trials=2000, random_seed=7),
        rate_sampler=DegradationRateSampler("normal", mean=0.8, std=0.2),
        noise_sampler=NoiseSampler("uniform", low=0.0, high=1.5),
        threshold_sampler=FailureThresholdSampler("triangular",
                                                  low=25, mode=30, high=35),
    )
    rul = engine.run_rul_uncertainty(base)
    logger.info("RUL  P10=%.1f  P50=%.1f  P90=%.1f  P(fail)=%.3f",
                rul.p10, rul.p50, rul.p90, rul.probability_of_failure)
    risk = engine.run_failure_probability(base)
    logger.info("RISK  P(fail)=%.3f  E[loss]=%.0f  VaR=%.0f  CVaR=%.0f",
                risk.probability_of_failure, risk.expected_loss,
                risk.value_at_risk, risk.conditional_value_at_risk)
    health = engine.run_health_distribution(base)
    logger.info("HEALTH  P50=%.1f  mean=%.1f",
                health.statistics.p50, health.statistics.mean)
    fleet = [
        AssetSimulatorConfig(horizon=120, model="linear", degradation_rate=r,
                             initial_health=95, failure_threshold=30,
                             noise_std=0.5)
        for r in (0.6, 0.8, 1.2)
    ]
    port = engine.run_portfolio_distribution(fleet)
    logger.info("FLEET  risk=%.3f  E[failures]=%.2f  worst_p=%.3f  concentration=%.3f",
                port.portfolio_risk, port.expected_fleet_failures,
                port.worst_asset_probability, port.risk_concentration)
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
    parser = argparse.ArgumentParser(description="Monte Carlo risk engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a Monte Carlo demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered Monte Carlo engines:", list_monte_carlo_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())