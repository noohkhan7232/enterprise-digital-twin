#!/usr/bin/env python3
"""Health-trend analysis for wind turbine acoustic predictive maintenance.

This module is the trend-analysis stage of the Week-5 predictive-maintenance
framework.  It sits **between** the health index (Step 1) and the RUL predictor
(Step 2): before extrapolating a trajectory to failure, a platform needs to
understand the *shape* of that trajectory — is health steady, drifting,
declining, or accelerating toward collapse? — and whether a **regime change**
(a sudden shift in behaviour, e.g. the onset of a fault) has occurred.  Those
answers gate and contextualise the RUL estimate: extrapolating across a regime
change, or trusting a trend with no statistical support, produces dangerously
wrong remaining-life numbers.

The analyzer consumes a health-index trajectory (the smoothed history from
:class:`~src.predictive.health_index.HealthIndexEngine`) and produces a
:class:`HealthTrendResult` containing a robust trend estimate, a trend
classification, detected change points, an operational early-warning level, and
calibrated confidence scores — all in pure NumPy, with no SciPy dependency.

Trend estimation
----------------
Three complementary estimators, because each fails differently:

* **Linear (OLS) slope** — the maximum-likelihood slope under Gaussian noise;
  efficient but sensitive to outliers and dropouts.
* **Moving-average slope** — the slope of a smoothed trajectory; suppresses
  per-sample noise before estimating direction.
* **Theil-Sen (robust) slope** — the median of all pairwise slopes; resistant to
  up to ~29% contaminated points, so a denoising artefact or a single corrupt
  clip cannot swing the verdict.  This is the slope the classification uses,
  because in the field the data is never clean.

Trend classification
--------------------
The robust slope and the trajectory's curvature (second derivative from a
quadratic fit) jointly determine one of four states:

* ``IMPROVING`` — health rising beyond the dead band.
* ``STABLE`` — slope within the dead band; no meaningful change.
* ``DEGRADING`` — health falling at a roughly constant rate.
* ``ACCELERATING`` — health falling *and* the decline is steepening (negative
  curvature), the most urgent pattern because RUL is collapsing faster than a
  linear extrapolation implies.

Change-point detection
----------------------
A sliding-window mean-shift scan (a CUSUM-style statistic) locates the index
that best splits the trajectory into two regimes of differing mean, returns the
**magnitude** of the shift, and scores its significance against the trajectory's
own scatter.  Detecting the onset of a fault is often more actionable than the
slow trend it triggers.

Early warning
-------------
The current health level, the trend class, and any change point are fused into a
four-level warning (``NORMAL`` / ``WATCH`` / ``WARNING`` / ``CRITICAL``) — the
single triage signal an operator acts on.

Confidence scoring
-----------------
* ``trend_confidence`` — how well a line explains the trajectory (R², adjusted
  for sample size); a noisy or short history yields low confidence even with a
  steep slope.
* ``change_confidence`` — the detected shift magnitude relative to the
  trajectory's noise; a small wiggle in a noisy signal scores low.

Architecture
------------
* :class:`HealthTrendConfig` — frozen, validated configuration.
* :class:`HealthTrendResult` — the immutable analysis result.
* :class:`TrendDirection` / :class:`EarlyWarning` — classification enums.
* :class:`HealthTrendAnalyzer` — the analyzer; registry- and tracker-integrated,
  with helpers to analyse a :class:`HealthIndexEngine` directly and to gate a
  downstream RUL prediction.

Usage::

    from src.predictive.health_trend_analyzer import (
        HealthTrendAnalyzer, HealthTrendConfig,
    )

    analyzer = HealthTrendAnalyzer(HealthTrendConfig())
    result = analyzer.analyze(health_engine.history(smoothed=True))

    print(result.trend.value, result.warning.value)
    print(result.robust_slope, result.trend_confidence)
    if result.has_change_point:
        print(result.change_index, result.change_magnitude)

CLI::

    python src/predictive/health_trend_analyzer.py --demo
"""

from __future__ import annotations

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

logger = logging.getLogger("health_trend_analyzer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named trend analyzers.
HEALTH_TREND_REGISTRY: dict[str, type] = {}

ANALYZER_NAME: Final[str] = "acoustic_health_trend"

#: Maximum points used for exact Theil-Sen before pairwise subsampling kicks in.
_THEIL_SEN_EXACT_CAP: Final[int] = 300


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    """Qualitative direction of a health trajectory."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    ACCELERATING = "accelerating"


class EarlyWarning(str, Enum):
    """Operational early-warning levels, least to most severe."""

    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_health_trend_analyzer(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a health-trend analyzer by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = HEALTH_TREND_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Health-trend analyzer '{name}' already registered to "
                f"{existing.__name__}"
            )
        HEALTH_TREND_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered health-trend analyzer '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_health_trend_analyzer(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered health-trend analyzer by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the analyzer constructor.

    Returns:
        An instantiated analyzer.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in HEALTH_TREND_REGISTRY:
        available = ", ".join(sorted(HEALTH_TREND_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown health-trend analyzer '{name}'. Available: {available}"
        )
    return HEALTH_TREND_REGISTRY[name](**kwargs)


def list_health_trend_analyzers() -> list[str]:
    """Return the sorted names of registered health-trend analyzers.

    Returns:
        Sorted registry keys.
    """
    return sorted(HEALTH_TREND_REGISTRY)


# ---------------------------------------------------------------------------
# Pure estimators (NumPy only)
# ---------------------------------------------------------------------------


def linear_slope(y: "np.ndarray") -> tuple[float, float, float]:
    """Ordinary-least-squares slope, intercept, and R² of ``y`` over its index.

    Args:
        y: Health values ``(n,)``.

    Returns:
        Tuple ``(slope, intercept, r_squared)``.
    """
    n = y.size
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0, 0.0
    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, y, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = np.polyval(coef, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, intercept, max(0.0, min(1.0, r2))


def moving_average(y: "np.ndarray", window: int) -> "np.ndarray":
    """Centered simple moving average.

    Args:
        y: Input series ``(n,)``.
        window: Window length (``>= 1``); ``1`` returns the input unchanged.

    Returns:
        The smoothed series ``(max(1, n - window + 1),)``.
    """
    if window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="valid")


def moving_average_slope(y: "np.ndarray", window: int) -> float:
    """Slope of the moving-average-smoothed series.

    Args:
        y: Health values ``(n,)``.
        window: Moving-average window.

    Returns:
        The OLS slope of the smoothed series.
    """
    smoothed = moving_average(y, window)
    slope, _, _ = linear_slope(smoothed)
    return slope


def theil_sen_slope(y: "np.ndarray", *, rng_seed: int = 0) -> float:
    """Theil-Sen robust slope (median of pairwise slopes).

    For ``n <= _THEIL_SEN_EXACT_CAP`` all pairwise slopes are used; for larger
    series a deterministic random subsample of pairs is used to bound cost at
    ``O(n)`` while preserving the breakdown-point robustness.

    Args:
        y: Health values ``(n,)``.
        rng_seed: Seed for deterministic pair subsampling on long series.

    Returns:
        The robust slope estimate.
    """
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    if n <= _THEIL_SEN_EXACT_CAP:
        i, j = np.triu_indices(n, k=1)
        slopes = (y[j] - y[i]) / (x[j] - x[i])
    else:  # pragma: no cover - long-series path
        rng = np.random.default_rng(rng_seed)
        m = n * 20
        i = rng.integers(0, n, size=m)
        j = rng.integers(0, n, size=m)
        valid = i != j
        i, j = i[valid], j[valid]
        slopes = (y[j] - y[i]) / (x[j] - x[i])
    return float(np.median(slopes))


def quadratic_curvature(y: "np.ndarray") -> float:
    """Second derivative of a quadratic fit (trajectory curvature).

    Negative curvature on a falling trajectory indicates an *accelerating*
    decline; positive curvature indicates a decelerating one.

    Args:
        y: Health values ``(n,)``.

    Returns:
        The second derivative ``2a`` of the fitted ``a·x² + b·x + c``.
    """
    n = y.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    a, _, _ = np.polyfit(x, y, 2)
    return float(2.0 * a)


def detect_change_point(
    y: "np.ndarray", min_segment: int
) -> tuple[int, float, float]:
    """Locate the most significant mean-shift change point.

    Scans every split that leaves at least ``min_segment`` points on each side
    and selects the one maximising the absolute difference of segment means
    (a CUSUM-style statistic).  Significance is the shift magnitude normalised by
    the pooled within-segment standard deviation.

    Args:
        y: Health values ``(n,)``.
        min_segment: Minimum points required in each segment.

    Returns:
        Tuple ``(index, magnitude, significance)``.  ``index`` is ``-1`` when no
        split is possible; *magnitude* is the absolute mean shift; *significance*
        is the magnitude in pooled-standard-deviation units.
    """
    n = y.size
    if n < 2 * min_segment:
        return -1, 0.0, 0.0

    best_idx = -1
    best_mag = 0.0
    best_sig = 0.0
    for k in range(min_segment, n - min_segment + 1):
        left, right = y[:k], y[k:]
        mag = abs(float(right.mean()) - float(left.mean()))
        # Pooled standard deviation of the two segments.
        var_l = float(left.var(ddof=1)) if left.size > 1 else 0.0
        var_r = float(right.var(ddof=1)) if right.size > 1 else 0.0
        pooled = math.sqrt(max(1e-12, (var_l + var_r) / 2.0))
        sig = mag / pooled
        if mag > best_mag:
            best_mag = mag
            best_idx = k
            best_sig = sig
    return best_idx, best_mag, best_sig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthTrendConfig:
    """Configuration for the :class:`HealthTrendAnalyzer`.

    Attributes:
        min_history: Minimum trajectory points required to analyse.
        slope_dead_band: Absolute slope (health per step) below which the trend
            is STABLE rather than improving/degrading.
        accel_curvature_threshold: Curvature (second derivative) below which a
            falling trend is classified ACCELERATING.
        ma_window: Moving-average window for the moving-average slope.
        change_min_segment: Minimum points each side of a change point.
        change_significance_threshold: Pooled-σ units a shift must exceed to
            count as a real change point.
        warn_health: Current-health level at/below which warning escalates.
        critical_health: Current-health level at/below which warning is CRITICAL.
        watch_slope: Degradation slope magnitude at/above which to at least WATCH.
        warning_slope: Degradation slope magnitude at/above which to WARN.
        nan_policy: How to treat NaNs: ``"interpolate"`` (default), ``"drop"``,
            or ``"raise"``.
    """

    min_history:                   int = 4
    slope_dead_band:               float = 0.1
    accel_curvature_threshold:     float = -0.05
    ma_window:                     int = 3
    change_min_segment:            int = 3
    change_significance_threshold: float = 1.5
    warn_health:                   float = 60.0
    critical_health:               float = 40.0
    watch_slope:                   float = 0.3
    warning_slope:                 float = 1.0
    nan_policy:                    str = "interpolate"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_history < 3:
            raise ValueError("min_history must be >= 3")
        if self.slope_dead_band < 0:
            raise ValueError("slope_dead_band must be >= 0")
        if self.accel_curvature_threshold > 0:
            raise ValueError("accel_curvature_threshold must be <= 0")
        if self.ma_window < 1:
            raise ValueError("ma_window must be >= 1")
        if self.change_min_segment < 1:
            raise ValueError("change_min_segment must be >= 1")
        if self.change_significance_threshold < 0:
            raise ValueError("change_significance_threshold must be >= 0")
        if not (0.0 <= self.critical_health <= self.warn_health <= 100.0):
            raise ValueError("require 0 <= critical_health <= warn_health <= 100")
        if self.watch_slope < 0 or self.warning_slope < 0:
            raise ValueError("slope thresholds must be >= 0")
        if self.watch_slope > self.warning_slope:
            raise ValueError("watch_slope must be <= warning_slope")
        if self.nan_policy not in {"interpolate", "drop", "raise"}:
            raise ValueError("nan_policy must be interpolate|drop|raise")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthTrendResult:
    """The result of one health-trend analysis.

    Attributes:
        trend: The qualitative :class:`TrendDirection`.
        warning: The :class:`EarlyWarning` level.
        linear_slope: OLS slope (health per step).
        moving_average_slope: Slope of the moving-average-smoothed series.
        robust_slope: Theil-Sen robust slope (used for classification).
        curvature: Second derivative from a quadratic fit.
        current_health: The most recent health value.
        mean_health: Mean of the analysed trajectory.
        trend_confidence: Confidence in the trend direction ``[0, 1]``.
        change_index: Index of the detected change point (``-1`` if none).
        change_magnitude: Absolute mean shift at the change point.
        change_significance: Shift magnitude in pooled-σ units.
        change_confidence: Confidence that a change point is real ``[0, 1]``.
        has_change_point: Whether a significant change point was found.
        n_observations: Number of points analysed (after NaN handling).
        warnings: Non-fatal diagnostics.
    """

    trend:                TrendDirection
    warning:              EarlyWarning
    linear_slope:         float
    moving_average_slope: float
    robust_slope:         float
    curvature:            float
    current_health:       float
    mean_health:          float
    trend_confidence:     float
    change_index:         int = -1
    change_magnitude:     float = 0.0
    change_significance:  float = 0.0
    change_confidence:    float = 0.0
    has_change_point:     bool = False
    n_observations:       int = 0
    warnings:             list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "trend": self.trend.value,
            "warning": self.warning.value,
            "linear_slope": self.linear_slope,
            "moving_average_slope": self.moving_average_slope,
            "robust_slope": self.robust_slope,
            "curvature": self.curvature,
            "current_health": self.current_health,
            "mean_health": self.mean_health,
            "trend_confidence": self.trend_confidence,
            "change_index": self.change_index,
            "change_magnitude": self.change_magnitude,
            "change_significance": self.change_significance,
            "change_confidence": self.change_confidence,
            "has_change_point": self.has_change_point,
            "n_observations": self.n_observations,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# HealthTrendAnalyzer
# ---------------------------------------------------------------------------


@register_health_trend_analyzer(ANALYZER_NAME)
class HealthTrendAnalyzer:
    """Analyses health-index trajectories for degradation patterns.

    Estimates the trend with three complementary methods, classifies it,
    detects change points, generates an early-warning level, and scores its own
    confidence.  Registry- and tracker-integrated.

    Args:
        config: The analyzer configuration.
        experiment_tracker: Optional tracker for logging analyses.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: HealthTrendConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or HealthTrendConfig()
        self.tracker = experiment_tracker
        self._n_analyses = 0
        logger.info(
            "HealthTrendAnalyzer ready | min_history=%d | dead_band=%.2f",
            self.config.min_history, self.config.slope_dead_band,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, trajectory: Sequence[float] | "np.ndarray"
    ) -> HealthTrendResult:
        """Analyse a health trajectory.

        Args:
            trajectory: Health-index values in chronological order (oldest
                first).

        Returns:
            A :class:`HealthTrendResult`.

        Raises:
            ValueError: When the trajectory is too short after NaN handling, is
                not one-dimensional, or contains NaNs under ``nan_policy="raise"``.
        """
        y_raw = np.asarray(trajectory, dtype=float)
        if y_raw.ndim != 1:
            raise ValueError("trajectory must be one-dimensional")

        warnings: list[str] = []
        y = self._handle_nans(y_raw, warnings)

        if y.size < self.config.min_history:
            raise ValueError(
                f"trajectory has {y.size} usable points; need at least "
                f"{self.config.min_history}"
            )

        # --- Trend estimation (three methods) ---
        lin_slope, _, r2 = linear_slope(y)
        ma_slope = moving_average_slope(y, self.config.ma_window)
        robust = theil_sen_slope(y)
        curvature = quadratic_curvature(y)
        current_health = float(y[-1])
        mean_health = float(y.mean())

        # --- Trend classification (robust slope + curvature) ---
        trend = self._classify_trend(robust, curvature)

        # --- Trend confidence (R² shrunk for short series) ---
        trend_confidence = self._trend_confidence(r2, y.size)

        # --- Change-point detection ---
        ci, mag, sig = detect_change_point(y, self.config.change_min_segment)
        has_change = (
            ci >= 0 and sig >= self.config.change_significance_threshold
        )
        change_confidence = self._change_confidence(sig) if ci >= 0 else 0.0

        # --- Early warning ---
        warning = self._early_warning(
            current_health, trend, robust, has_change,
        )

        result = HealthTrendResult(
            trend=trend,
            warning=warning,
            linear_slope=lin_slope,
            moving_average_slope=ma_slope,
            robust_slope=robust,
            curvature=curvature,
            current_health=current_health,
            mean_health=mean_health,
            trend_confidence=trend_confidence,
            change_index=ci if has_change else -1,
            change_magnitude=mag,
            change_significance=sig,
            change_confidence=change_confidence,
            has_change_point=has_change,
            n_observations=int(y.size),
            warnings=warnings,
        )
        self._log_result(result)
        self._n_analyses += 1
        return result

    def analyze_engine(self, health_engine: Any) -> HealthTrendResult:
        """Analyse a Step-1 health engine's smoothed history directly.

        Args:
            health_engine: A
                :class:`~src.predictive.health_index.HealthIndexEngine` (or any
                object exposing ``history(smoothed=True)``).

        Returns:
            A :class:`HealthTrendResult`.
        """
        return self.analyze(health_engine.history(smoothed=True))

    def should_predict_rul(self, result: HealthTrendResult) -> bool:
        """Gate downstream RUL prediction on trend quality.

        A RUL extrapolation is only meaningful when the machine is actually
        degrading with reasonable trend support; this returns ``True`` when a
        RUL prediction is worth computing.

        Args:
            result: A prior :class:`HealthTrendResult`.

        Returns:
            ``True`` when the trajectory is degrading/accelerating with non-
            trivial trend confidence, else ``False``.
        """
        degrading = result.trend in (
            TrendDirection.DEGRADING, TrendDirection.ACCELERATING,
        )
        return bool(degrading and result.trend_confidence >= 0.25)

    # ------------------------------------------------------------------
    # Classification / scoring internals
    # ------------------------------------------------------------------

    def _classify_trend(
        self, slope: float, curvature: float
    ) -> TrendDirection:
        """Classify the trend from the robust slope and curvature.

        Args:
            slope: The robust (Theil-Sen) slope.
            curvature: The trajectory curvature.

        Returns:
            The :class:`TrendDirection`.
        """
        if slope > self.config.slope_dead_band:
            return TrendDirection.IMPROVING
        if slope >= -self.config.slope_dead_band:
            return TrendDirection.STABLE
        # Degrading: distinguish accelerating from steady.
        if curvature <= self.config.accel_curvature_threshold:
            return TrendDirection.ACCELERATING
        return TrendDirection.DEGRADING

    def _trend_confidence(self, r_squared: float, n: int) -> float:
        """Compute trend confidence from R², shrunk for short series.

        Args:
            r_squared: Goodness of fit of the linear trend.
            n: Number of observations.

        Returns:
            Confidence in ``[0, 1]``.
        """
        # Sample-size shrinkage: a high R² on 4 points is less trustworthy than
        # on 40.  Multiply by a saturating factor n / (n + k).
        shrink = n / (n + 6.0)
        return float(max(0.0, min(1.0, r_squared * shrink)))

    def _change_confidence(self, significance: float) -> float:
        """Map a change significance (in σ units) to a ``[0, 1]`` confidence.

        Args:
            significance: Shift magnitude in pooled-σ units.

        Returns:
            Confidence in ``[0, 1]`` via a saturating transform.
        """
        # Logistic-style saturation centred near the significance threshold.
        return float(1.0 - math.exp(-max(0.0, significance) / 2.0))

    def _early_warning(
        self, current_health: float, trend: TrendDirection,
        slope: float, has_change: bool,
    ) -> EarlyWarning:
        """Fuse level, trend, slope, and change into an early-warning level.

        Args:
            current_health: The latest health value.
            trend: The classified trend.
            slope: The robust slope.
            has_change: Whether a significant change point was found.

        Returns:
            The :class:`EarlyWarning` level.
        """
        cfg = self.config
        score = 0

        # Contribution from absolute health level.
        if current_health <= cfg.critical_health:
            score += 3
        elif current_health <= cfg.warn_health:
            score += 2
        elif current_health <= (cfg.warn_health + 100.0) / 2.0:
            score += 1

        # Contribution from trend direction / rate.
        mag = abs(slope)
        if trend == TrendDirection.ACCELERATING:
            score += 3
        elif trend == TrendDirection.DEGRADING:
            if mag >= cfg.warning_slope:
                score += 2
            elif mag >= cfg.watch_slope:
                score += 1

        # A fresh regime change raises vigilance, but only when the machine is
        # not pristine — a mean shift on a high, stable health signal is not by
        # itself a cause for alarm (it may be a benign operating-point change).
        if has_change and current_health <= (cfg.warn_health + 100.0) / 2.0:
            score += 1

        if score >= 5:
            return EarlyWarning.CRITICAL
        if score >= 3:
            return EarlyWarning.WARNING
        if score >= 1:
            return EarlyWarning.WATCH
        return EarlyWarning.NORMAL

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    def _handle_nans(
        self, y: "np.ndarray", warnings: list[str]
    ) -> "np.ndarray":
        """Apply the configured NaN policy to the trajectory.

        Args:
            y: The raw trajectory.
            warnings: Accumulator for diagnostics.

        Returns:
            The cleaned trajectory.

        Raises:
            ValueError: Under ``nan_policy="raise"`` when NaNs are present.
        """
        mask = ~np.isfinite(y)
        if not mask.any():
            return y
        policy = self.config.nan_policy
        if policy == "raise":
            raise ValueError("trajectory contains non-finite values")
        if policy == "drop":
            warnings.append(f"dropped {int(mask.sum())} non-finite values")
            return y[~mask]
        # interpolate (default)
        warnings.append(f"interpolated {int(mask.sum())} non-finite values")
        idx = np.arange(y.size)
        good = ~mask
        if good.sum() == 0:
            raise ValueError("trajectory is entirely non-finite")
        return np.interp(idx, idx[good], y[good])

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_result(self, result: HealthTrendResult) -> None:
        """Log a result to the experiment tracker (failure-safe).

        Args:
            result: The result to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "trend_robust_slope": result.robust_slope,
                    "trend_confidence": result.trend_confidence,
                    "change_magnitude": result.change_magnitude,
                    "change_confidence": result.change_confidence,
                },
                step=self._n_analyses,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic trend-analysis demo.

    Returns:
        Exit code 0.
    """
    rng = np.random.default_rng(0)
    scenarios = {
        "stable": np.full(20, 85.0) + rng.normal(0, 1.0, 20),
        "degrading": 90.0 - 1.5 * np.arange(20) + rng.normal(0, 1.0, 20),
        "accelerating": 95.0 - 0.1 * np.arange(20) ** 2 + rng.normal(0, 1.0, 20),
        "regime_change": np.concatenate([
            np.full(10, 88.0), np.full(10, 55.0)
        ]) + rng.normal(0, 1.0, 20),
    }
    analyzer = HealthTrendAnalyzer(HealthTrendConfig())
    for name, traj in scenarios.items():
        r = analyzer.analyze(traj)
        logger.info(
            "[%-13s] trend=%-12s warning=%-8s slope=%6.2f conf=%.2f change@%d(mag=%.1f)",
            name, r.trend.value, r.warning.value, r.robust_slope,
            r.trend_confidence, r.change_index, r.change_magnitude,
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
    parser = argparse.ArgumentParser(description="Health-trend analyzer")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic trend-analysis demo.")
    parser.add_argument("--list-analyzers", action="store_true")
    args = parser.parse_args(argv)

    if args.list_analyzers:
        print("Registered health-trend analyzers:",
              list_health_trend_analyzers())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())