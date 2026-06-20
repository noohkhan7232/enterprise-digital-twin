#!/usr/bin/env python3
"""Remaining-Useful-Life (RUL) prediction engine for predictive maintenance.

This module is the second stage of the Week-5 predictive-maintenance framework.
It consumes the **health-index trajectories** produced by
:class:`~src.predictive.health_index.HealthIndexEngine` (Step 1) and extrapolates
them to a **failure threshold** to estimate how much useful life a machine has
left — the central quantity an industrial predictive-maintenance platform exists
to produce.

The estimate is only as trustworthy as its uncertainty, so every prediction
carries a confidence interval derived from the statistical uncertainty of the
trajectory fit (not a hand-waved fudge factor).  This is what lets an operator
distinguish "fail in 30 ± 3 days, schedule the crew" from "fail in 30 ± 25 days,
keep watching."

Degradation models
------------------
Two complementary extrapolation models are provided; the right one depends on
the failure physics:

* **Linear** — ``health(t) = a + b·t``.  The natural model for steady,
  constant-rate wear.  The threshold-crossing time has a closed form and its
  uncertainty is propagated analytically.
* **Exponential** — ``health(t) = H0·exp(-k·t)``.  The natural model for
  *accelerating* degradation (fatigue-crack growth, runaway bearing spall),
  fitted as a log-linear regression so the same closed-form machinery applies in
  log-space.

An ``AUTO`` mode fits both and selects the one that explains the observed
trajectory better (higher R²), which is the right default when the failure mode
is unknown a priori.

Uncertainty quantification
--------------------------
The confidence interval on RUL is computed by the **delta method**: the
threshold-crossing time is a differentiable function of the regression
coefficients, whose covariance comes from ordinary least squares, so the
variance of the crossing time follows by first-order error propagation.  The
Student-t critical value (computed in pure NumPy via a Cornish-Fisher expansion,
no SciPy dependency) accounts for the small sample sizes typical of early-life
trajectories.  Wider scatter around the trend, or fewer observations, correctly
widens the interval.

Design properties
-----------------
* **NumPy only** — no SciPy, no PyTorch.  Fully deterministic and testable.
* **Registry- and tracker-integrated**, mirroring the Step-1 engine and the
  model zoo so a platform can construct predictors by name and log predictions.
* **Defensive** — minimum-history validation, non-degrading (healthy/flat)
  trajectories handled explicitly as "no foreseeable failure," and noisy inputs
  surfaced through honest, wide intervals rather than false precision.

Usage::

    from src.predictive.rul_predictor import RULPredictor, RULConfig

    predictor = RULPredictor(RULConfig(failure_threshold=30.0))
    health_curve = engine.history(smoothed=True)   # from Step 1
    prediction = predictor.predict(health_curve)

    print(prediction.rul, prediction.ci_low, prediction.ci_high)
    print(prediction.model_used, prediction.r_squared)

CLI::

    python src/predictive/rul_predictor.py --demo
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

logger = logging.getLogger("rul_predictor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sentinel for "no foreseeable failure" (non-degrading trajectory).
RUL_INFINITE: Final[float] = float("inf")

#: Registry of named RUL predictors (mirrors the health-engine pattern).
RUL_PREDICTOR_REGISTRY: dict[str, type] = {}

PREDICTOR_NAME: Final[str] = "trajectory_rul_predictor"


# ---------------------------------------------------------------------------
# Degradation-model enum
# ---------------------------------------------------------------------------


class DegradationModel(str, Enum):
    """Supported degradation-extrapolation models."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    AUTO = "auto"  # fit both, select by goodness of fit


#: Concrete (non-AUTO) models.
_CONCRETE_MODELS: Final[tuple[str, ...]] = (
    DegradationModel.LINEAR.value, DegradationModel.EXPONENTIAL.value,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_rul_predictor(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a RUL predictor by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = RUL_PREDICTOR_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"RUL predictor '{name}' already registered to {existing.__name__}"
            )
        RUL_PREDICTOR_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered RUL predictor '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_rul_predictor(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered RUL predictor by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the predictor constructor.

    Returns:
        An instantiated predictor.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in RUL_PREDICTOR_REGISTRY:
        available = ", ".join(sorted(RUL_PREDICTOR_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown RUL predictor '{name}'. Available: {available}")
    return RUL_PREDICTOR_REGISTRY[name](**kwargs)


def list_rul_predictors() -> list[str]:
    """Return the sorted names of registered RUL predictors.

    Returns:
        Sorted registry keys.
    """
    return sorted(RUL_PREDICTOR_REGISTRY)


# ---------------------------------------------------------------------------
# Pure statistical helpers (NumPy only)
# ---------------------------------------------------------------------------


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Args:
        p: Probability in ``(0, 1)``.

    Returns:
        The standard-normal quantile.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def student_t_ppf(p: float, dof: int) -> float:
    """Student-t quantile via a Cornish-Fisher expansion (NumPy only).

    Accurate to a few thousandths for ``dof >= 5`` — sufficient for confidence
    intervals — and avoids a SciPy dependency.

    Args:
        p: Probability in ``(0, 1)``.
        dof: Degrees of freedom (``>= 1``).

    Returns:
        The Student-t quantile.
    """
    if dof <= 0:
        raise ValueError("dof must be >= 1")
    z = _normal_ppf(p)
    if dof > 200:
        return z
    g1 = (z**3 + z) / (4 * dof)
    g2 = (5*z**5 + 16*z**3 + 3*z) / (96 * dof**2)
    g3 = (3*z**7 + 19*z**5 + 17*z**3 - 15*z) / (384 * dof**3)
    return z + g1 + g2 + g3


def _ols_fit(
    x: "np.ndarray", y: "np.ndarray"
) -> tuple[float, float, "np.ndarray", float]:
    """Ordinary least-squares fit of ``y = a + b·x`` with covariance and R².

    Args:
        x: Independent variable ``(n,)``.
        y: Dependent variable ``(n,)``.

    Returns:
        Tuple ``(a, b, cov, r_squared)`` where *cov* is the 2×2 coefficient
        covariance matrix (``[[var_a, cov_ab], [cov_ab, var_b]]``).
    """
    n = x.size
    A = np.vstack([np.ones(n), x]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = A @ coef
    resid = y - yhat
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    try:
        ata_inv = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate x
        ata_inv = np.linalg.pinv(A.T @ A)
    cov = s2 * ata_inv
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return a, b, cov, r2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RULConfig:
    """Configuration for the :class:`RULPredictor`.

    Attributes:
        failure_threshold: Health index at which the machine is considered
            failed (RUL is the time until the trajectory crosses this).
        model: Degradation model (``linear`` | ``exponential`` | ``auto``).
        min_history: Minimum number of trajectory points required to predict.
        confidence_level: Confidence level for the RUL interval (e.g. 0.95).
        time_per_step: Real-time duration of one trajectory step (e.g. days),
            used to express RUL in physical units.
        max_horizon: Cap on reported finite RUL (steps); predictions beyond this
            are clamped and flagged as effectively "no foreseeable failure."
        slope_epsilon: Minimum degradation slope magnitude (per step) for a
            trajectory to count as degrading; flatter trends yield infinite RUL.
        smooth_window: Optional moving-average window applied before fitting
            (``1`` disables); robustifies the fit against per-step noise.
        recent_fraction: Fraction of the most-recent trajectory used for the
            fit (``1.0`` uses all); focusing on recent history tracks regime
            changes (e.g. the onset of accelerated wear).
    """

    failure_threshold: float = 30.0
    model:             str = DegradationModel.AUTO.value
    min_history:       int = 5
    confidence_level:  float = 0.95
    time_per_step:     float = 1.0
    max_horizon:       float = 1.0e6
    slope_epsilon:     float = 1.0e-3
    smooth_window:     int = 1
    recent_fraction:   float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.failure_threshold < 100.0):
            raise ValueError("failure_threshold must be in [0, 100)")
        if self.model not in {m.value for m in DegradationModel}:
            raise ValueError(
                f"model must be one of {[m.value for m in DegradationModel]}"
            )
        if self.min_history < 3:
            raise ValueError("min_history must be >= 3 (need a fit plus residual dof)")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
        if self.time_per_step <= 0:
            raise ValueError("time_per_step must be > 0")
        if self.max_horizon <= 0:
            raise ValueError("max_horizon must be > 0")
        if self.slope_epsilon < 0:
            raise ValueError("slope_epsilon must be >= 0")
        if self.smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        if not (0.0 < self.recent_fraction <= 1.0):
            raise ValueError("recent_fraction must be in (0, 1]")


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RULPrediction:
    """The result of one RUL prediction.

    All RUL fields are expressed in **physical time units** (steps multiplied by
    ``config.time_per_step``).  When the trajectory is not degrading, ``rul`` is
    infinite and the interval bounds are infinite.

    Attributes:
        rul: Point estimate of remaining useful life.
        ci_low: Lower confidence bound on RUL (clamped at 0).
        ci_high: Upper confidence bound on RUL.
        confidence_level: The confidence level used.
        failure_threshold: The threshold the trajectory was extrapolated to.
        current_health: The most recent health value.
        model_used: The degradation model actually applied.
        slope: Fitted degradation rate (health per step; negative = degrading).
        r_squared: Goodness of fit of the selected model.
        n_observations: Number of trajectory points used in the fit.
        is_degrading: Whether the trajectory was found to be degrading.
        time_per_step: The step-to-time conversion used.
        warnings: Any non-fatal diagnostics raised during prediction.
    """

    rul:               float
    ci_low:            float
    ci_high:           float
    confidence_level:  float
    failure_threshold: float
    current_health:    float
    model_used:        str
    slope:             float
    r_squared:         float
    n_observations:    int
    is_degrading:      bool
    time_per_step:     float = 1.0
    warnings:          list[str] = field(default_factory=list)

    @property
    def rul_steps(self) -> float:
        """RUL expressed in trajectory steps rather than physical time.

        Returns:
            ``rul / time_per_step`` (infinite passes through).
        """
        if math.isinf(self.rul):
            return self.rul
        return self.rul / self.time_per_step

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; infinities are rendered as ``None``.
        """
        def _num(v: float) -> float | None:
            return None if math.isinf(v) or math.isnan(v) else v

        return {
            "rul": _num(self.rul),
            "ci_low": _num(self.ci_low),
            "ci_high": _num(self.ci_high),
            "confidence_level": self.confidence_level,
            "failure_threshold": self.failure_threshold,
            "current_health": self.current_health,
            "model_used": self.model_used,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "n_observations": self.n_observations,
            "is_degrading": self.is_degrading,
            "time_per_step": self.time_per_step,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# RULPredictor
# ---------------------------------------------------------------------------


@register_rul_predictor(PREDICTOR_NAME)
class RULPredictor:
    """Predicts remaining useful life from health-index trajectories.

    Fits a degradation model to a health trajectory, extrapolates to the failure
    threshold, and quantifies the uncertainty of the crossing time.

    Args:
        config: The predictor configuration.
        experiment_tracker: Optional tracker for logging predictions.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: RULConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or RULConfig()
        self.tracker = experiment_tracker
        self._n_predictions = 0
        logger.info(
            "RULPredictor ready | model=%s | threshold=%.1f | min_history=%d",
            self.config.model, self.config.failure_threshold,
            self.config.min_history,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, trajectory: Sequence[float] | "np.ndarray"
    ) -> RULPrediction:
        """Predict remaining useful life from a health trajectory.

        Args:
            trajectory: Sequence of health-index values in chronological order
                (oldest first), as produced by the Step-1 health engine.

        Returns:
            A :class:`RULPrediction`.

        Raises:
            ValueError: When the trajectory is shorter than ``min_history`` or
                contains non-finite values.
        """
        y_full = np.asarray(trajectory, dtype=float)
        if y_full.ndim != 1:
            raise ValueError("trajectory must be one-dimensional")
        if y_full.size < self.config.min_history:
            raise ValueError(
                f"trajectory has {y_full.size} points; need at least "
                f"{self.config.min_history}"
            )
        if not np.all(np.isfinite(y_full)):
            raise ValueError("trajectory contains non-finite values")

        warnings: list[str] = []
        current_health = float(y_full[-1])

        # Optional smoothing + recent-window focus (analysis of the trajectory).
        y = self._preprocess(y_full)
        x = np.arange(y.size, dtype=float)

        # Already at/below threshold -> failed now.
        if current_health <= self.config.failure_threshold:
            warnings.append("current health already at/below failure threshold")
            return self._degenerate_prediction(
                rul=0.0, current_health=current_health, slope=0.0,
                r2=0.0, n=int(y.size), is_degrading=True,
                model_used=self._resolve_model_label(), warnings=warnings,
            )

        # Fit the degradation model(s) and select.
        model_used, a, b, cov, r2 = self._fit_selected(x, y)

        # Non-degrading: slope not meaningfully negative -> no foreseeable failure.
        if b >= -self.config.slope_epsilon:
            warnings.append("trajectory is not degrading; RUL is unbounded")
            return self._degenerate_prediction(
                rul=RUL_INFINITE, current_health=current_health, slope=b,
                r2=r2, n=int(y.size), is_degrading=False,
                model_used=model_used, warnings=warnings,
            )

        rul_steps, ci_lo_steps, ci_hi_steps = self._crossing_time_with_ci(
            model_used, a, b, cov, x_last=float(x[-1]), n=int(y.size),
        )

        # Clamp to the reporting horizon.
        if rul_steps > self.config.max_horizon:
            warnings.append("RUL exceeds max_horizon; reported as horizon cap")
            rul_steps = self.config.max_horizon

        tps = self.config.time_per_step
        prediction = RULPrediction(
            rul=rul_steps * tps,
            ci_low=max(0.0, ci_lo_steps) * tps,
            ci_high=ci_hi_steps * tps if math.isfinite(ci_hi_steps)
            else RUL_INFINITE,
            confidence_level=self.config.confidence_level,
            failure_threshold=self.config.failure_threshold,
            current_health=current_health,
            model_used=model_used,
            slope=b,
            r_squared=r2,
            n_observations=int(y.size),
            is_degrading=True,
            time_per_step=tps,
            warnings=warnings,
        )
        self._log_prediction(prediction)
        self._n_predictions += 1
        return prediction

    def predict_from_engine(self, engine: Any) -> RULPrediction:
        """Predict directly from a Step-1 health engine's smoothed history.

        Args:
            engine: A :class:`~src.predictive.health_index.HealthIndexEngine`
                (or any object exposing ``history(smoothed=True)``).

        Returns:
            A :class:`RULPrediction`.
        """
        trajectory = engine.history(smoothed=True)
        return self.predict(trajectory)

    # ------------------------------------------------------------------
    # Trajectory preprocessing / analysis
    # ------------------------------------------------------------------

    def _preprocess(self, y_full: "np.ndarray") -> "np.ndarray":
        """Apply optional smoothing and recent-window focus before fitting.

        Args:
            y_full: The full health trajectory.

        Returns:
            The preprocessed trajectory used for the fit.
        """
        y = y_full
        w = self.config.smooth_window
        if w > 1 and y.size >= w:
            kernel = np.ones(w) / w
            y = np.convolve(y, kernel, mode="valid")
        if self.config.recent_fraction < 1.0:
            keep = max(self.config.min_history,
                       int(math.ceil(y.size * self.config.recent_fraction)))
            y = y[-keep:]
        return y

    # ------------------------------------------------------------------
    # Model fitting / selection
    # ------------------------------------------------------------------

    def _fit_selected(
        self, x: "np.ndarray", y: "np.ndarray"
    ) -> tuple[str, float, float, "np.ndarray", float]:
        """Fit the configured model (or auto-select) and return its parameters.

        Args:
            x: Step indices.
            y: Health values.

        Returns:
            Tuple ``(model_used, a, b, cov, r_squared)`` in the model's fitting
            space (linear: health; exponential: log-health).
        """
        model = self.config.model
        if model == DegradationModel.LINEAR.value:
            a, b, cov, r2 = _ols_fit(x, y)
            return DegradationModel.LINEAR.value, a, b, cov, r2
        if model == DegradationModel.EXPONENTIAL.value:
            return self._fit_exponential(x, y)
        # AUTO: fit both, prefer the higher R².
        a_l, b_l, cov_l, r2_l = _ols_fit(x, y)
        exp_label, a_e, b_e, cov_e, r2_e = self._fit_exponential(x, y)
        if r2_e > r2_l:
            return exp_label, a_e, b_e, cov_e, r2_e
        return DegradationModel.LINEAR.value, a_l, b_l, cov_l, r2_l

    def _fit_exponential(
        self, x: "np.ndarray", y: "np.ndarray"
    ) -> tuple[str, float, float, "np.ndarray", float]:
        """Fit an exponential model as a log-linear regression.

        ``health = H0·exp(-k·t)`` becomes ``log(health) = log(H0) - k·t``.  Only
        strictly-positive health values can be log-transformed; if too few
        remain the fit falls back to linear.

        Args:
            x: Step indices.
            y: Health values.

        Returns:
            Tuple ``(model_used, a, b, cov, r_squared)`` in log-space (so *a* is
            ``log(H0)`` and *b* is ``-k``).  R² is reported on the log fit.
        """
        positive = y > 1e-9
        if positive.sum() < max(3, self.config.min_history - 1):
            # Not enough positive points to log-fit; fall back to linear.
            a, b, cov, r2 = _ols_fit(x, y)
            return DegradationModel.LINEAR.value, a, b, cov, r2
        xp = x[positive]
        logy = np.log(y[positive])
        a, b, cov, r2 = _ols_fit(xp, logy)
        return DegradationModel.EXPONENTIAL.value, a, b, cov, r2

    # ------------------------------------------------------------------
    # Threshold crossing + confidence interval
    # ------------------------------------------------------------------

    def _crossing_time_with_ci(
        self, model_used: str, a: float, b: float, cov: "np.ndarray",
        *, x_last: float, n: int,
    ) -> tuple[float, float, float]:
        """Solve for the failure-threshold crossing time and its CI (delta method).

        Args:
            model_used: The fitted model label.
            a: Intercept (health-space for linear, log-space for exponential).
            b: Slope (health/step, or log-health/step for exponential).
            cov: Coefficient covariance from the fit.
            x_last: The last step index (current time).
            n: Number of observations used in the fit.

        Returns:
            Tuple ``(rul_steps, ci_low_steps, ci_high_steps)``.
        """
        thr = self.config.failure_threshold
        # Transform the threshold into the model's fitting space.
        if model_used == DegradationModel.EXPONENTIAL.value:
            # log(thr) = a + b·t_fail  (b<0)
            if thr <= 1e-9:
                # Exponential never reaches zero; treat as unbounded.
                return RUL_INFINITE, RUL_INFINITE, RUL_INFINITE
            target = math.log(thr)
        else:
            target = thr

        t_fail = (target - a) / b
        rul_steps = t_fail - x_last

        # Delta method: t_fail = (target - a)/b.
        #   d/da = -1/b ;  d/db = -(target - a)/b^2
        d_a = -1.0 / b
        d_b = -(target - a) / (b * b)
        var_tfail = (
            d_a * d_a * cov[0, 0]
            + d_b * d_b * cov[1, 1]
            + 2.0 * d_a * d_b * cov[0, 1]
        )
        se_tfail = math.sqrt(var_tfail) if var_tfail > 0 else 0.0

        dof = max(n - 2, 1)
        p = 1.0 - (1.0 - self.config.confidence_level) / 2.0
        tcrit = student_t_ppf(p, dof)
        margin = tcrit * se_tfail

        return rul_steps, rul_steps - margin, rul_steps + margin

    # ------------------------------------------------------------------
    # Degenerate-case helper
    # ------------------------------------------------------------------

    def _degenerate_prediction(
        self, *, rul: float, current_health: float, slope: float, r2: float,
        n: int, is_degrading: bool, model_used: str, warnings: list[str],
    ) -> RULPrediction:
        """Build a prediction for non-fittable / boundary cases.

        Args:
            rul: The RUL value in steps (0 or infinite).
            current_health: Current health.
            slope: Fitted or assumed slope.
            r2: Goodness of fit.
            n: Observation count.
            is_degrading: Whether the machine is degrading.
            model_used: Model label.
            warnings: Accumulated warnings.

        Returns:
            A :class:`RULPrediction`.
        """
        tps = self.config.time_per_step
        rul_time = rul * tps if math.isfinite(rul) else RUL_INFINITE
        prediction = RULPrediction(
            rul=rul_time,
            ci_low=rul_time if math.isfinite(rul_time) else RUL_INFINITE,
            ci_high=rul_time if math.isfinite(rul_time) else RUL_INFINITE,
            confidence_level=self.config.confidence_level,
            failure_threshold=self.config.failure_threshold,
            current_health=current_health,
            model_used=model_used,
            slope=slope,
            r_squared=r2,
            n_observations=n,
            is_degrading=is_degrading,
            time_per_step=tps,
            warnings=warnings,
        )
        self._log_prediction(prediction)
        self._n_predictions += 1
        return prediction

    def _resolve_model_label(self) -> str:
        """Return a concrete model label for degenerate predictions.

        Returns:
            The configured model, or ``linear`` when configured as ``auto``.
        """
        if self.config.model == DegradationModel.AUTO.value:
            return DegradationModel.LINEAR.value
        return self.config.model

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_prediction(self, prediction: RULPrediction) -> None:
        """Log a prediction to the experiment tracker (failure-safe).

        Args:
            prediction: The prediction to log.
        """
        if self.tracker is None:
            return
        try:
            rul_metric = (prediction.rul if math.isfinite(prediction.rul)
                          else -1.0)
            self.tracker.log_metrics(
                {
                    "rul": rul_metric,
                    "rul_ci_low": (prediction.ci_low
                                   if math.isfinite(prediction.ci_low) else -1.0),
                    "rul_ci_high": (prediction.ci_high
                                    if math.isfinite(prediction.ci_high) else -1.0),
                    "rul_slope": prediction.slope,
                    "rul_r_squared": prediction.r_squared,
                },
                step=self._n_predictions,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic RUL demo.

    Returns:
        Exit code 0.
    """
    rng = np.random.default_rng(0)
    trajectory = 95.0 - 2.0 * np.arange(25) + rng.normal(0, 1.5, 25)
    predictor = RULPredictor(RULConfig(failure_threshold=30.0, time_per_step=1.0))
    pred = predictor.predict(trajectory)
    logger.info(
        "RUL=%.1f steps  CI=[%.1f, %.1f]  model=%s  R2=%.3f  slope=%.2f",
        pred.rul, pred.ci_low, pred.ci_high, pred.model_used,
        pred.r_squared, pred.slope,
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
    parser = argparse.ArgumentParser(description="RUL predictor")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic RUL demo.")
    parser.add_argument("--list-predictors", action="store_true")
    args = parser.parse_args(argv)

    if args.list_predictors:
        print("Registered RUL predictors:", list_rul_predictors())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())