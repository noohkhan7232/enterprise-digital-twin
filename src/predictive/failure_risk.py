#!/usr/bin/env python3
"""Failure-risk estimation for wind turbine acoustic predictive maintenance.

This module is the third stage of the Week-5 predictive-maintenance framework.
It converts a :class:`~src.predictive.rul_predictor.RULPrediction` — a remaining-
useful-life estimate with a confidence interval (Step 2) — into **calibrated
failure probabilities over operational time horizons** (e.g. "11% chance of
failure in the next 7 days, 52% in 30, 99% in 90").  Those horizon probabilities
are what a maintenance planner actually schedules against: a point RUL of
"30 days" is far less actionable than the risk curve that says whether the next
inspection window is safe.

From RUL to a survival distribution
-----------------------------------
A single RUL number describes a *distribution* of failure times, and the right
way to reason about horizon risk is survival analysis.  This engine fits a
survival model whose mean equals the predicted RUL, then reads off the failure
CDF, survival function, and hazard rate at any horizon.

Two models are provided, covering the regimes that matter for rotating
machinery:

* **Exponential** — constant hazard, memoryless.  ``S(t) = exp(-t/RUL)``.  The
  conservative default: it makes no assumption about *when* in the remaining
  life the risk concentrates, so it assigns appreciable risk even at short
  horizons.
* **Weibull** — ``S(t) = exp(-(t/eta)^beta)``.  With shape ``beta > 1`` the
  hazard *increases* with age, which is the textbook signature of mechanical
  wear-out (bearing spall, gear pitting, fatigue): low probability of imminent
  failure early, accelerating sharply toward end of life.  ``beta = 1`` recovers
  the exponential exactly, so the Weibull is the strict generalisation.  The
  scale ``eta`` is solved from the predicted RUL via the Weibull mean
  ``E[T] = eta · Gamma(1 + 1/beta)``.

All special functions used (``exp``, ``Gamma``) are Python standard library, so
the engine is **NumPy-only** with no SciPy dependency.

Uncertainty propagation
-----------------------
The RUL confidence interval is propagated into the risk estimate: the optimistic
RUL bound (``ci_high``) yields the lower risk bound and the pessimistic bound
(``ci_low``) yields the upper risk bound, because shorter remaining life means
higher failure probability.  Every horizon therefore carries a risk interval,
not just a point probability.

Risk levels
-----------
The dominant-horizon probability is mapped to a four-level operational category
(LOW / MEDIUM / HIGH / CRITICAL) with configurable thresholds, giving operators
an immediate triage signal on top of the continuous probabilities.

Infinite RUL
------------
A healthy or flat machine has infinite predicted RUL; the engine maps this to
zero failure probability at every horizon and a LOW risk level, rather than
dividing by infinity — the correct, safe behaviour for a machine showing no
degradation.

Usage::

    from src.predictive.failure_risk import FailureRiskEngine, FailureRiskConfig

    engine = FailureRiskEngine(FailureRiskConfig(horizons=(7, 30, 90)))
    risk = engine.predict_from_rul(rul_prediction)

    print(risk.risk_level.value)
    for h, p in risk.horizon_risks.items():
        print(h, p, risk.horizon_intervals[h])

CLI::

    python src/predictive/failure_risk.py --demo
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

from src.predictive.rul_predictor import RULPrediction

logger = logging.getLogger("failure_risk")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named failure-risk engines.
FAILURE_RISK_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "survival_failure_risk"

#: Default analysis horizons (days), matching the platform's 7/30/90 cadence.
DEFAULT_HORIZONS: Final[tuple[float, ...]] = (7.0, 30.0, 90.0)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SurvivalModel(str, Enum):
    """Supported survival models."""

    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"


class RiskLevel(str, Enum):
    """Operational risk categories, ordered from least to most severe."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_probability(
        cls, probability: float, *, medium: float, high: float, critical: float
    ) -> "RiskLevel":
        """Map a failure probability to a risk category.

        Args:
            probability: Failure probability in ``[0, 1]``.
            medium: Lower bound of the MEDIUM band.
            high: Lower bound of the HIGH band.
            critical: Lower bound of the CRITICAL band.

        Returns:
            The corresponding :class:`RiskLevel`.
        """
        if probability >= critical:
            return cls.CRITICAL
        if probability >= high:
            return cls.HIGH
        if probability >= medium:
            return cls.MEDIUM
        return cls.LOW


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_failure_risk_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a failure-risk engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = FAILURE_RISK_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Failure-risk engine '{name}' already registered to "
                f"{existing.__name__}"
            )
        FAILURE_RISK_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered failure-risk engine '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_failure_risk_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered failure-risk engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in FAILURE_RISK_REGISTRY:
        available = ", ".join(sorted(FAILURE_RISK_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown failure-risk engine '{name}'. Available: {available}"
        )
    return FAILURE_RISK_REGISTRY[name](**kwargs)


def list_failure_risk_engines() -> list[str]:
    """Return the sorted names of registered failure-risk engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(FAILURE_RISK_REGISTRY)


# ---------------------------------------------------------------------------
# Pure survival functions (NumPy / stdlib only)
# ---------------------------------------------------------------------------


def exponential_survival(t: "float | np.ndarray", mean_life: float) -> "float | np.ndarray":
    """Exponential survival function ``S(t) = exp(-t / mean_life)``.

    Args:
        t: Time(s) at which to evaluate survival (``>= 0``).
        mean_life: Mean time to failure (the predicted RUL).

    Returns:
        Survival probability/probabilities in ``[0, 1]``.
    """
    arr = np.asarray(t, dtype=float)
    if mean_life <= 0:
        out = np.zeros_like(arr)
    else:
        out = np.exp(-np.maximum(0.0, arr) / mean_life)
    return float(out) if np.isscalar(t) else out


def weibull_survival(
    t: "float | np.ndarray", scale: float, shape: float
) -> "float | np.ndarray":
    """Weibull survival function ``S(t) = exp(-(t / scale) ** shape)``.

    Args:
        t: Time(s) at which to evaluate survival (``>= 0``).
        scale: Weibull scale parameter ``eta`` (``> 0``).
        shape: Weibull shape parameter ``beta`` (``> 0``); ``>1`` is wear-out.

    Returns:
        Survival probability/probabilities in ``[0, 1]``.
    """
    arr = np.maximum(0.0, np.asarray(t, dtype=float))
    if scale <= 0 or shape <= 0:
        out = np.zeros_like(arr)
    else:
        out = np.exp(-np.power(arr / scale, shape))
    return float(out) if np.isscalar(t) else out


def weibull_scale_from_mean(mean_life: float, shape: float) -> float:
    """Solve the Weibull scale ``eta`` so the distribution mean equals *mean_life*.

    Uses ``E[T] = eta · Gamma(1 + 1/shape)``.

    Args:
        mean_life: Target mean time to failure (the predicted RUL).
        shape: Weibull shape parameter ``beta`` (``> 0``).

    Returns:
        The scale parameter ``eta``.
    """
    if shape <= 0:
        raise ValueError("shape must be > 0")
    return mean_life / math.gamma(1.0 + 1.0 / shape)


def hazard_rate(
    t: float, mean_life: float, model: str, shape: float
) -> float:
    """Instantaneous hazard rate at time *t* for the chosen model.

    Args:
        t: Time at which to evaluate the hazard (``>= 0``).
        mean_life: Mean time to failure (the predicted RUL).
        model: ``"exponential"`` or ``"weibull"``.
        shape: Weibull shape parameter (ignored for exponential).

    Returns:
        The hazard rate (failures per unit time).
    """
    if mean_life <= 0:
        return float("inf")
    if model == SurvivalModel.WEIBULL.value:
        eta = weibull_scale_from_mean(mean_life, shape)
        if eta <= 0:
            return float("inf")
        return (shape / eta) * (max(0.0, t) / eta) ** (shape - 1.0)
    # Exponential: constant hazard 1 / mean_life.
    return 1.0 / mean_life


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureRiskConfig:
    """Configuration for the :class:`FailureRiskEngine`.

    Attributes:
        model: Survival model (``exponential`` | ``weibull``).
        horizons: Time horizons (in the same units as the RUL) at which to
            report failure probability.
        weibull_shape: Weibull shape ``beta`` (``>1`` for wear-out); ignored for
            the exponential model.
        dominant_horizon: Horizon whose probability drives the risk level;
            defaults to the largest horizon when ``None``.
        medium_threshold: Failure probability at/above which risk is MEDIUM.
        high_threshold: Failure probability at/above which risk is HIGH.
        critical_threshold: Failure probability at/above which risk is CRITICAL.
        propagate_ci: Propagate the RUL confidence interval into risk intervals.
    """

    model:              str = SurvivalModel.EXPONENTIAL.value
    horizons:           tuple[float, ...] = DEFAULT_HORIZONS
    weibull_shape:      float = 2.0
    dominant_horizon:   float | None = None
    medium_threshold:   float = 0.25
    high_threshold:     float = 0.50
    critical_threshold: float = 0.75
    propagate_ci:       bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.model not in {m.value for m in SurvivalModel}:
            raise ValueError(
                f"model must be one of {[m.value for m in SurvivalModel]}"
            )
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("all horizons must be > 0")
        if self.weibull_shape <= 0:
            raise ValueError("weibull_shape must be > 0")
        if not (0.0 < self.medium_threshold <= self.high_threshold
                <= self.critical_threshold < 1.0):
            raise ValueError(
                "thresholds must satisfy 0 < medium <= high <= critical < 1"
            )
        if self.dominant_horizon is not None and self.dominant_horizon <= 0:
            raise ValueError("dominant_horizon must be > 0 when set")


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureRiskPrediction:
    """The result of one failure-risk estimation.

    Attributes:
        horizon_risks: Mapping of horizon -> failure probability ``P(T <= h)``.
        horizon_intervals: Mapping of horizon -> ``(risk_low, risk_high)`` from
            propagating the RUL confidence interval (empty when not propagated).
        risk_level: The overall :class:`RiskLevel` from the dominant horizon.
        dominant_horizon: The horizon driving ``risk_level``.
        dominant_probability: Failure probability at the dominant horizon.
        model_used: The survival model applied.
        mean_life: The mean time to failure used (the predicted RUL).
        hazard_at_zero: Hazard rate at the current time.
        weibull_shape: The shape parameter (for the Weibull model).
        is_degrading: Whether the source RUL indicated degradation.
        confidence_level: Confidence level inherited from the RUL prediction.
        warnings: Non-fatal diagnostics.
    """

    horizon_risks:        dict[float, float]
    horizon_intervals:    dict[float, tuple[float, float]] = field(default_factory=dict)
    risk_level:           RiskLevel = RiskLevel.LOW
    dominant_horizon:     float = 0.0
    dominant_probability: float = 0.0
    model_used:           str = SurvivalModel.EXPONENTIAL.value
    mean_life:            float = float("inf")
    hazard_at_zero:       float = 0.0
    weibull_shape:        float = 2.0
    is_degrading:         bool = False
    confidence_level:     float = 0.95
    warnings:             list[str] = field(default_factory=list)

    def survival_at(self, horizon: float) -> float:
        """Return the survival probability ``S(h) = 1 - F(h)`` at a horizon.

        Args:
            horizon: A horizon present in ``horizon_risks``.

        Returns:
            The survival probability, or ``nan`` if the horizon is absent.
        """
        if horizon not in self.horizon_risks:
            return float("nan")
        return 1.0 - self.horizon_risks[horizon]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation; infinite mean life renders as ``None``.
        """
        return {
            "horizon_risks": {str(k): v for k, v in self.horizon_risks.items()},
            "horizon_intervals": {
                str(k): list(v) for k, v in self.horizon_intervals.items()
            },
            "risk_level": self.risk_level.value,
            "dominant_horizon": self.dominant_horizon,
            "dominant_probability": self.dominant_probability,
            "model_used": self.model_used,
            "mean_life": (None if math.isinf(self.mean_life) else self.mean_life),
            "hazard_at_zero": (
                None if math.isinf(self.hazard_at_zero) else self.hazard_at_zero
            ),
            "weibull_shape": self.weibull_shape,
            "is_degrading": self.is_degrading,
            "confidence_level": self.confidence_level,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# FailureRiskEngine
# ---------------------------------------------------------------------------


@register_failure_risk_engine(ENGINE_NAME)
class FailureRiskEngine:
    """Converts RUL predictions into horizon failure probabilities.

    Fits a survival model whose mean equals the predicted RUL and evaluates the
    failure CDF, survival function, and hazard rate at the configured horizons,
    propagating the RUL confidence interval into risk intervals.

    Args:
        config: The engine configuration.
        experiment_tracker: Optional tracker for logging predictions.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: FailureRiskConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or FailureRiskConfig()
        self.tracker = experiment_tracker
        self._n_predictions = 0
        logger.info(
            "FailureRiskEngine ready | model=%s | horizons=%s",
            self.config.model, self.config.horizons,
        )

    # ------------------------------------------------------------------
    # Survival / CDF / hazard primitives bound to config
    # ------------------------------------------------------------------

    def survival_function(
        self, t: "float | np.ndarray", mean_life: float
    ) -> "float | np.ndarray":
        """Survival probability ``S(t)`` under the configured model.

        Args:
            t: Time(s) at which to evaluate survival.
            mean_life: Mean time to failure (the predicted RUL).

        Returns:
            Survival probability/probabilities.
        """
        if self.config.model == SurvivalModel.WEIBULL.value:
            if math.isinf(mean_life) or mean_life <= 0:
                arr = np.asarray(t, dtype=float)
                out = np.ones_like(arr) if math.isinf(mean_life) else np.zeros_like(arr)
                return float(out) if np.isscalar(t) else out
            eta = weibull_scale_from_mean(mean_life, self.config.weibull_shape)
            return weibull_survival(t, eta, self.config.weibull_shape)
        if math.isinf(mean_life):
            arr = np.asarray(t, dtype=float)
            out = np.ones_like(arr)
            return float(out) if np.isscalar(t) else out
        return exponential_survival(t, mean_life)

    def failure_cdf(
        self, t: "float | np.ndarray", mean_life: float
    ) -> "float | np.ndarray":
        """Failure CDF ``F(t) = 1 - S(t)`` under the configured model.

        Args:
            t: Time(s) at which to evaluate the CDF.
            mean_life: Mean time to failure (the predicted RUL).

        Returns:
            Failure probability/probabilities.
        """
        s = self.survival_function(t, mean_life)
        return 1.0 - s if np.isscalar(t) else 1.0 - np.asarray(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_rul(
        self, rul_prediction: RULPrediction
    ) -> FailureRiskPrediction:
        """Estimate failure risk from a RUL prediction.

        Args:
            rul_prediction: The :class:`RULPrediction` to convert.

        Returns:
            A :class:`FailureRiskPrediction`.
        """
        warnings: list[str] = list(rul_prediction.warnings)
        mean_life = rul_prediction.rul

        # Infinite (non-degrading) RUL -> zero risk everywhere.
        if math.isinf(mean_life) or not rul_prediction.is_degrading:
            risks = {float(h): 0.0 for h in self.config.horizons}
            intervals = (
                {float(h): (0.0, 0.0) for h in self.config.horizons}
                if self.config.propagate_ci else {}
            )
            warnings.append("non-degrading RUL; failure risk is zero at all horizons")
            prediction = FailureRiskPrediction(
                horizon_risks=risks,
                horizon_intervals=intervals,
                risk_level=RiskLevel.LOW,
                dominant_horizon=self._dominant_horizon(),
                dominant_probability=0.0,
                model_used=self.config.model,
                mean_life=float("inf"),
                hazard_at_zero=0.0,
                weibull_shape=self.config.weibull_shape,
                is_degrading=False,
                confidence_level=rul_prediction.confidence_level,
                warnings=warnings,
            )
            self._log_prediction(prediction)
            self._n_predictions += 1
            return prediction

        # Already failed (RUL == 0) -> certain failure at every horizon.
        if mean_life <= 0:
            risks = {float(h): 1.0 for h in self.config.horizons}
            intervals = (
                {float(h): (1.0, 1.0) for h in self.config.horizons}
                if self.config.propagate_ci else {}
            )
            warnings.append("RUL is zero; failure is certain")
            prediction = FailureRiskPrediction(
                horizon_risks=risks,
                horizon_intervals=intervals,
                risk_level=RiskLevel.CRITICAL,
                dominant_horizon=self._dominant_horizon(),
                dominant_probability=1.0,
                model_used=self.config.model,
                mean_life=0.0,
                hazard_at_zero=float("inf"),
                weibull_shape=self.config.weibull_shape,
                is_degrading=True,
                confidence_level=rul_prediction.confidence_level,
                warnings=warnings,
            )
            self._log_prediction(prediction)
            self._n_predictions += 1
            return prediction

        # General case: evaluate the failure CDF at each horizon.
        risks: dict[float, float] = {}
        intervals: dict[float, tuple[float, float]] = {}
        for h in self.config.horizons:
            hf = float(h)
            risks[hf] = float(_clip01(self.failure_cdf(hf, mean_life)))
            if self.config.propagate_ci:
                intervals[hf] = self._risk_interval(hf, rul_prediction)

        dominant_h = self._dominant_horizon()
        dominant_p = risks.get(dominant_h, risks[float(self.config.horizons[-1])])
        risk_level = RiskLevel.from_probability(
            dominant_p, medium=self.config.medium_threshold,
            high=self.config.high_threshold, critical=self.config.critical_threshold,
        )
        hz0 = hazard_rate(0.0, mean_life, self.config.model, self.config.weibull_shape)

        prediction = FailureRiskPrediction(
            horizon_risks=risks,
            horizon_intervals=intervals,
            risk_level=risk_level,
            dominant_horizon=dominant_h,
            dominant_probability=dominant_p,
            model_used=self.config.model,
            mean_life=mean_life,
            hazard_at_zero=hz0,
            weibull_shape=self.config.weibull_shape,
            is_degrading=True,
            confidence_level=rul_prediction.confidence_level,
            warnings=warnings,
        )
        self._log_prediction(prediction)
        self._n_predictions += 1
        return prediction

    def predict_from_health_engine(
        self, health_engine: Any, rul_predictor: Any
    ) -> FailureRiskPrediction:
        """Estimate failure risk straight from a health engine + RUL predictor.

        Convenience that chains Steps 1->2->3: it asks the RUL predictor to
        predict from the health engine's trajectory, then converts that RUL
        prediction to failure risk.

        Args:
            health_engine: A Step-1 health engine exposing
                ``history(smoothed=True)``.
            rul_predictor: A Step-2 predictor exposing ``predict_from_engine``.

        Returns:
            A :class:`FailureRiskPrediction`.
        """
        rul_prediction = rul_predictor.predict_from_engine(health_engine)
        return self.predict_from_rul(rul_prediction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_horizon(self) -> float:
        """Return the horizon driving the risk level.

        Returns:
            The configured dominant horizon, or the largest horizon.
        """
        if self.config.dominant_horizon is not None:
            return float(self.config.dominant_horizon)
        return float(max(self.config.horizons))

    def _risk_interval(
        self, horizon: float, rul_prediction: RULPrediction
    ) -> tuple[float, float]:
        """Propagate the RUL CI into a risk interval at a horizon.

        Shorter remaining life means higher risk, so the optimistic RUL bound
        (``ci_high``) gives the lower risk bound and the pessimistic bound
        (``ci_low``) gives the upper risk bound.

        Args:
            horizon: The horizon at which to evaluate.
            rul_prediction: The source RUL prediction.

        Returns:
            Tuple ``(risk_low, risk_high)``.
        """
        ci_low = rul_prediction.ci_low
        ci_high = rul_prediction.ci_high

        # Upper risk bound from the pessimistic (shorter) RUL bound.
        if math.isinf(ci_low) or ci_low <= 0:
            risk_high = 1.0 if ci_low <= 0 and not math.isinf(ci_low) else \
                float(_clip01(self.failure_cdf(horizon, rul_prediction.rul)))
        else:
            risk_high = float(_clip01(self.failure_cdf(horizon, ci_low)))

        # Lower risk bound from the optimistic (longer) RUL bound.
        if math.isinf(ci_high):
            risk_low = 0.0
        else:
            risk_low = float(_clip01(self.failure_cdf(horizon, max(1e-9, ci_high))))

        lo, hi = min(risk_low, risk_high), max(risk_low, risk_high)
        return lo, hi

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_prediction(self, prediction: FailureRiskPrediction) -> None:
        """Log a prediction to the experiment tracker (failure-safe).

        Args:
            prediction: The prediction to log.
        """
        if self.tracker is None:
            return
        try:
            metrics = {
                f"failure_risk_h{int(h)}": p
                for h, p in prediction.horizon_risks.items()
            }
            metrics["dominant_failure_risk"] = prediction.dominant_probability
            self.tracker.log_metrics(metrics, step=self._n_predictions)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip01(value: "float | np.ndarray") -> "float | np.ndarray":
    """Clip a value (or array) to ``[0, 1]``, mapping NaN to 0.

    Args:
        value: The value(s) to clip.

    Returns:
        The clipped value(s).
    """
    arr = np.nan_to_num(np.asarray(value, dtype=float), nan=0.0)
    clipped = np.clip(arr, 0.0, 1.0)
    return float(clipped) if np.isscalar(value) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    ) else clipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic failure-risk demo.

    Returns:
        Exit code 0.
    """
    from src.predictive.rul_predictor import RULConfig, RULPredictor

    rng = np.random.default_rng(0)
    trajectory = 95.0 - 2.0 * np.arange(25) + rng.normal(0, 1.5, 25)
    rul_pred = RULPredictor(RULConfig(failure_threshold=30.0)).predict(trajectory)

    for model in ("exponential", "weibull"):
        engine = FailureRiskEngine(FailureRiskConfig(model=model, horizons=(7, 30, 90)))
        risk = engine.predict_from_rul(rul_pred)
        logger.info(
            "[%s] RUL=%.1f -> risk %s | P(7)=%.2f P(30)=%.2f P(90)=%.2f",
            model, rul_pred.rul, risk.risk_level.value,
            risk.horizon_risks[7.0], risk.horizon_risks[30.0],
            risk.horizon_risks[90.0],
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
    parser = argparse.ArgumentParser(description="Failure-risk engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic failure-risk demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered failure-risk engines:", list_failure_risk_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())