#!/usr/bin/env python3
"""Health Index Engine for wind turbine acoustic predictive maintenance.

This module is the first stage of the Week-5 Remaining-Useful-Life (RUL)
framework.  It converts the per-clip diagnostic signals produced by the Week-4
models — the anomaly autoencoder's reconstruction error and the classifiers'
fault probabilities — into a single, continuous **health index** in ``[0, 100]``
that trends smoothly over a machine's life and is robust to the noise inherent
in per-clip scores.

A health index is the backbone of an industrial predictive-maintenance system:
it is the one number an operator watches, the input the RUL regressor
extrapolates to a failure threshold, and the quantity whose trend the
early-warning logic monitors.  This engine produces it with three properties
that matter for that role:

* **Continuous (0–100)** — 100 is a pristine machine, 0 is failed.  The mapping
  from raw diagnostic signals is smooth and bounded, so small changes in the
  underlying acoustics move the index by small, interpretable amounts.
* **Trend-aware** — raw per-clip scores are noisy; the engine maintains a
  history and reports an exponentially-smoothed index plus an explicit
  degradation rate (the slope of recent health), so transient spikes do not
  trigger false alarms while genuine trends are surfaced early.
* **Degradation-aware** — mechanical wear is largely irreversible.  An optional
  monotonic mode prevents the index from spuriously *recovering* after a real
  degradation event (a denoising artefact or a quiet operating period should not
  read as the bearing healing itself), which is the correct prior for fatigue,
  wear, and crack growth.

Signal fusion
-------------
The engine accepts any subset of three diagnostic signals and fuses the
available ones with configurable weights:

* ``anomaly_score`` — reconstruction error from
  :class:`~src.models.anomaly_autoencoder.AnomalyAutoencoder`, interpreted
  relative to a normal-data threshold.  Scores at or below threshold read as
  fully healthy; scores above it decay the health exponentially.
* ``fault_probability`` — ``1 - P(normal)`` from any classifier's
  ``predict_proba``; the more probability mass on fault classes, the lower the
  health.
* ``deviation`` — an optional pre-normalised ``[0, 1]`` deviation from a
  user-supplied feature monitor, for sites with bespoke sensors.

Each signal is mapped to its own ``[0, 100]`` health contribution and the
contributions are combined as a weighted average over whichever signals are
present, so the engine degrades gracefully when only one signal is available.

Architecture
------------
* :class:`HealthIndexConfig` — frozen, validated configuration.
* :class:`HealthState` — the immutable result of one update (smoothed index,
  raw index, status, trend, degradation rate, contributions, timestamp).
* :class:`HealthStatus` — a four-level operational status enum.
* :class:`HealthIndexEngine` — stateful streaming engine plus stateless batch
  scoring; registry- and tracker-integrated; exposes a pure scoring function
  suitable for ONNX export of the core mapping.

The pure mapping ``raw_health_from_signals`` is dependency-free (NumPy only) and
deterministic, so the whole engine is testable without PyTorch and the core
scoring path can be traced to ONNX.

Usage::

    from src.predictive.health_index import HealthIndexEngine, HealthIndexConfig

    engine = HealthIndexEngine(HealthIndexConfig(monotonic=True))
    engine.set_anomaly_threshold(0.02)          # from the AE's normal set

    for clip in stream:
        state = engine.update(
            anomaly_score=ae.anomaly_score(clip).item(),
            fault_probability=1.0 - clf.predict_proba(clip)[0, 0].item(),
        )
        print(state.index, state.status.value, state.degradation_rate)

CLI::

    python src/predictive/health_index.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

logger = logging.getLogger("health_index")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Health-index bounds.
HEALTH_MAX: Final[float] = 100.0
HEALTH_MIN: Final[float] = 0.0

#: Registry of named health-index engines (mirrors the model registry pattern).
HEALTH_ENGINE_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "acoustic_health_index"

#: Recognised diagnostic signal names.
SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "anomaly_score", "fault_probability", "deviation",
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_health_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a health-index engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = HEALTH_ENGINE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Health engine '{name}' already registered to {existing.__name__}"
            )
        HEALTH_ENGINE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered health engine '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_health_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered health-index engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in HEALTH_ENGINE_REGISTRY:
        available = ", ".join(sorted(HEALTH_ENGINE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown health engine '{name}'. Available: {available}")
    return HEALTH_ENGINE_REGISTRY[name](**kwargs)


def list_health_engines() -> list[str]:
    """Return the sorted names of registered health-index engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(HEALTH_ENGINE_REGISTRY)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class HealthStatus(str, Enum):
    """Operational health status derived from the health index.

    The thresholds are configurable; the default bands are a common
    predictive-maintenance convention.
    """

    HEALTHY = "healthy"      # index >= healthy_threshold
    DEGRADING = "degrading"  # warning_threshold <= index < healthy_threshold
    WARNING = "warning"      # critical_threshold <= index < warning_threshold
    CRITICAL = "critical"    # index < critical_threshold

    @classmethod
    def from_index(
        cls, index: float, healthy: float, warning: float, critical: float
    ) -> "HealthStatus":
        """Map a health index to a status band.

        Args:
            index: Health index in ``[0, 100]``.
            healthy: Lower bound of the HEALTHY band.
            warning: Lower bound of the DEGRADING band.
            critical: Lower bound of the WARNING band.

        Returns:
            The corresponding :class:`HealthStatus`.
        """
        if index >= healthy:
            return cls.HEALTHY
        if index >= warning:
            return cls.DEGRADING
        if index >= critical:
            return cls.WARNING
        return cls.CRITICAL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthIndexConfig:
    """Configuration for the :class:`HealthIndexEngine`.

    Attributes:
        weight_anomaly: Fusion weight for the anomaly-score signal.
        weight_fault: Fusion weight for the fault-probability signal.
        weight_deviation: Fusion weight for the optional deviation signal.
        anomaly_scale: Divisor controlling how fast health decays above the
            anomaly threshold; defaults to the threshold itself when ``None``.
        fault_gamma: Exponent shaping the ``P(normal) -> health`` curve
            (``>1`` is stricter, ``<1`` is more lenient).
        smoothing_alpha: EWMA weight on the newest sample (``0 < a <= 1``);
            smaller values smooth more heavily.
        monotonic: When ``True``, the smoothed index is constrained to be
            non-increasing (degradation-aware, irreversible-wear prior).
        degradation_window: Number of recent points used to estimate the
            degradation rate (slope).
        history_size: Maximum retained history length (older points dropped).
        healthy_threshold: Index at/above which status is HEALTHY.
        warning_threshold: Index at/above which status is at worst DEGRADING.
        critical_threshold: Index at/above which status is at worst WARNING.
        clip_signals: Clip incoming signals to sane ranges before mapping.
    """

    weight_anomaly:      float = 0.5
    weight_fault:        float = 0.4
    weight_deviation:    float = 0.1
    anomaly_scale:       float | None = None
    fault_gamma:         float = 1.0
    smoothing_alpha:     float = 0.3
    monotonic:           bool = False
    degradation_window:  int = 5
    history_size:        int = 512
    healthy_threshold:   float = 80.0
    warning_threshold:   float = 60.0
    critical_threshold:  float = 40.0
    clip_signals:        bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if min(self.weight_anomaly, self.weight_fault, self.weight_deviation) < 0:
            raise ValueError("signal weights must be non-negative")
        if (self.weight_anomaly + self.weight_fault + self.weight_deviation) <= 0:
            raise ValueError("at least one signal weight must be positive")
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.fault_gamma <= 0:
            raise ValueError("fault_gamma must be > 0")
        if self.degradation_window < 2:
            raise ValueError("degradation_window must be >= 2")
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if not (self.critical_threshold <= self.warning_threshold
                <= self.healthy_threshold):
            raise ValueError(
                "thresholds must satisfy critical <= warning <= healthy"
            )
        if self.anomaly_scale is not None and self.anomaly_scale <= 0:
            raise ValueError("anomaly_scale must be > 0 when set")


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthState:
    """The result of one health-index update.

    Attributes:
        index: The smoothed (and optionally monotonic) health index ``[0, 100]``.
        raw_index: The instantaneous health from the current signals only.
        status: The operational :class:`HealthStatus`.
        trend: ``"improving"``, ``"stable"``, or ``"degrading"``.
        degradation_rate: Slope of recent health (index points per update;
            negative means degrading).
        contributions: Per-signal health contributions that were fused.
        step: The update index (0-based) within this engine's life.
        timestamp: ISO-8601 UTC timestamp of the update.
    """

    index:            float
    raw_index:        float
    status:           HealthStatus
    trend:            str
    degradation_rate: float
    contributions:    dict[str, float] = field(default_factory=dict)
    step:             int = 0
    timestamp:        str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the state.
        """
        return {
            "index": self.index,
            "raw_index": self.raw_index,
            "status": self.status.value,
            "trend": self.trend,
            "degradation_rate": self.degradation_rate,
            "contributions": dict(self.contributions),
            "step": self.step,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Pure signal -> health mappings (NumPy-only, ONNX-traceable, vectorised)
# ---------------------------------------------------------------------------


def anomaly_health(
    anomaly_score: "float | np.ndarray",
    threshold: float,
    scale: float | None = None,
) -> "float | np.ndarray":
    """Map an anomaly score to a health contribution in ``[0, 100]``.

    Scores at or below *threshold* are fully healthy (100); above it, health
    decays exponentially with the normalised exceedance, so the mapping is
    smooth, bounded, and monotonic in the score.

    Args:
        anomaly_score: Reconstruction-error score(s); higher is worse.
        threshold: The normal-data anomaly threshold.
        scale: Decay scale; defaults to ``threshold`` (or 1.0 if threshold ~0).

    Returns:
        Health contribution(s) in ``[0, 100]``.
    """
    s = np.asarray(anomaly_score, dtype=float)
    denom = scale if (scale is not None and scale > 0) else (
        threshold if threshold > 1e-12 else 1.0
    )
    exceedance = np.maximum(0.0, s - threshold) / denom
    health = HEALTH_MAX * np.exp(-exceedance)
    # NaN-safe: a NaN signal reads as fully degraded, not as a NaN health.
    health = np.nan_to_num(health, nan=HEALTH_MIN)
    return _scalar_or_array(health, anomaly_score)


def fault_health(
    fault_probability: "float | np.ndarray", gamma: float = 1.0
) -> "float | np.ndarray":
    """Map a fault probability to a health contribution in ``[0, 100]``.

    ``health = 100 * (1 - fault_probability) ** gamma``; i.e. the probability of
    the *normal* class raised to ``gamma`` and scaled.

    Args:
        fault_probability: ``1 - P(normal)``; higher is worse, in ``[0, 1]``.
        gamma: Curve-shaping exponent (``>1`` stricter, ``<1`` more lenient).

    Returns:
        Health contribution(s) in ``[0, 100]``.
    """
    p_fault = np.clip(np.asarray(fault_probability, dtype=float), 0.0, 1.0)
    health = HEALTH_MAX * np.power(1.0 - p_fault, gamma)
    health = np.nan_to_num(health, nan=HEALTH_MIN)
    return _scalar_or_array(health, fault_probability)


def deviation_health(
    deviation: "float | np.ndarray",
) -> "float | np.ndarray":
    """Map a pre-normalised ``[0, 1]`` deviation to health in ``[0, 100]``.

    Args:
        deviation: Deviation in ``[0, 1]``; higher is worse.

    Returns:
        Health contribution(s) in ``[0, 100]``.
    """
    d = np.clip(np.nan_to_num(np.asarray(deviation, dtype=float), nan=1.0), 0.0, 1.0)
    return _scalar_or_array(HEALTH_MAX * (1.0 - d), deviation)


def raw_health_from_signals(
    *,
    anomaly_score: float | None = None,
    fault_probability: float | None = None,
    deviation: float | None = None,
    anomaly_threshold: float = 0.0,
    anomaly_scale: float | None = None,
    fault_gamma: float = 1.0,
    weight_anomaly: float = 0.5,
    weight_fault: float = 0.4,
    weight_deviation: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """Fuse available diagnostic signals into a single raw health value.

    Only the provided (non-``None``) signals contribute; weights are
    renormalised over whichever signals are present, so the function degrades
    gracefully to any subset.

    Args:
        anomaly_score: Optional reconstruction-error score.
        fault_probability: Optional ``1 - P(normal)``.
        deviation: Optional pre-normalised ``[0, 1]`` deviation.
        anomaly_threshold: Threshold for the anomaly mapping.
        anomaly_scale: Decay scale for the anomaly mapping.
        fault_gamma: Exponent for the fault mapping.
        weight_anomaly: Weight for the anomaly contribution.
        weight_fault: Weight for the fault contribution.
        weight_deviation: Weight for the deviation contribution.

    Returns:
        Tuple ``(raw_health, contributions)`` where *contributions* maps each
        present signal to its health contribution.

    Raises:
        ValueError: When no signal is provided.
    """
    contributions: dict[str, float] = {}
    weights: dict[str, float] = {}

    if anomaly_score is not None:
        contributions["anomaly_score"] = float(
            anomaly_health(anomaly_score, anomaly_threshold, anomaly_scale)
        )
        weights["anomaly_score"] = weight_anomaly
    if fault_probability is not None:
        contributions["fault_probability"] = float(
            fault_health(fault_probability, fault_gamma)
        )
        weights["fault_probability"] = weight_fault
    if deviation is not None:
        contributions["deviation"] = float(deviation_health(deviation))
        weights["deviation"] = weight_deviation

    if not contributions:
        raise ValueError(
            "At least one of anomaly_score, fault_probability, or deviation "
            "must be provided"
        )

    total_w = sum(weights.values())
    if total_w <= 0:
        # All present signals had zero weight: fall back to a plain mean.
        raw = float(np.mean(list(contributions.values())))
    else:
        raw = float(
            sum(contributions[k] * weights[k] for k in contributions) / total_w
        )
    return _clip_health(raw), contributions


# ---------------------------------------------------------------------------
# Health Index Engine
# ---------------------------------------------------------------------------


@register_health_engine(ENGINE_NAME)
class HealthIndexEngine:
    """Stateful, trend- and degradation-aware health-index engine.

    Maintains a history of health values, applies exponential smoothing and an
    optional monotonic (irreversible-degradation) constraint, and reports an
    operational status, trend direction, and degradation rate on every update.

    Args:
        config: The engine configuration.
        experiment_tracker: Optional tracker for logging each update.
        anomaly_threshold: Initial anomaly threshold (settable later).
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: HealthIndexConfig | None = None,
        experiment_tracker: Any = None,
        anomaly_threshold: float = 0.0,
    ) -> None:
        self.config = config or HealthIndexConfig()
        self.tracker = experiment_tracker
        self._anomaly_threshold = float(anomaly_threshold)
        self._raw_history: list[float] = []
        self._smoothed_history: list[float] = []
        self._smoothed: float | None = None
        self._monotonic_floor: float = HEALTH_MAX
        self._step: int = 0
        logger.info(
            "HealthIndexEngine ready | alpha=%.2f | monotonic=%s",
            self.config.smoothing_alpha, self.config.monotonic,
        )

    # ------------------------------------------------------------------
    # Configuration of the anomaly threshold
    # ------------------------------------------------------------------

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set the anomaly threshold used to map reconstruction error to health.

        Args:
            threshold: The normal-data anomaly threshold (e.g. from the
                autoencoder's ``estimate_threshold``).
        """
        if threshold < 0:
            raise ValueError("anomaly threshold must be >= 0")
        self._anomaly_threshold = float(threshold)

    @property
    def anomaly_threshold(self) -> float:
        """The current anomaly threshold.

        Returns:
            The threshold value.
        """
        return self._anomaly_threshold

    # ------------------------------------------------------------------
    # Streaming update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        anomaly_score: float | None = None,
        fault_probability: float | None = None,
        deviation: float | None = None,
        timestamp: str | None = None,
    ) -> HealthState:
        """Ingest one observation and return the updated health state.

        Args:
            anomaly_score: Optional reconstruction-error score.
            fault_probability: Optional ``1 - P(normal)``.
            deviation: Optional pre-normalised ``[0, 1]`` deviation.
            timestamp: Optional ISO-8601 timestamp; generated when omitted.

        Returns:
            The :class:`HealthState` for this update.
        """
        raw, contributions = raw_health_from_signals(
            anomaly_score=anomaly_score,
            fault_probability=fault_probability,
            deviation=deviation,
            anomaly_threshold=self._anomaly_threshold,
            anomaly_scale=self.config.anomaly_scale,
            fault_gamma=self.config.fault_gamma,
            weight_anomaly=self.config.weight_anomaly,
            weight_fault=self.config.weight_fault,
            weight_deviation=self.config.weight_deviation,
        )

        # Exponential smoothing (trend-awareness).
        if self._smoothed is None:
            smoothed = raw
        else:
            a = self.config.smoothing_alpha
            smoothed = a * raw + (1.0 - a) * self._smoothed

        # Monotonic constraint (degradation-awareness).
        if self.config.monotonic:
            self._monotonic_floor = min(self._monotonic_floor, smoothed)
            smoothed = self._monotonic_floor

        smoothed = _clip_health(smoothed)
        self._smoothed = smoothed

        # Bookkeeping + history (bounded).
        self._raw_history.append(raw)
        self._smoothed_history.append(smoothed)
        self._trim_history()

        degradation_rate = self._degradation_rate()
        trend = self._trend(degradation_rate)
        status = HealthStatus.from_index(
            smoothed, self.config.healthy_threshold,
            self.config.warning_threshold, self.config.critical_threshold,
        )

        state = HealthState(
            index=smoothed,
            raw_index=raw,
            status=status,
            trend=trend,
            degradation_rate=degradation_rate,
            contributions=contributions,
            step=self._step,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )
        self._step += 1
        self._log_state(state)
        return state

    # ------------------------------------------------------------------
    # Batch / stateless scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        *,
        anomaly_scores: Sequence[float] | None = None,
        fault_probabilities: Sequence[float] | None = None,
        deviations: Sequence[float] | None = None,
    ) -> "np.ndarray":
        """Score a batch of observations without updating engine state.

        Computes the instantaneous (unsmoothed) health for each observation —
        useful for scoring a whole dataset at once or for offline analysis.

        Args:
            anomaly_scores: Optional sequence of anomaly scores.
            fault_probabilities: Optional sequence of ``1 - P(normal)`` values.
            deviations: Optional sequence of deviations.

        Returns:
            Array ``(N,)`` of instantaneous health values.

        Raises:
            ValueError: When no sequence is provided or lengths disagree.
        """
        provided = {
            "anomaly_scores": anomaly_scores,
            "fault_probabilities": fault_probabilities,
            "deviations": deviations,
        }
        present = {k: v for k, v in provided.items() if v is not None}
        if not present:
            raise ValueError("At least one signal sequence must be provided")
        lengths = {len(v) for v in present.values()}
        if len(lengths) != 1:
            raise ValueError(f"Signal sequences must share a length, got {lengths}")
        n = lengths.pop()

        out = np.empty(n, dtype=float)
        for i in range(n):
            raw, _ = raw_health_from_signals(
                anomaly_score=(anomaly_scores[i] if anomaly_scores is not None else None),
                fault_probability=(fault_probabilities[i] if fault_probabilities is not None else None),
                deviation=(deviations[i] if deviations is not None else None),
                anomaly_threshold=self._anomaly_threshold,
                anomaly_scale=self.config.anomaly_scale,
                fault_gamma=self.config.fault_gamma,
                weight_anomaly=self.config.weight_anomaly,
                weight_fault=self.config.weight_fault,
                weight_deviation=self.config.weight_deviation,
            )
            out[i] = raw
        return out

    # ------------------------------------------------------------------
    # Trend / degradation analysis
    # ------------------------------------------------------------------

    def _degradation_rate(self) -> float:
        """Estimate the recent degradation rate as a least-squares slope.

        Returns:
            Slope of the recent smoothed history in index points per update
            (negative means health is falling).
        """
        window = self.config.degradation_window
        hist = self._smoothed_history[-window:]
        if len(hist) < 2:
            return 0.0
        x = np.arange(len(hist), dtype=float)
        y = np.asarray(hist, dtype=float)
        # slope of the OLS line y = m x + b
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def _trend(self, degradation_rate: float, *, dead_band: float = 0.1) -> str:
        """Classify the trend direction from the degradation rate.

        Args:
            degradation_rate: The recent slope.
            dead_band: Magnitude below which the trend is considered stable.

        Returns:
            ``"improving"``, ``"stable"``, or ``"degrading"``.
        """
        if degradation_rate > dead_band:
            return "improving"
        if degradation_rate < -dead_band:
            return "degrading"
        return "stable"

    # ------------------------------------------------------------------
    # History access / reset
    # ------------------------------------------------------------------

    @property
    def current_index(self) -> float:
        """The most recent smoothed health index.

        Returns:
            The current index, or 100.0 before any update.
        """
        return self._smoothed if self._smoothed is not None else HEALTH_MAX

    def history(self, *, smoothed: bool = True) -> list[float]:
        """Return the recorded health history.

        Args:
            smoothed: Return the smoothed history (else the raw history).

        Returns:
            A copy of the requested history list.
        """
        return list(self._smoothed_history if smoothed else self._raw_history)

    def reset(self) -> None:
        """Clear all engine state (history, smoothing, monotonic floor)."""
        self._raw_history.clear()
        self._smoothed_history.clear()
        self._smoothed = None
        self._monotonic_floor = HEALTH_MAX
        self._step = 0
        logger.debug("HealthIndexEngine reset")

    def _trim_history(self) -> None:
        """Bound the retained history to ``config.history_size``."""
        cap = self.config.history_size
        if len(self._raw_history) > cap:
            self._raw_history = self._raw_history[-cap:]
            self._smoothed_history = self._smoothed_history[-cap:]

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_state(self, state: HealthState) -> None:
        """Log a state to the experiment tracker (failure-safe).

        Args:
            state: The state to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "health_index": state.index,
                    "health_raw": state.raw_index,
                    "degradation_rate": state.degradation_rate,
                },
                step=state.step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_health(value: float) -> float:
    """Clip a health value to ``[HEALTH_MIN, HEALTH_MAX]``.

    Args:
        value: The value to clip.

    Returns:
        The clipped value.
    """
    if math.isnan(value):
        return HEALTH_MIN
    return float(min(HEALTH_MAX, max(HEALTH_MIN, value)))


def _scalar_or_array(
    result: "np.ndarray", original: Any
) -> "float | np.ndarray":
    """Return a Python float when the input was scalar, else the array.

    Args:
        result: The computed NumPy array.
        original: The original input (to detect scalar vs. array).

    Returns:
        A float for scalar inputs, otherwise the array.
    """
    if np.isscalar(original) or (
        isinstance(original, np.ndarray) and original.ndim == 0
    ):
        return float(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short synthetic degradation demo.

    Returns:
        Exit code 0.
    """
    engine = HealthIndexEngine(HealthIndexConfig(monotonic=True))
    engine.set_anomaly_threshold(0.02)
    # Simulate a slow bearing degradation: anomaly score climbs over time.
    rng = np.random.default_rng(42)
    scores = np.concatenate([
        rng.normal(0.012, 0.002, 20),   # healthy
        np.linspace(0.015, 0.08, 30),   # degrading
    ])
    for s in scores:
        state = engine.update(anomaly_score=float(max(0.0, s)))
    logger.info(
        "Final: index=%.1f status=%s trend=%s rate=%.2f",
        state.index, state.status.value, state.trend, state.degradation_rate,
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
    parser = argparse.ArgumentParser(description="Health Index Engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a synthetic degradation demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered health engines:", list_health_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())