#!/usr/bin/env python3
"""Scenario & what-if simulation engine for the digital-twin platform.

Week-6 Step-2 elevates the :class:`~src.simulation.asset_state_simulator.
AssetStateSimulator` from a *forward* simulator into a true **digital-twin
what-if capability**.  Where Step 1 answers "given these operating conditions,
how will health evolve?", this engine answers the question a reliability
engineer actually asks:

    "What happens to remaining life, failure risk, and cost if we change the
     future operating conditions of this asset?"

The engine runs a **baseline** simulation and a **scenario** (counterfactual)
simulation under modified conditions, then computes a structured, explainable
set of deltas — health, failure time, remaining useful life, failure risk, cost,
and downtime — together with a human-readable summary.  Every delta is grounded
in the same Week-5 prognostic engines used in production, so a what-if answer is
computed by exactly the machinery that scores the real asset.

============================================================================
Scenario taxonomy
============================================================================
Five named scenario archetypes plus a fully-custom escape hatch span the
operational levers an operator can pull:

* ``NORMAL`` — identity scenario; the counterfactual equals the baseline.  Used
  as a control and to validate that the engine introduces no spurious delta.
* ``ACCELERATED_DEGRADATION`` — multiply the degradation rate by a factor > 1,
  modelling harsher duty (higher load, temperature, or contamination).
* ``MAINTENANCE_INTERVENTION`` — restore a chosen amount of health at a chosen
  time, modelling an inspection-and-repair action; the asset then continues to
  degrade from the restored level.
* ``SUDDEN_FAULT`` — inject an instantaneous health drop at a chosen time,
  modelling an abrupt fault event overlaid on the baseline trajectory.
* ``LOAD_REDUCTION`` — multiply the degradation rate by a factor < 1, modelling
  de-rating or a lighter duty cycle.
* ``CUSTOM`` — override any subset of simulator-config fields, for scenarios not
  captured by the archetypes.

============================================================================
Transform mechanisms
============================================================================
Two mechanisms realise the scenarios, chosen so the validated Step-1 simulator
is never modified:

1. **Config mutation** — ``ACCELERATED_DEGRADATION``, ``LOAD_REDUCTION`` and
   ``CUSTOM`` derive a new frozen :class:`AssetSimulatorConfig` from the baseline
   (via :func:`dataclasses.replace`) and re-simulate.  The degradation maths is
   reused verbatim.
2. **Trajectory overlay** — ``MAINTENANCE_INTERVENTION`` and ``SUDDEN_FAULT`` are
   applied as post-simulation transforms on the health array (a health *gain* or
   *loss* applied from the event time onward, re-clipped to ``[0, 100]``).  This
   is the correct mechanism because the base simulator has no native "repair" or
   "inject" concept, and overlaying keeps the baseline degradation shape intact
   between events.

============================================================================
Architecture
============================================================================
* **Pure NumPy** — no SciPy, no PyTorch; deterministic under seeds.
* **Frozen, validated configuration** (:class:`ScenarioConfig`).
* **Immutable, JSON-serialisable result** (:class:`ScenarioResult`).
* **Registry-compatible**, mirroring every Week-1–6 module.
* **Composes the Week-5 prognostic stack** for RUL and risk deltas.

Usage::

    from src.simulation.asset_state_simulator import AssetSimulatorConfig
    from src.simulation.scenario_engine import (
        ScenarioEngine, ScenarioConfig, ScenarioType,
    )

    baseline = AssetSimulatorConfig(
        horizon=120, model="linear", degradation_rate=0.8,
        initial_health=95, failure_threshold=30, noise_std=0.0,
    )
    scenario = ScenarioConfig(
        scenario_type=ScenarioType.MAINTENANCE_INTERVENTION,
        intervention_time=60.0, restoration_amount=35.0,
    )
    result = ScenarioEngine(baseline).run(scenario)
    print(result.summary)
    print(result.to_dict())

CLI::

    python src/simulation/scenario_engine.py --demo
"""

from __future__ import annotations

import dataclasses
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
    DEFAULT_FAILURE_THRESHOLD,
    HEALTH_MAX,
    HEALTH_MIN,
    AssetSimulatorConfig,
    AssetStateSimulator,
    SimulationResult,
)

logger = logging.getLogger("scenario_engine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named scenario engines.
SCENARIO_ENGINE_REGISTRY: dict[str, type] = {}

ENGINE_NAME: Final[str] = "digital_twin_scenario_engine"

#: Sentinel for a remaining-useful-life / failure-time delta that is undefined
#: because a finite value is compared against "never fails".
_UNDEFINED_DELTA: Final[float] = float("nan")

#: Default RUL failure threshold used when chaining the Week-5 RUL predictor.
_DEFAULT_RUL_THRESHOLD: Final[float] = DEFAULT_FAILURE_THRESHOLD

#: Default per-event maintenance and failure costs (currency-agnostic units),
#: matching the Week-5 maintenance decision agent defaults so scenario economics
#: are consistent with production decisioning.
DEFAULT_MAINTENANCE_COST: Final[float] = 5_000.0
DEFAULT_FAILURE_COST: Final[float] = 50_000.0

#: Default downtime (hours) attributed to a maintenance intervention and to an
#: unplanned in-service failure, matching the Week-5 agent's scheduled/failure
#: downtime bands.
DEFAULT_INTERVENTION_DOWNTIME_H: Final[float] = 8.0
DEFAULT_FAILURE_DOWNTIME_H: Final[float] = 72.0

_EPS: Final[float] = 1e-9


# ---------------------------------------------------------------------------
# Scenario-type enum
# ---------------------------------------------------------------------------


class ScenarioType(str, Enum):
    """Supported what-if scenario archetypes."""

    NORMAL = "normal"
    ACCELERATED_DEGRADATION = "accelerated_degradation"
    MAINTENANCE_INTERVENTION = "maintenance_intervention"
    SUDDEN_FAULT = "sudden_fault"
    LOAD_REDUCTION = "load_reduction"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_scenario_engine(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a scenario engine by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = SCENARIO_ENGINE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Scenario engine '{name}' already registered to {existing.__name__}"
            )
        SCENARIO_ENGINE_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered scenario engine '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_scenario_engine(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered scenario engine by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An instantiated scenario engine.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in SCENARIO_ENGINE_REGISTRY:
        available = ", ".join(sorted(SCENARIO_ENGINE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown scenario engine '{name}'. Available: {available}")
    return SCENARIO_ENGINE_REGISTRY[name](**kwargs)


def list_scenario_engines() -> list[str]:
    """Return the sorted names of registered scenario engines.

    Returns:
        Sorted registry keys.
    """
    return sorted(SCENARIO_ENGINE_REGISTRY)


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """Immutable, validated configuration for a single what-if scenario.

    Only the fields relevant to the chosen ``scenario_type`` are used; the
    others are ignored.  Validation enforces that the required fields for the
    chosen type are present and well-formed.

    Attributes:
        scenario_type: The scenario archetype to apply.
        name: Optional human-readable scenario name (used in the report).
        degradation_factor: For ``ACCELERATED_DEGRADATION`` (``> 1``) and
            ``LOAD_REDUCTION`` (``0 <= factor < 1``), the multiplier applied to
            the baseline degradation rate.
        intervention_time: For ``MAINTENANCE_INTERVENTION``, the time at which
            health is restored.
        restoration_amount: For ``MAINTENANCE_INTERVENTION``, the health added at
            the intervention time (clipped to ``[0, 100]``).
        fault_time: For ``SUDDEN_FAULT``, the time of the injected fault.
        fault_magnitude: For ``SUDDEN_FAULT``, the instantaneous health drop.
        overrides: For ``CUSTOM``, a mapping of :class:`AssetSimulatorConfig`
            field names to override values.
        maintenance_cost: Cost attributed to a maintenance intervention.
        failure_cost: Cost attributed to an unplanned in-service failure.
        intervention_downtime_h: Downtime (hours) for a maintenance intervention.
        failure_downtime_h: Downtime (hours) for an unplanned failure.
    """

    scenario_type:           str
    name:                    str = ""
    degradation_factor:      float = 1.0
    intervention_time:       float | None = None
    restoration_amount:      float = 0.0
    fault_time:              float | None = None
    fault_magnitude:         float = 0.0
    overrides:               dict[str, Any] = field(default_factory=dict)
    maintenance_cost:        float = DEFAULT_MAINTENANCE_COST
    failure_cost:            float = DEFAULT_FAILURE_COST
    intervention_downtime_h: float = DEFAULT_INTERVENTION_DOWNTIME_H
    failure_downtime_h:      float = DEFAULT_FAILURE_DOWNTIME_H

    def __post_init__(self) -> None:
        """Validate the scenario configuration.

        Raises:
            ValueError: On an unknown type or malformed type-specific fields.
        """
        valid = {s.value for s in ScenarioType}
        if self.scenario_type not in valid:
            raise ValueError(
                f"scenario_type must be one of {sorted(valid)}, got "
                f"'{self.scenario_type}'"
            )
        if self.maintenance_cost < 0 or self.failure_cost < 0:
            raise ValueError("costs must be >= 0")
        if self.intervention_downtime_h < 0 or self.failure_downtime_h < 0:
            raise ValueError("downtime must be >= 0")

        if self.scenario_type == ScenarioType.ACCELERATED_DEGRADATION.value:
            if self.degradation_factor <= 1.0:
                raise ValueError(
                    "accelerated_degradation requires degradation_factor > 1"
                )
        elif self.scenario_type == ScenarioType.LOAD_REDUCTION.value:
            if not (0.0 <= self.degradation_factor < 1.0):
                raise ValueError(
                    "load_reduction requires 0 <= degradation_factor < 1"
                )
        elif self.scenario_type == ScenarioType.MAINTENANCE_INTERVENTION.value:
            if self.intervention_time is None or self.intervention_time < 0:
                raise ValueError(
                    "maintenance_intervention requires intervention_time >= 0"
                )
            if self.restoration_amount <= 0:
                raise ValueError(
                    "maintenance_intervention requires restoration_amount > 0"
                )
        elif self.scenario_type == ScenarioType.SUDDEN_FAULT.value:
            if self.fault_time is None or self.fault_time < 0:
                raise ValueError("sudden_fault requires fault_time >= 0")
            if self.fault_magnitude <= 0:
                raise ValueError("sudden_fault requires fault_magnitude > 0")
        elif self.scenario_type == ScenarioType.CUSTOM.value:
            if not self.overrides:
                raise ValueError("custom requires a non-empty overrides mapping")
            allowed = {f.name for f in dataclasses.fields(AssetSimulatorConfig)}
            bad = set(self.overrides) - allowed
            if bad:
                raise ValueError(
                    f"custom overrides contain unknown config fields: {sorted(bad)}"
                )


# ---------------------------------------------------------------------------
# Scenario result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioResult:
    """Immutable, JSON-serialisable result of a what-if comparison.

    Deltas follow the convention *scenario minus baseline*: a positive
    ``health_delta`` means the scenario is healthier than the baseline, a
    positive ``rul_delta`` means the scenario has longer remaining life, and a
    negative ``risk_delta`` means the scenario is less risky.

    Attributes:
        scenario_type: The applied scenario archetype.
        scenario_name: The human-readable scenario name.
        times: Shared sample-time grid ``(N,)``.
        baseline_health: Baseline health trajectory ``(N,)``.
        scenario_health: Scenario health trajectory ``(N,)``.
        health_delta: Per-sample health difference ``(N,)`` (scenario − baseline).
        mean_health_delta: Mean of ``health_delta``.
        final_health_delta: Health difference at the final sample.
        baseline_failure_time: Baseline failure time (``-1`` if it never fails).
        scenario_failure_time: Scenario failure time (``-1`` if it never fails).
        failure_time_delta: Scenario − baseline failure time; ``NaN`` when only
            one of the two trajectories fails (use ``failure_avoided`` /
            ``failure_induced`` to interpret).
        failure_avoided: True when the baseline fails but the scenario does not.
        failure_induced: True when the scenario fails but the baseline does not.
        baseline_rul: Baseline remaining useful life (may be ``inf``).
        scenario_rul: Scenario remaining useful life (may be ``inf``).
        rul_delta: Scenario − baseline RUL; ``NaN`` when exactly one is infinite.
        baseline_risk: Baseline dominant-horizon failure probability.
        scenario_risk: Scenario dominant-horizon failure probability.
        risk_delta: Scenario − baseline risk.
        cost_impact: Expected cost of the scenario minus the baseline (negative
            means the scenario saves money).
        downtime_impact: Expected downtime hours of the scenario minus baseline.
        summary: Human-readable explainability summary.
    """

    scenario_type:          str
    scenario_name:          str
    times:                  "np.ndarray"
    baseline_health:        "np.ndarray"
    scenario_health:        "np.ndarray"
    health_delta:           "np.ndarray"
    mean_health_delta:      float
    final_health_delta:     float
    baseline_failure_time:  float
    scenario_failure_time:  float
    failure_time_delta:     float
    failure_avoided:        bool
    failure_induced:        bool
    baseline_rul:           float
    scenario_rul:           float
    rul_delta:              float
    baseline_risk:          float
    scenario_risk:          float
    risk_delta:             float
    cost_impact:            float
    downtime_impact:        float
    summary:                str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Non-finite floats (``inf``/``NaN``) are rendered as ``None`` so the
        payload is valid JSON.

        Returns:
            Dictionary representation; arrays are rendered as lists.
        """
        def _jsonsafe(x: float) -> float | None:
            return None if (math.isinf(x) or math.isnan(x)) else float(x)

        return {
            "scenario_type": self.scenario_type,
            "scenario_name": self.scenario_name,
            "times": [float(t) for t in self.times],
            "baseline_health": [float(h) for h in self.baseline_health],
            "scenario_health": [float(h) for h in self.scenario_health],
            "health_delta": [float(d) for d in self.health_delta],
            "mean_health_delta": float(self.mean_health_delta),
            "final_health_delta": float(self.final_health_delta),
            "baseline_failure_time": _jsonsafe(self.baseline_failure_time),
            "scenario_failure_time": _jsonsafe(self.scenario_failure_time),
            "failure_time_delta": _jsonsafe(self.failure_time_delta),
            "failure_avoided": self.failure_avoided,
            "failure_induced": self.failure_induced,
            "baseline_rul": _jsonsafe(self.baseline_rul),
            "scenario_rul": _jsonsafe(self.scenario_rul),
            "rul_delta": _jsonsafe(self.rul_delta),
            "baseline_risk": float(self.baseline_risk),
            "scenario_risk": float(self.scenario_risk),
            "risk_delta": float(self.risk_delta),
            "cost_impact": float(self.cost_impact),
            "downtime_impact": float(self.downtime_impact),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------


@register_scenario_engine(ENGINE_NAME)
class ScenarioEngine:
    """Runs baseline-vs-scenario what-if comparisons on an asset configuration.

    The engine owns a baseline :class:`AssetSimulatorConfig`.  Each call to
    :meth:`run` simulates that baseline, derives and simulates the counterfactual
    for the supplied :class:`ScenarioConfig`, and returns a fully-populated
    :class:`ScenarioResult` with explainable deltas.  RUL and risk deltas are
    computed by chaining the Week-5 prognostic engines when they are available;
    if they cannot be imported the engine degrades gracefully to health- and
    failure-time-based deltas (the RUL/risk fields become ``NaN``/``0`` rather
    than raising), preserving operability in a minimal deployment.

    Args:
        baseline_config: The asset configuration to use as the baseline.
        experiment_tracker: Optional tracker for logging scenario summaries.
        rul_threshold: Failure threshold for the chained RUL predictor.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        baseline_config: AssetSimulatorConfig | None = None,
        experiment_tracker: Any = None,
        rul_threshold: float = _DEFAULT_RUL_THRESHOLD,
    ) -> None:
        self.baseline_config = baseline_config or AssetSimulatorConfig()
        self.tracker = experiment_tracker
        self.rul_threshold = float(rul_threshold)
        self._n_scenarios = 0
        logger.info(
            "ScenarioEngine ready | baseline model=%s | horizon=%.1f",
            self.baseline_config.model, self.baseline_config.horizon,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, scenario: ScenarioConfig) -> ScenarioResult:
        """Run a baseline-vs-scenario comparison.

        Args:
            scenario: The what-if scenario to evaluate.

        Returns:
            A populated :class:`ScenarioResult`.
        """
        baseline_result = AssetStateSimulator(self.baseline_config).simulate()
        scenario_result = self._simulate_scenario(scenario)
        out = self.compare(baseline_result, scenario_result, scenario)
        self._log_result(out)
        self._n_scenarios += 1
        return out

    def compare(
        self,
        baseline: SimulationResult,
        scenario: SimulationResult,
        scenario_cfg: ScenarioConfig,
    ) -> ScenarioResult:
        """Compute structured deltas between two simulation results.

        The two results must share a sample grid (they do when both derive from
        the engine's baseline horizon/timestep).  Deltas follow the convention
        *scenario minus baseline*.

        Args:
            baseline: The baseline simulation result.
            scenario: The scenario simulation result.
            scenario_cfg: The scenario configuration (for naming and economics).

        Returns:
            A populated :class:`ScenarioResult`.

        Raises:
            ValueError: When the two trajectories have different lengths.
        """
        if baseline.n_samples != scenario.n_samples:
            raise ValueError(
                "baseline and scenario trajectories must share a sample grid"
            )
        bh = np.asarray(baseline.health, dtype=float)
        sh = np.asarray(scenario.health, dtype=float)
        delta = sh - bh

        # --- Failure-time delta with avoided / induced handling ---
        b_ft = baseline.failure_time
        s_ft = scenario.failure_time
        b_failed = baseline.crossed_failure
        s_failed = scenario.crossed_failure
        failure_avoided = b_failed and not s_failed
        failure_induced = s_failed and not b_failed
        if b_failed and s_failed:
            failure_time_delta = float(s_ft - b_ft)
        else:
            # Exactly one (or neither) fails -> a finite difference is undefined.
            failure_time_delta = _UNDEFINED_DELTA

        # --- RUL & risk deltas via the Week-5 chain (graceful if absent) ---
        b_rul, b_risk = self._prognostics(bh)
        s_rul, s_risk = self._prognostics(sh)
        rul_delta = self._safe_delta(s_rul, b_rul)
        risk_delta = float(s_risk - b_risk)

        # --- Cost & downtime impact ---
        cost_impact, downtime_impact = self._economics(
            baseline, scenario, scenario_cfg,
        )

        summary = self._summarise(
            scenario_cfg, delta, failure_avoided, failure_induced,
            failure_time_delta, rul_delta, risk_delta, cost_impact,
            downtime_impact,
        )

        return ScenarioResult(
            scenario_type=scenario_cfg.scenario_type,
            scenario_name=scenario_cfg.name or scenario_cfg.scenario_type,
            times=np.asarray(baseline.times, dtype=float),
            baseline_health=bh,
            scenario_health=sh,
            health_delta=delta,
            mean_health_delta=float(np.mean(delta)),
            final_health_delta=float(delta[-1]),
            baseline_failure_time=float(b_ft),
            scenario_failure_time=float(s_ft),
            failure_time_delta=failure_time_delta,
            failure_avoided=failure_avoided,
            failure_induced=failure_induced,
            baseline_rul=b_rul,
            scenario_rul=s_rul,
            rul_delta=rul_delta,
            baseline_risk=b_risk,
            scenario_risk=s_risk,
            risk_delta=risk_delta,
            cost_impact=cost_impact,
            downtime_impact=downtime_impact,
            summary=summary,
        )

    def generate_report(self, results: Sequence[ScenarioResult]) -> str:
        """Render a multi-scenario comparison report as plain text.

        Args:
            results: One or more scenario results (e.g. a portfolio of what-ifs).

        Returns:
            A formatted, human-readable report.

        Raises:
            ValueError: When *results* is empty.
        """
        if not results:
            raise ValueError("generate_report requires at least one result")
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("DIGITAL TWIN — WHAT-IF SCENARIO REPORT")
        lines.append("=" * 72)
        lines.append(
            f"Baseline: model={self.baseline_config.model} "
            f"horizon={self.baseline_config.horizon:g} "
            f"rate={self.baseline_config.degradation_rate:g} "
            f"threshold={self.baseline_config.failure_threshold:g}"
        )
        lines.append("-" * 72)
        header = (f"{'Scenario':<26}{'ΔMeanH':>9}{'ΔRUL':>9}"
                  f"{'ΔRisk':>9}{'ΔCost':>11}{'ΔDown(h)':>10}")
        lines.append(header)
        lines.append("-" * 72)
        for r in results:
            rul = "n/a" if math.isnan(r.rul_delta) or math.isinf(r.rul_delta) \
                else f"{r.rul_delta:+.1f}"
            lines.append(
                f"{r.scenario_name[:25]:<26}"
                f"{r.mean_health_delta:>+9.1f}"
                f"{rul:>9}"
                f"{r.risk_delta:>+9.3f}"
                f"{r.cost_impact:>+11.0f}"
                f"{r.downtime_impact:>+10.1f}"
            )
        lines.append("-" * 72)
        for r in results:
            lines.append(f"\n[{r.scenario_name}] {r.summary}")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scenario realisation
    # ------------------------------------------------------------------

    def _simulate_scenario(self, scenario: ScenarioConfig) -> SimulationResult:
        """Simulate the counterfactual trajectory for *scenario*.

        Config-mutation scenarios derive a new config and re-simulate; overlay
        scenarios simulate the baseline and transform the resulting health array.

        Args:
            scenario: The scenario configuration.

        Returns:
            The scenario :class:`SimulationResult`.
        """
        stype = scenario.scenario_type
        base = self.baseline_config

        if stype == ScenarioType.NORMAL.value:
            return AssetStateSimulator(base).simulate()

        if stype == ScenarioType.ACCELERATED_DEGRADATION.value:
            cfg = self._scaled_rate_config(base, scenario.degradation_factor)
            return AssetStateSimulator(cfg).simulate()

        if stype == ScenarioType.LOAD_REDUCTION.value:
            cfg = self._scaled_rate_config(base, scenario.degradation_factor)
            return AssetStateSimulator(cfg).simulate()

        if stype == ScenarioType.CUSTOM.value:
            cfg = dataclasses.replace(base, **scenario.overrides)
            return AssetStateSimulator(cfg).simulate()

        if stype == ScenarioType.MAINTENANCE_INTERVENTION.value:
            base_result = AssetStateSimulator(base).simulate()
            return self._overlay_health(
                base_result, scenario.intervention_time,
                +float(scenario.restoration_amount),
            )

        # SUDDEN_FAULT
        base_result = AssetStateSimulator(base).simulate()
        return self._overlay_health(
            base_result, scenario.fault_time,
            -float(scenario.fault_magnitude),
        )

    @staticmethod
    def _scaled_rate_config(
        base: AssetSimulatorConfig, factor: float
    ) -> AssetSimulatorConfig:
        """Return a config with degradation rates scaled by *factor*.

        Both the primary ``degradation_rate`` and (for completeness) the
        piecewise ``segment_rates`` and the sudden-fault ``fault_baseline_rate``
        are scaled, so the scenario is well-defined for every degradation model.

        Args:
            base: The baseline configuration.
            factor: The multiplicative scaling factor.

        Returns:
            A new frozen configuration.
        """
        return dataclasses.replace(
            base,
            degradation_rate=base.degradation_rate * factor,
            fault_baseline_rate=base.fault_baseline_rate * factor,
            segment_rates=tuple(r * factor for r in base.segment_rates),
        )

    def _overlay_health(
        self, base_result: SimulationResult, event_time: float, delta: float
    ) -> SimulationResult:
        """Apply a health gain/loss from *event_time* onward and rebuild a result.

        Args:
            base_result: The baseline simulation result to transform.
            event_time: The time at which the overlay takes effect.
            delta: The health change applied from the event onward (positive for
                restoration, negative for a fault).

        Returns:
            A new :class:`SimulationResult` with the overlaid trajectory and
            recomputed failure bookkeeping (against the baseline threshold).
        """
        times = np.asarray(base_result.times, dtype=float)
        health = np.asarray(base_result.health, dtype=float).copy()
        idx = int(np.searchsorted(times, event_time, side="left"))
        if idx < health.size:
            health[idx:] = np.clip(health[idx:] + delta, HEALTH_MIN, HEALTH_MAX)

        threshold = self.baseline_config.failure_threshold
        at_or_below = health <= (threshold + _EPS)
        crossed = bool(at_or_below.any())
        first = int(np.argmax(at_or_below)) if crossed else -1
        fail_time = float(times[first]) if crossed else -1.0

        # Preserve any pre-existing fault steps from the baseline result and add
        # the overlay event when it is a fault (negative delta).
        fault_steps = set(base_result.fault_steps)
        if delta < 0 and idx < health.size:
            fault_steps.add(idx)

        return dataclasses.replace(
            base_result,
            health=health,
            fault_steps=tuple(sorted(fault_steps)),
            final_health=float(health[-1]),
            crossed_failure=crossed,
            first_failure_step=first,
            failure_time=fail_time,
        )

    # ------------------------------------------------------------------
    # Prognostics chain (Week-5)
    # ------------------------------------------------------------------

    def _prognostics(self, health: "np.ndarray") -> tuple[float, float]:
        """Compute (RUL, dominant-horizon risk) for a trajectory via Week-5.

        Degrades gracefully: if the Week-5 engines cannot be imported, returns
        ``(NaN, 0.0)`` so the scenario comparison still produces health- and
        failure-time deltas.

        Args:
            health: A health trajectory.

        Returns:
            Tuple ``(rul, risk)``; ``rul`` may be ``inf`` for a non-degrading
            trajectory.
        """
        try:
            from src.predictive.failure_risk import (
                FailureRiskConfig,
                FailureRiskEngine,
            )
            from src.predictive.rul_predictor import RULConfig, RULPredictor
        except Exception as exc:  # noqa: BLE001
            logger.debug("Week-5 prognostics unavailable: %s", exc)
            return _UNDEFINED_DELTA, 0.0

        try:
            rul_pred = RULPredictor(
                RULConfig(failure_threshold=self.rul_threshold, model="linear")
            ).predict(health)
            risk_pred = FailureRiskEngine(FailureRiskConfig()).predict_from_rul(
                rul_pred
            )
            return float(rul_pred.rul), float(risk_pred.dominant_probability)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Prognostics computation failed: %s", exc)
            return _UNDEFINED_DELTA, 0.0

    @staticmethod
    def _safe_delta(scenario_val: float, baseline_val: float) -> float:
        """Difference of two possibly-infinite values, ``NaN`` when ambiguous.

        ``inf - inf`` and ``finite - inf`` are undefined for ranking purposes, so
        they collapse to ``NaN``; two finite values subtract normally.

        Args:
            scenario_val: The scenario value.
            baseline_val: The baseline value.

        Returns:
            The delta, or ``NaN`` when exactly one operand is infinite or both
            are infinite.
        """
        s_inf = math.isinf(scenario_val)
        b_inf = math.isinf(baseline_val)
        if s_inf or b_inf:
            if s_inf and b_inf:
                return 0.0  # both never fail -> no difference
            return _UNDEFINED_DELTA
        return float(scenario_val - baseline_val)

    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    def _economics(
        self,
        baseline: SimulationResult,
        scenario: SimulationResult,
        cfg: ScenarioConfig,
    ) -> tuple[float, float]:
        """Compute the cost and downtime impact of the scenario vs baseline.

        The model is intentionally transparent: each in-service failure carries
        ``failure_cost`` and ``failure_downtime_h``; a maintenance intervention
        carries ``maintenance_cost`` and ``intervention_downtime_h``.  The impact
        is the scenario's expected cost/downtime minus the baseline's.

        Args:
            baseline: The baseline result.
            scenario: The scenario result.
            cfg: The scenario configuration (supplies the cost/downtime rates).

        Returns:
            Tuple ``(cost_impact, downtime_impact)``; negative means the scenario
            is cheaper / less down than the baseline.
        """
        def failure_cost(result: SimulationResult) -> float:
            return cfg.failure_cost if result.crossed_failure else 0.0

        def failure_down(result: SimulationResult) -> float:
            return cfg.failure_downtime_h if result.crossed_failure else 0.0

        base_cost = failure_cost(baseline)
        scen_cost = failure_cost(scenario)
        base_down = failure_down(baseline)
        scen_down = failure_down(scenario)

        # A maintenance intervention adds its own planned cost/downtime.
        if cfg.scenario_type == ScenarioType.MAINTENANCE_INTERVENTION.value:
            scen_cost += cfg.maintenance_cost
            scen_down += cfg.intervention_downtime_h

        return float(scen_cost - base_cost), float(scen_down - base_down)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(
        cfg: ScenarioConfig,
        delta: "np.ndarray",
        failure_avoided: bool,
        failure_induced: bool,
        failure_time_delta: float,
        rul_delta: float,
        risk_delta: float,
        cost_impact: float,
        downtime_impact: float,
    ) -> str:
        """Compose a human-readable explainability summary.

        Args:
            cfg: The scenario configuration.
            delta: The per-sample health delta.
            failure_avoided: Whether the scenario averts a baseline failure.
            failure_induced: Whether the scenario causes a new failure.
            failure_time_delta: The failure-time delta (may be ``NaN``).
            rul_delta: The RUL delta (may be ``NaN``).
            risk_delta: The risk delta.
            cost_impact: The cost impact.
            downtime_impact: The downtime impact.

        Returns:
            A single-paragraph summary.
        """
        name = cfg.name or cfg.scenario_type.replace("_", " ")
        parts: list[str] = [f"Scenario '{name}':"]

        mean_d = float(np.mean(delta))
        if abs(mean_d) < 0.05:
            parts.append("negligible change in mean health")
        elif mean_d > 0:
            parts.append(f"mean health improved by {mean_d:.1f} points")
        else:
            parts.append(f"mean health worsened by {abs(mean_d):.1f} points")

        if failure_avoided:
            parts.append("and the projected failure is AVOIDED")
        elif failure_induced:
            parts.append("and a new failure is INDUCED")
        elif not math.isnan(failure_time_delta):
            if failure_time_delta > _EPS:
                parts.append(f"failure deferred by {failure_time_delta:.1f} time units")
            elif failure_time_delta < -_EPS:
                parts.append(
                    f"failure brought forward by {abs(failure_time_delta):.1f} "
                    "time units"
                )
            else:
                parts.append("failure timing unchanged")

        if not math.isnan(rul_delta) and not math.isinf(rul_delta):
            if rul_delta > _EPS:
                parts.append(f"RUL extended by {rul_delta:.1f}")
            elif rul_delta < -_EPS:
                parts.append(f"RUL shortened by {abs(rul_delta):.1f}")

        if abs(risk_delta) >= 0.005:
            direction = "reduced" if risk_delta < 0 else "increased"
            parts.append(f"failure risk {direction} by {abs(risk_delta):.3f}")

        if abs(cost_impact) >= 1.0:
            verb = "saves" if cost_impact < 0 else "adds"
            parts.append(f"and {verb} {abs(cost_impact):,.0f} in expected cost")

        return " ".join(parts) + "."

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_result(self, result: ScenarioResult) -> None:
        """Log a scenario summary to the experiment tracker (failure-safe).

        Args:
            result: The scenario result to log.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {
                    "scenario_mean_health_delta": result.mean_health_delta,
                    "scenario_final_health_delta": result.final_health_delta,
                    "scenario_risk_delta": result.risk_delta,
                    "scenario_cost_impact": result.cost_impact,
                    "scenario_downtime_impact": result.downtime_impact,
                    "scenario_failure_avoided": float(result.failure_avoided),
                    "scenario_failure_induced": float(result.failure_induced),
                },
                step=self._n_scenarios,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)
        try:
            log_params = getattr(self.tracker, "log_params", None)
            if callable(log_params):
                log_params({"scenario_type": result.scenario_type})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_params failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short what-if demo across the scenario archetypes.

    Returns:
        Exit code 0.
    """
    baseline = AssetSimulatorConfig(
        horizon=120, timestep=1.0, model="linear", degradation_rate=0.7,
        initial_health=95.0, failure_threshold=30.0, noise_std=0.0,
    )
    engine = ScenarioEngine(baseline)
    scenarios = [
        ScenarioConfig(scenario_type=ScenarioType.NORMAL.value, name="Baseline control"),
        ScenarioConfig(scenario_type=ScenarioType.ACCELERATED_DEGRADATION.value,
                       name="Harsh duty +80%", degradation_factor=1.8),
        ScenarioConfig(scenario_type=ScenarioType.LOAD_REDUCTION.value,
                       name="De-rate -40%", degradation_factor=0.6),
        ScenarioConfig(scenario_type=ScenarioType.MAINTENANCE_INTERVENTION.value,
                       name="Mid-life overhaul", intervention_time=60.0,
                       restoration_amount=35.0),
        ScenarioConfig(scenario_type=ScenarioType.SUDDEN_FAULT.value,
                       name="Bearing strike @40", fault_time=40.0,
                       fault_magnitude=30.0),
    ]
    results = [engine.run(s) for s in scenarios]
    print(engine.generate_report(results))
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
    parser = argparse.ArgumentParser(description="Digital-twin scenario engine")
    parser.add_argument("--demo", action="store_true",
                        help="Run a what-if scenario demo.")
    parser.add_argument("--list-engines", action="store_true")
    args = parser.parse_args(argv)

    if args.list_engines:
        print("Registered scenario engines:", list_scenario_engines())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())