#!/usr/bin/env python3
"""Platform API — the executive dashboard and REST API layer.

Week-8 Phase-5 exposes the entire Enterprise Digital Twin platform through a
production-style API.  Executives (or a dashboard front-end) submit raw fleet
telemetry and receive fleet overviews, top risks, root causes, decisions,
scenario outcomes, Monte Carlo risk metrics, and full executive reports over
REST.

============================================================================
Architecture — transport-agnostic core, optional FastAPI shell
============================================================================
All request processing lives in :class:`PlatformAPIServer`.  Its
:meth:`PlatformAPIServer.dispatch` method performs the exact routing,
validation, composition, and serialisation that an HTTP layer would, returning
a ``(status_code, body)`` pair.  This keeps the whole API framework-independent,
deterministic, and fully testable with no web dependencies — matching the
platform's pure-Python standard.

:func:`create_app` builds a FastAPI application on top of the server via a lazy,
guarded import and wires every route to ``dispatch`` through dependency
injection.  If FastAPI is not installed, :func:`create_app` raises a clear
:class:`RuntimeError`, but the server itself remains fully functional.

Clients send raw telemetry — per-asset health trajectories plus optional
subsystem evidence, a budget, and a scenario flag.  The server builds a
:class:`FleetSnapshot` with the Fleet Digital Twin and then composes the
Executive Intelligence, Root-Cause, Scenario, Executive Decision, and Monte
Carlo modules.  No business logic is duplicated here.

============================================================================
Endpoints
============================================================================
* ``GET  /health``                — liveness probe.
* ``POST /fleet/summary``         — executive intelligence report summary.
* ``POST /fleet/top-risks``       — top assets by executive priority.
* ``POST /fleet/root-causes``     — dominant root-cause findings.
* ``POST /fleet/decisions``       — executive decision portfolio.
* ``POST /fleet/scenarios``       — scenario recommendations.
* ``POST /fleet/monte-carlo``     — Monte Carlo fleet risk statistics.
* ``POST /fleet/executive-report``— full executive intelligence report.

Run (when FastAPI + uvicorn are installed)::

    uvicorn src.api.platform_api:app --reload

Demo (always available)::

    python src/api/platform_api.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.simulation.asset_state_simulator import AssetSimulatorConfig  # noqa: E402
from src.risk.monte_carlo_engine import (  # noqa: E402
    MonteCarloConfig,
    MonteCarloEngine,
)
from src.fleet.fleet_digital_twin import (  # noqa: E402
    AssetInput,
    FleetDigitalTwinConfig,
    FleetDigitalTwinEngine,
    FleetSnapshot,
)
from src.executive.executive_decision_engine import (  # noqa: E402
    ExecutiveDecisionConfig,
    ExecutiveDecisionEngine,
)
from src.agent.root_cause_analysis_agent import AssetEvidence  # noqa: E402
from src.agent.scenario_planning_agent import (  # noqa: E402
    ScenarioPlanningAgent,
    ScenarioPlanningConfig,
)
from src.agent.executive_intelligence_agent import (  # noqa: E402
    ExecutiveIntelligenceAgent,
    ExecutiveIntelligenceConfig,
)

logger = logging.getLogger("platform_api")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION: Final[str] = "1.0"
_MIN_TRAJECTORY: Final[int] = 5
_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "temperature", "vibration", "pressure", "load", "lubrication",
    "electrical", "environmental", "operational",
)


def _jsonsafe(x: Any) -> Any:
    """Recursively render a value JSON-safe (non-finite floats -> ``None``).

    Args:
        x: Any value (scalar, list, tuple, or dict).

    Returns:
        A JSON-serialisable structure.
    """
    if isinstance(x, float):
        return None if (math.isinf(x) or math.isnan(x)) else x
    if isinstance(x, dict):
        return {k: _jsonsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonsafe(v) for v in x]
    if isinstance(x, (np.floating,)):
        return _jsonsafe(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlatformAPIError(Exception):
    """A structured API error carrying an HTTP status code.

    Attributes:
        status_code: The HTTP status code to return.
        message: A human-readable error message.
        detail: Optional additional detail.
    """

    def __init__(self, status_code: int, message: str,
                 detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = str(message)
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable error body."""
        return {
            "status_code": self.status_code,
            "message": self.message,
            "detail": _jsonsafe(self.detail),
        }


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformAPIResponse:
    """A uniform API response envelope.

    Attributes:
        status: ``"ok"`` for success, ``"error"`` otherwise.
        endpoint: The endpoint that produced the response.
        data: The response payload (``None`` on error).
        error: The error body (``None`` on success).
    """

    status:   str
    endpoint: str
    data:     dict[str, Any] | None = None
    error:    dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "status": self.status,
            "endpoint": self.endpoint,
            "data": _jsonsafe(self.data),
            "error": _jsonsafe(self.error),
        }

    @staticmethod
    def ok(endpoint: str, data: dict[str, Any]) -> "PlatformAPIResponse":
        """Build a success response.

        Args:
            endpoint: The endpoint name.
            data: The payload.

        Returns:
            A success :class:`PlatformAPIResponse`.
        """
        return PlatformAPIResponse(status="ok", endpoint=endpoint, data=data)

    @staticmethod
    def fail(endpoint: str, err: PlatformAPIError) -> "PlatformAPIResponse":
        """Build an error response.

        Args:
            endpoint: The endpoint name.
            err: The error.

        Returns:
            An error :class:`PlatformAPIResponse`.
        """
        return PlatformAPIResponse(status="error", endpoint=endpoint,
                                   error=err.to_dict())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformAPIConfig:
    """Configuration for the :class:`PlatformAPIServer`.

    Attributes:
        title: API title.
        version: API version.
        currency: Currency label for monetary figures.
        top_n: Number of top items returned by list endpoints.
        include_scenarios_default: Whether the summary/report endpoints run
            scenario analysis when the request does not specify.
        mc_trials: Monte Carlo trial count.
        mc_seed: Monte Carlo random seed (for deterministic responses).
        mc_failure_threshold: Simulator failure threshold for Monte Carlo.
        mc_horizon: Simulator horizon for Monte Carlo.
        max_assets: Maximum number of assets accepted in one request.
    """

    title:                      str = "Enterprise Digital Twin Platform API"
    version:                    str = API_VERSION
    currency:                   str = "USD"
    top_n:                      int = 5
    include_scenarios_default:  bool = True
    mc_trials:                  int = 1000
    mc_seed:                    int = 0
    mc_failure_threshold:       float = 30.0
    mc_horizon:                 float = 200.0
    max_assets:                 int = 10_000

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid parameters.
        """
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.mc_trials < 1:
            raise ValueError("mc_trials must be >= 1")
        if self.mc_horizon <= 0:
            raise ValueError("mc_horizon must be > 0")
        if not (0.0 < self.mc_failure_threshold < 100.0):
            raise ValueError("mc_failure_threshold must be in (0, 100)")
        if self.max_assets < 1:
            raise ValueError("max_assets must be >= 1")


# ---------------------------------------------------------------------------
# Platform API server (transport-agnostic core)
# ---------------------------------------------------------------------------


class PlatformAPIServer:
    """The transport-agnostic API core composing the whole platform.

    The server owns one instance of each composed module and exposes a handler
    per endpoint plus a :meth:`dispatch` method that mirrors HTTP routing.  It
    holds no per-request mutable state and is deterministic.

    Args:
        config: The API configuration.
        intelligence_agent: Optional Executive Intelligence Agent (created if
            None) — itself composes the Copilot, Root-Cause, and Scenario agents.
        experiment_tracker: Optional tracker for request counts.
    """

    def __init__(
        self,
        config: PlatformAPIConfig | None = None,
        intelligence_agent: ExecutiveIntelligenceAgent | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or PlatformAPIConfig()
        cfg = self.config
        if intelligence_agent is not None:
            self.intelligence = intelligence_agent
        else:
            # Build a currency-matched scenario agent and inject it, so the
            # scenario endpoint reports figures in the configured currency
            # (the intelligence agent otherwise defaults its scenario agent).
            scenario_agent = ScenarioPlanningAgent(
                ScenarioPlanningConfig(currency=cfg.currency))
            self.intelligence = ExecutiveIntelligenceAgent(
                ExecutiveIntelligenceConfig(
                    top_n=cfg.top_n,
                    include_scenarios_default=cfg.include_scenarios_default,
                    currency=cfg.currency),
                scenario_agent=scenario_agent)
        # Direct handles for the decision / scenario / Monte Carlo endpoints.
        self.fleet_engine = FleetDigitalTwinEngine(FleetDigitalTwinConfig())
        self.scenario_agent = self.intelligence.scenario_agent
        self.root_cause_agent = self.intelligence.root_cause_agent
        self.monte_carlo = MonteCarloEngine(
            MonteCarloConfig(n_trials=cfg.mc_trials, random_seed=cfg.mc_seed))
        self.tracker = experiment_tracker
        self._n_requests = 0
        # Route table: (method, path) -> handler.
        self._routes: dict[tuple[str, str], Callable[[Any], PlatformAPIResponse]] = {
            ("GET", "/health"): lambda body: self.health(),
            ("POST", "/fleet/summary"): self.fleet_summary,
            ("POST", "/fleet/top-risks"): self.top_risks,
            ("POST", "/fleet/root-causes"): self.root_causes,
            ("POST", "/fleet/decisions"): self.decisions,
            ("POST", "/fleet/scenarios"): self.scenarios,
            ("POST", "/fleet/monte-carlo"): self.monte_carlo_stats,
            ("POST", "/fleet/executive-report"): self.executive_report,
        }
        logger.info("PlatformAPIServer ready | version=%s", cfg.version)

    # ------------------------------------------------------------------
    # Dispatch (mirrors HTTP routing)
    # ------------------------------------------------------------------

    def dispatch(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Route a request to its handler exactly as the HTTP layer would.

        Args:
            method: The HTTP method (e.g. ``"GET"``, ``"POST"``).
            path: The request path (e.g. ``"/fleet/summary"``).
            body: The parsed request body, if any.

        Returns:
            A ``(status_code, response_body)`` pair.  ``status_code`` is 200 on
            success, 400 on validation error, 404 for an unknown path, 405 for
            an unsupported method, and 500 for an unexpected internal error.
        """
        method = str(method).upper()
        path = self._normalise_path(path)
        handler = self._routes.get((method, path))
        if handler is None:
            # Distinguish unknown path (404) from wrong method (405).
            known_paths = {p for _, p in self._routes}
            if path in known_paths:
                return 405, PlatformAPIResponse.fail(
                    path, PlatformAPIError(
                        405, f"Method {method} not allowed for {path}")).to_dict()
            return 404, PlatformAPIResponse.fail(
                path, PlatformAPIError(404, f"Unknown endpoint: {path}")).to_dict()
        try:
            response = handler(body)
            self._n_requests += 1
            self._log()
            return 200, response.to_dict()
        except PlatformAPIError as exc:
            return exc.status_code, PlatformAPIResponse.fail(path, exc).to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error at %s", path)
            return 500, PlatformAPIResponse.fail(
                path, PlatformAPIError(500, "Internal server error",
                                       detail=str(exc))).to_dict()

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalise a request path (strip query and trailing slash).

        Args:
            path: The raw path.

        Returns:
            The normalised path.
        """
        p = str(path).split("?", 1)[0]
        if len(p) > 1 and p.endswith("/"):
            p = p.rstrip("/")
        return p or "/"

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def health(self) -> PlatformAPIResponse:
        """Return a liveness response.

        Returns:
            A success response with ``{"status": "healthy"}``.
        """
        return PlatformAPIResponse.ok("/health", {"status": "healthy",
                                                   "version": self.config.version})

    def fleet_summary(self, body: dict[str, Any] | None) -> PlatformAPIResponse:
        """Return an executive intelligence report summary.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the report summary.
        """
        snapshot, evidence, budget, scenarios = self._parse_request(body)
        report = self.intelligence.generate_report(
            snapshot, evidence_items=evidence, budget=budget,
            include_scenarios=scenarios)
        data = {
            "fleet_overview": report.fleet_overview,
            "current_health": report.current_health,
            "current_risk": report.current_risk,
            "current_rul": report.current_rul,
            "executive_summary": report.executive_summary,
            "confidence": report.confidence,
            "currency": report.currency,
        }
        return PlatformAPIResponse.ok("/fleet/summary", data)

    def top_risks(self, body: dict[str, Any] | None) -> PlatformAPIResponse:
        """Return the top assets by executive priority.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the top risks.
        """
        snapshot, _, _, _ = self._parse_request(body)
        risks = self.intelligence.risk_assessment(snapshot)
        return PlatformAPIResponse.ok(
            "/fleet/top-risks", {"top_risks": [r.to_dict() for r in risks],
                                 "count": len(risks)})

    def root_causes(self, body: dict[str, Any] | None) -> PlatformAPIResponse:
        """Return dominant root-cause findings.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the root-cause findings.
        """
        snapshot, evidence, _, _ = self._parse_request(body)
        findings = self.intelligence.root_cause_assessment(snapshot, evidence)
        return PlatformAPIResponse.ok(
            "/fleet/root-causes",
            {"root_causes": [f.to_dict() for f in findings],
             "count": len(findings),
             "evidence_supplied": bool(evidence)})

    def decisions(self, body: dict[str, Any] | None) -> PlatformAPIResponse:
        """Return the executive decision portfolio.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the decision portfolio.
        """
        snapshot, _, budget, _ = self._parse_request(body)
        portfolio = self.intelligence.decision_assessment(snapshot, budget)
        return PlatformAPIResponse.ok("/fleet/decisions", portfolio.to_dict())

    def scenarios(self, body: dict[str, Any] | None) -> PlatformAPIResponse:
        """Return scenario recommendations.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the scenario plan summary and ranking.
        """
        snapshot, evidence, budget, _ = self._parse_request(body)
        effective = (budget if budget is not None
                     else float(snapshot.fleet_expected_cost))
        plan = self.scenario_agent.plan(snapshot, effective,
                                        evidence_items=evidence)
        data = {
            "ranking": plan.ranking.to_dict(),
            "summary": plan.summary.to_dict(),
            "budget_scenarios": [r.to_dict() for r in plan.budget_scenarios],
            "delay_scenarios": [r.to_dict() for r in plan.delay_scenarios],
            "load_scenarios": [r.to_dict() for r in plan.load_scenarios],
            "growth_scenarios": [r.to_dict() for r in plan.growth_scenarios],
            "currency": plan.currency,
        }
        return PlatformAPIResponse.ok("/fleet/scenarios", data)

    def monte_carlo_stats(
        self, body: dict[str, Any] | None
    ) -> PlatformAPIResponse:
        """Return Monte Carlo fleet risk statistics.

        Derives an :class:`AssetSimulatorConfig` per asset from its health
        trajectory (initial health and an estimated degradation rate) and runs
        the Monte Carlo portfolio distribution.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the portfolio risk distribution.
        """
        snapshot, _, _, _ = self._parse_request(body)
        configs = self._simulator_configs(body)
        dist = self.monte_carlo.run_portfolio_distribution(configs)
        return PlatformAPIResponse.ok("/fleet/monte-carlo", dist.to_dict())

    def executive_report(
        self, body: dict[str, Any] | None
    ) -> PlatformAPIResponse:
        """Return the full executive intelligence report.

        Args:
            body: The fleet request body.

        Returns:
            A success response with the complete report.
        """
        snapshot, evidence, budget, scenarios = self._parse_request(body)
        report = self.intelligence.generate_report(
            snapshot, evidence_items=evidence, budget=budget,
            include_scenarios=scenarios)
        return PlatformAPIResponse.ok("/fleet/executive-report",
                                      report.to_dict())

    # ------------------------------------------------------------------
    # Validation & parsing layer
    # ------------------------------------------------------------------

    def _parse_request(
        self, body: dict[str, Any] | None
    ) -> tuple[FleetSnapshot, tuple[AssetEvidence, ...] | None, float | None, bool | None]:
        """Validate a fleet request body and build the snapshot.

        Args:
            body: The request body.

        Returns:
            Tuple ``(snapshot, evidence, budget, include_scenarios)``.

        Raises:
            PlatformAPIError: With status 400 on any validation failure.
        """
        if body is None or not isinstance(body, dict):
            raise PlatformAPIError(400, "Request body must be a JSON object")
        assets_raw = body.get("assets")
        if not isinstance(assets_raw, list) or not assets_raw:
            raise PlatformAPIError(400, "'assets' must be a non-empty list")
        if len(assets_raw) > self.config.max_assets:
            raise PlatformAPIError(
                400, f"too many assets (max {self.config.max_assets})")

        asset_inputs = []
        seen_ids = set()
        for i, a in enumerate(assets_raw):
            if not isinstance(a, dict):
                raise PlatformAPIError(400, f"asset[{i}] must be an object")
            aid = a.get("asset_id")
            if not isinstance(aid, str) or not aid.strip():
                raise PlatformAPIError(
                    400, f"asset[{i}] requires a non-empty 'asset_id'")
            if aid in seen_ids:
                raise PlatformAPIError(400, f"duplicate asset_id '{aid}'")
            seen_ids.add(aid)
            traj = a.get("health_trajectory")
            if not isinstance(traj, (list, tuple)) or len(traj) < _MIN_TRAJECTORY:
                raise PlatformAPIError(
                    400, f"asset '{aid}' requires a 'health_trajectory' of at "
                    f"least {_MIN_TRAJECTORY} numeric points")
            try:
                arr = np.asarray(traj, dtype=float)
            except (ValueError, TypeError):
                raise PlatformAPIError(
                    400, f"asset '{aid}' health_trajectory must be numeric")
            if not np.all(np.isfinite(arr)):
                raise PlatformAPIError(
                    400, f"asset '{aid}' health_trajectory must be finite")
            asset_inputs.append(AssetInput(
                asset_id=aid,
                asset_type=str(a.get("asset_type", "asset")),
                location=str(a.get("location", "unknown")),
                health_trajectory=arr))

        try:
            snapshot = self.fleet_engine.build_fleet_snapshot(asset_inputs)
        except ValueError as exc:
            raise PlatformAPIError(400, f"invalid fleet input: {exc}")

        evidence = self._parse_evidence(body.get("evidence"))
        budget = self._parse_budget(body.get("budget"))
        scenarios = body.get("include_scenarios")
        if scenarios is not None and not isinstance(scenarios, bool):
            raise PlatformAPIError(400, "'include_scenarios' must be a boolean")
        return snapshot, evidence, budget, scenarios

    @staticmethod
    def _parse_evidence(
        raw: Any,
    ) -> tuple[AssetEvidence, ...] | None:
        """Parse optional per-asset evidence.

        Args:
            raw: The raw evidence list, or ``None``.

        Returns:
            A tuple of :class:`AssetEvidence`, or ``None`` when absent.

        Raises:
            PlatformAPIError: With status 400 on malformed evidence.
        """
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise PlatformAPIError(400, "'evidence' must be a list")
        items = []
        for i, e in enumerate(raw):
            if not isinstance(e, dict):
                raise PlatformAPIError(400, f"evidence[{i}] must be an object")
            aid = e.get("asset_id")
            if not isinstance(aid, str) or not aid.strip():
                raise PlatformAPIError(
                    400, f"evidence[{i}] requires a non-empty 'asset_id'")
            kwargs = {}
            for key in _EVIDENCE_KEYS:
                if key in e:
                    val = e[key]
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        raise PlatformAPIError(
                            400, f"evidence[{i}].{key} must be a number")
                    kwargs[key] = float(val)
            try:
                items.append(AssetEvidence(asset_id=aid, **kwargs))
            except ValueError as exc:
                raise PlatformAPIError(400, f"evidence[{i}]: {exc}")
        return tuple(items)

    @staticmethod
    def _parse_budget(raw: Any) -> float | None:
        """Parse an optional budget.

        Args:
            raw: The raw budget value, or ``None``.

        Returns:
            The budget, or ``None`` when absent.

        Raises:
            PlatformAPIError: With status 400 on an invalid budget.
        """
        if raw is None:
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PlatformAPIError(400, "'budget' must be a number")
        if raw < 0 or not math.isfinite(raw):
            raise PlatformAPIError(400, "'budget' must be finite and >= 0")
        return float(raw)

    def _simulator_configs(
        self, body: dict[str, Any] | None
    ) -> list[AssetSimulatorConfig]:
        """Build per-asset simulator configs from the request trajectories.

        The degradation rate is estimated from each trajectory's net decline
        per step (floored at a small positive value so every asset degrades).

        Args:
            body: The request body (already validated).

        Returns:
            A list of :class:`AssetSimulatorConfig`, one per asset.
        """
        cfg = self.config
        assets_raw = body["assets"]  # validated by _parse_request
        configs = []
        for idx, a in enumerate(assets_raw):
            arr = np.asarray(a["health_trajectory"], dtype=float)
            initial = float(np.clip(arr[0], 0.0, 100.0))
            span = max(len(arr) - 1, 1)
            slope = (arr[0] - arr[-1]) / span
            rate = float(max(slope, 0.01))
            configs.append(AssetSimulatorConfig(
                horizon=cfg.mc_horizon,
                initial_health=initial,
                degradation_rate=rate,
                failure_threshold=cfg.mc_failure_threshold,
                random_seed=cfg.mc_seed + idx))
        return configs

    # ------------------------------------------------------------------
    # Tracker
    # ------------------------------------------------------------------

    def _log(self) -> None:
        """Log the request count to the tracker (failure-safe)."""
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics({"api_requests": float(self._n_requests)},
                                     step=self._n_requests)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI application factory (optional shell)
# ---------------------------------------------------------------------------


def create_app(server: PlatformAPIServer | None = None):  # type: ignore[no-untyped-def]
    """Build a FastAPI application wrapping a :class:`PlatformAPIServer`.

    The web framework is imported lazily so that importing this module — and
    using the server core — never requires FastAPI.  Routes delegate to
    :meth:`PlatformAPIServer.dispatch` through dependency injection.

    Args:
        server: The server to wrap (a default one is created if None).

    Returns:
        A configured FastAPI application.

    Raises:
        RuntimeError: When FastAPI is not installed.
    """
    try:
        from fastapi import Body, Depends, FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised only sans FastAPI
        raise RuntimeError(
            "FastAPI is not installed. Install it with 'pip install fastapi "
            "uvicorn' to serve the API, or use PlatformAPIServer.dispatch "
            "directly.") from exc

    srv = server or PlatformAPIServer()
    app = FastAPI(title=srv.config.title, version=srv.config.version)

    def get_server() -> PlatformAPIServer:
        """Dependency provider yielding the server instance."""
        return srv

    @app.get("/health")
    def _health(s: PlatformAPIServer = Depends(get_server)):  # noqa: ANN202
        code, body = s.dispatch("GET", "/health")
        return JSONResponse(status_code=code, content=body)

    def _make_post(path: str):  # type: ignore[no-untyped-def]
        def _handler(payload: dict = Body(default=None),  # noqa: ANN202
                     s: PlatformAPIServer = Depends(get_server)):
            code, body = s.dispatch("POST", path, payload)
            return JSONResponse(status_code=code, content=body)
        return _handler

    for path in ("/fleet/summary", "/fleet/top-risks", "/fleet/root-causes",
                 "/fleet/decisions", "/fleet/scenarios", "/fleet/monte-carlo",
                 "/fleet/executive-report"):
        app.post(path)(_make_post(path))

    return app


def get_app():  # type: ignore[no-untyped-def]
    """Return a FastAPI app if FastAPI is available, else ``None``.

    Returns:
        A FastAPI application, or ``None`` when FastAPI is not installed.
    """
    try:
        return create_app()
    except RuntimeError:
        return None


#: Module-level ASGI app for ``uvicorn src.api.platform_api:app`` (None if
#: FastAPI is absent).
app = get_app()


# ---------------------------------------------------------------------------
# Demo CLI
# ---------------------------------------------------------------------------


def _demo_request() -> dict[str, Any]:
    """Build a deterministic demo request body.

    Returns:
        A fleet request dictionary.
    """
    rng = np.random.default_rng(85)
    specs = [("WTG-001", 0.4, "vibration"), ("WTG-014", 0.9, "lubrication"),
             ("WTG-027", 1.6, "electrical"), ("WTG-042", 2.6, "vibration"),
             ("WTG-051", 1.1, "temperature"), ("WTG-068", 0.6, "load")]
    assets, evidence = [], []
    for aid, rate, cause in specs:
        traj = np.clip(96 - rate * np.arange(45) + rng.normal(0, 0.3, 45),
                       0, 100)
        assets.append({"asset_id": aid, "asset_type": "wind_turbine",
                       "location": "North Sea",
                       "health_trajectory": [round(float(x), 2) for x in traj]})
        evidence.append({"asset_id": aid, cause: 0.8})
    return {"assets": assets, "evidence": evidence, "budget": 15000,
            "include_scenarios": True}


def _demo() -> int:
    """Run a demo exercising every endpoint through ``dispatch``.

    Returns:
        Exit code 0.
    """
    server = PlatformAPIServer()
    body = _demo_request()

    print("=== GET /health ===")
    print(server.dispatch("GET", "/health"))
    print()
    for path in ("/fleet/summary", "/fleet/top-risks", "/fleet/root-causes",
                 "/fleet/decisions", "/fleet/scenarios", "/fleet/monte-carlo",
                 "/fleet/executive-report"):
        code, resp = server.dispatch("POST", path, body)
        print(f"=== POST {path} -> {code} ===")
        data = resp.get("data") or {}
        if path == "/fleet/summary":
            print(f"  health={data['current_health']:.0f} "
                  f"risk={data['current_risk']:.2f} "
                  f"confidence={data['confidence']:.2f}")
        elif path == "/fleet/top-risks":
            for r in data["top_risks"][:3]:
                print(f"  {r['asset_id']:10s} priority={r['priority_score']:.2f} "
                      f"tier={r['risk_tier']}")
        elif path == "/fleet/root-causes":
            for f in data["root_causes"][:3]:
                print(f"  {f['subject']:12s} conf={f['confidence']:.2f}")
        elif path == "/fleet/decisions":
            print(f"  strategy={data['strategy']} "
                  f"selected={len(data['selected_asset_ids'])} "
                  f"roi={data['total_roi']:.2f}")
        elif path == "/fleet/scenarios":
            print(f"  recommended={data['summary']['recommended_scenario']}")
        elif path == "/fleet/monte-carlo":
            print(f"  portfolio_risk={data['portfolio_risk']:.3f} "
                  f"expected_failures={data['expected_fleet_failures']:.2f}")
        elif path == "/fleet/executive-report":
            print(f"  summary: {data['executive_summary'][:80]}...")
        print()
    print("=== error handling ===")
    print("  unknown path:", server.dispatch("GET", "/nope")[0])
    print("  wrong method:", server.dispatch("GET", "/fleet/summary")[0])
    print("  empty body:", server.dispatch("POST", "/fleet/summary", {})[0])
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
        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Platform API")
    parser.add_argument("--demo", action="store_true",
                        help="Run an API demo through dispatch.")
    parser.add_argument("--routes", action="store_true",
                        help="List the registered routes.")
    args = parser.parse_args(argv)

    if args.routes:
        srv = PlatformAPIServer()
        for method, path in sorted(srv._routes):
            print(f"{method:5s} {path}")
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())