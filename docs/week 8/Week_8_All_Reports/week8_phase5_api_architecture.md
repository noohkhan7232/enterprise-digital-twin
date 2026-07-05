# Week 8 — Phase 5: Executive Dashboard & API Layer

## Architecture Documentation

**Component:** `src/api/platform_api.py`
**Role:** Production-style REST API and executive dashboard backend over the whole platform
**Inputs (per request):** raw fleet telemetry — per-asset health trajectories, plus optional subsystem evidence, budget, and scenario flag
**Outputs:** uniform JSON response envelopes wrapping fleet overviews, top risks, root causes, decisions, scenarios, Monte Carlo metrics, and full executive reports
**Status:** Implemented and validated — 158 tests passing with zero skips; all Week 5–8 Phase 4 suites unchanged and coexisting (17/17 modules import cleanly).

---

## 1. Architecture

The defining decision is a **transport-agnostic core with an optional FastAPI shell**.

All request processing lives in `PlatformAPIServer`. Its `dispatch(method, path, body)` method performs the exact routing, validation, composition, and serialisation that an HTTP layer would, returning a `(status_code, body)` pair. `create_app()` builds a FastAPI application on top of the server via a lazy, guarded import and wires every route to `dispatch` through dependency injection.

This separation has three concrete benefits. First, the entire API — every endpoint, every status code, every error path — is testable with no web dependencies, by driving `dispatch` directly; the 158-test suite runs without FastAPI installed. Second, it honours the platform's pure-Python, deterministic standard: the core has no framework coupling. Third, it keeps the HTTP layer trivially thin — each FastAPI route is a one-line delegation to `dispatch`, so there is no logic that the tests cannot reach.

```
   HTTP client / dashboard
        │  JSON
        ▼
   FastAPI app  (create_app — optional, lazy import)
        │  delegates every route to ↓
        ▼
   PlatformAPIServer.dispatch(method, path, body)  ── routing + status codes
        ├── validation layer  (_parse_request)      ── 400 on bad input
        ├── handler           (per endpoint)         ── composes the platform
        └── serialization     (_jsonsafe + to_dict)  ── uniform envelope
        ▼
   ExecutiveIntelligenceAgent ── Fleet Twin · Executive · Copilot ·
                                  Root-Cause · Scenario
   MonteCarloEngine          ── portfolio risk distribution
        ▼
   PlatformAPIResponse  (status · endpoint · data · error)
```

The server composes the platform through the Phase-4 Executive Intelligence Agent (which itself orchestrates the Fleet, Executive, Copilot, Root-Cause, and Scenario modules) plus a direct Monte Carlo engine for the risk-metrics endpoint. No business logic is duplicated in the API layer — it validates input, calls the composed modules, and serialises their frozen outputs.

---

## 2. API Contracts

Every response shares one envelope:

```json
{ "status": "ok" | "error", "endpoint": "/fleet/...", "data": { ... } | null, "error": { ... } | null }
```

The request body for every `POST /fleet/*` endpoint is the same fleet-telemetry object:

```json
{
  "assets": [
    { "asset_id": "WTG-001", "asset_type": "wind_turbine",
      "location": "North Sea", "health_trajectory": [96.0, 95.5, ...] }
  ],
  "evidence": [ { "asset_id": "WTG-001", "vibration": 0.8, "lubrication": 0.2 } ],
  "budget": 15000,
  "include_scenarios": true
}
```

`assets` is required (each needs an `asset_id` and a `health_trajectory` of at least five numeric points); `evidence`, `budget`, and `include_scenarios` are optional. `asset_type` and `location` default when omitted.

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/health` | GET | `{"status": "healthy", "version": "..."}` |
| `/fleet/summary` | POST | Executive report summary (overview, health, risk, RUL, summary, confidence) |
| `/fleet/top-risks` | POST | Top assets by executive priority score |
| `/fleet/root-causes` | POST | Dominant root-cause findings |
| `/fleet/decisions` | POST | Executive decision portfolio |
| `/fleet/scenarios` | POST | Scenario plan (budget / delay / load / growth) + ranking + summary |
| `/fleet/monte-carlo` | POST | Monte Carlo portfolio risk distribution |
| `/fleet/executive-report` | POST | Full executive intelligence report |

---

## 3. Endpoint Design

Clients submit raw telemetry rather than a pre-built snapshot, which is the realistic integration: the dashboard sends what it has (health histories), and the server runs the Fleet Digital Twin to produce the `FleetSnapshot` before any analysis. Each endpoint then composes exactly the modules it needs — `/fleet/decisions` runs the Executive Decision Engine, `/fleet/scenarios` runs the Scenario Planning Agent, `/fleet/executive-report` runs the full Executive Intelligence orchestration, and so on. The `/fleet/monte-carlo` endpoint derives an `AssetSimulatorConfig` per asset from its trajectory (initial health plus an estimated degradation rate, floored so every asset degrades) and runs the real `run_portfolio_distribution` — genuine Monte Carlo, not a summary statistic.

---

## 4. Validation Strategy

`_parse_request` is the single validation gate. It rejects, with HTTP 400 and a structured `PlatformAPIError`, any of: a non-object body; a missing or empty `assets` list; more assets than `max_assets`; an asset without a non-empty `asset_id`; a duplicate `asset_id`; a `health_trajectory` shorter than five points, non-numeric, or non-finite; a negative, non-numeric, or non-finite `budget`; malformed `evidence` (non-list, missing id, non-numeric or out-of-range indicator); and a non-boolean `include_scenarios`. Anything the Fleet Digital Twin itself rejects is caught and re-raised as a 400. This means a malformed request can never reach the composed engines.

---

## 5. Serialization Strategy

Every output is a frozen dataclass with a `to_dict`, and `_jsonsafe` recursively converts the result into strictly JSON-safe values: non-finite floats become `null`, NumPy scalars become native Python numbers, and tuples become lists. The uniform `PlatformAPIResponse` envelope is the only thing the HTTP layer ever serialises, so clients get a predictable shape on both success and failure. The test suite round-trips every endpoint's response through `json.dumps`/`json.loads`.

---

## 6. Error Handling

`dispatch` maps failures to standard HTTP semantics: 404 for an unknown path, 405 for a known path called with the wrong method, 400 for any validation failure (carrying the offending detail), and 500 for an unexpected internal error (caught, logged, and returned without leaking a traceback to the client). Successful requests return 200. The error envelope always carries a `status_code`, `message`, and optional `detail`. Request counting only increments on success, so error volume is distinguishable from throughput.

---

## 7. Integration Flow

A single `/fleet/executive-report` call exercises the entire platform: the telemetry builds a snapshot (Fleet Digital Twin); the snapshot is scored for priority and explained (Executive Intelligence + Copilot); evidence is attributed to causes (Root-Cause); a budget is optimised (Executive Decision Engine); futures are projected (Scenario Planning); and everything is synthesised into one report. The `/fleet/monte-carlo` endpoint adds probabilistic fleet risk via the Monte Carlo engine. Because all of this runs through the composed modules, the API inherits their determinism end to end — identical requests yield byte-identical responses, including the Monte Carlo metrics (fixed seed).

---

## 8. Testing Summary

158 tests pass with zero skips, covering config validation; the response envelope and error objects; all eight endpoints (payloads, value ranges, field presence); routing (404 / 405 / path normalisation / method case-insensitivity); the full validation layer (every rejection path); serialization (JSON round-trips, no residual tuples, non-finite handling); determinism (per-endpoint and across server instances, including Monte Carlo); the failure-safe tracker; the FastAPI factory's graceful degradation when the framework is absent; edge cases (single-asset, large, and minimal fleets; no evidence / scenario / budget); and integration (shared composed agents, end-to-end traversal of every endpoint, and consistency between the summary and full-report endpoints). The seven Week 7 / Week 8 prior suites continue to pass unchanged, and all seventeen platform modules import and coexist cleanly.

---

## 9. Future Dashboard UI Integration

The API is dashboard-ready. A front-end would call `/health` for a status indicator, `/fleet/summary` for the landing tiles (health, risk, RUL, confidence), `/fleet/top-risks` for a ranked asset table, `/fleet/root-causes` for a cause breakdown, `/fleet/decisions` for the recommended maintenance plan, `/fleet/scenarios` for an interactive what-if panel, `/fleet/monte-carlo` for a risk-distribution chart, and `/fleet/executive-report` for a printable briefing. Because every response is a flat JSON envelope with stable keys, the UI can bind directly to the fields. To deploy with HTTP, install `fastapi` and `uvicorn` and run `uvicorn src.api.platform_api:app`; the module-level `app` is constructed automatically when FastAPI is present and is `None` otherwise, so importing the module is always safe. Natural next steps include authentication middleware, response caching keyed on the request hash (safe given determinism), and server-sent events for streaming long scenario sweeps — all of which layer on top of the existing `dispatch` core without changing it.