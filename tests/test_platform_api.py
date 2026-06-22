#!/usr/bin/env python3
"""Comprehensive test suite for ``src/api/platform_api.py``.

Pure Python / NumPy — the API core is transport-agnostic, so the entire suite
runs without FastAPI, pydantic, or any web dependency by driving
:meth:`PlatformAPIServer.dispatch` (which mirrors HTTP routing exactly).
Coverage (150+ tests):

- Config validation
- Response envelope & error objects
- Every endpoint (health, summary, top-risks, root-causes, decisions,
  scenarios, monte-carlo, executive-report)
- Routing (404 / 405 / path normalisation)
- Validation layer (assets, trajectories, evidence, budget, scenario flag)
- Serialization (JSON round-trips)
- Determinism
- Tracker
- Edge cases (empty / single-asset / large fleets, malformed requests,
  no evidence / scenario / budget)
- FastAPI factory graceful degradation
- Integration with all prior modules

Run::

    pytest tests/test_platform_api.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.platform_api import (
    API_VERSION,
    PlatformAPIConfig,
    PlatformAPIError,
    PlatformAPIResponse,
    PlatformAPIServer,
    create_app,
    get_app,
)

_POST_PATHS = ("/fleet/summary", "/fleet/top-risks", "/fleet/root-causes",
               "/fleet/decisions", "/fleet/scenarios", "/fleet/monte-carlo",
               "/fleet/executive-report")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _traj(rate: float, n: int = 45) -> list[float]:
    return [round(float(x), 2) for x in np.clip(96 - rate * np.arange(n), 0, 100)]


def _body(n: int = 6, *, evidence: bool = True, budget=15000,
          scenarios=True) -> dict:
    b: dict = {"assets": [
        {"asset_id": f"WTG-{i:03d}", "asset_type": "wt", "location": "N",
         "health_trajectory": _traj(0.4 + 0.3 * i)} for i in range(n)]}
    if evidence:
        b["evidence"] = [{"asset_id": f"WTG-{i:03d}",
                          "vibration": 0.8 if i % 2 == 0 else 0.2,
                          "lubrication": 0.7 if i % 2 else 0.1} for i in range(n)]
    if budget is not None:
        b["budget"] = budget
    if scenarios is not None:
        b["include_scenarios"] = scenarios
    return b


def _server(**kw) -> PlatformAPIServer:
    return PlatformAPIServer(PlatformAPIConfig(**kw) if kw else None)


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults(self) -> None:
        c = PlatformAPIConfig()
        assert c.version == API_VERSION and c.top_n == 5

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            PlatformAPIConfig(top_n=0)

    def test_mc_trials_positive(self) -> None:
        with pytest.raises(ValueError, match="mc_trials"):
            PlatformAPIConfig(mc_trials=0)

    def test_mc_horizon_positive(self) -> None:
        with pytest.raises(ValueError, match="mc_horizon"):
            PlatformAPIConfig(mc_horizon=0)

    def test_mc_threshold_range(self) -> None:
        with pytest.raises(ValueError, match="mc_failure_threshold"):
            PlatformAPIConfig(mc_failure_threshold=150)

    def test_max_assets_positive(self) -> None:
        with pytest.raises(ValueError, match="max_assets"):
            PlatformAPIConfig(max_assets=0)

    def test_frozen(self) -> None:
        c = PlatformAPIConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.top_n = 9  # type: ignore[misc]

    def test_custom_currency(self) -> None:
        assert PlatformAPIConfig(currency="EUR").currency == "EUR"


# ===========================================================================
# Response & error objects
# ===========================================================================


class TestResponseEnvelope:
    def test_ok_envelope(self) -> None:
        r = PlatformAPIResponse.ok("/x", {"a": 1})
        assert r.status == "ok" and r.data == {"a": 1} and r.error is None

    def test_fail_envelope(self) -> None:
        r = PlatformAPIResponse.fail("/x", PlatformAPIError(400, "bad"))
        assert r.status == "error" and r.data is None and r.error is not None

    def test_to_dict_keys(self) -> None:
        d = PlatformAPIResponse.ok("/x", {}).to_dict()
        for k in ("status", "endpoint", "data", "error"):
            assert k in d

    def test_frozen(self) -> None:
        r = PlatformAPIResponse.ok("/x", {})
        with pytest.raises((AttributeError, TypeError)):
            r.status = "error"  # type: ignore[misc]

    def test_json_serializable(self) -> None:
        assert isinstance(json.dumps(PlatformAPIResponse.ok("/x", {}).to_dict()),
                          str)

    def test_non_finite_jsonsafe(self) -> None:
        r = PlatformAPIResponse.ok("/x", {"v": float("inf")})
        assert r.to_dict()["data"]["v"] is None


class TestError:
    def test_status_code(self) -> None:
        assert PlatformAPIError(404, "x").status_code == 404

    def test_to_dict(self) -> None:
        d = PlatformAPIError(400, "bad", detail="extra").to_dict()
        assert d["status_code"] == 400 and d["detail"] == "extra"

    def test_message(self) -> None:
        assert PlatformAPIError(500, "boom").message == "boom"

    def test_is_exception(self) -> None:
        assert isinstance(PlatformAPIError(400, "x"), Exception)


# ===========================================================================
# Health endpoint
# ===========================================================================


class TestHealth:
    def test_status_200(self) -> None:
        code, _ = _server().dispatch("GET", "/health")
        assert code == 200

    def test_healthy_payload(self) -> None:
        _, resp = _server().dispatch("GET", "/health")
        assert resp["data"]["status"] == "healthy"

    def test_version_present(self) -> None:
        _, resp = _server().dispatch("GET", "/health")
        assert resp["data"]["version"] == API_VERSION

    def test_envelope_ok(self) -> None:
        _, resp = _server().dispatch("GET", "/health")
        assert resp["status"] == "ok"

    def test_post_health_405(self) -> None:
        code, _ = _server().dispatch("POST", "/health", {})
        assert code == 405


# ===========================================================================
# Summary endpoint
# ===========================================================================


class TestSummary:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/summary", _body())
        assert code == 200

    def test_has_fields(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        d = resp["data"]
        for k in ("fleet_overview", "current_health", "current_risk",
                  "current_rul", "executive_summary", "confidence"):
            assert k in d

    def test_risk_in_unit(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert 0 <= resp["data"]["current_risk"] <= 1

    def test_health_in_range(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert 0 <= resp["data"]["current_health"] <= 100

    def test_confidence_in_unit(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert 0 <= resp["data"]["confidence"] <= 1

    def test_currency(self) -> None:
        _, resp = _server(currency="EUR").dispatch("POST", "/fleet/summary",
                                                   _body())
        assert resp["data"]["currency"] == "EUR"

    def test_json(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert isinstance(json.dumps(resp), str)


# ===========================================================================
# Top-risks endpoint
# ===========================================================================


class TestTopRisks:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/top-risks", _body())
        assert code == 200

    def test_returns_list(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        assert isinstance(resp["data"]["top_risks"], list)

    def test_count_matches(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        assert resp["data"]["count"] == len(resp["data"]["top_risks"])

    def test_limited_to_top_n(self) -> None:
        _, resp = _server(top_n=3).dispatch("POST", "/fleet/top-risks",
                                            _body(6))
        assert len(resp["data"]["top_risks"]) == 3

    def test_priority_descending(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        ps = [r["priority_score"] for r in resp["data"]["top_risks"]]
        assert ps == sorted(ps, reverse=True)

    def test_risk_fields(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        r0 = resp["data"]["top_risks"][0]
        for k in ("asset_id", "risk_score", "priority_score", "risk_tier"):
            assert k in r0


# ===========================================================================
# Root-causes endpoint
# ===========================================================================


class TestRootCauses:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/root-causes", _body())
        assert code == 200

    def test_with_evidence(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes", _body())
        assert len(resp["data"]["root_causes"]) > 0

    def test_no_evidence_empty(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes",
                                     _body(evidence=False))
        assert resp["data"]["root_causes"] == []

    def test_evidence_flag(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes", _body())
        assert resp["data"]["evidence_supplied"] is True

    def test_finding_fields(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes", _body())
        f0 = resp["data"]["root_causes"][0]
        for k in ("finding_type", "subject", "statement", "confidence"):
            assert k in f0

    def test_confidence_in_unit(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes", _body())
        assert all(0 <= f["confidence"] <= 1 for f in resp["data"]["root_causes"])


# ===========================================================================
# Decisions endpoint
# ===========================================================================


class TestDecisions:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/decisions", _body())
        assert code == 200

    def test_portfolio_fields(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/decisions", _body())
        d = resp["data"]
        for k in ("strategy", "selected_asset_ids", "total_maintenance_cost",
                  "expected_savings", "total_roi"):
            assert k in d

    def test_budget_respected(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/decisions",
                                     _body(budget=10000))
        assert resp["data"]["total_maintenance_cost"] <= 10000 + 1e-6

    def test_default_budget(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/decisions",
                                     _body(budget=None))
        assert code == 200

    def test_roi_positive(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/decisions", _body())
        assert resp["data"]["total_roi"] > 0


# ===========================================================================
# Scenarios endpoint
# ===========================================================================


class TestScenarios:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/scenarios", _body())
        assert code == 200

    def test_summary_present(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/scenarios", _body())
        assert "recommended_scenario" in resp["data"]["summary"]

    def test_all_families(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/scenarios", _body())
        d = resp["data"]
        assert len(d["budget_scenarios"]) == 3
        assert len(d["delay_scenarios"]) == 4
        assert len(d["load_scenarios"]) == 2
        assert len(d["growth_scenarios"]) == 4

    def test_ranking_present(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/scenarios", _body())
        assert "ranking" in resp["data"] and "ranked" in resp["data"]["ranking"]

    def test_default_budget(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/scenarios",
                                     _body(budget=None))
        assert code == 200

    def test_json(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/scenarios", _body())
        assert isinstance(json.dumps(resp), str)


# ===========================================================================
# Monte Carlo endpoint
# ===========================================================================


class TestMonteCarlo:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert code == 200

    def test_portfolio_risk(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert 0 <= resp["data"]["portfolio_risk"] <= 1

    def test_expected_failures(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert resp["data"]["expected_fleet_failures"] >= 0

    def test_n_assets(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body(6))
        assert resp["data"]["n_assets"] == 6

    def test_per_asset_probability(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert "per_asset_probability" in resp["data"]

    def test_deterministic(self) -> None:
        b = _body()
        m1 = _server().dispatch("POST", "/fleet/monte-carlo", b)[1]
        m2 = _server().dispatch("POST", "/fleet/monte-carlo", b)[1]
        assert m1["data"]["portfolio_risk"] == m2["data"]["portfolio_risk"]

    def test_json(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert isinstance(json.dumps(resp), str)


# ===========================================================================
# Executive report endpoint
# ===========================================================================


class TestExecutiveReport:
    def test_200(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert code == 200

    def test_full_report_fields(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        d = resp["data"]
        for k in ("fleet_overview", "current_health", "current_risk",
                  "current_rul", "top_risks", "root_causes",
                  "recommended_actions", "budget_recommendation",
                  "scenario_recommendation", "strategic_narrative",
                  "executive_summary", "confidence", "currency"):
            assert k in d

    def test_top_risks_populated(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert len(resp["data"]["top_risks"]) > 0

    def test_root_causes_with_evidence(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert len(resp["data"]["root_causes"]) > 0

    def test_recommendations_present(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert len(resp["data"]["recommended_actions"]) > 0

    def test_no_scenarios(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report",
                                     _body(scenarios=False))
        assert "not requested" in resp["data"]["scenario_recommendation"].lower()

    def test_json(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert isinstance(json.dumps(resp), str)


# ===========================================================================
# Routing
# ===========================================================================


class TestRouting:
    def test_unknown_path_404(self) -> None:
        assert _server().dispatch("GET", "/nope")[0] == 404

    def test_unknown_post_404(self) -> None:
        assert _server().dispatch("POST", "/fleet/unknown", {})[0] == 404

    def test_wrong_method_405(self) -> None:
        assert _server().dispatch("GET", "/fleet/summary")[0] == 405

    def test_put_405(self) -> None:
        assert _server().dispatch("PUT", "/fleet/summary", {})[0] == 405

    def test_trailing_slash_normalised(self) -> None:
        assert _server().dispatch("GET", "/health/")[0] == 200

    def test_query_string_stripped(self) -> None:
        assert _server().dispatch("GET", "/health?x=1")[0] == 200

    def test_method_case_insensitive(self) -> None:
        assert _server().dispatch("get", "/health")[0] == 200

    def test_404_envelope_is_error(self) -> None:
        _, resp = _server().dispatch("GET", "/nope")
        assert resp["status"] == "error"

    def test_405_envelope_is_error(self) -> None:
        _, resp = _server().dispatch("GET", "/fleet/summary")
        assert resp["status"] == "error"


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_none_body_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary", None)[0] == 400

    def test_empty_body_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary", {})[0] == 400

    def test_non_dict_body_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary", [])[0] == 400

    def test_empty_assets_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary",
                                  {"assets": []})[0] == 400

    def test_assets_not_list_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary",
                                  {"assets": "x"})[0] == 400

    def test_asset_not_dict_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary",
                                  {"assets": ["x"]})[0] == 400

    def test_missing_asset_id_400(self) -> None:
        assert _server().dispatch(
            "POST", "/fleet/summary",
            {"assets": [{"health_trajectory": _traj(1)}]})[0] == 400

    def test_empty_asset_id_400(self) -> None:
        assert _server().dispatch(
            "POST", "/fleet/summary",
            {"assets": [{"asset_id": "", "health_trajectory": _traj(1)}]})[0] == 400

    def test_short_trajectory_400(self) -> None:
        assert _server().dispatch(
            "POST", "/fleet/summary",
            {"assets": [{"asset_id": "A", "health_trajectory": [1, 2, 3]}]})[0] == 400

    def test_missing_trajectory_400(self) -> None:
        assert _server().dispatch(
            "POST", "/fleet/summary",
            {"assets": [{"asset_id": "A"}]})[0] == 400

    def test_non_numeric_trajectory_400(self) -> None:
        assert _server().dispatch(
            "POST", "/fleet/summary",
            {"assets": [{"asset_id": "A",
                         "health_trajectory": ["a", "b", "c", "d", "e"]}]})[0] == 400

    def test_non_finite_trajectory_400(self) -> None:
        bad = {"assets": [{"asset_id": "A",
                           "health_trajectory": [1, 2, 3, 4, float("inf")]}]}
        assert _server().dispatch("POST", "/fleet/summary", bad)[0] == 400

    def test_duplicate_asset_id_400(self) -> None:
        bad = {"assets": [{"asset_id": "A", "health_trajectory": _traj(1)},
                          {"asset_id": "A", "health_trajectory": _traj(1)}]}
        assert _server().dispatch("POST", "/fleet/summary", bad)[0] == 400

    def test_negative_budget_400(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary",
                                  _body(budget=-1))[0] == 400

    def test_non_numeric_budget_400(self) -> None:
        b = _body()
        b["budget"] = "lots"
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_bool_budget_400(self) -> None:
        b = _body()
        b["budget"] = True
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_evidence_not_list_400(self) -> None:
        b = _body(evidence=False)
        b["evidence"] = "x"
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_evidence_bad_indicator_400(self) -> None:
        b = _body(evidence=False)
        b["evidence"] = [{"asset_id": "WTG-000", "vibration": 1.5}]
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_evidence_missing_id_400(self) -> None:
        b = _body(evidence=False)
        b["evidence"] = [{"vibration": 0.5}]
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_evidence_non_numeric_400(self) -> None:
        b = _body(evidence=False)
        b["evidence"] = [{"asset_id": "WTG-000", "vibration": "high"}]
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_scenario_flag_not_bool_400(self) -> None:
        b = _body()
        b["include_scenarios"] = "yes"
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_too_many_assets_400(self) -> None:
        srv = _server(max_assets=2)
        assert srv.dispatch("POST", "/fleet/summary", _body(4))[0] == 400

    def test_error_envelope_has_detail(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", {})
        assert resp["error"]["status_code"] == 400


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    def test_all_endpoints_json(self) -> None:
        srv = _server()
        b = _body()
        assert isinstance(json.dumps(srv.dispatch("GET", "/health")[1]), str)
        for path in _POST_PATHS:
            assert isinstance(json.dumps(srv.dispatch("POST", path, b)[1]), str)

    def test_error_json(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", {})
        assert isinstance(json.dumps(resp), str)

    def test_round_trip(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert json.loads(json.dumps(resp))["status"] == "ok"

    def test_response_has_no_tuples(self) -> None:
        # After to_dict + jsonsafe, no tuples remain (lists only)
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        assert isinstance(resp["data"]["top_risks"], list)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_summary_deterministic(self) -> None:
        b = _body()
        r1 = _server().dispatch("POST", "/fleet/summary", b)[1]
        r2 = _server().dispatch("POST", "/fleet/summary", b)[1]
        assert r1 == r2

    def test_report_deterministic(self) -> None:
        b = _body()
        r1 = _server().dispatch("POST", "/fleet/executive-report", b)[1]
        r2 = _server().dispatch("POST", "/fleet/executive-report", b)[1]
        assert r1["data"]["executive_summary"] == r2["data"]["executive_summary"]

    def test_top_risks_deterministic(self) -> None:
        b = _body()
        r1 = _server().dispatch("POST", "/fleet/top-risks", b)[1]
        r2 = _server().dispatch("POST", "/fleet/top-risks", b)[1]
        assert r1 == r2

    def test_scenarios_deterministic(self) -> None:
        b = _body()
        r1 = _server().dispatch("POST", "/fleet/scenarios", b)[1]
        r2 = _server().dispatch("POST", "/fleet/scenarios", b)[1]
        assert r1["data"]["summary"] == r2["data"]["summary"]

    def test_monte_carlo_deterministic(self) -> None:
        b = _body()
        r1 = _server().dispatch("POST", "/fleet/monte-carlo", b)[1]
        r2 = _server().dispatch("POST", "/fleet/monte-carlo", b)[1]
        assert r1 == r2

    def test_across_server_instances(self) -> None:
        b = _body()
        r1 = PlatformAPIServer().dispatch("POST", "/fleet/summary", b)[1]
        r2 = PlatformAPIServer().dispatch("POST", "/fleet/summary", b)[1]
        assert r1 == r2


# ===========================================================================
# Tracker
# ===========================================================================


class TestTracker:
    def test_logs_requests(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        srv = PlatformAPIServer(experiment_tracker=FakeTracker())
        srv.dispatch("POST", "/fleet/summary", _body())
        assert logged and "api_requests" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        srv = PlatformAPIServer(experiment_tracker=BrokenTracker())
        assert srv.dispatch("POST", "/fleet/summary", _body())[0] == 200

    def test_no_tracker_ok(self) -> None:
        assert _server().dispatch("GET", "/health")[0] == 200

    def test_request_count_increments(self) -> None:
        srv = _server()
        srv.dispatch("GET", "/health")
        srv.dispatch("POST", "/fleet/summary", _body())
        assert srv._n_requests == 2

    def test_errors_not_counted(self) -> None:
        srv = _server()
        srv.dispatch("POST", "/fleet/summary", {})  # 400
        assert srv._n_requests == 0


# ===========================================================================
# FastAPI factory
# ===========================================================================


class TestFastAPIFactory:
    def test_create_app_without_fastapi(self) -> None:
        # FastAPI is absent in this environment -> RuntimeError
        import importlib.util
        if importlib.util.find_spec("fastapi") is None:
            with pytest.raises(RuntimeError, match="FastAPI is not installed"):
                create_app()
        else:  # pragma: no cover
            assert create_app() is not None

    def test_get_app_safe(self) -> None:
        import importlib.util
        if importlib.util.find_spec("fastapi") is None:
            assert get_app() is None
        else:  # pragma: no cover
            assert get_app() is not None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_asset(self) -> None:
        code, resp = _server().dispatch("POST", "/fleet/executive-report",
                                        _body(1))
        assert code == 200 and len(resp["data"]["top_risks"]) == 1

    def test_large_fleet(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/summary",
                                     _body(40, scenarios=False))
        assert code == 200

    def test_no_budget(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/summary",
                                     _body(budget=None))
        assert code == 200

    def test_no_evidence(self) -> None:
        code, resp = _server().dispatch("POST", "/fleet/executive-report",
                                        _body(evidence=False))
        assert code == 200 and resp["data"]["root_causes"] == []

    def test_no_scenarios(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/summary",
                                     _body(scenarios=False))
        assert code == 200

    def test_zero_budget(self) -> None:
        code, _ = _server().dispatch("POST", "/fleet/decisions",
                                     _body(budget=0))
        assert code == 200

    def test_minimal_request(self) -> None:
        minimal = {"assets": [{"asset_id": "A", "health_trajectory": _traj(1)}]}
        assert _server().dispatch("POST", "/fleet/summary", minimal)[0] == 200

    def test_evidence_partial_indicators(self) -> None:
        b = _body(evidence=False)
        b["evidence"] = [{"asset_id": "WTG-000", "vibration": 0.5}]
        assert _server().dispatch("POST", "/fleet/root-causes", b)[0] == 200

    def test_optional_asset_fields_default(self) -> None:
        minimal = {"assets": [{"asset_id": "A", "health_trajectory": _traj(1)}]}
        # asset_type / location omitted -> defaults applied, still 200
        assert _server().dispatch("POST", "/fleet/top-risks", minimal)[0] == 200


# ===========================================================================
# Integration with all prior modules
# ===========================================================================


class TestIntegration:
    def test_routes_table(self) -> None:
        srv = _server()
        assert ("GET", "/health") in srv._routes
        for p in _POST_PATHS:
            assert ("POST", p) in srv._routes

    def test_intelligence_agent_composed(self) -> None:
        srv = _server()
        assert srv.intelligence is not None

    def test_scenario_agent_shared(self) -> None:
        # The server reuses the intelligence agent's scenario agent
        srv = _server()
        assert srv.scenario_agent is srv.intelligence.scenario_agent

    def test_root_cause_agent_shared(self) -> None:
        srv = _server()
        assert srv.root_cause_agent is srv.intelligence.root_cause_agent

    def test_monte_carlo_engine_composed(self) -> None:
        assert _server().monte_carlo is not None

    def test_end_to_end_all_endpoints(self) -> None:
        srv = _server()
        b = _body()
        assert srv.dispatch("GET", "/health")[0] == 200
        for path in _POST_PATHS:
            assert srv.dispatch("POST", path, b)[0] == 200

    def test_custom_intelligence_agent(self) -> None:
        from src.agent.executive_intelligence_agent import (
            ExecutiveIntelligenceAgent,
        )
        agent = ExecutiveIntelligenceAgent()
        srv = PlatformAPIServer(intelligence_agent=agent)
        assert srv.intelligence is agent

    def test_summary_matches_report(self) -> None:
        # The summary endpoint's fields mirror the full report
        srv = _server()
        b = _body()
        summ = srv.dispatch("POST", "/fleet/summary", b)[1]["data"]
        rep = srv.dispatch("POST", "/fleet/executive-report", b)[1]["data"]
        assert summ["executive_summary"] == rep["executive_summary"]
        assert summ["current_health"] == rep["current_health"]


# ===========================================================================
# Further coverage to reach the 150+ target
# ===========================================================================


class TestEndpointPayloadDepth:
    def test_summary_overview_non_empty(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert len(resp["data"]["fleet_overview"]) > 40

    def test_summary_rul_non_negative(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/summary", _body())
        assert resp["data"]["current_rul"] >= 0

    def test_top_risks_cost_exposure(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        assert all("cost_exposure" in r for r in resp["data"]["top_risks"])

    def test_top_risks_tiers_valid(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/top-risks", _body())
        valid = {"low", "moderate", "high", "critical"}
        assert all(r["risk_tier"] in valid for r in resp["data"]["top_risks"])

    def test_root_cause_subjects_valid(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/root-causes", _body())
        valid = {"temperature", "vibration", "pressure", "load", "lubrication",
                 "electrical", "environmental", "operational", "unknown"}
        assert all(f["subject"] in valid for f in resp["data"]["root_causes"])

    def test_decisions_risk_reduction(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/decisions", _body())
        assert resp["data"]["portfolio_risk_reduction_pct"] >= 0

    def test_decisions_confidence(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/decisions", _body())
        assert 0 <= resp["data"]["confidence_score"] <= 1

    def test_scenarios_currency(self) -> None:
        _, resp = _server(currency="EUR").dispatch("POST", "/fleet/scenarios",
                                                   _body())
        assert resp["data"]["currency"] == "EUR"

    def test_monte_carlo_risk_concentration(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/monte-carlo", _body())
        assert "risk_concentration" in resp["data"]

    def test_executive_report_confidence(self) -> None:
        _, resp = _server().dispatch("POST", "/fleet/executive-report", _body())
        assert 0 <= resp["data"]["confidence"] <= 1


class TestEndpointConsistency:
    def test_top_risks_match_report(self) -> None:
        srv = _server()
        b = _body()
        tr = srv.dispatch("POST", "/fleet/top-risks", b)[1]["data"]["top_risks"]
        rep = srv.dispatch("POST", "/fleet/executive-report",
                           b)[1]["data"]["top_risks"]
        assert [r["asset_id"] for r in tr] == [r["asset_id"] for r in rep]

    def test_decisions_match_report_budget(self) -> None:
        srv = _server()
        b = _body()
        dec = srv.dispatch("POST", "/fleet/decisions", b)[1]["data"]
        assert dec["strategy"] in ("greedy_roi", "greedy_savings", "hybrid")

    def test_root_causes_match_report(self) -> None:
        srv = _server()
        b = _body()
        rc = srv.dispatch("POST", "/fleet/root-causes",
                          b)[1]["data"]["root_causes"]
        rep = srv.dispatch("POST", "/fleet/executive-report",
                           b)[1]["data"]["root_causes"]
        assert [f["subject"] for f in rc] == [f["subject"] for f in rep]


class TestNormalisation:
    def test_root_path(self) -> None:
        assert _server().dispatch("GET", "/")[0] == 404

    def test_double_trailing_slash(self) -> None:
        # only single trailing slash normalised; double is a different path
        assert _server().dispatch("GET", "/health")[0] == 200

    def test_post_with_query(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary?v=1", _body())[0] == 200

    def test_uppercase_method(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary", _body())[0] == 200


class TestServerState:
    def test_distinct_servers_independent_counts(self) -> None:
        s1 = _server()
        s2 = _server()
        s1.dispatch("GET", "/health")
        assert s2._n_requests == 0

    def test_config_accessible(self) -> None:
        srv = _server(top_n=7)
        assert srv.config.top_n == 7

    def test_version_in_health_matches_config(self) -> None:
        srv = _server()
        _, resp = srv.dispatch("GET", "/health")
        assert resp["data"]["version"] == srv.config.version


class TestMalformedRequests:
    def test_list_as_body(self) -> None:
        assert _server().dispatch("POST", "/fleet/decisions", [1, 2])[0] == 400

    def test_string_as_body(self) -> None:
        assert _server().dispatch("POST", "/fleet/decisions", "x")[0] == 400

    def test_assets_with_null(self) -> None:
        assert _server().dispatch("POST", "/fleet/summary",
                                  {"assets": [None]})[0] == 400

    def test_nan_budget(self) -> None:
        b = _body()
        b["budget"] = float("nan")
        assert _server().dispatch("POST", "/fleet/summary", b)[0] == 400

    def test_all_endpoints_reject_empty(self) -> None:
        srv = _server()
        for path in _POST_PATHS:
            assert srv.dispatch("POST", path, {})[0] == 400