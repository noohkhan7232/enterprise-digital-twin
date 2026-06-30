"""Enterprise Observability, Reliability & Production Operations platform.

Week 11 Phase 5 of the Enterprise Digital Twin & Decision Intelligence Platform.
Self-contained, deterministic, pure-Python (+ NumPy) observability: metrics,
tracing, structured logging, reliability, SLI/SLO, incident management, capacity
planning, an operations dashboard and a production-readiness assessment.
Integration with earlier subsystems (MLOps, monitoring, workflow, scheduler,
CI/CD, deployment) is by composition.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .observability_models import (
    Clock, IdGenerator, ManualClock, ValidationError,
    Severity, MetricType, SpanStatus, IncidentSeverity, IncidentStatus, SLIType,
    ResourceKind, ReadinessLevel,
    MetricPoint, MetricSeries, TraceSpan, TraceContext, LogRecord, TimelineEvent,
    Incident, IncidentTimeline, ReliabilityScore, SLI, SLO, ErrorBudget,
    CapacityForecast, ReadinessCheck, OperationsSnapshot, ProductionReport, RunbookEntry,
    percentile,
)
from .metrics_engine import MetricsEngine, create_metrics_engine
from .tracing_engine import Tracer, create_tracer
from .structured_logger import StructuredLogger, ListSink, create_logger
from .reliability_engine import ReliabilityEngine, create_reliability_engine
from .sli_slo_engine import SLISLOEngine, create_sli_slo_engine
from .incident_manager import IncidentManager, create_incident_manager
from .capacity_planner import CapacityPlanner, create_capacity_planner
from .operations_dashboard import OperationsDashboard, create_operations_dashboard
from .production_readiness import ProductionReadiness, create_production_readiness

__version__ = "11.5.0"

__all__ = [
    "__version__",
    "Clock", "IdGenerator", "ManualClock", "ValidationError",
    "Severity", "MetricType", "SpanStatus", "IncidentSeverity", "IncidentStatus",
    "SLIType", "ResourceKind", "ReadinessLevel",
    "MetricPoint", "MetricSeries", "TraceSpan", "TraceContext", "LogRecord",
    "TimelineEvent", "Incident", "IncidentTimeline", "ReliabilityScore", "SLI", "SLO",
    "ErrorBudget", "CapacityForecast", "ReadinessCheck", "OperationsSnapshot",
    "ProductionReport", "RunbookEntry", "percentile",
    "MetricsEngine", "create_metrics_engine",
    "Tracer", "create_tracer",
    "StructuredLogger", "ListSink", "create_logger",
    "ReliabilityEngine", "create_reliability_engine",
    "SLISLOEngine", "create_sli_slo_engine",
    "IncidentManager", "create_incident_manager",
    "CapacityPlanner", "create_capacity_planner",
    "OperationsDashboard", "create_operations_dashboard",
    "ProductionReadiness", "create_production_readiness",
    "run_demo",
]


def _demo_metrics() -> Dict[str, Any]:
    engine = MetricsEngine(clock=Clock())
    engine.record_many("inference_latency_ms", [float(v) for v in range(10, 110)],
                       category="inference", metric_type=MetricType.HISTOGRAM, unit="ms")
    engine.record_many("requests", [100.0, 110.0, 120.0, 130.0, 140.0],
                       category="application", metric_type=MetricType.COUNTER)
    return {"aggregate": engine.aggregate("inference_latency_ms"),
            "trend": engine.trend("requests")}


def _demo_tracing() -> Dict[str, Any]:
    tracer = Tracer(clock=Clock())
    tracer.record_span("api", trace_id="t1", start_time=0.0, end_time=12.0, span_id="api")
    tracer.record_span("db", trace_id="t1", start_time=1.0, end_time=9.0, span_id="db", parent_id="api")
    tracer.record_span("cache", trace_id="t1", start_time=1.0, end_time=2.0, span_id="cache", parent_id="api")
    tracer.record_span("query", trace_id="t1", start_time=2.0, end_time=8.0, span_id="query", parent_id="db")
    return tracer.export("t1")


def _demo_reliability() -> Dict[str, Any]:
    engine = ReliabilityEngine()
    engine.record_outcome(True, count=9990)
    engine.record_outcome(False, count=10)
    engine.set_observation_period(2592000.0)
    engine.record_failure_window(1000.0, 1600.0)
    return engine.reliability_score().to_dict()


def _demo_capacity() -> Dict[str, Any]:
    planner = CapacityPlanner(headroom=0.2)
    forecast = planner.forecast(ResourceKind.CPU,
                                [0.40, 0.43, 0.46, 0.49, 0.52, 0.55],
                                horizon=6, capacity_limit=0.8, strategy="linear", unit="ratio")
    return forecast.to_dict()


def _demo_production_readiness(root: str) -> Dict[str, Any]:
    return ProductionReadiness(root, clock=Clock(), reliability_score=0.99,
                               tests_passed=350, tests_failed=0, coverage=0.92).summary()


def run_demo(root: str = ".") -> Dict[str, Any]:
    """Run all subsystem demos and return a single deterministic report."""
    return {
        "version": __version__,
        "metrics": _demo_metrics(),
        "tracing": _demo_tracing(),
        "reliability": _demo_reliability(),
        "capacity": _demo_capacity(),
        "production_readiness": _demo_production_readiness(root),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Observability platform CLI demos")
    parser.add_argument("demo", nargs="?", default="all",
                        choices=["all", "metrics", "tracing", "reliability", "capacity", "readiness"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    dispatch = {
        "metrics": _demo_metrics,
        "tracing": _demo_tracing,
        "reliability": _demo_reliability,
        "capacity": _demo_capacity,
        "readiness": lambda: _demo_production_readiness(args.root),
        "all": lambda: run_demo(args.root),
    }
    result = dispatch[args.demo]()
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())