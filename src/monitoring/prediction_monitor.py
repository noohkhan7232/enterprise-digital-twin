"""Prediction monitoring and the end-to-end monitoring CLI demo.

Tracks prediction and confidence distributions, latency percentiles,
throughput, error / success rates, inference time and prediction variance, and
detects prediction-distribution drift. Running this module as a script executes
a deterministic, end-to-end monitoring pipeline::

    python src/monitoring/prediction_monitor.py --demo
"""

from __future__ import annotations

if __package__ in (None, ""):  # pragma: no cover - exercised only as a script
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from monitoring.data_drift_detector import js_distance, population_stability_index
from monitoring.monitoring_models import (
    Clock,
    DeterministicIdGenerator,
    DriftMethod,
    DriftSeverity,
    IdGenerator,
    LogicalClock,
    MonitoringConfiguration,
    MonitoringError,
    PredictionDriftResult,
    PredictionStatistics,
    ValidationError,
)

__all__ = [
    "PredictionMonitor",
    "create_prediction_monitor",
    "run_demo",
    "main",
]

logger = logging.getLogger("monitoring.prediction_monitor")
logger.addHandler(logging.NullHandler())


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


class PredictionMonitor:
    """Thread-safe, deterministic monitor for model prediction telemetry."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="prediction")
        self._predictions: List[float] = []
        self._confidences: List[float] = []
        self._latencies: List[float] = []
        self._inference_times: List[float] = []
        self._errors = 0
        self._total = 0
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    # -- recording ---------------------------------------------------------- #
    def record(
        self,
        predictions: Sequence[float],
        *,
        confidences: Optional[Sequence[float]] = None,
        latencies_ms: Optional[Sequence[float]] = None,
        inference_times_ms: Optional[Sequence[float]] = None,
        errors: int = 0,
    ) -> None:
        """Record a batch of prediction telemetry."""
        preds = [float(p) for p in predictions]
        if errors < 0:
            raise ValidationError("errors must be non-negative")
        with self._lock:
            self._predictions.extend(preds)
            self._total += len(preds)
            self._errors += int(errors)
            if confidences is not None:
                self._confidences.extend(float(c) for c in confidences)
            if latencies_ms is not None:
                self._latencies.extend(float(x) for x in latencies_ms)
            if inference_times_ms is not None:
                self._inference_times.extend(float(x) for x in inference_times_ms)

    def reset(self) -> None:
        with self._lock:
            self._predictions.clear()
            self._confidences.clear()
            self._latencies.clear()
            self._inference_times.clear()
            self._errors = 0
            self._total = 0

    # -- statistics --------------------------------------------------------- #
    def statistics(self, *, elapsed_seconds: Optional[float] = None) -> PredictionStatistics:
        """Aggregate all recorded telemetry into prediction statistics."""
        with self._lock:
            preds = np.asarray(self._predictions, dtype=float)
            confs = np.asarray(self._confidences, dtype=float)
            lats = np.asarray(self._latencies, dtype=float)
            infs = np.asarray(self._inference_times, dtype=float)
            errors = self._errors
            total = self._total
        return self._build_statistics(preds, confs, lats, infs, errors, total, elapsed_seconds)

    def compute_statistics(
        self,
        predictions: Sequence[float],
        *,
        confidences: Optional[Sequence[float]] = None,
        latencies_ms: Optional[Sequence[float]] = None,
        inference_times_ms: Optional[Sequence[float]] = None,
        errors: int = 0,
        total: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> PredictionStatistics:
        """Compute statistics for a single batch without mutating state."""
        preds = np.asarray([float(p) for p in predictions], dtype=float)
        confs = np.asarray(list(confidences), dtype=float) if confidences is not None else np.array([])
        lats = np.asarray(list(latencies_ms), dtype=float) if latencies_ms is not None else np.array([])
        infs = np.asarray(list(inference_times_ms), dtype=float) if inference_times_ms is not None else np.array([])
        total = total if total is not None else preds.size
        return self._build_statistics(preds, confs, lats, infs, errors, total, elapsed_seconds)

    def _build_statistics(
        self,
        preds: np.ndarray,
        confs: np.ndarray,
        lats: np.ndarray,
        infs: np.ndarray,
        errors: int,
        total: int,
        elapsed_seconds: Optional[float],
    ) -> PredictionStatistics:
        if preds.size == 0:
            return PredictionStatistics(count=0, error_rate=0.0, success_rate=1.0)
        error_rate = float(errors / total) if total else 0.0
        throughput = float(total / elapsed_seconds) if elapsed_seconds and elapsed_seconds > 0 else None
        return PredictionStatistics(
            count=int(preds.size),
            mean=float(np.mean(preds)),
            std=float(np.std(preds)),
            minimum=float(np.min(preds)),
            maximum=float(np.max(preds)),
            variance=float(np.var(preds)),
            q05=_percentile(preds, 5),
            q50=_percentile(preds, 50),
            q95=_percentile(preds, 95),
            positive_rate=float(np.mean(preds >= 0.5)),
            confidence_mean=float(np.mean(confs)) if confs.size else None,
            confidence_std=float(np.std(confs)) if confs.size else None,
            latency_p50_ms=_percentile(lats, 50) if lats.size else None,
            latency_p95_ms=_percentile(lats, 95) if lats.size else None,
            latency_p99_ms=_percentile(lats, 99) if lats.size else None,
            throughput_per_s=throughput,
            error_rate=error_rate,
            success_rate=float(1.0 - error_rate),
            inference_time_ms=float(np.mean(infs)) if infs.size else None,
        )

    # -- prediction drift --------------------------------------------------- #
    def prediction_drift(
        self,
        reference: Sequence[float],
        current: Sequence[float],
        method: DriftMethod = DriftMethod.JS_DISTANCE,
    ) -> PredictionDriftResult:
        """Detect drift between reference and current prediction distributions."""
        ref = [float(x) for x in reference]
        cur = [float(x) for x in current]
        if not ref or not cur:
            raise ValidationError("prediction drift requires non-empty inputs")
        bins = self._config.num_bins
        if method is DriftMethod.JS_DISTANCE:
            score = js_distance(ref, cur, bins)
            threshold = self._config.js_threshold
        elif method is DriftMethod.PSI:
            score = population_stability_index(ref, cur, bins)
            threshold = self._config.psi_threshold
        else:
            raise ValidationError(f"Unsupported prediction drift method: {method}")
        from monitoring.data_drift_detector import DataDriftDetector

        severity = DataDriftDetector.classify_severity(score, threshold)
        return PredictionDriftResult(
            drift_score=score,
            method=method,
            severity=severity,
            drifted=score >= threshold,
            threshold=threshold,
            reference_size=len(ref),
            current_size=len(cur),
        )


def create_prediction_monitor(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> PredictionMonitor:
    """Factory for a configured :class:`PredictionMonitor`."""
    if deterministic:
        return PredictionMonitor(config=config, clock=LogicalClock(),
                                 id_generator=DeterministicIdGenerator(seed="prediction"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return PredictionMonitor(config=config, clock=SystemClock(),
                             id_generator=SequentialIdGenerator())


# --------------------------------------------------------------------------- #
# End-to-end CLI demo
# --------------------------------------------------------------------------- #
def run_demo(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the deterministic end-to-end monitoring pipeline demo.

    Reference dataset -> production dataset -> drift detection -> health
    evaluation -> alert generation -> dashboard snapshot -> executive summary.
    """
    from monitoring.data_drift_detector import create_data_drift_detector
    from monitoring.concept_drift_detector import create_concept_drift_detector
    from monitoring.data_quality_monitor import create_data_quality_monitor
    from monitoring.model_health_monitor import create_model_health_monitor
    from monitoring.monitoring_dashboard import create_monitoring_dashboard
    from monitoring.monitoring_models import FeatureType, MonitoringConfiguration

    config = MonitoringConfiguration()
    rng = np.random.default_rng(config.seed)
    n = 4000

    # 1. Reference dataset.
    reference = {
        "amount": rng.normal(100.0, 20.0, n),
        "age": rng.normal(45.0, 12.0, n),
        "tenure": rng.normal(30.0, 8.0, n),
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
    }
    # 2. Production dataset (drift injected on amount + region; tenure stable).
    production = {
        "amount": rng.normal(140.0, 28.0, n),
        "age": rng.normal(46.0, 12.0, n),
        "tenure": rng.normal(30.0, 8.0, n),
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.2, 0.2, 0.3, 0.3]),
    }
    feature_types = {
        "amount": FeatureType.NUMERICAL, "age": FeatureType.NUMERICAL,
        "tenure": FeatureType.NUMERICAL, "region": FeatureType.CATEGORICAL,
    }

    # 3. Drift detection.
    drift_detector = create_data_drift_detector(config=config)
    drift_results = drift_detector.detect_dataset(reference, production, feature_types)
    drift_summary = drift_detector.summarize(drift_results)

    concept_detector = create_concept_drift_detector(config=config)
    concept = concept_detector.detect_performance_drift(0.92, 0.86)

    # Prediction telemetry.
    monitor = create_prediction_monitor(config=config)
    ref_preds = rng.uniform(0.0, 1.0, n)
    prod_preds = np.clip(rng.uniform(0.0, 1.0, n) + 0.15, 0.0, 1.0)
    latencies = rng.normal(120.0, 30.0, n)
    monitor.record(prod_preds, confidences=prod_preds, latencies_ms=latencies, errors=180)
    prediction_stats = monitor.statistics(elapsed_seconds=60.0)
    prediction_drift = monitor.prediction_drift(ref_preds, prod_preds)

    # 4. Data quality.
    quality_monitor = create_data_quality_monitor(config=config)
    dirty = np.array(production["amount"], dtype=float)
    dirty[:40] = np.nan
    quality = quality_monitor.evaluate(
        {"amount": dirty, "age": production["age"], "tenure": production["tenure"]},
        valid_ranges={"age": (0.0, 120.0)},
    )

    # 5. Health evaluation.
    health_monitor = create_model_health_monitor(config=config)
    health = health_monitor.evaluate(
        accuracy_trend=[0.92, 0.91, 0.90, 0.88, 0.86],
        latency_trend=[100.0, 110.0, 120.0, 125.0, 130.0],
        drift_score=drift_summary["overall_drift_score"],
        prediction_stability=1.0 - prediction_drift.drift_score,
        availability=0.999,
        resource_usage=0.55,
        reliability=1.0 - prediction_stats.error_rate,
        freshness_seconds=120.0,
    )

    # 6 & 7. Alerts + dashboard + executive summary.
    dashboard = create_monitoring_dashboard(config=config)
    report = dashboard.build_report(
        model_id="fraud-detector",
        drift_results=drift_results,
        concept_drift=concept,
        prediction_drift=prediction_drift,
        health=health,
        quality=quality,
    )
    snapshot = dashboard.build_snapshot(
        model_id="fraud-detector",
        drift_results=drift_results,
        health=health,
        prediction_stats=prediction_stats,
        quality=quality,
        alerts=report.alerts,
    )
    executive_summary = dashboard.executive_summary(report, snapshot)

    result = {
        "drift_summary": drift_summary,
        "drift_results": [d.to_dict() for d in drift_results],
        "concept_drift": concept.to_dict(),
        "prediction_statistics": prediction_stats.to_dict(),
        "prediction_drift": prediction_drift.to_dict(),
        "quality": quality.to_dict(),
        "health": health.to_dict(),
        "report": report.to_dict(),
        "dashboard": snapshot.to_dict(),
        "executive_summary": executive_summary,
    }

    if output_dir is not None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        for name, payload in (
            ("monitoring_report.json", report.to_dict()),
            ("dashboard.json", snapshot.to_dict()),
            ("drift.json", {"summary": drift_summary,
                            "results": [d.to_dict() for d in drift_results]}),
            ("health.json", health.to_dict()),
        ):
            with open(os.path.join(output_dir, name), "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, indent=2, sort_keys=True))

    return result


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Enterprise production monitoring")
    parser.add_argument("--demo", action="store_true", help="Run the end-to-end demo")
    parser.add_argument("--output-dir", default=None, help="Directory for exported JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress info logging")
    args = parser.parse_args(argv)

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.demo:
        result = run_demo(output_dir=args.output_dir)
        summary = {
            "model_status": result["dashboard"]["model_status"],
            "health_level": result["health"]["level"],
            "health_score": round(result["health"]["overall"], 4),
            "overall_drift_score": round(result["drift_summary"]["overall_drift_score"], 4),
            "top_drifted_features": result["dashboard"]["top_drifted_features"][:3],
            "active_alerts": len(result["dashboard"]["active_alerts"]),
            "executive_summary": result["executive_summary"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())