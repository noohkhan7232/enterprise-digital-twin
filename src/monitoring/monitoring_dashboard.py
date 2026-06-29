"""Alert engine and monitoring dashboard.

Provides a deterministic, thread-safe :class:`AlertEngine` implementing the
Observer pattern (threshold, trend, composite and repeated alerts with
deduplication) and a :class:`MonitoringDashboard` that assembles monitoring
reports, dashboard snapshots and executive summaries from drift, health,
prediction and data-quality signals.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from monitoring.data_drift_detector import DataDriftDetector
from monitoring.monitoring_models import (
    AlertLevel,
    AlertPolicy,
    AlertType,
    Clock,
    Comparison,
    ConceptDriftResult,
    DataDriftResult,
    DeterministicIdGenerator,
    HealthScore,
    IdGenerator,
    LogicalClock,
    MonitoringAlert,
    MonitoringConfiguration,
    MonitoringError,
    MonitoringReport,
    MonitoringStatistics,
    PredictionDriftResult,
    PredictionStatistics,
    QualityMetrics,
    DashboardSnapshot,
    ValidationError,
)

__all__ = [
    "AlertEngine",
    "MonitoringDashboard",
    "create_alert_engine",
    "create_monitoring_dashboard",
    "AlertObserver",
]

logger = logging.getLogger("monitoring.dashboard")
logger.addHandler(logging.NullHandler())

AlertObserver = Union[Callable[[MonitoringAlert], None], Any]

_LEVEL_ORDER = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.HIGH: 2, AlertLevel.CRITICAL: 3}


def _fingerprint(metric: str, level: AlertLevel, entity: str) -> str:
    digest = hashlib.sha256(f"{metric}|{level.value}|{entity}".encode("utf-8")).hexdigest()
    return digest[:16]


def default_alert_policies(config: MonitoringConfiguration) -> Tuple[AlertPolicy, ...]:
    """Build the standard alert policy set from a configuration."""
    return (
        AlertPolicy("overall_drift", "overall_drift_score", AlertLevel.HIGH,
                    Comparison.GT, config.drift_threshold),
        AlertPolicy("prediction_drift", "prediction_drift_score", AlertLevel.WARNING,
                    Comparison.GT, config.js_threshold),
        AlertPolicy("concept_drift", "concept_drift_score", AlertLevel.HIGH,
                    Comparison.GT, 0.05),
        AlertPolicy("health_warn", "health_score", AlertLevel.WARNING,
                    Comparison.LT, config.health_healthy),
        AlertPolicy("health_critical", "health_score", AlertLevel.CRITICAL,
                    Comparison.LT, config.health_warning),
        AlertPolicy("null_rate", "null_rate", AlertLevel.WARNING,
                    Comparison.GT, config.null_rate_threshold),
        AlertPolicy("duplicate_rate", "duplicate_rate", AlertLevel.WARNING,
                    Comparison.GT, config.duplicate_rate_threshold),
        AlertPolicy("error_rate", "error_rate", AlertLevel.HIGH,
                    Comparison.GT, config.error_rate_threshold),
        AlertPolicy("latency_p95", "latency_p95_ms", AlertLevel.WARNING,
                    Comparison.GT, config.latency_p95_threshold_ms),
    )


class AlertEngine:
    """Deterministic alert engine with Observer-pattern fan-out."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        policies: Optional[Sequence[AlertPolicy]] = None,
        repeat_threshold: int = 2,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="alert")
        self._policies: List[AlertPolicy] = list(
            policies if policies is not None else default_alert_policies(self._config)
        )
        self._repeat_threshold = max(2, int(repeat_threshold))
        self._observers: List[AlertObserver] = []
        self._seen: Dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def policies(self) -> Tuple[AlertPolicy, ...]:
        return tuple(self._policies)

    # -- Observer pattern --------------------------------------------------- #
    def subscribe(self, observer: AlertObserver) -> None:
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def unsubscribe(self, observer: AlertObserver) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def _notify(self, alert: MonitoringAlert) -> None:
        for observer in list(self._observers):
            if callable(observer):
                observer(alert)
            elif hasattr(observer, "on_alert"):
                observer.on_alert(alert)

    def add_policy(self, policy: AlertPolicy) -> None:
        with self._lock:
            self._policies.append(policy)

    # -- alert construction ------------------------------------------------- #
    def _make_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        metric: str,
        value: Optional[float],
        threshold: Optional[float],
        entity: str,
        message: str,
    ) -> MonitoringAlert:
        fingerprint = _fingerprint(metric, level, entity)
        with self._lock:
            count = self._seen.get(fingerprint, 0) + 1
            self._seen[fingerprint] = count
        effective_type = AlertType.REPEATED if count >= self._repeat_threshold else alert_type
        alert = MonitoringAlert(
            alert_id=self._ids.generate("alert"),
            level=level,
            alert_type=effective_type,
            title=title,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            entity=entity,
            fingerprint=fingerprint,
            count=count,
            created_at=self._clock.now(),
        )
        self._notify(alert)
        return alert

    def evaluate_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        entity: str = "",
        slopes: Optional[Mapping[str, float]] = None,
    ) -> List[MonitoringAlert]:
        """Evaluate threshold and trend policies, returning deduped alerts.

        Only the most severe firing policy per metric is retained.
        """
        slopes = dict(slopes or {})
        best: Dict[str, AlertPolicy] = {}
        best_value: Dict[str, float] = {}
        for policy in self._policies:
            if not policy.enabled:
                continue
            if policy.alert_type is AlertType.TREND:
                source = slopes
            else:
                source = metrics
            if policy.metric not in source:
                continue
            value = float(source[policy.metric])
            if not policy.evaluate(value):
                continue
            current = best.get(policy.metric)
            if current is None or _LEVEL_ORDER[policy.level] > _LEVEL_ORDER[current.level]:
                best[policy.metric] = policy
                best_value[policy.metric] = value
        alerts: List[MonitoringAlert] = []
        for metric in sorted(best):
            policy = best[metric]
            value = best_value[metric]
            alerts.append(self._make_alert(
                level=policy.level,
                alert_type=policy.alert_type,
                title=f"{metric} breached",
                metric=metric,
                value=value,
                threshold=policy.threshold,
                entity=entity,
                message=f"{metric}={value:.4f} {policy.comparison.value} {policy.threshold:.4f}",
            ))
        return alerts

    def composite_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        *,
        entity: str = "",
        components: Sequence[str] = (),
    ) -> MonitoringAlert:
        """Emit a composite alert summarising several co-occurring conditions."""
        return self._make_alert(
            level=level,
            alert_type=AlertType.COMPOSITE,
            title=name,
            metric="composite",
            value=None,
            threshold=None,
            entity=entity,
            message=message + (f" [{', '.join(components)}]" if components else ""),
        )

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


class MonitoringDashboard:
    """Assembles monitoring reports, snapshots and executive summaries."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        alert_engine: Optional[AlertEngine] = None,
        top_k: int = 5,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="dashboard")
        self._alerts = alert_engine or AlertEngine(config=self._config)
        self._drift = DataDriftDetector(config=self._config)
        self._top_k = int(top_k)
        self._reports = 0
        self._alert_count = 0
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    @property
    def alert_engine(self) -> AlertEngine:
        return self._alerts

    def _metric_bundle(
        self,
        overall_drift_score: float,
        concept_drift: Optional[ConceptDriftResult],
        prediction_drift: Optional[PredictionDriftResult],
        health: Optional[HealthScore],
        quality: Optional[QualityMetrics],
        prediction_stats: Optional[PredictionStatistics],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {"overall_drift_score": overall_drift_score}
        if concept_drift is not None:
            metrics["concept_drift_score"] = concept_drift.drift_score
        if prediction_drift is not None:
            metrics["prediction_drift_score"] = prediction_drift.drift_score
        if health is not None:
            metrics["health_score"] = health.overall
        if quality is not None:
            metrics["null_rate"] = quality.null_rate
            metrics["duplicate_rate"] = quality.duplicate_rate
        if prediction_stats is not None:
            metrics["error_rate"] = prediction_stats.error_rate
            if prediction_stats.latency_p95_ms is not None:
                metrics["latency_p95_ms"] = prediction_stats.latency_p95_ms
        return metrics

    def build_report(
        self,
        *,
        model_id: str,
        drift_results: Sequence[DataDriftResult],
        concept_drift: Optional[ConceptDriftResult] = None,
        prediction_drift: Optional[PredictionDriftResult] = None,
        health: Optional[HealthScore] = None,
        quality: Optional[QualityMetrics] = None,
        prediction_stats: Optional[PredictionStatistics] = None,
    ) -> MonitoringReport:
        """Assemble a consolidated monitoring report with generated alerts."""
        if not model_id:
            raise ValidationError("model_id is required")
        overall = self._drift.overall_drift_score(drift_results)
        metrics = self._metric_bundle(overall, concept_drift, prediction_drift,
                                      health, quality, prediction_stats)
        alerts = list(self._alerts.evaluate_metrics(metrics, entity=model_id))

        # Composite escalation: drift and health both unhealthy.
        drift_alert = any(a.metric == "overall_drift_score" for a in alerts)
        health_alert = any(a.metric == "health_score" for a in alerts)
        if drift_alert and health_alert:
            alerts.append(self._alerts.composite_alert(
                "model_degradation", AlertLevel.CRITICAL,
                "Concurrent drift and health degradation detected",
                entity=model_id, components=["overall_drift_score", "health_score"],
            ))

        ordered = tuple(sorted(
            alerts, key=lambda a: (-_LEVEL_ORDER[a.level], a.metric, a.alert_id)))
        with self._lock:
            self._reports += 1
            self._alert_count += len(ordered)
        summary = (
            f"Model {model_id}: drift={overall:.3f}, "
            f"health={(health.overall if health else 1.0):.3f}, alerts={len(ordered)}"
        )
        return MonitoringReport(
            report_id=self._ids.generate("report"),
            created_at=self._clock.now(),
            model_id=model_id,
            drift_results=tuple(drift_results),
            concept_drift=concept_drift,
            prediction_drift=prediction_drift,
            health=health,
            quality=quality,
            alerts=ordered,
            overall_drift_score=overall,
            summary=summary,
        )

    def build_snapshot(
        self,
        *,
        model_id: str,
        drift_results: Sequence[DataDriftResult],
        health: Optional[HealthScore] = None,
        prediction_stats: Optional[PredictionStatistics] = None,
        quality: Optional[QualityMetrics] = None,
        alerts: Sequence[MonitoringAlert] = (),
    ) -> DashboardSnapshot:
        """Assemble a point-in-time dashboard snapshot."""
        if not model_id:
            raise ValidationError("model_id is required")
        overall = self._drift.overall_drift_score(drift_results)
        ranking = self._drift.feature_drift_ranking(drift_results)[: self._top_k]
        drift_summary = {
            "overall_drift_score": overall,
            "feature_count": len(drift_results),
            "drifted_count": len(self._drift.drifted_features(drift_results)),
        }
        prediction_trends: Dict[str, float] = {}
        latency_trends: Dict[str, float] = {}
        if prediction_stats is not None:
            prediction_trends = {
                "mean": prediction_stats.mean,
                "variance": prediction_stats.variance,
                "positive_rate": prediction_stats.positive_rate or 0.0,
                "error_rate": prediction_stats.error_rate,
            }
            if prediction_stats.latency_p50_ms is not None:
                latency_trends["p50"] = prediction_stats.latency_p50_ms
            if prediction_stats.latency_p95_ms is not None:
                latency_trends["p95"] = prediction_stats.latency_p95_ms
            if prediction_stats.latency_p99_ms is not None:
                latency_trends["p99"] = prediction_stats.latency_p99_ms
        model_status = health.level.value if health is not None else "UNKNOWN"
        return DashboardSnapshot(
            snapshot_id=self._ids.generate("snapshot"),
            model_id=model_id,
            created_at=self._clock.now(),
            health=health,
            drift_summary=drift_summary,
            top_drifted_features=tuple(ranking),
            prediction_trends=prediction_trends,
            latency_trends=latency_trends,
            data_quality=quality,
            active_alerts=tuple(alerts),
            model_status=model_status,
        )

    def executive_summary(
        self, report: MonitoringReport, snapshot: DashboardSnapshot
    ) -> str:
        """Produce a deterministic, human-readable executive summary."""
        lines = [
            f"Executive Monitoring Summary for model '{report.model_id}'",
            f"Status: {snapshot.model_status}",
            f"Overall drift score: {report.overall_drift_score:.3f} "
            f"({dict(snapshot.drift_summary).get('drifted_count', 0)} of "
            f"{dict(snapshot.drift_summary).get('feature_count', 0)} features drifted)",
        ]
        if report.health is not None:
            lines.append(f"Health score: {report.health.overall:.3f} ({report.health.level.value})")
        if snapshot.top_drifted_features:
            top = ", ".join(f"{name} ({score:.3f})" for name, score in snapshot.top_drifted_features[:3])
            lines.append(f"Top drifted features: {top}")
        if report.quality is not None:
            lines.append(
                f"Data quality: completeness={report.quality.completeness:.3f}, "
                f"validity={report.quality.validity_rate:.3f}, issues={report.quality.issue_count}"
            )
        critical = sum(1 for a in report.alerts if a.level is AlertLevel.CRITICAL)
        high = sum(1 for a in report.alerts if a.level is AlertLevel.HIGH)
        lines.append(f"Active alerts: {len(report.alerts)} (critical={critical}, high={high})")
        if critical:
            lines.append("Recommendation: immediate investigation required; consider rollback.")
        elif high:
            lines.append("Recommendation: schedule retraining and review drifted features.")
        else:
            lines.append("Recommendation: continue monitoring; no action required.")
        return "\n".join(lines)

    def statistics(self) -> MonitoringStatistics:
        """Return aggregate dashboard statistics."""
        with self._lock:
            return MonitoringStatistics(
                total_reports=self._reports,
                total_alerts=self._alert_count,
                generated_at=self._clock.now(),
            )


def create_alert_engine(
    *, config: Optional[MonitoringConfiguration] = None,
    policies: Optional[Sequence[AlertPolicy]] = None, deterministic: bool = True,
) -> AlertEngine:
    """Factory for a configured :class:`AlertEngine`."""
    if deterministic:
        return AlertEngine(config=config, clock=LogicalClock(),
                           id_generator=DeterministicIdGenerator(seed="alert"), policies=policies)
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return AlertEngine(config=config, clock=SystemClock(),
                       id_generator=SequentialIdGenerator(), policies=policies)


def create_monitoring_dashboard(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> MonitoringDashboard:
    """Factory for a configured :class:`MonitoringDashboard`."""
    if deterministic:
        return MonitoringDashboard(config=config, clock=LogicalClock(),
                                   id_generator=DeterministicIdGenerator(seed="dashboard"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return MonitoringDashboard(config=config, clock=SystemClock(),
                               id_generator=SequentialIdGenerator())