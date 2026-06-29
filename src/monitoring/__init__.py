"""Enterprise Production Monitoring & Data Drift Intelligence subsystem.

Week 11 — Phase 2. Pure Python + NumPy. Deterministic, thread-safe and
JSON-serializable. Integrates with the MLOps platform, model registry,
experiment tracker, workflow engine, scheduler, event bus, integration layer
and executive copilot *by composition* — it imports none of them.
"""

from __future__ import annotations

from monitoring.monitoring_models import (
    AlertLevel,
    AlertPolicy,
    AlertType,
    Clock,
    Comparison,
    ConceptDriftResult,
    DashboardSnapshot,
    DataDriftResult,
    DeterministicIdGenerator,
    DistributionSnapshot,
    DriftMethod,
    DriftSeverity,
    FeatureStatistics,
    FeatureType,
    FixedClock,
    HealthLevel,
    HealthScore,
    IdGenerator,
    LogicalClock,
    ModelHealthStatus,
    MonitoringAlert,
    MonitoringConfiguration,
    MonitoringError,
    MonitoringReport,
    MonitoringStatistics,
    PredictionDriftResult,
    PredictionStatistics,
    QualityIssue,
    QualityIssueType,
    QualityMetrics,
    SequentialIdGenerator,
    SerializationError,
    SystemClock,
    ValidationError,
    freeze_mapping,
    thaw_mapping,
)
from monitoring.data_drift_detector import (
    DataDriftDetector,
    DriftComputationError,
    create_data_drift_detector,
    histogram_distance,
    js_distance,
    kl_divergence,
    ks_statistic,
    population_stability_index,
)
from monitoring.concept_drift_detector import (
    ConceptDriftDetector,
    create_concept_drift_detector,
)
from monitoring.prediction_monitor import (
    PredictionMonitor,
    create_prediction_monitor,
    run_demo,
)
from monitoring.model_health_monitor import (
    DEFAULT_HEALTH_WEIGHTS,
    ModelHealthMonitor,
    create_model_health_monitor,
)
from monitoring.data_quality_monitor import (
    DataQualityMonitor,
    create_data_quality_monitor,
)
from monitoring.monitoring_dashboard import (
    AlertEngine,
    MonitoringDashboard,
    create_alert_engine,
    create_monitoring_dashboard,
)

__version__ = "11.2.0"

__all__ = [
    "__version__",
    # models
    "FeatureStatistics", "DistributionSnapshot", "DataDriftResult", "ConceptDriftResult",
    "PredictionStatistics", "PredictionDriftResult", "ModelHealthStatus", "HealthScore",
    "MonitoringAlert", "MonitoringReport", "QualityMetrics", "QualityIssue",
    "DashboardSnapshot", "MonitoringStatistics", "MonitoringConfiguration", "AlertPolicy",
    # enums
    "FeatureType", "DriftMethod", "DriftSeverity", "AlertLevel", "AlertType",
    "HealthLevel", "QualityIssueType", "Comparison",
    # infra
    "Clock", "SystemClock", "FixedClock", "LogicalClock",
    "IdGenerator", "SequentialIdGenerator", "DeterministicIdGenerator",
    "freeze_mapping", "thaw_mapping",
    # exceptions
    "MonitoringError", "ValidationError", "SerializationError", "DriftComputationError",
    # drift
    "DataDriftDetector", "create_data_drift_detector",
    "population_stability_index", "kl_divergence", "js_distance",
    "ks_statistic", "histogram_distance",
    "ConceptDriftDetector", "create_concept_drift_detector",
    # prediction
    "PredictionMonitor", "create_prediction_monitor", "run_demo",
    # health
    "ModelHealthMonitor", "create_model_health_monitor", "DEFAULT_HEALTH_WEIGHTS",
    # quality
    "DataQualityMonitor", "create_data_quality_monitor",
    # dashboard
    "AlertEngine", "MonitoringDashboard", "create_alert_engine", "create_monitoring_dashboard",
]