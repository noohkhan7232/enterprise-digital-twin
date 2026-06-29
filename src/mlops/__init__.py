"""Enterprise MLOps subsystem for the Digital Twin & Decision Intelligence Platform.

This package provides a deterministic, dependency-injected MLOps stack:
immutable domain models, an experiment tracker, a model registry with a
lifecycle/promotion engine, a metadata-only artifact store and a
reproducibility engine. It is pure Python + NumPy and integrates with the
existing platform by composition only.
"""

from __future__ import annotations

from mlops.experiment_models import (
    ArtifactMetadata,
    ArtifactType,
    Clock,
    DeterministicIdGenerator,
    Experiment,
    ExperimentMetadata,
    ExperimentMetrics,
    ExperimentRun,
    ExperimentStatistics,
    ExperimentStatus,
    EnvironmentSnapshot,
    FixedClock,
    HyperParameters,
    IdGenerator,
    LogicalClock,
    MetricDirection,
    MLOpsError,
    ModelArtifact,
    ModelCard,
    ModelStage,
    ModelVersion,
    PromotionDecision,
    PromotionRequest,
    PromotionStatus,
    DatasetVersion,
    RegisteredModel,
    RegistryStatistics,
    ReproducibilitySnapshot,
    SemanticVersion,
    SequentialIdGenerator,
    SerializationError,
    SystemClock,
    ValidationError,
)
from mlops.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    DuplicateArtifactError,
    Sha256HashStrategy,
    create_artifact_store,
)
from mlops.reproducibility import (
    EnvironmentProvider,
    ReproducibilityEngine,
    ReproducibilityError,
    StaticEnvironmentProvider,
    SystemEnvironmentProvider,
    create_reproducibility_engine,
)
from mlops.model_registry import (
    DefaultPromotionPolicy,
    DuplicateModelVersionError,
    InvalidStageTransitionError,
    ModelNotFoundError,
    ModelRegistry,
    PromotionPolicy,
    RegistryError,
    VersionNotFoundError,
    WeightedScoreComparator,
    create_model_registry,
)
from mlops.experiment_tracker import (
    DuplicateExperimentError,
    DuplicateRunError,
    ExperimentNotFoundError,
    ExperimentTracker,
    ExperimentTrackerError,
    RunNotFoundError,
    create_experiment_tracker,
    run_demo,
)

__version__ = "11.1.0"

__all__ = [
    "__version__",
    # models
    "MLOpsError", "ValidationError", "SerializationError",
    "ExperimentStatus", "ModelStage", "PromotionStatus", "ArtifactType", "MetricDirection",
    "SemanticVersion", "HyperParameters", "ExperimentMetrics", "ExperimentMetadata",
    "Experiment", "ExperimentRun", "ArtifactMetadata", "ModelArtifact", "DatasetVersion",
    "EnvironmentSnapshot", "ModelCard", "RegisteredModel", "ModelVersion",
    "PromotionRequest", "PromotionDecision", "RegistryStatistics", "ExperimentStatistics",
    "ReproducibilitySnapshot",
    # infra
    "Clock", "SystemClock", "FixedClock", "LogicalClock",
    "IdGenerator", "SequentialIdGenerator", "DeterministicIdGenerator",
    # artifact store
    "ArtifactStore", "ArtifactStoreError", "ArtifactNotFoundError", "DuplicateArtifactError",
    "Sha256HashStrategy", "create_artifact_store",
    # reproducibility
    "ReproducibilityEngine", "ReproducibilityError", "EnvironmentProvider",
    "SystemEnvironmentProvider", "StaticEnvironmentProvider", "create_reproducibility_engine",
    # registry
    "ModelRegistry", "RegistryError", "ModelNotFoundError", "VersionNotFoundError",
    "DuplicateModelVersionError", "InvalidStageTransitionError", "PromotionPolicy",
    "DefaultPromotionPolicy", "WeightedScoreComparator", "create_model_registry",
    # tracker
    "ExperimentTracker", "ExperimentTrackerError", "ExperimentNotFoundError",
    "RunNotFoundError", "DuplicateExperimentError", "DuplicateRunError",
    "create_experiment_tracker", "run_demo",
]