"""Immutable domain models for the Enterprise MLOps subsystem.

This module defines the foundational, deterministic, JSON-serializable value
objects used across the MLOps platform (experiment tracking, model registry,
artifact store and reproducibility engine).

Design rules enforced here:

* Every aggregate is an immutable ``@dataclass(frozen=True, slots=True)``.
* Every model exposes ``to_dict()`` / ``from_dict()`` for lossless,
  deterministic JSON round-tripping.
* No wall-clock or randomness is consumed inside this module; timestamps and
  identifiers are supplied by callers so behaviour is fully reproducible.
* Mapping-like fields are stored as sorted tuples of key/value pairs so that
  instances remain hashable, comparable and deterministically serialisable.

The module is self-contained (pure Python + NumPy only) and integrates with the
existing platform by composition: nothing here imports from prior weeks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MLOpsError",
    "ValidationError",
    "SerializationError",
    "ExperimentStatus",
    "ModelStage",
    "PromotionStatus",
    "ArtifactType",
    "MetricDirection",
    "SemanticVersion",
    "HyperParameters",
    "ExperimentMetrics",
    "ExperimentMetadata",
    "Experiment",
    "ExperimentRun",
    "ArtifactMetadata",
    "ModelArtifact",
    "DatasetVersion",
    "EnvironmentSnapshot",
    "ModelCard",
    "RegisteredModel",
    "ModelVersion",
    "PromotionRequest",
    "PromotionDecision",
    "RegistryStatistics",
    "ExperimentStatistics",
    "ReproducibilitySnapshot",
    "freeze_mapping",
    "thaw_mapping",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class MLOpsError(Exception):
    """Base exception for every error raised by the MLOps subsystem."""


class ValidationError(MLOpsError):
    """Raised when a domain invariant is violated during construction."""


class SerializationError(MLOpsError):
    """Raised when a payload cannot be deserialised into a domain model."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment or run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class ModelStage(str, Enum):
    """Deterministic model lifecycle stages."""

    REGISTERED = "REGISTERED"
    VALIDATION = "VALIDATION"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"


class PromotionStatus(str, Enum):
    """Outcome of a promotion evaluation."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ArtifactType(str, Enum):
    """Supported artifact categories (metadata-only)."""

    SERIALIZED_MODEL = "SERIALIZED_MODEL"
    EVALUATION_REPORT = "EVALUATION_REPORT"
    ENGINEERING_REPORT = "ENGINEERING_REPORT"
    PUBLICATION_FIGURE = "PUBLICATION_FIGURE"
    FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
    SHAP_RESULT = "SHAP_RESULT"
    CONFUSION_MATRIX = "CONFUSION_MATRIX"
    ROC_CURVE = "ROC_CURVE"
    PR_CURVE = "PR_CURVE"
    CALIBRATION_PLOT = "CALIBRATION_PLOT"


class MetricDirection(str, Enum):
    """Whether a metric is better when larger or smaller."""

    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


# --------------------------------------------------------------------------- #
# Helpers for deterministic, immutable, JSON-friendly collections
# --------------------------------------------------------------------------- #
_SCALAR_TYPES = (str, int, float, bool, type(None))

FrozenMap = Tuple[Tuple[str, Any], ...]


def _coerce_scalar(value: Any) -> Any:
    """Validate that *value* is a JSON scalar, returning it unchanged."""
    if isinstance(value, bool):
        return value
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise ValidationError(
        f"Unsupported non-scalar mapping value of type {type(value).__name__!r}"
    )


def freeze_mapping(data: Optional[Mapping[str, Any]]) -> FrozenMap:
    """Convert a mapping into a deterministic, hashable tuple of pairs."""
    if data is None:
        return ()
    if isinstance(data, tuple):
        # Already frozen; revalidate for safety and re-sort.
        items = [(str(k), _coerce_scalar(v)) for k, v in data]
    else:
        items = [(str(k), _coerce_scalar(v)) for k, v in dict(data).items()]
    return tuple(sorted(items, key=lambda kv: kv[0]))


def thaw_mapping(frozen: FrozenMap) -> Dict[str, Any]:
    """Convert a frozen mapping back into a plain dictionary."""
    return {k: v for k, v in frozen}


def _coerce_str_tuple(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Coerce an iterable of values into a deterministic tuple of strings."""
    if values is None:
        return ()
    return tuple(str(v) for v in values)


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name!r} must be a non-empty string")


def _require_non_negative(value: float, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{field_name!r} must be numeric")
    if value < 0:
        raise ValidationError(f"{field_name!r} must be non-negative, got {value}")


# --------------------------------------------------------------------------- #
# Semantic version value object
# --------------------------------------------------------------------------- #
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-.]+))?$"
)


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    """A strict ``MAJOR.MINOR.PATCH[-prerelease]`` semantic version.

    Ordering follows semver precedence: numeric core ascending, and a release
    (no prerelease) outranks any prerelease of the same core.
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(f"SemanticVersion.{name} must be a non-negative int")
        if self.prerelease is not None:
            if not isinstance(self.prerelease, str) or not self.prerelease:
                raise ValidationError("SemanticVersion.prerelease must be a non-empty string")
            if not re.fullmatch(r"[0-9A-Za-z-.]+", self.prerelease):
                raise ValidationError("SemanticVersion.prerelease has invalid characters")

    @classmethod
    def parse(cls, text: str) -> "SemanticVersion":
        """Parse a semantic version string, raising ``ValidationError`` if invalid."""
        if not isinstance(text, str):
            raise ValidationError("Semantic version must be a string")
        match = _SEMVER_RE.match(text.strip())
        if not match:
            raise ValidationError(f"Invalid semantic version: {text!r}")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
        )

    @staticmethod
    def is_valid(text: str) -> bool:
        """Return ``True`` if *text* is a valid semantic version."""
        return isinstance(text, str) and bool(_SEMVER_RE.match(text.strip()))

    # Ordering: dataclass(order=True) compares field tuple; we must ensure a
    # release outranks a prerelease. We override the comparison helpers.
    def _precedence(self) -> Tuple[int, int, int, int, str]:
        # prerelease present -> rank 0 (lower); absent -> rank 1 (higher).
        has_release = 1 if self.prerelease is None else 0
        return (self.major, self.minor, self.patch, has_release, self.prerelease or "")

    def __lt__(self, other: "SemanticVersion") -> bool:  # type: ignore[override]
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._precedence() < other._precedence()

    def __le__(self, other: "SemanticVersion") -> bool:  # type: ignore[override]
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._precedence() <= other._precedence()

    def __gt__(self, other: "SemanticVersion") -> bool:  # type: ignore[override]
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._precedence() > other._precedence()

    def __ge__(self, other: "SemanticVersion") -> bool:  # type: ignore[override]
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._precedence() >= other._precedence()

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        return f"{core}-{self.prerelease}" if self.prerelease else core

    def to_dict(self) -> Dict[str, Any]:
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticVersion":
        try:
            return cls(
                major=int(data["major"]),
                minor=int(data["minor"]),
                patch=int(data["patch"]),
                prerelease=data.get("prerelease"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SerializationError(f"Cannot deserialise SemanticVersion: {exc}") from exc


# --------------------------------------------------------------------------- #
# Core experiment models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HyperParameters:
    """An immutable, ordered collection of training hyper-parameters."""

    values: FrozenMap = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_mapping(self.values))

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.values:
            if k == key:
                return v
        return default

    def with_value(self, key: str, value: Any) -> "HyperParameters":
        data = thaw_mapping(self.values)
        data[str(key)] = value
        return HyperParameters(freeze_mapping(data))

    def as_dict(self) -> Dict[str, Any]:
        return thaw_mapping(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {"values": thaw_mapping(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HyperParameters":
        return cls(freeze_mapping(data.get("values", {})))


@dataclass(frozen=True, slots=True)
class ExperimentMetrics:
    """Standard evaluation metrics plus an extensible additional mapping."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    loss: Optional[float] = None
    additional: FrozenMap = ()

    _STANDARD = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss")

    def __post_init__(self) -> None:
        object.__setattr__(self, "additional", freeze_mapping(self.additional))

    def get(self, name: str, default: Any = None) -> Any:
        """Return a metric value by name, searching standard then additional."""
        if name in self._STANDARD:
            return getattr(self, name)
        for k, v in self.additional:
            if k == name:
                return v
        return default

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {k: getattr(self, k) for k in self._STANDARD}
        data.update(thaw_mapping(self.additional))
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "loss": self.loss,
            "additional": thaw_mapping(self.additional),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentMetrics":
        return cls(
            accuracy=data.get("accuracy"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            f1=data.get("f1"),
            roc_auc=data.get("roc_auc"),
            pr_auc=data.get("pr_auc"),
            loss=data.get("loss"),
            additional=freeze_mapping(data.get("additional", {})),
        )


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Environment and provenance metadata captured for an experiment."""

    dataset_version: str = ""
    git_commit: str = ""
    random_seed: int = 0
    numpy_seed: int = 0
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    created_at: str = ""
    notes: str = ""
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "git_commit": self.git_commit,
            "random_seed": self.random_seed,
            "numpy_seed": self.numpy_seed,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "created_at": self.created_at,
            "notes": self.notes,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentMetadata":
        return cls(
            dataset_version=data.get("dataset_version", ""),
            git_commit=data.get("git_commit", ""),
            random_seed=int(data.get("random_seed", 0)),
            numpy_seed=int(data.get("numpy_seed", 0)),
            python_version=data.get("python_version", ""),
            platform=data.get("platform", ""),
            hostname=data.get("hostname", ""),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
            tags=_coerce_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class Experiment:
    """An experiment groups one or more runs under a shared objective."""

    experiment_id: str
    name: str
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    created_at: str = ""
    parent_experiment_id: Optional[str] = None
    run_ids: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.experiment_id, "experiment_id")
        _require_non_empty(self.name, "name")
        if not isinstance(self.status, ExperimentStatus):
            object.__setattr__(self, "status", ExperimentStatus(self.status))
        object.__setattr__(self, "run_ids", _coerce_str_tuple(self.run_ids))
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))

    def with_run(self, run_id: str) -> "Experiment":
        if run_id in self.run_ids:
            return self
        return replace(self, run_ids=self.run_ids + (run_id,))

    def with_status(self, status: ExperimentStatus) -> "Experiment":
        return replace(self, status=status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "parent_experiment_id": self.parent_experiment_id,
            "run_ids": list(self.run_ids),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Experiment":
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            status=ExperimentStatus(data.get("status", "CREATED")),
            metadata=ExperimentMetadata.from_dict(data.get("metadata", {})),
            created_at=data.get("created_at", ""),
            parent_experiment_id=data.get("parent_experiment_id"),
            run_ids=_coerce_str_tuple(data.get("run_ids", ())),
            tags=_coerce_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """A single training run with full metrics, resource usage and provenance."""

    run_id: str
    experiment_id: str
    status: ExperimentStatus = ExperimentStatus.CREATED
    parent_experiment_id: Optional[str] = None
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    training_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    validation_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    test_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    started_at: str = ""
    completed_at: Optional[str] = None
    dataset_version: str = ""
    git_commit: str = ""
    random_seed: int = 0
    numpy_seed: int = 0
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    notes: str = ""
    tags: Tuple[str, ...] = ()
    artifact_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.experiment_id, "experiment_id")
        if not isinstance(self.status, ExperimentStatus):
            object.__setattr__(self, "status", ExperimentStatus(self.status))
        for name in ("training_time", "inference_time", "memory_usage", "cpu_usage"):
            _require_non_negative(getattr(self, name), name)
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))
        object.__setattr__(self, "artifact_ids", _coerce_str_tuple(self.artifact_ids))

    def with_artifact(self, artifact_id: str) -> "ExperimentRun":
        if artifact_id in self.artifact_ids:
            return self
        return replace(self, artifact_ids=self.artifact_ids + (artifact_id,))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "parent_experiment_id": self.parent_experiment_id,
            "hyperparameters": self.hyperparameters.to_dict(),
            "training_metrics": self.training_metrics.to_dict(),
            "validation_metrics": self.validation_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dataset_version": self.dataset_version,
            "git_commit": self.git_commit,
            "random_seed": self.random_seed,
            "numpy_seed": self.numpy_seed,
            "python_version": self.python_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "notes": self.notes,
            "tags": list(self.tags),
            "artifact_ids": list(self.artifact_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            status=ExperimentStatus(data.get("status", "CREATED")),
            parent_experiment_id=data.get("parent_experiment_id"),
            hyperparameters=HyperParameters.from_dict(data.get("hyperparameters", {})),
            training_metrics=ExperimentMetrics.from_dict(data.get("training_metrics", {})),
            validation_metrics=ExperimentMetrics.from_dict(data.get("validation_metrics", {})),
            test_metrics=ExperimentMetrics.from_dict(data.get("test_metrics", {})),
            training_time=float(data.get("training_time", 0.0)),
            inference_time=float(data.get("inference_time", 0.0)),
            memory_usage=float(data.get("memory_usage", 0.0)),
            cpu_usage=float(data.get("cpu_usage", 0.0)),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            dataset_version=data.get("dataset_version", ""),
            git_commit=data.get("git_commit", ""),
            random_seed=int(data.get("random_seed", 0)),
            numpy_seed=int(data.get("numpy_seed", 0)),
            python_version=data.get("python_version", ""),
            platform=data.get("platform", ""),
            hostname=data.get("hostname", ""),
            notes=data.get("notes", ""),
            tags=_coerce_str_tuple(data.get("tags", ())),
            artifact_ids=_coerce_str_tuple(data.get("artifact_ids", ())),
        )


# --------------------------------------------------------------------------- #
# Artifact models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Metadata describing a tracked artifact (no binary payload is stored)."""

    artifact_id: str
    artifact_type: ArtifactType
    name: str
    content_hash: str = ""
    size_bytes: int = 0
    created_at: str = ""
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    description: str = ""
    properties: FrozenMap = ()
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_id, "artifact_id")
        _require_non_empty(self.name, "name")
        if not isinstance(self.artifact_type, ArtifactType):
            object.__setattr__(self, "artifact_type", ArtifactType(self.artifact_type))
        _require_non_negative(self.size_bytes, "size_bytes")
        object.__setattr__(self, "properties", freeze_mapping(self.properties))
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "description": self.description,
            "properties": thaw_mapping(self.properties),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactMetadata":
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            name=data["name"],
            content_hash=data.get("content_hash", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            created_at=data.get("created_at", ""),
            experiment_id=data.get("experiment_id"),
            run_id=data.get("run_id"),
            description=data.get("description", ""),
            properties=freeze_mapping(data.get("properties", {})),
            tags=_coerce_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A serialized-model artifact reference with its descriptive metadata."""

    artifact_id: str
    name: str
    framework: str = ""
    format: str = ""
    uri: str = ""
    content_hash: str = ""
    size_bytes: int = 0
    metadata: Optional[ArtifactMetadata] = None

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_id, "artifact_id")
        _require_non_empty(self.name, "name")
        _require_non_negative(self.size_bytes, "size_bytes")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "framework": self.framework,
            "format": self.format,
            "uri": self.uri,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata.to_dict() if self.metadata is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelArtifact":
        meta = data.get("metadata")
        return cls(
            artifact_id=data["artifact_id"],
            name=data["name"],
            framework=data.get("framework", ""),
            format=data.get("format", ""),
            uri=data.get("uri", ""),
            content_hash=data.get("content_hash", ""),
            size_bytes=int(data.get("size_bytes", 0)),
            metadata=ArtifactMetadata.from_dict(meta) if meta else None,
        )


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """An immutable, content-addressed dataset version descriptor."""

    dataset_id: str
    name: str
    version: str
    content_hash: str = ""
    row_count: int = 0
    feature_count: int = 0
    created_at: str = ""
    description: str = ""
    schema_hash: str = ""
    source_uri: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.dataset_id, "dataset_id")
        _require_non_empty(self.name, "name")
        if not SemanticVersion.is_valid(self.version):
            raise ValidationError(f"DatasetVersion.version is not valid semver: {self.version!r}")
        _require_non_negative(self.row_count, "row_count")
        _require_non_negative(self.feature_count, "feature_count")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "created_at": self.created_at,
            "description": self.description,
            "schema_hash": self.schema_hash,
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetVersion":
        return cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            version=data["version"],
            content_hash=data.get("content_hash", ""),
            row_count=int(data.get("row_count", 0)),
            feature_count=int(data.get("feature_count", 0)),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            schema_hash=data.get("schema_hash", ""),
            source_uri=data.get("source_uri", ""),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """A deterministic snapshot of a runtime environment."""

    python_version: str = ""
    numpy_version: str = ""
    platform: str = ""
    hostname: str = ""
    processor: str = ""
    cpu_count: int = 0
    dependencies: FrozenMap = ()
    environment_variables: FrozenMap = ()
    captured_at: str = ""

    def __post_init__(self) -> None:
        _require_non_negative(self.cpu_count, "cpu_count")
        object.__setattr__(self, "dependencies", freeze_mapping(self.dependencies))
        object.__setattr__(
            self, "environment_variables", freeze_mapping(self.environment_variables)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "dependencies": thaw_mapping(self.dependencies),
            "environment_variables": thaw_mapping(self.environment_variables),
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentSnapshot":
        return cls(
            python_version=data.get("python_version", ""),
            numpy_version=data.get("numpy_version", ""),
            platform=data.get("platform", ""),
            hostname=data.get("hostname", ""),
            processor=data.get("processor", ""),
            cpu_count=int(data.get("cpu_count", 0)),
            dependencies=freeze_mapping(data.get("dependencies", {})),
            environment_variables=freeze_mapping(data.get("environment_variables", {})),
            captured_at=data.get("captured_at", ""),
        )


@dataclass(frozen=True, slots=True)
class ModelCard:
    """A complete, human-readable model card for responsible deployment."""

    model_id: str
    model_name: str
    version: str
    purpose: str = ""
    training_data: str = ""
    evaluation: str = ""
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    limitations: Tuple[str, ...] = ()
    responsible_ai_notes: Tuple[str, ...] = ()
    deployment_stage: ModelStage = ModelStage.REGISTERED
    dependencies: FrozenMap = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.model_name, "model_name")
        if not SemanticVersion.is_valid(self.version):
            raise ValidationError(f"ModelCard.version is not valid semver: {self.version!r}")
        if not isinstance(self.deployment_stage, ModelStage):
            object.__setattr__(self, "deployment_stage", ModelStage(self.deployment_stage))
        object.__setattr__(self, "limitations", _coerce_str_tuple(self.limitations))
        object.__setattr__(
            self, "responsible_ai_notes", _coerce_str_tuple(self.responsible_ai_notes)
        )
        object.__setattr__(self, "dependencies", freeze_mapping(self.dependencies))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "purpose": self.purpose,
            "training_data": self.training_data,
            "evaluation": self.evaluation,
            "metrics": self.metrics.to_dict(),
            "limitations": list(self.limitations),
            "responsible_ai_notes": list(self.responsible_ai_notes),
            "deployment_stage": self.deployment_stage.value,
            "dependencies": thaw_mapping(self.dependencies),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCard":
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            version=data["version"],
            purpose=data.get("purpose", ""),
            training_data=data.get("training_data", ""),
            evaluation=data.get("evaluation", ""),
            metrics=ExperimentMetrics.from_dict(data.get("metrics", {})),
            limitations=_coerce_str_tuple(data.get("limitations", ())),
            responsible_ai_notes=_coerce_str_tuple(data.get("responsible_ai_notes", ())),
            deployment_stage=ModelStage(data.get("deployment_stage", "REGISTERED")),
            dependencies=freeze_mapping(data.get("dependencies", {})),
            created_at=data.get("created_at", ""),
        )


# --------------------------------------------------------------------------- #
# Registry models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ModelVersion:
    """A single, immutable version of a registered model with full lineage."""

    model_id: str
    version: str
    stage: ModelStage = ModelStage.REGISTERED
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    dataset_version: str = ""
    artifact_ids: Tuple[str, ...] = ()
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    model_size_bytes: int = 0
    training_time_s: float = 0.0
    inference_time_ms: float = 0.0
    validation_passed: bool = False
    created_at: str = ""
    promoted_at: Optional[str] = None
    description: str = ""
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        if not SemanticVersion.is_valid(self.version):
            raise ValidationError(f"ModelVersion.version is not valid semver: {self.version!r}")
        if not isinstance(self.stage, ModelStage):
            object.__setattr__(self, "stage", ModelStage(self.stage))
        for name in ("latency_ms", "memory_mb", "model_size_bytes",
                     "training_time_s", "inference_time_ms"):
            _require_non_negative(getattr(self, name), name)
        object.__setattr__(self, "artifact_ids", _coerce_str_tuple(self.artifact_ids))
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))

    @property
    def semantic_version(self) -> SemanticVersion:
        return SemanticVersion.parse(self.version)

    def lineage(self) -> Dict[str, Any]:
        """Return the complete provenance chain for this version."""
        return {
            "dataset_version": self.dataset_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "artifact_ids": list(self.artifact_ids),
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage.value,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "artifact_ids": list(self.artifact_ids),
            "metrics": self.metrics.to_dict(),
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "model_size_bytes": self.model_size_bytes,
            "training_time_s": self.training_time_s,
            "inference_time_ms": self.inference_time_ms,
            "validation_passed": self.validation_passed,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "description": self.description,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelVersion":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            stage=ModelStage(data.get("stage", "REGISTERED")),
            experiment_id=data.get("experiment_id"),
            run_id=data.get("run_id"),
            dataset_version=data.get("dataset_version", ""),
            artifact_ids=_coerce_str_tuple(data.get("artifact_ids", ())),
            metrics=ExperimentMetrics.from_dict(data.get("metrics", {})),
            latency_ms=float(data.get("latency_ms", 0.0)),
            memory_mb=float(data.get("memory_mb", 0.0)),
            model_size_bytes=int(data.get("model_size_bytes", 0)),
            training_time_s=float(data.get("training_time_s", 0.0)),
            inference_time_ms=float(data.get("inference_time_ms", 0.0)),
            validation_passed=bool(data.get("validation_passed", False)),
            created_at=data.get("created_at", ""),
            promoted_at=data.get("promoted_at"),
            description=data.get("description", ""),
            tags=_coerce_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """A registered model aggregate that owns an ordered set of versions."""

    model_id: str
    name: str
    description: str = ""
    created_at: str = ""
    version_ids: Tuple[str, ...] = ()
    current_production_version: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.name, "name")
        object.__setattr__(self, "version_ids", _coerce_str_tuple(self.version_ids))
        object.__setattr__(self, "tags", _coerce_str_tuple(self.tags))

    def with_version(self, version: str) -> "RegisteredModel":
        if version in self.version_ids:
            return self
        return replace(self, version_ids=self.version_ids + (version,))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "version_ids": list(self.version_ids),
            "current_production_version": self.current_production_version,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegisteredModel":
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            version_ids=_coerce_str_tuple(data.get("version_ids", ())),
            current_production_version=data.get("current_production_version"),
            tags=_coerce_str_tuple(data.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    """A request to advance a model version to a new lifecycle stage."""

    model_id: str
    version: str
    from_stage: ModelStage
    to_stage: ModelStage
    requested_by: str = ""
    approved_by: Optional[str] = None
    reason: str = ""
    requested_at: str = ""
    baseline_version: Optional[str] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    require_metric_improvement: bool = True
    primary_metric: str = "f1"
    primary_metric_direction: MetricDirection = MetricDirection.MAXIMIZE

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        if not SemanticVersion.is_valid(self.version):
            raise ValidationError(f"PromotionRequest.version invalid semver: {self.version!r}")
        if not isinstance(self.from_stage, ModelStage):
            object.__setattr__(self, "from_stage", ModelStage(self.from_stage))
        if not isinstance(self.to_stage, ModelStage):
            object.__setattr__(self, "to_stage", ModelStage(self.to_stage))
        if not isinstance(self.primary_metric_direction, MetricDirection):
            object.__setattr__(
                self, "primary_metric_direction",
                MetricDirection(self.primary_metric_direction),
            )

    @property
    def is_approved(self) -> bool:
        return bool(self.approved_by and str(self.approved_by).strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "requested_by": self.requested_by,
            "approved_by": self.approved_by,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "baseline_version": self.baseline_version,
            "max_latency_ms": self.max_latency_ms,
            "max_memory_mb": self.max_memory_mb,
            "require_metric_improvement": self.require_metric_improvement,
            "primary_metric": self.primary_metric,
            "primary_metric_direction": self.primary_metric_direction.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromotionRequest":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            from_stage=ModelStage(data["from_stage"]),
            to_stage=ModelStage(data["to_stage"]),
            requested_by=data.get("requested_by", ""),
            approved_by=data.get("approved_by"),
            reason=data.get("reason", ""),
            requested_at=data.get("requested_at", ""),
            baseline_version=data.get("baseline_version"),
            max_latency_ms=data.get("max_latency_ms"),
            max_memory_mb=data.get("max_memory_mb"),
            require_metric_improvement=bool(data.get("require_metric_improvement", True)),
            primary_metric=data.get("primary_metric", "f1"),
            primary_metric_direction=MetricDirection(
                data.get("primary_metric_direction", "MAXIMIZE")
            ),
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The deterministic outcome of evaluating a promotion request."""

    model_id: str
    version: str
    from_stage: ModelStage
    to_stage: ModelStage
    status: PromotionStatus
    reasons: Tuple[str, ...] = ()
    checks: FrozenMap = ()
    decided_at: str = ""
    decided_by: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.from_stage, ModelStage):
            object.__setattr__(self, "from_stage", ModelStage(self.from_stage))
        if not isinstance(self.to_stage, ModelStage):
            object.__setattr__(self, "to_stage", ModelStage(self.to_stage))
        if not isinstance(self.status, PromotionStatus):
            object.__setattr__(self, "status", PromotionStatus(self.status))
        object.__setattr__(self, "reasons", _coerce_str_tuple(self.reasons))
        object.__setattr__(self, "checks", freeze_mapping(self.checks))

    @property
    def approved(self) -> bool:
        return self.status is PromotionStatus.APPROVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "checks": thaw_mapping(self.checks),
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromotionDecision":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            from_stage=ModelStage(data["from_stage"]),
            to_stage=ModelStage(data["to_stage"]),
            status=PromotionStatus(data["status"]),
            reasons=_coerce_str_tuple(data.get("reasons", ())),
            checks=freeze_mapping(data.get("checks", {})),
            decided_at=data.get("decided_at", ""),
            decided_by=data.get("decided_by", ""),
        )


# --------------------------------------------------------------------------- #
# Statistics & reproducibility
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RegistryStatistics:
    """Aggregate statistics computed over a model registry."""

    total_models: int = 0
    total_versions: int = 0
    versions_by_stage: FrozenMap = ()
    production_models: int = 0
    archived_versions: int = 0
    generated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions_by_stage", freeze_mapping(self.versions_by_stage))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_models": self.total_models,
            "total_versions": self.total_versions,
            "versions_by_stage": thaw_mapping(self.versions_by_stage),
            "production_models": self.production_models,
            "archived_versions": self.archived_versions,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryStatistics":
        return cls(
            total_models=int(data.get("total_models", 0)),
            total_versions=int(data.get("total_versions", 0)),
            versions_by_stage=freeze_mapping(data.get("versions_by_stage", {})),
            production_models=int(data.get("production_models", 0)),
            archived_versions=int(data.get("archived_versions", 0)),
            generated_at=data.get("generated_at", ""),
        )


@dataclass(frozen=True, slots=True)
class ExperimentStatistics:
    """Aggregate statistics computed over tracked experiments and runs."""

    total_experiments: int = 0
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    running_runs: int = 0
    average_training_time: float = 0.0
    best_metric_value: Optional[float] = None
    best_run_id: Optional[str] = None
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "running_runs": self.running_runs,
            "average_training_time": self.average_training_time,
            "best_metric_value": self.best_metric_value,
            "best_run_id": self.best_run_id,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentStatistics":
        return cls(
            total_experiments=int(data.get("total_experiments", 0)),
            total_runs=int(data.get("total_runs", 0)),
            completed_runs=int(data.get("completed_runs", 0)),
            failed_runs=int(data.get("failed_runs", 0)),
            running_runs=int(data.get("running_runs", 0)),
            average_training_time=float(data.get("average_training_time", 0.0)),
            best_metric_value=data.get("best_metric_value"),
            best_run_id=data.get("best_run_id"),
            generated_at=data.get("generated_at", ""),
        )


@dataclass(frozen=True, slots=True)
class ReproducibilitySnapshot:
    """A complete, deterministic snapshot enabling exact run reproduction."""

    snapshot_id: str
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    random_seed: int = 0
    numpy_seed: int = 0
    config: FrozenMap = ()
    dataset_version: str = ""
    git_commit: str = ""
    runtime_info: FrozenMap = ()
    hardware_info: FrozenMap = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.snapshot_id, "snapshot_id")
        object.__setattr__(self, "config", freeze_mapping(self.config))
        object.__setattr__(self, "runtime_info", freeze_mapping(self.runtime_info))
        object.__setattr__(self, "hardware_info", freeze_mapping(self.hardware_info))

    def diff(self, other: "ReproducibilitySnapshot") -> Dict[str, Any]:
        """Return a deterministic mapping of fields that differ from *other*."""
        differences: Dict[str, Any] = {}
        left, right = self.to_dict(), other.to_dict()
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                differences[key] = {"self": left.get(key), "other": right.get(key)}
        return differences

    def matches(self, other: "ReproducibilitySnapshot") -> bool:
        """Return ``True`` if the two snapshots are reproducibly identical."""
        ignore = {"snapshot_id", "created_at"}
        left = {k: v for k, v in self.to_dict().items() if k not in ignore}
        right = {k: v for k, v in other.to_dict().items() if k not in ignore}
        env_ignore = {"captured_at"}
        left["environment"] = {k: v for k, v in left["environment"].items() if k not in env_ignore}
        right["environment"] = {k: v for k, v in right["environment"].items() if k not in env_ignore}
        return left == right

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "environment": self.environment.to_dict(),
            "random_seed": self.random_seed,
            "numpy_seed": self.numpy_seed,
            "config": thaw_mapping(self.config),
            "dataset_version": self.dataset_version,
            "git_commit": self.git_commit,
            "runtime_info": thaw_mapping(self.runtime_info),
            "hardware_info": thaw_mapping(self.hardware_info),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReproducibilitySnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            environment=EnvironmentSnapshot.from_dict(data.get("environment", {})),
            random_seed=int(data.get("random_seed", 0)),
            numpy_seed=int(data.get("numpy_seed", 0)),
            config=freeze_mapping(data.get("config", {})),
            dataset_version=data.get("dataset_version", ""),
            git_commit=data.get("git_commit", ""),
            runtime_info=freeze_mapping(data.get("runtime_info", {})),
            hardware_info=freeze_mapping(data.get("hardware_info", {})),
            created_at=data.get("created_at", ""),
        )


# --------------------------------------------------------------------------- #
# Deterministic infrastructure: clocks and identifier generators
# --------------------------------------------------------------------------- #
import hashlib
import itertools
import threading
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

__all__ += [
    "Clock",
    "SystemClock",
    "FixedClock",
    "LogicalClock",
    "IdGenerator",
    "SequentialIdGenerator",
    "DeterministicIdGenerator",
]

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@runtime_checkable
class Clock(Protocol):
    """Abstraction over time so all components remain reproducible."""

    def now(self) -> str:
        """Return the current timestamp as an ISO-8601 string."""
        ...


class SystemClock:
    """A wall-clock implementation for production use."""

    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class FixedClock:
    """A clock that always returns the same instant (useful for tests)."""

    def __init__(self, instant: str = "2024-01-01T00:00:00+00:00") -> None:
        self._instant = instant

    def now(self) -> str:
        return self._instant


class LogicalClock:
    """A monotonic logical clock yielding deterministic, ordered timestamps."""

    def __init__(self, start: int = 0, step_seconds: int = 1) -> None:
        if step_seconds <= 0:
            raise ValidationError("LogicalClock.step_seconds must be positive")
        self._counter = itertools.count(start)
        self._step = step_seconds
        self._lock = threading.Lock()

    def now(self) -> str:
        with self._lock:
            tick = next(self._counter)
        return (_EPOCH + timedelta(seconds=tick * self._step)).isoformat()


@runtime_checkable
class IdGenerator(Protocol):
    """Abstraction over identifier creation for reproducible runs."""

    def generate(self, prefix: str) -> str:
        """Return a new identifier carrying the given semantic prefix."""
        ...


class SequentialIdGenerator:
    """Thread-safe, monotonically increasing identifiers (e.g. ``exp-000001``)."""

    def __init__(self, width: int = 6, start: int = 1) -> None:
        if width <= 0:
            raise ValidationError("SequentialIdGenerator.width must be positive")
        self._counters: Dict[str, "itertools.count[int]"] = {}
        self._width = width
        self._start = start
        self._lock = threading.Lock()

    def generate(self, prefix: str) -> str:
        with self._lock:
            counter = self._counters.get(prefix)
            if counter is None:
                counter = itertools.count(self._start)
                self._counters[prefix] = counter
            value = next(counter)
        return f"{prefix}-{value:0{self._width}d}"


class DeterministicIdGenerator:
    """Content-stable identifiers derived from a seed and per-prefix counter.

    Given the same seed and call sequence, identifiers are byte-for-byte
    reproducible across processes and platforms.
    """

    def __init__(self, seed: str = "mlops", length: int = 12) -> None:
        if length <= 0 or length > 64:
            raise ValidationError("DeterministicIdGenerator.length must be in 1..64")
        self._seed = seed
        self._length = length
        self._counters: Dict[str, "itertools.count[int]"] = {}
        self._lock = threading.Lock()

    def generate(self, prefix: str) -> str:
        with self._lock:
            counter = self._counters.get(prefix)
            if counter is None:
                counter = itertools.count(0)
                self._counters[prefix] = counter
            value = next(counter)
        digest = hashlib.sha256(f"{self._seed}:{prefix}:{value}".encode("utf-8")).hexdigest()
        return f"{prefix}-{digest[: self._length]}"