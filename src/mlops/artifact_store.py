"""Deterministic, metadata-only artifact store for the MLOps subsystem.

The artifact store records *metadata* about artifacts (serialized models,
evaluation reports, engineering reports, publication figures, feature
importance, SHAP results, confusion matrices, ROC / PR curves and calibration
plots). It deliberately stores no binary payloads; instead every artifact is
content-addressed by a deterministic SHA-256 digest of its descriptive
properties, which is sufficient for lineage, reproducibility and registry
integration.

The store is thread-safe, dependency-injected (clock, id generator, hashing
strategy) and fully JSON-serialisable.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from mlops.experiment_models import (
    ArtifactMetadata,
    ArtifactType,
    Clock,
    DeterministicIdGenerator,
    IdGenerator,
    LogicalClock,
    MLOpsError,
    ModelArtifact,
    SerializationError,
    ValidationError,
)

__all__ = [
    "ArtifactStoreError",
    "ArtifactNotFoundError",
    "DuplicateArtifactError",
    "HashStrategy",
    "Sha256HashStrategy",
    "ArtifactStore",
    "create_artifact_store",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ArtifactStoreError(MLOpsError):
    """Base error for the artifact store."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when a requested artifact does not exist."""


class DuplicateArtifactError(ArtifactStoreError):
    """Raised when registering an artifact whose id already exists."""


# --------------------------------------------------------------------------- #
# Hashing strategy (Strategy pattern)
# --------------------------------------------------------------------------- #
class HashStrategy(Protocol):
    """Abstraction over content-hash computation."""

    def compute(self, payload: Mapping[str, Any]) -> str:
        """Return a deterministic hash for the given metadata payload."""
        ...


class Sha256HashStrategy:
    """Deterministic SHA-256 over canonicalised JSON."""

    def compute(self, payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Artifact store
# --------------------------------------------------------------------------- #
class ArtifactStore:
    """A thread-safe, in-memory registry of artifact metadata."""

    def __init__(
        self,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        hash_strategy: Optional[HashStrategy] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="artifact")
        self._hasher: HashStrategy = hash_strategy or Sha256HashStrategy()
        self._artifacts: Dict[str, ArtifactMetadata] = {}
        self._lock = threading.RLock()

    # -- creation ----------------------------------------------------------- #
    def create_artifact(
        self,
        artifact_type: ArtifactType,
        name: str,
        *,
        artifact_id: Optional[str] = None,
        size_bytes: int = 0,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        description: str = "",
        properties: Optional[Mapping[str, Any]] = None,
        tags: Optional[Tuple[str, ...]] = None,
    ) -> ArtifactMetadata:
        """Create and register a new artifact, returning its metadata."""
        if not isinstance(artifact_type, ArtifactType):
            artifact_type = ArtifactType(artifact_type)
        props = dict(properties or {})
        with self._lock:
            aid = artifact_id or self._ids.generate("artifact")
            if aid in self._artifacts:
                raise DuplicateArtifactError(f"Artifact id already exists: {aid!r}")
            content_hash = self._hasher.compute(
                {
                    "artifact_type": artifact_type.value,
                    "name": name,
                    "size_bytes": size_bytes,
                    "description": description,
                    "properties": props,
                }
            )
            metadata = ArtifactMetadata(
                artifact_id=aid,
                artifact_type=artifact_type,
                name=name,
                content_hash=content_hash,
                size_bytes=size_bytes,
                created_at=self._clock.now(),
                experiment_id=experiment_id,
                run_id=run_id,
                description=description,
                properties=props,
                tags=tags or (),
            )
            self._artifacts[aid] = metadata
            return metadata

    def register(self, metadata: ArtifactMetadata) -> ArtifactMetadata:
        """Register a pre-built :class:`ArtifactMetadata` instance."""
        if not isinstance(metadata, ArtifactMetadata):
            raise ValidationError("register() expects an ArtifactMetadata instance")
        with self._lock:
            if metadata.artifact_id in self._artifacts:
                raise DuplicateArtifactError(
                    f"Artifact id already exists: {metadata.artifact_id!r}"
                )
            self._artifacts[metadata.artifact_id] = metadata
            return metadata

    def register_model_artifact(self, artifact: ModelArtifact) -> ArtifactMetadata:
        """Register a :class:`ModelArtifact`, deriving metadata when absent."""
        if not isinstance(artifact, ModelArtifact):
            raise ValidationError("register_model_artifact() expects a ModelArtifact")
        metadata = artifact.metadata
        if metadata is None:
            with self._lock:
                content_hash = artifact.content_hash or self._hasher.compute(
                    {
                        "name": artifact.name,
                        "framework": artifact.framework,
                        "format": artifact.format,
                        "uri": artifact.uri,
                        "size_bytes": artifact.size_bytes,
                    }
                )
                metadata = ArtifactMetadata(
                    artifact_id=artifact.artifact_id,
                    artifact_type=ArtifactType.SERIALIZED_MODEL,
                    name=artifact.name,
                    content_hash=content_hash,
                    size_bytes=artifact.size_bytes,
                    created_at=self._clock.now(),
                    properties={
                        "framework": artifact.framework,
                        "format": artifact.format,
                        "uri": artifact.uri,
                    },
                )
        return self.register(metadata)

    # -- retrieval ---------------------------------------------------------- #
    def get(self, artifact_id: str) -> ArtifactMetadata:
        """Return the metadata for *artifact_id* or raise if absent."""
        with self._lock:
            try:
                return self._artifacts[artifact_id]
            except KeyError as exc:
                raise ArtifactNotFoundError(f"Unknown artifact: {artifact_id!r}") from exc

    def exists(self, artifact_id: str) -> bool:
        with self._lock:
            return artifact_id in self._artifacts

    def list_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[ArtifactMetadata, ...]:
        """Return artifacts matching the given filters, ordered by id."""
        with self._lock:
            items = list(self._artifacts.values())
        if artifact_type is not None:
            atype = ArtifactType(artifact_type)
            items = [a for a in items if a.artifact_type is atype]
        if experiment_id is not None:
            items = [a for a in items if a.experiment_id == experiment_id]
        if run_id is not None:
            items = [a for a in items if a.run_id == run_id]
        return tuple(sorted(items, key=lambda a: a.artifact_id))

    def by_type(self, artifact_type: ArtifactType) -> Tuple[ArtifactMetadata, ...]:
        return self.list_artifacts(artifact_type=artifact_type)

    def find_by_content_hash(self, content_hash: str) -> Tuple[ArtifactMetadata, ...]:
        with self._lock:
            items = [a for a in self._artifacts.values() if a.content_hash == content_hash]
        return tuple(sorted(items, key=lambda a: a.artifact_id))

    def count(self) -> int:
        with self._lock:
            return len(self._artifacts)

    def artifact_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._artifacts))

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "artifacts": [
                    self._artifacts[k].to_dict() for k in sorted(self._artifacts)
                ]
            }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def export_json(self, path: str, *, indent: int = 2) -> str:
        """Write the store to *path* as JSON and return the path."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(indent=indent))
        return path

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        hash_strategy: Optional[HashStrategy] = None,
    ) -> "ArtifactStore":
        store = cls(clock=clock, id_generator=id_generator, hash_strategy=hash_strategy)
        try:
            for entry in data.get("artifacts", []):
                store.register(ArtifactMetadata.from_dict(entry))
        except (KeyError, TypeError) as exc:
            raise SerializationError(f"Cannot deserialise ArtifactStore: {exc}") from exc
        return store


def create_artifact_store(
    *,
    deterministic: bool = True,
    seed: str = "artifact",
) -> ArtifactStore:
    """Factory returning a configured :class:`ArtifactStore`."""
    if deterministic:
        return ArtifactStore(
            clock=LogicalClock(),
            id_generator=DeterministicIdGenerator(seed=seed),
            hash_strategy=Sha256HashStrategy(),
        )
    from mlops.experiment_models import SequentialIdGenerator, SystemClock

    return ArtifactStore(
        clock=SystemClock(),
        id_generator=SequentialIdGenerator(),
        hash_strategy=Sha256HashStrategy(),
    )