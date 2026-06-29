"""Enterprise model registry with a deterministic lifecycle.

Lifecycle state machine::

    REGISTERED -> VALIDATION -> STAGING -> PRODUCTION -> ARCHIVED

The registry provides registration with duplicate detection and semantic
version validation, a policy-driven promotion engine (Strategy pattern), a
deterministic model-comparison engine, complete lineage tracking, model-card
generation and JSON export. It is thread-safe and dependency-injected.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

from mlops.experiment_models import (
    Clock,
    DeterministicIdGenerator,
    ExperimentMetrics,
    IdGenerator,
    LogicalClock,
    MLOpsError,
    MetricDirection,
    ModelCard,
    ModelStage,
    ModelVersion,
    PromotionDecision,
    PromotionRequest,
    PromotionStatus,
    RegisteredModel,
    RegistryStatistics,
    SemanticVersion,
    SerializationError,
    ValidationError,
)

__all__ = [
    "RegistryError",
    "ModelNotFoundError",
    "VersionNotFoundError",
    "DuplicateModelVersionError",
    "InvalidStageTransitionError",
    "PromotionPolicy",
    "DefaultPromotionPolicy",
    "ModelComparator",
    "WeightedScoreComparator",
    "ModelRegistry",
    "create_model_registry",
    "ALLOWED_TRANSITIONS",
]

# Forward lifecycle transitions.
_FORWARD: Dict[ModelStage, ModelStage] = {
    ModelStage.REGISTERED: ModelStage.VALIDATION,
    ModelStage.VALIDATION: ModelStage.STAGING,
    ModelStage.STAGING: ModelStage.PRODUCTION,
}

# Every stage except ARCHIVED may be archived.
ALLOWED_TRANSITIONS: Dict[ModelStage, Tuple[ModelStage, ...]] = {
    ModelStage.REGISTERED: (ModelStage.VALIDATION, ModelStage.ARCHIVED),
    ModelStage.VALIDATION: (ModelStage.STAGING, ModelStage.ARCHIVED),
    ModelStage.STAGING: (ModelStage.PRODUCTION, ModelStage.ARCHIVED),
    ModelStage.PRODUCTION: (ModelStage.ARCHIVED,),
    ModelStage.ARCHIVED: (),
}

_QUALITY_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class RegistryError(MLOpsError):
    """Base error for the model registry."""


class ModelNotFoundError(RegistryError):
    """Raised when a model id is unknown."""


class VersionNotFoundError(RegistryError):
    """Raised when a (model, version) pair is unknown."""


class DuplicateModelVersionError(RegistryError):
    """Raised when registering a version that already exists."""


class InvalidStageTransitionError(RegistryError):
    """Raised when a requested lifecycle transition is not allowed."""


# --------------------------------------------------------------------------- #
# Promotion policy (Strategy pattern)
# --------------------------------------------------------------------------- #
class PromotionPolicy(Protocol):
    """Abstraction over promotion-gate evaluation."""

    def evaluate(
        self,
        request: PromotionRequest,
        candidate: ModelVersion,
        baseline: Optional[ModelVersion],
        decided_at: str,
    ) -> PromotionDecision:
        """Return a deterministic promotion decision."""
        ...


class DefaultPromotionPolicy:
    """The default enterprise promotion policy.

    A promotion is APPROVED only if every applicable gate passes:

    * the lifecycle transition is structurally valid;
    * validation is complete when entering STAGING or PRODUCTION;
    * an approver is recorded when entering STAGING or PRODUCTION;
    * the primary metric is no worse than the baseline (when required);
    * latency is within the configured budget;
    * memory is within the configured budget.
    """

    def evaluate(
        self,
        request: PromotionRequest,
        candidate: ModelVersion,
        baseline: Optional[ModelVersion],
        decided_at: str,
    ) -> PromotionDecision:
        checks: Dict[str, bool] = {}
        reasons: List[str] = []

        gated_stage = request.to_stage in (ModelStage.STAGING, ModelStage.PRODUCTION)

        # Transition validity.
        transition_ok = request.to_stage in ALLOWED_TRANSITIONS.get(request.from_stage, ())
        checks["transition_valid"] = transition_ok
        if not transition_ok:
            reasons.append(
                f"Invalid transition {request.from_stage.value} -> {request.to_stage.value}"
            )

        # Validation complete.
        if gated_stage:
            validation_ok = bool(candidate.validation_passed)
            checks["validation_complete"] = validation_ok
            if not validation_ok:
                reasons.append("Validation has not been completed for this version")

            approval_ok = request.is_approved
            checks["approval_exists"] = approval_ok
            if not approval_ok:
                reasons.append("No approver recorded for the promotion request")

        # Metric improvement.
        if request.require_metric_improvement and baseline is not None:
            improved = self._metric_not_worse(request, candidate, baseline)
            checks["metric_improved"] = improved
            if not improved:
                reasons.append(
                    f"Primary metric {request.primary_metric!r} did not improve over baseline"
                )

        # Latency budget.
        if request.max_latency_ms is not None:
            latency_ok = candidate.latency_ms <= request.max_latency_ms
            checks["latency_acceptable"] = latency_ok
            if not latency_ok:
                reasons.append(
                    f"Latency {candidate.latency_ms}ms exceeds budget {request.max_latency_ms}ms"
                )

        # Memory budget.
        if request.max_memory_mb is not None:
            memory_ok = candidate.memory_mb <= request.max_memory_mb
            checks["memory_acceptable"] = memory_ok
            if not memory_ok:
                reasons.append(
                    f"Memory {candidate.memory_mb}MB exceeds budget {request.max_memory_mb}MB"
                )

        approved = all(checks.values()) if checks else transition_ok
        status = PromotionStatus.APPROVED if approved else PromotionStatus.REJECTED
        if approved and not reasons:
            reasons.append("All promotion gates satisfied")

        return PromotionDecision(
            model_id=request.model_id,
            version=request.version,
            from_stage=request.from_stage,
            to_stage=request.to_stage,
            status=status,
            reasons=tuple(reasons),
            checks=checks,
            decided_at=decided_at,
            decided_by=request.approved_by or request.requested_by,
        )

    @staticmethod
    def _metric_not_worse(
        request: PromotionRequest, candidate: ModelVersion, baseline: ModelVersion
    ) -> bool:
        cand = candidate.metrics.get(request.primary_metric)
        base = baseline.metrics.get(request.primary_metric)
        if cand is None or base is None:
            return False
        if request.primary_metric_direction is MetricDirection.MAXIMIZE:
            return float(cand) >= float(base)
        return float(cand) <= float(base)


# --------------------------------------------------------------------------- #
# Comparison engine (Strategy pattern)
# --------------------------------------------------------------------------- #
class ModelComparator(Protocol):
    """Abstraction over deterministic multi-metric version comparison."""

    def score(self, version: ModelVersion) -> float:
        """Return a scalar score (higher is better)."""
        ...


class WeightedScoreComparator:
    """Deterministic weighted scoring across quality and cost dimensions.

    Quality metrics contribute positively; cost dimensions (latency, memory,
    model size, inference time) contribute negatively after min-max
    normalisation over the compared set. The comparator is pure and order
    independent.
    """

    def __init__(
        self,
        quality_weights: Optional[Mapping[str, float]] = None,
        cost_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._quality = dict(quality_weights or {"f1": 0.5, "roc_auc": 0.3, "accuracy": 0.2})
        self._cost = dict(
            cost_weights
            or {"latency_ms": 0.4, "memory_mb": 0.3, "inference_time_ms": 0.2, "model_size_bytes": 0.1}
        )

    def score(self, version: ModelVersion) -> float:
        quality = 0.0
        for name, weight in sorted(self._quality.items()):
            value = version.metrics.get(name)
            if value is not None:
                quality += weight * float(value)
        return quality

    def rank(self, versions: List[ModelVersion]) -> List[Tuple[ModelVersion, float]]:
        """Return ``(version, composite_score)`` pairs sorted best-first."""
        if not versions:
            return []
        cost_values: Dict[str, List[float]] = {name: [] for name in self._cost}
        for name in self._cost:
            for v in versions:
                cost_values[name].append(float(getattr(v, name)))

        def composite(v: ModelVersion) -> float:
            score = self.score(v)
            for name, weight in sorted(self._cost.items()):
                values = cost_values[name]
                lo, hi = min(values), max(values)
                if hi > lo:
                    normalised = (float(getattr(v, name)) - lo) / (hi - lo)
                    score -= weight * normalised
            return score

        scored = [(v, composite(v)) for v in versions]
        # Deterministic ordering: score desc, then semantic version desc, then version string.
        scored.sort(key=lambda pair: (-pair[1], _neg_semver_key(pair[0].version), pair[0].version))
        return scored


def _neg_semver_key(version: str) -> Tuple[int, int, int, int]:
    sv = SemanticVersion.parse(version)
    has_release = 1 if sv.prerelease is None else 0
    return (-sv.major, -sv.minor, -sv.patch, -has_release)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelRegistry:
    """A thread-safe registry of models and their immutable versions."""

    def __init__(
        self,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        promotion_policy: Optional[PromotionPolicy] = None,
        comparator: Optional[ModelComparator] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="model")
        self._policy: PromotionPolicy = promotion_policy or DefaultPromotionPolicy()
        self._comparator = comparator or WeightedScoreComparator()
        self._models: Dict[str, RegisteredModel] = {}
        self._versions: Dict[Tuple[str, str], ModelVersion] = {}
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------- #
    def register_model(
        self,
        name: str,
        version: str,
        *,
        model_id: Optional[str] = None,
        metrics: Optional[ExperimentMetrics] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        dataset_version: str = "",
        artifact_ids: Tuple[str, ...] = (),
        latency_ms: float = 0.0,
        memory_mb: float = 0.0,
        model_size_bytes: int = 0,
        training_time_s: float = 0.0,
        inference_time_ms: float = 0.0,
        validation_passed: bool = False,
        description: str = "",
        tags: Tuple[str, ...] = (),
    ) -> ModelVersion:
        """Register a new model version, creating the model if needed."""
        if not SemanticVersion.is_valid(version):
            raise ValidationError(f"Invalid semantic version: {version!r}")
        with self._lock:
            mid = model_id or self._stable_model_id(name)
            now = self._clock.now()
            model = self._models.get(mid)
            if model is None:
                model = RegisteredModel(model_id=mid, name=name, created_at=now)
            key = (mid, version)
            if key in self._versions:
                raise DuplicateModelVersionError(
                    f"Version {version} already exists for model {mid}"
                )
            mv = ModelVersion(
                model_id=mid,
                version=version,
                stage=ModelStage.REGISTERED,
                experiment_id=experiment_id,
                run_id=run_id,
                dataset_version=dataset_version,
                artifact_ids=artifact_ids,
                metrics=metrics or ExperimentMetrics(),
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                model_size_bytes=model_size_bytes,
                training_time_s=training_time_s,
                inference_time_ms=inference_time_ms,
                validation_passed=validation_passed,
                created_at=now,
                description=description,
                tags=tags,
            )
            self._versions[key] = mv
            self._models[mid] = model.with_version(version)
            return mv

    def _stable_model_id(self, name: str) -> str:
        for mid, model in self._models.items():
            if model.name == name:
                return mid
        return self._ids.generate("model")

    # -- lookups ------------------------------------------------------------ #
    def get_model(self, model_id: str) -> RegisteredModel:
        with self._lock:
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise ModelNotFoundError(f"Unknown model: {model_id!r}") from exc

    def get_version(self, model_id: str, version: str) -> ModelVersion:
        with self._lock:
            try:
                return self._versions[(model_id, version)]
            except KeyError as exc:
                raise VersionNotFoundError(
                    f"Unknown version {version!r} for model {model_id!r}"
                ) from exc

    def versions(self, model_id: str) -> Tuple[ModelVersion, ...]:
        """Return all versions of a model, ordered by semantic version ascending."""
        with self._lock:
            self.get_model(model_id)
            items = [v for (mid, _), v in self._versions.items() if mid == model_id]
        return tuple(sorted(items, key=lambda v: v.semantic_version))

    def history(self, model_id: str) -> Tuple[ModelVersion, ...]:
        """Alias for the ordered version history of a model."""
        return self.versions(model_id)

    def latest(self, model_id: str, stage: Optional[ModelStage] = None) -> ModelVersion:
        """Return the highest semantic version, optionally filtered by stage."""
        items = list(self.versions(model_id))
        if stage is not None:
            items = [v for v in items if v.stage is stage]
        if not items:
            raise VersionNotFoundError(
                f"No versions for model {model_id!r}"
                + (f" in stage {stage.value}" if stage else "")
            )
        return max(items, key=lambda v: v.semantic_version)

    def list_models(self) -> Tuple[RegisteredModel, ...]:
        with self._lock:
            return tuple(self._models[k] for k in sorted(self._models))

    # -- validation flag ---------------------------------------------------- #
    def mark_validated(
        self, model_id: str, version: str, passed: bool = True
    ) -> ModelVersion:
        """Record the outcome of validation for a version."""
        with self._lock:
            mv = self.get_version(model_id, version)
            updated = replace(mv, validation_passed=bool(passed))
            self._versions[(model_id, version)] = updated
            return updated

    # -- promotion ---------------------------------------------------------- #
    def evaluate_promotion(self, request: PromotionRequest) -> PromotionDecision:
        """Evaluate a promotion request without mutating registry state."""
        with self._lock:
            candidate = self.get_version(request.model_id, request.version)
            baseline = self._baseline_for(request)
            if candidate.stage is not request.from_stage:
                return PromotionDecision(
                    model_id=request.model_id,
                    version=request.version,
                    from_stage=request.from_stage,
                    to_stage=request.to_stage,
                    status=PromotionStatus.REJECTED,
                    reasons=(
                        f"Current stage {candidate.stage.value} does not match "
                        f"requested from_stage {request.from_stage.value}",
                    ),
                    checks={"stage_matches": False},
                    decided_at=self._clock.now(),
                    decided_by=request.approved_by or request.requested_by,
                )
            return self._policy.evaluate(request, candidate, baseline, self._clock.now())

    def promote(self, request: PromotionRequest) -> PromotionDecision:
        """Evaluate and, if approved, apply a promotion transition."""
        with self._lock:
            decision = self.evaluate_promotion(request)
            if not decision.approved:
                return decision
            candidate = self.get_version(request.model_id, request.version)
            now = self._clock.now()
            promoted = replace(candidate, stage=request.to_stage, promoted_at=now)
            self._versions[(request.model_id, request.version)] = promoted
            if request.to_stage is ModelStage.PRODUCTION:
                self._set_production(request.model_id, request.version, now)
            return decision

    def _set_production(self, model_id: str, version: str, now: str) -> None:
        model = self._models[model_id]
        previous = model.current_production_version
        if previous is not None and previous != version:
            prev_version = self._versions.get((model_id, previous))
            if prev_version is not None and prev_version.stage is ModelStage.PRODUCTION:
                self._versions[(model_id, previous)] = replace(
                    prev_version, stage=ModelStage.ARCHIVED, promoted_at=now
                )
        self._models[model_id] = replace(model, current_production_version=version)

    def rollback(self, model_id: str, target_version: str) -> ModelVersion:
        """Roll production back to *target_version*, archiving the current one."""
        with self._lock:
            model = self.get_model(model_id)
            target = self.get_version(model_id, target_version)
            now = self._clock.now()
            current = model.current_production_version
            if current is not None and current != target_version:
                cur = self._versions.get((model_id, current))
                if cur is not None:
                    self._versions[(model_id, current)] = replace(
                        cur, stage=ModelStage.ARCHIVED, promoted_at=now
                    )
            promoted = replace(target, stage=ModelStage.PRODUCTION, promoted_at=now)
            self._versions[(model_id, target_version)] = promoted
            self._models[model_id] = replace(model, current_production_version=target_version)
            return promoted

    def archive(self, model_id: str, version: str) -> ModelVersion:
        """Archive a version. Always permitted unless already archived."""
        with self._lock:
            mv = self.get_version(model_id, version)
            if mv.stage is ModelStage.ARCHIVED:
                raise InvalidStageTransitionError(
                    f"Version {version} of {model_id} is already archived"
                )
            now = self._clock.now()
            archived = replace(mv, stage=ModelStage.ARCHIVED, promoted_at=now)
            self._versions[(model_id, version)] = archived
            model = self._models[model_id]
            if model.current_production_version == version:
                self._models[model_id] = replace(model, current_production_version=None)
            return archived

    def _baseline_for(self, request: PromotionRequest) -> Optional[ModelVersion]:
        if request.baseline_version is not None:
            return self._versions.get((request.model_id, request.baseline_version))
        model = self._models.get(request.model_id)
        if model is not None and model.current_production_version:
            return self._versions.get(
                (request.model_id, model.current_production_version)
            )
        return None

    # -- lineage ------------------------------------------------------------ #
    def lineage(self, model_id: str, version: str) -> Dict[str, Any]:
        """Return the complete lineage chain for a version."""
        return self.get_version(model_id, version).lineage()

    # -- comparison --------------------------------------------------------- #
    def compare_versions(
        self, model_id: str, versions: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Any]:
        """Compare versions of a model, returning a deterministic ranking."""
        if versions is None:
            selected = list(self.versions(model_id))
        else:
            selected = [self.get_version(model_id, v) for v in versions]
        return self.compare(selected)

    def compare(self, versions: List[ModelVersion]) -> Dict[str, Any]:
        """Compare an explicit list of versions and rank them deterministically."""
        ranked = self._comparator.rank(list(versions)) if hasattr(
            self._comparator, "rank"
        ) else [(v, self._comparator.score(v)) for v in versions]
        if not hasattr(self._comparator, "rank"):
            ranked.sort(key=lambda pair: (-pair[1], pair[0].version))
        rows = []
        for rank, (mv, score) in enumerate(ranked, start=1):
            row = {
                "rank": rank,
                "model_id": mv.model_id,
                "version": mv.version,
                "stage": mv.stage.value,
                "composite_score": round(float(score), 12),
                "accuracy": mv.metrics.accuracy,
                "precision": mv.metrics.precision,
                "recall": mv.metrics.recall,
                "f1": mv.metrics.f1,
                "roc_auc": mv.metrics.roc_auc,
                "latency_ms": mv.latency_ms,
                "memory_mb": mv.memory_mb,
                "training_time_s": mv.training_time_s,
                "inference_time_ms": mv.inference_time_ms,
                "model_size_bytes": mv.model_size_bytes,
            }
            rows.append(row)
        return {
            "count": len(rows),
            "ranking": rows,
            "best_version": rows[0]["version"] if rows else None,
        }

    # -- model cards -------------------------------------------------------- #
    def generate_model_card(
        self,
        model_id: str,
        version: str,
        *,
        purpose: str = "",
        training_data: str = "",
        evaluation: str = "",
        limitations: Tuple[str, ...] = (),
        responsible_ai_notes: Tuple[str, ...] = (),
        dependencies: Optional[Mapping[str, str]] = None,
    ) -> ModelCard:
        """Generate a complete model card for a registered version."""
        model = self.get_model(model_id)
        mv = self.get_version(model_id, version)
        return ModelCard(
            model_id=model_id,
            model_name=model.name,
            version=version,
            purpose=purpose,
            training_data=training_data or mv.dataset_version,
            evaluation=evaluation,
            metrics=mv.metrics,
            limitations=limitations,
            responsible_ai_notes=responsible_ai_notes,
            deployment_stage=mv.stage,
            dependencies=dependencies or {},
            created_at=self._clock.now(),
        )

    # -- statistics --------------------------------------------------------- #
    def statistics(self) -> RegistryStatistics:
        """Compute aggregate statistics over the registry."""
        with self._lock:
            by_stage: Dict[str, int] = {stage.value: 0 for stage in ModelStage}
            production_models = 0
            archived = 0
            for mv in self._versions.values():
                by_stage[mv.stage.value] += 1
                if mv.stage is ModelStage.ARCHIVED:
                    archived += 1
            for model in self._models.values():
                if model.current_production_version:
                    production_models += 1
            return RegistryStatistics(
                total_models=len(self._models),
                total_versions=len(self._versions),
                versions_by_stage=by_stage,
                production_models=production_models,
                archived_versions=archived,
                generated_at=self._clock.now(),
            )

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "models": [self._models[k].to_dict() for k in sorted(self._models)],
                "versions": [
                    self._versions[k].to_dict()
                    for k in sorted(self._versions)
                ],
            }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def export_json(self, path: str, *, indent: int = 2) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(indent=indent))
        return path

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        promotion_policy: Optional[PromotionPolicy] = None,
        comparator: Optional[ModelComparator] = None,
    ) -> "ModelRegistry":
        registry = cls(
            clock=clock,
            id_generator=id_generator,
            promotion_policy=promotion_policy,
            comparator=comparator,
        )
        try:
            for entry in data.get("models", []):
                model = RegisteredModel.from_dict(entry)
                registry._models[model.model_id] = model
            for entry in data.get("versions", []):
                mv = ModelVersion.from_dict(entry)
                registry._versions[(mv.model_id, mv.version)] = mv
        except (KeyError, TypeError) as exc:
            raise SerializationError(f"Cannot deserialise ModelRegistry: {exc}") from exc
        return registry


def create_model_registry(*, deterministic: bool = True, seed: str = "model") -> ModelRegistry:
    """Factory returning a configured :class:`ModelRegistry`."""
    if deterministic:
        return ModelRegistry(
            clock=LogicalClock(),
            id_generator=DeterministicIdGenerator(seed=seed),
        )
    from mlops.experiment_models import SequentialIdGenerator, SystemClock

    return ModelRegistry(clock=SystemClock(), id_generator=SequentialIdGenerator())