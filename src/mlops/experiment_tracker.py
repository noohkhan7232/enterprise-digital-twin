"""Deterministic experiment tracker and end-to-end MLOps CLI demo.

The :class:`ExperimentTracker` records experiments and runs together with their
full provenance (seeds, git commit, dataset version, environment), the standard
train / validation / test metrics, resource usage and artifacts. All time and
identity is dependency-injected, so tracked state is fully reproducible.

Running this module as a script executes a deterministic demo that exercises
the whole platform::

    python src/mlops/experiment_tracker.py --demo
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standalone-script bootstrap: ensure the ``src`` directory is importable when
# this file is executed directly (``python src/mlops/experiment_tracker.py``).
# When imported as part of the ``mlops`` package this is a no-op.
# --------------------------------------------------------------------------- #
if __package__ in (None, ""):  # pragma: no cover - exercised only as a script
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import threading
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mlops.experiment_models import (
    ArtifactType,
    Clock,
    DeterministicIdGenerator,
    Experiment,
    ExperimentMetadata,
    ExperimentMetrics,
    ExperimentRun,
    ExperimentStatistics,
    ExperimentStatus,
    HyperParameters,
    IdGenerator,
    LogicalClock,
    MLOpsError,
    SerializationError,
    ValidationError,
)
from mlops.artifact_store import ArtifactStore, create_artifact_store
from mlops.reproducibility import (
    EnvironmentProvider,
    StaticEnvironmentProvider,
)

__all__ = [
    "ExperimentTrackerError",
    "ExperimentNotFoundError",
    "RunNotFoundError",
    "DuplicateExperimentError",
    "DuplicateRunError",
    "ExperimentTracker",
    "create_experiment_tracker",
    "run_demo",
    "main",
]

logger = logging.getLogger("mlops.experiment_tracker")
logger.addHandler(logging.NullHandler())

_VALID_SPLITS = ("training", "validation", "test")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ExperimentTrackerError(MLOpsError):
    """Base error for the experiment tracker."""


class ExperimentNotFoundError(ExperimentTrackerError):
    """Raised when an experiment id is unknown."""


class RunNotFoundError(ExperimentTrackerError):
    """Raised when a run id is unknown."""


class DuplicateExperimentError(ExperimentTrackerError):
    """Raised when an experiment id already exists."""


class DuplicateRunError(ExperimentTrackerError):
    """Raised when a run id already exists."""


# --------------------------------------------------------------------------- #
# Experiment tracker
# --------------------------------------------------------------------------- #
class ExperimentTracker:
    """A thread-safe, reproducible tracker for experiments and runs."""

    def __init__(
        self,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        environment_provider: Optional[EnvironmentProvider] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="experiment")
        self._env: EnvironmentProvider = environment_provider or StaticEnvironmentProvider()
        self._artifacts: ArtifactStore = artifact_store or ArtifactStore(
            clock=self._clock, id_generator=self._ids
        )
        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, ExperimentRun] = {}
        self._lock = threading.RLock()

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._artifacts

    # -- experiments -------------------------------------------------------- #
    def create_experiment(
        self,
        name: str,
        *,
        experiment_id: Optional[str] = None,
        description: str = "",
        parent_experiment_id: Optional[str] = None,
        dataset_version: str = "",
        git_commit: str = "",
        random_seed: int = 0,
        numpy_seed: int = 0,
        notes: str = "",
        tags: Tuple[str, ...] = (),
    ) -> Experiment:
        """Create and store a new experiment."""
        with self._lock:
            eid = experiment_id or self._ids.generate("exp")
            if eid in self._experiments:
                raise DuplicateExperimentError(f"Experiment id already exists: {eid!r}")
            now = self._clock.now()
            metadata = ExperimentMetadata(
                dataset_version=dataset_version,
                git_commit=git_commit,
                random_seed=random_seed,
                numpy_seed=numpy_seed,
                python_version=self._env.python_version(),
                platform=self._env.platform(),
                hostname=self._env.hostname(),
                created_at=now,
                notes=notes,
                tags=tags,
            )
            experiment = Experiment(
                experiment_id=eid,
                name=name,
                description=description,
                status=ExperimentStatus.CREATED,
                metadata=metadata,
                created_at=now,
                parent_experiment_id=parent_experiment_id,
                tags=tags,
            )
            self._experiments[eid] = experiment
            logger.info("Created experiment %s (%s)", eid, name)
            return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        with self._lock:
            try:
                return self._experiments[experiment_id]
            except KeyError as exc:
                raise ExperimentNotFoundError(f"Unknown experiment: {experiment_id!r}") from exc

    def list_experiments(self) -> Tuple[Experiment, ...]:
        with self._lock:
            return tuple(self._experiments[k] for k in sorted(self._experiments))

    # -- runs --------------------------------------------------------------- #
    def start_run(
        self,
        experiment_id: str,
        *,
        run_id: Optional[str] = None,
        hyperparameters: Optional[Mapping[str, Any]] = None,
        dataset_version: str = "",
        git_commit: str = "",
        random_seed: int = 0,
        numpy_seed: int = 0,
        notes: str = "",
        tags: Tuple[str, ...] = (),
    ) -> ExperimentRun:
        """Start a new run within an experiment."""
        with self._lock:
            experiment = self.get_experiment(experiment_id)
            rid = run_id or self._ids.generate("run")
            if rid in self._runs:
                raise DuplicateRunError(f"Run id already exists: {rid!r}")
            now = self._clock.now()
            run = ExperimentRun(
                run_id=rid,
                experiment_id=experiment_id,
                status=ExperimentStatus.RUNNING,
                parent_experiment_id=experiment.parent_experiment_id,
                hyperparameters=HyperParameters(dict(hyperparameters or {})),
                dataset_version=dataset_version or experiment.metadata.dataset_version,
                git_commit=git_commit or experiment.metadata.git_commit,
                random_seed=random_seed,
                numpy_seed=numpy_seed,
                python_version=self._env.python_version(),
                platform=self._env.platform(),
                hostname=self._env.hostname(),
                started_at=now,
                notes=notes,
                tags=tags,
            )
            self._runs[rid] = run
            self._experiments[experiment_id] = experiment.with_run(rid).with_status(
                ExperimentStatus.RUNNING
            )
            logger.info("Started run %s for experiment %s", rid, experiment_id)
            return run

    def get_run(self, run_id: str) -> ExperimentRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise RunNotFoundError(f"Unknown run: {run_id!r}") from exc

    def runs_for_experiment(self, experiment_id: str) -> Tuple[ExperimentRun, ...]:
        with self._lock:
            self.get_experiment(experiment_id)
            items = [r for r in self._runs.values() if r.experiment_id == experiment_id]
        return tuple(sorted(items, key=lambda r: r.run_id))

    def list_runs(self) -> Tuple[ExperimentRun, ...]:
        with self._lock:
            return tuple(self._runs[k] for k in sorted(self._runs))

    # -- logging ------------------------------------------------------------ #
    def log_metrics(
        self, run_id: str, split: str, metrics: ExperimentMetrics | Mapping[str, Any]
    ) -> ExperimentRun:
        """Record metrics for the ``training``/``validation``/``test`` split."""
        if split not in _VALID_SPLITS:
            raise ValidationError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
        metric_obj = (
            metrics
            if isinstance(metrics, ExperimentMetrics)
            else ExperimentMetrics.from_dict(dict(metrics))
        )
        field = f"{split}_metrics"
        with self._lock:
            run = self.get_run(run_id)
            updated = replace(run, **{field: metric_obj})
            self._runs[run_id] = updated
            logger.info("Logged %s metrics for run %s", split, run_id)
            return updated

    def log_resource_usage(
        self,
        run_id: str,
        *,
        training_time: float = 0.0,
        inference_time: float = 0.0,
        memory_usage: float = 0.0,
        cpu_usage: float = 0.0,
    ) -> ExperimentRun:
        """Record resource usage for a run."""
        with self._lock:
            run = self.get_run(run_id)
            updated = replace(
                run,
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
            )
            self._runs[run_id] = updated
            return updated

    def log_artifact(
        self,
        run_id: str,
        artifact_type: ArtifactType,
        name: str,
        *,
        size_bytes: int = 0,
        description: str = "",
        properties: Optional[Mapping[str, Any]] = None,
        tags: Tuple[str, ...] = (),
    ):
        """Register an artifact and attach it to the run."""
        with self._lock:
            run = self.get_run(run_id)
            metadata = self._artifacts.create_artifact(
                artifact_type,
                name,
                size_bytes=size_bytes,
                experiment_id=run.experiment_id,
                run_id=run_id,
                description=description,
                properties=properties,
                tags=tags,
            )
            self._runs[run_id] = run.with_artifact(metadata.artifact_id)
            logger.info("Logged artifact %s for run %s", metadata.artifact_id, run_id)
            return metadata

    def finalize_run(
        self, run_id: str, status: ExperimentStatus = ExperimentStatus.COMPLETED
    ) -> ExperimentRun:
        """Mark a run as completed (or failed) and stamp completion time."""
        if not isinstance(status, ExperimentStatus):
            status = ExperimentStatus(status)
        with self._lock:
            run = self.get_run(run_id)
            updated = replace(run, status=status, completed_at=self._clock.now())
            self._runs[run_id] = updated
            logger.info("Finalised run %s with status %s", run_id, status.value)
            return updated

    # -- statistics --------------------------------------------------------- #
    def statistics(self, primary_metric: str = "f1") -> ExperimentStatistics:
        """Compute aggregate statistics across experiments and runs."""
        with self._lock:
            runs = list(self._runs.values())
            completed = [r for r in runs if r.status is ExperimentStatus.COMPLETED]
            failed = [r for r in runs if r.status is ExperimentStatus.FAILED]
            running = [r for r in runs if r.status is ExperimentStatus.RUNNING]
            total_training = sum(r.training_time for r in runs)
            avg_training = total_training / len(runs) if runs else 0.0
            best_value: Optional[float] = None
            best_run: Optional[str] = None
            for r in sorted(runs, key=lambda x: x.run_id):
                value = r.validation_metrics.get(primary_metric)
                if value is None:
                    continue
                if best_value is None or float(value) > best_value:
                    best_value = float(value)
                    best_run = r.run_id
            return ExperimentStatistics(
                total_experiments=len(self._experiments),
                total_runs=len(runs),
                completed_runs=len(completed),
                failed_runs=len(failed),
                running_runs=len(running),
                average_training_time=avg_training,
                best_metric_value=best_value,
                best_run_id=best_run,
                generated_at=self._clock.now(),
            )

    # -- serialisation ------------------------------------------------------ #
    def export_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Return a single experiment with its runs as a JSON-ready dict."""
        experiment = self.get_experiment(experiment_id)
        runs = self.runs_for_experiment(experiment_id)
        return {
            "experiment": experiment.to_dict(),
            "runs": [r.to_dict() for r in runs],
        }

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "experiments": [
                    self._experiments[k].to_dict() for k in sorted(self._experiments)
                ],
                "runs": [self._runs[k].to_dict() for k in sorted(self._runs)],
            }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def export_json(self, path: str, *, indent: int = 2) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(indent=indent))
        return path

    def export_experiment_json(
        self, experiment_id: str, path: str, *, indent: int = 2
    ) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.export_experiment(experiment_id), indent=indent, sort_keys=True))
        return path

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        environment_provider: Optional[EnvironmentProvider] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> "ExperimentTracker":
        tracker = cls(
            clock=clock,
            id_generator=id_generator,
            environment_provider=environment_provider,
            artifact_store=artifact_store,
        )
        try:
            for entry in data.get("experiments", []):
                experiment = Experiment.from_dict(entry)
                tracker._experiments[experiment.experiment_id] = experiment
            for entry in data.get("runs", []):
                run = ExperimentRun.from_dict(entry)
                tracker._runs[run.run_id] = run
        except (KeyError, TypeError) as exc:
            raise SerializationError(f"Cannot deserialise ExperimentTracker: {exc}") from exc
        return tracker


def create_experiment_tracker(*, deterministic: bool = True) -> ExperimentTracker:
    """Factory returning a configured :class:`ExperimentTracker`."""
    if deterministic:
        return ExperimentTracker(
            clock=LogicalClock(),
            id_generator=DeterministicIdGenerator(seed="experiment"),
            environment_provider=StaticEnvironmentProvider(),
        )
    from mlops.experiment_models import SequentialIdGenerator, SystemClock
    from mlops.reproducibility import SystemEnvironmentProvider

    return ExperimentTracker(
        clock=SystemClock(),
        id_generator=SequentialIdGenerator(),
        environment_provider=SystemEnvironmentProvider(),
    )


# --------------------------------------------------------------------------- #
# CLI demo: full end-to-end pipeline
# --------------------------------------------------------------------------- #
def run_demo(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute the deterministic end-to-end MLOps pipeline demo.

    Experiment creation -> training run -> metric logging -> artifact
    registration -> model registration -> model comparison -> promotion to
    production -> registry export.
    """
    from mlops.model_registry import create_model_registry
    from mlops.experiment_models import ModelStage, PromotionRequest

    tracker = create_experiment_tracker(deterministic=True)
    registry = create_model_registry(deterministic=True)

    # 1. Experiment creation.
    experiment = tracker.create_experiment(
        "fraud-detection",
        description="Gradient-boosted fraud classifier",
        dataset_version="1.4.0",
        git_commit="a1b2c3d4e5f6",
        random_seed=42,
        numpy_seed=7,
        tags=("fraud", "tabular"),
    )

    decisions: List[Dict[str, Any]] = []
    versions = ["1.0.0", "1.1.0"]
    metric_table = {
        "1.0.0": dict(accuracy=0.940, precision=0.910, recall=0.880, f1=0.895, roc_auc=0.952),
        "1.1.0": dict(accuracy=0.951, precision=0.922, recall=0.901, f1=0.911, roc_auc=0.964),
    }
    perf_table = {
        "1.0.0": dict(latency_ms=42.0, memory_mb=312.0, inference_time_ms=3.1, model_size_bytes=1_240_000, training_time_s=58.0),
        "1.1.0": dict(latency_ms=39.0, memory_mb=305.0, inference_time_ms=2.9, model_size_bytes=1_180_000, training_time_s=61.0),
    }
    model_id: Optional[str] = None

    for version in versions:
        # 2. Training run.
        run = tracker.start_run(
            experiment.experiment_id,
            hyperparameters={"learning_rate": 0.05, "max_depth": 6, "n_estimators": 400},
            random_seed=42,
            numpy_seed=7,
            tags=(version,),
        )
        # 3. Metric logging.
        tracker.log_metrics(run.run_id, "training", ExperimentMetrics(**metric_table[version]))
        tracker.log_metrics(run.run_id, "validation", ExperimentMetrics(**metric_table[version]))
        tracker.log_metrics(run.run_id, "test", ExperimentMetrics(**metric_table[version]))
        tracker.log_resource_usage(
            run.run_id,
            training_time=perf_table[version]["training_time_s"],
            inference_time=perf_table[version]["inference_time_ms"] / 1000.0,
            memory_usage=perf_table[version]["memory_mb"],
            cpu_usage=0.75,
        )
        # 4. Artifact registration.
        model_artifact = tracker.log_artifact(
            run.run_id, ArtifactType.SERIALIZED_MODEL, f"fraud-model-{version}",
            size_bytes=perf_table[version]["model_size_bytes"],
            properties={"framework": "numpy-gbm", "format": "npz"},
        )
        tracker.log_artifact(run.run_id, ArtifactType.ROC_CURVE, f"roc-{version}")
        tracker.log_artifact(run.run_id, ArtifactType.CONFUSION_MATRIX, f"cm-{version}")
        tracker.finalize_run(run.run_id)

        # 5. Model registration.
        mv = registry.register_model(
            "fraud-detector",
            version,
            model_id=model_id,
            metrics=ExperimentMetrics(**metric_table[version]),
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            dataset_version="1.4.0",
            artifact_ids=(model_artifact.artifact_id,),
            latency_ms=perf_table[version]["latency_ms"],
            memory_mb=perf_table[version]["memory_mb"],
            model_size_bytes=perf_table[version]["model_size_bytes"],
            training_time_s=perf_table[version]["training_time_s"],
            inference_time_ms=perf_table[version]["inference_time_ms"],
        )
        model_id = mv.model_id
        # Advance through validation -> staging.
        registry.promote(PromotionRequest(model_id=model_id, version=version,
                          from_stage=ModelStage.REGISTERED, to_stage=ModelStage.VALIDATION))
        registry.mark_validated(model_id, version, True)
        registry.promote(PromotionRequest(model_id=model_id, version=version,
                          from_stage=ModelStage.VALIDATION, to_stage=ModelStage.STAGING,
                          approved_by="ml-lead"))

    # 6. Model comparison.
    comparison = registry.compare_versions(model_id)

    # 7. Promotion to production (best version, with gates).
    best_version = comparison["best_version"]
    baseline = registry.get_model(model_id).current_production_version
    promotion = registry.promote(
        PromotionRequest(
            model_id=model_id,
            version=best_version,
            from_stage=ModelStage.STAGING,
            to_stage=ModelStage.PRODUCTION,
            requested_by="ml-engineer",
            approved_by="ml-lead",
            reason="Best composite score on holdout",
            baseline_version=baseline,
            max_latency_ms=100.0,
            max_memory_mb=512.0,
            primary_metric="f1",
        )
    )
    decisions.append(promotion.to_dict())

    card = registry.generate_model_card(
        model_id, best_version,
        purpose="Detect fraudulent transactions in real time.",
        training_data="Transactions dataset v1.4.0 (anonymised).",
        evaluation="Stratified hold-out with temporal split.",
        limitations=("Trained on a single region; recalibrate before reuse.",),
        responsible_ai_notes=("Monitor false-positive rate across cohorts.",),
        dependencies={"python": "3.12.3", "numpy": "2.4.4"},
    )

    result = {
        "experiment": tracker.export_experiment(experiment.experiment_id),
        "registry": registry.to_dict(),
        "comparison": comparison,
        "model_card": card.to_dict(),
        "promotion_decision": promotion.to_dict(),
        "experiment_statistics": tracker.statistics().to_dict(),
        "registry_statistics": registry.statistics().to_dict(),
    }

    if output_dir is not None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        tracker.export_experiment_json(
            experiment.experiment_id, os.path.join(output_dir, "experiment.json")
        )
        registry.export_json(os.path.join(output_dir, "registry.json"))
        with open(os.path.join(output_dir, "model_card.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(card.to_dict(), indent=2, sort_keys=True))
        with open(os.path.join(output_dir, "comparison.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(comparison, indent=2, sort_keys=True))

    return result


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Enterprise MLOps experiment tracker")
    parser.add_argument("--demo", action="store_true", help="Run the end-to-end demo")
    parser.add_argument("--output-dir", default=None, help="Directory for exported JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress info logging")
    args = parser.parse_args(argv)

    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.demo:
        result = run_demo(output_dir=args.output_dir)
        summary = {
            "experiment_id": result["experiment"]["experiment"]["experiment_id"],
            "runs": len(result["experiment"]["runs"]),
            "best_version": result["comparison"]["best_version"],
            "promotion_status": result["promotion_decision"]["status"],
            "production_version": result["model_card"]["version"],
            "registry_statistics": result["registry_statistics"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())