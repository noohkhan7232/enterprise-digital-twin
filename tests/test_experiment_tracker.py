"""Tests for the experiment tracker and the end-to-end CLI demo."""

from __future__ import annotations

import json
import os
import tempfile
import threading

import pytest

from mlops.artifact_store import ArtifactStore
from mlops.experiment_models import (
    ArtifactType,
    DeterministicIdGenerator,
    ExperimentMetrics,
    ExperimentStatus,
    LogicalClock,
    ValidationError,
)
from mlops.experiment_tracker import (
    DuplicateExperimentError,
    DuplicateRunError,
    ExperimentNotFoundError,
    ExperimentTracker,
    RunNotFoundError,
    create_experiment_tracker,
    main,
    run_demo,
)


def new_tracker():
    return create_experiment_tracker(deterministic=True)


def started_run(tracker):
    exp = tracker.create_experiment("demo", dataset_version="1.0.0")
    return exp, tracker.start_run(exp.experiment_id, hyperparameters={"lr": 0.1})


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def test_create_experiment_status():
    exp = new_tracker().create_experiment("demo")
    assert exp.status is ExperimentStatus.CREATED


def test_create_experiment_captures_environment():
    exp = new_tracker().create_experiment("demo")
    assert exp.metadata.python_version == "3.12.3"
    assert exp.metadata.platform


def test_create_experiment_seeds():
    exp = new_tracker().create_experiment("demo", random_seed=42, numpy_seed=7)
    assert exp.metadata.random_seed == 42
    assert exp.metadata.numpy_seed == 7


def test_duplicate_experiment_rejected():
    t = new_tracker()
    t.create_experiment("demo", experiment_id="fixed")
    with pytest.raises(DuplicateExperimentError):
        t.create_experiment("demo2", experiment_id="fixed")


def test_get_unknown_experiment():
    with pytest.raises(ExperimentNotFoundError):
        new_tracker().get_experiment("missing")


def test_list_experiments_sorted():
    t = new_tracker()
    t.create_experiment("b", experiment_id="exp-b")
    t.create_experiment("a", experiment_id="exp-a")
    assert [e.experiment_id for e in t.list_experiments()] == ["exp-a", "exp-b"]


def test_parent_experiment_link():
    t = new_tracker()
    parent = t.create_experiment("parent")
    child = t.create_experiment("child", parent_experiment_id=parent.experiment_id)
    assert child.parent_experiment_id == parent.experiment_id


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def test_start_run_status_running():
    t = new_tracker()
    _, run = started_run(t)
    assert run.status is ExperimentStatus.RUNNING


def test_start_run_registers_with_experiment():
    t = new_tracker()
    exp, run = started_run(t)
    assert run.run_id in t.get_experiment(exp.experiment_id).run_ids


def test_start_run_inherits_dataset_version():
    t = new_tracker()
    exp = t.create_experiment("demo", dataset_version="3.2.1")
    run = t.start_run(exp.experiment_id)
    assert run.dataset_version == "3.2.1"


def test_start_run_captures_hyperparameters():
    t = new_tracker()
    _, run = started_run(t)
    assert run.hyperparameters.get("lr") == 0.1


def test_start_run_unknown_experiment():
    with pytest.raises(ExperimentNotFoundError):
        new_tracker().start_run("missing")


def test_duplicate_run_rejected():
    t = new_tracker()
    exp = t.create_experiment("demo")
    t.start_run(exp.experiment_id, run_id="fixed")
    with pytest.raises(DuplicateRunError):
        t.start_run(exp.experiment_id, run_id="fixed")


def test_get_unknown_run():
    with pytest.raises(RunNotFoundError):
        new_tracker().get_run("missing")


def test_runs_for_experiment():
    t = new_tracker()
    exp = t.create_experiment("demo")
    t.start_run(exp.experiment_id)
    t.start_run(exp.experiment_id)
    assert len(t.runs_for_experiment(exp.experiment_id)) == 2


# --------------------------------------------------------------------------- #
# Metric & resource logging
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("split", ["training", "validation", "test"])
def test_log_metrics_each_split(split):
    t = new_tracker()
    _, run = started_run(t)
    updated = t.log_metrics(run.run_id, split, ExperimentMetrics(f1=0.88))
    assert getattr(updated, f"{split}_metrics").f1 == 0.88


def test_log_metrics_from_mapping():
    t = new_tracker()
    _, run = started_run(t)
    updated = t.log_metrics(run.run_id, "validation", {"accuracy": 0.95})
    assert updated.validation_metrics.accuracy == 0.95


def test_log_metrics_invalid_split():
    t = new_tracker()
    _, run = started_run(t)
    with pytest.raises(ValidationError):
        t.log_metrics(run.run_id, "holdout", ExperimentMetrics())


def test_log_resource_usage():
    t = new_tracker()
    _, run = started_run(t)
    updated = t.log_resource_usage(run.run_id, training_time=12.0, memory_usage=256.0, cpu_usage=0.5)
    assert updated.training_time == 12.0
    assert updated.memory_usage == 256.0
    assert updated.cpu_usage == 0.5


def test_log_artifact_attaches_to_run():
    t = new_tracker()
    _, run = started_run(t)
    meta = t.log_artifact(run.run_id, ArtifactType.ROC_CURVE, "roc")
    assert meta.artifact_id in t.get_run(run.run_id).artifact_ids
    assert t.artifact_store.exists(meta.artifact_id)


def test_log_artifact_sets_provenance():
    t = new_tracker()
    exp, run = started_run(t)
    meta = t.log_artifact(run.run_id, ArtifactType.SHAP_RESULT, "shap")
    assert meta.run_id == run.run_id
    assert meta.experiment_id == exp.experiment_id


@pytest.mark.parametrize("status", [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED])
def test_finalize_run(status):
    t = new_tracker()
    _, run = started_run(t)
    updated = t.finalize_run(run.run_id, status)
    assert updated.status is status
    assert updated.completed_at is not None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_statistics_basic():
    t = new_tracker()
    exp = t.create_experiment("demo")
    r1 = t.start_run(exp.experiment_id)
    t.log_metrics(r1.run_id, "validation", ExperimentMetrics(f1=0.8))
    t.log_resource_usage(r1.run_id, training_time=10.0)
    t.finalize_run(r1.run_id)
    r2 = t.start_run(exp.experiment_id)
    t.log_metrics(r2.run_id, "validation", ExperimentMetrics(f1=0.9))
    t.log_resource_usage(r2.run_id, training_time=20.0)
    t.finalize_run(r2.run_id)
    stats = t.statistics(primary_metric="f1")
    assert stats.total_runs == 2
    assert stats.completed_runs == 2
    assert stats.average_training_time == pytest.approx(15.0)
    assert stats.best_metric_value == pytest.approx(0.9)
    assert stats.best_run_id == r2.run_id


def test_statistics_counts_running_and_failed():
    t = new_tracker()
    exp = t.create_experiment("demo")
    t.start_run(exp.experiment_id)  # running
    r = t.start_run(exp.experiment_id)
    t.finalize_run(r.run_id, ExperimentStatus.FAILED)
    stats = t.statistics()
    assert stats.running_runs == 1
    assert stats.failed_runs == 1


def test_statistics_empty():
    stats = new_tracker().statistics()
    assert stats.total_runs == 0
    assert stats.average_training_time == 0.0


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_export_experiment_structure():
    t = new_tracker()
    exp, run = started_run(t)
    payload = t.export_experiment(exp.experiment_id)
    assert payload["experiment"]["experiment_id"] == exp.experiment_id
    assert len(payload["runs"]) == 1


def test_tracker_json_roundtrip():
    t = new_tracker()
    exp, run = started_run(t)
    t.log_metrics(run.run_id, "validation", ExperimentMetrics(f1=0.9))
    reloaded = ExperimentTracker.from_dict(json.loads(t.to_json()))
    assert reloaded.to_dict() == t.to_dict()


def test_export_experiment_json_file():
    t = new_tracker()
    exp, _ = started_run(t)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "experiment.json")
        t.export_experiment_json(exp.experiment_id, path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    assert data["experiment"]["name"] == "demo"


def test_custom_artifact_store_injection():
    store = ArtifactStore(clock=LogicalClock(), id_generator=DeterministicIdGenerator(seed="x"))
    t = ExperimentTracker(clock=LogicalClock(), id_generator=DeterministicIdGenerator(seed="x"),
                          artifact_store=store)
    exp = t.create_experiment("demo")
    run = t.start_run(exp.experiment_id)
    t.log_artifact(run.run_id, ArtifactType.ROC_CURVE, "roc")
    assert store.count() == 1


# --------------------------------------------------------------------------- #
# Determinism & thread-safety
# --------------------------------------------------------------------------- #
def test_demo_is_deterministic():
    first = run_demo()
    second = run_demo()
    assert first == second


def test_demo_promotes_to_production():
    result = run_demo()
    assert result["promotion_decision"]["status"] == "APPROVED"
    assert result["model_card"]["deployment_stage"] == "PRODUCTION"


def test_demo_registry_has_two_versions():
    result = run_demo()
    assert result["registry_statistics"]["total_versions"] == 2


def test_demo_comparison_picks_best():
    result = run_demo()
    assert result["comparison"]["best_version"] == "1.1.0"


def test_cli_main_demo_writes_files():
    with tempfile.TemporaryDirectory() as d:
        code = main(["--demo", "--quiet", "--output-dir", d])
        assert code == 0
        for name in ("experiment.json", "registry.json", "model_card.json", "comparison.json"):
            assert os.path.exists(os.path.join(d, name))


def test_cli_main_no_args_returns_zero():
    assert main(["--quiet"]) == 0


def test_thread_safe_concurrent_runs():
    t = new_tracker()
    exp = t.create_experiment("demo")

    def worker():
        for _ in range(20):
            run = t.start_run(exp.experiment_id)
            t.log_metrics(run.run_id, "validation", ExperimentMetrics(f1=0.9))
            t.finalize_run(run.run_id)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(t.list_runs()) == 6 * 20