"""Tests for the metadata-only artifact store."""

from __future__ import annotations

import json
import threading

import pytest

from mlops.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    DuplicateArtifactError,
    Sha256HashStrategy,
    create_artifact_store,
)
from mlops.experiment_models import (
    ArtifactMetadata,
    ArtifactType,
    DeterministicIdGenerator,
    LogicalClock,
    ModelArtifact,
    ValidationError,
)


def new_store():
    return create_artifact_store(deterministic=True)


@pytest.mark.parametrize("atype", list(ArtifactType))
def test_create_each_artifact_type(atype):
    store = new_store()
    meta = store.create_artifact(atype, f"name-{atype.value}")
    assert meta.artifact_type is atype
    assert store.exists(meta.artifact_id)


@pytest.mark.parametrize("atype", list(ArtifactType))
def test_each_type_listed_by_type(atype):
    store = new_store()
    store.create_artifact(atype, "x")
    assert len(store.by_type(atype)) == 1


def test_create_assigns_content_hash():
    store = new_store()
    meta = store.create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.9})
    assert len(meta.content_hash) == 64


def test_content_hash_is_deterministic():
    a = new_store().create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.9})
    b = new_store().create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.9})
    assert a.content_hash == b.content_hash


def test_content_hash_changes_with_properties():
    a = new_store().create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.9})
    b = new_store().create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.8})
    assert a.content_hash != b.content_hash


def test_duplicate_id_rejected():
    store = new_store()
    meta = store.create_artifact(ArtifactType.ROC_CURVE, "roc")
    with pytest.raises(DuplicateArtifactError):
        store.register(meta)


def test_explicit_duplicate_id_rejected():
    store = new_store()
    store.create_artifact(ArtifactType.ROC_CURVE, "roc", artifact_id="fixed")
    with pytest.raises(DuplicateArtifactError):
        store.create_artifact(ArtifactType.PR_CURVE, "pr", artifact_id="fixed")


def test_get_unknown_raises():
    with pytest.raises(ArtifactNotFoundError):
        new_store().get("missing")


def test_exists_false_for_unknown():
    assert new_store().exists("missing") is False


def test_register_requires_metadata_instance():
    with pytest.raises(ValidationError):
        new_store().register({"artifact_id": "x"})


def test_register_model_artifact_requires_type():
    with pytest.raises(ValidationError):
        new_store().register_model_artifact("not-a-model-artifact")


def test_register_prebuilt_metadata():
    store = new_store()
    meta = ArtifactMetadata(artifact_id="a-1", artifact_type=ArtifactType.SHAP_RESULT, name="shap")
    assert store.register(meta).artifact_id == "a-1"
    assert store.count() == 1


def test_register_model_artifact_derives_metadata():
    store = new_store()
    artifact = ModelArtifact(artifact_id="m-art", name="model", framework="numpy", format="npz")
    meta = store.register_model_artifact(artifact)
    assert meta.artifact_type is ArtifactType.SERIALIZED_MODEL
    assert meta.content_hash


def test_register_model_artifact_with_metadata():
    store = new_store()
    inner = ArtifactMetadata(artifact_id="m-art", artifact_type=ArtifactType.SERIALIZED_MODEL, name="m")
    artifact = ModelArtifact(artifact_id="m-art", name="model", metadata=inner)
    assert store.register_model_artifact(artifact).artifact_id == "m-art"


@pytest.mark.parametrize("n", [0, 1, 5, 25, 100])
def test_count_tracks_size(n):
    store = new_store()
    for i in range(n):
        store.create_artifact(ArtifactType.EVALUATION_REPORT, f"r{i}")
    assert store.count() == n


def test_filter_by_run_and_experiment():
    store = new_store()
    store.create_artifact(ArtifactType.ROC_CURVE, "a", run_id="r1", experiment_id="e1")
    store.create_artifact(ArtifactType.PR_CURVE, "b", run_id="r2", experiment_id="e1")
    store.create_artifact(ArtifactType.PR_CURVE, "c", run_id="r1", experiment_id="e2")
    assert len(store.list_artifacts(run_id="r1")) == 2
    assert len(store.list_artifacts(experiment_id="e1")) == 2
    assert len(store.list_artifacts(run_id="r1", experiment_id="e2")) == 1


def test_list_is_sorted_by_id():
    store = new_store()
    for i in range(5):
        store.create_artifact(ArtifactType.ROC_CURVE, f"r{i}", artifact_id=f"id-{4 - i}")
    ids = [a.artifact_id for a in store.list_artifacts()]
    assert ids == sorted(ids)


def test_artifact_ids_sorted():
    store = new_store()
    store.create_artifact(ArtifactType.ROC_CURVE, "a", artifact_id="z")
    store.create_artifact(ArtifactType.ROC_CURVE, "b", artifact_id="a")
    assert store.artifact_ids() == ("a", "z")


def test_find_by_content_hash():
    store = new_store()
    meta = store.create_artifact(ArtifactType.ROC_CURVE, "roc", properties={"auc": 0.9})
    found = store.find_by_content_hash(meta.content_hash)
    assert found and found[0].artifact_id == meta.artifact_id


def test_to_json_is_valid_and_sorted():
    store = new_store()
    store.create_artifact(ArtifactType.ROC_CURVE, "roc")
    payload = json.loads(store.to_json())
    assert "artifacts" in payload and len(payload["artifacts"]) == 1


def test_export_and_reload():
    import tempfile, os
    store = new_store()
    store.create_artifact(ArtifactType.ROC_CURVE, "roc", run_id="r1")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "artifacts.json")
        store.export_json(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    reloaded = ArtifactStore.from_dict(data)
    assert reloaded.count() == 1


def test_from_dict_roundtrip_equivalent():
    store = new_store()
    store.create_artifact(ArtifactType.CONFUSION_MATRIX, "cm", run_id="r1")
    store.create_artifact(ArtifactType.CALIBRATION_PLOT, "cal", run_id="r2")
    reloaded = ArtifactStore.from_dict(store.to_dict())
    assert reloaded.to_dict() == store.to_dict()


def test_sha256_strategy_direct():
    strategy = Sha256HashStrategy()
    h1 = strategy.compute({"a": 1, "b": 2})
    h2 = strategy.compute({"b": 2, "a": 1})
    assert h1 == h2 and len(h1) == 64


def test_thread_safe_concurrent_creates():
    store = new_store()

    def worker(start):
        for i in range(50):
            store.create_artifact(ArtifactType.ROC_CURVE, f"r-{start}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.count() == 8 * 50


def test_large_store_performance():
    store = new_store()
    for i in range(2000):
        store.create_artifact(ArtifactType.EVALUATION_REPORT, f"r{i}")
    assert store.count() == 2000
    assert len(store.by_type(ArtifactType.EVALUATION_REPORT)) == 2000


def test_non_deterministic_factory_sequential_ids():
    store = create_artifact_store(deterministic=False)
    meta = store.create_artifact(ArtifactType.ROC_CURVE, "roc")
    assert meta.artifact_id.startswith("artifact-")