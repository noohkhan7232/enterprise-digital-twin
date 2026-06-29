"""Tests for the model registry, lifecycle, promotion and comparison engines."""

from __future__ import annotations

import json
import threading

import pytest

from mlops.experiment_models import (
    ExperimentMetrics,
    MetricDirection,
    ModelStage,
    ModelVersion,
    PromotionRequest,
    PromotionStatus,
    ValidationError,
)
from mlops.model_registry import (
    ALLOWED_TRANSITIONS,
    DefaultPromotionPolicy,
    DuplicateModelVersionError,
    InvalidStageTransitionError,
    ModelNotFoundError,
    ModelRegistry,
    VersionNotFoundError,
    WeightedScoreComparator,
    create_model_registry,
)


def new_registry():
    return create_model_registry(deterministic=True)


def register(registry, version="1.0.0", model_id=None, **overrides):
    params = dict(
        metrics=ExperimentMetrics(f1=0.9, roc_auc=0.95, accuracy=0.92),
        latency_ms=40.0, memory_mb=200.0, model_size_bytes=1000,
        dataset_version="ds-1.0.0", run_id="run-1", experiment_id="exp-1",
    )
    params.update(overrides)
    return registry.register_model("fraud", version, model_id=model_id, **params)


def advance_to_staging(registry, model_id, version, approver="alice"):
    registry.promote(PromotionRequest(model_id=model_id, version=version,
                     from_stage=ModelStage.REGISTERED, to_stage=ModelStage.VALIDATION))
    registry.mark_validated(model_id, version, True)
    registry.promote(PromotionRequest(model_id=model_id, version=version,
                     from_stage=ModelStage.VALIDATION, to_stage=ModelStage.STAGING,
                     approved_by=approver))


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_register_returns_registered_stage():
    assert register(new_registry()).stage is ModelStage.REGISTERED


def test_register_creates_model_record():
    r = new_registry()
    mv = register(r)
    assert r.get_model(mv.model_id).name == "fraud"


def test_duplicate_version_rejected():
    r = new_registry()
    mv = register(r)
    with pytest.raises(DuplicateModelVersionError):
        register(r, model_id=mv.model_id)


@pytest.mark.parametrize("bad", ["1.0", "x", "1.0.0.0", "v1"])
def test_invalid_semver_rejected(bad):
    with pytest.raises(ValidationError):
        new_registry().register_model("fraud", bad)


def test_same_name_reuses_model_id():
    r = new_registry()
    a = r.register_model("dup", "1.0.0")
    b = r.register_model("dup", "1.1.0")
    assert a.model_id == b.model_id


def test_unknown_model_raises():
    with pytest.raises(ModelNotFoundError):
        new_registry().get_model("nope")


def test_unknown_version_raises():
    r = new_registry()
    mv = register(r)
    with pytest.raises(VersionNotFoundError):
        r.get_version(mv.model_id, "9.9.9")


# --------------------------------------------------------------------------- #
# Lifecycle transitions
# --------------------------------------------------------------------------- #
def test_promote_registered_to_validation():
    r = new_registry()
    mv = register(r)
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.REGISTERED, to_stage=ModelStage.VALIDATION))
    assert dec.approved
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.VALIDATION


def test_full_lifecycle_to_production():
    r = new_registry()
    mv = register(r)
    advance_to_staging(r, mv.model_id, "1.0.0")
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                    approved_by="alice", max_latency_ms=100, max_memory_mb=512))
    assert dec.approved
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.PRODUCTION
    assert r.get_model(mv.model_id).current_production_version == "1.0.0"


def test_stage_mismatch_rejected():
    r = new_registry()
    mv = register(r)
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                    approved_by="alice"))
    assert not dec.approved
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.REGISTERED


@pytest.mark.parametrize("stage", list(ModelStage))
def test_allowed_transitions_table_complete(stage):
    assert stage in ALLOWED_TRANSITIONS


def test_archived_has_no_transitions():
    assert ALLOWED_TRANSITIONS[ModelStage.ARCHIVED] == ()


# --------------------------------------------------------------------------- #
# Promotion gates
# --------------------------------------------------------------------------- #
def test_reject_without_validation():
    r = new_registry()
    mv = register(r)
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.REGISTERED, to_stage=ModelStage.VALIDATION))
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.VALIDATION, to_stage=ModelStage.STAGING,
                    approved_by="alice"))
    assert not dec.approved
    assert dict(dec.checks)["validation_complete"] is False


def test_reject_without_approval():
    r = new_registry()
    mv = register(r)
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.REGISTERED, to_stage=ModelStage.VALIDATION))
    r.mark_validated(mv.model_id, "1.0.0", True)
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.VALIDATION, to_stage=ModelStage.STAGING))
    assert not dec.approved
    assert dict(dec.checks)["approval_exists"] is False


def test_reject_on_latency():
    r = new_registry()
    mv = register(r, latency_ms=200.0)
    advance_to_staging(r, mv.model_id, "1.0.0")
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                    approved_by="alice", max_latency_ms=100))
    assert not dec.approved and dict(dec.checks)["latency_acceptable"] is False


def test_reject_on_memory():
    r = new_registry()
    mv = register(r, memory_mb=1024.0)
    advance_to_staging(r, mv.model_id, "1.0.0")
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                    from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                    approved_by="alice", max_memory_mb=512))
    assert not dec.approved and dict(dec.checks)["memory_acceptable"] is False


def test_reject_on_metric_regression():
    r = new_registry()
    mv = register(r, version="1.0.0", metrics=ExperimentMetrics(f1=0.9))
    advance_to_staging(r, mv.model_id, "1.0.0")
    register(r, version="1.1.0", model_id=mv.model_id, metrics=ExperimentMetrics(f1=0.7))
    advance_to_staging(r, mv.model_id, "1.1.0")
    dec = r.promote(PromotionRequest(model_id=mv.model_id, version="1.1.0",
                    from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                    approved_by="a", baseline_version="1.0.0", primary_metric="f1"))
    assert not dec.approved and dict(dec.checks)["metric_improved"] is False


def test_minimize_metric_direction():
    policy = DefaultPromotionPolicy()
    cand = ModelVersion(model_id="m", version="1.1.0", metrics=ExperimentMetrics(loss=0.2),
                        validation_passed=True)
    base = ModelVersion(model_id="m", version="1.0.0", metrics=ExperimentMetrics(loss=0.3))
    req = PromotionRequest(model_id="m", version="1.1.0", from_stage=ModelStage.STAGING,
                           to_stage=ModelStage.PRODUCTION, approved_by="a", baseline_version="1.0.0",
                           primary_metric="loss", primary_metric_direction=MetricDirection.MINIMIZE)
    dec = policy.evaluate(req, cand, base, "t")
    assert dict(dec.checks)["metric_improved"] is True


def test_evaluate_promotion_does_not_mutate():
    r = new_registry()
    mv = register(r)
    advance_to_staging(r, mv.model_id, "1.0.0")
    r.evaluate_promotion(PromotionRequest(model_id=mv.model_id, version="1.0.0",
                         from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION,
                         approved_by="a"))
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.STAGING


def test_production_promotion_archives_previous():
    r = new_registry()
    mv = register(r, version="1.0.0")
    advance_to_staging(r, mv.model_id, "1.0.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    register(r, version="1.1.0", model_id=mv.model_id, metrics=ExperimentMetrics(f1=0.95))
    advance_to_staging(r, mv.model_id, "1.1.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.1.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.ARCHIVED
    assert r.get_version(mv.model_id, "1.1.0").stage is ModelStage.PRODUCTION


# --------------------------------------------------------------------------- #
# Rollback & archive
# --------------------------------------------------------------------------- #
def test_rollback_restores_previous():
    r = new_registry()
    mv = register(r, version="1.0.0")
    advance_to_staging(r, mv.model_id, "1.0.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    register(r, version="1.1.0", model_id=mv.model_id, metrics=ExperimentMetrics(f1=0.95))
    advance_to_staging(r, mv.model_id, "1.1.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.1.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    r.rollback(mv.model_id, "1.0.0")
    assert r.get_version(mv.model_id, "1.0.0").stage is ModelStage.PRODUCTION
    assert r.get_version(mv.model_id, "1.1.0").stage is ModelStage.ARCHIVED
    assert r.get_model(mv.model_id).current_production_version == "1.0.0"


def test_archive_version():
    r = new_registry()
    mv = register(r)
    archived = r.archive(mv.model_id, "1.0.0")
    assert archived.stage is ModelStage.ARCHIVED


def test_archive_twice_rejected():
    r = new_registry()
    mv = register(r)
    r.archive(mv.model_id, "1.0.0")
    with pytest.raises(InvalidStageTransitionError):
        r.archive(mv.model_id, "1.0.0")


def test_archive_clears_production_pointer():
    r = new_registry()
    mv = register(r)
    advance_to_staging(r, mv.model_id, "1.0.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    r.archive(mv.model_id, "1.0.0")
    assert r.get_model(mv.model_id).current_production_version is None


# --------------------------------------------------------------------------- #
# latest / history / versions
# --------------------------------------------------------------------------- #
def test_history_sorted_ascending():
    r = new_registry()
    mv = register(r, version="1.0.0")
    register(r, version="2.0.0", model_id=mv.model_id)
    register(r, version="1.5.0", model_id=mv.model_id)
    assert [v.version for v in r.history(mv.model_id)] == ["1.0.0", "1.5.0", "2.0.0"]


def test_latest_returns_highest():
    r = new_registry()
    mv = register(r, version="1.0.0")
    register(r, version="2.3.0", model_id=mv.model_id)
    register(r, version="2.2.9", model_id=mv.model_id)
    assert r.latest(mv.model_id).version == "2.3.0"


def test_latest_by_stage():
    r = new_registry()
    mv = register(r, version="1.0.0")
    advance_to_staging(r, mv.model_id, "1.0.0")
    register(r, version="1.1.0", model_id=mv.model_id)
    assert r.latest(mv.model_id, stage=ModelStage.STAGING).version == "1.0.0"


def test_latest_missing_stage_raises():
    r = new_registry()
    mv = register(r)
    with pytest.raises(VersionNotFoundError):
        r.latest(mv.model_id, stage=ModelStage.PRODUCTION)


def test_list_models_sorted():
    r = new_registry()
    r.register_model("b-model", "1.0.0", model_id="m-b")
    r.register_model("a-model", "1.0.0", model_id="m-a")
    assert [m.model_id for m in r.list_models()] == ["m-a", "m-b"]


# --------------------------------------------------------------------------- #
# Comparison engine
# --------------------------------------------------------------------------- #
def test_compare_versions_ranks_best_first():
    r = new_registry()
    mv = register(r, version="1.0.0", metrics=ExperimentMetrics(f1=0.80, roc_auc=0.85, accuracy=0.9),
                  latency_ms=50, memory_mb=300)
    register(r, version="1.1.0", model_id=mv.model_id,
             metrics=ExperimentMetrics(f1=0.92, roc_auc=0.96, accuracy=0.94),
             latency_ms=40, memory_mb=280)
    result = r.compare_versions(mv.model_id)
    assert result["count"] == 2
    assert result["best_version"] == "1.1.0"
    assert result["ranking"][0]["rank"] == 1


def test_compare_is_deterministic():
    r = new_registry()
    mv = register(r, version="1.0.0", metrics=ExperimentMetrics(f1=0.8))
    register(r, version="1.1.0", model_id=mv.model_id, metrics=ExperimentMetrics(f1=0.9))
    first = r.compare_versions(mv.model_id)
    second = r.compare_versions(mv.model_id)
    assert first == second


def test_compare_specific_versions():
    r = new_registry()
    mv = register(r, version="1.0.0")
    register(r, version="1.1.0", model_id=mv.model_id)
    register(r, version="1.2.0", model_id=mv.model_id)
    result = r.compare_versions(mv.model_id, versions=("1.0.0", "1.2.0"))
    assert result["count"] == 2


def test_compare_empty_list():
    assert new_registry().compare([])["count"] == 0


def test_comparator_score_monotonic():
    cmp = WeightedScoreComparator()
    low = ModelVersion(model_id="m", version="1.0.0", metrics=ExperimentMetrics(f1=0.5, roc_auc=0.5, accuracy=0.5))
    high = ModelVersion(model_id="m", version="1.0.0", metrics=ExperimentMetrics(f1=0.9, roc_auc=0.9, accuracy=0.9))
    assert cmp.score(high) > cmp.score(low)


def test_comparator_rank_orders_by_quality():
    cmp = WeightedScoreComparator()
    a = ModelVersion(model_id="m", version="1.0.0", metrics=ExperimentMetrics(f1=0.6, roc_auc=0.6, accuracy=0.6))
    b = ModelVersion(model_id="m", version="1.1.0", metrics=ExperimentMetrics(f1=0.9, roc_auc=0.9, accuracy=0.9))
    ranked = cmp.rank([a, b])
    assert ranked[0][0].version == "1.1.0"


# --------------------------------------------------------------------------- #
# Lineage & model cards
# --------------------------------------------------------------------------- #
def test_lineage_chain():
    r = new_registry()
    mv = register(r, dataset_version="ds-9", run_id="run-9", experiment_id="exp-9")
    lineage = r.lineage(mv.model_id, "1.0.0")
    assert lineage["dataset_version"] == "ds-9"
    assert lineage["run_id"] == "run-9"
    assert lineage["experiment_id"] == "exp-9"


def test_generate_model_card():
    r = new_registry()
    mv = register(r, metrics=ExperimentMetrics(f1=0.91))
    card = r.generate_model_card(mv.model_id, "1.0.0", purpose="Detect fraud",
                                 limitations=("Region-specific",),
                                 responsible_ai_notes=("Monitor drift",),
                                 dependencies={"numpy": "2.4.4"})
    assert card.purpose == "Detect fraud"
    assert card.metrics.f1 == 0.91
    assert card.limitations == ("Region-specific",)
    assert dict(card.dependencies)["numpy"] == "2.4.4"


def test_model_card_json_roundtrip():
    r = new_registry()
    mv = register(r)
    card = r.generate_model_card(mv.model_id, "1.0.0", purpose="p")
    from mlops.experiment_models import ModelCard
    assert ModelCard.from_dict(json.loads(json.dumps(card.to_dict()))) == card


# --------------------------------------------------------------------------- #
# Statistics & serialisation
# --------------------------------------------------------------------------- #
def test_statistics_counts():
    r = new_registry()
    mv = register(r, version="1.0.0")
    register(r, version="1.1.0", model_id=mv.model_id)
    stats = r.statistics()
    assert stats.total_models == 1
    assert stats.total_versions == 2
    assert dict(stats.versions_by_stage)["REGISTERED"] == 2


def test_statistics_production_count():
    r = new_registry()
    mv = register(r)
    advance_to_staging(r, mv.model_id, "1.0.0")
    r.promote(PromotionRequest(model_id=mv.model_id, version="1.0.0",
              from_stage=ModelStage.STAGING, to_stage=ModelStage.PRODUCTION, approved_by="a"))
    assert r.statistics().production_models == 1


def test_registry_json_roundtrip():
    r = new_registry()
    mv = register(r, version="1.0.0")
    register(r, version="1.1.0", model_id=mv.model_id)
    reloaded = ModelRegistry.from_dict(json.loads(r.to_json()))
    assert reloaded.to_dict() == r.to_dict()


def test_export_registry_file():
    import tempfile, os
    r = new_registry()
    register(r)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "registry.json")
        r.export_json(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    assert "models" in data and "versions" in data


# --------------------------------------------------------------------------- #
# Scale & thread-safety
# --------------------------------------------------------------------------- #
def test_large_registry():
    r = new_registry()
    mid = register(r, version="1.0.0").model_id
    for i in range(1, 200):
        register(r, version=f"1.{i}.0", model_id=mid)
    assert r.statistics().total_versions == 200
    assert r.latest(mid).version == "1.199.0"


def test_thread_safe_registration():
    r = new_registry()
    mid = register(r, version="1.0.0").model_id

    def worker(base):
        for i in range(20):
            register(r, version=f"{base}.{i}.0", model_id=mid)

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(2, 8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert r.statistics().total_versions == 1 + 6 * 20