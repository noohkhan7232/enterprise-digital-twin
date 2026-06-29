"""Tests for the immutable MLOps domain models and deterministic infrastructure."""

from __future__ import annotations

import dataclasses
import json

import pytest

from mlops.experiment_models import (
    ArtifactMetadata,
    ArtifactType,
    DatasetVersion,
    DeterministicIdGenerator,
    EnvironmentSnapshot,
    Experiment,
    ExperimentMetadata,
    ExperimentMetrics,
    ExperimentRun,
    ExperimentStatistics,
    ExperimentStatus,
    FixedClock,
    HyperParameters,
    LogicalClock,
    MetricDirection,
    ModelArtifact,
    ModelCard,
    ModelStage,
    ModelVersion,
    PromotionDecision,
    PromotionRequest,
    PromotionStatus,
    RegisteredModel,
    RegistryStatistics,
    ReproducibilitySnapshot,
    SemanticVersion,
    SequentialIdGenerator,
    SerializationError,
    ValidationError,
    freeze_mapping,
    thaw_mapping,
)


# --------------------------------------------------------------------------- #
# Representative instances for generic round-trip coverage.
# --------------------------------------------------------------------------- #
def all_instances():
    return [
        HyperParameters({"lr": 0.01, "depth": 6, "shuffle": True}),
        ExperimentMetrics(accuracy=0.9, f1=0.88, additional={"mcc": 0.7}),
        ExperimentMetadata(dataset_version="1.0.0", tags=["a", "b"], random_seed=42),
        Experiment(experiment_id="exp-1", name="demo", tags=["x"]),
        ExperimentRun(run_id="run-1", experiment_id="exp-1"),
        ArtifactMetadata(artifact_id="a-1", artifact_type=ArtifactType.ROC_CURVE, name="roc"),
        ModelArtifact(artifact_id="a-1", name="model"),
        DatasetVersion(dataset_id="d-1", name="ds", version="2.1.0", row_count=1000),
        EnvironmentSnapshot(python_version="3.12.3", dependencies={"numpy": "2.4.4"}),
        ModelCard(model_id="m-1", model_name="M", version="1.0.0"),
        RegisteredModel(model_id="m-1", name="M", version_ids=["1.0.0"]),
        ModelVersion(model_id="m-1", version="1.2.3", metrics=ExperimentMetrics(f1=0.9)),
        PromotionRequest(model_id="m-1", version="1.0.0", from_stage=ModelStage.STAGING,
                         to_stage=ModelStage.PRODUCTION),
        PromotionDecision(model_id="m-1", version="1.0.0", from_stage=ModelStage.STAGING,
                          to_stage=ModelStage.PRODUCTION, status=PromotionStatus.APPROVED),
        RegistryStatistics(total_models=2, versions_by_stage={"PRODUCTION": 1}),
        ExperimentStatistics(total_runs=3, completed_runs=2),
        ReproducibilitySnapshot(snapshot_id="s-1", random_seed=1, numpy_seed=2),
    ]


INSTANCES = all_instances()


@pytest.mark.parametrize("obj", INSTANCES)
def test_to_dict_from_dict_roundtrip(obj):
    restored = type(obj).from_dict(obj.to_dict())
    assert restored == obj


@pytest.mark.parametrize("obj", INSTANCES)
def test_json_roundtrip(obj):
    restored = type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))
    assert restored == obj


@pytest.mark.parametrize("obj", INSTANCES)
def test_instances_are_hashable(obj):
    assert isinstance(hash(obj), int)


@pytest.mark.parametrize("obj", INSTANCES)
def test_instances_are_frozen(obj):
    field_name = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field_name, "mutated")


@pytest.mark.parametrize("obj", INSTANCES)
def test_to_dict_is_json_serialisable(obj):
    assert isinstance(json.dumps(obj.to_dict()), str)


@pytest.mark.parametrize("obj", INSTANCES)
def test_equal_objects_have_equal_hash(obj):
    twin = type(obj).from_dict(obj.to_dict())
    assert hash(twin) == hash(obj)


# --------------------------------------------------------------------------- #
# SemanticVersion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,major,minor,patch,pre",
    [
        ("1.0.0", 1, 0, 0, None),
        ("0.0.1", 0, 0, 1, None),
        ("10.20.30", 10, 20, 30, None),
        ("2.3.1-rc.1", 2, 3, 1, "rc.1"),
        ("1.0.0-alpha", 1, 0, 0, "alpha"),
        ("12.0.0-beta.2", 12, 0, 0, "beta.2"),
    ],
)
def test_semver_parse_valid(text, major, minor, patch, pre):
    sv = SemanticVersion.parse(text)
    assert (sv.major, sv.minor, sv.patch, sv.prerelease) == (major, minor, patch, pre)


@pytest.mark.parametrize(
    "text", ["1.0", "1", "1.0.0.0", "v1.0.0", "01.0.0", "1.0.0-", "abc", "", "1.0.x"]
)
def test_semver_parse_invalid(text):
    with pytest.raises(ValidationError):
        SemanticVersion.parse(text)


@pytest.mark.parametrize(
    "text,valid",
    [("1.0.0", True), ("2.3.4-rc.1", True), ("1.0", False), ("x", False), ("1.0.0.0", False)],
)
def test_semver_is_valid(text, valid):
    assert SemanticVersion.is_valid(text) is valid


@pytest.mark.parametrize(
    "lower,higher",
    [
        ("1.0.0", "2.0.0"),
        ("1.0.0", "1.1.0"),
        ("1.0.0", "1.0.1"),
        ("1.0.0-rc.1", "1.0.0"),
        ("1.0.0-alpha", "1.0.0-beta"),
        ("1.9.9", "2.0.0"),
    ],
)
def test_semver_ordering(lower, higher):
    assert SemanticVersion.parse(lower) < SemanticVersion.parse(higher)
    assert SemanticVersion.parse(higher) > SemanticVersion.parse(lower)


def test_semver_equality():
    assert SemanticVersion.parse("1.2.3") == SemanticVersion(1, 2, 3)


@pytest.mark.parametrize(
    "start,method,expected",
    [
        ("1.2.3", "bump_major", "2.0.0"),
        ("1.2.3", "bump_minor", "1.3.0"),
        ("1.2.3", "bump_patch", "1.2.4"),
    ],
)
def test_semver_bump(start, method, expected):
    sv = SemanticVersion.parse(start)
    assert str(getattr(sv, method)()) == expected


@pytest.mark.parametrize("text", ["1.0.0", "2.3.1-rc.1", "0.0.0", "99.99.99"])
def test_semver_str_roundtrip(text):
    assert str(SemanticVersion.parse(text)) == text


@pytest.mark.parametrize("text", ["1.0.0", "2.3.1-rc.1"])
def test_semver_dict_roundtrip(text):
    sv = SemanticVersion.parse(text)
    assert SemanticVersion.from_dict(sv.to_dict()) == sv


def test_semver_negative_rejected():
    with pytest.raises(ValidationError):
        SemanticVersion(-1, 0, 0)


def test_semver_from_dict_invalid():
    with pytest.raises(SerializationError):
        SemanticVersion.from_dict({"major": "x", "minor": 0, "patch": 0})


# --------------------------------------------------------------------------- #
# Mapping helpers
# --------------------------------------------------------------------------- #
def test_freeze_mapping_sorted_and_hashable():
    frozen = freeze_mapping({"b": 1, "a": 2})
    assert frozen == (("a", 2), ("b", 1))
    assert isinstance(hash(frozen), int)


def test_freeze_mapping_none():
    assert freeze_mapping(None) == ()


def test_thaw_mapping_roundtrip():
    data = {"a": 1, "b": "two", "c": True}
    assert thaw_mapping(freeze_mapping(data)) == data


def test_freeze_mapping_rejects_non_scalar():
    with pytest.raises(ValidationError):
        freeze_mapping({"a": [1, 2, 3]})


@pytest.mark.parametrize("value", [1, "s", 3.14, True, None])
def test_freeze_mapping_accepts_scalars(value):
    assert thaw_mapping(freeze_mapping({"k": value})) == {"k": value}


# --------------------------------------------------------------------------- #
# HyperParameters
# --------------------------------------------------------------------------- #
def test_hyperparameters_get():
    hp = HyperParameters({"lr": 0.1, "depth": 5})
    assert hp.get("lr") == 0.1
    assert hp.get("missing", 99) == 99


def test_hyperparameters_with_value_is_immutable():
    hp = HyperParameters({"lr": 0.1})
    hp2 = hp.with_value("depth", 7)
    assert hp.get("depth") is None
    assert hp2.get("depth") == 7


def test_hyperparameters_as_dict():
    assert HyperParameters({"a": 1}).as_dict() == {"a": 1}


def test_hyperparameters_empty():
    assert HyperParameters().as_dict() == {}


# --------------------------------------------------------------------------- #
# ExperimentMetrics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "loss"])
def test_metrics_standard_get(name):
    metrics = ExperimentMetrics(**{name: 0.5})
    assert metrics.get(name) == 0.5


def test_metrics_additional_get():
    metrics = ExperimentMetrics(additional={"mcc": 0.42})
    assert metrics.get("mcc") == 0.42
    assert metrics.get("absent") is None


def test_metrics_as_dict_merges():
    metrics = ExperimentMetrics(accuracy=0.9, additional={"mcc": 0.4})
    data = metrics.as_dict()
    assert data["accuracy"] == 0.9 and data["mcc"] == 0.4


def test_metrics_approx():
    metrics = ExperimentMetrics(f1=0.333333)
    assert metrics.f1 == pytest.approx(0.333333)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage", list(ModelStage))
def test_model_stage_value_roundtrip(stage):
    assert ModelStage(stage.value) is stage


@pytest.mark.parametrize("status", list(ExperimentStatus))
def test_experiment_status_roundtrip(status):
    assert ExperimentStatus(status.value) is status


@pytest.mark.parametrize("atype", list(ArtifactType))
def test_artifact_type_roundtrip(atype):
    assert ArtifactType(atype.value) is atype


@pytest.mark.parametrize("direction", list(MetricDirection))
def test_metric_direction_roundtrip(direction):
    assert MetricDirection(direction.value) is direction


def test_artifact_type_has_ten_members():
    assert len(list(ArtifactType)) == 10


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #
def test_experiment_requires_id():
    with pytest.raises(ValidationError):
        Experiment(experiment_id="", name="x")


def test_experiment_requires_name():
    with pytest.raises(ValidationError):
        Experiment(experiment_id="e", name="  ")


def test_run_requires_ids():
    with pytest.raises(ValidationError):
        ExperimentRun(run_id="", experiment_id="e")


@pytest.mark.parametrize("field", ["training_time", "inference_time", "memory_usage", "cpu_usage"])
def test_run_rejects_negative_resources(field):
    with pytest.raises(ValidationError):
        ExperimentRun(run_id="r", experiment_id="e", **{field: -1.0})


def test_artifact_metadata_requires_name():
    with pytest.raises(ValidationError):
        ArtifactMetadata(artifact_id="a", artifact_type=ArtifactType.ROC_CURVE, name="")


def test_artifact_metadata_negative_size_rejected():
    with pytest.raises(ValidationError):
        ArtifactMetadata(artifact_id="a", artifact_type=ArtifactType.ROC_CURVE, name="n", size_bytes=-5)


def test_dataset_version_invalid_semver():
    with pytest.raises(ValidationError):
        DatasetVersion(dataset_id="d", name="ds", version="not-semver")


def test_model_version_invalid_semver():
    with pytest.raises(ValidationError):
        ModelVersion(model_id="m", version="1.0")


def test_model_card_invalid_semver():
    with pytest.raises(ValidationError):
        ModelCard(model_id="m", model_name="M", version="x")


@pytest.mark.parametrize("field", ["latency_ms", "memory_mb", "model_size_bytes",
                                    "training_time_s", "inference_time_ms"])
def test_model_version_rejects_negative_perf(field):
    with pytest.raises(ValidationError):
        ModelVersion(model_id="m", version="1.0.0", **{field: -1})


# --------------------------------------------------------------------------- #
# Behavioural helpers on models
# --------------------------------------------------------------------------- #
def test_experiment_with_run_appends_once():
    exp = Experiment(experiment_id="e", name="n")
    exp2 = exp.with_run("r1").with_run("r1")
    assert exp2.run_ids == ("r1",)


def test_experiment_with_status():
    exp = Experiment(experiment_id="e", name="n")
    assert exp.with_status(ExperimentStatus.RUNNING).status is ExperimentStatus.RUNNING


def test_run_with_artifact():
    run = ExperimentRun(run_id="r", experiment_id="e")
    assert run.with_artifact("a1").artifact_ids == ("a1",)


def test_registered_model_with_version():
    model = RegisteredModel(model_id="m", name="M")
    assert model.with_version("1.0.0").version_ids == ("1.0.0",)


def test_model_version_semantic_version_property():
    assert ModelVersion(model_id="m", version="1.2.3").semantic_version == SemanticVersion(1, 2, 3)


def test_model_version_lineage_contents():
    mv = ModelVersion(model_id="m", version="1.0.0", dataset_version="d-1",
                      experiment_id="e", run_id="r", artifact_ids=("a1",))
    lineage = mv.lineage()
    assert lineage["dataset_version"] == "d-1"
    assert lineage["experiment_id"] == "e"
    assert lineage["run_id"] == "r"
    assert lineage["artifact_ids"] == ["a1"]
    assert lineage["stage"] == "REGISTERED"


def test_promotion_request_is_approved():
    req = PromotionRequest(model_id="m", version="1.0.0", from_stage=ModelStage.STAGING,
                           to_stage=ModelStage.PRODUCTION, approved_by="alice")
    assert req.is_approved is True


def test_promotion_request_not_approved():
    req = PromotionRequest(model_id="m", version="1.0.0", from_stage=ModelStage.STAGING,
                           to_stage=ModelStage.PRODUCTION)
    assert req.is_approved is False


def test_promotion_decision_approved_property():
    dec = PromotionDecision(model_id="m", version="1.0.0", from_stage=ModelStage.STAGING,
                            to_stage=ModelStage.PRODUCTION, status=PromotionStatus.APPROVED)
    assert dec.approved is True


# --------------------------------------------------------------------------- #
# ReproducibilitySnapshot diff/matches
# --------------------------------------------------------------------------- #
def test_repro_snapshot_matches_ignores_id_and_time():
    env = EnvironmentSnapshot(python_version="3.12.3")
    a = ReproducibilitySnapshot(snapshot_id="s1", environment=env, random_seed=1, created_at="t1")
    b = ReproducibilitySnapshot(snapshot_id="s2", environment=env, random_seed=1, created_at="t2")
    assert a.matches(b)


def test_repro_snapshot_diff_detects_seed():
    a = ReproducibilitySnapshot(snapshot_id="s1", random_seed=1)
    b = ReproducibilitySnapshot(snapshot_id="s1", random_seed=2)
    assert "random_seed" in a.diff(b)


def test_repro_snapshot_requires_id():
    with pytest.raises(ValidationError):
        ReproducibilitySnapshot(snapshot_id="")


# --------------------------------------------------------------------------- #
# Deterministic infrastructure
# --------------------------------------------------------------------------- #
def test_fixed_clock_constant():
    clock = FixedClock("2024-05-01T00:00:00+00:00")
    assert clock.now() == clock.now() == "2024-05-01T00:00:00+00:00"


def test_logical_clock_monotonic():
    clock = LogicalClock()
    first, second, third = clock.now(), clock.now(), clock.now()
    assert first < second < third


def test_sequential_id_generator_format():
    gen = SequentialIdGenerator()
    assert gen.generate("exp") == "exp-000001"
    assert gen.generate("exp") == "exp-000002"
    assert gen.generate("run") == "run-000001"


def test_deterministic_id_generator_reproducible():
    a = DeterministicIdGenerator(seed="s")
    b = DeterministicIdGenerator(seed="s")
    assert [a.generate("run") for _ in range(4)] == [b.generate("run") for _ in range(4)]


def test_deterministic_id_generator_seed_sensitive():
    a = DeterministicIdGenerator(seed="s1")
    b = DeterministicIdGenerator(seed="s2")
    assert a.generate("run") != b.generate("run")


def test_logical_clock_invalid_step():
    with pytest.raises(ValidationError):
        LogicalClock(step_seconds=0)


def test_deterministic_id_invalid_length():
    with pytest.raises(ValidationError):
        DeterministicIdGenerator(length=0)


# --------------------------------------------------------------------------- #
# Backward-compatibility: from_dict tolerates missing optional keys
# --------------------------------------------------------------------------- #
def test_experiment_from_minimal_dict():
    exp = Experiment.from_dict({"experiment_id": "e", "name": "n"})
    assert exp.status is ExperimentStatus.CREATED


def test_run_from_minimal_dict():
    run = ExperimentRun.from_dict({"run_id": "r", "experiment_id": "e"})
    assert run.training_time == 0.0


def test_model_version_from_minimal_dict():
    mv = ModelVersion.from_dict({"model_id": "m", "version": "1.0.0"})
    assert mv.stage is ModelStage.REGISTERED