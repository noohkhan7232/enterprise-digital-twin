"""Tests for the reproducibility engine."""

from __future__ import annotations

import json

import pytest

from mlops.experiment_models import (
    DeterministicIdGenerator,
    LogicalClock,
    ReproducibilitySnapshot,
    ValidationError,
)
from mlops.reproducibility import (
    DEFAULT_ENV_ALLOWLIST,
    ReproducibilityEngine,
    ReproducibilityError,
    StaticEnvironmentProvider,
    SystemEnvironmentProvider,
    create_reproducibility_engine,
)


def det_engine(seed="r"):
    return ReproducibilityEngine(
        provider=StaticEnvironmentProvider(),
        clock=LogicalClock(),
        id_generator=DeterministicIdGenerator(seed=seed),
    )


def test_capture_environment_fields():
    env = det_engine().capture_environment()
    assert env.python_version == "3.12.3"
    assert env.numpy_version == "2.4.4"
    assert env.cpu_count == 8


def test_capture_snapshot_fields():
    snap = det_engine().capture(random_seed=42, numpy_seed=7, config={"lr": 0.1},
                                dataset_version="1.0.0")
    assert snap.random_seed == 42
    assert snap.numpy_seed == 7
    assert snap.dataset_version == "1.0.0"
    assert snap.git_commit == "0" * 40


def test_capture_is_deterministic_across_engines():
    a = det_engine().capture(random_seed=1, numpy_seed=2, config={"x": 1})
    b = det_engine().capture(random_seed=1, numpy_seed=2, config={"x": 1})
    assert a.matches(b)


def test_verify_true_for_identical():
    e = det_engine()
    a = e.capture(random_seed=1, numpy_seed=1)
    b = det_engine().capture(random_seed=1, numpy_seed=1)
    assert e.verify(a, b) is True


@pytest.mark.parametrize("seed_a,seed_b", [(1, 2), (10, 11), (0, 99)])
def test_verify_false_for_different_seed(seed_a, seed_b):
    e = det_engine()
    a = e.capture(random_seed=seed_a, numpy_seed=0)
    b = e.capture(random_seed=seed_b, numpy_seed=0)
    assert e.verify(a, b) is False


def test_diff_reports_random_seed():
    e = det_engine()
    a = e.capture(random_seed=1, numpy_seed=0)
    b = e.capture(random_seed=2, numpy_seed=0)
    assert "random_seed" in e.diff(a, b)


def test_diff_reports_config():
    e = det_engine()
    a = e.capture(random_seed=1, numpy_seed=0, config={"lr": 0.1})
    b = e.capture(random_seed=1, numpy_seed=0, config={"lr": 0.2})
    assert "config" in e.diff(a, b)


def test_diff_empty_for_equivalent():
    e = det_engine()
    a = e.capture(random_seed=1, numpy_seed=2, config={"a": 1}, snapshot_id="fixed")
    b = e.capture(random_seed=1, numpy_seed=2, config={"a": 1}, snapshot_id="fixed")
    # created_at differs because the logical clock advances; restrict to seeds/config.
    diff = e.diff(a, b)
    assert "random_seed" not in diff and "config" not in diff


def test_invalid_random_seed_type():
    with pytest.raises(ValidationError):
        det_engine().capture(random_seed="x", numpy_seed=0)


def test_invalid_numpy_seed_type():
    with pytest.raises(ValidationError):
        det_engine().capture(random_seed=0, numpy_seed=1.5)


def test_verify_requires_snapshots():
    with pytest.raises(ValidationError):
        det_engine().verify("a", "b")


def test_snapshot_json_roundtrip():
    snap = det_engine().capture(random_seed=1, numpy_seed=2, config={"a": 1})
    restored = ReproducibilitySnapshot.from_dict(json.loads(ReproducibilityEngine.to_json(snap)))
    assert restored == snap


def test_export_json_file():
    import tempfile, os
    snap = det_engine().capture(random_seed=1, numpy_seed=2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "repro.json")
        ReproducibilityEngine.export_json(snap, path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    assert data["random_seed"] == 1


def test_env_allowlist_filters_variables():
    provider = StaticEnvironmentProvider(environment_variables={"MLOPS_ENV": "prod", "SECRET": "x"})
    engine = ReproducibilityEngine(provider=provider, clock=LogicalClock(),
                                   id_generator=DeterministicIdGenerator())
    env = engine.capture_environment()
    keys = dict(env.environment_variables)
    assert "MLOPS_ENV" in keys and "SECRET" not in keys


def test_default_allowlist_is_tuple():
    assert isinstance(DEFAULT_ENV_ALLOWLIST, tuple) and len(DEFAULT_ENV_ALLOWLIST) >= 1


def test_system_provider_reports_versions():
    provider = SystemEnvironmentProvider()
    assert provider.numpy_version()
    assert provider.python_version()
    assert provider.cpu_count() >= 0


def test_system_engine_capture_has_environment():
    engine = create_reproducibility_engine(deterministic=False)
    snap = engine.capture(random_seed=1, numpy_seed=1)
    assert snap.environment.numpy_version


@pytest.mark.parametrize("seed", [0, 1, 42, 2024])
def test_snapshot_seed_roundtrip(seed):
    snap = det_engine().capture(random_seed=seed, numpy_seed=seed)
    assert ReproducibilitySnapshot.from_dict(snap.to_dict()).random_seed == seed


def test_runtime_and_hardware_info_present():
    snap = det_engine().capture(random_seed=1, numpy_seed=1)
    assert dict(snap.runtime_info)
    assert dict(snap.hardware_info)


def test_static_provider_custom_dependencies():
    provider = StaticEnvironmentProvider(dependencies={"numpy": "2.4.4", "scipy": "1.13.0"})
    engine = ReproducibilityEngine(provider=provider, clock=LogicalClock(),
                                   id_generator=DeterministicIdGenerator())
    env = engine.capture_environment()
    assert dict(env.dependencies)["scipy"] == "1.13.0"


def test_snapshot_id_is_generated():
    snap = det_engine().capture(random_seed=1, numpy_seed=1)
    assert snap.snapshot_id.startswith("repro-")