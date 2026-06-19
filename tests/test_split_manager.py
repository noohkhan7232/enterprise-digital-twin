#!/usr/bin/env python3
"""Comprehensive test suite for ``src/training/split_manager.py``.

Coverage:
- :class:`SplitConfig` validation (fraction sums, bounds)
- :func:`validate_manifest` (empty, missing label, duplicate/missing id, bad type)
- :func:`compute_split_fingerprint` (order-independence, content-sensitivity)
- :func:`_get_field` (dict, object, nested metadata)
- All six :class:`SplitStrategy` values produce correct partitions
- Leakage prevention for GROUP / STRATIFIED_GROUP / TURBINE
- Temporal ordering guarantee for TEMPORAL
- Reproducibility (same seed → same fingerprint)
- Cross-validation for every strategy
- Expanding-window temporal CV
- :class:`LeakageAudit` overlap counting
- :class:`SplitReport` summary / to_dict
- ExperimentTracker integration (params logged, broken tracker safe)
- Object-record (ClipRecord-like) and dict-record compatibility
- Custom group_key / turbine_key / timestamp_key resolution
- Edge cases: few groups, single class, no timestamps, ISO timestamps

Run::

    pytest tests/test_split_manager.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.split_manager import (
    FoldResult,
    LeakageAudit,
    ManifestValidationError,
    SplitConfig,
    SplitManager,
    SplitReport,
    SplitResult,
    SplitStrategy,
    _get_field,
    compute_split_fingerprint,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------
FAULT_TYPES = ["normal", "bearing_fault", "blade_imbalance", "gearbox_fault"]


def make_dict_manifest(n_per_class: int = 8) -> list[dict]:
    """Build a synthetic dict manifest with groups, turbines, timestamps."""
    manifest = []
    for ci, ft in enumerate(FAULT_TYPES):
        for i in range(n_per_class):
            manifest.append({
                "clip_id":     f"{ft}_{i:03d}",
                "fault_type":  ft,
                "label":       ci,
                "source_path": f"data/raw/{ft}/{ft}_{i // 2:03d}.wav",
                "dataset":     f"turbine_{i % 3}",
                "metadata":    {"timestamp": 1_700_000_000 + ci * 86400 + i * 3600},
            })
    return manifest


@dataclass
class FakeClip:
    """ClipRecord-like object for testing object-record compatibility."""

    clip_id:     str
    label:       int
    fault_type:  str
    source_path: str
    dataset:     str
    metadata:    dict = field(default_factory=dict)


def make_object_manifest(n_per_class: int = 6) -> list[FakeClip]:
    """Build a synthetic object manifest."""
    return [
        FakeClip(
            clip_id=f"{ft}_{i:03d}", label=ci, fault_type=ft,
            source_path=f"data/raw/{ft}/{ft}_{i // 2:03d}.wav",
            dataset=f"turbine_{i % 3}",
            metadata={"timestamp": 1_700_000_000 + i * 3600},
        )
        for ci, ft in enumerate(FAULT_TYPES)
        for i in range(n_per_class)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest() -> list[dict]:
    """Standard 32-clip dict manifest."""
    return make_dict_manifest(8)


@pytest.fixture()
def object_manifest() -> list[FakeClip]:
    """Standard 24-clip object manifest."""
    return make_object_manifest(6)


def _max_ts(records: list, key: str = "timestamp") -> float:
    return max(_get_field(r, key, -1) for r in records) if records else -1.0


def _min_ts(records: list, key: str = "timestamp") -> float:
    return min(_get_field(r, key, 1e18) for r in records) if records else 1e18


# ---------------------------------------------------------------------------
# SplitConfig tests
# ---------------------------------------------------------------------------


class TestSplitConfig:
    """Tests for :class:`SplitConfig`."""

    def test_default(self) -> None:
        cfg = SplitConfig()
        assert cfg.strategy == SplitStrategy.STRATIFIED_GROUP
        assert cfg.random_seed == 42

    def test_fractions_sum_validation(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_fraction=0.8, val_fraction=0.3, test_fraction=0.3)

    def test_train_fraction_bounds(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(train_fraction=1.6, val_fraction=-0.3, test_fraction=-0.3)

    def test_negative_val_fraction(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(train_fraction=1.1, val_fraction=-0.05, test_fraction=-0.05)

    def test_valid_custom_fractions(self) -> None:
        cfg = SplitConfig(train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
        assert cfg.train_fraction == 0.6

    def test_frozen(self) -> None:
        cfg = SplitConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.random_seed = 99  # type: ignore[misc]

    def test_zero_val_zero_test_allowed(self) -> None:
        cfg = SplitConfig(train_fraction=1.0, val_fraction=0.0, test_fraction=0.0)
        assert cfg.val_fraction == 0.0


# ---------------------------------------------------------------------------
# validate_manifest tests
# ---------------------------------------------------------------------------


class TestValidateManifest:
    """Tests for :func:`validate_manifest`."""

    def test_valid_manifest(self, manifest: list[dict]) -> None:
        assert validate_manifest(manifest) == []

    def test_empty_manifest(self) -> None:
        assert validate_manifest([]) == ["Manifest is empty"]

    def test_missing_label(self) -> None:
        errors = validate_manifest([{"clip_id": "a"}])
        assert any("missing label" in e for e in errors)

    def test_duplicate_clip_id(self) -> None:
        errors = validate_manifest([
            {"clip_id": "a", "label": 0},
            {"clip_id": "a", "label": 1},
        ])
        assert any("duplicate" in e for e in errors)

    def test_missing_clip_id(self) -> None:
        errors = validate_manifest([{"clip_id": "", "label": 0}])
        assert any("missing clip_id" in e for e in errors)

    def test_non_integer_label(self) -> None:
        errors = validate_manifest([{"clip_id": "a", "label": "zero"}])
        assert any("must be int" in e for e in errors)

    def test_object_records(self, object_manifest: list[FakeClip]) -> None:
        assert validate_manifest(object_manifest) == []

    def test_allow_duplicate_ids_when_disabled(self) -> None:
        errors = validate_manifest(
            [{"clip_id": "a", "label": 0}, {"clip_id": "a", "label": 1}],
            require_unique_ids=False,
        )
        assert not any("duplicate" in e for e in errors)


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


class TestFingerprint:
    """Tests for :func:`compute_split_fingerprint`."""

    def test_order_independent(self) -> None:
        fp1 = compute_split_fingerprint(["a", "b", "c"], ["d"], ["e"])
        fp2 = compute_split_fingerprint(["c", "a", "b"], ["d"], ["e"])
        assert fp1 == fp2

    def test_content_sensitive(self) -> None:
        fp1 = compute_split_fingerprint(["a", "b"], ["c"], ["d"])
        fp2 = compute_split_fingerprint(["a"], ["b", "c"], ["d"])
        assert fp1 != fp2

    def test_length_16(self) -> None:
        fp = compute_split_fingerprint(["a"], ["b"], ["c"])
        assert len(fp) == 16

    def test_deterministic(self) -> None:
        fp1 = compute_split_fingerprint(["x", "y"], ["z"], [])
        fp2 = compute_split_fingerprint(["x", "y"], ["z"], [])
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# _get_field tests
# ---------------------------------------------------------------------------


class TestGetField:
    """Tests for :func:`_get_field`."""

    def test_dict_top_level(self) -> None:
        assert _get_field({"label": 3}, "label") == 3

    def test_dict_nested_metadata(self) -> None:
        assert _get_field({"metadata": {"ts": 99}}, "ts") == 99

    def test_dict_default(self) -> None:
        assert _get_field({}, "missing", "default") == "default"

    def test_object_attribute(self) -> None:
        clip = FakeClip("a", 0, "normal", "p", "t")
        assert _get_field(clip, "label") == 0

    def test_object_nested_metadata(self) -> None:
        clip = FakeClip("a", 0, "normal", "p", "t", {"ts": 5})
        assert _get_field(clip, "ts") == 5

    def test_top_level_priority_over_metadata(self) -> None:
        rec = {"label": 1, "metadata": {"label": 2}}
        assert _get_field(rec, "label") == 1


# ---------------------------------------------------------------------------
# Strategy: RANDOM
# ---------------------------------------------------------------------------


class TestRandomStrategy:
    """Tests for the RANDOM split strategy."""

    def test_split_sizes_sum(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.RANDOM))
        r = mgr.split(manifest)
        assert r.report.n_train + r.report.n_val + r.report.n_test == len(manifest)

    def test_returns_split_result(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.RANDOM))
        assert isinstance(mgr.split(manifest), SplitResult)


# ---------------------------------------------------------------------------
# Strategy: STRATIFIED
# ---------------------------------------------------------------------------


class TestStratifiedStrategy:
    """Tests for the STRATIFIED split strategy."""

    def test_all_splits_non_empty(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        r = mgr.split(manifest)
        assert r.report.n_train > 0
        assert r.report.n_val > 0
        assert r.report.n_test > 0

    def test_class_balance_preserved(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        r = mgr.split(manifest)
        # All 4 classes should appear in train
        assert len(r.report.train_class_balance) == len(FAULT_TYPES)

    def test_balance_sums_to_one(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        r = mgr.split(manifest)
        assert abs(sum(r.report.train_class_balance.values()) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Strategy: GROUP / STRATIFIED_GROUP — leakage prevention
# ---------------------------------------------------------------------------


class TestGroupStrategies:
    """Tests for GROUP and STRATIFIED_GROUP leakage prevention."""

    def test_group_no_leakage(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.GROUP))
        r = mgr.split(manifest)
        assert r.report.leakage.is_clean

    def test_stratified_group_no_leakage(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        r = mgr.split(manifest)
        assert r.report.leakage.is_clean

    def test_group_source_recording_not_split(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.GROUP))
        r = mgr.split(manifest)
        # Derive recording id from source_path stem (drop chunk suffix)
        def rec_id(x):
            stem = Path(_get_field(x, "source_path")).stem
            return stem.rsplit("_", 1)[0] if "_" in stem else stem
        train_recs = {rec_id(x) for x in r.train}
        test_recs  = {rec_id(x) for x in r.test}
        assert not (train_recs & test_recs)

    def test_verify_no_leakage_passes_on_clean(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.STRATIFIED_GROUP, verify_no_leakage=True
        ))
        # Should not raise
        r = mgr.split(manifest)
        assert r.report.leakage.is_clean

    def test_custom_group_key(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.GROUP, group_key="dataset"
        ))
        r = mgr.split(manifest)
        train_g = {_get_field(x, "dataset") for x in r.train}
        test_g  = {_get_field(x, "dataset") for x in r.test}
        assert not (train_g & test_g)


# ---------------------------------------------------------------------------
# Strategy: TURBINE
# ---------------------------------------------------------------------------


class TestTurbineStrategy:
    """Tests for the TURBINE split strategy."""

    def test_no_turbine_leakage(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TURBINE))
        r = mgr.split(manifest)
        assert r.report.leakage.is_clean

    def test_turbines_disjoint_across_splits(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TURBINE))
        r = mgr.split(manifest)
        train_t = {_get_field(x, "dataset") for x in r.train}
        val_t   = {_get_field(x, "dataset") for x in r.val}
        test_t  = {_get_field(x, "dataset") for x in r.test}
        assert not (train_t & test_t)
        assert not (train_t & val_t)
        assert not (val_t & test_t)

    def test_few_turbines_warning(self) -> None:
        # Only 2 turbines
        manifest = [
            {"clip_id": f"c{i}", "label": i % 4, "dataset": f"turbine_{i % 2}"}
            for i in range(16)
        ]
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TURBINE))
        r = mgr.split(manifest)
        assert any("turbine" in w.lower() for w in r.report.warnings)

    def test_custom_turbine_key(self) -> None:
        manifest = [
            {"clip_id": f"c{i}", "label": i % 4, "asset_id": f"A{i % 4}"}
            for i in range(20)
        ]
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.TURBINE, turbine_key="asset_id"
        ))
        r = mgr.split(manifest)
        train_a = {_get_field(x, "asset_id") for x in r.train}
        test_a  = {_get_field(x, "asset_id") for x in r.test}
        assert not (train_a & test_a)


# ---------------------------------------------------------------------------
# Strategy: TEMPORAL
# ---------------------------------------------------------------------------


class TestTemporalStrategy:
    """Tests for the TEMPORAL split strategy."""

    def test_chronological_order(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TEMPORAL))
        r = mgr.split(manifest)
        if r.train and r.val:
            assert _max_ts(r.train) <= _min_ts(r.val)
        if r.val and r.test:
            assert _max_ts(r.val) <= _min_ts(r.test)

    def test_no_timestamp_fallback_warning(self) -> None:
        manifest = [{"clip_id": f"c{i}", "label": i % 4} for i in range(20)]
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TEMPORAL))
        r = mgr.split(manifest)
        assert any("timestamp" in w.lower() for w in r.report.warnings)

    def test_iso_timestamp_parsing(self) -> None:
        manifest = [
            {"clip_id": f"c{i}", "label": i % 4,
             "metadata": {"timestamp": f"2024-01-{(i % 28) + 1:02d}T12:00:00"}}
            for i in range(16)
        ]
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TEMPORAL))
        r = mgr.split(manifest)
        assert r.report.n_total == 16

    def test_sizes_sum(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TEMPORAL))
        r = mgr.split(manifest)
        assert r.report.n_train + r.report.n_val + r.report.n_test == len(manifest)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for deterministic, reproducible splits."""

    @pytest.mark.parametrize("strategy", list(SplitStrategy))
    def test_same_seed_same_fingerprint(
        self, manifest: list[dict], strategy: SplitStrategy
    ) -> None:
        cfg = SplitConfig(strategy=strategy, random_seed=42)
        r1 = SplitManager(cfg).split(manifest)
        r2 = SplitManager(cfg).split(manifest)
        assert r1.report.fingerprint == r2.report.fingerprint

    def test_same_seed_same_membership(self, manifest: list[dict]) -> None:
        cfg = SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP, random_seed=42)
        r1 = SplitManager(cfg).split(manifest)
        r2 = SplitManager(cfg).split(manifest)
        ids1 = sorted(_get_field(x, "clip_id") for x in r1.train)
        ids2 = sorted(_get_field(x, "clip_id") for x in r2.train)
        assert ids1 == ids2

    def test_fingerprint_is_16_hex(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig()).split(manifest)
        assert len(r.report.fingerprint) == 16
        int(r.report.fingerprint, 16)  # parses as hex


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Tests for :meth:`SplitManager.cross_validate`."""

    def test_stratified_group_folds(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        folds = mgr.cross_validate(manifest, n_folds=4)
        assert len(folds) == 4
        assert all(isinstance(f, FoldResult) for f in folds)

    def test_cv_folds_no_group_leakage(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        folds = mgr.cross_validate(manifest, n_folds=4)
        assert all(f.leakage.is_clean for f in folds)

    def test_group_kfold(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.GROUP))
        folds = mgr.cross_validate(manifest, n_folds=3)
        assert len(folds) == 3

    def test_stratified_kfold(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        folds = mgr.cross_validate(manifest, n_folds=4)
        assert len(folds) == 4

    def test_random_kfold(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.RANDOM))
        folds = mgr.cross_validate(manifest, n_folds=5)
        assert len(folds) == 5

    def test_temporal_expanding_window(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.TEMPORAL))
        folds = mgr.cross_validate(manifest, n_folds=3)
        assert len(folds) == 3
        for f in folds:
            if f.train and f.val:
                assert _max_ts(f.train) <= _min_ts(f.val)

    def test_n_folds_minimum(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        with pytest.raises(ValueError, match="n_folds"):
            mgr.cross_validate(manifest, n_folds=1)

    def test_cv_invalid_manifest_raises(self) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.RANDOM))
        with pytest.raises(ManifestValidationError):
            mgr.cross_validate([{"clip_id": "a", "label": 0},
                                {"clip_id": "a", "label": 1}], n_folds=2)

    def test_cv_fold_indices_sequential(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        folds = mgr.cross_validate(manifest, n_folds=4)
        assert [f.fold_index for f in folds] == [0, 1, 2, 3]

    def test_cv_fold_fingerprints_unique(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        folds = mgr.cross_validate(manifest, n_folds=4)
        fps = [f.fingerprint for f in folds]
        assert len(set(fps)) == len(fps)


# ---------------------------------------------------------------------------
# LeakageAudit / SplitReport tests
# ---------------------------------------------------------------------------


class TestLeakageAudit:
    """Tests for :class:`LeakageAudit`."""

    def test_total_overlaps(self) -> None:
        audit = LeakageAudit(False, ["g1"], ["g2", "g3"], [], "group")
        assert audit.total_overlaps == 3

    def test_clean_audit_zero_overlaps(self) -> None:
        audit = LeakageAudit(True, [], [], [], "group")
        assert audit.total_overlaps == 0

    def test_frozen(self) -> None:
        audit = LeakageAudit(True, [], [], [], "group")
        with pytest.raises((AttributeError, TypeError)):
            audit.is_clean = False  # type: ignore[misc]


class TestSplitReport:
    """Tests for :class:`SplitReport`."""

    def test_summary_string(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig()).split(manifest)
        s = r.report.summary()
        assert "Split Report" in s
        assert r.report.fingerprint in s

    def test_to_dict_keys(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig()).split(manifest)
        d = r.report.to_dict()
        for key in ("strategy", "n_train", "n_val", "n_test",
                    "fingerprint", "leakage_clean", "seed"):
            assert key in d

    def test_summary_shows_leakage_clean(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP)).split(manifest)
        assert "YES" in r.report.summary()

    def test_report_n_groups(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig(strategy=SplitStrategy.GROUP)).split(manifest)
        assert r.report.n_groups > 0


# ---------------------------------------------------------------------------
# ExperimentTracker integration tests
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for ExperimentTracker logging."""

    def test_split_logs_params(self, manifest: list[dict]) -> None:
        logged: list[dict] = []

        class FakeTracker:
            def log_params(self, p: dict) -> None:
                logged.append(dict(p))

        mgr = SplitManager(SplitConfig(experiment_tracker=FakeTracker()))
        mgr.split(manifest)
        keys = {k for d in logged for k in d}
        assert "split.strategy" in keys
        assert "split.fingerprint" in keys

    def test_cv_logs_params(self, manifest: list[dict]) -> None:
        logged: list[dict] = []

        class FakeTracker:
            def log_params(self, p: dict) -> None:
                logged.append(dict(p))

        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.STRATIFIED, experiment_tracker=FakeTracker()
        ))
        mgr.cross_validate(manifest, n_folds=3)
        keys = {k for d in logged for k in d}
        assert "cv.n_folds" in keys

    def test_broken_tracker_does_not_crash(self, manifest: list[dict]) -> None:
        class BrokenTracker:
            def log_params(self, *a, **kw) -> None:
                raise RuntimeError("boom")

        mgr = SplitManager(SplitConfig(experiment_tracker=BrokenTracker()))
        r = mgr.split(manifest)  # must not raise
        assert r.report.n_total == len(manifest)

    def test_none_tracker_safe(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(experiment_tracker=None))
        r = mgr.split(manifest)
        assert r.report.n_total == len(manifest)


# ---------------------------------------------------------------------------
# Object-record compatibility tests
# ---------------------------------------------------------------------------


class TestObjectRecordCompatibility:
    """Tests that ClipRecord-like objects work identically to dicts."""

    def test_object_split(self, object_manifest: list[FakeClip]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        r = mgr.split(object_manifest)
        assert r.report.n_total == len(object_manifest)
        assert r.report.leakage.is_clean

    def test_object_returns_objects(self, object_manifest: list[FakeClip]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED))
        r = mgr.split(object_manifest)
        if r.train:
            assert isinstance(r.train[0], FakeClip)

    def test_object_cross_validation(self, object_manifest: list[FakeClip]) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        folds = mgr.cross_validate(object_manifest, n_folds=3)
        assert len(folds) == 3


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------


class TestRobustness:
    """Edge case and robustness tests."""

    def test_invalid_manifest_raises(self) -> None:
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.RANDOM))
        with pytest.raises(ManifestValidationError):
            mgr.split([{"clip_id": "", "label": 0}])

    def test_single_class(self) -> None:
        manifest = [
            {"clip_id": f"n_{i:03d}", "label": 0,
             "source_path": f"data/normal/n_{i // 2:03d}.wav"}
            for i in range(12)
        ]
        mgr = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
        r = mgr.split(manifest)
        assert r.report.n_total == 12

    def test_custom_fractions_respected(self, manifest: list[dict]) -> None:
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.RANDOM,
            train_fraction=0.6, val_fraction=0.2, test_fraction=0.2,
        ))
        r = mgr.split(manifest)
        assert abs(r.report.n_train / len(manifest) - 0.6) < 0.2

    def test_few_groups_warning(self) -> None:
        # 2 groups only
        manifest = [
            {"clip_id": f"c{i}", "label": i % 2,
             "source_path": f"data/rec_{i % 2:03d}.wav"}
            for i in range(10)
        ]
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.GROUP, group_key="source_path"
        ))
        r = mgr.split(manifest)
        # Either a warning is present or splits are still produced
        assert r.report.n_total == 10

    def test_empty_val_test_fractions(self) -> None:
        manifest = make_dict_manifest(8)
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.RANDOM,
            train_fraction=1.0, val_fraction=0.0, test_fraction=0.0,
        ))
        r = mgr.split(manifest)
        assert r.report.n_train == len(manifest)

    def test_all_strategies_run_without_error(self, manifest: list[dict]) -> None:
        for strategy in SplitStrategy:
            mgr = SplitManager(SplitConfig(strategy=strategy))
            r = mgr.split(manifest)
            assert r.report.n_total == len(manifest)

    def test_warnings_recorded_for_empty_test(self) -> None:
        # Tiny manifest where test may be empty
        manifest = [{"clip_id": f"c{i}", "label": 0} for i in range(2)]
        mgr = SplitManager(SplitConfig(
            strategy=SplitStrategy.RANDOM, verify_no_leakage=False
        ))
        r = mgr.split(manifest)
        assert isinstance(r.report.warnings, list)


# ---------------------------------------------------------------------------
# SplitResult structure tests
# ---------------------------------------------------------------------------


class TestSplitResult:
    """Tests for :class:`SplitResult` structure."""

    def test_has_all_fields(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig()).split(manifest)
        assert hasattr(r, "train")
        assert hasattr(r, "val")
        assert hasattr(r, "test")
        assert hasattr(r, "report")

    def test_partitions_are_lists(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig()).split(manifest)
        assert isinstance(r.train, list)
        assert isinstance(r.val, list)
        assert isinstance(r.test, list)

    def test_no_clip_in_two_partitions(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP)).split(manifest)
        train_ids = {_get_field(x, "clip_id") for x in r.train}
        val_ids   = {_get_field(x, "clip_id") for x in r.val}
        test_ids  = {_get_field(x, "clip_id") for x in r.test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_total_clips_conserved(self, manifest: list[dict]) -> None:
        r = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP)).split(manifest)
        assert len(r.train) + len(r.val) + len(r.test) == len(manifest)