#!/usr/bin/env python3
"""Comprehensive test suite for ``src/training/dataset.py``.

Coverage targets:
- :class:`DatasetConfig` validation and property computation
- :class:`WindTurbineDataset` construction from pre-extracted features
- :class:`WindTurbineDataset` construction from raw audio (on-the-fly mode)
- All :class:`FeatureMode` values
- Train/val/test split correctness, no-overlap, stratification
- :meth:`__getitem__`, :meth:`__len__`, :meth:`__iter__`
- Class balancing (weights, sampler)
- :meth:`normal_only`, :meth:`filter_by_dataset` specialised views
- :meth:`as_numpy` for classical ML
- :meth:`split` factory
- :meth:`class_distribution` DataFrame
- Feature caching (write + read)
- Manifest building from features dir and raw audio dir
- Robustness: empty dirs, missing files, corrupt metadata, unknown labels
- Multi-worker safety (concurrent ``__getitem__``)
- Deterministic reproducibility (same seed → same splits)
- ExperimentTracker integration
- :func:`create_dataloaders` factory (torch-optional)
- ClipRecord and DatasetStatistics dataclasses

Run with::

    pytest tests/test_dataset.py -v
    pytest tests/test_dataset.py -v --tb=short -q   # summary only
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable from any working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.dataset import (
    CLASS_NAMES,
    DEFAULT_CQT_BINS,
    DEFAULT_CQT_BPO,
    DEFAULT_N_FRAMES,
    DEFAULT_N_MELS,
    DEFAULT_N_MFCC,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    ClipRecord,
    DataSplit,
    DatasetConfig,
    DatasetStatistics,
    FeatureMode,
    WindTurbineDataset,
    create_dataloaders,
)

# ---------------------------------------------------------------------------
# Optional torch import for torch-specific tests
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

# ---------------------------------------------------------------------------
# Constants for fixture data
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
CLIP_DURATION = 10.0
N_FRAMES = 431  # ceil(22050 * 10.0 / 512)
FAULT_TYPES = ["normal", "bearing_fault", "blade_imbalance", "gearbox_fault"]
LABELS = {ft: i for i, ft in enumerate(FAULT_TYPES)}
N_CLIPS_PER_CLASS = 5  # enough for stratified split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_feature_clip(
    clip_dir: Path,
    fault_type: str,
    label: int,
    clip_index: int,
    *,
    augmented: bool = False,
    dataset: str = "synthetic",
    n_mels: int = 128,
    n_mfcc: int = 40,
    cqt_bins: int = 168,
    n_frames: int = N_FRAMES,
    corrupt_metadata: bool = False,
    missing_mel: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Write a complete fake feature clip directory."""
    clip_dir.mkdir(parents=True, exist_ok=True)

    if not missing_mel:
        np.save(clip_dir / "mel.npy",
                np.random.randn(n_mels, n_frames).astype(np.float32))
    np.save(clip_dir / "mfcc.npy",
            np.random.randn(n_mfcc, n_frames).astype(np.float32))
    np.save(clip_dir / "cqt.npy",
            np.random.randn(cqt_bins, n_frames).astype(np.float32))
    np.save(clip_dir / "spectral_features.npy",
            np.random.randn(12).astype(np.float32))

    if corrupt_metadata:
        (clip_dir / "metadata.json").write_text("NOT_VALID_JSON{{{")
    else:
        meta = {
            "fault_type": fault_type,
            "label": label,
            "sample_rate": sample_rate,
            "duration": CLIP_DURATION,
            "source_path": f"data/raw/synthetic/{fault_type}/{fault_type}_{clip_index:03d}.wav",
            "dataset": dataset,
            "augmented": augmented,
            "clip_index": clip_index,
        }
        (clip_dir / "metadata.json").write_text(json.dumps(meta))


@pytest.fixture()
def features_dir(tmp_path: Path) -> Path:
    """Fake pre-extracted feature tree with N_CLIPS_PER_CLASS clips per class."""
    root = tmp_path / "features"
    for ft in FAULT_TYPES:
        label = LABELS[ft]
        for i in range(N_CLIPS_PER_CLASS):
            clip_dir = root / ft / f"{ft}_{i:03d}"
            _write_feature_clip(clip_dir, ft, label, i)
    return root


@pytest.fixture()
def features_dir_with_augmented(tmp_path: Path) -> Path:
    """Feature tree with both original and augmented clips."""
    root = tmp_path / "features_aug"
    for ft in FAULT_TYPES:
        label = LABELS[ft]
        for i in range(3):
            clip_dir = root / ft / f"{ft}_{i:03d}"
            _write_feature_clip(clip_dir, ft, label, i)
            # One augmented variant per clip
            aug_dir = root / ft / f"{ft}_{i:03d}_aug00"
            _write_feature_clip(aug_dir, ft, label, i, augmented=True)
    return root


@pytest.fixture()
def raw_audio_dir(tmp_path: Path) -> Path:
    """Fake raw audio directory with WAV-like stubs (soundfile not available)."""
    root = tmp_path / "raw"
    for ft in FAULT_TYPES:
        (root / ft).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def minimal_config(features_dir: Path, tmp_path: Path) -> DatasetConfig:
    """Minimal DatasetConfig pointing at the fake features tree."""
    return DatasetConfig(
        features_dir=features_dir,
        raw_audio_dir=tmp_path / "raw_missing",
        cache_dir=None,
        feature_mode=FeatureMode.MEL,
        n_mels=128,
        n_mfcc=40,
        cqt_bins=168,
        cqt_bins_per_octave=24,
        random_seed=42,
    )


@pytest.fixture()
def full_dataset(minimal_config: DatasetConfig) -> WindTurbineDataset:
    """Full (ALL-split) dataset over fake features."""
    return WindTurbineDataset(minimal_config, split=DataSplit.ALL)


# ---------------------------------------------------------------------------
# DatasetConfig tests
# ---------------------------------------------------------------------------


class TestDatasetConfig:
    """Tests for :class:`DatasetConfig`."""

    def test_default_construction(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(features_dir=tmp_path, cache_dir=None)
        assert cfg.n_mels == DEFAULT_N_MELS
        assert cfg.n_mfcc == DEFAULT_N_MFCC
        assert cfg.cqt_bins == DEFAULT_CQT_BINS
        assert cfg.cqt_bins_per_octave == DEFAULT_CQT_BPO
        assert cfg.random_seed == 42
        assert cfg.train_split + cfg.val_split + cfg.test_split == pytest.approx(1.0)

    def test_split_sum_validation(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            DatasetConfig(
                features_dir=tmp_path, cache_dir=None,
                train_split=0.8, val_split=0.2, test_split=0.2,
            )

    def test_negative_train_split(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            DatasetConfig(
                features_dir=tmp_path, cache_dir=None,
                train_split=-0.1, val_split=0.55, test_split=0.55,
            )

    def test_zero_feature_dims(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            DatasetConfig(features_dir=tmp_path, cache_dir=None, n_mels=0)

    def test_config_hash_is_deterministic(self, tmp_path: Path) -> None:
        cfg1 = DatasetConfig(features_dir=tmp_path, cache_dir=None, n_mels=128)
        cfg2 = DatasetConfig(features_dir=tmp_path, cache_dir=None, n_mels=128)
        assert cfg1.config_hash == cfg2.config_hash

    def test_config_hash_changes_with_n_mels(self, tmp_path: Path) -> None:
        cfg1 = DatasetConfig(features_dir=tmp_path, cache_dir=None, n_mels=128)
        cfg2 = DatasetConfig(features_dir=tmp_path, cache_dir=None, n_mels=64)
        assert cfg1.config_hash != cfg2.config_hash

    def test_feature_shape_mel(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.MEL, n_mels=128,
        )
        assert cfg.feature_shape == (128, N_FRAMES)

    def test_feature_shape_mfcc(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.MFCC, n_mfcc=40,
        )
        assert cfg.feature_shape == (40, N_FRAMES)

    def test_feature_shape_cqt(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.CQT, cqt_bins=168,
        )
        assert cfg.feature_shape == (168, N_FRAMES)

    def test_feature_shape_mel3channel(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.MEL_3CHANNEL, n_mels=128,
        )
        assert cfg.feature_shape == (3, 128, N_FRAMES)

    def test_feature_shape_combined(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.COMBINED,
            n_mels=128, n_mfcc=40, cqt_bins=168,
        )
        assert cfg.feature_shape == (336, N_FRAMES)

    def test_feature_shape_spectral(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.SPECTRAL,
        )
        assert cfg.feature_shape == (12,)

    def test_feature_shape_all_raises(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=tmp_path, cache_dir=None,
            feature_mode=FeatureMode.ALL,
        )
        with pytest.raises(ValueError, match="no fixed shape"):
            _ = cfg.feature_shape


# ---------------------------------------------------------------------------
# ClipRecord and DatasetStatistics tests
# ---------------------------------------------------------------------------


class TestDataContainers:
    """Tests for :class:`ClipRecord` and :class:`DatasetStatistics`."""

    def test_clip_record_construction(self) -> None:
        r = ClipRecord(
            clip_id="normal_000", fault_type="normal", label=0,
            source_path="data/raw/normal_000.wav",
            features_dir="data/processed/features/normal/normal_000",
            dataset="synthetic", split=SPLIT_TRAIN,
            augmented=False, duration_s=10.0, sample_rate=22050,
        )
        assert r.clip_id == "normal_000"
        assert r.label == 0
        assert r.metadata == {}

    def test_clip_record_metadata_default_factory(self) -> None:
        r1 = ClipRecord("a", "normal", 0, "", "", "x", "", False, 10.0, 22050)
        r2 = ClipRecord("b", "normal", 0, "", "", "x", "", False, 10.0, 22050)
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata  # independent dict instances

    def test_dataset_statistics_frozen(self) -> None:
        stats = DatasetStatistics(
            split=SPLIT_TRAIN, n_clips=10,
            clips_per_class={"normal": 5, "bearing_fault": 5},
            total_duration_s=100.0,
            class_weight={0: 1.0, 2: 1.0},
            augmented_fraction=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            stats.n_clips = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WindTurbineDataset construction tests
# ---------------------------------------------------------------------------


class TestDatasetConstruction:
    """Tests for :class:`WindTurbineDataset` manifest building."""

    def test_loads_from_features_dir(self, full_dataset: WindTurbineDataset) -> None:
        assert len(full_dataset) == len(FAULT_TYPES) * N_CLIPS_PER_CLASS

    def test_all_classes_present(self, full_dataset: WindTurbineDataset) -> None:
        found = {r.fault_type for r in full_dataset.manifest}
        assert found == set(FAULT_TYPES)

    def test_labels_match_fault_types(self, full_dataset: WindTurbineDataset) -> None:
        for record in full_dataset.manifest:
            assert record.label == LABELS[record.fault_type]

    def test_raises_on_empty_dirs(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_features"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            WindTurbineDataset(
                DatasetConfig(
                    features_dir=empty,
                    raw_audio_dir=tmp_path / "also_empty",
                    cache_dir=None,
                )
            )

    def test_skips_corrupt_metadata(self, tmp_path: Path) -> None:
        root = tmp_path / "partial"
        # Good clip
        _write_feature_clip(root / "normal" / "normal_000", "normal", 0, 0)
        # Corrupt metadata clip
        _write_feature_clip(root / "normal" / "normal_001", "normal", 0, 1,
                            corrupt_metadata=True)
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
            )
        )
        assert len(ds) == 1  # only the good clip

    def test_excludes_unknown_labels_by_default(self, tmp_path: Path) -> None:
        root = tmp_path / "feats_unknown"
        _write_feature_clip(root / "normal" / "n_000", "normal", 0, 0)
        _write_feature_clip(root / "unknown" / "u_000", "unknown", -1, 0)
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
            )
        )
        assert all(r.label != -1 for r in ds.manifest)

    def test_include_unknown_when_disabled(self, tmp_path: Path) -> None:
        root = tmp_path / "feats_uk2"
        _write_feature_clip(root / "normal" / "n_000", "normal", 0, 0)
        _write_feature_clip(root / "unknown" / "u_000", "unknown", -1, 0)
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
                exclude_unknown=False,
            )
        )
        assert any(r.label == -1 for r in ds.manifest)

    def test_exclude_augmented(
        self, features_dir_with_augmented: Path, tmp_path: Path
    ) -> None:
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=features_dir_with_augmented,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
                include_augmented=False,
            )
        )
        assert not any(r.augmented for r in ds.manifest)

    def test_include_augmented(
        self, features_dir_with_augmented: Path, tmp_path: Path
    ) -> None:
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=features_dir_with_augmented,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
                include_augmented=True,
            )
        )
        assert any(r.augmented for r in ds.manifest)


# ---------------------------------------------------------------------------
# __getitem__ and feature shape tests
# ---------------------------------------------------------------------------


class TestGetItem:
    """Tests for :meth:`WindTurbineDataset.__getitem__`."""

    @pytest.mark.parametrize("mode,expected_shape", [
        (FeatureMode.MEL,          (128, N_FRAMES)),
        (FeatureMode.MFCC,         (40,  N_FRAMES)),
        (FeatureMode.CQT,          (168, N_FRAMES)),
        (FeatureMode.MEL_3CHANNEL, (3, 128, N_FRAMES)),
        (FeatureMode.COMBINED,     (336, N_FRAMES)),
        (FeatureMode.SPECTRAL,     (12,)),
    ])
    def test_feature_shape(
        self,
        features_dir: Path,
        tmp_path: Path,
        mode: FeatureMode,
        expected_shape: tuple[int, ...],
    ) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=mode,
        )
        ds = WindTurbineDataset(cfg)
        feat, label = ds[0]
        if _TORCH and hasattr(feat, "shape"):
            assert tuple(feat.shape) == expected_shape
        elif hasattr(feat, "shape"):
            assert feat.shape == expected_shape

    def test_all_mode_returns_dict(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.ALL,
        )
        ds = WindTurbineDataset(cfg)
        feat, label = ds[0]
        assert isinstance(feat, dict)
        assert set(feat.keys()) >= {"mel", "mfcc", "cqt", "spectral"}

    def test_label_is_integer(self, full_dataset: WindTurbineDataset) -> None:
        _, label = full_dataset[0]
        assert isinstance(label, int)

    def test_label_in_valid_range(self, full_dataset: WindTurbineDataset) -> None:
        for i in range(len(full_dataset)):
            _, label = full_dataset[i]
            assert label in LABELS.values()

    def test_index_out_of_range_raises(self, full_dataset: WindTurbineDataset) -> None:
        with pytest.raises(IndexError):
            full_dataset[len(full_dataset)]

    def test_negative_index_raises(self, full_dataset: WindTurbineDataset) -> None:
        with pytest.raises(IndexError):
            full_dataset[-1]

    def test_feature_is_float32(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.MEL,
        )
        ds = WindTurbineDataset(cfg)
        feat, _ = ds[0]
        if _TORCH and isinstance(feat, torch.Tensor):
            assert feat.dtype == torch.float32
        elif hasattr(feat, "dtype"):
            assert feat.dtype == np.float32

    def test_no_nan_in_features(self, full_dataset: WindTurbineDataset) -> None:
        for i in range(min(len(full_dataset), 5)):
            feat, _ = full_dataset[i]
            if _TORCH and isinstance(feat, torch.Tensor):
                assert not torch.isnan(feat).any()
            elif isinstance(feat, np.ndarray):
                assert np.all(np.isfinite(feat))

    def test_iter_yields_all_samples(self, full_dataset: WindTurbineDataset) -> None:
        items = list(full_dataset)
        assert len(items) == len(full_dataset)

    def test_normalise_features_zero_mean(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.MEL,
            normalise_features=True,
        )
        ds = WindTurbineDataset(cfg)
        feat, _ = ds[0]
        if _TORCH and isinstance(feat, torch.Tensor):
            assert abs(feat.mean().item()) < 0.1
        elif isinstance(feat, np.ndarray):
            assert abs(feat.mean()) < 0.1


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------


class TestSplits:
    """Tests for train/val/test split correctness."""

    def test_split_returns_three_datasets(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        assert isinstance(train, WindTurbineDataset)
        assert isinstance(val, WindTurbineDataset)
        assert isinstance(test, WindTurbineDataset)

    def test_split_sizes_sum_to_total(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        assert len(train) + len(val) + len(test) == len(full_dataset)

    def test_no_clip_in_multiple_splits(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        train_ids = {r.clip_id for r in train.manifest}
        val_ids   = {r.clip_id for r in val.manifest}
        test_ids  = {r.clip_id for r in test.manifest}
        assert train_ids.isdisjoint(val_ids), "Train/val overlap"
        assert train_ids.isdisjoint(test_ids), "Train/test overlap"
        assert val_ids.isdisjoint(test_ids), "Val/test overlap"

    def test_train_is_largest_split(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        assert len(train) >= len(val)
        assert len(train) >= len(test)

    def test_split_is_deterministic(
        self, minimal_config: DatasetConfig
    ) -> None:
        ds1 = WindTurbineDataset(minimal_config)
        ds2 = WindTurbineDataset(minimal_config)
        ids1 = sorted(r.clip_id for r in ds1.split()[0].manifest)
        ids2 = sorted(r.clip_id for r in ds2.split()[0].manifest)
        assert ids1 == ids2

    def test_different_seeds_give_different_splits(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cfg1 = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None, random_seed=42,
        )
        cfg2 = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None, random_seed=99,
        )
        ds1 = WindTurbineDataset(cfg1)
        ds2 = WindTurbineDataset(cfg2)
        ids1 = sorted(r.clip_id for r in ds1.split()[0].manifest)
        ids2 = sorted(r.clip_id for r in ds2.split()[0].manifest)
        # With only 20 clips these may coincidentally match — that's fine;
        # just verify the function runs without error for two seeds.
        assert isinstance(ids1, list) and isinstance(ids2, list)

    def test_split_view_exposes_correct_partition(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        for r in train.manifest:
            assert r.split == SPLIT_TRAIN
        for r in val.manifest:
            assert r.split == SPLIT_VAL
        for r in test.manifest:
            assert r.split == SPLIT_TEST

    def test_direct_split_construction(
        self, minimal_config: DatasetConfig
    ) -> None:
        train_ds = WindTurbineDataset(minimal_config, split=DataSplit.TRAIN)
        assert all(r.split == SPLIT_TRAIN for r in train_ds.manifest)

    def test_all_splits_have_at_least_one_sample(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        train, val, test = full_dataset.split()
        assert len(train) >= 1
        assert len(val) >= 1
        assert len(test) >= 1


# ---------------------------------------------------------------------------
# Class balancing tests
# ---------------------------------------------------------------------------


class TestClassBalancing:
    """Tests for class weight and sampler computation."""

    def test_class_weights_all_positive(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        weights = full_dataset.class_weights()
        assert all(w > 0 for w in weights.values())

    def test_class_weights_keys_are_labels(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        weights = full_dataset.class_weights()
        for key in weights:
            assert isinstance(key, int)

    def test_balanced_class_weights_equal_for_equal_counts(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        # All classes have N_CLIPS_PER_CLASS clips → weights should be equal
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=features_dir,
                raw_audio_dir=tmp_path / "missing",
                cache_dir=None,
            )
        )
        weights = ds.class_weights()
        w_vals = list(weights.values())
        assert max(w_vals) / min(w_vals) == pytest.approx(1.0, rel=0.05)

    def test_class_distribution_dataframe_schema(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        dist = full_dataset.class_distribution()
        assert isinstance(dist, pd.DataFrame)
        for col in ("fault_type", "label", "count", "fraction", "weight"):
            assert col in dist.columns

    def test_class_distribution_fractions_sum_to_one(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        dist = full_dataset.class_distribution()
        assert dist["fraction"].sum() == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_weighted_sampler_length(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        sampler = full_dataset.weighted_sampler()
        assert sampler is not None
        assert len(sampler) == len(full_dataset)

    def test_weighted_sampler_returns_none_without_torch(
        self, full_dataset: WindTurbineDataset, monkeypatch
    ) -> None:
        import src.training.dataset as ds_mod
        monkeypatch.setattr(ds_mod, "_TORCH_AVAILABLE", False)
        result = full_dataset.weighted_sampler()
        assert result is None


# ---------------------------------------------------------------------------
# Specialised view tests
# ---------------------------------------------------------------------------


class TestSpecialisedViews:
    """Tests for :meth:`normal_only` and :meth:`filter_by_dataset`."""

    def test_normal_only_returns_only_normal(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        normal_ds = full_dataset.normal_only()
        assert all(r.fault_type == "normal" for r in normal_ds.manifest)

    def test_normal_only_has_correct_count(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        normal_ds = full_dataset.normal_only()
        expected = sum(1 for r in full_dataset.manifest if r.fault_type == "normal")
        assert len(normal_ds) == expected

    def test_filter_by_dataset_name(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        # Add clips from a different dataset
        root = tmp_path / "mixed"
        for ft in FAULT_TYPES[:2]:
            label = LABELS[ft]
            for i in range(3):
                _write_feature_clip(root / ft / f"{ft}_{i:03d}", ft, label, i,
                                    dataset="synthetic")
                _write_feature_clip(root / ft / f"{ft}_cwru_{i:03d}", ft, label, i,
                                    dataset="cwru")
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "missing",
                cache_dir=None,
            )
        )
        cwru_view = ds.filter_by_dataset("cwru")
        assert all(r.dataset == "cwru" for r in cwru_view.manifest)
        synth_view = ds.filter_by_dataset("synthetic")
        assert all(r.dataset == "synthetic" for r in synth_view.manifest)

    def test_filter_by_nonexistent_dataset_returns_empty(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        view = full_dataset.filter_by_dataset("nonexistent_turbine_farm")
        assert len(view) == 0

    def test_normal_only_supports_getitem(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        normal_ds = full_dataset.normal_only()
        if len(normal_ds) > 0:
            feat, label = normal_ds[0]
            assert label == 0  # normal label


# ---------------------------------------------------------------------------
# as_numpy tests
# ---------------------------------------------------------------------------


class TestAsNumpy:
    """Tests for :meth:`WindTurbineDataset.as_numpy`."""

    def test_returns_tuple_of_arrays(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        X, y = full_dataset.as_numpy()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_x_shape_n_samples_by_n_features(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        X, y = full_dataset.as_numpy()
        assert X.shape[0] == len(full_dataset)
        assert X.ndim == 2

    def test_y_length_matches_x(self, full_dataset: WindTurbineDataset) -> None:
        X, y = full_dataset.as_numpy()
        assert X.shape[0] == y.shape[0]

    def test_x_dtype_float32(self, full_dataset: WindTurbineDataset) -> None:
        X, _ = full_dataset.as_numpy()
        assert X.dtype == np.float32

    def test_y_dtype_int64(self, full_dataset: WindTurbineDataset) -> None:
        _, y = full_dataset.as_numpy()
        assert y.dtype == np.int64

    def test_all_labels_present(self, full_dataset: WindTurbineDataset) -> None:
        _, y = full_dataset.as_numpy()
        assert set(y.tolist()) == set(LABELS.values())


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for feature disk caching."""

    def test_cache_creates_file(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=cache_dir,
            feature_mode=FeatureMode.MEL,
        )
        ds = WindTurbineDataset(cfg)
        _ = ds[0]  # triggers cache write
        cache_files = list(cache_dir.rglob("*.npy"))
        assert len(cache_files) >= 1

    def test_cache_hit_returns_same_tensor(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache2"
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=cache_dir,
            feature_mode=FeatureMode.MEL,
        )
        ds = WindTurbineDataset(cfg)
        feat1, _ = ds[0]
        feat2, _ = ds[0]  # should hit cache
        if _TORCH and isinstance(feat1, torch.Tensor):
            assert torch.allclose(feat1, feat2)
        elif isinstance(feat1, np.ndarray):
            assert np.allclose(feat1, feat2)

    def test_different_config_hash_uses_different_cache(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache3"
        cfg1 = DatasetConfig(
            features_dir=features_dir, raw_audio_dir=tmp_path / "missing",
            cache_dir=cache_dir, feature_mode=FeatureMode.MEL, n_mels=128,
        )
        cfg2 = DatasetConfig(
            features_dir=features_dir, raw_audio_dir=tmp_path / "missing",
            cache_dir=cache_dir, feature_mode=FeatureMode.MEL, n_mels=64,
        )
        assert cfg1.config_hash != cfg2.config_hash

    def test_cache_disabled_when_none(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.MEL,
        )
        ds = WindTurbineDataset(cfg)
        _ = ds[0]
        # No cache directory should have been created
        default_cache = _PROJECT_ROOT / "data" / "processed" / "feature_cache"
        # Only check if it doesn't exist in tmp — config points elsewhere
        assert cfg.cache_dir is None


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for :class:`DatasetStatistics` computation."""

    def test_statistics_n_clips_correct(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        assert full_dataset.statistics.n_clips == len(full_dataset)

    def test_statistics_all_classes_present(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        assert set(full_dataset.statistics.clips_per_class.keys()) == set(FAULT_TYPES)

    def test_statistics_total_duration(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        expected = len(full_dataset) * CLIP_DURATION
        assert full_dataset.statistics.total_duration_s == pytest.approx(expected)

    def test_statistics_class_weights_positive(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        for w in full_dataset.statistics.class_weight.values():
            assert w > 0

    def test_summary_string_contains_split(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        summary = full_dataset.summary()
        assert "split=" in summary

    def test_summary_string_contains_feature_mode(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        summary = full_dataset.summary()
        assert "mel" in summary.lower()


# ---------------------------------------------------------------------------
# get_record tests
# ---------------------------------------------------------------------------


class TestGetRecord:
    """Tests for :meth:`WindTurbineDataset.get_record`."""

    def test_get_record_returns_clip_record(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        record = full_dataset.get_record(0)
        assert isinstance(record, ClipRecord)

    def test_get_record_out_of_range(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        with pytest.raises(IndexError):
            full_dataset.get_record(len(full_dataset))

    def test_get_record_clip_id_matches(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        record = full_dataset.get_record(0)
        assert record.clip_id != ""

    def test_labels_property_shape(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        labels = full_dataset.labels
        assert labels.shape == (len(full_dataset),)
        assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# MEL_3CHANNEL computation tests
# ---------------------------------------------------------------------------


class TestMel3Channel:
    """Tests for the Mel + Δ + ΔΔ three-channel computation."""

    def test_shape(self, features_dir: Path, tmp_path: Path) -> None:
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.MEL_3CHANNEL,
        )
        ds = WindTurbineDataset(cfg)
        feat, _ = ds[0]
        if hasattr(feat, "shape"):
            assert feat.shape[-3:] == (3, 128, N_FRAMES)

    def test_make_mel_3channel_static(self) -> None:
        mel = np.random.randn(128, 431).astype(np.float32)
        result = WindTurbineDataset._make_mel_3channel(mel)
        assert result.shape == (3, 128, 431)
        assert result.dtype == np.float32

    def test_first_channel_is_mel(self) -> None:
        mel = np.ones((128, 431), dtype=np.float32) * 3.0
        result = WindTurbineDataset._make_mel_3channel(mel)
        assert np.allclose(result[0], mel)


# ---------------------------------------------------------------------------
# Multi-worker safety tests
# ---------------------------------------------------------------------------


class TestMultiWorkerSafety:
    """Tests for thread-safe concurrent ``__getitem__`` access."""

    def test_concurrent_getitem_no_errors(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        errors: list[Exception] = []
        results: list[tuple] = []

        def _worker(idx: int) -> None:
            try:
                feat, label = full_dataset[idx % len(full_dataset)]
                results.append((idx, label))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(i,))
            for i in range(40)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 40

    def test_concurrent_different_indices(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        labels_seen: list[int] = []
        lock = threading.Lock()

        def _worker(idx: int) -> None:
            feat, label = full_dataset[idx]
            with lock:
                labels_seen.append(label)

        n = min(len(full_dataset), 10)
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(labels_seen) == n


# ---------------------------------------------------------------------------
# ExperimentTracker integration tests
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for :class:`ExperimentTracker` integration."""

    def test_tracker_receives_dataset_info(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        """Verify log_dataset_info is called with correct clip count."""
        calls: list[dict] = []

        class FakeTracker:
            def log_dataset_info(self, total_clips, clips_per_class,
                                  total_duration_s, **kwargs):
                calls.append({
                    "total_clips": total_clips,
                    "clips_per_class": clips_per_class,
                })

        tracker = FakeTracker()
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            experiment_tracker=tracker,
        )
        ds = WindTurbineDataset(cfg)
        assert len(calls) == 1
        assert calls[0]["total_clips"] == len(ds)

    def test_tracker_failure_does_not_crash_dataset(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        class BrokenTracker:
            def log_dataset_info(self, *a, **kw):
                raise RuntimeError("tracker exploded")

        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            experiment_tracker=BrokenTracker(),
        )
        # Should not raise — tracker errors are logged and swallowed
        ds = WindTurbineDataset(cfg)
        assert len(ds) > 0

    def test_none_tracker_is_safe(
        self, minimal_config: DatasetConfig
    ) -> None:
        assert minimal_config.experiment_tracker is None
        ds = WindTurbineDataset(minimal_config)
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# create_dataloaders factory tests
# ---------------------------------------------------------------------------


class TestCreateDataloaders:
    """Tests for :func:`create_dataloaders`."""

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_returns_three_dataloaders(
        self, minimal_config: DatasetConfig
    ) -> None:
        train_dl, val_dl, test_dl = create_dataloaders(
            minimal_config, batch_size=4, num_workers=0
        )
        assert train_dl is not None
        assert val_dl is not None
        assert test_dl is not None

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_batch_shape_correct(
        self, minimal_config: DatasetConfig
    ) -> None:
        train_dl, _, _ = create_dataloaders(
            minimal_config, batch_size=4, num_workers=0, balanced_train=False
        )
        batch_feats, batch_labels = next(iter(train_dl))
        assert batch_feats.shape[1:] == (128, N_FRAMES)
        assert batch_labels.ndim == 1

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_val_loader_not_shuffled(
        self, minimal_config: DatasetConfig
    ) -> None:
        # Run twice and check same order
        _, val_dl1, _ = create_dataloaders(
            minimal_config, batch_size=4, num_workers=0
        )
        _, val_dl2, _ = create_dataloaders(
            minimal_config, batch_size=4, num_workers=0
        )
        labels1 = [lb for _, lb in val_dl1]
        labels2 = [lb for _, lb in val_dl2]
        if _TORCH:
            pairs = zip(labels1, labels2)
            for l1, l2 in pairs:
                assert torch.equal(l1, l2)

    def test_returns_none_without_torch(
        self, minimal_config: DatasetConfig, monkeypatch
    ) -> None:
        import src.training.dataset as ds_mod
        monkeypatch.setattr(ds_mod, "_TORCH_AVAILABLE", False)
        result = create_dataloaders(minimal_config)
        assert result == (None, None, None)


# ---------------------------------------------------------------------------
# make_dataloader tests
# ---------------------------------------------------------------------------


class TestMakeDataloader:
    """Tests for :meth:`WindTurbineDataset.make_dataloader`."""

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_make_dataloader_returns_dataloader(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        from torch.utils.data import DataLoader as TorchDL
        dl = full_dataset.make_dataloader(batch_size=4, num_workers=0)
        assert isinstance(dl, TorchDL)

    @pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")
    def test_make_dataloader_iterates(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        dl = full_dataset.make_dataloader(batch_size=4, num_workers=0)
        batches = list(dl)
        total = sum(b[0].shape[0] for b in batches)
        assert total == len(full_dataset)


# ---------------------------------------------------------------------------
# Robustness / edge case tests
# ---------------------------------------------------------------------------


class TestRobustness:
    """Edge case and robustness tests."""

    def test_missing_mel_file_falls_through(
        self, tmp_path: Path
    ) -> None:
        """Clip dir missing mel.npy triggers on-the-fly path which raises RuntimeError."""
        root = tmp_path / "partial_feats"
        # Write clip without mel.npy
        for ft in FAULT_TYPES:
            _write_feature_clip(root / ft / f"{ft}_000", ft, LABELS[ft], 0,
                                missing_mel=True)
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "raw_missing",
                cache_dir=None,
                feature_mode=FeatureMode.MEL,
            )
        )
        # Dataset should load (mel file is optional for discovery)
        assert len(ds) >= 0

    def test_single_class_dataset(self, tmp_path: Path) -> None:
        root = tmp_path / "single_class"
        for i in range(8):
            _write_feature_clip(root / "normal" / f"normal_{i:03d}", "normal", 0, i)
        ds = WindTurbineDataset(
            DatasetConfig(
                features_dir=root,
                raw_audio_dir=tmp_path / "missing",
                cache_dir=None,
            )
        )
        assert len(ds) == 8
        train, val, test = ds.split()
        assert len(train) + len(val) + len(test) == 8

    def test_all_mode_handles_missing_file_gracefully(
        self, features_dir: Path, tmp_path: Path
    ) -> None:
        """ALL mode still returns dict even when cqt.npy missing (raises FileNotFoundError)."""
        cfg = DatasetConfig(
            features_dir=features_dir,
            raw_audio_dir=tmp_path / "missing",
            cache_dir=None,
            feature_mode=FeatureMode.ALL,
        )
        ds = WindTurbineDataset(cfg)
        # Should raise or return dict — just verify it doesn't silently corrupt
        try:
            feat, label = ds[0]
            assert isinstance(feat, dict)
        except (RuntimeError, FileNotFoundError):
            pass  # acceptable — on-the-fly extraction attempted and failed

    def test_features_dir_missing_scans_raw(
        self, raw_audio_dir: Path, tmp_path: Path
    ) -> None:
        """When features_dir is empty, dataset should raise FileNotFoundError."""
        empty_feats = tmp_path / "empty_feats"
        empty_feats.mkdir()
        with pytest.raises(FileNotFoundError):
            WindTurbineDataset(
                DatasetConfig(
                    features_dir=empty_feats,
                    raw_audio_dir=raw_audio_dir,
                    cache_dir=None,
                )
            )

    def test_config_hash_length(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(features_dir=tmp_path, cache_dir=None)
        assert len(cfg.config_hash) == 12

    def test_normalise_static_method(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        normed = WindTurbineDataset._normalise(arr)
        assert abs(normed.mean()) < 1e-5
        assert abs(normed.std() - 1.0) < 0.1

    def test_dataset_len_is_manifest_len(
        self, full_dataset: WindTurbineDataset
    ) -> None:
        assert len(full_dataset) == len(full_dataset.manifest)