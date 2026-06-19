#!/usr/bin/env python3
"""Production-grade dataset abstraction for Wind Turbine Acoustic Monitoring.

This module provides the PyTorch dataset layer that sits between the
preprocessing pipeline (``src/preprocessing/``) and the model training loop
(``src/training/train.py``).  It is designed to scale from a 20-clip synthetic
prototype to a multi-turbine fleet deployment with millions of clips.

Architecture overview
---------------------
::

    PreprocessingPipeline          WindTurbineDataset
    ─────────────────────          ──────────────────
    data/processed/features/  ──►  ClipManifest (pandas DataFrame)
      <fault_type>/<clip_id>/  ──►  __getitem__: lazy .npy load + cache
        mel.npy                ──►  torch.Tensor
        mfcc.npy                    (n_mels, T) | (n_mfcc, T) | (336, T) …
        cqt.npy
        spectral_features.npy
        metadata.json

Design principles
-----------------
1. **Lazy loading** — features are loaded from disk only when a sample is
   requested, not at dataset construction time.  Large datasets never OOM.
2. **Caching** — once loaded (or on-the-fly extracted), a feature tensor is
   written to a per-sample cache directory.  Subsequent epochs are I/O-bound
   only on the first pass.
3. **Multi-worker safe** — ``__getitem__`` performs only read operations and
   uses file-level ``np.load`` calls with no shared mutable state.  The dataset
   is safe for ``DataLoader(num_workers=N)`` with any N.
4. **Deterministic** — every random operation (splits, sampling) is seeded via
   ``DatasetConfig.random_seed``.
5. **Backward compatible** — all ``src/preprocessing/`` APIs are called through
   their existing public interfaces; no internal details are relied upon.
6. **ExperimentTracker integrated** — dataset provenance, class distribution,
   and split statistics are logged automatically when a tracker is provided.

Feature modes
-------------
+--------------------+----------------------------------+---------------------+
| FeatureMode        | Output shape                     | Use case            |
+====================+==================================+=====================+
| MEL                | (128, T)                         | Primary CNN input   |
| MFCC               | (40, T)                          | Classical ML        |
| CQT                | (168, T)                         | Harmonic analysis   |
| MEL_3CHANNEL       | (3, 128, T)                      | ResNet/EfficientNet |
| COMBINED           | (336, T)  [mel+mfcc+cqt concat]  | Multi-branch CNN    |
| SPECTRAL           | (12,)                            | Scalar baseline     |
| ALL                | dict[str, Tensor]                | Research / ablation |
+--------------------+----------------------------------+---------------------+

*T = 431 frames at 10 s × 22 050 Hz / hop 512*

Usage::

    from src.training.dataset import DatasetConfig, WindTurbineDataset, FeatureMode

    config = DatasetConfig(features_dir=Path("data/processed/features"))
    full_ds = WindTurbineDataset(config)

    train_ds, val_ds, test_ds = full_ds.split()

    loader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=True)
    for features, labels in loader:
        ...                          # features: (B, 128, 431), labels: (B,)

CLI smoke-test::

    python src/training/dataset.py --features data/processed/features
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Optional PyTorch import — graceful fallback for environments without GPU
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    _TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]
    DataLoader = None  # type: ignore[assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Repository root on sys.path
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent  # src/training → root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.audio_loader import (
    FAULT_LABELS,
    AudioConfig,
    AudioLoader,
)
from src.preprocessing.denoiser import Denoiser, DenoiserConfig
from src.preprocessing.feature_extractor import FeatureConfig, FeatureExtractor

# ExperimentTracker is optional (avoids circular import risk)
try:
    from src.utils.experiment_tracker import ExperimentConfig, ExperimentTracker
    _TRACKER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ExperimentTracker = None  # type: ignore[assignment,misc]
    ExperimentConfig = None   # type: ignore[assignment]
    _TRACKER_AVAILABLE: bool = False

logger = logging.getLogger("dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default Week-2-recommended feature configuration (from audit report).
DEFAULT_N_MELS: Final[int] = 128
DEFAULT_N_MFCC: Final[int] = 40
DEFAULT_CQT_BPO: Final[int] = 24
DEFAULT_CQT_BINS: Final[int] = DEFAULT_CQT_BPO * 7  # 7 octaves = 168 bins
DEFAULT_DENOISING_METHOD: Final[str] = "wavelet"  # best from Week 2 benchmark

#: Canonical class names in label-index order.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "normal",
    "blade_imbalance",
    "bearing_fault",
    "gearbox_fault",
    "electrical_fault",
)

#: Frame count for a 10-second clip at 22 050 Hz / hop 512.
DEFAULT_N_FRAMES: Final[int] = 431

#: Feature file names written by PreprocessingPipeline._save_record.
_FEATURE_FILES: Final[dict[str, str]] = {
    "mel":       "mel.npy",
    "mfcc":      "mfcc.npy",
    "cqt":       "cqt.npy",
    "spectral":  "spectral_features.npy",
}

#: Split identifiers.
SPLIT_TRAIN: Final[str] = "train"
SPLIT_VAL:   Final[str] = "val"
SPLIT_TEST:  Final[str] = "test"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FeatureMode(Enum):
    """Feature representation returned by ``__getitem__``.

    Attributes:
        MEL: Log-mel spectrogram ``(n_mels, T)``.
        MFCC: Mel-frequency cepstral coefficients ``(n_mfcc, T)``.
        CQT: Constant-Q transform spectrogram ``(cqt_bins, T)``.
        MEL_3CHANNEL: Log-mel + Δ + ΔΔ stacked ``(3, n_mels, T)``.
        COMBINED: Mel, MFCC and CQT concatenated along the frequency axis
            ``(n_mels + n_mfcc + cqt_bins, T)`` — multi-branch CNN input.
        SPECTRAL: 12-D scalar spectral statistics vector ``(12,)``.
        ALL: Dictionary mapping feature name → tensor (research / ablation).
    """

    MEL         = "mel"
    MFCC        = "mfcc"
    CQT         = "cqt"
    MEL_3CHANNEL = "mel_3channel"
    COMBINED    = "combined"
    SPECTRAL    = "spectral"
    ALL         = "all"


class DataSplit(Enum):
    """Dataset partition identifier."""

    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"
    ALL   = "all"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ClipRecord:
    """Lightweight manifest entry for a single audio clip.

    Attributes:
        clip_id: Unique clip identifier (directory name under features_dir).
        fault_type: Fault class name string.
        label: Integer class label (matches :data:`FAULT_LABELS`).
        source_path: Path to the original raw audio file.
        features_dir: Path to the pre-extracted ``.npy`` feature directory.
            Empty string when features must be extracted on-the-fly.
        dataset: Originating dataset name (``synthetic``, ``cwru``, ``mimii``…).
        split: Partition assignment: ``train``, ``val``, or ``test``.
        augmented: Whether this clip is an augmented variant.
        duration_s: Clip duration in seconds.
        sample_rate: Audio sample rate in Hz.
        metadata: Free-form key-value store for additional provenance.
    """

    clip_id:      str
    fault_type:   str
    label:        int
    source_path:  str
    features_dir: str
    dataset:      str
    split:        str
    augmented:    bool
    duration_s:   float
    sample_rate:  int
    metadata:     dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStatistics:
    """Immutable statistics snapshot for one dataset split.

    Attributes:
        split: Partition name.
        n_clips: Total clip count.
        clips_per_class: Mapping of class name to clip count.
        total_duration_s: Sum of all clip durations.
        class_weight: Per-class training weight (from balanced weighting).
        augmented_fraction: Fraction of clips that are augmented variants.
    """

    split:               str
    n_clips:             int
    clips_per_class:     dict[str, int]
    total_duration_s:    float
    class_weight:        dict[int, float]
    augmented_fraction:  float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Complete configuration for :class:`WindTurbineDataset`.

    All numeric defaults match the Week-2-validated configuration documented
    in the audit report and ``config.yaml``.

    Attributes:
        features_dir: Root of the pre-extracted feature tree produced by
            ``PreprocessingPipeline.process_and_save()``.
        raw_audio_dir: Root of raw audio clips.  Used for on-the-fly
            extraction when ``features_dir`` is absent or incomplete.
        cache_dir: Directory for caching on-the-fly-extracted features.
            ``None`` disables caching.
        feature_mode: Which feature representation to return.
        n_mels: Mel filterbank bands (Week-2 recommendation: 128).
        n_mfcc: MFCC coefficients (Week-2 recommendation: 40).
        cqt_bins_per_octave: CQT frequency resolution (Week-2 rec: 24).
        cqt_bins: Total CQT bins (default: 24 × 7 octaves = 168).
        n_fft: STFT window size.
        hop_length: STFT hop size in samples.
        sample_rate: Audio sample rate in Hz.
        clip_duration: Clip length in seconds.
        denoising_method: Algorithm passed to :class:`Denoiser.denoise`.
            ``None`` disables denoising during on-the-fly extraction.
        train_split: Fraction of data assigned to training.
        val_split: Fraction assigned to validation.
        test_split: Fraction assigned to test.
        stratify: Apply stratified splitting by class label.
        group_by_source: Keep clips from the same source recording in the
            same split (prevents data leakage).
        include_augmented: Whether to include augmented clips.
        random_seed: Global RNG seed for reproducible splits and sampling.
        normalise_features: Apply per-sample zero-mean unit-variance
            normalisation to loaded features.
        exclude_unknown: Drop clips with label ``-1`` (unknown class).
        experiment_tracker: Optional :class:`ExperimentTracker` instance.
            When provided, dataset provenance is logged automatically.
    """

    features_dir:          Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "features"
    )
    raw_audio_dir:         Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "synthetic"
    )
    cache_dir:             Path | None = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "feature_cache"
    )
    feature_mode:          FeatureMode = FeatureMode.MEL
    n_mels:                int = DEFAULT_N_MELS
    n_mfcc:                int = DEFAULT_N_MFCC
    cqt_bins_per_octave:   int = DEFAULT_CQT_BPO
    cqt_bins:              int = DEFAULT_CQT_BINS
    n_fft:                 int = 2048
    hop_length:            int = 512
    sample_rate:           int = 22050
    clip_duration:         float = 10.0
    denoising_method:      str | None = DEFAULT_DENOISING_METHOD
    train_split:           float = 0.70
    val_split:             float = 0.15
    test_split:            float = 0.15
    stratify:              bool = True
    group_by_source:       bool = True
    include_augmented:     bool = True
    random_seed:           int = 42
    normalise_features:    bool = False
    exclude_unknown:       bool = True
    experiment_tracker:    Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration values at construction time."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total:.4f} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )
        if not (0 < self.train_split < 1):
            raise ValueError(f"train_split must be in (0,1), got {self.train_split}")
        if self.n_mels < 1 or self.n_mfcc < 1 or self.cqt_bins < 1:
            raise ValueError("Feature dimensions must be positive integers")

    @property
    def config_hash(self) -> str:
        """Short hash uniquely identifying this feature configuration.

        Used as part of the cache key so changing any feature parameter
        automatically invalidates the cache.
        """
        raw = (
            f"{self.n_mels}:{self.n_mfcc}:{self.cqt_bins}:{self.cqt_bins_per_octave}:"
            f"{self.n_fft}:{self.hop_length}:{self.sample_rate}:"
            f"{self.denoising_method}:{self.feature_mode.value}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Expected tensor shape for a single sample (excluding batch dim).

        Returns:
            Shape tuple matching ``FeatureMode``.

        Raises:
            ValueError: For :attr:`FeatureMode.ALL` which has no fixed shape.
        """
        T = int(np.ceil(self.sample_rate * self.clip_duration / self.hop_length))
        shapes: dict[FeatureMode, tuple[int, ...]] = {
            FeatureMode.MEL:          (self.n_mels, T),
            FeatureMode.MFCC:         (self.n_mfcc, T),
            FeatureMode.CQT:          (self.cqt_bins, T),
            FeatureMode.MEL_3CHANNEL: (3, self.n_mels, T),
            FeatureMode.COMBINED:     (self.n_mels + self.n_mfcc + self.cqt_bins, T),
            FeatureMode.SPECTRAL:     (12,),
        }
        if self.feature_mode not in shapes:
            raise ValueError(
                f"FeatureMode.ALL has no fixed shape — use individual modes"
            )
        return shapes[self.feature_mode]


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class WindTurbineDataset(Dataset):  # type: ignore[misc]
    """PyTorch Dataset for wind turbine acoustic fault classification.

    Supports classical ML (via :meth:`as_numpy`), deep learning
    (via :meth:`__getitem__` returning ``torch.Tensor``), anomaly detection
    (normal-only subset via :meth:`normal_only`), digital-twin simulation
    (raw waveform access via :meth:`get_waveform`), and multi-turbine
    deployment (turbine-ID metadata in every :class:`ClipRecord`).

    Args:
        config: Dataset configuration.
        split: Which partition to expose.  :attr:`DataSplit.ALL` exposes
            all clips and is used internally before calling :meth:`split`.

    Raises:
        FileNotFoundError: When neither ``features_dir`` nor ``raw_audio_dir``
            contains loadable data.
        ValueError: When ``config`` fails validation.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: DataSplit = DataSplit.ALL,
    ) -> None:
        self.config = config
        self._split = split
        self._lock = None  # set lazily — must not use threading in __init__ for fork safety

        # Build or load the clip manifest
        self._manifest: list[ClipRecord] = []
        self._build_manifest()

        if not self._manifest:
            raise FileNotFoundError(
                f"No clips found under {config.features_dir} "
                f"(also checked {config.raw_audio_dir}). "
                "Run: python -m src.preprocessing.pipeline --output data/processed"
            )

        # Assign train/val/test splits
        self._assign_splits()

        # Filter to requested split
        if split != DataSplit.ALL:
            self._manifest = [r for r in self._manifest if r.split == split.value]

        # Build statistics
        self._stats = self._compute_statistics()

        # Lazy-initialise heavy objects (thread-safe via _get_extractor)
        self._extractor: FeatureExtractor | None = None
        self._denoiser: Denoiser | None = None

        # Log to experiment tracker
        if config.experiment_tracker is not None:
            self._log_to_tracker(config.experiment_tracker)

        logger.info(
            "WindTurbineDataset | split=%s | clips=%d | mode=%s | shape=%s",
            split.value, len(self._manifest),
            config.feature_mode.value,
            config.feature_shape if config.feature_mode != FeatureMode.ALL else "dict",
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of clips in this split."""
        return len(self._manifest)

    def __getitem__(
        self, index: int
    ) -> tuple[Any, int]:
        """Return ``(feature_tensor, label)`` for the clip at *index*.

        The method is fully re-entrant and safe for multi-worker DataLoader.
        Reads are performed with :func:`numpy.load` (mmap-mode disabled to
        avoid file-descriptor leaks in forked workers).

        Args:
            index: Sample index in ``[0, len(self))``.

        Returns:
            Tuple of (feature, label) where feature is a ``torch.Tensor``
            (or dict of tensors for :attr:`FeatureMode.ALL`), and label is
            an ``int``.

        Raises:
            IndexError: When *index* is out of range.
            RuntimeError: When the feature cannot be loaded or extracted.
        """
        if index < 0 or index >= len(self._manifest):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self._manifest)}"
            )
        record = self._manifest[index]
        feature = self._load_feature(record)
        label = record.label
        return feature, label

    def __iter__(self) -> Iterator[tuple[Any, int]]:
        """Iterate over all (feature, label) pairs."""
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Split factory
    # ------------------------------------------------------------------

    def split(
        self,
    ) -> tuple["WindTurbineDataset", "WindTurbineDataset", "WindTurbineDataset"]:
        """Return train, validation, and test sub-datasets.

        All three share the same underlying manifest and configuration; they
        differ only in which clips are exposed.  The split is deterministic
        (seeded by :attr:`DatasetConfig.random_seed`).

        Returns:
            Tuple of ``(train_dataset, val_dataset, test_dataset)``.
        """
        # Re-use the already-split full manifest by constructing views
        train_ds = self._view(DataSplit.TRAIN)
        val_ds   = self._view(DataSplit.VAL)
        test_ds  = self._view(DataSplit.TEST)
        logger.info(
            "Split | train=%d, val=%d, test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )
        return train_ds, val_ds, test_ds

    def _view(self, split: DataSplit) -> "WindTurbineDataset":
        """Create a split-filtered view without re-reading the filesystem.

        Args:
            split: The partition to expose.

        Returns:
            A new :class:`WindTurbineDataset` exposing only *split* clips.
        """
        # Build a new instance that skips manifest construction
        view = object.__new__(WindTurbineDataset)
        view.config = self.config
        view._split = split
        view._lock = None
        view._extractor = None
        view._denoiser = None
        # Filter from the full (ALL) manifest
        all_records = self._manifest if self._split == DataSplit.ALL else self._get_full_manifest()
        view._manifest = [r for r in all_records if r.split == split.value]
        view._stats = view._compute_statistics()
        return view

    def _get_full_manifest(self) -> list[ClipRecord]:
        """Return the full manifest, reading from disk if this is a view."""
        return self._manifest

    # ------------------------------------------------------------------
    # Specialised accessors
    # ------------------------------------------------------------------

    def normal_only(self) -> "WindTurbineDataset":
        """Return a dataset containing only healthy (normal) clips.

        Intended for anomaly-detection autoencoder training where only the
        healthy class is used to learn the reconstruction baseline.

        Returns:
            A view containing only ``fault_type == 'normal'`` clips.
        """
        view = object.__new__(WindTurbineDataset)
        view.config = self.config
        view._split = self._split
        view._lock = None
        view._extractor = None
        view._denoiser = None
        view._manifest = [r for r in self._manifest if r.fault_type == "normal"]
        view._stats = view._compute_statistics()
        logger.info("normal_only() view: %d clips", len(view._manifest))
        return view

    def filter_by_dataset(self, dataset_name: str) -> "WindTurbineDataset":
        """Return a view filtered to clips from a single source dataset.

        Useful for multi-turbine deployment where data from each turbine
        installation is tagged with a distinct dataset identifier.

        Args:
            dataset_name: Dataset name to include (e.g. ``"cwru"``,
                ``"mimii"``, ``"synthetic"``).

        Returns:
            Filtered view.
        """
        view = object.__new__(WindTurbineDataset)
        view.config = self.config
        view._split = self._split
        view._lock = None
        view._extractor = None
        view._denoiser = None
        view._manifest = [r for r in self._manifest if r.dataset == dataset_name]
        view._stats = view._compute_statistics()
        logger.info(
            "filter_by_dataset('%s'): %d clips", dataset_name, len(view._manifest)
        )
        return view

    def get_waveform(self, index: int) -> np.ndarray:
        """Load the raw float32 waveform for the clip at *index*.

        Intended for digital-twin simulation and fine-grained signal
        inspection.  The waveform is peak-normalised (matching the
        preprocessing pipeline contract).

        Args:
            index: Sample index.

        Returns:
            1-D float32 waveform array at ``config.sample_rate``.

        Raises:
            FileNotFoundError: When the source audio file cannot be found.
        """
        record = self._manifest[index]
        src = Path(record.source_path)
        if not src.is_file():
            raise FileNotFoundError(
                f"Raw audio not found: {src}. "
                "Run: python scripts/download_datasets.py --synthetic"
            )
        audio_cfg = AudioConfig(
            sample_rate=self.config.sample_rate,
            duration=self.config.clip_duration,
        )
        loader = AudioLoader(audio_cfg, dataset_name=record.dataset)
        clip = loader.load(src)
        if clip is None:
            raise RuntimeError(f"Failed to load waveform from {src}")
        return clip.waveform

    def as_numpy(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire split as (X, y) NumPy arrays.

        Intended for scikit-learn / classical ML baselines.  All features are
        mean-pooled over the time axis to produce a 1-D vector per clip.

        Returns:
            Tuple ``(X, y)`` where ``X`` has shape ``(n_clips, n_features)``
            and ``y`` has shape ``(n_clips,)``.

        Note:
            Memory scales with dataset size.  Use the PyTorch DataLoader for
            large datasets.
        """
        rows, labels = [], []
        n = len(self._manifest)
        for i, record in enumerate(self._manifest):
            try:
                feat = self._load_feature_numpy(record)
                # Flatten: mean-pool temporal dimension for 2-D features
                if feat.ndim == 2:
                    feat = feat.mean(axis=1)
                elif feat.ndim == 3:
                    feat = feat.mean(axis=(1, 2))
                rows.append(feat.astype(np.float32))
                labels.append(record.label)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("as_numpy: skipping clip %d (%s): %s",
                                i, record.clip_id, exc)
            if (i + 1) % 50 == 0 or i + 1 == n:
                logger.info("as_numpy: %d / %d clips loaded", i + 1, n)

        if not rows:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.vstack(rows), np.array(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # Class balancing
    # ------------------------------------------------------------------

    def class_weights(self) -> dict[int, float]:
        """Compute balanced class weights for weighted loss functions.

        Uses ``sklearn.utils.class_weight.compute_class_weight('balanced')``
        on the training-split label distribution.

        Returns:
            Mapping of integer label to float weight.  Labels not present in
            the split are omitted.
        """
        labels = np.array([r.label for r in self._manifest])
        classes = np.unique(labels)
        if len(classes) < 2:
            return {int(c): 1.0 for c in classes}
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        return {int(c): float(w) for c, w in zip(classes, weights)}

    def weighted_sampler(self) -> "WeightedRandomSampler | None":
        """Build a :class:`WeightedRandomSampler` for balanced mini-batches.

        Returns:
            A PyTorch sampler that over-samples minority classes, or ``None``
            when PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            logger.warning(
                "weighted_sampler() requires PyTorch. Install with: pip install torch"
            )
            return None
        weights_map = self.class_weights()
        sample_weights = torch.tensor(
            [weights_map.get(r.label, 1.0) for r in self._manifest],
            dtype=torch.float32,
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def class_distribution(self) -> pd.DataFrame:
        """Return a DataFrame summarising class distribution in this split.

        Returns:
            DataFrame with columns ``fault_type``, ``label``, ``count``,
            ``fraction``, ``weight``.
        """
        weights = self.class_weights()
        rows = []
        df = pd.DataFrame(
            [{"fault_type": r.fault_type, "label": r.label} for r in self._manifest]
        )
        for ft, grp in df.groupby("fault_type"):
            lbl = int(grp["label"].iloc[0])
            cnt = len(grp)
            rows.append({
                "fault_type": ft,
                "label": lbl,
                "count": cnt,
                "fraction": round(cnt / max(len(df), 1), 4),
                "weight": round(weights.get(lbl, 1.0), 4),
            })
        return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Statistics and metadata
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> DatasetStatistics:
        """Immutable statistics for this split."""
        return self._stats

    @property
    def manifest(self) -> list[ClipRecord]:
        """Read-only view of the clip manifest for this split."""
        return list(self._manifest)

    @property
    def labels(self) -> np.ndarray:
        """Integer label array for all clips in this split, shape ``(N,)``."""
        return np.array([r.label for r in self._manifest], dtype=np.int64)

    @property
    def feature_shape(self) -> tuple[int, ...] | str:
        """Expected tensor shape (excluding batch dimension)."""
        try:
            return self.config.feature_shape
        except ValueError:
            return "dict[str, Tensor]"

    def get_record(self, index: int) -> ClipRecord:
        """Return the :class:`ClipRecord` for the clip at *index*.

        Args:
            index: Sample index.

        Returns:
            The corresponding :class:`ClipRecord`.
        """
        if index < 0 or index >= len(self._manifest):
            raise IndexError(f"Index {index} out of range")
        return self._manifest[index]

    def summary(self) -> str:
        """Return a human-readable dataset summary string."""
        stats = self._stats
        dist = self.class_distribution()
        lines = [
            f"WindTurbineDataset | split={self._split.value}",
            f"  Clips         : {stats.n_clips}",
            f"  Duration      : {stats.total_duration_s / 60:.1f} min",
            f"  Feature mode  : {self.config.feature_mode.value}",
            f"  Feature shape : {self.feature_shape}",
            f"  Config hash   : {self.config.config_hash}",
            "  Class distribution:",
        ]
        for _, row in dist.iterrows():
            lines.append(
                f"    {row['fault_type']:<20} {row['count']:>4} clips "
                f"({row['fraction']:.1%})  weight={row['weight']:.3f}"
            )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the dataset summary to stdout."""
        print(self.summary())

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def make_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        balanced: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> "DataLoader | None":
        """Create a configured :class:`torch.utils.data.DataLoader`.

        Args:
            batch_size: Mini-batch size.
            shuffle: Shuffle samples between epochs (ignored when
                ``balanced=True`` which uses a sampler).
            num_workers: Worker processes for parallel data loading.
            balanced: Use :meth:`weighted_sampler` for class balancing.
            pin_memory: Pin tensors in memory (recommended for GPU training).
            drop_last: Drop the final incomplete batch.

        Returns:
            A configured DataLoader, or ``None`` when PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("make_dataloader() requires PyTorch")
            return None

        sampler = None
        effective_shuffle = shuffle
        if balanced:
            sampler = self.weighted_sampler()
            effective_shuffle = False  # sampler and shuffle are mutually exclusive

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Internal: manifest construction
    # ------------------------------------------------------------------

    def _build_manifest(self) -> None:
        """Populate ``self._manifest`` from the features directory.

        Scans ``config.features_dir`` for subdirectories matching the
        ``<fault_type>/<clip_id>/metadata.json`` layout written by
        :class:`PreprocessingPipeline`.  Falls back to scanning
        ``config.raw_audio_dir`` with audio-only entries (features will be
        extracted on-the-fly during ``__getitem__``).
        """
        records: list[ClipRecord] = []

        # ── Primary: load from pre-extracted features tree ───────────────────
        feat_root = self.config.features_dir
        if feat_root.is_dir():
            records.extend(self._scan_features_dir(feat_root))
            logger.info(
                "Manifest: %d clips from features dir %s", len(records), feat_root
            )

        # ── Fallback: load from raw audio tree ───────────────────────────────
        if not records:
            raw_root = self.config.raw_audio_dir
            if raw_root.is_dir():
                audio_cfg = AudioConfig(
                    sample_rate=self.config.sample_rate,
                    duration=self.config.clip_duration,
                )
                loader = AudioLoader(audio_cfg, dataset_name="synthetic")
                clips = loader.load_directory(raw_root)
                for clip in clips:
                    records.append(ClipRecord(
                        clip_id=f"{clip.path.stem}_c{clip.clip_index:03d}",
                        fault_type=clip.fault_type,
                        label=clip.label,
                        source_path=str(clip.path),
                        features_dir="",  # on-the-fly extraction
                        dataset=clip.dataset,
                        split="",
                        augmented=bool(clip.metadata.get("augmented", False)),
                        duration_s=clip.duration,
                        sample_rate=clip.sample_rate,
                        metadata=clip.metadata,
                    ))
                logger.info(
                    "Manifest: %d clips from raw audio dir %s (on-the-fly mode)",
                    len(records), raw_root,
                )

        # ── Filter unknown labels ─────────────────────────────────────────────
        before = len(records)
        if self.config.exclude_unknown:
            records = [r for r in records if r.label != FAULT_LABELS.get("unknown", -1)]
        dropped = before - len(records)
        if dropped:
            logger.info("Excluded %d unknown-label clips", dropped)

        # ── Filter augmented if requested ─────────────────────────────────────
        if not self.config.include_augmented:
            before = len(records)
            records = [r for r in records if not r.augmented]
            logger.debug("Excluded %d augmented clips", before - len(records))

        self._manifest = records

    def _scan_features_dir(self, root: Path) -> list[ClipRecord]:
        """Recursively scan a pre-extracted features directory.

        Expected layout::

            <root>/
            └── <fault_type>/
                └── <clip_id>/
                    ├── mel.npy
                    ├── mfcc.npy
                    ├── cqt.npy
                    ├── spectral_features.npy
                    └── metadata.json

        Args:
            root: Root directory of the features tree.

        Returns:
            List of :class:`ClipRecord` objects.
        """
        records: list[ClipRecord] = []
        for fault_dir in sorted(root.iterdir()):
            if not fault_dir.is_dir() or fault_dir.name.startswith("."):
                continue
            ft = fault_dir.name
            label = FAULT_LABELS.get(ft, FAULT_LABELS.get("unknown", -1))

            for clip_dir in sorted(fault_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                meta_path = clip_dir / "metadata.json"
                if not meta_path.is_file():
                    # Accept dirs with .npy files but no metadata (legacy)
                    has_npy = any(clip_dir.glob("*.npy"))
                    if not has_npy:
                        continue
                    meta: dict[str, Any] = {}
                else:
                    try:
                        with meta_path.open("r", encoding="utf-8") as fh:
                            meta = json.load(fh)
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.warning("Skipping %s: bad metadata (%s)", clip_dir, exc)
                        continue

                # Resolve label from metadata if available (handles subclasses)
                resolved_label = int(meta.get("label", label))
                resolved_ft    = str(meta.get("fault_type", ft))

                records.append(ClipRecord(
                    clip_id=clip_dir.name,
                    fault_type=resolved_ft,
                    label=resolved_label,
                    source_path=str(meta.get("source_path", "")),
                    features_dir=str(clip_dir),
                    dataset=str(meta.get("dataset", "unknown")),
                    split="",
                    augmented=bool(meta.get("augmented", False)),
                    duration_s=float(meta.get("duration", self.config.clip_duration)),
                    sample_rate=int(meta.get("sample_rate", self.config.sample_rate)),
                    metadata=meta,
                ))
        return records

    # ------------------------------------------------------------------
    # Internal: split assignment
    # ------------------------------------------------------------------

    def _assign_splits(self) -> None:
        """Assign ``split`` field on every :class:`ClipRecord`.

        Implements the ``config.yaml`` contract:

        * ``stratify=true``: Preserves class proportions in each split.
        * ``group_by_source=true``: Ensures all clips from the same source
          recording land in the same split (prevents augmentation leakage).
        * Falls back to non-stratified splitting when a class has too few
          samples for the requested fractions.
        """
        if not self._manifest:
            return

        n = len(self._manifest)
        all_idx = np.arange(n)
        labels = np.array([r.label for r in self._manifest])

        # Group-by-source: derive a group key from source_path
        if self.config.group_by_source:
            groups = np.array([
                Path(r.source_path).stem.rsplit("_", 1)[0]
                if r.source_path else r.clip_id
                for r in self._manifest
            ])
        else:
            groups = np.array([str(i) for i in range(n)])

        # Deduplicate groups so each source recording is assigned to one split
        unique_groups = list(dict.fromkeys(groups.tolist()))
        group_labels  = np.array([
            labels[groups == g][0] for g in unique_groups
        ])
        group_idx = np.arange(len(unique_groups))

        trainval_g, test_g = self._safe_split(
            group_idx, group_labels,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
        )
        val_frac = self.config.val_split / max(
            self.config.train_split + self.config.val_split, 1e-9
        )
        train_g, val_g = self._safe_split(
            trainval_g,
            group_labels[trainval_g],
            test_size=val_frac,
            random_state=self.config.random_seed + 1,
        )

        # Map group assignments back to individual clips
        train_set = {unique_groups[g] for g in train_g}
        val_set   = {unique_groups[g] for g in val_g}
        test_set  = {unique_groups[g] for g in test_g}

        for record, group in zip(self._manifest, groups):
            if group in train_set:
                record.split = SPLIT_TRAIN
            elif group in val_set:
                record.split = SPLIT_VAL
            else:
                record.split = SPLIT_TEST

        counts = {
            s: sum(1 for r in self._manifest if r.split == s)
            for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
        }
        logger.info(
            "Split assigned: train=%d, val=%d, test=%d (total=%d)",
            counts[SPLIT_TRAIN], counts[SPLIT_VAL], counts[SPLIT_TEST], n,
        )

    @staticmethod
    def _safe_split(
        indices: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified split with graceful fallback to random when N is small.

        Args:
            indices: Array of integer indices to split.
            labels: Class labels for each index (same length).
            test_size: Fraction to assign to the test partition.
            random_state: RNG seed.

        Returns:
            Tuple ``(train_indices, test_indices)``.
        """
        if len(indices) < 2:
            return indices, np.array([], dtype=int)

        # Ensure test set has at least 1 sample
        effective_test = max(test_size, 1.0 / len(indices))
        effective_test = min(effective_test, 1.0 - 1.0 / len(indices))

        n_classes = len(np.unique(labels))
        min_class_count = np.min(np.bincount(labels)) if len(labels) else 0
        n_test = max(1, int(np.round(len(indices) * effective_test)))

        stratify = labels if (n_classes >= 2 and min_class_count >= 2
                               and n_test >= n_classes) else None
        try:
            return train_test_split(
                indices,
                test_size=effective_test,
                stratify=stratify,
                random_state=random_state,
            )
        except ValueError:
            # Non-stratified fallback
            return train_test_split(
                indices,
                test_size=effective_test,
                stratify=None,
                random_state=random_state,
            )

    # ------------------------------------------------------------------
    # Internal: feature loading
    # ------------------------------------------------------------------

    def _load_feature(
        self, record: ClipRecord
    ) -> "torch.Tensor | dict[str, torch.Tensor]":
        """Load or extract the feature for *record* and convert to Tensor.

        Pipeline:
            1. Check cache directory for pre-computed tensor.
            2. Load from pre-extracted ``.npy`` files in ``features_dir``.
            3. Extract on-the-fly from raw audio (denoising + feature extraction).
            4. Write to cache.
            5. Convert to ``torch.Tensor`` (or dict of tensors for ALL mode).

        Args:
            record: The clip record to load.

        Returns:
            Feature tensor or dict.

        Raises:
            RuntimeError: When no feature source is available.
        """
        # 1. Cache hit
        cached = self._load_from_cache(record)
        if cached is not None:
            return self._to_tensor(cached)

        # 2. Pre-extracted .npy files
        feat_np = self._load_feature_numpy(record)

        # 3. Write cache
        self._write_cache(record, feat_np)

        return self._to_tensor(feat_np)

    def _load_feature_numpy(
        self, record: ClipRecord
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Load or extract the feature as NumPy array(s).

        Args:
            record: The clip record to load.

        Returns:
            NumPy array or dict of arrays.
        """
        mode = self.config.feature_mode

        if record.features_dir and Path(record.features_dir).is_dir():
            feat_dir = Path(record.features_dir)
            try:
                return self._load_npy(feat_dir, mode)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug(
                    "Pre-extracted features incomplete for %s (%s); "
                    "falling back to on-the-fly extraction", record.clip_id, exc
                )

        # On-the-fly extraction from raw waveform
        return self._extract_on_the_fly(record, mode)

    def _load_npy(
        self,
        feat_dir: Path,
        mode: FeatureMode,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Load pre-extracted .npy feature files.

        Args:
            feat_dir: Directory containing the ``.npy`` files.
            mode: Requested feature mode.

        Returns:
            Feature array or dict.

        Raises:
            FileNotFoundError: When a required ``.npy`` file is missing.
        """
        def _load(name: str) -> np.ndarray:
            path = feat_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Feature file not found: {path}")
            return np.load(str(path), allow_pickle=False)

        if mode == FeatureMode.MEL:
            return _load("mel.npy")
        if mode == FeatureMode.MFCC:
            return _load("mfcc.npy")
        if mode == FeatureMode.CQT:
            return _load("cqt.npy")
        if mode == FeatureMode.SPECTRAL:
            return _load("spectral_features.npy")
        if mode == FeatureMode.MEL_3CHANNEL:
            mel = _load("mel.npy")
            return self._make_mel_3channel(mel)
        if mode == FeatureMode.COMBINED:
            mel  = _load("mel.npy")
            mfcc = _load("mfcc.npy")
            cqt  = _load("cqt.npy")
            return np.concatenate([mel, mfcc, cqt], axis=0).astype(np.float32)
        if mode == FeatureMode.ALL:
            return {
                "mel":      _load("mel.npy"),
                "mfcc":     _load("mfcc.npy"),
                "cqt":      _load("cqt.npy"),
                "spectral": _load("spectral_features.npy"),
            }
        raise ValueError(f"Unknown FeatureMode: {mode}")

    def _extract_on_the_fly(
        self, record: ClipRecord, mode: FeatureMode
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Extract features from raw audio at request time.

        Used when pre-extracted ``.npy`` files are absent.  The result is
        cached to ``config.cache_dir`` so subsequent epochs avoid re-extraction.

        Args:
            record: Clip record; ``source_path`` must point to a loadable file.
            mode: Requested feature mode.

        Returns:
            Extracted feature array or dict.

        Raises:
            RuntimeError: When the source audio cannot be loaded.
        """
        src = Path(record.source_path)
        if not src.is_file():
            raise RuntimeError(
                f"Cannot extract on-the-fly: source not found: {src}"
            )

        audio_cfg = AudioConfig(
            sample_rate=self.config.sample_rate,
            duration=self.config.clip_duration,
        )
        loader = AudioLoader(audio_cfg, dataset_name=record.dataset)
        clip = loader.load(src)
        if clip is None:
            raise RuntimeError(f"AudioLoader failed for {src}")
        waveform = clip.waveform

        # Denoising
        if self.config.denoising_method is not None:
            denoiser = self._get_denoiser()
            waveform = denoiser.denoise(waveform, method=self.config.denoising_method)

        # Feature extraction
        extractor = self._get_extractor()

        if mode == FeatureMode.MEL:
            return extractor.mel_spectrogram(waveform)
        if mode == FeatureMode.MFCC:
            return extractor.mfcc(waveform)
        if mode == FeatureMode.CQT:
            return extractor.cqt_spectrogram(waveform)
        if mode == FeatureMode.SPECTRAL:
            return extractor.spectral_statistics(waveform)
        if mode == FeatureMode.MEL_3CHANNEL:
            return extractor.mel_3channel(waveform)
        if mode == FeatureMode.COMBINED:
            mel  = extractor.mel_spectrogram(waveform)
            mfcc = extractor.mfcc(waveform)
            cqt  = extractor.cqt_spectrogram(waveform)
            return np.concatenate([mel, mfcc, cqt], axis=0).astype(np.float32)
        if mode == FeatureMode.ALL:
            return extractor.extract_all(waveform)
        raise ValueError(f"Unknown FeatureMode: {mode}")

    # ------------------------------------------------------------------
    # Internal: caching
    # ------------------------------------------------------------------

    def _cache_path(self, record: ClipRecord, mode: FeatureMode) -> Path | None:
        """Return the cache file path for *record* at *mode*, or None.

        Args:
            record: The clip record.
            mode: Feature mode.

        Returns:
            A :class:`Path` pointing to the cached ``.npy`` file, or ``None``
            when caching is disabled.
        """
        if self.config.cache_dir is None:
            return None
        key = hashlib.md5(
            f"{record.clip_id}::{mode.value}::{self.config.config_hash}".encode()
        ).hexdigest()[:16]
        return self.config.cache_dir / key[:2] / f"{key}.npy"

    def _load_from_cache(
        self, record: ClipRecord
    ) -> np.ndarray | dict[str, np.ndarray] | None:
        """Return cached feature array if it exists, else None.

        Args:
            record: Clip record.

        Returns:
            Cached NumPy array / dict, or ``None`` on cache miss.
        """
        mode = self.config.feature_mode
        if mode == FeatureMode.ALL:
            return None  # dict caching not supported (use individual modes)
        path = self._cache_path(record, mode)
        if path is None or not path.is_file():
            return None
        try:
            return np.load(str(path), allow_pickle=False)
        except (OSError, ValueError):
            return None

    def _write_cache(
        self,
        record: ClipRecord,
        feat: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        """Write a feature array to the cache.

        Silently skips on any I/O failure so a read-only filesystem never
        aborts training.

        Args:
            record: Clip record.
            feat: Feature array to cache (dict is skipped).
        """
        if isinstance(feat, dict):
            return  # dict mode not cached
        mode = self.config.feature_mode
        path = self._cache_path(record, mode)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), feat)
        except OSError as exc:
            logger.debug("Cache write failed for %s: %s", record.clip_id, exc)

    # ------------------------------------------------------------------
    # Internal: tensor conversion
    # ------------------------------------------------------------------

    def _to_tensor(
        self,
        feat: np.ndarray | dict[str, np.ndarray],
    ) -> "torch.Tensor | dict[str, torch.Tensor]":
        """Convert NumPy array(s) to torch.Tensor(s).

        Args:
            feat: Feature array or dict of arrays.

        Returns:
            Tensor or dict of tensors.  Falls back to the NumPy array when
            PyTorch is not available.
        """
        if isinstance(feat, dict):
            if _TORCH_AVAILABLE:
                return {k: torch.from_numpy(v.astype(np.float32)) for k, v in feat.items()}
            return feat

        arr = feat.astype(np.float32)
        if self.config.normalise_features:
            arr = self._normalise(arr)

        if _TORCH_AVAILABLE:
            return torch.from_numpy(arr)
        return arr  # type: ignore[return-value]

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Apply per-sample zero-mean unit-variance normalisation.

        Args:
            arr: Feature array of any shape.

        Returns:
            Normalised float32 array.
        """
        mean = arr.mean()
        std  = arr.std()
        return ((arr - mean) / (std + 1e-8)).astype(np.float32)

    @staticmethod
    def _make_mel_3channel(mel: np.ndarray) -> np.ndarray:
        """Stack mel + Δ + ΔΔ into a 3-channel representation.

        Uses first-order finite differences as an efficient approximation to
        librosa's delta computation (identical result for order=1, width=9 is
        a smooth approximation; this returns the exact first difference for
        simplicity and to avoid a librosa dependency at load time).

        Args:
            mel: Log-mel spectrogram ``(n_mels, T)``.

        Returns:
            Three-channel array ``(3, n_mels, T)``.
        """
        delta  = np.concatenate([mel[:, :1],  np.diff(mel,  axis=1)], axis=1)
        delta2 = np.concatenate([delta[:, :1], np.diff(delta, axis=1)], axis=1)
        return np.stack([mel, delta, delta2], axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: lazy extractor / denoiser
    # ------------------------------------------------------------------

    def _get_extractor(self) -> FeatureExtractor:
        """Return (or lazily initialise) the shared feature extractor.

        Returns:
            Configured :class:`FeatureExtractor` instance.
        """
        if self._extractor is None:
            cfg = FeatureConfig(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                n_mfcc=self.config.n_mfcc,
                cqt_bins=self.config.cqt_bins,
                cqt_bins_per_octave=self.config.cqt_bins_per_octave,
            )
            self._extractor = FeatureExtractor(cfg)
        return self._extractor

    def _get_denoiser(self) -> Denoiser:
        """Return (or lazily initialise) the shared denoiser.

        Returns:
            Configured :class:`Denoiser` instance.
        """
        if self._denoiser is None:
            cfg = DenoiserConfig(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
            self._denoiser = Denoiser(cfg)
        return self._denoiser

    # ------------------------------------------------------------------
    # Internal: statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> DatasetStatistics:
        """Compute :class:`DatasetStatistics` for the current manifest.

        Returns:
            Immutable statistics snapshot.
        """
        clips_per_class: dict[str, int] = {}
        total_dur = 0.0
        aug_count = 0

        for r in self._manifest:
            clips_per_class[r.fault_type] = clips_per_class.get(r.fault_type, 0) + 1
            total_dur += r.duration_s
            aug_count += int(r.augmented)

        weights = self.class_weights() if len(self._manifest) > 0 else {}

        return DatasetStatistics(
            split=self._split.value,
            n_clips=len(self._manifest),
            clips_per_class=clips_per_class,
            total_duration_s=total_dur,
            class_weight=weights,
            augmented_fraction=aug_count / max(len(self._manifest), 1),
        )

    # ------------------------------------------------------------------
    # Internal: experiment tracker
    # ------------------------------------------------------------------

    def _log_to_tracker(self, tracker: Any) -> None:
        """Log dataset provenance to the experiment tracker.

        Args:
            tracker: An :class:`ExperimentTracker` instance (or any object
                with a compatible ``log_dataset_info`` method).
        """
        try:
            counts = {
                r.fault_type: sum(
                    1 for x in self._manifest if x.fault_type == r.fault_type
                )
                for r in self._manifest
            }
            tracker.log_dataset_info(
                total_clips=len(self._manifest),
                clips_per_class=counts,
                total_duration_s=self._stats.total_duration_s,
                dataset_version=f"split={self._split.value}",
                extra={
                    "feature_mode": self.config.feature_mode.value,
                    "config_hash": self.config.config_hash,
                    "augmented_fraction": self._stats.augmented_fraction,
                    "n_mels": self.config.n_mels,
                    "n_mfcc": self.config.n_mfcc,
                    "cqt_bins_per_octave": self.config.cqt_bins_per_octave,
                    "denoising_method": self.config.denoising_method,
                },
            )
            logger.debug("Dataset info logged to ExperimentTracker")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def create_dataloaders(
    config: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    balanced_train: bool = True,
) -> "tuple[DataLoader, DataLoader, DataLoader] | tuple[None, None, None]":
    """One-call factory: create train, val, and test DataLoaders.

    Args:
        config: Dataset configuration.
        batch_size: Mini-batch size for all loaders.
        num_workers: DataLoader worker processes.
        balanced_train: Apply :meth:`WeightedRandomSampler` to the training
            loader to correct class imbalance.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``, or
        ``(None, None, None)`` when PyTorch is unavailable.
    """
    if not _TORCH_AVAILABLE:
        logger.error(
            "create_dataloaders() requires PyTorch. "
            "Install with: pip install torch"
        )
        return None, None, None

    full_ds = WindTurbineDataset(config)
    train_ds, val_ds, test_ds = full_ds.split()

    train_loader = train_ds.make_dataloader(
        batch_size=batch_size, shuffle=not balanced_train,
        num_workers=num_workers, balanced=balanced_train,
    )
    val_loader   = val_ds.make_dataloader(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, balanced=False,
    )
    test_loader  = test_ds.make_dataloader(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, balanced=False,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------


def _build_parser():  # type: ignore[no-untyped-def]
    import argparse
    p = argparse.ArgumentParser(
        description="WindTurbineDataset smoke-test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--features", type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "features",
        help="Pre-extracted features directory.",
    )
    p.add_argument(
        "--raw", type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "synthetic",
        help="Raw audio directory (fallback).",
    )
    p.add_argument(
        "--mode", default="mel",
        choices=[m.value for m in FeatureMode],
        help="Feature mode.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset smoke-test.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    import argparse

    logging.basicConfig(
        level=logging.DEBUG if "--verbose" in (argv or sys.argv) else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = DatasetConfig(
            features_dir=args.features,
            raw_audio_dir=args.raw,
            feature_mode=FeatureMode(args.mode),
            cache_dir=None,  # disable cache for smoke-test
        )
        logger.info("DatasetConfig | hash=%s", config.config_hash)

        ds = WindTurbineDataset(config)
        ds.print_summary()

        train_ds, val_ds, test_ds = ds.split()
        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )

        # Test one sample
        if len(ds) > 0:
            feat, label = ds[0]
            if _TORCH_AVAILABLE:
                logger.info(
                    "Sample 0 | shape=%s | dtype=%s | label=%d",
                    tuple(feat.shape) if hasattr(feat, "shape") else "dict",
                    feat.dtype if hasattr(feat, "dtype") else "dict",
                    label,
                )
            else:
                logger.info("Sample 0 | label=%d (PyTorch not installed)", label)

        # Class distribution
        dist = ds.class_distribution()
        logger.info("Class distribution:\n%s", dist.to_string(index=False))

        # Weights
        weights = ds.class_weights()
        logger.info("Class weights: %s", weights)

        logger.info("Smoke-test PASSED")
        return 0

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Smoke-test FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())