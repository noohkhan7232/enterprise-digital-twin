#!/usr/bin/env python3
"""Dataset statistics generator for the Wind Turbine Acoustic Monitoring project.

Produces a research-grade dataset report covering clip inventory, per-class
duration statistics, per-clip audio quality metrics, spectral feature
summaries, class-balance analysis, and publication-ready figures.  All
outputs land in ``data/processed/reports/`` and ``docs/figures/`` so every
experiment in the paper is traceable to a fully audited data state.

Why a dedicated statistics script?
-----------------------------------
Models are bounded by data quality.  Missed class imbalance causes silent
recall collapse on minority faults.  Undetected silent clips or clipping
events poison mean-feature estimates.  Unbalanced class durations bias
gradient updates in proportion to their share of total audio.  This script
surfaces all of these risks before a single model weight is updated, and
writes machine-readable artefacts that CI can gate on.

Usage::

    python scripts/generate_dataset_statistics.py
    python scripts/generate_dataset_statistics.py --input data/raw/synthetic
    python scripts/generate_dataset_statistics.py --input data/raw/synthetic \\
        --output data/processed/reports --figures docs/figures --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Repository root on sys.path so src.* imports work when this script is
# invoked from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.audio_loader import (
    FAULT_LABELS,
    AudioClip,
    AudioConfig,
    AudioLoader,
)
from src.preprocessing.feature_extractor import (
    SPECTRAL_STATISTIC_NAMES,
    FeatureConfig,
    FeatureExtractor,
)
from src.preprocessing.pipeline import LOADER_REGISTRY, PipelineConfig

logger = logging.getLogger("generate_dataset_statistics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Classes expected in the synthetic / field dataset.
EXPECTED_CLASSES: Final[tuple[str, ...]] = (
    "normal",
    "bearing_fault",
    "blade_imbalance",
    "gearbox_fault",
)

#: Maximum acceptable class-count imbalance ratio before a warning is raised.
IMBALANCE_THRESHOLD: Final[float] = 0.20

#: Seaborn palette matched to the project's class taxonomy.
CLASS_PALETTE: Final[dict[str, str]] = {
    "normal": "#2196F3",
    "bearing_fault": "#F44336",
    "blade_imbalance": "#FF9800",
    "gearbox_fault": "#4CAF50",
    "electrical_fault": "#9C27B0",
    "abnormal": "#795548",
    "unknown": "#9E9E9E",
}

DPI: Final[int] = 300


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticsConfig:
    """Configuration for the dataset statistics generator.

    Attributes:
        input_dir: Root directory of raw audio clips (recursively traversed).
        output_dir: Directory for JSON / CSV / Markdown reports.
        figures_dir: Directory for PNG figure exports.
        loader_name: Dataset loader key (see ``LOADER_REGISTRY``).
        audio_sample_rate: Sample rate audio is resampled to before analysis.
        audio_duration: Target clip duration in seconds.
        imbalance_threshold: Relative class-size deviation that triggers a
            warning (default 0.20 = 20 %).
        random_seed: Seed for reproducible sampling in figures.
        figure_dpi: Raster export resolution.
        max_clips_per_class_for_features: Cap on clips analysed per class for
            the (expensive) feature extraction pass; use -1 for all clips.
    """

    input_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "synthetic"
    )
    output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed" / "reports"
    )
    figures_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "docs" / "figures"
    )
    loader_name: str = "synthetic"
    audio_sample_rate: int = 22050
    audio_duration: float = 10.0
    imbalance_threshold: float = IMBALANCE_THRESHOLD
    random_seed: int = 42
    figure_dpi: int = DPI
    max_clips_per_class_for_features: int = 50


# ---------------------------------------------------------------------------
# Per-clip audio metrics
# ---------------------------------------------------------------------------


def _compute_audio_metrics(clip: AudioClip) -> dict[str, float]:
    """Compute interpretable per-clip audio quality metrics.

    All metrics are computed from the waveform already in memory; no
    additional file I/O is performed.

    Args:
        clip: Loaded audio clip.

    Returns:
        Dictionary with keys ``rms_energy``, ``peak_amplitude``,
        ``dynamic_range_db``, ``zero_crossing_rate``, ``dc_offset``,
        and ``clipping_fraction``.
    """
    w = clip.waveform.astype(np.float64)
    if w.size == 0:
        return {
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "dynamic_range_db": 0.0,
            "zero_crossing_rate": 0.0,
            "dc_offset": 0.0,
            "clipping_fraction": 0.0,
        }

    rms = float(np.sqrt(np.mean(w**2)))
    peak = float(np.max(np.abs(w)))

    # Dynamic range: difference between loudest and quietest 1/8-second frames
    frame = max(1, clip.sample_rate // 8)
    n_frames = len(w) // frame
    if n_frames >= 2:
        frame_rms = [
            float(np.sqrt(np.mean(w[i * frame: (i + 1) * frame] ** 2)))
            for i in range(n_frames)
        ]
        loudest = max(frame_rms)
        quietest = min(frame_rms)
        dyn_db = (
            20.0 * np.log10((loudest + 1e-12) / (quietest + 1e-12))
            if quietest > 1e-12
            else 0.0
        )
    else:
        dyn_db = 0.0

    # Zero-crossing rate (mean per sample)
    zcr = float(np.mean(np.abs(np.diff(np.sign(w))) / 2))

    dc_offset = float(np.mean(w))

    # Fraction of samples within 1 % of full scale → clipping indicator
    clipping = float(np.mean(np.abs(w) > 0.99))

    return {
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "dynamic_range_db": round(float(dyn_db), 3),
        "zero_crossing_rate": round(zcr, 6),
        "dc_offset": round(dc_offset, 6),
        "clipping_fraction": round(clipping, 6),
    }


# ---------------------------------------------------------------------------
# Per-clip spectral metrics (scipy-only, no librosa dependency at this layer)
# ---------------------------------------------------------------------------


def _compute_spectral_metrics(
    clip: AudioClip,
    n_fft: int = 2048,
    hop: int = 512,
) -> dict[str, float]:
    """Compute per-clip spectral metrics using scipy's STFT.

    Computes frame-wise spectral centroid, bandwidth, and roll-off (85 %)
    directly from the power spectrogram, then returns per-clip means.

    Args:
        clip: Loaded audio clip.
        n_fft: FFT window size.
        hop: Hop size between frames.

    Returns:
        Dictionary with ``spectral_centroid_hz``, ``spectral_bandwidth_hz``,
        ``spectral_rolloff_hz``, and ``spectral_flatness``.
    """
    w = clip.waveform.astype(np.float64)
    sr = clip.sample_rate
    if w.size < n_fft:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
        }

    try:
        freqs, _times, stft = sps.stft(
            w, fs=sr, nperseg=n_fft, noverlap=n_fft - hop
        )
        power = np.abs(stft) ** 2  # shape (n_freqs, n_frames)

        # Per-frame spectral centroid
        power_sum = power.sum(axis=0) + 1e-15
        centroid = float(np.mean((freqs[:, None] * power).sum(axis=0) / power_sum))

        # Per-frame spectral bandwidth (weighted std around centroid)
        frame_centroids = (freqs[:, None] * power).sum(axis=0) / power_sum
        bandwidth = float(
            np.mean(
                np.sqrt(
                    ((freqs[:, None] - frame_centroids[None, :]) ** 2 * power).sum(
                        axis=0
                    )
                    / power_sum
                )
            )
        )

        # Spectral roll-off (85 % of cumulative energy)
        cumulative = np.cumsum(power, axis=0)
        threshold = 0.85 * cumulative[-1, :]
        rolloff_indices = np.argmax(cumulative >= threshold[None, :], axis=0)
        rolloff = float(np.mean(freqs[rolloff_indices]))

        # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
        mean_power = power.mean(axis=1) + 1e-15
        geo_mean = float(np.exp(np.mean(np.log(mean_power + 1e-15))))
        arith_mean = float(np.mean(mean_power))
        flatness = round(geo_mean / arith_mean, 6) if arith_mean > 0 else 0.0

    except (ValueError, RuntimeError) as exc:
        logger.debug("STFT failed for %s: %s", clip.path.name, exc)
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
        }

    return {
        "spectral_centroid_hz": round(centroid, 3),
        "spectral_bandwidth_hz": round(bandwidth, 3),
        "spectral_rolloff_hz": round(rolloff, 3),
        "spectral_flatness": flatness,
    }


# ---------------------------------------------------------------------------
# Feature dimension probe
# ---------------------------------------------------------------------------


def _probe_feature_dims(
    clip: AudioClip,
    feature_config: FeatureConfig,
) -> dict[str, list[int]]:
    """Extract feature shapes from a single representative clip.

    Args:
        clip: A healthy clip used as the shape probe (not stored).
        feature_config: Feature extraction configuration.

    Returns:
        Mapping of feature name to shape list; empty on failure.
    """
    extractor = FeatureExtractor(feature_config)
    dims: dict[str, list[int]] = {}
    probes = {
        "mel_spectrogram": extractor.mel_spectrogram,
        "mfcc": extractor.mfcc,
        "cqt_spectrogram": extractor.cqt_spectrogram,
        "spectral_statistics": extractor.spectral_statistics,
        "mel_3channel": extractor.mel_3channel,
    }
    for name, fn in probes.items():
        try:
            dims[name] = list(fn(clip.waveform).shape)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Feature probe failed for '%s': %s", name, exc)
    return dims


# ---------------------------------------------------------------------------
# Main statistics engine
# ---------------------------------------------------------------------------


class DatasetStatisticsGenerator:
    """Compute, export, and visualise comprehensive dataset statistics.

    The generator is designed to be re-entrant: calling :meth:`run` on the
    same instance twice with different configurations is safe (state is
    accumulated only within a single :meth:`run` call and returned, not
    stored on the instance).

    Args:
        config: Statistics generator configuration.
    """

    def __init__(self, config: StatisticsConfig) -> None:
        self.config = config
        self._setup_directories()
        sns.set_theme(style="whitegrid", font_scale=1.15)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full statistics pipeline.

        Steps:
            1. Load all clips from ``config.input_dir``.
            2. Compute per-clip audio and spectral metrics.
            3. Probe feature dimensions on one representative clip.
            4. Aggregate into per-class and global summary tables.
            5. Detect class imbalance.
            6. Export JSON / CSV / Markdown reports.
            7. Generate and save all figures.
            8. Write the research dataset report.

        Returns:
            Full report dictionary (also written to disk as JSON).

        Raises:
            SystemExit: If no clips can be loaded from the input directory.
        """
        wall_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("Dataset Statistics Generator")
        logger.info("Input : %s", self.config.input_dir)
        logger.info("Output: %s", self.config.output_dir)
        logger.info("=" * 70)

        # 1. Load -----------------------------------------------------------
        clips = self._load_clips()
        if not clips:
            logger.error(
                "No audio clips found under %s. "
                "Run: python scripts/download_datasets.py --synthetic",
                self.config.input_dir,
            )
            sys.exit(1)

        # 2. Per-clip metrics -----------------------------------------------
        clip_records, audio_df = self._analyse_clips(clips)

        # 3. Feature dimensions --------------------------------------------
        feature_dims = self._probe_features(clips)

        # 4. Aggregate -------------------------------------------------------
        global_stats = self._global_statistics(audio_df)
        class_stats = self._class_statistics(audio_df)
        duration_stats = self._duration_statistics(audio_df)

        # 5. Class balance --------------------------------------------------
        balance = self._class_balance(audio_df)

        # 6. Assemble full report -------------------------------------------
        elapsed = round(time.perf_counter() - wall_start, 2)
        report: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dir": str(self.config.input_dir),
            "elapsed_seconds": elapsed,
            "clip_inventory": {
                "total_clips": len(clips),
                "total_duration_seconds": round(audio_df["duration_s"].sum(), 3),
                "total_duration_hours": round(audio_df["duration_s"].sum() / 3600, 5),
                "classes_found": sorted(audio_df["fault_type"].unique().tolist()),
                "clips_per_class": audio_df["fault_type"].value_counts().to_dict(),
            },
            "duration_statistics": duration_stats,
            "audio_metrics": {
                "global": global_stats,
                "per_class": class_stats,
            },
            "feature_dimensions": feature_dims,
            "class_balance": balance,
            "config": {
                "loader": self.config.loader_name,
                "sample_rate": self.config.audio_sample_rate,
                "clip_duration": self.config.audio_duration,
            },
        }

        # 7. Export ---------------------------------------------------------
        self._export_json(report)
        self._export_csv(audio_df)
        self._export_markdown_summary(report, audio_df, class_stats)

        # 8. Figures --------------------------------------------------------
        self._plot_class_distribution(audio_df)
        self._plot_duration_distribution(audio_df)
        self._plot_rms_distribution(audio_df)
        self._plot_spectral_centroid_distribution(audio_df)

        # 9. Research report ------------------------------------------------
        self._write_research_report(report, audio_df, balance)

        logger.info("=" * 70)
        logger.info(
            "Complete in %.2f s  |  %d clips  |  %.1f min total audio",
            elapsed,
            len(clips),
            audio_df["duration_s"].sum() / 60,
        )
        logger.info("Reports : %s", self.config.output_dir)
        logger.info("Figures : %s", self.config.figures_dir)
        logger.info("=" * 70)
        return report

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------

    def _load_clips(self) -> list[AudioClip]:
        """Load all audio clips from the input directory.

        Returns:
            List of loaded ``AudioClip`` objects.
        """
        audio_cfg = AudioConfig(
            sample_rate=self.config.audio_sample_rate,
            duration=self.config.audio_duration,
        )
        loader_cls = LOADER_REGISTRY.get(self.config.loader_name, AudioLoader)
        if loader_cls is AudioLoader:
            loader: AudioLoader = AudioLoader(
                audio_cfg, dataset_name=self.config.loader_name
            )
        else:
            loader = loader_cls(audio_cfg)

        clips = loader.load_directory(self.config.input_dir)
        logger.info("Loaded %d clips from %s", len(clips), self.config.input_dir)
        return clips

    # ------------------------------------------------------------------
    # Step 2: Per-clip analysis
    # ------------------------------------------------------------------

    def _analyse_clips(
        self, clips: list[AudioClip]
    ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        """Compute audio and spectral metrics for every clip.

        Args:
            clips: All loaded clips.

        Returns:
            Tuple of (list of raw per-clip record dicts, tidy DataFrame).
        """
        records: list[dict[str, Any]] = []
        n = len(clips)
        log_every = max(1, n // 10)

        for idx, clip in enumerate(clips, start=1):
            audio_m = _compute_audio_metrics(clip)
            spectral_m = _compute_spectral_metrics(clip)
            records.append(
                {
                    "path": str(clip.path),
                    "fault_type": clip.fault_type,
                    "label": clip.label,
                    "dataset": clip.dataset,
                    "clip_index": clip.clip_index,
                    "sample_rate": clip.sample_rate,
                    "duration_s": round(clip.duration, 4),
                    "num_samples": clip.num_samples,
                    **audio_m,
                    **spectral_m,
                }
            )
            if idx % log_every == 0 or idx == n:
                logger.info("  Analysed %d / %d clips", idx, n)

        df = pd.DataFrame(records)
        logger.info("Per-clip analysis complete: %d records", len(df))
        return records, df

    # ------------------------------------------------------------------
    # Step 3: Feature dimension probe
    # ------------------------------------------------------------------

    def _probe_features(self, clips: list[AudioClip]) -> dict[str, list[int]]:
        """Probe feature array shapes using one healthy clip.

        Tries clips in order until one succeeds (skips empty / corrupt).

        Args:
            clips: All loaded clips.

        Returns:
            Mapping of feature name to shape list; empty on total failure.
        """
        feature_cfg = FeatureConfig(sample_rate=self.config.audio_sample_rate)
        for clip in clips:
            if clip.num_samples >= feature_cfg.n_fft and np.all(
                np.isfinite(clip.waveform)
            ):
                dims = _probe_feature_dims(clip, feature_cfg)
                if dims:
                    logger.info(
                        "Feature dimensions probed from '%s':", clip.path.name
                    )
                    for name, shape in dims.items():
                        logger.info("  %-22s %s", name, shape)
                    return dims
        logger.warning("Feature dimension probe failed for all clips")
        return {}

    # ------------------------------------------------------------------
    # Step 4a: Global statistics
    # ------------------------------------------------------------------

    def _global_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute global (dataset-wide) audio quality statistics.

        Args:
            df: Per-clip metrics DataFrame.

        Returns:
            Nested dictionary with mean / std / min / max for each metric.
        """
        metric_cols = [
            "rms_energy", "peak_amplitude", "dynamic_range_db",
            "zero_crossing_rate", "spectral_centroid_hz",
            "spectral_bandwidth_hz", "spectral_rolloff_hz", "spectral_flatness",
        ]
        stats: dict[str, Any] = {}
        for col in metric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            stats[col] = {
                "mean": round(float(s.mean()), 6),
                "std": round(float(s.std()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "median": round(float(s.median()), 6),
            }
        return stats

    # ------------------------------------------------------------------
    # Step 4b: Per-class statistics
    # ------------------------------------------------------------------

    def _class_statistics(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Compute per-class mean audio metrics.

        Args:
            df: Per-clip metrics DataFrame.

        Returns:
            Nested dict ``{fault_type: {metric: value}}``.
        """
        metric_cols = [
            "rms_energy", "peak_amplitude", "dynamic_range_db",
            "zero_crossing_rate", "spectral_centroid_hz",
            "spectral_bandwidth_hz", "spectral_rolloff_hz", "spectral_flatness",
            "duration_s",
        ]
        available = [c for c in metric_cols if c in df.columns]
        result: dict[str, dict[str, Any]] = {}
        for fault_type, group in df.groupby("fault_type"):
            result[str(fault_type)] = {
                col: round(float(group[col].mean()), 6)
                for col in available
            }
        return result

    # ------------------------------------------------------------------
    # Step 4c: Duration statistics
    # ------------------------------------------------------------------

    def _duration_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute dataset-level and per-class duration statistics.

        Args:
            df: Per-clip metrics DataFrame.

        Returns:
            Dictionary with global and per-class duration breakdowns.
        """
        dur = df["duration_s"]
        per_class: dict[str, dict[str, float]] = {}
        for ft, grp in df.groupby("fault_type"):
            per_class[str(ft)] = {
                "total_s": round(float(grp["duration_s"].sum()), 3),
                "mean_s": round(float(grp["duration_s"].mean()), 3),
                "min_s": round(float(grp["duration_s"].min()), 3),
                "max_s": round(float(grp["duration_s"].max()), 3),
                "clips": int(len(grp)),
            }
        return {
            "total_seconds": round(float(dur.sum()), 3),
            "total_minutes": round(float(dur.sum() / 60), 3),
            "total_hours": round(float(dur.sum() / 3600), 5),
            "mean_seconds": round(float(dur.mean()), 3),
            "std_seconds": round(float(dur.std()), 3),
            "min_seconds": round(float(dur.min()), 3),
            "max_seconds": round(float(dur.max()), 3),
            "per_class": per_class,
        }

    # ------------------------------------------------------------------
    # Step 5: Class balance
    # ------------------------------------------------------------------

    def _class_balance(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyse class distribution and flag imbalance.

        Imbalance is defined as any class whose clip count deviates more
        than ``config.imbalance_threshold`` from the mean count.

        Args:
            df: Per-clip metrics DataFrame.

        Returns:
            Dictionary with counts, fractions, imbalance ratio, and a list
            of imbalanced classes.
        """
        counts = df["fault_type"].value_counts().to_dict()
        total = sum(counts.values())
        fractions = {k: round(v / total, 4) for k, v in counts.items()}
        mean_count = total / max(len(counts), 1)

        imbalanced: list[str] = [
            ft
            for ft, cnt in counts.items()
            if abs(cnt - mean_count) / mean_count > self.config.imbalance_threshold
        ]

        # Imbalance ratio: max count / min count
        if counts:
            imbalance_ratio = round(max(counts.values()) / max(min(counts.values()), 1), 4)
        else:
            imbalance_ratio = 1.0

        result: dict[str, Any] = {
            "counts": {k: int(v) for k, v in counts.items()},
            "fractions": fractions,
            "imbalance_ratio": imbalance_ratio,
            "imbalanced_classes": imbalanced,
            "is_balanced": len(imbalanced) == 0,
            "threshold_used": self.config.imbalance_threshold,
        }

        if imbalanced:
            logger.warning(
                "CLASS IMBALANCE DETECTED  (threshold %.0f%%): %s",
                self.config.imbalance_threshold * 100,
                ", ".join(imbalanced),
            )
        else:
            logger.info(
                "Class balance OK  (imbalance ratio = %.2f)", imbalance_ratio
            )
        return result

    # ------------------------------------------------------------------
    # Step 6: Export reports
    # ------------------------------------------------------------------

    def _export_json(self, report: dict[str, Any]) -> None:
        """Write the full report as indented JSON.

        Args:
            report: Complete statistics dictionary.
        """
        path = self.config.output_dir / "dataset_statistics.json"
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=_json_default)
            logger.info("JSON report -> %s", path)
        except OSError as exc:
            logger.error("Failed to write JSON report: %s", exc)

    def _export_csv(self, df: pd.DataFrame) -> None:
        """Write per-clip metrics as a flat CSV.

        Args:
            df: Per-clip metrics DataFrame.
        """
        path = self.config.output_dir / "dataset_statistics.csv"
        try:
            df.to_csv(path, index=False)
            logger.info("CSV report  -> %s  (%d rows, %d cols)", path, len(df), len(df.columns))
        except OSError as exc:
            logger.error("Failed to write CSV: %s", exc)

    def _export_markdown_summary(
        self,
        report: dict[str, Any],
        df: pd.DataFrame,
        class_stats: dict[str, dict[str, Any]],
    ) -> None:
        """Write a human-readable Markdown summary.

        Args:
            report: Full report dictionary.
            df: Per-clip DataFrame.
            class_stats: Per-class mean metrics.
        """
        path = self.config.output_dir / "dataset_summary.md"
        inv = report["clip_inventory"]
        dur = report["duration_statistics"]
        bal = report["class_balance"]
        feat = report["feature_dimensions"]

        # Per-class table rows
        class_rows = ""
        for ft in sorted(class_stats):
            m = class_stats[ft]
            cnt = bal["counts"].get(ft, 0)
            frac = bal["fractions"].get(ft, 0.0)
            class_rows += (
                f"| {ft} | {cnt} | {frac:.1%} | "
                f"{m.get('duration_s', 0):.2f} s | "
                f"{m.get('rms_energy', 0):.4f} | "
                f"{m.get('spectral_centroid_hz', 0):.0f} Hz |\n"
            )

        # Feature dimension table
        feat_rows = ""
        for name, shape in feat.items():
            feat_rows += f"| `{name}` | `{shape}` | {_numel(shape):,} |\n"

        # Global metrics table
        gm = report["audio_metrics"]["global"]
        global_rows = ""
        for metric, vals in gm.items():
            global_rows += (
                f"| {metric} | {vals['mean']:.4f} | {vals['std']:.4f} | "
                f"{vals['min']:.4f} | {vals['max']:.4f} |\n"
            )

        balance_note = (
            "✅ Dataset is **balanced** within the "
            f"{self.config.imbalance_threshold:.0%} threshold."
            if bal["is_balanced"]
            else (
                f"⚠️ **Imbalance detected** in: "
                f"{', '.join(bal['imbalanced_classes'])}. "
                f"Imbalance ratio = {bal['imbalance_ratio']:.2f}."
            )
        )

        md = textwrap.dedent(f"""\
            # Dataset Summary — Wind Turbine Acoustic Monitoring

            Generated: `{report['generated_at']}`
            Input directory: `{report['input_dir']}`

            ---

            ## Clip Inventory

            | Metric | Value |
            |--------|-------|
            | Total clips | **{inv['total_clips']}** |
            | Total duration | **{dur['total_minutes']:.1f} min** ({dur['total_hours']:.4f} h) |
            | Mean clip duration | {dur['mean_seconds']:.2f} s |
            | Min clip duration | {dur['min_seconds']:.2f} s |
            | Max clip duration | {dur['max_seconds']:.2f} s |
            | Classes found | {', '.join(f'`{c}`' for c in sorted(inv['classes_found']))} |

            ---

            ## Class Distribution

            {balance_note}

            Imbalance ratio (max / min class count): **{bal['imbalance_ratio']:.2f}**

            | Class | Clips | Share | Mean Duration | Mean RMS | Mean Centroid |
            |-------|------:|------:|:-------------:|:--------:|:-------------:|
            {class_rows}
            ---

            ## Audio Quality Metrics (Global)

            | Metric | Mean | Std | Min | Max |
            |--------|-----:|----:|----:|----:|
            {global_rows}
            ---

            ## Feature Dimensions

            | Feature Set | Shape | Total Elements |
            |-------------|-------|:--------------:|
            {feat_rows}
            ---

            ## Configuration

            | Parameter | Value |
            |-----------|-------|
            | Sample rate | {self.config.audio_sample_rate} Hz |
            | Clip duration | {self.config.audio_duration} s |
            | Loader | `{self.config.loader_name}` |
            | Imbalance threshold | {self.config.imbalance_threshold:.0%} |
            """)

        try:
            path.write_text(md, encoding="utf-8")
            logger.info("Markdown summary -> %s", path)
        except OSError as exc:
            logger.error("Failed to write Markdown summary: %s", exc)

    # ------------------------------------------------------------------
    # Step 7: Figures
    # ------------------------------------------------------------------

    def _save_fig(self, fig: plt.Figure, stem: str) -> None:
        """Save and close a figure.

        Args:
            fig: Matplotlib figure.
            stem: Output filename stem (no extension).
        """
        path = self.config.figures_dir / f"{stem}.png"
        try:
            fig.tight_layout()
            fig.savefig(path, dpi=self.config.figure_dpi, bbox_inches="tight")
            logger.info("Figure saved -> %s", path)
        except OSError as exc:
            logger.error("Failed to save figure %s: %s", path, exc)
        finally:
            plt.close(fig)

    def _class_palette(self, classes: list[str]) -> list[str]:
        """Return per-class colours consistent with the project palette.

        Args:
            classes: Ordered class names.

        Returns:
            List of hex colour strings.
        """
        return [CLASS_PALETTE.get(c, "#607D8B") for c in classes]

    def _plot_class_distribution(self, df: pd.DataFrame) -> None:
        """Bar chart: clips per class (count and share).

        Args:
            df: Per-clip metrics DataFrame.
        """
        counts = df["fault_type"].value_counts().sort_index()
        classes = list(counts.index)
        palette = self._class_palette(classes)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Count bars ---
        ax = axes[0]
        bars = ax.bar(
            [c.replace("_", "\n") for c in classes],
            counts.values,
            color=palette,
            width=0.6,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.bar_label(bars, padding=4, fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Clips", fontsize=12)
        ax.set_title("Clips per Fault Class", fontsize=13, fontweight="bold")
        ax.set_ylim(0, counts.max() * 1.20)
        ax.grid(axis="y", alpha=0.4, zorder=1)

        # --- Pie chart ---
        ax = axes[1]
        wedge_labels = [c.replace("_", "\n") for c in classes]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=wedge_labels,
            colors=palette,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title("Class Share", fontsize=13, fontweight="bold")

        fig.suptitle(
            "Dataset Class Distribution",
            fontsize=14,
            fontweight="bold",
        )
        self._save_fig(fig, "ds_class_distribution")

    def _plot_duration_distribution(self, df: pd.DataFrame) -> None:
        """Histogram of clip durations, faceted by class.

        Args:
            df: Per-clip metrics DataFrame.
        """
        classes = sorted(df["fault_type"].unique())
        palette = self._class_palette(classes)
        n = len(classes)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows),
                                 squeeze=False)
        axes_flat = axes.ravel()

        for idx, (cls, color) in enumerate(zip(classes, palette)):
            ax = axes_flat[idx]
            data = df.loc[df["fault_type"] == cls, "duration_s"]
            ax.hist(data, bins=15, color=color, edgecolor="white",
                    linewidth=0.8, alpha=0.85)
            ax.axvline(data.mean(), color="black", linewidth=1.5,
                       linestyle="--", label=f"mean = {data.mean():.2f}s")
            ax.set_title(cls.replace("_", " ").title(), fontsize=12,
                         fontweight="bold")
            ax.set_xlabel("Duration (s)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.4)

        # Hide unused subplots
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        fig.suptitle("Clip Duration Distribution by Class",
                     fontsize=14, fontweight="bold")
        self._save_fig(fig, "ds_duration_distribution")

    def _plot_rms_distribution(self, df: pd.DataFrame) -> None:
        """Overlaid KDE + rug plot of RMS energy per class.

        Args:
            df: Per-clip metrics DataFrame.
        """
        classes = sorted(df["fault_type"].unique())
        palette = self._class_palette(classes)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Overlaid KDE ---
        ax = axes[0]
        for cls, color in zip(classes, palette):
            data = df.loc[df["fault_type"] == cls, "rms_energy"].dropna()
            if data.empty:
                continue
            ax.hist(data, bins=15, color=color, alpha=0.55,
                    edgecolor="white", linewidth=0.6,
                    label=cls.replace("_", " "), density=True)
        ax.set_xlabel("RMS Energy", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("RMS Energy Distribution (all classes)", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

        # --- Box plot per class ---
        ax = axes[1]
        bp_data = [
            df.loc[df["fault_type"] == cls, "rms_energy"].dropna().values
            for cls in classes
        ]
        bp = ax.boxplot(
            bp_data,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(
            [c.replace("_", "\n") for c in classes], fontsize=10
        )
        ax.set_ylabel("RMS Energy", fontsize=12)
        ax.set_title("RMS Energy — Box Plots per Class", fontsize=13,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.4)

        fig.suptitle("RMS Energy Analysis", fontsize=14, fontweight="bold")
        self._save_fig(fig, "ds_rms_distribution")

    def _plot_spectral_centroid_distribution(self, df: pd.DataFrame) -> None:
        """Violin + strip plot of spectral centroid per class.

        Spectral centroid is the most discriminative single feature for
        separating fault classes (established in Week 1 notebook).

        Args:
            df: Per-clip metrics DataFrame.
        """
        if "spectral_centroid_hz" not in df.columns:
            logger.warning("spectral_centroid_hz missing; skipping figure")
            return

        classes = sorted(df["fault_type"].unique())
        palette_dict = {c: CLASS_PALETTE.get(c, "#607D8B") for c in classes}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # --- Violin ---
        ax = axes[0]
        plot_df = df[["fault_type", "spectral_centroid_hz"]].dropna()
        # Build per-class data lists for matplotlib violin
        vp_data = [
            plot_df.loc[plot_df["fault_type"] == cls, "spectral_centroid_hz"].values
            for cls in classes
        ]
        vp = ax.violinplot(vp_data, positions=range(1, len(classes) + 1),
                           showmedians=True, showextrema=True)
        for pc, cls in zip(vp["bodies"], classes):
            pc.set_facecolor(palette_dict[cls])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=10)
        ax.set_ylabel("Spectral Centroid (Hz)", fontsize=12)
        ax.set_title("Spectral Centroid — Violin Plot", fontsize=13,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))

        # --- Scatter (per-class mean ± std) ---
        ax = axes[1]
        means = [
            plot_df.loc[plot_df["fault_type"] == cls, "spectral_centroid_hz"].mean()
            for cls in classes
        ]
        stds = [
            plot_df.loc[plot_df["fault_type"] == cls, "spectral_centroid_hz"].std()
            for cls in classes
        ]
        xs = range(len(classes))
        ax.bar(xs, means, yerr=stds, color=[palette_dict[c] for c in classes],
               width=0.5, capsize=6, edgecolor="white", linewidth=1.5,
               error_kw={"elinewidth": 1.5, "capthick": 1.5}, zorder=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=10)
        ax.set_ylabel("Spectral Centroid (Hz)", fontsize=12)
        ax.set_title("Mean ± Std Spectral Centroid", fontsize=13,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.4, zorder=1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))

        fig.suptitle("Spectral Centroid Distribution by Fault Class",
                     fontsize=14, fontweight="bold")
        self._save_fig(fig, "ds_spectral_centroid_distribution")

    # ------------------------------------------------------------------
    # Step 8: Research report
    # ------------------------------------------------------------------

    def _write_research_report(
        self,
        report: dict[str, Any],
        df: pd.DataFrame,
        balance: dict[str, Any],
    ) -> None:
        """Write the research-oriented dataset assessment Markdown document.

        Covers dataset strengths, weaknesses, model-training risks, and
        actionable recommendations — structured for direct inclusion in the
        paper's Data section.

        Args:
            report: Full statistics report.
            df: Per-clip metrics DataFrame.
            balance: Class balance analysis dictionary.
        """
        inv = report["clip_inventory"]
        dur = report["duration_statistics"]
        gm = report["audio_metrics"]["global"]
        feat = report["feature_dimensions"]

        total_clips = inv["total_clips"]
        total_min = dur["total_minutes"]
        classes = sorted(inv["classes_found"])
        n_classes = len(classes)
        imbalance_ratio = balance["imbalance_ratio"]
        is_balanced = balance["is_balanced"]

        # Derive per-class clip counts for the report body
        counts = balance["counts"]
        min_class = min(counts, key=counts.get) if counts else "N/A"
        max_class = max(counts, key=counts.get) if counts else "N/A"
        min_cnt = counts.get(min_class, 0)
        max_cnt = counts.get(max_class, 0)

        # Dynamic range statistics
        dr_mean = gm.get("dynamic_range_db", {}).get("mean", 0.0)
        rms_mean = gm.get("rms_energy", {}).get("mean", 0.0)
        rms_std = gm.get("rms_energy", {}).get("std", 0.0)
        centroid_mean = gm.get("spectral_centroid_hz", {}).get("mean", 0.0)

        # Clipping check
        if "clipping_fraction" in df.columns:
            clipped_clips = int((df["clipping_fraction"] > 0.01).sum())
        else:
            clipped_clips = 0

        # Silent check (RMS below noise floor)
        if "rms_energy" in df.columns:
            silent_clips = int((df["rms_energy"] < 1e-4).sum())
        else:
            silent_clips = 0

        # Feature memory footprint
        feat_memory: dict[str, str] = {}
        for name, shape in feat.items():
            n_elem = _numel(shape)
            bytes_fp32 = n_elem * 4
            feat_memory[name] = _human_bytes(bytes_fp32)

        balance_section = (
            "The dataset is **well-balanced**: all classes fall within "
            f"the ±{self.config.imbalance_threshold:.0%} deviation threshold."
            if is_balanced
            else (
                f"The dataset is **imbalanced** (ratio = {imbalance_ratio:.2f}x). "
                f"The largest class (`{max_class}`, {max_cnt} clips) has "
                f"{imbalance_ratio:.2f}× more samples than the smallest "
                f"(`{min_class}`, {min_cnt} clips). "
                "Weighted loss or stratified sampling is required."
            )
        )

        feat_table = "\n".join(
            f"| `{n}` | `{s}` | {_numel(s):,} | {feat_memory.get(n, '?')} |"
            for n, s in feat.items()
        )

        md = textwrap.dedent(f"""\
            # Research Dataset Report — Wind Turbine Acoustic Monitoring

            Generated: `{report['generated_at']}`
            Input: `{report['input_dir']}`

            ---

            ## Executive Summary

            The dataset contains **{total_clips} audio clips** from
            **{n_classes} fault classes** ({', '.join(f'`{c}`' for c in classes)}),
            totalling **{total_min:.1f} minutes** of acoustic recordings at
            {self.config.audio_sample_rate} Hz.  Each clip is
            {dur['mean_seconds']:.1f} s (mean), spanning
            {dur['min_seconds']:.1f}–{dur['max_seconds']:.1f} s.

            ---

            ## 1. Dataset Strengths

            ### 1.1 Acoustic Diversity
            - **{n_classes} fault classes** cover the primary wind-turbine failure
              modes: bearing wear (high-frequency impulse trains), blade imbalance
              (low-frequency 1P modulation), gearbox damage (gear-mesh harmonic
              ladders), and healthy baseline (broadband aerodynamic noise).
            - Mean spectral centroid of **{centroid_mean:,.0f} Hz** confirms
              energy is distributed across a broad frequency band, not confined
              to a single octave — a prerequisite for learning meaningful
              spectral features.
            - Mean dynamic range of **{dr_mean:.1f} dB** indicates the recordings
              contain both quiet background periods and transient fault events,
              enabling the anomaly detector to learn a meaningful reconstruction
              threshold.

            ### 1.2 Feature Richness
            Five feature representations are available for each clip:

            | Feature | Shape | Elements | FP32 memory |
            |---------|-------|:--------:|:-----------:|
            {feat_table}

            The mel spectrogram provides the primary CNN input; MFCCs support
            classical-ML baselines; CQT aligns with harmonic fault ladders;
            the 3-channel stack enables ImageNet-pretrained backbone transfer.

            ### 1.3 Reproducibility
            All clips are generated by a seeded synthetic pipeline
            (`random_seed = 42`), guaranteeing byte-identical datasets across
            machines and enabling exact ablation comparisons in the paper.

            ---

            ## 2. Dataset Weaknesses

            ### 2.1 Synthetic Origin
            The dataset is **entirely synthetic**: signals are constructed from
            sinusoids, impulse trains, and pink noise rather than captured from
            physical turbines.  Domain shift between synthetic and field audio
            is the primary deployment risk (see §4).

            ### 2.2 Compound Fault Coverage
            Each clip contains **a single fault type**.  Real turbines often
            exhibit simultaneous faults (e.g., gearbox wear *and* bearing
            degradation).  Compound fault clips are absent, which may cause
            classifiers to over-specialise to clean single-fault patterns.

            ### 2.3 Fixed Operating Conditions
            Rotor speed, load, and wind speed are fixed per clip rather than
            varying within clips.  Speed-dependent frequency shifts (gear-mesh
            frequency scales linearly with RPM) are not represented, limiting
            generalisation to variable-speed turbines.

            ### 2.4 Signal Quality Flags
            - **Potentially clipped clips:** {clipped_clips} / {total_clips}
              ({clipped_clips / max(total_clips, 1):.1%})
            - **Potentially silent clips:** {silent_clips} / {total_clips}
              ({silent_clips / max(total_clips, 1):.1%})
            - RMS coefficient of variation: {rms_std / max(rms_mean, 1e-12):.2f}
              (> 0.30 indicates high inter-clip loudness variability requiring
              per-clip normalisation in the DataLoader).

            ---

            ## 3. Risks for Model Training

            ### 3.1 Class Balance
            {balance_section}

            **Risk level:** {"🟢 Low" if is_balanced else "🔴 High"}
            **Mitigation:** {"None required." if is_balanced else "Apply class-weighted cross-entropy (`torch.nn.CrossEntropyLoss(weight=...)`) and stratified train/val/test splits (`config.yaml: stratify: true`)."}

            ### 3.2 Train / Val / Test Leakage
            Because multiple clips may originate from a single synthetic
            session (same seed family), naive random splits risk intra-session
            leakage.  The pipeline enforces `group_by_source: true` in
            `config.yaml` to group chunks of the same recording together.

            ### 3.3 Feature Distribution Shift Under Wind Noise
            The Week 1 notebook established that wind contamination at 5 dB SNR
            shifts the spectral centroid by tens of percent.  Models trained on
            clean synthetic features will degrade under deployment noise unless
            SNR-controlled augmentation (`WindNoiseGenerator`) is applied.

            ### 3.4 Overfitting Risk
            With {total_clips} clips across {n_classes} classes
            ({total_clips // max(n_classes, 1)} clips/class on average), a CNN
            classifier with {sum(feat.get("mel_spectrogram", [0])):,}+ input
            elements will overfit without strong regularisation.
            Early stopping (`patience = 10` epochs on `val_f1_macro`) and
            data augmentation (`variants_per_clip = 3`) are mandatory.

            ---

            ## 4. Recommendations

            ### 4.1 Immediate Actions
            1. **Run augmentation pipeline**: expand to
               `{total_clips * 4}` clips (3 wind-noise variants + original)
               before training.  Execute:
               ```
               python -m src.preprocessing.pipeline \\
                   --input data/raw/synthetic --output data/processed
               ```
            2. **Validate splits**: confirm no recording appears in both
               train and test partitions via `validate_dataset()`.
            3. **Inspect clipped clips**: review the {clipped_clips} flagged
               clips (> 1 % of samples near full scale) — they may contain
               ADC saturation artefacts from the synthetic generator.

            ### 4.2 Data Collection (Field Recording)
            4. **Collect real turbine recordings** at 3+ wind speeds
               (cut-in, rated, cut-out) to close the synthetic-to-real gap.
               Target ≥ 60 min per fault class at each wind speed.
            5. **Acquire compound fault recordings**: simultaneously
               introduce bearing wear and blade imbalance on a test rig.
            6. **Vary rotor speed**: sweep 5–20 RPM to capture
               speed-dependent gear-mesh frequency shifts.

            ### 4.3 Model Training Configuration
            7. **Loss function**: `CrossEntropyLoss` with class weights
               `w_c = total / (n_classes * count_c)` if imbalance ratio > 1.5.
            8. **Metric**: optimise `val_f1_macro` (not accuracy) to avoid
               majority-class collapse.
            9. **Feature normalisation**: apply per-feature-dimension
               standardisation fitted on the training split only; do **not**
               normalise on the full dataset to prevent val/test leakage.
            10. **Denoising ablation**: train two model variants
                (with / without spectral subtraction pre-processing) and
                report Δ F1-macro as a quantitative denoising benefit estimate.

            ### 4.4 Evaluation Protocol
            11. **SNR robustness curves**: evaluate F1-macro vs wind-noise
                SNR at {-5}, 0, 5, 10, 15, 20 dB using the
                `WindNoiseGenerator`; report in the paper's Table II.
            12. **MTTD target**: verify mean time-to-detection ≤ 60 s
                (config.yaml: `monitoring.mttd_target_seconds`) on the held-out
                test set before claiming deployment readiness.

            ---

            ## 5. Appendix: Full Feature Dimension Reference

            | Feature Set | Shape | Memory (FP32/clip) |
            |-------------|-------|:------------------:|
            {feat_table}

            *Memory estimates per clip at float32 precision.*

            ---

            *Report generated by `scripts/generate_dataset_statistics.py`.*
            """)

        path = self.config.output_dir / "research_dataset_report.md"
        try:
            path.write_text(md, encoding="utf-8")
            logger.info("Research report -> %s", path)
        except OSError as exc:
            logger.error("Failed to write research report: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_directories(self) -> None:
        """Create output directories if they do not exist."""
        for directory in (self.config.output_dir, self.config.figures_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("Cannot create directory %s: %s", directory, exc)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for NumPy scalars and Path objects.

    Args:
        obj: Object that the default JSON encoder cannot handle.

    Returns:
        A JSON-serialisable equivalent.

    Raises:
        TypeError: For types with no known conversion.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serialisable")


def _numel(shape: list[int]) -> int:
    """Compute the total number of elements from a shape list.

    Args:
        shape: List of dimension sizes.

    Returns:
        Product of all dimensions, or 0 for an empty shape.
    """
    if not shape:
        return 0
    result = 1
    for dim in shape:
        result *= dim
    return result


def _human_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n_bytes: Raw byte count.

    Returns:
        String such as '12.3 KB' or '1.4 MB'.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate comprehensive dataset statistics for the Wind Turbine "
            "Acoustic Monitoring project."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "synthetic",
        help="Root directory of raw audio clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "reports",
        help="Output directory for JSON / CSV / Markdown reports.",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=_PROJECT_ROOT / "docs" / "figures",
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--loader",
        choices=sorted(LOADER_REGISTRY),
        default="synthetic",
        help="Dataset loader to use.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate to resample all clips to (Hz).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Target clip duration in seconds.",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=IMBALANCE_THRESHOLD,
        help=(
            "Relative class-size deviation (fraction) that triggers an "
            "imbalance warning."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: parse arguments, run the generator, and exit.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = StatisticsConfig(
        input_dir=args.input,
        output_dir=args.output,
        figures_dir=args.figures,
        loader_name=args.loader,
        audio_sample_rate=args.sample_rate,
        audio_duration=args.duration,
        imbalance_threshold=args.imbalance_threshold,
    )

    try:
        generator = DatasetStatisticsGenerator(config)
        report = generator.run()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 – top-level guard only
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return 1

    # Print a one-line summary to stdout for CI integration
    inv = report.get("clip_inventory", {})
    bal = report.get("class_balance", {})
    print(
        f"\nDataset statistics complete: "
        f"{inv.get('total_clips', 0)} clips | "
        f"{inv.get('total_duration_seconds', 0) / 60:.1f} min | "
        f"{len(inv.get('classes_found', []))} classes | "
        f"balanced={bal.get('is_balanced', '?')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())