#!/usr/bin/env python3
"""Master preprocessing pipeline for the Wind Turbine Acoustic Monitoring project.

Workflow::

    Load Audio -> Denoise -> Extract Features -> Augment -> Save Features
               -> Generate Dataset Statistics

Why preprocessing quality determines downstream performance
------------------------------------------------------------
Every model in this project -- the anomaly-detection autoencoder, the fault
classifier, and the RUL regressor -- learns exclusively from the feature
tensors produced here. Preprocessing is therefore the ceiling on achievable
performance: residual wind noise becomes label noise the classifier must
absorb; inconsistent normalization or clip lengths become distribution shift
between training and deployment; silent data corruption becomes invisible
evaluation bias. A model can never recover information that denoising
destroyed or that feature extraction failed to encode. This pipeline
therefore validates inputs, applies controlled denoising, extracts all
feature representations consistently, augments reproducibly (seeded), and
emits a ``dataset_report.json`` so every experiment in the paper is traceable
to an exact, auditable data state.

Supported datasets: synthetic, MIMII, CWRU, and (future) WindTurbineSound --
selected via the ``loader_name`` configuration field.

Usage::

    python -m src.preprocessing.pipeline --input data/raw/synthetic --output data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.preprocessing.audio_loader import (
    FAULT_LABELS,
    AudioClip,
    AudioConfig,
    AudioLoader,
    CWRULoader,
    MIMIILoader,
)
from src.preprocessing.augmentation import AugmentationConfig, AugmentationPipeline
from src.preprocessing.denoiser import Denoiser, DenoiserConfig
from src.preprocessing.feature_extractor import FeatureConfig, FeatureExtractor

logger = logging.getLogger("pipeline")

#: Loader registry: maps ``loader_name`` to an AudioLoader factory.
LOADER_REGISTRY: Final[dict[str, type[AudioLoader]]] = {
    "generic": AudioLoader,
    "synthetic": AudioLoader,
    "mimii": MIMIILoader,
    "cwru": CWRULoader,
    "windturbine": AudioLoader,  # future dataset; generic layout for now
}

#: Progress is logged every this many processed clips.
PROGRESS_LOG_INTERVAL: Final[int] = 25


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end preprocessing pipeline.

    Attributes:
        input_dir: Directory containing the raw dataset.
        output_dir: Root output directory for features and reports.
        save_features: If True, feature arrays are written to disk.
        save_augmented_audio: If True, augmented waveforms are also saved as
            ``.npy`` files alongside their features.
        enable_denoising: If True, clips are denoised before feature extraction.
        enable_augmentation: If True, augmented variants are generated.
        n_augmentations_per_clip: Augmented variants per source clip.
        max_workers: Worker threads for :meth:`parallel_process_dataset`.
        random_seed: Base seed for reproducible augmentation.
        loader_name: Dataset loader to use (see :data:`LOADER_REGISTRY`).
        denoise_method: Denoising algorithm passed to :meth:`Denoiser.denoise`.
    """

    input_dir: Path = field(default_factory=lambda: Path("data/raw/synthetic"))
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))
    save_features: bool = True
    save_augmented_audio: bool = False
    enable_denoising: bool = True
    enable_augmentation: bool = True
    n_augmentations_per_clip: int = 3
    max_workers: int = 4
    random_seed: int = 42
    loader_name: str = "generic"
    denoise_method: str = "spectral_subtraction"

    @property
    def features_dir(self) -> Path:
        """Directory where per-clip feature folders are written."""
        return self.output_dir / "features"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """End-to-end dataset preprocessing orchestrator.

    Wires together :class:`AudioLoader`, :class:`Denoiser`,
    :class:`FeatureExtractor`, and :class:`AugmentationPipeline` and runs
    them over complete datasets with progress tracking, per-clip error
    recovery, and reproducible augmentation.

    Args:
        config: Pipeline configuration. Defaults to :class:`PipelineConfig`.
        audio_config: Audio loading configuration shared by all components.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        audio_config: AudioConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.audio_config = audio_config or AudioConfig()

        loader_class = LOADER_REGISTRY.get(self.config.loader_name)
        if loader_class is None:
            logger.warning("Unknown loader '%s'; falling back to generic",
                           self.config.loader_name)
            loader_class = AudioLoader
        if loader_class is AudioLoader:
            self.loader: AudioLoader = AudioLoader(
                self.audio_config, dataset_name=self.config.loader_name
            )
        else:
            self.loader = loader_class(self.audio_config)

        self.denoiser = Denoiser(
            DenoiserConfig(sample_rate=self.audio_config.sample_rate)
        )
        self.feature_extractor = FeatureExtractor(
            FeatureConfig(sample_rate=self.audio_config.sample_rate)
        )
        self.augmenter = AugmentationPipeline(
            AugmentationConfig(
                sample_rate=self.audio_config.sample_rate,
                variants_per_clip=self.config.n_augmentations_per_clip,
                random_seed=self.config.random_seed,
            )
        )

    # -- single clip ------------------------------------------------------------

    def process_clip(
        self,
        clip: AudioClip,
        denoise: bool | None = None,
    ) -> dict[str, Any] | None:
        """Denoise one clip and extract every feature representation.

        Args:
            clip: Input audio clip.
            denoise: Override for ``config.enable_denoising`` (used to skip
                re-denoising of augmented variants).

        Returns:
            Dictionary with keys ``mel``, ``mfcc``, ``cqt``,
            ``spectral_features`` (float32 arrays) and ``metadata`` (clip
            provenance and labels), or None when processing fails.
        """
        denoise = self.config.enable_denoising if denoise is None else denoise
        try:
            waveform = clip.waveform
            if denoise:
                waveform = self.denoiser.denoise(
                    waveform, method=self.config.denoise_method
                )
            return {
                "mel": self.feature_extractor.mel_spectrogram(waveform),
                "mfcc": self.feature_extractor.mfcc(waveform),
                "cqt": self.feature_extractor.cqt_spectrogram(waveform),
                "spectral_features": self.feature_extractor.spectral_statistics(waveform),
                "metadata": {
                    "source_path": str(clip.path),
                    "clip_index": clip.clip_index,
                    "label": clip.label,
                    "fault_type": clip.fault_type,
                    "dataset": clip.dataset,
                    "duration": clip.duration,
                    "sample_rate": clip.sample_rate,
                    "denoised": denoise,
                    "augmented": bool(clip.metadata.get("augmented", False)),
                    "variant_index": clip.metadata.get("variant_index"),
                },
            }
        except (ValueError, RuntimeError, MemoryError) as exc:
            logger.error("Failed to process %s (clip %d): %s",
                         clip.path, clip.clip_index, exc)
            return None

    def _expand_clip(self, clip: AudioClip) -> list[dict[str, Any]]:
        """Process a source clip plus its augmented variants.

        Args:
            clip: Source clip.

        Returns:
            Feature records for the original and (if enabled) each variant;
            empty when the original itself fails.
        """
        records: list[dict[str, Any]] = []
        original = self.process_clip(clip)
        if original is None:
            return records
        records.append(original)

        if self.config.enable_augmentation:
            try:
                variants = self.augmenter.augment(clip)
            except (ValueError, RuntimeError) as exc:
                logger.error("Augmentation failed for %s: %s", clip.path, exc)
                variants = []
            for variant in variants:
                record = self.process_clip(variant, denoise=False)
                if record is not None:
                    if self.config.save_augmented_audio:
                        record["waveform"] = variant.waveform
                    records.append(record)
        return records

    # -- dataset processing --------------------------------------------------------

    def process_dataset(self, directory: Path | str | None = None) -> dict[str, Any]:
        """Process every clip in a dataset sequentially.

        Args:
            directory: Dataset directory. Defaults to ``config.input_dir``.

        Returns:
            Summary dictionary with keys ``records`` (feature dictionaries),
            ``processed``, ``failed``, ``augmented``, and ``elapsed_seconds``.
        """
        directory = Path(directory or self.config.input_dir)
        start = time.perf_counter()
        clips = self.loader.load_directory(directory)
        if not clips:
            logger.error("No clips loaded from %s", directory)
            return {"records": [], "processed": 0, "failed": 0,
                    "augmented": 0, "elapsed_seconds": 0.0}

        records: list[dict[str, Any]] = []
        failed = 0
        for index, clip in enumerate(clips, start=1):
            expanded = self._expand_clip(clip)
            if not expanded:
                failed += 1
            records.extend(expanded)
            if index % PROGRESS_LOG_INTERVAL == 0 or index == len(clips):
                rate = index / max(time.perf_counter() - start, 1e-9)
                logger.info("Progress: %d/%d source clips (%.1f clips/s)",
                            index, len(clips), rate)

        elapsed = time.perf_counter() - start
        augmented = sum(1 for r in records if r["metadata"]["augmented"])
        logger.info(
            "Dataset processed: %d records (%d augmented), %d failures, %.1f s total",
            len(records), augmented, failed, elapsed,
        )
        return {"records": records, "processed": len(records),
                "failed": failed, "augmented": augmented,
                "elapsed_seconds": elapsed}

    def parallel_process_dataset(
        self,
        directory: Path | str | None = None,
    ) -> dict[str, Any]:
        """Process a dataset with a thread pool.

        Uses :class:`concurrent.futures.ThreadPoolExecutor` with
        ``config.max_workers`` workers. Threading is effective here because
        librosa/NumPy release the GIL inside FFT-heavy native code.

        Args:
            directory: Dataset directory. Defaults to ``config.input_dir``.

        Returns:
            Summary dictionary identical in shape to :meth:`process_dataset`.
        """
        directory = Path(directory or self.config.input_dir)
        start = time.perf_counter()
        clips = self.loader.load_directory(directory)
        if not clips:
            logger.error("No clips loaded from %s", directory)
            return {"records": [], "processed": 0, "failed": 0,
                    "augmented": 0, "elapsed_seconds": 0.0}

        records: list[dict[str, Any]] = []
        failed = 0
        completed = 0
        logger.info("Parallel processing %d clips with %d workers",
                    len(clips), self.config.max_workers)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self._expand_clip, clip): clip for clip in clips}
            for future in as_completed(futures):
                clip = futures[future]
                try:
                    expanded = future.result()
                except Exception as exc:  # noqa: BLE001 - worker isolation boundary
                    logger.error("Worker failed on %s: %s", clip.path, exc)
                    expanded = []
                if not expanded:
                    failed += 1
                records.extend(expanded)
                completed += 1
                if completed % PROGRESS_LOG_INTERVAL == 0 or completed == len(clips):
                    rate = completed / max(time.perf_counter() - start, 1e-9)
                    logger.info("Progress: %d/%d source clips (%.1f clips/s)",
                                completed, len(clips), rate)

        elapsed = time.perf_counter() - start
        augmented = sum(1 for r in records if r["metadata"]["augmented"])
        logger.info(
            "Parallel processing done: %d records (%d augmented), %d failures, %.1f s",
            len(records), augmented, failed, elapsed,
        )
        return {"records": records, "processed": len(records),
                "failed": failed, "augmented": augmented,
                "elapsed_seconds": elapsed}

    # -- persistence -----------------------------------------------------------------

    def process_and_save(
        self,
        directory: Path | str | None = None,
        parallel: bool = False,
    ) -> dict[str, Any]:
        """Process a dataset and persist features to ``data/processed/features/``.

        Layout::

            features/<fault_type>/<clip_id>/
                mel.npy
                mfcc.npy
                cqt.npy
                spectral_features.npy
                metadata.json

        Args:
            directory: Dataset directory. Defaults to ``config.input_dir``.
            parallel: If True, uses :meth:`parallel_process_dataset`.

        Returns:
            Dataset statistics dictionary (see
            :meth:`generate_dataset_statistics`).
        """
        runner = self.parallel_process_dataset if parallel else self.process_dataset
        summary = runner(directory)
        records: list[dict[str, Any]] = summary["records"]

        saved = 0
        if self.config.save_features:
            for record in records:
                if self._save_record(record):
                    saved += 1
            logger.info("Saved %d/%d feature records to %s",
                        saved, len(records), self.config.features_dir)

        statistics = self.generate_dataset_statistics(summary)
        return statistics

    def _save_record(self, record: dict[str, Any]) -> bool:
        """Write one feature record to disk.

        Args:
            record: Feature record from :meth:`process_clip`.

        Returns:
            True on success, False on failure (logged, never raised).
        """
        meta = record["metadata"]
        stem = Path(meta["source_path"]).stem
        variant = meta.get("variant_index")
        clip_id = f"{stem}_c{meta['clip_index']:03d}"
        if meta["augmented"] and variant is not None:
            clip_id += f"_aug{variant:02d}"

        target = self.config.features_dir / meta["fault_type"] / clip_id
        try:
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "mel.npy", record["mel"])
            np.save(target / "mfcc.npy", record["mfcc"])
            np.save(target / "cqt.npy", record["cqt"])
            np.save(target / "spectral_features.npy", record["spectral_features"])
            if "waveform" in record:
                np.save(target / "waveform.npy", record["waveform"])
            with (target / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
            return True
        except (OSError, ValueError) as exc:
            logger.error("Failed to save %s: %s", target, exc)
            return False

    # -- reporting --------------------------------------------------------------------

    def generate_dataset_statistics(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Compute dataset statistics and write ``dataset_report.json``.

        The report captures class distribution, feature dimensions, total
        duration, and augmentation statistics for use in paper tables.

        Args:
            summary: Output of :meth:`process_dataset` or
                :meth:`parallel_process_dataset`.

        Returns:
            Statistics dictionary with keys ``total_clips``, ``classes``,
            ``clips_per_class``, ``duration_hours``, ``augmented_clips``,
            ``processing_time``, ``failed_clips``, and ``feature_dimensions``.
        """
        records: list[dict[str, Any]] = summary["records"]
        clips_per_class: dict[str, int] = {}
        total_duration = 0.0
        for record in records:
            meta = record["metadata"]
            clips_per_class[meta["fault_type"]] = (
                clips_per_class.get(meta["fault_type"], 0) + 1
            )
            total_duration += float(meta["duration"])

        feature_dimensions: dict[str, list[int]] = {}
        if records:
            sample = records[0]
            for key in ("mel", "mfcc", "cqt", "spectral_features"):
                feature_dimensions[key] = list(np.asarray(sample[key]).shape)

        statistics: dict[str, Any] = {
            "total_clips": summary["processed"],
            "classes": sorted(clips_per_class),
            "clips_per_class": clips_per_class,
            "duration_hours": round(total_duration / 3600.0, 4),
            "augmented_clips": summary["augmented"],
            "processing_time": round(summary["elapsed_seconds"], 2),
            "failed_clips": summary["failed"],
            "feature_dimensions": feature_dimensions,
            "config": {
                "loader": self.config.loader_name,
                "denoising_enabled": self.config.enable_denoising,
                "denoise_method": self.config.denoise_method,
                "augmentation_enabled": self.config.enable_augmentation,
                "n_augmentations_per_clip": self.config.n_augmentations_per_clip,
                "random_seed": self.config.random_seed,
            },
        }

        report_path = self.config.output_dir / "dataset_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(statistics, handle, indent=2)
            logger.info("Dataset report written to %s", report_path)
        except OSError as exc:
            logger.error("Failed to write dataset report: %s", exc)
        return statistics

    def validate_dataset(self, directory: Path | str | None = None) -> dict[str, Any]:
        """Validate a raw dataset and write ``validation_report.json``.

        Checks for missing files (empty class directories), corrupted files
        (unreadable audio), empty clips (silent waveforms), and invalid
        labels (fault types missing from :data:`FAULT_LABELS`).

        Args:
            directory: Dataset directory. Defaults to ``config.input_dir``.

        Returns:
            Validation report with per-issue file lists and an ``is_valid``
            flag.
        """
        directory = Path(directory or self.config.input_dir)
        report: dict[str, Any] = {
            "directory": str(directory),
            "missing_files": [],
            "corrupted_files": [],
            "empty_clips": [],
            "invalid_labels": [],
            "total_checked": 0,
        }

        if not directory.is_dir():
            report["missing_files"].append(str(directory))
            report["is_valid"] = False
            logger.error("Dataset directory does not exist: %s", directory)
            return report

        clips = self.loader.load_directory(directory)
        audio_files = [
            p for p in directory.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
        loaded_paths = {clip.path for clip in clips}
        report["total_checked"] = len(audio_files)

        for path in audio_files:
            if path.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"} and \
                    path not in loaded_paths:
                report["corrupted_files"].append(str(path))

        for clip in clips:
            if clip.num_samples == 0 or float(np.max(np.abs(clip.waveform))) < 1e-10:
                report["empty_clips"].append(f"{clip.path}#{clip.clip_index}")
            if clip.fault_type not in FAULT_LABELS:
                report["invalid_labels"].append(
                    f"{clip.path}: '{clip.fault_type}'"
                )

        issues = (len(report["missing_files"]) + len(report["corrupted_files"])
                  + len(report["empty_clips"]) + len(report["invalid_labels"]))
        report["is_valid"] = issues == 0
        logger.info("Validation: %d files checked, %d issues found",
                    report["total_checked"], issues)

        report_path = self.config.output_dir / "validation_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            logger.info("Validation report written to %s", report_path)
        except OSError as exc:
            logger.error("Failed to write validation report: %s", exc)
        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the full preprocessing pipeline over a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic"),
                        help="Raw dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("data/processed"),
                        help="Output directory for features and reports.")
    parser.add_argument("--loader", choices=sorted(LOADER_REGISTRY),
                        default="synthetic", help="Dataset loader to use.")
    parser.add_argument("--parallel", action="store_true",
                        help="Use the thread-pool executor.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker threads for parallel processing.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation.")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Disable denoising.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: run the full pipeline and print final statistics.

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

    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        enable_denoising=not args.no_denoise,
        enable_augmentation=not args.no_augment,
        max_workers=args.workers,
        loader_name=args.loader,
    )
    pipeline = PreprocessingPipeline(config)

    # 1. Validate the raw dataset.
    validation = pipeline.validate_dataset()
    if not validation.get("is_valid", False):
        logger.warning("Validation found issues; continuing with valid clips only")

    # 2-4. Run the full pipeline, save features, and generate the report.
    statistics = pipeline.process_and_save(parallel=args.parallel)
    if statistics["total_clips"] == 0:
        logger.error("Pipeline produced no clips")
        return 1

    # 5. Print final statistics.
    logger.info("Final dataset statistics:")
    logger.info("  total clips:      %d", statistics["total_clips"])
    logger.info("  classes:          %s", ", ".join(statistics["classes"]))
    for fault_type, count in sorted(statistics["clips_per_class"].items()):
        logger.info("    %-18s %d", fault_type, count)
    logger.info("  duration:         %.2f h", statistics["duration_hours"])
    logger.info("  augmented clips:  %d", statistics["augmented_clips"])
    logger.info("  failed clips:     %d", statistics["failed_clips"])
    logger.info("  processing time:  %.2f s", statistics["processing_time"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
