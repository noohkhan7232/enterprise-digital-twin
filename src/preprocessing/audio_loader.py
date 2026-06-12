#!/usr/bin/env python3
"""Audio loading foundation for the Wind Turbine Acoustic Monitoring pipeline.

This module provides the canonical entry point for getting audio into the
preprocessing pipeline. Every downstream stage (denoising, feature
extraction, augmentation) consumes :class:`AudioClip` objects produced here.

Components:

* :class:`AudioConfig` -- global loading configuration (sample rate, clip
  duration, channel handling, normalization).
* :class:`AudioClip` -- the standard container passed through the pipeline.
* :class:`AudioLoader` -- generic WAV/FLAC loader with directory traversal
  and long-recording chunking.
* :class:`MIMIILoader` -- loader aware of the MIMII directory layout.
* :class:`CWRULoader` -- loader for CWRU Bearing Dataset ``.mat`` files.

Usage::

    python -m src.preprocessing.audio_loader --path data/raw/synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import librosa
import numpy as np
import soundfile as sf
from scipy.io import loadmat

logger = logging.getLogger("audio_loader")

#: Audio file extensions handled by :class:`AudioLoader`.
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".flac", ".ogg", ".mp3"})

#: Canonical fault-type -> integer label mapping used across the project.
FAULT_LABELS: Final[dict[str, int]] = {
    "normal": 0,
    "blade_imbalance": 1,
    "bearing_fault": 2,
    "gearbox_fault": 3,
    "electrical_fault": 4,
    "abnormal": 5,  # generic anomaly (e.g., MIMII 'abnormal' recordings)
    "unknown": -1,
}

#: Native sample rate of the CWRU drive-end recordings (12 kHz subset).
CWRU_NATIVE_SAMPLE_RATE: Final[int] = 12_000


# ---------------------------------------------------------------------------
# Configuration and data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """Global audio loading configuration.

    Attributes:
        sample_rate: Target sample rate in Hz; all audio is resampled to this.
        duration: Target clip duration in seconds (clips are padded/truncated).
        mono: If True, multi-channel audio is downmixed to mono.
        normalize: If True, waveforms are peak-normalized after loading.
    """

    sample_rate: int = 22050
    duration: float = 10.0
    mono: bool = True
    normalize: bool = True

    @property
    def num_samples(self) -> int:
        """Number of samples in one clip at the target sample rate."""
        return int(self.sample_rate * self.duration)


@dataclass
class AudioClip:
    """Standard audio container passed through the preprocessing pipeline.

    Attributes:
        waveform: 1-D float32 array of audio samples.
        sample_rate: Sample rate of ``waveform`` in Hz.
        path: Source file the clip was loaded from.
        label: Integer class label (see :data:`FAULT_LABELS`); -1 if unknown.
        fault_type: Human-readable fault class name.
        dataset: Name of the originating dataset (e.g., 'synthetic', 'mimii').
        clip_index: Index of this clip within its source recording (0 for
            single-clip files; increments for chunked long recordings).
        duration: Clip duration in seconds.
        metadata: Free-form extra information (machine id, channel, SNR, ...).
    """

    waveform: np.ndarray
    sample_rate: int
    path: Path
    label: int = FAULT_LABELS["unknown"]
    fault_type: str = "unknown"
    dataset: str = "unknown"
    clip_index: int = 0
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive duration from the waveform when not provided."""
        if self.duration <= 0.0 and self.sample_rate > 0:
            self.duration = len(self.waveform) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of samples in the waveform."""
        return int(len(self.waveform))


# ---------------------------------------------------------------------------
# Generic audio loader
# ---------------------------------------------------------------------------


class AudioLoader:
    """Load audio files into :class:`AudioClip` objects.

    Handles resampling to the configured rate, mono downmixing, peak
    normalization, fixed-length padding/truncation, and chunking of long
    recordings.

    Args:
        config: Audio loading configuration. Defaults to :class:`AudioConfig`.
        dataset_name: Dataset tag stamped onto produced clips.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        dataset_name: str = "generic",
    ) -> None:
        self.config = config or AudioConfig()
        self.dataset_name = dataset_name

    # -- public API -----------------------------------------------------------

    def load(self, filepath: Path | str) -> AudioClip | None:
        """Load a single audio file as one fixed-length clip.

        The waveform is resampled to ``config.sample_rate``, optionally
        downmixed and normalized, then padded or truncated to
        ``config.duration`` seconds.

        Args:
            filepath: Path to the audio file.

        Returns:
            An :class:`AudioClip`, or None if the file could not be loaded.
        """
        filepath = Path(filepath)
        waveform = self._read_audio(filepath)
        if waveform is None:
            return None

        waveform = self._fit_length(waveform, self.config.num_samples)
        fault_type = self._infer_fault_type(filepath)
        return AudioClip(
            waveform=waveform,
            sample_rate=self.config.sample_rate,
            path=filepath,
            label=FAULT_LABELS.get(fault_type, FAULT_LABELS["unknown"]),
            fault_type=fault_type,
            dataset=self.dataset_name,
            clip_index=0,
            duration=self.config.duration,
        )

    def load_directory(self, directory: Path | str) -> list[AudioClip]:
        """Recursively load every supported audio file under a directory.

        Files that fail to load are skipped with a logged warning so a single
        corrupt file cannot abort a large ingestion run.

        Args:
            directory: Root directory to traverse.

        Returns:
            List of successfully loaded clips (possibly empty).
        """
        directory = Path(directory)
        if not directory.is_dir():
            logger.error("Directory not found: %s", directory)
            return []

        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info("Loading %d audio files from %s", len(files), directory)

        clips: list[AudioClip] = []
        for filepath in files:
            clip = self.load(filepath)
            if clip is not None:
                clips.append(clip)

        logger.info("Loaded %d/%d clips from %s", len(clips), len(files), directory)
        return clips

    def load_long_recording(
        self,
        filepath: Path | str,
        chunk_duration: float | None = None,
    ) -> list[AudioClip]:
        """Load a long recording and split it into fixed-length chunks.

        Trailing audio shorter than half a chunk is discarded; otherwise the
        final chunk is zero-padded to full length.

        Args:
            filepath: Path to the long audio file.
            chunk_duration: Chunk length in seconds. Defaults to
                ``config.duration``.

        Returns:
            List of chunk clips with increasing ``clip_index`` (possibly empty).
        """
        filepath = Path(filepath)
        chunk_duration = chunk_duration or self.config.duration
        waveform = self._read_audio(filepath)
        if waveform is None:
            return []

        chunk_samples = int(self.config.sample_rate * chunk_duration)
        if chunk_samples <= 0:
            logger.error("Invalid chunk duration: %s s", chunk_duration)
            return []

        fault_type = self._infer_fault_type(filepath)
        clips: list[AudioClip] = []
        for clip_index, start in enumerate(range(0, len(waveform), chunk_samples)):
            chunk = waveform[start : start + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                break  # discard short trailing remainder
            chunk = self._fit_length(chunk, chunk_samples)
            clips.append(
                AudioClip(
                    waveform=chunk,
                    sample_rate=self.config.sample_rate,
                    path=filepath,
                    label=FAULT_LABELS.get(fault_type, FAULT_LABELS["unknown"]),
                    fault_type=fault_type,
                    dataset=self.dataset_name,
                    clip_index=clip_index,
                    duration=chunk_duration,
                    metadata={"chunk_start_seconds": start / self.config.sample_rate},
                )
            )

        logger.info("Chunked %s into %d clips of %.1f s",
                    filepath.name, len(clips), chunk_duration)
        return clips

    def resample(self, audio: np.ndarray, orig_sample_rate: int) -> np.ndarray:
        """Resample audio to the configured target sample rate.

        Args:
            audio: 1-D waveform array.
            orig_sample_rate: Sample rate of ``audio`` in Hz.

        Returns:
            Waveform resampled to ``config.sample_rate`` (unchanged if rates
            already match).
        """
        if orig_sample_rate == self.config.sample_rate:
            return audio.astype(np.float32)
        resampled = librosa.resample(
            audio.astype(np.float32),
            orig_sr=orig_sample_rate,
            target_sr=self.config.sample_rate,
        )
        logger.debug("Resampled %d Hz -> %d Hz (%d -> %d samples)",
                     orig_sample_rate, self.config.sample_rate,
                     len(audio), len(resampled))
        return resampled

    @staticmethod
    def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """Peak-normalize a waveform.

        Args:
            audio: 1-D waveform array.
            peak: Target absolute peak amplitude.

        Returns:
            Normalized waveform; silent input is returned unchanged.
        """
        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_abs < 1e-12:
            return audio.astype(np.float32)
        return (audio / max_abs * peak).astype(np.float32)

    # -- internals --------------------------------------------------------------

    def _read_audio(self, filepath: Path) -> np.ndarray | None:
        """Read, resample, downmix, and normalize an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            Processed 1-D float32 waveform, or None on failure.
        """
        if not filepath.is_file():
            logger.error("File not found: %s", filepath)
            return None
        try:
            waveform, orig_sr = librosa.load(
                filepath, sr=None, mono=self.config.mono
            )
        except (sf.LibsndfileError, RuntimeError, EOFError, OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", filepath, exc)
            return None

        if waveform.ndim > 1:  # safety net when mono=False upstream
            waveform = np.mean(waveform, axis=0)

        waveform = self.resample(waveform, int(orig_sr))
        if self.config.normalize:
            waveform = self.normalize(waveform)
        return waveform

    @staticmethod
    def _fit_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Pad with zeros or truncate a waveform to an exact sample count."""
        if len(audio) >= target_samples:
            return audio[:target_samples].astype(np.float32)
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded

    @staticmethod
    def _infer_fault_type(filepath: Path) -> str:
        """Infer fault type from a file path using known class keywords.

        Checks both parent directory names and the file stem against the
        canonical fault classes (longest match first, so 'bearing_fault'
        wins over a bare 'fault' substring).

        Args:
            filepath: Path to inspect.

        Returns:
            Matched fault class name, or 'unknown'.
        """
        haystack = "/".join(part.lower() for part in filepath.parts)
        for fault_type in sorted(FAULT_LABELS, key=len, reverse=True):
            if fault_type != "unknown" and fault_type in haystack:
                return fault_type
        return "unknown"


# ---------------------------------------------------------------------------
# MIMII loader
# ---------------------------------------------------------------------------


class MIMIILoader(AudioLoader):
    """Loader for the MIMII industrial machine sound dataset.

    Expects the standard MIMII layout::

        <root>/<machine_type>/<machine_id>/<normal|abnormal>/*.wav

    e.g. ``data/raw/mimii/fan/id_00/normal/00000000.wav``. The condition
    folder maps to fault type ('normal' or 'abnormal') and machine
    type/id are recorded in clip metadata.

    Args:
        config: Audio loading configuration.
    """

    CONDITIONS: Final[frozenset[str]] = frozenset({"normal", "abnormal"})

    def __init__(self, config: AudioConfig | None = None) -> None:
        super().__init__(config=config, dataset_name="mimii")

    def load_directory(self, directory: Path | str) -> list[AudioClip]:
        """Load all MIMII recordings under ``directory``.

        Args:
            directory: MIMII dataset root (or any sub-tree of it).

        Returns:
            Clips annotated with machine type, machine id, and condition.
        """
        directory = Path(directory)
        clips = super().load_directory(directory)
        for clip in clips:
            self._annotate(clip)
        labelled = sum(1 for c in clips if c.fault_type in self.CONDITIONS)
        logger.info("MIMII: annotated %d/%d clips with machine metadata",
                    labelled, len(clips))
        return clips

    def _annotate(self, clip: AudioClip) -> None:
        """Attach MIMII machine metadata parsed from the clip's path."""
        parts = [part.lower() for part in clip.path.parts]
        for index, part in enumerate(parts):
            if part in self.CONDITIONS and index >= 2:
                clip.fault_type = part
                clip.label = FAULT_LABELS.get(part, FAULT_LABELS["unknown"])
                clip.metadata.update(
                    machine_type=parts[index - 2],
                    machine_id=parts[index - 1],
                    condition=part,
                )
                return
        logger.debug("Non-standard MIMII path, metadata skipped: %s", clip.path)


# ---------------------------------------------------------------------------
# CWRU loader
# ---------------------------------------------------------------------------


class CWRULoader(AudioLoader):
    """Loader for CWRU Bearing Dataset MATLAB ``.mat`` files.

    Extracts the drive-end (``*_DE_time``) and, when available, fan-end
    (``*_FE_time``) accelerometer channels, resamples them from the native
    CWRU rate to the configured sample rate, and chunks them into
    fixed-length clips.

    Args:
        config: Audio loading configuration.
        native_sample_rate: Sample rate of the source ``.mat`` recordings.
            Defaults to the 12 kHz CWRU subset.
    """

    #: Filename keyword -> project fault class mapping for CWRU files.
    FILENAME_FAULT_MAP: Final[dict[str, str]] = {
        "normal": "normal",
        "inner_race": "bearing_fault",
        "outer_race": "bearing_fault",
        "ball": "bearing_fault",
        "ir": "bearing_fault",
        "or": "bearing_fault",
        "b0": "bearing_fault",
    }

    def __init__(
        self,
        config: AudioConfig | None = None,
        native_sample_rate: int = CWRU_NATIVE_SAMPLE_RATE,
    ) -> None:
        super().__init__(config=config, dataset_name="cwru")
        self.native_sample_rate = native_sample_rate

    def load(self, filepath: Path | str) -> AudioClip | None:
        """Load the first clip of the drive-end channel of a ``.mat`` file.

        Args:
            filepath: Path to the CWRU ``.mat`` file.

        Returns:
            The first drive-end clip, or None if the file yields no clips.
        """
        clips = self.load_mat(filepath)
        return clips[0] if clips else None

    def load_directory(self, directory: Path | str) -> list[AudioClip]:
        """Load every ``.mat`` file under ``directory``.

        Args:
            directory: Directory containing CWRU ``.mat`` files.

        Returns:
            All chunked clips from every channel of every file.
        """
        directory = Path(directory)
        if not directory.is_dir():
            logger.error("Directory not found: %s", directory)
            return []

        mat_files = sorted(directory.rglob("*.mat"))
        logger.info("Loading %d CWRU .mat files from %s", len(mat_files), directory)
        clips: list[AudioClip] = []
        for mat_file in mat_files:
            clips.extend(self.load_mat(mat_file))
        logger.info("CWRU: produced %d clips from %d files", len(clips), len(mat_files))
        return clips

    def load_mat(self, filepath: Path | str) -> list[AudioClip]:
        """Extract DE/FE channels from one ``.mat`` file as chunked clips.

        Args:
            filepath: Path to the CWRU ``.mat`` file.

        Returns:
            Clips for each available channel, chunked to ``config.duration``
            (possibly empty on read failure).
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            logger.error("File not found: %s", filepath)
            return []
        try:
            mat_data: dict[str, Any] = loadmat(str(filepath))
        except (OSError, ValueError, NotImplementedError) as exc:
            logger.warning("Failed to parse %s: %s", filepath, exc)
            return []

        fault_type = self._fault_type_from_filename(filepath)
        clips: list[AudioClip] = []
        for channel in ("DE_time", "FE_time"):
            signal = self._extract_channel(mat_data, channel)
            if signal is None:
                if channel == "DE_time":
                    logger.warning("No DE_time channel found in %s", filepath.name)
                continue
            clips.extend(self._chunk_signal(signal, filepath, fault_type, channel))

        logger.info("%s: %d clips (%s)", filepath.name, len(clips), fault_type)
        return clips

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _extract_channel(mat_data: dict[str, Any], suffix: str) -> np.ndarray | None:
        """Find a variable ending with ``suffix`` (e.g. 'X097_DE_time').

        Args:
            mat_data: Parsed ``.mat`` contents.
            suffix: Channel suffix to match.

        Returns:
            Flattened float32 signal, or None when absent.
        """
        for key, value in mat_data.items():
            if key.endswith(suffix) and isinstance(value, np.ndarray):
                return np.asarray(value, dtype=np.float32).flatten()
        return None

    def _chunk_signal(
        self,
        signal: np.ndarray,
        filepath: Path,
        fault_type: str,
        channel: str,
    ) -> list[AudioClip]:
        """Resample a raw channel and split it into fixed-length clips."""
        resampled = self.resample(signal, self.native_sample_rate)
        if self.config.normalize:
            resampled = self.normalize(resampled)

        chunk_samples = self.config.num_samples
        clips: list[AudioClip] = []
        for clip_index, start in enumerate(range(0, len(resampled), chunk_samples)):
            chunk = resampled[start : start + chunk_samples]
            if len(chunk) < chunk_samples // 2:
                break
            chunk = self._fit_length(chunk, chunk_samples)
            clips.append(
                AudioClip(
                    waveform=chunk,
                    sample_rate=self.config.sample_rate,
                    path=filepath,
                    label=FAULT_LABELS.get(fault_type, FAULT_LABELS["unknown"]),
                    fault_type=fault_type,
                    dataset=self.dataset_name,
                    clip_index=clip_index,
                    duration=self.config.duration,
                    metadata={
                        "channel": channel,
                        "native_sample_rate": self.native_sample_rate,
                    },
                )
            )
        return clips

    def _fault_type_from_filename(self, filepath: Path) -> str:
        """Map a CWRU filename to a project fault class.

        Args:
            filepath: CWRU ``.mat`` file path.

        Returns:
            Matched fault class, or 'unknown'.
        """
        stem = filepath.stem.lower()
        for keyword, fault_type in self.FILENAME_FAULT_MAP.items():
            if keyword in stem:
                return fault_type
        return "unknown"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Load audio datasets into AudioClip objects (smoke test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", type=Path, default=Path("data/raw/synthetic"),
                        help="Directory or file to load.")
    parser.add_argument("--loader", choices=("generic", "mimii", "cwru"),
                        default="generic", help="Which loader to use.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: load a path and report a summary.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when nothing could be loaded.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    loaders: dict[str, AudioLoader] = {
        "generic": AudioLoader(dataset_name="synthetic"),
        "mimii": MIMIILoader(),
        "cwru": CWRULoader(),
    }
    loader = loaders[args.loader]

    if args.path.is_file():
        clip = loader.load(args.path)
        clips = [clip] if clip is not None else []
    else:
        clips = loader.load_directory(args.path)

    if not clips:
        logger.error("No clips loaded from %s", args.path)
        return 1

    by_class: dict[str, int] = {}
    for clip in clips:
        by_class[clip.fault_type] = by_class.get(clip.fault_type, 0) + 1

    logger.info("Loaded %d clips total:", len(clips))
    for fault_type, count in sorted(by_class.items()):
        logger.info("  %-18s %d clips", fault_type, count)
    sample = clips[0]
    logger.info("Example clip: %s | %d Hz | %.1f s | label=%d | dataset=%s",
                sample.path.name, sample.sample_rate, sample.duration,
                sample.label, sample.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
