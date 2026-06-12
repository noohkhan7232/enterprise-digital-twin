#!/usr/bin/env python3
"""Acoustic feature extraction for the Wind Turbine Acoustic Monitoring pipeline.

Transforms preprocessed waveforms into the feature representations consumed
by the anomaly detection, fault classification, and RUL models:

* **Log-mel spectrograms** -- primary CNN input.
* **MFCCs** -- compact cepstral features for classical ML baselines.
* **CQT spectrograms** -- log-frequency representation suited to harmonic
  fault signatures (gear-mesh tones, sidebands).
* **Spectral statistics** -- interpretable scalar descriptors for trend
  analysis and RUL regression.
* **3-channel mel stacks** -- mel + delta + delta-delta, formatted for
  image-style CNN backbones.

All extractors return ``float32`` NumPy arrays.

Usage::

    python -m src.preprocessing.feature_extractor --path data/raw/synthetic/normal/normal_000.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import librosa
import numpy as np

logger = logging.getLogger("feature_extractor")

#: Names of the scalar descriptors produced by ``spectral_statistics``.
SPECTRAL_STATISTIC_NAMES: Final[tuple[str, ...]] = (
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "rms_mean",
    "rms_std",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for all feature extraction transforms.

    Attributes:
        sample_rate: Expected input sample rate in Hz.
        n_fft: FFT window size for STFT-based features.
        hop_length: Hop size in samples between analysis frames.
        n_mels: Number of mel filterbank bands.
        n_mfcc: Number of MFCC coefficients.
        fmin: Lowest analysis frequency in Hz (also CQT minimum frequency).
        fmax: Highest mel analysis frequency in Hz; None means ``sample_rate / 2``.
        cqt_bins: Total number of CQT frequency bins.
        cqt_bins_per_octave: CQT frequency resolution per octave.
        log_offset: Small constant added before log compression for stability.
    """

    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 40
    fmin: float = 20.0
    fmax: float | None = None
    cqt_bins: int = 84
    cqt_bins_per_octave: int = 12
    log_offset: float = 1e-10

    @property
    def effective_fmax(self) -> float:
        """Upper mel frequency bound, defaulting to the Nyquist frequency."""
        return self.fmax if self.fmax is not None else self.sample_rate / 2.0


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Compute acoustic features from preprocessed waveforms.

    All methods accept a 1-D ``float32`` waveform at ``config.sample_rate``
    and return ``float32`` arrays. Inputs are validated once via
    :meth:`_validate`, so corrupt clips fail fast with a clear error.

    Args:
        config: Feature extraction configuration. Defaults to
            :class:`FeatureConfig`.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    # -- spectrogram features ---------------------------------------------------

    def mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute a log-scaled mel spectrogram.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            Log-mel spectrogram of shape ``(n_mels, num_frames)``, float32, in dB.
        """
        audio = self._validate(audio)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.effective_fmax,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel + self.config.log_offset, ref=np.max)
        logger.debug("mel_spectrogram: %s", log_mel.shape)
        return log_mel.astype(np.float32)

    def mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel-frequency cepstral coefficients.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            MFCC matrix of shape ``(n_mfcc, num_frames)``, float32.
        """
        audio = self._validate(audio)
        coefficients = librosa.feature.mfcc(
            y=audio,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.effective_fmax,
        )
        logger.debug("mfcc: %s", coefficients.shape)
        return coefficients.astype(np.float32)

    def cqt_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute a log-scaled constant-Q transform spectrogram.

        The CQT's logarithmic frequency axis aligns naturally with harmonic
        fault signatures such as gear-mesh tones and their sidebands.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            Log-magnitude CQT of shape ``(cqt_bins, num_frames)``, float32, in dB.
        """
        audio = self._validate(audio)
        cqt = librosa.cqt(
            y=audio,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            n_bins=self.config.cqt_bins,
            bins_per_octave=self.config.cqt_bins_per_octave,
        )
        log_cqt = librosa.amplitude_to_db(
            np.abs(cqt) + self.config.log_offset, ref=np.max
        )
        logger.debug("cqt_spectrogram: %s", log_cqt.shape)
        return log_cqt.astype(np.float32)

    # -- scalar descriptors -------------------------------------------------------

    def spectral_statistics(self, audio: np.ndarray) -> np.ndarray:
        """Compute interpretable scalar spectral descriptors.

        Aggregates frame-wise spectral centroid, bandwidth, rolloff, flatness,
        zero-crossing rate, and RMS energy into per-clip mean and standard
        deviation. Feature order follows :data:`SPECTRAL_STATISTIC_NAMES`.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            Feature vector of shape ``(12,)``, float32.
        """
        audio = self._validate(audio)
        sr = self.config.sample_rate
        kwargs = {"n_fft": self.config.n_fft, "hop_length": self.config.hop_length}

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, **kwargs)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, **kwargs)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, **kwargs)
        flatness = librosa.feature.spectral_flatness(y=audio, **kwargs)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        rms = librosa.feature.rms(
            y=audio, frame_length=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        statistics = np.array(
            [
                centroid.mean(), centroid.std(),
                bandwidth.mean(), bandwidth.std(),
                rolloff.mean(), rolloff.std(),
                flatness.mean(), flatness.std(),
                zcr.mean(), zcr.std(),
                rms.mean(), rms.std(),
            ],
            dtype=np.float32,
        )
        logger.debug("spectral_statistics: %s", statistics.shape)
        return statistics

    # -- composite extractors -------------------------------------------------------

    def mel_3channel(self, audio: np.ndarray) -> np.ndarray:
        """Build a 3-channel mel representation for image-style CNN backbones.

        Stacks the log-mel spectrogram with its first (delta) and second
        (delta-delta) temporal derivatives, mirroring RGB image input.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            Array of shape ``(3, n_mels, num_frames)``, float32, ordered as
            (log-mel, delta, delta-delta).
        """
        log_mel = self.mel_spectrogram(audio)
        delta = librosa.feature.delta(log_mel, order=1)
        delta2 = librosa.feature.delta(log_mel, order=2)
        stacked = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)
        logger.debug("mel_3channel: %s", stacked.shape)
        return stacked

    def extract_all(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Run every extractor on one waveform.

        Individual extractor failures are logged and skipped so one bad
        transform cannot abort a batch feature-extraction run.

        Args:
            audio: 1-D waveform at ``config.sample_rate``.

        Returns:
            Mapping of feature name to float32 array with keys:
            ``mel_spectrogram``, ``mfcc``, ``cqt_spectrogram``,
            ``spectral_statistics``, ``mel_3channel``. Failed extractors
            are omitted.
        """
        extractors = {
            "mel_spectrogram": self.mel_spectrogram,
            "mfcc": self.mfcc,
            "cqt_spectrogram": self.cqt_spectrogram,
            "spectral_statistics": self.spectral_statistics,
            "mel_3channel": self.mel_3channel,
        }
        features: dict[str, np.ndarray] = {}
        for name, extractor in extractors.items():
            try:
                features[name] = extractor(audio)
            except (ValueError, RuntimeError, librosa.ParameterError) as exc:
                logger.error("Extractor '%s' failed: %s", name, exc)
        logger.info("extract_all: produced %d/%d feature sets",
                    len(features), len(extractors))
        return features

    # -- internals ---------------------------------------------------------------

    def _validate(self, audio: np.ndarray) -> np.ndarray:
        """Validate and coerce an input waveform.

        Args:
            audio: Candidate waveform array.

        Returns:
            1-D float32 waveform.

        Raises:
            ValueError: If the input is empty, not 1-D, or contains
                non-finite values.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D waveform, got shape {audio.shape}")
        if audio.size == 0:
            raise ValueError("Empty waveform")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Waveform contains NaN or Inf values")
        if audio.size < self.config.n_fft:
            logger.warning(
                "Waveform shorter than n_fft (%d < %d); features may be degraded",
                audio.size, self.config.n_fft,
            )
        return audio


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract acoustic features from an audio file (smoke test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", type=Path, default=None,
                        help="Audio file to analyze. Omit to use a synthetic test tone.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: extract all features and report their shapes.

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

    config = FeatureConfig()
    extractor = FeatureExtractor(config)

    if args.path is not None:
        if not args.path.is_file():
            logger.error("File not found: %s", args.path)
            return 1
        try:
            audio, _ = librosa.load(args.path, sr=config.sample_rate, mono=True)
        except (RuntimeError, EOFError, OSError, ValueError) as exc:
            logger.error("Failed to load %s: %s", args.path, exc)
            return 1
        logger.info("Loaded %s (%.1f s)", args.path, len(audio) / config.sample_rate)
    else:
        logger.info("No --path given; using a 10 s synthetic test tone")
        t = np.arange(int(config.sample_rate * 10.0)) / config.sample_rate
        rng = np.random.default_rng(42)
        audio = (
            0.5 * np.sin(2.0 * np.pi * 440.0 * t)
            + 0.1 * rng.standard_normal(t.size)
        ).astype(np.float32)

    features = extractor.extract_all(audio)
    if not features:
        logger.error("All extractors failed")
        return 1

    logger.info("Extracted features:")
    for name, array in features.items():
        logger.info("  %-22s shape=%-18s dtype=%s",
                    name, str(array.shape), array.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
