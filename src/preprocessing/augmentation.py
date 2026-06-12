#!/usr/bin/env python3
"""Wind-noise synthesis and data augmentation for turbine acoustic monitoring.

This module implements the project's core research contribution -- the
:class:`WindNoiseGenerator` -- and an :class:`AugmentationPipeline` that
produces robust training variants of every clip.

Wind model:

* **1/f² spectrum** -- Brownian (red) spectral shaping matching the
  low-frequency energy roll-off of natural atmospheric wind.
* **Gust modulation** -- stochastic low-frequency (0.1-2 Hz) amplitude
  envelopes simulating gust events and lulls.
* **Turbulence component** -- band-limited stochastic fluctuations whose
  intensity scales with wind speed (2-15 m/s).
* **SNR-controlled mixing** -- deterministic mixing of fault signals with
  generated noise at precise target SNRs (5-25 dB by default).

Usage::

    python -m src.preprocessing.augmentation --input data/raw/synthetic --output data/augmented
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
from scipy import signal as sps

from src.preprocessing.audio_loader import AudioClip, AudioConfig, AudioLoader

try:
    from audiomentations import AddGaussianSNR
except ImportError:  # pragma: no cover - optional dependency guard
    AddGaussianSNR = None  # type: ignore[assignment]

logger = logging.getLogger("augmentation")

#: Speed of sound used by the distance simulation (m/s at ~15 degC).
SPEED_OF_SOUND: Final[float] = 340.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for wind-noise synthesis and augmentation.

    Attributes:
        sample_rate: Sample rate of all processed audio in Hz.
        snr_range_db: (min, max) target signal-to-noise ratio in dB for
            wind-noise mixing.
        wind_speed_range: (min, max) simulated wind speed in m/s.
        gust_frequency_range_hz: (min, max) gust modulation frequency in Hz.
        gust_depth_range: (min, max) gust modulation depth in [0, 1].
        turbulence_band_hz: (low, high) band of the turbulence component in Hz.
        turbulence_intensity: Base turbulence mix level at the maximum wind speed.
        max_time_shift_seconds: Maximum circular time shift magnitude.
        amplitude_scale_range: (min, max) random gain factors.
        gaussian_snr_range_db: (min, max) SNR for additive Gaussian noise.
        distance_range_m: (min, max) simulated source distance in metres.
        reference_distance_m: Distance at which the source level is defined.
        variants_per_clip: Number of augmented variants produced per input clip.
        random_seed: Base seed for reproducible augmentation.
    """

    sample_rate: int = 22050
    snr_range_db: tuple[float, float] = (5.0, 25.0)
    wind_speed_range: tuple[float, float] = (2.0, 15.0)
    gust_frequency_range_hz: tuple[float, float] = (0.1, 2.0)
    gust_depth_range: tuple[float, float] = (0.3, 0.8)
    turbulence_band_hz: tuple[float, float] = (200.0, 2000.0)
    turbulence_intensity: float = 0.35
    max_time_shift_seconds: float = 1.0
    amplitude_scale_range: tuple[float, float] = (0.6, 1.4)
    gaussian_snr_range_db: tuple[float, float] = (20.0, 40.0)
    distance_range_m: tuple[float, float] = (20.0, 200.0)
    reference_distance_m: float = 20.0
    variants_per_clip: int = 3
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Wind noise generator
# ---------------------------------------------------------------------------


class WindNoiseGenerator:
    """Physically-motivated synthetic wind noise generator.

    Produces wind noise with a 1/f² power spectrum, stochastic gust
    modulation, and a wind-speed-dependent turbulence component, and mixes
    it with fault signals at exact target SNRs.

    Args:
        config: Augmentation configuration. Defaults to
            :class:`AugmentationConfig`.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def generate(
        self,
        num_samples: int,
        wind_speed: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate one wind-noise realization.

        Pipeline: white noise -> 1/f² spectral shaping -> gust modulation ->
        additive band-limited turbulence scaled by wind speed.

        Args:
            num_samples: Length of the noise signal in samples.
            wind_speed: Simulated wind speed in m/s. Drawn uniformly from
                ``config.wind_speed_range`` when None.
            rng: Seeded random generator. A default seeded generator is
                created when None.

        Returns:
            Unit-variance wind noise of shape ``(num_samples,)``, float32.

        Raises:
            ValueError: If ``num_samples`` is not positive or ``wind_speed``
                is non-positive.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        rng = rng or np.random.default_rng(self.config.random_seed)
        low, high = self.config.wind_speed_range
        wind_speed = float(wind_speed) if wind_speed is not None else float(
            rng.uniform(low, high)
        )
        if wind_speed <= 0.0:
            raise ValueError(f"wind_speed must be positive, got {wind_speed}")

        # Base broadband component with 1/f^2 spectrum.
        white = rng.standard_normal(num_samples)
        wind = self._shape_spectrum(white)

        # Slow gusting envelope.
        wind = self._add_gusts(wind, rng)

        # Wind-speed-dependent turbulence: stronger and relatively louder
        # at higher wind speeds.
        speed_factor = (wind_speed - low) / max(high - low, 1e-9)
        turbulence = self._turbulence(num_samples, rng)
        wind = wind + self.config.turbulence_intensity * (0.3 + 0.7 * speed_factor) * turbulence

        # Overall level grows roughly with the 6th power of wind speed in
        # aeroacoustics; here we normalize to unit variance and let
        # mix_at_snr control absolute levels deterministically.
        wind = wind / (np.std(wind) + 1e-12)
        logger.debug("Generated wind noise: %d samples @ %.1f m/s",
                     num_samples, wind_speed)
        return wind.astype(np.float32)

    def _shape_spectrum(self, white: np.ndarray) -> np.ndarray:
        """Apply 1/f² (Brownian) power spectral shaping to white noise.

        Scales FFT amplitudes by 1/f, which yields a 1/f² power spectrum.

        Args:
            white: White Gaussian noise array.

        Returns:
            Spectrally shaped noise, unit variance.
        """
        n = len(white)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.config.sample_rate)
        scaling = np.ones_like(frequencies)
        scaling[1:] = 1.0 / frequencies[1:]  # amplitude 1/f -> power 1/f^2
        scaling[0] = 0.0  # remove DC
        shaped = np.fft.irfft(spectrum * scaling, n=n)
        return shaped / (np.std(shaped) + 1e-12)

    def _add_gusts(
        self,
        noise: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Modulate noise with a stochastic low-frequency gust envelope.

        The envelope sums 2-4 sinusoids with random frequencies in
        ``config.gust_frequency_range_hz`` and random phases, scaled to a
        random modulation depth, and is clipped to remain non-negative.

        Args:
            noise: Wind noise signal to modulate.
            rng: Seeded random generator.

        Returns:
            Gust-modulated noise of the same length.
        """
        t = np.arange(len(noise)) / self.config.sample_rate
        f_low, f_high = self.config.gust_frequency_range_hz
        d_low, d_high = self.config.gust_depth_range
        depth = rng.uniform(d_low, d_high)

        num_components = int(rng.integers(2, 5))
        envelope = np.zeros_like(t)
        for _ in range(num_components):
            frequency = rng.uniform(f_low, f_high)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            envelope += np.sin(2.0 * np.pi * frequency * t + phase)
        envelope /= num_components

        modulation = np.clip(1.0 + depth * envelope, 0.05, None)
        return noise * modulation

    def _turbulence(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate the band-limited turbulence component.

        Args:
            num_samples: Signal length in samples.
            rng: Seeded random generator.

        Returns:
            Unit-variance band-passed noise simulating micro-scale turbulence.
        """
        nyquist = self.config.sample_rate / 2.0
        low, high = self.config.turbulence_band_hz
        low_n = max(low / nyquist, 1e-4)
        high_n = min(high / nyquist, 0.999)
        sos = sps.butter(4, [low_n, high_n], btype="bandpass", output="sos")
        turbulence = sps.sosfilt(sos, rng.standard_normal(num_samples))
        return turbulence / (np.std(turbulence) + 1e-12)

    @staticmethod
    def mix_at_snr(
        signal_audio: np.ndarray,
        noise: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """Mix a signal with noise at an exact target SNR.

        The noise is rescaled so that ``10 * log10(P_signal / P_noise)``
        equals ``snr_db``, then added to the signal. The mixture is peak
        normalized only if it would clip.

        Args:
            signal_audio: Clean signal waveform.
            noise: Noise waveform (truncated/tiled to the signal length).
            snr_db: Target signal-to-noise ratio in dB.

        Returns:
            Mixture waveform, float32, same length as ``signal_audio``.

        Raises:
            ValueError: If the signal is silent (SNR undefined).
        """
        signal_audio = np.asarray(signal_audio, dtype=np.float64)
        noise = np.asarray(noise, dtype=np.float64)

        if len(noise) < len(signal_audio):
            repeats = int(np.ceil(len(signal_audio) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[: len(signal_audio)]

        signal_power = float(np.mean(signal_audio**2))
        noise_power = float(np.mean(noise**2))
        if signal_power < 1e-15:
            raise ValueError("Cannot mix at SNR: signal is silent")
        if noise_power < 1e-15:
            logger.warning("Noise is silent; returning signal unchanged")
            return signal_audio.astype(np.float32)

        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = noise * np.sqrt(target_noise_power / noise_power)

        mixture = signal_audio + noise
        peak = float(np.max(np.abs(mixture)))
        if peak > 1.0:
            mixture = mixture / peak * 0.95
        return mixture.astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline:
    """Produce augmented training variants of :class:`AudioClip` objects.

    Each variant applies, in order: circular time shift, amplitude scaling,
    distance simulation, wind-noise mixing at a random SNR, and light
    Gaussian sensor noise. Individual transform failures are logged and the
    untransformed audio is carried forward, so a single bad transform cannot
    abort a dataset run.

    Args:
        config: Augmentation configuration. Defaults to
            :class:`AugmentationConfig`.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self.wind_generator = WindNoiseGenerator(self.config)
        if AddGaussianSNR is None:
            logger.warning(
                "audiomentations not installed; using NumPy fallback for "
                "Gaussian noise"
            )

    # -- public API ----------------------------------------------------------

    def augment(
        self,
        clip: AudioClip,
        rng: np.random.Generator | None = None,
    ) -> list[AudioClip]:
        """Create augmented variants of one clip.

        Args:
            clip: Source clip to augment.
            rng: Seeded random generator. Derived from ``config.random_seed``
                and the clip identity when None.

        Returns:
            ``config.variants_per_clip`` new clips with augmentation
            parameters recorded in ``metadata``. The source clip is not
            modified.
        """
        rng = rng or np.random.default_rng(
            self.config.random_seed
            + (hash((str(clip.path), clip.clip_index)) % 100_000)
        )
        variants: list[AudioClip] = []
        for variant_index in range(self.config.variants_per_clip):
            audio = clip.waveform.astype(np.float32).copy()
            applied: dict[str, float] = {}

            for name, transform in (
                ("time_shift", self._apply_time_shift),
                ("amplitude_scaling", self._apply_amplitude_scaling),
                ("distance_sim", self._apply_distance_sim),
                ("wind_noise", self._apply_wind_noise),
                ("gaussian_noise", self._apply_gaussian_noise),
            ):
                try:
                    audio, parameter = transform(audio, rng)
                    applied[name] = parameter
                except (ValueError, RuntimeError) as exc:
                    logger.error("Transform '%s' failed on %s: %s",
                                 name, clip.path.name, exc)

            variants.append(
                replace(
                    clip,
                    waveform=audio,
                    metadata={
                        **clip.metadata,
                        "augmented": True,
                        "variant_index": variant_index,
                        "augmentations": applied,
                    },
                )
            )
        logger.debug("Augmented %s -> %d variants", clip.path.name, len(variants))
        return variants

    def augment_dataset(self, clips: list[AudioClip]) -> list[AudioClip]:
        """Augment every clip in a dataset.

        Args:
            clips: Source clips.

        Returns:
            All augmented variants (``len(clips) * variants_per_clip`` on
            success; clips that fail entirely are skipped with a logged error).
        """
        augmented: list[AudioClip] = []
        for clip in clips:
            try:
                augmented.extend(self.augment(clip))
            except (ValueError, RuntimeError) as exc:
                logger.error("Skipping %s: %s", clip.path, exc)
        logger.info("Augmented %d clips -> %d variants", len(clips), len(augmented))
        return augmented

    # -- individual transforms ---------------------------------------------------

    def _apply_wind_noise(
        self,
        audio: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Mix in generated wind noise at a random SNR from the config range.

        Args:
            audio: Input waveform.
            rng: Seeded random generator.

        Returns:
            Tuple of (mixed waveform, applied SNR in dB).
        """
        snr_db = float(rng.uniform(*self.config.snr_range_db))
        wind = self.wind_generator.generate(len(audio), rng=rng)
        return self.wind_generator.mix_at_snr(audio, wind, snr_db), snr_db

    def _apply_time_shift(
        self,
        audio: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Apply a random circular time shift.

        Args:
            audio: Input waveform.
            rng: Seeded random generator.

        Returns:
            Tuple of (shifted waveform, shift in seconds).
        """
        max_shift = int(self.config.max_time_shift_seconds * self.config.sample_rate)
        if max_shift == 0:
            return audio, 0.0
        shift = int(rng.integers(-max_shift, max_shift + 1))
        return np.roll(audio, shift).astype(np.float32), shift / self.config.sample_rate

    def _apply_amplitude_scaling(
        self,
        audio: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Apply a random gain factor.

        Args:
            audio: Input waveform.
            rng: Seeded random generator.

        Returns:
            Tuple of (scaled waveform, gain factor).
        """
        gain = float(rng.uniform(*self.config.amplitude_scale_range))
        scaled = np.clip(audio * gain, -1.0, 1.0)
        return scaled.astype(np.float32), gain

    def _apply_gaussian_noise(
        self,
        audio: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Add light Gaussian sensor noise at a random SNR.

        Uses ``audiomentations.AddGaussianSNR`` when available, otherwise a
        NumPy fallback with identical semantics.

        Args:
            audio: Input waveform.
            rng: Seeded random generator.

        Returns:
            Tuple of (noisy waveform, applied SNR in dB).
        """
        snr_db = float(rng.uniform(*self.config.gaussian_snr_range_db))
        if AddGaussianSNR is not None:
            transform = AddGaussianSNR(
                min_snr_db=snr_db, max_snr_db=snr_db, p=1.0
            )
            noisy = transform(samples=audio, sample_rate=self.config.sample_rate)
            return noisy.astype(np.float32), snr_db
        noise = rng.standard_normal(len(audio))
        return WindNoiseGenerator.mix_at_snr(audio, noise, snr_db), snr_db

    def _apply_distance_sim(
        self,
        audio: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Simulate microphone placement at a random distance.

        Applies inverse-distance attenuation relative to
        ``config.reference_distance_m`` and a distance-dependent low-pass
        filter approximating atmospheric high-frequency absorption.

        Args:
            audio: Input waveform.
            rng: Seeded random generator.

        Returns:
            Tuple of (processed waveform, simulated distance in metres).
        """
        distance = float(rng.uniform(*self.config.distance_range_m))
        attenuation = self.config.reference_distance_m / max(
            distance, self.config.reference_distance_m
        )

        # Atmospheric absorption: cutoff drops as distance grows
        # (~8 kHz near reference distance down to ~2 kHz at long range).
        nyquist = self.config.sample_rate / 2.0
        d_low, d_high = self.config.distance_range_m
        distance_factor = (distance - d_low) / max(d_high - d_low, 1e-9)
        cutoff_hz = 8000.0 - 6000.0 * distance_factor
        cutoff_n = min(max(cutoff_hz / nyquist, 1e-3), 0.999)

        sos = sps.butter(2, cutoff_n, btype="lowpass", output="sos")
        processed = sps.sosfilt(sos, audio * attenuation)
        return processed.astype(np.float32), distance


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Augment an audio dataset with synthetic wind noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic"),
                        help="Directory of source audio clips.")
    parser.add_argument("--output", type=Path, default=Path("data/augmented"),
                        help="Directory to write augmented WAV files.")
    parser.add_argument("--variants", type=int, default=3,
                        help="Augmented variants per source clip.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: augment a directory of clips and write WAV files.

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

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile is required to write augmented audio. "
                     "Install dependencies with: pip install -r requirements.txt")
        return 1

    config = AugmentationConfig(variants_per_clip=args.variants)
    loader = AudioLoader(AudioConfig(), dataset_name="synthetic")
    pipeline = AugmentationPipeline(config)

    clips = loader.load_directory(args.input)
    if not clips:
        logger.error("No source clips found under %s", args.input)
        return 1

    augmented = pipeline.augment_dataset(clips)
    if not augmented:
        logger.error("Augmentation produced no clips")
        return 1

    written = 0
    for clip in augmented:
        variant = clip.metadata.get("variant_index", 0)
        destination = (
            args.output / clip.fault_type
            / f"{clip.path.stem}_aug{variant:02d}.wav"
        )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, clip.waveform, clip.sample_rate, subtype="PCM_16")
            written += 1
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to write %s: %s", destination, exc)

    logger.info("Wrote %d/%d augmented clips to %s", written, len(augmented), args.output)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
