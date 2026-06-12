#!/usr/bin/env python3
"""Comprehensive test suite for the preprocessing stack.

Covers:

* ``audio_loader.py`` -- loading, resampling, normalization, chunking.
* ``feature_extractor.py`` -- output shapes, dtypes, and numerical health.
* ``augmentation.py`` -- wind-noise synthesis, SNR mixing, variant counts.
* ``denoiser.py`` -- every algorithm preserves shape and produces finite output.
* ``pipeline.py`` -- end-to-end processing, statistics, and validation reports.
* Robustness -- empty/NaN/Inf waveforms and corrupted files never crash the
  pipeline.

Run with::

    pytest tests/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.audio_loader import (
    FAULT_LABELS,
    AudioClip,
    AudioConfig,
    AudioLoader,
)
from src.preprocessing.augmentation import (
    AugmentationConfig,
    AugmentationPipeline,
    WindNoiseGenerator,
)
from src.preprocessing.denoiser import Denoiser, DenoiserConfig
from src.preprocessing.feature_extractor import FeatureConfig, FeatureExtractor
from src.preprocessing.pipeline import PipelineConfig, PreprocessingPipeline

SAMPLE_RATE = 22050
CLIP_DURATION = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def sine_wave() -> np.ndarray:
    """One-second 440 Hz sine wave at the test sample rate."""
    t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture()
def audio_config() -> AudioConfig:
    """Loader configuration with short clips to keep tests fast."""
    return AudioConfig(sample_rate=SAMPLE_RATE, duration=CLIP_DURATION)


@pytest.fixture()
def loader(audio_config: AudioConfig) -> AudioLoader:
    """Generic audio loader under test."""
    return AudioLoader(audio_config, dataset_name="test")


@pytest.fixture()
def wav_file(tmp_path: Path, sine_wave: np.ndarray) -> Path:
    """A valid 1-second WAV file on disk."""
    path = tmp_path / "normal" / "normal_000.wav"
    path.parent.mkdir(parents=True)
    sf.write(path, sine_wave, SAMPLE_RATE, subtype="PCM_16")
    return path


@pytest.fixture()
def dataset_dir(tmp_path: Path, sine_wave: np.ndarray,
                rng: np.random.Generator) -> Path:
    """A small two-class dataset directory (2 clips per class)."""
    root = tmp_path / "dataset"
    for fault_type in ("normal", "bearing_fault"):
        class_dir = root / fault_type
        class_dir.mkdir(parents=True)
        for index in range(2):
            wave = sine_wave + 0.05 * rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
            sf.write(class_dir / f"{fault_type}_{index:03d}.wav",
                     wave, SAMPLE_RATE, subtype="PCM_16")
    return root


@pytest.fixture()
def clip(sine_wave: np.ndarray) -> AudioClip:
    """An in-memory AudioClip for component-level tests."""
    return AudioClip(
        waveform=sine_wave,
        sample_rate=SAMPLE_RATE,
        path=Path("memory/normal_clip.wav"),
        label=FAULT_LABELS["normal"],
        fault_type="normal",
        dataset="test",
    )


# ---------------------------------------------------------------------------
# Audio loader
# ---------------------------------------------------------------------------


class TestAudioLoader:
    """Tests for ``audio_loader.AudioLoader``."""

    def test_load_valid_wav(self, loader: AudioLoader, wav_file: Path) -> None:
        """A valid WAV loads into a correctly-shaped, labelled AudioClip."""
        result = loader.load(wav_file)
        assert result is not None
        assert result.sample_rate == SAMPLE_RATE
        assert result.num_samples == CLIP_SAMPLES
        assert result.waveform.dtype == np.float32
        assert result.fault_type == "normal"
        assert result.label == FAULT_LABELS["normal"]

    def test_load_invalid_path_returns_none(self, loader: AudioLoader,
                                            tmp_path: Path) -> None:
        """A missing file returns None instead of raising."""
        assert loader.load(tmp_path / "does_not_exist.wav") is None

    def test_resample_changes_length(self, loader: AudioLoader) -> None:
        """Resampling from half the target rate roughly doubles the length."""
        original = np.sin(np.linspace(0, 100, 11025)).astype(np.float32)
        resampled = loader.resample(original, orig_sample_rate=11025)
        assert resampled.dtype == np.float32
        assert abs(len(resampled) - 2 * len(original)) <= 2

    def test_resample_noop_at_target_rate(self, loader: AudioLoader,
                                          sine_wave: np.ndarray) -> None:
        """Resampling at the target rate preserves length exactly."""
        assert len(loader.resample(sine_wave, SAMPLE_RATE)) == len(sine_wave)

    def test_normalize_peak(self, loader: AudioLoader) -> None:
        """Normalization scales the absolute peak to the requested value."""
        audio = np.array([0.5, -0.25, 0.1], dtype=np.float32)
        normalized = loader.normalize(audio, peak=0.95)
        assert np.isclose(np.max(np.abs(normalized)), 0.95, atol=1e-6)

    def test_normalize_silence_is_safe(self, loader: AudioLoader) -> None:
        """Normalizing silence returns silence without dividing by zero."""
        silence = np.zeros(100, dtype=np.float32)
        assert np.array_equal(loader.normalize(silence), silence)

    def test_long_recording_chunking(self, loader: AudioLoader,
                                     tmp_path: Path) -> None:
        """A 3-second recording chunks into three 1-second clips."""
        long_wave = np.sin(
            2.0 * np.pi * 440.0 * np.arange(3 * CLIP_SAMPLES) / SAMPLE_RATE
        ).astype(np.float32)
        path = tmp_path / "long.wav"
        sf.write(path, long_wave, SAMPLE_RATE, subtype="PCM_16")

        chunks = loader.load_long_recording(path, chunk_duration=CLIP_DURATION)
        assert len(chunks) == 3
        assert [c.clip_index for c in chunks] == [0, 1, 2]
        assert all(c.num_samples == CLIP_SAMPLES for c in chunks)

    def test_load_directory(self, loader: AudioLoader, dataset_dir: Path) -> None:
        """Directory loading finds every clip and infers both classes."""
        clips = loader.load_directory(dataset_dir)
        assert len(clips) == 4
        assert {c.fault_type for c in clips} == {"normal", "bearing_fault"}


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    """Tests for ``feature_extractor.FeatureExtractor``."""

    @pytest.fixture()
    def extractor(self) -> FeatureExtractor:
        """Feature extractor with default research configuration."""
        return FeatureExtractor(FeatureConfig(sample_rate=SAMPLE_RATE))

    def test_mel_spectrogram_shape(self, extractor: FeatureExtractor,
                                   sine_wave: np.ndarray) -> None:
        """Mel spectrogram has n_mels rows and float32 dtype."""
        mel = extractor.mel_spectrogram(sine_wave)
        assert mel.shape[0] == extractor.config.n_mels
        assert mel.ndim == 2
        assert mel.dtype == np.float32

    def test_mfcc_shape(self, extractor: FeatureExtractor,
                        sine_wave: np.ndarray) -> None:
        """MFCC matrix has n_mfcc rows."""
        mfcc = extractor.mfcc(sine_wave)
        assert mfcc.shape[0] == extractor.config.n_mfcc
        assert mfcc.dtype == np.float32

    def test_cqt_shape(self, extractor: FeatureExtractor,
                       sine_wave: np.ndarray) -> None:
        """CQT spectrogram has cqt_bins rows."""
        cqt = extractor.cqt_spectrogram(sine_wave)
        assert cqt.shape[0] == extractor.config.cqt_bins
        assert cqt.dtype == np.float32

    def test_spectral_statistics_dimensions(self, extractor: FeatureExtractor,
                                            sine_wave: np.ndarray) -> None:
        """Spectral statistics form a fixed 12-dimensional vector."""
        stats = extractor.spectral_statistics(sine_wave)
        assert stats.shape == (12,)
        assert stats.dtype == np.float32

    def test_mel_3channel_shape(self, extractor: FeatureExtractor,
                                sine_wave: np.ndarray) -> None:
        """3-channel mel stack is (3, n_mels, frames)."""
        stacked = extractor.mel_3channel(sine_wave)
        assert stacked.shape[0] == 3
        assert stacked.shape[1] == extractor.config.n_mels

    def test_no_nan_in_any_feature(self, extractor: FeatureExtractor,
                                   sine_wave: np.ndarray) -> None:
        """Every extractor produces fully finite output."""
        features = extractor.extract_all(sine_wave)
        assert set(features) == {
            "mel_spectrogram", "mfcc", "cqt_spectrogram",
            "spectral_statistics", "mel_3channel",
        }
        for name, array in features.items():
            assert np.all(np.isfinite(array)), f"{name} contains NaN/Inf"


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Tests for ``augmentation.WindNoiseGenerator`` and ``AugmentationPipeline``."""

    @pytest.fixture()
    def aug_config(self) -> AugmentationConfig:
        """Fast augmentation configuration for short test clips."""
        return AugmentationConfig(
            sample_rate=SAMPLE_RATE,
            variants_per_clip=2,
            max_time_shift_seconds=0.1,
        )

    def test_wind_noise_generation(self, aug_config: AugmentationConfig,
                                   rng: np.random.Generator) -> None:
        """Generated wind noise has the requested length, is finite, and
        has approximately unit variance."""
        generator = WindNoiseGenerator(aug_config)
        noise = generator.generate(CLIP_SAMPLES, rng=rng)
        assert noise.shape == (CLIP_SAMPLES,)
        assert noise.dtype == np.float32
        assert np.all(np.isfinite(noise))
        assert np.isclose(np.std(noise), 1.0, atol=0.05)

    def test_wind_noise_reproducible(self, aug_config: AugmentationConfig) -> None:
        """Identical seeds produce identical noise realizations."""
        generator = WindNoiseGenerator(aug_config)
        first = generator.generate(CLIP_SAMPLES, rng=np.random.default_rng(7))
        second = generator.generate(CLIP_SAMPLES, rng=np.random.default_rng(7))
        assert np.array_equal(first, second)

    def test_wind_noise_rejects_invalid_length(self,
                                               aug_config: AugmentationConfig) -> None:
        """Non-positive sample counts raise ValueError."""
        with pytest.raises(ValueError):
            WindNoiseGenerator(aug_config).generate(0)

    def test_snr_mixing_accuracy(self, rng: np.random.Generator) -> None:
        """mix_at_snr hits the requested SNR within 0.1 dB (no clipping)."""
        t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
        signal = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        noise = rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
        target_snr = 10.0

        mixture = WindNoiseGenerator.mix_at_snr(signal, noise, target_snr)
        residual = mixture.astype(np.float64) - signal
        achieved = 10.0 * np.log10(np.mean(signal.astype(np.float64) ** 2)
                                   / np.mean(residual ** 2))
        assert mixture.shape == signal.shape
        assert abs(achieved - target_snr) < 0.1

    def test_snr_mixing_rejects_silent_signal(self,
                                              rng: np.random.Generator) -> None:
        """Mixing a silent signal raises ValueError (SNR undefined)."""
        with pytest.raises(ValueError):
            WindNoiseGenerator.mix_at_snr(
                np.zeros(100), rng.standard_normal(100), 10.0)

    def test_augmentation_count(self, aug_config: AugmentationConfig,
                                clip: AudioClip) -> None:
        """augment() returns exactly variants_per_clip new clips."""
        pipeline = AugmentationPipeline(aug_config)
        variants = pipeline.augment(clip)
        assert len(variants) == aug_config.variants_per_clip
        assert all(v.metadata["augmented"] for v in variants)

    def test_waveform_length_preserved(self, aug_config: AugmentationConfig,
                                       clip: AudioClip) -> None:
        """Every augmented variant keeps the source waveform length."""
        pipeline = AugmentationPipeline(aug_config)
        for variant in pipeline.augment(clip):
            assert variant.num_samples == clip.num_samples
            assert np.all(np.isfinite(variant.waveform))

    def test_source_clip_not_modified(self, aug_config: AugmentationConfig,
                                      clip: AudioClip) -> None:
        """Augmentation never mutates the source clip waveform."""
        original = clip.waveform.copy()
        AugmentationPipeline(aug_config).augment(clip)
        assert np.array_equal(clip.waveform, original)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class TestDenoiser:
    """Tests for ``denoiser.Denoiser``."""

    @pytest.fixture()
    def denoiser(self) -> Denoiser:
        """Denoiser with the default research configuration."""
        return Denoiser(DenoiserConfig(sample_rate=SAMPLE_RATE))

    @pytest.fixture()
    def noisy_wave(self, sine_wave: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        """Sine wave corrupted with broadband noise."""
        return (sine_wave
                + 0.2 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)

    @pytest.mark.parametrize("method", [
        "spectral_subtraction", "wiener", "wavelet", "noisereduce",
    ])
    def test_method_output_health(self, denoiser: Denoiser,
                                  noisy_wave: np.ndarray, method: str) -> None:
        """Each algorithm preserves shape and produces finite float32 output."""
        denoised = denoiser.denoise(noisy_wave, method=method)
        assert denoised.shape == noisy_wave.shape
        assert denoised.dtype == np.float32
        assert np.all(np.isfinite(denoised))

    def test_unknown_method_returns_input(self, denoiser: Denoiser,
                                          noisy_wave: np.ndarray) -> None:
        """An unknown method name degrades gracefully to the input signal."""
        result = denoiser.denoise(noisy_wave, method="quantum_filter")
        assert np.array_equal(result, noisy_wave)

    def test_estimate_snr_with_reference(self, denoiser: Denoiser,
                                         sine_wave: np.ndarray,
                                         noisy_wave: np.ndarray) -> None:
        """Reference-based SNR of a noisy signal is finite and plausible."""
        snr = denoiser.estimate_snr(noisy_wave, clean_reference=sine_wave)
        assert np.isfinite(snr)
        assert -10.0 < snr < 40.0

    def test_evaluate_denoising_keys(self, denoiser: Denoiser,
                                     sine_wave: np.ndarray,
                                     noisy_wave: np.ndarray) -> None:
        """Evaluation returns the documented result dictionary."""
        denoised = denoiser.denoise(noisy_wave, method="wavelet")
        report = denoiser.evaluate_denoising(
            noisy_wave, denoised, "wavelet", clean_reference=sine_wave)
        assert set(report) == {"input_snr", "output_snr",
                               "snr_improvement", "method"}
        assert report["method"] == "wavelet"

    def test_benchmark_methods_ranked(self, denoiser: Denoiser,
                                      sine_wave: np.ndarray,
                                      noisy_wave: np.ndarray) -> None:
        """Benchmarking ranks all methods by output SNR with full metrics."""
        results = denoiser.benchmark_methods(noisy_wave,
                                             clean_reference=sine_wave)
        assert len(results) == 4
        assert [row["rank"] for row in results] == [1, 2, 3, 4]
        snrs = [row["output_snr"] for row in results]
        assert snrs == sorted(snrs, reverse=True)
        for row in results:
            assert "processing_time_seconds" in row
            assert "energy_preservation" in row


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    """Tests for ``pipeline.PreprocessingPipeline``."""

    @pytest.fixture()
    def pipeline(self, dataset_dir: Path, tmp_path: Path,
                 audio_config: AudioConfig) -> PreprocessingPipeline:
        """Pipeline over the small temp dataset with fast settings."""
        config = PipelineConfig(
            input_dir=dataset_dir,
            output_dir=tmp_path / "processed",
            enable_denoising=False,
            enable_augmentation=True,
            n_augmentations_per_clip=1,
            loader_name="generic",
        )
        return PreprocessingPipeline(config, audio_config=audio_config)

    def test_process_clip(self, pipeline: PreprocessingPipeline,
                          clip: AudioClip) -> None:
        """process_clip returns all feature arrays plus metadata."""
        record = pipeline.process_clip(clip)
        assert record is not None
        assert set(record) == {"mel", "mfcc", "cqt",
                               "spectral_features", "metadata"}
        assert record["metadata"]["fault_type"] == "normal"
        for key in ("mel", "mfcc", "cqt", "spectral_features"):
            assert np.all(np.isfinite(record[key]))

    def test_process_dataset(self, pipeline: PreprocessingPipeline) -> None:
        """process_dataset expands 4 sources into originals + variants."""
        summary = pipeline.process_dataset()
        assert summary["failed"] == 0
        assert summary["processed"] == 4 * (1 + 1)  # original + 1 variant
        assert summary["augmented"] == 4
        assert summary["elapsed_seconds"] > 0

    def test_generate_dataset_statistics(self, pipeline: PreprocessingPipeline) -> None:
        """Statistics include all documented keys and a JSON report on disk."""
        summary = pipeline.process_dataset()
        stats = pipeline.generate_dataset_statistics(summary)
        for key in ("total_clips", "classes", "clips_per_class",
                    "duration_hours", "augmented_clips", "processing_time"):
            assert key in stats
        assert stats["total_clips"] == summary["processed"]
        assert sorted(stats["classes"]) == ["bearing_fault", "normal"]
        assert (pipeline.config.output_dir / "dataset_report.json").is_file()

    def test_validate_dataset_clean(self, pipeline: PreprocessingPipeline) -> None:
        """A healthy dataset validates with no issues."""
        report = pipeline.validate_dataset()
        assert report["is_valid"]
        assert report["corrupted_files"] == []
        assert report["empty_clips"] == []
        assert (pipeline.config.output_dir / "validation_report.json").is_file()

    def test_validate_dataset_flags_corrupted_file(
            self, pipeline: PreprocessingPipeline, dataset_dir: Path) -> None:
        """A garbage WAV file is reported as corrupted, not raised."""
        bad_file = dataset_dir / "normal" / "corrupted.wav"
        bad_file.write_bytes(b"this is definitely not audio data")
        report = pipeline.validate_dataset()
        assert not report["is_valid"]
        assert any("corrupted.wav" in path for path in report["corrupted_files"])


# ---------------------------------------------------------------------------
# Robustness: the pipeline must never crash
# ---------------------------------------------------------------------------


class TestRobustness:
    """Degenerate-input tests across the whole stack."""

    @pytest.fixture()
    def denoiser(self) -> Denoiser:
        """Denoiser with the default configuration."""
        return Denoiser(DenoiserConfig(sample_rate=SAMPLE_RATE))

    def test_denoiser_empty_waveform(self, denoiser: Denoiser) -> None:
        """Empty input passes through every method without raising."""
        empty = np.array([], dtype=np.float32)
        for method in ("spectral_subtraction", "wiener",
                       "wavelet", "noisereduce"):
            result = denoiser.denoise(empty, method=method)
            assert result.size == 0

    def test_denoiser_nan_waveform(self, denoiser: Denoiser,
                                   sine_wave: np.ndarray) -> None:
        """NaN samples are sanitized and the output stays finite."""
        corrupted = sine_wave.copy()
        corrupted[::100] = np.nan
        result = denoiser.denoise(corrupted, method="wavelet")
        assert np.all(np.isfinite(result))

    def test_denoiser_inf_waveform(self, denoiser: Denoiser,
                                   sine_wave: np.ndarray) -> None:
        """Inf samples are sanitized and the output stays finite."""
        corrupted = sine_wave.copy()
        corrupted[::200] = np.inf
        result = denoiser.denoise(corrupted, method="wiener")
        assert np.all(np.isfinite(result))

    def test_feature_extractor_rejects_nan(self, sine_wave: np.ndarray) -> None:
        """The feature extractor fails fast with ValueError on NaN input."""
        extractor = FeatureExtractor(FeatureConfig(sample_rate=SAMPLE_RATE))
        corrupted = sine_wave.copy()
        corrupted[0] = np.nan
        with pytest.raises(ValueError):
            extractor.mel_spectrogram(corrupted)

    def test_pipeline_process_clip_nan_returns_none(
            self, tmp_path: Path, audio_config: AudioConfig) -> None:
        """process_clip on a NaN waveform returns None instead of raising."""
        pipeline = PreprocessingPipeline(
            PipelineConfig(input_dir=tmp_path, output_dir=tmp_path / "out",
                           enable_denoising=False),
            audio_config=audio_config,
        )
        bad_clip = AudioClip(
            waveform=np.full(CLIP_SAMPLES, np.nan, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            path=Path("memory/bad.wav"),
        )
        assert pipeline.process_clip(bad_clip) is None

    def test_corrupted_file_skipped_by_loader(self, loader: AudioLoader,
                                              tmp_path: Path,
                                              sine_wave: np.ndarray) -> None:
        """A corrupted file is skipped while valid neighbours still load."""
        good = tmp_path / "normal" / "good.wav"
        good.parent.mkdir(parents=True)
        sf.write(good, sine_wave, SAMPLE_RATE, subtype="PCM_16")
        (tmp_path / "normal" / "broken.wav").write_bytes(b"\x00garbage\x00")

        clips = loader.load_directory(tmp_path)
        assert len(clips) == 1
        assert clips[0].path == good

    def test_empty_dataset_directory(self, tmp_path: Path,
                                     audio_config: AudioConfig) -> None:
        """An empty dataset directory yields an empty, well-formed summary."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pipeline = PreprocessingPipeline(
            PipelineConfig(input_dir=empty_dir, output_dir=tmp_path / "out"),
            audio_config=audio_config,
        )
        summary = pipeline.process_dataset()
        assert summary["processed"] == 0
        assert summary["records"] == []
