#!/usr/bin/env python3
"""Research-grade acoustic denoising for the Wind Turbine Acoustic Monitoring project.

Why wind noise is a problem
---------------------------
Microphones mounted near a turbine record the faults we care about *and* the
environment they live in. Atmospheric wind noise dominates the low-frequency
band with a 1/f² spectrum, is amplitude-modulated by gusts, and can sit only
a few dB below (or above) early-stage fault signatures such as bearing impact
transients and gear-mesh sidebands. Left untreated, wind noise inflates the
false-negative rate of anomaly detectors and blurs the spectral structure the
fault classifier depends on.

Why classical denoising still matters
-------------------------------------
Learned denoisers are powerful but data-hungry, opaque, and risky to deploy
as the *only* defense: they can hallucinate structure and silently fail out
of distribution. Classical methods (spectral subtraction, Wiener filtering,
wavelet shrinkage, spectral gating) are deterministic, interpretable,
cheap enough for edge deployment, and provide the *reference baselines*
every learned method must beat in the paper's evaluation. They also serve
as robust pre-cleaning so downstream models train on better-conditioned
inputs.

Road to the Denoising Autoencoder (Week 4)
------------------------------------------
This module establishes the benchmark harness (:meth:`Denoiser.benchmark_methods`)
and the SNR evaluation protocol that the Week 4 Denoising Autoencoder will be
measured against. The DAE will be trained on pairs produced by the
:class:`~src.preprocessing.augmentation.WindNoiseGenerator` (clean clip,
SNR-controlled noisy mixture) and must outperform the best classical method
here on output SNR at equal or better signature preservation.

Usage::

    python -m src.preprocessing.denoiser --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pywt
from scipy import signal as sps

try:
    import noisereduce as nr
except ImportError:  # pragma: no cover - optional dependency guard
    nr = None  # type: ignore[assignment]

logger = logging.getLogger("denoiser")

#: Wavelet families validated for fault-signature preservation.
SUPPORTED_WAVELETS: Final[frozenset[str]] = frozenset({"db4", "db8", "sym8"})

#: Methods accepted by :meth:`Denoiser.denoise`.
SUPPORTED_METHODS: Final[tuple[str, ...]] = (
    "spectral_subtraction",
    "wiener",
    "wavelet",
    "noisereduce",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenoiserConfig:
    """Configuration for all denoising algorithms.

    Attributes:
        sample_rate: Sample rate of processed audio in Hz.
        spectral_subtraction_alpha: Over-subtraction factor; values > 1
            subtract more than the estimated noise magnitude.
        spectral_floor: Fraction of the noisy magnitude kept as a floor to
            avoid musical-noise artifacts.
        wiener_window_size: Window length (samples) of the Wiener filter;
            small windows preserve impulsive fault transients.
        wavelet_name: Wavelet family ('db4', 'db8', or 'sym8').
        wavelet_level: Wavelet decomposition depth.
        noise_reduce_stationary: If True, noisereduce assumes stationary noise.
        noise_reduce_prop_decrease: Proportion of noise reduction applied by
            noisereduce, in [0, 1].
        save_intermediate_results: If True, intermediate artifacts (noise
            profile, STFT magnitudes) are kept on the instance for inspection.
        enable_logging: If False, this module's logger is silenced.
        n_fft: STFT window size for spectral subtraction.
        hop_length: STFT hop size in samples.
        noise_estimate_seconds: Leading duration assumed noise-dominated when
            estimating the noise profile.
        snr_noise_frame_fraction: Fraction of lowest-energy frames treated as
            noise for blind SNR estimation.
    """

    sample_rate: int = 22050
    spectral_subtraction_alpha: float = 2.0
    spectral_floor: float = 0.05
    wiener_window_size: int = 29
    wavelet_name: str = "db8"
    wavelet_level: int = 5
    noise_reduce_stationary: bool = False
    noise_reduce_prop_decrease: float = 0.9
    save_intermediate_results: bool = False
    enable_logging: bool = True
    n_fft: int = 2048
    hop_length: int = 512
    noise_estimate_seconds: float = 0.5
    snr_noise_frame_fraction: float = 0.1


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class Denoiser:
    """Classical denoising algorithms with a unified evaluation harness.

    All methods accept a 1-D waveform at ``config.sample_rate`` and return a
    ``float32`` waveform of identical length. Invalid input is sanitized
    (NaN/Inf replaced, with a logged warning) and failures degrade gracefully
    by returning the input unchanged -- the pipeline never crashes.

    Args:
        config: Denoiser configuration. Defaults to :class:`DenoiserConfig`.
    """

    def __init__(self, config: DenoiserConfig | None = None) -> None:
        self.config = config or DenoiserConfig()
        self.intermediate: dict[str, np.ndarray] = {}
        if not self.config.enable_logging:
            logger.setLevel(logging.CRITICAL)
        if self.config.wavelet_name not in SUPPORTED_WAVELETS:
            logger.warning(
                "Wavelet '%s' is not in the validated set %s; proceeding anyway",
                self.config.wavelet_name, sorted(SUPPORTED_WAVELETS),
            )

    # -- denoising algorithms -------------------------------------------------

    def spectral_subtraction(
        self,
        audio: np.ndarray,
        noise_profile: np.ndarray | None = None,
    ) -> np.ndarray:
        """Denoise via magnitude spectral subtraction (Boll, 1979).

        Computes the STFT, estimates the noise magnitude spectrum (from
        ``noise_profile`` or the leading ``noise_estimate_seconds`` of the
        clip), over-subtracts it scaled by ``spectral_subtraction_alpha``,
        applies a spectral floor to suppress musical noise, and reconstructs
        with the noisy phase via ISTFT.

        Args:
            audio: 1-D noisy waveform.
            noise_profile: Optional noise-only waveform used to estimate the
                noise spectrum. Falls back to the leading clip segment.

        Returns:
            Denoised float32 waveform, same length as the input.
        """
        audio = self._validate(audio)
        if audio.size == 0:
            return audio
        cfg = self.config

        _, _, stft = sps.stft(
            audio, fs=cfg.sample_rate, nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Noise magnitude estimate: explicit profile, else leading frames.
        if noise_profile is not None and noise_profile.size:
            noise_profile = self._validate(noise_profile)
            _, _, noise_stft = sps.stft(
                noise_profile, fs=cfg.sample_rate, nperseg=cfg.n_fft,
                noverlap=cfg.n_fft - cfg.hop_length,
            )
            noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        else:
            noise_frames = max(
                1,
                int(cfg.noise_estimate_seconds * cfg.sample_rate / cfg.hop_length),
            )
            noise_magnitude = np.mean(
                magnitude[:, :noise_frames], axis=1, keepdims=True
            )

        # Over-subtraction with spectral flooring.
        subtracted = magnitude - cfg.spectral_subtraction_alpha * noise_magnitude
        floored = np.maximum(subtracted, cfg.spectral_floor * magnitude)

        if cfg.save_intermediate_results:
            self.intermediate["noise_magnitude"] = noise_magnitude
            self.intermediate["stft_magnitude"] = magnitude

        _, reconstructed = sps.istft(
            floored * np.exp(1j * phase), fs=cfg.sample_rate,
            nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length,
        )
        return self._fit_length(reconstructed, len(audio))

    def wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """Denoise with a local adaptive Wiener filter.

        Uses :func:`scipy.signal.wiener` with a configurable, deliberately
        small window: short windows track local statistics, which preserves
        the impulsive transients characteristic of bearing defects.

        Args:
            audio: 1-D noisy waveform.

        Returns:
            Denoised float32 waveform, same length as the input.
        """
        audio = self._validate(audio)
        if audio.size == 0:
            return audio
        window = self.config.wiener_window_size
        if window % 2 == 0:
            window += 1  # scipy requires an odd window
            logger.debug("Adjusted Wiener window to odd size %d", window)
        try:
            filtered = sps.wiener(audio.astype(np.float64), mysize=window)
        except (ValueError, FloatingPointError) as exc:
            logger.error("Wiener filter failed: %s; returning input", exc)
            return audio
        return np.nan_to_num(filtered).astype(np.float32)

    def wavelet_denoise(self, audio: np.ndarray) -> np.ndarray:
        """Denoise via wavelet decomposition with soft thresholding.

        Decomposes the signal with the configured wavelet family
        ('db4', 'db8', or 'sym8'), estimates the noise level from the
        finest detail coefficients (median absolute deviation), applies the
        universal threshold ``sigma * sqrt(2 * ln N)`` with soft
        thresholding, and reconstructs.

        Args:
            audio: 1-D noisy waveform.

        Returns:
            Denoised float32 waveform, same length as the input.
        """
        audio = self._validate(audio)
        if audio.size == 0:
            return audio
        cfg = self.config

        if cfg.wavelet_name not in pywt.wavelist():
            logger.error("Invalid wavelet '%s'; returning input", cfg.wavelet_name)
            return audio

        max_level = pywt.dwt_max_level(len(audio), pywt.Wavelet(cfg.wavelet_name))
        level = min(cfg.wavelet_level, max_level)
        if level < 1:
            logger.error("Signal too short for wavelet decomposition; returning input")
            return audio
        if level < cfg.wavelet_level:
            logger.debug("Clamped wavelet level %d -> %d", cfg.wavelet_level, level)

        try:
            coefficients = pywt.wavedec(audio, cfg.wavelet_name, level=level)
            # Robust noise estimate from the finest detail band.
            sigma = np.median(np.abs(coefficients[-1])) / 0.6745
            threshold = sigma * np.sqrt(2.0 * np.log(len(audio)))
            denoised_coefficients = [coefficients[0]] + [
                pywt.threshold(c, threshold, mode="soft")
                for c in coefficients[1:]
            ]
            reconstructed = pywt.waverec(denoised_coefficients, cfg.wavelet_name)
        except (ValueError, RuntimeError) as exc:
            logger.error("Wavelet denoising failed: %s; returning input", exc)
            return audio
        return self._fit_length(reconstructed, len(audio))

    def noise_reduce(self, audio: np.ndarray) -> np.ndarray:
        """Denoise with the ``noisereduce`` spectral gating library.

        Supports stationary mode (single global noise estimate) and
        non-stationary mode (time-varying estimate), controlled by
        ``config.noise_reduce_stationary``.

        Args:
            audio: 1-D noisy waveform.

        Returns:
            Denoised float32 waveform; the input unchanged when the
            ``noisereduce`` package is unavailable or processing fails.
        """
        audio = self._validate(audio)
        if audio.size == 0:
            return audio
        if nr is None:
            logger.error("noisereduce is not installed; returning input. "
                         "Install dependencies with: pip install -r requirements.txt")
            return audio
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.config.sample_rate,
                stationary=self.config.noise_reduce_stationary,
                prop_decrease=self.config.noise_reduce_prop_decrease,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error("noisereduce failed: %s; returning input", exc)
            return audio
        return np.nan_to_num(reduced).astype(np.float32)

    # -- evaluation ------------------------------------------------------------

    def estimate_snr(
        self,
        audio: np.ndarray,
        clean_reference: np.ndarray | None = None,
    ) -> float:
        """Estimate the signal-to-noise ratio in dB.

        With a clean reference: ``SNR = 10 * log10(P_clean / P_residual)``
        where the residual is ``audio - clean_reference``. Without one, a
        blind estimate is used: the quietest ``snr_noise_frame_fraction`` of
        frames define the noise floor, and
        ``SNR = 10 * log10((P_total - P_noise) / P_noise)``.

        Args:
            audio: Waveform to evaluate.
            clean_reference: Optional clean signal of equal length.

        Returns:
            Estimated SNR in dB (clamped to [-60, 100]); 0.0 for degenerate
            input.
        """
        audio = self._validate(audio)
        if audio.size == 0:
            return 0.0

        if clean_reference is not None:
            clean = self._validate(clean_reference)[: len(audio)]
            residual = audio[: len(clean)] - clean
            signal_power = float(np.mean(clean**2))
            noise_power = float(np.mean(residual**2))
        else:
            frame = self.config.n_fft
            hop = self.config.hop_length
            if len(audio) < frame:
                return 0.0
            frame_energies = np.array([
                float(np.mean(audio[start : start + frame] ** 2))
                for start in range(0, len(audio) - frame, hop)
            ])
            if frame_energies.size == 0:
                return 0.0
            k = max(1, int(len(frame_energies) * self.config.snr_noise_frame_fraction))
            noise_power = float(np.mean(np.sort(frame_energies)[:k]))
            total_power = float(np.mean(frame_energies))
            signal_power = max(total_power - noise_power, 0.0)

        if noise_power < 1e-15:
            return 100.0
        if signal_power < 1e-15:
            return -60.0
        return float(np.clip(10.0 * np.log10(signal_power / noise_power), -60.0, 100.0))

    def evaluate_denoising(
        self,
        noisy: np.ndarray,
        denoised: np.ndarray,
        method: str,
        clean_reference: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Quantify denoising quality for one method.

        Args:
            noisy: Input waveform before denoising.
            denoised: Output waveform after denoising.
            method: Name of the method evaluated.
            clean_reference: Optional clean signal for reference-based SNR.

        Returns:
            Dictionary with keys ``input_snr``, ``output_snr``,
            ``snr_improvement`` (all in dB), and ``method``.
        """
        input_snr = self.estimate_snr(noisy, clean_reference)
        output_snr = self.estimate_snr(denoised, clean_reference)
        result = {
            "input_snr": round(input_snr, 2),
            "output_snr": round(output_snr, 2),
            "snr_improvement": round(output_snr - input_snr, 2),
            "method": method,
        }
        logger.info("%s: input %.2f dB -> output %.2f dB (improvement %+.2f dB)",
                    method, input_snr, output_snr, output_snr - input_snr)
        return result

    # -- orchestration -----------------------------------------------------------

    def denoise(self, audio: np.ndarray, method: str = "spectral_subtraction") -> np.ndarray:
        """Route a waveform to the selected denoising algorithm.

        Args:
            audio: 1-D noisy waveform.
            method: One of :data:`SUPPORTED_METHODS`.

        Returns:
            Denoised float32 waveform; the input unchanged when ``method``
            is unknown or the algorithm fails.
        """
        dispatch = {
            "spectral_subtraction": self.spectral_subtraction,
            "wiener": self.wiener_filter,
            "wavelet": self.wavelet_denoise,
            "noisereduce": self.noise_reduce,
        }
        algorithm = dispatch.get(method)
        if algorithm is None:
            logger.error("Unknown method '%s' (expected one of %s); returning input",
                         method, ", ".join(SUPPORTED_METHODS))
            return self._validate(audio)

        start = time.perf_counter()
        denoised = algorithm(audio)
        elapsed = time.perf_counter() - start
        logger.info("Method '%s' completed in %.3f s", method, elapsed)
        return denoised

    def benchmark_methods(
        self,
        noisy: np.ndarray,
        clean_reference: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Benchmark every denoising method on one waveform.

        Runs all of :data:`SUPPORTED_METHODS` and ranks the results by output
        SNR (descending). Intended to generate the comparison tables and
        figures for the paper, and the classical baselines the Week 4
        Denoising Autoencoder must beat.

        Args:
            noisy: Noisy input waveform.
            clean_reference: Optional clean signal for reference-based SNR.

        Returns:
            Ranked list of dictionaries with keys ``rank``, ``method``,
            ``input_snr``, ``output_snr``, ``snr_improvement``,
            ``processing_time_seconds``, and ``energy_preservation`` (output
            signal energy as a fraction of input energy).
        """
        noisy = self._validate(noisy)
        if noisy.size == 0:
            logger.error("Cannot benchmark an empty signal")
            return []

        input_energy = float(np.sum(noisy.astype(np.float64) ** 2)) + 1e-15
        results: list[dict[str, Any]] = []

        for method in SUPPORTED_METHODS:
            start = time.perf_counter()
            denoised = self.denoise(noisy, method=method)
            elapsed = time.perf_counter() - start

            evaluation = self.evaluate_denoising(
                noisy, denoised, method, clean_reference
            )
            evaluation["processing_time_seconds"] = round(elapsed, 4)
            evaluation["energy_preservation"] = round(
                float(np.sum(denoised.astype(np.float64) ** 2)) / input_energy, 4
            )
            results.append(evaluation)

        results.sort(key=lambda row: row["output_snr"], reverse=True)
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
        logger.info("Benchmark complete; best method: %s (%.2f dB output SNR)",
                    results[0]["method"], results[0]["output_snr"])
        return results

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _validate(audio: np.ndarray) -> np.ndarray:
        """Sanitize an input waveform without ever raising.

        Coerces to 1-D float32, replaces NaN/Inf with zeros (logged), and
        returns an empty array for empty input (logged).

        Args:
            audio: Candidate waveform.

        Returns:
            Sanitized 1-D float32 waveform (possibly empty).
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            logger.error("Received empty signal")
            return audio
        if not np.all(np.isfinite(audio)):
            bad = int(np.sum(~np.isfinite(audio)))
            logger.warning("Replacing %d NaN/Inf samples with zeros", bad)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return audio

    @staticmethod
    def _fit_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Pad with zeros or truncate to an exact sample count."""
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if len(audio) >= target_samples:
            return audio[:target_samples]
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark classical denoising methods on synthetic turbine audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snr", type=float, default=5.0,
                        help="Wind-noise mixing SNR in dB for the demo signal.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Demo clip duration in seconds.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: demonstrate and benchmark all denoising methods.

    Builds a synthetic fault-like signal (gear-mesh harmonics + bearing-style
    impulses), corrupts it with generated wind noise at the requested SNR,
    runs every denoising method, and prints a ranked benchmark table.

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

    config = DenoiserConfig()
    denoiser = Denoiser(config)
    rng = np.random.default_rng(42)

    # 1. Synthetic fault-like clean signal: gear-mesh harmonics + impulses.
    num_samples = int(config.sample_rate * args.duration)
    t = np.arange(num_samples) / config.sample_rate
    clean = (
        0.4 * np.sin(2.0 * np.pi * 420.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 840.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 1260.0 * t)
    ).astype(np.float32)
    impulse_period = int(config.sample_rate / 100.0)  # ~100 Hz impacts
    clean[::impulse_period] += 0.5

    # 2. Corrupt with artificial wind noise at the requested SNR.
    try:
        from src.preprocessing.augmentation import WindNoiseGenerator
        wind = WindNoiseGenerator().generate(num_samples, rng=rng)
    except ImportError:
        logger.warning("Augmentation module unavailable; using shaped noise fallback")
        wind = rng.standard_normal(num_samples).astype(np.float32)
    noisy = WindNoiseGeneratorMix = None  # placeholder to satisfy linters
    del WindNoiseGeneratorMix
    noise_power = float(np.mean(wind**2))
    signal_power = float(np.mean(clean**2))
    wind = wind * np.sqrt(signal_power / (noise_power * 10.0 ** (args.snr / 10.0)))
    noisy = (clean + wind).astype(np.float32)
    logger.info("Demo signal: %.1f s at %d Hz, wind noise mixed at %.1f dB SNR",
                args.duration, config.sample_rate, args.snr)

    # 3-4. Apply all methods and compare SNR improvements.
    results = denoiser.benchmark_methods(noisy, clean_reference=clean)
    if not results:
        logger.error("Benchmark produced no results")
        return 1

    # 5. Print the benchmark table.
    header = (f"{'rank':>4} | {'method':<22} | {'in SNR':>8} | {'out SNR':>8} | "
              f"{'gain dB':>8} | {'time s':>8} | {'energy':>7}")
    logger.info("%s", header)
    logger.info("%s", "-" * len(header))
    for row in results:
        logger.info(
            "%4d | %-22s | %8.2f | %8.2f | %+8.2f | %8.4f | %7.4f",
            row["rank"], row["method"], row["input_snr"], row["output_snr"],
            row["snr_improvement"], row["processing_time_seconds"],
            row["energy_preservation"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
