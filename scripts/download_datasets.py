#!/usr/bin/env python3
"""Dataset management for the Wind Turbine Acoustic Monitoring project.

This script manages acquisition of the acoustic datasets used throughout the
project. It supports:

* **CWRU Bearing Dataset** -- direct download of vibration ``.mat`` files.
* **MIMII Dataset** -- guided manual download (Zenodo, ~100 GB total).
* **ToyADMOS Dataset** -- guided manual download (Zenodo).
* **WindTurbineSound Dataset** -- guided manual download.
* **Synthetic data generation** -- physically-motivated synthetic turbine
  audio for pipeline smoke tests and controlled experiments.

Usage::

    python scripts/download_datasets.py --status
    python scripts/download_datasets.py --cwru
    python scripts/download_datasets.py --synthetic
    python scripts/download_datasets.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - dependency guard
    sf = None  # type: ignore[assignment]

logger = logging.getLogger("download_datasets")

#: Repository root resolved relative to this file (scripts/ -> project root).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Default location for all raw datasets.
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata describing one supported dataset.

    Attributes:
        name: Short machine-friendly identifier (also the target sub-folder).
        full_name: Human-readable dataset title.
        url: Landing page or direct download base URL.
        description: One-line summary of the dataset contents.
        approx_size: Approximate on-disk size after download.
        auto_downloadable: Whether this script can fetch it without manual steps.
    """

    name: str
    full_name: str
    url: str
    description: str
    approx_size: str
    auto_downloadable: bool

    @property
    def target_dir(self) -> Path:
        """Directory under ``data/raw/`` where this dataset is stored."""
        return RAW_DATA_DIR / self.name


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for synthetic turbine audio generation.

    Attributes:
        sample_rate: Output sample rate in Hz.
        duration_seconds: Length of each generated clip in seconds.
        clips_per_class: Number of clips generated per fault class.
        classes: Fault classes to synthesize.
        output_dir: Root directory for the generated WAV files.
        rotor_frequency_hz: Nominal rotor rotation frequency (1P) in Hz.
        random_seed: Base seed for reproducible generation.
    """

    sample_rate: int = 22050
    duration_seconds: float = 10.0
    clips_per_class: int = 5
    classes: tuple[str, ...] = (
        "normal",
        "bearing_fault",
        "blade_imbalance",
        "gearbox_fault",
    )
    output_dir: Path = field(default_factory=lambda: RAW_DATA_DIR / "synthetic")
    rotor_frequency_hz: float = 0.3
    random_seed: int = 42

    @property
    def num_samples(self) -> int:
        """Number of audio samples per clip."""
        return int(self.sample_rate * self.duration_seconds)


#: Registry of all supported datasets.
DATASETS: Final[dict[str, DatasetConfig]] = {
    "cwru": DatasetConfig(
        name="cwru",
        full_name="CWRU Bearing Dataset",
        url="https://engineering.case.edu/bearingdatacenter",
        description="Case Western Reserve University bearing fault vibration data.",
        approx_size="~700 MB (full), ~25 MB (sample subset)",
        auto_downloadable=True,
    ),
    "mimii": DatasetConfig(
        name="mimii",
        full_name="MIMII Dataset",
        url="https://zenodo.org/record/3384388",
        description="Malfunctioning industrial machine sounds (fan, pump, slider, valve).",
        approx_size="~100 GB (all SNR variants)",
        auto_downloadable=False,
    ),
    "toyadmos": DatasetConfig(
        name="toyadmos",
        full_name="ToyADMOS Dataset",
        url="https://zenodo.org/record/3351307",
        description="Anomaly detection in machine operating sounds (toy car, conveyor, train).",
        approx_size="~45 GB",
        auto_downloadable=False,
    ),
    "windturbine": DatasetConfig(
        name="windturbine",
        full_name="WindTurbineSound Dataset",
        url="https://zenodo.org/search?q=wind+turbine+acoustic",
        description="Field recordings of operating wind turbines under varying wind conditions.",
        approx_size="varies by source",
        auto_downloadable=False,
    ),
}

#: Small, representative CWRU files (normal baseline + fault conditions).
#: File IDs follow the official CWRU Bearing Data Center numbering scheme.
CWRU_SAMPLE_FILES: Final[dict[str, str]] = {
    "normal_0hp_97.mat": "https://engineering.case.edu/sites/default/files/97.mat",
    "inner_race_007_105.mat": "https://engineering.case.edu/sites/default/files/105.mat",
    "ball_007_118.mat": "https://engineering.case.edu/sites/default/files/118.mat",
    "outer_race_007_130.mat": "https://engineering.case.edu/sites/default/files/130.mat",
}


# ---------------------------------------------------------------------------
# Dataset manager
# ---------------------------------------------------------------------------


class DatasetManager:
    """Manage download, inspection, and synthesis of project datasets.

    The manager is intentionally side-effect free at construction time:
    directories are only created when a download or generation method runs.

    Args:
        raw_data_dir: Root directory for raw datasets. Defaults to
            ``<project_root>/data/raw``.
        synthetic_config: Configuration for synthetic audio generation.
    """

    DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 60
    DOWNLOAD_MAX_RETRIES: Final[int] = 3

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        synthetic_config: SyntheticDataConfig | None = None,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.synthetic_config = synthetic_config or SyntheticDataConfig()

    # -- status -------------------------------------------------------------

    def show_dataset_status(self) -> dict[str, int]:
        """Log the on-disk status of every supported dataset.

        Returns:
            Mapping of dataset name to the number of data files found locally.
        """
        logger.info("=" * 72)
        logger.info("Dataset status (root: %s)", self.raw_data_dir)
        logger.info("=" * 72)

        status: dict[str, int] = {}
        for config in DATASETS.values():
            file_count = self._count_data_files(config.target_dir)
            status[config.name] = file_count
            state = f"{file_count} files" if file_count else "NOT DOWNLOADED"
            logger.info(
                "%-16s | %-28s | %-12s | %s",
                config.name,
                config.full_name,
                state,
                config.approx_size,
            )

        synthetic_dir = self.synthetic_config.output_dir
        synthetic_count = self._count_data_files(synthetic_dir)
        status["synthetic"] = synthetic_count
        state = f"{synthetic_count} files" if synthetic_count else "NOT GENERATED"
        logger.info("%-16s | %-28s | %s", "synthetic", "Synthetic turbine audio", state)
        logger.info("=" * 72)
        return status

    @staticmethod
    def _count_data_files(directory: Path) -> int:
        """Count data files (non-hidden, non-placeholder) under ``directory``."""
        if not directory.is_dir():
            return 0
        return sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )

    # -- CWRU ---------------------------------------------------------------

    def download_cwru(self) -> list[Path]:
        """Download a representative subset of the CWRU Bearing Dataset.

        Fetches one healthy baseline and three fault-condition ``.mat`` files
        from the official CWRU Bearing Data Center. Existing files are skipped.

        Returns:
            Paths of all files present locally after the operation.

        Raises:
            OSError: If the target directory cannot be created.
        """
        config = DATASETS["cwru"]
        config.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s subset to %s", config.full_name, config.target_dir)

        downloaded: list[Path] = []
        for filename, url in CWRU_SAMPLE_FILES.items():
            destination = config.target_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                logger.info("Skipping %s (already present)", filename)
                downloaded.append(destination)
                continue
            try:
                self._download_file(url, destination)
                downloaded.append(destination)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.error("Failed to download %s: %s", filename, exc)
                logger.error(
                    "If the URL has changed, download manually from %s", config.url
                )

        logger.info("CWRU download complete: %d/%d files available",
                    len(downloaded), len(CWRU_SAMPLE_FILES))
        return downloaded

    def _download_file(self, url: str, destination: Path) -> None:
        """Download ``url`` to ``destination`` with retries and atomic write.

        Args:
            url: Source URL.
            destination: Final file path. A ``.part`` temp file is used during
                transfer and renamed on success.

        Raises:
            urllib.error.URLError: If all retry attempts fail.
        """
        temp_path = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None

        for attempt in range(1, self.DOWNLOAD_MAX_RETRIES + 1):
            try:
                logger.info(
                    "Downloading %s (attempt %d/%d)",
                    url, attempt, self.DOWNLOAD_MAX_RETRIES,
                )
                request = urllib.request.Request(
                    url, headers={"User-Agent": "wind-turbine-acoustics/1.0"}
                )
                with urllib.request.urlopen(
                    request, timeout=self.DOWNLOAD_TIMEOUT_SECONDS
                ) as response, temp_path.open("wb") as handle:
                    while chunk := response.read(1024 * 256):
                        handle.write(chunk)
                temp_path.replace(destination)
                logger.info(
                    "Saved %s (%.2f MB)",
                    destination.name,
                    destination.stat().st_size / 1e6,
                )
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                logger.warning("Attempt %d failed: %s", attempt, exc)
                temp_path.unlink(missing_ok=True)

        raise urllib.error.URLError(
            f"All {self.DOWNLOAD_MAX_RETRIES} attempts failed for {url}: {last_error}"
        )

    # -- manual-download instructions ----------------------------------------

    def download_mimii_instructions(self) -> None:
        """Log step-by-step instructions for obtaining the MIMII dataset."""
        config = DATASETS["mimii"]
        self._log_instructions(
            config,
            steps=(
                f"Visit {config.url}",
                "Choose the SNR variant (-6 dB, 0 dB, or 6 dB) and machine type "
                "(fan recordings are most relevant to turbine acoustics).",
                "Download the desired ZIP archives (each ~10 GB).",
                f"Extract the archives into {config.target_dir}/",
                "Re-run this script with --status to verify.",
            ),
        )

    def download_toyadmos_instructions(self) -> None:
        """Log step-by-step instructions for obtaining the ToyADMOS dataset."""
        config = DATASETS["toyadmos"]
        self._log_instructions(
            config,
            steps=(
                f"Visit {config.url}",
                "Download the case archives (ToyCar, ToyConveyor, or ToyTrain).",
                "Concatenate the multi-part 7z archives if required, then extract.",
                f"Place the extracted folders under {config.target_dir}/",
                "Re-run this script with --status to verify.",
            ),
        )

    def download_windturbine_instructions(self) -> None:
        """Log step-by-step instructions for obtaining wind turbine recordings."""
        config = DATASETS["windturbine"]
        self._log_instructions(
            config,
            steps=(
                f"Search Zenodo / IEEE DataPort for turbine acoustic datasets: {config.url}",
                "Verify the license permits research use before downloading.",
                "Prefer recordings with documented wind speed and turbine state metadata.",
                f"Place WAV/FLAC files under {config.target_dir}/",
                "Re-run this script with --status to verify.",
            ),
        )

    @staticmethod
    def _log_instructions(config: DatasetConfig, steps: tuple[str, ...]) -> None:
        """Log a formatted manual-download instruction block."""
        logger.info("-" * 72)
        logger.info("%s -- manual download required (%s)",
                    config.full_name, config.approx_size)
        logger.info("%s", config.description)
        for index, step in enumerate(steps, start=1):
            logger.info("  %d. %s", index, step)
        logger.info("-" * 72)

    # -- synthetic data -------------------------------------------------------

    def create_sample_data(self) -> list[Path]:
        """Generate synthetic turbine audio clips for every fault class.

        Produces ``clips_per_class`` WAV files per class under
        ``data/raw/synthetic/<class>/``, each ``duration_seconds`` long at
        ``sample_rate`` Hz. Signals combine sinusoids, pink noise, impulse
        trains, and harmonic stacks chosen to mimic each fault signature.

        Returns:
            Paths of all WAV files written.

        Raises:
            RuntimeError: If the ``soundfile`` dependency is not installed.
            OSError: If output directories or files cannot be written.
        """
        if sf is None:
            raise RuntimeError(
                "soundfile is required for synthetic data generation. "
                "Install dependencies with: pip install -r requirements.txt"
            )

        config = self.synthetic_config
        written: list[Path] = []
        generators = {
            "normal": self._synthesize_normal,
            "bearing_fault": self._synthesize_bearing_fault,
            "blade_imbalance": self._synthesize_blade_imbalance,
            "gearbox_fault": self._synthesize_gearbox_fault,
        }

        logger.info(
            "Generating synthetic data: %d classes x %d clips, %.0f s @ %d Hz",
            len(config.classes),
            config.clips_per_class,
            config.duration_seconds,
            config.sample_rate,
        )

        for class_name in config.classes:
            generator = generators.get(class_name)
            if generator is None:
                logger.warning("No generator registered for class '%s'; skipping",
                               class_name)
                continue

            class_dir = config.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for clip_index in range(config.clips_per_class):
                seed = config.random_seed + hash(class_name) % 10_000 + clip_index
                rng = np.random.default_rng(seed)
                signal = generator(rng)
                signal = self._normalize(signal)

                destination = class_dir / f"{class_name}_{clip_index:03d}.wav"
                try:
                    sf.write(destination, signal, config.sample_rate, subtype="PCM_16")
                except (OSError, RuntimeError) as exc:
                    logger.error("Failed to write %s: %s", destination, exc)
                    continue
                written.append(destination)
                logger.debug("Wrote %s", destination)

            logger.info("Class '%s': %d clips written to %s",
                        class_name, config.clips_per_class, class_dir)

        logger.info("Synthetic generation complete: %d files", len(written))
        return written

    # -- signal building blocks ----------------------------------------------

    def _time_axis(self) -> np.ndarray:
        """Return the time axis vector for one clip."""
        config = self.synthetic_config
        return np.arange(config.num_samples, dtype=np.float64) / config.sample_rate

    def _pink_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Generate pink (1/f) noise via spectral shaping of white noise.

        Args:
            rng: Seeded random generator for reproducibility.

        Returns:
            Pink noise array of length ``num_samples``, unit variance.
        """
        n = self.synthetic_config.num_samples
        white = rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(n, d=1.0 / self.synthetic_config.sample_rate)
        # Avoid division by zero at DC; shape amplitude by 1/sqrt(f).
        scaling = np.ones_like(frequencies)
        scaling[1:] = 1.0 / np.sqrt(frequencies[1:])
        pink = np.fft.irfft(spectrum * scaling, n=n)
        return pink / (np.std(pink) + 1e-12)

    def _harmonic_stack(
        self,
        fundamental_hz: float,
        num_harmonics: int,
        decay: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build a stack of decaying harmonics with random phases.

        Args:
            fundamental_hz: Fundamental frequency in Hz.
            num_harmonics: Number of harmonics to sum (including fundamental).
            decay: Per-harmonic amplitude decay factor in (0, 1].
            rng: Seeded random generator.

        Returns:
            Summed harmonic signal of length ``num_samples``.
        """
        t = self._time_axis()
        signal = np.zeros_like(t)
        for k in range(1, num_harmonics + 1):
            amplitude = decay ** (k - 1)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal += amplitude * np.sin(2.0 * np.pi * fundamental_hz * k * t + phase)
        return signal

    def _impulse_train(
        self,
        repetition_hz: float,
        resonance_hz: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a train of exponentially-decaying resonant impulses.

        Models bearing-defect impacts: each impact excites a structural
        resonance that rings down rapidly.

        Args:
            repetition_hz: Impact repetition rate (defect frequency) in Hz.
            resonance_hz: Excited structural resonance frequency in Hz.
            rng: Seeded random generator (adds jitter to timing and amplitude).

        Returns:
            Impulse train signal of length ``num_samples``.
        """
        config = self.synthetic_config
        signal = np.zeros(config.num_samples, dtype=np.float64)
        period_samples = config.sample_rate / repetition_hz
        ring_length = int(0.01 * config.sample_rate)  # 10 ms ring-down
        ring_t = np.arange(ring_length) / config.sample_rate
        ring = np.exp(-ring_t / 0.002) * np.sin(2.0 * np.pi * resonance_hz * ring_t)

        position = 0.0
        while position < config.num_samples - ring_length:
            jitter = rng.uniform(-0.02, 0.02) * period_samples
            start = int(position + jitter)
            if 0 <= start < config.num_samples - ring_length:
                amplitude = rng.uniform(0.7, 1.0)
                signal[start : start + ring_length] += amplitude * ring
            position += period_samples
        return signal

    @staticmethod
    def _normalize(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
        """Peak-normalize a signal to ``peak`` to avoid clipping on export."""
        max_abs = np.max(np.abs(signal))
        if max_abs < 1e-12:
            return signal
        return (signal / max_abs * peak).astype(np.float32)

    # -- per-class synthesizers ------------------------------------------------

    def _synthesize_normal(self, rng: np.random.Generator) -> np.ndarray:
        """Healthy baseline: broadband pink noise + stable blade-pass tone.

        Args:
            rng: Seeded random generator.

        Returns:
            Synthetic 'normal operation' signal.
        """
        config = self.synthetic_config
        blade_pass_hz = 3.0 * config.rotor_frequency_hz  # 3 blades -> 3P
        aerodynamic = self._pink_noise(rng)
        blade_pass = 0.3 * self._harmonic_stack(blade_pass_hz, 4, 0.6, rng)
        hum = 0.05 * np.sin(2.0 * np.pi * 50.0 * self._time_axis())
        return aerodynamic + blade_pass + hum

    def _synthesize_bearing_fault(self, rng: np.random.Generator) -> np.ndarray:
        """Bearing wear: baseline + high-frequency resonant impulse train.

        Impacts repeat at a characteristic outer-race defect frequency and
        excite a kHz-range structural resonance.

        Args:
            rng: Seeded random generator.

        Returns:
            Synthetic 'bearing fault' signal.
        """
        baseline = self._synthesize_normal(rng)
        bpfo_hz = rng.uniform(85.0, 115.0)  # outer-race defect frequency
        resonance_hz = rng.uniform(3000.0, 5000.0)
        impacts = 0.8 * self._impulse_train(bpfo_hz, resonance_hz, rng)
        return baseline + impacts

    def _synthesize_blade_imbalance(self, rng: np.random.Generator) -> np.ndarray:
        """Blade imbalance: baseline with strong 1P amplitude modulation.

        An unbalanced rotor modulates aerodynamic noise at the rotation
        frequency and boosts the 1P tonal component.

        Args:
            rng: Seeded random generator.

        Returns:
            Synthetic 'blade imbalance' signal.
        """
        config = self.synthetic_config
        t = self._time_axis()
        rotor_hz = config.rotor_frequency_hz * rng.uniform(0.9, 1.1)
        modulation_depth = rng.uniform(0.5, 0.8)
        modulation = 1.0 + modulation_depth * np.sin(2.0 * np.pi * rotor_hz * t)
        baseline = self._synthesize_normal(rng)
        one_p_tone = 0.4 * self._harmonic_stack(rotor_hz, 3, 0.5, rng)
        return baseline * modulation + one_p_tone

    def _synthesize_gearbox_fault(self, rng: np.random.Generator) -> np.ndarray:
        """Gearbox fault: gear-mesh harmonics with shaft-rate sidebands.

        A damaged gear produces strong tonal energy at the gear-mesh frequency
        and its harmonics, amplitude-modulated at the shaft rotation rate
        (which creates the characteristic sidebands).

        Args:
            rng: Seeded random generator.

        Returns:
            Synthetic 'gearbox fault' signal.
        """
        t = self._time_axis()
        baseline = self._synthesize_normal(rng)
        mesh_hz = rng.uniform(380.0, 460.0)
        shaft_hz = rng.uniform(18.0, 25.0)
        mesh_tones = self._harmonic_stack(mesh_hz, 5, 0.7, rng)
        sideband_modulation = 1.0 + 0.6 * np.sin(2.0 * np.pi * shaft_hz * t)
        return baseline + 0.6 * mesh_tones * sideband_modulation


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Dataset management for Wind Turbine Acoustic Monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--status", action="store_true",
                        help="Show local status of all datasets.")
    parser.add_argument("--cwru", action="store_true",
                        help="Download the CWRU bearing dataset subset.")
    parser.add_argument("--mimii", action="store_true",
                        help="Show MIMII manual download instructions.")
    parser.add_argument("--toyadmos", action="store_true",
                        help="Show ToyADMOS manual download instructions.")
    parser.add_argument("--windturbine", action="store_true",
                        help="Show WindTurbineSound manual download instructions.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic turbine audio samples.")
    parser.add_argument("--all", action="store_true",
                        help="Run every action above.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

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

    actions_requested = any(
        (args.status, args.cwru, args.mimii, args.toyadmos,
         args.windturbine, args.synthetic, args.all)
    )
    if not actions_requested:
        parser.print_help()
        return 0

    manager = DatasetManager()
    try:
        if args.status or args.all:
            manager.show_dataset_status()
        if args.cwru or args.all:
            manager.download_cwru()
        if args.mimii or args.all:
            manager.download_mimii_instructions()
        if args.toyadmos or args.all:
            manager.download_toyadmos_instructions()
        if args.windturbine or args.all:
            manager.download_windturbine_instructions()
        if args.synthetic or args.all:
            manager.create_sample_data()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 1
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        logger.error("Fatal error: %s", exc)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
