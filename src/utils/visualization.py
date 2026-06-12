#!/usr/bin/env python3
"""Publication-quality visualization for Wind Turbine Acoustic Monitoring.

Why visualization is critical for acoustic fault diagnosis
----------------------------------------------------------
Acoustic fault signatures are fundamentally *visual* in the time-frequency
domain: bearing impacts appear as periodic vertical striations, gear damage
as harmonic ladders with sidebands, blade imbalance as low-frequency
amplitude banding. Spectrogram inspection is how domain experts validate
that a model is reacting to physics rather than artifacts, and how data
problems (clipping, residual wind noise, label errors) are caught before
they silently poison training.

For explainable AI, visualization is the delivery mechanism: a Grad-CAM
heatmap overlaid on a mel spectrogram (Week 8) shows *which* time-frequency
regions drove a fault prediction, letting engineers confirm that a
"bearing fault" call attends to impact bands and not to gusts of wind.
Without this, the system is an unauditable black box that maintenance teams
rightly will not trust.

Every figure produced here is publication-ready (configurable DPI, vector
formats, consistent styling) and doubles as GitHub documentation and model
debugging output.

Usage::

    python -m src.utils.visualization --output docs/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # headless backend for CI and servers

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("visualization")

#: Export formats supported by :class:`VisualizationManager`.
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset({"png", "pdf", "svg"})

#: Default class names matching the project fault taxonomy.
DEFAULT_CLASS_NAMES: Final[tuple[str, ...]] = (
    "normal",
    "blade_imbalance",
    "bearing_fault",
    "gearbox_fault",
    "electrical_fault",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for figure generation and export.

    Attributes:
        dpi: Raster export resolution (300+ for print publication).
        figure_size: Default (width, height) in inches.
        save_format: Export format: 'png', 'pdf', or 'svg'.
        output_dir: Default directory for saved figures.
        style: Seaborn style name (e.g., 'whitegrid', 'ticks').
        font_scale: Seaborn font scaling for axis labels and titles.
        sample_rate: Sample rate assumed for time axes in Hz.
        hop_length: STFT hop used for spectrogram time axes.
        colormap: Default colormap for spectrograms.
    """

    dpi: int = 300
    figure_size: tuple[float, float] = (10.0, 6.0)
    save_format: str = "png"
    output_dir: Path = field(default_factory=lambda: Path("docs/figures"))
    style: str = "whitegrid"
    font_scale: float = 1.2
    sample_rate: int = 22050
    hop_length: int = 512
    colormap: str = "magma"


# ---------------------------------------------------------------------------
# Visualization manager
# ---------------------------------------------------------------------------


class VisualizationManager:
    """Generate and export all project figures with consistent styling.

    Every public plot method builds a figure, saves it via :meth:`_save`,
    closes it (preventing memory leaks in long runs), and returns the saved
    path or None on failure -- plotting errors are logged, never raised, so
    a broken figure cannot abort a training or evaluation run.

    Args:
        config: Visualization configuration. Defaults to
            :class:`VisualizationConfig`.
        class_names: Ordered class names used for label axes.
    """

    def __init__(
        self,
        config: VisualizationConfig | None = None,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ) -> None:
        self.config = config or VisualizationConfig()
        self.class_names = class_names
        if self.config.save_format not in SUPPORTED_FORMATS:
            logger.warning("Unsupported format '%s'; falling back to png",
                           self.config.save_format)
            object.__setattr__(self.config, "save_format", "png")
        sns.set_theme(style=self.config.style, font_scale=self.config.font_scale)

    # -- signal-level figures ----------------------------------------------------

    def plot_waveform(
        self,
        waveform: np.ndarray,
        title: str = "Waveform",
        filename: str = "waveform",
    ) -> Path | None:
        """Plot a time-domain waveform.

        Args:
            waveform: 1-D audio samples at ``config.sample_rate``.
            title: Figure title.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            waveform = self._validate_array(waveform, ndim=1)
            t = np.arange(len(waveform)) / self.config.sample_rate
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.plot(t, waveform, linewidth=0.5, color="#1f77b4")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(title)
            ax.set_xlim(0, t[-1] if t.size else 1)
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_waveform failed: %s", exc)
            return None

    def plot_spectrogram(
        self,
        mel_spectrogram: np.ndarray,
        title: str = "Log-Mel Spectrogram",
        filename: str = "spectrogram",
    ) -> Path | None:
        """Plot a log-mel spectrogram with a calibrated colorbar.

        Args:
            mel_spectrogram: Log-mel matrix ``(n_mels, n_frames)`` in dB.
            title: Figure title.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            mel_spectrogram = self._validate_array(mel_spectrogram, ndim=2)
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            image = librosa.display.specshow(
                mel_spectrogram,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                x_axis="time",
                y_axis="mel",
                cmap=self.config.colormap,
                ax=ax,
            )
            fig.colorbar(image, ax=ax, format="%+2.0f dB", label="Power (dB)")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel frequency (Hz)")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_spectrogram failed: %s", exc)
            return None

    def compare_spectrograms(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        titles: tuple[str, str] = ("Original (noisy)", "Denoised"),
        filename: str = "spectrogram_comparison",
    ) -> Path | None:
        """Plot original vs. denoised spectrograms side-by-side.

        Both panels share the color scale so the noise-floor reduction is
        visually honest.

        Args:
            original: Log-mel spectrogram before denoising.
            denoised: Log-mel spectrogram after denoising.
            titles: Panel titles (left, right).
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            original = self._validate_array(original, ndim=2)
            denoised = self._validate_array(denoised, ndim=2)
            vmin = min(original.min(), denoised.min())
            vmax = max(original.max(), denoised.max())

            width, height = self.config.figure_size
            fig, axes = plt.subplots(1, 2, figsize=(width * 1.6, height),
                                     sharey=True)
            for ax, data, title in zip(axes, (original, denoised), titles):
                image = librosa.display.specshow(
                    data, sr=self.config.sample_rate,
                    hop_length=self.config.hop_length,
                    x_axis="time", y_axis="mel",
                    cmap=self.config.colormap, vmin=vmin, vmax=vmax, ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("Time (s)")
            axes[0].set_ylabel("Mel frequency (Hz)")
            axes[1].set_ylabel("")
            fig.colorbar(image, ax=axes, format="%+2.0f dB", label="Power (dB)",
                         shrink=0.9)
            return self._save(fig, filename, tight=False)
        except (ValueError, RuntimeError) as exc:
            logger.error("compare_spectrograms failed: %s", exc)
            return None

    # -- dataset figures ------------------------------------------------------------

    def plot_feature_distributions(
        self,
        features: dict[str, np.ndarray],
        filename: str = "feature_distributions",
    ) -> Path | None:
        """Plot histograms and boxplots for feature sets (MFCC, spectral stats).

        Each feature set gets a histogram (pooled values) and a per-dimension
        boxplot row.

        Args:
            features: Mapping of feature-set name to a 2-D array
                ``(n_dims, n_observations)`` or 1-D vector.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            if not features:
                raise ValueError("No feature sets provided")
            n_rows = len(features)
            width, height = self.config.figure_size
            fig, axes = plt.subplots(n_rows, 2,
                                     figsize=(width * 1.4, height * 0.8 * n_rows),
                                     squeeze=False)
            for row, (name, values) in enumerate(features.items()):
                values = np.atleast_2d(self._validate_array(np.asarray(values)))
                pooled = values.ravel()
                sns.histplot(pooled, bins=60, kde=True, ax=axes[row][0],
                             color="#1f77b4")
                axes[row][0].set_title(f"{name}: value distribution")
                axes[row][0].set_xlabel("Value")

                max_dims = min(values.shape[0], 20)  # keep boxplots readable
                axes[row][1].boxplot([values[i] for i in range(max_dims)],
                                     showfliers=False)
                axes[row][1].set_title(f"{name}: per-dimension spread")
                axes[row][1].set_xlabel("Dimension")
                axes[row][1].set_ylabel("Value")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_feature_distributions failed: %s", exc)
            return None

    def plot_class_distribution(
        self,
        clips_per_class: dict[str, int],
        title: str = "Dataset Class Distribution",
        filename: str = "class_distribution",
    ) -> Path | None:
        """Plot the number of clips per fault class.

        Args:
            clips_per_class: Mapping of class name to clip count.
            title: Figure title.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            if not clips_per_class:
                raise ValueError("Empty class distribution")
            names = list(clips_per_class)
            counts = [clips_per_class[name] for name in names]
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            bars = ax.bar(names, counts, color=sns.color_palette("deep", len(names)))
            ax.bar_label(bars, padding=3)
            ax.set_xlabel("Fault class")
            ax.set_ylabel("Number of clips")
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_class_distribution failed: %s", exc)
            return None

    # -- model evaluation figures -------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalized: bool = True,
        filename: str = "confusion_matrix",
    ) -> Path | None:
        """Plot a confusion matrix annotated with fault labels.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            normalized: If True, rows are normalized to recall fractions.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            labels = np.arange(len(self.class_names))
            matrix = sk_confusion_matrix(y_true, y_pred, labels=labels).astype(float)
            if normalized:
                row_sums = matrix.sum(axis=1, keepdims=True)
                matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                                   where=row_sums > 0)
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.heatmap(matrix, annot=True,
                        fmt=".2f" if normalized else ".0f",
                        cmap="Blues", cbar=True, square=True,
                        xticklabels=self.class_names,
                        yticklabels=self.class_names, ax=ax)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalized else ""))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_confusion_matrix failed: %s", exc)
            return None

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        filename: str = "roc_curves",
    ) -> Path | None:
        """Plot one-vs-rest ROC curves for every class.

        Args:
            y_true: Ground-truth integer labels.
            y_proba: Class probability matrix ``(n_samples, n_classes)``.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            classes = np.arange(y_proba.shape[1])
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            for index in classes:
                fpr, tpr, _ = roc_curve(y_bin[:, index], y_proba[:, index])
                name = (self.class_names[index]
                        if index < len(self.class_names) else f"class {index}")
                ax.plot(fpr, tpr, linewidth=1.8,
                        label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Multi-class ROC Curves (one-vs-rest)")
            ax.legend(loc="lower right", fontsize="small")
            return self._save(fig, filename)
        except (ValueError, IndexError) as exc:
            logger.error("plot_roc_curves failed: %s", exc)
            return None

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        filename: str = "precision_recall_curves",
    ) -> Path | None:
        """Plot one-vs-rest precision-recall curves for every class.

        Args:
            y_true: Ground-truth integer labels.
            y_proba: Class probability matrix ``(n_samples, n_classes)``.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            classes = np.arange(y_proba.shape[1])
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            for index in classes:
                precision, recall, _ = precision_recall_curve(
                    y_bin[:, index], y_proba[:, index])
                ap = average_precision_score(y_bin[:, index], y_proba[:, index])
                name = (self.class_names[index]
                        if index < len(self.class_names) else f"class {index}")
                ax.plot(recall, precision, linewidth=1.8,
                        label=f"{name} (AP = {ap:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Multi-class Precision-Recall Curves (one-vs-rest)")
            ax.legend(loc="lower left", fontsize="small")
            return self._save(fig, filename)
        except (ValueError, IndexError) as exc:
            logger.error("plot_precision_recall_curves failed: %s", exc)
            return None

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        filename: str = "training_history",
    ) -> Path | None:
        """Plot training curves (loss, accuracy, F1) over epochs.

        Keys ending in ``loss`` share the left panel; all other metrics
        (accuracy, F1, ...) share the right panel. Validation curves are
        plotted dashed when keys are prefixed with ``val_``.

        Args:
            history: Mapping of metric name to per-epoch values, e.g.
                ``{'loss': [...], 'val_loss': [...], 'accuracy': [...],
                'f1': [...]}``.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            if not history:
                raise ValueError("Empty training history")
            width, height = self.config.figure_size
            fig, (ax_loss, ax_metric) = plt.subplots(
                1, 2, figsize=(width * 1.5, height))
            for name, values in history.items():
                axis = ax_loss if name.endswith("loss") else ax_metric
                style = "--" if name.startswith("val_") else "-"
                axis.plot(range(1, len(values) + 1), values, style,
                          linewidth=1.8, label=name)
            ax_loss.set_title("Loss")
            ax_metric.set_title("Metrics")
            for axis in (ax_loss, ax_metric):
                axis.set_xlabel("Epoch")
                axis.legend(fontsize="small")
            ax_loss.set_ylabel("Loss")
            ax_metric.set_ylabel("Score")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_training_history failed: %s", exc)
            return None

    # -- explainability figures ------------------------------------------------------------

    def plot_gradcam_overlay(
        self,
        spectrogram: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        title: str = "Grad-CAM Explanation",
        filename: str = "gradcam_overlay",
    ) -> Path | None:
        """Overlay a Grad-CAM heatmap on a spectrogram (Week 8 explainability).

        The heatmap is bilinearly resized to the spectrogram resolution,
        min-max normalized, and alpha-blended on top, highlighting the
        time-frequency regions that drove the model's prediction.

        Args:
            spectrogram: Log-mel spectrogram ``(n_mels, n_frames)``.
            heatmap: Grad-CAM activation map (any 2-D shape).
            alpha: Heatmap opacity in [0, 1].
            title: Figure title.
            filename: Output filename stem.

        Returns:
            Path of the saved figure, or None on failure.
        """
        try:
            spectrogram = self._validate_array(spectrogram, ndim=2)
            heatmap = self._validate_array(heatmap, ndim=2)
            zoom_factors = (spectrogram.shape[0] / heatmap.shape[0],
                            spectrogram.shape[1] / heatmap.shape[1])
            resized = ndimage.zoom(heatmap, zoom_factors, order=1)
            span = resized.max() - resized.min()
            resized = (resized - resized.min()) / (span + 1e-12)

            fig, ax = plt.subplots(figsize=self.config.figure_size)
            base = librosa.display.specshow(
                spectrogram, sr=self.config.sample_rate,
                hop_length=self.config.hop_length,
                x_axis="time", y_axis="mel", cmap="gray_r", ax=ax,
            )
            overlay = ax.imshow(
                resized, aspect="auto", origin="lower", cmap="jet",
                alpha=alpha, extent=base.get_extent(),
            )
            fig.colorbar(overlay, ax=ax, label="Model attention")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel frequency (Hz)")
            return self._save(fig, filename)
        except (ValueError, RuntimeError) as exc:
            logger.error("plot_gradcam_overlay failed: %s", exc)
            return None

    # -- paper automation ---------------------------------------------------------------------

    def generate_paper_figures(
        self,
        figure_inputs: dict[str, dict[str, Any]],
        paper_dir: Path | str = Path("paper/figures"),
    ) -> list[Path]:
        """Generate sequentially numbered figures for the paper directory.

        Each entry maps a plot method name to its keyword arguments; figures
        are saved as ``Figure1.<fmt>``, ``Figure2.<fmt>``, ... in input order.

        Example::

            manager.generate_paper_figures({
                "plot_waveform": {"waveform": audio},
                "plot_spectrogram": {"mel_spectrogram": mel},
            })

        Args:
            figure_inputs: Ordered mapping of method name to kwargs
                (``filename`` is overridden automatically).
            paper_dir: Destination directory for paper figures.

        Returns:
            Paths of all successfully generated figures.
        """
        paper_dir = Path(paper_dir)
        original_dir = self.config.output_dir
        object.__setattr__(self.config, "output_dir", paper_dir)
        generated: list[Path] = []
        try:
            for number, (method_name, kwargs) in enumerate(
                    figure_inputs.items(), start=1):
                method = getattr(self, method_name, None)
                if method is None or not callable(method):
                    logger.error("Unknown plot method '%s'; skipping", method_name)
                    continue
                kwargs = {**kwargs, "filename": f"Figure{number}"}
                path = method(**kwargs)
                if path is not None:
                    generated.append(path)
        finally:
            object.__setattr__(self.config, "output_dir", original_dir)
        logger.info("Paper figures: %d/%d generated in %s",
                    len(generated), len(figure_inputs), paper_dir)
        return generated

    # -- internals -------------------------------------------------------------------------------

    def _save(
        self,
        fig: plt.Figure,
        filename: str,
        tight: bool = True,
    ) -> Path | None:
        """Save and close a figure, logging path and generation time.

        Args:
            fig: Matplotlib figure to save.
            filename: Output filename stem (extension added from config).
            tight: If True, apply tight layout before saving.

        Returns:
            Saved path, or None on failure.
        """
        start = time.perf_counter()
        destination = (self.config.output_dir
                       / f"{filename}.{self.config.save_format}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if tight:
                fig.tight_layout()
            fig.savefig(destination, dpi=self.config.dpi,
                        format=self.config.save_format, bbox_inches="tight")
            elapsed = time.perf_counter() - start
            logger.info("Figure saved: %s (%.2f s)", destination, elapsed)
            return destination
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Failed to save figure %s: %s", destination, exc)
            return None
        finally:
            plt.close(fig)

    @staticmethod
    def _validate_array(array: np.ndarray, ndim: int | None = None) -> np.ndarray:
        """Validate a numeric array input.

        Args:
            array: Candidate array.
            ndim: Required dimensionality, if any.

        Returns:
            Float64 array with NaN/Inf replaced by zeros.

        Raises:
            ValueError: If the array is empty or has the wrong rank.
        """
        array = np.asarray(array, dtype=np.float64)
        if array.size == 0:
            raise ValueError("Empty array")
        if ndim is not None and array.ndim != ndim:
            raise ValueError(f"Expected {ndim}-D array, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            logger.warning("Array contains NaN/Inf; replacing with zeros")
            array = np.nan_to_num(array)
        return array


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate demonstration figures from synthetic audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("docs/figures"),
                        help="Output directory for figures.")
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/figures"),
                        help="Output directory for numbered paper figures.")
    parser.add_argument("--format", choices=sorted(SUPPORTED_FORMATS),
                        default="png", help="Figure export format.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: generate demonstration and paper figures.

    Synthesizes a fault-like signal with wind noise, denoises it, and
    produces waveform, spectrogram, comparison, and paper figures.

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

    config = VisualizationConfig(output_dir=args.output, save_format=args.format)
    manager = VisualizationManager(config)

    # 1. Build a synthetic noisy fault signal and its denoised version.
    from src.preprocessing.augmentation import WindNoiseGenerator
    from src.preprocessing.denoiser import Denoiser
    from src.preprocessing.feature_extractor import FeatureExtractor

    rng = np.random.default_rng(42)
    num_samples = config.sample_rate * 10
    t = np.arange(num_samples) / config.sample_rate
    clean = (0.4 * np.sin(2 * np.pi * 420.0 * t)
             + 0.2 * np.sin(2 * np.pi * 840.0 * t)).astype(np.float32)
    wind = WindNoiseGenerator().generate(num_samples, rng=rng)
    noisy = WindNoiseGenerator.mix_at_snr(clean, wind, snr_db=8.0)
    denoised = Denoiser().denoise(noisy, method="spectral_subtraction")

    extractor = FeatureExtractor()
    mel_noisy = extractor.mel_spectrogram(noisy)
    mel_denoised = extractor.mel_spectrogram(denoised)

    # 2-4. Waveform, spectrogram, and comparison figures.
    results = [
        manager.plot_waveform(noisy, title="Synthetic Turbine Signal (noisy)"),
        manager.plot_spectrogram(mel_noisy),
        manager.compare_spectrograms(mel_noisy, mel_denoised),
        manager.plot_class_distribution(
            {"normal": 20, "bearing_fault": 20, "blade_imbalance": 20,
             "gearbox_fault": 20}),
    ]

    # 5. Numbered paper figures.
    paper = manager.generate_paper_figures(
        {
            "plot_waveform": {"waveform": noisy,
                              "title": "Turbine Acoustic Signal"},
            "plot_spectrogram": {"mel_spectrogram": mel_noisy},
            "compare_spectrograms": {"original": mel_noisy,
                                     "denoised": mel_denoised},
        },
        paper_dir=args.paper_dir,
    )

    succeeded = sum(1 for r in results if r is not None) + len(paper)
    logger.info("Generated %d figures", succeeded)
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
