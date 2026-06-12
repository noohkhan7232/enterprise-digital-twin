#!/usr/bin/env python3
"""Research-grade evaluation metrics for Wind Turbine Acoustic Monitoring.

Provides the unified evaluation protocol used across the project and in the
paper:

* **Classification metrics** -- accuracy, precision, recall, F1, ROC-AUC,
  and PR-AUC (macro and weighted averaging).
* **Confusion matrix** and **per-class metrics** tables.
* **Temporal detection metrics** -- Mean Time To Detection (MTTD) and
  per-episode fault detection latency, the operationally critical numbers
  for condition monitoring.
* **Dataset-level evaluation reports** with JSON / CSV export and
  publication-ready Markdown / LaTeX tables.
* **Bootstrap confidence intervals** so every reported number carries
  statistical uncertainty.

Usage::

    python -m src.utils.metrics --output data/processed/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("metrics")

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
class MetricsConfig:
    """Configuration for the evaluation protocol.

    Attributes:
        class_names: Ordered class names; index equals integer label.
        bootstrap_iterations: Resamples used for confidence intervals.
        confidence_level: Two-sided confidence level in (0, 1).
        random_seed: Seed for reproducible bootstrap resampling.
        zero_division: Value reported when a metric is undefined
            (e.g., precision with no positive predictions).
    """

    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    zero_division: float = 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class MetricsEvaluator:
    """Compute, aggregate, and export evaluation metrics.

    All methods accept integer label arrays (and, where relevant, an
    ``(n_samples, n_classes)`` probability matrix) and validate inputs
    before computing. Failures in optional metrics (e.g., ROC-AUC with a
    missing class) are logged and reported as NaN rather than raised, so an
    evaluation run never crashes a training loop.

    Args:
        config: Evaluation configuration. Defaults to :class:`MetricsConfig`.
    """

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self.config = config or MetricsConfig()

    # -- classification metrics ----------------------------------------------

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute headline classification metrics.

        Args:
            y_true: Ground-truth integer labels, shape ``(n_samples,)``.
            y_pred: Predicted integer labels, shape ``(n_samples,)``.
            y_proba: Optional class probabilities, shape
                ``(n_samples, n_classes)``; required for ROC-AUC and PR-AUC.

        Returns:
            Dictionary with ``accuracy``, ``precision_macro``,
            ``recall_macro``, ``f1_macro``, ``precision_weighted``,
            ``recall_weighted``, ``f1_weighted``, ``roc_auc_macro``, and
            ``pr_auc_macro`` (AUCs are NaN when probabilities are absent or
            degenerate).
        """
        y_true, y_pred = self._validate_labels(y_true, y_pred)
        zero_div = self.config.zero_division
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(
                y_true, y_pred, average="macro", zero_division=zero_div)),
            "recall_macro": float(recall_score(
                y_true, y_pred, average="macro", zero_division=zero_div)),
            "f1_macro": float(f1_score(
                y_true, y_pred, average="macro", zero_division=zero_div)),
            "precision_weighted": float(precision_score(
                y_true, y_pred, average="weighted", zero_division=zero_div)),
            "recall_weighted": float(recall_score(
                y_true, y_pred, average="weighted", zero_division=zero_div)),
            "f1_weighted": float(f1_score(
                y_true, y_pred, average="weighted", zero_division=zero_div)),
            "roc_auc_macro": float("nan"),
            "pr_auc_macro": float("nan"),
        }

        if y_proba is not None:
            try:
                classes = np.arange(y_proba.shape[1])
                y_true_bin = label_binarize(y_true, classes=classes)
                if y_true_bin.shape[1] == 1:  # binary edge case
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                metrics["roc_auc_macro"] = float(roc_auc_score(
                    y_true_bin, y_proba, average="macro", multi_class="ovr"))
                metrics["pr_auc_macro"] = float(average_precision_score(
                    y_true_bin, y_proba, average="macro"))
            except (ValueError, IndexError) as exc:
                logger.warning("AUC computation failed (reported as NaN): %s", exc)

        logger.info("Classification: acc=%.4f f1_macro=%.4f roc_auc=%.4f",
                    metrics["accuracy"], metrics["f1_macro"],
                    metrics["roc_auc_macro"])
        return metrics

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Compute the confusion matrix as a labelled DataFrame.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            normalize: If True, rows are normalized to sum to 1.

        Returns:
            DataFrame indexed by true class with predicted-class columns.
        """
        y_true, y_pred = self._validate_labels(y_true, y_pred)
        labels = np.arange(len(self.config.class_names))
        matrix = sk_confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums,
                               out=np.zeros_like(matrix), where=row_sums > 0)
        names = list(self.config.class_names)
        frame = pd.DataFrame(matrix, index=names, columns=names)
        frame.index.name = "true"
        frame.columns.name = "predicted"
        return frame

    def per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Compute precision, recall, F1, and support for every class.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.

        Returns:
            DataFrame with one row per class and columns ``precision``,
            ``recall``, ``f1``, ``support``.
        """
        y_true, y_pred = self._validate_labels(y_true, y_pred)
        labels = np.arange(len(self.config.class_names))
        zero_div = self.config.zero_division
        precision = precision_score(y_true, y_pred, labels=labels,
                                    average=None, zero_division=zero_div)
        recall = recall_score(y_true, y_pred, labels=labels,
                              average=None, zero_division=zero_div)
        f1 = f1_score(y_true, y_pred, labels=labels,
                      average=None, zero_division=zero_div)
        support = np.array([int(np.sum(y_true == label)) for label in labels])
        frame = pd.DataFrame({
            "precision": np.round(precision, 4),
            "recall": np.round(recall, 4),
            "f1": np.round(f1, 4),
            "support": support,
        }, index=list(self.config.class_names))
        frame.index.name = "class"
        return frame

    # -- temporal detection metrics ----------------------------------------------

    @staticmethod
    def mean_time_to_detection(
        fault_onset_times: np.ndarray,
        detection_times: np.ndarray,
    ) -> dict[str, float]:
        """Compute Mean Time To Detection over a set of fault events.

        For each fault event ``i``, the detection delay is
        ``detection_times[i] - fault_onset_times[i]``. Events that were never
        detected are encoded as ``NaN`` in ``detection_times`` and counted as
        misses (excluded from the mean).

        Args:
            fault_onset_times: Fault onset timestamps in seconds, shape ``(n,)``.
            detection_times: First-detection timestamps in seconds (NaN for
                undetected events), shape ``(n,)``.

        Returns:
            Dictionary with ``mttd_seconds``, ``median_ttd_seconds``,
            ``detected_events``, ``missed_events``, and ``detection_rate``.
        """
        onsets = np.asarray(fault_onset_times, dtype=np.float64)
        detections = np.asarray(detection_times, dtype=np.float64)
        if onsets.shape != detections.shape:
            raise ValueError(
                f"Shape mismatch: onsets {onsets.shape} vs detections {detections.shape}"
            )
        delays = detections - onsets
        valid = np.isfinite(delays) & (delays >= 0)
        detected = int(np.sum(valid))
        missed = int(delays.size - detected)

        result = {
            "mttd_seconds": float(np.mean(delays[valid])) if detected else float("nan"),
            "median_ttd_seconds": float(np.median(delays[valid])) if detected else float("nan"),
            "detected_events": detected,
            "missed_events": missed,
            "detection_rate": detected / delays.size if delays.size else 0.0,
        }
        logger.info("MTTD: %.2f s over %d/%d detected events",
                    result["mttd_seconds"], detected, delays.size)
        return result

    @staticmethod
    def fault_detection_latency(
        true_timeline: np.ndarray,
        predicted_timeline: np.ndarray,
        frame_duration_seconds: float,
    ) -> dict[str, Any]:
        """Compute per-episode detection latency from frame-level timelines.

        A fault *episode* is a maximal run of consecutive fault-positive
        frames in ``true_timeline`` (any non-zero label). Its latency is the
        time from episode onset to the first fault-positive frame in
        ``predicted_timeline`` at or after the onset and within the episode.

        Args:
            true_timeline: Frame-level ground-truth labels (0 = normal),
                shape ``(n_frames,)``.
            predicted_timeline: Frame-level predictions (0 = normal),
                shape ``(n_frames,)``.
            frame_duration_seconds: Duration of one frame in seconds.

        Returns:
            Dictionary with ``episodes`` (count), ``latencies_seconds``
            (per detected episode), ``mean_latency_seconds``,
            ``missed_episodes``, and ``detection_rate``.
        """
        true_timeline = np.asarray(true_timeline).astype(int)
        predicted_timeline = np.asarray(predicted_timeline).astype(int)
        if true_timeline.shape != predicted_timeline.shape:
            raise ValueError("Timelines must have identical shape")
        if frame_duration_seconds <= 0:
            raise ValueError("frame_duration_seconds must be positive")

        fault_active = true_timeline != 0
        boundaries = np.flatnonzero(np.diff(np.concatenate(([0], fault_active.view(np.int8), [0]))))
        episodes = list(zip(boundaries[0::2], boundaries[1::2]))

        latencies: list[float] = []
        missed = 0
        for start, end in episodes:
            hits = np.flatnonzero(predicted_timeline[start:end] != 0)
            if hits.size:
                latencies.append(float(hits[0]) * frame_duration_seconds)
            else:
                missed += 1

        result: dict[str, Any] = {
            "episodes": len(episodes),
            "latencies_seconds": [round(value, 4) for value in latencies],
            "mean_latency_seconds": float(np.mean(latencies)) if latencies else float("nan"),
            "missed_episodes": missed,
            "detection_rate": (len(latencies) / len(episodes)) if episodes else 0.0,
        }
        logger.info("Detection latency: %.2f s mean over %d/%d episodes",
                    result["mean_latency_seconds"], len(latencies), len(episodes))
        return result

    # -- statistical rigor -----------------------------------------------------------

    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_iterations: int | None = None,
    ) -> dict[str, float]:
        """Estimate a metric's confidence interval via bootstrap resampling.

        Resamples (true, predicted) pairs with replacement ``n_iterations``
        times and reports the percentile interval at
        ``config.confidence_level``.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            metric_fn: Callable mapping ``(y_true, y_pred)`` to a scalar,
                e.g. ``sklearn.metrics.accuracy_score``.
            n_iterations: Override for ``config.bootstrap_iterations``.

        Returns:
            Dictionary with ``mean``, ``lower``, ``upper``, ``std``, and
            ``confidence_level``.
        """
        y_true, y_pred = self._validate_labels(y_true, y_pred)
        n_iterations = n_iterations or self.config.bootstrap_iterations
        rng = np.random.default_rng(self.config.random_seed)
        n = len(y_true)

        scores = np.empty(n_iterations, dtype=np.float64)
        for iteration in range(n_iterations):
            indices = rng.integers(0, n, size=n)
            try:
                scores[iteration] = float(metric_fn(y_true[indices], y_pred[indices]))
            except ValueError:
                scores[iteration] = np.nan

        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            logger.error("All bootstrap iterations failed")
            return {"mean": float("nan"), "lower": float("nan"),
                    "upper": float("nan"), "std": float("nan"),
                    "confidence_level": self.config.confidence_level}

        alpha = 1.0 - self.config.confidence_level
        result = {
            "mean": float(np.mean(scores)),
            "lower": float(np.percentile(scores, 100 * alpha / 2)),
            "upper": float(np.percentile(scores, 100 * (1 - alpha / 2))),
            "std": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
            "confidence_level": self.config.confidence_level,
        }
        # Sanity check against the normal-approximation interval.
        sem = stats.sem(scores) if scores.size > 1 else 0.0
        logger.debug("Bootstrap: mean=%.4f [%.4f, %.4f] (SEM %.5f)",
                     result["mean"], result["lower"], result["upper"], sem)
        return result

    # -- dataset-level reporting -------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Produce a complete dataset-level evaluation report.

        Combines headline metrics (with bootstrap confidence intervals for
        accuracy and macro F1), the confusion matrix, and per-class metrics.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            y_proba: Optional class probability matrix.
            model_name: Identifier recorded in the report.

        Returns:
            Nested report dictionary suitable for JSON export.
        """
        metrics = self.classification_metrics(y_true, y_pred, y_proba)
        report: dict[str, Any] = {
            "model": model_name,
            "n_samples": int(len(np.asarray(y_true))),
            "metrics": {key: round(value, 4) for key, value in metrics.items()},
            "confidence_intervals": {
                "accuracy": self.bootstrap_confidence_interval(
                    y_true, y_pred, accuracy_score),
                "f1_macro": self.bootstrap_confidence_interval(
                    y_true, y_pred,
                    lambda t, p: f1_score(t, p, average="macro",
                                          zero_division=self.config.zero_division)),
            },
            "confusion_matrix": self.confusion_matrix(y_true, y_pred).to_dict(),
            "per_class": self.per_class_metrics(y_true, y_pred).to_dict(orient="index"),
        }
        logger.info("Evaluation report built for '%s' (%d samples)",
                    model_name, report["n_samples"])
        return report

    # -- export -----------------------------------------------------------------------

    @staticmethod
    def export_json(report: dict[str, Any], path: Path | str) -> bool:
        """Write an evaluation report to JSON.

        Args:
            report: Report dictionary (e.g., from :meth:`evaluate`).
            path: Destination file path.

        Returns:
            True on success, False on failure (logged).
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, default=float)
            logger.info("Report exported to %s", path)
            return True
        except (OSError, TypeError) as exc:
            logger.error("JSON export failed: %s", exc)
            return False

    def export_csv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        path: Path | str,
    ) -> bool:
        """Write per-class metrics to CSV.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            path: Destination file path.

        Returns:
            True on success, False on failure (logged).
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.per_class_metrics(y_true, y_pred).to_csv(path)
            logger.info("Per-class metrics exported to %s", path)
            return True
        except (OSError, ValueError) as exc:
            logger.error("CSV export failed: %s", exc)
            return False

    def publication_table(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fmt: str = "markdown",
    ) -> str:
        """Render a publication-ready per-class metrics table.

        Args:
            y_true: Ground-truth integer labels.
            y_pred: Predicted integer labels.
            fmt: Output format: ``markdown`` or ``latex``.

        Returns:
            Formatted table string (empty string on failure).
        """
        frame = self.per_class_metrics(y_true, y_pred)
        try:
            if fmt == "latex":
                return frame.to_latex(
                    float_format="%.3f",
                    caption="Per-class fault classification performance.",
                    label="tab:per_class_metrics",
                )
            return frame.to_markdown(floatfmt=".3f")
        except (ValueError, ImportError) as exc:
            logger.error("Table rendering failed (%s): %s", fmt, exc)
            return frame.to_string()

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _validate_labels(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce label arrays.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            Tuple of 1-D integer arrays.

        Raises:
            ValueError: On empty input or shape mismatch.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.size == 0:
            raise ValueError("Empty label arrays")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        return y_true.astype(int), y_pred.astype(int)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Demonstrate the evaluation metrics module on synthetic predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/evaluation"),
                        help="Directory for exported reports.")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of synthetic prediction samples.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point: evaluate synthetic predictions end-to-end.

    Simulates an imperfect classifier over the project's five fault classes,
    computes the full report (including MTTD and detection latency on a
    simulated timeline), exports JSON/CSV, and prints a publication table.

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

    config = MetricsConfig()
    evaluator = MetricsEvaluator(config)
    rng = np.random.default_rng(config.random_seed)
    n_classes = len(config.class_names)

    # Simulate an ~85%-accurate classifier with calibrated-ish probabilities.
    y_true = rng.integers(0, n_classes, size=args.samples)
    flip = rng.random(args.samples) < 0.15
    y_pred = np.where(flip, rng.integers(0, n_classes, size=args.samples), y_true)
    y_proba = np.full((args.samples, n_classes), 0.05)
    y_proba[np.arange(args.samples), y_pred] = 0.8
    y_proba /= y_proba.sum(axis=1, keepdims=True)

    try:
        report = evaluator.evaluate(y_true, y_pred, y_proba, model_name="demo_classifier")

        # Temporal metrics on a simulated monitoring timeline.
        onsets = np.array([100.0, 400.0, 900.0])
        detections = np.array([112.0, 431.0, np.nan])  # third fault missed
        report["mttd"] = evaluator.mean_time_to_detection(onsets, detections)

        timeline_true = np.zeros(600, dtype=int)
        timeline_true[120:180] = 2
        timeline_true[400:480] = 3
        timeline_pred = np.zeros(600, dtype=int)
        timeline_pred[131:180] = 2
        timeline_pred[407:480] = 3
        report["detection_latency"] = evaluator.fault_detection_latency(
            timeline_true, timeline_pred, frame_duration_seconds=1.0)

        evaluator.export_json(report, args.output / "evaluation_report.json")
        evaluator.export_csv(y_true, y_pred, args.output / "per_class_metrics.csv")

        logger.info("Publication table (markdown):\n%s",
                    evaluator.publication_table(y_true, y_pred))
    except ValueError as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
