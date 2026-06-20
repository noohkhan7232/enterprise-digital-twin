#!/usr/bin/env python3
"""Publication-quality benchmarking framework for acoustic fault classifiers.

This module provides :class:`ModelBenchmark`, an evaluation harness that runs
every registered **supervised** classifier in the project through an identical
protocol and produces a side-by-side comparison suitable for a research paper
or a commercial datasheet.  It orchestrates the existing components — the model
registry, :class:`~src.utils.metrics.MetricsEvaluator`, ``BaseModel`` parameter
accounting, and the :class:`~src.utils.experiment_tracker.ExperimentTracker` —
without modifying any of them.

What it measures, per model
---------------------------
* **Quality** — accuracy, macro precision / recall / F1, and ROC-AUC.
* **Efficiency** — inference latency (ms/batch), throughput (samples/s),
  parameter count, and memory footprint (parameter bytes plus a peak-activation
  estimate).

What it produces
----------------
* **Ranking tables** — models ordered by any metric, rendered as plain text.
* **A summary CSV** — one row per model per dataset, every metric a column.
* **A JSON report** — the full nested result tree, including per-fold values
  and confidence intervals, for programmatic downstream use.
* **ExperimentTracker logs** — each model's metrics and the artifacts.

Evaluation protocol
-------------------
* **Multiple datasets** — the same models are evaluated on each provided
  dataset (e.g. mel vs. CQT features, or different turbine fleets), and results
  are reported per dataset.
* **k-fold** — when ``k_folds > 1``, each model is evaluated on ``k`` folds and
  the reported metric is the across-fold mean with a confidence interval.
* **Confidence intervals** — computed with the Student-t distribution, the
  correct choice for the small sample sizes typical of k-fold (``k`` is usually
  5–10).

The anomaly autoencoder is intentionally excluded: it is unsupervised and does
not implement the classification ``predict`` contract, so it cannot be ranked
on classification metrics.  A separate anomaly-detection benchmark (ROC over
normal-vs-fault scores) is the right tool for it.

Design for testability
-----------------------
The framework accepts an **evaluable** for each model — any object exposing
``predict(x) -> labels`` and ``predict_proba(x) -> probs`` — and a **dataset**
exposing ``iter_batches() -> (features, labels)`` (or a plain list of such
tuples).  This lets the statistical machinery (metrics, ranking, CI, CSV/JSON
emission) be unit-tested with lightweight fakes and **without PyTorch**, while
the same code path drives real ``BaseModel`` instances and DataLoaders in
production.

Usage::

    from src.evaluation.model_benchmark import ModelBenchmark, BenchmarkConfig

    benchmark = ModelBenchmark(
        BenchmarkConfig(models=["acoustic_cnn", "resnet_acoustic"], k_folds=5),
    )
    report = benchmark.run(datasets={"mel": mel_loader, "cqt": cqt_loader})
    benchmark.to_csv("benchmark.csv")
    benchmark.to_json("benchmark.json")
    print(benchmark.ranking_table(metric="f1_macro"))

CLI::

    python src/evaluation/model_benchmark.py --list-models
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SKLEARN_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional PyTorch (only needed to build real models, not for the statistics)
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("model_benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registered supervised classifiers eligible for the benchmark.
SUPERVISED_MODELS: Final[tuple[str, ...]] = (
    "acoustic_cnn",
    "resnet_acoustic",
    "cnn_bilstm",
    "cnn_bilstm_attention",
    "acoustic_transformer",
)

#: Quality metrics computed for every model.
QUALITY_METRICS: Final[tuple[str, ...]] = (
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
)

#: Efficiency metrics computed for every model.
EFFICIENCY_METRICS: Final[tuple[str, ...]] = (
    "latency_ms", "throughput_sps", "n_parameters", "memory_mb",
)

#: All metric names, in CSV column order.
ALL_METRICS: Final[tuple[str, ...]] = QUALITY_METRICS + EFFICIENCY_METRICS

_BYTES_PER_FLOAT32: Final[int] = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        models: Registry names of the models to benchmark. Defaults to all
            supervised classifiers.
        num_classes: Number of fault classes.
        class_names: Optional class names for metric reporting.
        k_folds: Number of evaluation folds (1 disables k-fold aggregation).
        confidence_level: Confidence level for intervals (e.g. 0.95).
        device: Device preference for real models (``auto`` | ``cuda`` | ``cpu``).
        warmup_batches: Latency-measurement warmup batches (excluded from timing).
        timing_batches: Number of batches timed for latency / throughput.
        batch_size: Batch size assumed for throughput reporting.
        output_dir: Directory for CSV / JSON artifacts.
        random_seed: Seed for any stochastic evaluation steps.
    """

    models:           tuple[str, ...] = SUPERVISED_MODELS
    num_classes:      int = 5
    class_names:      tuple[str, ...] | None = None
    k_folds:          int = 1
    confidence_level: float = 0.95
    device:           str = "cpu"
    warmup_batches:   int = 2
    timing_batches:   int = 10
    batch_size:       int = 32
    output_dir:       Path = field(
        default_factory=lambda: _PROJECT_ROOT / "benchmark_results"
    )
    random_seed:      int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.models:
            raise ValueError("models must be a non-empty sequence")
        if self.k_folds < 1:
            raise ValueError(f"k_folds must be >= 1, got {self.k_folds}")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    """A metric's central value and (optional) confidence interval.

    Attributes:
        mean: Mean across folds (or the single value when ``k_folds == 1``).
        ci_low: Lower confidence bound (``nan`` when not computed).
        ci_high: Upper confidence bound (``nan`` when not computed).
        std: Standard deviation across folds.
        values: The per-fold values.
    """

    mean:    float
    ci_low:  float = float("nan")
    ci_high: float = float("nan")
    std:     float = 0.0
    values:  list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "std": self.std, "values": list(self.values),
        }


@dataclass
class ModelBenchmarkResult:
    """All benchmark metrics for one model on one dataset.

    Attributes:
        model_name: Registry name of the model.
        dataset_name: Name of the dataset evaluated.
        metrics: Mapping of metric name to :class:`MetricSummary`.
        n_samples: Number of samples evaluated.
        error: Error message when the model failed to evaluate, else ``None``.
    """

    model_name:   str
    dataset_name: str
    metrics:      dict[str, MetricSummary] = field(default_factory=dict)
    n_samples:    int = 0
    error:        str | None = None

    def value(self, metric: str) -> float:
        """Return the mean value of a metric (``nan`` when absent).

        Args:
            metric: Metric name.

        Returns:
            The metric's mean value.
        """
        summary = self.metrics.get(metric)
        return summary.mean if summary is not None else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "error": self.error,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute the mean and a Student-t confidence interval.

    The t-distribution is used because k-fold sample sizes are small; for a
    single value the interval is undefined (``nan`` bounds).

    Args:
        values: The per-fold metric values.
        confidence: Confidence level in ``(0, 1)``.

    Returns:
        Tuple ``(mean, ci_low, ci_high)``.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan"), float("nan")
    if _SCIPY_AVAILABLE:
        sem = _scipy_stats.sem(arr)
        half = sem * _scipy_stats.t.ppf((1 + confidence) / 2.0, arr.size - 1)
    else:  # pragma: no cover - normal approximation fallback
        sem = arr.std(ddof=1) / np.sqrt(arr.size)
        half = sem * 1.96
    return mean, float(mean - half), float(mean + half)


def _summarise(values: Sequence[float], confidence: float) -> MetricSummary:
    """Build a :class:`MetricSummary` from per-fold values.

    Args:
        values: Per-fold metric values.
        confidence: Confidence level for the interval.

    Returns:
        A populated :class:`MetricSummary`.
    """
    arr = np.asarray(list(values), dtype=float)
    valid = arr[~np.isnan(arr)]
    mean, lo, hi = compute_confidence_interval(values, confidence)
    std = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
    return MetricSummary(mean=mean, ci_low=lo, ci_high=hi, std=std,
                         values=[float(v) for v in arr])


# ---------------------------------------------------------------------------
# Metric computation (self-contained, sklearn-backed)
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    y_proba: "np.ndarray | None",
    num_classes: int,
) -> dict[str, float]:
    """Compute the quality metric suite for one evaluation.

    Uses scikit-learn directly so the benchmark is self-contained and does not
    depend on the exact signature of any project metrics class.  Degenerate
    cases (missing classes, no probabilities) yield ``nan`` for the affected
    metric rather than raising.

    Args:
        y_true: Ground-truth labels ``(N,)``.
        y_pred: Predicted labels ``(N,)``.
        y_proba: Class probabilities ``(N, num_classes)`` or ``None``.
        num_classes: Total number of classes.

    Returns:
        Dictionary with accuracy, precision, recall, F1, and ROC-AUC.
    """
    out: dict[str, float] = {m: float("nan") for m in QUALITY_METRICS}
    if y_true.size == 0 or not _SKLEARN_AVAILABLE:
        return out

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    out["precision_macro"] = float(prec)
    out["recall_macro"] = float(rec)
    out["f1_macro"] = float(f1)

    if y_proba is not None and y_proba.ndim == 2:
        try:
            n_present = len(np.unique(y_true))
            if n_present >= 2:
                if y_proba.shape[1] == 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    out["roc_auc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro",
                        labels=list(range(y_proba.shape[1])),
                    ))
        except (ValueError, IndexError) as exc:
            logger.debug("ROC-AUC unavailable: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Evaluable / dataset protocols (duck-typed)
# ---------------------------------------------------------------------------


def _iter_batches(dataset: Any) -> Iterable[tuple[Any, Any]]:
    """Yield ``(features, labels)`` batches from a dataset-like object.

    Accepts an object with an ``iter_batches()`` method, a DataLoader-style
    iterable of ``(features, labels)`` tuples, or a plain list of such tuples.

    Args:
        dataset: The dataset-like object.

    Yields:
        ``(features, labels)`` batches.
    """
    if hasattr(dataset, "iter_batches"):
        yield from dataset.iter_batches()
    else:
        yield from dataset


def _to_numpy(x: Any) -> "np.ndarray":
    """Convert a tensor / array / sequence to a NumPy array.

    Args:
        x: A torch tensor, NumPy array, or sequence.

    Returns:
        A NumPy array.
    """
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# ModelBenchmark
# ---------------------------------------------------------------------------


class ModelBenchmark:
    """Benchmark harness comparing supervised classifiers.

    Args:
        config: The benchmark configuration.
        experiment_tracker: Optional tracker for logging metrics / artifacts.
        model_factory: Optional callable ``(name) -> evaluable`` used to build
            a model from its registry name.  When ``None``, a real
            :class:`BaseModel` is constructed via the registry (requires
            PyTorch).  Supplying a factory lets tests inject fakes.
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        experiment_tracker: Any = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.tracker = experiment_tracker
        self.model_factory = model_factory
        self.results: list[ModelBenchmarkResult] = []
        logger.info(
            "ModelBenchmark ready | models=%d | k_folds=%d | device=%s",
            len(self.config.models), self.config.k_folds, self.config.device,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, name: str) -> Any:
        """Construct an evaluable model by registry name.

        Args:
            name: Registry name of the model.

        Returns:
            An evaluable exposing ``predict`` / ``predict_proba``.

        Raises:
            RuntimeError: When no factory is given and PyTorch is unavailable.
        """
        if self.model_factory is not None:
            return self.model_factory(name)
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "Building real models requires PyTorch; pass model_factory "
                "to benchmark with fakes."
            )
        from src.models.base_model import ModelConfig, build_model

        cfg = ModelConfig(
            model_name=name,
            num_classes=self.config.num_classes,
            class_names=(list(self.config.class_names)
                         if self.config.class_names else None),
        )
        model = build_model(name, cfg)
        model.to_device(self.config.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Efficiency measurement
    # ------------------------------------------------------------------

    def _measure_efficiency(
        self, model: Any, batches: Sequence[tuple[Any, Any]]
    ) -> dict[str, float]:
        """Measure latency, throughput, parameters, and memory for a model.

        Args:
            model: The evaluable model.
            batches: A small list of ``(features, labels)`` batches for timing.

        Returns:
            Dictionary of efficiency metrics.
        """
        out: dict[str, float] = {m: float("nan") for m in EFFICIENCY_METRICS}

        # Parameter count + parameter memory (works on real BaseModels).
        n_params = float("nan")
        if hasattr(model, "count_parameters"):
            try:
                n_params = float(model.count_parameters().total)
            except Exception as exc:  # noqa: BLE001
                logger.debug("count_parameters failed: %s", exc)
        out["n_parameters"] = n_params
        if not np.isnan(n_params):
            out["memory_mb"] = n_params * _BYTES_PER_FLOAT32 / (1024 ** 2)

        # Latency / throughput: warm up, then time a few batches.
        if not batches:
            return out
        timing = batches[: self.config.timing_batches] or batches
        n_warm = min(self.config.warmup_batches, len(timing))
        try:
            for features, _ in timing[:n_warm]:
                model.predict(features)
            durations: list[float] = []
            sample_counts: list[int] = []
            for features, _ in timing:
                t0 = time.perf_counter()
                model.predict(features)
                durations.append(time.perf_counter() - t0)
                sample_counts.append(len(_to_numpy(features)))
            mean_dur = float(np.mean(durations)) if durations else float("nan")
            mean_n = float(np.mean(sample_counts)) if sample_counts else 0.0
            out["latency_ms"] = mean_dur * 1000.0
            out["throughput_sps"] = (mean_n / mean_dur) if mean_dur > 0 else float("nan")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Latency measurement failed: %s", exc)
        return out

    # ------------------------------------------------------------------
    # Single-fold quality evaluation
    # ------------------------------------------------------------------

    def _evaluate_quality(
        self, model: Any, batches: Iterable[tuple[Any, Any]]
    ) -> tuple[dict[str, float], int]:
        """Run a model over batches and compute quality metrics.

        Args:
            model: The evaluable model.
            batches: An iterable of ``(features, labels)`` batches.

        Returns:
            Tuple ``(metrics, n_samples)``.
        """
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        for features, labels in batches:
            preds = _to_numpy(model.predict(features)).reshape(-1)
            all_pred.append(preds)
            all_true.append(_to_numpy(labels).reshape(-1))
            if hasattr(model, "predict_proba"):
                try:
                    all_proba.append(_to_numpy(model.predict_proba(features)))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("predict_proba failed: %s", exc)

        if not all_true:
            return {m: float("nan") for m in QUALITY_METRICS}, 0

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_proba = (
            np.concatenate(all_proba)
            if all_proba and len(all_proba) == len(all_pred) else None
        )
        metrics = compute_quality_metrics(
            y_true, y_pred, y_proba, self.config.num_classes
        )
        return metrics, int(y_true.size)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self, datasets: dict[str, Any]
    ) -> list[ModelBenchmarkResult]:
        """Benchmark every configured model on every dataset.

        For each ``(model, dataset)`` pair, quality metrics are computed over
        ``k_folds`` folds (the dataset is materialised into batches and split
        into contiguous fold groups) and aggregated with confidence intervals;
        efficiency metrics are measured once.

        Args:
            datasets: Mapping of dataset name to a dataset-like object.

        Returns:
            The list of :class:`ModelBenchmarkResult`, one per model per dataset.
        """
        self.results = []
        for dataset_name, dataset in datasets.items():
            batches = list(_iter_batches(dataset))
            logger.info("Dataset '%s': %d batches", dataset_name, len(batches))
            for model_name in self.config.models:
                result = self._benchmark_one(model_name, dataset_name, batches)
                self.results.append(result)
                self._log_result(result)
        logger.info("Benchmark complete: %d results", len(self.results))
        return self.results

    def _benchmark_one(
        self, model_name: str, dataset_name: str,
        batches: list[tuple[Any, Any]],
    ) -> ModelBenchmarkResult:
        """Benchmark a single model on a single dataset.

        Args:
            model_name: Registry name of the model.
            dataset_name: Name of the dataset.
            batches: Materialised ``(features, labels)`` batches.

        Returns:
            The :class:`ModelBenchmarkResult`.
        """
        result = ModelBenchmarkResult(model_name=model_name, dataset_name=dataset_name)
        try:
            model = self._build_model(model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build '%s': %s", model_name, exc)
            result.error = f"build failed: {exc}"
            return result

        # k-fold quality
        folds = self._make_folds(batches, self.config.k_folds)
        per_metric: dict[str, list[float]] = {m: [] for m in QUALITY_METRICS}
        total_samples = 0
        try:
            for fold_batches in folds:
                metrics, n = self._evaluate_quality(model, fold_batches)
                total_samples += n
                for m in QUALITY_METRICS:
                    per_metric[m].append(metrics[m])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation of '%s' failed: %s", model_name, exc)
            result.error = f"evaluation failed: {exc}"
            return result

        for m in QUALITY_METRICS:
            result.metrics[m] = _summarise(per_metric[m], self.config.confidence_level)

        # efficiency (single measurement)
        eff = self._measure_efficiency(model, batches)
        for m in EFFICIENCY_METRICS:
            result.metrics[m] = MetricSummary(mean=eff[m], values=[eff[m]])

        result.n_samples = total_samples
        return result

    @staticmethod
    def _make_folds(
        batches: list[tuple[Any, Any]], k: int
    ) -> list[list[tuple[Any, Any]]]:
        """Partition batches into ``k`` contiguous folds.

        Args:
            batches: The materialised batches.
            k: Number of folds.

        Returns:
            A list of ``k`` batch groups (fewer if there are too few batches).
        """
        if k <= 1 or len(batches) < 2:
            return [batches]
        k = min(k, len(batches))
        fold_size = int(np.ceil(len(batches) / k))
        return [
            batches[i:i + fold_size]
            for i in range(0, len(batches), fold_size)
        ]

    # ------------------------------------------------------------------
    # Ranking / reporting
    # ------------------------------------------------------------------

    def ranking(
        self, metric: str = "f1_macro", *, dataset: str | None = None,
        descending: bool = True,
    ) -> list[tuple[str, float]]:
        """Rank models by a metric.

        Args:
            metric: Metric to rank by.
            dataset: Restrict to one dataset (``None`` averages across datasets).
            descending: Sort highest-first (set ``False`` for latency, etc.).

        Returns:
            A list of ``(model_name, value)`` pairs in ranked order.
        """
        agg: dict[str, list[float]] = {}
        for r in self.results:
            if dataset is not None and r.dataset_name != dataset:
                continue
            agg.setdefault(r.model_name, []).append(r.value(metric))
        ranked = [
            (name, float(np.nanmean(vals))) for name, vals in agg.items()
        ]
        ranked.sort(key=lambda kv: (np.isnan(kv[1]), kv[1]),
                    reverse=descending)
        # Push NaNs to the end regardless of sort direction.
        non_nan = [x for x in ranked if not np.isnan(x[1])]
        nan = [x for x in ranked if np.isnan(x[1])]
        non_nan.sort(key=lambda kv: kv[1], reverse=descending)
        return non_nan + nan

    def ranking_table(
        self, metric: str = "f1_macro", *, dataset: str | None = None,
    ) -> str:
        """Render a ranking as a plain-text table.

        Args:
            metric: Metric to rank by.
            dataset: Restrict to one dataset.

        Returns:
            A formatted table string.
        """
        descending = metric not in ("latency_ms", "memory_mb", "n_parameters")
        rows = self.ranking(metric, dataset=dataset, descending=descending)
        lines = [
            f"Ranking by {metric}"
            + (f" on '{dataset}'" if dataset else " (mean across datasets)"),
            f"{'Rank':<5}{'Model':<26}{metric:>14}",
            "-" * 45,
        ]
        for i, (name, val) in enumerate(rows, 1):
            val_str = "n/a" if np.isnan(val) else f"{val:.4f}"
            lines.append(f"{i:<5}{name:<26}{val_str:>14}")
        return "\n".join(lines)

    def summary_table(self) -> str:
        """Render a full metric table (all models × all metrics).

        Returns:
            A formatted table string.
        """
        header = f"{'Model':<24}{'Dataset':<12}" + "".join(
            f"{m[:11]:>13}" for m in ALL_METRICS
        )
        lines = [header, "-" * len(header)]
        for r in self.results:
            row = f"{r.model_name:<24}{r.dataset_name:<12}"
            for m in ALL_METRICS:
                v = r.value(m)
                row += f"{'n/a' if np.isnan(v) else f'{v:.3f}':>13}"
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path | None = None) -> Path:
        """Write the benchmark summary to a CSV file.

        One row per model per dataset; one column per metric (mean), plus the
        confidence bounds for quality metrics.

        Args:
            path: Destination CSV path (defaults to ``output_dir/benchmark.csv``).

        Returns:
            The resolved :class:`Path` written.
        """
        path = Path(path) if path else self.config.output_dir / "benchmark.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["model_name", "dataset_name", "n_samples"]
        for m in ALL_METRICS:
            fieldnames.append(m)
            if m in QUALITY_METRICS:
                fieldnames += [f"{m}_ci_low", f"{m}_ci_high"]

        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                row: dict[str, Any] = {
                    "model_name": r.model_name,
                    "dataset_name": r.dataset_name,
                    "n_samples": r.n_samples,
                }
                for m in ALL_METRICS:
                    summary = r.metrics.get(m)
                    row[m] = "" if summary is None else _fmt(summary.mean)
                    if m in QUALITY_METRICS and summary is not None:
                        row[f"{m}_ci_low"] = _fmt(summary.ci_low)
                        row[f"{m}_ci_high"] = _fmt(summary.ci_high)
                writer.writerow(row)

        logger.info("Wrote benchmark CSV -> %s", path)
        self._log_artifact(path, "benchmark_csv")
        return path

    def to_json(self, path: str | Path | None = None) -> Path:
        """Write the full benchmark report to a JSON file.

        Args:
            path: Destination JSON path (defaults to ``output_dir/benchmark.json``).

        Returns:
            The resolved :class:`Path` written.
        """
        path = Path(path) if path else self.config.output_dir / "benchmark.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "config": _config_to_dict(self.config),
            "results": [r.to_dict() for r in self.results],
            "rankings": {
                m: self.ranking(
                    m, descending=(m not in ("latency_ms", "memory_mb", "n_parameters"))
                )
                for m in ALL_METRICS
            },
        }
        with path.open("w") as fh:
            json.dump(report, fh, indent=2, default=_json_default)

        logger.info("Wrote benchmark JSON -> %s", path)
        self._log_artifact(path, "benchmark_json")
        return path

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log_result(self, result: ModelBenchmarkResult) -> None:
        """Log a single result's metrics to the tracker (failure-safe).

        Args:
            result: The result to log.
        """
        if self.tracker is None:
            return
        try:
            flat = {
                f"{result.dataset_name}/{result.model_name}/{m}": s.mean
                for m, s in result.metrics.items()
            }
            self.tracker.log_metrics(flat)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)

    def _log_artifact(self, path: Path, kind: str) -> None:
        """Log an artifact to the tracker (failure-safe).

        Args:
            path: Artifact path.
            kind: Artifact type label.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_artifact(str(path), description=kind, artifact_type=kind)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_artifact failed: %s", exc)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a float for CSV, leaving NaN cells blank.

    Args:
        value: The value to format.

    Returns:
        A string ("" for NaN).
    """
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.6f}"


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for NumPy scalars and paths.

    Args:
        obj: The object to encode.

    Returns:
        A JSON-serialisable representation.

    Raises:
        TypeError: When the object cannot be serialised.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _config_to_dict(config: BenchmarkConfig) -> dict[str, Any]:
    """Serialise a :class:`BenchmarkConfig` to a JSON-friendly dict.

    Args:
        config: The configuration.

    Returns:
        Dictionary representation with the output path stringified.
    """
    d = asdict(config)
    d["output_dir"] = str(config.output_dir)
    d["models"] = list(config.models)
    if config.class_names is not None:
        d["class_names"] = list(config.class_names)
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Model benchmark")
    parser.add_argument("--list-models", action="store_true",
                        help="List the supervised models eligible for benchmarking.")
    args = parser.parse_args(argv)

    if args.list_models:
        print("Supervised models eligible for benchmarking:")
        for name in SUPERVISED_MODELS:
            print(f"  • {name}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())