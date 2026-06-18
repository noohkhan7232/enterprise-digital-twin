#!/usr/bin/env python3
"""Production-grade experiment tracking for Wind Turbine Acoustic Monitoring.

Provides local experiment tracking with optional MLflow integration.  Every
experiment run is stored as a structured :class:`RunRecord`, persisted to
``data/processed/experiments/<experiment_name>/<run_id>/``, and rolled into
project-wide CSV and JSON summary files so the full research audit trail is
always one ``grep`` away.

Design principles
-----------------
* **MLflow-first, local-fallback:** if ``mlflow`` is importable the tracker
  creates a real MLflow run and mirrors every log call; if not, all data land
  in a plain JSON/CSV store with an identical public API.
* **Thread-safe:** a per-instance ``threading.Lock`` protects all shared state
  so the tracker can be used safely from parallel training workers.
* **Context-manager support:** ``with ExperimentTracker(...) as tracker:``
  guarantees ``end_run()`` is called even if training raises.
* **Reproducibility-first:** every run captures git commit, Python version,
  platform info, and a SHA-256 prefix for each logged artifact.
* **Zero mandatory dependencies beyond stdlib + numpy/pandas/matplotlib.**

Usage::

    tracker = ExperimentTracker()
    with tracker.start_run(run_name="baseline_cnn") as run:
        tracker.log_params({"n_mels": 128, "n_mfcc": 40})
        tracker.log_metrics({"accuracy": 0.91, "f1_macro": 0.89})
        tracker.log_figure(fig, "confusion_matrix.png")

    tracker.compare_experiments()

CLI::

    python src/utils/experiment_tracker.py --demo
    python src/utils/experiment_tracker.py --compare --experiment wind-turbine-acoustics
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Generator, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------
try:
    import mlflow
    import mlflow.tracking

    _MLFLOW_AVAILABLE: bool = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Optional PyYAML import
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml

    _YAML_AVAILABLE: bool = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE: bool = False

logger = logging.getLogger("experiment_tracker")

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent  # src/utils/ -> root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default MLflow experiment name matching config.yaml.
DEFAULT_EXPERIMENT_NAME: Final[str] = "wind-turbine-acoustics"

#: Default MLflow tracking URI matching config.yaml.
DEFAULT_TRACKING_URI: Final[str] = "mlruns"

#: Root directory for local experiment storage.
DEFAULT_EXPERIMENTS_DIR: Final[Path] = (
    _PROJECT_ROOT / "data" / "processed" / "experiments"
)

#: Metric keys used for composite ranking.
RANKING_METRIC_WEIGHTS: Final[dict[str, float]] = {
    "f1_macro": 0.40,
    "accuracy": 0.30,
    "roc_auc_macro": 0.30,
}

#: Recognised artifact MIME types for validation.
SUPPORTED_ARTIFACT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".svg", ".pdf", ".csv", ".json", ".yaml", ".yml",
     ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".txt", ".md"}
)

#: Status values for a :class:`RunRecord`.
RUN_STATUS_CREATED: Final[str] = "created"
RUN_STATUS_RUNNING: Final[str] = "running"
RUN_STATUS_FINISHED: Final[str] = "finished"
RUN_STATUS_FAILED: Final[str] = "failed"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Complete record for a single experiment run.

    Attributes:
        run_id: Unique identifier, e.g. ``run_1718000000000_ab12cd34``.
        experiment_name: Parent experiment name (maps to an MLflow experiment).
        run_name: Optional human-friendly name for the run.
        status: One of ``created``, ``running``, ``finished``, ``failed``.
        start_time: Unix timestamp of run start.
        end_time: Unix timestamp of run end (0.0 if still running).
        duration_seconds: Wall-clock duration; 0.0 while running.
        params: Arbitrary key-value hyperparameter dictionary.
        metrics: Metric name → scalar value.
        metric_history: Metric name → list of (step, value) tuples for
            per-epoch tracking.
        artifacts: List of artifact descriptors (path, hash, description).
        tags: Arbitrary string key-value tags.
        git_commit: Short SHA of HEAD at run start.
        python_version: Python interpreter version string.
        platform_info: OS / hardware summary string.
        notes: Free-text researcher notes.
        composite_score: Ranking score computed by
            :meth:`ExperimentTracker._rank_run`.
    """

    run_id: str = ""
    experiment_name: str = ""
    run_name: str = ""
    status: str = RUN_STATUS_CREATED
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metric_history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    python_version: str = ""
    platform_info: str = ""
    notes: str = ""
    composite_score: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for :class:`ExperimentTracker`.

    Attributes:
        experiment_name: MLflow experiment name; also the local subdirectory.
        tracking_uri: MLflow tracking URI (``mlruns`` for local file store).
        experiments_dir: Root directory for local JSON/CSV storage.
        figures_dir: Directory where tracked figures are copied.
        use_mlflow: Attempt to use MLflow if installed.
        auto_log_system_info: Capture git / Python / OS info on run start.
        figure_dpi: DPI for comparison figure exports.
        ranking_metric_weights: Metric → weight dict for composite ranking.
    """

    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    tracking_uri: str = DEFAULT_TRACKING_URI
    experiments_dir: Path = field(default_factory=lambda: DEFAULT_EXPERIMENTS_DIR)
    figures_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "docs" / "figures"
    )
    use_mlflow: bool = True
    auto_log_system_info: bool = True
    figure_dpi: int = 300
    ranking_metric_weights: dict[str, float] = field(
        default_factory=lambda: dict(RANKING_METRIC_WEIGHTS)
    )


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Production-grade experiment tracker with MLflow and local-JSON backends.

    All public methods are thread-safe and idempotent.  Failures in logging
    helpers are logged at WARNING level and never propagate to the caller so
    a broken tracker never aborts a training run.

    Args:
        config: Tracker configuration.  Defaults to :class:`ExperimentConfig`.

    Example::

        config = ExperimentConfig(experiment_name="cnn_ablation")
        tracker = ExperimentTracker(config)

        with tracker.start_run("baseline") as run:
            tracker.log_params({"n_mels": 128, "n_mfcc": 40})
            tracker.log_metrics({"accuracy": 0.91, "f1_macro": 0.89})

        tracker.compare_experiments()
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()
        self._lock = threading.Lock()
        self._active_run: RunRecord | None = None
        self._mlflow_run = None  # mlflow.ActiveRun or None
        self._mlflow_enabled: bool = False
        self._all_runs: list[RunRecord] = []

        self._setup_directories()
        self._setup_mlflow()
        self._load_existing_runs()

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: str = "",
        tags: dict[str, str] | None = None,
        notes: str = "",
    ) -> Generator["ExperimentTracker", None, None]:
        """Start a new experiment run as a context manager.

        On exit the run is ended automatically; if the block raises an
        exception the run is marked *failed* and the exception re-raised.

        Args:
            run_name: Human-friendly run name.
            tags: Optional string key-value tags attached to the run.
            notes: Free-text researcher notes.

        Yields:
            This :class:`ExperimentTracker` instance (for fluent use).

        Raises:
            RuntimeError: If a run is already active on this tracker instance.
        """
        self._begin_run(run_name=run_name, tags=tags or {}, notes=notes)
        try:
            yield self
            self._finish_run(status=RUN_STATUS_FINISHED)
        except Exception:
            self._finish_run(status=RUN_STATUS_FAILED)
            raise

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def _begin_run(
        self,
        run_name: str,
        tags: dict[str, str],
        notes: str,
    ) -> RunRecord:
        """Internal: create and register a new :class:`RunRecord`.

        Args:
            run_name: Human-friendly name.
            tags: Initial tags dict.
            notes: Researcher notes.

        Returns:
            The newly created :class:`RunRecord`.

        Raises:
            RuntimeError: If a run is already active.
        """
        with self._lock:
            if self._active_run is not None:
                raise RuntimeError(
                    f"Run '{self._active_run.run_id}' is already active. "
                    "Call end_run() before starting a new one."
                )
            run_id = _make_run_id()
            run = RunRecord(
                run_id=run_id,
                experiment_name=self.config.experiment_name,
                run_name=run_name or run_id,
                status=RUN_STATUS_RUNNING,
                start_time=time.time(),
                tags=dict(tags),
                notes=notes,
            )
            if self.config.auto_log_system_info:
                run.git_commit = _get_git_commit()
                run.python_version = sys.version.split()[0]
                run.platform_info = _get_platform_info()

            self._active_run = run

        if self._mlflow_enabled:
            try:
                self._mlflow_run = mlflow.start_run(
                    run_name=run.run_name,
                    tags={**tags, "run_id": run_id},
                )
            except Exception as exc:
                logger.warning("MLflow start_run failed: %s", exc)

        logger.info(
            "Run started  | id=%s | name=%s | experiment=%s",
            run.run_id, run.run_name, run.experiment_name,
        )
        return run

    def _finish_run(self, status: str) -> None:
        """Internal: finalise the active run and persist it to disk.

        Args:
            status: Final status string.
        """
        with self._lock:
            if self._active_run is None:
                return
            run = self._active_run
            run.status = status
            run.end_time = time.time()
            run.duration_seconds = round(run.end_time - run.start_time, 3)
            run.composite_score = self._rank_run(run)
            self._active_run = None
            self._all_runs.append(run)

        self._persist_run(run)
        self._update_summary_files(run)

        if self._mlflow_enabled and self._mlflow_run is not None:
            try:
                mlflow.end_run(
                    status="FINISHED" if status == RUN_STATUS_FINISHED else "FAILED"
                )
                self._mlflow_run = None
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)

        logger.info(
            "Run %s  | id=%s | duration=%.1fs | composite=%.4f",
            status, run.run_id, run.duration_seconds, run.composite_score,
        )

    def end_run(self, failed: bool = False) -> None:
        """End the active run explicitly (alternative to the context manager).

        Args:
            failed: If True, mark the run as *failed* instead of *finished*.
        """
        status = RUN_STATUS_FAILED if failed else RUN_STATUS_FINISHED
        self._finish_run(status)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the active run.

        Covers all tracked parameters specified in the project:
        MFCC count, mel bins, CQT settings, denoising method,
        dataset version, and random seed.

        Args:
            params: Dictionary mapping parameter names to values.
                Nested dicts are flattened with ``__`` as separator.
        """
        flat = _flatten_dict(params)
        with self._lock:
            if self._active_run is None:
                logger.warning("log_params called with no active run; skipping")
                return
            self._active_run.params.update(flat)

        if self._mlflow_enabled:
            try:
                mlflow.log_params(
                    {k: str(v) for k, v in flat.items()}
                )
            except Exception as exc:
                logger.warning("MLflow log_params failed: %s", exc)

        logger.debug("Params logged: %s", list(flat.keys()))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log scalar metrics for the active run.

        Tracked metrics include accuracy, precision, recall, F1, ROC AUC,
        training time, and inference time.

        Args:
            metrics: Mapping of metric name to scalar value.
            step: Optional training step / epoch number for per-epoch
                history tracking.
        """
        with self._lock:
            if self._active_run is None:
                logger.warning("log_metrics called with no active run; skipping")
                return
            run = self._active_run
            for name, value in metrics.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    logger.warning("Metric '%s' is not numeric (%s); skipping",
                                   name, type(value).__name__)
                    continue
                run.metrics[name] = scalar
                if step is not None:
                    if name not in run.metric_history:
                        run.metric_history[name] = []
                    run.metric_history[name].append((step, scalar))

        if self._mlflow_enabled:
            try:
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items()
                     if _is_finite_scalar(v)},
                    step=step,
                )
            except Exception as exc:
                logger.warning("MLflow log_metrics failed: %s", exc)

        logger.debug("Metrics logged (step=%s): %s", step, list(metrics.keys()))

    def log_artifact(
        self,
        path: Path | str,
        description: str = "",
        artifact_type: str = "file",
    ) -> None:
        """Track a file artifact (PNG, CSV, JSON, YAML, model checkpoint).

        The file is copied into the run's artifact directory and registered
        with its SHA-256 prefix for integrity checking.

        Args:
            path: Source file path.
            description: Human-readable description of the artifact.
            artifact_type: Category label (e.g., ``figure``, ``model``,
                ``report``).
        """
        path = Path(path)
        if not path.is_file():
            logger.warning("Artifact not found: %s", path)
            return

        ext = path.suffix.lower()
        if ext not in SUPPORTED_ARTIFACT_EXTENSIONS:
            logger.warning("Unsupported artifact extension '%s'; logging anyway", ext)

        file_hash = _hash_file(path)

        with self._lock:
            if self._active_run is None:
                logger.warning("log_artifact called with no active run; skipping")
                return
            run = self._active_run
            dest_dir = self._run_artifacts_dir(run)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            try:
                shutil.copy2(path, dest)
            except OSError as exc:
                logger.error("Failed to copy artifact %s: %s", path, exc)
                return

            run.artifacts.append({
                "source": str(path),
                "dest": str(dest),
                "description": description,
                "type": artifact_type,
                "sha256_prefix": file_hash,
                "size_bytes": str(path.stat().st_size),
            })

        if self._mlflow_enabled:
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:
                logger.warning("MLflow log_artifact failed: %s", exc)

        logger.debug("Artifact logged: %s (%s, sha256:%s)",
                     path.name, artifact_type, file_hash)

    def log_figure(
        self,
        fig: plt.Figure,
        filename: str,
        description: str = "",
        close_after: bool = True,
    ) -> Path | None:
        """Save a matplotlib figure and register it as an artifact.

        Args:
            fig: Matplotlib figure to save.
            filename: Output filename (with extension, e.g. ``roc.png``).
            description: Human-readable description for the artifact registry.
            close_after: Whether to close *fig* after saving.

        Returns:
            Path of the saved figure file, or None on failure.
        """
        with self._lock:
            if self._active_run is None:
                logger.warning("log_figure called with no active run; skipping")
                return None
            run = self._active_run
            dest_dir = self._run_artifacts_dir(run)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename

        try:
            fig.savefig(dest, dpi=self.config.figure_dpi, bbox_inches="tight")
            logger.debug("Figure saved: %s", dest)
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to save figure %s: %s", filename, exc)
            return None
        finally:
            if close_after:
                plt.close(fig)

        # Register artifact directly — the file is already in the run
        # artifacts directory, so copying via log_artifact would fail.
        file_hash = _hash_file(dest)
        with self._lock:
            if self._active_run is not None:
                self._active_run.artifacts.append({
                    "source": str(dest),
                    "dest": str(dest),
                    "description": description,
                    "type": "figure",
                    "sha256_prefix": file_hash,
                    "size_bytes": str(dest.stat().st_size),
                })
        if self._mlflow_enabled:
            try:
                mlflow.log_artifact(str(dest))
            except Exception as exc:
                logger.warning("MLflow log_artifact (figure) failed: %s", exc)
        return dest

    def log_dataset_info(
        self,
        total_clips: int,
        clips_per_class: dict[str, int],
        total_duration_s: float,
        dataset_version: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log dataset provenance and inventory.

        Args:
            total_clips: Total number of clips in the dataset.
            clips_per_class: Mapping of fault-type name to clip count.
            total_duration_s: Total audio duration in seconds.
            dataset_version: Optional version string or git tag for the data.
            extra: Additional key-value metadata.
        """
        info: dict[str, Any] = {
            "dataset.total_clips": total_clips,
            "dataset.total_duration_s": round(total_duration_s, 3),
            "dataset.total_duration_min": round(total_duration_s / 60, 2),
            "dataset.n_classes": len(clips_per_class),
            "dataset.version": dataset_version,
        }
        for ft, cnt in clips_per_class.items():
            info[f"dataset.clips_{ft}"] = cnt
        if extra:
            info.update({f"dataset.{k}": v for k, v in extra.items()})

        self.log_params(info)
        logger.info(
            "Dataset info logged: %d clips, %d classes, %.1f min",
            total_clips, len(clips_per_class), total_duration_s / 60,
        )

    def log_model_info(
        self,
        model_name: str,
        n_parameters: int,
        architecture: str = "",
        framework: str = "PyTorch",
        checkpoint_path: Path | str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log model architecture metadata.

        Args:
            model_name: Short model identifier (e.g., ``CNN_Mel128``).
            n_parameters: Total trainable parameter count.
            architecture: Description of the architecture layers.
            framework: Deep learning framework (default ``PyTorch``).
            checkpoint_path: Path to the saved model checkpoint, if any.
            extra: Additional key-value metadata.
        """
        info: dict[str, Any] = {
            "model.name": model_name,
            "model.n_parameters": n_parameters,
            "model.architecture": architecture,
            "model.framework": framework,
        }
        if extra:
            info.update({f"model.{k}": v for k, v in extra.items()})
        self.log_params(info)

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.is_file():
                self.log_artifact(ckpt, description="model checkpoint",
                                  artifact_type="model")

        logger.info(
            "Model info logged: %s | %s params | %s",
            model_name, f"{n_parameters:,}", framework,
        )

    def log_feature_config(
        self,
        n_mfcc: int | None = None,
        n_mels: int | None = None,
        n_fft: int | None = None,
        hop_length: int | None = None,
        cqt_bins: int | None = None,
        cqt_bins_per_octave: int | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        denoising_method: str | None = None,
        dataset_version: str | None = None,
        random_seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log the complete feature extraction configuration.

        Maps directly to ``config.yaml: feature_extraction`` fields.
        All arguments are optional — only non-None values are logged.

        Args:
            n_mfcc: Number of MFCC coefficients.
            n_mels: Number of mel filterbank bands.
            n_fft: FFT window size.
            hop_length: STFT hop length in samples.
            cqt_bins: Total CQT frequency bins.
            cqt_bins_per_octave: CQT frequency resolution per octave.
            fmin: Lowest analysis frequency in Hz.
            fmax: Highest mel analysis frequency in Hz.
            denoising_method: Algorithm name from ``SUPPORTED_METHODS``.
            dataset_version: Dataset identifier / version tag.
            random_seed: Global RNG seed for reproducibility.
            extra: Additional feature parameters.
        """
        params: dict[str, Any] = {}
        local_vars = {
            "feature.n_mfcc": n_mfcc,
            "feature.n_mels": n_mels,
            "feature.n_fft": n_fft,
            "feature.hop_length": hop_length,
            "feature.cqt_bins": cqt_bins,
            "feature.cqt_bins_per_octave": cqt_bins_per_octave,
            "feature.fmin": fmin,
            "feature.fmax": fmax,
            "feature.denoising_method": denoising_method,
            "feature.dataset_version": dataset_version,
            "feature.random_seed": random_seed,
        }
        for key, val in local_vars.items():
            if val is not None:
                params[key] = val
        if extra:
            params.update({f"feature.{k}": v for k, v in extra.items()})
        if params:
            self.log_params(params)
        logger.debug("Feature config logged: %s", list(params.keys()))

    def log_tag(self, key: str, value: str) -> None:
        """Attach a string tag to the active run.

        Args:
            key: Tag key.
            value: Tag value (must be a string).
        """
        with self._lock:
            if self._active_run is None:
                return
            self._active_run.tags[key] = str(value)

        if self._mlflow_enabled:
            try:
                mlflow.set_tag(key, value)
            except Exception as exc:
                logger.warning("MLflow set_tag failed: %s", exc)

    def add_notes(self, notes: str) -> None:
        """Append free-text researcher notes to the active run.

        Args:
            notes: Text to append (a newline is inserted between chunks).
        """
        with self._lock:
            if self._active_run is None:
                return
            sep = "\n" if self._active_run.notes else ""
            self._active_run.notes += sep + notes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_experiment_summary(self) -> dict[str, Path]:
        """Write full experiment-level summary files.

        Combines all runs (including previously persisted ones) into:

        * ``<experiments_dir>/<experiment_name>/experiment_summary.json``
        * ``<experiments_dir>/<experiment_name>/experiment_summary.csv``

        Returns:
            Dictionary with keys ``json`` and ``csv`` pointing to the
            written files.
        """
        base = self.config.experiments_dir / self.config.experiment_name
        base.mkdir(parents=True, exist_ok=True)

        all_records = self._load_all_run_records()
        ranked = self._rank_all_runs(all_records)

        # JSON summary
        json_path = base / "experiment_summary.json"
        summary = {
            "experiment_name": self.config.experiment_name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runs": len(ranked),
            "best_run_id": ranked[0].run_id if ranked else "",
            "best_composite_score": ranked[0].composite_score if ranked else 0.0,
            "runs": [asdict(r) for r in ranked],
        }
        try:
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=_json_default)
            logger.info("Experiment summary JSON -> %s", json_path)
        except OSError as exc:
            logger.error("Failed to write experiment summary JSON: %s", exc)

        # CSV summary (flat)
        csv_path = base / "experiment_summary.csv"
        rows = [_run_to_flat_dict(r) for r in ranked]
        if rows:
            try:
                flat_df = pd.DataFrame(rows)
                flat_df.to_csv(csv_path, index=False)
                logger.info("Experiment summary CSV  -> %s", csv_path)
            except (OSError, ValueError) as exc:
                logger.error("Failed to write experiment summary CSV: %s", exc)

        return {"json": json_path, "csv": csv_path}

    # ------------------------------------------------------------------
    # Comparison and ranking
    # ------------------------------------------------------------------

    def compare_experiments(
        self,
        output_path: Path | str | None = None,
        top_n: int | None = None,
    ) -> Path | None:
        """Generate a publication-quality experiment comparison figure.

        Plots each tracked metric as a grouped bar chart across all finished
        runs and saves the figure.

        Args:
            output_path: Destination PNG path.  Defaults to
                ``<figures_dir>/experiment_comparison.png``.
            top_n: If given, only the top-N runs by composite score are shown.

        Returns:
            Path of the saved figure, or None if there are no finished runs.
        """
        all_records = self._load_all_run_records()
        finished = [r for r in all_records if r.status == RUN_STATUS_FINISHED]
        if not finished:
            logger.warning("No finished runs found; skipping comparison figure")
            return None

        ranked = self._rank_all_runs(finished)
        if top_n is not None:
            ranked = ranked[:top_n]

        df = pd.DataFrame([_run_to_flat_dict(r) for r in ranked])

        metric_cols = sorted(
            [c for c in df.columns if c.startswith("metric_")],
        )
        if not metric_cols:
            logger.warning("No metric columns found; skipping comparison figure")
            return None

        # Limit to 6 panels for readability
        plot_metrics = metric_cols[:6]
        n_metrics = len(plot_metrics)
        ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols

        sns.set_theme(style="whitegrid", font_scale=1.1)
        palette = sns.color_palette("deep", n_colors=len(df))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4.5 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        run_labels = [
            f"{r.run_name[:18]}\n({r.run_id[-8:]})" for r in ranked
        ]

        for idx, metric_col in enumerate(plot_metrics):
            ax = axes_flat[idx]
            values = df[metric_col].fillna(0).values
            bars = ax.bar(
                run_labels, values,
                color=palette,
                width=0.55,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
            metric_name = metric_col.replace("metric_", "").replace("_", " ")
            ax.set_title(metric_name.title(), fontsize=12, fontweight="bold")
            ax.set_ylabel(metric_name, fontsize=10)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.grid(axis="y", alpha=0.4, zorder=1)

        # Hide unused panels
        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        # Composite score summary panel (always last)
        if n_metrics < nrows * ncols:
            ax_comp = axes_flat[n_metrics]
            ax_comp.set_visible(True)
        else:
            # Add an extra axes below the grid
            fig.set_size_inches(fig.get_size_inches()[0],
                                fig.get_size_inches()[1] + 3.5)
            ax_comp = fig.add_axes([0.1, -0.12, 0.8, 0.10])

        scores = df["composite_score"].fillna(0).values if "composite_score" in df.columns else np.zeros(len(df))
        bar_colors = ["#4CAF50" if i == 0 else "#FF9800" if i == 1 else "#2196F3"
                      for i in range(len(ranked))]
        comp_bars = ax_comp.barh(
            list(reversed(run_labels)),
            list(reversed(scores)),
            color=list(reversed(bar_colors)),
            height=0.55,
            edgecolor="white",
            linewidth=1.4,
        )
        ax_comp.bar_label(comp_bars, fmt="%.4f", padding=4, fontsize=9)
        ax_comp.set_xlabel(
            "Composite Score (40% F1 + 30% Acc + 30% ROC-AUC)", fontsize=10
        )
        ax_comp.set_title("Run Ranking", fontsize=12, fontweight="bold")
        ax_comp.set_xlim(0, 1.15)
        ax_comp.grid(axis="x", alpha=0.4)
        if scores.size > 0 and np.max(scores) > 0:
            ax_comp.axvline(np.max(scores), color="red", linewidth=1.0,
                            linestyle="--", alpha=0.5, label="Best score")
            ax_comp.legend(fontsize=9)

        fig.suptitle(
            f"Experiment Comparison — {self.config.experiment_name}\n"
            f"({len(ranked)} runs, ranked by composite score)",
            fontsize=14,
            fontweight="bold",
        )

        dest = Path(
            output_path
            if output_path is not None
            else self.config.figures_dir / "experiment_comparison.png"
        )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(dest, dpi=self.config.figure_dpi, bbox_inches="tight")
            logger.info("Comparison figure -> %s", dest)
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to save comparison figure: %s", exc)
            return None
        finally:
            plt.close(fig)

        return dest

    def get_best_run(self) -> RunRecord | None:
        """Return the highest-ranked finished run.

        Returns:
            The :class:`RunRecord` with the highest composite score, or None
            if no finished runs exist.
        """
        finished = [
            r for r in self._load_all_run_records()
            if r.status == RUN_STATUS_FINISHED
        ]
        if not finished:
            return None
        return self._rank_all_runs(finished)[0]

    def recommend_best(self) -> str:
        """Print and return a human-readable best-run recommendation.

        Returns:
            Formatted recommendation string (also printed to stdout).
        """
        best = self.get_best_run()
        if best is None:
            msg = "No finished runs found. Run at least one experiment first."
            print(msg)
            return msg

        lines = [
            "=" * 64,
            "BEST EXPERIMENT RECOMMENDATION",
            "=" * 64,
            f"  Run ID          : {best.run_id}",
            f"  Run name        : {best.run_name}",
            f"  Composite score : {best.composite_score:.4f}",
            f"  Status          : {best.status}",
            f"  Duration        : {best.duration_seconds:.1f} s",
            "",
            "  Key metrics:",
        ]
        for name, val in sorted(best.metrics.items()):
            lines.append(f"    {name:<30} {val:.4f}")
        lines += [
            "",
            "  Key parameters:",
        ]
        for name, val in sorted(best.params.items()):
            lines.append(f"    {name:<30} {val}")
        lines += ["=" * 64]
        msg = "\n".join(lines)
        print(msg)
        logger.info("Best run: %s (score=%.4f)", best.run_id, best.composite_score)
        return msg

    def get_run_history(self) -> pd.DataFrame:
        """Return all runs as a flat pandas DataFrame.

        Returns:
            DataFrame with one row per run, columns for all params and metrics
            prefixed with ``param_`` and ``metric_``.
        """
        records = self._load_all_run_records()
        if not records:
            return pd.DataFrame()
        rows = [_run_to_flat_dict(r) for r in records]
        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        """Print a concise experiment summary table to stdout."""
        df = self.get_run_history()
        if df.empty:
            print(f"No runs found for experiment '{self.config.experiment_name}'.")
            return
        display_cols = (
            ["run_id", "run_name", "status", "duration_seconds", "composite_score"]
            + [c for c in df.columns if c.startswith("metric_")]
        )
        display_cols = [c for c in display_cols if c in df.columns]
        print(f"\nExperiment: {self.config.experiment_name}")
        print(f"Total runs: {len(df)}")
        print()
        try:
            print(df[display_cols].to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            print(df[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_directories(self) -> None:
        """Create required output directories."""
        for d in (
            self.config.experiments_dir / self.config.experiment_name,
            self.config.figures_dir,
        ):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("Cannot create directory %s: %s", d, exc)

    def _setup_mlflow(self) -> None:
        """Initialise MLflow if available and configured."""
        if not _MLFLOW_AVAILABLE or not self.config.use_mlflow:
            logger.info(
                "MLflow %s — using local JSON/CSV backend",
                "not installed" if not _MLFLOW_AVAILABLE else "disabled",
            )
            self._mlflow_enabled = False
            return

        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._mlflow_enabled = True
            logger.info(
                "MLflow enabled | uri=%s | experiment=%s",
                self.config.tracking_uri,
                self.config.experiment_name,
            )
        except Exception as exc:
            logger.warning(
                "MLflow setup failed (%s) — falling back to local backend", exc
            )
            self._mlflow_enabled = False

    def _load_existing_runs(self) -> None:
        """Load previously persisted run records into ``_all_runs``."""
        exp_dir = self.config.experiments_dir / self.config.experiment_name
        if not exp_dir.is_dir():
            return
        loaded = 0
        for run_dir in sorted(exp_dir.iterdir()):
            record_path = run_dir / "run_record.json"
            if record_path.is_file():
                try:
                    run = _load_run_record(record_path)
                    # Avoid duplicates if the process restarts mid-experiment
                    if not any(r.run_id == run.run_id for r in self._all_runs):
                        self._all_runs.append(run)
                        loaded += 1
                except (OSError, json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Failed to load %s: %s", record_path, exc)
        if loaded:
            logger.debug("Loaded %d existing run records", loaded)

    def _run_dir(self, run: RunRecord) -> Path:
        """Return the per-run output directory path."""
        return (
            self.config.experiments_dir
            / run.experiment_name
            / run.run_id
        )

    def _run_artifacts_dir(self, run: RunRecord) -> Path:
        """Return the per-run artifacts subdirectory path."""
        return self._run_dir(run) / "artifacts"

    def _persist_run(self, run: RunRecord) -> None:
        """Write a run's record to disk as JSON and CSV.

        Args:
            run: Completed run record.
        """
        run_dir = self._run_dir(run)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)

            record_path = run_dir / "run_record.json"
            with record_path.open("w", encoding="utf-8") as fh:
                json.dump(asdict(run), fh, indent=2, default=_json_default)

            flat_path = run_dir / "run_summary.csv"
            pd.DataFrame([_run_to_flat_dict(run)]).to_csv(flat_path, index=False)

            logger.debug("Run persisted -> %s", run_dir)
        except OSError as exc:
            logger.error("Failed to persist run %s: %s", run.run_id, exc)

    def _update_summary_files(self, run: RunRecord) -> None:
        """Append the finished run to experiment-level summary files.

        Args:
            run: Completed run record to append.
        """
        self.save_experiment_summary()

    def _load_all_run_records(self) -> list[RunRecord]:
        """Return all run records, reloading from disk to catch external writes.

        Returns:
            List of :class:`RunRecord` objects.
        """
        # Re-load to pick up any runs written by other processes
        self._load_existing_runs()
        with self._lock:
            return list(self._all_runs)

    def _rank_run(self, run: RunRecord) -> float:
        """Compute a composite ranking score for a single run.

        Args:
            run: Run whose metrics are used for scoring.

        Returns:
            Normalised composite score in [0, 1] using the single-run
            metric values directly (cross-run normalisation is done in
            :meth:`_rank_all_runs`).
        """
        score = 0.0
        total_weight = 0.0
        for metric, weight in self.config.ranking_metric_weights.items():
            val = run.metrics.get(metric)
            if val is not None and _is_finite_scalar(val):
                score += float(val) * weight
                total_weight += weight
        if total_weight > 1e-9:
            return round(score / total_weight, 6)
        return 0.0

    def _rank_all_runs(self, runs: list[RunRecord]) -> list[RunRecord]:
        """Rank a list of runs using min-max normalised composite scoring.

        Args:
            runs: List of :class:`RunRecord` objects to rank.

        Returns:
            The same list sorted by composite score descending, with each
            record's ``composite_score`` field updated.
        """
        if not runs:
            return []

        # Gather per-metric min/max for normalisation
        metric_ranges: dict[str, tuple[float, float]] = {}
        for metric in self.config.ranking_metric_weights:
            vals = [
                r.metrics[metric]
                for r in runs
                if metric in r.metrics and _is_finite_scalar(r.metrics[metric])
            ]
            if vals:
                metric_ranges[metric] = (min(vals), max(vals))

        for run in runs:
            score = 0.0
            total_w = 0.0
            for metric, weight in self.config.ranking_metric_weights.items():
                val = run.metrics.get(metric)
                if val is None or not _is_finite_scalar(val):
                    continue
                lo, hi = metric_ranges.get(metric, (val, val))
                span = hi - lo
                normed = (float(val) - lo) / span if span > 1e-9 else 1.0
                score += normed * weight
                total_w += weight
            run.composite_score = round(score / total_w, 6) if total_w > 1e-9 else 0.0

        return sorted(runs, key=lambda r: r.composite_score, reverse=True)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    """Generate a timestamped unique run identifier.

    Returns:
        String of the form ``run_<ms_timestamp>_<8-char-uuid>``.
    """
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{uid}"


def _get_git_commit() -> str:
    """Return the short SHA of the current git HEAD, or empty string.

    Returns:
        7-character git commit hash, or ``''`` if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _get_platform_info() -> str:
    """Return a compact OS / hardware summary string.

    Returns:
        String such as ``Linux 6.1.0 | Python 3.10.14 | x86_64``.
    """
    return (
        f"{platform.system()} {platform.release()} | "
        f"Python {sys.version.split()[0]} | "
        f"{platform.machine()}"
    )


def _hash_file(path: Path) -> str:
    """Compute the first 12 hex characters of a file's SHA-256 hash.

    Args:
        path: File to hash.

    Returns:
        12-character hex string, or empty string on I/O error.
    """
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except OSError:
        return ""


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for NumPy scalars, Path, and datetime.

    Args:
        obj: Object that the default encoder cannot handle.

    Returns:
        JSON-serialisable equivalent.

    Raises:
        TypeError: For completely unknown types.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serialisable")


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = "__",
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        parent_key: Key prefix for the current recursion level.
        sep: Separator between nested key levels.

    Returns:
        Flat dictionary with compound keys joined by *sep*.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _is_finite_scalar(val: Any) -> bool:
    """Return True if *val* is a finite numeric scalar.

    Args:
        val: Value to test.

    Returns:
        True for finite int / float; False otherwise.
    """
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False


def _run_to_flat_dict(run: RunRecord) -> dict[str, Any]:
    """Convert a :class:`RunRecord` to a flat dictionary for CSV export.

    Args:
        run: Run record to flatten.

    Returns:
        Dict with scalar fields plus ``param_*`` and ``metric_*`` prefixed
        columns; ``artifacts`` and ``metric_history`` are omitted.
    """
    flat: dict[str, Any] = {
        "run_id": run.run_id,
        "experiment_name": run.experiment_name,
        "run_name": run.run_name,
        "status": run.status,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "duration_seconds": run.duration_seconds,
        "git_commit": run.git_commit,
        "python_version": run.python_version,
        "composite_score": run.composite_score,
        "n_artifacts": len(run.artifacts),
        "notes": run.notes,
    }
    for k, v in run.params.items():
        flat[f"param_{k}"] = v
    for k, v in run.metrics.items():
        flat[f"metric_{k}"] = v
    for k, v in run.tags.items():
        flat[f"tag_{k}"] = v
    return flat


def _load_run_record(path: Path) -> RunRecord:
    """Deserialise a :class:`RunRecord` from a JSON file.

    Args:
        path: Path to the ``run_record.json`` file.

    Returns:
        Reconstructed :class:`RunRecord`.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields are missing.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RunRecord(**{k: v for k, v in data.items() if k in RunRecord.__dataclass_fields__})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Experiment tracking for Wind Turbine Acoustic Monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate experiment comparison figure and print ranking.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print experiment run summary table.",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Print best-run recommendation.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a self-contained demo with three synthetic experiment runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _run_demo(tracker: "ExperimentTracker") -> None:
    """Execute three synthetic experiment runs to demonstrate the tracker.

    Args:
        tracker: Configured :class:`ExperimentTracker` instance.
    """
    import random

    rng = random.Random(42)

    scenarios = [
        {
            "run_name": "mfcc40_mel128_specSub",
            "params": {
                "n_mfcc": 40, "n_mels": 128, "n_fft": 2048,
                "hop_length": 512, "denoising_method": "spectral_subtraction",
                "random_seed": 42, "dataset_version": "synthetic_v1",
            },
            "metrics": {
                "accuracy": 0.912, "precision_macro": 0.908,
                "recall_macro": 0.914, "f1_macro": 0.911,
                "roc_auc_macro": 0.971, "training_time_s": 47.3,
                "inference_time_ms": 3.1,
            },
        },
        {
            "run_name": "mfcc20_mel64_wiener",
            "params": {
                "n_mfcc": 20, "n_mels": 64, "n_fft": 2048,
                "hop_length": 512, "denoising_method": "wiener",
                "random_seed": 42, "dataset_version": "synthetic_v1",
            },
            "metrics": {
                "accuracy": 0.843, "precision_macro": 0.838,
                "recall_macro": 0.846, "f1_macro": 0.842,
                "roc_auc_macro": 0.932, "training_time_s": 31.0,
                "inference_time_ms": 2.8,
            },
        },
        {
            "run_name": "mfcc80_mel256_wavelet",
            "params": {
                "n_mfcc": 80, "n_mels": 256, "n_fft": 2048,
                "hop_length": 512, "denoising_method": "wavelet",
                "random_seed": 42, "dataset_version": "synthetic_v1",
            },
            "metrics": {
                "accuracy": 0.881, "precision_macro": 0.875,
                "recall_macro": 0.883, "f1_macro": 0.879,
                "roc_auc_macro": 0.953, "training_time_s": 62.5,
                "inference_time_ms": 4.2,
            },
        },
    ]

    logger.info("Running demo with %d synthetic experiment runs", len(scenarios))

    for scenario in scenarios:
        with tracker.start_run(run_name=scenario["run_name"]) as t:
            t.log_feature_config(
                n_mfcc=scenario["params"]["n_mfcc"],
                n_mels=scenario["params"]["n_mels"],
                n_fft=scenario["params"]["n_fft"],
                hop_length=scenario["params"]["hop_length"],
                denoising_method=scenario["params"]["denoising_method"],
                random_seed=scenario["params"]["random_seed"],
                dataset_version=scenario["params"]["dataset_version"],
            )
            t.log_dataset_info(
                total_clips=20,
                clips_per_class={
                    "normal": 5, "bearing_fault": 5,
                    "blade_imbalance": 5, "gearbox_fault": 5,
                },
                total_duration_s=200.0,
            )
            # Simulate per-epoch metric logging
            for epoch in range(1, 4):
                t.log_metrics(
                    {"train_loss": rng.uniform(0.8, 1.2) / epoch,
                     "val_loss": rng.uniform(0.9, 1.3) / epoch},
                    step=epoch,
                )
            t.log_metrics(scenario["metrics"])
            t.log_model_info(
                model_name="CNN_Baseline",
                n_parameters=1_240_000,
                architecture="3×Conv + 2×Dense",
            )
            t.log_tag("stage", "week3_baseline")
            t.add_notes(f"Demo run for {scenario['run_name']}")

    tracker.save_experiment_summary()
    tracker.compare_experiments()
    tracker.recommend_best()
    tracker.print_summary()


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

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

    config = ExperimentConfig(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
    )
    tracker = ExperimentTracker(config)

    if not any([args.demo, args.compare, args.summary, args.recommend]):
        parser.print_help()
        return 0

    try:
        if args.demo:
            _run_demo(tracker)

        if args.summary:
            tracker.print_summary()

        if args.recommend:
            tracker.recommend_best()

        if args.compare:
            path = tracker.compare_experiments()
            if path:
                print(f"\nComparison figure saved to: {path}")

    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal error: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())