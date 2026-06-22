#!/usr/bin/env python3
"""Production-grade training engine for Wind Turbine Acoustic Monitoring.

This module provides :class:`Trainer`, the orchestration layer that turns a
:class:`~src.models.base_model.BaseModel` and a set of DataLoaders into a
trained, evaluated, checkpointed model.  It is designed to be the single entry
point a training CLI or notebook calls, and to be robust enough for unattended
runs on industrial-client infrastructure.

Responsibilities
----------------
The trainer owns every concern that sits *between* the model and the data:

* **Optimisation loop** — forward / backward / step, with mixed-precision
  autocast and gradient scaling on CUDA.
* **Gradient clipping** — global-norm clipping to prevent exploding gradients
  on the small, augmentation-heavy datasets typical of condition monitoring.
* **Learning-rate scheduling** — cosine (the ``config.yaml`` default), step,
  plateau, and one-cycle policies.
* **Early stopping** — patience-based, monitoring any metric in either
  direction, matching the ``early_stopping_metric=val_f1_macro`` contract.
* **Checkpointing & resume** — a checkpoint captures *everything* needed to
  resume bit-identically: model, optimiser, scheduler, AMP scaler, RNG states
  (Python / NumPy / Torch / CUDA), and the full :class:`TrainingState`.
* **Best-model tracking** — the best checkpoint (by the monitored metric) is
  saved separately and never overwritten by a worse epoch.
* **Metrics** — loss, accuracy, precision, recall, F1, ROC-AUC every epoch.
* **Experiment logging** — every metric and artifact flows to
  :class:`~src.utils.experiment_tracker.ExperimentTracker` (MLflow-backed) and,
  optionally, TensorBoard.
* **Reproducibility** — a fixed seed plus deterministic cuDNN makes runs
  repeatable; the RNG state is checkpointed so a resumed run continues the same
  random stream.
* **Fault tolerance** — training survives a single bad batch (logged and
  skipped), always flushes a checkpoint on interruption, and degrades
  gracefully when optional dependencies (AMP, TensorBoard, MLflow) are absent.

Design notes for reviewers
--------------------------
* The trainer never mutates the APIs of the model, dataloader, or tracker
  layers — it composes them through their public interfaces only.
* All torch-dependent behaviour is guarded; the module imports and its config
  classes validate even when PyTorch is not installed (for CI / docs).
* :class:`TrainingState` is a plain (mutable) dataclass intentionally — it is
  the evolving record of a run and is serialised into every checkpoint.

Usage::

    from src.training.trainer import Trainer, TrainerConfig
    from src.training.dataloader import create_production_loaders
    from src.models.cnn_classifier import build_acoustic_cnn

    bundle = create_production_loaders(Path("data/processed/features"))
    model = build_acoustic_cnn(num_classes=5)

    trainer = Trainer(
        model=model,
        train_loader=bundle.train,
        val_loader=bundle.val,
        test_loader=bundle.test,
        config=TrainerConfig(epochs=100, early_stopping_patience=10),
    )
    state = trainer.train()                 # runs to completion or early stop
    test_metrics = trainer.test()           # evaluates best model on test set

    # Resume a crashed run
    trainer.resume_training("checkpoints/last.pt")

CLI::

    python src/training/trainer.py --smoke   # synthetic end-to-end smoke test
"""

from __future__ import annotations

import json
import logging
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    # from __future__ import annotations
    import torch
    
    import torch.nn as nn

    _TORCH_AVAILABLE: bool = True
    _TORCH_VERSION: str = torch.__version__
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False
    _TORCH_VERSION: str = "not installed"

# ---------------------------------------------------------------------------
# Optional TensorBoard
# ---------------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]
    _TENSORBOARD_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# scikit-learn metrics
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SKLEARN_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseModel, resolve_device, set_global_seed

try:
    from src.utils.experiment_tracker import ExperimentTracker  # noqa: F401

    _TRACKER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ExperimentTracker = Any  # type: ignore[assignment,misc]
    _TRACKER_AVAILABLE: bool = False

logger = logging.getLogger("trainer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_FORMAT_VERSION: Final[str] = "1.0"

#: Supported learning-rate scheduler names.
SCHEDULERS: Final[tuple[str, ...]] = (
    "cosine", "step", "plateau", "onecycle", "none",
)

#: Supported optimiser names.
OPTIMIZERS: Final[tuple[str, ...]] = ("adam", "adamw", "sgd")

#: Metric names produced every epoch.
METRIC_NAMES: Final[tuple[str, ...]] = (
    "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
)

#: Default checkpoint file names.
LAST_CHECKPOINT_NAME: Final[str] = "last.pt"
BEST_CHECKPOINT_NAME: Final[str] = "best.pt"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the :class:`Trainer`.

    Defaults match the ``config.yaml`` training contract.

    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        optimizer: Optimiser name (``adam`` | ``adamw`` | ``sgd``).
        weight_decay: L2 weight decay.
        momentum: SGD momentum (ignored for Adam/AdamW).
        scheduler: LR scheduler (``cosine`` | ``step`` | ``plateau`` |
            ``onecycle`` | ``none``).
        scheduler_step_size: Epoch period for the step scheduler.
        scheduler_gamma: Decay factor for step / plateau schedulers.
        warmup_epochs: Linear LR warmup epochs (0 disables).
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_metric: Metric to monitor (e.g. ``val_f1_macro``).
        early_stopping_mode: ``max`` to maximise the metric, ``min`` to minimise.
        early_stopping_min_delta: Minimum change counted as improvement.
        gradient_clip_norm: Global gradient-norm clip value (0 disables).
        mixed_precision: Enable AMP autocast + GradScaler on CUDA.
        device: Device preference (``auto`` | ``cuda`` | ``mps`` | ``cpu``).
        multi_gpu: Wrap the model in ``DataParallel`` when multiple GPUs exist.
        num_classes: Number of output classes (for metric computation).
        random_seed: Global RNG seed.
        deterministic: Configure cuDNN for deterministic algorithms.
        checkpoint_dir: Directory for checkpoints.
        save_every_n_epochs: Periodic checkpoint cadence (0 = only last/best).
        log_every_n_steps: Step-level logging cadence within an epoch.
        tensorboard: Enable TensorBoard logging when available.
        tensorboard_dir: TensorBoard log directory.
        label_smoothing: Cross-entropy label smoothing factor.
        class_weights: Optional per-class loss weights (length ``num_classes``).
        max_grad_skip: Consecutive bad batches tolerated before aborting.
    """

    epochs:                    int = 100
    learning_rate:             float = 1e-3
    optimizer:                 str = "adam"
    weight_decay:              float = 1e-4
    momentum:                  float = 0.9
    scheduler:                 str = "cosine"
    scheduler_step_size:       int = 30
    scheduler_gamma:           float = 0.1
    warmup_epochs:             int = 0
    early_stopping_patience:   int = 10
    early_stopping_metric:     str = "val_f1_macro"
    early_stopping_mode:       str = "max"
    early_stopping_min_delta:  float = 1e-4
    gradient_clip_norm:        float = 1.0
    mixed_precision:           bool = True
    device:                    str = "auto"
    multi_gpu:                 bool = False
    num_classes:               int = 5
    random_seed:               int = 42
    deterministic:             bool = True
    checkpoint_dir:            Path = field(
        default_factory=lambda: _PROJECT_ROOT / "checkpoints"
    )
    save_every_n_epochs:       int = 0
    log_every_n_steps:         int = 10
    tensorboard:               bool = False
    tensorboard_dir:           Path = field(
        default_factory=lambda: _PROJECT_ROOT / "runs"
    )
    label_smoothing:           float = 0.0
    class_weights:             tuple[float, ...] | None = None
    max_grad_skip:             int = 5

    def __post_init__(self) -> None:
        """Validate configuration values at construction time."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {OPTIMIZERS}, got '{self.optimizer}'"
            )
        if self.scheduler not in SCHEDULERS:
            raise ValueError(
                f"scheduler must be one of {SCHEDULERS}, got '{self.scheduler}'"
            )
        if self.early_stopping_mode not in ("min", "max"):
            raise ValueError(
                f"early_stopping_mode must be 'min' or 'max', "
                f"got '{self.early_stopping_mode}'"
            )
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Mutable record of a training run, serialised into every checkpoint.

    Attributes:
        epoch: Last completed epoch (0-based; -1 before training starts).
        global_step: Total optimiser steps taken.
        best_metric: Best value of the monitored metric so far.
        best_epoch: Epoch at which ``best_metric`` was achieved.
        epochs_without_improvement: Early-stopping patience counter.
        should_stop: Whether early stopping has triggered.
        history: Per-epoch metric history (metric name -> list of values).
        wall_time_seconds: Cumulative training wall-clock time.
        finished: Whether the run completed (vs interrupted).
    """

    epoch:                       int = -1
    global_step:                 int = 0
    best_metric:                 float = float("nan")
    best_epoch:                  int = -1
    epochs_without_improvement:  int = 0
    should_stop:                 bool = False
    history:                     dict[str, list[float]] = field(default_factory=dict)
    wall_time_seconds:           float = 0.0
    finished:                    bool = False

    def record(self, metrics: dict[str, float]) -> None:
        """Append a dict of metrics to the history.

        Args:
            metrics: Mapping of metric name to scalar value.
        """
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the state.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingState":
        """Reconstruct a state from its dictionary form.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A :class:`TrainingState` instance.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------


class _EarlyStopping:
    """Patience-based early-stopping monitor.

    Args:
        patience: Epochs without improvement before stopping.
        mode: ``max`` or ``min``.
        min_delta: Minimum change counted as an improvement.
    """

    def __init__(self, patience: int, mode: str, min_delta: float) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Register a new metric value.

        Args:
            value: The monitored metric for the current epoch.

        Returns:
            Tuple ``(improved, should_stop)``.
        """
        if self.best is None or math.isnan(self.best):
            self.best = value
            self.counter = 0
            return True, False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            return True, False

        self.counter += 1
        should_stop = self.patience > 0 and self.counter >= self.patience
        return False, should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Production-grade training engine.

    Orchestrates the full training lifecycle for a
    :class:`~src.models.base_model.BaseModel`: optimisation with mixed
    precision and gradient clipping, LR scheduling, early stopping,
    checkpointing and exact resume, best-model tracking, full metric
    computation, and experiment logging.

    Args:
        model: The model to train (a :class:`BaseModel` subclass).
        train_loader: Training DataLoader yielding ``(features, labels)``.
        val_loader: Validation DataLoader (optional but recommended).
        test_loader: Test DataLoader (optional; used by :meth:`test`).
        config: Training configuration.
        experiment_tracker: Optional :class:`ExperimentTracker` instance.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
    """

    def __init__(
        self,
        model: "BaseModel",
        train_loader: Any,
        val_loader: Any = None,
        test_loader: Any = None,
        config: TrainerConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "Trainer requires PyTorch. Install with: pip install torch"
            )
        self.config = config or TrainerConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tracker = experiment_tracker

        # Reproducibility
        set_global_seed(self.config.random_seed, deterministic=self.config.deterministic)

        # Device
        self.device = torch.device(resolve_device(self.config.device))
        self.model.to(self.device)

        # Multi-GPU
        self._wrap_multi_gpu()

        # Optimiser, scheduler, scaler, loss
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()
        self.criterion = self._build_criterion()

        # Early stopping + state
        self.early_stopping = _EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode=self.config.early_stopping_mode,
            min_delta=self.config.early_stopping_min_delta,
        )
        self.state = TrainingState()

        # TensorBoard
        self.tb_writer = self._build_tensorboard()

        # Checkpoint dir
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer ready | device=%s | optimizer=%s | scheduler=%s | "
            "amp=%s | epochs=%d",
            self.device, self.config.optimizer, self.config.scheduler,
            self._amp_enabled, self.config.epochs,
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _wrap_multi_gpu(self) -> None:
        """Wrap the model in DataParallel when multiple GPUs are available."""
        if (
            self.config.multi_gpu
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            self.model = nn.DataParallel(self.model)
            logger.info("Wrapped model in DataParallel across %d GPUs",
                        torch.cuda.device_count())

    def _unwrap(self) -> "BaseModel":
        """Return the underlying model, unwrapping DataParallel if present.

        Returns:
            The base model.
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module  # type: ignore[return-value]
        return self.model

    def _build_optimizer(self) -> "torch.optim.Optimizer":
        """Construct the optimiser from config.

        Returns:
            A configured optimiser.
        """
        params = self.model.parameters()
        name = self.config.optimizer
        if name == "adam":
            return torch.optim.Adam(
                params, lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        # sgd
        return torch.optim.SGD(
            params, lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> Any:
        """Construct the LR scheduler from config.

        Returns:
            A scheduler instance, or ``None`` when disabled.
        """
        sched = self.config.scheduler
        opt = self.optimizer
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.config.epochs,
            )
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        if sched == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=self.config.early_stopping_mode,
                factor=self.config.scheduler_gamma,
                patience=max(1, self.config.early_stopping_patience // 2),
            )
        if sched == "onecycle":
            steps = max(1, self._safe_loader_len(self.train_loader))
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.config.learning_rate,
                epochs=self.config.epochs, steps_per_epoch=steps,
            )
        return None

    def _build_scaler(self) -> Any:
        """Construct the AMP gradient scaler.

        Returns:
            A ``GradScaler`` (enabled only on CUDA with mixed precision).
        """
        self._amp_enabled = (
            self.config.mixed_precision
            and self.device.type == "cuda"
        )
        try:
            # PyTorch 2.x preferred API
            return torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        except (AttributeError, TypeError):  # pragma: no cover
            return torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

    def _build_criterion(self) -> Any:
        """Construct the loss function from config.

        Returns:
            A ``CrossEntropyLoss`` with optional class weights and smoothing.
        """
        weight = None
        if self.config.class_weights is not None:
            weight = torch.tensor(
                self.config.class_weights, dtype=torch.float32, device=self.device,
            )
        try:
            return nn.CrossEntropyLoss(
                weight=weight, label_smoothing=self.config.label_smoothing,
            )
        except TypeError:  # pragma: no cover  (older torch without label_smoothing)
            return nn.CrossEntropyLoss(weight=weight)

    def _build_tensorboard(self) -> Any:
        """Construct a TensorBoard writer when enabled and available.

        Returns:
            A ``SummaryWriter`` or ``None``.
        """
        if not (self.config.tensorboard and _TENSORBOARD_AVAILABLE):
            if self.config.tensorboard and not _TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard requested but not installed; skipping.")
            return None
        self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(self.config.tensorboard_dir))

    @staticmethod
    def _safe_loader_len(loader: Any) -> int:
        """Return ``len(loader)`` or a fallback when unavailable.

        Args:
            loader: A DataLoader (or any iterable).

        Returns:
            The loader length, or 1 when it cannot be determined.
        """
        try:
            return len(loader)
        except (TypeError, AttributeError):
            return 1

    # ------------------------------------------------------------------
    # Autocast context
    # ------------------------------------------------------------------

    def _autocast(self) -> Any:
        """Return an autocast context manager (no-op when AMP disabled).

        Returns:
            A context manager.
        """
        if not self._amp_enabled:
            from contextlib import nullcontext
            return nullcontext()
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except (AttributeError, TypeError):  # pragma: no cover
            return torch.cuda.amp.autocast(enabled=True)

    # ------------------------------------------------------------------
    # Public: train
    # ------------------------------------------------------------------

    def train(self) -> TrainingState:
        """Run the full training loop to completion or early stop.

        Each epoch: train one pass, validate, step the scheduler, check early
        stopping, and checkpoint (last + best).  A keyboard interrupt or
        unexpected error flushes a checkpoint before propagating so no progress
        is lost.

        Returns:
            The final :class:`TrainingState`.
        """
        logger.info("Starting training for up to %d epochs", self.config.epochs)
        start_epoch = self.state.epoch + 1
        t_start = time.perf_counter()

        try:
            for epoch in range(start_epoch, self.config.epochs):
                self.state.epoch = epoch
                epoch_t0 = time.perf_counter()

                train_metrics = self._train_one_epoch(epoch)
                val_metrics = (
                    self.validate() if self.val_loader is not None else {}
                )

                # Scheduler step
                self._step_scheduler(val_metrics)

                # Assemble epoch metrics
                epoch_metrics = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "lr": self._current_lr(),
                    "epoch_seconds": time.perf_counter() - epoch_t0,
                }
                self.state.record(epoch_metrics)
                self._log_epoch(epoch, epoch_metrics)

                # Early stopping + best tracking
                monitored = epoch_metrics.get(
                    self.config.early_stopping_metric, float("nan")
                )
                improved, should_stop = self.early_stopping.update(monitored)
                if improved and not math.isnan(monitored):
                    self.state.best_metric = monitored
                    self.state.best_epoch = epoch
                    self.state.epochs_without_improvement = 0
                    self.save_checkpoint(self._best_path(), is_best=True)
                else:
                    self.state.epochs_without_improvement += 1

                # Always save 'last'
                self.save_checkpoint(self._last_path())
                if (
                    self.config.save_every_n_epochs > 0
                    and (epoch + 1) % self.config.save_every_n_epochs == 0
                ):
                    self.save_checkpoint(
                        self.config.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                    )

                if should_stop:
                    self.state.should_stop = True
                    logger.info(
                        "Early stopping at epoch %d (best %s=%.4f at epoch %d)",
                        epoch, self.config.early_stopping_metric,
                        self.state.best_metric, self.state.best_epoch,
                    )
                    break

            self.state.finished = True

        except KeyboardInterrupt:  # pragma: no cover
            logger.warning("Training interrupted; flushing checkpoint…")
            self.save_checkpoint(self._last_path())
            raise
        except Exception as exc:
            logger.error("Training failed at epoch %d: %s", self.state.epoch, exc)
            self.save_checkpoint(self._last_path())
            raise
        finally:
            self.state.wall_time_seconds += time.perf_counter() - t_start
            if self.tb_writer is not None:
                self.tb_writer.flush()

        logger.info(
            "Training complete | epochs=%d | best %s=%.4f @ epoch %d | %.1fs",
            self.state.epoch + 1, self.config.early_stopping_metric,
            self.state.best_metric, self.state.best_epoch,
            self.state.wall_time_seconds,
        )
        return self.state

    # ------------------------------------------------------------------
    # Public: validate / test
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Evaluate the model on the validation loader.

        Returns:
            Dictionary of metrics (loss, accuracy, precision, recall, F1,
            ROC-AUC).  Empty when no validation loader is set.
        """
        if self.val_loader is None:
            return {}
        return self._evaluate(self.val_loader, split="val")

    def test(self, *, use_best: bool = True) -> dict[str, float]:
        """Evaluate on the test loader, optionally loading the best checkpoint.

        Args:
            use_best: Load the best checkpoint before evaluating.

        Returns:
            Dictionary of test metrics.  Empty when no test loader is set.
        """
        if self.test_loader is None:
            logger.warning("test() called with no test_loader")
            return {}
        if use_best and self._best_path().is_file():
            logger.info("Loading best checkpoint for test evaluation")
            self.load_checkpoint(self._best_path(), resume=False)
        metrics = self._evaluate(self.test_loader, split="test")
        logger.info(
            "Test results | acc=%.4f f1=%.4f auc=%.4f",
            metrics.get("accuracy", float("nan")),
            metrics.get("f1_macro", float("nan")),
            metrics.get("roc_auc", float("nan")),
        )
        return metrics

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        skipped = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        for step, batch in enumerate(self.train_loader):
            try:
                loss, preds, labels, proba = self._train_step(batch)
            except RuntimeError as exc:
                skipped += 1
                logger.warning("Skipping bad batch %d: %s", step, exc)
                if skipped > self.config.max_grad_skip:
                    raise RuntimeError(
                        f"Exceeded max_grad_skip ({self.config.max_grad_skip}) "
                        "consecutive bad batches"
                    ) from exc
                continue

            skipped = 0
            total_loss += loss
            n_batches += 1
            self.state.global_step += 1
            all_preds.append(preds)
            all_labels.append(labels)
            all_proba.append(proba)

            if step % self.config.log_every_n_steps == 0:
                logger.debug(
                    "epoch %d step %d | loss=%.4f", epoch, step, loss
                )
            # OneCycle steps per batch
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self._compute_metrics(
            np.concatenate(all_labels) if all_labels else np.array([]),
            np.concatenate(all_preds) if all_preds else np.array([]),
            np.concatenate(all_proba) if all_proba else np.array([]),
        )
        metrics["loss"] = avg_loss
        return metrics

    def _train_step(
        self, batch: Any
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Run one optimisation step.

        Args:
            batch: A ``(features, labels)`` tuple from the loader.

        Returns:
            Tuple ``(loss_value, predictions, labels, probabilities)``.
        """
        features, labels = self._unpack_batch(batch)
        features = features.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            logits = self.model(features)
            loss = self.criterion(logits, labels)

        # Backward with AMP scaling
        self.scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        if self.config.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            proba = torch.softmax(logits.float(), dim=-1)
            preds = torch.argmax(proba, dim=-1)

        return (
            float(loss.detach().cpu()),
            preds.cpu().numpy(),
            labels.cpu().numpy(),
            proba.detach().cpu().numpy(),
        )

    def _evaluate(self, loader: Any, *, split: str) -> dict[str, float]:
        """Evaluate the model on a loader without gradient updates.

        Args:
            loader: DataLoader to evaluate.
            split: Split name for logging (``val`` / ``test``).

        Returns:
            Dictionary of metrics.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                features, labels = self._unpack_batch(batch)
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)

                proba = torch.softmax(logits.float(), dim=-1)
                preds = torch.argmax(proba, dim=-1)

                total_loss += float(loss.detach().cpu())
                n_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_proba.append(proba.detach().cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        metrics = self._compute_metrics(
            np.concatenate(all_labels) if all_labels else np.array([]),
            np.concatenate(all_preds) if all_preds else np.array([]),
            np.concatenate(all_proba) if all_proba else np.array([]),
        )
        metrics["loss"] = avg_loss
        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, float]:
        """Compute the full metric suite.

        Args:
            y_true: Ground-truth labels ``(N,)``.
            y_pred: Predicted labels ``(N,)``.
            y_proba: Class probabilities ``(N, num_classes)``.

        Returns:
            Dictionary with accuracy, precision, recall, F1, and ROC-AUC.
            Missing/degenerate cases yield ``nan`` rather than raising.
        """
        out: dict[str, float] = {
            "accuracy": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
            "roc_auc": float("nan"),
        }
        if y_true.size == 0 or not _SKLEARN_AVAILABLE:
            return out

        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0,
        )
        out["precision_macro"] = float(prec)
        out["recall_macro"] = float(rec)
        out["f1_macro"] = float(f1)

        # ROC-AUC needs probabilities and >= 2 classes present
        try:
            n_present = len(np.unique(y_true))
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2 and n_present >= 2:
                if y_proba.shape[1] == 2:
                    out["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba[:, 1])
                    )
                else:
                    out["roc_auc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr",
                        average="macro", labels=list(range(y_proba.shape[1])),
                    ))
        except (ValueError, IndexError) as exc:
            logger.debug("ROC-AUC unavailable: %s", exc)

        return out

    # ------------------------------------------------------------------
    # Scheduler / LR
    # ------------------------------------------------------------------

    def _step_scheduler(self, val_metrics: dict[str, float]) -> None:
        """Advance the LR scheduler appropriately for its type.

        Args:
            val_metrics: Validation metrics (used by plateau scheduler).
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            return  # stepped per-batch in the train loop
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_key = self.config.early_stopping_metric.replace("val_", "")
            value = val_metrics.get(metric_key, val_metrics.get("loss", 0.0))
            self.scheduler.step(value)
        else:
            self.scheduler.step()

    def _current_lr(self) -> float:
        """Return the current learning rate.

        Returns:
            The first parameter group's learning rate.
        """
        return float(self.optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, path: str | Path, *, is_best: bool = False
    ) -> Path:
        """Save a full training checkpoint for exact resume.

        Captures model, optimiser, scheduler, AMP scaler, RNG states, and the
        :class:`TrainingState`.  Written atomically (temp file + rename) so an
        interrupted save never corrupts an existing checkpoint.

        Args:
            path: Destination ``.pt`` file path.
            is_best: Tag this checkpoint as the best so far (metadata only).

        Returns:
            The resolved :class:`Path` written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format_version":       CHECKPOINT_FORMAT_VERSION,
            "model_state_dict":     self._unwrap().state_dict(),
            "model_config":         self._unwrap().config.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict":    self.scaler.state_dict(),
            "training_state":       self.state.to_dict(),
            "trainer_config":       asdict(self.config),
            "rng_state":            self._capture_rng_state(),
            "is_best":              is_best,
            "provenance": {
                "saved_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "torch_version": _TORCH_VERSION,
                "platform":      platform.platform(),
            },
        }

        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, str(tmp))
        tmp.replace(path)  # atomic on POSIX

        logger.debug("Saved checkpoint -> %s (is_best=%s)", path, is_best)
        if is_best and self.tracker is not None:
            self._log_artifact_safe(self.tracker, path, "best_checkpoint")
        return path

    def load_checkpoint(
        self, path: str | Path, *, resume: bool = True,
        map_location: str | None = None,
    ) -> None:
        """Load a checkpoint into the trainer.

        Args:
            path: Checkpoint file path.
            resume: When ``True``, restore optimiser / scheduler / scaler /
                RNG / training state for exact continuation.  When ``False``,
                only the model weights are loaded (e.g. for test evaluation).
            map_location: Device map for loading; defaults to the trainer device.

        Raises:
            FileNotFoundError: When *path* does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        loc = map_location or str(self.device)
        checkpoint = torch.load(str(path), map_location=loc, weights_only=False)

        self._unwrap().load_state_dict(checkpoint["model_state_dict"])

        if not resume:
            logger.info("Loaded model weights from %s (no resume)", path)
            return

        if checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            self.scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except (ValueError, KeyError) as exc:  # pragma: no cover
                logger.warning("Could not restore AMP scaler: %s", exc)

        self.state = TrainingState.from_dict(checkpoint["training_state"])
        self._restore_rng_state(checkpoint.get("rng_state"))

        # Restore early-stopping bookkeeping
        self.early_stopping.best = (
            self.state.best_metric if not math.isnan(self.state.best_metric) else None
        )
        self.early_stopping.counter = self.state.epochs_without_improvement

        logger.info(
            "Resumed from %s | epoch=%d | best %s=%.4f",
            path, self.state.epoch, self.config.early_stopping_metric,
            self.state.best_metric,
        )

    def resume_training(self, path: str | Path) -> TrainingState:
        """Resume a previously-interrupted run and continue to completion.

        Args:
            path: Checkpoint to resume from.

        Returns:
            The final :class:`TrainingState`.
        """
        self.load_checkpoint(path, resume=True)
        logger.info("Resuming training from epoch %d", self.state.epoch + 1)
        return self.train()

    # ------------------------------------------------------------------
    # RNG state
    # ------------------------------------------------------------------

    def _capture_rng_state(self) -> dict[str, Any]:
        """Capture Python / NumPy / Torch / CUDA RNG states.

        Returns:
            Dictionary of RNG states (CUDA omitted when unavailable).
        """
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        """Restore RNG states captured by :meth:`_capture_rng_state`.

        Args:
            state: The RNG state dict, or ``None`` to skip.
        """
        if not state:
            return
        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(_as_byte_tensor(state["torch"]))
            if "cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(
                    [_as_byte_tensor(s) for s in state["cuda"]]
                )
        except (KeyError, TypeError, RuntimeError) as exc:  # pragma: no cover
            logger.warning("Could not fully restore RNG state: %s", exc)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics to console, tracker, and TensorBoard.

        Args:
            epoch: Epoch index.
            metrics: Epoch metric dictionary.
        """
        logger.info(
            "epoch %3d | train_loss=%.4f val_loss=%.4f val_f1=%.4f "
            "val_acc=%.4f lr=%.2e",
            epoch,
            metrics.get("train_loss", float("nan")),
            metrics.get("val_loss", float("nan")),
            metrics.get("val_f1_macro", float("nan")),
            metrics.get("val_accuracy", float("nan")),
            metrics.get("lr", float("nan")),
        )

        if self.tracker is not None:
            try:
                self.tracker.log_metrics(metrics, step=epoch)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tracker log_metrics failed: %s", exc)

        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    self.tb_writer.add_scalar(key, value, epoch)

    @staticmethod
    def _log_artifact_safe(tracker: Any, path: Path, kind: str) -> None:
        """Log an artifact to the tracker, swallowing failures.

        Args:
            tracker: ExperimentTracker instance.
            path: Artifact path.
            kind: Artifact type label.
        """
        try:
            tracker.log_artifact(str(path), description=kind, artifact_type=kind)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tracker log_artifact failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    # def _unpack_batch(batch: Any) -> tuple["torch.Tensor", "torch.Tensor"]:
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a ``(features, labels)`` batch.

        Args:
            batch: A tuple/list of ``(features, labels)``.

        Returns:
            Tuple ``(features, labels)``.

        Raises:
            ValueError: When the batch is not a 2-element sequence.
        """
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Each batch must be a (features, labels) tuple; "
                f"got {type(batch).__name__}"
            )
        return batch[0], batch[1]

    def _best_path(self) -> Path:
        """Path to the best checkpoint."""
        return self.config.checkpoint_dir / BEST_CHECKPOINT_NAME

    def _last_path(self) -> Path:
        """Path to the last checkpoint."""
        return self.config.checkpoint_dir / LAST_CHECKPOINT_NAME

    def get_history(self) -> dict[str, list[float]]:
        """Return the per-epoch metric history.

        Returns:
            Mapping of metric name to a list of per-epoch values.
        """
        return dict(self.state.history)

    def close(self) -> None:
        """Release resources (TensorBoard writer)."""
        if self.tb_writer is not None:
            self.tb_writer.close()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _as_byte_tensor(value: Any) -> Any:
    """Coerce a stored RNG state back to a ``torch.ByteTensor``.

    ``torch.get_rng_state`` returns a ByteTensor; after a checkpoint round-trip
    it is already a tensor, so this is largely a safety shim.

    Args:
        value: A tensor or tensor-like RNG state.

    Returns:
        A ``torch.ByteTensor``.
    """
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.to(torch.uint8)
    if _TORCH_AVAILABLE:
        return torch.tensor(value, dtype=torch.uint8)
    return value


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> int:
    """Run a synthetic end-to-end training smoke test.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot run smoke test.")
        return 1

    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(42)
    n, n_classes = 64, 3
    # Separable synthetic data
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(torch.randn(n // n_classes, 1, 32, 32) + c * 2.0)
        ys.append(torch.full((n // n_classes,), c, dtype=torch.long))
    X = torch.cat(xs)
    y = torch.cat(ys)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    # Minimal model
    class _Net(BaseModel):
        def build_layers(self) -> None:
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(8, self.config.num_classes),
            )

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.net(x)

    from src.models.base_model import ModelConfig
    model = _Net(ModelConfig(num_classes=n_classes, input_shape=(32, 32)))

    import tempfile
    ckpt_dir = Path(tempfile.mkdtemp())
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=loader, test_loader=loader,
        config=TrainerConfig(
            epochs=5, mixed_precision=False, num_classes=n_classes,
            early_stopping_patience=10, checkpoint_dir=ckpt_dir,
            gradient_clip_norm=1.0, scheduler="cosine",
        ),
    )
    state = trainer.train()
    test_metrics = trainer.test()
    trainer.close()

    logger.info("Smoke test | final train_f1=%.3f | test_f1=%.3f",
                state.history.get("train_f1_macro", [float("nan")])[-1],
                test_metrics.get("f1_macro", float("nan")))
    logger.info("Checkpoints: %s", list(ckpt_dir.glob("*.pt")))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Trainer smoke test")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a synthetic end-to-end smoke test.")
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke_test()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())