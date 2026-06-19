#!/usr/bin/env python3
"""Enterprise-grade DataLoader layer for Wind Turbine Acoustic Monitoring.

This module sits directly above :class:`~src.training.dataset.WindTurbineDataset`
and is responsible for every aspect of *how* data flows into a training loop:
worker management, device placement, batch sizing, sampling strategy, seeding,
memory management, and distributed-training coordination.

Why a dedicated DataLoader layer?
----------------------------------
PyTorch's raw ``DataLoader`` exposes a large surface of interacting options
(``num_workers``, ``pin_memory``, ``prefetch_factor``, ``persistent_workers``,
``sampler``, ``worker_init_fn`` …) that must be tuned consistently across the
train / val / test splits and adapted to the available hardware.  A single
mis-configuration — e.g. ``pin_memory=True`` on a CPU-only machine, or
``persistent_workers=True`` with ``num_workers=0`` — causes silent performance
degradation or a crash that surfaces only at epoch 2.

:class:`DataLoaderManager` encodes the correct combination of these settings
for every deployment scenario the project targets:

* **Local development** (laptop, no GPU, 4 cores): 0–2 workers, no pin-memory,
  small batch size derived from available RAM.
* **Single-GPU training** (research server): auto pin-memory, prefetch_factor=2,
  workers tuned to CPU count.
* **Multi-GPU / DDP** (cloud cluster): :class:`torch.utils.data.DistributedSampler`
  replaces the weighted sampler; each rank sees a disjoint shard.
* **Edge inference** (nacelle hardware): batch_size=1, num_workers=0, no
  pin-memory, synchronous loading.

Integration with existing modules
----------------------------------
* Reads from :class:`~src.training.dataset.WindTurbineDataset` via its
  public ``make_dataloader()`` interface — no internal APIs touched.
* Logs hardware profile, configuration, and memory snapshots to
  :class:`~src.utils.experiment_tracker.ExperimentTracker`.
* :class:`DataLoaderConfig` follows the frozen-dataclass convention used
  throughout :mod:`src.preprocessing` and :mod:`src.utils`.

Concurrency contract
---------------------
All ``DataLoader`` instances produced by this module satisfy:

1. **Deterministic worker seeds** — each worker is seeded as
   ``base_seed + worker_id``, so runs with the same seed produce identical
   data orderings.
2. **No shared mutable state** — worker functions are module-level named
   functions (not lambdas) so they pickle correctly across processes.
3. **Persistent workers** — enabled only when ``num_workers > 0`` to avoid
   the hidden ``StopIteration`` bug in PyTorch < 1.9.

Usage::

    from src.training.dataloader import DataLoaderConfig, DataLoaderManager
    from src.training.dataset import DatasetConfig, FeatureMode

    ds_cfg = DatasetConfig(features_dir=Path("data/processed/features"))
    dl_cfg = DataLoaderConfig(batch_size=32, num_workers=4)

    manager = DataLoaderManager(dataset_config=ds_cfg, dataloader_config=dl_cfg)
    train_loader, val_loader, test_loader = manager.create_all_loaders()

    for batch_features, batch_labels in train_loader:
        ...

CLI::

    python src/training/dataloader.py --features data/processed/features
    python src/training/dataloader.py --features data/processed/features --benchmark
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import (
        DataLoader,
        DistributedSampler,
        RandomSampler,
        SequentialSampler,
        WeightedRandomSampler,
    )

    _TORCH_AVAILABLE: bool = True
    _TORCH_VERSION: str = torch.__version__
    # prefetch_factor was introduced in PyTorch 1.7.0
    _PREFETCH_SUPPORTED: bool = tuple(
        int(x) for x in torch.__version__.split(".")[:2] if x.isdigit()
    ) >= (1, 7)
except ImportError:
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    DistributedSampler = None  # type: ignore[assignment]
    RandomSampler = None  # type: ignore[assignment]
    SequentialSampler = None  # type: ignore[assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False
    _TORCH_VERSION: str = "not installed"
    _PREFETCH_SUPPORTED: bool = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.dataset import (
    CLASS_NAMES,
    DatasetConfig,
    DataSplit,
    FeatureMode,
    WindTurbineDataset,
)

try:
    from src.utils.experiment_tracker import ExperimentConfig, ExperimentTracker
    _TRACKER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ExperimentTracker = None  # type: ignore[assignment,misc]
    ExperimentConfig = None   # type: ignore[assignment]
    _TRACKER_AVAILABLE: bool = False

logger = logging.getLogger("dataloader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum workers cap — beyond this the overhead outweighs the benefit.
_MAX_WORKERS_CAP: Final[int] = 12

#: Minimum free RAM per worker (GB) for audio + feature processing.
_RAM_PER_WORKER_GB: Final[float] = 0.5

#: Default memory fraction of available GPU VRAM to target for one batch.
_GPU_BATCH_MEMORY_FRACTION: Final[float] = 0.25

#: Default memory fraction of available CPU RAM to target for one batch.
_CPU_BATCH_MEMORY_FRACTION: Final[float] = 0.10

#: Bytes per float32 element.
_FLOAT32_BYTES: Final[int] = 4

#: Sentinel for "let the manager decide automatically".
AUTO: Final[str] = "auto"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class MemorySnapshot(NamedTuple):
    """Point-in-time memory measurement.

    Attributes:
        system_total_gb: Total system RAM in GB.
        system_available_gb: Free system RAM in GB.
        system_used_pct: System RAM usage percentage.
        cuda_available: Whether a CUDA device is visible.
        cuda_device_name: GPU model name, or empty string.
        cuda_total_gb: Total GPU VRAM in GB (0 when no GPU).
        cuda_allocated_gb: Currently allocated GPU VRAM in GB.
        cuda_reserved_gb: Reserved (cached) GPU VRAM in GB.
        timestamp: Unix timestamp of measurement.
    """

    system_total_gb:    float
    system_available_gb: float
    system_used_pct:    float
    cuda_available:     bool
    cuda_device_name:   str
    cuda_total_gb:      float
    cuda_allocated_gb:  float
    cuda_reserved_gb:   float
    timestamp:          float


class LoaderBundle(NamedTuple):
    """Typed container for all three DataLoader splits.

    Attributes:
        train: Training DataLoader (with optional weighted sampler).
        val: Validation DataLoader (sequential, no shuffling).
        test: Test DataLoader (sequential, no shuffling).
        config: The :class:`DataLoaderConfig` used to create these loaders.
        hardware_profile: Hardware profile dict logged at creation time.
    """

    train:            Any  # DataLoader | None
    val:              Any  # DataLoader | None
    test:             Any  # DataLoader | None
    config:           "DataLoaderConfig"
    hardware_profile: dict[str, Any]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataLoaderConfig:
    """Complete configuration for :class:`DataLoaderManager`.

    Every numeric field has a documented rationale tied to the project's
    hardware targets and the audit-report recommendations.

    Attributes:
        batch_size: Mini-batch size.  Use ``"auto"`` to let the manager
            derive a size from available memory and feature shape.
        num_workers: Parallel loading workers.  Use ``"auto"`` to derive
            from CPU count and available RAM.  ``0`` disables
            multiprocessing (required on Windows and when debugging).
        pin_memory: Pin CPU tensors for faster H2D transfer.  Use
            ``"auto"`` to enable only when a CUDA device is present.
        prefetch_factor: Batches to prefetch per worker.  Ignored when
            ``num_workers=0``.  Use ``"auto"`` to set 2 on GPU, 1 on CPU.
        persistent_workers: Keep worker processes alive between epochs.
            Significantly reduces epoch startup latency.  Automatically
            set to ``False`` when ``num_workers=0``.
        drop_last_train: Drop the final incomplete training batch to keep
            all batches the same size (important for BatchNorm stability).
        shuffle_train: Shuffle the training set.  Ignored when
            ``balanced_sampling=True`` (sampler handles ordering).
        balanced_sampling: Use :class:`WeightedRandomSampler` on the
            training loader to correct class imbalance.
        distributed: Enable :class:`DistributedSampler` for multi-GPU /
            multi-node DDP training.  Requires ``torch.distributed`` to
            be initialised before creating loaders.
        distributed_rank: This process's rank in the distributed group.
            Ignored when ``distributed=False``.
        distributed_world_size: Total number of processes in the group.
            Ignored when ``distributed=False``.
        seed: Base RNG seed for worker initialisation and sampler.
        max_workers_cap: Hard upper bound on auto-derived worker count.
        memory_target_gb: Target memory budget (GB) for auto batch sizing.
            On CPU this is RAM; on GPU this is VRAM.
        timeout: DataLoader worker timeout in seconds (0 = no timeout).
        collate_fn: Optional custom collation function.
            ``None`` uses PyTorch's default.
        experiment_tracker: Optional tracker for logging hardware profile
            and loader statistics.
    """

    batch_size:            int | str = 32
    num_workers:           int | str = AUTO
    pin_memory:            bool | str = AUTO
    prefetch_factor:       int | str = AUTO
    persistent_workers:    bool = True
    drop_last_train:       bool = True
    shuffle_train:         bool = True
    balanced_sampling:     bool = True
    distributed:           bool = False
    distributed_rank:      int = 0
    distributed_world_size: int = 1
    seed:                  int = 42
    max_workers_cap:       int = _MAX_WORKERS_CAP
    memory_target_gb:      float = 1.0
    timeout:               int = 0
    collate_fn:            Any = field(default=None, compare=False)
    experiment_tracker:    Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        if isinstance(self.batch_size, int) and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if isinstance(self.num_workers, int) and self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.memory_target_gb <= 0:
            raise ValueError(
                f"memory_target_gb must be positive, got {self.memory_target_gb}"
            )
        if self.distributed and not _TORCH_AVAILABLE:
            raise ValueError(
                "distributed=True requires PyTorch. "
                "Install with: pip install torch"
            )

    @property
    def effective_batch_size(self) -> int:
        """Return the configured batch size as an integer.

        Returns:
            Configured batch size; ``AUTO`` defers to the manager.
        """
        if self.batch_size == AUTO:
            return -1  # sentinel: manager will compute
        return int(self.batch_size)

    @property
    def effective_num_workers(self) -> int:
        """Return the configured worker count as an integer.

        Returns:
            Configured worker count; ``AUTO`` defers to the manager.
        """
        if self.num_workers == AUTO:
            return -1  # sentinel: manager will compute
        return int(self.num_workers)


# ---------------------------------------------------------------------------
# Module-level worker init function (must be picklable — no lambda)
# ---------------------------------------------------------------------------


def _seed_worker(worker_id: int) -> None:
    """Seed all RNGs in a DataLoader worker process.

    Called by PyTorch at worker startup via ``worker_init_fn``.  Sets
    ``random``, ``numpy``, and ``torch`` seeds to ``base_seed + worker_id``
    where ``base_seed`` is injected via :func:`torch.initial_seed`.

    Args:
        worker_id: Zero-based worker index passed by PyTorch.
    """
    worker_seed = (
        torch.initial_seed() % (2 ** 32)
        if _TORCH_AVAILABLE
        else 42 + worker_id
    )
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def take_memory_snapshot() -> MemorySnapshot:
    """Capture a point-in-time memory measurement.

    Returns:
        :class:`MemorySnapshot` with system RAM and GPU VRAM fields.
    """
    vm = psutil.virtual_memory()

    cuda_available   = False
    cuda_device_name = ""
    cuda_total_gb    = 0.0
    cuda_allocated   = 0.0
    cuda_reserved    = 0.0

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_available   = True
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_total_gb    = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 3
        )
        cuda_allocated   = round(torch.cuda.memory_allocated() / 1024 ** 3, 3)
        cuda_reserved    = round(torch.cuda.memory_reserved() / 1024 ** 3, 3)

    return MemorySnapshot(
        system_total_gb    = round(vm.total    / 1024 ** 3, 2),
        system_available_gb= round(vm.available/ 1024 ** 3, 2),
        system_used_pct    = vm.percent,
        cuda_available     = cuda_available,
        cuda_device_name   = cuda_device_name,
        cuda_total_gb      = cuda_total_gb,
        cuda_allocated_gb  = cuda_allocated,
        cuda_reserved_gb   = cuda_reserved,
        timestamp          = time.time(),
    )


def _derive_num_workers(max_cap: int = _MAX_WORKERS_CAP) -> int:
    """Derive a safe worker count from available hardware.

    Reserves one logical core for the main process, then caps by available
    RAM (``_RAM_PER_WORKER_GB`` per worker) and ``max_cap``.

    Args:
        max_cap: Hard upper bound on returned value.

    Returns:
        Non-negative integer worker count.
    """
    cpu = os.cpu_count() or 1
    cpu_based = max(0, cpu - 1)

    vm = psutil.virtual_memory()
    available_gb = vm.available / 1024 ** 3
    mem_based = max(0, int(available_gb / _RAM_PER_WORKER_GB))

    workers = min(cpu_based, mem_based, max_cap)
    logger.debug(
        "Auto workers: cpu_based=%d, mem_based=%d (%.1f GB), cap=%d → %d",
        cpu_based, mem_based, available_gb, max_cap, workers,
    )
    return workers


def _derive_batch_size(
    feature_shape: tuple[int, ...],
    memory_target_gb: float = 1.0,
    use_gpu: bool = False,
) -> int:
    """Derive a batch size that fits within the memory budget.

    Rounds the result down to the nearest power of two for GPU alignment.

    Args:
        feature_shape: Shape of one feature tensor (excluding batch dim).
        memory_target_gb: Target memory budget in GB.
        use_gpu: When ``True``, check available GPU VRAM instead of RAM.

    Returns:
        Power-of-two batch size in ``[1, 4096]``.
    """
    if not feature_shape:
        return 32

    sample_bytes = math.prod(feature_shape) * _FLOAT32_BYTES

    if use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available():
        total_vram   = torch.cuda.get_device_properties(0).total_memory
        available_gb = (
            total_vram - torch.cuda.memory_reserved()
        ) * _GPU_BATCH_MEMORY_FRACTION / 1024 ** 3
    else:
        vm = psutil.virtual_memory()
        available_gb = (
            vm.available * _CPU_BATCH_MEMORY_FRACTION / 1024 ** 3
        )

    target_gb  = min(memory_target_gb, available_gb)
    raw_batch  = max(1, int(target_gb * 1024 ** 3 / sample_bytes))

    # Nearest power of two ≤ raw_batch, capped at 4096
    power = min(12, max(0, int(math.log2(raw_batch))))
    result = 2 ** power
    logger.debug(
        "Auto batch size: feature=%s sample_bytes=%d target_gb=%.2f → %d",
        feature_shape, sample_bytes, target_gb, result,
    )
    return result


def _derive_pin_memory(use_gpu: bool | None = None) -> bool:
    """Determine whether to pin CPU tensors for H2D transfer.

    Args:
        use_gpu: Override. ``None`` auto-detects CUDA availability.

    Returns:
        ``True`` only when a CUDA device is present (or ``use_gpu=True``).
    """
    if use_gpu is not None:
        return bool(use_gpu)
    return bool(_TORCH_AVAILABLE and torch.cuda.is_available())


def _derive_prefetch_factor(num_workers: int, use_gpu: bool) -> int | None:
    """Choose ``prefetch_factor`` for the DataLoader.

    Args:
        num_workers: Resolved worker count.
        use_gpu: Whether a GPU is in use.

    Returns:
        ``2`` on GPU with workers, ``1`` on CPU with workers,
        ``None`` when ``num_workers=0`` (PyTorch requirement).
    """
    if num_workers == 0 or not _PREFETCH_SUPPORTED:
        return None
    return 2 if use_gpu else 1


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------


def build_hardware_profile() -> dict[str, Any]:
    """Build a comprehensive hardware profile for logging.

    Returns:
        Dictionary with CPU, RAM, CUDA, and PyTorch version details.
    """
    snap = take_memory_snapshot()
    profile: dict[str, Any] = {
        "cpu_logical_cores": os.cpu_count() or 1,
        "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
        "ram_total_gb":       snap.system_total_gb,
        "ram_available_gb":   snap.system_available_gb,
        "ram_used_pct":       snap.system_used_pct,
        "cuda_available":     snap.cuda_available,
        "cuda_device_name":   snap.cuda_device_name,
        "cuda_total_gb":      snap.cuda_total_gb,
        "torch_version":      _TORCH_VERSION,
        "python_version":     f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform":           sys.platform,
    }
    if snap.cuda_available and _TORCH_AVAILABLE:
        try:
            profile["cuda_version"]  = torch.version.cuda or "?"
            profile["cudnn_version"] = str(torch.backends.cudnn.version())
        except Exception:  # noqa: BLE001
            pass
    return profile


# ---------------------------------------------------------------------------
# DataLoaderManager
# ---------------------------------------------------------------------------


class DataLoaderManager:
    """Enterprise-grade DataLoader factory for Wind Turbine Acoustic Monitoring.

    Manages the full lifecycle of train / val / test DataLoaders:

    * Resolves ``AUTO`` settings to concrete values based on live hardware.
    * Applies reproducible worker seeding via ``worker_init_fn``.
    * Applies class-balanced weighted sampling to the training loader.
    * Activates :class:`DistributedSampler` for DDP runs.
    * Monitors and logs memory before and after loader creation.
    * Integrates with :class:`ExperimentTracker` for full provenance.

    Args:
        dataset_config: Configuration for the underlying
            :class:`~src.training.dataset.WindTurbineDataset`.
        dataloader_config: DataLoader-layer configuration.  Defaults to
            :class:`DataLoaderConfig` with all sensible defaults.

    Raises:
        RuntimeError: If dataset construction fails.
        ValueError: If ``dataloader_config`` is invalid.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataLoaderConfig | None = None,
    ) -> None:
        self.dataset_config    = dataset_config
        self.dataloader_config = dataloader_config or DataLoaderConfig()

        # Hardware profile captured once at construction
        self._hardware         = build_hardware_profile()
        self._use_gpu: bool    = self._hardware["cuda_available"]

        # Resolve AUTO settings to concrete values
        self._batch_size  = self._resolve_batch_size()
        self._num_workers = self._resolve_num_workers()
        self._pin_memory  = self._resolve_pin_memory()
        self._prefetch    = self._resolve_prefetch_factor()

        logger.info(
            "DataLoaderManager | batch=%d workers=%d pin=%s prefetch=%s gpu=%s",
            self._batch_size, self._num_workers,
            self._pin_memory, self._prefetch, self._use_gpu,
        )

        # Build the full dataset (ALL split) once; views are cheap
        snap_before = take_memory_snapshot()
        try:
            self._full_dataset = WindTurbineDataset(
                dataset_config, split=DataSplit.ALL
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"DataLoaderManager: dataset not found — {exc}\n"
                "Run: python -m src.preprocessing.pipeline "
                "--input data/raw/synthetic --output data/processed"
            ) from exc

        snap_after = take_memory_snapshot()
        ram_delta = snap_before.system_available_gb - snap_after.system_available_gb
        logger.info(
            "Dataset loaded: %d clips | RAM delta: %.2f GB",
            len(self._full_dataset), ram_delta,
        )

        # Split into train / val / test sub-datasets
        self._train_ds, self._val_ds, self._test_ds = (
            self._full_dataset.split()
        )

        # Experiment tracker
        tracker = (
            self.dataloader_config.experiment_tracker
            or (dataset_config.experiment_tracker if hasattr(dataset_config, "experiment_tracker") else None)
        )
        if tracker is not None:
            self._log_to_tracker(tracker)

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    def create_train_loader(self) -> "DataLoader | None":
        """Create the training DataLoader.

        Applies:
        * Weighted random sampling when ``balanced_sampling=True``.
        * :class:`DistributedSampler` when ``distributed=True``.
        * Worker seeding via :func:`_seed_worker`.
        * ``drop_last=drop_last_train`` to prevent incomplete batches.

        Returns:
            Configured ``DataLoader``, or ``None`` if PyTorch is absent.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("create_train_loader: PyTorch not installed")
            return None

        sampler = self._build_train_sampler()

        loader = self._make_loader(
            dataset    = self._train_ds,
            batch_size = self._batch_size,
            sampler    = sampler,
            shuffle    = (
                self.dataloader_config.shuffle_train
                if sampler is None else False
            ),
            drop_last  = self.dataloader_config.drop_last_train,
        )
        logger.info(
            "Train loader: %d clips | %d batches/epoch | sampler=%s",
            len(self._train_ds),
            math.ceil(len(self._train_ds) / self._batch_size),
            type(sampler).__name__ if sampler is not None else "None",
        )
        return loader

    def create_validation_loader(self) -> "DataLoader | None":
        """Create the validation DataLoader.

        Validation always uses sequential ordering with no shuffling and
        no class balancing so metrics are computed on the true distribution.

        Returns:
            Configured ``DataLoader``, or ``None`` if PyTorch is absent.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("create_validation_loader: PyTorch not installed")
            return None

        sampler = (
            DistributedSampler(
                self._val_ds,
                num_replicas = self.dataloader_config.distributed_world_size,
                rank         = self.dataloader_config.distributed_rank,
                shuffle      = False,
            )
            if self.dataloader_config.distributed
            else None
        )

        loader = self._make_loader(
            dataset    = self._val_ds,
            batch_size = self._batch_size,
            sampler    = sampler,
            shuffle    = False,
            drop_last  = False,
        )
        logger.info(
            "Val loader: %d clips | %d batches",
            len(self._val_ds),
            math.ceil(len(self._val_ds) / max(self._batch_size, 1)),
        )
        return loader

    def create_test_loader(self) -> "DataLoader | None":
        """Create the test DataLoader.

        Test always uses sequential ordering with no shuffling, no class
        balancing, and no drop-last (every sample must be evaluated).

        Returns:
            Configured ``DataLoader``, or ``None`` if PyTorch is absent.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("create_test_loader: PyTorch not installed")
            return None

        sampler = (
            DistributedSampler(
                self._test_ds,
                num_replicas = self.dataloader_config.distributed_world_size,
                rank         = self.dataloader_config.distributed_rank,
                shuffle      = False,
            )
            if self.dataloader_config.distributed
            else None
        )

        loader = self._make_loader(
            dataset    = self._test_ds,
            batch_size = self._batch_size,
            sampler    = sampler,
            shuffle    = False,
            drop_last  = False,
        )
        logger.info(
            "Test loader: %d clips | %d batches",
            len(self._test_ds),
            math.ceil(len(self._test_ds) / max(self._batch_size, 1)),
        )
        return loader

    def create_all_loaders(self) -> LoaderBundle:
        """Create train, validation, and test DataLoaders in one call.

        Returns:
            :class:`LoaderBundle` containing all three loaders, the
            resolved config, and the hardware profile.
        """
        t0 = time.perf_counter()
        train = self.create_train_loader()
        val   = self.create_validation_loader()
        test  = self.create_test_loader()
        elapsed = time.perf_counter() - t0

        logger.info("All loaders created in %.3f s", elapsed)

        snap = take_memory_snapshot()
        logger.info(
            "Post-loader memory: RAM %.1f GB free | GPU %.3f GB allocated",
            snap.system_available_gb, snap.cuda_allocated_gb,
        )

        return LoaderBundle(
            train            = train,
            val              = val,
            test             = test,
            config           = self.dataloader_config,
            hardware_profile = self._hardware,
        )

    # ------------------------------------------------------------------
    # Dataset accessors
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> WindTurbineDataset:
        """The training split :class:`WindTurbineDataset`."""
        return self._train_ds

    @property
    def val_dataset(self) -> WindTurbineDataset:
        """The validation split :class:`WindTurbineDataset`."""
        return self._val_ds

    @property
    def test_dataset(self) -> WindTurbineDataset:
        """The test split :class:`WindTurbineDataset`."""
        return self._test_ds

    @property
    def full_dataset(self) -> WindTurbineDataset:
        """The unsplit :class:`WindTurbineDataset` (DataSplit.ALL)."""
        return self._full_dataset

    @property
    def resolved_batch_size(self) -> int:
        """The batch size after AUTO resolution."""
        return self._batch_size

    @property
    def resolved_num_workers(self) -> int:
        """The worker count after AUTO resolution."""
        return self._num_workers

    @property
    def hardware_profile(self) -> dict[str, Any]:
        """Hardware profile captured at manager construction."""
        return dict(self._hardware)

    # ------------------------------------------------------------------
    # Memory monitoring
    # ------------------------------------------------------------------

    def memory_snapshot(self) -> MemorySnapshot:
        """Capture a current memory snapshot.

        Returns:
            :class:`MemorySnapshot` with system RAM and GPU VRAM.
        """
        return take_memory_snapshot()

    def log_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Capture and log a memory snapshot.

        Args:
            label: Optional label for the log line (e.g. ``"before_epoch_3"``).

        Returns:
            :class:`MemorySnapshot`.
        """
        snap = take_memory_snapshot()
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%sMemory | RAM: %.1f/%.1f GB (%.0f%%) | "
            "GPU: %s | VRAM: %.3f/%.3f GB allocated",
            prefix,
            snap.system_available_gb, snap.system_total_gb, snap.system_used_pct,
            snap.cuda_device_name or "N/A",
            snap.cuda_allocated_gb, snap.cuda_total_gb,
        )
        return snap

    def estimate_epoch_memory_gb(self) -> float:
        """Estimate peak RAM usage for one training epoch.

        Accounts for the batch tensor, gradient buffers (2× forward pass),
        and DataLoader prefetch buffer.

        Returns:
            Estimated peak RAM in GB (conservative upper bound).
        """
        try:
            shape = self.dataset_config.feature_shape
        except ValueError:
            shape = (128, 431)  # fallback for ALL mode

        sample_bytes   = math.prod(shape) * _FLOAT32_BYTES
        batch_bytes    = sample_bytes * self._batch_size
        prefetch       = (self._prefetch or 1) * (self._num_workers or 1)
        gradient_factor = 3.0  # batch + grad + optimizer state (Adam)

        total = batch_bytes * (prefetch + 1) * gradient_factor
        return round(total / 1024 ** 3, 4)

    def warn_if_memory_constrained(self) -> bool:
        """Log a warning and return ``True`` if memory may be insufficient.

        Compares :meth:`estimate_epoch_memory_gb` against available RAM.

        Returns:
            ``True`` when a potential OOM condition is detected.
        """
        snap     = take_memory_snapshot()
        estimate = self.estimate_epoch_memory_gb()
        available = (
            snap.cuda_total_gb - snap.cuda_allocated_gb
            if snap.cuda_available else snap.system_available_gb
        )

        if estimate > available * 0.8:
            logger.warning(
                "Potential OOM: estimated %.3f GB per epoch but only "
                "%.3f GB available. Consider reducing batch_size or num_workers.",
                estimate, available,
            )
            return True

        logger.debug(
            "Memory OK: estimated %.3f GB / %.3f GB available",
            estimate, available,
        )
        return False

    # ------------------------------------------------------------------
    # Benchmark utility
    # ------------------------------------------------------------------

    def benchmark(
        self,
        n_batches: int = 10,
        split: str = "train",
    ) -> dict[str, float]:
        """Measure DataLoader throughput and latency.

        Iterates ``n_batches`` mini-batches, recording total time,
        per-batch latency, and samples-per-second.

        Args:
            n_batches: Number of batches to time.
            split: Which loader to benchmark (``"train"``, ``"val"``,
                ``"test"``).

        Returns:
            Dictionary with ``total_s``, ``mean_batch_ms``,
            ``samples_per_sec``, ``n_batches``, and ``batch_size`` keys.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("benchmark() requires PyTorch")
            return {}

        loaders = {
            "train": self.create_train_loader,
            "val":   self.create_validation_loader,
            "test":  self.create_test_loader,
        }
        create_fn = loaders.get(split)
        if create_fn is None:
            raise ValueError(f"Unknown split '{split}'. Choose train/val/test.")

        loader = create_fn()
        if loader is None:
            return {}

        logger.info("Benchmarking %s loader (%d batches)…", split, n_batches)
        times: list[float] = []
        total_samples = 0

        t_start = time.perf_counter()
        for i, (feats, labels) in enumerate(loader):
            t0 = time.perf_counter()
            # Simulate minimal forward pass (move to device)
            if self._use_gpu and isinstance(feats, torch.Tensor):
                feats = feats.cuda(non_blocking=self._pin_memory)
            times.append(time.perf_counter() - t0)
            total_samples += (
                feats.shape[0]
                if hasattr(feats, "shape") else len(labels)
            )
            if i + 1 >= n_batches:
                break
        t_total = time.perf_counter() - t_start

        mean_ms   = (sum(times) / len(times) * 1000) if times else 0.0
        throughput = total_samples / max(t_total, 1e-9)

        result = {
            "total_s":        round(t_total, 3),
            "mean_batch_ms":  round(mean_ms, 2),
            "samples_per_sec": round(throughput, 1),
            "n_batches":      len(times),
            "batch_size":     self._batch_size,
        }
        logger.info(
            "Benchmark [%s]: %.1f samples/s | %.2f ms/batch | %d batches in %.2f s",
            split, throughput, mean_ms, len(times), t_total,
        )
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable configuration and hardware summary.

        Returns:
            Multi-line string suitable for logging or printing.
        """
        hw = self._hardware
        lines = [
            "DataLoaderManager Summary",
            "─" * 50,
            f"  Dataset clips  : {len(self._full_dataset)}",
            f"    train         : {len(self._train_ds)}",
            f"    val           : {len(self._val_ds)}",
            f"    test          : {len(self._test_ds)}",
            f"  Batch size      : {self._batch_size}",
            f"  Num workers     : {self._num_workers}",
            f"  Pin memory      : {self._pin_memory}",
            f"  Prefetch factor : {self._prefetch}",
            f"  Balanced sampl. : {self.dataloader_config.balanced_sampling}",
            f"  Distributed     : {self.dataloader_config.distributed}",
            f"  Drop last       : {self.dataloader_config.drop_last_train}",
            f"  Feature mode    : {self.dataset_config.feature_mode.value}",
            f"  Seed            : {self.dataloader_config.seed}",
            "─" * 50,
            f"  CPU cores       : {hw['cpu_logical_cores']} logical",
            f"  RAM             : {hw['ram_available_gb']:.1f} / {hw['ram_total_gb']:.1f} GB free",
            f"  GPU             : {hw['cuda_device_name'] or 'None'}",
            f"  VRAM            : {hw['cuda_total_gb']:.1f} GB",
            f"  PyTorch         : {hw['torch_version']}",
            f"  Est. epoch RAM  : {self.estimate_epoch_memory_gb():.3f} GB",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print :meth:`summary` to stdout."""
        print(self.summary())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_batch_size(self) -> int:
        """Resolve configured batch_size or derive from hardware.

        Returns:
            Concrete integer batch size.
        """
        if self.dataloader_config.batch_size != AUTO:
            return int(self.dataloader_config.batch_size)

        try:
            shape = self.dataset_config.feature_shape
        except ValueError:
            shape = (128, 431)  # ALL mode fallback

        return _derive_batch_size(
            feature_shape    = shape,
            memory_target_gb = self.dataloader_config.memory_target_gb,
            use_gpu          = self._use_gpu,
        )

    def _resolve_num_workers(self) -> int:
        """Resolve configured num_workers or derive from hardware.

        Returns:
            Non-negative integer worker count.
        """
        if self.dataloader_config.num_workers != AUTO:
            return int(self.dataloader_config.num_workers)
        return _derive_num_workers(self.dataloader_config.max_workers_cap)

    def _resolve_pin_memory(self) -> bool:
        """Resolve configured pin_memory or derive from GPU presence.

        Returns:
            Boolean pin-memory flag.
        """
        if self.dataloader_config.pin_memory != AUTO:
            return bool(self.dataloader_config.pin_memory)
        return _derive_pin_memory(self._use_gpu)

    def _resolve_prefetch_factor(self) -> int | None:
        """Resolve prefetch_factor or derive from context.

        Returns:
            Integer prefetch factor, or ``None`` when inapplicable.
        """
        if self.dataloader_config.prefetch_factor != AUTO:
            val = self.dataloader_config.prefetch_factor
            if val is None or val == 0:
                return None
            return int(val)
        return _derive_prefetch_factor(self._num_workers, self._use_gpu)

    def _build_train_sampler(self) -> Any:
        """Build the appropriate sampler for the training loader.

        Priority:
        1. :class:`DistributedSampler` when DDP is active.
        2. :class:`WeightedRandomSampler` when balanced_sampling is True.
        3. ``None`` (DataLoader uses shuffle=True).

        Returns:
            A sampler instance or ``None``.
        """
        if self.dataloader_config.distributed:
            return DistributedSampler(
                self._train_ds,
                num_replicas = self.dataloader_config.distributed_world_size,
                rank         = self.dataloader_config.distributed_rank,
                shuffle      = True,
                seed         = self.dataloader_config.seed,
            )

        if self.dataloader_config.balanced_sampling:
            sampler = self._train_ds.weighted_sampler()
            if sampler is not None:
                return sampler
            logger.debug(
                "weighted_sampler() returned None (PyTorch absent); "
                "falling back to shuffle"
            )
        return None

    def _make_loader(
        self,
        dataset:    WindTurbineDataset,
        batch_size: int,
        sampler:    Any,
        shuffle:    bool,
        drop_last:  bool,
    ) -> "DataLoader":
        """Construct a DataLoader with all resolved settings.

        Handles the ``prefetch_factor`` / ``persistent_workers`` quirks
        across PyTorch versions.

        Args:
            dataset: The :class:`WindTurbineDataset` split.
            batch_size: Resolved batch size.
            sampler: Sampler or ``None``.
            shuffle: Whether to shuffle (ignored when sampler is set).
            drop_last: Whether to drop the last incomplete batch.

        Returns:
            Configured :class:`torch.utils.data.DataLoader`.
        """
        effective_shuffle = shuffle if sampler is None else False
        persistent = (
            self.dataloader_config.persistent_workers
            and self._num_workers > 0
        )

        # Build generator for reproducible shuffle
        generator = None
        if _TORCH_AVAILABLE:
            generator = torch.Generator()
            generator.manual_seed(self.dataloader_config.seed)

        kwargs: dict[str, Any] = dict(
            dataset            = dataset,
            batch_size         = batch_size,
            shuffle            = effective_shuffle,
            num_workers        = self._num_workers,
            sampler            = sampler,
            pin_memory         = self._pin_memory,
            drop_last          = drop_last,
            worker_init_fn     = _seed_worker,
            persistent_workers = persistent,
            timeout            = self.dataloader_config.timeout,
            generator          = generator,
        )

        if self.dataloader_config.collate_fn is not None:
            kwargs["collate_fn"] = self.dataloader_config.collate_fn

        if self._prefetch is not None and self._num_workers > 0:
            kwargs["prefetch_factor"] = self._prefetch

        return DataLoader(**kwargs)

    def _log_to_tracker(self, tracker: Any) -> None:
        """Log hardware profile and resolved settings to ExperimentTracker.

        Args:
            tracker: An :class:`ExperimentTracker` instance.
        """
        try:
            tracker.log_params({
                "dataloader.batch_size":    self._batch_size,
                "dataloader.num_workers":   self._num_workers,
                "dataloader.pin_memory":    self._pin_memory,
                "dataloader.prefetch":      self._prefetch,
                "dataloader.balanced":      self.dataloader_config.balanced_sampling,
                "dataloader.distributed":   self.dataloader_config.distributed,
                "dataloader.seed":          self.dataloader_config.seed,
                "dataloader.drop_last":     self.dataloader_config.drop_last_train,
                "hw.cuda_available":        self._hardware["cuda_available"],
                "hw.cuda_device":           self._hardware["cuda_device_name"],
                "hw.ram_total_gb":          self._hardware["ram_total_gb"],
                "hw.cpu_cores":             self._hardware["cpu_logical_cores"],
                "hw.torch_version":         self._hardware["torch_version"],
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_production_loaders(
    features_dir: Path,
    *,
    feature_mode: FeatureMode = FeatureMode.MEL,
    batch_size: int | str = 32,
    num_workers: int | str = AUTO,
    balanced: bool = True,
    seed: int = 42,
    experiment_tracker: Any = None,
) -> LoaderBundle:
    """One-call factory for production-ready DataLoaders.

    Creates a :class:`DataLoaderManager` with sensible defaults and returns
    a :class:`LoaderBundle` ready for the training loop.

    Args:
        features_dir: Pre-extracted features directory (output of
            ``PreprocessingPipeline.process_and_save()``).
        feature_mode: Feature representation to use.
        batch_size: Mini-batch size, or ``"auto"`` for memory-derived size.
        num_workers: Worker processes, or ``"auto"`` for hardware-derived count.
        balanced: Apply weighted sampling to the training loader.
        seed: Global RNG seed.
        experiment_tracker: Optional :class:`ExperimentTracker` instance.

    Returns:
        :class:`LoaderBundle` with all three loaders.

    Raises:
        RuntimeError: When the features directory contains no clips.
    """
    ds_cfg = DatasetConfig(
        features_dir      = features_dir,
        feature_mode      = feature_mode,
        random_seed       = seed,
        experiment_tracker= experiment_tracker,
    )
    dl_cfg = DataLoaderConfig(
        batch_size          = batch_size,
        num_workers         = num_workers,
        balanced_sampling   = balanced,
        seed                = seed,
        experiment_tracker  = experiment_tracker,
    )
    manager = DataLoaderManager(ds_cfg, dl_cfg)
    return manager.create_all_loaders()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():  # type: ignore[no-untyped-def]
    import argparse
    p = argparse.ArgumentParser(
        description="DataLoaderManager smoke-test and benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--features", type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "features",
        help="Pre-extracted features directory.",
    )
    p.add_argument("--batch-size", default="auto",
                   help="Batch size or 'auto'.")
    p.add_argument("--workers", default="auto",
                   help="Worker count or 'auto'.")
    p.add_argument("--mode", default="mel",
                   choices=[m.value for m in FeatureMode])
    p.add_argument("--benchmark", action="store_true",
                   help="Run throughput benchmark after smoke-test.")
    p.add_argument("--n-batches", type=int, default=5,
                   help="Batches to time in benchmark mode.")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    logging.basicConfig(
        level=logging.DEBUG if "--verbose" in (argv or sys.argv) else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        batch_size = int(args.batch_size) if args.batch_size != "auto" else AUTO
        num_workers = int(args.workers) if args.workers != "auto" else AUTO

        ds_cfg = DatasetConfig(
            features_dir = args.features,
            feature_mode = FeatureMode(args.mode),
            cache_dir    = None,
        )
        dl_cfg = DataLoaderConfig(
            batch_size  = batch_size,
            num_workers = num_workers,
        )
        manager = DataLoaderManager(ds_cfg, dl_cfg)
        manager.print_summary()

        _ = manager.warn_if_memory_constrained()
        _ = manager.log_memory_snapshot("post-init")

        bundle = manager.create_all_loaders()
        logger.info(
            "Loaders ready | train=%s val=%s test=%s",
            "OK" if bundle.train else "None",
            "OK" if bundle.val   else "None",
            "OK" if bundle.test  else "None",
        )

        if args.benchmark and bundle.train is not None:
            result = manager.benchmark(n_batches=args.n_batches, split="train")
            logger.info("Benchmark: %s", result)

        return 0

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())