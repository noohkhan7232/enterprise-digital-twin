#!/usr/bin/env python3
"""Abstract base model for Wind Turbine Acoustic Monitoring.

This module defines :class:`BaseModel`, the abstract foundation every neural
network in the project inherits from, plus the model registry and lifecycle
utilities (checkpointing, ONNX export, parameter accounting, device management,
reproducibility, and experiment-tracker integration).

Why an abstract base model?
---------------------------
A production acoustic-monitoring platform will host many architectures — a CNN
classifier, a denoising autoencoder, an RUL regressor, possibly a transformer —
each of which must support the same operational contract:

* **Checkpoint round-trips** — ``save_model`` / ``load_model`` must persist not
  just weights but the full config and metadata so a model can be reconstructed
  on a different machine with no source code beyond this package.
* **ONNX export** — the nacelle edge runtime consumes ONNX, so every model must
  export with correct dynamic axes.
* **Parameter accounting & summaries** — model cards and the research paper need
  exact parameter counts and layer tables.
* **Reproducibility** — a model created with a given seed must be bit-identical
  across runs.
* **Registry construction** — the training CLI builds models by name from
  config, never by importing concrete classes directly.

:class:`BaseModel` encodes all of this once so concrete architectures only need
to implement :meth:`~BaseModel.forward` and (optionally)
:meth:`~BaseModel.build_layers`.

torch-optional design
---------------------
The whole project degrades gracefully when PyTorch is absent (CI linting,
documentation builds, config validation).  :class:`BaseModel` therefore uses a
**conditional base class**: it inherits ``torch.nn.Module`` when torch is
installed and ``object`` otherwise, with :class:`abc.ABCMeta` as the metaclass
in both cases (so the abstract contract is always enforced).  Every
torch-dependent method guards on ``_TORCH_AVAILABLE`` and raises a clear,
actionable error when called without torch.

Registry usage
--------------
::

    from src.models.base_model import BaseModel, ModelConfig, register_model

    @register_model("acoustic_cnn")
    class AcousticCNN(BaseModel):
        def build_layers(self) -> None:
            self.conv = nn.Conv2d(1, 32, 3)
            self.head = nn.Linear(32, self.config.num_classes)

        def forward(self, x):
            return self.head(self.conv(x).mean(dim=(-1, -2)))

    model = build_model("acoustic_cnn", ModelConfig(num_classes=5))
    model.save_model("checkpoints/cnn_best.pt", metadata={"val_f1": 0.93})
    restored = AcousticCNN.load_model("checkpoints/cnn_best.pt")

CLI::

    python src/models/base_model.py --list      # list registered models
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import sys
import time
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, NamedTuple, Type

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import + conditional base class
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE: bool = True
    _TORCH_VERSION: str = torch.__version__
    _ModuleBase: type = nn.Module
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False
    _TORCH_VERSION: str = "not installed"
    _ModuleBase = object

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.utils.experiment_tracker import ExperimentTracker  # noqa: F401

    _TRACKER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ExperimentTracker = Any  # type: ignore[assignment,misc]
    _TRACKER_AVAILABLE: bool = False

logger = logging.getLogger("base_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Checkpoint format version — bump on breaking serialization changes.
CHECKPOINT_FORMAT_VERSION: Final[str] = "1.0"

#: Canonical fault class names in label order (matches the dataset layer).
DEFAULT_CLASS_NAMES: Final[tuple[str, ...]] = (
    "normal",
    "blade_imbalance",
    "bearing_fault",
    "gearbox_fault",
    "electrical_fault",
)

#: Default ONNX opset — 17 is broadly supported by edge runtimes (2024+).
DEFAULT_ONNX_OPSET: Final[int] = 17

#: Global model registry: name -> BaseModel subclass.
MODEL_REGISTRY: dict[str, Type["BaseModel"]] = {}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class ParameterCount(NamedTuple):
    """Parameter accounting for a model.

    Attributes:
        total: Total parameter count.
        trainable: Parameters with ``requires_grad=True``.
        non_trainable: Frozen parameters.
        buffers: Non-parameter buffer element count (e.g. BatchNorm stats).
        size_mb: Approximate model size in megabytes (float32).
    """

    total:         int
    trainable:     int
    non_trainable: int
    buffers:       int
    size_mb:       float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Architecture-agnostic base configuration for every model.

    Subclass configs may add architecture-specific fields, but every model in
    the project shares these common fields so the registry, checkpointing, and
    tracker integration can rely on them.

    Attributes:
        model_name: Human-readable / registry name of the architecture.
        num_classes: Number of output classes (classification) or targets.
        input_shape: Feature tensor shape excluding the batch dimension,
            e.g. ``(128, 431)`` for a mel spectrogram or ``(3, 128, 431)``
            for a 3-channel stack.
        in_channels: Number of input channels (1 for single spectrogram,
            3 for mel+Δ+ΔΔ).
        dropout: Dropout probability used by subclasses that support it.
        random_seed: Seed for deterministic initialisation.
        class_names: Ordered class names; defaults to the project taxonomy.
        extra: Free-form dict for architecture-specific hyperparameters that
            must survive a checkpoint round-trip.
    """

    model_name:   str = "base_model"
    num_classes:  int = len(DEFAULT_CLASS_NAMES)
    input_shape:  tuple[int, ...] = (128, 431)
    in_channels:  int = 1
    dropout:      float = 0.3
    random_seed:  int = 42
    class_names:  tuple[str, ...] = DEFAULT_CLASS_NAMES
    extra:        dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values at construction time."""
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if len(self.input_shape) == 0:
            raise ValueError("input_shape must be non-empty")

    @property
    def config_hash(self) -> str:
        """Short deterministic hash of the architecture-defining fields.

        Returns:
            First 12 hex characters of an MD5 over the config fields.
        """
        raw = (
            f"{self.model_name}:{self.num_classes}:{self.input_shape}:"
            f"{self.in_channels}:{self.dropout}:{sorted(self.extra.items())}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @property
    def batched_input_shape(self) -> tuple[int, ...]:
        """Input shape with the channel dimension prepended when needed.

        For a 2-D feature ``(F, T)`` with ``in_channels`` channels, returns
        ``(in_channels, F, T)``.  For an already-channelled shape, returns it
        unchanged.

        Returns:
            Channel-first input shape (excluding batch).
        """
        if len(self.input_shape) == 2 and self.in_channels >= 1:
            return (self.in_channels, *self.input_shape)
        return self.input_shape

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a JSON-compatible dictionary.

        Returns:
            Dictionary representation with tuples converted to lists.
        """
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["class_names"] = list(self.class_names)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Reconstruct a config from its dictionary form.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A :class:`ModelConfig` instance.
        """
        payload = dict(data)
        if "input_shape" in payload:
            payload["input_shape"] = tuple(payload["input_shape"])
        if "class_names" in payload:
            payload["class_names"] = tuple(payload["class_names"])
        # Drop unknown keys for forward compatibility
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(**payload)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_model(name: str):  # type: ignore[no-untyped-def]
    """Class decorator that registers a :class:`BaseModel` subclass by name.

    Args:
        name: Unique registry key used by :func:`build_model`.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: Type["BaseModel"]) -> Type["BaseModel"]:
        existing = MODEL_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model name '{name}' already registered to "
                f"{existing.__name__}; choose a unique name."
            )
        MODEL_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered model '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_model(
    name: str,
    config: ModelConfig | None = None,
    **kwargs: Any,
) -> "BaseModel":
    """Instantiate a registered model by name.

    Args:
        name: Registry key (see :func:`register_model`).
        config: Model configuration; defaults to ``ModelConfig(model_name=name)``.
        **kwargs: Forwarded to the model constructor after ``config``.

    Returns:
        An instantiated :class:`BaseModel` subclass.

    Raises:
        KeyError: When *name* is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    cls = MODEL_REGISTRY[name]
    cfg = config or ModelConfig(model_name=name)
    logger.info("Building model '%s' (%s)", name, cls.__name__)
    return cls(cfg, **kwargs)


def list_models() -> list[str]:
    """Return the sorted list of registered model names.

    Returns:
        Sorted registry keys.
    """
    return sorted(MODEL_REGISTRY)


def is_registered(name: str) -> bool:
    """Check whether a model name is registered.

    Args:
        name: Registry key to check.

    Returns:
        ``True`` when *name* is present in the registry.
    """
    return name in MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducible model construction and training.

    Seeds ``random``, ``numpy``, and (when available) ``torch`` on both CPU
    and CUDA.  When *deterministic* is set, also configures cuDNN for
    deterministic algorithms.

    Args:
        seed: The seed value.
        deterministic: Configure cuDNN/torch for deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.debug("Global seed set to %d (deterministic=%s)", seed, deterministic)


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device string from a preference.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or an
            explicit device string like ``"cuda:1"``.

    Returns:
        A concrete device string.  ``"auto"`` resolves to CUDA, then MPS,
        then CPU based on availability.  Without torch, always ``"cpu"``.
    """
    if not _TORCH_AVAILABLE:
        return "cpu"
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------


class BaseModel(_ModuleBase, metaclass=ABCMeta):  # type: ignore[misc,valid-type]
    """Abstract base class for every neural network in the project.

    Concrete subclasses must implement :meth:`forward`.  They may optionally
    override :meth:`build_layers` (called by ``__init__`` after seeding) to
    construct their layers; this keeps initialisation deterministic and
    centralised.

    Subclasses inherit the full operational contract:

    * :meth:`save_model` / :meth:`load_model` — checkpoint round-trips.
    * :meth:`export_onnx` — edge-runtime export.
    * :meth:`count_parameters` — exact parameter accounting.
    * :meth:`summary` — layer-by-layer table.
    * :meth:`to_device` / :attr:`device` — device management.
    * :meth:`log_to_tracker` — ExperimentTracker integration.

    Args:
        config: Model configuration.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
    """

    #: Set by :func:`register_model`; ``None`` for unregistered subclasses.
    _registry_name: str | None = None

    def __init__(self, config: ModelConfig | None = None) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                f"{type(self).__name__} requires PyTorch to instantiate. "
                "Install with: pip install torch"
            )
        super().__init__()
        self.config: ModelConfig = config or ModelConfig(
            model_name=self._registry_name or "base_model"
        )
        # Deterministic construction
        set_global_seed(self.config.random_seed)
        # Let subclass build its layers
        self.build_layers()

    # ------------------------------------------------------------------
    # Abstract / overridable
    # ------------------------------------------------------------------

    def build_layers(self) -> None:
        """Construct the model's layers.

        Called automatically by ``__init__`` after seeding.  The default is a
        no-op so subclasses may instead build layers directly in their own
        ``__init__`` if preferred.  Overriding this method is the recommended
        pattern because it keeps construction deterministic and centralised.
        """
        return None

    @abstractmethod
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the forward pass.

        Args:
            x: Input tensor of shape ``(batch, *input_shape)``.

        Returns:
            Output tensor (logits for classifiers).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @property
    def device(self) -> "torch.device":
        """The device of the model's first parameter (or CPU when empty).

        Returns:
            ``torch.device`` instance.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to_device(self, device: str = "auto") -> "BaseModel":
        """Move the model to a resolved device.

        Args:
            device: Device preference; see :func:`resolve_device`.

        Returns:
            ``self`` for chaining.
        """
        resolved = resolve_device(device)
        self.to(resolved)
        logger.info("Model '%s' moved to %s", self.config.model_name, resolved)
        return self

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------

    def count_parameters(self) -> ParameterCount:
        """Count parameters and estimate model size.

        Returns:
            :class:`ParameterCount` with total / trainable / non-trainable
            counts, buffer element count, and float32 size in MB.
        """
        total = trainable = 0
        param_bytes = 0
        for p in self.parameters():
            n = p.numel()
            total += n
            param_bytes += n * p.element_size()
            if p.requires_grad:
                trainable += n

        buffer_elems = 0
        buffer_bytes = 0
        for b in self.buffers():
            buffer_elems += b.numel()
            buffer_bytes += b.numel() * b.element_size()

        size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
        return ParameterCount(
            total         = total,
            trainable     = trainable,
            non_trainable = total - trainable,
            buffers       = buffer_elems,
            size_mb       = round(size_mb, 4),
        )

    @property
    def num_parameters(self) -> int:
        """Total parameter count.

        Returns:
            Number of parameters in the model.
        """
        return self.count_parameters().total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, *, max_depth: int = 1) -> str:
        """Return a human-readable layer-by-layer summary.

        Args:
            max_depth: Module nesting depth to display (1 = top-level children).

        Returns:
            Multi-line summary string with per-module parameter counts and a
            totals footer.
        """
        counts = self.count_parameters()
        col_name, col_type, col_params = 28, 22, 14
        header = (
            f"{'Layer':<{col_name}}{'Type':<{col_type}}"
            f"{'Params':>{col_params}}"
        )
        sep = "─" * (col_name + col_type + col_params)
        lines = [
            f"Model: {self.config.model_name}  "
            f"(registry: {self._registry_name or 'unregistered'})",
            sep, header, sep,
        ]

        for name, module in self.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            type_name = type(module).__name__
            lines.append(
                f"{name:<{col_name}}{type_name:<{col_type}}"
                f"{module_params:>{col_params},}"
            )

        lines.extend([
            sep,
            f"Total params       : {counts.total:,}",
            f"Trainable params   : {counts.trainable:,}",
            f"Non-trainable params: {counts.non_trainable:,}",
            f"Buffers (elements) : {counts.buffers:,}",
            f"Estimated size     : {counts.size_mb:.2f} MB",
            f"Input shape        : {self.config.batched_input_shape}",
            f"Output classes     : {self.config.num_classes}",
            sep,
        ])
        return "\n".join(lines)

    def print_summary(self, *, max_depth: int = 1) -> None:
        """Print :meth:`summary` to stdout.

        Args:
            max_depth: Module nesting depth to display.
        """
        print(self.summary(max_depth=max_depth))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _build_checkpoint(
        self, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Assemble the checkpoint dictionary.

        Args:
            metadata: Optional user metadata (epoch, metrics, notes).

        Returns:
            A fully-populated checkpoint dict.
        """
        counts = self.count_parameters()
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_name":     self.config.model_name,
            "registry_name":  self._registry_name,
            "class_name":     type(self).__name__,
            "config":         self.config.to_dict(),
            "state_dict":     self.state_dict(),
            "parameter_count": {
                "total":     counts.total,
                "trainable": counts.trainable,
                "size_mb":   counts.size_mb,
            },
            "metadata":       metadata or {},
            "provenance": {
                "saved_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "torch_version": _TORCH_VERSION,
                "python_version": platform.python_version(),
                "platform":      platform.platform(),
                "config_hash":   self.config.config_hash,
            },
        }

    def save_model(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        experiment_tracker: Any = None,
    ) -> Path:
        """Save the model to a checkpoint file.

        The checkpoint contains the format version, full config, state dict,
        parameter accounting, user metadata, and provenance (torch version,
        platform, timestamp, config hash) — everything needed to reconstruct
        the model elsewhere.

        Args:
            path: Destination ``.pt`` file path.
            metadata: Optional user metadata (e.g. ``{"epoch": 10, "val_f1": 0.9}``).
            experiment_tracker: Optional tracker; the checkpoint is logged as
                an artifact when provided.

        Returns:
            The resolved :class:`Path` written.

        Raises:
            RuntimeError: When PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("save_model requires PyTorch")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self._build_checkpoint(metadata)
        torch.save(checkpoint, str(path))
        logger.info(
            "Saved '%s' -> %s (%.2f MB)",
            self.config.model_name, path,
            checkpoint["parameter_count"]["size_mb"],
        )

        if experiment_tracker is not None:
            self._log_artifact_safe(experiment_tracker, path)
        return path

    @classmethod
    def load_model(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        strict: bool = True,
    ) -> "BaseModel":
        """Load a model from a checkpoint file.

        Reconstructs the config, instantiates the model, loads the state dict,
        and moves it to the resolved device.

        Args:
            path: Checkpoint ``.pt`` file path.
            device: Device preference; see :func:`resolve_device`.
            strict: Pass-through to ``load_state_dict`` strictness.

        Returns:
            The reconstructed model in eval mode on the requested device.

        Raises:
            RuntimeError: When PyTorch is unavailable.
            FileNotFoundError: When *path* does not exist.
            ValueError: When the checkpoint format is unrecognised.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("load_model requires PyTorch")

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(
                f"Unrecognised checkpoint format at {path}: "
                "missing 'config' or 'state_dict'."
            )

        config = ModelConfig.from_dict(checkpoint["config"])

        # Resolve the concrete class: prefer the registry, fall back to cls
        registry_name = checkpoint.get("registry_name")
        target_cls: Type[BaseModel] = cls
        if registry_name and registry_name in MODEL_REGISTRY:
            target_cls = MODEL_REGISTRY[registry_name]

        model = target_cls(config)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        model.to_device(device)
        model.eval()

        logger.info(
            "Loaded '%s' from %s (saved %s)",
            config.model_name, path,
            checkpoint.get("provenance", {}).get("saved_at", "unknown"),
        )
        return model

    @staticmethod
    def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
        """Read a checkpoint's metadata without instantiating the model.

        Args:
            path: Checkpoint file path.

        Returns:
            Dictionary with format version, model name, config, parameter
            count, metadata, and provenance.

        Raises:
            RuntimeError: When PyTorch is unavailable.
            FileNotFoundError: When *path* does not exist.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("inspect_checkpoint requires PyTorch")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        return {
            "format_version":  ckpt.get("format_version"),
            "model_name":      ckpt.get("model_name"),
            "registry_name":   ckpt.get("registry_name"),
            "class_name":      ckpt.get("class_name"),
            "config":          ckpt.get("config"),
            "parameter_count": ckpt.get("parameter_count"),
            "metadata":        ckpt.get("metadata"),
            "provenance":      ckpt.get("provenance"),
        }

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        path: str | Path,
        *,
        opset: int = DEFAULT_ONNX_OPSET,
        dynamic_batch: bool = True,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        experiment_tracker: Any = None,
    ) -> Path:
        """Export the model to ONNX for edge-runtime deployment.

        A dummy input is generated from :attr:`ModelConfig.batched_input_shape`.
        When *dynamic_batch* is set, axis 0 is marked dynamic so the exported
        graph accepts any batch size.

        Args:
            path: Destination ``.onnx`` file path.
            opset: ONNX opset version.
            dynamic_batch: Mark the batch axis as dynamic.
            input_names: Names for graph inputs (default ``["input"]``).
            output_names: Names for graph outputs (default ``["output"]``).
            experiment_tracker: Optional tracker; the file is logged when set.

        Returns:
            The resolved :class:`Path` written.

        Raises:
            RuntimeError: When PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("export_onnx requires PyTorch")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        in_names = input_names or ["input"]
        out_names = output_names or ["output"]
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                in_names[0]:  {0: "batch"},
                out_names[0]: {0: "batch"},
            }

        was_training = self.training
        self.eval()
        dummy = torch.randn(1, *self.config.batched_input_shape, device=self.device)
        try:
            torch.onnx.export(
                self,
                dummy,
                str(path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dynamic_axes,
            )
        finally:
            if was_training:
                self.train()

        logger.info(
            "Exported '%s' to ONNX -> %s (opset %d, dynamic_batch=%s)",
            self.config.model_name, path, opset, dynamic_batch,
        )
        if experiment_tracker is not None:
            self._log_artifact_safe(experiment_tracker, path, artifact_type="onnx")
        return path

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run inference and return class predictions.

        Args:
            x: Input batch of shape ``(batch, *input_shape)``.

        Returns:
            1-D tensor of predicted class indices.

        Raises:
            RuntimeError: When PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("predict requires PyTorch")
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.to(self.device))
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run inference and return class probabilities (softmax).

        Args:
            x: Input batch of shape ``(batch, *input_shape)``.

        Returns:
            Tensor of shape ``(batch, num_classes)`` summing to 1 per row.

        Raises:
            RuntimeError: When PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("predict_proba requires PyTorch")
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.to(self.device))
            return torch.softmax(logits, dim=-1)

    def freeze(self) -> "BaseModel":
        """Freeze all parameters (set ``requires_grad=False``).

        Returns:
            ``self`` for chaining.
        """
        for p in self.parameters():
            p.requires_grad = False
        logger.debug("Froze all parameters of '%s'", self.config.model_name)
        return self

    def unfreeze(self) -> "BaseModel":
        """Unfreeze all parameters (set ``requires_grad=True``).

        Returns:
            ``self`` for chaining.
        """
        for p in self.parameters():
            p.requires_grad = True
        logger.debug("Unfroze all parameters of '%s'", self.config.model_name)
        return self

    # ------------------------------------------------------------------
    # ExperimentTracker integration
    # ------------------------------------------------------------------

    def log_to_tracker(self, tracker: Any) -> None:
        """Log model architecture details to an ExperimentTracker.

        Args:
            tracker: An :class:`ExperimentTracker` instance (or compatible).
        """
        try:
            counts = self.count_parameters()
            tracker.log_model_info(
                model_name=self.config.model_name,
                n_parameters=counts.total,
                architecture=type(self).__name__,
                extra={
                    "trainable_params":  counts.trainable,
                    "non_trainable_params": counts.non_trainable,
                    "size_mb":           counts.size_mb,
                    "num_classes":       self.config.num_classes,
                    "input_shape":       list(self.config.batched_input_shape),
                    "config_hash":       self.config.config_hash,
                    "registry_name":     self._registry_name,
                },
            )
            logger.debug("Logged model info to ExperimentTracker")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker model logging failed: %s", exc)

    @staticmethod
    def _log_artifact_safe(
        tracker: Any, path: Path, artifact_type: str = "checkpoint"
    ) -> None:
        """Log an artifact to a tracker, swallowing any failure.

        Args:
            tracker: An ExperimentTracker instance.
            path: Artifact file path.
            artifact_type: Artifact category label.
        """
        try:
            tracker.log_artifact(
                str(path),
                description=f"{artifact_type} for model",
                artifact_type=artifact_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker artifact logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation including parameter count."""
        if not _TORCH_AVAILABLE:
            return f"{type(self).__name__}(torch unavailable)"
        try:
            n = self.num_parameters
            return (
                f"{type(self).__name__}(name='{self.config.model_name}', "
                f"params={n:,}, classes={self.config.num_classes})"
            )
        except Exception:  # noqa: BLE001
            return f"{type(self).__name__}(name='{self.config.model_name}')"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: list registered models / show environment.

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

    parser = argparse.ArgumentParser(description="BaseModel registry inspector")
    parser.add_argument("--list", action="store_true",
                        help="List registered models.")
    args = parser.parse_args(argv)

    print(f"PyTorch available : {_TORCH_AVAILABLE} ({_TORCH_VERSION})")
    print(f"ExperimentTracker : {_TRACKER_AVAILABLE}")
    print(f"Default device    : {resolve_device('auto')}")
    print(f"Checkpoint version: {CHECKPOINT_FORMAT_VERSION}")

    if args.list or True:
        models = list_models()
        print(f"\nRegistered models ({len(models)}):")
        if models:
            for name in models:
                print(f"  • {name} -> {MODEL_REGISTRY[name].__name__}")
        else:
            print("  (none — concrete models register themselves on import)")

    return 0


if __name__ == "__main__":
    sys.exit(main())