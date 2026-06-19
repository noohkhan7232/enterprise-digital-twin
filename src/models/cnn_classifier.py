#!/usr/bin/env python3
"""Production-grade CNN classifier for wind turbine acoustic fault detection.

This module implements :class:`CNNClassifier`, a ResNet-style convolutional
network optimised for 128-band log-mel spectrograms (the Week-2-recommended
feature representation).  It builds on :class:`~src.models.base_model.BaseModel`
and therefore inherits the full operational contract — checkpointing, ONNX
export, parameter accounting, device management, reproducible construction, and
ExperimentTracker integration — for free.

Architecture rationale
----------------------
The input is a single-channel log-mel spectrogram of shape ``(1, 128, 431)``
(mel bins × frames).  A spectrogram is an image-like tensor where:

* the **frequency axis** (128 mel bins) carries the harmonic signature of a
  fault — gear-mesh sidebands, bearing defect frequencies, blade-pass tones;
* the **time axis** (431 frames) carries the temporal evolution and
  modulation of those components.

A ResNet-style stack is the right inductive bias: stacked 3×3 convolutions with
residual skip connections learn local time-frequency patterns and compose them
hierarchically, while batch normalisation stabilises training on the small,
augmentation-heavy datasets typical of condition monitoring.  The design choices
that matter for hitting the **F1 > 0.85** target:

* **Residual blocks** — skip connections prevent the vanishing-gradient problem
  and let the network go deep enough (3 stages × 2 blocks) to separate the four
  fault classes without overfitting 20–2000 clips.
* **BatchNorm after every conv** — essential for stable convergence at the
  learning rate (1e-3) and batch size (32) fixed in ``config.yaml``.
* **Adaptive average pooling** — collapses the variable time dimension to a
  fixed 1×1 before the classifier head, so the network accepts clips of any
  length and the ONNX graph exports with a clean dynamic time axis.
* **Dropout before the head** — regularises the high-capacity classifier on
  small data.
* **Kaiming initialisation** — matched to ReLU activations for healthy initial
  gradient scale.

Spatial flow (input 128×431)::

    Input              (1,   128, 431)
    Stem conv 7×7 /2   (32,   64, 216)
    MaxPool 3×3 /2     (32,   32, 108)
    Stage 1 (×2)       (64,   32, 108)
    Stage 2 (×2, /2)   (128,  16,  54)
    Stage 3 (×2, /2)   (256,   8,  27)
    AdaptiveAvgPool    (256,   1,   1)
    Dropout + Linear   (num_classes,)

Configuration
-------------
Architecture hyperparameters live in ``ModelConfig.extra`` so they survive a
checkpoint round-trip:

* ``stem_channels`` (default 32)
* ``stage_channels`` (default ``[64, 128, 256]``)
* ``blocks_per_stage`` (default 2)
* ``zero_init_residual`` (default True — improves convergence)

Usage::

    from src.models.cnn_classifier import CNNClassifier
    from src.models.base_model import ModelConfig, build_model

    model = build_model("acoustic_cnn", ModelConfig(num_classes=5))
    logits = model(torch.randn(8, 1, 128, 431))   # (8, 5)
    model.export_onnx("cnn.onnx")                  # dynamic batch + time

CLI::

    python src/models/cnn_classifier.py --summary
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseModel, ModelConfig, register_model

logger = logging.getLogger("cnn_classifier")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

#: Registry name for the production CNN.
MODEL_NAME: Final[str] = "acoustic_cnn"

DEFAULT_STEM_CHANNELS:   Final[int] = 32
DEFAULT_STAGE_CHANNELS:  Final[tuple[int, ...]] = (64, 128, 256)
DEFAULT_BLOCKS_PER_STAGE: Final[int] = 2


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Pre-activation-free basic residual block (ResNet-18 style).

        Two 3×3 convolutions each followed by BatchNorm, with a skip
        connection.  When ``stride != 1`` or the channel count changes, the
        skip path applies a 1×1 convolution + BatchNorm to match dimensions.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Stride of the first convolution (2 downsamples spatially).
            zero_init_bn: Initialise the second BN's weight to zero so the
                block starts as an identity mapping (improves convergence).
        """

        expansion: int = 1

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            zero_init_bn: bool = True,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=1, bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            # Skip connection: project when shape changes
            self.downsample: nn.Module | None = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1,
                        stride=stride, bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )

            self._zero_init_bn = zero_init_bn

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the residual block.

            Args:
                x: Input tensor ``(B, in_channels, H, W)``.

            Returns:
                Output tensor ``(B, out_channels, H', W')``.
            """
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return self.relu(out)

else:  # pragma: no cover
    ResidualBlock = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# CNN classifier
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class CNNClassifier(BaseModel):
    """ResNet-style CNN for wind-turbine acoustic fault classification.

    Optimised for 128-band log-mel spectrograms.  Inherits the full lifecycle
    contract from :class:`~src.models.base_model.BaseModel` (checkpointing,
    ONNX export, parameter accounting, device management, tracker integration).

    The architecture is configured through ``ModelConfig.extra``:

    * ``stem_channels`` (int): channels after the stem conv (default 32).
    * ``stage_channels`` (list[int]): channels per residual stage
      (default ``[64, 128, 256]``).
    * ``blocks_per_stage`` (int): residual blocks per stage (default 2).
    * ``zero_init_residual`` (bool): zero-init the last BN in each block so
      blocks start as identity mappings (default True).

    Args:
        config: Model configuration.  ``input_shape`` should be the mel
            spectrogram shape ``(128, 431)`` (or any ``(F, T)``); the network
            adapts to the frequency axis and is time-agnostic via adaptive
            pooling.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
    """

    def build_layers(self) -> None:
        """Construct the stem, residual stages, and classifier head.

        Reads architecture hyperparameters from ``self.config.extra`` and
        applies weight initialisation after construction.
        """
        cfg = self.config
        extra = cfg.extra or {}

        stem_channels: int = int(extra.get("stem_channels", DEFAULT_STEM_CHANNELS))
        stage_channels: list[int] = list(
            extra.get("stage_channels", DEFAULT_STAGE_CHANNELS)
        )
        blocks_per_stage: int = int(
            extra.get("blocks_per_stage", DEFAULT_BLOCKS_PER_STAGE)
        )
        self._zero_init_residual: bool = bool(
            extra.get("zero_init_residual", True)
        )

        in_ch = cfg.in_channels

        # ── Stem: 7×7 conv stride 2 + BN + ReLU + 3×3 maxpool stride 2 ───────
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ─────────────────────────────────────────────────
        stages: list[nn.Module] = []
        current = stem_channels
        for stage_idx, out_ch in enumerate(stage_channels):
            # First stage keeps spatial size; later stages downsample by 2
            stride = 1 if stage_idx == 0 else 2
            blocks: list[nn.Module] = []
            for block_idx in range(blocks_per_stage):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(
                    ResidualBlock(
                        current, out_ch, stride=block_stride,
                        zero_init_bn=self._zero_init_residual,
                    )
                )
                current = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self._final_channels = current

        # ── Head: adaptive pool + dropout + linear ───────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(current, cfg.num_classes)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming init to convs and normal init to the classifier.

        When ``zero_init_residual`` is set, the last BatchNorm in every
        residual block is zero-initialised so each block begins as an identity
        mapping — a standard trick that improves early-training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

        # Zero-init the residual branch's last BN (identity start)
        if self._zero_init_residual:
            for module in self.modules():
                if isinstance(module, ResidualBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _ensure_channel_dim(self, x: "torch.Tensor") -> "torch.Tensor":
        """Ensure the input has a channel dimension.

        Accepts ``(B, F, T)`` (no channel) and inserts a channel axis to make
        ``(B, 1, F, T)``, so callers can pass raw 2-D spectrograms directly.

        Args:
            x: Input tensor, 3-D or 4-D.

        Returns:
            4-D tensor ``(B, C, F, T)``.
        """
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the full forward pass.

        Args:
            x: Input batch. Either ``(B, C, F, T)`` or ``(B, F, T)`` — a
               channel axis is inserted automatically when absent.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        x = self._ensure_channel_dim(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the pooled feature embedding before the classifier head.

        Useful for transfer learning, clustering, and anomaly detection where
        the penultimate representation is more informative than the logits.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Feature embedding of shape ``(B, final_channels)``.
        """
        x = self._ensure_channel_dim(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the embedding returned by :meth:`extract_features`.

        Returns:
            Number of channels feeding the classifier head.
        """
        return self._final_channels


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_acoustic_cnn(
    num_classes: int = 5,
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    dropout: float = 0.3,
    stem_channels: int = DEFAULT_STEM_CHANNELS,
    stage_channels: tuple[int, ...] = DEFAULT_STAGE_CHANNELS,
    blocks_per_stage: int = DEFAULT_BLOCKS_PER_STAGE,
    random_seed: int = 42,
) -> "CNNClassifier":
    """Construct a :class:`CNNClassifier` with explicit hyperparameters.

    Args:
        num_classes: Number of fault classes.
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channel count (1 for mel, 3 for mel+Δ+ΔΔ).
        dropout: Dropout probability before the classifier head.
        stem_channels: Channels after the stem convolution.
        stage_channels: Channels per residual stage.
        blocks_per_stage: Residual blocks per stage.
        random_seed: Seed for deterministic construction.

    Returns:
        A constructed :class:`CNNClassifier`.

    Raises:
        RuntimeError: When PyTorch is unavailable.
    """
    config = ModelConfig(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        input_shape=input_shape,
        in_channels=in_channels,
        dropout=dropout,
        random_seed=random_seed,
        extra={
            "stem_channels": stem_channels,
            "stage_channels": list(stage_channels),
            "blocks_per_stage": blocks_per_stage,
            "zero_init_residual": True,
        },
    )
    return CNNClassifier(config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the model summary and a forward-pass smoke test.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Acoustic CNN classifier")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="Export ONNX to this path.")
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_acoustic_cnn(num_classes=args.num_classes)
    if args.summary or True:
        model.print_summary()

    # Forward-pass smoke test
    dummy = torch.randn(2, *model.config.batched_input_shape)
    out = model(dummy)
    logger.info("Forward pass OK: input %s -> output %s",
                tuple(dummy.shape), tuple(out.shape))

    if args.onnx is not None:
        model.export_onnx(args.onnx)
        logger.info("Exported ONNX to %s", args.onnx)

    return 0


if __name__ == "__main__":
    sys.exit(main())