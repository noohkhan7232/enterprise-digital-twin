#!/usr/bin/env python3
"""Production-grade residual network for wind turbine acoustic fault classification.

This module implements :class:`ResNetAcoustic`, a deeper, attention-enhanced
residual network that builds on the same :class:`~src.models.base_model.BaseModel`
foundation as :class:`~src.models.cnn_classifier.CNNClassifier` but targets a
higher accuracy ceiling (**F1 > 0.90**) through three architectural upgrades:

1. **Squeeze-and-Excitation (SE) attention** — each residual block recalibrates
   its channel responses, letting the network emphasise the frequency bands that
   carry a given fault's signature (gear-mesh sidebands, bearing defect
   frequencies, blade-pass tones) and suppress irrelevant channels.
2. **Multi-scale stem** — parallel 3×3 / 5×5 / 7×7 convolutions capture acoustic
   features at several time-frequency resolutions simultaneously, which matters
   because fault signatures span narrow tonal components *and* broadband
   modulation.
3. **Configurable depth** — a ResNet-style ``layers`` specification
   (``[2,2,2,2]`` ≈ ResNet-18) lets the same code scale from a small model for
   20-clip pilots to a deep model for large fleets.

Relationship to ``cnn_classifier.py``
--------------------------------------
``CNNClassifier`` (registry name ``"acoustic_cnn"``) remains the lightweight
baseline.  ``ResNetAcoustic`` (registry name ``"resnet_acoustic"``) is the
higher-capacity production model.  Both coexist in the registry; neither
modifies the other.  This module reuses ``BaseModel``'s entire operational
contract — checkpointing, ONNX export, parameter accounting, device management,
reproducible construction, and ExperimentTracker integration — unchanged.

Feature-representation support
------------------------------
The network is representation-agnostic: it adapts to the channel count and
frequency axis of whatever feature the dataset layer produces.

* **Mel spectrogram** — ``input_shape=(128, T)``, ``in_channels=1``
* **MFCC** — ``input_shape=(40, T)``, ``in_channels=1``
* **CQT** — ``input_shape=(168, T)``, ``in_channels=1``
* **Hybrid stacks** — ``in_channels=3`` (e.g. mel + Δ + ΔΔ) or a combined
  feature passed as a multi-channel tensor.

Dynamic input length is handled by adaptive average pooling before the head, so
the model accepts clips of any duration and exports to ONNX with a dynamic time
axis.

Spatial flow (mel-128 input, resnet18 config)::

    Input               (1,   128, 431)
    Multi-scale stem /2  (64,   64, 216)   3×[3×3,5×5,7×7] → concat → 1×1 fuse
    MaxPool /2           (64,   32, 108)
    Stage 0 (×2)         (64,   32, 108)   SE-residual blocks, stride 1
    Stage 1 (×2, /2)     (128,  16,  54)
    Stage 2 (×2, /2)     (256,   8,  27)
    Stage 3 (×2, /2)     (512,   4,  14)
    AdaptiveAvgPool      (512,   1,   1)
    Dropout + Linear     (num_classes,)

Configuration (via ``ModelConfig.extra``)
------------------------------------------
* ``layers`` (list[int]): blocks per stage (default ``[2, 2, 2, 2]``)
* ``stage_channels`` (list[int]): channels per stage (default ``[64,128,256,512]``)
* ``stem_channels`` (int): channels after the multi-scale stem (default 64)
* ``se_reduction`` (int): SE bottleneck reduction ratio (default 16)
* ``use_se`` (bool): enable SE attention (default True)
* ``multiscale_stem`` (bool): use the multi-scale stem (default True)
* ``zero_init_residual`` (bool): identity-start residual blocks (default True)

Usage::

    from src.models.resnet_acoustic import build_resnet_acoustic
    model = build_resnet_acoustic(num_classes=5, depth="resnet18")
    logits = model(torch.randn(8, 1, 128, 431))   # (8, 5)
    emb = model.extract_features(torch.randn(8, 1, 128, 431))  # (8, 512)

CLI::

    python src/models/resnet_acoustic.py --summary --depth resnet18
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch import Tensor

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseModel, ModelConfig, register_model

logger = logging.getLogger("resnet_acoustic")

# ---------------------------------------------------------------------------
# Constants / depth presets
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "resnet_acoustic"

DEFAULT_STEM_CHANNELS:  Final[int] = 64
DEFAULT_STAGE_CHANNELS: Final[tuple[int, ...]] = (64, 128, 256, 512)
DEFAULT_LAYERS:         Final[tuple[int, ...]] = (2, 2, 2, 2)
DEFAULT_SE_REDUCTION:   Final[int] = 16

#: Named depth presets (``layers`` per stage).
DEPTH_PRESETS: Final[dict[str, tuple[int, ...]]] = {
    "resnet10": (1, 1, 1, 1),
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}

#: Multi-scale stem kernel sizes (each path same-padded to align outputs).
_STEM_KERNELS: Final[tuple[int, ...]] = (3, 5, 7)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation channel-attention block.

        Recalibrates channel-wise feature responses by (1) *squeezing* spatial
        information into a per-channel descriptor via global average pooling,
        then (2) learning per-channel *excitation* weights through a small
        bottleneck MLP, and (3) rescaling the input channels by those weights.

        Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018).

        Args:
            channels: Number of input/output channels.
            reduction: Bottleneck reduction ratio (channels // reduction).
        """

        def __init__(self, channels: int, reduction: int = DEFAULT_SE_REDUCTION) -> None:
            super().__init__()
            bottleneck = max(1, channels // reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, bottleneck, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, channels, bias=True),
                nn.Sigmoid(),
            )

        def forward(self, x: Tensor) -> Tensor:
            """Apply channel attention.

            Args:
                x: Input tensor ``(B, C, H, W)``.

            Returns:
                Channel-reweighted tensor of the same shape.
            """
            b, c, _, _ = x.shape
            s = self.pool(x).view(b, c)
            w = self.fc(s).view(b, c, 1, 1)
            return x * w

    class ResidualBlock(nn.Module):
        """SE-enhanced basic residual block.

        Two 3×3 convolutions (each followed by BatchNorm), an optional
        Squeeze-and-Excitation recalibration, and a skip connection.  When the
        stride or channel count changes, the skip path applies a 1×1
        convolution + BatchNorm to match dimensions.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Stride of the first convolution (2 downsamples spatially).
            use_se: Insert an :class:`SEBlock` before the residual addition.
            se_reduction: SE bottleneck reduction ratio.
            zero_init_bn: Zero-init the second BN so the block starts as identity.
        """

        expansion: int = 1

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            use_se: bool = True,
            se_reduction: int = DEFAULT_SE_REDUCTION,
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
            self.se = (
                SEBlock(out_channels, se_reduction) if use_se else None
            )

            # self.downsample: nn.Module | None = None
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1,
                        stride=stride, bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            self._zero_init_bn = zero_init_bn

        def forward(self, x: Tensor) -> Tensor:
            """Run the residual block.

            Args:
                x: Input tensor ``(B, in_channels, H, W)``.

            Returns:
                Output tensor ``(B, out_channels, H', W')``.
            """
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.se is not None:
                out = self.se(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return self.relu(out)

    class MultiScaleStem(nn.Module):
        """Multi-scale convolutional stem.

        Applies parallel convolutions at several kernel sizes (3×3, 5×5, 7×7),
        each same-padded and strided identically so their outputs align, then
        concatenates and fuses them with a 1×1 convolution.  This captures
        acoustic structure at multiple time-frequency resolutions in the first
        layer.

        Args:
            in_channels: Input channel count.
            out_channels: Fused output channel count.
            stride: Stride applied by every path.
        """

        def __init__(
            self, in_channels: int, out_channels: int, stride: int = 2
        ) -> None:
            super().__init__()
            per_path = max(1, out_channels // len(_STEM_KERNELS))
            self.paths = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, per_path, kernel_size=k,
                        stride=stride, padding=k // 2, bias=False,
                    ),
                    nn.BatchNorm2d(per_path),
                    nn.ReLU(inplace=True),
                )
                for k in _STEM_KERNELS
            ])
            fused = per_path * len(_STEM_KERNELS)
            self.fuse = nn.Sequential(
                nn.Conv2d(fused, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            """Run the multi-scale stem.

            Args:
                x: Input tensor ``(B, in_channels, H, W)``.

            Returns:
                Fused tensor ``(B, out_channels, H', W')``.
            """
            feats = [path(x) for path in self.paths]
            return self.fuse(torch.cat(feats, dim=1))

else:  # pragma: no cover
    SEBlock = None  # type: ignore[assignment,misc]
    ResidualBlock = None  # type: ignore[assignment,misc]
    MultiScaleStem = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# ResNetAcoustic
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class ResNetAcoustic(BaseModel):
    """SE-ResNet for wind-turbine acoustic fault classification.

    A configurable-depth residual network with Squeeze-and-Excitation attention
    and a multi-scale stem, optimised for spectrogram-like acoustic features.
    Inherits the full lifecycle contract from
    :class:`~src.models.base_model.BaseModel`.

    Architecture is configured through ``ModelConfig.extra`` (see module
    docstring).  Defaults correspond to an SE-ResNet-18.

    Args:
        config: Model configuration.  ``input_shape`` is the feature shape
            ``(F, T)`` (mel/MFCC/CQT) and ``in_channels`` the channel count
            (1 for single-feature, 3 for hybrid stacks).

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
    """

    def build_layers(self) -> None:
        """Construct the stem, SE-residual stages, and classifier head.

        Reads architecture hyperparameters from ``self.config.extra`` and
        applies weight initialisation after construction.
        """
        cfg = self.config
        extra = cfg.extra or {}

        layers: list[int] = list(extra.get("layers", DEFAULT_LAYERS))
        stage_channels: list[int] = list(
            extra.get("stage_channels", DEFAULT_STAGE_CHANNELS)
        )
        stem_channels: int = int(extra.get("stem_channels", DEFAULT_STEM_CHANNELS))
        self._use_se: bool = bool(extra.get("use_se", True))
        self._se_reduction: int = int(extra.get("se_reduction", DEFAULT_SE_REDUCTION))
        self._multiscale: bool = bool(extra.get("multiscale_stem", True))
        self._zero_init_residual: bool = bool(extra.get("zero_init_residual", True))

        if len(layers) != len(stage_channels):
            raise ValueError(
                f"layers ({len(layers)}) and stage_channels "
                f"({len(stage_channels)}) must have equal length"
            )

        in_ch = cfg.in_channels

        # ── Stem (multi-scale or plain) ──────────────────────────────────────
        if self._multiscale:
            self.stem = MultiScaleStem(in_ch, stem_channels, stride=2)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, stem_channels, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual stages ──────────────────────────────────────────────────
        # stages: list[nn.Module] = []
        stages = []
        current = stem_channels
        for stage_idx, (out_ch, n_blocks) in enumerate(
            zip(stage_channels, layers)
        ):
            stride = 1 if stage_idx == 0 else 2
            # blocks: list[nn.Module] = []
            blocks = []
            for block_idx in range(n_blocks):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(
                    ResidualBlock(
                        current, out_ch, stride=block_stride,
                        use_se=self._use_se, se_reduction=self._se_reduction,
                        zero_init_bn=self._zero_init_residual,
                    )
                )
                current = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self._final_channels = current

        # ── Head ──────────────────────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(current, cfg.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming init to convs and normal init to linear layers.

        With ``zero_init_residual`` set, each residual block's second
        BatchNorm is zero-initialised so the block starts as an identity
        mapping, improving early-training stability for deep configurations.
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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self._zero_init_residual:
            for module in self.modules():
                if isinstance(module, ResidualBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _ensure_channel_dim(self, x: Tensor) -> Tensor:
        """Insert a channel dimension when the input is 3-D.

        Args:
            x: Input tensor, 3-D ``(B, F, T)`` or 4-D ``(B, C, F, T)``.

        Returns:
            4-D tensor ``(B, C, F, T)``.
        """
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Run the full forward pass.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        x = self._ensure_channel_dim(x)
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def extract_features(self, x: Tensor) -> Tensor:
        """Return the pooled embedding before the classifier head.

        Useful for transfer learning, clustering, anomaly detection, and the
        digital-twin similarity search the platform roadmap calls for.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Embedding of shape ``(B, final_channels)``.
        """
        x = self._ensure_channel_dim(x)
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the :meth:`extract_features` embedding.

        Returns:
            Number of channels feeding the classifier head.
        """
        return self._final_channels


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_resnet_acoustic(
    num_classes: int = 5,
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    depth: str = "resnet18",
    layers: tuple[int, ...] | None = None,
    stage_channels: tuple[int, ...] = DEFAULT_STAGE_CHANNELS,
    stem_channels: int = DEFAULT_STEM_CHANNELS,
    dropout: float = 0.3,
    use_se: bool = True,
    se_reduction: int = DEFAULT_SE_REDUCTION,
    multiscale_stem: bool = True,
    random_seed: int = 42,
) -> ResNetAcoustic:
    """Construct a :class:`ResNetAcoustic` with explicit hyperparameters.

    Args:
        num_classes: Number of fault classes.
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channels (1 for mel/MFCC/CQT, 3 for hybrid stacks).
        depth: Named preset (``resnet10`` | ``resnet18`` | ``resnet34``).
            Ignored when ``layers`` is given explicitly.
        layers: Explicit blocks-per-stage; overrides ``depth`` when provided.
        stage_channels: Channels per stage.
        stem_channels: Channels after the stem.
        dropout: Dropout probability before the head.
        use_se: Enable Squeeze-and-Excitation attention.
        se_reduction: SE bottleneck reduction ratio.
        multiscale_stem: Use the multi-scale stem.
        random_seed: Seed for deterministic construction.

    Returns:
        A constructed :class:`ResNetAcoustic`.

    Raises:
        ValueError: When ``depth`` is unknown and ``layers`` is not given.
        RuntimeError: When PyTorch is unavailable.
    """
    if layers is None:
        if depth not in DEPTH_PRESETS:
            raise ValueError(
                f"Unknown depth '{depth}'. Choose from {list(DEPTH_PRESETS)} "
                "or pass layers explicitly."
            )
        layers = DEPTH_PRESETS[depth]

    config = ModelConfig(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        input_shape=input_shape,
        in_channels=in_channels,
        dropout=dropout,
        random_seed=random_seed,
        extra={
            "layers": list(layers),
            "stage_channels": list(stage_channels),
            "stem_channels": stem_channels,
            "use_se": use_se,
            "se_reduction": se_reduction,
            "multiscale_stem": multiscale_stem,
            "zero_init_residual": True,
        },
    )
    return ResNetAcoustic(config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print summary and run a forward-pass smoke test.

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

    parser = argparse.ArgumentParser(description="SE-ResNet acoustic classifier")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--depth", default="resnet18",
                        choices=list(DEPTH_PRESETS))
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_resnet_acoustic(
        num_classes=args.num_classes, depth=args.depth,
        in_channels=args.in_channels,
    )
    if args.summary or True:
        model.print_summary()

    dummy = torch.randn(2, *model.config.batched_input_shape)
    out = model(dummy)
    logger.info("Forward pass OK: %s -> %s | embedding dim %d",
                tuple(dummy.shape), tuple(out.shape), model.feature_dim)

    if args.onnx is not None:
        model.export_onnx(args.onnx)
        logger.info("Exported ONNX to %s", args.onnx)

    return 0


if __name__ == "__main__":
    sys.exit(main())