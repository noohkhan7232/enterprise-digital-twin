#!/usr/bin/env python3
"""Vision-Transformer encoder for wind turbine acoustic fault classification.

This module implements :class:`AcousticTransformer`, a ViT-style transformer
that treats a time-frequency spectrogram as a grid of patches, embeds each
patch as a token, and classifies the clip from a stack of transformer encoder
layers.  It is the attention-native counterpart to the convolutional and
recurrent models in this package and completes the model zoo with a pure
self-attention architecture suitable for research publication.

It builds on two already-shipped, independently-tested foundations rather than
reimplementing them:

* :class:`~src.models.base_model.BaseModel` — the full lifecycle contract
  (checkpointing, ONNX export, parameter accounting, device management,
  reproducible construction, ExperimentTracker integration), and
* :class:`~src.models.attention_modules.SelfAttentionBlock` — the multi-head
  self-attention + residual/LayerNorm + GELU-feedforward encoder layer, reused
  directly as the transformer block.

Registered as ``"acoustic_transformer"``, it coexists with every other model
(``acoustic_cnn``, ``resnet_acoustic``, ``cnn_bilstm``,
``cnn_bilstm_attention``, ``anomaly_autoencoder``) and is driven by the shared
:class:`~src.training.trainer.Trainer` through the registry by name.

Architecture
------------
::

    Input  (B, C, F, T)
      │  Patch embedding: Conv2d(C, embed_dim, kernel=patch, stride=patch)
      ▼
    (B, embed_dim, F/patch, T/patch)
      │  flatten spatial grid → token sequence
      ▼
    (B, N, embed_dim)              N = (F/patch)·(T/patch) patch tokens
      │  + sinusoidal positional encoding (length-independent)
      ▼
    Transformer encoder stack ×L   each: MHA → +residual → LN
      │                                   FFN(GELU) → +residual → LN
      ▼
    (B, N, embed_dim)
      │  global mean pooling over tokens
      ▼
    (B, embed_dim)
      │  LayerNorm → Dropout → Linear
      ▼
    (B, num_classes)

Why sinusoidal positional encoding?
-----------------------------------
Learned positional embeddings are fixed to a single sequence length, which
would break the **dynamic input length** requirement — clips of different
durations produce different patch counts.  Sinusoidal encodings are defined
per-position and therefore length-independent: a position-``p`` token always
receives the same encoding regardless of total length.  The encoding is
precomputed once into a registered buffer (so it moves with ``.to(device)`` and
is captured in checkpoints) and sliced to the actual token count at runtime —
an ONNX-exportable operation.  Longer-than-buffer inputs fall back to on-the-fly
computation, so correctness never depends on the buffer size.

Feature-representation support
------------------------------
Representation-agnostic via ``in_channels`` / ``input_shape``: mel ``(128,T)``,
MFCC ``(40,T)``, CQT ``(168,T)`` run single-channel; hybrid stacks run with
``in_channels=3``.  The patch convolution adapts to any frequency/time size; a
build-time check ensures the patch size does not exceed the frequency axis.

Configuration (via ``ModelConfig.extra``)
------------------------------------------
* ``embed_dim`` (int): token / model dimension (default 192)
* ``num_heads`` (int): attention heads; must divide ``embed_dim`` (default 6)
* ``num_layers`` (int): transformer encoder layers (default 6)
* ``mlp_ratio`` (float): feed-forward width as a multiple of ``embed_dim``
  (default 4.0)
* ``dropout`` (float): dropout throughout (default 0.1)
* ``patch_size`` (int): square patch edge in time-frequency bins (default 16)
* ``max_positions`` (int): precomputed positional-encoding length (default 4096)

Usage::

    from src.models.acoustic_transformer import build_acoustic_transformer
    model = build_acoustic_transformer(num_classes=5, embed_dim=192, num_heads=6)
    logits = model(torch.randn(8, 1, 128, 431))               # (8, 5)
    emb = model.extract_features(torch.randn(8, 1, 128, 431))   # (8, 192)
    maps = model.attention_maps(torch.randn(8, 1, 128, 431))    # list of (B,N,N)

CLI::

    python src/models/acoustic_transformer.py --summary
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Final

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

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseModel, ModelConfig, register_model

# Reuse the transformer encoder layer from the attention framework.
if _TORCH_AVAILABLE:
    from src.models.attention_modules import SelfAttentionBlock
else:  # pragma: no cover
    SelfAttentionBlock = None  # type: ignore[assignment,misc]

logger = logging.getLogger("acoustic_transformer")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "acoustic_transformer"

DEFAULT_EMBED_DIM:     Final[int] = 192
DEFAULT_NUM_HEADS:     Final[int] = 6
DEFAULT_NUM_LAYERS:    Final[int] = 6
DEFAULT_MLP_RATIO:     Final[float] = 4.0
DEFAULT_DROPOUT:       Final[float] = 0.1
DEFAULT_PATCH_SIZE:    Final[int] = 16
DEFAULT_MAX_POSITIONS: Final[int] = 4096


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class PatchEmbedding(nn.Module):
        """Convolutional patch embedding for spectrogram inputs.

        Splits a ``(B, C, F, T)`` feature map into non-overlapping square
        patches and projects each to ``embed_dim`` with a strided convolution
        (the standard ViT patch-embedding trick), then flattens the spatial
        grid into a token sequence.

        Args:
            in_channels: Input channel count.
            embed_dim: Output token dimension.
            patch_size: Square patch edge length.
        """

        def __init__(
            self, in_channels: int, embed_dim: int, patch_size: int
        ) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size,
            )

        def forward(
            self, x: "torch.Tensor"
        ) -> "tuple[torch.Tensor, tuple[int, int]]":
            """Embed patches and flatten to a token sequence.

            Args:
                x: Input feature map ``(B, C, F, T)``.

            Returns:
                Tuple ``(tokens, grid)`` where *tokens* is ``(B, N, embed_dim)``
                with ``N = (F // patch) * (T // patch)`` and *grid* is the
                ``(F', T')`` patch-grid shape (useful for reshaping attention
                maps back to a 2-D layout).
            """
            x = self.proj(x)                      # (B, E, F', T')
            b, e, fp, tp = x.shape
            # (B, E, F', T') -> (B, F'*T', E); row-major over (F', T')
            return x.flatten(2).transpose(1, 2), (fp, tp)

    class SinusoidalPositionalEncoding(nn.Module):
        """Length-independent sinusoidal positional encoding.

        Precomputes encodings up to ``max_positions`` into a registered buffer
        and slices to the actual sequence length at runtime; longer sequences
        are computed on the fly.  Being position-based (not length-based), the
        encoding supports dynamic input length and exports to ONNX.

        Args:
            embed_dim: Token dimension.
            max_positions: Precomputed buffer length.
        """

        def __init__(self, embed_dim: int, max_positions: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.register_buffer(
                "pe", self._build(max_positions, embed_dim), persistent=False
            )

        @staticmethod
        def _build(n_pos: int, dim: int) -> "torch.Tensor":
            """Construct a sinusoidal positional-encoding table.

            Args:
                n_pos: Number of positions.
                dim: Embedding dimension.

            Returns:
                Encoding tensor ``(n_pos, dim)``.
            """
            pe = torch.zeros(n_pos, dim)
            position = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            return pe

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Add positional encodings to a token sequence.

            Args:
                x: Token sequence ``(B, N, embed_dim)``.

            Returns:
                Position-encoded sequence ``(B, N, embed_dim)``.
            """
            n = x.shape[1]
            if n <= self.pe.shape[0]:
                pe = self.pe[:n]
            else:  # pragma: no cover - rare very-long inputs
                pe = self._build(n, self.embed_dim).to(x.device, x.dtype)
            return x + pe.unsqueeze(0).to(x.dtype)

else:  # pragma: no cover
    PatchEmbedding = None  # type: ignore[assignment,misc]
    SinusoidalPositionalEncoding = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# AcousticTransformer
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class AcousticTransformer(BaseModel):
    """Vision-Transformer encoder for acoustic fault classification.

    Embeds spectrogram patches as tokens, adds sinusoidal positional encodings,
    refines them through a stack of transformer encoder layers (reused from
    :mod:`src.models.attention_modules`), mean-pools over tokens, and
    classifies.  Inherits the full lifecycle contract from
    :class:`~src.models.base_model.BaseModel`.

    Configuration is read from ``ModelConfig.extra`` (see module docstring).

    Args:
        config: Model configuration. ``input_shape`` is the feature shape
            ``(F, T)`` and ``in_channels`` the channel count.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
        ValueError: When ``embed_dim`` is not divisible by ``num_heads`` or the
            patch size exceeds the frequency axis.
    """

    def build_layers(self) -> None:
        """Construct patch embedding, positional encoding, encoder stack, head.

        Raises:
            ValueError: When ``embed_dim`` is not divisible by ``num_heads`` or
                the patch size exceeds the input frequency dimension.
        """
        cfg = self.config
        extra = cfg.extra or {}

        self._embed_dim: int = int(extra.get("embed_dim", DEFAULT_EMBED_DIM))
        self._num_heads: int = int(extra.get("num_heads", DEFAULT_NUM_HEADS))
        self._num_layers: int = int(extra.get("num_layers", DEFAULT_NUM_LAYERS))
        self._mlp_ratio: float = float(extra.get("mlp_ratio", DEFAULT_MLP_RATIO))
        dropout: float = float(extra.get("dropout", cfg.dropout))
        self._patch_size: int = int(extra.get("patch_size", DEFAULT_PATCH_SIZE))
        max_positions: int = int(extra.get("max_positions", DEFAULT_MAX_POSITIONS))

        if self._embed_dim % self._num_heads != 0:
            raise ValueError(
                f"embed_dim ({self._embed_dim}) must be divisible by "
                f"num_heads ({self._num_heads})"
            )
        freq_dim = cfg.input_shape[-2] if len(cfg.input_shape) >= 2 else cfg.input_shape[0]
        if self._patch_size > freq_dim:
            raise ValueError(
                f"patch_size ({self._patch_size}) cannot exceed the frequency "
                f"dimension ({freq_dim})"
            )

        # ── Patch embedding + positional encoding ────────────────────────────
        self.patch_embed = PatchEmbedding(
            cfg.in_channels, self._embed_dim, self._patch_size
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            self._embed_dim, max_positions
        )
        self.pos_dropout = nn.Dropout(dropout)

        # ── Transformer encoder stack (reused SelfAttentionBlock) ─────────────
        ff_dim = int(self._embed_dim * self._mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim=self._embed_dim,
                num_heads=self._num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(self._num_layers)
        ])

        # ── Head ──────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(self._embed_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._embed_dim, cfg.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialise patch-conv (Kaiming), linears (Xavier), norms (1/0)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _ensure_channel_dim(self, x: "torch.Tensor") -> "torch.Tensor":
        """Insert a channel dimension when the input is 3-D.

        Args:
            x: Input tensor, 3-D ``(B, F, T)`` or 4-D ``(B, C, F, T)``.

        Returns:
            4-D tensor ``(B, C, F, T)``.
        """
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def _tokenize(self, x: "torch.Tensor") -> "torch.Tensor":
        """Patch-embed and position-encode an input to a token sequence.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Position-encoded token sequence ``(B, N, embed_dim)``.
        """
        x = self._ensure_channel_dim(x)
        tokens, _ = self.patch_embed(x)
        tokens = self.pos_encoding(tokens)
        return self.pos_dropout(tokens)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the full forward pass.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        tokens = self._tokenize(x)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)               # global mean pooling
        pooled = self.head_dropout(pooled)
        return self.classifier(pooled)

    def extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the pooled embedding before the classifier head.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Embedding of shape ``(B, embed_dim)``.
        """
        tokens = self._tokenize(x)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.mean(dim=1)

    def attention_maps(self, x: "torch.Tensor") -> "list[torch.Tensor]":
        """Return per-layer self-attention maps for interpretability.

        Each map shows token-to-token attention, revealing which time-frequency
        patches the model relates — valuable for diagnostics and for the
        explainability expected of a deployed industrial model.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            A list of ``num_layers`` tensors, each ``(B, N, N)`` (head-averaged).
        """
        tokens = self._tokenize(x)
        maps: list[torch.Tensor] = []
        for layer in self.encoder_layers:
            tokens, weights = layer(tokens, return_weights=True)
            maps.append(weights)
        return maps

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the :meth:`extract_features` embedding.

        Returns:
            The transformer embedding dimension.
        """
        return self._embed_dim

    @property
    def num_heads(self) -> int:
        """Number of attention heads per encoder layer.

        Returns:
            The configured head count.
        """
        return self._num_heads

    @property
    def num_layers(self) -> int:
        """Number of transformer encoder layers.

        Returns:
            The configured layer count.
        """
        return self._num_layers


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_acoustic_transformer(
    num_classes: int = 5,
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    mlp_ratio: float = DEFAULT_MLP_RATIO,
    dropout: float = DEFAULT_DROPOUT,
    patch_size: int = DEFAULT_PATCH_SIZE,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    random_seed: int = 42,
) -> "AcousticTransformer":
    """Construct an :class:`AcousticTransformer` with explicit hyperparameters.

    Args:
        num_classes: Number of fault classes.
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channels (1 for mel/MFCC/CQT, 3 for hybrid stacks).
        embed_dim: Token / model dimension.
        num_heads: Attention heads; must divide ``embed_dim``.
        num_layers: Transformer encoder layers.
        mlp_ratio: Feed-forward width as a multiple of ``embed_dim``.
        dropout: Dropout throughout.
        patch_size: Square patch edge length.
        max_positions: Precomputed positional-encoding length.
        random_seed: Seed for deterministic construction.

    Returns:
        A constructed :class:`AcousticTransformer`.

    Raises:
        ValueError: When ``embed_dim`` is not divisible by ``num_heads`` or the
            patch size exceeds the frequency axis.
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
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
            "patch_size": patch_size,
            "max_positions": max_positions,
        },
    )
    return AcousticTransformer(config)


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

    parser = argparse.ArgumentParser(description="Acoustic transformer classifier")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_acoustic_transformer(
        num_classes=args.num_classes, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        patch_size=args.patch_size, in_channels=args.in_channels,
    )
    if args.summary or True:
        model.print_summary()

    dummy = torch.randn(2, *model.config.batched_input_shape)
    out = model(dummy)
    logger.info(
        "Forward pass OK: %s -> %s | embed %d | heads %d | layers %d",
        tuple(dummy.shape), tuple(out.shape),
        model.feature_dim, model.num_heads, model.num_layers,
    )
    maps = model.attention_maps(dummy)
    logger.info("Attention maps: %d layers, each %s",
                len(maps), tuple(maps[0].shape))

    if args.onnx is not None:
        model.export_onnx(args.onnx)
        logger.info("Exported ONNX to %s", args.onnx)

    return 0


if __name__ == "__main__":
    sys.exit(main())