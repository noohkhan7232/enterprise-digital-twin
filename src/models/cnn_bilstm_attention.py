#!/usr/bin/env python3
"""CNN-BiLSTM with multi-head self-attention for temporal acoustic fault diagnosis.

This module implements :class:`CNNBiLSTMAttention`, the most expressive temporal
model in the project.  It composes three already-shipped, independently-tested
building blocks rather than reimplementing them:

* the **CNN feature extractor + BiLSTM** pattern from
  :mod:`src.models.cnn_bilstm`,
* the **multi-head self-attention** block (:class:`SelfAttentionBlock`) and
  **attention pooling** (:class:`AttentionPooling`) from
  :mod:`src.models.attention_modules`.

It inherits the full operational contract from
:class:`~src.models.base_model.BaseModel` (checkpointing, ONNX export,
parameter accounting, device management, reproducible construction,
ExperimentTracker integration) and is registered as ``"cnn_bilstm_attention"``.

Architecture
------------
::

    Input  (B, 1, F, T)
      │
      ▼  CNN feature extractor  (freq downsampled, time preserved)
    (B, C, F', T')
      │  AdaptiveAvgPool(freq) + reshape
      ▼
    (B, T', C·freq_pool)          per-frame feature sequence
      │  BiLSTM (configurable depth)
      ▼
    (B, T', 2·hidden)             temporally-contextualised sequence
      │  Multi-head self-attention refinement  (configurable heads, seq→seq)
      ▼
    (B, T', 2·hidden)             globally re-weighted sequence
      │  Attention pooling  (seq→vector)
      ▼
    (B, 2·hidden)                 attention-summarised embedding
      │  Dropout + Linear
      ▼
    (B, num_classes)

Why this should outperform ``cnn_bilstm`` and ``resnet_acoustic``
----------------------------------------------------------------
``cnn_bilstm`` already keeps the time axis, but its single additive-attention
pooling can only express *one* view of which time steps matter.  Inserting a
**multi-head self-attention** stage between the BiLSTM and the pooling lets each
time step attend to every other time step before pooling — so a fault impact at
*t* can be related to its echoes and modulation elsewhere in the clip — and the
multiple heads capture several such relational patterns in parallel.  The
attention-pooling stage then collapses the refined sequence with a learned,
mask-aware weighting.  Against ``resnet_acoustic`` (which discards temporal
order via global pooling), this model retains and *models* the temporal
structure that non-stationary faults exhibit, which is exactly where the macro-F1
gain is expected.

The chain is well-typed end to end: the self-attention stage is
sequence-to-sequence ``(B,T,D) → (B,T,D)`` so the pooling stage that follows
``(B,T,D) → (B,D)`` composes cleanly — and the multi-head count remains fully
configurable.

ONNX / variable length / mixed precision
----------------------------------------
The CNN preserves a proportional time axis, the LSTM and self-attention accept
any sequence length, and attention pooling reduces any length to a fixed
vector — so the model handles clips of any duration and exports to ONNX with a
dynamic time axis (opset 17 supports LSTM, attention, and softmax).  No dtype is
hardcoded in the forward path, and the attention framework computes softmax in
fp32, so the model is mixed-precision safe.

Configuration (via ``ModelConfig.extra``)
------------------------------------------
* ``cnn_channels`` (list[int]): channels per CNN block (default ``[32,64,128,128]``)
* ``freq_pool`` (int): adaptive frequency size before the LSTM (default 4)
* ``lstm_hidden`` (int): LSTM hidden size per direction (default 128)
* ``lstm_layers`` (int): stacked BiLSTM layers (default 2)
* ``lstm_dropout`` (float): dropout between LSTM layers (default 0.2)
* ``num_heads`` (int): self-attention heads; must divide ``2·lstm_hidden``
  (default 8)
* ``attention_ff_dim`` (int | None): self-attention feed-forward width
  (default ``4·embed_dim``)
* ``attention_dropout`` (float): self-attention dropout (default 0.1)
* ``pool_attention_dim`` (int): attention-pooling scoring hidden size (default 128)
* ``bidirectional`` (bool): bidirectional LSTM (default True)

Usage::

    from src.models.cnn_bilstm_attention import build_cnn_bilstm_attention
    model = build_cnn_bilstm_attention(num_classes=5, num_heads=8)
    logits = model(torch.randn(8, 1, 128, 431))               # (8, 5)
    emb = model.extract_features(torch.randn(8, 1, 128, 431))   # (8, 256)
    maps = model.attention_maps(torch.randn(8, 1, 128, 431))    # dict of weights

CLI::

    python src/models/cnn_bilstm_attention.py --summary
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

# Reuse the attention framework (import guarded so the module still loads
# without torch for config/registry tooling).
if _TORCH_AVAILABLE:
    from src.models.attention_modules import AttentionPooling, SelfAttentionBlock
else:  # pragma: no cover
    AttentionPooling = None  # type: ignore[assignment,misc]
    SelfAttentionBlock = None  # type: ignore[assignment,misc]

logger = logging.getLogger("cnn_bilstm_attention")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "cnn_bilstm_attention"

DEFAULT_CNN_CHANNELS:   Final[tuple[int, ...]] = (32, 64, 128, 128)
DEFAULT_FREQ_POOL:      Final[int] = 4
DEFAULT_LSTM_HIDDEN:    Final[int] = 128
DEFAULT_LSTM_LAYERS:    Final[int] = 2
DEFAULT_LSTM_DROPOUT:   Final[float] = 0.2
DEFAULT_NUM_HEADS:      Final[int] = 8
DEFAULT_ATTENTION_DROPOUT: Final[float] = 0.1
DEFAULT_POOL_ATTENTION_DIM: Final[int] = 128


# ---------------------------------------------------------------------------
# Building block (CNN extractor reused from the cnn_bilstm pattern)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _ConvBlock(nn.Module):
        """Conv → BN → ReLU → MaxPool, with configurable per-axis pooling.

        Later blocks pool frequency only (``time_pool=1``) so the time axis is
        preserved for the recurrent / attention stages.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            freq_pool: Pooling stride along the frequency axis.
            time_pool: Pooling stride along the time axis.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            freq_pool: int = 2,
            time_pool: int = 2,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(
                kernel_size=(freq_pool, time_pool),
                stride=(freq_pool, time_pool),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Run the conv block.

            Args:
                x: Input tensor ``(B, C, F, T)``.

            Returns:
                Output tensor ``(B, C_out, F', T')``.
            """
            return self.pool(self.relu(self.bn(self.conv(x))))

else:  # pragma: no cover
    _ConvBlock = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# CNNBiLSTMAttention
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class CNNBiLSTMAttention(BaseModel):
    """CNN-BiLSTM with multi-head self-attention and attention pooling.

    Composes a CNN feature extractor, a bidirectional LSTM, a multi-head
    self-attention refinement (from :mod:`src.models.attention_modules`), and a
    mask-aware attention-pooling stage, followed by a linear classifier.
    Inherits the full lifecycle contract from
    :class:`~src.models.base_model.BaseModel`.

    Configuration is read from ``ModelConfig.extra`` (see module docstring).

    Args:
        config: Model configuration. ``input_shape`` is the feature shape
            ``(F, T)`` and ``in_channels`` the channel count.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
        ValueError: When ``2·lstm_hidden`` is not divisible by ``num_heads``.
    """

    def build_layers(self) -> None:
        """Construct the CNN, BiLSTM, self-attention, pooling, and head.

        Reads architecture hyperparameters from ``self.config.extra`` and
        applies weight initialisation after construction.

        Raises:
            ValueError: When the self-attention embedding dimension
                (``2·lstm_hidden`` if bidirectional) is not divisible by
                ``num_heads``.
        """
        cfg = self.config
        extra = cfg.extra or {}

        cnn_channels: list[int] = list(extra.get("cnn_channels", DEFAULT_CNN_CHANNELS))
        self._freq_pool: int = int(extra.get("freq_pool", DEFAULT_FREQ_POOL))
        lstm_hidden: int = int(extra.get("lstm_hidden", DEFAULT_LSTM_HIDDEN))
        lstm_layers: int = int(extra.get("lstm_layers", DEFAULT_LSTM_LAYERS))
        lstm_dropout: float = float(extra.get("lstm_dropout", DEFAULT_LSTM_DROPOUT))
        num_heads: int = int(extra.get("num_heads", DEFAULT_NUM_HEADS))
        attention_dropout: float = float(
            extra.get("attention_dropout", DEFAULT_ATTENTION_DROPOUT)
        )
        attention_ff_dim = extra.get("attention_ff_dim", None)
        pool_attention_dim: int = int(
            extra.get("pool_attention_dim", DEFAULT_POOL_ATTENTION_DIM)
        )
        self._bidirectional: bool = bool(extra.get("bidirectional", True))

        # ── CNN feature extractor ────────────────────────────────────────────
        blocks: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i, out_ch in enumerate(cnn_channels):
            time_pool = 2 if i < 2 else 1
            blocks.append(_ConvBlock(in_ch, out_ch, freq_pool=2, time_pool=time_pool))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self._cnn_out_channels = in_ch
        self.freq_pool = nn.AdaptiveAvgPool2d((self._freq_pool, None))
        self._seq_feature_dim = self._cnn_out_channels * self._freq_pool

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._seq_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=self._bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self._embed_dim = lstm_hidden * (2 if self._bidirectional else 1)

        # ── Multi-head self-attention refinement (seq → seq) ──────────────────
        if self._embed_dim % num_heads != 0:
            raise ValueError(
                f"Self-attention embed_dim ({self._embed_dim} = "
                f"{'2*' if self._bidirectional else ''}lstm_hidden) must be "
                f"divisible by num_heads ({num_heads}). Adjust lstm_hidden or "
                "num_heads."
            )
        self.self_attention = SelfAttentionBlock(
            embed_dim=self._embed_dim,
            num_heads=num_heads,
            ff_dim=attention_ff_dim,
            dropout=attention_dropout,
        )

        # ── Attention pooling (seq → vector) ──────────────────────────────────
        self.attention_pool = AttentionPooling(
            input_dim=self._embed_dim,
            attention_dim=pool_attention_dim,
        )

        # ── Head ──────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(self._embed_dim, cfg.num_classes)

        self._num_heads = num_heads
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialise CNN, LSTM, and head weights.

        The self-attention and pooling submodules initialise themselves; here
        we cover the CNN (Kaiming), the LSTM (Xavier input / orthogonal
        recurrent), and the classifier (Xavier).
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        # Classifier head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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

    def _to_sequence(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the CNN extractor and reshape into a time sequence.

        Args:
            x: Input tensor ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Sequence tensor ``(B, T', C·freq_pool)``.
        """
        x = self._ensure_channel_dim(x)
        x = self.cnn(x)
        x = self.freq_pool(x)                  # (B, C, freq_pool, T')
        b, c, f, t = x.shape
        return x.permute(0, 3, 1, 2).reshape(b, t, c * f)

    def _encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run CNN → BiLSTM → self-attention to get the refined sequence.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Refined sequence ``(B, T', embed_dim)``.
        """
        seq = self._to_sequence(x)
        lstm_out, _ = self.lstm(seq)           # (B, T', embed_dim)
        return self.self_attention(lstm_out)   # (B, T', embed_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the full forward pass.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        refined = self._encode(x)
        pooled = self.attention_pool(refined)  # (B, embed_dim)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the attention-pooled embedding before the classifier head.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Embedding of shape ``(B, embed_dim)``.
        """
        refined = self._encode(x)
        return self.attention_pool(refined)

    def attention_maps(self, x: "torch.Tensor") -> "dict[str, torch.Tensor]":
        """Return attention weights from both attention stages.

        Provides interpretability into *where* the model attends: the
        self-attention matrix shows time-step-to-time-step relationships, and
        the pooling weights show which time steps dominate the final embedding.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Dictionary with:
            ``"self_attention"`` → ``(B, T', T')`` head-averaged attention,
            ``"pooling"`` → ``(B, T')`` pooling weights.
        """
        seq = self._to_sequence(x)
        lstm_out, _ = self.lstm(seq)
        refined, self_attn_w = self.self_attention(lstm_out, return_weights=True)
        _, pool_w = self.attention_pool(refined, return_weights=True)
        return {"self_attention": self_attn_w, "pooling": pool_w}

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the :meth:`extract_features` embedding.

        Returns:
            The embedding dimension (``2·lstm_hidden`` if bidirectional).
        """
        return self._embed_dim

    @property
    def num_heads(self) -> int:
        """Number of self-attention heads.

        Returns:
            The configured head count.
        """
        return self._num_heads


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_cnn_bilstm_attention(
    num_classes: int = 5,
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    cnn_channels: tuple[int, ...] = DEFAULT_CNN_CHANNELS,
    freq_pool: int = DEFAULT_FREQ_POOL,
    lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
    lstm_layers: int = DEFAULT_LSTM_LAYERS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
    num_heads: int = DEFAULT_NUM_HEADS,
    attention_dropout: float = DEFAULT_ATTENTION_DROPOUT,
    attention_ff_dim: int | None = None,
    pool_attention_dim: int = DEFAULT_POOL_ATTENTION_DIM,
    bidirectional: bool = True,
    dropout: float = 0.3,
    random_seed: int = 42,
) -> "CNNBiLSTMAttention":
    """Construct a :class:`CNNBiLSTMAttention` with explicit hyperparameters.

    Args:
        num_classes: Number of fault classes.
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channels (1 for mel/MFCC/CQT, 3 for hybrid stacks).
        cnn_channels: Channels per CNN block.
        freq_pool: Adaptive frequency size before the LSTM.
        lstm_hidden: LSTM hidden size per direction.
        lstm_layers: Stacked BiLSTM layers (configurable depth).
        lstm_dropout: Dropout between LSTM layers.
        num_heads: Self-attention heads; must divide ``2·lstm_hidden``.
        attention_dropout: Self-attention dropout.
        attention_ff_dim: Self-attention feed-forward width (default 4×embed).
        pool_attention_dim: Attention-pooling scoring hidden size.
        bidirectional: Use a bidirectional LSTM.
        dropout: Dropout before the classifier head.
        random_seed: Seed for deterministic construction.

    Returns:
        A constructed :class:`CNNBiLSTMAttention`.

    Raises:
        ValueError: When ``2·lstm_hidden`` is not divisible by ``num_heads``.
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
            "cnn_channels": list(cnn_channels),
            "freq_pool": freq_pool,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout,
            "num_heads": num_heads,
            "attention_dropout": attention_dropout,
            "attention_ff_dim": attention_ff_dim,
            "pool_attention_dim": pool_attention_dim,
            "bidirectional": bidirectional,
        },
    )
    return CNNBiLSTMAttention(config)


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

    parser = argparse.ArgumentParser(description="CNN-BiLSTM-Attention classifier")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_cnn_bilstm_attention(
        num_classes=args.num_classes, num_heads=args.num_heads,
        in_channels=args.in_channels,
    )
    if args.summary or True:
        model.print_summary()

    dummy = torch.randn(2, *model.config.batched_input_shape)
    out = model(dummy)
    logger.info("Forward pass OK: %s -> %s | embedding dim %d | heads %d",
                tuple(dummy.shape), tuple(out.shape),
                model.feature_dim, model.num_heads)

    maps = model.attention_maps(dummy)
    logger.info("Attention maps: self=%s pooling=%s",
                tuple(maps["self_attention"].shape), tuple(maps["pooling"].shape))

    if args.onnx is not None:
        model.export_onnx(args.onnx)
        logger.info("Exported ONNX to %s", args.onnx)

    return 0


if __name__ == "__main__":
    sys.exit(main())