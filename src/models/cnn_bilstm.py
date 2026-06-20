#!/usr/bin/env python3
"""CNN-BiLSTM with attention pooling for temporal acoustic fault classification.

This module implements :class:`CNNBiLSTM`, a hybrid architecture that pairs a
convolutional feature extractor with a bidirectional LSTM and learned attention
pooling.  It builds on the same :class:`~src.models.base_model.BaseModel`
foundation as :class:`~src.models.cnn_classifier.CNNClassifier` and
:class:`~src.models.resnet_acoustic.ResNetAcoustic`, and is registered as
``"cnn_bilstm"``.

Why a recurrent model — the temporal argument
---------------------------------------------
The two existing convolutional models (``acoustic_cnn``, ``resnet_acoustic``)
collapse the time axis with ``AdaptiveAvgPool2d(1)`` before the classifier.
That is the right inductive bias for *stationary* fault signatures, but it
**discards temporal order**: a bearing fault whose amplitude envelope modulates
over the clip, or a gearbox fault with drifting sidebands, produces the same
global-pooled descriptor as a stationary equivalent.  Many real wind-turbine
faults are *non-stationary* — they evolve within a 10-second clip as the rotor
loads and unloads.

:class:`CNNBiLSTM` preserves the time axis end-to-end:

1. **CNN feature extractor** — convolutional blocks that downsample the
   *frequency* axis aggressively while preserving *time* resolution, yielding a
   per-frame feature sequence rather than a single global vector.
2. **Bidirectional LSTM** — models how those per-frame features evolve, in both
   the forward and backward temporal directions, so the representation at each
   step is informed by the whole clip.
3. **Attention pooling** — a learned, softmax-normalised weighting over time
   steps that focuses the final representation on the most diagnostic moments
   (e.g. the instant a defect impacts), instead of averaging everything.

This is expected to outperform the pure-convolutional models specifically on
**temporal fault patterns** while remaining competitive on stationary ones.

Design notes for reviewers
--------------------------
* **ONNX-friendly recurrence** — the LSTM runs on the full (unpacked) sequence;
  ``pack_padded_sequence`` is deliberately avoided because it does not export
  cleanly to ONNX.  Opset 17 supports LSTM and the attention softmax, so the
  whole graph exports with a dynamic time axis.
* **Variable length** — the CNN preserves a proportional time dimension, the
  LSTM accepts any sequence length, and attention pooling reduces any length to
  a fixed vector, so the model handles clips of any duration.
* **Mixed precision** — no dtype is hardcoded in the forward path; the LSTM and
  attention run correctly under ``autocast``.
* **Reuses BaseModel** — checkpointing, ONNX export, parameter accounting,
  device management, reproducible construction, and ExperimentTracker
  integration are all inherited unchanged.

Feature-representation support
------------------------------
Representation-agnostic via ``in_channels`` / ``input_shape``: mel ``(128,T)``,
MFCC ``(40,T)``, CQT ``(168,T)`` run single-channel; hybrid stacks run with
``in_channels=3``.

Spatial / sequence flow (mel-128 input)::

    Input                (1,   128, 431)
    CNN block 1 (32)     (32,   64, 215)   pool (2,2)
    CNN block 2 (64)     (64,   32, 107)   pool (2,2)
    CNN block 3 (128)    (128,  16, 107)   pool (2,1) — time preserved
    CNN block 4 (128)    (128,   8, 107)   pool (2,1)
    AdaptiveAvgPool freq (128,   4, 107)   freq → 4
    Reshape to sequence  (107, 512)        (B, T, C·F)
    BiLSTM (hidden 128)  (107, 256)        2·hidden
    Attention pooling    (256,)            softmax over T
    Dropout + Linear     (num_classes,)

Configuration (via ``ModelConfig.extra``)
------------------------------------------
* ``cnn_channels`` (list[int]): channels per CNN block (default ``[32,64,128,128]``)
* ``freq_pool`` (int): adaptive frequency size before the LSTM (default 4)
* ``lstm_hidden`` (int): LSTM hidden size per direction (default 128)
* ``lstm_layers`` (int): stacked LSTM layers (default 2)
* ``lstm_dropout`` (float): dropout between LSTM layers (default 0.2)
* ``attention_dim`` (int): attention scoring hidden size (default 128)
* ``bidirectional`` (bool): use a bidirectional LSTM (default True)

Usage::

    from src.models.cnn_bilstm import build_cnn_bilstm
    model = build_cnn_bilstm(num_classes=5)
    logits = model(torch.randn(8, 1, 128, 431))            # (8, 5)
    emb = model.extract_features(torch.randn(8, 1, 128, 431))  # (8, 256)
    weights = model.attention_weights(torch.randn(8, 1, 128, 431))  # (8, T)

CLI::

    python src/models/cnn_bilstm.py --summary
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

logger = logging.getLogger("cnn_bilstm")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

MODEL_NAME: Final[str] = "cnn_bilstm"

DEFAULT_CNN_CHANNELS:  Final[tuple[int, ...]] = (32, 64, 128, 128)
DEFAULT_FREQ_POOL:     Final[int] = 4
DEFAULT_LSTM_HIDDEN:   Final[int] = 128
DEFAULT_LSTM_LAYERS:   Final[int] = 2
DEFAULT_LSTM_DROPOUT:  Final[float] = 0.2
DEFAULT_ATTENTION_DIM: Final[int] = 128


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Convolutional block: Conv → BN → ReLU → MaxPool.

        The pooling kernel is configurable so later blocks can reduce the
        frequency axis while preserving time resolution for the LSTM.

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

    class AttentionPooling(nn.Module):
        """Additive (Bahdanau-style) attention pooling over the time axis.

        Scores each time step with a small MLP, normalises the scores with a
        softmax over time, and returns the weighted sum of the sequence.  This
        collapses a variable-length sequence to a fixed vector while focusing on
        the most diagnostic time steps.

        Args:
            input_dim: Feature dimension of each time step.
            attention_dim: Hidden size of the scoring MLP.
        """

        def __init__(self, input_dim: int, attention_dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, attention_dim)
            self.score = nn.Linear(attention_dim, 1, bias=False)

        def forward(
            self, x: "torch.Tensor", *, return_weights: bool = False
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Pool a sequence over time with learned attention.

            Args:
                x: Sequence tensor ``(B, T, D)``.
                return_weights: Also return the attention weights ``(B, T)``.

            Returns:
                Pooled tensor ``(B, D)``, or ``(pooled, weights)`` when
                *return_weights* is set.
            """
            energy = self.score(torch.tanh(self.proj(x)))  # (B, T, 1)
            weights = torch.softmax(energy, dim=1)          # over time
            pooled = torch.sum(x * weights, dim=1)          # (B, D)
            if return_weights:
                return pooled, weights.squeeze(-1)
            return pooled

else:  # pragma: no cover
    ConvBlock = None  # type: ignore[assignment,misc]
    AttentionPooling = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# CNNBiLSTM
# ---------------------------------------------------------------------------


@register_model(MODEL_NAME)
class CNNBiLSTM(BaseModel):
    """CNN-BiLSTM with attention pooling for temporal fault classification.

    A convolutional feature extractor feeds a bidirectional LSTM whose outputs
    are attention-pooled over time and classified.  Optimised for
    *non-stationary* acoustic fault signatures that evolve within a clip.
    Inherits the full lifecycle contract from
    :class:`~src.models.base_model.BaseModel`.

    Configuration is read from ``ModelConfig.extra`` (see module docstring).

    Args:
        config: Model configuration. ``input_shape`` is the feature shape
            ``(F, T)`` and ``in_channels`` the channel count.

    Raises:
        RuntimeError: When instantiated without PyTorch installed.
    """

    def build_layers(self) -> None:
        """Construct the CNN extractor, BiLSTM, attention, and head.

        Reads architecture hyperparameters from ``self.config.extra`` and
        applies weight initialisation after construction.
        """
        cfg = self.config
        extra = cfg.extra or {}

        cnn_channels: list[int] = list(
            extra.get("cnn_channels", DEFAULT_CNN_CHANNELS)
        )
        self._freq_pool: int = int(extra.get("freq_pool", DEFAULT_FREQ_POOL))
        lstm_hidden: int = int(extra.get("lstm_hidden", DEFAULT_LSTM_HIDDEN))
        lstm_layers: int = int(extra.get("lstm_layers", DEFAULT_LSTM_LAYERS))
        lstm_dropout: float = float(extra.get("lstm_dropout", DEFAULT_LSTM_DROPOUT))
        attention_dim: int = int(extra.get("attention_dim", DEFAULT_ATTENTION_DIM))
        self._bidirectional: bool = bool(extra.get("bidirectional", True))

        # ── CNN feature extractor ────────────────────────────────────────────
        # Blocks 0-1 pool both axes (2,2); later blocks pool frequency only
        # (2,1) so the time axis is preserved for the LSTM.
        blocks: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i, out_ch in enumerate(cnn_channels):
            time_pool = 2 if i < 2 else 1
            blocks.append(ConvBlock(in_ch, out_ch, freq_pool=2, time_pool=time_pool))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self._cnn_out_channels = in_ch

        # Collapse the frequency axis to a fixed small size; keep time dynamic.
        self.freq_pool = nn.AdaptiveAvgPool2d((self._freq_pool, None))

        # Per-timestep feature dimension fed to the LSTM.
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
        self._lstm_out_dim = lstm_hidden * (2 if self._bidirectional else 1)

        # ── Attention pooling + head ──────────────────────────────────────────
        self.attention = AttentionPooling(self._lstm_out_dim, attention_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(self._lstm_out_dim, cfg.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming init to convs, Xavier to linears, orthogonal to LSTM.

        Orthogonal initialisation of the LSTM recurrent weights is a standard
        technique that stabilises gradient flow through time.
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
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

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
        """Run the CNN extractor and reshape the result into a time sequence.

        Args:
            x: Input tensor ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Sequence tensor ``(B, T', C·freq_pool)`` ready for the LSTM.
        """
        x = self._ensure_channel_dim(x)
        x = self.cnn(x)                       # (B, C, F', T')
        x = self.freq_pool(x)                 # (B, C, freq_pool, T')
        b, c, f, t = x.shape
        # (B, C, F, T) -> (B, T, C, F) -> (B, T, C*F); preserves time order
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        return x

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the full forward pass.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        seq = self._to_sequence(x)            # (B, T, D)
        lstm_out, _ = self.lstm(seq)          # (B, T, 2*hidden)
        pooled = self.attention(lstm_out)     # (B, 2*hidden)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the attention-pooled embedding before the classifier head.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Embedding of shape ``(B, lstm_out_dim)``.
        """
        seq = self._to_sequence(x)
        lstm_out, _ = self.lstm(seq)
        return self.attention(lstm_out)

    def attention_weights(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the per-time-step attention weights for interpretability.

        These weights reveal *when* in the clip the model focuses, which is
        valuable for diagnostics and for building trust with industrial
        operators reviewing model decisions.

        Args:
            x: Input batch ``(B, C, F, T)`` or ``(B, F, T)``.

        Returns:
            Attention weights of shape ``(B, T')`` summing to 1 along time.
        """
        seq = self._to_sequence(x)
        lstm_out, _ = self.lstm(seq)
        _, weights = self.attention(lstm_out, return_weights=True)
        return weights

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the :meth:`extract_features` embedding.

        Returns:
            The LSTM output dimension feeding the classifier head.
        """
        return self._lstm_out_dim


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def build_cnn_bilstm(
    num_classes: int = 5,
    input_shape: tuple[int, ...] = (128, 431),
    *,
    in_channels: int = 1,
    cnn_channels: tuple[int, ...] = DEFAULT_CNN_CHANNELS,
    freq_pool: int = DEFAULT_FREQ_POOL,
    lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
    lstm_layers: int = DEFAULT_LSTM_LAYERS,
    lstm_dropout: float = DEFAULT_LSTM_DROPOUT,
    attention_dim: int = DEFAULT_ATTENTION_DIM,
    bidirectional: bool = True,
    dropout: float = 0.3,
    random_seed: int = 42,
) -> "CNNBiLSTM":
    """Construct a :class:`CNNBiLSTM` with explicit hyperparameters.

    Args:
        num_classes: Number of fault classes.
        input_shape: Feature shape ``(F, T)`` (or ``(C, F, T)``).
        in_channels: Input channels (1 for mel/MFCC/CQT, 3 for hybrid stacks).
        cnn_channels: Channels per CNN block.
        freq_pool: Adaptive frequency size before the LSTM.
        lstm_hidden: LSTM hidden size per direction.
        lstm_layers: Stacked LSTM layers.
        lstm_dropout: Dropout between LSTM layers (only when layers > 1).
        attention_dim: Attention scoring hidden size.
        bidirectional: Use a bidirectional LSTM.
        dropout: Dropout before the classifier head.
        random_seed: Seed for deterministic construction.

    Returns:
        A constructed :class:`CNNBiLSTM`.

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
            "cnn_channels": list(cnn_channels),
            "freq_pool": freq_pool,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "lstm_dropout": lstm_dropout,
            "attention_dim": attention_dim,
            "bidirectional": bidirectional,
        },
    )
    return CNNBiLSTM(config)


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

    parser = argparse.ArgumentParser(description="CNN-BiLSTM acoustic classifier")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        logger.error("PyTorch not installed; cannot instantiate model.")
        return 1

    model = build_cnn_bilstm(num_classes=args.num_classes,
                             in_channels=args.in_channels)
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