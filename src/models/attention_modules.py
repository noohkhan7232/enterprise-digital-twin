#!/usr/bin/env python3
"""Reusable attention framework for acoustic fault diagnosis.

This module provides a self-contained, composable library of attention
mechanisms designed to be shared across every model in the project — the
existing :class:`~src.models.cnn_bilstm.CNNBiLSTM`, future transformer
encoders, anomaly-detection autoencoders, and remaining-useful-life (RUL)
regressors.  Each component is a standalone ``nn.Module`` with a uniform,
introspectable interface so attention can be dropped into any architecture and
its weights extracted for feature-importance analysis and visualization.

Why a shared framework?
-----------------------
Attention is reimplemented ad-hoc in many codebases, leading to subtle
inconsistencies in masking, numerical stability, and ONNX-exportability.  A
single reviewed framework guarantees that every model in the platform:

* handles **variable sequence length** identically (additive masking, never
  ``pack_padded_sequence`` which does not export to ONNX);
* is **ONNX-exportable** (no data-dependent Python control flow in ``forward``);
* is **mixed-precision safe** (softmax computed in fp32 then cast back, so AMP
  never produces NaNs from fp16 overflow);
* can **return its attention weights** on request for feature importance and
  heatmap visualization;
* is **fault-tolerant** (shape validation with actionable error messages).

Components
----------
+----------------------------+-------------------------------------------------+
| Module                     | Operates on / produces                          |
+============================+=================================================+
| :class:`TemporalAttention` | (B,T,D) → (B,D); additive attention over time   |
| :class:`MultiHeadTemporal  | (B,T,D) → (B,D); H parallel temporal heads      |
| Attention`                 |                                                 |
| :class:`ChannelAttention`  | (B,C,H,W) → (B,C,H,W); SE-style channel gate    |
| :class:`SpatialAttention`  | (B,C,H,W) → (B,C,H,W); freq-time spatial map    |
| :class:`SelfAttentionBlock`| (B,T,D) → (B,T,D); scaled dot-product self-attn |
| :class:`AttentionPooling`  | (B,T,D) → (B,D); learned pooling                |
+----------------------------+-------------------------------------------------+

Registry
--------
A lightweight registry (:func:`register_attention`, :func:`build_attention`)
lets architectures construct attention blocks by name from config, mirroring
the model registry in :mod:`src.models.base_model` without depending on it.

Backward compatibility
-----------------------
This module is additive.  It does **not** modify the ``AttentionPooling`` class
already defined inside :mod:`src.models.cnn_bilstm`; that remains the canonical
one used by the shipped model.  The :class:`AttentionPooling` here is a richer,
mask-aware superset that new models may adopt.

Usage::

    from src.models.attention_modules import (
        TemporalAttention, SelfAttentionBlock, build_attention,
    )

    attn = TemporalAttention(input_dim=256, attention_dim=128)
    pooled, weights = attn(sequence, mask=mask, return_weights=True)

    # Or via the registry
    block = build_attention("self_attention", embed_dim=256, num_heads=8)

CLI::

    python src/models/attention_modules.py --list
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Final, Type

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE: bool = True
    _ModuleBase: type = nn.Module
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE: bool = False
    _ModuleBase = object

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("attention_modules")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Large negative value used as the additive mask fill (ONNX-friendly; avoids
#: ``float('-inf')`` which can produce NaNs after softmax on fully-masked rows).
_MASK_FILL: Final[float] = -1e9

#: Global attention-module registry: name -> nn.Module subclass.
ATTENTION_REGISTRY: dict[str, Type] = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_attention(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering an attention module by name.

    Args:
        name: Unique registry key used by :func:`build_attention`.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: Type) -> Type:
        existing = ATTENTION_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Attention '{name}' already registered to {existing.__name__}"
            )
        ATTENTION_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered attention '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def build_attention(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered attention module by name.

    Args:
        name: Registry key (see :func:`register_attention`).
        **kwargs: Forwarded to the module constructor.

    Returns:
        An instantiated attention module.

    Raises:
        KeyError: When *name* is not registered.
        RuntimeError: When PyTorch is unavailable.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("build_attention requires PyTorch")
    if name not in ATTENTION_REGISTRY:
        available = ", ".join(sorted(ATTENTION_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown attention '{name}'. Available: {available}")
    return ATTENTION_REGISTRY[name](**kwargs)


def list_attention_modules() -> list[str]:
    """Return the sorted list of registered attention-module names.

    Returns:
        Sorted registry keys.
    """
    return sorted(ATTENTION_REGISTRY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_softmax(scores: "torch.Tensor", dim: int) -> "torch.Tensor":
    """Numerically-stable, mixed-precision-safe softmax.

    Computes the softmax in float32 regardless of the input dtype, then casts
    back.  Under AMP this prevents fp16 overflow in ``exp`` from producing NaNs.

    Args:
        scores: Raw attention scores.
        dim: Dimension to normalise over.

    Returns:
        Softmax weights in the original dtype.
    """
    return torch.softmax(scores.float(), dim=dim).to(scores.dtype)


def _apply_mask(
    scores: "torch.Tensor", mask: "torch.Tensor | None"
) -> "torch.Tensor":
    """Apply an additive boolean mask to attention scores.

    Args:
        scores: Attention scores; masked positions are set to a large negative
            value before softmax.
        mask: Boolean tensor broadcastable to *scores*, ``True`` for valid
            positions.  ``None`` is a no-op.

    Returns:
        Masked scores (or *scores* unchanged when *mask* is ``None``).
    """
    if mask is None:
        return scores
    return scores.masked_fill(~mask, _MASK_FILL)


def _validate_sequence(x: "torch.Tensor", name: str) -> None:
    """Validate that a tensor is a 3-D sequence ``(B, T, D)``.

    Args:
        x: Tensor to validate.
        name: Module name for the error message.

    Raises:
        ValueError: When *x* is not 3-dimensional.
    """
    if x.dim() != 3:
        raise ValueError(
            f"{name} expects a 3-D sequence (B, T, D); got shape {tuple(x.shape)}"
        )


def _validate_feature_map(x: "torch.Tensor", name: str) -> None:
    """Validate that a tensor is a 4-D feature map ``(B, C, H, W)``.

    Args:
        x: Tensor to validate.
        name: Module name for the error message.

    Raises:
        ValueError: When *x* is not 4-dimensional.
    """
    if x.dim() != 4:
        raise ValueError(
            f"{name} expects a 4-D feature map (B, C, H, W); "
            f"got shape {tuple(x.shape)}"
        )


# ===========================================================================
# Attention modules
# ===========================================================================

if _TORCH_AVAILABLE:

    @register_attention("temporal")
    class TemporalAttention(nn.Module):
        """Additive (Bahdanau-style) attention pooling over the time axis.

        Scores each time step with a small MLP, normalises with a masked
        softmax over time, and returns the weighted sum.  Reduces a
        variable-length sequence ``(B, T, D)`` to a fixed vector ``(B, D)``.

        Args:
            input_dim: Feature dimension of each time step.
            attention_dim: Hidden size of the scoring MLP.
            dropout: Dropout applied to the attention weights.
        """

        _registry_name: str | None = None

        def __init__(
            self, input_dim: int, attention_dim: int = 128, dropout: float = 0.0
        ) -> None:
            super().__init__()
            if input_dim < 1 or attention_dim < 1:
                raise ValueError("input_dim and attention_dim must be >= 1")
            self.proj = nn.Linear(input_dim, attention_dim)
            self.score = nn.Linear(attention_dim, 1, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.input_dim = input_dim

        def forward(
            self,
            x: "torch.Tensor",
            *,
            mask: "torch.Tensor | None" = None,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Pool a sequence over time with additive attention.

            Args:
                x: Sequence tensor ``(B, T, D)``.
                mask: Optional boolean mask ``(B, T)``, ``True`` for valid steps.
                return_weights: Also return attention weights ``(B, T)``.

            Returns:
                Pooled tensor ``(B, D)``, or ``(pooled, weights)``.
            """
            _validate_sequence(x, "TemporalAttention")
            energy = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # (B,T)
            energy = _apply_mask(energy, mask)
            weights = _stable_softmax(energy, dim=1)                   # (B,T)
            weights = self.dropout(weights)
            pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)       # (B,D)
            if return_weights:
                return pooled, weights
            return pooled

    @register_attention("multihead_temporal")
    class MultiHeadTemporalAttention(nn.Module):
        """Multi-head additive temporal attention.

        Runs ``num_heads`` independent additive-attention heads over the time
        axis and concatenates their pooled outputs, then projects back to
        ``input_dim``.  Each head can focus on a different temporal pattern
        (e.g. onset vs. steady-state vs. decay of a fault transient).

        Args:
            input_dim: Feature dimension of each time step.
            num_heads: Number of parallel attention heads.
            attention_dim: Per-head scoring hidden size.
            dropout: Dropout on the attention weights.
        """

        _registry_name: str | None = None

        def __init__(
            self,
            input_dim: int,
            num_heads: int = 4,
            attention_dim: int = 64,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if num_heads < 1:
                raise ValueError("num_heads must be >= 1")
            self.num_heads = num_heads
            self.input_dim = input_dim
            self.proj = nn.Linear(input_dim, attention_dim * num_heads)
            self.score = nn.Linear(attention_dim, 1, bias=False)
            self.attention_dim = attention_dim
            self.out = nn.Linear(input_dim * num_heads, input_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: "torch.Tensor",
            *,
            mask: "torch.Tensor | None" = None,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Pool a sequence with multiple temporal attention heads.

            Args:
                x: Sequence tensor ``(B, T, D)``.
                mask: Optional boolean mask ``(B, T)``.
                return_weights: Also return per-head weights ``(B, H, T)``.

            Returns:
                Pooled tensor ``(B, D)``, or ``(pooled, weights)``.
            """
            _validate_sequence(x, "MultiHeadTemporalAttention")
            b, t, _ = x.shape
            # (B, T, H*a) -> (B, H, T, a)
            proj = torch.tanh(self.proj(x)).view(b, t, self.num_heads, self.attention_dim)
            proj = proj.permute(0, 2, 1, 3)
            energy = self.score(proj).squeeze(-1)            # (B, H, T)
            if mask is not None:
                energy = _apply_mask(energy, mask.unsqueeze(1))
            weights = _stable_softmax(energy, dim=2)         # (B, H, T)
            weights = self.dropout(weights)
            # Weighted sum per head -> (B, H, D) -> concat -> (B, H*D)
            pooled = torch.einsum("bht,btd->bhd", weights, x)
            pooled = pooled.reshape(b, self.num_heads * self.input_dim)
            pooled = self.out(pooled)                        # (B, D)
            if return_weights:
                return pooled, weights
            return pooled

    @register_attention("channel")
    class ChannelAttention(nn.Module):
        """Squeeze-and-Excitation channel attention for conv feature maps.

        Squeezes spatial information into a per-channel descriptor (average +
        max pooling), learns per-channel gates through a bottleneck MLP, and
        rescales the channels.  Operates on ``(B, C, H, W)`` maps.

        Args:
            channels: Number of channels.
            reduction: Bottleneck reduction ratio.
        """

        _registry_name: str | None = None

        def __init__(self, channels: int, reduction: int = 16) -> None:
            super().__init__()
            bottleneck = max(1, channels // reduction)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.mlp = nn.Sequential(
                nn.Linear(channels, bottleneck, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, channels, bias=True),
            )
            self.channels = channels

        def forward(
            self,
            x: "torch.Tensor",
            *,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Apply channel attention.

            Args:
                x: Feature map ``(B, C, H, W)``.
                return_weights: Also return channel gates ``(B, C)``.

            Returns:
                Reweighted map ``(B, C, H, W)``, or ``(map, weights)``.
            """
            _validate_feature_map(x, "ChannelAttention")
            b, c, _, _ = x.shape
            avg = self.mlp(self.avg_pool(x).view(b, c))
            mx = self.mlp(self.max_pool(x).view(b, c))
            gate = torch.sigmoid(avg + mx)                  # (B, C)
            out = x * gate.view(b, c, 1, 1)
            if return_weights:
                return out, gate
            return out

    @register_attention("spatial")
    class SpatialAttention(nn.Module):
        """Spatial attention for conv feature maps.

        Computes a single-channel spatial attention map from the channel-wise
        average and max responses, passed through a convolution and sigmoid,
        then rescales the input.  The map ``(B, H, W)`` highlights the
        time-frequency regions the model attends to — directly usable as a
        diagnostic heatmap.

        Args:
            kernel_size: Convolution kernel size for the attention map
                (odd, so the output preserves spatial size).
        """

        _registry_name: str | None = None

        def __init__(self, kernel_size: int = 7) -> None:
            super().__init__()
            if kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd")
            self.conv = nn.Conv2d(
                2, 1, kernel_size=kernel_size,
                padding=kernel_size // 2, bias=False,
            )

        def forward(
            self,
            x: "torch.Tensor",
            *,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Apply spatial attention.

            Args:
                x: Feature map ``(B, C, H, W)``.
                return_weights: Also return the spatial map ``(B, H, W)``.

            Returns:
                Reweighted map ``(B, C, H, W)``, or ``(map, weights)``.
            """
            _validate_feature_map(x, "SpatialAttention")
            avg = torch.mean(x, dim=1, keepdim=True)        # (B,1,H,W)
            mx, _ = torch.max(x, dim=1, keepdim=True)       # (B,1,H,W)
            attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
            out = x * attn
            if return_weights:
                return out, attn.squeeze(1)
            return out

    @register_attention("self_attention")
    class SelfAttentionBlock(nn.Module):
        """Scaled dot-product multi-head self-attention block.

        A transformer-style encoder block: multi-head self-attention with a
        residual connection and LayerNorm, followed by a position-wise
        feed-forward network with its own residual + LayerNorm.  This is the
        reusable core for future transformer encoders over acoustic sequences.

        Args:
            embed_dim: Token / feature dimension.
            num_heads: Number of attention heads (must divide ``embed_dim``).
            ff_dim: Hidden size of the feed-forward network.
            dropout: Dropout probability.

        Raises:
            ValueError: When ``embed_dim`` is not divisible by ``num_heads``.
        """

        _registry_name: str | None = None

        def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            ff_dim: int | None = None,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim ({embed_dim}) must be divisible by "
                    f"num_heads ({num_heads})"
                )
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            ff_dim = ff_dim or embed_dim * 4

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
            self.dropout = nn.Dropout(dropout)

        def _split_heads(self, x: "torch.Tensor") -> "torch.Tensor":
            """Reshape ``(B, T, D)`` to ``(B, H, T, d)``."""
            b, t, _ = x.shape
            return x.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def forward(
            self,
            x: "torch.Tensor",
            *,
            mask: "torch.Tensor | None" = None,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Run the self-attention encoder block.

            Args:
                x: Sequence tensor ``(B, T, D)``.
                mask: Optional boolean mask ``(B, T)``, ``True`` for valid steps.
                return_weights: Also return averaged attention ``(B, T, T)``.

            Returns:
                Encoded sequence ``(B, T, D)``, or ``(encoded, weights)``.
            """
            _validate_sequence(x, "SelfAttentionBlock")
            b, t, _ = x.shape

            q = self._split_heads(self.q_proj(x))   # (B,H,T,d)
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                # mask (B,T) -> (B,1,1,T) broadcast over heads and query positions
                scores = _apply_mask(scores, mask.view(b, 1, 1, t))
            weights = _stable_softmax(scores, dim=-1)        # (B,H,T,T)
            weights = self.dropout(weights)

            context = torch.matmul(weights, v)               # (B,H,T,d)
            context = context.permute(0, 2, 1, 3).reshape(b, t, self.embed_dim)
            attn_out = self.out_proj(context)

            # Residual + norm, then FFN + residual + norm
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ff(x))

            if return_weights:
                return x, weights.mean(dim=1)                # (B,T,T) head-averaged
            return x

    @register_attention("attention_pooling")
    class AttentionPooling(nn.Module):
        """Mask-aware learned attention pooling over time.

        A richer, mask-aware superset of the pooling used inside
        :mod:`src.models.cnn_bilstm`, suitable for new models.  Reduces a
        sequence ``(B, T, D)`` to ``(B, D)`` with a learned, optionally
        temperature-scaled attention distribution.

        Args:
            input_dim: Feature dimension of each time step.
            attention_dim: Hidden size of the scoring MLP.
            temperature: Softmax temperature (>1 softens, <1 sharpens).
        """

        _registry_name: str | None = None

        def __init__(
            self,
            input_dim: int,
            attention_dim: int = 128,
            temperature: float = 1.0,
        ) -> None:
            super().__init__()
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            self.proj = nn.Linear(input_dim, attention_dim)
            self.score = nn.Linear(attention_dim, 1, bias=False)
            self.temperature = temperature
            self.input_dim = input_dim

        def forward(
            self,
            x: "torch.Tensor",
            *,
            mask: "torch.Tensor | None" = None,
            return_weights: bool = False,
        ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
            """Pool a sequence over time.

            Args:
                x: Sequence tensor ``(B, T, D)``.
                mask: Optional boolean mask ``(B, T)``.
                return_weights: Also return attention weights ``(B, T)``.

            Returns:
                Pooled tensor ``(B, D)``, or ``(pooled, weights)``.
            """
            _validate_sequence(x, "AttentionPooling")
            energy = self.score(torch.tanh(self.proj(x))).squeeze(-1)
            energy = energy / self.temperature
            energy = _apply_mask(energy, mask)
            weights = _stable_softmax(energy, dim=1)
            pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
            if return_weights:
                return pooled, weights
            return pooled

else:  # pragma: no cover
    TemporalAttention = None  # type: ignore[assignment,misc]
    MultiHeadTemporalAttention = None  # type: ignore[assignment,misc]
    ChannelAttention = None  # type: ignore[assignment,misc]
    SpatialAttention = None  # type: ignore[assignment,misc]
    SelfAttentionBlock = None  # type: ignore[assignment,misc]
    AttentionPooling = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Feature-importance / visualization helpers
# ---------------------------------------------------------------------------


def extract_attention_map(
    module: Any,
    x: "torch.Tensor",
    **forward_kwargs: Any,
) -> "Any":
    """Run a module and return only its attention weights.

    A uniform accessor for feature-importance analysis: works with any module
    in this framework that accepts ``return_weights=True``.

    Args:
        module: An attention module from this framework.
        x: Input tensor appropriate for the module.
        **forward_kwargs: Extra keyword arguments (e.g. ``mask``).

    Returns:
        The attention-weight tensor.

    Raises:
        RuntimeError: When PyTorch is unavailable.
        TypeError: When the module does not support ``return_weights``.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("extract_attention_map requires PyTorch")
    out = module(x, return_weights=True, **forward_kwargs)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(
            f"{type(module).__name__} did not return (output, weights); "
            "it may not support return_weights"
        )
    return out[1]


def attention_to_numpy(weights: "torch.Tensor") -> "Any":
    """Detach attention weights to a NumPy array for visualization.

    Args:
        weights: Attention-weight tensor on any device.

    Returns:
        A NumPy array (float32) ready for heatmap plotting.

    Raises:
        RuntimeError: When PyTorch is unavailable.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("attention_to_numpy requires PyTorch")
    return weights.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: list registered attention modules.

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

    parser = argparse.ArgumentParser(description="Attention module registry")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    print(f"PyTorch available: {_TORCH_AVAILABLE}")
    modules = list_attention_modules()
    print(f"\nRegistered attention modules ({len(modules)}):")
    for name in modules:
        print(f"  • {name} -> {ATTENTION_REGISTRY[name].__name__}")

    if _TORCH_AVAILABLE and (args.list or True):
        # Quick smoke test of each
        print("\nSmoke test (random inputs):")
        seq = torch.randn(2, 16, 64)
        fmap = torch.randn(2, 32, 8, 8)
        for name in ("temporal", "multihead_temporal", "attention_pooling"):
            m = build_attention(name, input_dim=64)
            out = m(seq)
            print(f"  {name:22s} (2,16,64) -> {tuple(out.shape)}")
        m = build_attention("self_attention", embed_dim=64, num_heads=8)
        print(f"  {'self_attention':22s} (2,16,64) -> {tuple(m(seq).shape)}")
        for name, ch in (("channel", 32), ("spatial", None)):
            m = build_attention(name, channels=ch) if ch else build_attention(name)
            print(f"  {name:22s} (2,32,8,8) -> {tuple(m(fmap).shape)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())