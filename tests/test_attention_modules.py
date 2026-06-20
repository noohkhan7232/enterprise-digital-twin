#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/attention_modules.py``.

Coverage:
- Registry (6 modules registered when torch present, build/list, duplicate reject)
- :class:`TemporalAttention` pooling, masking, weight normalisation (torch)
- :class:`MultiHeadTemporalAttention` multi-head pooling + per-head weights (torch)
- :class:`ChannelAttention` channel gating on feature maps (torch)
- :class:`SpatialAttention` spatial map generation (torch)
- :class:`SelfAttentionBlock` shape preservation, masking, residual (torch)
- :class:`AttentionPooling` mask-aware pooling + temperature (torch)
- Dynamic sequence length across all sequence modules (torch)
- Mixed-precision (autocast) forward passes (torch)
- Feature-importance extraction via return_weights (torch)
- Visualization helpers (extract_attention_map, attention_to_numpy) (torch)
- Fault tolerance: shape validation errors, divisibility checks (torch)
- ONNX export of a self-attention block (torch + onnx)
- Gradient flow through every module (torch)
- torch-absent graceful errors

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_attention_modules.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.attention_modules import (
    ATTENTION_REGISTRY,
    build_attention,
    list_attention_modules,
    register_attention,
)

try:
    import torch
    import torch.nn as nn

    from src.models.attention_modules import (
        AttentionPooling,
        ChannelAttention,
        MultiHeadTemporalAttention,
        SelfAttentionBlock,
        SpatialAttention,
        TemporalAttention,
        attention_to_numpy,
        extract_attention_map,
    )

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")

# Reference shapes
B, T, D = 4, 20, 64
FMAP = (4, 32, 8, 16)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the attention registry."""

    @torch_only
    def test_six_modules_registered(self) -> None:
        expected = {
            "temporal", "multihead_temporal", "channel",
            "spatial", "self_attention", "attention_pooling",
        }
        assert expected.issubset(set(list_attention_modules()))

    @torch_only
    def test_build_temporal(self) -> None:
        m = build_attention("temporal", input_dim=64)
        assert isinstance(m, TemporalAttention)

    @torch_only
    def test_build_self_attention(self) -> None:
        m = build_attention("self_attention", embed_dim=64, num_heads=8)
        assert isinstance(m, SelfAttentionBlock)

    @torch_only
    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown attention"):
            build_attention("does_not_exist")

    @torch_only
    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_attention("temporal")
            class _Other(nn.Module):
                pass

    @torch_only
    def test_registry_names_stamped(self) -> None:
        assert TemporalAttention._registry_name == "temporal"
        assert SelfAttentionBlock._registry_name == "self_attention"


# ---------------------------------------------------------------------------
# TemporalAttention
# ---------------------------------------------------------------------------


class TestTemporalAttention:
    """Tests for additive temporal attention."""

    @torch_only
    def test_pools_to_fixed_dim(self) -> None:
        attn = TemporalAttention(input_dim=D)
        out = attn(torch.randn(B, T, D))
        assert out.shape == (B, D)

    @torch_only
    def test_returns_weights(self) -> None:
        attn = TemporalAttention(input_dim=D)
        out, w = attn(torch.randn(B, T, D), return_weights=True)
        assert out.shape == (B, D)
        assert w.shape == (B, T)

    @torch_only
    def test_weights_sum_to_one(self) -> None:
        attn = TemporalAttention(input_dim=D)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)

    @torch_only
    def test_masking_zeros_padded(self) -> None:
        attn = TemporalAttention(input_dim=D)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, T // 2:] = False  # mask out second half
        _, w = attn(torch.randn(B, T, D), mask=mask, return_weights=True)
        # Masked positions get ~0 weight
        assert torch.allclose(w[:, T // 2:], torch.zeros(B, T - T // 2), atol=1e-5)
        assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)

    @torch_only
    def test_variable_length(self) -> None:
        attn = TemporalAttention(input_dim=D)
        for t in (5, 20, 100):
            out = attn(torch.randn(2, t, D))
            assert out.shape == (2, D)

    @torch_only
    def test_invalid_shape_raises(self) -> None:
        attn = TemporalAttention(input_dim=D)
        with pytest.raises(ValueError, match="3-D sequence"):
            attn(torch.randn(B, D))  # 2-D, invalid

    @torch_only
    def test_invalid_dims_raise(self) -> None:
        with pytest.raises(ValueError):
            TemporalAttention(input_dim=0)


# ---------------------------------------------------------------------------
# MultiHeadTemporalAttention
# ---------------------------------------------------------------------------


class TestMultiHeadTemporalAttention:
    """Tests for multi-head temporal attention."""

    @torch_only
    def test_pools_to_fixed_dim(self) -> None:
        attn = MultiHeadTemporalAttention(input_dim=D, num_heads=4)
        out = attn(torch.randn(B, T, D))
        assert out.shape == (B, D)

    @torch_only
    def test_per_head_weights(self) -> None:
        attn = MultiHeadTemporalAttention(input_dim=D, num_heads=4)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert w.shape == (B, 4, T)

    @torch_only
    def test_per_head_weights_sum_to_one(self) -> None:
        attn = MultiHeadTemporalAttention(input_dim=D, num_heads=4)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert torch.allclose(w.sum(dim=2), torch.ones(B, 4), atol=1e-5)

    @torch_only
    def test_masking(self) -> None:
        attn = MultiHeadTemporalAttention(input_dim=D, num_heads=2)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -5:] = False
        _, w = attn(torch.randn(B, T, D), mask=mask, return_weights=True)
        assert torch.allclose(w[:, :, -5:], torch.zeros(B, 2, 5), atol=1e-5)

    @torch_only
    def test_invalid_heads_raise(self) -> None:
        with pytest.raises(ValueError):
            MultiHeadTemporalAttention(input_dim=D, num_heads=0)


# ---------------------------------------------------------------------------
# ChannelAttention
# ---------------------------------------------------------------------------


class TestChannelAttention:
    """Tests for channel attention."""

    @torch_only
    def test_shape_preserved(self) -> None:
        attn = ChannelAttention(channels=32, reduction=8)
        out = attn(torch.randn(*FMAP))
        assert out.shape == FMAP

    @torch_only
    def test_returns_channel_gates(self) -> None:
        attn = ChannelAttention(channels=32)
        _, gate = attn(torch.randn(*FMAP), return_weights=True)
        assert gate.shape == (FMAP[0], FMAP[1])

    @torch_only
    def test_gates_in_unit_range(self) -> None:
        attn = ChannelAttention(channels=32)
        _, gate = attn(torch.randn(*FMAP), return_weights=True)
        assert (gate >= 0).all() and (gate <= 1).all()

    @torch_only
    def test_small_channels(self) -> None:
        attn = ChannelAttention(channels=8, reduction=16)
        assert attn(torch.randn(2, 8, 4, 4)).shape == (2, 8, 4, 4)

    @torch_only
    def test_invalid_shape_raises(self) -> None:
        attn = ChannelAttention(channels=32)
        with pytest.raises(ValueError, match="4-D feature map"):
            attn(torch.randn(B, T, D))  # 3-D, invalid


# ---------------------------------------------------------------------------
# SpatialAttention
# ---------------------------------------------------------------------------


class TestSpatialAttention:
    """Tests for spatial attention."""

    @torch_only
    def test_shape_preserved(self) -> None:
        attn = SpatialAttention(kernel_size=7)
        out = attn(torch.randn(*FMAP))
        assert out.shape == FMAP

    @torch_only
    def test_returns_spatial_map(self) -> None:
        attn = SpatialAttention()
        _, m = attn(torch.randn(*FMAP), return_weights=True)
        assert m.shape == (FMAP[0], FMAP[2], FMAP[3])  # (B, H, W)

    @torch_only
    def test_map_in_unit_range(self) -> None:
        attn = SpatialAttention()
        _, m = attn(torch.randn(*FMAP), return_weights=True)
        assert (m >= 0).all() and (m <= 1).all()

    @torch_only
    def test_even_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            SpatialAttention(kernel_size=6)

    @torch_only
    def test_variable_spatial_size(self) -> None:
        attn = SpatialAttention()
        for hw in ((8, 8), (16, 32), (4, 64)):
            out = attn(torch.randn(2, 16, *hw))
            assert out.shape == (2, 16, *hw)


# ---------------------------------------------------------------------------
# SelfAttentionBlock
# ---------------------------------------------------------------------------


class TestSelfAttentionBlock:
    """Tests for the self-attention encoder block."""

    @torch_only
    def test_shape_preserved(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        out = attn(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    @torch_only
    def test_returns_attention_matrix(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert w.shape == (B, T, T)

    @torch_only
    def test_attention_rows_sum_to_one(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        attn.eval()
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert torch.allclose(w.sum(dim=-1), torch.ones(B, T), atol=1e-4)

    @torch_only
    def test_masking(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=4)
        attn.eval()
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -5:] = False
        _, w = attn(torch.randn(B, T, D), mask=mask, return_weights=True)
        # Masked key positions get ~0 weight from every query
        assert torch.allclose(w[:, :, -5:], torch.zeros(B, T, 5), atol=1e-5)

    @torch_only
    def test_non_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            SelfAttentionBlock(embed_dim=64, num_heads=7)

    @torch_only
    def test_variable_length(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        for t in (5, 20, 50):
            out = attn(torch.randn(2, t, D))
            assert out.shape == (2, t, D)

    @torch_only
    def test_invalid_shape_raises(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        with pytest.raises(ValueError, match="3-D sequence"):
            attn(torch.randn(B, D))


# ---------------------------------------------------------------------------
# AttentionPooling
# ---------------------------------------------------------------------------


class TestAttentionPooling:
    """Tests for mask-aware attention pooling."""

    @torch_only
    def test_pools(self) -> None:
        attn = AttentionPooling(input_dim=D)
        out = attn(torch.randn(B, T, D))
        assert out.shape == (B, D)

    @torch_only
    def test_weights_sum_to_one(self) -> None:
        attn = AttentionPooling(input_dim=D)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-5)

    @torch_only
    def test_temperature_validation(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            AttentionPooling(input_dim=D, temperature=0.0)

    @torch_only
    def test_temperature_affects_sharpness(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, T, D)
        sharp = AttentionPooling(input_dim=D, temperature=0.1)
        soft = AttentionPooling(input_dim=D, temperature=10.0)
        # Copy weights so only temperature differs
        soft.load_state_dict(sharp.state_dict())
        _, w_sharp = sharp(x, return_weights=True)
        _, w_soft = soft(x, return_weights=True)
        # Lower temperature → higher max weight (sharper)
        assert w_sharp.max(dim=1).values.mean() >= w_soft.max(dim=1).values.mean()

    @torch_only
    def test_masking(self) -> None:
        attn = AttentionPooling(input_dim=D)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, -3:] = False
        _, w = attn(torch.randn(B, T, D), mask=mask, return_weights=True)
        assert torch.allclose(w[:, -3:], torch.zeros(B, 3), atol=1e-5)


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


class TestMixedPrecision:
    """Tests for AMP compatibility."""

    @torch_only
    def test_temporal_autocast_cpu(self) -> None:
        attn = TemporalAttention(input_dim=D)
        x = torch.randn(B, T, D)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = attn(x)
            assert out.shape == (B, D)
        except RuntimeError:
            pytest.skip("CPU autocast bf16 unsupported")

    @torch_only
    def test_self_attention_autocast_cpu(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        attn.eval()
        x = torch.randn(B, T, D)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = attn(x)
            assert out.shape == (B, T, D)
        except RuntimeError:
            pytest.skip("CPU autocast bf16 unsupported")


# ---------------------------------------------------------------------------
# Visualization / feature importance helpers
# ---------------------------------------------------------------------------


class TestVisualizationHelpers:
    """Tests for feature-importance and visualization helpers."""

    @torch_only
    def test_extract_attention_map(self) -> None:
        attn = TemporalAttention(input_dim=D)
        w = extract_attention_map(attn, torch.randn(B, T, D))
        assert w.shape == (B, T)

    @torch_only
    def test_extract_with_kwargs(self) -> None:
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        mask = torch.ones(B, T, dtype=torch.bool)
        w = extract_attention_map(attn, torch.randn(B, T, D), mask=mask)
        assert w.shape == (B, T, T)

    @torch_only
    def test_attention_to_numpy(self) -> None:
        attn = TemporalAttention(input_dim=D)
        _, w = attn(torch.randn(B, T, D), return_weights=True)
        arr = attention_to_numpy(w)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (B, T)

    @torch_only
    def test_extract_on_non_attention_raises(self) -> None:
        # A module that doesn't return weights
        plain = nn.Linear(D, D)
        with pytest.raises((TypeError, Exception)):
            extract_attention_map(plain, torch.randn(B, T, D))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests that gradients flow through every module."""

    @torch_only
    @pytest.mark.parametrize("factory", [
        lambda: TemporalAttention(input_dim=D),
        lambda: MultiHeadTemporalAttention(input_dim=D, num_heads=4),
        lambda: AttentionPooling(input_dim=D),
        lambda: SelfAttentionBlock(embed_dim=D, num_heads=8),
    ])
    def test_sequence_module_gradients(self, factory) -> None:
        module = factory()
        module.train()
        x = torch.randn(B, T, D, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None
        for p in module.parameters():
            assert p.grad is not None

    @torch_only
    @pytest.mark.parametrize("factory", [
        lambda: ChannelAttention(channels=32),
        lambda: SpatialAttention(),
    ])
    def test_feature_map_module_gradients(self, factory) -> None:
        module = factory()
        module.train()
        x = torch.randn(*FMAP, requires_grad=True)
        out = module(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Tests for ONNX export of attention modules."""

    @torch_only
    def test_self_attention_exports(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        attn = SelfAttentionBlock(embed_dim=D, num_heads=8)
        attn.eval()
        path = tmp_path / "self_attn.onnx"
        torch.onnx.export(
            attn, torch.randn(1, T, D), str(path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 1: "seq"},
                          "output": {0: "batch", 1: "seq"}},
            opset_version=17,
        )
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_temporal_attention_exports(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        attn = TemporalAttention(input_dim=D)
        attn.eval()
        path = tmp_path / "temporal.onnx"
        torch.onnx.export(
            attn, torch.randn(1, T, D), str(path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 1: "seq"}},
            opset_version=17,
        )
        assert path.is_file()


# ---------------------------------------------------------------------------
# torch-absent
# ---------------------------------------------------------------------------


class TestTorchAbsent:
    """Tests for graceful behaviour when PyTorch is absent."""

    def test_registry_accessor_works(self) -> None:
        # list_attention_modules never raises, even without torch
        assert isinstance(list_attention_modules(), list)

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_build_raises_without_torch(self) -> None:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_attention("temporal", input_dim=64)