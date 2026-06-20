#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/acoustic_transformer.py``.

Coverage (60+ tests):
- Registry integration (registered, distinct, BaseModel subclass, coexistence)
- Composition: reuses SelfAttentionBlock from the attention framework
- PatchEmbedding tokenisation across feature shapes (torch)
- SinusoidalPositionalEncoding length-independence + dynamic length (torch)
- Construction with default and custom hyperparameters (torch)
- embed_dim/num_heads divisibility + patch-size guards (torch)
- Forward pass for mel / MFCC / CQT / hybrid inputs (torch)
- Dynamic + variable sequence length support (torch)
- Batch-size invariance (1 … N) (torch)
- Feature embedding extraction + feature_dim (torch)
- Attention-map extraction (per-layer, shapes, row-normalisation) (torch)
- Build-factory tests (presets, overrides, validation) (torch)
- ONNX export with dynamic axes (torch + onnx)
- Mixed-precision (autocast) forward (torch)
- Checkpoint save/load round-trip preserving arch + weights (torch)
- Parameter counting (torch)
- Device transfer (torch)
- Learnability smoke test → separable-data accuracy (torch)
- Inherited BaseModel functionality (predict, summary) (torch)
- torch-absent graceful errors

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_acoustic_transformer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import (
    MODEL_REGISTRY,
    BaseModel,
    ModelConfig,
    build_model,
    is_registered,
)
from src.models.acoustic_transformer import (
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PATCH_SIZE,
    MODEL_NAME,
    AcousticTransformer,
    build_acoustic_transformer,
)

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")

MEL_SHAPE = (128, 431)
MFCC_SHAPE = (40, 431)
CQT_SHAPE = (168, 431)
N_CLASSES = 5


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for registry integration and coexistence."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is AcousticTransformer

    def test_model_name_value(self) -> None:
        assert MODEL_NAME == "acoustic_transformer"

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(AcousticTransformer, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert AcousticTransformer._registry_name == MODEL_NAME

    def test_forward_concrete(self) -> None:
        assert not getattr(AcousticTransformer.forward, "__isabstractmethod__", False)

    def test_coexists_with_all_models(self) -> None:
        import src.models.cnn_classifier  # noqa: F401
        import src.models.resnet_acoustic  # noqa: F401
        import src.models.cnn_bilstm  # noqa: F401
        import src.models.cnn_bilstm_attention  # noqa: F401
        import src.models.anomaly_autoencoder  # noqa: F401
        from src.models.base_model import list_models
        for name in ("acoustic_cnn", "resnet_acoustic", "cnn_bilstm",
                     "cnn_bilstm_attention", "anomaly_autoencoder", MODEL_NAME):
            assert name in list_models()

    def test_reuses_attention_framework(self) -> None:
        import inspect
        import src.models.acoustic_transformer as m
        src = inspect.getsource(m)
        assert "from src.models.attention_modules import SelfAttentionBlock" in src


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------


class TestPatchEmbedding:
    """Tests for the convolutional patch embedding."""

    @torch_only
    def test_tokenises_mel(self) -> None:
        from src.models.acoustic_transformer import PatchEmbedding
        pe = PatchEmbedding(1, 192, 16)
        tokens, grid = pe(torch.randn(2, 1, 128, 431))
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == 192
        assert grid == (8, 26)  # 128/16, 431//16
        assert tokens.shape[1] == grid[0] * grid[1]

    @torch_only
    def test_tokenises_mfcc(self) -> None:
        from src.models.acoustic_transformer import PatchEmbedding
        pe = PatchEmbedding(1, 96, 8)
        tokens, grid = pe(torch.randn(2, 1, 40, 431))
        assert grid == (5, 53)
        assert tokens.shape == (2, 5 * 53, 96)

    @torch_only
    def test_multichannel(self) -> None:
        from src.models.acoustic_transformer import PatchEmbedding
        pe = PatchEmbedding(3, 64, 16)
        tokens, _ = pe(torch.randn(2, 3, 128, 256))
        assert tokens.shape[2] == 64


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    @torch_only
    def test_adds_encoding(self) -> None:
        from src.models.acoustic_transformer import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(64, 4096)
        x = torch.zeros(2, 100, 64)
        out = pe(x)
        # Output equals the positional encoding when input is zero
        assert out.shape == (2, 100, 64)
        assert not torch.allclose(out, x)

    @torch_only
    def test_length_independent(self) -> None:
        from src.models.acoustic_transformer import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(64, 4096)
        out_short = pe(torch.zeros(1, 100, 64))
        out_long = pe(torch.zeros(1, 300, 64))
        # First 100 positions identical regardless of total length
        assert torch.allclose(out_short[0], out_long[0, :100], atol=1e-5)

    @torch_only
    def test_variable_length(self) -> None:
        from src.models.acoustic_transformer import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(32, 4096)
        for n in (10, 100, 1000):
            out = pe(torch.zeros(2, n, 32))
            assert out.shape == (2, n, 32)

    @torch_only
    def test_longer_than_buffer(self) -> None:
        # Sequence longer than max_positions falls back to on-the-fly build
        from src.models.acoustic_transformer import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(16, 64)
        out = pe(torch.zeros(1, 200, 16))
        assert out.shape == (1, 200, 16)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_default(self) -> None:
        assert isinstance(build_acoustic_transformer(num_classes=5), AcousticTransformer)

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        assert isinstance(build_model(MODEL_NAME, cfg), AcousticTransformer)

    @torch_only
    def test_has_components(self) -> None:
        model = build_acoustic_transformer()
        assert hasattr(model, "patch_embed")
        assert hasattr(model, "pos_encoding")
        assert hasattr(model, "encoder_layers")
        assert hasattr(model, "classifier")

    @torch_only
    def test_num_layers_matches(self) -> None:
        model = build_acoustic_transformer(num_layers=4)
        assert len(model.encoder_layers) == 4
        assert model.num_layers == 4

    @torch_only
    def test_num_heads_property(self) -> None:
        model = build_acoustic_transformer(num_heads=8, embed_dim=256)
        assert model.num_heads == 8

    @torch_only
    def test_embed_dim_feature_dim(self) -> None:
        model = build_acoustic_transformer(embed_dim=128, num_heads=4)
        assert model.feature_dim == 128

    @torch_only
    def test_divisibility_guard(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            build_acoustic_transformer(embed_dim=100, num_heads=7)

    @torch_only
    def test_patch_size_guard(self) -> None:
        # patch larger than freq dim (40) must raise
        with pytest.raises(ValueError, match="patch_size"):
            build_acoustic_transformer(input_shape=(40, 431), patch_size=64)

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_acoustic_transformer(random_seed=42)
        m2 = build_acoustic_transformer(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    @torch_only
    def test_mlp_ratio_affects_ff(self) -> None:
        small = build_acoustic_transformer(mlp_ratio=1.0)
        large = build_acoustic_transformer(mlp_ratio=4.0)
        assert (large.count_parameters().total
                > small.count_parameters().total)


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


class TestBuildFactory:
    """Tests for the build_acoustic_transformer factory."""

    @torch_only
    def test_returns_instance(self) -> None:
        assert isinstance(build_acoustic_transformer(), AcousticTransformer)

    @torch_only
    def test_custom_num_classes(self) -> None:
        model = build_acoustic_transformer(num_classes=10)
        assert model.config.num_classes == 10

    @torch_only
    def test_custom_embed_and_heads(self) -> None:
        model = build_acoustic_transformer(embed_dim=384, num_heads=12)
        assert model.feature_dim == 384
        assert model.num_heads == 12

    @torch_only
    def test_custom_patch_size(self) -> None:
        model = build_acoustic_transformer(patch_size=8)
        # Smaller patches -> more tokens -> forward still works
        out = model(torch.randn(1, 1, *MEL_SHAPE))
        assert out.shape == (1, N_CLASSES)

    @torch_only
    def test_in_channels_propagates(self) -> None:
        model = build_acoustic_transformer(in_channels=3)
        assert model.config.in_channels == 3

    @torch_only
    def test_arch_stored_in_extra(self) -> None:
        model = build_acoustic_transformer(embed_dim=256, num_heads=8,
                                            num_layers=3)
        extra = model.config.extra
        assert extra["embed_dim"] == 256
        assert extra["num_heads"] == 8
        assert extra["num_layers"] == 3


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for forward passes across feature representations."""

    @torch_only
    def test_mel_input(self) -> None:
        model = build_acoustic_transformer(num_classes=5, input_shape=MEL_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_mfcc_input(self) -> None:
        model = build_acoustic_transformer(num_classes=5, input_shape=MFCC_SHAPE,
                                            patch_size=8)
        model.eval()
        assert model(torch.randn(4, 1, *MFCC_SHAPE)).shape == (4, 5)

    @torch_only
    def test_cqt_input(self) -> None:
        model = build_acoustic_transformer(num_classes=5, input_shape=CQT_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *CQT_SHAPE)).shape == (4, 5)

    @torch_only
    def test_hybrid_3channel(self) -> None:
        model = build_acoustic_transformer(num_classes=5, in_channels=3)
        model.eval()
        assert model(torch.randn(4, 3, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_3d_input_auto_channel(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        assert model(torch.randn(4, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_output_finite(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        out = model(torch.randn(4, 1, *MEL_SHAPE))
        assert torch.all(torch.isfinite(out))

    @torch_only
    def test_logits_shape_various_classes(self) -> None:
        for nc in (2, 5, 10):
            model = build_acoustic_transformer(num_classes=nc)
            model.eval()
            assert model(torch.randn(2, 1, *MEL_SHAPE)).shape == (2, nc)


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


class TestBatchSize:
    """Tests for batch-size invariance."""

    @torch_only
    def test_batch_one(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        assert model(torch.randn(1, 1, *MEL_SHAPE)).shape == (1, 5)

    @torch_only
    def test_batch_large(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        assert model(torch.randn(16, 1, *MEL_SHAPE)).shape == (16, 5)

    @torch_only
    def test_various_batches(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        for b in (1, 2, 8, 32):
            assert model(torch.randn(b, 1, *MEL_SHAPE)).shape == (b, 5)


# ---------------------------------------------------------------------------
# Dynamic / variable sequence length
# ---------------------------------------------------------------------------


class TestDynamicLength:
    """Tests for dynamic and variable sequence length support."""

    @torch_only
    def test_variable_time(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        for T in (128, 256, 431, 640):
            assert model(torch.randn(2, 1, 128, T)).shape == (2, 5)

    @torch_only
    def test_variable_freq(self) -> None:
        model = build_acoustic_transformer(num_classes=5, patch_size=8)
        model.eval()
        for F in (64, 128, 168):
            assert model(torch.randn(2, 1, F, 256)).shape == (2, 5)

    @torch_only
    def test_short_clip(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        # Short clip (few patches) still classifies
        assert model(torch.randn(2, 1, 128, 64)).shape == (2, 5)

    @torch_only
    def test_long_clip(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        assert model(torch.randn(1, 1, 128, 1000)).shape == (1, 5)

    @torch_only
    def test_consistency_same_input(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for embedding extraction."""

    @torch_only
    def test_extract_features_shape(self) -> None:
        model = build_acoustic_transformer(embed_dim=192, num_heads=6)
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, 192)

    @torch_only
    def test_feature_dim_matches(self) -> None:
        model = build_acoustic_transformer(embed_dim=256, num_heads=8)
        assert model.feature_dim == 256

    @torch_only
    def test_embedding_differs_from_logits(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        feats = model.extract_features(torch.randn(2, 1, *MEL_SHAPE))
        assert feats.shape[1] != 5

    @torch_only
    def test_features_finite(self) -> None:
        model = build_acoustic_transformer()
        model.eval()
        feats = model.extract_features(torch.randn(2, 1, *MEL_SHAPE))
        assert torch.all(torch.isfinite(feats))

    @torch_only
    def test_features_dynamic_length(self) -> None:
        model = build_acoustic_transformer(embed_dim=192, num_heads=6)
        model.eval()
        for T in (256, 431):
            feats = model.extract_features(torch.randn(2, 1, 128, T))
            assert feats.shape == (2, 192)


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------


class TestAttentionMaps:
    """Tests for attention-map extraction."""

    @torch_only
    def test_returns_per_layer(self) -> None:
        model = build_acoustic_transformer(num_layers=4)
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        assert len(maps) == 4

    @torch_only
    def test_map_is_square(self) -> None:
        model = build_acoustic_transformer(num_layers=2)
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        for m in maps:
            assert m.dim() == 3
            assert m.shape[1] == m.shape[2]  # (B, N, N)

    @torch_only
    def test_rows_sum_to_one(self) -> None:
        model = build_acoustic_transformer(num_layers=2)
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        for m in maps:
            sums = m.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    @torch_only
    def test_batch_dim(self) -> None:
        model = build_acoustic_transformer(num_layers=3)
        model.eval()
        maps = model.attention_maps(torch.randn(5, 1, *MEL_SHAPE))
        for m in maps:
            assert m.shape[0] == 5

    @torch_only
    def test_maps_finite(self) -> None:
        model = build_acoustic_transformer(num_layers=2)
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        for m in maps:
            assert torch.all(torch.isfinite(m))


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


class TestParameterCounting:
    """Tests for parameter accounting."""

    @torch_only
    def test_total_positive(self) -> None:
        model = build_acoustic_transformer()
        assert model.count_parameters().total > 0

    @torch_only
    def test_all_trainable_by_default(self) -> None:
        model = build_acoustic_transformer()
        counts = model.count_parameters()
        assert counts.trainable == counts.total

    @torch_only
    def test_more_layers_more_params(self) -> None:
        small = build_acoustic_transformer(num_layers=2)
        large = build_acoustic_transformer(num_layers=6)
        assert large.count_parameters().total > small.count_parameters().total

    @torch_only
    def test_larger_embed_more_params(self) -> None:
        small = build_acoustic_transformer(embed_dim=128, num_heads=4)
        large = build_acoustic_transformer(embed_dim=256, num_heads=8)
        assert large.count_parameters().total > small.count_parameters().total

    @torch_only
    def test_freeze_reduces_trainable(self) -> None:
        model = build_acoustic_transformer()
        model.freeze()
        assert model.count_parameters().trainable == 0


# ---------------------------------------------------------------------------
# Device transfer
# ---------------------------------------------------------------------------


class TestDeviceTransfer:
    """Tests for device movement."""

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_acoustic_transformer()
        model.to_device("cpu")
        assert model.device.type == "cpu"

    @torch_only
    def test_params_on_cpu(self) -> None:
        model = build_acoustic_transformer()
        model.to_device("cpu")
        for p in model.parameters():
            assert p.device.type == "cpu"

    @pytest.mark.skipif(
        not (_TORCH and torch.cuda.is_available()),
        reason="CUDA not available",
    )
    def test_to_device_cuda(self) -> None:
        model = build_acoustic_transformer()
        model.to_device("cuda")
        assert model.device.type == "cuda"
        out = model(torch.randn(2, 1, *MEL_SHAPE, device="cuda"))
        assert out.is_cuda


# ---------------------------------------------------------------------------
# Inherited functionality
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests for inherited BaseModel methods."""

    @torch_only
    def test_predict(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        preds = model.predict(torch.randn(4, 1, *MEL_SHAPE))
        assert preds.shape == (4,)
        assert preds.min() >= 0 and preds.max() < 5

    @torch_only
    def test_predict_proba(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        proba = model.predict_proba(torch.randn(4, 1, *MEL_SHAPE))
        assert proba.shape == (4, 5)
        assert torch.allclose(proba.sum(dim=-1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_summary(self) -> None:
        model = build_acoustic_transformer()
        s = model.summary()
        assert "Total params" in s
        assert MODEL_NAME in s

    @torch_only
    def test_log_to_tracker(self) -> None:
        calls = []

        class FakeTracker:
            def log_model_info(self, model_name, n_parameters, architecture, **kw):
                calls.append(architecture)

        model = build_acoustic_transformer()
        model.log_to_tracker(FakeTracker())
        assert calls == ["AcousticTransformer"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint round-trips."""

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        model = build_acoustic_transformer(num_classes=5)
        path = tmp_path / "vit.pt"
        model.save_model(path, metadata={"val_f1": 0.92})
        restored = AcousticTransformer.load_model(path)
        assert restored.config.num_classes == 5
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_via_basemodel(self, tmp_path: Path) -> None:
        model = build_acoustic_transformer()
        path = tmp_path / "vit.pt"
        model.save_model(path)
        assert isinstance(BaseModel.load_model(path), AcousticTransformer)

    @torch_only
    def test_custom_arch_survives(self, tmp_path: Path) -> None:
        model = build_acoustic_transformer(embed_dim=128, num_heads=4,
                                            num_layers=3, patch_size=8)
        path = tmp_path / "vit_custom.pt"
        model.save_model(path)
        restored = AcousticTransformer.load_model(path)
        assert restored.feature_dim == 128
        assert restored.num_heads == 4
        assert restored.num_layers == 3

    @torch_only
    def test_restored_same_output(self, tmp_path: Path) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            out1 = model(x)
        path = tmp_path / "vit.pt"
        model.save_model(path)
        restored = AcousticTransformer.load_model(path)
        restored.eval()
        with torch.no_grad():
            out2 = restored(x)
        assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Tests for ONNX export."""

    @torch_only
    def test_export_creates_file(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_acoustic_transformer(num_classes=5)
        path = tmp_path / "vit.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_acoustic_transformer(num_classes=5)
        path = tmp_path / "vit.onnx"
        model.export_onnx(path)
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        path = tmp_path / "vit.onnx"
        model.export_onnx(path, dynamic_batch=True)
        sess = ort.InferenceSession(str(path))
        for bs in (1, 4):
            x = np.random.randn(bs, 1, *MEL_SHAPE).astype(np.float32)
            out = sess.run(None, {"input": x})[0]
            assert out.shape == (bs, 5)


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


class TestMixedPrecision:
    """Tests for AMP compatibility."""

    @torch_only
    def test_autocast_cpu_forward(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(x)
            assert out.shape == (2, 5)
        except RuntimeError:
            pytest.skip("CPU autocast bf16 unsupported")

    @pytest.mark.skipif(
        not (_TORCH and torch.cuda.is_available()),
        reason="CUDA not available",
    )
    def test_autocast_cuda_forward(self) -> None:
        model = build_acoustic_transformer(num_classes=5).to_device("cuda")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE, device="cuda")
        with torch.autocast(device_type="cuda"):
            out = model(x)
        assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Learnability
# ---------------------------------------------------------------------------


class TestLearnability:
    """Tests that the transformer can learn."""

    @torch_only
    def test_can_learn_separable(self) -> None:
        torch.manual_seed(0)
        model = build_acoustic_transformer(
            num_classes=3, input_shape=(32, 64),
            embed_dim=64, num_heads=4, num_layers=2, patch_size=8,
        )
        model.train()

        n_per = 6
        xs, ys = [], []
        for cls in range(3):
            xs.append(torch.randn(n_per, 1, 32, 64) + cls * 3.0)
            ys.append(torch.full((n_per,), cls, dtype=torch.long))
        X, y = torch.cat(xs), torch.cat(ys)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        first = None
        for _ in range(80):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        assert loss.item() < first

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(dim=-1) == y).float().mean().item()
        assert acc > 0.8, f"accuracy {acc:.2f} too low"

    @torch_only
    def test_gradients_flow(self) -> None:
        model = build_acoustic_transformer(num_classes=5)
        model.train()
        loss = nn.CrossEntropyLoss()(
            model(torch.randn(2, 1, *MEL_SHAPE)), torch.tensor([0, 1])
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    @torch_only
    def test_loss_decreases(self) -> None:
        torch.manual_seed(1)
        model = build_acoustic_transformer(
            num_classes=2, input_shape=(32, 64),
            embed_dim=32, num_heads=4, num_layers=1, patch_size=8,
        )
        model.train()
        X = torch.cat([torch.randn(8, 1, 32, 64) - 2,
                       torch.randn(8, 1, 32, 64) + 2])
        y = torch.cat([torch.zeros(8), torch.ones(8)]).long()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        losses = []
        for _ in range(40):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# torch-absent
# ---------------------------------------------------------------------------


class TestTorchAbsent:
    """Tests for graceful behaviour when PyTorch is absent."""

    def test_registered_without_torch(self) -> None:
        assert is_registered(MODEL_NAME)

    def test_defaults_accessible(self) -> None:
        assert DEFAULT_EMBED_DIM > 0
        assert DEFAULT_NUM_HEADS > 0
        assert DEFAULT_NUM_LAYERS > 0
        assert DEFAULT_PATCH_SIZE > 0

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_factory_raises_without_torch(self) -> None:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_acoustic_transformer()