#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/cnn_bilstm_attention.py``.

Coverage:
- Registry integration (registered, distinct, BaseModel subclass, coexistence)
- Composition: reuses SelfAttentionBlock + AttentionPooling from the framework
- Construction with default and custom hyperparameters (torch)
- embed_dim/num_heads divisibility guard (torch)
- Forward pass for mel / MFCC / CQT / hybrid inputs (torch)
- Dynamic sequence length support (torch)
- Channel-dim auto-insertion for 3-D input (torch)
- Feature embedding extraction + feature_dim (torch)
- Attention-map extraction (both self-attention + pooling stages) (torch)
- Configurable num_heads and BiLSTM depth (torch)
- Inherited BaseModel functionality (params, summary, predict) (torch)
- Checkpoint save/load round-trip preserving architecture + weights (torch)
- ONNX export with dynamic axes (torch + onnx)
- Mixed-precision (autocast) forward pass (torch)
- ExperimentTracker integration (torch)
- Deterministic construction (torch)
- Learnability smoke test on a temporal pattern (torch)
- torch-absent graceful errors

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_cnn_bilstm_attention.py -v
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
from src.models.cnn_bilstm_attention import (
    MODEL_NAME,
    CNNBiLSTMAttention,
    build_cnn_bilstm_attention,
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
# Registry / composition
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for registry integration and coexistence."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is CNNBiLSTMAttention

    def test_coexists_with_all_models(self) -> None:
        import src.models.cnn_classifier  # noqa: F401
        import src.models.resnet_acoustic  # noqa: F401
        import src.models.cnn_bilstm  # noqa: F401
        from src.models.base_model import list_models
        for name in ("acoustic_cnn", "resnet_acoustic", "cnn_bilstm", MODEL_NAME):
            assert name in list_models()

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(CNNBiLSTMAttention, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert CNNBiLSTMAttention._registry_name == MODEL_NAME

    def test_reuses_attention_framework(self) -> None:
        import inspect
        import src.models.cnn_bilstm_attention as m
        src = inspect.getsource(m)
        assert "from src.models.attention_modules import" in src
        assert "SelfAttentionBlock" in src
        assert "AttentionPooling" in src


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_default(self) -> None:
        assert isinstance(build_cnn_bilstm_attention(num_classes=5), CNNBiLSTMAttention)

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        assert isinstance(build_model(MODEL_NAME, cfg), CNNBiLSTMAttention)

    @torch_only
    def test_has_all_stages(self) -> None:
        model = build_cnn_bilstm_attention()
        assert hasattr(model, "cnn")
        assert isinstance(model.lstm, nn.LSTM)
        assert hasattr(model, "self_attention")
        assert hasattr(model, "attention_pool")
        assert hasattr(model, "classifier")

    @torch_only
    def test_configurable_heads(self) -> None:
        for h in (1, 2, 4, 8):
            model = build_cnn_bilstm_attention(num_heads=h)
            assert model.num_heads == h

    @torch_only
    def test_configurable_lstm_depth(self) -> None:
        for depth in (1, 2, 3):
            model = build_cnn_bilstm_attention(lstm_layers=depth)
            assert model.lstm.num_layers == depth

    @torch_only
    def test_divisibility_guard(self) -> None:
        # 2*64 = 128, not divisible by 7
        with pytest.raises(ValueError, match="divisible"):
            build_cnn_bilstm_attention(lstm_hidden=64, num_heads=7)

    @torch_only
    def test_bidirectional_embed_dim(self) -> None:
        model = build_cnn_bilstm_attention(lstm_hidden=128, bidirectional=True)
        assert model.feature_dim == 256

    @torch_only
    def test_unidirectional_embed_dim(self) -> None:
        model = build_cnn_bilstm_attention(lstm_hidden=128, bidirectional=False,
                                            num_heads=4)
        assert model.feature_dim == 128

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_cnn_bilstm_attention(random_seed=42)
        m2 = build_cnn_bilstm_attention(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for forward passes across feature representations."""

    @torch_only
    def test_mel_input(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5, input_shape=MEL_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_mfcc_input(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5, input_shape=MFCC_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *MFCC_SHAPE)).shape == (4, 5)

    @torch_only
    def test_cqt_input(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5, input_shape=CQT_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *CQT_SHAPE)).shape == (4, 5)

    @torch_only
    def test_hybrid_3channel(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5, in_channels=3)
        model.eval()
        assert model(torch.randn(4, 3, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_3d_input_auto_channel(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        assert model(torch.randn(4, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_variable_length(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        for T in (216, 431, 862):
            assert model(torch.randn(2, 1, 128, T)).shape == (2, 5)

    @torch_only
    def test_batch_size_one(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        assert model(torch.randn(1, 1, *MEL_SHAPE)).shape == (1, 5)

    @torch_only
    def test_output_finite(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        out = model(torch.randn(4, 1, *MEL_SHAPE))
        assert torch.all(torch.isfinite(out))

    @torch_only
    def test_unidirectional_forward(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5, bidirectional=False,
                                            num_heads=4)
        model.eval()
        assert model(torch.randn(2, 1, *MEL_SHAPE)).shape == (2, 5)


# ---------------------------------------------------------------------------
# Feature & attention extraction
# ---------------------------------------------------------------------------


class TestFeatureAndAttention:
    """Tests for embedding and attention-map extraction."""

    @torch_only
    def test_extract_features_shape(self) -> None:
        model = build_cnn_bilstm_attention()
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, model.feature_dim)

    @torch_only
    def test_attention_maps_keys(self) -> None:
        model = build_cnn_bilstm_attention()
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        assert "self_attention" in maps
        assert "pooling" in maps

    @torch_only
    def test_self_attention_map_shape(self) -> None:
        model = build_cnn_bilstm_attention()
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        sa = maps["self_attention"]
        # (B, T, T) head-averaged
        assert sa.dim() == 3
        assert sa.shape[0] == 2
        assert sa.shape[1] == sa.shape[2]

    @torch_only
    def test_pooling_weights_shape_and_sum(self) -> None:
        model = build_cnn_bilstm_attention()
        model.eval()
        maps = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))
        pw = maps["pooling"]
        assert pw.dim() == 2 and pw.shape[0] == 2
        assert torch.allclose(pw.sum(dim=1), torch.ones(2), atol=1e-4)

    @torch_only
    def test_self_attention_rows_sum_to_one(self) -> None:
        model = build_cnn_bilstm_attention()
        model.eval()
        sa = model.attention_maps(torch.randn(2, 1, *MEL_SHAPE))["self_attention"]
        assert torch.allclose(sa.sum(dim=-1),
                              torch.ones(sa.shape[0], sa.shape[1]), atol=1e-4)

    @torch_only
    def test_embedding_differs_from_logits(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        feats = model.extract_features(torch.randn(2, 1, *MEL_SHAPE))
        assert feats.shape[1] != 5


# ---------------------------------------------------------------------------
# Inherited functionality
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests that BaseModel methods work."""

    @torch_only
    def test_count_parameters(self) -> None:
        model = build_cnn_bilstm_attention()
        counts = model.count_parameters()
        assert counts.total > 0
        assert counts.trainable == counts.total

    @torch_only
    def test_summary(self) -> None:
        model = build_cnn_bilstm_attention()
        s = model.summary()
        assert "Total params" in s
        assert MODEL_NAME in s

    @torch_only
    def test_predict_proba_sums_to_one(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        proba = model.predict_proba(torch.randn(4, 1, *MEL_SHAPE))
        assert proba.shape == (4, 5)
        assert torch.allclose(proba.sum(dim=-1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_predict(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        preds = model.predict(torch.randn(4, 1, *MEL_SHAPE))
        assert preds.shape == (4,)
        assert preds.min() >= 0 and preds.max() < 5

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_cnn_bilstm_attention()
        model.to_device("cpu")
        assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint round-trips."""

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        path = tmp_path / "m.pt"
        model.save_model(path, metadata={"val_f1": 0.94})
        restored = CNNBiLSTMAttention.load_model(path)
        assert restored.config.num_classes == 5
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_via_basemodel(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm_attention()
        path = tmp_path / "m.pt"
        model.save_model(path)
        restored = BaseModel.load_model(path)
        assert isinstance(restored, CNNBiLSTMAttention)

    @torch_only
    def test_custom_arch_survives_round_trip(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm_attention(lstm_hidden=64, lstm_layers=1,
                                            num_heads=4, bidirectional=False)
        path = tmp_path / "m_custom.pt"
        model.save_model(path)
        restored = CNNBiLSTMAttention.load_model(path)
        assert restored.feature_dim == 64
        assert restored.num_heads == 4

    @torch_only
    def test_restored_same_output(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            out1 = model(x)
        path = tmp_path / "m.pt"
        model.save_model(path)
        restored = CNNBiLSTMAttention.load_model(path)
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
        model = build_cnn_bilstm_attention(num_classes=5)
        path = tmp_path / "m.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_cnn_bilstm_attention(num_classes=5)
        path = tmp_path / "m.onnx"
        model.export_onnx(path)
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_cnn_bilstm_attention(num_classes=5)
        model.eval()
        path = tmp_path / "m.onnx"
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
        model = build_cnn_bilstm_attention(num_classes=5)
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
        model = build_cnn_bilstm_attention(num_classes=5).to_device("cuda")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE, device="cuda")
        with torch.autocast(device_type="cuda"):
            out = model(x)
        assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# ExperimentTracker integration
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for ExperimentTracker logging."""

    @torch_only
    def test_log_to_tracker(self) -> None:
        calls = []

        class FakeTracker:
            def log_model_info(self, model_name, n_parameters, architecture, **kw):
                calls.append(architecture)

        model = build_cnn_bilstm_attention()
        model.log_to_tracker(FakeTracker())
        assert calls == ["CNNBiLSTMAttention"]

    @torch_only
    def test_save_with_tracker(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        model = build_cnn_bilstm_attention()
        model.save_model(tmp_path / "m.pt", experiment_tracker=FakeTracker())
        assert "checkpoint" in artifacts


# ---------------------------------------------------------------------------
# Learnability (temporal pattern)
# ---------------------------------------------------------------------------


class TestLearnability:
    """Verify the model learns a temporal pattern (necessary for F1 gains)."""

    @torch_only
    def test_learns_temporal_pattern(self) -> None:
        torch.manual_seed(0)
        model = build_cnn_bilstm_attention(
            num_classes=3, input_shape=(32, 64),
            cnn_channels=(16, 32), freq_pool=2,
            lstm_hidden=32, lstm_layers=1, num_heads=4,
            pool_attention_dim=32,
        )
        model.train()

        # Classes differ only in the temporal position of an energy band.
        n_per = 6
        T = 64
        xs, ys = [], []
        for cls in range(3):
            base = torch.randn(n_per, 1, 32, T) * 0.1
            start = cls * (T // 3)
            base[:, :, :, start:start + T // 3] += 2.0
            xs.append(base)
            ys.append(torch.full((n_per,), cls, dtype=torch.long))
        X, y = torch.cat(xs), torch.cat(ys)

        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()
        first = None
        for _ in range(100):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        assert loss.item() < first

        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        assert acc > 0.8, f"temporal accuracy {acc:.2f} too low"

    @torch_only
    def test_gradients_flow(self) -> None:
        model = build_cnn_bilstm_attention(num_classes=5)
        model.train()
        loss = nn.CrossEntropyLoss()(
            model(torch.randn(2, 1, *MEL_SHAPE)), torch.tensor([0, 1])
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# torch-absent
# ---------------------------------------------------------------------------


class TestTorchAbsent:
    """Tests for graceful behaviour when PyTorch is absent."""

    def test_registered_without_torch(self) -> None:
        assert is_registered(MODEL_NAME)

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_factory_raises_without_torch(self) -> None:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_cnn_bilstm_attention()