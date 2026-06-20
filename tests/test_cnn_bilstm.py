#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/cnn_bilstm.py``.

Coverage:
- Registry integration (registered, distinct, BaseModel subclass, coexistence)
- :class:`ConvBlock` pooling behaviour (torch)
- :class:`AttentionPooling` shape, weight-normalisation, variable length (torch)
- Construction with default and custom hyperparameters (torch)
- Forward pass for mel / MFCC / CQT / hybrid inputs (torch)
- Variable-length sequence support via dynamic time axis (torch)
- Channel-dim auto-insertion for 3-D input (torch)
- Feature embedding extraction + feature_dim (torch)
- Attention-weight extraction for interpretability (torch)
- Unidirectional vs bidirectional configuration (torch)
- Inherited BaseModel functionality (params, summary, predict) (torch)
- Checkpoint save/load round-trip preserving architecture + weights (torch)
- ONNX export with dynamic axes incl. LSTM (torch + onnx)
- Mixed-precision (autocast) forward pass (torch)
- ExperimentTracker integration (torch)
- Deterministic construction (torch)
- Learnability smoke test on a temporal pattern (torch)
- torch-absent graceful errors

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_cnn_bilstm.py -v
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
from src.models.cnn_bilstm import (
    DEFAULT_CNN_CHANNELS,
    DEFAULT_LSTM_HIDDEN,
    MODEL_NAME,
    CNNBiLSTM,
    build_cnn_bilstm,
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
# Registry / inheritance
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for registry integration and coexistence with other models."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is CNNBiLSTM

    def test_coexists_with_other_models(self) -> None:
        # Importing this module must not disturb the others
        from src.models.cnn_classifier import MODEL_NAME as CNN
        from src.models.resnet_acoustic import MODEL_NAME as RESNET
        assert is_registered(CNN)
        assert is_registered(RESNET)
        assert len({MODEL_NAME, CNN, RESNET}) == 3

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(CNNBiLSTM, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert CNNBiLSTM._registry_name == MODEL_NAME

    def test_forward_concrete(self) -> None:
        assert not getattr(CNNBiLSTM.forward, "__isabstractmethod__", False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_default(self) -> None:
        assert isinstance(build_cnn_bilstm(num_classes=5), CNNBiLSTM)

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        assert isinstance(build_model(MODEL_NAME, cfg), CNNBiLSTM)

    @torch_only
    def test_has_components(self) -> None:
        model = build_cnn_bilstm()
        assert hasattr(model, "cnn")
        assert isinstance(model.lstm, nn.LSTM)
        assert hasattr(model, "attention")
        assert hasattr(model, "classifier")

    @torch_only
    def test_bidirectional_feature_dim(self) -> None:
        model = build_cnn_bilstm(lstm_hidden=128, bidirectional=True)
        assert model.feature_dim == 256  # 2 * hidden

    @torch_only
    def test_unidirectional_feature_dim(self) -> None:
        model = build_cnn_bilstm(lstm_hidden=128, bidirectional=False)
        assert model.feature_dim == 128

    @torch_only
    def test_lstm_is_bidirectional(self) -> None:
        model = build_cnn_bilstm(bidirectional=True)
        assert model.lstm.bidirectional is True

    @torch_only
    def test_custom_lstm_layers(self) -> None:
        model = build_cnn_bilstm(lstm_layers=3)
        assert model.lstm.num_layers == 3

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_cnn_bilstm(random_seed=42)
        m2 = build_cnn_bilstm(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------


class TestConvBlock:
    """Tests for the convolutional block."""

    @torch_only
    def test_pools_both_axes(self) -> None:
        from src.models.cnn_bilstm import ConvBlock
        block = ConvBlock(1, 32, freq_pool=2, time_pool=2)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 32, 32, 32)

    @torch_only
    def test_preserves_time(self) -> None:
        from src.models.cnn_bilstm import ConvBlock
        block = ConvBlock(1, 32, freq_pool=2, time_pool=1)
        x = torch.randn(2, 1, 64, 100)
        out = block(x)
        assert out.shape == (2, 32, 32, 100)  # time preserved


# ---------------------------------------------------------------------------
# AttentionPooling
# ---------------------------------------------------------------------------


class TestAttentionPooling:
    """Tests for attention pooling."""

    @torch_only
    def test_pools_to_fixed_dim(self) -> None:
        from src.models.cnn_bilstm import AttentionPooling
        attn = AttentionPooling(256, 128)
        x = torch.randn(4, 107, 256)
        out = attn(x)
        assert out.shape == (4, 256)

    @torch_only
    def test_weights_sum_to_one(self) -> None:
        from src.models.cnn_bilstm import AttentionPooling
        attn = AttentionPooling(64, 32)
        x = torch.randn(4, 50, 64)
        _, weights = attn(x, return_weights=True)
        assert weights.shape == (4, 50)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_variable_length(self) -> None:
        from src.models.cnn_bilstm import AttentionPooling
        attn = AttentionPooling(64, 32)
        for T in (10, 50, 200):
            x = torch.randn(2, T, 64)
            out = attn(x)
            assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# Forward pass — feature representations
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for forward passes across feature representations."""

    @torch_only
    def test_mel_input(self) -> None:
        model = build_cnn_bilstm(num_classes=5, input_shape=MEL_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_mfcc_input(self) -> None:
        model = build_cnn_bilstm(num_classes=5, input_shape=MFCC_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *MFCC_SHAPE)).shape == (4, 5)

    @torch_only
    def test_cqt_input(self) -> None:
        model = build_cnn_bilstm(num_classes=5, input_shape=CQT_SHAPE)
        model.eval()
        assert model(torch.randn(4, 1, *CQT_SHAPE)).shape == (4, 5)

    @torch_only
    def test_hybrid_3channel(self) -> None:
        model = build_cnn_bilstm(num_classes=5, in_channels=3)
        model.eval()
        assert model(torch.randn(4, 3, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_3d_input_auto_channel(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        assert model(torch.randn(4, *MEL_SHAPE)).shape == (4, 5)

    @torch_only
    def test_variable_length(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        for T in (216, 431, 862):
            assert model(torch.randn(2, 1, 128, T)).shape == (2, 5)

    @torch_only
    def test_batch_size_one(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        assert model(torch.randn(1, 1, *MEL_SHAPE)).shape == (1, 5)

    @torch_only
    def test_output_finite(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        out = model(torch.randn(4, 1, *MEL_SHAPE))
        assert torch.all(torch.isfinite(out))

    @torch_only
    def test_unidirectional_forward(self) -> None:
        model = build_cnn_bilstm(num_classes=5, bidirectional=False)
        model.eval()
        assert model(torch.randn(2, 1, *MEL_SHAPE)).shape == (2, 5)


# ---------------------------------------------------------------------------
# Feature extraction & attention
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for embedding and attention-weight extraction."""

    @torch_only
    def test_extract_features_shape(self) -> None:
        model = build_cnn_bilstm()
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, model.feature_dim)

    @torch_only
    def test_feature_dim_property(self) -> None:
        model = build_cnn_bilstm(lstm_hidden=128, bidirectional=True)
        assert model.feature_dim == 256

    @torch_only
    def test_attention_weights_shape(self) -> None:
        model = build_cnn_bilstm()
        model.eval()
        weights = model.attention_weights(torch.randn(4, 1, *MEL_SHAPE))
        assert weights.dim() == 2
        assert weights.shape[0] == 4

    @torch_only
    def test_attention_weights_sum_to_one(self) -> None:
        model = build_cnn_bilstm()
        model.eval()
        weights = model.attention_weights(torch.randn(4, 1, *MEL_SHAPE))
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-4)

    @torch_only
    def test_embedding_differs_from_logits(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        feats = model.extract_features(torch.randn(2, 1, *MEL_SHAPE))
        assert feats.shape[1] != 5


# ---------------------------------------------------------------------------
# Inherited functionality
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests that BaseModel methods work on the CNN-BiLSTM."""

    @torch_only
    def test_count_parameters(self) -> None:
        model = build_cnn_bilstm()
        counts = model.count_parameters()
        assert counts.total > 0
        assert counts.trainable == counts.total

    @torch_only
    def test_summary(self) -> None:
        model = build_cnn_bilstm()
        s = model.summary()
        assert "Total params" in s
        assert MODEL_NAME in s

    @torch_only
    def test_predict(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        preds = model.predict(torch.randn(4, 1, *MEL_SHAPE))
        assert preds.shape == (4,)
        assert preds.min() >= 0 and preds.max() < 5

    @torch_only
    def test_predict_proba(self) -> None:
        model = build_cnn_bilstm(num_classes=5)
        proba = model.predict_proba(torch.randn(4, 1, *MEL_SHAPE))
        assert proba.shape == (4, 5)
        assert torch.allclose(proba.sum(dim=-1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_cnn_bilstm()
        model.to_device("cpu")
        assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint round-trips."""

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm(num_classes=5)
        path = tmp_path / "bilstm.pt"
        model.save_model(path, metadata={"val_f1": 0.93})
        restored = CNNBiLSTM.load_model(path)
        assert restored.config.num_classes == 5
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_via_basemodel(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm()
        path = tmp_path / "bilstm.pt"
        model.save_model(path)
        restored = BaseModel.load_model(path)
        assert isinstance(restored, CNNBiLSTM)

    @torch_only
    def test_custom_arch_survives_round_trip(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm(lstm_hidden=64, lstm_layers=1,
                                  bidirectional=False)
        path = tmp_path / "bilstm_custom.pt"
        model.save_model(path)
        restored = CNNBiLSTM.load_model(path)
        assert restored.feature_dim == 64

    @torch_only
    def test_restored_same_output(self, tmp_path: Path) -> None:
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            out1 = model(x)
        path = tmp_path / "bilstm.pt"
        model.save_model(path)
        restored = CNNBiLSTM.load_model(path)
        restored.eval()
        with torch.no_grad():
            out2 = restored(x)
        assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Tests for ONNX export (including the LSTM)."""

    @torch_only
    def test_export_creates_file(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_cnn_bilstm(num_classes=5)
        path = tmp_path / "bilstm.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_cnn_bilstm(num_classes=5)
        path = tmp_path / "bilstm.onnx"
        model.export_onnx(path)
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        path = tmp_path / "bilstm.onnx"
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
        model = build_cnn_bilstm(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(x)
            assert out.shape == (2, 5)
        except RuntimeError:
            pytest.skip("CPU autocast bf16 unsupported on this build")

    @pytest.mark.skipif(
        not (_TORCH and torch.cuda.is_available()),
        reason="CUDA not available",
    )
    def test_autocast_cuda_forward(self) -> None:
        model = build_cnn_bilstm(num_classes=5).to_device("cuda")
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

        model = build_cnn_bilstm()
        model.log_to_tracker(FakeTracker())
        assert calls == ["CNNBiLSTM"]

    @torch_only
    def test_save_with_tracker(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        model = build_cnn_bilstm()
        model.save_model(tmp_path / "b.pt", experiment_tracker=FakeTracker())
        assert "checkpoint" in artifacts


# ---------------------------------------------------------------------------
# Learnability smoke test (temporal pattern)
# ---------------------------------------------------------------------------


class TestLearnability:
    """Verify the network can learn a temporal pattern.

    The classes here differ only in their *temporal structure* (a pulse at
    different time positions), not their global statistics — exactly the regime
    where a recurrent model should outperform global-pooling CNNs.
    """

    @torch_only
    def test_learns_temporal_pattern(self) -> None:
        torch.manual_seed(0)
        model = build_cnn_bilstm(
            num_classes=3, input_shape=(32, 64),
            cnn_channels=(16, 32), freq_pool=2,
            lstm_hidden=32, lstm_layers=1, attention_dim=32,
        )
        model.train()

        # 3 classes: a high-energy band placed in different TIME thirds.
        # Global statistics are identical; only temporal position differs.
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
        model = build_cnn_bilstm(num_classes=5)
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
            build_cnn_bilstm()