#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/cnn_classifier.py``.

Coverage:
- Registry integration (CNN registered, retrievable, BaseModel subclass)
- Architecture construction (stem, residual stages, head)
- :class:`ResidualBlock` skip-connection and downsample behaviour (torch)
- Forward pass output shape for mel-128 input (torch)
- Channel-dim auto-insertion for 3-D input (torch)
- Variable time-axis acceptance via adaptive pooling (torch)
- BatchNorm presence after every conv (torch)
- Weight initialisation (Kaiming conv, normal linear, zero-init residual) (torch)
- Parameter counting / summary inherited from BaseModel (torch)
- Checkpoint save/load round-trip preserves architecture + weights (torch)
- ONNX export with dynamic batch + time axes (torch + onnx)
- :meth:`extract_features` embedding shape (torch)
- predict / predict_proba inherited behaviour (torch)
- ExperimentTracker integration (torch)
- Custom architecture config via ModelConfig.extra (torch)
- Deterministic construction with fixed seed (torch)
- torch-absent graceful errors
- A lightweight learnability smoke test (overfit a tiny batch → F1 target signal)

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_cnn_classifier.py -v
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
from src.models.cnn_classifier import (
    DEFAULT_STAGE_CHANNELS,
    DEFAULT_STEM_CHANNELS,
    MODEL_NAME,
    CNNClassifier,
    build_acoustic_cnn,
)

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")

# Mel-128 reference shapes
MEL_SHAPE = (128, 431)
N_CLASSES = 5


# ---------------------------------------------------------------------------
# Registry / inheritance tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for registry and BaseModel inheritance."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is CNNClassifier

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(CNNClassifier, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert CNNClassifier._registry_name == MODEL_NAME

    def test_forward_is_concrete(self) -> None:
        assert not getattr(CNNClassifier.forward, "__isabstractmethod__", False)

    def test_build_layers_overridden(self) -> None:
        assert CNNClassifier.build_layers is not BaseModel.build_layers

    def test_constants(self) -> None:
        assert MODEL_NAME == "acoustic_cnn"
        assert DEFAULT_STEM_CHANNELS == 32
        assert DEFAULT_STAGE_CHANNELS == (64, 128, 256)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_via_factory(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        assert isinstance(model, CNNClassifier)

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        model = build_model(MODEL_NAME, cfg)
        assert isinstance(model, CNNClassifier)

    @torch_only
    def test_has_stem(self) -> None:
        model = build_acoustic_cnn()
        assert hasattr(model, "stem")

    @torch_only
    def test_has_stages(self) -> None:
        model = build_acoustic_cnn()
        assert hasattr(model, "stages")

    @torch_only
    def test_has_classifier_head(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        assert hasattr(model, "classifier")
        assert model.classifier.out_features == 5

    @torch_only
    def test_has_adaptive_pool(self) -> None:
        model = build_acoustic_cnn()
        assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)

    @torch_only
    def test_has_dropout(self) -> None:
        model = build_acoustic_cnn(dropout=0.4)
        assert isinstance(model.dropout, nn.Dropout)
        assert model.dropout.p == 0.4

    @torch_only
    def test_custom_architecture(self) -> None:
        model = build_acoustic_cnn(
            stem_channels=16,
            stage_channels=(32, 64),
            blocks_per_stage=1,
        )
        assert model.feature_dim == 64

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_acoustic_cnn(random_seed=42)
        m2 = build_acoustic_cnn(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# ResidualBlock tests
# ---------------------------------------------------------------------------


class TestResidualBlock:
    """Tests for the residual block."""

    @torch_only
    def test_identity_shape_preserved(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        block = ResidualBlock(32, 32, stride=1)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    @torch_only
    def test_downsample_changes_shape(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 8, 8)

    @torch_only
    def test_downsample_path_exists(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        block = ResidualBlock(32, 64, stride=2)
        assert block.downsample is not None

    @torch_only
    def test_no_downsample_when_shape_unchanged(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        block = ResidualBlock(32, 32, stride=1)
        assert block.downsample is None

    @torch_only
    def test_has_two_batchnorms(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        block = ResidualBlock(32, 32)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for the forward pass."""

    @torch_only
    def test_output_shape_mel128(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        x = torch.randn(4, 1, 128, 431)
        out = model(x)
        assert out.shape == (4, 5)

    @torch_only
    def test_accepts_3d_input(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        x = torch.randn(4, 128, 431)  # no channel dim
        out = model(x)
        assert out.shape == (4, 5)

    @torch_only
    def test_variable_time_axis(self) -> None:
        # AdaptiveAvgPool makes the network time-agnostic
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        for T in (216, 431, 862):
            x = torch.randn(2, 1, 128, T)
            out = model(x)
            assert out.shape == (2, 5)

    @torch_only
    def test_batch_size_one(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        x = torch.randn(1, 1, 128, 431)
        out = model(x)
        assert out.shape == (1, 5)

    @torch_only
    def test_output_is_finite(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        x = torch.randn(4, 1, 128, 431)
        out = model(x)
        assert torch.all(torch.isfinite(out))

    @torch_only
    def test_3channel_input(self) -> None:
        model = build_acoustic_cnn(num_classes=5, in_channels=3)
        model.eval()
        x = torch.randn(2, 3, 128, 431)
        out = model(x)
        assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for embedding extraction."""

    @torch_only
    def test_extract_features_shape(self) -> None:
        model = build_acoustic_cnn()
        model.eval()
        x = torch.randn(4, 1, 128, 431)
        feats = model.extract_features(x)
        assert feats.shape == (4, model.feature_dim)

    @torch_only
    def test_feature_dim_matches_last_stage(self) -> None:
        model = build_acoustic_cnn(stage_channels=(64, 128, 256))
        assert model.feature_dim == 256

    @torch_only
    def test_features_before_classifier(self) -> None:
        model = build_acoustic_cnn()
        model.eval()
        x = torch.randn(2, 1, 128, 431)
        feats = model.extract_features(x)
        # feature_dim should not equal num_classes (it's the embedding)
        assert feats.shape[1] != model.config.num_classes


# ---------------------------------------------------------------------------
# BatchNorm / weight init tests
# ---------------------------------------------------------------------------


class TestBatchNormAndInit:
    """Tests for BatchNorm presence and weight initialisation."""

    @torch_only
    def test_has_batchnorms(self) -> None:
        model = build_acoustic_cnn()
        bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        assert bn_count > 0

    @torch_only
    def test_conv_count_matches_bn(self) -> None:
        model = build_acoustic_cnn()
        conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        # Every conv is followed by a BN (1:1 in this architecture)
        assert conv_count == bn_count

    @torch_only
    def test_conv_has_no_bias(self) -> None:
        model = build_acoustic_cnn()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                assert m.bias is None

    @torch_only
    def test_zero_init_residual_bn2(self) -> None:
        from src.models.cnn_classifier import ResidualBlock
        model = build_acoustic_cnn()
        for m in model.modules():
            if isinstance(m, ResidualBlock):
                assert torch.allclose(m.bn2.weight, torch.zeros_like(m.bn2.weight))

    @torch_only
    def test_classifier_weight_small(self) -> None:
        model = build_acoustic_cnn()
        # Linear init std=0.01 → weights should be small
        assert model.classifier.weight.std().item() < 0.1


# ---------------------------------------------------------------------------
# Inherited BaseModel functionality tests
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests that inherited BaseModel methods work on the CNN."""

    @torch_only
    def test_count_parameters(self) -> None:
        model = build_acoustic_cnn()
        counts = model.count_parameters()
        assert counts.total > 0
        assert counts.trainable == counts.total

    @torch_only
    def test_summary(self) -> None:
        model = build_acoustic_cnn()
        s = model.summary()
        assert "Total params" in s
        assert "acoustic_cnn" in s

    @torch_only
    def test_predict(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        x = torch.randn(4, 1, 128, 431)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert preds.min() >= 0 and preds.max() < 5

    @torch_only
    def test_predict_proba_sums_to_one(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        x = torch.randn(4, 1, 128, 431)
        proba = model.predict_proba(x)
        assert proba.shape == (4, 5)
        assert torch.allclose(proba.sum(dim=-1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_acoustic_cnn()
        model.to_device("cpu")
        assert model.device.type == "cpu"

    @torch_only
    def test_freeze_unfreeze(self) -> None:
        model = build_acoustic_cnn()
        model.freeze()
        assert model.count_parameters().trainable == 0
        model.unfreeze()
        assert model.count_parameters().trainable == model.count_parameters().total


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for save/load round-trips with the CNN."""

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        model = build_acoustic_cnn(num_classes=5)
        path = tmp_path / "cnn.pt"
        model.save_model(path, metadata={"val_f1": 0.88})

        restored = CNNClassifier.load_model(path)
        assert restored.config.num_classes == 5
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_via_basemodel_resolves_cnn(self, tmp_path: Path) -> None:
        model = build_acoustic_cnn()
        path = tmp_path / "cnn.pt"
        model.save_model(path)
        restored = BaseModel.load_model(path)
        assert isinstance(restored, CNNClassifier)

    @torch_only
    def test_custom_arch_survives_round_trip(self, tmp_path: Path) -> None:
        model = build_acoustic_cnn(stem_channels=16, stage_channels=(32, 64),
                                    blocks_per_stage=1)
        path = tmp_path / "cnn_custom.pt"
        model.save_model(path)
        restored = CNNClassifier.load_model(path)
        assert restored.feature_dim == 64

    @torch_only
    def test_restored_produces_same_output(self, tmp_path: Path) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, 128, 431)
        with torch.no_grad():
            out1 = model(x)
        path = tmp_path / "cnn.pt"
        model.save_model(path)
        restored = CNNClassifier.load_model(path)
        restored.eval()
        with torch.no_grad():
            out2 = restored(x)
        assert torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX export tests
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Tests for ONNX export."""

    @torch_only
    def test_export_creates_file(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_acoustic_cnn(num_classes=5)
        path = tmp_path / "cnn.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_acoustic_cnn(num_classes=5)
        path = tmp_path / "cnn.onnx"
        model.export_onnx(path)
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

    @torch_only
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_acoustic_cnn(num_classes=5)
        model.eval()
        path = tmp_path / "cnn.onnx"
        model.export_onnx(path, dynamic_batch=True)
        sess = ort.InferenceSession(str(path))
        # Different batch sizes should both work
        for bs in (1, 8):
            x = np.random.randn(bs, 1, 128, 431).astype(np.float32)
            out = sess.run(None, {"input": x})[0]
            assert out.shape == (bs, 5)


# ---------------------------------------------------------------------------
# ExperimentTracker integration tests
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for ExperimentTracker logging."""

    @torch_only
    def test_log_to_tracker(self) -> None:
        calls = []

        class FakeTracker:
            def log_model_info(self, model_name, n_parameters, architecture, **kw):
                calls.append((model_name, n_parameters, architecture))

        model = build_acoustic_cnn()
        model.log_to_tracker(FakeTracker())
        assert len(calls) == 1
        assert calls[0][2] == "CNNClassifier"

    @torch_only
    def test_save_with_tracker(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        model = build_acoustic_cnn()
        model.save_model(tmp_path / "cnn.pt", experiment_tracker=FakeTracker())
        assert "checkpoint" in artifacts


# ---------------------------------------------------------------------------
# Learnability smoke test (F1 target signal)
# ---------------------------------------------------------------------------


class TestLearnability:
    """Verify the architecture can actually learn (overfit a tiny batch).

    This is a *smoke test*, not a full training run: it confirms the network
    has the capacity and gradient flow to drive training loss down and reach
    high F1 on a trivially separable batch — a necessary condition for hitting
    the F1 > 0.85 target on real data.
    """

    @torch_only
    def test_can_overfit_tiny_batch(self) -> None:
        torch.manual_seed(0)
        # Small model for speed
        model = build_acoustic_cnn(
            num_classes=3, stem_channels=8,
            stage_channels=(16, 32), blocks_per_stage=1,
        )
        model.train()

        # 3 classes, each with a distinct constant-offset pattern (separable)
        n_per = 4
        xs, ys = [], []
        for cls in range(3):
            x = torch.randn(n_per, 1, 32, 64) + cls * 3.0
            xs.append(x)
            ys.append(torch.full((n_per,), cls, dtype=torch.long))
        X = torch.cat(xs)
        y = torch.cat(ys)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        first_loss = None
        for step in range(60):
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = loss.item()

        # Loss should drop substantially
        assert loss.item() < first_loss

        # Compute F1 on the training batch (overfit → should be high)
        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=-1)
        # Macro F1 by hand
        from collections import Counter
        f1s = []
        for cls in range(3):
            tp = int(((preds == cls) & (y == cls)).sum())
            fp = int(((preds == cls) & (y != cls)).sum())
            fn = int(((preds != cls) & (y == cls)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s)
        # On a separable batch after 60 steps, F1 should exceed the 0.85 target
        assert macro_f1 > 0.85, f"macro F1 {macro_f1:.3f} below target"

    @torch_only
    def test_gradients_flow(self) -> None:
        model = build_acoustic_cnn(num_classes=5)
        model.train()
        x = torch.randn(2, 1, 128, 431)
        y = torch.tensor([0, 1])
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        # Every trainable parameter should have a gradient
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# torch-absent tests
# ---------------------------------------------------------------------------


class TestTorchAbsent:
    """Tests for graceful behaviour when PyTorch is absent."""

    def test_cnn_registered_even_without_torch(self) -> None:
        # Registration happens at import time regardless of torch
        assert is_registered(MODEL_NAME)

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_factory_raises_without_torch(self) -> None:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_acoustic_cnn()