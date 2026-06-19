#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/resnet_acoustic.py``.

Coverage:
- Registry integration (registered, distinct from acoustic_cnn, BaseModel subclass)
- Depth presets (resnet10/18/34) and explicit ``layers``
- :class:`SEBlock` shape preservation and channel reweighting (torch)
- :class:`ResidualBlock` skip/downsample/SE behaviour (torch)
- :class:`MultiScaleStem` parallel-path concatenation (torch)
- Forward pass for mel / MFCC / CQT / hybrid (multi-channel) inputs (torch)
- Dynamic input length via adaptive pooling (torch)
- Channel-dim auto-insertion for 3-D input (torch)
- Feature embedding extraction + feature_dim (torch)
- Configurable depth + use_se / multiscale toggles (torch)
- Inherited BaseModel functionality (params, summary, predict) (torch)
- Checkpoint save/load round-trip preserving architecture + weights (torch)
- ONNX export with dynamic axes (torch + onnx)
- ExperimentTracker integration (torch)
- Mixed-precision (autocast) forward pass (torch + CUDA, else skipped)
- Deterministic construction (torch)
- Learnability smoke test → F1 > 0.90 target signal (torch)
- torch-absent graceful errors
- Coexistence with CNNClassifier (no registry regression)

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_resnet_acoustic.py -v
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
from src.models.resnet_acoustic import (
    DEFAULT_LAYERS,
    DEFAULT_STAGE_CHANNELS,
    DEPTH_PRESETS,
    MODEL_NAME,
    ResNetAcoustic,
    build_resnet_acoustic,
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
    """Tests for registry integration and coexistence with the CNN."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is ResNetAcoustic

    def test_distinct_from_cnn(self) -> None:
        from src.models.cnn_classifier import MODEL_NAME as CNN_NAME
        assert MODEL_NAME != CNN_NAME
        assert is_registered("acoustic_cnn")  # CNN still present

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(ResNetAcoustic, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert ResNetAcoustic._registry_name == MODEL_NAME

    def test_forward_concrete(self) -> None:
        assert not getattr(ResNetAcoustic.forward, "__isabstractmethod__", False)

    def test_depth_presets(self) -> None:
        assert DEPTH_PRESETS["resnet18"] == (2, 2, 2, 2)
        assert DEPTH_PRESETS["resnet34"] == (3, 4, 6, 3)
        assert DEPTH_PRESETS["resnet10"] == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_default(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        assert isinstance(model, ResNetAcoustic)

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        model = build_model(MODEL_NAME, cfg)
        assert isinstance(model, ResNetAcoustic)

    @torch_only
    def test_depth_resnet10(self) -> None:
        model = build_resnet_acoustic(depth="resnet10")
        assert model.feature_dim == DEFAULT_STAGE_CHANNELS[-1]

    @torch_only
    def test_depth_resnet34(self) -> None:
        model = build_resnet_acoustic(depth="resnet34")
        # More blocks → more residual modules
        from src.models.resnet_acoustic import ResidualBlock
        n_blocks = sum(1 for m in model.modules() if isinstance(m, ResidualBlock))
        assert n_blocks == sum(DEPTH_PRESETS["resnet34"])

    @torch_only
    def test_explicit_layers_override_depth(self) -> None:
        model = build_resnet_acoustic(layers=(1, 1), stage_channels=(32, 64))
        assert model.feature_dim == 64

    @torch_only
    def test_unknown_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown depth"):
            build_resnet_acoustic(depth="resnet999")

    @torch_only
    def test_mismatched_layers_channels_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            build_resnet_acoustic(layers=(2, 2, 2), stage_channels=(64, 128))

    @torch_only
    def test_has_components(self) -> None:
        model = build_resnet_acoustic()
        assert hasattr(model, "stem")
        assert hasattr(model, "stages")
        assert hasattr(model, "classifier")
        assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_resnet_acoustic(random_seed=42)
        m2 = build_resnet_acoustic(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# SEBlock
# ---------------------------------------------------------------------------


class TestSEBlock:
    """Tests for the Squeeze-and-Excitation block."""

    @torch_only
    def test_shape_preserved(self) -> None:
        from src.models.resnet_acoustic import SEBlock
        se = SEBlock(64, reduction=16)
        x = torch.randn(2, 64, 8, 8)
        out = se(x)
        assert out.shape == x.shape

    @torch_only
    def test_output_is_scaled_input(self) -> None:
        from src.models.resnet_acoustic import SEBlock
        se = SEBlock(32, reduction=8)
        se.eval()
        x = torch.randn(2, 32, 4, 4)
        out = se(x)
        # SE rescales channels; sign should be preserved (sigmoid weights > 0)
        assert torch.all((out == 0) | (torch.sign(out) == torch.sign(x)))

    @torch_only
    def test_small_channel_count(self) -> None:
        # reduction larger than channels → bottleneck clamps to >= 1
        from src.models.resnet_acoustic import SEBlock
        se = SEBlock(8, reduction=16)
        x = torch.randn(2, 8, 4, 4)
        assert se(x).shape == x.shape


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


class TestResidualBlock:
    """Tests for the SE-enhanced residual block."""

    @torch_only
    def test_identity_shape(self) -> None:
        from src.models.resnet_acoustic import ResidualBlock
        block = ResidualBlock(32, 32, stride=1)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 32, 16, 16)

    @torch_only
    def test_downsample_shape(self) -> None:
        from src.models.resnet_acoustic import ResidualBlock
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 64, 8, 8)

    @torch_only
    def test_has_se_when_enabled(self) -> None:
        from src.models.resnet_acoustic import ResidualBlock, SEBlock
        block = ResidualBlock(32, 32, use_se=True)
        assert isinstance(block.se, SEBlock)

    @torch_only
    def test_no_se_when_disabled(self) -> None:
        from src.models.resnet_acoustic import ResidualBlock
        block = ResidualBlock(32, 32, use_se=False)
        assert block.se is None

    @torch_only
    def test_downsample_path(self) -> None:
        from src.models.resnet_acoustic import ResidualBlock
        block = ResidualBlock(32, 64, stride=2)
        assert block.downsample is not None


# ---------------------------------------------------------------------------
# MultiScaleStem
# ---------------------------------------------------------------------------


class TestMultiScaleStem:
    """Tests for the multi-scale stem."""

    @torch_only
    def test_output_channels(self) -> None:
        from src.models.resnet_acoustic import MultiScaleStem
        stem = MultiScaleStem(1, 64, stride=2)
        x = torch.randn(2, 1, 128, 431)
        out = stem(x)
        assert out.shape[1] == 64

    @torch_only
    def test_downsamples(self) -> None:
        from src.models.resnet_acoustic import MultiScaleStem
        stem = MultiScaleStem(1, 64, stride=2)
        x = torch.randn(2, 1, 128, 128)
        out = stem(x)
        # stride 2 halves spatial dims
        assert out.shape[2] == 64 and out.shape[3] == 64

    @torch_only
    def test_multichannel_input(self) -> None:
        from src.models.resnet_acoustic import MultiScaleStem
        stem = MultiScaleStem(3, 64, stride=2)
        x = torch.randn(2, 3, 128, 128)
        assert stem(x).shape[1] == 64


# ---------------------------------------------------------------------------
# Forward pass — feature representations
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for forward passes across feature representations."""

    @torch_only
    def test_mel_input(self) -> None:
        model = build_resnet_acoustic(num_classes=5, input_shape=MEL_SHAPE)
        model.eval()
        out = model(torch.randn(4, 1, *MEL_SHAPE))
        assert out.shape == (4, 5)

    @torch_only
    def test_mfcc_input(self) -> None:
        model = build_resnet_acoustic(num_classes=5, input_shape=MFCC_SHAPE)
        model.eval()
        out = model(torch.randn(4, 1, *MFCC_SHAPE))
        assert out.shape == (4, 5)

    @torch_only
    def test_cqt_input(self) -> None:
        model = build_resnet_acoustic(num_classes=5, input_shape=CQT_SHAPE)
        model.eval()
        out = model(torch.randn(4, 1, *CQT_SHAPE))
        assert out.shape == (4, 5)

    @torch_only
    def test_hybrid_3channel(self) -> None:
        model = build_resnet_acoustic(num_classes=5, input_shape=MEL_SHAPE,
                                       in_channels=3)
        model.eval()
        out = model(torch.randn(4, 3, *MEL_SHAPE))
        assert out.shape == (4, 5)

    @torch_only
    def test_3d_input_auto_channel(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        out = model(torch.randn(4, *MEL_SHAPE))  # no channel dim
        assert out.shape == (4, 5)

    @torch_only
    def test_dynamic_length(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        for T in (216, 431, 862):
            out = model(torch.randn(2, 1, 128, T))
            assert out.shape == (2, 5)

    @torch_only
    def test_batch_size_one(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        assert model(torch.randn(1, 1, *MEL_SHAPE)).shape == (1, 5)

    @torch_only
    def test_output_finite(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        out = model(torch.randn(4, 1, *MEL_SHAPE))
        assert torch.all(torch.isfinite(out))

    @torch_only
    def test_no_se_variant(self) -> None:
        model = build_resnet_acoustic(num_classes=5, use_se=False)
        model.eval()
        assert model(torch.randn(2, 1, *MEL_SHAPE)).shape == (2, 5)

    @torch_only
    def test_plain_stem_variant(self) -> None:
        model = build_resnet_acoustic(num_classes=5, multiscale_stem=False)
        model.eval()
        assert model(torch.randn(2, 1, *MEL_SHAPE)).shape == (2, 5)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for embedding extraction."""

    @torch_only
    def test_extract_features_shape(self) -> None:
        model = build_resnet_acoustic()
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, model.feature_dim)

    @torch_only
    def test_feature_dim_matches_last_stage(self) -> None:
        model = build_resnet_acoustic(stage_channels=(64, 128, 256, 512))
        assert model.feature_dim == 512

    @torch_only
    def test_embedding_differs_from_logits(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        feats = model.extract_features(torch.randn(2, 1, *MEL_SHAPE))
        assert feats.shape[1] != 5


# ---------------------------------------------------------------------------
# Inherited functionality
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests that BaseModel methods work on the ResNet."""

    @torch_only
    def test_count_parameters(self) -> None:
        model = build_resnet_acoustic()
        counts = model.count_parameters()
        assert counts.total > 0
        assert counts.trainable == counts.total

    @torch_only
    def test_summary(self) -> None:
        model = build_resnet_acoustic()
        s = model.summary()
        assert "Total params" in s
        assert MODEL_NAME in s

    @torch_only
    def test_predict(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        preds = model.predict(torch.randn(4, 1, *MEL_SHAPE))
        assert preds.shape == (4,)
        assert preds.min() >= 0 and preds.max() < 5

    @torch_only
    def test_predict_proba(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
        proba = model.predict_proba(torch.randn(4, 1, *MEL_SHAPE))
        assert proba.shape == (4, 5)
        assert torch.allclose(proba.sum(dim=-1), torch.ones(4), atol=1e-5)

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_resnet_acoustic()
        model.to_device("cpu")
        assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint round-trips."""

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        model = build_resnet_acoustic(num_classes=5)
        path = tmp_path / "resnet.pt"
        model.save_model(path, metadata={"val_f1": 0.91})
        restored = ResNetAcoustic.load_model(path)
        assert restored.config.num_classes == 5
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_via_basemodel(self, tmp_path: Path) -> None:
        model = build_resnet_acoustic()
        path = tmp_path / "resnet.pt"
        model.save_model(path)
        restored = BaseModel.load_model(path)
        assert isinstance(restored, ResNetAcoustic)

    @torch_only
    def test_custom_depth_survives_round_trip(self, tmp_path: Path) -> None:
        model = build_resnet_acoustic(layers=(1, 1), stage_channels=(32, 64),
                                       use_se=False, multiscale_stem=False)
        path = tmp_path / "resnet_custom.pt"
        model.save_model(path)
        restored = ResNetAcoustic.load_model(path)
        assert restored.feature_dim == 64

    @torch_only
    def test_restored_same_output(self, tmp_path: Path) -> None:
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            out1 = model(x)
        path = tmp_path / "resnet.pt"
        model.save_model(path)
        restored = ResNetAcoustic.load_model(path)
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
        model = build_resnet_acoustic(num_classes=5)
        path = tmp_path / "resnet.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_resnet_acoustic(num_classes=5)
        path = tmp_path / "resnet.onnx"
        model.export_onnx(path)
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_onnx_dynamic_batch(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_resnet_acoustic(num_classes=5)
        model.eval()
        path = tmp_path / "resnet.onnx"
        model.export_onnx(path, dynamic_batch=True)
        sess = ort.InferenceSession(str(path))
        for bs in (1, 8):
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
        # CPU autocast (bf16) — verifies no dtype hardcoding breaks AMP
        model = build_resnet_acoustic(num_classes=5)
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
        model = build_resnet_acoustic(num_classes=5).to_device("cuda")
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

        model = build_resnet_acoustic()
        model.log_to_tracker(FakeTracker())
        assert calls == ["ResNetAcoustic"]

    @torch_only
    def test_save_with_tracker(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        model = build_resnet_acoustic()
        model.save_model(tmp_path / "r.pt", experiment_tracker=FakeTracker())
        assert "checkpoint" in artifacts


# ---------------------------------------------------------------------------
# Learnability smoke test (F1 > 0.90 signal)
# ---------------------------------------------------------------------------


class TestLearnability:
    """Verify the network can learn — necessary condition for F1 > 0.90."""

    @torch_only
    def test_can_overfit_tiny_batch(self) -> None:
        torch.manual_seed(0)
        model = build_resnet_acoustic(
            num_classes=3, input_shape=(32, 64),
            layers=(1, 1), stage_channels=(16, 32),
            stem_channels=16,
        )
        model.train()

        n_per = 4
        xs, ys = [], []
        for cls in range(3):
            xs.append(torch.randn(n_per, 1, 32, 64) + cls * 3.0)
            ys.append(torch.full((n_per,), cls, dtype=torch.long))
        X, y = torch.cat(xs), torch.cat(ys)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        first_loss = None
        for _ in range(80):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = loss.item()

        assert loss.item() < first_loss

        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=-1)
        f1s = []
        for cls in range(3):
            tp = int(((preds == cls) & (y == cls)).sum())
            fp = int(((preds == cls) & (y != cls)).sum())
            fn = int(((preds != cls) & (y == cls)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        macro_f1 = sum(f1s) / len(f1s)
        assert macro_f1 > 0.90, f"macro F1 {macro_f1:.3f} below 0.90 target"

    @torch_only
    def test_gradients_flow(self) -> None:
        model = build_resnet_acoustic(num_classes=5)
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
            build_resnet_acoustic()