#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/anomaly_autoencoder.py``.

Coverage (50+ tests):
- Registry integration (registered, distinct, BaseModel subclass, coexistence)
- Threshold estimation (std/percentile/iqr/max, validation) — torch-independent
- AnomalyThreshold dataclass (fields, to_dict, frozen)
- Construction for both AE and VAE modes, configurable latent (torch)
- encode / decode / forward / reconstruct shape correctness (torch)
- Exact-shape reconstruction for variable input length (torch)
- VAE reparameterisation + KL loss + deterministic eval (torch)
- anomaly_score (mse/mae), reconstruction_error_map (torch)
- predict_anomaly + anomaly_confidence (torch)
- Feature embedding extraction + feature_dim (torch)
- Inherited BaseModel functionality; disabled predict/predict_proba (torch)
- Checkpoint save/load round-trip (AE and VAE) (torch)
- ONNX export (reconstruction-only graph) (torch + onnx)
- Mixed-precision (autocast) forward (torch)
- ExperimentTracker integration (torch)
- Learnability: AE reconstructs normal better than anomalous after training (torch)
- Anomaly separation: trained AE scores anomalies higher than normal (torch)
- torch-absent graceful errors

Tests requiring PyTorch are guarded with ``@torch_only``.

Run::

    pytest tests/test_anomaly_autoencoder.py -v
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
from src.models.anomaly_autoencoder import (
    MODES,
    SCORE_METRICS,
    THRESHOLD_METHODS,
    AnomalyAutoencoder,
    AnomalyThreshold,
    MODEL_NAME,
    build_anomaly_autoencoder,
)

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")

MEL_SHAPE = (128, 431)
SMALL_SHAPE = (32, 64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for registry integration and coexistence."""

    def test_registered(self) -> None:
        assert is_registered(MODEL_NAME)
        assert MODEL_REGISTRY[MODEL_NAME] is AnomalyAutoencoder

    def test_coexists_with_all_models(self) -> None:
        import src.models.cnn_classifier  # noqa: F401
        import src.models.resnet_acoustic  # noqa: F401
        import src.models.cnn_bilstm  # noqa: F401
        import src.models.cnn_bilstm_attention  # noqa: F401
        from src.models.base_model import list_models
        for name in ("acoustic_cnn", "resnet_acoustic", "cnn_bilstm",
                     "cnn_bilstm_attention", MODEL_NAME):
            assert name in list_models()

    def test_is_basemodel_subclass(self) -> None:
        assert issubclass(AnomalyAutoencoder, BaseModel)

    def test_registry_name_stamped(self) -> None:
        assert AnomalyAutoencoder._registry_name == MODEL_NAME


# ---------------------------------------------------------------------------
# Threshold estimation (torch-independent)
# ---------------------------------------------------------------------------


class TestThresholdEstimation:
    """Tests for threshold estimation (pure NumPy, no torch needed)."""

    REF = np.array([0.01, 0.02, 0.015, 0.012, 0.018,
                    0.011, 0.014, 0.016, 0.013, 0.017])

    def test_std_method(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(self.REF, method="std", k=3.0)
        assert isinstance(t, AnomalyThreshold)
        assert t.value == pytest.approx(self.REF.mean() + 3.0 * self.REF.std())

    def test_percentile_method(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(
            self.REF, method="percentile", percentile=95
        )
        assert t.value == pytest.approx(float(np.percentile(self.REF, 95)))

    def test_iqr_method(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(self.REF, method="iqr")
        assert t.method == "iqr"
        assert t.value > 0

    def test_max_method(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(self.REF, method="max")
        assert t.value == pytest.approx(float(self.REF.max()))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AnomalyAutoencoder.estimate_threshold([])

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            AnomalyAutoencoder.estimate_threshold(self.REF, method="bogus")

    def test_records_statistics(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(self.REF, method="std")
        assert t.mean == pytest.approx(self.REF.mean())
        assert t.std == pytest.approx(self.REF.std())
        assert t.n_samples == len(self.REF)

    def test_k_affects_threshold(self) -> None:
        t1 = AnomalyAutoencoder.estimate_threshold(self.REF, method="std", k=1.0)
        t3 = AnomalyAutoencoder.estimate_threshold(self.REF, method="std", k=3.0)
        assert t3.value > t1.value

    @torch_only
    def test_accepts_torch_tensor(self) -> None:
        t = AnomalyAutoencoder.estimate_threshold(torch.tensor(self.REF))
        assert isinstance(t, AnomalyThreshold)


class TestAnomalyThreshold:
    """Tests for the AnomalyThreshold dataclass."""

    def test_to_dict(self) -> None:
        t = AnomalyThreshold(value=0.5, method="std", mean=0.1, std=0.05,
                             n_samples=10)
        d = t.to_dict()
        assert d["value"] == 0.5 and d["method"] == "std"
        assert d["n_samples"] == 10

    def test_frozen(self) -> None:
        t = AnomalyThreshold(value=0.5, method="std", mean=0.1, std=0.05,
                             n_samples=10)
        with pytest.raises((AttributeError, TypeError)):
            t.value = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for model construction."""

    @torch_only
    def test_builds_ae(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        assert model.mode == "ae"

    @torch_only
    def test_builds_vae(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        assert model.mode == "vae"

    @torch_only
    def test_builds_via_build_model(self) -> None:
        cfg = ModelConfig(model_name=MODEL_NAME, num_classes=5)
        assert isinstance(build_model(MODEL_NAME, cfg), AnomalyAutoencoder)

    @torch_only
    def test_configurable_latent_dim(self) -> None:
        for dim in (16, 64, 256):
            model = build_anomaly_autoencoder(latent_dim=dim)
            assert model.feature_dim == dim

    @torch_only
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            build_anomaly_autoencoder(mode="gan")

    @torch_only
    def test_invalid_score_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="score_metric"):
            build_anomaly_autoencoder(score_metric="rmse")

    @torch_only
    def test_vae_has_mu_logvar(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        assert hasattr(model, "fc_mu")
        assert hasattr(model, "fc_logvar")

    @torch_only
    def test_ae_has_single_latent(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        assert hasattr(model, "fc_latent")

    @torch_only
    def test_deterministic_construction(self) -> None:
        m1 = build_anomaly_autoencoder(random_seed=42)
        m2 = build_anomaly_autoencoder(random_seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Encode / decode / reconstruct
# ---------------------------------------------------------------------------


class TestEncodeDecodeReconstruct:
    """Tests for the core autoencoder operations."""

    @torch_only
    def test_encode_ae_shape(self) -> None:
        model = build_anomaly_autoencoder(mode="ae", latent_dim=64)
        model.eval()
        z = model.encode(torch.randn(4, 1, *MEL_SHAPE))
        assert z.shape == (4, 64)

    @torch_only
    def test_encode_vae_returns_mu_logvar(self) -> None:
        model = build_anomaly_autoencoder(mode="vae", latent_dim=32)
        model.eval()
        mu, logvar = model.encode(torch.randn(4, 1, *MEL_SHAPE))
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    @torch_only
    def test_decode_shape(self) -> None:
        model = build_anomaly_autoencoder(mode="ae", latent_dim=64)
        model.eval()
        z = torch.randn(4, 64)
        recon = model.decode(z, output_size=MEL_SHAPE)
        assert recon.shape == (4, 1, *MEL_SHAPE)

    @torch_only
    def test_reconstruct_matches_input_shape(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        x = torch.randn(4, 1, *MEL_SHAPE)
        recon = model.reconstruct(x)
        assert recon.shape == x.shape

    @torch_only
    def test_forward_returns_recon_and_info(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        recon, info = model(torch.randn(2, 1, *MEL_SHAPE))
        assert recon.shape == (2, 1, *MEL_SHAPE)
        assert "z" in info

    @torch_only
    def test_vae_forward_has_mu_logvar(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        model.eval()
        _, info = model(torch.randn(2, 1, *MEL_SHAPE))
        assert "mu" in info and "logvar" in info

    @torch_only
    def test_reconstruct_exact_shape_variable_length(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        for T in (100, 256, 431, 600):
            x = torch.randn(2, 1, 128, T)
            assert model.reconstruct(x).shape == x.shape

    @torch_only
    def test_3d_input_auto_channel(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        recon = model.reconstruct(torch.randn(2, *MEL_SHAPE))
        assert recon.shape == (2, 1, *MEL_SHAPE)

    @torch_only
    def test_hybrid_3channel(self) -> None:
        model = build_anomaly_autoencoder(mode="ae", in_channels=3)
        model.eval()
        x = torch.randn(2, 3, *MEL_SHAPE)
        assert model.reconstruct(x).shape == x.shape


# ---------------------------------------------------------------------------
# VAE specifics
# ---------------------------------------------------------------------------


class TestVAE:
    """Tests for variational-autoencoder behaviour."""

    @torch_only
    def test_reparam_deterministic_in_eval(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        r1 = model.reconstruct(x)
        r2 = model.reconstruct(x)
        assert torch.allclose(r1, r2)  # no sampling in eval

    @torch_only
    def test_reparam_stochastic_in_train(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        model.train()
        x = torch.randn(2, 1, *MEL_SHAPE)
        torch.manual_seed(1)
        r1, _ = model(x)
        torch.manual_seed(2)
        r2, _ = model(x)
        assert not torch.allclose(r1, r2)  # sampling differs

    @torch_only
    def test_kl_loss_non_negative(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        model.train()
        x = torch.randn(4, 1, *MEL_SHAPE)
        recon, info = model(x)
        _, comps = model.loss_function(x, recon, info)
        assert comps["kl"] >= -1e-4

    @torch_only
    def test_ae_loss_has_zero_kl(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.train()
        x = torch.randn(4, 1, *MEL_SHAPE)
        recon, info = model(x)
        _, comps = model.loss_function(x, recon, info)
        assert comps["kl"] == 0.0

    @torch_only
    def test_loss_total_is_finite(self) -> None:
        for mode in ("ae", "vae"):
            model = build_anomaly_autoencoder(mode=mode)
            model.train()
            x = torch.randn(4, 1, *MEL_SHAPE)
            recon, info = model(x)
            loss, _ = model.loss_function(x, recon, info)
            assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------


class TestAnomalyScoring:
    """Tests for anomaly scoring and prediction."""

    @torch_only
    def test_anomaly_score_shape(self) -> None:
        model = build_anomaly_autoencoder()
        scores = model.anomaly_score(torch.randn(8, 1, *MEL_SHAPE))
        assert scores.shape == (8,)

    @torch_only
    def test_anomaly_score_non_negative(self) -> None:
        model = build_anomaly_autoencoder()
        scores = model.anomaly_score(torch.randn(8, 1, *MEL_SHAPE))
        assert (scores >= 0).all()

    @torch_only
    def test_mae_metric(self) -> None:
        model = build_anomaly_autoencoder(score_metric="mae")
        scores = model.anomaly_score(torch.randn(4, 1, *MEL_SHAPE))
        assert scores.shape == (4,)

    @torch_only
    def test_error_map_shape(self) -> None:
        model = build_anomaly_autoencoder()
        emap = model.reconstruction_error_map(torch.randn(4, 1, *MEL_SHAPE))
        assert emap.shape == (4, *MEL_SHAPE)

    @torch_only
    def test_predict_anomaly_bool(self) -> None:
        model = build_anomaly_autoencoder()
        flags = model.predict_anomaly(torch.randn(8, 1, *MEL_SHAPE), threshold=0.5)
        assert flags.dtype == torch.bool
        assert flags.shape == (8,)

    @torch_only
    def test_predict_anomaly_with_threshold_object(self) -> None:
        model = build_anomaly_autoencoder()
        thr = AnomalyThreshold(value=0.0, method="std", mean=0.0, std=1.0,
                               n_samples=5)
        flags = model.predict_anomaly(torch.randn(4, 1, *MEL_SHAPE), threshold=thr)
        # threshold 0 → all positive-error samples flagged
        assert flags.all()

    @torch_only
    def test_confidence_in_unit_range(self) -> None:
        model = build_anomaly_autoencoder()
        conf = model.anomaly_confidence(torch.randn(8, 1, *MEL_SHAPE),
                                        threshold=0.5)
        assert (conf >= 0).all() and (conf <= 1).all()

    @torch_only
    def test_score_does_not_change_training_mode(self) -> None:
        model = build_anomaly_autoencoder()
        model.train()
        model.anomaly_score(torch.randn(2, 1, *MEL_SHAPE))
        assert model.training  # restored


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for latent embedding extraction."""

    @torch_only
    def test_extract_features_ae(self) -> None:
        model = build_anomaly_autoencoder(mode="ae", latent_dim=64)
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, 64)

    @torch_only
    def test_extract_features_vae_returns_mu(self) -> None:
        model = build_anomaly_autoencoder(mode="vae", latent_dim=32)
        model.eval()
        feats = model.extract_features(torch.randn(4, 1, *MEL_SHAPE))
        assert feats.shape == (4, 32)

    @torch_only
    def test_feature_dim(self) -> None:
        model = build_anomaly_autoencoder(latent_dim=100)
        assert model.feature_dim == 100


# ---------------------------------------------------------------------------
# Inherited functionality
# ---------------------------------------------------------------------------


class TestInheritedFunctionality:
    """Tests for inherited BaseModel methods and disabled classifier methods."""

    @torch_only
    def test_count_parameters(self) -> None:
        model = build_anomaly_autoencoder()
        assert model.count_parameters().total > 0

    @torch_only
    def test_summary(self) -> None:
        model = build_anomaly_autoencoder()
        assert MODEL_NAME in model.summary()

    @torch_only
    def test_predict_disabled(self) -> None:
        model = build_anomaly_autoencoder()
        with pytest.raises(NotImplementedError, match="unsupervised"):
            model.predict(torch.randn(2, 1, *MEL_SHAPE))

    @torch_only
    def test_predict_proba_disabled(self) -> None:
        model = build_anomaly_autoencoder()
        with pytest.raises(NotImplementedError, match="unsupervised"):
            model.predict_proba(torch.randn(2, 1, *MEL_SHAPE))

    @torch_only
    def test_to_device_cpu(self) -> None:
        model = build_anomaly_autoencoder()
        model.to_device("cpu")
        assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint round-trips."""

    @torch_only
    def test_save_load_ae(self, tmp_path: Path) -> None:
        model = build_anomaly_autoencoder(mode="ae", latent_dim=64)
        path = tmp_path / "ae.pt"
        model.save_model(path, metadata={"trained_on": "normal"})
        restored = AnomalyAutoencoder.load_model(path)
        assert restored.mode == "ae"
        assert restored.feature_dim == 64

    @torch_only
    def test_save_load_vae(self, tmp_path: Path) -> None:
        model = build_anomaly_autoencoder(mode="vae", latent_dim=32)
        path = tmp_path / "vae.pt"
        model.save_model(path)
        restored = AnomalyAutoencoder.load_model(path)
        assert restored.mode == "vae"

    @torch_only
    def test_load_via_basemodel(self, tmp_path: Path) -> None:
        model = build_anomaly_autoencoder()
        path = tmp_path / "ae.pt"
        model.save_model(path)
        assert isinstance(BaseModel.load_model(path), AnomalyAutoencoder)

    @torch_only
    def test_weights_preserved(self, tmp_path: Path) -> None:
        model = build_anomaly_autoencoder()
        path = tmp_path / "ae.pt"
        model.save_model(path)
        restored = AnomalyAutoencoder.load_model(path)
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_restored_same_reconstruction(self, tmp_path: Path) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        with torch.no_grad():
            r1 = model.reconstruct(x)
        path = tmp_path / "ae.pt"
        model.save_model(path)
        restored = AnomalyAutoencoder.load_model(path)
        restored.eval()
        with torch.no_grad():
            r2 = restored.reconstruct(x)
        assert torch.allclose(r1, r2, atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Tests for ONNX export (reconstruction-only graph)."""

    @torch_only
    def test_export_creates_file(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_anomaly_autoencoder(mode="ae")
        path = tmp_path / "ae.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_validates(self, tmp_path: Path) -> None:
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_anomaly_autoencoder(mode="ae")
        path = tmp_path / "ae.onnx"
        model.export_onnx(path)
        onnx.checker.check_model(onnx.load(str(path)))

    @torch_only
    def test_vae_exports(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        model = build_anomaly_autoencoder(mode="vae")
        path = tmp_path / "vae.onnx"
        model.export_onnx(path)
        assert path.is_file()

    @torch_only
    def test_onnx_reconstruction_runs(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        path = tmp_path / "ae.onnx"
        model.export_onnx(path, dynamic_batch=True)
        sess = ort.InferenceSession(str(path))
        x = np.random.randn(3, 1, *MEL_SHAPE).astype(np.float32)
        out = sess.run(None, {"input": x})[0]
        assert out.shape == (3, 1, *MEL_SHAPE)


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


class TestMixedPrecision:
    """Tests for AMP compatibility."""

    @torch_only
    def test_autocast_cpu_reconstruct(self) -> None:
        model = build_anomaly_autoencoder(mode="ae")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                recon = model.reconstruct(x)
            assert recon.shape == x.shape
        except RuntimeError:
            pytest.skip("CPU autocast bf16 unsupported")

    @pytest.mark.skipif(
        not (_TORCH and torch.cuda.is_available()),
        reason="CUDA not available",
    )
    def test_autocast_cuda_reconstruct(self) -> None:
        model = build_anomaly_autoencoder(mode="vae").to_device("cuda")
        model.eval()
        x = torch.randn(2, 1, *MEL_SHAPE, device="cuda")
        with torch.autocast(device_type="cuda"):
            recon = model.reconstruct(x)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# ExperimentTracker integration
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    @torch_only
    def test_log_to_tracker(self) -> None:
        calls = []

        class FakeTracker:
            def log_model_info(self, model_name, n_parameters, architecture, **kw):
                calls.append(architecture)

        model = build_anomaly_autoencoder()
        model.log_to_tracker(FakeTracker())
        assert calls == ["AnomalyAutoencoder"]

    @torch_only
    def test_onnx_export_logs_artifact(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        model = build_anomaly_autoencoder()
        model.export_onnx(tmp_path / "ae.onnx", experiment_tracker=FakeTracker())
        assert "onnx" in artifacts


# ---------------------------------------------------------------------------
# Learnability & anomaly separation
# ---------------------------------------------------------------------------


class TestLearnability:
    """Tests that training improves reconstruction and separates anomalies."""

    @torch_only
    def test_reconstruction_improves(self) -> None:
        torch.manual_seed(0)
        model = build_anomaly_autoencoder(
            mode="ae", input_shape=SMALL_SHAPE, latent_dim=32,
            enc_channels=(16, 32),
        )
        model.train()
        # Structured "normal" data: a fixed low-rank pattern + small noise
        base = torch.sin(torch.linspace(0, 6, SMALL_SHAPE[1])).view(1, 1, 1, -1)
        normal = base.expand(16, 1, *SMALL_SHAPE) + 0.05 * torch.randn(16, 1, *SMALL_SHAPE)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(60):
            opt.zero_grad()
            recon, info = model(normal)
            loss, _ = model.loss_function(normal, recon, info)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        assert loss.item() < first

    @torch_only
    def test_anomalies_score_higher(self) -> None:
        torch.manual_seed(0)
        model = build_anomaly_autoencoder(
            mode="ae", input_shape=SMALL_SHAPE, latent_dim=32,
            enc_channels=(16, 32),
        )
        model.train()
        base = torch.sin(torch.linspace(0, 6, SMALL_SHAPE[1])).view(1, 1, 1, -1)
        normal = base.expand(24, 1, *SMALL_SHAPE) + 0.05 * torch.randn(24, 1, *SMALL_SHAPE)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(120):
            opt.zero_grad()
            recon, info = model(normal)
            loss, _ = model.loss_function(normal, recon, info)
            loss.backward()
            opt.step()

        # Anomalies: different structure (random, no sinusoidal pattern)
        anomalous = torch.randn(24, 1, *SMALL_SHAPE)
        normal_scores = model.anomaly_score(normal)
        anom_scores = model.anomaly_score(anomalous)
        # Mean anomaly score should exceed mean normal score
        assert anom_scores.mean() > normal_scores.mean()

    @torch_only
    def test_threshold_separates(self) -> None:
        torch.manual_seed(0)
        model = build_anomaly_autoencoder(
            mode="ae", input_shape=SMALL_SHAPE, latent_dim=32,
            enc_channels=(16, 32),
        )
        model.train()
        base = torch.sin(torch.linspace(0, 6, SMALL_SHAPE[1])).view(1, 1, 1, -1)
        normal = base.expand(32, 1, *SMALL_SHAPE) + 0.05 * torch.randn(32, 1, *SMALL_SHAPE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(120):
            opt.zero_grad()
            recon, info = model(normal)
            loss, _ = model.loss_function(normal, recon, info)
            loss.backward()
            opt.step()

        normal_scores = model.anomaly_score(normal)
        thr = AnomalyAutoencoder.estimate_threshold(
            normal_scores, method="percentile", percentile=95
        )
        anomalous = torch.randn(16, 1, *SMALL_SHAPE)
        # Most anomalies should exceed the 95th-percentile normal threshold
        flags = model.predict_anomaly(anomalous, threshold=thr)
        assert flags.float().mean() > 0.5

    @torch_only
    def test_gradients_flow(self) -> None:
        model = build_anomaly_autoencoder(mode="vae")
        model.train()
        x = torch.randn(2, 1, *MEL_SHAPE)
        recon, info = model(x)
        loss, _ = model.loss_function(x, recon, info)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Constants & torch-absent
# ---------------------------------------------------------------------------


class TestConstantsAndTorchAbsent:
    """Tests for module constants and torch-absent behaviour."""

    def test_modes(self) -> None:
        assert MODES == ("ae", "vae")

    def test_score_metrics(self) -> None:
        assert "mse" in SCORE_METRICS and "mae" in SCORE_METRICS

    def test_threshold_methods(self) -> None:
        for m in ("std", "percentile", "iqr", "max"):
            assert m in THRESHOLD_METHODS

    def test_registered_without_torch(self) -> None:
        assert is_registered(MODEL_NAME)

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_factory_raises_without_torch(self) -> None:
        with pytest.raises(RuntimeError, match="PyTorch"):
            build_anomaly_autoencoder()