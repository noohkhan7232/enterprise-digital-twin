#!/usr/bin/env python3
"""Comprehensive test suite for ``src/models/base_model.py``.

Coverage:
- :class:`ModelConfig` validation, hashing, to_dict/from_dict round-trips
- :func:`register_model` / :func:`build_model` / :func:`list_models` registry
- Duplicate-registration rejection and idempotent same-class re-registration
- :func:`set_global_seed` reproducibility
- :func:`resolve_device` resolution logic
- :class:`BaseModel` abstract enforcement (ABCMeta, abstract forward)
- :meth:`BaseModel.count_parameters` accounting (torch)
- :meth:`BaseModel.summary` / :meth:`print_summary` (torch)
- :meth:`BaseModel.save_model` / :meth:`load_model` round-trip (torch)
- :meth:`BaseModel.inspect_checkpoint` (torch)
- :meth:`BaseModel.export_onnx` (torch + onnx)
- :meth:`BaseModel.to_device` / :attr:`device` (torch)
- :meth:`BaseModel.predict` / :meth:`predict_proba` (torch)
- :meth:`BaseModel.freeze` / :meth:`unfreeze` (torch)
- :meth:`BaseModel.log_to_tracker` ExperimentTracker integration
- torch-absent graceful errors
- :class:`ParameterCount` structure

Tests that require PyTorch are guarded with ``@pytest.mark.skipif(not _TORCH)``.

Run::

    pytest tests/test_base_model.py -v
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
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_CLASS_NAMES,
    MODEL_REGISTRY,
    BaseModel,
    ModelConfig,
    ParameterCount,
    build_model,
    is_registered,
    list_models,
    register_model,
    resolve_device,
    set_global_seed,
)

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Test model definitions (only usable when torch present)
# ---------------------------------------------------------------------------

if _TORCH:

    class _TinyClassifier(BaseModel):
        """Minimal concrete model for testing."""

        def build_layers(self) -> None:
            c, *rest = self.config.batched_input_shape
            flat = c
            for dim in rest:
                flat *= dim
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(flat, self.config.num_classes)
            self.dropout = nn.Dropout(self.config.dropout)

        def forward(self, x):
            x = self.flatten(x)
            x = self.dropout(x)
            return self.fc(x)


def _register_tiny(name: str = "tiny_test_clf"):
    """Register the tiny classifier under a unique name (idempotent)."""
    if not _TORCH:
        return None
    if name not in MODEL_REGISTRY:
        register_model(name)(_TinyClassifier)
    return name


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for :class:`ModelConfig`."""

    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.num_classes == len(DEFAULT_CLASS_NAMES)
        assert cfg.input_shape == (128, 431)
        assert cfg.random_seed == 42
        assert cfg.in_channels == 1

    def test_num_classes_validation(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            ModelConfig(num_classes=0)

    def test_dropout_validation(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)

    def test_in_channels_validation(self) -> None:
        with pytest.raises(ValueError, match="in_channels"):
            ModelConfig(in_channels=0)

    def test_empty_input_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="input_shape"):
            ModelConfig(input_shape=())

    def test_config_hash_deterministic(self) -> None:
        assert ModelConfig(num_classes=5).config_hash == ModelConfig(num_classes=5).config_hash

    def test_config_hash_sensitive(self) -> None:
        assert ModelConfig(num_classes=5).config_hash != ModelConfig(num_classes=3).config_hash

    def test_config_hash_length(self) -> None:
        assert len(ModelConfig().config_hash) == 12

    def test_batched_input_shape_2d(self) -> None:
        cfg = ModelConfig(input_shape=(128, 431), in_channels=1)
        assert cfg.batched_input_shape == (1, 128, 431)

    def test_batched_input_shape_3channel(self) -> None:
        cfg = ModelConfig(input_shape=(128, 431), in_channels=3)
        assert cfg.batched_input_shape == (3, 128, 431)

    def test_batched_input_shape_already_channelled(self) -> None:
        cfg = ModelConfig(input_shape=(3, 128, 431), in_channels=3)
        assert cfg.batched_input_shape == (3, 128, 431)

    def test_to_dict(self) -> None:
        cfg = ModelConfig(model_name="cnn", num_classes=4)
        d = cfg.to_dict()
        assert d["model_name"] == "cnn"
        assert d["num_classes"] == 4
        assert isinstance(d["input_shape"], list)

    def test_from_dict_round_trip(self) -> None:
        cfg = ModelConfig(model_name="cnn", num_classes=4, extra={"depth": 3})
        cfg2 = ModelConfig.from_dict(cfg.to_dict())
        assert cfg2.model_name == "cnn"
        assert cfg2.num_classes == 4
        assert cfg2.input_shape == (128, 431)
        assert cfg2.extra == {"depth": 3}

    def test_from_dict_drops_unknown_keys(self) -> None:
        d = ModelConfig().to_dict()
        d["future_field"] = "xyz"
        cfg = ModelConfig.from_dict(d)
        assert not hasattr(cfg, "future_field")

    def test_frozen(self) -> None:
        cfg = ModelConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.num_classes = 99  # type: ignore[misc]

    def test_default_class_names(self) -> None:
        assert ModelConfig().class_names == DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the model registry."""

    def test_register_and_retrieve(self) -> None:
        @register_model("reg_test_a")
        class _ModelA(BaseModel):
            def forward(self, x):  # pragma: no cover
                return x

        assert is_registered("reg_test_a")
        assert MODEL_REGISTRY["reg_test_a"] is _ModelA
        assert _ModelA._registry_name == "reg_test_a"

    def test_duplicate_name_different_class_raises(self) -> None:
        @register_model("reg_test_dup")
        class _First(BaseModel):
            def forward(self, x):  # pragma: no cover
                return x

        with pytest.raises(ValueError, match="already registered"):
            @register_model("reg_test_dup")
            class _Second(BaseModel):
                def forward(self, x):  # pragma: no cover
                    return x

    def test_same_class_reregister_idempotent(self) -> None:
        @register_model("reg_test_idem")
        class _Model(BaseModel):
            def forward(self, x):  # pragma: no cover
                return x

        # Re-registering the same class is a no-op, not an error
        register_model("reg_test_idem")(_Model)
        assert MODEL_REGISTRY["reg_test_idem"] is _Model

    def test_list_models_sorted(self) -> None:
        @register_model("reg_test_zzz")
        class _Z(BaseModel):
            def forward(self, x):  # pragma: no cover
                return x

        @register_model("reg_test_aaa")
        class _A(BaseModel):
            def forward(self, x):  # pragma: no cover
                return x

        models = list_models()
        assert models == sorted(models)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            build_model("definitely_not_registered_xyz")

    def test_is_registered_false(self) -> None:
        assert not is_registered("never_registered_abc")


# ---------------------------------------------------------------------------
# Reproducibility / device tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for seeding and device resolution."""

    def test_set_global_seed_numpy(self) -> None:
        set_global_seed(123)
        v1 = np.random.rand()
        set_global_seed(123)
        v2 = np.random.rand()
        assert v1 == v2

    def test_set_global_seed_runs(self) -> None:
        set_global_seed(0)
        set_global_seed(2 ** 31)

    @torch_only
    def test_set_global_seed_torch(self) -> None:
        set_global_seed(42)
        t1 = torch.rand(5)
        set_global_seed(42)
        t2 = torch.rand(5)
        assert torch.equal(t1, t2)

    def test_resolve_device_cpu_explicit(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self) -> None:
        result = resolve_device("auto")
        assert result in ("cuda", "mps", "cpu")

    def test_resolve_device_explicit_passthrough(self) -> None:
        # Explicit non-auto strings pass through unchanged (when torch present)
        if _TORCH:
            assert resolve_device("cuda:1") == "cuda:1"
        else:
            assert resolve_device("cuda:1") == "cpu"


# ---------------------------------------------------------------------------
# Abstract enforcement tests
# ---------------------------------------------------------------------------


class TestAbstractEnforcement:
    """Tests that BaseModel enforces its abstract contract."""

    def test_basemodel_uses_abcmeta(self) -> None:
        from abc import ABCMeta
        assert type(BaseModel) is ABCMeta

    def test_forward_is_abstract(self) -> None:
        assert getattr(BaseModel.forward, "__isabstractmethod__", False)

    @torch_only
    def test_cannot_instantiate_without_forward(self) -> None:
        class _Incomplete(BaseModel):
            pass  # no forward

        with pytest.raises(TypeError):
            _Incomplete(ModelConfig())

    @pytest.mark.skipif(_TORCH, reason="Only when torch absent")
    def test_instantiation_without_torch_raises(self) -> None:
        class _M(BaseModel):
            def forward(self, x):
                return x

        with pytest.raises(RuntimeError, match="PyTorch"):
            _M(ModelConfig())


# ---------------------------------------------------------------------------
# Parameter counting tests
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Tests for :class:`ParameterCount` and counting."""

    def test_namedtuple_structure(self) -> None:
        pc = ParameterCount(total=100, trainable=80, non_trainable=20,
                            buffers=5, size_mb=0.4)
        assert pc.total == 100
        assert pc.trainable == 80
        assert pc.non_trainable == 20
        assert pc.buffers == 5
        assert pc.size_mb == 0.4

    @torch_only
    def test_count_parameters(self) -> None:
        _register_tiny("tiny_count")
        model = build_model("tiny_count", ModelConfig(num_classes=5))
        counts = model.count_parameters()
        assert counts.total > 0
        assert counts.trainable == counts.total  # nothing frozen
        assert counts.non_trainable == 0
        assert counts.size_mb > 0

    @torch_only
    def test_num_parameters_property(self) -> None:
        _register_tiny("tiny_numparams")
        model = build_model("tiny_numparams", ModelConfig())
        assert model.num_parameters == model.count_parameters().total

    @torch_only
    def test_freeze_changes_counts(self) -> None:
        _register_tiny("tiny_freeze")
        model = build_model("tiny_freeze", ModelConfig())
        model.freeze()
        counts = model.count_parameters()
        assert counts.trainable == 0
        assert counts.non_trainable == counts.total

    @torch_only
    def test_unfreeze_restores(self) -> None:
        _register_tiny("tiny_unfreeze")
        model = build_model("tiny_unfreeze", ModelConfig())
        model.freeze()
        model.unfreeze()
        counts = model.count_parameters()
        assert counts.trainable == counts.total


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for the model summary."""

    @torch_only
    def test_summary_is_string(self) -> None:
        _register_tiny("tiny_summary")
        model = build_model("tiny_summary", ModelConfig())
        s = model.summary()
        assert isinstance(s, str)

    @torch_only
    def test_summary_contains_total_params(self) -> None:
        _register_tiny("tiny_summary2")
        model = build_model("tiny_summary2", ModelConfig())
        s = model.summary()
        assert "Total params" in s
        assert "Trainable params" in s

    @torch_only
    def test_summary_contains_model_name(self) -> None:
        _register_tiny("tiny_summary3")
        model = build_model("tiny_summary3", ModelConfig(model_name="tiny_summary3"))
        assert "tiny_summary3" in model.summary()

    @torch_only
    def test_print_summary(self, capsys) -> None:
        _register_tiny("tiny_summary4")
        model = build_model("tiny_summary4", ModelConfig())
        model.print_summary()
        out = capsys.readouterr().out
        assert "Total params" in out


# ---------------------------------------------------------------------------
# Checkpoint save/load tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for save_model / load_model / inspect_checkpoint."""

    @torch_only
    def test_save_creates_file(self, tmp_path: Path) -> None:
        _register_tiny("tiny_save")
        model = build_model("tiny_save", ModelConfig(num_classes=5))
        path = tmp_path / "model.pt"
        result = model.save_model(path)
        assert result.is_file()

    @torch_only
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        _register_tiny("tiny_roundtrip")
        model = build_model("tiny_roundtrip", ModelConfig(num_classes=5))
        path = tmp_path / "model.pt"
        model.save_model(path, metadata={"epoch": 10, "val_f1": 0.92})

        restored = _TinyClassifier.load_model(path)
        assert restored.config.num_classes == 5
        # Weights identical
        for p1, p2 in zip(model.parameters(), restored.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    @torch_only
    def test_load_preserves_metadata(self, tmp_path: Path) -> None:
        _register_tiny("tiny_meta")
        model = build_model("tiny_meta", ModelConfig())
        path = tmp_path / "model.pt"
        model.save_model(path, metadata={"epoch": 7, "note": "best"})
        info = BaseModel.inspect_checkpoint(path)
        assert info["metadata"]["epoch"] == 7
        assert info["metadata"]["note"] == "best"

    @torch_only
    def test_inspect_checkpoint_provenance(self, tmp_path: Path) -> None:
        _register_tiny("tiny_prov")
        model = build_model("tiny_prov", ModelConfig())
        path = tmp_path / "model.pt"
        model.save_model(path)
        info = BaseModel.inspect_checkpoint(path)
        assert info["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert "torch_version" in info["provenance"]
        assert "config_hash" in info["provenance"]

    @torch_only
    def test_load_resolves_registry_class(self, tmp_path: Path) -> None:
        _register_tiny("tiny_registry_load")
        model = build_model("tiny_registry_load", ModelConfig())
        path = tmp_path / "model.pt"
        model.save_model(path)
        # Load via BaseModel — should resolve concrete class from registry
        restored = BaseModel.load_model(path)
        assert isinstance(restored, _TinyClassifier)

    @torch_only
    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BaseModel.load_model(tmp_path / "nonexistent.pt")

    @torch_only
    def test_load_invalid_checkpoint_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pt"
        torch.save({"not_a_checkpoint": True}, str(bad))
        with pytest.raises(ValueError, match="Unrecognised checkpoint"):
            BaseModel.load_model(bad)

    @torch_only
    def test_loaded_model_in_eval_mode(self, tmp_path: Path) -> None:
        _register_tiny("tiny_eval")
        model = build_model("tiny_eval", ModelConfig())
        path = tmp_path / "model.pt"
        model.save_model(path)
        restored = BaseModel.load_model(path)
        assert not restored.training  # eval mode

    def test_save_without_torch_raises(self) -> None:
        if _TORCH:
            pytest.skip("torch present")
        with pytest.raises(RuntimeError):
            BaseModel.inspect_checkpoint("/tmp/x.pt")


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
        _register_tiny("tiny_onnx")
        model = build_model("tiny_onnx", ModelConfig(num_classes=5))
        path = tmp_path / "model.onnx"
        result = model.export_onnx(path)
        assert result.is_file()

    @torch_only
    def test_export_restores_training_mode(self, tmp_path: Path) -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed")
        _register_tiny("tiny_onnx_mode")
        model = build_model("tiny_onnx_mode", ModelConfig())
        model.train()
        model.export_onnx(tmp_path / "m.onnx")
        assert model.training  # restored to train mode

    def test_export_without_torch_raises(self) -> None:
        if _TORCH:
            pytest.skip("torch present")
        # Cannot instantiate without torch, so this path is covered by
        # the instantiation test; just assert the guard constant.
        from src.models.base_model import _TORCH_AVAILABLE
        assert _TORCH_AVAILABLE is False


# ---------------------------------------------------------------------------
# Device management tests
# ---------------------------------------------------------------------------


class TestDeviceManagement:
    """Tests for device placement."""

    @torch_only
    def test_device_property(self) -> None:
        _register_tiny("tiny_device")
        model = build_model("tiny_device", ModelConfig())
        assert isinstance(model.device, torch.device)

    @torch_only
    def test_to_device_cpu(self) -> None:
        _register_tiny("tiny_device_cpu")
        model = build_model("tiny_device_cpu", ModelConfig())
        model.to_device("cpu")
        assert model.device.type == "cpu"

    @torch_only
    def test_to_device_returns_self(self) -> None:
        _register_tiny("tiny_device_self")
        model = build_model("tiny_device_self", ModelConfig())
        assert model.to_device("cpu") is model


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------


class TestInference:
    """Tests for predict / predict_proba."""

    @torch_only
    def test_predict_shape(self) -> None:
        _register_tiny("tiny_predict")
        model = build_model("tiny_predict", ModelConfig(num_classes=5))
        x = torch.randn(4, *model.config.batched_input_shape)
        preds = model.predict(x)
        assert preds.shape == (4,)

    @torch_only
    def test_predict_valid_classes(self) -> None:
        _register_tiny("tiny_predict_cls")
        model = build_model("tiny_predict_cls", ModelConfig(num_classes=5))
        x = torch.randn(8, *model.config.batched_input_shape)
        preds = model.predict(x)
        assert preds.min() >= 0
        assert preds.max() < 5

    @torch_only
    def test_predict_proba_sums_to_one(self) -> None:
        _register_tiny("tiny_proba")
        model = build_model("tiny_proba", ModelConfig(num_classes=5))
        x = torch.randn(4, *model.config.batched_input_shape)
        proba = model.predict_proba(x)
        assert proba.shape == (4, 5)
        sums = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    @torch_only
    def test_predict_proba_non_negative(self) -> None:
        _register_tiny("tiny_proba_nn")
        model = build_model("tiny_proba_nn", ModelConfig(num_classes=5))
        x = torch.randn(4, *model.config.batched_input_shape)
        proba = model.predict_proba(x)
        assert (proba >= 0).all()


# ---------------------------------------------------------------------------
# Forward pass / determinism tests
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for forward pass and deterministic construction."""

    @torch_only
    def test_forward_output_shape(self) -> None:
        _register_tiny("tiny_fwd")
        model = build_model("tiny_fwd", ModelConfig(num_classes=5))
        model.eval()
        x = torch.randn(2, *model.config.batched_input_shape)
        out = model(x)
        assert out.shape == (2, 5)

    @torch_only
    def test_deterministic_construction(self) -> None:
        _register_tiny("tiny_det")
        m1 = build_model("tiny_det", ModelConfig(random_seed=42))
        m2 = build_model("tiny_det", ModelConfig(random_seed=42))
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    @torch_only
    def test_repr_contains_params(self) -> None:
        _register_tiny("tiny_repr")
        model = build_model("tiny_repr", ModelConfig())
        r = repr(model)
        assert "params=" in r


# ---------------------------------------------------------------------------
# ExperimentTracker integration tests
# ---------------------------------------------------------------------------


class TestExperimentTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    @torch_only
    def test_log_to_tracker(self) -> None:
        calls = []

        class FakeTracker:
            def log_model_info(self, model_name, n_parameters, architecture, **kw):
                calls.append({
                    "model_name": model_name,
                    "n_parameters": n_parameters,
                    "architecture": architecture,
                })

        _register_tiny("tiny_tracker")
        model = build_model("tiny_tracker", ModelConfig())
        model.log_to_tracker(FakeTracker())
        assert len(calls) == 1
        assert calls[0]["n_parameters"] == model.num_parameters

    @torch_only
    def test_broken_tracker_does_not_crash(self) -> None:
        class BrokenTracker:
            def log_model_info(self, *a, **kw):
                raise RuntimeError("boom")

        _register_tiny("tiny_tracker_broken")
        model = build_model("tiny_tracker_broken", ModelConfig())
        # Must not raise
        model.log_to_tracker(BrokenTracker())

    @torch_only
    def test_save_with_tracker_logs_artifact(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_artifact(self, path, description, artifact_type):
                artifacts.append((path, artifact_type))

        _register_tiny("tiny_tracker_artifact")
        model = build_model("tiny_tracker_artifact", ModelConfig())
        model.save_model(tmp_path / "m.pt", experiment_tracker=FakeTracker())
        assert len(artifacts) == 1
        assert artifacts[0][1] == "checkpoint"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module constants."""

    def test_checkpoint_format_version(self) -> None:
        assert CHECKPOINT_FORMAT_VERSION == "1.0"

    def test_default_class_names_count(self) -> None:
        assert len(DEFAULT_CLASS_NAMES) == 5

    def test_default_class_names_content(self) -> None:
        assert "normal" in DEFAULT_CLASS_NAMES
        assert "bearing_fault" in DEFAULT_CLASS_NAMES