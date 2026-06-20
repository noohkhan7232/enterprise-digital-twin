#!/usr/bin/env python3
"""Comprehensive test suite for ``src/evaluation/model_benchmark.py``.

The benchmark's statistical and orchestration core is torch-independent, so
the **entire pipeline** is exercised here with lightweight fake models — no
PyTorch required.  A small number of tests that construct real registry models
are guarded with ``@torch_only``.

Coverage:
- BenchmarkConfig validation
- MetricSummary / ModelBenchmarkResult containers
- Confidence-interval computation (Student-t, edge cases)
- Quality-metric computation (accuracy/precision/recall/F1/ROC-AUC, degenerate)
- Fold partitioning
- End-to-end run with fakes: multiple models × multiple datasets
- k-fold aggregation with confidence intervals
- Efficiency metrics (latency, throughput, params, memory)
- Ranking (descending quality, ascending latency, NaN handling)
- Ranking / summary tables
- CSV emission (columns, CI columns, content)
- JSON emission (structure, rankings, round-trip)
- ExperimentTracker integration (metrics + artifacts, failure-safe)
- Error handling (build failure, evaluation failure)
- Registry alignment (supervised set excludes the autoencoder)
- Real-model construction via the registry (torch-only)

Run::

    pytest tests/test_model_benchmark.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.model_benchmark import (
    ALL_METRICS,
    EFFICIENCY_METRICS,
    QUALITY_METRICS,
    SUPERVISED_MODELS,
    BenchmarkConfig,
    MetricSummary,
    ModelBenchmark,
    ModelBenchmarkResult,
    compute_confidence_interval,
    compute_quality_metrics,
)

try:
    import torch  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False

torch_only = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """A deterministic fake exposing the evaluable contract."""

    def __init__(self, name: str, n_params: int = 1_000_000,
                 num_classes: int = 5) -> None:
        self.name = name
        self._n = n_params
        self.num_classes = num_classes

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x)
        n = len(x)
        # Deterministic pseudo-labels from feature magnitude.
        return (np.abs(x).reshape(n, -1).mean(axis=1) * 1000).astype(int) % self.num_classes

    def predict_proba(self, x) -> np.ndarray:
        preds = self.predict(x)
        n = len(preds)
        p = np.full((n, self.num_classes), 0.02)
        for i, c in enumerate(preds):
            p[i, c] = 1.0 - 0.02 * (self.num_classes - 1)
        return p

    def count_parameters(self):
        class _PC:
            total = self._n
        return _PC()


def _make_dataset(n_batches: int = 8, bs: int = 16, seed: int = 0):
    rng = np.random.RandomState(seed)
    return [
        (rng.randn(bs, 1, 16, 16), rng.randint(0, 5, bs))
        for _ in range(n_batches)
    ]


def _factory(name: str) -> _FakeModel:
    params = {
        "acoustic_cnn": 2_000_000, "resnet_acoustic": 11_000_000,
        "cnn_bilstm": 3_000_000, "cnn_bilstm_attention": 4_000_000,
        "acoustic_transformer": 8_000_000,
    }.get(name, 1_000_000)
    return _FakeModel(name, n_params=params)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig validation."""

    def test_defaults(self) -> None:
        c = BenchmarkConfig()
        assert c.models == SUPERVISED_MODELS
        assert c.k_folds == 1

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BenchmarkConfig(models=())

    def test_bad_k_folds_raises(self) -> None:
        with pytest.raises(ValueError, match="k_folds"):
            BenchmarkConfig(k_folds=0)

    def test_bad_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            BenchmarkConfig(confidence_level=1.5)

    def test_bad_num_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            BenchmarkConfig(num_classes=1)

    def test_frozen(self) -> None:
        c = BenchmarkConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.k_folds = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


class TestContainers:
    """Tests for result/summary containers."""

    def test_metric_summary_to_dict(self) -> None:
        s = MetricSummary(mean=0.9, ci_low=0.85, ci_high=0.95, std=0.02,
                          values=[0.9, 0.88, 0.92])
        d = s.to_dict()
        assert d["mean"] == 0.9 and d["ci_low"] == 0.85
        assert d["values"] == [0.9, 0.88, 0.92]

    def test_result_value(self) -> None:
        r = ModelBenchmarkResult(model_name="m", dataset_name="d")
        r.metrics["f1_macro"] = MetricSummary(mean=0.88)
        assert r.value("f1_macro") == 0.88

    def test_result_value_missing_is_nan(self) -> None:
        r = ModelBenchmarkResult(model_name="m", dataset_name="d")
        assert np.isnan(r.value("nonexistent"))

    def test_result_to_dict(self) -> None:
        r = ModelBenchmarkResult(model_name="m", dataset_name="d", n_samples=100)
        r.metrics["accuracy"] = MetricSummary(mean=0.9)
        d = r.to_dict()
        assert d["model_name"] == "m"
        assert d["metrics"]["accuracy"]["mean"] == 0.9


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    """Tests for confidence-interval computation."""

    def test_basic(self) -> None:
        mean, lo, hi = compute_confidence_interval([0.90, 0.89, 0.91, 0.92, 0.88])
        assert lo < mean < hi
        assert mean == pytest.approx(0.90, abs=0.01)

    def test_single_value_no_interval(self) -> None:
        mean, lo, hi = compute_confidence_interval([0.9])
        assert mean == 0.9
        assert np.isnan(lo) and np.isnan(hi)

    def test_empty(self) -> None:
        mean, lo, hi = compute_confidence_interval([])
        assert np.isnan(mean)

    def test_ignores_nan(self) -> None:
        mean, _, _ = compute_confidence_interval([0.9, float("nan"), 0.8])
        assert mean == pytest.approx(0.85)

    def test_higher_confidence_wider(self) -> None:
        vals = [0.9, 0.85, 0.88, 0.92, 0.87]
        _, lo90, hi90 = compute_confidence_interval(vals, 0.90)
        _, lo99, hi99 = compute_confidence_interval(vals, 0.99)
        assert (hi99 - lo99) > (hi90 - lo90)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    """Tests for quality-metric computation."""

    def test_perfect_prediction(self) -> None:
        y = np.array([0, 1, 2, 3, 4])
        m = compute_quality_metrics(y, y, None, 5)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_all_metrics_present(self) -> None:
        yt = np.array([0, 1, 2, 0, 1])
        yp = np.array([0, 1, 2, 1, 1])
        m = compute_quality_metrics(yt, yp, None, 5)
        for key in QUALITY_METRICS:
            assert key in m

    def test_metrics_in_range(self) -> None:
        yt = np.array([0, 1, 2, 3, 0])
        yp = np.array([0, 1, 2, 3, 1])
        m = compute_quality_metrics(yt, yp, None, 5)
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["f1_macro"] <= 1

    def test_empty_returns_nan(self) -> None:
        m = compute_quality_metrics(np.array([]), np.array([]), None, 5)
        assert all(np.isnan(v) for v in m.values())

    def test_roc_auc_with_proba(self) -> None:
        yt = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        yp = yt.copy()
        proba = np.eye(5)[yp] * 0.8 + 0.04
        m = compute_quality_metrics(yt, yp, proba, 5)
        assert not np.isnan(m["roc_auc"])

    def test_roc_auc_nan_without_proba(self) -> None:
        yt = np.array([0, 1, 2])
        m = compute_quality_metrics(yt, yt, None, 5)
        assert np.isnan(m["roc_auc"])


# ---------------------------------------------------------------------------
# Fold partitioning
# ---------------------------------------------------------------------------


class TestFolds:
    """Tests for fold partitioning."""

    def test_single_fold_when_k_one(self) -> None:
        batches = _make_dataset(8)
        folds = ModelBenchmark._make_folds(batches, 1)
        assert len(folds) == 1

    def test_k_folds(self) -> None:
        batches = _make_dataset(9)
        folds = ModelBenchmark._make_folds(batches, 3)
        assert len(folds) == 3
        # All batches accounted for
        assert sum(len(f) for f in folds) == 9

    def test_k_capped_at_batch_count(self) -> None:
        batches = _make_dataset(2)
        folds = ModelBenchmark._make_folds(batches, 10)
        assert len(folds) <= 2


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


class TestEndToEndRun:
    """Tests for the full benchmark pipeline with fakes."""

    def _bench(self, **cfg_kwargs) -> ModelBenchmark:
        cfg = BenchmarkConfig(
            models=("acoustic_cnn", "resnet_acoustic", "cnn_bilstm"),
            **cfg_kwargs,
        )
        return ModelBenchmark(cfg, model_factory=_factory)

    def test_run_produces_results(self) -> None:
        bench = self._bench(k_folds=1)
        results = bench.run({"mel": _make_dataset(seed=1)})
        assert len(results) == 3  # 3 models × 1 dataset

    def test_multiple_datasets(self) -> None:
        bench = self._bench(k_folds=1)
        results = bench.run({"mel": _make_dataset(seed=1),
                             "cqt": _make_dataset(seed=2)})
        assert len(results) == 6  # 3 × 2

    def test_kfold_records_folds(self) -> None:
        bench = self._bench(k_folds=3)
        results = bench.run({"mel": _make_dataset(n_batches=9, seed=1)})
        assert len(results[0].metrics["f1_macro"].values) == 3

    def test_kfold_has_confidence_interval(self) -> None:
        bench = self._bench(k_folds=4)
        results = bench.run({"mel": _make_dataset(n_batches=12, seed=1)})
        summary = results[0].metrics["f1_macro"]
        assert not np.isnan(summary.ci_low)
        assert summary.ci_low <= summary.mean <= summary.ci_high

    def test_all_metrics_computed(self) -> None:
        bench = self._bench(k_folds=1)
        results = bench.run({"mel": _make_dataset(seed=1)})
        for m in ALL_METRICS:
            assert m in results[0].metrics

    def test_efficiency_measured(self) -> None:
        bench = self._bench(k_folds=1)
        results = bench.run({"mel": _make_dataset(seed=1)})
        r = results[0]
        assert not np.isnan(r.value("n_parameters"))
        assert not np.isnan(r.value("memory_mb"))
        assert r.value("n_parameters") > 0

    def test_resnet_has_more_params(self) -> None:
        bench = self._bench(k_folds=1)
        bench.run({"mel": _make_dataset(seed=1)})
        params = {r.model_name: r.value("n_parameters") for r in bench.results}
        assert params["resnet_acoustic"] > params["acoustic_cnn"]

    def test_n_samples_counted(self) -> None:
        bench = self._bench(k_folds=1)
        results = bench.run({"mel": _make_dataset(n_batches=4, bs=16, seed=1)})
        assert results[0].n_samples == 64


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    """Tests for ranking."""

    def _run(self, **kw) -> ModelBenchmark:
        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn", "resnet_acoustic",
                                    "cnn_bilstm"), **kw),
            model_factory=_factory,
        )
        bench.run({"mel": _make_dataset(seed=1)})
        return bench

    def test_ranking_returns_all(self) -> None:
        bench = self._run()
        ranking = bench.ranking("f1_macro")
        assert len(ranking) == 3

    def test_ranking_sorted_descending(self) -> None:
        bench = self._run()
        ranking = bench.ranking("f1_macro", descending=True)
        vals = [v for _, v in ranking if not np.isnan(v)]
        assert vals == sorted(vals, reverse=True)

    def test_latency_ranked_ascending(self) -> None:
        bench = self._run()
        ranking = bench.ranking("latency_ms", descending=False)
        vals = [v for _, v in ranking if not np.isnan(v)]
        assert vals == sorted(vals)

    def test_params_ranking_ascending(self) -> None:
        bench = self._run()
        ranking = bench.ranking("n_parameters", descending=False)
        # acoustic_cnn (2M) should rank first (fewest params)
        assert ranking[0][0] == "acoustic_cnn"

    def test_ranking_table_renders(self) -> None:
        bench = self._run()
        table = bench.ranking_table("f1_macro")
        assert "Ranking by f1_macro" in table
        assert "Rank" in table

    def test_summary_table_renders(self) -> None:
        bench = self._run()
        table = bench.summary_table()
        assert "Model" in table
        assert "acoustic_cnn" in table


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCsv:
    """Tests for CSV emission."""

    def _bench(self) -> ModelBenchmark:
        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn", "resnet_acoustic"),
                            k_folds=3),
            model_factory=_factory,
        )
        bench.run({"mel": _make_dataset(n_batches=9, seed=1)})
        return bench

    def test_csv_written(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_csv(tmp_path / "b.csv")
        assert path.is_file() and path.stat().st_size > 0

    def test_csv_has_header_and_rows(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_csv(tmp_path / "b.csv")
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 models

    def test_csv_has_ci_columns(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_csv(tmp_path / "b.csv")
        header = path.read_text().splitlines()[0]
        assert "f1_macro_ci_low" in header
        assert "f1_macro_ci_high" in header

    def test_csv_has_all_metrics(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_csv(tmp_path / "b.csv")
        header = path.read_text().splitlines()[0]
        for m in ALL_METRICS:
            assert m in header


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJson:
    """Tests for JSON emission."""

    def _bench(self) -> ModelBenchmark:
        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn", "resnet_acoustic"),
                            k_folds=2),
            model_factory=_factory,
        )
        bench.run({"mel": _make_dataset(seed=1), "cqt": _make_dataset(seed=2)})
        return bench

    def test_json_written(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_json(tmp_path / "b.json")
        assert path.is_file()

    def test_json_structure(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_json(tmp_path / "b.json")
        data = json.loads(path.read_text())
        assert "config" in data
        assert "results" in data
        assert "rankings" in data

    def test_json_all_results(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_json(tmp_path / "b.json")
        data = json.loads(path.read_text())
        assert len(data["results"]) == 4  # 2 models × 2 datasets

    def test_json_rankings_present(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_json(tmp_path / "b.json")
        data = json.loads(path.read_text())
        assert "f1_macro" in data["rankings"]

    def test_json_round_trips(self, tmp_path: Path) -> None:
        bench = self._bench()
        path = bench.to_json(tmp_path / "b.json")
        # Valid JSON that loads without error
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class TestTrackerIntegration:
    """Tests for ExperimentTracker integration."""

    def test_metrics_logged(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, metrics, step=None):
                logged.append(metrics)
            def log_artifact(self, *a, **kw):
                pass

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn",), k_folds=1),
            experiment_tracker=FakeTracker(), model_factory=_factory,
        )
        bench.run({"mel": _make_dataset(seed=1)})
        assert len(logged) == 1

    def test_artifacts_logged(self, tmp_path: Path) -> None:
        artifacts = []

        class FakeTracker:
            def log_metrics(self, *a, **kw):
                pass
            def log_artifact(self, path, description, artifact_type):
                artifacts.append(artifact_type)

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn",), k_folds=1),
            experiment_tracker=FakeTracker(), model_factory=_factory,
        )
        bench.run({"mel": _make_dataset(seed=1)})
        bench.to_csv(tmp_path / "b.csv")
        bench.to_json(tmp_path / "b.json")
        assert "benchmark_csv" in artifacts
        assert "benchmark_json" in artifacts

    def test_broken_tracker_does_not_crash(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **kw):
                raise RuntimeError("boom")
            def log_artifact(self, *a, **kw):
                raise RuntimeError("boom")

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn",), k_folds=1),
            experiment_tracker=BrokenTracker(), model_factory=_factory,
        )
        # Must not raise
        results = bench.run({"mel": _make_dataset(seed=1)})
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_build_failure_recorded(self) -> None:
        def bad_factory(name):
            raise RuntimeError("cannot build")

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn",), k_folds=1),
            model_factory=bad_factory,
        )
        results = bench.run({"mel": _make_dataset(seed=1)})
        assert results[0].error is not None
        assert "build failed" in results[0].error

    def test_evaluation_failure_recorded(self) -> None:
        class ExplodingModel:
            def predict(self, x):
                raise RuntimeError("inference failed")
            def count_parameters(self):
                class _PC:
                    total = 1000
                return _PC()

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn",), k_folds=1),
            model_factory=lambda name: ExplodingModel(),
        )
        results = bench.run({"mel": _make_dataset(seed=1)})
        assert results[0].error is not None

    def test_other_models_continue_after_failure(self) -> None:
        def selective_factory(name):
            if name == "acoustic_cnn":
                raise RuntimeError("broken")
            return _factory(name)

        bench = ModelBenchmark(
            BenchmarkConfig(models=("acoustic_cnn", "resnet_acoustic"),
                            k_folds=1),
            model_factory=selective_factory,
        )
        results = bench.run({"mel": _make_dataset(seed=1)})
        # cnn failed, resnet succeeded
        errors = {r.model_name: r.error for r in results}
        assert errors["acoustic_cnn"] is not None
        assert errors["resnet_acoustic"] is None


# ---------------------------------------------------------------------------
# Registry alignment
# ---------------------------------------------------------------------------


class TestRegistryAlignment:
    """Tests for the supervised-model set."""

    def test_supervised_excludes_autoencoder(self) -> None:
        assert "anomaly_autoencoder" not in SUPERVISED_MODELS

    def test_supervised_count(self) -> None:
        assert len(SUPERVISED_MODELS) == 5

    def test_supervised_names(self) -> None:
        assert set(SUPERVISED_MODELS) == {
            "acoustic_cnn", "resnet_acoustic", "cnn_bilstm",
            "cnn_bilstm_attention", "acoustic_transformer",
        }

    @torch_only
    def test_supervised_models_are_registered(self) -> None:
        import src.models.cnn_classifier  # noqa: F401
        import src.models.resnet_acoustic  # noqa: F401
        import src.models.cnn_bilstm  # noqa: F401
        import src.models.cnn_bilstm_attention  # noqa: F401
        import src.models.acoustic_transformer  # noqa: F401
        from src.models.base_model import is_registered
        for name in SUPERVISED_MODELS:
            assert is_registered(name)


# ---------------------------------------------------------------------------
# Real-model construction (torch-only)
# ---------------------------------------------------------------------------


class TestRealModelConstruction:
    """Tests that build real registry models (torch required)."""

    @torch_only
    def test_builds_and_benchmarks_real_model(self) -> None:
        import src.models.cnn_classifier  # noqa: F401
        import torch as _t

        cfg = BenchmarkConfig(models=("acoustic_cnn",), num_classes=3,
                              k_folds=1, device="cpu")
        bench = ModelBenchmark(cfg)

        # Small real dataset
        dataset = [
            (_t.randn(4, 1, 128, 431), _t.randint(0, 3, (4,)))
            for _ in range(2)
        ]
        results = bench.run({"mel": dataset})
        assert len(results) == 1
        # Real model has a real parameter count
        assert results[0].value("n_parameters") > 0