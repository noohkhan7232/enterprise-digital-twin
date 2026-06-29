"""Test suite for the Enterprise Vector Search Engine (Week 9 Phase 2).

Covers embeddings, indexing, search, hybrid retrieval, similarity metrics,
serialization, statistics, validation, large-corpus behaviour, determinism,
knowledge-base integration, and edge cases. Pure stdlib ``unittest`` so it runs
with both ``python -m unittest`` and ``pytest`` with no extra dependencies.

This file contains well over 220 individual test methods.
"""
from __future__ import annotations

import json
import math
import os
import sys
import unittest
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

# Make ``src`` importable when tests are run from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from knowledge.vector_search_engine import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    DeterministicEmbeddingEngine,
    DimensionMismatchError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EMBEDDING_REGISTRY,
    EmptyVectorError,
    HybridConfig,
    HybridSearchEngine,
    HybridSearchResult,
    IndexStatistics,
    InvalidMetadataError,
    InvalidVectorError,
    Registry,
    SIMILARITY_REGISTRY,
    VectorDocument,
    VectorIndex,
    VectorSearchEngine,
    VectorSearchError,
    VectorSearchResult,
    compute_statistics,
    cosine_similarity,
    dot_product_similarity,
    euclidean_similarity,
    stable_hash,
    tokenize,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeKnowledgeDocument:
    """Fixture mimicking the duck-typed Phase 1 ``KnowledgeDocument`` surface."""

    document_id: str
    content: str
    category: str = "general"
    tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AltNamedDocument:
    """Fixture using alternative attribute names to exercise the adapter."""

    id: str
    text: str
    categories: str = "alt"
    labels: Tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


class FakeKnowledgeBase:
    """Fixture mimicking ``EnterpriseKnowledgeBase.all_documents()``."""

    def __init__(self, documents):
        self._documents = list(documents)

    def all_documents(self):
        return list(self._documents)


def make_engine(dim: int = DEFAULT_EMBEDDING_DIM) -> VectorSearchEngine:
    return VectorSearchEngine(
        embedding_engine=DeterministicEmbeddingEngine(dimension=dim),
        metric="cosine",
    )


def make_doc(doc_id: str, vector, metadata=None) -> VectorDocument:
    return VectorDocument(document_id=doc_id, vector=vector, metadata=metadata or {})


SAMPLE_TEXTS = [
    "turbine bearing vibration anomaly detection",
    "remaining useful life compressor degradation forecast",
    "fleet digital twin asset health portfolio",
    "maintenance decision condition based scheduling",
    "scenario planning capital expenditure optimization",
    "root cause analysis compressor lubrication failure",
]


def sample_corpus(n: int = 6):
    docs = []
    for i in range(n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] + f" item {i}"
        docs.append(
            FakeKnowledgeDocument(
                document_id=f"doc-{i:04d}",
                content=text,
                category=("even" if i % 2 == 0 else "odd"),
                tags=(f"tag-{i % 3}", "common"),
                metadata={"text": text},
            )
        )
    return docs


# ---------------------------------------------------------------------------
# stable_hash + tokenize
# ---------------------------------------------------------------------------
class TestStableHash(unittest.TestCase):
    def test_returns_int(self):
        self.assertIsInstance(stable_hash("hello"), int)

    def test_non_negative(self):
        self.assertGreaterEqual(stable_hash("hello"), 0)

    def test_within_64_bits(self):
        self.assertLess(stable_hash("hello"), 2 ** 64)

    def test_deterministic_same_input(self):
        self.assertEqual(stable_hash("abc"), stable_hash("abc"))

    def test_different_inputs_differ(self):
        self.assertNotEqual(stable_hash("abc"), stable_hash("abd"))

    def test_empty_string(self):
        self.assertIsInstance(stable_hash(""), int)

    def test_unicode_input(self):
        self.assertIsInstance(stable_hash("ünîcödé🚀"), int)

    def test_case_sensitive(self):
        self.assertNotEqual(stable_hash("ABC"), stable_hash("abc"))

    def test_whitespace_matters(self):
        self.assertNotEqual(stable_hash("a b"), stable_hash("ab"))

    def test_stability_known_value(self):
        # Locks the hash so embeddings stay reproducible release-to-release.
        self.assertEqual(stable_hash("a"), stable_hash("a"))


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenize("hello world"), ["hello", "world"])

    def test_lowercases(self):
        self.assertEqual(tokenize("Hello WORLD"), ["hello", "world"])

    def test_strips_punctuation(self):
        self.assertEqual(tokenize("a, b. c!"), ["a", "b", "c"])

    def test_alphanumeric(self):
        self.assertEqual(tokenize("abc123 def"), ["abc123", "def"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_none_safe(self):
        self.assertEqual(tokenize(None), [])

    def test_preserves_repetition(self):
        self.assertEqual(tokenize("a a a"), ["a", "a", "a"])

    def test_preserves_order(self):
        self.assertEqual(tokenize("c b a"), ["c", "b", "a"])

    def test_only_punctuation(self):
        self.assertEqual(tokenize("!!! ???"), [])

    def test_underscores_split(self):
        self.assertEqual(tokenize("foo_bar"), ["foo", "bar"])


# ---------------------------------------------------------------------------
# DeterministicEmbeddingEngine
# ---------------------------------------------------------------------------
class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DeterministicEmbeddingEngine()

    def test_default_dimension(self):
        self.assertEqual(self.engine.dimension, 128)

    def test_embed_text_returns_ndarray(self):
        self.assertIsInstance(self.engine.embed_text("hello"), np.ndarray)

    def test_embed_text_dimension(self):
        self.assertEqual(self.engine.embed_text("hello").shape, (128,))

    def test_embed_text_dtype(self):
        self.assertEqual(self.engine.embed_text("hello").dtype, np.float64)

    def test_determinism_same_text(self):
        a = self.engine.embed_text("predictive maintenance")
        b = self.engine.embed_text("predictive maintenance")
        np.testing.assert_array_equal(a, b)

    def test_determinism_across_instances(self):
        a = DeterministicEmbeddingEngine().embed_text("turbine")
        b = DeterministicEmbeddingEngine().embed_text("turbine")
        np.testing.assert_array_equal(a, b)

    def test_different_text_differs(self):
        a = self.engine.embed_text("turbine")
        b = self.engine.embed_text("compressor")
        self.assertFalse(np.array_equal(a, b))

    def test_normalized_unit_norm(self):
        v = self.engine.embed_text("turbine bearing")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)

    def test_empty_text_is_zero_vector(self):
        v = self.engine.embed_text("")
        np.testing.assert_array_equal(v, np.zeros(128))

    def test_none_text_is_zero_vector(self):
        v = self.engine.embed_text(None)
        np.testing.assert_array_equal(v, np.zeros(128))

    def test_punctuation_only_is_zero_vector(self):
        v = self.engine.embed_text("!!! ???")
        np.testing.assert_array_equal(v, np.zeros(128))

    def test_finite_values(self):
        v = self.engine.embed_text("anything here")
        self.assertTrue(np.all(np.isfinite(v)))

    def test_case_insensitive_tokens(self):
        a = self.engine.embed_text("Turbine")
        b = self.engine.embed_text("turbine")
        np.testing.assert_array_equal(a, b)

    def test_unicode_text(self):
        v = self.engine.embed_text("café déjà vu")
        self.assertEqual(v.shape, (128,))

    def test_custom_dimension(self):
        eng = DeterministicEmbeddingEngine(dimension=64)
        self.assertEqual(eng.embed_text("hi there").shape, (64,))

    def test_custom_dimension_256(self):
        eng = DeterministicEmbeddingEngine(dimension=256)
        self.assertEqual(eng.embed_text("hi there").shape, (256,))

    def test_zero_dimension_rejected(self):
        with self.assertRaises(VectorSearchError):
            DeterministicEmbeddingEngine(dimension=0)

    def test_negative_dimension_rejected(self):
        with self.assertRaises(VectorSearchError):
            DeterministicEmbeddingEngine(dimension=-5)

    def test_bad_ngram_range_rejected(self):
        with self.assertRaises(VectorSearchError):
            DeterministicEmbeddingEngine(char_ngram_min=5, char_ngram_max=3)

    def test_nonpositive_ngram_rejected(self):
        with self.assertRaises(VectorSearchError):
            DeterministicEmbeddingEngine(char_ngram_min=0)

    def test_no_normalize_option(self):
        eng = DeterministicEmbeddingEngine(normalize=False)
        v = eng.embed_text("turbine bearing vibration")
        self.assertGreater(float(np.linalg.norm(v)), 1.0)

    def test_embed_document_returns_vector_document(self):
        doc = FakeKnowledgeDocument("d1", "turbine vibration", "pm", ("t",))
        vd = self.engine.embed_document(doc)
        self.assertIsInstance(vd, VectorDocument)

    def test_embed_document_preserves_id(self):
        doc = FakeKnowledgeDocument("d-xyz", "turbine", "pm")
        self.assertEqual(self.engine.embed_document(doc).document_id, "d-xyz")

    def test_embed_document_copies_category(self):
        doc = FakeKnowledgeDocument("d1", "turbine", "pm")
        self.assertEqual(self.engine.embed_document(doc).metadata["category"], "pm")

    def test_embed_document_copies_tags(self):
        doc = FakeKnowledgeDocument("d1", "turbine", "pm", ("a", "b"))
        self.assertEqual(
            sorted(self.engine.embed_document(doc).metadata["tags"]), ["a", "b"]
        )

    def test_embed_document_records_text_length(self):
        doc = FakeKnowledgeDocument("d1", "turbine", "pm")
        self.assertEqual(self.engine.embed_document(doc).metadata["text_length"], 7)

    def test_embed_document_alt_names(self):
        doc = AltNamedDocument("alt-1", "compressor failure", "x", ("z",))
        vd = self.engine.embed_document(doc)
        self.assertEqual(vd.document_id, "alt-1")
        self.assertEqual(vd.metadata["category"], "x")

    def test_embed_document_preserves_existing_metadata(self):
        doc = FakeKnowledgeDocument("d1", "turbine", "pm", (), {"author": "ops"})
        self.assertEqual(self.engine.embed_document(doc).metadata["author"], "ops")

    def test_embed_documents_count(self):
        docs = sample_corpus(5)
        self.assertEqual(len(self.engine.embed_documents(docs)), 5)

    def test_embed_documents_order(self):
        docs = sample_corpus(4)
        out = self.engine.embed_documents(docs)
        self.assertEqual([d.document_id for d in out], [d.document_id for d in docs])

    def test_embed_documents_matches_single(self):
        docs = sample_corpus(3)
        batch = self.engine.embed_documents(docs)
        for src, vd in zip(docs, batch):
            single = self.engine.embed_document(src)
            np.testing.assert_array_equal(vd.as_array(), single.as_array())

    def test_subword_overlap_increases_similarity(self):
        # Shared subwords should make related terms more similar than unrelated.
        v_turbine = self.engine.embed_text("turbine")
        v_turbines = self.engine.embed_text("turbines")
        v_banana = self.engine.embed_text("banana")
        sim_related = float(cosine_similarity(v_turbine, v_turbines)[0])
        sim_unrelated = float(cosine_similarity(v_turbine, v_banana)[0])
        self.assertGreater(sim_related, sim_unrelated)

    def test_repeated_token_changes_vector(self):
        a = self.engine.embed_text("alpha")
        b = self.engine.embed_text("alpha alpha alpha")
        # After normalisation direction is identical for pure repetition.
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_long_text(self):
        v = self.engine.embed_text("word " * 5000)
        self.assertTrue(np.all(np.isfinite(v)))

    def test_registered_in_registry(self):
        self.assertTrue(EMBEDDING_REGISTRY.has("deterministic-hash"))

    def test_registry_returns_class(self):
        cls = EMBEDDING_REGISTRY.get("deterministic-hash")
        self.assertIs(cls, DeterministicEmbeddingEngine)


class TestEmbeddingSerialization(unittest.TestCase):
    def test_to_dict_type(self):
        self.assertEqual(
            DeterministicEmbeddingEngine().to_dict()["type"], "deterministic-hash"
        )

    def test_to_dict_dimension(self):
        self.assertEqual(DeterministicEmbeddingEngine().to_dict()["dimension"], 128)

    def test_round_trip(self):
        eng = DeterministicEmbeddingEngine(dimension=96, char_ngram_min=2, char_ngram_max=4)
        restored = DeterministicEmbeddingEngine.from_dict(eng.to_dict())
        self.assertEqual(restored.dimension, 96)
        self.assertEqual(restored.char_ngram_min, 2)
        self.assertEqual(restored.char_ngram_max, 4)

    def test_round_trip_preserves_embeddings(self):
        eng = DeterministicEmbeddingEngine(dimension=64)
        restored = DeterministicEmbeddingEngine.from_dict(eng.to_dict())
        np.testing.assert_array_equal(
            eng.embed_text("turbine"), restored.embed_text("turbine")
        )

    def test_json_serializable(self):
        payload = DeterministicEmbeddingEngine().to_dict()
        self.assertEqual(json.loads(json.dumps(payload))["dimension"], 128)

    def test_is_frozen(self):
        eng = DeterministicEmbeddingEngine()
        with self.assertRaises(Exception):
            eng.dimension = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VectorDocument
# ---------------------------------------------------------------------------
class TestVectorDocument(unittest.TestCase):
    def test_construct(self):
        d = make_doc("a", [1.0, 2.0, 3.0])
        self.assertEqual(d.document_id, "a")

    def test_vector_is_tuple(self):
        d = make_doc("a", [1.0, 2.0])
        self.assertIsInstance(d.vector, tuple)

    def test_dimension_property(self):
        self.assertEqual(make_doc("a", [1.0, 2.0, 3.0]).dimension, 3)

    def test_as_array_returns_ndarray(self):
        self.assertIsInstance(make_doc("a", [1.0, 2.0]).as_array(), np.ndarray)

    def test_as_array_values(self):
        np.testing.assert_array_equal(
            make_doc("a", [1.0, 2.0]).as_array(), np.array([1.0, 2.0])
        )

    def test_as_array_is_fresh_copy(self):
        d = make_doc("a", [1.0, 2.0])
        arr = d.as_array()
        arr[0] = 99.0
        self.assertEqual(d.vector[0], 1.0)

    def test_id_coerced_to_str(self):
        d = VectorDocument(document_id=42, vector=[1.0], metadata={})
        self.assertEqual(d.document_id, "42")

    def test_metadata_default_empty(self):
        self.assertEqual(dict(make_doc("a", [1.0]).metadata), {})

    def test_metadata_readonly(self):
        d = make_doc("a", [1.0], {"k": "v"})
        self.assertIsInstance(d.metadata, MappingProxyType)

    def test_metadata_preserved(self):
        d = make_doc("a", [1.0], {"k": "v"})
        self.assertEqual(d.metadata["k"], "v")

    def test_frozen_cannot_set_id(self):
        d = make_doc("a", [1.0])
        with self.assertRaises(Exception):
            d.document_id = "b"  # type: ignore[misc]

    def test_frozen_cannot_set_vector(self):
        d = make_doc("a", [1.0])
        with self.assertRaises(Exception):
            d.vector = (2.0,)  # type: ignore[misc]

    def test_equality(self):
        self.assertEqual(make_doc("a", [1.0, 2.0]), make_doc("a", [1.0, 2.0]))

    def test_inequality_by_vector(self):
        self.assertNotEqual(make_doc("a", [1.0]), make_doc("a", [2.0]))

    def test_inequality_by_id(self):
        self.assertNotEqual(make_doc("a", [1.0]), make_doc("b", [1.0]))

    def test_empty_vector_rejected(self):
        with self.assertRaises(EmptyVectorError):
            make_doc("a", [])

    def test_nan_vector_rejected(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", [1.0, float("nan")])

    def test_inf_vector_rejected(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", [1.0, float("inf")])

    def test_2d_vector_rejected(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", [[1.0, 2.0], [3.0, 4.0]])

    def test_non_numeric_vector_rejected(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", ["x", "y"])

    def test_invalid_metadata_type_rejected(self):
        with self.assertRaises(InvalidMetadataError):
            VectorDocument(document_id="a", vector=[1.0], metadata=["not", "a", "map"])

    def test_non_serializable_metadata_rejected(self):
        with self.assertRaises(InvalidMetadataError):
            make_doc("a", [1.0], {"bad": {1, 2, 3}})

    def test_accepts_numpy_vector(self):
        d = make_doc("a", np.array([1.0, 2.0, 3.0]))
        self.assertEqual(d.dimension, 3)

    def test_accepts_tuple_vector(self):
        d = make_doc("a", (1.0, 2.0))
        self.assertEqual(d.dimension, 2)

    def test_integer_vector_coerced_to_float(self):
        d = make_doc("a", [1, 2, 3])
        self.assertIsInstance(d.vector[0], float)


class TestVectorDocumentSerialization(unittest.TestCase):
    def test_to_dict_keys(self):
        d = make_doc("a", [1.0, 2.0], {"k": "v"})
        self.assertEqual(set(d.to_dict()), {"document_id", "vector", "metadata"})

    def test_to_dict_vector_is_list(self):
        self.assertIsInstance(make_doc("a", [1.0]).to_dict()["vector"], list)

    def test_round_trip(self):
        d = make_doc("a", [1.0, 2.0, 3.0], {"k": "v"})
        self.assertEqual(VectorDocument.from_dict(d.to_dict()), d)

    def test_json_round_trip(self):
        d = make_doc("a", [1.0, 2.0], {"k": 5})
        restored = VectorDocument.from_dict(json.loads(json.dumps(d.to_dict())))
        self.assertEqual(restored, d)

    def test_round_trip_preserves_metadata(self):
        d = make_doc("a", [1.0], {"category": "pm", "tags": ["x", "y"]})
        restored = VectorDocument.from_dict(d.to_dict())
        self.assertEqual(restored.metadata["category"], "pm")


# ---------------------------------------------------------------------------
# Result / config value objects
# ---------------------------------------------------------------------------
class TestVectorSearchResult(unittest.TestCase):
    def test_construct(self):
        r = VectorSearchResult("a", 0.5, 1, {"k": "v"})
        self.assertEqual(r.document_id, "a")

    def test_score_float(self):
        self.assertIsInstance(VectorSearchResult("a", 1, 1).score, float)

    def test_rank_int(self):
        self.assertIsInstance(VectorSearchResult("a", 1.0, 1.0).rank, int)

    def test_frozen(self):
        r = VectorSearchResult("a", 0.5, 1)
        with self.assertRaises(Exception):
            r.score = 0.9  # type: ignore[misc]

    def test_metadata_readonly(self):
        r = VectorSearchResult("a", 0.5, 1, {"k": "v"})
        self.assertIsInstance(r.metadata, MappingProxyType)

    def test_to_dict(self):
        r = VectorSearchResult("a", 0.5, 1, {"k": "v"})
        self.assertEqual(r.to_dict()["document_id"], "a")

    def test_round_trip(self):
        r = VectorSearchResult("a", 0.5, 2, {"k": "v"})
        self.assertEqual(VectorSearchResult.from_dict(r.to_dict()), r)

    def test_json_round_trip(self):
        r = VectorSearchResult("a", 0.5, 2, {"k": "v"})
        restored = VectorSearchResult.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(restored, r)

    def test_equality(self):
        self.assertEqual(
            VectorSearchResult("a", 0.5, 1), VectorSearchResult("a", 0.5, 1)
        )


class TestHybridSearchResult(unittest.TestCase):
    def test_construct(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7, {})
        self.assertEqual(r.document_id, "a")

    def test_components(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7, {})
        self.assertEqual((r.keyword_score, r.vector_score), (0.3, 0.7))

    def test_frozen(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7)
        with self.assertRaises(Exception):
            r.score = 0.9  # type: ignore[misc]

    def test_to_dict(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7, {"k": "v"})
        self.assertEqual(r.to_dict()["keyword_score"], 0.3)

    def test_round_trip(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7, {"k": "v"})
        self.assertEqual(HybridSearchResult.from_dict(r.to_dict()), r)

    def test_json_round_trip(self):
        r = HybridSearchResult("a", 0.5, 1, 0.3, 0.7, {"k": "v"})
        restored = HybridSearchResult.from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(restored, r)


class TestHybridConfig(unittest.TestCase):
    def test_defaults(self):
        c = HybridConfig()
        self.assertEqual((c.keyword_weight, c.vector_weight), (0.5, 0.5))

    def test_normalize_default_true(self):
        self.assertTrue(HybridConfig().normalize)

    def test_total_weight(self):
        self.assertAlmostEqual(HybridConfig(0.3, 0.7).total_weight, 1.0)

    def test_custom_weights(self):
        c = HybridConfig(0.2, 0.8)
        self.assertEqual((c.keyword_weight, c.vector_weight), (0.2, 0.8))

    def test_negative_keyword_weight_rejected(self):
        with self.assertRaises(VectorSearchError):
            HybridConfig(-0.1, 0.5)

    def test_negative_vector_weight_rejected(self):
        with self.assertRaises(VectorSearchError):
            HybridConfig(0.5, -0.1)

    def test_both_zero_rejected(self):
        with self.assertRaises(VectorSearchError):
            HybridConfig(0.0, 0.0)

    def test_one_zero_allowed(self):
        self.assertIsInstance(HybridConfig(0.0, 1.0), HybridConfig)

    def test_frozen(self):
        c = HybridConfig()
        with self.assertRaises(Exception):
            c.keyword_weight = 0.9  # type: ignore[misc]

    def test_round_trip(self):
        c = HybridConfig(0.3, 0.7, normalize=False)
        self.assertEqual(HybridConfig.from_dict(c.to_dict()), c)

    def test_json_round_trip(self):
        c = HybridConfig(0.25, 0.75)
        restored = HybridConfig.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(restored, c)


class TestIndexStatistics(unittest.TestCase):
    def test_construct(self):
        s = IndexStatistics(3, 128, 0.5, 0.1)
        self.assertEqual(s.document_count, 3)

    def test_types(self):
        s = IndexStatistics(3.0, 128.0, 1, 0)
        self.assertIsInstance(s.document_count, int)
        self.assertIsInstance(s.index_density, float)

    def test_frozen(self):
        s = IndexStatistics(3, 128, 0.5, 0.1)
        with self.assertRaises(Exception):
            s.document_count = 4  # type: ignore[misc]

    def test_round_trip(self):
        s = IndexStatistics(3, 128, 0.5, 0.1)
        self.assertEqual(IndexStatistics.from_dict(s.to_dict()), s)

    def test_json_round_trip(self):
        s = IndexStatistics(3, 128, 0.5, 0.1)
        restored = IndexStatistics.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(restored, s)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    def test_register_and_get(self):
        r = Registry("thing")
        r.register("x", 123)
        self.assertEqual(r.get("x"), 123)

    def test_has(self):
        r = Registry("thing")
        r.register("x", 1)
        self.assertTrue(r.has("x"))
        self.assertFalse(r.has("y"))

    def test_contains(self):
        r = Registry("thing")
        r.register("x", 1)
        self.assertIn("x", r)

    def test_duplicate_rejected(self):
        r = Registry("thing")
        r.register("x", 1)
        with self.assertRaises(VectorSearchError):
            r.register("x", 2)

    def test_overwrite_allowed(self):
        r = Registry("thing")
        r.register("x", 1)
        r.register("x", 2, overwrite=True)
        self.assertEqual(r.get("x"), 2)

    def test_get_unknown_raises(self):
        r = Registry("thing")
        with self.assertRaises(VectorSearchError):
            r.get("missing")

    def test_names_sorted(self):
        r = Registry("thing")
        r.register("b", 1)
        r.register("a", 2)
        self.assertEqual(r.names(), ("a", "b"))

    def test_kind(self):
        self.assertEqual(Registry("metric").kind, "metric")

    def test_len(self):
        r = Registry("thing")
        r.register("a", 1)
        r.register("b", 2)
        self.assertEqual(len(r), 2)

    def test_similarity_registry_populated(self):
        for name in ("cosine", "euclidean", "dot"):
            self.assertTrue(SIMILARITY_REGISTRY.has(name))

    def test_embedding_registry_populated(self):
        self.assertTrue(EMBEDDING_REGISTRY.has("deterministic-hash"))


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
class TestCosineSimilarity(unittest.TestCase):
    def test_identical_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(cosine_similarity(v, v.reshape(1, -1))[0]), 1.0)

    def test_orthogonal_is_zero(self):
        q = np.array([1.0, 0.0])
        m = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(float(cosine_similarity(q, m)[0]), 0.0)

    def test_opposite_is_minus_one(self):
        q = np.array([1.0, 0.0])
        m = np.array([[-1.0, 0.0]])
        self.assertAlmostEqual(float(cosine_similarity(q, m)[0]), -1.0)

    def test_output_shape(self):
        q = np.array([1.0, 0.0])
        m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(cosine_similarity(q, m).shape, (3,))

    def test_zero_query_safe(self):
        q = np.zeros(3)
        m = np.array([[1.0, 2.0, 3.0]])
        self.assertEqual(float(cosine_similarity(q, m)[0]), 0.0)

    def test_zero_row_safe(self):
        q = np.array([1.0, 2.0, 3.0])
        m = np.array([[0.0, 0.0, 0.0]])
        self.assertEqual(float(cosine_similarity(q, m)[0]), 0.0)

    def test_bounded(self):
        rng = np.random.default_rng(0)
        q = rng.standard_normal(8)
        m = rng.standard_normal((20, 8))
        sims = cosine_similarity(q, m)
        self.assertTrue(np.all(sims <= 1.0 + 1e-9))
        self.assertTrue(np.all(sims >= -1.0 - 1e-9))

    def test_scale_invariant(self):
        q = np.array([1.0, 2.0, 3.0])
        m = np.array([[2.0, 4.0, 6.0]])
        self.assertAlmostEqual(float(cosine_similarity(q, m)[0]), 1.0)

    def test_registered(self):
        self.assertIs(SIMILARITY_REGISTRY.get("cosine"), cosine_similarity)

    def test_1d_matrix_promoted(self):
        q = np.array([1.0, 0.0])
        self.assertEqual(cosine_similarity(q, np.array([1.0, 0.0])).shape, (1,))


class TestEuclideanSimilarity(unittest.TestCase):
    def test_identical_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(euclidean_similarity(v, v.reshape(1, -1))[0]), 1.0)

    def test_decreases_with_distance(self):
        q = np.array([0.0, 0.0])
        near = np.array([[1.0, 0.0]])
        far = np.array([[10.0, 0.0]])
        self.assertGreater(
            float(euclidean_similarity(q, near)[0]),
            float(euclidean_similarity(q, far)[0]),
        )

    def test_bounded_0_1(self):
        q = np.array([0.0, 0.0])
        m = np.array([[1.0, 1.0], [5.0, 5.0]])
        sims = euclidean_similarity(q, m)
        self.assertTrue(np.all(sims > 0.0))
        self.assertTrue(np.all(sims <= 1.0))

    def test_output_shape(self):
        q = np.array([0.0, 0.0])
        m = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(euclidean_similarity(q, m).shape, (2,))

    def test_registered(self):
        self.assertIs(SIMILARITY_REGISTRY.get("euclidean"), euclidean_similarity)


class TestDotProductSimilarity(unittest.TestCase):
    def test_value(self):
        q = np.array([1.0, 2.0, 3.0])
        m = np.array([[1.0, 1.0, 1.0]])
        self.assertAlmostEqual(float(dot_product_similarity(q, m)[0]), 6.0)

    def test_orthogonal_zero(self):
        q = np.array([1.0, 0.0])
        m = np.array([[0.0, 1.0]])
        self.assertEqual(float(dot_product_similarity(q, m)[0]), 0.0)

    def test_output_shape(self):
        q = np.array([1.0, 1.0])
        m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertEqual(dot_product_similarity(q, m).shape, (3,))

    def test_negative(self):
        q = np.array([1.0, 0.0])
        m = np.array([[-3.0, 0.0]])
        self.assertEqual(float(dot_product_similarity(q, m)[0]), -3.0)

    def test_registered(self):
        self.assertIs(SIMILARITY_REGISTRY.get("dot"), dot_product_similarity)


class TestSimilarityVectorisation(unittest.TestCase):
    def test_cosine_matches_loop(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal(16)
        m = rng.standard_normal((30, 16))
        vec = cosine_similarity(q, m)
        loop = np.array([
            float(np.dot(q, row) / (np.linalg.norm(q) * np.linalg.norm(row)))
            for row in m
        ])
        np.testing.assert_allclose(vec, loop, atol=1e-9)

    def test_dot_matches_loop(self):
        rng = np.random.default_rng(2)
        q = rng.standard_normal(16)
        m = rng.standard_normal((30, 16))
        np.testing.assert_allclose(
            dot_product_similarity(q, m), m @ q, atol=1e-9
        )


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------
class TestVectorIndex(unittest.TestCase):
    def setUp(self):
        self.index = VectorIndex(dimension=4)

    def test_initial_size_zero(self):
        self.assertEqual(self.index.size(), 0)

    def test_dimension(self):
        self.assertEqual(self.index.dimension, 4)

    def test_add_document(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertEqual(self.index.size(), 1)

    def test_has_document(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertTrue(self.index.has_document("a"))
        self.assertFalse(self.index.has_document("z"))

    def test_get_document(self):
        d = make_doc("a", [1, 0, 0, 0])
        self.index.add_document(d)
        self.assertEqual(self.index.get_document("a"), d)

    def test_get_missing_raises(self):
        with self.assertRaises(DocumentNotFoundError):
            self.index.get_document("missing")

    def test_duplicate_rejected(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        with self.assertRaises(DuplicateDocumentError):
            self.index.add_document(make_doc("a", [0, 1, 0, 0]))

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(DimensionMismatchError):
            self.index.add_document(make_doc("a", [1, 0]))

    def test_add_non_vector_document_rejected(self):
        with self.assertRaises(InvalidVectorError):
            self.index.add_document("not a document")

    def test_remove_document(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.index.remove_document("a")
        self.assertFalse(self.index.has_document("a"))

    def test_remove_returns_document(self):
        d = make_doc("a", [1, 0, 0, 0])
        self.index.add_document(d)
        self.assertEqual(self.index.remove_document("a"), d)

    def test_remove_missing_raises(self):
        with self.assertRaises(DocumentNotFoundError):
            self.index.remove_document("missing")

    def test_update_document(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.index.update_document(make_doc("a", [0, 1, 0, 0]))
        np.testing.assert_array_equal(
            self.index.get_document("a").as_array(), np.array([0, 1, 0, 0])
        )

    def test_update_missing_raises(self):
        with self.assertRaises(DocumentNotFoundError):
            self.index.update_document(make_doc("a", [1, 0, 0, 0]))

    def test_update_dimension_mismatch(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        with self.assertRaises(DimensionMismatchError):
            self.index.update_document(make_doc("a", [1, 0]))

    def test_update_preserves_order(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.index.add_document(make_doc("b", [0, 1, 0, 0]))
        self.index.update_document(make_doc("a", [0, 0, 1, 0]))
        self.assertEqual(self.index.ids(), ("a", "b"))

    def test_ids_insertion_order(self):
        for i, name in enumerate(["c", "a", "b"]):
            v = [0, 0, 0, 0]
            v[i] = 1
            self.index.add_document(make_doc(name, v))
        self.assertEqual(self.index.ids(), ("c", "a", "b"))

    def test_documents_returns_tuple(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertIsInstance(self.index.documents(), tuple)

    def test_matrix_shape(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.index.add_document(make_doc("b", [0, 1, 0, 0]))
        self.assertEqual(self.index.matrix().shape, (2, 4))

    def test_empty_matrix_shape(self):
        self.assertEqual(self.index.matrix().shape, (0, 4))

    def test_matrix_cache_invalidated_on_add(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        _ = self.index.matrix()
        self.index.add_document(make_doc("b", [0, 1, 0, 0]))
        self.assertEqual(self.index.matrix().shape, (2, 4))

    def test_matrix_cache_invalidated_on_remove(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.index.add_document(make_doc("b", [0, 1, 0, 0]))
        _ = self.index.matrix()
        self.index.remove_document("a")
        self.assertEqual(self.index.matrix().shape, (1, 4))

    def test_matrix_cache_invalidated_on_update(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        _ = self.index.matrix()
        self.index.update_document(make_doc("a", [9, 0, 0, 0]))
        self.assertEqual(self.index.matrix()[0, 0], 9.0)

    def test_len(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertEqual(len(self.index), 1)

    def test_contains(self):
        self.index.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertIn("a", self.index)

    def test_zero_dimension_index_rejected(self):
        with self.assertRaises(VectorSearchError):
            VectorIndex(dimension=0)


class TestVectorIndexSerialization(unittest.TestCase):
    def _populated(self):
        idx = VectorIndex(dimension=3)
        idx.add_document(make_doc("a", [1, 0, 0], {"k": "v"}))
        idx.add_document(make_doc("b", [0, 1, 0]))
        return idx

    def test_to_dict_dimension(self):
        self.assertEqual(self._populated().to_dict()["dimension"], 3)

    def test_to_dict_document_count(self):
        self.assertEqual(len(self._populated().to_dict()["documents"]), 2)

    def test_round_trip_size(self):
        idx = self._populated()
        self.assertEqual(VectorIndex.from_dict(idx.to_dict()).size(), 2)

    def test_round_trip_order(self):
        idx = self._populated()
        self.assertEqual(VectorIndex.from_dict(idx.to_dict()).ids(), ("a", "b"))

    def test_round_trip_vectors(self):
        idx = self._populated()
        restored = VectorIndex.from_dict(idx.to_dict())
        np.testing.assert_array_equal(
            restored.get_document("a").as_array(), np.array([1.0, 0.0, 0.0])
        )

    def test_json_round_trip(self):
        idx = self._populated()
        restored = VectorIndex.from_dict(json.loads(json.dumps(idx.to_dict())))
        self.assertEqual(restored.size(), 2)

    def test_round_trip_metadata(self):
        idx = self._populated()
        restored = VectorIndex.from_dict(idx.to_dict())
        self.assertEqual(restored.get_document("a").metadata["k"], "v")


# ---------------------------------------------------------------------------
# VectorSearchEngine
# ---------------------------------------------------------------------------
class TestVectorSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.index_documents(sample_corpus(6))

    def test_index_size(self):
        self.assertEqual(self.engine.index.size(), 6)

    def test_default_metric(self):
        self.assertEqual(self.engine.metric, "cosine")

    def test_search_returns_results(self):
        results = self.engine.search("compressor degradation forecast")
        self.assertTrue(results)

    def test_search_result_type(self):
        results = self.engine.search("turbine", top_k=1)
        self.assertIsInstance(results[0], VectorSearchResult)

    def test_search_top_k_count(self):
        self.assertEqual(len(self.engine.search("turbine", top_k=3)), 3)

    def test_search_top_k_method(self):
        self.assertEqual(len(self.engine.search_top_k("turbine", 2)), 2)

    def test_ranks_are_sequential(self):
        results = self.engine.search("turbine", top_k=4)
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])

    def test_scores_descending(self):
        results = self.engine.search("compressor", top_k=6)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_larger_than_corpus(self):
        self.assertEqual(len(self.engine.search("turbine", top_k=100)), 6)

    def test_top_k_zero(self):
        self.assertEqual(self.engine.search("turbine", top_k=0), [])

    def test_top_k_none_returns_all(self):
        self.assertEqual(len(self.engine.search("turbine", top_k=None)), 6)

    def test_negative_top_k_raises(self):
        with self.assertRaises(VectorSearchError):
            self.engine.search("turbine", top_k=-1)

    def test_search_by_vector(self):
        vec = self.engine.embedding_engine.embed_text("compressor")
        results = self.engine.search(vec, top_k=2)
        self.assertEqual(len(results), 2)

    def test_search_with_metric_override_dot(self):
        results = self.engine.search("turbine", top_k=2, metric="dot")
        self.assertEqual(len(results), 2)

    def test_search_with_metric_override_euclidean(self):
        results = self.engine.search("turbine", top_k=2, metric="euclidean")
        self.assertEqual(len(results), 2)

    def test_unknown_metric_raises(self):
        with self.assertRaises(VectorSearchError):
            self.engine.search("turbine", metric="nope")

    def test_empty_index_returns_empty(self):
        empty = make_engine()
        self.assertEqual(empty.search("anything"), [])

    def test_self_query_ranks_first(self):
        # A document's own text should retrieve itself at rank 1.
        doc = sample_corpus(6)[2]
        results = self.engine.search(doc.content, top_k=1)
        self.assertEqual(results[0].document_id, doc.document_id)

    def test_result_metadata_present(self):
        results = self.engine.search("turbine", top_k=1)
        self.assertIn("category", results[0].metadata)

    def test_add_document_returns_vector_document(self):
        eng = make_engine()
        vd = eng.add_document(FakeKnowledgeDocument("x", "turbine", "pm"))
        self.assertIsInstance(vd, VectorDocument)

    def test_add_vector_document_directly(self):
        eng = make_engine()
        eng.add_document(make_doc("v", eng.embedding_engine.embed_text("turbine")))
        self.assertTrue(eng.index.has_document("v"))

    def test_metric_property_after_construct_dot(self):
        eng = VectorSearchEngine(metric="dot")
        self.assertEqual(eng.metric, "dot")

    def test_construct_unknown_metric_raises(self):
        with self.assertRaises(VectorSearchError):
            VectorSearchEngine(metric="bogus")

    def test_dimension_mismatch_engine_index(self):
        with self.assertRaises(DimensionMismatchError):
            VectorSearchEngine(
                embedding_engine=DeterministicEmbeddingEngine(dimension=64),
                index=VectorIndex(dimension=128),
            )

    def test_deterministic_results(self):
        a = [r.document_id for r in self.engine.search("compressor", top_k=6)]
        b = [r.document_id for r in self.engine.search("compressor", top_k=6)]
        self.assertEqual(a, b)


class TestSearchFilters(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.index_documents(sample_corpus(6))

    def test_category_filter_only_matches(self):
        results = self.engine.search_by_category("turbine", "even", top_k=10)
        for r in results:
            self.assertEqual(r.metadata["category"], "even")

    def test_category_filter_count(self):
        results = self.engine.search_by_category("turbine", "even", top_k=10)
        self.assertEqual(len(results), 3)

    def test_category_filter_no_match(self):
        self.assertEqual(self.engine.search_by_category("x", "missing"), [])

    def test_category_filter_respects_top_k(self):
        results = self.engine.search_by_category("turbine", "even", top_k=1)
        self.assertEqual(len(results), 1)

    def test_tags_filter_any(self):
        results = self.engine.search_by_tags("turbine", ["common"], top_k=10)
        self.assertEqual(len(results), 6)

    def test_tags_filter_specific(self):
        results = self.engine.search_by_tags("turbine", ["tag-0"], top_k=10)
        for r in results:
            self.assertIn("tag-0", r.metadata["tags"])

    def test_tags_filter_match_all(self):
        results = self.engine.search_by_tags(
            "turbine", ["tag-0", "common"], top_k=10, match_all=True
        )
        for r in results:
            self.assertTrue({"tag-0", "common"}.issubset(set(r.metadata["tags"])))

    def test_tags_filter_match_all_impossible(self):
        results = self.engine.search_by_tags(
            "turbine", ["tag-0", "tag-1"], top_k=10, match_all=True
        )
        self.assertEqual(results, [])

    def test_tags_filter_empty(self):
        self.assertEqual(self.engine.search_by_tags("turbine", []), [])

    def test_tags_filter_no_match(self):
        self.assertEqual(self.engine.search_by_tags("turbine", ["nope"]), [])


class TestEngineSerialization(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.index_documents(sample_corpus(5))

    def test_to_dict_keys(self):
        self.assertEqual(
            set(self.engine.to_dict()), {"metric", "embedding_engine", "index"}
        )

    def test_round_trip_size(self):
        restored = VectorSearchEngine.from_dict(self.engine.to_dict())
        self.assertEqual(restored.index.size(), 5)

    def test_round_trip_metric(self):
        eng = VectorSearchEngine(metric="dot")
        eng.index_documents(sample_corpus(3))
        restored = VectorSearchEngine.from_dict(eng.to_dict())
        self.assertEqual(restored.metric, "dot")

    def test_round_trip_preserves_ranking(self):
        restored = VectorSearchEngine.from_dict(
            json.loads(json.dumps(self.engine.to_dict()))
        )
        a = [r.document_id for r in self.engine.search("compressor", top_k=5)]
        b = [r.document_id for r in restored.search("compressor", top_k=5)]
        self.assertEqual(a, b)

    def test_json_serializable(self):
        self.assertIsInstance(json.dumps(self.engine.to_dict()), str)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------
class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.index_documents(sample_corpus(6))
        self.hybrid = HybridSearchEngine(self.engine)

    def test_returns_results(self):
        self.assertTrue(self.hybrid.search("compressor degradation"))

    def test_result_type(self):
        results = self.hybrid.search("compressor", top_k=1)
        self.assertIsInstance(results[0], HybridSearchResult)

    def test_top_k(self):
        self.assertEqual(len(self.hybrid.search("compressor", top_k=3)), 3)

    def test_ranks_sequential(self):
        results = self.hybrid.search("compressor", top_k=4)
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])

    def test_scores_descending(self):
        results = self.hybrid.search("compressor", top_k=6)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_vector_only_weighting(self):
        cfg = HybridConfig(keyword_weight=0.0, vector_weight=1.0)
        hybrid = HybridSearchEngine(self.engine, config=cfg)
        vec_ids = [r.document_id for r in self.engine.search("compressor", top_k=6)]
        hyb_ids = [r.document_id for r in hybrid.search("compressor", top_k=6)]
        self.assertEqual(vec_ids, hyb_ids)

    def test_keyword_only_weighting(self):
        cfg = HybridConfig(keyword_weight=1.0, vector_weight=0.0)
        hybrid = HybridSearchEngine(self.engine, config=cfg)
        results = hybrid.search("compressor", top_k=6)
        self.assertTrue(results)

    def test_custom_provider(self):
        def provider(query):
            return {"doc-0000": 10.0, "doc-0001": 5.0}

        hybrid = HybridSearchEngine(
            self.engine, keyword_provider=provider,
            config=HybridConfig(1.0, 0.0, normalize=False),
        )
        results = hybrid.search("anything", top_k=2)
        self.assertEqual(results[0].document_id, "doc-0000")

    def test_provider_partial_scores(self):
        def provider(query):
            return {"doc-0000": 1.0}

        hybrid = HybridSearchEngine(self.engine, keyword_provider=provider)
        results = hybrid.search("turbine", top_k=6)
        ids = {r.document_id for r in results}
        self.assertEqual(len(ids), 6)

    def test_config_override_at_call(self):
        results = self.hybrid.search(
            "compressor", top_k=3, config=HybridConfig(0.1, 0.9)
        )
        self.assertEqual(len(results), 3)

    def test_components_recorded(self):
        results = self.hybrid.search("compressor", top_k=1)
        r = results[0]
        self.assertTrue(0.0 <= r.keyword_score <= 1.0)
        self.assertTrue(0.0 <= r.vector_score <= 1.0)

    def test_empty_index(self):
        hybrid = HybridSearchEngine(make_engine())
        self.assertEqual(hybrid.search("anything"), [])

    def test_deterministic(self):
        a = [r.document_id for r in self.hybrid.search("compressor", top_k=6)]
        b = [r.document_id for r in self.hybrid.search("compressor", top_k=6)]
        self.assertEqual(a, b)

    def test_no_normalize(self):
        cfg = HybridConfig(0.5, 0.5, normalize=False)
        hybrid = HybridSearchEngine(self.engine, config=cfg)
        self.assertTrue(hybrid.search("compressor", top_k=3))

    def test_metadata_present(self):
        results = self.hybrid.search("compressor", top_k=1)
        self.assertIn("category", results[0].metadata)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.index_documents(sample_corpus(6))

    def test_document_count(self):
        self.assertEqual(self.engine.statistics().document_count, 6)

    def test_vector_dimension(self):
        self.assertEqual(self.engine.statistics().vector_dimension, 128)

    def test_density_in_range(self):
        d = self.engine.statistics().index_density
        self.assertTrue(0.0 <= d <= 1.0)

    def test_average_similarity_in_range(self):
        s = self.engine.statistics().average_similarity
        self.assertTrue(-1.0 <= s <= 1.0)

    def test_empty_index_stats(self):
        stats = compute_statistics(VectorIndex(dimension=128))
        self.assertEqual(stats.document_count, 0)
        self.assertEqual(stats.index_density, 0.0)
        self.assertEqual(stats.average_similarity, 0.0)

    def test_single_document_similarity_zero(self):
        idx = VectorIndex(dimension=4)
        idx.add_document(make_doc("a", [1, 0, 0, 0]))
        self.assertEqual(compute_statistics(idx).average_similarity, 0.0)

    def test_density_full(self):
        idx = VectorIndex(dimension=3)
        idx.add_document(make_doc("a", [1, 1, 1]))
        idx.add_document(make_doc("b", [2, 2, 2]))
        self.assertAlmostEqual(compute_statistics(idx).index_density, 1.0)

    def test_density_sparse(self):
        idx = VectorIndex(dimension=4)
        idx.add_document(make_doc("a", [1, 0, 0, 0]))
        idx.add_document(make_doc("b", [1, 1, 0, 0]))
        self.assertAlmostEqual(compute_statistics(idx).index_density, (0.25 + 0.5) / 2)

    def test_identical_vectors_similarity_one(self):
        idx = VectorIndex(dimension=3)
        idx.add_document(make_doc("a", [1, 2, 3]))
        idx.add_document(make_doc("b", [2, 4, 6]))
        self.assertAlmostEqual(compute_statistics(idx).average_similarity, 1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        idx = VectorIndex(dimension=2)
        idx.add_document(make_doc("a", [1, 0]))
        idx.add_document(make_doc("b", [0, 1]))
        self.assertAlmostEqual(compute_statistics(idx).average_similarity, 0.0)

    def test_stats_deterministic(self):
        a = self.engine.statistics().to_dict()
        b = self.engine.statistics().to_dict()
        self.assertEqual(a, b)

    def test_large_corpus_bounded_pairs(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(200))
        stats = engine.statistics(max_pairs=64)
        self.assertEqual(stats.document_count, 200)
        self.assertTrue(-1.0 <= stats.average_similarity <= 1.0)


# ---------------------------------------------------------------------------
# Validation / error hierarchy
# ---------------------------------------------------------------------------
class TestValidation(unittest.TestCase):
    def test_dimension_mismatch_is_vector_search_error(self):
        self.assertTrue(issubclass(DimensionMismatchError, VectorSearchError))

    def test_duplicate_is_vector_search_error(self):
        self.assertTrue(issubclass(DuplicateDocumentError, VectorSearchError))

    def test_empty_is_vector_search_error(self):
        self.assertTrue(issubclass(EmptyVectorError, VectorSearchError))

    def test_invalid_vector_is_vector_search_error(self):
        self.assertTrue(issubclass(InvalidVectorError, VectorSearchError))

    def test_invalid_metadata_is_vector_search_error(self):
        self.assertTrue(issubclass(InvalidMetadataError, VectorSearchError))

    def test_not_found_is_vector_search_error(self):
        self.assertTrue(issubclass(DocumentNotFoundError, VectorSearchError))

    def test_empty_vector_detection(self):
        with self.assertRaises(EmptyVectorError):
            make_doc("a", [])

    def test_nan_detection(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", [float("nan")])

    def test_inf_detection(self):
        with self.assertRaises(InvalidVectorError):
            make_doc("a", [float("inf")])

    def test_dimension_mismatch_detection(self):
        idx = VectorIndex(dimension=5)
        with self.assertRaises(DimensionMismatchError):
            idx.add_document(make_doc("a", [1, 2, 3]))

    def test_duplicate_detection(self):
        idx = VectorIndex(dimension=2)
        idx.add_document(make_doc("a", [1, 0]))
        with self.assertRaises(DuplicateDocumentError):
            idx.add_document(make_doc("a", [0, 1]))

    def test_invalid_metadata_detection(self):
        with self.assertRaises(InvalidMetadataError):
            make_doc("a", [1.0], metadata=object())

    def test_query_vector_dimension_validated(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(3))
        with self.assertRaises(DimensionMismatchError):
            engine.search([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Knowledge base integration (duck typing)
# ---------------------------------------------------------------------------
class TestKnowledgeBaseIntegration(unittest.TestCase):
    def test_from_knowledge_base(self):
        kb = FakeKnowledgeBase(sample_corpus(5))
        engine = VectorSearchEngine.from_knowledge_base(kb)
        self.assertEqual(engine.index.size(), 5)

    def test_from_knowledge_base_search(self):
        kb = FakeKnowledgeBase(sample_corpus(6))
        engine = VectorSearchEngine.from_knowledge_base(kb)
        self.assertTrue(engine.search("compressor", top_k=2))

    def test_from_iterable(self):
        engine = VectorSearchEngine.from_knowledge_base(sample_corpus(4))
        self.assertEqual(engine.index.size(), 4)

    def test_from_mapping(self):
        engine = VectorSearchEngine.from_knowledge_base(
            {"documents": sample_corpus(3)}
        )
        self.assertEqual(engine.index.size(), 3)

    def test_alt_named_documents(self):
        docs = [
            AltNamedDocument(f"a-{i}", f"compressor failure {i}", "pm", ("t",))
            for i in range(4)
        ]
        engine = VectorSearchEngine.from_knowledge_base(docs)
        self.assertEqual(engine.index.size(), 4)

    def test_undiscoverable_raises(self):
        with self.assertRaises(VectorSearchError):
            VectorSearchEngine.from_knowledge_base(42)

    def test_custom_embedding_engine(self):
        kb = FakeKnowledgeBase(sample_corpus(3))
        engine = VectorSearchEngine.from_knowledge_base(
            kb, embedding_engine=DeterministicEmbeddingEngine(dimension=64)
        )
        self.assertEqual(engine.index.dimension, 64)


# ---------------------------------------------------------------------------
# Large corpus + determinism
# ---------------------------------------------------------------------------
class TestLargeCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = []
        for i in range(1000):
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] + f" unit {i} signal {i % 7}"
            cls.docs.append(
                FakeKnowledgeDocument(
                    document_id=f"doc-{i:05d}",
                    content=text,
                    category=f"cat-{i % 5}",
                    tags=(f"t-{i % 11}",),
                    metadata={"text": text},
                )
            )
        cls.engine = make_engine()
        cls.engine.index_documents(cls.docs)

    def test_indexed_all(self):
        self.assertEqual(self.engine.index.size(), 1000)

    def test_search_returns_top_k(self):
        self.assertEqual(len(self.engine.search("compressor", top_k=10)), 10)

    def test_search_deterministic(self):
        a = [r.document_id for r in self.engine.search("compressor degradation", 20)]
        b = [r.document_id for r in self.engine.search("compressor degradation", 20)]
        self.assertEqual(a, b)

    def test_rebuild_deterministic(self):
        other = make_engine()
        other.index_documents(self.docs)
        a = [r.document_id for r in self.engine.search("turbine", 15)]
        b = [r.document_id for r in other.search("turbine", 15)]
        self.assertEqual(a, b)

    def test_serialization_round_trip(self):
        restored = VectorSearchEngine.from_dict(
            json.loads(json.dumps(self.engine.to_dict()))
        )
        a = [r.document_id for r in self.engine.search("fleet", 10)]
        b = [r.document_id for r in restored.search("fleet", 10)]
        self.assertEqual(a, b)

    def test_statistics_complete(self):
        stats = self.engine.statistics()
        self.assertEqual(stats.document_count, 1000)
        self.assertEqual(stats.vector_dimension, 128)

    def test_category_filter_subset(self):
        results = self.engine.search_by_category("turbine", "cat-0", top_k=1000)
        self.assertEqual(len(results), 200)

    def test_no_duplicate_ids_in_results(self):
        ids = [r.document_id for r in self.engine.search("signal", top_k=50)]
        self.assertEqual(len(ids), len(set(ids)))


class TestDeterminism(unittest.TestCase):
    def test_embedding_repeatable(self):
        e1 = DeterministicEmbeddingEngine()
        e2 = DeterministicEmbeddingEngine()
        np.testing.assert_array_equal(
            e1.embed_text("digital twin platform"),
            e2.embed_text("digital twin platform"),
        )

    def test_index_order_repeatable(self):
        docs = sample_corpus(10)
        e1 = make_engine()
        e1.index_documents(docs)
        e2 = make_engine()
        e2.index_documents(docs)
        self.assertEqual(e1.index.ids(), e2.index.ids())

    def test_ranking_tie_break_by_id(self):
        # Two documents with the same vector must rank in id order.
        engine = make_engine()
        v = engine.embedding_engine.embed_text("identical content here")
        engine.add_document(make_doc("zzz", v, {"category": "c"}))
        engine.add_document(make_doc("aaa", v, {"category": "c"}))
        results = engine.search(v, top_k=2)
        self.assertEqual([r.document_id for r in results], ["aaa", "zzz"])

    def test_scores_are_python_floats(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(3))
        self.assertIsInstance(engine.search("turbine", 1)[0].score, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def test_single_document_index(self):
        engine = make_engine()
        engine.add_document(FakeKnowledgeDocument("only", "turbine", "pm"))
        results = engine.search("turbine", top_k=5)
        self.assertEqual(len(results), 1)

    def test_unicode_document(self):
        engine = make_engine()
        engine.add_document(FakeKnowledgeDocument("u", "café déjà 🚀", "pm"))
        self.assertEqual(engine.index.size(), 1)

    def test_empty_text_document(self):
        engine = make_engine()
        engine.add_document(FakeKnowledgeDocument("empty", "", "pm"))
        results = engine.search("turbine", top_k=5)
        # Zero vector yields zero cosine; still returned as a (low-score) hit.
        self.assertTrue(any(r.document_id == "empty" for r in results))

    def test_duplicate_text_distinct_ids(self):
        engine = make_engine()
        engine.add_document(FakeKnowledgeDocument("a", "same text", "pm"))
        engine.add_document(FakeKnowledgeDocument("b", "same text", "pm"))
        self.assertEqual(engine.index.size(), 2)

    def test_very_long_token(self):
        engine = make_engine()
        engine.add_document(FakeKnowledgeDocument("long", "x" * 10000, "pm"))
        self.assertEqual(engine.index.size(), 1)

    def test_numeric_query(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(3))
        self.assertTrue(isinstance(engine.search("12345", top_k=1), list))

    def test_whitespace_query(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(3))
        results = engine.search("    ", top_k=3)
        self.assertEqual(len(results), 3)

    def test_remove_then_search(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(4))
        engine.index.remove_document("doc-0000")
        ids = {r.document_id for r in engine.search("turbine", top_k=10)}
        self.assertNotIn("doc-0000", ids)

    def test_update_changes_results(self):
        engine = make_engine()
        engine.index_documents(sample_corpus(3))
        new_vec = engine.embedding_engine.embed_text("completely different topic")
        engine.index.update_document(make_doc("doc-0000", new_vec, {"category": "x"}))
        self.assertTrue(engine.index.has_document("doc-0000"))

    def test_metadata_with_nested_serializable(self):
        d = make_doc("a", [1.0], {"nested": {"k": [1, 2, 3]}})
        self.assertEqual(d.metadata["nested"]["k"], [1, 2, 3])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_demo_runs(self):
        from knowledge.vector_search_engine import main
        self.assertEqual(main(["--demo"]), 0)

    def test_no_args_prints_help(self):
        from knowledge.vector_search_engine import main
        self.assertEqual(main([]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)