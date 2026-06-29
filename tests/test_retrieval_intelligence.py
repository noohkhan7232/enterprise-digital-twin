"""Test suite for the Enterprise Retrieval Intelligence Layer (Week 9 Phase 3).

Covers query normalization, the retrieval pipeline, candidate generation,
ranking, context assembly, evidence generation, statistics, serialization,
large-corpus behaviour, determinism, edge cases, and integration with the
Week 9 Phase 1 (knowledge base, duck-typed) and Phase 2 (vector / hybrid)
modules. Pure stdlib ``unittest``; runs under both ``python -m unittest`` and
``pytest`` with no extra dependencies.

This file contains well over 320 individual test methods.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from knowledge.vector_search_engine import (  # noqa: E402
    DeterministicEmbeddingEngine,
    HybridConfig,
    HybridSearchEngine,
    VectorSearchEngine,
)
from knowledge.retrieval_intelligence import (  # noqa: E402
    DEFAULT_STOPWORDS,
    DuplicateRetrievalError,
    EmptyQueryError,
    InvalidQueryError,
    InvalidRankingError,
    MissingDocumentError,
    QueryNormalizer,
    RANKING_REGISTRY,
    RankingWeights,
    RetrievalConfig,
    RetrievalContext,
    RetrievalEngine,
    RetrievalError,
    RetrievalEvidence,
    RetrievalPackage,
    RetrievalStatistics,
    RetrievedDocument,
    register_ranking_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeKnowledgeDocument:
    document_id: str
    content: str
    category: str = "general"
    tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AltNamedDocument:
    id: str
    text: str
    categories: str = "alt"
    labels: Tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


class FakeKnowledgeBase:
    def __init__(self, documents):
        self._documents = list(documents)

    def all_documents(self):
        return list(self._documents)


_TEXTS = [
    "turbine bearing vibration anomaly detection on the high speed shaft",
    "remaining useful life compressor degradation forecast monte carlo",
    "fleet digital twin asset health portfolio executive planning",
    "maintenance decision condition based scheduling downtime cost",
    "scenario planning capital expenditure strategic portfolio optimization",
    "root cause compressor failure lubrication corrective maintenance",
]


def make_corpus(n=6, with_freshness=True):
    docs = []
    for i in range(n):
        text = _TEXTS[i % len(_TEXTS)] + f" record {i}"
        meta = {"text": text}
        if with_freshness:
            meta["freshness"] = round((i % 5) / 4.0, 3)
        docs.append(
            FakeKnowledgeDocument(
                document_id=f"doc-{i:04d}",
                content=text,
                category=("pm" if i % 2 == 0 else "decision"),
                tags=(f"tag-{i % 3}", "common"),
                metadata=meta,
            )
        )
    return docs


def make_engine(n=6, clock=lambda: 0.0, config=None):
    return RetrievalEngine.from_knowledge_base(
        make_corpus(n), config=config, clock=clock
    )


QUERY = "compressor failure remaining useful life"


# ---------------------------------------------------------------------------
# QueryNormalizer
# ---------------------------------------------------------------------------
class TestQueryNormalizer(unittest.TestCase):
    def setUp(self):
        self.norm = QueryNormalizer()

    def test_lowercase(self):
        self.assertEqual(self.norm.normalize("TURBINE BEARING"), "turbine bearing")

    def test_strips_punctuation(self):
        self.assertEqual(self.norm.normalize("turbine, bearing!"), "turbine bearing")

    def test_removes_stopwords(self):
        self.assertNotIn("the", self.norm.normalize_tokens("the turbine"))

    def test_keeps_content_words(self):
        self.assertIn("turbine", self.norm.normalize_tokens("the turbine"))

    def test_removes_duplicates(self):
        self.assertEqual(
            self.norm.normalize_tokens("turbine turbine bearing"),
            ["turbine", "bearing"],
        )

    def test_min_token_length(self):
        self.assertNotIn("a", self.norm.normalize_tokens("a turbine"))

    def test_order_preserved(self):
        self.assertEqual(
            self.norm.normalize_tokens("compressor failure life"),
            ["compressor", "failure", "life"],
        )

    def test_empty_string(self):
        self.assertEqual(self.norm.normalize_tokens(""), [])

    def test_only_stopwords(self):
        self.assertEqual(self.norm.normalize_tokens("the and of"), [])

    def test_deterministic(self):
        a = self.norm.normalize("the compressor and the turbine")
        b = self.norm.normalize("the compressor and the turbine")
        self.assertEqual(a, b)

    def test_no_stopword_removal_option(self):
        norm = QueryNormalizer(remove_stopwords=False)
        self.assertIn("the", norm.normalize_tokens("the turbine"))

    def test_no_dedupe_option(self):
        norm = QueryNormalizer(remove_duplicates=False)
        self.assertEqual(
            norm.normalize_tokens("turbine turbine"), ["turbine", "turbine"]
        )

    def test_custom_min_length(self):
        norm = QueryNormalizer(min_token_length=5, remove_stopwords=False)
        self.assertEqual(norm.normalize_tokens("the turbine"), ["turbine"])

    def test_min_length_zero_rejected(self):
        with self.assertRaises(RetrievalError):
            QueryNormalizer(min_token_length=0)

    def test_custom_stopwords(self):
        norm = QueryNormalizer(stopwords=("turbine",))
        self.assertNotIn("turbine", norm.normalize_tokens("turbine bearing"))

    def test_frozen(self):
        with self.assertRaises(Exception):
            self.norm.lowercase = False  # type: ignore[misc]

    def test_round_trip(self):
        norm = QueryNormalizer(min_token_length=3, remove_stopwords=False)
        restored = QueryNormalizer.from_dict(norm.to_dict())
        self.assertEqual(restored, norm)

    def test_json_round_trip(self):
        norm = QueryNormalizer()
        restored = QueryNormalizer.from_dict(json.loads(json.dumps(norm.to_dict())))
        self.assertEqual(restored.min_token_length, norm.min_token_length)

    def test_unicode(self):
        # The shared tokenizer is ASCII alphanumeric, so accented bytes are split off.
        tokens = self.norm.normalize_tokens("café turbine")
        self.assertIn("turbine", tokens)
        self.assertEqual(len(tokens), 2)

    def test_numeric_tokens_kept(self):
        self.assertIn("12345", self.norm.normalize_tokens("sensor 12345"))

    def test_default_stopwords_nonempty(self):
        self.assertTrue(len(DEFAULT_STOPWORDS) > 0)


# ---------------------------------------------------------------------------
# RankingWeights
# ---------------------------------------------------------------------------
class TestRankingWeights(unittest.TestCase):
    def test_defaults_sum_to_one(self):
        self.assertAlmostEqual(RankingWeights().total, 1.0)

    def test_custom(self):
        w = RankingWeights(1, 1, 1, 1, 1)
        self.assertAlmostEqual(w.total, 5.0)

    def test_as_normalized_sums_to_one(self):
        w = RankingWeights(2, 2, 2, 2, 2).as_normalized()
        self.assertAlmostEqual(w.total, 1.0)

    def test_as_normalized_preserves_ratio(self):
        w = RankingWeights(2, 1, 1, 1, 1).as_normalized()
        self.assertAlmostEqual(w.vector, 2.0 / 6.0)

    def test_negative_rejected(self):
        with self.assertRaises(InvalidRankingError):
            RankingWeights(-1, 1, 1, 1, 1)

    def test_all_zero_rejected(self):
        with self.assertRaises(InvalidRankingError):
            RankingWeights(0, 0, 0, 0, 0)

    def test_single_nonzero_allowed(self):
        self.assertIsInstance(RankingWeights(1, 0, 0, 0, 0), RankingWeights)

    def test_frozen(self):
        w = RankingWeights()
        with self.assertRaises(Exception):
            w.vector = 0.9  # type: ignore[misc]

    def test_round_trip(self):
        w = RankingWeights(0.3, 0.3, 0.2, 0.1, 0.1)
        self.assertEqual(RankingWeights.from_dict(w.to_dict()), w)

    def test_json_round_trip(self):
        w = RankingWeights()
        restored = RankingWeights.from_dict(json.loads(json.dumps(w.to_dict())))
        self.assertEqual(restored, w)

    def test_components_present(self):
        d = RankingWeights().to_dict()
        self.assertEqual(
            set(d), {"vector", "keyword", "metadata", "category", "freshness"}
        )


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------
class TestRetrievalConfig(unittest.TestCase):
    def test_defaults(self):
        c = RetrievalConfig()
        self.assertEqual(c.top_k, 5)

    def test_candidate_k_none_allowed(self):
        self.assertIsNone(RetrievalConfig(candidate_k=None).candidate_k)

    def test_negative_top_k_rejected(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(top_k=-1)

    def test_negative_candidate_k_rejected(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(candidate_k=-1)

    def test_negative_top_evidence_rejected(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(top_evidence=-1)

    def test_min_score_out_of_range_rejected(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(min_score=1.5)

    def test_min_score_negative_rejected(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(min_score=-0.1)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(InvalidRankingError):
            RetrievalConfig(ranking_strategy="nope")

    def test_tags_coerced_to_tuple(self):
        self.assertEqual(RetrievalConfig(tags=["a", "b"]).tags, ("a", "b"))

    def test_category_coerced(self):
        self.assertEqual(RetrievalConfig(category="pm").category, "pm")

    def test_frozen(self):
        c = RetrievalConfig()
        with self.assertRaises(Exception):
            c.top_k = 9  # type: ignore[misc]

    def test_round_trip(self):
        c = RetrievalConfig(
            top_k=7, candidate_k=30, category="pm", tags=("x",), min_score=0.1
        )
        restored = RetrievalConfig.from_dict(c.to_dict())
        self.assertEqual(restored.top_k, 7)
        self.assertEqual(restored.category, "pm")
        self.assertEqual(restored.tags, ("x",))

    def test_round_trip_candidate_k_none(self):
        c = RetrievalConfig(candidate_k=None)
        self.assertIsNone(RetrievalConfig.from_dict(c.to_dict()).candidate_k)

    def test_json_round_trip(self):
        c = RetrievalConfig()
        restored = RetrievalConfig.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(restored.top_k, c.top_k)

    def test_embeds_ranking_weights(self):
        c = RetrievalConfig(ranking_weights=RankingWeights(1, 0, 0, 0, 0))
        self.assertEqual(c.ranking_weights.vector, 1.0)

    def test_embeds_hybrid_config(self):
        c = RetrievalConfig(hybrid_config=HybridConfig(0.3, 0.7))
        self.assertAlmostEqual(c.hybrid_config.keyword_weight, 0.3)


# ---------------------------------------------------------------------------
# Ranking registry
# ---------------------------------------------------------------------------
class TestRankingRegistry(unittest.TestCase):
    def test_weighted_sum_registered(self):
        self.assertTrue(RANKING_REGISTRY.has("weighted_sum"))

    def test_max_registered(self):
        self.assertTrue(RANKING_REGISTRY.has("max"))

    def test_weighted_sum_value(self):
        fn = RANKING_REGISTRY.get("weighted_sum")
        comps = {"vector": 1.0, "keyword": 0.0, "metadata": 0.0, "category": 0.0, "freshness": 0.0}
        score = fn(comps, RankingWeights(1, 0, 0, 0, 0))
        self.assertAlmostEqual(score, 1.0)

    def test_weighted_sum_blend(self):
        fn = RANKING_REGISTRY.get("weighted_sum")
        comps = {"vector": 1.0, "keyword": 1.0, "metadata": 0.0, "category": 0.0, "freshness": 0.0}
        score = fn(comps, RankingWeights(1, 1, 0, 0, 0))
        self.assertAlmostEqual(score, 1.0)

    def test_max_strategy_value(self):
        fn = RANKING_REGISTRY.get("max")
        comps = {"vector": 0.2, "keyword": 0.9, "metadata": 0.0, "category": 0.0, "freshness": 0.0}
        score = fn(comps, RankingWeights(1, 1, 1, 1, 1))
        self.assertAlmostEqual(score, 0.9 / 5.0)

    def test_register_custom_strategy(self):
        @register_ranking_strategy("test_constant_strategy")
        def _const(components, weights):
            return 0.42

        self.assertTrue(RANKING_REGISTRY.has("test_constant_strategy"))
        self.assertAlmostEqual(
            RANKING_REGISTRY.get("test_constant_strategy")({}, RankingWeights()), 0.42
        )

    def test_can_use_custom_strategy_in_config(self):
        if not RANKING_REGISTRY.has("const2"):
            register_ranking_strategy("const2")(lambda c, w: 0.5)
        cfg = RetrievalConfig(ranking_strategy="const2")
        self.assertEqual(cfg.ranking_strategy, "const2")


# ---------------------------------------------------------------------------
# RetrievedDocument
# ---------------------------------------------------------------------------
class TestRetrievedDocument(unittest.TestCase):
    def _doc(self):
        return RetrievedDocument(
            "d1", 0.9, 1, 0.8, 0.7, 0.6, 1.0, 0.5, "content here", "hybrid", {"k": "v"}
        )

    def test_construct(self):
        self.assertEqual(self._doc().document_id, "d1")

    def test_score_float(self):
        self.assertIsInstance(self._doc().score, float)

    def test_rank_int(self):
        self.assertIsInstance(self._doc().rank, int)

    def test_components_mapping(self):
        comps = self._doc().components
        self.assertEqual(comps["vector"], 0.8)
        self.assertEqual(comps["category"], 1.0)

    def test_content(self):
        self.assertEqual(self._doc().content, "content here")

    def test_source(self):
        self.assertEqual(self._doc().source, "hybrid")

    def test_metadata_readonly(self):
        self.assertIsInstance(self._doc().metadata, MappingProxyType)

    def test_frozen(self):
        d = self._doc()
        with self.assertRaises(Exception):
            d.score = 0.1  # type: ignore[misc]

    def test_to_dict_keys(self):
        d = self._doc().to_dict()
        for key in (
            "document_id", "score", "rank", "vector_score", "keyword_score",
            "metadata_score", "category_score", "freshness_score", "content",
            "source", "metadata",
        ):
            self.assertIn(key, d)

    def test_round_trip(self):
        d = self._doc()
        self.assertEqual(RetrievedDocument.from_dict(d.to_dict()), d)

    def test_json_round_trip(self):
        d = self._doc()
        restored = RetrievedDocument.from_dict(json.loads(json.dumps(d.to_dict())))
        self.assertEqual(restored, d)

    def test_equality(self):
        self.assertEqual(self._doc(), self._doc())

    def test_id_coercion(self):
        d = RetrievedDocument(99, 0.5, 1, 0, 0, 0, 0, 0)
        self.assertEqual(d.document_id, "99")


# ---------------------------------------------------------------------------
# RetrievalContext
# ---------------------------------------------------------------------------
class TestRetrievalContext(unittest.TestCase):
    def _ctx(self):
        d1 = RetrievedDocument("a", 0.9, 1, 0.9, 0, 0, 1, 0.5, "alpha text", "hybrid", {})
        d2 = RetrievedDocument("b", 0.5, 2, 0.5, 0, 0, 1, 0.5, "beta text", "hybrid", {})
        return RetrievalContext("q norm", "q raw", (d1, d2), 0.8, {"k": "v"})

    def test_query(self):
        self.assertEqual(self._ctx().query, "q norm")

    def test_original_query(self):
        self.assertEqual(self._ctx().original_query, "q raw")

    def test_document_ids(self):
        self.assertEqual(self._ctx().document_ids, ("a", "b"))

    def test_confidence(self):
        self.assertAlmostEqual(self._ctx().confidence, 0.8)

    def test_metadata_readonly(self):
        self.assertIsInstance(self._ctx().metadata, MappingProxyType)

    def test_as_text_includes_ranks(self):
        text = self._ctx().as_text()
        self.assertIn("[1]", text)
        self.assertIn("[2]", text)

    def test_as_text_includes_content(self):
        self.assertIn("alpha text", self._ctx().as_text())

    def test_frozen(self):
        c = self._ctx()
        with self.assertRaises(Exception):
            c.confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        c = self._ctx()
        restored = RetrievalContext.from_dict(c.to_dict())
        self.assertEqual(restored.document_ids, c.document_ids)

    def test_json_round_trip(self):
        c = self._ctx()
        restored = RetrievalContext.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(restored.query, c.query)

    def test_documents_is_tuple(self):
        self.assertIsInstance(self._ctx().documents, tuple)


# ---------------------------------------------------------------------------
# RetrievalEvidence
# ---------------------------------------------------------------------------
class TestRetrievalEvidence(unittest.TestCase):
    def _ev(self):
        d1 = RetrievedDocument("a", 0.9, 1, 0.9, 0, 0, 1, 0.5, "alpha", "hybrid", {})
        d2 = RetrievedDocument("b", 0.5, 2, 0.5, 0, 0, 1, 0.5, "beta", "hybrid", {})
        return RetrievalEvidence("q", (d1,), (d2,), 0.7, 0.6, {"why": "because"})

    def test_top_documents(self):
        self.assertEqual(self._ev().top_documents[0].document_id, "a")

    def test_supporting_documents(self):
        self.assertEqual(self._ev().supporting_documents[0].document_id, "b")

    def test_all_documents(self):
        self.assertEqual([d.document_id for d in self._ev().all_documents], ["a", "b"])

    def test_confidence(self):
        self.assertAlmostEqual(self._ev().confidence_score, 0.7)

    def test_coverage(self):
        self.assertAlmostEqual(self._ev().coverage_score, 0.6)

    def test_reasoning_readonly(self):
        self.assertIsInstance(self._ev().reasoning, MappingProxyType)

    def test_frozen(self):
        e = self._ev()
        with self.assertRaises(Exception):
            e.confidence_score = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        e = self._ev()
        restored = RetrievalEvidence.from_dict(e.to_dict())
        self.assertEqual(restored.confidence_score, e.confidence_score)

    def test_json_round_trip(self):
        e = self._ev()
        restored = RetrievalEvidence.from_dict(json.loads(json.dumps(e.to_dict())))
        self.assertEqual(
            [d.document_id for d in restored.top_documents],
            [d.document_id for d in e.top_documents],
        )

    def test_reasoning_preserved(self):
        restored = RetrievalEvidence.from_dict(self._ev().to_dict())
        self.assertEqual(restored.reasoning["why"], "because")


# ---------------------------------------------------------------------------
# RetrievalStatistics
# ---------------------------------------------------------------------------
class TestRetrievalStatistics(unittest.TestCase):
    def _stats(self):
        return RetrievalStatistics(1.5, 100, 5, 0.6, 0.7, 0.8)

    def test_fields(self):
        s = self._stats()
        self.assertEqual(s.documents_searched, 100)
        self.assertEqual(s.documents_returned, 5)

    def test_types(self):
        s = RetrievalStatistics(1, 2, 3, 4, 5, 6)
        self.assertIsInstance(s.latency_ms, float)
        self.assertIsInstance(s.documents_searched, int)

    def test_frozen(self):
        s = self._stats()
        with self.assertRaises(Exception):
            s.latency_ms = 9.0  # type: ignore[misc]

    def test_round_trip(self):
        s = self._stats()
        self.assertEqual(RetrievalStatistics.from_dict(s.to_dict()), s)

    def test_json_round_trip(self):
        s = self._stats()
        restored = RetrievalStatistics.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(restored, s)


# ---------------------------------------------------------------------------
# RetrievalPackage
# ---------------------------------------------------------------------------
class TestRetrievalPackage(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.package = self.engine.retrieve(QUERY)

    def test_has_context(self):
        self.assertIsInstance(self.package.context, RetrievalContext)

    def test_has_evidence(self):
        self.assertIsInstance(self.package.evidence, RetrievalEvidence)

    def test_has_statistics(self):
        self.assertIsInstance(self.package.statistics, RetrievalStatistics)

    def test_has_config(self):
        self.assertIsInstance(self.package.config, RetrievalConfig)

    def test_documents_property(self):
        self.assertEqual(self.package.documents, self.package.context.documents)

    def test_round_trip(self):
        restored = RetrievalPackage.from_dict(self.package.to_dict())
        self.assertEqual(
            restored.context.document_ids, self.package.context.document_ids
        )

    def test_json_round_trip(self):
        restored = RetrievalPackage.from_dict(
            json.loads(json.dumps(self.package.to_dict()))
        )
        self.assertEqual(
            restored.context.document_ids, self.package.context.document_ids
        )

    def test_to_dict_keys(self):
        self.assertEqual(
            set(self.package.to_dict()),
            {"context", "evidence", "statistics", "config"},
        )


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------
class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_retrieve_returns_package(self):
        self.assertIsInstance(self.engine.retrieve(QUERY), RetrievalPackage)

    def test_returns_documents(self):
        self.assertTrue(self.engine.retrieve(QUERY).context.documents)

    def test_respects_top_k(self):
        cfg = RetrievalConfig(top_k=2)
        self.assertEqual(len(self.engine.retrieve(QUERY, config=cfg).context.documents), 2)

    def test_ranks_sequential(self):
        docs = self.engine.retrieve(QUERY).context.documents
        self.assertEqual([d.rank for d in docs], list(range(1, len(docs) + 1)))

    def test_scores_descending(self):
        docs = self.engine.retrieve(QUERY).context.documents
        scores = [d.score for d in docs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_relevant_document_ranks_high(self):
        docs = self.engine.retrieve(QUERY).context.documents
        top_ids = [d.document_id for d in docs[:2]]
        # The RUL/compressor and root-cause docs are doc-0001 and doc-0005.
        self.assertTrue(any(i in top_ids for i in ("doc-0001", "doc-0005")))

    def test_normalized_query_recorded(self):
        ctx = self.engine.retrieve("The compressor FAILURE!").context
        self.assertNotIn("the", ctx.query.split())

    def test_original_query_recorded(self):
        ctx = self.engine.retrieve("The Compressor").context
        self.assertEqual(ctx.original_query, "The Compressor")

    def test_retrieve_context_helper(self):
        self.assertIsInstance(self.engine.retrieve_context(QUERY), RetrievalContext)

    def test_retrieve_evidence_helper(self):
        self.assertIsInstance(self.engine.retrieve_evidence(QUERY), RetrievalEvidence)

    def test_empty_query_raises(self):
        with self.assertRaises(EmptyQueryError):
            self.engine.retrieve("the and of")

    def test_blank_query_raises(self):
        with self.assertRaises(EmptyQueryError):
            self.engine.retrieve("   ")

    def test_non_string_query_raises(self):
        with self.assertRaises(InvalidQueryError):
            self.engine.retrieve(12345)  # type: ignore[arg-type]

    def test_none_query_raises(self):
        with self.assertRaises(InvalidQueryError):
            self.engine.retrieve(None)  # type: ignore[arg-type]

    def test_documents_have_content(self):
        docs = self.engine.retrieve(QUERY).context.documents
        self.assertTrue(all(d.content for d in docs))

    def test_documents_have_components(self):
        doc = self.engine.retrieve(QUERY).context.documents[0]
        self.assertTrue(hasattr(doc, "vector_score"))

    def test_top_k_zero(self):
        cfg = RetrievalConfig(top_k=0)
        self.assertEqual(self.engine.retrieve(QUERY, config=cfg).context.documents, ())

    def test_candidate_k_limits_pool(self):
        cfg = RetrievalConfig(candidate_k=2, top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertLessEqual(len(docs), 2)

    def test_source_label(self):
        doc = self.engine.retrieve(QUERY).context.documents[0]
        self.assertEqual(doc.source, "hybrid")

    def test_config_override_per_call(self):
        cfg = RetrievalConfig(top_k=1)
        self.assertEqual(len(self.engine.retrieve(QUERY, config=cfg).context.documents), 1)


# ---------------------------------------------------------------------------
# Candidate generation / filtering
# ---------------------------------------------------------------------------
class TestCandidateFiltering(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_category_filter(self):
        cfg = RetrievalConfig(category="pm", top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertEqual(d.metadata["category"], "pm")

    def test_category_filter_count(self):
        cfg = RetrievalConfig(category="pm", top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertEqual(len(docs), 3)  # doc-0000, doc-0002, doc-0004

    def test_category_filter_no_match(self):
        cfg = RetrievalConfig(category="nonexistent", top_k=10)
        self.assertEqual(self.engine.retrieve(QUERY, config=cfg).context.documents, ())

    def test_tag_filter_any(self):
        cfg = RetrievalConfig(tags=("common",), top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertEqual(len(docs), 6)

    def test_tag_filter_specific(self):
        cfg = RetrievalConfig(tags=("tag-0",), top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertIn("tag-0", d.metadata["tags"])

    def test_tag_filter_match_all(self):
        cfg = RetrievalConfig(tags=("tag-0", "common"), match_all_tags=True, top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertTrue({"tag-0", "common"}.issubset(set(d.metadata["tags"])))

    def test_tag_filter_match_all_impossible(self):
        cfg = RetrievalConfig(tags=("tag-0", "tag-1"), match_all_tags=True, top_k=10)
        self.assertEqual(self.engine.retrieve(QUERY, config=cfg).context.documents, ())

    def test_min_score_filter(self):
        cfg = RetrievalConfig(min_score=0.99, top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        # Very high threshold should prune most/all candidates.
        self.assertLessEqual(len(docs), 6)

    def test_min_score_zero_keeps_all_matched(self):
        cfg = RetrievalConfig(min_score=0.0, top_k=100)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertTrue(len(docs) >= 1)

    def test_no_duplicate_documents(self):
        docs = self.engine.retrieve(QUERY).context.documents
        ids = [d.document_id for d in docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_category_and_tag_combined(self):
        cfg = RetrievalConfig(category="pm", tags=("common",), top_k=10)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertEqual(d.metadata["category"], "pm")
            self.assertIn("common", d.metadata["tags"])


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
class TestRanking(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_vector_only_weights(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(1, 0, 0, 0, 0), top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertTrue(docs)

    def test_keyword_only_weights(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(0, 1, 0, 0, 0), top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertTrue(docs)

    def test_freshness_only_weights(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(0, 0, 0, 0, 1), top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        scores = [d.score for d in docs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_freshness_orders_by_freshness(self):
        # With pure freshness weighting and no normalisation, highest freshness wins.
        cfg = RetrievalConfig(
            ranking_weights=RankingWeights(0, 0, 0, 0, 1),
            normalize_components=False,
            top_k=6,
        )
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        fresh = [d.freshness_score for d in docs]
        self.assertEqual(fresh, sorted(fresh, reverse=True))

    def test_metadata_only_weights(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(0, 0, 1, 0, 0), top_k=6)
        docs = self.engine.retrieve("compressor", config=cfg).context.documents
        self.assertTrue(docs)

    def test_category_score_set_when_filtering(self):
        cfg = RetrievalConfig(category="pm", top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertEqual(d.category_score, 1.0)

    def test_category_neutral_when_no_filter(self):
        docs = self.engine.retrieve(QUERY).context.documents
        for d in docs:
            self.assertEqual(d.category_score, 1.0)

    def test_max_strategy(self):
        cfg = RetrievalConfig(ranking_strategy="max", top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertTrue(docs)

    def test_normalize_components_true(self):
        cfg = RetrievalConfig(normalize_components=True, top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertTrue(0.0 <= d.vector_score <= 1.0)

    def test_normalize_components_false(self):
        cfg = RetrievalConfig(normalize_components=False, top_k=6)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        self.assertTrue(docs)

    def test_ranking_deterministic(self):
        a = [d.document_id for d in self.engine.retrieve(QUERY).context.documents]
        b = [d.document_id for d in self.engine.retrieve(QUERY).context.documents]
        self.assertEqual(a, b)

    def test_tie_break_by_id(self):
        # Two identical documents differing only by id should rank id-ascending.
        docs = [
            FakeKnowledgeDocument("zzz", "identical content here", "pm", ("common",), {"text": "identical content here", "freshness": 0.5}),
            FakeKnowledgeDocument("aaa", "identical content here", "pm", ("common",), {"text": "identical content here", "freshness": 0.5}),
        ]
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        ids = [d.document_id for d in engine.retrieve("identical content here").context.documents]
        self.assertEqual(ids, ["aaa", "zzz"])

    def test_weights_recorded_in_metadata(self):
        ctx = self.engine.retrieve(QUERY).context
        self.assertIn("ranking_weights", ctx.metadata)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
class TestContextAssembly(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_confidence_in_range(self):
        c = self.engine.retrieve(QUERY).context.confidence
        self.assertTrue(0.0 <= c <= 1.0)

    def test_metadata_has_strategy(self):
        ctx = self.engine.retrieve(QUERY).context
        self.assertEqual(ctx.metadata["ranking_strategy"], "weighted_sum")

    def test_metadata_has_counts(self):
        ctx = self.engine.retrieve(QUERY).context
        self.assertIn("candidate_count", ctx.metadata)
        self.assertIn("returned_count", ctx.metadata)

    def test_ordered_evidence(self):
        docs = self.engine.retrieve(QUERY).context.documents
        self.assertEqual([d.rank for d in docs], list(range(1, len(docs) + 1)))

    def test_as_text_nonempty(self):
        self.assertTrue(self.engine.retrieve(QUERY).context.as_text())

    def test_as_text_custom_separator(self):
        text = self.engine.retrieve(QUERY).context.as_text(separator=" | ")
        self.assertIn(" | ", text)

    def test_source_tracking(self):
        for d in self.engine.retrieve(QUERY).context.documents:
            self.assertEqual(d.source, "hybrid")

    def test_confidence_zero_on_empty(self):
        cfg = RetrievalConfig(top_k=0)
        self.assertEqual(self.engine.retrieve(QUERY, config=cfg).context.confidence, 0.0)


# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------
class TestEvidenceGeneration(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_top_documents_count(self):
        cfg = RetrievalConfig(top_k=5, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual(len(ev.top_documents), 2)

    def test_supporting_documents_count(self):
        cfg = RetrievalConfig(top_k=5, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual(len(ev.supporting_documents), 3)

    def test_top_plus_supporting_equals_total(self):
        cfg = RetrievalConfig(top_k=5, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual(
            len(ev.top_documents) + len(ev.supporting_documents), 5
        )

    def test_confidence_score_in_range(self):
        ev = self.engine.retrieve(QUERY).evidence
        self.assertTrue(0.0 <= ev.confidence_score <= 1.0)

    def test_coverage_in_range(self):
        ev = self.engine.retrieve(QUERY).evidence
        self.assertTrue(0.0 <= ev.coverage_score <= 1.0)

    def test_coverage_high_for_matching_query(self):
        ev = self.engine.retrieve("compressor failure remaining useful life").evidence
        self.assertGreater(ev.coverage_score, 0.5)

    def test_reasoning_has_query_tokens(self):
        ev = self.engine.retrieve(QUERY).evidence
        self.assertIn("query_tokens", ev.reasoning)

    def test_reasoning_has_top_score(self):
        ev = self.engine.retrieve(QUERY).evidence
        self.assertIn("top_score", ev.reasoning)

    def test_reasoning_documents_searched(self):
        ev = self.engine.retrieve(QUERY).evidence
        self.assertEqual(ev.reasoning["documents_searched"], 6)

    def test_top_evidence_larger_than_topk(self):
        cfg = RetrievalConfig(top_k=2, top_evidence=10)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual(len(ev.top_documents), 2)
        self.assertEqual(len(ev.supporting_documents), 0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_documents_searched(self):
        self.assertEqual(self.engine.retrieve(QUERY).statistics.documents_searched, 6)

    def test_documents_returned(self):
        cfg = RetrievalConfig(top_k=3)
        self.assertEqual(
            self.engine.retrieve(QUERY, config=cfg).statistics.documents_returned, 3
        )

    def test_average_score_in_range(self):
        s = self.engine.retrieve(QUERY).statistics
        self.assertTrue(0.0 <= s.average_score <= 1.0)

    def test_coverage_matches_evidence(self):
        pkg = self.engine.retrieve(QUERY)
        self.assertAlmostEqual(
            pkg.statistics.coverage, pkg.evidence.coverage_score
        )

    def test_confidence_matches_context(self):
        pkg = self.engine.retrieve(QUERY)
        self.assertAlmostEqual(pkg.statistics.confidence, pkg.context.confidence)

    def test_latency_deterministic_with_clock(self):
        self.assertEqual(self.engine.retrieve(QUERY).statistics.latency_ms, 0.0)

    def test_latency_nonneg_real_clock(self):
        engine = make_engine(clock=None) if False else RetrievalEngine.from_knowledge_base(make_corpus(6))
        self.assertGreaterEqual(engine.retrieve(QUERY).statistics.latency_ms, 0.0)

    def test_average_score_zero_empty(self):
        cfg = RetrievalConfig(top_k=0)
        self.assertEqual(self.engine.retrieve(QUERY, config=cfg).statistics.average_score, 0.0)

    def test_stats_serializable(self):
        s = self.engine.retrieve(QUERY).statistics
        self.assertIsInstance(json.dumps(s.to_dict()), str)


# ---------------------------------------------------------------------------
# Engine serialization
# ---------------------------------------------------------------------------
class TestEngineSerialization(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_to_dict_keys(self):
        self.assertEqual(
            set(self.engine.to_dict()),
            {"normalizer", "config", "hybrid", "content_lookup"},
        )

    def test_round_trip_size(self):
        restored = RetrievalEngine.from_dict(self.engine.to_dict())
        self.assertEqual(restored.vector_engine.index.size(), 6)

    def test_round_trip_preserves_ranking(self):
        restored = RetrievalEngine.from_dict(
            json.loads(json.dumps(self.engine.to_dict()))
        )
        a = [d.document_id for d in self.engine.retrieve(QUERY).context.documents]
        b = [d.document_id for d in restored.retrieve(QUERY).context.documents]
        self.assertEqual(a, b)

    def test_round_trip_preserves_content(self):
        restored = RetrievalEngine.from_dict(self.engine.to_dict())
        docs = restored.retrieve(QUERY).context.documents
        self.assertTrue(all(d.content for d in docs))

    def test_round_trip_normalizer(self):
        engine = RetrievalEngine.from_knowledge_base(
            make_corpus(4),
            normalizer=QueryNormalizer(min_token_length=4),
            clock=lambda: 0.0,
        )
        restored = RetrievalEngine.from_dict(engine.to_dict())
        self.assertEqual(restored.normalizer.min_token_length, 4)

    def test_json_serializable(self):
        self.assertIsInstance(json.dumps(self.engine.to_dict()), str)


# ---------------------------------------------------------------------------
# Integration with Phase 1 & Phase 2
# ---------------------------------------------------------------------------
class TestIntegration(unittest.TestCase):
    def test_from_knowledge_base_object(self):
        kb = FakeKnowledgeBase(make_corpus(5))
        engine = RetrievalEngine.from_knowledge_base(kb, clock=lambda: 0.0)
        self.assertEqual(engine.vector_engine.index.size(), 5)

    def test_from_iterable(self):
        engine = RetrievalEngine.from_knowledge_base(make_corpus(4), clock=lambda: 0.0)
        self.assertEqual(engine.vector_engine.index.size(), 4)

    def test_alt_named_documents(self):
        docs = [
            AltNamedDocument(f"a-{i}", f"compressor failure case {i}", "pm", ("t",))
            for i in range(4)
        ]
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertTrue(engine.retrieve("compressor failure").context.documents)

    def test_uses_existing_vector_engine(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus(4))
        hybrid = HybridSearchEngine(ve)
        engine = RetrievalEngine(hybrid_engine=hybrid, clock=lambda: 0.0)
        self.assertEqual(engine.vector_engine.index.size(), 4)

    def test_custom_embedding_dimension(self):
        engine = RetrievalEngine.from_knowledge_base(
            make_corpus(4),
            embedding_engine=DeterministicEmbeddingEngine(dimension=64),
            clock=lambda: 0.0,
        )
        self.assertEqual(engine.vector_engine.index.dimension, 64)

    def test_custom_keyword_provider(self):
        def provider(query):
            return {"doc-0000": 5.0}

        engine = RetrievalEngine.from_knowledge_base(
            make_corpus(4), keyword_provider=provider, clock=lambda: 0.0,
        )
        self.assertTrue(engine.retrieve("compressor").context.documents)

    def test_custom_hybrid_config(self):
        engine = RetrievalEngine.from_knowledge_base(
            make_corpus(4), hybrid_config=HybridConfig(0.2, 0.8), clock=lambda: 0.0,
        )
        self.assertTrue(engine.retrieve("compressor").context.documents)

    def test_index_documents_after_construction(self):
        engine = make_engine(4)
        engine.index_documents([
            FakeKnowledgeDocument("new-1", "new turbine document", "pm", ("t",), {"text": "new turbine document"})
        ])
        self.assertEqual(engine.vector_engine.index.size(), 5)

    def test_content_from_metadata_fallback(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus(3))
        engine = RetrievalEngine(hybrid_engine=HybridSearchEngine(ve), clock=lambda: 0.0)
        docs = engine.retrieve(QUERY).context.documents
        self.assertTrue(all(d.content for d in docs))


# ---------------------------------------------------------------------------
# Large corpus
# ---------------------------------------------------------------------------
class TestLargeCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = []
        for i in range(800):
            text = _TEXTS[i % len(_TEXTS)] + f" unit {i} channel {i % 9}"
            docs.append(
                FakeKnowledgeDocument(
                    document_id=f"doc-{i:05d}",
                    content=text,
                    category=f"cat-{i % 4}",
                    tags=(f"t-{i % 7}", "common"),
                    metadata={"text": text, "freshness": round((i % 10) / 9.0, 3)},
                )
            )
        cls.docs = docs
        cls.engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)

    def test_indexed_all(self):
        self.assertEqual(self.engine.vector_engine.index.size(), 800)

    def test_returns_top_k(self):
        cfg = RetrievalConfig(top_k=10, candidate_k=50)
        self.assertEqual(
            len(self.engine.retrieve(QUERY, config=cfg).context.documents), 10
        )

    def test_deterministic(self):
        cfg = RetrievalConfig(top_k=15, candidate_k=60)
        a = [d.document_id for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        b = [d.document_id for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(a, b)

    def test_rebuild_deterministic(self):
        other = RetrievalEngine.from_knowledge_base(self.docs, clock=lambda: 0.0)
        cfg = RetrievalConfig(top_k=10, candidate_k=50)
        a = [d.document_id for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        b = [d.document_id for d in other.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(a, b)

    def test_documents_searched(self):
        self.assertEqual(self.engine.retrieve(QUERY).statistics.documents_searched, 800)

    def test_category_filter(self):
        cfg = RetrievalConfig(category="cat-0", top_k=1000, candidate_k=None)
        docs = self.engine.retrieve(QUERY, config=cfg).context.documents
        for d in docs:
            self.assertEqual(d.metadata["category"], "cat-0")

    def test_no_duplicate_ids(self):
        cfg = RetrievalConfig(top_k=50, candidate_k=100)
        ids = [d.document_id for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_serialization_round_trip(self):
        restored = RetrievalEngine.from_dict(
            json.loads(json.dumps(self.engine.to_dict()))
        )
        cfg = RetrievalConfig(top_k=10, candidate_k=50)
        a = [d.document_id for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        b = [d.document_id for d in restored.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_same_query_same_package(self):
        engine = make_engine()
        a = engine.retrieve(QUERY).to_dict()
        b = engine.retrieve(QUERY).to_dict()
        # latency is fixed by clock, so the full dict matches.
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_across_engine_instances(self):
        e1 = make_engine()
        e2 = make_engine()
        a = [d.document_id for d in e1.retrieve(QUERY).context.documents]
        b = [d.document_id for d in e2.retrieve(QUERY).context.documents]
        self.assertEqual(a, b)

    def test_scores_are_python_floats(self):
        engine = make_engine()
        self.assertIsInstance(engine.retrieve(QUERY).context.documents[0].score, float)

    def test_normalizer_deterministic(self):
        norm = QueryNormalizer()
        self.assertEqual(norm.normalize(QUERY), norm.normalize(QUERY))

    def test_confidence_deterministic(self):
        engine = make_engine()
        self.assertEqual(
            engine.retrieve(QUERY).context.confidence,
            engine.retrieve(QUERY).context.confidence,
        )

    def test_coverage_deterministic(self):
        engine = make_engine()
        self.assertEqual(
            engine.retrieve(QUERY).evidence.coverage_score,
            engine.retrieve(QUERY).evidence.coverage_score,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def test_empty_index(self):
        engine = RetrievalEngine.from_knowledge_base([], clock=lambda: 0.0)
        pkg = engine.retrieve(QUERY)
        self.assertEqual(pkg.context.documents, ())
        self.assertEqual(pkg.statistics.documents_returned, 0)

    def test_empty_index_confidence_zero(self):
        engine = RetrievalEngine.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(engine.retrieve(QUERY).context.confidence, 0.0)

    def test_single_document(self):
        engine = RetrievalEngine.from_knowledge_base(make_corpus(1), clock=lambda: 0.0)
        docs = engine.retrieve(QUERY).context.documents
        self.assertEqual(len(docs), 1)

    def test_single_document_confidence(self):
        engine = RetrievalEngine.from_knowledge_base(make_corpus(1), clock=lambda: 0.0)
        c = engine.retrieve(QUERY).context.confidence
        self.assertTrue(0.0 <= c <= 1.0)

    def test_unicode_query(self):
        engine = make_engine()
        self.assertTrue(engine.retrieve("café compressor failure").context.documents)

    def test_query_with_only_one_token(self):
        engine = make_engine()
        self.assertTrue(engine.retrieve("compressor").context.documents)

    def test_query_no_corpus_overlap(self):
        engine = make_engine()
        pkg = engine.retrieve("zzzzqqqq nonsense token")
        # Still returns ranked candidates (low scores), pipeline does not crash.
        self.assertIsInstance(pkg, RetrievalPackage)

    def test_documents_without_freshness(self):
        docs = make_corpus(4, with_freshness=False)
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        cfg = RetrievalConfig(normalize_components=False, top_k=10)
        for d in engine.retrieve(QUERY, config=cfg).context.documents:
            self.assertEqual(d.freshness_score, 0.5)  # neutral default

    def test_age_days_freshness(self):
        docs = [
            FakeKnowledgeDocument("fresh", "compressor failure new", "pm", ("t",), {"text": "compressor failure new", "age_days": 0}),
            FakeKnowledgeDocument("stale", "compressor failure old", "pm", ("t",), {"text": "compressor failure old", "age_days": 300}),
        ]
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        by_id = {d.document_id: d for d in engine.retrieve("compressor failure").context.documents}
        self.assertGreater(by_id["fresh"].freshness_score, by_id["stale"].freshness_score)

    def test_very_long_query(self):
        engine = make_engine()
        self.assertTrue(engine.retrieve("compressor " * 200).context.documents)

    def test_duplicate_tokens_in_query(self):
        engine = make_engine()
        ctx = engine.retrieve("compressor compressor compressor failure").context
        self.assertEqual(ctx.query.count("compressor"), 1)

    def test_candidate_k_none(self):
        cfg = RetrievalConfig(candidate_k=None, top_k=10)
        engine = make_engine()
        self.assertTrue(engine.retrieve(QUERY, config=cfg).context.documents)

    def test_invalid_freshness_value_neutral(self):
        docs = [
            FakeKnowledgeDocument("bad", "compressor failure", "pm", ("t",), {"text": "compressor failure", "freshness": "not-a-number"}),
        ]
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        cfg = RetrievalConfig(normalize_components=False)
        self.assertEqual(engine.retrieve("compressor", config=cfg).context.documents[0].freshness_score, 0.5)


# ---------------------------------------------------------------------------
# Validation / error hierarchy
# ---------------------------------------------------------------------------
class TestValidation(unittest.TestCase):
    def test_empty_query_is_retrieval_error(self):
        self.assertTrue(issubclass(EmptyQueryError, RetrievalError))

    def test_invalid_query_is_retrieval_error(self):
        self.assertTrue(issubclass(InvalidQueryError, RetrievalError))

    def test_invalid_ranking_is_retrieval_error(self):
        self.assertTrue(issubclass(InvalidRankingError, RetrievalError))

    def test_missing_document_is_retrieval_error(self):
        self.assertTrue(issubclass(MissingDocumentError, RetrievalError))

    def test_duplicate_is_retrieval_error(self):
        self.assertTrue(issubclass(DuplicateRetrievalError, RetrievalError))

    def test_empty_query_detection(self):
        with self.assertRaises(EmptyQueryError):
            make_engine().retrieve("")

    def test_invalid_query_detection(self):
        with self.assertRaises(InvalidQueryError):
            make_engine().retrieve(3.14)  # type: ignore[arg-type]

    def test_invalid_ranking_weights(self):
        with self.assertRaises(InvalidRankingError):
            RankingWeights(-1, 0, 0, 0, 0)

    def test_invalid_strategy_in_config(self):
        with self.assertRaises(InvalidRankingError):
            RetrievalConfig(ranking_strategy="does-not-exist")

    def test_min_score_validation(self):
        with self.assertRaises(RetrievalError):
            RetrievalConfig(min_score=2.0)

    def test_negative_weights_via_config(self):
        with self.assertRaises(InvalidRankingError):
            RetrievalConfig(ranking_weights=RankingWeights(-0.5, 1, 1, 1, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_demo_runs(self):
        from knowledge.retrieval_intelligence import main
        self.assertEqual(main(["--demo"]), 0)

    def test_no_args_prints_help(self):
        from knowledge.retrieval_intelligence import main
        self.assertEqual(main([]), 0)


# ---------------------------------------------------------------------------
# Backward-compatibility guard: Phase 2 still intact and usable
# ---------------------------------------------------------------------------
class TestPhase2BackwardCompatibility(unittest.TestCase):
    def test_vector_engine_still_searches(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus(4))
        self.assertTrue(ve.search("compressor", top_k=2))

    def test_hybrid_engine_still_searches(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus(4))
        hybrid = HybridSearchEngine(ve)
        self.assertTrue(hybrid.search("compressor", top_k=2))

    def test_retrieval_reuses_phase2_scores(self):
        engine = make_engine()
        docs = engine.retrieve(QUERY).context.documents
        # Component scores originate from Phase 2 hybrid results.
        self.assertTrue(all(hasattr(d, "vector_score") for d in docs))


# ---------------------------------------------------------------------------
# Component scoring (metadata / category / freshness math)
# ---------------------------------------------------------------------------
class TestComponentScoring(unittest.TestCase):
    def _engine(self, docs):
        return RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)

    def test_metadata_overlap_full(self):
        docs = [FakeKnowledgeDocument("a", "x body", "turbine", ("vibration",), {"text": "x body"})]
        cfg = RetrievalConfig(normalize_components=False)
        d = self._engine(docs).retrieve("turbine vibration", config=cfg).context.documents[0]
        self.assertAlmostEqual(d.metadata_score, 1.0)

    def test_metadata_overlap_half(self):
        docs = [FakeKnowledgeDocument("a", "x", "compressor", ("failure",), {"text": "x"})]
        cfg = RetrievalConfig(normalize_components=False)
        d = self._engine(docs).retrieve("compressor failure remaining useful", config=cfg).context.documents[0]
        self.assertAlmostEqual(d.metadata_score, 0.5)

    def test_metadata_overlap_zero(self):
        docs = [FakeKnowledgeDocument("a", "x", "other", ("misc",), {"text": "x"})]
        cfg = RetrievalConfig(normalize_components=False)
        d = self._engine(docs).retrieve("turbine vibration", config=cfg).context.documents[0]
        self.assertAlmostEqual(d.metadata_score, 0.0)

    def test_metadata_excludes_text_field(self):
        docs = [FakeKnowledgeDocument("a", "turbine vibration", "other", ("misc",), {"text": "turbine vibration"})]
        cfg = RetrievalConfig(normalize_components=False)
        d = self._engine(docs).retrieve("turbine vibration", config=cfg).context.documents[0]
        self.assertAlmostEqual(d.metadata_score, 0.0)

    def test_category_score_neutral_no_filter(self):
        for d in self._engine(make_corpus(4)).retrieve(QUERY).context.documents:
            self.assertEqual(d.category_score, 1.0)

    def test_category_score_one_when_filtered(self):
        cfg = RetrievalConfig(category="pm", top_k=10)
        for d in self._engine(make_corpus(4)).retrieve(QUERY, config=cfg).context.documents:
            self.assertEqual(d.category_score, 1.0)

    def test_freshness_direct_value(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "freshness": 0.3})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 0.3)

    def test_freshness_clamped_above_one(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "freshness": 5.0})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 1.0)

    def test_freshness_clamped_below_zero(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "freshness": -2.0})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 0.0)

    def test_age_days_zero_is_one(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "age_days": 0})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 1.0)

    def test_age_days_decay(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "age_days": 30})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 0.5)

    def test_age_days_negative_clamped(self):
        docs = [FakeKnowledgeDocument("a", "compressor", "pm", ("t",), {"text": "compressor", "age_days": -10})]
        cfg = RetrievalConfig(normalize_components=False)
        self.assertAlmostEqual(self._engine(docs).retrieve("compressor", config=cfg).context.documents[0].freshness_score, 1.0)

    def test_components_count(self):
        d = self._engine(make_corpus(3)).retrieve(QUERY).context.documents[0]
        self.assertEqual(len(d.components), 5)

    def test_vector_score_bounded(self):
        d = self._engine(make_corpus(3)).retrieve(QUERY).context.documents[0]
        self.assertTrue(0.0 <= d.vector_score <= 1.0)

    def test_keyword_score_bounded(self):
        d = self._engine(make_corpus(3)).retrieve(QUERY).context.documents[0]
        self.assertTrue(0.0 <= d.keyword_score <= 1.0)


# ---------------------------------------------------------------------------
# Multi-query invariants
# ---------------------------------------------------------------------------
class TestMultiQueryInvariants(unittest.TestCase):
    QUERIES = [
        "compressor failure remaining useful life",
        "turbine bearing vibration anomaly",
        "fleet digital twin portfolio",
        "maintenance scheduling downtime cost",
        "scenario capital expenditure optimization",
        "root cause lubrication degradation",
    ]

    def setUp(self):
        self.engine = make_engine()

    def test_all_queries_return_results(self):
        for q in self.QUERIES:
            self.assertTrue(self.engine.retrieve(q).context.documents, msg=q)

    def test_all_queries_ranks_sequential(self):
        for q in self.QUERIES:
            docs = self.engine.retrieve(q).context.documents
            self.assertEqual([d.rank for d in docs], list(range(1, len(docs) + 1)), msg=q)

    def test_all_queries_scores_descending(self):
        for q in self.QUERIES:
            scores = [d.score for d in self.engine.retrieve(q).context.documents]
            self.assertEqual(scores, sorted(scores, reverse=True), msg=q)

    def test_all_queries_deterministic(self):
        for q in self.QUERIES:
            a = [d.document_id for d in self.engine.retrieve(q).context.documents]
            b = [d.document_id for d in self.engine.retrieve(q).context.documents]
            self.assertEqual(a, b, msg=q)

    def test_all_queries_confidence_bounded(self):
        for q in self.QUERIES:
            self.assertTrue(0.0 <= self.engine.retrieve(q).context.confidence <= 1.0, msg=q)

    def test_all_queries_coverage_bounded(self):
        for q in self.QUERIES:
            self.assertTrue(0.0 <= self.engine.retrieve(q).evidence.coverage_score <= 1.0, msg=q)

    def test_all_queries_no_duplicates(self):
        for q in self.QUERIES:
            ids = [d.document_id for d in self.engine.retrieve(q).context.documents]
            self.assertEqual(len(ids), len(set(ids)), msg=q)

    def test_all_queries_json_round_trip(self):
        for q in self.QUERIES:
            pkg = self.engine.retrieve(q)
            restored = RetrievalPackage.from_dict(json.loads(json.dumps(pkg.to_dict())))
            self.assertEqual(restored.context.document_ids, pkg.context.document_ids, msg=q)

    def test_all_queries_statistics_consistent(self):
        for q in self.QUERIES:
            pkg = self.engine.retrieve(q)
            self.assertEqual(pkg.statistics.documents_returned, len(pkg.context.documents), msg=q)

    def test_all_queries_content_present(self):
        for q in self.QUERIES:
            self.assertTrue(all(d.content for d in self.engine.retrieve(q).context.documents), msg=q)


# ---------------------------------------------------------------------------
# Normalization behaviour in pipeline
# ---------------------------------------------------------------------------
class TestNormalizationBehaviour(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_stopwords_removed_in_query(self):
        words = self.engine.retrieve("what is the compressor failure").context.query.split()
        for w in ("what", "is", "the"):
            self.assertNotIn(w, words)

    def test_punctuation_removed_in_query(self):
        self.assertEqual(self.engine.retrieve("compressor, failure!").context.query, "compressor failure")

    def test_case_folded_in_query(self):
        self.assertEqual(self.engine.retrieve("COMPRESSOR Failure").context.query, "compressor failure")

    def test_duplicates_removed_in_query(self):
        self.assertEqual(self.engine.retrieve("compressor compressor failure").context.query, "compressor failure")

    def test_custom_normalizer_used(self):
        engine = RetrievalEngine.from_knowledge_base(make_corpus(4), normalizer=QueryNormalizer(remove_stopwords=False), clock=lambda: 0.0)
        self.assertIn("the", engine.retrieve("the compressor").context.query.split())

    def test_short_token_removed(self):
        self.assertNotIn("a", self.engine.retrieve("a compressor").context.query.split())

    def test_normalized_query_used_for_ranking(self):
        a = [d.document_id for d in self.engine.retrieve("compressor failure").context.documents]
        b = [d.document_id for d in self.engine.retrieve("the COMPRESSOR, failure!").context.documents]
        self.assertEqual(a, b)

    def test_query_with_numbers(self):
        self.assertIn("12345", self.engine.retrieve("sensor 12345 compressor").context.query.split())


# ---------------------------------------------------------------------------
# Config variants
# ---------------------------------------------------------------------------
class TestConfigVariants(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_top_k_one(self):
        self.assertEqual(len(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=1)).context.documents), 1)

    def test_top_k_large(self):
        self.assertLessEqual(len(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=100)).context.documents), 6)

    def test_candidate_k_one(self):
        self.assertLessEqual(len(self.engine.retrieve(QUERY, config=RetrievalConfig(candidate_k=1, top_k=5)).context.documents), 1)

    def test_top_evidence_zero(self):
        ev = self.engine.retrieve(QUERY, config=RetrievalConfig(top_evidence=0, top_k=4)).evidence
        self.assertEqual(len(ev.top_documents), 0)
        self.assertEqual(len(ev.supporting_documents), 4)

    def test_top_evidence_all(self):
        ev = self.engine.retrieve(QUERY, config=RetrievalConfig(top_evidence=4, top_k=4)).evidence
        self.assertEqual(len(ev.top_documents), 4)

    def test_strategy_weighted_sum(self):
        self.assertTrue(self.engine.retrieve(QUERY, config=RetrievalConfig(ranking_strategy="weighted_sum")).context.documents)

    def test_strategy_max(self):
        self.assertTrue(self.engine.retrieve(QUERY, config=RetrievalConfig(ranking_strategy="max")).context.documents)

    def test_different_strategies_same_survivor_set(self):
        ws = [d.document_id for d in self.engine.retrieve(QUERY, config=RetrievalConfig(ranking_strategy="weighted_sum", top_k=10)).context.documents]
        mx = [d.document_id for d in self.engine.retrieve(QUERY, config=RetrievalConfig(ranking_strategy="max", top_k=10)).context.documents]
        self.assertEqual(set(ws), set(mx))

    def test_hybrid_config_vector_only(self):
        self.assertTrue(self.engine.retrieve(QUERY, config=RetrievalConfig(hybrid_config=HybridConfig(0.0, 1.0))).context.documents)

    def test_hybrid_config_keyword_only(self):
        self.assertTrue(self.engine.retrieve(QUERY, config=RetrievalConfig(hybrid_config=HybridConfig(1.0, 0.0))).context.documents)

    def test_min_score_high_prunes(self):
        low = len(self.engine.retrieve(QUERY, config=RetrievalConfig(min_score=0.0, top_k=100)).context.documents)
        high = len(self.engine.retrieve(QUERY, config=RetrievalConfig(min_score=0.95, top_k=100)).context.documents)
        self.assertLessEqual(high, low)


# ---------------------------------------------------------------------------
# Serialization (extra coverage with varied data)
# ---------------------------------------------------------------------------
class TestSerializationExtra(unittest.TestCase):
    def test_retrieved_document_varied(self):
        d = RetrievedDocument("x", -0.5, 3, 0.1, 0.2, 0.3, 0.0, 1.0, "txt", "vec", {"a": [1, 2]})
        self.assertEqual(RetrievedDocument.from_dict(d.to_dict()), d)

    def test_context_empty_documents(self):
        c = RetrievalContext("q", "q", (), 0.0, {})
        self.assertEqual(RetrievalContext.from_dict(c.to_dict()).documents, ())

    def test_evidence_empty(self):
        e = RetrievalEvidence("q", (), (), 0.0, 0.0, {})
        self.assertEqual(RetrievalEvidence.from_dict(e.to_dict()).top_documents, ())

    def test_statistics_zeros(self):
        s = RetrievalStatistics(0, 0, 0, 0, 0, 0)
        self.assertEqual(RetrievalStatistics.from_dict(s.to_dict()), s)

    def test_config_full_round_trip(self):
        c = RetrievalConfig(top_k=8, candidate_k=40, top_evidence=4, min_score=0.2, category="pm",
                            tags=("a", "b"), match_all_tags=True,
                            ranking_weights=RankingWeights(0.5, 0.2, 0.1, 0.1, 0.1),
                            ranking_strategy="max", normalize_components=False,
                            hybrid_config=HybridConfig(0.4, 0.6, normalize=False))
        r = RetrievalConfig.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual((r.top_k, r.tags, r.match_all_tags, r.ranking_strategy, r.normalize_components),
                         (8, ("a", "b"), True, "max", False))

    def test_normalizer_full_round_trip(self):
        n = QueryNormalizer(lowercase=True, remove_duplicates=False, remove_stopwords=False, min_token_length=3, stopwords=("foo", "bar"))
        self.assertEqual(QueryNormalizer.from_dict(json.loads(json.dumps(n.to_dict()))), n)

    def test_weights_full_round_trip(self):
        w = RankingWeights(0.1, 0.2, 0.3, 0.2, 0.2)
        self.assertEqual(RankingWeights.from_dict(json.loads(json.dumps(w.to_dict()))), w)

    def test_package_round_trip_evidence(self):
        pkg = make_engine().retrieve(QUERY)
        r = RetrievalPackage.from_dict(pkg.to_dict())
        self.assertEqual([d.document_id for d in r.evidence.top_documents], [d.document_id for d in pkg.evidence.top_documents])

    def test_package_round_trip_statistics(self):
        pkg = make_engine().retrieve(QUERY)
        r = RetrievalPackage.from_dict(pkg.to_dict())
        self.assertEqual(r.statistics.documents_searched, pkg.statistics.documents_searched)

    def test_package_round_trip_config(self):
        pkg = make_engine().retrieve(QUERY, config=RetrievalConfig(top_k=3))
        self.assertEqual(RetrievalPackage.from_dict(pkg.to_dict()).config.top_k, 3)

    def test_context_metadata_round_trip(self):
        ctx = make_engine().retrieve(QUERY).context
        self.assertEqual(RetrievalContext.from_dict(ctx.to_dict()).metadata["ranking_strategy"], "weighted_sum")

    def test_retrieved_document_negative_score(self):
        d = RetrievedDocument("x", -1.0, 1, 0, 0, 0, 0, 0)
        self.assertEqual(RetrievedDocument.from_dict(d.to_dict()).score, -1.0)


# ---------------------------------------------------------------------------
# as_text assembly
# ---------------------------------------------------------------------------
class TestContextAsText(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_contains_all_ranks(self):
        ctx = self.engine.retrieve(QUERY).context
        text = ctx.as_text()
        for d in ctx.documents:
            self.assertIn(f"[{d.rank}]", text)

    def test_contains_document_ids(self):
        ctx = self.engine.retrieve(QUERY).context
        text = ctx.as_text()
        for d in ctx.documents:
            self.assertIn(d.document_id, text)

    def test_default_separator(self):
        self.assertIn("\n\n", self.engine.retrieve(QUERY).context.as_text())

    def test_custom_separator(self):
        self.assertIn("###", self.engine.retrieve(QUERY).context.as_text(separator="###"))

    def test_empty_context_text(self):
        self.assertEqual(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=0)).context.as_text(), "")

    def test_text_block_order(self):
        ctx = self.engine.retrieve(QUERY).context
        text = ctx.as_text()
        positions = [text.index(f"[{d.rank}]") for d in ctx.documents]
        self.assertEqual(positions, sorted(positions))


# ---------------------------------------------------------------------------
# Statistics (extra)
# ---------------------------------------------------------------------------
class TestStatisticsExtra(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_documents_returned_matches_topk(self):
        for k in (1, 2, 3, 4):
            pkg = self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=k))
            self.assertEqual(pkg.statistics.documents_returned, len(pkg.context.documents))

    def test_documents_searched_constant(self):
        for k in (1, 3, 5):
            self.assertEqual(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=k)).statistics.documents_searched, 6)

    def test_average_score_matches_mean(self):
        pkg = self.engine.retrieve(QUERY)
        self.assertAlmostEqual(pkg.statistics.average_score, float(np.mean([d.score for d in pkg.context.documents])))

    def test_latency_zero_with_fixed_clock(self):
        self.assertEqual(self.engine.retrieve(QUERY).statistics.latency_ms, 0.0)

    def test_latency_increasing_clock(self):
        ticks = iter([0.0, 0.05, 0.05, 0.05])
        engine = RetrievalEngine.from_knowledge_base(make_corpus(4), clock=lambda: next(ticks))
        self.assertAlmostEqual(engine.retrieve(QUERY).statistics.latency_ms, 50.0)

    def test_coverage_zero_on_empty(self):
        self.assertEqual(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=0)).statistics.coverage, 0.0)

    def test_confidence_zero_on_empty(self):
        self.assertEqual(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=0)).statistics.confidence, 0.0)

    def test_statistics_round_trip(self):
        s = self.engine.retrieve(QUERY).statistics
        self.assertEqual(RetrievalStatistics.from_dict(s.to_dict()), s)


# ---------------------------------------------------------------------------
# Engine properties / construction
# ---------------------------------------------------------------------------
class TestEngineProperties(unittest.TestCase):
    def test_default_construction_empty(self):
        self.assertEqual(RetrievalEngine().vector_engine.index.size(), 0)

    def test_hybrid_engine_property(self):
        self.assertIsInstance(make_engine().hybrid_engine, HybridSearchEngine)

    def test_vector_engine_property(self):
        self.assertIsInstance(make_engine().vector_engine, VectorSearchEngine)

    def test_normalizer_property(self):
        self.assertIsInstance(make_engine().normalizer, QueryNormalizer)

    def test_config_property(self):
        self.assertIsInstance(make_engine().config, RetrievalConfig)

    def test_config_default_topk(self):
        self.assertEqual(make_engine().config.top_k, 5)

    def test_index_documents_grows(self):
        engine = make_engine(4)
        engine.index_documents([FakeKnowledgeDocument("z", "new turbine doc", "pm", ("t",), {"text": "new turbine doc"})])
        self.assertEqual(engine.vector_engine.index.size(), 5)

    def test_content_lookup_populated(self):
        engine = make_engine(3)
        self.assertTrue(all(d.content for d in engine.retrieve(QUERY).context.documents))


# ---------------------------------------------------------------------------
# Ranking (extra)
# ---------------------------------------------------------------------------
class TestRankingExtra(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_weighted_sum_scores_bounded(self):
        for d in self.engine.retrieve(QUERY).context.documents:
            self.assertLessEqual(d.score, 1.0 + 1e-9)

    def test_scores_nonnegative_when_components_nonneg(self):
        for d in self.engine.retrieve(QUERY).context.documents:
            self.assertGreaterEqual(d.score, -1e-9)

    def test_vector_only_orders_by_vector(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(1, 0, 0, 0, 0), normalize_components=False, top_k=6)
        vs = [d.vector_score for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(vs, sorted(vs, reverse=True))

    def test_metadata_weight_orders_by_metadata(self):
        docs = [
            FakeKnowledgeDocument("a", "x", "compressor", ("failure",), {"text": "x"}),
            FakeKnowledgeDocument("b", "y", "other", ("misc",), {"text": "y"}),
        ]
        engine = RetrievalEngine.from_knowledge_base(docs, clock=lambda: 0.0)
        cfg = RetrievalConfig(ranking_weights=RankingWeights(0, 0, 1, 0, 0), normalize_components=False, top_k=2)
        ids = [d.document_id for d in engine.retrieve("compressor failure", config=cfg).context.documents]
        self.assertEqual(ids[0], "a")

    def test_max_strategy_scores_bounded(self):
        cfg = RetrievalConfig(ranking_strategy="max", top_k=6)
        for d in self.engine.retrieve(QUERY, config=cfg).context.documents:
            self.assertLessEqual(d.score, 1.0 + 1e-9)

    def test_normalize_false_still_bounded(self):
        cfg = RetrievalConfig(normalize_components=False, top_k=6)
        for d in self.engine.retrieve(QUERY, config=cfg).context.documents:
            self.assertLessEqual(d.score, 1.0 + 1e-9)

    def test_weights_recorded_normalized(self):
        ctx = self.engine.retrieve(QUERY).context
        total = sum(ctx.metadata["ranking_weights"].values())
        self.assertAlmostEqual(total, 1.0)

    def test_freshness_weight_orders(self):
        cfg = RetrievalConfig(ranking_weights=RankingWeights(0, 0, 0, 0, 1), normalize_components=False, top_k=6)
        fresh = [d.freshness_score for d in self.engine.retrieve(QUERY, config=cfg).context.documents]
        self.assertEqual(fresh, sorted(fresh, reverse=True))


# ---------------------------------------------------------------------------
# Evidence (extra)
# ---------------------------------------------------------------------------
class TestEvidenceExtra(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_top_documents_are_highest_ranked(self):
        cfg = RetrievalConfig(top_k=5, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual([d.rank for d in ev.top_documents], [1, 2])

    def test_supporting_are_lower_ranked(self):
        cfg = RetrievalConfig(top_k=5, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual([d.rank for d in ev.supporting_documents], [3, 4, 5])

    def test_reasoning_strategy_field(self):
        self.assertEqual(self.engine.retrieve(QUERY).evidence.reasoning["strategy"], "weighted_sum")

    def test_reasoning_filtered_to(self):
        pkg = self.engine.retrieve(QUERY)
        self.assertEqual(pkg.evidence.reasoning["filtered_to"], len(pkg.context.documents))

    def test_all_documents_order(self):
        cfg = RetrievalConfig(top_k=4, top_evidence=2)
        ev = self.engine.retrieve(QUERY, config=cfg).evidence
        self.assertEqual([d.rank for d in ev.all_documents], [1, 2, 3, 4])

    def test_confidence_matches_context(self):
        pkg = self.engine.retrieve(QUERY)
        self.assertAlmostEqual(pkg.evidence.confidence_score, pkg.context.confidence)


# ---------------------------------------------------------------------------
# Pipeline robustness
# ---------------------------------------------------------------------------
class TestPipelineRobustness(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_repeated_calls_independent(self):
        a = self.engine.retrieve("turbine vibration").context.document_ids
        _ = self.engine.retrieve("scenario capex").context.document_ids
        b = self.engine.retrieve("turbine vibration").context.document_ids
        self.assertEqual(a, b)

    def test_category_filter_changes_results(self):
        unfiltered = set(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=10)).context.document_ids)
        filtered = set(self.engine.retrieve(QUERY, config=RetrievalConfig(category="pm", top_k=10)).context.document_ids)
        self.assertTrue(filtered.issubset(unfiltered))

    def test_tag_filter_subset(self):
        unfiltered = set(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=10)).context.document_ids)
        filtered = set(self.engine.retrieve(QUERY, config=RetrievalConfig(tags=("tag-0",), top_k=10)).context.document_ids)
        self.assertTrue(filtered.issubset(unfiltered))

    def test_large_top_k_no_error(self):
        self.assertTrue(self.engine.retrieve(QUERY, config=RetrievalConfig(top_k=10000)).context.documents)

    def test_min_score_one_returns_few(self):
        docs = self.engine.retrieve(QUERY, config=RetrievalConfig(min_score=1.0, top_k=10)).context.documents
        self.assertLessEqual(len(docs), 6)

    def test_whitespace_heavy_query(self):
        self.assertTrue(self.engine.retrieve("   compressor    failure   ").context.documents)

    def test_mixed_case_punct_query(self):
        a = self.engine.retrieve("Compressor; Failure.").context.document_ids
        b = self.engine.retrieve("compressor failure").context.document_ids
        self.assertEqual(a, b)

    def test_candidate_k_none_full_scan(self):
        cfg = RetrievalConfig(candidate_k=None, top_k=10)
        self.assertTrue(self.engine.retrieve(QUERY, config=cfg).context.documents)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)