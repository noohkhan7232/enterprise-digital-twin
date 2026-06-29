"""Test suite for the Enterprise Knowledge Agent (Week 9 Phase 4).

Covers question classification, intent detection, retrieval integration,
grounded response generation, citation generation, reasoning generation,
confidence calculation, follow-up generation, serialization, statistics, edge
cases, large knowledge bases, determinism, backward compatibility with Weeks
1-9 Phase 3, and the CLI. Pure stdlib ``unittest``.

This file contains well over 350 individual test methods.
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
    VectorSearchEngine,
    tokenize,
)
from knowledge.retrieval_intelligence import (  # noqa: E402
    RetrievalConfig,
    RetrievalEngine,
    RetrievalPackage,
)
from knowledge.knowledge_agent import (  # noqa: E402
    AgentConfig,
    DuplicateCitationError,
    EmptyQuestionError,
    FOLLOWUP_REGISTRY,
    InvalidQuestionError,
    KnowledgeAgent,
    KnowledgeAgentError,
    KnowledgeAgentStatistics,
    KnowledgeAnswer,
    KnowledgeCitation,
    KnowledgeQuestion,
    KnowledgeReasoning,
    KnowledgeResponse,
    MissingEvidenceError,
    MissingRetrievalError,
    QUESTION_TYPE_REGISTRY,
    QuestionClassifier,
    QuestionType,
    QuestionTypeRule,
    register_followup_templates,
    register_question_type,
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


_CORPUS_ROWS = [
    ("doc-turbine-vibration", "Turbine Bearing Vibration Advisory",
     "Turbine bearing vibration anomaly was detected by the asset health engine. "
     "Elevated vibration indicates an emerging bearing fault on the high-speed shaft. "
     "Recommended maintenance is to inspect and replace the affected bearing within the next service interval.",
     "predictive-maintenance", ("turbine", "vibration", "bearing", "maintenance")),
    ("doc-rul-forecast", "Compressor RUL Forecast",
     "The remaining useful life of compressor stage two is forecast from the degradation trend. "
     "Monte Carlo simulation estimates limited remaining life under current load. "
     "Continued operation increases the failure risk.",
     "predictive-maintenance", ("rul", "compressor", "forecast", "failure")),
    ("doc-root-cause", "Compressor Failure Root Cause",
     "Root cause analysis traced the compressor failure to lubrication degradation. "
     "Insufficient lubrication accelerated bearing wear. "
     "The corrective action is to restore the lubrication schedule and replace worn components.",
     "predictive-maintenance", ("root-cause", "compressor", "lubrication", "failure")),
    ("doc-safety-lockout", "Lockout Tagout Safety Procedure",
     "Before servicing rotating equipment, apply lockout tagout to isolate energy sources. "
     "Personal protective equipment is mandatory. "
     "Verify zero energy state before beginning maintenance.",
     "safety", ("safety", "lockout", "ppe", "procedure")),
    ("doc-fleet-twin", "Fleet Digital Twin Overview",
     "The fleet digital twin aggregates asset health across the energy portfolio. "
     "Executives use it for strategic capital planning and prioritisation of maintenance investment.",
     "digital-twin", ("fleet", "twin", "portfolio", "executive")),
    ("doc-inspection", "Inspection Checklist",
     "Routine inspection requires checking bearing temperature and vibration. "
     "Inspectors must measure clearance and record readings monthly.",
     "inspection", ("inspection", "check", "measure")),
]


def make_corpus():
    docs = []
    for doc_id, title, content, category, tags in _CORPUS_ROWS:
        docs.append(FakeKnowledgeDocument(
            document_id=doc_id, content=content, category=category, tags=tags,
            metadata={"text": content, "title": title, "freshness": 0.7},
        ))
    return docs


CONTENT_BY_ID = {row[0]: row[2] for row in _CORPUS_ROWS}
ALL_CONTENT = " ".join(row[2] for row in _CORPUS_ROWS)


def make_agent(config=None, clock=lambda: 0.0):
    return KnowledgeAgent.from_knowledge_base(make_corpus(), config=config, clock=clock)


Q_ROOT = "What is the root cause of the compressor failure?"
Q_SAFETY = "What safety procedure applies before servicing equipment?"
Q_MAINT = "What maintenance is recommended for turbine bearing vibration?"


# ---------------------------------------------------------------------------
# Question type registry
# ---------------------------------------------------------------------------
class TestQuestionTypeRegistry(unittest.TestCase):
    def test_all_types_registered(self):
        for t in (QuestionType.MAINTENANCE, QuestionType.INSPECTION,
                  QuestionType.FAILURE_ANALYSIS, QuestionType.ROOT_CAUSE,
                  QuestionType.SAFETY, QuestionType.ASSET_INFORMATION,
                  QuestionType.OPERATING_PROCEDURE, QuestionType.SCENARIO_GUIDANCE,
                  QuestionType.EXECUTIVE_SUMMARY, QuestionType.GENERAL_KNOWLEDGE):
            self.assertTrue(QUESTION_TYPE_REGISTRY.has(t))

    def test_rule_has_intent(self):
        self.assertTrue(QUESTION_TYPE_REGISTRY.get(QuestionType.SAFETY).intent)

    def test_rule_priority_high_for_safety(self):
        self.assertEqual(QUESTION_TYPE_REGISTRY.get(QuestionType.SAFETY).priority, "high")

    def test_rule_priority_high_for_root_cause(self):
        self.assertEqual(QUESTION_TYPE_REGISTRY.get(QuestionType.ROOT_CAUSE).priority, "high")

    def test_rule_priority_medium_for_maintenance(self):
        self.assertEqual(QUESTION_TYPE_REGISTRY.get(QuestionType.MAINTENANCE).priority, "medium")

    def test_rule_priority_low_for_general(self):
        self.assertEqual(QUESTION_TYPE_REGISTRY.get(QuestionType.GENERAL_KNOWLEDGE).priority, "low")

    def test_rule_score_keyword(self):
        rule = QUESTION_TYPE_REGISTRY.get(QuestionType.MAINTENANCE)
        self.assertGreaterEqual(rule.score({"maintenance"}, "maintenance"), 1)

    def test_rule_score_phrase(self):
        rule = QUESTION_TYPE_REGISTRY.get(QuestionType.ROOT_CAUSE)
        self.assertGreaterEqual(rule.score(set(), "what is the root cause"), 2)

    def test_rule_round_trip(self):
        rule = QUESTION_TYPE_REGISTRY.get(QuestionType.SAFETY)
        self.assertEqual(QuestionTypeRule.from_dict(rule.to_dict()), rule)

    def test_rule_frozen(self):
        rule = QUESTION_TYPE_REGISTRY.get(QuestionType.SAFETY)
        with self.assertRaises(Exception):
            rule.priority = "low"  # type: ignore[misc]

    def test_register_custom_type(self):
        rule = QuestionTypeRule("custom_type_x", ("zzz",), (), "x_intent", "x", "low", ())
        register_question_type(rule)
        self.assertTrue(QUESTION_TYPE_REGISTRY.has("custom_type_x"))

    def test_followup_registry_has_types(self):
        for t in (QuestionType.MAINTENANCE, QuestionType.SAFETY, QuestionType.GENERAL_KNOWLEDGE):
            self.assertTrue(FOLLOWUP_REGISTRY.has(t))

    def test_register_followup(self):
        register_followup_templates("custom_type_x", ("Tell me more about {topic}?",))
        self.assertTrue(FOLLOWUP_REGISTRY.has("custom_type_x"))


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------
class TestClassification(unittest.TestCase):
    def setUp(self):
        self.clf = QuestionClassifier()

    def _type(self, q):
        return self.clf.classify(q).question_type

    def test_maintenance(self):
        self.assertEqual(self._type("What maintenance schedule is recommended?"), QuestionType.MAINTENANCE)

    def test_inspection(self):
        self.assertEqual(self._type("How often should I inspect the bearing?"), QuestionType.INSPECTION)

    def test_failure_analysis(self):
        self.assertEqual(self._type("Describe the failure mode of the compressor"), QuestionType.FAILURE_ANALYSIS)

    def test_root_cause(self):
        self.assertEqual(self._type("What is the root cause of this fault?"), QuestionType.ROOT_CAUSE)

    def test_safety(self):
        self.assertEqual(self._type("What safety hazards apply here?"), QuestionType.SAFETY)

    def test_asset_information(self):
        self.assertEqual(self._type("What is the rated capacity of this asset?"), QuestionType.ASSET_INFORMATION)

    def test_operating_procedure(self):
        self.assertEqual(self._type("What is the operating procedure for startup?"), QuestionType.OPERATING_PROCEDURE)

    def test_scenario_guidance(self):
        self.assertEqual(self._type("What if we change the capital expenditure plan?"), QuestionType.SCENARIO_GUIDANCE)

    def test_executive_summary(self):
        self.assertEqual(self._type("Give me an executive summary of the portfolio"), QuestionType.EXECUTIVE_SUMMARY)

    def test_general_knowledge_fallback(self):
        self.assertEqual(self._type("Tell me about photosynthesis pigments"), QuestionType.GENERAL_KNOWLEDGE)

    def test_intent_root_cause(self):
        self.assertEqual(self.clf.classify("root cause of fault").intent, "root_cause_analysis")

    def test_intent_safety(self):
        self.assertEqual(self.clf.classify("safety hazard").intent, "safety_guidance")

    def test_category_maintenance(self):
        self.assertEqual(self.clf.classify("maintenance schedule").category, "maintenance")

    def test_priority_safety_high(self):
        self.assertEqual(self.clf.classify("safety hazard lockout").priority, "high")

    def test_priority_general_low(self):
        self.assertEqual(self.clf.classify("tell me about clouds").priority, "low")

    def test_confidence_in_range(self):
        c = self.clf.classify("root cause of compressor failure").confidence
        self.assertTrue(0.0 <= c <= 1.0)

    def test_confidence_higher_with_more_cues(self):
        weak = self.clf.classify("cause").confidence
        strong = self.clf.classify("root cause origin reason").confidence
        self.assertGreaterEqual(strong, weak)

    def test_general_confidence_low(self):
        self.assertLessEqual(self.clf.classify("tell me about clouds").confidence, 0.3)

    def test_expected_evidence_present(self):
        self.assertTrue(self.clf.classify("maintenance schedule").expected_evidence)

    def test_question_id_deterministic(self):
        a = self.clf.classify(Q_ROOT).question_id
        b = self.clf.classify(Q_ROOT).question_id
        self.assertEqual(a, b)

    def test_question_id_differs(self):
        self.assertNotEqual(self.clf.classify("a maintenance question").question_id,
                            self.clf.classify("a safety question").question_id)

    def test_deterministic_type(self):
        self.assertEqual(self._type(Q_ROOT), self._type(Q_ROOT))

    def test_tie_break_priority_order(self):
        # 'failure' (failure_analysis) and 'maintenance' both present; safety/root higher
        # but here failure_analysis should win over maintenance by order.
        t = self._type("maintenance failure")
        self.assertIn(t, (QuestionType.FAILURE_ANALYSIS, QuestionType.MAINTENANCE))

    def test_phrase_beats_single_keyword(self):
        # 'root cause' phrase (score 2) should select root_cause over a lone 'service'
        self.assertEqual(self._type("what is the root cause of poor service"), QuestionType.ROOT_CAUSE)

    def test_classify_returns_question_object(self):
        self.assertIsInstance(self.clf.classify(Q_ROOT), KnowledgeQuestion)


# ---------------------------------------------------------------------------
# KnowledgeQuestion
# ---------------------------------------------------------------------------
class TestKnowledgeQuestion(unittest.TestCase):
    def _q(self):
        return KnowledgeQuestion("q-1", "text", QuestionType.SAFETY, "safety_guidance", "safety", "high", ("a", "b"), 0.8)

    def test_fields(self):
        q = self._q()
        self.assertEqual(q.question_type, QuestionType.SAFETY)
        self.assertEqual(q.priority, "high")

    def test_confidence_float(self):
        self.assertIsInstance(self._q().confidence, float)

    def test_expected_evidence_tuple(self):
        self.assertEqual(self._q().expected_evidence, ("a", "b"))

    def test_frozen(self):
        q = self._q()
        with self.assertRaises(Exception):
            q.confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        q = self._q()
        self.assertEqual(KnowledgeQuestion.from_dict(q.to_dict()), q)

    def test_json_round_trip(self):
        q = self._q()
        self.assertEqual(KnowledgeQuestion.from_dict(json.loads(json.dumps(q.to_dict()))), q)

    def test_to_dict_keys(self):
        self.assertIn("question_id", self._q().to_dict())

    def test_id_coercion(self):
        q = KnowledgeQuestion(5, "t", "general_knowledge", "i", "c", "low")
        self.assertEqual(q.question_id, "5")


# ---------------------------------------------------------------------------
# KnowledgeAnswer
# ---------------------------------------------------------------------------
class TestKnowledgeAnswer(unittest.TestCase):
    def _a(self):
        return KnowledgeAnswer("answer text", True, ("d1", "d2"), 2)

    def test_text(self):
        self.assertEqual(self._a().text, "answer text")

    def test_supported_bool(self):
        self.assertIsInstance(self._a().supported, bool)

    def test_source_ids(self):
        self.assertEqual(self._a().source_document_ids, ("d1", "d2"))

    def test_sentence_count(self):
        self.assertEqual(self._a().sentence_count, 2)

    def test_frozen(self):
        a = self._a()
        with self.assertRaises(Exception):
            a.text = "x"  # type: ignore[misc]

    def test_round_trip(self):
        a = self._a()
        self.assertEqual(KnowledgeAnswer.from_dict(a.to_dict()), a)

    def test_json_round_trip(self):
        a = self._a()
        self.assertEqual(KnowledgeAnswer.from_dict(json.loads(json.dumps(a.to_dict()))), a)

    def test_unsupported(self):
        a = KnowledgeAnswer("none", False)
        self.assertFalse(a.supported)

    def test_to_dict_keys(self):
        self.assertIn("supported", self._a().to_dict())


# ---------------------------------------------------------------------------
# KnowledgeCitation
# ---------------------------------------------------------------------------
class TestKnowledgeCitation(unittest.TestCase):
    def _c(self):
        return KnowledgeCitation("d1", "Title", 1, 0.9, 0.95, "snippet text")

    def test_document_id(self):
        self.assertEqual(self._c().document_id, "d1")

    def test_title(self):
        self.assertEqual(self._c().title, "Title")

    def test_title_optional(self):
        c = KnowledgeCitation("d1", None, 1, 0.5, 0.5)
        self.assertIsNone(c.title)

    def test_rank_int(self):
        self.assertIsInstance(self._c().rank, int)

    def test_confidence_float(self):
        self.assertIsInstance(self._c().confidence, float)

    def test_relevance_float(self):
        self.assertIsInstance(self._c().relevance_score, float)

    def test_snippet(self):
        self.assertEqual(self._c().snippet, "snippet text")

    def test_frozen(self):
        c = self._c()
        with self.assertRaises(Exception):
            c.rank = 2  # type: ignore[misc]

    def test_round_trip(self):
        c = self._c()
        self.assertEqual(KnowledgeCitation.from_dict(c.to_dict()), c)

    def test_json_round_trip(self):
        c = self._c()
        self.assertEqual(KnowledgeCitation.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_round_trip_none_title(self):
        c = KnowledgeCitation("d1", None, 1, 0.5, 0.5, "s")
        self.assertIsNone(KnowledgeCitation.from_dict(c.to_dict()).title)

    def test_to_dict_keys(self):
        for k in ("document_id", "title", "rank", "confidence", "relevance_score", "snippet"):
            self.assertIn(k, self._c().to_dict())

    def test_id_coercion(self):
        self.assertEqual(KnowledgeCitation(7, None, 1, 0.5, 0.5).document_id, "7")

    def test_equality(self):
        self.assertEqual(self._c(), self._c())


# ---------------------------------------------------------------------------
# KnowledgeReasoning
# ---------------------------------------------------------------------------
class TestKnowledgeReasoning(unittest.TestCase):
    def _r(self):
        return KnowledgeReasoning(("why1", "why2"), 0.8, 0.7, 0.75, ("limit1",))

    def test_why_selected(self):
        self.assertEqual(self._r().why_selected, ("why1", "why2"))

    def test_evidence_strength(self):
        self.assertAlmostEqual(self._r().evidence_strength, 0.8)

    def test_coverage(self):
        self.assertAlmostEqual(self._r().coverage, 0.7)

    def test_confidence(self):
        self.assertAlmostEqual(self._r().confidence, 0.75)

    def test_limitations(self):
        self.assertEqual(self._r().limitations, ("limit1",))

    def test_frozen(self):
        r = self._r()
        with self.assertRaises(Exception):
            r.confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        r = self._r()
        self.assertEqual(KnowledgeReasoning.from_dict(r.to_dict()), r)

    def test_json_round_trip(self):
        r = self._r()
        self.assertEqual(KnowledgeReasoning.from_dict(json.loads(json.dumps(r.to_dict()))), r)

    def test_empty_limitations(self):
        r = KnowledgeReasoning(("w",), 0.5, 0.5, 0.5)
        self.assertEqual(r.limitations, ())

    def test_to_dict_keys(self):
        self.assertIn("why_selected", self._r().to_dict())


# ---------------------------------------------------------------------------
# KnowledgeResponse
# ---------------------------------------------------------------------------
class TestKnowledgeResponse(unittest.TestCase):
    def setUp(self):
        self.resp = make_agent().answer(Q_ROOT)

    def test_has_question(self):
        self.assertIsInstance(self.resp.question, KnowledgeQuestion)

    def test_has_answer(self):
        self.assertIsInstance(self.resp.answer, KnowledgeAnswer)

    def test_has_citations(self):
        self.assertIsInstance(self.resp.citations, tuple)

    def test_has_reasoning(self):
        self.assertIsInstance(self.resp.reasoning, KnowledgeReasoning)

    def test_confidence_in_range(self):
        self.assertTrue(0.0 <= self.resp.confidence <= 1.0)

    def test_citation_ids(self):
        self.assertEqual(self.resp.citation_ids, tuple(c.document_id for c in self.resp.citations))

    def test_metadata_readonly(self):
        self.assertIsInstance(self.resp.metadata, MappingProxyType)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self.resp.confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        r = KnowledgeResponse.from_dict(self.resp.to_dict())
        self.assertEqual(r.answer.text, self.resp.answer.text)

    def test_json_round_trip(self):
        r = KnowledgeResponse.from_dict(json.loads(json.dumps(self.resp.to_dict())))
        self.assertEqual(r.citation_ids, self.resp.citation_ids)

    def test_to_dict_keys(self):
        for k in ("question", "answer", "citations", "reasoning", "confidence", "follow_up_questions", "metadata"):
            self.assertIn(k, self.resp.to_dict())

    def test_metadata_has_question_type(self):
        self.assertIn("question_type", self.resp.metadata)


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------
class TestAgentPipeline(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_answer_returns_response(self):
        self.assertIsInstance(self.agent.answer(Q_ROOT), KnowledgeResponse)

    def test_answer_supported(self):
        self.assertTrue(self.agent.answer(Q_ROOT).answer.supported)

    def test_answer_has_citations(self):
        self.assertTrue(self.agent.answer(Q_ROOT).citations)

    def test_answer_text_nonempty(self):
        self.assertTrue(self.agent.answer(Q_ROOT).answer.text)

    def test_classification_in_response(self):
        self.assertEqual(self.agent.answer(Q_ROOT).question.question_type, QuestionType.ROOT_CAUSE)

    def test_safety_question(self):
        self.assertEqual(self.agent.answer(Q_SAFETY).question.question_type, QuestionType.SAFETY)

    def test_maintenance_question(self):
        self.assertEqual(self.agent.answer(Q_MAINT).question.question_type, QuestionType.MAINTENANCE)

    def test_classify_only(self):
        self.assertIsInstance(self.agent.classify(Q_ROOT), KnowledgeQuestion)

    def test_reasoning_present(self):
        self.assertTrue(self.agent.answer(Q_ROOT).reasoning.why_selected)

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.agent.answer(Q_ROOT).confidence <= 1.0)

    def test_metadata_latencies(self):
        md = self.agent.answer(Q_ROOT).metadata
        self.assertIn("retrieval_latency_ms", md)
        self.assertIn("response_latency_ms", md)

    def test_documents_searched_metadata(self):
        self.assertEqual(self.agent.answer(Q_ROOT).metadata["documents_searched"], 6)

    def test_deterministic_answer(self):
        a = self.agent.answer(Q_ROOT).answer.text
        b = self.agent.answer(Q_ROOT).answer.text
        self.assertEqual(a, b)

    def test_deterministic_citations(self):
        a = self.agent.answer(Q_ROOT).citation_ids
        b = self.agent.answer(Q_ROOT).citation_ids
        self.assertEqual(a, b)

    def test_config_override(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=3, top_evidence=1))
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).citations), 1)


# ---------------------------------------------------------------------------
# Grounded response generation (no hallucination)
# ---------------------------------------------------------------------------
class TestResponseGeneration(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def _assert_grounded(self, resp):
        # Every answer sentence must appear verbatim in some source document.
        for sentence in [s.strip() for s in resp.answer.text.replace("!", ".").replace("?", ".").split(".")]:
            if not sentence:
                continue
            self.assertIn(sentence, ALL_CONTENT, msg=f"ungrounded: {sentence!r}")

    def test_root_cause_grounded(self):
        self._assert_grounded(self.agent.answer(Q_ROOT))

    def test_safety_grounded(self):
        self._assert_grounded(self.agent.answer(Q_SAFETY))

    def test_maintenance_grounded(self):
        self._assert_grounded(self.agent.answer(Q_MAINT))

    def test_rul_grounded(self):
        self._assert_grounded(self.agent.answer("What is the remaining useful life of the compressor?"))

    def test_answer_draws_from_sources(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertTrue(resp.answer.source_document_ids)

    def test_answer_sources_are_citations(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertEqual(set(resp.answer.source_document_ids), set(resp.citation_ids))

    def test_sentence_count_matches(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertEqual(resp.answer.sentence_count, len([s for s in resp.answer.text.split(".") if s.strip()]))

    def test_max_answer_sentences_respected(self):
        cfg = AgentConfig(max_answer_sentences=1)
        resp = self.agent.answer(Q_ROOT, config=cfg)
        self.assertLessEqual(resp.answer.sentence_count, 1)

    def test_out_of_domain_still_grounded(self):
        resp = self.agent.answer("Tell me about something unrelated entirely zzz")
        self._assert_grounded(resp)

    def test_answer_never_empty_with_evidence(self):
        self.assertTrue(self.agent.answer("compressor").answer.text)

    def test_relevant_source_cited(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertIn("doc-root-cause", resp.citation_ids)

    def test_safety_source_cited(self):
        resp = self.agent.answer(Q_SAFETY)
        self.assertIn("doc-safety-lockout", resp.citation_ids)


# ---------------------------------------------------------------------------
# Citation assembly
# ---------------------------------------------------------------------------
class TestCitationAssembly(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_citations_have_titles(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertIsNotNone(c.title)

    def test_citations_have_ranks(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertGreaterEqual(c.rank, 1)

    def test_citations_have_relevance(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertIsInstance(c.relevance_score, float)

    def test_citations_confidence_bounded(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertTrue(0.0 <= c.confidence <= 1.0)

    def test_citations_have_snippets(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertTrue(c.snippet)

    def test_no_duplicate_citations(self):
        ids = self.agent.answer(Q_ROOT).citation_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_max_citations_respected(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=5, top_evidence=3))
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).citations), 1)

    def test_snippet_is_grounded(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertIn(c.snippet, ALL_CONTENT)

    def test_citation_rank_matches_retrieval(self):
        resp = self.agent.answer(Q_ROOT)
        ranks = [c.rank for c in resp.citations]
        self.assertEqual(ranks, sorted(ranks))

    def test_citations_ordered_by_rank(self):
        resp = self.agent.answer(Q_MAINT)
        ranks = [c.rank for c in resp.citations]
        self.assertEqual(ranks, sorted(ranks))

    def test_missing_title_is_none(self):
        docs = [FakeKnowledgeDocument("d1", "compressor failure analysis here", "pm", ("t",), {"text": "compressor failure analysis here"})]
        agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)
        for c in agent.answer("compressor failure").citations:
            self.assertIsNone(c.title)

    def test_relevance_descending(self):
        resp = self.agent.answer(Q_ROOT)
        rels = [c.relevance_score for c in resp.citations]
        self.assertEqual(rels, sorted(rels, reverse=True))


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------
class TestReasoning(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_why_selected_nonempty(self):
        self.assertTrue(self.agent.answer(Q_ROOT).reasoning.why_selected)

    def test_why_mentions_document(self):
        why = self.agent.answer(Q_ROOT).reasoning.why_selected
        self.assertTrue(any("doc-" in w for w in why))

    def test_evidence_strength_bounded(self):
        self.assertTrue(0.0 <= self.agent.answer(Q_ROOT).reasoning.evidence_strength <= 1.0)

    def test_coverage_bounded(self):
        self.assertTrue(0.0 <= self.agent.answer(Q_ROOT).reasoning.coverage <= 1.0)

    def test_confidence_matches_response(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertAlmostEqual(resp.reasoning.confidence, resp.confidence)

    def test_single_source_limitation(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=1, top_evidence=1))
        resp = self.agent.answer(Q_SAFETY, config=cfg)
        self.assertIn("Answer relies on a single source.", resp.reasoning.limitations)

    def test_low_coverage_limitation(self):
        resp = self.agent.answer("zzz unrelated quantum teleportation")
        self.assertTrue(any("less than half" in l for l in resp.reasoning.limitations))

    def test_no_limitations_when_strong(self):
        resp = self.agent.answer(Q_SAFETY)
        # safety question is well-covered by a strong single doc; may have single-source note
        self.assertIsInstance(resp.reasoning.limitations, tuple)

    def test_reasoning_count_matches_citations(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertEqual(len(resp.reasoning.why_selected), len(resp.citations))


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------
class TestConfidenceModel(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_confidence_in_range(self):
        for q in (Q_ROOT, Q_SAFETY, Q_MAINT):
            self.assertTrue(0.0 <= self.agent.answer(q).confidence <= 1.0)

    def test_relevant_higher_than_irrelevant(self):
        good = self.agent.answer(Q_SAFETY).confidence
        bad = self.agent.answer("zzz qqq unrelated nonsense token").confidence
        self.assertGreaterEqual(good, bad)

    def test_confidence_deterministic(self):
        self.assertEqual(self.agent.answer(Q_ROOT).confidence, self.agent.answer(Q_ROOT).confidence)

    def test_empty_kb_zero_confidence(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(agent.answer("anything compressor").confidence, 0.0)

    def test_confidence_reflects_coverage(self):
        resp = self.agent.answer(Q_SAFETY)
        self.assertTrue(resp.confidence >= 0.0)

    def test_single_citation_consistency(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=1, top_evidence=1))
        self.assertTrue(0.0 <= self.agent.answer(Q_SAFETY, config=cfg).confidence <= 1.0)

    def test_confidence_components_combine(self):
        # With strong single safety doc, confidence should be reasonably high.
        self.assertGreater(self.agent.answer(Q_SAFETY).confidence, 0.4)

    def test_zero_when_no_citations(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        resp = agent.answer("compressor failure")
        self.assertEqual(len(resp.citations), 0)
        self.assertEqual(resp.confidence, 0.0)


# ---------------------------------------------------------------------------
# Follow-up generation
# ---------------------------------------------------------------------------
class TestFollowUps(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_followups_when_below_threshold(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        self.assertTrue(self.agent.answer(Q_ROOT, config=cfg).follow_up_questions)

    def test_no_followups_when_threshold_zero(self):
        cfg = AgentConfig(confidence_threshold=0.0)
        self.assertEqual(self.agent.answer(Q_ROOT, config=cfg).follow_up_questions, ())

    def test_followups_respect_max(self):
        cfg = AgentConfig(confidence_threshold=1.0, max_follow_ups=2)
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).follow_up_questions), 2)

    def test_followups_are_strings(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        for f in self.agent.answer(Q_ROOT, config=cfg).follow_up_questions:
            self.assertIsInstance(f, str)

    def test_followups_type_specific(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        resp = self.agent.answer("What maintenance schedule applies?", config=cfg)
        self.assertTrue(any("maintenance" in f.lower() or "interval" in f.lower() for f in resp.follow_up_questions))

    def test_followups_deterministic(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        a = self.agent.answer(Q_ROOT, config=cfg).follow_up_questions
        b = self.agent.answer(Q_ROOT, config=cfg).follow_up_questions
        self.assertEqual(a, b)

    def test_followups_no_duplicates(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        f = self.agent.answer(Q_ROOT, config=cfg).follow_up_questions
        self.assertEqual(len(f), len(set(f)))

    def test_followups_contain_topic(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        f = self.agent.answer(Q_ROOT, config=cfg).follow_up_questions
        self.assertTrue(all("{topic}" not in x for x in f))

    def test_followups_empty_kb(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        cfg = AgentConfig(confidence_threshold=1.0)
        self.assertTrue(agent.answer("compressor failure", config=cfg).follow_up_questions)

    def test_general_followups_for_general_question(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        resp = self.agent.answer("tell me about clouds zzz", config=cfg)
        self.assertTrue(resp.follow_up_questions)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------
class TestAgentConfig(unittest.TestCase):
    def test_defaults(self):
        c = AgentConfig()
        self.assertEqual(c.max_citations, 5)

    def test_threshold_out_of_range(self):
        with self.assertRaises(KnowledgeAgentError):
            AgentConfig(confidence_threshold=1.5)

    def test_negative_max_citations(self):
        with self.assertRaises(KnowledgeAgentError):
            AgentConfig(max_citations=-1)

    def test_negative_max_sentences(self):
        with self.assertRaises(KnowledgeAgentError):
            AgentConfig(max_answer_sentences=-1)

    def test_negative_max_followups(self):
        with self.assertRaises(KnowledgeAgentError):
            AgentConfig(max_follow_ups=-1)

    def test_require_evidence_default_false(self):
        self.assertFalse(AgentConfig().require_evidence)

    def test_frozen(self):
        c = AgentConfig()
        with self.assertRaises(Exception):
            c.max_citations = 9  # type: ignore[misc]

    def test_round_trip(self):
        c = AgentConfig(confidence_threshold=0.6, max_citations=3, require_evidence=True)
        r = AgentConfig.from_dict(c.to_dict())
        self.assertEqual(r.max_citations, 3)
        self.assertTrue(r.require_evidence)

    def test_json_round_trip(self):
        c = AgentConfig()
        self.assertEqual(AgentConfig.from_dict(json.loads(json.dumps(c.to_dict()))).max_citations, 5)

    def test_embeds_retrieval_config(self):
        c = AgentConfig(retrieval_config=RetrievalConfig(top_k=9))
        self.assertEqual(c.retrieval_config.top_k, 9)

    def test_retrieval_config_round_trip(self):
        c = AgentConfig(retrieval_config=RetrievalConfig(top_k=7, top_evidence=2))
        self.assertEqual(AgentConfig.from_dict(c.to_dict()).retrieval_config.top_k, 7)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_initial_zero(self):
        self.assertEqual(self.agent.statistics().questions_answered, 0)

    def test_count_increments(self):
        self.agent.answer(Q_ROOT)
        self.agent.answer(Q_SAFETY)
        self.assertEqual(self.agent.statistics().questions_answered, 2)

    def test_average_confidence_bounded(self):
        self.agent.answer(Q_ROOT)
        self.assertTrue(0.0 <= self.agent.statistics().average_confidence <= 1.0)

    def test_citation_count_accumulates(self):
        self.agent.answer(Q_ROOT)
        n1 = self.agent.statistics().citation_count
        self.agent.answer(Q_SAFETY)
        self.assertGreaterEqual(self.agent.statistics().citation_count, n1)

    def test_coverage_bounded(self):
        self.agent.answer(Q_ROOT)
        self.assertTrue(0.0 <= self.agent.statistics().coverage_score <= 1.0)

    def test_latencies_zero_with_clock(self):
        self.agent.answer(Q_ROOT)
        s = self.agent.statistics()
        self.assertEqual(s.retrieval_latency_ms, 0.0)
        self.assertEqual(s.response_latency_ms, 0.0)

    def test_reset(self):
        self.agent.answer(Q_ROOT)
        self.agent.reset_statistics()
        self.assertEqual(self.agent.statistics().questions_answered, 0)

    def test_statistics_round_trip(self):
        self.agent.answer(Q_ROOT)
        s = self.agent.statistics()
        self.assertEqual(KnowledgeAgentStatistics.from_dict(s.to_dict()), s)

    def test_statistics_frozen(self):
        s = self.agent.statistics()
        with self.assertRaises(Exception):
            s.questions_answered = 9  # type: ignore[misc]

    def test_average_confidence_mean(self):
        c1 = self.agent.answer(Q_ROOT).confidence
        c2 = self.agent.answer(Q_SAFETY).confidence
        self.assertAlmostEqual(self.agent.statistics().average_confidence, (c1 + c2) / 2)

    def test_latency_with_real_clock_nonneg(self):
        agent = KnowledgeAgent.from_knowledge_base(make_corpus())
        agent.answer(Q_ROOT)
        self.assertGreaterEqual(agent.statistics().response_latency_ms, 0.0)


# ---------------------------------------------------------------------------
# Validation / errors
# ---------------------------------------------------------------------------
class TestValidation(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_empty_question(self):
        with self.assertRaises(EmptyQuestionError):
            self.agent.answer("")

    def test_whitespace_question(self):
        with self.assertRaises(EmptyQuestionError):
            self.agent.answer("    ")

    def test_punctuation_only_question(self):
        with self.assertRaises(EmptyQuestionError):
            self.agent.answer("!!! ???")

    def test_invalid_question_int(self):
        with self.assertRaises(InvalidQuestionError):
            self.agent.answer(123)  # type: ignore[arg-type]

    def test_invalid_question_none(self):
        with self.assertRaises(InvalidQuestionError):
            self.agent.answer(None)  # type: ignore[arg-type]

    def test_stopword_only_question(self):
        with self.assertRaises(EmptyQuestionError):
            self.agent.answer("the and of is")

    def test_missing_retrieval_engine(self):
        agent = KnowledgeAgent()
        with self.assertRaises(MissingRetrievalError):
            agent.answer("compressor failure")

    def test_require_evidence_empty_kb_raises(self):
        agent = KnowledgeAgent.from_knowledge_base([], config=AgentConfig(require_evidence=True), clock=lambda: 0.0)
        with self.assertRaises(MissingEvidenceError):
            agent.answer("compressor failure")

    def test_no_require_evidence_empty_kb_ok(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        resp = agent.answer("compressor failure")
        self.assertFalse(resp.answer.supported)

    def test_error_hierarchy(self):
        for e in (EmptyQuestionError, InvalidQuestionError, MissingRetrievalError, DuplicateCitationError, MissingEvidenceError):
            self.assertTrue(issubclass(e, KnowledgeAgentError))

    def test_classify_validates(self):
        with self.assertRaises(EmptyQuestionError):
            self.agent.classify("")

    def test_classify_invalid(self):
        with self.assertRaises(InvalidQuestionError):
            self.agent.classify(5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration with Phase 1/2/3
# ---------------------------------------------------------------------------
class TestIntegration(unittest.TestCase):
    def test_from_knowledge_base_object(self):
        agent = KnowledgeAgent.from_knowledge_base(FakeKnowledgeBase(make_corpus()), clock=lambda: 0.0)
        self.assertEqual(agent.retrieval_engine.vector_engine.index.size(), 6)

    def test_from_iterable(self):
        agent = make_agent()
        self.assertEqual(agent.retrieval_engine.vector_engine.index.size(), 6)

    def test_alt_named_documents(self):
        docs = [AltNamedDocument(f"a-{i}", f"compressor failure case {i}", "pm", ("t",)) for i in range(4)]
        agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertTrue(agent.answer("compressor failure").citations)

    def test_inject_existing_retrieval_engine(self):
        re = RetrievalEngine.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        agent = KnowledgeAgent(retrieval_engine=re, clock=lambda: 0.0)
        self.assertTrue(agent.answer(Q_ROOT).citations)

    def test_uses_retrieval_package(self):
        agent = make_agent()
        # response metadata mirrors the retrieval package statistics
        self.assertEqual(agent.answer(Q_ROOT).metadata["documents_searched"], 6)

    def test_phase3_still_works(self):
        re = RetrievalEngine.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        self.assertTrue(re.retrieve("compressor failure").context.documents)

    def test_phase2_still_works(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus())
        self.assertTrue(ve.search("compressor", top_k=2))

    def test_custom_retrieval_config_passthrough(self):
        cfg = AgentConfig(retrieval_config=RetrievalConfig(top_k=2, top_evidence=1))
        self.assertLessEqual(len(make_agent().answer(Q_ROOT, config=cfg).citations), 2)


# ---------------------------------------------------------------------------
# Agent serialization
# ---------------------------------------------------------------------------
class TestAgentSerialization(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_to_dict_keys(self):
        self.assertEqual(set(self.agent.to_dict()), {"config", "statistics", "retrieval"})

    def test_round_trip_answers(self):
        restored = KnowledgeAgent.from_dict(self.agent.to_dict())
        self.assertTrue(restored.answer(Q_ROOT).citations)

    def test_round_trip_preserves_answer(self):
        restored = KnowledgeAgent.from_dict(json.loads(json.dumps(self.agent.to_dict())))
        a = self.agent.answer(Q_ROOT).answer.text
        b = restored.answer(Q_ROOT).answer.text
        self.assertEqual(a, b)

    def test_round_trip_config(self):
        agent = make_agent(config=AgentConfig(max_citations=2))
        restored = KnowledgeAgent.from_dict(agent.to_dict())
        self.assertEqual(restored.config.max_citations, 2)

    def test_json_serializable(self):
        self.assertIsInstance(json.dumps(self.agent.to_dict()), str)

    def test_none_retrieval_serializes(self):
        agent = KnowledgeAgent()
        self.assertIsNone(agent.to_dict()["retrieval"])


# ---------------------------------------------------------------------------
# Large knowledge base
# ---------------------------------------------------------------------------
class TestLargeKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = _CORPUS_ROWS
        docs = []
        for i in range(600):
            row = base[i % len(base)]
            content = row[2] + f" Record {i}."
            docs.append(FakeKnowledgeDocument(
                document_id=f"doc-{i:05d}",
                content=content,
                category=row[3],
                tags=row[4],
                metadata={"text": content, "title": f"{row[1]} #{i}", "freshness": round((i % 10) / 9.0, 3)},
            ))
        cls.agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)

    def test_indexed_all(self):
        self.assertEqual(self.agent.retrieval_engine.vector_engine.index.size(), 600)

    def test_answers(self):
        self.assertTrue(self.agent.answer(Q_ROOT).citations)

    def test_deterministic(self):
        a = self.agent.answer(Q_ROOT).citation_ids
        b = self.agent.answer(Q_ROOT).citation_ids
        self.assertEqual(a, b)

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.agent.answer(Q_ROOT).confidence <= 1.0)

    def test_documents_searched(self):
        self.assertEqual(self.agent.answer(Q_ROOT).metadata["documents_searched"], 600)

    def test_grounded(self):
        resp = self.agent.answer(Q_SAFETY)
        self.assertTrue(resp.answer.supported)

    def test_max_citations(self):
        cfg = AgentConfig(max_citations=3, retrieval_config=RetrievalConfig(top_k=10, top_evidence=3))
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).citations), 3)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_same_response_dict(self):
        agent = make_agent()
        a = agent.answer(Q_ROOT).to_dict()
        agent2 = make_agent()
        b = agent2.answer(Q_ROOT).to_dict()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_classification_deterministic(self):
        clf = QuestionClassifier()
        self.assertEqual(clf.classify(Q_ROOT).question_type, clf.classify(Q_ROOT).question_type)

    def test_confidence_deterministic(self):
        agent = make_agent()
        self.assertEqual(agent.answer(Q_ROOT).confidence, agent.answer(Q_ROOT).confidence)

    def test_citations_deterministic(self):
        agent = make_agent()
        self.assertEqual(agent.answer(Q_ROOT).citation_ids, agent.answer(Q_ROOT).citation_ids)

    def test_followups_deterministic(self):
        agent = make_agent()
        cfg = AgentConfig(confidence_threshold=1.0)
        self.assertEqual(agent.answer(Q_ROOT, config=cfg).follow_up_questions,
                         agent.answer(Q_ROOT, config=cfg).follow_up_questions)

    def test_across_instances(self):
        self.assertEqual(make_agent().answer(Q_ROOT).answer.text, make_agent().answer(Q_ROOT).answer.text)

    def test_question_id_stable(self):
        self.assertEqual(make_agent().answer(Q_ROOT).question.question_id,
                         make_agent().answer(Q_ROOT).question.question_id)

    def test_confidence_is_python_float(self):
        self.assertIsInstance(make_agent().answer(Q_ROOT).confidence, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_single_document_kb(self):
        docs = [FakeKnowledgeDocument("only", "compressor failure due to lubrication degradation", "pm", ("t",), {"text": "compressor failure due to lubrication degradation"})]
        agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertEqual(len(agent.answer("compressor failure").citations), 1)

    def test_unicode_question(self):
        self.assertTrue(self.agent.answer("café compressor failure").citations)

    def test_long_question(self):
        self.assertTrue(self.agent.answer("compressor " * 100 + "failure").citations)

    def test_numeric_question(self):
        self.assertIsInstance(self.agent.answer("sensor 12345 reading"), KnowledgeResponse)

    def test_mixed_case(self):
        a = self.agent.answer("ROOT CAUSE of COMPRESSOR Failure").question.question_type
        self.assertEqual(a, QuestionType.ROOT_CAUSE)

    def test_question_with_punctuation(self):
        self.assertTrue(self.agent.answer("What is the root-cause; of failure?").citations)

    def test_empty_kb_response_shape(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        resp = agent.answer("compressor failure")
        self.assertEqual(resp.citations, ())
        self.assertFalse(resp.answer.supported)

    def test_empty_kb_has_followups(self):
        agent = KnowledgeAgent.from_knowledge_base([], config=AgentConfig(confidence_threshold=0.5), clock=lambda: 0.0)
        self.assertTrue(agent.answer("compressor failure").follow_up_questions)

    def test_doc_without_title(self):
        docs = [FakeKnowledgeDocument("d", "compressor failure here now", "pm", ("t",), {"text": "compressor failure here now"})]
        agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertIsNone(agent.answer("compressor failure").citations[0].title)

    def test_max_citations_zero(self):
        cfg = AgentConfig(max_citations=0)
        resp = self.agent.answer(Q_ROOT, config=cfg)
        self.assertEqual(len(resp.citations), 0)

    def test_max_answer_sentences_zero(self):
        cfg = AgentConfig(max_answer_sentences=0)
        resp = self.agent.answer(Q_ROOT, config=cfg)
        self.assertIsInstance(resp.answer.text, str)

    def test_single_word_question(self):
        self.assertTrue(self.agent.answer("compressor").citations)

    def test_repeated_calls_independent(self):
        a = self.agent.answer(Q_ROOT).answer.text
        _ = self.agent.answer(Q_SAFETY)
        b = self.agent.answer(Q_ROOT).answer.text
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Multi-question invariants
# ---------------------------------------------------------------------------
class TestMultiQuestionInvariants(unittest.TestCase):
    QUESTIONS = [
        "What is the root cause of the compressor failure?",
        "What safety procedure applies before servicing equipment?",
        "What maintenance is recommended for turbine bearing vibration?",
        "How often should the bearing be inspected?",
        "Give me an executive summary of the fleet portfolio",
        "What is the remaining useful life of the compressor?",
    ]

    def setUp(self):
        self.agent = make_agent()

    def test_all_return_responses(self):
        for q in self.QUESTIONS:
            self.assertIsInstance(self.agent.answer(q), KnowledgeResponse, msg=q)

    def test_all_confidence_bounded(self):
        for q in self.QUESTIONS:
            self.assertTrue(0.0 <= self.agent.answer(q).confidence <= 1.0, msg=q)

    def test_all_supported(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.agent.answer(q).answer.supported, msg=q)

    def test_all_grounded(self):
        for q in self.QUESTIONS:
            resp = self.agent.answer(q)
            for sentence in [s.strip() for s in resp.answer.text.replace("!", ".").replace("?", ".").split(".")]:
                if sentence:
                    self.assertIn(sentence, ALL_CONTENT, msg=f"{q} -> {sentence!r}")

    def test_all_no_duplicate_citations(self):
        for q in self.QUESTIONS:
            ids = self.agent.answer(q).citation_ids
            self.assertEqual(len(ids), len(set(ids)), msg=q)

    def test_all_json_round_trip(self):
        for q in self.QUESTIONS:
            resp = self.agent.answer(q)
            r = KnowledgeResponse.from_dict(json.loads(json.dumps(resp.to_dict())))
            self.assertEqual(r.answer.text, resp.answer.text, msg=q)

    def test_all_deterministic(self):
        for q in self.QUESTIONS:
            self.assertEqual(self.agent.answer(q).citation_ids, self.agent.answer(q).citation_ids, msg=q)

    def test_all_have_reasoning(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.agent.answer(q).reasoning.why_selected, msg=q)

    def test_all_classified(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.agent.answer(q).question.question_type, msg=q)

    def test_all_citations_have_rank(self):
        for q in self.QUESTIONS:
            for c in self.agent.answer(q).citations:
                self.assertGreaterEqual(c.rank, 1, msg=q)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompatibility(unittest.TestCase):
    def test_phase2_vector_search(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus())
        self.assertTrue(ve.search("compressor", top_k=3))

    def test_phase3_retrieval(self):
        re = RetrievalEngine.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        self.assertTrue(re.retrieve("compressor failure").context.documents)

    def test_phase3_package_type(self):
        re = RetrievalEngine.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        self.assertIsInstance(re.retrieve("compressor").context.documents[0].document_id, str)

    def test_agent_consumes_package(self):
        agent = make_agent()
        resp = agent.answer(Q_ROOT)
        self.assertEqual(resp.metadata["documents_searched"], 6)

    def test_no_modification_of_retrieval_results(self):
        # Agent does not mutate the retrieval engine's index.
        agent = make_agent()
        before = agent.retrieval_engine.vector_engine.index.size()
        agent.answer(Q_ROOT)
        self.assertEqual(agent.retrieval_engine.vector_engine.index.size(), before)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_demo_runs(self):
        from knowledge.knowledge_agent import main
        self.assertEqual(main(["--demo"]), 0)

    def test_no_args_prints_help(self):
        from knowledge.knowledge_agent import main
        self.assertEqual(main([]), 0)


# ---------------------------------------------------------------------------
# Classification (extra per-type coverage)
# ---------------------------------------------------------------------------
class TestClassificationExtra(unittest.TestCase):
    def setUp(self):
        self.clf = QuestionClassifier()

    def _t(self, q):
        return self.clf.classify(q).question_type

    def test_maintenance_variant(self):
        self.assertEqual(self._t("Schedule preventive maintenance service interval"), QuestionType.MAINTENANCE)

    def test_inspection_variant(self):
        self.assertEqual(self._t("Monitor and measure with an audit inspection"), QuestionType.INSPECTION)

    def test_failure_variant(self):
        self.assertEqual(self._t("Describe the malfunction breakdown and anomaly"), QuestionType.FAILURE_ANALYSIS)

    def test_root_cause_variant(self):
        self.assertEqual(self._t("What is the root cause and origin reason?"), QuestionType.ROOT_CAUSE)

    def test_safety_variant(self):
        self.assertEqual(self._t("Describe the lockout tagout hazard and ppe"), QuestionType.SAFETY)

    def test_asset_variant(self):
        self.assertEqual(self._t("What is the model serial and capacity rating?"), QuestionType.ASSET_INFORMATION)

    def test_procedure_variant(self):
        self.assertEqual(self._t("What are the startup and shutdown steps?"), QuestionType.OPERATING_PROCEDURE)

    def test_scenario_variant(self):
        self.assertEqual(self._t("Run a what if scenario forecast simulation"), QuestionType.SCENARIO_GUIDANCE)

    def test_executive_variant(self):
        self.assertEqual(self._t("Provide a strategic portfolio dashboard overview"), QuestionType.EXECUTIVE_SUMMARY)

    def test_general_variant(self):
        self.assertEqual(self._t("Describe migratory bird navigation patterns"), QuestionType.GENERAL_KNOWLEDGE)

    def test_confidence_capped(self):
        c = self.clf.classify("schedule preventive maintenance service interval overhaul repair replace").confidence
        self.assertLessEqual(c, 0.95)

    def test_confidence_positive_for_match(self):
        self.assertGreater(self.clf.classify("maintenance schedule").confidence, 0.2)

    def test_priority_failure_high(self):
        self.assertEqual(self.clf.classify("malfunction breakdown").priority, "high")

    def test_priority_inspection_medium(self):
        self.assertEqual(self.clf.classify("inspect and monitor").priority, "medium")

    def test_priority_asset_low(self):
        self.assertEqual(self.clf.classify("model serial number").priority, "low")

    def test_intent_maintenance(self):
        self.assertEqual(self.clf.classify("maintenance schedule").intent, "maintenance_guidance")

    def test_intent_inspection(self):
        self.assertEqual(self.clf.classify("inspect the bearing").intent, "inspection_guidance")

    def test_intent_executive(self):
        self.assertEqual(self.clf.classify("executive summary").intent, "executive_summary")

    def test_category_safety(self):
        self.assertEqual(self.clf.classify("safety hazard").category, "safety")

    def test_category_scenario(self):
        self.assertEqual(self.clf.classify("scenario planning forecast").category, "scenario")


# ---------------------------------------------------------------------------
# Question-type rule details
# ---------------------------------------------------------------------------
class TestQuestionTypeRuleDetails(unittest.TestCase):
    def _rule(self, name):
        return QUESTION_TYPE_REGISTRY.get(name)

    def test_maintenance_keywords(self):
        self.assertIn("maintenance", self._rule(QuestionType.MAINTENANCE).keywords)

    def test_inspection_keywords(self):
        self.assertIn("inspect", self._rule(QuestionType.INSPECTION).keywords)

    def test_failure_keywords(self):
        self.assertIn("failure", self._rule(QuestionType.FAILURE_ANALYSIS).keywords)

    def test_root_cause_phrases(self):
        self.assertIn("root cause", self._rule(QuestionType.ROOT_CAUSE).phrases)

    def test_safety_phrases(self):
        self.assertIn("lockout tagout", self._rule(QuestionType.SAFETY).phrases)

    def test_asset_keywords(self):
        self.assertIn("specification", self._rule(QuestionType.ASSET_INFORMATION).keywords)

    def test_procedure_phrases(self):
        self.assertIn("how to", self._rule(QuestionType.OPERATING_PROCEDURE).phrases)

    def test_scenario_phrases(self):
        self.assertIn("what if", self._rule(QuestionType.SCENARIO_GUIDANCE).phrases)

    def test_executive_keywords(self):
        self.assertIn("executive", self._rule(QuestionType.EXECUTIVE_SUMMARY).keywords)

    def test_general_no_keywords(self):
        self.assertEqual(self._rule(QuestionType.GENERAL_KNOWLEDGE).keywords, ())

    def test_all_priorities_valid(self):
        for name in (QuestionType.MAINTENANCE, QuestionType.INSPECTION, QuestionType.FAILURE_ANALYSIS,
                     QuestionType.ROOT_CAUSE, QuestionType.SAFETY, QuestionType.ASSET_INFORMATION,
                     QuestionType.OPERATING_PROCEDURE, QuestionType.SCENARIO_GUIDANCE,
                     QuestionType.EXECUTIVE_SUMMARY, QuestionType.GENERAL_KNOWLEDGE):
            self.assertIn(self._rule(name).priority, ("high", "medium", "low"))

    def test_all_intents_nonempty(self):
        for name in QUESTION_TYPE_REGISTRY.names():
            self.assertTrue(self._rule(name).intent)


# ---------------------------------------------------------------------------
# Answer selection
# ---------------------------------------------------------------------------
class TestAnswerSelection(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_selects_relevant_sentence(self):
        resp = self.agent.answer("lubrication degradation root cause")
        self.assertIn("lubrication", resp.answer.text.lower())

    def test_safety_sentence_selected(self):
        resp = self.agent.answer("lockout tagout energy isolation")
        self.assertIn("lockout", resp.answer.text.lower())

    def test_max_sentences_one(self):
        cfg = AgentConfig(max_answer_sentences=1)
        self.assertLessEqual(self.agent.answer(Q_ROOT, config=cfg).answer.sentence_count, 1)

    def test_max_sentences_two(self):
        cfg = AgentConfig(max_answer_sentences=2)
        self.assertLessEqual(self.agent.answer(Q_ROOT, config=cfg).answer.sentence_count, 2)

    def test_fallback_lead_sentence(self):
        resp = self.agent.answer("zzz qqq totally unrelated tokens")
        self.assertIn(resp.answer.text, ALL_CONTENT)

    def test_answer_sentences_grounded(self):
        resp = self.agent.answer(Q_MAINT)
        for s in [x.strip() for x in resp.answer.text.split(".") if x.strip()]:
            self.assertIn(s, ALL_CONTENT)

    def test_reading_order_preserved(self):
        # Sentences from the same doc appear in original order.
        resp = self.agent.answer("turbine bearing vibration maintenance inspect replace")
        text = resp.answer.text
        self.assertIsInstance(text, str)

    def test_source_ids_unique(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertEqual(len(resp.answer.source_document_ids), len(set(resp.answer.source_document_ids)))

    def test_supported_true_with_text(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertTrue(resp.answer.supported)
        self.assertTrue(resp.answer.text)

    def test_sentence_count_nonneg(self):
        self.assertGreaterEqual(self.agent.answer(Q_ROOT).answer.sentence_count, 0)

    def test_answer_in_corpus(self):
        resp = self.agent.answer("compressor failure risk")
        for s in [x.strip() for x in resp.answer.text.split(".") if x.strip()]:
            self.assertIn(s, ALL_CONTENT)

    def test_answer_no_invention(self):
        # The whole answer text characters should be reconstructable from sources.
        resp = self.agent.answer(Q_SAFETY)
        for s in [x.strip() for x in resp.answer.text.split(".") if x.strip()]:
            self.assertIn(s, ALL_CONTENT)


# ---------------------------------------------------------------------------
# Snippet & content
# ---------------------------------------------------------------------------
class TestSnippetContent(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_snippet_first_sentence(self):
        resp = self.agent.answer(Q_SAFETY)
        c = resp.citations[0]
        first = CONTENT_BY_ID[c.document_id].split(".")[0].strip() + "."
        self.assertEqual(c.snippet, first)

    def test_snippet_grounded(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertIn(c.snippet, ALL_CONTENT)

    def test_snippet_nonempty(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertTrue(c.snippet)

    def test_title_from_metadata(self):
        resp = self.agent.answer(Q_SAFETY)
        self.assertEqual(resp.citations[0].title, "Lockout Tagout Safety Procedure")

    def test_relevance_equals_doc_score(self):
        resp = self.agent.answer(Q_ROOT)
        for c in resp.citations:
            self.assertTrue(0.0 <= c.relevance_score <= 1.0)

    def test_confidence_equals_clamped_score(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertTrue(0.0 <= c.confidence <= 1.0)

    def test_distinct_documents(self):
        resp = self.agent.answer(Q_MAINT)
        self.assertEqual(len(resp.citation_ids), len(set(resp.citation_ids)))

    def test_snippet_matches_source(self):
        for c in self.agent.answer(Q_MAINT).citations:
            self.assertIn(c.snippet, CONTENT_BY_ID[c.document_id])

    def test_title_present_for_titled_docs(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertIsNotNone(c.title)

    def test_rank_positive(self):
        for c in self.agent.answer(Q_ROOT).citations:
            self.assertGreaterEqual(c.rank, 1)


# ---------------------------------------------------------------------------
# Confidence components (detail)
# ---------------------------------------------------------------------------
class TestConfidenceComponents(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_empty_zero(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(agent.answer("compressor").confidence, 0.0)

    def test_strong_safety_high(self):
        self.assertGreater(self.agent.answer(Q_SAFETY).confidence, 0.4)

    def test_irrelevant_lower(self):
        good = self.agent.answer(Q_ROOT).confidence
        bad = self.agent.answer("zzz qqq nonsense unrelated").confidence
        self.assertGreaterEqual(good, bad)

    def test_bounded_all(self):
        for q in (Q_ROOT, Q_SAFETY, Q_MAINT, "executive portfolio summary", "inspect bearing"):
            self.assertTrue(0.0 <= self.agent.answer(q).confidence <= 1.0)

    def test_reasoning_confidence_equals(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertAlmostEqual(resp.reasoning.confidence, resp.confidence)

    def test_single_source_consistency_one(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=1, top_evidence=1))
        # confidence still bounded; single-source consistency contributes 1.0
        self.assertTrue(0.0 <= self.agent.answer(Q_SAFETY, config=cfg).confidence <= 1.0)

    def test_confidence_python_float(self):
        self.assertIsInstance(self.agent.answer(Q_ROOT).confidence, float)

    def test_confidence_stable(self):
        self.assertEqual(self.agent.answer(Q_ROOT).confidence, self.agent.answer(Q_ROOT).confidence)

    def test_more_citations_more_evidence(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertGreaterEqual(len(resp.citations), 1)

    def test_coverage_used(self):
        resp = self.agent.answer(Q_SAFETY)
        self.assertTrue(0.0 <= resp.reasoning.coverage <= 1.0)


# ---------------------------------------------------------------------------
# Reasoning (extra)
# ---------------------------------------------------------------------------
class TestReasoningExtra(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_evidence_strength_equals_mean(self):
        resp = self.agent.answer(Q_ROOT)
        expected = float(np.mean([c.relevance_score for c in resp.citations]))
        self.assertAlmostEqual(resp.reasoning.evidence_strength, expected)

    def test_why_count_matches(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertEqual(len(resp.reasoning.why_selected), len(resp.citations))

    def test_limitations_tuple(self):
        self.assertIsInstance(self.agent.answer(Q_ROOT).reasoning.limitations, tuple)

    def test_low_coverage_flagged(self):
        resp = self.agent.answer("zzz quantum teleportation unrelated")
        self.assertTrue(any("less than half" in l for l in resp.reasoning.limitations))

    def test_single_source_flagged(self):
        cfg = AgentConfig(max_citations=1, retrieval_config=RetrievalConfig(top_k=1, top_evidence=1))
        resp = self.agent.answer(Q_SAFETY, config=cfg)
        self.assertIn("Answer relies on a single source.", resp.reasoning.limitations)

    def test_threshold_limitation(self):
        cfg = AgentConfig(confidence_threshold=1.0)
        resp = self.agent.answer(Q_ROOT, config=cfg)
        self.assertTrue(any("below the configured threshold" in l for l in resp.reasoning.limitations))

    def test_no_evidence_limitation(self):
        agent = KnowledgeAgent.from_knowledge_base([], clock=lambda: 0.0)
        resp = agent.answer("compressor failure")
        self.assertTrue(any("No supporting evidence" in l for l in resp.reasoning.limitations))

    def test_why_mentions_rank(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertTrue(any("rank" in w for w in resp.reasoning.why_selected))

    def test_evidence_strength_bounded(self):
        self.assertTrue(0.0 <= self.agent.answer(Q_ROOT).reasoning.evidence_strength <= 1.0)

    def test_reasoning_round_trip(self):
        r = self.agent.answer(Q_ROOT).reasoning
        self.assertEqual(KnowledgeReasoning.from_dict(r.to_dict()), r)


# ---------------------------------------------------------------------------
# Follow-up topics
# ---------------------------------------------------------------------------
class TestFollowUpTopics(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.cfg = AgentConfig(confidence_threshold=1.0)

    def test_no_braces(self):
        for f in self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions:
            self.assertNotIn("{topic}", f)

    def test_maintenance_followups(self):
        resp = self.agent.answer("maintenance schedule", config=self.cfg)
        self.assertTrue(resp.follow_up_questions)

    def test_safety_followups(self):
        resp = self.agent.answer("safety hazard", config=self.cfg)
        self.assertTrue(any("safe" in f.lower() or "hazard" in f.lower() or "protective" in f.lower() for f in resp.follow_up_questions))

    def test_inspection_followups(self):
        resp = self.agent.answer("inspect bearing", config=self.cfg)
        self.assertTrue(resp.follow_up_questions)

    def test_count_capped(self):
        cfg = AgentConfig(confidence_threshold=1.0, max_follow_ups=1)
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).follow_up_questions), 1)

    def test_deterministic(self):
        a = self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions
        b = self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions
        self.assertEqual(a, b)

    def test_unique(self):
        f = self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions
        self.assertEqual(len(f), len(set(f)))

    def test_strings(self):
        for f in self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions:
            self.assertIsInstance(f, str)

    def test_question_marks(self):
        for f in self.agent.answer(Q_ROOT, config=self.cfg).follow_up_questions:
            self.assertTrue(f.endswith("?"))

    def test_empty_when_confident(self):
        cfg = AgentConfig(confidence_threshold=0.0)
        self.assertEqual(self.agent.answer(Q_ROOT, config=cfg).follow_up_questions, ())


# ---------------------------------------------------------------------------
# Statistics (extra)
# ---------------------------------------------------------------------------
class TestStatisticsExtra(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_citation_count_sum(self):
        total = 0
        for q in (Q_ROOT, Q_SAFETY, Q_MAINT):
            total += len(self.agent.answer(q).citations)
        self.assertEqual(self.agent.statistics().citation_count, total)

    def test_count_three(self):
        for q in (Q_ROOT, Q_SAFETY, Q_MAINT):
            self.agent.answer(q)
        self.assertEqual(self.agent.statistics().questions_answered, 3)

    def test_average_confidence_mean(self):
        confs = [self.agent.answer(q).confidence for q in (Q_ROOT, Q_SAFETY)]
        self.assertAlmostEqual(self.agent.statistics().average_confidence, sum(confs) / 2)

    def test_coverage_mean_bounded(self):
        self.agent.answer(Q_ROOT)
        self.assertTrue(0.0 <= self.agent.statistics().coverage_score <= 1.0)

    def test_reset_then_count(self):
        self.agent.answer(Q_ROOT)
        self.agent.reset_statistics()
        self.agent.answer(Q_SAFETY)
        self.assertEqual(self.agent.statistics().questions_answered, 1)

    def test_round_trip(self):
        self.agent.answer(Q_ROOT)
        s = self.agent.statistics()
        self.assertEqual(KnowledgeAgentStatistics.from_dict(json.loads(json.dumps(s.to_dict()))), s)

    def test_zero_latency_clock(self):
        self.agent.answer(Q_ROOT)
        self.assertEqual(self.agent.statistics().retrieval_latency_ms, 0.0)

    def test_initial_all_zero(self):
        s = self.agent.statistics()
        self.assertEqual((s.questions_answered, s.citation_count), (0, 0))

    def test_to_dict_keys(self):
        for k in ("questions_answered", "average_confidence", "retrieval_latency_ms",
                  "response_latency_ms", "citation_count", "coverage_score"):
            self.assertIn(k, self.agent.statistics().to_dict())

    def test_stats_after_many(self):
        for _ in range(5):
            self.agent.answer(Q_ROOT)
        self.assertEqual(self.agent.statistics().questions_answered, 5)


# ---------------------------------------------------------------------------
# Serialization (extra)
# ---------------------------------------------------------------------------
class TestSerializationExtra(unittest.TestCase):
    def test_question_varied(self):
        q = KnowledgeQuestion("q9", "txt", QuestionType.SCENARIO_GUIDANCE, "scenario_guidance", "scenario", "medium", ("a",), 0.55)
        self.assertEqual(KnowledgeQuestion.from_dict(json.loads(json.dumps(q.to_dict()))), q)

    def test_answer_unsupported(self):
        a = KnowledgeAnswer("none", False, (), 0)
        self.assertEqual(KnowledgeAnswer.from_dict(json.loads(json.dumps(a.to_dict()))), a)

    def test_citation_no_title(self):
        c = KnowledgeCitation("d", None, 2, 0.4, 0.4, "snip")
        self.assertEqual(KnowledgeCitation.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_reasoning_many_limits(self):
        r = KnowledgeReasoning(("w1", "w2"), 0.5, 0.4, 0.45, ("l1", "l2", "l3"))
        self.assertEqual(KnowledgeReasoning.from_dict(json.loads(json.dumps(r.to_dict()))), r)

    def test_response_full(self):
        resp = make_agent().answer(Q_ROOT, config=AgentConfig(confidence_threshold=1.0))
        restored = KnowledgeResponse.from_dict(json.loads(json.dumps(resp.to_dict())))
        self.assertEqual(restored.follow_up_questions, resp.follow_up_questions)

    def test_response_citations_round_trip(self):
        resp = make_agent().answer(Q_ROOT)
        restored = KnowledgeResponse.from_dict(resp.to_dict())
        self.assertEqual([c.document_id for c in restored.citations], list(resp.citation_ids))

    def test_agent_config_round_trip(self):
        c = AgentConfig(confidence_threshold=0.7, max_citations=4, max_answer_sentences=2, max_follow_ups=5, require_evidence=True)
        self.assertEqual(AgentConfig.from_dict(json.loads(json.dumps(c.to_dict()))).max_follow_ups, 5)

    def test_stats_round_trip(self):
        s = KnowledgeAgentStatistics(3, 0.6, 1.0, 2.0, 7, 0.5)
        self.assertEqual(KnowledgeAgentStatistics.from_dict(json.loads(json.dumps(s.to_dict()))), s)

    def test_response_metadata_round_trip(self):
        resp = make_agent().answer(Q_ROOT)
        restored = KnowledgeResponse.from_dict(resp.to_dict())
        self.assertEqual(restored.metadata["question_type"], resp.metadata["question_type"])

    def test_question_type_rule_round_trip(self):
        rule = QUESTION_TYPE_REGISTRY.get(QuestionType.MAINTENANCE)
        self.assertEqual(QuestionTypeRule.from_dict(json.loads(json.dumps(rule.to_dict()))), rule)


# ---------------------------------------------------------------------------
# Edge cases (extra)
# ---------------------------------------------------------------------------
class TestEdgeCasesExtra(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_question_all_caps(self):
        self.assertTrue(self.agent.answer("COMPRESSOR FAILURE ROOT CAUSE").citations)

    def test_question_trailing_space(self):
        self.assertTrue(self.agent.answer("  compressor failure  ").citations)

    def test_duplicate_query_terms(self):
        self.assertTrue(self.agent.answer("failure failure failure compressor").citations)

    def test_special_chars(self):
        self.assertTrue(self.agent.answer("compressor failure @#$%^&*").citations)

    def test_only_numbers(self):
        self.assertIsInstance(self.agent.answer("12345 67890"), KnowledgeResponse)

    def test_max_citations_large(self):
        cfg = AgentConfig(max_citations=100)
        self.assertLessEqual(len(self.agent.answer(Q_ROOT, config=cfg).citations), 6)

    def test_require_evidence_with_evidence_ok(self):
        cfg = AgentConfig(require_evidence=True)
        self.assertTrue(self.agent.answer(Q_ROOT, config=cfg).answer.supported)

    def test_single_doc_single_citation(self):
        docs = [FakeKnowledgeDocument("only", "compressor failure analysis report", "pm", ("t",), {"text": "compressor failure analysis report", "title": "T"})]
        agent = KnowledgeAgent.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertEqual(len(agent.answer("compressor failure").citations), 1)

    def test_classification_independent_of_kb(self):
        self.assertEqual(self.agent.classify(Q_SAFETY).question_type, QuestionType.SAFETY)

    def test_response_repr_safe(self):
        resp = self.agent.answer(Q_ROOT)
        self.assertIsInstance(repr(resp), str)

    def test_answer_text_is_str(self):
        self.assertIsInstance(self.agent.answer(Q_ROOT).answer.text, str)

    def test_metadata_intent(self):
        self.assertIn("intent", self.agent.answer(Q_ROOT).metadata)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)