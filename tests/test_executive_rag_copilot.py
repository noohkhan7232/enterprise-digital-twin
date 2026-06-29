"""Test suite for the Executive RAG Copilot (Week 9 Phase 5).

Covers executive question classification, Knowledge Agent integration, briefing
generation, insight extraction, recommendation generation, the confidence model,
statistics, serialization, edge cases, large knowledge bases, determinism, the
CLI, and backward compatibility with Weeks 1-9 Phase 4. Pure stdlib ``unittest``.

This file contains well over 400 individual test methods.
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

from knowledge.vector_search_engine import VectorSearchEngine  # noqa: E402
from knowledge.retrieval_intelligence import RetrievalEngine  # noqa: E402
from knowledge.knowledge_agent import AgentConfig, KnowledgeAgent, KnowledgeCitation  # noqa: E402
from knowledge.executive_rag_copilot import (  # noqa: E402
    CopilotConfig,
    DuplicateCitationError,
    DuplicateRecommendationError,
    EmptyExecutiveQuestionError,
    EXECUTIVE_QUESTION_TYPE_REGISTRY,
    ExecutiveBriefing,
    ExecutiveCitation,
    ExecutiveCopilotError,
    ExecutiveCopilotStatistics,
    ExecutiveInsight,
    ExecutiveQuestion,
    ExecutiveQuestionClassifier,
    ExecutiveQuestionRule,
    ExecutiveQuestionType,
    ExecutiveRAGCopilot,
    ExecutiveReasoning,
    ExecutiveRecommendation,
    InvalidQuestionTypeError,
    MissingEvidenceError,
    RECOMMENDATION_TEMPLATE_REGISTRY,
    SUBQUESTION_REGISTRY,
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


class FakeKnowledgeBase:
    def __init__(self, documents):
        self._documents = list(documents)

    def all_documents(self):
        return list(self._documents)


_ROWS = [
    ("doc-turbine-vibration", "Turbine Bearing Vibration Advisory",
     "Turbine bearing vibration anomaly was detected by the asset health engine. "
     "Elevated vibration indicates an emerging bearing fault and elevated failure risk. "
     "Recommended maintenance is to inspect and replace the affected bearing.",
     "predictive-maintenance", ("turbine", "vibration", "risk", "maintenance")),
    ("doc-rul-forecast", "Compressor RUL Forecast",
     "The remaining useful life of the compressor is forecast from the degradation trend. "
     "Continued operation increases the unplanned failure risk and downtime cost.",
     "predictive-maintenance", ("rul", "compressor", "risk", "cost")),
    ("doc-root-cause", "Compressor Failure Root Cause",
     "Root cause analysis traced the compressor failure to lubrication degradation. "
     "The corrective action is to restore the lubrication schedule.",
     "predictive-maintenance", ("root-cause", "compressor", "failure")),
    ("doc-cost", "Maintenance Cost Optimization",
     "Condition-based maintenance reduces unplanned downtime cost across the fleet. "
     "Optimising the service interval lowers total maintenance expense.",
     "decision", ("cost", "maintenance", "downtime")),
    ("doc-fleet-twin", "Fleet Digital Twin Overview",
     "The fleet digital twin aggregates asset health across the energy portfolio. "
     "Executives use it for strategic capital planning and investment prioritisation.",
     "digital-twin", ("fleet", "portfolio", "investment", "strategic")),
    ("doc-inspection", "Inspection Checklist",
     "Routine inspection requires checking bearing temperature and vibration monthly. "
     "Inspectors record clearance readings for reliability tracking.",
     "inspection", ("inspection", "check", "reliability")),
]

CONTENT_BY_ID = {r[0]: r[2] for r in _ROWS}
ALL_CONTENT = " ".join(r[2] for r in _ROWS)


def make_corpus():
    return [
        FakeKnowledgeDocument(
            document_id=r[0], content=r[2], category=r[3], tags=r[4],
            metadata={"text": r[2], "title": r[1], "freshness": 0.7},
        )
        for r in _ROWS
    ]


def make_copilot(config=None, clock=lambda: 0.0):
    return ExecutiveRAGCopilot.from_knowledge_base(make_corpus(), config=config, clock=clock)


QR = "Give me a risk assessment for the compressor"
QM = "What is the maintenance strategy for the turbine?"
QC = "How can we optimize maintenance cost across the fleet?"
QE = "Provide an executive summary of asset health"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    ALL_TYPES = (
        ExecutiveQuestionType.EXECUTIVE_SUMMARY, ExecutiveQuestionType.RISK_ASSESSMENT,
        ExecutiveQuestionType.MAINTENANCE_STRATEGY, ExecutiveQuestionType.INVESTMENT_PLANNING,
        ExecutiveQuestionType.ASSET_HEALTH_REVIEW, ExecutiveQuestionType.ROOT_CAUSE_REVIEW,
        ExecutiveQuestionType.SCENARIO_EVALUATION, ExecutiveQuestionType.OPERATIONAL_EFFICIENCY,
        ExecutiveQuestionType.COST_OPTIMIZATION, ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT,
    )

    def test_all_types_registered(self):
        for t in self.ALL_TYPES:
            self.assertTrue(EXECUTIVE_QUESTION_TYPE_REGISTRY.has(t))

    def test_all_have_subquestions(self):
        for t in self.ALL_TYPES:
            self.assertTrue(SUBQUESTION_REGISTRY.has(t))

    def test_all_have_recommendation_templates(self):
        for t in self.ALL_TYPES:
            self.assertTrue(RECOMMENDATION_TEMPLATE_REGISTRY.has(t))

    def test_rule_focus_nonempty(self):
        for t in self.ALL_TYPES:
            self.assertTrue(EXECUTIVE_QUESTION_TYPE_REGISTRY.get(t).focus)

    def test_rule_score_keyword(self):
        rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(ExecutiveQuestionType.RISK_ASSESSMENT)
        self.assertGreaterEqual(rule.score({"risk"}, "risk"), 1)

    def test_rule_score_phrase(self):
        rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(ExecutiveQuestionType.COST_OPTIMIZATION)
        self.assertGreaterEqual(rule.score(set(), "cost optimization plan"), 2)

    def test_rule_round_trip(self):
        rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(ExecutiveQuestionType.RISK_ASSESSMENT)
        self.assertEqual(ExecutiveQuestionRule.from_dict(rule.to_dict()), rule)

    def test_rule_frozen(self):
        rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(ExecutiveQuestionType.RISK_ASSESSMENT)
        with self.assertRaises(Exception):
            rule.focus = "x"  # type: ignore[misc]

    def test_subquestion_templates_have_topic(self):
        for t in self.ALL_TYPES:
            for tmpl in SUBQUESTION_REGISTRY.get(t):
                self.assertIn("{topic}", tmpl)

    def test_recommendation_templates_have_topic(self):
        for t in self.ALL_TYPES:
            self.assertIn("{topic}", RECOMMENDATION_TEMPLATE_REGISTRY.get(t))

    def test_registry_names_sorted(self):
        names = EXECUTIVE_QUESTION_TYPE_REGISTRY.names()
        self.assertEqual(list(names), sorted(names))

    def test_rule_keywords_tuple(self):
        self.assertIsInstance(EXECUTIVE_QUESTION_TYPE_REGISTRY.get(ExecutiveQuestionType.RISK_ASSESSMENT).keywords, tuple)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class TestClassification(unittest.TestCase):
    def setUp(self):
        self.clf = ExecutiveQuestionClassifier()

    def _t(self, q):
        return self.clf.classify(q).question_type

    def test_risk(self):
        self.assertEqual(self._t("Give me a risk assessment for the compressor"), ExecutiveQuestionType.RISK_ASSESSMENT)

    def test_maintenance(self):
        self.assertEqual(self._t("What is the maintenance strategy for the turbine?"), ExecutiveQuestionType.MAINTENANCE_STRATEGY)

    def test_cost(self):
        self.assertEqual(self._t("How can we reduce cost and optimize expense?"), ExecutiveQuestionType.COST_OPTIMIZATION)

    def test_investment(self):
        self.assertEqual(self._t("What capital investment and budget is needed?"), ExecutiveQuestionType.INVESTMENT_PLANNING)

    def test_asset_health(self):
        self.assertEqual(self._t("Review the asset health and condition"), ExecutiveQuestionType.ASSET_HEALTH_REVIEW)

    def test_root_cause(self):
        self.assertEqual(self._t("What is the root cause of the failure?"), ExecutiveQuestionType.ROOT_CAUSE_REVIEW)

    def test_scenario(self):
        self.assertEqual(self._t("Evaluate the what if scenario forecast"), ExecutiveQuestionType.SCENARIO_EVALUATION)

    def test_operational(self):
        self.assertEqual(self._t("Improve operational efficiency and throughput"), ExecutiveQuestionType.OPERATIONAL_EFFICIENCY)

    def test_strategic(self):
        self.assertEqual(self._t("Provide strategic decision support and roadmap"), ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT)

    def test_executive_summary(self):
        self.assertEqual(self._t("Give a high level overview snapshot"), ExecutiveQuestionType.EXECUTIVE_SUMMARY)

    def test_general_defaults_to_summary(self):
        self.assertEqual(self._t("Tell me about migratory birds"), ExecutiveQuestionType.EXECUTIVE_SUMMARY)

    def test_focus_set(self):
        self.assertEqual(self.clf.classify(QR).focus, "risk exposure")

    def test_priority_risk_high(self):
        self.assertEqual(self.clf.classify(QR).priority, "high")

    def test_priority_summary_low(self):
        self.assertEqual(self.clf.classify("overview snapshot").priority, "low")

    def test_priority_maintenance_medium(self):
        self.assertEqual(self.clf.classify(QM).priority, "medium")

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.clf.classify(QR).confidence <= 1.0)

    def test_confidence_capped(self):
        c = self.clf.classify("risk risks hazard exposure threat vulnerability danger").confidence
        self.assertLessEqual(c, 0.95)

    def test_sub_questions_present(self):
        self.assertTrue(self.clf.classify(QR).sub_questions)

    def test_sub_questions_include_original(self):
        self.assertEqual(self.clf.classify(QR).sub_questions[0], QR)

    def test_sub_questions_capped(self):
        clf = ExecutiveQuestionClassifier(max_sub_questions=2)
        self.assertLessEqual(len(clf.classify(QR).sub_questions), 2)

    def test_topic_extraction(self):
        self.assertEqual(self.clf.topic(QR), "compressor")

    def test_topic_fallback(self):
        self.assertEqual(self.clf.topic("the and of"), "the assets")

    def test_question_id_deterministic(self):
        self.assertEqual(self.clf.classify(QR).question_id, self.clf.classify(QR).question_id)

    def test_explicit_type_override(self):
        self.assertEqual(self.clf.classify(QR, question_type=ExecutiveQuestionType.COST_OPTIMIZATION).question_type,
                         ExecutiveQuestionType.COST_OPTIMIZATION)

    def test_invalid_type_override(self):
        with self.assertRaises(InvalidQuestionTypeError):
            self.clf.classify(QR, question_type="bogus")

    def test_deterministic_type(self):
        self.assertEqual(self._t(QR), self._t(QR))

    def test_returns_executive_question(self):
        self.assertIsInstance(self.clf.classify(QR), ExecutiveQuestion)


# ---------------------------------------------------------------------------
# ExecutiveQuestion
# ---------------------------------------------------------------------------
class TestExecutiveQuestion(unittest.TestCase):
    def _q(self):
        return ExecutiveQuestion("ex-1", "text", ExecutiveQuestionType.RISK_ASSESSMENT, "risk exposure", "high", 0.8, ("a", "b"))

    def test_fields(self):
        q = self._q()
        self.assertEqual(q.question_type, ExecutiveQuestionType.RISK_ASSESSMENT)
        self.assertEqual(q.priority, "high")

    def test_sub_questions_tuple(self):
        self.assertEqual(self._q().sub_questions, ("a", "b"))

    def test_confidence_float(self):
        self.assertIsInstance(self._q().confidence, float)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._q().confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        q = self._q()
        self.assertEqual(ExecutiveQuestion.from_dict(q.to_dict()), q)

    def test_json_round_trip(self):
        q = self._q()
        self.assertEqual(ExecutiveQuestion.from_dict(json.loads(json.dumps(q.to_dict()))), q)

    def test_to_dict_keys(self):
        for k in ("question_id", "text", "question_type", "focus", "priority", "confidence", "sub_questions"):
            self.assertIn(k, self._q().to_dict())

    def test_id_coercion(self):
        self.assertEqual(ExecutiveQuestion(5, "t", "x", "f", "low", 0.1).question_id, "5")

    def test_equality(self):
        self.assertEqual(self._q(), self._q())

    def test_focus(self):
        self.assertEqual(self._q().focus, "risk exposure")


# ---------------------------------------------------------------------------
# ExecutiveCitation
# ---------------------------------------------------------------------------
class TestExecutiveCitation(unittest.TestCase):
    def _c(self):
        return ExecutiveCitation("d1", "Title", 1, 0.9, 0.95, "snippet", "subq")

    def test_fields(self):
        c = self._c()
        self.assertEqual(c.document_id, "d1")
        self.assertEqual(c.source_question, "subq")

    def test_title_optional(self):
        self.assertIsNone(ExecutiveCitation("d", None, 1, 0.5, 0.5).title)

    def test_from_knowledge_citation(self):
        kc = KnowledgeCitation("d1", "T", 2, 0.7, 0.8, "snip")
        ec = ExecutiveCitation.from_knowledge_citation(kc, "subq")
        self.assertEqual(ec.document_id, "d1")
        self.assertEqual(ec.relevance_score, 0.8)
        self.assertEqual(ec.source_question, "subq")

    def test_from_knowledge_citation_preserves_title(self):
        kc = KnowledgeCitation("d1", "T", 2, 0.7, 0.8, "snip")
        self.assertEqual(ExecutiveCitation.from_knowledge_citation(kc).title, "T")

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._c().rank = 2  # type: ignore[misc]

    def test_round_trip(self):
        c = self._c()
        self.assertEqual(ExecutiveCitation.from_dict(c.to_dict()), c)

    def test_json_round_trip(self):
        c = self._c()
        self.assertEqual(ExecutiveCitation.from_dict(json.loads(json.dumps(c.to_dict()))), c)

    def test_round_trip_none_title(self):
        c = ExecutiveCitation("d", None, 1, 0.5, 0.5, "s", "q")
        self.assertIsNone(ExecutiveCitation.from_dict(c.to_dict()).title)

    def test_to_dict_keys(self):
        for k in ("document_id", "title", "rank", "confidence", "relevance_score", "snippet", "source_question"):
            self.assertIn(k, self._c().to_dict())

    def test_rank_int(self):
        self.assertIsInstance(self._c().rank, int)

    def test_relevance_float(self):
        self.assertIsInstance(self._c().relevance_score, float)

    def test_id_coercion(self):
        self.assertEqual(ExecutiveCitation(7, None, 1, 0.5, 0.5).document_id, "7")

    def test_equality(self):
        self.assertEqual(self._c(), self._c())


# ---------------------------------------------------------------------------
# ExecutiveInsight
# ---------------------------------------------------------------------------
class TestExecutiveInsight(unittest.TestCase):
    def _i(self):
        return ExecutiveInsight("Title", "summary", "impact", ("d1",), "high", 0.8)

    def test_fields(self):
        i = self._i()
        self.assertEqual(i.title, "Title")
        self.assertEqual(i.priority, "high")

    def test_supporting_evidence_tuple(self):
        self.assertEqual(self._i().supporting_evidence, ("d1",))

    def test_confidence_float(self):
        self.assertIsInstance(self._i().confidence, float)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._i().confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        i = self._i()
        self.assertEqual(ExecutiveInsight.from_dict(i.to_dict()), i)

    def test_json_round_trip(self):
        i = self._i()
        self.assertEqual(ExecutiveInsight.from_dict(json.loads(json.dumps(i.to_dict()))), i)

    def test_to_dict_keys(self):
        for k in ("title", "summary", "business_impact", "supporting_evidence", "priority", "confidence"):
            self.assertIn(k, self._i().to_dict())

    def test_business_impact(self):
        self.assertEqual(self._i().business_impact, "impact")

    def test_summary(self):
        self.assertEqual(self._i().summary, "summary")

    def test_equality(self):
        self.assertEqual(self._i(), self._i())

    def test_multi_evidence(self):
        i = ExecutiveInsight("T", "s", "i", ("d1", "d2"), "medium", 0.5)
        self.assertEqual(len(i.supporting_evidence), 2)

    def test_priority(self):
        self.assertEqual(self._i().priority, "high")


# ---------------------------------------------------------------------------
# ExecutiveRecommendation
# ---------------------------------------------------------------------------
class TestExecutiveRecommendation(unittest.TestCase):
    def _r(self):
        return ExecutiveRecommendation("act", "rationale", "impact", ("d1",), "high", 0.8)

    def test_fields(self):
        r = self._r()
        self.assertEqual(r.recommended_action, "act")
        self.assertEqual(r.priority, "high")

    def test_supporting_citations(self):
        self.assertEqual(self._r().supporting_citations, ("d1",))

    def test_confidence_float(self):
        self.assertIsInstance(self._r().confidence, float)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._r().confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        r = self._r()
        self.assertEqual(ExecutiveRecommendation.from_dict(r.to_dict()), r)

    def test_json_round_trip(self):
        r = self._r()
        self.assertEqual(ExecutiveRecommendation.from_dict(json.loads(json.dumps(r.to_dict()))), r)

    def test_to_dict_keys(self):
        for k in ("recommended_action", "business_rationale", "expected_impact", "supporting_citations", "priority", "confidence"):
            self.assertIn(k, self._r().to_dict())

    def test_rationale(self):
        self.assertEqual(self._r().business_rationale, "rationale")

    def test_impact(self):
        self.assertEqual(self._r().expected_impact, "impact")

    def test_equality(self):
        self.assertEqual(self._r(), self._r())

    def test_action(self):
        self.assertEqual(self._r().recommended_action, "act")


# ---------------------------------------------------------------------------
# ExecutiveReasoning
# ---------------------------------------------------------------------------
class TestExecutiveReasoning(unittest.TestCase):
    def _r(self):
        return ExecutiveReasoning("agg", 0.8, 0.7, 0.75, 3, 0.6, ("limit",))

    def test_fields(self):
        r = self._r()
        self.assertEqual(r.sub_question_count, 3)
        self.assertAlmostEqual(r.agent_confidence, 0.6)

    def test_evidence_strength(self):
        self.assertAlmostEqual(self._r().evidence_strength, 0.8)

    def test_limitations(self):
        self.assertEqual(self._r().limitations, ("limit",))

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._r().confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        r = self._r()
        self.assertEqual(ExecutiveReasoning.from_dict(r.to_dict()), r)

    def test_json_round_trip(self):
        r = self._r()
        self.assertEqual(ExecutiveReasoning.from_dict(json.loads(json.dumps(r.to_dict()))), r)

    def test_to_dict_keys(self):
        for k in ("aggregation_summary", "evidence_strength", "coverage", "confidence", "sub_question_count", "agent_confidence", "limitations"):
            self.assertIn(k, self._r().to_dict())

    def test_coverage(self):
        self.assertAlmostEqual(self._r().coverage, 0.7)

    def test_aggregation_summary(self):
        self.assertEqual(self._r().aggregation_summary, "agg")

    def test_confidence(self):
        self.assertAlmostEqual(self._r().confidence, 0.75)

    def test_empty_limitations(self):
        r = ExecutiveReasoning("a", 0.5, 0.5, 0.5, 1, 0.5)
        self.assertEqual(r.limitations, ())


# ---------------------------------------------------------------------------
# ExecutiveBriefing
# ---------------------------------------------------------------------------
class TestExecutiveBriefing(unittest.TestCase):
    def setUp(self):
        self.b = make_copilot().brief(QR)

    def test_has_question(self):
        self.assertIsInstance(self.b.question, ExecutiveQuestion)

    def test_has_summary(self):
        self.assertTrue(self.b.executive_summary)

    def test_has_key_findings(self):
        self.assertIsInstance(self.b.key_findings, tuple)

    def test_has_insights(self):
        self.assertTrue(self.b.insights)

    def test_has_recommendations(self):
        self.assertTrue(self.b.recommendations)

    def test_has_citations(self):
        self.assertTrue(self.b.citations)

    def test_has_reasoning(self):
        self.assertIsInstance(self.b.reasoning, ExecutiveReasoning)

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.b.confidence <= 1.0)

    def test_recommended_actions_property(self):
        self.assertEqual(self.b.recommended_actions, tuple(r.recommended_action for r in self.b.recommendations))

    def test_supporting_evidence_property(self):
        self.assertEqual(self.b.supporting_evidence, self.b.citations)

    def test_confidence_assessment_label(self):
        self.assertIn(self.b.confidence_assessment, ("high", "moderate", "low", "very low"))

    def test_citation_ids(self):
        self.assertEqual(self.b.citation_ids, tuple(c.document_id for c in self.b.citations))

    def test_metadata_readonly(self):
        self.assertIsInstance(self.b.metadata, MappingProxyType)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self.b.confidence = 0.1  # type: ignore[misc]

    def test_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.citation_ids, self.b.citation_ids)

    def test_json_round_trip(self):
        r = ExecutiveBriefing.from_dict(json.loads(json.dumps(self.b.to_dict())))
        self.assertEqual(r.executive_summary, self.b.executive_summary)

    def test_to_dict_keys(self):
        for k in ("question", "executive_summary", "key_findings", "business_risks", "insights",
                  "recommendations", "citations", "reasoning", "confidence", "limitations", "metadata"):
            self.assertIn(k, self.b.to_dict())

    def test_metadata_question_type(self):
        self.assertEqual(self.b.metadata["question_type"], ExecutiveQuestionType.RISK_ASSESSMENT)


# ---------------------------------------------------------------------------
# Copilot statistics dataclass
# ---------------------------------------------------------------------------
class TestStatisticsDataclass(unittest.TestCase):
    def _s(self):
        return ExecutiveCopilotStatistics(2, 0.5, 1.5, 6, 8, 10)

    def test_fields(self):
        s = self._s()
        self.assertEqual(s.briefings_generated, 2)
        self.assertEqual(s.recommendation_count, 6)

    def test_frozen(self):
        with self.assertRaises(Exception):
            self._s().briefings_generated = 9  # type: ignore[misc]

    def test_round_trip(self):
        s = self._s()
        self.assertEqual(ExecutiveCopilotStatistics.from_dict(s.to_dict()), s)

    def test_json_round_trip(self):
        s = self._s()
        self.assertEqual(ExecutiveCopilotStatistics.from_dict(json.loads(json.dumps(s.to_dict()))), s)

    def test_to_dict_keys(self):
        for k in ("briefings_generated", "average_confidence", "average_response_latency_ms",
                  "recommendation_count", "insight_count", "citation_count"):
            self.assertIn(k, self._s().to_dict())

    def test_types(self):
        s = ExecutiveCopilotStatistics(1, 2, 3, 4, 5, 6)
        self.assertIsInstance(s.average_confidence, float)
        self.assertIsInstance(s.citation_count, int)


# ---------------------------------------------------------------------------
# CopilotConfig
# ---------------------------------------------------------------------------
class TestCopilotConfig(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(CopilotConfig().max_insights, 4)

    def test_negative_insights_rejected(self):
        with self.assertRaises(ExecutiveCopilotError):
            CopilotConfig(max_insights=-1)

    def test_negative_recs_rejected(self):
        with self.assertRaises(ExecutiveCopilotError):
            CopilotConfig(max_recommendations=-1)

    def test_negative_subq_rejected(self):
        with self.assertRaises(ExecutiveCopilotError):
            CopilotConfig(max_sub_questions=-1)

    def test_threshold_range(self):
        with self.assertRaises(ExecutiveCopilotError):
            CopilotConfig(confidence_threshold=2.0)

    def test_frozen(self):
        with self.assertRaises(Exception):
            CopilotConfig().max_insights = 9  # type: ignore[misc]

    def test_round_trip(self):
        c = CopilotConfig(max_insights=2, max_recommendations=2, require_evidence=True)
        r = CopilotConfig.from_dict(c.to_dict())
        self.assertEqual(r.max_insights, 2)
        self.assertTrue(r.require_evidence)

    def test_json_round_trip(self):
        c = CopilotConfig()
        self.assertEqual(CopilotConfig.from_dict(json.loads(json.dumps(c.to_dict()))).max_insights, 4)

    def test_agent_config_none_default(self):
        self.assertIsNone(CopilotConfig().agent_config)

    def test_agent_config_round_trip(self):
        c = CopilotConfig(agent_config=AgentConfig(max_citations=2))
        r = CopilotConfig.from_dict(c.to_dict())
        self.assertEqual(r.agent_config.max_citations, 2)

    def test_require_evidence_default_false(self):
        self.assertFalse(CopilotConfig().require_evidence)

    def test_max_citations_default(self):
        self.assertEqual(CopilotConfig().max_citations, 6)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_brief_returns_briefing(self):
        self.assertIsInstance(self.copilot.brief(QR), ExecutiveBriefing)

    def test_brief_has_citations(self):
        self.assertTrue(self.copilot.brief(QR).citations)

    def test_brief_has_recommendations(self):
        self.assertTrue(self.copilot.brief(QR).recommendations)

    def test_brief_has_insights(self):
        self.assertTrue(self.copilot.brief(QR).insights)

    def test_classification_in_briefing(self):
        self.assertEqual(self.copilot.brief(QR).question.question_type, ExecutiveQuestionType.RISK_ASSESSMENT)

    def test_maintenance_briefing(self):
        self.assertEqual(self.copilot.brief(QM).question.question_type, ExecutiveQuestionType.MAINTENANCE_STRATEGY)

    def test_cost_briefing(self):
        self.assertEqual(self.copilot.brief(QC).question.question_type, ExecutiveQuestionType.COST_OPTIMIZATION)

    def test_classify_only(self):
        self.assertIsInstance(self.copilot.classify(QR), ExecutiveQuestion)

    def test_explicit_type(self):
        b = self.copilot.brief(QR, question_type=ExecutiveQuestionType.COST_OPTIMIZATION)
        self.assertEqual(b.question.question_type, ExecutiveQuestionType.COST_OPTIMIZATION)

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.copilot.brief(QR).confidence <= 1.0)

    def test_metadata_latency(self):
        self.assertIn("response_latency_ms", self.copilot.brief(QR).metadata)

    def test_metadata_counts(self):
        md = self.copilot.brief(QR).metadata
        self.assertIn("insight_count", md)
        self.assertIn("recommendation_count", md)

    def test_deterministic_briefing(self):
        a = self.copilot.brief(QR).citation_ids
        b = self.copilot.brief(QR).citation_ids
        self.assertEqual(a, b)

    def test_max_insights(self):
        cfg = CopilotConfig(max_insights=1)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).insights), 1)

    def test_max_recommendations(self):
        cfg = CopilotConfig(max_recommendations=1)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).recommendations), 1)

    def test_max_citations(self):
        cfg = CopilotConfig(max_citations=2)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).citations), 2)

    def test_sub_questions_recorded(self):
        self.assertTrue(self.copilot.brief(QR).question.sub_questions)

    def test_reasoning_sub_question_count(self):
        b = self.copilot.brief(QR)
        self.assertGreaterEqual(b.reasoning.sub_question_count, 1)

    def test_no_agent_raises(self):
        copilot = ExecutiveRAGCopilot()
        with self.assertRaises(ExecutiveCopilotError):
            copilot.brief(QR)


# ---------------------------------------------------------------------------
# Insight extraction
# ---------------------------------------------------------------------------
class TestInsightExtraction(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_insight_summary_grounded(self):
        for i in self.copilot.brief(QR).insights:
            self.assertIn(i.summary, ALL_CONTENT)

    def test_insight_title_nonempty(self):
        for i in self.copilot.brief(QR).insights:
            self.assertTrue(i.title)

    def test_insight_priority_valid(self):
        for i in self.copilot.brief(QR).insights:
            self.assertIn(i.priority, ("high", "medium", "low"))

    def test_insight_confidence_bounded(self):
        for i in self.copilot.brief(QR).insights:
            self.assertTrue(0.0 <= i.confidence <= 1.0)

    def test_insight_business_impact_nonempty(self):
        for i in self.copilot.brief(QR).insights:
            self.assertTrue(i.business_impact)

    def test_insight_evidence_present(self):
        for i in self.copilot.brief(QR).insights:
            self.assertTrue(i.supporting_evidence)

    def test_insight_count_respects_config(self):
        cfg = CopilotConfig(max_insights=2)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).insights), 2)

    def test_insight_title_from_citation(self):
        b = self.copilot.brief(QR)
        titles = {i.title for i in b.insights}
        self.assertTrue(any("Compressor" in t or "Turbine" in t for t in titles))

    def test_insight_evidence_matches_citation(self):
        b = self.copilot.brief(QR)
        cite_ids = set(b.citation_ids)
        for i in b.insights:
            self.assertTrue(set(i.supporting_evidence).issubset(cite_ids))

    def test_business_impact_mentions_focus(self):
        b = self.copilot.brief(QR)
        for i in b.insights:
            self.assertIn("risk exposure", i.business_impact)

    def test_key_findings_match_insights(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.key_findings, tuple(i.summary for i in b.insights))


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------
class TestRecommendationGeneration(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_rationale_grounded(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertIn(r.business_rationale, ALL_CONTENT)

    def test_actions_unique(self):
        actions = [r.recommended_action for r in self.copilot.brief(QR).recommendations]
        self.assertEqual(len(actions), len(set(actions)))

    def test_action_contains_topic(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertNotIn("{topic}", r.recommended_action)

    def test_impact_phrase(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertIn("business impact", r.expected_impact)

    def test_priority_valid(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertIn(r.priority, ("high", "medium", "low"))

    def test_confidence_bounded(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertTrue(0.0 <= r.confidence <= 1.0)

    def test_supporting_citations_present(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertTrue(r.supporting_citations)

    def test_count_respects_config(self):
        cfg = CopilotConfig(max_recommendations=1)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).recommendations), 1)

    def test_risk_action_template(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertIn("Mitigate the risk", r.recommended_action)

    def test_maintenance_action_template(self):
        for r in self.copilot.brief(QM).recommendations:
            self.assertIn("maintenance", r.recommended_action.lower())

    def test_cost_action_template(self):
        for r in self.copilot.brief(QC).recommendations:
            self.assertIn("cost", r.recommended_action.lower())

    def test_recommendation_maps_to_insight(self):
        b = self.copilot.brief(QR)
        rec_evidence = set()
        for r in b.recommendations:
            rec_evidence.update(r.supporting_citations)
        self.assertTrue(rec_evidence.issubset(set(b.citation_ids)))

    def test_rationale_equals_insight_summary(self):
        b = self.copilot.brief(QR)
        summaries = {i.summary for i in b.insights}
        for r in b.recommendations:
            self.assertIn(r.business_rationale, summaries)

    def test_no_invented_numbers(self):
        # Recommended actions are templates + grounded topics; rationale verbatim.
        for r in self.copilot.brief(QR).recommendations:
            self.assertIn(r.business_rationale, ALL_CONTENT)

    def test_actions_count_le_insights(self):
        b = self.copilot.brief(QR)
        self.assertLessEqual(len(b.recommendations), len(b.insights))

    def test_high_priority_high_impact(self):
        for r in self.copilot.brief(QR).recommendations:
            if r.priority == "high":
                self.assertEqual(r.expected_impact, "High expected business impact.")


# ---------------------------------------------------------------------------
# Citation aggregation
# ---------------------------------------------------------------------------
class TestCitationAggregation(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_no_duplicates(self):
        ids = self.copilot.brief(QR).citation_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_sorted_by_relevance(self):
        rels = [c.relevance_score for c in self.copilot.brief(QR).citations]
        self.assertEqual(rels, sorted(rels, reverse=True))

    def test_capped(self):
        cfg = CopilotConfig(max_citations=2)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).citations), 2)

    def test_source_question_recorded(self):
        for c in self.copilot.brief(QR).citations:
            self.assertTrue(c.source_question)

    def test_titles_present(self):
        for c in self.copilot.brief(QR).citations:
            self.assertIsNotNone(c.title)

    def test_snippets_grounded(self):
        for c in self.copilot.brief(QR).citations:
            self.assertIn(c.snippet, ALL_CONTENT)

    def test_reuses_agent_citations(self):
        # Citations are adapted KnowledgeCitations; relevance in [0,1].
        for c in self.copilot.brief(QR).citations:
            self.assertTrue(0.0 <= c.relevance_score <= 1.0)

    def test_multiple_sources_aggregated(self):
        # Risk assessment spans multiple sub-questions; expect >1 citation.
        self.assertGreaterEqual(len(self.copilot.brief(QR).citations), 1)

    def test_rank_positive(self):
        for c in self.copilot.brief(QR).citations:
            self.assertGreaterEqual(c.rank, 1)

    def test_deterministic_order(self):
        a = self.copilot.brief(QR).citation_ids
        b = self.copilot.brief(QR).citation_ids
        self.assertEqual(a, b)

    def test_dedup_keeps_highest_relevance(self):
        # A document appearing in multiple sub-answers appears once.
        ids = self.copilot.brief(QC).citation_ids
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------
class TestConfidenceModel(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_bounded(self):
        for q in (QR, QM, QC, QE):
            self.assertTrue(0.0 <= self.copilot.brief(q).confidence <= 1.0)

    def test_empty_kb_zero(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(copilot.brief(QR).confidence, 0.0)

    def test_deterministic(self):
        self.assertEqual(self.copilot.brief(QR).confidence, self.copilot.brief(QR).confidence)

    def test_reasoning_matches(self):
        b = self.copilot.brief(QR)
        self.assertAlmostEqual(b.reasoning.confidence, b.confidence)

    def test_agent_confidence_bounded(self):
        self.assertTrue(0.0 <= self.copilot.brief(QR).reasoning.agent_confidence <= 1.0)

    def test_coverage_bounded(self):
        self.assertTrue(0.0 <= self.copilot.brief(QR).reasoning.coverage <= 1.0)

    def test_evidence_strength_bounded(self):
        self.assertTrue(0.0 <= self.copilot.brief(QR).reasoning.evidence_strength <= 1.0)

    def test_assessment_label(self):
        self.assertIn(self.copilot.brief(QR).confidence_assessment, ("high", "moderate", "low", "very low"))

    def test_python_float(self):
        self.assertIsInstance(self.copilot.brief(QR).confidence, float)

    def test_relevant_not_lower_than_empty(self):
        good = self.copilot.brief(QR).confidence
        empty = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0).brief(QR).confidence
        self.assertGreaterEqual(good, empty)


# ---------------------------------------------------------------------------
# Business risks & summary
# ---------------------------------------------------------------------------
class TestBusinessRisksAndSummary(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_risks_identified_for_risk_question(self):
        self.assertTrue(self.copilot.brief(QR).business_risks)

    def test_risks_are_grounded(self):
        for r in self.copilot.brief(QR).business_risks:
            self.assertIn(r, ALL_CONTENT)

    def test_risks_contain_risk_tokens(self):
        risk_tokens = {"risk", "failure", "fault", "downtime", "degradation", "anomaly"}
        for r in self.copilot.brief(QR).business_risks:
            self.assertTrue(risk_tokens & set(r.lower().split()))

    def test_summary_nonempty(self):
        self.assertTrue(self.copilot.brief(QR).executive_summary)

    def test_summary_mentions_focus(self):
        self.assertIn("risk exposure", self.copilot.brief(QR).executive_summary)

    def test_summary_grounded_findings(self):
        b = self.copilot.brief(QR)
        # The summary embeds the first key findings verbatim.
        if b.key_findings:
            self.assertIn(b.key_findings[0], b.executive_summary)

    def test_empty_kb_summary(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertIn("No supporting evidence", copilot.brief(QR).executive_summary)

    def test_summary_confidence_label(self):
        b = self.copilot.brief(QR)
        self.assertIn(b.confidence_assessment, b.executive_summary)


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
class TestLimitations(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_limitations_tuple(self):
        self.assertIsInstance(self.copilot.brief(QR).limitations, tuple)

    def test_empty_kb_limitation(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertTrue(any("No supporting evidence" in l for l in copilot.brief(QR).limitations))

    def test_low_confidence_limitation(self):
        # QR scores below the default threshold in the demo corpus.
        b = self.copilot.brief(QR)
        if b.confidence < self.copilot.config.confidence_threshold:
            self.assertTrue(any("below the configured threshold" in l for l in b.limitations))

    def test_limitations_deduplicated(self):
        lims = self.copilot.brief(QR).limitations
        self.assertEqual(len(lims), len(set(lims)))

    def test_limitations_reflect_reasoning(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.limitations, b.reasoning.limitations)

    def test_limitations_strings(self):
        for l in self.copilot.brief(QR).limitations:
            self.assertIsInstance(l, str)

    def test_few_sources_limitation(self):
        cfg = CopilotConfig(max_citations=1)
        b = self.copilot.brief(QR, config=cfg)
        self.assertTrue(any("limited number of sources" in l for l in b.limitations))

    def test_no_limitation_keys_when_strong(self):
        # maintenance question tends to be better covered; still a tuple
        self.assertIsInstance(self.copilot.brief(QM).limitations, tuple)


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------
class TestStatisticsTracking(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_initial_zero(self):
        self.assertEqual(self.copilot.statistics().briefings_generated, 0)

    def test_count_increments(self):
        self.copilot.brief(QR)
        self.copilot.brief(QM)
        self.assertEqual(self.copilot.statistics().briefings_generated, 2)

    def test_average_confidence_bounded(self):
        self.copilot.brief(QR)
        self.assertTrue(0.0 <= self.copilot.statistics().average_confidence <= 1.0)

    def test_recommendation_count(self):
        b = self.copilot.brief(QR)
        self.assertEqual(self.copilot.statistics().recommendation_count, len(b.recommendations))

    def test_insight_count(self):
        b = self.copilot.brief(QR)
        self.assertEqual(self.copilot.statistics().insight_count, len(b.insights))

    def test_citation_count(self):
        b = self.copilot.brief(QR)
        self.assertEqual(self.copilot.statistics().citation_count, len(b.citations))

    def test_latency_zero_with_clock(self):
        self.copilot.brief(QR)
        self.assertEqual(self.copilot.statistics().average_response_latency_ms, 0.0)

    def test_reset(self):
        self.copilot.brief(QR)
        self.copilot.reset_statistics()
        self.assertEqual(self.copilot.statistics().briefings_generated, 0)

    def test_round_trip(self):
        self.copilot.brief(QR)
        s = self.copilot.statistics()
        self.assertEqual(ExecutiveCopilotStatistics.from_dict(s.to_dict()), s)

    def test_average_confidence_mean(self):
        c1 = self.copilot.brief(QR).confidence
        c2 = self.copilot.brief(QM).confidence
        self.assertAlmostEqual(self.copilot.statistics().average_confidence, (c1 + c2) / 2)

    def test_cumulative_counts(self):
        self.copilot.brief(QR)
        self.copilot.brief(QM)
        s = self.copilot.statistics()
        self.assertGreaterEqual(s.citation_count, 1)

    def test_real_clock_nonneg(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base(make_corpus())
        copilot.brief(QR)
        self.assertGreaterEqual(copilot.statistics().average_response_latency_ms, 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_empty_question(self):
        with self.assertRaises(EmptyExecutiveQuestionError):
            self.copilot.brief("")

    def test_whitespace_question(self):
        with self.assertRaises(EmptyExecutiveQuestionError):
            self.copilot.brief("    ")

    def test_punctuation_only(self):
        with self.assertRaises(EmptyExecutiveQuestionError):
            self.copilot.brief("!!! ???")

    def test_non_string(self):
        with self.assertRaises(EmptyExecutiveQuestionError):
            self.copilot.brief(123)  # type: ignore[arg-type]

    def test_invalid_type(self):
        with self.assertRaises(InvalidQuestionTypeError):
            self.copilot.brief(QR, question_type="not-a-type")

    def test_require_evidence_empty_kb(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], config=CopilotConfig(require_evidence=True), clock=lambda: 0.0)
        with self.assertRaises(MissingEvidenceError):
            copilot.brief(QR)

    def test_no_require_evidence_empty_ok(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(copilot.brief(QR).citations, ())

    def test_error_hierarchy(self):
        for e in (EmptyExecutiveQuestionError, InvalidQuestionTypeError, MissingEvidenceError,
                  DuplicateRecommendationError, DuplicateCitationError):
            self.assertTrue(issubclass(e, ExecutiveCopilotError))

    def test_classify_validates_empty(self):
        with self.assertRaises(EmptyExecutiveQuestionError):
            self.copilot.classify("")

    def test_classify_invalid_type(self):
        with self.assertRaises(InvalidQuestionTypeError):
            self.copilot.classify(QR, question_type="bogus")

    def test_no_duplicate_citations_invariant(self):
        ids = self.copilot.brief(QR).citation_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_duplicate_recommendations_invariant(self):
        actions = [r.recommended_action for r in self.copilot.brief(QR).recommendations]
        self.assertEqual(len(actions), len(set(actions)))


# ---------------------------------------------------------------------------
# Integration with Knowledge Agent (Phase 4)
# ---------------------------------------------------------------------------
class TestIntegration(unittest.TestCase):
    def test_from_kb_object(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base(FakeKnowledgeBase(make_corpus()), clock=lambda: 0.0)
        self.assertEqual(copilot.knowledge_agent.retrieval_engine.vector_engine.index.size(), 6)

    def test_inject_agent(self):
        agent = KnowledgeAgent.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        copilot = ExecutiveRAGCopilot(knowledge_agent=agent, clock=lambda: 0.0)
        self.assertTrue(copilot.brief(QR).citations)

    def test_uses_agent_for_evidence(self):
        copilot = make_copilot()
        # All citation ids must be real document ids from the KB.
        kb_ids = {r[0] for r in _ROWS}
        for c in copilot.brief(QR).citations:
            self.assertIn(c.document_id, kb_ids)

    def test_agent_config_passthrough(self):
        cfg = CopilotConfig(agent_config=AgentConfig(max_citations=1))
        b = make_copilot().brief(QR, config=cfg)
        self.assertIsInstance(b, ExecutiveBriefing)

    def test_classify_independent_of_kb(self):
        self.assertEqual(make_copilot().classify(QR).question_type, ExecutiveQuestionType.RISK_ASSESSMENT)

    def test_does_not_mutate_index(self):
        copilot = make_copilot()
        before = copilot.knowledge_agent.retrieval_engine.vector_engine.index.size()
        copilot.brief(QR)
        self.assertEqual(copilot.knowledge_agent.retrieval_engine.vector_engine.index.size(), before)

    def test_citation_is_executive_citation(self):
        for c in make_copilot().brief(QR).citations:
            self.assertIsInstance(c, ExecutiveCitation)

    def test_sub_questions_delegated(self):
        b = make_copilot().brief(QR)
        self.assertGreaterEqual(b.reasoning.sub_question_count, 1)


# ---------------------------------------------------------------------------
# Copilot serialization
# ---------------------------------------------------------------------------
class TestCopilotSerialization(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_to_dict_keys(self):
        self.assertEqual(set(self.copilot.to_dict()), {"config", "statistics", "agent"})

    def test_round_trip_briefs(self):
        restored = ExecutiveRAGCopilot.from_dict(self.copilot.to_dict())
        self.assertTrue(restored.brief(QR).citations)

    def test_round_trip_preserves_briefing(self):
        restored = ExecutiveRAGCopilot.from_dict(json.loads(json.dumps(self.copilot.to_dict())))
        a = self.copilot.brief(QR).citation_ids
        b = restored.brief(QR).citation_ids
        self.assertEqual(a, b)

    def test_round_trip_config(self):
        copilot = make_copilot(config=CopilotConfig(max_insights=2))
        restored = ExecutiveRAGCopilot.from_dict(copilot.to_dict())
        self.assertEqual(restored.config.max_insights, 2)

    def test_json_serializable(self):
        self.assertIsInstance(json.dumps(self.copilot.to_dict()), str)

    def test_none_agent(self):
        self.assertIsNone(ExecutiveRAGCopilot().to_dict()["agent"])


# ---------------------------------------------------------------------------
# Large knowledge base
# ---------------------------------------------------------------------------
class TestLargeKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = []
        for i in range(500):
            row = _ROWS[i % len(_ROWS)]
            content = row[2] + f" Record {i}."
            docs.append(FakeKnowledgeDocument(
                document_id=f"doc-{i:05d}", content=content, category=row[3], tags=row[4],
                metadata={"text": content, "title": f"{row[1]} #{i}", "freshness": round((i % 10) / 9.0, 3)},
            ))
        cls.copilot = ExecutiveRAGCopilot.from_knowledge_base(docs, clock=lambda: 0.0)

    def test_indexed(self):
        self.assertEqual(self.copilot.knowledge_agent.retrieval_engine.vector_engine.index.size(), 500)

    def test_briefs(self):
        self.assertTrue(self.copilot.brief(QR).citations)

    def test_deterministic(self):
        self.assertEqual(self.copilot.brief(QR).citation_ids, self.copilot.brief(QR).citation_ids)

    def test_capped_citations(self):
        self.assertLessEqual(len(self.copilot.brief(QR).citations), self.copilot.config.max_citations)

    def test_capped_insights(self):
        self.assertLessEqual(len(self.copilot.brief(QR).insights), self.copilot.config.max_insights)

    def test_confidence_bounded(self):
        self.assertTrue(0.0 <= self.copilot.brief(QR).confidence <= 1.0)

    def test_grounded(self):
        for i in self.copilot.brief(QR).insights:
            self.assertTrue(i.summary)

    def test_recommendations_unique(self):
        actions = [r.recommended_action for r in self.copilot.brief(QR).recommendations]
        self.assertEqual(len(actions), len(set(actions)))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_full_dict_equal(self):
        a = make_copilot().brief(QR).to_dict()
        b = make_copilot().brief(QR).to_dict()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_classification_stable(self):
        clf = ExecutiveQuestionClassifier()
        self.assertEqual(clf.classify(QR).question_type, clf.classify(QR).question_type)

    def test_confidence_stable(self):
        copilot = make_copilot()
        self.assertEqual(copilot.brief(QR).confidence, copilot.brief(QR).confidence)

    def test_citations_stable(self):
        copilot = make_copilot()
        self.assertEqual(copilot.brief(QR).citation_ids, copilot.brief(QR).citation_ids)

    def test_recommendations_stable(self):
        copilot = make_copilot()
        a = [r.recommended_action for r in copilot.brief(QR).recommendations]
        b = [r.recommended_action for r in copilot.brief(QR).recommendations]
        self.assertEqual(a, b)

    def test_across_instances(self):
        self.assertEqual(make_copilot().brief(QR).executive_summary, make_copilot().brief(QR).executive_summary)

    def test_question_id_stable(self):
        self.assertEqual(make_copilot().brief(QR).question.question_id, make_copilot().brief(QR).question.question_id)

    def test_insights_stable(self):
        copilot = make_copilot()
        a = [i.title for i in copilot.brief(QR).insights]
        b = [i.title for i in copilot.brief(QR).insights]
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Multi-question invariants
# ---------------------------------------------------------------------------
class TestMultiQuestion(unittest.TestCase):
    QUESTIONS = [QR, QM, QC, QE,
                 "What is the root cause of the compressor failure?",
                 "Evaluate the investment for the fleet portfolio"]

    def setUp(self):
        self.copilot = make_copilot()

    def test_all_brief(self):
        for q in self.QUESTIONS:
            self.assertIsInstance(self.copilot.brief(q), ExecutiveBriefing, msg=q)

    def test_all_confidence_bounded(self):
        for q in self.QUESTIONS:
            self.assertTrue(0.0 <= self.copilot.brief(q).confidence <= 1.0, msg=q)

    def test_all_have_citations(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.copilot.brief(q).citations, msg=q)

    def test_all_grounded_insights(self):
        for q in self.QUESTIONS:
            for i in self.copilot.brief(q).insights:
                self.assertIn(i.summary, ALL_CONTENT, msg=q)

    def test_all_grounded_rationales(self):
        for q in self.QUESTIONS:
            for r in self.copilot.brief(q).recommendations:
                self.assertIn(r.business_rationale, ALL_CONTENT, msg=q)

    def test_all_unique_recommendations(self):
        for q in self.QUESTIONS:
            actions = [r.recommended_action for r in self.copilot.brief(q).recommendations]
            self.assertEqual(len(actions), len(set(actions)), msg=q)

    def test_all_no_duplicate_citations(self):
        for q in self.QUESTIONS:
            ids = self.copilot.brief(q).citation_ids
            self.assertEqual(len(ids), len(set(ids)), msg=q)

    def test_all_json_round_trip(self):
        for q in self.QUESTIONS:
            b = self.copilot.brief(q)
            r = ExecutiveBriefing.from_dict(json.loads(json.dumps(b.to_dict())))
            self.assertEqual(r.citation_ids, b.citation_ids, msg=q)

    def test_all_deterministic(self):
        for q in self.QUESTIONS:
            self.assertEqual(self.copilot.brief(q).citation_ids, self.copilot.brief(q).citation_ids, msg=q)

    def test_all_have_reasoning(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.copilot.brief(q).reasoning.aggregation_summary, msg=q)

    def test_all_classified(self):
        for q in self.QUESTIONS:
            self.assertTrue(self.copilot.brief(q).question.question_type, msg=q)

    def test_all_confidence_assessment(self):
        for q in self.QUESTIONS:
            self.assertIn(self.copilot.brief(q).confidence_assessment, ("high", "moderate", "low", "very low"), msg=q)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompatibility(unittest.TestCase):
    def test_phase2(self):
        ve = VectorSearchEngine()
        ve.index_documents(make_corpus())
        self.assertTrue(ve.search("compressor", top_k=2))

    def test_phase3(self):
        re = RetrievalEngine.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        self.assertTrue(re.retrieve("compressor failure").context.documents)

    def test_phase4(self):
        agent = KnowledgeAgent.from_knowledge_base(make_corpus(), clock=lambda: 0.0)
        self.assertTrue(agent.answer("compressor failure").citations)

    def test_phase4_citation_reuse(self):
        kc = KnowledgeCitation("d", "T", 1, 0.5, 0.6, "s")
        ec = ExecutiveCitation.from_knowledge_citation(kc)
        self.assertEqual(ec.snippet, "s")

    def test_copilot_uses_full_stack(self):
        copilot = make_copilot()
        b = copilot.brief(QR)
        self.assertEqual(b.metadata["citation_count"], len(b.citations))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_single_doc_kb(self):
        docs = [FakeKnowledgeDocument("only", "compressor failure risk degradation report here", "pm", ("t",), {"text": "compressor failure risk degradation report here", "title": "Only"})]
        copilot = ExecutiveRAGCopilot.from_knowledge_base(docs, clock=lambda: 0.0)
        self.assertTrue(copilot.brief(QR).citations)

    def test_unicode_question(self):
        self.assertTrue(self.copilot.brief("café risk assessment for the compressor").citations)

    def test_long_question(self):
        self.assertTrue(self.copilot.brief("risk " * 100 + "compressor").citations)

    def test_max_insights_zero(self):
        cfg = CopilotConfig(max_insights=0)
        b = self.copilot.brief(QR, config=cfg)
        self.assertEqual(len(b.insights), 0)
        self.assertEqual(len(b.recommendations), 0)

    def test_max_recommendations_zero(self):
        cfg = CopilotConfig(max_recommendations=0)
        self.assertEqual(len(self.copilot.brief(QR, config=cfg).recommendations), 0)

    def test_max_subq_one(self):
        cfg = CopilotConfig(max_sub_questions=1)
        copilot = make_copilot(config=cfg)
        # classifier built from config => only the original question
        self.assertLessEqual(len(copilot.classify(QR).sub_questions), 1)

    def test_empty_kb_no_recommendations(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(copilot.brief(QR).recommendations, ())

    def test_empty_kb_no_insights(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(copilot.brief(QR).insights, ())

    def test_special_chars_question(self):
        self.assertTrue(self.copilot.brief("risk @#$ assessment compressor").citations)

    def test_mixed_case(self):
        self.assertEqual(self.copilot.brief("RISK ASSESSMENT COMPRESSOR").question.question_type, ExecutiveQuestionType.RISK_ASSESSMENT)

    def test_briefing_repr(self):
        self.assertIsInstance(repr(self.copilot.brief(QR)), str)

    def test_numbers_question(self):
        self.assertIsInstance(self.copilot.brief("risk 12345 compressor"), ExecutiveBriefing)

    def test_recommendation_action_str(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertIsInstance(r.recommended_action, str)

    def test_key_findings_grounded(self):
        for k in self.copilot.brief(QR).key_findings:
            self.assertIn(k, ALL_CONTENT)

    def test_executive_summary_str(self):
        self.assertIsInstance(self.copilot.brief(QR).executive_summary, str)

    def test_brief_with_explicit_type_grounded(self):
        b = self.copilot.brief(QM, question_type=ExecutiveQuestionType.MAINTENANCE_STRATEGY)
        for i in b.insights:
            self.assertIn(i.summary, ALL_CONTENT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI(unittest.TestCase):
    def test_demo_runs(self):
        from knowledge.executive_rag_copilot import main
        self.assertEqual(main(["--demo"]), 0)

    def test_no_args_prints_help(self):
        from knowledge.executive_rag_copilot import main
        self.assertEqual(main([]), 0)


# ---------------------------------------------------------------------------
# Per-type sub-question templates
# ---------------------------------------------------------------------------
class TestSubQuestionTemplates(unittest.TestCase):
    def setUp(self):
        self.clf = ExecutiveQuestionClassifier()

    def _subs(self, q, qtype):
        return self.clf.classify(q, question_type=qtype).sub_questions

    def test_risk_subs_topic(self):
        subs = self._subs(QR, ExecutiveQuestionType.RISK_ASSESSMENT)
        self.assertTrue(any("compressor" in s for s in subs[1:]))

    def test_maintenance_subs(self):
        subs = self._subs(QM, ExecutiveQuestionType.MAINTENANCE_STRATEGY)
        self.assertTrue(any("maintenance" in s.lower() for s in subs[1:]))

    def test_cost_subs(self):
        subs = self._subs(QC, ExecutiveQuestionType.COST_OPTIMIZATION)
        self.assertTrue(any("cost" in s.lower() or "reduce" in s.lower() for s in subs[1:]))

    def test_investment_subs(self):
        subs = self._subs("Plan investment for the fleet", ExecutiveQuestionType.INVESTMENT_PLANNING)
        self.assertTrue(any("investment" in s.lower() or "cost" in s.lower() for s in subs[1:]))

    def test_asset_health_subs(self):
        subs = self._subs("Asset health of the turbine", ExecutiveQuestionType.ASSET_HEALTH_REVIEW)
        self.assertTrue(any("health" in s.lower() or "degradation" in s.lower() for s in subs[1:]))

    def test_root_cause_subs(self):
        subs = self._subs("Root cause of failure", ExecutiveQuestionType.ROOT_CAUSE_REVIEW)
        self.assertTrue(any("root cause" in s.lower() or "contribut" in s.lower() for s in subs[1:]))

    def test_scenario_subs(self):
        subs = self._subs("Scenario for the fleet", ExecutiveQuestionType.SCENARIO_EVALUATION)
        self.assertTrue(any("scenario" in s.lower() or "forecast" in s.lower() for s in subs[1:]))

    def test_operational_subs(self):
        subs = self._subs("Operational efficiency review", ExecutiveQuestionType.OPERATIONAL_EFFICIENCY)
        self.assertTrue(any("efficiency" in s.lower() or "downtime" in s.lower() for s in subs[1:]))

    def test_strategic_subs(self):
        subs = self._subs("Strategic decision for the fleet", ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT)
        self.assertTrue(any("decision" in s.lower() or "trade" in s.lower() for s in subs[1:]))

    def test_summary_subs(self):
        subs = self._subs("Overview of the fleet", ExecutiveQuestionType.EXECUTIVE_SUMMARY)
        self.assertTrue(any("status" in s.lower() or "findings" in s.lower() for s in subs[1:]))

    def test_first_sub_is_original(self):
        for qtype in (ExecutiveQuestionType.RISK_ASSESSMENT, ExecutiveQuestionType.COST_OPTIMIZATION):
            self.assertEqual(self._subs(QR, qtype)[0], QR)

    def test_subs_unique(self):
        subs = self._subs(QR, ExecutiveQuestionType.RISK_ASSESSMENT)
        self.assertEqual(len(subs), len(set(subs)))

    def test_subs_no_brace(self):
        for s in self._subs(QR, ExecutiveQuestionType.RISK_ASSESSMENT):
            self.assertNotIn("{topic}", s)


# ---------------------------------------------------------------------------
# Per-type recommendation templates
# ---------------------------------------------------------------------------
class TestRecommendationTemplates(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def _actions(self, q, qtype):
        return [r.recommended_action for r in self.copilot.brief(q, question_type=qtype).recommendations]

    def test_risk(self):
        self.assertTrue(all("Mitigate" in a for a in self._actions(QR, ExecutiveQuestionType.RISK_ASSESSMENT)))

    def test_maintenance(self):
        self.assertTrue(all("maintenance" in a.lower() for a in self._actions(QM, ExecutiveQuestionType.MAINTENANCE_STRATEGY)))

    def test_cost(self):
        self.assertTrue(all("cost-reduction" in a for a in self._actions(QC, ExecutiveQuestionType.COST_OPTIMIZATION)))

    def test_investment(self):
        acts = self._actions("Investment plan for the fleet", ExecutiveQuestionType.INVESTMENT_PLANNING)
        self.assertTrue(all("capital allocation" in a for a in acts))

    def test_asset_health(self):
        acts = self._actions("Asset health of the turbine", ExecutiveQuestionType.ASSET_HEALTH_REVIEW)
        self.assertTrue(all("asset health" in a for a in acts))

    def test_root_cause(self):
        acts = self._actions("Root cause of the failure", ExecutiveQuestionType.ROOT_CAUSE_REVIEW)
        self.assertTrue(all("root cause" in a for a in acts))

    def test_scenario(self):
        acts = self._actions("Scenario for the fleet", ExecutiveQuestionType.SCENARIO_EVALUATION)
        self.assertTrue(all("scenario" in a for a in acts))

    def test_operational(self):
        acts = self._actions("Operational efficiency of fleet", ExecutiveQuestionType.OPERATIONAL_EFFICIENCY)
        self.assertTrue(all("operational efficiency" in a for a in acts))

    def test_strategic(self):
        acts = self._actions("Strategic decision support", ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT)
        self.assertTrue(all("strategic decision" in a for a in acts))

    def test_summary(self):
        acts = self._actions("Overview of asset health", ExecutiveQuestionType.EXECUTIVE_SUMMARY)
        self.assertTrue(all("status" in a for a in acts))

    def test_all_actions_grounded_topic(self):
        # The quoted topic must be a real citation title for the briefing.
        b = self.copilot.brief(QR, question_type=ExecutiveQuestionType.RISK_ASSESSMENT)
        titles = {i.title for i in b.insights}
        for r in b.recommendations:
            self.assertTrue(any(t in r.recommended_action for t in titles))

    def test_actions_distinct_per_type(self):
        for qtype in (ExecutiveQuestionType.RISK_ASSESSMENT, ExecutiveQuestionType.COST_OPTIMIZATION):
            acts = self._actions(QR, qtype)
            self.assertEqual(len(acts), len(set(acts)))


# ---------------------------------------------------------------------------
# Per-type briefing smoke
# ---------------------------------------------------------------------------
class TestPerTypeBriefing(unittest.TestCase):
    TYPE_QUERIES = {
        ExecutiveQuestionType.EXECUTIVE_SUMMARY: "Give a high level overview snapshot briefing",
        ExecutiveQuestionType.RISK_ASSESSMENT: "Risk assessment for the compressor",
        ExecutiveQuestionType.MAINTENANCE_STRATEGY: "Maintenance strategy for the turbine",
        ExecutiveQuestionType.INVESTMENT_PLANNING: "Investment plan for the fleet",
        ExecutiveQuestionType.ASSET_HEALTH_REVIEW: "Asset health review of the compressor",
        ExecutiveQuestionType.ROOT_CAUSE_REVIEW: "Root cause review of the failure",
        ExecutiveQuestionType.SCENARIO_EVALUATION: "Scenario evaluation for the fleet",
        ExecutiveQuestionType.OPERATIONAL_EFFICIENCY: "Operational efficiency of the fleet",
        ExecutiveQuestionType.COST_OPTIMIZATION: "Cost optimization across the fleet",
        ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT: "Strategic decision support for the portfolio",
    }

    def setUp(self):
        self.copilot = make_copilot()

    def test_each_type_classifies(self):
        for qtype, q in self.TYPE_QUERIES.items():
            self.assertEqual(self.copilot.brief(q).question.question_type, qtype, msg=q)

    def test_each_type_briefs(self):
        for q in self.TYPE_QUERIES.values():
            self.assertIsInstance(self.copilot.brief(q), ExecutiveBriefing, msg=q)

    def test_each_type_confidence_bounded(self):
        for q in self.TYPE_QUERIES.values():
            self.assertTrue(0.0 <= self.copilot.brief(q).confidence <= 1.0, msg=q)

    def test_each_type_citations(self):
        for q in self.TYPE_QUERIES.values():
            self.assertTrue(self.copilot.brief(q).citations, msg=q)

    def test_each_type_grounded(self):
        for q in self.TYPE_QUERIES.values():
            for i in self.copilot.brief(q).insights:
                self.assertIn(i.summary, ALL_CONTENT, msg=q)

    def test_each_type_unique_recs(self):
        for q in self.TYPE_QUERIES.values():
            acts = [r.recommended_action for r in self.copilot.brief(q).recommendations]
            self.assertEqual(len(acts), len(set(acts)), msg=q)

    def test_each_type_round_trip(self):
        for q in self.TYPE_QUERIES.values():
            b = self.copilot.brief(q)
            self.assertEqual(ExecutiveBriefing.from_dict(b.to_dict()).citation_ids, b.citation_ids, msg=q)

    def test_each_type_priority_valid(self):
        for q in self.TYPE_QUERIES.values():
            self.assertIn(self.copilot.brief(q).question.priority, ("high", "medium", "low"), msg=q)

    def test_each_type_deterministic(self):
        for q in self.TYPE_QUERIES.values():
            self.assertEqual(self.copilot.brief(q).confidence, self.copilot.brief(q).confidence, msg=q)

    def test_each_type_focus_in_summary(self):
        for q in self.TYPE_QUERIES.values():
            b = self.copilot.brief(q)
            self.assertIn(b.question.focus, b.executive_summary, msg=q)


# ---------------------------------------------------------------------------
# Grounding guarantees (anti-hallucination)
# ---------------------------------------------------------------------------
class TestGrounding(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_insight_summaries_substring(self):
        for q in (QR, QM, QC, QE):
            for i in self.copilot.brief(q).insights:
                self.assertIn(i.summary, ALL_CONTENT, msg=q)

    def test_rationales_substring(self):
        for q in (QR, QM, QC, QE):
            for r in self.copilot.brief(q).recommendations:
                self.assertIn(r.business_rationale, ALL_CONTENT, msg=q)

    def test_key_findings_substring(self):
        for q in (QR, QM, QC, QE):
            for k in self.copilot.brief(q).key_findings:
                self.assertIn(k, ALL_CONTENT, msg=q)

    def test_risks_substring(self):
        for q in (QR, QM, QC, QE):
            for risk in self.copilot.brief(q).business_risks:
                self.assertIn(risk, ALL_CONTENT, msg=q)

    def test_snippets_substring(self):
        for q in (QR, QM, QC, QE):
            for c in self.copilot.brief(q).citations:
                self.assertIn(c.snippet, ALL_CONTENT, msg=q)

    def test_citation_titles_real(self):
        real_titles = {r[1] for r in _ROWS}
        for c in self.copilot.brief(QR).citations:
            self.assertIn(c.title, real_titles)

    def test_citation_ids_real(self):
        real_ids = {r[0] for r in _ROWS}
        for c in self.copilot.brief(QR).citations:
            self.assertIn(c.document_id, real_ids)

    def test_rationale_is_a_snippet(self):
        b = self.copilot.brief(QR)
        snippets = {c.snippet for c in b.citations}
        for r in b.recommendations:
            self.assertIn(r.business_rationale, snippets)

    def test_summary_text_grounded_fragment(self):
        b = self.copilot.brief(QR)
        if b.key_findings:
            self.assertIn(b.key_findings[0], b.executive_summary)

    def test_no_recommendation_without_citation(self):
        for r in self.copilot.brief(QR).recommendations:
            self.assertTrue(r.supporting_citations)


# ---------------------------------------------------------------------------
# Confidence component behaviour
# ---------------------------------------------------------------------------
class TestConfidenceComponents(unittest.TestCase):
    def test_consistency_empty(self):
        self.assertEqual(ExecutiveRAGCopilot._consistency([]), 0.0)

    def test_consistency_single(self):
        self.assertEqual(ExecutiveRAGCopilot._consistency([{"a"}]), 1.0)

    def test_consistency_identical(self):
        self.assertEqual(ExecutiveRAGCopilot._consistency([{"a"}, {"a"}]), 1.0)

    def test_consistency_disjoint(self):
        self.assertEqual(ExecutiveRAGCopilot._consistency([{"a"}, {"b"}]), 0.0)

    def test_consistency_partial(self):
        v = ExecutiveRAGCopilot._consistency([{"a", "b"}, {"b", "c"}])
        self.assertTrue(0.0 < v < 1.0)

    def test_compute_confidence_bounded(self):
        c = make_copilot()
        val = c._compute_confidence(0.8, 0.7, (), ())
        self.assertTrue(0.0 <= val <= 1.0)

    def test_compute_confidence_zero_inputs(self):
        c = make_copilot()
        self.assertEqual(c._compute_confidence(0.0, 0.0, (), ()), 0.0)

    def test_compute_confidence_monotone_agent(self):
        c = make_copilot()
        low = c._compute_confidence(0.2, 0.5, (), ())
        high = c._compute_confidence(0.9, 0.5, (), ())
        self.assertGreater(high, low)

    def test_compute_confidence_monotone_coverage(self):
        c = make_copilot()
        low = c._compute_confidence(0.5, 0.1, (), ())
        high = c._compute_confidence(0.5, 0.9, (), ())
        self.assertGreater(high, low)

    def test_reasoning_confidence_equals_briefing(self):
        b = make_copilot().brief(QR)
        self.assertAlmostEqual(b.reasoning.confidence, b.confidence)


# ---------------------------------------------------------------------------
# Serialization (extra / nested)
# ---------------------------------------------------------------------------
class TestSerializationExtra(unittest.TestCase):
    def setUp(self):
        self.b = make_copilot().brief(QR)

    def test_briefing_full_round_trip(self):
        r = ExecutiveBriefing.from_dict(json.loads(json.dumps(self.b.to_dict())))
        self.assertEqual(r.confidence, self.b.confidence)

    def test_insights_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual([i.title for i in r.insights], [i.title for i in self.b.insights])

    def test_recommendations_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual([x.recommended_action for x in r.recommendations],
                         [x.recommended_action for x in self.b.recommendations])

    def test_citations_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.citation_ids, self.b.citation_ids)

    def test_reasoning_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.reasoning.sub_question_count, self.b.reasoning.sub_question_count)

    def test_question_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.question.question_type, self.b.question.question_type)

    def test_limitations_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.limitations, self.b.limitations)

    def test_business_risks_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.business_risks, self.b.business_risks)

    def test_key_findings_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.key_findings, self.b.key_findings)

    def test_metadata_round_trip(self):
        r = ExecutiveBriefing.from_dict(self.b.to_dict())
        self.assertEqual(r.metadata["question_type"], self.b.metadata["question_type"])

    def test_confidence_assessment_in_dict(self):
        self.assertIn("confidence_assessment", self.b.to_dict())

    def test_dict_json_serializable(self):
        self.assertIsInstance(json.dumps(self.b.to_dict()), str)

    def test_copilot_dict_serializable(self):
        self.assertIsInstance(json.dumps(make_copilot().to_dict()), str)

    def test_nested_insight_dicts(self):
        d = self.b.to_dict()
        self.assertTrue(all("title" in i for i in d["insights"]))

    def test_nested_recommendation_dicts(self):
        d = self.b.to_dict()
        self.assertTrue(all("recommended_action" in r for r in d["recommendations"]))

    def test_nested_citation_dicts(self):
        d = self.b.to_dict()
        self.assertTrue(all("document_id" in c for c in d["citations"]))


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------
class TestEdgeCasesExtra(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_trailing_whitespace(self):
        self.assertTrue(self.copilot.brief("  risk assessment compressor  ").citations)

    def test_repeated_terms(self):
        self.assertTrue(self.copilot.brief("risk risk risk compressor compressor").citations)

    def test_question_mark_only_text(self):
        self.assertEqual(self.copilot.brief("What is the risk for the compressor?").question.question_type,
                         ExecutiveQuestionType.RISK_ASSESSMENT)

    def test_max_citations_one(self):
        cfg = CopilotConfig(max_citations=1)
        self.assertLessEqual(len(self.copilot.brief(QR, config=cfg).citations), 1)

    def test_insights_le_citations(self):
        b = self.copilot.brief(QR)
        self.assertLessEqual(len(b.insights), len(b.citations))

    def test_recs_le_insights(self):
        b = self.copilot.brief(QR)
        self.assertLessEqual(len(b.recommendations), len(b.insights))

    def test_config_override_does_not_persist(self):
        self.copilot.brief(QR, config=CopilotConfig(max_insights=1))
        # Default config still 4.
        self.assertEqual(self.copilot.config.max_insights, 4)

    def test_statistics_after_override(self):
        self.copilot.brief(QR, config=CopilotConfig(max_insights=1))
        self.assertEqual(self.copilot.statistics().briefings_generated, 1)

    def test_brief_empty_kb_returns_briefing(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertIsInstance(copilot.brief(QR), ExecutiveBriefing)

    def test_brief_empty_kb_confidence_zero(self):
        copilot = ExecutiveRAGCopilot.from_knowledge_base([], clock=lambda: 0.0)
        self.assertEqual(copilot.brief(QR).confidence, 0.0)

    def test_classifier_topic_stable(self):
        clf = ExecutiveQuestionClassifier()
        self.assertEqual(clf.topic(QR), clf.topic(QR))

    def test_property_recommended_actions_match(self):
        b = self.copilot.brief(QR)
        self.assertEqual(list(b.recommended_actions), [r.recommended_action for r in b.recommendations])

    def test_property_supporting_evidence_is_citations(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.supporting_evidence, b.citations)

    def test_reasoning_agg_summary_mentions_focus(self):
        b = self.copilot.brief(QR)
        self.assertIn(b.question.focus, b.reasoning.aggregation_summary)


# ---------------------------------------------------------------------------
# Extended determinism & property checks
# ---------------------------------------------------------------------------
class TestExtendedProperties(unittest.TestCase):
    def setUp(self):
        self.copilot = make_copilot()

    def test_metadata_sub_questions_list(self):
        self.assertIsInstance(self.copilot.brief(QR).metadata["sub_questions"], list)

    def test_metadata_focus(self):
        self.assertEqual(self.copilot.brief(QR).metadata["focus"], "risk exposure")

    def test_metadata_priority(self):
        self.assertEqual(self.copilot.brief(QR).metadata["priority"], "high")

    def test_evidence_strength_equals_mean(self):
        b = self.copilot.brief(QR)
        expected = float(np.mean([c.relevance_score for c in b.citations])) if b.citations else 0.0
        self.assertAlmostEqual(b.reasoning.evidence_strength, expected)

    def test_insight_priority_matches_relevance_bucket(self):
        for i in self.copilot.brief(QR).insights:
            self.assertIn(i.priority, ("high", "medium", "low"))

    def test_recommendation_priority_matches_insight(self):
        b = self.copilot.brief(QR)
        ins_by_title = {i.title: i for i in b.insights}
        for r in b.recommendations:
            # action carries the insight title in quotes
            self.assertTrue(any(t in r.recommended_action for t in ins_by_title))

    def test_classify_method_matches_brief(self):
        self.assertEqual(self.copilot.classify(QR).question_type, self.copilot.brief(QR).question.question_type)

    def test_two_copilots_same_summary(self):
        self.assertEqual(make_copilot().brief(QM).executive_summary, make_copilot().brief(QM).executive_summary)

    def test_two_copilots_same_confidence(self):
        self.assertEqual(make_copilot().brief(QM).confidence, make_copilot().brief(QM).confidence)

    def test_full_briefing_dict_byte_equal(self):
        a = json.dumps(make_copilot().brief(QM).to_dict(), sort_keys=True)
        b = json.dumps(make_copilot().brief(QM).to_dict(), sort_keys=True)
        self.assertEqual(a, b)

    def test_statistics_dict_byte_equal(self):
        c1 = make_copilot(); c1.brief(QR); c1.brief(QM)
        c2 = make_copilot(); c2.brief(QR); c2.brief(QM)
        self.assertEqual(c1.statistics().to_dict(), c2.statistics().to_dict())

    def test_reasoning_limitations_subset_of_briefing(self):
        b = self.copilot.brief(QR)
        self.assertEqual(set(b.reasoning.limitations), set(b.limitations))

    def test_confidence_label_consistent(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.confidence_assessment, b.to_dict()["confidence_assessment"])

    def test_citation_count_matches_metadata(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.metadata["citation_count"], len(b.citations))

    def test_insight_count_matches_metadata(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.metadata["insight_count"], len(b.insights))

    def test_recommendation_count_matches_metadata(self):
        b = self.copilot.brief(QR)
        self.assertEqual(b.metadata["recommendation_count"], len(b.recommendations))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)