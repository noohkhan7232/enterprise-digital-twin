"""Enterprise Knowledge Agent -- Week 9, Phase 4.

The grounded, explainable answering layer of the Enterprise Knowledge Layer for
the Enterprise Digital Twin & Decision Intelligence Platform. The Knowledge
Agent converts a natural-language question into a :class:`KnowledgeResponse`:
a grounded answer, citations, structured reasoning, a calibrated confidence
score, and -- when confidence is low -- suggested follow-up questions.

**No LLM. No generation.** Every word of every answer is selected verbatim from
the retrieval evidence produced by the Week 9 Phase 3 :class:`RetrievalEngine`.
The agent never invents facts; if there is no evidence, it says so and lowers its
confidence. The whole pipeline is deterministic (stable hashing, deterministic
tie-breaking, injectable clock).

This module is **additive**. It imports and composes the frozen Phase 1
(knowledge base, duck-typed), Phase 2 (vector / hybrid search), and Phase 3
(retrieval intelligence) modules without modifying, renaming, or duplicating any
existing model or public API.

Pipeline
--------
``question`` -> classification -> retrieval -> evidence analysis ->
response generation -> citation assembly -> reasoning -> ``KnowledgeResponse``

Engineering constraints honoured
--------------------------------
Pure Python + NumPy. Deterministic. Frozen dataclasses with full
``to_dict`` / ``from_dict`` JSON round trips. Registry pattern for question
types and follow-up templates. Dependency injection of the retrieval engine and
clock. No LLM / OpenAI / HuggingFace / external API.

CLI demo
--------
``python src/knowledge/knowledge_agent.py --demo``
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

# --- Phase 1/2/3 integration ----------------------------------------------
try:  # pragma: no cover - exercised via both import styles
    from .vector_search_engine import (
        Registry,
        VectorSearchEngine,
        VectorSearchError,
        stable_hash,
        tokenize,
    )
    from .retrieval_intelligence import (
        EmptyQueryError,
        QueryNormalizer,
        RetrievalConfig,
        RetrievalContext,
        RetrievalEngine,
        RetrievalError,
        RetrievalEvidence,
        RetrievalPackage,
        RetrievedDocument,
    )
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vector_search_engine import (  # type: ignore
        Registry,
        VectorSearchEngine,
        VectorSearchError,
        stable_hash,
        tokenize,
    )
    from retrieval_intelligence import (  # type: ignore
        EmptyQueryError,
        QueryNormalizer,
        RetrievalConfig,
        RetrievalContext,
        RetrievalEngine,
        RetrievalError,
        RetrievalEvidence,
        RetrievalPackage,
        RetrievedDocument,
    )

__all__ = [
    # exceptions
    "KnowledgeAgentError",
    "EmptyQuestionError",
    "InvalidQuestionError",
    "MissingRetrievalError",
    "DuplicateCitationError",
    "MissingEvidenceError",
    # registries / question types
    "QUESTION_TYPE_REGISTRY",
    "FOLLOWUP_REGISTRY",
    "QuestionType",
    "QuestionTypeRule",
    "register_question_type",
    "register_followup_templates",
    "QuestionClassifier",
    # config
    "AgentConfig",
    # dataclasses
    "KnowledgeQuestion",
    "KnowledgeAnswer",
    "KnowledgeCitation",
    "KnowledgeReasoning",
    "KnowledgeResponse",
    "KnowledgeAgentStatistics",
    # agent
    "KnowledgeAgent",
]

_EPS = 1e-12
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class KnowledgeAgentError(Exception):
    """Base class for every error raised by the Knowledge Agent."""


class EmptyQuestionError(KnowledgeAgentError):
    """Raised when a question is empty or has no usable tokens."""


class InvalidQuestionError(KnowledgeAgentError):
    """Raised when a question is not a string."""


class MissingRetrievalError(KnowledgeAgentError):
    """Raised when the agent has no retrieval engine to answer with."""


class DuplicateCitationError(KnowledgeAgentError):
    """Raised when assembled citations contain duplicate document ids."""


class MissingEvidenceError(KnowledgeAgentError):
    """Raised when no evidence is available and evidence is required."""


# ---------------------------------------------------------------------------
# Question types & classification
# ---------------------------------------------------------------------------
class QuestionType:
    """Canonical question-type identifiers (string constants)."""

    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    FAILURE_ANALYSIS = "failure_analysis"
    ROOT_CAUSE = "root_cause"
    SAFETY = "safety"
    ASSET_INFORMATION = "asset_information"
    OPERATING_PROCEDURE = "operating_procedure"
    SCENARIO_GUIDANCE = "scenario_guidance"
    EXECUTIVE_SUMMARY = "executive_summary"
    GENERAL_KNOWLEDGE = "general_knowledge"


#: Deterministic priority order used to break classification ties.
QUESTION_TYPE_ORDER: Tuple[str, ...] = (
    QuestionType.SAFETY,
    QuestionType.ROOT_CAUSE,
    QuestionType.FAILURE_ANALYSIS,
    QuestionType.MAINTENANCE,
    QuestionType.INSPECTION,
    QuestionType.OPERATING_PROCEDURE,
    QuestionType.SCENARIO_GUIDANCE,
    QuestionType.ASSET_INFORMATION,
    QuestionType.EXECUTIVE_SUMMARY,
    QuestionType.GENERAL_KNOWLEDGE,
)


@dataclass(frozen=True)
class QuestionTypeRule:
    """A deterministic rule describing how to recognise a question type.

    Attributes:
        name: Canonical question-type identifier (see :class:`QuestionType`).
        keywords: Single-token cues; each match adds 1 to the type score.
        phrases: Multi-word cues matched against the lowercased raw question;
            each match adds 2 to the type score.
        intent: Canonical intent label for the type.
        category: Knowledge-domain category associated with the type.
        priority: ``"high"`` / ``"medium"`` / ``"low"`` triage priority.
        expected_evidence: Evidence cues a good answer is expected to contain.
    """

    name: str
    keywords: Tuple[str, ...]
    phrases: Tuple[str, ...]
    intent: str
    category: str
    priority: str
    expected_evidence: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keywords", tuple(self.keywords))
        object.__setattr__(self, "phrases", tuple(self.phrases))
        object.__setattr__(self, "expected_evidence", tuple(self.expected_evidence))

    def score(self, token_set: set, raw_lower: str) -> int:
        """Return the rule's match score for a tokenised + raw question."""
        kw = sum(1 for k in self.keywords if k in token_set)
        ph = sum(2 for p in self.phrases if p in raw_lower)
        return kw + ph

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "keywords": list(self.keywords),
            "phrases": list(self.phrases),
            "intent": self.intent,
            "category": self.category,
            "priority": self.priority,
            "expected_evidence": list(self.expected_evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuestionTypeRule":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            keywords=tuple(data.get("keywords", ())),
            phrases=tuple(data.get("phrases", ())),
            intent=data["intent"],
            category=data["category"],
            priority=data["priority"],
            expected_evidence=tuple(data.get("expected_evidence", ())),
        )


QUESTION_TYPE_REGISTRY = Registry("question-type")
FOLLOWUP_REGISTRY = Registry("followup-template")


def register_question_type(rule: QuestionTypeRule) -> QuestionTypeRule:
    """Register a :class:`QuestionTypeRule` under its name."""
    QUESTION_TYPE_REGISTRY.register(rule.name, rule, overwrite=True)
    return rule


def register_followup_templates(question_type: str, templates: Sequence[str]) -> None:
    """Register follow-up question templates for ``question_type``."""
    FOLLOWUP_REGISTRY.register(question_type, tuple(templates), overwrite=True)


# Default rule set ----------------------------------------------------------
register_question_type(QuestionTypeRule(
    QuestionType.MAINTENANCE,
    ("maintenance", "maintain", "service", "servicing", "repair", "replace",
     "replacement", "lubricate", "lubrication", "overhaul", "schedule",
     "scheduling", "interval", "preventive"),
    ("maintenance schedule", "preventive maintenance"),
    "maintenance_guidance", "maintenance", "medium",
    ("maintenance", "schedule", "interval"),
))
register_question_type(QuestionTypeRule(
    QuestionType.INSPECTION,
    ("inspect", "inspection", "check", "examine", "examination", "monitor",
     "monitoring", "measure", "measurement", "audit"),
    ("visual inspection",),
    "inspection_guidance", "inspection", "medium",
    ("inspection", "check", "monitor"),
))
register_question_type(QuestionTypeRule(
    QuestionType.FAILURE_ANALYSIS,
    ("failure", "fail", "failed", "fault", "faults", "breakdown", "malfunction",
     "defect", "defects", "anomaly", "degradation"),
    ("failure mode", "failure analysis"),
    "failure_analysis", "failure", "high",
    ("failure", "fault", "anomaly"),
))
register_question_type(QuestionTypeRule(
    QuestionType.ROOT_CAUSE,
    ("cause", "causes", "reason", "reasons", "root", "origin"),
    ("root cause", "why did", "due to", "because of"),
    "root_cause_analysis", "root-cause", "high",
    ("cause", "root", "origin"),
))
register_question_type(QuestionTypeRule(
    QuestionType.SAFETY,
    ("safety", "hazard", "hazards", "danger", "dangerous", "risk", "protective",
     "lockout", "tagout", "ppe", "compliance"),
    ("safety procedure", "lockout tagout"),
    "safety_guidance", "safety", "high",
    ("safety", "hazard", "risk"),
))
register_question_type(QuestionTypeRule(
    QuestionType.ASSET_INFORMATION,
    ("asset", "equipment", "specification", "specifications", "spec", "specs",
     "model", "serial", "rating", "capacity", "manufacturer", "datasheet"),
    ("asset information", "serial number"),
    "asset_information", "asset", "low",
    ("asset", "specification", "rating"),
))
register_question_type(QuestionTypeRule(
    QuestionType.OPERATING_PROCEDURE,
    ("procedure", "procedures", "operate", "operating", "operation", "startup",
     "shutdown", "steps", "instructions", "sop"),
    ("operating procedure", "how to", "step by step", "start up", "shut down"),
    "operating_procedure", "procedure", "medium",
    ("procedure", "steps", "operation"),
))
register_question_type(QuestionTypeRule(
    QuestionType.SCENARIO_GUIDANCE,
    ("scenario", "plan", "planning", "forecast", "projection", "simulate",
     "simulation", "capex", "expenditure", "optimization"),
    ("what if", "scenario planning", "capital expenditure"),
    "scenario_guidance", "scenario", "medium",
    ("scenario", "forecast", "plan"),
))
register_question_type(QuestionTypeRule(
    QuestionType.EXECUTIVE_SUMMARY,
    ("executive", "summary", "overview", "portfolio", "strategic", "strategy",
     "kpi", "kpis", "report", "dashboard"),
    ("executive summary", "strategic overview"),
    "executive_summary", "executive", "low",
    ("summary", "portfolio", "strategic"),
))
register_question_type(QuestionTypeRule(
    QuestionType.GENERAL_KNOWLEDGE,
    (),
    (),
    "general_information", "general", "low",
    (),
))

register_followup_templates(QuestionType.MAINTENANCE, (
    "What is the recommended maintenance interval for {topic}?",
    "Which maintenance procedure applies to {topic}?",
    "Are there preventive maintenance steps documented for {topic}?",
))
register_followup_templates(QuestionType.INSPECTION, (
    "What inspection checklist covers {topic}?",
    "How frequently should {topic} be inspected?",
    "Which measurements are required when inspecting {topic}?",
))
register_followup_templates(QuestionType.FAILURE_ANALYSIS, (
    "What are the known failure modes for {topic}?",
    "What symptoms precede a {topic} failure?",
    "Is there historical failure data for {topic}?",
))
register_followup_templates(QuestionType.ROOT_CAUSE, (
    "What is the documented root cause for {topic}?",
    "Which conditions contribute to {topic}?",
    "What corrective actions address {topic}?",
))
register_followup_templates(QuestionType.SAFETY, (
    "What safety precautions apply to {topic}?",
    "Are there hazard controls documented for {topic}?",
    "What protective equipment is required for {topic}?",
))
register_followup_templates(QuestionType.ASSET_INFORMATION, (
    "What are the specifications for {topic}?",
    "What is the rated capacity of {topic}?",
    "Which asset records mention {topic}?",
))
register_followup_templates(QuestionType.OPERATING_PROCEDURE, (
    "What are the operating steps for {topic}?",
    "Is there a startup procedure for {topic}?",
    "What is the shutdown sequence for {topic}?",
))
register_followup_templates(QuestionType.SCENARIO_GUIDANCE, (
    "What scenarios have been modelled for {topic}?",
    "What is the forecast impact for {topic}?",
    "Which assumptions drive the {topic} scenario?",
))
register_followup_templates(QuestionType.EXECUTIVE_SUMMARY, (
    "What is the executive summary for {topic}?",
    "Which KPIs are relevant to {topic}?",
    "How does {topic} affect the portfolio?",
))
register_followup_templates(QuestionType.GENERAL_KNOWLEDGE, (
    "Can you provide more detail about {topic}?",
    "Which documents describe {topic}?",
    "What additional context is available on {topic}?",
))


class QuestionClassifier:
    """Deterministic, rule-based question classifier (no ML, no LLM).

    Scores every registered :class:`QuestionTypeRule` against the question and
    selects the highest-scoring type, breaking ties via
    :data:`QUESTION_TYPE_ORDER`. Produces an intent, category, priority,
    expected-evidence cues, and a bounded classification confidence.
    """

    def __init__(self, order: Sequence[str] = QUESTION_TYPE_ORDER) -> None:
        self._order = tuple(order)

    def classify(self, question: str) -> "KnowledgeQuestion":
        """Classify ``question`` into a :class:`KnowledgeQuestion`."""
        raw_lower = question.lower()
        token_set = set(tokenize(question))

        best_name = QuestionType.GENERAL_KNOWLEDGE
        best_score = -1
        best_index = len(self._order)
        for name in self._order:
            if not QUESTION_TYPE_REGISTRY.has(name):
                continue
            rule = QUESTION_TYPE_REGISTRY.get(name)
            score = rule.score(token_set, raw_lower)
            idx = self._order.index(name)
            if score > best_score or (score == best_score and idx < best_index):
                if score > 0 or name == QuestionType.GENERAL_KNOWLEDGE:
                    if score > best_score or (score == best_score and idx < best_index):
                        best_name, best_score, best_index = name, score, idx

        rule = QUESTION_TYPE_REGISTRY.get(best_name)
        if best_score > 0:
            confidence = min(0.95, 0.35 + 0.2 * best_score)
        else:
            confidence = 0.2
        question_id = "q-" + format(stable_hash(question), "016x")[:12]
        return KnowledgeQuestion(
            question_id=question_id,
            text=question,
            question_type=rule.name,
            intent=rule.intent,
            category=rule.category,
            priority=rule.priority,
            expected_evidence=rule.expected_evidence,
            confidence=float(confidence),
        )


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnowledgeQuestion:
    """A classified question with inferred understanding.

    Attributes:
        question_id: Deterministic identifier derived from the text.
        text: Raw question text.
        question_type: Canonical :class:`QuestionType`.
        intent: Canonical intent label.
        category: Knowledge-domain category.
        priority: Triage priority (``high`` / ``medium`` / ``low``).
        expected_evidence: Evidence cues expected in a good answer.
        confidence: Classification confidence in ``[0, 1]``.
    """

    question_id: str
    text: str
    question_type: str
    intent: str
    category: str
    priority: str
    expected_evidence: Tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", str(self.question_id))
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "question_type", str(self.question_type))
        object.__setattr__(self, "intent", str(self.intent))
        object.__setattr__(self, "category", str(self.category))
        object.__setattr__(self, "priority", str(self.priority))
        object.__setattr__(self, "expected_evidence", tuple(self.expected_evidence))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "question_id": self.question_id,
            "text": self.text,
            "question_type": self.question_type,
            "intent": self.intent,
            "category": self.category,
            "priority": self.priority,
            "expected_evidence": list(self.expected_evidence),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeQuestion":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            question_id=data["question_id"],
            text=data["text"],
            question_type=data["question_type"],
            intent=data["intent"],
            category=data["category"],
            priority=data["priority"],
            expected_evidence=tuple(data.get("expected_evidence", ())),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class KnowledgeAnswer:
    """A grounded answer assembled verbatim from retrieval evidence.

    Attributes:
        text: The answer text (concatenated evidence sentences).
        supported: Whether the answer is backed by at least one citation.
        source_document_ids: Ids of the documents the answer draws on.
        sentence_count: Number of evidence sentences in the answer.
    """

    text: str
    supported: bool
    source_document_ids: Tuple[str, ...] = ()
    sentence_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(
            self, "source_document_ids", tuple(str(s) for s in self.source_document_ids)
        )
        object.__setattr__(self, "sentence_count", int(self.sentence_count))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "text": self.text,
            "supported": self.supported,
            "source_document_ids": list(self.source_document_ids),
            "sentence_count": self.sentence_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeAnswer":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            text=data["text"],
            supported=bool(data["supported"]),
            source_document_ids=tuple(data.get("source_document_ids", ())),
            sentence_count=int(data.get("sentence_count", 0)),
        )


@dataclass(frozen=True)
class KnowledgeCitation:
    """A single citation backing the answer.

    Attributes:
        document_id: Cited document identifier.
        title: Human-readable title if available in metadata, else ``None``.
        rank: Retrieval rank of the document (1-based).
        confidence: Per-citation confidence in ``[0, 1]``.
        relevance_score: Underlying retrieval relevance score.
        snippet: Evidence text drawn from the document.
    """

    document_id: str
    title: Optional[str]
    rank: int
    confidence: float
    relevance_score: float
    snippet: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(
            self, "title", None if self.title is None else str(self.title)
        )
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "relevance_score", float(self.relevance_score))
        object.__setattr__(self, "snippet", str(self.snippet))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "rank": self.rank,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeCitation":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            document_id=data["document_id"],
            title=data.get("title"),
            rank=int(data["rank"]),
            confidence=float(data["confidence"]),
            relevance_score=float(data["relevance_score"]),
            snippet=str(data.get("snippet", "")),
        )


@dataclass(frozen=True)
class KnowledgeReasoning:
    """Structured, human-readable reasoning trace for an answer.

    Attributes:
        why_selected: Ordered explanations of why each cited document was used.
        evidence_strength: Mean citation relevance, in ``[0, 1]``.
        coverage: Query-token coverage of the evidence, in ``[0, 1]``.
        confidence: Overall answer confidence, in ``[0, 1]``.
        limitations: Caveats deterministically derived from the metrics.
    """

    why_selected: Tuple[str, ...]
    evidence_strength: float
    coverage: float
    confidence: float
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "why_selected", tuple(str(w) for w in self.why_selected))
        object.__setattr__(self, "evidence_strength", float(self.evidence_strength))
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "limitations", tuple(str(l) for l in self.limitations))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "why_selected": list(self.why_selected),
            "evidence_strength": self.evidence_strength,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeReasoning":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            why_selected=tuple(data.get("why_selected", ())),
            evidence_strength=float(data.get("evidence_strength", 0.0)),
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            limitations=tuple(data.get("limitations", ())),
        )


@dataclass(frozen=True)
class KnowledgeResponse:
    """The complete, grounded response to a question.

    Attributes:
        question: The classified :class:`KnowledgeQuestion`.
        answer: The grounded :class:`KnowledgeAnswer`.
        citations: Supporting :class:`KnowledgeCitation` list.
        reasoning: The :class:`KnowledgeReasoning` trace.
        confidence: Overall response confidence, in ``[0, 1]``.
        follow_up_questions: Suggested follow-ups (empty when confident).
        metadata: Auxiliary metadata (latencies, retrieval summary, ...).
    """

    question: KnowledgeQuestion
    answer: KnowledgeAnswer
    citations: Tuple[KnowledgeCitation, ...]
    reasoning: KnowledgeReasoning
    confidence: float
    follow_up_questions: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(
            self, "follow_up_questions", tuple(str(f) for f in self.follow_up_questions)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def citation_ids(self) -> Tuple[str, ...]:
        """Return the cited document ids in order."""
        return tuple(c.document_id for c in self.citations)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "question": self.question.to_dict(),
            "answer": self.answer.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "reasoning": self.reasoning.to_dict(),
            "confidence": self.confidence,
            "follow_up_questions": list(self.follow_up_questions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeResponse":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            question=KnowledgeQuestion.from_dict(data["question"]),
            answer=KnowledgeAnswer.from_dict(data["answer"]),
            citations=tuple(
                KnowledgeCitation.from_dict(c) for c in data.get("citations", [])
            ),
            reasoning=KnowledgeReasoning.from_dict(data["reasoning"]),
            confidence=float(data["confidence"]),
            follow_up_questions=tuple(data.get("follow_up_questions", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class KnowledgeAgentStatistics:
    """Cumulative operational statistics for a :class:`KnowledgeAgent`.

    Attributes:
        questions_answered: Number of questions processed.
        average_confidence: Mean response confidence.
        retrieval_latency_ms: Mean retrieval latency per question.
        response_latency_ms: Mean end-to-end response latency per question.
        citation_count: Total citations generated across all questions.
        coverage_score: Mean evidence coverage across all questions.
    """

    questions_answered: int
    average_confidence: float
    retrieval_latency_ms: float
    response_latency_ms: float
    citation_count: int
    coverage_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "questions_answered", int(self.questions_answered))
        object.__setattr__(self, "average_confidence", float(self.average_confidence))
        object.__setattr__(self, "retrieval_latency_ms", float(self.retrieval_latency_ms))
        object.__setattr__(self, "response_latency_ms", float(self.response_latency_ms))
        object.__setattr__(self, "citation_count", int(self.citation_count))
        object.__setattr__(self, "coverage_score", float(self.coverage_score))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "questions_answered": self.questions_answered,
            "average_confidence": self.average_confidence,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "response_latency_ms": self.response_latency_ms,
            "citation_count": self.citation_count,
            "coverage_score": self.coverage_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeAgentStatistics":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            questions_answered=int(data.get("questions_answered", 0)),
            average_confidence=float(data.get("average_confidence", 0.0)),
            retrieval_latency_ms=float(data.get("retrieval_latency_ms", 0.0)),
            response_latency_ms=float(data.get("response_latency_ms", 0.0)),
            citation_count=int(data.get("citation_count", 0)),
            coverage_score=float(data.get("coverage_score", 0.0)),
        )


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    """Configuration controlling agent behaviour.

    Attributes:
        confidence_threshold: Below this overall confidence, follow-up questions
            are generated.
        max_citations: Maximum number of citations to attach.
        max_answer_sentences: Maximum evidence sentences in an answer.
        max_follow_ups: Maximum follow-up questions to suggest.
        require_evidence: When ``True``, answering with no evidence raises
            :class:`MissingEvidenceError`; otherwise an unsupported response is
            returned.
        retrieval_config: Phase 3 :class:`RetrievalConfig` used per question.
    """

    confidence_threshold: float = 0.5
    max_citations: int = 5
    max_answer_sentences: int = 4
    max_follow_ups: int = 3
    require_evidence: bool = False
    retrieval_config: RetrievalConfig = field(
        default_factory=lambda: RetrievalConfig(top_k=5, top_evidence=3)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.confidence_threshold) <= 1.0):
            raise KnowledgeAgentError("confidence_threshold must be in [0, 1]")
        for attr in ("max_citations", "max_answer_sentences", "max_follow_ups"):
            if int(getattr(self, attr)) < 0:
                raise KnowledgeAgentError(f"{attr} must be non-negative")
        object.__setattr__(self, "confidence_threshold", float(self.confidence_threshold))
        object.__setattr__(self, "max_citations", int(self.max_citations))
        object.__setattr__(self, "max_answer_sentences", int(self.max_answer_sentences))
        object.__setattr__(self, "max_follow_ups", int(self.max_follow_ups))
        object.__setattr__(self, "require_evidence", bool(self.require_evidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "max_citations": self.max_citations,
            "max_answer_sentences": self.max_answer_sentences,
            "max_follow_ups": self.max_follow_ups,
            "require_evidence": self.require_evidence,
            "retrieval_config": self.retrieval_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentConfig":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            max_citations=int(data.get("max_citations", 5)),
            max_answer_sentences=int(data.get("max_answer_sentences", 4)),
            max_follow_ups=int(data.get("max_follow_ups", 3)),
            require_evidence=bool(data.get("require_evidence", False)),
            retrieval_config=RetrievalConfig.from_dict(
                data.get("retrieval_config", {})
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _split_sentences(text: str) -> List[str]:
    """Split ``text`` into trimmed sentences, deterministically."""
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.findall(text)]
    return [p for p in parts if p]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Knowledge agent
# ---------------------------------------------------------------------------
class KnowledgeAgent:
    """Evidence-grounded, deterministic question-answering agent (no LLM).

    Dependency-injected with a Phase 3 :class:`RetrievalEngine`, a
    :class:`QuestionClassifier`, an :class:`AgentConfig`, and a clock. Converts a
    question into a fully traced :class:`KnowledgeResponse` whose every assertion
    is selected verbatim from retrieval evidence.

    Args:
        retrieval_engine: The injected Phase 3 retrieval engine.
        classifier: Question classifier (defaults to :class:`QuestionClassifier`).
        config: Agent configuration (defaults to :class:`AgentConfig`).
        clock: Monotonic clock for deterministic latency in tests.
    """

    def __init__(
        self,
        retrieval_engine: Optional[RetrievalEngine] = None,
        classifier: Optional[QuestionClassifier] = None,
        config: Optional[AgentConfig] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retrieval = retrieval_engine
        self._classifier = classifier or QuestionClassifier()
        self._config = config or AgentConfig()
        self._clock = clock
        # cumulative statistics accumulators
        self._n = 0
        self._sum_conf = 0.0
        self._sum_retr_ms = 0.0
        self._sum_resp_ms = 0.0
        self._sum_citations = 0
        self._sum_coverage = 0.0

    # -- properties ---------------------------------------------------------
    @property
    def retrieval_engine(self) -> Optional[RetrievalEngine]:
        """Return the injected retrieval engine (may be ``None``)."""
        return self._retrieval

    @property
    def classifier(self) -> QuestionClassifier:
        """Return the question classifier."""
        return self._classifier

    @property
    def config(self) -> AgentConfig:
        """Return the agent configuration."""
        return self._config

    # -- construction helper ------------------------------------------------
    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: Any,
        config: Optional[AgentConfig] = None,
        classifier: Optional[QuestionClassifier] = None,
        clock: Callable[[], float] = time.perf_counter,
        **retrieval_kwargs: Any,
    ) -> "KnowledgeAgent":
        """Build an agent (and its Phase 3 retrieval engine) from a knowledge base."""
        retrieval_engine = RetrievalEngine.from_knowledge_base(
            knowledge_base, clock=clock, **retrieval_kwargs
        )
        return cls(
            retrieval_engine=retrieval_engine,
            classifier=classifier,
            config=config,
            clock=clock,
        )

    # -- public API: classification only ------------------------------------
    def classify(self, question: str) -> KnowledgeQuestion:
        """Classify ``question`` without retrieving (validation included)."""
        self._validate_question(question)
        return self._classifier.classify(question)

    # -- public API: full pipeline ------------------------------------------
    def answer(
        self, question: str, *, config: Optional[AgentConfig] = None
    ) -> KnowledgeResponse:
        """Answer ``question`` and return a grounded :class:`KnowledgeResponse`.

        Raises:
            InvalidQuestionError: if ``question`` is not a string.
            EmptyQuestionError: if it has no usable tokens.
            MissingRetrievalError: if no retrieval engine is configured.
            MissingEvidenceError: if evidence is required but none is found.
        """
        cfg = config or self._config
        t_start = self._clock()

        # 1. validation
        self._validate_question(question)
        if self._retrieval is None:
            raise MissingRetrievalError("no retrieval engine configured")

        # 2. classification
        kq = self._classifier.classify(question)

        # 3. retrieval (Phase 3)
        try:
            package = self._retrieval.retrieve(
                question, config=cfg.retrieval_config
            )
        except EmptyQueryError as exc:
            raise EmptyQuestionError(str(exc)) from exc
        retrieval_ms = float(package.statistics.latency_ms)

        # 4. evidence analysis + 5. response generation
        evidence_docs = list(package.context.documents)
        if not evidence_docs and cfg.require_evidence:
            raise MissingEvidenceError("no retrieval evidence for question")

        query_tokens = set(tokenize(package.context.query))
        answer, used_docs = self._generate_answer(evidence_docs, query_tokens, cfg)

        # 6. citation assembly
        citations = self._assemble_citations(used_docs, cfg)
        self._validate_citations(citations)

        # confidence + 7. reasoning
        confidence = self._compute_confidence(package, citations)
        reasoning = self._build_reasoning(
            package, citations, confidence, query_tokens, cfg.confidence_threshold
        )

        # follow-ups
        follow_ups: Tuple[str, ...] = ()
        if confidence < cfg.confidence_threshold:
            follow_ups = self._generate_follow_ups(kq, package, query_tokens, cfg)

        # 8. response
        t_end = self._clock()
        total_ms = max(0.0, (t_end - t_start) * 1000.0)
        response_ms = max(0.0, total_ms - retrieval_ms)
        metadata = {
            "question_type": kq.question_type,
            "intent": kq.intent,
            "priority": kq.priority,
            "retrieval_latency_ms": retrieval_ms,
            "response_latency_ms": response_ms,
            "documents_searched": package.statistics.documents_searched,
            "documents_returned": package.statistics.documents_returned,
        }
        response = KnowledgeResponse(
            question=kq,
            answer=answer,
            citations=citations,
            reasoning=reasoning,
            confidence=confidence,
            follow_up_questions=follow_ups,
            metadata=metadata,
        )

        # statistics accumulation
        self._n += 1
        self._sum_conf += confidence
        self._sum_retr_ms += retrieval_ms
        self._sum_resp_ms += response_ms
        self._sum_citations += len(citations)
        self._sum_coverage += package.evidence.coverage_score
        return response

    # -- pipeline stages ----------------------------------------------------
    def _validate_question(self, question: Any) -> None:
        if not isinstance(question, str):
            raise InvalidQuestionError("question must be a string")
        if not tokenize(question):
            raise EmptyQuestionError("question is empty or has no usable tokens")

    def _generate_answer(
        self,
        evidence_docs: Sequence[RetrievedDocument],
        query_tokens: set,
        cfg: AgentConfig,
    ) -> Tuple[KnowledgeAnswer, List[RetrievedDocument]]:
        """Select evidence sentences verbatim; never generate new text."""
        if not evidence_docs:
            return (
                KnowledgeAnswer(
                    text="No supporting evidence was found in the knowledge base "
                    "for this question.",
                    supported=False,
                    source_document_ids=(),
                    sentence_count=0,
                ),
                [],
            )

        # Score sentences from the top documents by query-token overlap.
        scored: List[Tuple[int, float, int, str, RetrievedDocument]] = []
        for doc in evidence_docs[: cfg.max_citations]:
            sentences = _split_sentences(doc.content)
            for s_idx, sentence in enumerate(sentences):
                s_tokens = set(tokenize(sentence))
                overlap = len(query_tokens & s_tokens)
                scored.append((doc.rank, float(overlap), s_idx, sentence, doc))

        # Prefer sentences with overlap; fall back to the top document's lead.
        with_overlap = [s for s in scored if s[1] > 0]
        if with_overlap:
            # Highest overlap first; deterministic tie-break by (rank, sentence index).
            with_overlap.sort(key=lambda t: (-t[1], t[0], t[2]))
            selected = with_overlap[: cfg.max_answer_sentences]
        else:
            top_doc = evidence_docs[0]
            lead = _split_sentences(top_doc.content)[:1]
            selected = [
                (top_doc.rank, 0.0, i, s, top_doc) for i, s in enumerate(lead)
            ]

        # Present in reading order: by document rank, then sentence index.
        selected_sorted = sorted(selected, key=lambda t: (t[0], t[2]))
        sentences_text = [t[3] for t in selected_sorted]
        used_docs_ordered: List[RetrievedDocument] = []
        seen = set()
        for t in selected_sorted:
            doc = t[4]
            if doc.document_id not in seen:
                seen.add(doc.document_id)
                used_docs_ordered.append(doc)

        text = " ".join(s.rstrip() for s in sentences_text).strip()
        answer = KnowledgeAnswer(
            text=text,
            supported=bool(used_docs_ordered) and bool(text),
            source_document_ids=tuple(d.document_id for d in used_docs_ordered),
            sentence_count=len(sentences_text),
        )
        return answer, used_docs_ordered

    def _assemble_citations(
        self, used_docs: Sequence[RetrievedDocument], cfg: AgentConfig
    ) -> Tuple[KnowledgeCitation, ...]:
        citations: List[KnowledgeCitation] = []
        seen = set()
        for doc in used_docs[: cfg.max_citations]:
            if doc.document_id in seen:
                continue
            seen.add(doc.document_id)
            title = doc.metadata.get("title")
            snippet = self._best_snippet(doc)
            citations.append(
                KnowledgeCitation(
                    document_id=doc.document_id,
                    title=None if title is None else str(title),
                    rank=doc.rank,
                    confidence=_clamp01(doc.score),
                    relevance_score=float(doc.score),
                    snippet=snippet,
                )
            )
        return tuple(citations)

    @staticmethod
    def _best_snippet(doc: RetrievedDocument) -> str:
        sentences = _split_sentences(doc.content)
        return sentences[0] if sentences else doc.content

    def _validate_citations(self, citations: Sequence[KnowledgeCitation]) -> None:
        ids = [c.document_id for c in citations]
        if len(ids) != len(set(ids)):
            raise DuplicateCitationError("duplicate citations detected")

    def _compute_confidence(
        self, package: RetrievalPackage, citations: Sequence[KnowledgeCitation]
    ) -> float:
        retrieval_conf = _clamp01(package.evidence.confidence_score)
        coverage = _clamp01(package.evidence.coverage_score)
        if citations:
            citation_quality = _clamp01(
                float(np.mean([c.relevance_score for c in citations]))
            )
        else:
            citation_quality = 0.0
        consistency = self._evidence_consistency(citations)
        overall = (
            0.40 * retrieval_conf
            + 0.30 * coverage
            + 0.20 * citation_quality
            + 0.10 * consistency
        )
        return _clamp01(overall)

    @staticmethod
    def _evidence_consistency(citations: Sequence[KnowledgeCitation]) -> float:
        if not citations:
            return 0.0
        if len(citations) == 1:
            return 1.0
        token_sets = [set(tokenize(c.snippet)) for c in citations]
        sims: List[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                sims.append(_jaccard(token_sets[i], token_sets[j]))
        return _clamp01(float(np.mean(sims))) if sims else 0.0

    def _build_reasoning(
        self,
        package: RetrievalPackage,
        citations: Sequence[KnowledgeCitation],
        confidence: float,
        query_tokens: set,
        threshold: float,
    ) -> KnowledgeReasoning:
        why: List[str] = []
        for c in citations:
            why.append(
                f"Document '{c.document_id}' selected at rank {c.rank} "
                f"with relevance {c.relevance_score:.3f}."
            )
        if not why:
            why.append("No documents met the retrieval criteria for this question.")

        if citations:
            evidence_strength = _clamp01(
                float(np.mean([c.relevance_score for c in citations]))
            )
        else:
            evidence_strength = 0.0
        coverage = _clamp01(package.evidence.coverage_score)

        limitations: List[str] = []
        if not citations:
            limitations.append("No supporting evidence was retrieved.")
        if coverage < 0.5:
            limitations.append("Evidence covers less than half of the query terms.")
        if len(citations) == 1:
            limitations.append("Answer relies on a single source.")
        if confidence < threshold:
            limitations.append("Overall confidence is below the configured threshold.")
        if package.evidence.confidence_score < 0.3:
            limitations.append("Underlying retrieval confidence is low.")

        return KnowledgeReasoning(
            why_selected=tuple(why),
            evidence_strength=evidence_strength,
            coverage=coverage,
            confidence=confidence,
            limitations=tuple(limitations),
        )

    def _generate_follow_ups(
        self,
        kq: KnowledgeQuestion,
        package: RetrievalPackage,
        query_tokens: set,
        cfg: AgentConfig,
    ) -> Tuple[str, ...]:
        # Topic = first uncovered query token, else category, else first token.
        covered: set = set()
        for doc in package.context.documents:
            covered.update(tokenize(doc.content))
        uncovered = [t for t in sorted(query_tokens) if t not in covered]
        if uncovered:
            topic = uncovered[0]
        elif query_tokens:
            topic = sorted(query_tokens)[0]
        else:
            topic = kq.category

        templates: Tuple[str, ...] = ()
        if FOLLOWUP_REGISTRY.has(kq.question_type):
            templates = FOLLOWUP_REGISTRY.get(kq.question_type)
        elif FOLLOWUP_REGISTRY.has(QuestionType.GENERAL_KNOWLEDGE):
            templates = FOLLOWUP_REGISTRY.get(QuestionType.GENERAL_KNOWLEDGE)

        out: List[str] = []
        seen = set()
        for tmpl in templates:
            q = tmpl.format(topic=topic)
            if q not in seen:
                seen.add(q)
                out.append(q)
            if len(out) >= cfg.max_follow_ups:
                break
        return tuple(out)

    # -- statistics ---------------------------------------------------------
    def statistics(self) -> KnowledgeAgentStatistics:
        """Return cumulative :class:`KnowledgeAgentStatistics`."""
        n = self._n
        if n == 0:
            return KnowledgeAgentStatistics(0, 0.0, 0.0, 0.0, 0, 0.0)
        return KnowledgeAgentStatistics(
            questions_answered=n,
            average_confidence=self._sum_conf / n,
            retrieval_latency_ms=self._sum_retr_ms / n,
            response_latency_ms=self._sum_resp_ms / n,
            citation_count=self._sum_citations,
            coverage_score=self._sum_coverage / n,
        )

    def reset_statistics(self) -> None:
        """Reset all cumulative statistics accumulators to zero."""
        self._n = 0
        self._sum_conf = 0.0
        self._sum_retr_ms = 0.0
        self._sum_resp_ms = 0.0
        self._sum_citations = 0
        self._sum_coverage = 0.0

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the agent (config + retrieval)."""
        return {
            "config": self._config.to_dict(),
            "statistics": self.statistics().to_dict(),
            "retrieval": (
                None if self._retrieval is None else self._retrieval.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], clock: Callable[[], float] = time.perf_counter
    ) -> "KnowledgeAgent":
        """Reconstruct an agent from :meth:`to_dict` output.

        The retrieval engine is rebuilt from its serialised state; runtime-only
        callables (custom keyword provider, clock) revert to defaults.
        """
        retrieval = None
        if data.get("retrieval") is not None:
            retrieval = RetrievalEngine.from_dict(data["retrieval"])
        return cls(
            retrieval_engine=retrieval,
            config=AgentConfig.from_dict(data.get("config", {})),
            clock=clock,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DemoDocument:
    """Fixture mimicking the Phase 1 ``KnowledgeDocument`` surface (demo only)."""

    document_id: str
    content: str
    category: str
    tags: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _demo_corpus() -> List[_DemoDocument]:
    rows = [
        ("doc-turbine-vibration",
         "Turbine Bearing Vibration Advisory",
         "Turbine bearing vibration anomaly was detected by the asset health "
         "engine. Elevated vibration indicates an emerging bearing fault on the "
         "high-speed shaft. Recommended maintenance is to inspect and replace the "
         "affected bearing within the next service interval.",
         "predictive-maintenance", ("turbine", "vibration", "bearing", "maintenance")),
        ("doc-rul-forecast",
         "Compressor RUL Forecast",
         "The remaining useful life of compressor stage two is forecast from the "
         "degradation trend. Monte Carlo simulation estimates limited remaining "
         "life under current load. Continued operation increases the failure risk.",
         "predictive-maintenance", ("rul", "compressor", "forecast", "failure")),
        ("doc-root-cause",
         "Compressor Failure Root Cause",
         "Root cause analysis traced the compressor failure to lubrication "
         "degradation. Insufficient lubrication accelerated bearing wear. The "
         "corrective action is to restore the lubrication schedule and replace "
         "worn components.",
         "predictive-maintenance", ("root-cause", "compressor", "lubrication", "failure")),
        ("doc-safety-lockout",
         "Lockout Tagout Safety Procedure",
         "Before servicing rotating equipment, apply lockout tagout to isolate "
         "energy sources. Personal protective equipment is mandatory. Verify zero "
         "energy state before beginning maintenance.",
         "safety", ("safety", "lockout", "ppe", "procedure")),
        ("doc-fleet-twin",
         "Fleet Digital Twin Overview",
         "The fleet digital twin aggregates asset health across the energy "
         "portfolio. Executives use it for strategic capital planning and "
         "prioritisation of maintenance investment.",
         "digital-twin", ("fleet", "twin", "portfolio", "executive")),
    ]
    corpus = []
    for doc_id, title, content, category, tags in rows:
        corpus.append(
            _DemoDocument(
                document_id=doc_id,
                content=content,
                category=category,
                tags=tags,
                metadata={"text": content, "title": title, "freshness": 0.7},
            )
        )
    return corpus


def _run_demo() -> int:
    """Run the deterministic command-line demonstration of the agent."""
    print("=" * 74)
    print("Enterprise Knowledge Agent -- Week 9 Phase 4 -- Demo")
    print("=" * 74)

    agent = KnowledgeAgent.from_knowledge_base(_demo_corpus(), clock=lambda: 0.0)
    print(f"\nKnowledge base indexed: "
          f"{agent.retrieval_engine.vector_engine.index.size()} documents.")

    questions = [
        "What is the root cause of the compressor failure?",
        "What safety procedure applies before servicing rotating equipment?",
        "What maintenance is recommended for turbine bearing vibration?",
        "What is the remaining useful life of the compressor?",
        "Tell me about quantum teleportation logistics",  # out-of-domain
    ]

    for q in questions:
        resp = agent.answer(q)
        print("\n" + "-" * 74)
        print(f"Q: {q}")
        print(f"   type={resp.question.question_type}  "
              f"intent={resp.question.intent}  priority={resp.question.priority}")
        print(f"   confidence={resp.confidence:.4f}  supported={resp.answer.supported}")
        print(f"   A: {resp.answer.text}")
        if resp.citations:
            print("   Citations:")
            for c in resp.citations:
                title = c.title or "(untitled)"
                print(f"     - [{c.rank}] {c.document_id} \"{title}\" "
                      f"rel={c.relevance_score:.3f} conf={c.confidence:.3f}")
        if resp.reasoning.limitations:
            print(f"   Limitations: {list(resp.reasoning.limitations)}")
        if resp.follow_up_questions:
            print("   Suggested follow-ups:")
            for f in resp.follow_up_questions:
                print(f"     * {f}")

    stats = agent.statistics()
    print("\n" + "=" * 74)
    print("Agent statistics:")
    print(f"  questions_answered  = {stats.questions_answered}")
    print(f"  average_confidence  = {stats.average_confidence:.4f}")
    print(f"  citation_count      = {stats.citation_count}")
    print(f"  coverage_score      = {stats.coverage_score:.4f}")
    print(f"  retrieval_latency_ms= {stats.retrieval_latency_ms:.4f}")
    print(f"  response_latency_ms = {stats.response_latency_ms:.4f}")

    resp = agent.answer(questions[0])
    restored = KnowledgeResponse.from_dict(json.loads(json.dumps(resp.to_dict())))
    print(f"\nJSON round-trip preserves answer: "
          f"{restored.answer.text == resp.answer.text}")
    print("=" * 74)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enterprise Knowledge Agent (Week 9 Phase 4)."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run the deterministic CLI demo."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point. Returns a process exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.demo:
        return _run_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())