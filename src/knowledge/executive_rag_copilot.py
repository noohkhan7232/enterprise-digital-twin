"""Executive RAG Copilot -- Week 9, Phase 5.

The executive orchestration tier of the Enterprise Knowledge Layer for the
Enterprise Digital Twin & Decision Intelligence Platform. The Copilot converts a
high-level executive question into a fully grounded, evidence-backed
:class:`ExecutiveBriefing`: an executive summary, key findings, business risks,
structured insights, deterministic recommendations, aggregated citations, a
calibrated confidence assessment, and explicit limitations.

**This is not an LLM chatbot.** There is no generative model and no invented
content anywhere. The Copilot is a deterministic *orchestrator*: it classifies
the executive question, expands it into focused sub-questions, delegates every
piece of reasoning and retrieval to the frozen Week 9 Phase 4
:class:`KnowledgeAgent` (which in turn composes the Phase 3 retrieval engine and
Phase 2 search), and then aggregates and frames those grounded results. Every
factual sentence in a briefing is selected verbatim from Knowledge Agent
evidence; citations are reused directly from the agent -- no citation, retrieval,
or knowledge-reasoning logic is reimplemented here.

This module is **additive**: it imports and composes the existing modules and
modifies, renames, and duplicates nothing.

Pipeline
--------
``executive question`` -> classification -> knowledge-agent invocation
(sub-questions) -> evidence aggregation -> insight extraction ->
recommendation generation -> briefing assembly -> confidence assessment ->
``ExecutiveBriefing``

Engineering constraints honoured
--------------------------------
Pure Python + NumPy. Deterministic (stable hashing, deterministic tie-breaking,
injectable clock). Dependency injection of the Knowledge Agent, classifier,
config, and clock. Registry pattern for executive question types, sub-question
templates, and recommendation templates. Frozen dataclasses with full
``to_dict`` / ``from_dict`` JSON round trips. No OpenAI / HuggingFace / external
API / LLM. No hallucination. 100% backward compatible with Weeks 1-9 Phase 4.

CLI demo
--------
``python src/knowledge/executive_rag_copilot.py --demo``
"""
from __future__ import annotations

import argparse
import json
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

# --- Phase 2/3/4 integration ----------------------------------------------
try:  # pragma: no cover - exercised via both import styles
    from .vector_search_engine import Registry, stable_hash, tokenize
    from .retrieval_intelligence import QueryNormalizer
    from .knowledge_agent import (
        AgentConfig,
        KnowledgeAgent,
        KnowledgeCitation,
        KnowledgeResponse,
    )
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vector_search_engine import Registry, stable_hash, tokenize  # type: ignore
    from retrieval_intelligence import QueryNormalizer  # type: ignore
    from knowledge_agent import (  # type: ignore
        AgentConfig,
        KnowledgeAgent,
        KnowledgeCitation,
        KnowledgeResponse,
    )

__all__ = [
    # exceptions
    "ExecutiveCopilotError",
    "EmptyExecutiveQuestionError",
    "InvalidQuestionTypeError",
    "MissingEvidenceError",
    "DuplicateRecommendationError",
    "DuplicateCitationError",
    # registries / types
    "ExecutiveQuestionType",
    "ExecutiveQuestionRule",
    "EXECUTIVE_QUESTION_TYPE_REGISTRY",
    "SUBQUESTION_REGISTRY",
    "RECOMMENDATION_TEMPLATE_REGISTRY",
    "register_executive_question_type",
    "register_subquestions",
    "register_recommendation_template",
    "ExecutiveQuestionClassifier",
    # config
    "CopilotConfig",
    # dataclasses
    "ExecutiveQuestion",
    "ExecutiveCitation",
    "ExecutiveInsight",
    "ExecutiveRecommendation",
    "ExecutiveReasoning",
    "ExecutiveBriefing",
    "ExecutiveCopilotStatistics",
    # copilot
    "ExecutiveRAGCopilot",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ExecutiveCopilotError(Exception):
    """Base class for every error raised by the Executive RAG Copilot."""


class EmptyExecutiveQuestionError(ExecutiveCopilotError):
    """Raised when an executive question is empty or has no usable tokens."""


class InvalidQuestionTypeError(ExecutiveCopilotError):
    """Raised when an explicitly supplied question type is not registered."""


class MissingEvidenceError(ExecutiveCopilotError):
    """Raised when no evidence is aggregated and evidence is required."""


class DuplicateRecommendationError(ExecutiveCopilotError):
    """Raised when assembled recommendations contain duplicate actions."""


class DuplicateCitationError(ExecutiveCopilotError):
    """Raised when aggregated citations contain duplicate document ids."""


# ---------------------------------------------------------------------------
# Executive question types
# ---------------------------------------------------------------------------
class ExecutiveQuestionType:
    """Canonical executive question-type identifiers (string constants)."""

    EXECUTIVE_SUMMARY = "executive_summary"
    RISK_ASSESSMENT = "risk_assessment"
    MAINTENANCE_STRATEGY = "maintenance_strategy"
    INVESTMENT_PLANNING = "investment_planning"
    ASSET_HEALTH_REVIEW = "asset_health_review"
    ROOT_CAUSE_REVIEW = "root_cause_review"
    SCENARIO_EVALUATION = "scenario_evaluation"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    STRATEGIC_DECISION_SUPPORT = "strategic_decision_support"


#: Deterministic priority order used to break classification ties.
EXECUTIVE_QUESTION_TYPE_ORDER: Tuple[str, ...] = (
    ExecutiveQuestionType.RISK_ASSESSMENT,
    ExecutiveQuestionType.ROOT_CAUSE_REVIEW,
    ExecutiveQuestionType.ASSET_HEALTH_REVIEW,
    ExecutiveQuestionType.MAINTENANCE_STRATEGY,
    ExecutiveQuestionType.COST_OPTIMIZATION,
    ExecutiveQuestionType.INVESTMENT_PLANNING,
    ExecutiveQuestionType.OPERATIONAL_EFFICIENCY,
    ExecutiveQuestionType.SCENARIO_EVALUATION,
    ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT,
    ExecutiveQuestionType.EXECUTIVE_SUMMARY,
)

#: Deterministic priority label per executive question type.
_TYPE_PRIORITY: Dict[str, str] = {
    ExecutiveQuestionType.RISK_ASSESSMENT: "high",
    ExecutiveQuestionType.ROOT_CAUSE_REVIEW: "high",
    ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT: "high",
    ExecutiveQuestionType.ASSET_HEALTH_REVIEW: "medium",
    ExecutiveQuestionType.MAINTENANCE_STRATEGY: "medium",
    ExecutiveQuestionType.COST_OPTIMIZATION: "medium",
    ExecutiveQuestionType.INVESTMENT_PLANNING: "medium",
    ExecutiveQuestionType.OPERATIONAL_EFFICIENCY: "medium",
    ExecutiveQuestionType.SCENARIO_EVALUATION: "medium",
    ExecutiveQuestionType.EXECUTIVE_SUMMARY: "low",
}


@dataclass(frozen=True)
class ExecutiveQuestionRule:
    """A deterministic rule for recognising an executive question type.

    Attributes:
        name: Canonical type identifier.
        keywords: Single-token cues (``+1`` to score each).
        phrases: Multi-word cues (``+2`` to score each).
        focus: Business focus label associated with the type.
    """

    name: str
    keywords: Tuple[str, ...]
    phrases: Tuple[str, ...]
    focus: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keywords", tuple(self.keywords))
        object.__setattr__(self, "phrases", tuple(self.phrases))

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
            "focus": self.focus,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveQuestionRule":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            keywords=tuple(data.get("keywords", ())),
            phrases=tuple(data.get("phrases", ())),
            focus=data["focus"],
        )


EXECUTIVE_QUESTION_TYPE_REGISTRY = Registry("executive-question-type")
SUBQUESTION_REGISTRY = Registry("subquestion-template")
RECOMMENDATION_TEMPLATE_REGISTRY = Registry("recommendation-template")


def register_executive_question_type(rule: ExecutiveQuestionRule) -> ExecutiveQuestionRule:
    """Register an :class:`ExecutiveQuestionRule` under its name."""
    EXECUTIVE_QUESTION_TYPE_REGISTRY.register(rule.name, rule, overwrite=True)
    return rule


def register_subquestions(question_type: str, templates: Sequence[str]) -> None:
    """Register sub-question templates for an executive question type."""
    SUBQUESTION_REGISTRY.register(question_type, tuple(templates), overwrite=True)


def register_recommendation_template(question_type: str, template: str) -> None:
    """Register the recommendation-action template for a question type."""
    RECOMMENDATION_TEMPLATE_REGISTRY.register(question_type, str(template), overwrite=True)


# Default rule set ----------------------------------------------------------
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.EXECUTIVE_SUMMARY,
    ("summary", "overview", "brief", "briefing", "status", "snapshot"),
    ("executive summary", "high level overview"),
    "executive overview",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.RISK_ASSESSMENT,
    ("risk", "risks", "hazard", "hazards", "exposure", "threat", "vulnerability", "danger"),
    ("risk assessment", "risk exposure"),
    "risk exposure",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.MAINTENANCE_STRATEGY,
    ("maintenance", "servicing", "upkeep", "preventive", "overhaul", "repair"),
    ("maintenance strategy", "maintenance plan"),
    "maintenance strategy",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.INVESTMENT_PLANNING,
    ("investment", "capital", "capex", "budget", "funding", "spend", "allocation"),
    ("investment plan", "capital allocation"),
    "investment planning",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.ASSET_HEALTH_REVIEW,
    ("health", "condition", "degradation", "wear", "reliability", "integrity"),
    ("asset health", "health review", "condition assessment"),
    "asset health",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.ROOT_CAUSE_REVIEW,
    ("root", "cause", "causes", "origin", "reason", "failure"),
    ("root cause", "root cause review", "why did"),
    "root cause",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.SCENARIO_EVALUATION,
    ("scenario", "forecast", "projection", "simulate", "simulation", "outlook"),
    ("what if", "scenario evaluation", "scenario analysis"),
    "scenario evaluation",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.OPERATIONAL_EFFICIENCY,
    ("efficiency", "throughput", "utilization", "productivity", "downtime", "uptime"),
    ("operational efficiency", "operating efficiency"),
    "operational efficiency",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.COST_OPTIMIZATION,
    ("cost", "costs", "optimize", "optimise", "optimization", "savings", "reduce", "expense"),
    ("cost optimization", "cost reduction", "reduce cost"),
    "cost optimization",
))
register_executive_question_type(ExecutiveQuestionRule(
    ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT,
    ("strategic", "strategy", "decision", "prioritize", "prioritise", "roadmap", "tradeoff"),
    ("strategic decision", "decision support"),
    "strategic decision support",
))

register_subquestions(ExecutiveQuestionType.EXECUTIVE_SUMMARY, (
    "What is the overall status of {topic}?",
    "What are the key findings about {topic}?",
))
register_subquestions(ExecutiveQuestionType.RISK_ASSESSMENT, (
    "What are the failure risks for {topic}?",
    "What safety hazards relate to {topic}?",
))
register_subquestions(ExecutiveQuestionType.MAINTENANCE_STRATEGY, (
    "What maintenance is recommended for {topic}?",
    "What is the maintenance schedule for {topic}?",
))
register_subquestions(ExecutiveQuestionType.INVESTMENT_PLANNING, (
    "What investment is justified for {topic}?",
    "What is the cost impact of {topic}?",
))
register_subquestions(ExecutiveQuestionType.ASSET_HEALTH_REVIEW, (
    "What is the asset health of {topic}?",
    "What degradation affects {topic}?",
))
register_subquestions(ExecutiveQuestionType.ROOT_CAUSE_REVIEW, (
    "What is the root cause of {topic}?",
    "What contributes to {topic} failure?",
))
register_subquestions(ExecutiveQuestionType.SCENARIO_EVALUATION, (
    "What scenarios affect {topic}?",
    "What is the forecast for {topic}?",
))
register_subquestions(ExecutiveQuestionType.OPERATIONAL_EFFICIENCY, (
    "What affects the operational efficiency of {topic}?",
    "What downtime relates to {topic}?",
))
register_subquestions(ExecutiveQuestionType.COST_OPTIMIZATION, (
    "What costs relate to {topic}?",
    "How can cost be reduced for {topic}?",
))
register_subquestions(ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT, (
    "What decision factors apply to {topic}?",
    "What are the trade-offs for {topic}?",
))

_REC_TEMPLATES = {
    ExecutiveQuestionType.EXECUTIVE_SUMMARY: "Review the documented status of '{topic}'.",
    ExecutiveQuestionType.RISK_ASSESSMENT: "Mitigate the risk identified for '{topic}'.",
    ExecutiveQuestionType.MAINTENANCE_STRATEGY: "Prioritise condition-based maintenance for '{topic}'.",
    ExecutiveQuestionType.INVESTMENT_PLANNING: "Evaluate capital allocation toward '{topic}'.",
    ExecutiveQuestionType.ASSET_HEALTH_REVIEW: "Monitor the asset health of '{topic}'.",
    ExecutiveQuestionType.ROOT_CAUSE_REVIEW: "Address the documented root cause for '{topic}'.",
    ExecutiveQuestionType.SCENARIO_EVALUATION: "Plan for the evaluated scenario affecting '{topic}'.",
    ExecutiveQuestionType.OPERATIONAL_EFFICIENCY: "Improve operational efficiency related to '{topic}'.",
    ExecutiveQuestionType.COST_OPTIMIZATION: "Pursue cost-reduction opportunities in '{topic}'.",
    ExecutiveQuestionType.STRATEGIC_DECISION_SUPPORT: "Incorporate '{topic}' into the strategic decision.",
}
for _t, _tmpl in _REC_TEMPLATES.items():
    register_recommendation_template(_t, _tmpl)

#: Risk-signalling tokens used to surface business risks from grounded evidence.
_RISK_TOKENS = frozenset({
    "risk", "risks", "failure", "failures", "fault", "faults", "hazard", "hazards",
    "danger", "dangerous", "degradation", "anomaly", "breakdown", "malfunction",
    "downtime", "wear", "defect", "defects", "unplanned",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _priority_label(score: float, base: str = "medium") -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _impact_phrase(priority: str) -> str:
    return {
        "high": "High expected business impact.",
        "medium": "Moderate expected business impact.",
        "low": "Limited expected business impact.",
    }.get(priority, "Moderate expected business impact.")


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "moderate"
    if value >= 0.25:
        return "low"
    return "very low"


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class ExecutiveQuestionClassifier:
    """Deterministic, rule-based executive question classifier (no LLM)."""

    def __init__(
        self,
        order: Sequence[str] = EXECUTIVE_QUESTION_TYPE_ORDER,
        normalizer: Optional[QueryNormalizer] = None,
        max_sub_questions: int = 3,
    ) -> None:
        self._order = tuple(order)
        self._normalizer = normalizer or QueryNormalizer()
        self._max_sub_questions = int(max_sub_questions)

    def topic(self, question: str) -> str:
        """Return a deterministic topic token for ``question``.

        Uses the last normalised (content) token, which most often names the
        subject of an executive question; falls back to ``"the assets"``.
        """
        tokens = self._normalizer.normalize_tokens(question)
        if tokens:
            return tokens[-1]
        return "the assets"

    def classify(
        self, question: str, question_type: Optional[str] = None
    ) -> "ExecutiveQuestion":
        """Classify ``question`` into an :class:`ExecutiveQuestion`.

        Args:
            question: Raw executive question text.
            question_type: Optional explicit type override; must be registered.
        """
        if question_type is not None:
            if not EXECUTIVE_QUESTION_TYPE_REGISTRY.has(question_type):
                raise InvalidQuestionTypeError(
                    f"unknown executive question type '{question_type}'"
                )
            best_name = question_type
            best_score = 1
        else:
            raw_lower = question.lower()
            token_set = set(tokenize(question))
            best_name = ExecutiveQuestionType.EXECUTIVE_SUMMARY
            best_score = 0
            best_index = self._order.index(ExecutiveQuestionType.EXECUTIVE_SUMMARY)
            for name in self._order:
                rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(name)
                score = rule.score(token_set, raw_lower)
                if score <= 0:
                    continue
                idx = self._order.index(name)
                if score > best_score or (score == best_score and idx < best_index):
                    best_name, best_score, best_index = name, score, idx

        rule = EXECUTIVE_QUESTION_TYPE_REGISTRY.get(best_name)
        confidence = min(0.95, 0.35 + 0.2 * best_score) if best_score > 0 else 0.2
        topic = self.topic(question)
        sub_questions = self._build_sub_questions(best_name, question, topic)
        question_id = "ex-" + format(stable_hash(question), "016x")[:12]
        return ExecutiveQuestion(
            question_id=question_id,
            text=question,
            question_type=best_name,
            focus=rule.focus,
            priority=_TYPE_PRIORITY.get(best_name, "medium"),
            confidence=float(confidence),
            sub_questions=sub_questions,
        )

    def _build_sub_questions(
        self, question_type: str, question: str, topic: str
    ) -> Tuple[str, ...]:
        subs: List[str] = [question]
        if SUBQUESTION_REGISTRY.has(question_type):
            for tmpl in SUBQUESTION_REGISTRY.get(question_type):
                subs.append(tmpl.format(topic=topic))
        # Deduplicate, preserve order, cap.
        seen = set()
        out: List[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= self._max_sub_questions:
                break
        return tuple(out)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutiveQuestion:
    """A classified executive question with derived sub-questions.

    Attributes:
        question_id: Deterministic identifier derived from the text.
        text: Raw executive question.
        question_type: Canonical :class:`ExecutiveQuestionType`.
        focus: Business focus label.
        priority: Triage priority (``high`` / ``medium`` / ``low``).
        confidence: Classification confidence in ``[0, 1]``.
        sub_questions: Focused sub-questions delegated to the Knowledge Agent.
    """

    question_id: str
    text: str
    question_type: str
    focus: str
    priority: str
    confidence: float
    sub_questions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", str(self.question_id))
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "question_type", str(self.question_type))
        object.__setattr__(self, "focus", str(self.focus))
        object.__setattr__(self, "priority", str(self.priority))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "sub_questions", tuple(str(s) for s in self.sub_questions))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "question_id": self.question_id,
            "text": self.text,
            "question_type": self.question_type,
            "focus": self.focus,
            "priority": self.priority,
            "confidence": self.confidence,
            "sub_questions": list(self.sub_questions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveQuestion":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            question_id=data["question_id"],
            text=data["text"],
            question_type=data["question_type"],
            focus=data["focus"],
            priority=data["priority"],
            confidence=float(data.get("confidence", 0.0)),
            sub_questions=tuple(data.get("sub_questions", ())),
        )


@dataclass(frozen=True)
class ExecutiveCitation:
    """An executive-level citation adapted from a Knowledge Agent citation.

    The citation fields mirror :class:`KnowledgeCitation` (which is reused, not
    reimplemented) plus the sub-question that surfaced the evidence.

    Attributes:
        document_id: Cited document identifier.
        title: Document title if available, else ``None``.
        rank: Retrieval rank.
        confidence: Citation confidence in ``[0, 1]``.
        relevance_score: Underlying retrieval relevance.
        snippet: Grounded evidence text.
        source_question: The sub-question that produced this citation.
    """

    document_id: str
    title: Optional[str]
    rank: int
    confidence: float
    relevance_score: float
    snippet: str = ""
    source_question: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(self, "title", None if self.title is None else str(self.title))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "relevance_score", float(self.relevance_score))
        object.__setattr__(self, "snippet", str(self.snippet))
        object.__setattr__(self, "source_question", str(self.source_question))

    @classmethod
    def from_knowledge_citation(
        cls, citation: KnowledgeCitation, source_question: str = ""
    ) -> "ExecutiveCitation":
        """Adapt a Phase 4 :class:`KnowledgeCitation` (reuse, do not duplicate)."""
        return cls(
            document_id=citation.document_id,
            title=citation.title,
            rank=citation.rank,
            confidence=citation.confidence,
            relevance_score=citation.relevance_score,
            snippet=citation.snippet,
            source_question=source_question,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "rank": self.rank,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "snippet": self.snippet,
            "source_question": self.source_question,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveCitation":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            document_id=data["document_id"],
            title=data.get("title"),
            rank=int(data["rank"]),
            confidence=float(data["confidence"]),
            relevance_score=float(data["relevance_score"]),
            snippet=str(data.get("snippet", "")),
            source_question=str(data.get("source_question", "")),
        )


@dataclass(frozen=True)
class ExecutiveInsight:
    """A structured, evidence-grounded executive insight.

    Attributes:
        title: Insight title (the cited document's title or id).
        summary: Grounded summary text (verbatim evidence snippet).
        business_impact: Deterministic business-impact framing.
        supporting_evidence: Document ids backing the insight.
        priority: ``high`` / ``medium`` / ``low``.
        confidence: Insight confidence in ``[0, 1]``.
    """

    title: str
    summary: str
    business_impact: str
    supporting_evidence: Tuple[str, ...]
    priority: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(self, "business_impact", str(self.business_impact))
        object.__setattr__(self, "supporting_evidence", tuple(str(s) for s in self.supporting_evidence))
        object.__setattr__(self, "priority", str(self.priority))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "title": self.title,
            "summary": self.summary,
            "business_impact": self.business_impact,
            "supporting_evidence": list(self.supporting_evidence),
            "priority": self.priority,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveInsight":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            title=data["title"],
            summary=data["summary"],
            business_impact=data["business_impact"],
            supporting_evidence=tuple(data.get("supporting_evidence", ())),
            priority=data["priority"],
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class ExecutiveRecommendation:
    """A deterministic, evidence-backed executive recommendation.

    The recommended action is a deterministic template instantiated on a topic
    drawn from cited evidence; the rationale is verbatim grounded evidence. No
    information is invented.

    Attributes:
        recommended_action: The recommended action (template + evidence topic).
        business_rationale: Grounded rationale (verbatim evidence snippet).
        expected_impact: Deterministic impact framing from priority.
        supporting_citations: Document ids backing the recommendation.
        priority: ``high`` / ``medium`` / ``low``.
        confidence: Recommendation confidence in ``[0, 1]``.
    """

    recommended_action: str
    business_rationale: str
    expected_impact: str
    supporting_citations: Tuple[str, ...]
    priority: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "recommended_action", str(self.recommended_action))
        object.__setattr__(self, "business_rationale", str(self.business_rationale))
        object.__setattr__(self, "expected_impact", str(self.expected_impact))
        object.__setattr__(self, "supporting_citations", tuple(str(s) for s in self.supporting_citations))
        object.__setattr__(self, "priority", str(self.priority))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "recommended_action": self.recommended_action,
            "business_rationale": self.business_rationale,
            "expected_impact": self.expected_impact,
            "supporting_citations": list(self.supporting_citations),
            "priority": self.priority,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveRecommendation":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            recommended_action=data["recommended_action"],
            business_rationale=data["business_rationale"],
            expected_impact=data["expected_impact"],
            supporting_citations=tuple(data.get("supporting_citations", ())),
            priority=data["priority"],
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class ExecutiveReasoning:
    """Structured reasoning trace for an executive briefing.

    Attributes:
        aggregation_summary: How evidence across sub-questions was aggregated.
        evidence_strength: Mean citation relevance, in ``[0, 1]``.
        coverage: Mean Knowledge Agent coverage across sub-questions.
        confidence: Overall briefing confidence, in ``[0, 1]``.
        sub_question_count: Number of sub-questions delegated to the agent.
        agent_confidence: Mean Knowledge Agent confidence across sub-questions.
        limitations: Caveats aggregated from the agent and the copilot.
    """

    aggregation_summary: str
    evidence_strength: float
    coverage: float
    confidence: float
    sub_question_count: int
    agent_confidence: float
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregation_summary", str(self.aggregation_summary))
        object.__setattr__(self, "evidence_strength", float(self.evidence_strength))
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "sub_question_count", int(self.sub_question_count))
        object.__setattr__(self, "agent_confidence", float(self.agent_confidence))
        object.__setattr__(self, "limitations", tuple(str(l) for l in self.limitations))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "aggregation_summary": self.aggregation_summary,
            "evidence_strength": self.evidence_strength,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "sub_question_count": self.sub_question_count,
            "agent_confidence": self.agent_confidence,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveReasoning":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            aggregation_summary=data["aggregation_summary"],
            evidence_strength=float(data.get("evidence_strength", 0.0)),
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            sub_question_count=int(data.get("sub_question_count", 0)),
            agent_confidence=float(data.get("agent_confidence", 0.0)),
            limitations=tuple(data.get("limitations", ())),
        )


@dataclass(frozen=True)
class ExecutiveBriefing:
    """The complete grounded executive briefing (the Copilot's response).

    Attributes:
        question: The classified :class:`ExecutiveQuestion`.
        executive_summary: Grounded summary text.
        key_findings: Grounded key findings (verbatim evidence).
        business_risks: Grounded risk statements (verbatim evidence).
        insights: Structured :class:`ExecutiveInsight` list.
        recommendations: :class:`ExecutiveRecommendation` list.
        citations: Aggregated :class:`ExecutiveCitation` list.
        reasoning: The :class:`ExecutiveReasoning` trace.
        confidence: Overall confidence in ``[0, 1]``.
        limitations: Aggregated limitations.
        metadata: Auxiliary metadata (latency, counts, ...).
    """

    question: ExecutiveQuestion
    executive_summary: str
    key_findings: Tuple[str, ...]
    business_risks: Tuple[str, ...]
    insights: Tuple[ExecutiveInsight, ...]
    recommendations: Tuple[ExecutiveRecommendation, ...]
    citations: Tuple[ExecutiveCitation, ...]
    reasoning: ExecutiveReasoning
    confidence: float
    limitations: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "executive_summary", str(self.executive_summary))
        object.__setattr__(self, "key_findings", tuple(str(k) for k in self.key_findings))
        object.__setattr__(self, "business_risks", tuple(str(r) for r in self.business_risks))
        object.__setattr__(self, "insights", tuple(self.insights))
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "limitations", tuple(str(l) for l in self.limitations))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def recommended_actions(self) -> Tuple[str, ...]:
        """Return the recommended-action strings (the briefing's action list)."""
        return tuple(r.recommended_action for r in self.recommendations)

    @property
    def supporting_evidence(self) -> Tuple[ExecutiveCitation, ...]:
        """Return the aggregated supporting evidence (citations)."""
        return self.citations

    @property
    def confidence_assessment(self) -> str:
        """Return a deterministic confidence label for the briefing."""
        return _confidence_label(self.confidence)

    @property
    def citation_ids(self) -> Tuple[str, ...]:
        """Return the cited document ids."""
        return tuple(c.document_id for c in self.citations)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "question": self.question.to_dict(),
            "executive_summary": self.executive_summary,
            "key_findings": list(self.key_findings),
            "business_risks": list(self.business_risks),
            "insights": [i.to_dict() for i in self.insights],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "citations": [c.to_dict() for c in self.citations],
            "reasoning": self.reasoning.to_dict(),
            "confidence": self.confidence,
            "confidence_assessment": self.confidence_assessment,
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveBriefing":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            question=ExecutiveQuestion.from_dict(data["question"]),
            executive_summary=data["executive_summary"],
            key_findings=tuple(data.get("key_findings", ())),
            business_risks=tuple(data.get("business_risks", ())),
            insights=tuple(ExecutiveInsight.from_dict(i) for i in data.get("insights", [])),
            recommendations=tuple(
                ExecutiveRecommendation.from_dict(r) for r in data.get("recommendations", [])
            ),
            citations=tuple(ExecutiveCitation.from_dict(c) for c in data.get("citations", [])),
            reasoning=ExecutiveReasoning.from_dict(data["reasoning"]),
            confidence=float(data["confidence"]),
            limitations=tuple(data.get("limitations", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExecutiveCopilotStatistics:
    """Cumulative operational statistics for an :class:`ExecutiveRAGCopilot`.

    Attributes:
        briefings_generated: Number of briefings produced.
        average_confidence: Mean briefing confidence.
        average_response_latency_ms: Mean end-to-end briefing latency.
        recommendation_count: Total recommendations produced.
        insight_count: Total insights produced.
        citation_count: Total citations produced.
    """

    briefings_generated: int
    average_confidence: float
    average_response_latency_ms: float
    recommendation_count: int
    insight_count: int
    citation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "briefings_generated", int(self.briefings_generated))
        object.__setattr__(self, "average_confidence", float(self.average_confidence))
        object.__setattr__(self, "average_response_latency_ms", float(self.average_response_latency_ms))
        object.__setattr__(self, "recommendation_count", int(self.recommendation_count))
        object.__setattr__(self, "insight_count", int(self.insight_count))
        object.__setattr__(self, "citation_count", int(self.citation_count))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "briefings_generated": self.briefings_generated,
            "average_confidence": self.average_confidence,
            "average_response_latency_ms": self.average_response_latency_ms,
            "recommendation_count": self.recommendation_count,
            "insight_count": self.insight_count,
            "citation_count": self.citation_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutiveCopilotStatistics":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            briefings_generated=int(data.get("briefings_generated", 0)),
            average_confidence=float(data.get("average_confidence", 0.0)),
            average_response_latency_ms=float(data.get("average_response_latency_ms", 0.0)),
            recommendation_count=int(data.get("recommendation_count", 0)),
            insight_count=int(data.get("insight_count", 0)),
            citation_count=int(data.get("citation_count", 0)),
        )


# ---------------------------------------------------------------------------
# Copilot configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CopilotConfig:
    """Configuration for the Executive RAG Copilot.

    Attributes:
        max_sub_questions: Sub-questions delegated per executive question
            (including the original).
        max_insights: Maximum insights extracted per briefing.
        max_recommendations: Maximum recommendations per briefing.
        max_citations: Maximum aggregated citations per briefing.
        confidence_threshold: Below this confidence, a limitation is recorded.
        require_evidence: When ``True``, a briefing with no evidence raises
            :class:`MissingEvidenceError`.
        agent_config: Optional Phase 4 :class:`AgentConfig` for sub-question
            answering (``None`` uses the agent's default).
    """

    max_sub_questions: int = 3
    max_insights: int = 4
    max_recommendations: int = 3
    max_citations: int = 6
    confidence_threshold: float = 0.5
    require_evidence: bool = False
    agent_config: Optional[AgentConfig] = None

    def __post_init__(self) -> None:
        for attr in ("max_sub_questions", "max_insights", "max_recommendations", "max_citations"):
            if int(getattr(self, attr)) < 0:
                raise ExecutiveCopilotError(f"{attr} must be non-negative")
        if not (0.0 <= float(self.confidence_threshold) <= 1.0):
            raise ExecutiveCopilotError("confidence_threshold must be in [0, 1]")
        object.__setattr__(self, "max_sub_questions", int(self.max_sub_questions))
        object.__setattr__(self, "max_insights", int(self.max_insights))
        object.__setattr__(self, "max_recommendations", int(self.max_recommendations))
        object.__setattr__(self, "max_citations", int(self.max_citations))
        object.__setattr__(self, "confidence_threshold", float(self.confidence_threshold))
        object.__setattr__(self, "require_evidence", bool(self.require_evidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "max_sub_questions": self.max_sub_questions,
            "max_insights": self.max_insights,
            "max_recommendations": self.max_recommendations,
            "max_citations": self.max_citations,
            "confidence_threshold": self.confidence_threshold,
            "require_evidence": self.require_evidence,
            "agent_config": None if self.agent_config is None else self.agent_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CopilotConfig":
        """Reconstruct from :meth:`to_dict` output."""
        ac = data.get("agent_config")
        return cls(
            max_sub_questions=int(data.get("max_sub_questions", 3)),
            max_insights=int(data.get("max_insights", 4)),
            max_recommendations=int(data.get("max_recommendations", 3)),
            max_citations=int(data.get("max_citations", 6)),
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            require_evidence=bool(data.get("require_evidence", False)),
            agent_config=None if ac is None else AgentConfig.from_dict(ac),
        )


# ---------------------------------------------------------------------------
# Executive RAG Copilot
# ---------------------------------------------------------------------------
class ExecutiveRAGCopilot:
    """Deterministic executive orchestration layer over the Knowledge Agent.

    The Copilot coordinates existing components -- it never reimplements
    retrieval, citations, or knowledge reasoning. It is dependency-injected with
    a Phase 4 :class:`KnowledgeAgent`, an :class:`ExecutiveQuestionClassifier`,
    a :class:`CopilotConfig`, and a clock.

    Args:
        knowledge_agent: The injected Phase 4 Knowledge Agent.
        classifier: Executive question classifier.
        config: Copilot configuration.
        clock: Monotonic clock for deterministic latency in tests.
    """

    def __init__(
        self,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        classifier: Optional[ExecutiveQuestionClassifier] = None,
        config: Optional[CopilotConfig] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._agent = knowledge_agent
        self._config = config or CopilotConfig()
        self._classifier = classifier or ExecutiveQuestionClassifier(
            max_sub_questions=self._config.max_sub_questions
        )
        self._clock = clock
        self._n = 0
        self._sum_conf = 0.0
        self._sum_latency = 0.0
        self._sum_recs = 0
        self._sum_insights = 0
        self._sum_citations = 0

    # -- properties ---------------------------------------------------------
    @property
    def knowledge_agent(self) -> Optional[KnowledgeAgent]:
        """Return the injected Knowledge Agent (may be ``None``)."""
        return self._agent

    @property
    def classifier(self) -> ExecutiveQuestionClassifier:
        """Return the executive question classifier."""
        return self._classifier

    @property
    def config(self) -> CopilotConfig:
        """Return the copilot configuration."""
        return self._config

    # -- construction helper ------------------------------------------------
    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: Any,
        config: Optional[CopilotConfig] = None,
        classifier: Optional[ExecutiveQuestionClassifier] = None,
        clock: Callable[[], float] = time.perf_counter,
        **agent_kwargs: Any,
    ) -> "ExecutiveRAGCopilot":
        """Build a Copilot (and its Knowledge Agent) from a knowledge base."""
        agent = KnowledgeAgent.from_knowledge_base(
            knowledge_base, clock=clock, **agent_kwargs
        )
        return cls(knowledge_agent=agent, classifier=classifier, config=config, clock=clock)

    # -- classification only ------------------------------------------------
    def classify(self, question: str, question_type: Optional[str] = None) -> ExecutiveQuestion:
        """Classify ``question`` without generating a briefing (validated)."""
        self._validate_question(question)
        return self._classifier.classify(question, question_type=question_type)

    # -- main pipeline ------------------------------------------------------
    def brief(
        self,
        question: str,
        *,
        question_type: Optional[str] = None,
        config: Optional[CopilotConfig] = None,
    ) -> ExecutiveBriefing:
        """Generate a grounded :class:`ExecutiveBriefing` for ``question``.

        Raises:
            EmptyExecutiveQuestionError: if the question has no usable tokens.
            InvalidQuestionTypeError: if ``question_type`` is not registered.
            MissingEvidenceError: if evidence is required but none is found.
        """
        cfg = config or self._config
        t0 = self._clock()

        # 1. validation + classification
        self._validate_question(question)
        if self._agent is None:
            raise ExecutiveCopilotError("no knowledge agent configured")
        eq = self._classifier.classify(question, question_type=question_type)

        # 2. knowledge-agent invocation across sub-questions
        responses: List[Tuple[str, KnowledgeResponse]] = []
        for sub_q in eq.sub_questions:
            try:
                resp = self._agent.answer(sub_q, config=cfg.agent_config)
            except Exception:
                # A malformed sub-question never breaks the briefing.
                continue
            responses.append((sub_q, resp))

        # 3. evidence aggregation
        citations = self._aggregate_citations(responses, cfg)
        self._validate_citations(citations)
        if not citations and cfg.require_evidence:
            raise MissingEvidenceError("no evidence aggregated for executive question")

        agent_conf = (
            float(np.mean([r.confidence for _, r in responses])) if responses else 0.0
        )
        coverage = (
            float(np.mean([r.reasoning.coverage for _, r in responses])) if responses else 0.0
        )

        # 4. insight extraction
        insights = self._extract_insights(eq, citations, cfg)

        # 5. recommendation generation
        recommendations = self._generate_recommendations(eq, insights, cfg)
        self._validate_recommendations(recommendations)

        # 6. briefing assembly
        key_findings = tuple(i.summary for i in insights)
        business_risks = tuple(
            i.summary for i in insights
            if _RISK_TOKENS & set(tokenize(i.summary))
        )
        confidence = self._compute_confidence(agent_conf, coverage, recommendations, insights)
        limitations = self._build_limitations(responses, citations, coverage, confidence, cfg)
        executive_summary = self._assemble_summary(eq, citations, key_findings, confidence)

        # 7. reasoning
        reasoning = ExecutiveReasoning(
            aggregation_summary=(
                f"Aggregated {len(citations)} unique evidence source(s) from "
                f"{len(responses)} knowledge-agent analyses for focus '{eq.focus}'."
            ),
            evidence_strength=(
                float(np.mean([c.relevance_score for c in citations])) if citations else 0.0
            ),
            coverage=_clamp01(coverage),
            confidence=confidence,
            sub_question_count=len(responses),
            agent_confidence=_clamp01(agent_conf),
            limitations=limitations,
        )

        t1 = self._clock()
        latency_ms = max(0.0, (t1 - t0) * 1000.0)
        metadata = {
            "question_type": eq.question_type,
            "focus": eq.focus,
            "priority": eq.priority,
            "response_latency_ms": latency_ms,
            "sub_questions": list(eq.sub_questions),
            "insight_count": len(insights),
            "recommendation_count": len(recommendations),
            "citation_count": len(citations),
        }
        briefing = ExecutiveBriefing(
            question=eq,
            executive_summary=executive_summary,
            key_findings=key_findings,
            business_risks=business_risks,
            insights=insights,
            recommendations=recommendations,
            citations=citations,
            reasoning=reasoning,
            confidence=confidence,
            limitations=limitations,
            metadata=metadata,
        )

        # statistics
        self._n += 1
        self._sum_conf += confidence
        self._sum_latency += latency_ms
        self._sum_recs += len(recommendations)
        self._sum_insights += len(insights)
        self._sum_citations += len(citations)
        return briefing

    # -- pipeline stages ----------------------------------------------------
    def _validate_question(self, question: Any) -> None:
        if not isinstance(question, str):
            raise EmptyExecutiveQuestionError("executive question must be a string")
        if not tokenize(question):
            raise EmptyExecutiveQuestionError("executive question is empty")

    def _aggregate_citations(
        self,
        responses: Sequence[Tuple[str, KnowledgeResponse]],
        cfg: CopilotConfig,
    ) -> Tuple[ExecutiveCitation, ...]:
        best: Dict[str, ExecutiveCitation] = {}
        for sub_q, resp in responses:
            for kc in resp.citations:
                ec = ExecutiveCitation.from_knowledge_citation(kc, source_question=sub_q)
                existing = best.get(ec.document_id)
                if existing is None or ec.relevance_score > existing.relevance_score:
                    best[ec.document_id] = ec
        ordered = sorted(
            best.values(), key=lambda c: (-c.relevance_score, c.rank, c.document_id)
        )
        return tuple(ordered[: cfg.max_citations])

    def _extract_insights(
        self,
        eq: ExecutiveQuestion,
        citations: Sequence[ExecutiveCitation],
        cfg: CopilotConfig,
    ) -> Tuple[ExecutiveInsight, ...]:
        insights: List[ExecutiveInsight] = []
        for c in citations[: cfg.max_insights]:
            priority = _priority_label(c.relevance_score)
            title = c.title if c.title else c.document_id
            business_impact = (
                f"{priority.capitalize()}-priority evidence for {eq.focus} "
                f"(relevance {c.relevance_score:.2f})."
            )
            insights.append(ExecutiveInsight(
                title=title,
                summary=c.snippet,  # verbatim, grounded
                business_impact=business_impact,
                supporting_evidence=(c.document_id,),
                priority=priority,
                confidence=_clamp01(c.confidence),
            ))
        return tuple(insights)

    def _generate_recommendations(
        self,
        eq: ExecutiveQuestion,
        insights: Sequence[ExecutiveInsight],
        cfg: CopilotConfig,
    ) -> Tuple[ExecutiveRecommendation, ...]:
        template = (
            RECOMMENDATION_TEMPLATE_REGISTRY.get(eq.question_type)
            if RECOMMENDATION_TEMPLATE_REGISTRY.has(eq.question_type)
            else "Review the documented evidence for '{topic}'."
        )
        recs: List[ExecutiveRecommendation] = []
        seen_actions = set()
        for insight in insights[: cfg.max_recommendations]:
            topic = insight.title
            action = template.format(topic=topic)
            if action in seen_actions:
                continue
            seen_actions.add(action)
            recs.append(ExecutiveRecommendation(
                recommended_action=action,
                business_rationale=insight.summary,  # verbatim, grounded
                expected_impact=_impact_phrase(insight.priority),
                supporting_citations=tuple(insight.supporting_evidence),
                priority=insight.priority,
                confidence=insight.confidence,
            ))
        return tuple(recs)

    def _assemble_summary(
        self,
        eq: ExecutiveQuestion,
        citations: Sequence[ExecutiveCitation],
        key_findings: Sequence[str],
        confidence: float,
    ) -> str:
        if not citations:
            return (
                f"No supporting evidence was found in the knowledge base for "
                f"the {eq.focus} question. Confidence is {_confidence_label(confidence)}."
            )
        head = " ".join(key_findings[:2]).strip()
        return (
            f"Executive briefing on {eq.focus}: synthesised {len(citations)} "
            f"evidence source(s) with {_confidence_label(confidence)} confidence. "
            f"{head}"
        ).strip()

    def _compute_confidence(
        self,
        agent_conf: float,
        coverage: float,
        recommendations: Sequence[ExecutiveRecommendation],
        insights: Sequence[ExecutiveInsight],
    ) -> float:
        rec_consistency = self._consistency([set(r.supporting_citations) for r in recommendations])
        ins_consistency = self._consistency([set(i.supporting_evidence) for i in insights])
        overall = (
            0.40 * _clamp01(agent_conf)
            + 0.30 * _clamp01(coverage)
            + 0.15 * rec_consistency
            + 0.15 * ins_consistency
        )
        return _clamp01(overall)

    @staticmethod
    def _consistency(sets: Sequence[set]) -> float:
        if not sets:
            return 0.0
        if len(sets) == 1:
            return 1.0
        sims: List[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                sims.append(_jaccard(sets[i], sets[j]))
        return _clamp01(float(np.mean(sims))) if sims else 0.0

    def _build_limitations(
        self,
        responses: Sequence[Tuple[str, KnowledgeResponse]],
        citations: Sequence[ExecutiveCitation],
        coverage: float,
        confidence: float,
        cfg: CopilotConfig,
    ) -> Tuple[str, ...]:
        limitations: List[str] = []
        seen = set()

        def _add(msg: str) -> None:
            if msg not in seen:
                seen.add(msg)
                limitations.append(msg)

        if not citations:
            _add("No supporting evidence was aggregated for this question.")
        if coverage < 0.5:
            _add("Aggregated evidence covers less than half of the query terms.")
        if len(citations) < 2:
            _add("Briefing relies on a limited number of sources.")
        if confidence < cfg.confidence_threshold:
            _add("Overall confidence is below the configured threshold.")
        # Roll up agent-level limitations (deduplicated).
        for _, resp in responses:
            for lim in resp.reasoning.limitations:
                _add(lim)
        return tuple(limitations)

    def _validate_citations(self, citations: Sequence[ExecutiveCitation]) -> None:
        ids = [c.document_id for c in citations]
        if len(ids) != len(set(ids)):
            raise DuplicateCitationError("duplicate citations detected")

    def _validate_recommendations(self, recs: Sequence[ExecutiveRecommendation]) -> None:
        actions = [r.recommended_action for r in recs]
        if len(actions) != len(set(actions)):
            raise DuplicateRecommendationError("duplicate recommendations detected")

    # -- statistics ---------------------------------------------------------
    def statistics(self) -> ExecutiveCopilotStatistics:
        """Return cumulative :class:`ExecutiveCopilotStatistics`."""
        n = self._n
        if n == 0:
            return ExecutiveCopilotStatistics(0, 0.0, 0.0, 0, 0, 0)
        return ExecutiveCopilotStatistics(
            briefings_generated=n,
            average_confidence=self._sum_conf / n,
            average_response_latency_ms=self._sum_latency / n,
            recommendation_count=self._sum_recs,
            insight_count=self._sum_insights,
            citation_count=self._sum_citations,
        )

    def reset_statistics(self) -> None:
        """Reset all cumulative statistics accumulators to zero."""
        self._n = 0
        self._sum_conf = 0.0
        self._sum_latency = 0.0
        self._sum_recs = 0
        self._sum_insights = 0
        self._sum_citations = 0

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the copilot."""
        return {
            "config": self._config.to_dict(),
            "statistics": self.statistics().to_dict(),
            "agent": None if self._agent is None else self._agent.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], clock: Callable[[], float] = time.perf_counter
    ) -> "ExecutiveRAGCopilot":
        """Reconstruct a copilot from :meth:`to_dict` output."""
        agent = None
        if data.get("agent") is not None:
            agent = KnowledgeAgent.from_dict(data["agent"])
        return cls(
            knowledge_agent=agent,
            config=CopilotConfig.from_dict(data.get("config", {})),
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
    ]
    corpus = []
    for doc_id, title, content, category, tags in rows:
        corpus.append(_DemoDocument(
            document_id=doc_id, content=content, category=category, tags=tags,
            metadata={"text": content, "title": title, "freshness": 0.7},
        ))
    return corpus


def _run_demo() -> int:
    """Run the deterministic command-line demonstration of the copilot."""
    print("=" * 76)
    print("Executive RAG Copilot -- Week 9 Phase 5 -- Demo")
    print("=" * 76)

    copilot = ExecutiveRAGCopilot.from_knowledge_base(_demo_corpus(), clock=lambda: 0.0)
    print(f"\nKnowledge base indexed: "
          f"{copilot.knowledge_agent.retrieval_engine.vector_engine.index.size()} documents.")

    questions = [
        "Give me a risk assessment for the compressor",
        "What is the maintenance strategy for the turbine?",
        "How can we optimize maintenance cost across the fleet?",
        "Provide an executive summary of asset health",
    ]

    for q in questions:
        b = copilot.brief(q)
        print("\n" + "-" * 76)
        print(f"Q: {q}")
        print(f"   type={b.question.question_type}  focus={b.question.focus}  "
              f"priority={b.question.priority}")
        print(f"   confidence={b.confidence:.4f} ({b.confidence_assessment})")
        print(f"   Executive summary: {b.executive_summary}")
        if b.business_risks:
            print(f"   Business risks: {len(b.business_risks)} identified")
        print(f"   Recommendations:")
        for r in b.recommendations:
            print(f"     * {r.recommended_action}")
            print(f"       rationale: {r.business_rationale}")
            print(f"       impact: {r.expected_impact}  priority={r.priority}")
        print(f"   Citations: {list(b.citation_ids)}")
        if b.limitations:
            print(f"   Limitations: {list(b.limitations)}")

    stats = copilot.statistics()
    print("\n" + "=" * 76)
    print("Copilot statistics:")
    print(f"  briefings_generated         = {stats.briefings_generated}")
    print(f"  average_confidence          = {stats.average_confidence:.4f}")
    print(f"  recommendation_count        = {stats.recommendation_count}")
    print(f"  insight_count               = {stats.insight_count}")
    print(f"  citation_count              = {stats.citation_count}")
    print(f"  average_response_latency_ms = {stats.average_response_latency_ms:.4f}")

    b = copilot.brief(questions[0])
    restored = ExecutiveBriefing.from_dict(json.loads(json.dumps(b.to_dict())))
    print(f"\nJSON round-trip preserves briefing: "
          f"{restored.citation_ids == b.citation_ids and restored.executive_summary == b.executive_summary}")
    print("=" * 76)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executive RAG Copilot (Week 9 Phase 5)."
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