"""Enterprise Retrieval Intelligence Layer -- Week 9, Phase 3.

The evidence-preparation layer of the Enterprise Knowledge Layer for the
Enterprise Digital Twin & Decision Intelligence Platform. It sits directly on
top of the Week 9 Phase 2 vector / hybrid search engines and turns a raw user
query into a ranked, scored, fully traced *retrieval package* that the future
Enterprise RAG / Knowledge Agent (Phase 4) can consume without any further
search-time logic.

This module is **additive**. It imports and composes the frozen Phase 2 module
(:mod:`knowledge.vector_search_engine`) and the Phase 1 Enterprise Knowledge
Base through the same documented duck-typed adapter; it does not modify,
re-implement, or duplicate any existing model.

Pipeline
--------
``query`` -> normalize -> hybrid retrieval -> candidate generation ->
candidate filtering -> evidence ranking -> context assembly -> retrieval package

Ranking
-------
Final relevance is a configurable weighted combination of five component
scores: vector, keyword, metadata, category, and freshness (see
:class:`RankingWeights`). Components are optionally min-max normalised across the
candidate set before weighting so heterogeneous signals stay comparable.

Engineering constraints honoured
--------------------------------
Pure Python + NumPy. Deterministic (BLAKE2b-based embeddings inherited from
Phase 2, deterministic tie-breaking, injectable clock so even latency is
reproducible in tests). Frozen dataclasses. Registry pattern for ranking
strategies. Full ``to_dict`` / ``from_dict`` JSON round trips. No LLM, no
OpenAI/HuggingFace, no external APIs, no vector databases.

CLI demo
--------
``python src/knowledge/retrieval_intelligence.py --demo``
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
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

# --- Phase 2 integration ---------------------------------------------------
# Robust import: works both as a package module (``knowledge.retrieval_intelligence``)
# and when executed directly as a script (``python .../retrieval_intelligence.py``).
try:  # pragma: no cover - exercised via both import styles
    from .vector_search_engine import (
        DeterministicEmbeddingEngine,
        HybridConfig,
        HybridSearchEngine,
        HybridSearchResult,
        Registry,
        VectorDocument,
        VectorSearchEngine,
        VectorSearchError,
        VectorSearchResult,
        _discover_documents,
        _doc_category,
        _doc_id,
        _doc_metadata,
        _doc_tags,
        _doc_text,
        tokenize,
    )
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vector_search_engine import (  # type: ignore
        DeterministicEmbeddingEngine,
        HybridConfig,
        HybridSearchEngine,
        HybridSearchResult,
        Registry,
        VectorDocument,
        VectorSearchEngine,
        VectorSearchError,
        VectorSearchResult,
        _discover_documents,
        _doc_category,
        _doc_id,
        _doc_metadata,
        _doc_tags,
        _doc_text,
        tokenize,
    )

__all__ = [
    # exceptions
    "RetrievalError",
    "EmptyQueryError",
    "InvalidQueryError",
    "InvalidRankingError",
    "MissingDocumentError",
    "DuplicateRetrievalError",
    # registry
    "RANKING_REGISTRY",
    "register_ranking_strategy",
    # config / weights
    "QueryNormalizer",
    "RankingWeights",
    "RetrievalConfig",
    # value objects
    "RetrievedDocument",
    "RetrievalContext",
    "RetrievalEvidence",
    "RetrievalStatistics",
    "RetrievalPackage",
    # engine
    "RetrievalEngine",
    # helpers
    "DEFAULT_STOPWORDS",
]

_EPS = 1e-12

#: Compact, deterministic English stop-word set used by query normalisation.
DEFAULT_STOPWORDS: Tuple[str, ...] = (
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "which", "with",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RetrievalError(Exception):
    """Base class for every error raised by the retrieval intelligence layer."""


class EmptyQueryError(RetrievalError):
    """Raised when a query is empty or normalises to no usable tokens."""


class InvalidQueryError(RetrievalError):
    """Raised when a query is not a string (or is ``None``)."""


class InvalidRankingError(RetrievalError):
    """Raised when ranking weights or produced ranks are invalid."""


class MissingDocumentError(RetrievalError):
    """Raised when a generated candidate references an absent document."""


class DuplicateRetrievalError(RetrievalError):
    """Raised when the assembled evidence contains duplicate document ids."""


# ---------------------------------------------------------------------------
# Registry of ranking strategies
# ---------------------------------------------------------------------------
RANKING_REGISTRY = Registry("ranking-strategy")


def register_ranking_strategy(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering a ranking strategy under ``name``.

    A ranking strategy is a callable
    ``(components: Mapping[str, float], weights: RankingWeights) -> float``
    returning a single scalar relevance score.
    """

    def _decorator(func: Callable) -> Callable:
        RANKING_REGISTRY.register(name, func)
        return func

    return _decorator


@register_ranking_strategy("weighted_sum")
def _weighted_sum(components: Mapping[str, float], weights: "RankingWeights") -> float:
    """Default strategy: normalised weighted sum of the five components."""
    w = weights.as_normalized()
    return (
        w.vector * float(components.get("vector", 0.0))
        + w.keyword * float(components.get("keyword", 0.0))
        + w.metadata * float(components.get("metadata", 0.0))
        + w.category * float(components.get("category", 0.0))
        + w.freshness * float(components.get("freshness", 0.0))
    )


@register_ranking_strategy("max")
def _max_strategy(components: Mapping[str, float], weights: "RankingWeights") -> float:
    """Take the maximum weighted component (useful for recall-oriented runs)."""
    w = weights.as_normalized()
    contributions = [
        w.vector * float(components.get("vector", 0.0)),
        w.keyword * float(components.get("keyword", 0.0)),
        w.metadata * float(components.get("metadata", 0.0)),
        w.category * float(components.get("category", 0.0)),
        w.freshness * float(components.get("freshness", 0.0)),
    ]
    return max(contributions) if contributions else 0.0


# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QueryNormalizer:
    """Deterministic query normaliser.

    Cleans a raw query into a canonical token sequence. The transformation is
    pure and order-stable so identical raw queries always normalise identically.

    Attributes:
        lowercase: Lower-case the query (always applied via tokenisation).
        remove_duplicates: Drop repeated tokens, keeping first occurrence.
        remove_stopwords: Drop tokens present in :attr:`stopwords`.
        min_token_length: Drop tokens shorter than this (noise filtering).
        stopwords: Stop-word set used when ``remove_stopwords`` is enabled.
    """

    lowercase: bool = True
    remove_duplicates: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    stopwords: Tuple[str, ...] = DEFAULT_STOPWORDS

    def __post_init__(self) -> None:
        if int(self.min_token_length) < 1:
            raise RetrievalError("min_token_length must be >= 1")
        object.__setattr__(self, "lowercase", bool(self.lowercase))
        object.__setattr__(self, "remove_duplicates", bool(self.remove_duplicates))
        object.__setattr__(self, "remove_stopwords", bool(self.remove_stopwords))
        object.__setattr__(self, "min_token_length", int(self.min_token_length))
        object.__setattr__(self, "stopwords", tuple(self.stopwords))

    def normalize_tokens(self, query: str) -> List[str]:
        """Return the cleaned, ordered token list for ``query``."""
        tokens = tokenize(query)  # already lowercases + strips punctuation
        stop = set(self.stopwords) if self.remove_stopwords else set()
        cleaned: List[str] = []
        seen: set = set()
        for tok in tokens:
            if len(tok) < self.min_token_length:
                continue
            if tok in stop:
                continue
            if self.remove_duplicates and tok in seen:
                continue
            seen.add(tok)
            cleaned.append(tok)
        return cleaned

    def normalize(self, query: str) -> str:
        """Return the normalised query as a single space-joined string."""
        return " ".join(self.normalize_tokens(query))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "lowercase": self.lowercase,
            "remove_duplicates": self.remove_duplicates,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "stopwords": list(self.stopwords),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QueryNormalizer":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            lowercase=bool(data.get("lowercase", True)),
            remove_duplicates=bool(data.get("remove_duplicates", True)),
            remove_stopwords=bool(data.get("remove_stopwords", True)),
            min_token_length=int(data.get("min_token_length", 2)),
            stopwords=tuple(data.get("stopwords", DEFAULT_STOPWORDS)),
        )


# ---------------------------------------------------------------------------
# Ranking weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RankingWeights:
    """Non-negative weights for the five relevance components.

    Attributes:
        vector: Weight on the dense vector similarity component.
        keyword: Weight on the lexical keyword component.
        metadata: Weight on query/metadata token overlap (tags + category).
        category: Weight on requested-category match.
        freshness: Weight on document freshness/recency.
    """

    vector: float = 0.45
    keyword: float = 0.30
    metadata: float = 0.10
    category: float = 0.10
    freshness: float = 0.05

    def __post_init__(self) -> None:
        values = [self.vector, self.keyword, self.metadata, self.category, self.freshness]
        coerced = []
        for v in values:
            fv = float(v)
            if fv < 0.0:
                raise InvalidRankingError("ranking weights must be non-negative")
            coerced.append(fv)
        if sum(coerced) <= _EPS:
            raise InvalidRankingError("at least one ranking weight must be positive")
        object.__setattr__(self, "vector", coerced[0])
        object.__setattr__(self, "keyword", coerced[1])
        object.__setattr__(self, "metadata", coerced[2])
        object.__setattr__(self, "category", coerced[3])
        object.__setattr__(self, "freshness", coerced[4])

    @property
    def total(self) -> float:
        """Return the sum of all five weights."""
        return self.vector + self.keyword + self.metadata + self.category + self.freshness

    def as_normalized(self) -> "RankingWeights":
        """Return an equivalent set of weights summing to 1.0."""
        total = self.total
        return RankingWeights(
            vector=self.vector / total,
            keyword=self.keyword / total,
            metadata=self.metadata / total,
            category=self.category / total,
            freshness=self.freshness / total,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "vector": self.vector,
            "keyword": self.keyword,
            "metadata": self.metadata,
            "category": self.category,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RankingWeights":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            vector=float(data.get("vector", 0.45)),
            keyword=float(data.get("keyword", 0.30)),
            metadata=float(data.get("metadata", 0.10)),
            category=float(data.get("category", 0.10)),
            freshness=float(data.get("freshness", 0.05)),
        )


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for a single retrieval run.

    Attributes:
        top_k: Number of documents returned in the final package.
        candidate_k: Candidate pool size before ranking (``None`` = all docs).
        top_evidence: How many ranked documents are designated *top* evidence;
            the remainder of ``top_k`` become *supporting* evidence.
        min_score: Minimum hybrid (pre-ranking) score for a candidate to survive
            filtering, in ``[0, 1]``.
        category: Optional category restriction applied during filtering.
        tags: Optional tag restriction applied during filtering.
        match_all_tags: Require all tags (``True``) or any tag (``False``).
        ranking_weights: Component weights for evidence ranking.
        ranking_strategy: Registered ranking strategy name.
        normalize_components: Min-max normalise each component across candidates.
        hybrid_config: Weighting for the underlying hybrid retrieval step.
    """

    top_k: int = 5
    candidate_k: Optional[int] = 25
    top_evidence: int = 3
    min_score: float = 0.0
    category: Optional[str] = None
    tags: Tuple[str, ...] = ()
    match_all_tags: bool = False
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    ranking_strategy: str = "weighted_sum"
    normalize_components: bool = True
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)

    def __post_init__(self) -> None:
        if int(self.top_k) < 0:
            raise RetrievalError("top_k must be non-negative")
        if self.candidate_k is not None and int(self.candidate_k) < 0:
            raise RetrievalError("candidate_k must be non-negative or None")
        if int(self.top_evidence) < 0:
            raise RetrievalError("top_evidence must be non-negative")
        if not (0.0 <= float(self.min_score) <= 1.0):
            raise RetrievalError("min_score must lie in [0, 1]")
        if not RANKING_REGISTRY.has(self.ranking_strategy):
            raise InvalidRankingError(
                f"unknown ranking strategy '{self.ranking_strategy}'"
            )
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(
            self, "candidate_k",
            None if self.candidate_k is None else int(self.candidate_k),
        )
        object.__setattr__(self, "top_evidence", int(self.top_evidence))
        object.__setattr__(self, "min_score", float(self.min_score))
        object.__setattr__(
            self, "category", None if self.category is None else str(self.category)
        )
        object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))
        object.__setattr__(self, "match_all_tags", bool(self.match_all_tags))
        object.__setattr__(self, "normalize_components", bool(self.normalize_components))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "top_evidence": self.top_evidence,
            "min_score": self.min_score,
            "category": self.category,
            "tags": list(self.tags),
            "match_all_tags": self.match_all_tags,
            "ranking_weights": self.ranking_weights.to_dict(),
            "ranking_strategy": self.ranking_strategy,
            "normalize_components": self.normalize_components,
            "hybrid_config": self.hybrid_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalConfig":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            top_k=int(data.get("top_k", 5)),
            candidate_k=(
                None if data.get("candidate_k") is None else int(data["candidate_k"])
            ),
            top_evidence=int(data.get("top_evidence", 3)),
            min_score=float(data.get("min_score", 0.0)),
            category=data.get("category"),
            tags=tuple(data.get("tags", ())),
            match_all_tags=bool(data.get("match_all_tags", False)),
            ranking_weights=RankingWeights.from_dict(data.get("ranking_weights", {})),
            ranking_strategy=str(data.get("ranking_strategy", "weighted_sum")),
            normalize_components=bool(data.get("normalize_components", True)),
            hybrid_config=HybridConfig.from_dict(data.get("hybrid_config", {})),
        )


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievedDocument:
    """A single ranked, scored, fully traced retrieval hit.

    Attributes:
        document_id: Source document identifier.
        score: Final combined relevance score (higher is better).
        rank: 1-based rank within the retrieval package.
        vector_score: Vector-similarity component (post-normalisation).
        keyword_score: Keyword component (post-normalisation).
        metadata_score: Query/metadata overlap component.
        category_score: Requested-category match component.
        freshness_score: Freshness/recency component.
        content: Document text snippet used as evidence.
        source: Provenance label (e.g. ``"hybrid"``).
        metadata: Copy of the document metadata.
    """

    document_id: str
    score: float
    rank: int
    vector_score: float
    keyword_score: float
    metadata_score: float
    category_score: float
    freshness_score: float
    content: str = ""
    source: str = "hybrid"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        for attr in (
            "score", "vector_score", "keyword_score", "metadata_score",
            "category_score", "freshness_score",
        ):
            object.__setattr__(self, attr, float(getattr(self, attr)))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def components(self) -> Dict[str, float]:
        """Return the five raw component scores as a mapping."""
        return {
            "vector": self.vector_score,
            "keyword": self.keyword_score,
            "metadata": self.metadata_score,
            "category": self.category_score,
            "freshness": self.freshness_score,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "metadata_score": self.metadata_score,
            "category_score": self.category_score,
            "freshness_score": self.freshness_score,
            "content": self.content,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievedDocument":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            document_id=data["document_id"],
            score=float(data["score"]),
            rank=int(data["rank"]),
            vector_score=float(data.get("vector_score", 0.0)),
            keyword_score=float(data.get("keyword_score", 0.0)),
            metadata_score=float(data.get("metadata_score", 0.0)),
            category_score=float(data.get("category_score", 0.0)),
            freshness_score=float(data.get("freshness_score", 0.0)),
            content=str(data.get("content", "")),
            source=str(data.get("source", "hybrid")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievalContext:
    """Ordered, assembled context ready for downstream consumption.

    Attributes:
        query: Normalised query.
        original_query: Raw query as supplied by the caller.
        documents: Ranked evidence documents in descending relevance.
        confidence: Confidence in the assembled context, in ``[0, 1]``.
        metadata: Assembly metadata (strategy, weights, counts, ...).
    """

    query: str
    original_query: str
    documents: Tuple[RetrievedDocument, ...]
    confidence: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", str(self.query))
        object.__setattr__(self, "original_query", str(self.original_query))
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def document_ids(self) -> Tuple[str, ...]:
        """Return the ordered evidence document ids."""
        return tuple(doc.document_id for doc in self.documents)

    def as_text(self, separator: str = "\n\n") -> str:
        """Assemble the evidence into a numbered context block for RAG prompts."""
        blocks = [
            f"[{doc.rank}] ({doc.document_id}) {doc.content}".rstrip()
            for doc in self.documents
        ]
        return separator.join(blocks)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "query": self.query,
            "original_query": self.original_query,
            "documents": [doc.to_dict() for doc in self.documents],
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalContext":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            query=data["query"],
            original_query=data.get("original_query", data["query"]),
            documents=tuple(
                RetrievedDocument.from_dict(d) for d in data.get("documents", [])
            ),
            confidence=float(data.get("confidence", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievalEvidence:
    """Structured evidence package separating top and supporting documents.

    Attributes:
        query: Normalised query.
        top_documents: Highest-ranked evidence (the most load-bearing).
        supporting_documents: Lower-ranked corroborating evidence.
        confidence_score: Confidence the evidence answers the query, ``[0, 1]``.
        coverage_score: Fraction of query tokens covered by the evidence, ``[0, 1]``.
        reasoning: Reasoning/trace metadata explaining how evidence was selected.
    """

    query: str
    top_documents: Tuple[RetrievedDocument, ...]
    supporting_documents: Tuple[RetrievedDocument, ...]
    confidence_score: float
    coverage_score: float
    reasoning: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", str(self.query))
        object.__setattr__(self, "top_documents", tuple(self.top_documents))
        object.__setattr__(
            self, "supporting_documents", tuple(self.supporting_documents)
        )
        object.__setattr__(self, "confidence_score", float(self.confidence_score))
        object.__setattr__(self, "coverage_score", float(self.coverage_score))
        object.__setattr__(self, "reasoning", MappingProxyType(dict(self.reasoning)))

    @property
    def all_documents(self) -> Tuple[RetrievedDocument, ...]:
        """Return top followed by supporting documents."""
        return tuple(self.top_documents) + tuple(self.supporting_documents)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "query": self.query,
            "top_documents": [d.to_dict() for d in self.top_documents],
            "supporting_documents": [d.to_dict() for d in self.supporting_documents],
            "confidence_score": self.confidence_score,
            "coverage_score": self.coverage_score,
            "reasoning": dict(self.reasoning),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalEvidence":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            query=data["query"],
            top_documents=tuple(
                RetrievedDocument.from_dict(d) for d in data.get("top_documents", [])
            ),
            supporting_documents=tuple(
                RetrievedDocument.from_dict(d)
                for d in data.get("supporting_documents", [])
            ),
            confidence_score=float(data.get("confidence_score", 0.0)),
            coverage_score=float(data.get("coverage_score", 0.0)),
            reasoning=dict(data.get("reasoning", {})),
        )


@dataclass(frozen=True)
class RetrievalStatistics:
    """Operational statistics for a retrieval run.

    Attributes:
        latency_ms: Wall-clock latency in milliseconds (the only field that is
            not value-deterministic; an injectable clock makes it reproducible
            in tests).
        documents_searched: Number of documents scanned during retrieval.
        documents_returned: Number of documents in the final package.
        average_score: Mean final score across returned documents.
        coverage: Query-token coverage of the returned evidence, ``[0, 1]``.
        confidence: Confidence in the returned evidence, ``[0, 1]``.
    """

    latency_ms: float
    documents_searched: int
    documents_returned: int
    average_score: float
    coverage: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "documents_searched", int(self.documents_searched))
        object.__setattr__(self, "documents_returned", int(self.documents_returned))
        object.__setattr__(self, "average_score", float(self.average_score))
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "latency_ms": self.latency_ms,
            "documents_searched": self.documents_searched,
            "documents_returned": self.documents_returned,
            "average_score": self.average_score,
            "coverage": self.coverage,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalStatistics":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            latency_ms=float(data.get("latency_ms", 0.0)),
            documents_searched=int(data.get("documents_searched", 0)),
            documents_returned=int(data.get("documents_returned", 0)),
            average_score=float(data.get("average_score", 0.0)),
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class RetrievalPackage:
    """The complete output of one retrieval run -- the *retrieval package*.

    Bundles the assembled :class:`RetrievalContext`, the structured
    :class:`RetrievalEvidence`, the run :class:`RetrievalStatistics`, and the
    :class:`RetrievalConfig` used to produce them.
    """

    context: RetrievalContext
    evidence: RetrievalEvidence
    statistics: RetrievalStatistics
    config: RetrievalConfig

    @property
    def documents(self) -> Tuple[RetrievedDocument, ...]:
        """Return the ranked context documents."""
        return self.context.documents

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "context": self.context.to_dict(),
            "evidence": self.evidence.to_dict(),
            "statistics": self.statistics.to_dict(),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalPackage":
        """Reconstruct from :meth:`to_dict` output."""
        return cls(
            context=RetrievalContext.from_dict(data["context"]),
            evidence=RetrievalEvidence.from_dict(data["evidence"]),
            statistics=RetrievalStatistics.from_dict(data["statistics"]),
            config=RetrievalConfig.from_dict(data["config"]),
        )


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------
def _min_max(values: Sequence[float]) -> List[float]:
    """Deterministically min-max normalise ``values`` to ``[0, 1]``.

    A constant vector maps to all-``1.0`` (the signal is uninformative but not
    destroyed), matching the convention used by the Phase 2 hybrid engine.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= _EPS:
        return [1.0 for _ in values]
    return [(v - lo) / span for v in values]


def _metadata_tokens(metadata: Mapping[str, Any]) -> set:
    """Collect tokens from tags + category + scalar string metadata (excl. text)."""
    tokens: set = set()
    category = metadata.get("category")
    if category is not None:
        tokens.update(tokenize(str(category)))
    for tag in metadata.get("tags", ()) or ():
        tokens.update(tokenize(str(tag)))
    for key, value in metadata.items():
        if key in ("text", "tags", "category"):
            continue
        if isinstance(value, str):
            tokens.update(tokenize(value))
    return tokens


def _freshness_from_metadata(metadata: Mapping[str, Any]) -> float:
    """Derive a freshness score in ``[0, 1]`` from metadata, defaulting neutral.

    Priority: explicit ``freshness`` in ``[0, 1]`` -> ``age_days`` decay
    (``1 / (1 + age_days / 30)``) -> neutral ``0.5``.
    """
    if "freshness" in metadata:
        try:
            return float(min(1.0, max(0.0, float(metadata["freshness"]))))
        except (TypeError, ValueError):
            return 0.5
    if "age_days" in metadata:
        try:
            age = max(0.0, float(metadata["age_days"]))
            return 1.0 / (1.0 + age / 30.0)
        except (TypeError, ValueError):
            return 0.5
    return 0.5


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


# ---------------------------------------------------------------------------
# Retrieval engine
# ---------------------------------------------------------------------------
class RetrievalEngine:
    """The Retrieval Intelligence pipeline.

    Composes a Phase 2 :class:`HybridSearchEngine` (candidate generation), a
    :class:`QueryNormalizer` (query processing), and a configurable weighted
    ranking strategy (evidence ranking) into a single deterministic pipeline
    that yields a :class:`RetrievalPackage`.

    Args:
        hybrid_engine: The hybrid search engine providing candidates. When
            omitted a fresh one is created from ``vector_engine``.
        vector_engine: Optional vector engine used to build the hybrid engine
            when one is not supplied directly.
        normalizer: Query normaliser (defaults to :class:`QueryNormalizer`).
        config: Default :class:`RetrievalConfig` (overridable per call).
        content_lookup: Optional ``{document_id: content}`` mapping used to
            attach richer evidence text. When absent, content falls back to the
            indexed metadata ``text`` field.
        clock: Monotonic clock returning seconds; injectable for deterministic
            latency in tests. Defaults to :func:`time.perf_counter`.
    """

    def __init__(
        self,
        hybrid_engine: Optional[HybridSearchEngine] = None,
        vector_engine: Optional[VectorSearchEngine] = None,
        normalizer: Optional[QueryNormalizer] = None,
        config: Optional[RetrievalConfig] = None,
        content_lookup: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if hybrid_engine is None:
            base_vector = vector_engine or VectorSearchEngine()
            hybrid_engine = HybridSearchEngine(base_vector)
        self._hybrid = hybrid_engine
        self._normalizer = normalizer or QueryNormalizer()
        self._config = config or RetrievalConfig()
        self._content_lookup: Dict[str, str] = dict(content_lookup or {})
        self._clock = clock

    # -- properties ---------------------------------------------------------
    @property
    def hybrid_engine(self) -> HybridSearchEngine:
        """Return the underlying :class:`HybridSearchEngine`."""
        return self._hybrid

    @property
    def vector_engine(self) -> VectorSearchEngine:
        """Return the underlying :class:`VectorSearchEngine`."""
        return self._hybrid.vector_engine

    @property
    def normalizer(self) -> QueryNormalizer:
        """Return the query normaliser."""
        return self._normalizer

    @property
    def config(self) -> RetrievalConfig:
        """Return the default retrieval configuration."""
        return self._config

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: Any,
        embedding_engine: Optional[DeterministicEmbeddingEngine] = None,
        hybrid_config: Optional[HybridConfig] = None,
        config: Optional[RetrievalConfig] = None,
        normalizer: Optional[QueryNormalizer] = None,
        keyword_provider: Optional[Callable[[str], Mapping[str, float]]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "RetrievalEngine":
        """Build a fully wired engine from an ``EnterpriseKnowledgeBase``.

        Documents are embedded and indexed via Phase 2, a hybrid engine is
        constructed, and per-document content is captured for richer evidence.
        No knowledge-base method is modified.
        """
        documents = _discover_documents(knowledge_base)
        vector_engine = VectorSearchEngine(embedding_engine=embedding_engine)
        vector_engine.index_documents(documents)
        hybrid = HybridSearchEngine(
            vector_engine,
            keyword_provider=keyword_provider,
            config=hybrid_config or HybridConfig(),
        )
        content_lookup = {_doc_id(d): _doc_text(d) for d in documents}
        return cls(
            hybrid_engine=hybrid,
            normalizer=normalizer,
            config=config,
            content_lookup=content_lookup,
            clock=clock,
        )

    def index_documents(self, documents: Iterable[Any]) -> None:
        """Embed and index more documents, capturing their content for evidence."""
        for doc in documents:
            self.vector_engine.add_document(doc)
            try:
                self._content_lookup[_doc_id(doc)] = _doc_text(doc)
            except VectorSearchError:
                continue

    # -- content resolution -------------------------------------------------
    def _content_for(self, document_id: str, metadata: Mapping[str, Any]) -> str:
        if document_id in self._content_lookup:
            return self._content_lookup[document_id]
        return str(metadata.get("text", ""))

    # -- pipeline -----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalPackage:
        """Run the full retrieval pipeline for ``query`` and return a package.

        Raises:
            InvalidQueryError: if ``query`` is not a string.
            EmptyQueryError: if ``query`` normalises to no usable tokens.
        """
        cfg = config or self._config
        start = self._clock()

        # 1. validation + normalisation -----------------------------------
        if not isinstance(query, str):
            raise InvalidQueryError("query must be a string")
        original_query = query
        norm_tokens = self._normalizer.normalize_tokens(query)
        if not norm_tokens:
            raise EmptyQueryError("query is empty after normalisation")
        norm_query = " ".join(norm_tokens)
        query_token_set = set(norm_tokens)

        index = self.vector_engine.index
        documents_searched = index.size()

        # 2. hybrid retrieval (candidate pool) -----------------------------
        hybrid_results: List[HybridSearchResult] = self._hybrid.search(
            norm_query, top_k=cfg.candidate_k, config=cfg.hybrid_config
        )

        # 3. candidate generation ------------------------------------------
        candidates = self._generate_candidates(hybrid_results)

        # 4. candidate filtering -------------------------------------------
        candidates = self._filter_candidates(candidates, cfg)

        # 5. evidence ranking ----------------------------------------------
        ranked = self._rank_candidates(candidates, cfg, query_token_set)
        ranked = ranked[: cfg.top_k] if cfg.top_k is not None else ranked

        # internal invariant: no duplicates
        ids = [d.document_id for d in ranked]
        if len(ids) != len(set(ids)):
            raise DuplicateRetrievalError("duplicate documents in ranked evidence")
        if [d.rank for d in ranked] != list(range(1, len(ranked) + 1)):
            raise InvalidRankingError("ranks are not a contiguous 1-based sequence")

        # 6. context assembly ----------------------------------------------
        coverage = self._coverage(ranked, query_token_set)
        confidence = self._confidence(ranked, cfg)
        assembly_meta = {
            "ranking_strategy": cfg.ranking_strategy,
            "ranking_weights": cfg.ranking_weights.as_normalized().to_dict(),
            "normalized_components": cfg.normalize_components,
            "candidate_count": len(candidates),
            "returned_count": len(ranked),
        }
        context = RetrievalContext(
            query=norm_query,
            original_query=original_query,
            documents=tuple(ranked),
            confidence=confidence,
            metadata=assembly_meta,
        )

        # 7. evidence package ----------------------------------------------
        top_docs = tuple(ranked[: cfg.top_evidence])
        supporting = tuple(ranked[cfg.top_evidence:])
        reasoning = {
            "query_tokens": list(norm_tokens),
            "documents_searched": documents_searched,
            "candidate_count": len(candidates),
            "filtered_to": len(ranked),
            "strategy": cfg.ranking_strategy,
            "top_score": ranked[0].score if ranked else 0.0,
        }
        evidence = RetrievalEvidence(
            query=norm_query,
            top_documents=top_docs,
            supporting_documents=supporting,
            confidence_score=confidence,
            coverage_score=coverage,
            reasoning=reasoning,
        )

        # 8. statistics + package ------------------------------------------
        elapsed_ms = max(0.0, (self._clock() - start) * 1000.0)
        avg_score = (
            float(np.mean([d.score for d in ranked])) if ranked else 0.0
        )
        statistics = RetrievalStatistics(
            latency_ms=elapsed_ms,
            documents_searched=documents_searched,
            documents_returned=len(ranked),
            average_score=avg_score,
            coverage=coverage,
            confidence=confidence,
        )
        return RetrievalPackage(
            context=context, evidence=evidence, statistics=statistics, config=cfg
        )

    def retrieve_context(self, query: str, *, config: Optional[RetrievalConfig] = None) -> RetrievalContext:
        """Convenience wrapper returning only the :class:`RetrievalContext`."""
        return self.retrieve(query, config=config).context

    def retrieve_evidence(self, query: str, *, config: Optional[RetrievalConfig] = None) -> RetrievalEvidence:
        """Convenience wrapper returning only the :class:`RetrievalEvidence`."""
        return self.retrieve(query, config=config).evidence

    # -- pipeline stages ----------------------------------------------------
    def _generate_candidates(
        self, hybrid_results: Sequence[HybridSearchResult]
    ) -> List[Dict[str, Any]]:
        index = self.vector_engine.index
        candidates: List[Dict[str, Any]] = []
        for res in hybrid_results:
            if not index.has_document(res.document_id):
                # Defensive: hybrid result must map to an indexed document.
                raise MissingDocumentError(
                    f"candidate '{res.document_id}' is not in the index"
                )
            metadata = dict(index.get_document(res.document_id).metadata)
            candidates.append(
                {
                    "document_id": res.document_id,
                    "hybrid_score": float(res.score),
                    "vector_score": float(res.vector_score),
                    "keyword_score": float(res.keyword_score),
                    "metadata": metadata,
                }
            )
        return candidates

    def _filter_candidates(
        self, candidates: List[Dict[str, Any]], cfg: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        wanted_tags = set(cfg.tags)
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for cand in candidates:
            if cand["document_id"] in seen:  # duplicate removal
                continue
            if cand["hybrid_score"] < cfg.min_score:
                continue
            meta = cand["metadata"]
            if cfg.category is not None and str(meta.get("category")) != cfg.category:
                continue
            if wanted_tags:
                doc_tags = {str(t) for t in meta.get("tags", ())}
                if cfg.match_all_tags:
                    if not wanted_tags.issubset(doc_tags):
                        continue
                elif not (wanted_tags & doc_tags):
                    continue
            seen.add(cand["document_id"])
            out.append(cand)
        return out

    def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        cfg: RetrievalConfig,
        query_tokens: set,
    ) -> List[RetrievedDocument]:
        if not candidates:
            return []

        # Raw component scores per candidate.
        vector_raw = [c["vector_score"] for c in candidates]
        keyword_raw = [c["keyword_score"] for c in candidates]
        metadata_raw: List[float] = []
        category_raw: List[float] = []
        freshness_raw: List[float] = []
        for c in candidates:
            meta = c["metadata"]
            meta_tokens = _metadata_tokens(meta)
            if query_tokens:
                overlap = len(query_tokens & meta_tokens) / len(query_tokens)
            else:
                overlap = 0.0
            metadata_raw.append(overlap)
            if cfg.category is None:
                category_raw.append(1.0)
            else:
                category_raw.append(
                    1.0 if str(meta.get("category")) == cfg.category else 0.0
                )
            freshness_raw.append(_freshness_from_metadata(meta))

        if cfg.normalize_components:
            vector_c = _min_max(vector_raw)
            keyword_c = _min_max(keyword_raw)
            metadata_c = _min_max(metadata_raw)
            freshness_c = _min_max(freshness_raw)
            category_c = category_raw  # already binary/neutral
        else:
            vector_c, keyword_c, metadata_c = vector_raw, keyword_raw, metadata_raw
            category_c, freshness_c = category_raw, freshness_raw

        strategy = RANKING_REGISTRY.get(cfg.ranking_strategy)
        scored: List[Tuple[str, float, Dict[str, float], Dict[str, Any]]] = []
        for i, c in enumerate(candidates):
            components = {
                "vector": vector_c[i],
                "keyword": keyword_c[i],
                "metadata": metadata_c[i],
                "category": category_c[i],
                "freshness": freshness_c[i],
            }
            score = float(strategy(components, cfg.ranking_weights))
            scored.append((c["document_id"], score, components, c["metadata"]))

        # Deterministic ordering: score desc, id asc.
        scored.sort(key=lambda t: (-t[1], t[0]))

        ranked: List[RetrievedDocument] = []
        for rank, (doc_id, score, comps, meta) in enumerate(scored, 1):
            ranked.append(
                RetrievedDocument(
                    document_id=doc_id,
                    score=score,
                    rank=rank,
                    vector_score=comps["vector"],
                    keyword_score=comps["keyword"],
                    metadata_score=comps["metadata"],
                    category_score=comps["category"],
                    freshness_score=comps["freshness"],
                    content=self._content_for(doc_id, meta),
                    source="hybrid",
                    metadata=meta,
                )
            )
        return ranked

    # -- metrics ------------------------------------------------------------
    def _coverage(self, ranked: Sequence[RetrievedDocument], query_tokens: set) -> float:
        if not query_tokens:
            return 0.0
        covered: set = set()
        for doc in ranked:
            covered.update(tokenize(doc.content))
            covered.update(tokenize(str(doc.metadata.get("category", ""))))
            for tag in doc.metadata.get("tags", ()) or ():
                covered.update(tokenize(str(tag)))
        return _clamp01(len(query_tokens & covered) / len(query_tokens))

    def _confidence(self, ranked: Sequence[RetrievedDocument], cfg: RetrievalConfig) -> float:
        if not ranked:
            return 0.0
        head = ranked[: max(1, cfg.top_evidence)]
        mean_top = float(np.mean([d.score for d in head]))
        # Separation bonus: how clearly the top result leads the pack.
        if len(ranked) > 1:
            margin = ranked[0].score - ranked[1].score
        else:
            margin = ranked[0].score
        confidence = 0.7 * mean_top + 0.3 * _clamp01(margin)
        return _clamp01(confidence)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the engine state."""
        return {
            "normalizer": self._normalizer.to_dict(),
            "config": self._config.to_dict(),
            "hybrid": {
                "config": self._hybrid.config.to_dict(),
                "vector_engine": self.vector_engine.to_dict(),
            },
            "content_lookup": dict(self._content_lookup),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetrievalEngine":
        """Reconstruct an engine from :meth:`to_dict` output.

        Note: a custom ``keyword_provider`` and a custom ``clock`` are runtime
        callables and cannot be serialised; the reconstructed engine uses the
        Phase 2 built-in keyword fallback and the default clock.
        """
        vector_engine = VectorSearchEngine.from_dict(data["hybrid"]["vector_engine"])
        hybrid = HybridSearchEngine(
            vector_engine,
            config=HybridConfig.from_dict(data["hybrid"]["config"]),
        )
        return cls(
            hybrid_engine=hybrid,
            normalizer=QueryNormalizer.from_dict(data.get("normalizer", {})),
            config=RetrievalConfig.from_dict(data.get("config", {})),
            content_lookup=dict(data.get("content_lookup", {})),
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
         "Turbine bearing vibration anomaly detected by the asset health engine "
         "indicating elevated failure risk on the high-speed shaft.",
         "predictive-maintenance", ("turbine", "vibration", "bearing"), 0.9),
        ("doc-rul-forecast",
         "Remaining useful life forecast for compressor stage two derived from "
         "degradation trend analysis and Monte Carlo risk simulation.",
         "predictive-maintenance", ("rul", "compressor", "forecast"), 0.8),
        ("doc-fleet-twin",
         "Fleet digital twin aggregates asset health across the energy portfolio "
         "for executive decision intelligence and capital planning.",
         "digital-twin", ("fleet", "twin", "portfolio"), 0.6),
        ("doc-maintenance-policy",
         "Maintenance decision engine recommends condition-based intervention "
         "scheduling to minimise unplanned downtime cost across the plant.",
         "decision", ("maintenance", "policy", "downtime"), 0.5),
        ("doc-scenario-plan",
         "Scenario planning agent simulates capital expenditure trade-offs for "
         "strategic portfolio optimization under demand uncertainty.",
         "decision", ("scenario", "capex", "strategy"), 0.4),
        ("doc-root-cause",
         "Root cause analysis traces compressor failure to lubrication "
         "degradation and recommends corrective maintenance.",
         "predictive-maintenance", ("root-cause", "compressor", "lubrication"), 0.95),
    ]
    corpus = []
    for doc_id, content, category, tags, freshness in rows:
        corpus.append(
            _DemoDocument(
                document_id=doc_id,
                content=content,
                category=category,
                tags=tags,
                metadata={"text": content, "freshness": freshness},
            )
        )
    return corpus


def _run_demo() -> int:
    """Run the deterministic command-line demonstration of the pipeline."""
    print("=" * 72)
    print("Enterprise Retrieval Intelligence Layer -- Week 9 Phase 3 -- Demo")
    print("=" * 72)

    engine = RetrievalEngine.from_knowledge_base(
        _demo_corpus(), clock=lambda: 0.0
    )
    print(f"\nIndexed {engine.vector_engine.index.size()} documents "
          f"at dimension {engine.vector_engine.index.dimension}.")

    query = "What causes compressor failure and remaining useful life?"
    print(f"\nRaw query     : {query!r}")
    print(f"Normalised    : {engine.normalizer.normalize(query)!r}")

    package = engine.retrieve(query)
    ctx, ev, stats = package.context, package.evidence, package.statistics

    print("\nRanked evidence (context):")
    for doc in ctx.documents:
        print(f"  #{doc.rank}  {doc.document_id:<24} score={doc.score:.4f}  "
              f"[vec={doc.vector_score:.2f} kw={doc.keyword_score:.2f} "
              f"meta={doc.metadata_score:.2f} cat={doc.category_score:.2f} "
              f"fresh={doc.freshness_score:.2f}]")

    print(f"\nTop evidence docs    : {[d.document_id for d in ev.top_documents]}")
    print(f"Supporting docs      : {[d.document_id for d in ev.supporting_documents]}")
    print(f"Confidence score     : {ev.confidence_score:.4f}")
    print(f"Coverage score       : {ev.coverage_score:.4f}")

    print("\nStatistics:")
    print(f"  documents_searched = {stats.documents_searched}")
    print(f"  documents_returned = {stats.documents_returned}")
    print(f"  average_score      = {stats.average_score:.4f}")
    print(f"  coverage           = {stats.coverage:.4f}")
    print(f"  confidence         = {stats.confidence:.4f}")
    print(f"  latency_ms         = {stats.latency_ms:.4f} (deterministic clock)")

    print("\nCategory-filtered retrieval (category='predictive-maintenance'):")
    cfg = RetrievalConfig(top_k=3, category="predictive-maintenance")
    for doc in engine.retrieve(query, config=cfg).context.documents:
        print(f"  #{doc.rank}  {doc.document_id:<24} score={doc.score:.4f}")

    restored = RetrievalPackage.from_dict(json.loads(json.dumps(package.to_dict())))
    same = restored.context.document_ids == ctx.document_ids
    print(f"\nJSON round-trip preserves ranking: {same}")

    print("\nAssembled RAG context block (preview):")
    preview = ctx.as_text()
    print(preview[:300] + ("..." if len(preview) > 300 else ""))
    print("=" * 72)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enterprise Retrieval Intelligence Layer (Week 9 Phase 3)."
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