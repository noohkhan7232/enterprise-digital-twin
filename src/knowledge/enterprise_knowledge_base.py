#!/usr/bin/env python3
"""Enterprise Knowledge Base — the platform's foundational knowledge layer.

Week-9 Phase-1 introduces enterprise knowledge to a platform that already has
predictive, simulation, decision, and agentic intelligence but no structured
corpus of operational know-how.  This module provides that corpus: a registry of
typed :class:`KnowledgeDocument` records with deterministic keyword, tag,
category, and exact-match search; explainable relevance scoring; document,
category, and fleet collections; corpus statistics with a coverage score; strict
validation; and full JSON serialisation.

It is deliberately a *foundation*.  There is no database, no vector store, no
language model, and no embeddings — all search and scoring are exact, rule-based,
and reproducible.  This is the substrate on which the Week-9 Phase-2 vector
search engine will build: the document contract, the registry, and the scoring
interface defined here are forward-compatible with an embedding layer that ranks
the same documents by semantic similarity rather than lexical overlap.

============================================================================
Capabilities
============================================================================
* :class:`KnowledgeDocument` — the frozen, validated, serialisable record.
* :class:`EnterpriseKnowledgeBase` — register / remove / get / list documents;
  keyword / tag / category / exact search; relevance scoring; collections;
  statistics; serialisation.

============================================================================
Engineering standards
============================================================================
Frozen dataclasses, registry pattern, pure Python and NumPy, deterministic
behaviour, no external services, full validation, JSON-serialisable outputs,
failure-safe tracker integration, and a CLI demo.

Usage::

    from src.knowledge.enterprise_knowledge_base import (
        EnterpriseKnowledgeBase, KnowledgeDocument, KnowledgeCategory,
    )
    kb = EnterpriseKnowledgeBase()
    kb.register_document(KnowledgeDocument(
        document_id="DOC-001", title="Gearbox Bearing Inspection",
        category=KnowledgeCategory.INSPECTION_PROCEDURE.value,
        tags=("bearing", "vibration", "gearbox"),
        content="Procedure for inspecting gearbox bearings for vibration faults."))
    print(kb.search_keywords("bearing vibration")[0].document_id)

CLI::

    python src/knowledge/enterprise_knowledge_base.py --demo
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("enterprise_knowledge_base")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named knowledge-base implementations (platform convention).
KNOWLEDGE_BASE_REGISTRY: dict[str, type] = {}

KB_NAME: Final[str] = "enterprise_knowledge_base"

#: Deterministic stop-word set excluded from tokenisation.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
    "with", "by", "at", "as", "be", "this", "that", "it", "from", "into",
})

_TOKEN_RE: Final[re.Pattern] = re.compile(r"[^a-z0-9]+")
_JSON_SCALARS: Final[tuple[type, ...]] = (str, int, float, bool, type(None))


def _tokenize(text: str) -> list[str]:
    """Tokenise text deterministically (lowercase, split, drop stop-words).

    Args:
        text: Input text.

    Returns:
        A list of content tokens.
    """
    return [t for t in _TOKEN_RE.split(text.lower()) if t and t not in _STOPWORDS]


def _is_json_safe(value: Any) -> bool:
    """Return whether *value* is recursively JSON-serialisable.

    Args:
        value: The value to check.

    Returns:
        ``True`` when the value is composed only of JSON-safe types.
    """
    if isinstance(value, _JSON_SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v)
                   for k, v in value.items())
    return False


# ---------------------------------------------------------------------------
# Knowledge categories
# ---------------------------------------------------------------------------


class KnowledgeCategory(str, Enum):
    """Supported enterprise knowledge categories."""

    MAINTENANCE_MANUAL = "maintenance_manual"
    INSPECTION_PROCEDURE = "inspection_procedure"
    OPERATING_PROCEDURE = "operating_procedure"
    FAILURE_CATALOG = "failure_catalog"
    ROOT_CAUSE_PLAYBOOK = "root_cause_playbook"
    SAFETY_PROCEDURE = "safety_procedure"
    ASSET_SPECIFICATION = "asset_specification"
    ENGINEERING_REPORT = "engineering_report"
    EXECUTIVE_REPORT = "executive_report"
    UNKNOWN = "unknown"


_VALID_CATEGORIES: Final[frozenset[str]] = frozenset(c.value for c in KnowledgeCategory)
_NON_UNKNOWN_CATEGORIES: Final[tuple[str, ...]] = tuple(
    c.value for c in KnowledgeCategory if c != KnowledgeCategory.UNKNOWN)


def normalize_category(value: Any) -> str:
    """Normalise a category to its canonical string value.

    Args:
        value: A :class:`KnowledgeCategory` or a string.

    Returns:
        The canonical category string.

    Raises:
        ValueError: When the category is not recognised.
    """
    if isinstance(value, KnowledgeCategory):
        return value.value
    if isinstance(value, str) and value in _VALID_CATEGORIES:
        return value
    raise ValueError(
        f"invalid category '{value}'; must be one of {sorted(_VALID_CATEGORIES)}")


# ---------------------------------------------------------------------------
# Knowledge document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeDocument:
    """A single enterprise knowledge document.

    Attributes:
        document_id: Unique identifier (non-empty).
        title: Human-readable title (non-empty).
        document_type: Free-form document type label.
        source: Provenance of the document.
        category: One of the :class:`KnowledgeCategory` values.
        tags: Normalised tuple of lowercase tags.
        content: The document body (non-empty).
        created_at: Creation timestamp (ISO string; caller-supplied).
        version: Version label.
        metadata: JSON-safe metadata mapping.
    """

    document_id:   str = ""
    title:         str = ""
    document_type: str = ""
    source:        str = ""
    category:      str = KnowledgeCategory.UNKNOWN.value
    tags:          tuple[str, ...] = ()
    content:       str = ""
    created_at:    str = ""
    version:       str = "1.0"
    metadata:      Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the document.

        Raises:
            ValueError: On empty required fields, invalid category, malformed
                tags, or non-JSON-safe metadata.
        """
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string (empty "
                             "documents are not allowed)")
        object.__setattr__(self, "category", normalize_category(self.category))
        if not isinstance(self.tags, (list, tuple)):
            raise ValueError("tags must be a list or tuple of strings")
        norm_tags = []
        for t in self.tags:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("each tag must be a non-empty string")
            norm_tags.append(t.strip().lower())
        object.__setattr__(self, "tags", tuple(norm_tags))
        for fld in ("document_type", "source", "created_at", "version"):
            if not isinstance(getattr(self, fld), str):
                raise ValueError(f"{fld} must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if not _is_json_safe(self.metadata):
            raise ValueError("metadata must contain only JSON-safe values")
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "source": self.source,
            "category": self.category,
            "tags": list(self.tags),
            "content": self.content,
            "created_at": self.created_at,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeDocument":
        """Reconstruct a document from a dictionary.

        Args:
            data: A mapping produced by :meth:`to_dict`.

        Returns:
            A :class:`KnowledgeDocument`.

        Raises:
            ValueError: When required fields are missing or invalid.
        """
        if not isinstance(data, Mapping):
            raise ValueError("from_dict requires a mapping")
        return cls(
            document_id=data.get("document_id", ""),
            title=data.get("title", ""),
            document_type=data.get("document_type", ""),
            source=data.get("source", ""),
            category=data.get("category", KnowledgeCategory.UNKNOWN.value),
            tags=tuple(data.get("tags", ()) or ()),
            content=data.get("content", ""),
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    # -- cached token views (computed lazily, deterministically) -------

    def token_sets(self) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        """Return (title, content, tag) token sets for scoring.

        Returns:
            A tuple of three frozensets.
        """
        return (
            frozenset(_tokenize(self.title)),
            frozenset(_tokenize(self.content)),
            frozenset(_tokenize(" ".join(self.tags))),
        )


# ---------------------------------------------------------------------------
# Search result, collection, statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A ranked search hit.

    Attributes:
        document_id: The matched document's identifier.
        title: The matched document's title.
        category: The matched document's category.
        score: The relevance score in ``[0, 1]``.
        matched_terms: The query terms that matched.
    """

    document_id:   str
    title:         str
    category:      str
    score:         float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "category": self.category,
            "score": float(self.score),
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class KnowledgeCollection:
    """A named collection of documents.

    Attributes:
        name: The collection name.
        kind: ``"document"``, ``"category"``, or ``"fleet"``.
        document_ids: The member document identifiers (sorted).
        metadata: JSON-safe collection metadata.
    """

    name:         str
    kind:         str
    document_ids: tuple[str, ...]
    metadata:     Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.document_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "document_ids": list(self.document_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KnowledgeStatistics:
    """Aggregate statistics over a knowledge base.

    Attributes:
        document_count: Number of documents.
        category_distribution: ``(category, count)`` pairs, most frequent first.
        tag_distribution: ``(tag, count)`` pairs, most frequent first.
        coverage_score: Fraction of non-unknown categories represented.
        categories_covered: Count of distinct non-unknown categories present.
        distinct_tags: Number of distinct tags.
    """

    document_count:         int
    category_distribution:  tuple[tuple[str, int], ...]
    tag_distribution:       tuple[tuple[str, int], ...]
    coverage_score:         float
    categories_covered:     int
    distinct_tags:          int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "document_count": self.document_count,
            "category_distribution": [[c, n] for c, n in self.category_distribution],
            "tag_distribution": [[t, n] for t, n in self.tag_distribution],
            "coverage_score": float(self.coverage_score),
            "categories_covered": self.categories_covered,
            "distinct_tags": self.distinct_tags,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Configuration for the :class:`EnterpriseKnowledgeBase`.

    Attributes:
        title_weight: Keyword-scoring weight on title matches.
        content_weight: Keyword-scoring weight on content matches.
        tag_weight: Keyword-scoring weight on tag matches.
        keyword_component_weight: Combined-relevance weight on keyword overlap.
        tag_component_weight: Combined-relevance weight on tag overlap.
        category_component_weight: Combined-relevance weight on category match.
        default_top_n: Default number of results returned by ranked search.
    """

    title_weight:              float = 3.0
    content_weight:            float = 1.0
    tag_weight:                float = 2.0
    keyword_component_weight:  float = 0.50
    tag_component_weight:      float = 0.30
    category_component_weight: float = 0.20
    default_top_n:             int = 10

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On non-positive weights or an invalid default.
        """
        for name in ("title_weight", "content_weight", "tag_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if (self.title_weight + self.content_weight + self.tag_weight) <= 0:
            raise ValueError("keyword weights must sum to > 0")
        for name in ("keyword_component_weight", "tag_component_weight",
                     "category_component_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.default_top_n < 1:
            raise ValueError("default_top_n must be >= 1")

    @property
    def keyword_weight_sum(self) -> float:
        """Return the sum of the keyword field weights."""
        return self.title_weight + self.content_weight + self.tag_weight


# ---------------------------------------------------------------------------
# Registry helpers (platform convention)
# ---------------------------------------------------------------------------


def register_knowledge_base(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a knowledge-base implementation by name."""
    def decorator(cls: type) -> type:
        existing = KNOWLEDGE_BASE_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Knowledge base '{name}' already registered to {existing.__name__}")
        KNOWLEDGE_BASE_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return decorator


def build_knowledge_base(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered knowledge base by name."""
    if name not in KNOWLEDGE_BASE_REGISTRY:
        available = ", ".join(sorted(KNOWLEDGE_BASE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown knowledge base '{name}'. Available: {available}")
    return KNOWLEDGE_BASE_REGISTRY[name](**kwargs)


def list_knowledge_bases() -> list[str]:
    """Return the sorted names of registered knowledge bases."""
    return sorted(KNOWLEDGE_BASE_REGISTRY)


# ---------------------------------------------------------------------------
# Enterprise Knowledge Base
# ---------------------------------------------------------------------------


@register_knowledge_base(KB_NAME)
class EnterpriseKnowledgeBase:
    """A deterministic, in-memory enterprise knowledge base.

    Holds a registry of :class:`KnowledgeDocument` records and provides
    rule-based search, relevance scoring, collections, statistics, and
    serialisation.  No database, vector store, language model, or embeddings are
    used; all behaviour is exact and reproducible.

    Args:
        config: The knowledge-base configuration.
        experiment_tracker: Optional tracker for logging operation counts.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: KnowledgeBaseConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or KnowledgeBaseConfig()
        self.tracker = experiment_tracker
        self._documents: dict[str, KnowledgeDocument] = {}
        self._collections: dict[str, KnowledgeCollection] = {}
        self._n_ops = 0
        logger.info("EnterpriseKnowledgeBase ready")

    # ------------------------------------------------------------------
    # Registry: register / remove / get / list
    # ------------------------------------------------------------------

    def register_document(self, document: KnowledgeDocument) -> None:
        """Register a document.

        Args:
            document: The document to register.

        Raises:
            TypeError: When *document* is not a :class:`KnowledgeDocument`.
            ValueError: When a document with the same id is already registered.
        """
        if not isinstance(document, KnowledgeDocument):
            raise TypeError("register_document requires a KnowledgeDocument")
        if document.document_id in self._documents:
            raise ValueError(
                f"duplicate document_id '{document.document_id}'")
        self._documents[document.document_id] = document
        self._record()

    def remove_document(self, document_id: str) -> KnowledgeDocument:
        """Remove and return a document by id.

        Args:
            document_id: The identifier to remove.

        Returns:
            The removed document.

        Raises:
            KeyError: When the id is not present.
        """
        if document_id not in self._documents:
            raise KeyError(f"unknown document_id '{document_id}'")
        self._record()
        return self._documents.pop(document_id)

    def get_document(self, document_id: str) -> KnowledgeDocument:
        """Return a document by id.

        Args:
            document_id: The identifier to fetch.

        Returns:
            The document.

        Raises:
            KeyError: When the id is not present.
        """
        if document_id not in self._documents:
            raise KeyError(f"unknown document_id '{document_id}'")
        return self._documents[document_id]

    def has_document(self, document_id: str) -> bool:
        """Return whether a document id is registered."""
        return document_id in self._documents

    def list_documents(
        self, *, category: Any = None, tag: str | None = None,
    ) -> tuple[KnowledgeDocument, ...]:
        """List documents, optionally filtered, sorted by id for determinism.

        Args:
            category: Optional category filter.
            tag: Optional tag filter (case-insensitive).

        Returns:
            A tuple of documents sorted by identifier.
        """
        cat = normalize_category(category) if category is not None else None
        tg = tag.strip().lower() if tag else None
        out = []
        for doc in self._documents.values():
            if cat is not None and doc.category != cat:
                continue
            if tg is not None and tg not in doc.tags:
                continue
            out.append(doc)
        out.sort(key=lambda d: d.document_id)
        return tuple(out)

    def __len__(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _keyword_score(
        self, document: KnowledgeDocument, query_tokens: frozenset[str],
    ) -> tuple[float, frozenset[str]]:
        """Compute a document's weighted keyword score against query tokens.

        Args:
            document: The document.
            query_tokens: The query token set.

        Returns:
            ``(score, matched_terms)`` with score in ``[0, 1]``.
        """
        if not query_tokens:
            return 0.0, frozenset()
        title_t, content_t, tag_t = document.token_sets()
        cfg = self.config
        s = (cfg.title_weight * len(query_tokens & title_t)
             + cfg.content_weight * len(query_tokens & content_t)
             + cfg.tag_weight * len(query_tokens & tag_t))
        denom = len(query_tokens) * cfg.keyword_weight_sum
        score = float(s / denom) if denom > 0 else 0.0
        matched = query_tokens & (title_t | content_t | tag_t)
        return float(np.clip(score, 0.0, 1.0)), matched

    def _tag_overlap(
        self, document: KnowledgeDocument, query_tags: Sequence[str],
    ) -> float:
        """Compute fractional tag overlap.

        Args:
            document: The document.
            query_tags: The query tags.

        Returns:
            The overlap fraction in ``[0, 1]``.
        """
        q = frozenset(t.strip().lower() for t in query_tags if t and t.strip())
        if not q:
            return 0.0
        return float(len(q & frozenset(document.tags)) / len(q))

    def relevance_score(
        self, document: KnowledgeDocument, *, query: str | None = None,
        tags: Sequence[str] | None = None, category: Any = None,
    ) -> float:
        """Compute a deterministic combined relevance score.

        Combines keyword overlap, tag overlap, and category match with the
        configured component weights, over whichever signals are supplied.

        Args:
            document: The document to score.
            query: Optional free-text query.
            tags: Optional query tags.
            category: Optional target category.

        Returns:
            The combined relevance score in ``[0, 1]`` (0.0 if no signal given).
        """
        cfg = self.config
        parts: list[float] = []
        weights: list[float] = []
        if query:
            s, _ = self._keyword_score(document, frozenset(_tokenize(query)))
            parts.append(s); weights.append(cfg.keyword_component_weight)
        if tags:
            parts.append(self._tag_overlap(document, tags))
            weights.append(cfg.tag_component_weight)
        if category is not None:
            match = 1.0 if document.category == normalize_category(category) else 0.0
            parts.append(match); weights.append(cfg.category_component_weight)
        wsum = sum(weights)
        if wsum <= 0:
            return 0.0
        return float(np.clip(sum(p * w for p, w in zip(parts, weights)) / wsum,
                             0.0, 1.0))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_keywords(
        self, query: str, *, top_n: int | None = None, min_score: float = 0.0,
    ) -> tuple[SearchResult, ...]:
        """Rank documents by weighted keyword overlap.

        Args:
            query: The free-text query.
            top_n: Maximum results (defaults to the configured value).
            min_score: Minimum score to include.

        Returns:
            Ranked search results (score descending, id ascending).
        """
        q = frozenset(_tokenize(query or ""))
        if not q:
            return ()
        results = []
        for doc in self._documents.values():
            score, matched = self._keyword_score(doc, q)
            if score > min_score:
                results.append(SearchResult(
                    document_id=doc.document_id, title=doc.title,
                    category=doc.category, score=score,
                    matched_terms=tuple(sorted(matched))))
        results.sort(key=lambda r: (-r.score, r.document_id))
        n = self.config.default_top_n if top_n is None else top_n
        return tuple(results[:n]) if n is not None else tuple(results)

    def search_tags(
        self, tags: Sequence[str], *, match_all: bool = False,
        top_n: int | None = None,
    ) -> tuple[SearchResult, ...]:
        """Search by tag overlap.

        Args:
            tags: The query tags.
            match_all: When True, require all query tags to be present.
            top_n: Maximum results.

        Returns:
            Ranked search results by tag overlap.
        """
        q = frozenset(t.strip().lower() for t in tags if t and t.strip())
        if not q:
            return ()
        results = []
        for doc in self._documents.values():
            inter = q & frozenset(doc.tags)
            if not inter:
                continue
            if match_all and inter != q:
                continue
            score = len(inter) / len(q)
            results.append(SearchResult(
                document_id=doc.document_id, title=doc.title,
                category=doc.category, score=float(score),
                matched_terms=tuple(sorted(inter))))
        results.sort(key=lambda r: (-r.score, r.document_id))
        n = self.config.default_top_n if top_n is None else top_n
        return tuple(results[:n]) if n is not None else tuple(results)

    def search_category(self, category: Any) -> tuple[SearchResult, ...]:
        """Return all documents in a category as search results.

        Args:
            category: The target category.

        Returns:
            Search results (score 1.0) sorted by id.

        Raises:
            ValueError: When the category is invalid.
        """
        cat = normalize_category(category)
        results = [SearchResult(d.document_id, d.title, d.category, 1.0, ())
                   for d in self._documents.values() if d.category == cat]
        results.sort(key=lambda r: r.document_id)
        return tuple(results)

    def search_exact(
        self, phrase: str, *, fields: Sequence[str] = ("title", "content"),
    ) -> tuple[SearchResult, ...]:
        """Return documents containing an exact (case-insensitive) phrase.

        Args:
            phrase: The phrase to match.
            fields: Document fields to search.

        Returns:
            Search results (score 1.0) sorted by id.
        """
        p = (phrase or "").strip().lower()
        if not p:
            return ()
        results = []
        for doc in self._documents.values():
            hay = " ".join(str(getattr(doc, f, "")) for f in fields).lower()
            if p in hay:
                results.append(SearchResult(doc.document_id, doc.title,
                                            doc.category, 1.0, (p,)))
        results.sort(key=lambda r: r.document_id)
        return tuple(results)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def make_collection(
        self, name: str, document_ids: Sequence[str], *, kind: str = "document",
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeCollection:
        """Build a named document collection.

        Args:
            name: The collection name.
            document_ids: Member identifiers (must all exist).
            kind: The collection kind.
            metadata: Optional metadata.

        Returns:
            The collection.

        Raises:
            ValueError: When the name is empty or an id is unknown.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("collection name must be a non-empty string")
        ids = []
        for did in document_ids:
            if did not in self._documents:
                raise ValueError(f"unknown document_id '{did}' in collection")
            ids.append(did)
        coll = KnowledgeCollection(name=name, kind=kind,
                                   document_ids=tuple(sorted(set(ids))),
                                   metadata=dict(metadata or {}))
        self._collections[name] = coll
        self._record()
        return coll

    def category_collection(self, category: Any) -> KnowledgeCollection:
        """Build a collection of all documents in a category.

        Args:
            category: The target category.

        Returns:
            A category collection.
        """
        cat = normalize_category(category)
        ids = sorted(d.document_id for d in self._documents.values()
                     if d.category == cat)
        coll = KnowledgeCollection(name=f"category:{cat}", kind="category",
                                   document_ids=tuple(ids),
                                   metadata={"category": cat})
        self._collections[coll.name] = coll
        return coll

    def fleet_collection(self, fleet_id: str) -> KnowledgeCollection:
        """Build a collection of documents associated with a fleet.

        A document belongs to the fleet when its ``metadata['fleet_id']`` equals
        *fleet_id* or the fleet id appears (case-insensitively) in its tags.

        Args:
            fleet_id: The fleet identifier.

        Returns:
            A fleet collection.
        """
        fid = str(fleet_id).strip()
        flow = fid.lower()
        ids = []
        for d in self._documents.values():
            if str(d.metadata.get("fleet_id", "")) == fid or flow in d.tags:
                ids.append(d.document_id)
        coll = KnowledgeCollection(name=f"fleet:{fid}", kind="fleet",
                                   document_ids=tuple(sorted(ids)),
                                   metadata={"fleet_id": fid})
        self._collections[coll.name] = coll
        return coll

    def get_collection(self, name: str) -> KnowledgeCollection:
        """Return a previously created collection by name.

        Raises:
            KeyError: When the collection is not present.
        """
        if name not in self._collections:
            raise KeyError(f"unknown collection '{name}'")
        return self._collections[name]

    def list_collections(self) -> tuple[str, ...]:
        """Return the sorted names of stored collections."""
        return tuple(sorted(self._collections))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> KnowledgeStatistics:
        """Compute aggregate statistics over the knowledge base.

        Returns:
            A :class:`KnowledgeStatistics`.
        """
        docs = list(self._documents.values())
        cat_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        for d in docs:
            cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
            for t in d.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        cat_dist = tuple(sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0])))
        tag_dist = tuple(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))
        covered = int(np.sum([1 for c in _NON_UNKNOWN_CATEGORIES
                              if cat_counts.get(c, 0) > 0]))
        coverage = float(covered / len(_NON_UNKNOWN_CATEGORIES))
        return KnowledgeStatistics(
            document_count=len(docs),
            category_distribution=cat_dist,
            tag_distribution=tag_dist,
            coverage_score=coverage,
            categories_covered=covered,
            distinct_tags=len(tag_counts),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire knowledge base to a dictionary."""
        return {
            "documents": [self._documents[i].to_dict()
                          for i in sorted(self._documents)],
            "collections": [self._collections[n].to_dict()
                            for n in sorted(self._collections)],
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any],
        config: KnowledgeBaseConfig | None = None,
    ) -> "EnterpriseKnowledgeBase":
        """Reconstruct a knowledge base from a dictionary.

        Args:
            data: A mapping produced by :meth:`to_dict`.
            config: Optional configuration.

        Returns:
            A populated :class:`EnterpriseKnowledgeBase`.

        Raises:
            ValueError: When the payload is malformed.
        """
        if not isinstance(data, Mapping):
            raise ValueError("from_dict requires a mapping")
        kb = cls(config=config)
        for d in data.get("documents", []):
            kb.register_document(KnowledgeDocument.from_dict(d))
        for c in data.get("collections", []):
            name = c.get("name", "")
            ids = [i for i in c.get("document_ids", []) if i in kb._documents]
            if name:
                kb._collections[name] = KnowledgeCollection(
                    name=name, kind=c.get("kind", "document"),
                    document_ids=tuple(sorted(ids)),
                    metadata=dict(c.get("metadata", {}) or {}))
        return kb

    def to_json(self) -> str:
        """Serialise the knowledge base to a JSON string."""
        return json.dumps(self.to_dict())

    # ------------------------------------------------------------------
    # Tracker
    # ------------------------------------------------------------------

    def _record(self) -> None:
        """Increment the operation counter and log it (failure-safe)."""
        self._n_ops += 1
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics({"kb_operations": float(self._n_ops)},
                                     step=self._n_ops)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _seed_corpus() -> EnterpriseKnowledgeBase:
    """Build a small deterministic demonstration corpus.

    Returns:
        A populated knowledge base.
    """
    kb = EnterpriseKnowledgeBase()
    docs = [
        ("KB-001", "Gearbox Bearing Inspection Procedure",
         KnowledgeCategory.INSPECTION_PROCEDURE, ("bearing", "vibration", "gearbox"),
         "Step-by-step procedure for inspecting gearbox bearings for vibration and wear faults.",
         {"fleet_id": "FLEET-NORTH"}),
        ("KB-002", "Lubrication System Maintenance Manual",
         KnowledgeCategory.MAINTENANCE_MANUAL, ("lubrication", "oil", "gearbox"),
         "Maintenance manual covering lubrication system service, oil quality, and refill intervals.",
         {"fleet_id": "FLEET-NORTH"}),
        ("KB-003", "Bearing Fault Failure Catalog",
         KnowledgeCategory.FAILURE_CATALOG, ("bearing", "failure", "vibration"),
         "Catalog of bearing failure modes, their vibration signatures, and typical causes.",
         {}),
        ("KB-004", "Vibration Root Cause Playbook",
         KnowledgeCategory.ROOT_CAUSE_PLAYBOOK, ("vibration", "bearing", "diagnosis"),
         "Playbook mapping vibration anomalies to root causes and recommended investigations.",
         {"fleet_id": "FLEET-NORTH"}),
        ("KB-005", "Turbine Electrical Safety Procedure",
         KnowledgeCategory.SAFETY_PROCEDURE, ("electrical", "safety", "lockout"),
         "Safety procedure for electrical isolation and lockout during turbine maintenance.",
         {}),
        ("KB-006", "Wind Turbine Asset Specification",
         KnowledgeCategory.ASSET_SPECIFICATION, ("turbine", "specification", "gearbox"),
         "Technical specification sheet for the wind turbine, including gearbox and bearing ratings.",
         {"fleet_id": "FLEET-NORTH"}),
    ]
    for did, title, cat, tags, content, meta in docs:
        kb.register_document(KnowledgeDocument(
            document_id=did, title=title, document_type="reference",
            source="engineering", category=cat.value, tags=tags,
            content=content, created_at="2026-01-01T00:00:00Z", version="1.0",
            metadata=meta))
    return kb


def _demo() -> int:
    """Run a short knowledge-base demo.

    Returns:
        Exit code 0.
    """
    kb = _seed_corpus()
    print(f"=== Corpus: {len(kb)} documents ===\n")
    print("=== Keyword search: 'bearing vibration' ===")
    for r in kb.search_keywords("bearing vibration", top_n=4):
        print(f"  {r.document_id}  score={r.score:.3f}  {r.title}  matched={list(r.matched_terms)}")
    print()
    print("=== Tag search: ['gearbox','lubrication'] ===")
    for r in kb.search_tags(["gearbox", "lubrication"]):
        print(f"  {r.document_id}  score={r.score:.3f}  {r.title}")
    print()
    print("=== Category search: failure_catalog ===")
    for r in kb.search_category(KnowledgeCategory.FAILURE_CATALOG):
        print(f"  {r.document_id}  {r.title}")
    print()
    print("=== Exact search: 'lockout' ===")
    for r in kb.search_exact("lockout"):
        print(f"  {r.document_id}  {r.title}")
    print()
    print("=== Fleet collection: FLEET-NORTH ===")
    coll = kb.fleet_collection("FLEET-NORTH")
    print(f"  {len(coll)} documents: {list(coll.document_ids)}")
    print()
    print("=== Statistics ===")
    s = kb.statistics()
    print(f"  documents={s.document_count}  categories_covered={s.categories_covered}/9 "
          f"coverage={s.coverage_score:.2f}  distinct_tags={s.distinct_tags}")
    print(f"  top tags: {s.tag_distribution[:5]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Enterprise knowledge base")
    parser.add_argument("--demo", action="store_true", help="Run a demo.")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--list-kbs", action="store_true")
    args = parser.parse_args(argv)

    if args.list_categories:
        print("Categories:", [c.value for c in KnowledgeCategory])
        return 0
    if args.list_kbs:
        print("Registered knowledge bases:", list_knowledge_bases())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())