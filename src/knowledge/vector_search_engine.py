"""Enterprise Vector Search Engine -- Week 9, Phase 2.

Semantic retrieval foundation for the Enterprise Knowledge Layer of the
Enterprise Digital Twin & Decision Intelligence Platform.

This module is **additive**. It does not modify, re-implement, or duplicate any
frozen Week 1-9 Phase 1 module. In particular it does **not** redefine the
``KnowledgeDocument``, ``SearchResult``, or ``EnterpriseKnowledgeBase`` types
that ship with the Enterprise Knowledge Base (Week 9 Phase 1). Instead it
*composes* them through a small, explicitly documented, duck-typed adapter
layer so that the vector engine remains backward compatible with the existing
public API regardless of incidental field-name differences.

Integration contract (assumed Phase 1 surface)
-----------------------------------------------
The engine reads the following logical fields from any object that behaves like
a ``KnowledgeDocument``. Each field is resolved through a prioritized list of
attribute names so the adapter tolerates the common naming conventions used
across the platform:

============  =======================================================
Logical field  Attribute names tried (in order)
============  =======================================================
identifier     ``document_id``, ``doc_id``, ``id``, ``key``
text body      ``content``, ``text``, ``body``, ``summary``
category       ``category``, ``categories``
tags           ``tags``, ``labels``
metadata       ``metadata``, ``meta``, ``attributes``
============  =======================================================

``EnterpriseKnowledgeBase`` instances are consumed through
:meth:`VectorSearchEngine.from_knowledge_base`, which discovers the document
collection through ``all_documents()`` / ``list_documents()`` /
``documents()`` / ``iter_documents()`` / the ``documents`` attribute. No method
of the knowledge base is overridden or required to change.

Engineering constraints honoured
---------------------------------
* Pure Python + NumPy only. No FAISS / Pinecone / Weaviate / Chroma / OpenAI /
  HuggingFace / sentence-transformers / network access of any kind.
* Deterministic outputs (stable hashing via :mod:`hashlib`, deterministic
  tie-breaking, no reliance on the salted built-in ``hash``).
* Frozen dataclasses for every value object.
* Registry pattern for embedding engines and similarity metrics.
* Full ``to_dict`` / ``from_dict`` JSON-serialisable round trips.
* Production-grade validation with a typed exception hierarchy.
* CLI demo: ``python src/knowledge/vector_search_engine.py --demo``.

Public API
----------
Value objects:        :class:`VectorDocument`, :class:`VectorSearchResult`,
                      :class:`HybridSearchResult`, :class:`IndexStatistics`,
                      :class:`HybridConfig`
Embedding:            :class:`DeterministicEmbeddingEngine`
Index:                :class:`VectorIndex`
Search:               :class:`VectorSearchEngine`, :class:`HybridSearchEngine`
Similarity metrics:   :func:`cosine_similarity`, :func:`euclidean_similarity`,
                      :func:`dot_product_similarity`
Registries:           :data:`EMBEDDING_REGISTRY`, :data:`SIMILARITY_REGISTRY`
Exceptions:           :class:`VectorSearchError` (+ subclasses)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, replace
from hashlib import blake2b
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
    Union,
)

import numpy as np

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    # exceptions
    "VectorSearchError",
    "DimensionMismatchError",
    "DuplicateDocumentError",
    "DocumentNotFoundError",
    "EmptyVectorError",
    "InvalidVectorError",
    "InvalidMetadataError",
    # registries
    "Registry",
    "EMBEDDING_REGISTRY",
    "SIMILARITY_REGISTRY",
    "register_embedding_engine",
    "register_similarity",
    # similarity
    "cosine_similarity",
    "euclidean_similarity",
    "dot_product_similarity",
    # value objects
    "VectorDocument",
    "VectorSearchResult",
    "HybridSearchResult",
    "IndexStatistics",
    "HybridConfig",
    # engines / index
    "DeterministicEmbeddingEngine",
    "VectorIndex",
    "VectorSearchEngine",
    "HybridSearchEngine",
    # helpers
    "stable_hash",
    "tokenize",
]

DEFAULT_EMBEDDING_DIM: int = 128
_EPS: float = 1e-12
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class VectorSearchError(Exception):
    """Base class for every error raised by the vector search engine."""


class DimensionMismatchError(VectorSearchError):
    """Raised when a vector's dimension does not match the expected dimension."""


class DuplicateDocumentError(VectorSearchError):
    """Raised when adding a document whose id already exists in the index."""


class DocumentNotFoundError(VectorSearchError):
    """Raised when an operation references a document id absent from the index."""


class EmptyVectorError(VectorSearchError):
    """Raised when a vector has zero length (no components at all)."""


class InvalidVectorError(VectorSearchError):
    """Raised when a vector contains NaN/Inf values or is not 1-D numeric."""


class InvalidMetadataError(VectorSearchError):
    """Raised when metadata is not a JSON-serialisable mapping."""


# ---------------------------------------------------------------------------
# Registry pattern
# ---------------------------------------------------------------------------
class Registry:
    """A minimal, deterministic name -> object registry.

    Used to register pluggable embedding engines and similarity metrics. The
    registry preserves insertion order and rejects duplicate names so that the
    platform's component wiring stays explicit and reproducible.
    """

    def __init__(self, kind: str) -> None:
        self._kind = str(kind)
        self._items: "Dict[str, Any]" = {}

    @property
    def kind(self) -> str:
        """Return the human-readable kind of objects this registry holds."""
        return self._kind

    def register(self, name: str, obj: Any, *, overwrite: bool = False) -> Any:
        """Register ``obj`` under ``name``.

        Args:
            name: Unique registry key.
            obj: The object (callable, class, or instance) to register.
            overwrite: When ``False`` (default) re-registering an existing name
                raises :class:`VectorSearchError`.

        Returns:
            The registered object (to allow decorator usage).
        """
        key = str(name)
        if not overwrite and key in self._items:
            raise VectorSearchError(
                f"{self._kind} '{key}' is already registered"
            )
        self._items[key] = obj
        return obj

    def get(self, name: str) -> Any:
        """Return the object registered under ``name`` or raise ``KeyError``."""
        key = str(name)
        if key not in self._items:
            raise VectorSearchError(
                f"unknown {self._kind} '{key}'. "
                f"available: {sorted(self._items)}"
            )
        return self._items[key]

    def has(self, name: str) -> bool:
        """Return ``True`` if ``name`` is registered."""
        return str(name) in self._items

    def names(self) -> Tuple[str, ...]:
        """Return registered names sorted deterministically."""
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:  # pragma: no cover - thin
        return str(name) in self._items

    def __len__(self) -> int:  # pragma: no cover - thin
        return len(self._items)


SIMILARITY_REGISTRY = Registry("similarity-metric")
EMBEDDING_REGISTRY = Registry("embedding-engine")


def register_similarity(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering a vectorised similarity metric under ``name``."""

    def _decorator(func: Callable) -> Callable:
        SIMILARITY_REGISTRY.register(name, func)
        return func

    return _decorator


def register_embedding_engine(name: str) -> Callable[[type], type]:
    """Decorator registering an embedding-engine class under ``name``."""

    def _decorator(cls: type) -> type:
        EMBEDDING_REGISTRY.register(name, cls)
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------
def stable_hash(text: str) -> int:
    """Return a deterministic, process-independent 64-bit hash of ``text``.

    The platform requires reproducible embeddings across runs and machines, so
    Python's salted built-in ``hash`` cannot be used. :func:`stable_hash` uses
    BLAKE2b with an 8-byte digest, which is fast, deterministic, and uniformly
    distributed.

    Args:
        text: Arbitrary unicode string.

    Returns:
        A non-negative integer in ``[0, 2**64)``.
    """
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def tokenize(text: str) -> List[str]:
    """Tokenise ``text`` into lowercase alphanumeric word tokens.

    Deterministic and locale-independent. Punctuation and whitespace act as
    separators; the result preserves token order and repetition.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _coerce_vector(values: Any, *, expected_dim: Optional[int] = None) -> np.ndarray:
    """Validate and convert ``values`` into a 1-D float64 NumPy vector.

    Raises:
        EmptyVectorError: if the vector has zero length.
        InvalidVectorError: if the vector is not 1-D numeric or holds NaN/Inf.
        DimensionMismatchError: if ``expected_dim`` is set and not matched.
    """
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # non-numeric content
        raise InvalidVectorError(f"vector is not numeric: {exc}") from exc
    if arr.ndim != 1:
        raise InvalidVectorError(
            f"vector must be 1-D, got {arr.ndim}-D shape {arr.shape}"
        )
    if arr.size == 0:
        raise EmptyVectorError("vector has zero length")
    if not np.all(np.isfinite(arr)):
        raise InvalidVectorError("vector contains NaN or infinite values")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise DimensionMismatchError(
            f"expected dimension {expected_dim}, got {arr.shape[0]}"
        )
    return arr


def _validate_metadata(metadata: Any) -> Dict[str, Any]:
    """Validate that ``metadata`` is a JSON-serialisable mapping; return a copy."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise InvalidMetadataError(
            f"metadata must be a mapping, got {type(metadata).__name__}"
        )
    plain = {str(k): v for k, v in metadata.items()}
    try:
        json.dumps(plain, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidMetadataError(
            f"metadata is not JSON-serialisable: {exc}"
        ) from exc
    return plain


# Duck-typed accessors for Phase 1 ``KnowledgeDocument`` objects ------------
_ID_ATTRS = ("document_id", "doc_id", "id", "key")
_TEXT_ATTRS = ("content", "text", "body", "summary")
_CATEGORY_ATTRS = ("category", "categories")
_TAG_ATTRS = ("tags", "labels")
_META_ATTRS = ("metadata", "meta", "attributes")


def _first_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    return default


def _doc_id(doc: Any) -> str:
    value = _first_attr(doc, _ID_ATTRS)
    if value is None:
        raise InvalidMetadataError("document is missing an identifier field")
    return str(value)


def _doc_text(doc: Any) -> str:
    value = _first_attr(doc, _TEXT_ATTRS, default="")
    return "" if value is None else str(value)


def _doc_category(doc: Any) -> Optional[str]:
    value = _first_attr(doc, _CATEGORY_ATTRS)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _doc_tags(doc: Any) -> Tuple[str, ...]:
    value = _first_attr(doc, _TAG_ATTRS, default=())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(v) for v in value))
    return (str(value),)


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    value = _first_attr(doc, _META_ATTRS, default={})
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


# ---------------------------------------------------------------------------
# Similarity metrics (vectorised)
# ---------------------------------------------------------------------------
def _as_matrix(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise InvalidVectorError(
            f"similarity matrix must be 2-D, got {arr.ndim}-D"
        )
    return arr


@register_similarity("cosine")
def cosine_similarity(query: Any, matrix: Any) -> np.ndarray:
    """Cosine similarity between ``query`` (d,) and each row of ``matrix`` (n,d).

    Returns a ``(n,)`` array in ``[-1, 1]``. Zero-norm vectors yield ``0.0``
    similarity rather than ``NaN`` so the metric is total over the input domain.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    m = _as_matrix(matrix)
    q_norm = float(np.linalg.norm(q))
    m_norm = np.linalg.norm(m, axis=1)
    denom = m_norm * q_norm
    dots = m @ q
    out = np.zeros_like(dots, dtype=np.float64)
    nonzero = denom > _EPS
    out[nonzero] = dots[nonzero] / denom[nonzero]
    return out


@register_similarity("euclidean")
def euclidean_similarity(query: Any, matrix: Any) -> np.ndarray:
    """Euclidean *similarity* in ``(0, 1]`` derived from L2 distance.

    Computed as ``1 / (1 + ||row - query||_2)`` so that identical vectors score
    ``1.0`` and similarity decreases monotonically with distance. This keeps the
    metric directly comparable with the bounded :func:`cosine_similarity`.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    m = _as_matrix(matrix)
    dist = np.linalg.norm(m - q, axis=1)
    return 1.0 / (1.0 + dist)


@register_similarity("dot")
def dot_product_similarity(query: Any, matrix: Any) -> np.ndarray:
    """Raw dot-product similarity between ``query`` and each row of ``matrix``."""
    q = np.asarray(query, dtype=np.float64).ravel()
    m = _as_matrix(matrix)
    return m @ q


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorDocument:
    """An immutable embedded document: an id, its dense vector, and metadata.

    The vector is stored as a tuple of floats so the dataclass is genuinely
    immutable and JSON-serialisable. Use :meth:`as_array` for numeric work.

    Attributes:
        document_id: Stable identifier shared with the source ``KnowledgeDocument``.
        vector: Dense embedding as a tuple of floats (length == dimension).
        metadata: JSON-serialisable mapping (category, tags, etc.).
    """

    document_id: str
    vector: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        arr = _coerce_vector(self.vector)
        object.__setattr__(self, "vector", tuple(float(x) for x in arr))
        object.__setattr__(
            self, "metadata", MappingProxyType(_validate_metadata(self.metadata))
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return len(self.vector)

    def as_array(self) -> np.ndarray:
        """Return the vector as a fresh ``float64`` NumPy array."""
        return np.asarray(self.vector, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the document."""
        return {
            "document_id": self.document_id,
            "vector": [float(x) for x in self.vector],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorDocument":
        """Reconstruct a :class:`VectorDocument` from :meth:`to_dict` output."""
        return cls(
            document_id=data["document_id"],
            vector=tuple(float(x) for x in data["vector"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class VectorSearchResult:
    """A single ranked hit returned by the vector search engine.

    Attributes:
        document_id: Identifier of the retrieved document.
        score: Similarity score under the active metric (higher is better).
        rank: 1-based rank within the result list (deterministic ties).
        metadata: Copy of the document metadata for downstream consumers.
    """

    document_id: str
    score: float
    rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(
            self, "metadata", MappingProxyType(_validate_metadata(self.metadata))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorSearchResult":
        """Reconstruct a :class:`VectorSearchResult` from :meth:`to_dict`."""
        return cls(
            document_id=data["document_id"],
            score=float(data["score"]),
            rank=int(data["rank"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class HybridSearchResult:
    """A ranked hit produced by hybrid (keyword + vector) retrieval.

    Attributes:
        document_id: Identifier of the retrieved document.
        score: Combined weighted score (higher is better).
        rank: 1-based rank within the hybrid result list.
        keyword_score: Normalised keyword component prior to weighting.
        vector_score: Normalised vector component prior to weighting.
        metadata: Copy of the document metadata.
    """

    document_id: str
    score: float
    rank: int
    keyword_score: float
    vector_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "keyword_score", float(self.keyword_score))
        object.__setattr__(self, "vector_score", float(self.vector_score))
        object.__setattr__(
            self, "metadata", MappingProxyType(_validate_metadata(self.metadata))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the hybrid result."""
        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "keyword_score": self.keyword_score,
            "vector_score": self.vector_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HybridSearchResult":
        """Reconstruct a :class:`HybridSearchResult` from :meth:`to_dict`."""
        return cls(
            document_id=data["document_id"],
            score=float(data["score"]),
            rank=int(data["rank"]),
            keyword_score=float(data["keyword_score"]),
            vector_score=float(data["vector_score"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class IndexStatistics:
    """Aggregate statistics describing a populated :class:`VectorIndex`.

    Attributes:
        document_count: Number of indexed documents.
        vector_dimension: Embedding dimension (0 when the index is empty).
        index_density: Mean fraction of non-zero components across vectors,
            in ``[0, 1]`` (0.0 for an empty index).
        average_similarity: Mean pairwise cosine similarity across the corpus,
            in ``[-1, 1]`` (0.0 when fewer than two documents are present).
    """

    document_count: int
    vector_dimension: int
    index_density: float
    average_similarity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_count", int(self.document_count))
        object.__setattr__(self, "vector_dimension", int(self.vector_dimension))
        object.__setattr__(self, "index_density", float(self.index_density))
        object.__setattr__(
            self, "average_similarity", float(self.average_similarity)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the statistics."""
        return {
            "document_count": self.document_count,
            "vector_dimension": self.vector_dimension,
            "index_density": self.index_density,
            "average_similarity": self.average_similarity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndexStatistics":
        """Reconstruct :class:`IndexStatistics` from :meth:`to_dict` output."""
        return cls(
            document_count=int(data["document_count"]),
            vector_dimension=int(data["vector_dimension"]),
            index_density=float(data["index_density"]),
            average_similarity=float(data["average_similarity"]),
        )


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for weighted hybrid retrieval.

    Attributes:
        keyword_weight: Non-negative weight applied to the keyword component.
        vector_weight: Non-negative weight applied to the vector component.
        normalize: When ``True`` each component is min-max normalised to
            ``[0, 1]`` across the candidate set before weighting, which keeps
            the two heterogeneous score scales comparable.
    """

    keyword_weight: float = 0.5
    vector_weight: float = 0.5
    normalize: bool = True

    def __post_init__(self) -> None:
        kw = float(self.keyword_weight)
        vw = float(self.vector_weight)
        if kw < 0.0 or vw < 0.0:
            raise VectorSearchError("hybrid weights must be non-negative")
        if (kw + vw) <= _EPS:
            raise VectorSearchError("at least one hybrid weight must be positive")
        object.__setattr__(self, "keyword_weight", kw)
        object.__setattr__(self, "vector_weight", vw)
        object.__setattr__(self, "normalize", bool(self.normalize))

    @property
    def total_weight(self) -> float:
        """Return the sum of the two component weights."""
        return self.keyword_weight + self.vector_weight

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        return {
            "keyword_weight": self.keyword_weight,
            "vector_weight": self.vector_weight,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HybridConfig":
        """Reconstruct a :class:`HybridConfig` from :meth:`to_dict` output."""
        return cls(
            keyword_weight=float(data.get("keyword_weight", 0.5)),
            vector_weight=float(data.get("vector_weight", 0.5)),
            normalize=bool(data.get("normalize", True)),
        )


# ---------------------------------------------------------------------------
# Deterministic embedding engine
# ---------------------------------------------------------------------------
@register_embedding_engine("deterministic-hash")
@dataclass(frozen=True)
class DeterministicEmbeddingEngine:
    """Deterministic text embedder using token + character n-gram feature hashing.

    The engine produces fixed-dimensional dense embeddings with **no external
    dependency** -- no API calls, no model downloads, no randomness. Identical
    text always maps to an identical vector, on any machine, in any process.

    Algorithm
    ---------
    1. Tokenise the text into lowercase alphanumeric tokens.
    2. Expand each token into a feature set: the whole token plus its character
       n-grams (``char_ngram_min`` .. ``char_ngram_max``), capturing subword
       signal so morphologically related terms share components.
    3. Hash every feature with :func:`stable_hash`. The hash selects a bucket in
       ``[0, dimension)`` and a sign (``+1`` / ``-1``) from independent bits,
       implementing the signed feature-hashing trick that keeps the expected
       dot product unbiased.
    4. Accumulate signed counts into the vector with ``numpy.add.at`` (vectorised
       scatter-add) and L2-normalise the result.

    Attributes:
        dimension: Output dimension (defaults to 128).
        char_ngram_min: Minimum character n-gram length (inclusive).
        char_ngram_max: Maximum character n-gram length (inclusive).
        use_token_features: Include whole-token features in addition to n-grams.
        normalize: L2-normalise the final vector.
    """

    dimension: int = DEFAULT_EMBEDDING_DIM
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    use_token_features: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        dim = int(self.dimension)
        if dim <= 0:
            raise VectorSearchError("embedding dimension must be positive")
        if self.char_ngram_min <= 0 or self.char_ngram_max <= 0:
            raise VectorSearchError("character n-gram lengths must be positive")
        if self.char_ngram_min > self.char_ngram_max:
            raise VectorSearchError(
                "char_ngram_min must not exceed char_ngram_max"
            )
        object.__setattr__(self, "dimension", dim)
        object.__setattr__(self, "char_ngram_min", int(self.char_ngram_min))
        object.__setattr__(self, "char_ngram_max", int(self.char_ngram_max))
        object.__setattr__(self, "use_token_features", bool(self.use_token_features))
        object.__setattr__(self, "normalize", bool(self.normalize))

    # -- feature extraction -------------------------------------------------
    def _features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        features: List[str] = []
        for token in tokens:
            if self.use_token_features:
                features.append("t:" + token)
            padded = f"#{token}#"
            upper = min(self.char_ngram_max, len(padded))
            for n in range(self.char_ngram_min, upper + 1):
                for i in range(len(padded) - n + 1):
                    features.append("g:" + padded[i : i + n])
        return features

    # -- public API ---------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a raw string into a deterministic ``float64`` vector.

        Empty or token-free text maps to the zero vector (a valid, finite
        embedding). The returned array always has length :attr:`dimension`.
        """
        if text is None:
            text = ""
        vec = np.zeros(self.dimension, dtype=np.float64)
        features = self._features(str(text))
        if not features:
            return vec
        hashes = np.fromiter(
            (stable_hash(f) for f in features), dtype=np.uint64, count=len(features)
        )
        indices = (hashes % np.uint64(self.dimension)).astype(np.int64)
        signs = np.where(((hashes >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0)
        np.add.at(vec, indices, signs)
        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > _EPS:
                vec = vec / norm
        return vec

    def embed_document(self, document: Any) -> VectorDocument:
        """Embed a ``KnowledgeDocument``-like object into a :class:`VectorDocument`.

        The source identifier, category, and tags are copied into the resulting
        document's metadata so downstream filtering does not require the original
        knowledge base. Any pre-existing metadata on the source document is
        preserved (and not overwritten by reserved keys).
        """
        doc_id = _doc_id(document)
        text = _doc_text(document)
        vector = self.embed_text(text)
        metadata: Dict[str, Any] = dict(_doc_metadata(document))
        category = _doc_category(document)
        if category is not None:
            metadata.setdefault("category", category)
        tags = _doc_tags(document)
        if tags:
            metadata.setdefault("tags", list(tags))
        metadata.setdefault("text_length", len(text))
        return VectorDocument(document_id=doc_id, vector=vector, metadata=metadata)

    def embed_documents(self, documents: Iterable[Any]) -> List[VectorDocument]:
        """Embed an iterable of documents, preserving input order."""
        return [self.embed_document(doc) for doc in documents]

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the engine config."""
        return {
            "type": "deterministic-hash",
            "dimension": self.dimension,
            "char_ngram_min": self.char_ngram_min,
            "char_ngram_max": self.char_ngram_max,
            "use_token_features": self.use_token_features,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeterministicEmbeddingEngine":
        """Reconstruct an engine from :meth:`to_dict` output."""
        return cls(
            dimension=int(data.get("dimension", DEFAULT_EMBEDDING_DIM)),
            char_ngram_min=int(data.get("char_ngram_min", 3)),
            char_ngram_max=int(data.get("char_ngram_max", 5)),
            use_token_features=bool(data.get("use_token_features", True)),
            normalize=bool(data.get("normalize", True)),
        )


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------
class VectorIndex:
    """An in-memory, insertion-ordered store of :class:`VectorDocument` objects.

    The index enforces a single shared dimension, rejects duplicate ids and
    invalid vectors, and lazily materialises a dense ``(n, d)`` matrix for
    vectorised similarity search. The matrix cache is invalidated automatically
    on any mutation so results never go stale.

    Args:
        dimension: Fixed embedding dimension for every stored vector.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        dim = int(dimension)
        if dim <= 0:
            raise VectorSearchError("index dimension must be positive")
        self._dimension = dim
        self._documents: "Dict[str, VectorDocument]" = {}
        self._matrix_cache: Optional[np.ndarray] = None
        self._id_cache: Optional[Tuple[str, ...]] = None

    # -- properties ---------------------------------------------------------
    @property
    def dimension(self) -> int:
        """Return the fixed embedding dimension of the index."""
        return self._dimension

    # -- cache management ---------------------------------------------------
    def _invalidate(self) -> None:
        self._matrix_cache = None
        self._id_cache = None

    # -- mutation -----------------------------------------------------------
    def add_document(self, document: VectorDocument) -> None:
        """Add ``document`` to the index.

        Raises:
            DuplicateDocumentError: if the id already exists.
            DimensionMismatchError: if the vector dimension is wrong.
        """
        if not isinstance(document, VectorDocument):
            raise InvalidVectorError(
                f"expected VectorDocument, got {type(document).__name__}"
            )
        if document.dimension != self._dimension:
            raise DimensionMismatchError(
                f"index expects dimension {self._dimension}, "
                f"document has {document.dimension}"
            )
        if document.document_id in self._documents:
            raise DuplicateDocumentError(
                f"document '{document.document_id}' already exists"
            )
        self._documents[document.document_id] = document
        self._invalidate()

    def remove_document(self, document_id: str) -> VectorDocument:
        """Remove and return the document with ``document_id``.

        Raises:
            DocumentNotFoundError: if the id is absent.
        """
        key = str(document_id)
        if key not in self._documents:
            raise DocumentNotFoundError(f"document '{key}' not found")
        removed = self._documents.pop(key)
        self._invalidate()
        return removed

    def update_document(self, document: VectorDocument) -> None:
        """Replace an existing document in place, preserving insertion order.

        Raises:
            DocumentNotFoundError: if the id is absent.
            DimensionMismatchError: if the vector dimension is wrong.
        """
        if not isinstance(document, VectorDocument):
            raise InvalidVectorError(
                f"expected VectorDocument, got {type(document).__name__}"
            )
        if document.document_id not in self._documents:
            raise DocumentNotFoundError(
                f"document '{document.document_id}' not found"
            )
        if document.dimension != self._dimension:
            raise DimensionMismatchError(
                f"index expects dimension {self._dimension}, "
                f"document has {document.dimension}"
            )
        self._documents[document.document_id] = document
        self._invalidate()

    # -- access -------------------------------------------------------------
    def get_document(self, document_id: str) -> VectorDocument:
        """Return the document with ``document_id`` or raise ``DocumentNotFoundError``."""
        key = str(document_id)
        if key not in self._documents:
            raise DocumentNotFoundError(f"document '{key}' not found")
        return self._documents[key]

    def has_document(self, document_id: str) -> bool:
        """Return ``True`` if ``document_id`` is present in the index."""
        return str(document_id) in self._documents

    def size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def __len__(self) -> int:  # pragma: no cover - thin
        return len(self._documents)

    def __contains__(self, document_id: object) -> bool:  # pragma: no cover
        return str(document_id) in self._documents

    def ids(self) -> Tuple[str, ...]:
        """Return document ids in insertion order."""
        if self._id_cache is None:
            self._id_cache = tuple(self._documents.keys())
        return self._id_cache

    def documents(self) -> Tuple[VectorDocument, ...]:
        """Return all documents in insertion order."""
        return tuple(self._documents.values())

    def matrix(self) -> np.ndarray:
        """Return the cached dense ``(n, d)`` matrix of stacked vectors.

        For an empty index this returns a ``(0, dimension)`` array.
        """
        if self._matrix_cache is None:
            if self._documents:
                self._matrix_cache = np.vstack(
                    [doc.as_array() for doc in self._documents.values()]
                )
            else:
                self._matrix_cache = np.zeros((0, self._dimension), dtype=np.float64)
            self._id_cache = tuple(self._documents.keys())
        return self._matrix_cache

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the index."""
        return {
            "dimension": self._dimension,
            "documents": [doc.to_dict() for doc in self._documents.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorIndex":
        """Reconstruct a :class:`VectorIndex` from :meth:`to_dict` output."""
        index = cls(dimension=int(data["dimension"]))
        for raw in data.get("documents", []):
            index.add_document(VectorDocument.from_dict(raw))
        return index


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------
def _rank_scores(
    ids: Sequence[str],
    scores: Sequence[float],
    top_k: Optional[int],
) -> List[Tuple[str, float, int]]:
    """Rank ``(id, score)`` pairs deterministically and return the top ``k``.

    Sorting is by descending score with id as a stable, deterministic tie
    breaker (ascending lexicographic). Returns ``(id, score, rank)`` triples
    with 1-based ranks.
    """
    paired = list(zip(ids, (float(s) for s in scores)))
    paired.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        if top_k < 0:
            raise VectorSearchError("top_k must be non-negative")
        paired = paired[:top_k]
    return [(doc_id, score, rank) for rank, (doc_id, score) in enumerate(paired, 1)]


# ---------------------------------------------------------------------------
# Vector search engine
# ---------------------------------------------------------------------------
class VectorSearchEngine:
    """Semantic retrieval over a :class:`VectorIndex`.

    Composes a :class:`DeterministicEmbeddingEngine` (to embed queries and
    documents) with a :class:`VectorIndex` (storage) and a registered similarity
    metric (default ``cosine``). The engine is the semantic counterpart to the
    keyword / category / tag / exact search already provided by the Enterprise
    Knowledge Base, and is the foundation the Enterprise RAG layer will build on.

    Args:
        embedding_engine: Engine used to embed text. Defaults to a 128-dim
            :class:`DeterministicEmbeddingEngine`.
        index: Existing index to wrap. When omitted a new empty index is created
            with the embedding engine's dimension.
        metric: Name of a registered similarity metric (``cosine`` by default).
    """

    def __init__(
        self,
        embedding_engine: Optional[DeterministicEmbeddingEngine] = None,
        index: Optional[VectorIndex] = None,
        metric: str = "cosine",
    ) -> None:
        self._embedding_engine = embedding_engine or DeterministicEmbeddingEngine()
        if index is None:
            index = VectorIndex(dimension=self._embedding_engine.dimension)
        if index.dimension != self._embedding_engine.dimension:
            raise DimensionMismatchError(
                f"embedding engine dimension {self._embedding_engine.dimension} "
                f"does not match index dimension {index.dimension}"
            )
        self._index = index
        if not SIMILARITY_REGISTRY.has(metric):
            raise VectorSearchError(f"unknown similarity metric '{metric}'")
        self._metric = metric

    # -- properties ---------------------------------------------------------
    @property
    def index(self) -> VectorIndex:
        """Return the underlying :class:`VectorIndex`."""
        return self._index

    @property
    def embedding_engine(self) -> DeterministicEmbeddingEngine:
        """Return the underlying embedding engine."""
        return self._embedding_engine

    @property
    def metric(self) -> str:
        """Return the name of the active similarity metric."""
        return self._metric

    # -- ingestion ----------------------------------------------------------
    def add_document(self, document: Any) -> VectorDocument:
        """Embed and add a ``KnowledgeDocument``-like object to the index.

        Returns the resulting :class:`VectorDocument`.
        """
        if isinstance(document, VectorDocument):
            vector_doc = document
        else:
            vector_doc = self._embedding_engine.embed_document(document)
        self._index.add_document(vector_doc)
        return vector_doc

    def index_documents(self, documents: Iterable[Any]) -> List[VectorDocument]:
        """Embed and add many documents, returning the created vector documents."""
        created: List[VectorDocument] = []
        for doc in documents:
            created.append(self.add_document(doc))
        return created

    @classmethod
    def from_knowledge_base(
        cls,
        knowledge_base: Any,
        embedding_engine: Optional[DeterministicEmbeddingEngine] = None,
        metric: str = "cosine",
    ) -> "VectorSearchEngine":
        """Build an engine from an ``EnterpriseKnowledgeBase``-like object.

        The document collection is discovered through whichever of the following
        the knowledge base exposes (tried in order): ``all_documents()``,
        ``list_documents()``, ``documents()``, ``iter_documents()``, or a
        ``documents`` attribute. No method of the knowledge base is modified.
        """
        engine = cls(embedding_engine=embedding_engine, metric=metric)
        documents = _discover_documents(knowledge_base)
        engine.index_documents(documents)
        return engine

    # -- query embedding ----------------------------------------------------
    def _resolve_query_vector(self, query: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
        if isinstance(query, str):
            return self._embedding_engine.embed_text(query)
        return _coerce_vector(query, expected_dim=self._index.dimension)

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        top_k: Optional[int] = 10,
        *,
        metric: Optional[str] = None,
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> List[VectorSearchResult]:
        """Return the ``top_k`` most similar documents to ``query``.

        Args:
            query: Either query text (embedded on the fly) or a precomputed
                vector of the index dimension.
            top_k: Maximum number of results (``None`` returns all). Must be
                non-negative.
            metric: Optional override of the active similarity metric.
            candidate_ids: Optional restriction of the search to a subset of
                document ids (used by category / tag filtered search).

        Returns:
            A ranked list of :class:`VectorSearchResult`.
        """
        metric_name = metric or self._metric
        similarity_fn = SIMILARITY_REGISTRY.get(metric_name)
        query_vec = self._resolve_query_vector(query)

        if candidate_ids is None:
            ids = list(self._index.ids())
            matrix = self._index.matrix()
        else:
            ids = [cid for cid in candidate_ids if self._index.has_document(cid)]
            if ids:
                matrix = np.vstack(
                    [self._index.get_document(cid).as_array() for cid in ids]
                )
            else:
                matrix = np.zeros((0, self._index.dimension), dtype=np.float64)

        if not ids:
            return []

        scores = similarity_fn(query_vec, matrix)
        ranked = _rank_scores(ids, scores, top_k)
        return [
            VectorSearchResult(
                document_id=doc_id,
                score=score,
                rank=rank,
                metadata=dict(self._index.get_document(doc_id).metadata),
            )
            for doc_id, score, rank in ranked
        ]

    def search_top_k(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        k: int,
        *,
        metric: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Explicit-``k`` convenience wrapper around :meth:`search`."""
        return self.search(query, top_k=k, metric=metric)

    def search_by_category(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        category: str,
        top_k: Optional[int] = 10,
        *,
        metric: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search restricted to documents whose metadata ``category`` matches."""
        target = str(category)
        candidates = [
            doc.document_id
            for doc in self._index.documents()
            if str(doc.metadata.get("category")) == target
        ]
        return self.search(
            query, top_k=top_k, metric=metric, candidate_ids=candidates
        )

    def search_by_tags(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        tags: Sequence[str],
        top_k: Optional[int] = 10,
        *,
        match_all: bool = False,
        metric: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search restricted to documents matching the requested ``tags``.

        Args:
            tags: Tags to filter on.
            match_all: When ``True`` a document must carry every requested tag;
                otherwise any overlapping tag qualifies.
        """
        wanted = {str(t) for t in tags}
        candidates: List[str] = []
        for doc in self._index.documents():
            doc_tags = {str(t) for t in doc.metadata.get("tags", [])}
            if not wanted:
                continue
            if match_all:
                if wanted.issubset(doc_tags):
                    candidates.append(doc.document_id)
            elif wanted & doc_tags:
                candidates.append(doc.document_id)
        return self.search(
            query, top_k=top_k, metric=metric, candidate_ids=candidates
        )

    # -- statistics ---------------------------------------------------------
    def statistics(self, max_pairs: int = 4096) -> IndexStatistics:
        """Return :class:`IndexStatistics` for the underlying index."""
        return compute_statistics(self._index, max_pairs=max_pairs)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the entire engine."""
        return {
            "metric": self._metric,
            "embedding_engine": self._embedding_engine.to_dict(),
            "index": self._index.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorSearchEngine":
        """Reconstruct a :class:`VectorSearchEngine` from :meth:`to_dict`."""
        engine = DeterministicEmbeddingEngine.from_dict(data["embedding_engine"])
        index = VectorIndex.from_dict(data["index"])
        return cls(embedding_engine=engine, index=index, metric=data.get("metric", "cosine"))


def _discover_documents(knowledge_base: Any) -> List[Any]:
    """Return the document collection of an ``EnterpriseKnowledgeBase``-like object."""
    for method in ("all_documents", "list_documents", "documents", "iter_documents"):
        if hasattr(knowledge_base, method):
            attr = getattr(knowledge_base, method)
            collection = attr() if callable(attr) else attr
            return list(collection)
    if isinstance(knowledge_base, Mapping) and "documents" in knowledge_base:
        return list(knowledge_base["documents"])
    if isinstance(knowledge_base, Iterable):
        return list(knowledge_base)
    raise VectorSearchError(
        "could not discover documents on the supplied knowledge base"
    )


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------
def _min_max_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a mapping of scores to ``[0, 1]`` deterministically.

    A constant input maps to all-``1.0`` (every candidate is equally and fully
    relevant on that axis), which avoids collapsing the signal to zero.
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= _EPS:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


class HybridSearchEngine:
    """Weighted hybrid retrieval combining keyword and vector relevance.

    The engine composes an existing :class:`VectorSearchEngine` with a
    *keyword score provider*. The provider supplies a ``{document_id: score}``
    mapping for a query; this is intentionally generic so it can be backed by
    the Enterprise Knowledge Base's existing keyword search, an exact-match
    scorer, or any custom lexical signal -- without this module reaching into
    or duplicating that implementation.

    Args:
        vector_engine: The semantic engine providing the vector component.
        keyword_provider: Callable ``(query: str) -> Mapping[str, float]``
            returning lexical scores keyed by document id. Optional; when
            omitted a deterministic built-in token-overlap scorer is used that
            reads document text from the index metadata's ``text`` field if
            present, otherwise falls back to a pure vector ranking.
        config: :class:`HybridConfig` controlling weights and normalisation.
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        keyword_provider: Optional[Callable[[str], Mapping[str, float]]] = None,
        config: Optional[HybridConfig] = None,
    ) -> None:
        self._vector_engine = vector_engine
        self._keyword_provider = keyword_provider
        self._config = config or HybridConfig()

    @property
    def config(self) -> HybridConfig:
        """Return the active :class:`HybridConfig`."""
        return self._config

    @property
    def vector_engine(self) -> VectorSearchEngine:
        """Return the wrapped :class:`VectorSearchEngine`."""
        return self._vector_engine

    def _keyword_scores(self, query: str) -> Dict[str, float]:
        if self._keyword_provider is not None:
            provided = self._keyword_provider(query)
            return {str(k): float(v) for k, v in provided.items()}
        # Built-in deterministic token-overlap fallback over indexed metadata.
        query_tokens = set(tokenize(query))
        scores: Dict[str, float] = {}
        for doc in self._vector_engine.index.documents():
            text = str(doc.metadata.get("text", ""))
            doc_tokens = set(tokenize(text))
            if query_tokens and doc_tokens:
                overlap = len(query_tokens & doc_tokens)
                scores[doc.document_id] = float(overlap) / float(len(query_tokens))
            else:
                scores[doc.document_id] = 0.0
        return scores

    def search(
        self,
        query: str,
        top_k: Optional[int] = 10,
        *,
        config: Optional[HybridConfig] = None,
    ) -> List[HybridSearchResult]:
        """Return ``top_k`` hybrid-ranked results for ``query``.

        Vector and keyword scores are gathered over the union of candidate
        documents, optionally min-max normalised, then linearly combined using
        the configured weights. Ranking ties break deterministically by id.
        """
        cfg = config or self._config

        # Vector component over the whole index.
        vector_results = self._vector_engine.search(query, top_k=None)
        vector_scores = {r.document_id: r.score for r in vector_results}

        # Keyword component.
        keyword_scores = self._keyword_scores(query)

        all_ids = sorted(set(vector_scores) | set(keyword_scores))
        if not all_ids:
            return []

        vec_full = {doc_id: vector_scores.get(doc_id, 0.0) for doc_id in all_ids}
        kw_full = {doc_id: keyword_scores.get(doc_id, 0.0) for doc_id in all_ids}

        if cfg.normalize:
            vec_norm = _min_max_normalize(vec_full)
            kw_norm = _min_max_normalize(kw_full)
        else:
            vec_norm = vec_full
            kw_norm = kw_full

        combined = {
            doc_id: cfg.keyword_weight * kw_norm[doc_id]
            + cfg.vector_weight * vec_norm[doc_id]
            for doc_id in all_ids
        }

        ranked = _rank_scores(all_ids, [combined[i] for i in all_ids], top_k)
        results: List[HybridSearchResult] = []
        for doc_id, score, rank in ranked:
            metadata = (
                dict(self._vector_engine.index.get_document(doc_id).metadata)
                if self._vector_engine.index.has_document(doc_id)
                else {}
            )
            results.append(
                HybridSearchResult(
                    document_id=doc_id,
                    score=score,
                    rank=rank,
                    keyword_score=kw_norm[doc_id],
                    vector_score=vec_norm[doc_id],
                    metadata=metadata,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------
def compute_statistics(index: VectorIndex, max_pairs: int = 4096) -> IndexStatistics:
    """Compute :class:`IndexStatistics` for ``index``.

    * ``index_density`` is the mean fraction of non-zero components per vector.
    * ``average_similarity`` is the mean off-diagonal cosine similarity. For
      corpora larger than roughly ``sqrt(2 * max_pairs)`` documents a
      deterministic evenly-spaced subsample of rows is used so the computation
      stays bounded while remaining reproducible.

    Args:
        index: The populated (or empty) index.
        max_pairs: Soft cap on the number of pairwise comparisons.

    Returns:
        An :class:`IndexStatistics` value object.
    """
    n = index.size()
    if n == 0:
        return IndexStatistics(0, index.dimension, 0.0, 0.0)

    matrix = index.matrix()
    density = float(np.mean(np.count_nonzero(matrix, axis=1) / index.dimension))

    if n < 2:
        return IndexStatistics(n, index.dimension, density, 0.0)

    # Deterministic subsample to bound pairwise work.
    max_rows = max(2, int(math.floor(math.sqrt(2.0 * max_pairs))))
    if n > max_rows:
        sel = np.linspace(0, n - 1, num=max_rows, dtype=np.int64)
        sel = np.unique(sel)
        sub = matrix[sel]
    else:
        sub = matrix

    norms = np.linalg.norm(sub, axis=1)
    safe = np.where(norms > _EPS, norms, 1.0)
    unit = sub / safe[:, None]
    sims = unit @ unit.T
    m = sub.shape[0]
    # Mean of strictly-upper-triangular entries (unique unordered pairs).
    iu = np.triu_indices(m, k=1)
    if iu[0].size == 0:
        avg_sim = 0.0
    else:
        avg_sim = float(np.mean(sims[iu]))
    return IndexStatistics(n, index.dimension, density, avg_sim)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DemoDocument:
    """Lightweight stand-in for a Phase 1 ``KnowledgeDocument`` (demo/tests only).

    This is **not** a redefinition of the production model -- it is a fixture
    that exposes the documented duck-typed fields so the demo runs standalone
    without importing the frozen Enterprise Knowledge Base.
    """

    document_id: str
    content: str
    category: str
    tags: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _demo_corpus() -> List[_DemoDocument]:
    return [
        _DemoDocument(
            "doc-turbine-vibration",
            "Turbine bearing vibration anomaly detected by predictive maintenance "
            "asset health engine indicating elevated failure risk.",
            "predictive-maintenance",
            ("turbine", "vibration", "bearing", "anomaly"),
            {"text": "turbine bearing vibration anomaly predictive maintenance"},
        ),
        _DemoDocument(
            "doc-rul-forecast",
            "Remaining useful life forecast for compressor stage two derived from "
            "degradation trend and Monte Carlo risk simulation.",
            "predictive-maintenance",
            ("rul", "compressor", "forecast", "monte-carlo"),
            {"text": "remaining useful life compressor degradation monte carlo"},
        ),
        _DemoDocument(
            "doc-fleet-twin",
            "Fleet digital twin aggregates asset health across the energy "
            "portfolio for executive decision intelligence.",
            "digital-twin",
            ("fleet", "digital-twin", "portfolio", "executive"),
            {"text": "fleet digital twin asset health portfolio executive"},
        ),
        _DemoDocument(
            "doc-maintenance-policy",
            "Maintenance decision engine recommends condition-based intervention "
            "scheduling to minimise unplanned downtime cost.",
            "decision",
            ("maintenance", "policy", "downtime", "cost"),
            {"text": "maintenance decision condition based scheduling downtime cost"},
        ),
        _DemoDocument(
            "doc-scenario-plan",
            "Scenario planning agent simulates capital expenditure trade-offs for "
            "strategic portfolio optimization under demand uncertainty.",
            "decision",
            ("scenario", "capex", "strategy", "optimization"),
            {"text": "scenario planning capital expenditure strategic portfolio optimization"},
        ),
        _DemoDocument(
            "doc-root-cause",
            "Root cause analysis agent traces compressor failure to lubrication "
            "degradation and recommends corrective maintenance action.",
            "predictive-maintenance",
            ("root-cause", "compressor", "lubrication", "failure"),
            {"text": "root cause compressor failure lubrication maintenance"},
        ),
    ]


def _run_demo() -> int:
    """Run the deterministic command-line demonstration of the engine."""
    print("=" * 70)
    print("Enterprise Vector Search Engine -- Week 9 Phase 2 -- Demo")
    print("=" * 70)

    engine = VectorSearchEngine(
        embedding_engine=DeterministicEmbeddingEngine(dimension=DEFAULT_EMBEDDING_DIM),
        metric="cosine",
    )
    corpus = _demo_corpus()
    engine.index_documents(corpus)
    print(f"\nIndexed {engine.index.size()} documents "
          f"at dimension {engine.index.dimension}.")

    query = "compressor failure remaining useful life"
    print(f"\nSemantic search (cosine) for: {query!r}")
    for result in engine.search_top_k(query, k=3):
        print(f"  #{result.rank}  {result.document_id:<24} "
              f"score={result.score:.4f}  "
              f"category={result.metadata.get('category')}")

    print("\nCategory-filtered search ('predictive-maintenance'):")
    for result in engine.search_by_category(query, "predictive-maintenance", top_k=3):
        print(f"  #{result.rank}  {result.document_id:<24} score={result.score:.4f}")

    print("\nTag-filtered search (tags={'compressor'}):")
    for result in engine.search_by_tags(query, ["compressor"], top_k=3):
        print(f"  #{result.rank}  {result.document_id:<24} score={result.score:.4f}")

    print("\nHybrid search (keyword 0.4 / vector 0.6):")
    hybrid = HybridSearchEngine(engine, config=HybridConfig(0.4, 0.6, normalize=True))
    for result in hybrid.search(query, top_k=3):
        print(f"  #{result.rank}  {result.document_id:<24} "
              f"score={result.score:.4f}  "
              f"(kw={result.keyword_score:.3f}, vec={result.vector_score:.3f})")

    stats = engine.statistics()
    print("\nIndex statistics:")
    print(f"  document_count    = {stats.document_count}")
    print(f"  vector_dimension  = {stats.vector_dimension}")
    print(f"  index_density     = {stats.index_density:.4f}")
    print(f"  average_similarity= {stats.average_similarity:.4f}")

    payload = engine.to_dict()
    restored = VectorSearchEngine.from_dict(json.loads(json.dumps(payload)))
    same = [r.document_id for r in engine.search_top_k(query, 3)] == [
        r.document_id for r in restored.search_top_k(query, 3)
    ]
    print(f"\nJSON round-trip preserves ranking: {same}")
    print("=" * 70)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enterprise Vector Search Engine (Week 9 Phase 2)."
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