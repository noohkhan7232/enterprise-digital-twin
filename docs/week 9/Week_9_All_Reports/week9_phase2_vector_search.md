# Week 9 — Phase 2: Enterprise Vector Search Engine

**Module:** `src/knowledge/vector_search_engine.py`
**Tests:** `tests/test_vector_search_engine.py` (300 tests)
**Status:** Complete — additive only, no Week 1–9 Phase 1 file modified.

---

## 1. Purpose

Phase 2 adds the **semantic retrieval foundation** for the Enterprise Knowledge
Layer. The Enterprise Knowledge Base (Week 9 Phase 1) already supports keyword,
category, tag, and exact search. This phase introduces dense-vector semantic
search and weighted hybrid retrieval that together become the substrate for the
upcoming Enterprise RAG layer.

The engine is pure Python + NumPy, fully deterministic, dependency-free (no
FAISS / Pinecone / Weaviate / Chroma / OpenAI / HuggingFace / network), and
serialises cleanly to JSON.

---

## 2. Position in the architecture

```
src/knowledge/
    enterprise_knowledge_base.py   # Week 9 Phase 1 (FROZEN — untouched)
    vector_search_engine.py        # Week 9 Phase 2 (NEW — this module)
```

The new module **composes** the Phase 1 models rather than re-implementing
them. It does not redefine, rename, or subclass `KnowledgeDocument`,
`SearchResult`, or `EnterpriseKnowledgeBase`, and it changes no public API.

### Integration contract (duck-typed adapter)

Because the Phase 1 source is frozen, integration is performed through a thin,
documented adapter that reads logical fields from any `KnowledgeDocument`-like
object using a prioritised list of attribute names. This keeps the engine
backward compatible regardless of incidental naming, and is verified by tests
(`TestKnowledgeBaseIntegration`).

| Logical field | Attribute names tried (in order)              |
| ------------- | --------------------------------------------- |
| identifier    | `document_id`, `doc_id`, `id`, `key`          |
| text body     | `content`, `text`, `body`, `summary`          |
| category      | `category`, `categories`                      |
| tags          | `tags`, `labels`                              |
| metadata      | `metadata`, `meta`, `attributes`              |

`EnterpriseKnowledgeBase` instances are consumed via
`VectorSearchEngine.from_knowledge_base(kb)`, which discovers the document
collection through `all_documents()` / `list_documents()` / `documents()` /
`iter_documents()` / a `documents` attribute / a plain iterable.

> If the real Phase 1 field names differ from every fallback above, only the
> `_*_ATTRS` tuples at the top of the module need a one-line extension — no
> structural change is required.

---

## 3. Components

### Value objects (frozen, JSON-serialisable)

| Type                 | Fields                                                            |
| -------------------- | ---------------------------------------------------------------- |
| `VectorDocument`     | `document_id`, `vector`, `metadata`                              |
| `VectorSearchResult` | `document_id`, `score`, `rank`, `metadata`                       |
| `HybridSearchResult` | `document_id`, `score`, `rank`, `keyword_score`, `vector_score`, `metadata` |
| `IndexStatistics`    | `document_count`, `vector_dimension`, `index_density`, `average_similarity` |
| `HybridConfig`       | `keyword_weight`, `vector_weight`, `normalize`                   |

Every value object exposes `to_dict()` / `from_dict()` and round-trips through
`json.dumps` / `json.loads`.

### `DeterministicEmbeddingEngine`

Deterministic dense embeddings (default dimension **128**) with no external
dependency. Algorithm:

1. **Tokenise** text into lowercase alphanumeric tokens.
2. **Feature expansion** — each token contributes a whole-token feature plus
   character n-grams (default 3–5) for subword signal.
3. **Signed feature hashing** — every feature is hashed with `stable_hash`
   (BLAKE2b, 64-bit). The hash selects a bucket in `[0, dim)` and a sign from
   independent bits (the hashing-trick, keeping the expected dot product
   unbiased).
4. **Scatter-add + L2 normalise** — counts are accumulated with `numpy.add.at`
   and the vector is L2-normalised.

Methods: `embed_text()`, `embed_document()`, `embed_documents()`. Identical
input always yields an identical vector, across processes and machines (Python's
salted built-in `hash` is deliberately avoided).

### `VectorIndex`

Insertion-ordered in-memory store enforcing a single dimension, rejecting
duplicate ids / invalid vectors, with a lazily materialised `(n, d)` matrix
cache that is invalidated on every mutation.

API: `add_document`, `remove_document`, `update_document`, `get_document`,
`has_document`, `size`, plus `ids()`, `documents()`, `matrix()`, and
`to_dict` / `from_dict`.

### Similarity metrics (vectorised, registered)

| Name        | Function                  | Range        | Notes                                   |
| ----------- | ------------------------- | ------------ | --------------------------------------- |
| `cosine`    | `cosine_similarity`       | `[-1, 1]`    | Zero-norm vectors score `0.0` (no NaN). |
| `euclidean` | `euclidean_similarity`    | `(0, 1]`     | `1 / (1 + ‖row − q‖₂)`, monotonic.      |
| `dot`       | `dot_product_similarity`  | unbounded    | Raw inner product.                      |

All three are registered in `SIMILARITY_REGISTRY` and selectable per query.

### `VectorSearchEngine`

Composes embedding engine + index + metric.

* `search(query, top_k=10, *, metric=None, candidate_ids=None)` — `query` may be
  text (embedded on the fly) or a precomputed vector.
* `search_top_k(query, k, *, metric=None)`
* `search_by_category(query, category, top_k=10, *, metric=None)`
* `search_by_tags(query, tags, top_k=10, *, match_all=False, metric=None)`
* `statistics()`, `to_dict()` / `from_dict()`, `from_knowledge_base(...)`.

Ranking is deterministic: descending score with ascending `document_id` as the
tie-breaker.

### `HybridSearchEngine`

Weighted combination of a keyword signal and the vector signal:

```
combined = keyword_weight · norm(keyword_score) + vector_weight · norm(vector_score)
```

Weights and min-max normalisation are configurable via `HybridConfig`. The
keyword signal is supplied by an injectable provider
`(query: str) -> Mapping[str, float]`, so it can be backed by the Enterprise
Knowledge Base's existing keyword search **without this module reaching into or
duplicating that implementation**. When no provider is given, a deterministic
token-overlap fallback is used over indexed metadata.

---

## 4. CLI demo

```bash
python src/knowledge/vector_search_engine.py --demo
```

The demo indexes a six-document platform-themed corpus and prints semantic
search, category-filtered search, tag-filtered search, hybrid search, index
statistics, and a JSON round-trip equivalence check — all deterministic.

---

## 5. Validation & error model

A typed exception hierarchy rooted at `VectorSearchError`:

`DimensionMismatchError`, `DuplicateDocumentError`, `DocumentNotFoundError`,
`EmptyVectorError`, `InvalidVectorError`, `InvalidMetadataError`.

Validated conditions: dimension mismatch, duplicate ids, empty (zero-length)
vectors, invalid vectors (NaN/Inf/non-1-D/non-numeric), and non-JSON-serialisable
metadata.

---

## 6. Index statistics

`compute_statistics(index, max_pairs=4096)` reports `document_count`,
`vector_dimension`, `index_density` (mean fraction of non-zero components), and
`average_similarity` (mean off-diagonal cosine over a deterministic
evenly-spaced subsample bounded by `max_pairs`).

---

## 7. Engineering compliance

| Constraint                  | Status                                              |
| --------------------------- | --------------------------------------------------- |
| Frozen dataclasses          | ✅ all value objects                                 |
| Registry pattern            | ✅ `SIMILARITY_REGISTRY`, `EMBEDDING_REGISTRY`       |
| Pure Python + NumPy         | ✅ no other runtime dependency                       |
| Deterministic outputs       | ✅ BLAKE2b hashing, deterministic tie-breaking       |
| JSON serialisable           | ✅ `to_dict` / `from_dict` everywhere                |
| No external vector DB / API | ✅ none used                                         |
| Production docstrings       | ✅ module, classes, methods                          |
| CLI demo                    | ✅ `--demo`                                          |
| Backward compatibility      | ✅ additive only; no Phase 1 file touched            |

---

## 8. Testing

`tests/test_vector_search_engine.py` — **300 tests** across 28 classes,
covering embeddings, indexing, search, hybrid retrieval, similarity metrics,
serialization, statistics, validation, knowledge-base integration, a
1,000-document corpus, determinism, the CLI, and edge cases.

```bash
python -m unittest tests.test_vector_search_engine        # or: pytest tests/
```

All tests pass deterministically on repeated runs.

---

## 9. Usage example

```python
from knowledge.vector_search_engine import (
    VectorSearchEngine, DeterministicEmbeddingEngine,
    HybridSearchEngine, HybridConfig,
)

# Build directly from the existing Enterprise Knowledge Base.
engine = VectorSearchEngine.from_knowledge_base(knowledge_base)

# Semantic search.
for hit in engine.search_top_k("compressor remaining useful life", k=5):
    print(hit.rank, hit.document_id, round(hit.score, 4))

# Filtered semantic search.
engine.search_by_category("turbine vibration", "predictive-maintenance")
engine.search_by_tags("turbine vibration", ["compressor"], match_all=True)

# Hybrid retrieval, reusing the knowledge base's keyword scorer.
hybrid = HybridSearchEngine(
    engine,
    keyword_provider=lambda q: knowledge_base.keyword_scores(q),  # any lexical signal
    config=HybridConfig(keyword_weight=0.4, vector_weight=0.6),
)
hybrid.search("compressor failure", top_k=5)

# Persist and restore.
import json
blob = json.dumps(engine.to_dict())
restored = VectorSearchEngine.from_dict(json.loads(blob))
```

---

## 10. Forward path (Week 9 Phase 3 — Enterprise RAG)

This engine is the retrieval primitive for RAG: `from_knowledge_base` builds the
vector store from existing documents, `HybridSearchEngine` provides the
candidate ranker, and `VectorSearchResult.metadata` carries the context payload
needed for grounded generation — all without modifying any frozen module.