# Week 9 — Phase 3: Enterprise Retrieval Intelligence Layer

**Module:** `src/knowledge/retrieval_intelligence.py`
**Tests:** `tests/test_retrieval_intelligence.py` (331 tests)
**Status:** Complete — additive only. No Week 1–9 Phase 2 file modified; Phase 2's 300 tests remain green (631 tests total across both phases).

---

## 1. Purpose

Phase 2 delivered semantic and hybrid *search*. Phase 3 delivers retrieval
*intelligence*: it converts a raw query into a ranked, scored, fully traced
**retrieval package** — assembled context, structured evidence, and operational
statistics — that the future Enterprise RAG / Knowledge Agent (Phase 4) can
consume directly, with no search-time logic left to implement downstream.

The layer is pure Python + NumPy, deterministic (including latency, via an
injectable clock), dependency-free (no LLM / OpenAI / HuggingFace / external API
/ vector database), and fully JSON-serialisable.

---

## 2. Architecture

```
src/knowledge/
    enterprise_knowledge_base.py   # Phase 1 (FROZEN — untouched, duck-typed)
    vector_search_engine.py        # Phase 2 (FROZEN — imported & composed)
    retrieval_intelligence.py      # Phase 3 (NEW — this module)
```

Phase 3 **composes** Phase 2 rather than reaching around it. It imports the real
`VectorSearchEngine`, `HybridSearchEngine`, `HybridConfig`, `HybridSearchResult`,
`Registry`, and the shared `tokenize`/duck-typed document accessors, and reuses
them unchanged. The Phase 1 knowledge base is consumed through the same
documented duck-typed adapter established in Phase 2 (id/text/category/tags/
metadata resolved through prioritised attribute names), so nothing in Phase 1 or
Phase 2 is renamed, rewritten, or duplicated.

Component ownership:

| Concern              | Owner                                  |
| -------------------- | -------------------------------------- |
| Embedding            | Phase 2 `DeterministicEmbeddingEngine` |
| Vector / hybrid scan | Phase 2 `VectorSearchEngine` / `HybridSearchEngine` |
| Query processing     | Phase 3 `QueryNormalizer`              |
| Candidate filtering  | Phase 3 `RetrievalEngine`              |
| Evidence ranking     | Phase 3 ranking registry + `RankingWeights` |
| Context / evidence   | Phase 3 value objects                  |

---

## 3. Retrieval pipeline

```
query
  -> query normalization      (lowercase, token cleanup, dedupe, stopword/noise filtering)
  -> hybrid retrieval         (Phase 2 HybridSearchEngine -> candidate pool)
  -> candidate generation     (attach indexed metadata, trace provenance)
  -> candidate filtering      (category, tags, min-score, duplicate removal)
  -> evidence ranking         (weighted 5-component score, deterministic order)
  -> context assembly         (ordered evidence, confidence, source tracking)
  -> retrieval package        (context + evidence + statistics + config)
```

Each stage is a small, independently testable method on `RetrievalEngine`. The
single public entry point is `retrieve(query, *, config=None) -> RetrievalPackage`
(`retrieve_context` / `retrieve_evidence` are convenience wrappers).

### Query processing

`QueryNormalizer` is a frozen, deterministic transformer: lower-casing and
punctuation stripping (via the shared tokenizer), configurable stop-word removal
(compact built-in English set), minimum-token-length noise filtering, and
order-preserving duplicate removal. Identical raw queries always normalise
identically.

### Candidate generation & filtering

Candidates come from one Phase 2 hybrid search over the normalised query (bounded
by `candidate_k`). Filtering then applies category match, tag match (any/all),
a `min_score` floor on the hybrid score, and a duplicate-id guard. Filtering in
this layer (rather than via Phase 2's vector-only filtered search) preserves both
the keyword and vector component scores for ranking.

---

## 4. Ranking strategy

Final relevance is a configurable weighted combination of five components, each
optionally min-max normalised across the candidate set before weighting:

| Component  | Source / definition                                                      |
| ---------- | ------------------------------------------------------------------------ |
| vector     | Phase 2 hybrid vector component                                          |
| keyword    | Phase 2 hybrid keyword component                                         |
| metadata   | Overlap of query tokens with tags + category + scalar metadata (excludes the `text` field, which keyword already covers) |
| category   | 1.0 when the requested category matches (acts as a gate under filtering) |
| freshness  | Explicit `freshness∈[0,1]` → `age_days` decay `1/(1+age/30)` → neutral `0.5` |

Weights are expressed with `RankingWeights` (non-negative, at least one
positive, auto-normalisable). Strategies live in a `RANKING_REGISTRY`
(`weighted_sum` default, `max` provided; custom strategies register via
`@register_ranking_strategy`). Ordering is deterministic: score descending with
ascending `document_id` as the tie-breaker, producing a contiguous 1-based rank
sequence (asserted as an internal invariant).

---

## 5. Evidence assembly

`RetrievalContext` carries the ordered `RetrievedDocument` evidence, an assembled
confidence, source tracking, and assembly metadata; `as_text()` renders a
numbered context block ready for a RAG prompt. `RetrievalEvidence` separates the
load-bearing **top documents** (first `top_evidence` ranks) from **supporting
documents**, and adds:

* **confidence** = `0.7·mean(top scores) + 0.3·clamp(score margin between #1 and #2)`, clamped to `[0, 1]`.
* **coverage** = fraction of query tokens covered by the union of the evidence's content + tags + category tokens.
* **reasoning** = a trace dict (query tokens, documents searched, candidate/filtered counts, strategy, top score).

`RetrievalStatistics` reports latency, documents searched/returned, average
score, coverage, and confidence. `RetrievalPackage` bundles context + evidence +
statistics + the effective config; every object round-trips through
`to_dict`/`from_dict` and JSON.

---

## 6. Engineering decisions & trade-offs

* **Compose, don't fork.** The layer imports Phase 2 directly (including a robust
  package/script dual-import) and reuses its registry, tokenizer, and document
  accessors. Trade-off: a hard dependency on Phase 2's public surface — accepted,
  since that surface is frozen and versioned with the platform.
* **Duck-typed Phase 1 integration.** Retains backward compatibility across
  incidental field-name differences without importing or modifying the frozen
  knowledge base. Trade-off: field resolution is by convention, documented and
  test-covered, rather than statically typed.
* **Injectable clock.** Latency is the only non-value-deterministic quantity; an
  injectable `clock` makes even latency reproducible in tests while defaulting to
  `time.perf_counter` in production. Trade-off: a small constructor parameter for
  full determinism — worth it.
* **Filtering after hybrid.** Keeps both component scores available for ranking
  at the cost of scoring a slightly larger candidate pool; bounded by `candidate_k`.
* **Category as a gate.** Because category filtering removes non-matching docs,
  the surviving `category_score` is constant; the component is retained for
  explicitness and future soft-preference use, with weight controllable.
* **Min-max normalisation.** Makes heterogeneous component scales comparable and
  matches Phase 2's hybrid convention (constant input → all `1.0`). Trade-off:
  scores become relative to the candidate set; `normalize_components=False`
  exposes absolute component values when needed.

---

## 7. Performance considerations

* Candidate generation is one Phase 2 hybrid pass (vectorised cosine over the
  cached index matrix); ranking is `O(C · 5)` over the `C ≤ candidate_k`
  survivors, dominated by cheap token-set operations.
* `candidate_k` bounds ranking work independently of corpus size; the
  800-document large-corpus test exercises this and runs in milliseconds.
* All vector math stays in NumPy; no per-document Python loops over embeddings at
  query time beyond candidate metadata assembly.
* Determinism holds across engine rebuilds and JSON round trips (explicitly
  tested), so retrieval packages can be cached and replayed safely.

---

## 8. Validation & error model

A typed hierarchy rooted at `RetrievalError`: `EmptyQueryError`,
`InvalidQueryError`, `InvalidRankingError`, `MissingDocumentError`,
`DuplicateRetrievalError`. Validated conditions: empty/whitespace/stopword-only
queries, non-string queries, invalid ranking weights / unknown strategy,
candidates missing from the index, and duplicate ids in assembled evidence.

---

## 9. Testing

`tests/test_retrieval_intelligence.py` — **331 tests** covering query
normalization, the full pipeline, candidate generation/filtering, ranking
(per-component math and ordering), context assembly, evidence generation,
statistics, serialization, an 800-document corpus, determinism, edge cases, the
CLI, and explicit Phase 1 + Phase 2 integration and backward-compatibility
guards.

```bash
python -m unittest tests.test_retrieval_intelligence        # 331 tests
python -m unittest tests.test_vector_search_engine          # 300 tests (Phase 2, still green)
python src/knowledge/retrieval_intelligence.py --demo       # deterministic demo
```

---

## 10. Usage example

```python
from knowledge.retrieval_intelligence import RetrievalEngine, RetrievalConfig, RankingWeights

engine = RetrievalEngine.from_knowledge_base(knowledge_base)

pkg = engine.retrieve("What causes compressor failure and remaining useful life?")
for d in pkg.context.documents:
    print(d.rank, d.document_id, round(d.score, 3))

prompt_context = pkg.context.as_text()          # numbered evidence block for RAG
top = pkg.evidence.top_documents                # load-bearing evidence
print(pkg.statistics.confidence, pkg.statistics.coverage)

# Tuned run: predictive-maintenance only, vector-leaning ranking.
cfg = RetrievalConfig(
    top_k=5, category="predictive-maintenance",
    ranking_weights=RankingWeights(vector=0.6, keyword=0.25, metadata=0.1, category=0.0, freshness=0.05),
)
engine.retrieve("compressor failure", config=cfg)
```

---

## 11. Forward integration — Week 9 Phase 4 (Knowledge Agent)

This layer is the agent's retrieval tool. A Phase 4 Knowledge Agent will:

1. call `retrieve()` to obtain a `RetrievalPackage` for each sub-question;
2. ground its response in `context.as_text()` and cite `evidence.top_documents`;
3. gate answers on `confidence_score` / `coverage_score` (e.g. ask a follow-up or
   widen `candidate_k` when coverage is low);
4. use `reasoning` and `statistics` for transparent, auditable decision traces.

Because the package is deterministic and JSON-serialisable, agent runs are
fully reproducible and replayable — all without modifying any frozen module.