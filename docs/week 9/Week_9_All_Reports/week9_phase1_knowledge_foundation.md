# Week 9 — Phase 1: Enterprise Knowledge Foundation

## Architecture Documentation

**Component:** `src/knowledge/enterprise_knowledge_base.py`
**Role:** The platform's foundational enterprise knowledge layer
**Inputs:** Typed knowledge documents (manuals, procedures, catalogs, playbooks, reports)
**Outputs:** Ranked search results, collections, statistics — all frozen and JSON-serialisable
**Status:** Implemented and validated — 184 tests passing with zero skips; all Week 5–8 suites unchanged and coexisting (18/18 modules import cleanly).

---

## 1. Purpose

Through Week 8 the platform reasoned exceptionally well over *live* asset state — it could predict, simulate, decide, explain, attribute causes, and plan scenarios. What it lacked was *institutional memory*: the body of maintenance manuals, inspection and operating procedures, failure catalogs, root-cause playbooks, safety procedures, asset specifications, and prior engineering and executive reports that an enterprise accumulates and that experienced engineers draw on. Week 9 Phase 1 introduces that memory as a first-class, structured layer.

This phase is deliberately a *foundation*, not the finished knowledge system. There is no database, no vector store, no language model, and no embeddings. All retrieval and scoring are exact, lexical, and rule-based. The reasons are the same ones that have governed every prior layer: determinism (identical queries return identical, reproducible results), auditability (every score traces to a token-overlap rule, not an opaque model), and zero external dependencies (pure Python and NumPy). This foundation is what the Week 9 Phase 2 vector search engine will build on — the document contract, the registry, and the relevance interface defined here are forward-compatible with an embedding layer that ranks the same documents by semantic similarity rather than lexical overlap.

---

## 2. The Knowledge Document

The atomic unit is the frozen `KnowledgeDocument`, carrying the ten fields the specification requires: `document_id`, `title`, `document_type`, `source`, `category`, `tags`, `content`, `created_at`, `version`, and `metadata`. Construction validates rigorously — a document must have a non-empty id, title, and content; its category must be one of the recognised values; its tags are normalised to a lowercase tuple; and its metadata must be a JSON-safe mapping. The validation runs in `__post_init__` and normalises in place, so a constructed document is always well-formed and immutable. Because the record is frozen, it can be shared freely across the platform without any risk of aliasing or mutation — the same discipline that has governed every dataclass in the system.

Ten categories are supported: maintenance manual, inspection procedure, operating procedure, failure catalog, root-cause playbook, safety procedure, asset specification, engineering report, executive report, and unknown. The `unknown` category is the explicit residual for documents whose type is not yet classified, and it is excluded from the coverage metric so that uncategorised material does not inflate apparent corpus breadth.

---

## 3. The Knowledge Registry

The `EnterpriseKnowledgeBase` holds an in-memory registry of documents keyed by id and exposes the four required operations. `register_document` rejects a non-document and rejects a duplicate id (the first line of defence against corpus corruption). `remove_document` returns the removed record and raises on an unknown id. `get_document` raises on an unknown id rather than returning `None`, so a missing document is a loud error, not a silent one. `list_documents` returns documents sorted by id — deterministically — and supports optional category and tag filters. A class-level registry (`KNOWLEDGE_BASE_REGISTRY` with register/build/list helpers) follows the platform convention so alternative knowledge-base implementations can be discovered by name.

---

## 4. Search

Four exact, deterministic search modes are provided. **Keyword search** tokenises the query (lowercase, split on non-alphanumeric boundaries, stop-words removed) and scores each document by a weighted overlap of query tokens against the document's title, content, and tag tokens; results are ranked by score descending with ties broken by id. **Tag search** ranks by fractional tag overlap and supports an all-tags-required mode. **Category search** returns every document in a category at unit score. **Exact-match search** returns documents containing a case-insensitive phrase in the chosen fields. Every mode returns the same frozen `SearchResult` shape — id, title, category, score, and matched terms — so a caller (and, later, the vector engine) sees one uniform ranked-result contract.

---

## 5. Relevance Scoring

The keyword score is a transparent weighted overlap. For a query token set `Q` and a document's title, content, and tag token sets, the score is

```
score = ( w_title·|Q ∩ title| + w_content·|Q ∩ content| + w_tag·|Q ∩ tag| )
        / ( |Q| · (w_title + w_content + w_tag) )
```

with default weights of 3, 1, and 2 for title, content, and tags. Dividing by `|Q|` times the weight sum guarantees the score lands in `[0, 1]`: a document that matches every query term in all three fields scores exactly 1, while a content-only match scores a small fraction. Title matches are weighted highest because a term in the title is a stronger relevance signal than the same term buried in the body. The combined `relevance_score` blends keyword overlap, tag overlap, and category match with configurable component weights, over whichever signals the caller supplies — so the same scoring interface serves single-signal and multi-signal queries alike. Every computation is closed-form and deterministic; there is no randomness anywhere in the layer.

---

## 6. Collections

Three collection types are supported, each producing a frozen `KnowledgeCollection`. A **document collection** is an explicit, named set of ids (validated to exist). A **category collection** gathers every document in a category. A **fleet collection** gathers documents associated with a fleet, where association is declared either through a `fleet_id` in metadata or through a matching tag. Collections are stored by name on the knowledge base and are themselves serialisable, so a curated view of the corpus — "everything relevant to the North Sea fleet", say — is a first-class, persistable object.

---

## 7. Statistics

`statistics` produces a frozen `KnowledgeStatistics` carrying the document count, the category and tag distributions (each sorted most-frequent-first for stable presentation), the number of distinct tags, the count of non-unknown categories represented, and a coverage score. Coverage is the fraction of the nine non-unknown categories that have at least one document — a single, interpretable measure of how complete the institutional memory is. A corpus with maintenance manuals and failure catalogs but no safety procedures scores below one, surfacing the gap.

---

## 8. Validation and Serialization

Validation is layered. At the document level, `__post_init__` rejects empty ids, titles, and content; invalid categories; malformed tags; and non-JSON-safe metadata. At the registry level, `register_document` rejects duplicates and non-documents. At the collection level, `make_collection` rejects unknown ids and empty names. Serialization is complete and symmetric: every frozen type exposes `to_dict`, the document and the whole knowledge base both support `from_dict`, and a `to_json` convenience is provided. A knowledge base survives a full `to_dict`/`from_dict` round-trip — documents, collections, and metadata intact — and the round-tripped base is byte-identical in its search behaviour, which the test suite verifies.

---

## 9. Validation Summary

184 tests pass with zero skips, covering the category enum and normalisation; document construction, validation, and serialisation; config validation; the four registry operations; all four search modes (ranking, bounding, filtering, edge cases); relevance scoring (each component, weighting, and determinism); the three collection types; statistics (counts, distributions, coverage, the unknown-exclusion rule); full serialization round-trips including via JSON; determinism across instances; the failure-safe tracker; the class registry; and scale and edge cases (500-document corpora, unicode, very long content, many tags, register-remove-register, and empty bases). All Week 5–8 suites continue to pass unchanged, and all eighteen platform modules import and coexist cleanly.

---

## 10. Forward Path to Phase 2

This layer is engineered to become the substrate for vector search. Three design choices make that transition clean. First, the `KnowledgeDocument` is the stable unit an embedding layer will encode — no field needs to change for a vector to be attached. Second, the `SearchResult` contract already expresses "ranked documents with scores", so a semantic ranker can return the identical shape and be swapped in behind the same call sites. Third, the `relevance_score` interface already abstracts "how relevant is this document to this query", so Phase 2 can add a semantic component alongside the existing lexical ones rather than replacing the scoring path. The result is that Week 9 Phase 2 will extend this foundation — adding embeddings and approximate nearest-neighbour ranking — without modifying the document model, the registry, or the result contract established here.