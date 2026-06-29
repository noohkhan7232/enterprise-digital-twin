# Week 9 — Phase 4: Enterprise Knowledge Agent

**Module:** `src/knowledge/knowledge_agent.py`
**Tests:** `tests/test_knowledge_agent.py` (366 tests)
**Status:** Complete — additive only. No Phase 1–3 file modified; the full four-phase regression is **997 tests, all green** (Phase 2: 300, Phase 3: 331, Phase 4: 366).

---

## 1. Purpose

The Knowledge Agent converts a natural-language question into a
`KnowledgeResponse`: a **grounded** answer, citations, a structured reasoning
trace, a calibrated confidence score, and — when confidence is low — suggested
follow-up questions.

**There is no LLM and no text generation.** Every sentence of every answer is
selected *verbatim* from the retrieval evidence produced by the Week 9 Phase 3
`RetrievalEngine`. The agent cannot invent facts: if there is no evidence, it
says so and lowers its confidence. The entire pipeline is deterministic (stable
hashing, deterministic tie-breaking, injectable clock).

---

## 2. Architecture

```
src/knowledge/
    enterprise_knowledge_base.py   # Phase 1 (FROZEN — duck-typed)
    vector_search_engine.py        # Phase 2 (FROZEN — composed)
    retrieval_intelligence.py      # Phase 3 (FROZEN — composed)
    knowledge_agent.py             # Phase 4 (NEW — this module)
```

The agent **composes** the lower layers by dependency injection: it holds a
Phase 3 `RetrievalEngine` (which itself wraps the Phase 2 vector/hybrid engines)
and never reaches around it. It imports the real `RetrievalEngine`,
`RetrievalConfig`, `RetrievalPackage`, `RetrievedDocument`, plus the shared
`Registry`/`tokenize`/`stable_hash` helpers, and reuses them unchanged. Nothing
in Phase 1–3 is modified, renamed, or duplicated, and no public API changes.

Dependency injection points: the retrieval engine, the `QuestionClassifier`, the
`AgentConfig`, and the `clock`. `KnowledgeAgent.from_knowledge_base(kb)` is the
convenience wiring that builds the Phase 3 engine and the agent together.

---

## 3. Knowledge Agent pipeline

```
question
  -> classification        (deterministic rule-based question understanding)
  -> retrieval             (Phase 3 RetrievalEngine.retrieve -> RetrievalPackage)
  -> evidence analysis     (sentence scoring by query-token overlap)
  -> response generation   (verbatim sentence selection — no generation)
  -> citation assembly     (one citation per contributing document, deduped)
  -> reasoning             (why-selected, strength, coverage, limitations)
  -> KnowledgeResponse     (+ confidence, + follow-ups when below threshold)
```

Each stage is a discrete method on `KnowledgeAgent`. The single public entry
point is `answer(question, *, config=None) -> KnowledgeResponse`; `classify()`
exposes the understanding stage in isolation.

---

## 4. Question understanding

`QuestionClassifier` is deterministic and rule-based (no ML, no LLM). Each of the
ten supported types is described by a frozen `QuestionTypeRule` registered in
`QUESTION_TYPE_REGISTRY`:

Maintenance, Inspection, Failure Analysis, Root Cause, Safety, Asset
Information, Operating Procedure, Scenario Guidance, Executive Summary, General
Knowledge.

Classification scores every rule (`+1` per keyword cue, `+2` per multi-word
phrase cue), selects the maximum, and breaks ties via a fixed priority order
(`QUESTION_TYPE_ORDER`) so safety/root-cause/failure outrank softer types. The
result is a `KnowledgeQuestion` carrying a deterministic `question_id`, the
inferred **intent**, **category**, **priority** (high/medium/low), **expected
evidence** cues, and a bounded classification **confidence**.

---

## 5. Evidence analysis & grounded response generation

The agent reads the ranked `RetrievedDocument` list from the Phase 3 package and
splits each top document's content into sentences. Sentences are scored by
query-token overlap; the highest-overlap sentences (deterministically tie-broken
by document rank then sentence index) are selected up to
`max_answer_sentences`, then re-ordered into natural reading order. When no
sentence overlaps the query, the agent falls back to the top-ranked document's
lead sentence so the answer is always anchored to the best retrieval hit.

The answer text is the concatenation of these verbatim sentences. `supported` is
true only when at least one document contributed. This is the core
anti-hallucination guarantee, enforced by tests that assert every answer
sentence appears verbatim in the source corpus.

---

## 6. Citation model

One `KnowledgeCitation` is emitted per document that contributed to the answer
(deduplicated; a duplicate triggers `DuplicateCitationError`). Each citation
carries: `document_id`, `title` (from metadata when available, else `None`),
`rank` (the retrieval rank), `confidence` (clamped document score),
`relevance_score` (raw document score), and a grounded `snippet` (the document's
lead sentence). Citations are ordered by rank, capped at `max_citations`.

---

## 7. Reasoning engine

`KnowledgeReasoning` provides an auditable trace: per-document **why-selected**
statements (rank + relevance), **evidence strength** (mean citation relevance),
**coverage** (from the Phase 3 evidence), the overall **confidence**, and a
deterministic list of **limitations** — e.g. no evidence retrieved, coverage
below half the query terms, single-source reliance, confidence below the
effective threshold, or low underlying retrieval confidence.

---

## 8. Confidence model

Overall confidence is a fixed convex combination of four signals, each in
`[0, 1]`, clamped to `[0, 1]`:

```
confidence = 0.40 · retrieval_confidence
           + 0.30 · coverage
           + 0.20 · citation_quality          (mean citation relevance)
           + 0.10 · evidence_consistency       (mean pairwise snippet Jaccard)
```

`retrieval_confidence` and `coverage` come from the Phase 3 evidence;
`citation_quality` and `evidence_consistency` are computed over the assembled
citations (a single citation is trivially consistent at `1.0`; no citations
yields `0.0`). When confidence falls below `AgentConfig.confidence_threshold`,
the agent generates up to `max_follow_ups` deterministic follow-up questions from
`FOLLOWUP_REGISTRY`, templated on the first query term not covered by the
evidence.

---

## 9. Design decisions & trade-offs

* **Extraction, not generation.** Verbatim sentence selection makes every answer
  trivially auditable and eliminates hallucination by construction. Trade-off:
  answers read as assembled evidence rather than prose synthesis — exactly the
  intended behaviour for a deterministic, no-LLM enterprise agent.
* **Rule-based classification.** Transparent, deterministic, and editable via the
  registry. Trade-off: not as flexible as a learned classifier, but fully
  explainable and dependency-free.
* **Dependency injection.** The agent accepts an injected retrieval engine,
  classifier, config, and clock, so it is testable in isolation and composes
  cleanly with the frozen lower layers. The injected clock makes latency
  reproducible in tests while defaulting to `time.perf_counter`.
* **Graceful vs strict evidence.** By default, a question with no evidence yields
  an honest unsupported response (confidence `0`, follow-ups); setting
  `require_evidence=True` turns that into a `MissingEvidenceError` for callers
  that must fail closed.
* **Effective per-call threshold.** Reasoning limitations and follow-up gating
  both honour the threshold from the per-call `AgentConfig`, keeping a single
  source of truth for "low confidence".

---

## 10. Performance considerations

* The agent adds only sentence splitting and token-set overlap on top of one
  Phase 3 retrieval pass; cost is `O(C · S)` over the `C ≤ max_citations`
  documents and their `S` sentences — independent of corpus size.
* The 600-document large-KB test answers in milliseconds and is deterministic
  across rebuilds and JSON round trips.
* Responses are fully JSON-serialisable, so agent runs can be cached, replayed,
  and audited without recomputation.

---

## 11. Validation & error model

`KnowledgeAgentError` root with `EmptyQuestionError`, `InvalidQuestionError`,
`MissingRetrievalError`, `DuplicateCitationError`, and `MissingEvidenceError`.
Validated conditions: empty / whitespace / punctuation-only / stopword-only
questions, non-string questions, a missing retrieval engine, duplicate
citations, and (optionally) missing evidence.

---

## 12. Testing

`tests/test_knowledge_agent.py` — **366 tests** covering question
classification and intent detection (all ten types), retrieval integration,
grounded response generation (verbatim/anti-hallucination assertions), citation
generation, reasoning generation, the confidence model, follow-up generation,
serialization of all six core dataclasses plus the agent, statistics, edge
cases, a 600-document knowledge base, determinism, backward compatibility with
Phases 2–3, and the CLI.

```bash
python -m unittest tests.test_knowledge_agent         # 366 tests
python src/knowledge/knowledge_agent.py --demo        # deterministic demo
```

---

## 13. Usage example

```python
from knowledge.knowledge_agent import KnowledgeAgent, AgentConfig

agent = KnowledgeAgent.from_knowledge_base(knowledge_base)
resp = agent.answer("What is the root cause of the compressor failure?")

print(resp.answer.text)                  # grounded, verbatim from evidence
for c in resp.citations:
    print(c.rank, c.document_id, c.title, round(c.relevance_score, 3))
print(resp.confidence, resp.reasoning.limitations)
print(resp.follow_up_questions)          # populated only when confidence is low

agent.statistics()                       # cumulative operational metrics
```

---

## 14. Forward integration — Week 9 Phase 5 (Executive RAG Copilot)

The Knowledge Agent is the grounding tier beneath the Executive RAG Copilot. A
Phase 5 Copilot will:

1. decompose an executive question into sub-questions and call `answer()` for each;
2. compose a briefing strictly from the returned `KnowledgeResponse` answers and
   `citations`, preserving end-to-end traceability;
3. use per-response `confidence`, `reasoning.limitations`, and `follow_up_questions`
   to decide when to widen retrieval or surface caveats to the executive;
4. roll up `KnowledgeAgentStatistics` into copilot-level reporting.

Because every `KnowledgeResponse` is deterministic and JSON-serialisable, the
Copilot's briefings remain fully reproducible and auditable — and, as with every
phase so far, this is achieved without modifying any frozen module.