# Week 9 — Phase 5: Executive RAG Copilot

**Module:** `src/knowledge/executive_rag_copilot.py`
**Tests:** `tests/test_executive_rag_copilot.py` (405 tests)
**Status:** Complete — additive only. No Week 1–9 Phase 4 file was modified, no
class renamed, no public API changed. The full five-phase regression is
**1,402 tests, all green** (Phase 2: 300, Phase 3: 331, Phase 4: 366, Phase 5:
405).

---

## 1. Purpose

The Executive RAG Copilot is the executive-facing orchestration tier of the
Enterprise Knowledge Layer. It converts a high-level executive question into a
fully grounded :class:`ExecutiveBriefing`: an executive summary, key findings,
business risks, structured insights, deterministic recommendations, aggregated
citations, a calibrated confidence assessment, and explicit limitations.

**It is not an LLM chatbot.** There is no generative model and no invented
content. The Copilot is a deterministic *orchestrator*: it classifies the
question, expands it into focused sub-questions, delegates all retrieval and
knowledge reasoning to the frozen Phase 4 `KnowledgeAgent`, and then aggregates
and frames those grounded results. Every factual sentence in a briefing is
selected verbatim from Knowledge Agent evidence, and citations are reused
directly from the agent.

---

## 2. Architecture

```
src/knowledge/
    enterprise_knowledge_base.py   # Phase 1 (FROZEN — duck-typed)
    vector_search_engine.py        # Phase 2 (FROZEN — composed)
    retrieval_intelligence.py      # Phase 3 (FROZEN — composed)
    knowledge_agent.py             # Phase 4 (FROZEN — composed)
    executive_rag_copilot.py       # Phase 5 (NEW — this module)
```

The Copilot composes the entire stack by **dependency injection** and reaches no
deeper than the `KnowledgeAgent`: it holds an injected agent, an
`ExecutiveQuestionClassifier`, a `CopilotConfig`, and a `clock`. It imports
`KnowledgeAgent`, `AgentConfig`, `KnowledgeResponse`, and `KnowledgeCitation`
from Phase 4, plus the shared `Registry` / `tokenize` / `stable_hash` helpers and
the Phase 3 `QueryNormalizer`, and reuses them unchanged.

The central design discipline is **orchestration, not reimplementation**. The
Copilot contains no retrieval code, no scoring code, and no citation-extraction
code; those concerns live in the frozen layers. `ExecutiveCitation` is a thin
adapter (`from_knowledge_citation`) over the agent's `KnowledgeCitation` — the
citation system is reused, never duplicated.

---

## 3. Executive Copilot pipeline

```
executive question
  -> classification          (deterministic rule-based, + sub-question expansion)
  -> knowledge-agent invocation   (one KnowledgeAgent.answer per sub-question)
  -> evidence aggregation    (dedupe + rank citations across sub-answers)
  -> insight extraction      (structured, evidence-grounded insights)
  -> recommendation generation    (deterministic templates + grounded rationale)
  -> briefing assembly       (summary, findings, risks, actions, evidence)
  -> confidence assessment   (four-signal convex combination)
  -> ExecutiveBriefing
```

The single public entry point is
`brief(question, *, question_type=None, config=None) -> ExecutiveBriefing`;
`classify()` exposes the classification stage in isolation.

---

## 4. Knowledge Agent integration

For each sub-question the Copilot calls `KnowledgeAgent.answer(sub_q, config=...)`
and collects the resulting `KnowledgeResponse` objects. From those it reuses:

* **Citations** — each `KnowledgeCitation` is adapted into an
  `ExecutiveCitation` carrying the originating sub-question; no new citation
  logic is written.
* **Coverage and confidence** — the agent's per-response `reasoning.coverage` and
  `confidence` feed the executive confidence model.
* **Limitations** — agent-level limitations are rolled up (deduplicated) into the
  briefing's limitations.

A malformed sub-question never breaks a briefing: the invocation is guarded so a
single failed sub-question is skipped rather than aborting the pipeline.

---

## 5. Insight engine

One `ExecutiveInsight` is extracted per top aggregated citation (capped by
`max_insights`). Each insight contains: a **title** (the cited document's title,
or its id), a **summary** (the citation's grounded snippet — verbatim evidence),
a deterministic **business impact** framing (priority bucket + focus + relevance
score), **supporting evidence** (document ids), a **priority**
(`high`/`medium`/`low` from the relevance bucket), and a **confidence** (the
citation confidence). Because the summary is the verbatim agent snippet, insights
cannot hallucinate.

---

## 6. Recommendation engine

One `ExecutiveRecommendation` is generated per top insight (capped by
`max_recommendations`). Each contains a **recommended action**, a **business
rationale**, an **expected impact**, **supporting citations**, a **priority**, and
a **confidence**. The design keeps recommendations grounded while still being
actionable:

* The **recommended action** is a deterministic per-question-type template
  (`RECOMMENDATION_TEMPLATE_REGISTRY`) instantiated on a topic drawn from the
  cited evidence (the insight's document title). This is an imperative action,
  not a factual claim, and it references only real evidence titles — so actions
  are distinct per source and never invent facts or numbers.
* The **business rationale** is the insight's summary — verbatim grounded
  evidence.
* The **expected impact** is a deterministic restatement of the priority bucket
  (high/medium/low), not a fabricated quantitative projection.

This is the literal realisation of "recommendations must never invent
information": every recommendation is an action template plus cited evidence.

---

## 7. Executive briefing generation

`ExecutiveBriefing` assembles the seven required sections:

* **Executive Summary** — a deterministic framing sentence (counts + confidence
  label) followed by the leading key findings, which are verbatim evidence.
* **Key Findings** — the insight summaries (verbatim evidence).
* **Business Risks** — the subset of insight summaries whose tokens intersect a
  fixed risk-signal vocabulary (`failure`, `risk`, `degradation`, ...), surfaced
  verbatim.
* **Recommended Actions** — exposed via the `recommended_actions` property over
  the recommendations.
* **Supporting Evidence** — the aggregated `ExecutiveCitation` list (the
  `supporting_evidence` property).
* **Confidence Assessment** — a deterministic label (`high`/`moderate`/`low`/
  `very low`) via the `confidence_assessment` property.
* **Limitations** — deduplicated copilot-level caveats plus rolled-up agent
  limitations.

All seven sections, plus the `ExecutiveQuestion`, insights, recommendations,
citations, and `ExecutiveReasoning`, are frozen and fully JSON round-trippable.

---

## 8. Confidence model

Overall confidence is a fixed convex combination of four signals, each in
`[0, 1]`, clamped to `[0, 1]`:

```
confidence = 0.40 · agent_confidence        (mean KnowledgeAgent confidence)
           + 0.30 · evidence_coverage        (mean KnowledgeAgent coverage)
           + 0.15 · recommendation_consistency
           + 0.15 · insight_consistency
```

`recommendation_consistency` and `insight_consistency` are the mean pairwise
Jaccard overlap of, respectively, the recommendations' and the insights'
supporting-document sets (trivially `1.0` for a single item, `0.0` when empty).
When confidence falls below `CopilotConfig.confidence_threshold`, a limitation is
recorded.

---

## 9. Citation system

Citations are **reused, not reimplemented**. `ExecutiveCitation` mirrors the
Phase 4 `KnowledgeCitation` fields and adds the originating sub-question;
`ExecutiveCitation.from_knowledge_citation` is the only adaptation point.
Aggregation deduplicates by `document_id` (keeping the highest-relevance
occurrence), sorts deterministically by `(-relevance, rank, document_id)`, and
caps at `max_citations`. Duplicate ids trigger `DuplicateCitationError` as an
invariant guard.

---

## 10. Statistics, validation, and error model

`statistics()` returns a cumulative `ExecutiveCopilotStatistics`: briefings
generated, average confidence, average response latency, and total
recommendation / insight / citation counts. `reset_statistics()` clears them.

`ExecutiveCopilotError` is the root, with `EmptyExecutiveQuestionError`,
`InvalidQuestionTypeError`, `MissingEvidenceError`, `DuplicateRecommendationError`,
and `DuplicateCitationError`. Validated conditions: empty / whitespace /
punctuation-only / non-string questions; unknown explicit question types; missing
evidence when `require_evidence=True`; and duplicate-recommendation /
duplicate-citation invariants.

---

## 11. Engineering decisions & trade-offs

* **Orchestration over reimplementation.** The Copilot owns aggregation and
  framing only; retrieval, scoring, citations, and knowledge reasoning remain in
  the frozen layers. This keeps the deterministic guarantees of the lower tiers
  intact and means a change to retrieval automatically propagates upward.
* **Sub-question expansion.** Each executive question is broadened into a small,
  deterministic set of focused sub-questions, improving evidence coverage without
  any generative step. Trade-off: more agent calls per briefing (bounded by
  `max_sub_questions`), in exchange for broader, better-grounded evidence.
* **Actions reference evidence titles.** Using the cited document title as the
  action topic keeps recommendations distinct per source and grounded, at the
  cost of slightly more verbose action phrasing.
* **Extractive everywhere.** Summaries, findings, risks, and rationales are all
  verbatim agent evidence — auditable and hallucination-free by construction.
* **Out-of-domain confidence.** A question with weak corpus overlap can still
  return a moderate confidence when a single document dominates; the coverage and
  single-source limitations make this explicit in every briefing.

---

## 12. Performance analysis

* Per briefing, cost is `S` Knowledge Agent calls (`S ≤ max_sub_questions`) plus
  `O(C)` aggregation/insight/recommendation work over `C ≤ max_citations`
  citations — independent of corpus size beyond the agent's own retrieval cost.
* The 500-document large-KB test set builds, briefs, round-trips, and re-runs
  deterministically in milliseconds with the injected zero clock.
* Briefings are fully JSON-serialisable, so executive outputs can be cached,
  replayed, and audited without recomputation. Determinism is verified by
  byte-equal `to_dict()` comparisons across independently constructed copilots.

---

## 13. Testing

`tests/test_executive_rag_copilot.py` — **405 tests** covering: the registries;
classification of all ten executive question types (plus focus, priority,
confidence, topic extraction, and sub-question expansion); every core dataclass
(frozen + JSON round trips); the full pipeline; insight extraction;
recommendation generation (grounding, uniqueness, per-type templates); citation
aggregation (dedup, ordering, agent reuse); the confidence model and its
components; business risks and summary assembly; limitations; statistics;
validation and the error hierarchy; Knowledge Agent integration; copilot
serialization; a 500-document knowledge base; determinism; multi-question
invariants; backward compatibility with Phases 2–4; the CLI; and edge cases.

```bash
python -m unittest tests.test_executive_rag_copilot      # 405 tests
python src/knowledge/executive_rag_copilot.py --demo     # deterministic demo
```

---

## 14. Usage example

```python
from knowledge.executive_rag_copilot import ExecutiveRAGCopilot, CopilotConfig

copilot = ExecutiveRAGCopilot.from_knowledge_base(knowledge_base)
briefing = copilot.brief("Give me a risk assessment for the compressor")

print(briefing.executive_summary)
for finding in briefing.key_findings:        # verbatim grounded evidence
    print("-", finding)
for rec in briefing.recommendations:
    print(rec.recommended_action, "|", rec.business_rationale, "|", rec.expected_impact)
print(briefing.confidence, briefing.confidence_assessment, briefing.limitations)

copilot.statistics()                          # cumulative operational metrics
```

---

## 15. Future integration with external LLMs (without changing this architecture)

The Copilot is deliberately structured so that an external LLM can be added later
as an **optional presentation layer**, never as the source of truth:

1. **LLM as renderer, not reasoner.** The deterministic briefing — its grounded
   findings, citations, and confidence — remains the canonical artifact. An LLM
   would only re-express the already-grounded `ExecutiveBriefing` into prose; it
   never decides what is true, what to cite, or how confident to be.
2. **Injected through the existing seams.** The Copilot already accepts injected
   collaborators (agent, classifier, config, clock). An optional
   `briefing_renderer` callable could be injected the same way; when absent,
   today's deterministic behaviour is unchanged. No dataclass, registry, or
   public method needs to change.
3. **Grounding enforced post-hoc.** Because every claim in the briefing is a
   verbatim corpus snippet with a citation, an LLM rendering can be validated
   against the briefing's citations and rejected if it introduces unsupported
   statements — preserving the no-hallucination guarantee even with a generative
   front end.
4. **Determinism preserved by default.** The core pipeline stays pure-Python,
   pure-NumPy, and deterministic; the LLM is strictly an opt-in, side-channel
   enhancement layered on top of an unchanged, frozen-compatible core.

As with every phase in Weeks 1–9, this capability path is available **without
modifying any existing module** — the deterministic Knowledge Layer remains the
foundation, and any future LLM sits above it as a thin, swappable renderer.