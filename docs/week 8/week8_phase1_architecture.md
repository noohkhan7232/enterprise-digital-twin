# Week 8 — Phase 1: Decision Copilot Agent

## Architecture & Hardening Notes

**Component:** `src/agent/decision_copilot_agent.py`
**Role:** Rule-based, LLM-free explanation and reasoning layer over all prior engines
**Inputs:** `FleetAsset`, `FleetSnapshot` (Phase 2), `ExecutiveDecisionPortfolio` (Phase 3) — composed, never modified
**Outputs:** `AssetExplanation`, `FleetExplanation`, `PortfolioExplanation`, `QuestionAnswer`, `ExecutiveBrief`
**Status:** Implemented and validated — 163 tests passing with zero skips; all Week 5–7 suites unchanged and coexisting (13/13 modules import cleanly).

---

## 1. Purpose

Every prior engine produces a decision or a diagnosis; none explains itself in language a non-specialist can act on. The Decision Copilot Agent is the platform's explanation and reasoning layer. It renders the structured outputs of the predictive, fleet, and executive engines into executive-friendly prose, answers natural-language questions about those outputs with rule-based reasoning, and assembles an executive brief.

The defining constraint is **no LLM dependency**. All language is produced by deterministic templates over the structured fields of the engine outputs; all reasoning is rule-based keyword classification and lookup. This is a deliberate architectural choice with three consequences that matter for an enterprise decision system: every sentence is traceable to a specific field and rule (auditable), identical inputs always yield identical text (reproducible), and the agent runs with no model weights or inference runtime (deployable anywhere the rest of the platform runs).

---

## 2. Architecture

```
   FleetAsset ──────────────────┐
   FleetSnapshot ───────────────┤
   ExecutiveDecisionPortfolio ──┤
                                ▼
                  ┌──────────────────────────────┐
                  │     DecisionCopilotAgent       │
                  │                                │
                  │  explain_asset()               │
                  │  explain_fleet()               │
                  │  explain_portfolio()           │
                  │  answer_question() ── classify │
                  │      │            └─ resolve    │
                  │      ├─ why / why-not           │
                  │      ├─ what-if                 │
                  │      └─ which asset/risk/action │
                  │  generate_executive_brief()     │
                  └──────────────┬─────────────────┘
                                 ▼
   AssetExplanation · FleetExplanation · PortfolioExplanation
   QuestionAnswer · ExecutiveBrief   (frozen · JSON-serialisable)
```

The agent composes the frozen Phase-2/Phase-3 contracts through a small `CopilotContext` (an optional snapshot plus an optional portfolio) and modifies no prior module.

---

## 3. Capabilities

**explain_asset(asset).** Produces six grounded statements — health, trend, RUL, failure probability, risk level, and recommended action — and a composed summary. Because a frozen `FleetAsset` records condition rather than a slope, the trend statement is *inferred* qualitatively from the health band and risk score (critical or high-risk → "deteriorating"; warning or moderate → "trending downward"; otherwise "stable"). This inference is documented and deterministic.

**explain_fleet(snapshot).** Summarises fleet health, names the critical assets, characterises risk concentration qualitatively (well distributed / moderately concentrated / highly concentrated, keyed to the Herfindahl index), and identifies the top maintenance opportunities by expected savings.

**explain_portfolio(portfolio).** Explains the selected assets, the cost-versus-savings economics, the risk reduction (before → after with a percentage), the ROI, and budget coverage.

**answer_question(question, context).** Classifies the question into one of six intents via ordered keyword rules and routes to a dedicated handler. Intent ordering is significant: *why-not* is tested before *why* (so "why wasn't X selected" is not misread as "why"), and *what-if* before the *which-* family (so "what if the asset fails" is not misread as "which asset"). Named assets are resolved by scanning the question for any known asset id (case-insensitive), choosing the earliest occurrence for determinism.

**generate_executive_brief(context).** Assembles an executive summary, ranked key risks, top opportunities, recommendations (propagated from the portfolio where available), and a confidence statement that qualifies the portfolio confidence score as high / moderate / limited.

---

## 4. Rule-Based Reasoning Design

The question-answering is a transparent decision tree, not a learned model:

1. **Classification** — a lower-cased keyword scan maps the question to a `QuestionIntent`. Negator phrases ("why not", "why wasn't", "why didn't", …) capture the why-not branch before the plain "why" branch.
2. **Resolution** — the handler attempts to bind the question to a specific asset from the context.
3. **Reasoning** — each handler applies a fixed rule over the resolved asset and context (for example, "why-not" reports either that the asset was in fact selected, or that its priority was lower / the budget was exhausted, citing its risk score and cost).
4. **Confidence** — a deterministic confidence is assigned by how completely the context supports the answer: ~0.85–0.90 when a named asset is resolved with the needed context, ~0.70 for qualitative what-if reasoning, ~0.30–0.40 for fallbacks.

Every answer carries `supporting_facts` — the raw field values the prose rests on — so a reviewer can verify the explanation against the source data.

---

## 5. Hardening Notes

**Validation versus failure-safety — a deliberate split.** The three `explain_*` methods *validate* their inputs and raise (`TypeError` for a malformed object missing required fields, `ValueError` for an empty snapshot), because explaining a malformed decision object is a programming error that should surface loudly. By contrast, `answer_question` is *failure-safe*: it never raises on a missing or empty context, an empty or whitespace question, or a `None` context — it returns a low-confidence fallback instead, because a question-answering surface that crashes on an unexpected query is unacceptable in an interactive setting. `generate_executive_brief` sits between the two: it requires at least one of a snapshot or portfolio and raises `ValueError` otherwise, since a brief about nothing is meaningless.

**Non-finite handling.** Infinite or NaN RUL (a healthy asset never crossing the failure threshold) is rendered as "no end-of-life is projected within the modelling horizon" rather than printing `inf`. All serialised floats pass through `_jsonsafe`, rendering non-finite values as `null`.

**Unknown enumerations degrade gracefully.** Unrecognised maintenance actions or health bands fall back to neutral phrasing ("a maintenance review is advised", "of indeterminate condition") rather than raising, so a future engine that adds an action value cannot break the copilot.

**Determinism under multiple matches.** When a question names more than one asset, resolution picks the earliest occurrence in the question text, with the asset id as a tie-breaker — so multi-asset questions resolve identically across runs.

**Failure-safe tracker.** The interaction counter is logged inside a `try/except (# noqa: BLE001)`, so a tracker fault can never interrupt an explanation.

**A corrected validation defect.** During bring-up, a self-check asserted that the ROI statement would contain the token "ROI"; the engine in fact spells out "return on investment" for executive readability, so the check — not the engine — was wrong and was corrected. This is the desired failure mode: the implementation favoured the clearer phrasing and the test was brought into line.

---

## 6. Engineering Standards

The agent meets the platform standard in full: seven frozen, validated dataclasses each with a `to_dict`; the `register` / `build` / `list` registry triad under `COPILOT_REGISTRY`; failure-safe tracker integration; pure Python and NumPy with no LLM, PyTorch, SciPy, or pandas anywhere in the module; JSON-serialisable outputs with non-finite floats rendered as `null`; full determinism with stable tie-breaking; and a CLI `--demo` that exercises all five capabilities end-to-end on a synthetic fleet.

---

## 7. Validation Summary

163 tests pass with zero skips, covering the intent enum, config validation and risk labelling, the context object, the registry, all three `explain_*` methods (including infinite RUL, every action and band, and currency propagation), all six question intents plus classification precedence and asset resolution, the executive brief (snapshot-only, portfolio-only, and combined), serialization, determinism, failure-safe behaviour, and edge cases (empty and whitespace questions, `None` context, malformed objects, empty/single/large fleets, zero and unconstrained budgets). The four Week-7 suites (Monte Carlo 202, Fleet 157, Executive 182, Strategic 210) continue to pass unchanged, and all thirteen platform modules import and coexist cleanly.

---

## 8. Week 8 Trajectory

This agent is the deterministic substrate for the remaining Week-8 work. Its structured `QuestionAnswer` and `*Explanation` outputs, each carrying supporting facts and a confidence score, are exactly the grounded, auditable units a retrieval-augmented or agentic layer needs: an LLM-backed interface can later sit *in front of* this agent for fluent phrasing and open-ended dialogue while delegating every factual claim to these rule-based, traceable explanations — keeping the authoritative reasoning verifiable even as the conversational surface becomes more flexible.