# Presentation Notes

Speaker notes and talk tracks for both decks (`executive_presentation.md`,
`technical_presentation.md`). Use these as full-sentence cues during rehearsal; on stage, rely on
the one-line slide cues. Timings are guides, not constraints.

---

## General Delivery Guidance

- **Open on the problem, not the product.** The audience must feel the systemic pain before the
  solution lands.
- **Speak to outcomes.** For every technical capability, state the business or engineering outcome.
- **Be deliberately honest.** Proactively distinguish verified properties from configured targets,
  and self-assessment from external certification. With senior audiences this builds, not
  undermines, credibility.
- **Let the demo carry proof.** Determinism and the passing test suite are stronger than any slide.
- **Pace.** Executive: ~45 s/slide. Technical: ~60–75 s/slide plus the live demo block.

## Executive Deck — Talk Tracks

**Slides 2–3 (Problem / Why it matters).** "Picture an operations team running a digital twin, a
prediction model per machine, a document search system and an agent that drafts work orders. Built
separately, those four things disagree about what an asset even is. When something goes wrong, no
one can quickly say which model produced which result from which data. That is where cost and risk
come from."

**Slides 7–9 (Business value).** Tie each to a decision a leader makes: trust ("can we defend this
result?"), detection ("how fast do we know something broke?"), change ("how risky is shipping?").

**Slide 12 (ROI).** "I won't show invented savings figures — the inputs are specific to your
operation. Instead, here are the levers the platform moves: audit effort, incident impact,
deployment risk and platform lock-in. Plug in your numbers and the model follows."

**Slides 13–14 (Rigour / Future).** Land credibility (tests, transparent readiness, minimal
dependencies), then point forward to decision intelligence as the natural next step.

## Technical Deck — Talk Tracks

**Slide 4 (Composition).** "The single most important decision: subsystems exchange immutable value
objects and nothing else. No shared mutable state, no reaching into internals. That is why a
ten-layer system stays testable — every layer is exercisable in isolation."

**Slide 6 (Determinism).** "Because we inject the clock and the identifier source, a test can run a
report twice and assert the two serialisations are byte-identical. Determinism turns reproducibility
into an assertion rather than a hope."

**Slides 11–12 (MLOps / Monitoring).** Emphasise provenance-by-construction and the data-vs-concept
drift distinction; these are the two ideas technical audiences probe most.

**Slide 17 (SLOs).** "These are configured objectives the SLO engine evaluates at runtime — they are
not measured latencies from a benchmark. I'm flagging that explicitly because conflating the two is
a common and misleading move."

**Slide 19 (Lessons).** Speak from experience: composition kept the system sane; determinism made
testing cheap; honest gate failures were more useful than green-by-suppression; additive discipline
meant Week 11 never broke Week 1.

## Handling Q&A

- **"Is it production-deployed?"** "It is a fully engineered reference platform verified by an
  extensive deterministic suite and built to production standards. I report verified engineering
  properties, not field telemetry."
- **"Where are the benchmarks?"** "Deliberately absent. I describe methodology and configured
  targets; sustained-load benchmarking is named future work. Inventing numbers would undermine the
  whole point."
- **"Why not use an existing MLOps/observability stack?"** "The contracts allow exactly that — heavy
  engines can sit behind them. The goal here was a coherent, dependency-light reference that
  demonstrates the integration discipline."
- **"How big is it?"** "Roughly 10,620 measured lines across 30 production-engineering modules, with
  1,503 tests; capability-layer internals are pluggable and described architecturally."
- **"What would you do next?"** "Sustained-load study, formalising the cross-capability contracts,
  agent observability, and decision-quality evaluation."

## Failure Recovery (mid-talk)

If a live command fails, narrate the expected output, switch to the recording or screenshot, and
continue. Never debug on stage. The narrative must survive a total AV failure — rehearse delivering
the close from slides alone.