# Week 8 — Phase 2: Root Cause Analysis Agent

## Architecture Document

**Component:** `src/agent/root_cause_analysis_agent.py`
**Role:** Evidence-driven, rule-based causal attribution layer
**Inputs:** `AssetEvidence` (per-subsystem anomaly indicators); optional `FleetSnapshot` for risk-weighted fleet analysis
**Outputs:** `RootCauseReport`, `FleetRCAReport`, `ExecutiveRCASummary`
**Status:** Implemented and validated — 180 tests passing with zero skips; all Week 5–8 Phase 1 suites unchanged and coexisting (14/14 modules import cleanly).

---

## 1. Purpose

The platform predicts that an asset is degrading (health, risk, RUL) and what to do about it (maintenance action), but it cannot explain *why* the degradation is occurring. The Root Cause Analysis Agent adds that explanatory layer. It attributes degradation to a primary cause and contributing causes across nine categories, scores each cause, quantifies confidence, and recommends targeted investigations — at both the single-asset and fleet level.

A central design decision shapes the whole module: **root-cause attribution requires evidence, not just a health score.** A low health value or high risk score tells you an asset is unwell, but it carries no information about *which subsystem* is responsible. The agent therefore consumes an `AssetEvidence` record — eight normalised per-subsystem anomaly indicators — and drives attribution from that evidence. This is the honest engineering position: a causal claim must rest on subsystem-level signals, so the contract makes those signals explicit rather than pretending to localise a cause from aggregate health alone.

Like the Decision Copilot Agent, the RCA agent is **rule-based and LLM-free**. Every attribution is a deterministic function of the evidence, so each conclusion is auditable (it traces to indicator values and thresholds) and reproducible (identical evidence always yields the identical report).

---

## 2. Architecture

```
   AssetEvidence (8 per-subsystem indicators in [0,1])
        ▼
   RootCauseAnalysisAgent
        ├── score_causes()       per-cause score · contribution % · confidence
        ├── analyze()            primary + contributing + evidence + actions
        │      ├── single        one dominant signal
        │      ├── multi-cause   several elevated signals
        │      ├── conflicting   top two near-equal
        │      └── unknown       all signals below the evidence floor
        ├── analyze_fleet()      distribution · most-common · highest-risk ·
        │                        concentration  (FleetSnapshot for risk weight)
        └── executive_summary()  top-5 causes · actions · distribution
        ▼
   RootCauseReport / FleetRCAReport / ExecutiveRCASummary
        (frozen · JSON-serialisable · deterministic)
```

The nine cause categories are Temperature, Vibration, Pressure, Load, Lubrication, Electrical, Environmental, Operational, and Unknown. The first eight carry an evidence indicator; Unknown is the residual attribution when no signal clears the detection floor.

---

## 3. Cause Scoring Engine

For each of the eight evidence categories the agent computes three quantities:

```
cause_score              = indicator value in [0, 1]
contribution_percentage  = 100 · cause_score / Σ cause_scores   (0 if no evidence)
confidence (per-cause)   = the indicator value, clamped to [0, 1]
```

`cause_score` is the raw anomaly strength; `contribution_percentage` expresses each cause's share of the total observed evidence (the contributions sum to 100% whenever any evidence is present); and the per-cause confidence reflects how strong that specific signal is. Scores are returned sorted by magnitude, with ties broken by the fixed category enumeration order so the ordering is deterministic.

---

## 4. Attribution Logic and Multi-Cause Analysis

`analyze` examines the sorted scores and resolves one of four regimes:

**Unknown.** If the strongest indicator is below `evidence_floor` (default 0.15), no cause can be localised: the primary cause is `unknown`, with a low confidence that is highest when there is genuinely no signal and falls toward zero as the evidence approaches the floor (the most ambiguous region). Unknown confidence is capped at 0.5 so it can never rival a strongly localised cause.

**Single cause.** One indicator clears the floor and dominates the rest. The primary cause is assigned, no contributing causes qualify, and confidence is high.

**Multi-cause.** Several indicators are elevated. A non-primary cause becomes a *contributing cause* when its score is at least `contributing_fraction` (default 0.50) of the primary score and also clears the floor. Confidence is moderate, reflecting the shared attribution.

**Conflicting evidence.** When the second-strongest score is within `conflict_margin` (default 0.10) of the primary, the report is flagged `is_conflicting`, an explicit note is added to the evidence, and confidence is reduced.

Confidence is a blend of evidence strength and separation:

```
separation = (primary_score − second_score) / primary_score
confidence = (evidence_weight · primary_score
              + separation_weight · separation) / weight_sum
```

with weights 0.60 / 0.40 by default. The separation term is what naturally lowers confidence for conflicting evidence — no special-casing is required, which keeps the scoring honest and continuous.

---

## 5. Investigation Recommendations

Each cause category maps to a concrete investigation, ordered with the primary cause first and contributing causes following (de-duplicated):

| Cause | Recommended investigation |
|-------|---------------------------|
| Vibration | Inspect bearings and check rotating-assembly alignment and balance. |
| Lubrication | Inspect the lubrication system and assess oil quality and level. |
| Load | Review the load profile against the rated operating envelope. |
| Electrical | Inspect the electrical subsystem, connections, and insulation. |
| Operational | Review operating procedures and control set-points. |
| Temperature | Inspect the cooling system and verify thermal sensors. |
| Pressure | Inspect the hydraulic and pressure subsystem for leaks or blockage. |
| Environmental | Review environmental exposure and protective measures. |
| Unknown | Conduct a broad diagnostic inspection; evidence is insufficient to localise. |

These directly cover the platform's required examples (inspect bearings, inspect lubrication system, review load profile, inspect electrical subsystem, review operating procedures).

---

## 6. Fleet RCA

`analyze_fleet` runs the single-asset analysis across a collection of evidence records and aggregates:

- **Top causes across the fleet** — a frequency table of primary causes, most prevalent first.
- **Most common cause** — the modal primary cause (ties broken by category order).
- **Highest-risk cause** — the cause carrying the greatest *aggregate risk*. When a `FleetSnapshot` is supplied, risk is the sum of the corresponding assets' risk scores; otherwise the agent falls back to per-report confidence as a risk proxy. This distinction matters: the most *common* cause and the most *dangerous* cause are frequently different, and leadership needs both.
- **Cause concentration** — a Herfindahl index over the primary-cause distribution, ranging from `1/k` (causes spread evenly) to `1` (a single cause dominates the fleet).

---

## 7. Executive RCA Summary

`executive_summary` condenses a fleet report into the top-five root causes, a de-duplicated set of recommended actions for those causes, the full fleet cause distribution as `(cause, percentage)` pairs, and a natural-language summary naming the most common cause, the highest-risk cause, and whether the fleet's causes are focused or spread (keyed to the concentration index).

---

## 8. Engineering Standards

The agent meets the platform standard in full: seven frozen, validated dataclasses each with a `to_dict`; the `register` / `build` / `list` registry triad under `RCA_AGENT_REGISTRY`; failure-safe tracker integration; pure Python and NumPy with no LLM, PyTorch, SciPy, or pandas; JSON-serialisable outputs with non-finite floats rendered as `null`; and full determinism with stable tie-breaking by the fixed category order. Input validation is strict — `AssetEvidence` rejects out-of-range indicators and empty ids, `RCAConfig` rejects invalid thresholds, `analyze` rejects non-evidence objects, and `analyze_fleet` rejects an empty fleet — while a CLI `--demo` exercises single-asset, unknown-cause, fleet, and executive flows end-to-end.

---

## 9. Validation Summary

180 tests pass with zero skips, covering the cause enum and investigation map, config and evidence validation, the registry, cause scoring (scores, contributions summing to 100%, per-cause confidence, deterministic ordering), all four attribution regimes (single, multi, conflicting, unknown, including floor-boundary behaviour), every category's investigation mapping, fleet RCA (distribution, most-common and risk-weighted highest-risk causes, Herfindahl concentration with exact-value checks), the executive summary, serialization, determinism, failure-safe tracker behaviour, and edge cases (all-equal evidence, all-unknown fleets, large fleets, custom thresholds). The five Week 7 / Week 8 Phase 1 suites continue to pass unchanged, and all fourteen platform modules import and coexist cleanly.