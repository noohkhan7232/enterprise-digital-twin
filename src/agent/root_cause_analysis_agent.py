#!/usr/bin/env python3
"""Root Cause Analysis Agent — evidence-driven causal attribution.

Week-8 Phase-2 adds the platform's root-cause layer.  The prior engines predict
*that* an asset is degrading (health, risk, RUL) and *what* to do about it
(maintenance action); none explains *why* the degradation is occurring.  The
Root Cause Analysis Agent closes that gap.

Genuine root-cause attribution requires evidence beyond a health score: a low
health value tells you an asset is unwell but cannot localise the cause.  The
agent therefore consumes per-subsystem anomaly indicators — an
:class:`AssetEvidence` record carrying eight normalised cause signals — and
attributes degradation to a primary cause and contributing causes with a
confidence score, supporting evidence, and recommended investigations.  It
composes the platform by consuming :class:`FleetAsset` / :class:`FleetSnapshot`
risk for fleet-level weighting, but drives attribution from evidence.

Like the Decision Copilot Agent, this agent is **rule-based and LLM-free**: every
attribution is a deterministic function of the evidence, fully auditable and
reproducible.

============================================================================
Cause categories
============================================================================
Temperature · Vibration · Pressure · Load · Lubrication · Electrical ·
Environmental · Operational · Unknown.

============================================================================
Capabilities
============================================================================
* ``analyze(evidence)``            — single-asset root-cause report.
* ``score_causes(evidence)``       — per-cause scores, contributions, confidence.
* ``analyze_fleet(items)``         — fleet-wide top causes, concentration.
* ``executive_summary(report)``    — top-5 causes, actions, distribution.

============================================================================
Architecture
============================================================================
::

    AssetEvidence (per-subsystem indicators)
        ▼
    RootCauseAnalysisAgent
        ├── score_causes()      cause_score · contribution_% · confidence
        ├── analyze()           primary + contributing + evidence + actions
        ├── analyze_fleet()     distribution · most-common · highest-risk
        └── executive_summary() top-5 · actions · distribution
        ▼
    RootCauseReport / FleetRCAReport / ExecutiveRCASummary
        (frozen · JSON-serialisable · deterministic)

Usage::

    from src.agent.root_cause_analysis_agent import (
        RootCauseAnalysisAgent, AssetEvidence,
    )
    agent = RootCauseAnalysisAgent()
    report = agent.analyze(AssetEvidence(asset_id="WTG-042", vibration=0.85))
    print(report.primary_cause, report.confidence)

CLI::

    python src/agent/root_cause_analysis_agent.py --demo
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fleet.fleet_digital_twin import FleetSnapshot  # noqa: E402

logger = logging.getLogger("root_cause_analysis_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Registry of named root cause analysis agents.
RCA_AGENT_REGISTRY: dict[str, type] = {}

AGENT_NAME: Final[str] = "root_cause_analysis_agent"

_EPS: Final[float] = 1e-12


def _jsonsafe(x: float) -> float | None:
    """Render a non-finite float as ``None`` for JSON.

    Args:
        x: A float that may be ``inf`` or ``NaN``.

    Returns:
        ``None`` when non-finite, else the float.
    """
    return None if (math.isinf(x) or math.isnan(x)) else float(x)


# ---------------------------------------------------------------------------
# Cause-category enum
# ---------------------------------------------------------------------------


class CauseCategory(str, Enum):
    """Supported root-cause categories (definition order is the tie-break order)."""

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    LOAD = "load"
    LUBRICATION = "lubrication"
    ELECTRICAL = "electrical"
    ENVIRONMENTAL = "environmental"
    OPERATIONAL = "operational"
    UNKNOWN = "unknown"


#: The cause categories that carry an evidence signal (UNKNOWN excluded).
_EVIDENCE_CATEGORIES: Final[tuple[CauseCategory, ...]] = (
    CauseCategory.TEMPERATURE, CauseCategory.VIBRATION, CauseCategory.PRESSURE,
    CauseCategory.LOAD, CauseCategory.LUBRICATION, CauseCategory.ELECTRICAL,
    CauseCategory.ENVIRONMENTAL, CauseCategory.OPERATIONAL,
)

#: Tie-break index for each category (enum definition order).
_CATEGORY_INDEX: Final[dict[CauseCategory, int]] = {
    c: i for i, c in enumerate(CauseCategory)
}

#: Recommended investigation per cause category.
_INVESTIGATION: Final[dict[CauseCategory, str]] = {
    CauseCategory.TEMPERATURE: "Inspect the cooling system and verify thermal sensors.",
    CauseCategory.VIBRATION: "Inspect bearings and check rotating-assembly alignment and balance.",
    CauseCategory.PRESSURE: "Inspect the hydraulic and pressure subsystem for leaks or blockage.",
    CauseCategory.LOAD: "Review the load profile against the rated operating envelope.",
    CauseCategory.LUBRICATION: "Inspect the lubrication system and assess oil quality and level.",
    CauseCategory.ELECTRICAL: "Inspect the electrical subsystem, connections, and insulation.",
    CauseCategory.ENVIRONMENTAL: "Review environmental exposure and the condition of protective measures.",
    CauseCategory.OPERATIONAL: "Review operating procedures and control set-points.",
    CauseCategory.UNKNOWN: "Conduct a broad diagnostic inspection; evidence is insufficient to localise the cause.",
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_root_cause_analysis_agent(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a root cause analysis agent by name.

    Args:
        name: Unique registry key.

    Returns:
        The class decorator.

    Raises:
        ValueError: When *name* is already registered to a different class.
    """
    def decorator(cls: type) -> type:
        existing = RCA_AGENT_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Root cause analysis agent '{name}' already registered to "
                f"{existing.__name__}"
            )
        RCA_AGENT_REGISTRY[name] = cls
        cls._registry_name = name
        logger.debug("Registered root cause analysis agent '%s' -> %s",
                     name, cls.__name__)
        return cls

    return decorator


def build_root_cause_analysis_agent(name: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Instantiate a registered root cause analysis agent by name.

    Args:
        name: Registry key.
        **kwargs: Forwarded to the constructor.

    Returns:
        An instantiated agent.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in RCA_AGENT_REGISTRY:
        available = ", ".join(sorted(RCA_AGENT_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown root cause analysis agent '{name}'. Available: {available}"
        )
    return RCA_AGENT_REGISTRY[name](**kwargs)


def list_root_cause_analysis_agents() -> list[str]:
    """Return the sorted names of registered root cause analysis agents.

    Returns:
        Sorted registry keys.
    """
    return sorted(RCA_AGENT_REGISTRY)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RCAConfig:
    """Configuration for the :class:`RootCauseAnalysisAgent`.

    Attributes:
        evidence_floor: Minimum indicator value to count as evidence; below this
            for all categories the cause is ``UNKNOWN``. In ``(0, 1)``.
        contributing_fraction: A non-primary cause is "contributing" when its
            score is at least this fraction of the primary score. In ``(0, 1]``.
        conflict_margin: When the second-strongest score is within this fraction
            of the primary, the evidence is flagged conflicting. In ``[0, 1)``.
        evidence_weight: Weight on raw evidence strength in the confidence blend.
        separation_weight: Weight on primary/second separation in the confidence
            blend. ``evidence_weight + separation_weight`` should be ``1``.
        top_n: Number of items in fleet top-cause and executive lists.
    """

    evidence_floor:        float = 0.15
    contributing_fraction: float = 0.50
    conflict_margin:       float = 0.10
    evidence_weight:       float = 0.60
    separation_weight:     float = 0.40
    top_n:                 int = 5

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On invalid thresholds or weights.
        """
        if not (0.0 < self.evidence_floor < 1.0):
            raise ValueError("evidence_floor must be in (0, 1)")
        if not (0.0 < self.contributing_fraction <= 1.0):
            raise ValueError("contributing_fraction must be in (0, 1]")
        if not (0.0 <= self.conflict_margin < 1.0):
            raise ValueError("conflict_margin must be in [0, 1)")
        if self.evidence_weight < 0 or self.separation_weight < 0:
            raise ValueError("confidence weights must be >= 0")
        if self.evidence_weight + self.separation_weight <= 0:
            raise ValueError("confidence weights must sum to > 0")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")

    @property
    def weight_sum(self) -> float:
        """Return the sum of the two confidence weights."""
        return self.evidence_weight + self.separation_weight


# ---------------------------------------------------------------------------
# Evidence input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetEvidence:
    """Per-subsystem anomaly indicators for one asset.

    Each indicator is a normalised anomaly strength in ``[0, 1]``, where higher
    means stronger evidence that the corresponding subsystem is driving
    degradation.

    Attributes:
        asset_id: The asset identifier.
        temperature: Thermal-anomaly indicator.
        vibration: Vibration-anomaly indicator.
        pressure: Pressure-anomaly indicator.
        load: Load / overload indicator.
        lubrication: Lubrication-degradation indicator.
        electrical: Electrical-anomaly indicator.
        environmental: Environmental-stress indicator.
        operational: Operational-deviation indicator.
    """

    asset_id:      str
    temperature:   float = 0.0
    vibration:     float = 0.0
    pressure:      float = 0.0
    load:          float = 0.0
    lubrication:   float = 0.0
    electrical:    float = 0.0
    environmental: float = 0.0
    operational:   float = 0.0

    def __post_init__(self) -> None:
        """Validate the evidence indicators.

        Raises:
            ValueError: When ``asset_id`` is empty or an indicator is out of range.
        """
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        for cat in _EVIDENCE_CATEGORIES:
            v = getattr(self, cat.value)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"indicator '{cat.value}' must be in [0, 1], got {v}")

    def indicators(self) -> dict[CauseCategory, float]:
        """Return the indicator values keyed by cause category.

        Returns:
            Mapping of evidence category to indicator value.
        """
        return {cat: float(getattr(self, cat.value))
                for cat in _EVIDENCE_CATEGORIES}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        d: dict[str, Any] = {"asset_id": self.asset_id}
        for cat in _EVIDENCE_CATEGORIES:
            d[cat.value] = float(getattr(self, cat.value))
        return d


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CauseScore:
    """The score of a single cause category for one asset.

    Attributes:
        cause: The cause category.
        cause_score: The raw evidence-derived score in ``[0, 1]``.
        contribution_percentage: Share of total evidence attributed to this cause.
        confidence: Per-cause evidence confidence in ``[0, 1]``.
    """

    cause:                    str
    cause_score:              float
    contribution_percentage:  float
    confidence:               float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "cause": self.cause,
            "cause_score": _jsonsafe(self.cause_score),
            "contribution_percentage": _jsonsafe(self.contribution_percentage),
            "confidence": _jsonsafe(self.confidence),
        }


@dataclass(frozen=True)
class RootCauseReport:
    """A root-cause report for one asset.

    Attributes:
        asset_id: The asset identifier.
        primary_cause: The most likely cause category.
        contributing_causes: Other causes above the contributing threshold.
        confidence: Confidence in the primary attribution in ``[0, 1]``.
        evidence: Human-readable supporting-evidence statements.
        investigation_actions: Recommended investigations, primary cause first.
        cause_scores: The full per-cause scoring.
        is_conflicting: Whether the top two causes are near-equal (ambiguous).
        is_unknown: Whether evidence was insufficient to localise a cause.
    """

    asset_id:               str
    primary_cause:          str
    contributing_causes:    tuple[str, ...]
    confidence:             float
    evidence:               tuple[str, ...]
    investigation_actions:  tuple[str, ...]
    cause_scores:           tuple[CauseScore, ...]
    is_conflicting:         bool
    is_unknown:             bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "asset_id": self.asset_id,
            "primary_cause": self.primary_cause,
            "contributing_causes": list(self.contributing_causes),
            "confidence": _jsonsafe(self.confidence),
            "evidence": list(self.evidence),
            "investigation_actions": list(self.investigation_actions),
            "cause_scores": [c.to_dict() for c in self.cause_scores],
            "is_conflicting": self.is_conflicting,
            "is_unknown": self.is_unknown,
        }


@dataclass(frozen=True)
class CauseFrequency:
    """A cause's prevalence across a fleet.

    Attributes:
        cause: The cause category.
        count: Number of assets whose primary cause is this category.
        percentage: Share of assets attributed to this cause.
        total_risk: Sum of risk scores of assets with this primary cause.
    """

    cause:       str
    count:       int
    percentage:  float
    total_risk:  float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "cause": self.cause,
            "count": self.count,
            "percentage": _jsonsafe(self.percentage),
            "total_risk": _jsonsafe(self.total_risk),
        }


@dataclass(frozen=True)
class FleetRCAReport:
    """A fleet-wide root-cause analysis.

    Attributes:
        asset_count: Number of assets analysed.
        top_causes: Cause frequencies, most prevalent first.
        most_common_cause: The most frequently attributed primary cause.
        highest_risk_cause: The cause carrying the greatest aggregate risk.
        cause_concentration: Herfindahl index over the primary-cause distribution.
        reports: The per-asset reports.
    """

    asset_count:          int
    top_causes:           tuple[CauseFrequency, ...]
    most_common_cause:    str
    highest_risk_cause:   str
    cause_concentration:  float
    reports:              tuple[RootCauseReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "asset_count": self.asset_count,
            "top_causes": [c.to_dict() for c in self.top_causes],
            "most_common_cause": self.most_common_cause,
            "highest_risk_cause": self.highest_risk_cause,
            "cause_concentration": _jsonsafe(self.cause_concentration),
            "reports": [r.to_dict() for r in self.reports],
        }


@dataclass(frozen=True)
class ExecutiveRCASummary:
    """An executive-level fleet root-cause summary.

    Attributes:
        top_causes: The top-N cause frequencies.
        recommended_actions: De-duplicated investigations for the top causes.
        cause_distribution: ``(cause, percentage)`` pairs across the fleet.
        summary: A composed natural-language summary.
    """

    top_causes:           tuple[CauseFrequency, ...]
    recommended_actions:  tuple[str, ...]
    cause_distribution:   tuple[tuple[str, float], ...]
    summary:              str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "top_causes": [c.to_dict() for c in self.top_causes],
            "recommended_actions": list(self.recommended_actions),
            "cause_distribution": [[c, _jsonsafe(p)]
                                   for c, p in self.cause_distribution],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Root Cause Analysis Agent
# ---------------------------------------------------------------------------


@register_root_cause_analysis_agent(AGENT_NAME)
class RootCauseAnalysisAgent:
    """Rule-based, evidence-driven root-cause attribution over the platform.

    The agent scores the eight cause categories from an asset's anomaly
    indicators, attributes a primary and contributing causes with a confidence
    score, recommends investigations, and aggregates results across a fleet.  It
    holds no per-call mutable state and is fully deterministic.

    Args:
        config: The RCA configuration.
        experiment_tracker: Optional tracker for logging analysis counts.
    """

    _registry_name: str | None = None

    def __init__(
        self,
        config: RCAConfig | None = None,
        experiment_tracker: Any = None,
    ) -> None:
        self.config = config or RCAConfig()
        self.tracker = experiment_tracker
        self._n_analyses = 0
        logger.info("RootCauseAnalysisAgent ready | floor=%.2f",
                    self.config.evidence_floor)

    # ------------------------------------------------------------------
    # Cause scoring
    # ------------------------------------------------------------------

    def score_causes(self, evidence: AssetEvidence) -> tuple[CauseScore, ...]:
        """Score every cause category from an asset's evidence.

        Args:
            evidence: The asset's per-subsystem indicators.

        Returns:
            A tuple of :class:`CauseScore`, ordered by score (descending) then
            by category order.

        Raises:
            TypeError: When *evidence* lacks an ``indicators`` method.
        """
        if not hasattr(evidence, "indicators"):
            raise TypeError("evidence must be an AssetEvidence-like object")
        ind = evidence.indicators()
        total = sum(ind.values())
        scores = []
        for cat in _EVIDENCE_CATEGORIES:
            v = ind[cat]
            contribution = (100.0 * v / total) if total > _EPS else 0.0
            scores.append(CauseScore(
                cause=cat.value,
                cause_score=float(v),
                contribution_percentage=float(contribution),
                confidence=float(np.clip(v, 0.0, 1.0)),
            ))
        scores.sort(key=lambda s: (-s.cause_score,
                                   _CATEGORY_INDEX[CauseCategory(s.cause)]))
        return tuple(scores)

    # ------------------------------------------------------------------
    # Single-asset analysis
    # ------------------------------------------------------------------

    def analyze(self, evidence: AssetEvidence) -> RootCauseReport:
        """Produce a root-cause report for one asset.

        Args:
            evidence: The asset's per-subsystem indicators.

        Returns:
            A :class:`RootCauseReport`.

        Raises:
            TypeError: When *evidence* is not an :class:`AssetEvidence`.
        """
        if not isinstance(evidence, AssetEvidence):
            raise TypeError("analyze requires an AssetEvidence instance")
        cfg = self.config
        scores = self.score_causes(evidence)
        top = scores[0]
        second = scores[1] if len(scores) > 1 else None
        top_val = top.cause_score
        second_val = second.cause_score if second is not None else 0.0

        self._n_analyses += 1
        self._log()

        # Unknown: no category clears the evidence floor.
        if top_val < cfg.evidence_floor:
            unknown_conf = float(np.clip(
                0.5 * (1.0 - min(top_val / cfg.evidence_floor, 1.0)), 0.0, 0.5))
            return RootCauseReport(
                asset_id=evidence.asset_id,
                primary_cause=CauseCategory.UNKNOWN.value,
                contributing_causes=(),
                confidence=unknown_conf,
                evidence=("Evidence across all monitored subsystems is below "
                          "the detection threshold; no cause can be localised.",),
                investigation_actions=(_INVESTIGATION[CauseCategory.UNKNOWN],),
                cause_scores=scores,
                is_conflicting=False,
                is_unknown=True,
            )

        primary_cat = CauseCategory(top.cause)
        # Contributing causes: above the contributing fraction and the floor.
        contributing = tuple(
            s.cause for s in scores[1:]
            if s.cause_score >= cfg.contributing_fraction * top_val
            and s.cause_score >= cfg.evidence_floor)

        conflicting = (second_val >= (1.0 - cfg.conflict_margin) * top_val
                       and second_val >= cfg.evidence_floor)

        separation = ((top_val - second_val) / top_val
                      if top_val > _EPS else 1.0)
        if second is None:
            separation = 1.0
        confidence = float(np.clip(
            (cfg.evidence_weight * top_val
             + cfg.separation_weight * separation) / cfg.weight_sum,
            0.0, 1.0))

        evidence_stmts = self._evidence_statements(scores, primary_cat,
                                                   contributing, conflicting)
        actions = self._investigations(primary_cat, contributing)

        return RootCauseReport(
            asset_id=evidence.asset_id,
            primary_cause=primary_cat.value,
            contributing_causes=contributing,
            confidence=confidence,
            evidence=evidence_stmts,
            investigation_actions=actions,
            cause_scores=scores,
            is_conflicting=bool(conflicting),
            is_unknown=False,
        )

    def _evidence_statements(
        self, scores: Sequence[CauseScore], primary: CauseCategory,
        contributing: Sequence[str], conflicting: bool,
    ) -> tuple[str, ...]:
        """Compose human-readable supporting-evidence statements.

        Args:
            scores: The per-cause scores (sorted).
            primary: The primary cause category.
            contributing: The contributing cause names.
            conflicting: Whether the evidence is conflicting.

        Returns:
            A tuple of evidence statements.
        """
        by_cause = {s.cause: s for s in scores}
        stmts = [
            f"Primary evidence: {primary.value} anomaly at "
            f"{by_cause[primary.value].cause_score:.0%} "
            f"({by_cause[primary.value].contribution_percentage:.0f}% of total "
            "evidence)."
        ]
        for c in contributing:
            s = by_cause[c]
            stmts.append(
                f"Contributing: {c} anomaly at {s.cause_score:.0%} "
                f"({s.contribution_percentage:.0f}% of total evidence).")
        if conflicting:
            stmts.append(
                "Note: the two strongest signals are close in magnitude; the "
                "primary attribution is not clear-cut.")
        return tuple(stmts)

    def _investigations(
        self, primary: CauseCategory, contributing: Sequence[str]
    ) -> tuple[str, ...]:
        """Build the ordered investigation actions.

        Args:
            primary: The primary cause category.
            contributing: The contributing cause names.

        Returns:
            Investigation actions, primary first, de-duplicated.
        """
        actions = [_INVESTIGATION[primary]]
        for c in contributing:
            act = _INVESTIGATION[CauseCategory(c)]
            if act not in actions:
                actions.append(act)
        return tuple(actions)

    # ------------------------------------------------------------------
    # Fleet analysis
    # ------------------------------------------------------------------

    def analyze_fleet(
        self, evidence_items: Sequence[AssetEvidence], *,
        snapshot: FleetSnapshot | None = None,
    ) -> FleetRCAReport:
        """Analyse root causes across a fleet.

        Args:
            evidence_items: One :class:`AssetEvidence` per asset.
            snapshot: Optional fleet snapshot supplying per-asset risk scores for
                the highest-risk-cause computation.

        Returns:
            A :class:`FleetRCAReport`.

        Raises:
            ValueError: When *evidence_items* is empty.
        """
        items = list(evidence_items)
        if not items:
            raise ValueError("analyze_fleet requires at least one evidence item")

        risk_by_asset: dict[str, float] = {}
        if snapshot is not None and snapshot.assets:
            risk_by_asset = {a.asset_id: float(a.risk_score)
                             for a in snapshot.assets}

        reports = tuple(self.analyze(e) for e in items)

        # Tally primary causes and accumulate risk per cause.
        counts: dict[str, int] = {}
        risk_totals: dict[str, float] = {}
        for r in reports:
            counts[r.primary_cause] = counts.get(r.primary_cause, 0) + 1
            risk = risk_by_asset.get(r.asset_id, r.confidence)
            risk_totals[r.primary_cause] = (
                risk_totals.get(r.primary_cause, 0.0) + risk)

        n = len(reports)
        freqs = []
        for cause in sorted(counts, key=lambda c: (-counts[c],
                                                    self._cause_order(c))):
            freqs.append(CauseFrequency(
                cause=cause,
                count=counts[cause],
                percentage=100.0 * counts[cause] / n,
                total_risk=float(risk_totals.get(cause, 0.0)),
            ))
        top_causes = tuple(freqs)

        most_common = top_causes[0].cause if top_causes else CauseCategory.UNKNOWN.value
        highest_risk = (max(risk_totals,
                            key=lambda c: (risk_totals[c], -self._cause_order(c)))
                        if risk_totals else CauseCategory.UNKNOWN.value)

        # Herfindahl concentration over the cause distribution.
        fractions = np.array([f.count / n for f in top_causes], dtype=float)
        concentration = float(np.sum(fractions ** 2)) if fractions.size else 0.0

        return FleetRCAReport(
            asset_count=n,
            top_causes=top_causes,
            most_common_cause=most_common,
            highest_risk_cause=highest_risk,
            cause_concentration=concentration,
            reports=reports,
        )

    @staticmethod
    def _cause_order(cause: str) -> int:
        """Return the tie-break index of a cause name.

        Args:
            cause: The cause category value.

        Returns:
            The category's definition-order index.
        """
        try:
            return _CATEGORY_INDEX[CauseCategory(cause)]
        except ValueError:
            return len(_CATEGORY_INDEX)

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def executive_summary(
        self, fleet_report: FleetRCAReport
    ) -> ExecutiveRCASummary:
        """Summarise a fleet RCA report for executives.

        Args:
            fleet_report: The fleet root-cause report.

        Returns:
            An :class:`ExecutiveRCASummary`.

        Raises:
            TypeError: When *fleet_report* is not a :class:`FleetRCAReport`.
        """
        if not isinstance(fleet_report, FleetRCAReport):
            raise TypeError("executive_summary requires a FleetRCAReport")
        top = fleet_report.top_causes[:self.config.top_n]

        actions: list[str] = []
        for f in top:
            try:
                act = _INVESTIGATION[CauseCategory(f.cause)]
            except ValueError:
                act = _INVESTIGATION[CauseCategory.UNKNOWN]
            if act not in actions:
                actions.append(act)

        distribution = tuple((f.cause, f.percentage)
                             for f in fleet_report.top_causes)

        if top:
            lead = top[0]
            summary = (
                f"Across {fleet_report.asset_count} assets, the most common root "
                f"cause is {lead.cause} ({lead.percentage:.0f}% of assets). The "
                f"cause carrying the greatest aggregate risk is "
                f"{fleet_report.highest_risk_cause}. Cause concentration is "
                f"{fleet_report.cause_concentration:.2f} "
                + ("(focused on a few causes)."
                   if fleet_report.cause_concentration >= 0.4
                   else "(spread across several causes)."))
        else:
            summary = ("No assets were analysed; no root-cause summary is "
                       "available.")

        return ExecutiveRCASummary(
            top_causes=top,
            recommended_actions=tuple(actions),
            cause_distribution=distribution,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Tracker integration
    # ------------------------------------------------------------------

    def _log(self) -> None:
        """Log the analysis count to the tracker (failure-safe)."""
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(
                {"rca_analyses": float(self._n_analyses)},
                step=self._n_analyses)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracker log_metrics failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a short root-cause demo over a synthetic fleet.

    Returns:
        Exit code 0.
    """
    rng = np.random.default_rng(82)
    # Synthesise evidence: each asset has a dominant subsystem plus noise.
    dominant = ["vibration", "lubrication", "temperature", "electrical",
                "load", "vibration", "operational", "vibration"]
    items = []
    for i, dom in enumerate(dominant):
        kwargs = {c.value: float(np.clip(rng.uniform(0.0, 0.2), 0, 1))
                  for c in _EVIDENCE_CATEGORIES}
        kwargs[dom] = float(np.clip(rng.uniform(0.6, 0.95), 0, 1))
        items.append(AssetEvidence(asset_id=f"WTG-{i:03d}", **kwargs))

    agent = RootCauseAnalysisAgent()

    print("=== Single-asset RCA ===")
    r = agent.analyze(items[0])
    print(f"{r.asset_id}: primary={r.primary_cause} "
          f"contributing={list(r.contributing_causes)} "
          f"confidence={r.confidence:.2f}")
    for e in r.evidence:
        print(f"   - {e}")
    print("   Investigations:")
    for a in r.investigation_actions:
        print(f"     * {a}")
    print()

    print("=== Unknown-cause example ===")
    weak = AssetEvidence(asset_id="WTG-X", vibration=0.05, temperature=0.08)
    ur = agent.analyze(weak)
    print(f"{ur.asset_id}: primary={ur.primary_cause} "
          f"confidence={ur.confidence:.2f} is_unknown={ur.is_unknown}")
    print()

    print("=== Fleet RCA ===")
    fleet = agent.analyze_fleet(items)
    print(f"assets={fleet.asset_count} most_common={fleet.most_common_cause} "
          f"highest_risk={fleet.highest_risk_cause} "
          f"concentration={fleet.cause_concentration:.2f}")
    for f in fleet.top_causes:
        print(f"   {f.cause:14s} count={f.count} ({f.percentage:.0f}%)")
    print()

    print("=== Executive RCA summary ===")
    summary = agent.executive_summary(fleet)
    print(summary.summary)
    print("Recommended actions:")
    for a in summary.recommended_actions:
        print(f"   * {a}")
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
    parser = argparse.ArgumentParser(description="Root cause analysis agent")
    parser.add_argument("--demo", action="store_true",
                        help="Run a root-cause demo.")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args(argv)

    if args.list_agents:
        print("Registered root cause analysis agents:",
              list_root_cause_analysis_agents())
        return 0
    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())