"""Deterministic SLI/SLO engine with error-budget and burn-rate computation.

Supports availability, latency, error-rate and freshness SLIs; reliability SLOs
with directional targets; error-budget calculation, burn rate and compliance
reporting. Pure Python, deterministic, thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .observability_models import (
    ErrorBudget, SLI, SLIType, SLO, ValidationError, clamp, percentile,
)

__all__ = ["SLISLOEngine", "create_sli_slo_engine"]


class SLISLOEngine:
    """Computes SLIs, evaluates SLOs and tracks error budgets."""

    def __init__(self) -> None:
        self._slos: Dict[str, SLO] = {}
        self._slis: Dict[str, SLI] = {}
        self._lock = threading.RLock()

    # -- SLI construction --------------------------------------------------- #
    @staticmethod
    def availability_sli(name: str, successful: int, total: int) -> SLI:
        if total < 0 or successful < 0 or successful > total:
            raise ValidationError("invalid availability counts")
        value = 1.0 if total == 0 else successful / total
        return SLI(name, SLIType.AVAILABILITY, round(value, 6), "ratio")

    @staticmethod
    def latency_sli(name: str, latencies: Sequence[float], pct: float = 95.0) -> SLI:
        value = percentile(latencies, pct)
        return SLI(name, SLIType.LATENCY, value, "ms")

    @staticmethod
    def error_rate_sli(name: str, errors: int, total: int) -> SLI:
        if total < 0 or errors < 0 or errors > total:
            raise ValidationError("invalid error counts")
        value = 0.0 if total == 0 else errors / total
        return SLI(name, SLIType.ERROR_RATE, round(value, 6), "ratio")

    @staticmethod
    def freshness_sli(name: str, age_seconds: float) -> SLI:
        return SLI(name, SLIType.FRESHNESS, round(float(age_seconds), 6), "s")

    # -- registration ------------------------------------------------------- #
    def register_slo(self, slo: SLO) -> None:
        with self._lock:
            self._slos[slo.name] = slo

    def record_sli(self, sli: SLI) -> None:
        with self._lock:
            self._slis[sli.name] = sli

    def slo(self, name: str) -> SLO:
        with self._lock:
            if name not in self._slos:
                raise ValidationError(f"unknown SLO: {name}")
            return self._slos[name]

    # -- evaluation --------------------------------------------------------- #
    def is_compliant(self, slo_name: str, sli_value: float) -> bool:
        return self.slo(slo_name).is_met(sli_value)

    def error_budget(self, slo_name: str, sli_value: float,
                     *, elapsed_fraction: float = 1.0) -> ErrorBudget:
        """Compute the error budget for an availability/error-rate style SLO.

        For ``gte`` SLOs (e.g. availability), the budget is ``1 - target`` and
        consumption is how far below target the SLI sits. For ``lte`` SLOs
        (e.g. error rate), the budget is the allowed maximum and consumption is
        the observed value. ``elapsed_fraction`` scales the burn-rate baseline.
        """
        slo = self.slo(slo_name)
        if slo.comparison == "gte":
            budget_total = round(1.0 - slo.target, 6)
            consumed = max(0.0, slo.target - sli_value)
        else:
            budget_total = round(slo.target, 6)
            consumed = max(0.0, sli_value)
        consumed = round(min(consumed, budget_total) if budget_total > 0 else consumed, 6)
        remaining = round(max(0.0, budget_total - consumed), 6)
        # Burn rate: consumed fraction of budget vs elapsed fraction of window.
        elapsed = clamp(elapsed_fraction, 1e-9, 1.0)
        consumed_fraction = 0.0 if budget_total <= 0 else consumed / budget_total
        burn_rate = round(consumed_fraction / elapsed, 6)
        return ErrorBudget(
            slo_name=slo_name, target=slo.target, actual=round(float(sli_value), 6),
            window_seconds=slo.window_seconds, budget_total=budget_total,
            budget_consumed=consumed, budget_remaining=remaining, burn_rate=burn_rate,
        )

    def compliance_report(self, sli_values: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        """Report compliance for every registered SLO.

        SLI values may be supplied explicitly or resolved from recorded SLIs
        that share the SLO's name.
        """
        with self._lock:
            slos = dict(self._slos)
            recorded = {n: s.value for n, s in self._slis.items()}
        provided = dict(recorded)
        if sli_values:
            provided.update(sli_values)
        results: Dict[str, Any] = {}
        compliant_count = 0
        for name, slo in sorted(slos.items()):
            if name not in provided:
                results[name] = {"evaluated": False, "compliant": None}
                continue
            value = provided[name]
            compliant = slo.is_met(value)
            compliant_count += int(compliant)
            budget = self.error_budget(name, value)
            results[name] = {
                "evaluated": True, "compliant": compliant, "sli_value": round(value, 6),
                "target": slo.target, "comparison": slo.comparison,
                "budget_remaining": budget.budget_remaining, "burn_rate": budget.burn_rate,
            }
        evaluated = [r for r in results.values() if r.get("evaluated")]
        return {
            "slo_count": len(slos),
            "evaluated": len(evaluated),
            "compliant": compliant_count,
            "compliance_rate": round(compliant_count / len(evaluated), 6) if evaluated else 1.0,
            "results": results,
        }


def create_sli_slo_engine() -> SLISLOEngine:
    return SLISLOEngine()