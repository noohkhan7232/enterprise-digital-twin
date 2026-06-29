"""Deterministic data-quality monitoring (pure NumPy).

Detects missing values, duplicate records, invalid ranges, outliers, schema
violations, null percentage, data freshness, consistency and completeness, and
aggregates them into :class:`QualityMetrics` with a deterministic list of
:class:`QualityIssue` records.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from monitoring.monitoring_models import (
    Clock,
    DeterministicIdGenerator,
    DriftSeverity,
    IdGenerator,
    LogicalClock,
    MonitoringConfiguration,
    MonitoringError,
    QualityIssue,
    QualityIssueType,
    QualityMetrics,
    ValidationError,
)

__all__ = ["DataQualityMonitor", "create_data_quality_monitor"]


def _severity_from_rate(rate: float) -> DriftSeverity:
    if rate <= 0.0:
        return DriftSeverity.NONE
    if rate < 0.01:
        return DriftSeverity.LOW
    if rate < 0.05:
        return DriftSeverity.MODERATE
    if rate < 0.2:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


def _numeric(values: Sequence[Any]) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr


class DataQualityMonitor:
    """Evaluates the quality of a tabular dataset deterministically."""

    def __init__(
        self,
        config: Optional[MonitoringConfiguration] = None,
        clock: Optional[Clock] = None,
        id_generator: Optional[IdGenerator] = None,
        freshness_budget_seconds: float = 3600.0,
    ) -> None:
        self._config = config or MonitoringConfiguration()
        self._clock: Clock = clock or LogicalClock()
        self._ids: IdGenerator = id_generator or DeterministicIdGenerator(seed="quality")
        self._freshness_budget = float(freshness_budget_seconds)
        self._lock = threading.RLock()

    @property
    def config(self) -> MonitoringConfiguration:
        return self._config

    def evaluate(
        self,
        dataset: Mapping[str, Sequence[Any]],
        *,
        valid_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
        expected_columns: Optional[Sequence[str]] = None,
        freshness_seconds: Optional[float] = None,
    ) -> QualityMetrics:
        """Evaluate dataset quality and return aggregate metrics + issues."""
        if not dataset:
            raise ValidationError("dataset must contain at least one feature")
        valid_ranges = dict(valid_ranges or {})
        features = {k: list(v) for k, v in dataset.items()}
        lengths = [len(v) for v in features.values()]
        n_rows = max(lengths) if lengths else 0
        min_rows = min(lengths) if lengths else 0
        issues = []

        # Missing values / completeness.
        total_cells = 0
        missing_cells = 0
        for name, values in sorted(features.items()):
            arr = _numeric(values)
            if arr is None:
                col_missing = sum(1 for v in values if v is None or v == "")
            else:
                col_missing = int(np.isnan(arr).sum())
            total_cells += len(values)
            missing_cells += col_missing
            if col_missing > 0:
                rate = col_missing / len(values) if values else 0.0
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MISSING_VALUES, feature_name=name,
                    severity=_severity_from_rate(rate), count=col_missing, rate=rate,
                    message=f"{col_missing} missing values in {name}",
                ))
        null_rate = (missing_cells / total_cells) if total_cells else 0.0
        completeness = 1.0 - null_rate
        if null_rate > self._config.null_rate_threshold:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.NULL_PERCENTAGE,
                severity=_severity_from_rate(null_rate), count=missing_cells, rate=null_rate,
                message=f"Overall null rate {null_rate:.3f} exceeds threshold",
            ))

        # Duplicate records.
        duplicate_rate = 0.0
        if min_rows > 0:
            names = sorted(features)
            rows = [tuple(str(features[name][i]) for name in names) for i in range(min_rows)]
            unique = len(set(rows))
            dup_count = min_rows - unique
            duplicate_rate = dup_count / min_rows
            if dup_count > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.DUPLICATE_RECORDS,
                    severity=_severity_from_rate(duplicate_rate), count=dup_count,
                    rate=duplicate_rate, message=f"{dup_count} duplicate rows",
                ))

        # Invalid ranges / validity.
        invalid_total = 0
        checked_total = 0
        for name, (lo, hi) in sorted(valid_ranges.items()):
            if name not in features:
                continue
            arr = _numeric(features[name])
            if arr is None:
                continue
            clean = arr[~np.isnan(arr)]
            checked_total += clean.size
            invalid = int(np.sum((clean < lo) | (clean > hi)))
            invalid_total += invalid
            if invalid > 0:
                rate = invalid / clean.size if clean.size else 0.0
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INVALID_RANGE, feature_name=name,
                    severity=_severity_from_rate(rate), count=invalid, rate=rate,
                    message=f"{invalid} values of {name} outside [{lo}, {hi}]",
                    details={"min": float(lo), "max": float(hi)},
                ))
        validity_rate = 1.0 - (invalid_total / checked_total) if checked_total else 1.0

        # Outliers.
        outlier_total = 0
        numeric_total = 0
        z = self._config.outlier_z_threshold
        for name, values in sorted(features.items()):
            arr = _numeric(values)
            if arr is None:
                continue
            clean = arr[~np.isnan(arr)]
            if clean.size < 2:
                continue
            std = float(np.std(clean))
            numeric_total += clean.size
            if std == 0.0:
                continue
            scores = np.abs((clean - float(np.mean(clean))) / std)
            outliers = int(np.sum(scores > z))
            outlier_total += outliers
            if outliers > 0:
                rate = outliers / clean.size
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.OUTLIER, feature_name=name,
                    severity=_severity_from_rate(rate), count=outliers, rate=rate,
                    message=f"{outliers} outliers (|z|>{z}) in {name}",
                ))
        outlier_rate = (outlier_total / numeric_total) if numeric_total else 0.0

        # Schema violations.
        schema_violations = 0
        if expected_columns is not None:
            expected = set(expected_columns)
            actual = set(features)
            missing_cols = sorted(expected - actual)
            extra_cols = sorted(actual - expected)
            schema_violations = len(missing_cols) + len(extra_cols)
            if schema_violations > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.SCHEMA_VIOLATION,
                    severity=DriftSeverity.HIGH if schema_violations else DriftSeverity.NONE,
                    count=schema_violations, rate=0.0,
                    message="Schema mismatch",
                    details={"missing": ",".join(missing_cols), "unexpected": ",".join(extra_cols)},
                ))

        # Freshness.
        if freshness_seconds is not None and freshness_seconds > self._freshness_budget:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.FRESHNESS,
                severity=DriftSeverity.HIGH, count=0,
                rate=float(freshness_seconds / self._freshness_budget),
                message=f"Data is stale by {freshness_seconds:.0f}s",
            ))

        consistency_score = 1.0 - duplicate_rate
        ordered = tuple(sorted(issues, key=lambda i: (i.issue_type.value, i.feature_name)))

        return QualityMetrics(
            total_records=n_rows,
            completeness=completeness,
            null_rate=null_rate,
            duplicate_rate=duplicate_rate,
            outlier_rate=outlier_rate,
            validity_rate=validity_rate,
            consistency_score=consistency_score,
            schema_violation_count=schema_violations,
            freshness_seconds=freshness_seconds,
            issue_count=len(ordered),
            issues=ordered,
            created_at=self._clock.now(),
        )


def create_data_quality_monitor(
    *, config: Optional[MonitoringConfiguration] = None, deterministic: bool = True
) -> DataQualityMonitor:
    """Factory for a configured :class:`DataQualityMonitor`."""
    if deterministic:
        return DataQualityMonitor(config=config, clock=LogicalClock(),
                                  id_generator=DeterministicIdGenerator(seed="quality"))
    from monitoring.monitoring_models import SequentialIdGenerator, SystemClock

    return DataQualityMonitor(config=config, clock=SystemClock(),
                              id_generator=SequentialIdGenerator())