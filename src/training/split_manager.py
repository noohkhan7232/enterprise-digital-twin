#!/usr/bin/env python3
"""Enterprise-grade dataset split management for Wind Turbine Acoustic Monitoring.

This module provides a standalone, reusable split engine that operates on a
*manifest* — a list of clip records — independently of the PyTorch dataset
layer.  It centralises every splitting strategy the project needs and, critically,
guarantees **no data leakage** between train / validation / test partitions.

Why a dedicated split manager?
------------------------------
:class:`~src.training.dataset.WindTurbineDataset` already performs a stratified,
group-aware split internally.  That is correct for the common case, but a
production acoustic-monitoring platform needs more:

* **Turbine-aware splitting** — when data is collected from a fleet, the test
  set must contain *entirely unseen turbines* to measure true generalisation.
  A model that has seen turbine T7 in training and is tested on a different
  clip from T7 reports optimistically biased accuracy.
* **Temporal splitting** — for remaining-useful-life and drift studies, the
  test set must be chronologically *after* the training set.  Random splits
  leak future information into the past.
* **Cross-validation** — research papers require k-fold estimates with
  confidence intervals, not a single split.
* **Leakage auditing** — every split must be provably free of group / turbine
  overlap, and that proof must be recorded for the experiment log.

This module encodes all of these as first-class, independently testable
strategies, each producing a fully-audited :class:`SplitResult`.

Design principles
-----------------
1. **Manifest-based** — operates on ``list[dict]`` or ``list[ClipRecord]``;
   never touches the filesystem or loads features.  Fast and pure.
2. **Leakage-proof by construction** — group / turbine splits partition at the
   group level then map back to clips; leakage is additionally *verified* after
   the fact and recorded in :class:`SplitReport`.
3. **Reproducible** — every split is seeded and produces a deterministic
   SHA-256 fingerprint that uniquely identifies the partition.
4. **Backward compatible** — consumes the same :class:`ClipRecord` schema as
   :mod:`src.training.dataset` without modifying it.
5. **Tracker-integrated** — split provenance, sizes, class balance, and the
   leakage audit are logged to :class:`ExperimentTracker`.

Strategies
----------
+----------------------+----------------------------------------------------+
| SplitStrategy        | Guarantee                                          |
+======================+====================================================+
| RANDOM               | i.i.d. random partition                            |
| STRATIFIED           | preserves class proportions in every split         |
| GROUP                | no source recording spans two splits               |
| STRATIFIED_GROUP     | class proportions + group integrity                |
| TURBINE              | no turbine spans two splits (fleet generalisation) |
| TEMPORAL             | train precedes val precedes test chronologically   |
+----------------------+----------------------------------------------------+

Usage::

    from src.training.split_manager import (
        SplitManager, SplitConfig, SplitStrategy,
    )

    manager = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
    result = manager.split(manifest)            # manifest: list[ClipRecord|dict]

    print(result.report.summary())              # human-readable audit
    train_records = result.train                # list[ClipRecord|dict]

    # Cross-validation
    for fold in manager.cross_validate(manifest, n_folds=5):
        train_fold, val_fold = fold.train, fold.val

CLI::

    python src/training/split_manager.py --demo
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

import numpy as np

try:
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        train_test_split,
    )

    _SKLEARN_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ClipRecord is reused for type-compatibility but the manager also accepts dicts.
try:
    from src.training.dataset import ClipRecord  # noqa: F401

    _CLIPRECORD_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ClipRecord = Any  # type: ignore[assignment,misc]
    _CLIPRECORD_AVAILABLE: bool = False

try:
    from src.utils.experiment_tracker import ExperimentTracker  # noqa: F401

    _TRACKER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    ExperimentTracker = Any  # type: ignore[assignment,misc]
    _TRACKER_AVAILABLE: bool = False

logger = logging.getLogger("split_manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_TRAIN: Final[str] = "train"
SPLIT_VAL:   Final[str] = "val"
SPLIT_TEST:  Final[str] = "test"

#: Metadata key inspected for temporal ordering (Unix epoch seconds or ISO str).
_TIMESTAMP_KEYS: Final[tuple[str, ...]] = (
    "timestamp", "recorded_at", "datetime", "acquisition_time", "epoch",
)

#: Metadata / record keys inspected to resolve a turbine identifier.
_TURBINE_KEYS: Final[tuple[str, ...]] = (
    "turbine_id", "turbine", "asset_id", "unit_id",
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SplitStrategy(Enum):
    """Splitting strategy.

    Attributes:
        RANDOM: Independent identically-distributed random partition.
        STRATIFIED: Preserves per-class proportions across splits.
        GROUP: Partitions by group key so no group spans two splits.
        STRATIFIED_GROUP: Combines class balance with group integrity.
        TURBINE: Partitions by turbine ID for fleet-generalisation testing.
        TEMPORAL: Chronological — train precedes val precedes test.
    """

    RANDOM            = "random"
    STRATIFIED        = "stratified"
    GROUP             = "group"
    STRATIFIED_GROUP  = "stratified_group"
    TURBINE           = "turbine"
    TEMPORAL          = "temporal"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for :class:`SplitManager`.

    Attributes:
        strategy: Which :class:`SplitStrategy` to apply.
        train_fraction: Proportion of data assigned to training.
        val_fraction: Proportion assigned to validation.
        test_fraction: Proportion assigned to test.
        random_seed: Seed for all stochastic operations.
        group_key: Record field (or metadata key) used as the group
            identifier for GROUP / STRATIFIED_GROUP strategies.  Defaults
            to deriving a recording id from ``source_path``.
        label_key: Record field holding the integer class label.
        turbine_key: Record/metadata key holding the turbine identifier.
            Used for the TURBINE strategy; defaults to auto-detection from
            :data:`_TURBINE_KEYS`.
        timestamp_key: Record/metadata key holding the timestamp.  Used for
            TEMPORAL; defaults to auto-detection from :data:`_TIMESTAMP_KEYS`.
        shuffle: Shuffle before splitting (ignored for TEMPORAL).
        verify_no_leakage: Run the post-split leakage audit and raise on
            failure for group-based strategies.
        min_test_per_class: Soft target for minimum test clips per class;
            triggers a warning when unmet (does not raise).
        experiment_tracker: Optional :class:`ExperimentTracker` instance.
    """

    strategy:           SplitStrategy = SplitStrategy.STRATIFIED_GROUP
    train_fraction:     float = 0.70
    val_fraction:       float = 0.15
    test_fraction:      float = 0.15
    random_seed:        int = 42
    group_key:          str | None = None
    label_key:          str = "label"
    turbine_key:        str | None = None
    timestamp_key:      str | None = None
    shuffle:            bool = True
    verify_no_leakage:  bool = True
    min_test_per_class: int = 1
    experiment_tracker: Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate fractions and strategy availability."""
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fractions must sum to 1.0, got {total:.4f} "
                f"(train={self.train_fraction}, val={self.val_fraction}, "
                f"test={self.test_fraction})"
            )
        if not (0 < self.train_fraction <= 1):
            raise ValueError(
                f"train_fraction must be in (0, 1], got {self.train_fraction}"
            )
        if self.val_fraction < 0 or self.test_fraction < 0:
            raise ValueError("val_fraction and test_fraction must be non-negative")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageAudit:
    """Result of a post-split leakage check.

    Attributes:
        is_clean: ``True`` when no group/turbine spans multiple splits.
        train_val_overlap: Group keys appearing in both train and val.
        train_test_overlap: Group keys appearing in both train and test.
        val_test_overlap: Group keys appearing in both val and test.
        audited_key: The record field used for the audit (group / turbine).
    """

    is_clean:           bool
    train_val_overlap:  list[str]
    train_test_overlap: list[str]
    val_test_overlap:   list[str]
    audited_key:        str

    @property
    def total_overlaps(self) -> int:
        """Total number of leaking keys across all split pairs."""
        return (
            len(self.train_val_overlap)
            + len(self.train_test_overlap)
            + len(self.val_test_overlap)
        )


@dataclass(frozen=True)
class SplitReport:
    """Full audit trail for a single split operation.

    Attributes:
        strategy: The strategy that produced the split.
        n_total: Total clip count.
        n_train: Training clip count.
        n_val: Validation clip count.
        n_test: Test clip count.
        train_class_balance: Class label -> fraction within train.
        val_class_balance: Class label -> fraction within val.
        test_class_balance: Class label -> fraction within test.
        n_groups: Number of distinct group keys (when applicable).
        leakage: The :class:`LeakageAudit` result.
        fingerprint: Deterministic SHA-256 (first 16 hex) of the partition.
        warnings: Non-fatal issues detected during splitting.
        seed: The random seed used.
    """

    strategy:            str
    n_total:             int
    n_train:             int
    n_val:               int
    n_test:              int
    train_class_balance: dict[int, float]
    val_class_balance:   dict[int, float]
    test_class_balance:  dict[int, float]
    n_groups:            int
    leakage:             LeakageAudit
    fingerprint:         str
    warnings:            list[str]
    seed:                int

    def summary(self) -> str:
        """Return a human-readable multi-line audit summary.

        Returns:
            Formatted string suitable for logging or printing.
        """
        lines = [
            "Split Report",
            "─" * 52,
            f"  Strategy      : {self.strategy}",
            f"  Total clips   : {self.n_total}",
            f"    train        : {self.n_train}  ({self.n_train / max(self.n_total, 1):.1%})",
            f"    val          : {self.n_val}  ({self.n_val / max(self.n_total, 1):.1%})",
            f"    test         : {self.n_test}  ({self.n_test / max(self.n_total, 1):.1%})",
            f"  Distinct groups: {self.n_groups}",
            f"  Fingerprint   : {self.fingerprint}",
            f"  Seed          : {self.seed}",
            f"  Leakage clean : {'✅ YES' if self.leakage.is_clean else '❌ NO'}",
        ]
        if not self.leakage.is_clean:
            lines.append(
                f"    ⚠ {self.leakage.total_overlaps} overlapping "
                f"{self.leakage.audited_key} keys detected!"
            )
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    • {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-compatible dictionary.

        Returns:
            Dictionary with all report fields flattened.
        """
        return {
            "strategy":            self.strategy,
            "n_total":             self.n_total,
            "n_train":             self.n_train,
            "n_val":               self.n_val,
            "n_test":              self.n_test,
            "train_class_balance": self.train_class_balance,
            "val_class_balance":   self.val_class_balance,
            "test_class_balance":  self.test_class_balance,
            "n_groups":            self.n_groups,
            "leakage_clean":       self.leakage.is_clean,
            "leakage_overlaps":    self.leakage.total_overlaps,
            "fingerprint":         self.fingerprint,
            "warnings":            self.warnings,
            "seed":                self.seed,
        }


@dataclass(frozen=True)
class SplitResult:
    """Container for one train/val/test partition plus its audit report.

    Attributes:
        train: Training records (same type as input — dict or ClipRecord).
        val: Validation records.
        test: Test records.
        report: The :class:`SplitReport` audit trail.
    """

    train:  list[Any]
    val:    list[Any]
    test:   list[Any]
    report: SplitReport


@dataclass(frozen=True)
class FoldResult:
    """One cross-validation fold.

    Attributes:
        fold_index: Zero-based fold number.
        train: Training records for this fold.
        val: Validation records for this fold.
        train_class_balance: Class fractions within the training portion.
        val_class_balance: Class fractions within the validation portion.
        leakage: Leakage audit for this fold (group strategies only).
        fingerprint: Deterministic fingerprint for this fold.
    """

    fold_index:          int
    train:               list[Any]
    val:                 list[Any]
    train_class_balance: dict[int, float]
    val_class_balance:   dict[int, float]
    leakage:             LeakageAudit
    fingerprint:         str


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class ManifestValidationError(ValueError):
    """Raised when a manifest fails pre-split validation."""


def _get_field(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a dataclass-like record.

    Checks top-level attributes/keys first, then a nested ``metadata`` dict.

    Args:
        record: A dict or object with attributes.
        key: Field name to read.
        default: Value returned when the key is absent.

    Returns:
        The field value, or *default*.
    """
    if isinstance(record, dict):
        if key in record:
            return record[key]
        meta = record.get("metadata", {})
        if isinstance(meta, dict) and key in meta:
            return meta[key]
        return default

    # Object with attributes
    if hasattr(record, key):
        return getattr(record, key)
    meta = getattr(record, "metadata", {})
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def validate_manifest(
    manifest: Sequence[Any],
    label_key: str = "label",
    *,
    require_unique_ids: bool = True,
) -> list[str]:
    """Validate a manifest before splitting.

    Checks for: empty manifest, missing labels, missing / duplicate clip ids.

    Args:
        manifest: Sequence of clip records (dicts or objects).
        label_key: Field name holding the class label.
        require_unique_ids: Treat duplicate ``clip_id`` values as errors.

    Returns:
        List of human-readable error strings; empty when valid.
    """
    errors: list[str] = []

    if len(manifest) == 0:
        errors.append("Manifest is empty")
        return errors

    seen_ids: set[str] = set()
    for i, record in enumerate(manifest):
        clip_id = _get_field(record, "clip_id")
        if clip_id in (None, ""):
            errors.append(f"Record {i}: missing clip_id")
        elif require_unique_ids and clip_id in seen_ids:
            errors.append(f"Record {i}: duplicate clip_id '{clip_id}'")
        elif clip_id is not None:
            seen_ids.add(clip_id)

        label = _get_field(record, label_key)
        if label is None:
            errors.append(f"Record {i}: missing label ('{label_key}')")
        elif not isinstance(label, (int, np.integer)):
            errors.append(
                f"Record {i}: label must be int, got {type(label).__name__}"
            )

    return errors


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def compute_split_fingerprint(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> str:
    """Compute an order-independent SHA-256 fingerprint of a partition.

    Two partitions with the same clip-id membership produce the same
    fingerprint regardless of internal ordering, enabling exact
    reproducibility checks across runs and machines.

    Args:
        train_ids: Clip ids in the training split.
        val_ids: Clip ids in the validation split.
        test_ids: Clip ids in the test split.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    payload = "|".join([
        ",".join(sorted(str(x) for x in train_ids)),
        ",".join(sorted(str(x) for x in val_ids)),
        ",".join(sorted(str(x) for x in test_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SplitManager
# ---------------------------------------------------------------------------


class SplitManager:
    """Enterprise-grade dataset split engine.

    Produces leakage-audited, reproducible train/val/test partitions and
    cross-validation folds for any manifest of clip records.

    Args:
        config: Split configuration.  Defaults to a stratified-group split.

    Raises:
        ValueError: When ``config`` is invalid.
        RuntimeError: When scikit-learn is required but unavailable.
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.config = config or SplitConfig()
        if not _SKLEARN_AVAILABLE:
            logger.warning(
                "scikit-learn not available; only RANDOM and TEMPORAL "
                "strategies will function."
            )
        logger.info(
            "SplitManager | strategy=%s | %.2f/%.2f/%.2f | seed=%d",
            self.config.strategy.value,
            self.config.train_fraction,
            self.config.val_fraction,
            self.config.test_fraction,
            self.config.random_seed,
        )

    # ------------------------------------------------------------------
    # Public API: single split
    # ------------------------------------------------------------------

    def split(self, manifest: Sequence[Any]) -> SplitResult:
        """Partition *manifest* into train / val / test.

        The strategy is taken from :attr:`SplitConfig.strategy`.  The result
        is fully audited for leakage and assigned a reproducible fingerprint.

        Args:
            manifest: Sequence of clip records (dicts or :class:`ClipRecord`).

        Returns:
            :class:`SplitResult` with the three partitions and an audit report.

        Raises:
            ManifestValidationError: When the manifest fails validation.
            RuntimeError: When a group-based leakage check fails and
                ``verify_no_leakage=True``.
        """
        errors = validate_manifest(manifest, self.config.label_key)
        if errors:
            raise ManifestValidationError(
                f"Manifest validation failed with {len(errors)} error(s): "
                + "; ".join(errors[:5])
                + (" …" if len(errors) > 5 else "")
            )

        records = list(manifest)
        warnings: list[str] = []
        strategy = self.config.strategy

        if strategy == SplitStrategy.TEMPORAL:
            train, val, test = self._split_temporal(records, warnings)
        elif strategy in (SplitStrategy.GROUP, SplitStrategy.STRATIFIED_GROUP):
            train, val, test = self._split_grouped(records, warnings, stratified=(
                strategy == SplitStrategy.STRATIFIED_GROUP
            ))
        elif strategy == SplitStrategy.TURBINE:
            train, val, test = self._split_turbine(records, warnings)
        elif strategy == SplitStrategy.STRATIFIED:
            train, val, test = self._split_stratified(records, warnings)
        else:  # RANDOM
            train, val, test = self._split_random(records, warnings)

        report = self._build_report(
            records, train, val, test, warnings
        )

        if (
            self.config.verify_no_leakage
            and not report.leakage.is_clean
            and strategy in (
                SplitStrategy.GROUP,
                SplitStrategy.STRATIFIED_GROUP,
                SplitStrategy.TURBINE,
            )
        ):
            raise RuntimeError(
                f"Leakage detected in {strategy.value} split: "
                f"{report.leakage.total_overlaps} overlapping "
                f"{report.leakage.audited_key} keys. "
                "This indicates a bug — please report."
            )

        if self.config.experiment_tracker is not None:
            self._log_to_tracker(self.config.experiment_tracker, report)

        logger.info(
            "Split complete | train=%d val=%d test=%d | leakage_clean=%s | fp=%s",
            report.n_train, report.n_val, report.n_test,
            report.leakage.is_clean, report.fingerprint,
        )
        return SplitResult(train=train, val=val, test=test, report=report)

    # ------------------------------------------------------------------
    # Public API: cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        manifest: Sequence[Any],
        n_folds: int = 5,
    ) -> list[FoldResult]:
        """Generate cross-validation folds.

        Uses the appropriate sklearn splitter for the configured strategy:
        :class:`StratifiedGroupKFold`, :class:`GroupKFold`,
        :class:`StratifiedKFold`, or :class:`KFold`.  TEMPORAL strategy uses
        an expanding-window scheme.

        Args:
            manifest: Sequence of clip records.
            n_folds: Number of folds (>= 2).

        Returns:
            List of :class:`FoldResult`, one per fold.

        Raises:
            ManifestValidationError: When the manifest is invalid.
            ValueError: When ``n_folds < 2``.
            RuntimeError: When scikit-learn is required but unavailable.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        errors = validate_manifest(manifest, self.config.label_key)
        if errors:
            raise ManifestValidationError(
                f"Manifest validation failed: {'; '.join(errors[:5])}"
            )

        records = list(manifest)
        labels = np.array(
            [int(_get_field(r, self.config.label_key, 0)) for r in records]
        )
        n = len(records)

        if self.config.strategy == SplitStrategy.TEMPORAL:
            return self._cross_validate_temporal(records, n_folds)

        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "cross_validate requires scikit-learn for non-temporal strategies"
            )

        groups = self._resolve_groups(records)
        splitter, split_args = self._make_cv_splitter(
            n_folds, labels, groups
        )

        folds: list[FoldResult] = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(n), labels, *split_args)
        ):
            train_recs = [records[i] for i in train_idx]
            val_recs   = [records[i] for i in val_idx]
            leakage    = self._audit_leakage(train_recs, val_recs, [], groups_map=None)
            fp = compute_split_fingerprint(
                [_get_field(r, "clip_id", str(i)) for i, r in zip(train_idx, train_recs)],
                [_get_field(r, "clip_id", str(i)) for i, r in zip(val_idx, val_recs)],
                [],
            )
            folds.append(FoldResult(
                fold_index          = fold_idx,
                train               = train_recs,
                val                 = val_recs,
                train_class_balance = self._class_balance(train_recs),
                val_class_balance   = self._class_balance(val_recs),
                leakage             = leakage,
                fingerprint         = fp,
            ))

        logger.info(
            "Cross-validation | %d folds | strategy=%s",
            len(folds), self.config.strategy.value,
        )
        if self.config.experiment_tracker is not None:
            self._log_cv_to_tracker(self.config.experiment_tracker, folds)
        return folds

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _split_random(
        self, records: list[Any], warnings: list[str]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Random i.i.d. split.

        Args:
            records: All clip records.
            warnings: Mutable list collecting non-fatal issues.

        Returns:
            Tuple ``(train, val, test)``.
        """
        rng = np.random.default_rng(self.config.random_seed)
        idx = np.arange(len(records))
        if self.config.shuffle:
            rng.shuffle(idx)
        return self._partition_indices(records, idx)

    def _split_stratified(
        self, records: list[Any], warnings: list[str]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Stratified split preserving class proportions.

        Args:
            records: All clip records.
            warnings: Mutable list collecting non-fatal issues.

        Returns:
            Tuple ``(train, val, test)``.
        """
        labels = np.array(
            [int(_get_field(r, self.config.label_key, 0)) for r in records]
        )
        idx = np.arange(len(records))
        trainval_idx, test_idx = self._safe_split(
            idx, labels, self.config.test_fraction, self.config.random_seed
        )
        val_frac = self.config.val_fraction / max(
            self.config.train_fraction + self.config.val_fraction, 1e-9
        )
        train_idx, val_idx = self._safe_split(
            trainval_idx, labels[trainval_idx], val_frac,
            self.config.random_seed + 1,
        )
        return (
            [records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx],
        )

    def _split_grouped(
        self, records: list[Any], warnings: list[str], *, stratified: bool
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Group-aware split; optionally stratified by class.

        Partitions at the group level so no group spans two splits, then
        maps groups back to clips.

        Args:
            records: All clip records.
            warnings: Mutable list collecting non-fatal issues.
            stratified: Preserve class proportions across splits.

        Returns:
            Tuple ``(train, val, test)``.
        """
        groups = self._resolve_groups(records)
        return self._partition_by_group(records, groups, warnings, stratified)

    def _split_turbine(
        self, records: list[Any], warnings: list[str]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Turbine-aware split (no turbine spans two splits).

        Args:
            records: All clip records.
            warnings: Mutable list collecting non-fatal issues.

        Returns:
            Tuple ``(train, val, test)``.
        """
        turbines = self._resolve_turbines(records)
        unique_turbines = set(turbines)
        if len(unique_turbines) < 3:
            warnings.append(
                f"Only {len(unique_turbines)} distinct turbine(s); turbine-aware "
                "split cannot guarantee all three partitions are non-empty."
            )
        return self._partition_by_group(
            records, turbines, warnings, stratified=False
        )

    def _split_temporal(
        self, records: list[Any], warnings: list[str]
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Chronological split: train precedes val precedes test.

        Args:
            records: All clip records.
            warnings: Mutable list collecting non-fatal issues.

        Returns:
            Tuple ``(train, val, test)``.
        """
        timestamps = self._resolve_timestamps(records)
        if timestamps is None:
            warnings.append(
                "No timestamp metadata found; falling back to insertion order "
                "for temporal split."
            )
            order = np.arange(len(records))
        else:
            order = np.argsort(timestamps, kind="stable")

        n = len(records)
        n_test = int(round(n * self.config.test_fraction))
        n_val  = int(round(n * self.config.val_fraction))
        n_test = max(0, min(n_test, n))
        n_val  = max(0, min(n_val, n - n_test))

        test_idx  = order[n - n_test:] if n_test > 0 else np.array([], dtype=int)
        val_idx   = order[n - n_test - n_val: n - n_test] if n_val > 0 else np.array([], dtype=int)
        train_idx = order[: n - n_test - n_val]

        return (
            [records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx],
        )

    # ------------------------------------------------------------------
    # Partition helpers
    # ------------------------------------------------------------------

    def _partition_indices(
        self, records: list[Any], idx: np.ndarray
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Slice a (possibly shuffled) index array into 3 fractions.

        Args:
            records: All clip records.
            idx: Ordering of indices to partition.

        Returns:
            Tuple ``(train, val, test)``.
        """
        n = len(idx)
        n_train = int(round(n * self.config.train_fraction))
        n_val   = int(round(n * self.config.val_fraction))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train: n_train + n_val]
        test_idx  = idx[n_train + n_val:]
        return (
            [records[i] for i in train_idx],
            [records[i] for i in val_idx],
            [records[i] for i in test_idx],
        )

    def _partition_by_group(
        self,
        records: list[Any],
        groups: Sequence[str],
        warnings: list[str],
        stratified: bool,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Partition clips by group key, optionally stratifying.

        Each unique group is assigned wholesale to exactly one split.

        Args:
            records: All clip records.
            groups: Per-record group identifier.
            warnings: Mutable list collecting non-fatal issues.
            stratified: Stratify groups by their majority class label.

        Returns:
            Tuple ``(train, val, test)``.
        """
        groups_arr = np.asarray(groups)
        unique_groups = list(dict.fromkeys(groups_arr.tolist()))

        # Representative label per group (majority class)
        labels = np.array(
            [int(_get_field(r, self.config.label_key, 0)) for r in records]
        )
        group_label: dict[str, int] = {}
        for g in unique_groups:
            mask = groups_arr == g
            group_label[g] = int(Counter(labels[mask].tolist()).most_common(1)[0][0])

        group_idx = np.arange(len(unique_groups))
        group_labels = np.array([group_label[g] for g in unique_groups])

        seed = self.config.random_seed
        if stratified:
            trainval_g, test_g = self._safe_split(
                group_idx, group_labels, self.config.test_fraction, seed
            )
            val_frac = self.config.val_fraction / max(
                self.config.train_fraction + self.config.val_fraction, 1e-9
            )
            train_g, val_g = self._safe_split(
                trainval_g, group_labels[trainval_g], val_frac, seed + 1
            )
        else:
            rng = np.random.default_rng(seed)
            shuffled = group_idx.copy()
            rng.shuffle(shuffled)
            n_g = len(shuffled)
            n_test_g, n_val_g = self._allocate_group_counts(n_g, warnings)
            test_g  = shuffled[:n_test_g]
            val_g   = shuffled[n_test_g: n_test_g + n_val_g]
            train_g = shuffled[n_test_g + n_val_g:]

        train_set = {unique_groups[g] for g in train_g}
        val_set   = {unique_groups[g] for g in val_g}
        test_set  = {unique_groups[g] for g in test_g}

        train, val, test = [], [], []
        for record, g in zip(records, groups_arr):
            if g in train_set:
                train.append(record)
            elif g in val_set:
                val.append(record)
            else:
                test.append(record)
        return train, val, test

    def _allocate_group_counts(
        self, n_groups: int, warnings: list[str]
    ) -> tuple[int, int]:
        """Allocate group counts to test and val, guaranteeing non-empty splits.

        When a fraction is positive and there are enough groups, at least one
        group is assigned to that partition.  Train is never starved below one
        group.

        Args:
            n_groups: Total number of distinct groups.
            warnings: Mutable list collecting non-fatal issues.

        Returns:
            Tuple ``(n_test_groups, n_val_groups)``.
        """
        n_test = int(round(n_groups * self.config.test_fraction))
        n_val  = int(round(n_groups * self.config.val_fraction))

        if self.config.test_fraction > 0:
            n_test = max(1, n_test)
        if self.config.val_fraction > 0:
            n_val = max(1, n_val)

        # Never starve train below one group
        if n_test + n_val >= n_groups:
            n_test = min(n_test, max(0, n_groups - 1))
            n_val  = min(n_val, max(0, n_groups - n_test - 1))
            if (self.config.val_fraction > 0 and n_val == 0) or (
                self.config.test_fraction > 0 and n_test == 0
            ):
                warnings.append(
                    f"Only {n_groups} distinct group(s); cannot fill all "
                    "partitions — some splits may be empty."
                )
        return n_test, n_val

    @staticmethod
    def _safe_split(
        indices: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified split with graceful fallback to random.

        Mirrors the contract used in
        :meth:`src.training.dataset.WindTurbineDataset._safe_split` so both
        layers behave identically at small sample sizes.

        Args:
            indices: Indices to split.
            labels: Class labels aligned to *indices*.
            test_size: Fraction assigned to the second partition.
            random_state: RNG seed.

        Returns:
            Tuple ``(first_indices, second_indices)``.
        """
        if len(indices) < 2:
            return indices, np.array([], dtype=int)

        effective = max(test_size, 1.0 / len(indices))
        effective = min(effective, 1.0 - 1.0 / len(indices))

        n_classes = len(np.unique(labels))
        min_count = int(np.min(np.bincount(labels))) if len(labels) else 0
        n_test = max(1, int(round(len(indices) * effective)))
        stratify = (
            labels
            if (_SKLEARN_AVAILABLE and n_classes >= 2
                and min_count >= 2 and n_test >= n_classes)
            else None
        )

        if not _SKLEARN_AVAILABLE:
            rng = np.random.default_rng(random_state)
            shuffled = np.array(indices).copy()
            rng.shuffle(shuffled)
            cut = len(shuffled) - n_test
            return shuffled[:cut], shuffled[cut:]

        try:
            return train_test_split(
                indices, test_size=effective,
                stratify=stratify, random_state=random_state,
            )
        except ValueError:
            return train_test_split(
                indices, test_size=effective,
                stratify=None, random_state=random_state,
            )

    # ------------------------------------------------------------------
    # Cross-validation helpers
    # ------------------------------------------------------------------

    def _make_cv_splitter(
        self,
        n_folds: int,
        labels: np.ndarray,
        groups: Sequence[str] | None,
    ) -> tuple[Any, tuple]:
        """Select the appropriate sklearn CV splitter for the strategy.

        Args:
            n_folds: Number of folds.
            labels: Class labels.
            groups: Per-record group keys, or ``None``.

        Returns:
            Tuple ``(splitter, extra_split_args)`` where *extra_split_args*
            is passed positionally to ``splitter.split``.
        """
        strategy = self.config.strategy
        if strategy in (SplitStrategy.GROUP, SplitStrategy.TURBINE):
            return GroupKFold(n_splits=n_folds), (np.asarray(groups),)
        if strategy == SplitStrategy.STRATIFIED_GROUP:
            return (
                StratifiedGroupKFold(
                    n_splits=n_folds, shuffle=True,
                    random_state=self.config.random_seed,
                ),
                (np.asarray(groups),),
            )
        if strategy == SplitStrategy.STRATIFIED:
            return (
                StratifiedKFold(
                    n_splits=n_folds, shuffle=True,
                    random_state=self.config.random_seed,
                ),
                (),
            )
        # RANDOM
        from sklearn.model_selection import KFold
        return (
            KFold(n_splits=n_folds, shuffle=True,
                  random_state=self.config.random_seed),
            (),
        )

    def _cross_validate_temporal(
        self, records: list[Any], n_folds: int
    ) -> list[FoldResult]:
        """Expanding-window temporal cross-validation.

        Fold *k* trains on the first ``(k+1)`` blocks and validates on block
        ``(k+1)``; train always precedes val chronologically.

        Args:
            records: All clip records.
            n_folds: Number of folds.

        Returns:
            List of :class:`FoldResult`.
        """
        timestamps = self._resolve_timestamps(records)
        order = (
            np.argsort(timestamps, kind="stable")
            if timestamps is not None
            else np.arange(len(records))
        )
        blocks = np.array_split(order, n_folds + 1)

        folds: list[FoldResult] = []
        for k in range(n_folds):
            train_idx = np.concatenate(blocks[: k + 1])
            val_idx   = blocks[k + 1]
            train_recs = [records[i] for i in train_idx]
            val_recs   = [records[i] for i in val_idx]
            fp = compute_split_fingerprint(
                [_get_field(r, "clip_id", str(i)) for i, r in zip(train_idx, train_recs)],
                [_get_field(r, "clip_id", str(i)) for i, r in zip(val_idx, val_recs)],
                [],
            )
            folds.append(FoldResult(
                fold_index          = k,
                train               = train_recs,
                val                 = val_recs,
                train_class_balance = self._class_balance(train_recs),
                val_class_balance   = self._class_balance(val_recs),
                leakage             = LeakageAudit(True, [], [], [], "temporal"),
                fingerprint         = fp,
            ))
        return folds

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def _resolve_groups(self, records: list[Any]) -> list[str]:
        """Resolve a group key for every record.

        Uses ``config.group_key`` when set; otherwise derives a recording id
        from ``source_path`` (stem with trailing ``_NNN`` chunk index removed),
        falling back to ``clip_id``.

        Args:
            records: All clip records.

        Returns:
            List of group identifier strings.
        """
        key = self.config.group_key
        groups: list[str] = []
        for r in records:
            if key is not None:
                val = _get_field(r, key)
                groups.append(str(val) if val is not None else "_nogroup")
                continue
            src = _get_field(r, "source_path", "")
            if src:
                stem = Path(str(src)).stem
                groups.append(stem.rsplit("_", 1)[0] if "_" in stem else stem)
            else:
                groups.append(str(_get_field(r, "clip_id", "_nogroup")))
        return groups

    def _resolve_turbines(self, records: list[Any]) -> list[str]:
        """Resolve a turbine identifier for every record.

        Uses ``config.turbine_key`` when set; otherwise auto-detects from
        :data:`_TURBINE_KEYS`, falling back to the ``dataset`` field (which
        carries the turbine/source name in multi-turbine deployments).

        Args:
            records: All clip records.

        Returns:
            List of turbine identifier strings.
        """
        key = self.config.turbine_key
        turbines: list[str] = []
        for r in records:
            val = None
            if key is not None:
                val = _get_field(r, key)
            else:
                for candidate in _TURBINE_KEYS:
                    val = _get_field(r, candidate)
                    if val is not None:
                        break
                if val is None:
                    val = _get_field(r, "dataset")
            turbines.append(str(val) if val is not None else "_noturbine")
        return turbines

    def _resolve_timestamps(self, records: list[Any]) -> np.ndarray | None:
        """Resolve a numeric timestamp for every record.

        Uses ``config.timestamp_key`` when set; otherwise auto-detects from
        :data:`_TIMESTAMP_KEYS`.  ISO-8601 strings are parsed to epoch.

        Args:
            records: All clip records.

        Returns:
            Float array of timestamps, or ``None`` when none are available.
        """
        key = self.config.timestamp_key
        raw: list[Any] = []
        for r in records:
            val = None
            if key is not None:
                val = _get_field(r, key)
            else:
                for candidate in _TIMESTAMP_KEYS:
                    val = _get_field(r, candidate)
                    if val is not None:
                        break
            raw.append(val)

        if all(v is None for v in raw):
            return None

        out = np.zeros(len(raw), dtype=float)
        for i, v in enumerate(raw):
            out[i] = self._to_epoch(v, fallback=float(i))
        return out

    @staticmethod
    def _to_epoch(value: Any, fallback: float) -> float:
        """Convert a timestamp value to epoch seconds.

        Args:
            value: Numeric epoch, ISO-8601 string, or ``None``.
            fallback: Returned when *value* cannot be parsed.

        Returns:
            Epoch seconds as a float.
        """
        if value is None:
            return fallback
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                return fallback
        return fallback

    # ------------------------------------------------------------------
    # Audit / reporting
    # ------------------------------------------------------------------

    def _audit_leakage(
        self,
        train: list[Any],
        val: list[Any],
        test: list[Any],
        groups_map: dict[int, str] | None = None,
    ) -> LeakageAudit:
        """Verify no group/turbine key spans multiple splits.

        Args:
            train: Training records.
            val: Validation records.
            test: Test records.
            groups_map: Unused placeholder for API symmetry.

        Returns:
            :class:`LeakageAudit` describing any overlaps.
        """
        strategy = self.config.strategy
        if strategy == SplitStrategy.TURBINE:
            key_name = "turbine"
            resolver = self._resolve_turbines
        elif strategy in (SplitStrategy.GROUP, SplitStrategy.STRATIFIED_GROUP):
            key_name = "group"
            resolver = self._resolve_groups
        else:
            # Non-group strategies: audit by clip_id (always leakage-free)
            key_name = "clip_id"
            resolver = lambda recs: [
                str(_get_field(r, "clip_id", id(r))) for r in recs
            ]

        train_keys = set(resolver(train))
        val_keys   = set(resolver(val))
        test_keys  = set(resolver(test))

        tv = sorted(train_keys & val_keys)
        tt = sorted(train_keys & test_keys)
        vt = sorted(val_keys & test_keys)

        return LeakageAudit(
            is_clean           = not (tv or tt or vt),
            train_val_overlap  = tv,
            train_test_overlap = tt,
            val_test_overlap   = vt,
            audited_key        = key_name,
        )

    def _class_balance(self, records: list[Any]) -> dict[int, float]:
        """Compute class-label fractions within a record list.

        Args:
            records: Records to summarise.

        Returns:
            Mapping of integer label to fraction (sums to 1.0).
        """
        if not records:
            return {}
        labels = [int(_get_field(r, self.config.label_key, 0)) for r in records]
        counts = Counter(labels)
        total = len(labels)
        return {int(k): round(v / total, 4) for k, v in sorted(counts.items())}

    def _build_report(
        self,
        all_records: list[Any],
        train: list[Any],
        val: list[Any],
        test: list[Any],
        warnings: list[str],
    ) -> SplitReport:
        """Assemble the :class:`SplitReport` audit trail.

        Args:
            all_records: The full manifest.
            train: Training records.
            val: Validation records.
            test: Test records.
            warnings: Accumulated non-fatal warnings.

        Returns:
            A fully-populated :class:`SplitReport`.
        """
        leakage = self._audit_leakage(train, val, test)

        # Soft check: minimum test clips per class
        test_balance = self._class_balance(test)
        test_counts = Counter(
            int(_get_field(r, self.config.label_key, 0)) for r in test
        )
        for label, count in test_counts.items():
            if count < self.config.min_test_per_class:
                warnings.append(
                    f"Class {label} has only {count} test clip(s) "
                    f"(target {self.config.min_test_per_class})."
                )
        # Empty-split warnings
        if not val:
            warnings.append("Validation split is empty.")
        if not test:
            warnings.append("Test split is empty.")

        groups = self._resolve_groups(all_records)
        fingerprint = compute_split_fingerprint(
            [str(_get_field(r, "clip_id", i)) for i, r in enumerate(train)],
            [str(_get_field(r, "clip_id", i)) for i, r in enumerate(val)],
            [str(_get_field(r, "clip_id", i)) for i, r in enumerate(test)],
        )

        return SplitReport(
            strategy            = self.config.strategy.value,
            n_total             = len(all_records),
            n_train             = len(train),
            n_val               = len(val),
            n_test              = len(test),
            train_class_balance = self._class_balance(train),
            val_class_balance   = self._class_balance(val),
            test_class_balance  = test_balance,
            n_groups            = len(set(groups)),
            leakage             = leakage,
            fingerprint         = fingerprint,
            warnings            = warnings,
            seed                = self.config.random_seed,
        )

    # ------------------------------------------------------------------
    # ExperimentTracker integration
    # ------------------------------------------------------------------

    def _log_to_tracker(self, tracker: Any, report: SplitReport) -> None:
        """Log split provenance to an ExperimentTracker.

        Args:
            tracker: An :class:`ExperimentTracker` instance.
            report: The split report to log.
        """
        try:
            tracker.log_params({
                "split.strategy":       report.strategy,
                "split.n_train":        report.n_train,
                "split.n_val":          report.n_val,
                "split.n_test":         report.n_test,
                "split.n_groups":       report.n_groups,
                "split.fingerprint":    report.fingerprint,
                "split.seed":           report.seed,
                "split.leakage_clean":  report.leakage.is_clean,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker logging failed: %s", exc)

    def _log_cv_to_tracker(
        self, tracker: Any, folds: list[FoldResult]
    ) -> None:
        """Log cross-validation provenance to an ExperimentTracker.

        Args:
            tracker: An :class:`ExperimentTracker` instance.
            folds: The CV folds.
        """
        try:
            tracker.log_params({
                "cv.strategy":   self.config.strategy.value,
                "cv.n_folds":    len(folds),
                "cv.seed":       self.config.random_seed,
                "cv.fingerprints": [f.fingerprint for f in folds],
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperimentTracker CV logging failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Run a self-contained demonstration of all strategies.

    Returns:
        Exit code 0.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build a synthetic manifest: 4 classes, 3 turbines, timestamps
    fault_types = ["normal", "bearing_fault", "blade_imbalance", "gearbox_fault"]
    manifest: list[dict[str, Any]] = []
    for ci, ft in enumerate(fault_types):
        for i in range(8):
            manifest.append({
                "clip_id":     f"{ft}_{i:03d}",
                "fault_type":  ft,
                "label":       ci,
                "source_path": f"data/raw/{ft}/{ft}_{i // 2:03d}.wav",
                "dataset":     f"turbine_{i % 3}",
                "metadata":    {"timestamp": 1_700_000_000 + ci * 86400 + i * 3600},
            })

    for strategy in SplitStrategy:
        print(f"\n{'=' * 56}\nStrategy: {strategy.value}\n{'=' * 56}")
        cfg = SplitConfig(strategy=strategy, verify_no_leakage=True)
        manager = SplitManager(cfg)
        try:
            result = manager.split(manifest)
            print(result.report.summary())
        except (RuntimeError, ManifestValidationError) as exc:
            print(f"  Split failed: {exc}")

    # Cross-validation demo
    print(f"\n{'=' * 56}\nCross-validation (StratifiedGroup, 4 folds)\n{'=' * 56}")
    cv_manager = SplitManager(SplitConfig(strategy=SplitStrategy.STRATIFIED_GROUP))
    folds = cv_manager.cross_validate(manifest, n_folds=4)
    for fold in folds:
        print(
            f"  Fold {fold.fold_index}: train={len(fold.train)} "
            f"val={len(fold.val)} clean={fold.leakage.is_clean} "
            f"fp={fold.fingerprint}"
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="SplitManager demo and self-test",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run the full strategy demonstration.")
    args = parser.parse_args(argv)

    if args.demo:
        return _demo()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())