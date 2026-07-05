# Week 11 — Phase 1: Enterprise MLOps Platform

> Subsystem: `src/mlops/`
> Status: production-ready, deterministic, pure Python + NumPy.
> Integration model: **composition only** — this subsystem adds new files and
> does not modify, rename, move or re-architect any code from Weeks 1–10.

---

## 1. Business Motivation

Industrial AI systems fail in production not because models are inaccurate, but
because the *process* around them is unaccountable. Teams cannot answer basic
questions: Which dataset trained the model now serving traffic? Who approved its
promotion? Can we reproduce last quarter's result bit-for-bit? Which version do
we roll back to when a regression ships?

This subsystem answers those questions deterministically. It provides experiment
tracking, a governed model registry, a metadata artifact store and a
reproducibility engine that together form an auditable chain from dataset to
deployment. It deliberately avoids dependencies on MLflow, Weights & Biases or
Kubeflow: every behaviour is owned, inspectable and reproducible, which is a hard
requirement in regulated and safety-critical industrial settings.

The design optimises for three properties valued by principal-engineering review:
**determinism** (identical inputs produce byte-identical outputs), **governance**
(promotions pass explicit gates and leave an audit trail) and **composability**
(clean value objects and injected strategies, no hidden global state).

## 2. Architecture

The subsystem is layered. A foundation of immutable value objects underpins four
independent services, each constructed through dependency injection.

```
                         experiment_models.py
        (immutable dataclasses · enums · SemanticVersion · Clock · IdGenerator)
                                    |
        +---------------------+-----+-------------------+--------------------+
        |                     |                         |                    |
 experiment_tracker.py   model_registry.py        artifact_store.py   reproducibility.py
 (experiments / runs)    (lifecycle / promotion   (metadata-only      (capture / verify
                          / comparison / cards)     content-addressed)   environments)
```

Cross-cutting design patterns:

- **Dependency Injection** — every service accepts a `Clock`, an `IdGenerator`
  and (where relevant) an `EnvironmentProvider`, `HashStrategy`, `PromotionPolicy`
  or `ModelComparator`. Defaults are deterministic; production variants are
  supplied through factories.
- **Strategy** — promotion gating (`PromotionPolicy`), scoring
  (`ModelComparator`), hashing (`HashStrategy`) and environment capture
  (`EnvironmentProvider`) are all swappable behaviours.
- **Factory** — `create_experiment_tracker`, `create_model_registry`,
  `create_artifact_store` and `create_reproducibility_engine` assemble configured
  services for either deterministic or system-backed operation.
- **Registry** — the model registry and artifact/experiment stores are explicit
  registries with controlled mutation behind a re-entrant lock.

All aggregates are `@dataclass(frozen=True, slots=True)`, hashable and
JSON-serialisable via symmetric `to_dict()` / `from_dict()`.

## 3. Experiment Tracking

`ExperimentTracker` records experiments and their runs with complete provenance:
experiment and run identifiers, parent experiment, timestamps, dataset version,
git commit, random and NumPy seeds, Python version, platform, hostname,
hyperparameters, training/validation/test metrics, training and inference time,
memory and CPU usage, notes and tags.

The lifecycle is: create an experiment, start a run (status `RUNNING`), log
metrics per split, log resource usage, register artifacts, then finalise the run
(`COMPLETED` or `FAILED`). Because every model is immutable, each mutation
produces a new value object stored under a lock, so concurrent runs never corrupt
shared state. `statistics()` aggregates run counts, average training time and the
best validation score deterministically.

## 4. Model Registry

`ModelRegistry` governs the model lifecycle as an explicit state machine:

```
REGISTERED -> VALIDATION -> STAGING -> PRODUCTION -> ARCHIVED
```

It supports `register_model` (with duplicate detection and semantic-version
validation), `promote`, `rollback`, `archive`, `compare_versions`, `latest` and
`history`. Registering a duplicate `(model_id, version)` raises
`DuplicateModelVersionError`; an invalid version raises `ValidationError`.
Promoting a version to `PRODUCTION` automatically archives the previously
production version, and `rollback` reinstates an earlier version while archiving
the current one.

## 5. Artifact Store

`ArtifactStore` records metadata for ten artifact types — serialized models,
evaluation reports, engineering reports, publication figures, feature importance,
SHAP results, confusion matrices, ROC curves, PR curves and calibration plots. No
binary payloads are stored. Each artifact is content-addressed by a deterministic
SHA-256 digest of its descriptive properties, so identical artifacts hash
identically across processes. Artifacts can be filtered by type, experiment or
run, looked up by content hash, and exported to JSON.

## 6. Model Lineage

Every `ModelVersion` exposes a complete provenance chain:

```
DatasetVersion -> Experiment -> Run -> Artifacts -> ModelVersion -> DeploymentStage
```

`registry.lineage(model_id, version)` returns this chain as a JSON-ready mapping,
making it trivial to answer "what produced this production model?" during an
audit. Because the chain is carried on immutable versions, lineage cannot drift
after the fact.

## 7. Promotion Workflow

`DefaultPromotionPolicy` approves a promotion only when every applicable gate
passes:

1. the lifecycle transition is structurally valid;
2. validation is complete when entering `STAGING` or `PRODUCTION`;
3. an approver is recorded when entering `STAGING` or `PRODUCTION`;
4. the primary metric is no worse than the baseline (when required), respecting
   `MAXIMIZE` / `MINIMIZE` direction;
5. latency is within the configured budget;
6. memory is within the configured budget.

The policy returns a `PromotionDecision` carrying the status, the per-gate check
results and human-readable reasons. `evaluate_promotion` computes the decision
without mutating state; `promote` applies the transition only on approval. This
separation supports dry-run governance previews.

## 8. Reproducibility

`ReproducibilityEngine` captures a `ReproducibilitySnapshot` containing the Python
and NumPy versions, dependency versions, whitelisted environment variables, random
and NumPy seeds, a configuration snapshot, the dataset version, the git commit, and
runtime and hardware information. Environment introspection is performed through an
injected `EnvironmentProvider`: `StaticEnvironmentProvider` for fully reproducible
captures and `SystemEnvironmentProvider` for real environments. `verify()` compares
two snapshots while ignoring volatile fields (identifiers and timestamps), and
`diff()` reports exactly which inputs changed.

## 9. Engineering Decisions

- **Determinism over convenience.** No component reads the wall clock or a random
  source directly; time and identity are injected. Default identifiers are derived
  from a seed and a per-prefix counter via SHA-256, so runs are reproducible.
- **Hashable immutability.** Mapping-like fields are stored as sorted tuples of
  key/value pairs rather than dictionaries, keeping aggregates frozen, hashable
  and deterministically serialisable.
- **Semantic-version precedence.** `SemanticVersion` implements full semver
  ordering, including the rule that a release outranks any prerelease of the same
  core, used for deterministic tie-breaking during comparison.
- **Audit-friendly serialisation.** Every `to_json` sorts keys, producing stable
  diffs in version control and reproducible export artifacts.
- **Thread safety.** Each service guards its mutable index with a re-entrant lock
  while exposing only immutable value objects, eliminating data races.

## 10. Performance

All operations are in-memory dictionary or list operations. Registration,
retrieval and promotion are O(1) on the relevant index. `versions`, `history`,
`compare_versions` and `statistics` are O(n log n) in the number of versions due
to deterministic sorting. The model comparator normalises cost metrics in a single
linear pass over the compared set. The suite exercises stores and registries at
thousands of entries within interactive latency.

## 11. Complexity

| Operation | Complexity |
|-----------|------------|
| `register_model`, `get_version`, `promote`, `rollback`, `archive` | O(1) |
| `versions`, `history`, `latest` | O(n log n) over a model's versions |
| `compare_versions` | O(n log n) over the compared set |
| `statistics` (registry / tracker) | O(n) over versions / runs |
| `ArtifactStore.list_artifacts` | O(n) filter + O(n log n) sort |
| reproducibility `capture` / `verify` / `diff` | O(k) in captured fields |

## 12. Enterprise Applications

- **Regulated deployment.** Promotion gates and immutable lineage produce an audit
  trail suitable for model-risk-management review.
- **Industrial digital twins.** Deterministic snapshots allow a model serving a
  twin to be reproduced exactly during incident analysis.
- **Champion/challenger workflows.** The comparison engine ranks candidate versions
  deterministically across quality and cost dimensions.
- **Safe rollback.** One call reinstates a known-good production version while
  archiving the regressed one.

## 13. Future Integration with CI/CD

The JSON exports (`experiment.json`, `registry.json`, `model_card.json`,
`comparison.json`) are designed as CI artifacts. A pipeline can: capture a
reproducibility snapshot at train time and fail the build if `verify()` detects
drift; call `evaluate_promotion` as a required status check before merge; gate
deployment on `PromotionDecision.status`; and publish the generated model card to
a documentation portal. Because promotion is policy-driven, organisation-specific
gates plug in by supplying an alternative `PromotionPolicy` — no core change
required.