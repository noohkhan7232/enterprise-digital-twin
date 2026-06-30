# Developer Guide

This guide is for engineers contributing to the platform. It describes the repository structure,
coding standards, testing strategy, design principles and contribution workflow.

---

## 1. Repository Structure

```
src/            Capability and production-engineering packages
scripts/        CI/CD validation (quality gates, release, readiness)
deployment/     Docker, Kubernetes, automation and the health check
configs/        YAML policy and configuration
tests/          Deterministic automated test suite (one file per module)
docs/           Documentation, research paper and guides
.github/        Issue/PR templates, contribution policy, workflows
```

Each subsystem under `src/` is internally complete and externally minimal, exposing a small surface
of factory functions and immutable value objects. Tests mirror modules one-to-one, which keeps the
mapping between code and verification obvious.

## 2. Coding Standards

- **Typing.** All public functions and methods are fully type-annotated. Type clarity is treated as
  part of the interface contract.
- **Immutability.** Domain types are frozen, slotted dataclasses that carry their own
  serialisation (`to_dict`/`from_dict`) and round-trip losslessly to and from JSON.
- **Determinism.** No component reads a wall clock or a random source directly; time and identity
  are injected. Collections are ordered before serialisation and numerical results are rounded
  consistently.
- **Thread safety.** Stateful engines guard mutable internals with re-entrant locks and expose
  immutable snapshots; readers never observe partial updates.
- **No placeholders in code.** Modules contain no TODOs, stubs or pseudocode; behaviour is complete
  and tested.
- **Documentation.** Modules and public types carry concise docstrings describing intent and
  contracts, not implementation trivia.

## 3. Testing Strategy

The suite contains 1,503 deterministic, framework-agnostic tests. The strategy has four levels:

1. **Value-object tests** isolate serialisation, validation and immutability defects.
2. **Engine tests** isolate computational defects — percentiles, trends, availability, error
   budgets, critical paths, forecasts and readiness scoring are checked against known inputs.
3. **Edge-case tests** cover empty and single-element inputs, boundary thresholds, illegal state
   transitions and exhausted budgets.
4. **Determinism tests** assert that repeated execution with identical inputs produces identical
   outputs, including fully serialised reports.

Tests rely only on standard assertions and parameterisation, with no fixtures, network access or
external services, so a failure points to a single component. Running the full suite re-verifies
the contracts of every subsystem.

```bash
PYTHONPATH=src:scripts pytest tests/ -q
```

## 4. Design Principles

| Principle | How it is applied |
|-----------|-------------------|
| SOLID | Narrow responsibilities; extension via new types and injected strategies; small interfaces; dependencies on abstractions |
| Dependency injection | Clocks, identifier sources, strategies, weights and probes are injected |
| Immutable domain models | Frozen, slotted, self-serialising value objects |
| Determinism | Injected time/identity; ordered, rounded outputs |
| Thread safety | Re-entrant locks with immutable snapshots |
| Composition over inheritance | Subsystems compose outputs; inheritance reserved for genuine specialisation |

These principles are not aspirational: dependency injection is what makes the system testable
offline, immutability is what makes values safe to share, and determinism is what makes outputs
reproducible.

## 5. Contribution Workflow

1. **Open an issue.** Use the bug report or feature request template under
   `.github/ISSUE_TEMPLATE/` to describe the change and its motivation.
2. **Branch.** Create a topic branch from the default branch.
3. **Implement additively.** Add new modules and tests; do not modify locked prior work unless the
   change is explicitly scoped to it. Maintain backward compatibility.
4. **Test.** Add tests at the level where the behaviour is introduced and run the full suite
   locally. New behaviour must be deterministic and covered.
5. **Validate.** Ensure the CI/CD quality gates pass; address any honest gate failures rather than
   suppressing them.
6. **Open a pull request.** Complete the pull-request template, including the testing and
   compatibility checklist. Keep pull requests focused and reviewable.
7. **Review.** Address feedback constructively. Merges require passing checks and review approval.

By convention, contributions preserve the architectural invariants: a single dependency direction,
integration by composition, and additive change.