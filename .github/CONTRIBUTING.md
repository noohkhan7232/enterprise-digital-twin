# Contributing

Thank you for your interest in contributing to the Enterprise Digital Twin & Decision Intelligence
Platform. This document explains how to propose changes and the standards contributions are expected
to meet.

## Guiding Principles

The platform is built on a small number of invariants. Contributions are expected to preserve them:

- **Additive change.** Add new modules and tests; do not modify locked prior work unless a change is
  explicitly scoped to it. Maintain backward compatibility.
- **Integration by composition.** Subsystems exchange immutable value objects; do not introduce
  shared mutable state or reach into another subsystem's internals.
- **Determinism.** Inject time and identity; order and round outputs. Behaviour must be reproducible.
- **Immutability, typing and thread safety.** Public domain types are frozen, slotted and fully
  typed; stateful components are thread-safe.

## Getting Started

```bash
git clone https://github.com/<org>/enterprise-digital-twin.git
cd enterprise-digital-twin
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyyaml pytest pytest-cov
```

## Development Workflow

1. **Open an issue** using the appropriate template to describe the change and its motivation.
2. **Create a topic branch** from the default branch.
3. **Implement** the change additively, with full type annotations and no placeholder logic.
4. **Add tests** at the level where the behaviour is introduced — value-object, engine, edge-case
   and (where applicable) determinism tests. Tests must be deterministic and framework-agnostic.
5. **Run the full suite** and the quality gates locally:
   ```bash
   PYTHONPATH=src:scripts pytest tests/ -q
   ```
6. **Update documentation** if interfaces or behaviour changed.
7. **Open a pull request** and complete the pull-request template, including the engineering and
   testing checklists.

## Coding Standards

- Full type annotations on public functions and methods.
- Immutable, self-serialising domain types (`to_dict`/`from_dict`, lossless JSON round-trip).
- No direct reads of wall-clock time or randomness; inject them.
- Concise docstrings describing intent and contracts.
- No TODOs, stubs or pseudocode in merged code.

## Testing Expectations

The suite is the contract. New behaviour must be covered, and the full suite (which re-verifies all
prior subsystems) must pass. Honest quality-gate failures should be addressed, not suppressed.

## Commit and PR Hygiene

- Keep pull requests focused and reviewable.
- Write clear commit messages describing the *why*, not only the *what*.
- Reference related issues (e.g. `Closes #123`).

## Reporting Security Issues

Please do not open public issues for security vulnerabilities. Report them privately to the
maintainers via the contact address in the repository metadata.

## Code of Conduct

Participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).