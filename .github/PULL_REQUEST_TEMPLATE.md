## Description

Summarise the change and the motivation behind it. Link any related issues (e.g. `Closes #123`).

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Documentation update
- [ ] Refactor (no functional change)
- [ ] Breaking change (please justify; the platform favours additive, backward-compatible change)

## Affected subsystem(s)

List the layers or subsystems touched by this change.

## Engineering checklist

- [ ] Change is **additive**; locked prior work is not modified (or modification is explicitly
      justified and scoped)
- [ ] Integration is **by composition**; no shared mutable state introduced across subsystems
- [ ] Public types remain **immutable** and serialisable where applicable
- [ ] Behaviour is **deterministic** (time and identity injected, outputs ordered/rounded)
- [ ] Code is fully **type-annotated**
- [ ] No TODOs, stubs or placeholder logic

## Testing checklist

- [ ] Added tests at the level where the behaviour is introduced
- [ ] Added a determinism test where applicable
- [ ] Full suite passes locally: `PYTHONPATH=src:scripts pytest tests/ -q`
- [ ] CI/CD quality gates pass (honest gate failures addressed, not suppressed)

## Documentation

- [ ] Updated relevant documentation (guides, FAQ, architecture overview) if behaviour or interfaces
      changed

## Notes for reviewers

Anything reviewers should focus on, or known limitations.