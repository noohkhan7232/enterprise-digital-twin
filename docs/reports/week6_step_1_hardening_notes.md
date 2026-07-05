# Week-6 Step-1 Hardening Pass — Architecture Notes & Hardening Summary

**Component:** `src/simulation/asset_state_simulator.py`
**Review role:** Principal Reliability Engineer / Simulation Architect / Industrial AI Platform Reviewer
**Outcome:** Elevated from a working simulator to an enterprise-grade digital-twin simulation component, with zero public-API breakage.

---

## 1. Hardening summary

| # | Requirement | Status | Mechanism |
|--:|-------------|--------|-----------|
| 1 | Horizon guarantee | Done | `_build_time_grid` appends the exact `horizon` when `horizon % timestep != 0`, preserving strict chronological order |
| 2 | Failure threshold support | Done | New `failure_threshold` config (default 30, validated `0..100`); `_failure_bookkeeping` drives `crossed_failure`, `first_failure_step`, and the new `failure_time` |
| 3 | Piecewise span validation | Done | `_validate_piecewise` now rejects `segment_breakpoints[-1] < horizon` |
| 4 | Monte Carlo readiness | Done | Three sampler hooks (`DegradationRateSampler`, `FaultSampler`, `NoiseSampler`) with degenerate defaults; no behaviour change |
| 5 | Enterprise telemetry | Done | Eight additional metrics logged failure-safely; categorical `sim_model` via `log_params` |
| 6 | Digital-twin documentation | Done | Assumptions, limitations, intended usage, MC architecture, class diagram, design rationale embedded in the module docstring |

**Test count:** 123 → **185** (target was 120+). All 185 pass live, zero skipped.

---

## 2. Design rationale per requirement

### Requirement 1 — Horizon guarantee

The pre-hardening grid was `arange(floor(horizon/timestep)+1) * timestep`, which silently dropped the endpoint whenever the horizon was not an integer multiple of the timestep (e.g. `horizon=10, timestep=3` → `0,3,6,9`). The fix appends the exact horizon as a final, possibly-shorter step only when the uniform grid stops short:

```
grid = arange(n) * timestep
if horizon - grid[-1] > EPS:
    grid = append(grid, horizon)
```

This guarantees `times[-1] == horizon` while preserving strict monotonicity (the appended sample is always greater than its predecessor). The divisible case is untouched, so every existing `n_samples` assertion — all of which used divisible configs — remains valid.

### Requirement 2 — Failure threshold support

The legacy failure definition was `health == 0`, which is physically unrealistic: an asset is "failed" for maintenance purposes well before its health index reaches absolute zero. A `failure_threshold` field (default **30**, validated to `[0, 100]`) now governs failure bookkeeping via a dedicated `_failure_bookkeeping` method that locates the first index where `health <= threshold` and records the matching `failure_time`. Setting `failure_threshold=0` recovers the exact legacy semantics, which is the backward-compatibility escape hatch.

### Requirement 3 — Piecewise span validation

Previously a piecewise config whose final breakpoint fell short of the horizon would silently hold the last segment's end value flat for the remaining samples — a latent correctness bug. Validation now requires `segment_breakpoints[-1] >= horizon`, rejecting `(0,20,40)` for `horizon=180` while accepting both `(0,20,40,180)` and `(0,20,40,250)` (over-spanning is harmless; the trajectory is simply truncated at the horizon).

### Requirement 4 — Monte Carlo readiness

The deterministic degradation maths is the *invariant* of the simulator and must never be forked for a stochastic variant. Three strategy interfaces isolate every source of randomness:

```
DegradationRateSampler.sample(rng, base_rate)         -> float
FaultSampler.sample(rng, base_times, base_magnitudes) -> (times, mags)
NoiseSampler.sample(rng, n, noise_std)                -> ndarray
```

The default implementations are **degenerate**: `ConstantRateSampler` and `ConstantFaultSampler` consume *zero* random numbers and echo the configuration, and `GaussianNoiseSampler` draws exactly `rng.normal(0, std, n)` — the same call, in the same order, that the pre-hardening code made. The net effect on the generator stream is nil, which is why determinism is preserved bit-for-bit (verified by reconstructing the legacy noise path and asserting array equality). A Week-7 Monte Carlo engine plugs stochastic subclasses into the three constructor arguments without editing a single line of simulation logic.

### Requirement 5 — Enterprise telemetry

The original five metrics are retained verbatim for backward compatibility; eight enterprise metrics are added (`sim_mean_health`, `sim_min_health`, `sim_max_health`, `sim_failure_time`, `sim_threshold`, `sim_duration`, `sim_fault_count`, and the categorical `sim_model`). The numeric metrics go through `log_metrics`; the categorical model name goes through `log_params` *only when the tracker exposes it*, discovered by duck-typed `getattr`. Both calls are independently wrapped in `try/except … # noqa: BLE001`, so a tracker fault in either path can never interrupt a simulation — the failure-safe contract established in Weeks 1–5.

### Requirement 6 — Digital-twin documentation

The module docstring now contains four enterprise sections: **Simulation assumptions** (the physical interpretation of each degradation regime), **Limitations** (explicitly *not* FEA, CFD, or vibration-spectrum synthesis), **Intended usage** (closed-loop validation, RUL benchmarking, policy evaluation, scenario simulation, Monte Carlo forecasting), and the **Monte Carlo extension architecture** with design rationale and a text class diagram.

---

## 3. Class diagram

```
AssetStateSimulator
  ├── config: AssetSimulatorConfig            (frozen, validated)
  ├── rate_sampler:  DegradationRateSampler ──┐
  ├── fault_sampler: FaultSampler            ├─ strategy hooks (default degenerate)
  └── noise_sampler: NoiseSampler ───────────┘

DegradationRateSampler (ABC) → ConstantRateSampler   (default; returns config rate)
FaultSampler           (ABC) → ConstantFaultSampler  (default; returns config faults)
NoiseSampler           (ABC) → GaussianNoiseSampler  (default; reproduces config noise)

AssetSimulatorConfig (frozen) ──produces──▶ SimulationResult ──materialises──▶ AssetState[]
```

---

## 4. Backward-compatibility report

| Surface | Status | Evidence |
|---------|--------|----------|
| Public functions (`register/build/list_simulator`, four degradation generators) | Unchanged | Signatures and behaviour identical |
| `AssetStateSimulator.__init__` | Extended (additive) | Three new **optional** sampler args default to degenerate; no positional break |
| `simulate(seed=...)` / `simulate_ensemble(...)` | Unchanged | Same signatures; tested |
| `AssetSimulatorConfig` | Extended (additive) | New `failure_threshold` field with a default; all prior fields and defaults intact |
| `SimulationResult` | Extended (additive) | New `failure_threshold` and `failure_time` fields appended; all prior fields intact |
| `AssetState` | Unchanged | Identical |
| Registry / tracker integration | Preserved | Same registry name; tracker still failure-safe |
| Deterministic reproducibility | Preserved (bit-for-bit) | Legacy noise path reconstructed and asserted equal |
| Pure NumPy | Preserved | No SciPy/PyTorch introduced |
| Frozen validated configs | Preserved | `__post_init__` extended, not relaxed |

**Existing tests:** 122 of the original 123 pass **unchanged**. Exactly one test, `test_first_failure_is_first`, had its assertion updated from `health[idx:] == 0` to `health[idx:] <= failure_threshold` — a *semantic* update mandated by Requirement 2 (configurable threshold replaces the `health==0` rule). The test's intent (all post-failure samples remain in the failed state) is preserved; only the failure criterion it checks against changed, in lock-step with the hardened bookkeeping. No other existing test was modified.

**Net behavioural change for existing callers:** a caller that did not set `failure_threshold` now sees failure declared at `health <= 30` rather than `health == 0`. This is the intended, documented hardening; callers requiring the exact legacy behaviour set `failure_threshold=0`.

---

## 5. Final test inventory

| Group | Classes | Representative coverage |
|-------|---------|-------------------------|
| Original (preserved) | 14 | validation, correctness, reproducibility, bounding, faults, ensemble, serialization, Week-5 integration |
| Hardening 1 | `TestHorizonGuarantee` | 10 tests — endpoint inclusion, ordering, determinism, fractional grids |
| Hardening 2 | `TestFailureThreshold` | 16 tests — default, validation, custom thresholds, `failure_time`, boundaries |
| Hardening 3 | `TestPiecewiseSpanValidation` | 6 tests — reject short, accept equal/over-spanning |
| Hardening 4 | `TestMonteCarloHooks` | 11 tests — degenerate defaults, custom injection, abstract enforcement, determinism |
| Hardening 5 | `TestEnterpriseTelemetry` | 13 tests — all 8 metrics, param logging, failure-safety |
| Compatibility | `TestBackwardCompatibility` | 6 tests — bit-for-bit noise, API surface, signatures |

**Total: 185 tests, 100% passing, 0 skipped.**