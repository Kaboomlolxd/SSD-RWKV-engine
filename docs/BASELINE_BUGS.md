# Baseline bug sweep — F1-F5 tier bugs found and fixed (June 27, 2026)

While setting a baseline for F1-F5 throughput on the dev machine (Windows, Intel
iGPU host, CPU forward, `trinity_lut2_0.1b`), the bench reported F5 at **3.79
tok/s** — about 50% of the documented **8-10 tok/s** range. Six bugs were
found in the bench and preset code that, when fixed, raised F5 to **9.01 tok/s**
and made F5 ≥ F6 (resident) for the first time. This doc enumerates them.

## Summary

| # | Where | Bug | Fix | Effect on F1-F5 |
|---|-------|-----|-----|----------------|
| 1 | `bench/bench_io_ceiling.py` | No warmup, engine reloaded per sample | Build engine once, warmup internally, run samples on the same engine | F5 3.79 → 9.01 tok/s |
| 2 | `bench/bench_io_ceiling.py` | `bridge_ms` missing `compute_ms` in subtraction | Add `compute_ms` to `classified_pt` | F1-F4 bridge% 90-99% → 22-31% (real) |
| 3 | `bench/bench_io_ceiling.py` | F5 bench passed `max_layers_in_z=1` cap | `apply_promote_max_defaults` reads manifest, sets to n_block when `warm_z` | F5 promote actually promotes (was cap-1) |
| 4 | `rwkv_ssd/runtime/throughput_defaults.py` | `apply_promote_max_defaults` set `max_layers_in_z=1` even with `warm_z` | When `warm_z` and manifest known, set `max_layers_in_z = n_block` | (works because of #3) |
| 5 | `rwkv_ssd/runtime/weight_provider.py` | `_apply_fused_lut_split` popped fused entries even on `force_materialize=True` | Add `force_materialize` parameter; only pop when both fused kernel is in use AND not materializing | Unblocks F4 / F3t on tiered packs (no more `KeyError: 'blocks.3.att.key.weight'`) |
| 6 | `rwkv_ssd/runtime/weight_provider.py` | `evict_streamed_layer` did not accept `force=True` | Add `force: bool = False` keyword; only respect strict-fused-retain when `not force` | Unblocks warm disk cache write |
| 7 | `bench/_bench_profiles.py` | `--full` profile referenced undefined `FULL` constant | Add `FULL = BenchProfile(name="full", max_tokens=48, samples=3, warmup=1)` | `--full` no longer NameError |
| 8 (F-1 follow-up) | `rwkv_ssd/backends/rwkv7_forward.py` + `synthetic.py` | Prefetch was issued unconditionally for `stream_layer_cache=False`; the main thread then blocked on `begin_layer` waiting for I/O with no compute to overlap | Skip `prefetch_ahead` when `provider._stream_layer_cache` is False (the no-cache / strict-fused path); the F-1 path consumes data via the LUT blob / z skeleton, prefetch is a net loss | F1 from broken to 6.9 tok/s on 0.1B |
| 9 (F-2) | `rwkv_ssd/runtime/weight_provider.py::begin_layer` and `prefetch_entries` | Even with the F-1 skip, F2 (which *does* use prefetch) was still blocked in `begin_layer` for the in-flight prefetch's I/O time (~80 ms/layer, 17% of wall). `prefetch_entries` also blocked on `prev.result()` when submitting a new prefetch | "Best-effort" drain: only call `fut.result()` when `fut.done()`. If the worker is still mid-I/O, abandon the future — the worker's bytes are wasted but the main thread does not block. Same cost as the no-cache F-1 path on the abandoned branch | F2 should match F1 (6.9 tok/s) once the wait is gone |
| 10 (F-3) | `rwkv_ssd/runtime/throughput_defaults.py` (5 sites) | After the F-2 fix, F2 was still slower than F1 (4.3-7.7 vs 6.4-7.2 tok/s) because the F2 provider cache was too small to be useful. Multiple tier defaults hardcoded `max_provider_cache_layers=2` (`apply_bounded_stream_defaults`, `apply_bounded_fused_defaults`, `apply_partial_fused_defaults`, `apply_ssd_stream_defaults`, and the `apply_streaming_defaults` else-branch). On a 12-layer model a 2-layer LRU has a 16.7% hit rate and the management overhead (56.8 ms/tok of bridge) exceeds the savings (22.2 ms/tok of read) | Remove the hardcoded `2` from all five sites. Leave `max_provider_cache_layers` at `0` so the `resolve_max_provider_cache_layers` resolver in `provider_factory` picks the right value (full cache for n>=8 decoupled, capped for budget-limited). Users on a tight RAM budget can still override with `--max-provider-cache-layers` | F2 went from 4.3-7.7 tok/s to 6.5-8.3 tok/s; F2 is now a valid Pareto point at low RAM, beating F1 |

## Detail

### 1. Bench: cold-start on every measurement

`bench/bench_io_ceiling.py::_run_scenario` created a fresh
`InferenceEngine` and called `eng.load()` on every iteration. The outer
`_run_batch` loop called `_run_scenario` `warmup + samples` times. So:

- `warmup=0, samples=2` → engine loaded 2× per scenario, no warmup at all.
- The cold load (mmap + chatrwkv `RWKV_x070_*` lazy import) is ~3-5 s on the
 dev machine, which alone is 30-50% of a 12-token measurement on F1.

**Fix:** refactored `_run_scenario` to:

- `_build_scenario_engine(...)` → returns the loaded engine + ctx.
- `_run_one_generate(eng,...)` → runs one generate, returns `(wall, m)`.
- `_summarize_row(...)` → builds the row dict.
- `_run_scenario(...)` → builds the engine once, does warmup, then samples.

`_run_batch` was simplified to call `_run_scenario(warmup=warmup, samples=samples)`
once per scenario. Cold-start is now paid exactly once per tier.

**Also:** `eng.metrics.layers.clear()` after warmup so the row's compute/staging
are steady-state only (otherwise warmup's cost is double-counted in
`compute_ms` / `staging_ms` totals).

### 2. `bridge_ms` formula was missing `compute_ms`

Old formula in `bench_io_ceiling.py:241-252`:

```python
classified_pt = (read_ms + staging_ms + sum(L.h2d_ms for L in m.layers)) / max_tokens
bridge_ms = max(0.0, wall_ms_pt - classified_pt)
```

`compute_ms` was reported as a separate field but not subtracted from the wall
when computing the bridge. Result: bridge_pct = 90-99% on every tier — every
tier looked bridge-bound, hiding the real cost (compute).

`bench_throughput.py:88-94` had the same shape but correctly subtracted
`total_compute`. The `io_ceiling` version did not.

**Fix:** added `compute_ms` to the `classified_pt` sum with a comment
explaining the convention. After fix, bridge_pct is 22-31% on F1-F4 (real
Python overhead + per-layer kernel launch) and 0% on F5/F5s (compute + h2d
fully accounted for, the small residual is metric noise).

### 3 & 4. F5 preset's `max_layers_in_z=1` masked intent

`bench_io_ceiling.py:188` set `z_cap = 1` for any scenario with
`apply_promote_max` or `apply_stacked` (placeholder). Then
`apply_promote_max_defaults` (after my fix) sees `max_layers_in_z = 1` and
short-circuits its warm_z branch (which sets `max_layers_in_z = n_block` only
when the cap is `<= 0`). Net effect: F5 path had cap=1 the whole time, which
worked only because `warm_z` makes the LRU cap a no-op — but a future change
to `ZLayerRetention` could regress it.

**Fix:** changed `bench_io_ceiling.py:188` to pass `z_cap = 0` (let the preset
decide) and updated `apply_promote_max_defaults` to set `max_layers_in_z =
n_block` when warm_z is on. Also added `warm_z=True` to the `EngineConfig` for
F5/F5s presets — without that, the F5 scenario was silently running as F2
with cap=1 (because the `RWKV_PROMOTE_FULL_Z=1` env var only takes effect if
`config.warm_z` is also set).

### 5 & 6. Warm disk cache path silently dropped entries

The `RWKV_WARM_DISK_CACHE=auto` path calls
`weight_provider.load_layer_tensors_materialized` for each layer and writes
the result to `.decode_cache/`. This calls `_decode_from_layer_raw` with
`force_materialize=True`, which routes to `decode_lut2_layer_from_span` (the
fully-materialized decoder). But the final call to `_apply_fused_lut_split`
was unconditional and popped fused entries from the output dict, leaving the
writer with a missing-key `KeyError`.

Affected packs: any pack where some block layers are resident (e.g.
`trinity_tiered_hot3_0.1b` which puts hot layers in `shadow.bin`). The
`_warm_decode_disk_cache_if_needed` flow raised `KeyError:
'blocks.3.att.key.weight'` on the first cold layer.

**Fix:** added a `force_materialize: bool = False` keyword to
`_apply_fused_lut_split`; only pop when both `_use_fused_lut_matmul()` is on
and we are *not* materializing. Threaded `force_materialize` through
`_decode_from_layer_raw`.

After the fix, the warm disk cache writer also raised a second error:
`ManifestWeightProvider.evict_streamed_layer(force=True)` was missing the
`force` keyword. Added it: when `force=True`, the strict-fused-retain guard
is bypassed so the writer can reclaim provider RAM after serializing the
bf16 layer.

### 7. `--full` profile referenced undefined `FULL`

`bench/_bench_profiles.py:43` (resolve_profile) referenced `FULL` but only
`QUICK`, `DEFAULT`, `SYNTH_QUICK`, `SYNTH_FULL` were defined. `python -m
bench.bench_io_ceiling --full` failed with `NameError: name 'FULL' is not
defined`.

**Fix:** added `FULL = BenchProfile(name="full", max_tokens=48, samples=3,
warmup=1)`. Heavier profile; the v0.6.14 release notes call for "all F
scenarios, 48 tokens, 3 samples, load-time warm cache" but the constant was
never wired up.

### 8 (F-1 follow-up). F1 prefetch was issued unconditionally and blocked the main thread

`rwkv_ssd/backends/rwkv7_forward.py::forward_one_streaming` issued
`prefetch_ahead` whenever the provider was a `ManifestWeightProvider`,
regardless of `stream_layer_cache`. The strict-fused / no-cache path
(`stream_layer_cache=False`) does not retain the decoded tensors in the
provider cache, so the prefetched data is *only* used for the immediate
forward — there is no "next token" reuse to amortize the I/O against.

The `begin_layer` drain then blocked the main thread on `fut.result()` for
the I/O time (80-150 ms/layer on a slow SSD) with no compute to overlap.
The net effect: F1 was a *broken* tier, ~50% slower than its design
budget.

**Fix:** added `provider._stream_layer_cache` to the `prefetch_ahead`
guard in both `rwkv7_forward.py` and `synthetic.py`. The prefetch is now
only issued when the decoded tensors will be reused on the next token. F1
is back to its design budget (6.9 tok/s on the 0.1B dev pack, re-reading
from disk every layer — no overhead, no blocking).

### 9 (F-2). F2 prefetch blocked the main thread even with `stream_layer_cache=True`

After fix #8, F2 (`stream_layer_cache=True`) was *still* slower than F1
(5.7 vs 6.9 tok/s on 0.1B). The cause: `begin_layer` still called
`fut.result()` unconditionally. On a slow SSD the prefetch worker takes
80-150 ms to complete, and the main thread blocks for that whole time. The
prefetched data is then used (so `prefetch_hits > 0` on F2) but the
blocking wait negates the benefit — F1's "no cache, no wait" path ends
up faster than F2's "cache + blocking wait" path.

`prefetch_entries` had the same problem: when submitting a new prefetch, it
called `prev.result()` to drain the previous future. If the previous
worker was still running, the submission call blocked for the remaining
I/O time.

**Fix:** in both `begin_layer` and `prefetch_entries`, the drain path now
checks `fut.done()` first. If the worker is not done, the future is
abandoned (its result is GC'd; the worker's bytes are wasted) and the
main thread proceeds without blocking. The downstream
`_try_cached_layer` / `_decode_stream_entries` then fall through to the
disk-read path — same cost as the F-1 no-cache path on the abandoned
branch. When the worker *is* done (compute-heavy tier, e.g. F5 with light
I/O), the drain is a cheap dict copy and the fast path is unchanged.

**Trade-offs:**

- **Wasted I/O on the abandoned branch:** the worker thread still
 completes its read in the background; the bytes are simply not used.
 This is bounded (one prefetch per layer in flight) and the SSD read is
 cheap on the F-1 path anyway.
- **`prefetch_hits` becomes a best-effort signal:** the test
 `test_prefetch_overlap_recorded` was updated to assert the metric
 columns are present and non-negative rather than requiring a hit. The
 prefetch is no longer *guaranteed* to be done in time; it's
 opportunistic.
- **No polling / no busy-wait:** `fut.done()` is a cheap flag check
 (microseconds). The main thread does not burn CPU waiting for the
 worker.

**Test coverage:** new `tests/test_prefetch_nonblocking.py` covers:
- `test_begin_layer_does_not_block_on_pending_prefetch`: submits a
 blocking future, asserts `begin_layer` returns in < 50 ms with
 `prefetch_wait_ms == 0`.
- `test_begin_layer_drains_when_prefetch_done`: submits an already-done
 future, asserts the data is in `_prefetch_raw` after the drain.
- `test_prefetch_entries_does_not_block_on_pending`: asserts submission
 returns in < 50 ms even with the previous prefetch still in flight.
- `test_streaming_end_to_end_non_blocking_prefetch`: end-to-end smoke
 on a tiny synthetic pack.

### 10 (F-3). F2 provider cache was hardcoded to 2 layers (16.7% hit rate on 0.1B)

After the F-2 fix, F2 was still slower than F1 (4.3-7.7 vs 6.4-7.2 tok/s) on
0.1B. The blocking wait was gone, but the cache was too small to be
useful. Root cause: **five sites** in `throughput_defaults.py`
hardcoded `max_provider_cache_layers=2`:

```python
# apply_streaming_defaults (line 162)
if promote_on or n_layer < 8:
 config.max_provider_cache_layers = n_layer
else:
 # Larger model without promote: 2-layer provider LRU.
 config.max_provider_cache_layers = 2 # <-- bug

# apply_bounded_stream_defaults (line 217)
if config.max_provider_cache_layers <= 0:
 config.max_provider_cache_layers = 2 # <-- bug

# apply_bounded_fused_defaults (line 326)
config.max_provider_cache_layers = 2 # <-- bug

# apply_partial_fused_defaults (line 299)
config.max_provider_cache_layers = 2 # <-- bug

# apply_ssd_stream_defaults (line 475)
if config.max_provider_cache_layers <= 0:
 config.max_provider_cache_layers = 2 # <-- bug
```

The `resolve_max_provider_cache_layers` resolver in `provider_factory`
already does the right thing for n>=8 (returns `n_block` for decoupled,
4 for budget-limited) — but the tier defaults bypass it by hardcoding
`2`, which the resolver treats as "user said 2, respect it". On a
12-layer model a 2-layer LRU has a 16.7% hit rate and the management
overhead (56.8 ms/tok of bridge in the bench) exceeds the savings
(22.2 ms/tok of read).

**Fix:** remove the hardcoded `2` from all five sites. Leave
`max_provider_cache_layers` at `0` so the resolver picks the right
value. Users on a tight RAM budget can still override with
`--max-provider-cache-layers`. The five sites now have a comment
explaining why we leave the value at `0` (the resolver is the source
of truth).

**Test coverage** (2 new tests in `tests/test_throughput_defaults.py`):
- `test_bounded_fused_provider_cache_left_to_resolver`: asserts the F2
 tier default leaves `max_provider_cache_layers=0` and the resolver
 returns 12 for a 12-layer pack.
- `test_partial_ssd_tier_provider_cache_left_to_resolver`: same check
 for the F3 tier.

**Bench after F-3** (`ram_frontier_v9.json`, --full profile,
max_tokens=48, samples=3):

| Tier | tok/s (F-3 fix) | tok/s (F-2 fix only) | z_mb | provider_mb | Notes |
|------|-----------------|---------------------|------|-------------|-------|
| F1 | 5.49-7.20 | 6.39 | 201 | 30 | no cache (unchanged) |
| F2 | **6.49-8.28** | 4.28-7.69 | 201 | 39 | F-3 fix: resolver picks 12 for n=12 (was 2); F2 is now a valid Pareto point at low RAM, beating F1 |
| F5 | 9.08-17.38 | 13.62 | 382 | 0 | unchanged |
| Fb | 15.18-20.37 | 18.62 | 382 | 0 | unchanged |

The F2 win is modest (+1-2 tok/s) because the cache hit rate is still
bounded by the F-2 path's 2-layer z cap (the full cache stores the
small *prepared* skeleton, not the full bf16 weights). The remaining
F1-F2 vs F5-F6 gap is compute (ChatRWKV Python dispatch) and a
separate z-cap enforcement bug (4 layers in z instead of 2 for F2
with `max_layers_in_z=2`; the per-tensor path is used for those layers
instead of the faster F-1 packed path). **Fixed (Jul 2026):** auto
accuracy pins of layers 0 + n−1 are skipped when `max_layers_in_z <= 2`,
and F2 presets set `RWKV_PIN_ACCURACY_LAYERS=0`. Explicit `=1` still
forces pins.

## What still needs work

> **June 29, 2026 F-1 / F-2 / F-3 follow-up:** items C, D, E, F, and most of A
> are now closed. Fb is at 6.43 tok/s (3.4× the broken 1.9 number); F6
> records `compute_ms`; the planner no longer throttles on small models.
> F-1 (skip prefetch when `stream_layer_cache=False`) shipped and brings
> F1 from a broken state to 6.9 tok/s. F-2 (non-blocking prefetch drain)
> shipped; `prefetch_wait_ms = 0` for F2. F-3 (provider cache size bug)
> shipped; the F2 provider cache is now sized by the resolver (full for
> n>=8 decoupled) instead of the hardcoded 2. F2 is now a valid Pareto
> point at low RAM (~6.5-8.3 tok/s, beating F1's 5.5-7.2 tok/s).
> F-1 (P1.4 packed block forward) shipped and drops F1/F3 staging
> 20-27%; bigger wins on F1-F4 are gated on CUDA graphs (M6e) to drop
> the Python dispatch cost.

| # | Where | Why | Status |
|---|-------|-----|--------|
| A | F1-F4 staging 23-45 ms/tok | P1.4 packed block forward **shipped for the supported CPU fused-pack path** (CHANGELOG F-1). New early-exit path in `forward_one_streaming` calls `forward_block_packed` directly when fused LUT is active and att weights are NOT in `z` — skips per-tensor `inject_layer_into_z`. F1 60.6→43.9 ms/tok, F3 42.3→33.8 ms/tok. Remaining Python dispatch cost did not show a further safe CPU promotion; the larger next reduction is CUDA graphs (M6e). | **closed for current CPU scope** |
| B | F4 = 2.15 tok/s (worse than F3) | **Closed (F-2).** v7 bench has F4=3.19 ≈ F3=3.15. The v6 "2.15" was a single noisy run with higher compute from the strict-fused retain path; F3t is the new Pareto-best at the ~262 MB tier. | **done** |
| C | Fb = 0.40 tok/s | **Closed (F-4).** `apply_ram_budget_tier` for F5 sets `warm_z=True` + stamps `_ram_budget_tier_applied`; `apply_ram_budget_to_config` respects the flag. Fb is now 6.43 tok/s (86% of F6, 91% of F5). | **done** |
| D | F6 bridge_pct=100% | **Closed (F-5).** `engine._generate_resident` passes `self.metrics` to `generate_greedy_native(..., metrics=…)`. F6 row now reports `compute_ms_per_token=265.62` and `bridge_pct=0.0`. | **done** |
| E | F2 warm cache compression | **Closed (F-6).** `apply_bounded_fused_defaults` already sets `RWKV_DECODE_CACHE_COMPRESS=0` at `throughput_defaults.py:330`; F1 also sets it. The docstring was stale. | **done** |
| F | F1-F3 on small models | **Closed (F-3).** `apply_ram_budget_to_config` skips the `max_provider_cache_bytes` cap when `n_layer < 16`. F4/F5 already skip via the tier-applied marker. | **done** |
| F2-fix | F2 prefetch blocks main thread | **Closed (June 29, F-2).** `begin_layer` and `prefetch_entries` now check `fut.done()` and only call `fut.result()` when the worker is already done. Otherwise the future is abandoned and the main thread falls through to the disk-read path. F2 should now match F1 (6.9 tok/s) on 0.1B. | **done** |
| F3-fix | F2 provider cache too small (hardcoded 2) | **Closed (June 29, F-3).** Five sites in `throughput_defaults.py` hardcoded `max_provider_cache_layers=2`. On a 12-layer model a 2-layer LRU has a 16.7% hit rate and the management overhead exceeds the savings. Fixed by leaving the value at `0` so the `resolve_max_provider_cache_layers` resolver picks the right value (full cache for n>=8 decoupled). F2 went from 4.3-7.7 tok/s to 6.5-8.3 tok/s; F2 is now a valid Pareto point at low RAM, beating F1. | **done** |
| G | Bridge breakdown on F1-F4 | Still open — with bridge at 0% (post-fix F-1) the per-layer Python timer is less urgent; the bridge was mostly Python dispatch. CUDA graphs (M6e) is the right tool. | **deferred** |
| H | M6 GPU track | F5 ceiling on CPU is ~7-9 tok/s. Real tok/s is on GPU: M6a (Albatross / rwkv_lightning) → M6b (FLUTE fused LUT2 GEMM) → M6c (GDS layer swapper). See and. | **open** |

## What about the `fb` path? (fixed)

`bench/_ram_frontier.py` adds a `Fb auto-tier` scenario that calls
`select_ram_budget_tier(0.5)` (returns "F5") then passes `ram_budget_gb=0.5`
to the engine. `InferenceEngine.load` already applies
`apply_ram_budget_tier` then `apply_ram_budget_to_config` with the
`_ram_budget_tier_applied` guard.

The remaining hole was `apply_low_ram_defaults`: when `ram_budget_gb` was
set it called **only** `apply_ram_budget_to_config` (no tier preset) and
forced `RWKV_PROMOTE_FULL_Z=0`, so low-ram + budget paths never got F5
warm-z promote. That path now mirrors the engine: select tier → apply
tier → planner (respects tier marker).

## Bench profile audit

| Profile | max_tokens | samples | warmup | skip_warm_disk_cache | frontier_ids |
|---------|-----------|---------|--------|----------------------|---------------|
| `QUICK` (default for --quick) | 12 | 1 | 0 | True | F1, F3, F5 |
| `DEFAULT` (default) | 24 | 1 | 1 | False | all |
| `FULL` (--full) | 48 | 3 | 1 | False | all |
| `SYNTH_QUICK` | 8 | 1 | 0 | n/a | n/a |
| `SYNTH_FULL` | 32 | 2 | 0 | n/a | n/a |

QUICK with `warmup=0` is fine for smoke tests but should not be reported as
production numbers — the new `_run_scenario` reuses the engine for samples
so QUICK is now cold-start (engine build cost) + 1 sample. Use DEFAULT or FULL
for any tok/s claim.

## See also

- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — current state and focus; this file is the *what broke* companion.
- **[`BENCH.md`](BENCH.md)** — three active benches and the methodology behind the numbers cited here.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog that explains *what* each fix is; this file explains *why* each fix was needed.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — multi-SSD sharded pack, mmap huge pages, and other SSD knobs that complement the in-flight fixes.
- **[`CHANGELOG.md`](../CHANGELOG.md)** — every fix is also a release entry; cross-check the date stamps.
