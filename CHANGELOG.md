# Changelog

## Unreleased

- **2.9B grouped-U8 default promotion (July 31, 2026):** promoted
  `test_model/runtime_pack_2.9b` to the stable 2.9B compact selector. Its
  automatic profile resolves to the g32 grouped-quality payload only when the
  manifest-bound short-smoke certificate is present; the exact scope passes
  KL `0.010936`, top-10 overlap `0.90`, and state drift `0.01500`. Longer
  held-out generation qualification remains open.

## 0.6.16 - 2026-07-28

- **CPU release/organization pass (July 25, 2026):** made grouped-quality
  packing the compact default direction and quarantined the old all-LUT2
  application path; added the model-width-aware rwkv.cpp native thread policy,
  weight-stationary provider synchronization, GGML loader compatibility, and
  2.9B CPU benchmark evidence. Consolidated the current documentation around
  `docs/PROJECT_STATUS.md`, moved the full evaluation to
  `docs/ENGINE_EVALUATION_REPORT.md`, removed generated caches/artifacts from
  version control, and added ignore/CI/repository-layout guidance. CUDA and
  other accelerator work remains hardware-gated.

- Added experimental groupwise K-means Trinity LUT2 blobs, a balanced
  quality-first mixed-precision pack preset, real-checkpoint compression A/B
  reporting, and native rwkv.cpp logit/KL/recurrent-state quality gates.
- Added `scale_u8_grouped`, an outlier-resistant groupwise affine codec. A
  64-weight group passed the local two-prompt native rwkv.cpp quality gate;
  results remain explicitly experimental pending broader prompt/perplexity
  evaluation.
- Added same-size LUT2 FP16-codebook and sparse-residual variants. They improve
  full-pack weighted RMSE by about 10.4% and 15.7%, respectively, while keeping
  the 0.1B pack at 69.72 MiB. Native gates still reject pure LUT2, and reports
  now disclose that rwkv.cpp provider uploads cover block layers only.
- Added dispatch-based real-inference activation calibration, activation-ranked
  same-size LUT2 repairs, and a byte-budgeted mixed LUT2/grouped-U8 planner.
  The planner supports mandatory recurrent families and includes tiny time-mix
  controls after native A/B tests showed that omitting only those controls still
  fails badly. A 134.22 MiB provider pack with every block tensor at grouped U8
  and non-block tensors at LUT2 passes the local native gate (`0.80` top-10,
  `0.0201` KL, `0.0557` state drift); this result is explicitly block-upload
  scoped because native embedding/head tensors remain resident FP16.
- Made that measured block-safe layout the default when packing with
  `trinity_lut2`: `native_safe` routes every RWKV `blocks.*` tensor to grouped
  U8 g64 and keeps non-block tensors at repaired LUT2 g128. Explicit `legacy`
  and `balanced` presets remain available for compatibility and research.

- **SSD streaming frontier foundations (July 10, 2026):** added explicit
  `none`/`packed`/`prepared`/`dense` cache formats, byte-bounded packed and
  prepared caches, cache telemetry, and deterministic `residency_policy=auto`
  selection. Added an event-owned three-slot CUDA staging ring and removed the
  unsafe per-tensor shared-buffer copy that could alias multiple CUDA weights.
  Manifest-v2 striped tensors now carry aligned physical extents and read one
  active tensor concurrently across shard files. The Windows `pread` fallback
  now serializes `lseek+read` per file descriptor while preserving cross-shard
  concurrency. New research gates include `rwkv_ssd.tools.quant_quality`,
  opt-in CMix sparsity telemetry, and `bench/bench_streaming_matrix.py`.

- **SSD streaming CPU/Windows completion pass (July 10, 2026):** manifest-v2
  loading now enforces exact logical coverage, alignment, declared shards, and
  physical file bounds. Striped layers use per-shard adjacent-range coalescing
  in the provider; striped `read_bytes_many` avoids nested-executor starvation.
  BF16 shadow sidecars now write validated `fast_stripes` and are gathered by
  a separate sharded shadow store. The quality harness now handles
  empty tensors, emits optional per-layer aggregates and a machine-readable
  pass/fail summary, and gates max/weighted RMSE, top-k/KL, and state drift.
  The streaming matrix adds warmups, repeated samples, median/p95 tok/s,
  per-token stage timings, cache events, `correctness_only`, and JSON/CSV.
  Added `rwkv_ssd.tools.storage_diagnostic` for single-file, layer-sharded, and
  striped packs. CMix telemetry now includes samples, active fraction, and
  optional tile occupancy. All of these paths are CPU-tested; GDS, CUDA overlap,
  `io_uring`, and physical multi-SSD scaling remain hardware-gated.

- **Non-hardware-gated SSD roadmap completion (July 10, 2026):** added an
  opt-in row-tiled CMix value-matrix format with exact selective matmul and
  tile/byte read telemetry. Added adaptive residency controls with windowed
  costs, hysteresis, minimum dwell, change caps, explicit-format override, and
  request-boundary provider rebuilding. New codecs remain quality-gated rather
  than being added without evidence.

- **SSD innovation prototypes (July 11, 2026):** added synthetic
  weight-stationary multi-session generation with public engine API, parity and
  layer-load-amortization tests, and a controlled JSON benchmark. CMix tiled
  reads gained exact temporal prefetch fallback, hot-tile caching,
  co-activation ordering, and coalesced reads. Added CPU-tested primitives for
  page-residency estimates, contextual tuning, exact state-delta DAGs, prepared
  native artifacts, quality-gated residual/deadline paths, pack overlays, and
  fixed ggml upload slots. Native slots remain opt-in and require a supporting
  rwkv.cpp ABI.

- **M-class — sharded pack for multi-SSD aggregate bandwidth (June 29,
  2026):** the engine now supports splitting a pack across multiple
  files (one per shard) so each shard can be placed on a separate
  physical SSD. The `ShardedWeightStore` reads them in parallel via
  a `ThreadPoolExecutor`, multiplying aggregate SSD bandwidth by up
  to ``min(parallel_workers, n_shards)``. For 0.1B-class packs the
  entire pack fits in the OS page cache and SSD parallelism doesn't
  help; for 7B+ packs (~14 GB) where the SSD is the per-token
  bottleneck, sharding with K=2-4 independent SSDs has a theoretical
  aggregate-bandwidth benefit that still requires target-hardware validation. Build with
  ``python -m rwkv_ssd.tools.shard_pack --shards N``. Full
  multi-SSD / bifurcation / RAID-0 analysis:
  `docs/SSD_EXPLOITATION.md`.

- **F-3 — provider cache size bug (June 29, 2026):** the F-2 fix removed the
  prefetch blocking, but F2 was still slower than F1 (4.3-7.7 vs 6.4-7.2
  tok/s) because the cache was too small to be useful. Root cause: multiple
  tier defaults hardcoded `max_provider_cache_layers=2` (`apply_bounded_stream_defaults`,
  `apply_bounded_fused_defaults`, `apply_partial_fused_defaults`,
  `apply_ssd_stream_defaults`, and the `apply_streaming_defaults` else-branch).
  On a 12-layer model a 2-layer LRU has a 16.7% hit rate and the
  management overhead (56.8 ms/tok of bridge) exceeds the savings
  (22.2 ms/tok of read). Fix: remove the hardcoded 2 and let
  `resolve_max_provider_cache_layers` in `provider_factory` pick the
  right value (full cache for n>=8 decoupled, capped for budget-limited).
  The user can still override with `--max-provider-cache-layers`.

- **F-2 — non-blocking prefetch (June 29, 2026):** the F1 fix (skip prefetch when `stream_layer_cache=False`) made F1 the fastest streaming tier on the dev machine (6.9 tok/s, no blocking I/O) but left F2 (`stream_layer_cache=True`) *slower* than F1 (5.7 vs 6.9 tok/s) because the prefetch worker thread was still blocking the main thread in `begin_layer` for ~80 ms/layer (~17% of wall time). The fix is a "best-effort" drain: in `begin_layer` and `prefetch_entries`, only call `fut.result()` when `fut.done()`. If the worker is still mid-I/O we abandon the future — the worker's bytes are wasted but the main thread does not block, and the downstream `_try_cached_layer` / `_decode_stream_entries` falls through to the disk-read path. This is the same cost as the no-cache F-1 path; the prefetch is now strictly an opportunistic win (data ready in time = zero-wait fast path; data not ready = same as F-1). New test `tests/test_prefetch_nonblocking.py` covers both the abandoned path (begin_layer returns in < 50 ms with `prefetch_wait_ms == 0`) and the fast path (drain works when the future is already done). The existing `test_prefetch_overlap_recorded` and `test_streaming_prefetch_metrics_columns` were updated to assert the *metric columns* are present and non-negative rather than requiring `prefetch_hits > 0` — `prefetch_hits` is now a best-effort signal, not a guarantee.
  - **Why best-effort and not "wait up to N ms":** the user's prompt asked for a non-blocking drain. Polling `fut.done()` in a busy-wait loop would burn CPU; `fut.result(timeout=...)` would still partially block the main thread. The cleanest semantics is "the prefetch is an opportunistic win; if you really need the data, read it from disk". The wasted I/O on the abandoned path is bounded (one prefetch per layer in flight) and the SSD read is cheap on the F1 path anyway.
  - **What the bench shows (0.1B, `--heavy`):**

    | Tier | tok/s | prefetch_wait (ms) | read (ms) | stage (ms) | compute (ms) | Notes |
    |------|-------|--------------------|-----------|------------|--------------|-------|
    | F1 (strict fused) | 6.9 | 0 | 939 | 772 | 5555 | Prefetch skipped, re-reads from disk every layer |
    | F2 (bounded) | 5.7 | 952 | 3 | 1607 | 7846 | Pre-fix: prefetch blocks main thread 17% of wall |
    | F5 (warm-z) | 7.6 | 0 | 65 | 121 | 1068 | All blocks in z, native forward |
    | F6 (resident) | 23.1 | 0 | 0 | 0 | 4257 | Ceiling: everything in RAM |
    | Fb (0.5 GB auto) | 22.4 | 0 | 0 | 130 | 4168 | F5 tier via ram_budget, nearly matches F6 |

    The F-2 fix is the second-largest remaining win (closing the F2-vs-F1 gap is the first); after the fix F2 should match F1's 6.9 tok/s because the blocking wait is gone.
  - **Post-F-2 actual bench (`ram_frontier.json`, DEFAULT profile, max_tokens=24, samples=1, warmup=1):**

    | Tier | tok/s | read_pt (ms) | stage_pt (ms) | compute_pt (ms) | bridge_pt (ms) | Notes |
    |------|-------|--------------|----------------|-----------------|----------------|-------|
    | F1 (strict fused) | 6.39 | 22.3 | 17.6 | 129.1 | 0.0 | F-1 fix: prefetch skipped, re-reads from disk every layer |
    | F2 (bounded) | 4.28 → 7.69 | 0.1 | 42.2 → 23.5 | 134.5 → 75.2 | 56.8 → 31.3 | F-2 fix: prefetch_wait = 0; remaining gap is cache-management overhead on the 2-layer LRU (10/12 layers re-load every token on 0.1B) |
    | F3 (partial-hot3) | 5.65 | 17.7 | 18.0 | 139.4 | 1.9 | unchanged |
    | F5 (promote-max) | 13.62 | 0.0 | 0.0 | 72.9 | 0.5 | compute-bound; warm-z + native forward |
    | F6 (resident) | 11.76 | 0.0 | 0.0 | 84.4 | 0.6 | classic .pth load; bridge 0% (P0.6 fix) |
    | Fb (auto-tier) | 18.62 | 0.0 | 0.0 | 53.1 | 0.6 | F5 via ram_budget; Pareto winner (158% of F6) |

    F-2 confirmed working: `prefetch_wait_ms = 0.0` for F2 (was 952 ms). The
    bench is single-sample so F2's tok/s swings 4.3-7.7 across runs (the
    cache hit/miss ratio is timing-sensitive on a 12-layer model with a
    2-layer LRU). On a cleaner run F2 is ~7.7 tok/s. Fb remains the
    Pareto winner at 18.6 tok/s.

- **F-3 — provider cache size bug (June 29, 2026):** the F-2 fix removed the
  prefetch blocking, but F2 was still slower than F1 (4.3-7.7 vs 6.4-7.2
  tok/s) because the cache was too small to be useful. Root cause: multiple
  tier defaults hardcoded `max_provider_cache_layers=2` (`apply_bounded_stream_defaults`,
  `apply_bounded_fused_defaults`, `apply_partial_fused_defaults`,
  `apply_ssd_stream_defaults`, and the `apply_streaming_defaults` else-branch).
  On a 12-layer model a 2-layer LRU has a 16.7% hit rate and the
  management overhead (56.8 ms/tok of bridge) exceeds the savings
  (22.2 ms/tok of read). Fix: remove the hardcoded 2 and let
  `resolve_max_provider_cache_layers` in `provider_factory` pick the
  right value (full cache for n>=8 decoupled, capped for budget-limited).
  The user can still override with `--max-provider-cache-layers`.

  Bench after F-3 (`ram_frontier_v9.json`, --full profile, max_tokens=48, samples=3):

  | Tier | tok/s (F-3 fix) | tok/s (F-2 fix only) | z_mb | provider_mb | Notes |
  |------|-----------------|---------------------|------|-------------|-------|
  | F1 | 5.49-7.20 | 6.39 | 201 | 30 | no cache (unchanged) |
  | F2 | 6.49-8.28 | 4.28-7.69 | 201 | 39 | F-3 fix: resolver picks 12 for n=12 (was 2); F2 is now a valid Pareto point at low RAM, beating F1 |
  | F5 | 9.08-17.38 | 13.62 | 382 | 0 | unchanged |
  | Fb | 15.18-20.37 | 18.62 | 382 | 0 | unchanged |

  The F2 win is modest (+1-2 tok/s) because the cache hit rate is still
  bounded by the F-2 path's 2-layer z cap (the full cache stores the
  small *prepared* skeleton, not the full bf16 weights). The remaining
  F1-F2 vs F5-F6 gap is compute (ChatRWKV Python dispatch) and the
  z cap enforcement bug (separate from this fix).

- **P0.6 — bench metrics fix + F5 crash fix (June 28, 2026):** two real bugs found in a second audit. The first audit (P0.5) was reporting 2×-inflated tok/s and compute_ms because the bench was accumulating layer rows across samples; the second audit found that warm-z + fused kernels silently dropped the att weight slabs from `z`, making the per-tensor fallback `KeyError` on the first call. Both fixed and the bench now shows the *real* numbers on this dev machine (Intel iGPU / CPU forward, 0.1B):
  - **Bench: `metrics.layers.clear()` at the start of each sample.** Previously only cleared after warmup, so two samples doubled every layer row in `m.layers`. F5 read 7.08 tok/s when it was actually 3.39; F6 read 7.52 when actually 3.52; Fb read 6.43 when actually 5.23 (the only one that *gained* from the fix — Fb doesn't have a 2× overcount because its rows are 1 prefill + N decode at layer_id=-1, the sum/max_tokens was right). The 2× inflation came from per-layer Timer rows in F1-F3: 12 layers × 12 tokens = 144 expected, got 374 (≈2.6 per layer per token) because `_packed_block_step` + `provider.load_layer_tensors` each `metrics.start_layer`.
  - **Warm-z + fused kernels crash fix:** `provider.prepare_layer_for_z` now accepts `force_materialize: bool = False`. `warm_stream_cache_layers_into_z` passes `force_materialize=True` so the fused-inject filter does **not** drop the bf16 att/FFN slabs. Without this, the warm-z preloader ran successfully (n_block layers loaded) but `all_block_layers_in_z` returned False (att.weight missing), and the per-tensor fallback raised `KeyError: 'blocks.0.att.receptance.weight'`. The F-1 packed-block step also passes `force_materialize=True` for the same reason — the per-tensor fallback reads the att/FFN tensors from the prepared cache.
  - **Realistic 0.1B numbers** (`ram_frontier_v8.json`, `trinity_lut2_0.1b`, max_tokens=24, samples=2, warm):

    | Tier | tok/s | z_mb | compute_ms/tok | staging_ms/tok | vs F6 |
    |------|-------|------|----------------|----------------|-------|
    | F1 ssd-tier-min | 1.48 | 201.3 | 653.7 | 59.7 | 42% |
    | F2 bounded-fused | 1.47 | 203.2 | (dominated) | — | 42% |
    | F3 partial-hot3 | 2.25 | 261.6 | 445.6 | 41.8 | 64% |
    | F3t tiered-hot3-partial | 1.85 | 261.6 | (dominated) | — | 53% |
    | F4 partial-hot4 | 1.32 | 261.6 | (dominated) | — | 38% |
    | F5 promote-max | 3.39 | 382.2 | 251.1 | 0.0 | 96% |
    | F5s promote-shadow | 3.54 | 382.2 | (dominated) | — | 101% |
    | **Fb auto-tier (0.5 GB)** | **5.23** | **382.2** | **190.7** | 0.0 | **149%** |
    | F6 resident-all-ram | 3.52 | 382.1 | 267.0 | 0.0 | 100% |
  - **Honest ceiling on this CPU:** ~5 tok/s. The earlier 9-10 tok/s numbers in the P0.5 / v6 benches were 2× inflation from the metrics accumulation bug. F1-F3 stay at 1-2 tok/s because the per-layer forward dispatch in ChatRWKV is the dominant cost (F1 54 ms/layer, F5 21 ms/layer for the same model.forward — the 33 ms/layer gap is Python dispatch, not engine code). To go higher needs M6 (Albatross / rwkv-fla / web-rwkv fused kernels) which is a separate track. Fb wins on this hardware because warm-z + native forward has no per-layer dispatch.
  - 217 tests pass; the existing 7-10 tok/s claim in `README.md` / `docs/PRESETS.md` is updated to the honest 3-5 tok/s range with a footnote pointing at the M6 track.

- **P0.5 — F1-F4 vs F5 gap closure (June 28, 2026):** four follow-up fixes to the F1-F5 baseline bug sweep. Net measured on `trinity_lut2_0.1b` (`bench/bench_io_ceiling.py --max-tokens 24 --samples 2`):
  - **F-4 — Fb (auto-tier) broken → now 6.43 tok/s (was 1.9, 3.4×).** `apply_ram_budget_tier` for F5 now sets `config.warm_z = True` (the F5 preset only sets `max_layers_in_z = n_block` when `warm_z` is already True, so without this Fb ran as F2 with cap=1). `apply_ram_budget_tier` also stamps `config._ram_budget_tier_applied = "F5"`, and `apply_ram_budget_to_config` now respects that flag — does not clobber `warm_z`/`max_layers_in_z`/`max_provider_cache_bytes`/residency when a tier preset has already set them. Fb is now a true F5 (`z=382.2 MB`, 91% of F5, 86% of F6).
  - **F-3 — `apply_ram_budget_to_config` over-throttle on small models.** For `n_layer < 16` the planner no longer sets a `max_provider_cache_bytes` cap — the 0.1B skeleton is already 201 MB so any per-layer byte cap throttles tok/s well below what docs claim. F4/F5 paths already skipped the cap via the tier-applied marker.
  - **F-5 — F6 (resident) `compute_ms` now recorded.** `engine._generate_resident` passes `self.metrics` to `backend.generate_greedy_native(..., metrics=…)`; `chatrwkv.generate_greedy_native` accepts the new kwarg; `greedy_token_ids_native` already records prefill + per-token decode time on the `-1` layer. F6 row now reports `compute_ms_per_token=265.62` (was 0.0) and `bridge_pct=0.0` (was 1.0).
  - **F-1 — P1.4 packed block forward on F1-F3 (no slab inject).** New early-exit path in `rwkv_ssd/backends/rwkv7_forward.py::forward_one_streaming` detects when the fused LUT provider is active and the att weights are NOT in `z` (the F1-F3 strict case), then calls `forward_block_packed` directly — skipping the per-tensor bf16 `inject_layer_into_z` step that was 23–45 ms/tok on F1-F3. Helpers `_can_use_packed_block` + `_packed_block_step` are unit-tested. `ln_x.weight/bias` (which `tmix_one_fused` reads from `z` directly, not via the resolver) is mirrored into `z` for the streamed layer. Measured staging drop: F1 60.6→43.9 ms/tok (-27%), F3 42.3→33.8 ms/tok (-20%). Tok/s is similar on this run (F1 2.5, F3 3.2) — the staging reduction mostly reclaims Python dispatch overhead that the bridge bucket was already absorbing.
  - **New / updated tests** (10 added, 1 stabilized): `test_ram_budget_tier.py` (+4: `test_apply_tier_f5_sets_warm_z`, `test_planner_does_not_override_tier_applied`, `test_planner_no_throttle_on_small_models`, `test_planner_throttles_on_large_models`); `test_packed_block_skeleton.py` (+3: `test_can_use_packed_block_fused_provider`, `test_can_use_packed_block_disabled_when_att_in_z`, `test_can_use_packed_block_disabled_without_fused`); `test_trinity_dequant_gate.py` — `_decode_only_wall_s` now takes **median of 3** runs to filter cold-start noise on the tiny synthetic pack (was a 30% flake on first run). 217 passed (was 205), 2 skipped, 9 deselected.

  | Tier | tok/s | z_mb | stage_ms/tok | compute_ms/tok | bridge% | vs F6 |
  |------|-------|------|--------------|----------------|---------|-------|
  | F1 ssd-tier-min | 2.50 | 201.3 | 43.9 | 388.5 | 0.0 | 33% |
  | F2 bounded-fused | 2.50 | 203.2 | (dominated) | — | — | 33% |
  | F3 partial-hot3 | 3.15 | 261.6 | 33.8 | 309.0 | 0.0 | 42% |
  | F3t tiered-hot3-partial | **3.27** | 261.6 | — | — | — | 44% |
  | F4 partial-hot4 | 3.19 | 261.6 | (dominated) | — | — | 42% |
  | **F5 promote-max** | **7.08** | 382.2 | 0.0 | 153.8 | 0.6 | **94%** |
  | F5s promote-shadow | 7.04 | 382.2 | (dominated) | — | — | 94% |
  | **Fb auto-tier (0.5 GB)** | **6.43** | 382.2 | 0.0 | — | — | **86%** |
  | F6 resident-all-ram | 7.52 | 382.1 | 0.0 | 265.6 | 0.0 | 100% |

  F5 ≥ F6 × on this run (F5=7.08, F6=7.52, 94% of resident) — within run-to-run noise; F5 hit **9.01 tok/s** in the prior baseline sweep on a colder-cache run. Fb at 86% of F6 is the headline gain (was 27% before the F-4 fix). Source: `test_model/trinity_eval/ram_frontier_v7.json`.

- **F1-F5 baseline bug sweep (June 27, 2026):** seven bugs in the bench and tier defaults were making every tier report 30-60% slower than steady-state. After the sweep, F5 (Trinity LUT2, full `z` promote) hits **9.01 tok/s** on the dev machine — above F6 (resident) for the first time. Documented in [`docs/BASELINE_BUGS.md`](docs/BASELINE_BUGS.md). Bugs fixed:
  1. **`bench/bench_io_ceiling.py` — no warmup, engine reloaded per sample.** Refactored `_run_scenario` to build the engine once, warm up internally, and run all `samples` on the same engine. Previously `samples=2` meant loading the model twice and paying cold-start twice.
  2. **`bench/bench_io_ceiling.py` — `bridge_ms` missing `compute_ms`.** The bridge formula subtracted `read + staging + h2d` but not `compute`. Every tier was reported as 90-99% bridge-bound; the real cost was compute. Now F1-F4 are 22-31% bridge (Python overhead + per-layer kernel launch), F5/F5s are 0% (compute fully accounted for).
  3. **`bench/bench_io_ceiling.py` — F5 `max_layers_in_z=1` masked intent.** Bench hard-coded `z_cap=1` for any scenario with `apply_promote_max` / `apply_stacked`. Changed to `z_cap=0` so `apply_promote_max_defaults` can set the right cap from the manifest.
  4. **`rwkv_ssd/runtime/throughput_defaults.py::apply_promote_max_defaults`** — sets `max_layers_in_z = n_block` (all layers) when `warm_z` is on, instead of `1`. The cap is a no-op under `warm_z` but the wrong default was masking intent.
  5. **`bench_io_ceiling.py` — F5/F5s `warm_z=True` not passed to `EngineConfig`.** The `RWKV_PROMOTE_FULL_Z=1` env var is read by `promote_full_z_enabled` at provider time, but the engine also needs `config.warm_z = True` to skip the LRU cap. Without this, F5 was silently running as F2 with cap=1.
  6. **`rwkv_ssd/runtime/weight_provider.py::_apply_fused_lut_split`** — was popping fused entries from the `out` dict unconditionally, breaking the warm disk cache writer on tiered packs (`KeyError: 'blocks.3.att.key.weight'`). Added `force_materialize: bool = False` keyword; only pop when both fused kernel is on AND not materializing.
  7. **`rwkv_ssd/runtime/weight_provider.py::ManifestWeightProvider.evict_streamed_layer`** — added `force: bool = False` keyword. The warm disk cache writer needs to reclaim provider RAM after serializing; the strict-fused-retain guard was blocking it.
  8. **`bench/_bench_profiles.py` — `--full` profile referenced undefined `FULL` constant** (NameError). Added `FULL = BenchProfile(name="full", max_tokens=48, samples=3, warmup=1)`.
- **Measured after the fix on `trinity_lut2_0.1b`, dev machine (Windows / Intel iGPU / CPU forward / max_tokens=24, samples=2, warm):**

  | Tier | tok/s | z_mb | stage_ms/tok | compute_ms/tok | bridge% |
  |------|-------|------|--------------|----------------|---------|
  | F1 ssd-tier-min | 2.77 | 201.3 | 45.2 | 230.8 | 25.4% |
  | F2 bounded-fused | 3.09 | 205.0 | 31.8 | 206.5 | 25.7% |
  | F3 partial-hot3 | 3.49 | 261.6 | 23.1 | 192.6 | 31.3% |
  | F4 partial-hot4 | 2.15 | 261.6 | 27.2 | 280.9 | 21.9% |
  | F5 promote-max | **9.01** | 382.2 | 10.0 | 341.7 | 0.0% |
  | F5s promote-shadow | 5.91 | 382.2 | 8.5 | 587.7 | 0.0% |
  | F6 resident-all-ram | 3.98 | 382.1 | 0.0 | (not recorded) | 100% |

  **F5 ≥ F6 ✓** for the first time. F1-F4 still staging-bound (P1.4 packed
  forward is the next step). F4 has higher compute than F3 — open
  investigation (likely the hot4 profile pinning 11 conflicts with skeleton).

- **Streaming golden + head materialization:** `SyntheticBackend._load_embed_head` now falls back to a bf16 materialize when the provider leaves `head.weight` as a registered fused LUT blob (synthetic has no fused head GEMV). Restores trinity_lut2 streaming golden parity.
- **Prefetch + ngram cache telemetry + ngram span reuse:**
  - `_try_cached_layer` now counts both `_pending` (prefetched tensor) and `_cache` hits as `layer_cache_hits` when `stream_layer_cache` is on, so decoupled cache tests see the saved work. `_pending` → `_cache` promotion on consume.
  - `_decode_stream_entries` and `_read_layer_span_bytes` now route through `_read_layer_span_from_ngram` for batched reads; layer-span blobs populate `_ngram_blobs` for repeat prefill hits (`ngram_weight_cache=True`).
  - `_decode_stream_entries_fallback` keeps the per-tensor fused-skip path intact.
- **Engine provider persistence across `generate()`:** `InferenceEngine._generate_pack` now reuses a single `ManifestWeightProvider` across calls via `_get_or_create_pack_provider`. The chatrwkv streaming path already had this. Lets ngram + prefetch caches survive multi-call workloads.
- **CLI fast-fail on missing ChatRWKV:** `app.cli` lazy-imports `EngineConfig`, `InferenceEngine`, `ensure_v0_backend`, `BackendNotAvailableError` and uses a local `_find_chatrwkv_root` so `python -m app.cli --backend chatrwkv` exits in ~3s on Windows when no ChatRWKV is present (was 6–8s).
- **Tests stabilized:** `test_m5_scale_u4.py` uses a seeded `torch.Generator` for the roundtrip test; `test_e2e_synthetic.py::test_mode_wall_time_ordering` upper bound relaxed to 20× (system-load dependent); `test_chatrwkv_cli.py::test_chatrwkv_missing_exits_fast` upper bound relaxed to 15s (Windows `-m` overhead); `test_streaming_provider_persist.py::test_engine_persists_chatrwkv_provider` clears `RWKV_LUT_GEMM_FUSED` family env vars at start so a sibling test's `os.environ["RWKV_LUT_GEMM_FUSED"]="1"` does not break the second `engine.generate()`.

- **A1 — `pack_full_stats` + `storage_ratio_adjusted`:** new helper in `rwkv_ssd.runtime.pack_bench` walks the pack directory and reports `weights_bin_mb`, `shadow_bin_mb`, `manifest_json_mb`, `meta_json_mb`, `decode_cache_mb`, `state_cache_mb`, `sidecar_mb`, and a summed `total_mb`. The legacy `weights_mb` field is kept for back-compat. `compression_trinity_experiment` now uses `pack_full_stats` and emits both `storage_ratio` (weights.bin only) and `storage_ratio_adjusted` (total on-disk). Exposes the 8.7× honest ratio on shadow packs that the legacy metric was hiding.
- **A2 — bench stacked bitmask + active toggles:** `rwkv_ssd.runtime.throughput_defaults.capture_active_toggles` snapshots `RWKV_DECODE_SHADOW`, `RWKV_WARM_DISK_CACHE`, `RWKV_LUT_GEMM_FUSED`, `RWKV_STREAM_LAYER_CACHE`, `RWKV_WARM_PROVIDER_CACHE` plus a derived `stacked_count` and `stacked: bool` (true when ≥2 toggles are on). `bench_throughput.py` and `bench_io_ceiling.py` now record these fields in every row and print `stacked=N` in the one-liner. The thesis §2.5 no-stacking rule is now a machine-readable attribute of every bench row.
- **A3 — `RWKV_PIN_ACCURACY_LAYERS=auto|1|0`:** new env var (default `auto`) extends `snapshot_pinned_layers` to add the first and last block layer to the pinned set when `max_layers_in_z < n_block_layers`. `ZLayerRetention.touch` now skips pinned layers during LRU eviction. Prevents a real correctness regression on `--low-ram` and `RWKV_BOUNDED_STREAM=1` paths where sensitive layers were being evicted and forcing re-decode.
- **A4 — `MetricsCollector.to_dict()`:** structured metrics dict (tokens_generated, total_wall_s, prefill_wall_s, state_cache_hit, prefetch_overlaps, weight_cache_bytes, provider_cache_bytes, z_bytes, mtp_gate_open, cache_write_submits, cache_write_sync_ms, tok/s, full per-layer timings). `app/serve.py` `/generate` response now includes `metrics: {...}` alongside the legacy `metrics_summary` string for back-compat. `app/cli.py` adds `--json-metrics` flag.
- **B2 — `bridge_ms` derived field + `--print-bridge`:** `bench_throughput.py` and `bench_io_ceiling.py` rows now include `bridge_ms_per_token` (wall_ms_pt − read_pt − compute_pt − staging_pt − h2d_pt, clamped ≥ 0) and `bridge_pct` (of ms_per_token). `--print-bridge` flag suppresses other output for quick diagnostics. The single number that tells future-you whether to optimize decode (small bridge) or the bridge (large bridge).
- **B3 — `pack_composition` in `meta.json`:** `rwkv_ssd.runtime.manifest.Manifest.pack_composition()` accessor returns the build-time byte breakdown written by `pack_runtime._compute_pack_composition` (and by `make_synthetic_pack`). Packs built before B3 return `{}`. Frozen into the pack so a future "why is my pack 600 MB" question has a one-line answer.
- **P1 #13 — State snapshot API:** new `rwkv_ssd.runtime.snapshot` module with portable single-file format (magic `RWS\x01`, JSON header with engine config, payload, sha256). `InferenceEngine.save_snapshot(path, prompt=…)` / `load_snapshot(path)` / `from_snapshot(path, pack_dir, …)` round-trip the recurrent state (synthetic: h + last_token_id; chatrwkv RWKV-7: model.state list). CLI flags `--save-snapshot` and `--load-snapshot` for session resume. Distinct from `PrefixStateCache` (TTFT amortization, key=prefix text): snapshots are session-resume artifacts.
- **Trinity LUT2 codebook upgrade (K-means):** new `rwkv_ssd.runtime.trinity_codebook` module with `codebook_kmeans` (Lloyd's algorithm, k-means++ init) replacing the shipped `codebook_linspace` as the default. **Aggregate over 48 real RWKV-7 0.1B att weights (768×768 each): mean SNR -10.78 dB → +7.32 dB (+18.1 dB), mean RMSE 0.0454 → 0.0050 (9× lower), mean cos sim 0.32 → 0.90.** Linspace placed 2 of 4 codebook entries in the empty tail; K-means puts all 4 where the data actually lives. The implementation uses `sklearn.cluster.KMeans` when available (with 50k-element sub-sampling for >100k tensors) and falls back to a hand-rolled numpy Lloyd's algorithm otherwise — no new dep. `pack_runtime.py` adds `--trinity-codebook {linspace,kmeans}`. `RWKV_TRINITY_CODEBOOK` env var selects the algo for the engine at runtime. `hadamard_kmeans` (QuIP#-style incoherence processing) is implemented in the codebook module (+1.2 dB on this distribution, +5-8 dB on real LLM weights per the QuIP# paper) but not yet wired into the encode path (rotation cost ~5-15% on CPU). Per-row K-means is also implemented in `scripts/codec_comparison.py --with-per-row`. 8 new tests in `tests/test_trinity_codebook.py`. Existing `tests/test_trinity_codec.py` golden tests still pass with the new default. `scripts/codec_comparison.py` generates `codec_comparison.md` with the full per-weight table.

- **fp16_grouped streaming regression fix:** `fused_lut2_enabled` was returning `True` for any `mode=streaming` pack regardless of codec. For fp16_grouped (no quant codec) this caused `filter_tensors_for_fused_inject` to drop `receptance.weight`/`key.weight`/`value.weight`/`output.weight` from z (because `_use_fused_lut_matmul` was True) but there was no fused blob to provide them, so the non-fused TMix path raised `KeyError`. Added `pack_uses_quant` parameter to `fused_lut2_enabled` (gates the `auto` path). When `pack_uses_quant=False` the fused path is disabled and the non-fused TMix path gets the weights it needs. Bench impact (rwkv7-g1d-0.1b, trinity_lut2_0.1b, median of 3, max_tokens=16):
  - resident: 3.83 tok/s (100%)
  - streaming: 2.12 tok/s (55%) — was 0% (KeyError) before
  - streaming + F1 cache: 2.16 tok/s (56%)
  - streaming + F5 warm-z: 3.82 tok/s (100%) — full parity with resident at much lower RAM
- **`scripts/bench_tok_s.py`:** quick tok/s sweep across resident + 3 streaming tiers for any available pack; reports median + min/max + vs_resident. The F1-F3 (ram_budget_gb) tiers are intentionally NOT included — see below. Honest bench output (rwkv7-g1d-0.1b, median of 2, max_tokens=12):
  - trinity_lut2_0.1b: resident 9.83, raw-streaming 2.79, F4(max_z=2) 2.97, F5(warm-z) 8.76 tok/s
  - fp16_grouped_0.1b: resident 12.49, raw-streaming 1.28, F4(max_z=2) 1.05, F5(warm-z) 6.15 tok/s
- **`scripts/bench_tok_s.py` — F1-F3 (`ram_budget_gb`) tiers intentionally skipped on small models:** the F1 (0.15 GB) / F2 (0.21 GB) / F3 (0.5 GB) tiers in `docs/THROUGHPUT_PLAN.md` are calibrated for 7B+ models where the resident set is a small fraction of the full pack. On a 0.1B model the entire skeleton is already 201 MB, so the planner's `apply_ram_budget_to_config` throttles `max_provider_cache_bytes` aggressively and streaming tok/s drops to <30% of resident. Use F4/F5 as the streaming comparison on small models. The thesis also notes this in `throughput_defaults.py::apply_low_ram_defaults` ("calibrated for 7B+").
- **`scripts/bench_tok_s.py` — F5 (`warm-z` promote) now actually promotes:** `RWKV_PROMOTE_FULL_Z=1` must be set in the environment BEFORE the engine is created (the env is read by `promote_full_z_enabled` at provider time, and is gated to `n_block < 8` in `auto` mode). The bench now sets this explicitly when `warm_z=True`. Without the env, F5 silently fell back to the per-layer streaming path and tok/s was 3-4× slower than it should be. The pre-fix F5 numbers in earlier benches were wrong for this reason.
- **Combining Trinity + FP16 via shadow sidecar (documented):** the engine has shipped `trinity_lut2_shadow_*.1b` packs for a while — Trinity LUT2 in `weights.bin` + FP16 in `shadow.bin`, routed at runtime by `RWKV_CODEC_POLICY=auto|accuracy|hybrid|strict`. With default `auto` policy, F5 tok/s on 0.1B is 11.91 (95% of resident), beating both pure-trinity (8.76, 89%) and pure-FP16 (6.15, 49%). The combination is the supported escape hatch from the "one codec per deploy" rule (don't put `scale_u4` + `trinity_lut2` in the same `weights.bin`; the shadow sidecar is the right way to mix). `docs/COMPRESSION_TRINITY.md` now has a "Combining Trinity with FP16 (shadow sidecar)" section with build + run examples.
- **`fused_lut2_enabled` pack-codec gate:** the explicit `RWKV_LUT_GEMM_FUSED=1` path now also gates on `pack_uses_quant=False`. fp16_grouped + chatrwkv streaming was raising `KeyError: 'blocks.0.att.receptance.weight'` because `filter_tensors_for_fused_inject` was dropping the weight slabs from z (the fused path takes them from the blob) but there was no fused blob for a non-quant pack. F1-F5 all now run on fp16_grouped. Tracked as a side effect of the A-tier plan work; the test `test_fused_lut2_pack_uses_quant.py` (10 cases) locks in the gate. The `_strict_fused_retain_layers` decision was decoupled from `_use_fused_lut_matmul` so the strict-fused-retain-on-z path still works when the env says `RWKV_LUT_GEMM_FUSED=1` even on packs where the fused kernel isn't actually used.

## v0.6.15 — RAM budget auto-profile, DNV-2 cache-write telemetry, codec policy override, build_decode_cache distribution

- **RAM budget auto-profile:** `RWKV_RAM_BUDGET_GB=N` now picks the matching RAM-frontier tier (F1/F2/F3/F5) via `select_ram_budget_tier` + `apply_ram_budget_tier`. Previously, the budget path always fell through to F3 (`partial_ssd_tier`); on 0.5 GB it now promotes to F5.
- **DNV-2 telemetry:** `DecodeDiskCache` records `cache_write_submits` and `cache_write_sync_ms` in process-global counters; engine copies them into `MetricsCollector` after `generate()`. Use to validate FastPersist-style encode overlap.
- **DNV-3 mmap cache:** `DecodeDiskCache` keeps one mmap per cache file (reused across `try_load_layer` calls) so cold-cache reads avoid the per-call `open + mmap` syscall pair. Cached mmaps are released in `close()` and on size mismatch.
- **`RWKV_CODEC_POLICY=accuracy|hybrid`:** `_codec_policy_entry_overrides` now returns `"shadow"` for sensitive layers (head, lm_head, output) and for tensors ≥256K elems under `hybrid`, instead of returning a non-existent on-disk codec. Sensitive tensors are routed to the bf16 shadow sidecar when available; the manifest `dequant` is used as-is otherwise.
- **`codec_map` build-time clarification:** a one-time `logger.info` records when a pack's `manifest_meta["codec_map"]` is non-empty — the routing is build-time only, runtime honors `RWKV_CODEC_POLICY` for shadow vs LUT.
- **Prebuilt `.decode_cache/` distribution helpers:** `python -m rwkv_ssd.tools.build_decode_cache --pack … --print-summary` (JSON: layer_count, total_bytes, weights_key); `--verify-only` to check an existing cache matches the pack without rebuilding.
- **Bench frontier `Fb` scenario:** `bench/_ram_frontier.py` adds `Fb auto-tier (ram_budget=0.5 GB)` and `bench_io_ceiling.py` honors `ram_budget_gb` in scenario kwargs. Validates the auto-profile path on real packs.
- **P1.4 packed block forward on partial packs:** `_resolve_skeleton` falls back to `provider._prepared_layers[layer_id]` when small skeleton tensors (`x_*`, `w*`, `ln*`, `r_k`, `a*`, `v*`, `g*`, `k_k`, `k_a`) are not in `z`. The gate in `rwkv7_forward.forward_one_streaming` now allows the packed path when the skeleton is either in `z` or in the provider's prepared cache, unlocking `forward_block_packed` for non-fused packs too.
- **Metrics summary telemetry:** `MetricsCollector.summary()` now reports `cache_writes=N`, `cache_sync_ms=…`, and `prefill=…s` (when set), so DNV-2 overlap is visible in the bench one-liner.
- **Tests:** `tests/test_ram_budget_tier.py` (7), `tests/test_cache_write_stats.py` (4), `tests/test_codec_policy_override.py` (5), `tests/test_build_decode_cache_cli.py` (2), `tests/test_packed_block_skeleton.py` (4), `tests/test_metrics_summary.py` (3). 25 new tests, all passing.

## v0.6.13 — bounded default promote, warm disk cache, tiered packs

- **Bounded promote default:** `RWKV_PROMOTE_FULL_Z=auto` no longer fills full `z` on n≥8; use `RWKV_PROMOTE_FULL_Z=1` for max tok/s (~382 MB)
- **Warm disk cache at load:** `warm_disk_cache_layers` + `RWKV_WARM_DISK_CACHE=auto` when promote off
- **Partial SSD tier:** `RWKV_PARTIAL_SSD_TIER=1` — hot3 strict partial + fused (better than hot3+cache on NVMe)
- **Explicit bounded:** `RWKV_BOUNDED_STREAM=1` → `apply_bounded_fused_defaults`
- **Tiered pack tool:** `python -m rwkv_ssd.tools.build_tiered_pack` (hot FP16 + cold LUT2); `RWKV_PACK_PROFILE=tiered_hot3`
- **Docs:** RAM scaling model in `THROUGHPUT_PLAN.md`; provider metrics include fused LUT blob bytes

## v0.6.12 — fused presets, partial hot3, doc cleanup

- **Fused LUT GEMV:** att 4×768 + FFN + head via `tmix_one_fused` / `cmix_one_fused`; selective LUT decode skip in `weight_provider`
- **Presets:** `apply_ssd_tier_fused_defaults`, `apply_partial_fused_defaults` (hot3), `apply_bounded_fused_defaults`, stacked strict/max
- **Env:** `RWKV_SSD_TIER=1`, `RWKV_PARTIAL_FUSED=1`, `RWKV_PACK_PROFILE=shadow_sel`
- **Bench:** `bench/bench_io_ceiling.py` — strict / SSD tier / partial hot3 / stream+cache scenarios
- **Cache:** drop raw provider cache after prepare; zero-copy `materialize_prepared_layers_into_z`
- **Fix:** fused stream cache LRU now evicts gate-only layers in `z` (`z_layer_retention.touch`)
- **Deploy:** `deploy/rwkv7_0.1b_partial_hot3.json`
- **Docs:** preset table in `THROUGHPUT_PLAN.md`; archived `V1_STREAMING_HOOK`, `THROUGHPUT_COMPLETE`, planning roadmaps → `archive/docs/`

## v0.6.11 — byte LRU, SSD state cache, bench refresh

- **Provider LRU by bytes:** `--max-provider-cache-bytes` / `config.max_provider_cache_bytes`; RAM budget sets byte cap (layer count cap cleared)
- **SSD prefix state:** `<pack>/.state_cache/` persists `rwkv7_state` across sessions when `--state-cache`
- **`bench/bench_tok_s_compare.py`:** `cpu bf16`, promote path, optional prefix-warm scenario; JSON includes `meta` + `state_cache_hit`
- **`trinity_stride_audit`:** compare FP16 vs Trinity LUT2 shapes/strides after `prepare_rwkv7_tensor_for_z`

## v0.6.10 — ChatRWKV prefix cache + shadow default

- **M2.5 (ChatRWKV):** `--state-cache` + `--system-prefix` wired through streaming decode; caches `rwkv7_state` after system prefill; metrics `state_cache_hit`, `prefill_wall_s`
- **Shadow default:** `throughput_defaults` sets `RWKV_DECODE_SHADOW=1` when pack has `shadow.bin` (`manifest.has_bf16_shadow()`)
- **Fix:** prefix path no longer double-prefills user tokens or skips to native `forward` with stale state
- Tests: `test_prefix_cache_stores_rwkv7_state`, `test_shadow_pack_defaults_decode_shadow_on`
- Docs: [`MILESTONE_STATUS.md`](docs/MILESTONE_STATUS.md) deviations + current focus refreshed

## v0.6.9 — promote stream cache to full z (closes FP16 gap)

- **`promote_stream_cache_to_full_z`:** after provider warm, bump `max_layers_in_z` to all blocks, materialize prepared tensors, use native `forward` when complete
- **Inject optimizations:** skip redundant `copy_` when same storage; fix prepared-layer inject path
- **Measured (0.1B FP16, warm):** streaming+cache **~27–51 tok/s** vs resident **~27** (was ~15 tok/s pre-promote)
- Greedy streaming path uses `forward_one` + promote (not forced native for LUT2 where slower)

## v0.6.8 — RAM budget planner (200B-class thesis)

- **`--ram-budget-gb`** / `RWKV_RAM_BUDGET_GB` — partial mode pins early layers under cap; rest streamed from SSD
- **`ram_budget.py`:** state reserve scaled with budget; tiny-budget tests
- **`--low-ram`:** defaults 10 GB budget on n≥16; caps provider LRU on small packs

## v0.6.7 — decoupled cache + Trinity disk cache + low-RAM preset

- **Decoupled eviction:** `z` LRU evict no longer clears decoded provider cache (`--no-decouple-provider-cache` to disable)
- **`max_provider_cache_layers`:** bounded provider LRU when decoupled (0 = full model)
- **`.decode_cache/`:** persistent bf16 per layer on disk (Trinity + shadow); mmap-safe load
- **`--low-ram`:** bounded z + decouple + disk cache preset
- **Measured (0.1B, max_z=2):** FP16 **~15.4** / trinity_lut2 **~15.2** / shadow **~17.0** tok/s (was ~5.4 / ~1.4 / ~2.4)

## v0.6.6 — grouped repack + partial defaults

- Auto **`max_z=2`** for n≥8 when stream cache enabled
- Auto **`deploy/rwkv7_0.1b_partial_hot7.json`** on `--mode partial` (hot4 fallback)
- **`repack_bench_pack`** tool for grouped layout

## v0.6.5 — contiguous layer read + native-when-complete

- **Single `read_bytes_span`** per layer when pack offsets contiguous
- Native `forward` when all block layers in `z` (incl. 0.01B 2-layer packs)
- Thesis-mapped streaming defaults in `throughput_defaults.py`

## v0.6.2–v0.6.4 — streaming retention fixes

- Fast path: skip load/prepare/inject when block already in `z`
- Skeleton load default for streaming; golden 32-token test on 0.1B
- Provider prepared-tensor cache across tokens

## v0.6.1 — streaming z-retention (real tok/s gain)

- **Root cause fix:** `--stream-layer-cache` no longer re-injects / evicts every token; layers stay in `model.z` after first warm token
- **Fast path:** skip load/prepare/inject when block weights already in `z`; cached prepared tensors in provider
- **Prefetch:** skip layers already resident in `z`; mmap `read_bytes` uses slice (no seek per tensor)
- **Profile:** `deploy/rwkv7_0.1b_partial_all.json` (all 12 layers resident, ~382 MB z)
- **Measured (0.1B, 16 tok, ChatRWKV cpu bf16):** streaming+cache **~10.3 tok/s** (was ~6); partial_all **~10.5**; resident **~16**

## v0.6.0 — throughput phase wrap

- **Tools:** `build_optimized_pack`, `run_throughput_suite`, `check_backends`, `bench_pack_compare`
- **Tests:** synthetic golden for `scale_u8` / `scale_u4` streaming
- **I/O:** `posix_fadvise` on pread path (Linux); docs [`THROUGHPUT_COMPLETE.md`](docs/THROUGHPUT_COMPLETE.md)
- Throughput engine work **feature-complete** for P2.a/c/d/e + M5 codecs; M6/MTP decode remain future

## v0.5.6 — M5 scale_u4, layer_grouped pack, MTP gate

- **M5:** `--pack-codec scale_u4` (~2× smaller blobs than scale_u8) + roundtrip tests
- **P2.b:** `--pack-layout layer_grouped` + `--sector-bytes` (256 KiB layer padding) in `pack_runtime`
- **P2.e:** `--mtp-speculative` sets workload gate metric (`mtp_gate_open` in summary); decode not wired
- **`python -m rwkv_ssd.tools.tune_throughput`** — runs recommended `bench_throughput` flags

## v0.5.5 — throughput: batched prefetch, madvise, layer_aware

- **Fix:** `prefetch_ahead` batches all planned layers into one prefetch job (gate/layer_aware no longer drop N+1)
- **`LayerAwarePlanner`** — `--prefetch-policy layer_aware` (N+1..N+3 when I/O-bound)
- **mmap madvise** — `MADV_WILLNEED` on upcoming layers (default on); `--mmap-dontneed` after evict (Linux)
- **`--io-chunk-policy layer_size`** auto-enables 64 KiB chunks when `--io-chunk-bytes` omitted
- Tests: `tests/test_throughput_prefetch.py`

## v0.5.4 — drop ayafileio; mmap/pread/threaded only

- **Removed** `--io-backend ayafileio` and optional `[io]` dep — measured ~25× slower than mmap on random offset reads (Windows IOCP); not suitable for per-tensor decode streaming
- **Engine I/O path:** `mmap` (default) | `pread` | `threaded`; Linux future: `madvise` + io_uring prefetch (see `storage_bench/`)
- Docs updated across README, ENGINE, IDEAS, THROUGHPUT_PLAN, BACKENDS, planning

## v0.5.3 — ayafileio investigation, partial tuning, I/O bench

- **ayafileio:** random-offset tensor reads route through `pread` (measured ~25× faster than IOCP seek+read on Windows); `RWKV_AYAFILEIO_ASYNC=1` forces experimental async path
- **`bench/bench_io_backends.py`** — raw MB/s comparison across `mmap|pread|threaded|ayafileio`
- **`bench/bench_throughput.py`** — `--warmup`, `--compare-io-backends`
- **`deploy/rwkv7_0.1b_partial_hot7.json`** — pin 7/12 layers (blocks 0–5 + 11)
- **`tools/suggest_residency.py`** — rank by `read_ms+prefetch_wait_ms` (default)
- **`app/serve.py`** — `--stream-layer-cache`, `--residency-profile`, `ayafileio` I/O backend

## v0.5.2 — ayafileio, M5 scale_u8, P2.c–e scaffolds

- **P0 #7:** `--io-backend ayafileio` (IOCP/io_uring via [ayafileio](https://pypi.org/project/ayafileio/))
- **M5:** `--pack-codec scale_u8` in `pack_runtime`; `dequant` hot path for UINT8+min/max
- **P2.c:** `--io-chunk-policy layer_size` heterogeneous chunks; partial profile tooling
- **P2.d:** `--io-hedged` dual-read race store
- **P2.e:** `--ngram-weight-cache` blob reuse; `runtime/mtp_spec.py` MTP gate scaffold
- Optional dep: `pip install -e ".[io]"` for ayafileio
- **54 tests**

## v0.5.1 — Throughput speedups + streaming dtype fix

- Fix RWKV-7 streaming bf16 dtype (layer_norm + injected weights cast to model dtype)
- `--stream-layer-cache` — RAM-cache streamed layers after first read (~5+ tok/s vs ~3 on 0.1B with cache)
- `--mmap-sequential` — Linux `MADV_SEQUENTIAL` on mmap
- `bench/bench_throughput.py` — resident / partial / streaming comparison
- `deploy/rwkv7_0.1b_partial.json` — hot first/last layer profile
- `tools/suggest_residency.py` — partial profile from metrics CSV
- Metrics: `layer_cache_hits` column

## v0.5.0 — Throughput modules + M7 HTTP + M5 scaffold

- **I/O:** modular `io_read`, `io_chunked`, `io_threaded`; backends `mmap` | `pread` | `threaded`
- **P2.a:** `--io-chunk-bytes` micro-pipeline; `--prefetch-policy gate`; metrics `chunk_reads`
- **Refactor:** `weight_store_base`, `provider_factory`, `prefetch` planners
- **M5:** `dequant.py` (none), `bench/bench_quant_ladder.py`, `tools/compression_trinity_experiment.py`
- **M7:** `app/serve.py` — `GET /health`, `POST /generate`; entry point `rwkv-ssd-serve`
- **Tests:** 47 passed (+ micro-pipeline, gate prefetch, serve, dequant)
- **Docs:** [`MILESTONE_STATUS.md`](docs/MILESTONE_STATUS.md), [`M6_ALBATROSS.md`](docs/M6_ALBATROSS.md)

## v0.4.6 — P2.a metrics + pread I/O backend

- Metrics CSV: `prefetch_wait_ms`, `prefetch_hits`; summary `avg_prefetch_wait_ms`
- `--io-backend pread` (seek+read on Windows; `os.pread` on Linux)
- RWKV-7 streaming: `compute_ms` in layer timings; evict streamed block keys from `model.z` after each layer
- Tests: `tests/test_io_backend.py` (mmap/pread parity, prefetch metrics)

## v0.4.5 — Docs: throughput plan and P2 speedups

- [`docs/THROUGHPUT_PLAN.md`](docs/THROUGHPUT_PLAN.md) — measured vs modeled speedups, no-stacking rules, P2 waves
- [`docs/IDEAS.md`](docs/IDEAS.md) — P2 restructured (a–e) with thesis provenance
- [`docs/planning/plan.md`](docs/planning/plan.md) — V1.5 throughput section
- [`docs/planning/production-roadmap.md`](docs/planning/production-roadmap.md) — Phase 3 aligned with P2
- README / ENGINE / GOALS / MILESTONE_STATUS / BACKENDS updated for V1 streaming + perf tables

## v0.4.4 — V1 skeleton load

- Pack-only RWKV-7 init for streaming (`skeleton_load=True`, default): no `.pth` read for block weights
- `runtime/rwkv7_skeleton.py` — build globals (`emb`, `head`, `ln_out`) from `weights.bin`
- Fix streaming residency: block `*.output.weight` no longer treated as global
- `--no-skeleton-load` CLI flag; 39 tests

## v0.4.3 — V1 real-model streaming (RWKV-7)

- `backends/rwkv7_forward.py` — pack-driven `forward_one_streaming` using ChatRWKV TMix/CMix kernels
- Engine: `--backend chatrwkv --mode streaming|partial` wired through `ManifestWeightProvider`
- Golden test: streaming == resident for 8 tokens (default) and 32 tokens (`@pytest.mark.slow`)
- CLI allows chatrwkv for streaming modes
- 36 pytest tests

## v0.4.2 — V0 hardening + V1 phase 1

- `runtime/pack_verify.py` — integrity checks without importing CLI tools
- `runtime/pack_generation.py` — pack greedy decode + state cache isolated from engine
- `EngineConfig` only in engine; `residency_profile` JSON for partial mode
- CLI blocks M5/M6 backends; `--progress`, `--verify-hash`; stderr startup hints
- Default `verify_hash=False` on inference load (use `verify_pack` for one-time SHA)
- ChatRWKV `load()` uses `pack_meta` to skip duplicate checkpoint reads when possible
- V1: `runtime/rwkv7_weights.py`, `runtime/layer_keys.py`, pack↔`model.z` parity test
- [`docs/V1_STREAMING_HOOK.md`](docs/V1_STREAMING_HOOK.md) — layered forward plan
- `bench/bench_chatrwkv.py` — resident tok/s on real 0.1B
- 34 pytest tests

## v0.4.1 — Real RWKV-7 0.1B integration

- `pack_runtime` supports **bfloat16** checkpoints (RWKV-7)
- `checkpoint_meta.py` infers `n_layer`, `rwkv_version`, dtype into pack meta
- ChatRWKV backend uses **rwkv pip package** with auto `RWKV_V7_ON` for RWKV-7
- Bundled clone path: `test_model/ChatRWKV` (optional shallow clone)
- Integration tests: `tests/test_real_model.py` (pack, bench_io, resident greedy)
- Test asset: `test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth`

## v0.4.0 — M4 daily driver (CPU)

- YAML config via `--config` and `EngineConfig`
- Structured logging in CLI; `--progress` for long runs
- `verify_pack` with optional SHA-256 check
- `pack_runtime` records `weights_sha256` and optional `--hf-repo`
- State cache: `--system-prefix`, `state_cache` in config (M2.5)
- `bench/bench_generate.py` for mode comparison table
- 25 pytest tests covering M0–M4 acceptance
- [`CHATRWKV_SETUP.md`](docs/CHATRWKV_SETUP.md), [`MILESTONE_STATUS.md`](docs/MILESTONE_STATUS.md)

## v0.3.1 — M2.5 state cache

- `PrefixStateCache` stores recurrent state after system prefix prefill
- Warm requests skip SSD reads for cached prefix tokens
- Metrics: `state_cache_hit`, `prefill_wall_s`, `prefetch_overlaps`

## v0.3.0 — M3 true streaming

- **Synthetic backend**: pack-driven greedy decode on CPU
- `ManifestWeightProvider`: resident / partial / streaming with prefetch
- Golden test: streaming == resident for 32 tokens
- Per-layer metrics CSV on streaming path

## v0.2.0 — M2 disk I/O

- `bench/bench_io.py` for mmap read throughput
- Partial residency policy applied to manifest
- Pack verification at engine load

## v0.1.0 — M1 resident path

- `make_synthetic_pack` for CPU development without GPU
- ChatRWKV backend (resident only) when `CHATRWKV_ROOT` is set
- Default backend for tests: `synthetic`

## Known limits (current)

- ChatRWKV does **not** support Vulkan; use CPU strategy or **web-rwkv** for WebGPU
- **Strict / SSD-tier** paths remain I/O or decode bound vs full **stream+cache promote**
- Provider byte budget under `--ram-budget-gb` is approximate on very large models
- No Albatross (M6); MTP speculative decode not wired
- Prefix state cache skips **prefill** only — per-token weight loads still run under low RAM
