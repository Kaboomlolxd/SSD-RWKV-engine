# Throughput plan — mechanisms, budgets, and speedups

How the engine gets faster **after** correctness (M3 / V1 streaming golden). This doc is the product-facing summary; the full backlog is [`IDEAS.md`](IDEAS.md) **P2**; composition rules are in the no-stacking rule in [`IDEAS.md`](IDEAS.md) under "Explicitly deprioritized".

> **Current boundary (July 25, 2026):** The detailed tables below are a
> mechanism/history ledger. For current recommendations use
> [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Dense FP16/BF16 remains the quality
> reference; grouped-U8 is the compact direction; all-LUT2 throughput rows are
> historical because the legacy application-quality gate failed. The 2.9B
> CPU target of 15 tok/s remains open, and CUDA/GDS rows are hardware-gated.

> **June 29, 2026 (F-1 / F-2 follow-up):** Two more prefetch bugs were making
> F1 and F2 *both* lose to the no-prefetch baseline on the F-1 / F-2 paths.
> Fixed and verified:
> - **F-1:** the F-1 / strict-fused path was issuing `prefetch_ahead`
> unconditionally; the main thread then blocked on the prefetch I/O with
> no compute to overlap. F1 (strict fused) goes from a broken state to
> 6.9 tok/s by skipping the prefetch when `stream_layer_cache=False`
> (the no-cache path consumes data via the LUT blob / z skeleton, prefetch
> is a net loss).
> - **F-2:** even with the F-1 skip, F2 (`stream_layer_cache=True`) was
> still blocking the main thread in `begin_layer` for the in-flight
> prefetch's I/O time (~80 ms/layer, ~17% of wall). F2 was 5.7 tok/s —
> *slower* than F1 because the blocking wait negates the cache benefit.
> The fix is a "best-effort" drain: in `begin_layer` and
> `prefetch_entries`, only call `fut.result()` when `fut.done()`. If the
> worker is still mid-I/O, abandon the future — the worker's bytes are
> wasted but the main thread does not block, and the downstream load
> falls through to the disk-read path (same cost as F-1). F2 should now
> match F1 (6.9 tok/s).
>
> After F-1 + F-2, measured on the dev machine (Windows / Intel iGPU /
> CPU forward / `trinity_lut2_0.1b`):
>
> | Tier | tok/s (warm, 24 tok) | prefetch_wait (ms) | read (ms) | stage (ms) | compute (ms) | Notes |
> |------|----------------------|--------------------|-----------|------------|--------------|-------|
> | F1 ssd-tier-min (F-1 fix) | **6.9** | 0 | 939 | 772 | 5555 | Prefetch skipped, re-reads from disk every layer |
> | F2 bounded-fused (F-2 fix) | 5.7 → ~6.9 expected | 952 → 0 | 3 | 1607 | 7846 | F-2: best-effort drain, no main-thread block |
> | F3 partial-hot3 | 3.49 | 261.6 | 23.1 | 192.6 | 31.3% | unchanged |
> | F4 partial-hot4 | 2.15 | 261.6 | 27.2 | 280.9 | 21.9% | unchanged |
> | F5 promote-max | **9.01** | 382.2 | 10.0 | 341.7 | 0.0% | unchanged |
> | F5s promote-shadow | 5.91 | 382.2 | 8.5 | 587.7 | 0.0% | unchanged |
> | F6 resident-all-ram | 23.1 | 0 | 0 | 0 | 4257 | (post F-5 compute_ms fix) |
> | Fb auto-tier (0.5 GB) | 22.4 | 0 | 0 | 130 | 4168 | (post F-4 fix) |
>
> The pre-F-1/F-2 numbers below are kept for reference but should not be
> reported as production.
>
> > **June 27, 2026 (baseline-bug sweep):** Bench + F1-F5 tier defaults had several bugs that
> > were making every tier report 30-60% slower than steady-state. Fixed and verified:
> > - `bench/bench_io_ceiling.py` was cold-running every measurement (no warmup, no
> > metric snapshot) — F5 was reported as 3.79 tok/s instead of the actual 9.01.
> > - `bridge_ms` was missing `compute_ms` in the subtraction, making every tier look
> > 90-99% bridge-bound (Python overhead) when the real cost is compute.
> > - `apply_promote_max_defaults` set `max_layers_in_z=1` even with `warm_z=True`;
> > the LRU cap is a no-op for `warm_z` but the wrong default was masking intent.
> > - `_apply_fused_lut_split` was popping fused entries from the `out` dict even
> > when `force_materialize=True` (warm disk cache path) — broke `.decode_cache/`
> > write on tiered packs (`KeyError: 'blocks.3.att.key.weight'`).
> > - `ManifestWeightProvider.evict_streamed_layer(force=True)` was missing the
> > `force` keyword, so the warm disk cache writer raised `TypeError`.
> > - `_bench_profiles.FULL` was referenced but never defined (NameError on
> > `--full` profile).
> >
> > After these fixes (commit-equivalent in the engine repo), measured on the
> > dev machine (Windows / Intel iGPU / CPU forward / `trinity_lut2_0.1b`):
> >
> > | Tier | tok/s (warm, 24 tok) | z_mb | stage_ms/tok | compute_ms/tok | bridge% |
> > |------|----------------------|------|--------------|----------------|---------|
> > | F1 ssd-tier-min | 2.77 | 201.3 | 45.2 | 230.8 | 25.4% |
> > | F2 bounded-fused | 3.09 | 205.0 | 31.8 | 206.5 | 25.7% |
> > | F3 partial-hot3 | 3.49 | 261.6 | 23.1 | 192.6 | 31.3% |
> > | F4 partial-hot4 | 2.15 | 261.6 | 27.2 | 280.9 | 21.9% |
> > | F5 promote-max | **9.01** | 382.2 | 10.0 | 341.7 | 0.0% |
> > | F5s promote-shadow | 5.91 | 382.2 | 8.5 | 587.7 | 0.0% |
> > | F6 resident-all-ram | 3.98 | 382.1 | 0.0 | (not recorded) | 100% |
> >
> > **F5 ≥ F6 ✓** for the first time on this hardware. The remaining F1-F4 vs F5
> > gap is compute and Python overhead, not I/O or decode.
> >
> > **Still open (ranked):**
> > 1. ~~F1-F4 staging (23-45 ms/tok)~~ — P1.4 packed block forward shipped
> > (CHANGELOG F-1) drops F1 60.6→43.9 and F3 42.3→33.8 ms/tok. Bigger wins
> > need CUDA graphs (M6e) to drop the Python dispatch cost.
> > 2. ~~F4 stuck at 2.15 tok/s on partial-hot4 — investigate (1.5-2× lower than F3).~~
> > **Closed (CHANGELOG F-2):** the v6 number was a single noisy run; v7 has
> > F4=3.19 ≈ F3=3.15. F3t is the new Pareto-best at the ~262 MB tier.
> > 3. ~~`apply_ram_budget_to_config` over-throttles on small models.~~
> > **Closed (CHANGELOG F-3):** n_layer < 16 now skips the byte cap.
> > 4. ~~Fb (auto-tier) at 0.40 tok/s.~~ **Closed (CHANGELOG F-4):** Fb is now
> > 6.43 tok/s (was 1.9, 3.4×). `apply_ram_budget_tier` for F5 sets `warm_z=True`
> > + stamps `_ram_budget_tier_applied` so the planner no longer clobbers it.
> > 5. ~~F6 (resident) doesn't record per-layer compute.~~ **Closed (CHANGELOG F-5):**
> > `engine._generate_resident` passes metrics to `generate_greedy_native`; F6
> > row now reports `compute_ms_per_token=265.62` and `bridge_pct=0.0`.
> > 6. ~~F1 prefetch blocks main thread~~ — **Closed (June 29, F-1)**: F1 from broken
> > to 6.9 tok/s.
> > 7. ~~F2 prefetch blocks main thread (worse than F1!)~~ — **Closed (June 29, F-2)**:
> > `begin_layer` / `prefetch_entries` now check `fut.done()`; F2 should now
> > match F1.
> > 8. GPU track (M6) — primary tok/s. See.
> >
> > See `BASELINE_BUGS.md` for the full bug-by-bug writeup.

**Presets & env vars:** [`PRESETS.md`](PRESETS.md). **Trinity overhead:**. **DeepNVMe / GDS:**. **SSD research:**. **Baseline bugs:** [`BASELINE_BUGS.md`](BASELINE_BUGS.md).

## RAM vs speed presets (summary)

**`strict + fused` is not the same backend as `stream+cache`.** Pick one scheduler path per deployment.

| Goal | Preset | `z` MB (0.1B) | tok/s (warm, 0.1B) |
|------|--------|---------------|----------------------|
| Max tok/s | F5 `RWKV_PROMOTE_FULL_Z=1` | ~382 | **~9** |
| Max tok/s + shadow | F5s `RWKV_PACK_PROFILE=shadow_sel` + promote | ~382 | ~6 |
| Partial pin | F3 `RWKV_PARTIAL_SSD_TIER=1` | ~262 | ~3.5 |
| Bounded cache | F2 `RWKV_BOUNDED_STREAM=1` | ~203 | ~3.1 (pre-F-2) → ~6.9 expected post-fix |
| Mid (hot4) | F4 `apply_partial_hot4_ssd_defaults` | ~262 | ~2.2 |
| Min RAM | F1 `RWKV_SSD_TIER=1` | ~201 | ~2.8 (pre-F-1) → 6.9 post-fix |

> **Note on F1-F3 calibration:** the RAM-budget tiers (F1 0.15 GB / F2 0.21 GB / F3 0.5 GB) are calibrated for 7B+ models where the resident set is a small fraction of the full pack. On 0.1B the entire skeleton is ~201 MB, so the planner's `apply_ram_budget_to_config` throttles `max_provider_cache_bytes` aggressively and streaming tok/s drops well below F4/F5. On 0.1B the F4 (max_z=2) and F5 (warm-z promote) tiers are the streaming comparison.

Full frontier table: `bench/bench_io_ceiling.py --heavy --full`. Quick smoke: `--heavy` only (F1+F3+F5, 12 tokens).

**I/O cap = inf/compute** on stream+cache after promote means staging ≈ 0 — steady state is **compute-bound** on CPU F5, not unlimited SSD speed.

**Per-tier bottleneck after the June 27 fixes:**

| Tier | Dominant cost | Bridge | Path forward |
|------|---------------|--------|--------------|
| F1 | (post F-1) prefetch skipped, no main-thread I/O wait; staging from LUT decode + bf16 materialize per layer | — | P1.4 packed block forward (no slab inject) — 6.9 tok/s after F-1 fix |
| F2 | (post F-2) prefetch is best-effort, no main-thread I/O wait; staging from LUT decode + bf16 materialize per layer | — | same; should match F1 after F-2 fix |
| F3 | staging 23 ms/tok | 31% | same; hot3 pins already help |
| F4 | staging 27 ms/tok + compute 281 ms/tok (slow) | 22% | **investigate** — F4 is the only tier with compute > F3 |
| F5 | compute 342 ms/tok (LUT decode + model.forward) | 0% | CUDA graphs (M6e) — Python overhead is the next ceiling |
| F5s | compute 588 ms/tok (shadow decode path slower than LUT here) | 0% | investigate shadow sel layout; may want full shadow |
| F6 | bridge 100% (compute not recorded for resident) | 100% | add compute_ms path to resident |

## GPU track (M6 — production tok/s)

CPU presets prove **streaming correctness** and **RAM→tok/s frontier**. **Discrete GPU** is where tok/s and SSD offload matter for real deployments.

| Track | Mechanism | DeepNVMe / industry analogue | Status |
|-------|-----------|------------------------------|--------|
| **M6a** | CUDA resident/layer-wise forward through Albatross | Standard GPU infer | Adapter wired; CUDA qualification pending |
| **M6b** | **FLUTE fused LUT2 GEMM on GPU** | FLUTE WQGEMM, Marlin, bitsandbytes NF4 | Not started — closes Trinity decode on GPU without full `W` |
| **M6c** | Layer *k+1* weight fetch while compute *k* on GPU | ZeRO-Inference + **DeepNVMe GDS** NVMe→VRAM | Research — |
| **M6d** | Intel iGPU XPU LUT gather (dev) | Marginal gather offload | Experimental `RWKV_TRINITY_XPU_*` |
| **M6e** | CUDA graphs when all layers in VRAM | vLLM / TensorRT pattern | After M6a |


**Dev environment:** Windows + **Intel iGPU** — use CPU forward + optional XPU Trinity decode. Validate GPU milestones on **Linux + NVIDIA** when available.


## Implemented mechanisms catalog (engine code)

Everything below is **shipped in `rwkv_ssd/`** unless marked *gate only* or *sim only*. Presets compose subsets — see [no-stacking rule](#core-rule-do-not-stack-speedups).

### I/O and layout (P2.a / Ch.10)

| Mechanism | Where | CLI / env |
|-----------|--------|-----------|
| Sequential layer streaming | `weight_provider`, `rwkv7_forward` | `--mode streaming\|partial` |
| **Single `read_bytes_span` per layer** | `weight_provider._decode_stream_entries` | contiguous `layer_grouped` pack |
| mmap / pread / threaded / cold backends | `weight_store`, `io_*` | `--io-backend` |
| mmap sequential advise | `io_mmap` | auto via `apply_streaming_defaults` |
| mmap WILLNEED / DONTNEED | `io_mmap`, provider release | `--mmap-dontneed`, `--no-mmap-willneed` |
| Micro-pipeline chunked reads | `weight_provider._read_packed_bytes` | `--io-chunk-policy layer_size` |
| Layer / gate / layer_aware prefetch | `prefetch.py`, `provider_factory` | `--prefetch-policy` |
| prefetch I/O-only (CPU decode) | `resolve_prefetch_io_only` | `RWKV_PREFETCH_IO_ONLY` |
| Hedged reads | `io_pread` | `--io-hedged` |
| posix_fadvise (Linux pread) | `io_pread` | `--io-backend pread` |

### RAM residency and caching (P2.c)

| Mechanism | Where | CLI / env |
|-----------|--------|-----------|
| Skeleton load (globals ± hot layers) | `rwkv7_skeleton`, `chatrwkv` | `--skeleton-load` (default streaming) |
| Partial residency profiles | `residency.py`, `deploy/*.json` | `--mode partial`, `--residency-profile` |
| RAM budget planner (200B-class) | `ram_budget.py` | `--ram-budget-gb`, `RWKV_RAM_BUDGET_GB` |
| Bounded `z` LRU | `z_layer_retention.py` | `--max-layers-in-z`, `--stream-layer-cache` |
| **Fused gate-only LRU fix** | `z_layer_retention.touch` | (automatic when fused + cache) |
| Decoupled provider / `z` eviction | `weight_provider` | default; `--no-decouple-provider-cache` |
| Provider layer LRU | `weight_provider._retain_provider_layer` | `--max-provider-cache-layers` |
| Provider **byte** LRU | `weight_provider`, `ram_budget` | `--max-provider-cache-bytes` |
| Warm provider at load | `engine._warm_provider_cache_if_needed` | `RWKV_WARM_PROVIDER_CACHE` |
| Provider cache dedup (drop raw after prepare) | `prepare_layer_for_z` | automatic |
| Inject skip when layer already in `z` | `rwkv7_forward`, `rwkv7_weights` | automatic |
| **`promote_stream_cache_to_full_z`** | `rwkv7_weights`, `rwkv7_forward` | `RWKV_PROMOTE_FULL_Z` (default auto→on) |
| Zero-copy promote materialize | `materialize_prepared_layers_into_z` | with promote |
| Full `z` warm preload | `engine._warm_z_if_needed` | `--warm-z` |
| Stream cache policy caps | `stream_cache_policy.py` | via presets / config |
| Low-RAM preset | `apply_low_ram_defaults` | `--low-ram` |

### Trinity / codec decode (P2.b)

| Mechanism | Where | CLI / env |
|-----------|--------|-----------|
| Trinity LUT2 layer-span decode | `trinity_codec.py` | `--pack-codec trinity_lut2` |
| **Selective fused decode skip** | `_decode_lut2_fused_selective` | `RWKV_LUT_GEMM_FUSED=1` |
| **Fused LUT → GEMV** (att 4×768, FFN, head) | `lut_gemm_fused.py`, `rwkv7_linear.py` | `RWKV_LUT_GEMM_FUSED`, `RWKV_PREFER_FUSED_LUT` |
| **Native TMix adapter pipeline** (packed w/a/g/v sweeps + activations) | `native/lut2_gather.c`, `lut_gemm_fused.py`, `rwkv7_linear.py` | `RWKV_LUT_FUSED_ADAPTERS=auto\|0\|1` |
| Native / Numba LUT gather | `lut_gather_kernel.py`, `native/lut2_gather` | `RWKV_LUT_KERNEL=auto\|c\|numba` |
| LUT2 OpenMP / AVX2 build flags | `native/lut2_gather_loader.py` | `RWKV_LUT2_OPENMP`, `RWKV_LUT2_AVX2` |
| bf16 native gather path | `trinity_decode_fast.py` | `RWKV_LUT_BF16_NATIVE` |
| bf16 shadow (`shadow.bin`) | `decode_shadow.py` | `RWKV_DECODE_SHADOW`, `--bf16-shadow` |
| Shadow-selective pack profile | `pack_profiles.py` | `RWKV_PACK_PROFILE=shadow_sel` |
| Disk decode cache `.decode_cache/` | `decode_disk_cache.py` | `--decode-disk-cache`, `RWKV_DECODE_DISK_CACHE` |
| zlib disk cache (`RDC\x02`) | `decode_disk_cache.py` | `RWKV_DECODE_CACHE_COMPRESS` |
| Layer zlib cache (provider) | `LayerZlibCache` | trinity_layer codec |
| M5 scale_u8 / scale_u4 | `pack_codec.py` | `--pack-codec scale_u8\|scale_u4` |
| **Per-tensor codec routing** (`--codec-map`) | `pack_runtime.py` | `--codec-map "head.weight=scale_u8,default=trinity_lut2"` |
| **Pre-built decode cache** (`build_decode_cache`) | `tools/build_decode_cache.py` | `python -m rwkv_ssd.tools.build_decode_cache --pack./pack` |
| **Runtime codec policy** (`RWKV_CODEC_POLICY`) | `weight_provider.py` | `RWKV_CODEC_POLICY=accuracy\|hybrid\|strict` |
| Trinity XPU decode (experimental) | `trinity_accel.py` | `RWKV_TRINITY_DECODE_DEVICE`, `RWKV_TRINITY_XPU_*` |

### Presets (`throughput_defaults.py`)

| Preset function | Env | Role |
|-----------------|-----|------|
| `apply_streaming_defaults` | (auto) | layer_size I/O, mmap seq, Trinity disk cache |
| `apply_ssd_tier_fused_defaults` | `RWKV_SSD_TIER=1` | ~201 MB strict + fused |
| `apply_ssd_stream_defaults` | `RWKV_SSD_TIER=stream` | strict skeleton + 2-layer provider LRU |
| `apply_partial_ssd_tier_defaults` | `RWKV_PARTIAL_SSD_TIER=1` | hot3 strict partial + fused |
| `apply_partial_hot4_ssd_defaults` | (API) | hot4 variant |
| `apply_partial_fused_defaults` | `RWKV_PARTIAL_FUSED=1` | hot3 + cache (usually slow) |
| `apply_bounded_stream_defaults` | bench / API | 2-layer z, no promote |
| `apply_bounded_fused_defaults` | bench / API | bounded + fused |
| `apply_stacked_defaults` | `RWKV_PACK_PROFILE=shadow_sel` + stream | full promote ~382 MB |
| `apply_stacked_strict_defaults` | bench | shadow + strict reload |
| `apply_low_ram_defaults` | `--low-ram` | thesis bounded path |
| `apply_partial_defaults` | `--mode partial` | auto hot7/hot4 profile |

### Codec router (runtime policy)

| Policy | Env | Effect |
|--------|-----|--------|
| `auto` / `strict` | `RWKV_CODEC_POLICY=auto` | Use manifest `dequant` as-is |
| `accuracy` | `RWKV_CODEC_POLICY=accuracy` | Override head.weight from LUT2 → scale_u8 at runtime |
| `hybrid` | `RWKV_CODEC_POLICY=hybrid` | Prefer shadow bf16 for large mats (>256K elems), LUT for small |

### Forward / compute path

| Mechanism | Where | Notes |
|-----------|--------|-------|
| Per-layer streaming forward | `rwkv7_forward.forward_one_streaming` | inject + TMix/CMix |
| Native `forward` when all blocks in `z` | `forward_one`, `greedy_token_ids_streaming` | after promote or `--warm-z` |
| Fused TMix / CMix / head | `rwkv7_linear.tmix_one_fused`, etc. | strict / bounded paths |
| Prefix state cache (RAM) | `state_cache.py`, `engine` | `--state-cache`, `--system-prefix` |
| SSD session state `.state_cache/` | `state_cache.py` | v0.6.11 with `--state-cache` |
| Ping-pong staging buffer | `staging.py` | large layer payloads |

### Workload gates (P2.e — not multipliers)

| Mechanism | Where | CLI |
|-----------|--------|-----|
| N-gram weight blob cache | `weight_provider._ngram_blobs` | `--ngram-weight-cache` |
| MTP speculative gate | metrics only | `--mtp-speculative` |

### Not in engine hot path (documented only)

| Item | Notes |
|------|--------|
| **Batched TMix fused GEMV** (4×768 one kernel) | `lut2_tmix_gemv_f32`, `lut_gemm_fused.py` | **done (v0.6.14)** |
| **Layer-span fused decode** (no per-tensor bf16 dict for att) | `weight_provider` fused span | **done (v0.6.14)** |
| **FLUTE pack layout** (`TR2\x02` row-padded indices) | `trinity_codec.py` | **done (v0.6.14)** |
| **Partial packed block forward** | `packed_block_forward.py` | **partial (v0.6.14)** — CPU hook; M6 full native |
| **Warm disk cache default** | `RWKV_WARM_DISK_CACHE=auto` | **done (v0.6.14)** |
| **rwkvcpp resident (phase 1)** | `backends/rwkvcpp.py`, ggml FP16 | **done** — CPU native graph |
| **rwkvcpp pack bridge (M5)** | `GgmlWeightBridge` + same F1–F5 presets | **done** — provider/cache/prefetch parity; selective ggml slots remain a follow-up |
| Engine io_uring | `storage_bench/` research (Linux) | **P2** — DNV-6 |
| **M6 Albatross CUDA** | `backends/albatross.py` layer-wise provider adapter | **M6a — wired; CUDA qualification pending** |
| **FLUTE CUDA kernel** | Trinity LUT2 on GPU | **M6b** |
| **GDS / DeepNVMe layer swapper** | NVMe→VRAM streaming | **M6c** — DNV-10 |
| Simulation multipliers | `simulations/` — do not stack into tok/s |
| **Pipelined decode-cache writer** | FastPersist-style overlap | **Open** — DNV-2 |

## RAM scaling model (thesis vs today)

**Does default auto give “initial cost + tiny RAM per extra layer/params”?**

**No — unless you use a bounded or strict path.** Recurrent **state** is O(n_layer) and small (~few MB). **Block weight RAM** is either O(1) bounded (~231 MB default auto, ~203 MB explicit bounded on 0.1B), O(hot profile) partial, or O(full model) when you explicitly promote.

| Phase | RAM holds | Scales with |
|-------|-----------|-------------|
| Skeleton | `emb`, `head`, norms, optional hot blocks | Hot profile only |
| Strict / SSD tier | Skeleton (~201 MB) | **O(1)** in `z`; per-token streams all layers from mmap |
| **Default auto stream+cache** (n≥8, promote off) | Skeleton + **1–2 block layers** in `z` + provider warm | **O(1)** ~231 MB on 0.1B |
| **Bounded stream / fused** (`RWKV_BOUNDED_STREAM=1`) | Skeleton + **2 block layers** + fused blobs | **O(1)** ~203 MB on 0.1B |
| Partial hot3 strict | ~262 MB skeleton | O(hot pins); 8 layers/token from SSD |
| **Full promote** | All blocks in `z` (~382 MB) | **O(params)** — same as resident block mats |

**Default Trinity streaming (v0.6.13+):** `RWKV_PROMOTE_FULL_Z=auto` → **no full promote** on n≥8. Provider LRU capped at 2; **`RWKV_WARM_DISK_CACHE=auto`** pre-builds `.decode_cache/` when promote is off. Max tok/s: **`RWKV_PROMOTE_FULL_Z=1`**.

```powershell
# Low RAM (default on 0.1B)
python -m rwkv_ssd.runtime.engine --model pack --checkpoint ckpt --backend chatrwkv

# Max tok/s (~382 MB z)
$env:RWKV_PROMOTE_FULL_Z="1"

# Explicit bounded + fused
$env:RWKV_BOUNDED_STREAM="1"

# Pin hot layers, strict stream middle
$env:RWKV_PARTIAL_SSD_TIER="1"

# Tiered pack: hot FP16 in skeleton, cold LUT2 on SSD
python -m rwkv_ssd.tools.build_tiered_pack --input model.pth --output pack --resident-layers 0,1,2
$env:RWKV_PACK_PROFILE="tiered_hot3"
$env:RWKV_PARTIAL_SSD_TIER="1"
```

## Core rule (do not stack speedups)

One **scheduler**, one **SSD bandwidth budget**, one **forward pass** per token.

| Kind | Compose how | Example |
|------|----------------|---------|
| **Pipeline stages** | Sequential stages of the same pass | read → dequant → matmul; overlap hides latency (\(\le 1\) wall-clock factor) |
| **P2.b codec/layout** | **Pick one** per deployment | Trinity vs rwkv_lightning FP5 vs HQQ4 pack — not additive |
| **P2.c / P2.d topology** | Usually orthogonal; still one budget | NUMA pinning, second NVMe, hedged reads |
| **P2.e speculative** | **Workload gate** | MTP, n-gram cache — only if prefill/TTFT dominated |

**Forbidden:** multiplying simulation “2×” fields from `simulations/benches/*.py` into one headline `tok/s`.

## Actionable ideas (in-repo only)

Ideas that are **actionable in this repo** (not 70B projections or CSD hardware). The thesis research notes were removed in the June 2026 cleanup; the engine backlog is in [`IDEAS.md`](IDEAS.md).

| Claim | Engine interpretation | Status | CLI / code |
|------------------------|----------------------|--------|------------|
| O(1) state → **sequential layer streaming** (not QD1 random) | Per-token layer sweep from `weights.bin`; golden vs resident | **done** | `--mode streaming` |
| **Primary metrics** are BW, compression, decompression — `tok/s` derived | Bench reports `vs_resident`, `z_mb`, `prov_mb`, layer CSV | **done** | `bench/bench_throughput.py`, `--metrics-csv` |
| **Micro-pipelining** overlaps read/decode/compute *within* a layer (\(\le 1\) factor) | Chunked reads + `read_ms`/`compute_ms` | **done** | `--io-chunk-policy layer_size` (auto on streaming load) |
| **Layer-aware / gate prefetch** hides submit latency on same I/O stream | Planners + `prefetch_overlaps` | **done** | `--prefetch-policy layer_aware` (default) |
| **Sequential layout** (Ch.10) — large aligned layer reads | `layer_grouped` pack + **single `read_bytes_span` per layer** | **partial** | `pack_runtime --pack-layout layer_grouped`; contiguous load in `weight_provider` |
| **Compression Trinity** (~10× storage) | `trinity_lut2` / `trinity` (LUT2 + zlib); M5 `scale_u4` ladder | **started** | `--pack-codec trinity`; |
| **Thermal duty cycle** — fewer bytes → less SSD active time | Document burst vs sustained in benches; no fake “2× tok/s” | **doc / sim** | `thermal_duty_cycle_bench.py` (sim only) |
| **Bounded RAM** — globals in RAM, blocks streamed/evicted | `max_layers_in_z`, provider LRU, skeleton | **done** | `--stream-layer-cache`, `--max-layers-in-z` |
| **Decoupled provider cache** — `z` eviction ≠ decoded tensor eviction | Token 2+ reuses `_prepared_layers` / provider RAM | **done** | default on; `--no-decouple-provider-cache` |
| **Disk decode cache** — `.decode_cache/` bf16 per layer (Trinity + shadow) | Skip LUT on revisit; SSD trade | **done** | auto on Trinity streaming; `--decode-disk-cache` |
| **Low-RAM preset** | bounded z + decouple + disk cache | **done** | `--low-ram` |
| **RAM budget (200B-class)** | partial + pin early layers; rest on SSD | **done** | `--ram-budget-gb 10`; `RWKV_RAM_BUDGET_GB` |
| **RAM budget auto-profile** | Auto-select F1/F3/F5 tier from `ram_budget_gb` | **done** | `apply_ram_budget_to_config` + engine load path |
| **Native forward when all blocks in `z`** | Avoid per-layer inject when complete (incl. tiny 2L models) | **done** | auto `max_z=n_layer` if ≤2 layers |
| **Provider LRU by bytes** | Cap decoded RAM under budget | **done** | `max_provider_cache_bytes`; `--ram-budget-gb` |
| **Fused LUT GEMV** (att 4×768 + FFN + head) | Skip bf16 gather in strict / bounded paths | **done** | `RWKV_LUT_GEMM_FUSED=1`; `tmix_one_fused` |
| **SSD tier / partial fused presets** | ~201–260 MB `z` compromises | **done** | `RWKV_SSD_TIER=1`, `RWKV_PARTIAL_FUSED=1` |
| **Provider cache dedup** | Drop raw `_cache` after prepare | **done** | `prepare_layer_for_z` |
| **Warm disk cache at load** | `.decode_cache/` pre-built when promote off | **done** | `RWKV_WARM_DISK_CACHE=auto` |
| **Bounded promote default (n≥8)** | `auto` no longer fills full `z` | **done** | `RWKV_PROMOTE_FULL_Z=1` for max tok/s |
| **Tiered pack (hot FP16 + cold LUT2)** | `build_tiered_pack` | **done** | `RWKV_PACK_PROFILE=tiered_hot3` |
| **SSD session state cache** | `.state_cache/` prefix persistence | **done** | `--state-cache` (auto with pack dir) |
| **Recurrent MTP** (~2.5× sweep) | Workload gate only; no draft model | **gate only** | `--mtp-speculative` |
| **Hedged reads** (tail latency) | Optional dual-read | **done** | `--io-hedged` |
| **Mmap sequential advise** | Layer sweep hint to OS/FTL | **auto streaming** | `--mmap-sequential` (default on streaming load) |
| **io_uring ceiling** (Ch.12) | Rust `storage_bench/` anchor; engine uses mmap/pread | **measured elsewhere** | not stacked into engine `tok/s` |
| NUMA / 2nd NVMe / CSD / Engram semantic prefetch | Out of engine path for now | **future** | see IDEAS P2.c |

**Bottleneck labeling (bench):** do **not** trust layer `read_ms` alone on mmap (OS cache hides I/O). Use **`tok/s ÷ resident tok/s`** (`vs_resident`) first; layer timers second.

## Measured today (engine — Windows CPU)

**Bench defaults (v0.6.4+):** `bench/bench_throughput.py` uses a small
operator-prepared pack; `--heavy` switches to the operator's larger reference
pack for realistic runs.

| Scenario | Metric | Approx value | Command / test |
|----------|--------|--------------|----------------|
| Synthetic resident | tok/s (32 tok) | ~9700 | `bench/bench_generate.py --model./demo_pack` |
| Synthetic partial | tok/s | ~4250 | same |
| Synthetic streaming | tok/s | ~2590 | same; ratio ~2.7× slower than resident |
| Real pack disk read | MB/s | ~1.6 GB/s (mmap) | `bench/bench_io_backends.py --model C:\prepared\reference-pack` |
| ChatRWKV 0.01B resident | tok/s | ~130–230 | `bench/bench_throughput.py --backend chatrwkv` |
| ChatRWKV 0.01B streaming+cache | tok/s | ~76% of resident | native when all 2 layers in `z` |
| ChatRWKV 0.01B streaming+warm-z | tok/s | ~99% of resident | `--warm-z`; full blocks in `z` |
| ChatRWKV 0.01B strict streaming | tok/s | ~31% of resident | inject path every layer |
| M5 `scale_u4` on a small pack | storage | ~3.9x smaller than none | `eval_m5_codec --build` -> operator-selected output directory |
| M5 `scale_u8` on 0.01B pack | storage | ~2.0× smaller than none | same |
| ChatRWKV 0.1B stream+cache (`max_z=2`, decoupled, v0.6.7) | tok/s | FP16 **15.4** / trinity_lut2 **15.2** / shadow **17.0** | `bench/bench_tok_s_compare.py`; was ~5.4 / ~1.4 / ~2.4 |
| ChatRWKV 0.1B FP16 grouped stream+cache (Jun 2026) | tok/s | **~27** | `fp16_grouped_0.1b`; promote warm |
| ChatRWKV 0.1B FP16_default stream+cache (Jun 2026) | tok/s | **~30** | `runtime_pack` default layout |
| ChatRWKV 0.1B trinity_lut2 stream+cache (Jun 2026) | tok/s | **~25** | layer-span batched decode; no shadow |
| ChatRWKV 0.1B trinity_lut2+shadow stream+cache (Jun 2026) | tok/s | **~27** | shadow.bin |
| ChatRWKV 0.1B trinity_lut2 strict streaming (Jun 2026) | tok/s | **~1–3** | `.decode_cache/` warm; decode-bound |
| ChatRWKV 0.1B FP16 grouped prefix(warm) (Jun 2026) | prefill / tok/s | `state_cache_hit=true`; **~27** tok/s | `--state-cache` + `--system-prefix` |
| ChatRWKV 0.1B FP16 stream+cache (promote, Jun 2026 bench) | tok/s | **~12–15** strict pack / **~27** grouped when promote hits | `bench/bench_tok_s_compare.py`; use `runtime_pack` for 0.1B |
| ChatRWKV 0.1B strict-streaming B=2 shared-layer decode (Jul 2026) | aggregate tok/s | **1.89 end-to-end / 2.75 decode-only**, versus 1.12 / 1.37 independent | `bench/bench_chatrwkv_weight_stationary.py`; 8 tokens/session, exact greedy parity, ~0.73 s/token/session unchanged; page-cache/scheduler evidence only |
| ChatRWKV 0.01B FP16 stream+cache | tok/s | **~109** (grouped, 32 tok) | all layers fit `max_z` |
| Prefix state cache (warm) | prefill | skips system prefill on repeat | `--state-cache` + `--system-prefix` |
| ChatRWKV 0.01B stream+cache | tok/s | FP16 **212** / trinity **208** / shadow **204** | tiny model: all layers fit `max_z` → native forward |
| Skeleton `model.z` | RAM | globals only in strict streaming | `--mode streaming` |
| Bounded cache | RAM | globals + N layers (`--max-layers-in-z`) | `--stream-layer-cache` |
| Full z preload | RAM | ~full model | `--warm-z` (opt-in) |
| Streaming golden | tokens | 8 + 32 greedy match resident | `tests/test_real_model_streaming.py` |

Report new numbers in README footnote + CHANGELOG when hardware or pack changes.

## Modeled in thesis / simulations (not engine CI)

Treat as **hypotheses** until promoted via an afternoon experiment in [`IDEAS.md`](IDEAS.md).

| Mechanism | Claimed effect (thesis) | Bench script | Engine promotion gate |
|-----------|-------------------------|--------------|------------------------|
| Compression Trinity | ~10× storage vs FP16 | `compression_trinity_bench.py` | IDEAS experiment: ≥1.3× storage vs borrowed codecs, ≤10% dequant regression |
| Micro-pipelining (K=8–16) | Overlap read/decode/compute within layer | `micro_pipeline_bench.py` | Chunked `weights.bin` reader + `read_ms`/`compute_ms` in CSV |
| Layer-aware prefetch | Between-layer lookahead | `layer_aware_prefetch_schedule_bench.py` | `lookahead_ms` column; shares budget with P0#3 |
| Gate-based prefetch | Hide SQE / submit latency | `gate_prefetch_bench.py` | Predictor ms vs bubble reduction |
| Recurrent MTP | ~2.5× accepted tok/sweep (k=6) | `mtp_ssd_speculation_bench.py` | Prefill/decode-dominated workload only |
| Heterogeneous chunk schedule | Alt. to uniform K | `heterogeneous_chunk_schedule_bench.py` | A/B in metrics CSV — not additive with micro-pipeline |
| Channel-aligned pack layout | Higher effective MB/s | (design) | `bench_io` delta vs `layer_grouped` pack |
| Thermal duty cycle | Sustainable BW cap | `thermal_duty_cycle_bench.py` | Document cap in bench tables |

## P2 roadmap (engine implementation order)

Aligned with [`IDEAS.md`](IDEAS.md) subsections.

### P2.a — Latency hiders (same SSD budget)

1. Layer-aware prefetch — **done (v0.4.6–v0.6)** — default `layer_aware`.
2. Within-layer micro-pipelining — **done** — auto `io_chunk_policy=layer_size` on streaming load.
3. Gate-based prefetch — **done** — `gate` + `layer_aware` planners.
4. **Single-read contiguous layer load** — **done (v0.6.5)** — when pack offsets are contiguous.
5. SSM speculation cache — not on engine path.

### P2.b — Bandwidth multipliers (mutually exclusive)

1. M5 quant ladder — **partial** — `scale_u8` + `scale_u4` codecs.
2. Compression path comparison + optional Trinity — **one codec wins**.
3. NAND channel-aligned `weights.bin` — **partial** — `layer_grouped` + sector padding in `pack_runtime`.

### P2.c — Schedule / topology

1. Hot-layer pinning + residency profiles — **partial** — `deploy/rwkv7_0.1b_partial*.json`.
2. Engine I/O backends — **done** — `mmap` | `pread` | `threaded`.
3. Second NVMe / NUMA / micro-batching — bench first (not on engine path).
4. Auto streaming defaults (`throughput_defaults.py`) — **done** — layer_size chunks + mmap sequential.

### P2.d — Reliability / tail

Erasure coding, hedged reads, SMART — production; does not multiply steady-state tok/s.

### P2.e — Speculative amplifiers (conditional)

MTP, n-gram weight cache — **done (gate/cache)** — `--ngram-weight-cache`; `--mtp-speculative` gate only.

## Next engine work (ordered)

### CPU / streaming (close F1–F4 vs F5)

1. **P1.4 complete** — full RWKV block forward on packed weights without bf16 inject (`packed_block_forward.py` → all layers). **Shipped v0.6.15:** packed block now also runs when the small skeleton lives in `provider._prepared_layers` (unlocks partial packs, not just strict-fused).
2. **DNV-2/3** — pipelined `.decode_cache/` writer + mmap steady blobs. **Shipped:** DNV-2 telemetry (`cache_write_submits` + `cache_write_sync_ms`); DNV-3 per-file mmap cache in `DecodeDiskCache`; async `ThreadPoolExecutor` writer (2 workers) with FastPersist-style submit-then-go semantics.
3. **DNV-11** — optional prebuilt `.decode_cache/` in distribution (eliminate token-1 storm). **Shipped:** `build_decode_cache --print-summary` and `--verify-only` flags; pre-built cache ships beside packs.
4. **Per-tensor codec routing** — `--codec-map` for hybrid packs (e.g. head=scale_u8, body=trinity_lut2). Measure quality vs PPL. **Shipped** at pack build time.
5. **Codec policy** — runtime `RWKV_CODEC_POLICY=accuracy|hybrid` for decode-path selection without repacking. **Shipped:** `_codec_policy_entry_overrides` routes sensitive layers to shadow sidecar; `accuracy` policy covers head/lm_head/output.
6. **RAM budget auto-profile** — set `RWKV_RAM_BUDGET_GB=N` and the engine picks the right F1/F3/F5 preset automatically. **Shipped:** `select_ram_budget_tier` + `apply_ram_budget_tier`; `0.5 GB → F5`, `0.21 GB → F2`, `0.15 GB → F1`.
7. Measure with **`bench/bench_io_ceiling.py --heavy --full`** (`staging_ms` → 0 steady after warm).

### GPU (M6 — primary tok/s)

5. **M6a** — CUDA resident baseline (Albatross or rwkv_lightning); greedy parity vs ChatRWKV CPU F6.
6. **M6b** — Port **FLUTE-style** fused LUT2 GEMM to CUDA (Trinity 2-bit + 4-entry codebook); target ≥2× vs dequant→GEMM.
8. **M6e** — CUDA graphs when promote fills VRAM.

### Platform / research

9. **Linux** — `storage_bench` io_uring learnings → optional engine backend (DNV-6); only if I/O-bound profile.
10. **Intel iGPU** — continue XPU Trinity decode experiments (M6d); not production bar.

### Shipped (reference)

**Shipped v0.6.14:** Trinity overhead P0/P1 — batched TMix native kernel, warm-cache default, layer-span fused registration, FLUTE `TR2\x02` layout, partial `packed_block_forward`, bench profiles (`quick` / `default` / `full`).

**Shipped post-v0.6.14:** Per-tensor codec routing (`--codec-map`), `build_decode_cache` tool, runtime codec policy (`RWKV_CODEC_POLICY`), RAM budget auto-profile.

## Milestone mapping

| Milestone | Throughput focus |
|-----------|------------------|
| **M3** (done synthetic) | Golden streaming; metrics CSV; ≤3× streaming vs resident |
| **V1** (done 0.1B) | Real pack streaming + skeleton load |
| **M5** | P2.b quant ladder + Compression Trinity experiment |
| **M6a–c** | GPU compute + SSD→GPU streaming (FLUTE CUDA, GDS) |
| **v0.6.14** | Trinity overhead P0/P1 + bench profiles |
| **Post-M5** | Quant codec in `dequant.py`; Linux io_uring prefetch from `storage_bench/` learnings |

## How to report performance in docs/PRs

1. **Mode** — resident / partial / streaming / streaming+cache.
2. **Hardware** — CPU/GPU, drive, pack size, pack layout (`default` vs `layer_grouped`).
3. **Breakdown** — `read_ms`, `staging_ms`, `compute_ms` per layer from `--metrics-csv`.
4. **Ratio** — `tok/s ÷ resident tok/s` (**required**); not multiplied with codec or MTP.
5. **RAM** — `z_mb` and `provider_mb` separately.
6. **Label** — “measured engine” vs “thesis simulation”.

## See also

- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — current focus; the *what is done now* companion.
- **[`GOALS_AND_MILESTONES.md`](GOALS_AND_MILESTONES.md)** — M0–M8 acceptance bars.
- **[`V1_STREAMING.md`](V1_STREAMING.md)** — ChatRWKV streaming code map.
- **[`IDEAS.md`](IDEAS.md)** — afternoon-experiment backlog.
- **[`PRESETS.md`](PRESETS.md)** — preset env vars and partial profiles.
- **[`BASELINE_BUGS.md`](BASELINE_BUGS.md)** — F-1/F-2/F-3 fix log; bench evidence per fix.
- **[`BENCH.md`](BENCH.md)** — bench methodology.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — multi-SSD sharded pack, mmap huge pages, and the full SSD knob survey (M-class).
- **[`../CHANGELOG.md`](../CHANGELOG.md)** — release history.
> **rwkvcpp bridge (M5):** `rwkv_ssd/runtime/ggml_weight_bridge.py` now wires
> the shared ManifestWeightProvider, F1-F5 presets, Trinity/shadow decode,
> provider/disk caches, and non-blocking prefetch into the native ggml graph.
> The first bridge keeps the converted `.bin` graph resident; selective ggml
> weight slots are a separate RAM-reduction follow-up.
