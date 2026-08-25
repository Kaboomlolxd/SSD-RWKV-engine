# Milestone status (roadmap ledger: dev CPU/iGPU · long-term target GPU + SSD)

The long-term roadmap target remains GPU inference with SSD-backed weights.
That target is not the current release posture; the current release is
CPU-first and is defined in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Current release gate (last verified August 24, 2026)

The complete local Python gate passed **577 tests**, with 18 skipped. The run
used `python -m pytest -q --override-ini "addopts="`, so it included marked
parity and serving checks. Native qualification must be rerun against the
newly public upstream submodule pin.
This ledger keeps historical milestone numbers below; [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
is the current recommendation source.

Current verification: **August 24, 2026** — full local Python suite **577
passed, 18 skipped**.

The **July 28, 2026** result of **634 passed, 22 skipped** and the earlier
**7/7** native result are historical snapshots, retained below only for
traceability. Accelerator paths remain hardware-gated.

**Historical post-remediation snapshot (July 23, 2026; 0.1B scope):** CPU
safety and deployment fixes from the engine evaluation were implemented at
that point. The current 2.9B selector and release boundary are maintained in
`PROJECT_STATUS.md`; the paragraph below is retained as a dated ledger entry.
Historical
`trinity_lut2_*` payloads were removed after failing real-model quality gates.
Automatic production resolution uses the checked-in
mixed `trinity_grouped_0.1b` scale-U8 grouped g256 compact pack only when it
has a passing certificate: 326 large tensors are grouped-U8 and 76
one-dimensional control/normalization vectors are dense BF16. Otherwise the
`trinity_safe_0.1b` FP16/BF16 fallback is selected. Explicit grouped
selection remains diagnostic and fails preflight without its certificate. The
corrected grouped pack passes the current short real-model parity gate.
rwkv.cpp resident GGML is the default fast CPU path, and its provider-backed
native layer-local ABI now powers the bounded F1-F4 acceptance path. The 2.9B
g32 grouped-U8 pack is now the default compact selector and carries a
short-smoke certificate from the corrected three-prompt quality gates, but
long-run quality qualification remains open. CUDA, XPU, MPS, Albatross, and other
accelerator claims remain hardware-gated.

**Production target:** **GPU inference** (CUDA datacenter / consumer NVIDIA) with **weights on SSD** when VRAM is bounded. Current dev machine lacks discrete NVIDIA GPU; **CPU + Intel iGPU** is the correctness and preset-tuning path. GPU wins (M6, FLUTE CUDA, DeepNVMe GDS) are **first-class milestones**, not optional.

## Current CPU implementation status (July 31, 2026)

The CPU parity and low-RAM portions of the current plan are implemented and
green on the checked-in fixture. ChatRWKV resident/pack streaming and
rwkv.cpp resident/provider modes share a four-way conformance harness with
exact greedy token/text parity, configurable top-10/KL/state guardrails,
state restore, follow-up generation, sampling determinism, cancellation, and
capability-specific failures. The native rwkv.cpp layer path keeps a bounded
layer plan and reports owned provider bytes separately from file-backed mmap
bytes.

The native F-tier gate passes both cold and warm measurements: on the 0.1B
fixture F1-F4 are 50.06-53.31 tok/s versus 53.07-54.15 tok/s for F5; on the
larger 2.9B grouped-U8 diagnostic F1-F4 are 1.90-2.25 tok/s and remain inside
the explicit 12 GB RSS, 1 GiB provider, and 512 MiB native-layer budgets.
The 2.9B result is not a quality promotion.

HTTP serving now has a bounded spawned process-worker pool, versioned state
envelopes, worker health/restart handling, cancellation acknowledgements, and
aggregated queue/latency/RSS metrics; the default remains one worker.

## July 15 architecture slice

DeepEmbed variant detection now distinguishes the qkv/DEA sidecar contract
from RWKV7a DeepEmbed-v1. The latter uses upstream ChatRWKV's
`RWKV_DE_VERSION=1` path and is supported for resident inference and CPU
layer streaming; qkv/DEA now has a sidecar-backed CPU reference stream.
Verification for this slice includes DeepEmbed/capability tests and the real
RWKV7a resident vs
streaming parity gate; see
[`RESEARCH_AND_ARCHITECTURE.md`](RESEARCH_AND_ARCHITECTURE.md).

## Historical roadmap snapshot (June 27, 2026)

**Historical product snapshot:** V1 streaming correctness and the Trinity
overhead P0/P1 work were done at that point. The old F-tier rates and the
"close the F1-F4 gap" wording below are retained as an audit trail; the
current native rwkv.cpp F-tier acceptance results are recorded above and in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

The July 10 implementation slice closed the Pareto cache-format selector,
byte-bounded cache accounting, validated/coalesced striped layer reads, CUDA
staging safety, quantization quality gates, activation telemetry, the repeated
frontier benchmark, and a Windows-safe storage diagnostic. Remaining work is
hardware-gated or quality-gated: M6 GPU/GDS validation, physical 2.9B/7B+
measurements, and new quantization codecs without measured quality evidence.

| Track | Status |
|-------|--------|
| M3 / V1 golden (FP16 streaming == resident) | **Done** |
| P2.a prefetch + micro-pipeline + madvise | **Done** |
| P2.c layer cache, partial profiles, `layer_grouped` | **Done** |
| **Trinity overhead P0/P1** (v0.6.14) | **Done** — batched TMix GEMV, warm disk cache default, FLUTE `TR2\x02` layout, `packed_block_forward`, zlib engine guard |
| **Fused LUT GEMV + SSD/partial presets (v0.6.12–0.6.13)** | **Done** |
| Trinity codecs on engine path | **Live** — LUT decode still caps F1–F4 tok/s until fuse/promote/cache |
| **Streaming tok/s (FP16 + Trinity promote)** | **Done on CPU** — **15-20 tok/s 0.1B** (F5/Fb Pareto winner, post F-1/F-2/F-3). The win is the **compute path** (native forward via `model.forward()`), not SSD bandwidth — the 49 MB pack fits in the OS page cache after warmup. See "What F5/Fb actually wins on" below. |
| **F5 ≥ F6** (F5 ~17 vs F6 ~13 tok/s) | **Done on CPU** — F5 wins because the warm-z-to-full-z path avoids the per-layer Python dispatch that F6 hits when reading from `.pth`. |
| **F1 prefetch skip** (no-cache path) | **Done (June 29, F-1)** — F1 from broken to 5.5-7.2 tok/s on 0.1B by skipping `prefetch_ahead` when `stream_layer_cache=False` |
| **F2 non-blocking prefetch** (cache path) | **Done (June 29, F-2)** — `begin_layer` / `prefetch_entries` no longer block on the in-flight prefetch future. F2 → 6.5-8.3 tok/s; F2 is now a valid Pareto point at low RAM. |
| **F3 provider cache size** | **Done (June 29, F-3)** — five tier defaults hardcoded `max_provider_cache_layers=2` (16.7% hit rate on 0.1B; 56.8 ms/tok of bridge). Removed the hardcodes; `resolve_max_provider_cache_layers` now picks full cache for n>=8 decoupled. |
| **M-class sharded pack** (multi-SSD) | **Done (June 29)** — `python -m rwkv_ssd.tools.shard_pack --shards N` splits the pack into N files. `ShardedWeightStore` reads in parallel via `ThreadPoolExecutor`. For 0.1B the pack fits in page cache (no speedup); for 7B+ where the SSD is the bottleneck, K SSDs give up to Kx aggregate read bandwidth. [`docs/SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md) |
| **Manifest-v2 striped sharding** | **Done on CPU (July 10)** — exact logical/physical validation, adjacent extent coalescing, provider integration, and BF16 shadow shard writing/gather/decode. Physical multi-SSD scaling remains hardware-gated. [`docs/SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md) |
| **Explicit cache-format/F profiles** | **Done (July 10)** — `cache_format`, packed/prepared byte caps, and expanded cache telemetry separate dense `z`, prepared tensors, packed LUT blobs, and LUT2 indices. |
| **CUDA event-owned staging ring** | **Foundation (July 10)** — three-slot CUDA ring with event ownership; provider overlap consumer is gated for the GPU benchmark slice. |
| **CMix selective reads** | **Done on CPU (July 10)** — opt-in telemetry plus an explicit row-tiled value-matrix sidecar, exact selective matmul, and tile/byte read metrics. Existing dense/LUT packs safely fall back. |
| **Adaptive residency retiering** | **Done on CPU (July 10)** — window, hysteresis, minimum dwell, change cap, explicit-format override, and request-boundary provider rebuild. |
| **Streaming matrix + storage diagnostic** | **Done on CPU (July 10)** — warmups/repeats, median/p95 tok/s, stage timings, cache events, JSON/CSV, `correctness_only`, and JSON diagnostics for single/layer/striped packs. Artificial caps are never reported as physical scaling. |
| **Real ChatRWKV weight-stationary batching** | **Done on CPU (July 12)** — dense layer-outer/session-inner decode with exact greedy parity. Clean two-session/8-token diagnostic: **1.12→1.89 aggregate end-to-end tok/s (1.69x)** and **1.37→2.75 aggregate decode-only tok/s (2.01x)**; per-session latency unchanged at about 0.73 s/token. Short requests remain prefill-bound. |
| **DeepEmbed variant-aware packing** | **Done on CPU (July 15)** — qkv/DEA emits `DeepEmbed.bin`; RWKV7a-v1 keeps `s_emb`/`s_emb_x` in packed tensors and records `deepembed_streaming_supported=true`. |
| **RWKV7a DeepEmbed-v1 resident + CPU streaming** | **Done on CPU (July 15)** — native ChatRWKV `RWKV_DE_VERSION=1`, skeleton/provider row derivation, greedy parity for `"Hi"` under the shared prefill/decode contract. |
| **qkv/DEA DeepEmbed CPU streaming** | **Done as a reference path (July 15)** — sidecar rows plus provider-loaded ordinary layers match the resident reference; fused production path remains open. |
| **F1–F4 vs F5 gap** | **Closed for the native rwkv.cpp gate** — the current 0.1B and 2.9B cold/warm acceptance matrices meet the requested ratios and memory budgets. The older ChatRWKV/Python-dispatch figures in this historical table remain useful for explaining why the native layer-local path was required. |
| M2.5 prefix state cache | **Done** |
| Low-RAM / 200B thesis path | **Partial** — `--ram-budget-gb`, partial hot3; byte LRU shipped v0.6.11. **Open:** over-throttles on small models (F1-F3 on 0.1B report <F4/F5). |
| **M6 GPU compute** (Albatross / FLUTE CUDA / rwkv_lightning) | **Not started** — **next major track** |
| **rwkvcpp phase 1** (resident ggml CPU) | **Done** — DLL wired, `bench_backend_compare.py`, ~59 tok/s 0.1B resident |
| **rwkvcpp streaming (M5)** | **Done for the native layer ABI** — bounded layer plan, dense/grouped-U8 uploads, chunked prefill, provider scheduling, and F-tier metrics are implemented; longer 2.9B quality certification remains open. |
| **M6a-M6c + sharded pack** (multi-SSD) | 7B+ packs on multi-socket with NVMe bifurcation | M6-class future |
| **M6b GPU↔SSD** (GDS / DeepNVMe patterns, layer swapper) | **Research** — |
| Intel iGPU / XPU Trinity decode | **Experimental** — `RWKV_TRINITY_DECODE_DEVICE=xpu` |
| M7 HTTP serve | **Done** — bounded spawned workers, state envelopes, cancellation, restart, and aggregated metrics |
| Bench profiles (quick / default / full) | **Done** — `bench/_bench_profiles.py` (with `FULL` constant fix) |

Presets: [`PRESETS.md`](PRESETS.md) · Trinity overhead: · DeepNVMe: · Benches: [`BENCH.md`](BENCH.md) · **Baseline bugs:** [`BASELINE_BUGS.md`](BASELINE_BUGS.md).

---

## Deviations from the original plan

These are intentional shifts discovered while closing the streaming throughput gap:

| Original assumption | What we learned | Current approach |
|-------------------|-----------------|------------------|
| Streaming is ~3× slower than resident; optimize later | Per-token **re-inject** of all layers dominated tok/s (~5–15 vs ~27 resident on 0.1B) | **Decoupled provider cache** + **`promote_stream_cache_to_full_z`** — after warm, FP16 streaming ≈ resident |
| `--warm-z` = only way to get native `forward` | Full preload at start uses too much RAM for thesis | **Promote on warm decode** — grow `z` retention after provider cache fills |
| State cache = serving win for HTTP first | Prefix cache skips **prefill**, not per-token weight loads | **`--state-cache` + `--system-prefix`** on ChatRWKV streaming; biggest win on repeated system prompts |
| Shadow = optional bench trick | LUT2 decode is the Trinity tok/s bottleneck | **Auto `RWKV_DECODE_SHADOW=1`** when `shadow.bin` exists; trades disk for decode speed |
| Throughput phase “complete” at v0.6.0 | Trinity LUT2 still far from FP16 without shadow/promote | Extended **v0.6.7–v0.6.10**; `tok_s_compare.json` is **stale** (pre-promote) |
| `max_layers_in_z` = layer count LRU | 200B needs **GB budget**, not layer count | **`--ram-budget-gb 10`** pins early layers; stricter byte LRU still TODO |
| Compression Trinity = storage win | zlib layer bundles kill decode on 0.1B | Ship **`trinity_lut2` + shadow** or **`.decode_cache/`**, not zlib `trinity_layer` on fast SSD |
| **CPU-only thesis framing** | Product is **GPU inference**; CPU path is dev + low-RAM research | M6 + GDS are **primary** tok/s track; CPU presets validate streaming correctness |
| **Trinity tok/s = disk size** | Decode ~290 ms/layer dominates read ~0.5 ms on NVMe | — fuse / cache / promote |

---

## Shipped milestones

| Milestone | Item | Status |
|-----------|------|--------|
| **M0–M4** | CLI, benches, residency, daily driver | Done |
| **M3** | Synthetic streaming golden | Done |
| **V1** | Real RWKV-7 pack streaming + skeleton load + golden | Done |
| **M2.5** | Prefix state cache (synthetic + ChatRWKV) | Done (v0.6.10) |
| **M5** | `scale_u8`, `scale_u4` codecs + eval gate | Done |
| **M7** | HTTP serve (`app/serve.py`) | Done |

### Release notes (v0.6.2 → v0.6.10)

| Version | Highlights |
|---------|------------|
| **v0.6.2–v0.6.5** | Contiguous layer read (`read_bytes_span`); auto `io_chunk_policy=layer_size`; native `forward` when all blocks in `z` (≤2-layer packs) |
| **v0.6.6** | `max_z=2` default for n≥8 + stream cache; auto `partial` hot7 profile; `repack_bench_pack` |
| **v0.6.7** | Decoupled `z` / provider eviction; Trinity `.decode_cache/`; shadow disk cache; `--low-ram` |
| **v0.6.8** | `--ram-budget-gb` partial planner (`RWKV_RAM_BUDGET_GB`); caps provider LRU under budget |
| **v0.6.9** | `promote_stream_cache_to_full_z` — FP16 streaming ≈ resident after warm (~27 tok/s on 0.1B vs ~15 pre-promote) |
| **v0.6.10** | ChatRWKV prefix state cache; shadow default when `shadow.bin` present; inject skip when weights already in `z` |

| **v0.6.14** | Trinity overhead P0/P1 — batched TMix GEMV, warm-cache default, layer-span fused, FLUTE layout, packed block forward, bench profiles |

See [`CHANGELOG.md`](../CHANGELOG.md) for full history.

---

## Trinity overhead backlog (post-P0/P1)

| ID | Work | Status |
|----|------|--------|
| P0.1–P0.4 | Warm cache default, batched TMix, zlib guard, tiered+partial default | **Done** |
| P1.1–P1.3 | Batched GEMV, layer-span fused, FLUTE pack layout | **Done** |
| P1.4 | Full native block forward on packed weights (M6 hook) | **Partial** — `packed_block_forward.py` on CPU |
| P2.1 | **FLUTE-style CUDA kernel** for Trinity LUT2 | **Open** — M6b |
| DNV-2/3/11 | Pipelined `.decode_cache/` writes, mmap steady, ship cache | **Open** — |

---

## GPU milestone track (M6 expanded)

| Sub-track | Goal | Status |
|-----------|------|--------|
| **M6a** | CUDA resident forward — Albatross / rwkv_lightning / ChatRWKV GPU | Not started |
| **M6b** | **FLUTE fused LUT2 GEMM on GPU** — no full bf16 `W` in VRAM | Not started |
| **M6c** | **GPU layer streaming from SSD** — ZeRO-Inference pattern; overlap fetch L+1 while compute L | Research — DeepNVMe GDS + pinned H2D |
| **M6d** | Intel iGPU / XPU Trinity decode (`RWKV_TRINITY_XPU_*`) | Experimental dev only |
| **M6e** | CUDA graphs / single native forward when layers promoted to VRAM | After M6a |

**DeepNVMe wins on GPU path:** GDS NVMe→VRAM (+10–37% vs bounce), bulk async layer fetch, `ds_nvme_tune` queue depth — see DNV-10 in. Irrelevant for strict LUT on CPU until decode is amortized.

The following are intentionally not marked complete from this Windows
CPU/iGPU run: CUDA/GDS, GPU-fused RWKV kernels, physical multi-SSD bandwidth
scaling, large 2.9B/7B measurements, and large-model or quantizer training.

---

## Remaining work (current CPU scope, July 28, 2026)

1. Extend the 2.9B grouped-U8 quality certificate to multiple prompts and
   longer generations; the current throughput/ABI result is not a quality
   promotion because its KL probe exceeded the configured gate.
2. Continue profiling the native grouped-U8 GEMV, recurrent-state update, and
   vocabulary head so the compact CPU path improves without relaxing parity.
3. Keep the full CPU conformance, native CTest, F-tier, and multiprocess HTTP
   checks in CI when their model/native assets are
   available.
4. GPU, CUDA/GDS, Albatross, and physical multi-SSD scaling remain separate
   hardware-gated work and are outside the current CPU implementation scope.

---

## Measured baselines (dev machine — Windows, Intel iGPU host, CPU forward)

Measured **June 29, 2026** on `trinity_lut2_0.1b` after F-1/F-2/F-3
([`BASELINE_BUGS.md`](BASELINE_BUGS.md)). All numbers are warm
steady-state (engine built once, warmup run, then `samples` measurements);
`--max-tokens 24 --samples 2`.

### The "auto" preset (Fb)

`RWKV_RAM_BUDGET_GB=0.5` calls `select_ram_budget_tier(0.5)` which
returns `"F5"` (since 0.5*1024 = 512 MB ≥ F5's 382 MB floor). The
`apply_ram_budget_tier(tier="F5")` then runs the F5 preset
(`apply_promote_max_defaults` + `warm_z=True`). So "auto" = the
warm-z-to-full-z F5 path. The `Fb` name just labels the origin
(budget-selected) vs. an explicit F5.

### What F5/Fb actually wins on (0.1B, no SSD bottleneck)

The 0.1B pack is 49 MB — fits in the OS page cache after the first
read. After warmup, **all reads are from RAM, not SSD**. So the
F5/Fb tok/s win over F1/F2 is **not** from SSD bandwidth; it's
from the **compute path**:

- **F1/F2:** per-layer Python dispatch (for-loop, dict lookups,
  `inject_layer_into_z`, provider cache retain/evict). The
  per-tensor path has ~120-180 ms/tok of bridge overhead.
- **F5/Fb:** all weights in `z`, `model.forward()` is called once
  per token. The native path avoids the per-layer Python loop.

| Tier | tok/s | RAM | ms/tok | compute | bridge | wall source |
|------|-------|-----|--------|---------|--------|--------------|
| F1   | 5.5-7.2 | 231 MB | 180 | 151 | 0    | per-tensor dispatch |
| F2   | 6.5-8.3 | 241 MB | 136 | 0   | 136  | per-tensor dispatch |
| F5   | 13-19   | 382 MB |  58 |  57 | 1    | **native forward** |
| Fb   | 15-20   | 382 MB |  55 |  54 | 1    | **native forward** |
| F6   | 12-19   | 382 MB |  77 |  76 | 0    | native forward |

**On 0.1B, the F-tier choice is about the Python dispatch, not the
SSD.** The "small RAM" tiers (F1, F2) lose because of per-layer
overhead; the "full RAM" tiers (F5, Fb) win by using the native
forward that ChatRWKV provides.

### What changes for 7B+ (the SSD actually matters)

For 7B+ (~14 GB pack) on a 16 GB RAM system, the pack does **not**
fit in the page cache. The SSD becomes the bottleneck:

- **F1** (per-token re-read) on 7B+: 14 GB / 1.6 GB/s ≈ **9 sec/token**
  — catastrophic. Not viable.
- **F5** (full z) on 7B+ needs 14 GB RAM — only works on 24+ GB
  systems.
- **F3** (~280 MB z on 7B+) is the practical low-RAM tier; ~12 GB
  still streams from SSD per token at ~7 sec/token.
- **Sharded pack** (4 SSDs in parallel on 7B+): 14 GB / 6.4 GB/s
  ≈ **2 sec/token** — the practical 7B+ ceiling without going F5.

**Bottom line:** on 0.1B the SSD is *not* the bottleneck; on 7B+ it
is. The F-tier win on 0.1B is the compute path; the F-tier win on
7B+ is staying in RAM. The `SSD_EXPLOITATION.md` doc covers the SSD
side; this section covers the F-tier / compute-path side.

> **Pre-fix numbers (do not report):** F5 was reported as 3.79 tok/s, F1 as
> 1.04. The bench cold-started on every measurement and the bridge_ms
> formula was wrong. See [`BASELINE_BUGS.md`](BASELINE_BUGS.md) for the
> bug-by-bug writeup and the code fixes.

Use **`tok/s ÷ F5 or F6`** — not layer `read_ms` alone on mmap. Run
**`--max-tokens 24 --samples 2`** (DEFAULT profile) or **`--full`**
(max_tokens=48, samples=3, warmup=1) for production-like numbers.

---

## Key CLI flags (current)

| Flag | Purpose |
|------|---------|
| `--mode streaming` | Per-token layer sweep from `weights.bin` |
| `--stream-layer-cache` | Retain decoded layers in `model.z` (bounded) |
| `--max-layers-in-z N` | LRU cap for block layers in `z` |
| `--no-decouple-provider-cache` | Tie provider eviction to `z` eviction (old behavior) |
| `--max-provider-cache-layers N` | Cap decoded tensor LRU (0 = full model when decoupled) |
| `--decode-disk-cache` | `.decode_cache/` bf16 per layer (auto on Trinity) |
| `--low-ram` | Bounded z + decouple + disk cache preset |
| `--ram-budget-gb 10` | Partial: pin early layers under RAM cap |
| `--state-cache` + `--system-prefix` | Skip system prefill on repeat requests |
| `--warm-z` | Full `z` preload at start (benchmark only) |

**Env presets:** `RWKV_SSD_TIER=1` (minimal RAM + fused) · `RWKV_PARTIAL_FUSED=1` (hot3 compromise) · `RWKV_LUT_GEMM_FUSED=1` · `RWKV_PROMOTE_FULL_Z=0|1`.

Env: `RWKV_RAM_BUDGET_GB`, `RWKV_DECODE_SHADOW`, `RWKV_SSD_SYSTEM_PREFIX` (enables state cache).

---

## Verify

```powershell
pip install -e ".[dev]"
$env:RWKV_JIT_ON='0'

python -m pytest tests/ -q
# Fast (no ChatRWKV checkpoint): default deselects slow markers

python -m pytest tests/test_state_cache.py tests/test_decouple_provider_cache.py `
 tests/test_native_materialize.py tests/test_ram_budget.py -q

python -m rwkv_ssd.tools.check_backends
python -m rwkv_ssd.tools.run_throughput_suite --heavy
python bench/bench_io_ceiling.py --heavy # quick: F1+F3+F5, 12 tok
python bench/bench_io_ceiling.py --heavy --full # all F tiers, warm cache at load
```

Recommended streaming smoke (0.1B):

```powershell
python -m app.cli --model C:\prepared\reference-pack `
 --checkpoint C:\models\rwkv-model.pth `
 --backend chatrwkv --mode streaming --stream-layer-cache `
 --strategy "cpu bf16" --state-cache `
 --system-prefix "You are a helpful assistant." `
 "Hello"
```

## See also

- **[`BASELINE_BUGS.md`](BASELINE_BUGS.md)** — the F-1/F-2/F-3 bug log with bench evidence per fix; this file tracks *what shipped*, that file tracks *what broke and why*.
- **[`PRESETS.md`](PRESETS.md)** — env-var decision tree, preset catalog (F1–F6 + Fb), and the auto-tier mechanism.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog (what each preset does internally) and the no-stacking rules.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — multi-SSD sharded pack, mmap huge pages, and the full SSD knob survey.
- **[`CHANGELOG.md`](../CHANGELOG.md)** — chronological list of every release since v0.6.0.
