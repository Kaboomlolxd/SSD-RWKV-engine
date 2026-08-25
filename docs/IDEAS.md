# Engine ideas backlog

Prioritized for **SSD per-layer streaming during decode**, not re-building rwkv_lightning / web-rwkv / RWKV-Infer.

**Status (July 15, 2026):** the non-hardware-gated CPU architecture slice is
complete for the downloaded small checkpoints. RWKV7a DeepEmbed-v1 has native
ChatRWKV resident support plus CPU layer streaming; real Mamba-2 and
Transformer references have cached-state parity tests. These references are
contracts and correctness tools, not production throughput backends.


**Plan summary (presets, measured vs modeled speedups):** [`PRESETS.md`](PRESETS.md) · [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md) · **SSD research / stacks:** [`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md) · archived timelines in [`../archive/docs/planning/README.md`](../archive/docs/planning/README.md).

## P0 — required for “it works”

1. **Per-layer weight injection (ChatRWKV)** — stream `weights.bin` slice into block parameters each forward.
2. **Golden test** — `resident` vs `streaming` greedy tokens identical on one small RWKV-7.
3. **Prefetch layer N+1** while computing layer N (ping-pong already scaffolded).
4. **`pread` + optional `madvise`** — **Done (v0.5.5):** `--io-backend pread|threaded`; `MADV_WILLNEED`/`DONTNEED` on Linux (`--no-mmap-willneed`, `--mmap-dontneed`).
5. **Artifact verify at startup** — checksum manifest, fail fast on corruption.
6. **HuggingFace source ingestion** — `meta.json` records HF repo ID; `pack_runtime` accepts `--hf-repo` / snapshot path.
7. **Cross-platform async I/O** — **Done (v0.5.4):** `--io-backend mmap|pread|threaded`. ayafileio evaluated and **rejected** for decode (random offset reads ~25× slower than mmap on Windows). Linux next: `madvise(MADV_WILLNEED)` + io_uring prefetch thread.
8. **Prefix / state cache** — repeated system prompts skip prefill SSD work; **RWKV-Infer** and **rwkv_lightning** already ship this — biggest TTFT win in serving; prototype early (see M2.5).

> **Engine status (June 27, 2026):** V1 streaming, Trinity overhead P0/P1, fused presets, M7 HTTP, per-tensor codec routing (`--codec-map`), `build_decode_cache` tool, RAM budget auto-profile, `RWKV_CODEC_POLICY` runtime router, **Trinity LUT2 K-means codebook** (+20 dB SNR vs legacy linspace; Hadamard+kmeans helper module ready for next codec pass). **F1-F5 baseline bug sweep** shipped (7 bugs, see [`BASELINE_BUGS.md`](BASELINE_BUGS.md)) — F5 = 9.01 tok/s, F5 ≥ F6 for the first time on the dev machine. **M6 GPU track** (M6a–c: Albatross, FLUTE CUDA, GDS) is **primary tok/s** path; dev on Intel iGPU + CPU forward. See [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md).
>
> **F1-F3 calibration note:** the F1 (0.15 GB) / F2 (0.21 GB) / F3 (0.5 GB) `ram_budget_gb` tiers are designed for 7B+ models where the resident set is a small fraction of the full pack. On 0.1B (full skeleton ~201 MB) the planner throttles `max_provider_cache_bytes` aggressively and streaming tok/s drops well below F4/F5. `scripts/bench_tok_s.py` skips F1-F3 on 0.1B and uses F4 (`max_z=2`) / F5 (`RWKV_PROMOTE_FULL_Z=1`) for the streaming comparison. The decision tree in `PRESETS.md` documents the full mapping.

> **F1-F5 open work (post June 27, 2026 sweep):** measured warm on 0.1B —
> F1 2.77 / F2 3.09 / F3 3.49 / F4 2.15 / F5 9.01 / F6 3.98 tok/s. F1-F4
> staging 23-45 ms/tok is the bottleneck. Open work: (1) P1.4 packed block
> forward that doesn't need per-tensor bf16 inject — would close the F1-F4
> gap to F5 staging (≤10 ms/tok). (2) F4 = 2.15 < F3 = 3.49 — hot4
> residency profile (`rwkv7_0.1b_partial_hot4.json`) needs investigation,
> likely a pin conflict with the skeleton. (3) Fb (auto-tier) at 0.40 tok/s —
> the planner runs after the tier preset and overrides it; need to call
> `apply_ram_budget_tier` instead of `apply_ram_budget_to_config` for the
> auto path. (4) F6 doesn't record per-layer compute, bridge=100% is
> misleading; add a metrics path. (5) `apply_ram_budget_to_config` over-
> throttles provider cache on n_layer<16; for small models leave it
> uncapped. (6) M6 GPU track — primary tok/s ceiling. See
> [`BASELINE_BUGS.md`](BASELINE_BUGS.md) §"What still needs work" for the
> full list.

## P0.5 — F1-F4 staging elimination (post-baseline-sweep)

> P0.5 is a new tier inserted between P0 and P1. These are the items the
> F1-F4 baseline audit identified as the next concrete engine work to close
> the F1-F4 vs F5 gap. Each item is sized to "afternoon experiment" at most.

- **F-1 — P1.4 packed block forward on F1-F3 (no slab inject).**
 Status: **done (Unreleased)** — `forward_one_streaming` now has an
 early-exit path that calls `forward_block_packed` directly when the fused
 LUT provider is active and the att weights are NOT in `z` (the F1-F3
 strict case). Skips the per-tensor `inject_layer_into_z` step. Helpers
 `_can_use_packed_block` + `_packed_block_step` are unit-tested. `ln_x.weight/bias`
 (which `tmix_one_fused` reads from `z` directly) is mirrored into `z` for
 the streamed layer. Measured staging drop: F1 60.6→43.9 ms/tok (-27%),
 F3 42.3→33.8 ms/tok (-20%). Tok/s is similar on the bench run — the
 staging reduction mostly reclaims Python dispatch overhead that the
 bridge bucket was already absorbing. The bigger win on F1-F3 will come
 from shrinking the Python dispatch cost itself (CUDA graphs on M6e).

- **F-2 — F4 (hot4) regression: 2.15 < F3 (hot3) 3.49 tok/s.**
 Status: **investigated — not a real regression.** The v7 bench has F4
 (3.19) ≥ F3 (3.15) within noise; the prior "F4=2.15" was a specific run
 with higher compute from the strict-fused retain path on the hot4 layer
 set. F4 still dominated by F3t (tiered-hot3-partial) on the v7 run; the
 partial hot4 profile (`rwkv7_0.1b_partial_hot4.json`, pins 0,1,2,11) is
 correct, but the deploy profile needs the same number of hot layers as
 F3 to win. `apply_partial_hot4_ssd_defaults` correctly delegates to
 `apply_partial_ssd_tier_defaults` with the hot4 profile. Marked closed.

- **F-3 — `apply_ram_budget_to_config` over-throttle on small models.**
 Status: **done (Unreleased)** — when `n_layer < 16`, the planner no
 longer sets a `max_provider_cache_bytes` cap. The 0.1B skeleton is
 already 201 MB so any per-layer byte cap throttles tok/s well below
 what docs claim. F4/F5 paths already skipped the cap via the
 tier-applied marker (see F-4). Tested by `test_planner_no_throttle_on_small_models`
 and `test_planner_throttles_on_large_models` (n_layer=32 still gets a
 cap — that's the 7B+ path the planner is calibrated for).

- **F-4 — `Fb` (auto-tier) broken: 0.40 tok/s.**
 Status: **done (Unreleased)** — three changes in `rwkv_ssd/runtime/ram_budget.py`:
 1. `apply_ram_budget_tier` for F5 now sets `config.warm_z = True` (the
 F5 preset only sets `max_layers_in_z = n_block` when `warm_z` is
 already True, so without this Fb ran as F2 with cap=1).
 2. `apply_ram_budget_tier` stamps `config._ram_budget_tier_applied = <tier>`.
 3. `apply_ram_budget_to_config` respects that flag — does not clobber
 `warm_z`/`max_layers_in_z`/`max_provider_cache_bytes`/residency when
 a tier preset has already set them. Fb at 0.5 GB → F5 → z=382.2 MB,
 6.43 tok/s (was 1.9 in the v6 bench, 3.4× improvement, 86% of F6).

- **F-5 — F6 (resident) compute_ms not recorded.**
 Status: **done (Unreleased)** — `engine._generate_resident` passes
 `self.metrics` to `backend.generate_greedy_native(..., metrics=…)`;
 `chatrwkv.generate_greedy_native` accepts the new kwarg; the inner
 `greedy_token_ids_native` already records prefill + per-token decode
 time on the `-1` layer row. F6 row now reports
 `compute_ms_per_token=265.62` (was 0.0) and `bridge_pct=0.0` (was 1.0).

- **F-6 — `apply_bounded_fused_defaults` missing
 `RWKV_DECODE_CACHE_COMPRESS=0`.**
 Status: **done.** `apply_bounded_fused_defaults` already sets
 `RWKV_DECODE_CACHE_COMPRESS=0` at line 330 of `throughput_defaults.py`
 (verified during the F-1 review). Marked closed.

## P1 — product quality

9. **Albatross `faster3a_2605`** — pin this variant after M3; **`rwkv_lightning`** is the higher-level alternative (batch + cache + HTTP bundled).
10. **Residency profiles** — JSON: which layers/tensors stay resident vs streamed; cross-reference P0 state cache (cache = prompt prefix, residency = which weights never leave disk).
11. **Quantization ladder (comparison, not one codec)** — span **rwkv.cpp** Q4_1/Q5_K, **web-rwkv** NF4/INT8, **RWKV-Infer** HQQ4, **rwkv_lightning** FP8/INT8/FP6/FP5/HQQ4; dense FP16/BF16 remains the quality reference while grouped-U8 is the compact default direction (see M5).
12. **HTTP streaming API** — **rwkv_lightning** already has OpenAI-compatible + `/big_batch` + state endpoints; **borrow or wrap**, don’t greenfield unless SSD stream forces it. **web-rwkv** has a runtime API reference too.
13. **State snapshot API** — save/load recurrent state blob for session resume (engine API; distinct from prefix cache).
 - Status: **done (Unreleased)** — `rwkv_ssd.runtime.snapshot` (single-file `RWS\x01` format with engine config + payload + sha256); `engine.save_snapshot(path, prompt=…)` / `load_snapshot(path)` / `from_snapshot(path, pack_dir)`; CLI `--save-snapshot` / `--load-snapshot`. Tests: `tests/test_state_snapshot.py` (7).
14. **LoRA adapter slots** — small resident deltas + streamed base.

## P2 — throughput mechanisms

> Single scheduler, single SSD bandwidth budget. No item here is an independent multiplier on a `tok/s` headline. See the no-stacking rule in [`IDEAS.md`](IDEAS.md) under "Explicitly deprioritized" and the rejection of "stacking simulation speedups" under that section.

> **No item in P2.a–P2.e is an independent multiplier on a tok/s headline.** The defended pipeline is one scheduler with one SSD bandwidth budget. **P2.b** items are mutually exclusive codec/layout choices. **P2.c** and **P2.d** items are usually orthogonal and do not multiply. **P2.e** items are conditional and must declare their workload gate. See thesis §2.5 for the originating discipline.

### P2.a — Latency hiders (overlap/predict within one scheduler)

- **Within-layer micro-pipelining (K=8–16 chunk overlap)**
 - From thesis: Ch.8 (micro-pipelining); `simulations/benches/micro_pipeline_bench.py`
 - Status: **done (v0.5.0)** — `--io-chunk-bytes`, `chunk_reads` in metrics CSV
 - Verify: `tests/test_micro_pipeline.py`
 - No-stack: alternative to heterogeneous chunk schedule (P2.c)
- **Layer-aware prefetch schedule (between-layer lookahead)**
 - From thesis: `simulations/benches/layer_aware_prefetch_schedule_bench.py`
 - Status: **done (v0.5.5)** — `LayerAwarePlanner`, batched prefetch; metrics CSV unchanged
 - Precondition: lookahead budget per layer
 - Verify: same CSV (`prefetch_wait_ms` column)
 - No-stack: shares bandwidth budget with P0 #3
- **Gate-based prefetch (SSM gate signal as cheap predictor)**
 - From thesis: `simulations/benches/gate_prefetch_bench.py`; thesis A5 #2
 - Status: **done (v0.5.5)** — `gate` + `layer_aware`; SSM gate hook still optional future
 - Verify: `tests/test_gate_prefetch.py` (greedy parity)
 - No-stack: default over `ssd_speculative_prefetch_bench.py` (thesis A5 #2)
- **SSM speculation cache (compact state trajectory)**
 - From thesis: `simulations/benches/ssm_speculation_cache_bench.py`; thesis A5 #3
 - Status: inspired
 - Precondition: recurrent-model state serializer
 - Verify: hit rate × state-size cost
 - No-stack: preferred over `ssd_speculative_prefetch_bench.py` for recurrent models

### P2.b — Bandwidth multipliers (mutually exclusive)

> Trinity **tok/s** is decode-overhead bound, not SSD bytes — elimination strategies:.

- **NAND channel-aligned I/O layout (sector packing, channel alignment)**
 - From thesis: Ch.16 (channel-aligned I/O); Appendix B “New idea”
 - Status: new-idea
 - Precondition: SSD geometry profile in `pack_runtime`
 - Verify: `bench/bench_io.py` MB/s delta vs current mmap layout
 - No-stack: alternative to current `weights.bin` layout, not additive
- **Adaptive precision / multi-bitwidth runtime switching**
 - From thesis: Ch.16; `simulations/benches/temporal_weight_locality_bench.py`
 - Status: inspired
 - Precondition: multi-bitwidth pack variants in M5 ladder
 - Verify: per-token quality hit vs bandwidth win
 - No-stack: mutually exclusive with P1 #11 codec choice per layer (router)
- **Compression path comparison (M5 evaluation)**
 - From thesis: `simulations/benches/compression_path_comparison_bench.py`; Ch.6
 - Status: inspired; outcome decides codec
 - Precondition: M5 ladder operational
 - Verify: size + latency + quality table; pick one
 - No-stack: replaces the codec choice, does not stack
- **Compression Trinity (thesis baseline codec stack)**
 - From thesis: Ch.6; `simulations/benches/compression_trinity_bench.py`
 - Status: thesis-defended; **not on engine path** until afternoon experiment passes (see below)
 - Precondition: same gate as Compression Trinity experiment
 - Verify: storage ratio + dequant overhead vs borrowed codecs
 - No-stack: one of the P2.b choices, not additive with rwkv_lightning FP5 / HQQ4 packs

### P2.c — Schedule / topology (place work correctly)

- **Heterogeneous chunk schedule (uniform-K vs 128 KiB vs hybrid)**
 - From thesis: `simulations/benches/heterogeneous_chunk_schedule_bench.py`; thesis A5 #13
 - Status: new-idea
 - Precondition: scheduler that picks policy per layer
 - Verify: same CSV, two policies side by side
 - No-stack: alternative to fixed K, not additive
- **Hot-layer pinning**
 - From thesis: n/a (engine-specific)
 - Status: engine-specific; residency profiles (P1 #10) are the hook
 - Precondition: metrics CSV with bubble pattern
 - Verify: RSS saved + tok/s delta
- **Second NVMe / striped read**
 - From thesis: Ch.11
 - Status: inspired
 - Verify: `bench/bench_io.py` MB/s scaling with drive count
- **NUMA-aware I/O threads**
 - From thesis: Ch.11; `simulations/benches/numa_topology_bench.py`
 - Status: inspired
 - Verify: inter-socket fabric cost
- **Micro-batching**
 - From thesis: Ch.5
 - Status: **real ChatRWKV CPU implementation (July 12, 2026)** — dense
   layer-outer/session-inner decode through `InferenceEngine.generate_batch()`;
  synthetic path also retained. rwkv.cpp has a provider bridge and native
  weight-stationary path; shared-sweep batching and long real-model quality
  certification remain open.
 - Verify: exact greedy parity passed at B=2. Clean 0.1B/8-token diagnostic
   improved aggregate end-to-end throughput 1.12→1.89 tok/s and aggregate
   decode-only throughput 1.37→2.75 tok/s; per-session latency stayed about
   0.73 s/token. Continue with B=1..N and longer repeated workloads.
 - Scope: capacity win, not single-session latency or physical-SSD evidence;
   short requests remain prefill-bound.
- **Concurrent model loading (cold start)**
 - From thesis: n/a; compare **runai-model-streamer** / **fastsafetensors**
 - Status: engine evaluation
 - Verify: time-to-first-token vs `pack_runtime` on same checkpoint
 - No-stack: bulk load ≠ per-token stream; TTFT only
- **Asymmetric RAID power scaling (active-drive count with model size)**
 - From thesis: `simulations/benches/asymmetric_raid_power_bench.py`; Appendix B
 - Status: new-idea
 - Precondition: per-machine power budget
 - Verify: power curve vs MB/s
 - No-stack: orthogonal to codec/layout choice
- **Die-thermal balance (per-die thermistor)**
 - From thesis: `simulations/benches/die_thermal_balance_bench.py`
 - Status: inspired
 - Verify: thermistor log + throttle-event reduction

### P2.d — Reliability / tail (orthogonal, low-contention)

- **Hedged reads on mirrored layout**
 - From thesis: Ch.14
 - Status: inspired
 - Precondition: RAID-1/10
 - Verify: p99 latency vs read-amp
- **S.M.A.R.T.-aware scheduling**
 - From thesis: Ch.14
 - Status: inspired
 - Verify: SMART poll latency × throttle response
- **Read-disturb-aware rotation**
 - From thesis: Ch.14; `simulations/benches/read_disturb_rotation_bench.py`
 - Status: inspired
 - Verify: hot-region wear counter
- **Application-layer erasure coding for streamed weights**
 - From thesis: `simulations/benches/erasure_coding_bench.py`; Appendix B
 - Status: new-idea
 - Precondition: RAID-style layout
 - Verify: rebuild time vs storage overhead
- **pSLC state endurance (state-parking durability)**
 - From thesis: `simulations/benches/pslc_state_endurance_bench.py`
 - Status: new-idea; only useful if state-parking is hot
 - Verify: endurance counter delta under repeated park cycles

### P2.e — Speculative amplifiers (only if prefill/decode dominated)

- **Recurrent MTP with state-rolling extrapolation**
 - From thesis: Ch.9; `simulations/benches/mtp_ssd_speculation_bench.py`
 - Status: inspired
 - Precondition: prefill/TTFT dominated workload; verifier cost ≤ savings
 - Verify: accepted tokens per sweep × verifier cost
 - No-stack: throughput transform on decode loop, not byte multiplier
- **N-gram weight cache (prefill-time only)**
 - From thesis: `simulations/benches/ngram_weight_cache_bench.py`; Appendix B
 - Status: new-idea
 - Precondition: token-pattern reuse in workload
 - Verify: cache hit rate × RSS
 - No-stack: prefill-only; do not multiply onto decode tok/s

- **DSpark confidence-scheduled speculative decoding**
 - From paper: `rwkv_ssd/runtime/dspark.py`; [DSpark](https://arxiv.org/abs/2607.05147)
 - Status: research primitive; not enabled in packed backends
 - Precondition: trained proposal/correction head, calibrated confidence, and a target verifier
 - Verify: exact acceptance rate, accepted tokens per verification sweep, and end-to-end tok/s
 - Adaptation: Transformer attention can draft a parallel block; Mamba and RWKV use the same correction/verifier interface while retaining state-rolled proposal cost

## P3 — only if P0–P1 are done

> Forward-looking SSD exploits, excessive disk caching, stack combinations, and strict/auto roadmap:.

20. **INT8 / GPTQ stream** — only if dequant on GPU beats smaller packs on disk.
21. **HRWKV7 (hybrid RWKV-7 + GQA)** — track **RWKV-Infer**; only RWKV layers need SSD delivery; stretch milestone M8.
22. **NVIDIA GPUDirect Storage (GDS)** — **M6c** production path for SSD→GPU layer streaming; DeepNVMe `gds_handle` (+10–37% vs bounce). See DNV-10. Dev on Intel iGPU cannot validate GDS — needs Linux + NVIDIA.
23. **Custom weight compression in hot path** — only with measured win vs quantized packs + fast GPU dequant.

### Codec quality roadmap (post K-means — shipped in v0.6.16)

**v0.6.16 baseline (measured on 48 real RWKV-7 0.1B att weights, 768×768 each):**

| Strategy | mean SNR | mean RMSE | mean cos sim | median encode (ms) |
|----------|----------|-----------|--------------|--------------------|
| `linspace` (legacy) | -10.78 dB | 0.0454 | 0.32 | 48 |
| `kmeans` (default) | **+7.32 dB** | **0.0050** | **0.90** | 280 |

The K-means codebook is **+18.1 dB SNR / 9× lower RMSE / +0.58 cos sim** on real RWKV weights. The historical per-weight table is [`../archive/bench/codec_comparison.md`](../archive/bench/codec_comparison.md); new runs default to `bench/results/codec_comparison.md`. The SOTA 2-bit codecs that beat it require deeper work:

| Technique | SNR on real RWKV (vs kmeans) | Effort | Refs |
|-----------|------------------------------|--------|------|
| **K-means (shipped v0.6.16)** | 0 dB baseline (+7.3 dB abs) | done | IDEAS S2 |
| Per-row K-means | -1 to +1 dB on small matmuls, more on big | done (script) | IDEAS S3 |
| Hadamard + K-means (QuIP#) | +1-2 dB on RWKV, +5-8 dB on LLM | 1 week (rotation in encode/decode) | [QuIP#](https://arxiv.org/abs/2402.04396) |
| 2:4 structured sparsity | halves the bit-stream to 1 bit/w (16×) | 2-3 weeks (Marlin-style CPU kernel) | [Marlin](https://arxiv.org/abs/2406.09994) |
| AQLM learned additive codebooks | +0.5-3 dB on LLM, training pass required | 2-4 weeks (calibration step) | [AQLM](https://arxiv.org/abs/2401.06118) |
| QTIP trellis coded quant | 2-bit QTIP ≈ 4-bit GPTQ on Llama-2-7B | 1-2 weeks (trellis state, custom kernel) | [QTIP](https://arxiv.org/abs/2406.11235) |
| BitNet b1.58 (ternary, from-scratch) | 1.58 bits, FP16 parity at 3B+ | training, not PTQ | [BitNet b1.58](https://arxiv.org/abs/2402.17764) |

**Hadamard + K-means** is the right next step for a 1-week effort: it adds ~5-8 dB
SNR on real LLM weight distributions (per QuIP#) and only adds the rotation
multiplication on encode and the inverse on decode. The codec module
(`trinity_codebook.py`) is already wired; only the `encode_trinity_lut2`
function needs the rotation step added.

**Per-row K-means** is also implemented in the comparison script
(`--with-per-row`) and gives marginal SNR improvement on the 0.1B att weights
(the 768-row matrices already share similar distributions; per-row helps
more on big FFN matrices with wider dynamic range). Not yet wired as a
default — keep `kmeans` (per-tensor) for the simple 1-codebook-per-tensor
model.

## Explicitly deprioritized

- Stacking thesis simulation “speedups” into one tok/s number.
- **Weak / wrong-fit ideas from out-there sweep** — archived in [`archive/ideas/OUT_THERE_ARCHIVED.md`](../archive/ideas/OUT_THERE_ARCHIVED.md) (VcLLM/NVDEC, ngram decode cache, layer-only prefetch, etc.).
- **Platform / GPU + bounded RAM:** · **LUT3/4 ladder:** 
- Mamba / Transformer as first-class in this repo.
- Ouroboros, CSD in-storage scan, Engram Sector-Nine unless second product.
- 50+ scripts in `simulations/` — not engine CI.
- **`storage_bench/` (Rust io_uring)** — research artifact (Linux sequential ceiling); engine decode uses **mmap/pread/threaded**; future Linux overlap via io_uring prefetch, not third-party async seek+read wrappers.
- Re-implementing features already better in **rwkv_lightning / web-rwkv / RWKV-Infer** without a borrow audit.
- **Compression Trinity** stays off the engine path until the afternoon experiment below passes; see **P2.b** for the codec ladder instead.

## Experiments worth one afternoon each

| Experiment | Pass criterion |
|------------|----------------|
| `bench_io` on HDD vs NVMe vs RAM disk | Read BW explains bubble |
| Partial residency sweep | Smallest resident set for &lt;5% tok regression |
| **rwkv-fla** resident vs ChatRWKV resident | Same greedy tokens; record tok/s |
| **web-rwkv** NF4/INT8 resident vs FP16 stream | Size/latency trade documented |
| **rwkv_lightning** HTTP throughput vs CLI | Overhead % for same prompt |
| **runai-model-streamer** vs `pack_runtime` cold-start | Time-to-first-token on same checkpoint |
| Albatross `faster3a_2605` vs ChatRWKV resident | Parity + speed table |
| **Compression Trinity vs borrowed codec ladder** | Trinity wins ≥1.3× on measured storage ratio with ≤10% dequant overhead regression in `bench_io` + a simple forward pass. Compare against rwkv_lightning FP5 and RWKV-Infer HQQ4 on the same ≤3B RWKV-7 checkpoint. If pass, move to P2.b; if fail, document result and shelve Trinity. |

---

## After core is stable — exploratory additions (no commitment)

### RWKV-8 DeepEmbed (announced May 2025)

Token-conditioned memory in the RWKV stack. **Status:** preview / few production models. **Would add:** another resident vs streamed tensor family beside layer weights. **Deferred:** changes model contract before M3 streaming is proven.

### RWKV-8 ROSA (announced Oct 2025) + ROSA-Tuning (Feb 2026)

Symbolic / retrieval side-channel for long context on other architectures. **Status:** experimental. **Would add:** extra retrieval I/O orthogonal to layer-weight streaming. **Deferred:** modifies the model, not the SSD layer stream.

### DeepSeek Engram (released Jan 2026)

Episodic memory with **three-tier hierarchy (HBM / DRAM / NVMe)** — see https://github.com/deepseek-ai/Engram. **Would add:** design reference for **residency profiles** (what stays in RAM vs streams from SSD). **Deferred:** separate memory subsystem.

### July 15 architecture pass

- **DeepEmbed is not Engram:** both suggest deterministic lookup plus tiered
  storage, but DeepEmbed is a RWKV model contract and Engram is a separate
  conditional-memory architecture.
- **Variant-aware DeepEmbed:** qkv/DEA still uses `DeepEmbed.bin` and now has
  a provider-driven CPU reference stream with resident parity. RWKV7a
  DeepEmbed-v1 is detected separately, uses upstream ChatRWKV's
  `RWKV_DE_VERSION=1`, keeps `s_emb`/`s_emb_x` in the pack, and passes
  resident/CPU-streaming greedy parity.
- **Real small references:** `research/real_sequence_models.py` loads the
 local operator-supplied Mamba and Transformer fixtures
  checkpoints without requiring `transformers`, `mamba_ssm`, or a tokenizer.
  Fixed IDs are deliberate; tokenizer compatibility is not claimed.
- **Exact chunked prefill:** ordinary RWKV-7 prompt prefill uses a layer-outer
  sequence schedule controlled by `RWKV_STREAM_PREFILL_CHUNK`.
- **Next architecture work:** qkv/DEA batching, state-aware chunk scheduling,
  and fused kernels. Do not turn the CPU reference models into a production
  claim until optimized kernels and real model quality are measured.

Hardware/training gates remain CUDA/GDS, GPU-fused kernels, physical multi-SSD
scaling, large 2.9B/7B measurements, production Mamba/Transformer backends,
and large-model or quantizer training.

Detailed evidence and source links: [`RESEARCH_AND_ARCHITECTURE.md`](RESEARCH_AND_ARCHITECTURE.md).

### HRWKV7

Hybrid RWKV-7 + GQA (**RWKV-Infer**). **Would add:** test whether streaming engine generalizes to hybrid blocks. **Deferred:** M8 stretch only.

### GDS / fastsafetensors at scale

**Would add:** faster **initial** weight availability on cloud iron. **Deferred:** hardware and FS constraints; benchmark against mmap pack first in P2.c.

## See also

- **[`SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md)** — implementation contract for cache formats, Pareto F tiers, striped sharding, GPU staging, and quality gates for speculative ideas.

- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — current state and the F-1/F-2/F-3 status; this file is the *what to do next* companion.
- **[`BASELINE_BUGS.md`](BASELINE_BUGS.md)** — bugs that motivated several P0.5 / P0.6 items in this doc.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — current mechanism catalog and measured tok/s; the no-stacking rules.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — M-class items (sharded pack, huge pages, mmap readahead) for the SSD path.
