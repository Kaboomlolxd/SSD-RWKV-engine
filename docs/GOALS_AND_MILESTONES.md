# Goals, baselines, and milestones

This document defines what “good enough” means for the **RWKV SSD inference engine**. Targets are intentionally modest: a working, explainable, maintainable product—not a commercial cloud rival or thesis headline `tok/s`.

Use this as the checklist for releases. If a milestone is not met, do not start the next one.

**Related:** [`ENGINE.md`](../ENGINE.md) · [`IDEAS.md`](IDEAS.md) · [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md) · **Current status:** [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)

---

## Status snapshot (September 1, 2026)

| Milestone | Done? | Notes |
|-----------|-------|-------|
| M3 / V1 streaming golden | Yes | Synthetic + 0.1B ChatRWKV FP16 |
| M2.5 prefix cache | Yes | Synthetic + **ChatRWKV streaming** (v0.6.10) |
| M5 scale_u8/u4 | Yes | Trinity on path separately (not promoted as default) |
| M7 HTTP | Yes | `app/serve.py` |
| V1.5 throughput (FP16) | Yes | Decouple + promote closes resident gap |
| V1.5 throughput (Trinity CPU) | Yes for the native CPU gate | rwkv.cpp F1–F4 cold/warm ratios and memory budgets pass; long-run lossy quality remains open |
| Trinity overhead P0/P1 | Yes | Batched TMix, warm cache default, layer-span fused, partial tier default |
| CPU optimization continuation | Yes for measured safe paths | grouped decode, DeepEmbed lookup/batching, sequence scratch reuse, state-publication elision, and opt-in zstd/packed-head A/B paths |
| **M6 GPU compute + SSD offload** | No | **Primary tok/s track** — Albatross, FLUTE CUDA, GDS |
| M6d Intel iGPU / XPU dev | Partial | Trinity XPU decode experimental |
| 200B / 10 GB RAM thesis | Partial | `--ram-budget-gb`; byte LRU shipped |

**Dev vs target:** Engine developed on **Windows + Intel iGPU** (CPU forward, optional XPU LUT decode). **Production target is GPU** (CUDA) with SSD-backed weights when VRAM is bounded — same streaming inject path, faster compute and GDS I/O.

Deviations from original milestone ordering are documented in [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md).

---

## Product vision (one sentence)

A developer can run a **local RWKV-7 model** from a **packed weight file on SSD**, get **correct tokens**, and see **why** each step is slow (disk vs copy vs compute)—with a path to faster backends later.

---

## Explicit non-goals (for now)

- Multi-node or multi-tenant serving at scale 
- Training, fine-tuning, or a LoRA marketplace 
- Custom 2-bit / ANS compression in the hot path 
- Beating H100 VRAM throughput or matching vLLM feature parity 
- Additional model families as first-class production models in this repo.
- Publishing or defending simulation suites as “measured E2E” 
- **Re-implementing** what **rwkv_lightning**, **web-rwkv**, or **RWKV-Infer** already ship (HTTP, state cache, batch quant) without a documented borrow/wrap decision 

---

## Baseline dimensions

### 1. Correctness

| Requirement | Baseline (must have) | Stretch (later) |
|-------------|----------------------|-----------------|
| Token match vs reference | Greedy decode: **identical** `resident` vs `streaming` on a **≤3B** RWKV-7 checkpoint for **≥32** generated tokens | Same on **7B**; sampling match within fixed seed |
| State lifecycle | Prefill + decode without silent state corruption over **≥500** tokens | Soak **24h** without drift |
| Pack integrity | `verify_pack` passes; corrupt/truncated pack **fails at load** with clear error | SHA-256 in manifest |
| Backend scope | **ChatRWKV** is the reference backend | Albatross matches ChatRWKV on greedy decode |

### 2. Performance

Measure what you control. Report **per-layer** and **per-token** breakdowns, not one heroic `tok/s` without context.

| Metric | Baseline target | Notes |
|--------|-----------------|-------|
| **Streaming correctness** | Achieved before optimizing speed | Non-negotiable gate |
| **Disk read throughput** | Document achieved MB/s for your pack on your drive (`bench_io` or OS tools) | Compare to drive spec (~30–70% of sequential spec is fine) |
| **Resident decode (small model)** | Stable generation on **≤3B** without OOM on **8GB+** GPU | “Works on my machine” documented |
| **Streaming decode (small model)** | Completes multi-token runs without crash; **≤3×** slower than resident on same hardware | Ratio matters more than absolute tok/s |
| **Partial streaming** | Middle layers streamed; **≤2×** slower than full resident on same model | Easier than full streaming |
| **GPU bubble** | Metrics show read/h2d/compute ms per layer; you can name the bottleneck | “Unknown slowness” = fail |
| **vs rwkv.cpp (CPU)** | Optional table: Q4 latency vs FP16 stream—not required to beat GPU path | Comparison only |
| **vs Albatross** | Optional: **≥2×** tok/s vs ChatRWKV on same model when integrated | V1+ only |

**Do not** commit to absolute tok/s (e.g. “50 tok/s on 7B”) until you have one hardware profile and a frozen pack format.

**Do not** multiply independent “speedups” (compression × micro-pipeline × MTP × prefetch) into one headline number. Compose through a **single scheduler and SSD budget** — see thesis §2.5 and [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md). Mechanism backlog: [`IDEAS.md`](IDEAS.md) P2.a–e.

### 3. Usability

| Requirement | Baseline | Stretch |
|-------------|----------|---------|
| Install | `pip install -r requirements.txt` + documented env vars (`CHATRWKV_ROOT`) | `pip install -e.` one-liner |
| Run inference | One CLI command with `--model`, `--checkpoint`, `--prompt` | Config file optional |
| Modes | `--mode resident \| partial \| streaming` documented with trade-offs | Presets in JSON |
| Errors | Missing ChatRWKV, bad pack, OOM → **actionable message** (what to set / fix) | Error codes doc |
| Docs | README + ENGINE + this file; **&lt;15 min** cold start for another dev | Tuning guide |
| Platforms | **Windows or Linux** dev path documented (pick one primary) | Both CI-smoke tested |

### 4. Polish

| Requirement | Baseline | Stretch |
|-------------|----------|---------|
| Logging | INFO: load pack, mode, tokens/s summary; WARN on fallback | Structured JSON logs |
| Metrics | `--metrics-csv` per-layer timings on streaming path | Prometheus / HTTP `/metrics` |
| Versioning | `manifest.json` has `version`; incompatible pack rejected | Migration tool |
| Reproducibility | `meta.json` records source checkpoint label / hash; preflight reports pack, tokenizer, and native identities | Locked manifest in release |
| Repo hygiene | Engine vs `simulations/` clearly separated | CI runs verify_pack + smoke test |
| Release tags | Git tag `v0.x` with short CHANGELOG for milestones | Stable / preview channels |

### 5. Reliability (minimal)

| Requirement | Baseline |
|-------------|----------|
| Clean shutdown | Ctrl+C / process exit does not corrupt open files |
| Repeated runs | **10** consecutive CLI runs same prompt → same greedy output |
| Disk full / missing file | Clear failure, no hang |
| GPU missing | Falls back to CPU with explicit warning or clean error |

---

## Milestones

### M0 — Project baseline (done / maintain)

**Goal:** Repo structure and contracts exist.

- [x] `rwkv_ssd/` package, pack format, CLI skeleton 
- [x] Docs: README, ENGINE, BACKENDS, simulations separated 
- [x] `CHATRWKV_ROOT` documented — 
- [x] `storage_bench/` (Rust `io_uring`) — **research artifact only**; engine decode I/O: **mmap / pread / threaded**

**Exit:** New contributor knows where engine vs simulations live.

---

### M1 — Resident inference (correctness foundation)

**Goal:** Real tokens from ChatRWKV using the repo tooling.

| Item | Acceptance |
|------|------------|
| Pack | `pack_runtime` + `verify_pack` on your chosen **≤3B** RWKV-7 `.pth` |
| CLI | `python -m app.cli --mode resident` prints coherent text |
| Repeatability | Same prompt + greedy → same output **3/3** runs |
| Docs | README quick start works on your primary OS |

**Performance baseline:** Enough tok/s to iterate (no minimum; record number in README footnote).

**Usability baseline:** If ChatRWKV missing, CLI exits in &lt;2s with fix instructions.

**Exit:** You trust the reference path before touching SSD streaming.

**Estimated effort:** Small (days) once ChatRWKV is cloned.

---

### M2 — Pack + disk I/O validated

**Goal:** Prove the weight file on disk is readable at useful speed, independent of model hook.

| Item | Acceptance |
|------|------------|
| `bench_io` | Reports MB/s for full streamed tensor set |
| Streaming trace | `--mode streaming` runs layer I/O loop + metrics CSV without crash |
| Partial | `--mode partial` residency matches policy (manifest flags) |

**Performance baseline:** Record `bench_io` MB/s; streaming may still use ChatRWKV RAM for compute (hook not required yet).

**Exit:** Disk path is not hypothetical.

---

### M2.5 — State cache prototype — **done (extended v0.6.10)**

**Goal:** Validate prefix / state reuse before full M4 polish — **P0** in [`IDEAS.md`](IDEAS.md).

| Item | Acceptance | Status |
|------|------------|--------|
| Reference | Study **RWKV-Infer** and **rwkv_lightning** state / prefix APIs | Done |
| Prototype (synthetic) | Cache recurrent state; second request skips prefix prefill I/O | Done (v0.3.1) |
| Prototype (ChatRWKV) | Same for `rwkv7_state` on streaming path | Done (v0.6.10) |
| Measure | `state_cache_hit`, `prefill_wall_s`; lower read_ms on warm request | Done |

**Lesson learned:** Prefix cache skips **prefill**, not per-token weight loads under low RAM. Pair with `--stream-layer-cache` + promote for decode throughput.

**Note:** Does not replace HTTP borrow (M7); complements it for chat workloads.

---

### M3 — True streaming inference (core product)

**Goal:** Generated tokens use weights loaded from `weights.bin` per layer, not a full RAM copy of all layers.

| Item | Acceptance |
|------|------------|
| Golden test | Greedy: `streaming` == `resident` for **≥32** tokens (≤3B model) |
| Memory | Process RSS / VRAM **lower** in streaming than resident (document rough MB saved) |
| Metrics | CSV has non-zero `read_ms` / `h2d_ms` for streamed layers |
| Prefetch | Layer N+1 read overlaps layer N compute at least once (visible in timings or logs) |

**Performance baseline:**

- Streaming completes without crash for **≥100** tokens. 
- Streaming ≤ **3×** resident wall time on same hardware (same model, greedy). 

**Usability baseline:** `--mode` help text explains when to use each mode.

**Exit:** This is the **“am I crazy?”** milestone from the original plan.

**Estimated effort:** Largest technical chunk (1–3 weeks focused).

---

### M4 — Usable daily driver

**Goal:** You would actually use it locally for experiments.

| Item | Acceptance |
|------|------------|
| Config | Optional `config.yaml` or env: pack path, mode, max_tokens, device |
| Partial tuning | One documented “sweet spot” partial profile for your GPU |
| Errors | All failure paths tested once (bad manifest, wrong checkpoint, OOM) |
| CHANGELOG | `v0.1.0` notes: what works, what does not |

**Performance baseline:** Document one table: resident / partial / streaming tok/s **on your machine**.

**Polish baseline:** No debug print spam; progress line for long generations optional.

**Exit:** Friends could run it following README without a call.

---

### M5 — Quantized comparison branch (P2.b bandwidth)

**Goal:** Document a **quantization ladder**, not only rwkv.cpp Q4_1. Pick **one** codec/layout for streaming packs (mutually exclusive with Compression Trinity unless the IDEAS experiment promotes Trinity).

| Item | Acceptance |
|------|------------|
| Formats | **≥3** quantized paths benchmarked with size + latency + subjective quality, e.g. rwkv.cpp Q4_1/Q5_K, web-rwkv NF4/INT8, rwkv_lightning FP8/INT8/FP6/FP5/HQQ4, RWKV-Infer HQQ4 (pick what you can run) |
| Packs | Optional second/third `weights.bin` variants via `pack_runtime` |
| Scope | FP16 streaming remains default |

**Performance baseline:** Smallest pack **≤50%** of FP16 pack size; table in README/CHANGELOG.

**Exit:** Quant choice is data-driven, not ideology; if the Compression Trinity experiment in [`IDEAS.md`](IDEAS.md) passes its gate, add Trinity to the P2.b codec ladder.

---

### M6 — Fast GPU backend + SSD offload (Albatross / FLUTE / DeepNVMe)

**Goal:** Production inference on **GPU** with optional **SSD layer streaming** when weights exceed VRAM. Dev environment uses **Intel iGPU** for XPU decode experiments; **CUDA** is the acceptance target.

| Sub | Item | Acceptance |
|-----|------|------------|
| **M6a** | CUDA resident forward — **Albatross `faster3a_2605`**, **rwkv_lightning**, or ChatRWKV CUDA | Greedy parity with CPU resident; record GPU tok/s |
| **M6b** | **FLUTE-style fused LUT2 GEMM on GPU** — dequant+matmul without full bf16 `W` in VRAM | Match CPU fused numerics within tolerance; ≥2× vs dequant→GEMM on GPU |
| **M6c** | **GPU layer streaming from SSD** — ZeRO-Inference / DeepNVMe pattern | Overlap layer *k+1* fetch (GDS or aio→pinned→H2D) with layer *k* compute; metrics show overlap |
| **M6d** | Intel **iGPU / XPU** Trinity LUT decode (dev only) | Document vs CPU gather on same layer; not production bar |
| **M6e** | CUDA graph or single native forward when all layers in VRAM | After promote / full VRAM resident |

**DeepNVMe-inspired (M6c):** Linux libaio or **NVIDIA GDS** (`gds_handle`) for NVMe→VRAM DMA; pinned buffer pool; queue-depth tuning (`ds_nvme_tune` analogue). See DNV-10, DNV-1, DNV-2.

**Borrow path:** Wrap **rwkv_lightning** if it ships CUDA batch + state cache + HTTP without forking.

**Exit:** GPU tok/s measured on same checkpoint as CPU F5/F6; streaming-from-SSD path documented with read/h2d/compute breakdown.

---

### M6 (legacy one-liner)

Former “M6 fast backend” — expanded above into **M6a–M6e** because GPU compute and GPU↔SSD I/O are separate acceptance tables.

---

### M7 — Light serving (optional product)

**Goal:** HTTP for local tools, not production cloud.

| Item | Acceptance |
|------|------------|
| Reference | **rwkv_lightning** OpenAI-compatible API + state endpoints; **web-rwkv** runtime API |
| Deliverable | Borrow/wrap preferred over greenfield; minimal custom API only if SSD stream can’t compose |
| Limits | Max prompt, max tokens, timeout, cancel, `/health` |

**Performance baseline:** Single client; **≤10%** overhead vs CLI for same workload.

**Exit:** Scriptable local use without re-building lightning from scratch.

---

### M8 — HRWKV7 stretch (optional)

**Goal:** Prove streaming engine generalizes to hybrid **RWKV-7 + GQA** (**RWKV-Infer** class).

| Item | Acceptance |
|------|------------|
| Scope | Only RWKV-attached layers streamed from SSD; GQA blocks per Infer layout |
| Bar | Greedy parity on one small HRWKV7 checkpoint if available |

**Exit:** Optional; post-M6 only.

---

## Milestone map (timeline sketch)

Solo builder, focused weeks—not calendar promises:

```
M0 ──► M1 ──► M2 ──► M2.5 ──► M3 ──► M4
 │ │
 │ ├──► M5 (quant ladder)
 │ └──► M6 (fast backend)
 │ └──► M7 (HTTP borrow)
 │ └──► M8 (HRWKV7 stretch)
```

**Minimum shippable product:** **M3 + M4** 
**Nice product:** **M3 + M4 + M2.5 + M6a** (GPU resident) 
**Full product (SSD thesis):** **M3 + M4 + M6a + M6b + M6c** (GPU + SSD streaming) 
**Research-complete:** M5 optional; ignore `simulations/` unless exploring ideas

---

## Release naming

| Tag | Milestones | Audience |
|-----|------------|----------|
| `v0.1.0` | M1 | You only |
| `v0.2.0` | M2 | You + disk benchmarks |
| `v0.3.0` | M3 | First real “SSD engine” |
| `v0.4.0` | M4 | Others can run it |
| `v0.5.0` | M5 | Quant ladder |
| `v0.5.1` | M6a | GPU resident compute |
| `v0.5.2` | M6b–c | FLUTE CUDA + SSD→GPU streaming |
| `v0.6.0` | M7 | HTTP (borrow) |
| `v0.7.0` | M8 | HRWKV7 stretch (optional) |

---

## Definition of done (global)

A milestone is **done** only if:

1. **Acceptance table** for that milestone is checked. 
2. **One paragraph** added to README or CHANGELOG with measured numbers (even if unimpressive). 
3. **Known limitations** listed honestly (model size, OS, GPU). 
4. No claim in docs that simulations = engine measurements.

---

## Weekly checkpoint questions

1. Can I generate tokens in `resident` mode right now? 
2. Does `streaming` still match `resident` on greedy decode? 
3. What was the bottleneck last run (disk / H2D / compute)? 
4. Did I add scope that belongs in `simulations/` instead of the engine? 
5. Would another dev know what to install from README alone? 
6. Is any feature here already implemented better in **rwkv_lightning / web-rwkv / RWKV-Infer**? If yes, have we re-evaluated **borrowing** it?

If any answer is “no” for M3+, fix that before new features from [`IDEAS.md`](IDEAS.md).

---

## Summary table (pin on your wall)

| Area | Minimum bar |
|------|-------------|
| **Correctness** | Greedy streaming == resident (small RWKV-7) |
| **Performance** | Explainable timings; streaming ≤3× resident |
| **Usability** | One CLI, clear errors, 15-min README path |
| **Polish** | metrics CSV, versioned pack, tagged releases |
| **Reliability** | 10 repeat runs, clean failures |

That is enough to build a real inference engine without boiling the ocean.

## See also

- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — what is done now; this file is the *what we aimed for* companion.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog that maps to the M0–M8 items.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — the multi-SSD sharded pack pushes the M3 streaming tier toward M4 territory on 7B+ packs.
- **[`IDEAS.md`](IDEAS.md)** — afternoon-experiment backlog.
