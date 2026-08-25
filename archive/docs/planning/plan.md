> **Repo note (2026):** This is a planning document. The active product is the **RWKV SSD inference engine** — see [`ENGINE.md`](../../ENGINE.md) at repo root and [`docs/BACKENDS.md`](../BACKENDS.md). Implementation lives in `rwkv_ssd/`, not in `simulations/`.

### What’s already shipping elsewhere (don’t re-invent)

| Project | One-line value for us |
|---------|------------------------|
| [rwkv_lightning](https://github.com/RWKV-Vibe/rwkv_lightning) | Batch + state cache + OpenAI HTTP + many quants (CUDA/ROCm) |
| [web-rwkv](https://github.com/cryscan/web-rwkv) | WebGPU inference, NF4/INT8, async runtime API |
| [RWKV-Infer](https://github.com/OpenMOSE/RWKV-Infer) | Multi-batch, dynamic state cache, HRWKV7 hybrid |
| [rwkv-fla](https://github.com/fla-org/flash-linear-attention) | Triton RWKV kernels, cross-vendor GPU |
| [Albatross](https://github.com/BlinkDL/Albatross) | Fastest RWKV-7 CUDA variants (`faster3a_2605`, `faster4_2605_cpp`) |
| Engine I/O | **mmap / pread / threaded** | Default mmap; ayafileio evaluated and rejected (v0.5.4) |
| [runai-model-streamer](https://github.com/run-ai/runai-model-streamer) | Concurrent safetensors load to GPU (cold-start benchmark) |
| [fastsafetensors](https://github.com/fastsafetensors/fastsafetensors) | GDS-class fast load (vLLM ecosystem; v0.2.x in 2026 CUDA stacks) |

**Our unique work:** per-token **layer-weight streaming from SSD during decode**, not another fast RWKV server.

---

Yes. Here is the plan I would actually follow.

The headline recommendation is:

**Build the main prototype on ChatRWKV/PyTorch, and use rwkv.cpp as the quantized sidecar/reference backend.**
That gives the best solo-builder tradeoff:

* ChatRWKV is the easiest place to understand and modify RWKV inference, and its own README explicitly says to start from `src/model_run.py` when building a custom inference engine. It also already has a v2 branch with `stream` / `split` strategies and INT8 support. ([GitHub][1])
* RWKV itself is a linear-time, constant-space recurrent architecture with no KV-cache, which is exactly the structural property your thesis depends on. ([GitHub][2])
* rwkv.cpp already supports FP16 plus quantized INT4 / INT5 / INT8 inference, has a Python wrapper, and supports cuBLAS for GPU-assisted execution. ([GitHub][3])

That means:

* **V0/V1 primary engine**: ChatRWKV + custom SSD streaming runtime
* **V1/V2 quantized branch**: borrow or integrate rwkv.cpp Q4/Q5 support
* **V3**: the hard thesis-grade stuff you intentionally do not build first

This fits your thesis direction, because the defended core is the O(1)-state architecture, sequential layer streaming, and overlap, while many of the more aggressive mechanisms are explicitly extensions or future work. 

---

## What the product is

A prototype local inference engine for RWKV that:

* keeps the recurrent state in RAM/VRAM,
* keeps only hot tensors resident,
* streams most layer weights from SSD in layer order,
* overlaps disk read, host staging, GPU copy, and compute,
* emits real tokens.

Not a thesis proof. Not a 70B system. A real, shippable prototype.

---

## Core design choices

### 1. Model family

Use **RWKV**, not Transformer, not full Mamba first.

Why:

* the no-KV-cache, recurrent-state property is the whole point of your storage bet. ([GitHub][2])
* ChatRWKV is the clearest editable reference. ([GitHub][1])

### 2. Main codebase

Use **Python + PyTorch + ChatRWKV** first.

Why:

* easiest to modify,
* easiest to profile,
* easiest to get tokens out,
* easiest to add pinned-memory + CUDA streams.

PyTorch’s own guidance confirms that `to(device)` already uses `cudaMemcpyAsync`, and overlap requires pinned host memory, a separate CUDA stream, and GPU DMA capability. ([PyTorch Documentation][4])

### 3. Disk path

Support two I/O backends from the start:

**Backend A: `mmap` + `madvise`**
Use Python `mmap`, and optionally `mmap.madvise()` with `MADV_SEQUENTIAL`, `MADV_WILLNEED`, and `MADV_DONTNEED`. Python exposes these flags directly on supported systems. Linux documents that `MADV_SEQUENTIAL` enables aggressive read-ahead and that `MADV_WILLNEED` signals near-future access. ([Python documentation][5])

**Backend B: `pread` + `posix_fadvise`**
Use explicit offset-based reads for more deterministic profiling. On Linux, `POSIX_MADV_SEQUENTIAL` / `WILLNEED` semantics exist for memory advice, and `posix_fadvise` is the file-side analog for access pattern hints. ([man7.org][6])

Start with `mmap`. Keep `pread` as a switch if page-cache behavior gets weird.

### 4. Weight format

Use **safetensors for source ingestion**, and a **flat runtime pack format** for streaming.

Why:

* safetensors supports lazy and partial loading and is easy to manipulate safely. ([Hugging Face][7])
* for runtime, a single large packed file plus manifest is better than hundreds of tiny files.

So:

* ingest from `.pth` / `.safetensors`
* convert once to `weights.bin + manifest.json`
* runtime only reads from that packed format

### 5. Quantization path

Use **existing Q4/Q5 support**, but not in V0.

The best practical answer to “use their Q4 support?” is:

* **Yes, use rwkv.cpp’s Q4/Q5 support in V1/V2**
* **No, do not make Q4 the first blocker for your streaming runtime**

Reason:

* rwkv.cpp already supports `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and FP16, with conversion and quantization tooling. ([GitHub][8])
* but your main product risk is not “can Q4 exist?” It is “can SSD streaming + overlap + recurrent state actually work?”

So:

* **V0**: FP16 or BF16-like simple path
* **V1**: try ChatRWKV v2’s built-in easier quantized/streaming options, especially INT8, because they are already in the same stack. ([GitHub][1])
* **V2**: integrate rwkv.cpp Q4/Q5 as a serious quantized mode or reference backend

That is the best difficulty/reward trade.

---

## Architecture

### Runtime layout

Keep these resident:

* tokenizer
* embedding table if small enough
* output head if practical
* recurrent state
* tiny normalization / helper tensors
* sampling state

Stream these:

* big per-layer weight blocks
* optional per-layer auxiliary weights

### File layout

Do **not** shard per layer into separate files.

Use:

* `weights.bin`
* `manifest.json`
* optional `meta.json`

`manifest.json` should contain:

* tensor name
* layer id
* dtype / quant type
* shape
* byte offset
* byte length
* alignment
* residency flag (`resident` vs `streamed`)
* dequant policy (`none`, `q4_rwkvcpp`, later maybe `int8`)

### Memory layout

Use:

* one mapped or explicitly opened runtime file
* two pinned host staging buffers
* two device buffers
* one compute stream
* one H2D copy stream
* one I/O worker thread

That is your basic ping-pong engine.

---

## Repo layout

Use something like this:

```text
rwkv_ssd/          # actual package name in this repo
  tools/
    convert_from_chatrwkv.py
    pack_runtime.py
    repack_rwkvcpp_quant.py
    inspect_manifest.py

  runtime/
    manifest.py
    io_mmap.py
    io_pread.py
    staging.py
    residency.py
    scheduler.py
    executor_rwkv.py
    sampler.py
    state.py
    metrics.py

  backends/
    chatrwkv_ref/
    rwkvcpp_ref/

  bench/
    bench_io.py
    bench_h2d.py
    bench_layer_loop.py
    bench_tokps.py
    bench_temp.py

  app/
    cli.py
    server.py
```

---

## Main loop

The core loop should be this simple:

1. Load prompt and initialize RWKV state.
2. For each layer:

   * issue read/prefetch for layer `N+1`
   * if needed, fill pinned host buffer for layer `N`
   * async H2D copy layer `N` into device buffer A/B
   * run layer `N` on compute stream
   * swap buffers
3. Final logits
4. Sample next token
5. Update recurrent state
6. Repeat

The key metric is not just `tok/s`. It is:

* SSD read time
* host staging time
* H2D copy time
* layer compute time
* GPU bubble time
* total loop time

That is much closer to the thesis’s actual core than chasing one headline number. 

---

# V0 — “real tokens, simple stack”

This version proves the product concept with minimal pain.

## Goal

A working RWKV inference engine that streams most layer weights from SSD and generates tokens correctly.

## Stack

* ChatRWKV / `src/model_run.py` as the reference execution path. ([GitHub][1])
* Python + PyTorch
* `mmap` backend first
* flat runtime pack format
* FP16 weights
* K=2 ping-pong only
* one-request decode only

## Features

Include:

* prompt encode
* recurrent state handling
* single-user token generation
* mapped file or offset reads
* pinned staging buffers
* async H2D copy with a separate stream
* per-layer metrics logging

Do not include:

* batching
* quantization beyond “whatever is easiest”
* MTP
* LoRA hot-swap
* server mode
* prefix caching
* RAID-specific tuning

## Why this is the right V0

PyTorch already supports the mechanics you need for async device copies, and Python’s `mmap` exposes `madvise`. ([PyTorch Documentation][4])

Also, PyTorch’s `torch.from_file()` can create CPU tensors backed by a memory-mapped file, but those mapped tensors cannot be created in pinned memory, which is why the right design is still:
**mapped file → pinned staging buffer → device buffer**. ([PyTorch Documentation][9])

## V0 deliverables

You should have:

* a model packer
* a runtime manifest
* a decode loop
* a benchmark script
* a CLI like:

```bash
python -m app.cli --model ./runtime_pack --prompt "Hello"
```

## V0 success criteria

* tokens are correct relative to the in-memory reference
* most large layer weights are streamed, not permanently loaded
* the engine can complete multi-token generation without crashes
* you can report per-layer read/copy/compute/bubble numbers

## V0 target result

This is your “am I crazy?” checkpoint.
If V0 works, the thesis idea is no longer just a thesis idea.

---

# V1 — “practical speedups with low pain”

This version adds the highest-value optimizations that are still manageable.

## Goal

Improve latency and memory footprint without changing the whole architecture.

## Add these features

### 1. Add a second I/O backend

Implement `pread` + advisory hints in addition to `mmap`.

Why:

* easier profiling
* easier to isolate page-cache weirdness
* lets you benchmark page-cache-driven streaming vs explicit reads

### 2. Add `madvise` / access-pattern hints

If using `mmap`, apply:

* `MADV_SEQUENTIAL` for mapped ranges
* `MADV_WILLNEED` for next layer
* `MADV_DONTNEED` for consumed layers if memory pressure matters

Linux documents these as readahead/caching hints for mapped memory. ([man7.org][10])

### 3. Add a resident-vs-streamed policy

Keep always resident:

* embeddings
* output head
* norms / tiny tensors
* first and last layer if they dominate latency less

Stream only the heavy middle layers.

This is one of the easiest real wins.

### 4. Add ChatRWKV v2 strategy support

Leverage v2 `stream` / `split` strategies and its model conversion path where useful, because ChatRWKV already says those help loading speed and CPU RAM use. ([GitHub][1])

### 5. Add an in-memory fallback mode

You need three modes:

* full in-memory
* partial streaming
* full streaming

That lets you answer where the slowdown comes from.

### 6. Add a lightweight server wrapper

Not production-grade. Just enough to benchmark repeated requests.

## Optional V1 quantization choice

This is where you decide between two branches:

### Branch A: easier

Use ChatRWKV’s already-supported easier quantized modes and strategies first, especially INT8-style paths in the same ecosystem. ([GitHub][1])

### Branch B: better compression

Start using rwkv.cpp’s quantized formats as offline artifacts or a side backend:

* `Q4_1` as the main candidate
* `Q5_1` as the quality-safe candidate
* `Q8_0` if Q4 hurts too much

rwkv.cpp’s own reference table shows the usual size/latency/quality trade space and warns to benchmark perplexity and latency on representative data. ([GitHub][8])

My recommendation:

* **V1 default: stay in the ChatRWKV family**
* **V1b: build a rwkv.cpp branch for comparison**

## V1 success criteria

* clearly lower bubble time than V0
* lower RAM footprint
* clean comparison across in-memory / partial / full streaming
* one quantized mode benchmarked

**Engine note (June 2026):** V1 golden + skeleton load are done. **Active work:** V1.5+ throughput (v0.6.7–v0.6.10) — decouple, promote, disk cache, prefix cache; FP16 streaming ≈ resident after warm; Trinity LUT2 gap remains. See [`MILESTONE_STATUS.md`](../MILESTONE_STATUS.md).

---

# V1.5 — Throughput mechanisms (P2): speedups without stacking

> **Full tables, measured numbers, and promotion gates:** [`../THROUGHPUT_PLAN.md`](../THROUGHPUT_PLAN.md) · **Backlog:** [`../IDEAS.md`](../IDEAS.md) P2.a–e · **Theory:** thesis [§2.5](../thesis/thesis_draft.md#25-pipeline-composition-and-non-stacking-rules)

After V1 correctness, speed comes from **one scheduler** and **one SSD bandwidth budget**. Do **not** multiply simulation speedups into a single `tok/s` headline.

## Composition (from thesis §2.5)

| Category | Rule | Plan items |
|----------|------|------------|
| **Pipeline stages** | Stack as stages of one forward pass | mmap/pread read → staging → H2D → matmul; micro-pipeline overlap within layer |
| **P2.b bandwidth** | **Mutually exclusive** | Compression Trinity vs M5 borrowed codecs vs channel-aligned pack — pick one |
| **P2.c topology** | Orthogonal; same budget | NUMA, striped NVMe, hot-layer pin, heterogeneous **or** uniform-K chunks |
| **P2.e speculative** | Workload-gated | MTP (~2.5× tok/sweep in thesis model); n-gram cache prefill-only |

## Measured engine baselines (record on every release)

| Workload | tok/s or BW | Notes |
|----------|-------------|--------|
| Synthetic streaming / resident | ~2590 / ~9700 | `bench/bench_generate.py` |
| ChatRWKV 0.1B FP16 stream+cache + promote | ~27–51 warm / ~27 resident | v0.6.9; was ~15 pre-promote |
| ChatRWKV 0.1B strict streaming | ~31% of resident | inject every layer |
| Trinity LUT2 + shadow (0.1B) | ~15–17 | LUT decode bound |
| Pack mmap read | ~1.6 GB/s | `bench/bench_io_backends.py` (Windows) |

## Implementation waves (maps to IDEAS P2)

**Wave A — latency hiders (P2.a)**  
Layer prefetch (extend metrics), micro-pipelining K=8–16 (chunked reader), gate-based prefetch (preferred over blind speculative SSD read).

**Wave B — bandwidth (P2.b)**  
M5 quant ladder; Compression Trinity afternoon experiment; optional channel-aligned pack layout.

**Wave C — topology (P2.c)**  
mmap + madvise (Linux), hot-layer pinning, second NVMe, NUMA threads, micro-batching bench.

**Wave D — tail + speculative (P2.d / P2.e)**  
Reliability for RAID deployments; MTP / n-gram only when workload is prefill-dominated.

## V1.5 success criteria

* Every PR that claims “faster” includes metrics CSV or `bench_io` with **mode** and **hardware** labeled.
* No headline combines Trinity × micro-pipeline × MTP.
* At least one P2.a item shows non-zero overlap in `read_ms` / `compute_ms` / `lookahead_ms`.
* M5 picks **one** codec path for packs; Trinity only if IDEAS experiment passes.

---

# V2 — “serious prototype”

This is the version that starts to feel like an actual product.

## Goal

Add one or two bigger wins that are still realistic for one person.

## Add these features

### 1. Quantized runtime mode

This is the main V2 upgrade.

Use rwkv.cpp’s Q4/Q5 support in one of two ways:

**Option 1: sidecar backend**
Just expose a rwkv.cpp backend mode in your engine and compare it directly. This is the easiest way to “use their Q4 support.” rwkv.cpp already provides a C library and Python wrapper. ([GitHub][8])

**Option 2: import-and-repack**
Convert with rwkv.cpp tools, then repack selected quantized tensors into your own runtime format and write dequant/load adapters.

This is harder, but keeps one unified engine.

I would start with Option 1.

### 2. Small batching

Not massive serving. Just batch size 2–4 for nearby requests.

Because RWKV keeps bounded state, batching is simpler than for a giant KV-cache system, but still adds scheduler complexity. Keep it small.

### 3. Prefix/state reuse

Do **state reuse**, not Transformer-style KV prefix caching.

vLLM’s APC is useful for understanding the concept, but its own docs are explicit that APC helps **prefill**, not decode. ([vLLM][11])

For RWKV, the analogous easy feature is:

* cache recurrent state snapshots for repeated system prompts or common prefixes
* resume from the cached state instead of replaying the whole prompt

This is one of the best “product features” you can add without rewriting the engine.

### 4. Better scheduler policy

Add simple heuristics:

* prefetch next layer only if copy queue is not backlogged
* pin hot layers if they repeatedly dominate bubbles
* choose `mmap` vs `pread` based on file size and host memory pressure

### 5. Better metrics

Add:

* p50 / p95 layer bubble
* SSD temp
* achieved SSD throughput
* H2D overlap ratio
* tok/s by request length
* prompt-time vs decode-time split

## V2 success criteria

* meaningful improvement over V1
* one quantized mode that is actually usable
* state reuse or small batching working
* enough stability that another person can run it

At this stage, you have a real open-source demo.

---

# V3 — possible, valuable, but too hard for the first serious build

This is the “possible but not first” bucket.

## Keep these out of the critical path

### 1. Custom 2-bit codec stack

Your thesis’s custom LUT 2-bit + ANS + sparsity accounting is exactly the kind of thing that can eat months. Keep it out of the main build. 

### 2. `io_uring` / SPDK / kernel-bypass storage

Valuable, but the complexity is real. Your current prototype should first answer whether the architecture is promising using normal Linux tools.

### 3. GPUDirect Storage / fastsafetensors / Run:ai Model Streamer

**Updated (2026):** **fastsafetensors** (e.g. v0.2.x in vLLM’s CUDA requirements) and **runai-model-streamer** make **fast bulk load to GPU** much more tractable than when this plan was first written — often large multiples over a naive Hugging Face safetensors path on GDS-capable hardware.

Treat them as **P2**: benchmark cold-start and first-load against your custom **mmap** pack path; **borrow if faster**. They still do **not** replace per-token **layer streaming during decode** (they optimize getting weights *into* memory once, not re-reading layers from SSD every token).

True **GDS** at scale still implies FSx-for-Lustre / EFA / specific cloud iron — keep full GDS integration as **P3**, not V0.

### 4. MTP / speculative recurrent decoding

Still worth trying later, but it changes validation complexity a lot. The thesis also treats this as part of a modeled stack, not the first thing to prove. 

### 5. RAID striping / NUMA tuning / topology pinning

Good later if a single NVMe drive is not enough. Bad first blocker.

### 6. LoRA hot-swap in the streamed runtime

Interesting product feature. Too many moving parts for the first serious build.

### 7. Custom CUDA kernels for dequant + matmul fusion

This is exactly where solo “vibe coding” goes to die.

---

## Which outside tools are foundation vs reference

### Use as foundation

* **ChatRWKV** for primary editable runtime. ([GitHub][1])
* **PyTorch** for async copy, pinned memory, streams. ([PyTorch Documentation][4])
* **safetensors** for safe lazy source format. ([Hugging Face][7])
* **rwkv.cpp** as quantized sidecar / comparison backend. ([GitHub][8])

### Use as reference only

* **vLLM** for ideas on prefix caching, loading infrastructure, and benchmarking; not the main runtime for this RWKV engine. Its APC is about KV-cache reuse and mainly helps prefilling, not decode. ([vLLM][13])
* **llama.cpp** as a baseline for how a practical low-bit local inference project is structured, but not as your core codebase because your structural bet is recurrent O(1) state, not Transformer+KV.
* **Unsloth** only as a convenient Transformer baseline if you want a “normal stack” comparison, not as the RWKV foundation.

---

## The exact recommendation on Q4

Since you specifically asked:

**Yes, use their Q4 support — but in V2 or as a V1 side branch, not as the first blocker.**

Best sequence:

1. **V0**: FP16 streaming engine works
2. **V1**: improve overlap and file layout
3. **V1b/V2**: add rwkv.cpp `Q4_1` and `Q5_1` comparison mode
4. only after that decide whether to fully integrate quantized runtime packing into your own engine

If I had to pick one quantized target first, I would pick **`Q4_1`** for “aggressive but still reasonable,” and **`Q5_1`** as the safer backup. rwkv.cpp’s published reference table includes both formats and shows the expected size/latency tradeoffs. ([GitHub][8])

---

## The single most important constraint

Do not let V0 become “custom compression research.”

Your real product question is much simpler:

**Can a recurrent RWKV engine stream most of its weights from SSD, overlap enough of the storage path, and still generate tokens at a not-embarrassing speed?**

That is the real first win.


## Design reference: DeepSeek Engram + RWKV on SSD

[DeepSeek Engram](https://github.com/deepseek-ai/Engram) (Jan 2026) uses a **three-tier memory hierarchy (HBM / DRAM / NVMe)** for episodic lookup tables. It is **not** a substitute for this engine’s core loop (stream **layer weights** each decode step).

Use Engram as a **residency-profile reference**: what stays in fast tiers vs what streams from SSD — especially if you later add a RAM-hot / SSD-cold split beside layer streaming. Deferred until M3 streaming is proven; see [`docs/IDEAS.md`](../IDEAS.md) appendix.

[1]: https://github.com/BlinkDL/ChatRWKV?utm_source=chatgpt.com "GitHub - BlinkDL/ChatRWKV: ChatRWKV is like ChatGPT but powered by RWKV (100% RNN) language model, and open source."
[2]: https://github.com/BlinkDL/RWKV-LM?utm_source=chatgpt.com "GitHub - BlinkDL/RWKV-LM: RWKV (pronounced RwaKuv) is an RNN with great LLM performance, which can also be directly trained like a GPT transformer (parallelizable). We are at RWKV-7 \"Goose\". So it's combining the best of RNN and transformer - great performance, linear time, constant space (no kv-cache), fast training, infinite ctx_len, and free sentence embedding."
[3]: https://github.com/RWKV/rwkv.cpp?utm_source=chatgpt.com "GitHub - RWKV/rwkv.cpp: INT4/INT5/INT8 and FP16 inference on CPU for RWKV language model"
[4]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html?utm_source=chatgpt.com "A guide on good usage of non_blocking and pin_memory() in PyTorch — PyTorch Tutorials 2.11.0+cu130 documentation"
[5]: https://docs.python.org/3.9/library/mmap.html?utm_source=chatgpt.com "mmap — Memory-mapped file support — Python 3.9.24 documentation"
[6]: https://man7.org/linux/man-pages/man3/posix_madvise.3p.html?utm_source=chatgpt.com "posix_madvise(3p) - Linux manual page"
[7]: https://huggingface.co/docs/safetensors/index?utm_source=chatgpt.com "Safetensors · Hugging Face"
[8]: https://github.com/RWKV/rwkv.cpp "GitHub - RWKV/rwkv.cpp: INT4/INT5/INT8 and FP16 inference on CPU for RWKV language model · GitHub"
[9]: https://docs.pytorch.org/docs/2.8/generated/torch.from_file.html?utm_source=chatgpt.com "torch.from_file — PyTorch 2.8 documentation"
[10]: https://www.man7.org/linux/man-pages/man2/madvise.2.html?utm_source=chatgpt.com "madvise(2) - Linux manual page"
[11]: https://docs.vllm.ai/features/automatic_prefix_caching.html?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
[12]: https://docs.vllm.ai/en/v0.18.2/models/extensions/fastsafetensor.html?utm_source=chatgpt.com "Loading model weights with fastsafetensors - vLLM"
[13]: https://docs.vllm.ai/usage/automatic_prefix_caching.html?utm_source=chatgpt.com "Automatic Prefix Caching - vLLM"
