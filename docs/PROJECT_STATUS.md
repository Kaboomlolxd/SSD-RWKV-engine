# Project status

This is the current source of truth for what the SSD engine can do, what has
been measured, and what should be used for the talking application. Historical
ideas and exploratory results remain available, but they are not production
recommendations unless they appear in the tables below.

For the full architecture/path matrix, working-tree risks, and production
readiness decision, see [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md). This file
remains the shorter recommendation page.

Updated: 2026-07-31

## Current recommendation

The default compact profile is the promoted grouped-U8 policy selected by the
operator's prepared pack and manifest-bound certificate. It keeps non-matrix
control tensors dense and groups the large matrices. The certificate accepts
only its exact declared checkpoint, tokenizer, prompts, and artifact hashes;
long-run held-out generation qualification remains open. A previous larger
group candidate exceeded the configured KL gate, so grouped quantization is
not a blanket quality guarantee.
The former all-LUT2 reproduction pack has been removed after its
embedding/head representation failed the application quality gate. The
grouped pack is the quality-preserving compact direction, but it is not a
15 tok/s CPU solution on this machine: BF16 resident remains the speed/quality
baseline when the full checkpoint fits in RAM.

The current CPU path is release-safe within the tested model/pack boundary:
the four-way ChatRWKV/rwkv.cpp conformance harness passes exact greedy
token/text parity on the checked-in 0.1B fixture, including provider streaming;
the minimum native top-10 overlap is 0.90, KL stays below 0.005, and relative
state error stays below 0.016 for the certification prompts. The mixed-codec
QKV fast path is enabled and the short-prompt prefill regression is fixed.
Treat the measured tok/s below as local CPU evidence, not a hardware-independent
SLA.

For non-resident grouped-U8 packs, `rwkvcpp` now selects the native CPU SG8
graph automatically. The complete 2.9B matrix pack uses packed-only GGML
residency, with embedding/head included in the provider upload and dense
storage retained only for small controls. This removes the large dense matrix
payload from the native skeleton, but the native U8 route remains a
qualification-bound compact path rather than a general quality guarantee.
The historical larger 2.9B g64 native one-prompt recheck remains outside the
current short-smoke certificate: its top-10 overlap was 0.80, relative state
L2 0.0176, and KL 0.0754 against the dense reference. The configured KL gate
is 0.05; the current g32 smoke improves that short diagnostic, but larger
grouped-U8 measurements below remain performance/ABI evidence rather than
long-run generation qualification.

The native layer ABI now has transient and provider-owned persistent dense
borrowing, plus explicit eviction invalidation. This removes the dense
upload-copy from eligible bounded-provider paths without turning strict F1 or
an mmap-only temporary view into a hidden full-model cache.

## Latest 2.9B CPU comparison

The grouped result below comes from the real `InferenceEngine` path, while the
16-token parity run uses the same checkpoint and pack for both resident and
pack-streaming reference calls. Auto warming and explicit F3/strict mode have
different cold-start tradeoffs, so both are recorded.

| Configuration | Result | Tracked weights | Interpretation |
|---|---:|---:|---|
| Normal BF16 resident | **~2.34 tok/s** | ~5,895 MB | Existing quality/speed baseline |
| Existing all-LUT2, F1/F3 | **0.27–0.54 tok/s** | ~2,037–2,226 MB | Archived; application quality failed |
| Grouped quality, auto warm | **~1.36 tok/s end-to-end** | 671 MB `z` + ~4.30 GB provider pools | ~43–60s cold load; ~2.5 tok/s decode compute-only |
| Grouped quality, explicit strict/F3-style cold | **~0.23 tok/s first request** | 671 MB skeleton before request; packed retention grows after sweep | ~8.5s load + ~8.7s first two-token request |
| Grouped quality, warmed short request | **~1.36 tok/s** | 671 MB `z` + 3.77 GB packed + 527 MB prepared | Actual engine path after load/warm |
| Historical native rwkv.cpp grouped-U8 g64, packed-only | **~2.62 tok/s** | Shape-only GGML matrices + 3.32 GB packed upload | Warm post-upload decode at 4 threads; the g32 quality run did not measure throughput |
| Native rwkv.cpp F1-F5 acceptance matrix | **1.85-2.25 tok/s** | F1-F4 provider-owned bytes 3-462 MB; mmap payload reported separately | 2.9B, 8 generated tokens x 2 samples, explicit 12 GB RSS / 1 GiB provider / 512 MiB native budgets |
| Resident vs grouped streaming, 16 greedy tokens | **exact token match** | Same checkpoint/pack | Pack-streaming parity certificate passed |
| Actual `InferenceEngine` resident vs streaming, 8 tokens | **exact token match** | Same 2.9B checkpoint/pack | `tests/test_real_model_streaming.py` passed |

### Current rwkv.cpp F-tier gate status

The acceptance harness reports cold and warm rows separately and now enforces
the requested ratios and memory domains. The current raw 0.1B native run
(64 generated tokens x 5 samples, `RWKV_CPU_THREADS=1`) measured:

| Tier | Cold tok/s | Warm tok/s | Warm/F5 | Gate |
|---|---:|---:|---:|---|
| F1 | 51.26 | 50.98 | 0.961x | pass |
| F2 | 53.31 | 52.13 | 0.982x | pass |
| F3 | 52.63 | 49.69 | 0.936x | pass |
| F4 | 50.06 | 50.46 | 0.951x | pass |
| F5 | 54.15 | 53.07 | 1.000x | reference |

The larger real 2.9B grouped-U8 run is diagnostic beyond the pack's
short-smoke certificate (long-run quality remains open), but it passes the
native throughput and memory acceptance gates. With 8 generated tokens x 2 samples,
12 GB RSS, a 1 GiB provider budget, and a 512 MiB native-layer budget it
measured:

| Tier | Cold tok/s | Warm tok/s | Cold/F5 | Warm/F5 | Provider-owned MB | Mmap MB | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| F1 | 2.02 | 1.90 | 0.992x | 1.026x | 2.9 | 2,937 | pass |
| F2 | 2.02 | 1.99 | 0.994x | 1.075x | 0.4 | 367 | pass |
| F3 | 2.24 | 1.90 | 1.100x | 1.026x | 370.1 | 2,570 | pass |
| F4 | 2.25 | 2.12 | 1.105x | 1.146x | 461.9 | 2,478 | pass |
| F5 | 2.03 | 1.85 | 1.000x | 1.000x | 188.7 | 0 | reference |

`provider_cache_bytes` is process-owned provider memory. Stable mmap-backed
packed records are not charged as a second heap copy; they are reported as
`provider_mmap_bytes`, while `provider_layer_cache_bytes` and
`provider_resident_bytes` expose the two owned subdomains. This fixes the
previous double-counting of resident native views and keeps the explicit
provider cap auditable. F5 is a high-RAM reference and is not expected to fit
the 12 GB F-tier RSS envelope on this 2.9B host.

Earlier all-LUT2 runs measured approximately `0.40 / 0.39 tok/s` for F1/F3.
The spread is expected on these very short CPU runs.

## Capability matrix

| Area | Status | Evidence / entry point |
|---|---|---|
| CPU synthetic streaming | Supported | `tests/`, `bench/bench_generate.py` |
| CPU ChatRWKV resident | Supported reference path | `bench/bench_f1_f3.py --backend chatrwkv --tiers F6` |
| CPU ChatRWKV BF16 streaming | Supported, quality baseline | operator-prepared RWKV pack + checkpoint |
| CPU legacy LUT2 F1/F3 | Removed; superseded by grouped-U8 and failed quality gates | historical benchmark results only |
| CPU native-safe grouped-U8 direct GEMV | Promoted compact direction; exact pack certificate and bounded native F1-F4 throughput/memory gates required, while long-run quality remains open | `rwkv_ssd/native/lut2_gather.c`, `docs/MODEL_IMPORT.md` |
| rwkv.cpp native grouped-U8 | Packed-only CPU graph; ~2.62 tok/s warm at four threads, with native ABI and malformed-record checks | `backends/rwkvcpp_ref/rwkv_native_u8.inc`, `docs/BACKENDS.md` |
| rwkv.cpp native GGML | Supported CPU backend; ~2.8 tok/s FP16, ~3.5 tok/s Q5_1, and ~4.2–4.6 tok/s Q4_K best observed on 2.9B | `docs/BACKENDS.md`, `tests/test_rwkvcpp_backend.py` |
| ChatRWKV/rwkv.cpp four-way parity | Exact greedy token/text match on checked-in 0.1B prompts; configurable logit/state guardrails pass | `bench/bench_backend_conformance.py`, `tests/test_parity_conformance.py` |
| Archived non-RWKV adapters | Retained as research fixtures only; excluded from the maintained engine, CLI, HTTP, and F-tier product contract | `rwkv_ssd/backends/sequence.py`, archived backend modules |
| HTTP process workers | Bounded spawned pool, session routing/state envelopes, cancellation/restart/metrics contract covered by synthetic integration tests | `app/worker_pool.py`, `tests/test_worker_pool.py`, `tests/test_serve.py` |
| Intel XPU | LUT decode works; BF16 matrix compute unavailable | Intel Arc 140V, `torch 2.13.0+xpu`; see below |
| CUDA/GDS | Roadmap/research only | `docs/ACCELERATORS.md`, `docs/SSD_EXPLOITATION.md` |
| rwkv.cpp | Supported CPU backend; grouped-U8 native path is CPU-only and quality-qualified per pack | `docs/BACKENDS.md` |

## XPU status

The XPU environment is restored at `.venv-xpu` using the local Python 3.12
runtime and its existing XPU package set. The verified runtime is:

- Intel Arc 140V GPU, 16 GB shared/system-visible memory
- Python 3.12.10 from the venv launcher
- `torch 2.13.0+xpu`
- `torch.xpu.is_available() == True`, one device detected
- `xpu_compute_available == False`: the synchronized BF16 matrix probe fails
  with `could not make an engine with allocator`
- XPU smoke tests: **17 passed**

The supported split is therefore CPU model computation plus XPU LUT2 decode.
The engine intentionally disables CPU fused GEMV on XPU; it uses accelerator-
side LUT gathering followed by a CPU Torch forward pass. Grouped LUT2 gather
parity passes, but grouped-U8 remains outside this accelerator path.

### XPU tok/s measurements

These are short, single-sample decode-only measurements with
`RWKV_TRINITY_XPU_AUTO=1`, `RWKV_TRINITY_XPU_MIN_NUMEL=1`, `mmap`, and CPU
`bf16` model compute. F6 is a resident CPU reference and does not use XPU
decode. Full JSON evidence is in
[`bench/results/xpu_chatrwkv_lut2_20260721.json`](../bench/results/xpu_chatrwkv_lut2_20260721.json).

| Model / pack | F6 resident | F1 strict | F3 hot3 |
|---|---:|---:|---:|
| 0.1B LUT2 | 20.14 tok/s | 10.26 tok/s | 13.63 tok/s |
| 2.9B all-LUT2 | 2.79 tok/s | 0.54 tok/s | 0.69 tok/s |

The 2.9B result does not justify enabling XPU by default: F1 is effectively
the same as the CPU result, and the F3 difference is from a very short two-
token run. The normal BF16 pack remains the talking-app recommendation. The
native-safe grouped pack was not rerun on XPU because its 3.69 GB traffic is
grouped-U8, for which no XPU direct decode path is implemented.

## Production entry points

| Need | Use |
|---|---|
| Run the application | `python -m app.cli --help` |
| CPU/RWKV real-model throughput | `bench/bench_f1_f3.py` or `bench/bench_throughput.py` |
| RAM/F-tier frontier | `bench/bench_io_ceiling.py` |
| Streaming/storage matrix | `bench/bench_streaming_matrix.py` |
| Synthetic regression | `bench/bench_generate.py` |
| Pack validation | `python -m rwkv_ssd.tools.verify_pack <pack>` |
| Full test suite | `python -m pytest tests/ -q` |

See [`../bench/README.md`](../bench/README.md) for the benchmark catalog and
reporting rules. See [`README.md`](README.md) for the documentation map.

## Work queue

1. Resolve the Intel BF16 matrix-engine failure with a compatible Torch/
   oneAPI/driver pairing, then measure XPU model compute separately from LUT
   decode.
2. Continue reducing the CPU grouped-U8 GEMV/state-update floor and profile
   the vocabulary head separately; native rwkv.cpp is ~2.62 tok/s warm at its
   best measured thread count and still far from 15 tok/s.
3. Keep CUDA/XPU hardware-gated work separate from the CPU release path.
4. Treat the old all-LUT2 2.9B pack as an archived reproduction artifact; new
   native-safe packs should use the grouped-U8 matrix policy.
5. Move only clearly stale scripts/results into `archive/` after references
   are audited; organization should not silently delete research history.
