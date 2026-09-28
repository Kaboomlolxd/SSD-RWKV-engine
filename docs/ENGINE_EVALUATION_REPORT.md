# RWKV SSD Engine — Production Readiness Evaluation

**Evaluation date:** 2026-07-31 default-promotion follow-up
**Repository:** `C:\Users\Kaboom\Desktop\SSD mamba`
**Scope:** correctness, real-model inference, storage/compute backends, RAM
tiers, performance, native loading, serving, packaging, documentation, and
release risks. CUDA and other hardware-gated paths are explicitly outside the
local certification boundary.

## Executive verdict

The engine is **ready for a constrained CPU release**, but not for a
general-purpose, hardware-independent multi-backend production claim.

The core design is promising and substantial portions work: synthetic pack/provider paths are healthy; ChatRWKV with raw and grouped FP16 packs produced meaningful output and resident/streaming behavior; RAM tiers form a useful latency/memory frontier; and the test suite covers more of the runtime than the default invocation exposes.

The release boundary is intentionally narrow:

- The archived all-LUT2 and shadow packs fail the application-quality gate.
  The promoted compact selector is now the default 2.9B path and
  auto-resolves to the g32 grouped-quality pack,
  whose manifest-bound certificate covers the current short smoke.
- rwkv.cpp resident CPU inference is the default fast CPU path with a matching
  GGML model. Its provider-backed pack path now includes a bounded native
  layer-local ABI and passes the F1-F4 throughput/memory acceptance matrix;
  longer 2.9B quality certification remains open.
- The complete local CPU release gate remains green: the September 1 follow-up
  passed **600 tests, with 18 skipped** in
  `python -m pytest -q --override-ini "addopts="`; the earlier vendored native
  gate was **8/8 CTest targets** when configured with `-DGGML_CCACHE=OFF`. The
  local ccache wrapper is unreliable on this host and is not part of the gate.
- The wheel and native runtime still require an explicit asset/binary bundle;
  `rwkv-ssd-preflight` documents and checks that boundary.
- The HTTP server has bounded admission and a configurable spawned process
  worker pool with session envelopes, cancellation, restart handling, and
  aggregated metrics. It remains a local/protected service boundary rather
  than an internet-facing deployment.

Recommended status: **release the CPU/reference surface with explicit model and
pack qualification; keep long-run grouped-U8 quality claims, CUDA/XPU, and
Albatross experimental.** The g32 grouped-U8 pack is the compact 2.9B
default within its short-smoke certificate scope; dense FP16/BF16 remains the
broad quality/speed reference, while rwkv.cpp is the production fast CPU
backend within the certified boundary.

## Post-remediation status (2026-07-31)

This report records the original evaluation findings. The non-hardware-gated
remediation requested from that evaluation has now been applied:

- Native DLL resolution is deterministic and limited to the DLL directory plus
  explicit `RWKV_LUT2_DLL_DIRS` entries; sanitized-PATH regression tests pass.
- The cross-backend benchmark validates scenario schemas and completes its
  resident rwkv.cpp comparison without the former `KeyError: 'mode'` crash.
- rwkv.cpp validates native token IDs before tokenizer lookup and raises an
  actionable compatibility/codec error instead of leaking a tokenizer
  `KeyError`.
- Real RWKV-7 lossy packs require a passing, hash-bound quality
  certificate at runtime. Historical `trinity_lut2_*` and shadow payloads were
  removed from the working tree after the audit. Automatic requests for those
  names now resolve to the checked-in mixed `trinity_grouped_0.1b` pack:
  326 large tensors use `scale_u8_grouped` g256 and 76 one-dimensional
  control/normalization vectors remain dense BF16 (about 189 MiB total); if it is absent, resolution
  falls back to the exact-parity `trinity_safe_0.1b` FP16/BF16 pack. Explicit
  LUT2 selection reaches only a separately rebuilt legacy pack and fails closed
  when that payload or its certificate is absent.
- The mixed grouped g256 pack is the best current compact CPU candidate and is
  intentionally the default requested by this release pass. Keeping the 76
  one-dimensional control/normalization vectors dense removes the recurrent
  drift caused by quantizing those vectors, without increasing the aligned
  198,250,504-byte payload. It passes the current short real-model
  resident-vs-streaming greedy parity checks with fused decode both disabled
  and enabled; use the dense fallback for longer reference comparisons.
- The rwkv.cpp CPU pass removes duplicate provider uploads on prefix-cache
  misses and follow-up decoding, with one weight-stationary synchronization by
  default. `RWKVCPP_SYNC_EVERY_TOKEN=1` remains available for strict upload
  diagnostics.
- The four-way ChatRWKV/rwkv.cpp conformance harness passes exact greedy
  token and text parity on the checked-in 0.1B certification fixture. Its
  configurable top-10, KL, recurrent-state, state-restore, follow-up,
  sampling, cancellation, and deadline checks are covered by the full suite.
- The native rwkv.cpp layer ABI now has a reusable one-block plan, dense and
  grouped-U8 layer uploads, chunked prefill, bounded native slots, and
  eviction-aware provider scheduling. The 0.1B F1-F4 cold/warm rows pass the
  60/80/80/80 percent gates; the 2.9B diagnostic also passes throughput and
  memory gates while remaining quality-gated.
- The grouped-U8 fused CPU GEMV now caches the active group scale/min-max pair
  per output row and avoids a per-element group division. The rebuilt native
  LUT DLL is portable by default (no required OpenMP runtime); the native
  loader is confirmed to load and the grouped GEMV tests remain green.
- The checked-in grouped pack passes structural/hash verification: 402 tensors,
  198,250,504 bytes, with 326 `scale_u8_grouped` g256 tensors and 76 dense
  one-dimensional tensors. Its weights SHA-256 is
  `7a8045d04d71df792c6c79a11893fbfe8a5400d12b3d287f2d1dba5e772ff7fa`.
- A 2.9B follow-up probe now uses the same native-safe principle: 580 dense
  non-matrix control tensors and 482 grouped-U8 g32 matrices, including the
  embedding and output head. The default selector
  `runtime_pack_2.9b` resolves to the grouped-quality payload, which is
  3,689,844,744 bytes and passes the corrected three-prompt native smoke with
  a manifest-bound short-smoke certificate: maximum KL `0.010936`, minimum
  top-10 overlap `0.90`, and maximum state relative L2 `0.034156`. Long-run/held-out
  quality certification remains open. The previous g64 artifact recorded KL
  `0.07539133727550507` against a configured maximum of `0.05`.
  The current F-tier matrix measured 1.90-2.25 tok/s for F1-F4 cold/warm
  rows under the explicit RSS/provider/native budgets. The best local CPU
  resident results were approximately 2.8 tok/s for FP16 rwkv.cpp,
  3.5 tok/s for Q5_1, and 4.2–4.6 tok/s for Q4_K; an earlier grouped-U8
  provider run was about 1.36 tok/s warmed end-to-end and is superseded by
  the bounded native F-tier matrix. The 15 tok/s target was not reached.
- The HTTP server now has bounded admission, body limits, API-key/Bearer auth,
  explicit CORS origins, health diagnostics, Prometheus-style metrics, and a
  bounded spawned process-worker pool with stable session routing, versioned
  state envelopes, cancellation acknowledgements, and bounded worker restart.
- Mamba2 and Transformer adapters now expose the common generation, streaming,
  follow-up, state snapshot/restore, seeded sampling, cancellation/deadline,
  metrics, prefix-reuse, and qualified batching contract against their own
  resident references. Kimi-K3-0.18B now has a resident CPU KDA/MLA+sparse-MoE
  adapter that runs without Triton through a narrow PyTorch compatibility
  layer; FP16/BF16 state restore and explicit low-RAM capability boundaries are
  covered by the opt-in real-checkpoint contract.
- Native LUT2 assets are package data, `rwkv-ssd-preflight` validates external
  runtime assets, and the deployment example now names the CPU reference
  backend and required checkpoint explicitly.
- A Windows CPU-only CI workflow covers default/software tests, native loader
  isolation, wheel contents, preflight, and benchmark exit status.

The follow-up native rwkv.cpp pass now implements grouped-U8 directly in the
vendored C++ backend. The `SG8\x01` loader validates magic, group size, exact
payload length, finite min/max metadata, and shape; packed-only mode omits
large matrix payloads from dense GGML storage. The Python bridge uploads all
manifest layers, including `emb.weight` and `head.weight`, while retaining
dense storage for small controls. A real 2.9B packed-only smoke now evaluates
successfully after the global-matrix upload fix. The historical g64 warm native
decode measured approximately 1.83 / 2.45 / 2.62 / 2.19 tok/s at 1 / 2 / 4 /
8 threads. That historical g64 one-prompt quality recheck reached top-10
overlap 0.80 and state relative L2 0.0176, but KL 0.0754 exceeded the 0.05
gate. The current g32 three-prompt smoke passes those short-run gates; a
longer held-out certificate remains open.

The archived all-LUT2 Trinity codec remains blocked from production because its
candidate generated incorrect real-model tokens. The g32 grouped-U8 pack is now
the automatic 2.9B compact direction within its short-smoke certificate scope.
Native provider-backed rwkv.cpp layer streaming is implemented and
acceptance-tested, but long-run 2.9B generation quality remains open.
CUDA/XPU/MPS/Albatross and other accelerator validation is intentionally still
hardware-gated and is not represented as completed here.

## CPU continuation update (2026-09-01)

The non-hardware-gated CPU continuation added and validated grouped decode
vectorization, direct DeepEmbed sidecar row reads, shared-layer qkv/DEA prompt
prefill/decode reference batching, recurrent-state publication elision, native
sequence scratch reuse, whole-pack zstd cold storage, and the packed native
vocabulary-head memory option. Focused A/B probes retained only the options
whose overhead was worthwhile: grouped-U8 decode improved **5.05x**, grouped
LUT2 decode **4.63x**, and DeepEmbed sidecar lookup was approximately
**19x–1,202x** faster depending on access pattern. The two-session DeepEmbed
batch probe improved wall time **1.17x** with **50% fewer layer loads**; focused
multi-layer probes reached **1.28x–1.75x**.

The six-prompt, 32-position-per-prompt native quality probe passed its
configured gates (top-10 **0.90**, KL **0.010936**, state relative L2
**0.034156**), but it is supplemental evidence rather than a replacement for
the existing manifest-bound three-prompt/eight-position certificate. The zstd
probe reduced the 2.9B physical weights file from **3,689.8 MB to 3,128.8 MB**
but increased load-plus-full-read time from **2,842.2 ms to 8,898.5 ms**, so
raw mmap remains the hot-path default. No additional safe CPU optimization was
promoted after these A/B measurements; the remaining work is qualification and
hardware-gated acceleration.

## Scope and environment

The original evaluation was performed against a heavily dirty worktree. The
subsequent remediation pass intentionally changed the CPU source, tests, native
LUT binary, and documentation; those changes are summarized in the
post-remediation status above.

Observed environment:

| Area | Result |
|---|---|
| OS | Windows 11 |
| Python | 3.14.3 |
| PyTorch | 2.13.0, CPU build |
| Torch threads | 8 by default for the tier run |
| CUDA / XPU / MPS | Unavailable in this environment |
| ChatRWKV | Available through the pinned ChatRWKV submodule |
| rwkv.cpp | DLL available at backends\rwkvcpp_ref\build\bin\librwkv.dll |
| Albatross | Unavailable and not wired into this environment |
| madvise / posix_fadvise / io_uring | Unavailable |

This is meaningful CPU coverage, but it is not a hardware certification for GPU or accelerator paths.

## Test evidence

### Automated tests

The default pytest configuration excludes chatrwkv, slow, and integration markers in pyproject.toml.

| Run | Result | Interpretation |
|---|---:|---|
| Complete local Python suite (`--override-ini "addopts="`) | **600 passed, 18 skipped** (618 collected) | September 1 CPU continuation gate; includes parity, sequence-backend, Kimi-K3, serving, worker-pool, state-envelope, native bridge, and default-pack profile coverage |
| Focused rwkv.cpp/backend bridge gate | **27 passed, 1 deselected** | Thread policy, native bridge, tokenizer guard, and provider synchronization covered |
| rwkv.cpp native CTest gate | **8 passed** | Configured with `-DGGML_CCACHE=OFF`; includes the layer-local plan/upload/chunked-prefill/eviction ABI target |
| Native SG8 ABI smoke | **Pass** | Bad magic, short/truncated payloads, and NaN metadata rejected; valid record accepted |
| Native/real-model follow-up | **Pass (short-smoke + supplemental long probe)** | Six-prompt/32-position teacher-forced probe also passes; manifest-bound certificate remains scoped to three prompts/eight positions and held-out/free-running quality remains open |
| Python compilation | Pass | compileall completed successfully |
| Dependency consistency | Pass | pip check completed successfully |
| Ruff / mypy / bandit / pip-audit | Not installed | Static/security audit was not available in the environment |
| Windows CPU CI workflow | Present | Default/software, native loader, wheel, preflight, and benchmark smoke jobs are defined |

The passing suite is useful evidence of breadth, but the repository still
requires real-model, backend, native, and codec tests to be run explicitly in
release CI. The local numbers are not a CUDA/XPU/MPS certification.

## Original findings (before remediation)

The following subsections preserve the full audit trail from the July 23
evaluation. Each issue was either fixed, quarantined, or narrowed into an
explicit release boundary below.

### Confirmed failures and bugs

#### 1. Windows native DLL dependency isolation — resolved

rwkv_ssd/native/lut2_gather_loader.py:108-153 claims to avoid walking the process PATH, but it appends fixed locations including:

~~~text
C:/Strawberry/c/bin
C:/msys64/ucrt64/bin
C:/msys64/mingw64/bin
~~~

If those locations exist and contain libgomp-1.dll, they are returned even when the test sanitizes PATH or passes an explicit dependency directory. The failures are:

~~~text
test_windows_dependency_search_does_not_walk_process_path: failed
test_windows_dependency_search_accepts_explicit_directory: failed
~~~

This makes deployment non-deterministic and can load a compiler/runtime DLL from outside the application bundle. Severity is **P1**, or **P2** for a tightly controlled local deployment.

#### 2. Cross-backend benchmark harness crash — resolved

bench/bench_backend_compare.py:157-174 adds an rwkv.cpp scenario with backend but no mode, then unconditionally reads spec["mode"]. The script raises:

~~~text
KeyError: 'mode'
~~~

ChatRWKV rows run before the crash, but the documented comparison never completes. Severity: **P1 for release/benchmark tooling**.

#### 3. rwkv.cpp + Trinity streaming failure — narrowed, not fully certified

The resident rwkv.cpp path works with a matching FP16 GGML file, and raw FP16 pack streaming produced valid output. The Trinity pack-backed rwkv.cpp streaming path fails during decoding with:

~~~text
KeyError: 0
~~~

The error is raised by the rwkv.cpp world tokenizer because token ID 0 is not present in rwkv_vocab_v20230424.txt. The exact upstream cause is not isolated: likely candidates include Trinity codec quality, the pack-to-GGML bridge, or invalid native output handling. Regardless of the root cause, the advertised path is not certifiable. Severity: **P0/P1 release blocker**.

Post-remediation, `decode_text()` converts this native tokenizer failure into
an actionable compatibility error, and the provider path no longer performs
the duplicate synchronization found on prefix-cache misses, follow-up
decodes, and every token by default. A fresh 0.1B CPU run completed F1/F3/F6
rwkv.cpp scenarios; this closes the crash, not the pack-quality gate.

Relevant implementation areas include rwkv_ssd/backends/rwkvcpp.py:302-305, rwkv_ssd/backends/rwkvcpp.py:429-557, and the bridge calls in rwkv_ssd/runtime/engine.py:943-944 and :1702.

#### 4. All-LUT2 Trinity inference quality — quarantined

On the real 0.1B checkpoint:

- Resident raw pack: meaningful greedy output.
- Grouped FP16 pack: exact resident parity in the tested case.
- trinity_lut2_0.1b: repeated token 43538 starting at the first generated token.
- trinity_lut2_shadow_0.1b: the same repeated-token behavior.
- Exact 8-token parity: false, first difference at position 0.
- 32-token parity: false, first difference at position 0.

The corresponding bench/results/rwkvcpp-quality-lut2-symmetric-g64.json artifact fails top-k overlap, KL, and state-drift gates. A separate native_safe candidate passes its own quality gates, but it is not the current trinity_lut2_0.1b production candidate. Severity: **P0** for Trinity promotion.

## Backend and storage assessment

| Backend or mode | Verified result | Production assessment |
|---|---|---|
| ChatRWKV + raw/FP16 grouped packs | Real 0.1B inference works; grouped FP16 has exact resident parity in the tested case | Best current reference path; limited CPU/0.1B evidence only |
| ChatRWKV F1/F2/F3/F5/F6 RAM tiers | Functional latency/memory frontier; see measurements below | Promising beta capability, subject to real quality gates and larger-model tests |
| Native pack/provider path | Synthetic correctness and performance tests pass | Good regression target; needs clean artifact/release validation |
| rwkv.cpp resident + matching GGML | Functional; 2.9B best observed approximately 2.8 tok/s FP16, 3.5 tok/s Q5_1, 4.2–4.6 tok/s Q4_K | Usable CPU backend; native thread count is model-width aware and explicit overrides remain available |
| rwkv.cpp raw FP16 pack streaming | Valid output observed | Experimental provider bridge; use parity-qualified packs only |
| rwkv.cpp grouped-U8 streaming | Native SG8 packed-only graph evaluates correctly; historical g64 warm decode peaked around 2.62 tok/s at 4 threads, while current g32 short-smoke KL is 0.010936 | CPU experimental / pack-quality gated |
| Mamba2 pack backend | Common generation/streaming/state/sampling/cancellation/metrics contract and qualified layer-stationary batch path pass self-reference tests | CPU/reference surface; broader checkpoint and performance qualification remains open |
| Transformer/Llama pack backend | Common generation/streaming/KV-state/sampling/cancellation/metrics contract and qualified equal-length batch path pass self-reference tests | CPU/reference surface; model-format coverage remains capability-gated |
| CUDA / XPU / MPS | Not available for this evaluation | No production claim can be made |
| Albatross | Not available and not wired | Experimental/unavailable |

The engine has a reasonably rich storage strategy surface: resident weights, partial residency, strict SSD streaming, bounded/fused provider caching, mmap/read-style I/O, prefetch, and state parking/caching. The important remaining issue is not feature count; it is proving the same model quality and operational behavior across every advertised combination.

## Performance and RAM tiers

### Historical ChatRWKV tier run

The source `trinity_lut2_0.1b` payload for this run has been removed after
failing the quality gate. Rebuild the pack separately if this historical
measurement needs to be reproduced.

Command shape:

~~~text
# Historical command shape; the source pack was removed after this run.
bench_f1_f3.py --backend chatrwkv --pack C:\prepared\reference-pack --checkpoint C:\models\rwkv-model.pth --tiers F6,F1,F2,F3,F5 --max-tokens 16 --samples 2
~~~

CPU-only, 8 Torch threads. Rates are **decode-only tok/s**; prompt prefill is excluded. z is the runtime resident/weight footprint reported by the benchmark, provider is provider-side cached memory, and tracked is their sum.

| Tier | Strategy | tok/s | z MB | Provider MB | Tracked MB |
|---|---|---:|---:|---:|---:|
| F6 | Resident | 14.67 | 382.1 | 0.0 | 382.1 |
| F1 | Strict SSD | 8.58 | 201.3 | 39.3 | 240.6 |
| F2 | Bounded fused | 9.53 | 203.2 | 39.3 | 242.5 |
| F3 | Partial hot3 | 11.82 | 261.6 | 26.2 | 287.8 |
| F5 | Promote/full z | 12.64 | 382.2 | 0.0 | 382.2 |

The measured frontier is directionally sensible: F1 cuts tracked memory by roughly 37% versus resident at a roughly 41% decode-rate reduction; F2 recovers some throughput with nearly the same footprint; F3 buys most of the resident speed while using less memory; and F5 converges on resident behavior. These are useful planning numbers, not a quality-certified Trinity result, because the tested Trinity pack emits degenerate output.

### Other performance runs (earlier reference measurements)

| Workload | Mode | Result | Notes |
|---|---|---:|---|
| Synthetic pack | Resident | ~4,179 tok/s | Regression/streaming-tax measurement; not comparable to real RWKV |
| Synthetic pack | Partial | ~1,845 tok/s | Approximately 44% of synthetic resident rate |
| Synthetic pack | Streaming | ~1,048 tok/s | Approximately 25% of synthetic resident rate |
| rwkv.cpp real 0.1B FP16 GGML | 1 thread | ~18.9 tok/s | 16 generated tokens, 2 samples |
| rwkv.cpp real 0.1B FP16 GGML | 8 threads | ~7.8 tok/s | More threads were slower; OpenMP overhead dominates this small model |
| Mamba2 real pack | Resident | ~176.6 tok/s | Fixed 16-token CPU test, repeated twice |
| Mamba2 real pack | Partial | ~175.0 tok/s | Fixed-token preview measurement |
| Mamba2 real pack | Streaming | ~168.4 tok/s | Fixed-token preview measurement |
| Mamba2 weight-stationary batch B=4 | Streaming | ~1,609 aggregate tok/s | 64 output tokens; batch throughput, not single-stream latency |
| Llama/Transformer real pack | Resident | ~543.6 tok/s | SDPA off; fixed 16-token CPU test |
| Llama/Transformer real pack | Partial | ~522.0 tok/s | Fixed-token preview measurement |
| Llama/Transformer real pack | Streaming | ~384.4 tok/s | Fixed-token preview measurement |

The numbers show that SSD streaming and provider caching can be useful on the tested architecture, but they do not yet establish production SLOs. A release benchmark should add cold-load latency, prompt prefill, steady-state rate, p50/p95 inter-token latency, memory high-water mark, concurrent requests, and disk-class metadata.

### rwkv.cpp 2.9B CPU pass

The rwkv.cpp pass was measured separately from ChatRWKV/PyTorch F-tier
numbers. The available DLL reported AVX, AVX2, FMA, and F16C, with no AVX512
or NEON. The best observed resident rates were:

| Artifact | Best observed result | Status |
|---|---:|---|
| FP16 GGML | ~2.8 tok/s | Quality/speed reference for the native backend |
| Q5_1 GGML | ~3.5 tok/s | Experimental quantized comparison; short greedy run diverged from FP16 |
| Q4_K GGML | ~4.2–4.6 tok/s | Fastest observed native artifact; short greedy run matched only a prefix |
| Q4_0 GGML | ~3.7 tok/s | Experimental comparison |
| Grouped-U8 provider pack (earlier run) | ~1.36 tok/s warmed end-to-end | Compact SSD direction; superseded by the bounded native layer F-tier matrix |
| Native grouped-U8 rwkv.cpp packed-only | ~1.83 / 2.45 / **2.62** / 2.19 tok/s at 1 / 2 / 4 / 8 threads | Warm post-upload decode; 4 threads best on the measured AVX2 host |

The native thread policy now uses manifest width before model construction:
one thread below width 1280, two to four threads for medium and large widths,
and no automatic eight-thread oversubscription on the memory-bound 2.9B host.
`RWKV_CPU_THREADS=N` remains authoritative. These results do not reach the
15 tok/s target; materially higher throughput is hardware-gated.

The current bounded native layer acceptance run is the authoritative F-tier
measurement: the 0.1B fixture measured F1/F2/F3/F4 at 51.26/53.31/52.63/50.06
cold tok/s and 50.98/52.13/49.69/50.46 warm tok/s, versus F5 at
54.15/53.07. The 2.9B grouped-U8 diagnostic measured 2.02/2.02/2.24/2.25
cold and 1.90/1.99/1.90/2.12 warm tok/s for F1-F4, with provider-owned and
mmap-backed bytes reported separately. See `PROJECT_STATUS.md` for the full
memory and ratio tables.

## Serving and operational readiness

app/serve.py and the worker-pool integration pass synthetic health,
OpenAI-style chat, streaming SSE, session parking, queue saturation,
cancellation, state incompatibility, and worker-restart tests. The service has
bounded admission/body handling, API-key/Bearer hooks, explicit CORS origins,
backend/pack health diagnostics, and Prometheus-style counters. The default is
one spawned worker for memory safety; `--workers N` / `RWKV_SSD_WORKERS` allow a
validated bounded pool. The service should still be classified as a **local or
protected service boundary**, not an internet-facing multi-tenant deployment.

Operational limitations:

- Each worker owns one engine/native backend instance; bounded process workers
  provide concurrent execution while avoiding shared allocator/model state.
  Stateless requests use round-robin assignment and session requests use
  stable routing/state envelopes.
- Authentication and CORS are opt-in deployment controls and require an
  explicit operator configuration; they are not a substitute for a gateway.
- Queue wait plus generation deadlines, cancellation acknowledgements, worker
  health/restart, and per-worker RSS/latency aggregation are covered by the
  local integration contract. TLS, external rate limiting, and multi-host
  deployment remain intentionally out of scope.

Recommended deployment boundary: bind to localhost or a protected private
network, place a real gateway/auth layer in front, and treat the current
server as a local/operator interface rather than a public multi-tenant edge.

## Packaging and release engineering

A no-isolation wheel build succeeded and produced:

~~~text
rwkv_ssd-0.6.16-py3-none-any.whl
~~~

The wheel now includes the Python package data for the portable LUT2 source and
native asset, while rwkv.cpp binaries, converted GGML checkpoints, tokenizer
assets, and ChatRWKV remain external runtime dependencies. The release
contract is therefore a wheel plus an explicit runtime/model bundle, validated
by `rwkv-ssd-preflight`; it is not a self-contained model distribution.

`deploy/example_config.yaml` names the backend, checkpoint, and required pack
inputs rather than pretending that a bare wheel can infer them. The current
backend document uses one readiness table and marks the provider-backed
rwkv.cpp path experimental until its longer quality certificate is complete.

## Original prioritized blockers and current disposition

| Priority | Finding | Impact | Exit condition |
|---|---|---|---|
| P0 | All-LUT2 output degenerates | Incorrect model behavior can pass a speed benchmark while producing unusable text | **Quarantined.** Grouped-quality pack is the compact direction; old LUT2 remains opt-in only |
| P0/P1 | rwkv.cpp Trinity streaming failed token decoding | Advertised backend/codec combination could crash during generation | **Narrowed.** Native ID guard and synchronization fixes land; longer quality certificate remains open |
| P1 | Cross-backend benchmark crashed on missing mode | Release comparisons were incomplete | **Closed.** Scenario validation and the rwkv.cpp row now complete |
| P1 | DLL loader searched fixed external compiler paths | Non-deterministic native dependency resolution | **Closed.** Search is explicit and isolated by regression tests |
| P1 | Wheel omitted native/backend assets | Deployment from the published artifact was incomplete | **Documented.** Wheel + explicit runtime/model bundle is the contract; preflight checks it |
| P1 | No visible full release CI matrix | Real backend/codec regressions could land behind fast defaults | **Improved.** Windows CPU workflow covers default/native/package/preflight smoke; accelerator CI remains open |
| P1 | Docs and deployment example overclaimed capability | Operators could select a failing default | **Closed in this pass.** Canonical status and backend boundaries were consolidated |
| P2 | Unicode CLI help fails under default cp1252 console | Windows operator experience is brittle | Emit UTF-8-safe help or use ASCII punctuation in CLI output |
| P2 | Local server lacked production controls | Unsafe direct exposure and poor observability | **Closed for the local service boundary.** Body/request limits, auth hooks, diagnostics, metrics, bounded queueing, cancellation, and a spawned worker pool are covered; TLS, external rate limiting, and multi-host deployment remain separate |
| P2 | Hardware coverage incomplete | GPU/accelerator regressions are invisible | Add CUDA/XPU/MPS/Albatross runners or explicitly remove those claims |

## Recommended release tiers

### Tier A — constrained CPU preview / release candidate

Support rwkv.cpp as the fast CPU backend and retain ChatRWKV as the RWKV-7
correctness oracle on explicitly tested checkpoint/pack pairs, with a
documented CPU configuration and reference greedy-output fixture. Resident,
provider, and SSD-tier modes can be offered where the pair has passed parity
and memory tests.

This tier is the strongest current candidate, but it should not be called a general model/backend guarantee. The present real evidence is centered on the 0.1B checkpoint.

### Tier B — beta memory tiers

Offer F1/F2/F3/F5/F6 as a beta feature with clear memory/latency expectations. Keep the tier planner and provider accounting, but require per-pack quality certification before enabling a tier automatically. Report decode-only and end-to-end metrics separately.

### Tier C — experimental resident native backend

The native rwkv.cpp resident and layer-local provider paths are functional for
an experimental CPU release. The native grouped-U8 packed-only graph can be
offered for the experimental all-grouped pack after upload, but it must carry the
same checkpoint/pack/tokenizer certificate and should not be called
quality-equivalent to FP16 until the KL and longer-generation gates pass. Do
not promise that higher thread counts improve throughput.

### Tier D — experimental / blocked

Keep removed Trinity LUT2/shadow payloads and long-run grouped-U8 quality
claims, Kimi-K3 low-RAM layer streaming (the resident CPU path is supported),
CUDA/XPU/MPS, and Albatross out of the fully supported matrix until they have
the required quality, packaging, and hardware evidence. Mamba2/Transformer
common engine behavior is covered, while broader model-family qualification
remains explicit.

## Follow-up roadmap

### P0: extend the quality certificate

1. Keep all-LUT2 promotion blocked unless a pack certificate passes greedy
   parity, top-k overlap, KL, state-drift, and repetition checks.
2. A supplemental six-prompt/32-position teacher-forced probe passes the
   configured gates; review its ignored local artifact before changing the
   manifest-bound certificate, then extend qualification to held-out,
   sustained, and free-running generation.
3. Add a longer real-model rwkv.cpp provider-streaming certificate, including
   tokenizer compatibility and quality thresholds.
4. Continue profiling native grouped-U8 direct GEMV, transposed adapter GEMV,
   recurrent state updates, and the vocabulary head separately; the packed
   vocabulary-head A/B is slower than dense BLAS, while the current four-thread
   result remains below the 15 tok/s target.

### P1: make the runtime bundle reproducible

1. Keep the explicit wheel + runtime/model bundle contract and record loaded
   DLL paths in diagnostics.
2. Add clean-venv real-model smoke coverage to CI, not only software/native
   tests.
3. **Implemented in the CPU preflight path.** `rwkv_ssd.tools.preflight` now
   reports pack identity, codec/layout, source checkpoint hash (and checks it
   against the pack when declared), tokenizer asset fingerprint/version, and
   the loaded native library hash plus bridge ABI label. Publication of a
   complete release bundle remains open.
4. Add longer marked real-model/backend jobs when model artifacts are
   available to the runner.

### P2: improve product behavior and scale

1. Make CLI output UTF-8-safe on Windows.
2. **Implemented for the local service boundary.** Structured metrics cover
   load/prefill/decode, I/O wait, provider/resident bytes, queue wait,
   cancellations, failures, worker health, latency percentiles, and RSS.
3. **Implemented.** The spawned worker pool provides bounded queueing,
   per-session routing/state envelopes, cancellation, deadlines, admission
   control, and bounded worker restart; external gateway hardening is separate.
4. **Implemented.** Health checks cover backend/pack/native/tokenizer and
   memory-related diagnostics; storage-class and hardware measurements remain
   deployment-specific.
5. Continue cold/warm model-size and storage-class benchmarking with p50/p95
   latency and memory high-water reporting.
6. **Common contract implemented.** Continue real-checkpoint tokenizer,
   prompt, batch, and architecture-specific parity qualification for Mamba2
   and Transformer assets.

## Suggested production acceptance gates

Before calling a backend/codec combination production-ready, require all of the following for each supported checkpoint family:

- Clean-install smoke test from the published artifact and documented asset bundle.
- Deterministic model/pack/tokenizer/native ABI compatibility check before generation.
- Resident versus streaming greedy parity on at least 8 and 32 generated tokens for multiple prompts.
- Quantitative logit/top-k/KL/state-drift thresholds, plus a repetition/invalid-token guard.
- Benchmark completion with no skipped scenario caused by a malformed spec.
- Cold load, prompt prefill, steady decode, p50/p95 inter-token latency, and memory high-water measurements.
- At least two samples and 16 generated tokens for headline throughput claims when the model permits it.
- Concurrency and overload tests for the serving surface, or an explicit local-only classification.
- Native dependency isolation test on a clean Windows environment.
- Reproducible CI result tied to checkpoint, pack, code revision, device, thread count, I/O backend, and sampling policy.

## Reproducibility notes

Representative commands used or supported by the repository:

~~~powershell
# Fast/default suite (the repository's pytest defaults exclude marked real-model tests)
python -m pytest

# Full marked coverage should be run explicitly in release CI
python -m pytest -m ""

# Static import/bytecode and dependency checks
python -m compileall -q rwkv_ssd app bench tests
python -m pip check

# Historical real ChatRWKV measurement shape; rebuild the source pack first.
python bench\bench_f1_f3.py --backend chatrwkv --pack C:\prepared\reference-pack --checkpoint C:\models\rwkv-model.pth --tiers F6,F1,F2,F3,F5 --max-tokens 16 --samples 2

# Cross-backend comparison (must complete every declared row)
python bench\bench_backend_compare.py

# Windows help smoke test (use UTF-8 output on cp1252 consoles)
$env:PYTHONIOENCODING = "utf-8"
python bench\bench_io_ceiling.py --help

# Wheel build shape used for the packaging check
python -m build --wheel --no-isolation
~~~

The full-suite command should be verified against the repository's marker/addopts behavior in CI; the important release requirement is that marked real-model/backend tests are not silently omitted.

## Evidence paths

Key code and test locations:

- tests/test_native_loader.py — Windows dependency-search isolation.
- bench/bench_backend_compare.py — cross-backend benchmark scenario validation.
- bench/bench_f1_f3.py — RAM-tier benchmark.
- rwkv_ssd/native/lut2_gather_loader.py — native DLL resolution.
- rwkv_ssd/backends/rwkvcpp.py — resident/provider bridge and tokenizer-facing generation.
- rwkv_ssd/runtime/engine.py — runtime/native bridge calls.
- app/serve.py — HTTP serving surface.
- pyproject.toml — package data, pytest defaults, and optional dependencies.
- deploy/example_config.yaml — deployment default configuration.
- docs/BACKENDS.md — backend capability documentation and readiness boundary.
- bench/results/rwkvcpp-quality-lut2-symmetric-g64.json — failing LUT2 quality artifact.
- bench/results/rwkvcpp-quality-default-native-safe.json — passing comparison candidate that is not the current Trinity LUT2 pack.
- the archived reference pack's manifest.json and meta.json — archived codec metadata.
- bench/RESULTS.md — benchmark metadata and comparison policy.

## Final assessment

Native-U8 evidence is implemented in
`backends/rwkvcpp_ref/rwkv_native_u8.inc`; the historical g64 one-prompt
recheck is recorded in
`bench/results/rwkvcpp_native_u8_2.9b_g64_1prompt.json`, and the current g32
three-prompt smoke is recorded in
`bench/results/rwkvcpp_native_u8_2.9b_g32_3prompt.json`; the stable default
selector recheck is recorded in
`bench/results/rwkvcpp_native_u8_2.9b_default_3prompt.json`. A later local
six-prompt/32-position teacher-forced probe is recorded in the ignored working
tree artifact `bench/results/rwkvcpp_native_u8_2.9b_g32_6prompt_32tf.json`.

The project now has a credible CPU release surface and a useful SSD/RAM-tier
engine. The important production decision is the boundary: dense FP16/BF16 is
the broad quality/speed reference, while the g32 grouped-quality pack is the
default compact direction within its explicit short-smoke certificate scope.
The native rwkv.cpp grouped-U8 graph has explicit ABI validation, but longer
generation quality remains open; historical all-LUT2 and accelerator paths are
experimental. This prevents a fast but degenerate codec from being presented
as a general inference result.

The remaining work is primarily qualification rather than another broad feature
pass: longer 2.9B quality certificates, a reproducible runtime/model bundle,
and hardware-specific throughput gates. On the measured CPU, the 15 tok/s
2.9B goal remains unmet; the best native Q4_K result was approximately
4.2–4.6 tok/s. The source, tests, docs, and local CI now reflect that reality.
