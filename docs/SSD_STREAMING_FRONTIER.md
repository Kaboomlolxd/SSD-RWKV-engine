# SSD streaming frontier

This document is the implementation contract for SSD streaming. It separates
what is implemented on the CPU/Windows development path from work that still
needs the target Linux/NVIDIA or physical multi-SSD hardware.

## Compression quality frontier (CPU/Windows, experimental)

Trinity now supports `groupwise_kmeans`: two-bit indices with a learned
four-value codebook per contiguous weight group. This follows the broad
groupwise/local-codebook lesson from [GPTQ](https://arxiv.org/abs/2210.17323)
and [AQLM](https://arxiv.org/abs/2401.06118), while remaining a repository-
native LUT codec. It is not an implementation of either paper. The optional
`balanced` quality preset follows the mixed-precision/saliency lesson of
[AWQ](https://arxiv.org/abs/2306.00978), [SpQR](https://arxiv.org/abs/2306.03078),
and [SqueezeLLM](https://arxiv.org/abs/2306.07629): do not spend the same bit
budget on every tensor.

Implemented:

- Historical all-LUT2 packs were removed after their quality gates failed;
  benchmark results remain archived. Automatic requests for `trinity_lut2_*` now select the checked-in
  mixed `trinity_grouped_0.1b` compact pack (326 large tensors use
  `scale_u8_grouped` g256; 76 one-dimensional control/normalization vectors
  remain dense BF16), with the dense `trinity_safe_0.1b` pack as a fallback
  when it is absent. Use
  `RWKV_PACK_PROFILE=lut2` only when intentionally reproducing a separately
  rebuilt all-LUT2 path.
- The checked-in mixed grouped g256 pack is about 189 MiB for the 0.1B model.
  It is the best current compact CPU candidate and passes the current short
  real-model resident-vs-streaming greedy parity test with fused decode off/on.
- Selecting `--pack-codec trinity_lut2` for a new build still defaults to the
  native-safe mixed build policy: block tensors use grouped U8 and sensitive
  non-block tensors retain the safer representation. Use
  `--trinity-quality-preset legacy` only when intentionally reproducing an
  all-LUT2 research pack.
- Group sizes 64/128/256/512 can be A/B tested with deterministic learned
  codebooks.
- The A/B schema reports encoded bytes, compression ratio, weighted tensor
  RMSE, activation-output RMSE, and machine-readable gates.
- `--trinity-quality-preset balanced` keeps vectors and small/control tensors
  dense, stores embedding/head tensors with `scale_u8`, and applies grouped
  two-bit coding only to large projection matrices.
- Grouped-U8 blobs decode through normal layer spans, striped packs, ChatRWKV,
  and the real rwkv.cpp `rwkv_set_tensor` upload ABI. CPU ChatRWKV may use the
  grouped-U8 fused GEMV path; the older all-LUT2 grouped-codebook path remains
  separate and is not promoted by this default.
- `bench/bench_rwkvcpp_quality.py` compares resident FP16 and provider-backed
  native logits/state, with top-k, KL, and recurrent-state gates.
- `scale_u8_grouped` provides outlier-resistant affine 8-bit weights with a
  min/max pair per configurable group. It is supported by ordinary and span
  decoding and by provider-backed rwkv.cpp uploads.

Measured on the local RWKV-7 0.1B checkpoint (402 tensors):

- Global K-means LUT2: weighted RMSE `0.02797`, approximately 47 MiB payload.
- Groupwise K-means g128: weighted RMSE `0.01990` (about 29% lower),
  approximately 70 MiB payload.
- The all-tensor g128 pack still **failed** native sequence gates (top-10
  overlap 0, maximum KL 9.82, state relative-L2 drift 1.29). It is therefore
  experimental and is not selected as a default codec.
- Per-tensor `scale_u8` also failed (`0.50` minimum top-10 overlap, `1.40`
  maximum KL, `0.202` state drift). Groupwise U8 g128 nearly passed, and g64
  passed the declared two-prompt native gate: minimum top-10 overlap `0.80`,
  maximum KL `0.0201`, maximum state drift `0.0557`. The g64 pack was 206.22
  MiB versus the approximately 365 MiB FP16 payload. This is an initial gate,
  not a perplexity claim; broader held-out prompts remain required.

Build and evaluate a quality-first pack:

```powershell
# Default quality-gated mixed path. The explicit size flags shown here are the
# defaults and may be omitted.
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.pack_runtime `
  --input .\model.pth --output .\pack-native-safe `
  --pack-codec trinity_lut2 `
  --trinity-group-size 128 --scale-group-size 64

# Older balanced experimental policy.
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.pack_runtime `
  --input .\model.pth --output .\pack-g128-balanced `
  --pack-codec trinity_lut2 --trinity-codebook groupwise_kmeans `
  --trinity-group-size 128 --trinity-quality-preset balanced `
  --pack-layout layer_grouped

.venv-cpu\Scripts\python.exe bench\bench_compression_ab.py `
  --checkpoint .\model.pth --group-sizes 64,128,256,512 `
  --json-out .\bench\results\compression-ab.json

.venv-cpu\Scripts\python.exe bench\bench_rwkvcpp_quality.py `
  --checkpoint .\model-FP16.bin --candidate-pack .\pack-g128-balanced `
  --strict --json-out .\bench\results\native-quality.json

# Quality-first affine option that passed the local native smoke gate.
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.pack_runtime `
  --input .\model.pth --output .\pack-u8-g64 `
  --pack-codec scale_u8_grouped --trinity-group-size 64 `
  --pack-layout layer_grouped
```

Next research candidates, in priority order:

1. Extend the implemented activation-RMS calibration into sampled Hessian or
   covariance statistics, then optimize recurrent drift rather than only local
   projection error.
2. Extend the implemented tensor-level mixed-codec planner to group-level
   promotions driven by measured native KL/state-drift reduction per byte.
3. Sparse residual sidecars selected against real RWKV activations, accepted
   only when the native logit/state gate passes at a declared byte budget.
4. Error-cancelling paired-layer optimization: tune adjacent layer codebooks
   jointly against recurrent-state drift instead of minimizing each weight
   tensor independently.
5. Randomized Hadamard/incoherence processing inspired by
   [QuIP#](https://arxiv.org/abs/2402.04396), but only after a fused inverse
   transform and full sequence gates exist.
6. Request-specific precision: keep a compact base pack plus promoted overlay
   extents selected by domain/profile, using the existing pack-overlay layer.

### Same-size LUT2 experiments

The following variants keep the old groupwise-g128 payload at exactly 69.72
MiB on the 0.1B checkpoint:

- `groupwise_kmeans_fp16` stores codebooks as FP16. This permits g64 groups at
  the same byte count as FP32-codebook g128. Full-pack weighted RMSE improved
  from `0.01990` to `0.01782` (about 10.4%).
- `groupwise_kmeans_residual` uses the FP16-codebook savings for two sparse
  FP16 error repairs per group. Full-pack weighted RMSE improved to `0.01678`
  (about 15.7%), and the 12-tensor probe cut max absolute error from `7.31`
  to `2.11` without adding bytes.

Neither pure-LUT2 variant passed the native recurrent gate. The repaired g128
pack still produced minimum top-10 overlap `0.0`, maximum KL `5.66`, and state
relative-L2 drift `1.235`. Therefore the implementation is a real RMSE and
max-error improvement, but not yet a deployable all-block 2-bit quantizer.

Rejected same-size experiments are retained as explicit research options:

- symmetric codebooks reduced signed bias but worsened tensor RMSE and did not
  improve recurrent drift;
- weight-magnitude saliency was worse than selecting the largest residuals;
- grouping along the input axis was worse than the existing flat grouping on
  this RWKV checkpoint;
- replacing even one complete recurrent layer with LUT2 caused the native gate
  to fail. Replacing only `blocks.6.ffn.value.weight` was closer, but still
  missed (`KL 0.144`, state drift `0.106`).

This agrees with the main findings of [GPTQ](https://arxiv.org/abs/2210.17323),
[AWQ](https://arxiv.org/abs/2306.00978), [SpQR](https://arxiv.org/abs/2306.03078),
[SqueezeLLM](https://arxiv.org/abs/2306.07629), and
[AQLM](https://arxiv.org/abs/2401.06118): successful 2-bit post-training
quantization needs activation/Hessian-aware allocation, sparse outliers, or
multiple additive codebooks. Scalar K-means alone is not enough for RWKV's
recurrently amplified errors. Rotations from
[QuIP#](https://arxiv.org/abs/2402.04396) or
[QTIP](https://arxiv.org/abs/2406.11235) remain promising, but require a fused
inverse transform before they preserve the SSD-streaming throughput objective.

### Activation calibration and mixed-codec result

`rwkv_ssd.tools.calibrate_activations` observes real resident ChatRWKV
`aten.mm`/`aten.mv` inputs without changing model outputs and writes per-input-
feature RMS statistics. `groupwise_residual_activation` keeps the ordinary
K-means levels but spends same-size sparse repairs on large `abs(error) * RMS`
terms. This improved the repaired-LUT2 native result from maximum KL `5.66` and
state drift `1.235` to `4.73` and `1.124`; it still failed, so it remains an
allocation improvement rather than a deployable pure-2-bit result.

`rwkv_ssd.tools.plan_mixed_precision` compares calibrated repaired LUT2 g128
against grouped U8 g64, ranks promotions by weighted error reduction per extra
byte, supports explicit mandatory recurrent tensor-family suffixes, and now
includes small time-mix/control tensors. The latter is essential: promoting all
168 substantial block matrices while leaving 230 tiny block controls at LUT2
failed (`0.20` top-10 overlap, KL `4.93`, state drift `1.117`), despite those
controls occupying only about 68 KiB in their LUT2 form.

Promoting all 398 block tensors to grouped U8 while retaining LUT2 for the four
non-block tensors produced a 134.22 MiB pack and passed the native two-prompt
gate with the same block-upload metrics as all-tensor grouped U8 g64: top-10
overlap `0.80`, maximum KL `0.0201`, and state drift `0.0557`. This is a useful
rwkv.cpp provider pack-size optimization, but not a fully quantized-graph
result: embedding, head, and other non-block tensors remain resident FP16 in
the current native benchmark. Standalone runtimes that consume those LUT2
embedding/head tensors need a separate end-to-end quality gate.

Example calibrated plan:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.calibrate_activations `
  --checkpoint .\model.pth --pack .\resident-fp16-pack `
  --output .\activation-rms.pt

.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.plan_mixed_precision `
  --checkpoint .\model.pth --activation-stats .\activation-rms.pt `
  --output .\mixed-plan.json --extra-budget-mb 70 `
  --mandatory-suffix .ffn.key.weight --mandatory-suffix .ffn.value.weight `
  --mandatory-suffix .att.key.weight --mandatory-suffix .att.value.weight `
  --mandatory-suffix .att.receptance.weight --mandatory-suffix .att.output.weight

.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.pack_runtime `
  --input .\model.pth --output .\pack-mixed `
  --pack-codec trinity_lut2 `
  --trinity-codebook groupwise_residual_activation `
  --trinity-group-size 128 --scale-group-size 64 `
  --activation-stats .\activation-rms.pt --codec-map-json .\mixed-plan.json
```

### Comparison with conventional quantization

The local rwkv.cpp Q4_K checkpoint was 248.28 MiB and failed the same two-prompt
gate (minimum top-10 overlap `0.20`, maximum KL `1.90`, state drift `0.240`).
The repository's groupwise-U8 g64 pack was smaller at 206.22 MiB and passed
(`0.80`, `0.0201`, `0.0557`). On this checkpoint and smoke gate, groupwise U8
is better than this conventional Q4_K baseline. Pure LUT2 is much smaller
(69.72 MiB) but substantially less accurate.

The rwkv.cpp provider quality benchmark currently uploads block layers only.
Embedding, head, and non-block tensors remain resident from the FP16 ggml
checkpoint. JSON reports now state this scope explicitly; results must not be
described as a fully quantized native graph.

## Implemented and CPU-tested

### Cache residency and F tiers

The runtime reports three independent residency pools:

| Pool | Contents | Main control |
| --- | --- | --- |
| Dense `z` | decoded backend tensors | `warm_z`, `max_layers_in_z` |
| Prepared | decoded/provider tensors | `prepared_cache_bytes` |
| Packed | compressed LUT2 blobs and native indices | `packed_cache_bytes` |

`cache_format=none|packed|prepared|dense` selects an explicit point. With
`residency_policy=auto`, an explicit format still wins; otherwise the stable
load-time map is F1 = packed, F2/F3/F4 = prepared, and F5 = dense. A synthetic
Pareto test also checks pack composition and byte-budget decisions. The legacy
`residency_policy=static` behavior remains unchanged. The selected format and
all cache byte pools are available in metrics/JSON immediately after load and
after generation.

Packed and prepared byte caps are tested across multiple layers, prepared
reuse, LUT/TMix aliases, and eviction. `0` keeps the legacy unlimited-cache
meaning; use `cache_format=none` for a true no-cache profile.

### Manifest-v2 striped packs

Manifest loading now rejects striped entries unless all of these hold:

- logical extents are non-overlapping and cover exactly `entry.length`;
- every stripe shard is declared in `weights_files`;
- physical offsets and padded physical lengths satisfy the entry alignment;
- every physical extent fits the actual shard file.

`ShardedWeightStore.read_layer_coalesced()` groups extents by shard, merges
adjacent physical ranges, issues leaf reads concurrently, and reconstructs
logical tensors. The provider uses this path for normal and materialized
striped-layer loads. Legacy layer-affinity packs retain their one-shard span
path. A nested-executor starvation case in striped `read_bytes_many()` is also
covered by regression tests.

BF16 shadow sidecars are striped with the same command. `fast_offset` remains
the stable logical shadow address, while `fast_stripes` records aligned
physical extents in `shadow.shard.N.bin`. Manifest loading validates exact
coverage, declared files, alignment, and physical bounds; the runtime gathers
and coalesces those extents through the CPU sharded store before shadow decode.

### Quantization quality gates

`rwkv_ssd.tools.quant_quality` provides:

- robust equal-empty tensor, empty-logit-batch, and empty-state-sequence handling;
- per-tensor RMSE, mean/max absolute error, relative L2, and cosine;
- optional per-layer weighted/max RMSE aggregates;
- logit top-k overlap and candidate-to-reference KL;
- recurrent-state final/mean/maximum relative-L2 drift;
- a machine-readable `summary` containing pack completeness, max/weighted
  RMSE, minimum top-k overlap, maximum KL, maximum state drift, gate booleans,
  and the final pass/fail result.

Exact pack gate example:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.quant_quality `
  --reference .\packs\fp16 `
  --candidate .\packs\lut2 `
  --per-layer `
  --max-rmse 0.050 `
  --max-weighted-rmse 0.020 `
  --strict --json
```

Exact logit/state gate example:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.quant_quality `
  --reference-logits .\quality\fp16-logits.pt `
  --candidate-logits .\quality\lut2-logits.pt `
  --reference-state .\quality\fp16-state.pt `
  --candidate-state .\quality\lut2-state.pt `
  --top-k 10 `
  --min-top-k-overlap 0.90 `
  --max-kl 0.020 `
  --max-state-relative-l2 0.050 `
  --strict --json
```

`--strict` exits with status 1 when any supplied gate fails.

### Controlled benchmark and storage diagnostics

`bench/bench_streaming_matrix.py` supports warmup tokens, repeated samples,
median/p95 tok/s, read/staging/compute milliseconds per token, cache hits and
evictions, JSON/CSV output, and a `correctness_only` profile.

Model interpretation is deliberately narrow:

- **0.1B:** correctness, warm-cache behavior, and compute-bound smoke tests;
- **2.9B:** RAM/cache accounting and longer CPU validation;
- **7B+ on Linux/NVIDIA:** real storage, H2D, CUDA, and physical scaling tests.

An artificial `RWKV_SSD_IO_CAP_MBPS` or `--io-caps-mbps` value is a controlled
throttle only. Every result marks `physical_ssd_scaling_claim=false`.

`rwkv_ssd.tools.storage_diagnostic` works with single-file, layer-affinity,
and manifest-v2 striped packs. It reports pack file sizes, sequential tensor
throughput, per-layer latency, available shard concurrency, and repeated-read
OS/page-cache behavior. JSON output can be copied to the 3060 laptop for later
comparison.

### Activation sparsity telemetry

`RWKV_CMIX_SPARSITY=1` remains opt-in and reports sample count, zero fraction,
and active fraction. Optional tile statistics use:

```powershell
$env:RWKV_CMIX_SPARSITY="1"
$env:RWKV_CMIX_TILE_STATS="1"
$env:RWKV_CMIX_TILE_SIZE="32"
```

Tile telemetry reports tile count, active-tile fraction, and a compact
occupancy histogram. `rwkv_ssd.runtime.cmix_sparse.pack_value_matrix` can
explicitly repack a dense `[in,out]` value matrix into row tiles. Register its
`TiledValueMatrix` with the provider and set `RWKV_CMIX_SELECTIVE_READS=1` to
read only tiles containing active input rows. Results match dense matmul;
metrics report tiles read/skipped and bytes read. Existing dense/LUT packs do
not opt in automatically.

### Adaptive residency

`adaptive_residency=true` enables a windowed controller for
`none|packed|prepared|dense`. It requires a sustained measured cost improvement,
applies hysteresis and a minimum token dwell, and stops at a configured change
limit. Decisions are applied only between requests after the old provider is
drained and closed. An explicit `cache_format` disables adaptation.

### Innovation prototypes

The CPU path also includes bounded prototypes for the next research tranche:

- `InferenceEngine.generate_batch()` supports synthetic batching, dense ChatRWKV
  layer-outer/session-inner prefill/decode, and the native rwkv.cpp equivalent.
  One streamed layer advances every active recurrent state before eviction. The
  ChatRWKV 0.1B CPU gate has exact greedy output parity and, for two sessions
  with eight decode tokens each, measured 1.12 to 1.89 aggregate end-to-end
  tok/s and 1.37 to 2.75 aggregate decode-only tok/s. Per-session token
  latency remained about 0.73 seconds. The native rwkv.cpp 0.1B warm probe
  measured 2.67–3.27 aggregate tok/s for two sessions generating four tokens
  each, versus 1.34–1.60 tok/s independently, with exact output parity. These
  are capacity results, not single-session latency or physical-SSD claims.
- CMix tiled matrices support exact temporal tile prefetch with synchronous
  miss fallback, a byte-bounded hot-tile cache, co-activation physical ordering,
  and adjacent-tile read coalescing.
- Optional observed page-residency tracking can suppress redundant prefetch
  hints (`RWKV_PAGE_RESIDENCY=1`). It is a portable recent-read estimate, not a
  native Windows/Linux page-table query.
- Content-addressed pack overlays, exact compressed recurrent-state deltas,
  prepared native artifact caching, activation-weighted residual experiments,
  contextual multi-knob tuning, and deadline-aware shadow/packed routing have
  isolated CPU tests. Residual and deadline fallbacks remain quality-gated.
- A byte-bounded hierarchical recurrent-state store now parks hot sessions in
  a RAM LRU and spills exact portable snapshots to local SSD. GPU-memory and
  remote tiers remain open; the CPU filesystem benchmark reports its scope.
- Content-addressed profiles can now serve tensor spans directly, remove
  profiles transactionally, and garbage-collect only unreferenced chunks.
- Runtime bottleneck diagnosis, streamed-byte admission, JSONL decision traces,
  persistent machine-bound tuning profiles, runtime capability probes, and
  load-enforced quality certificates have isolated CPU and serving tests.
- The ggml bridge has an opt-in fixed slot arena. It activates only when both
  slot variables are set and the loaded native library exposes the slot ABI;
  otherwise the legacy copying upload remains unchanged.

## CPU-testable, not hardware-validated

- Concurrent shard scheduling and coalesced stripe gathers are validated with
  normal files and synthetic concurrency, not independent physical SSDs.
- CUDA staging ownership is implemented and gated, but real stream overlap and
  throughput have not been measured on this Windows CPU development machine.
- Artificial bandwidth caps validate benchmark behavior, not NVMe controller,
  filesystem, thermal, RAID, or PCIe scaling.

## Linux/NVIDIA-only validation

- GPUDirect Storage and NVMe-to-VRAM paths;
- CUDA packed kernels and real copy/compute stream overlap;
- Linux `io_uring` and `O_DIRECT` experiments;
- physical multi-SSD scaling on 7B+ packs.

No throughput claim for these items should be inferred from the CPU tests or
artificial caps.

## Deferred research

- NF4, AQLM, QuIP#, QTIP, or residual sidecars without measured quality data;
- tensor/expert parallel execution, which is separate from storage sharding.

## Exact commands

Build a legacy layer-affinity shard pack:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.shard_pack `
  --input .\runtime_pack `
  --output .\runtime_pack_layer_sharded `
  --shards 4 `
  --strategy layer
```

Build a manifest-v2 striped pack (BF16 shadow sidecars are striped too):

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.shard_pack `
  --input .\runtime_pack `
  --output .\runtime_pack_striped `
  --shards 4 `
  --strategy stripe `
  --stripe-bytes 67108864
```

Run the repeated benchmark matrix and write JSON plus CSV:

```powershell
.venv-cpu\Scripts\python.exe bench\bench_streaming_matrix.py `
  --model .\runtime_pack `
  --backend synthetic `
  --cache-formats none,packed,prepared,dense `
  --io-caps-mbps 0,250,500 `
  --warmup-tokens 2 `
  --samples 5 `
  --json-out .\bench\results\streaming-matrix.json `
  --csv-out .\bench\results\streaming-matrix.csv
```

Run the small-model correctness profile:

```powershell
.venv-cpu\Scripts\python.exe bench\bench_streaming_matrix.py `
  --model .\runtime_pack `
  --backend synthetic `
  --profile correctness_only
```

Run the storage diagnostic:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.storage_diagnostic `
  --pack .\runtime_pack_striped `
  --workers 4 `
  --repeats 3 `
  --json-out .\bench\results\storage-diagnostic.json
```

Run the weight-stationary synthetic benchmark:

```powershell
.venv-cpu\Scripts\python.exe bench\bench_weight_stationary.py `
  --model .\runtime_pack `
  --prompts "a,longer prompt,third" `
  --max-tokens 8 --samples 3 `
  --json-out .\bench\results\weight-stationary.json
```

Run the real ChatRWKV independent-vs-shared-layer benchmark:

```powershell
$env:RWKV_PROMOTE_FULL_Z='0'
.venv-cpu\Scripts\python.exe bench\bench_chatrwkv_weight_stationary.py `
  --pack C:\prepared\reference-pack `
  --checkpoint C:\models\rwkv-model.pth `
  --max-tokens 8 --samples 3 `
  --json-out .\bench\results\chatrwkv-weight-stationary.json
```

This benchmark's headline tok/s divides generated decode tokens by total wall
time, including independent prompt prefill. That is an end-to-end serving
metric, not decode-only throughput. Short runs can therefore erase the batching
gain. The current batch metrics also under-report `prefill_wall_s` and
`z_bytes`, and their layer rows include prefill; do not compare those fields
directly with independent decode rows until the instrumentation is aligned.

Run the CPU fast suite with the repository's restricted temp path:

```powershell
.venv-cpu\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp .pytest-tmp\ssd-streaming-fast
```

For transfer to a CUDA laptop, use the accelerator and deployment guidance in
[`ACCELERATORS.md`](ACCELERATORS.md) and [`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md).
Hardware numbers must come from the target SSD, filesystem, GPU, and model.
