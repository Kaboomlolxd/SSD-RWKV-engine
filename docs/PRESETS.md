# Throughput presets and environment variables

> **Release qualification note (July 2026):** the measured Trinity LUT2
> frontier below is a historical engineering reference, not a production
> quality claim. The historical LUT2/shadow payloads were removed and are
> quarantined in the archive as benchmark evidence;
> after failing real-model generation gates. Automatic runtime resolution uses
> the mixed `trinity_grouped_0.1b` g256 compact pack only when it has a
> passing certificate (326 large tensors grouped-U8; 76 one-dimensional
> vectors dense BF16); otherwise it uses the `trinity_safe_0.1b` parity-
> validated fallback. Explicit grouped selection remains diagnostic and
> fails preflight without its certificate. The grouped pack passes the
> current short real-model resident-vs-streaming parity checks; longer
> quality certificates are still required.
> Production benchmarks must still identify the resolved pack and evidence.

Single reference for **RAM vs tok/s** paths. Implementation: `rwkv_ssd/runtime/throughput_defaults.py`. Deploy examples: [`../deploy/README.md`](../deploy/README.md).

**Rule:** pick **one** scheduler path per deployment. Do not stack independent “2×” claims — see [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md#core-rule-do-not-stack-speedups).

## Decision tree

```text
Set `RWKV_RAM_BUDGET_GB=N`?
 yes → auto-profile: pick F1/F2/F3/F5 by floor (≥0.5 GB → F5, ≥0.21 → F2, ≥0.15 → F1)
 no → Enough RAM for full promote (~382 MB z on 0.1B; ~full bf16 on 2.9B)?
 yes → F5 promote-max (`RWKV_PROMOTE_FULL_Z=1`) — max tok/s
 no → Need best speed under ~280 MB (0.1B)?
 yes → F4 partial-hot4 or F3 partial-hot3 (`RWKV_PARTIAL_SSD_TIER=1`)
 no → Need ~203 MB bounded cache?
 yes → F2 bounded-fused (`RWKV_BOUNDED_STREAM=1`)
 no → F1 ssd-tier-min (`RWKV_SSD_TIER=1`) — ~201 MB skeleton
```

> **Caveat for 0.1B / small models:** the F1–F3 `ram_budget_gb` floors are calibrated to 0.1B `z` MB. On 2.9B the same env picks the same *tier logic*, but absolute RAM is much larger (F1 ~671 MB skeleton + ~1.3 GB provider LUT blobs). Prefer explicit `RWKV_SSD_TIER` / `RWKV_PARTIAL_SSD_TIER` / `RWKV_PROMOTE_FULL_Z` on large packs.

Bench the slope: `python bench/bench_io_ceiling.py --heavy` → see [`BENCH.md`](BENCH.md). For a promoted compact model, use the operator's prepared pack; its auto profile resolves a certified sibling only when the manifest-bound certificate is present. Historical all-LUT2 A/B results remain in the archive and are not shipped.

**Auto-profile tier map** (`select_ram_budget_tier`):

| Budget (GB) | Tier | Approx `z` MB (0.1B) |
|-------------|------|----------------------|
| ≥ 0.39 | F5 | ~382 |
| ≥ 0.27 | F3 | ~262 |
| ≥ 0.21 | F2 | ~203 |
| < 0.21 | F1 | ~201 |

## Frontier table (0.1B Trinity LUT2, historical / measured warm)

| ID | Preset | `z` MB | tok/s (warm) | Env / API |
|----|--------|--------|--------------|-----------|
| F1 | ssd-tier-min | ~201 | **~6.4–7.2** | `RWKV_SSD_TIER=1` |
| F2 | bounded-fused | ~203 | **~6.5–8.3** | `RWKV_BOUNDED_STREAM=1` |
| F3 | partial-hot3 | ~262 | ~5.7–6.1 | `RWKV_PARTIAL_SSD_TIER=1` |
| F3t | tiered-hot3-partial | ~262 | ~6.1 | `RWKV_PACK_PROFILE=tiered_hot3` + partial |
| F4 | partial-hot4 | ~262 | ~5.9 | `apply_partial_hot4_ssd_defaults` |
| F5 | promote-max | ~382 | **~9–14** | `RWKV_PROMOTE_FULL_Z=1` |
| F5s | promote-shadow | ~382 | ~14 | `RWKV_PACK_PROFILE=shadow_sel` + promote |
| **Fb** | **auto-tier (0.5 GB)** | **~382** | **~6–18**† | `RWKV_RAM_BUDGET_GB=0.5` |
| F6 | resident-all-ram | ~full ckpt | ~11.8 | `--mode resident` |

† Fb numbers vary with hardware and whether the engine took the tier-first path (`apply_ram_budget_tier` then planner). Post-fix low-ram + budget mirrors `InferenceEngine.load`.

### 2.9B tier reference (default grouped-quality path)

The historical all-LUT2 numbers are retained for comparison only. The active
compact profile is an operator-prepared `weights.bin` with native-safe
non-matrix controls and grouped-U8 matrices. Its size and short-smoke metrics
belong to the exact checkpoint/pack certificate; long-run held-out generation
qualification remains open, and the path is not a 15 tok/s CPU solution on
the measured machine.

Current native rwkv.cpp acceptance (`bench/bench_f1_f3.py`, 2.9B grouped-quality pack, 8 generated tokens × 2 samples, `RWKV_CPU_THREADS=1`):

| ID | Cold tok/s | Warm tok/s | Provider-owned MB | Mmap MB | Gate |
|----|-----------:|-----------:|------------------:|--------:|------|
| F1 ssd-tier-min | 2.02 | 1.90 | 2.9 | 2,937 | pass |
| F2 bounded-fused | 2.02 | 1.99 | 0.4 | 367 | pass |
| F3 partial-hot3 | 2.24 | 1.90 | 370.1 | 2,570 | pass |
| F4 partial-hot4 | 2.25 | 2.12 | 461.9 | 2,478 | pass |
| F5 promote-max | 2.03 | 1.85 | 188.7 | 0 | reference |

The F1-F4 ratios pass the 0.60/0.80/0.80/0.80 cold-and-warm gates and the
explicit 12 GB RSS, 1 GiB provider, and 512 MiB native-layer budgets. The
grouped-quality pack has a short-smoke certificate for the measured
three-prompt logits/state scope, but still lacks long-run held-out generation
qualification, so this is a throughput/ABI acceptance result rather than a
general model-quality claim.
`provider_cache_bytes` counts process-owned provider memory; stable packed
views are reported separately as `provider_mmap_bytes`. For native rwkv.cpp,
leave `RWKV_CPU_THREADS=auto` or use the model-width policy in
[`BACKENDS.md`](BACKENDS.md); set an explicit integer only for a measured host.

### Native rwkv.cpp grouped-U8 result

The historical g64 all-grouped matrix pack was loaded through the native CPU
graph with about 0.24 s of packed-only initialization; its 480 available SG8
matrix records uploaded in about 5 s on the measured Windows AVX2 host. Warm
one-token decode measured approximately 1.83 / 2.45 / **2.62** / 2.19 tok/s
at 1 / 2 / 4 / 8 native threads. Four threads was best; eight threads slowed
down. These are historical g64 post-upload native decode figures and must not
be treated as a g32 throughput measurement or compared directly with cold
F1/F3 end-to-end rates.

The native route is enabled automatically for non-resident grouped-U8 packs.
Set `RWKVCPP_NATIVE_U8=0` to force the ordinary dense GGML bridge, or set
`RWKVCPP_NATIVE_U8_PACKED_ONLY=1` to require shape-only matrix residency for a
pack whose every 2-D manifest tensor is grouped-U8. The native route is
CPU-only and remains subject to the pack's quality certificate.

### Backend note (rwkvcpp)

F1–F5 presets apply to both **`--backend chatrwkv`** and the experimental
provider-backed **`--backend rwkvcpp`** path. rwkvcpp uses the shared
ManifestWeightProvider and uploads decoded pack layers through
`GgmlWeightBridge` into the native GGML graph. Resident native-GGML results
must be compared separately: the best local 2.9B results were ~2.8 tok/s FP16,
~3.5 tok/s Q5_1, and ~4.2–4.6 tok/s Q4_K.

For a non-resident pack with grouped-U8 matrices, the bridge now selects
rwkv.cpp's native SG8 graph automatically. A complete all-grouped matrix pack
uses packed-only residency: the converted `.bin` contributes tensor shapes and
small dense controls, while the provider uploads grouped records for block
matrices, embedding, and head. `RWKVCPP_NATIVE_U8=0` forces the ordinary dense
bridge for A/B tests.

The first bridge keeps the converted `.bin` graph resident; provider/cache,
prefetch, shadow, disk-cache, prefix-state, and F1–F5 semantics are shared.
Selective ggml weight slots remain a separate RAM-reduction follow-up.

Compare backends now:

```powershell
python bench/bench_backend_compare.py --max-tokens 16 --samples 3 --threads 8 `
  --json-out bench/results/chatrwkv_vs_rwkvcpp_0.1b.json
```

## Historical notes (0.1B prefetch bugs — fixed)

* **F-1:** no-cache path issued `prefetch_ahead` unconditionally → skip when `stream_layer_cache=False`.
* **F-2:** cache path blocked on in-flight prefetch → best-effort drain when `fut.done()`.
* **F-3:** provider cache hardcoded to 2 layers → resolver picks full cache for n≥8 decoupled.
* **F2 z-cap:** accuracy pins of layers 0 + n−1 stacked on the 2-layer LRU → 4 layers in `z`. Auto pins now skip when `max_layers_in_z≤2`; F2 presets set `RWKV_PIN_ACCURACY_LAYERS=0`.

Provider cache on F2 is **full** (decoupled from the 2-layer `z` window), not a 2-layer LRU. See [`BASELINE_BUGS.md`](BASELINE_BUGS.md).

Deprecated (not frontier): `RWKV_PARTIAL_FUSED=1`, unfused stream+cache — see `archive/bench/io_ceiling_legacy_scenarios.py`.

## Preset table (detail)

| Goal | Env / API | `z` MB (0.1B) | Stream cache |
|------|-----------|---------------|--------------|
| Max tok/s | `RWKV_PROMOTE_FULL_Z=1` | ~382 | yes + full promote |
| Default auto (n≥8) | (none) | ~2-layer `z` + full provider cache | yes |
| Bounded fused (F2) | `RWKV_BOUNDED_STREAM=1` | ~203 | yes, 2-layer `z`, full provider |
| Partial hot4 strict | `apply_partial_hot4_ssd_defaults` | ~262 | **no** |
| Partial hot3 strict | `RWKV_PARTIAL_SSD_TIER=1` | ~262 | **no** |
| SSD tier strict (F1) | `RWKV_SSD_TIER=1` | ~201 | **no** |
| Stacked max | `RWKV_PACK_PROFILE=shadow_sel` + promote | ~382 | yes + shadow pack |

Bench: `python bench/bench_io_ceiling.py --max-tokens 24 --samples 2`

> **Important:** use `--max-tokens 24+` and `--samples 2+` for tok/s claims. The bench warms internally; a single short sample is noisy.

## Environment variables

| Variable | Values | Effect |
|----------|--------|--------|
| `RWKV_PROMOTE_FULL_Z` | `0` / `1` / `auto` | Full `z` after warm; `auto` = off for n≥8 |
| `RWKV_SSD_TIER` | `1` / `stream` | Strict fused (~201 MB) or SSD stream tier |
| `RWKV_PARTIAL_SSD_TIER` | `1` | Partial hot3 + strict fused |
| `RWKV_PARTIAL_FUSED` | `1` | Partial hot3 + bounded cache (usually slower) |
| `RWKV_BOUNDED_STREAM` | `1` | `apply_bounded_fused_defaults` (F2) |
| `RWKV_LUT_GEMM_FUSED` | `0` / `1` / `auto` | Fused att/FFN/head GEMV |
| `RWKV_LUT_SMALL_TRANSPOSED` | `auto` / `0` / `1` | Grouped-U8 TMix `x @ W` GEMV; `auto` enables it when the native AVX2 ABI is available, `0` forces dense BF16 adapters, `1` forces the packed path |
| `RWKV_LUT_FUSED_ADAPTERS` | `auto` / `0` / `1` | Native C two-sweep grouped-U8 TMix adapter pipeline; `0` keeps the Python/Torch adapter fallback |
| `RWKV_LUT_ACTIVATION_FP32` | `auto` / `0` / `1` | Fused CPU activation dtype; `auto` keeps packed blocks in FP32 when Torch reports no AVX512-BF16, avoiding repeated BF16/FP32 conversions; mixed dense/packed tiers stay on their native dtype |
| `RWKV_LUT_ACTIVATION_INT8` | `0` / `1` | Experimental SG8 activation quantization; can materially change greedy tokens, so leave at `0` for quality-certified runs |
| `RWKV_LUT_ACTIVATION_INT8_HEAD` | `0` / `1` | Experimental INT8 activation quantization for the vocabulary head; independently gated because head perturbations can change argmax |
| `RWKVCPP_NATIVE_U8` | `auto` / `0` / `1` | Enable the CPU-native rwkv.cpp SG8 grouped-U8 graph for non-resident grouped matrix packs; `0` forces dense GGML uploads |
| `RWKVCPP_NATIVE_U8_PACKED_ONLY` | `auto` / `0` / `1` | Omit dense 2-D matrix payloads from the native GGML skeleton; use `1` only for an all-grouped matrix manifest |
| `RWKV_STREAM_FUSED_PREFILL_MIN_TOKENS` | positive integer | Use the fused token path for prompts shorter than this threshold (default 8) |
| `RWKV_PIN_ACCURACY_LAYERS` | `auto` / `0` / `1` | Auto-pin first+last block when `max_z≥3`; F2 defaults `0` |
| `RWKV_TRINITY_CODEBOOK` | `linspace` / `kmeans` | Pack-time codebook. `kmeans` (default) is +18 dB SNR vs `linspace` but ~3000× slower on 2560×2560 — use `--trinity-codebook linspace` for fast 2.9B+ packs |
| `RWKV_CPU_THREADS` | integer / `auto` / `off` | Backend thread override. Native rwkv.cpp `auto`: 1 below `n_embd` 1280, 2–4 for wider models; ChatRWKV Torch uses its own bounded width-aware policy |
| `RWKV_CODEC_POLICY` | `auto` / `strict` / `accuracy` / `hybrid` | Runtime codec selection |
| `RWKV_SSD_HEALTH` | `1` | Health-conscious streaming |
| `RWKV_SSD_IO_CAP_MBPS` | float | Cap read bandwidth (MB/s) |
| `RWKV_WARM_DISK_CACHE` | `auto` / `0` / `1` | Prefill `.decode_cache/` at load; F2 defaults `0` when provider cache is full |
| `RWKV_CACHE_FORMAT` | `auto` / `none` / `packed` / `prepared` / `dense` | Select which residency pool may remain warm |
| `RWKV_PACKED_CACHE_BYTES` | integer bytes | Cap retained LUT2 blobs and native packed indices |
| `RWKV_PREPARED_CACHE_BYTES` | integer bytes | Cap decoded/prepared provider tensors |
| `RWKV_RESIDENCY_POLICY` | `static` / `auto` | Select one Pareto cache format at engine load from the F tier, byte budget, and pack composition |
| `RWKV_ADAPTIVE_RESIDENCY` | `0` / `1` | Enable hysteretic cache retiering between requests; explicit cache formats disable it |
| `RWKV_ADAPTIVE_RESIDENCY_WINDOW` | positive integer | Cost sample window (default 8) |
| `RWKV_ADAPTIVE_RESIDENCY_MIN_DWELL_TOKENS` | non-negative integer | Minimum tokens between changes (default 32) |
| `RWKV_ADAPTIVE_RESIDENCY_HYSTERESIS` | `0 <= x < 1` | Required fractional improvement (default 0.15) |
| `RWKV_ADAPTIVE_RESIDENCY_MAX_CHANGES` | non-negative integer | Engine-lifetime retier cap (default 2) |
| `RWKV_SESSION_PROMOTION` | `0` / `1` | Opt in to request-lifetime-aware dense layer promotion between ChatRWKV requests; alternative to adaptive residency |
| `RWKV_SESSION_EXPECTED_TOKENS` | non-negative integer | Expected total tokens in the session; `0` uses the next request's `max_tokens` |
| `RWKV_SESSION_PROMOTION_BYTES` | integer bytes | Hard cap on additional RAM retained by session promotions |
| `RWKV_SESSION_PROMOTION_POLICY` | `benefit_per_byte` / `highest_stall` / `lru` | A/B ordering within the same cap |
| `RWKV_CMIX_SPARSITY` | `0` / `1` | Opt-in post-ReLU CMix samples plus zero/active fractions; measurement only |
| `RWKV_CMIX_TILE_STATS` | `0` / `1` | Optional tile occupancy histogram; requires `RWKV_CMIX_SPARSITY=1` |
| `RWKV_CMIX_TILE_SIZE` | positive integer | Tile width for occupancy telemetry (default 32) |
| `RWKV_CMIX_SELECTIVE_READS` | `0` / `1` | Use a registered `TiledValueMatrix`; otherwise dense/LUT behavior is unchanged |
| `RWKV_CMIX_SPECULATIVE_PREFETCH` | `0` / `1` | Prefetch the previous token's active CMix tiles; exact misses fall back synchronously |
| `RWKV_CMIX_HOT_CACHE_BYTES` | integer bytes | Byte cap for frequently active CMix tile payloads |
| `RWKV_PAGE_RESIDENCY` | `0` / `1` | Skip redundant hints using portable recent-read residency estimates |
| `RWKV_PAGE_RESIDENCY_TTL_S` | positive seconds | Residency estimate lifetime (default 30) |
| `RWKV_CODEC_DEADLINE_MS` | non-negative milliseconds | Deadline before an eligible shadow read may use its packed fallback |
| `RWKV_CODEC_MAX_FALLBACKS` | non-negative integer | Maximum deadline fallbacks in one provider lifetime |
| `RWKV_CODEC_FALLBACK_QUALITY_BUDGET` | non-negative float | Cumulative fallback quality-cost cap |
| `RWKV_PREPARED_ARTIFACT_CACHE` | directory | Disposable content-addressed native ggml payload cache |
| `RWKV_GGML_SLOT_COUNT` | positive integer | Opt-in native layer slot count; requires slot-capable rwkv.cpp ABI |
| `RWKV_GGML_SLOT_BYTES` | positive integer | Bytes per native layer slot; must be set with slot count |
| `RWKV_ADMISSION_MAX_STREAM_BYTES` | integer bytes | Reject serving requests whose predicted streamed bytes exceed one request budget; `0` disables |
| `RWKV_STATE_PARKING_DIR` | directory | Enable server-side recurrent session parking keyed by request `session_id` |
| `RWKV_STATE_PARKING_RAM_BYTES` | integer bytes | RAM LRU cap for parked states; larger/cold states remain on SSD |

See also [`SSD_HEALTH.md`](SSD_HEALTH.md).

## Residency profiles

| File | Pinned layers | When used |
|------|---------------|-----------|
| `deploy/rwkv7_0.1b_partial_hot3.json` | 0, 1, 2 (+ auto last) | 0.1B / mid-size hot3 |
| `deploy/rwkv7_0.1b_partial_hot4.json` | 0, 1, 2, 11 | 0.1B hot4 |
| `deploy/rwkv7_0.1b_partial_hot7.json` | 0–5, 11 | **only** exact 12-layer packs |
| `deploy/rwkv7_0.1b_partial_all.json` | all 12 (~382 MB) | full partial |
| `deploy/rwkv7_2.9b_partial_hot3.json` | 0, 1, 2 (+ auto last=31) | `n_layer≥24` (F3 / partial defaults) |

## Pack tools

| Tool | Purpose |
|------|---------|
| `pack_runtime` | Standard FP16 / Trinity / M5 codecs; `--codec-map` for per-tensor routing |
| `build_tiered_pack` | Hot layers FP16 resident, cold `trinity_lut2` |
| `build_decode_cache` | Pre-built `.decode_cache/` at pack time |
| `add_bf16_shadow` | Add `shadow.bin` for fast decode |
| `build_optimized_pack` | Multi-profile pack matrix |
| `shard_pack` | Build layer-affinity shards or validated manifest-v2 stripes |
| `quant_quality` | Compare pack/logit/state quality with strict gates |
| `storage_diagnostic` | CPU/Windows JSON diagnostics for single/layer/striped packs |

Tiered example:

```powershell
python -m rwkv_ssd.tools.build_tiered_pack `
  --input C:\models\rwkv-model.pth `
  --output C:\prepared\rwkv-model-tiered.pack `
  --resident-layers 0,1,2
$env:RWKV_PACK_PROFILE="tiered_hot3"
$env:RWKV_PARTIAL_SSD_TIER="1"
```

Layer-affinity and striped pack examples:

```powershell
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.shard_pack `
  --input .\runtime_pack --output .\runtime_pack_layer_sharded `
  --shards 4 --strategy layer

.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.shard_pack `
  --input .\runtime_pack --output .\runtime_pack_striped `
  --shards 4 --strategy stripe --stripe-bytes 67108864
```

Striped output also writes aligned `shadow.shard.N.bin` files when a BF16
shadow is present. Exact quality, benchmark, diagnostic, and restricted
pytest commands are in [`SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md#exact-commands).

## RAM scaling (short)

- **State** — always O(n_layer), small.
- **Block weights in RAM** — O(1) on strict F1 (~201 MB skeleton on 0.1B; ~671 MB on 2.9B), 2-layer `z` on F2, O(hot profile) on partial, O(params) with `RWKV_PROMOTE_FULL_Z=1`.
- **Provider cache** — full decoupled cache for n≥8 on F2/default (not a 2-layer LRU).

For strict accounting, use `RWKV_CACHE_FORMAT=none`; for a packed-only F1
variant use `RWKV_CACHE_FORMAT=packed` plus `RWKV_PACKED_CACHE_BYTES`. The
three pools are reported independently in metrics. See
[`SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md) for the cache/F-tier
model and benchmark decision rule.

Details: [`THROUGHPUT_PLAN.md#ram-scaling-model`](THROUGHPUT_PLAN.md#ram-scaling-model-thesis-vs-today).

## See also

- **[`BACKENDS.md`](BACKENDS.md)** — backend matrix, rwkvcpp phase 1, **streaming rwkvcpp roadmap (M5)**.
- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — current focus and Pareto winner.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog and no-stacking rules.
- **[`BASELINE_BUGS.md`](BASELINE_BUGS.md)** — F-1/F-2/F-3 fix log.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — sharded pack / multi-SSD.
- **[`SSD_STREAMING_FRONTIER.md`](SSD_STREAMING_FRONTIER.md)** — cache-format/F-tier contract and research gates.
- **[`BENCH.md`](BENCH.md)** — how to validate a preset on your hardware.
- **[`CHANGELOG.md`](../CHANGELOG.md)** — preset history.
### rwkvcpp bridge status

`--backend rwkvcpp` now uses the shared ManifestWeightProvider for F1-F5:
Trinity/LUT2 and shadow decode, disk/provider caches, non-blocking prefetch,
RAM-budget selection, and prefix state snapshots are synchronized into the
native ggml graph through `GgmlWeightBridge`. F5/provider-cache layers upload
once and reuse their native tensors; strict F1 re-uploads layers as the
provider evicts them. The first bridge version keeps the native `.bin` graph
resident, preserving ggml speed while the selective ggml-slot loader is
developed separately.
