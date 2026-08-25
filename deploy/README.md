# Deploy examples

## `example_config.yaml`

Copy and edit for daily use:

```bash
python -m app.cli --config deploy/example_config.yaml --prompt "Hello"
```

Key fields:

| Field | Purpose |
|-------|---------|
| `mode` | `resident` / `partial` / `streaming` |
| `stream_layer_cache` | Retain decoded block layers in provider / `z` (throughput) |
| `max_layers_in_z` | LRU cap for block layers in `z` (explicit value honored; auto=2 for n≥8) |
| `system_prefix` | Fixed system prompt for state cache (M2.5) |
| `state_cache` | Enable prefix state reuse (synthetic + ChatRWKV streaming) |
| `low_ram` | Bounded z + decouple + disk cache preset |
| `ram_budget_gb` | Auto-select F1–F5 tier then planner (tier-first) |
| `decode_disk_cache` | `auto` / `1` / `0` — `.decode_cache/` bf16 per layer |
| `decouple_provider_cache` | Default `true` — provider LRU independent of `z` eviction |
| `max_provider_cache_layers` | Cap decoded tensors in RAM (0 = resolver / full model) |
| `max_provider_cache_bytes` | Byte cap for provider LRU (RAM budget sets this) |
| `metrics_csv` | Per-layer timing output |

## Throughput presets (RAM vs speed)

Full table: [`docs/PRESETS.md`](../docs/PRESETS.md).

| Goal | Command |
|------|---------|
| **Max tok/s** (~382 MB `z` on 0.1B) | `$env:RWKV_PROMOTE_FULL_Z="1"` + a certified FP16/BF16 streaming pack |
| **Default auto (n≥8)** | 2-layer `z` + full decoupled provider cache |
| **Minimal RAM (F1)** | `$env:RWKV_SSD_TIER="1"` (~201 MB `z` on 0.1B; ~671 MB on 2.9B) |
| **Pin early layers (F3)** | `$env:RWKV_PARTIAL_SSD_TIER="1"` — hot3 strict |
| **Pin + stream cache** | `$env:RWKV_PARTIAL_FUSED="1"` — usually **slower** than partial strict |
| **Bounded fused (F2)** | `$env:RWKV_BOUNDED_STREAM="1"` (~203 MB `z`, 2-layer window) |

```powershell
$pack = "C:\prepared\rwkv-model.pack"
$checkpoint = "C:\models\rwkv-model.pth"

# Minimal RAM (strict + fused)
$env:RWKV_SSD_TIER="1"
python -m app.cli --model $pack --checkpoint $checkpoint `
  --backend chatrwkv --mode streaming --max-tokens 32

# Hot3 strict (recommended partial compromise)
$env:RWKV_PARTIAL_SSD_TIER="1"
python -m app.cli --model $pack --checkpoint $checkpoint `
  --backend chatrwkv --mode streaming --max-tokens 32

# Compact-path F1 smoke (use a pack with a verified quality certificate)
$env:RWKV_CPU_THREADS="8"
$env:RWKV_SSD_TIER="1"
python bench/bench_f1_f3.py --pack $pack `
  --checkpoint $checkpoint --tiers F6,F1 --max-tokens 8

# Compare all presets (0.1B; use a certified pack for quality claims)
python bench/bench_io_ceiling.py --heavy --light --samples 1 --warmup 0

# ChatRWKV vs rwkvcpp (resident rwkv.cpp comparison; long-run provider quality remains open)
python bench/bench_backend_compare.py --max-tokens 16 --samples 3 --threads 8
```

## rwkv.cpp (experimental CPU backend)

Full doc: [`docs/BACKENDS.md`](../docs/BACKENDS.md).

```powershell
# Build once
cmake -S backends/rwkvcpp_ref -B backends/rwkvcpp_ref/build
cmake --build backends/rwkvcpp_ref/build --config Release

# Convert the operator's RWKV checkpoint (or set RWKVCPP_GGML_PATH to an
# existing matching GGML file)
$ggml = "C:\models\rwkv-model-F16.bin"
python backends/rwkvcpp_ref/python/convert_pytorch_to_ggml.py `
  $checkpoint $ggml FP16

# Resident generate (matching GGML model is required)
python -m app.cli --model $pack --backend rwkvcpp --mode resident `
  --checkpoint $ggml --max-tokens 16
```

For provider-backed rwkv.cpp streaming, provision the native root/DLL and a
matching GGML model, then run `rwkv-ssd-preflight --backend rwkvcpp`. Trinity
packs are rejected unless their quality certificate passes. The current CPU
release does not make a CUDA or accelerator performance claim.

## Partial residency profiles

| File | Policy |
|------|--------|
| `rwkv7_0.1b_partial_hot3.json` | Pin layers 0–2 (+ auto last) — 0.1B / mid-size |
| `rwkv7_0.1b_partial_hot4.json` | Pin layers 0–2, 11 |
| `rwkv7_0.1b_partial_hot7.json` | Pin layers 0–5, 11 — **12-layer packs only** |
| `rwkv7_0.1b_partial.json` | Pin layers 0, 11 + embed/head |
| `rwkv7_0.1b_partial_all.json` | All 12 blocks resident (~382 MB z) |
| `rwkv7_2.9b_partial_hot3.json` | Pin layers 0–2 (+ auto last=31) — used when `n_layer≥24` |

Auto-applied when `--mode partial` and no profile passed: hot7 only for exact 12-layer packs; hot3 (2.9B file when present) for `n_layer≥24`; hot3 otherwise for `8≤n<24`.

```bash
python -m app.cli --model <pack-dir> --mode partial \
  --residency-profile deploy/rwkv7_0.1b_partial_hot7.json ...
python rwkv_ssd/tools/suggest_residency.py layers.csv --top 4 --json-out deploy/my_partial.json
```

## Streaming throughput (default high-RAM path)

```bash
python -m app.cli --model <pack-dir> \
  --checkpoint <checkpoint.pth> \
  --backend chatrwkv --mode streaming \
  --stream-layer-cache --strategy "cpu bf16"
```

After warm tokens, `promote_stream_cache_to_full_z` lifts retention so streaming ≈ resident native `forward` when `RWKV_PROMOTE_FULL_Z=1`.

**Repeated system prompt (chat / serve):**

```bash
python -m app.cli ... --state-cache \
  --system-prefix "You are a helpful assistant." "User question here"
```

**Thesis low-RAM (~10 GB, 200B-class planning):**

```bash
python -m app.cli --model <pack> --low-ram --ram-budget-gb 10 \
  --backend chatrwkv --mode partial --strategy "cpu bf16"
```

**Trinity pack with shadow (experimental only):**

Shadow is enabled automatically when `shadow.bin` is present (`RWKV_DECODE_SHADOW=1`).
It does not bypass the model-level quality certificate; disable with
`RWKV_DECODE_SHADOW=0` when diagnosing a certified pack.

```bash
python -m rwkv_ssd.tools.add_bf16_shadow --pack <pack-dir>
```

## Environment overrides

```bash
set RWKV_SSD_PACK=C:\path\to\runtime_pack
set RWKV_SSD_MODE=streaming
set RWKV_SSD_TIER=1
set RWKV_PARTIAL_SSD_TIER=1
set RWKV_PARTIAL_FUSED=1
set RWKV_BOUNDED_STREAM=1
set RWKV_WARM_DISK_CACHE=auto
set RWKV_PROMOTE_FULL_Z=1
set RWKV_PACK_PROFILE=tiered_hot3
set RWKV_LUT_GEMM_FUSED=1
set RWKV_PIN_ACCURACY_LAYERS=0
set RWKV_CPU_THREADS=8
set RWKV_SSD_SYSTEM_PREFIX=You are a helpful assistant.
set RWKV_RAM_BUDGET_GB=10
set RWKV_DECODE_SHADOW=auto
set RWKV_DECODE_DISK_CACHE=1
set RWKV_PACK_PROFILE=shadow_sel
```

`RWKV_SSD_SYSTEM_PREFIX` also enables `state_cache`.
