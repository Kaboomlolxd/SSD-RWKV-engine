# Inference backends

Run `python -m rwkv_ssd.tools.check_backends` or
`python -m rwkv_ssd.tools.preflight --help` to inspect the current machine.
Importability is not a production guarantee: a backend is supported only for
the model format, device, pack codec, and quality gate listed below.

## Capability matrix

| Backend | Device | Resident | Pack streaming | Current role |
|---|---|---:|---:|---|
| `rwkvcpp` | CPU/native GGML | yes | provider bridge; native grouped-U8 packed-only path | Default real CPU backend |
| `chatrwkv` | CPU PyTorch | yes | yes, including F tiers | RWKV-7 reference and compatibility path |
| `chatrwkv` | Intel XPU + CPU compute | preview | LUT decode path | Hardware-gated development path |
| `synthetic` | CPU | yes | yes | Deterministic CI/golden backend |
| `albatross` | CUDA | yes, through the pack provider | yes, layer-wise | External Albatross checkout; availability and variant are hardware-gated |
| `rwkv-fla`, `rwkv_lightning`, `web-rwkv` | external | n/a | n/a | Integration references/proxies |

CUDA, GDS, XPU, MPS, and Albatross are hardware-gated. No local result should
be read as certification for those paths. Albatross requires `ALBATROSS_ROOT`
and an external variant exposing the layer-wise SSD adapter protocol; a
monolithic full-model variant is rejected so it cannot silently bypass the
provider tiers.

```powershell
$env:ALBATROSS_ROOT = 'C:\path\to\Albatross'
$env:ALBATROSS_VARIANT = 'faster3a_2605'  # optional; auto-detected otherwise
python -m app.cli --model C:\path\to\runtime_pack --backend albatross `
  --mode streaming --device cuda --strategy 'cuda fp16'
```

The adapter consumes dense tensors materialized by `ManifestWeightProvider`,
so existing packed formats remain valid. F1-F4 retain their current bounded
provider behavior; F5 warms all layers through that same provider and reports
the resulting cache accounting. CUDA execution and performance qualification
are intentionally not part of the CPU-only development checks.

## Default selection

The CLI defaults to `rwkvcpp` for a real CPU RWKV model. Use an explicit
backend when the contract requires another implementation:

```bash
python -m app.cli --model ./demo_pack --backend synthetic --mode streaming
python -m app.cli --model ./runtime_pack --backend chatrwkv --mode streaming
python -m app.cli --model ./runtime_pack --backend rwkvcpp --mode resident
```

`chatrwkv` is the reference when validating pack parity, DeepEmbed, or F-tier
behavior. `rwkvcpp` is the native CPU speed/format path and requires a
matching GGML model. Synthetic is the fast, dependency-light regression path.

Obsolete non-RWKV adapters are no longer shipped. Historical benchmark records
are not part of the maintained RWKV release contract.

## rwkv.cpp

The vendored source is at `backends/rwkvcpp_ref`; the adapter is
`rwkv_ssd/backends/rwkvcpp.py` and the provider bridge is coordinated by
`rwkv_ssd/runtime/engine.py`.

### Build and convert

```powershell
cmake -S backends/rwkvcpp_ref -B backends/rwkvcpp_ref/build
cmake --build backends/rwkvcpp_ref/build --config Release
python backends/rwkvcpp_ref/python/convert_pytorch_to_ggml.py `
  C:\models\rwkv-model.pth `
  C:\prepared\rwkv-model-FP16.bin FP16
```

On Windows the build commonly emits
`backends/rwkvcpp_ref/build/bin/librwkv.dll`; set `RWKVCPP_DLL` if the DLL is
elsewhere. Set `RWKVCPP_GGML_PATH` for the converted model, or enable
`RWKVCPP_AUTO_CONVERT=1` for an opt-in first-load conversion.

### Runtime controls

| Variable | Meaning |
|---|---|
| `RWKVCPP_ROOT` | Override the vendored rwkv.cpp root |
| `RWKVCPP_DLL` | Explicit native DLL path |
| `RWKVCPP_GGML_PATH` | Explicit matching GGML model path |
| `RWKVCPP_AUTO_CONVERT` | Opt-in checkpoint-to-GGML conversion |
| `RWKVCPP_GPU_LAYERS` | Native offload setting; hardware-gated and default 0 |
| `RWKV_CPU_THREADS` | Native thread count; integer is authoritative, `auto` is model-width aware |
| `RWKVCPP_SYNC_EVERY_TOKEN` | Diagnostic per-token upload mode; default is weight-stationary |
| `RWKVCPP_NATIVE_U8` | `auto` / `0` / `1`; select or disable the CPU-native SG8 grouped-U8 graph |
| `RWKVCPP_NATIVE_U8_PACKED_ONLY` | `auto` / `0` / `1`; omit dense 2-D matrix payloads when every manifest matrix is grouped-U8 |
| `RWKV_GGML_SLOT_COUNT` / `RWKV_GGML_SLOT_BYTES` | Opt-in bounded native upload slots; both are required |
| `RWKVCPP_LAYER_CACHE_BYTES` | Maximum decoded active-layer payload retained by the optional native layer LRU; accepts decimal, `0x` values, or `auto` |

When `RWKV_CPU_THREADS` is unset or `auto`, the engine provides manifest
`n_embd` before native model construction and uses this policy:

| Model width | Automatic native threads |
|---:|---:|
| `<1280` | 1 |
| `1280–2047` | 2–4, host-scaled |
| `>=2048` | 2–4, host-scaled and capped |

The policy is intentionally not “all host threads”: single-token quantized
GEMV is memory/cache bound on the measured Windows AVX2 host, and eight native
threads slowed the 2.9B runs. `RWKV_CPU_THREADS=N` remains the escape hatch for
benchmarking another CPU.

### Native layer-local execution and cache accounting

For a non-resident `rwkvcpp` request, the required native ABI executes one
RWKV block at a time. The GGML graph plan is reusable, but only the active
block's weights, recurrent state, and activations are live in that plan. The
provider may retain compressed source records separately; it must not retain a
second unbounded copy of the model to qualify an F1-F4 tier. F5/promoted
resident mode intentionally selects the complete native graph as the speed
reference.

`RWKVCPP_LAYER_CACHE_BYTES` caps the optional native decoded-layer LRU. A value
of `0` disables that LRU; `auto` keeps the profile-selected cap (currently
smaller for F1 than for F2-F4). The cap covers decoded payloads owned by the
native layer cache, not the active layer plan, provider packed cache, process
overhead, or the GGML model file mapping. The runtime reports those domains
separately:

| Metric | Meaning |
|---|---|
| `native_upload_bytes` | Bytes passed from the Python bridge into the native ABI |
| `native_packed_bytes` | Packed provider-record bytes in those uploads (for example SG8 records) |
| `native_active_bytes` | Decoded byte footprint installed in the active native layer plan |
| `native_layer_cache_bytes` | Current decoded-payload LRU usage, bounded by the configured cap |
| `native_layer_cache_hits` / `misses` / `evictions` | Native LRU tensor/cache events |
| `native_decoded_cache_hits` | Complete-layer restores that avoided provider decode/materialization |
| `provider_cache_bytes` | Process-owned provider bytes, deduplicated across aliases |
| `provider_layer_cache_bytes` / `provider_resident_bytes` | Evictable layer bytes versus resident provider-owned bytes |
| `provider_mmap_bytes` | Logical packed payload retained through stable file-backed views; not heap RAM |

The optional cached-token ABI is reported as
`supports_native_cached_layer_step`. When it is present, the backend can run
a token step against a ready cached layer. When it is absent, the backend
falls back to the qualified layer-step ABI and still preserves bounded
layer-local execution; it does not silently switch F1-F4 to a full resident
graph. A requested F1-F4 tier fails at load time if the required native
layer-streaming ABI itself is missing.

Dense FP16/BF16/FP32 layers also have a borrowed upload ABI. Strict F1 uses the
transient form, whose owner is released at the next `rwkv_layer_begin`; a
bounded provider cache may opt into the persistent form to avoid copying a
decoded dense layer into the active plan. The native plan stores pointers only,
not a second model copy. Provider eviction calls the invalidation ABI before
releasing its tensor/view, so stale borrowed pointers cannot be restored. A
provider must not claim persistent borrowing merely because the source pack is
mmap-backed; it must retain the decoded owner within its configured cache.

### Measured CPU results

The July 2026 local 2.9B native comparison measured the following best
observed decode rates:

| Artifact | Rate | Interpretation |
|---|---:|---|
| FP16 GGML | ~2.8 tok/s | Safe native quality/speed reference |
| Q5_1 GGML | ~3.5 tok/s | Experimental; short greedy run diverged from FP16 |
| Q4_K GGML | ~4.2–4.6 tok/s | Fastest observed native artifact; short prefix matched |
| Q4_0 GGML | ~3.7 tok/s | Experimental comparison |

These are resident native-GGML results, not SSD provider results. The
grouped-U8 provider pack measured about 1.36 tok/s warmed end-to-end on the
same machine. The 15 tok/s 2.9B goal was not reached; a higher result is
hardware-gated.

The historical g64 native grouped-U8 packed-only run on the same 2.9B model
measured approximately 1.83 / 2.45 / **2.62** / 2.19 tok/s at 1 / 2 / 4 / 8
threads after upload; four threads won on this AVX2 host. This is historical
g64 warm native decode, not a current g32 throughput measurement or cold
SSD/provider throughput.

### Native grouped-U8 in rwkv.cpp

The vendored source contains a CPU-only `SG8\x01` ABI. Each grouped record
stores a `uint32` group size, one float32 min/max pair per group, and one UINT8
value per matrix element. The loader exposes two flags through
`rwkv_init_from_file_ex`:

| Flag | Meaning |
|---:|---|
| `1` | Enable native grouped-U8 graph nodes while retaining dense GGML storage as a fallback |
| `2` | Add packed-only residency; large matrix payloads are shape-only in GGML and must be uploaded through `rwkv_set_tensor_scale_u8_grouped` |

The Python wrapper and engine select flags `3` automatically for a complete
all-grouped 2-D streaming pack. The bridge includes the embedding in layer 0
and the output head in global layer 9999; both are required for a complete
packed-only graph. RWKV-7 transposed `w/a/v/g` adapter matrices use a native
row-major `x @ W` kernel, while small elementwise controls are decoded into
their existing dense GGML tensors. Upload records are copied synchronously,
so the provider may release its staging span after the call.

The path validates SG8 magic, group size, exact payload length, finite affine
metadata, and matrix shape. Missing uploads fail at evaluation with a data
error instead of producing partial logits. CUDA/Metal/other accelerator
offload keeps the ordinary GGML path; native grouped-U8 is deliberately not an
accelerator claim.

### Provider-backed streaming boundary

The provider path can reuse the engine's pack, cache, prefetch, residency, and
metrics infrastructure. Its default synchronization is weight-stationary:
after the provider uploads the graph's required weights, the engine avoids
re-uploading every layer for every token. `RWKVCPP_SYNC_EVERY_TOKEN=1` exists
for bridge diagnostics and comparisons.

The path remains quality-qualified per pack: the old all-LUT2 2.9B pack is
archived because it fails the application-quality gate, while the promoted
g32 grouped-quality pack is the default compact direction with a short-smoke
certificate. The current four-way
0.1B conformance run passes exact greedy token/text parity, with minimum native
top-10 overlap 0.90, KL below 0.005, and relative state error below 0.016.

The current raw 0.1B native F-tier acceptance run (64 generated tokens x 5
samples, one CPU thread) measured 51.26/50.98, 53.31/52.13, 52.63/49.69,
50.06/50.46, and 54.15/53.07 tok/s for F1/F2/F3/F4/F5 (cold/warm). Every
F1-F4 cold and warm ratio gate passed. The larger real 2.9B grouped-U8 run
does not certify lossy quality, but its bounded native F1-F4 throughput and
memory gates passed with 8 generated tokens x 2 samples, a 12 GB RSS envelope,
a 1 GiB provider cap, and a 512 MiB native cap: F1 2.02/1.90, F2 2.02/1.99,
F3 2.24/1.90, and F4 2.25/2.12 tok/s (cold/warm), against F5 2.03/1.85.

Provider memory accounting now separates process-owned bytes from stable
file-backed views. `provider_cache_bytes` is the owned total;
`provider_layer_cache_bytes` and `provider_resident_bytes` are its subdomains;
`provider_mmap_bytes` reports retained logical payload through mmap without
pretending it is a second heap copy. This prevents resident native views from
being double-counted while keeping the explicit provider cap auditable.

The active F-tier benchmark (`bench/bench_f1_f3.py`) reports cold and warm
observations independently. Its F1/F2/F3/F4 ratios are acceptance gates
against F5 (60%, 80%, 80%, and 80%, respectively), not labels that imply a
tier has passed. Use `--enforce-f-tier-gates` only for a measured acceptance
run with sustained generation, multiple samples, and the required fixture
coverage. RSS high-water, fixed process overhead, provider-owned bytes, and
file-backed mmap payloads are reported separately from the native layer cache.

## ChatRWKV and F tiers

ChatRWKV is the most useful CPU reference for pack correctness and residency
behavior. It supports resident, partial, strict SSD, bounded provider cache,
promote-to-full-`z`, prefix state cache, and the F-tier presets. See
[`PRESETS.md`](PRESETS.md) for the decision tree and
[`V1_STREAMING.md`](V1_STREAMING.md) for the code map.

The RWKV7a DeepEmbed-v1 checkpoint is supported by the native ChatRWKV path.
The older qkv/DEA DeepEmbed contract has a correctness-first CPU reference
adapter with a `DeepEmbed.bin` sidecar; it is not a fused rwkv.cpp backend.

## Other backends

### Synthetic

Use `--backend synthetic` for deterministic resident-vs-streaming tests and
benchmark harness validation. It is not a language-model quality benchmark.

### Removed experimental backends

Obsolete generic sequence-model adapters were removed. RWKV is the maintained
runtime contract, keeping the capability matrix, state path, and CPU
performance work focused on the SSD/F-tier pipeline.

### Accelerators and external engines

`docs/ACCELERATORS.md` records the current XPU/CUDA/MPS boundary. The external
backend references in the original evaluation are useful for design
comparison, but the repository does not claim their kernels or throughput.
