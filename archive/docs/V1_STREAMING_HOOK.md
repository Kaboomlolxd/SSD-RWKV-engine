# V1 — ChatRWKV streaming hook

**Status:** **V1 complete** — skeleton load, golden tests, decoupled cache, promote-to-full-`z`, prefix state cache on ChatRWKV (v0.6.10). Active work: Trinity LUT2 tok/s gap.

V0 proved SSD streaming on the **synthetic** backend (M3). V1 wires the same `ManifestWeightProvider` path into **real RWKV-7** inference.

## Why this is non-trivial

ChatRWKV RWKV-7 does **not** use `nn.Module.state_dict()`. Weights live in a flat dict `model.z`, populated once at load from the full `.pth`. `forward_one` loops `for i in range(n_layer)` and reads `z['blocks.{i}.…']` for every tensor in that block.

True streaming means:

1. **Resident:** `emb.weight`, `head.weight`, `ln_out.*` (and optional hot layers per residency profile).
2. **Per token / per layer step:** read layer `i` tensors from `weights.bin` → `prepare_rwkv7_tensor_for_z` → `inject_layer_into_z(z, …)` → run the same TMix/CMix ops as `forward_one` for layer `i` only.
3. **Evict** layer `i` tensors from RAM after the layer compute (strict `streaming`), or retain a **small LRU cap** (`--stream-layer-cache`, default `--max-layers-in-z 1`).

Prefetch layer `i+1` while computing layer `i` (already implemented in `ManifestWeightProvider`).

### Streaming shapes (v0.6.4+)

| CLI | RAM in `model.z` | Throughput |
|-----|------------------|------------|
| `--mode streaming` | globals only (+ skeleton hot layers) | M3 proof; strict inject every layer |
| `--stream-layer-cache` | globals + ≤N block layers (`--max-layers-in-z`) | warm layers; **promote** to full `z` after provider warm (v0.6.9) |
| `--warm-z` | full model in `z` at load | native `forward` immediately; benchmark / max RAM |
| `--low-ram` / `--ram-budget-gb` | partial pin + bounded provider LRU | thesis 200B-class path |
| `--state-cache` + `--system-prefix` | same `z` policy; skips system **prefill** on repeat | serving / chat workloads |

Bench defaults: **0.01B** pack (`bench/bench_throughput.py`); `--heavy` for **0.1B**.

## Code map

| Piece | Location |
|-------|----------|
| Pack I/O + prefetch | `runtime/weight_provider.py` |
| RWKV-7 layout transforms | `runtime/rwkv7_weights.py` |
| Layer grouping | `runtime/layer_keys.py` |
| Golden reference (V0) | `backends/synthetic.py` |
| Resident path (V0) | `backends/chatrwkv.py` → full `.pth` load |
| **V1 forward** | `backends/rwkv7_forward.py` — `forward_one_streaming` + `greedy_token_ids_streaming` |

## Acceptance (V1 M3 bar on real model)

- [x] Greedy `streaming` == resident for **≥32** tokens on `test_model` 0.1B (`test_real_model_streaming.py`)
- [x] Lower `model.z` footprint in streaming (~201 MB vs ~382 MB on 0.1B — emb+head stay resident)
- [x] Metrics CSV shows non-zero `read_ms` per streamed layer when using `--mode streaming`

## Implementation order

1. **Parity** — done (`test_rwkv7_pack_z_parity.py`).
2. **Layer forward** — done (`rwkv7_forward.py`).
3. **Full decode loop** — done; CLI `--mode streaming` uses pack injection each layer per token.
4. **Golden test** — done (8 default, 32 with `@pytest.mark.slow`).
5. **Skeleton load** — done (`build_rwkv7_skeleton_from_pack`, default for streaming).

## Next — post-V1 ([`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md))

| Priority | Item | Status |
|----------|------|--------|
| 1 | Decouple provider / `z` eviction | **done (v0.6.7)** |
| 2 | Disk decode cache + shadow | **done (v0.6.7–v0.6.10)** |
| 3 | Promote stream cache → full `z` | **done (v0.6.9)** — FP16 ≈ resident |
| 4 | Prefix state cache on ChatRWKV | **done (v0.6.10)** |
| 5 | RAM budget partial planner | **done (v0.6.8)**; byte LRU TODO |
| 6 | Trinity LUT2 ≈ FP16 without shadow | **in progress** |
| 7 | Re-bench `tok_s_compare.json` | **TODO** |
| 8 | M6 Albatross compute | [`M6_ALBATROSS.md`](M6_ALBATROSS.md) |

Do not stack thesis simulation speedups; see thesis §2.5.

## Out of scope here

- Albatross / rwkv.cpp backends (M6)
- Quantized packs until M5 + Trinity experiment
- GPU H2D overlap (ping-pong scaffold exists; enable when `device=cuda`)

## References

- [`IDEAS.md`](IDEAS.md) P0–P2
- [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)
- [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)
- ChatRWKV `rwkv/model.py` — `self.z` and `forward_one`
