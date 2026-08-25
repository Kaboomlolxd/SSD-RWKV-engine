# V1 — ChatRWKV pack streaming

**Status:** **done** (skeleton load, golden tests, decoupled cache, promote-to-full-`z`, prefix state cache, fused LUT presets). Historical design notes: [`../archive/docs/V1_STREAMING_HOOK.md`](../archive/docs/V1_STREAMING_HOOK.md).

ChatRWKV RWKV-7 keeps weights in a flat dict `model.z`. Streaming reads middle block tensors from `weights.bin` each token (or retains a bounded LRU in `z`).

## Code map

| Piece | Location |
|-------|----------|
| Pack I/O + prefetch | `rwkv_ssd/runtime/weight_provider.py` |
| RWKV-7 layout / promote | `rwkv_ssd/runtime/rwkv7_weights.py` |
| Layer grouping | `rwkv_ssd/runtime/layer_keys.py` |
| Skeleton globals | `rwkv_ssd/runtime/rwkv7_skeleton.py` |
| Streaming forward | `rwkv_ssd/backends/rwkv7_forward.py` |
| Fused att / FFN / head | `rwkv_ssd/runtime/rwkv7_linear.py`, `lut_gemm_fused.py` |
| Presets | `rwkv_ssd/runtime/throughput_defaults.py` |
| Golden (synthetic) | `rwkv_ssd/backends/synthetic.py` |
| Resident + streaming load | `rwkv_ssd/backends/chatrwkv.py` |

## Modes (quick)

| Shape | CLI / env | `z` RAM (0.1B Trinity) |
|-------|-----------|------------------------|
| Strict | `--mode streaming` (no cache) | ~201 MB skeleton |
| SSD tier + fused | `RWKV_SSD_TIER=1` | ~201 MB; mmap + fused GEMV |
| Partial hot3 + fused | `RWKV_PARTIAL_FUSED=1` | ~260 MB; pin 0–2, stream 3–10 |
| Bounded fused | `apply_bounded_fused_defaults` / bench | ~203 MB; 2-layer `z` |
| Default auto stream+cache | `apply_streaming_defaults` | ~231 MB; 1–2 layer `z` (promote off on n≥8) |
| Stream + cache | default Trinity streaming | ~382 MB after promote |
| Partial / RAM budget | `--mode partial`, `--ram-budget-gb` | profile-dependent |

Full preset table: [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md#ram-vs-speed-presets-june-2026).

## Smoke test

```powershell
python -m app.cli --model C:\prepared\reference-pack `
 --checkpoint C:\models\rwkv-model.pth `
 --backend chatrwkv --mode streaming --stream-layer-cache `
 --strategy "cpu bf16" --max-tokens 32
```

I/O ceiling / RAM frontier (inf-compute model + Pareto presets): `python bench/bench_io_ceiling.py --heavy`

## See also

- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog (prefetch, fused, promote); V1 is the runtime that uses those mechanisms.
- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — V1 is "Done" — see the measured baselines for current V1 performance.
- **[`PRESETS.md`](PRESETS.md)** — env-var decisions that pick the V1 streaming preset.
