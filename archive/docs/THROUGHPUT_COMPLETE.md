# Throughput phase — status (extended v0.6.10)

What the engine implements for **SSD per-layer streaming during decode** (P2 + M5), and the **v0.6.7–v0.6.10 extension** that closes the FP16 streaming vs resident gap. This does **not** include M6 GPU backends or thesis simulation multipliers.

> **Note:** The throughput phase was originally marked “complete” at v0.6.0. Work continued through **v0.6.10** after measuring that decoupled caching and `z` promotion were required for real-world tok/s — see [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md) deviations table.

## Done (engine path)

### P2.a — latency hiders

| Mechanism | CLI / tool |
|-----------|------------|
| Layer / gate / layer_aware prefetch | `--prefetch-policy` |
| Micro-pipeline chunked reads | `--io-chunk-bytes`, `--io-chunk-policy layer_size` (auto on streaming load) |
| mmap madvise WILLNEED / DONTNEED | `--no-mmap-willneed`, `--mmap-dontneed` |
| Contiguous prefetch + single `read_bytes_span` per layer | `weight_provider._decode_stream_entries` |
| posix_fadvise on pread (Linux) | `--io-backend pread` |

### P2.c — topology / caching

| Mechanism | CLI / tool |
|-----------|------------|
| Bounded layer cache in `model.z` + provider LRU | `--stream-layer-cache`, `--max-layers-in-z` |
| **Decoupled provider / `z` eviction** (v0.6.7) | default on; `--no-decouple-provider-cache` |
| **`promote_stream_cache_to_full_z`** (v0.6.9) | auto after warm; native `forward` when all blocks in `z` |
| Inject skip when weights already in `z` | `rwkv7_weights.py`, `rwkv7_forward.py` |
| Native `forward` when all block layers resident | auto `max_z=n_layer` for ≤2-layer packs |
| Partial residency profiles | `--residency-profile`; auto hot7 via `throughput_defaults` |
| `layer_grouped` pack + sector padding | `pack_runtime --pack-layout layer_grouped` |
| **`--ram-budget-gb`** partial planner (v0.6.8) | pin layers 0…K under cap; `RWKV_RAM_BUDGET_GB` |
| **`--low-ram` preset** (v0.6.7) | bounded z + decouple + disk cache |

### P2.b — bandwidth (mutually exclusive codecs)

| Mechanism | CLI / tool |
|-----------|------------|
| M5 codecs `none`, `scale_u8`, `scale_u4` | `--pack-codec`; `eval_m5_codec` |
| Trinity `trinity_lut2` / `trinity` on engine path | `pack_runtime --pack-codec trinity_lut2` |
| **bf16 shadow** (`shadow.bin`) | `--bf16-shadow`; `add_bf16_shadow` |
| **Disk decode cache** `.decode_cache/` (v0.6.7) | auto on Trinity streaming; `--decode-disk-cache` |
| Shadow default when pack has shadow (v0.6.10) | `RWKV_DECODE_SHADOW=1` via `throughput_defaults` |

### P2.d / P2.e / I/O / bench

| Mechanism | CLI / tool |
|-----------|------------|
| Hedged read_bytes | `--io-hedged` |
| N-gram blob cache, MTP workload gate | `--ngram-weight-cache`, `--mtp-speculative` |
| mmap, pread, threaded | `--io-backend` |
| **Prefix state cache** — synthetic + ChatRWKV (v0.6.10) | `--state-cache`, `--system-prefix` |
| Bottleneck: **tok/s vs resident** | `bench/bench_throughput.py` (`vs_resident=`) |
| tok/s compare across codecs | `bench/bench_tok_s_compare.py` |

## Golden tests

- Synthetic resident == streaming (all modes)
- Real RWKV-7 0.1B resident == streaming (8 + 32 tokens)
- Synthetic **scale_u8 / scale_u4** packs: resident == streaming
- Prefix state cache: cold vs warm read_ms; token parity with no cache
- Decouple provider cache, RAM budget, stream promote (unit tests)

Trinity **real-model** golden tests may require full eval packs under `test_model/trinity_eval/`.

## Not done (explicitly out of scope or next)

| Item | Reason / next step |
|------|-------------------|
| **Trinity LUT2 ≈ FP16 resident** without shadow | **Met at stream+cache** after promote + bf16 gather; strict streaming still decode-bound |
| **Provider LRU by bytes** | Layer-count cap insufficient for 200B under 10 GB |
| **M6 Albatross / rwkv_lightning** | Separate compute integration |
| **MTP speculative decode** | Gate only; no draft model |
| **Engine io_uring** | `storage_bench/` on Linux; engine uses mmap/pread |
| **Session state on SSD** | `.state_cache/` discussed, not implemented |
| **NUMA / 2nd NVMe / micro-batch** | Hardware-specific |
| **M8 HRWKV7** | Optional stretch |
| **Fresh `tok_s_compare.json`** | Refreshed Jun 2026 — trinity_lut2 stream+cache ≈ FP16 |

## Recommended workflow

```powershell
pip install -e ".[dev]"
$env:RWKV_JIT_ON='0'

# Daily tok/s (0.01B default; --heavy for 0.1B)
python bench/bench_throughput.py --backend chatrwkv --strategy "cpu bf16"

# Codec compare (refresh results after promote)
python bench/bench_tok_s_compare.py --max-tokens 32 --samples 3 --warmup 1

# Streaming + prefix cache smoke
python -m app.cli --model test_model/trinity_eval/fp16_grouped `
  --checkpoint test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth `
  --backend chatrwkv --mode streaming --stream-layer-cache `
  --strategy "cpu bf16" --state-cache `
  --system-prefix "You are a helpful assistant." "Hello"

# Low-RAM thesis preset
python -m app.cli --model <pack> --low-ram --ram-budget-gb 10 ...

python -m pytest tests/ -q
```

## Cross-links

- [`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md) — rules and measured table
- [`MILESTONE_STATUS.md`](MILESTONE_STATUS.md) — current focus and deviations
- [`TRINITY_DECODE_ROADMAP.md`](TRINITY_DECODE_ROADMAP.md) — Trinity-specific decode path
- [`IDEAS.md`](IDEAS.md) — full backlog
