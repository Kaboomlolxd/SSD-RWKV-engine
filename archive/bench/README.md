# Archived benchmarks

Superseded by the three active benches in [`../../bench/`](../../bench/README.md):

| Active | Role |
|--------|------|
| `bench/bench_throughput.py` | Real ChatRWKV tok/s + RAM |
| `bench/bench_io_ceiling.py` | Inf-compute / RAM frontier |
| `bench/bench_generate.py` | Synthetic ratio |

## Archived scripts

| Script | Was used for | Use instead |
|--------|--------------|-------------|
| `bench_tok_s_compare.py` | Codec tok/s matrix | `bench_io_ceiling.py` frontier + `bench_throughput.py` |
| `bench_full_matrix.py` | Multi-scenario matrix | `bench_io_ceiling.py` |
| `bench_trinity_decode.py` | Per-layer decode ms | Trinity dev only |
| `bench_trinity_per_token.py` | Trinity breakdown | Trinity dev only |
| `bench_trinity_toks.py` | Trinity tok sweep | archived |
| `bench_lut_kernel.py` | LUT kernel compare | `tests/` + native build |
| `bench_io_backends.py` | Raw mmap/pread GB/s | `_raw_gbs` in `bench_io_ceiling.py` |
| `bench_io.py` | Raw mmap read smoke | `tests/test_bench_io.py` |
| `bench_cold_ssd.py` | Cold cache SSD | `bench_io_ceiling.py --diagnostics` |
| `bench_chatrwkv.py` | Resident baseline | `bench_throughput.py` resident mode |
| `bench_pack_compare.py` | Pack variants | pack tools + frontier |
| `bench_quant_ladder.py` | M5 codec ladder | `tools/compression_trinity_experiment` |
| `bench_memory_tier.py` | RAM tier scenarios | `bench_io_ceiling.py` frontier |
| `bench_prefetch_io_shadow.py` | Prefetch + shadow | integrated in engine |
| `bench_optimization_verify.py` | Regression checks | `pytest tests/` |
| `io_ceiling_legacy_scenarios.py` | Subsumed presets | `--legacy` on io_ceiling |
| `_parse_bench_json.py` | JSON helper | unused |

Run archived script example:

```bash
python archive/bench/bench_tok_s_compare.py --max-tokens 32
```
