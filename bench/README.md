# Benchmark catalog

This directory is cataloged by purpose. The scripts remain in one directory
for compatibility with existing commands; this catalog separates production
reporting from focused and exploratory experiments.

The project-wide interpretation of the latest results is in
[`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md). The reporting contract
is in [`RESULTS.md`](RESULTS.md).

## Canonical production benches

Run these when a result may influence a launcher, preset, or project status.

| Script | Measures | Typical command |
|---|---|---|
| `bench_f1_f3.py` | Real ChatRWKV F1/F2/F3/F4/F5/F6 tiers | `python bench/bench_f1_f3.py --backend chatrwkv --tiers F6,F1,F3` |
| `bench_throughput.py` | Real-model tok/s, RAM, and per-layer metrics | `python bench/bench_throughput.py --heavy` |
| `bench_streaming_matrix.py` | Warmups, repeated samples, cache and stage timing | `python bench/bench_streaming_matrix.py --help` |
| `bench_io_ceiling.py` | I/O ceiling and RAM frontier | `python bench/bench_io_ceiling.py --heavy` |
| `bench_generate.py` | Synthetic resident/partial/streaming regression | `python bench/bench_generate.py --json` |

For an RWKV-7 ChatRWKV CPU comparison, supply the operator's prepared pack
and original checkpoint explicitly:

```powershell
$env:RWKV_LUT_KERNEL = "c"
$pack = "C:\prepared\rwkv-model.pack"
$checkpoint = "C:\models\rwkv-model.pth"
.\.venv-cpu\Scripts\python.exe bench\bench_f1_f3.py `
  --backend chatrwkv `
  --pack $pack `
  --checkpoint $checkpoint `
  --tiers F6,F1,F3 --max-tokens 16 --samples 2 `
  --prompt "Throughput bench prompt" --strategy "cpu bf16" `
  --io-backend mmap --decode-disk-cache 0
```

Change only the pack and tiers for an A/B comparison. Do not use a different
checkpoint or prompt for the competing pack.

## Focused engineering benches

These answer one mechanism question and must not be combined into a headline
speedup without a separate end-to-end run.

| Scope | Scripts |
|---|---|
| Backend comparison | `bench_backend_compare.py`, `bench_rwkvcpp_quality.py` |
| Weight scheduling | `bench_weight_stationary.py`, `bench_chatrwkv_weight_stationary.py`, `bench_rwkvcpp_weight_stationary.py` |
| Codec quality/size | `bench_compression_ab.py`, `bench_lut_bitwidth_research.py` |
| Storage/layout | `bench_shard_simulation.py`, `bench_streaming_matrix.py` |
| Context/state | `bench_context_frontier.py`, `bench_state_parking.py` |
| Shadow/cache behavior | `bench_shadow_sel_quick.py`, `bench_io_ceiling.py` |
| CPU optimization A/B probes | `bench_cpu_optimizations.py` (grouped decode, DeepEmbed sidecar/batch paths, state publication, whole-pack zstd); `bench_rwkvcpp_weight_stationary.py` (native shared prefill/decode) |

The CPU optimization probe uses a real uncompressed pack for the storage A/B
and creates disposable synthetic fixtures for the other mechanisms:

```powershell
python bench/bench_cpu_optimizations.py `
  --pack C:\prepared\runtime_pack_2.9b_grouped_quality `
  --repeats 3
```

The latest local run measured 5.05× grouped-U8 decode, 4.63× grouped-LUT2
decode, approximately 19×–1,202× DeepEmbed sidecar lookup improvement across
access patterns, 1.17× two-session DeepEmbed batch wall time with 50% fewer
layer loads, and 2,842.2 ms raw-mmap versus 8,898.5 ms zstd load-plus-read on
the 2.9B pack. These focused numbers are mechanism evidence, not an additive
end-to-end tok/s claim; the zstd result is why raw mmap remains the hot-path
default.

For native rwkv.cpp batch capacity, use `bench_rwkvcpp_weight_stationary.py`.
The latest three-sample 0.1B run matched independent output and measured
2.67 aggregate tok/s for two shared sessions versus 1.34 tok/s independently,
about 2.00× wall-time speedup. It covers native layer-outer/session-inner
prefill and decode on normal files with a warm page cache.

## Exploratory benches

`bench_out_there.py` and `bench_additive_opportunities.py` are idea-screening
tools. Their output is useful for prioritization but is not a production
baseline. New one-off experiments should have a clear question, an explicit
gate, and a JSON output schema.

## Results and archival rules

- Put reproducible JSON in [`results/`](results/) with a dated name.
- Put reusable benchmark helpers in `bench/`; do not put one-off application
  scripts there.
- Put stale scripts, superseded reports, and abandoned experiments in
  [`../archive/bench/`](../archive/bench/) after checking references.
- Keep raw application reports outside the engine repository or summarize them
  in `docs/PROJECT_STATUS.md`; do not let them silently become engine baselines.
- Every new result must state whether its tok/s is decode-only or end-to-end.
