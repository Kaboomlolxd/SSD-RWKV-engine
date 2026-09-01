# Benchmark index

> Current commands and production/focused/exploratory boundaries live in
> [`../bench/README.md`](../bench/README.md). Current interpretation and latest
> results live in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

**Active:** only three scripts in [`../bench/`](../bench/). Archived benches live in [`../archive/bench/`](../archive/bench/) (codec matrices, Trinity microbenches, legacy scenarios).

> **Release baseline (July 2026):** the historical `trinity_lut2_*` and
> shadow-pack payloads were removed because they fail real-model generation-quality
> gates. In automatic profile mode, requests for a historical LUT2 path first
> resolve to the mixed `trinity_grouped_0.1b` compact pack (326 large tensors
> use `scale_u8_grouped` g256; 76 one-dimensional control/normalization vectors
> remain dense BF16), then to the parity-validated `trinity_safe_0.1b` FP16/BF16
> fallback if the grouped pack is absent. The grouped pack passes the current
> short resident-vs-streaming parity gate; use the dense fallback for broader
> reference claims until longer certificates are available.
> Explicit LUT2 numbers remain historical engineering measurements.

> **June 27, 2026 bench audit** — the bench and the F1-F5 tier defaults had
> several bugs that were making every tier report 30-60% slower than
> steady-state. See [`BASELINE_BUGS.md`](BASELINE_BUGS.md) for the bug
> sweep. After fixes: F5 = 9.01 tok/s on the dev machine, F5 ≥ F6.

## Frontier reference points (F5 vs F6)

| ID | What it is | Same RAM? | Use for |
|----|------------|-----------|---------|
| **F5** | Certified FP16/BF16 pack, stream+cache, **full `z` promote** | ~382 MB | **Streaming ceiling** — best this engine does from SSD |
| **F6** | Classic **resident** — load `.pth`, no pack stream | ~382 MB | **External baseline** — normal ChatRWKV all-RAM path |

They are **not the same code path** (pack decode+promote vs direct checkpoint load), but **`z_mb` is similar** (~full bf16 weights). On your machine F5 can match or beat F6 tok/s — that means the **tok/s gap to close is at F1–F4** (low RAM), not at max RAM.

**Compare low-RAM tiers to F5** for streaming work. Keep **F6** as sanity check vs "just load the model."

> **June 27, 2026 update:** F5 = **9.01 tok/s** (warm, 0.1B, dev machine). F6
> = 3.98 tok/s. **F5 ≥ F6 ✓** for the first time — the pack-stream+promote path
> is now faster than loading the.pth. The F1-F4 vs F5 gap is the remaining
> work; see [`BASELINE_BUGS.md`](BASELINE_BUGS.md) §"What still needs work".

## Trinity (grouped compact default; legacy LUT2 removed)

| Aspect | Status |
|--------|--------|
| **Engine path** | Historical LUT2 requests auto-resolve to `trinity_grouped_0.1b`; explicit LUT2 requires a separately rebuilt pack |
| **Compact path** | Mixed `scale_u8_grouped` g256 (326 large tensors) plus dense BF16 vectors (76 tensors); about 189 MiB for the checked-in 0.1B pack |
| **Correctness path** | `trinity_safe_0.1b` reference / other certified FP16/BF16 pack with resident-vs-streaming parity |
| **Quality boundary** | Corrected mixed grouped g256 passes the current short real-model resident-vs-streaming greedy parity gate with fused decode off/on |
| **tok/s** | Compact-pack measurements are useful for engineering; production claims must identify the selected pack and evidence |
| **Archived** | Trinity *microbenches* (`bench_trinity_*`) in `archive/bench/` — codec still shipped |

See ·.

## Run these

| Bench | What it measures | Example |
|-------|------------------|---------|
| **`bench_throughput.py`** | **Real** RWKV-7 (ChatRWKV): tok/s, `z_mb`, `provider_mb`, `vs_resident`, layer CSV | `python bench/bench_throughput.py --backend chatrwkv --heavy` |
| **`bench_io_ceiling.py`** | **Inf-compute estimate** + **RAM frontier**: Pareto-best preset per `z` tier, I/O ceiling tok/s, Δtok/s/MB | `python bench/bench_io_ceiling.py --heavy` |
| **`bench_f1_f3.py`** | **F1/F2/F3/F4/F5/F6 cold/warm acceptance** on real packs (ChatRWKV or native rwkv.cpp) | `python bench/bench_f1_f3.py --backend rwkvcpp --tiers F1,F2,F3,F4,F5` |
| **`bench_rwkvcpp_weight_stationary.py`** | **Native rwkv.cpp CPU shared prefill/decode** versus independent sessions | `python bench/bench_rwkvcpp_weight_stationary.py --pack C:\prepared\runtime-pack --checkpoint C:\models\rwkv-model.bin` |
| **`bench_backend_compare.py`** | **ChatRWKV vs rwkvcpp** (resident + streaming tiers) | `python bench/bench_backend_compare.py --max-tokens 16 --samples 3` |
| **`bench_generate.py`** | **Synthetic** toy pack: resident / partial / streaming tok/s + RAM | `python bench/bench_generate.py --json` |
| **`scripts/bench_tok_s.py`** | **Quick tok/s sweep** (resident + raw + F4 + F5). Use for one-shot comparisons without the full suite. F1-F3 (ram_budget_gb) tiers are intentionally skipped on small packs — see [§ RAM vs speed presets](THROUGHPUT_PLAN.md). | `python scripts/bench_tok_s.py --pack C:\prepared\reference-pack --checkpoint C:\models\rwkv-model.pth` |

All-in-one:

```bash
python -m rwkv_ssd.tools.run_throughput_suite
python -m rwkv_ssd.tools.run_throughput_suite --heavy # 0.1B real + frontier
```

### Outputs

| Bench | Default JSON |
|-------|----------------|
| Frontier | operator-generated `ram_frontier.json` evidence |
| Throughput | stdout + optional `--json-out` |
| Synthetic | stdout or `--json-out` |

### `bench_io_ceiling.py` flags

| Flag | Use |
|------|-----|
| `--heavy` | 0.1B pack (**quick profile** by default); verify the resolved pack and certificate before using results |
| `--full` | All F scenarios, 48 tokens, 3 samples, warm disk cache at load |
| `--quick` | Explicit quick: F1+F3+F5 only, skip load-time warm |
| `--legacy` | Archived subsumed scenarios (regression only) |
| `--diagnostics` | D0/D1 cold baselines |

## Archived (do not use for product numbers)

Moved to `archive/bench/`: `bench_tok_s_compare`, `bench_full_matrix`, `bench_trinity_*`, `bench_io_backends`, `bench_cold_ssd`, `bench_lut_kernel`, `bench_pack_compare`, `bench_quant_ladder`, `bench_chatrwkv`, `bench_memory_tier`, `bench_prefetch_io_shadow`, `bench_optimization_verify`, `bench_io`, etc.

See [`../archive/README.md`](../archive/README.md).

## Research-only (not engine CI)

| Path | Role |
|------|------|
| [`../storage_bench/`](../storage_bench/) | Linux Rust io_uring sequential ceiling |

## Reporting rules

- **RAM vs tok/s claims:** use frontier output (`bench_io_ceiling.py`) — Pareto points only
- **Real steady tok/s:** `bench_throughput.py` with `--heavy`, report `z_mb` + `provider_mb`
- **Synthetic ratio:** `bench_generate.py` (not comparable to ChatRWKV absolute tok/s)
- Do not multiply speedups across independent bench rows

### Native rwkv.cpp F-tier acceptance

`bench/bench_f1_f3.py` is the acceptance harness for the native CPU layer
path. It records the first request as the cold observation and a subsequent
request as warm; those are reported separately along with median/p95 token
latency, prefill/decode time, RSS high-water, streamed/provider bytes, native
upload bytes, active decoded bytes, cache usage, and cache hits. The fixed
interpreter/process baseline is reported separately from model and provider
memory.

F1-F4 are compared with the F5 resident/native fast path using both cold and
warm ratios. The required minimums are F1 ≥ 0.60× F5 and F2/F3/F4 ≥ 0.80× F5.
The JSON gate is descriptive unless `--enforce-f-tier-gates` is supplied; a
row name alone never certifies a tier. Use sustained runs (at least 64–128
generated tokens, three samples where practical), then validate the 0.1B
fixture and a larger real checkpoint before publishing a result.

For native-cache telemetry, `native_upload_bytes` is the Python-to-ABI
payload, `native_packed_bytes` is the packed provider representation,
`native_active_bytes` is the decoded active-layer footprint, and
`native_layer_cache_bytes` is the bounded native decoded-payload LRU. These
are different byte domains and must not be added as if they were independent
resident model copies.

## See also

- **[`MILESTONE_STATUS.md`](MILESTONE_STATUS.md)** — current Pareto winner and F1–F5 tok/s table that these benches populate.
- **[`BASELINE_BUGS.md`](BASELINE_BUGS.md)** — bench evidence per fix; explains *why* the numbers changed.
- **[`THROUGHPUT_PLAN.md`](THROUGHPUT_PLAN.md)** — mechanism catalog that the benches measure.
- **[`PRESETS.md`](PRESETS.md)** — env-var decisions that affect which preset a bench scenario runs.
- **[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md)** — the sharded pack bench results (multi-SSD comparison) live in the SSD exploitation doc, not here.
