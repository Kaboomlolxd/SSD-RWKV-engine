# Archived ideas — out-there sweep (June 2026)

Ideas moved here after `bench/bench_out_there.py --full` showed **weak or wrong fit**
for the current engine / dev machine. Not deleted — thesis sims remain in `simulations/benches/`.

**Source:** [`docs/OUT_THERE_RESULTS.md`](../../docs/OUT_THERE_RESULTS.md)

## Idea problems (do not prioritize on engine path)

| Idea | Sweep ID | Verdict | Why archived |
|------|----------|---------|--------------|
| **VcLLM / NVDEC weight decode** | SIM10 Path D | **Wrong tool** | 0.04× tok/s vs baseline; NVDEC expects video NAL units, not arbitrary bitstreams |
| **N-gram weight cache (decode)** | E08, SIM04 | **Workload mismatch** | 0.11% UCB hit @ 1 GB; +6% tok/s on single-user F1 — prefill-only niche |
| **Layer-only prefetch (no gate)** | E07 | **Regresses** | −7% vs F1 baseline when compute-bound; gate prefetch wins (+26%) |
| **Prefix state library (generic traffic)** | SIM03 | **Low hit rate** | 4.85% hits, 1.05× mean latency — only wins with repeated system prefixes (see E09) |
| **Engram three-tier (current contract)** | S07, SIM16 | **Deferred** | Model contract change; sim neutral (SSD ≈ RAM @ batch 512) |
| **HRWKV7 hybrid** | S05 | **Deferred** | M8 stretch; no pack/engine path |
| **Compression Trinity on hot path** | IDEAS experiment | **Shelves codec** | 10× storage validated (SIM11); engine uses LUT2; Trinity stack not default codec |

## Environment / hardware — not tested (not idea failures)

| Item | IDs | Blocker |
|------|-----|---------|
| GDS NVMe→VRAM | S01, SIM17 | Linux + NVIDIA |
| FLUTE CUDA GEMV | S02 | Not wired in engine |
| Dual-NVMe topology | S03 | Single drive dev box |
| Shared decode cache multi-process | S08 | Manual ops |

## Implementation gaps (idea may be fine)

Tracked in [`docs/ACTIVE_WINS.md`](../../docs/ACTIVE_WINS.md) — not archived as idea failures.

| Item | IDs | Status |
|------|-----|--------|
| shadow_sel streaming | E02 | **In progress** — decode retain + zero-copy |
| LoRA delta stream | S06, SIM15 | Sim pass; engine slot missing |
| Low-rank factored pack | S04 | No pack tooling |
| 0.01B quant ladder | E11–E13 | Missing ckpt + weights.bin |

## Restore

Move an item back to `docs/IDEAS.md` P2 when pass criteria are met on real engine hardware.
