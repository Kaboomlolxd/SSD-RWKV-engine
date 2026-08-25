# SSD health and temperature for RWKV streaming

This doc translates NVMe thermal/endurance research into **actionable settings**
for the RWKV SSD inference engine. It complements
[`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md) (performance knobs) with
**health-first** guidance.

## TL;DR

| Goal | Best approach | Trade-off |
|------|---------------|-----------|
| Lowest SSD temperature | `--mode resident` or `RWKV_PROMOTE_FULL_Z=1` after warm | Needs ~full model RAM |
| Low RAM + cooler SSD | `RWKV_SSD_HEALTH=1` + pre-built decode cache | Slightly lower tok/s vs F5 |
| Monitor before/after | `python -m rwkv_ssd.tools.ssd_health` | OS-dependent sensors |
| Cap sustained read BW | `RWKV_SSD_IO_CAP_MBPS=2000` | Artificial slowdown for cooling |
| Avoid | `RWKV_SSD_TIER=1` (strict, no cache) on long runs | Re-reads entire pack every token |

```powershell
# Health-conscious preset (partial hot3 + disk decode cache + compressed cache)
$env:RWKV_SSD_HEALTH="1"
python -m app.cli --model ./runtime_pack --backend chatrwkv --mode streaming --max-tokens 32

# One-time: build decode cache so runtime avoids LUT decode + repeated pack reads
python -m rwkv_ssd.tools.build_decode_cache --pack ./runtime_pack --compress

# Optional: cap SSD bandwidth to leave thermal headroom (MB/s)
$env:RWKV_SSD_IO_CAP_MBPS="2500"
```

---

## What stresses an SSD in this engine

The engine is **read-mostly**. Writes are limited to `.decode_cache/` under the
pack directory (per-layer bf16 blobs, ~5 MB each on 0.1B-class models).

### Read load (temperature)

Per generated token in **streaming** mode, the engine may read one layer span
from `weights.bin` (or shard files). On a 12-layer 0.1B Trinity pack (~49 MB
total), strict streaming without cache re-touches most of the pack **every
token** — a sustained sequential read pattern similar to a large file copy.

NVMe controllers typically throttle between **70–85 °C**, dropping bandwidth
30–40% until they cool. Sustained 100% read duty (no idle gaps) is the main
thermal risk, not random 4K IOPS.

Research and field data ([Swissbit HCTM](https://www.swissbit.com/files/public/Documents/TechNotes/AN4105en_HCTM_for_NVMe.pdf),
consumer NVMe thermal tests) show:

- Controller junction temp rises in **30–90 s** under full sequential read
- **Compute/decode gaps** between layer reads let the controller idle-cool
- **Smaller compressed reads** shorten the active read phase per token
- **Page-cache hits** avoid physical NAND reads entirely after warm-up

The engine's `simulations/benches/thermal_duty_cycle_bench.py` models this:
compression + compute overlap creates idle gaps that keep the controller below
the throttle threshold.

### Write load (endurance)

| Source | When | Wear impact |
|--------|------|-------------|
| `.decode_cache/` layer files | First visit per layer, or `build_decode_cache` | Low — one write per layer per pack version |
| Pack `weights.bin` | Never at runtime | None |
| OS swap / page cache | If RAM is exhausted | Avoid — keep RAM budget sane |

Consumer NVMe endurance (300–600 TBW) is ample for inference. Pre-building
`.decode_cache/` once at pack time is **better** than lazy per-layer writes
during serving (fewer surprise write bursts, predictable layout).

---

## Preset ranking (health vs speed)

From best to worst for **SSD temperature** on long streaming runs:

1. **F6 resident** — zero SSD I/O after load
2. **F5 promote-max** (`RWKV_PROMOTE_FULL_Z=1`) — one warm pass, then RAM only
3. **`RWKV_SSD_HEALTH=1`** — partial hot3 + stream cache + warm disk cache
4. **F3 partial-hot3** (`RWKV_PARTIAL_SSD_TIER=1`) — fewer layers streamed per token
5. **F2 bounded** — small LRU; may re-read many layers (variable thermal duty)
6. **F1 strict** (`RWKV_SSD_TIER=1`) — **worst for temperature**: full pack sweep every token

For 7B+ models where the pack does not fit in RAM, combine:

- Trinity / tiered packs (smaller cold reads)
- Pre-built `.decode_cache/` (`build_decode_cache --compress`)
- Partial residency profile (pin hot layers)
- Optional `RWKV_SSD_IO_CAP_MBPS` if the drive still thermally throttles

---

## Engine knobs that affect SSD health

### Reduce physical reads

| Knob | Effect |
|------|--------|
| `stream_layer_cache` / `RWKV_STREAM_LAYER_CACHE` | Retain decoded layers in RAM |
| `RWKV_PROMOTE_FULL_Z=1` | All block weights in `z` after warm |
| `RWKV_WARM_DISK_CACHE=1` | Pre-build `.decode_cache/` at load |
| `build_decode_cache --compress` | One-time offline cache; zlib reads are smaller |
| Partial residency (`RWKV_PARTIAL_SSD_TIER=1`) | Pin early/last layers; stream middle only |
| Trinity / tiered pack | Smaller bytes per cold layer |

### Avoid extra SSD traffic

| Knob | Why avoid for health |
|------|----------------------|
| `--mmap-dontneed` / `mmap_dontneed=True` | Evicts OS page cache → forces NAND re-reads |
| `io_backend=cold` | Bypasses page cache intentionally (bench only) |
| `RWKV_SSD_TIER=1` without cache | Maximum read duty cycle |

### Thermal duty cycle (compute overlap)

| Knob | Effect |
|------|--------|
| `RWKV_PREFETCH_IO_ONLY=auto` (default on CPU+quant) | Prefetch raw bytes while computing previous layer — creates read/compute alternation |
| Fused LUT (`RWKV_LUT_GEMM_FUSED=1`) | Longer compute per layer → more SSD idle time per token |
| `RWKV_SSD_IO_CAP_MBPS=N` | Hard cap on read bandwidth; trades tok/s for cooler sustained operation |

### Write gentleness

| Knob | Effect |
|------|--------|
| `RWKV_DECODE_CACHE_WRITERS=1` | Serial cache writes (default in health preset) |
| `RWKV_DECODE_CACHE_COMPRESS=1` | Smaller cache files, less write amplification |
| Pre-build cache offline | No runtime write spikes |

---

## Hardware and OS (outside the engine)

These often matter more than software tuning:

1. **M.2 heatsink** — bare Gen4 drives can throttle within ~90 s sustained read; a basic heatsink often eliminates throttling entirely.
2. **Case airflow** — direct a fan over the M.2 slot; avoid sandwiching the drive between GPU and motherboard without airflow.
3. **Separate pack from OS disk** — put `runtime_pack/` on a dedicated NVMe so inference does not compete with OS paging.
4. **Leave 10–20% free space** — NVMe FTL works better with headroom; TRIM enabled (default on Windows 10+ / modern Linux).
5. **Power plan** — on Windows, Balanced/High Performance with PCIe link power saving off for consistent NVMe latency (see vendor guidance).
6. **Firmware** — keep NVMe firmware and chipset drivers current.

---

## Monitoring

```bash
# Snapshot SMART / health (best-effort per OS)
python -m rwkv_ssd.tools.ssd_health

# JSON for automation
python -m rwkv_ssd.tools.ssd_health --json

# Watch temperature during a run (poll every 5s)
python -m rwkv_ssd.tools.ssd_health --watch 5
```

On Linux with `smartmontools` installed, the tool also reports NVMe SMART
attributes (temperature, percentage used, media errors). On Windows it uses
`Get-PhysicalDisk` and, when permitted, storage reliability counters.

**During inference**, use `--metrics-csv layers.csv` and check `disk_cache_hits`
vs `read_ms` — high cache hits mean fewer physical pack reads.

---

## `RWKV_SSD_HEALTH=1` preset

Implemented in `rwkv_ssd/runtime/throughput_defaults.py` as
`apply_ssd_health_defaults`. It:

- Uses **partial hot3** residency (layers 0–2 pinned; fewer cold reads per token)
- Enables **stream layer cache** + decoupled provider cache
- Turns on **decode disk cache** with warm-at-load
- Sets **compressed cache writes** and **single cache writer**
- Keeps **mmap page cache** (`mmap_dontneed=False`)
- Does **not** enable strict F1 full-pack re-read path

Override with the usual tier env vars (`RWKV_SSD_TIER`, `RWKV_PROMOTE_FULL_Z`,
etc.) — explicit tier flags take precedence over `RWKV_SSD_HEALTH`.

---

## Read disturb (long-horizon note)

Repeated reads to the same NAND blocks (months of 24/7 inference) can
gradually increase bit error rates on consumer TLC/QLC. Mitigations:

- Rotate pack files across drives or re-pack periodically
- Prefer SLC/enterprise SSDs for always-on production
- Monitor `media_errors` / `Percentage Used` via `ssd_health`

For typical dev and batch inference workloads, this is negligible compared to
thermal throttling during a single long session.

---

## See also

- [`SSD_EXPLOITATION.md`](SSD_EXPLOITATION.md) — performance / multi-SSD sharding
- [`PRESETS.md`](PRESETS.md) — throughput tier decision tree
- [`simulations/benches/thermal_duty_cycle_bench.py`](../simulations/benches/thermal_duty_cycle_bench.py) — thermal duty-cycle model
- `python -m rwkv_ssd.tools.build_decode_cache --help` — offline cache builder
