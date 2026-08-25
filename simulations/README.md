# Feasibility simulations (legacy)

Parameterized Python scripts from the earlier **thesis / SSD-native feasibility** work. They model compression, pipelining, thermal duty cycle, MTP, etc.

**These are not the inference engine.** The product code is [`rwkv_ssd/`](../rwkv_ssd/) and [`app/cli.py`](../app/cli.py).

## Layout

| Path | Contents |
|------|----------|
| `benches/` | `*_bench.py` scripts |
| `results/` | `*_metrics.csv` outputs (and new runs write here) |
| `run_suite.py` | Run all benches sequentially |

## Run

```bash
python simulations/run_suite.py --list
python simulations/run_suite.py
```

Requires CUDA for GPU-timed scripts (see `docs/thesis/TESTING_GUIDE.md`).

## Storage micro-benchmark

Real measured sequential read (Linux): [`storage_bench/`](../storage_bench/) (Rust `io_uring`), not `io_uring_simulation_bench.py`.
