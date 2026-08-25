# Storage micro-benchmark (research artifact)

Measures **sequential read bandwidth** with `O_DIRECT` + `io_uring` on **Linux**. Useful for feasibility papers or a storage ceiling reference.

**Not the inference engine decode path.** Engine I/O today is **`mmap`** (default), **`pread`**, or **`threaded`** — explicit offset reads per tensor. This bench informs a future **Linux io_uring prefetch thread** (overlap read with compute), not a drop-in replacement for mmap on Windows.

## Build

```bash
cd storage_bench
cargo build --release
```

## Run

Create `raw_layer_fp16.bin` in the working directory, then:

```bash
./target/release/rwkv_storage_bench
```

Writes `io_uring_metrics.csv` in the current directory.

Compare engine-side reads against an operator-supplied pack:
`python bench/bench_io_ceiling.py --pack C:\prepared\model.pack --checkpoint C:\models\model.pth --json-out C:\prepared\io-ceiling.json`
