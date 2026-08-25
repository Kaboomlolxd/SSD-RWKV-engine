"""
SSD-resident microbenchmark for Engram-style memory serving paths.

What this script does
---------------------
It benchmarks filesystem-backed lookup variants using real file reads, so it can be used as a
practical prototype for comparing the serving-path behavior of:

1. engram_hash                     - original hashed cold path, K random reads per lookup
2. engram_nine                    - hot tier served in 1 read, cold tier still K reads
3. engram_nine_blockpack          - hot tier in 1 read, cold tier in 1 packed read
4. engram_nine_blockpack_bloom    - same as BlockPack, but skips guaranteed misses via Bloom

What it does NOT do
-------------------
It does not reproduce the full DeepSeek model. It isolates storage behavior. mHC is orthogonal
for address generation; if you want an end-to-end approximation, set --overlap-us to estimate
how much lookup time is hidden by preceding compute.

Usage example
-------------
python blockpack_benchmark.py \
  --workdir ./blockpack_data \
  --num-queries 20000 \
  --heads 2 \
  --head-dim 64 \
  --dtype-bytes 2 \
  --hot-keys 50000 \
  --cold-keys 500000 \
  --hot-prob 0.55 \
  --miss-prob 0.10 \
  --csv-out ./results.csv \
  --prepare

For a more SSD-like setup on a machine with ~8 GB RAM, make the data files materially larger
than RAM and consider enabling --thrash-before-run with a large --thrash-bytes value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import resource
import shutil
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

BLOCK_SIZE = 4096
UINT64 = struct.Struct("<Q")


def stable_u64(value: int, seed: int = 0) -> int:
    h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little", signed=False))
    h.update(UINT64.pack(value & 0xFFFFFFFFFFFFFFFF))
    return int.from_bytes(h.digest(), "little", signed=False)


class BloomFilter:
    def __init__(self, n_items: int, bits_per_item: int = 10, num_hashes: int = 7):
        self.n_items = max(1, n_items)
        self.bits_per_item = max(1, bits_per_item)
        self.num_hashes = max(1, num_hashes)
        self.num_bits = self.n_items * self.bits_per_item
        self.data = bytearray((self.num_bits + 7) // 8)

    def _positions(self, key: int) -> Iterable[int]:
        h1 = stable_u64(key, 17)
        h2 = stable_u64(key, 29) | 1
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, key: int) -> None:
        for pos in self._positions(key):
            self.data[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, key: int) -> bool:
        for pos in self._positions(key):
            if not (self.data[pos >> 3] & (1 << (pos & 7))):
                return False
        return True


@dataclass
class Query:
    key: int
    is_hot: bool
    is_miss: bool


@dataclass
class VariantStats:
    variant: str
    num_queries: int
    hot_prob: float
    miss_prob: float
    overlap_us: float
    mean_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    qps: float
    read_calls: int
    bytes_read: int
    checksum: int
    rss_mb: float
    end_to_end_mean_us: float
    end_to_end_p95_us: float


class FileRegion:
    def __init__(self, path: Path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = path.stat().st_size

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass

    def read(self, offset: int, nbytes: int) -> bytes:
        return os.pread(self.fd, nbytes, offset)


class DataLayout:
    def __init__(self, workdir: Path, heads: int, head_dim: int, dtype_bytes: int, hot_keys: int, cold_keys: int):
        self.workdir = workdir
        self.heads = heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.hot_keys = hot_keys
        self.cold_keys = cold_keys
        self.row_bytes_head = head_dim * dtype_bytes
        self.row_bytes_concat = heads * self.row_bytes_head
        self.packed_extent_bytes = max(BLOCK_SIZE, align_up(self.row_bytes_concat, BLOCK_SIZE))
        self.total_member_keys = hot_keys + cold_keys
        self.hot_file = FileRegion(workdir / "hot.bin")
        self.concat_file = FileRegion(workdir / "concat.bin")
        self.blockpack_cold_file = FileRegion(workdir / "blockpack_cold.bin")
        self.naive_head_files = [FileRegion(workdir / f"cold_head_{h}.bin") for h in range(heads)]

    def close(self) -> None:
        self.hot_file.close()
        self.concat_file.close()
        self.blockpack_cold_file.close()
        for f in self.naive_head_files:
            f.close()


class BenchHarness:
    def __init__(self, data: DataLayout, bloom: BloomFilter, overlap_us: float = 0.0):
        self.data = data
        self.bloom = bloom
        self.overlap_us = float(overlap_us)
        self.hot_set = set(range(data.hot_keys))
        self.member_limit = data.total_member_keys

    def _cold_index(self, key: int, head: int) -> int:
        cold_id = key - self.data.hot_keys
        if cold_id < 0:
            cold_id = stable_u64(key, head + 1) % self.data.cold_keys
        return cold_id

    def _consume(self, blob: bytes) -> int:
        if not blob:
            return 0
        step = max(1, len(blob) // 8)
        acc = 0
        for i in range(0, len(blob), step):
            acc ^= blob[i]
        return acc

    def lookup_engram_hash(self, q: Query) -> Tuple[int, int, int]:
        # Original-style hashed cold path: K separate random reads.
        checksum = 0
        read_calls = 0
        bytes_read = 0
        for h, region in enumerate(self.data.naive_head_files):
            idx = self._cold_index(q.key if not q.is_hot else q.key + self.data.hot_keys, h)
            off = idx * self.data.row_bytes_head
            blob = region.read(off, self.data.row_bytes_head)
            checksum ^= self._consume(blob)
            read_calls += 1
            bytes_read += len(blob)
        return checksum, read_calls, bytes_read

    def lookup_engram_nine(self, q: Query) -> Tuple[int, int, int]:
        checksum = 0
        read_calls = 0
        bytes_read = 0
        if q.is_hot and not q.is_miss:
            off = q.key * self.data.row_bytes_concat
            blob = self.data.hot_file.read(off, self.data.row_bytes_concat)
            checksum ^= self._consume(blob)
            read_calls += 1
            bytes_read += len(blob)
            return checksum, read_calls, bytes_read
        return self.lookup_engram_hash(q)

    def lookup_blockpack(self, q: Query, use_bloom: bool) -> Tuple[int, int, int]:
        checksum = 0
        read_calls = 0
        bytes_read = 0
        if q.is_hot and not q.is_miss:
            off = q.key * self.data.row_bytes_concat
            blob = self.data.hot_file.read(off, self.data.row_bytes_concat)
            checksum ^= self._consume(blob)
            return checksum, 1, len(blob)

        if use_bloom and (q.key not in self.bloom):
            return checksum, read_calls, bytes_read

        cold_id = self._cold_index(q.key if not q.is_hot else q.key + self.data.hot_keys, 0)
        off = cold_id * self.data.packed_extent_bytes
        blob = self.data.blockpack_cold_file.read(off, self.data.packed_extent_bytes)
        checksum ^= self._consume(blob)
        read_calls += 1
        bytes_read += len(blob)
        return checksum, read_calls, bytes_read

    def run_variant(self, name: str, queries: Sequence[Query], lookup: Callable[[Query], Tuple[int, int, int]]) -> VariantStats:
        latencies_us: List[float] = []
        end_to_end_latencies_us: List[float] = []
        checksum = 0
        read_calls = 0
        bytes_read = 0
        t0 = time.perf_counter()
        for q in queries:
            s = time.perf_counter_ns()
            csum, rc, br = lookup(q)
            e = time.perf_counter_ns()
            latency_us = (e - s) / 1000.0
            end_to_end_us = max(0.0, latency_us - self.overlap_us)
            latencies_us.append(latency_us)
            end_to_end_latencies_us.append(end_to_end_us)
            checksum ^= csum
            read_calls += rc
            bytes_read += br
        total_s = time.perf_counter() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return VariantStats(
            variant=name,
            num_queries=len(queries),
            hot_prob=sum(1 for q in queries if q.is_hot) / len(queries),
            miss_prob=sum(1 for q in queries if q.is_miss) / len(queries),
            overlap_us=self.overlap_us,
            mean_us=safe_mean(latencies_us),
            p50_us=percentile(latencies_us, 50),
            p95_us=percentile(latencies_us, 95),
            p99_us=percentile(latencies_us, 99),
            qps=(len(queries) / total_s) if total_s > 0 else float("inf"),
            read_calls=read_calls,
            bytes_read=bytes_read,
            checksum=checksum,
            rss_mb=rss_mb,
            end_to_end_mean_us=safe_mean(end_to_end_latencies_us),
            end_to_end_p95_us=percentile(end_to_end_latencies_us, 95),
        )


def safe_mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def write_random_file(path: Path, size_bytes: int, seed: int) -> None:
    rng = random.Random(seed)
    chunk = 1 << 20
    with open(path, "wb") as f:
        remaining = size_bytes
        while remaining > 0:
            take = min(chunk, remaining)
            buf = bytearray(rng.getrandbits(8) for _ in range(take))
            f.write(buf)
            remaining -= take


def prepare_data(workdir: Path, heads: int, head_dim: int, dtype_bytes: int, hot_keys: int, cold_keys: int, force: bool = False) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    row_bytes_head = head_dim * dtype_bytes
    row_bytes_concat = heads * row_bytes_head
    packed_extent_bytes = max(BLOCK_SIZE, align_up(row_bytes_concat, BLOCK_SIZE))

    targets = [
        (workdir / "hot.bin", hot_keys * row_bytes_concat, 11),
        (workdir / "concat.bin", (hot_keys + cold_keys) * row_bytes_concat, 13),
        (workdir / "blockpack_cold.bin", cold_keys * packed_extent_bytes, 17),
    ]
    for h in range(heads):
        targets.append((workdir / f"cold_head_{h}.bin", cold_keys * row_bytes_head, 19 + h))

    for path, size_bytes, seed in targets:
        if force or (not path.exists()) or (path.stat().st_size != size_bytes):
            print(f"Preparing {path.name}: {size_bytes / (1024**2):.2f} MiB", file=sys.stderr)
            write_random_file(path, size_bytes, seed)


def build_bloom(total_member_keys: int) -> BloomFilter:
    bloom = BloomFilter(total_member_keys)
    for key in range(total_member_keys):
        bloom.add(key)
    return bloom


def generate_queries(num_queries: int, hot_keys: int, cold_keys: int, hot_prob: float, miss_prob: float, seed: int) -> List[Query]:
    rng = random.Random(seed)
    queries: List[Query] = []
    total_member_keys = hot_keys + cold_keys
    for _ in range(num_queries):
        is_miss = rng.random() < miss_prob
        is_hot = (not is_miss) and (rng.random() < hot_prob)
        if is_miss:
            key = total_member_keys + rng.randrange(max(1, total_member_keys * 4))
        elif is_hot:
            key = rng.randrange(hot_keys)
        else:
            key = hot_keys + rng.randrange(cold_keys)
        queries.append(Query(key=key, is_hot=is_hot, is_miss=is_miss))
    return queries


def maybe_thrash_cache(workdir: Path, thrash_bytes: int) -> None:
    if thrash_bytes <= 0:
        return
    path = workdir / "thrash.bin"
    if (not path.exists()) or (path.stat().st_size != thrash_bytes):
        print(f"Preparing thrash file: {thrash_bytes / (1024**2):.2f} MiB", file=sys.stderr)
        write_random_file(path, thrash_bytes, 101)
    fd = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        chunk = BLOCK_SIZE
        while offset < thrash_bytes:
            _ = os.pread(fd, chunk, offset)
            offset += chunk
    finally:
        os.close(fd)


def write_csv(path: Path, rows: Sequence[VariantStats]) -> None:
    fieldnames = list(VariantStats.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("./blockpack_data"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2 for fp16-like rows, 1 for int8-like rows")
    ap.add_argument("--hot-keys", type=int, default=50_000)
    ap.add_argument("--cold-keys", type=int, default=500_000)
    ap.add_argument("--num-queries", type=int, default=20_000)
    ap.add_argument("--hot-prob", type=float, default=0.55)
    ap.add_argument("--miss-prob", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--csv-out", type=Path, default=Path("./blockpack_results.csv"))
    ap.add_argument("--overlap-us", type=float, default=0.0, help="Approximate hidden time from preceding compute")
    ap.add_argument("--thrash-before-run", action="store_true")
    ap.add_argument("--thrash-bytes", type=int, default=0, help="Read a large file before each variant to disturb cache")
    args = ap.parse_args()

    if args.prepare:
        prepare_data(
            workdir=args.workdir,
            heads=args.heads,
            head_dim=args.head_dim,
            dtype_bytes=args.dtype_bytes,
            hot_keys=args.hot_keys,
            cold_keys=args.cold_keys,
            force=args.force,
        )

    data = DataLayout(
        workdir=args.workdir,
        heads=args.heads,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
        hot_keys=args.hot_keys,
        cold_keys=args.cold_keys,
    )

    try:
        bloom = build_bloom(args.hot_keys + args.cold_keys)
        queries = generate_queries(
            num_queries=args.num_queries,
            hot_keys=args.hot_keys,
            cold_keys=args.cold_keys,
            hot_prob=args.hot_prob,
            miss_prob=args.miss_prob,
            seed=args.seed,
        )
        harness = BenchHarness(data=data, bloom=bloom, overlap_us=args.overlap_us)

        variants: List[Tuple[str, Callable[[Query], Tuple[int, int, int]]]] = [
            ("engram_hash", harness.lookup_engram_hash),
            ("engram_nine", harness.lookup_engram_nine),
            ("engram_nine_blockpack", lambda q: harness.lookup_blockpack(q, use_bloom=False)),
            ("engram_nine_blockpack_bloom", lambda q: harness.lookup_blockpack(q, use_bloom=True)),
        ]

        rows: List[VariantStats] = []
        for name, fn in variants:
            if args.thrash_before_run and args.thrash_bytes > 0:
                maybe_thrash_cache(args.workdir, args.thrash_bytes)
            stats = harness.run_variant(name, queries, fn)
            rows.append(stats)
            print(
                f"{stats.variant:28s} mean={stats.mean_us:8.3f}us p95={stats.p95_us:8.3f}us "
                f"reads={stats.read_calls:8d} bytes={stats.bytes_read / (1024**2):8.2f}MiB "
                f"e2e_mean={stats.end_to_end_mean_us:8.3f}us",
                file=sys.stderr,
            )

        write_csv(args.csv_out, rows)
        print(args.csv_out)
        return 0
    finally:
        data.close()


if __name__ == "__main__":
    raise SystemExit(main())
