# RWKV architecture research

This repository maintains one model family: RWKV. Research code is kept only
when it strengthens the RWKV SSD pipeline or provides a measurable path toward
lower latency, lower memory use, or better pack quality.

## Maintained research paths

- DeepEmbed support and sidecar loading for RWKV-7 variants.
- DSpark-style speculative primitives, gated behind trained artifacts and exact
  target verification.
- F1-F5 residency policies, provider caching, asynchronous reads, and state
  parking.
- Trinity and grouped-U8 pack codecs with explicit quality and size gates.
- Native `rwkv.cpp`, ChatRWKV reference, and external Albatross parity work.

Research utilities are not production features merely because they exist.
Promotion requires deterministic CPU tests, a documented artifact contract,
measured end-to-end results, and parity against a trusted RWKV reference.

## Hardware boundary

The local release gate is CPU-only. CUDA, GDS, physical multi-SSD scaling, and
Albatross execution require separate hardware qualification. Synthetic tests
can validate interfaces and failure behavior, but they cannot certify GPU
correctness or throughput.

## Verification

```powershell
python -m pytest -q
python -m pytest -q --override-ini "addopts="
python -m compileall -q rwkv_ssd app bench tests
python -m rwkv_ssd.tools.preflight --help
```

Large model checkpoints, generated packs, and benchmark payloads are local
artifacts and are intentionally excluded from Git.
