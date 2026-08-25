# Full matrix analysis (ChatRWKV, matched config)

> **Doc note (June 2026):** This matrix predates **v0.6.7–v0.6.10** (decouple, promote, disk cache). `streaming_cache` numbers here do **not** include `promote_stream_cache_to_full_z` — re-run `bench/bench_full_matrix.py` for post-promote FP16 results. See [`docs/MILESTONE_STATUS.md`](../../../docs/MILESTONE_STATUS.md).

**Config (all modes):** `strategy=cpu bf16`, `io=mmap`, prefetch on, 32 tokens, median of 3 runs (+1 warmup).

**Modes:**
- **resident** — all weights in RAM (ceiling).
- **partial** — layers 0,1 (+ last on 0.1B) + embed/head resident; middle streams.
- **streaming_strict** — reload every layer every token.
- **streaming_cache** — bounded `max_z=2` (0.01B) / policy cap (0.1B).

Artifacts: `full_matrix.json`, `full_matrix.csv`.

---

## 0.01B (2 block layers)

| Pack | Mode | tok/s | % read | % staging | % compute | ms/tok read+staging+compute |
|------|------|-------|--------|-----------|-----------|---------------------------|
| FP16 | resident | **225** | — | — | — | (preload; layer timers ~0) |
| FP16 | partial | 139 | 39% | 0% | 62% | 2.6 + 0 + 4.2 |
| FP16 | streaming_strict | 52 | 7% | 45% | 49% | 0.8 + 5.4 + 5.9 |
| FP16 | streaming_cache | 133 | 5% | 11% | 84% | 0.1 + 0.2 + 1.4 |
| trinity_lut2 | resident | **198** | — | — | — | |
| trinity_lut2 | partial | 42 | **81%** | 0% | 19% | **18.9** + 0 + 4.4 |
| trinity_lut2 | streaming_strict | 26 | 1% | **80%** | 19% | 0.2 + **24.1** + 5.8 |
| trinity_lut2 | streaming_cache | 129 | 0% | 31% | 68% | 0 + 0.6 + 1.4 |
| trinity_lut2+shadow | resident | **213** | — | — | — | |
| trinity_lut2+shadow | partial | 43 | 82% | 0% | 18% | 18.8 + 0 + 4.3 |
| trinity_lut2+shadow | streaming_strict | **64** | 14% | 30% | 56% | 1.3 + **2.9** + 5.4 |
| trinity_lut2+shadow | streaming_cache | **135** | 3% | 6% | 91% | 0.05 + 0.09 + 1.4 |

**Disk:** FP16 72 MiB | trinity_lut2 **9 MiB** | trinity+shadow 80 MiB (9 + 71 shadow).

### 0.01B takeaways

1. **Trinity does shrink disk (~8×)** — but **staging (LUT decode) dominates** strict streaming (80% vs 45% for FP16).
2. **Shadow fixes strict streaming:** 26 → **64 tok/s** (~2.5× LUT), beats FP16 strict (52). Staging drops **24 → 3 ms/tok**.
3. **Resident ceiling** is similar: FP16 225 vs trinity 198 vs shadow 213 (one-time decode at load).
4. **Partial on 0.01B** ≈ both blocks resident; trinity still shows high **read** because middle tensors are tiny and timer attributes differ — not a win over resident.
5. **streaming_cache** makes LUT-only viable (129 tok/s) without shadow; shadow edges to **135 tok/s**.

---

## 0.1B (12 block layers)

| Pack | Mode | tok/s | % read | % staging | % compute | ms/tok read+staging+compute |
|------|------|-------|--------|-----------|-----------|---------------------------|
| FP16 | resident | **26.5** | — | — | — | |
| FP16 | partial | 3.3 | 22% | 53% | 25% | 39.7 + 98.7 + 46.4 |
| FP16 | streaming_strict | 2.4 | 18% | 61% | 21% | 42.8 + 145 + 50.8 |
| FP16 | streaming_cache | **5.4** | 4% | 70% | 26% | 5.4 + 98.4 + 37.4 |
| trinity_lut2 | resident | **27.4** | — | — | — | |
| trinity_lut2 | partial | 1.7 | 40% | 33% | 28% | 74.2 + 61.9 + 51.8 |
| trinity_lut2 | streaming_strict | **0.9** | 0% | **84%** | 16% | 0.7 + **303** + 57.8 |
| trinity_lut2 | streaming_cache | 1.4 | 0% | **83%** | 16% | 0.7 + **311** + 61.3 |
| trinity_lut2+shadow | resident | **27.4** | — | — | — | |
| trinity_lut2+shadow | partial | 2.5 | 43% | 33% | 24% | 114.7 + 86.9 + 62.7 |
| trinity_lut2+shadow | streaming_strict | **1.9** | 25% | 53% | 22% | 66 + **139** + 59.3 |
| trinity_lut2+shadow | streaming_cache | 2.4 | 24% | 52% | 24% | 65 + 140 + 64.1 |

**Disk:** FP16 384 MiB | trinity_lut2 **49 MiB** | trinity+shadow 432 MiB (49 + 382 shadow).

### 0.1B takeaways

1. **Smaller weights.bin does NOT mean fast streaming** — trinity strict read is **0.7 ms/tok** vs FP16 **43 ms/tok**, but **staging is 303 ms/tok** (LUT) vs 145 ms/tok (FP16 bf16 inject).
2. **Shadow helps LUT strict** (0.9 → 1.9 tok/s) but **reads full shadow layer** (~66 ms/tok read) — you traded decode for bandwidth.
3. **FP16 streaming_cache (5.4 tok/s)** still beats trinity+shadow cache (2.4) on 0.1B — CPU compute + inject path dominates.
4. **Partial** is the practical middle: FP16 3.3, shadow 2.5, LUT 1.7 tok/s — still heavy staging on streamed middle layers.
5. **Resident** ~27 tok/s for all packs (weights already decoded in RAM).

---

## Answer: “Trinity compressed a lot — shouldn’t SSD/IOPS be fine?”

**Yes for read bytes; no for end-to-end tok/s.**

| 0.1B strict streaming | weights on disk | ms/tok **read** | ms/tok **staging** |
|-----------------------|-----------------|-----------------|---------------------|
| FP16 | 384 MiB | 42.8 | 145 |
| trinity_lut2 | **49 MiB** | **0.7** | **303** |
| trinity+shadow | 432 MiB | 66 | 139 |

Trinity **does** cut sequential read (mmap-friendly). **Random IOPS are not the bottleneck** on fast NVMe here — **decode/inject (staging)** is. Layer timers sum to more than wall on some runs because of prefetch overlap accounting; use **ms/tok** columns for comparisons.

**chunk_reads:** FP16 strict ~68k vs trinity ~2k — fewer, larger logical reads with Trinity packs; still slow because of CPU decode.

---

## Recommended deployment profiles

| Goal | 0.01B | 0.1B |
|------|-------|------|
| Max tok/s | resident FP16 or shadow | resident FP16 |
| SSD + bounded RAM | streaming_cache + **shadow** | streaming_cache **FP16**; shadow optional |
| Smallest disk + acceptable speed | trinity_lut2 + cache | not trinity_lut2 alone |
| Strict streaming / no RAM cache | **trinity_lut2+shadow** | FP16 or shadow (still slow) |

---

## Reproduce

```powershell
$env:RWKV_JIT_ON='0'
python bench/bench_full_matrix.py --max-tokens 32 --samples 3 --warmup 1
```

Partial profiles: `bench/profiles/partial_0.01b.json`, `bench/profiles/partial_0.1b.json`.
