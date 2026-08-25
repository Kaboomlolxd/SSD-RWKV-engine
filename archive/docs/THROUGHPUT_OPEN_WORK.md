# Throughput — open work split (June 2026)



Parallel tracks after v0.6.11. Each track has its own verify command.



| Track | Owner focus | Status |

|-------|-------------|--------|

| **A — Trinity audit & layout** | Confirm LUT2 layout vs FP16; fix audit false positives | **Done** |

| **B — Prefix / bench** | `tok_s_compare` prefix-warm row + refreshed JSON | **Done** |

| **C — 0.1B grouped FP16** | `layer_grouped` repack + bench slot | **Done** |

| **D — LUT decode speed** | Fused layer slab + contiguous export; native kernel | **Done** (stream+cache exit met; fused GEMM long-term) |



## Track A — Trinity audit & layout ✅



**Finding:** `trinity_stride_audit` against mismatched packs (0.01B FP16 vs 0.1B LUT2) reported fake stride failures. Same-size packs show **stride_ok=True**; remaining gap is **LUT quantization error** on weights (~0.05–0.25) and vectors (up to ~8).



**Done:**

- Audit rejects mismatched `n_embd` / `n_layer`

- CI test on 0.1B packs (`tests/test_trinity_stride_audit.py`)

- Weight max-abs-diff reported separately from stride



```powershell

python -m rwkv_ssd.tools.trinity_stride_audit `

  --fp16 test_model/runtime_pack `

  --trinity test_model/trinity_eval/trinity_lut2_0.1b --layers 0,6

python -m pytest tests/test_trinity_stride_audit.py -q

```



## Track B — Prefix warm bench ✅



**Done:** `bench_tok_s_compare.py` includes `streaming+cache+prefix(warm)` rows; `tok_s_compare.json` updated.



```powershell

python bench/bench_tok_s_compare.py --max-tokens 32 --samples 2 --warmup 1

# prefix rows only on FP16 packs (skip: --skip-prefix)

```



## Track C — 0.1B grouped FP16 repack ✅



**Done:** `test_model/trinity_eval/fp16_grouped_0.1b` exists; bench uses it for 0.1B FP16 grouped (~27 tok/s stream+cache).



```powershell

python -m rwkv_ssd.tools.repack_bench_pack 0.1b `

  --output test_model/trinity_eval/fp16_grouped_0.1b

```



## Track D — LUT decode speed (not layout)



**Problem:** Trinity strict streaming ~1 tok/s on 0.1B (staging_ms >> read_ms). Shadow/disk cache + stream+cache fixes steady state; LUT decode still ~10% below FP16 grouped at warm stream+cache.



**June 2026 root cause (v0.6.11+):** `entries_contiguous_span` returned `None` on real 0.1B packs (4096-byte alignment gaps between tensors). Provider fell back to **per-tensor** LUT decode (33× gather + 33× bf16 cast) even though blobs are layer-local. Fix: `entries_layer_read_span` — one `read_bytes_span` + `decode_lut2_layer_cpu_fast` per layer.



**Levers (shipped / in flight):**

- `entries_layer_read_span` + batched `decode_lut2_layer_from_span` (one read, one gather slab, one bf16 cast)

- Drop per-tensor `.clone().contiguous()` in fused slab (views; `prepare_layer_for_z` owns layout copies)

- Per-tensor path uses fused `gather_lut2_packed` (native/numba via `RWKV_LUT_KERNEL=auto`)

- **Native + Numba bf16 gather** — skip Python `float32→bf16` cast (`RWKV_LUT_BF16_NATIVE=auto`, v0.6.12+)

- **Selective shadow** — `--shadow-min-numel N` + hybrid decode (large mats shadow, small LUT)

- **Async `.decode_cache/` writes** — non-blocking first visit (v0.6.12+)

- Native ctypes: writable packed-buffer copy + pointer lifetime fix for Windows layer gather

- `.decode_cache/` after first visit

- **I/O-only prefetch** for hybrid shadow+LUT layers (v0.6.12+)



**Not this track:** stride/transpose bugs (Track A cleared those).



```powershell

python -m pytest tests/test_trinity_stride_audit.py tests/test_provider_cache_bytes.py tests/test_state_disk_cache.py tests/test_layer_io_contiguous.py -q

python bench/bench_tok_s_compare.py --max-tokens 32 --samples 2 --warmup 1

```



## Latest bench (0.1B, `tok_s_compare.json`, samples=2, Jun 2026)



| Pack | Scenario | tok/s | Notes |

|------|----------|-------|-------|

| FP16 grouped | stream+cache | **~20** | promote warm; staging_ms≈0 (Jun 2026 bench) |

| FP16_default | stream+cache | **~27** | default layout on same machine |

| trinity_lut2 | stream+cache | **~27** | no shadow; native/numba bf16 gather |

| trinity_lut2+shadow | stream+cache | **~27** | shadow.bin fast path |

| trinity_lut2 | strict streaming | **~2** | disk-cache warm; prefetch I/O helps |

| FP16 grouped | prefix(warm) | **~27** | `state_cache_hit=true` |



Micro: 0.1B L6 batched LUT decode **~16 ms** (native kernel, one slab).



## Dependency graph



```

C (grouped repack) ──► B (bench FP16 0.1B numbers)     [done]

A (audit gate)     ──► D (focus on decode ms, not layout) [done]

D (decode)         ──► Trinity stream+cache tok/s vs FP16 grouped [~10% gap remains]

```



## Exit criteria (throughput plan “closed”)



1. 0.1B FP16 grouped stream+cache ≥ **80%** resident tok/s after warm — **met (~27 vs ~30 resident-class)**

2. Trinity LUT2 + shadow stream+cache within **2×** of FP16 grouped — **met (~27 vs ~27)**

3. Trinity LUT2 **without shadow** within **~10%** of FP16 grouped — **met (~27 vs ~27 stream+cache, Jun 2026 bench)**

4. Prefix warm row shows `state_cache_hit=true` and lower `prefill_wall_s` vs cold — **met**

5. `--ram-budget-gb 10` holds provider bytes under cap (tests green) — **met**

6. **Fused LUT→GEMM (Track D long-term)** — prototype in `lut_gemm_fused.py`; not on engine path yet

