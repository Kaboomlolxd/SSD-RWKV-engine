# RWKV SSD Engine opportunity atlas

Last audited: **July 12, 2026** against the current uncommitted worktree.

This is an opportunity map, not a claim that modeled multipliers are measured
product performance. Every proposal shares one scheduler, one storage/RAM
budget, and one forward-pass workload; mutually exclusive mechanisms are
compared, never stacked.

## Baseline and evidence labels

Already present: V1 streaming, cache-format selection, byte-bounded caches,
layer-affinity and striped packs, CMix selective reads, adaptive residency,
state snapshots and state-delta DAGs, pack overlays, prepared-artifact caching,
synthetic weight-stationary batching, and real dense ChatRWKV
layer-outer/session-inner decode.

The 0.1B CPU pack is normally page-cache resident and Python per-layer dispatch
dominates. It is valid for correctness and scheduler overhead, but not for an
SSD-bandwidth claim. Physical 2.9B/7B+, Linux/NVIDIA, multi-drive, power, and
thermal results remain explicitly gated.

- **shipped**: normal runtime path with tests.
- **prototype**: isolated implementation/tool and CPU tests.
- **open**: design only.
- **hardware-gated**: target hardware is required for meaningful evidence.
- **evidence-gated**: broader quality or workload data is required first.

## Prioritized shortlist

Score is `user impact × confidence of validation ÷ implementation effort`, on
relative 1–5 inputs. It ranks experiments, not roadmap commitments.

| Rank | Opportunity | Impact | Confidence | Effort | Score | Status / gate | Likely interface impact | First acceptance experiment |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | Persistent contextual autotuning | 4 | 5 | 2 | 10.0 | prototype | profile/diagnostic artifact | bounded probes across restarts; same quality, lower median/p95 latency |
| 2 | Session-length-aware promotion | 4 | 5 | 2 | 10.0 | **prototype** | future request hint after validation | short/medium/long replay; lower total latency at identical RAM cap |
| 3 | Amortization-aware promotion order | 4 | 5 | 2 | 10.0 | **prototype** | residency policy | LRU vs highest-stall vs benefit/byte on 2.9B traces |
| 4 | Cross-model chunk reuse and overlays | 4 | 4 | 3 | 5.3 | **prototype** | external store/profile artifacts | two related checkpoints; unique bytes, metadata and read overhead |
| 5 | Activation-aware mixed precision | 5 | 3 | 3 | 5.0 | prototype, evidence-gated | calibration/quality profile | KL, state drift, top-k, bytes, throughput vs uniform precision |
| 6 | Deadline/QoS runtime routing | 4 | 3 | 3 | 4.0 | prototype, quality-gated | per-request policy hint | missed deadlines and quality under one cache budget |
| 7 | Hierarchical state parking/DAG branching | 3 | 4 | 3 | 4.0 | **CPU prototype**; remote/GPU tiers open | state-tier policy | parity, bytes, restore p95 and recovery |
| 8 | Native layer slots for rwkv.cpp | 5 | 3 | 4 | 3.8 | ABI prototype; build-gated | provider/native slot ABI | greedy parity and RSS vs resident converted graph |
| 9 | Real weight-stationary multi-session batching | 5 | 3 | 4 | 3.8 | **CPU-tested ChatRWKV and rwkv.cpp implementations**; broader qualification open | backend batch/state API | parity, sweeps/session and tok/s |
| 10 | Topology-aware sharding | 5 | 2 | 4 | 2.5 | placement prototype; hardware-gated | diagnostic/placement plan | capped two-tier emulation, then physical NUMA/PCIe/multi-SSD |
| 11 | CUDA fused packed-block execution | 5 | 2 | 5 | 2.0 | open, Linux/NVIDIA-gated; LUT2 reserved | CUDA provider/kernel ABI | parity and kernel throughput vs decode→GEMM |
| 12 | GPU read/decode/H2D/compute pipeline | 5 | 2 | 5 | 2.0 | staging foundation; hardware-gated | GPU swapper/provider events | median/p95 stages, VRAM/RSS, GDS vs pinned bounce |

The first hardware-independent implementation slice deliberately does not edit
LUT2 codecs or kernels.

## Five genuinely additive experiments

### 1. Session-length-aware promotion — opt-in CPU prototype

`runtime/promotion_planner.py` selects a layer only when:

`promotion_ms < expected_remaining_tokens × staging_ms_saved_per_token`

Short sessions remain packed/streamed; longer sessions can progressively
promote. This differs from adaptive residency, which reacts to measured cache
cost. The ChatRWKV engine can now observe one request, create a plan for the
next request boundary, force-materialize selected dense layers, pin them in
`model.z`, discard duplicate provider copies, and verify the actual total-RAM
delta against `session_promotion_bytes`. A cap violation rolls the promotion
back. The two controllers cannot be enabled together.

The CLI/config surface is `--session-promotion`,
`--session-expected-tokens`, `--session-promotion-bytes`, and
`--session-promotion-policy`. Metrics report promoted layer IDs, actual bytes,
promotion wall time, estimated net benefit, and remaining-token estimate.

Acceptance: short, medium, and long replays; total wall time must fall without
exceeding the same resident-byte cap. Report prediction error and bad-promotion
cost.

### 2. Heterogeneous-drive placement — prototype, hardware-gated

`runtime/storage_placement.py` and `tools/plan_storage_placement.py` place layer
payloads using measured bandwidth, latency, access frequency, and capacity.
They emit a reviewed plan and modeled round-robin A/B; they do not move files.
This extends sharding without assuming equal drives.

Acceptance: bandwidth-capped fast/slow emulation, then two physical drive
classes. Report median/p95 stall, bytes/drive, queue depth, and topology. A
simulation retains `physical_hardware_validated=false`.

### 3. Cross-model content-addressed weight store — prototype

`PackChunkStore` already stores immutable SHA-256 chunks. It now reports
repository-wide logical, unique, and metadata bytes, references, deduplication
ratio, and a stable digest per profile. Next compare fixed-size with
tensor-aligned/content-defined chunks; fixed chunks can miss shifted tensors.

Acceptance: related base/fine-tuned checkpoints, unique download bytes,
metadata, cold/warm random reads, materialization hash, and load parity.
“Near-identical” means changed chunks, never fuzzy hashes.

The chunk repository now also supports direct `WeightStore` reads without
materializing `weights.bin`, mutation locking, profile checksum enforcement,
transactional profile removal, and reference-safe garbage collection.

### 4. Amortization-aware promotion order — prototype

The planner compares LRU, highest-stall-first, and:

`(future staging saved − promotion cost) / additional resident bytes`

All share one cap and reject unprofitable promotions. The greedy ratio is an
experiment baseline, not proof of optimal knapsack allocation.

Acceptance: 2.9B traces, identical cap, promotion bytes/cost, total latency,
and median/p95 token latency.

### 5. Portable session handoff — CPU-tested foundation

New engine snapshots carry exact manifest identity, backend identity, recurrent
state, last token, prompt, and greedy/temperature metadata. Restore rejects an
incompatible local pack/backend before installing state. Legacy snapshots stay
readable without identity enforcement.

Acceptance: CPU→GPU deterministic greedy continuation against a compatible
local pack, mismatch/corruption cases, transfer bytes, and resume latency. GPU
migration remains hardware-gated; sampler RNG state for stochastic bit parity
is still open.

### 6. Hierarchical recurrent-state parking — CPU prototype

`runtime/state_parking.py` provides an exact byte-bounded RAM LRU backed by
atomic portable snapshots on SSD. Every state is persisted before it can be
evicted, restores are deep copies, corrupt disk states fail soft, and synthetic,
RWKV-7 tensor-list, and rwkv.cpp-style external NumPy states round-trip exactly.
The current tier pair is RAM→local SSD; GPU memory and remote/object storage are
still open and require explicit transfer/latency policy.

The local OpenAI-compatible server accepts `session_id`, restores parked state,
continues without re-prefill, and parks the updated state. Admission control can
reject requests before execution when predicted streamed bytes exceed one
configured budget.

The local synthetic benchmark parked eight 256 KiB states under a 512 KiB RAM
cap with exact parity. It measured approximately 2.12 ms median write and
11.00 ms median disk restore, with 2,101,064 physical bytes including snapshot
metadata. These are Windows CPU/filesystem numbers, not a production p95 claim.

## Broader opportunity lanes

| Lane | Good/interesting ideas | Judgment |
|---|---|---|
| GPU execution/storage | native packed blocks, CUDA graphs, event-owned swapping, GDS/pinned alternatives, predictive VRAM windows, mixed placement | highest eventual impact; Linux/NVIDIA-gated. Build measurement before claims |
| Scheduling/runtime | unified stage cost, lifetime prediction, persistent tuning, safe probes, diagnosis, thermal/health inputs | strong core fit. Keep bounded planners inside one scheduler |
| Quantization/quality | activation scaling, mixed-bit allocation, residuals, paired layers, transforms, precision overlays, certificates | recurrent error makes this evidence-gated; sequence/state gates are mandatory |
| Pack/storage | content store, COW overlays, cold compression, topology maps, recovery shards, remote snapshots, profile compiler | reuse/compiler are additive; reliability follows hot-path proof |
| Memory/session | state parking, prefix reuse, migration, batching/coalescing, resident adapters, branches, time travel | distinctive RWKV advantage; migration/DAGs are lower risk than state prediction |
| Backend ecosystem | rwkv.cpp slots, Albatross, runtime probes, provider ABI, native execution plan | native CPU slots prove the contract before GPU |
| Serving/product | QoS, admission/fairness, observability, bundles, decision traces | useful after planner decisions are measurable and stable |
| Radical research | learned residency, token precision, request compilation, activation reuse, model filesystem, P2P, CXL, RWKV/GQA | interesting but later; most expand scope or require new hardware/data |

## Common validation contract

Every experiment records correctness (resident greedy parity, state drift,
snapshot round trips), performance (TTFT, warm median/p95 tok/s, stage times),
resources (RSS, provider/cache bytes, VRAM, power/temp), quality (RMSE, output
error, top-k, KL, state drift), reliability, and evidence scope.

No idea advances solely on simulated multipliers or a page-cache-resident 0.1B
pack. Public interface/schema changes wait until its experiment passes.

## Additional completed CPU infrastructure

- Storage diagnostics can populate heterogeneous drive tiers; a reviewed plan
  can be applied by a non-destructive repacker that preserves alignment and
  sidecar metadata, recomputes hashes, reloads the output manifest, and verifies
  every tensor byte against the source.
- Runtime diagnosis identifies the largest measured read/staging/compute
  component and recommends one next action. Optional JSONL traces explain each
  streamed, cached, prepared/resident, and promoted layer decision.
- Contextual autotuning profiles persist atomically and invalidate when the
  machine/runtime/backend/pack fingerprint changes.
- Backend capability probes separate declared support from loaded callable and
  native ABI support.
- Optional quality certificates bind declared gates and evidence scope to exact
  weight hashes. Failed gates, tampering, or changed weights stop manifest load.

## Software-interface gates discovered by implementation

- rwkv.cpp now has a real registered host-slot ABI with generation-checked
  native uploads and a passing DLL/model integration gate. It eliminates the
  temporary ctypes copy. The converted graph still owns resident weight
  buffers; reducing native graph RSS further requires a per-layer native graph.
- ChatRWKV now has real dense layer-outer/session-inner decode. On the local
  0.1B CPU A/B, two sessions matched independent greedy output exactly. A clean
  eight-token-per-session diagnostic measured 1.12 to 1.89 aggregate end-to-end
  tok/s (1.69x) and 1.37 to 2.75 aggregate decode-only tok/s (2.01x). Decode
  step latency stayed approximately 0.73 seconds: the implementation improves
  multi-session capacity, not single-session latency. Eight shared decode
  sweeps loaded 12 layers each (96 loads). A three-token repeated run was
  prefill-dominated and slightly slower in batch mode; short-request gains are
  therefore not claimed. Native rwkv.cpp now has the same layer-outer/session-
  inner schedule through its registered per-layer ABI. On the real 0.1B CPU
  pack, two warm sessions generating four tokens each measured 2.67–3.27
  aggregate tok/s versus 1.34–1.60 tok/s independently, with exact greedy
  output parity. This is capacity evidence, not a single-session latency or
  physical-SSD claim.

  Current benchmark observability has three known gaps: batch prefill time and
  `z_bytes` report zero, and batch layer rows include prefill while independent
  rows are decode-only. Use outer wall time for the end-to-end comparison and
  instrument the decode boundary for decode-only claims until those metrics are
  corrected. The 0.1B pack is page-cache resident, so this result is scheduler
  and staging evidence, not physical SSD scaling evidence.

## Reproduction

```powershell
# Deterministic planner A/B.
.venv-cpu\Scripts\python.exe bench\bench_additive_opportunities.py `
  --json-out bench\results\additive-opportunities.json

# Drive plan from a measured tier description.
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.plan_storage_placement `
  --pack .\runtime_pack --drives-json .\drives.json --json-out .\placement.json

# Cross-model repository savings.
.venv-cpu\Scripts\python.exe -m rwkv_ssd.tools.pack_overlay `
  --store .\weight-store stats

# Exact RAM→SSD recurrent-state parking.
.venv-cpu\Scripts\python.exe bench\bench_state_parking.py `
  --root .\.state-parking-bench --json-out bench\results\state-parking.json

# Real ChatRWKV independent-vs-shared-layer A/B. Use 8+ decode tokens and
# repeated samples for a throughput claim; very short runs are prefill-heavy.
$env:RWKV_PROMOTE_FULL_Z='0'
.venv-cpu\Scripts\python.exe bench\bench_chatrwkv_weight_stationary.py `
  --pack C:\prepared\reference-pack `
  --checkpoint C:\models\rwkv-model.pth `
  --max-tokens 8 --samples 3 `
  --json-out .\bench\results\chatrwkv-weight-stationary.json
```

Primary baselines: [MILESTONE_STATUS.md](MILESTONE_STATUS.md),
[SSD_STREAMING_FRONTIER.md](SSD_STREAMING_FRONTIER.md), [IDEAS.md](IDEAS.md),
[THROUGHPUT_PLAN.md](THROUGHPUT_PLAN.md), and [BACKENDS.md](BACKENDS.md).
