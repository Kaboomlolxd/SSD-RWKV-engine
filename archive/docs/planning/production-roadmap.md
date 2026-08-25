# RWKV SSD Runtime Plan — Production-Ready Version

> **Repo note:** Product code is in `rwkv_ssd/` + `app/`. Start with [`ENGINE.md`](../../ENGINE.md). Simulations moved to `simulations/`; not part of engine CI.

> **Engine status (June 2026):** V1 streaming golden is shipped. Throughput phase extended **v0.6.7–v0.6.10** (decouple, promote, disk cache, prefix cache, RAM budget). FP16 streaming matches resident after warm; Trinity LUT2 and byte-based RAM caps are open. Operational “production ready” gates below are **not** met yet — this remains a research-grade prototype with measurable benches. Live status: [`MILESTONE_STATUS.md`](../MILESTONE_STATUS.md).

## Headline Recommendation

**Build the main engine on ChatRWKV + PyTorch using RWKV-7 as the initial base model, use rwkv.cpp as a quantized sidecar / comparison backend, and treat production-readiness as a parallel workstream from the start rather than a final cleanup phase.**

Then add RWKV-8 features only after the RWKV-7 engine is stable, observable, testable, and deployable.

This is the best path if the end goal is not merely a research demo, but a **production-ready RWKV inference engine**.

That means the roadmap is no longer only about model execution. It must cover five tracks in parallel:

1. **Core inference engine** — recurrent correctness, streaming runtime, quantization, scheduling.
2. **Serving plane** — API, batching, admission control, cancellation, timeouts, state reuse.
3. **Operability** — metrics, tracing, logging, health checks, runbooks, rollback, canaries.
4. **Release engineering** — packaging, CI, compatibility matrix, reproducible builds, upgrade path.
5. **Security and safety** — artifact integrity, config validation, isolation, input limits, resource controls.

The plan should end with a system that is not just "working," but is actually ready to be operated by other people.

---

## Definition of "Production Ready"

For this plan, **production ready** means all of the following are true:

### Functional correctness

- recurrent decode is correct,
- streamed execution matches resident execution within tolerance,
- quantized modes are validated against reference modes,
- failure modes are explicit rather than silent,
- APIs are stable enough for external users.

### Reliability

- the engine can run for long sessions without leaks or state corruption,
- cancellation and timeout paths are safe,
- partial failures degrade gracefully,
- startup and shutdown are deterministic,
- corrupted packs and incompatible artifacts fail early and clearly.

### Performance predictability

- performance is benchmarked and versioned,
- regressions are detected automatically,
- resident / partial / full streaming modes are measurable,
- throughput and latency are explained by metrics,
- scheduler behavior is tunable and observable.

### Operability

- the engine exposes metrics, logs, and traces,
- on-call operators can tell what is wrong,
- health endpoints exist,
- there are runbooks for common failure classes,
- release rollback is supported.

### Security and control

- model artifacts are checksummed,
- configs are validated,
- request limits are enforced,
- resource exhaustion paths are guarded,
- unsafe file and runtime assumptions are blocked by default.

### Distribution and maintenance

- the engine is packaged cleanly,
- CI covers correctness, performance smoke tests, and packaging,
- versioning is documented,
- supported deployment modes are documented,
- another engineer can build, run, benchmark, and deploy it without reverse engineering the repo.

If the roadmap does not end with all of those, it ends with a strong prototype, not a production-ready engine.

---

## Product Goal

Deliver a **production-ready RWKV inference engine** that:

- runs RWKV-7 reliably in recurrent mode,
- supports resident and SSD-streamed execution modes,
- exposes a real service interface,
- supports observability and operational control,
- has a defined artifact format and deployment story,
- can later absorb RWKV-8 features without destabilizing the core runtime.

The engine should be useful in three modes:

1. **Developer mode** — local debugging and benchmarking.
2. **Research mode** — controlled experiments for streaming, quantization, and future RWKV-8 features.
3. **Production mode** — stable serving with monitored latency, controlled degradation, and reproducible deployment.

---

## Product Boundaries

### In scope

- RWKV recurrent inference,
- SSD streaming for large model data,
- quantized reference / sidecar backend,
- API server,
- state reuse,
- structured observability,
- packaging and deployment,
- configuration system,
- compatibility rules,
- release gates,
- rollback and failure handling,
- production docs and runbooks.

### Out of scope for initial GA

- distributed multi-node inference,
- training,
- fine-tuning service,
- unbounded multi-tenant scheduling,
- zero-copy GDS-only architectures,
- custom CUDA kernel research as a prerequisite,
- custom compression research as the mainline path,
- experimental RWKV-8 features in the initial GA.

These can arrive later, but not at the cost of shipping a stable core product.

---

## Strategic Architecture Decision

### Base line: RWKV-7 first

RWKV-7 should remain the first production target.

Reason:

- the recurrent runtime can be proven with fewer moving parts,
- the serving and observability story can be stabilized first,
- the storage/runtime path can be validated before introducing new token-conditioned memory systems,
- it creates a production baseline that later RWKV-8 work can extend instead of destabilize.

### Base codebase: ChatRWKV first

Use **ChatRWKV as the editable foundation**.

Reason:

- fastest path to a real recurrent engine,
- easiest place to iterate on step execution,
- easiest way to keep a reference runtime close at hand,
- cleanest route for a solo or small-team build.

### Sidecar backend: rwkv.cpp

Use **rwkv.cpp** as:

- a quantized comparison backend,
- a performance sanity-check backend,
- a fallback or secondary execution path,
- a possible later integration target.

Do not make it the primary implementation foundation.

---

## The Core Principle for the Whole Roadmap

The engine must be built as a **single schedulable system**, not as a pile of independent speedups.

So the product should be modeled as one integrated runtime with:

- one request lifecycle,
- one recurrent state model,
- one artifact format,
- one scheduling policy per deployment mode,
- one observability layer,
- one release process.

This matters because production systems fail at boundaries. The plan must explicitly manage those boundaries.

---

## Product Architecture

## 1. Control Plane

Responsible for:

- loading engine configuration,
- validating artifact compatibility,
- deciding runtime mode,
- enforcing request limits,
- routing to model backend,
- controlling lifecycle and upgrades.

Key responsibilities:

- configuration schema validation,
- artifact manifest validation,
- model compatibility checks,
- startup self-tests,
- backend selection,
- rollout and rollback hooks.

## 2. Data Plane

Responsible for the actual inference path.

Subcomponents:

- tokenizer,
- recurrent state manager,
- model executor,
- tensor I/O backend,
- pinned staging manager,
- scheduler,
- sampling module,
- optional feature providers.

## 3. Serving Plane

Responsible for request handling.

Subcomponents:

- HTTP / gRPC API,
- request queue,
- admission control,
- cancellation,
- timeout enforcement,
- batching / micro-batching,
- state snapshot cache,
- response streaming.

## 4. Observability Plane

Responsible for operational visibility.

Subcomponents:

- structured logs,
- metrics exporter,
- trace hooks,
- health endpoints,
- debug snapshot tooling,
- benchmark reporters.

## 5. Artifact Plane

Responsible for model distribution and integrity.

Subcomponents:

- model packer,
- manifest format,
- checksum generation,
- artifact verifier,
- compatibility/version metadata,
- migration tooling.

---

## Required Engine Interfaces

These interfaces should exist early and remain stable.

### Recurrent model interface

```python
class RecurrentModel:
    def prefill(self, tokens, state=None):
        ...
    def step(self, token_id, state):
        ...
    def clone_state(self, state):
        ...
    def serialize_state(self, state):
        ...
    def deserialize_state(self, blob):
        ...
```

### Tensor I/O interface

```python
class TensorIO:
    def open(self):
        ...
    def read_tensor(self, tensor_name):
        ...
    def prefetch_tensor(self, tensor_name):
        ...
    def close(self):
        ...
```

### Scheduler interface

```python
class RuntimeScheduler:
    def begin_request(self, request_ctx):
        ...
    def begin_step(self, token_id, state):
        ...
    def get_layer_payload(self, layer_id):
        ...
    def end_step(self, result):
        ...
```

### Feature provider interface

```python
class FeatureProvider:
    def begin_step(self, token_id, state):
        ...
    def get_layer_feature(self, layer_id):
        ...
    def end_step(self):
        ...
```

### Serving interface

```python
class InferenceService:
    def generate(self, request):
        ...
    def stream_generate(self, request):
        ...
    def cancel(self, request_id):
        ...
    def health(self):
        ...
```

These interfaces matter because production systems survive through stable seams.

---

## Runtime Modes

The engine should explicitly support these modes:

1. **Resident mode**
   - everything feasible stays resident,
   - reference correctness and lowest operational complexity.

2. **Partial streaming mode**
   - hot tensors resident,
   - heavy middle tensors streamed,
   - practical first production streaming target.

3. **Full streaming mode**
   - most large weights streamed,
   - highest risk and highest memory savings,
   - production supported only after full qualification.

4. **Quantized mode**
   - uses quantized artifacts or sidecar backend,
   - must expose quality and performance trade-offs explicitly.

5. **Experimental feature mode**
   - DeepEmbed, ROSA, or other future features,
   - gated behind feature flags,
   - explicitly separated from production support commitments.

---

## Release Policy

The project should have three release channels:

### Stable

- production supported,
- strict compatibility rules,
- rollback guaranteed,
- performance baselines enforced.

### Preview

- near-production features,
- wider logging,
- opt-in deployment,
- weaker compatibility guarantees.

### Experimental

- RWKV-8 features,
- new schedulers,
- new storage paths,
- no production guarantee.

This is necessary because the engine will otherwise be torn between research velocity and operational stability.

---

# Roadmap

# Phase 0 — Product Spec, Contracts, and Success Gates

## Goal

Freeze the product shape before deeper implementation.

## Deliverables

- architecture spec,
- supported deployment modes,
- artifact format spec,
- config schema,
- request/response API schema,
- compatibility policy,
- benchmark methodology,
- release criteria document,
- risk register.

## Must define now

- what counts as correctness,
- what counts as supported mode,
- what metrics are mandatory,
- what deployment assumptions are allowed,
- what rollback means,
- what production SLA / SLO targets are initially reasonable.

## Exit criteria

- all core interfaces frozen at v0 shape,
- artifact manifest defined,
- runtime modes defined,
- release checklist drafted,
- no critical ambiguity about product scope.

---

# Phase 1 — RWKV-7 Recurrent Correctness Core

## Goal

Build the correct recurrent RWKV-7 engine on ChatRWKV.

## Work items

- **audit** [rwkv_lightning](https://github.com/RWKV-Vibe/rwkv_lightning), [web-rwkv](https://github.com/cryscan/web-rwkv), [RWKV-Infer](https://github.com/OpenMOSE/RWKV-Infer), [rwkv-fla](https://github.com/fla-org/flash-linear-attention) — decide per feature: **integrate, borrow, or re-implement** (default: borrow),
- implement real `prefill()` and `step()` execution,
- remove GPT-style full-context recompute from true decode path,
- implement state snapshot / clone / restore,
- define deterministic sampling hooks,
- build correctness harness against reference path,
- add crash-safe state lifecycle.

## Production hardening in parallel

- define structured internal error model,
- add assertions that fail early on invalid states,
- add unit tests for state invariants,
- add fuzz tests for tokenizer / input edge cases,
- add leak and long-run soak tests.

## Exit criteria

- recurrent execution is correct,
- state lifecycle is deterministic,
- no hidden full-context recomputation remains in serving path,
- test suite covers core state transitions,
- engine can survive long generation sessions without corruption.

---

# Phase 2 — Artifact Format and Streaming Runtime

## Goal

Add SSD-capable artifact packaging and streaming runtime.

## Work items

- implement packer from source checkpoints,
- define `weights.bin + manifest.json + meta.json`,
- implement `mmap` backend,
- implement optional `pread` backend,
- implement resident-vs-streamed policy,
- implement pinned host staging,
- implement H2D copy stream,
- implement basic ping-pong scheduler,
- add artifact checksum verification.

## Production hardening in parallel

- add artifact compatibility versioning,
- add corruption detection and descriptive load errors,
- add startup verification mode,
- add deterministic fallback to resident mode when streaming unsupported,
- add pack validation tool.

## Exit criteria

- packed artifacts load reproducibly,
- streaming path matches resident path within tolerance,
- corrupted or incompatible artifacts fail fast,
- metrics can explain the streaming path,
- resident / partial / full streaming modes are selectable and validated.

---

# Phase 3 — Scheduler, Throughput Mechanisms, and Performance Discipline

## Goal

Make the runtime faster and more controllable without sacrificing correctness — under **one scheduler and one SSD bandwidth budget** (thesis §2.5; [`../THROUGHPUT_PLAN.md`](../THROUGHPUT_PLAN.md)).

## Work items — engine core

- extend layer prefetch (read-ahead) with `lookahead_ms` in metrics CSV,
- add hot-layer pinning tied to residency profiles,
- add automated performance baseline capture (`bench_generate`, `bench_io`, `bench_chatrwkv`),
- state-reuse / prefix cache for repeated prompts (M2.5 path).

## Work items — P2.a latency hiders ([`IDEAS.md`](../IDEAS.md))

- within-layer micro-pipelining (K=8–16) after chunked `weights.bin` reader exists,
- gate-based prefetch (SSM gate signal) — default over blind speculative SSD prefetch,
- optional SSM speculation cache for recurrent models.

## Work items — P2.b bandwidth (mutually exclusive codec)

- quantized comparison **ladder** (rwkv.cpp Q4_1/Q5_K, web-rwkv NF4/INT8, rwkv_lightning FP8/INT8/FP6/FP5/HQQ4, RWKV-Infer HQQ4),
- **Compression Trinity** afternoon experiment — promote to pack path only if gate passes,
- optional NAND channel-aligned pack layout (bench_io delta).

## Work items — P2.c topology

- mmap / pread / threaded engine I/O (ayafileio rejected v0.5.4),
- Linux: madvise + io_uring prefetch from `storage_bench/` learnings,
- second NVMe / striped read benchmarks,
- NUMA-aware reader threads,
- heterogeneous chunk schedule **or** uniform-K — document A/B, do not stack gains.

## Work items — P2.e (conditional)

- recurrent MTP / n-gram weight cache — only for prefill-dominated workloads; separate reporting from decode tok/s.

## Production hardening in parallel

- regression benchmarks in CI,
- defined latency budgets per mode,
- defined memory budgets per mode,
- defined fallback rules when quantized mode fails,
- observability for scheduler decisions,
- **no-stacking** review on any performance PR (see [`../THROUGHPUT_PLAN.md`](../THROUGHPUT_PLAN.md)).

## Exit criteria

- runtime regressions are automatically detectable,
- quantized mode is clearly labeled and benchmarked; **one** primary codec chosen for streaming packs,
- state reuse is correct and safe,
- scheduler behavior is observable (`read_ms`, `h2d_ms`, `compute_ms`, prefetch overlaps),
- mode-specific budgets are documented,
- no release note multiplies independent simulation speedups.

---

# Phase 4 — Serving Plane and Production API

## Goal

Turn the engine into a real service.

## Work items

- implement HTTP and/or gRPC API,
- add request schema validation,
- add streaming token responses,
- add cancellation,
- add timeouts,
- add admission control,
- add queueing policy,
- add micro-batching where safe,
- add per-request context objects,
- add request IDs and trace correlation.

## Production hardening in parallel

- enforce input size limits,
- enforce max generation limits,
- protect against pathological prompts,
- add overload responses,
- add graceful shutdown behavior,
- add readiness and liveness endpoints,
- add configuration for concurrency caps.

## Exit criteria

- service can run continuously,
- cancellation works,
- timeouts work,
- overload behavior is controlled,
- request boundaries are clean,
- service API is documented and versioned.

---

# Phase 5 — Observability, Operations, and Failure Management

## Goal

Make the service operable by someone other than the original author.

## Work items

- add structured logs,
- add metrics exporter,
- add tracing hooks,
- add per-stage latency metrics,
- add model-load metrics,
- add artifact integrity metrics,
- add scheduler decision metrics,
- add state-cache hit metrics,
- add health summaries.

## Minimum dashboards / runbooks

Need runbooks for:

- slow startup,
- artifact verification failure,
- streaming backend failure,
- latency regression,
- GPU bubble spike,
- host memory pressure,
- SSD saturation / abnormal read latency,
- request queue overload,
- quantized mode mismatch.

## Exit criteria

- operator can identify bottleneck class from metrics,
- logs are structured and searchable,
- alerts can be defined from exported metrics,
- health endpoints reflect real readiness,
- there is a documented procedure for common incidents.

---

# Phase 6 — Packaging, Deployment, and Release Engineering

## Goal

Make the engine shippable and reproducible.

## Work items

- containerize supported deployment path,
- provide local binary / package install path,
- define supported OS / CUDA / driver matrix,
- add CI for tests + package build + smoke benchmark,
- add reproducible model pack build process,
- accept **Hugging Face repo IDs** in `pack_runtime` / `meta.json` (modern input path),
- add versioned manifests,
- add upgrade / migration tooling,
- add rollback tooling,
- add sample deployment configs.

## Production hardening in parallel

- sign or checksum artifacts,
- freeze dependency ranges for stable releases,
- test startup on clean environment,
- test downgrade / rollback path,
- publish release notes format,
- define deprecation policy.

## Exit criteria

- another engineer can deploy from docs,
- CI can produce artifacts reproducibly,
- rollback is tested,
- compatibility matrix is documented,
- release notes and migration docs exist.

---

# Phase 7 — GA Readiness and Production Qualification

## Goal

Qualify the RWKV-7 engine for general production release.

## Qualification matrix

The release must pass all of these:

### Correctness

- golden-output tests,
- resident vs streamed equivalence tests,
- quantized vs reference acceptance bounds,
- restart and restore correctness,
- state snapshot correctness.

### Reliability

- 24h / 72h soak tests,
- repeated load/unload cycles,
- cancellation storm tests,
- queue overload tests,
- corrupted artifact tests,
- degraded-storage tests.

### Performance

- versioned baseline benchmarks,
- cold-start latency benchmark,
- warm-start latency benchmark,
- per-token latency distribution,
- prefill / decode split,
- sustained throughput benchmark,
- memory footprint benchmark.

### Operability

- dashboard coverage,
- alert coverage,
- runbook review,
- logging verification,
- canary deployment validation.

### Security / safety

- input limit enforcement,
- config validation coverage,
- artifact integrity checks,
- no unsafe default deployment settings,
- documented resource isolation assumptions.

### Documentation

- operator guide,
- deployment guide,
- performance tuning guide,
- artifact format guide,
- troubleshooting guide,
- release / rollback guide.

## Exit criteria

- stable release branch created,
- documented supported modes,
- documented unsupported modes,
- production checklist signed off,
- rollback plan verified,
- initial SLOs published.

At this point the RWKV-7 engine is legitimately production ready.

---

# Phase 8 — RWKV-8 DeepEmbed as Preview Feature

## Goal

Track only — **not committed.**

## Status (2026)

Announced **May 2025**; still **preview** — no production RWKV-8 DeepEmbed models broadly available. Do **not** block GA on this.

## Rule

If revisited post-GA: **preview / experimental provider** only, not a rewrite of the SSD streaming core.

## Requirements

- provider abstraction already exists,
- resident version first,
- mapped version second,
- token-major bundle layout,
- explicit feature flag,
- separate benchmark suite,
- separate failure handling,
- clear support statement.

## Exit criteria

- preview feature does not regress stable core,
- observability covers DeepEmbed-specific overhead,
- documentation clearly labels support level.

---

# Phase 9 — RWKV-8 ROSA as Experimental / Preview Feature

## Goal

Track only — **not committed.**

## Status (2026)

Announced **Oct 2025**; still **experimental**. **ROSA-Tuning** paper (Feb 2026) validates long-context retrieval ideas on **Qwen3-class** models — **orthogonal** to SSD layer streaming (changes the **model**, not the weight I/O path).

## Rule

If revisited: experimental side-channel only; must not destabilize M3 streaming + resident parity.

## Requirements

- clear provider boundary,
- CPU reference path first,
- native optimized path later,
- explicit feature flags,
- separate correctness harness,
- separate latency accounting,
- no degradation of stable modes.

## Exit criteria

- ROSA can be enabled without destabilizing GA runtime,
- metrics expose ROSA overhead,
- support level is explicit,
- rollback is trivial.

---

## Cross-Cutting Workstreams

These workstreams run across all phases.

## A. Testing

Test categories:

- unit tests,
- property tests,
- golden-output tests,
- artifact validation tests,
- streaming equivalence tests,
- quantized acceptance tests,
- load tests,
- soak tests,
- fault-injection tests,
- packaging smoke tests.

## B. Performance governance

Need:

- benchmark corpus,
- stable benchmark hardware definition,
- regression thresholds,
- perf dashboards,
- per-release benchmark summary.

## C. Config governance

Need:

- schema validation,
- default profiles,
- safe production defaults,
- explicit experimental flags,
- deprecated setting handling.

## D. Documentation

Need:

- architecture doc,
- operator doc,
- developer doc,
- artifact doc,
- deployment doc,
- tuning guide,
- FAQ and troubleshooting.

## E. Incident readiness

Need:

- issue severity model,
- release rollback procedure,
- incident template,
- log bundle export command,
- reproducibility checklist.

## F. Library currency

Older citations in planning docs (e.g. **pytorch-lightning 1.9.5**, **torch 1.13.1+cu117**) are **out of date**. V1 compute targets (**rwkv_lightning**, **rwkv-fla**) expect **torch ≥2.5 / 2.7** and modern CUDA/ROCm stacks.

Maintain a separate **pinned dependency matrix** doc before GA — do not mix legacy thesis pins with engine pins.

---

## Recommended Repo Layout

```text
rwkv_ssd/
  tools/
    convert_from_chatrwkv.py
    pack_runtime.py
    verify_pack.py
    repack_rwkvcpp_quant.py
    migrate_manifest.py

  runtime/
    config.py
    errors.py
    manifest.py
    artifact_verify.py
    io_mmap.py
    io_pread.py
    staging.py
    scheduler.py
    residency.py
    state.py
    executor_rwkv.py
    sampler.py
    metrics.py
    tracing.py
    health.py

  service/
    api_http.py
    api_grpc.py
    request_models.py
    queueing.py
    admission.py
    batching.py
    cancellation.py
    lifecycle.py

  backends/
    chatrwkv_ref/
    rwkvcpp_ref/

  features/
    deepembed_provider.py
    rosa_runtime.py
    rosa_reference.py

  tests/
    unit/
    integration/
    performance/
    soak/
    fault_injection/

  bench/
    bench_correctness.py
    bench_io.py
    bench_h2d.py
    bench_tokps.py
    bench_overlap.py
    bench_startup.py
    bench_residency_modes.py

  deploy/
    docker/
    k8s/
    systemd/
    example_configs/

  docs/
    architecture.md
    artifact_format.md
    deployment.md
    tuning.md
    operations.md
    rollback.md
    troubleshooting.md
```

---

## Production SLO Starter Set

Initial production targets should be modest and explicit.

Example starter SLOs:

- successful request completion rate,
- p95 startup time,
- p95 first-token latency by mode,
- p95 per-token latency by mode,
- crash-free uptime window,
- artifact load failure rate,
- rollback success rate.

These do not need to be ambitious at first. They need to be real.

---

## Major Risks and Mitigations

### Risk 1: research creep

Mitigation:
- stable/preview/experimental channel split,
- GA scope frozen before RWKV-8 features.

### Risk 2: performance claims become unexplainable

Mitigation:
- benchmark governance,
- always-on metrics,
- regression tracking from Phase 3 onward.

### Risk 3: streaming path is correct but fragile

Mitigation:
- artifact verification,
- startup self-checks,
- resident-mode fallback,
- fault-injection testing.

### Risk 4: service instability under load

Mitigation:
- admission control,
- queue caps,
- cancellation and timeout support,
- overload behavior defined explicitly.

### Risk 5: experimental features destabilize stable runtime

Mitigation:
- feature flags,
- provider boundaries,
- release channels,
- trivial rollback.

---

## Final Recommendation

The strongest practical plan is:

1. **Use ChatRWKV as the base codebase.**
2. **Make RWKV-7 the first production target.**
3. **Treat production readiness as a parallel workstream from day one.**
4. **Ship a stable RWKV-7 engine before promoting RWKV-8 features.**
5. **Keep rwkv.cpp as a sidecar / comparison backend until the core runtime is mature.**
6. **Gate DeepEmbed behind preview mode.**
7. **Gate ROSA behind experimental or preview mode.**

That gives you a roadmap that is still ambitious, but now ends in a system that can actually be deployed, monitored, upgraded, rolled back, and operated.

---

## Short Version

The previous plan ended at **"real open-source demo."**

This revised plan ends at **"production-ready RWKV-7 engine with stable serving, observability, artifact integrity, release engineering, and rollback, plus RWKV-8 features added only behind preview/experimental gates."**

---

## Exploratory additions (no commitment — post-GA or never)

| Topic | Status | Relation to SSD streaming engine |
|-------|--------|--------------------------------|
| **RWKV-8 DeepEmbed** | May 2025 announce; preview | Extra memory tensors beside layer weights — defer until M3 proven |
| **RWKV-8 ROSA** | Oct 2025 announce; experimental | Model-side retrieval; orthogonal to per-layer SSD reads |
| **DeepSeek Engram** | Jan 2026 release | Three-tier HBM/DRAM/NVMe — **residency profile** reference only |
| **HRWKV7** | RWKV-Infer | Hybrid blocks; only RWKV layers streamed — see milestone M8 |
| **GDS / fastsafetensors at scale** | P2 benchmark | Bulk load, not per-token layer stream — borrow if cold-start wins |

See [`docs/IDEAS.md`](../IDEAS.md) appendix for paragraphs on each.
