# Engine architecture and runbook

## Design goal

RWKV SSD keeps recurrent state resident while large per-layer weights travel
through a bounded path:

```text
weights.bin -> store/read -> provider cache -> staging -> backend compute
                   ^              |              |
                   |              +-- prefetch --+
                   +-- manifest, codec, residency policy
```

The same engine also supports resident inference for reference comparisons.
The streaming path is deliberately measurable: per-layer reads, staging,
provider bytes, cache hits, and compute are recorded in `MetricsCollector`.

## Backend roles

| Backend | Use | Qualification |
|---|---|---|
| `rwkvcpp` | Default real-CPU RWKV backend; native GGML resident and provider bridge | CPU/native validation only; matching GGML model required |
| `chatrwkv` | RWKV-7 PyTorch compatibility/reference path, including DeepEmbed variants | Best reference for pack parity and F-tier experiments |
| `synthetic` | Small deterministic pack and streaming golden tests | CI/reference only |
| `albatross` | External layer-wise CUDA adapter over the shared pack provider | Hardware-gated; not locally certified |

The supported runtime surface is deliberately RWKV-only. The obsolete generic
sequence-model executors and their state/fixture plumbing were removed so the
CLI, service, snapshots, and performance work share one coherent model contract.

Capabilities are declared in `rwkv_ssd/backends/capabilities.py` and selected
through `rwkv_ssd/backends/factory.py`. Do not infer production support from a
backend being importable; use `rwkv-ssd-preflight` and the capability probe.

## Runtime modes

| Mode | Behavior |
|---|---|
| `resident` | Materialize all model weights; quality/speed reference |
| `partial` | Keep a configured hot subset resident and stream the remainder |
| `streaming` | Read or decode streamed layers through the provider on each sweep |

F-tier environment controls are documented in
[`docs/PRESETS.md`](docs/PRESETS.md). Use one primary residency policy per
run; do not combine historical benchmark claims from different warm/cold
conditions.

## rwkv.cpp path

Build the vendored backend:

```powershell
cmake -S backends/rwkvcpp_ref -B backends/rwkvcpp_ref/build
cmake --build backends/rwkvcpp_ref/build --config Release
```

Convert the checkpoint to a matching GGML file with
`backends/rwkvcpp_ref/python/convert_pytorch_to_ggml.py`, or set
`RWKVCPP_GGML_PATH` to an existing conversion. On Windows the expected DLL is
usually `librwkv.dll`; `RWKVCPP_DLL` overrides discovery.

The engine passes manifest `n_embd` into the backend before construction so
the automatic native thread policy can select a model-width tier. Explicit
`RWKV_CPU_THREADS=N` always wins. The measured policy is intentionally
conservative for single-token memory-bound GEMV:

- width below 1280: one native thread;
- width 1280–2047: two to four threads, host-scaled;
- width 2048 and above: two to four threads, capped to avoid the observed
  oversubscription slowdown.

`RWKVCPP_SYNC_EVERY_TOKEN=1` restores the diagnostic per-token upload path.
The default is weight-stationary after the provider has synchronized the graph,
which avoids duplicate uploads on follow-up and prefix-cache requests.

On the local AVX2 Windows CPU, the 2.9B GGML measurements were approximately
2.8 tok/s for FP16 and 4.2–4.6 tok/s for Q4_K at their best observed thread
counts. The compact grouped-U8 provider path is a different compute path and
measured about 1.36 tok/s warmed end-to-end on the same machine. Neither meets
the aspirational 15 tok/s target; CUDA is the remaining hardware-gated route
for a substantially higher ceiling.

## Pack contract

Each runtime pack contains:

- `weights.bin`: aligned tensor payloads, optionally codec-packed;
- `manifest.json`: tensor names, shapes, dtype/codec, offsets, lengths, and
  residency;
- `meta.json`: checkpoint/model identity and layout metadata.

Validate before running (replace the placeholder with the operator's pack):

```bash
python -m rwkv_ssd.tools.verify_pack ./prepared-model.pack
python -m rwkv_ssd.tools.preflight --help
```

Large checkpoints and generated pack payloads are local artifacts. They are
ignored by Git so a source checkout stays reviewable; keep their hashes and
creation commands in benchmark result metadata instead.

The generic import workflow, supported architecture matrix, and backend-specific
prerequisites are documented in [`docs/MODEL_IMPORT.md`](docs/MODEL_IMPORT.md).

## Development gates

```bash
python -m pytest tests/ -q
python -m compileall -q rwkv_ssd app bench tests
python bench/bench_backend_compare.py --help
```

For the release evidence and known boundaries, read
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md),
[`docs/BACKENDS.md`](docs/BACKENDS.md), and
[`docs/ENGINE_EVALUATION_REPORT.md`](docs/ENGINE_EVALUATION_REPORT.md). For
the complete path and production-readiness audit, read
[`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md).
