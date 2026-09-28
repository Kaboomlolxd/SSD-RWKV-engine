# RWKV SSD Inference Engine

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

RWKV SSD is an inference engine for recurrent models whose large per-layer
weights can be stored on SSD and loaded through a bounded, instrumented
provider during decode. Recurrent state stays resident; the runtime handles
pack layout, reads, prefetch, staging, residency tiers, native kernels, and
backend dispatch.

The project is CPU-first. The default real-CPU backend is `rwkvcpp` (rwkv.cpp
GGML); `chatrwkv` is the compatibility/reference path for RWKV-7 PyTorch and
DeepEmbed variants. Synthetic packs provide the dependency-light correctness
golden path. CUDA, GDS, XPU, MPS, and Albatross remain hardware-gated and are
not certified by the local release gate.

## Current release posture

The current source-backed recommendation is in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md). In brief:

- FP16/BF16 resident inference is the quality and speed reference.
- The default product workflow is to import an operator-supplied model with
  [`docs/MODEL_IMPORT.md`](docs/MODEL_IMPORT.md), then run preflight on the
  generated pack. The compact grouped-U8 profile is available, but its
  manifest-bound certificate is scoped to the exact qualified artifact and
  short smoke; longer held-out quality qualification remains open.
- The old all-LUT2 2.9B pack was removed after its application-quality gate
  failed; historical benchmark results remain available in the archive.
- rwkv.cpp resident FP16 and provider-backed paths are usable for CPU
  experiments. The provider bridge uses a weight-stationary synchronization by
  default; set `RWKVCPP_SYNC_EVERY_TOKEN=1` only for strict upload diagnostics.

The full evaluation and remediation record is
[`docs/ENGINE_EVALUATION_REPORT.md`](docs/ENGINE_EVALUATION_REPORT.md).

The current CPU continuation has measured, opt-in-safe fast paths rather than
stacked headline multipliers: grouped-U8 decode is **5.05×** faster and grouped
LUT2 decode **4.63×** faster in focused A/B probes; DeepEmbed sidecar lookup is
approximately **19×–1,202×** faster depending on access pattern; and the
shared-layer DeepEmbed reference batch path is **1.17×** faster with **50% fewer
layer loads** in the maintained two-session probe. Native sequence prefill
reuses its scratch buffers, uninterrupted generation elides intermediate state
publication, and whole-pack zstd is available for cold storage but remains
opt-in because raw mmap is faster on hot reads. Run
`bench/bench_cpu_optimizations.py` for the focused probes.

## Quick start

Install the runtime and development checks:

```bash
python -m pip install -e ".[dev]"
```

Try the CPU command line with a tiny synthetic pack. This verifies the installation without requiring real model files:

```powershell
python -m rwkv_ssd.tools.make_synthetic_pack --output .\demo_pack
python -m app.cli --model .\demo_pack --backend synthetic --mode resident --prompt "Hello" --max-tokens 16
```

For a real RWKV checkpoint, make a pack and check which CPU backends your machine can use. Checkpoints, packs, tokenizers, and GGML files are supplied by the operator and stay outside this repository:

```powershell
rwkv-ssd-pack `
  --input C:\models\rwkv-model.pth `
  --output C:\prepared\rwkv-model.pack `
  --model-family auto `
  --pack-layout layer_grouped
rwkv-ssd doctor `
  --pack C:\prepared\rwkv-model.pack `
  --checkpoint C:\models\rwkv-model.pth
```

Doctor reports the host CPU and RAM, pack integrity, and backend files it can find. It recommends `rwkvcpp` when its native library and matching GGML model are available. That is the faster native CPU path. `chatrwkv` is the PyTorch reference path and needs ChatRWKV plus the original checkpoint. Doctor never changes the backend selection for you.

Run one prompt from the terminal:

```powershell
rwkv-ssd --model C:\prepared\rwkv-model.pack `
  --backend chatrwkv --checkpoint C:\models\rwkv-model.pth `
  --mode streaming --prompt "Hello" --max-tokens 32
```

After selecting a model with `--model` or `--config`, run `rwkv-ssd` without a prompt in a terminal to start an interactive chat. Use `--prompt-file PATH` for a one-shot prompt from a UTF-8 text file. For scripts, pass `--prompt` or `--prompt-file`; the command will not wait for input when stdin is piped.

### Local browser chat

Copy [`rwkv-ssd.example.yaml`](rwkv-ssd.example.yaml) to `rwkv-ssd.yaml`, edit the pack, checkpoint, and backend paths, then run:

```powershell
./run-rwkv-ssd.ps1
```

The script starts the service on `127.0.0.1` and opens the browser. You can also run `rwkv-ssd-serve --config rwkv-ssd.yaml --open-browser` from a terminal. The OpenAI-compatible API remains available at `/v1/chat/completions`. If you configure an API key, enter it in the browser's Connection section for that session.

Use `--mode streaming` when the full model does not fit comfortably in RAM. Cache, residency, codec, and I/O tuning lives in the advanced notes in [`docs/PRESETS.md`](docs/PRESETS.md), [`docs/BACKENDS.md`](docs/BACKENDS.md), and [`docs/MODEL_IMPORT.md`](docs/MODEL_IMPORT.md).

This getting-started path covers RWKV CPU inference. CUDA and other accelerator paths remain hardware-gated.

## Where to read next

| Question | Document |
|---|---|
| What is supported now? | [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) |
| How do I import and prepare a model? | [`docs/MODEL_IMPORT.md`](docs/MODEL_IMPORT.md) |
| What is the full path/readiness audit? | [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) |
| How do I choose RAM/speed tiers? | [`docs/PRESETS.md`](docs/PRESETS.md) |
| Which backend should I use? | [`docs/BACKENDS.md`](docs/BACKENDS.md) |
| How does pack streaming work? | [`docs/V1_STREAMING.md`](docs/V1_STREAMING.md) and [`docs/SSD_STREAMING_FRONTIER.md`](docs/SSD_STREAMING_FRONTIER.md) |
| How do I run and report benchmarks? | [`bench/README.md`](bench/README.md) and [`bench/RESULTS.md`](bench/RESULTS.md) |
| How do I run local HTTP serving? | [`docs/HTTP_SERVING.md`](docs/HTTP_SERVING.md) |
| What is hardware-gated? | [`docs/ACCELERATORS.md`](docs/ACCELERATORS.md) |
| What remains open? | [`docs/GOALS_AND_MILESTONES.md`](docs/GOALS_AND_MILESTONES.md) and [`docs/IDEAS.md`](docs/IDEAS.md) |

## Repository layout

| Path | Purpose |
|---|---|
| [`rwkv_ssd/`](rwkv_ssd/) | Engine runtime, providers, codecs, backends, native loader, and tools |
| [`app/`](app/) | CLI and local HTTP serving surface |
| [`bench/`](bench/) | Active benchmark scripts and checked-in result metadata |
| [`tests/`](tests/) | Regression tests plus opt-in real-model/backend gates |
| [`deploy/`](deploy/) | Example deployment configuration and residency profiles |
| [`docs/`](docs/) | Current status, contracts, runbooks, research, and roadmap |
| [`archive/`](archive/) | Superseded plans, legacy benches, and historical experiments |
| [`backends/rwkvcpp_ref/`](backends/rwkvcpp_ref/) | Vendored rwkv.cpp source and build instructions |
| Operator model directories | Supply checkpoints/packs outside the source tree; large payloads are intentionally not versioned |

See [`REPO_LAYOUT.md`](REPO_LAYOUT.md) for the compact file map and
[`archive/README.md`](archive/README.md) for the archive policy.

## Validation

The latest unrestricted local CPU validation passed **600 tests**, with 18
skipped. The GitHub software selection passed **564 tests**, with 13 skipped
and 18 deselected. The gates include manifest path-security,
quality-certificate, preflight, serving, and incremental-stream tests. These
results cover the local CPU/native environment only; they do not certify CUDA
or another accelerator.

Headline throughput numbers must identify the checkpoint, pack, backend, tier,
thread count, warm/cold state, and whether prompt prefill is included. See
[`bench/RESULTS.md`](bench/RESULTS.md) before adding new numbers.

## License

This project is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not
permitted under the included license. See the license text for the complete
terms and permitted noncommercial uses.

The repository includes third-party source and submodules with their own
licenses. Those terms remain applicable to the corresponding third-party
components.
