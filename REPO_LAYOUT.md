# Repository layout

The repository is organized around the engine, its repeatable gates, and
evidence. Large model/payload files are local artifacts and are intentionally
ignored; metadata and source commands remain reviewable.

```text
rwkv_ssd/           Engine library: runtime, codecs, providers, backends, tools
app/                CLI and local HTTP serving surface
tests/              Unit, regression, and opt-in real-model gates
bench/              Active benchmarks and result metadata
deploy/             Deployment examples and residency profiles
docs/               Current status, contracts, runbooks, research, roadmap
archive/            Superseded docs, legacy benches, and old experiments
backends/           Vendored rwkv.cpp reference backend
operator-models/    User-supplied checkpoints and prepared packs (outside Git)
research/           Small architecture/state reference models, not production backends
storage_bench/      Linux storage ceiling research artifact
simulations/        Legacy feasibility simulations
papers/             Optional paper drafts
```

## Primary entry points

| Need | File |
|---|---|
| Current recommendation | [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) |
| Full path/readiness audit | [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) |
| Backend contracts and rwkv.cpp | [`docs/BACKENDS.md`](docs/BACKENDS.md) |
| RAM/speed tiers and environment variables | [`docs/PRESETS.md`](docs/PRESETS.md) |
| Pack streaming and cache details | [`docs/SSD_STREAMING_FRONTIER.md`](docs/SSD_STREAMING_FRONTIER.md) |
| Benchmark catalog | [`bench/README.md`](bench/README.md) |
| Result metadata policy | [`bench/RESULTS.md`](bench/RESULTS.md) |
| Full evaluation report | [`docs/ENGINE_EVALUATION_REPORT.md`](docs/ENGINE_EVALUATION_REPORT.md) |
| Archive policy | [`archive/README.md`](archive/README.md) |

## Model artifacts

Checkpoints, converted GGML files, tokenizer bundles, generated packs, and
benchmark output belong in an operator-selected model/output directory, not in
the source distribution. Use [`docs/MODEL_IMPORT.md`](docs/MODEL_IMPORT.md) to
create and validate them. Do not commit a generated cache or model payload to
make a test pass locally. A pack is not release-ready merely because it passes
structural verification: lossy real RWKV-7 packs also need a passing
`quality_certificate.json` bound to the manifest, sidecar metadata, and every
declared artifact.

For the current 2.9B grouped-U8 experiment, the reproducible build and release
check are:

```powershell
python -m rwkv_ssd.tools.pack_runtime `
  --input C:\models\rwkv-model.pth `
  --output C:\prepared\rwkv-model-grouped.pack `
  --model-family auto `
  --pack-codec scale_u8_grouped --scale-group-size 32 `
  --pack-layout layer_grouped

# Supply measured metrics and gates from the exact checkpoint/pack/tokenizer
# run; this command refuses to write a passing certificate when a gate fails.
python -m rwkv_ssd.tools.quality_certificate `
  --pack C:\prepared\rwkv-model-grouped.pack `
  --metrics C:\evidence\model-metrics.json `
  --gates C:\evidence\model-gates.json `
  --scope "checkpoint=<sha256>;pack=<weights-sha256>;tokenizer=<sha256>;prompts=<declared-set>"

python -m rwkv_ssd.tools.preflight `
  --pack C:\prepared\rwkv-model-grouped.pack --backend rwkvcpp `
  --checkpoint C:\models\matching-rwkv-model-FP16.bin --json
```

The promoted compact profile's size and metrics are properties of the exact
operator-supplied checkpoint and pack, not repository-wide constants. Record
them in the pack's release evidence. Never manufacture a certificate from
throughput or structural verification alone.

## Organization rules

1. New current evidence belongs in `bench/results/` with checkpoint, pack,
   backend, device, thread, and warm/cold metadata.
2. New operator guidance belongs in the canonical document named by
   [`docs/README.md`](docs/README.md); do not create another status table.
3. Superseded plans and exploratory benches move under `archive/` and retain
   their original date/context.
4. CUDA/XPU/MPS/Albatross claims must remain marked hardware-gated until a
   matching runner and quality gate exists.

5. Use [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) for the
   architecture/path matrix, evidence boundaries, working-tree risks, and
   production-readiness decision. Keep `PROJECT_STATUS.md` focused on the
   current recommendation rather than duplicating the full audit.
