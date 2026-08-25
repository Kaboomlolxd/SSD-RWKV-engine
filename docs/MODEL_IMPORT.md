# Model import and preparation

The supported distribution workflow is RWKV-specific:

```text
source checkpoint or HF directory/repository
        -> rwkv-ssd-pack
        -> self-contained runtime pack
        -> preflight
        -> rwkv-ssd / rwkv-ssd-serve
```

Model weights, generated packs, tokenizer files, and converted GGML files are
operator-supplied artifacts. They are intentionally not included in the
source checkout or Python wheel. The commands below use placeholders so they
can be copied to any machine without depending on a developer's test tree.

## Install the importer

For local PyTorch/RWKV inputs, the base package is sufficient. For Hugging Face
repository downloads, install the optional Hub dependency:

```powershell
python -m pip install -e ".[hf]"
```

The installed `rwkv-ssd-pack` command is equivalent to
`python -m rwkv_ssd.tools.pack_runtime`.

## Supported packed architectures

The importer detects tensor naming conventions by default. The maintained
packed runtime is RWKV-focused:

| Family | Input preparation | Runtime path | Boundary |
|---|---|---|---|
| RWKV-5/6/7 | `.pth`, `.pt`, or safetensors | `chatrwkv` reference; `rwkvcpp` when a matching GGML model/native build is supplied | RWKV-7 streaming is the qualified reference path; other versions need their own qualification |
An unsupported or ambiguous architecture remains `unknown` and fails
preflight. Do not force an unrelated family to make a pack appear valid.

## Local model directory or checkpoint

For an HF directory, copy the config, tokenizer, and generation metadata into
the resulting pack so the runtime does not need the original directory:

```powershell
rwkv-ssd-pack `
  --input C:\models\my-model `
  --output C:\prepared\my-model.pack `
  --model-family auto `
  --pack-layout layer_grouped `
  --copy-hf-metadata
```

For a single RWKV checkpoint, use the same command. A `.pth` has no HF
metadata to copy, so RWKV reference execution must retain the original
checkpoint as an explicit runtime input:

```powershell
rwkv-ssd-pack `
  --input C:\models\rwkv-model.pth `
  --output C:\prepared\rwkv-model.pack `
  --model-family auto `
  --pack-layout layer_grouped

rwkv-ssd-preflight `
  --pack C:\prepared\rwkv-model.pack `
  --backend chatrwkv `
  --checkpoint C:\models\rwkv-model.pth
```

Use `--model-family` only for a verified RWKV checkpoint with a non-standard
naming scheme.

## Hugging Face repository ID

The importer can download the selected weights and the small config/tokenizer
bundle into the local HF cache before writing the pack:

```powershell
rwkv-ssd-pack `
  --hf-repo org/model-name `
  --output C:\prepared\model-name.pack `
  --model-family auto `
  --pack-layout layer_grouped `
  --copy-hf-metadata
```

The output is portable after creation. The input repository is not contacted
by the runtime when the copied metadata is sufficient; use the pack's
`manifest.json` and `meta.json` as the identity record.

## Compact codecs and quality gates

Start with `--pack-codec none` (the default) when establishing parity. A
compact codec is a separate model qualification decision:

```powershell
rwkv-ssd-pack `
  --input C:\models\rwkv-model.pth `
  --output C:\prepared\rwkv-model-u8.pack `
  --model-family rwkv7 `
  --pack-codec scale_u8_grouped `
  --scale-group-size 32 `
  --pack-layout layer_grouped

rwkv-ssd-preflight `
  --pack C:\prepared\rwkv-model-u8.pack `
  --backend chatrwkv `
  --checkpoint C:\models\rwkv-model.pth
```

Structural verification is not a quality certificate. For a lossy real-model
pack, compare against the resident reference using the declared prompts,
tokenizer, checkpoint hash, and generation length, then issue and verify a
manifest-bound `quality_certificate.json`. A codec that has not passed that
process remains an experiment regardless of its storage ratio or speed.

## Run the prepared pack

Self-contained RWKV packs can run without the source checkpoint:

```powershell
rwkv-ssd `
  --model C:\prepared\my-model.pack `
  --backend rwkvcpp `
  --mode streaming `
  --prompt "Hello" `
  --max-tokens 32
```

RWKV reference streaming needs the original checkpoint for the ChatRWKV
adapter, while `rwkvcpp` additionally needs a matching converted GGML model
and native library. Those are backend prerequisites, not hidden pack contents.

## Reproducibility checklist

Keep the following beside any deployed pack or release record:

1. `manifest.json`, `meta.json`, and the pack hash;
2. source checkpoint or HF revision hash;
3. tokenizer/config hashes when they are not copied into the pack;
4. selected backend, native library/submodule revision, device, and thread
   settings;
5. the exact preflight result and quality certificate, if the pack is lossy.

This keeps the repository small while making an operator-created model pack
auditable and repeatable.
