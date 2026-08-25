# Architecture research and implementation status

Updated July 15, 2026. This is the evidence-backed boundary between the
production RWKV-7 SSD path and the research baselines. A paper result or a
simulator result is not counted as an end-to-end engine speedup.

## DeepEmbed: related to Engram, but not the same thing

DeepEmbed is a RWKV-v7-derived model family, not a generic storage format. The
public [RWKV-DeepEmbed-mmap](https://github.com/Beortext/RWKV-DeepEmbed-mmap)
implementation contains an older qkv/DEA contract with context-indexed
`s_emb`, `k_emb`, and `v_emb` tables. It removes those tables from the main
model file and stores them in a custom mmap `.bin` file with a binary tensor
region, a JSON index, and a 16-byte `<QQ>` footer. A token lookup is therefore
part of the model forward, not merely a different RWKV quantizer.

The downloaded `rwkv7a-g1d-0.1b-20260212-ctx8192.pth` is a different,
supported contract: ChatRWKV's native DeepEmbed-v1 path, selected with
`RWKV_DE_VERSION=1`. Its `s_emb` and `s_emb_x` tensors stay in the ordinary
packed layer tensors; for a token row the CPU streaming path derives
`s_emb[token] + emb[token] @ s_emb_x.T`. It does not need the qkv/DEA sidecar.

Engram is a separate conditional-memory architecture. Its paper describes
deterministic N-gram addressing and an optional host-memory offload path; the
official [DeepSeek Engram implementation](https://github.com/deepseek-ai/Engram)
also labels its Python code as a demonstration that mocks the surrounding
attention/MoE/mHC components. Both designs motivate the same systems idea:
deterministic lookup enables tiered storage and prefetch. DeepEmbed cannot be
declared “Engram support”, and ordinary RWKV-7 cannot consume DeepEmbed
tensors. That shared storage intuition does not make the model contracts
interchangeable: this repository's DeepEmbed support is not Engram support,
and Engram checkpoints/routing are not implemented by adding DeepEmbed keys.

### What is implemented here

| Capability | Status | Notes |
|---|---|---|
| Checkpoint and variant detection | Done | `infer_checkpoint_meta()` records `deepembed`, layers, tensor keys, and distinguishes `qkv_dea` from `rwkv7a_v1` under `rwkv7_deepembed`. |
| qkv/DEA sidecar generation | Done | `pack_runtime` writes `DeepEmbed.bin` for the qkv/DEA contract. Use `--no-deepembed-sidecar` to opt out. |
| Sidecar validation/lookup | Done | `rwkv_ssd.runtime.deepembed.DeepEmbedSidecar` validates offsets, shapes, dtypes, index bounds, and performs exact token-row lookup. |
| qkv/DEA resident reference forward | Done | `DeepEmbedReferenceModel` is a CPU-safe adapter for resident experiments and uses the sidecar when present. |
| RWKV7a DeepEmbed-v1 resident | Done | ChatRWKV is imported with `RWKV_DE_VERSION=1`; native ChatRWKV merges the v1 tables in RAM. |
| RWKV7a DeepEmbed-v1 CPU layer streaming | Done | The skeleton/provider path derives the requested `s_emb` row and uses the v1-aware CMix path. Resident vs streaming greedy parity passes on the downloaded 0.1B checkpoint. |
| qkv/DEA CPU layer streaming | Done as a reference path | `DeepEmbedReferenceModel.forward_streaming()` loads ordinary block tensors through the provider and lookup rows through `DeepEmbed.bin`; parity is covered by a synthetic complete-contract test. Fused production kernels remain open. |
| rwkv.cpp DeepEmbed | Not supported upstream | The official README lists v4, v5, v6, v7 and GGML FP16/INT4/INT5/INT8 formats; no DeepEmbed tensor or sidecar API is present. The vendored source has the same boundary. |

Pack and inspect a sidecar:

```powershell
python -m rwkv_ssd.tools.pack_runtime `
  --input deepembed-model.pth `
  --output .\runtime_pack_deepembed
python -m rwkv_ssd.tools.verify_pack .\runtime_pack_deepembed
python -m app.cli --model .\runtime_pack_deepembed `
  --checkpoint deepembed-model.pth --backend chatrwkv --mode resident `
  --prompt "Hello" --max-tokens 16
```

Both contracts now have CPU correctness paths. RWKV7a-v1 uses native ChatRWKV;
qkv/DEA uses `DeepEmbedReferenceModel.forward_streaming()` with a required
`DeepEmbed.bin` sidecar. Both remain Python/reference paths rather than fused
production kernels. The next DeepEmbed work is batching and a fused provider
transaction for the qkv/DEA rows; that is a separate model contract and should
not be hidden behind a generic "Engram" flag.

## Improvements completed in the current pass

The existing runtime prototypes remain the source of truth for prefetch,
adaptive residency, promotion, snapshots, striped packs, CMix selective reads,
quality-gated codecs, and real ChatRWKV weight-stationary batching. This pass
adds the missing exact prompt-side schedule: `prefill_text_streaming()` now
processes a prompt in layer-outer chunks, loading each streamed layer once per
chunk and running the ChatRWKV sequence kernels. Set
`RWKV_STREAM_PREFILL_CHUNK=N` to tune the chunk size; the default is 256 tokens.
If an old ChatRWKV package lacks the sequence kernel, the code falls back to
the exact token loop.

The defended metric remains:

```text
accepted tokens per full streamed weight sweep
```

The short-token decode loop is still recurrent. Weight-stationary batching is
the capacity optimization for multiple sessions; it should not be reported as
a single-session latency improvement.

## Small real Mamba-2 and Transformer references

The downloaded checkpoints are now exercised by
`research/real_sequence_models.py`:

- an operator-supplied HF Mamba-2 directory is a real checkpoint with
  527,240 parameters;
- an operator-supplied HF Transformer directory is a real Llama-style causal Transformer
  with 1,032,272 parameters.

The dependency-light references implement the tensor contracts needed for CPU
correctness work: RMSNorm, Mamba-2 convolution/state-space recurrence and
cached state, or RoPE/causal attention/KV cache/SwiGLU. They produce finite
logits and full-sequence versus token-by-token cached decoding parity. This is
an executable architectural baseline, not a first-class high-throughput
Mamba/Transformer backend.

Run the real-checkpoint gate with:

```powershell
.venv-cpu\Scripts\python.exe -m pytest -q `
  --basetemp .pytest-tmp\codex-real-sequence `
  tests/test_real_sequence_models.py -m integration
```

Tokenizer files are present beside both checkpoints. The reference harness
intentionally uses fixed token IDs so it has no dependency on a tokenizer
package and does not claim HF tokenizer compatibility. Native ChatRWKV uses
its own tokenizer.

## Tiny Mamba, attention, and hybrid baselines

`research/tiny_sequence_models.py` contains three deliberately isolated
baselines, all configurable around the 0.01B scale:

- `mamba`: selective diagonal state-space recurrence with explicit recurrent
  state, intended for chunk/state-size experiments;
- `attention`: causal multi-head attention with the same embedding, layer, and
  feed-forward budget;
- `hybrid`: alternating Mamba and attention blocks for placement experiments.

Run the research benchmark:

```powershell
python research/tiny_sequence_models.py --kind all --seq-len 64
python research/tiny_sequence_models.py --kind all --train-steps 25 `
  --json-out bench/results/tiny-sequence-baselines.json
```

The toy training task is a deterministic copy/shift pattern. It is a harness
for comparing state/update mechanics, not evidence that any baseline is a
useful pretrained language model. The default configuration is approximately
ten million parameters; the benchmark records the exact count. A July 15 CPU
smoke with `seq_len=8` reported the following one-run snapshot:

| Baseline | Parameters | Forward tok/s |
|---|---:|---:|
| Mamba | 12,654,848 | 490.6 |
| Attention | 12,609,792 | 1,550.1 |
| Hybrid | 12,632,320 | 516.5 |

These numbers are kernel/hardware observations for the toy harness, not a
claim about language-model quality or a production SSD backend.

## DSpark proposal research path

`rwkv_ssd/runtime/dspark.py` contains the model-agnostic mechanisms from
[DSpark](https://arxiv.org/abs/2607.05147): a parallel or state-rolled base
block, a small sequential correction head, conditional confidence estimates,
and a capacity-aware verification-prefix scheduler. `DSparkDrafter` accepts
`[batch, steps, vocab]` base logits and `[batch, steps, hidden]` states, so the
same research interface can consume a Transformer attention block, a Mamba
state-rolled block, or an RWKV recurrent block.

The Markov head is the inexpensive default; the RNN head is available when a
recurrent correction state is worth the extra work. The confidence head emits
conditional per-position survival probabilities, whose cumulative product is
used by the scheduler. `verify_speculative` implements standard lossless
rejection sampling, including the residual target distribution after the first
rejection and the target bonus token when the whole draft is accepted.

The tiny harness exposes this path through `dspark_propose_tiny()`. Attention
can produce a base block in parallel, while Mamba and the hybrid model still
roll their recurrent state across the block. Therefore this is a shared
proposal/correction and verification research path, not a claim that DSpark
makes recurrent backbones parallel during decoding. Production use remains
gated on trained drafter heads, calibrated confidence, a target-model
verification loop, and measured hardware capacity curves; random research
heads are not enabled in the packed backends.

The dependency-light real-checkpoint harness exposes the same contract through
`research/real_sequence_models.py`: `ReferenceLlama` and `ReferenceMamba2`
provide `forward_hidden()` plus `logits_from_hidden()`, and
`dspark_propose_reference()` applies the common drafter to either architecture.
For recurrent or cached models, the masked proposal state must be discarded
until target verification accepts tokens; only the accepted target path may
commit the KV/SSM/RWKV state.

## Paper-derived decisions

| Research | Useful design signal | Decision in this repository |
|---|---|---|
| [RWKV-7 “Goose”](https://arxiv.org/abs/2503.14456) | Token-dependent decay/mixing and recurrent inference with no KV cache. | Preserve resident recurrent state; keep weight streaming and state updates as separate scheduling concerns. |
| [RWKV-X hybrid](https://arxiv.org/abs/2504.21463) | Hybrid recurrent/attention blocks can trade local recurrence for retrieval capacity. | Keep hybrid placement as a research axis; do not assume every block has the same streaming transaction. |
| [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | SSM/attention duality and chunkwise state updates; the paper reports a 2–8x faster core layer in its setting. | Use chunkwise/layer-outer prefill as the first implementation target; do not transfer the paper's speed number to SSD decode. |
| [FlashAttention](https://arxiv.org/abs/2205.14135) / [FlashAttention-2](https://arxiv.org/abs/2307.08691) | IO-aware tiling reduces high-bandwidth-memory traffic in attention. | Apply the principle to contiguous layer reads and fused attention kernels; CPU eager attention is only a correctness reference. |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | More expressive discretization, complex state transitions, and MIMO are promising but recent. | Keep as a research follow-up; the tiny harness is the safe place to compare state choices before production kernels. |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Gating and delta updates complement each other; hybrid attention/SSM variants are a useful retrieval/long-context direction. | Include a hybrid baseline and measure placement/state quality separately from RWKV SSD I/O. |
| [DSpark](https://arxiv.org/abs/2607.05147) | Confidence-scheduled speculative decoding combines a parallel/state-rolled proposal block with a lightweight Markov or RNN correction head and exact verification. | Keep one model-agnostic research interface for Transformer, Mamba, and RWKV; require trained heads, calibrated confidence, and a target verifier before enabling a production backend. |
| [Engram](https://arxiv.org/abs/2601.07372) | Deterministic lookup allows conditional memory and host-side prefetch. | Reuse the tiering/layout lesson for DeepEmbed sidecars, but keep the model contracts separate. |
| [LLM in a Flash](https://arxiv.org/abs/2312.11514) | Reduce bytes transferred and favor contiguous storage transactions. | Keep layer-grouped packs, coalesced spans, shadows, and one-sweep batching as the storage-side priorities. |

## Still gated by hardware or training

CUDA/GDS, a fused GPU LUT2 or Mamba/attention/RWKV kernel, physical
multi-SSD scaling, large 2.9B/7B measurements, new quantizer training, and
high-quality Mamba/attention language models are not honestly finishable on
this Windows CPU/iGPU development host. Production-quality Mamba/Transformer
backends, rwkv.cpp DeepEmbed support, and large-model training remain open
engineering work even though small CPU references are complete.

## Verification snapshot

The implementation was checked with the following commands on July 15, 2026:

```powershell
python -m pytest --basetemp .pytest-tmp\codex-suite-final2 -q
# 487 passed, 2 skipped, 23 deselected

python -m pytest --basetemp .pytest-tmp\codex-targeted `
  tests/test_deepembed.py tests/test_real_sequence_models.py `
  tests/test_backend_capability_probe.py -q
# DeepEmbed and real-reference tests pass; ChatRWKV integration is opt-in

python -m pytest -o addopts= --basetemp .pytest-tmp\codex-real-final2 `
  tests/test_real_model_streaming.py -q
# 3 passed: resident/streaming parity, 32-token parity, and batch parity

python -m rwkv_ssd.tools.pack_runtime `
  --input C:\models\rwkv-model.pth `
  --output .pytest-tmp\codex-pack-0.01b --model-family rwkv7 --no-hash
python -m rwkv_ssd.tools.verify_pack .pytest-tmp\codex-pack-0.01b
# OK: 72 tensors, 71012864 bytes, version=1
```

The separate provider-persistence regression was also rerun after the final
cache fix and passed. The default pytest selection intentionally excludes
`chatrwkv`, `slow`, and `integration` markers; use `-o addopts=` for those
explicit gates.

The real RWKV7a validation pack is an operator-prepared RWKV7a pack. It
contains 462 tensors and a 2,015,294,976-byte `weights.bin`, records
`deepembed_variant=rwkv7a_v1`, and intentionally has no `DeepEmbed.bin`.
ChatRWKV resident and CPU streaming greedy decoding for `"Hi"` both produced
the same token IDs under the shared prefill/decode contract; sequence prefill
matched resident argmax decisions.
These are correctness results, not throughput measurements.
