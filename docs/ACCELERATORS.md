# Accelerator support

The engine separates three concerns that are easy to conflate:

1. choosing a Torch device;
2. decoding a packed layer onto that device; and
3. running the RWKV recurrent kernels on that device.

The Intel path below is the first integrated-GPU implementation. Other
accelerators are deliberately listed as plans until their runtime, operator
coverage, and memory behavior have been verified on real hardware.

## Current status

| Platform | Status | Recommended path |
|---|---|---|
| Intel Arc discrete or Intel integrated GPU | Experimental, implemented | `--backend chatrwkv --device xpu --strategy "xpu bf16"` |
| AMD integrated GPU / APU | Planned | ROCm/HIP Torch path after an APU-capable wheel and driver matrix exist |
| Apple M-series | Planned | Torch MPS path; no Metal-specific engine code is wired yet |
| Snapdragon / ARM PCs | Planned | Architecture-neutral Torch CPU first, then QNN/ExecuTorch or an equivalent accelerator backend |
| `rwkv.cpp` on non-CPU accelerators | Not the current XPU path | Keep using `chatrwkv` for Intel XPU; SYCL/other native backends need separate work |

## Intel Arc / XPU

The XPU path uses a pack-only RWKV-7 skeleton. Global tensors are loaded onto
XPU, streamed LUT2 layers are decoded with Torch indexing on XPU, and the
ordinary Torch RWKV-7 matmuls execute on XPU when the runtime passes the
matrix-capability probe. The CPU fused LUT GEMV is disabled for this device
because it converts activations to NumPy and would move the dominant
projections back to the host.

Both `streaming` and `resident` modes are supported for ordinary RWKV-7. In
resident mode the engine warms the skeleton into `model.z` before generation;
in streaming/partial mode the existing layer scheduler supplies the dense
layer tensors to the same Torch forward implementation. Grouped LUT2 codebooks
(including FP16-codebook, residual, and input-layout variants) use device-side
gathering as well.

This requires a Torch build with `torch.xpu` support and a working Intel GPU
driver. `torch.xpu.is_available()` is only the runtime/enumeration check; the
engine also runs a tiny synchronized matrix operation before selecting XPU for
model computation. If allocation/gather works but matrix engines do not, an
`--device xpu` request safely uses CPU model computation and keeps XPU
available for Trinity decode. Force that split explicitly with
`--trinity-decode-device xpu` when needed.

On Intel Arc 140V, the repaired Windows environment is `.venv-xpu` with
Python 3.12 and `torch==2.13.0+xpu`. It enumerates the GPU and runs LUT2
gathers, but currently fails even a 2x2 BF16 matrix operation with `could not
make an engine with allocator`. The older local `torch==2.12.1+xpu` /
oneAPI 2025.3.x artifact set remains available for a future compatibility
comparison.

When using the repaired environment from PowerShell, expose the bundled Intel
and Torch DLL directories before launching Python:

```powershell
$env:PATH = `
  "$pwd\.venv-xpu\Library\bin;$pwd\.venv-xpu\Scripts;$pwd\.venv-xpu\Lib\site-packages\torch\lib;" + `
  $env:PATH
```

Example:

```powershell
python -c "import torch; print(torch.__version__, hasattr(torch, 'xpu'), torch.xpu.is_available())"
python -m app.cli `
  --model C:\prepared\reference-pack `
  --checkpoint C:\models\rwkv-model.pth `
  --backend chatrwkv --device xpu --strategy "xpu bf16" `
  --mode streaming --prompt "Hello" --max-tokens 16
```

`trinity_decode_device: xpu` can be set in a config file when explicit decode
placement is useful. For the CPU-compute/XPU-decode split used by the verified
benchmark, set `RWKV_TRINITY_XPU_AUTO=1`; `RWKV_TRINITY_DECODE_DEVICE=xpu` is
available to callers that construct configuration from environment values.

The qkv/DEA DeepEmbed contract is intentionally rejected on XPU for now. Its
sidecar-backed lookup and variant-specific kernels need an accelerator-aware
implementation. RWKV7a-v1 remains on the ordinary Torch skeleton path, but it
still needs a real Arc parity run before it should be considered production
ready.

## Roadmap

### AMD integrated graphics and APUs

1. Establish the supported ROCm-on-APU combinations, including the minimum
   Linux kernel, ROCm, and Torch versions and whether Windows support is viable.
2. Map the device to Torch's ROCm `cuda` namespace without assuming a discrete
   GPU or a large VRAM pool.
3. Reuse the dense LUT2-on-device path, then add HIP kernels only where Torch
   indexing/matmul is measurably insufficient.
4. Add UMA-aware residency limits: decoded layer bytes, recurrent state, and
   operating-system page cache must share system memory.
5. Run parity and sustained-throughput tests on at least one Ryzen APU and one
   Radeon integrated device before enabling automatic selection.

### Apple M-series

1. Add an explicit MPS capability probe and device validation rather than
   treating every `mps` string as available.
2. Audit bfloat16, float16, `index_select`/gather, layer norm, recurrent state
   updates, and vocabulary-head matmul on MPS. Use float16 or float32 where an
   operator lacks bfloat16 support.
3. Replace CUDA-only staging assumptions with a backend-neutral asynchronous
   transfer interface; MPS uses unified memory but still has synchronization
   costs.
4. Verify grouped LUT2 and RWKV7a-v1 parity, then add MPS hardware tests.

### Snapdragon and other ARM systems

1. Keep pack parsing, alignment, manifests, and dense Torch forward free of
   x86-specific assumptions; validate the CPU path on Windows ARM and Linux
   ARM64 first.
2. Add a capability-based backend interface for QNN, ExecuTorch, or another
   supported Snapdragon runtime instead of making the engine depend on one
   vendor SDK.
3. Define how recurrent state and layer streaming cross the CPU/NPU boundary,
   including copy cost, quantized operator coverage, and tokenizer placement.
4. Add ARM64 CI and a real Snapdragon device before advertising acceleration.

## GPU-gated problems to track

- **Binary availability:** Torch XPU, ROCm-on-APU, MPS, and Snapdragon runtimes
  have different Python/version/driver matrices. A device string alone is not
  proof that kernels are available.
- **Operator coverage:** grouped gather, residual repair, layer norm, state
  updates, BF16/FP16 conversion, and vocabulary-head matmul must all remain on
  the accelerator. Unsupported operators can create hidden host fallbacks.
- **Packed kernels:** the CPU fused LUT GEMV is not a GPU implementation. A
  native XPU/HIP/Metal LUT GEMV or a better dense materialization policy may be
  needed for larger models and low-memory modes.
- **UMA pressure:** integrated GPUs draw from system memory. Decoded layers,
  `model.z`, recurrent state, Torch allocator pools, and SSD page cache compete
  for the same budget; discrete-GPU VRAM heuristics are unsafe.
- **Overlap:** the current CUDA ping-pong staging ring is CUDA-specific. XPU,
  HIP, MPS, and vendor NPUs need their own event/queue and pinned/shared-memory
  semantics before asynchronous SSD-to-device overlap is claimed.
- **Custom RWKV kernels:** the vendored ChatRWKV kernels are pure Torch on the
  XPU skeleton, but optimized CUDA/Triton/Flash-style kernels are not portable
  automatically. Fused kernels need per-backend implementations and parity
  tests.
- **DeepEmbed:** qkv/DEA sidecar lookup, context-indexed tables, and variant
  adapters need device kernels or an explicit CPU placement policy.
- **Native backends:** rwkv.cpp currently exposes CPU/CUDA/Metal/BLAS-oriented
  paths, not a Python-wired XPU path. A future SYCL/HIP/Metal integration must
  preserve the pack-backed layer streaming contract rather than silently
  reverting to CPU.
- **Thermal and power behavior:** integrated GPUs can share package power with
  the CPU. Throughput comparisons need sustained runs, temperature, and power
  telemetry rather than one short token burst.

The next implementation gate is an Intel Arc parity/throughput run. After that,
the same capability probes and grouped-LUT tests should be reused for ROCm and
MPS instead of adding platform-specific fallbacks to the core provider.
