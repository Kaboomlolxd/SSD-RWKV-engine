# Trinity evaluation packs

These artifacts all use the same RWKV-7 0.1B checkpoint:
`rwkv7-g1d-0.1b-20260129-ctx8192.pth`.

## Active default

`trinity_grouped_0.1b` is the compact default for automatic resolution of a
historical `trinity_lut2_*` request. It contains 326 large tensors as
`scale_u8_grouped` with 256 weights per scale group; 76 one-dimensional
control/normalization vectors remain dense BF16 for recurrent quality.

| Pack | Weight payload | Status |
| --- | ---: | --- |
| `trinity_grouped_0.1b` | 198,250,504 bytes (about 189 MiB) | Active compact default |
| `trinity_safe_0.1b` | 382,763,008 bytes (about 365 MiB) | Dense correctness fallback |
| `trinity_lut2_*` | removed | Deprecated legacy/intermediate payloads; historical results remain in `archive/` |

The grouped pack is the best current compact CPU candidate. The corrected
mixed policy passes the current 32-token real-model resident-vs-streaming
greedy parity test with fused decode disabled and enabled. Use
`trinity_safe_0.1b` for broader reference comparisons, and do not use the
archived LUT2 artifacts for production quality claims.

The grouped weight-file SHA-256 is:

`7a8045d04d71df792c6c79a11893fbfe8a5400d12b3d287f2d1dba5e772ff7fa`

## Resolution and reproducibility

With the default `RWKV_PACK_PROFILE=auto`, an application pointed at
Historical `trinity_lut2_*` names resolve to the sibling
`trinity_grouped_0.1b`. If that pack is absent, automatic resolution falls
back to `trinity_safe_0.1b`.

Historical LUT2 experiments are represented by the checked-in benchmark
results and archive documentation. The deprecated weight payloads are not
kept in the working tree; set `RWKV_PACK_PROFILE=lut2` only when supplying a
separately rebuilt legacy pack.
