# Codec comparison on real RWKV-7 0.1B weights

Source: `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, 48 square weights, 2 bits/weight.

Each weight is quantized once per strategy, decoded back, and compared to the original FP32 source.
Lower MSE / higher SNR / higher cosine sim = better.

## Aggregate (mean over all 48 weights)

| Strategy | mean SNR (dB) | median SNR (dB) | min SNR (dB) | mean RMSE | mean rel RMSE | mean cos sim | median encode (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| linspace (per-tensor) | -10.78 | -10.77 | -21.20 | 0.04543 | 555.58% | 0.3223 | 48.21 |
| kmeans (per-tensor) | 7.32 | 7.51 | 4.74 | 0.00504 | 61.08% | 0.8995 | 280.45 |

## Per-weight results

| name | shape | linspace (per-tensor) | kmeans (per-tensor) |
|---|---|---|---|---|---|---|---|
| blocks.0.att.receptance.weight | [768, 768] | -13.8 dB / 0.175 | 7.8 dB / 0.914 |
| blocks.0.att.key.weight | [768, 768] | -6.8 dB / 0.026 | 5.6 dB / 0.852 |
| blocks.0.att.value.weight | [768, 768] | -18.6 dB / 0.044 | 6.8 dB / 0.890 |
| blocks.0.att.output.weight | [768, 768] | -9.5 dB / 0.021 | 7.0 dB / 0.894 |
| blocks.1.att.receptance.weight | [768, 768] | -11.8 dB / 0.420 | 8.3 dB / 0.922 |
| blocks.1.att.key.weight | [768, 768] | -10.8 dB / 0.759 | 8.5 dB / 0.927 |
| blocks.1.att.value.weight | [768, 768] | -13.3 dB / 0.574 | 6.7 dB / 0.888 |
| blocks.1.att.output.weight | [768, 768] | -15.5 dB / 0.067 | 5.5 dB / 0.849 |
| blocks.2.att.receptance.weight | [768, 768] | -11.4 dB / 0.056 | 7.5 dB / 0.908 |
| blocks.2.att.key.weight | [768, 768] | -9.1 dB / 0.680 | 8.2 dB / 0.920 |
| blocks.2.att.value.weight | [768, 768] | -2.1 dB / 0.051 | 7.9 dB / 0.915 |
| blocks.2.att.output.weight | [768, 768] | -13.1 dB / 0.043 | 7.5 dB / 0.906 |
| blocks.3.att.receptance.weight | [768, 768] | -16.6 dB / 0.610 | 7.0 dB / 0.896 |
| blocks.3.att.key.weight | [768, 768] | -9.8 dB / 0.723 | 7.7 dB / 0.912 |
| blocks.3.att.value.weight | [768, 768] | -10.0 dB / 0.114 | 6.4 dB / 0.879 |
| blocks.3.att.output.weight | [768, 768] | -14.8 dB / 0.468 | 5.1 dB / 0.832 |
| blocks.4.att.receptance.weight | [768, 768] | -16.9 dB / 0.237 | 7.3 dB / 0.902 |
| blocks.4.att.key.weight | [768, 768] | -13.6 dB / 0.515 | 7.8 dB / 0.914 |
| blocks.4.att.value.weight | [768, 768] | -8.9 dB / 0.199 | 6.7 dB / 0.885 |
| blocks.4.att.output.weight | [768, 768] | -14.7 dB / 0.530 | 5.4 dB / 0.845 |
| blocks.5.att.receptance.weight | [768, 768] | -12.0 dB / 0.077 | 6.7 dB / 0.887 |
| blocks.5.att.key.weight | [768, 768] | -9.8 dB / 0.420 | 7.7 dB / 0.912 |
| blocks.5.att.value.weight | [768, 768] | -10.0 dB / 0.739 | 8.3 dB / 0.923 |
| blocks.5.att.output.weight | [768, 768] | -6.8 dB / 0.013 | 8.4 dB / 0.925 |
| blocks.6.att.receptance.weight | [768, 768] | -15.7 dB / 0.089 | 5.6 dB / 0.852 |
| blocks.6.att.key.weight | [768, 768] | -9.5 dB / 0.026 | 6.4 dB / 0.878 |
| blocks.6.att.value.weight | [768, 768] | -9.5 dB / 0.335 | 7.2 dB / 0.900 |
| blocks.6.att.output.weight | [768, 768] | -13.4 dB / 0.073 | 7.3 dB / 0.901 |
| blocks.7.att.receptance.weight | [768, 768] | -10.9 dB / 0.558 | 7.1 dB / 0.898 |
| blocks.7.att.key.weight | [768, 768] | -12.5 dB / 0.358 | 7.3 dB / 0.901 |
| blocks.7.att.value.weight | [768, 768] | -10.2 dB / 0.726 | 8.0 dB / 0.917 |
| blocks.7.att.output.weight | [768, 768] | -0.1 dB / 0.136 | 7.9 dB / 0.916 |
| blocks.8.att.receptance.weight | [768, 768] | -10.7 dB / 0.534 | 7.2 dB / 0.899 |
| blocks.8.att.key.weight | [768, 768] | -2.3 dB / 0.039 | 7.4 dB / 0.905 |
| blocks.8.att.value.weight | [768, 768] | -10.0 dB / 0.634 | 7.7 dB / 0.912 |
| blocks.8.att.output.weight | [768, 768] | -12.1 dB / 0.028 | 8.4 dB / 0.925 |
| blocks.9.att.receptance.weight | [768, 768] | -15.2 dB / 0.527 | 6.4 dB / 0.877 |
| blocks.9.att.key.weight | [768, 768] | -21.2 dB / 0.167 | 4.7 dB / 0.815 |
| blocks.9.att.value.weight | [768, 768] | -10.3 dB / 0.520 | 7.9 dB / 0.916 |
| blocks.9.att.output.weight | [768, 768] | -0.0 dB / 0.084 | 8.2 dB / 0.922 |
| blocks.10.att.receptance.weight | [768, 768] | -11.0 dB / 0.597 | 7.3 dB / 0.903 |
| blocks.10.att.key.weight | [768, 768] | -9.3 dB / 0.030 | 7.9 dB / 0.915 |
| blocks.10.att.value.weight | [768, 768] | -7.7 dB / 0.376 | 8.1 dB / 0.919 |
| blocks.10.att.output.weight | [768, 768] | -9.3 dB / 0.022 | 8.2 dB / 0.922 |
| blocks.11.att.receptance.weight | [768, 768] | -11.3 dB / 0.737 | 8.1 dB / 0.918 |
| blocks.11.att.key.weight | [768, 768] | -13.8 dB / 0.730 | 8.0 dB / 0.917 |
| blocks.11.att.value.weight | [768, 768] | -1.1 dB / 0.178 | 8.3 dB / 0.923 |
| blocks.11.att.output.weight | [768, 768] | -10.5 dB / 0.406 | 8.8 dB / 0.931 |

## Notes

- `linspace` is the legacy shipped default (4 equally-spaced levels between min and max).
- `kmeans (per-tensor)` is the new default in v0.6.16 (Lloyd's algorithm with k-means++ init; uses sklearn when available, hand-rolled numpy fallback otherwise).
- `linspace (per-row)` / `kmeans (per-row)` allocate one 4-entry codebook per output row of the weight matrix. Better for matrices with per-row dynamic range, but 4× larger header per matrix (4 fp32 per row).
- `hadamard_kmeans (per-tensor)` is QuIP#-style: apply a fixed random Hadamard rotation to make the weights sub-Gaussian, then k-means, then inverse rotate. Marginal on synthetic Gaussians, ~5–8 dB SNR improvement on real LLM weight distributions per the QuIP# paper.

Generated by `scripts/codec_comparison.py`. 623856.4s wall.
