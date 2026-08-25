"""Comprehensive codec comparison: per-tensor MSE / RMSE / SNR / max-error
for each codebook strategy on real RWKV-7 0.1B weights.

Reports a markdown table that can be pasted into the docs.

Usage: python scripts/codec_comparison.py [--out table.md]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rwkv_ssd.runtime.trinity_codebook import (
    apply_rotation,
    codebook_kmeans,
    codebook_linspace,
    invert_rotation,
    random_hadamard_matrix,
)


def quantize_linspace(w_flat: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    idx = np.abs(w_flat[:, None] - codebook[None, :]).argmin(axis=1)
    return codebook[idx]


def quantize_per_row_linspace(w_2d: np.ndarray) -> np.ndarray:
    """Per-row linspace codebook (one per row)."""
    out = np.empty_like(w_2d)
    for r in range(w_2d.shape[0]):
        mn, mx = w_2d[r].min(), w_2d[r].max()
        cb = (
            np.linspace(mn, mx, 4, dtype=np.float32)
            if mx > mn
            else np.zeros(4, dtype=np.float32) + mn
        )
        out[r] = quantize_linspace(w_2d[r], cb)
    return out


def quantize_per_row_kmeans(w_2d: np.ndarray) -> np.ndarray:
    """Per-row K-means codebook (one per row)."""
    out = np.empty_like(w_2d)
    for r in range(w_2d.shape[0]):
        cb = codebook_kmeans(w_2d[r])
        out[r] = quantize_linspace(w_2d[r], cb)
    return out


def quantize_hadamard_kmeans(w_2d: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """QuIP#-style: rotate, K-means, inverse rotate. Round to original shape."""
    H = random_hadamard_matrix(w_2d.shape[0], seed=seed)
    w_rot = apply_rotation(w_2d, H)
    cb = codebook_kmeans(w_rot.flatten())
    dec_rot = quantize_linspace(w_rot.flatten(), cb).reshape(w_rot.shape)
    w_dec = invert_rotation(dec_rot, H)
    return w_dec[:, : w_2d.shape[1]]


def metrics(orig: np.ndarray, dec: np.ndarray) -> dict[str, float]:
    diff = (dec - orig).astype(np.float64)
    mse = float((diff**2).mean())
    rmse = mse**0.5
    var = float(orig.astype(np.float64).var())
    snr = 10 * np.log10(var / mse) if mse > 0 else float("inf")
    max_err = float(np.abs(diff).max())
    mean_abs = float(np.abs(orig).mean())
    cos_sim = float(
        np.dot(orig.flatten().astype(np.float64), dec.flatten().astype(np.float64))
        / (
            np.linalg.norm(orig.flatten().astype(np.float64))
            * np.linalg.norm(dec.flatten().astype(np.float64))
            + 1e-12
        )
    )
    return {
        "rmse": rmse,
        "mse": mse,
        "snr_db": snr,
        "max_err": max_err,
        "rel_rmse": rmse / mean_abs if mean_abs > 0 else 0.0,
        "cos_sim": cos_sim,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("bench/results/codec_comparison.md"),
        help="Output markdown path",
    )
    p.add_argument(
        "--limit-layers",
        type=int,
        default=12,
        help="How many of each layer to sample (12 covers all blocks)",
    )
    p.add_argument(
        "--only",
        default=None,
        choices=["att", "ffn", "head", None],
        help="Only quantize this tensor family (att / ffn / head). Default: both att and ffn.",
    )
    p.add_argument(
        "--with-per-row",
        action="store_true",
        help="Also include per-row codebook strategies (slower, better for big matmuls)",
    )
    p.add_argument(
        "--with-hadamard",
        action="store_true",
        help="Also include Hadamard+K-means (QuIP#-style, slow on CPU)",
    )
    args = p.parse_args()

    ckpt = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    if not ckpt.is_file():
        print("Real RWKV-7 checkpoint not present; cannot run comparison.")
        return

    print(f"Loading {ckpt} ...")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)

    # Collect square weights of meaningful size
    weights = []
    for name, t in state.items():
        if not (
            hasattr(t, "shape")
            and len(t.shape) == 2
            and t.shape[0] == t.shape[1]
            and t.shape[0] >= 64
            and "weight" in name
        ):
            continue
        if args.only == "att" and "att" not in name:
            continue
        if args.only == "ffn" and "ffn" not in name:
            continue
        if args.only == "head" and "head" not in name:
            continue
        if "emb" in name or "ln" in name or "lora" in name:
            continue
        # Limit to blocks 0..N-1 for speed (--limit-layers defaults to 12 = all)
        if "blocks." in name:
            try:
                layer_id = int(name.split("blocks.")[1].split(".")[0])
            except (IndexError, ValueError):
                layer_id = 0
            if layer_id >= args.limit_layers:
                continue
        weights.append((name, t.float().numpy()))

    print(f"Found {len(weights)} square weights in the checkpoint")

    # Warmup: do a tiny kmeans so sklearn is imported once and cached
    print("Warming up kmeans (sklearn cold import ~5s, one-time) ...")
    import time as _t

    _t0 = _t.perf_counter()
    codebook_kmeans(np.random.randn(1000).astype(np.float32))
    print(f"  warmup done in {_t.perf_counter() - _t0:.1f}s")

    # Group by name family
    families = defaultdict(list)
    for name, W in weights:
        family = name.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in name else name
        # Use a coarser family: blocks.X.att/ffn
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            fam = f"blocks.{parts[1]}.{parts[2]}"
        else:
            fam = parts[0]
        families[fam].append((name, W))

    # Per-weight, per-strategy metrics
    strategies = {
        "linspace (per-tensor)": lambda w: quantize_linspace(
            w.flatten(), codebook_linspace(w.flatten())
        ).reshape(w.shape),
        "kmeans (per-tensor)": lambda w: quantize_linspace(
            w.flatten(), codebook_kmeans(w.flatten())
        ).reshape(w.shape),
    }
    if args.with_per_row:
        strategies["linspace (per-row)"] = quantize_per_row_linspace
        strategies["kmeans (per-row)"] = quantize_per_row_kmeans
    if args.with_hadamard:
        strategies["hadamard_kmeans (per-tensor)"] = quantize_hadamard_kmeans

    rows = []
    timings = {name: [] for name in strategies}
    for wname, W in weights:
        row = {"name": wname, "shape": str(list(W.shape))}
        for strat_name, strat_fn in strategies.items():
            t0 = time.perf_counter()
            try:
                decoded = strat_fn(W.copy())
            except Exception as exc:
                decoded = W
                print(f"  WARN: {strat_name} failed on {wname}: {exc}", file=sys.stderr)
            timings[strat_name].append(time.perf_counter() - t0)
            m = metrics(W, decoded)
            row[strat_name] = m
        rows.append(row)

    # Aggregate per strategy: mean SNR, mean RMSE, mean rel RMSE
    summary = {}
    for strat_name in strategies:
        snrs = [r[strat_name]["snr_db"] for r in rows]
        rmses = [r[strat_name]["rmse"] for r in rows]
        rel_rmses = [r[strat_name]["rel_rmse"] for r in rows]
        max_errs = [r[strat_name]["max_err"] for r in rows]
        cos_sims = [r[strat_name]["cos_sim"] for r in rows]
        ts = timings[strat_name]
        summary[strat_name] = {
            "mean_snr": statistics.mean(snrs),
            "median_snr": statistics.median(snrs),
            "min_snr": min(snrs),
            "max_snr": max(snrs),
            "mean_rmse": statistics.mean(rmses),
            "mean_rel_rmse": statistics.mean(rel_rmses),
            "mean_max_err": statistics.mean(max_errs),
            "mean_cos_sim": statistics.mean(cos_sims),
            "median_encode_ms": statistics.median(ts) * 1000,
        }

    # Print to stdout
    print()
    print("=" * 100)
    print(f"Codec comparison on {len(weights)} real RWKV-7 0.1B weights ({ckpt.name})")
    print("=" * 100)
    print()
    print(
        f"{'strategy':<30s}  {'mean SNR':>10s}  {'median SNR':>11s}  {'min SNR':>8s}  {'mean RMSE':>10s}  {'rel RMSE':>10s}  {'mean cos':>9s}  {'encode ms':>10s}"
    )
    print("-" * 110)
    for strat_name in strategies:
        s = summary[strat_name]
        print(
            f"{strat_name:<30s}  {s['mean_snr']:>8.2f} dB  {s['median_snr']:>9.2f} dB  "
            f"{s['min_snr']:>6.2f} dB  {s['mean_rmse']:>10.5f}  {s['mean_rel_rmse']:>9.2%}  "
            f"{s['mean_cos_sim']:>9.4f}  {s['median_encode_ms']:>8.2f}"
        )

    # Per-weight table
    print()
    print("=" * 100)
    print("Per-weight results (SNR in dB, RMSE, cosine sim)")
    print("=" * 100)
    strat_cols = list(strategies.keys())
    header = f"{'name':<55s} " + " ".join(f"{s[:14]:>14s}" for s in strat_cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for s in strat_cols:
            m = r[s]
            cells.append(f"{m['snr_db']:>6.1f}dB {m['cos_sim']:.3f}")
        print(f"{r['name']:<55s} " + " ".join(f"{c:>14s}" for c in cells))

    # Write markdown
    md_lines = []
    md_lines.append("# Codec comparison on real RWKV-7 0.1B weights")
    md_lines.append("")
    md_lines.append(
        f"Source: `{ckpt.name}`, {len(weights)} square weights, 2 bits/weight."
    )
    md_lines.append("")
    md_lines.append(
        "Each weight is quantized once per strategy, decoded back, and compared to the original FP32 source."
    )
    md_lines.append("Lower MSE / higher SNR / higher cosine sim = better.")
    md_lines.append("")
    md_lines.append("## Aggregate (mean over all " + str(len(weights)) + " weights)")
    md_lines.append("")
    md_lines.append(
        "| Strategy | mean SNR (dB) | median SNR (dB) | min SNR (dB) | mean RMSE | mean rel RMSE | mean cos sim | median encode (ms) |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for strat_name in strategies:
        s = summary[strat_name]
        md_lines.append(
            f"| {strat_name} | {s['mean_snr']:.2f} | {s['median_snr']:.2f} | "
            f"{s['min_snr']:.2f} | {s['mean_rmse']:.5f} | {s['mean_rel_rmse']:.2%} | "
            f"{s['mean_cos_sim']:.4f} | {s['median_encode_ms']:.2f} |"
        )
    md_lines.append("")
    md_lines.append("## Per-weight results")
    md_lines.append("")
    header_md = "| name | shape | " + " | ".join(strat_cols) + " |"
    sep_md = "|---|---" * (2 + len(strat_cols)) + "|"
    md_lines.append(header_md)
    md_lines.append(sep_md)
    for r in rows:
        cells = [r["name"], r["shape"]]
        for s in strat_cols:
            m = r[s]
            cells.append(f"{m['snr_db']:.1f} dB / {m['cos_sim']:.3f}")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append(
        "- `linspace` is the legacy shipped default (4 equally-spaced levels between min and max)."
    )
    md_lines.append(
        "- `kmeans (per-tensor)` is the new default in v0.6.16 (Lloyd's algorithm with k-means++ init; uses sklearn when available, hand-rolled numpy fallback otherwise)."
    )
    md_lines.append(
        "- `linspace (per-row)` / `kmeans (per-row)` allocate one 4-entry codebook per output row of the weight matrix. Better for matrices with per-row dynamic range, but 4× larger header per matrix (4 fp32 per row)."
    )
    md_lines.append(
        "- `hadamard_kmeans (per-tensor)` is QuIP#-style: apply a fixed random Hadamard rotation to make the weights sub-Gaussian, then k-means, then inverse rotate. Marginal on synthetic Gaussians, ~5–8 dB SNR improvement on real LLM weight distributions per the QuIP# paper."
    )
    md_lines.append("")
    md_lines.append(
        f"Generated by `scripts/codec_comparison.py`. {time.perf_counter() - 0:.1f}s wall."
    )

    args.out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
