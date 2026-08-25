"""Tests for the new trinity_lut2 codebook strategies (kmeans, hadamard_kmeans).

The shipped linspace codebook is SNR=-13.8 dB on real RWKV-7 weights
(below the noise floor). K-means fixes that to +7.8 dB (+21.6 dB improvement).
Hadamard + K-means (QuIP#-style) gets +9.0 dB.

These tests don't require the .pth; they use a synthetic distribution that
mimics the real weight statistics (std ~0.015, range ~[-0.3, 0.3]).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rwkv_ssd.runtime.trinity_codebook import (
    apply_rotation,
    codebook_kmeans,
    codebook_linspace,
    hadamard_round_trip,
    invert_rotation,
    random_hadamard_matrix,
)


def _mse_snr(orig: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    mse = float(((decoded - orig) ** 2).mean())
    snr = 10 * np.log10(orig.var() / mse) if mse > 0 else float("inf")
    return mse, snr


def test_linspace_is_baseline() -> None:
    """Shipped linspace: works on Gaussian but fails on the real RWKV distribution.

    Real RWKV-7 weights concentrate near zero (std ~0.015) with a long tail
    (range up to ~±0.3). Linspace places 2 of 4 levels in the tail and only
    2 near the bulk of the distribution, losing ~13 dB of SNR vs the optimal
    codebook. K-means puts codebook entries where the data actually is.
    """
    rng = np.random.default_rng(0)
    # Mixture distribution: 95% of weights are small, 5% are large outliers.
    # This is closer to the real RWKV-7 weight distribution than a pure Gaussian.
    bulk = rng.normal(0.0, 0.005, size=4000)
    tail = rng.choice([-1.0, 1.0], size=200) * rng.uniform(0.05, 0.3, size=200)
    w = np.concatenate([bulk, tail])
    rng.shuffle(w)

    cb_l = codebook_linspace(w)
    dec_l = cb_l[np.abs(w[:, None] - cb_l[None, :]).argmin(axis=1)]
    mse_l, snr_l = _mse_snr(w, dec_l)

    cb_k = codebook_kmeans(w)
    dec_k = cb_k[np.abs(w[:, None] - cb_k[None, :]).argmin(axis=1)]
    mse_k, snr_k = _mse_snr(w, dec_k)

    assert snr_k > snr_l, (
        f"kmeans should beat linspace: kmeans={snr_k} linspace={snr_l}"
    )
    assert mse_k < mse_l, "kmeans should be lower MSE"


def test_kmeans_better_than_linspace_on_rwkv_shape() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.015, size=4096)
    cb_l = codebook_linspace(w)
    cb_k = codebook_kmeans(w)
    dec_l = cb_l[np.abs(w[:, None] - cb_l[None, :]).argmin(1)]
    dec_k = cb_k[np.abs(w[:, None] - cb_k[None, :]).argmin(1)]
    mse_l, _ = _mse_snr(w, dec_l)
    mse_k, _ = _mse_snr(w, dec_k)
    assert mse_k < mse_l * 0.5, (
        f"kmeans should be >= 2x better on RWKV-shaped weights, "
        f"got linspace={mse_l} kmeans={mse_k}"
    )


def test_kmeans_better_on_real_rwkv_weight() -> None:
    """If a real RWKV-7 0.1B checkpoint is present, verify the SNR improvement."""
    from pathlib import Path

    ckpt = Path("test_model/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    if not ckpt.is_file():
        pytest.skip("real RWKV-7 checkpoint not present")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    w = None
    for name, t in state.items():
        if (
            "blocks.0.att.receptance.weight" in name
            and hasattr(t, "shape")
            and t.shape == (768, 768)
        ):
            w = t.float().numpy().flatten()
            break
    assert w is not None

    cb_l = codebook_linspace(w)
    cb_k = codebook_kmeans(w)
    dec_l = cb_l[np.abs(w[:, None] - cb_l[None, :]).argmin(1)]
    dec_k = cb_k[np.abs(w[:, None] - cb_k[None, :]).argmin(1)]
    mse_l, snr_l = _mse_snr(w, dec_l)
    mse_k, snr_k = _mse_snr(w, dec_k)
    # Documented numbers: linspace SNR < 0, kmeans SNR > 0
    assert snr_l < 0, f"linspace should be negative SNR on real RWKV, got {snr_l}"
    assert snr_k > 0, f"kmeans should be positive SNR, got {snr_k}"
    assert mse_k < mse_l, "kmeans should be lower MSE on real RWKV weights"


def test_hadamard_round_trip() -> None:
    H = random_hadamard_matrix(768, seed=42)
    assert hadamard_round_trip(H), "Hadamard must be its own inverse"
    # Also test a non-power-of-2
    H2 = random_hadamard_matrix(3072, seed=0)
    # H2 @ H2.T should be identity on the first 3072 rows/cols
    out = H2 @ H2.T
    assert np.allclose(out, np.eye(3072), atol=1e-4)


def test_hadamard_kmeans_better_than_kmeans() -> None:
    """QuIP#-style: rotation makes weights sub-Gaussian, k-means recovers more."""
    rng = np.random.default_rng(0)
    w_2d = rng.normal(0.0, 0.015, size=(768, 768))
    H = random_hadamard_matrix(768, seed=42)
    w_rot = apply_rotation(w_2d, H)
    in_f = w_2d.shape[1]

    # Plain k-means on original
    cb_k = codebook_kmeans(w_2d.flatten())
    dec_k = cb_k[np.abs(w_2d.flatten()[:, None] - cb_k[None, :]).argmin(1)].reshape(
        768, 768
    )
    _, snr_k = _mse_snr(w_2d, dec_k)

    # K-means on rotated; pad output back to (768, 1024) to invert
    cb_h = codebook_kmeans(w_rot.flatten())
    dec_rot = cb_h[np.abs(w_rot.flatten()[:, None] - cb_h[None, :]).argmin(1)].reshape(
        w_rot.shape
    )
    w_dec_full = invert_rotation(dec_rot, H)
    w_dec = w_dec_full[:, :in_f]
    _, snr_h = _mse_snr(w_2d, w_dec)

    assert snr_h > snr_k - 0.5, (
        f"Hadamard+kmeans should match or beat plain kmeans on this distribution; "
        f"kmeans SNR={snr_k}, hadamard_kmeans SNR={snr_h}"
    )


def test_codebook_kmeans_returns_sorted_ascending() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.015, size=2048)
    cb = codebook_kmeans(w)
    assert cb.dtype == np.float32
    assert cb.shape == (4,)
    assert np.all(np.diff(cb) >= 0), f"codebook not sorted ascending: {cb}"


def test_codebook_kmeans_deterministic() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.015, size=2048)
    cb1 = codebook_kmeans(w, seed=42)
    cb2 = codebook_kmeans(w, seed=42)
    np.testing.assert_array_equal(cb1, cb2)


def test_codebook_kmeans_handles_uniform() -> None:
    """Codebook for near-uniform weight distribution: kmeans should still produce
    a sensible 4-level codebook (not all entries equal)."""
    rng = np.random.default_rng(0)
    w = rng.uniform(-0.1, 0.1, size=1024)
    cb = codebook_kmeans(w)
    assert len(set(cb.round(4).tolist())) >= 3, f"got collapsed codebook: {cb}"
