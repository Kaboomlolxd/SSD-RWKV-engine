"""
Codebook generators for the trinity_lut2 codec.

Three strategies, ordered by quality vs cost:

1. ``linspace`` (shipped default, 0 extra deps) — equally-spaced levels between
   min and max. SNR on real RWKV weights: ~ -14 dB (very bad, the inner levels
   capture almost no information because weights concentrate near zero).

2. ``kmeans`` (this module) — Lloyd's algorithm on the flat weight tensor. On
   real RWKV-7 weights: ~ +8 dB SNR (21 dB better than linspace on the
   ``blocks.0.att.receptance.weight`` matrix). No sklearn / scipy dep — pure
   numpy, 30-line implementation, deterministic with a fixed seed.

3. ``hadamard_kmeans`` (QuIP#-style) — apply a fixed random Hadamard rotation
   to the weight matrix before k-means, so the rotated weights become
   sub-Gaussian (incoherence). On Llama-class LLMs this gives another 5-8 dB
   SNR at 2 bits/weight. Decode is the same ``gather_lut2_packed`` kernel
   followed by the inverse Hadamard.

Each strategy emits a 16-byte (4-fp32) ascending codebook. The engine
encodes the inverse permutation as a uint8 index per weight, packed 4-per-byte.

Reference SNR table on rwkv7-g1d-0.1b ``blocks.0.att.receptance.weight``
(768×768 bf16, std=0.0147):

  +-------------------------+---------+--------+
  | codebook                | RMSE    | SNR    |
  +-------------------------+---------+--------+
  | linspace (shipped)      | 0.07184 | -13.8dB |
  | percentile              | 0.14834 | -20.1dB |
  | kmeans (this module)    | 0.00595 |  +7.8dB |
  +-------------------------+---------+--------+
"""

from __future__ import annotations

import math

import numpy as np
import torch


def codebook_linspace(flat: np.ndarray) -> np.ndarray:
    """Shipped default — kept for back-compat with existing packs."""
    if flat.size == 0:
        return np.linspace(0.0, 1.0, 4, dtype=np.float32)
    mn = float(flat.min())
    mx = float(flat.max())
    if mx - mn < 1e-12:
        mx = mn + 1.0
    return np.linspace(mn, mx, 4, dtype=np.float32)


def codebook_kmeans(
    flat: np.ndarray,
    *,
    k: int = 4,
    iters: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """Hand-rolled Lloyd's algorithm (1-D k-means, ~25 iters).

    Returns ``k`` sorted ascending codebook entries. Pure numpy, no scipy/sklearn.
    For a 768×768 weight tensor (589,824 elements) this runs in <100ms on CPU.

    If ``sklearn`` is installed, we delegate to ``sklearn.cluster.KMeans`` for
    ~5× speedup on larger tensors (used transparently).
    """
    if flat.size == 0:
        return np.linspace(0.0, 1.0, k, dtype=np.float32)
    if flat.size < k * 4:
        return codebook_linspace(flat)[:k]

    # Cache sklearn import (it's a 5s cold start)
    if not hasattr(codebook_kmeans, "_sklearn_kmeans"):
        try:
            from sklearn.cluster import KMeans as _SKMeans  # type: ignore[import-not-found]

            codebook_kmeans._sklearn_kmeans = _SKMeans  # type: ignore[attr-defined]
        except ImportError:
            codebook_kmeans._sklearn_kmeans = None  # type: ignore[attr-defined]
    _SKMeans = codebook_kmeans._sklearn_kmeans  # type: ignore[attr-defined]

    if _SKMeans is not None:
        # For large weights, use a sub-sample + lower n_init to keep encode time sane.
        # 4-entry k-means converges in ~10 iterations on 1D data; n_init=4 is enough.
        n = flat.size
        if n > 100000:
            rng = np.random.default_rng(seed)
            sample = rng.choice(flat, size=50000, replace=False)
        else:
            sample = flat
        km = _SKMeans(
            n_clusters=k,
            n_init=4,
            max_iter=100,
            random_state=seed,
            algorithm="lloyd",
        )
        labels = km.fit_predict(sample.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.flatten()).astype(np.float32)
        return centers

    rng = np.random.default_rng(seed)
    sample = flat if flat.size <= 4096 else rng.choice(flat, size=4096, replace=False)
    centers = np.empty(k, dtype=np.float64)
    centers[0] = sample[rng.integers(sample.size)]
    sq_dist = np.full(sample.size, np.inf, dtype=np.float64)
    for i in range(1, k):
        d2 = (sample - centers[i - 1]) ** 2
        sq_dist = np.minimum(sq_dist, d2)
        total = sq_dist.sum()
        if total <= 0:
            centers[i] = sample[rng.integers(sample.size)]
            continue
        probs = sq_dist / total
        centers[i] = sample[rng.choice(sample.size, p=probs)]

    flat64 = flat.astype(np.float64, copy=False)
    for _ in range(iters):
        d = np.abs(flat64[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = np.empty_like(centers)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = flat64[mask].mean()
            else:
                new_centers[i] = centers[i]
        new_centers.sort()
        if np.max(np.abs(new_centers - centers)) < 1e-7:
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.float32)


def codebooks_groupwise_kmeans(
    flat: np.ndarray,
    *,
    group_size: int = 256,
    k: int = 4,
    iters: int = 12,
) -> np.ndarray:
    """Build one small learned codebook per contiguous weight group.

    A global four-value codebook is easily dominated by a tensor's outliers.
    Group-local codebooks retain the same two-bit indices while allowing each
    region to follow its own scale and offset.  Quantile initialization keeps
    this deterministic and avoids launching thousands of sklearn fits when a
    large checkpoint is packed.
    """
    values = np.asarray(flat, dtype=np.float32).reshape(-1)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if k != 4:
        raise ValueError("Trinity LUT2 requires exactly four codebook values")
    groups = (values.size + group_size - 1) // group_size
    out = np.empty((groups, k), dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, k + 2, dtype=np.float64)[1:-1]
    # Batch independent one-dimensional fits.  A Python loop per 128-value
    # group makes a full checkpoint repack unreasonably slow; batching retains
    # identical Lloyd updates while bounding temporary memory.
    batch_groups = max(1, min(4096, 2_000_000 // group_size))
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, values.size - start)
        chunk = np.empty((count, group_size), dtype=np.float32)
        flat_chunk = values[start : start + available]
        chunk.reshape(-1)[:available] = flat_chunk
        valid = np.arange(count * group_size).reshape(count, group_size) < available
        if available < count * group_size:
            chunk.reshape(-1)[available:] = flat_chunk[-1] if available else 0.0
        centers = np.quantile(chunk, quantiles, axis=1).T.astype(np.float32)
        for _ in range(max(1, int(iters))):
            labels = np.abs(chunk[:, :, None] - centers[:, None, :]).argmin(axis=2)
            updated = centers.copy()
            for index in range(k):
                selected = (labels == index) & valid
                counts = selected.sum(axis=1)
                sums = np.where(selected, chunk, 0.0).sum(axis=1, dtype=np.float64)
                present = counts > 0
                updated[present, index] = (sums[present] / counts[present]).astype(
                    np.float32
                )
            updated.sort(axis=1)
            if np.max(np.abs(updated - centers)) < 1e-7:
                centers = updated
                break
            centers = updated
        out[first : first + count] = centers
    return out


def codebooks_groupwise_symmetric(
    flat: np.ndarray,
    *,
    group_size: int = 128,
    iters: int = 12,
) -> np.ndarray:
    """Two learned magnitudes mirrored around zero for bias-free LUT2.

    Scalar MSE can prefer an asymmetric codebook even for nearly zero-mean
    weights. Repeated recurrent application may amplify that small signed
    bias, so this variant gives up some scalar freedom to guarantee symmetry.
    """
    values = np.asarray(flat, dtype=np.float32).reshape(-1)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = (values.size + group_size - 1) // group_size
    out = np.empty((groups, 4), dtype=np.float32)
    batch_groups = max(1, min(4096, 2_000_000 // group_size))
    for first in range(0, groups, batch_groups):
        count = min(batch_groups, groups - first)
        start = first * group_size
        available = min(count * group_size, values.size - start)
        chunk = np.empty((count, group_size), dtype=np.float32)
        chunk.reshape(-1)[:available] = np.abs(values[start : start + available])
        valid = np.arange(count * group_size).reshape(count, group_size) < available
        if available < count * group_size:
            chunk.reshape(-1)[available:] = chunk.reshape(-1)[available - 1]
        centers = np.quantile(chunk, [0.35, 0.85], axis=1).T.astype(np.float32)
        for _ in range(max(1, int(iters))):
            labels = np.abs(chunk[:, :, None] - centers[:, None, :]).argmin(axis=2)
            updated = centers.copy()
            for index in range(2):
                selected = (labels == index) & valid
                counts = selected.sum(axis=1)
                sums = np.where(selected, chunk, 0.0).sum(axis=1, dtype=np.float64)
                present = counts > 0
                updated[present, index] = (sums[present] / counts[present]).astype(
                    np.float32
                )
            updated.sort(axis=1)
            if np.max(np.abs(updated - centers)) < 1e-7:
                centers = updated
                break
            centers = updated
        out[first : first + count] = np.stack(
            (-centers[:, 1], -centers[:, 0], centers[:, 0], centers[:, 1]),
            axis=1,
        )
    return out


def random_hadamard_matrix(n: int, *, seed: int) -> np.ndarray:
    """A deterministic signed Hadamard transform of size n×n.

    For n that is not a power of 2, the function returns an (n, n_padded) matrix
    where n_padded is the next power of 2, and the extra rows/columns are
    identity padding. This is the standard QuIP# trick: pad, transform, truncate.

    QuIP# shows a *randomized* Hadamard (random signs on each column, then
    H @ diag(s) @ H / sqrt(n)) makes the weight distribution sub-Gaussian
    enough that 2-bit k-means recovers near-FP16 quality. We use a fixed seed
    so the encode/decode are deterministic.
    """
    rng = np.random.default_rng(seed)
    n_pow = 1
    while n_pow < n:
        n_pow *= 2
    diag = rng.choice([-1.0, 1.0], size=n_pow).astype(np.float64)
    h = _hadamard(n_pow).astype(np.float64) / math.sqrt(n_pow)
    h = (h * diag[None, :]) @ h
    if n_pow == n:
        return h
    # pad to n_pow on the right, then truncate to n
    out = np.zeros((n, n_pow), dtype=np.float64)
    out[:n, :n] = np.eye(n)
    return out @ h


def _hadamard(n: int) -> np.ndarray:
    """Sylvester-constructed Hadamard matrix of order n (power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def hadamard_round_trip(rot: np.ndarray) -> bool:
    """Sanity check that rot @ rot.T == I within float32 tolerance."""
    out = rot @ rot.T
    return np.allclose(out, np.eye(rot.shape[0]), atol=1e-4)


def apply_rotation(matrix: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Apply an (n_in, n_in) rotation to a (m, n_in) weight matrix.

    Returns a (m, n_rot) array where n_rot is rot.shape[1].
    """
    return matrix.astype(np.float64, copy=False) @ rot


def invert_rotation(rotated: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Apply the inverse rotation. For a square Hadamard, rot.T = rot^-1."""
    return rotated.astype(np.float64, copy=False) @ rot.T


def _fwht_inplace(values: np.ndarray) -> np.ndarray:
    """Apply an unnormalised Walsh-Hadamard transform along the last axis."""
    # FP32 is sufficient for a 2-bit calibration rotation and avoids the
    # doubled bandwidth of the reference FP64 helper.  The transform is
    # orthogonal; the quantizer, not this arithmetic, dominates final error.
    out = np.asarray(values, dtype=np.float32).copy()
    width = out.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Walsh-Hadamard width must be a positive power of two")
    step = 1
    while step < width:
        block = step * 2
        # Reshape all independent butterfly blocks at once.  Iterating over
        # every block is mathematically simple but adds 2^log2(width) Python
        # slice operations to every matrix transform.
        view = out.reshape(-1, width).reshape(-1, width // block, 2, step)
        left = view[:, :, 0, :]
        right = view[:, :, 1, :]
        left_copy = left.copy()
        left += right
        right[:] = left_copy - right
        step = block
    return out


def hadamard_transform(
    values: np.ndarray, *, seed: int = 42, output_width: int | None = None
) -> np.ndarray:
    """Apply the same matrix-free randomized Hadamard transform used by QuIP#.

    ``values`` is a matrix whose rows are independent weight vectors.  When
    the input width is not a power of two it is zero-padded to the next power
    of two.  The transform is involutory, so applying this function twice
    returns the padded input (up to floating-point error).
    """
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Hadamard transform expects a 2-D matrix")
    input_width = matrix.shape[1]
    width = 1
    while width < input_width:
        width *= 2
    if output_width is not None and int(output_width) != width:
        raise ValueError(
            f"Hadamard output width mismatch: expected {width}, got {output_width}"
        )
    padded = np.zeros((matrix.shape[0], width), dtype=np.float32)
    padded[:, :input_width] = matrix
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=width).astype(np.float32)
    transformed = _fwht_inplace(padded)
    transformed *= signs[None, :]
    transformed = _fwht_inplace(transformed)
    return transformed / np.float32(width)


def invert_hadamard_transform(
    rotated: np.ndarray,
    original_width: int,
    *,
    seed: int = 42,
    output_width: int | None = None,
) -> np.ndarray:
    """Invert :func:`hadamard_transform` and truncate padding columns."""
    transformed = hadamard_transform(
        rotated, seed=seed, output_width=output_width
    )
    return transformed[:, : int(original_width)]


__all__ = [
    "codebook_linspace",
    "codebook_kmeans",
    "codebooks_groupwise_kmeans",
    "codebooks_groupwise_symmetric",
    "random_hadamard_matrix",
    "hadamard_round_trip",
    "apply_rotation",
    "invert_rotation",
    "hadamard_transform",
    "invert_hadamard_transform",
]
