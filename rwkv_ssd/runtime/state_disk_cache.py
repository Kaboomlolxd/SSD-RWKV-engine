"""Persist prefix recurrent state under ``<pack>/.state_cache/`` (optional M2.5 extension)."""

from __future__ import annotations

import hashlib
import mmap
import os
import struct
from pathlib import Path

import numpy as np
import torch

from rwkv_ssd.runtime.state_cache import RecurrentState, clone_rwkv7_state

_STATE_MAGIC = b"RSC\x01"
_STATE_VERSION = 1
_DTYPE_BF16 = 1
_DTYPE_FP32 = 2
_EXTERNAL_MAGIC = b"RSE\x01"


def _cache_dir(pack_dir: Path) -> Path:
    return Path(pack_dir) / ".state_cache"


def _path_for(pack_dir: Path, prefix_key: str) -> Path:
    digest = hashlib.sha256(prefix_key.encode("utf-8")).hexdigest()
    return _cache_dir(pack_dir) / f"{digest}.bin"


def _external_path(pack_dir: Path, prefix_key: str) -> Path:
    return _path_for(pack_dir, prefix_key).with_suffix(".external.bin")


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return _DTYPE_BF16
    if dtype == torch.float32:
        return _DTYPE_FP32
    raise ValueError(f"unsupported state tensor dtype: {dtype}")


def _dtype_from_code(code: int) -> torch.dtype:
    if code == _DTYPE_BF16:
        return torch.bfloat16
    if code == _DTYPE_FP32:
        return torch.float32
    raise ValueError(f"bad state dtype code: {code}")


class StateDiskCache:
    """Serialize ``RecurrentState.rwkv7_state`` for cross-session reuse."""

    def __init__(self, pack_dir: Path) -> None:
        self._pack_dir = Path(pack_dir)
        self._dir = _cache_dir(self._pack_dir)

    def load(self, prefix_key: str) -> RecurrentState | None:
        path = _path_for(self._pack_dir, prefix_key)
        if path.is_file():
            try:
                with path.open("rb") as fh:
                    header = fh.read(16)
                    if len(header) < 16 or header[:4] != _STATE_MAGIC:
                        raise ValueError("not a legacy recurrent state cache")
                    ver, last_id, n_tensors = struct.unpack("<III", header[4:16])
                    if ver != _STATE_VERSION or n_tensors <= 0:
                        return None
                    specs: list[tuple[list[int], torch.dtype, int]] = []
                    for _ in range(n_tensors):
                        ndim_b = fh.read(4)
                        if len(ndim_b) < 4:
                            return None
                        (ndim,) = struct.unpack("<I", ndim_b)
                        shape_b = fh.read(8 * ndim)
                        if len(shape_b) < 8 * ndim:
                            return None
                        shape = list(struct.unpack(f"<{ndim}q", shape_b))
                        tail = fh.read(12)
                        if len(tail) < 12:
                            return None
                        code, nbytes = struct.unpack("<IQ", tail)
                        specs.append((shape, _dtype_from_code(code), nbytes))
                    payload_off = fh.tell()
                    with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        blob = bytes(memoryview(mm)[payload_off:])
                    tensors: list[torch.Tensor] = []
                    off = 0
                    for shape, dt, nbytes in specs:
                        chunk = blob[off : off + nbytes]
                        off += nbytes
                        if len(chunk) != nbytes:
                            return None
                        t = torch.frombuffer(bytearray(chunk), dtype=dt).reshape(shape).clone()
                        tensors.append(t)
                    return RecurrentState(last_token_id=last_id, rwkv7_state=tensors)
            except (OSError, ValueError, RuntimeError, OverflowError, struct.error):
                pass
        external = _external_path(self._pack_dir, prefix_key)
        try:
            raw = external.read_bytes()
            if len(raw) < 20 or raw[:4] != _EXTERNAL_MAGIC:
                raise ValueError("not an external state cache")
            last_id, ndim = struct.unpack_from("<II", raw, 4)
            off = 12
            shape = struct.unpack_from(f"<{ndim}q", raw, off)
            off += 8 * ndim
            (nbytes,) = struct.unpack_from("<Q", raw, off)
            off += 8
            payload = raw[off : off + nbytes]
            if len(payload) != nbytes:
                raise ValueError("truncated external state cache")
            state = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
            return RecurrentState(last_token_id=last_id, external_state=state)
        except (OSError, ValueError, struct.error):
            pass
        # Sequence states use the versioned snapshot container.  Keep the
        # legacy RSC/RSE readers above unchanged so existing RWKV prefix
        # entries remain readable.
        try:
            from rwkv_ssd.runtime.snapshot import load_snapshot

            restored, _meta = load_snapshot(path)
            if restored.sequence_state is not None:
                return restored
        except (OSError, ValueError, RuntimeError, OverflowError, struct.error):
            pass
        return None

    def store(self, prefix_key: str, state: RecurrentState) -> None:
        if state.sequence_state is not None:
            from rwkv_ssd.runtime.snapshot import SnapshotMeta, save_snapshot

            target = _path_for(self._pack_dir, prefix_key)
            self._dir.mkdir(parents=True, exist_ok=True)
            save_snapshot(
                target,
                state,
                SnapshotMeta(
                    backend="sequence",
                    mode="prefix_cache",
                    model_family=state.sequence_state.kind,
                    last_token_id=int(state.last_token_id),
                    extras={
                        "sequence_position": int(state.sequence_state.position),
                        "sequence_context_limit": state.sequence_state.context_limit,
                    },
                ),
            )
            return
        if state.external_state is not None:
            arr = np.ascontiguousarray(np.asarray(state.external_state, dtype=np.float32))
            header = _EXTERNAL_MAGIC + struct.pack(
                "<II", int(state.last_token_id), arr.ndim
            )
            header += struct.pack(f"<{arr.ndim}q", *arr.shape)
            header += struct.pack("<Q", arr.nbytes)
            target = _external_path(self._pack_dir, prefix_key)
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(header + arr.tobytes())
            os.replace(tmp, target)
            return
        if state.rwkv7_state is None:
            return
        tensors = clone_rwkv7_state(state.rwkv7_state)
        meta_parts: list[bytes] = []
        payload_parts: list[bytes] = []
        for t in tensors:
            # State tensors may live on CUDA/XPU after generation.  NumPy
            # views are CPU-only, so make the serialization boundary explicit
            # instead of relying on the caller to keep recurrent state on CPU.
            tc = t.detach().contiguous().cpu()
            dt = tc.dtype
            nbytes = tc.numel() * tc.element_size()
            meta_parts.append(struct.pack("<I", len(tc.shape)))
            meta_parts.append(struct.pack(f"<{len(tc.shape)}q", *tc.shape))
            meta_parts.append(struct.pack("<IQ", _dtype_code(dt), nbytes))
            if dt == torch.bfloat16:
                payload_parts.append(tc.view(torch.uint16).numpy().tobytes())
            else:
                payload_parts.append(tc.numpy().tobytes())
        meta = b"".join(meta_parts)
        payload = b"".join(payload_parts)
        header = _STATE_MAGIC + struct.pack(
            "<III",
            _STATE_VERSION,
            int(state.last_token_id),
            len(tensors),
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        target = _path_for(self._pack_dir, prefix_key)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(header + meta + payload)
        os.replace(tmp, target)

    def clear(self) -> None:
        if not self._dir.is_dir():
            return
        for path in self._dir.glob("*.bin"):
            try:
                path.unlink()
            except OSError:
                pass
