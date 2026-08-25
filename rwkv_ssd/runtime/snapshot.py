"""Engine-level state snapshot (P1 #13).

Save/load the recurrent state of an ``InferenceEngine`` mid-generation so a
new session can resume from the same point. Distinct from ``PrefixStateCache``,
which is keyed by prefix text for TTFT amortization on identical system prompts.

File format (one file per snapshot):
  - Magic: ``b"RWS\\x01"`` (RWKV State Snapshot v1)
  - JSON header with engine config (mode, backend, max_layers_in_z,
    decouple_provider_cache, max_provider_cache_layers, model_family,
    version=1, prompt)
  - Per-tensor payload (synthetic: 1 bf16/fp32 tensor h + last_token_id;
    chatrwkv RWKV-7: list of tensors rwkv7_state)

Distinct from ``state_disk_cache.py`` which serializes ``RecurrentState`` for
cross-session prefix reuse under a pack. The snapshot here is a portable
artifact that includes enough engine config to reconstruct a working
``InferenceEngine`` and resume.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

from rwkv_ssd.runtime.state_cache import RecurrentState, clone_rwkv7_state

_SNAP_MAGIC = b"RWS\x01"
_SNAP_VERSION = 2
_DTYPE_BF16 = 1
_DTYPE_FP32 = 2
_DTYPE_F16 = 3
_DTYPE_I32 = 4

_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.bfloat16: _DTYPE_BF16,
    torch.float32: _DTYPE_FP32,
    torch.float16: _DTYPE_F16,
    torch.int32: _DTYPE_I32,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}


def pack_identity(pack_dir: str | Path) -> str:
    """Return the immutable identity used for portable session handoff.

    The manifest commits to tensor layout and, for verified production packs,
    the weight hashes. Hashing it is fast even for very large checkpoints and
    intentionally rejects a differently packed model unless the snapshot is
    explicitly migrated by a future compatibility layer.
    """
    root = Path(pack_dir)
    path = root / "manifest.json"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = [root / "config.json"]
    files.extend(sorted(root.glob("*.safetensors")))
    files = [file for file in files if file.is_file()]
    if not files:
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot_compatibility(
    meta: "SnapshotMeta",
    pack_dir: str | Path,
    *,
    backend: str | None = None,
) -> None:
    """Reject handoff to an incompatible pack/backend.

    Legacy snapshots without an identity remain loadable for backward
    compatibility; newly written engine snapshots always carry one.
    """
    expected = str(meta.extras.get("pack_identity", ""))
    if expected:
        actual = pack_identity(pack_dir)
        if actual != expected:
            raise ValueError(
                "snapshot pack identity mismatch: "
                f"expected {expected[:12]}... got {actual[:12]}..."
            )
    if backend is not None and str(backend).strip().lower() != meta.backend.strip().lower():
        raise ValueError(
            f"snapshot backend mismatch: expected {meta.backend!r} got {backend!r}"
        )


def _as_bool(value: object, default: bool = False) -> bool:
    """Parse JSON booleans without treating ``"false"`` as truthy."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


@dataclass
class SnapshotMeta:
    """Engine / config info carried in a snapshot file."""

    backend: str
    mode: str
    model_family: str
    max_layers_in_z: int = 1
    decouple_provider_cache: bool = True
    max_provider_cache_layers: int = 0
    prompt: str = ""
    last_token_id: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "backend": self.backend,
            "mode": self.mode,
            "model_family": self.model_family,
            "max_layers_in_z": self.max_layers_in_z,
            "decouple_provider_cache": self.decouple_provider_cache,
            "max_provider_cache_layers": self.max_provider_cache_layers,
            "prompt": self.prompt,
            "last_token_id": self.last_token_id,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SnapshotMeta:
        return cls(
            backend=str(d.get("backend", "synthetic")),
            mode=str(d.get("mode", "streaming")),
            model_family=str(d.get("model_family", "rwkv7")),
            max_layers_in_z=int(d.get("max_layers_in_z", 1)),
            decouple_provider_cache=_as_bool(
                d.get("decouple_provider_cache"), default=True
            ),
            max_provider_cache_layers=int(d.get("max_provider_cache_layers", 0)),
            prompt=str(d.get("prompt", "")),
            last_token_id=int(d.get("last_token_id", 0)),
            extras=dict(d.get("extras", {}) or {}),
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_specs(state: RecurrentState) -> list[tuple[list[int], int, int]]:
    """Return (shape, dtype_code, byte_count) for each tensor in state."""
    specs: list[tuple[list[int], int, int]] = []
    if state.h is not None:
        nbytes = state.h.numel() * state.h.element_size()
        specs.append((list(state.h.shape), _DTYPE_TO_CODE[state.h.dtype], nbytes))
    if state.rwkv7_state is not None:
        for t in state.rwkv7_state:
            nbytes = t.numel() * t.element_size()
            specs.append((list(t.shape), _DTYPE_TO_CODE[t.dtype], nbytes))
    return specs


def _state_kind(state: RecurrentState) -> str:
    kinds = [
        state.h is not None,
        state.rwkv7_state is not None,
        state.external_state is not None,
    ]
    if sum(kinds) > 1:
        raise ValueError("recurrent state must contain exactly one state representation")
    if state.h is not None:
        return "h"
    if state.rwkv7_state is not None:
        return "rwkv7"
    if state.external_state is not None:
        return "external"
    return "empty"


def _encode_state(state: RecurrentState) -> bytes:
    parts: list[bytes] = []
    if state.h is not None:
        t = state.h.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            parts.append(t.view(torch.uint16).numpy().tobytes())
        else:
            parts.append(t.numpy().tobytes())
    if state.rwkv7_state is not None:
        for t in state.rwkv7_state:
            tc = t.detach().contiguous().cpu()
            if tc.dtype == torch.bfloat16:
                parts.append(tc.view(torch.uint16).numpy().tobytes())
            else:
                parts.append(tc.numpy().tobytes())
    return b"".join(parts)


def _external_spec(state: RecurrentState) -> tuple[list[int], int, bytes] | None:
    """Encode NumPy-backed states used by rwkv.cpp snapshots."""
    if state.external_state is None:
        return None
    arr = np.asarray(state.external_state, dtype=np.float32)
    return list(arr.shape), _DTYPE_FP32, np.ascontiguousarray(arr).tobytes()


def _decode_state(
    payload: bytes,
    specs: list[tuple[list[int], int, int]],
    *,
    last_token_id: int = 0,
    state_kind: str = "legacy",
) -> RecurrentState:
    h: torch.Tensor | None = None
    rwkv7_state: list[torch.Tensor] | None = None
    cursor = 0
    for idx, (shape, code, nbytes) in enumerate(specs):
        dtype = _CODE_TO_DTYPE[code]
        chunk = payload[cursor : cursor + nbytes]
        if len(chunk) < nbytes:
            raise ValueError(
                f"snapshot payload truncated at tensor {idx}: wanted {nbytes} got {len(chunk)}"
            )
        t = torch.frombuffer(bytearray(chunk), dtype=dtype).reshape(shape).clone()
        if state_kind == "h":
            if idx != 0 or len(specs) != 1:
                raise ValueError("synthetic snapshot must contain exactly one h tensor")
            h = t
        else:
            if rwkv7_state is None:
                rwkv7_state = []
            rwkv7_state.append(t)
        cursor += nbytes
    return RecurrentState(
        h=h,
        rwkv7_state=rwkv7_state,
        last_token_id=last_token_id,
    )


def save_snapshot(
    path: str | Path,
    state: RecurrentState,
    meta: SnapshotMeta,
) -> Path:
    """Write a snapshot file at ``path`` containing ``state`` + ``meta``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = _tensor_specs(state)
    payload = _encode_state(state)
    external = _external_spec(state)
    if external is not None:
        payload += external[2]

    header = {
        "version": _SNAP_VERSION,
        "specs": [
            {"shape": list(s[0]), "dtype_code": s[1], "nbytes": s[2]} for s in specs
        ],
        "state_kind": _state_kind(state),
        "meta": meta.to_dict(),
    }
    if external is not None:
        header["external_spec"] = {
            "shape": external[0],
            "dtype_code": external[1],
            "nbytes": len(external[2]),
        }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(_SNAP_MAGIC)
        fh.write(struct.pack("<I", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(payload)
        payload_size = len(payload)
    payload_sha = hashlib.sha256(payload).hexdigest()
    with tmp.open("ab") as fh:
        fh.write(struct.pack("<Q", payload_size))
        fh.write(payload_sha.encode("utf-8"))
    tmp.replace(path)
    return path


def load_snapshot(path: str | Path) -> tuple[RecurrentState, SnapshotMeta]:
    """Read a snapshot file. Returns (state, meta)."""
    path = Path(path)
    with path.open("rb") as fh:
        header = fh.read(4)
        if header != _SNAP_MAGIC:
            raise ValueError(
                f"bad snapshot magic in {path}: {header!r} (expected {_SNAP_MAGIC!r})"
            )
        (header_len,) = struct.unpack("<I", fh.read(4))
        header_bytes = fh.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        version = int(header.get("version", 0))
        if version not in {1, _SNAP_VERSION}:
            raise ValueError(
                f"unsupported snapshot version {version} (supported 1 and {_SNAP_VERSION})"
            )
        specs_raw = header.get("specs") or []
        specs: list[tuple[list[int], int, int]] = [
            (list(s["shape"]), int(s["dtype_code"]), int(s["nbytes"]))
            for s in specs_raw
        ]
        total = sum(s[2] for s in specs)
        external_raw = header.get("external_spec") or None
        external_nbytes = int(external_raw.get("nbytes", 0)) if external_raw else 0
        total += external_nbytes
        payload_off = fh.tell()
        file_size = fh.seek(0, 2)
        fh.seek(payload_off)
        trailer_len = 8 + 64
        if file_size - payload_off < total + trailer_len:
            raise ValueError(
                f"snapshot payload truncated in {path}: "
                f"wanted {total + trailer_len} bytes got {file_size - payload_off}"
            )
        payload = fh.read(total)
        payload_size_raw = fh.read(8)
        (payload_size,) = struct.unpack("<Q", payload_size_raw)
        if payload_size != total:
            raise ValueError(
                f"snapshot payload size mismatch in {path}: "
                f"header={total} trailer={payload_size}"
            )
        sha_b = fh.read(64)
        sha_expected = sha_b.decode("ascii", errors="replace")
        if sha_expected:
            sha_got = hashlib.sha256(payload).hexdigest()
            if sha_got != sha_expected:
                raise ValueError(
                    f"snapshot payload checksum mismatch in {path}: "
                    f"expected {sha_expected[:12]}... got {sha_got[:12]}..."
                )
    meta_raw = header.get("meta", {})
    if not isinstance(meta_raw, dict):
        raise ValueError("snapshot meta must be a JSON object")
    meta = SnapshotMeta.from_dict(meta_raw)
    state_kind = header.get("state_kind")
    if state_kind is None:
        # v1 snapshots written before state_kind existed.  ChatRWKV's
        # multi-tensor state is unambiguous from its backend; for the rare
        # one-tensor case this also preserves the intended representation.
        state_kind = (
            "rwkv7"
            if meta.backend.strip().lower() == "chatrwkv"
            else ("h" if len(specs) == 1 else "rwkv7")
        )
    if state_kind not in {"h", "rwkv7", "external", "empty"}:
        raise ValueError(f"unsupported snapshot state kind {state_kind!r}")
    if state_kind in {"external", "empty"} and specs:
        raise ValueError(f"snapshot state kind {state_kind!r} cannot contain tensor specs")
    tensor_total = sum(s[2] for s in specs)
    state = _decode_state(
        payload[:tensor_total],
        specs,
        last_token_id=meta.last_token_id,
        state_kind=state_kind,
    )
    if external_raw and external_nbytes:
        shape = tuple(int(x) for x in external_raw.get("shape", []))
        code = int(external_raw.get("dtype_code", _DTYPE_FP32))
        if code != _DTYPE_FP32:
            raise ValueError("unsupported external snapshot dtype")
        external_blob = payload[tensor_total : tensor_total + external_nbytes]
        state.external_state = np.frombuffer(
            external_blob, dtype=np.float32
        ).reshape(shape).copy()
    return state, meta


__all__ = [
    "SnapshotMeta",
    "load_snapshot",
    "pack_identity",
    "save_snapshot",
    "verify_snapshot_compatibility",
]
