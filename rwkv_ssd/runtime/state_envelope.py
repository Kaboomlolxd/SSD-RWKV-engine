"""Versioned state handoff envelopes used by process-level serving.

The recurrent payload is intentionally backend-owned.  The envelope carries
the identities needed to prevent a parked state from being restored into a
different model, tokenizer, or backend implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkv_ssd.runtime.state_cache import RecurrentState


STATE_SERIALIZATION_VERSION = 1


def _fingerprint_files(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    if not found:
        return "missing"
    return digest.hexdigest()


def model_fingerprint(pack_dir: str | Path) -> str:
    """Hash the immutable model/pack files that define tensor identity.

    Runtime packs use ``manifest.json``.  Raw Hugging Face checkpoints such as
    Kimi-K3 do not have a manifest, so hash their config and safetensors shard
    set instead of allowing a state envelope to omit model identity.
    """
    root = Path(pack_dir)
    path = root / "manifest.json"
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        meta = root / "meta.json"
        if meta.is_file():
            digest.update(meta.read_bytes())
        return digest.hexdigest()
    files = [root / "config.json"]
    files.extend(sorted(root.glob("*.safetensors")))
    files = [file for file in files if file.is_file()]
    if not files:
        raise FileNotFoundError(path)
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(pack_dir: str | Path) -> str:
    """Hash local tokenizer assets without requiring optional HF packages."""
    return _fingerprint_files(
        Path(pack_dir),
        (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "rwkv_vocab_v20230424.txt",
            # Kimi-K3's custom tiktoken tokenizer and remote-code assets.
            "tiktoken.model",
            "encoding_k3.py",
            "tokenization_kimi.py",
            "added_tokens.json",
        ),
    )


@dataclass
class StateEnvelope:
    """Backend-specific state plus identities for safe IPC/session restore."""

    backend_kind: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    state_payload: RecurrentState
    last_token_id: int
    serialization_version: int = STATE_SERIALIZATION_VERSION

    def validate_against(
        self,
        *,
        backend_kind: str,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
    ) -> None:
        if int(self.serialization_version) != STATE_SERIALIZATION_VERSION:
            raise ValueError(
                f"unsupported state envelope version {self.serialization_version}"
            )
        expected_backend = str(backend_kind).strip().lower()
        if str(self.backend_kind).strip().lower() != expected_backend:
            raise ValueError(
                f"state backend mismatch: parked={self.backend_kind!r} "
                f"worker={backend_kind!r}"
            )
        if str(self.model_fingerprint) != str(model_fingerprint):
            raise ValueError("state model/pack fingerprint mismatch")
        if str(self.tokenizer_fingerprint) != str(tokenizer_fingerprint):
            raise ValueError("state tokenizer fingerprint mismatch")
        if int(self.last_token_id) != int(self.state_payload.last_token_id):
            raise ValueError("state envelope last_token_id does not match payload")


def make_state_envelope(
    state: RecurrentState,
    *,
    backend_kind: str,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
) -> StateEnvelope:
    return StateEnvelope(
        backend_kind=str(backend_kind),
        model_fingerprint=str(model_fingerprint),
        tokenizer_fingerprint=str(tokenizer_fingerprint),
        state_payload=state,
        last_token_id=int(state.last_token_id),
    )


__all__ = [
    "STATE_SERIALIZATION_VERSION",
    "StateEnvelope",
    "make_state_envelope",
    "model_fingerprint",
    "tokenizer_fingerprint",
]
