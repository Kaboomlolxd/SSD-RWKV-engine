"""Abstract recurrent model backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rwkv_ssd.runtime.state_cache import RecurrentState


class RecurrentBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, strategy: str, device: str) -> None: ...

    def prefill(self, prompt: str) -> tuple[list[int], Any]:
        raise NotImplementedError(f"{type(self).__name__} does not expose prefill()")

    def step(self, token_id: int, state: Any) -> tuple[int, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not expose step()")

    @abstractmethod
    def decode_text(self, token_ids: list[int]) -> str: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    def get_recurrent_state(self) -> RecurrentState | None:
        """Return the backend's current recurrent state for snapshotting.

        Synthetic backends return the post-decode ``h``; ChatRWKV RWKV-7 returns
        the ``model.state`` list. Returns ``None`` if the backend has not yet
        been driven past prefill or the state isn't accessible.
        """
        return None

    def set_recurrent_state(self, state: RecurrentState) -> None:
        """Restore a previously snapshot state. Default: no-op."""
        return None

    def probe_logits(self, state: RecurrentState | None = None) -> Any | None:
        """Return next-token logits when the backend exposes a probe path.

        This is deliberately optional.  The conformance harness uses it for
        resident/streaming diagnostics, while the normal generation contract
        remains usable by backends that only expose token IDs.
        """
        del state
        return None

    def capability_error(self, capability: str, detail: str | None = None) -> None:
        """Raise the common structured error for an unsupported operation."""
        from rwkv_ssd.runtime.errors import CapabilityNotSupportedError

        raise CapabilityNotSupportedError(capability, type(self).__name__, detail)

    def capabilities(self) -> dict[str, object]:
        """Return declared and observed capabilities for this backend."""
        from rwkv_ssd.backends.capabilities import probe_backend_capabilities

        return probe_backend_capabilities(self)

    def supports_capability(self, capability: str) -> bool:
        """Query a declared engine capability by its public name."""
        from rwkv_ssd.backends.capabilities import capability_supported

        return capability_supported(self, capability)
