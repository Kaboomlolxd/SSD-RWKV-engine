"""Streaming rwkv.cpp adapter.

The concrete implementation lives on :class:`RWKVCppBackend` so resident and
pack-backed inference share one tokenizer/state/ctypes lifecycle.  This name
is kept as a small adapter for callers that want to make the streaming choice
explicit and for future selective-ggml-slot work.
"""

from __future__ import annotations

from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend


class RWKVCppStreamingBackend(RWKVCppBackend):
    """Explicit streaming spelling of the provider-backed rwkv.cpp backend."""

    pass


__all__ = ["RWKVCppStreamingBackend"]

