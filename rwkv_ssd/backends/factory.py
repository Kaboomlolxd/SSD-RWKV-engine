"""Backend selection for the inference engine."""

from __future__ import annotations

from rwkv_ssd.backends.albatross import AlbatrossBackend
from rwkv_ssd.backends.base import RecurrentBackend
from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
from rwkv_ssd.backends.pack_backend import PackBackend
from rwkv_ssd.backends.rwkvcpp import RWKVCppBackend, is_rwkvcpp_available
from rwkv_ssd.backends.synthetic import SyntheticBackend
from rwkv_ssd.runtime.errors import BackendNotAvailableError

# Supported runtime surface. Sequence-model experiments remain available to
# their research fixtures, but are not production engine backends.
SUPPORTED_BACKENDS = frozenset(
    {
        "synthetic",
        "reference",
        "mock",
        "chatrwkv",
        "chat",
        "pytorch",
    }
)

# External CUDA backend. Selection remains availability-gated because the
# repository does not vendor CUDA kernels or the Albatross checkout.
ALBATROSS_BACKENDS = frozenset({"albatross", "alb"})
OPTIONAL_BACKENDS = frozenset({"rwkvcpp", "cpp"})


def create_backend(name: str) -> RecurrentBackend:
    key = name.lower().strip()
    if key in ("synthetic", "reference", "mock"):
        return SyntheticBackend()
    if key in ("chatrwkv", "chat", "pytorch"):
        return ChatRWKVBackend()
    if key in ("rwkvcpp", "cpp"):
        return RWKVCppBackend()
    if key in ("albatross", "alb"):
        return AlbatrossBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Choose: rwkvcpp (CPU), chatrwkv "
        "(RWKV reference), synthetic (tests), or albatross "
        "(external CUDA + layer-wise adapter)."
    )


def is_pack_backend(backend: RecurrentBackend) -> bool:
    return isinstance(backend, PackBackend)


def supports_streaming_mode(backend_name: str) -> bool:
    key = backend_name.lower().strip()
    if key in OPTIONAL_BACKENDS:
        return key in ("rwkvcpp", "cpp")
    return key in (*SUPPORTED_BACKENDS, *ALBATROSS_BACKENDS)


def ensure_v0_backend(name: str) -> None:
    key = name.lower().strip()
    if key in SUPPORTED_BACKENDS:
        return
    if key in OPTIONAL_BACKENDS:
        if is_rwkvcpp_available():
            return
        raise BackendNotAvailableError(
            f"Backend {name!r} requires a built rwkv.cpp shared library. "
            "Build backends/rwkvcpp_ref or set RWKVCPP_DLL."
        )
    if key in ALBATROSS_BACKENDS:
        reason = AlbatrossBackend.availability_error()
        if reason is not None:
            raise BackendNotAvailableError(
                f"Backend {name!r} is unavailable: {reason}."
            )
        return
    raise ValueError(f"Unknown backend {name!r}")
