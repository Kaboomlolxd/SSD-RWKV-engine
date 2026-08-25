"""Device selection with CPU fallback when an accelerator is unavailable."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

SUPPORTED_MANIFEST_VERSION = 1


def xpu_runtime_available() -> bool:
    """Return whether Torch can currently initialize an Intel XPU device.

    Some Torch builds expose ``torch.xpu`` even when the runtime/driver is
    absent. The availability call can also raise while the runtime is being
    initialized, so device selection treats those cases as unavailable.
    """
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except Exception as exc:
        logger.debug("Torch XPU probe failed: %s", exc)
        return False


def xpu_compute_available(dtype: torch.dtype = torch.bfloat16) -> bool:
    """Return whether the XPU can execute a real matrix operation.

    ``torch.xpu.is_available()`` only proves that the runtime can enumerate a
    device. Intel's gather/indexing path can also work when the matrix backend
    cannot create an engine, so enumeration alone is not sufficient for RWKV
    computation. Keep this probe tiny and synchronize it so asynchronous
    failures are reported before the model is loaded.
    """
    if not xpu_runtime_available():
        return False
    xpu = getattr(torch, "xpu", None)
    try:
        device = torch.device("xpu")
        lhs = torch.ones((2, 2), dtype=dtype, device=device)
        rhs = torch.ones((2, 2), dtype=dtype, device=device)
        torch.mm(lhs, rhs)
        if xpu is not None and hasattr(xpu, "synchronize"):
            xpu.synchronize()
        return True
    except Exception as exc:
        logger.warning(
            "Intel XPU is visible but matrix computation is unavailable for %s: %s",
            dtype,
            exc,
        )
        return False


def _strategy_dtype(strategy: str) -> torch.dtype:
    parts = strategy.strip().lower().split()
    if len(parts) > 1 and parts[1] == "fp16":
        return torch.float16
    if len(parts) > 1 and parts[1] == "fp32":
        return torch.float32
    return torch.bfloat16


def resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req if ":" in req else "cuda")
        logger.warning(
            "CUDA requested but not available — falling back to CPU. "
            "ChatRWKV does not use Vulkan; for GPU use cuda fp16 with an NVIDIA GPU."
        )
        return torch.device("cpu")
    if req.startswith("xpu"):
        if xpu_runtime_available():
            if xpu_compute_available():
                return torch.device(req if ":" in req else "xpu")
            logger.warning(
                "Intel XPU can allocate and run LUT decode, but its matrix "
                "compute engine is unavailable; using CPU for model computation. "
                "XPU LUT decode remains available with "
                "--trinity-decode-device xpu."
            )
            return torch.device("cpu")
        logger.warning(
            "Intel XPU requested but not available — install intel-extension-for-pytorch "
            "and GPU drivers; falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(req)


def is_accelerator_device(device: torch.device) -> bool:
    """True for CUDA / Intel XPU compute devices."""
    return device.type in ("cuda", "xpu")


def resolve_strategy(requested: str, device: torch.device, *, rwkv7: bool = False) -> str:
    """Map ChatRWKV STRATEGY to something valid on the active device."""
    s = requested.strip()
    s_lower = s.lower()
    if device.type == "cpu":
        if "cuda" in s_lower or "gpu" in s_lower or "xpu" in s_lower:
            return "cpu fp32"
        if "fp16" in s_lower or "bf16" in s_lower:
            return s if rwkv7 else "cpu fp32"
    if device.type == "xpu" and not xpu_compute_available(_strategy_dtype(s)):
        logger.warning(
            "XPU strategy %r cannot execute matrix computation; using CPU fp32.",
            s,
        )
        return "cpu fp32"
    return s
