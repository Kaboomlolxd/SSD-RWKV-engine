"""Small CPU reference shims for the Kimi-K3 remote-code dependencies.

Kimi-K3's public checkpoint currently imports the CUDA/Triton implementations
from ``fla-core`` unconditionally.  That is a good production dependency, but
it leaves a perfectly usable CPU reference implementation inaccessible on
machines that only have PyTorch.  This module supplies the same equations and
state layouts for the handful of FLA symbols used by the Kimi text model.

The shim is intentionally narrow.  It is installed only while Hugging Face
loads the remote-code model, and it does not pretend to implement unrelated
FLA operators or a low-RAM weight provider.  The model remains a resident CPU
reference backend until a Kimi-specific streamed execution plan exists.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import sys
import types
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_cache(function):
    """Compatibility decorator for FLA's tensor-aware memoization decorator.

    The decorated Kimi helper computes tiny index tensors.  A no-op decorator
    is preferable to caching arbitrary tensor arguments by object identity,
    which can retain request state longer than the engine's cache policy.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapped


def _activation(value: torch.Tensor, name: str | None) -> torch.Tensor:
    if name in ("silu", "swish"):
        return value * torch.sigmoid(value)
    if name is None:
        return value
    raise ValueError(f"unsupported CPU ShortConvolution activation {name!r}")


class ShortConvolution(nn.Conv1d):
    """Depthwise causal convolution with the FLA cache layout.

    FLA stores ``[batch, channels, kernel]`` containing the most recent raw
    projected inputs.  Keeping the current token in the last position makes
    the prefill and one-token decode paths mathematically identical.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        backend: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        del backend, kwargs
        super().__init__(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=0,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = int(hidden_size)
        self.activation = activation

    def _weight_2d(self) -> torch.Tensor:
        return self.weight[:, 0, :]

    def _decode_step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        *,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The upstream module receives [B, 1, D] for ordinary decode.
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"CPU ShortConvolution decode expects [B,1,D], got {tuple(x.shape)}")
        batch, _, channels = x.shape
        width = int(self.kernel_size[0])
        if channels != self.hidden_size:
            raise ValueError("ShortConvolution channel count changed during decode")
        if cache is None:
            cache = x.new_zeros((batch, channels, width))
        elif tuple(cache.shape) != (batch, channels, width):
            raise ValueError(
                "ShortConvolution cache shape changed: "
                f"expected {(batch, channels, width)}, got {tuple(cache.shape)}"
            )
        # Match causal_conv1d_update: shift left, then append the current raw
        # input.  Mutate a caller-owned cache in place when possible because
        # KimiDynamicCache documents that behavior.
        cache[..., :-1] = cache[..., 1:]
        cache[..., -1] = x[:, 0, :]
        y = (cache * self._weight_2d().to(dtype=cache.dtype)).sum(dim=-1)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        y = _activation(y, self.activation).unsqueeze(1)
        return y, cache if output_final_state else None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del chunk_indices, kwargs
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        batch, length, channels = x.shape
        if channels != self.hidden_size:
            raise ValueError("ShortConvolution channel count changed")

        # The Kimi runtime uses a normal rectangular batch.  Supporting the
        # packed form as well costs little and keeps the shim honest for
        # callers that use the model directly.
        if cu_seqlens is not None:
            if batch != 1:
                raise ValueError("packed CPU ShortConvolution expects batch=1")
            outputs: list[torch.Tensor] = []
            final_caches: list[torch.Tensor] = []
            for index in range(int(cu_seqlens.numel()) - 1):
                start = int(cu_seqlens[index].item())
                stop = int(cu_seqlens[index + 1].item())
                segment = x[:, start:stop, :]
                initial = None
                if cache is not None and index == 0:
                    initial = cache[index : index + 1]
                output, final = self._prefill(segment, initial, output_final_state=output_final_state)
                outputs.append(output)
                if final is not None:
                    final_caches.append(final)
            result = torch.cat(outputs, dim=1) if outputs else x[:, :0, :]
            final_state = torch.cat(final_caches, dim=0) if final_caches else None
        elif length == 1:
            result, final_state = self._decode_step(
                x, cache, output_final_state=output_final_state
            )
        else:
            result, final_state = self._prefill(
                x, cache, output_final_state=output_final_state
            )
        if residual is not None:
            result = result + residual
        return result, final_state

    def _prefill(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        *,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, channels = x.shape
        width = int(self.kernel_size[0])
        if cache is None:
            initial = x.new_zeros((batch, channels, width))
        else:
            if tuple(cache.shape) != (batch, channels, width):
                raise ValueError(
                    "ShortConvolution cache shape changed: "
                    f"expected {(batch, channels, width)}, got {tuple(cache.shape)}"
                )
            initial = cache
        # ``cache`` contains W values, with the last value being the most
        # recent token already consumed.  The next output therefore uses
        # cache[..., 1:] followed by the new input.  With an empty cache the
        # first W-1 entries are simply left padding zeros.
        sequence = torch.cat(
            [initial[..., 1:], x.transpose(1, 2)], dim=-1
        )
        windows = sequence.unfold(-1, width, 1)
        # unfold yields [B,D,T,W]; keep accumulation in the input dtype just
        # like the normal convolution path and let the model's dtype govern
        # the final output.
        result = (windows * self._weight_2d().to(dtype=windows.dtype).unsqueeze(0).unsqueeze(2)).sum(-1)
        if self.bias is not None:
            result = result + self.bias.to(dtype=result.dtype).view(1, -1, 1)
        result = _activation(result, self.activation).transpose(1, 2)
        if not output_final_state:
            return result, None
        final = sequence[..., -width:].contiguous()
        return result, final


class FusedRMSNormGated(nn.Module):
    """CPU equivalent of FLA's RMS norm followed by a gated activation."""

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = "swish",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.elementwise_affine = bool(elementwise_affine)
        self.eps = float(eps)
        self.activation = str(activation)
        if self.activation not in {"swish", "silu", "sigmoid"}:
            raise ValueError(f"unsupported CPU FusedRMSNormGated activation {activation!r}")
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del residual_in_fp32
        if residual is not None:
            x = x + residual
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        normalized = normalized.to(dtype=x.dtype)
        gate = g.to(dtype=normalized.dtype)
        if self.activation in {"swish", "silu"}:
            normalized = normalized * gate * torch.sigmoid(gate)
        else:
            normalized = normalized * torch.sigmoid(gate)
        return (normalized, x) if prenorm else normalized


def _normalize_qk(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1e-6)


def _kda_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    *,
    lower_bound: float | None,
) -> torch.Tensor:
    if A_log is None:
        return raw_gate.float()
    batch, _, heads, key_dim = raw_gate.shape
    gate = raw_gate.float()
    if dt_bias is not None:
        gate = gate + dt_bias.float().reshape(heads, key_dim).view(1, 1, heads, key_dim)
    a = A_log.float().reshape(1, 1, -1, 1).exp()
    if lower_bound is not None:
        return float(lower_bound) * torch.sigmoid(a * gate)
    # This is the non-safe FLA form.  It is kept here for compatibility with
    # other KDA checkpoints even though Kimi-K3 uses lower_bound=-5.
    return -a * F.softplus(gate)


def _recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del kwargs
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("CPU KDA packed execution expects q batch=1")
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for index in range(int(cu_seqlens.numel()) - 1):
            start = int(cu_seqlens[index].item())
            stop = int(cu_seqlens[index + 1].item())
            initial = initial_state[index : index + 1] if initial_state is not None else None
            output, state = _recurrent_kda(
                q[:, start:stop], k[:, start:stop], v[:, start:stop],
                g[:, start:stop], beta[:, start:stop],
                A_log=A_log, dt_bias=dt_bias, initial_state=initial,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                lower_bound=lower_bound,
            )
            outputs.append(output)
            if state is not None:
                states.append(state)
        final = torch.cat(states, dim=0) if states else None
        return torch.cat(outputs, dim=1), final

    dtype = v.dtype
    qf = _normalize_qk(q) if use_qk_l2norm_in_kernel else q.float()
    kf = _normalize_qk(k) if use_qk_l2norm_in_kernel else k.float()
    vf = v.float()
    gf = _kda_gate(g, A_log, dt_bias, lower_bound=lower_bound) if use_gate_in_kernel else g.float()
    betaf = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()
    batch, length, query_heads, key_dim = qf.shape
    value_heads = vf.shape[2]
    if value_heads % query_heads:
        raise ValueError("Kimi KDA value heads must be divisible by query heads")
    repeat = value_heads // query_heads
    qf = qf.repeat_interleave(repeat, dim=2) * (key_dim ** -0.5)
    kf = kf.repeat_interleave(repeat, dim=2)
    if initial_state is None:
        state = vf.new_zeros((batch, value_heads, key_dim, vf.shape[-1]), dtype=torch.float32)
    else:
        state = initial_state.float().clone()
    outputs = torch.empty_like(vf)
    for index in range(length):
        state = state * gf[:, index].exp().unsqueeze(-1)
        residual = vf[:, index] - torch.einsum("bhk,bhkv->bhv", kf[:, index], state)
        state = state + torch.einsum(
            "bhk,bhv->bhkv", betaf[:, index].unsqueeze(-1) * kf[:, index], residual
        )
        outputs[:, index] = torch.einsum("bhk,bhkv->bhv", qf[:, index], state)
    return outputs.to(dtype=dtype), state if output_final_state else None


def chunk_kda(*args: Any, **kwargs: Any):
    return _recurrent_kda(*args, **kwargs)


def fused_recurrent_kda(*args: Any, **kwargs: Any):
    return _recurrent_kda(*args, **kwargs)


def _prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1, dtype=torch.int32)


def _prepare_cu_seqlens_from_mask(
    mask: torch.Tensor, dtype: torch.dtype | None = torch.int32
) -> torch.Tensor:
    lens = _prepare_lens_from_mask(mask)
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


def _module(name: str, **exports: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(exports)
    return module


@contextmanager
def cpu_compat_modules() -> Iterator[None]:
    """Temporarily install the minimal ``fla`` module tree used by Kimi."""

    names = (
        "fla",
        "fla.modules",
        "fla.ops",
        "fla.ops.kda",
        "fla.ops.utils",
        "fla.ops.utils.index",
        "fla.utils",
    )
    previous = {name: sys.modules.get(name) for name in names}
    root = _module("fla")
    modules = _module(
        "fla.modules",
        ShortConvolution=ShortConvolution,
        FusedRMSNormGated=FusedRMSNormGated,
    )
    ops = _module("fla.ops")
    kda = _module(
        "fla.ops.kda",
        chunk_kda=chunk_kda,
        fused_recurrent_kda=fused_recurrent_kda,
    )
    ops_utils = _module("fla.ops.utils")
    index = _module(
        "fla.ops.utils.index",
        prepare_lens_from_mask=_prepare_lens_from_mask,
        prepare_cu_seqlens_from_mask=_prepare_cu_seqlens_from_mask,
    )
    utils = _module("fla.utils", tensor_cache=_tensor_cache)
    root.modules = modules
    root.ops = ops
    root.utils = utils
    ops.kda = kda
    ops.utils = ops_utils
    ops_utils.index = index
    for name, module in (
        ("fla", root),
        ("fla.modules", modules),
        ("fla.ops", ops),
        ("fla.ops.kda", kda),
        ("fla.ops.utils", ops_utils),
        ("fla.ops.utils.index", index),
        ("fla.utils", utils),
    ):
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


__all__ = [
    "FusedRMSNormGated",
    "ShortConvolution",
    "chunk_kda",
    "cpu_compat_modules",
    "fused_recurrent_kda",
]
