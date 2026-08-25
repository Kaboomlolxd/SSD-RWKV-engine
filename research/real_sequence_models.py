"""Dependency-light CPU references for the bundled real HF checkpoints.

This module deliberately implements only the inference contracts needed for
the small local Mamba-2 and Llama checkpoints.  It is a correctness harness,
not a replacement for the optimized ``transformers``/``mamba_ssm`` kernels:
all recurrent state transitions are explicit so full-sequence and cached
token-by-token execution can be compared on a CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from rwkv_ssd.runtime.safetensors_loader import load_safetensors_dir
from rwkv_ssd.runtime.dspark import DSparkDrafter, DSparkProposal


def load_local_hf_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load ``config.json`` and a local HF safetensors state dict."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = load_safetensors_dir(root)
    return config, state


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x @ weight.to(dtype=x.dtype).transpose(-1, -2)


@dataclass
class Mamba2State:
    conv: torch.Tensor
    ssm: torch.Tensor


class ReferenceMamba2:
    """CPU reference for the HF ``Mamba2ForCausalLM`` tensor layout."""

    architecture = "mamba2"

    def __init__(self, config: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> None:
        self.config = dict(config)
        self.state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        self.n_layer = int(config["num_hidden_layers"])
        self.hidden_size = int(config["hidden_size"])
        self.d_inner = int(config.get("intermediate_size", self.hidden_size * int(config.get("expand", 2))))
        self.state_size = int(config["state_size"])
        self.n_heads = int(config["num_heads"])
        self.head_dim = int(config["head_dim"])
        self.n_groups = int(config["n_groups"])
        self.conv_kernel = int(config["conv_kernel"])
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.state_size
        self.eps = float(config.get("layer_norm_epsilon", 1e-5))
        self.residual_in_fp32 = bool(config.get("residual_in_fp32", False))
        self.time_step_floor = float(
            config.get(
                "time_step_floor",
                config.get("time_step_limit", [0.0, float("inf")])[0]
                if isinstance(config.get("time_step_limit"), (list, tuple))
                else 0.0,
            )
        )
        configured_time_max = config.get("time_step_max")
        if configured_time_max is None:
            limit = config.get("time_step_limit")
            configured_time_max = (
                limit[1] if isinstance(limit, (list, tuple)) and len(limit) > 1 else float("inf")
            )
        self.time_step_max = float(configured_time_max)
        if self.n_heads * self.head_dim != self.d_inner:
            raise ValueError("Mamba-2 checkpoint has inconsistent num_heads * head_dim")
        if self.conv_dim + self.d_inner + self.n_heads != int(
            self.state[f"backbone.layers.0.mixer.in_proj.weight"].shape[0]
        ):
            raise ValueError("unsupported Mamba-2 in_proj layout")

    @classmethod
    def from_pretrained_local(cls, path: str | Path) -> "ReferenceMamba2":
        config, state = load_local_hf_checkpoint(path)
        if str(config.get("model_type", "")).lower() != "mamba2":
            raise ValueError("checkpoint is not model_type=mamba2")
        return cls(config, state)

    @property
    def vocab_size(self) -> int:
        return int(self.state["backbone.embeddings.weight"].shape[0])

    @property
    def parameter_count(self) -> int:
        return sum(int(t.numel()) for t in self.state.values())

    def _initial_state(self, batch: int) -> list[Mamba2State]:
        dtype = self.state["backbone.embeddings.weight"].dtype
        return [
            Mamba2State(
                torch.zeros(batch, self.conv_dim, self.conv_kernel - 1, dtype=dtype),
                torch.zeros(batch, self.n_heads, self.state_size, self.head_dim, dtype=torch.float32),
            )
            for _ in range(self.n_layer)
        ]

    def _layer_step(
        self,
        layer_id: int,
        x: torch.Tensor,
        previous: Mamba2State,
    ) -> tuple[torch.Tensor, Mamba2State]:
        p = f"backbone.layers.{layer_id}.mixer."
        hidden = _linear(x, self.state[p + "in_proj.weight"])
        conv_input, gate, dt_input = torch.split(
            hidden, [self.conv_dim, self.d_inner, self.n_heads], dim=-1
        )
        conv_input = conv_input.to(dtype=self.state[p + "conv1d.weight"].dtype)
        window = torch.cat((previous.conv, conv_input.unsqueeze(-1)), dim=-1)
        conv = F.conv1d(
            window,
            self.state[p + "conv1d.weight"],
            self.state[p + "conv1d.bias"],
            groups=self.conv_dim,
        )[:, :, -1]
        conv = F.silu(conv)
        u = conv[:, : self.d_inner]
        b = conv[:, self.d_inner : self.d_inner + self.n_groups * self.state_size]
        c = conv[:, self.d_inner + self.n_groups * self.state_size :]
        b = b.view(x.shape[0], self.n_groups, self.state_size)
        c = c.view(x.shape[0], self.n_groups, self.state_size)

        dt = F.softplus(dt_input + self.state[p + "dt_bias"].to(dtype=dt_input.dtype))
        dt = dt.clamp_min(self.time_step_floor)
        if self.time_step_max > 0:
            dt = dt.clamp_max(self.time_step_max)
        a = -torch.exp(self.state[p + "A_log"].float()).view(1, self.n_heads)
        discrete_a = torch.exp(dt.float() * a)

        # Each head consumes one group of B/C rows.  The checkpoint uses
        # n_heads divisible by n_groups, as required by the HF implementation.
        heads_per_group = self.n_heads // self.n_groups
        b_head = b.repeat_interleave(heads_per_group, dim=1)
        c_head = c.repeat_interleave(heads_per_group, dim=1)
        u_head = u.view(x.shape[0], self.n_heads, self.head_dim)
        ssm = discrete_a[:, :, None, None] * previous.ssm.float()
        ssm = ssm + (dt.float()[:, :, None, None] * b_head[:, :, :, None] * u_head[:, :, None, :])
        y = (ssm * c_head[:, :, :, None]).sum(dim=2)
        y = y + self.state[p + "D"].float().view(1, self.n_heads, 1) * u_head.float()
        y = y.reshape(x.shape[0], self.d_inner).to(dtype=x.dtype)
        y = _rms_norm(y, self.state[p + "norm.weight"].to(dtype=x.dtype), self.eps)
        if bool(self.config.get("norm_before_gate", True)):
            y = y * F.silu(gate.to(dtype=y.dtype))
        else:
            y = _rms_norm(y * F.silu(gate.to(dtype=y.dtype)), self.state[p + "norm.weight"], self.eps)
        y = _linear(y, self.state[p + "out_proj.weight"])
        next_conv = window[:, :, 1:].detach()
        return y, Mamba2State(next_conv, ssm.detach())

    def _block_step(
        self, layer_id: int, x: torch.Tensor, state: Mamba2State
    ) -> tuple[torch.Tensor, Mamba2State]:
        p = f"backbone.layers.{layer_id}."
        norm_weight = self.state[p + "norm.weight"]
        residual = x.float() if self.residual_in_fp32 else x
        normed = _rms_norm(x.to(dtype=norm_weight.dtype), norm_weight, self.eps)
        y, next_state = self._layer_step(layer_id, normed, state)
        return residual + y, next_state

    @torch.no_grad()
    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        states: list[Mamba2State] | None = None,
    ) -> tuple[torch.Tensor, list[Mamba2State]]:
        """Return final-layer hidden states plus the updated recurrent state."""
        ids = token_ids.to(dtype=torch.long, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        batch, steps = ids.shape
        current = states or self._initial_state(batch)
        if len(current) != self.n_layer:
            raise ValueError("wrong number of Mamba-2 layer states")
        outputs: list[torch.Tensor] = []
        emb = self.state["backbone.embeddings.weight"]
        for t in range(steps):
            x = emb.index_select(0, ids[:, t])
            next_states: list[Mamba2State] = []
            for layer_id, layer_state in enumerate(current):
                x, layer_state = self._block_step(layer_id, x, layer_state)
                next_states.append(layer_state)
            current = next_states
            outputs.append(x)
        return torch.stack(outputs, dim=1), current

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        final_norm = self.state["backbone.norm_f.weight"]
        normalized = _rms_norm(hidden.to(dtype=final_norm.dtype), final_norm, self.eps)
        return _linear(normalized, self.state["lm_head.weight"])

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        states: list[Mamba2State] | None = None,
    ) -> tuple[torch.Tensor, list[Mamba2State]]:
        hidden, current = self.forward_hidden(token_ids, states)
        return self.logits_from_hidden(hidden), current


@dataclass
class AttentionState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


class ReferenceLlama:
    """CPU reference for the small standard Llama/HF Transformer checkpoint."""

    architecture = "llama"

    def __init__(self, config: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> None:
        self.config = dict(config)
        self.state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        self.n_layer = int(config["num_hidden_layers"])
        self.hidden_size = int(config["hidden_size"])
        self.n_head = int(config["num_attention_heads"])
        self.n_kv_head = int(config.get("num_key_value_heads", self.n_head))
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.n_head))
        self.intermediate_size = int(config["intermediate_size"])
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        rope = config.get("rope_parameters", {})
        self.rope_theta = float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))
        rope_type = str(rope.get("rope_type", "default")).lower()
        if rope_type in {"default", "none"}:
            self.rope_scale = 1.0
        elif rope_type == "linear":
            self.rope_scale = float(rope.get("factor", 1.0))
        else:
            raise ValueError(f"unsupported RoPE scaling type {rope_type!r}")
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head:
            raise ValueError("ReferenceLlama requires num_attention_heads divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("ReferenceLlama RoPE requires an even head_dim")

    @classmethod
    def from_pretrained_local(cls, path: str | Path) -> "ReferenceLlama":
        config, state = load_local_hf_checkpoint(path)
        if str(config.get("model_type", "")).lower() not in {"llama", "transformer"}:
            raise ValueError("checkpoint is not a Llama-style Transformer")
        return cls(config, state)

    @property
    def vocab_size(self) -> int:
        return int(self.state["model.embed_tokens.weight"].shape[0])

    @property
    def parameter_count(self) -> int:
        return sum(int(t.numel()) for t in self.state.values())

    def _initial_state(self) -> AttentionState:
        return AttentionState([], [])

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, time, head_dim]
        half = self.head_dim // 2
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        angles = (positions.float() / self.rope_scale)[:, None] * inv[None, :]
        cos, sin = angles.cos().to(dtype=x.dtype), angles.sin().to(dtype=x.dtype)
        first, second = x[..., :half], x[..., half:]
        rotated = torch.cat((-second, first), dim=-1)
        return x * torch.cat((cos, cos), dim=-1)[None, None, :, :] + rotated * torch.cat((sin, sin), dim=-1)[None, None, :, :]

    def _layer(
        self,
        layer_id: int,
        x: torch.Tensor,
        state: AttentionState,
        position: int,
    ) -> tuple[torch.Tensor, AttentionState]:
        p = f"model.layers.{layer_id}."
        residual = x
        normed = _rms_norm(x, self.state[p + "input_layernorm.weight"].to(dtype=x.dtype), self.eps)
        q = _linear(normed, self.state[p + "self_attn.q_proj.weight"])
        k = _linear(normed, self.state[p + "self_attn.k_proj.weight"])
        v = _linear(normed, self.state[p + "self_attn.v_proj.weight"])
        batch, steps, _ = q.shape
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)
        q_norm = self.state.get(p + "self_attn.q_norm.weight")
        k_norm = self.state.get(p + "self_attn.k_norm.weight")
        if q_norm is not None:
            q = _rms_norm(q, q_norm.to(dtype=q.dtype), self.eps)
        if k_norm is not None:
            k = _rms_norm(k, k_norm.to(dtype=k.dtype), self.eps)
        positions = torch.arange(position, position + steps, dtype=torch.long)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        previous_k = state.keys[layer_id] if len(state.keys) > layer_id else None
        previous_v = state.values[layer_id] if len(state.values) > layer_id else None
        full_k = torch.cat((previous_k, k), dim=2) if previous_k is not None else k
        full_v = torch.cat((previous_v, v), dim=2) if previous_v is not None else v
        if self.n_kv_head == self.n_head:
            scores = (q.float() @ full_k.float().transpose(-1, -2)) / (self.head_dim ** 0.5)
        else:
            heads_per_group = self.n_head // self.n_kv_head
            grouped_q = q.view(batch, self.n_kv_head, heads_per_group, steps, self.head_dim)
            scores = (
                grouped_q.float()
                @ full_k.float().unsqueeze(2).transpose(-1, -2)
            ) / (self.head_dim ** 0.5)
        total = full_k.shape[2]
        allowed = torch.arange(total)[None, :] <= (position + torch.arange(steps))[:, None]
        if self.n_kv_head == self.n_head:
            scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1).to(dtype=x.dtype) @ full_v.to(dtype=x.dtype)
            attn = attn.transpose(1, 2).reshape(batch, steps, self.hidden_size)
        else:
            scores = scores.masked_fill(~allowed[None, None, None, :, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1).to(dtype=x.dtype) @ full_v.to(dtype=x.dtype).unsqueeze(2)
            attn = attn.reshape(batch, self.n_head, steps, self.head_dim).transpose(1, 2).reshape(
                batch, steps, self.hidden_size
            )
        x = residual + _linear(attn, self.state[p + "self_attn.o_proj.weight"])
        residual = x
        normed = _rms_norm(x, self.state[p + "post_attention_layernorm.weight"].to(dtype=x.dtype), self.eps)
        gate = _linear(normed, self.state[p + "mlp.gate_proj.weight"])
        up = _linear(normed, self.state[p + "mlp.up_proj.weight"])
        mlp = _linear(F.silu(gate) * up, self.state[p + "mlp.down_proj.weight"])
        keys = list(state.keys)
        values = list(state.values)
        while len(keys) <= layer_id:
            keys.append(full_k.detach())
            values.append(full_v.detach())
        keys[layer_id] = full_k.detach()
        values[layer_id] = full_v.detach()
        return residual + mlp, AttentionState(keys, values)

    @torch.no_grad()
    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        state: AttentionState | None = None,
    ) -> tuple[torch.Tensor, AttentionState]:
        """Return final-layer hidden states plus the updated KV cache."""
        ids = token_ids.to(dtype=torch.long, device="cpu")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        x = self.state["model.embed_tokens.weight"].index_select(0, ids.reshape(-1)).view(*ids.shape, self.hidden_size)
        current = state or self._initial_state()
        position = current.keys[0].shape[2] if current.keys else 0
        for layer_id in range(self.n_layer):
            x, current = self._layer(layer_id, x, current, position)
        return x, current

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = _rms_norm(
            hidden,
            self.state["model.norm.weight"].to(dtype=hidden.dtype),
            self.eps,
        )
        return _linear(normalized, self.state["lm_head.weight"])

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        state: AttentionState | None = None,
    ) -> tuple[torch.Tensor, AttentionState]:
        hidden, current = self.forward_hidden(token_ids, state)
        return self.logits_from_hidden(hidden), current


@torch.no_grad()
def dspark_propose_reference(
    model: ReferenceMamba2 | ReferenceLlama,
    anchor_tokens: torch.Tensor,
    *,
    proposal_length: int = 4,
    mask_token_id: int = 0,
    rank: int = 32,
    head: str = "markov",
    greedy: bool = True,
    temperature: float = 1.0,
    drafter: DSparkDrafter | None = None,
) -> DSparkProposal:
    """Run the shared DSpark adapter against either real CPU reference.

    The block is intentionally generated from a clean reference state. A
    streaming caller must keep its original target state and discard the
    masked proposal state's KV/SSM updates until verification accepts tokens.
    This makes the helper safe for both attention caches and recurrent Mamba
    state without pretending that a masked recurrent rollout is the target
    state.
    """

    if anchor_tokens.ndim != 1:
        raise ValueError("anchor_tokens must have shape [batch]")
    if proposal_length <= 0:
        raise ValueError("proposal_length must be positive")
    block_inputs = torch.full(
        (anchor_tokens.shape[0], proposal_length),
        int(mask_token_id),
        dtype=torch.long,
        device="cpu",
    )
    block_inputs[:, 0] = anchor_tokens.to(device="cpu", dtype=torch.long)
    hidden, _ = model.forward_hidden(block_inputs)
    base_logits = model.logits_from_hidden(hidden)
    if drafter is None:
        drafter = DSparkDrafter(
            vocab_size=model.vocab_size,
            hidden_size=model.hidden_size,
            rank=rank,
            head=head,
            with_confidence=True,
        )
    drafter = drafter.to(device=hidden.device, dtype=hidden.dtype)
    return drafter.propose(
        base_logits,
        hidden,
        anchor_tokens.to(device=hidden.device, dtype=torch.long),
        temperature=temperature,
        greedy=greedy,
    )


def load_real_reference(path: str | Path) -> ReferenceMamba2 | ReferenceLlama:
    """Dispatch a local HF directory to the matching CPU reference."""
    config_path = Path(path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "mamba2":
        return ReferenceMamba2.from_pretrained_local(path)
    if model_type in {"llama", "transformer"}:
        return ReferenceLlama.from_pretrained_local(path)
    raise ValueError(f"unsupported local reference model_type={model_type!r}")


__all__ = [
    "AttentionState",
    "Mamba2State",
    "ReferenceLlama",
    "ReferenceMamba2",
    "dspark_propose_reference",
    "load_local_hf_checkpoint",
    "load_real_reference",
]
