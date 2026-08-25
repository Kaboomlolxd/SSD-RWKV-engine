"""~0.01B Mamba/attention baselines for architecture experiments.

The defaults are intentionally close to ten million parameters rather than a
toy 100k network.  They are useful for measuring state size, decode behavior,
and hybrid placement on a laptop.  They are not pretrained language models;
the companion benchmark trains only when explicitly asked to do so.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from rwkv_ssd.runtime.dspark import DSparkDrafter, DSparkProposal


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layer: int = 8
    d_ff: int = 768
    n_head: int = 8
    d_state: int = 8
    max_seq_len: int = 512
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * self.weight


class TinyMambaBlock(nn.Module):
    """A compact selective diagonal SSM with an explicit recurrent step."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        d = config.d_model
        self.norm = RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.dt_proj = nn.Linear(d, d, bias=True)
        self.b_proj = nn.Linear(d, config.d_state, bias=False)
        self.c_proj = nn.Linear(d, config.d_state, bias=False)
        self.a_log = nn.Parameter(torch.zeros(d, config.d_state))
        self.d_skip = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, config.d_ff * 2),
            nn.SiLU(),
            nn.Linear(config.d_ff * 2, d),
        )

    def forward_recurrent(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, time, d_model]. Keeping the time loop explicit makes the
        # recurrence easy to compare to RWKV and to future SSD chunk kernels.
        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        batch, steps, d = u.shape
        if state is None:
            state = u.new_zeros(batch, d, self.a_log.shape[-1])
        a = -torch.exp(self.a_log).to(dtype=u.dtype, device=u.device)
        outputs: list[torch.Tensor] = []
        for t in range(steps):
            ut = u[:, t]
            dt = F.softplus(self.dt_proj(ut)).unsqueeze(-1)
            b = self.b_proj(ut).unsqueeze(1)
            c = self.c_proj(ut).unsqueeze(1)
            state = torch.exp(dt * a) * state + dt * ut.unsqueeze(-1) * b
            y = (state * c).sum(dim=-1) + self.d_skip.to(dtype=u.dtype) * ut
            outputs.append(y * torch.sigmoid(gate[:, t]))
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        return y + self.ff(self.ff_norm(y)), state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_recurrent(x)
        return y


class TinyAttentionBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        if config.d_model % config.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.norm = RMSNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff * 2),
            nn.SiLU(),
            nn.Linear(config.d_ff * 2, config.d_model),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        q = self.norm(x)
        attn, _ = self.attn(q, q, q, attn_mask=causal_mask, need_weights=False)
        x = x + attn
        return x + self.ff(self.ff_norm(x))


class TinyMambaLM(nn.Module):
    architecture = "mamba"

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(TinyMambaBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            x = x + layer(x)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.forward_hidden(tokens)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def decode_step(
        self, token: torch.Tensor, states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if token.ndim == 1:
            token = token[:, None]
        x = self.embed(token)
        next_states: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            previous = states[index] if states is not None else None
            y, next_state = layer.forward_recurrent(x, previous)
            x = x + y
            next_states.append(next_state)
        return self.lm_head(self.norm(x[:, -1])), next_states


class TinyAttentionLM(nn.Module):
    architecture = "attention"

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(TinyAttentionBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _mask(self, steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(steps, steps, device=device, dtype=torch.bool), diagonal=1)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        mask = self._mask(tokens.shape[1], tokens.device)
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.forward_hidden(tokens)
        return self.lm_head(self.norm(x))


class TinyHybridLM(nn.Module):
    """Alternating recurrent/attention baseline for placement experiments."""

    architecture = "hybrid"

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            TinyMambaBlock(config) if i % 2 == 0 else TinyAttentionBlock(config)
            for i in range(config.n_layer)
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            if isinstance(layer, TinyMambaBlock):
                x = x + layer(x)
            else:
                x = layer(x, mask)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.forward_hidden(tokens)
        return self.lm_head(self.norm(x))


@torch.no_grad()
def dspark_propose_tiny(
    model: TinyMambaLM | TinyAttentionLM | TinyHybridLM,
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
    """Apply a DSpark head to a tiny attention, Mamba, or hybrid backbone.

    Attention receives one anchor plus mask embeddings and produces the base
    block in one causal forward.  Mamba and the hybrid's recurrent layers
    still roll their state through the block; DSpark improves proposal
    coherence there, but this research helper makes no parallel-speed claim
    for a recurrent backbone.
    """

    if anchor_tokens.ndim != 1:
        raise ValueError("anchor_tokens must have shape [batch]")
    if proposal_length <= 0:
        raise ValueError("proposal_length must be positive")
    batch = int(anchor_tokens.shape[0])
    device = anchor_tokens.device
    block_inputs = torch.full(
        (batch, proposal_length),
        int(mask_token_id),
        dtype=torch.long,
        device=device,
    )
    block_inputs[:, 0] = anchor_tokens
    hidden = model.forward_hidden(block_inputs)
    base_logits = model.lm_head(model.norm(hidden))
    if drafter is None:
        drafter = DSparkDrafter(
            vocab_size=int(model.config.vocab_size),
            hidden_size=int(model.config.d_model),
            rank=rank,
            head=head,
            with_confidence=True,
        )
    drafter = drafter.to(device=hidden.device, dtype=hidden.dtype)
    return drafter.propose(
        base_logits,
        hidden,
        anchor_tokens,
        temperature=temperature,
        greedy=greedy,
    )


def build_tiny_model(kind: str, config: TinyConfig | None = None) -> nn.Module:
    cfg = config or TinyConfig()
    key = kind.strip().lower()
    if key == "mamba":
        return TinyMambaLM(cfg)
    if key in {"attention", "attn", "transformer"}:
        return TinyAttentionLM(cfg)
    if key == "hybrid":
        return TinyHybridLM(cfg)
    raise ValueError("kind must be mamba, attention, or hybrid")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synthetic_batch(config: TinyConfig, batch_size: int, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # A deterministic copy/shift pattern is enough to compare optimization
    # and decode mechanics without claiming language-model quality.
    base = torch.arange(seq_len + 1, device=device).unsqueeze(0).repeat(batch_size, 1)
    offsets = torch.arange(batch_size, device=device).unsqueeze(1) * 17
    tokens = (base + offsets) % config.vocab_size
    return tokens[:, :-1], tokens[:, 1:]


def benchmark_model(model: nn.Module, *, batch_size: int = 1, seq_len: int = 64, device: str = "cpu") -> dict[str, object]:
    model.eval().to(device)
    cfg = model.config  # type: ignore[attr-defined]
    dev = torch.device(device)
    tokens, _ = synthetic_batch(cfg, batch_size, seq_len, device=dev)
    with torch.no_grad():
        _ = model(tokens)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        start = time.perf_counter()
        logits = model(tokens)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start
    return {
        "architecture": getattr(model, "architecture", type(model).__name__),
        "config": asdict(cfg),
        "parameters": parameter_count(model),
        "parameter_billions": parameter_count(model) / 1e9,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "forward_ms": elapsed * 1000.0,
        "tokens_per_second": (batch_size * seq_len) / max(elapsed, 1e-9),
        "logits_shape": list(logits.shape),
    }


def train_toy(model: nn.Module, *, steps: int, batch_size: int, seq_len: int, lr: float, device: str) -> list[float]:
    model.train().to(device)
    cfg = model.config  # type: ignore[attr-defined]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(max(0, steps)):
        tokens, targets = synthetic_batch(cfg, batch_size, seq_len, device=torch.device(device))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens).reshape(-1, cfg.vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark research-only ~0.01B sequence baselines")
    parser.add_argument("--kind", choices=["mamba", "attention", "hybrid", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    kinds = ["mamba", "attention", "hybrid"] if args.kind == "all" else [args.kind]
    results: list[dict[str, object]] = []
    for kind in kinds:
        model = build_tiny_model(kind)
        losses = train_toy(model, steps=args.train_steps, batch_size=args.batch_size, seq_len=args.seq_len, lr=3e-4, device=args.device)
        result = benchmark_model(model, batch_size=args.batch_size, seq_len=args.seq_len, device=args.device)
        if losses:
            result["toy_training"] = {"steps": len(losses), "initial_loss": losses[0], "final_loss": losses[-1]}
        results.append(result)
    payload = {"research_only": True, "models": results}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "TinyAttentionLM",
    "TinyConfig",
    "TinyHybridLM",
    "TinyMambaLM",
    "benchmark_model",
    "build_tiny_model",
    "dspark_propose_tiny",
    "parameter_count",
    "train_toy",
]
