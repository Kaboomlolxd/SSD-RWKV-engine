"""Capture ChatRWKV linear-input RMS statistics from representative prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rwkv_ssd.backends.chatrwkv import ChatRWKVBackend
from rwkv_ssd.runtime.activation_calibration import ActivationStatsCollector
from rwkv_ssd.runtime.config import EngineConfig
from rwkv_ssd.runtime.engine import InferenceEngine


class _LinearCaptureMode(TorchDispatchMode):
    """Observe direct ``@`` calls whose RHS aliases a named model weight."""

    def __init__(
        self,
        by_pointer: dict[int, str],
        collector: ActivationStatsCollector,
    ) -> None:
        super().__init__()
        self.by_pointer = by_pointer
        self.collector = collector

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        mm = torch.ops.aten.mm.default
        mv = torch.ops.aten.mv.default
        addmm = torch.ops.aten.addmm.default
        if func in (mm, mv) and len(args) >= 2:
            activation, weight = args[0], args[1]
        elif func == addmm and len(args) >= 3:
            activation, weight = args[1], args[2]
        else:
            activation = weight = None
        if isinstance(weight, torch.Tensor) and isinstance(activation, torch.Tensor):
            name = self.by_pointer.get(int(weight.data_ptr()))
            if name is not None:
                self.collector.observe(name, activation)
        return func(*args, **kwargs)


def collect_activation_stats(
    checkpoint: Path,
    pack_dir: Path,
    prompts: list[str],
    output: Path,
    *,
    max_tokens: int = 1,
) -> dict[str, object]:
    collector = ActivationStatsCollector()
    config = EngineConfig(
        pack_dir=pack_dir,
        checkpoint_path=str(checkpoint),
        backend="chatrwkv",
        mode="resident",
        device="cpu",
        strategy="cpu bf16",
        max_tokens=max_tokens,
        cache_format="dense",
    )
    with InferenceEngine(config) as engine:
        if not isinstance(engine.backend, ChatRWKVBackend):
            raise TypeError("calibration requires ChatRWKVBackend")
        model = engine.backend._model
        if model is None or not hasattr(model, "z"):
            raise RuntimeError("ChatRWKV model weights are unavailable")
        import rwkv.model as rwkv_model  # type: ignore[import-untyped]

        by_object = {
            id(value): name
            for name, value in model.z.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2
        }
        by_pointer = {
            int(value.data_ptr()): name
            for name, value in model.z.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2
        }
        original = rwkv_model.matmul

        def calibrated_matmul(a, b, *args, **kwargs):
            name = by_object.get(id(b))
            if name is not None and isinstance(a, torch.Tensor):
                collector.observe(name, a)
            return original(a, b, *args, **kwargs)

        rwkv_model.matmul = calibrated_matmul
        try:
            with _LinearCaptureMode(by_pointer, collector):
                for prompt in prompts:
                    engine.generate(prompt)
        finally:
            rwkv_model.matmul = original
    collector.save(output)
    artifact = collector.to_artifact()
    return {
        "output": str(output),
        "tensors": len(artifact["rms"]),
        "samples": artifact["samples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        default="Hello,The future of storage is,Once upon a time,Explain recurrent neural networks",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()
    report = collect_activation_stats(
        args.checkpoint,
        args.pack,
        [item for item in args.prompts.split(",") if item],
        args.output,
        max_tokens=args.max_tokens,
    )
    print(report)


if __name__ == "__main__":
    main()
