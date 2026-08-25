"""Codec policy override routing (RWKV_CODEC_POLICY=accuracy|hybrid)."""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_provider import _codec_policy_entry_overrides


def _entry(name: str, numel: int, dequant: str = "trinity_lut2") -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=0,
        dtype="bfloat16",
        shape=[numel],
        offset=0,
        length=numel * 2,
        alignment=4096,
        residency="streamed",
        dequant=dequant,
    )


def test_accuracy_routes_head_to_shadow() -> None:
    """``accuracy`` policy routes head/lm_head/output to shadow sidecar."""
    head = _entry("head.weight", 65536 * 768)
    out = _entry("output.weight", 65536 * 768)
    assert _codec_policy_entry_overrides(head, "accuracy") == "shadow"
    assert _codec_policy_entry_overrides(out, "accuracy") == "shadow"


def test_accuracy_does_not_route_block_layers() -> None:
    """``accuracy`` only affects sensitive layers (head / output)."""
    att = _entry("blocks.0.att.receptance.weight", 768 * 768)
    assert _codec_policy_entry_overrides(att, "accuracy") is None


def test_hybrid_routes_large_to_shadow() -> None:
    """``hybrid`` keeps small tensors on LUT, large on shadow."""
    small = _entry("blocks.0.att.receptance.weight", 100_000)
    large = _entry("blocks.0.att.key.weight", 1_000_000)
    assert _codec_policy_entry_overrides(small, "hybrid") is None
    assert _codec_policy_entry_overrides(large, "hybrid") == "shadow"


def test_auto_and_strict_no_override() -> None:
    """``auto`` and ``strict`` never override."""
    head = _entry("head.weight", 65536 * 768)
    assert _codec_policy_entry_overrides(head, "auto") is None
    assert _codec_policy_entry_overrides(head, "strict") is None
    assert _codec_policy_entry_overrides(head, "") is None


def test_unknown_policy_no_override() -> None:
    """Unknown policy values fall back to no-override (graceful)."""
    head = _entry("head.weight", 65536 * 768)
    assert _codec_policy_entry_overrides(head, "garbage") is None
