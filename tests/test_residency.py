"""Residency policy tests (M2 partial)."""

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.residency import apply_residency_policy


def _entry(name: str, layer_id: int) -> TensorEntry:
    return TensorEntry(
        name=name,
        layer_id=layer_id,
        dtype="float32",
        shape=[4, 4],
        offset=0,
        length=64,
        alignment=4096,
        residency="streamed",
    )


def test_resident_mode_all_resident() -> None:
    entries = [_entry("blocks.1.weight", 1), _entry("blocks.2.weight", 2)]
    out = apply_residency_policy(entries, "resident", num_layers=4)
    assert all(e.residency == "resident" for e in out)


def test_partial_middle_streamed() -> None:
    entries = [
        _entry("embed.weight", 0),
        _entry("blocks.1.weight", 1),
        _entry("blocks.2.weight", 2),
        _entry("head.weight", 3),
    ]
    out = apply_residency_policy(entries, "partial", num_layers=4)
    by_name = {e.name: e.residency for e in out}
    assert by_name["embed.weight"] == "resident"
    assert by_name["blocks.1.weight"] == "streamed"
    assert by_name["blocks.2.weight"] == "streamed"


def test_streaming_does_not_resident_block_outputs() -> None:
    entries = [
        _entry("emb.weight", 0),
        _entry("blocks.3.att.output.weight", 3),
        _entry("blocks.3.ffn.value.weight", 3),
        _entry("head.weight", 9999),
        _entry("ln_out.weight", 9999),
    ]
    out = apply_residency_policy(entries, "streaming", num_layers=12)
    by_name = {e.name: e.residency for e in out}
    assert by_name["emb.weight"] == "resident"
    assert by_name["head.weight"] == "resident"
    assert by_name["ln_out.weight"] == "resident"
    assert by_name["blocks.3.att.output.weight"] == "streamed"
    assert by_name["blocks.3.ffn.value.weight"] == "streamed"
