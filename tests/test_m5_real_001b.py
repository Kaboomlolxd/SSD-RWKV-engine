"""M5 golden on real 0.01B quantized packs (pack-only; not vs FP16 resident)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from tests.chatrwkv_greedy import greedy_token_ids
from tests.conftest import REPO, require_checkpoint

PACKS = REPO / "test_model" / "packs_0.01b"
CKPT = REPO / "test_model" / "rwkv7-g1d-0.01b-bench.pth"

pytestmark = [
    pytest.mark.chatrwkv,
    pytest.mark.skipif(find_chatrwkv_root() is None, reason="ChatRWKV not available"),
]


@pytest.mark.parametrize("codec_dir", ["scale_u8", "scale_u4"])
def test_real_001b_codec_streaming_matches_warm_z(codec_dir: str) -> None:
    """
  Quantized packs differ from FP16 checkpoint resident decode.
  Compare strict streaming vs warm-z (all block layers from pack in z).
  """
    require_checkpoint()
    pack = PACKS / codec_dir
    if not (pack / "manifest.json").is_file() or not (
        pack / "weights.bin"
    ).is_file():
        pytest.skip(
            f"pack missing: {pack} (run: python -m rwkv_ssd.tools.eval_m5_codec "
            f"--build {CKPT} --output {PACKS})"
        )
    strict = greedy_token_ids(
        pack,
        CKPT,
        mode="streaming",
        max_tokens=8,
        stream_layer_cache=False,
    )
    warm = greedy_token_ids(
        pack,
        CKPT,
        mode="streaming",
        max_tokens=8,
        stream_layer_cache=True,
        warm_z=True,
    )
    assert strict == warm
