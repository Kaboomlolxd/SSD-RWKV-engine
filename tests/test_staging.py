import torch

from rwkv_ssd.runtime.staging import PingPongStaging


def test_cpu_staging_ring_preserves_legacy_api_and_rotates_slots() -> None:
    staging = PingPongStaging(16, torch.device("cpu"), dtype=torch.float32)
    assert staging.slots == 3
    assert staging.active_dev.numel() == 4
    first = staging.active_dev
    source = torch.arange(4, dtype=torch.float32)
    staging.h2d_async(source, first)
    staging.sync_copy_stream()
    assert torch.equal(first, source)
    staging.swap()
    assert staging.active_dev is not first
    staging.close()
