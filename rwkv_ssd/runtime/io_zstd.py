"""CPU store for optional whole-pack zstandard compression.

The compressed pack keeps the original logical tensor offsets in its manifest.
This store decompresses the frame once at open time and then serves the same
range/span API as the mmap and pread stores.  It is intentionally a cold-pack
feature: the uncompressed logical pack must fit in host RAM, and hot workloads
should continue to use the normal mmap pack.
"""

from __future__ import annotations

from pathlib import Path

from rwkv_ssd.runtime.manifest import TensorEntry
from rwkv_ssd.runtime.weight_store_base import WeightStore


class ZstdWeightStore(WeightStore):
    """Decompress a ``weights.bin.zst`` frame into a host-side byte buffer."""

    # The decompressed buffer is owned by this store for its entire lifetime,
    # so native consumers may borrow memoryviews just as they do from mmap.
    stable_memoryviews = True

    def __init__(self, weights_path: Path) -> None:
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise ImportError(
                "zstandard is required to open compressed packs; "
                "install with `pip install 'rwkv-ssd[zstd]'`"
            ) from exc

        self._path = Path(weights_path)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        self._data = self._decompress(self._path, zstd)
        if not self._data:
            raise OSError(f"compressed weights file is empty: {self._path}")

    @staticmethod
    def _decompress(path: Path, zstd: object) -> bytes | bytearray:
        # Use stream_reader rather than read_bytes()+decompress(). Large packs
        # should not briefly retain both the compressed and uncompressed image.
        dctx = zstd.ZstdDecompressor()  # type: ignore[attr-defined]
        expected = -1
        try:
            with path.open("rb") as header_file:
                expected = int(zstd.frame_content_size(header_file.read(64)))  # type: ignore[attr-defined]
            # zstandard reports these sentinel values as very large unsigned
            # integers.  Treating CONTENTSIZE_UNKNOWN as a real size would
            # attempt an effectively unbounded allocation for a valid
            # streaming frame (for example one produced by ``compressobj``).
            unknown = int(getattr(zstd, "CONTENTSIZE_UNKNOWN", 2**64 - 1))
            error = int(getattr(zstd, "CONTENTSIZE_ERROR", 2**64 - 2))
            if expected in (unknown, error):
                expected = -1
        except (OSError, ValueError):
            expected = -1

        if expected >= 0:
            data = bytearray(expected)
            position = 0
            with path.open("rb") as source, dctx.stream_reader(source) as reader:
                view = memoryview(data)
                while position < expected:
                    count = reader.readinto(view[position:])
                    if not count:
                        raise OSError(
                            f"short zstd frame: got {position} bytes, expected {expected}"
                        )
                    position += count
            return data

        # Older/external zstd frames may omit the content size. Keep a safe
        # streaming fallback for those files rather than requiring a repack.
        data = bytearray()
        with path.open("rb") as source, dctx.stream_reader(source) as reader:
            while True:
                chunk = reader.read(8 * 1024 * 1024)
                if not chunk:
                    break
                # Keep one mutable backing allocation. ``b"".join(chunks)``
                # would briefly duplicate a multi-GB logical pack at return.
                data.extend(chunk)
        return data

    @property
    def logical_size(self) -> int:
        return len(self._data)

    def _validate_span(self, offset: int, length: int) -> None:
        if offset < 0 or length < 0 or offset + length > len(self._data):
            raise OSError(
                f"read past decompressed weights: [{offset}, {offset + length})"
            )

    def read_bytes(self, entry: TensorEntry) -> bytes:
        self._validate_span(entry.offset, entry.length)
        return bytes(self._data[entry.offset : entry.offset + entry.length])

    def read_range(
        self, entry: TensorEntry, byte_offset: int, length: int, dest: memoryview
    ) -> None:
        if byte_offset < 0 or length < 0 or byte_offset + length > entry.length:
            raise ValueError(f"read past end of tensor {entry.name}")
        if len(dest) < length:
            raise ValueError(
                f"dest too small for {entry.name}: need {length}, have {len(dest)}"
            )
        self._validate_span(entry.offset + byte_offset, length)
        dest[:length] = self._data[entry.offset + byte_offset : entry.offset + byte_offset + length]

    def read_bytes_span(self, offset: int, length: int) -> bytes:
        self._validate_span(offset, length)
        return bytes(self._data[offset : offset + length])

    def read_bytearray_span(self, offset: int, length: int) -> bytearray:
        self._validate_span(offset, length)
        return bytearray(self._data[offset : offset + length])

    def read_memoryview_span(self, offset: int, length: int) -> memoryview:
        self._validate_span(offset, length)
        return memoryview(self._data)[offset : offset + length]

    def advise_prefetch(self, entries: list[TensorEntry]) -> bool:
        # The complete logical image is already resident after construction.
        return False

    def advise_release(self, entries: list[TensorEntry]) -> bool:
        return False

    def close(self) -> None:
        self._data = bytearray()


__all__ = ["ZstdWeightStore"]
