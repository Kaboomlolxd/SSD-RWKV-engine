"""Inspect a local CPU inference setup and recommend a usable backend."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Sequence

from rwkv_ssd.backends.chatrwkv import find_chatrwkv_root
from rwkv_ssd.backends.rwkvcpp import (
    find_rwkvcpp_dll,
    find_rwkvcpp_root,
)
from rwkv_ssd.tools.preflight import run_preflight


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return None
    return None


def _size_text(byte_count: int | None) -> str:
    if byte_count is None:
        return "unknown"
    return f"{byte_count / (1024 ** 3):.1f} GiB"


def _find_matching_ggml(checkpoint: Path) -> Path:
    """Find an existing GGML file without triggering optional auto-conversion."""
    explicit = os.environ.get("RWKVCPP_GGML_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"RWKVCPP_GGML_PATH not found: {explicit}")
    if checkpoint.suffix.lower() == ".bin" and checkpoint.is_file():
        return checkpoint
    if checkpoint.suffix.lower() == ".pth":
        stem = checkpoint.with_suffix("")
        for candidate in (
            stem.with_name(stem.name + "-FP16.bin"),
            stem.with_suffix(".bin"),
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No matching GGML file beside {checkpoint}; convert the checkpoint or set RWKVCPP_GGML_PATH"
        )
    if checkpoint.is_file():
        return checkpoint
    raise FileNotFoundError(f"Model path not found: {checkpoint}")


def _source_checkpoint_from_pack(pack: Path | None) -> Path | None:
    if pack is None:
        return None
    for metadata_path in (pack / "meta.json", pack / "manifest.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = metadata.get("source_checkpoint")
        if source is None and isinstance(metadata.get("meta"), dict):
            source = metadata["meta"].get("source_checkpoint")
        if not source:
            continue
        candidate = Path(str(source))
        if not candidate.is_absolute():
            beside_pack = pack / candidate
            if beside_pack.exists():
                candidate = beside_pack
        if candidate.exists():
            return candidate
    return None


def _backend_rows(
    pack: Path | None, checkpoint: Path | None
) -> dict[str, dict[str, Any]]:
    chat_root = find_chatrwkv_root()
    cpp_root = find_rwkvcpp_root()
    cpp_dll = find_rwkvcpp_dll(cpp_root) if cpp_root is not None else None
    ggml: Path | None = None
    ggml_error: str | None = None
    if checkpoint is not None:
        try:
            candidate = _find_matching_ggml(checkpoint)
            if candidate.is_file():
                ggml = candidate
        except (FileNotFoundError, OSError, ValueError) as exc:
            ggml_error = str(exc)

    pack_report = (
        run_preflight(pack, backend="synthetic", checkpoint=checkpoint)
        if pack is not None
        else None
    )
    pack_ready = bool(pack_report and pack_report["passed"])
    rows: dict[str, dict[str, Any]] = {
        "rwkvcpp": {
            "label": "rwkvcpp",
            "ready": (
                cpp_dll is not None
                and ggml is not None
                and (pack is None or pack_ready)
            ),
            "speed": "Fast native CPU execution when the matching GGML model is present.",
            "compatibility": "Narrower artifact requirements. Needs the native library and a matching converted GGML model.",
            "details": [
                f"native library: {cpp_dll or 'not found'}",
                "GGML model: "
                f"{ggml or ggml_error or 'not found; supply --checkpoint or RWKVCPP_GGML_PATH'}",
            ],
        },
        "chatrwkv": {
            "label": "chatrwkv",
            "ready": (
                chat_root is not None
                and checkpoint is not None
                and checkpoint.exists()
                and checkpoint.suffix.lower() != ".bin"
                and (pack is None or pack_ready)
            ),
            "speed": "Usually slower than rwkvcpp, with PyTorch reference behavior.",
            "compatibility": "Useful for reference and compatibility runs. Needs ChatRWKV source and the original checkpoint.",
            "details": [
                f"ChatRWKV source: {chat_root or 'not found'}",
                "checkpoint: "
                f"{checkpoint if checkpoint and checkpoint.exists() and checkpoint.suffix.lower() != '.bin' else 'not found; supply the original .pth checkpoint'}",
            ],
        },
    }
    if pack is not None:
        for backend in rows:
            rows[backend]["preflight"] = pack_report
    return rows


def build_report(
    pack: Path | None = None, checkpoint: Path | None = None
) -> dict[str, Any]:
    if checkpoint is None:
        checkpoint = _source_checkpoint_from_pack(pack)
    memory = _total_memory_bytes()
    rows = _backend_rows(pack, checkpoint)
    if rows["rwkvcpp"]["ready"]:
        recommended = "rwkvcpp"
        reason = (
            "Its native library and matching GGML model are available, so it is "
            "the preferred CPU speed path."
        )
    elif rows["chatrwkv"]["ready"]:
        recommended = "chatrwkv"
        reason = (
            "The reference source and checkpoint are available. It is a good "
            "compatibility path, though usually slower than rwkvcpp."
        )
    else:
        recommended = None
        reason = "Neither real-model CPU backend has all required files. Review the checks below and supply the missing artifacts."
    return {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine() or "unknown",
            "logical_cpus": os.cpu_count() or 1,
            "memory_bytes": memory,
            "memory": _size_text(memory),
        },
        "pack": str(pack) if pack else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "recommended_backend": recommended,
        "recommendation_reason": reason,
        "backends": rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    system = report["system"]
    print("RWKV SSD CPU doctor")
    print(f"System: {system['platform']}")
    print(f"CPU: {system['processor']} ({system['logical_cpus']} logical cores)")
    print(f"RAM: {system['memory']}")
    if report["pack"]:
        print(f"Pack: {report['pack']}")
    if report["checkpoint"]:
        print(f"Checkpoint: {report['checkpoint']}")
    print()
    for key, label in (
        ("rwkvcpp", "rwkv.cpp native"),
        ("chatrwkv", "ChatRWKV reference"),
    ):
        row = report["backends"][key]
        state = "READY" if row["ready"] else "NEEDS FILES"
        print(f"{label}: {state}")
        print(f"  {row['speed']}")
        print(f"  {row['compatibility']}")
        for detail in row["details"]:
            print(f"  {detail}")
        preflight = row.get("preflight")
        if preflight and key == "chatrwkv":
            for check in preflight["checks"]:
                mark = "PASS" if check["ok"] else "FAIL"
                print(f"  {mark} {check['name']}: {check['detail']}")
        print()
    if report["recommended_backend"]:
        print(f"Recommended: {report['recommended_backend']}")
    else:
        print("Recommended: none yet")
    print(report["recommendation_reason"])
    if not report["pack"]:
        print("Tip: add --pack PATH to verify pack integrity and backend compatibility.")
    elif not report["checkpoint"]:
        print(
            "Tip: add --checkpoint PATH to verify the source checkpoint or find "
            "its matching GGML model."
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="Prepared runtime pack to verify")
    parser.add_argument(
        "--checkpoint", type=Path, help="Original checkpoint or matching GGML file"
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    report = build_report(args.pack, args.checkpoint)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    if report["pack"]:
        ready = any(row["ready"] for row in report["backends"].values())
        if not ready:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
