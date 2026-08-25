"""Resolve runtime pack variants and certificate-gated default selectors."""

from __future__ import annotations

import os
from pathlib import Path

_PROFILE_ALIASES = {
    "lut2": "lut2",
    "lut": "lut2",
    "grouped": "grouped",
    "grouped_u8": "grouped",
    "grouped-u8": "grouped",
    "shadow_sel": "shadow_sel",
    "shadow-selective": "shadow_sel",
    "selective_shadow": "shadow_sel",
    "tiered_hot3": "tiered_hot3",
    "tiered": "tiered_hot3",
}

_DEFAULT_2_9B_PACK = "runtime_pack_2.9b"
_GROUPED_2_9B_PACK = "runtime_pack_2.9b_grouped_quality"


def pack_profile_from_env() -> str:
    raw = os.environ.get("RWKV_PACK_PROFILE", "auto").strip().lower()
    return _PROFILE_ALIASES.get(raw, raw)


def sibling_pack(pack_dir: Path, name: str) -> Path | None:
    candidate = pack_dir.parent / name
    return candidate if candidate.is_dir() else None


def _trinity_variant_sibling(pack_dir: Path, prefix: str) -> Path | None:
    """Find a named Trinity variant beside a historical LUT2 request."""
    base_name = pack_dir.name
    if base_name.startswith("trinity_lut2_"):
        suffix = base_name[len("trinity_lut2_"):]
        exact = sibling_pack(pack_dir, f"{prefix}_{suffix}")
        if exact is not None:
            return exact
    return sibling_pack(pack_dir, f"{prefix}_0.1b")


def resolve_trinity_pack(pack_dir: Path, profile: str | None = None) -> Path:
    """
    Pick a Trinity pack directory when a sibling variant exists.

    Profiles:
    - ``auto``: use ``RWKV_PACK_PROFILE``; historical LUT2 requests prefer a
      *certified* grouped-U8 compact pack, then the dense correctness fallback
    - ``shadow_sel``: selective bf16 shadow for large tensors (fast memcpy staging)
    - ``grouped``: all-tensor grouped-U8 compact pack
    - ``lut2``: keep ``pack_dir`` as-is

    The same policy also covers the local 2.9B release names: the stable
    ``runtime_pack_2.9b`` path promotes its certified grouped-U8 sibling in
    ``auto`` mode, while an explicit ``grouped`` profile selects that sibling
    for diagnostics.  Selection never bypasses Manifest's certificate check.
    """
    pack_dir = Path(pack_dir)
    profile = profile or pack_profile_from_env()
    if pack_dir.name == _DEFAULT_2_9B_PACK:
        grouped = sibling_pack(pack_dir, _GROUPED_2_9B_PACK)
        if profile == "grouped":
            if grouped is not None:
                return grouped
        elif profile == "auto":
            # Default promotion is certificate-gated.  The caller will still
            # verify the certificate contents and artifact hashes when the
            # resolved manifest is loaded.
            if (
                grouped is not None
                and (grouped / "quality_certificate.json").is_file()
            ):
                return grouped
    if profile == "grouped":
        grouped = _trinity_variant_sibling(pack_dir, "trinity_grouped")
        if grouped is not None:
            return grouped
    if profile == "shadow_sel":
        # trinity_lut2_shadow_sel_0.1b next to trinity_lut2_0.1b
        base_name = pack_dir.name
        if base_name.startswith("trinity_lut2_"):
            suffix = base_name[len("trinity_lut2_"):]
            sel = sibling_pack(pack_dir, f"trinity_lut2_shadow_sel_{suffix}")
            if sel is not None:
                return sel
        generic = sibling_pack(pack_dir, "trinity_lut2_shadow_sel_0.1b")
        if generic is not None:
            return generic
    if profile == "tiered_hot3":
        base_name = pack_dir.name
        if base_name.startswith("trinity_lut2_"):
            suffix = base_name[len("trinity_lut2_"):]
            tiered = sibling_pack(pack_dir, f"trinity_tiered_hot3_{suffix}")
            if tiered is not None:
                return tiered
        generic = sibling_pack(pack_dir, "trinity_tiered_hot3_0.1b")
        if generic is not None:
            return generic
    if profile == "auto" and pack_dir.name.startswith("trinity_lut2_"):
        # A certified grouped-U8 pack is the compact default for historical
        # LUT2 requests.  It is intentionally a sibling rather than an
        # in-place rewrite so old results remain reproducible.  If it is
        # unavailable or uncertified, retain the exact-parity dense fallback
        # behavior.  Explicit
        # ``RWKV_PACK_PROFILE=lut2`` still reaches the legacy pack and its
        # certificate gate instead of silently changing representations.
        grouped = _trinity_variant_sibling(pack_dir, "trinity_grouped")
        if grouped is not None and (grouped / "quality_certificate.json").is_file():
            return grouped
        safe = sibling_pack(pack_dir, "trinity_safe_0.1b")
        if safe is not None and not (pack_dir / "quality_certificate.json").is_file():
            return safe
    return pack_dir
