"""Content-addressed transcript prefix splitting for stateless clients."""

from __future__ import annotations

import hashlib
from collections.abc import Callable


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_transcript(rendered: str) -> tuple[str, str, str]:
    """Split a rendered chat transcript into cacheable prefix and fresh suffix.

    Returns ``(cache_key, stable_prefix, dynamic_suffix)``. The stable prefix
    is everything before the final ``user:`` turn; the suffix is the last user
    turn (and anything after it). When no boundary exists the whole prompt is
    treated as dynamic and no stable cache key is produced.
    """
    text = rendered.strip("\n")
    if not text:
        return "", "", ""
    marker = "\nuser:"
    idx = text.rfind(marker)
    if idx < 0:
        if text.startswith("user:"):
            normalized = text if text.endswith("\n") else text + "\n"
            return sha256_text(normalized), normalized, ""
        return "", "", text
    stable = text[:idx].strip("\n")
    suffix = text[idx + 1 :].strip("\n")
    if not stable:
        return "", "", text
    return sha256_text(stable + "\n"), stable + "\n", suffix


def longest_cached_prefix(
    rendered: str,
    *,
    has_entry: Callable[[str], bool],
) -> tuple[str, str, str]:
    """Find the longest cached transcript prefix using ``has_entry(key)``."""
    text = rendered.strip("\n")
    if not text:
        return "", "", ""
    parts: list[str] = []
    best_key = ""
    best_prefix = ""
    for line in text.split("\n"):
        parts.append(line)
        candidate = "\n".join(parts)
        key = sha256_text(candidate + "\n")
        if has_entry(key):
            best_key = key
            best_prefix = candidate + "\n"
    if best_key:
        suffix = text[len(best_prefix.rstrip("\n")) :].lstrip("\n")
        return best_key, best_prefix, suffix
    return split_transcript(rendered)
