"""POSIX fadvise on file descriptors (Linux read-ahead hint for pread path)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_POSIX_FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", None)
_POSIX_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)


def fadvise_willneed(fd: int, offset: int, length: int) -> bool:
    if _POSIX_FADV_WILLNEED is None or length <= 0:
        return False
    try:
        os.posix_fadvise(fd, offset, length, _POSIX_FADV_WILLNEED)
        return True
    except (OSError, AttributeError) as exc:
        logger.debug("posix_fadvise WILLNEED skipped: %s", exc)
        return False


def fadvise_dontneed(fd: int, offset: int, length: int) -> bool:
    if _POSIX_FADV_DONTNEED is None or length <= 0:
        return False
    try:
        os.posix_fadvise(fd, offset, length, _POSIX_FADV_DONTNEED)
        return True
    except (OSError, AttributeError) as exc:
        logger.debug("posix_fadvise DONTNEED skipped: %s", exc)
        return False
