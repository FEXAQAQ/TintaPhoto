"""The thumbnail cache on disk (PLAN.md section 4).

Thumbnails live in ``%LOCALAPPDATA%\\TintaPhoto\\cache\\thumbs`` as small
JPEGs. An entry is keyed by the photo's full path, its size and the time it
was last changed, so editing or replacing a photo never shows a stale picture.
When the folder grows past its limit, the entries that have not been looked at
for longest are deleted.

Nothing here is essential: if the cache cannot be written, thumbnails are
simply made again next time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from core import thumbnails

logger = logging.getLogger(__name__)


def _ensure_directory(path: Path) -> bool:
    """Create *path* if needed; return whether it can be used.

    The cache folder is chosen by the controller layer and passed in, because
    nothing in ``core`` is allowed to know where application data lives.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return True


#: Cache size limits offered in the settings dialog (PLAN.md section 3.10).
CACHE_LIMITS_BYTES: Final[tuple[int, ...]] = (
    500 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    5 * 1024 * 1024 * 1024,
)
DEFAULT_CACHE_LIMIT_BYTES: Final[int] = CACHE_LIMITS_BYTES[1]

#: How much of the limit is freed when the cache has to be trimmed, so that
#: trimming happens rarely rather than on almost every write.
_TRIM_TO: Final[float] = 0.85

#: How much may be written before the cache looks at its own size again.
#: Checking means listing and measuring every entry, which is far too much
#: work to do after each thumbnail: filling a folder of three hundred photos
#: would do it three hundred times, and because it is plain Python it blocks
#: the interface thread as well as the one doing the work.
_CHECK_AFTER_BYTES: Final[int] = 16 * 1024 * 1024

_SUFFIX: Final[str] = ".jpg"

#: Bumped whenever thumbnails start being made differently, so that entries
#: from an older version of the program are not shown. The orientation fix of
#: 2026-10-20 is why this exists.
_THUMBNAIL_VERSION: Final[int] = 2


class ThumbnailCache:
    """A size limited store of thumbnails on disk."""

    def __init__(
        self,
        directory: Path,
        limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    ) -> None:
        self._directory = directory
        self._limit_bytes = limit_bytes
        self._written_since_check = 0
        self._usable = _ensure_directory(self._directory)
        if not self._usable:
            logger.warning("The thumbnail cache at %s cannot be used", self._directory)

    @property
    def directory(self) -> Path:
        """Return the folder the thumbnails are kept in."""
        return self._directory

    @property
    def limit_bytes(self) -> int:
        """Return the size the cache is trimmed back to."""
        return self._limit_bytes

    def set_limit(self, limit_bytes: int) -> None:
        """Change the size limit and trim at once if it is now exceeded."""
        self._limit_bytes = limit_bytes
        self._written_since_check = 0
        self.trim()

    # -- keys ------------------------------------------------------------

    @staticmethod
    def key(path: Path) -> str | None:
        """Return the cache key of *path*: its identity plus its state.

        Returns ``None`` when the photo cannot be inspected, which means it
        cannot be cached either. The key also carries the version of the
        thumbnail format, so that changing how thumbnails are made retires the
        old ones instead of showing them.
        """
        try:
            stat = path.stat()
        except OSError:
            return None
        identity = (
            f"v{_THUMBNAIL_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        )
        return hashlib.sha1(identity.encode("utf-8", "replace")).hexdigest()

    def _entry(self, key: str) -> Path:
        return self._directory / f"{key}{_SUFFIX}"

    # -- reading and writing ---------------------------------------------

    def get(self, path: Path) -> NDArray[np.uint8] | None:
        """Return the stored thumbnail of *path*, or ``None`` if there is none."""
        if not self._usable:
            return None
        key = self.key(path)
        if key is None:
            return None
        entry = self._entry(key)
        try:
            data = entry.read_bytes()
        except OSError:
            return None

        thumbnail = thumbnails.decode(data)
        if thumbnail is None:
            logger.info("A damaged thumbnail was discarded")
            self._remove(entry)
            return None
        self._touch(entry)
        return thumbnail

    def put(self, path: Path, thumbnail: NDArray[np.uint8]) -> bool:
        """Store *thumbnail* for *path*; return whether it was written."""
        if not self._usable:
            return False
        key = self.key(path)
        if key is None:
            return False
        data = thumbnails.encode(thumbnail)
        if data is None:
            return False

        entry = self._entry(key)
        temporary = entry.with_suffix(entry.suffix + ".tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, entry)
        except OSError as error:
            logger.info("A thumbnail could not be cached: %s", error)
            self._remove(temporary)
            return False

        self._written_since_check += len(data)
        if self._written_since_check >= _CHECK_AFTER_BYTES:
            self._written_since_check = 0
            self.trim()
        return True

    # -- housekeeping ----------------------------------------------------

    def usage_bytes(self) -> int:
        """Return how much space the cached thumbnails take up."""
        return sum(size for _, _, size in self._entries())

    def clear(self) -> None:
        """Delete every cached thumbnail."""
        for entry, _, _ in self._entries():
            self._remove(entry)
        self._written_since_check = 0

    def trim(self) -> None:
        """Delete the least recently used entries until the cache fits."""
        entries = self._entries()
        total = sum(size for _, _, size in entries)
        if total <= self._limit_bytes:
            return

        target = int(self._limit_bytes * _TRIM_TO)
        entries.sort(key=lambda item: item[1])  # oldest use first
        for entry, _, size in entries:
            if total <= target:
                break
            if self._remove(entry):
                total -= size
        logger.info("The thumbnail cache was trimmed to %.0f MB", total / 1024 / 1024)

    def _entries(self) -> list[tuple[Path, float, int]]:
        """Return every cached file with the time it was last used and its size."""
        if not self._usable:
            return []
        found: list[tuple[Path, float, int]] = []
        try:
            for entry in self._directory.glob(f"*{_SUFFIX}"):
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                found.append((entry, stat.st_mtime, stat.st_size))
        except OSError as error:
            logger.info("The thumbnail cache could not be listed: %s", error)
        return found

    @staticmethod
    def _touch(entry: Path) -> None:
        """Mark an entry as used just now, which is what orders the trimming."""
        try:
            now = time.time()
            os.utime(entry, (now, now))
        except OSError:
            pass

    @staticmethod
    def _remove(entry: Path) -> bool:
        try:
            entry.unlink(missing_ok=True)
        except OSError:
            return False
        return True
