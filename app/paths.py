"""Application data locations and logging setup.

User data lives under ``%APPDATA%\\TintaPhoto`` (settings, presets, logs) and
regenerable data under ``%LOCALAPPDATA%\\TintaPhoto`` (thumbnail cache,
fallback sidecars), following PLAN.md sections 4 and 7.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

APP_NAME: Final[str] = "TintaPhoto"
APP_VERSION: Final[str] = "0.34.4"

_LOG_FILE_NAME: Final[str] = "tintaphoto.log"
_LOG_MAX_BYTES: Final[int] = 1_000_000
_LOG_BACKUP_COUNT: Final[int] = 3
_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def project_root() -> Path:
    """Return the directory containing the bundled resources.

    In a packaged build that is the folder PyInstaller unpacks the program
    into, which is where the ``resources`` folder is put as well.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return Path(__file__).resolve().parent.parent


def distribution_dir() -> Path:
    """Return the folder the program is distributed in.

    What the user sees: the folder holding the executable in a packaged
    build, and the project itself when running from source. The licence
    files live here rather than among the bundled resources, so that they
    can be read without opening the program (PLAN.md 13.5).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()



def resources_dir() -> Path:
    """Return the bundled ``resources`` directory."""
    return project_root() / "resources"


def devlog_file() -> Path:
    """Return the development log the menu command shows.

    It is the repository's own ``docs/DEVLOG.md``, bundled into the release
    beside the resources (see tintaphoto.spec), so that what the dialog
    shows and what the repository holds cannot drift apart.
    """
    return project_root() / "docs" / "DEVLOG.md"


def roaming_dir() -> Path:
    """Return ``%APPDATA%\\TintaPhoto``, falling back to the home directory."""
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    return root / APP_NAME


def local_dir() -> Path:
    """Return ``%LOCALAPPDATA%\\TintaPhoto``, falling back to the home directory."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / APP_NAME


def settings_file() -> Path:
    """Return the path of the application settings file (PLAN.md 7.3)."""
    return roaming_dir() / "settings.json"


def presets_dir() -> Path:
    """Return the directory holding preset files (PLAN.md 7.2)."""
    return roaming_dir() / "presets"


def filters_dir() -> Path:
    """Return the directory holding the film looks of the looks tab.

    They ship with the program rather than belonging to the user: each one
    was measured from a photograph and is part of what the program renders
    with, like the colour matrices (:mod:`core.filters`).
    """
    return resources_dir() / "filters"


def logs_dir() -> Path:
    """Return the directory holding log files."""
    return roaming_dir() / "logs"


def thumbnail_cache_dir() -> Path:
    """Return the thumbnail cache directory (PLAN.md section 4)."""
    return local_dir() / "cache" / "thumbs"


def fallback_sidecar_dir() -> Path:
    """Return the sidecar directory used for unwritable folders (PLAN.md 7.1)."""
    return local_dir() / "sidecars"


def ensure_dir(path: Path) -> bool:
    """Create *path* and its parents if needed; return whether it is usable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return True


def setup_logging(level: int = logging.INFO) -> None:
    """Send log records to a rotating file in :func:`logs_dir`.

    Failing to open the log file must never prevent the application from
    starting, so any error here falls back to logging to the console only.
    """
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    directory = logs_dir()
    if ensure_dir(directory):
        try:
            file_handler = RotatingFileHandler(
                directory / _LOG_FILE_NAME,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
        except OSError:
            pass
        else:
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
