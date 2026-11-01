"""Small Windows specific helpers used by the interface.

A few things cannot be expressed through Qt alone: forcing the native title
bar into dark mode (PLAN.md 3.6), asking Windows whether the user has turned
animation effects off, in which case all transitions must be immediate, and
showing an exported file in Explorer. All of them degrade silently on other
platforms and older builds.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import sys
from pathlib import Path
from typing import Final

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

logger = logging.getLogger(__name__)


#: DWMWA_USE_IMMERSIVE_DARK_MODE, and the value used by Windows 10 builds
#: before 19041 where the attribute had a different index.
_DWMWA_USE_IMMERSIVE_DARK_MODE: Final[int] = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY: Final[int] = 19

#: SPI_GETCLIENTAREAANIMATION
_SPI_GET_CLIENT_AREA_ANIMATION: Final[int] = 0x1042


def is_windows() -> bool:
    """Return whether the application is running on Windows."""
    return sys.platform == "win32"


def apply_dark_title_bar(window_id: int) -> bool:
    """Force the native title bar of *window_id* into dark mode.

    Returns whether the attribute was accepted. Failure is not an error: on
    other platforms, or on Windows builds without the attribute, the title bar
    simply stays light.
    """
    if not is_windows():
        return False
    try:
        dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        logger.debug("dwmapi is not available; leaving the title bar as it is")
        return False

    enabled = ctypes.c_int(1)
    handle = ctypes.c_void_p(window_id)
    for attribute in (_DWMWA_USE_IMMERSIVE_DARK_MODE, _DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY):
        result = dwmapi.DwmSetWindowAttribute(
            handle, ctypes.c_int(attribute), ctypes.byref(enabled), ctypes.sizeof(enabled)
        )
        if result == 0:
            return True
    logger.debug("The dark title bar attribute was rejected by this Windows build")
    return False


def display_profile() -> bytes | None:
    """Return the colour profile Windows has for the current screen.

    Returns ``None`` when there is none, which simply means the preview is
    shown as plain sRGB (PLAN.md section 6 step 13).
    """
    try:
        from PIL import ImageCms

        profile = ImageCms.get_display_profile()
    except (ImportError, OSError, ValueError, AttributeError):
        return None
    if profile is None:
        return None
    try:
        return profile.tobytes()
    except (OSError, ValueError, AttributeError):
        return None


def animations_enabled() -> bool:
    """Return whether Windows animation effects are turned on.

    When the user has turned them off, every transition must switch instantly
    instead of animating.
    """
    if not is_windows():
        return True
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return True
    value = ctypes.c_int(1)
    ok = user32.SystemParametersInfoW(
        ctypes.c_uint(_SPI_GET_CLIENT_AREA_ANIMATION),
        ctypes.c_uint(0),
        ctypes.byref(value),
        ctypes.c_uint(0),
    )
    if not ok:
        return True
    return bool(value.value)


def reveal(path: Path) -> bool:
    """Show *path* in Explorer, with the file itself selected.

    Offered when an export finishes (PLAN.md 3.11). Returns whether Explorer
    was asked; if it could not be, the folder is opened instead, and if that
    fails too nothing happens but a line in the log.
    """
    if is_windows():
        try:
            # Explorer takes the file to select as part of one argument and
            # reports a non zero exit code even when it works, so neither the
            # shell nor the result is of any use here.
            subprocess.Popen(["explorer", f"/select,{path}"])  # noqa: S603, S607
            return True
        except OSError:
            logger.info("Explorer could not be opened for %s", path)
    folder = path.parent
    if QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
        return True
    logger.info("The folder %s could not be opened", folder)
    return False
