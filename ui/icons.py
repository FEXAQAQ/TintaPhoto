"""Loading and recolouring of the SVG line icons in ``resources/icons``.

Icons are stored as SVG with ``stroke="currentColor"`` so that one file can be
rendered in any of the theme's text colours (PLAN.md 3.5). They are rasterised
at the device pixel ratio of the current screen so that they stay sharp at
125%, 150% and 200% display scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from app.paths import resources_dir
from ui import theme

logger = logging.getLogger(__name__)

_CURRENT_COLOR: Final[str] = "currentColor"

#: Display scale factors an icon is rasterised for, so that Windows display
#: scaling never has to stretch one (PLAN.md 3.8: 125%, 150%, 200%).
_SCALE_FACTORS: Final[tuple[float, ...]] = (1.0, 1.25, 1.5, 2.0)

_pixmap_cache: dict[tuple[str, str, int], QPixmap] = {}
_source_cache: dict[str, str] = {}

# Icon names, so callers never repeat a file name string.
ICON_MENU: Final[str] = "menu"
ICON_UNDO: Final[str] = "undo"
ICON_REDO: Final[str] = "redo"
ICON_SLIDERS: Final[str] = "sliders"
ICON_CROP: Final[str] = "crop"
ICON_FILTERS: Final[str] = "filters"
ICON_RETOUCH: Final[str] = "retouch"
ICON_INFO: Final[str] = "info"
ICON_FOLDER: Final[str] = "folder"
#: A framed picture: the welcome screen's "open file" button.
ICON_IMAGE: Final[str] = "image"
#: A cross: takes a folder off the welcome screen's recent list.
ICON_X: Final[str] = "x"
#: The mark on an edited thumbnail; drawn small, so it has thicker strokes
#: and filled dots rather than the 1.5 px outline of the others.
ICON_EDITED: Final[str] = "edited"
#: The arrow on a collapsible group's header: pointing down while the group
#: is open, to the right while it is closed (PLAN.md 3.1).
ICON_CHEVRON_DOWN: Final[str] = "chevron-down"
ICON_CHEVRON_RIGHT: Final[str] = "chevron-right"
#: The arrow on the filmstrip's handle while the strip is folded away: it
#: points the way the strip will come back (PLAN.md 3.1).
ICON_CHEVRON_UP: Final[str] = "chevron-up"
ICON_TURN_LEFT: Final[str] = "turn-left"
ICON_TURN_RIGHT: Final[str] = "turn-right"
#: Two triangles either side of a dashed axis: the crop tab's flips (C-08).
ICON_FLIP_HORIZONTAL: Final[str] = "flip-horizontal"
ICON_FLIP_VERTICAL: Final[str] = "flip-vertical"
#: A frame split in two: the toolbar button that opens the collage.
ICON_COLLAGE: Final[str] = "collage"
#: The white balance eyedropper on the white balance group's header (A-24).
ICON_PIPETTE: Final[str] = "pipette"
#: Saves the current settings as a preset, on the presets group's header.
ICON_PLUS: Final[str] = "plus"
#: The automatic light, on the light group's header: a wand and three sparks.
ICON_WAND: Final[str] = "wand"
#: The spot removal tool, on the spot group's header (A-32): a blemish
#: fading out, drawn dashed, with a spark where it was.
ICON_SPOT: Final[str] = "spot"


def icons_dir() -> Path:
    """Return the directory holding the icon files."""
    return resources_dir() / "icons"


def app_icon_file() -> Path:
    """Return the application's own icon, the one Windows shows."""
    return resources_dir() / "app.ico"


def app_icon() -> QIcon:
    """Return the application icon, or an empty one if it is not there.

    Unlike the line icons, this one is a drop of ink carrying the whole
    range of colour, drawn in ``resources/icons/app.svg`` and rendered to
    ``resources/app.ico`` by ``tools/make_app_icon.py``. A missing file
    leaves the window with the system's default icon rather than stopping
    the program.
    """
    path = app_icon_file()
    if not path.is_file():
        logger.warning("The application icon %s is missing", path)
        return QIcon()
    return QIcon(str(path))



def _read_source(name: str) -> str:
    """Return the SVG source of *name*, or an empty string when unreadable."""
    if name in _source_cache:
        return _source_cache[name]
    path = icons_dir() / f"{name}.svg"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Could not read icon %s", path)
        source = ""
    _source_cache[name] = source
    return source


def render_pixmap(name: str, color: str, pixels: int) -> QPixmap:
    """Rasterise icon *name* in *color* into a square of *pixels* device pixels."""
    key = (name, color, pixels)
    cached = _pixmap_cache.get(key)
    if cached is not None:
        return cached

    source = _read_source(name).replace(_CURRENT_COLOR, color)
    pixmap = QPixmap(pixels, pixels)
    pixmap.fill(Qt.GlobalColor.transparent)

    if source:
        renderer = QSvgRenderer(source.encode("utf-8"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        renderer.render(painter, QRectF(0, 0, pixels, pixels))
        painter.end()

    _pixmap_cache[key] = pixmap
    return pixmap


def load_icon(
    name: str,
    color: str = theme.TEXT_SECONDARY,
    active_color: str = theme.TEXT_PRIMARY,
    disabled_color: str = theme.TEXT_TERTIARY,
    size: int = theme.ICON_SIZE,
) -> QIcon:
    """Return icon *name* with a pixmap per state and per display scale factor.

    One pixmap is rasterised for each scale factor in :data:`_SCALE_FACTORS`,
    so Qt can pick an exact match instead of stretching a smaller one when the
    display is scaled.
    """
    modes = (
        (QIcon.Mode.Normal, color),
        (QIcon.Mode.Active, active_color),
        (QIcon.Mode.Selected, active_color),
        (QIcon.Mode.Disabled, disabled_color),
    )
    icon = QIcon()
    for mode, mode_color in modes:
        for factor in _SCALE_FACTORS:
            pixels = max(1, round(size * factor))
            icon.addPixmap(render_pixmap(name, mode_color, pixels), mode)
    return icon
