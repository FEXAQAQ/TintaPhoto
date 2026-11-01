"""Tests for the icon files (PLAN.md 3.5).

The loader itself needs a Qt application, so these tests check the files: every
icon referenced by the code exists, is valid SVG, draws with ``currentColor``
so it can be recoloured, and uses the documented 1.5 px stroke.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ui import icons, theme

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = PROJECT_ROOT / "resources" / "icons"

#: Every icon the interface asks for, taken from the ICON_* constants in
#: ui/icons.py rather than listed again here: a second list would only say
#: what it said when it was written.
REQUIRED_ICONS = tuple(
    sorted(
        value
        for name, value in vars(icons).items()
        if name.startswith("ICON_") and isinstance(value, str)
    )
)


#: The disclosure arrows of the collapsible groups. A chevron is a flat glyph
#: by nature and is drawn at half the size of the others, beside a 12 px
#: title, so it covers a third of the grid across its short axis where the
#: rest cover most of theirs.
ARROWS = ("chevron-down", "chevron-right", "chevron-up")


#: The application's own icon is not a line icon: it is filled, carries a
#: gradient and is drawn on a 256 grid, so the rules below are not its.
APP_ICON = "app.svg"


def _icon_files() -> list[Path]:
    return sorted(
        path for path in ICONS_DIR.glob("*.svg") if path.name != APP_ICON
    )


@pytest.mark.parametrize("name", REQUIRED_ICONS)
def test_required_icon_exists(name: str) -> None:
    assert (ICONS_DIR / f"{name}.svg").is_file()


def test_every_icon_is_valid_svg() -> None:
    for path in _icon_files():
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg"), path.name


def test_every_icon_can_be_recoloured_and_uses_the_documented_stroke() -> None:
    for path in _icon_files():
        root = ET.parse(path).getroot()
        assert root.get("stroke") == "currentColor", path.name
        assert root.get("stroke-width") == str(theme.ICON_STROKE_WIDTH), path.name
        assert root.get("viewBox") == "0 0 24 24", path.name


def _drawn_bounds(pixels: int, name: str) -> tuple[float, float, float, float]:
    """Return the drawn area of an icon as fractions of its box."""
    from ui.icons import render_pixmap

    image = render_pixmap(name, theme.TEXT_PRIMARY, pixels).toImage()
    xs = [x for x in range(pixels) for y in range(pixels) if image.pixelColor(x, y).alpha()]
    ys = [y for y in range(pixels) for x in range(pixels) if image.pixelColor(x, y).alpha()]
    assert xs and ys, f"{name} rendered nothing at {pixels} px"
    return (
        min(xs) / pixels,
        min(ys) / pixels,
        (max(xs) + 1) / pixels,
        (max(ys) + 1) / pixels,
    )


@pytest.mark.parametrize("name", REQUIRED_ICONS)
def test_icons_are_not_clipped_at_any_display_scale(name: str, qt_app: object) -> None:
    """The drawing must occupy the same relative box at every size.

    Rendering an icon into a box of the wrong size clips it against the icon
    rectangle, which shows up as a different relative bounding box.
    """
    small = _drawn_bounds(theme.ICON_SIZE, name)
    large = _drawn_bounds(theme.ICON_SIZE * 2, name)
    for edge_small, edge_large in zip(small, large):
        assert abs(edge_small - edge_large) < 0.06, (name, small, large)
    left, top, right, bottom = large
    assert right <= 1.0 and bottom <= 1.0, (name, large)
    # Every icon is drawn on the same 24 unit grid and spans most of it; the
    # flattest one is the menu icon, whose three lines cover half the height.
    across, down = (0.3, 0.3) if name in ARROWS else (0.6, 0.5)
    assert right - left > across and bottom - top >= down, (name, large)


# -- the application's own icon ------------------------------------------


def test_the_application_icon_is_there_and_has_every_size(qt_app: object) -> None:
    """Windows draws it at 16 px in a title bar and at 256 in Explorer."""
    from PIL import Image

    from ui import icons

    path = icons.app_icon_file()
    assert path.is_file(), path
    with Image.open(path) as drawn:
        sizes = {size for size, _ in drawn.ico.sizes()}
    for wanted in (16, 32, 48, 256):
        assert wanted in sizes, f"{wanted} px is missing from app.ico"


def test_the_application_icon_loads(qt_app: object) -> None:
    from ui import icons

    icon = icons.app_icon()
    assert not icon.isNull()
    assert not icon.pixmap(16, 16).isNull()
    assert not icon.pixmap(256, 256).isNull()


def test_the_icon_is_drawn_rather_than_fetched(qt_app: object) -> None:
    """Its source is an SVG in the project, like every other icon."""
    from ui import icons

    source = icons.icons_dir() / APP_ICON
    assert source.is_file()
    assert "linearGradient" in source.read_text(encoding="utf-8")
