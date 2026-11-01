"""Tests for the filmstrip, and the mark on an edited photo (PLAN.md 3.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QColor  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from ui import filmstrip as filmstrip_module  # noqa: E402
from ui.filmstrip import Filmstrip, Thumbnail  # noqa: E402

PHOTOS = [Path(f"D:/Photos/DSC0000{index}.ARW") for index in range(3)]


@pytest.fixture
def filmstrip(qt_app: object) -> Filmstrip:
    strip = Filmstrip()
    strip.set_photos(PHOTOS)
    return strip


def _thumbnail(strip: Filmstrip, path: Path) -> Thumbnail:
    found = strip.findChildren(Thumbnail)
    matching = [item for item in found if item.path == path]
    assert matching, f"{path.name} is not in the strip"
    return matching[0]


def test_only_the_edited_photos_are_marked(filmstrip: Filmstrip) -> None:
    filmstrip.set_edited_photos({PHOTOS[1]})

    assert not _thumbnail(filmstrip, PHOTOS[0]).edited
    assert _thumbnail(filmstrip, PHOTOS[1]).edited
    assert not _thumbnail(filmstrip, PHOTOS[2]).edited


def test_one_photo_can_be_marked_and_unmarked(filmstrip: Filmstrip) -> None:
    filmstrip.set_edited(PHOTOS[0], True)
    assert _thumbnail(filmstrip, PHOTOS[0]).edited

    filmstrip.set_edited(PHOTOS[0], False)
    assert not _thumbnail(filmstrip, PHOTOS[0]).edited


def test_marking_a_photo_that_is_not_in_the_strip_is_harmless(
    filmstrip: Filmstrip,
) -> None:
    filmstrip.set_edited(Path("D:/elsewhere/DSC09999.ARW"), True)


def test_the_mark_says_so_in_the_tooltip(filmstrip: Filmstrip) -> None:
    thumbnail = _thumbnail(filmstrip, PHOTOS[0])
    assert thumbnail.toolTip() == PHOTOS[0].name

    thumbnail.set_edited(True)
    assert thumbnail.toolTip() == tr("filmstrip.edited", name=PHOTOS[0].name)


def test_the_mark_is_drawn_in_the_bottom_right_corner(qt_app: object) -> None:
    """PLAN.md 3.1: 已編輯的照片右下角有小圖示."""
    thumbnail = Thumbnail(PHOTOS[0])
    # An even grey, so that anything drawn over it is a difference.
    thumbnail.set_picture(np.full((72, 96, 3), 128, dtype=np.uint8))

    plain = thumbnail.grab().toImage()
    thumbnail.set_edited(True)
    marked = thumbnail.grab().toImage()

    corner = (
        thumbnail.width() - theme.THUMBNAIL_BADGE_SIZE // 2 - theme.SPACE_XS,
        thumbnail.height() - theme.THUMBNAIL_BADGE_SIZE // 2 - theme.SPACE_XS,
    )
    assert marked.pixelColor(*corner) != plain.pixelColor(*corner)
    # The rest of the picture is untouched: the mark is a corner, not a tint.
    middle = (thumbnail.width() // 4, thumbnail.height() // 2)
    assert marked.pixelColor(*middle) == plain.pixelColor(*middle)


def test_the_current_photo_is_outlined_in_the_accent_colour(qt_app: object) -> None:
    """PLAN.md 3.5: the accent colour is for the current state, and which
    photo is being edited is the one piece of current state in the strip."""
    thumbnail = Thumbnail(PHOTOS[0])
    thumbnail.set_picture(np.full((72, 96, 3), 128, dtype=np.uint8))

    thumbnail.set_selected(True)
    chosen = thumbnail.grab().toImage()
    thumbnail.set_current(True)
    current = thumbnail.grab().toImage()

    middle = thumbnail.height() // 2
    assert current.pixelColor(0, middle) == QColor(theme.ACCENT)
    # A photo merely chosen alongside it is marked more quietly.
    assert chosen.pixelColor(0, middle) == QColor(theme.TEXT_SECONDARY)


def test_the_position_in_the_folder_sits_on_a_patch_of_its_own(
    filmstrip: Filmstrip,
) -> None:
    """PLAN.md 3.1: the position is shown at the right of the strip."""
    filmstrip.set_current(0, len(PHOTOS))
    label = filmstrip.findChild(QLabel, "FilmstripPosition")
    assert label is not None
    assert label.text() == tr("filmstrip.position", index=1, total=len(PHOTOS))
    assert label.height() >= theme.CHIP_HEIGHT


def test_a_thumbnail_marks_the_pointer_and_the_press(qt_app: object) -> None:
    """PLAN.md 3.8: every clickable element has a hover and a pressed state."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QEnterEvent, QMouseEvent
    from PySide6.QtWidgets import QApplication

    thumbnail = Thumbnail(PHOTOS[0])
    thumbnail.set_picture(np.full((72, 96, 3), 128, dtype=np.uint8))
    middle = thumbnail.height() // 2
    plain = thumbnail.grab().toImage().pixelColor(0, middle)

    where = QPointF(5, 5)
    QApplication.sendEvent(thumbnail, QEnterEvent(where, where, where))
    assert thumbnail.grab().toImage().pixelColor(0, middle) == QColor(
        theme.TEXT_SECONDARY
    )

    QApplication.sendEvent(
        thumbnail,
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            where,
            thumbnail.mapToGlobal(where),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    assert thumbnail.grab().toImage().pixelColor(0, middle) == QColor(
        theme.TEXT_PRIMARY
    )

    QApplication.sendEvent(thumbnail, QEvent(QEvent.Type.Leave))
    assert thumbnail.grab().toImage().pixelColor(0, middle) == plain


def test_the_current_photo_outranks_the_pointer(qt_app: object) -> None:
    """Hovering the photo being edited must not weaken its own mark."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication

    thumbnail = Thumbnail(PHOTOS[0])
    thumbnail.set_picture(np.full((72, 96, 3), 128, dtype=np.uint8))
    thumbnail.set_current(True)
    where = QPointF(5, 5)
    QApplication.sendEvent(thumbnail, QEnterEvent(where, where, where))

    middle = thumbnail.height() // 2
    assert thumbnail.grab().toImage().pixelColor(0, middle) == QColor(theme.ACCENT)


def test_a_thumbnail_is_rounded_at_its_corners(qt_app: object) -> None:
    """The picture used to be square inside a rounded block, so the corner
    was only ever rounded while the thumbnail was still loading."""
    thumbnail = Thumbnail(PHOTOS[0])
    thumbnail.set_picture(np.full((72, 96, 3), 255, dtype=np.uint8))
    picture = thumbnail.grab().toImage()

    # The very corner is outside the rounded shape, so the photo is not there.
    assert picture.pixelColor(0, 0) != QColor(theme.WHITE)
    # A little way in, it is.
    inside = theme.RADIUS_THUMB + theme.SPACE_XS
    assert picture.pixelColor(inside, inside) == QColor(theme.WHITE)


# -- the wheel (PLAN.md 3.3: on the filmstrip it scrolls) ------------------

MANY = [Path(f"D:/Photos/DSC1{index:04}.ARW") for index in range(40)]


@pytest.fixture
def long_strip(qt_app: object) -> Filmstrip:
    """A strip with far more photos than fit, shown so that it can scroll."""
    strip = Filmstrip()
    strip.set_photos(MANY)
    strip.resize(800, theme.FILMSTRIP_HEIGHT)
    strip.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    return strip


def _turn_wheel(target, notches: int = 1, sideways: bool = False) -> None:
    """Send *target* the turn of an ordinary wheel, downwards for positive."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    angle = QPoint(-120 * notches, 0) if sideways else QPoint(0, -120 * notches)
    point = QPointF(target.width() / 2, target.height() / 2)
    event = QWheelEvent(
        point,
        target.mapToGlobal(point),
        QPoint(),
        angle,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(target, event)


def _bar(strip: Filmstrip):
    return strip._scroll.horizontalScrollBar()


def test_the_wheel_moves_the_strip_sideways(
    long_strip: Filmstrip, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upright wheel used to go to a scroll bar the strip does not have."""
    from PySide6.QtWidgets import QApplication

    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    bar = _bar(long_strip)
    assert bar.maximum() > 0

    _turn_wheel(long_strip._scroll.viewport())
    step = QApplication.wheelScrollLines() * filmstrip_module.WHEEL_LINE
    assert bar.value() == step

    _turn_wheel(long_strip._scroll.viewport(), notches=-1)
    assert bar.value() == 0


def test_a_sideways_wheel_moves_it_the_same_way(
    long_strip: Filmstrip, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    _turn_wheel(long_strip._scroll.viewport(), sideways=True)
    assert _bar(long_strip).value() > 0


def test_the_wheel_works_anywhere_on_the_strip(
    long_strip: Filmstrip, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the position or the edges too, not only over the thumbnails.

    A real wheel over a thumbnail travels up to the strip on its own; Qt
    does not pass on one made up by a test, so this sends it to the strip,
    which is where it lands. The app itself was checked with a real wheel.
    """
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    _turn_wheel(long_strip)
    assert _bar(long_strip).value() > 0


def test_the_wheel_stops_at_either_end(
    long_strip: Filmstrip, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    bar = _bar(long_strip)
    _turn_wheel(long_strip._scroll.viewport(), notches=-3)
    assert bar.value() == 0
    _turn_wheel(long_strip._scroll.viewport(), notches=1000)
    assert bar.value() == bar.maximum()


def test_a_run_of_notches_glides_all_the_way(
    long_strip: Filmstrip, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each notch adds to where the glide is going, not to where it has got.

    The glide must also still be alive at the end: the wheel zoom once only
    ever worked the first time, because its animation was thrown away while
    still in use.
    """
    import time

    from PySide6.QtWidgets import QApplication

    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    bar = _bar(long_strip)
    step = QApplication.wheelScrollLines() * filmstrip_module.WHEEL_LINE
    for round_ in range(2):
        start = bar.value()
        for _ in range(3):
            _turn_wheel(long_strip._scroll.viewport())
            qt_app.processEvents()  # type: ignore[attr-defined]
        assert bar.value() < start + 3 * step, "the notches should glide, not jump"
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            qt_app.processEvents()  # type: ignore[attr-defined]
        assert bar.value() == min(bar.maximum(), start + 3 * step), round_


def test_choosing_a_photo_stops_a_glide(
    long_strip: Filmstrip, qt_app: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the glide would carry the strip away from the photo just chosen."""
    import time

    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: True)
    _turn_wheel(long_strip._scroll.viewport(), notches=5)
    long_strip.set_current(0, len(MANY))
    deadline = time.perf_counter() + 0.5
    while time.perf_counter() < deadline:
        qt_app.processEvents()  # type: ignore[attr-defined]
    assert _bar(long_strip).value() == 0
