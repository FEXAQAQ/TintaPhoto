"""Tests for the filmstrip's multiple choice (P-02).

The filmstrip can hold several photos chosen at once, so that one set of
settings can be pasted onto all of them. The preset list these tests once
shared a file with is now part of the looks tab (tests/test_looks_tab.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from ui.filmstrip import Filmstrip, Thumbnail  # noqa: E402

PHOTOS = [Path(f"D:/Photos/DSC0000{index}.ARW") for index in range(5)]


# -- choosing several photos (P-02) ---------------------------------------


@pytest.fixture
def filmstrip(qt_app: object) -> Filmstrip:
    strip = Filmstrip()
    strip.set_photos(PHOTOS)
    strip.set_current(0, len(PHOTOS))
    return strip


def _thumbnail(strip: Filmstrip, path: Path) -> Thumbnail:
    found = [item for item in strip.findChildren(Thumbnail) if item.path == path]
    assert found, f"{path.name} is not in the strip"
    return found[0]


def _click(strip: Filmstrip, path: Path, modifier: Qt.KeyboardModifier) -> None:
    thumbnail = _thumbnail(strip, path)
    QTest.mouseClick(
        thumbnail, Qt.MouseButton.LeftButton, modifier, thumbnail.rect().center()
    )


def test_one_photo_is_chosen_to_begin_with(filmstrip: Filmstrip) -> None:
    assert filmstrip.selected_photos() == [PHOTOS[0]]


def test_a_plain_click_asks_for_that_photo(filmstrip: Filmstrip) -> None:
    asked: list[Path] = []
    filmstrip.photo_selected.connect(asked.append)

    _click(filmstrip, PHOTOS[2], Qt.KeyboardModifier.NoModifier)

    assert asked == [PHOTOS[2]]


def test_control_adds_a_photo_without_changing_the_current_one(
    filmstrip: Filmstrip,
) -> None:
    asked: list[Path] = []
    filmstrip.photo_selected.connect(asked.append)

    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ControlModifier)

    assert filmstrip.selected_photos() == [PHOTOS[0], PHOTOS[3]]
    assert asked == [], "the photo being edited changed"
    assert _thumbnail(filmstrip, PHOTOS[3]).selected


def test_control_takes_a_photo_back_out(filmstrip: Filmstrip) -> None:
    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ControlModifier)
    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ControlModifier)

    assert filmstrip.selected_photos() == [PHOTOS[0]]


def test_the_current_photo_cannot_be_taken_out(filmstrip: Filmstrip) -> None:
    _click(filmstrip, PHOTOS[0], Qt.KeyboardModifier.ControlModifier)

    assert PHOTOS[0] in filmstrip.selected_photos()


def test_shift_chooses_everything_in_between(filmstrip: Filmstrip) -> None:
    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ShiftModifier)

    assert filmstrip.selected_photos() == PHOTOS[0:4]


def test_shift_works_backwards_too(filmstrip: Filmstrip) -> None:
    filmstrip.set_current(3, len(PHOTOS))

    _click(filmstrip, PHOTOS[1], Qt.KeyboardModifier.ShiftModifier)

    assert filmstrip.selected_photos() == PHOTOS[1:4]


def test_moving_to_another_photo_starts_a_new_choice(filmstrip: Filmstrip) -> None:
    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ControlModifier)

    filmstrip.set_current(1, len(PHOTOS))

    assert filmstrip.selected_photos() == [PHOTOS[1]]


def test_moving_within_the_choice_keeps_it(filmstrip: Filmstrip) -> None:
    _click(filmstrip, PHOTOS[3], Qt.KeyboardModifier.ShiftModifier)

    filmstrip.set_current(2, len(PHOTOS))

    assert filmstrip.selected_photos() == PHOTOS[0:4]


def test_the_choice_is_announced(filmstrip: Filmstrip) -> None:
    heard: list[list[Path]] = []
    filmstrip.selection_changed.connect(heard.append)

    _click(filmstrip, PHOTOS[2], Qt.KeyboardModifier.ControlModifier)

    assert heard and heard[-1] == [PHOTOS[0], PHOTOS[2]]


def test_the_current_photo_and_the_others_look_different(filmstrip: Filmstrip) -> None:
    """The one being edited is marked white, the rest more quietly."""
    import numpy as np

    _click(filmstrip, PHOTOS[1], Qt.KeyboardModifier.ControlModifier)
    for path in (PHOTOS[0], PHOTOS[1]):
        _thumbnail(filmstrip, path).set_picture(
            np.full((72, 96, 3), 128, dtype=np.uint8)
        )

    current = _thumbnail(filmstrip, PHOTOS[0]).grab().toImage()
    chosen = _thumbnail(filmstrip, PHOTOS[1]).grab().toImage()

    assert current.pixelColor(1, 30) != chosen.pixelColor(1, 30)


def test_a_photo_that_is_not_in_the_strip_is_ignored(filmstrip: Filmstrip) -> None:
    filmstrip.set_edited(Path("D:/elsewhere/other.ARW"), True)
    assert filmstrip.selected_photos() == [PHOTOS[0]]
