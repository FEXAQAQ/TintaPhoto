"""Tests for the watermark's position picker (PLAN.md 3.14)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402

from core import watermark  # noqa: E402
from ui import theme  # noqa: E402
from ui.widgets.anchor_picker import AnchorPicker  # noqa: E402


def _click(picker: AnchorPicker, index: int) -> None:
    centre = picker.cell_rect(index).center()
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
        event = QMouseEvent(
            kind,
            centre,
            picker.mapToGlobal(centre.toPoint()),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        if kind is QEvent.Type.MouseButtonPress:
            picker.mousePressEvent(event)
        else:
            picker.mouseReleaseEvent(event)


def test_nine_squares_on_the_grid(qt_app: object) -> None:
    picker = AnchorPicker()
    side = 3 * theme.ANCHOR_CELL + 2 * theme.ANCHOR_GAP
    assert (picker.width(), picker.height()) == (side, side)
    assert side % 4 == 0
    assert picker.cell_rect(8).bottomRight() == QPointF(side, side)


def test_it_starts_on_the_bottom_right(qt_app: object) -> None:
    assert AnchorPicker().anchor == watermark.ANCHOR_BOTTOM_RIGHT


def test_clicking_a_square_chooses_it(qt_app: object) -> None:
    picker = AnchorPicker()
    chosen: list[str] = []
    picker.chosen.connect(chosen.append)
    _click(picker, 0)
    assert chosen == [watermark.ANCHOR_TOP_LEFT]
    assert picker.anchor == watermark.ANCHOR_TOP_LEFT


def test_none_is_marked_once_the_watermark_is_dragged_off(qt_app: object) -> None:
    picker = AnchorPicker()
    picker.set_anchor(None)
    assert picker.anchor is None


def test_the_arrow_keys_move_the_choice(qt_app: object) -> None:
    picker = AnchorPicker()
    chosen: list[str] = []
    picker.chosen.connect(chosen.append)
    picker.set_anchor(watermark.ANCHOR_CENTRE)
    picker.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
    )
    picker.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
    )
    assert chosen == [watermark.ANCHOR_TOP, watermark.ANCHOR_TOP_LEFT]
    # At the edge it stays where it is.
    picker.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
    )
    assert picker.anchor == watermark.ANCHOR_TOP_LEFT
