"""Tests for the liquify brush over the photo (PLAN.md 3.16)."""

from __future__ import annotations

import pytest

from core import liquify

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from ui.liquify_overlay import TICK_MS, LiquifyOverlay  # noqa: E402

PHOTO = QRectF(100.0, 50.0, 400.0, 300.0)


def _overlay(tool: str = liquify.TOOL_PUSH) -> LiquifyOverlay:
    overlay = LiquifyOverlay()
    overlay.set_open(True)
    overlay.set_brush(tool, 0.05)
    return overlay


def test_a_push_paints_the_path_in_fractions_of_the_frame(qt_app: object) -> None:
    overlay = _overlay()
    drafts: list[list] = []
    painted: list[list] = []
    overlay.stroke_changed.connect(drafts.append)
    overlay.stroke_painted.connect(painted.append)

    assert overlay.press(QPointF(200.0, 200.0), PHOTO)
    assert overlay.painting
    assert not drafts  # one point of a push moves nothing yet
    overlay.move(QPointF(300.0, 200.0), PHOTO)
    assert drafts and drafts[-1][-1] == pytest.approx((0.5, 0.5))
    assert overlay.release(QPointF(400.0, 200.0), PHOTO)

    (stroke,) = painted
    assert stroke[0] == pytest.approx((0.25, 0.5))
    assert stroke[-1] == pytest.approx((0.75, 0.5))
    assert not overlay.painting


def test_a_held_swell_keeps_working_where_the_pointer_rests(qt_app: object) -> None:
    overlay = _overlay(liquify.TOOL_BLOAT)
    drafts: list[list] = []
    overlay.stroke_changed.connect(drafts.append)
    assert overlay.press(QPointF(300.0, 200.0), PHOTO)
    assert drafts == [[pytest.approx((0.5, 0.5))]]
    assert TICK_MS == round(1000 / liquify.TICKS_PER_SECOND)
    for _ in range(3):
        overlay._tick()
    assert len(drafts[-1]) == 4
    assert all(point == pytest.approx((0.5, 0.5)) for point in drafts[-1])

    painted: list[list] = []
    overlay.stroke_painted.connect(painted.append)
    overlay.release(QPointF(300.0, 200.0), PHOTO)
    assert len(painted[0]) == 4
    overlay._tick()  # the timer has stopped with the stroke
    assert len(drafts[-1]) == 4


def test_nothing_is_painted_while_closed_or_off_the_photo(qt_app: object) -> None:
    overlay = _overlay()
    overlay.set_open(False)
    assert not overlay.active
    assert not overlay.press(QPointF(200.0, 200.0), PHOTO)
    overlay.set_open(True)
    assert not overlay.press(QPointF(10.0, 10.0), PHOTO)


def test_closing_in_the_middle_of_a_stroke_drops_it(qt_app: object) -> None:
    overlay = _overlay(liquify.TOOL_TWIRL_CW)
    painted: list[list] = []
    overlay.stroke_painted.connect(painted.append)
    overlay.press(QPointF(200.0, 200.0), PHOTO)
    overlay.set_open(False)
    assert not overlay.painting
    assert not overlay.release(QPointF(200.0, 200.0), PHOTO)
    assert not painted


def test_the_brush_is_drawn_under_the_pointer(qt_app: object) -> None:
    overlay = _overlay()
    image = QImage(600, 400, QImage.Format.Format_ARGB32)
    image.fill(0)
    overlay.set_pointer(QPointF(300.0, 200.0))
    painter = QPainter(image)
    overlay.paint(painter, PHOTO)
    painter.end()
    # The circle is twenty pixels out (a twentieth of the 400 px long edge).
    assert image.pixelColor(320, 200).alpha() > 0
    assert image.pixelColor(310, 200).alpha() == 0
    assert image.pixelColor(301, 200).alpha() > 0  # the dot in the middle
