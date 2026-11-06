"""Tests for the portrait brush over the photo (PLAN.md 3.15)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from ui.skin_overlay import SkinOverlay  # noqa: E402

PHOTO = QRectF(100.0, 50.0, 400.0, 300.0)


def _overlay(qt_app: object) -> SkinOverlay:
    overlay = SkinOverlay()
    overlay.set_open(True)
    overlay.set_brush(True, True, 0.05)
    return overlay


def test_a_drag_paints_a_stroke_in_fractions_of_the_frame(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    painted: list[list] = []
    overlay.stroke_painted.connect(painted.append)

    assert overlay.press(QPointF(200.0, 200.0), PHOTO)
    assert overlay.painting
    overlay.move(QPointF(300.0, 200.0), PHOTO)
    assert overlay.release(QPointF(400.0, 200.0), PHOTO)

    (stroke,) = painted
    assert stroke[0] == pytest.approx((0.25, 0.5))
    assert stroke[-1] == pytest.approx((0.75, 0.5))
    assert not overlay.painting


def test_nothing_is_painted_with_the_brush_down_or_the_portrait_closed(
    qt_app: object,
) -> None:
    overlay = _overlay(qt_app)
    overlay.set_brush(False, True, 0.05)
    assert not overlay.active
    assert not overlay.press(QPointF(200.0, 200.0), PHOTO)
    overlay.set_brush(True, False, 0.05)
    overlay.set_open(False)
    assert not overlay.press(QPointF(200.0, 200.0), PHOTO)
    overlay.set_open(True)
    assert not overlay.press(QPointF(10.0, 10.0), PHOTO)  # off the photo


def test_the_skin_is_shown_as_a_wash_where_it_is(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    skin = np.zeros((30, 40), dtype=np.uint8)
    skin[:, :20] = 255
    overlay.set_skin(skin)
    overlay.set_showing(True)
    overlay.set_brush(False, True, 0.05)
    assert overlay.has_skin and overlay.showing

    canvas = QImage(600, 400, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0xFF000000)
    painter = QPainter(canvas)
    overlay.paint(painter, PHOTO)
    painter.end()
    washed = canvas.pixelColor(150, 200)
    clear = canvas.pixelColor(450, 200)
    assert washed.red() > 60 and clear.red() == 0

    overlay.set_skin(None)
    assert not overlay.has_skin
