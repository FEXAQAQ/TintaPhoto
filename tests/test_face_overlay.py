"""Tests for the face's points over the photo (PLAN.md 3.16, L-12)."""

from __future__ import annotations

import pytest

from core import liquify

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from ui.face_overlay import FaceOverlay  # noqa: E402

PHOTO = QRectF(100.0, 50.0, 400.0, 300.0)
FACE = liquify.FaceMarks(
    points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
)


def _overlay() -> FaceOverlay:
    overlay = FaceOverlay()
    overlay.set_open(True)
    guide = liquify.face_guide(FACE, (400, 300))
    assert guide is not None
    overlay.set_guides([guide])
    return overlay


def _on_screen(point: tuple[float, float]) -> QPointF:
    return QPointF(PHOTO.left() + point[0] * PHOTO.width(), PHOTO.top() + point[1] * PHOTO.height())


def test_a_point_is_dragged_and_handed_over_when_let_go(qt_app: object) -> None:
    overlay = _overlay()
    moved: list[tuple] = []
    overlay.handle_moved.connect(lambda *args: moved.append(args))
    nose = _on_screen(FACE.points[2])

    assert overlay.over_handle(nose, PHOTO)
    assert overlay.press(nose, PHOTO)
    assert overlay.dragging
    overlay.move(nose + QPointF(10.0, 0.0), PHOTO)
    assert moved[-1][:2] == (0, "nose") and moved[-1][4] is True
    assert overlay.release(nose + QPointF(20.0, 0.0), PHOTO)
    face, handle, x, y, dragging = moved[-1]
    assert (face, handle, dragging) == (0, "nose", False)
    assert (x, y) == pytest.approx((0.55, 0.47))
    assert not overlay.dragging


def test_a_press_away_from_every_point_is_left_for_panning(qt_app: object) -> None:
    overlay = _overlay()
    assert not overlay.press(QPointF(120.0, 60.0), PHOTO)
    assert not overlay.dragging
    overlay.set_open(False)
    assert not overlay.press(_on_screen(FACE.points[2]), PHOTO)


def test_the_nearest_point_is_the_one_taken(qt_app: object) -> None:
    overlay = _overlay()
    moved: list[tuple] = []
    overlay.handle_moved.connect(lambda *args: moved.append(args))
    eye = _on_screen(FACE.points[1])
    overlay.press(eye + QPointF(2.0, 1.0), PHOTO)
    overlay.release(eye, PHOTO)
    assert moved[-1][1] == "right_eye"


def test_the_points_and_the_reach_of_the_one_in_hand_are_drawn(qt_app: object) -> None:
    overlay = _overlay()
    guide = liquify.face_guide(FACE, (400, 300))
    assert guide is not None
    image = QImage(600, 400, QImage.Format.Format_ARGB32)
    image.fill(0)
    chin = _on_screen(guide.handles[liquify.HANDLES.index("chin")])
    overlay.move(chin, PHOTO)  # under the pointer: filled, its reach outlined
    painter = QPainter(image)
    overlay.paint(painter, PHOTO)
    painter.end()
    assert image.pixelColor(chin.toPoint()).alpha() > 0
    (centre, across, _down) = guide.reaches["chin"][0]
    edge = _on_screen(across)
    assert image.pixelColor(edge.toPoint()).alpha() > 0
