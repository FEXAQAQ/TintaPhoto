"""Tests for the spot removal brush drawn over the photo (PLAN.md 3.6, A-32)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402

from ui import theme  # noqa: E402
from ui.spot_overlay import SpotOverlay  # noqa: E402

#: The photo is drawn 400 by 200 pixels, from (100, 50).
PHOTO = QRectF(100.0, 50.0, 400.0, 200.0)
#: One dot in the middle, its source to the right, 20 pixels of radius on
#: screen (a twentieth of the 400 pixel long edge).
DOT = ([(0.5, 0.5)], [(0.75, 0.5)], 0.05)
#: A stroke along the top, its source below it.
LINE = ([(0.2, 0.2), (0.4, 0.2)], [(0.2, 0.5), (0.4, 0.5)], 0.02)


def _overlay(qt_app: object, spots: list | None = None) -> SpotOverlay:
    overlay = SpotOverlay()
    overlay.set_active(True)
    overlay.set_spots([DOT] if spots is None else spots)
    overlay.set_brush(0.02)
    return overlay


def _collect(signal) -> list:
    seen: list = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def _drag(overlay: SpotOverlay, points: list[tuple[float, float]]) -> None:
    overlay.press(QPointF(*points[0]), PHOTO)
    for point in points[1:]:
        overlay.move(QPointF(*point), PHOTO)
    overlay.release(QPointF(*points[-1]), PHOTO)


def test_dragging_paints_and_letting_go_hands_over_the_path(qt_app: object) -> None:
    overlay = _overlay(qt_app, [])
    painted = _collect(overlay.spot_painted)
    overlay.press(QPointF(150.0, 100.0), PHOTO)
    assert overlay.painting
    for x in range(151, 251):  # a hundred moves, a pixel apart
        overlay.move(QPointF(float(x), 100.0), PHOTO)
    overlay.release(QPointF(250.0, 100.0), PHOTO)

    assert not overlay.painting
    (path,), = painted
    assert path[0] == pytest.approx((0.125, 0.25))
    assert path[-1] == pytest.approx((0.375, 0.25))
    # Thinned as it is painted: a point every few pixels, not every one.
    assert 10 < len(path) < 60


def test_a_click_where_nothing_is_mended_is_a_dot(qt_app: object) -> None:
    overlay = _overlay(qt_app, [])
    painted = _collect(overlay.spot_painted)
    _drag(overlay, [(150.0, 100.0)])
    assert painted == [([pytest.approx((0.125, 0.25))],)]


def test_a_click_on_a_mended_stroke_selects_it_rather_than_painting(qt_app: object) -> None:
    overlay = _overlay(qt_app, [DOT, LINE])
    painted = _collect(overlay.spot_painted)
    _drag(overlay, [(220.0, 90.0)])  # on the line
    assert overlay.selected == 1
    assert painted == []


def test_painting_may_start_over_an_earlier_stroke(qt_app: object) -> None:
    overlay = _overlay(qt_app, [DOT])
    painted = _collect(overlay.spot_painted)
    _drag(overlay, [(300.0, 150.0), (320.0, 150.0), (340.0, 150.0)])
    assert len(painted) == 1
    assert overlay.selected == -1


def test_a_press_beside_the_photo_is_not_taken(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    assert not overlay.press(QPointF(50.0, 100.0), PHOTO)


def test_nothing_is_taken_while_the_tool_is_down(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    overlay.set_active(False)
    assert not overlay.press(QPointF(300.0, 150.0), PHOTO)


def test_the_selected_stroke_is_dragged_by_where_it_was_held(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    overlay.select(0)
    moved = _collect(overlay.spot_moved)
    _drag(overlay, [(305.0, 150.0), (345.0, 170.0)])
    assert moved[0] == (0, pytest.approx(0.6), pytest.approx(0.6), False, True)
    assert moved[-1] == (0, pytest.approx(0.6), pytest.approx(0.6), False, False)


def test_the_source_of_the_selected_stroke_moves_on_its_own(qt_app: object) -> None:
    overlay = _overlay(qt_app, [DOT, LINE])
    overlay.select(1)
    moved = _collect(overlay.spot_moved)
    # The source line runs along y = 150, from x = 180 to 260.
    _drag(overlay, [(220.0, 150.0), (220.0, 190.0)])
    assert moved[-1] == (1, pytest.approx(0.2), pytest.approx(0.7), True, False)


def test_a_tiny_stroke_can_still_be_taken_hold_of(qt_app: object) -> None:
    overlay = _overlay(qt_app, [([(0.5, 0.5)], [(0.7, 0.5)], 0.001)])
    overlay.select(0)
    reach = theme.SPOT_GRAB_RADIUS - 1
    assert overlay.over_spot(QPointF(300.0 + reach, 150.0), PHOTO)
    assert not overlay.over_spot(QPointF(300.0 + theme.SPOT_GRAB_RADIUS + 3, 150.0), PHOTO)


def test_only_the_selected_stroke_can_be_taken_hold_of(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    assert not overlay.over_spot(QPointF(300.0, 150.0), PHOTO)
    overlay.select(0)
    assert overlay.over_spot(QPointF(300.0, 150.0), PHOTO)


def test_a_selection_past_the_end_is_dropped(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    overlay.select(0)
    overlay.set_spots([])
    assert overlay.selected == -1


def test_putting_the_tool_down_forgets_the_selection_and_the_paint(qt_app: object) -> None:
    overlay = _overlay(qt_app)
    overlay.select(0)
    overlay.press(QPointF(150.0, 100.0), PHOTO)
    overlay.set_active(False)
    assert overlay.selected == -1
    assert not overlay.painting


def _drawn(overlay: SpotOverlay) -> QImage:
    image = QImage(600, 300, QImage.Format.Format_RGB32)
    image.fill(QColor("#808080"))
    painter = QPainter(image)
    overlay.paint(painter, PHOTO)
    painter.end()
    return image


def test_a_mended_stroke_keeps_a_faint_outline_and_a_selected_one_a_solid_one(
    qt_app: object,
) -> None:
    overlay = _overlay(qt_app)
    faint = max(QColor(_drawn(overlay).pixel(x, 150)).lightness() for x in range(270, 331))
    overlay.select(0)
    image = _drawn(overlay)
    solid = max(QColor(image.pixel(x, 150)).lightness() for x in range(270, 331))
    assert 128 < faint < solid
    # Inside the outline the photo is left alone: the tool covers nothing.
    assert QColor(image.pixel(300, 150)).name().upper() == "#808080"
    # The source is drawn once the stroke is selected: its outline crosses
    # the column through its middle 20 px above.
    column = [QColor(image.pixel(400, y)).lightness() for y in range(124, 137)]
    assert max(column) > 160


def test_what_is_being_painted_is_washed_over(qt_app: object) -> None:
    overlay = _overlay(qt_app, [])
    overlay.press(QPointF(150.0, 100.0), PHOTO)
    overlay.move(QPointF(250.0, 100.0), PHOTO)
    lightness = QColor(_drawn(overlay).pixel(200, 100)).lightness()
    assert 128 < lightness < 255
