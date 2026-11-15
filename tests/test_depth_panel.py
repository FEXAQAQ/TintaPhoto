"""Tests for the depth of field's panel and overlay (PLAN.md 2.13, 3.18)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest

from core import depth_of_field as dof
from core.depth_of_field import DEFAULT_DEPTH
from core.skin_range import SkinStroke

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from app.i18n import tr  # noqa: E402
from ui import theme  # noqa: E402
from ui.depth_overlay import DepthOverlay  # noqa: E402
from ui.depth_panel import BRUSH_BLUR, BRUSH_SHARP, DepthPanel  # noqa: E402

PHOTO = QRectF(100.0, 50.0, 400.0, 300.0)
STROKE = SkinStroke(points=((0.5, 0.5),), radius=0.03, snap=False)


@pytest.fixture
def panel(qt_app: object) -> Iterator[DepthPanel]:
    widget = DepthPanel()
    widget.resize(theme.PANEL_WIDTH, 700)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


# -- the panel ---------------------------------------------------------------


def test_it_is_as_wide_as_the_inspector_and_has_its_two_groups(panel: DepthPanel) -> None:
    assert panel.width() == theme.PANEL_WIDTH
    assert [group.header.text() for group in panel.groups()] == [
        tr("depth.group.effect"),
        tr("depth.group.range"),
    ]


def test_it_shows_a_depth_of_field_without_announcing_it(panel: DepthPanel) -> None:
    heard: list[object] = []
    panel.value_set.connect(lambda *args: heard.append(args))
    panel.shape_chosen.connect(heard.append)
    depth = replace(
        DEFAULT_DEPTH,
        amount=60.0,
        transition=20.0,
        highlights=80.0,
        shape=dof.SHAPE_BAND,
        bokeh=dof.BOKEH_HEXAGON,
    )
    panel.set_depth(depth)

    assert heard == []
    assert panel.rows()["amount"].value == 60.0
    assert panel.rows()["transition"].value == 20.0
    assert panel.rows()["highlights"].value == 80.0
    assert panel.shape_choice.value == dof.SHAPE_BAND
    assert panel.bokeh_choice.value == dof.BOKEH_HEXAGON


def test_the_caption_says_what_to_do_next(panel: DepthPanel) -> None:
    assert panel.caption.text() == tr("depth.hint.amount")
    panel.set_depth(replace(DEFAULT_DEPTH, amount=50.0))
    assert panel.caption.text() == tr("depth.hint")
    assert not panel.reset_button.isEnabled()
    panel.set_depth(replace(DEFAULT_DEPTH, amount=50.0, strokes=(STROKE, STROKE)))
    assert panel.caption.text() == tr("depth.strokes.many", count=2)
    assert panel.reset_button.isEnabled()
    panel.set_depth(replace(DEFAULT_DEPTH, centre=(0.3, 0.3)))
    assert panel.reset_button.isEnabled()


def test_the_switch_comes_first_and_says_when_it_is_off(panel: DepthPanel) -> None:
    heard: list[bool] = []
    panel.enabled_set.connect(heard.append)
    assert panel.enabled_switch.isChecked()

    panel.set_depth(replace(DEFAULT_DEPTH, amount=50.0, enabled=False))
    assert heard == [] and not panel.enabled_switch.isChecked()
    assert panel.caption.text() == tr("depth.hint.off")

    panel.enabled_switch.setChecked(True)
    assert heard == [True]


def test_the_choices_and_sliders_are_announced(panel: DepthPanel) -> None:
    values: list[tuple[str, float, bool]] = []
    shapes: list[str] = []
    bokehs: list[str] = []
    panel.value_set.connect(lambda name, value, dragging: values.append((name, value, dragging)))
    panel.shape_chosen.connect(shapes.append)
    panel.bokeh_chosen.connect(bokehs.append)

    panel.rows()["amount"].set_value(40.0, announce=True)
    panel.shape_choice.buttons()[dof.SHAPE_BAND].click()
    panel.bokeh_choice.buttons()[dof.BOKEH_HEXAGON].click()

    assert values and values[-1][:2] == ("amount", 40.0)
    assert shapes == [dof.SHAPE_BAND]
    assert bokehs == [dof.BOKEH_HEXAGON]


def test_one_brush_job_at_a_time(panel: DepthPanel) -> None:
    assert panel.brush is None
    panel.sharp_button.click()
    assert panel.brush == BRUSH_SHARP
    panel.blur_button.click()
    assert panel.brush == BRUSH_BLUR and not panel.sharp_button.isChecked()
    panel.blur_button.click()
    assert panel.brush is None


def test_every_button_names_its_key(panel: DepthPanel) -> None:
    assert tr("shortcut.portrait_add") in panel.sharp_button.toolTip()
    assert tr("shortcut.portrait_remove") in panel.blur_button.toolTip()
    assert tr("shortcut.depth") in panel.leave_button.toolTip()
    assert tr("shortcut.portrait_show") in panel.show_switch.toolTip()


# -- the overlay -------------------------------------------------------------


def _overlay(depth: dof.DepthOfField = DEFAULT_DEPTH) -> DepthOverlay:
    overlay = DepthOverlay()
    overlay.set_open(True)
    overlay.set_depth(depth)
    return overlay


def test_dragging_the_middle_moves_the_whole_shape(qt_app: object) -> None:
    overlay = _overlay()
    moves: list[tuple] = []
    overlay.focus_changed.connect(lambda *args: moves.append(args))
    middle = QPointF(300.0, 200.0)

    assert overlay.press(middle, PHOTO)
    overlay.move(QPointF(340.0, 230.0), PHOTO)
    assert moves[-1][3] is True
    assert overlay.release(QPointF(380.0, 260.0), PHOTO)

    centre, handle, ratio, dragging = moves[-1]
    assert dragging is False
    assert centre == pytest.approx((0.7, 0.7))
    assert handle == pytest.approx((0.7, 0.45))
    assert ratio == dof.DEFAULT_RATIO


def test_the_handles_turn_size_and_narrow_an_ellipse(qt_app: object) -> None:
    overlay = _overlay()
    moves: list[tuple] = []
    overlay.focus_changed.connect(lambda *args: moves.append(args))
    # The long axis ends a quarter of the height up from the middle.
    axis_end = QPointF(300.0, 125.0)

    assert overlay.press(axis_end, PHOTO)
    overlay.release(QPointF(400.0, 200.0), PHOTO)
    assert moves[-1][1] == pytest.approx((0.75, 0.5))

    # The short axis is now straight down, at 0.75 of the 100 pixel axis.
    overlay.set_depth(replace(DEFAULT_DEPTH, handle=(0.75, 0.5)))
    assert overlay.press(QPointF(300.0, 275.0), PHOTO)
    overlay.release(QPointF(300.0, 250.0), PHOTO)
    assert moves[-1][2] == pytest.approx(0.5)


def test_a_press_away_from_the_shape_is_left_for_panning(qt_app: object) -> None:
    overlay = _overlay()
    assert not overlay.press(QPointF(110.0, 60.0), PHOTO)
    assert not overlay.over(QPointF(110.0, 60.0), PHOTO)
    assert overlay.over(QPointF(300.0, 200.0), PHOTO)
    band = _overlay(replace(DEFAULT_DEPTH, shape=dof.SHAPE_BAND, handle=(0.5, 0.4)))
    assert band.over(QPointF(110.0, 200.0), PHOTO)
    assert not band.over(QPointF(300.0, 60.0), PHOTO)


def test_with_the_brush_taken_up_a_drag_paints(qt_app: object) -> None:
    overlay = _overlay()
    overlay.set_brush(True, True, 0.05)
    painted: list[list] = []
    overlay.stroke_painted.connect(painted.append)
    moves: list[tuple] = []
    overlay.focus_changed.connect(lambda *args: moves.append(args))

    assert overlay.press(QPointF(300.0, 200.0), PHOTO)
    overlay.move(QPointF(340.0, 200.0), PHOTO)
    overlay.release(QPointF(380.0, 200.0), PHOTO)

    assert moves == []
    assert painted and painted[0][0] == pytest.approx((0.5, 0.5))
    assert painted[0][-1] == pytest.approx((0.7, 0.5))


def test_it_draws_only_while_open(qt_app: object) -> None:
    picture = QImage(600, 400, QImage.Format.Format_ARGB32_Premultiplied)
    picture.fill(0)
    overlay = _overlay()
    overlay.set_showing(True)
    overlay.set_range(np.full((30, 40), 255, dtype=np.uint8))
    painter = QPainter(picture)
    overlay.paint(painter, PHOTO)
    painter.end()
    assert QImage(picture).pixelColor(110, 60).alpha() > 0

    closed = QImage(600, 400, QImage.Format.Format_ARGB32_Premultiplied)
    closed.fill(0)
    overlay.set_open(False)
    painter = QPainter(closed)
    overlay.paint(painter, PHOTO)
    painter.end()
    assert closed.pixelColor(110, 60).alpha() == 0
