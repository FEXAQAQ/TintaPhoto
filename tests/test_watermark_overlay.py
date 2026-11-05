"""Tests for the watermark placed over the photo (PLAN.md 3.14)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402

from core import watermark  # noqa: E402
from core.watermark import WatermarkSpec  # noqa: E402
from ui.watermark_overlay import WatermarkOverlay  # noqa: E402

#: The photo is 6000 x 4000 and drawn at a tenth of that, from (50, 20).
FRAME = (6000, 4000)
PHOTO = QRectF(50, 20, 600, 400)
#: A 1000 x 200 px watermark on the full size frame, drawn at half size.
PICTURE = np.zeros((100, 500, 4), dtype=np.uint8)


@pytest.fixture
def overlay(qt_app: object) -> WatermarkOverlay:
    placed = WatermarkOverlay()
    placed.set_frame(FRAME)
    placed.set_active(True)
    placed.set_picture(WatermarkSpec(text="x", margin=0), PICTURE, 0.5)
    return placed


def test_it_is_drawn_where_the_export_puts_it(overlay: WatermarkOverlay) -> None:
    # Bottom right, no margin: the box's corner meets the photo's.
    rect = overlay.screen_rect(PHOTO)
    assert rect == QRectF(550, 400, 100, 20)


def test_nothing_is_drawn_while_it_is_not_being_placed(overlay: WatermarkOverlay) -> None:
    overlay.set_active(False)
    assert overlay.screen_rect(PHOTO).isEmpty()
    assert not overlay.press(QPointF(600, 410), PHOTO)


def test_a_press_off_the_watermark_is_left_for_panning(overlay: WatermarkOverlay) -> None:
    assert not overlay.press(QPointF(100, 100), PHOTO)
    assert not overlay.dragging


def test_dragging_moves_it_and_says_so(overlay: WatermarkOverlay) -> None:
    moves: list[tuple[WatermarkSpec, bool]] = []
    overlay.moved.connect(lambda spec, settled: moves.append((spec, settled)))
    assert overlay.press(QPointF(600, 410), PHOTO)
    overlay.move(QPointF(400, 210), PHOTO)
    overlay.release(QPointF(400, 210), PHOTO)
    # 200 screen pixels left and up is 2000 frame pixels.
    x, y = watermark.layout(overlay.spec, FRAME, (1000, 200))
    assert (x, y) == pytest.approx((3000.0, 1800.0))
    assert [settled for _spec, settled in moves] == [False, True]
    assert not overlay.dragging


def test_a_drag_near_the_middle_snaps_onto_it(overlay: WatermarkOverlay) -> None:
    overlay.press(QPointF(600, 410), PHOTO)
    # The middle column puts the box's left at 2500 frame px, 300 on screen;
    # this lands 4 screen px off it, within the 8 px pull.
    overlay.move(QPointF(354, 410), PHOTO)
    assert overlay.spec.anchor == watermark.ANCHOR_BOTTOM
    assert overlay.spec.offset[0] == 0.0


def test_alt_lets_it_sit_off_the_lines(overlay: WatermarkOverlay) -> None:
    overlay.press(QPointF(600, 410), PHOTO)
    overlay.move(QPointF(354, 410), PHOTO, free=True)
    assert overlay.spec.offset[0] != 0.0


def test_a_double_click_puts_it_back_bottom_right(overlay: WatermarkOverlay) -> None:
    overlay.set_spec(WatermarkSpec(text="x", anchor="top-left", offset=(0.1, 0.1)))
    rect = overlay.screen_rect(PHOTO)
    assert overlay.double_click(rect.center(), PHOTO)
    assert overlay.spec.anchor == watermark.ANCHOR_BOTTOM_RIGHT
    assert overlay.spec.offset == (0.0, 0.0)


def test_a_new_spec_does_not_snatch_it_from_the_hand(overlay: WatermarkOverlay) -> None:
    overlay.press(QPointF(600, 410), PHOTO)
    overlay.set_spec(WatermarkSpec(text="x", anchor="top-left"))
    assert overlay.spec.anchor == watermark.ANCHOR_BOTTOM_RIGHT
