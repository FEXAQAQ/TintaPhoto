"""Tests for zooming and panning the preview (PLAN.md section 3.3).

These drive the real wheel events through the animated path, because that is
where the zoom broke: the animation was handed to Qt to delete when it
stopped while a reference to it was kept here, so the turn after the first
one raised instead of zooming.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui import theme  # noqa: E402
from ui.preview_view import (  # noqa: E402
    MINIMUM_ZOOM_CHANGE,
    ZOOM_ACTUAL,
    ZOOM_STEPS,
    PhotoCanvas,
    ViewBar,
    PreviewArea,
)

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
PHOTO = (1710, 2560)
#: A window the photo fits in at well under the smallest zoom step.
SMALL_WIDTH = 400
SMALL_HEIGHT = 300


def _settle(canvas: PhotoCanvas | None = None, timeout_ms: int = 4000) -> None:
    """Let the zoom animation run to its end.

    Waited out rather than slept through: under a full test run the
    animation can take longer than its own duration to get going, and a
    fixed sleep then reads the scale before it has moved at all.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(theme.DURATION_HOVER_MS, loop.quit)
        loop.exec()
        if canvas is None or not canvas.zooming:
            return


@pytest.fixture
def canvas(qt_app: object) -> Iterator[PhotoCanvas]:
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    widget.set_photo(np.full((*PHOTO, 3), 128, dtype=np.uint8))
    qt_app.processEvents()  # type: ignore[attr-defined]
    yield widget
    widget.hide()
    widget.deleteLater()


def turn_wheel(canvas: PhotoCanvas, up: bool, at: QPoint | None = None) -> None:
    """Send one notch of the wheel, as the mouse driver does."""
    position = QPointF(at or canvas.rect().center())
    event = QWheelEvent(
        position,
        QPointF(canvas.mapToGlobal(position.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 if up else -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas, event)
    _settle(canvas)


# -- the wheel ------------------------------------------------------------


def test_the_wheel_keeps_working_after_the_first_turn(canvas: PhotoCanvas) -> None:
    """Every turn must zoom, not just the first one."""
    seen = [canvas.scale]
    for _ in range(4):
        turn_wheel(canvas, up=True)
        seen.append(canvas.scale)
    assert seen == sorted(seen), seen
    assert len(set(seen)) == len(seen), f"a turn did nothing: {seen}"


def test_the_wheel_comes_back_down_to_fitting_the_window(canvas: PhotoCanvas) -> None:
    """Zooming in and out again must return to the fitted view."""
    for _ in range(3):
        turn_wheel(canvas, up=True)
    assert not canvas.fitting
    zoomed = canvas.scale

    for _ in range(6):
        turn_wheel(canvas, up=False)
    assert canvas.fitting, f"stuck at {canvas.scale} after zooming to {zoomed}"
    assert canvas.scale == pytest.approx(canvas.fit_scale(), abs=1e-6)


def test_the_wheel_fits_a_photo_far_larger_than_the_window(
    canvas: PhotoCanvas, qt_app: object
) -> None:
    """A photo that fits below the smallest step must still come all the way.

    The steps stop at a quarter, and the wheel used to stop there with them,
    which left the window holding a crop of a large photo with no way to see
    the whole of it.
    """
    canvas.resize(SMALL_WIDTH, SMALL_HEIGHT)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert canvas.fit_scale() < ZOOM_STEPS[0], "the window is not small enough to tell"

    canvas.zoom_to(ZOOM_ACTUAL, animate=False)
    for _ in range(8):
        turn_wheel(canvas, up=False)
    assert canvas.fitting, f"stuck at {canvas.scale}, fits at {canvas.fit_scale()}"
    assert canvas.scale == pytest.approx(canvas.fit_scale(), abs=1e-6)


def test_alternating_turns_end_where_they_started(canvas: PhotoCanvas) -> None:
    start = canvas.scale
    for _ in range(3):
        turn_wheel(canvas, up=True)
        turn_wheel(canvas, up=False)
    assert canvas.scale == pytest.approx(start, abs=1e-6)
    assert canvas.fitting


def test_every_turn_changes_the_zoom_enough_to_see(canvas: PhotoCanvas) -> None:
    """A step a hair above the fitted scale would feel like a dead notch."""
    previous = canvas.scale
    for _ in range(4):
        turn_wheel(canvas, up=True)
        assert canvas.scale >= previous * (1.0 + MINIMUM_ZOOM_CHANGE), (
            previous,
            canvas.scale,
        )
        previous = canvas.scale


def test_the_wheel_stops_at_the_largest_step(canvas: PhotoCanvas) -> None:
    for _ in range(12):
        turn_wheel(canvas, up=True)
    assert canvas.scale == pytest.approx(ZOOM_STEPS[-1])


def test_the_wheel_does_nothing_without_a_photo(qt_app: object) -> None:
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    try:
        turn_wheel(widget, up=True)
        assert not widget.has_photo
    finally:
        widget.hide()
        widget.deleteLater()


# -- zooming around the pointer -------------------------------------------


def test_zooming_keeps_the_point_under_the_pointer_in_place(canvas: PhotoCanvas) -> None:
    """PLAN.md 3.3: the point the pointer is on does not move."""
    pointer = QPoint(CANVAS_WIDTH // 4, CANVAS_HEIGHT // 3)
    before = canvas.widget_to_photo(pointer)
    turn_wheel(canvas, up=True, at=pointer)
    after = canvas.widget_to_photo(pointer)
    assert after.x() == pytest.approx(before.x(), abs=1.0)
    assert after.y() == pytest.approx(before.y(), abs=1.0)


def test_zooming_in_the_centre_keeps_the_centre(canvas: PhotoCanvas) -> None:
    centre = canvas.rect().center()
    before = canvas.widget_to_photo(centre)
    turn_wheel(canvas, up=True, at=centre)
    after = canvas.widget_to_photo(centre)
    assert after.x() == pytest.approx(before.x(), abs=1.0)
    assert after.y() == pytest.approx(before.y(), abs=1.0)


# -- fitting and actual size ----------------------------------------------


def test_a_new_photo_starts_fitted(canvas: PhotoCanvas) -> None:
    assert canvas.fitting
    assert canvas.scale == pytest.approx(canvas.fit_scale())
    assert canvas.scale < 1.0  # the sample is larger than the window


def test_the_fitted_photo_leaves_the_documented_margin(canvas: PhotoCanvas) -> None:
    drawn_width = PHOTO[1] * canvas.fit_scale()
    drawn_height = PHOTO[0] * canvas.fit_scale()
    assert drawn_width <= CANVAS_WIDTH - 2 * theme.CANVAS_MARGIN + 1
    assert drawn_height <= CANVAS_HEIGHT - 2 * theme.CANVAS_MARGIN + 1


def test_toggling_actual_size_and_back(canvas: PhotoCanvas) -> None:
    canvas.toggle_actual_size()
    _settle(canvas)
    assert canvas.scale == pytest.approx(ZOOM_ACTUAL)
    assert not canvas.fitting

    canvas.toggle_actual_size()
    _settle(canvas)
    assert canvas.fitting
    assert canvas.scale == pytest.approx(canvas.fit_scale(), abs=1e-6)


def test_resizing_the_window_keeps_the_photo_fitted(canvas: PhotoCanvas, qt_app: object) -> None:
    canvas.resize(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert canvas.fitting
    assert canvas.scale == pytest.approx(canvas.fit_scale(), abs=1e-6)


def test_resizing_does_not_disturb_a_chosen_zoom(canvas: PhotoCanvas, qt_app: object) -> None:
    canvas.zoom_to(ZOOM_ACTUAL, animate=False)
    canvas.resize(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)
    qt_app.processEvents()  # type: ignore[attr-defined]
    assert canvas.scale == pytest.approx(ZOOM_ACTUAL)
    assert not canvas.fitting


# -- panning --------------------------------------------------------------


def test_panning_cannot_push_the_photo_off_the_screen(canvas: PhotoCanvas) -> None:
    """PLAN.md 3.3: at least a corner of the photo stays visible."""
    canvas.zoom_to(ZOOM_STEPS[-1], animate=False)
    for _ in range(40):
        canvas._centre += QPointF(500, 500)  # as a very long drag would
        canvas._clamp_centre()
    visible = canvas.widget_to_photo(canvas.rect().bottomRight())
    assert visible.x() > 0
    assert visible.y() > 0


def test_a_new_photo_of_the_same_size_keeps_the_zoom(canvas: PhotoCanvas) -> None:
    """Developing the same photo again must not jump back to fitting."""
    canvas.zoom_to(ZOOM_ACTUAL, animate=False)
    canvas.set_photo(np.full((*PHOTO, 3), 200, dtype=np.uint8))
    assert canvas.scale == pytest.approx(ZOOM_ACTUAL)


def test_a_photo_of_a_different_size_is_fitted(canvas: PhotoCanvas) -> None:
    canvas.zoom_to(ZOOM_ACTUAL, animate=False)
    canvas.set_photo(np.full((800, 600, 3), 200, dtype=np.uint8))
    assert canvas.fitting


def test_the_photo_is_separated_from_the_background_by_a_hairline(
    qt_app: object,
) -> None:
    """A photo with dark corners would otherwise run into the background.

    PLAN.md 3.5 rules out a drop shadow, so the edge is one line of the same
    grey as every other separator in the window.
    """
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    # A photo small enough to sit at whole pixels at actual size, so the
    # line lands on one column rather than across two.
    widget.set_photo(np.full((100, 100, 3), 128, dtype=np.uint8))
    widget.zoom_to(ZOOM_ACTUAL, animate=False)
    qt_app.processEvents()  # type: ignore[attr-defined]
    try:
        picture = widget.grab().toImage()
        photo = widget.photo_rect()
        middle = int(photo.center().y())
        assert picture.pixelColor(int(photo.left()), middle) == QColor(theme.BORDER)
        # Inside the line the photo itself is untouched.
        inside = picture.pixelColor(int(photo.left()) + theme.SPACE_XS, middle)
        assert inside != QColor(theme.BORDER)
    finally:
        widget.hide()
        widget.deleteLater()


def test_the_floating_bar_is_exactly_twice_its_corner_radius_tall(
    qt_app: object,
) -> None:
    """A pill, not a rounded rectangle: the ends have to be semicircles.

    Qt adds a widget's border to the height a stylesheet asks for, so this
    is the check that the two have been kept in step.
    """
    bar = ViewBar()
    bar.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    try:
        assert bar.height() == theme.VIEW_BAR_HEIGHT
        assert bar.height() == 2 * theme.RADIUS_VIEW_BAR
    finally:
        bar.hide()
        bar.deleteLater()


def test_a_photo_moved_to_comes_up_over_the_documented_time(
    qt_app: object,
) -> None:
    """PLAN.md 3.6: 切換照片淡入 120 ms."""
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    widget.set_photo(np.full((200, 300, 3), 200, dtype=np.uint8))
    try:
        assert not widget.fading
        widget.set_photo(np.full((200, 300, 3), 60, dtype=np.uint8), fade=True)
        assert widget.fading
        # Part way through, the photo is not all there yet.
        assert widget._fade < 1.0

        deadline = time.monotonic() + 4.0
        while widget.fading and time.monotonic() < deadline:
            _settle()
        assert not widget.fading
        assert widget._fade == 1.0
    finally:
        widget.hide()
        widget.deleteLater()


def test_adjusting_a_photo_does_not_fade_it(qt_app: object) -> None:
    """The previews that follow a slider replace each other at once."""
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    widget.set_photo(np.full((200, 300, 3), 200, dtype=np.uint8))
    try:
        widget.set_photo(np.full((200, 300, 3), 90, dtype=np.uint8))
        assert not widget.fading
        assert widget._fade == 1.0
    finally:
        widget.hide()
        widget.deleteLater()


def _send_mouse(
    canvas: PhotoCanvas,
    kind: "QMouseEvent.Type",
    position: QPointF,
    held: bool = False,
) -> None:
    """Send one mouse event, the way the driver of a dragging hand does."""
    button = Qt.MouseButton.LeftButton
    event = QMouseEvent(
        kind,
        position,
        canvas.mapToGlobal(position),
        Qt.MouseButton.NoButton if held else button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)

# -- the zoom is measured against the photo (PLAN.md 3.3) ------------------

#: A preview developed at the size it is shown, for a photo of PHOTO_FULL.
PREVIEW = (600, 900)
PHOTO_FULL = (4000, 6000)


def _zoomed_canvas(qt_app: object) -> PhotoCanvas:
    """Return a canvas showing a small preview of a large photo."""
    widget = PhotoCanvas()
    widget.resize(CANVAS_WIDTH, CANVAS_HEIGHT)
    widget.show()
    qt_app.processEvents()  # type: ignore[attr-defined]
    widget.set_photo(
        np.full((*PREVIEW, 3), 128, dtype=np.uint8),
        frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
    )
    qt_app.processEvents()  # type: ignore[attr-defined]
    return widget


def test_fitting_is_measured_against_the_photo_not_the_preview(qt_app: object) -> None:
    """The percentage the view bar shows is the photo's own (PLAN.md 3.3)."""
    widget = _zoomed_canvas(qt_app)
    try:
        fit = min(
            (CANVAS_WIDTH - 2 * theme.CANVAS_MARGIN) / PHOTO_FULL[1],
            (CANVAS_HEIGHT - 2 * theme.CANVAS_MARGIN) / PHOTO_FULL[0],
        )
        assert widget.scale == pytest.approx(fit, rel=0.01)
        # Not the 1.5 the preview's own pixels would have given.
        assert widget.scale < 0.25
    finally:
        widget.hide()
        widget.deleteLater()


def test_actual_size_is_one_photo_pixel_to_one_screen_pixel(qt_app: object) -> None:
    widget = _zoomed_canvas(qt_app)
    try:
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        assert widget.scale == pytest.approx(1.0)
        # At actual size the whole photo is far larger than the window, so
        # what is on screen is a small part of it.
        region = widget.visible_region()
        assert region is not None
        assert (region[2] - region[0]) == pytest.approx(
            CANVAS_WIDTH / PHOTO_FULL[1], rel=0.05
        )
    finally:
        widget.hide()
        widget.deleteLater()


def test_a_window_is_drawn_in_the_place_it_was_taken_from(qt_app: object) -> None:
    widget = _zoomed_canvas(qt_app)
    try:
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        middle = (0.25, 0.25, 0.5, 0.5)
        widget.set_photo(
            np.full((1000, 1500, 3), 64, dtype=np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
            region=middle,
        )

        # The zoom survives a window replacing the whole photo, and the photo
        # is still the size it always was.
        assert widget.scale == pytest.approx(1.0)
        photo = widget.photo_rect()
        assert photo.width() == pytest.approx(PHOTO_FULL[1])
        assert widget._source.left() == pytest.approx(0.25 * PHOTO_FULL[1])
        assert widget._source.width() == pytest.approx(0.25 * PHOTO_FULL[1])
    finally:
        widget.hide()
        widget.deleteLater()


def test_real_pixels_are_only_shown_where_they_are_real(qt_app: object) -> None:
    """The mosaic the stretched preview would otherwise be (PLAN.md 3.3)."""
    widget = _zoomed_canvas(qt_app)
    try:
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        # The whole photo's preview, stretched over the window: smoothed.
        assert not widget.showing_photo_pixels

        region = widget.visible_region()
        assert region is not None
        width = round((region[2] - region[0]) * PHOTO_FULL[1])
        height = round((region[3] - region[1]) * PHOTO_FULL[0])
        widget.set_photo(
            np.full((height, width, 3), 64, dtype=np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
            region=region,
        )
        assert widget.showing_photo_pixels
    finally:
        widget.hide()
        widget.deleteLater()


def test_the_view_is_announced_with_what_part_of_the_photo_is_on_screen(
    qt_app: object,
) -> None:
    widget = _zoomed_canvas(qt_app)
    seen: list[tuple[float, object, bool]] = []
    widget.view_changed.connect(
        lambda scale, region, settled: seen.append((scale, region, settled))
    )
    try:
        # Fitted, there is no part of the photo to develop on its own.
        widget.fit_to_window()
        assert seen and seen[-1][1] is None

        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        scale, region, settled = seen[-1]
        assert scale == pytest.approx(1.0)
        assert region is not None
        assert 0.0 <= region[0] < region[2] <= 1.0
        assert 0.0 <= region[1] < region[3] <= 1.0
        # The zoom is over, so there is nothing to wait for.
        assert settled
    finally:
        widget.hide()
        widget.deleteLater()


def test_panning_says_where_the_photo_has_been_moved_to(qt_app: object) -> None:
    widget = _zoomed_canvas(qt_app)
    try:
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        before = widget.visible_region()
        seen: list[tuple[object, bool]] = []
        widget.view_changed.connect(
            lambda scale, region, settled: seen.append((region, settled))
        )

        middle = QPointF(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)
        moved = QPointF(CANVAS_WIDTH / 2 - 120, CANVAS_HEIGHT / 2)
        _send_mouse(widget, QMouseEvent.Type.MouseButtonPress, middle)
        _send_mouse(widget, QMouseEvent.Type.MouseMove, moved, held=True)
        _send_mouse(widget, QMouseEvent.Type.MouseButtonRelease, moved)

        after = widget.visible_region()
        assert before is not None and after is not None
        assert after[0] > before[0]
        # Moving says so as it goes; letting go says the hand is off.
        assert [settled for _region, settled in seen] == [False, True]
        assert seen[-1][0] == after
    finally:
        widget.hide()
        widget.deleteLater()


def _colour_at(canvas: PhotoCanvas, x: int, y: int) -> tuple[int, int, int]:
    """Return what is actually painted at a point of the canvas."""
    image = canvas.grab().toImage()
    pixel = image.pixelColor(x, y)
    return (pixel.red(), pixel.green(), pixel.blue())


def test_panning_uncovers_the_photo_rather_than_the_background(qt_app: object) -> None:
    """What the user saw instead: a black ring while the window caught up."""
    widget = _zoomed_canvas(qt_app)
    try:
        # The whole photo, in a colour of its own, is what the canvas holds.
        widget.set_photo(
            np.full((*PREVIEW, 3), 0, dtype=np.uint8) + np.array([200, 40, 40], np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
        )
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        region = widget.visible_region()
        assert region is not None

        # A window of it arrives, in another colour, covering the view.
        width = round((region[2] - region[0]) * PHOTO_FULL[1])
        height = round((region[3] - region[1]) * PHOTO_FULL[0])
        widget.set_photo(
            np.full((height, width, 3), 0, dtype=np.uint8) + np.array([40, 40, 200], np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
            region=region,
        )
        qt_app.processEvents()  # type: ignore[attr-defined]
        assert _colour_at(widget, CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)[2] > 150

        # Panning moves the window off the right hand side of the view; what
        # stands there is the photo, not the empty background.
        widget._centre += QPointF(CANVAS_WIDTH, 0.0)
        widget._clamp_centre()
        qt_app.processEvents()  # type: ignore[attr-defined]
        uncovered = _colour_at(widget, CANVAS_WIDTH - 40, CANVAS_HEIGHT // 2)
        assert uncovered[0] > 150, f"the photo should be under the window, not {uncovered}"
        assert uncovered[0] > uncovered[2]
    finally:
        widget.hide()
        widget.deleteLater()


def test_the_photo_underneath_is_let_go_of_when_another_one_arrives(
    qt_app: object,
) -> None:
    """It belongs to the frame it was developed for and to no other."""
    widget = _zoomed_canvas(qt_app)
    try:
        assert widget._backdrop is not None
        widget.set_photo(
            np.full((100, 150, 3), 90, dtype=np.uint8),
            frame=(3000, 2000),
            region=(0.1, 0.1, 0.2, 0.2),
        )
        assert widget._backdrop is None
    finally:
        widget.hide()
        widget.deleteLater()


def test_the_photo_underneath_can_be_replaced_without_moving_the_window(
    qt_app: object,
) -> None:
    """The window on screen is sharper; only what it does not cover changes."""
    widget = _zoomed_canvas(qt_app)
    try:
        widget.zoom_to(ZOOM_ACTUAL, animate=False)
        region = widget.visible_region()
        assert region is not None
        width = round((region[2] - region[0]) * PHOTO_FULL[1])
        height = round((region[3] - region[1]) * PHOTO_FULL[0])
        widget.set_photo(
            np.full((height, width, 3), 0, dtype=np.uint8) + np.array([40, 40, 200], np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
            region=region,
        )

        widget.set_backdrop(
            np.full((*PREVIEW, 3), 0, dtype=np.uint8) + np.array([200, 40, 40], np.uint8),
            frame=(PHOTO_FULL[1], PHOTO_FULL[0]),
        )
        qt_app.processEvents()  # type: ignore[attr-defined]

        assert widget.scale == pytest.approx(1.0)
        assert _colour_at(widget, CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)[2] > 150
        widget._centre += QPointF(CANVAS_WIDTH, 0.0)
        widget._clamp_centre()
        qt_app.processEvents()  # type: ignore[attr-defined]
        assert _colour_at(widget, CANVAS_WIDTH - 40, CANVAS_HEIGHT // 2)[0] > 150
    finally:
        widget.hide()
        widget.deleteLater()


def test_the_photo_underneath_is_left_alone_while_the_whole_photo_is_shown(
    qt_app: object,
) -> None:
    """Fitted, the photo on screen is already the whole of it."""
    widget = _zoomed_canvas(qt_app)
    try:
        before = widget._backdrop
        widget.set_backdrop(
            np.full((*PREVIEW, 3), 7, dtype=np.uint8), frame=(PHOTO_FULL[1], PHOTO_FULL[0])
        )
        assert widget._backdrop is before
    finally:
        widget.hide()
        widget.deleteLater()


# -- the white balance eyedropper (A-24) ------------------------------------


def _click(canvas: PhotoCanvas, position: QPointF) -> None:
    _send_mouse(canvas, QMouseEvent.Type.MouseButtonPress, position)
    _send_mouse(canvas, QMouseEvent.Type.MouseButtonRelease, position, held=True)


def test_a_click_with_the_eyedropper_says_where_on_the_photo(canvas: PhotoCanvas) -> None:
    picked: list[tuple[float, float]] = []
    canvas.white_balance_picked.connect(lambda across, down: picked.append((across, down)))
    canvas.set_picking(True)
    photo = canvas.photo_rect()

    _click(canvas, QPointF(photo.left() + photo.width() * 0.25, photo.top() + photo.height() * 0.75))

    assert len(picked) == 1
    assert picked[0][0] == pytest.approx(0.25, abs=0.01)
    assert picked[0][1] == pytest.approx(0.75, abs=0.01)


def test_a_click_beside_the_photo_is_not_a_pick(canvas: PhotoCanvas) -> None:
    picked: list[object] = []
    canvas.white_balance_picked.connect(lambda *args: picked.append(args))
    canvas.set_picking(True)
    photo = canvas.photo_rect()

    _click(canvas, QPointF(photo.left() - 8, photo.center().y()))

    assert picked == []


def test_without_the_eyedropper_a_click_is_no_pick(canvas: PhotoCanvas) -> None:
    picked: list[object] = []
    canvas.white_balance_picked.connect(lambda *args: picked.append(args))

    _click(canvas, canvas.photo_rect().center())

    assert picked == []


def test_the_eyedropper_shows_a_crosshair(canvas: PhotoCanvas) -> None:
    canvas.set_picking(True)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor
    canvas.set_picking(False)
    assert canvas.cursor().shape() != Qt.CursorShape.CrossCursor


# -- arriving from the welcome screen, and a photo taking over (PLAN.md 3.6, 3.7)


def _wait(predicate, timeout_ms: int = 4000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
        if predicate():
            return True
    return predicate()


@pytest.fixture
def area(qt_app: object) -> Iterator[PreviewArea]:
    view = PreviewArea()
    view.resize(800, 600)
    view.show()
    yield view
    view.close()
    view.deleteLater()


def test_leaving_the_welcome_screen_cross_fades_it_away(area: PreviewArea) -> None:
    area.show_photo(np.full((300, 400, 3), 200, dtype=np.uint8))

    snapshot = area.leaving
    assert snapshot is not None, "the welcome screen was cut away"
    assert area.canvas.fading, "the first photo did not fade in under it"
    assert _wait(lambda: area.leaving is None)


def test_a_photo_moved_to_from_another_photo_leaves_no_snapshot(area: PreviewArea) -> None:
    area.show_photo(np.full((300, 400, 3), 200, dtype=np.uint8))
    assert _wait(lambda: area.leaving is None)

    area.show_photo(np.full((300, 400, 3), 90, dtype=np.uint8), fade=True)
    assert area.leaving is None


def test_with_no_animation_the_welcome_screen_is_simply_gone(
    area: PreviewArea, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    area.show_photo(np.full((300, 400, 3), 200, dtype=np.uint8))
    assert area.leaving is None
    assert not area.canvas.fading


def test_a_new_picture_of_the_same_photo_blends_over_the_old(canvas: PhotoCanvas) -> None:
    """The camera's preview giving way to the developed photo (PLAN.md 3.7)."""
    canvas.set_photo(np.full((200, 300, 3), 40, dtype=np.uint8), frame=(3000, 2000))
    canvas.set_photo(
        np.full((400, 600, 3), 220, dtype=np.uint8), frame=(3000, 2000), blend=True
    )
    assert canvas.blending
    middle = canvas.photo_rect().center().toPoint()
    halfway = canvas.grab().toImage().pixelColor(middle).lightness()
    assert _wait(lambda: not canvas.blending)
    done = canvas.grab().toImage().pixelColor(middle).lightness()
    assert halfway < done


def test_a_picture_of_another_shape_is_swapped_rather_than_blended(
    canvas: PhotoCanvas,
) -> None:
    """Blending two framings would show both at once."""
    canvas.set_photo(np.full((200, 300, 3), 40, dtype=np.uint8), frame=(3000, 2000))
    canvas.set_photo(
        np.full((300, 300, 3), 220, dtype=np.uint8), frame=(2000, 2000), blend=True
    )
    assert not canvas.blending


def test_the_developing_sign_comes_up_and_goes(area: PreviewArea) -> None:
    """It sits on the photo page: over the camera's preview of a RAW file."""
    area.show_photo(np.full((300, 400, 3), 120, dtype=np.uint8))
    sign = area.developing
    area.set_developing(True)
    assert _wait(lambda: sign.shown == 1.0)
    assert sign.isVisible()
    # Level with the view bar, at the left.
    bar = area.view_bar.geometry()
    assert sign.x() == theme.CANVAS_MARGIN
    assert abs(sign.geometry().center().y() - bar.center().y()) <= 1

    area.set_developing(False)
    assert _wait(lambda: not sign.isVisible())


def test_the_developing_sign_needs_no_animation_to_be_seen(
    area: PreviewArea, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import native

    monkeypatch.setattr(native, "animations_enabled", lambda: False)
    area.show_photo(np.full((300, 400, 3), 120, dtype=np.uint8))
    area.set_developing(True)
    assert area.developing.isVisible() and area.developing.shown == 1.0
    area.set_developing(False)
    assert not area.developing.isVisible()


def test_the_view_bar_floats_over_the_photo_rather_than_under_it(
    area: PreviewArea, qt_app: object
) -> None:
    """PLAN.md 3.3: a zoomed photo carries on behind the bar, with no strip."""
    area.show_photo(np.full((3000, 4000, 3), 120, dtype=np.uint8))
    qt_app.processEvents()  # type: ignore[attr-defined]
    canvas = area.canvas
    bar = area.view_bar.geometry()
    page = area.view_bar.parentWidget()
    assert canvas.height() == page.height(), "no strip of its own under the canvas"
    assert bar.bottom() == page.height() - theme.SPACE_MD - 1
    assert abs(bar.center().x() - page.width() / 2) <= 1

    # Fitted, the photo stays clear of the bar...
    assert canvas.photo_rect().bottom() <= bar.top()
    # ...zoomed in, it carries on behind it to the bottom edge.
    canvas.zoom_to(1.0, animate=False)
    assert canvas.photo_rect().bottom() >= canvas.height()
    assert area.view_bar.isVisible()


def test_the_view_bar_keeps_centred_when_its_words_change_width(
    area: PreviewArea, qt_app: object
) -> None:
    area.show_photo(np.full((3000, 4000, 3), 120, dtype=np.uint8))
    area.view_bar.set_zoom(0.05)
    qt_app.processEvents()  # type: ignore[attr-defined]
    narrow = area.view_bar.width()
    area.view_bar.set_zoom(16.0)
    qt_app.processEvents()  # type: ignore[attr-defined]
    page = area.view_bar.parentWidget()
    assert area.view_bar.width() >= narrow
    assert abs(area.view_bar.geometry().center().x() - page.width() / 2) <= 1
