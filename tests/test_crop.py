"""Tests for the crop box and the crop tab (PLAN.md 3.2, C-01 to C-07)."""

from __future__ import annotations

import pytest

from core import geometry
from core.geometry import Geometry, Handle, Rect

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402

from ui import theme  # noqa: E402
from ui.crop_overlay import CropOverlay  # noqa: E402
from ui.crop_tab import CropTab  # noqa: E402

#: Where the photo is drawn in the tests, and how large it really is.
PHOTO = QRectF(100.0, 50.0, 400.0, 300.0)
SOURCE = (4000, 3000)


@pytest.fixture
def overlay(qt_app: object) -> CropOverlay:
    box = CropOverlay()
    box.set_source(SOURCE)
    box.set_geometry(Geometry(crop=Rect(0.25, 0.25, 0.5, 0.5)))
    box.set_active(True)
    return box


def _point(fraction_x: float, fraction_y: float) -> QPointF:
    """Return the point at those fractions of the drawn photo."""
    return QPointF(
        PHOTO.x() + fraction_x * PHOTO.width(), PHOTO.y() + fraction_y * PHOTO.height()
    )


# -- what the pointer is over ---------------------------------------------


def test_the_middle_of_the_box_moves_it(overlay: CropOverlay) -> None:
    assert overlay.handle_at(_point(0.5, 0.5), PHOTO) is Handle.INSIDE


def test_each_corner_is_its_own_handle(overlay: CropOverlay) -> None:
    assert overlay.handle_at(_point(0.25, 0.25), PHOTO) is Handle.TOP_LEFT
    assert overlay.handle_at(_point(0.75, 0.25), PHOTO) is Handle.TOP_RIGHT
    assert overlay.handle_at(_point(0.25, 0.75), PHOTO) is Handle.BOTTOM_LEFT
    assert overlay.handle_at(_point(0.75, 0.75), PHOTO) is Handle.BOTTOM_RIGHT


def test_each_edge_is_its_own_handle(overlay: CropOverlay) -> None:
    assert overlay.handle_at(_point(0.25, 0.5), PHOTO) is Handle.LEFT
    assert overlay.handle_at(_point(0.75, 0.5), PHOTO) is Handle.RIGHT
    assert overlay.handle_at(_point(0.5, 0.25), PHOTO) is Handle.TOP
    assert overlay.handle_at(_point(0.5, 0.75), PHOTO) is Handle.BOTTOM


def test_outside_the_box_turns_the_photo(overlay: CropOverlay) -> None:
    """PLAN.md 3.2: 拖曳框外旋轉照片."""
    assert overlay.handle_at(_point(0.02, 0.02), PHOTO) is Handle.OUTSIDE


def test_nothing_is_grabbed_when_the_box_is_hidden(overlay: CropOverlay) -> None:
    overlay.set_active(False)
    assert overlay.handle_at(_point(0.5, 0.5), PHOTO) is Handle.NONE


def test_every_handle_has_a_cursor_of_its_own(overlay: CropOverlay) -> None:
    cursors = {
        overlay.cursor_for(handle)
        for handle in (Handle.TOP_LEFT, Handle.TOP, Handle.LEFT, Handle.INSIDE)
    }
    assert len(cursors) == 4


# -- dragging --------------------------------------------------------------


def test_dragging_the_middle_moves_the_box(overlay: CropOverlay) -> None:
    moves: list[tuple[Rect, bool]] = []
    overlay.crop_changed.connect(lambda rect, dragging: moves.append((rect, dragging)))

    overlay.press(_point(0.5, 0.5), PHOTO)
    overlay.move(_point(0.6, 0.5), PHOTO)

    assert moves
    rect, dragging = moves[-1]
    assert dragging
    assert rect.x == pytest.approx(0.35, abs=0.01)
    assert rect.w == pytest.approx(0.5)


def test_dragging_a_corner_resizes_the_box(overlay: CropOverlay) -> None:
    sizes: list[Rect] = []
    overlay.crop_changed.connect(lambda rect, _dragging: sizes.append(rect))

    overlay.press(_point(0.25, 0.25), PHOTO)
    overlay.move(_point(0.35, 0.35), PHOTO)

    assert sizes[-1].x == pytest.approx(0.35, abs=0.01)
    assert sizes[-1].right == pytest.approx(0.75, abs=0.01)


def test_letting_go_ends_the_drag(overlay: CropOverlay) -> None:
    finished: list[bool] = []
    overlay.crop_changed.connect(lambda _rect, dragging: finished.append(dragging))

    overlay.press(_point(0.5, 0.5), PHOTO)
    overlay.move(_point(0.55, 0.5), PHOTO)
    overlay.release(_point(0.55, 0.5), PHOTO)

    assert finished[-1] is False, "the end of the drag was never announced"


def test_dragging_outside_turns_the_photo(overlay: CropOverlay) -> None:
    angles: list[tuple[float, bool]] = []
    overlay.angle_changed.connect(lambda angle, dragging: angles.append((angle, dragging)))

    overlay.press(_point(0.5, 0.05), PHOTO)
    overlay.move(_point(0.6, 0.06), PHOTO)
    overlay.release(_point(0.6, 0.06), PHOTO)

    assert angles, "turning the photo was never announced"
    assert angles[-1][1] is False
    assert angles[-1][0] != 0.0
    assert geometry.ANGLE_MIN <= angles[-1][0] <= geometry.ANGLE_MAX


def test_a_drag_cannot_leave_the_photo(overlay: CropOverlay) -> None:
    rects: list[Rect] = []
    overlay.crop_changed.connect(lambda rect, _dragging: rects.append(rect))

    overlay.press(_point(0.5, 0.5), PHOTO)
    overlay.move(_point(2.0, 0.5), PHOTO)

    for rect in rects:
        assert geometry.is_inside(rect, SOURCE, 0.0)


# -- the straighten tool (C-04) -------------------------------------------


def test_the_straighten_tool_measures_the_line_that_is_drawn(
    overlay: CropOverlay,
) -> None:
    angles: list[tuple[float, bool]] = []
    overlay.angle_changed.connect(lambda angle, dragging: angles.append((angle, dragging)))
    finished: list[int] = []
    overlay.straighten_finished.connect(lambda: finished.append(1))

    overlay.arm_straighten(True)
    assert overlay.straightening
    overlay.press(QPointF(150.0, 200.0), PHOTO)
    overlay.move(QPointF(450.0, 260.0), PHOTO)
    overlay.release(QPointF(450.0, 260.0), PHOTO)

    assert angles == [(pytest.approx(11.31, abs=0.01), False)]
    assert finished
    assert not overlay.straightening


def test_a_line_too_short_to_mean_anything_is_ignored(overlay: CropOverlay) -> None:
    angles: list[float] = []
    overlay.angle_changed.connect(lambda angle, _dragging: angles.append(angle))

    overlay.arm_straighten(True)
    overlay.press(QPointF(200.0, 200.0), PHOTO)
    overlay.release(QPointF(202.0, 201.0), PHOTO)

    assert not angles
    assert not overlay.straightening


def test_straightening_adds_to_the_angle_already_set(overlay: CropOverlay) -> None:
    overlay.set_geometry(Geometry(angle=5.0, crop=Rect(0.25, 0.25, 0.5, 0.5)))
    angles: list[float] = []
    overlay.angle_changed.connect(lambda angle, _dragging: angles.append(angle))

    overlay.arm_straighten(True)
    overlay.press(QPointF(150.0, 200.0), PHOTO)
    overlay.move(QPointF(450.0, 200.0), PHOTO)
    overlay.release(QPointF(450.0, 200.0), PHOTO)

    assert angles == [pytest.approx(5.0)]


# -- the thirds grid (C-07) -----------------------------------------------


def test_the_grid_appears_while_dragging(overlay: CropOverlay) -> None:
    assert not overlay.grid_visible

    overlay.press(_point(0.5, 0.5), PHOTO)

    assert overlay.grid_visible


def test_the_grid_can_be_kept_on_screen(overlay: CropOverlay) -> None:
    """PLAN.md section 8: O shows and hides the grid."""
    overlay.set_grid_always(True)
    assert overlay.grid_visible

    overlay.set_grid_always(False)
    assert not overlay.grid_visible


# -- drawing ---------------------------------------------------------------


def test_the_crop_box_darkens_what_it_leaves_out(qt_app: object) -> None:
    """PLAN.md 3.6: 框外變暗."""
    from PySide6.QtGui import QImage, QPainter

    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    box = CropOverlay()
    box.set_source(SOURCE)
    box.set_geometry(Geometry(crop=Rect(0.25, 0.25, 0.5, 0.5)))
    box.set_active(True)

    painter = QPainter(image)
    box.paint(painter, QRectF(0, 0, 200, 200))
    painter.end()

    inside = image.pixelColor(100, 100).value()
    outside = image.pixelColor(10, 10).value()
    assert outside < inside
    assert outside == pytest.approx(255 * (1 - theme.CROP_OUTSIDE_DIM), abs=2)


def test_nothing_is_drawn_when_the_box_is_hidden(qt_app: object) -> None:
    from PySide6.QtGui import QImage, QPainter

    image = QImage(100, 100, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    box = CropOverlay()
    box.set_active(False)

    painter = QPainter(image)
    box.paint(painter, QRectF(0, 0, 100, 100))
    painter.end()

    assert image.pixelColor(10, 10).value() == 255


# -- the crop tab ----------------------------------------------------------


@pytest.fixture
def crop_tab(qt_app: object) -> CropTab:
    tab = CropTab()
    tab.set_photo(Geometry())
    return tab


def test_the_ratios_are_offered(crop_tab: CropTab) -> None:
    """PLAN.md 3.2: 原始、自由、1:1、3:2、4:5、16:9, and 9:16 after them."""
    assert tuple(crop_tab.aspect_buttons()) == geometry.ASPECTS


def test_choosing_a_ratio_announces_it(crop_tab: CropTab) -> None:
    chosen: list[str] = []
    crop_tab.aspect_chosen.connect(chosen.append)

    crop_tab.aspect_buttons()["16:9"].click()

    assert chosen == ["16:9"]


def test_choosing_the_same_ratio_again_turns_it_on_its_side(crop_tab: CropTab) -> None:
    """PLAN.md 3.2: 再點一次已選的比例，切換直式／橫式."""
    flips: list[int] = []
    crop_tab.orientation_flipped.connect(lambda: flips.append(1))
    crop_tab.set_geometry(Geometry(aspect="3:2"))

    crop_tab.aspect_buttons()["3:2"].click()

    assert flips == [1]


def test_the_angle_slider_covers_the_documented_range(crop_tab: CropTab) -> None:
    """C-03: -45 to +45 degrees."""
    spec = crop_tab.angle_row.spec
    assert spec.minimum == geometry.ANGLE_MIN
    assert spec.maximum == geometry.ANGLE_MAX


def test_moving_the_angle_slider_announces_it(crop_tab: CropTab) -> None:
    angles: list[tuple[float, bool]] = []
    crop_tab.angle_changed.connect(lambda angle, dragging: angles.append((angle, dragging)))

    crop_tab.angle_row.set_value(12.5, announce=True)

    assert angles == [(pytest.approx(12.5), False)]


def test_the_turn_buttons_ask_for_a_quarter_turn_each_way(crop_tab: CropTab) -> None:
    turns: list[int] = []
    crop_tab.turn_requested.connect(turns.append)

    crop_tab.turn_left_button.click()
    crop_tab.turn_right_button.click()

    assert turns == [-1, 1]


def test_the_flip_buttons_ask_for_each_flip(crop_tab: CropTab) -> None:
    """C-08: one button flips left to right, the other top to bottom."""
    flips: list[bool] = []
    crop_tab.mirror_requested.connect(flips.append)

    crop_tab.mirror_horizontal_button.click()
    crop_tab.mirror_vertical_button.click()

    assert flips == [False, True]


def test_only_straighten_has_a_tooltip(crop_tab: CropTab) -> None:
    """The turns and flips say everything on the button; straighten keeps
    its hint, since nothing else tells the user to draw a line (PLAN.md 3.2)."""
    from app.i18n import tr

    crop_tab.retranslate()

    assert crop_tab.straighten_button.toolTip() == tr("crop.straighten.hint")
    for button in (
        crop_tab.turn_left_button,
        crop_tab.turn_right_button,
        crop_tab.mirror_horizontal_button,
        crop_tab.mirror_vertical_button,
    ):
        assert button.toolTip() == ""


def test_the_straighten_button_arms_the_tool(crop_tab: CropTab) -> None:
    armed: list[bool] = []
    crop_tab.straighten_toggled.connect(armed.append)

    crop_tab.straighten_button.click()

    assert armed == [True]


def test_showing_a_geometry_does_not_announce_it_again(crop_tab: CropTab) -> None:
    heard: list[object] = []
    crop_tab.aspect_chosen.connect(heard.append)
    crop_tab.angle_changed.connect(lambda *args: heard.append(args))

    crop_tab.set_geometry(Geometry(angle=-3.0, aspect="4:5"))

    assert not heard
    assert crop_tab.aspect_buttons()["4:5"].isChecked()
    assert crop_tab.angle_row.value == pytest.approx(-3.0)


def test_the_tab_is_empty_until_a_photo_is_open(qt_app: object) -> None:
    tab = CropTab()
    assert not tab.angle_row.isVisible()


def test_dragging_the_box_up_to_the_top_keeps_it_there(overlay: CropOverlay) -> None:
    """What the user sees: push against the top edge and it stays put."""
    rects: list[Rect] = []
    overlay.crop_changed.connect(lambda rect, _dragging: rects.append(rect))

    overlay.press(_point(0.5, 0.5), PHOTO)
    for height in (0.4, 0.3, 0.2, 0.1, 0.0, -0.2, -0.5):
        overlay.move(_point(0.5, height), PHOTO)

    assert rects, "the drag was never announced"
    assert rects[-1].y == pytest.approx(0.0, abs=1e-4)
    assert [rect.y for rect in rects] == sorted((rect.y for rect in rects), reverse=True)


def test_clicking_original_again_gives_the_whole_picture_again(
    crop_tab: CropTab,
) -> None:
    """PLAN.md 3.2 turns the chosen ratio over on a second click, but
    "original" names the whole picture, so it is applied afresh instead."""
    chosen: list[str] = []
    flips: list[int] = []
    crop_tab.aspect_chosen.connect(chosen.append)
    crop_tab.orientation_flipped.connect(lambda: flips.append(1))
    crop_tab.set_geometry(Geometry(aspect=geometry.ASPECT_ORIGINAL))

    crop_tab.aspect_buttons()[geometry.ASPECT_ORIGINAL].click()

    assert chosen == [geometry.ASPECT_ORIGINAL]
    assert flips == []


@pytest.mark.parametrize(
    "aspect", [geometry.ASPECT_FREE, geometry.ASPECT_SQUARE]
)
def test_the_other_two_with_nothing_to_turn_are_applied_again(
    crop_tab: CropTab, aspect: str
) -> None:
    """A free crop has no shape and a square is the same either way."""
    chosen: list[str] = []
    flips: list[int] = []
    crop_tab.aspect_chosen.connect(chosen.append)
    crop_tab.orientation_flipped.connect(lambda: flips.append(1))
    crop_tab.set_geometry(Geometry(aspect=aspect))

    crop_tab.aspect_buttons()[aspect].click()

    assert chosen == [aspect]
    assert flips == []


@pytest.mark.parametrize(
    "aspect",
    [
        geometry.ASPECT_3_2,
        geometry.ASPECT_4_5,
        geometry.ASPECT_16_9,
        geometry.ASPECT_9_16,
    ],
)
def test_the_ratios_with_two_sides_still_turn_over(
    crop_tab: CropTab, aspect: str
) -> None:
    flips: list[int] = []
    crop_tab.orientation_flipped.connect(lambda: flips.append(1))
    crop_tab.set_geometry(Geometry(aspect=aspect))

    crop_tab.aspect_buttons()[aspect].click()

    assert flips == [1]


def _settle_fold(group) -> None:
    """Let a group finish folding, the way the eye waits for it.

    A group opens and closes over the 160 ms of PLAN.md 3.6, so what is on
    screen is only settled once the animation has run out.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + 2.0
    while group.folding and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def test_9_16_has_a_button_of_its_own(crop_tab: CropTab) -> None:
    """X reaches it from 16:9 as well, but nobody finds it that way."""
    button = crop_tab.aspect_buttons()[geometry.ASPECT_9_16]

    assert button.text() == "9:16"
    assert button.isVisibleTo(crop_tab)


def test_the_ratio_buttons_have_no_tooltip(crop_tab: CropTab) -> None:
    """The ratios say everything on the button itself (PLAN.md 3.2)."""
    crop_tab.retranslate()

    for button in crop_tab.aspect_buttons().values():
        assert button.toolTip() == ""


def test_the_crop_tab_is_laid_out_in_headed_groups(crop_tab: CropTab) -> None:
    """Every tab of the panel marks its sections the same way (PLAN.md 3.4:
    同類元素外觀完全一致)."""
    from app.i18n import tr

    headers = [group.header.text() for group in crop_tab.groups()]
    assert headers == [
        tr("crop.group.aspect"),
        tr("crop.group.angle"),
        tr("crop.group.orientation"),
    ]


def test_closing_a_crop_group_folds_its_controls_away(crop_tab: CropTab) -> None:
    crop_tab.set_photo(Geometry())
    tools = crop_tab.groups()[-1]
    assert crop_tab.straighten_button.isVisibleTo(crop_tab)

    tools.header.click()
    _settle_fold(tools)
    assert not tools.is_open
    assert not crop_tab.straighten_button.isVisibleTo(crop_tab)
    # The ratios above it are untouched.
    assert crop_tab.aspect_buttons()["3:2"].isVisibleTo(crop_tab)

    tools.header.click()
    _settle_fold(tools)
    assert crop_tab.straighten_button.isVisibleTo(crop_tab)
