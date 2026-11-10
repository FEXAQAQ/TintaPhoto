"""Tests for cropping, rotation and straightening (PLAN.md 2.3, C-01 to C-07).

The acceptance items of stage 4 are all here in some form: a fixed ratio that
survives every drag, a rotated photo whose crop never takes in a black
corner, and a straighten line that comes out level.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from core import geometry
from core.decoder import DecodedPhoto
from core.geometry import Geometry, Handle, Rect

SOURCE = (6000, 4000)
ANGLES = (-45.0, -30.0, -12.5, -0.3, 0.0, 0.3, 7.0, 22.5, 45.0)
RATIOS = (None, 1.0, 3 / 2, 2 / 3, 5 / 4, 16 / 9)


def _photo(width: int = 60, height: int = 40, encoded: bool = False) -> DecodedPhoto:
    """Return a small photo whose pixels say where they came from."""
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, 0] = np.linspace(0.0, 1.0, width, dtype=np.float32)
    image[:, :, 1] = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    clipped = np.zeros((height, width), dtype=bool)
    clipped[0, 0] = True
    return DecodedPhoto(
        image=image,
        is_raw=not encoded,
        clipped=clipped,
        encoded=(
            np.full((height, width, 3), 200, dtype=np.uint8) if encoded else None
        ),
    )


# -- sizes -----------------------------------------------------------------


def test_a_quarter_turn_swaps_width_and_height() -> None:
    assert geometry.turned_size(SOURCE, 0) == SOURCE
    assert geometry.turned_size(SOURCE, 1) == (4000, 6000)
    assert geometry.turned_size(SOURCE, 2) == SOURCE
    assert geometry.turned_size(SOURCE, 3) == (4000, 6000)


def test_rotating_grows_the_frame() -> None:
    assert geometry.rotated_size((100, 100), 0.0) == (100, 100)
    assert geometry.rotated_size((100, 100), 45.0) == (141, 141)
    assert geometry.rotated_size((100, 100), -45.0) == (141, 141)


# -- what shape a crop must keep (C-02) ------------------------------------


def test_the_original_aspect_is_the_shape_of_the_photo() -> None:
    assert geometry.aspect_ratio("original", "landscape", SOURCE) == pytest.approx(1.5)
    assert geometry.aspect_ratio("original", "portrait", SOURCE) == pytest.approx(1 / 1.5)


def test_a_free_crop_has_no_shape_to_keep() -> None:
    assert geometry.aspect_ratio("free", "landscape", SOURCE) is None


def test_portrait_turns_every_fixed_ratio_on_its_side() -> None:
    for aspect, landscape in (("1:1", 1.0), ("3:2", 1.5), ("4:5", 1.25), ("16:9", 16 / 9)):
        assert geometry.aspect_ratio(aspect, "landscape") == pytest.approx(landscape)
        assert geometry.aspect_ratio(aspect, "portrait") == pytest.approx(1 / landscape)


# -- no black edges (C-05) -------------------------------------------------


def test_the_whole_frame_is_only_valid_without_rotation() -> None:
    assert geometry.is_inside(geometry.WHOLE_FRAME, SOURCE, 0.0)
    assert not geometry.is_inside(geometry.WHOLE_FRAME, SOURCE, 5.0)


def test_the_largest_crop_holds_nothing_but_photo() -> None:
    for angle in ANGLES:
        for ratio in RATIOS:
            rect = geometry.largest_rect(SOURCE, angle, ratio)
            assert geometry.is_inside(rect, SOURCE, angle), (angle, ratio)


def test_the_largest_crop_keeps_the_shape_it_was_asked_for() -> None:
    for angle in ANGLES:
        for ratio in RATIOS:
            if ratio is None:
                continue
            rect = geometry.largest_rect(SOURCE, angle, ratio)
            frame = geometry.rotated_size(SOURCE, angle)
            width = rect.w * frame[0]
            height = rect.h * frame[1]
            assert width / height == pytest.approx(ratio, rel=1e-6), (angle, ratio)


def test_the_largest_crop_is_really_the_largest() -> None:
    """A per cent larger must already take in a corner."""
    for angle in (-30.0, -7.0, 3.0, 22.5):
        for ratio in (1.0, 3 / 2, 16 / 9):
            rect = geometry.largest_rect(SOURCE, angle, ratio)
            assert not geometry.is_inside(rect.scaled(1.01), SOURCE, angle), (angle, ratio)


def test_without_rotation_the_original_shape_fills_the_frame() -> None:
    rect = geometry.largest_rect(SOURCE, 0.0, SOURCE[0] / SOURCE[1])
    assert rect.x == pytest.approx(0.0)
    assert rect.y == pytest.approx(0.0)
    assert rect.w == pytest.approx(1.0)
    assert rect.h == pytest.approx(1.0)


def test_a_crop_that_has_left_the_photo_is_brought_back() -> None:
    rect = geometry.WHOLE_FRAME
    for angle in (1.0, 5.0, 17.0, 45.0):
        fitted = geometry.fit_inside(rect, SOURCE, angle)
        assert geometry.is_inside(fitted, SOURCE, angle), angle
        assert fitted.w <= 1.0 and fitted.h <= 1.0


def test_changing_the_angle_keeps_the_shape_of_the_crop() -> None:
    """A 4:5 crop must still be 4:5 after the angle changes (C-02 and C-05)."""
    ratio = geometry.aspect_ratio("4:5", "landscape")
    rect = geometry.largest_rect(SOURCE, 0.0, ratio)

    carried = geometry.rebase(rect, SOURCE, 0.0, 12.0)
    fitted = geometry.fit_inside(carried, SOURCE, 12.0)

    frame_after = geometry.rotated_size(SOURCE, 12.0)
    shape_after = (fitted.w * frame_after[0]) / (fitted.h * frame_after[1])
    assert geometry.is_inside(fitted, SOURCE, 12.0)
    assert shape_after == pytest.approx(ratio, rel=1e-6)
    assert fitted.w * frame_after[0] < rect.w * SOURCE[0]  # it had to give way


def test_carrying_a_crop_to_another_angle_keeps_its_pixels() -> None:
    rect = Rect(0.2, 0.25, 0.4, 0.3)
    before = geometry.rotated_size(SOURCE, 0.0)
    after = geometry.rotated_size(SOURCE, 20.0)

    carried = geometry.rebase(rect, SOURCE, 0.0, 20.0)

    assert carried.w * after[0] == pytest.approx(rect.w * before[0])
    assert carried.h * after[1] == pytest.approx(rect.h * before[1])
    assert geometry.rebase(carried, SOURCE, 20.0, 0.0).x == pytest.approx(rect.x)


def test_a_crop_already_inside_is_left_alone() -> None:
    rect = geometry.largest_rect(SOURCE, 10.0, 1.0)
    assert geometry.fit_inside(rect, SOURCE, 10.0) is rect


# -- dragging the box (PLAN.md 3.2) ----------------------------------------


def test_dragging_a_corner_with_a_fixed_ratio_keeps_the_ratio() -> None:
    """Stage 4 acceptance: 選 4:5 後拖曳拉把，比例始終維持 4:5."""
    ratio = geometry.aspect_ratio("4:5", "portrait")  # 4 wide, 5 high
    frame = geometry.rotated_size(SOURCE, 0.0)
    rect = geometry.largest_rect(SOURCE, 0.0, ratio)

    for pointer in ((0.1, 0.1), (0.25, 0.4), (0.02, 0.3), (0.45, 0.45)):
        dragged = geometry.resize(rect, Handle.TOP_LEFT, pointer, SOURCE, 0.0, ratio)
        shape = (dragged.w * frame[0]) / (dragged.h * frame[1])
        assert shape == pytest.approx(ratio, rel=1e-6), pointer
        assert geometry.is_inside(dragged, SOURCE, 0.0)


def test_dragging_an_edge_with_a_fixed_ratio_keeps_the_ratio() -> None:
    ratio = geometry.aspect_ratio("16:9", "landscape")
    frame = geometry.rotated_size(SOURCE, 0.0)
    rect = geometry.largest_rect(SOURCE, 0.0, ratio)

    dragged = geometry.resize(rect, Handle.LEFT, (0.3, 0.5), SOURCE, 0.0, ratio)

    shape = (dragged.w * frame[0]) / (dragged.h * frame[1])
    assert shape == pytest.approx(ratio, rel=1e-6)


def test_a_free_crop_follows_the_pointer_on_one_axis_only() -> None:
    rect = Rect(0.2, 0.2, 0.6, 0.6)

    dragged = geometry.resize(rect, Handle.LEFT, (0.35, 0.9), SOURCE, 0.0, None)

    assert dragged.x == pytest.approx(0.35)
    assert dragged.y == pytest.approx(rect.y)
    assert dragged.bottom == pytest.approx(rect.bottom)


def test_a_drag_that_would_take_in_a_black_corner_is_refused() -> None:
    """Stage 4 acceptance: 旋轉 10° 後，裁切框內不會出現黑邊."""
    angle = 10.0
    rect = geometry.largest_rect(SOURCE, angle, None)

    dragged = geometry.resize(rect, Handle.TOP_LEFT, (0.0, 0.0), SOURCE, angle, None)

    assert dragged is rect
    assert geometry.is_inside(dragged, SOURCE, angle)


def test_a_drag_cannot_make_the_box_disappear() -> None:
    rect = Rect(0.2, 0.2, 0.6, 0.6)

    dragged = geometry.resize(rect, Handle.RIGHT, (0.2001, 0.5), SOURCE, 0.0, None)

    assert dragged.w >= geometry.MINIMUM_SIDE


def test_moving_the_box_stops_at_the_edge() -> None:
    rect = Rect(0.1, 0.1, 0.3, 0.3)

    moved = geometry.move(rect, (-0.5, 0.0), SOURCE, 0.0)

    assert geometry.is_inside(moved, SOURCE, 0.0)
    assert moved.x >= -1e-9


def test_moving_along_an_edge_still_works() -> None:
    """Pushing into the edge must not stop the other direction as well."""
    rect = Rect(0.0, 0.3, 0.3, 0.3)

    moved = geometry.move(rect, (-0.1, 0.2), SOURCE, 0.0)

    assert moved.y == pytest.approx(0.5)
    assert moved.x == pytest.approx(0.0)


def test_changing_the_ratio_keeps_the_middle_and_fits() -> None:
    angle = 8.0
    rect = geometry.largest_rect(SOURCE, angle, None)
    ratio = geometry.aspect_ratio("1:1", "landscape")

    reshaped = geometry.reshape(rect, SOURCE, angle, ratio)

    assert geometry.is_inside(reshaped, SOURCE, angle)
    frame = geometry.rotated_size(SOURCE, angle)
    shape = (reshaped.w * frame[0]) / (reshaped.h * frame[1])
    assert shape == pytest.approx(ratio, rel=0.01)


# -- straightening (C-04) --------------------------------------------------


def test_the_straighten_line_comes_out_level() -> None:
    """Stage 4 acceptance: 沿地平線畫線，套用後地平線水平."""
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.float32)
    start, end = (50.0, 100.0), (350.0, 160.0)
    cv2.line(image, (50, 100), (350, 160), (1.0, 1.0, 1.0), 2)

    angle = geometry.straighten_angle(start, end)
    assert angle == pytest.approx(math.degrees(math.atan2(60, 300)), abs=0.01)

    photo = DecodedPhoto(
        image=image, is_raw=True, clipped=np.zeros((height, width), dtype=bool)
    )
    turned = geometry.apply(photo, Geometry(angle=angle), crop=False)

    rows = []
    for column in range(20, turned.image.shape[1] - 20, 10):
        values = turned.image[:, column, 0]
        if values.max() > 0.3:
            rows.append(int(np.argmax(values)))
    assert rows, "the line disappeared"
    assert max(rows) - min(rows) <= 3, rows


def test_a_line_drawn_backwards_gives_the_same_angle() -> None:
    forward = geometry.straighten_angle((10.0, 10.0), (110.0, 30.0))
    backward = geometry.straighten_angle((110.0, 30.0), (10.0, 10.0))
    assert forward == pytest.approx(backward)


def test_a_line_along_a_vertical_edge_levels_the_other_way() -> None:
    """A building's edge is vertical, so it is read against the vertical."""
    angle = geometry.straighten_angle((100.0, 10.0), (120.0, 210.0))
    assert angle == pytest.approx(-math.degrees(math.atan2(20, 200)), abs=0.01)


def test_a_line_of_no_length_asks_for_no_rotation() -> None:
    assert geometry.straighten_angle((10.0, 10.0), (10.0, 10.0)) == 0.0


def test_the_straighten_angle_stays_within_the_slider() -> None:
    for start, end in (((0.0, 0.0), (10.0, 300.0)), ((0.0, 300.0), (10.0, 0.0))):
        angle = geometry.straighten_angle(start, end)
        assert geometry.ANGLE_MIN <= angle <= geometry.ANGLE_MAX


# -- doing it to the photo -------------------------------------------------


def test_nothing_is_done_when_there_is_nothing_to_do() -> None:
    """An untouched JPEG must stay the photo it was (JPG 原圖直通)."""
    photo = _photo(encoded=True)

    assert geometry.apply(photo, None) is photo
    assert geometry.apply(photo, Geometry()) is photo


def test_a_quarter_turn_turns_the_photo_clockwise() -> None:
    photo = _photo(width=60, height=40)
    marked = photo.image.copy()
    marked[0:2, 0:2] = 1.0
    photo = DecodedPhoto(image=marked, is_raw=True, clipped=photo.clipped)

    turned = geometry.apply(photo, Geometry(quarter_turns=1))

    assert turned.image.shape[:2] == (60, 40)
    # The corner that was top left is now top right.
    assert turned.image[0:2, -2:].max() == pytest.approx(1.0)


def test_two_quarter_turns_are_the_photo_upside_down() -> None:
    photo = _photo()

    turned = geometry.apply(photo, Geometry(quarter_turns=2))

    assert turned.image.shape == photo.image.shape
    assert np.allclose(turned.image, photo.image[::-1, ::-1])


def test_cropping_takes_the_asked_for_part() -> None:
    photo = _photo(width=100, height=100)

    cropped = geometry.apply(photo, Geometry(crop=Rect(0.25, 0.5, 0.5, 0.25)))

    assert cropped.image.shape[:2] == (25, 50)
    assert np.allclose(cropped.image, photo.image[50:75, 25:75])


def test_everything_that_belongs_to_the_photo_is_carried_along() -> None:
    photo = _photo(encoded=True)

    cropped = geometry.apply(photo, Geometry(quarter_turns=1, crop=Rect(0.0, 0.0, 0.5, 0.5)))

    assert cropped.encoded is not None
    assert cropped.encoded.shape[:2] == cropped.image.shape[:2]
    assert cropped.clipped.shape == cropped.image.shape[:2]
    assert cropped.is_raw == photo.is_raw


def test_the_crop_tab_sees_the_whole_rotated_photo() -> None:
    """PLAN.md 3.2: 預覽區顯示完整未裁切的照片."""
    photo = _photo(width=100, height=100)
    crop = Geometry(angle=10.0, crop=Rect(0.3, 0.3, 0.2, 0.2))

    whole = geometry.apply(photo, crop, crop=False)
    cropped = geometry.apply(photo, crop)

    assert whole.image.shape[:2] == geometry.rotated_size((100, 100), 10.0)[::-1]
    assert cropped.image.shape[0] < whole.image.shape[0]


def test_the_result_is_contiguous_so_qt_can_show_it() -> None:
    photo = _photo()

    cropped = geometry.apply(photo, Geometry(quarter_turns=3, crop=Rect(0.1, 0.1, 0.5, 0.5)))

    assert cropped.image.flags["C_CONTIGUOUS"]
    assert cropped.clipped.flags["C_CONTIGUOUS"]


# -- the stored form (PLAN.md 7.1) -----------------------------------------


def test_a_geometry_survives_being_stored_and_read_back() -> None:
    original = Geometry(
        quarter_turns=3,
        angle=-1.2,
        crop=Rect(0.05, 0.08, 0.9, 0.84),
        aspect="3:2",
        orientation="portrait",
    )

    assert Geometry.from_dict(original.to_dict()) == original


def test_nonsense_in_a_stored_geometry_falls_back() -> None:
    restored = Geometry.from_dict(
        {
            "quarter_turns": 7,
            "angle": 900.0,
            "crop": {"x": "left", "w": -1},
            "aspect": "golden",
            "orientation": "sideways",
        }
    )

    assert restored.quarter_turns == 3
    assert restored.angle == geometry.ANGLE_MAX
    assert restored.aspect == "original"
    assert restored.orientation == "landscape"
    assert geometry.is_inside(restored.crop, SOURCE, 0.0)


def test_an_untouched_geometry_knows_it() -> None:
    assert Geometry().is_default
    assert not Geometry(angle=0.5).is_default
    assert not Geometry(quarter_turns=1).is_default
    assert not Geometry(crop=Rect(0.1, 0.1, 0.8, 0.8)).is_default


def test_turning_the_photo_turns_the_crop_with_it() -> None:
    """A crop in the top left corner follows the picture round."""
    before = Geometry(crop=Rect(0.0, 0.0, 0.4, 0.2), orientation="landscape")

    after = before.turned(1)

    assert after.quarter_turns == 1
    assert after.orientation == "portrait"
    assert after.crop == Rect(0.8, 0.0, 0.2, 0.4)
    assert before.turned(4).crop == before.crop


def test_turning_four_times_comes_back_to_the_start() -> None:
    before = Geometry(crop=Rect(0.1, 0.2, 0.3, 0.4))
    after = before.turned(1).turned(1).turned(1).turned(1)
    assert after.quarter_turns == 0
    assert after.crop.x == pytest.approx(before.crop.x)
    assert after.crop.y == pytest.approx(before.crop.y)
    assert after.crop.w == pytest.approx(before.crop.w)
    assert after.crop.h == pytest.approx(before.crop.h)


# -- the size of what comes out (PLAN.md 3.11) ---------------------------


def test_the_output_size_of_an_untouched_photo_is_its_own() -> None:
    assert geometry.output_size((6000, 4000), None) == (6000, 4000)
    assert geometry.output_size((6000, 4000), Geometry()) == (6000, 4000)


def test_a_crop_makes_the_output_smaller() -> None:
    assert geometry.output_size(
        (6000, 4000), Geometry(crop=Rect(0.25, 0.25, 0.5, 0.5))
    ) == (3000, 2000)


def test_turning_the_photo_turns_the_output_size() -> None:
    assert geometry.output_size((6000, 4000), Geometry(quarter_turns=1)) == (4000, 6000)


def test_rotating_widens_the_frame_the_crop_is_taken_from() -> None:
    straight = geometry.output_size((6000, 4000), Geometry(crop=Rect(0.0, 0.0, 0.5, 0.5)))
    tilted = geometry.output_size(
        (6000, 4000), Geometry(angle=10.0, crop=Rect(0.0, 0.0, 0.5, 0.5))
    )
    assert tilted[0] > straight[0] and tilted[1] > straight[1]


def test_the_output_size_matches_what_applying_the_geometry_gives() -> None:
    """The dialog's figure and the exported file must not disagree."""
    frame = Geometry(quarter_turns=1, angle=4.0, crop=Rect(0.1, 0.15, 0.6, 0.5))
    photo = _photo(240, 160)
    applied = geometry.apply(photo, frame)
    assert geometry.output_size(photo.size, frame) == applied.size


# -- a drag that pushes past an edge (C-05) ------------------------------


def test_pushing_past_the_top_leaves_the_box_against_it() -> None:
    """The box must stay where it was pushed, not snap back to the grab.

    The overlay measures the whole drag from where the box was when the
    pointer went down, so refusing a movement that overshoots would send
    the box back to where the drag began, which is what it used to do.
    """
    start = Rect(0.25, 0.4, 0.5, 0.4)
    at_edge = geometry.move(start, (0.0, -0.4), (4000, 3000), 0.0)
    assert at_edge.y == pytest.approx(0.0)

    further = geometry.move(start, (0.0, -0.9), (4000, 3000), 0.0)
    assert further.y == pytest.approx(0.0)
    assert further.w == start.w and further.h == start.h


@pytest.mark.parametrize(
    "delta, expected",
    [
        ((0.0, -2.0), (0.25, 0.0)),
        ((0.0, 2.0), (0.25, 0.6)),
        ((-2.0, 0.0), (0.0, 0.4)),
        ((2.0, 0.0), (0.5, 0.4)),
        ((-2.0, -2.0), (0.0, 0.0)),
        ((2.0, 2.0), (0.5, 0.6)),
    ],
)
def test_a_drag_far_past_an_edge_stops_against_it(
    delta: tuple[float, float], expected: tuple[float, float]
) -> None:
    moved = geometry.move(Rect(0.25, 0.4, 0.5, 0.4), delta, (4000, 3000), 0.0)
    assert (moved.x, moved.y) == pytest.approx(expected, abs=1e-4)


def test_the_box_still_slides_along_the_edge_it_is_pressed_against() -> None:
    moved = geometry.move(Rect(0.25, 0.4, 0.5, 0.4), (0.1, -0.9), (4000, 3000), 0.0)
    assert moved.y == pytest.approx(0.0)
    assert moved.x == pytest.approx(0.35)


@pytest.mark.parametrize("angle", [0.0, 3.0, 10.0, -7.5])
def test_a_drag_never_sends_the_box_back_towards_where_it_started(
    angle: float,
) -> None:
    """Dragging further up may stop the box, but must never lower it."""
    start = Rect(0.25, 0.4, 0.5, 0.4)
    source = (4000, 3000)
    highest = start.y
    for step in range(1, 25):
        moved = geometry.move(start, (0.0, -0.05 * step), source, angle)
        # Where the edge is, is found by halving, so the last digit of two
        # neighbouring steps can differ; a hundredth of a pixel is not a
        # drop back.
        assert moved.y <= highest + 1e-4, f"the box dropped back at step {step}"
        assert geometry.is_inside(moved, source, angle)
        highest = moved.y


def test_a_drag_that_fits_is_left_exactly_as_it_is() -> None:
    moved = geometry.move(Rect(0.25, 0.4, 0.5, 0.4), (0.1, -0.1), (4000, 3000), 0.0)
    assert (moved.x, moved.y) == pytest.approx((0.35, 0.3))


# -- going back to the photo's own shape (C-02) --------------------------


@pytest.mark.parametrize("source", [(4000, 6000), (6024, 4024)])
def test_choosing_original_gives_the_whole_photo_back(
    source: tuple[int, int]
) -> None:
    """Its name is the photo's own frame, so that is what it must give."""
    square = geometry.with_aspect(Geometry(), source, geometry.ASPECT_SQUARE)
    assert not square.crop.is_whole_frame

    back = geometry.with_aspect(square, source, geometry.ASPECT_ORIGINAL)
    assert back.crop.is_whole_frame
    assert back.is_default


def test_choosing_original_after_a_tight_crop_opens_it_up_again() -> None:
    source = (4000, 6000)
    tight = geometry.with_crop(
        Geometry(aspect=geometry.ASPECT_FREE), source, Rect(0.6, 0.6, 0.2, 0.2)
    )

    back = geometry.with_aspect(tight, source, geometry.ASPECT_ORIGINAL)

    assert back.crop.is_whole_frame


def test_original_is_the_photo_s_own_way_round(source: tuple[int, int] = (4000, 6000)) -> None:
    back = geometry.with_aspect(Geometry(), source, geometry.ASPECT_ORIGINAL)
    assert back.orientation == geometry.ORIENTATION_PORTRAIT


def test_original_on_a_straightened_photo_holds_only_picture() -> None:
    """The whole frame no longer fits once the photo is turned a little."""
    source = (4000, 6000)
    tilted = geometry.with_aspect(Geometry(angle=5.0), source, geometry.ASPECT_ORIGINAL)

    assert not tilted.crop.is_whole_frame
    assert geometry.is_inside(tilted.crop, source, 5.0)
    # The crop is measured against the rotated frame, which is larger than
    # the photo, so that is what its shape has to be worked out against.
    frame = geometry.frame_size(source, tilted)
    shape = (tilted.crop.w * frame[0]) / (tilted.crop.h * frame[1])
    assert shape == pytest.approx(source[0] / source[1], rel=0.01)


def test_the_other_ratios_still_reshape_what_is_there() -> None:
    """Only "original" opens the box up; 1:1 keeps the crop you had."""
    source = (4000, 6000)
    tight = geometry.with_crop(
        Geometry(aspect=geometry.ASPECT_FREE), source, Rect(0.1, 0.1, 0.3, 0.3)
    )

    square = geometry.with_aspect(tight, source, geometry.ASPECT_SQUARE)

    assert not square.crop.is_whole_frame
    assert square.crop.w * source[0] == pytest.approx(square.crop.h * source[1], rel=0.01)


def test_9_16_is_upright_and_turns_into_16_9() -> None:
    """9:16 is written nine wide and sixteen high, so it starts upright, and
    X gives back the landscape shape it is the counterpart of."""
    source = (6000, 4000)

    assert geometry.written_ratio(geometry.ASPECT_9_16) == pytest.approx(9 / 16)
    assert (
        geometry.natural_orientation(geometry.ASPECT_9_16)
        == geometry.ORIENTATION_PORTRAIT
    )

    upright = geometry.with_aspect(Geometry(), source, geometry.ASPECT_9_16)
    assert upright.crop.w * source[0] == pytest.approx(
        upright.crop.h * source[1] * 9 / 16, rel=0.01
    )

    flipped = geometry.with_orientation_flipped(upright, source)
    assert geometry.aspect_ratio(
        flipped.aspect, flipped.orientation, source
    ) == pytest.approx(16 / 9)


def test_9_16_is_kept_when_a_photo_is_opened_again() -> None:
    """It is a stored ratio like any other (PLAN.md 7.1)."""
    stored = Geometry(aspect=geometry.ASPECT_9_16).to_dict()

    assert Geometry.from_dict(stored).aspect == geometry.ASPECT_9_16


@pytest.mark.parametrize(
    ("wide", "tall"),
    [("5:4", "4:5"), ("3:2", "2:3"), ("16:9", "9:16"), ("21:9", "9:21")],
)
def test_x_turns_a_ratio_into_its_twin(wide: str, tall: str) -> None:
    """Each way round has a name of its own, and X goes from one to the other."""
    source = (6000, 4000)
    chosen = geometry.with_aspect(Geometry(), source, wide)
    assert chosen.orientation == geometry.ORIENTATION_LANDSCAPE

    turned = geometry.with_orientation_flipped(chosen, source)
    assert turned.aspect == tall
    assert turned.orientation == geometry.ORIENTATION_PORTRAIT

    back = geometry.with_orientation_flipped(turned, source)
    assert back.aspect == wide
    assert back.crop.to_dict() == pytest.approx(chosen.crop.to_dict())


def test_21_9_is_wide() -> None:
    source = (6000, 4000)
    wide = geometry.with_aspect(Geometry(), source, geometry.ASPECT_21_9)

    assert wide.crop.w * source[0] == pytest.approx(
        wide.crop.h * source[1] * 21 / 9, rel=0.01
    )
    assert wide.crop.w == pytest.approx(1.0)


def test_a_quarter_turn_names_the_ratio_the_new_way_round() -> None:
    """16:9 on a photo turned a quarter is an upright crop, so 9:16."""
    source = (6000, 4000)
    wide = geometry.with_aspect(Geometry(), source, geometry.ASPECT_16_9)

    assert wide.turned(1).aspect == geometry.ASPECT_9_16


@pytest.mark.parametrize(
    ("stored", "orientation", "named"),
    [
        ("4:5", "landscape", "5:4"),
        ("3:2", "portrait", "2:3"),
        ("16:9", "portrait", "9:16"),
        ("9:16", "landscape", "16:9"),
        ("4:5", "portrait", "4:5"),
    ],
)
def test_an_older_sidecar_lights_the_right_button(
    stored: str, orientation: str, named: str
) -> None:
    """Before each way round had a name, 4:5 on its side was stored as 4:5
    lying down; it is read as 5:4, the same shape (PLAN.md 7.1)."""
    read = Geometry.from_dict({"aspect": stored, "orientation": orientation})

    assert read.aspect == named
    assert read.orientation == orientation


# -- turning the crop on its side (X, PLAN.md 3.2) -----------------------


@pytest.mark.parametrize("source", [(4000, 6000), (6024, 4024)])
@pytest.mark.parametrize("angle", [0.0, 5.0])
@pytest.mark.parametrize("aspect", [geometry.ASPECT_ORIGINAL, geometry.ASPECT_4_5])
def test_flipping_twice_gives_back_exactly_what_was_there(
    source: tuple[int, int], angle: float, aspect: str
) -> None:
    """PLAN.md 3.2 flips on X and on a second click of the chosen ratio, so
    doing it again and again must not walk the box across the picture."""
    start = geometry.with_aspect(Geometry(angle=angle), source, aspect)

    once = geometry.with_orientation_flipped(start, source)
    twice = geometry.with_orientation_flipped(once, source)

    assert once.crop != start.crop
    assert twice.crop.x == pytest.approx(start.crop.x, abs=1e-6)
    assert twice.crop.y == pytest.approx(start.crop.y, abs=1e-6)
    assert twice.crop.w == pytest.approx(start.crop.w, abs=1e-6)
    assert twice.crop.h == pytest.approx(start.crop.h, abs=1e-6)
    assert twice.orientation == start.orientation


def test_flipping_a_box_the_user_sized_keeps_its_size() -> None:
    source = (4000, 6000)
    sized = geometry.with_crop(
        geometry.with_aspect(Geometry(), source, geometry.ASPECT_4_5),
        source,
        Rect(0.2, 0.2, 0.3, 0.3),
    )

    flipped = geometry.with_orientation_flipped(sized, source)

    assert flipped.crop.w * flipped.crop.h == pytest.approx(
        sized.crop.w * sized.crop.h, rel=0.02
    )
    assert flipped.crop.centre == pytest.approx(sized.crop.centre, abs=1e-6)


def test_flipping_a_box_that_fills_the_picture_fills_it_the_other_way() -> None:
    source = (4000, 6000)
    whole = geometry.with_aspect(Geometry(), source, geometry.ASPECT_ORIGINAL)

    flipped = geometry.with_orientation_flipped(whole, source)

    assert flipped.crop.w == pytest.approx(1.0)
    assert flipped.crop.h == pytest.approx(source[0] / source[1] * source[0] / source[1], rel=0.01)
    assert geometry.is_inside(flipped.crop, source, 0.0)


def test_a_free_crop_has_no_side_to_stand_on() -> None:
    source = (4000, 6000)
    free = geometry.with_aspect(Geometry(), source, geometry.ASPECT_FREE)

    flipped = geometry.with_orientation_flipped(free, source)

    assert flipped.crop == free.crop


def test_clicking_the_chosen_ratio_again_keeps_flipping_between_two_boxes() -> None:
    """What the user sees when pressing the same button over and over."""
    source = (4000, 6000)
    geometry_now = geometry.with_aspect(Geometry(), source, geometry.ASPECT_ORIGINAL)

    seen = []
    for _ in range(6):
        geometry_now = geometry.with_orientation_flipped(geometry_now, source)
        seen.append((round(geometry_now.crop.w, 4), round(geometry_now.crop.h, 4)))

    assert len(set(seen)) == 2, f"the box should only ever be one of two: {seen}"


# -- the window on screen (PLAN.md 3.3 and section 4) ----------------------


def test_the_whole_frame_is_a_window_like_any_other() -> None:
    window = geometry.Window.whole((6000, 4000))

    assert window.is_whole
    assert window.size == (6000, 4000)
    assert (window.left, window.top, window.right, window.bottom) == (0, 0, 6000, 4000)


def test_a_window_hanging_over_the_edge_is_brought_inside() -> None:
    """What a photo panned to its corner asks for."""
    window = geometry.Window.around((600, 400), (-120.0, -80.0, 300.5, 410.0))

    assert (window.left, window.top) == (0, 0)
    assert (window.right, window.bottom) == (301, 400)


def test_a_window_is_never_empty() -> None:
    window = geometry.Window.around((600, 400), (599.6, 399.6, 599.7, 399.8))

    assert window.size == (1, 1)
    assert window.right <= 600 and window.bottom <= 400


def test_a_window_grows_by_the_margin_and_stops_at_the_frame() -> None:
    window = geometry.Window.around((600, 400), (100.0, 50.0, 200.0, 150.0))

    grown = window.grown(40)

    assert (grown.left, grown.top, grown.right, grown.bottom) == (60, 10, 240, 190)
    assert window.grown(0) is window
    assert geometry.Window.around((600, 400), (10.0, 10.0, 20.0, 20.0)).grown(80).left == 0


def test_a_window_knows_where_it_sits_inside_the_one_around_it() -> None:
    inner = geometry.Window.around((600, 400), (100.0, 50.0, 200.0, 150.0))
    outer = inner.grown(12)

    assert inner.inside(outer) == (12, 12, 112, 112)

    developed = np.zeros((outer.size[1], outer.size[0], 3), dtype=np.uint8)
    developed[12:112, 12:112] = 7

    trimmed = outer.trim(developed, inner)

    assert trimmed.shape == (100, 100, 3)
    assert np.all(trimmed == 7)


def test_a_window_scaled_down_keeps_its_place_in_the_picture() -> None:
    window = geometry.Window.around((6000, 4000), (3000.0, 1000.0, 4200.0, 2000.0))

    smaller = window.scaled(0.5)

    assert smaller.frame == (3000, 2000)
    assert (smaller.left, smaller.top, smaller.right, smaller.bottom) == (
        1500,
        500,
        2100,
        1000,
    )


def test_a_window_of_a_photo_holds_the_pixels_of_that_part() -> None:
    photo = _photo(width=60, height=40, encoded=True)
    window = geometry.Window((60, 40), left=10, top=5, right=30, bottom=25)

    part = geometry.window_of(photo, window)

    assert part.size == (20, 20)
    assert np.array_equal(part.image, photo.image[5:25, 10:30])
    assert part.encoded is not None
    assert part.encoded.shape == (20, 20, 3)
    assert part.clipped.shape == (20, 20)


def test_a_window_of_the_whole_frame_at_its_own_size_is_the_photo_itself() -> None:
    photo = _photo()

    assert geometry.window_of(photo, geometry.Window.whole(photo.size)) is photo


def test_a_window_cut_from_a_proxy_comes_back_at_the_size_asked_for() -> None:
    """The proxy is stretched over the window until the full size one arrives."""
    proxy = _photo(width=60, height=40)
    window = geometry.Window((120, 80), left=20, top=10, right=60, bottom=50)

    part = geometry.window_of(proxy, window)

    assert part.size == window.size == (40, 40)
    # The middle of the window is the middle of the same place in the proxy.
    assert part.image[20, 20, 0] == pytest.approx(proxy.image[15, 20, 0], abs=0.05)


def test_a_window_of_a_photo_carries_its_colour_and_its_camera() -> None:
    photo = DecodedPhoto(
        image=np.full((40, 60, 3), 0.4, dtype=np.float32),
        is_raw=True,
        clipped=np.zeros((40, 60), dtype=bool),
        camera_to_xyz=np.eye(3),
        rgb_xyz_matrix=np.eye(3),
        camera_whitebalance=(2.0, 1.0, 1.5),
    )

    part = geometry.window_of(photo, geometry.Window((60, 40), 10, 10, 30, 30))

    assert part.is_raw
    assert part.camera_whitebalance == (2.0, 1.0, 1.5)
    assert part.rgb_xyz_matrix is not None
    assert part.full_size == part.size


def test_a_proxy_says_what_it_was_shrunk_from_through_a_crop() -> None:
    """Zooming in needs the full size of what is on screen, crop included."""
    proxy = DecodedPhoto(
        image=np.zeros((400, 600, 3), dtype=np.float32),
        is_raw=False,
        clipped=np.zeros((400, 600), dtype=bool),
        shrunk_from=(6000, 4000),
    )
    half = geometry.with_crop(Geometry(), (6000, 4000), Rect(0.0, 0.0, 0.5, 0.5))

    framed = geometry.apply(proxy, half)

    assert framed.size == (300, 200)
    assert framed.full_size == (3000, 2000)


def test_a_window_takes_its_part_of_any_whole_frame_array() -> None:
    """Step 7's local brightness is cut from the frame the same way."""
    field = np.arange(40 * 60, dtype=np.float32).reshape(40, 60)
    window = geometry.Window((60, 40), left=10, top=5, right=30, bottom=25)

    assert np.array_equal(window.take(field), field[5:25, 10:30])


def test_a_window_stretches_what_it_takes_to_its_own_size() -> None:
    """The field is worked out on the proxy and used at full resolution."""
    field = np.linspace(0.0, 1.0, 30 * 20, dtype=np.float32).reshape(20, 30)
    window = geometry.Window((120, 80), left=40, top=20, right=80, bottom=60)

    taken = window.take(field)

    assert taken.shape == (40, 40)
    assert float(taken.min()) >= float(field.min())
    assert float(taken.max()) <= float(field.max())


def test_a_window_of_the_whole_frame_at_its_own_size_takes_nothing_new() -> None:
    field = np.zeros((40, 60), dtype=np.float32)
    assert geometry.Window.whole((60, 40)).take(field) is field


# -- flipping left to right and top to bottom (C-08) -----------------------


def _textured(width: int = 64, height: int = 40) -> DecodedPhoto:
    """Return a photo with no symmetry, smooth enough to rotate cleanly."""
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.stack(
        [x + 0.0 * y, y + 0.0 * x, 0.5 + 0.25 * np.sin(3.0 * x + 5.0 * y)], axis=2
    ).astype(np.float32)
    clipped = np.zeros((height, width), dtype=bool)
    clipped[0, 0] = True
    return DecodedPhoto(image=image, is_raw=True, clipped=clipped)


def _seen(photo: DecodedPhoto, shape: Geometry) -> np.ndarray:
    """Return what the crop tab shows: turned, mirrored and rotated, uncut."""
    return geometry.apply(photo, shape, crop=False).image


STANDINGS = (
    Geometry(),
    Geometry(quarter_turns=1),
    Geometry(quarter_turns=3, angle=7.5),
    Geometry(quarter_turns=2, angle=-12.0, mirrored=True),
)


@pytest.mark.parametrize("before", STANDINGS)
def test_flipping_left_to_right_mirrors_what_is_on_screen(before: Geometry) -> None:
    photo = _textured()

    after = geometry.with_mirror(before)

    assert np.allclose(_seen(photo, after), _seen(photo, before)[:, ::-1], atol=1e-4)


@pytest.mark.parametrize("before", STANDINGS)
def test_flipping_top_to_bottom_mirrors_what_is_on_screen(before: Geometry) -> None:
    photo = _textured()

    after = geometry.with_mirror(before, vertical=True)

    assert np.allclose(_seen(photo, after), _seen(photo, before)[::-1, :], atol=1e-4)


def test_a_flip_without_rotation_is_exact() -> None:
    photo = _textured()

    flipped = geometry.apply(photo, Geometry(mirrored=True))

    assert np.array_equal(flipped.image, photo.image[:, ::-1])
    assert np.array_equal(flipped.clipped, photo.clipped[:, ::-1])
    assert flipped.image.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("before", STANDINGS)
def test_flipping_twice_gives_everything_back(before: Geometry, vertical: bool) -> None:
    before = replace_crop(before, Rect(0.1, 0.2, 0.3, 0.4))

    twice = geometry.with_mirror(geometry.with_mirror(before, vertical), vertical)

    assert twice.quarter_turns == before.quarter_turns
    assert twice.mirrored == before.mirrored
    assert twice.angle == before.angle
    assert twice.crop.x == pytest.approx(before.crop.x)
    assert twice.crop.y == pytest.approx(before.crop.y)
    assert twice.crop.w == pytest.approx(before.crop.w)
    assert twice.crop.h == pytest.approx(before.crop.h)


def replace_crop(shape: Geometry, crop: Rect) -> Geometry:
    """Return *shape* with another crop box."""
    return Geometry(
        quarter_turns=shape.quarter_turns,
        angle=shape.angle,
        crop=crop,
        aspect=shape.aspect,
        orientation=shape.orientation,
        mirrored=shape.mirrored,
    )


def test_the_crop_box_goes_over_with_the_picture() -> None:
    before = Geometry(crop=Rect(0.0, 0.0, 0.4, 0.2))

    sideways = geometry.with_mirror(before)
    upside_down = geometry.with_mirror(before, vertical=True)

    assert sideways.crop.x == pytest.approx(0.6)
    assert sideways.crop.y == pytest.approx(0.0)
    assert upside_down.crop.x == pytest.approx(0.0)
    assert upside_down.crop.y == pytest.approx(0.8)
    assert (sideways.crop.w, sideways.crop.h) == (before.crop.w, before.crop.h)


def test_the_cropped_result_is_the_mirror_of_the_cropped_photo() -> None:
    photo = _textured(width=100, height=80)
    before = Geometry(quarter_turns=1, crop=Rect(0.1, 0.25, 0.5, 0.5))

    after = geometry.with_mirror(before)

    assert np.array_equal(
        geometry.apply(photo, after).image, geometry.apply(photo, before).image[:, ::-1]
    )


def test_a_flip_keeps_the_crop_on_the_picture() -> None:
    turned = geometry.turned_size(SOURCE, 1)
    before = geometry.with_angle(Geometry(quarter_turns=1), SOURCE, 20.0)
    before = geometry.with_crop(before, SOURCE, Rect(0.1, 0.1, 0.3, 0.3))

    for vertical in (False, True):
        after = geometry.with_mirror(before, vertical)
        assert geometry.is_inside(after.crop, turned, after.angle)


def test_a_flip_turns_the_angle_the_other_way() -> None:
    assert geometry.with_mirror(Geometry(angle=3.5)).angle == -3.5
    level = geometry.with_mirror(Geometry())
    assert level.angle == 0.0
    assert math.copysign(1.0, level.angle) == 1.0


@pytest.mark.parametrize("turn", [-1, 1])
def test_a_flipped_photo_still_turns_the_way_the_button_says(turn: int) -> None:
    """Turning right after a flip turns the flipped picture right."""
    photo = _textured()
    flipped = geometry.with_mirror(Geometry(quarter_turns=1))

    turned = geometry.with_quarter_turn(flipped, turn)

    assert np.array_equal(_seen(photo, turned), np.rot90(_seen(photo, flipped), k=-turn))


def test_both_flips_are_half_a_turn() -> None:
    both = geometry.with_mirror(geometry.with_mirror(Geometry()), vertical=True)

    assert not both.mirrored
    assert both.quarter_turns == 2


def test_a_flipped_photo_is_not_untouched() -> None:
    assert not Geometry(mirrored=True).is_default
    assert geometry.apply(_textured(), Geometry(mirrored=True), crop=False) is not None


def test_a_flip_is_stored_and_read_back() -> None:
    flipped = Geometry(quarter_turns=2, mirrored=True, angle=-1.5)

    assert flipped.to_dict()["mirrored"] is True
    assert Geometry.from_dict(flipped.to_dict()) == flipped


def test_an_older_or_odd_file_reads_as_not_flipped() -> None:
    assert Geometry.from_dict({"quarter_turns": 1}).mirrored is False
    assert Geometry.from_dict({"mirrored": "yes"}).mirrored is False
    assert Geometry.from_dict({"mirrored": 1}).mirrored is False
