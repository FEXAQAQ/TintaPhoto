"""Tests for the skin range the portrait brush shapes (PLAN.md 2.10)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core import geometry, pipeline, skin_range
from core.edit_state import Edit
from core.geometry import Geometry, Window
from core.skin_range import SkinRange, SkinStroke
from tests.test_whiten import APART, CHEEK, CHEST, JPEG, SIZE, _photo

WIDTH, HEIGHT = SIZE


def _at(x: float, y: float) -> tuple[float, float]:
    """Return pixel (x, y) of the test photo as fractions."""
    return ((x + 0.5) / WIDTH, (y + 0.5) / HEIGHT)


def _skin(photo, skin: SkinRange) -> np.ndarray:
    """Return the skin whitening works on, with *skin* as the range."""
    framed = geometry.apply(photo, Geometry(skin=skin))
    field = pipeline.body_field(framed)
    if field is None:
        return np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    planes = Window.whole(framed.size).take(field)
    return np.maximum(planes[..., 0] * planes[..., 1], planes[..., 2])


ADD_APART = SkinStroke(points=(_at(*APART),), radius=0.04, adding=True, snap=False)
REMOVE_CHEST = SkinStroke(points=(_at(*CHEST),), radius=0.04, adding=False)


# -- storing ---------------------------------------------------------------


def test_a_range_is_stored_and_read_back() -> None:
    skin = SkinRange(auto=False, strokes=(ADD_APART, REMOVE_CHEST))
    assert SkinRange.from_dict(skin.to_dict()) == skin
    shape = Geometry(skin=skin)
    assert Geometry.from_dict(shape.to_dict()).skin == skin


def test_odd_stored_ranges_fall_back() -> None:
    assert SkinRange.from_dict(None) == SkinRange()
    assert SkinRange.from_dict({"strokes": [{"points": [], "radius": 0.1, "mode": "add"}]}) == (
        SkinRange()
    )
    read = SkinRange.from_dict(
        {"strokes": [{"points": [[2.0, -1.0]], "radius": 9.0, "mode": "remove"}]}
    )
    assert read.strokes[0].points == ((1.0, 0.0),)
    assert read.strokes[0].radius == skin_range.RADIUS_MAX
    assert not read.strokes[0].adding


def test_a_painted_range_counts_as_a_change() -> None:
    reference = Edit(adjustments=JPEG)
    painted = Edit(adjustments=JPEG, geometry=Geometry(skin=SkinRange(strokes=(ADD_APART,))))
    assert reference.is_untouched(reference)
    assert not painted.is_untouched(reference)
    assert not Edit(adjustments=JPEG, geometry=Geometry(skin=SkinRange(auto=False))).is_untouched(
        reference
    )


# -- the strokes -------------------------------------------------------------


def test_adding_takes_in_skin_the_range_left_out() -> None:
    photo = _photo()
    x, y = APART
    assert _skin(photo, SkinRange())[y, x] < 0.05
    assert _skin(photo, SkinRange(strokes=(ADD_APART,)))[y, x] > 0.9


def test_taking_away_leaves_out_skin_that_was_found() -> None:
    photo = _photo()
    x, y = CHEST
    assert _skin(photo, SkinRange())[y, x] > 0.9
    assert _skin(photo, SkinRange(strokes=(REMOVE_CHEST,)))[y, x] < 0.05


def test_a_later_stroke_wins() -> None:
    photo = _photo()
    x, y = CHEST
    add_chest = replace(REMOVE_CHEST, adding=True, snap=False)
    assert _skin(photo, SkinRange(strokes=(REMOVE_CHEST, add_chest)))[y, x] > 0.9
    assert _skin(photo, SkinRange(strokes=(add_chest, REMOVE_CHEST)))[y, x] < 0.05


def test_without_finding_only_the_painted_skin_is_worked_on() -> None:
    photo = _photo()
    skin = _skin(photo, SkinRange(auto=False, strokes=(ADD_APART,)))
    x, y = APART
    assert skin[y, x] > 0.9
    assert skin[CHEEK[1], CHEEK[0]] == 0.0
    assert skin[CHEST[1], CHEST[0]] == 0.0


def test_a_stroke_that_keeps_to_skin_leaves_the_background_out() -> None:
    photo = _photo()
    # Across the edge of the skin coloured patch, half on the blue background.
    stroke = SkinStroke(points=(_at(330, 60),), radius=0.06, adding=True, snap=True)
    rough = replace(stroke, snap=False)
    kept = _skin(photo, SkinRange(auto=False, strokes=(stroke,)))
    loose = _skin(photo, SkinRange(auto=False, strokes=(rough,)))
    assert kept[60, 345] > 0.9  # on the patch
    assert loose[60, 312] > 0.9  # the background, under a stroke that does not keep to skin
    assert kept[60, 312] < 0.1


def test_a_stroke_keeps_to_the_colour_it_was_painted_over_without_faces() -> None:
    photo = _photo(with_faces=False)
    stroke = SkinStroke(points=(_at(345, 65),), radius=0.06, adding=True, snap=True)
    kept = _skin(photo, SkinRange(auto=False, strokes=(stroke,)))
    assert kept[65, 345] > 0.9
    assert kept[65, 318] < 0.1


# -- the sliders -------------------------------------------------------------


def test_whitening_reaches_skin_painted_where_no_face_was_found() -> None:
    photo = _photo(with_faces=False)
    framed = geometry.apply(photo, Geometry(skin=SkinRange(strokes=(ADD_APART,))))
    state = JPEG.with_value("whiten", 100.0)
    assert pipeline.has_whiten(state, framed)
    before = pipeline.develop(framed, JPEG)
    after = pipeline.develop(framed, state)
    x, y = APART
    assert after[y, x].astype(int).sum() > before[y, x].astype(int).sum() + 6
    assert np.array_equal(after[30, 30], before[30, 30])


def test_with_nothing_to_work_on_the_sliders_change_nothing() -> None:
    photo = _photo()
    off = geometry.apply(photo, Geometry(skin=SkinRange(auto=False)))
    state = JPEG.with_value("whiten", 100.0).with_value("skin", 100.0)
    assert not pipeline.has_whiten(state, off)
    assert not pipeline.has_skin(state, off)
    assert np.array_equal(pipeline.develop(off, state), pipeline.develop(off, JPEG))


def test_the_range_stays_on_the_photo_when_it_is_turned() -> None:
    photo = _photo()
    skin = SkinRange(auto=False, strokes=(ADD_APART,))
    shape = Geometry(skin=skin).turned(1)
    framed = geometry.apply(photo, shape)
    assert framed.skin == geometry.framed_skin(shape, photo.size)
    field = pipeline.body_field(framed)
    assert field is not None
    planes = Window.whole(framed.size).take(field)
    shown_x, shown_y = geometry.shown_point(_at(*APART), photo.size, shape)
    column = int(shown_x * framed.size[0])
    row = int(shown_y * framed.size[1])
    assert planes[row, column, skin_range.ADDED] > 0.9


def test_reaches_nothing() -> None:
    faces = _photo().faces
    assert not skin_range.reaches_nothing(faces, None)
    assert skin_range.reaches_nothing((), None)
    assert skin_range.reaches_nothing(faces, SkinRange(auto=False))
    assert not skin_range.reaches_nothing((), SkinRange(strokes=(ADD_APART,)))
    assert skin_range.reaches_nothing((), SkinRange(strokes=(REMOVE_CHEST,)))


def test_thinned_paths_keep_their_ends() -> None:
    points = [(0.1 + index * 0.0001, 0.5) for index in range(100)]
    kept = skin_range.thinned(points, 0.03, (1000, 1000))
    assert kept[0] == points[0]
    assert kept[-1] == points[-1]
    assert len(kept) < 20

