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



def _shaded_photo():
    """The test person with a band of shade across the chest: darker, bluer, greener."""
    from tests.test_whiten import _person_area

    photo = _photo()
    image = photo.image.copy()
    shade = np.zeros(image.shape[:2], dtype=bool)
    shade[490:520] = True  # about a third of the face's size (S = 80 px)
    shade &= _person_area()
    image[shade] *= np.array([0.30, 0.35, 0.40], dtype=np.float32)
    # A grey slab in the same shade, against the shaded skin.
    image[490:520, 330:400] = np.float32(0.05)
    return replace(photo, image=image)


def test_a_stroke_that_keeps_to_skin_takes_in_skin_in_shade() -> None:
    """2026-11-07: shade turned skin too blue for the brush, which left it out."""
    from core import faces

    photo = _shaded_photo()
    x, y = 150, 505
    # The colour of the cheeks alone does not take it in.
    tones = [face.tone for face in photo.faces]
    strict = faces.body_likeness(*faces.colour_features(photo.image), tones)
    assert strict[y, x] < 0.5

    stroke = SkinStroke(points=(_at(x, y),), radius=0.03, adding=True, snap=True)
    skin = _skin(photo, SkinRange(auto=False, strokes=(stroke,)))
    assert skin[y, x] > 0.9


def test_grey_in_the_same_shade_is_still_left_out() -> None:
    photo = _shaded_photo()
    stroke = SkinStroke(points=(_at(330, 505),), radius=0.05, adding=True, snap=True)
    skin = _skin(photo, SkinRange(auto=False, strokes=(stroke,)))
    assert skin[505, 345] < 0.1  # the grey slab
    assert skin[505, 310] > 0.9  # the shaded skin beside it


def _leaf_lit_photo():
    """The test person with a broad band of light through leaves across the chest.

    A stop darker and 0.9 stops less blue at its middle, fading in over 40 px
    on both sides, so that its middle lies well beyond the reach of shade
    from lit skin.
    """
    from tests.test_whiten import _person_area

    photo = _photo()
    rows = np.arange(HEIGHT, dtype=np.float32)
    strength = np.clip(np.minimum(rows - 460, 580 - rows) / 40.0, 0.0, 1.0)[:, None, None]
    tint = np.array([0.45, 0.42, 0.22], dtype=np.float32)
    shaded = photo.image * (1 + strength * (tint - 1))
    image = np.where(_person_area()[..., None], shaded, photo.image).astype(np.float32)
    return replace(photo, image=image)


def test_the_brush_takes_in_yellow_shade_where_its_centre_goes() -> None:
    """2026-11-07: the side of a waist lit through leaves stayed out of the brush."""
    photo = _leaf_lit_photo()
    over = SkinStroke(points=(_at(150, 520),), radius=0.03, adding=True, snap=True)
    skin = _skin(photo, SkinRange(auto=False, strokes=(over,)))
    assert skin[520, 150] > 0.9


def test_far_shade_is_left_out_where_the_centre_stays_on_lit_skin() -> None:
    photo = _leaf_lit_photo()
    # A wide brush over lit skin whose edge reaches the middle of the band.
    beside = SkinStroke(points=(_at(150, 440),), radius=0.12, adding=True, snap=True)
    skin = _skin(photo, SkinRange(auto=False, strokes=(beside,)))
    assert skin[440, 150] > 0.9
    assert skin[520, 150] < 0.1
