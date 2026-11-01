"""Tests for skin enhancement (PLAN.md 2.2 A-29, section 6 steps 2.6 and 10)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from core import color_management as cm
from core import exporter, face_detector, faces, geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import SKIN_MAX, EditState
from core.exporter import ExportOptions
from core.faces import Face
from core.geometry import Geometry, Rect, Window

JPEG = EditState(is_raw=False)
SIZE = (400, 360)  # width, height
SKIN = (205, 150, 120)
BACKGROUND = (70, 110, 150)
#: A blemish on the right cheek, out where colour decides what is skin.
PIMPLE = (269, 204)
#: Two eyes, the nose, two mouth corners and the middle of the face's box.
POINTS = ((160.0, 160.0), (240.0, 160.0), (200.0, 200.0), (172.0, 240.0), (228.0, 240.0), (200.0, 190.0))


def _linear(levels: tuple[int, int, int]) -> np.ndarray:
    return cm.srgb_to_linear(np.array(levels, dtype=np.float32) / np.float32(255.0))


def _picture(blemish: bool = False, pimple: bool = False) -> np.ndarray:
    """Return a face: an oval of skin on a blue background, with dark eyes."""
    width, height = SIZE
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = _linear(BACKGROUND)
    oval = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(oval, (200, 195), (95, 130), 0, 0, 360, 1, -1)
    image[oval > 0] = _linear(SKIN)
    for x, y in POINTS[:2]:
        cv2.circle(image, (int(x), int(y)), 9, (0.01, 0.01, 0.01), -1)
    if blemish:
        # A faint dark spot on the left cheek, a few pixels across.
        cv2.circle(image, (150, 215), 3, tuple(float(v) * 0.8 for v in _linear(SKIN)), -1)
    if pimple:
        # A distinct red blemish, far too strong to be faint detail.
        cv2.circle(image, PIMPLE, 3, tuple(float(v) for v in _linear((170, 85, 70))), -1)
    return image


def _photo(blemish: bool = False, with_faces: bool = True, pimple: bool = False) -> DecodedPhoto:
    image = _picture(blemish, pimple)
    found: tuple[Face, ...] | None = None
    if with_faces:
        points = np.array(POINTS, dtype=np.float64)
        tone = faces.measure_tone(image, points)
        assert tone is not None
        found = (Face(points=(), tone=tone).moved(points, SIZE),)
    return DecodedPhoto(
        image=image,
        is_raw=False,
        clipped=np.zeros(image.shape[:2], dtype=bool),
        color_space=ColorSpace.SRGB,
        faces=found,
    )


def _mask(photo: DecodedPhoto) -> np.ndarray:
    placed = faces.place(photo.faces or (), photo.size)
    mask = faces.skin_mask(photo.image, placed)
    assert mask is not None
    return mask


# -- the slider ------------------------------------------------------------


def test_the_slider_runs_from_nothing_to_a_hundred() -> None:
    assert JPEG.with_value("skin", -5.0).skin == 0.0
    assert JPEG.with_value("skin", 500.0).skin == SKIN_MAX == 100.0


def test_the_slider_counts_as_a_change() -> None:
    assert not JPEG.with_value("skin", 30.0).is_untouched(JPEG)


def test_the_slider_is_read_back_from_a_parameter_file() -> None:
    state = JPEG.with_value("skin", 42.0)
    assert EditState.from_dict(state.to_dict(), JPEG).skin == 42.0
    assert EditState.from_dict({}, JPEG).skin == 0.0


# -- faces and where they lie ----------------------------------------------


def test_a_face_lies_in_the_same_place_at_every_resolution() -> None:
    face = _photo().faces[0]
    small = face.in_pixels((200, 180))
    large = face.in_pixels(SIZE)
    np.testing.assert_allclose((small + 0.5) * 2.0 - 0.5, large, atol=1e-9)


def test_the_size_of_a_face_is_the_eyes_to_the_mouth() -> None:
    assert faces.face_size(_photo().faces[0], SIZE) == pytest.approx(80.0)


@pytest.mark.parametrize(
    "shape",
    [
        Geometry(quarter_turns=1),
        Geometry(quarter_turns=3, mirrored=True),
        Geometry(angle=8.0, crop=Rect(0.1, 0.1, 0.8, 0.8)),
        Geometry(crop=Rect(0.2, 0.15, 0.7, 0.7)),
    ],
)
def test_the_faces_move_with_the_photo(shape: Geometry) -> None:
    """A marked pixel ends up where the geometry step says the eye went."""
    photo = _photo()
    marked = photo.image.copy()
    marked[:] = 0.0
    eye = POINTS[0]
    cv2.circle(marked, (int(eye[0]), int(eye[1])), 3, (1.0, 1.0, 1.0), -1)
    moved = geometry.apply(replace(photo, image=marked), shape)
    assert moved.faces is not None
    x, y = moved.faces[0].in_pixels(moved.size)[0]
    weight = moved.image[..., 0]
    rows, columns = np.indices(weight.shape)
    assert abs((weight * columns).sum() / weight.sum() - x) <= 1.0
    assert abs((weight * rows).sum() / weight.sum() - y) <= 1.0


# -- the mask --------------------------------------------------------------


def test_the_mask_holds_the_skin_and_leaves_the_background() -> None:
    mask = _mask(_photo())
    assert mask[215, 150] > 0.9  # a cheek
    assert mask[110, 200] > 0.9  # the forehead
    assert mask[30, 30] == 0.0  # far away
    assert mask[195, 300] < 0.05  # the background just beside the face


def test_the_eyes_and_the_mouth_are_left_out() -> None:
    mask = _mask(_photo())
    for x, y in POINTS[:2]:
        assert mask[int(y), int(x)] < 0.1
    assert mask[240, 200] < 0.1


def test_a_blemish_leaves_no_hole_in_the_mask() -> None:
    """Its colour is unlike the cheeks, and it is still skin."""
    mask = _mask(_photo(pimple=True))
    assert mask[PIMPLE[1], PIMPLE[0]] > 0.9


def test_the_cheek_above_the_corners_of_the_mouth_is_skin() -> None:
    mask = _mask(_photo())
    assert mask[218, 185] > 0.9
    assert mask[218, 215] > 0.9


def test_there_is_no_mask_without_faces() -> None:
    photo = _photo(with_faces=False)
    assert faces.skin_mask(photo.image, faces.place((), photo.size)) is None


# -- smoothing (step 2.6) ----------------------------------------------------


def test_smoothing_takes_out_a_faint_spot() -> None:
    plain = _photo()
    spotted = _photo(blemish=True)
    placed = faces.place(spotted.faces or (), spotted.size)
    smoothed = pipeline.smooth_skin(spotted.image, placed, _mask(spotted), 100.0, False)
    before = np.abs(spotted.image[215, 150] - plain.image[215, 150]).sum()
    after = np.abs(smoothed[215, 150] - plain.image[215, 150]).sum()
    assert after < 0.6 * before


def test_smoothing_fills_in_a_distinct_blemish() -> None:
    plain = _photo()
    spotted = _photo(pimple=True)
    placed = faces.place(spotted.faces or (), spotted.size)
    smoothed = pipeline.smooth_skin(spotted.image, placed, _mask(spotted), 100.0, False)
    x, y = PIMPLE
    before = np.abs(spotted.image[y, x] - plain.image[y, x]).sum()
    after = np.abs(smoothed[y, x] - plain.image[y, x]).sum()
    assert after < 0.25 * before


def _spots(image: np.ndarray, nostrils=None) -> np.ndarray:
    split = cv2.transform(cv2.sqrt(image), np.asarray(cm.OPPONENT, dtype=np.float32))
    mask = np.ones(image.shape[:2], dtype=np.float32)
    spots = pipeline.skin_spots(split, mask, 80.0, nostrils)
    return np.zeros(image.shape[:2], np.float32) if spots is None else spots


def test_a_spot_is_found_and_a_line_is_not() -> None:
    image = np.empty((200, 200, 3), dtype=np.float32)
    image[:] = _linear(SKIN)
    dark = tuple(float(v) * 0.5 for v in _linear(SKIN))
    cv2.circle(image, (60, 60), 3, dark, -1)
    cv2.line(image, (20, 150), (180, 150), dark, 2)
    spots = _spots(image)
    assert spots[60, 60] > 0.9
    assert spots[145:156, 30:170].max() < 0.05


def test_the_nostrils_are_not_spots() -> None:
    image = np.empty((200, 200, 3), dtype=np.float32)
    image[:] = _linear(SKIN)
    cv2.circle(image, (100, 100), 3, tuple(float(v) * 0.4 for v in _linear(SKIN)), -1)
    assert _spots(image)[100, 100] > 0.9
    assert _spots(image, ((100.0, 100.0), (40.0, 20.0), 0.0))[100, 100] == 0.0


def test_smoothing_leaves_everything_outside_the_mask_exactly_as_it_was() -> None:
    photo = _photo(blemish=True)
    mask = _mask(photo)
    placed = faces.place(photo.faces or (), photo.size)
    smoothed = pipeline.smooth_skin(photo.image, placed, mask, 100.0, False)
    outside = mask == 0.0
    assert np.array_equal(smoothed[outside], photo.image[outside])


def test_smoothing_nothing_changes_nothing() -> None:
    photo = _photo(blemish=True)
    placed = faces.place(photo.faces or (), photo.size)
    assert pipeline.smooth_skin(photo.image, placed, _mask(photo), 0.0, False) is photo.image


# -- tone (step 10) ----------------------------------------------------------


def test_the_skin_is_brighter_and_less_yellow() -> None:
    photo = _photo()
    image = cm.apply_matrix(photo.image, cm.SRGB_TO_PROPHOTO)
    mask = _mask(photo)
    toned = pipeline.skin_tone(image, mask, 100.0)
    before = cm.prophoto_to_oklab(image[215:216, 150:151])[0, 0]
    after = cm.prophoto_to_oklab(toned[215:216, 150:151])[0, 0]
    assert after[0] > before[0] * 1.02
    hue_before = np.degrees(np.arctan2(before[2], before[1]))
    hue_after = np.degrees(np.arctan2(after[2], after[1]))
    assert hue_after < hue_before - 2.0
    outside = mask == 0.0
    assert np.array_equal(toned[outside], image[outside])


# -- the whole chain -------------------------------------------------------


def test_a_photo_without_faces_is_not_changed_by_the_slider() -> None:
    photo = _photo(with_faces=False)
    plain = pipeline.develop(photo, JPEG)
    assert np.array_equal(pipeline.develop(photo, JPEG.with_value("skin", 80.0)), plain)
    nobody = replace(photo, faces=())
    assert np.array_equal(pipeline.develop(nobody, JPEG.with_value("skin", 80.0)), plain)


def test_a_jpeg_without_faces_still_passes_through() -> None:
    photo = replace(_photo(with_faces=False), faces=(), encoded=np.zeros((360, 400, 3), np.uint8))
    assert pipeline.is_passthrough(photo, JPEG.with_value("skin", 50.0))
    with_face = replace(photo, faces=_photo().faces)
    assert not pipeline.is_passthrough(with_face, JPEG.with_value("skin", 50.0))


def test_the_slider_changes_the_face_and_not_the_background() -> None:
    photo = _photo(blemish=True)
    plain = pipeline.develop(photo, JPEG).astype(int)
    skin = pipeline.develop(photo, JPEG.with_value("skin", 100.0)).astype(int)
    assert skin[215, 140].sum() > plain[215, 140].sum() + 6
    assert np.array_equal(skin[:40, :40], plain[:40, :40])


def test_a_window_comes_out_as_the_whole_photo_does() -> None:
    """Zoomed in, a window with its margin matches the whole photo (PLAN.md 3.3)."""
    photo = _photo(blemish=True)
    state = JPEG.with_value("skin", 100.0)
    whole = pipeline.develop(photo, state).astype(int)
    inner = Window.around(SIZE, (120.0, 150.0, 260.0, 260.0))
    outer = inner.grown(pipeline.blur_margin(photo, state, SIZE))
    part = pipeline.develop(geometry.window_of(photo, outer), state, window=outer)
    trimmed = outer.trim(part, inner).astype(int)
    expected = whole[inner.top : inner.bottom, inner.left : inner.right]
    # The mask is drawn on a copy shrunk to the size of the face, cut at
    # slightly different places for a window, which can move a level or so.
    assert np.abs(trimmed - expected).max() <= 3
    assert np.abs(trimmed - expected).mean() < 0.5


# -- finding faces ---------------------------------------------------------


def test_the_model_is_bundled_and_loads() -> None:
    assert face_detector.MODEL_PATH.is_file()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert len(face_detector.find_in_picture(blank)) == 0


def test_a_photo_without_faces_finds_none() -> None:
    photo = _photo(with_faces=False)
    assert face_detector.find(replace(photo, image=np.full_like(photo.image, 0.2))) == ()


def test_faces_are_only_looked_for_when_the_slider_wants_them() -> None:
    photo = _photo(with_faces=False)
    assert face_detector.attach(photo, JPEG, None) is photo
    given = _photo().faces
    attached = face_detector.attach(photo, JPEG.with_value("skin", 50.0), None, given)
    assert attached.faces == given


def test_a_jpeg_without_faces_is_still_copied_when_exported(tmp_path) -> None:
    source = tmp_path / "a.jpg"
    source.write_bytes(b"")
    state = JPEG.with_value("skin", 50.0)
    options = ExportOptions()
    assert exporter.can_copy_original(source, state, None, options, faces=())
    assert not exporter.can_copy_original(source, state, None, options, faces=_photo().faces)
