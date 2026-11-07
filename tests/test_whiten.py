"""Tests for whitening (PLAN.md 2.2 A-30, section 6 step 10)."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core import color_management as cm
from core import exporter, face_detector, faces, geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import WHITEN_MAX, EditState
from core.exporter import ExportOptions
from core.faces import Face
from core.geometry import Window

JPEG = EditState(is_raw=False)
SIZE = (400, 600)  # width, height
SKIN = (205, 150, 120)
BACKGROUND = (70, 110, 150)
#: Two eyes, the nose, two mouth corners and the middle of the face's box.
POINTS = ((160.0, 160.0), (240.0, 160.0), (200.0, 200.0), (172.0, 240.0), (228.0, 240.0), (200.0, 190.0))
#: A cheek, the neck, the chest, and a patch the colour of skin standing
#: apart from the person.
CHEEK = (150, 215)
NECK = (200, 340)
CHEST = (120, 500)
APART = (360, 60)


def _linear(levels: tuple[int, int, int], stops: float = 0.0) -> np.ndarray:
    value = cm.srgb_to_linear(np.array(levels, dtype=np.float32) / np.float32(255.0))
    return value * np.float32(2.0**stops)


def _picture(grey: bool = False) -> np.ndarray:
    """Return a person: a face, a neck and shoulders, on a blue background."""
    width, height = SIZE
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = _linear(BACKGROUND)
    skin = tuple(float(v) for v in _linear(SKIN))
    person = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(person, (200, 195), (95, 130), 0, 0, 360, 1, -1)
    cv2.rectangle(person, (165, 300), (235, 400), 1, -1)
    cv2.ellipse(person, (200, 600), (170, 200), 0, 0, 360, 1, -1)
    image[person > 0] = skin
    # A patch of the same colour that touches nothing of the person.
    cv2.rectangle(image, (330, 20), (390, 110), skin, -1)
    for x, y in POINTS[:2]:
        cv2.circle(image, (int(x), int(y)), 9, (0.01, 0.01, 0.01), -1)
    if grey:
        image[:] = image.mean(axis=2, keepdims=True)
    return image


def _photo(grey: bool = False, with_faces: bool = True) -> DecodedPhoto:
    image = _picture(grey)
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


def _body(photo: DecodedPhoto) -> np.ndarray:
    """Return the skin whitening works on, over the whole photo."""
    field = pipeline.body_field(photo)
    assert field is not None
    planes = Window.whole(photo.size).take(field)
    return np.maximum(planes[..., 0] * planes[..., 1], planes[..., 2])


# -- the slider ------------------------------------------------------------


def test_the_slider_runs_from_nothing_to_a_hundred() -> None:
    assert JPEG.with_value("whiten", -5.0).whiten == 0.0
    assert JPEG.with_value("whiten", 500.0).whiten == WHITEN_MAX == 100.0


def test_the_slider_counts_as_a_change_and_is_kept() -> None:
    state = JPEG.with_value("whiten", 40.0)
    assert not state.is_untouched(JPEG)
    assert state.uses_faces
    assert EditState.from_dict(state.to_dict(), JPEG).whiten == 40.0
    assert EditState.from_dict({}, JPEG).whiten == 0.0


# -- the tone (step 10) ----------------------------------------------------


def _oklab(levels: tuple[int, int, int]) -> np.ndarray:
    return cm.prophoto_to_oklab(
        cm.apply_matrix(_linear(levels)[None, None, :], cm.SRGB_TO_PROPHOTO)
    )[0, 0]


def test_whitening_makes_skin_lighter_and_less_yellow() -> None:
    colour = cm.apply_matrix(_linear(SKIN)[None, None, :], cm.SRGB_TO_PROPHOTO)
    done = pipeline.whiten(colour, np.ones((1, 1), np.float32), 100.0)
    before = cm.prophoto_to_oklab(colour)[0, 0]
    after = cm.prophoto_to_oklab(done)[0, 0]
    assert after[0] > before[0] * 1.06
    assert after[2] < before[2] * 0.9
    # The red is left as it was, so the skin does not go grey.
    assert abs(after[1] - before[1]) < 1e-3


def test_shadows_are_lifted_less_than_lit_skin() -> None:
    lit = cm.apply_matrix(_linear(SKIN)[None, None, :], cm.SRGB_TO_PROPHOTO)
    shadow = lit * np.float32(0.15)
    mask = np.ones((1, 1), np.float32)
    rise = []
    for colour in (lit, shadow):
        before = cm.prophoto_to_oklab(colour)[0, 0, 0]
        after = cm.prophoto_to_oklab(pipeline.whiten(colour, mask, 100.0))[0, 0, 0]
        rise.append(after / before - 1.0)
    assert rise[1] < 0.7 * rise[0]


def test_whitening_leaves_what_the_mask_leaves_out() -> None:
    image = np.random.default_rng(3).random((20, 30, 3), dtype=np.float32)
    mask = np.zeros((20, 30), np.float32)
    mask[5:10, 5:10] = 1.0
    done = pipeline.whiten(image, mask, 100.0)
    outside = mask == 0.0
    assert np.array_equal(done[outside], image[outside])
    assert pipeline.whiten(image, mask, 0.0) is image


# -- the skin beyond the face ----------------------------------------------


def test_the_neck_and_the_chest_are_joined_to_the_face() -> None:
    body = _body(_photo())
    for x, y in (CHEEK, NECK, CHEST):
        assert body[y, x] > 0.9, (x, y)
    assert body[30, 30] == 0.0


def test_skin_coloured_things_apart_from_the_person_are_left_out() -> None:
    body = _body(_photo())
    x, y = APART
    assert body[y, x] < 0.05


def test_a_clear_edge_keeps_a_skin_coloured_wall_out() -> None:
    """A wall of the same colour touching the shoulder, only brighter."""
    photo = _photo()
    image = photo.image.copy()
    wall = np.zeros(image.shape[:2], dtype=bool)
    wall[420:, 360:] = True
    image[wall] = _linear(SKIN, 0.6)
    # The shoulder reaches it.
    image[440:, 300:361] = _linear(SKIN)
    body = _body(replace(photo, image=image))
    assert body[500, 320] > 0.9
    assert body[520, 390] < 0.05


def _person_area() -> np.ndarray:
    """Return where the person's neck, shoulders and chest are."""
    width, height = SIZE
    person = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(person, (165, 300), (235, 400), 1, -1)
    cv2.ellipse(person, (200, 600), (170, 200), 0, 0, 360, 1, -1)
    return person > 0


def test_shadows_of_leaves_do_not_cut_the_skin_into_patches() -> None:
    """Light through leaves: bands of shade, darker and a little bluer, across the body."""
    photo = _photo()
    image = photo.image.copy()
    shade = np.zeros(image.shape[:2], dtype=bool)
    for top in (410, 470, 530):
        shade[top : top + 24] = True
    shade &= _person_area()
    image[shade] *= np.array([0.35, 0.35, 0.37], dtype=np.float32)
    body = _body(replace(photo, image=image))
    # Lit skin below the bands, and the shaded skin itself.
    for x, y in ((120, 450), (280, 510), (200, 570), (120, 480)):
        assert body[y, x] > 0.9, (x, y)
    assert body[30, 30] == 0.0


def test_a_lock_of_hair_across_the_shoulders_does_not_cut_the_body_off() -> None:
    photo = _photo()
    image = photo.image.copy()
    cv2.line(image, (0, 412), (400, 412), (0.01, 0.01, 0.01), 6)
    body = _body(replace(photo, image=image))
    assert body[CHEST[1], CHEST[0]] > 0.9
    assert body[460, 300] > 0.9


def test_a_black_and_white_photo_whitens_only_the_face() -> None:
    body = _body(_photo(grey=True))
    assert body[CHEEK[1], CHEEK[0]] > 0.9
    assert body[CHEST[1], CHEST[0]] < 0.05


# -- the whole chain -------------------------------------------------------


def test_the_slider_whitens_the_face_and_the_chest_and_not_the_background() -> None:
    photo = _photo()
    plain = pipeline.develop(photo, JPEG).astype(int)
    white = pipeline.develop(photo, JPEG.with_value("whiten", 100.0)).astype(int)
    for x, y in (CHEEK, CHEST):
        assert white[y, x].sum() > plain[y, x].sum() + 9, (x, y)
    assert np.array_equal(white[:40, :40], plain[:40, :40])
    x, y = APART
    assert np.array_equal(white[y, x], plain[y, x])


def test_a_photo_without_faces_is_not_changed_by_the_slider() -> None:
    photo = _photo(with_faces=False)
    plain = pipeline.develop(photo, JPEG)
    assert np.array_equal(pipeline.develop(photo, JPEG.with_value("whiten", 80.0)), plain)


def test_a_jpeg_without_faces_still_passes_through_and_is_copied(tmp_path) -> None:
    photo = replace(
        _photo(with_faces=False), faces=(), encoded=np.zeros((600, 400, 3), np.uint8)
    )
    state = JPEG.with_value("whiten", 50.0)
    assert pipeline.is_passthrough(photo, state)
    assert not pipeline.is_passthrough(replace(photo, faces=_photo().faces), state)
    source = tmp_path / "a.jpg"
    source.write_bytes(b"")
    assert exporter.can_copy_original(source, state, None, ExportOptions(), faces=())


def test_a_window_whitens_as_the_whole_photo_does() -> None:
    """Zoomed in on the chest, far from the face (PLAN.md 3.3)."""
    photo = _photo()
    state = JPEG.with_value("whiten", 100.0)
    whole = pipeline.develop(photo, state).astype(int)
    inner = Window.around(SIZE, (60.0, 430.0, 200.0, 560.0))
    outer = inner.grown(pipeline.blur_margin(photo, state, SIZE))
    body = pipeline.body_field(photo)
    part = pipeline.develop(geometry.window_of(photo, outer), state, window=outer, body=body)
    trimmed = outer.trim(part, inner).astype(int)
    expected = whole[inner.top : inner.bottom, inner.left : inner.right]
    assert np.abs(trimmed - expected).max() <= 2


def test_faces_are_looked_for_when_only_whitening_wants_them() -> None:
    photo = _photo(with_faces=False)
    given = _photo().faces
    attached = face_detector.attach(photo, JPEG.with_value("whiten", 30.0), None, given)
    assert attached.faces == given
    assert face_detector.attach(photo, JPEG, None, given) is photo


def _cut_by_a_cloth(rough: bool = False) -> DecodedPhoto:
    """The test person with a blue cloth across the chest, as a bikini's top cuts it.

    The skin carries a little grain, as skin does. With *rough*, the part
    below the cloth is grained like gravel or a pavement the colour of skin.
    """
    photo = _photo()
    rng = np.random.default_rng(7)
    grain = rng.normal(1.0, 0.02, photo.image.shape[:2]).astype(np.float32)
    image = photo.image * grain[..., None]
    person = _person_area()
    below = person.copy()
    below[:500] = False
    if rough:
        speckle = cv2.GaussianBlur(rng.normal(1.0, 0.35, photo.image.shape[:2]), (0, 0), 1.0)
        image[below] *= np.clip(speckle, 0.4, 1.6).astype(np.float32)[below][..., None]
    band = person.copy()
    band[:470] = False
    band[500:] = False
    image[band] = _linear(BACKGROUND)
    return replace(photo, image=image.astype(np.float32))


def test_skin_cut_off_by_clothes_is_taken_too() -> None:
    """2026-11-07: a belly below a bikini's top was left out, the chest above it whitened."""
    body = _body(_cut_by_a_cloth())
    assert body[440, 200] > 0.9  # above the cloth
    for x, y in ((200, 550), (120, 580), (300, 580)):
        assert body[y, x] > 0.9, (x, y)
    assert body[485, 200] < 0.1  # the cloth itself
    assert body[30, 30] == 0.0


def test_a_rough_surface_the_colour_of_skin_past_a_gap_is_left_out() -> None:
    body = _body(_cut_by_a_cloth(rough=True))
    assert body[440, 200] > 0.9
    assert body[550, 200] < 0.1
