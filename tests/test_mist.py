"""Tests for the black mist (PLAN.md 2.2 A-33 and section 6 step 6.5)."""

from __future__ import annotations

import numpy as np

from core import color_management as cm
from core import pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import MIST_MAX, EditState
from core.geometry import Window

JPEG = EditState(is_raw=False)
MISTY = JPEG.with_value("mist", 100.0)


def _photo(image: np.ndarray, encoded: bool = False) -> DecodedPhoto:
    """Return linear sRGB *image* as a decoded sRGB JPEG."""
    working = cm.apply_matrix(image.astype(np.float32), cm.SRGB_TO_PROPHOTO)
    shape = image.shape[:2]
    return DecodedPhoto(
        image=np.ascontiguousarray(working),
        is_raw=False,
        clipped=np.zeros(shape, dtype=bool),
        color_space=ColorSpace.SRGB,
        encoded=np.zeros((*shape, 3), np.uint8) if encoded else None,
    )


def _lamp(shape: tuple[int, int] = (240, 360), level: float = 0.02) -> np.ndarray:
    """Return a dark grey frame with a small white lamp left of the middle."""
    image = np.full((*shape, 3), level, dtype=np.float32)
    image[110:130, 120:140] = 1.0
    return image


def _develop(photo: DecodedPhoto, state: EditState, **kwargs: object) -> np.ndarray:
    """Return *photo* developed to 16 bit, as floats from 0 to 1."""
    out = pipeline.develop(photo, state, bits=pipeline.SIXTEEN_BIT, **kwargs)  # type: ignore[arg-type]
    return out.astype(np.float64) / 65535.0


# -- the slider ------------------------------------------------------------


def test_the_slider_runs_from_nothing_to_its_strongest() -> None:
    assert JPEG.with_value("mist", -10.0).mist == 0.0
    assert JPEG.with_value("mist", 500.0).mist == MIST_MAX == 100.0


def test_mist_counts_as_an_adjustment_and_is_stored() -> None:
    misty = JPEG.with_value("mist", 40.0)

    assert not misty.is_neutral_colour
    assert not misty.is_untouched(JPEG)
    assert EditState.from_dict(misty.to_dict(), JPEG).mist == 40.0
    assert EditState.from_dict({"exposure": 0.5}, JPEG).mist == 0.0


def test_a_jpeg_with_mist_is_no_longer_passed_straight_through() -> None:
    photo = _photo(_lamp(), encoded=True)

    assert pipeline.is_passthrough(photo, JPEG)
    assert not pipeline.is_passthrough(photo, JPEG.with_value("mist", 1.0))


def test_at_zero_nothing_changes() -> None:
    photo = _photo(_lamp())

    assert np.array_equal(
        pipeline.develop(photo, JPEG), pipeline.develop(photo, JPEG.with_value("mist", 0.0))
    )
    assert pipeline.mist_field(photo, JPEG) is None


# -- what it does ------------------------------------------------------------


def test_a_lamp_glows_into_the_dark_around_it() -> None:
    photo = _photo(_lamp())
    plain = _develop(photo, JPEG)
    misty = _develop(photo, MISTY)

    # Beside the lamp the dark is lit by its glow, and less so further out.
    near = misty[120, 150].mean() - plain[120, 150].mean()
    far = misty[120, 300].mean() - plain[120, 300].mean()
    assert near > 0.03
    assert 0.0 <= far < near


def test_the_blacks_stay_deep_away_from_any_light() -> None:
    photo = _photo(np.full((120, 160, 3), 0.004, dtype=np.float32))

    plain = _develop(photo, JPEG)
    misty = _develop(photo, MISTY)

    # A flat frame spreads as much light into each pixel as it takes out.
    assert np.abs(misty - plain).max() < 1.0 / 255.0


def test_a_grey_stays_grey() -> None:
    image = _lamp()
    image[20:60, 20:60] = 0.3
    misty = _develop(_photo(image), MISTY)

    assert np.abs(misty - misty.mean(axis=2, keepdims=True)).max() < 2.0 / 255.0


def test_the_glow_takes_the_colour_of_its_light() -> None:
    image = _lamp()
    image[110:130, 120:140] = (1.0, 0.45, 0.1)
    plain = _develop(_photo(image), JPEG)
    misty = _develop(_photo(image), MISTY)

    red, green, blue = misty[120, 150] - plain[120, 150]
    assert red > green > blue > -1e-4


def test_light_is_neither_made_nor_lost_in_linear_light() -> None:
    image = _lamp().astype(np.float32)
    working = cm.apply_matrix(image, cm.SRGB_TO_PROPHOTO)
    field = pipeline.mist_field(_photo(image), MISTY)
    assert field is not None
    rows, columns = pipeline.mist_axes(field, Window.whole((360, 240)))
    glow = pipeline.sample_mist(field, rows, columns)

    # Without the JPEG's cut off light taken as brighter, which the lamp is,
    # the glow is exactly what the pixels give up; with it, more.
    misty = pipeline.apply_mist(working, glow, 100.0, False, 0.0)
    assert misty.sum() > working.sum()
    dark = _photo(np.full((60, 80, 3), 0.05, dtype=np.float32))
    field = pipeline.mist_field(dark, MISTY)
    assert field is not None
    rows, columns = pipeline.mist_axes(field, Window.whole((80, 60)))
    spread = pipeline.apply_mist(
        dark.image, pipeline.sample_mist(field, rows, columns), 100.0, False, 0.0
    )
    assert np.allclose(spread.sum(), dark.image.sum(), rtol=1e-4)


def test_the_exposure_does_not_change_what_the_filter_does() -> None:
    photo = _photo(_lamp())
    field = pipeline.mist_field(photo, MISTY)
    assert field is not None
    rows, columns = pipeline.mist_axes(field, Window.whole(photo.size))
    glow = pipeline.sample_mist(field, rows, columns)

    once = pipeline.apply_mist(photo.image, glow, 60.0, False, 0.0) * np.float32(2.0)
    brighter = pipeline.apply_mist(photo.image * np.float32(2.0), glow, 60.0, False, 1.0)

    assert np.allclose(once, brighter, atol=1e-6)
    assert pipeline.mist_key(MISTY) == pipeline.mist_key(MISTY.with_value("exposure", 1.0))
    assert pipeline.mist_key(MISTY) == pipeline.mist_key(MISTY.with_value("mist", 20.0))


# -- the frame ---------------------------------------------------------------


def test_a_window_comes_out_as_the_whole_photo_does() -> None:
    photo = _photo(_lamp())
    field = pipeline.mist_field(photo, MISTY)
    whole = pipeline.develop(photo, MISTY, mist=field)
    pane = Window(frame=photo.size, left=150, top=90, right=300, bottom=200)
    part = DecodedPhoto(
        **{**photo.__dict__, "image": np.ascontiguousarray(pane.take(photo.image)),
           "clipped": pane.take(photo.clipped)}
    )

    window = pipeline.develop(part, MISTY, window=pane, mist=field)

    # The lamp lies outside the window and still glows into it.
    assert np.array_equal(window, whole[90:200, 150:300])


def test_a_field_from_a_smaller_copy_is_found_by_a_larger_frame() -> None:
    large = _photo(np.repeat(np.repeat(_lamp(), 2, axis=0), 2, axis=1))
    small = _photo(_lamp())
    from_small = pipeline.develop(large, MISTY, mist=pipeline.mist_field(small, MISTY))
    own = pipeline.develop(large, MISTY)

    difference = np.abs(from_small.astype(np.int16) - own.astype(np.int16))
    assert np.percentile(difference, 99) <= 2


def test_a_large_photo_is_shrunk_before_it_is_blurred() -> None:
    photo = _photo(np.zeros((300, 2500, 3), dtype=np.float32))
    field = pipeline.mist_field(photo, MISTY)

    assert field is not None
    assert field.factor == 3
    assert field.glow.shape == (100, 834, 3)
    assert field.frame == (2500, 300)
