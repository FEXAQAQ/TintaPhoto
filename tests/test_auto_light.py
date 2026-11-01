"""Tests for the automatic light (core/auto_light.py).

The wand on the light group's header sets exposure, highlights, shadows,
whites and blacks from the photo itself, by trying values through the real
pipeline. These tests hold it to what PLAN.md section 6 promises: only those
five move, the answer does not depend on where they were, a dark photo is
lifted and a bright one brought down, a hazy one gets its black back, and a
white or a black that is already there is never taken away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import color_management as cm
from core import decoder, pipeline
from core.auto_light import AUTO_NAMES, auto_light
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import EditState

SAMPLES = Path(__file__).resolve().parent / "samples"
JPEG_REST = EditState.for_photo(False)


def _jpeg(encoded: np.ndarray) -> DecodedPhoto:
    """Return a JPEG photo holding *encoded* sRGB values from 0 to 255."""
    linear = cm.srgb_to_linear(encoded.astype(np.float32) / 255.0)
    return DecodedPhoto(
        image=cm.apply_matrix(linear, cm.SRGB_TO_PROPHOTO),
        is_raw=False,
        clipped=np.zeros(encoded.shape[:2], dtype=bool),
        color_space=ColorSpace.SRGB,
        encoded=encoded.astype(np.uint8),
    )


def _scene(low: float, high: float, height: int = 96, width: int = 144) -> DecodedPhoto:
    """Return a slightly coloured gradient running from *low* to *high*."""
    ramp = np.linspace(low, high, width, dtype=np.float32)
    rows = np.linspace(0.9, 1.1, height, dtype=np.float32)[:, None]
    grey = np.clip(ramp[None, :] * rows, 0.0, 1.0)
    tint = np.stack([grey, grey * 0.95, grey * 0.88], axis=-1)
    return _jpeg(np.round(np.clip(tint, 0.0, 1.0) * 255.0))


def _developed(photo: DecodedPhoto, state: EditState) -> np.ndarray:
    return pipeline.develop(pipeline.with_detail(photo, state), state).astype(np.float32) / 255.0


DARK = _scene(0.02, 0.30)
BRIGHT = _scene(0.55, 0.99)
HAZY = _scene(0.35, 0.65)
FULL = _scene(0.0, 1.0)


# -- what it touches ----------------------------------------------------------


def test_only_the_five_light_sliders_move() -> None:
    """Contrast, colour, white balance and the rest are the user's."""
    state = (
        JPEG_REST.with_value("contrast", 20.0)
        .with_value("vibrance", 15.0)
        .with_value("saturation", -5.0)
        .with_value("temperature", 10.0)
        .with_value("vignette", -20.0)
    )

    result = auto_light(HAZY, state)

    for name in AUTO_NAMES:
        state = state.with_value(name, getattr(result, name))
    assert result == state


def test_the_answer_does_not_depend_on_where_the_sliders_were() -> None:
    """Pressing it twice, or after moving them by hand, gives the same."""
    first = auto_light(DARK, JPEG_REST)
    moved = first.with_value("exposure", -1.5).with_value("blacks", 40.0)

    assert auto_light(DARK, first) == first
    assert auto_light(DARK, moved) == first


def test_the_photo_handed_in_is_not_changed() -> None:
    before = DARK.image.copy()

    auto_light(DARK, JPEG_REST)

    assert np.array_equal(DARK.image, before)


def test_the_answer_is_rounded_to_what_the_sliders_show() -> None:
    result = auto_light(HAZY, JPEG_REST)

    assert result.exposure == round(result.exposure, 2)
    for name in ("highlights", "shadows", "whites", "blacks"):
        assert getattr(result, name) == round(getattr(result, name))


# -- what it does ---------------------------------------------------------------


def test_a_dark_photo_is_lifted_and_a_bright_one_brought_down() -> None:
    assert auto_light(DARK, JPEG_REST).exposure > 0.0
    assert auto_light(BRIGHT, JPEG_REST).exposure < 0.0


def test_the_exposure_stays_within_its_limit() -> None:
    very_dark = _scene(0.0, 0.05)

    stops = auto_light(very_dark, JPEG_REST).exposure

    assert 0.0 < stops <= pipeline.AUTO_EXPOSURE_LIMIT_EV


def test_a_hazy_photo_gets_its_black_and_white_back() -> None:
    result = auto_light(HAZY, JPEG_REST)

    assert result.blacks < 0.0
    assert result.whites > 0.0
    before = _developed(HAZY, JPEG_REST)
    after = _developed(HAZY, result)
    assert after.min() < before.min()
    assert after.max() > before.max()


@pytest.mark.parametrize("photo", [DARK, BRIGHT, HAZY, FULL], ids=["dark", "bright", "hazy", "full"])
def test_a_white_is_never_pulled_down_and_a_black_never_lifted(photo: DecodedPhoto) -> None:
    result = auto_light(photo, JPEG_REST)

    assert result.whites >= 0.0
    assert result.blacks <= 0.0
    assert -pipeline.AUTO_HIGHLIGHTS_LIMIT <= result.highlights <= 0.0
    assert 0.0 <= result.shadows <= pipeline.AUTO_SHADOWS_LIMIT


def test_a_photo_that_already_reaches_black_and_white_is_left_alone() -> None:
    result = auto_light(FULL, JPEG_REST)

    assert result.whites == 0.0
    assert result.blacks == 0.0


def test_a_small_bright_subject_on_black_is_not_burnt_out() -> None:
    """The exposure stops where the brightest pixels reach white, however
    much black surrounds them: the black is the photo, not a mistake."""
    encoded = np.full((96, 144, 3), 6, dtype=np.uint8)
    encoded[40:56, 60:84] = (215, 200, 185)
    photo = _jpeg(encoded)

    result = auto_light(photo, JPEG_REST)

    # All that black asks for the whole limit; the subject says stop sooner.
    assert 0.0 <= result.exposure < pipeline.AUTO_EXPOSURE_LIMIT_EV
    subject = _developed(photo, result)[40:56, 60:84].max(axis=-1)
    assert np.median(subject) <= pipeline.AUTO_WHITE_TARGET
    assert np.mean(subject >= pipeline.AUTO_CLIPPED_LEVEL) < 0.25


def test_the_film_grain_does_not_change_the_answer() -> None:
    """Grain is noise; it is left out while measuring."""
    grainy = JPEG_REST.with_value("grain", 60.0)

    result = auto_light(HAZY, grainy)

    assert result.grain == 60.0
    plain = auto_light(HAZY, JPEG_REST)
    assert [getattr(result, name) for name in AUTO_NAMES] == [
        getattr(plain, name) for name in AUTO_NAMES
    ]


# -- on a real photo ----------------------------------------------------------


@pytest.mark.skipif(not (SAMPLES / "_DSC7753.ARW").is_file(), reason="sample RAW not present")
def test_a_sunset_keeps_its_white_sun_and_its_black_silhouettes() -> None:
    """The case that showed the first version wrong: it turned the sun into a
    grey disc and lifted the black shore to grey."""
    photo = decoder.decode(SAMPLES / "_DSC7753.ARW")
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    small = decoder.resize_photo(photo, pipeline.AUTO_LONG_EDGE)

    result = auto_light(small, state)

    assert result.whites == 0.0
    assert result.blacks == 0.0
    developed = _developed(small, result)
    assert np.percentile(developed.max(axis=-1), 99.9) >= pipeline.AUTO_WHITE_TARGET
