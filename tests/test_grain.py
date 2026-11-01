"""Tests for film grain (PLAN.md 2.2 A-25 and section 6 step 11.5)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core import color_management as cm
from core import geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto, grain_seed_for
from core.edit_state import GRAIN_MAX, EditState

JPEG = EditState(is_raw=False)


def _photo(
    levels: tuple[float, float, float] = (118.0, 118.0, 118.0),
    shape: tuple[int, int] = (160, 200),
    seed: int = 7,
    shrunk_from: tuple[int, int] | None = None,
) -> DecodedPhoto:
    """Return a flat sRGB colour, given in 8 bit levels, as a decoded JPEG."""
    encoded = np.array(levels, dtype=np.float32) / np.float32(255.0)
    linear = cm.srgb_to_linear(encoded)
    working = cm.apply_matrix(
        np.broadcast_to(linear, (*shape, 3)).astype(np.float32), cm.SRGB_TO_PROPHOTO
    )
    return DecodedPhoto(
        image=np.ascontiguousarray(working),
        is_raw=False,
        clipped=np.zeros(shape, dtype=bool),
        color_space=ColorSpace.SRGB,
        shrunk_from=shrunk_from,
        grain_seed=seed,
    )


def _neighbours(field: np.ndarray, lag: int) -> float:
    """Return how alike pixels *lag* apart across a row are, from -1 to 1."""
    centred = field - field.mean()
    return float((centred[:, :-lag] * centred[:, lag:]).mean() / centred.var())


def _develop(photo: DecodedPhoto, amount: float, bits: int = pipeline.SIXTEEN_BIT) -> np.ndarray:
    """Return *photo* developed with grain *amount*, in 8 bit levels."""
    out = pipeline.develop(photo, JPEG.with_value("grain", amount), bits=bits)
    scale = 255.0 / (65535.0 if bits == pipeline.SIXTEEN_BIT else 255.0)
    return out.astype(np.float64) * scale


# -- the slider ------------------------------------------------------------


def test_the_slider_runs_from_nothing_to_the_camera_s_strong_grain() -> None:
    assert JPEG.with_value("grain", -10.0).grain == 0.0
    assert JPEG.with_value("grain", 500.0).grain == GRAIN_MAX == 100.0


def test_grain_counts_as_an_adjustment() -> None:
    grained = JPEG.with_value("grain", 30.0)

    assert not grained.is_neutral_colour
    assert not grained.is_untouched(JPEG)


def test_a_jpeg_with_grain_is_no_longer_passed_straight_through() -> None:
    photo = _photo()
    photo = DecodedPhoto(**{**photo.__dict__, "encoded": np.zeros((160, 200, 3), np.uint8)})

    assert pipeline.is_passthrough(photo, JPEG)
    assert not pipeline.is_passthrough(photo, JPEG.with_value("grain", 1.0))


def test_grain_is_stored_and_read_back_and_an_old_file_has_none() -> None:
    grained = JPEG.with_value("grain", 50.0)

    assert EditState.from_dict(grained.to_dict(), JPEG).grain == 50.0
    assert EditState.from_dict({"exposure": 0.5}, JPEG).grain == 0.0


def test_at_zero_nothing_is_added() -> None:
    photo = _photo()

    assert np.array_equal(
        pipeline.develop(photo, JPEG), pipeline.develop(photo, JPEG.with_value("grain", 0.0))
    )


# -- the seed --------------------------------------------------------------


def test_the_seed_comes_from_the_file_name_alone() -> None:
    assert grain_seed_for(Path("C:/a/DSCF3335.JPG")) == grain_seed_for(Path("D:/b/DSCF3335.JPG"))
    assert grain_seed_for(Path("DSCF3335.JPG")) != grain_seed_for(Path("DSCF3336.JPG"))
    assert grain_seed_for(Path("京都 2026.JPG")) == grain_seed_for(Path("x/京都 2026.JPG"))


def test_the_same_photo_gets_the_same_grain_and_another_photo_other_grain() -> None:
    first = pipeline.grain_field(_photo(seed=1))
    again = pipeline.grain_field(_photo(seed=1))
    other = pipeline.grain_field(_photo(seed=2))

    assert np.array_equal(first, again)
    assert abs(np.corrcoef(first.ravel(), other.ravel())[0, 1]) < 0.03


# -- the shape of the grain ------------------------------------------------


def test_at_full_size_the_grain_has_the_camera_s_spread_and_size() -> None:
    """Measured from the camera: 0.22 beside each other, -0.10 two apart."""
    field = pipeline.grain_field(_photo(shape=(512, 512)))

    assert field.std() == pytest.approx(1.0, abs=0.03)
    assert abs(field.mean()) < 0.01
    assert _neighbours(field, 1) == pytest.approx(0.22, abs=0.03)
    assert _neighbours(field, 2) == pytest.approx(-0.10, abs=0.03)
    assert abs(_neighbours(field, 3)) < 0.03
    # The same down a column as across a row: grains are round.
    assert _neighbours(field.T, 1) == pytest.approx(_neighbours(field, 1), abs=0.02)


def test_a_window_at_full_size_has_exactly_the_grain_of_the_whole_frame() -> None:
    photo = _photo(shape=(300, 400))
    window = geometry.Window(frame=(400, 300), left=97, top=41, right=251, bottom=203)

    whole = pipeline.grain_field(photo)
    part = pipeline.grain_field(geometry.window_of(photo, window), window)

    assert np.array_equal(part, window.take(whole))


def test_a_developed_window_at_full_size_matches_the_whole_photo() -> None:
    photo = _photo(shape=(300, 400))
    window = geometry.Window(frame=(400, 300), left=97, top=41, right=251, bottom=203)
    state = JPEG.with_value("grain", 100.0)

    whole = pipeline.develop(photo, state)
    part = pipeline.develop(geometry.window_of(photo, window), state, window=window)

    assert np.array_equal(part, window.take(whole))


@pytest.mark.parametrize("scale", [0.2, 0.41, 0.7, 0.9])
def test_a_smaller_copy_has_as_much_grain_as_shrinking_the_full_size_would_leave(
    scale: float,
) -> None:
    """The preview gets the grain the export of that size has (PLAN.md 6 step 11.5)."""
    full = pipeline.grain_field(_photo(shape=(1024, 1024)))
    side = round(1024 * scale)
    shrunk = cv2.resize(full, (side, side), interpolation=cv2.INTER_AREA)

    copy = pipeline.grain_field(_photo(shape=(side, side), shrunk_from=(1024, 1024)))

    assert copy.std() == pytest.approx(shrunk.std(), rel=0.10)


# -- what it does to the photo ---------------------------------------------


def test_at_100_a_middle_grey_moves_by_the_camera_s_strong_grain() -> None:
    """The camera's strong grain measured 14.3 levels, its weak one 7.2."""
    strong = _develop(_photo(shape=(256, 256)), 100.0)
    weak = _develop(_photo(shape=(256, 256)), 50.0)

    assert strong.std() == pytest.approx(14.3, rel=0.05)
    assert weak.std() == pytest.approx(7.2, rel=0.05)
    assert strong.mean() == pytest.approx(118.0, abs=0.5)


def test_the_grain_is_grey() -> None:
    """The three channels move together, so no colour speckles appear."""
    out = _develop(_photo(shape=(256, 256)), 100.0)

    assert np.abs(out[..., 0] - out[..., 1]).max() < 0.05
    assert np.abs(out[..., 1] - out[..., 2]).max() < 0.05


def test_the_grain_fades_into_pure_black_and_pure_white() -> None:
    black = _develop(_photo(levels=(0.0, 0.0, 0.0), shape=(128, 128)), 100.0)
    white = _develop(_photo(levels=(255.0, 255.0, 255.0), shape=(128, 128)), 100.0)
    middle = _develop(_photo(shape=(128, 128)), 100.0)

    assert black.std() < 0.05
    assert white.std() < 0.05
    assert middle.std() > 10.0


def test_a_channel_at_the_end_of_its_range_is_rolled_off_rather_than_cut() -> None:
    """A saturated colour stays inside 0 to 1, and its other channels still move."""
    encoded = np.zeros((64, 64, 3), dtype=np.float32)
    encoded[..., 0] = 1.0
    encoded[..., 1] = 0.85
    encoded[..., 2] = 0.0
    field = np.random.default_rng(3).normal(0.0, 3.0, (64, 64)).astype(np.float32)

    grained = pipeline.apply_grain(encoded, field, 100.0)

    assert grained.min() >= 0.0
    assert grained.max() <= 1.0
    assert grained[..., 1].std() > 0.04
    assert grained[..., 0].max() < 1.0 + 1e-6


def test_the_photo_handed_in_is_not_changed() -> None:
    encoded = np.full((32, 32, 3), 0.5, dtype=np.float32)
    kept = encoded.copy()

    pipeline.apply_grain(encoded, np.ones((32, 32), np.float32), 100.0)

    assert np.array_equal(encoded, kept)


def test_strength_follows_the_slider() -> None:
    photo = _photo(shape=(256, 256))

    assert _develop(photo, 100.0).std() == pytest.approx(
        2.0 * _develop(photo, 50.0).std(), rel=0.03
    )


def test_a_field_of_the_wrong_size_is_refused() -> None:
    photo = _photo()

    with pytest.raises(ValueError):
        pipeline.develop(photo, JPEG.with_value("grain", 50.0), grain=np.zeros((3, 3), np.float32))


def test_a_field_handed_in_is_the_one_used() -> None:
    photo = _photo()
    state = JPEG.with_value("grain", 80.0)

    kept = pipeline.grain_field(photo)

    assert np.array_equal(pipeline.develop(photo, state, grain=kept), pipeline.develop(photo, state))
