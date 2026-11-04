"""Tests for the detail step: noise reduction and sharpening (PLAN.md 6 step 2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from core import color_management as cm
from core import geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import (
    RAW_COLOUR_NOISE_DEFAULT,
    RAW_SHARPENING_DEFAULT,
    SHARPENING_MAX,
    SHARPENING_MIN,
    EditState,
)
from tests.test_pipeline import CAMERA_RGB_XYZ_MATRIX

JPEG = EditState(is_raw=False)


def _noisy(
    level: float = 0.18, spread: float = 0.02, shape: tuple[int, int] = (240, 320), seed: int = 0
) -> np.ndarray:
    """Return a flat grey with independent noise in every channel."""
    rng = np.random.default_rng(seed)
    image = np.full((*shape, 3), level, dtype=np.float32)
    return np.clip(image + rng.normal(0.0, spread, image.shape), 0.0, None).astype(np.float32)


def _edge(shape: tuple[int, int] = (64, 96)) -> np.ndarray:
    """Return a dark half and a light half, side by side."""
    image = np.full((*shape, 3), 0.05, dtype=np.float32)
    image[:, shape[1] // 2 :] = 0.6
    return image


def _brightness(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=-1)


def _colourfulness(image: np.ndarray) -> float:
    return float(np.abs(image - image.mean(axis=-1, keepdims=True)).mean())


def _raw_photo(image: np.ndarray) -> DecodedPhoto:
    return DecodedPhoto(
        image=image.astype(np.float32),
        is_raw=True,
        clipped=np.zeros(image.shape[:2], dtype=bool),
        camera_to_xyz=cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX),
        camera_whitebalance=(1.0, 1.0, 1.0),
    )


# -- nothing asked for, nothing done --------------------------------------


def test_with_every_slider_at_zero_the_photo_is_handed_back_untouched() -> None:
    image = _noisy()

    assert pipeline.apply_detail(image, JPEG, 1.0, False) is image


def test_the_layers_add_back_up_to_the_photo() -> None:
    """With nothing let go, splitting and joining changes nothing."""
    image = _noisy()
    barely = JPEG.with_value("luminance_noise", 1e-9).with_value("colour_noise", 1e-9)

    result = pipeline.apply_detail(image, barely, 1.0, False)

    assert np.abs(result - image).max() < 1e-5


def test_the_photo_handed_in_is_not_changed() -> None:
    image = _noisy()
    kept = image.copy()
    state = JPEG.with_value("sharpening", 80.0).with_value("luminance_noise", 50.0)

    pipeline.apply_detail(image, state, 1.0, False)

    assert np.array_equal(image, kept)


# -- noise -----------------------------------------------------------------


def test_luminance_noise_reduction_smooths_the_grain() -> None:
    image = _noisy()
    state = JPEG.with_value("luminance_noise", 60.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert _brightness(result).std() < 0.6 * _brightness(image).std()


def test_more_luminance_noise_reduction_smooths_more() -> None:
    image = _noisy()

    def grain(amount: float) -> float:
        state = JPEG.with_value("luminance_noise", amount)
        return float(_brightness(pipeline.apply_detail(image, state, 1.0, False)).std())

    assert grain(100.0) < grain(50.0) < grain(20.0)


def test_luminance_noise_reduction_keeps_a_hard_edge() -> None:
    image = _edge()
    state = JPEG.with_value("luminance_noise", 100.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    middle = image.shape[1] // 2
    assert result[:, middle - 1].mean() == pytest.approx(0.05, abs=0.01)
    assert result[:, middle].mean() == pytest.approx(0.6, abs=0.02)


def test_colour_noise_reduction_takes_colour_out_of_a_grey() -> None:
    image = _noisy(spread=0.01)
    state = JPEG.with_value("colour_noise", 50.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert _colourfulness(result) < 0.5 * _colourfulness(image)
    # The brightness is not what the colour slider is about.
    assert _brightness(result).mean() == pytest.approx(_brightness(image).mean(), rel=0.02)


def test_colour_noise_reduction_leaves_a_real_colour_alone() -> None:
    image = np.full((96, 96, 3), (0.4, 0.15, 0.05), dtype=np.float32)
    state = JPEG.with_value("colour_noise", 100.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert np.abs(result - image).max() < 1e-4


def test_a_coloured_line_along_an_edge_survives_colour_noise_reduction() -> None:
    """The brightness guides the colour, so painted outlines are kept."""
    image = np.full((64, 64, 3), 0.5, dtype=np.float32)
    image[:, 30:32] = (0.25, 0.02, 0.02)  # a thin dark red line
    state = JPEG.with_value("colour_noise", 25.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    line = result[:, 30:32]
    assert (line[..., 0] - line[..., 1]).mean() > 0.8 * (0.25 - 0.02)


def test_the_noise_thresholds_mean_the_same_for_raw_and_jpeg() -> None:
    """A RAW is lifted onto a JPEG's middle grey before it is judged."""
    jpeg = _noisy(level=0.18, spread=0.02)
    raw = jpeg / np.float32(2.0**pipeline.RAW_BASE_EXPOSURE_EV)
    state = JPEG.with_value("luminance_noise", 60.0)

    as_jpeg = pipeline.apply_detail(jpeg, state, 1.0, False)
    as_raw = pipeline.apply_detail(raw, state, 1.0, True)

    lifted = as_raw * np.float32(2.0**pipeline.RAW_BASE_EXPOSURE_EV)
    assert np.abs(lifted - as_jpeg).max() < 1e-4


# -- sharpening ------------------------------------------------------------


def test_sharpening_steepens_an_edge() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", 100.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    middle = image.shape[1] // 2
    assert result[:, middle - 1].mean() < image[:, middle - 1].mean()
    assert result[:, middle].mean() > image[:, middle].mean()


def test_sharpening_leaves_a_flat_area_alone() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", SHARPENING_MAX)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert np.abs(result[:, :20] - image[:, :20]).max() < 1e-5
    assert np.abs(result[:, -20:] - image[:, -20:]).max() < 1e-5


def test_sharpening_keeps_the_colour_of_a_neutral_edge() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", SHARPENING_MAX)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert np.abs(result - result.mean(axis=-1, keepdims=True)).max() < 1e-5


def test_the_halo_of_a_hard_edge_is_held_back() -> None:
    """However far the slider goes, an edge is pushed by a bounded amount."""
    image = _edge()
    state = JPEG.with_value("sharpening", SHARPENING_MAX)

    result = pipeline.apply_detail(image, state, 1.0, False)

    middle = image.shape[1] // 2
    root_change = np.sqrt(result[:, middle]) - np.sqrt(image[:, middle])
    assert root_change.max() <= pipeline.SHARPEN_HALO_LIMIT + 1e-5


def test_masking_keeps_the_sharpening_off_the_grain() -> None:
    image = _noisy(spread=0.01)
    plain = JPEG.with_value("sharpening", 100.0)
    masked = plain.with_value("sharpen_masking", 100.0)

    grain = float(_brightness(pipeline.apply_detail(image, plain, 1.0, False)).std())
    kept_off = float(_brightness(pipeline.apply_detail(image, masked, 1.0, False)).std())

    assert kept_off < grain
    assert kept_off == pytest.approx(float(_brightness(image).std()), rel=0.1)


def test_masking_still_lets_a_strong_edge_be_sharpened() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", 100.0).with_value("sharpen_masking", 100.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    middle = image.shape[1] // 2
    assert result[:, middle].mean() > image[:, middle].mean()


# -- softening, below zero ------------------------------------------------


def test_softening_flattens_an_edge() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", -100.0)

    result = pipeline.apply_detail(image, state, 1.0, False)

    middle = image.shape[1] // 2
    assert result[:, middle - 1].mean() > image[:, middle - 1].mean()
    assert result[:, middle].mean() < image[:, middle].mean()
    # A blur adds nothing: what comes out stays between the two halves.
    assert result.min() >= image.min() - 1e-5
    assert result.max() <= image.max() + 1e-5


def test_softening_grows_with_the_slider() -> None:
    image = _noisy(spread=0.01)

    def grain(amount: float) -> float:
        state = JPEG.with_value("sharpening", amount)
        return float(_brightness(pipeline.apply_detail(image, state, 1.0, False)).std())

    assert grain(-100.0) < grain(-40.0) < float(_brightness(image).std())


def test_softening_keeps_the_colour_of_a_neutral_edge() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", SHARPENING_MIN)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert np.abs(result - result.mean(axis=-1, keepdims=True)).max() < 1e-5


def test_softening_leaves_a_flat_area_alone() -> None:
    image = _edge()
    state = JPEG.with_value("sharpening", SHARPENING_MIN)

    result = pipeline.apply_detail(image, state, 1.0, False)

    assert np.abs(result[:, :20] - image[:, :20]).max() < 1e-5


def test_the_masking_takes_no_part_in_softening() -> None:
    image = _noisy(spread=0.01)
    plain = JPEG.with_value("sharpening", -60.0)
    masked = plain.with_value("sharpen_masking", 100.0)

    assert np.array_equal(
        pipeline.apply_detail(image, plain, 1.0, False),
        pipeline.apply_detail(image, masked, 1.0, False),
    )


def test_a_softened_window_reaches_as_far_as_its_blur() -> None:
    state = JPEG.with_value("sharpening", -50.0)

    assert pipeline.detail_reach(state) >= np.ceil(
        pipeline.BLUR_REACH_SIGMAS * pipeline.SOFTEN_SIGMA
    )


# -- sizes are pixels of the full size photo -------------------------------


def test_a_smaller_copy_is_sharpened_less_per_pixel() -> None:
    """On a copy a fifth of the size, a one pixel radius is a fifth of a pixel."""
    image = _edge()
    state = JPEG.with_value("sharpening", 100.0)
    middle = image.shape[1] // 2

    full = pipeline.apply_detail(image, state, 1.0, False)[:, middle].mean()
    small = pipeline.apply_detail(image, state, 0.2, False)[:, middle].mean()

    assert image[:, middle].mean() <= small < full


def test_layers_finer_than_a_pixel_of_the_copy_are_left_out() -> None:
    levels = pipeline._detail_levels(
        pipeline.LUMINANCE_NOISE_SIGMAS, pipeline.LUMINANCE_NOISE_THRESHOLDS, 0.2, 100.0
    )

    assert all(sigma >= pipeline.DETAIL_FINEST_SIGMA for sigma, _, _ in levels)
    assert len(levels) < len(pipeline.LUMINANCE_NOISE_SIGMAS)


def test_the_scale_of_a_photo_is_its_share_of_the_full_size() -> None:
    photo = _raw_photo(np.zeros((40, 60, 3), dtype=np.float32))

    assert pipeline.detail_scale(photo) == 1.0
    shrunk = DecodedPhoto(
        image=photo.image, is_raw=True, clipped=photo.clipped, shrunk_from=(600, 400)
    )
    assert pipeline.detail_scale(shrunk) == pytest.approx(0.1)


def test_a_window_knows_how_large_it_would_be_at_full_size() -> None:
    proxy = _raw_photo(np.zeros((400, 600, 3), dtype=np.float32))
    photo = DecodedPhoto(
        image=proxy.image, is_raw=True, clipped=proxy.clipped, shrunk_from=(6000, 4000)
    )
    window = geometry.Window.around((6000, 4000), (1000.0, 1000.0, 2000.0, 1600.0)).scaled(0.1)

    part = geometry.window_of(photo, window)

    assert pipeline.detail_scale(part) == pytest.approx(0.1, rel=0.02)


# -- in the pipeline -------------------------------------------------------


def test_a_window_with_detail_comes_out_as_the_whole_photo_would_have() -> None:
    """PLAN.md 3.3, with the detail step looking at its neighbours."""
    image = _noisy(level=0.03, spread=0.006, shape=(240, 360), seed=4)
    image[80:160, 120:240] *= 3.0
    photo = _raw_photo(image)
    frame = photo.size

    for state in (
        EditState(is_raw=True).with_value("sharpening", 80.0).with_value("sharpen_masking", 30.0),
        EditState(is_raw=True).with_value("sharpening", -100.0),
        EditState(is_raw=True).with_value("luminance_noise", 80.0),
        EditState(is_raw=True).with_value("colour_noise", 80.0),
    ):
        asked = geometry.Window.around(frame, (150.0, 60.0, 290.0, 190.0))
        around = asked.grown(pipeline.blur_margin(photo, state, frame))
        windowed = around.trim(
            pipeline.develop(geometry.window_of(photo, around), state, window=around), asked
        )
        whole = pipeline.develop(photo, state)[
            asked.top : asked.bottom, asked.left : asked.right
        ]
        difference = np.abs(windowed.astype(int) - whole.astype(int))
        assert difference.max() <= 1
        assert (difference > 0).mean() < 0.03


def test_the_margin_covers_the_reach_of_the_detail_step() -> None:
    photo = _raw_photo(np.zeros((48, 64, 3), dtype=np.float32))
    colour = EditState(is_raw=True).with_value("colour_noise", 25.0)
    frame = (6000, 4000)

    assert pipeline.blur_margin(photo, colour, frame) >= np.ceil(
        pipeline.BLUR_REACH_SIGMAS * max(pipeline.COLOUR_NOISE_SIGMAS)
    )
    # Measured in pixels of the full size photo, so a copy needs less.
    assert pipeline.blur_margin(photo, colour, frame, scale=0.25) < pipeline.blur_margin(
        photo, colour, frame
    )


def test_a_jpeg_with_detail_is_no_longer_passed_through() -> None:
    pixels = np.full((40, 60, 3), 120, dtype=np.uint8)
    photo = DecodedPhoto(
        image=cm.apply_matrix(
            cm.srgb_to_linear(pixels.astype(np.float32) / 255.0), cm.SRGB_TO_PROPHOTO
        ),
        is_raw=False,
        clipped=np.zeros((40, 60), dtype=bool),
        color_space=ColorSpace.SRGB,
        encoded=pixels,
    )

    assert pipeline.is_passthrough(photo, JPEG)
    assert not pipeline.is_passthrough(photo, JPEG.with_value("sharpening", 10.0))
    assert not pipeline.is_passthrough(photo, JPEG.with_value("sharpening", -10.0))
    assert not pipeline.is_passthrough(photo, JPEG.with_value("colour_noise", 10.0))
    # The masking alone masks nothing, so the photo is still untouched.
    assert pipeline.is_passthrough(photo, JPEG.with_value("sharpen_masking", 50.0))


# -- where a photo rests ---------------------------------------------------


def test_a_raw_photo_rests_with_some_sharpening_and_colour_noise_reduction() -> None:
    rest = EditState.for_photo(True, 5600.0, 0.0)

    assert rest.sharpening == RAW_SHARPENING_DEFAULT
    assert rest.colour_noise == RAW_COLOUR_NOISE_DEFAULT
    assert rest.luminance_noise == 0.0
    assert rest.is_untouched(rest)


def test_a_jpeg_rests_with_none() -> None:
    rest = EditState.for_photo(False)

    assert not pipeline.has_detail(rest)


def test_moving_a_detail_slider_counts_as_a_change() -> None:
    rest = EditState.for_photo(True, 5600.0, 0.0)

    assert not rest.with_value("sharpening", 60.0).is_untouched(rest)
    assert not rest.with_value("luminance_noise", 10.0).is_untouched(rest)
    assert not rest.with_value("sharpen_masking", 30.0).is_untouched(rest)


def test_the_masking_does_not_count_without_sharpening() -> None:
    rest = EditState.for_photo(False)

    assert rest.with_value("sharpen_masking", 70.0).is_untouched(rest)
    softened = rest.with_value("sharpening", -30.0)
    assert softened.with_value("sharpen_masking", 70.0).detail_is_untouched(
        softened.with_value("sharpen_masking", 0.0)
    )


def test_the_detail_sliders_stay_in_their_ranges() -> None:
    state = EditState.for_photo(False)

    assert state.with_value("sharpening", 400.0).sharpening == SHARPENING_MAX
    assert state.with_value("sharpening", -400.0).sharpening == SHARPENING_MIN
    assert state.with_value("sharpening", -40.0).sharpening == -40.0
    assert state.with_value("luminance_noise", -5.0).luminance_noise == 0.0
    assert state.with_value("colour_noise", 140.0).colour_noise == 100.0


def test_an_older_parameter_file_gives_a_raw_photo_the_resting_detail() -> None:
    rest = EditState.for_photo(True, 5600.0, 0.0)

    state = EditState.from_dict({"exposure": 0.5}, rest)

    assert state.sharpening == RAW_SHARPENING_DEFAULT
    assert state.colour_noise == RAW_COLOUR_NOISE_DEFAULT
