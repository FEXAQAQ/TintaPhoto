"""Tests for the processing pipeline (PLAN.md section 6).

The properties checked here are the ones a photo editor must never get wrong:
neutral colours stay neutral, hue does not drift, nothing leaves the output
gamut, and an untouched JPEG comes out exactly as it went in.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core import color_management as cm
from core import filters, geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import EditState

#: The matrix LibRaw reports for the camera this was written against.
CAMERA_RGB_XYZ_MATRIX = np.array(
    [
        [0.7374, -0.2389, -0.0551],
        [-0.5435, 1.3162, 0.2519],
        [-0.1006, 0.1795, 0.6552],
        [0.0000, 0.0000, 0.0000],
    ]
)


def _image(*colours: tuple[float, float, float]) -> np.ndarray:
    return np.array([list(colours)], dtype=np.float32)


def _scene_mid_grey() -> float:
    """The linear RAW level that the base curve maps onto middle grey.

    The curve lifts sensor data by RAW_BASE_EXPOSURE_EV before its anchor, so
    middle grey arrives from well below it.
    """
    return pipeline.TONE_MID_INPUT / 2.0**pipeline.RAW_BASE_EXPOSURE_EV


def _raw_photo(image: np.ndarray, clipped: np.ndarray | None = None) -> DecodedPhoto:
    return DecodedPhoto(
        image=image.astype(np.float32),
        is_raw=True,
        clipped=np.zeros(image.shape[:2], dtype=bool) if clipped is None else clipped,
        camera_to_xyz=cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX),
        camera_whitebalance=(1.0, 1.0, 1.0),
    )


def _jpeg_photo(encoded: np.ndarray, color_space: ColorSpace = ColorSpace.SRGB) -> DecodedPhoto:
    linear = (
        cm.adobe_rgb_to_linear(encoded.astype(np.float32) / 255.0)
        if color_space is ColorSpace.ADOBE_RGB
        else cm.srgb_to_linear(encoded.astype(np.float32) / 255.0)
    )
    matrix = (
        cm.ADOBE_RGB_TO_PROPHOTO if color_space is ColorSpace.ADOBE_RGB else cm.SRGB_TO_PROPHOTO
    )
    return DecodedPhoto(
        image=cm.apply_matrix(linear, matrix),
        is_raw=False,
        clipped=np.zeros(encoded.shape[:2], dtype=bool),
        color_space=color_space,
        encoded=encoded.astype(np.uint8),
    )


# -- purity ---------------------------------------------------------------


def test_the_steps_do_not_modify_their_input() -> None:
    image = _image((0.3, 0.2, 0.1), (0.9, 0.8, 0.7))
    original = image.copy()
    pipeline.base_tone_curve(image)
    pipeline.black_and_white_points(image, -70.0, 70.0)
    pipeline.compress_to_gamut(image)
    pipeline.to_srgb8(image)
    pipeline.neutralise_clipped_highlights(image, np.ones(image.shape[:2], dtype=bool))
    assert np.array_equal(image, original)


def test_the_steps_return_float32() -> None:
    image = _image((0.3, 0.2, 0.1))
    assert pipeline.base_tone_curve(image).dtype == np.float32
    assert pipeline.black_and_white_points(image, -70.0, 70.0).dtype == np.float32
    assert pipeline.compress_to_gamut(image).dtype == np.float32
    assert pipeline.to_srgb(image).dtype == np.float32
    assert pipeline.to_srgb8(image).dtype == np.uint8


# -- step 4: clipped highlights -------------------------------------------


def test_a_clipped_magenta_highlight_becomes_neutral() -> None:
    """A highlight whose green has clipped must not stay magenta."""
    image = np.full((32, 32, 3), 1.2, dtype=np.float32)
    image[..., 1] = 1.0  # green clipped lower than red and blue
    clipped = np.ones(image.shape[:2], dtype=bool)
    result = pipeline.neutralise_clipped_highlights(image, clipped)
    centre = result[16, 16]
    assert centre.max() - centre.min() < 1e-3, centre


def test_pixels_that_did_not_clip_are_left_alone() -> None:
    image = np.full((64, 64, 3), 0.5, dtype=np.float32)
    image[32:, :, 0] = 0.4
    clipped = np.zeros(image.shape[:2], dtype=bool)
    assert np.array_equal(pipeline.neutralise_clipped_highlights(image, clipped), image)


def test_the_correction_fades_in_rather_than_showing_an_edge() -> None:
    """The softened mask must give a gradient, not a hard boundary."""
    image = np.full((64, 64, 3), 1.0, dtype=np.float32)
    image[..., 2] = 0.6
    clipped = np.zeros(image.shape[:2], dtype=bool)
    clipped[:, 32:] = True
    result = pipeline.neutralise_clipped_highlights(image, clipped)
    blue = result[32, :, 2]
    # Several pixels either side of the boundary take intermediate values.
    between = np.sum((blue > 0.61) & (blue < 0.99))
    assert between >= 4, blue[28:38]


def test_a_blown_sky_ends_up_close_to_white() -> None:
    """The point of the step: clipped pixels lose their colour cast."""
    image = np.full((48, 48, 3), 0.95, dtype=np.float32)
    image[..., 0] = 1.0  # red still at full scale, green and blue clipped lower
    image[..., 2] = 0.80
    clipped = np.ones(image.shape[:2], dtype=bool)

    def cast(photo: np.ndarray) -> float:
        middle = photo[24, 24]
        return float((middle.max() - middle.min()) / middle.max())

    before = cast(image)
    after = cast(pipeline.neutralise_clipped_highlights(image, clipped))
    assert before > 0.15
    assert after < 0.02, f"{before:.3f} -> {after:.3f}"


def test_the_corrected_highlight_keeps_its_brightness() -> None:
    image = np.full((16, 16, 3), 1.0, dtype=np.float32)
    image[..., 1] = 0.8
    result = pipeline.neutralise_clipped_highlights(image, np.ones((16, 16), dtype=bool))
    assert result[8, 8].max() == pytest.approx(1.0, abs=1e-3)


# -- step 5: colour conversion --------------------------------------------


def test_camera_neutral_stays_neutral_in_the_working_space() -> None:
    matrix = cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX)
    result = pipeline.camera_to_working_space(_image((0.4, 0.4, 0.4)), matrix)[0, 0]
    assert result.max() - result.min() < 1e-3, result


# -- step 9: base tone curve ----------------------------------------------


def test_middle_grey_maps_onto_itself() -> None:
    """A correctly exposed middle grey comes out as middle grey."""
    grey = _image((_scene_mid_grey(),) * 3)
    result = pipeline.base_tone_curve(grey)[0, 0]
    assert result[0] == pytest.approx(pipeline.TONE_MID_OUTPUT, abs=1e-4)


def test_the_baseline_gain_lifts_sensor_data_onto_the_curve() -> None:
    """Linear RAW is scaled to sensor saturation, not to a rendered grey."""
    assert pipeline.RAW_BASE_EXPOSURE_EV > 0
    assert _scene_mid_grey() < pipeline.TONE_MID_INPUT


def test_the_curve_is_monotonic() -> None:
    values = np.linspace(0.0, 8.0, 512, dtype=np.float32)
    image = np.stack([values] * 3, axis=-1)[np.newaxis, ...]
    result = pipeline.base_tone_curve(image)[0, :, 0]
    assert np.all(np.diff(result) >= -1e-7)


def test_the_curve_approaches_white_without_passing_it() -> None:
    bright = _image((4.0, 4.0, 4.0), (16.0, 16.0, 16.0), (64.0, 64.0, 64.0))
    result = pipeline.base_tone_curve(bright)[0, :, 0]
    assert np.all(result < 1.0)
    assert result[-1] > 0.95
    assert np.all(np.diff(result) > 0)


def test_black_stays_black() -> None:
    assert pipeline.base_tone_curve(_image((0.0, 0.0, 0.0)))[0, 0].max() == 0.0


def test_the_curve_has_a_toe_and_a_shoulder() -> None:
    """Around middle grey, darks fall away and brights are held back."""
    mid = pipeline.TONE_MID_OUTPUT
    dark = pipeline.base_tone_curve(_image((_scene_mid_grey() / 4,) * 3))[0, 0, 0]
    bright = pipeline.base_tone_curve(_image((_scene_mid_grey() * 4,) * 3))[0, 0, 0]
    assert dark < mid / 4  # toe
    assert bright < mid * 4  # shoulder


def test_the_curve_keeps_the_ratio_between_channels() -> None:
    """Scaling all three channels together is what keeps the hue (6 step 9)."""
    colour = _image((0.6, 0.3, 0.15))
    result = pipeline.base_tone_curve(colour)[0, 0]
    assert result[0] / result[1] == pytest.approx(0.6 / 0.3, rel=1e-4)
    assert result[1] / result[2] == pytest.approx(0.3 / 0.15, rel=1e-4)


def test_the_slope_at_middle_grey_matches_the_constant() -> None:
    """The measured log-log slope must be the one asked for."""
    step = 0.01
    mid = _scene_mid_grey()
    low = pipeline.base_tone_curve(_image((mid * (1 - step),) * 3))[0, 0, 0]
    high = pipeline.base_tone_curve(_image((mid * (1 + step),) * 3))[0, 0, 0]
    slope = (np.log(high) - np.log(low)) / (np.log(1 + step) - np.log(1 - step))
    assert slope == pytest.approx(pipeline.TONE_MID_SLOPE, abs=0.02)


# -- step 9.5: black and white points -------------------------------------


def test_the_points_sit_between_the_base_curve_and_the_gamut_compression() -> None:
    """Where the step sits is what makes 1.0 mean white (PLAN.md 6 step 9.5).

    The white point is free to push the top of the range past 1.0 because
    step 11 is still to come; what it must not do is reach the output with
    anything above 1.0 left in it.
    """
    photo = _raw_photo(np.full((16, 16, 3), 0.4, dtype=np.float32))
    state = EditState.for_photo(True).with_value("whites", 100.0)
    result = pipeline.develop(photo, state)
    assert int(result.max()) <= 255


def test_the_points_at_zero_leave_the_developed_photo_untouched() -> None:
    """Nothing may change for a photo whose two new sliders are at zero.

    This is the comparison CLAUDE.md asks for after any change to the
    pipeline: with both sliders where they rest, every pixel must be the one
    the previous version produced, which is what makes the step inert until
    it is asked for.
    """
    image = np.linspace(0.0, 0.6, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    photo = _raw_photo(image)
    state = EditState.for_photo(True)
    assert state.whites == 0.0 and state.blacks == 0.0
    developed = pipeline.develop(photo, state)
    assert np.array_equal(developed, pipeline.develop(photo, None))
    # The step itself hands back the array it was given, so the chain the
    # other steps form is the one it was before.
    assert pipeline.black_and_white_points(image, 0.0, 0.0) is image


@pytest.mark.parametrize("name", ["whites", "blacks"])
def test_touching_either_point_stops_a_jpeg_passing_through(name: str) -> None:
    """A JPEG is only left alone while every colour slider rests (X-08)."""
    pixels = np.full((8, 12, 3), (120, 130, 140), dtype=np.uint8)
    photo = _jpeg_photo(pixels)
    state = EditState(is_raw=False).with_value(name, -50.0)
    assert not pipeline.is_passthrough(photo, state)
    assert not np.array_equal(pipeline.develop(photo, state), pixels)


@pytest.mark.parametrize("name", ["whites", "blacks"])
def test_either_point_reaches_the_developed_photo(name: str) -> None:
    photo = _raw_photo(np.full((16, 16, 3), 0.2, dtype=np.float32))
    state = EditState.for_photo(True)
    lower = pipeline.develop(photo, state.with_value(name, -100.0))
    higher = pipeline.develop(photo, state.with_value(name, 100.0))
    assert int(higher[8, 8, 0]) > int(lower[8, 8, 0])


# -- step 11: gamut compression -------------------------------------------


def _is_in_srgb(image: np.ndarray) -> np.ndarray:
    srgb = cm.apply_matrix(image, cm.PROPHOTO_TO_SRGB)
    return np.all((srgb >= -1e-3) & (srgb <= 1.0 + 1e-3), axis=-1)


def test_colours_far_outside_srgb_are_brought_back_in() -> None:
    """A saturated neon or sunset must end up inside the output gamut."""
    wide = cm.apply_matrix(
        _image((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        cm.PROPHOTO_TO_XYZ,
    )
    wide = cm.apply_matrix(wide, cm.XYZ_TO_PROPHOTO)  # saturated ProPhoto primaries
    assert np.all(_is_in_srgb(pipeline.compress_to_gamut(wide)))


def test_colours_already_well_inside_the_gamut_are_untouched() -> None:
    modest = cm.apply_matrix(_image((0.4, 0.35, 0.3), (0.2, 0.25, 0.3)), cm.SRGB_TO_PROPHOTO)
    result = pipeline.compress_to_gamut(modest)
    assert np.allclose(result, modest, atol=2e-3)


def test_compression_keeps_hue_and_lightness() -> None:
    wide = cm.apply_matrix(_image((0.95, 0.1, 0.0), (0.0, 0.9, 0.95)), cm.PROPHOTO_TO_XYZ)
    wide = cm.apply_matrix(wide, cm.XYZ_TO_PROPHOTO)
    before = cm.prophoto_to_oklch(wide)
    after = cm.prophoto_to_oklch(pipeline.compress_to_gamut(wide))
    assert np.allclose(after[..., 0], np.clip(before[..., 0], 0.0, 1.0), atol=2e-3)
    assert np.allclose(after[..., 2], before[..., 2], atol=2e-3)


def test_neutrals_stay_neutral_through_compression() -> None:
    greys = cm.apply_matrix(_image((0.1,) * 3, (0.5,) * 3, (0.9,) * 3), cm.SRGB_TO_PROPHOTO)
    result = cm.apply_matrix(pipeline.compress_to_gamut(greys), cm.PROPHOTO_TO_SRGB)
    for pixel in result[0]:
        assert pixel.max() - pixel.min() < 2e-3, pixel


def test_brighter_than_white_converges_on_white() -> None:
    too_bright = cm.apply_matrix(_image((3.0, 2.4, 2.1)), cm.SRGB_TO_PROPHOTO)
    result = cm.apply_matrix(pipeline.compress_to_gamut(too_bright), cm.PROPHOTO_TO_SRGB)[0, 0]
    assert result.min() > 0.9
    assert result.max() - result.min() < 0.05, result


def test_a_gradient_stays_a_gradient_after_compression() -> None:
    """Compression must not flatten a ramp into a block of one colour.

    Chroma itself is not expected to keep rising along the ramp: the sRGB
    solid narrows as lightness goes up, so a bright red simply cannot hold as
    much chroma as a darker one. What must hold is that every step stays
    distinct and lightness keeps climbing.
    """
    ramp = np.linspace(0.3, 1.0, 64, dtype=np.float32)
    colours = np.stack([ramp, ramp * 0.1, ramp * 0.05], axis=-1)[np.newaxis, ...]
    result = pipeline.compress_to_gamut(colours)

    lightness = cm.prophoto_to_oklch(result)[0, :, 0]
    assert np.all(np.diff(lightness) > 0)
    assert np.all(_is_in_srgb(result))

    steps = pipeline.to_srgb8(result)[0]
    assert len(np.unique(steps, axis=0)) == len(steps)


def test_an_srgb_colour_just_inside_the_edge_is_left_alone() -> None:
    """Touching any slider must not fade a photo's strongest colours.

    The camera puts saturated colours right up to the edge of sRGB, so a
    compression that starts far inside it pales them as soon as a JPEG
    leaves passthrough. Ninety five per cent of the way to the edge must be
    left as it was.
    """
    lightness = np.float32([0.62, 0.72, 0.5])
    hue = np.radians(np.float32([40.0, 70.0, -100.0]))
    ceiling = pipeline._max_chroma(lightness, hue)
    lch = np.stack((lightness, ceiling * np.float32(0.95), hue), axis=-1)[np.newaxis, ...]
    colours = cm.oklch_to_prophoto(lch)
    result = cm.prophoto_to_oklch(pipeline.compress_to_gamut(colours))
    assert np.allclose(result[..., 1], lch[..., 1], atol=1e-4)


# -- step 12: output ------------------------------------------------------


def test_white_and_black_land_on_the_ends_of_the_range() -> None:
    white = cm.apply_matrix(_image((1.0, 1.0, 1.0)), cm.SRGB_TO_PROPHOTO)
    black = cm.apply_matrix(_image((0.0, 0.0, 0.0)), cm.SRGB_TO_PROPHOTO)
    assert pipeline.to_srgb8(white)[0, 0].tolist() == [255, 255, 255]
    assert pipeline.to_srgb8(black)[0, 0].tolist() == [0, 0, 0]


@pytest.mark.parametrize("step", [32, 64, 128, 192, 224])
def test_an_eight_bit_grey_survives_the_output_transform(step: int) -> None:
    """A grey that came from an 8 bit value must come back as that value."""
    level = step / 255.0
    grey = cm.apply_matrix(cm.srgb_to_linear(_image((level,) * 3)), cm.SRGB_TO_PROPHOTO)
    assert pipeline.to_srgb8(grey)[0, 0].tolist() == [step, step, step]


def test_values_outside_the_range_are_clipped_not_wrapped() -> None:
    extreme = cm.apply_matrix(_image((4.0, -1.0, 0.5)), cm.SRGB_TO_PROPHOTO)
    result = pipeline.to_srgb8(extreme)
    assert result.min() >= 0 and result.max() <= 255


def test_dither_changes_the_result_without_shifting_the_average() -> None:
    flat = cm.apply_matrix(
        cm.srgb_to_linear(np.full((64, 64, 3), 0.5019608, dtype=np.float32)),
        cm.SRGB_TO_PROPHOTO,
    )
    plain = pipeline.to_srgb8(flat)
    dithered = pipeline.to_srgb8(flat, dither=True)
    assert plain.std() == 0.0
    assert dithered.mean() == pytest.approx(plain.mean(), abs=0.2)


# -- the pipeline ---------------------------------------------------------


def test_an_untouched_srgb_jpeg_comes_out_exactly_as_it_went_in() -> None:
    """PLAN.md section 6: an unedited JPEG is shown pixel for pixel."""
    pixels = np.random.default_rng(1).integers(0, 256, (16, 24, 3), dtype=np.uint8)
    photo = _jpeg_photo(pixels)
    assert pipeline.is_passthrough(photo)
    assert np.array_equal(pipeline.develop(photo), pixels)


def test_an_adobe_rgb_photo_comes_out_more_saturated_than_its_own_numbers() -> None:
    """Adobe RGB numbers shown as sRGB look washed out; converting fixes that."""
    pixels = np.zeros((1, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (200, 60, 50)
    pixels[0, 1] = (40, 160, 90)
    pixels[0, 2] = (60, 90, 200)
    developed = pipeline.develop(_jpeg_photo(pixels, ColorSpace.ADOBE_RGB))

    def chroma(image: np.ndarray) -> float:
        values = image.astype(np.int32)
        return float(np.mean(values.max(axis=-1) - values.min(axis=-1)))

    assert chroma(developed) > chroma(pixels)


def test_an_adobe_rgb_jpeg_is_converted_rather_than_passed_through() -> None:
    """Its numbers mean different colours, so they cannot be shown as sRGB."""
    pixels = np.full((8, 8, 3), (200, 90, 40), dtype=np.uint8)
    photo = _jpeg_photo(pixels, ColorSpace.ADOBE_RGB)
    assert not pipeline.is_passthrough(photo)
    assert not np.array_equal(pipeline.develop(photo), pixels)


def test_a_raw_photo_develops_to_eight_bit_srgb() -> None:
    image = np.full((8, 8, 3), 0.2, dtype=np.float32)
    result = pipeline.develop(_raw_photo(image))
    assert result.dtype == np.uint8
    assert result.shape == (8, 8, 3)


def test_a_neutral_raw_photo_develops_to_a_neutral_grey() -> None:
    """Camera neutral must stay neutral all the way to the screen."""
    image = np.full((4, 4, 3), 0.25, dtype=np.float32)
    pixel = pipeline.develop(_raw_photo(image))[0, 0]
    assert int(pixel.max()) - int(pixel.min()) <= 1, pixel


def test_the_base_curve_lifts_a_dark_linear_raw_photo() -> None:
    """Linear RAW looks dark; the base curve is what gives it its shape."""
    image = np.full((4, 4, 3), _scene_mid_grey(), dtype=np.float32)
    developed = pipeline.develop(_raw_photo(image))[0, 0, 0]
    without_curve = pipeline.to_srgb8(
        pipeline.camera_to_working_space(image, cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX))
    )[0, 0, 0]
    assert developed > 100  # middle grey lands near 118 in 8 bit sRGB
    assert developed >= without_curve - 1


def test_a_raw_photo_without_a_camera_matrix_is_refused() -> None:
    photo = DecodedPhoto(
        image=np.zeros((2, 2, 3), dtype=np.float32),
        is_raw=True,
        clipped=np.zeros((2, 2), dtype=bool),
    )
    with pytest.raises(ValueError):
        pipeline.develop(photo)


def test_developing_twice_gives_the_same_result() -> None:
    """The pipeline is pure, so it cannot depend on anything but its input."""
    photo = _raw_photo(np.full((8, 8, 3), 0.3, dtype=np.float32))
    assert np.array_equal(pipeline.develop(photo), pipeline.develop(photo))


def test_the_whole_pipeline_leaves_nothing_outside_the_output_range() -> None:
    generator = np.random.default_rng(7)
    image = generator.random((32, 32, 3), dtype=np.float32) * 2.0
    result = pipeline.develop(_raw_photo(image))
    assert result.min() >= 0 and result.max() <= 255


# -- sixteen bit output (PLAN.md 6 step 12) -------------------------------


def test_sixteen_bit_output_fills_its_range() -> None:
    white = cm.apply_matrix(_image((1.0, 1.0, 1.0)), cm.SRGB_TO_PROPHOTO)
    black = cm.apply_matrix(_image((0.0, 0.0, 0.0)), cm.SRGB_TO_PROPHOTO)
    assert pipeline.to_srgb16(white)[0, 0].tolist() == [65535] * 3
    assert pipeline.to_srgb16(black)[0, 0].tolist() == [0] * 3


def test_sixteen_bit_and_eight_bit_agree_to_within_a_step() -> None:
    generator = np.random.default_rng(11)
    image = cm.apply_matrix(
        generator.random((8, 8, 3), dtype=np.float32), cm.SRGB_TO_PROPHOTO
    )
    eight = pipeline.to_srgb8(image).astype(np.float32)
    sixteen = pipeline.to_srgb16(image).astype(np.float32) / 257.0
    assert np.abs(eight - sixteen).max() <= 1.0


def test_developing_at_sixteen_bits_gives_sixteen_bit_pixels() -> None:
    photo = _raw_photo(np.full((8, 8, 3), 0.3, dtype=np.float32))
    result = pipeline.develop(photo, bits=pipeline.SIXTEEN_BIT)
    assert result.dtype == np.uint16
    assert result.shape == (8, 8, 3)


def test_an_untouched_jpeg_written_at_sixteen_bits_keeps_its_values() -> None:
    """There is no more information to be had, so each step is just scaled."""
    pixels = np.random.default_rng(3).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    result = pipeline.develop(_jpeg_photo(pixels), bits=pipeline.SIXTEEN_BIT)
    assert result.dtype == np.uint16
    assert np.array_equal(result // 257, pixels)
    assert result.max() == int(pixels.max()) * 257


def test_sixteen_bit_output_is_never_dithered() -> None:
    """65536 steps cannot band, so the noise would be for nothing."""
    flat = np.full((32, 32, 3), 0.25, dtype=np.float32)
    photo = _raw_photo(flat)
    once = pipeline.develop(photo, dither=True, bits=pipeline.SIXTEEN_BIT)
    again = pipeline.develop(photo, dither=True, bits=pipeline.SIXTEEN_BIT)
    assert np.array_equal(once, again)


def test_an_eight_bit_render_is_unchanged_by_the_depth_being_an_option() -> None:
    photo = _raw_photo(np.full((8, 8, 3), 0.3, dtype=np.float32))
    assert np.array_equal(
        pipeline.develop(photo), pipeline.develop(photo, bits=pipeline.EIGHT_BIT)
    )


def test_a_depth_that_is_not_offered_is_refused() -> None:
    photo = _raw_photo(np.full((4, 4, 3), 0.3, dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.develop(photo, bits=12)


def test_render_passes_the_depth_through_to_the_output() -> None:
    photo = _raw_photo(np.full((8, 8, 3), 0.3, dtype=np.float32))
    result = pipeline.render(photo, None, None, bits=pipeline.SIXTEEN_BIT)
    assert result.dtype == np.uint16


# -- one window of the photo (PLAN.md 3.3 and section 4) -------------------


def _busy_raw(height: int = 240, width: int = 360) -> DecodedPhoto:
    """Return a RAW photo with detail everywhere and a clipped corner."""
    rows = np.linspace(0.05, 0.9, height, dtype=np.float32)[:, None]
    columns = np.linspace(0.05, 0.9, width, dtype=np.float32)[None, :]
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:, :, 0] = rows * columns * 1.2
    image[:, :, 1] = rows
    image[:, :, 2] = columns
    clipped = np.zeros((height, width), dtype=bool)
    clipped[: height // 8, : width // 8] = True
    image[: height // 8, : width // 8] = 1.0
    return _raw_photo(image, clipped)


def _everything_turned_up() -> EditState:
    state = EditState(is_raw=True)
    for name, value in (
        ("exposure", 0.4),
        ("contrast", 25.0),
        ("highlights", -60.0),
        ("shadows", 45.0),
        ("blacks", -20.0),
        ("whites", 15.0),
        ("vignette", -70.0),
        ("vibrance", 30.0),
        ("saturation", 10.0),
    ):
        state = state.with_value(name, value)
    return state


def test_a_window_comes_out_as_the_whole_photo_would_have() -> None:
    """The one promise the zoomed in preview has to keep (PLAN.md 3.3)."""
    photo = _busy_raw()
    state = _everything_turned_up()
    frame = photo.size

    asked = geometry.Window.around(frame, (150.0, 60.0, 290.0, 190.0))
    around = asked.grown(pipeline.blur_margin(photo, state, frame))
    part = geometry.window_of(photo, around)

    windowed = around.trim(pipeline.develop(part, state, window=around), asked)
    whole = pipeline.develop(photo, state)[
        asked.top : asked.bottom, asked.left : asked.right
    ]

    assert windowed.shape == whole.shape
    difference = np.abs(windowed.astype(int) - whole.astype(int))
    # One step of 255 is the rounding of the blur being worked out over fewer
    # pixels; nothing may be off by more than that, and almost nothing is.
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.01


def test_a_window_told_nothing_of_its_frame_comes_out_wrong() -> None:
    """Why develop() takes a window at all, written down as a test."""
    photo = _busy_raw()
    state = _everything_turned_up()

    asked = geometry.Window.around(photo.size, (150.0, 60.0, 290.0, 190.0))
    part = geometry.window_of(photo, asked)

    on_its_own = pipeline.develop(part, state)
    whole = pipeline.develop(photo, state)[
        asked.top : asked.bottom, asked.left : asked.right
    ]

    assert np.abs(on_its_own.astype(int) - whole.astype(int)).max() > 8


def test_the_vignette_of_a_window_is_measured_from_the_middle_of_the_frame() -> None:
    """A window near the corner is darkened like a corner, not like a photo."""
    photo = _busy_raw()
    state = EditState(is_raw=True).with_value("vignette", -100.0)
    frame = photo.size

    corner = geometry.Window.around(frame, (0.0, 0.0, 60.0, 60.0))
    middle = geometry.Window.around(frame, (150.0, 90.0, 210.0, 150.0))

    darkened = pipeline.develop(geometry.window_of(photo, corner), state, window=corner)
    untouched = pipeline.develop(geometry.window_of(photo, middle), state, window=middle)
    plain = pipeline.develop(photo, EditState(is_raw=True))

    assert darkened.mean() < plain[0:60, 0:60].mean()
    assert untouched.mean() == pytest.approx(plain[90:150, 150:210].mean(), abs=1.0)


def test_a_window_of_the_whole_frame_develops_exactly_as_before() -> None:
    photo = _busy_raw(height=48, width=64)
    state = _everything_turned_up()

    assert np.array_equal(
        pipeline.develop(photo, state),
        pipeline.develop(photo, state, window=geometry.Window.whole(photo.size)),
    )


def test_a_window_that_does_not_fit_the_pixels_it_was_given_is_refused() -> None:
    photo = _busy_raw(height=48, width=64)

    with pytest.raises(ValueError):
        pipeline.develop(photo, window=geometry.Window((64, 48), 0, 0, 32, 24))


def test_the_margin_is_the_reach_of_whichever_blur_is_at_work() -> None:
    photo = _busy_raw(height=48, width=64)
    frame = (6000, 4000)
    resting = EditState(is_raw=True)

    #: Step 4 softens the clipped mask on every RAW photo, however it is set.
    assert pipeline.blur_margin(photo, resting, frame) == pytest.approx(
        np.ceil(pipeline.BLUR_REACH_SIGMAS * pipeline.CLIPPED_BLUR_FRACTION * 6000)
    )
    # Step 7 reaches much further, and only when it is asked for.
    lifted = resting.with_value("shadows", 30.0)
    assert pipeline.blur_margin(photo, lifted, frame) == pytest.approx(
        np.ceil(pipeline.BLUR_REACH_SIGMAS * pipeline.LOCAL_BRIGHTNESS_BLUR * 6000)
    )
    assert pipeline.blur_margin(photo, lifted, frame) > pipeline.blur_margin(
        photo, lifted, (600, 400)
    )


def test_a_jpeg_at_rest_needs_no_margin_at_all() -> None:
    """Nothing blurs, so nothing outside the window can reach into it."""
    photo = _jpeg_photo(np.full((40, 60, 3), 120, dtype=np.uint8))

    assert pipeline.blur_margin(photo, EditState(is_raw=False), (6000, 4000)) == 0


def test_an_untouched_jpeg_passes_a_window_through_pixel_for_pixel() -> None:
    """PLAN.md section 6: the original pixels, zoomed in as well."""
    pixels = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
    photo = _jpeg_photo(pixels)
    window = geometry.Window((60, 40), left=12, top=8, right=44, bottom=32)

    developed = pipeline.develop(geometry.window_of(photo, window), window=window)

    assert np.array_equal(developed, pixels[8:32, 12:44])


# -- the local brightness, worked out over the whole frame -----------------


def test_a_window_given_the_frames_local_brightness_needs_almost_no_margin() -> None:
    """What lets a zoomed in preview keep up (PLAN.md 3.3)."""
    photo = _busy_raw()
    state = _everything_turned_up()
    frame = photo.size

    field = pipeline.surroundings_field(photo, state)
    assert field is not None
    assert field.shape == (frame[1], frame[0])

    asked = geometry.Window.around(frame, (150.0, 60.0, 290.0, 190.0))
    around = asked.grown(
        pipeline.blur_margin(photo, state, frame, surroundings_apart=True)
    )
    windowed = around.trim(
        pipeline.develop(
            geometry.window_of(photo, around),
            state,
            window=around,
            surroundings=around.take(field),
        ),
        asked,
    )
    whole = pipeline.develop(photo, state)[
        asked.top : asked.bottom, asked.left : asked.right
    ]

    difference = np.abs(windowed.astype(int) - whole.astype(int))
    assert difference.max() <= 1
    # And on a real frame the margin it needs is a fraction of step 7's
    # reach: nineteen pixels of mask against two hundred and forty one.
    whole_frame = (6024, 4024)
    assert pipeline.blur_margin(
        photo, state, whole_frame, surroundings_apart=True
    ) * 8 < pipeline.blur_margin(photo, state, whole_frame)


def test_the_local_brightness_of_a_smaller_copy_says_the_same_thing() -> None:
    """It is a blur a hundredth of the frame wide, so the proxy will do."""
    photo = _busy_raw(height=480, width=720)
    state = _everything_turned_up()
    small = _busy_raw(height=120, width=180)

    whole = pipeline.surroundings_field(photo, state)
    proxy = pipeline.surroundings_field(small, state)
    assert whole is not None and proxy is not None

    stretched = cv2.resize(proxy, (720, 480), interpolation=cv2.INTER_CUBIC)
    # Both are in stops from middle grey, so this is a tenth of a stop.
    assert float(np.abs(whole - stretched).mean()) < 0.1


def test_nothing_is_worked_out_when_step_seven_is_at_rest() -> None:
    photo = _busy_raw(height=48, width=64)
    assert pipeline.surroundings_field(photo, EditState(is_raw=True)) is None


def test_the_local_brightness_outlives_every_slider_after_it() -> None:
    """Moving the contrast must not throw the kept field away."""
    state = _everything_turned_up()

    assert pipeline.surroundings_key(
        state.with_value("contrast", -40.0)
    ) == pipeline.surroundings_key(state.with_value("vignette", 80.0))
    assert pipeline.surroundings_key(
        state.with_value("exposure", 1.5)
    ) != pipeline.surroundings_key(state)
    assert pipeline.surroundings_key(
        state.with_value("shadows", 0.0).with_value("highlights", 0.0)
    ) != pipeline.surroundings_key(state)


def test_a_local_brightness_of_the_wrong_shape_is_refused() -> None:
    photo = _busy_raw(height=48, width=64)
    state = _everything_turned_up()

    with pytest.raises(ValueError):
        pipeline.develop(photo, state, surroundings=np.zeros((24, 32), dtype=np.float32))


def test_the_whole_photo_is_the_same_whoever_works_out_the_surroundings() -> None:
    """Handing develop() the field it would have made changes nothing."""
    photo = _busy_raw(height=240, width=360)
    state = _everything_turned_up()

    field = pipeline.surroundings_field(photo, state)

    assert np.array_equal(
        pipeline.develop(photo, state),
        pipeline.develop(photo, state, surroundings=field),
    )


# -- step 10: the base chroma of a RAW photo -------------------------------


def test_a_raw_photo_starts_with_its_base_chroma() -> None:
    assert pipeline.base_chroma(True) == pipeline.RAW_BASE_CHROMA
    assert pipeline.RAW_BASE_CHROMA > 1.0


def test_a_jpeg_gets_no_base_chroma() -> None:
    """A JPEG already carries the camera's colour."""
    assert pipeline.base_chroma(False) == 1.0
    assert pipeline.base_chroma(False, 0.0) == 1.0


@pytest.mark.parametrize(
    ("strength", "share"), [(0.0, 1.0), (50.0, 0.5), (100.0, 0.0), (150.0, 0.0)]
)
def test_a_filter_takes_over_from_the_base_chroma(strength: float, share: float) -> None:
    """A filter was measured all the way to the camera's colour."""
    expected = 1.0 + (pipeline.RAW_BASE_CHROMA - 1.0) * share
    assert pipeline.base_chroma(True, strength) == pytest.approx(expected)


def test_the_base_chroma_moves_only_chroma() -> None:
    colours = cm.apply_matrix(
        _image((0.6, 0.3, 0.2), (0.2, 0.4, 0.6), (0.3, 0.5, 0.2)), cm.SRGB_TO_PROPHOTO
    )
    before = cm.prophoto_to_oklch(colours)
    after = cm.prophoto_to_oklch(
        pipeline.vibrance_and_saturation(colours, 0.0, 0.0, pipeline.RAW_BASE_CHROMA)
    )
    assert np.allclose(after[..., 0], before[..., 0], atol=1e-4)
    assert np.allclose(after[..., 2], before[..., 2], atol=1e-4)
    assert np.allclose(after[..., 1], before[..., 1] * pipeline.RAW_BASE_CHROMA, rtol=1e-3)


def test_saturation_multiplies_on_top_of_the_base_chroma() -> None:
    colours = cm.apply_matrix(_image((0.6, 0.3, 0.2)), cm.SRGB_TO_PROPHOTO)
    before = float(cm.prophoto_to_oklch(colours)[0, 0, 1])
    after = float(
        cm.prophoto_to_oklch(pipeline.vibrance_and_saturation(colours, 0.0, 50.0, 1.35))[0, 0, 1]
    )
    assert after == pytest.approx(before * 1.35 * 1.5, rel=2e-3)


def test_saturation_at_its_lowest_still_gives_black_and_white_on_a_raw() -> None:
    colours = cm.apply_matrix(_image((0.6, 0.3, 0.2), (0.2, 0.4, 0.6)), cm.SRGB_TO_PROPHOTO)
    result = pipeline.vibrance_and_saturation(colours, 0.0, -100.0, pipeline.RAW_BASE_CHROMA)
    assert float(cm.prophoto_to_oklch(result)[..., 1].max()) < 1e-3


def test_vibrance_judges_a_raw_photo_by_its_colour_after_the_base_chroma() -> None:
    """With vibrance moved, the slow path must agree with the fast one at zero."""
    colours = cm.apply_matrix(
        _image((0.6, 0.3, 0.2), (0.2, 0.4, 0.6), (0.45, 0.42, 0.4)), cm.SRGB_TO_PROPHOTO
    )
    fast = pipeline.vibrance_and_saturation(colours, 0.0, 20.0, 1.35)
    lab = cm.prophoto_to_oklab(colours)
    lab[..., 1:] *= 1.35 * 1.2
    assert np.allclose(fast, cm.oklab_to_prophoto(lab), atol=1e-5)
    # Vibrance on the boosted colour reads its chroma after the boost.
    boosted = cm.scale_oklab_chroma(colours, 1.35)
    assert np.allclose(
        pipeline.vibrance_and_saturation(colours, 40.0, 0.0, 1.35),
        pipeline.vibrance_and_saturation(boosted, 40.0, 0.0),
        atol=1e-5,
    )


def _mean_chroma(encoded: np.ndarray) -> float:
    linear = cm.srgb_to_linear(encoded.astype(np.float32) / 255.0)
    lab = cm.srgb_linear_to_oklab(linear)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def test_a_raw_photo_at_rest_is_more_colourful_than_its_plain_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = np.linspace(0.02, 0.06, 60, dtype=np.float32)[:, None]
    columns = np.linspace(0.9, 1.1, 80, dtype=np.float32)[None, :]
    image = np.stack(
        [rows * columns, np.broadcast_to(rows, (60, 80)), rows / columns], axis=-1
    )
    photo = _raw_photo(image)
    state = EditState.for_photo(True)
    rested = pipeline.develop(photo, state)
    base = pipeline.RAW_BASE_CHROMA
    monkeypatch.setattr(pipeline, "RAW_BASE_CHROMA", 1.0)
    plain = pipeline.develop(photo, state)
    assert _mean_chroma(rested) == pytest.approx(_mean_chroma(plain) * base, rel=0.05)


# -- step 9.7: the film filter --------------------------------------------


def _a_look(identifier: str = "test") -> filters.Look:
    """Return a look that is felt in all three of its parts."""
    knots = np.linspace(0.0, 1.0, 17)
    return filters.Look(
        identifier=identifier,
        curves=(
            tuple(np.clip(knots * 1.15, 0.0, 1.0).tolist()),
            tuple(knots.tolist()),
            tuple((knots**1.25).tolist()),
        ),
        chroma=(-0.15, 0.08, 0.03),
        hue=(0.03, -0.04, 0.02),
        lightness=(0.02, 0.03, -0.02),
    )


def test_a_photo_wearing_no_filter_is_developed_exactly_as_before() -> None:
    """Adding the step may not move a pixel of a photo that does not use it."""
    photo = _busy_raw()
    state = _everything_turned_up()

    assert np.array_equal(
        pipeline.develop(photo, state), pipeline.develop(photo, state, look=_a_look())
    )


def test_a_look_is_only_worn_when_the_state_names_it() -> None:
    """Handing a look over is not the same as putting it on."""
    photo = _busy_raw()
    plain = EditState.for_photo(True)

    assert np.array_equal(
        pipeline.develop(photo, plain),
        pipeline.develop(photo, plain, look=_a_look()),
    )


def test_naming_a_filter_without_handing_it_over_is_an_error() -> None:
    photo = _busy_raw()
    state = EditState.for_photo(True).with_filter("test")

    with pytest.raises(ValueError, match="test"):
        pipeline.develop(photo, state)


def test_a_filter_changes_the_photo() -> None:
    photo = _busy_raw()
    state = EditState.for_photo(True)
    look = _a_look()

    assert not np.array_equal(
        pipeline.develop(photo, state),
        pipeline.develop(photo, state.with_filter("test"), look=look),
    )


def test_the_strength_of_a_filter_takes_it_back_towards_nothing() -> None:
    photo = _busy_raw()
    state = EditState.for_photo(True).with_filter("test")
    look = _a_look()

    plain = pipeline.develop(photo, state.with_filter(""))
    whole = pipeline.develop(photo, state, look=look).astype(np.int32)
    half = pipeline.develop(photo, state.with_value("filter_strength", 50.0), look=look)
    none = pipeline.develop(photo, state.with_value("filter_strength", 0.0), look=look)

    distance = np.abs(whole - plain.astype(np.int32)).mean()
    assert 0.0 < np.abs(half.astype(np.int32) - plain.astype(np.int32)).mean() < distance
    assert np.array_equal(none, plain)


def test_a_filter_comes_before_saturation() -> None:
    """The colour sliders work on what the filter leaves, not the other way."""
    photo = _busy_raw()
    look = _a_look()
    state = EditState.for_photo(True).with_filter("test")

    both = pipeline.develop(photo, state.with_value("saturation", 60.0), look=look)
    filtered = pipeline.develop(photo, state, look=look)
    assert not np.array_equal(both, filtered)


def test_a_filter_takes_a_jpeg_out_of_passthrough() -> None:
    """An untouched JPEG is passed through; one wearing a filter is not."""
    encoded = np.full((4, 6, 3), 180, dtype=np.uint8)
    photo = _jpeg_photo(encoded)
    state = EditState.for_photo(False)

    assert pipeline.is_passthrough(photo, state)
    assert not pipeline.is_passthrough(photo, state.with_filter("test"))
    assert np.array_equal(pipeline.develop(photo, state), encoded)
    assert not np.array_equal(
        pipeline.develop(photo, state.with_filter("test"), look=_a_look()), encoded
    )


def test_a_window_wearing_a_filter_comes_out_as_the_whole_photo_would_have() -> None:
    """Zooming in on a filtered photo must show what the whole one shows."""
    photo = _busy_raw()
    state = _everything_turned_up().with_filter("test")
    look = _a_look()
    frame = photo.size

    asked = geometry.Window.around(frame, (150.0, 60.0, 290.0, 190.0))
    around = asked.grown(pipeline.blur_margin(photo, state, frame))
    part = geometry.window_of(photo, around)

    windowed = around.trim(pipeline.develop(part, state, window=around, look=look), asked)
    whole = pipeline.develop(photo, state, look=look)[
        asked.top : asked.bottom, asked.left : asked.right
    ]

    # The same tolerance as without a filter: the blur of step 7 worked out
    # over fewer pixels, and nothing the filter adds on top of it.
    difference = np.abs(windowed.astype(int) - whole.astype(int))
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.01
