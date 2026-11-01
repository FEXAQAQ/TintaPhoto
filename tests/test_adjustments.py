"""Tests for the twelve adjustments (PLAN.md sections 2.2 and 6).

These are the properties the acceptance list of stage 2 asks for: exposure
doubles the light, saturation at its lowest gives a black and white photo,
nothing shifts hue, and a big tone change does not draw a halo along an edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import color_management as cm
from core import pipeline
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
#: What that camera recorded for one of the sample photos.
CAMERA_SHOT_MULTIPLIERS = (2.3359, 1.0, 1.6406)


def _working(*colours: tuple[float, float, float]) -> np.ndarray:
    """Return sRGB colours as linear working space values."""
    return cm.apply_matrix(
        cm.srgb_to_linear(np.array([list(colours)], dtype=np.float32)), cm.SRGB_TO_PROPHOTO
    )


def _grey(level: float, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    return np.full((*size, 3), level, dtype=np.float32)


def _raw_photo(image: np.ndarray) -> DecodedPhoto:
    return DecodedPhoto(
        image=image.astype(np.float32),
        is_raw=True,
        clipped=np.zeros(image.shape[:2], dtype=bool),
        camera_to_xyz=cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX),
        rgb_xyz_matrix=CAMERA_RGB_XYZ_MATRIX,
        camera_whitebalance=CAMERA_SHOT_MULTIPLIERS,
    )


def _jpeg_photo(encoded: np.ndarray) -> DecodedPhoto:
    linear = cm.srgb_to_linear(encoded.astype(np.float32) / 255.0)
    return DecodedPhoto(
        image=cm.apply_matrix(linear, cm.SRGB_TO_PROPHOTO),
        is_raw=False,
        clipped=np.zeros(encoded.shape[:2], dtype=bool),
        color_space=ColorSpace.SRGB,
        encoded=encoded.astype(np.uint8),
    )


def _luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.asarray(cm.PROPHOTO_LUMINANCE, dtype=np.float32)


# -- step 6: exposure -----------------------------------------------------


def test_one_stop_of_exposure_doubles_the_light() -> None:
    """PLAN.md stage 2 acceptance: +1 EV is twice the linear value."""
    image = _grey(0.18)
    brightened = pipeline.exposure(image, 1.0)
    assert float(brightened[0, 0, 0]) == pytest.approx(0.36, rel=1e-5)


@pytest.mark.parametrize("stops", [-2.0, -0.5, 0.5, 2.0, 5.0])
def test_exposure_is_a_power_of_two(stops: float) -> None:
    image = _grey(0.1)
    result = pipeline.exposure(image, stops)
    assert float(result[0, 0, 0]) == pytest.approx(0.1 * 2.0**stops, rel=1e-5)


def test_exposure_can_be_taken_back() -> None:
    image = _working((0.4, 0.3, 0.2))
    there_and_back = pipeline.exposure(pipeline.exposure(image, 1.5), -1.5)
    assert np.allclose(there_and_back, image, atol=1e-6)


def test_no_exposure_changes_nothing() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.exposure(image, 0.0) is image


def test_exposure_does_not_move_the_colour() -> None:
    image = _working((0.6, 0.3, 0.15))
    before = cm.prophoto_to_oklch(image)[0, 0]
    after = cm.prophoto_to_oklch(pipeline.exposure(image, 1.0))[0, 0]
    assert float(after[2]) == pytest.approx(float(before[2]), abs=1e-4)


# -- step 7: highlights and shadows ---------------------------------------


def test_lifting_the_shadows_brightens_the_dark_parts_most() -> None:
    dark = _grey(0.01)
    light = _grey(0.5)
    dark_gain = float(
        (_luminance(pipeline.highlights_and_shadows(dark, 0.0, 100.0)) / _luminance(dark)).mean()
    )
    light_gain = float(
        (_luminance(pipeline.highlights_and_shadows(light, 0.0, 100.0)) / _luminance(light)).mean()
    )
    cap = 2.0**pipeline.TONE_REGION_MAX_EV
    assert dark_gain > light_gain
    # Most of the promised lift reaches the shadows, and none of it is exceeded.
    assert 0.8 * cap < dark_gain <= cap
    assert light_gain < 1.2  # the midtones are mostly left alone


def test_pulling_the_highlights_darkens_the_bright_parts_most() -> None:
    bright = _grey(2.0)
    middle = _grey(0.18)
    bright_gain = float(
        (_luminance(pipeline.highlights_and_shadows(bright, -100.0, 0.0)) / _luminance(bright)).mean()
    )
    middle_gain = float(
        (_luminance(pipeline.highlights_and_shadows(middle, -100.0, 0.0)) / _luminance(middle)).mean()
    )
    floor = 2.0**-pipeline.TONE_REGION_MAX_EV
    assert bright_gain < middle_gain
    assert floor <= bright_gain < 0.6
    assert middle_gain > 0.8  # the midtones are mostly left alone


@pytest.mark.parametrize("highlights", [-100.0, 0.0, 100.0])
@pytest.mark.parametrize("shadows", [-100.0, 0.0, 100.0])
def test_the_tone_sliders_never_exceed_their_documented_reach(
    highlights: float, shadows: float
) -> None:
    """Whatever the photo, neither slider may move a region further than its cap."""
    generator = np.random.default_rng(5)
    image = (generator.random((64, 64, 3), dtype=np.float32) * 3.0).astype(np.float32)
    result = pipeline.highlights_and_shadows(image, highlights, shadows)
    stops = np.log2(np.maximum(_luminance(result), 1e-9) / np.maximum(_luminance(image), 1e-9))
    assert float(np.abs(stops).max()) <= pipeline.TONE_REGION_MAX_EV + 1e-3


def test_neither_slider_at_zero_changes_anything() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.highlights_and_shadows(image, 0.0, 0.0) is image


def _edge_gain() -> np.ndarray:
    """Return the gain applied across a hard black to white edge."""
    image = np.zeros((200, 400, 3), dtype=np.float32)
    image[:, :200] = 0.01
    image[:, 200:] = 1.00
    result = pipeline.highlights_and_shadows(image, -100.0, 100.0)
    return (_luminance(result) / _luminance(image))[100]


def test_a_big_tone_change_is_spread_out_rather_than_tracing_the_edge() -> None:
    """PLAN.md stage 2 acceptance: no obvious halo along a high contrast edge.

    What makes a halo obvious is a correction that follows the edge exactly,
    leaving a bright line hugging it. Judging each pixel by a blurred version
    of its surroundings is what spreads the change out instead, so what is
    checked here is that the gain changes gently from one pixel to the next.
    Without the blur the gain steps by more than three quarters in one pixel.
    """
    gain = _edge_gain()
    steepest = float(np.abs(np.diff(gain)).max())
    assert steepest < 0.40, f"the correction traces the edge: {steepest:.2f} per pixel"


def test_a_big_tone_change_does_not_overshoot_badly() -> None:
    """The spreading must not itself lift the bright side near the edge much."""
    gain = _edge_gain()
    far_side = float(gain[-1])
    just_inside = float(gain[205:240].max())
    assert (just_inside - far_side) / far_side < 0.20


def test_the_tone_adjustment_keeps_the_colour() -> None:
    image = _working((0.5, 0.25, 0.1))
    before = cm.prophoto_to_oklch(image)[0, 0]
    after = cm.prophoto_to_oklch(pipeline.highlights_and_shadows(image, -80.0, 80.0))[0, 0]
    assert float(after[2]) == pytest.approx(float(before[2]), abs=1e-3)


# -- step 8: contrast -----------------------------------------------------


def test_contrast_leaves_middle_grey_where_it_is() -> None:
    """Middle grey is the pivot, so it must not move at all."""
    image = _grey(pipeline.TONE_MID_OUTPUT)
    for amount in (-100.0, -50.0, 50.0, 100.0):
        result = pipeline.contrast(image, amount)
        assert float(result[0, 0, 0]) == pytest.approx(pipeline.TONE_MID_OUTPUT, rel=1e-4)


def test_more_contrast_pushes_tones_apart() -> None:
    dark = _grey(0.05)
    light = _grey(0.6)
    assert float(pipeline.contrast(dark, 100.0)[0, 0, 0]) < 0.05
    assert float(pipeline.contrast(light, 100.0)[0, 0, 0]) > 0.6


def test_less_contrast_draws_tones_together() -> None:
    dark = _grey(0.05)
    light = _grey(0.6)
    assert float(pipeline.contrast(dark, -100.0)[0, 0, 0]) > 0.05
    assert float(pipeline.contrast(light, -100.0)[0, 0, 0]) < 0.6


def test_contrast_is_monotonic() -> None:
    ramp = np.linspace(0.001, 4.0, 256, dtype=np.float32)
    image = np.stack([ramp] * 3, axis=-1)[np.newaxis, ...]
    result = pipeline.contrast(image, 100.0)[0, :, 0]
    assert np.all(np.diff(result) > 0)


def test_no_contrast_changes_nothing() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.contrast(image, 0.0) is image


def test_contrast_keeps_the_hue() -> None:
    image = _working((0.6, 0.3, 0.15))
    before = cm.prophoto_to_oklch(image)[0, 0]
    after = cm.prophoto_to_oklch(pipeline.contrast(image, 100.0))[0, 0]
    assert float(after[2]) == pytest.approx(float(before[2]), abs=1e-3)


# -- step 9.5: black and white points -------------------------------------


def test_neither_point_at_zero_changes_anything() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.black_and_white_points(image, 0.0, 0.0) is image


def test_crushing_the_blacks_takes_the_bottom_of_the_range_to_black() -> None:
    """Blacks at its lowest is a black point: below it there is nothing."""
    image = _grey(pipeline.BLACK_POINT_RANGE / 2)
    assert float(pipeline.black_and_white_points(image, -100.0, 0.0).max()) == 0.0


def test_lifting_the_blacks_turns_pure_black_into_a_dark_grey() -> None:
    result = pipeline.black_and_white_points(_grey(0.0), 100.0, 0.0)
    assert float(result.min()) > 0.0
    assert float(result.max()) < pipeline.TONE_MID_OUTPUT


def test_the_black_point_leaves_white_where_it_is() -> None:
    """Moving one end of the range must not move the other."""
    for blacks in (-100.0, -50.0, 50.0, 100.0):
        result = pipeline.set_black_point(_grey(1.0), blacks)
        assert float(result.mean()) == pytest.approx(1.0, abs=1e-5), blacks


def test_the_black_point_hardly_touches_middle_grey() -> None:
    """A black point is an offset, not an exposure: the midtones stay put."""
    grey = _grey(pipeline.TONE_MID_OUTPUT)
    for blacks in (-100.0, 100.0):
        moved = float(pipeline.set_black_point(grey, blacks).mean())
        stops = np.log2(moved / pipeline.TONE_MID_OUTPUT)
        assert abs(stops) < 0.1, (blacks, stops)


def test_opening_the_whites_sends_the_brightest_tones_past_white() -> None:
    """Above the white point there is nothing left but white (6 step 11)."""
    two_stops_up = pipeline.TONE_MID_OUTPUT * 2.0**pipeline.WHITE_PIVOT_EV
    assert float(pipeline.set_white_point(_grey(two_stops_up), 100.0).min()) > 1.0


def test_closing_the_whites_brings_a_blown_highlight_back() -> None:
    assert float(pipeline.set_white_point(_grey(1.0), -100.0).max()) < 1.0


def test_the_white_point_leaves_the_midtones_where_they_are() -> None:
    """A gain over the whole photo would be exposure, not a white point."""
    grey = _grey(pipeline.TONE_MID_OUTPUT)
    for whites in (-100.0, 100.0):
        moved = float(pipeline.set_white_point(grey, whites).mean())
        stops = np.log2(moved / pipeline.TONE_MID_OUTPUT)
        assert abs(stops) < 0.1, (whites, stops)


def test_the_white_point_leaves_black_alone() -> None:
    assert float(pipeline.set_white_point(_grey(0.0), 100.0).max()) == 0.0


def test_the_white_point_reaches_further_out_than_the_highlights_slider() -> None:
    """The two must not turn out to be the same control twice."""
    assert pipeline.WHITE_PIVOT_EV > pipeline.HIGHLIGHT_PIVOT_EV
    assert pipeline.WHITE_FALLOFF_PER_EV > pipeline.REGION_FALLOFF_PER_EV


@pytest.mark.parametrize("whites", [-100.0, 0.0, 100.0])
def test_the_white_point_never_exceeds_its_documented_reach(whites: float) -> None:
    generator = np.random.default_rng(7)
    image = (generator.random((64, 64, 3), dtype=np.float32) * 3.0).astype(np.float32)
    result = pipeline.set_white_point(image, whites)
    stops = np.log2(np.maximum(_luminance(result), 1e-9) / np.maximum(_luminance(image), 1e-9))
    assert float(np.abs(stops).max()) <= pipeline.WHITE_POINT_MAX_EV + 1e-3


@pytest.mark.parametrize(
    "blacks, whites",
    [(-100.0, -100.0), (-100.0, 100.0), (100.0, -100.0), (100.0, 100.0)],
)
def test_both_points_are_monotonic(blacks: float, whites: float) -> None:
    """Whatever the two are set to, brighter must never come out darker."""
    values = np.linspace(0.0, 4.0, 512, dtype=np.float32)
    ramp = np.stack([values] * 3, axis=-1)[np.newaxis, ...]
    result = pipeline.black_and_white_points(ramp, blacks, whites)[0, :, 0]
    assert np.all(np.diff(result) >= -1e-7)


@pytest.mark.parametrize(
    "blacks, whites",
    [(-100.0, -100.0), (-100.0, 100.0), (100.0, -100.0), (100.0, 100.0), (40.0, -40.0)],
)
def test_neither_point_puts_a_cast_on_a_neutral_grey(blacks: float, whites: float) -> None:
    """Both ends must leave a grey grey, at every brightness (PLAN.md 3.5)."""
    for level in (0.0, 0.02, 0.18, 0.6, 1.0):
        result = pipeline.black_and_white_points(_grey(level), blacks, whites)
        assert float(result.max() - result.min()) < 1e-6, level


def test_the_white_point_keeps_the_hue() -> None:
    """The brightest channel is what moves; the others follow it."""
    image = _working((0.9, 0.6, 0.3))
    before = cm.prophoto_to_oklch(image)[0, 0]
    after = cm.prophoto_to_oklch(pipeline.set_white_point(image, 100.0))[0, 0]
    assert float(after[2]) == pytest.approx(float(before[2]), abs=1e-3)


def test_the_two_points_are_a_pair_of_sliders_not_a_curve() -> None:
    """Each end answers to its own slider and leaves the other alone."""
    bright = pipeline.TONE_MID_OUTPUT * 2.0**pipeline.WHITE_PIVOT_EV
    only_blacks = pipeline.black_and_white_points(_grey(bright), -100.0, 0.0)
    both = pipeline.black_and_white_points(_grey(bright), -100.0, 100.0)
    assert float(both.mean()) > float(only_blacks.mean())


# -- step 9.6: vignette ---------------------------------------------------


def _corners(plane: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(plane[0, 0]),
        float(plane[0, -1]),
        float(plane[-1, 0]),
        float(plane[-1, -1]),
    )


def test_no_vignette_changes_nothing() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.set_vignette(image, 0.0, 50.0) is image


def test_the_midpoint_alone_changes_nothing() -> None:
    """A size with no amount behind it is not an adjustment."""
    image = _grey(0.4, size=(32, 32))
    for midpoint in (0.0, 25.0, 100.0):
        assert pipeline.set_vignette(image, 0.0, midpoint) is image


def test_a_vignette_darkens_the_corners_and_leaves_the_middle_alone() -> None:
    factor = pipeline.vignette_factor((64, 64), -100.0, 50.0)
    assert float(factor[32, 32]) == pytest.approx(1.0, abs=1e-6)
    assert all(corner < 0.5 for corner in _corners(factor))


def test_a_bright_vignette_lifts_the_corners() -> None:
    factor = pipeline.vignette_factor((64, 64), 100.0, 50.0)
    assert float(factor[32, 32]) == pytest.approx(1.0, abs=1e-6)
    assert all(corner > 2.0 for corner in _corners(factor))


def test_the_corner_moves_by_the_documented_amount() -> None:
    """The ends of the slider are worth VIGNETTE_MAX_EV, and no more."""
    for amount in (-100.0, -40.0, 40.0, 100.0):
        factor = pipeline.vignette_factor((64, 96), amount, 50.0)
        stops = np.log2(np.asarray(_corners(factor)))
        wanted = pipeline.VIGNETTE_MAX_EV * amount / 100.0
        assert float(np.abs(stops - wanted).max()) < 0.02, amount
        assert float(np.abs(np.log2(factor)).max()) <= abs(wanted) + 1e-3, amount


def test_the_vignette_is_the_same_in_every_corner() -> None:
    factor = pipeline.vignette_factor((48, 72), -70.0, 40.0)
    assert float(np.ptp(np.asarray(_corners(factor)))) < 1e-6


def test_a_higher_midpoint_leaves_more_of_the_frame_alone() -> None:
    reach = [
        float((pipeline.vignette_factor((64, 64), -100.0, midpoint) < 0.99).mean())
        for midpoint in (0.0, 25.0, 50.0, 75.0)
    ]
    assert reach == sorted(reach, reverse=True)
    assert reach[0] > reach[-1]


def test_the_midpoint_at_its_highest_still_has_a_transition() -> None:
    """Nothing divides by zero at the end of the slider, and nothing jumps."""
    factor = pipeline.vignette_factor((64, 64), -100.0, 100.0)
    assert np.all(np.isfinite(factor))
    assert float(factor[32, 32]) == pytest.approx(1.0, abs=1e-6)
    assert all(corner < 1.0 for corner in _corners(factor))


def test_the_falloff_arrives_and_leaves_without_an_edge() -> None:
    """It begins and ends with a slope of zero, so neither end shows."""
    size = 512
    factor = pipeline.vignette_factor((size, size), -100.0, 30.0)
    # Along the diagonal of a square frame the radius runs evenly from 0 in
    # the middle to 1 in the corner, so this is the falloff itself.
    diagonal = np.log2(np.diagonal(factor)[size // 2 :])
    slope = np.abs(np.diff(diagonal))
    assert float(slope[0]) < float(slope.max()) / 10.0
    assert float(slope[-1]) < float(slope.max()) / 10.0


def test_the_vignette_only_grows_outwards() -> None:
    row = pipeline.vignette_factor((2, 512), -80.0, 40.0)[0, 256:]
    assert np.all(np.diff(row) <= 1e-6)


def test_the_vignette_follows_the_frame_rather_than_the_pixels() -> None:
    """The preview and the full size export must come out the same."""
    small = pipeline.vignette_factor((40, 60), -80.0, 35.0)
    large = pipeline.vignette_factor((400, 600), -80.0, 35.0)
    for row, column in ((0, 0), (0.5, 0.5), (0.5, 0.0), (1.0, 1.0)):
        here = small[min(round(row * 40), 39), min(round(column * 60), 59)]
        there = large[min(round(row * 400), 399), min(round(column * 600), 599)]
        assert float(here) == pytest.approx(float(there), abs=2e-3), (row, column)


def test_the_vignette_puts_no_cast_on_a_neutral_grey() -> None:
    """Nothing but the light may change (PLAN.md 3.5: no colour of its own)."""
    for amount in (-100.0, -50.0, 50.0, 100.0):
        for level in (0.02, 0.18, 0.6):
            result = pipeline.set_vignette(_grey(level, size=(32, 32)), amount, 50.0)
            spread = result.max(axis=2) - result.min(axis=2)
            assert float(spread.max()) < 1e-6, (amount, level)


def test_the_vignette_keeps_the_hue() -> None:
    image = np.broadcast_to(_working((0.7, 0.4, 0.2)), (32, 32, 3)).copy()
    before = cm.prophoto_to_oklch(image)[..., 2]
    after = cm.prophoto_to_oklch(pipeline.set_vignette(image, -80.0, 30.0))[..., 2]
    assert float(np.abs(after - before).max()) < 1e-3


def test_a_bright_vignette_is_brought_back_to_white_rather_than_clipped() -> None:
    """Above 1.0 is step 11's to settle, which is why this step comes first."""
    lifted = pipeline.set_vignette(_grey(0.8, size=(32, 32)), 100.0, 20.0)
    assert float(lifted.max()) > 1.0
    compressed = pipeline.compress_to_gamut(lifted)
    assert float(compressed.max()) <= 1.0 + 1e-4
    assert float(compressed[0, 0].mean()) > float(compressed[16, 16].mean())


# -- step 10: vibrance and saturation -------------------------------------


def test_saturation_at_its_lowest_gives_a_black_and_white_photo() -> None:
    """PLAN.md stage 2 acceptance."""
    colours = _working((0.8, 0.2, 0.1), (0.1, 0.7, 0.3), (0.2, 0.3, 0.9))
    result = pipeline.vibrance_and_saturation(colours, 0.0, -100.0)
    assert float(cm.prophoto_to_oklch(result)[..., 1].max()) < 1e-3


def test_saturation_makes_colours_stronger() -> None:
    colours = _working((0.6, 0.35, 0.3))
    before = float(cm.prophoto_to_oklch(colours)[0, 0, 1])
    after = float(cm.prophoto_to_oklch(pipeline.vibrance_and_saturation(colours, 0.0, 50.0))[0, 0, 1])
    assert after == pytest.approx(before * 1.5, rel=0.02)


def test_saturation_does_not_shift_the_hue() -> None:
    """PLAN.md stage 2 acceptance: hue must not move."""
    colours = _working((0.8, 0.3, 0.2), (0.2, 0.6, 0.8), (0.5, 0.5, 0.1))
    for amount in (-60.0, -20.0, 40.0, 90.0):
        result = pipeline.vibrance_and_saturation(colours, 0.0, amount)
        before = cm.prophoto_to_oklch(colours)
        after = cm.prophoto_to_oklch(result)
        assert np.allclose(after[..., 2], before[..., 2], atol=2e-3), amount
        assert np.allclose(after[..., 0], before[..., 0], atol=2e-3), amount


def test_vibrance_lifts_the_pale_colours_more_than_the_strong_ones() -> None:
    pale = _working((0.55, 0.5, 0.45))
    strong = _working((0.9, 0.05, 0.05))

    def gain(image: np.ndarray) -> float:
        before = float(cm.prophoto_to_oklch(image)[0, 0, 1])
        after = float(
            cm.prophoto_to_oklch(pipeline.vibrance_and_saturation(image, 100.0, 0.0))[0, 0, 1]
        )
        return after / max(before, 1e-6)

    assert gain(pale) > gain(strong)


def _weight_at_hue(degrees: float, chroma: float = 0.05) -> float:
    """Return the vibrance weight for a colour at this hue."""
    radians = np.radians(degrees)
    a = np.float32([chroma * np.cos(radians)])
    b = np.float32([chroma * np.sin(radians)])
    return float(pipeline._vibrance_weight(np.float32([chroma]), a, b)[0])


def test_vibrance_holds_back_on_skin() -> None:
    """A face must not turn orange when vibrance is raised."""
    middle = (pipeline.SKIN_HUE_FROM_DEGREES + pipeline.SKIN_HUE_TO_DEGREES) / 2
    skin = _weight_at_hue(middle)
    other = _weight_at_hue(200.0)
    assert skin < other
    assert skin == pytest.approx(other * pipeline.SKIN_VIBRANCE_WEIGHT, rel=1e-5)


@pytest.mark.parametrize("degrees", [0.0, 20.0, 34.0, 76.0, 120.0, 200.0, 300.0, 359.0])
def test_colours_outside_the_skin_band_get_the_full_weight(degrees: float) -> None:
    """The band must be exactly the one PLAN.md names, not a nearby one."""
    assert _weight_at_hue(degrees) == pytest.approx(_weight_at_hue(200.0), rel=1e-5)


@pytest.mark.parametrize("degrees", [36.0, 55.0, 74.0])
def test_colours_inside_the_skin_band_are_held_back(degrees: float) -> None:
    assert _weight_at_hue(degrees) < _weight_at_hue(200.0)


def test_neither_colour_slider_at_zero_changes_anything() -> None:
    image = _working((0.4, 0.3, 0.2))
    assert pipeline.vibrance_and_saturation(image, 0.0, 0.0) is image


def test_a_grey_photo_cannot_be_made_colourful() -> None:
    grey = _grey(0.4)
    for vibrance, saturation in ((100.0, 0.0), (0.0, 100.0), (100.0, 100.0)):
        result = pipeline.vibrance_and_saturation(grey, vibrance, saturation)
        assert float(cm.prophoto_to_oklch(result)[..., 1].max()) < 2e-3


# -- step 3: white balance ------------------------------------------------


def test_leaving_a_raw_photo_at_its_shot_temperature_changes_nothing() -> None:
    """The sliders start where the camera was, so nothing should move."""
    kelvin, tint = cm.camera_neutral_to_temperature(
        1.0 / np.asarray(CAMERA_SHOT_MULTIPLIERS), CAMERA_RGB_XYZ_MATRIX
    )
    gains = pipeline.raw_white_balance_gains(
        kelvin, tint, CAMERA_SHOT_MULTIPLIERS, CAMERA_RGB_XYZ_MATRIX
    )
    assert np.allclose(gains, 1.0, atol=1e-3), gains


def test_a_warmer_setting_adds_red_and_removes_blue() -> None:
    kelvin, tint = cm.camera_neutral_to_temperature(
        1.0 / np.asarray(CAMERA_SHOT_MULTIPLIERS), CAMERA_RGB_XYZ_MATRIX
    )
    warmer = pipeline.raw_white_balance_gains(
        kelvin + 2000.0, tint, CAMERA_SHOT_MULTIPLIERS, CAMERA_RGB_XYZ_MATRIX
    )
    assert warmer[0] > 1.0
    assert warmer[2] < 1.0
    assert warmer[1] == pytest.approx(1.0)


def test_a_cooler_setting_does_the_opposite() -> None:
    kelvin, tint = cm.camera_neutral_to_temperature(
        1.0 / np.asarray(CAMERA_SHOT_MULTIPLIERS), CAMERA_RGB_XYZ_MATRIX
    )
    cooler = pipeline.raw_white_balance_gains(
        max(2000.0, kelvin - 2000.0), tint, CAMERA_SHOT_MULTIPLIERS, CAMERA_RGB_XYZ_MATRIX
    )
    assert cooler[0] < 1.0
    assert cooler[2] > 1.0


def test_a_jpeg_warms_up_without_getting_brighter() -> None:
    """The gains are normalised, so only the colour changes (6 step 3)."""
    for temperature in (-100.0, -40.0, 40.0, 100.0):
        gains = pipeline.jpeg_white_balance_gains(temperature, 0.0)
        weighted = float(np.asarray(gains, dtype=np.float64) @ cm.SRGB_TO_XYZ[1])
        assert weighted == pytest.approx(1.0, rel=1e-5), temperature
    warm = pipeline.jpeg_white_balance_gains(100.0, 0.0)
    assert warm[0] > warm[2]


def test_a_jpeg_at_zero_is_left_alone() -> None:
    assert np.allclose(pipeline.jpeg_white_balance_gains(0.0, 0.0), 1.0, atol=1e-6)


def test_the_jpeg_tint_moves_green_against_magenta() -> None:
    green = pipeline.jpeg_white_balance_gains(0.0, -100.0)
    magenta = pipeline.jpeg_white_balance_gains(0.0, 100.0)
    assert green[1] > magenta[1]


# -- the whole pipeline ---------------------------------------------------


def test_an_untouched_photo_develops_as_it_did_before_the_sliders() -> None:
    """PLAN.md stage 2 acceptance: at zero, nothing changed from stage 1."""
    photo = _raw_photo(_grey(0.05, size=(32, 32)))
    kelvin, tint = pipeline.shot_white_balance(photo)
    untouched = EditState.for_photo(True, kelvin, tint)

    with_state = pipeline.develop(photo, untouched)
    without_state = pipeline.develop(photo)
    assert np.array_equal(with_state, without_state)


def test_an_untouched_srgb_jpeg_is_still_passed_through() -> None:
    pixels = np.random.default_rng(3).integers(0, 256, (8, 12, 3), dtype=np.uint8)
    photo = _jpeg_photo(pixels)
    assert np.array_equal(pipeline.develop(photo, EditState(is_raw=False)), pixels)


def test_touching_a_slider_stops_the_jpeg_passing_through() -> None:
    pixels = np.full((8, 12, 3), (120, 130, 140), dtype=np.uint8)
    photo = _jpeg_photo(pixels)
    state = EditState(is_raw=False).with_value("exposure", 1.0)
    assert not pipeline.is_passthrough(photo, state)
    assert not np.array_equal(pipeline.develop(photo, state), pixels)


def test_exposure_reaches_the_developed_photo() -> None:
    photo = _raw_photo(_grey(0.02, size=(16, 16)))
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    darker = pipeline.develop(photo, state.with_value("exposure", -1.0))
    brighter = pipeline.develop(photo, state.with_value("exposure", 1.0))
    assert int(brighter[0, 0, 0]) > int(darker[0, 0, 0])


def test_the_sliders_do_not_disturb_the_order_of_the_steps() -> None:
    """Developing twice must give the same answer; nothing may be kept."""
    photo = _raw_photo(_grey(0.08, size=(16, 16)))
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo)).with_value(
        "vibrance", 60.0
    )
    assert np.array_equal(pipeline.develop(photo, state), pipeline.develop(photo, state))


def test_the_vignette_reaches_the_developed_photo() -> None:
    """Through the bands as well: the factor is cut up with the photo."""
    photo = _raw_photo(_grey(0.04, size=(64, 96)))
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    plain = pipeline.develop(photo, state)
    darkened = pipeline.develop(photo, state.with_value("vignette", -100.0))
    assert int(darkened[32, 48, 1]) == int(plain[32, 48, 1])
    assert int(darkened[0, 0, 1]) < int(plain[0, 0, 1])
    assert int(darkened[-1, -1, 1]) < int(plain[-1, -1, 1])


def test_the_vignette_reaches_the_photo_past_the_local_steps_too() -> None:
    """Highlights and shadows send the photo down the other of the two paths."""
    photo = _raw_photo(_grey(0.04, size=(64, 96)))
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo)).with_value(
        "shadows", 50.0
    )
    plain = pipeline.develop(photo, state)
    darkened = pipeline.develop(photo, state.with_value("vignette", -100.0))
    assert int(darkened[32, 48, 1]) == int(plain[32, 48, 1])
    assert int(darkened[0, 0, 1]) < int(plain[0, 0, 1])


def test_the_vignette_stops_the_jpeg_passing_through() -> None:
    pixels = np.full((8, 12, 3), (120, 130, 140), dtype=np.uint8)
    photo = _jpeg_photo(pixels)
    state = EditState(is_raw=False)
    assert pipeline.is_passthrough(photo, state.with_value("vignette_midpoint", 80.0))
    assert not pipeline.is_passthrough(photo, state.with_value("vignette", -30.0))
    assert not np.array_equal(
        pipeline.develop(photo, state.with_value("vignette", -30.0)), pixels
    )


def test_every_adjustment_leaves_the_output_inside_its_range() -> None:
    generator = np.random.default_rng(11)
    photo = _raw_photo(generator.random((24, 24, 3), dtype=np.float32))
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    for name, value in (
        ("exposure", 5.0),
        ("exposure", -5.0),
        ("contrast", 100.0),
        ("highlights", -100.0),
        ("shadows", 100.0),
        ("whites", 100.0),
        ("whites", -100.0),
        ("blacks", 100.0),
        ("blacks", -100.0),
        ("vibrance", 100.0),
        ("saturation", 100.0),
        ("saturation", -100.0),
        ("vignette", 100.0),
        ("vignette", -100.0),
    ):
        result = pipeline.develop(photo, state.with_value(name, value))
        assert result.min() >= 0 and result.max() <= 255, (name, value)
