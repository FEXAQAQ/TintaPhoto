"""Tests for the luminance curve (PLAN.md 2.2 A-34 and section 6 step 9.55)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import color_management as cm
from core import pipeline, presets, sidecar, tone_curve
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import Edit, EditState
from core.geometry import Window
from core.tone_curve import MAX_POINTS, MIN_GAP, STRAIGHT

JPEG = EditState(is_raw=False)
S_CURVE: tone_curve.Curve = ((0.0, 0.0), (0.25, 0.18), (0.75, 0.84), (1.0, 1.0))
LIFTED: tone_curve.Curve = ((0.0, 0.2), (1.0, 1.0))
APPLICATION = sidecar.Application(name="TintaPhoto", version="0.0.0")


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


def _ramp(shape: tuple[int, int] = (64, 256)) -> np.ndarray:
    """Return a grey ramp from black to white, in linear light."""
    across = cm.srgb_to_linear(np.linspace(0.0, 1.0, shape[1], dtype=np.float32))
    return np.repeat(np.tile(across, (shape[0], 1))[..., np.newaxis], 3, axis=2)


def _scene(shape: tuple[int, int] = (96, 128)) -> np.ndarray:
    """Return a frame of mixed colours and brightnesses, in linear light."""
    random = np.random.default_rng(7)
    return random.uniform(0.0, 1.0, (*shape, 3)).astype(np.float32) ** 2.2


def _develop(photo: DecodedPhoto, state: EditState, **kwargs: object) -> np.ndarray:
    """Return *photo* developed to 16 bit, as floats from 0 to 1."""
    out = pipeline.develop(photo, state, bits=pipeline.SIXTEEN_BIT, **kwargs)  # type: ignore[arg-type]
    return out.astype(np.float64) / 65535.0


# -- the shape -------------------------------------------------------------


def test_two_points_draw_a_straight_line() -> None:
    at = np.linspace(0.0, 1.0, 11)

    np.testing.assert_allclose(tone_curve.evaluate(STRAIGHT, at), at, atol=1e-12)
    np.testing.assert_allclose(tone_curve.evaluate(LIFTED, at), 0.2 + 0.8 * at, atol=1e-12)


def test_the_curve_passes_through_its_points() -> None:
    xs = np.array([x for x, _ in S_CURVE])
    ys = np.array([y for _, y in S_CURVE])

    np.testing.assert_allclose(tone_curve.evaluate(S_CURVE, xs), ys, atol=1e-12)


def test_rising_points_give_a_curve_that_rises_everywhere() -> None:
    steep: tone_curve.Curve = ((0.0, 0.0), (0.1, 0.02), (0.15, 0.6), (0.6, 0.65), (1.0, 1.0))

    for curve in (S_CURVE, steep):
        values = tone_curve.evaluate(curve, np.linspace(0.0, 1.0, 2001))
        assert np.all(np.diff(values) >= -1e-12)


def test_the_curve_never_leaves_the_heights_of_the_points_either_side() -> None:
    bumpy: tone_curve.Curve = ((0.0, 0.3), (0.3, 0.9), (0.5, 0.1), (0.8, 1.0), (1.0, 0.0))
    xs = [x for x, _ in bumpy]

    for index in range(len(bumpy) - 1):
        low, high = sorted((bumpy[index][1], bumpy[index + 1][1]))
        between = tone_curve.evaluate(bumpy, np.linspace(xs[index], xs[index + 1], 200))
        assert between.min() >= low - 1e-12
        assert between.max() <= high + 1e-12


def test_past_white_the_curve_carries_on_and_never_turns_down() -> None:
    falling: tone_curve.Curve = ((0.0, 0.0), (0.5, 0.9), (1.0, 0.6))
    beyond = np.array([1.0, 1.2, 1.5])

    assert np.all(np.diff(tone_curve.evaluate(S_CURVE, beyond)) > 0.0)
    np.testing.assert_allclose(tone_curve.evaluate(falling, beyond), 0.6)


# -- editing ---------------------------------------------------------------


def test_a_point_is_added_in_order_and_says_where_it_went() -> None:
    curve, index = tone_curve.with_point(STRAIGHT, 0.5, 0.7)

    assert curve == ((0.0, 0.0), (0.5, 0.7), (1.0, 1.0))
    assert index == 1


def test_a_point_cannot_be_added_on_top_of_another_or_past_the_limit() -> None:
    curve, index = tone_curve.with_point(S_CURVE, 0.25 + MIN_GAP / 2, 0.5)
    assert (curve, index) == (S_CURVE, -1)

    full = STRAIGHT
    for step in range(1, MAX_POINTS - 1):
        full, _ = tone_curve.with_point(full, step / MAX_POINTS, 0.5)
    assert len(full) == MAX_POINTS
    assert tone_curve.with_point(full, 0.97, 0.5) == (full, -1)


def test_a_moved_point_stays_between_its_neighbours() -> None:
    moved = tone_curve.moved(S_CURVE, 1, 0.9, 2.0)

    assert moved[1] == (pytest.approx(0.75 - MIN_GAP), 1.0)
    assert tone_curve.moved(S_CURVE, 2, 0.0, 0.5)[2][0] == pytest.approx(0.25 + MIN_GAP)


def test_the_ends_only_move_up_and_down() -> None:
    assert tone_curve.moved(STRAIGHT, 0, 0.4, 0.3) == ((0.0, 0.3), (1.0, 1.0))
    assert tone_curve.moved(STRAIGHT, 1, 0.4, 0.8) == ((0.0, 0.0), (1.0, 0.8))


def test_a_point_can_be_removed_but_not_an_end() -> None:
    assert tone_curve.without_point(S_CURVE, 1) == (S_CURVE[0], S_CURVE[2], S_CURVE[3])
    assert tone_curve.without_point(S_CURVE, 0) == S_CURVE
    assert tone_curve.without_point(S_CURVE, 3) == S_CURVE


# -- reading stored data ---------------------------------------------------


def test_a_stored_curve_is_read_back() -> None:
    assert tone_curve.cleaned([list(point) for point in S_CURVE]) == S_CURVE


@pytest.mark.parametrize("stored", [None, "curve", 3, [], [[0.5, 0.5]], [[0, "a"], [1]]])
def test_nonsense_reads_as_the_straight_curve(stored: object) -> None:
    assert tone_curve.cleaned(stored) == STRAIGHT


def test_stored_points_are_put_in_range_and_in_order() -> None:
    stored = [[0.1, -1], [0.5, 0.4], [0.3, 0.9], [float("nan"), 0.2], [0.8, 7], [0.9, True]]

    assert tone_curve.cleaned(stored) == ((0.0, 0.0), (0.5, 0.4), (1.0, 1.0))


def test_no_more_points_are_read_than_a_curve_may_hold() -> None:
    stored = [[step / 40, 0.5] for step in range(41)]

    assert len(tone_curve.cleaned(stored)) == MAX_POINTS


def test_points_on_the_diagonal_are_the_straight_curve() -> None:
    assert tone_curve.cleaned([[0, 0], [0.5, 0.5], [1, 1]]) == STRAIGHT
    assert JPEG.with_curve(((0.0, 0.0), (0.4, 0.4), (1.0, 1.0))).luma_curve == STRAIGHT


# -- the adjustment --------------------------------------------------------


def test_a_curve_counts_as_an_adjustment_and_is_stored() -> None:
    curved = JPEG.with_curve(S_CURVE)

    assert not curved.is_neutral_colour
    assert not curved.is_untouched(JPEG)
    assert EditState.from_dict(curved.to_dict(), JPEG).luma_curve == S_CURVE
    assert EditState.from_dict({"exposure": 0.5}, JPEG).luma_curve == STRAIGHT
    assert JPEG.to_dict()["luma_curve"] == [[0.0, 0.0], [1.0, 1.0]]


def test_a_jpeg_with_a_curve_is_no_longer_passed_straight_through() -> None:
    photo = _photo(_ramp(), encoded=True)

    assert pipeline.is_passthrough(photo, JPEG)
    assert not pipeline.is_passthrough(photo, JPEG.with_curve(S_CURVE))


def test_the_parameter_file_keeps_the_curve(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    store = tmp_path / "edits"

    sidecar.save(photo, Edit(JPEG.with_curve(S_CURVE)), Edit(JPEG), APPLICATION, store)
    read = sidecar.read(photo, JPEG, store)

    assert read is not None
    assert read.adjustments.luma_curve == S_CURVE


def test_a_curve_pasted_into_a_file_is_read_back(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    store = tmp_path / "edits"

    sidecar.update_adjustments(
        photo, {"luma_curve": [list(point) for point in S_CURVE]}, APPLICATION, store
    )
    read = sidecar.read(photo, JPEG, store)

    assert read is not None
    assert read.adjustments.luma_curve == S_CURVE


def test_a_preset_records_the_curve_only_when_it_was_drawn() -> None:
    plain = presets.create("plain", JPEG.with_value("contrast", 10.0), JPEG)
    curved = presets.create("curved", JPEG.with_curve(S_CURVE), JPEG)

    assert plain.luma_curve is None
    assert "luma_curve" not in plain.to_dict()
    assert plain.apply_to(JPEG.with_curve(LIFTED)).luma_curve == LIFTED
    assert not curved.is_empty
    assert curved.apply_to(JPEG).luma_curve == S_CURVE
    again = presets.Preset.from_dict(curved.to_dict())
    assert again is not None and again.luma_curve == S_CURVE


# -- the step --------------------------------------------------------------


def test_the_straight_curve_hands_the_photo_back_untouched() -> None:
    image = cm.apply_matrix(_scene(), cm.SRGB_TO_PROPHOTO)

    assert pipeline.luminance_curve(image, STRAIGHT) is image


def test_a_curve_that_all_but_draws_the_diagonal_all_but_leaves_the_photo() -> None:
    image = np.ascontiguousarray(cm.apply_matrix(_scene(), cm.SRGB_TO_PROPHOTO))
    nearly: tone_curve.Curve = ((0.0, 0.0), (0.5, 0.5 + 1e-9), (1.0, 1.0))

    shaped = pipeline.luminance_curve(image, nearly)

    np.testing.assert_allclose(shaped, image, atol=2e-6)


def test_the_step_does_not_change_what_it_was_given() -> None:
    image = np.ascontiguousarray(cm.apply_matrix(_scene(), cm.SRGB_TO_PROPHOTO))
    before = image.copy()

    pipeline.luminance_curve(image, S_CURVE)
    pipeline.luminance_curve(image, LIFTED)

    np.testing.assert_array_equal(image, before)


def test_a_grey_ramp_comes_out_where_the_curve_puts_it() -> None:
    out = _develop(_photo(_ramp()), JPEG.with_curve(S_CURVE))
    wanted = tone_curve.evaluate(S_CURVE, np.linspace(0.0, 1.0, 256))

    np.testing.assert_allclose(out[32, :, 1], wanted, atol=2e-3)


def test_neutral_grey_stays_neutral() -> None:
    for curve in (S_CURVE, LIFTED):
        out = _develop(_photo(_ramp()), JPEG.with_curve(curve))

        assert np.abs(out[..., 0] - out[..., 1]).max() < 1e-3
        assert np.abs(out[..., 2] - out[..., 1]).max() < 1e-3


def test_only_the_brightness_moves() -> None:
    image = np.ascontiguousarray(cm.apply_matrix(_scene(), cm.SRGB_TO_PROPHOTO))
    bright = pipeline._luminance(image) > 0.02

    shaped = pipeline.luminance_curve(image, S_CURVE)

    ratio = shaped[bright] / pipeline._luminance(shaped)[bright][:, np.newaxis]
    before = image[bright] / pipeline._luminance(image)[bright][:, np.newaxis]
    np.testing.assert_allclose(ratio, before, atol=1e-3)


def test_raising_the_left_end_lifts_pure_black_to_grey() -> None:
    black = np.zeros((16, 16, 3), dtype=np.float32)

    out = _develop(_photo(black), JPEG.with_curve(LIFTED))

    np.testing.assert_allclose(out, 0.2, atol=2e-3)


def test_deep_shadows_land_on_the_curve_and_keep_their_colour_when_darkened() -> None:
    darker: tone_curve.Curve = ((0.0, 0.0), (0.3, 0.2), (1.0, 1.0))
    image = np.full((8, 8, 3), 0.0004, dtype=np.float32)
    image[..., 0] = 0.0006
    before = float(pipeline._luminance(image)[0, 0])
    encoded = cm.linear_to_srgb(np.array([before], dtype=np.float32)).astype(np.float64)
    wanted = cm.srgb_to_linear(tone_curve.evaluate(darker, encoded).astype(np.float32))[0]

    shaped = pipeline.luminance_curve(image, darker)

    assert float(pipeline._luminance(shaped)[0, 0]) == pytest.approx(float(wanted), rel=5e-3)
    np.testing.assert_allclose(shaped[..., 0] / shaped[..., 1], 1.5, rtol=1e-4)


def test_brighter_than_white_stays_brighter_than_white() -> None:
    image = np.full((4, 4, 3), 1.5, dtype=np.float32)

    shaped = pipeline.luminance_curve(image, S_CURVE)

    assert shaped.min() > 1.0


def test_the_curve_acts_before_the_vignette() -> None:
    state = JPEG.with_curve(((0.0, 0.0), (0.5, 0.25), (1.0, 1.0))).with_value("vignette", -100.0)
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)

    out = _develop(_photo(flat), state)

    # The middle is the curve alone, and the corner is that, two stops down.
    middle = float(cm.srgb_to_linear(np.array([out[32, 32, 1]], dtype=np.float32))[0])
    corner = float(cm.srgb_to_linear(np.array([out[0, 0, 1]], dtype=np.float32))[0])
    assert corner == pytest.approx(middle / 4.0, rel=0.05)


def test_a_window_comes_out_as_the_whole_photo_does() -> None:
    scene = _scene()
    state = JPEG.with_curve(S_CURVE)
    whole = pipeline.develop(_photo(scene), state)
    window = Window(left=32, top=16, right=96, bottom=80, frame=(128, 96))

    part = pipeline.develop(_photo(scene[16:80, 32:96]), state, window=window)

    np.testing.assert_array_equal(part, whole[16:80, 32:96])
