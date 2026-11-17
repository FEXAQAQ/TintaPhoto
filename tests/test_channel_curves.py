"""Tests for the channel curves (PLAN.md 2.2 A-35 and section 6 step 9.56)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import color_management as cm
from core import pipeline, presets, sidecar, tone_curve
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import CHANNEL_CURVE_NAMES, CURVE_NAMES, Edit, EditState
from core.geometry import Window
from core.tone_curve import STRAIGHT

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
    random = np.random.default_rng(11)
    return random.uniform(0.0, 1.0, (*shape, 3)).astype(np.float32) ** 2.2


def _working(image: np.ndarray) -> np.ndarray:
    """Return linear sRGB *image* in the working space."""
    return np.ascontiguousarray(cm.apply_matrix(image, cm.SRGB_TO_PROPHOTO))


def _develop(photo: DecodedPhoto, state: EditState, **kwargs: object) -> np.ndarray:
    """Return *photo* developed to 16 bit, as floats from 0 to 1."""
    out = pipeline.develop(photo, state, bits=pipeline.SIXTEEN_BIT, **kwargs)  # type: ignore[arg-type]
    return out.astype(np.float64) / 65535.0


def _curves(state: EditState) -> tuple[tone_curve.Curve, ...]:
    """Return the four channel curves of *state*, in the step's order."""
    return (state.rgb_curve, state.red_curve, state.green_curve, state.blue_curve)


# -- the state -------------------------------------------------------------


@pytest.mark.parametrize("name", CURVE_NAMES[1:])
def test_every_channel_curve_counts_as_an_adjustment_and_is_stored(name: str) -> None:
    curved = JPEG.with_curve(S_CURVE, name)

    assert curved.curve(name) == S_CURVE
    assert not curved.is_neutral_colour
    assert not curved.curves_are_straight
    assert EditState.from_dict(curved.to_dict(), JPEG).curve(name) == S_CURVE
    on_diagonal: tone_curve.Curve = ((0.0, 0.0), (0.4, 0.4), (1.0, 1.0))
    assert JPEG.with_curve(on_diagonal, name).curve(name) == STRAIGHT


def test_a_parameter_file_without_channel_curves_reads_them_straight() -> None:
    state = EditState.from_dict({"luma_curve": [list(point) for point in S_CURVE]}, JPEG)

    assert state.luma_curve == S_CURVE
    assert all(state.curve(name) == STRAIGHT for name in CURVE_NAMES[1:])


def test_an_unknown_curve_is_refused() -> None:
    with pytest.raises(KeyError):
        JPEG.with_curve(S_CURVE, "alpha_curve")


def test_a_jpeg_with_a_channel_curve_is_no_longer_passed_straight_through() -> None:
    photo = _photo(_ramp(), encoded=True)

    assert pipeline.is_passthrough(photo, JPEG)
    assert not pipeline.is_passthrough(photo, JPEG.with_curve(LIFTED, "blue_curve"))


def test_the_parameter_file_keeps_every_curve(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"not really a photo")
    store = tmp_path / "store"
    state = JPEG.with_curve(S_CURVE, "rgb_curve").with_curve(LIFTED, "red_curve")

    sidecar.save(photo, Edit(state), Edit(JPEG), APPLICATION, store)
    read = sidecar.read(photo, JPEG, store)

    assert read is not None
    assert read.adjustments.rgb_curve == S_CURVE
    assert read.adjustments.red_curve == LIFTED
    assert read.adjustments.green_curve == STRAIGHT


def test_a_preset_records_only_the_curves_that_were_drawn() -> None:
    preset = presets.create("tinted", JPEG.with_curve(LIFTED, "blue_curve"), JPEG)

    assert preset.curves == {"blue_curve": LIFTED}
    assert set(preset.to_dict()) & set(CURVE_NAMES) == {"blue_curve"}
    applied = preset.apply_to(JPEG.with_curve(LIFTED, "red_curve"))
    assert applied.blue_curve == LIFTED
    assert applied.red_curve == LIFTED
    again = presets.Preset.from_dict(preset.to_dict())
    assert again is not None and again.curves == {"blue_curve": LIFTED}


def test_the_curves_copied_from_another_state_are_all_five() -> None:
    source = JPEG
    for name in CURVE_NAMES:
        source = source.with_curve(S_CURVE, name)

    copied = JPEG.with_curves_of(source)

    assert all(copied.curve(name) == S_CURVE for name in CURVE_NAMES)
    assert JPEG.with_curve(S_CURVE).with_curves_of(JPEG).curves_are_straight


def test_a_curve_is_found_with_itself_and_what_follows_it_straight() -> None:
    state = JPEG
    for name in CURVE_NAMES:
        state = state.with_curve(S_CURVE, name)

    assert state.before_curve("luma_curve").curves_are_straight
    for_rgb = state.before_curve("rgb_curve")
    assert for_rgb.luma_curve == S_CURVE
    assert all(for_rgb.curve(name) == STRAIGHT for name in CURVE_NAMES[1:])
    for name in CHANNEL_CURVE_NAMES:
        for_channel = state.before_curve(name)
        assert for_channel.luma_curve == S_CURVE
        assert for_channel.rgb_curve == S_CURVE
        assert all(for_channel.curve(other) == STRAIGHT for other in CHANNEL_CURVE_NAMES)


# -- the step --------------------------------------------------------------


def test_straight_curves_hand_the_photo_back_untouched() -> None:
    image = _working(_scene())

    assert pipeline.channel_curves(image, *_curves(JPEG)) is image


def _shown(image: np.ndarray) -> np.ndarray:
    """Return working space *image* in linear sRGB, where the curves act."""
    return cm.apply_matrix(image, cm.PROPHOTO_TO_SRGB)


def test_a_channel_with_straight_curves_keeps_its_value() -> None:
    scene = _scene()

    shaped = _shown(pipeline.channel_curves(_working(scene), STRAIGHT, S_CURVE, STRAIGHT, STRAIGHT))

    np.testing.assert_allclose(shaped[..., 1:], scene[..., 1:], atol=1e-5)
    assert np.abs(shaped[..., 0] - scene[..., 0]).max() > 0.01


def test_the_step_does_not_change_what_it_was_given() -> None:
    image = _working(_scene())
    before = image.copy()

    pipeline.channel_curves(image, S_CURVE, LIFTED, STRAIGHT, S_CURVE)

    np.testing.assert_array_equal(image, before)


def test_a_grey_ramp_comes_out_where_the_shared_curve_puts_it_and_stays_neutral() -> None:
    out = _develop(_photo(_ramp()), JPEG.with_curve(S_CURVE, "rgb_curve"))
    wanted = tone_curve.evaluate(S_CURVE, np.linspace(0.0, 1.0, 256))

    np.testing.assert_allclose(out[32, :, 1], wanted, atol=2e-3)
    assert np.abs(out[..., 0] - out[..., 1]).max() < 1e-3
    assert np.abs(out[..., 2] - out[..., 1]).max() < 1e-3


def test_a_channel_curve_follows_the_shared_one() -> None:
    state = JPEG.with_curve(S_CURVE, "rgb_curve").with_curve(LIFTED, "red_curve")

    out = _develop(_photo(_ramp()), state)

    across = np.linspace(0.0, 1.0, 256)
    shared = tone_curve.evaluate(S_CURVE, across)
    np.testing.assert_allclose(out[32, :, 0], tone_curve.evaluate(LIFTED, shared), atol=3e-3)
    np.testing.assert_allclose(out[32, :, 1], shared, atol=2e-3)


def test_one_channel_raised_tints_grey() -> None:
    out = _develop(_photo(_ramp()), JPEG.with_curve(LIFTED, "blue_curve"))

    middle = out[32, 128]
    assert middle[2] > middle[1] + 0.05
    assert middle[0] == pytest.approx(middle[1], abs=1e-3)


def test_raising_the_left_end_of_the_shared_curve_lifts_pure_black_to_grey() -> None:
    black = np.zeros((16, 16, 3), dtype=np.float32)

    out = _develop(_photo(black), JPEG.with_curve(LIFTED, "rgb_curve"))

    np.testing.assert_allclose(out, 0.2, atol=2e-3)


def test_an_s_curve_on_the_channels_deepens_colour_more_than_on_brightness() -> None:
    scene = _scene()

    on_channels = _develop(_photo(scene), JPEG.with_curve(S_CURVE, "rgb_curve"))
    on_brightness = _develop(_photo(scene), JPEG.with_curve(S_CURVE))

    def spread(image: np.ndarray) -> float:
        return float((image.max(axis=2) - image.min(axis=2)).mean())

    assert spread(on_channels) > spread(on_brightness) * 1.05


def test_below_nought_is_carried_through_and_past_white_stays_past_white() -> None:
    scene = np.full((4, 4, 3), 1.5, dtype=np.float32)
    scene[..., 2] = -0.01

    shaped = _shown(pipeline.channel_curves(_working(scene), S_CURVE, STRAIGHT, STRAIGHT, STRAIGHT))

    assert shaped[..., :2].min() > 1.0
    np.testing.assert_allclose(shaped[..., 2], -0.01, atol=1e-5)


def test_the_channel_curves_act_before_the_vignette() -> None:
    darker: tone_curve.Curve = ((0.0, 0.0), (0.5, 0.25), (1.0, 1.0))
    state = JPEG.with_curve(darker, "rgb_curve").with_value("vignette", -100.0)
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)

    out = _develop(_photo(flat), state)

    # The middle is the curve alone, and the corner is that, two stops down.
    middle = float(cm.srgb_to_linear(np.array([out[32, 32, 1]], dtype=np.float32))[0])
    corner = float(cm.srgb_to_linear(np.array([out[0, 0, 1]], dtype=np.float32))[0])
    assert corner == pytest.approx(middle / 4.0, rel=0.05)


def test_a_window_comes_out_as_the_whole_photo_does() -> None:
    scene = _scene()
    state = JPEG.with_curve(S_CURVE, "rgb_curve").with_curve(LIFTED, "green_curve")
    whole = pipeline.develop(_photo(scene), state)
    window = Window(left=32, top=16, right=96, bottom=80, frame=(128, 96))

    part = pipeline.develop(_photo(scene[16:80, 32:96]), state, window=window)

    np.testing.assert_array_equal(part, whole[16:80, 32:96])
