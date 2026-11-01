"""Tests for the colour matrices and conversions (PLAN.md section 6).

Two properties matter most and are required by the plan: neutral colours must
stay neutral through every conversion, and a round trip of white must be
accurate to better than 0.001.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import color_management as cm

TOLERANCE = 1e-3


def _image(*colours: tuple[float, float, float]) -> np.ndarray:
    """Return the colours as a 1 x N image."""
    return np.array([list(colours)], dtype=np.float32)


# -- matrices -------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix",
    [cm.SRGB_TO_XYZ, cm.ADOBE_RGB_TO_XYZ, cm.DISPLAY_P3_TO_XYZ, cm.PROPHOTO_TO_XYZ],
)
def test_primaries_map_white_to_their_white_point(matrix: np.ndarray) -> None:
    """RGB (1, 1, 1) must land on the space's own white point."""
    white = matrix @ np.ones(3)
    expected = cm.WHITE_D50 if matrix is cm.PROPHOTO_TO_XYZ else cm.WHITE_D65
    assert np.allclose(white, expected, atol=TOLERANCE)


@pytest.mark.parametrize(
    ("forward", "inverse"),
    [
        (cm.SRGB_TO_XYZ, cm.XYZ_TO_SRGB),
        (cm.ADOBE_RGB_TO_XYZ, cm.XYZ_TO_ADOBE_RGB),
        (cm.DISPLAY_P3_TO_XYZ, cm.XYZ_TO_DISPLAY_P3),
        (cm.PROPHOTO_TO_XYZ, cm.XYZ_TO_PROPHOTO),
    ],
)
def test_each_matrix_and_its_inverse_cancel(forward: np.ndarray, inverse: np.ndarray) -> None:
    assert np.allclose(forward @ inverse, np.eye(3), atol=1e-9)


def test_luminance_coefficients_sum_to_one() -> None:
    assert cm.PROPHOTO_LUMINANCE.sum() == pytest.approx(1.0, abs=TOLERANCE)


# -- chromatic adaptation -------------------------------------------------


def test_bradford_maps_one_white_point_onto_the_other() -> None:
    adapted = cm.ADAPT_D65_TO_D50 @ cm.WHITE_D65
    assert np.allclose(adapted, cm.WHITE_D50, atol=TOLERANCE)


def test_bradford_is_reversible() -> None:
    assert np.allclose(cm.ADAPT_D50_TO_D65 @ cm.ADAPT_D65_TO_D50, np.eye(3), atol=1e-9)


def test_adapting_a_white_point_to_itself_changes_nothing() -> None:
    assert np.allclose(cm.bradford_adaptation(cm.WHITE_D65, cm.WHITE_D65), np.eye(3), atol=1e-9)


# -- transfer functions ---------------------------------------------------


def test_srgb_transfer_function_round_trips() -> None:
    encoded = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(cm.linear_to_srgb(cm.srgb_to_linear(encoded)), encoded, atol=1e-5)


def test_srgb_transfer_function_anchors() -> None:
    assert cm.srgb_to_linear(np.float32([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert cm.srgb_to_linear(np.float32([1.0]))[0] == pytest.approx(1.0, abs=1e-6)
    # Middle grey: 0.5 encoded is about 0.214 in linear light.
    assert cm.srgb_to_linear(np.float32([0.5]))[0] == pytest.approx(0.2140, abs=1e-3)


def test_adobe_rgb_transfer_function_anchors() -> None:
    assert cm.adobe_rgb_to_linear(np.float32([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert cm.adobe_rgb_to_linear(np.float32([1.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert cm.adobe_rgb_to_linear(np.float32([0.5]))[0] == pytest.approx(0.2176, abs=1e-3)


# -- neutral stays neutral ------------------------------------------------


@pytest.mark.parametrize("level", [0.05, 0.18, 0.5, 1.0])
@pytest.mark.parametrize(
    "matrix",
    [cm.SRGB_TO_PROPHOTO, cm.ADOBE_RGB_TO_PROPHOTO, cm.DISPLAY_P3_TO_PROPHOTO],
)
def test_neutral_input_stays_neutral_in_the_working_space(
    matrix: np.ndarray, level: float
) -> None:
    """A neutral grey must have equal channels after adaptation to D50."""
    result = cm.apply_matrix(_image((level, level, level)), matrix)[0, 0]
    assert result.max() - result.min() < TOLERANCE, result


def test_white_survives_a_round_trip_through_the_working_space() -> None:
    """PLAN.md section 6: a round trip of sRGB white must stay within 0.001."""
    white = _image((1.0, 1.0, 1.0))
    working = cm.apply_matrix(white, cm.SRGB_TO_PROPHOTO)
    back = cm.apply_matrix(working, cm.PROPHOTO_TO_SRGB)
    assert np.allclose(back, white, atol=TOLERANCE)


@pytest.mark.parametrize("level", [0.0, 0.18, 0.5, 1.0])
def test_srgb_survives_a_round_trip_through_the_working_space(level: float) -> None:
    colours = _image((level, level, level), (level, 0.0, 0.0), (0.0, level, 0.0))
    working = cm.apply_matrix(colours, cm.SRGB_TO_PROPHOTO)
    back = cm.apply_matrix(working, cm.PROPHOTO_TO_SRGB)
    assert np.allclose(back, colours, atol=TOLERANCE)


# -- camera matrix --------------------------------------------------------

#: The matrix LibRaw reports for the camera this was written against.
CAMERA_RGB_XYZ_MATRIX = np.array(
    [
        [0.7374, -0.2389, -0.0551],
        [-0.5435, 1.3162, 0.2519],
        [-0.1006, 0.1795, 0.6552],
        [0.0000, 0.0000, 0.0000],
    ]
)


def test_camera_neutral_maps_onto_the_d65_white_point() -> None:
    """White balanced camera neutral must be D65 white (PLAN.md 6 step 5)."""
    matrix = cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX)
    white = matrix @ np.ones(3)
    assert np.allclose(white, cm.WHITE_D65, atol=TOLERANCE)


def test_camera_neutral_stays_neutral_in_the_working_space() -> None:
    matrix = cm.XYZ_TO_PROPHOTO @ cm.ADAPT_D65_TO_D50 @ cm.camera_to_xyz_matrix(
        CAMERA_RGB_XYZ_MATRIX
    )
    result = cm.apply_matrix(_image((0.5, 0.5, 0.5)), matrix)[0, 0]
    assert result.max() - result.min() < TOLERANCE, result


def test_a_singular_camera_matrix_is_rejected() -> None:
    with pytest.raises(ValueError):
        cm.camera_to_xyz_matrix(np.zeros((4, 3)))


# -- Oklab and OkLCh ------------------------------------------------------


def test_oklab_round_trips_through_xyz() -> None:
    colours = _image((0.2, 0.3, 0.4), (0.9, 0.1, 0.05), (0.5, 0.5, 0.5))
    assert np.allclose(cm.oklab_to_xyz(cm.xyz_to_oklab(colours)), colours, atol=1e-5)


def test_oklch_round_trips_through_oklab() -> None:
    colours = _image((0.6, 0.1, -0.05), (0.3, -0.08, 0.12))
    assert np.allclose(cm.oklch_to_oklab(cm.oklab_to_oklch(colours)), colours, atol=1e-6)


def test_white_has_no_chroma_in_oklch() -> None:
    """A neutral colour must sit exactly on the achromatic axis."""
    white = cm.apply_matrix(_image((1.0, 1.0, 1.0)), cm.SRGB_TO_PROPHOTO)
    lch = cm.prophoto_to_oklch(white)[0, 0]
    assert lch[0] == pytest.approx(1.0, abs=2e-3)  # L = 1 for white
    assert lch[1] < 2e-3, lch  # C = 0


@pytest.mark.parametrize("level", [0.1, 0.18, 0.5, 0.9])
def test_greys_have_no_chroma_in_oklch(level: float) -> None:
    grey = cm.apply_matrix(_image((level, level, level)), cm.SRGB_TO_PROPHOTO)
    assert cm.prophoto_to_oklch(grey)[0, 0, 1] < 2e-3


def test_prophoto_oklch_round_trip() -> None:
    colours = cm.apply_matrix(
        _image((0.8, 0.2, 0.1), (0.1, 0.6, 0.9), (0.18, 0.18, 0.18)), cm.SRGB_TO_PROPHOTO
    )
    assert np.allclose(cm.oklch_to_prophoto(cm.prophoto_to_oklch(colours)), colours, atol=1e-5)


def test_changing_only_chroma_keeps_lightness_and_hue() -> None:
    """Saturation and vibrance must not shift hue (PLAN.md 6 step 10)."""
    colours = cm.apply_matrix(_image((0.8, 0.3, 0.2), (0.2, 0.5, 0.7)), cm.SRGB_TO_PROPHOTO)
    lch = cm.prophoto_to_oklch(colours)
    boosted = lch.copy()
    boosted[..., 1] *= 1.3
    result = cm.prophoto_to_oklch(cm.oklch_to_prophoto(boosted))
    assert np.allclose(result[..., 0], lch[..., 0], atol=1e-4)
    assert np.allclose(result[..., 2], lch[..., 2], atol=1e-4)


def test_scaling_chroma_in_one_step_matches_the_long_way_round() -> None:
    """The folded matrices must give what going to Oklab and back gives."""
    colours = cm.apply_matrix(
        _image((0.8, 0.3, 0.2), (0.2, 0.5, 0.7), (0.05, 0.4, 0.1)), cm.SRGB_TO_PROPHOTO
    )
    lab = cm.prophoto_to_oklab(colours)
    lab[..., 1:] *= 1.35
    assert np.allclose(cm.scale_oklab_chroma(colours, 1.35), cm.oklab_to_prophoto(lab), atol=1e-5)


def test_scaling_chroma_keeps_lightness_and_hue() -> None:
    colours = cm.apply_matrix(_image((0.8, 0.3, 0.2), (0.2, 0.5, 0.7)), cm.SRGB_TO_PROPHOTO)
    before = cm.prophoto_to_oklch(colours)
    after = cm.prophoto_to_oklch(cm.scale_oklab_chroma(colours, 1.35))
    assert np.allclose(after[..., 0], before[..., 0], atol=1e-4)
    assert np.allclose(after[..., 2], before[..., 2], atol=1e-4)
    assert np.allclose(after[..., 1], before[..., 1] * 1.35, rtol=1e-3)


@pytest.mark.parametrize("level", [0.02, 0.18, 0.9])
def test_scaling_chroma_leaves_a_grey_grey(level: float) -> None:
    grey = cm.apply_matrix(_image((level,) * 3), cm.SRGB_TO_PROPHOTO)
    result = cm.apply_matrix(cm.scale_oklab_chroma(grey, 1.35), cm.PROPHOTO_TO_SRGB)[0, 0]
    assert result.max() - result.min() < 1e-3


def test_a_chroma_matrix_of_one_changes_nothing() -> None:
    assert np.allclose(cm.oklab_chroma_matrix(1.0), np.eye(3), atol=1e-12)


# -- dtype contract -------------------------------------------------------


def test_conversions_return_float32_and_do_not_modify_their_input() -> None:
    colours = _image((0.4, 0.5, 0.6))
    original = colours.copy()
    for result in (
        cm.apply_matrix(colours, cm.SRGB_TO_PROPHOTO),
        cm.prophoto_to_oklch(colours),
        cm.srgb_to_linear(colours),
        cm.linear_to_srgb(colours),
    ):
        assert result.dtype == np.float32
    assert np.array_equal(colours, original)


# -- the profile an exported file carries (PLAN.md 13.2) -----------------


def test_the_srgb_profile_is_built_rather_than_shipped() -> None:
    profile = cm.srgb_profile_bytes()
    assert len(profile) > 100
    # Every ICC profile says its own length first and carries this signature.
    assert profile[36:40] == b"acsp"
    assert int.from_bytes(profile[:4], "big") == len(profile)


def test_the_profile_is_only_built_once() -> None:
    assert cm.srgb_profile_bytes() is cm.srgb_profile_bytes()


def test_srgb_reaches_oklab_the_short_way_as_well_as_the_long_way() -> None:
    """The filter's shortcut must land where the working space route lands."""
    linear = np.array(
        [[[0.8, 0.3, 0.1], [0.05, 0.5, 0.9], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )

    short = cm.srgb_linear_to_oklab(linear)
    long_way = cm.prophoto_to_oklab(cm.apply_matrix(linear, cm.SRGB_TO_PROPHOTO))

    assert np.allclose(short, long_way, atol=1e-5)


def test_the_cone_matrices_are_each_other_undone() -> None:
    linear = np.array([[[0.8, 0.3, 0.1], [0.2, 0.6, 0.4]]], dtype=np.float32)

    there = cm.apply_matrix(linear, cm.SRGB_LINEAR_TO_CONE)

    assert np.allclose(cm.apply_matrix(there, cm.CONE_TO_SRGB_LINEAR), linear, atol=1e-5)


def test_a_neutral_grey_is_still_neutral_through_the_cone_matrices() -> None:
    grey = np.full((1, 4, 3), 0.42, dtype=np.float32)

    back = cm.apply_matrix(
        cm.apply_matrix(grey, cm.SRGB_LINEAR_TO_CONE), cm.CONE_TO_SRGB_LINEAR
    )

    assert float(back.max(axis=2).max() - back.min(axis=2).min()) < 1e-5


# -- brightness and colour apart, for the detail step (PLAN.md 6 step 2.5) --


@pytest.mark.parametrize("level", [0.0, 0.05, 0.42, 1.0])
def test_a_grey_has_no_colour_in_the_opponent_split(level: float) -> None:
    split = cm.OPPONENT @ np.array([level, level, level])

    assert split[0] == pytest.approx(level)
    assert split[1:] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_the_opponent_split_round_trips() -> None:
    colours = np.random.default_rng(3).random((50, 3))

    back = (cm.OPPONENT_INVERSE @ (cm.OPPONENT @ colours.T)).T

    assert np.abs(back - colours).max() < 1e-12


def test_a_change_of_brightness_moves_the_three_channels_together() -> None:
    """What lets the sharpening touch the brightness and not the colour."""
    assert cm.OPPONENT_INVERSE[:, 0] == pytest.approx([1.0, 1.0, 1.0])
