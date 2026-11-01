"""The white balance eyedropper (PLAN.md 2.2 A-24 and section 6 step 3)."""

from __future__ import annotations

import numpy as np
import pytest

from core import color_management as cm
from core import geometry, pipeline
from core.decoder import ColorSpace, DecodedPhoto
from core.edit_state import EditState
from core.geometry import Geometry, Rect

#: The matrix LibRaw reports for the camera this was written against.
CAMERA_RGB_XYZ_MATRIX = np.array(
    [
        [0.7374, -0.2389, -0.0551],
        [-0.5435, 1.3162, 0.2519],
        [-0.1006, 0.1795, 0.6552],
        [0.0000, 0.0000, 0.0000],
    ]
)
CAMERA_SHOT_MULTIPLIERS = (2.3359, 1.0, 1.6406)
SIZE = (24, 32)


def _raw_photo(image: np.ndarray, clipped: np.ndarray | None = None) -> DecodedPhoto:
    return DecodedPhoto(
        image=image.astype(np.float32),
        is_raw=True,
        clipped=np.zeros(image.shape[:2], dtype=bool) if clipped is None else clipped,
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


def _grey_card_under(kelvin: float, tint: float, level: float = 0.05) -> np.ndarray:
    """A grey card lit at *kelvin*, as the camera hands it over.

    The sensor reads the light's own neutral; LibRaw has then multiplied it
    by the balance the camera was set to.
    """
    neutral = cm.temperature_to_camera_neutral(kelvin, tint, CAMERA_RGB_XYZ_MATRIX)
    colour = neutral * np.asarray(CAMERA_SHOT_MULTIPLIERS) * level
    return np.broadcast_to(colour, (*SIZE, 3)).astype(np.float32)


def _spread(pixels: np.ndarray) -> int:
    """How far apart the three channels of a developed pixel are."""
    return int(pixels.max()) - int(pixels.min())


# -- RAW ----------------------------------------------------------------


@pytest.mark.parametrize(("kelvin", "tint"), [(3200.0, 0.0), (5600.0, 12.0), (8000.0, -20.0)])
def test_a_grey_card_gives_back_the_light_it_was_lit_by(kelvin: float, tint: float) -> None:
    picked = pipeline.pick_white_balance(_raw_photo(_grey_card_under(kelvin, tint)), (0.5, 0.5))
    assert picked.problem is None
    assert picked.temperature == pytest.approx(kelvin, rel=0.005)
    assert picked.tint == pytest.approx(tint, abs=1.0)


def test_the_picked_balance_develops_the_card_to_grey() -> None:
    photo = _raw_photo(_grey_card_under(3000.0, 25.0))
    plain = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    picked = pipeline.pick_white_balance(photo, (0.5, 0.5))
    balanced = plain.with_value("temperature", picked.temperature).with_value(
        "tint", picked.tint
    )
    assert _spread(pipeline.develop(photo, plain)[12, 16]) > 20
    assert _spread(pipeline.develop(photo, balanced)[12, 16]) <= 1


def test_a_spot_the_camera_already_saw_as_grey_gives_the_shot_balance() -> None:
    photo = _raw_photo(np.full((*SIZE, 3), 0.05, dtype=np.float32))
    picked = pipeline.pick_white_balance(photo, (0.5, 0.5))
    kelvin, tint = pipeline.shot_white_balance(photo)
    assert picked.temperature == pytest.approx(round(kelvin))
    assert picked.tint == pytest.approx(round(tint))


# -- JPEG ---------------------------------------------------------------


def test_a_neutral_jpeg_spot_asks_for_nothing() -> None:
    picked = pipeline.pick_white_balance(
        _jpeg_photo(np.full((*SIZE, 3), 128, dtype=np.uint8)), (0.5, 0.5)
    )
    assert (picked.temperature, picked.tint, picked.problem) == (0.0, 0.0, None)


@pytest.mark.parametrize("cast", [(150, 130, 110), (110, 125, 150), (128, 140, 128), (135, 120, 135)])
def test_the_picked_balance_develops_a_jpeg_spot_to_grey(cast: tuple[int, int, int]) -> None:
    photo = _jpeg_photo(np.full((*SIZE, 3), cast, dtype=np.uint8))
    picked = pipeline.pick_white_balance(photo, (0.5, 0.5))
    state = EditState(is_raw=False).with_value("temperature", picked.temperature).with_value(
        "tint", picked.tint
    )
    assert _spread(pipeline.develop(photo, state)[12, 16]) <= 1


def test_a_warm_jpeg_spot_is_cooled_and_a_green_one_pushed_to_magenta() -> None:
    warm = pipeline.pick_white_balance(
        _jpeg_photo(np.full((*SIZE, 3), (150, 130, 110), dtype=np.uint8)), (0.5, 0.5)
    )
    green = pipeline.pick_white_balance(
        _jpeg_photo(np.full((*SIZE, 3), (128, 150, 128), dtype=np.uint8)), (0.5, 0.5)
    )
    assert warm.temperature < 0.0
    assert green.tint > 0.0


def test_a_cast_beyond_the_jpeg_sliders_is_clamped_by_the_state() -> None:
    photo = _jpeg_photo(np.full((*SIZE, 3), (230, 120, 30), dtype=np.uint8))
    picked = pipeline.pick_white_balance(photo, (0.5, 0.5))
    assert picked.temperature < -100.0
    assert EditState(is_raw=False).with_value("temperature", picked.temperature).temperature == -100.0


# -- spots that cannot be used ------------------------------------------


def test_a_clipped_spot_is_refused() -> None:
    image = _grey_card_under(4000.0, 0.0)
    clipped = np.zeros(SIZE, dtype=bool)
    clipped[12, 16] = True
    picked = pipeline.pick_white_balance(_raw_photo(image, clipped), (0.5, 0.5))
    assert picked.problem == pipeline.PICK_CLIPPED


def test_a_clipped_pixel_elsewhere_does_not_matter() -> None:
    image = _grey_card_under(4000.0, 0.0)
    clipped = np.zeros(SIZE, dtype=bool)
    clipped[0, 0] = True
    picked = pipeline.pick_white_balance(_raw_photo(image, clipped), (0.5, 0.5))
    assert picked.problem is None


def test_a_dark_spot_is_refused() -> None:
    image = _grey_card_under(4000.0, 0.0, level=0.0002)
    assert pipeline.pick_white_balance(_raw_photo(image), (0.5, 0.5)).problem == pipeline.PICK_DARK


def test_a_spot_with_no_light_in_one_channel_is_refused() -> None:
    image = np.full((*SIZE, 3), (0.2, 0.2, 0.0), dtype=np.float32)
    assert pipeline.pick_white_balance(_raw_photo(image), (0.5, 0.5)).problem == pipeline.PICK_DARK


# -- the patch ----------------------------------------------------------


def test_one_odd_pixel_does_not_decide_the_result() -> None:
    image = _grey_card_under(5000.0, 0.0).copy()
    image[12, 16] = (0.3, 0.01, 0.01)
    picked = pipeline.pick_white_balance(_raw_photo(image), (0.5, 0.5))
    assert picked.temperature == pytest.approx(5000.0, rel=0.1)


@pytest.mark.parametrize("point", [(0.0, 0.0), (1.0, 1.0), (0.999, 0.0), (-0.2, 1.4)])
def test_a_point_at_or_past_the_edge_still_reads_the_photo(point: tuple[float, float]) -> None:
    picked = pipeline.pick_white_balance(_raw_photo(_grey_card_under(4500.0, 0.0)), point)
    assert picked.temperature == pytest.approx(4500.0, rel=0.005)


def test_picking_leaves_the_photo_alone() -> None:
    photo = _raw_photo(_grey_card_under(4500.0, 0.0))
    before = photo.image.copy()
    pipeline.pick_white_balance(photo, (0.5, 0.5))
    assert np.array_equal(photo.image, before)


def test_the_point_is_read_on_the_cropped_photo() -> None:
    """The point is where the click landed on the photo as it is shown."""
    image = np.empty((*SIZE, 3), dtype=np.float32)
    image[:, :16] = _grey_card_under(3000.0, 0.0)[:, :16]
    image[:, 16:] = _grey_card_under(7000.0, 0.0)[:, 16:]
    right_half = Geometry(crop=Rect(0.5, 0.0, 0.5, 1.0))
    shown = geometry.apply(_raw_photo(image), right_half)
    picked = pipeline.pick_white_balance(shown, (0.5, 0.5))
    assert picked.temperature == pytest.approx(7000.0, rel=0.005)
