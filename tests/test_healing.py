"""Tests for spot removal: painted strokes healed from nearby (PLAN.md 2.2, A-32, 6 step 1.6)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from core import exporter, geometry, healing
from core.decoder import DecodedPhoto
from core.edit_state import Edit, EditState
from core.exporter import ExportOptions
from core.geometry import Geometry, Rect
from core.healing import Spot


def _texture(width: int, height: int, seed: int = 1, level: float = 0.18) -> np.ndarray:
    """Return a grey photo with a fine, even texture, like skin or sky noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.0)
    noise /= noise.std()
    plane = level * np.exp2(0.15 * noise)
    return np.repeat(plane[..., None], 3, axis=2).astype(np.float32)


def _with_blemish(image: np.ndarray, centre: tuple[int, int], radius: int) -> np.ndarray:
    """Return *image* with a dark, warm blot at *centre*."""
    blotted = image.copy()
    ys, xs = np.mgrid[: image.shape[0], : image.shape[1]]
    inside = np.hypot(xs - centre[0], ys - centre[1]) <= radius
    blotted[inside] *= np.array([0.45, 0.3, 0.25], dtype=np.float32)
    return blotted


def _disk(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    return np.hypot(xs - centre[0], ys - centre[1]) <= radius


def _ring(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    """Return the surroundings a spot of *radius* is matched to."""
    return _disk(shape, centre, radius * 1.5) & ~_disk(shape, centre, radius)


def _spot(width: int, height: int, target: tuple[float, float], source: tuple[float, float],
          radius_px: float) -> Spot:
    """Return a spot given in pixel centres of a photo of *width* by *height*."""
    return Spot.dot(
        (target[0] + 0.5) / width,
        (target[1] + 0.5) / height,
        (source[0] + 0.5) / width,
        (source[1] + 0.5) / height,
        radius_px / max(width, height),
    )


def _stroke(width: int, height: int, path: list[tuple[float, float]],
            shift: tuple[float, float], radius_px: float) -> Spot:
    """Return a stroke along pixel centres *path*, its source *shift* pixels away."""
    return Spot(
        points=tuple(((x + 0.5) / width, (y + 0.5) / height) for x, y in path),
        radius=radius_px / max(width, height),
        offset=(shift[0] / width, shift[1] / height),
    )


def _source_of(width: int, height: int, target: tuple[float, float],
               offset: tuple[float, float]) -> tuple[float, float]:
    """Return the pixel a source offset puts the source of *target* at."""
    return target[0] + offset[0] * width, target[1] + offset[1] * height


# -- the spot itself ---------------------------------------------------------


def test_a_stroke_survives_being_stored() -> None:
    stroke = Spot(points=((0.25, 0.75), (0.3, 0.7), (0.35, 0.72)), radius=0.02,
                  offset=(0.05, -0.1))
    assert Spot.from_dict(stroke.to_dict()) == stroke
    assert healing.spots_from_list(healing.spots_to_list((stroke, stroke))) == (stroke, stroke)


def test_a_spot_stored_by_the_first_version_reads_as_a_dot() -> None:
    """0.28.0 stored a circle: a centre and a source centre."""
    spot = Spot.from_dict({"x": 0.25, "y": 0.75, "source_x": 0.3, "source_y": 0.7,
                           "radius": 0.02})
    assert spot == Spot.dot(0.25, 0.75, 0.3, 0.7, 0.02)
    assert spot is not None and spot.points == ((0.25, 0.75),)
    assert spot.source == pytest.approx((0.3, 0.7))


@pytest.mark.parametrize(
    "data",
    [None, "spot", [], {"x": 0.5}, {"x": "a", "y": 0.5, "source_x": 0.1, "source_y": 0.1,
                                     "radius": 0.01},
     {"x": float("nan"), "y": 0.5, "source_x": 0.1, "source_y": 0.1, "radius": 0.01},
     {"x": True, "y": 0.5, "source_x": 0.1, "source_y": 0.1, "radius": 0.01},
     {"points": [], "radius": 0.01, "offset": [0, 0]},
     {"points": [[0.5]], "radius": 0.01, "offset": [0, 0]},
     {"points": [[0.5, "a"]], "radius": 0.01, "offset": [0, 0]},
     {"points": [[0.5, 0.5]], "radius": 0.01},
     {"points": [[0.5, 0.5]], "offset": [0, 0]},
     {"points": [[0.5, 0.5]], "radius": 0.01, "offset": [0, float("inf")]}],
)
def test_what_is_not_a_spot_is_left_out(data: object) -> None:
    assert Spot.from_dict(data) is None
    assert healing.spots_from_list([data]) == ()


def test_a_hand_edited_spot_is_brought_back_into_range() -> None:
    spot = Spot.from_dict({"x": 2.0, "y": -1.0, "source_x": 0.5, "source_y": 0.5,
                           "radius": 5.0})
    assert spot is not None
    assert spot.target == (1.0, 0.0)
    assert spot.radius == healing.RADIUS_MAX
    wild = Spot.from_dict({"points": [[-3, 0.5], [0.5, 9]], "radius": 0.01, "offset": [5, -5]})
    assert wild is not None
    assert wild.points == ((0.0, 0.5), (0.5, 1.0))
    assert wild.offset == (1.0, -1.0)
    long = Spot.from_dict({"points": [[0.5, 0.5]] * (healing.POINTS_MAX + 5), "radius": 0.01,
                           "offset": [0, 0]})
    assert long is not None and len(long.points) == healing.POINTS_MAX
    tiny = Spot.from_dict({"x": 0.5, "y": 0.5, "source_x": 0.5, "source_y": 0.5,
                           "radius": 0.0})
    assert tiny is not None and tiny.radius == healing.RADIUS_MIN


def test_moving_a_stroke_takes_its_whole_path_and_its_source_along() -> None:
    stroke = Spot(points=((0.5, 0.5), (0.6, 0.5)), radius=0.01, offset=(0.1, -0.1))
    moved = stroke.moved_to((0.2, 0.3))
    assert moved.points == (pytest.approx((0.2, 0.3)), pytest.approx((0.3, 0.3)))
    assert moved.source == pytest.approx((0.3, 0.2))
    assert stroke.with_source((0.1, 0.9)).source == pytest.approx((0.1, 0.9))
    assert stroke.with_source((0.1, 0.9)).source_points[1] == pytest.approx((0.2, 0.9))
    assert stroke.with_radius(1.0).radius == healing.RADIUS_MAX


def test_a_path_is_thinned_to_a_quarter_radius_and_keeps_its_end() -> None:
    size = (1000, 500)
    radius = 0.02  # 20 px, so points closer than 5 px go
    path = [(0.1 + i * 0.001, 0.5) for i in range(101)]  # one pixel apart
    kept = healing.thinned(path, radius, size)
    assert kept[0] == path[0] and kept[-1] == path[-1]
    gaps = [(b[0] - a[0]) * 1000 for a, b in zip(kept[:-2], kept[1:-1])]
    assert all(gap >= 5 - 1e-9 for gap in gaps)
    assert len(kept) == 21
    assert healing.thinned([(0.5, 0.5)], radius, size) == ((0.5, 0.5),)
    assert healing.thinned([], radius, size) == ()


def test_a_list_of_spots_is_capped() -> None:
    many = [Spot.dot(0.5, 0.5, 0.6, 0.6).to_dict()] * (healing.SPOTS_MAX + 10)
    assert len(healing.spots_from_list(many)) == healing.SPOTS_MAX


# -- healing -------------------------------------------------------------------


def test_no_spots_leaves_the_photo_as_it_was() -> None:
    photo = DecodedPhoto(image=_texture(40, 30), is_raw=True,
                         clipped=np.zeros((30, 40), dtype=bool))
    assert healing.heal(photo, ()) is photo


def test_a_blemish_is_mended_and_nothing_else_moves() -> None:
    width, height = 200, 160
    clean = _texture(width, height)
    blotted = _with_blemish(clean, (80, 80), 8)
    spot = _spot(width, height, (80, 80), (130, 80), 12)
    healed, _ = healing.heal_image(blotted, None, (spot,))

    inside = _disk((height, width), (80, 80), 8)
    # The blot was 55 % to 75 % darker; it is now within a few percent of
    # what surrounds it.
    ratio = healed[inside].mean(axis=0) / clean[_ring((height, width), (80, 80), 12)].mean(axis=0)
    assert np.all(np.abs(ratio - 1.0) < 0.05)
    # Its texture came along: neither smeared flat nor doubled.
    assert 0.6 < healed[inside, 1].std() / clean[~inside, 1].std() < 1.5
    # Outside the circle not one pixel changed.
    outside = ~_disk((height, width), (80, 80), 12)
    assert np.array_equal(healed[outside], blotted[outside])
    # And the input was not touched.
    assert not np.array_equal(healed, blotted)


def test_texture_from_a_brighter_place_is_brought_to_the_light_around_the_spot() -> None:
    width, height = 240, 120
    image = _texture(width, height)
    # Two stops brighter on the right than on the left.
    image *= np.exp2(np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :, None]
    blotted = _with_blemish(image, (60, 60), 8)
    spot = _spot(width, height, (60, 60), (180, 60), 12)
    healed, _ = healing.heal_image(blotted, None, (spot,))
    inside = _disk((height, width), (60, 60), 8)
    ratio = healed[inside].mean() / image[_ring((height, width), (60, 60), 12)].mean()
    assert abs(ratio - 1.0) < 0.06


def test_the_colour_is_matched_as_well() -> None:
    width, height = 200, 120
    image = _texture(width, height)
    image[:, 100:] *= np.array([0.6, 1.0, 1.4], dtype=np.float32)  # a bluer right half
    blotted = _with_blemish(image, (40, 60), 8)
    spot = _spot(width, height, (40, 60), (160, 60), 12)
    healed, _ = healing.heal_image(blotted, None, (spot,))
    inside = _disk((height, width), (40, 60), 8)
    mended = healed[inside].mean(axis=0)
    assert mended[2] / mended[0] == pytest.approx(1.0, abs=0.05)


def test_a_jpeg_keeps_its_own_pixels_outside_the_spots_and_its_clipping_follows() -> None:
    """PLAN.md 6, "JPG 原圖直通": only the spots leave the original pixels."""
    width, height = 120, 80
    image = _texture(width, height)
    clipped = np.zeros((height, width), dtype=bool)
    clipped[35:46, 25:36] = True  # a blown highlight where the spot is
    encoded = np.clip(np.rint(image ** (1 / 2.2) * 255), 0, 255).astype(np.uint8)
    encoded[36:45, 26:35] = (90, 40, 30)  # the blemish, as the file holds it
    photo = DecodedPhoto(image=image, is_raw=False, clipped=clipped, encoded=encoded)
    spot = _spot(width, height, (30, 40), (80, 40), 10)
    healed = healing.heal(photo, (spot,))

    assert healed.encoded is not None
    outside = ~_disk((height, width), (30, 40), 10)
    assert np.array_equal(healed.encoded[outside], encoded[outside])
    inside = _disk((height, width), (30, 40), 5)
    assert healed.encoded[inside].mean() > encoded[inside].mean() + 40
    assert not healed.clipped[38:43, 28:33].any()
    assert photo.clipped[38:43, 28:33].all()


def test_a_spot_mends_the_same_way_at_any_size() -> None:
    width, height = 400, 300
    clean = _texture(width, height, seed=3)
    blotted = _with_blemish(clean, (150, 140), 14)
    spot = _spot(width, height, (150, 140), (250, 150), 20)
    big, _ = healing.heal_image(blotted, None, (spot,))
    small_source = cv2.resize(blotted, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    small, _ = healing.heal_image(small_source, None, (spot,))
    shrunk = cv2.resize(big, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    inside = _disk((height // 4, width // 4), ((150 + 0.5) / 4 - 0.5, (140 + 0.5) / 4 - 0.5), 3)
    assert np.allclose(small[inside].mean(axis=0), shrunk[inside].mean(axis=0), rtol=0.05)


def test_a_spot_at_the_edge_of_the_frame_still_mends() -> None:
    width, height = 120, 80
    clean = _texture(width, height)
    blotted = _with_blemish(clean, (2, 2), 6)
    spot = _spot(width, height, (2, 2), (40, 30), 9)
    healed, _ = healing.heal_image(blotted, None, (spot,))
    inside = _disk((height, width), (2, 2), 5)
    around = clean[_ring((height, width), (2, 2), 9)].mean()
    assert abs(healed[inside].mean() / around - 1.0) < 0.08
    assert np.isfinite(healed).all()


def test_a_spot_too_small_for_the_picture_changes_nothing() -> None:
    image = _texture(40, 30)
    spot = Spot.dot(0.5, 0.5, 0.2, 0.2, healing.RADIUS_MIN)
    healed, _ = healing.heal_image(image, None, (spot,))
    assert np.array_equal(healed, image)


# -- choosing a source ---------------------------------------------------------


def test_the_source_never_overlaps_the_spot_and_stays_in_the_frame() -> None:
    width, height = 300, 200
    image = _texture(width, height)
    for target in ((0.5, 0.5), (0.05, 0.05), (0.95, 0.9)):
        radius = 0.03
        offset = healing.find_offset(image, [target], radius)
        source = (target[0] + offset[0], target[1] + offset[1])
        gap = math.hypot(offset[0] * width, offset[1] * height)
        assert gap >= 2 * radius * width
        assert 0.0 <= source[0] <= 1.0 and 0.0 <= source[1] <= 1.0


def test_the_source_is_taken_along_an_edge_rather_than_across_it() -> None:
    width, height = 300, 300
    image = _texture(width, height)
    image[:, 150:] *= 4.0  # a bright wall right of x = 150
    # A spot on the dark side, close enough that its surroundings take in
    # the edge.
    target = ((140 + 0.5) / width, (150 + 0.5) / height)
    offset = healing.find_offset(image, [target], 8 / width)
    source_x = (target[0] + offset[0]) * width - 0.5
    # Same side of the edge, about as far from it.
    assert source_x + 8 <= 150
    assert abs(source_x - 140) <= 6


def test_the_source_avoids_another_blemish_nearby() -> None:
    width, height = 300, 300
    image = _texture(width, height, seed=5)
    image = _with_blemish(image, (150, 150), 8)
    # Dark blots everywhere around it but straight below.
    for angle in range(0, 360, 30):
        if angle == 90:
            continue
        x = 150 + 26 * math.cos(math.radians(angle))
        y = 150 + 26 * math.sin(math.radians(angle))
        image = _with_blemish(image, (round(x), round(y)), 8)
    offset = healing.find_offset(image, [(150.5 / width, 150.5 / height)], 10 / width)
    source_px = _source_of(width, height, (150, 150), offset)
    blots = [(150 + 26 * math.cos(math.radians(a)), 150 + 26 * math.sin(math.radians(a)))
             for a in range(0, 360, 30) if a != 90]
    assert min(math.hypot(source_px[0] - x, source_px[1] - y) for x, y in blots) > 10


# -- strokes ---------------------------------------------------------------------


def test_a_painted_line_of_blemishes_is_mended_along_its_length() -> None:
    width, height = 320, 200
    clean = _texture(width, height, seed=9)
    blotted = clean.copy()
    for x in range(80, 241, 20):  # a row of freckles
        blotted = _with_blemish(blotted, (x, 100), 4)
    stroke = _stroke(width, height, [(70, 100), (160, 100), (250, 100)], (0, 30), 8)
    healed, _ = healing.heal_image(blotted, None, (stroke,))

    row = np.zeros((height, width), dtype=bool)
    row[97:104, 80:241] = True
    around = clean[90:110, 70:250].mean()
    assert abs(healed[row].mean() / around - 1.0) < 0.05
    # Outside the painted band not one pixel changed.
    band = np.zeros((height, width), dtype=bool)
    band[88:113, 58:263] = True
    assert np.array_equal(healed[~band], blotted[~band])


def test_a_long_stroke_is_solved_iteratively_as_well_as_a_short_one_directly() -> None:
    width, height = 900, 200
    image = _texture(width, height, seed=4)
    image *= np.exp2(np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :, None]
    blotted = image.copy()
    blotted[96:105, 60:840] *= 0.4  # a dark wire across the whole frame
    stroke = _stroke(width, height, [(50, 100), (450, 100), (850, 100)], (0, 40), 12)
    healed, _ = healing.heal_image(blotted, None, (stroke,))
    for x0 in (100, 400, 750):  # light and shade both followed along it
        mended = healed[97:104, x0:x0 + 40].mean()
        around = image[85:115, x0:x0 + 40].mean()
        assert abs(mended / around - 1.0) < 0.06


def test_the_source_of_a_line_is_taken_beside_it_never_along_it() -> None:
    width, height = 400, 300
    image = _texture(width, height, seed=2)
    path = [(100.5 / width, 150.5 / height), (300.5 / width, 150.5 / height)]
    offset = healing.find_offset(image, path, 8 / max(width, height))
    # Clear of the line: the shift is mostly across it.
    assert abs(offset[1] * height) >= 2 * 8
    assert abs(offset[1] * height) > abs(offset[0] * width)


def test_the_source_of_a_stroke_by_an_edge_stays_on_its_side() -> None:
    width, height = 400, 400
    image = _texture(width, height, seed=6)
    image[:, 200:] *= 4.0  # a bright wall right of x = 200
    path = [(190.5 / width, 100.5 / height), (190.5 / width, 300.5 / height)]
    offset = healing.find_offset(image, path, 8 / width)
    shifted_x = 190 + offset[0] * width
    assert shifted_x + 8 <= 200


# -- where spots sit, and what they change elsewhere ----------------------------


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("angle", [0.0, 12.5, -30.0])
@pytest.mark.parametrize("crop", [Rect(), Rect(0.2, 0.25, 0.5, 0.45)])
def test_a_point_is_shown_where_the_framing_puts_its_pixel(
    turns: int, mirrored: bool, angle: float, crop: Rect
) -> None:
    width, height = 90, 60
    shape = Geometry(quarter_turns=turns, mirrored=mirrored, angle=angle, crop=crop)
    point = ((40 + 0.5) / width, (28 + 0.5) / height)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[27:30, 39:42] = 1.0
    photo = DecodedPhoto(image=image, is_raw=True, clipped=np.zeros((height, width), bool))
    framed = geometry.apply(photo, shape)
    plane = framed.image[..., 0]
    ys, xs = np.mgrid[: plane.shape[0], : plane.shape[1]]
    found = ((xs * plane).sum() / plane.sum(), (ys * plane).sum() / plane.sum())
    shown = geometry.shown_point(point, (width, height), shape)
    assert shown[0] * plane.shape[1] - 0.5 == pytest.approx(found[0], abs=0.3)
    assert shown[1] * plane.shape[0] - 0.5 == pytest.approx(found[1], abs=0.3)
    back = geometry.source_point(shown, (width, height), shape)
    assert back == pytest.approx(point, abs=1e-9)


def test_the_shown_scale_follows_the_crop() -> None:
    size = (6000, 4000)
    assert geometry.shown_scale(size, Geometry()) == 1.0
    assert geometry.shown_scale(size, Geometry(crop=Rect(0, 0, 0.5, 1.0))) == pytest.approx(
        2/3, abs=1e-3)


def test_spots_are_stored_with_the_geometry_and_mended_by_it() -> None:
    spot = Spot.dot(0.4, 0.5, 0.7, 0.5, 0.05)
    shape = Geometry(spots=(spot,))
    assert Geometry.from_dict(shape.to_dict()) == shape
    assert Geometry.from_dict({"quarter_turns": 1}).spots == ()
    assert shape.is_default  # the frame itself is untouched

    width, height = 100, 60
    blotted = _with_blemish(_texture(width, height), (40, 30), 3)
    photo = DecodedPhoto(image=blotted, is_raw=True, clipped=np.zeros((height, width), bool))
    framed = geometry.apply(photo, shape)
    assert not np.array_equal(framed.image, blotted)


def test_a_spot_counts_as_an_edit_and_stops_the_original_being_copied(tmp_path) -> None:
    reference = Edit(EditState.for_photo(False))
    spotted = Edit(reference.adjustments, Geometry(spots=(Spot.dot(0.5, 0.5, 0.6, 0.5),)))
    assert reference.is_untouched(reference)
    assert not spotted.is_untouched(reference)
    source = tmp_path / "photo.jpg"
    options = ExportOptions()
    assert exporter.can_copy_original(source, reference.adjustments, Geometry(), options)
    assert not exporter.can_copy_original(source, reference.adjustments, spotted.geometry,
                                          options)
