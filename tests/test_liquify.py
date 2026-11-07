"""Tests for liquify: a brush that pushes, swells and turns the picture (PLAN.md 2.11, 6 step 1.7)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core import exporter, geometry, liquify
from core.decoder import DecodedPhoto
from core.edit_state import Edit, EditState
from core.exporter import ExportOptions
from core.faces import Face, SkinTone
from core.geometry import Geometry, Rect
from core.liquify import Stroke
from core.skin_range import SkinRange, SkinStroke

SIZE = (300, 200)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Start every test with no fields kept, so none leans on another's."""
    with liquify._cache_lock:
        liquify._cache.clear()


def _line(tool: str, start: tuple[float, float], end: tuple[float, float], count: int,
          radius: float = 0.1, strength: float = 1.0) -> Stroke:
    """Return a stroke of *tool* from *start* to *end* through *count* points."""
    points = tuple(
        (start[0] + (end[0] - start[0]) * i / (count - 1),
         start[1] + (end[1] - start[1]) * i / (count - 1))
        for i in range(count)
    )
    return Stroke(tool=tool, points=points, radius=radius, strength=strength)


def _held(tool: str, at: tuple[float, float], ticks: int, radius: float = 0.1,
          strength: float = 1.0) -> Stroke:
    """Return *tool* held still at *at* for *ticks* points."""
    return Stroke(tool=tool, points=(at,) * ticks, radius=radius, strength=strength)


def _dot(size: tuple[int, int], at: tuple[float, float], radius: float) -> np.ndarray:
    """Return a black picture with a white disc of *radius* pixels at *at* (fractions)."""
    width, height = size
    ys, xs = np.mgrid[:height, :width]
    inside = (xs + 0.5 - at[0] * width) ** 2 + (ys + 0.5 - at[1] * height) ** 2 < radius ** 2
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[inside] = 1.0
    return image


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Return the middle of the light in *image*, in fractions."""
    plane = image[..., 0]
    ys, xs = np.mgrid[: plane.shape[0], : plane.shape[1]]
    total = plane.sum()
    return (
        float(((xs + 0.5) * plane).sum() / total / plane.shape[1]),
        float(((ys + 0.5) * plane).sum() / total / plane.shape[0]),
    )


def _photo(image: np.ndarray, encoded: np.ndarray | None = None) -> DecodedPhoto:
    height, width = image.shape[:2]
    return DecodedPhoto(
        image=image, is_raw=encoded is None, clipped=np.zeros((height, width), bool),
        encoded=encoded,
    )


def _texture(size: tuple[int, int], seed: int = 1) -> np.ndarray:
    width, height = size
    noise = np.random.default_rng(seed).random((height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.0)
    return np.repeat(noise[..., None], 3, axis=2)


def _map_slopes(shape: liquify.Field) -> np.ndarray:
    """Return the determinant of the map's slope at every node: positive where it does not fold."""
    step = shape.grid.step[0]
    down, across = shape.values.shape[1:]
    xs = np.arange(across) * step + shape.values[0]
    ys = np.arange(down)[:, None] * step + shape.values[1]
    dxdx = np.diff(xs, axis=1)[:-1, :] / step
    dydx = np.diff(ys, axis=1)[:-1, :] / step
    dxdy = np.diff(xs, axis=0)[:, :-1] / step
    dydy = np.diff(ys, axis=0)[:, :-1] / step
    return dxdx * dydy - dxdy * dydx


# -- storing ---------------------------------------------------------------


def test_a_stroke_survives_being_stored() -> None:
    stroke = _line(liquify.TOOL_PUSH, (0.2, 0.3), (0.4, 0.3), 5, radius=0.07, strength=0.4)
    assert Stroke.from_dict(stroke.to_dict()) == stroke
    assert liquify.strokes_from_list(liquify.strokes_to_list((stroke,))) == (stroke,)


@pytest.mark.parametrize(
    "data",
    [
        None,
        "push",
        {"tool": "melt", "points": [[0.5, 0.5]], "radius": 0.1, "strength": 0.5},
        {"tool": "push", "points": [], "radius": 0.1, "strength": 0.5},
        {"tool": "push", "points": [[0.5]], "radius": 0.1, "strength": 0.5},
        {"tool": "push", "points": [[0.5, "a"]], "radius": 0.1, "strength": 0.5},
        {"tool": "push", "points": [[0.5, 0.5]], "radius": True, "strength": 0.5},
        {"tool": "push", "points": [[0.5, 0.5]], "radius": 0.1},
    ],
)
def test_what_is_not_a_stroke_is_left_out(data: object) -> None:
    assert Stroke.from_dict(data) is None
    assert liquify.strokes_from_list([data]) == ()


def test_a_hand_edited_stroke_is_brought_back_into_range() -> None:
    stroke = Stroke.from_dict(
        {"tool": "bloat", "points": [[-1.0, 2.0]], "radius": 5.0, "strength": -3.0}
    )
    assert stroke == Stroke(
        tool="bloat", points=((0.0, 1.0),), radius=liquify.RADIUS_MAX,
        strength=liquify.STRENGTH_MIN,
    )


def test_the_list_of_strokes_and_the_points_of_one_are_capped() -> None:
    many = [Stroke(tool="push", points=((0.5, 0.5),)).to_dict()] * (liquify.STROKES_MAX + 5)
    assert len(liquify.strokes_from_list(many)) == liquify.STROKES_MAX
    long = {"tool": "push", "points": [[0.5, 0.5]] * (liquify.POINTS_MAX + 9), "radius": 0.1,
            "strength": 0.5}
    assert len(Stroke.from_dict(long).points) == liquify.POINTS_MAX  # type: ignore[union-attr]


def test_strokes_are_stored_with_the_geometry() -> None:
    stroke = _held(liquify.TOOL_BLOAT, (0.4, 0.5), 3)
    shape = Geometry(liquify=(stroke,))
    assert Geometry.from_dict(shape.to_dict()) == shape
    assert Geometry.from_dict({"quarter_turns": 1}).liquify == ()
    assert shape.is_default  # the frame itself is untouched


def test_a_push_keeps_points_a_quarter_radius_apart_and_a_held_brush_keeps_them_all() -> None:
    points = tuple((0.5 + i * 0.001, 0.5) for i in range(30))
    pushed = liquify.thinned(liquify.TOOL_PUSH, points, 0.04, (1000, 1000))
    assert pushed[0] == points[0] and pushed[-1] == points[-1]
    assert len(pushed) < len(points) / 5
    held = ((0.5, 0.5),) * 30
    assert liquify.thinned(liquify.TOOL_BLOAT, held, 0.04, (1000, 1000)) == held


def test_a_mirror_turns_a_twirl_the_other_way_and_leaves_the_rest() -> None:
    assert liquify.mirrored_tool(liquify.TOOL_TWIRL_CW) == liquify.TOOL_TWIRL_CCW
    assert liquify.mirrored_tool(liquify.TOOL_TWIRL_CCW) == liquify.TOOL_TWIRL_CW
    assert liquify.mirrored_tool(liquify.TOOL_PUSH) == liquify.TOOL_PUSH


# -- the field -------------------------------------------------------------


def test_no_strokes_leaves_the_photo_as_it_was() -> None:
    photo = _photo(_texture(SIZE))
    assert liquify.warp(photo, ()) is photo
    assert liquify.field((), SIZE) is None


def test_a_click_of_the_push_moves_nothing() -> None:
    photo = _photo(_texture(SIZE))
    still = Stroke(tool=liquify.TOOL_PUSH, points=((0.5, 0.5),))
    assert liquify.field((still,), SIZE) is None
    assert liquify.warp(photo, (still,)) is photo


def test_the_proxy_and_the_full_size_photo_share_one_field() -> None:
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.5, 0.5), 9)
    assert liquify.grid_for((6000, 4000)) == liquify.grid_for((2560, 1707))
    assert liquify.field((stroke,), (6000, 4000)) is liquify.field((stroke,), (2560, 1707))


def test_a_field_built_on_a_kept_one_is_the_same_as_one_built_afresh() -> None:
    first = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.5, 0.5), 9)
    second = _held(liquify.TOOL_TWIRL_CW, (0.45, 0.5), 10)
    liquify.field((first,), SIZE)
    built_on = liquify.field((first, second), SIZE)
    with liquify._cache_lock:
        liquify._cache.clear()
    afresh = liquify.field((first, second), SIZE)
    assert built_on is not None and afresh is not None
    assert np.array_equal(built_on.values, afresh.values)


@pytest.mark.parametrize(
    "stroke",
    [
        _line(liquify.TOOL_PUSH, (0.2, 0.5), (0.8, 0.5), 40, radius=0.08),
        _line(liquify.TOOL_PUSH, (0.5, 0.1), (0.5, 0.9), 40, radius=0.2),
        _held(liquify.TOOL_BLOAT, (0.5, 0.4), liquify.TICKS_PER_SECOND * 3),
        _held(liquify.TOOL_PUCKER, (0.3, 0.6), liquify.TICKS_PER_SECOND * 3),
        _held(liquify.TOOL_TWIRL_CW, (0.6, 0.5), liquify.TICKS_PER_SECOND * 3, radius=0.15),
    ],
)
def test_no_tool_folds_the_picture_over_even_at_full_strength(stroke: Stroke) -> None:
    shape = liquify.field((stroke,), SIZE)
    assert shape is not None
    # Zero is allowed: where a push runs out through the edge, the edge is
    # held and what lay beyond it is squeezed onto it, to within the
    # rounding of single precision.
    assert _map_slopes(shape).min() >= -1e-3


def test_strokes_laid_over_each_other_do_not_fold_the_picture() -> None:
    half = liquify.STRENGTH_DEFAULT
    strokes = (
        _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.6, 0.5), 20, radius=0.1, strength=half),
        _held(liquify.TOOL_BLOAT, (0.5, 0.45), liquify.TICKS_PER_SECOND, strength=half),
        _held(liquify.TOOL_PUCKER, (0.4, 0.55), liquify.TICKS_PER_SECOND, strength=half),
        _held(liquify.TOOL_TWIRL_CW, (0.5, 0.5), liquify.TICKS_PER_SECOND, radius=0.15,
              strength=half),
        _line(liquify.TOOL_PUSH, (0.5, 0.3), (0.5, 0.7), 20, radius=0.15, strength=half),
    )
    shape = liquify.field(strokes, SIZE)
    assert shape is not None
    assert _map_slopes(shape).min() > 0.0


def test_the_edges_are_held() -> None:
    # A push out through the left edge and one in from the bottom.
    strokes = (
        _line(liquify.TOOL_PUSH, (0.1, 0.5), (0.0, 0.5), 5, radius=0.2),
        _line(liquify.TOOL_PUSH, (0.5, 1.0), (0.5, 0.8), 5, radius=0.2),
    )
    shape = liquify.field(strokes, SIZE)
    assert shape is not None
    extent_x, extent_y = shape.grid.extent
    step = shape.grid.step[0]
    down, across = shape.values.shape[1:]
    lands_x = np.arange(across) * step + shape.values[0]
    lands_y = np.arange(down)[:, None] * step + shape.values[1]
    assert lands_x.min() >= 0.0 and lands_x.max() <= extent_x + 1e-6
    assert lands_y.min() >= 0.0 and lands_y.max() <= extent_y + 1e-6
    assert not shape.values[0][:, 0].any() and not shape.values[0][:, -1].any()
    assert not shape.values[1][0].any() and not shape.values[1][-1].any()


# -- the tools -------------------------------------------------------------


def test_a_push_carries_what_is_under_the_brush_along() -> None:
    image = _dot(SIZE, (0.3, 0.5), 8)
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.5), 10, radius=0.1, strength=1.0)
    moved = liquify.warp(_photo(image), (stroke,)).image
    before = _centroid(image)
    after = _centroid(moved)
    assert after[0] - before[0] > 0.1
    assert after[1] == pytest.approx(before[1], abs=0.005)


def test_a_gentler_push_carries_it_less_far() -> None:
    image = _dot(SIZE, (0.3, 0.5), 8)
    path = ((0.3, 0.5), (0.45, 0.5))
    full = Stroke(tool=liquify.TOOL_PUSH, points=path, radius=0.1, strength=1.0)
    half = Stroke(tool=liquify.TOOL_PUSH, points=path, radius=0.1, strength=0.5)
    far = _centroid(liquify.warp(_photo(image), (full,)).image)[0]
    near = _centroid(liquify.warp(_photo(image), (half,)).image)[0]
    assert 0.3 < near < far


def test_a_swell_makes_the_middle_larger_and_a_pinch_smaller() -> None:
    image = _dot(SIZE, (0.5, 0.5), 10)
    area = image[..., 0].sum()
    swell = _held(liquify.TOOL_BLOAT, (0.5, 0.5), liquify.TICKS_PER_SECOND)
    pinch = _held(liquify.TOOL_PUCKER, (0.5, 0.5), liquify.TICKS_PER_SECOND)
    swollen = liquify.warp(_photo(image), (swell,)).image
    pinched = liquify.warp(_photo(image), (pinch,)).image
    assert swollen[..., 0].sum() > area * 1.5
    assert pinched[..., 0].sum() < area / 1.5
    assert _centroid(swollen) == pytest.approx(_centroid(image), abs=0.003)


def test_a_twirl_turns_clockwise_or_anticlockwise_as_asked() -> None:
    # Clockwise on screen, with y down: what is right of the middle came from above it.
    clockwise = _held(liquify.TOOL_TWIRL_CW, (0.5, 0.5), 10)
    anticlockwise = _held(liquify.TOOL_TWIRL_CCW, (0.5, 0.5), 10)
    right = (0.55, 0.5)
    (from_cw,) = liquify.source_points((right,), (clockwise,), SIZE)
    (from_ccw,) = liquify.source_points((right,), (anticlockwise,), SIZE)
    assert from_cw[1] < 0.5 - 0.005
    assert from_ccw[1] > 0.5 + 0.005


def test_restoring_takes_the_picture_back_toward_the_photo() -> None:
    push = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.5, 0.5), 10, radius=0.1)
    restore = _held(liquify.TOOL_RECONSTRUCT, (0.45, 0.5), 60, radius=0.25)
    pushed = liquify.field((push,), SIZE)
    restored = liquify.field((push, restore), SIZE)
    assert pushed is not None and restored is not None
    assert np.abs(restored.values).max() < np.abs(pushed.values).max() * 0.05


def test_smoothing_evens_the_field_out() -> None:
    twirl = _held(liquify.TOOL_TWIRL_CW, (0.5, 0.5), 30, radius=0.05)
    smooth = _held(liquify.TOOL_SMOOTH, (0.5, 0.5), 30, radius=0.15)
    twirled = liquify.field((twirl,), SIZE)
    smoothed = liquify.field((twirl, smooth), SIZE)
    assert twirled is not None and smoothed is not None

    def roughness(shape: liquify.Field) -> float:
        return float(np.abs(np.diff(shape.values, axis=2)).max())

    assert roughness(smoothed) < roughness(twirled) * 0.8


# -- the pixels ------------------------------------------------------------


def test_what_the_brush_did_not_reach_is_left_exactly_as_it_was() -> None:
    image = _texture(SIZE)
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.4, 0.5), 6, radius=0.05)
    moved = liquify.warp(_photo(image), (stroke,)).image
    width, height = SIZE
    ys, xs = np.mgrid[:height, :width]
    # The brush reached at most a radius (a twentieth of the long edge, plus
    # a node) around its path.
    reach = 0.05 * width + 2
    near_x = np.clip((xs + 0.5), 0.3 * width, 0.4 * width)
    far = (xs + 0.5 - near_x) ** 2 + (ys + 0.5 - 0.5 * height) ** 2 > reach ** 2
    assert np.array_equal(moved[far], image[far])
    assert not np.array_equal(moved, image)


def test_a_jpeg_keeps_its_own_pixels_outside_the_brush_and_its_clipping_follows() -> None:
    width, height = SIZE
    image = _dot(SIZE, (0.3, 0.5), 8)
    encoded = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    clipped = image[..., 0] > 0.5
    photo = DecodedPhoto(image=image, is_raw=False, clipped=clipped, encoded=encoded)
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.5), 10, radius=0.1)
    moved = liquify.warp(photo, (stroke,))
    assert moved.encoded is not None
    assert _centroid(moved.encoded.astype(np.float32))[0] > 0.4
    assert _centroid(moved.clipped[..., None].astype(np.float32))[0] > 0.4
    ys, xs = np.mgrid[:height, :width]
    far = xs < 0.15 * width
    assert np.array_equal(moved.encoded[far], encoded[far])


def test_a_stroke_moves_the_picture_the_same_way_at_any_size() -> None:
    big_size = (1200, 800)
    small_size = (300, 200)
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.45), 10, radius=0.1)
    big = liquify.warp(_photo(_dot(big_size, (0.3, 0.5), 32)), (stroke,)).image
    small = liquify.warp(_photo(_dot(small_size, (0.3, 0.5), 8)), (stroke,)).image
    assert _centroid(big) == pytest.approx(_centroid(small), abs=0.003)


def test_the_faces_found_on_the_photo_move_with_it() -> None:
    tone = SkinTone(red=0.0, blue=0.0, spread=0.1, brightness=-2.0)
    face = Face(points=((0.3, 0.5), (0.35, 0.5)), tone=tone)
    image = _texture(SIZE)
    photo = DecodedPhoto(image=image, is_raw=True, clipped=np.zeros(image.shape[:2], bool),
                         faces=(face,))
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.5), 10, radius=0.1)
    moved = liquify.warp(photo, (stroke,))
    assert moved.faces is not None
    (eye, _other) = moved.faces[0].points
    assert eye[0] > 0.4
    assert liquify.source_points((eye,), (stroke,), SIZE)[0] == pytest.approx((0.3, 0.5), abs=1e-4)


# -- points through the field ----------------------------------------------


@pytest.mark.parametrize(
    "stroke",
    [
        _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.55), 10, radius=0.1, strength=0.6),
        _held(liquify.TOOL_BLOAT, (0.5, 0.5), 40),
        _held(liquify.TOOL_PUCKER, (0.5, 0.5), 40),
        _held(liquify.TOOL_TWIRL_CCW, (0.5, 0.5), 30, radius=0.2),
    ],
)
def test_a_point_is_shown_where_the_picture_fetches_it_from(stroke: Stroke) -> None:
    points = [(0.5, 0.5), (0.42, 0.47), (0.55, 0.6), (0.1, 0.1), (0.36, 0.52)]
    shown = liquify.shown_points(points, (stroke,), SIZE)
    back = liquify.source_points(shown, (stroke,), SIZE)
    assert np.allclose(back, points, atol=1e-5)


def test_the_framing_takes_the_strokes_into_account_both_ways() -> None:
    # A wide brush and a short push carry a small dot almost rigidly, so its
    # middle is where the point it was on is shown.
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.36, 0.5), 4, radius=0.25)
    crop = Rect(0.1, 0.1, 0.8, 0.8)
    shape = Geometry(quarter_turns=1, angle=4.0, crop=crop, liquify=(stroke,))
    image = _dot(SIZE, (0.3, 0.5), 4)
    framed = geometry.apply(_photo(image), shape)
    shown = geometry.shown_point((0.3, 0.5), SIZE, shape)
    assert shown == pytest.approx(_centroid(framed.image), abs=0.004)
    assert geometry.source_point(shown, SIZE, shape) == pytest.approx((0.3, 0.5), abs=1e-5)
    plain = Geometry(quarter_turns=1, angle=4.0, crop=crop)
    unpushed = geometry.shown_point((0.3, 0.5), SIZE, plain)
    assert abs(unpushed[1] - shown[1]) > 0.04  # the push, turned a quarter, runs down


def test_the_portrait_brush_stays_on_the_skin_it_was_painted_on() -> None:
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.5), 10, radius=0.1)
    skin = SkinRange(auto=False, strokes=(SkinStroke(points=((0.3, 0.5),), radius=0.02),))
    shape = Geometry(skin=skin, liquify=(stroke,))
    framed = geometry.apply(_photo(_texture(SIZE)), shape)
    assert framed.skin is not None
    (point,) = framed.skin.strokes[0].points
    assert point[0] > 0.4
    assert geometry.framed_skin(shape, SIZE) == framed.skin


# -- as an edit ------------------------------------------------------------


def test_a_stroke_counts_as_an_edit_and_stops_the_original_being_copied(tmp_path) -> None:
    reference = Edit(EditState.for_photo(False))
    shaped = Edit(
        reference.adjustments,
        Geometry(liquify=(_line(liquify.TOOL_PUSH, (0.3, 0.5), (0.4, 0.5), 4),)),
    )
    assert not shaped.is_untouched(reference)
    options = ExportOptions()
    source = tmp_path / "photo.jpg"
    assert exporter.can_copy_original(source, reference.adjustments, Geometry(), options)
    assert not exporter.can_copy_original(source, reference.adjustments, shaped.geometry, options)


# -- the draft (PLAN.md 3.16) ------------------------------------------------


def test_a_draft_moves_a_picture_on_screen_the_way_the_stroke_will() -> None:
    picture = (np.clip(_dot(SIZE, (0.3, 0.5), 8), 0, 1) * 255).astype(np.uint8)
    stroke = _line(liquify.TOOL_PUSH, (0.3, 0.5), (0.45, 0.5), 10, radius=0.1)
    drafted = liquify.draft(picture, stroke)
    assert drafted.dtype == np.uint8 and drafted.shape == picture.shape
    developed = liquify.warp(_photo(_dot(SIZE, (0.3, 0.5), 8)), (stroke,)).image
    assert _centroid(drafted.astype(np.float32)) == pytest.approx(_centroid(developed), abs=0.01)
    assert liquify.draft(picture, Stroke(tool=liquify.TOOL_PUSH, points=((0.5, 0.5),))) is picture


def test_a_draft_grid_is_only_as_fine_as_the_brush_needs() -> None:
    assert max(liquify.grid_for((3000, 2000), 129).nodes) == 129
    assert liquify.grid_for((3000, 2000), 129).nodes == (129, 86)
    with liquify._cache_lock:
        kept = len(liquify._cache)
    liquify.draft(np.zeros((200, 300, 3), np.uint8), _held(liquify.TOOL_BLOAT, (0.5, 0.5), 5))
    with liquify._cache_lock:
        assert len(liquify._cache) == kept  # drafts are kept nowhere


# -- the face sliders (PLAN.md 2.11 L-11) -----------------------------------

#: A face square to the camera, in fractions of a 300 by 200 photo: eyes 30
#: px apart, the nose between them and a little lower, the mouth below.
FACE = liquify.FaceMarks(
    points=((0.45, 0.4), (0.55, 0.4), (0.5, 0.47), (0.47, 0.54), (0.53, 0.54), (0.5, 0.48)),
)


def _shaped(**values: float) -> liquify.FaceShape:
    return liquify.FaceShape(faces=(FACE,), **values)


def _shown_x(point: tuple[float, float], shape: liquify.FaceShape) -> tuple[float, float]:
    (shown,) = liquify.shown_points((point,), (), SIZE, shape)
    return shown


def test_the_face_sliders_survive_being_stored() -> None:
    shape = _shaped(eyes=40.0, chin=-25.0)
    assert liquify.FaceShape.from_dict(shape.to_dict()) == shape
    geometry_now = Geometry(face_shape=shape)
    assert Geometry.from_dict(geometry_now.to_dict()) == geometry_now
    assert Geometry.from_dict({}).face_shape == liquify.DEFAULT_FACE_SHAPE


@pytest.mark.parametrize(
    "data",
    [None, "eyes", {"eyes": "a"}, {"eyes": True}, {"faces": [{"points": [[0.5, 0.5]]}]}],
)
def test_what_is_not_a_face_slider_rests(data: object) -> None:
    shape = liquify.FaceShape.from_dict(data)
    assert shape.is_default and shape.faces == ()


def test_a_hand_edited_face_slider_is_brought_back_into_range() -> None:
    shape = liquify.FaceShape.from_dict({"nose": 500, "mouth": -300})
    assert shape.nose == 100.0 and shape.mouth == -100.0
    with pytest.raises(KeyError):
        shape.with_value("ears", 10.0)


def test_sliders_at_rest_or_without_a_face_move_nothing() -> None:
    photo = _photo(_texture(SIZE))
    assert liquify.field((), SIZE, _shaped()) is None
    assert liquify.field((), SIZE, liquify.FaceShape(eyes=100.0)) is None
    assert liquify.warp(photo, (), liquify.FaceShape(eyes=100.0)) is photo


def test_the_eyes_grow_or_shrink_about_themselves() -> None:
    left_eye = FACE.points[0]
    image = _dot(SIZE, left_eye, 4)
    area = image[..., 0].sum()
    bigger = liquify.warp(_photo(image), (), _shaped(eyes=100.0)).image
    smaller = liquify.warp(_photo(image), (), _shaped(eyes=-100.0)).image
    assert bigger[..., 0].sum() > area * 1.3
    assert smaller[..., 0].sum() < area * 0.85
    assert _centroid(bigger) == pytest.approx(left_eye, abs=0.004)


@pytest.mark.parametrize(
    ("name", "point"),
    [
        ("nose", (0.52, 0.46)),  # beside the nose
        ("mouth", (0.53, 0.54)),  # a corner of the mouth
        ("face_width", (0.58, 0.52)),  # a cheek
    ],
)
def test_wider_moves_out_from_the_middle_and_narrower_in(
    name: str, point: tuple[float, float]
) -> None:
    wider = _shown_x(point, _shaped(**{name: 100.0}))
    narrower = _shown_x(point, _shaped(**{name: -100.0}))
    assert wider[0] > point[0] + 0.002
    assert narrower[0] < point[0] - 0.002
    assert wider[1] == pytest.approx(point[1], abs=0.002)


def test_a_longer_chin_moves_the_chin_down() -> None:
    chin = (0.5, 0.54 + 0.85 * 0.14 * 2 / 3)  # S is the eyes to mouth, 0.14 of the height
    longer = _shown_x(chin, _shaped(chin=100.0))
    shorter = _shown_x(chin, _shaped(chin=-100.0))
    assert longer[1] > chin[1] + 0.005 > shorter[1] + 0.01
    assert longer[0] == pytest.approx(chin[0], abs=0.002)


def test_a_tilted_face_is_shaped_along_its_own_lines() -> None:
    # The same face turned a quarter: its eyes one above the other, the
    # mouth to the left of them. The chin moves along the face, leftward.
    turned = liquify.FaceMarks(
        points=((0.5, 0.35), (0.5, 0.65), (0.45, 0.5), (0.4, 0.45), (0.4, 0.55), (0.45, 0.5)),
    )
    shape = liquify.FaceShape(chin=100.0, faces=(turned,))
    size = (300, 300)
    chin = (0.4 - 0.85 * 0.1, 0.5)
    (moved,) = liquify.shown_points((chin,), (), size, shape)
    assert moved[0] < chin[0] - 0.005
    assert moved[1] == pytest.approx(0.5, abs=0.002)


def test_every_face_slider_at_either_end_leaves_the_picture_unfolded() -> None:
    for value in (-100.0, 100.0):
        shape = _shaped(**{name: value for name in liquify.FACE_NAMES})
        built = liquify.field((), SIZE, shape)
        assert built is not None
        assert _map_slopes(built).min() > 0.0


def test_strokes_are_laid_over_the_face_sliders() -> None:
    shape = _shaped(eyes=60.0)
    stroke = _held(liquify.TOOL_BLOAT, FACE.points[2], 10, radius=0.05)
    both = liquify.field((stroke,), SIZE, shape)
    face_only = liquify.field((), SIZE, shape)
    with liquify._cache_lock:
        liquify._cache.clear()
    afresh = liquify.field((stroke,), SIZE, shape)
    assert both is not None and face_only is not None and afresh is not None
    assert np.array_equal(both.values, afresh.values)
    assert not np.array_equal(both.values, face_only.values)
    # Without the face sliders the stroke alone makes another field.
    alone = liquify.field((stroke,), SIZE)
    assert alone is not None and not np.array_equal(alone.values, both.values)


def test_a_face_slider_counts_as_an_edit_and_frames_the_points(tmp_path) -> None:
    reference = Edit(EditState.for_photo(False))
    shaped = Edit(reference.adjustments, Geometry(face_shape=_shaped(eyes=50.0)))
    assert not shaped.is_untouched(reference)
    options = ExportOptions()
    source = tmp_path / "photo.jpg"
    assert not exporter.can_copy_original(source, reference.adjustments, shaped.geometry, options)
    point = (0.53, 0.54)
    shape = Geometry(face_shape=_shaped(mouth=100.0))
    shown = geometry.shown_point(point, SIZE, shape)
    assert shown[0] > point[0] + 0.002
    assert geometry.source_point(shown, SIZE, shape) == pytest.approx(point, abs=1e-5)
