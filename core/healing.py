"""Spot removal: painting over blemishes and healing them from nearby.

PLAN.md section 6 step 1.6. Each :class:`Spot` is one stroke of the brush:
the path the pointer took, the brush's radius, and how far away the same
shape is taken from to cover it. A click is a stroke of one point, a round
dot. The texture is copied from the source, and then brought to the
brightness and colour around the stroke by a correction that is smooth
inside it and matches the surroundings on its edge - the healing brush of
every photo editor, worked out here from the published idea rather than from
anyone's code:

* The work is done on the logarithm of linear light, so the correction is a
  ratio. Texture taken from a brighter patch is darkened with its contrast,
  the way a shadow darkens the texture it falls on.
* The correction is the difference between target and source around the
  stroke, slightly blurred so that it carries no texture of its own, carried
  inward by solving Laplace's equation: the smoothest field that meets the
  edge. It is solved on a copy shrunk until the brush is a few pixels wide,
  directly when that leaves a few hundred unknowns and by conjugate
  gradients for a long stroke, and the smooth answer is stretched back.
* A soft edge blends the mended stroke into the photo, so nothing shows
  where it ends even when the answer is not perfect.

Strokes are held in fractions of the photo after its lens correction and
before it is turned or cropped, so that they stay on the dust or the
blemish they cover whatever is done to the frame afterwards; the radius is a
fraction of the long edge and the source a shift in fractions of the width
and height. The same numbers mend the preview proxy, a window of the full
size photo and the exported file alike.

:func:`find_offset` picks where a new stroke takes its texture from, the
way other editors pick it: the same shape nearby, where the surroundings look
most like the stroke's own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.decoder import DecodedPhoto

Point = tuple[float, float]

# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------

#: The radius of the brush, as a fraction of the long edge. The smallest is
#: about six pixels on a 6000 pixel photo, a speck of sensor dust; the
#: largest a tenth of the frame, which is already more than healing can hide.
RADIUS_MIN: Final[float] = 0.001
RADIUS_MAX: Final[float] = 0.1
#: Where the brush starts: sixty pixels across on a 6000 pixel photo.
RADIUS_DEFAULT: Final[float] = 0.005
#: More strokes than this in a parameter file is not something the interface
#: could have made, and the rest are left out.
SPOTS_MAX: Final[int] = 1000
#: The most points one stroke keeps; a longer path is thinned to this many.
POINTS_MAX: Final[int] = 2000
#: How close together a stroke's points are kept, in radii. Nearer points
#: change its shape by a hair's breadth and only cost time.
POINT_SPACING: Final[float] = 0.25

# --------------------------------------------------------------------------
# Healing
# --------------------------------------------------------------------------

#: Added before the logarithm is taken, so that black has one. About ten
#: stops under white: below it a difference is treated as an amount rather
#: than as a ratio, so the noise of the deepest shadows is not magnified.
LOG_FLOOR: Final[float] = 1e-3
#: How far out, in radii, the surroundings the correction meets are read.
SURROUNDINGS_REACH: Final[float] = 1.5
#: The blur that keeps the texture out of the correction, in radii.
SURROUNDINGS_BLUR: Final[float] = 0.15
#: The radius, in pixels, of the shrunk copy Laplace's equation is solved
#: on. The correction is smooth, so nine pixels across a radius hold it.
SOLVE_RADIUS: Final[float] = 9.0
#: Up to this many unknowns the equation is solved directly; beyond, by
#: conjugate gradients, which a long stroke needs.
DIRECT_UNKNOWNS: Final[int] = 600
#: How closely the conjugate gradients have to meet the equation, relative
#: to where they started, and how many steps they may take.
ITERATIVE_TOLERANCE: Final[float] = 1e-5
ITERATIVE_STEPS: Final[int] = 2000
#: The outer part of the stroke, in radii, over which the mended pixels give
#: way to the photo's own.
FEATHER: Final[float] = 0.3
#: Below this radius in pixels a stroke covers nothing worth mending, which
#: is what a small thumbnail of a photo with fine dust spots gives.
SMALLEST_RADIUS_PX: Final[float] = 0.5

# --------------------------------------------------------------------------
# Choosing a source
# --------------------------------------------------------------------------

#: How far from the stroke a source is looked for, in radii. Two radii would
#: touch; a little more leaves room for the soft edge.
SOURCE_DISTANCES: Final[tuple[float, ...]] = (2.4, 3.2, 4.2, 5.4)
#: How many directions are tried at each distance.
SOURCE_DIRECTIONS: Final[int] = 24
#: How near, in radii, the source may come to the stroke: past its edge and
#: the soft part around it, so neither is copied onto itself.
SOURCE_CLEARANCE: Final[float] = 1.2
#: The radius, in pixels, of the shrunk copy the search compares on.
SEARCH_RADIUS: Final[float] = 8.0
#: How much a difference in texture counts beside a difference in the
#: surroundings themselves, and how much a farther source is held against.
TEXTURE_WEIGHT: Final[float] = 0.02
DISTANCE_WEIGHT: Final[float] = 0.002


@dataclass(frozen=True)
class Spot:
    """One stroke of the brush and where its texture is taken from.

    The points are fractions of the photo's width and height, after the lens
    is corrected and before anything turns or crops it; the radius is a
    fraction of the long edge; the offset is how far the source lies from
    the stroke, in fractions of the width and height (PLAN.md 7.1).
    """

    points: tuple[Point, ...]
    radius: float = RADIUS_DEFAULT
    offset: Point = (0.0, 0.0)

    @classmethod
    def dot(
        cls, x: float, y: float, source_x: float, source_y: float, radius: float = RADIUS_DEFAULT
    ) -> "Spot":
        """Return the stroke of one click at (*x*, *y*), taken from the source given."""
        return cls(points=((x, y),), radius=radius, offset=(source_x - x, source_y - y))

    @property
    def target(self) -> Point:
        """Return where the stroke starts, which is what dragging it holds."""
        return self.points[0]

    @property
    def source(self) -> Point:
        """Return where the source of the stroke's first point is."""
        return self.points[0][0] + self.offset[0], self.points[0][1] + self.offset[1]

    @property
    def source_points(self) -> tuple[Point, ...]:
        """Return the path the texture is taken along."""
        dx, dy = self.offset
        return tuple((x + dx, y + dy) for x, y in self.points)

    def moved_to(self, point: Point) -> "Spot":
        """Return this stroke moved so it starts at *point*; the source comes along."""
        dx = point[0] - self.points[0][0]
        dy = point[1] - self.points[0][1]
        moved = tuple((_fraction(x + dx), _fraction(y + dy)) for x, y in self.points)
        return replace(self, points=moved)

    def with_source(self, point: Point) -> "Spot":
        """Return this stroke taking its first point's texture from *point*."""
        x, y = self.points[0]
        return replace(self, offset=(_shift(point[0] - x), _shift(point[1] - y)))

    def with_radius(self, radius: float) -> "Spot":
        """Return this stroke at another size, held within the allowed range."""
        return replace(self, radius=clamp_radius(radius))

    def to_dict(self) -> dict[str, Any]:
        """Return the stroke as it is stored (PLAN.md 7.1)."""
        return {
            "points": [[x, y] for x, y in self.points],
            "radius": self.radius,
            "offset": [self.offset[0], self.offset[1]],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Spot | None":
        """Build a stroke from stored data, or return ``None`` if it is not one.

        A spot stored by 0.28.0, the first version, was a circle with a
        centre and a source centre; it reads as a stroke of one point.
        """
        if not isinstance(data, dict):
            return None
        radius = _as_float(data.get("radius"))
        if radius is None:
            return None
        if "points" not in data:
            values = [_as_float(data.get(name)) for name in ("x", "y", "source_x", "source_y")]
            if any(value is None for value in values):
                return None
            x, y, source_x, source_y = (float(value) for value in values)  # type: ignore[arg-type]
            return cls.dot(
                _fraction(x), _fraction(y), _fraction(source_x), _fraction(source_y),
                clamp_radius(radius),
            )
        raw = data.get("points")
        offset = data.get("offset")
        if not isinstance(raw, list) or not raw or not isinstance(offset, list | tuple):
            return None
        points: list[Point] = []
        for item in raw[:POINTS_MAX]:
            if not isinstance(item, list | tuple) or len(item) != 2:
                return None
            x, y = _as_float(item[0]), _as_float(item[1])
            if x is None or y is None:
                return None
            points.append((_fraction(x), _fraction(y)))
        if len(offset) != 2:
            return None
        dx, dy = _as_float(offset[0]), _as_float(offset[1])
        if dx is None or dy is None:
            return None
        return cls(points=tuple(points), radius=clamp_radius(radius), offset=(_shift(dx), _shift(dy)))


def spots_to_list(spots: tuple[Spot, ...]) -> list[dict[str, Any]]:
    """Return *spots* as they are stored."""
    return [spot.to_dict() for spot in spots]


def spots_from_list(data: Any) -> tuple[Spot, ...]:
    """Read stored strokes, leaving out anything that is not one."""
    if not isinstance(data, list):
        return ()
    spots = (Spot.from_dict(item) for item in data[:SPOTS_MAX])
    return tuple(spot for spot in spots if spot is not None)


def clamp_radius(radius: float) -> float:
    """Return *radius* held between the smallest and the largest brush."""
    return max(RADIUS_MIN, min(RADIUS_MAX, float(radius)))


def thinned(points: Sequence[Point], radius: float, size: tuple[int, int]) -> tuple[Point, ...]:
    """Return the path *points* with points nearer than a quarter radius left out.

    *size* is the photo's width and height, which the fractions are of; the
    last point is always kept, so the stroke ends where the pointer did. A
    path still longer than :data:`POINTS_MAX` is evened out to that many.
    """
    if not points:
        return ()
    width, height = size
    spacing = radius * max(width, height) * POINT_SPACING
    kept = [points[0]]
    for point in points[1:]:
        last = kept[-1]
        if math.hypot((point[0] - last[0]) * width, (point[1] - last[1]) * height) >= spacing:
            kept.append(point)
    if len(points) > 1 and kept[-1] != points[-1]:
        kept.append(points[-1])
    if len(kept) > POINTS_MAX:
        step = (len(kept) - 1) / (POINTS_MAX - 1)
        kept = [kept[round(index * step)] for index in range(POINTS_MAX)]
    return tuple(kept)


# --------------------------------------------------------------------------
# Healing
# --------------------------------------------------------------------------


def heal(photo: "DecodedPhoto", spots: tuple[Spot, ...]) -> "DecodedPhoto":
    """Return *photo* with every stroke mended (PLAN.md 6 step 1.6).

    The strokes are mended one after another, so a later one may take its
    texture from where an earlier one has mended. A photo with no strokes
    comes back as it is.

    A JPEG's original encoded pixels are mended the same way, so a photo
    whose only change is its strokes is still shown and exported pixel for
    pixel everywhere outside them (PLAN.md section 6, "JPG 原圖直通"):
    taking it through the colour steps instead would change every pixel a
    little for the sake of a few strokes.
    """
    if not spots:
        return photo
    image, clipped = heal_image(photo.image, photo.clipped, spots)
    encoded = photo.encoded
    if encoded is not None:
        scale = np.float32(np.iinfo(np.uint8).max)
        mended, _ = heal_image(encoded.astype(np.float32) / scale, None, spots)
        encoded = np.clip(np.rint(mended * scale), 0, scale).astype(np.uint8)
    return replace(photo, image=image, clipped=clipped, encoded=encoded)


def heal_image(
    image: NDArray[np.float32], clipped: NDArray[np.bool_] | None, spots: tuple[Spot, ...]
) -> tuple[NDArray[np.float32], NDArray[np.bool_] | None]:
    """Return copies of *image* and its clipped mask with every stroke mended."""
    healed = np.array(image, dtype=np.float32, copy=True)
    mask = None if clipped is None else np.array(clipped, dtype=np.bool_, copy=True)
    for spot in spots:
        _heal_one(healed, mask, spot)
    return healed, mask


def _in_pixels(points: Sequence[Point], width: int, height: int) -> NDArray[np.float64]:
    """Return fractions of a frame as pixels, the middle of the first at 0."""
    return np.array([(x * width - 0.5, y * height - 0.5) for x, y in points], dtype=np.float64)


def _heal_one(image: NDArray[np.float32], clipped: NDArray[np.bool_] | None, spot: Spot) -> None:
    """Mend one stroke of *image*, in place."""
    height, width = image.shape[:2]
    radius = spot.radius * max(width, height)
    if radius < SMALLEST_RADIUS_PX:
        return
    path = _in_pixels(spot.points, width, height)
    shift_x = spot.offset[0] * width
    shift_y = spot.offset[1] * height

    reach = int(math.ceil(radius * SURROUNDINGS_REACH)) + 2
    left = max(0, int(math.floor(path[:, 0].min())) - reach)
    top = max(0, int(math.floor(path[:, 1].min())) - reach)
    right = min(width, int(math.ceil(path[:, 0].max())) + reach + 1)
    bottom = min(height, int(math.ceil(path[:, 1].max())) + reach + 1)
    if right - left < 2 or bottom - top < 2:
        return

    target = image[top:bottom, left:right]
    size = (right - left, bottom - top)
    middle = ((left + right - 1) / 2 + shift_x, (top + bottom - 1) / 2 + shift_y)
    source = cv2.getRectSubPix(np.ascontiguousarray(image), size, middle)

    distance = _distance_to_path(path, (left, top, right, bottom), reach) / np.float32(radius)

    floor = np.float32(LOG_FLOOR)
    target_log = np.log(np.maximum(target, 0.0) + floor)
    source_log = np.log(np.maximum(source, 0.0) + floor)
    correction = _correction(target_log - source_log, distance, radius)

    mended = np.exp(source_log + correction) - floor
    weight = _feather(distance)[..., None]
    image[top:bottom, left:right] = target + (mended - target) * weight

    if clipped is not None:
        source_clipped = cv2.getRectSubPix(
            np.ascontiguousarray(clipped, dtype=np.float32), size, middle
        )
        inside = weight[..., 0] > 0.5
        region = clipped[top:bottom, left:right]
        region[inside] = source_clipped[inside] > 0.5


def _distance_to_path(
    path: NDArray[np.float64], box: tuple[int, int, int, int], reach: float | None = None
) -> NDArray[np.float32]:
    """Return how far each pixel of *box* is from the polyline *path*, in pixels.

    *box* is left, top, right and bottom, right and bottom exclusive. Each
    segment only works out the part of the box within *reach* of it, so a
    long stroke costs what its own surroundings cost; farther pixels keep
    the farthest distance any part did.
    """
    left, top, right, bottom = box
    xs = np.arange(left, right, dtype=np.float32)
    ys = np.arange(top, bottom, dtype=np.float32)
    distance = np.full((bottom - top, right - left), np.inf, dtype=np.float32)
    if len(path) == 1:
        x, y = path[0]
        return np.hypot(xs[None, :] - np.float32(x), ys[:, None] - np.float32(y)).astype(
            np.float32
        )
    limit = float(max(right - left, bottom - top)) if reach is None else reach
    for (ax, ay), (bx, by) in zip(path[:-1], path[1:], strict=True):
        x0 = max(left, int(math.floor(min(ax, bx) - limit)))
        x1 = min(right, int(math.ceil(max(ax, bx) + limit)) + 1)
        y0 = max(top, int(math.floor(min(ay, by) - limit)))
        y1 = min(bottom, int(math.ceil(max(ay, by) + limit)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        px = xs[x0 - left : x1 - left][None, :] - np.float32(ax)
        py = ys[y0 - top : y1 - top][:, None] - np.float32(ay)
        dx = np.float32(bx - ax)
        dy = np.float32(by - ay)
        length = dx * dx + dy * dy
        if length > 0:
            t = np.clip((px * dx + py * dy) / length, 0.0, 1.0)
        else:
            t = np.zeros_like(px * py)
        part = np.hypot(px - t * dx, py - t * dy)
        window = distance[y0 - top : y1 - top, x0 - left : x1 - left]
        np.minimum(window, part, out=window)
    return distance


def _feather(distance: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return how much of the mended picture shows, by distance in radii."""
    start = 1.0 - FEATHER
    t = np.clip((1.0 - distance) / np.float32(FEATHER), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    return np.where(distance <= start, np.float32(1.0), weight).astype(np.float32)


def _correction(
    difference: NDArray[np.float32], distance: NDArray[np.float32], radius: float
) -> NDArray[np.float32]:
    """Return the smooth field that meets *difference* around the stroke.

    Around the stroke it is the difference itself, blurred so that it
    carries the light and the colour but not the texture; inside, the
    solution of Laplace's equation with that as its edge. Only the inside is
    ever used - the soft edge hands the photo back by the time the stroke
    ends - so all of it is worked out on a copy shrunk until the radius is
    :data:`SOLVE_RADIUS` pixels, and stretched back.
    """
    height, width = difference.shape[:2]
    known = (distance >= 1.0).astype(np.float32)
    scale = min(1.0, SOLVE_RADIUS / radius)
    if scale < 1.0:
        size = (max(3, round(width * scale)), max(3, round(height * scale)))
        # Shrunk as a weighted mean, so the blot inside the stroke does not
        # bleed into the pixels just outside it.
        weighted = cv2.resize(difference * known[..., None], size, interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(known, size, interpolation=cv2.INTER_AREA)
        values = weighted / np.maximum(coverage, np.float32(1e-6))[..., None]
        small_known = coverage > 0.999
    else:
        values = difference
        small_known = known > 0.5

    sigma = max(0.75, radius * scale * SURROUNDINGS_BLUR)
    usable = small_known.astype(np.float32)
    blurred = cv2.GaussianBlur(values * usable[..., None], (0, 0), sigma)
    weight = cv2.GaussianBlur(usable, (0, 0), sigma)[..., None]
    smooth = blurred / np.maximum(weight, np.float32(1e-6))
    solved = _laplace(smooth, small_known)
    if scale < 1.0:
        solved = cv2.resize(solved, (width, height), interpolation=cv2.INTER_LINEAR)
    return solved.astype(np.float32)


def _laplace(values: NDArray[np.float32], known: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Return *values* with the pixels not *known* filled by Laplace's equation.

    Every unknown pixel becomes the mean of its neighbours, which is one
    linear equation per pixel; the edge of the picture counts as a mirror.
    A few hundred of them are solved directly; more, by conjugate gradients.
    """
    unknown = ~known
    count = int(unknown.sum())
    filled = np.array(values, dtype=np.float32, copy=True)
    if count == 0 or count == known.size:
        # Nothing to fill, or nothing to meet.
        return filled
    if count <= DIRECT_UNKNOWNS:
        return _laplace_direct(filled, known)
    return _laplace_iterative(filled, known)


def _laplace_direct(values: NDArray[np.float32], known: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Solve Laplace's equation for the unknown pixels as one linear system."""
    height, width = known.shape
    unknown = np.flatnonzero(~known.ravel())
    index = np.full(known.size, -1, dtype=np.int64)
    index[unknown] = np.arange(unknown.size)
    flat = values.reshape(known.size, -1).astype(np.float64)
    count = unknown.size
    system = np.zeros((count, count), dtype=np.float64)
    right_side = np.zeros((count, flat.shape[1]), dtype=np.float64)
    rows = unknown // width
    cols = unknown % width
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_rows = rows + d_row
        n_cols = cols + d_col
        inside = (n_rows >= 0) & (n_rows < height) & (n_cols >= 0) & (n_cols < width)
        own = np.flatnonzero(inside)
        system[own, own] += 1.0
        neighbour = n_rows[own] * width + n_cols[own]
        other = index[neighbour]
        free = other >= 0
        system[own[free], other[free]] -= 1.0
        right_side[own[~free]] += flat[neighbour[~free]]
    flat[unknown] = np.linalg.solve(system, right_side)
    return flat.reshape(values.shape).astype(np.float32)


def _neighbour_sum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the sum of each pixel's four neighbours, nothing beyond the edge."""
    padded = np.pad(values, ((1, 1), (1, 1)) + ((0, 0),) * (values.ndim - 2))
    return padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]


def _laplace_iterative(
    values: NDArray[np.float32], known: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Solve Laplace's equation for the unknown pixels by conjugate gradients.

    The same equations as :func:`_laplace_direct`, without the matrix: each
    unknown pixel times the number of neighbours it has, less the unknown
    neighbours, equals the sum of the known ones. Started from the mean of
    the surroundings, which is close everywhere for a narrow stroke.
    """
    free = (~known).astype(np.float64)
    degree = _neighbour_sum(np.ones(known.shape, dtype=np.float64))
    filled = np.array(values, dtype=np.float32, copy=True)
    for channel in range(values.shape[2]):
        plane = values[..., channel].astype(np.float64)
        fixed = plane * (1.0 - free)
        right_side = _neighbour_sum(fixed) * free
        start = float(plane[known].mean())
        x = np.full(plane.shape, start) * free

        def apply(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return (degree * v - _neighbour_sum(v * free)) * free

        residual = right_side - apply(x)
        direction = residual.copy()
        before = float(np.sum(residual * residual))
        threshold = before * ITERATIVE_TOLERANCE**2
        for _step in range(ITERATIVE_STEPS):
            if before <= threshold or before == 0.0:
                break
            moved = apply(direction)
            step = before / float(np.sum(direction * moved))
            x += step * direction
            residual -= step * moved
            after = float(np.sum(residual * residual))
            direction = residual + (after / before) * direction
            before = after
        filled[..., channel] = np.where(known, plane, x).astype(np.float32)
    return filled


# --------------------------------------------------------------------------
# Choosing a source
# --------------------------------------------------------------------------


def find_offset(
    image: NDArray[np.float32], points: Sequence[Point], radius: float
) -> Point:
    """Return how far from the stroke along *points* its texture should come from.

    *points* are fractions of *image*, *radius* a fraction of its long edge,
    and the answer a shift in fractions of the width and height, as a
    :class:`Spot` holds them. The same shape is tried at a few distances and
    in many directions, never on or touching the stroke itself, and the one
    chosen is where the surroundings match the stroke's surroundings best -
    the light, the colour and the way they run - with the texture inside it
    as fine as the texture around the stroke, and nearer winning a close
    call. The search is run on a copy shrunk so the radius is a few pixels,
    so it costs the same for any size of brush.
    """
    height, width = image.shape[:2]
    radius_px = max(radius * max(width, height), SMALLEST_RADIUS_PX)
    path = _in_pixels(points, width, height)

    reach = radius_px * (max(SOURCE_DISTANCES) + SURROUNDINGS_REACH + 1) + 2
    left = max(0, int(math.floor(path[:, 0].min() - reach)))
    top = max(0, int(math.floor(path[:, 1].min() - reach)))
    right = min(width, int(math.ceil(path[:, 0].max() + reach)) + 1)
    bottom = min(height, int(math.ceil(path[:, 1].max() + reach)) + 1)
    scale = min(1.0, SEARCH_RADIUS / radius_px)
    part = np.maximum(image[top:bottom, left:right], 0.0)
    if scale < 1.0:
        size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
        part = cv2.resize(part, size, interpolation=cv2.INTER_AREA)
    part = np.log(part + np.float32(LOG_FLOOR))
    small_height, small_width = part.shape[:2]
    scale_x = small_width / (right - left)
    scale_y = small_height / (bottom - top)
    small_radius = radius_px * scale_x
    small_path = np.column_stack(
        ((path[:, 0] - left + 0.5) * scale_x - 0.5, (path[:, 1] - top + 0.5) * scale_y - 0.5)
    )
    distance = (
        _distance_to_path(
            small_path,
            (0, 0, small_width, small_height),
            small_radius * (1.0 + SOURCE_CLEARANCE + SURROUNDINGS_REACH),
        )
        / small_radius
    )
    inside = distance < 1.0
    ring = (distance >= 1.0) & (distance <= SURROUNDINGS_REACH)
    near = (distance < 1.0 + SOURCE_CLEARANCE).astype(np.float32)
    detail = part - cv2.GaussianBlur(part, (0, 0), max(0.75, small_radius * 0.5))
    target_texture = float(np.std(detail[ring])) if ring.any() else 0.0
    if not ring.any() or not inside.any():
        ring = distance <= SURROUNDINGS_REACH
    inside_float = inside.astype(np.float32)

    best: Point | None = None
    best_score = math.inf
    frame = (small_width, small_height)
    for distance_factor in SOURCE_DISTANCES:
        for step in range(SOURCE_DIRECTIONS):
            angle = 2.0 * math.pi * step / SOURCE_DIRECTIONS
            shift_x = math.cos(angle) * radius_px * distance_factor
            shift_y = math.sin(angle) * radius_px * distance_factor
            moved = path + (shift_x, shift_y)
            if not (
                moved[:, 0].min() + 0.5 >= radius_px
                and moved[:, 0].max() + 0.5 <= width - radius_px
                and moved[:, 1].min() + 0.5 >= radius_px
                and moved[:, 1].max() + 0.5 <= height - radius_px
            ):
                continue
            small_shift = (shift_x * scale_x, shift_y * scale_y)
            # Where the source would be: the stroke's own shape, moved.
            onto = np.float32([[1, 0, small_shift[0]], [0, 1, small_shift[1]]])
            source_area = cv2.warpAffine(inside_float, onto, frame, flags=cv2.INTER_LINEAR)
            if float(np.sum(source_area * near)) > 0.0:
                continue
            # What lies under the stroke and its surroundings, moved there.
            back = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
            shifted = cv2.warpAffine(part, onto, frame, flags=back, borderMode=cv2.BORDER_REFLECT)
            shifted_detail = cv2.warpAffine(
                detail, onto, frame, flags=back, borderMode=cv2.BORDER_REFLECT
            )
            surroundings = float(np.mean((shifted[ring] - part[ring]) ** 2))
            texture = float(np.std(shifted_detail[inside])) if inside.any() else 0.0
            texture_term = math.log((texture + 1e-3) / (target_texture + 1e-3)) ** 2
            score = (
                surroundings
                + TEXTURE_WEIGHT * texture_term
                + DISTANCE_WEIGHT * distance_factor
            )
            if score < best_score:
                best_score = score
                best = (shift_x, shift_y)
    if best is None:
        # A stroke with nowhere clear around it: take the far side of the frame.
        centre_x, centre_y = path.mean(axis=0)
        best = (width / 2 - centre_x, height / 2 - centre_y)
        if math.hypot(*best) < 2 * radius_px:
            best = (0.0, 2.4 * radius_px if centre_y < height / 2 else -2.4 * radius_px)
    return _shift(best[0] / width), _shift(best[1] / height)


def _fraction(value: float) -> float:
    """Return *value* held within the frame."""
    return max(0.0, min(1.0, float(value)))


def _shift(value: float) -> float:
    """Return a shift held to at most the frame's own size."""
    return max(-1.0, min(1.0, float(value)))


def _as_float(value: Any) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = [
    "RADIUS_DEFAULT",
    "RADIUS_MAX",
    "RADIUS_MIN",
    "Spot",
    "clamp_radius",
    "find_offset",
    "heal",
    "heal_image",
    "spots_from_list",
    "spots_to_list",
    "thinned",
]
