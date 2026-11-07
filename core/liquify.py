"""Liquify: pushing, swelling and turning parts of the picture with a brush.

PLAN.md section 6 step 1.7. Every stroke of the brush changes one field of
displacements laid over the photo: for each point of the picture it says
where on the photo the colour is fetched from. The picture is then taken
through that field once, so a hundred strokes cost one resampling.

* The field is held on a grid of :data:`GRID_LONG` nodes along the long
  edge, in units of the long edge, whatever size the photo is decoded at.
  The preview proxy, a window of the full size photo and the exported file
  are moved by the same numbers.
* A stroke is a row of dabs. Each dab is a small warp of its own - a push
  along the way the brush went, a swell or a pinch about its centre, a turn
  about it - and is put after what is already there: the new field fetches
  from where the dab says, and from there where the old field said.
  Restoring and smoothing work on the field itself.
* Every dab moves the picture by little enough that it cannot fold over on
  itself, so a long stroke, which is many dabs, cannot either.
* The edges of the photo are held: nothing is fetched from beyond them, and
  the outermost row and column only slide along themselves.

The idea is the displacement field of every warping tool, worked out here
from that idea rather than from anyone's code.

Strokes are held like the spots of :mod:`core.healing`: in fractions of the
photo after its lens correction and before it is turned or cropped, so they
stay on what they shaped. They are put after the spots, so a mended spot
moves with what is pushed around it.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from core.decoder import DecodedPhoto
    from core.skin_range import SkinRange

Point = tuple[float, float]

# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

#: Carries what is under the brush along the way it is dragged.
TOOL_PUSH: Final[str] = "push"
#: Swells what is under the brush out from its centre.
TOOL_BLOAT: Final[str] = "bloat"
#: Draws what is under the brush in toward its centre.
TOOL_PUCKER: Final[str] = "pucker"
#: Turns what is under the brush about its centre, clockwise on the photo
#: as it is held, before any mirror.
TOOL_TWIRL_CW: Final[str] = "twirl_cw"
TOOL_TWIRL_CCW: Final[str] = "twirl_ccw"
#: Takes what the other tools did back toward the photo as it was.
TOOL_RECONSTRUCT: Final[str] = "reconstruct"
#: Evens out what the other tools did, keeping its overall shape.
TOOL_SMOOTH: Final[str] = "smooth"

#: The tools, in the order the panel shows them (PLAN.md 3.16).
TOOLS: Final[tuple[str, ...]] = (
    TOOL_PUSH,
    TOOL_BLOAT,
    TOOL_PUCKER,
    TOOL_TWIRL_CW,
    TOOL_TWIRL_CCW,
    TOOL_RECONSTRUCT,
    TOOL_SMOOTH,
)

#: What a mirror makes of each tool: turning one way, seen in a mirror, is
#: turning the other.
MIRRORED: Final[dict[str, str]] = {
    TOOL_TWIRL_CW: TOOL_TWIRL_CCW,
    TOOL_TWIRL_CCW: TOOL_TWIRL_CW,
}

# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------

#: The radius of the brush, as a fraction of the long edge. The smallest is
#: thirty pixels on a 6000 pixel photo, about the size of an eye in a half
#: length portrait, and still five nodes of the grid; the largest a quarter
#: of the frame, for the line of a whole body.
RADIUS_MIN: Final[float] = 0.005
RADIUS_MAX: Final[float] = 0.25
RADIUS_DEFAULT: Final[float] = 0.05
#: How hard the brush works, from barely to fully.
STRENGTH_MIN: Final[float] = 0.01
STRENGTH_MAX: Final[float] = 1.0
STRENGTH_DEFAULT: Final[float] = 0.5
#: More strokes than this in a parameter file is not something the interface
#: could have made, and the rest are left out.
STROKES_MAX: Final[int] = 1000
#: The most points one stroke keeps.
POINTS_MAX: Final[int] = 2000
#: How close together a push keeps its points, in radii (as the spots do).
POINT_SPACING: Final[float] = 0.25

# --------------------------------------------------------------------------
# The field
# --------------------------------------------------------------------------

#: Nodes along the long edge of the grid the field is held on. A node every
#: six pixels of a 6000 pixel photo: the smallest brush spans ten of them,
#: and the field between them is smooth enough to be read bilinearly.
GRID_LONG: Final[int] = 1024
#: How far apart the dabs of a push are, in radii. A dab moves the picture
#: by this much at most, and the steepest the falloff gets is about 1.54
#: per radius, so the most a dab stretches anything is 0.2 x 1.54 = 0.31:
#: well short of 1, where it would fold.
PUSH_SPACING: Final[float] = 0.2
#: How far apart the dabs of the other tools are along a path, in radii.
DAB_SPACING: Final[float] = 0.25
#: What one dab of each tool does at full strength. The interface adds a
#: point to a held brush :data:`TICKS_PER_SECOND` times a second, so these
#: are also how fast a brush held still works.
#:
#: * swell and pinch: the share of the distance to the centre that is
#:   fetched from nearer it (or farther away); 2 % a dab, so holding the
#:   brush still for a second at full strength makes the middle about 1.8
#:   times the size;
#: * turn: radians at the centre;
#: * restore: the share of the displacement taken away at the centre;
#: * smooth: the share of the way to the blurred field at the centre.
BLOAT_RATE: Final[float] = 0.02
TWIRL_RATE: Final[float] = 0.05
RECONSTRUCT_RATE: Final[float] = 0.1
SMOOTH_RATE: Final[float] = 0.25
#: The blur the smoothing evens the field toward, in radii.
SMOOTH_SIGMA: Final[float] = 0.3
#: Past this many nodes the blur is worked out on a shrunk copy.
SMOOTH_BLUR_NODES: Final[float] = 4.0
#: How often the interface adds a point while the brush is held (PLAN.md
#: 3.16): the rates above are per point.
TICKS_PER_SECOND: Final[int] = 30
#: The draft of a stroke being painted is worked out on a coarser grid:
#: this many nodes across the brush's radius, which keeps a dab the same
#: few hundred nodes whatever the brush's size, and never fewer than
#: :data:`DRAFT_GRID_LEAST` along the long edge.
DRAFT_NODES_PER_RADIUS: Final[int] = 8
DRAFT_GRID_LEAST: Final[int] = 128
#: Fields kept for the strokes most recently asked about: twelve undo steps
#: of a 24 megapixel photo, about 70 MB.
CACHE_SIZE: Final[int] = 12
#: A pixel moved by less than this many pixels is left exactly as it was,
#: which keeps a JPEG pixel for pixel outside what the brush reached.
STILL_PX: Final[float] = 1e-3
#: Rows of pixels taken through the field at a time, which keeps the maps a
#: full size photo needs to a few megabytes.
STRIP_ROWS: Final[int] = 256
#: How many steps the inverse is allowed, and how closely it has to land,
#: in units of the long edge.
INVERSE_STEPS: Final[int] = 30
INVERSE_TOLERANCE: Final[float] = 1e-7
#: The most nodes the inverse looks through for where to start, and how
#: many points it compares with them at a time.
INVERSE_SEARCH_NODES: Final[int] = 40000
INVERSE_SEARCH_POINTS: Final[int] = 256


@dataclass(frozen=True)
class Stroke:
    """One stroke of the liquify brush (PLAN.md 7.1).

    The points are fractions of the photo's width and height, after the lens
    is corrected and before anything turns or crops it; the radius is a
    fraction of the long edge; the strength runs from
    :data:`STRENGTH_MIN` to 1.
    """

    tool: str
    points: tuple[Point, ...]
    radius: float = RADIUS_DEFAULT
    strength: float = STRENGTH_DEFAULT

    @cached_property
    def _hash(self) -> int:
        # A tuple does not keep its hash, and a stroke holds up to two
        # thousand points; the field cache asks for it on every look up.
        return hash((self.tool, self.points, self.radius, self.strength))

    def __hash__(self) -> int:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        """Return the stroke as it is stored (PLAN.md 7.1)."""
        return {
            "tool": self.tool,
            "points": [[x, y] for x, y in self.points],
            "radius": self.radius,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Stroke | None":
        """Build a stroke from stored data, or return ``None`` if it is not one."""
        if not isinstance(data, dict):
            return None
        tool = data.get("tool")
        radius = _as_float(data.get("radius"))
        strength = _as_float(data.get("strength"))
        raw = data.get("points")
        if tool not in TOOLS or radius is None or strength is None:
            return None
        if not isinstance(raw, list) or not raw:
            return None
        points: list[Point] = []
        for item in raw[:POINTS_MAX]:
            if not isinstance(item, list | tuple) or len(item) != 2:
                return None
            x, y = _as_float(item[0]), _as_float(item[1])
            if x is None or y is None:
                return None
            points.append((_fraction(x), _fraction(y)))
        return cls(
            tool=tool,
            points=tuple(points),
            radius=clamp_radius(radius),
            strength=clamp_strength(strength),
        )


def strokes_to_list(strokes: tuple[Stroke, ...]) -> list[dict[str, Any]]:
    """Return *strokes* as they are stored."""
    return [stroke.to_dict() for stroke in strokes]


def strokes_from_list(data: Any) -> tuple[Stroke, ...]:
    """Read stored strokes, leaving out anything that is not one."""
    if not isinstance(data, list):
        return ()
    strokes = (Stroke.from_dict(item) for item in data[:STROKES_MAX])
    return tuple(stroke for stroke in strokes if stroke is not None)


def clamp_radius(radius: float) -> float:
    """Return *radius* held between the smallest and the largest brush."""
    return max(RADIUS_MIN, min(RADIUS_MAX, float(radius)))


def clamp_strength(strength: float) -> float:
    """Return *strength* held between the gentlest brush and the full one."""
    return max(STRENGTH_MIN, min(STRENGTH_MAX, float(strength)))


def mirrored_tool(tool: str) -> str:
    """Return the tool that does on the photo what *tool* does in its mirror image."""
    return MIRRORED.get(tool, tool)


def thinned(
    tool: str, points: Sequence[Point], radius: float, size: tuple[int, int]
) -> tuple[Point, ...]:
    """Return the path *points* as a stroke of *tool* keeps it.

    A push is about where the brush went, so points nearer together than a
    quarter radius are left out, as the spots leave them (the last is always
    kept). The other tools work for as long as the brush is held, which is
    what repeated points at the same place say, so all of them are kept.
    Either way a path longer than :data:`POINTS_MAX` is evened out to that
    many.
    """
    if not points:
        return ()
    kept = [points[0]]
    if tool == TOOL_PUSH:
        width, height = size
        spacing = radius * max(width, height) * POINT_SPACING
        for point in points[1:]:
            last = kept[-1]
            if math.hypot((point[0] - last[0]) * width, (point[1] - last[1]) * height) >= spacing:
                kept.append(point)
        if len(points) > 1 and kept[-1] != points[-1]:
            kept.append(points[-1])
    else:
        kept = list(points)
    if len(kept) > POINTS_MAX:
        step = (len(kept) - 1) / (POINTS_MAX - 1)
        kept = [kept[round(index * step)] for index in range(POINTS_MAX)]
    return tuple(kept)


# --------------------------------------------------------------------------
# The grid and the field
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The nodes a field is held on, for a photo of a given shape.

    The nodes are the same distance apart both ways, :data:`GRID_LONG` of
    them along the long edge unless a draft asks for fewer, and positions on
    the grid are measured in that long edge: node (row *i*, column *j*) sits
    at ``(j * step, i * step)``.
    The first and last nodes are on the photo's edges, so a fraction *x* of
    the width is ``x * extent[0]`` on the grid.
    """

    #: Nodes across and down.
    nodes: tuple[int, int]

    @property
    def per_unit(self) -> int:
        """Return how many node spacings the long edge holds."""
        return max(self.nodes) - 1

    @property
    def step(self) -> tuple[float, float]:
        """Return how far apart the nodes are, across and down."""
        step = 1.0 / self.per_unit
        return step, step

    @property
    def extent(self) -> tuple[float, float]:
        """Return the grid's width and height: one of them is 1."""
        step = 1.0 / self.per_unit
        return (self.nodes[0] - 1) * step, (self.nodes[1] - 1) * step


def grid_for(size: tuple[int, int], long_nodes: int = GRID_LONG) -> Grid:
    """Return the grid for a photo of *size* (width, height).

    It depends only on the photo's shape, to the nearest node, so the
    proxy, the screen sized preview and the full size photo of the same
    picture share one grid and one field. The short edge's shape is off by
    half a node at most, a twentieth of a per cent. A draft asks for
    *long_nodes* fewer than :data:`GRID_LONG`.
    """
    width, height = max(int(size[0]), 1), max(int(size[1]), 1)
    long_edge = max(width, height)
    spacings = max(2, int(long_nodes)) - 1
    across = max(2, round(spacings * width / long_edge) + 1)
    down = max(2, round(spacings * height / long_edge) + 1)
    return Grid(nodes=(across, down))


@dataclass(frozen=True, eq=False)
class Field:
    """Where each point of the picture is fetched from, as a shift.

    ``values[0]`` and ``values[1]`` are the shift across and down at each
    node of *grid* (rows, columns), in units of the long edge: the picture
    at a point *p* is the photo at *p* plus the shift there. *box* is the
    rows and columns, end exclusive, outside which every shift is zero.
    """

    grid: Grid
    values: NDArray[np.float32]
    box: tuple[int, int, int, int]


_cache: OrderedDict[tuple[tuple[Stroke, ...], Grid], Field | None] = OrderedDict()
_cache_lock = threading.Lock()


def field(strokes: tuple[Stroke, ...], size: tuple[int, int]) -> Field | None:
    """Return the field *strokes* make on a photo of *size*, or ``None`` if they move nothing.

    The field is put together stroke by stroke from the longest run of the
    same strokes already worked out, so adding a stroke costs only that
    stroke and taking one back (undo) usually costs nothing.
    """
    if not strokes:
        return None
    grid = grid_for(size)
    key = (strokes, grid)
    start: NDArray[np.float32] | None = None
    done = 0
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        for (kept, kept_grid), kept_field in _cache.items():
            count = len(kept)
            if kept_grid == grid and done < count <= len(strokes) and strokes[:count] == kept:
                done = count
                start = None if kept_field is None else kept_field.values
    values = (
        np.zeros((2, grid.nodes[1], grid.nodes[0]), dtype=np.float32)
        if start is None
        else start.copy()
    )
    result = _built(values, grid, strokes[done:])
    with _cache_lock:
        _cache[key] = result
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def build_field(strokes: Sequence[Stroke], grid: Grid) -> Field | None:
    """Return the field *strokes* make on *grid*, worked out afresh and kept nowhere.

    What the draft of a stroke still being painted uses (PLAN.md 3.16): it
    changes with every point, so keeping it would only push the fields worth
    keeping out of the cache.
    """
    values = np.zeros((2, grid.nodes[1], grid.nodes[0]), dtype=np.float32)
    return _built(values, grid, strokes)


def _built(values: NDArray[np.float32], grid: Grid, strokes: Sequence[Stroke]) -> Field | None:
    """Add *strokes* to *values* in place and return the field they make."""
    for stroke in strokes:
        _apply_stroke(values, grid, stroke)
    return _finished(values, grid)


def _finished(values: NDArray[np.float32], grid: Grid) -> Field | None:
    """Return *values* as a field, or ``None`` if nothing in it moves."""
    moving = np.any(values != 0.0, axis=0)
    if not moving.any():
        return None
    rows = np.flatnonzero(moving.any(axis=1))
    columns = np.flatnonzero(moving.any(axis=0))
    values.setflags(write=False)
    return Field(
        grid=grid,
        values=values,
        box=(int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1),
    )


def _apply_stroke(values: NDArray[np.float32], grid: Grid, stroke: Stroke) -> None:
    """Add *stroke* to the field *values* on *grid*, in place."""
    if stroke.tool not in TOOLS:
        return
    across, down = grid.extent
    path = [(x * across, y * down) for x, y in stroke.points]
    radius = stroke.radius
    if stroke.tool == TOOL_PUSH:
        for start, end in zip(path, path[1:]):
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length == 0.0:
                continue
            count = max(1, math.ceil(length / (PUSH_SPACING * radius)))
            shift = ((end[0] - start[0]) / count, (end[1] - start[1]) / count)
            for step in range(1, count + 1):
                centre = (start[0] + shift[0] * step, start[1] + shift[1] * step)
                _dab(values, grid, stroke.tool, centre, radius, stroke.strength, shift)
        return
    _dab(values, grid, stroke.tool, path[0], radius, stroke.strength, (0.0, 0.0))
    for start, end in zip(path, path[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        count = max(1, math.ceil(length / (DAB_SPACING * radius)))
        for step in range(1, count + 1):
            share = step / count
            centre = (start[0] + (end[0] - start[0]) * share, start[1] + (end[1] - start[1]) * share)
            _dab(values, grid, stroke.tool, centre, radius, stroke.strength, (0.0, 0.0))


def falloff(distance: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return how much of a dab reaches *distance* (in radii) from its centre.

    ``(1 - t²)²``: all of it at the centre, none at the edge, and flat at
    both, so no dab leaves a crease where it ends.
    """
    inside = np.clip(np.float32(1.0) - distance * distance, 0.0, None)
    return (inside * inside).astype(np.float32)


def _dab(
    values: NDArray[np.float32],
    grid: Grid,
    tool: str,
    centre: Point,
    radius: float,
    strength: float,
    shift: Point,
) -> None:
    """Put one dab of *tool* on the field *values*, in place.

    *centre* and *radius* are in units of the long edge, and *shift* is how
    far a push's brush moved to get here.
    """
    across, down = grid.nodes
    step = grid.step[0]
    first_column = max(0, math.floor((centre[0] - radius) / step))
    last_column = min(across, math.ceil((centre[0] + radius) / step) + 1)
    first_row = max(0, math.floor((centre[1] - radius) / step))
    last_row = min(down, math.ceil((centre[1] + radius) / step) + 1)
    if first_column >= last_column or first_row >= last_row:
        return
    xs = np.arange(first_column, last_column, dtype=np.float32) * np.float32(step)
    ys = np.arange(first_row, last_row, dtype=np.float32) * np.float32(step)
    dx = xs[None, :] - np.float32(centre[0])
    dy = ys[:, None] - np.float32(centre[1])
    weight = falloff(np.sqrt(dx * dx + dy * dy) / np.float32(radius))
    if not weight.any():
        return
    patch = values[:, first_row:last_row, first_column:last_column]
    amount = np.float32(strength)

    if tool == TOOL_RECONSTRUCT:
        updated = patch * (np.float32(1.0) - amount * np.float32(RECONSTRUCT_RATE) * weight)
    elif tool == TOOL_SMOOTH:
        blurred = _smoothed(values, grid, (first_row, last_row, first_column, last_column), radius)
        updated = patch + (amount * np.float32(SMOOTH_RATE) * weight) * (blurred - patch)
    else:
        if tool == TOOL_PUSH:
            fetch_x = (-amount * np.float32(shift[0])) * weight
            fetch_y = (-amount * np.float32(shift[1])) * weight
        elif tool in (TOOL_BLOAT, TOOL_PUCKER):
            rate = amount * np.float32(BLOAT_RATE) * weight
            if tool == TOOL_BLOAT:
                rate = -rate
            fetch_x = rate * dx
            fetch_y = rate * dy
        else:
            # Turning the picture clockwise fetches each point from a little
            # anticlockwise of it (y points down, so a positive angle is
            # clockwise on screen).
            angle = amount * np.float32(TWIRL_RATE) * weight
            if tool == TOOL_TWIRL_CW:
                angle = -angle
            cosine = np.cos(angle)
            sine = np.sin(angle)
            fetch_x = dx * cosine - dy * sine - dx
            fetch_y = dx * sine + dy * cosine - dy
        old = _sample(values, grid, xs[None, :] + fetch_x, ys[:, None] + fetch_y)
        updated = np.stack((fetch_x + old[0], fetch_y + old[1]))

    # Hold the edges: nothing is fetched from beyond the photo, and the
    # outermost nodes only slide along their own edge.
    extent_x, extent_y = grid.extent
    updated[0] = np.clip(xs[None, :] + updated[0], 0.0, extent_x) - xs[None, :]
    updated[1] = np.clip(ys[:, None] + updated[1], 0.0, extent_y) - ys[:, None]
    if first_column == 0:
        updated[0, :, 0] = 0.0
    if last_column == across:
        updated[0, :, -1] = 0.0
    if first_row == 0:
        updated[1, 0, :] = 0.0
    if last_row == down:
        updated[1, -1, :] = 0.0
    # Outside the brush nothing changes, to the last bit.
    patch[...] = np.where(weight > 0.0, updated, patch)


def _smoothed(
    values: NDArray[np.float32],
    grid: Grid,
    box: tuple[int, int, int, int],
    radius: float,
) -> NDArray[np.float32]:
    """Return the field inside *box* blurred by :data:`SMOOTH_SIGMA` radii."""
    first_row, last_row, first_column, last_column = box
    sigma = SMOOTH_SIGMA * radius / grid.step[0]
    margin = max(1, math.ceil(3.0 * sigma))
    down, across = values.shape[1:]
    top = max(0, first_row - margin)
    bottom = min(down, last_row + margin)
    left = max(0, first_column - margin)
    right = min(across, last_column + margin)
    planes = []
    for plane in values[:, top:bottom, left:right]:
        around = np.ascontiguousarray(plane)
        if sigma > SMOOTH_BLUR_NODES:
            # The blur is wide and the field smooth: blur a shrunk copy and
            # stretch it back rather than run a kernel hundreds of nodes long.
            shrink = sigma / SMOOTH_BLUR_NODES
            size = (around.shape[1], around.shape[0])
            small = (max(2, round(size[0] / shrink)), max(2, round(size[1] / shrink)))
            shrunk = cv2.resize(around, small, interpolation=cv2.INTER_AREA)
            shrunk = cv2.GaussianBlur(
                shrunk, (0, 0), SMOOTH_BLUR_NODES, borderType=cv2.BORDER_REPLICATE
            )
            blurred = cv2.resize(shrunk, size, interpolation=cv2.INTER_LINEAR)
        else:
            blurred = cv2.GaussianBlur(around, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        planes.append(
            blurred[first_row - top : last_row - top, first_column - left : last_column - left]
        )
    return np.stack(planes)


def _sample(
    values: NDArray[np.float32],
    grid: Grid,
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return the field read bilinearly at points (*xs*, *ys*), in units of the long edge.

    What comes back is the two shifts, stacked in front of the shape of the
    points. Worked out here rather than by OpenCV, which rounds the position
    to a thirty second of a node: a field read back many times over would
    drift.
    """
    across, down = grid.nodes
    nodes_per_unit = np.float32(grid.per_unit)
    column = np.clip(xs * nodes_per_unit, 0.0, across - 1)
    row = np.clip(ys * nodes_per_unit, 0.0, down - 1)
    left = np.minimum(column.astype(np.intp), across - 2)
    top = np.minimum(row.astype(np.intp), down - 2)
    across_share = column - left.astype(np.float32)
    down_share = row - top.astype(np.float32)
    corner = top * across + left
    out = np.empty((2, *np.shape(xs)), dtype=np.float32)
    for plane, shifts in enumerate(values.reshape(2, -1)):
        upper_left = np.take(shifts, corner)
        upper_right = np.take(shifts, corner + 1)
        lower_left = np.take(shifts, corner + across)
        lower_right = np.take(shifts, corner + across + 1)
        upper = upper_left + (upper_right - upper_left) * across_share
        lower = lower_left + (lower_right - lower_left) * across_share
        out[plane] = upper + (lower - upper) * down_share
    return out


# --------------------------------------------------------------------------
# Moving the picture
# --------------------------------------------------------------------------


def warp(photo: "DecodedPhoto", strokes: tuple[Stroke, ...]) -> "DecodedPhoto":
    """Return *photo* taken through the field its strokes make (PLAN.md 6 step 1.7).

    The picture, its clipped mask and a JPEG's original encoded pixels are
    moved alike, and so are the faces found on it, so the skin sliders still
    find each face where it now is. Only the part the brush reached is
    resampled: everywhere else every pixel is exactly as it was, which keeps
    a JPEG shown and exported pixel for pixel there (PLAN.md section 6, "JPG
    原圖直通"). A photo with no strokes comes back as it is.
    """
    if not strokes:
        return photo
    height, width = photo.image.shape[:2]
    shape = field(strokes, (width, height))
    if shape is None:
        return photo
    image = warp_values(photo.image, shape, cv2.INTER_CUBIC)
    clipped = warp_values(photo.clipped.astype(np.uint8), shape, cv2.INTER_NEAREST) > 0
    encoded = (
        None if photo.encoded is None else warp_values(photo.encoded, shape, cv2.INTER_CUBIC)
    )
    faces = photo.faces
    if faces:
        faces = tuple(
            replace(face, points=shown_points(face.points, strokes, (width, height)))
            for face in faces
        )
    return replace(photo, image=image, clipped=clipped, encoded=encoded, faces=faces)


def warp_values(values: NDArray[Any], shape: Field, interpolation: int) -> NDArray[Any]:
    """Return a copy of *values* (rows, columns[, channels]) taken through *shape*."""
    height, width = values.shape[:2]
    first_row, last_row, first_column, last_column = shape.box
    across_nodes, down_nodes = shape.grid.nodes
    # A node reaches the pixels up to its neighbours; past them it is zero.
    left = _pixel(max(first_column - 1, 0), across_nodes, width, math.floor)
    right = _pixel(min(last_column, across_nodes - 1), across_nodes, width, math.ceil) + 1
    top = _pixel(max(first_row - 1, 0), down_nodes, height, math.floor)
    bottom = _pixel(min(last_row, down_nodes - 1), down_nodes, height, math.ceil) + 1
    out = np.array(values, copy=True)
    if left >= right or top >= bottom:
        return out

    # Pixels to the grid and back: a pixel's centre is a fraction of the
    # width (or height), and a fraction is that many grid extents.
    extent_x, extent_y = shape.grid.extent
    to_pixels_x = np.float32(width / extent_x)
    to_pixels_y = np.float32(height / extent_y)
    nodes_per_unit = np.float32(shape.grid.per_unit)
    columns = np.arange(left, right, dtype=np.float32)
    # The field is read bilinearly, one way and then the other, which is
    # the same sum as :func:`_sample` takes, for far fewer look ups.
    column = np.clip((columns + np.float32(0.5)) / to_pixels_x * nodes_per_unit, 0.0, across_nodes - 1)
    column_left = np.minimum(column.astype(np.intp), across_nodes - 2)
    across_share = column - column_left.astype(np.float32)
    first_node = int(column_left[0])
    last_node = int(column_left[-1]) + 2
    column_left -= first_node
    source = np.ascontiguousarray(values)
    for strip in range(top, bottom, STRIP_ROWS):
        end = min(bottom, strip + STRIP_ROWS)
        rows = np.arange(strip, end, dtype=np.float32)
        row = np.clip((rows + np.float32(0.5)) / to_pixels_y * nodes_per_unit, 0.0, down_nodes - 1)
        row_top = np.minimum(row.astype(np.intp), down_nodes - 2)
        down_share = (row - row_top.astype(np.float32))[:, None]
        shifts = []
        for plane in shape.values:
            upper = plane[row_top, first_node:last_node]
            lower = plane[row_top + 1, first_node:last_node]
            along = upper + (lower - upper) * down_share
            near = np.take(along, column_left, axis=1)
            far = np.take(along, column_left + 1, axis=1)
            shifts.append(near + (far - near) * across_share)
        shift_x = shifts[0] * to_pixels_x
        shift_y = shifts[1] * to_pixels_y
        map_x = columns[None, :] + shift_x
        map_y = rows[:, None] + shift_y
        moved = cv2.remap(source, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)
        still = np.maximum(np.abs(shift_x), np.abs(shift_y)) < STILL_PX
        if moved.ndim == 3:
            still = still[..., None]
        out[strip:end, left:right] = np.where(still, values[strip:end, left:right], moved)
    return out


def draft(values: NDArray[Any], stroke: Stroke) -> NDArray[Any]:
    """Return *values* with *stroke* put on it at once, roughly (PLAN.md 3.16).

    *values* is a picture already developed - what is on screen - and the
    stroke is in its own fractions and long edge. What the hand sees while it
    paints; letting go develops the photo properly. The grid is only as fine
    as the brush needs (:data:`DRAFT_NODES_PER_RADIUS`), and the picture is
    read bilinearly.
    """
    height, width = values.shape[:2]
    nodes = math.ceil(DRAFT_NODES_PER_RADIUS / max(stroke.radius, 1e-6)) + 1
    grid = grid_for((width, height), min(GRID_LONG, max(DRAFT_GRID_LEAST, nodes)))
    shape = build_field((stroke,), grid)
    if shape is None:
        return values
    return warp_values(values, shape, cv2.INTER_LINEAR)


def _pixel(node: int, nodes: int, pixels: int, rounding: Any) -> int:
    """Return the pixel at *node* of *nodes* along an edge of *pixels*, kept on the photo."""
    position = node / (nodes - 1) * pixels - 0.5
    return int(min(max(rounding(position), 0), pixels - 1))


# --------------------------------------------------------------------------
# Points through the field
# --------------------------------------------------------------------------


def source_points(
    points: Sequence[Point], strokes: tuple[Stroke, ...], size: tuple[int, int]
) -> tuple[Point, ...]:
    """Return where on the photo the picture at *points* is fetched from.

    The points are fractions of a photo of *size*, as the picture shows it
    after the strokes, and so is the answer, on the photo before them: what
    a click on the shaped picture means for a spot or the skin brush.
    """
    shape = field(strokes, size)
    if shape is None or not points:
        return tuple((float(x), float(y)) for x, y in points)
    across, down = shape.grid.extent
    at = np.array([(x * across, y * down) for x, y in points], dtype=np.float64)
    fetched = _fetched(shape, at)
    return tuple((float(x / across), float(y / down)) for x, y in fetched)


def shown_points(
    points: Sequence[Point], strokes: tuple[Stroke, ...], size: tuple[int, int]
) -> tuple[Point, ...]:
    """Return where the strokes show what is at *points* on the photo.

    The inverse of :func:`source_points`. Each point starts from the node
    of the grid that fetches nearest to it - a long push can carry a point
    far from where it was, so starting from the point itself may not get
    there - and is then brought in by Newton's method, a step taken only
    when it lands nearer. Where the brush squeezed a part away entirely the
    nearest point that fetches from it is given.
    """
    shape = field(strokes, size)
    if shape is None or not points:
        return tuple((float(x), float(y)) for x, y in points)
    across, down = shape.grid.extent
    wanted = np.array([(x * across, y * down) for x, y in points], dtype=np.float64)
    found = _nearest_node(shape, wanted)
    step = shape.grid.step[0]
    miss = _fetched(shape, found) - wanted
    for _ in range(INVERSE_STEPS):
        error = np.hypot(miss[:, 0], miss[:, 1])
        if error.max() < INVERSE_TOLERANCE:
            break
        # The slope of the map, read across one node either way.
        right = _fetched(shape, found + (step, 0.0))
        left = _fetched(shape, found - (step, 0.0))
        lower = _fetched(shape, found + (0.0, step))
        upper = _fetched(shape, found - (0.0, step))
        a = (right[:, 0] - left[:, 0]) / (2 * step)
        b = (lower[:, 0] - upper[:, 0]) / (2 * step)
        c = (right[:, 1] - left[:, 1]) / (2 * step)
        d = (lower[:, 1] - upper[:, 1]) / (2 * step)
        determinant = a * d - b * c
        safe = np.abs(determinant) > 1e-6
        determinant = np.where(safe, determinant, 1.0)
        move = np.stack(
            (
                np.where(safe, (d * miss[:, 0] - b * miss[:, 1]) / determinant, miss[:, 0]),
                np.where(safe, (a * miss[:, 1] - c * miss[:, 0]) / determinant, miss[:, 1]),
            ),
            axis=1,
        )
        # Halve a step that would land farther off, up to a few times.
        better = np.zeros(len(found), dtype=bool)
        trial = found
        trial_miss = miss
        for _halving in range(6):
            candidate = _on_grid(shape, found - move)
            candidate_miss = _fetched(shape, candidate) - wanted
            closer = np.hypot(candidate_miss[:, 0], candidate_miss[:, 1]) < error
            take = closer & ~better
            trial = np.where(take[:, None], candidate, trial)
            trial_miss = np.where(take[:, None], candidate_miss, trial_miss)
            better |= closer
            if better.all():
                break
            move = move * 0.5
        if not better.any():
            break
        found, miss = trial, trial_miss
    return tuple((float(x / across), float(y / down)) for x, y in found)


def _fetched(shape: Field, at: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return where the picture at points *at* (n by 2, units of the long edge) fetches from."""
    shift = _sample(
        shape.values, shape.grid, at[:, 0].astype(np.float32), at[:, 1].astype(np.float32)
    )
    return at + shift.T.astype(np.float64)


def _on_grid(shape: Field, at: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return points *at* held on the photo."""
    across, down = shape.grid.extent
    return np.stack((np.clip(at[:, 0], 0.0, across), np.clip(at[:, 1], 0.0, down)), axis=1)


def _nearest_node(shape: Field, wanted: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return, for each point, the node (or the point itself) that fetches nearest to it.

    Only the part of the grid the strokes moved is looked through, at most
    :data:`INVERSE_SEARCH_NODES` of its nodes, a few hundred points at a
    time.
    """
    first_row, last_row, first_column, last_column = shape.box
    count = (last_row - first_row + 2) * (last_column - first_column + 2)
    stride = max(1, math.ceil(math.sqrt(count / INVERSE_SEARCH_NODES)))
    step = shape.grid.step[0]
    down, across = shape.values.shape[1:]
    rows = np.arange(max(first_row - 1, 0), min(last_row + 1, down), stride)
    columns = np.arange(max(first_column - 1, 0), min(last_column + 1, across), stride)
    node_x, node_y = np.meshgrid(columns * step, rows * step)
    nodes = np.stack((node_x.ravel(), node_y.ravel()), axis=1)
    lands = nodes + np.stack(
        (
            shape.values[0][np.ix_(rows, columns)].ravel(),
            shape.values[1][np.ix_(rows, columns)].ravel(),
        ),
        axis=1,
    ).astype(np.float64)
    start = wanted.copy()
    own = _fetched(shape, wanted) - wanted
    own_error = own[:, 0] ** 2 + own[:, 1] ** 2
    for first in range(0, len(wanted), INVERSE_SEARCH_POINTS):
        part = wanted[first : first + INVERSE_SEARCH_POINTS]
        distance = (
            (part[:, None, 0] - lands[None, :, 0]) ** 2 + (part[:, None, 1] - lands[None, :, 1]) ** 2
        )
        nearest = np.argmin(distance, axis=1)
        nearer = distance[np.arange(len(part)), nearest] < own_error[first : first + len(part)]
        start[first : first + len(part)][nearer] = nodes[nearest[nearer]]
    return start


def moved_skin(
    skin: "SkinRange | None", strokes: tuple[Stroke, ...], size: tuple[int, int]
) -> "SkinRange | None":
    """Return the portrait brush's strokes moved to where the liquify strokes show them.

    The skin range is painted on the photo before the liquify, like the
    spots, so it stays on the skin when the skin is pushed around; its
    radii are left as they are.
    """
    if skin is None or not skin.strokes or not strokes or field(strokes, size) is None:
        return skin
    return replace(
        skin,
        strokes=tuple(
            replace(stroke, points=shown_points(stroke.points, strokes, size))
            for stroke in skin.strokes
        ),
    )


def _fraction(value: float) -> float:
    """Return *value* held between 0 and 1."""
    return max(0.0, min(1.0, float(value)))


def _as_float(value: Any) -> float | None:
    """Return *value* as a finite number, or ``None`` if it is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
