"""Where the photo stays sharp and how blurred the rest is (PLAN.md 2.13).

The depth of field is chosen by hand: a shape that stays sharp - an
ellipse, or a band across the frame - from whose edge the blur grows over a
transition, and the strokes of a brush that keep a place sharp or blur it
on top of that. :class:`DepthOfField` holds all of it with the amount of
blur, the shape the out of focus light takes, and how strongly bright
points of light are made to stand out in it.

Points are held like the portrait brush's strokes (:mod:`core.skin_range`):
in fractions of the photo after its lens correction and before it is turned
or cropped, so the sharp shape stays on what it was put on whatever is done
to the frame. :mod:`core.geometry` maps them onto the frame it makes
(:meth:`DepthOfField.mapped`).

What the pipeline is handed (:func:`blurriness`) is how blurred each pixel
of the frame is, from 0 (sharp) to 1 (the most the amount allows), on a
small copy; how that is turned into blurred light is the pipeline's
business (PLAN.md 6 step 6.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from core import skin_range
from core.skin_range import SkinStroke

Point = tuple[float, float]

#: The two shapes that stay sharp: an ellipse around a subject, or a band
#: across the frame - the ground a few steps away, a row of houses.
SHAPE_ELLIPSE: Final[str] = "ellipse"
SHAPE_BAND: Final[str] = "band"
SHAPES: Final[tuple[str, ...]] = (SHAPE_ELLIPSE, SHAPE_BAND)

#: The two shapes out of focus light takes: the round opening of a lens
#: wide open, or the six blades of one stopped down.
BOKEH_CIRCLE: Final[str] = "circle"
BOKEH_HEXAGON: Final[str] = "hexagon"
BOKEHS: Final[tuple[str, ...]] = (BOKEH_CIRCLE, BOKEH_HEXAGON)

#: The three sliders, which are amounts with nothing below zero.
AMOUNT_MIN: Final[float] = 0.0
AMOUNT_MAX: Final[float] = 100.0
TRANSITION_MIN: Final[float] = 0.0
TRANSITION_MAX: Final[float] = 100.0
TRANSITION_DEFAULT: Final[float] = 40.0
HIGHLIGHTS_MIN: Final[float] = 0.0
HIGHLIGHTS_MAX: Final[float] = 100.0
HIGHLIGHTS_DEFAULT: Final[float] = 50.0

#: How far beyond the sharp shape the blur reaches all it may, as a share of
#: the long edge, from the transition slider at zero to the slider at 100.
#: At zero the step is short rather than a cut, so the edge of the shape
#: does not show as a line.
TRANSITION_REACH: Final[tuple[float, float]] = (0.01, 0.5)

#: Where a new photo's sharp shape starts: an upright oval in the middle, as
#: wide as three quarters of its height, its half height a quarter of the
#: frame's - a person standing, a cup on a table - or a band across the
#: middle a fifth of the frame tall.
DEFAULT_CENTRE: Final[Point] = (0.5, 0.5)
DEFAULT_ELLIPSE_HANDLE: Final[Point] = (0.5, 0.25)
DEFAULT_RATIO: Final[float] = 0.75
DEFAULT_BAND_HANDLE: Final[Point] = (0.5, 0.4)
#: The narrowest an ellipse may be, as its short axis over its long one, and
#: the smallest the shape may be, as a share of the long edge.
RATIO_MIN: Final[float] = 0.1
RATIO_MAX: Final[float] = 1.0
SIZE_MIN: Final[float] = 0.005

#: At most this many strokes are read from a file, as for the skin.
STROKES_MAX: Final[int] = skin_range.STROKES_MAX

#: How small the blurriness is worked out: the long edge of the copy. It
#: changes slowly everywhere but at the brush's edges, which are soft too.
FIELD_LONG_EDGE: Final[int] = 1024


@dataclass(frozen=True)
class DepthOfField:
    """The depth of field of one photo (PLAN.md 2.13)."""

    #: How much the parts out of focus are blurred, 0 to 100.
    amount: float = 0.0
    #: Which shape stays sharp.
    shape: str = SHAPE_ELLIPSE
    #: The middle of the shape.
    centre: Point = DEFAULT_CENTRE
    #: For an ellipse, the end of its long axis; for a band, a point on its
    #: edge straight out from the middle line, which says how wide the band
    #: is and which way it runs.
    handle: Point = DEFAULT_ELLIPSE_HANDLE
    #: The ellipse's short axis over its long one.
    ratio: float = DEFAULT_RATIO
    #: How gradually the blur grows beyond the shape, 0 to 100.
    transition: float = TRANSITION_DEFAULT
    #: The shape out of focus light takes.
    bokeh: str = BOKEH_CIRCLE
    #: How strongly bright points of light stand out once out of focus.
    highlights: float = HIGHLIGHTS_DEFAULT
    #: The brush's strokes, in order: one that adds keeps what it covers
    #: sharp, one that takes away blurs it.
    strokes: tuple[SkinStroke, ...] = ()

    @property
    def is_default(self) -> bool:
        """Return whether nothing has been set: no blur, the shape where it starts, no strokes."""
        return self == DEFAULT_DEPTH

    @property
    def blurs(self) -> bool:
        """Return whether anything is blurred at all."""
        return self.amount > 0.0

    def with_stroke(self, stroke: SkinStroke) -> "DepthOfField":
        """Return this depth of field with *stroke* painted on top."""
        return replace(self, strokes=(*self.strokes, stroke)[-STROKES_MAX:])

    def with_shape(self, shape: str) -> "DepthOfField":
        """Return this depth of field with the other kind of sharp shape, where it starts."""
        if shape not in SHAPES or shape == self.shape:
            return self
        handle = DEFAULT_ELLIPSE_HANDLE if shape == SHAPE_ELLIPSE else DEFAULT_BAND_HANDLE
        return replace(self, shape=shape, centre=DEFAULT_CENTRE, handle=handle, ratio=DEFAULT_RATIO)

    def with_values(self, **values: float) -> "DepthOfField":
        """Return a copy with sliders set, each held within its range."""
        ranges = {
            "amount": (AMOUNT_MIN, AMOUNT_MAX),
            "transition": (TRANSITION_MIN, TRANSITION_MAX),
            "highlights": (HIGHLIGHTS_MIN, HIGHLIGHTS_MAX),
            "ratio": (RATIO_MIN, RATIO_MAX),
        }
        changed: dict[str, float] = {}
        for name, value in values.items():
            low, high = ranges[name]
            changed[name] = max(low, min(high, float(value)))
        return replace(self, **changed)

    def to_dict(self) -> dict[str, Any]:
        """Return the depth of field as it is stored (PLAN.md 7.1)."""
        return {
            "amount": self.amount,
            "shape": self.shape,
            "centre": list(self.centre),
            "handle": list(self.handle),
            "ratio": self.ratio,
            "transition": self.transition,
            "bokeh": self.bokeh,
            "highlights": self.highlights,
            "strokes": [stroke.to_dict() for stroke in self.strokes],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DepthOfField":
        """Build a depth of field from stored data; anything odd falls back."""
        if not isinstance(data, dict):
            return DEFAULT_DEPTH
        shape = data.get("shape")
        shape = shape if shape in SHAPES else SHAPE_ELLIPSE
        bokeh = data.get("bokeh")
        default_handle = DEFAULT_ELLIPSE_HANDLE if shape == SHAPE_ELLIPSE else DEFAULT_BAND_HANDLE
        raw = data.get("strokes")
        strokes: tuple[SkinStroke, ...] = ()
        if isinstance(raw, list):
            read = (SkinStroke.from_dict(item) for item in raw[:STROKES_MAX])
            strokes = tuple(replace(stroke, snap=False) for stroke in read if stroke is not None)
        depth = cls(
            shape=shape,
            centre=_as_point(data.get("centre"), DEFAULT_CENTRE),
            handle=_as_point(data.get("handle"), default_handle),
            bokeh=bokeh if bokeh in BOKEHS else BOKEH_CIRCLE,
            strokes=strokes,
        )
        values = {
            name: number
            for name in ("amount", "transition", "highlights", "ratio")
            if (number := _as_float(data.get(name))) is not None
        }
        return depth.with_values(**values)

    def mapped(
        self, matrix: NDArray[np.float64], size: tuple[int, int], frame: tuple[int, int]
    ) -> "DepthOfField":
        """Return the depth of field as it lies on a frame made of a photo of *size*.

        *matrix* takes pixels of the photo (OpenCV's convention) to pixels of
        the frame, which is *frame* pixels, as for
        :meth:`core.skin_range.SkinRange.mapped`. Turns, mirrors, rotations
        and cuts keep a pixel a pixel, so the shape keeps its size and the
        ellipse its ratio; the two points carry where it lies and which way.
        """
        width, height = size

        def moved(point: Point) -> Point:
            x, y, _ = matrix @ np.array((point[0] * width - 0.5, point[1] * height - 0.5, 1.0))
            return float((x + 0.5) / frame[0]), float((y + 0.5) / frame[1])

        strokes = skin_range.SkinRange(strokes=self.strokes).mapped(matrix, size, frame).strokes
        return replace(
            self, centre=moved(self.centre), handle=moved(self.handle), strokes=strokes
        )


DEFAULT_DEPTH: Final[DepthOfField] = DepthOfField()


def transition_reach(transition: float) -> float:
    """Return how far beyond the shape the blur is all there, as a share of the long edge."""
    low, high = TRANSITION_REACH
    share = max(TRANSITION_MIN, min(TRANSITION_MAX, transition)) / TRANSITION_MAX
    return low + (high - low) * share


def field_size(frame: tuple[int, int]) -> tuple[int, int]:
    """Return the size :func:`blurriness` is worked out at for a frame of *frame*."""
    width, height = frame
    long_edge = max(width, height, 1)
    if long_edge <= FIELD_LONG_EDGE:
        return max(1, width), max(1, height)
    scale = FIELD_LONG_EDGE / long_edge
    return max(1, round(width * scale)), max(1, round(height * scale))


def outside(depth: DepthOfField, size: tuple[int, int]) -> NDArray[np.float32]:
    """Return how far each pixel of a frame of *size* lies outside the sharp shape.

    In pixels of that frame, zero or less inside it. Pixel centres are
    measured, OpenCV's convention, as the strokes are drawn.
    """
    width, height = size
    columns = (np.arange(width, dtype=np.float32) + np.float32(0.5)) - np.float32(
        depth.centre[0] * width
    )
    rows = (np.arange(height, dtype=np.float32) + np.float32(0.5)) - np.float32(
        depth.centre[1] * height
    )
    axis_x = (depth.handle[0] - depth.centre[0]) * width
    axis_y = (depth.handle[1] - depth.centre[1]) * height
    reach = max(math.hypot(axis_x, axis_y), SIZE_MIN * max(width, height))
    angle = math.atan2(axis_y, axis_x)
    cos, sin = np.float32(math.cos(angle)), np.float32(math.sin(angle))
    # Along the handle's direction, and across it.
    along = columns[None, :] * cos + rows[:, None] * sin
    across = -columns[None, :] * sin + rows[:, None] * cos
    if depth.shape == SHAPE_BAND:
        return (np.abs(along) - np.float32(reach)).astype(np.float32)
    long_axis = np.float32(reach)
    short_axis = np.float32(max(reach * depth.ratio, SIZE_MIN * max(width, height)))
    u = along / long_axis
    v = across / short_axis
    rho = np.sqrt(u * u + v * v)
    # The distance to the ellipse, to first order: how far rho is past one
    # over how fast it grows. Exact on the edge, close enough beyond it.
    safe = np.maximum(rho, np.float32(1e-6))
    gradient = np.sqrt((u / long_axis) ** 2 + (v / short_axis) ** 2) / safe
    return ((rho - np.float32(1.0)) / np.maximum(gradient, np.float32(1e-9))).astype(np.float32)


def blurriness(depth: DepthOfField, size: tuple[int, int]) -> NDArray[np.float32]:
    """Return how blurred each pixel of a frame of *size* is, from 0 to 1.

    Zero inside the sharp shape, growing smoothly to one over the transition
    beyond it; then each stroke in order keeps what it covers sharp (adds)
    or blurs it (takes away), so a later stroke wins over an earlier one.
    """
    width, height = size
    reach = transition_reach(depth.transition) * max(width, height)
    step = np.clip(outside(depth, size) / np.float32(max(reach, 1e-6)), 0.0, 1.0)
    blurred = (step * step * (np.float32(3.0) - np.float32(2.0) * step)).astype(np.float32)
    for stroke in depth.strokes:
        painted = skin_range.stroke_mask(stroke, size)
        if stroke.adding:
            blurred *= np.float32(1.0) - painted
        else:
            np.maximum(blurred, painted, out=blurred)
    return blurred


def _as_point(value: Any, default: Point) -> Point:
    """Return *value* as a point within the frame and a little beyond, or *default*."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return default
    x, y = _as_float(value[0]), _as_float(value[1])
    if x is None or y is None:
        return default
    return max(-1.0, min(2.0, x)), max(-1.0, min(2.0, y))


def _as_float(value: Any) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "BOKEHS",
    "BOKEH_CIRCLE",
    "BOKEH_HEXAGON",
    "DEFAULT_DEPTH",
    "DepthOfField",
    "SHAPES",
    "SHAPE_BAND",
    "SHAPE_ELLIPSE",
    "blurriness",
    "field_size",
    "outside",
    "transition_reach",
]
