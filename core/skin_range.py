"""The skin the portrait sliders work on, as found and as painted (PLAN.md 2.10).

The skin beside a face is found by itself (:func:`core.faces.body_mask`),
and the brush of the portrait mode adds to it or takes from it. A
:class:`SkinRange` holds whether the skin is found at all and the strokes of
the brush, in order: each :class:`SkinStroke` is the path the pointer took,
the brush's radius, whether it adds or takes away, and whether it keeps to
the colour of skin.

Strokes are held like the spots of :mod:`core.healing`: in fractions of the
photo after its lens correction and before it is turned or cropped, so they
stay on the arm they were painted on whatever is done to the frame; the
radius is a fraction of the long edge. :mod:`core.geometry` maps them onto
the frame it makes (:meth:`SkinRange.mapped`).

What the pipeline is handed (:func:`field`) is three planes over the whole
frame on a small copy - the skin found, the share of it the strokes that
take away leave, and the skin the strokes add - so that the skin found can
be combined with the face's own sharper mask where it is developed, and the
order of the strokes still counts: a later stroke that takes away takes
from what an earlier one added, and the other way round.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core import faces as faces_module
from core.faces import Face, SkinTone

Point = tuple[float, float]

#: The radius of the brush, as a fraction of the long edge: the same scale
#: as the spot brush (PLAN.md 2.2, A-32), whose size slider runs over it in
#: thousandths. The skin of an arm wants a broad brush, so it starts wider.
RADIUS_MIN: Final[float] = 0.001
RADIUS_MAX: Final[float] = 0.1
RADIUS_DEFAULT: Final[float] = 0.03
#: At most this many strokes and points of each are read from a file.
STROKES_MAX: Final[int] = 1000
POINTS_MAX: Final[int] = 2000
#: Points of a path nearer than this share of the radius are not kept.
POINT_SPACING: Final[float] = 0.25
#: The edge of a stroke fades over this share of its radius.
FEATHER: Final[float] = 0.3
#: A stroke that keeps to skin keeps to the colour of the faces' skin, or,
#: with no face to take it from, to the colour it was painted over, which
#: may vary this much (stops of red and blue against green).
PAINTED_SPREAD: Final[float] = faces_module.COLOUR_MOST_SPREAD
#: Dips in that likeness narrower than this share of the radius - a
#: freckle, a hair - are filled, so a stroke does not come out speckled.
SNAP_HOLE_FILL: Final[float] = 0.3

#: The planes of :func:`field`.
FOUND: Final[int] = 0
KEPT: Final[int] = 1
ADDED: Final[int] = 2


@dataclass(frozen=True)
class SkinStroke:
    """One stroke of the portrait brush (PLAN.md 2.10)."""

    points: tuple[Point, ...]
    radius: float = RADIUS_DEFAULT
    #: True adds to the skin, false takes away from it.
    adding: bool = True
    #: Whether an adding stroke keeps to the colour of skin, so that one
    #: painted roughly over an arm leaves the sleeve beside it alone.
    snap: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the stroke as it is stored (PLAN.md 7.1)."""
        return {
            "points": [[x, y] for x, y in self.points],
            "radius": self.radius,
            "mode": "add" if self.adding else "remove",
            "snap": self.snap,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SkinStroke | None":
        """Build a stroke from stored data, or return ``None`` if it is not one."""
        if not isinstance(data, dict):
            return None
        radius = _as_float(data.get("radius"))
        raw = data.get("points")
        mode = data.get("mode")
        if radius is None or not isinstance(raw, list) or not raw or mode not in ("add", "remove"):
            return None
        points: list[Point] = []
        for item in raw[:POINTS_MAX]:
            if not isinstance(item, list | tuple) or len(item) != 2:
                return None
            x, y = _as_float(item[0]), _as_float(item[1])
            if x is None or y is None:
                return None
            points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
        return cls(
            points=tuple(points),
            radius=clamp_radius(radius),
            adding=mode == "add",
            snap=data.get("snap") is not False,
        )


@dataclass(frozen=True)
class SkinRange:
    """Whether the skin is found by itself, and the strokes that change it."""

    auto: bool = True
    strokes: tuple[SkinStroke, ...] = ()

    @property
    def is_default(self) -> bool:
        """Return whether the range is the one found, untouched by the brush."""
        return self.auto and not self.strokes

    @property
    def has_additions(self) -> bool:
        """Return whether any stroke adds skin."""
        return any(stroke.adding for stroke in self.strokes)

    def with_stroke(self, stroke: SkinStroke) -> "SkinRange":
        """Return this range with *stroke* painted on top."""
        return replace(self, strokes=(*self.strokes, stroke)[-STROKES_MAX:])

    def to_dict(self) -> dict[str, Any]:
        """Return the range as it is stored (PLAN.md 7.1)."""
        return {"auto": self.auto, "strokes": [stroke.to_dict() for stroke in self.strokes]}

    @classmethod
    def from_dict(cls, data: Any) -> "SkinRange":
        """Build a range from stored data; anything odd falls back to the default."""
        if not isinstance(data, dict):
            return cls()
        raw = data.get("strokes")
        strokes: tuple[SkinStroke, ...] = ()
        if isinstance(raw, list):
            read = (SkinStroke.from_dict(item) for item in raw[:STROKES_MAX])
            strokes = tuple(stroke for stroke in read if stroke is not None)
        return cls(auto=data.get("auto") is not False, strokes=strokes)

    def mapped(
        self, matrix: NDArray[np.float64], size: tuple[int, int], frame: tuple[int, int]
    ) -> "SkinRange":
        """Return the range as it lies on a frame made of a photo of *size*.

        *matrix* takes pixels of the photo (OpenCV's convention) to pixels
        of the frame, which is *frame* pixels; turns, mirrors, rotations and
        cuts keep a pixel a pixel, so the radius only changes with the long
        edge it is measured against.
        """
        if not self.strokes:
            return self
        width, height = size
        scale = max(width, height) / max(max(frame), 1)
        moved = []
        for stroke in self.strokes:
            points = np.array(
                [(x * width - 0.5, y * height - 0.5, 1.0) for x, y in stroke.points]
            )
            shown = points @ matrix[:2].T
            moved.append(
                replace(
                    stroke,
                    points=tuple(
                        (float((x + 0.5) / frame[0]), float((y + 0.5) / frame[1]))
                        for x, y in shown
                    ),
                    radius=stroke.radius * scale,
                )
            )
        return replace(self, strokes=tuple(moved))


DEFAULT_RANGE: Final[SkinRange] = SkinRange()


def clamp_radius(radius: float) -> float:
    """Return *radius* held between the smallest and the largest brush."""
    return max(RADIUS_MIN, min(RADIUS_MAX, float(radius)))


def thinned(points: Sequence[Point], radius: float, size: tuple[int, int]) -> tuple[Point, ...]:
    """Return the path *points* with points nearer than POINT_SPACING radii left out.

    *size* is the width and height the fractions are of. The last point is
    always kept, so the stroke ends where the pointer did.
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


def reaches_nothing(faces: tuple[Face, ...] | None, skin: SkinRange | None) -> bool:
    """Return whether the portrait sliders have no skin at all to work on.

    That is when no stroke adds any and the skin is not found by itself, or
    there is no face to find it from. A photo like that is not changed by
    the sliders, so it still counts as untouched (PLAN.md 2.2).
    """
    skin = DEFAULT_RANGE if skin is None else skin
    if skin.has_additions:
        return False
    return not (skin.auto and faces)


def field(
    image: NDArray[np.float32], faces: tuple[Face, ...] | None, skin: SkinRange | None
) -> NDArray[np.float32]:
    """Return the skin over the whole frame, as three planes on a small copy.

    *image* is the whole frame in its own linear colour, and *skin* the
    range as it lies on it. The planes, at :func:`core.faces.body_size`, are
    the skin found by itself (FOUND, nothing when the range does not find
    it), the share of that skin the strokes leave (KEPT) and the skin they
    add (ADDED). The skin to work on is ``max(found × kept, added)``, which is
    the strokes laid one over another in order.
    """
    skin = DEFAULT_RANGE if skin is None else skin
    height, width = image.shape[:2]
    size = faces_module.body_size((width, height))
    small = (
        cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size != (width, height) else image
    )
    planes = np.zeros((size[1], size[0], 3), dtype=np.float32)
    planes[..., KEPT] = 1.0
    if skin.auto and faces:
        found = faces_module.body_mask(small, faces)
        if found is not None:
            planes[..., FOUND] = found
    if not skin.strokes:
        return planes

    features = faces_module.colour_features(small)
    tones = [face.tone for face in faces or () if faces_module.is_coloured(face.tone)]
    # With faces to take the colour from, every stroke keeps to the same.
    shared = faces_module.body_likeness(*features, tones) if tones else None
    kept = np.ones(small.shape[:2], dtype=np.float32)
    added = np.zeros(small.shape[:2], dtype=np.float32)
    # From the last stroke back: what a stroke adds is then cut by every
    # stroke that takes away after it, and by none before it.
    for stroke in reversed(skin.strokes):
        painted = stroke_mask(stroke, size)
        if not stroke.adding:
            kept *= np.float32(1.0) - painted
            continue
        if stroke.snap:
            painted *= _likeness(features, shared, painted, stroke.radius * max(size))
        np.maximum(added, painted * kept, out=added)
    planes[..., KEPT] = kept
    planes[..., ADDED] = added
    return planes


def combined(planes: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the skin to work on from :func:`field`'s planes, ``max(found × kept, added)``."""
    return np.maximum(planes[..., FOUND] * planes[..., KEPT], planes[..., ADDED]).astype(
        np.float32
    )


def stroke_mask(stroke: SkinStroke, size: tuple[int, int]) -> NDArray[np.float32]:
    """Return how much of each pixel of a frame of *size* the stroke covers."""
    width, height = size
    radius = stroke.radius * max(width, height)
    drawn = np.zeros((height, width), dtype=np.uint8)
    # Drawn at four times the size in OpenCV's fixed point, so that a brush
    # narrower than a pixel of the small copy still leaves its trace.
    shift = 2
    unit = 1 << shift
    points = [
        (round((x * width - 0.5) * unit), round((y * height - 0.5) * unit))
        for x, y in stroke.points
    ]
    reach = max(1, round(radius * unit))
    for point in points:
        cv2.circle(drawn, point, reach, 255, -1, cv2.LINE_AA, shift)
    if len(points) > 1:
        thickness = max(1, round(2.0 * radius))
        for start, end in zip(points[:-1], points[1:], strict=True):
            cv2.line(drawn, start, end, 255, thickness, cv2.LINE_AA, shift)
    mask = drawn.astype(np.float32) * np.float32(1.0 / 255.0)
    sigma = FEATHER * radius / 2.0
    if sigma >= 0.5:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return np.clip(mask, 0.0, 1.0)


def _likeness(
    features: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    shared: NDArray[np.float32] | None,
    painted: NDArray[np.float32],
    radius: float,
) -> NDArray[np.float32]:
    """Return how much each pixel is skin, for a stroke that keeps to it.

    *shared* is the likeness to the colour of the faces, when there are
    faces with colour; otherwise the colour the stroke was painted over is
    read as a face's would be. Where there is no colour to go by - a black
    and white photo - everything is.
    """
    red, blue, brightness = features
    like = shared
    if like is None:
        under = painted > 0.5
        if not under.any():
            return np.ones(red.shape, dtype=np.float32)
        tone = SkinTone(
            red=float(np.median(red[under])),
            blue=float(np.median(blue[under])),
            spread=PAINTED_SPREAD,
            brightness=float(np.median(brightness[under])),
        )
        if not faces_module.is_coloured(tone):
            return np.ones(red.shape, dtype=np.float32)
        like = faces_module.body_likeness(red, blue, brightness, [tone])
    return faces_module.fill_holes(like, SNAP_HOLE_FILL * radius)


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
    "ADDED",
    "DEFAULT_RANGE",
    "FOUND",
    "KEPT",
    "RADIUS_DEFAULT",
    "RADIUS_MAX",
    "RADIUS_MIN",
    "SkinRange",
    "SkinStroke",
    "clamp_radius",
    "combined",
    "field",
    "reaches_nothing",
    "stroke_mask",
    "thinned",
]
