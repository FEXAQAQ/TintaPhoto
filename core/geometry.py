"""Cropping, rotation and straightening (PLAN.md 2.3 and section 6 step 2).

Four things happen here, always in this order: the photo is turned in
quarter turns, mirrored if the user flipped it, then rotated by the angle the
user asked for, then cropped.
Everything else in this module exists to answer two questions the crop box
keeps asking:

* **Where may the box be?** Rotating a rectangle leaves four empty triangles
  in the corners of the result. The crop box must stay off them, or the photo
  would come out with black edges (C-05). A box is allowed exactly when its
  four corners, mapped back through the rotation, land inside the photo.
* **How large may it be?** A box of a given shape, centred and turned by the
  angle, fits inside a photo of *w* by *h* when its own bounding box does:

      W·|cos| + H·|sin| <= w      W·|sin| + H·|cos| <= h

  which, with H = W / ratio, gives the width directly. That is the largest
  the box can be, and it is where a fresh crop starts.

The crop rectangle is stored in fractions of the rotated frame, which is what
PLAN.md section 7.1 asks for and what keeps it independent of the size the
photo happens to be decoded at: the preview and the exported file use the
same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core import healing
from core import lens as lens_module
from core.decoder import DecodedPhoto
from core.faces import Face
from core.healing import Spot
from core.lens import DEFAULT_SETTINGS, LensSettings

#: The fixed ratios of PLAN.md C-02, with the two open choices. 9:16 is the
#: upright full screen shape, which X reaches from 16:9 as well; it is named
#: here because that is how the social platforms name it, and a ratio nobody
#: can see on a button is a ratio nobody finds.
ASPECT_ORIGINAL: Final[str] = "original"
ASPECT_FREE: Final[str] = "free"
ASPECT_SQUARE: Final[str] = "1:1"
ASPECT_3_2: Final[str] = "3:2"
ASPECT_4_5: Final[str] = "4:5"
ASPECT_16_9: Final[str] = "16:9"
ASPECT_9_16: Final[str] = "9:16"

#: In the order the crop tab shows them, three to a row (PLAN.md 3.2).
ASPECTS: Final[tuple[str, ...]] = (
    ASPECT_ORIGINAL,
    ASPECT_FREE,
    ASPECT_SQUARE,
    ASPECT_3_2,
    ASPECT_4_5,
    ASPECT_16_9,
    ASPECT_9_16,
)

#: The ratios exactly as they are written: 4:5 is four wide and five high,
#: so it is a portrait shape until it is turned on its side (X).
_FIXED_RATIOS: Final[dict[str, float]] = {
    ASPECT_SQUARE: 1.0,
    ASPECT_3_2: 3 / 2,
    ASPECT_4_5: 4 / 5,
    ASPECT_16_9: 16 / 9,
    ASPECT_9_16: 9 / 16,
}

ORIENTATION_LANDSCAPE: Final[str] = "landscape"
ORIENTATION_PORTRAIT: Final[str] = "portrait"

#: Rotation, in degrees (C-03).
ANGLE_MIN: Final[float] = -45.0
ANGLE_MAX: Final[float] = 45.0
ANGLE_STEP: Final[float] = 0.1

#: The crop box may never become smaller than this fraction of the frame,
#: so that it cannot be lost by a careless drag.
MINIMUM_SIDE: Final[float] = 0.05

#: Iterations used when shrinking a box back inside the photo. Each one
#: halves the remaining error, so twenty are far more than enough for a
#: fraction that is later turned into whole pixels.
_SHRINK_STEPS: Final[int] = 20

#: How near a crop has to be to the largest box of its shape to count as
#: filling the picture, measured as a fraction of the frame's area.
_FULL_ENOUGH: Final[float] = 1e-3


_EPSILON: Final[float] = 1e-9


class Handle(Enum):
    """What the pointer has taken hold of (PLAN.md 3.2)."""

    NONE = "none"
    INSIDE = "inside"
    OUTSIDE = "outside"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def is_edge(self) -> bool:
        """Return whether this handle moves one edge."""
        return self in _EDGES


_EDGES: Final[frozenset[Handle]] = frozenset(
    {Handle.TOP, Handle.BOTTOM, Handle.LEFT, Handle.RIGHT}
)


@dataclass(frozen=True)
class Rect:
    """A rectangle in fractions of the frame it sits in (PLAN.md 7.1)."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    @property
    def right(self) -> float:
        """Return the right edge."""
        return self.x + self.w

    @property
    def bottom(self) -> float:
        """Return the bottom edge."""
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        """Return the middle of the rectangle."""
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def is_whole_frame(self) -> bool:
        """Return whether this is the whole frame, uncropped."""
        return (
            abs(self.x) < _EPSILON
            and abs(self.y) < _EPSILON
            and abs(self.w - 1.0) < _EPSILON
            and abs(self.h - 1.0) < _EPSILON
        )

    def scaled(self, factor: float) -> "Rect":
        """Return this rectangle scaled about its own centre."""
        middle_x, middle_y = self.centre
        width = self.w * factor
        height = self.h * factor
        return Rect(middle_x - width / 2, middle_y - height / 2, width, height)

    def moved_to(self, middle_x: float, middle_y: float) -> "Rect":
        """Return this rectangle centred on a new point."""
        return Rect(middle_x - self.w / 2, middle_y - self.h / 2, self.w, self.h)

    def in_pixels(self, frame: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return this rectangle as whole pixels of a *frame* wide by high."""
        width, height = frame
        left = max(0, min(width - 1, round(self.x * width)))
        top = max(0, min(height - 1, round(self.y * height)))
        right = max(left + 1, min(width, round(self.right * width)))
        bottom = max(top + 1, min(height, round(self.bottom * height)))
        return left, top, right, bottom

    def to_dict(self) -> dict[str, float]:
        """Return the rectangle as it is stored (PLAN.md 7.1)."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Rect":
        """Build a rectangle from stored data, falling back to the whole frame."""
        if not isinstance(data, dict):
            return cls()
        rect = cls(
            _as_float(data.get("x"), 0.0),
            _as_float(data.get("y"), 0.0),
            _as_float(data.get("w"), 1.0),
            _as_float(data.get("h"), 1.0),
        )
        return rect.sane()

    def sane(self) -> "Rect":
        """Return a rectangle with a sensible size inside the frame."""
        width = min(1.0, max(MINIMUM_SIDE, self.w))
        height = min(1.0, max(MINIMUM_SIDE, self.h))
        x = min(max(0.0, self.x), 1.0 - width)
        y = min(max(0.0, self.y), 1.0 - height)
        return Rect(x, y, width, height)


WHOLE_FRAME: Final[Rect] = Rect()


@dataclass(frozen=True)
class Geometry:
    """How the photo is turned, rotated and cropped (PLAN.md 7.1)."""

    quarter_turns: int = 0
    angle: float = 0.0
    crop: Rect = WHOLE_FRAME
    aspect: str = ASPECT_ORIGINAL
    orientation: str = ORIENTATION_LANDSCAPE
    #: Left and right exchanged after the quarter turns and before the angle
    #: (C-08). One switch covers both flips: upside down is mirrored and
    #: turned half way round.
    mirrored: bool = False
    #: Whether and how much the lens is corrected (PLAN.md 2.2 and 6 step
    #: 1.5). Kept here because it changes the frame the crop is measured
    #: against, and because like the crop it belongs to this one photo and is
    #: not carried by copy and paste (P-03).
    lens: LensSettings = DEFAULT_SETTINGS
    #: The spots mended on the photo (PLAN.md 2.2 and 6 step 1.6), in
    #: fractions of the frame after the lens correction and before any turn.
    #: Kept here for the lens correction's reasons: they belong to this one
    #: photo, are not carried by copy and paste (P-03), and every place that
    #: frames the photo mends it on the way.
    spots: tuple[Spot, ...] = ()

    @property
    def is_default(self) -> bool:
        """Return whether the frame is untouched: not turned, not cropped.

        The lens correction is not asked about: where it rests depends on
        the kind of photo, so it is compared with where the photo started
        instead (:meth:`core.edit_state.Edit.is_untouched`). Nor are the
        spots, which mend the picture without changing its frame.
        """
        return (
            self.quarter_turns == 0
            and not self.mirrored
            and self.angle == 0.0
            and self.crop.is_whole_frame
        )

    def turned(self, quarter_turns: int) -> "Geometry":
        """Return this geometry turned by *quarter_turns*, keeping the crop shape.

        Turning the photo turns the crop with it, so that a 4:5 crop stays
        4:5 of the picture rather than becoming 5:4 of the frame.

        The turn is the one seen on screen. A mirrored photo is turned before
        it is mirrored, and the mirror sends a turn the other way, so on a
        mirrored photo the stored turns go back where the screen goes on.
        """
        step = -quarter_turns if self.mirrored else quarter_turns
        turns = (self.quarter_turns + step) % 4
        crop = self.crop
        for _ in range(quarter_turns % 4):
            crop = Rect(1.0 - crop.bottom, crop.x, crop.h, crop.w)
        orientation = self.orientation
        if quarter_turns % 2:
            orientation = (
                ORIENTATION_PORTRAIT
                if self.orientation == ORIENTATION_LANDSCAPE
                else ORIENTATION_LANDSCAPE
            )
        return replace(self, quarter_turns=turns, crop=crop, orientation=orientation)

    def flipped_orientation(self) -> "Geometry":
        """Return this geometry with the crop turned on its side (X, PLAN.md 3.2)."""
        return replace(
            self,
            orientation=(
                ORIENTATION_PORTRAIT
                if self.orientation == ORIENTATION_LANDSCAPE
                else ORIENTATION_LANDSCAPE
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the geometry as it is stored (PLAN.md 7.1)."""
        return {
            "quarter_turns": self.quarter_turns,
            "mirrored": self.mirrored,
            "angle": self.angle,
            "crop": self.crop.to_dict(),
            "aspect": self.aspect,
            "orientation": self.orientation,
            "lens": self.lens.to_dict(),
            "spots": healing.spots_to_list(self.spots),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Geometry":
        """Build a geometry from stored data; anything odd falls back."""
        if not isinstance(data, dict):
            return cls()
        aspect = data.get("aspect")
        orientation = data.get("orientation")
        return cls(
            quarter_turns=int(_as_float(data.get("quarter_turns"), 0.0)) % 4,
            mirrored=data.get("mirrored") is True,
            angle=_clamp(_as_float(data.get("angle"), 0.0), ANGLE_MIN, ANGLE_MAX),
            crop=Rect.from_dict(data.get("crop")),
            aspect=aspect if aspect in ASPECTS else ASPECT_ORIGINAL,
            orientation=(
                orientation
                if orientation in (ORIENTATION_LANDSCAPE, ORIENTATION_PORTRAIT)
                else ORIENTATION_LANDSCAPE
            ),
            lens=LensSettings.from_dict(data.get("lens")),
            spots=healing.spots_from_list(data.get("spots")),
        )


DEFAULT_GEOMETRY: Final[Geometry] = Geometry()


# -------------------------------------------------------------------------
# Sizes and shapes
# -------------------------------------------------------------------------


def turned_size(size: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    """Return *size* after *quarter_turns*, as width and height."""
    width, height = size
    return (height, width) if quarter_turns % 2 else (width, height)


def rotated_size(size: tuple[int, int], angle: float) -> tuple[int, int]:
    """Return the size of the box a rotated photo needs."""
    width, height = size
    radians = math.radians(angle)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))
    return (
        max(1, int(round(width * cosine + height * sine))),
        max(1, int(round(width * sine + height * cosine))),
    )


def frame_size(size: tuple[int, int], geometry: Geometry) -> tuple[int, int]:
    """Return the size of the frame the crop rectangle is measured against."""
    return rotated_size(turned_size(size, geometry.quarter_turns), geometry.angle)


def output_size(size: tuple[int, int], geometry: Geometry | None) -> tuple[int, int]:
    """Return what a photo of *size* measures once *geometry* is applied.

    The size an exported file will have: the frame after the quarter turns
    and the rotation, with the crop taken out of it. It is worked out from
    the crop fractions rather than from pixels, so the preview proxy and the
    full size photo give the same answer apart from rounding.
    """
    if geometry is None:
        return size
    frame = frame_size(size, geometry)
    if geometry.crop.is_whole_frame:
        return frame
    left, top, right, bottom = geometry.crop.in_pixels(frame)
    return right - left, bottom - top


def _frame_matrix(
    size: tuple[int, int], geometry: Geometry, crop: bool = True
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """Return where :func:`apply` sends each pixel, and the size it ends up.

    A 3 by 3 matrix taking pixels of the photo before any turn (OpenCV's
    convention, the middle of the first pixel at 0) to pixels of the frame
    as it is shown: the quarter turns, the mirror, the rotation and, with
    *crop*, the cut, exactly as :func:`apply` does them.
    """
    width, height = size
    matrix = np.eye(3, dtype=np.float64)
    for _ in range(geometry.quarter_turns % 4):
        # A quarter turn clockwise: (x, y) goes to (height - 1 - y, x).
        turn = np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        matrix = turn @ matrix
        width, height = height, width
    if geometry.mirrored:
        mirror = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        matrix = mirror @ matrix
    frame = (width, height)
    if geometry.angle != 0.0:
        rotation, frame = _rotation((width, height), geometry.angle)
        matrix = np.vstack((rotation, (0.0, 0.0, 1.0))) @ matrix
    if crop and not geometry.crop.is_whole_frame:
        left, top, right, bottom = geometry.crop.in_pixels(frame)
        cut = np.array([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]])
        matrix = cut @ matrix
        frame = (right - left, bottom - top)
    return matrix, frame


def shown_point(
    point: tuple[float, float], size: tuple[int, int], geometry: Geometry, crop: bool = True
) -> tuple[float, float]:
    """Return where a point of the unturned photo is shown, in fractions.

    *point* is in fractions of a photo of *size* before any turn, the way a
    spot is held (PLAN.md 6 step 1.6); what comes back is in fractions of
    the frame :func:`apply` makes of it.
    """
    matrix, frame = _frame_matrix(size, geometry, crop)
    x = point[0] * size[0] - 0.5
    y = point[1] * size[1] - 0.5
    shown_x, shown_y, _ = matrix @ np.array((x, y, 1.0))
    return float((shown_x + 0.5) / frame[0]), float((shown_y + 0.5) / frame[1])


def source_point(
    point: tuple[float, float], size: tuple[int, int], geometry: Geometry, crop: bool = True
) -> tuple[float, float]:
    """Return where a point of the shown frame lies on the unturned photo.

    The inverse of :func:`shown_point`: *point* is in fractions of the frame
    as shown, and the answer in fractions of the photo of *size*.
    """
    matrix, frame = _frame_matrix(size, geometry, crop)
    x = point[0] * frame[0] - 0.5
    y = point[1] * frame[1] - 0.5
    source_x, source_y, _ = np.linalg.inv(matrix) @ np.array((x, y, 1.0))
    return float((source_x + 0.5) / size[0]), float((source_y + 0.5) / size[1])


def shown_scale(size: tuple[int, int], geometry: Geometry, crop: bool = True) -> float:
    """Return the shown frame's long edge over the unturned photo's.

    Turning and rotating keep a pixel a pixel and a crop only cuts, so a
    spot's radius - a fraction of the long edge - is shown as this many
    times that fraction of the shown frame's long edge, inverted.
    """
    _matrix, frame = _frame_matrix(size, geometry, crop)
    return max(frame) / max(max(size), 1)


def written_ratio(aspect: str, source: tuple[int, int] | None = None) -> float | None:
    """Return the ratio *aspect* names, or ``None`` when it names none.

    ``original`` means the shape the photo itself has, which is why the size
    of the photo has to be passed in for it.
    """
    if aspect == ASPECT_FREE:
        return None
    if aspect == ASPECT_ORIGINAL:
        if source is None or source[1] == 0:
            return None
        return source[0] / source[1]
    return _FIXED_RATIOS.get(aspect)


def natural_orientation(aspect: str, source: tuple[int, int] | None = None) -> str:
    """Return the way round *aspect* is written: 4:5 upright, 3:2 on its side."""
    ratio = written_ratio(aspect, source)
    if ratio is None or ratio >= 1.0:
        return ORIENTATION_LANDSCAPE
    return ORIENTATION_PORTRAIT


def aspect_ratio(
    aspect: str, orientation: str, source: tuple[int, int] | None = None
) -> float | None:
    """Return the width over height a crop must keep, or ``None`` if it is free.

    The name of a ratio says nothing about which way round it is wanted - a
    4:5 crop turned on its side is 5:4 - so *orientation* decides, and the
    name only gives the two numbers.
    """
    ratio = written_ratio(aspect, source)
    if ratio is None:
        return None
    wide = max(ratio, 1 / ratio)
    return 1 / wide if orientation == ORIENTATION_PORTRAIT else wide


# -------------------------------------------------------------------------
# Where the crop box may be (C-05)
# -------------------------------------------------------------------------


def to_source(
    point: tuple[float, float], frame: tuple[int, int], source: tuple[int, int], angle: float
) -> tuple[float, float]:
    """Return where a point of the rotated frame sits on the unrotated photo."""
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    offset_x = point[0] - frame[0] / 2
    offset_y = point[1] - frame[1] / 2
    # The inverse of the rotation the photo was given.
    return (
        offset_x * cosine - offset_y * sine + source[0] / 2,
        offset_x * sine + offset_y * cosine + source[1] / 2,
    )


def is_inside(rect: Rect, source: tuple[int, int], angle: float) -> bool:
    """Return whether *rect* holds nothing but photo (C-05).

    *rect* is in fractions of the rotated frame, and *source* is the size of
    the photo before it was rotated.
    """
    frame = rotated_size(source, angle)
    corners = (
        (rect.x * frame[0], rect.y * frame[1]),
        (rect.right * frame[0], rect.y * frame[1]),
        (rect.x * frame[0], rect.bottom * frame[1]),
        (rect.right * frame[0], rect.bottom * frame[1]),
    )
    tolerance = 0.5  # half a pixel, so that rounding cannot reject a good box
    for corner in corners:
        x, y = to_source(corner, frame, source, angle)
        if not (-tolerance <= x <= source[0] + tolerance):
            return False
        if not (-tolerance <= y <= source[1] + tolerance):
            return False
    return True


def largest_rect(source: tuple[int, int], angle: float, ratio: float | None) -> Rect:
    """Return the largest centred crop of *ratio* that holds nothing but photo.

    With the crop centred, its bounding box in the photo's own frame is
    ``W·|cos| + H·|sin|`` by ``W·|sin| + H·|cos|``, so both of those have to
    fit the photo. Solving for the width gives the answer outright, with no
    searching.
    """
    width, height = source
    frame = rotated_size(source, angle)
    radians = math.radians(angle)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))

    if ratio is None:
        # A free crop keeps the shape of the frame it starts from.
        ratio = frame[0] / frame[1]

    horizontal = cosine + sine / ratio
    vertical = sine + cosine / ratio
    crop_width = min(width / max(horizontal, _EPSILON), height / max(vertical, _EPSILON))
    crop_height = crop_width / ratio

    return Rect(
        (frame[0] - crop_width) / (2 * frame[0]),
        (frame[1] - crop_height) / (2 * frame[1]),
        crop_width / frame[0],
        crop_height / frame[1],
    )


def rebase(rect: Rect, source: tuple[int, int], was: float, now: float) -> Rect:
    """Return *rect* expressed against the frame of another angle.

    The crop is stored in fractions of the rotated frame, and that frame grows
    as the photo is turned. Reading the same fractions against a larger frame
    would quietly change both the size and the shape of the crop, so when the
    angle moves the rectangle is carried over in pixels instead: both frames
    are centred on the same point and share the same scale, so a position
    measured from the centre means the same thing in either.
    """
    if was == now:
        return rect
    before = rotated_size(source, was)
    after = rotated_size(source, now)
    middle_x, middle_y = rect.centre
    centre_x = (middle_x - 0.5) * before[0]
    centre_y = (middle_y - 0.5) * before[1]
    width = rect.w * before[0] / after[0]
    height = rect.h * before[1] / after[1]
    return Rect(
        centre_x / after[0] + 0.5 - width / 2,
        centre_y / after[1] + 0.5 - height / 2,
        width,
        height,
    )


def fit_inside(rect: Rect, source: tuple[int, int], angle: float) -> Rect:
    """Return *rect* brought back inside the photo, as little changed as possible.

    First the box is slid back towards the middle if its own centre has left
    the photo, then it is shrunk about that centre until every corner is on
    the picture. Shrinking keeps the shape, which is what makes a fixed ratio
    survive a change of angle (C-02 and C-05 together).
    """
    if is_inside(rect, source, angle):
        return rect

    frame = rotated_size(source, angle)
    middle_x, middle_y = rect.centre
    centre_rect = Rect(middle_x, middle_y, 0.0, 0.0)
    if not is_inside(centre_rect, source, angle):
        largest = largest_rect(source, angle, rect.w * frame[0] / (rect.h * frame[1]))
        middle_x, middle_y = largest.centre
        rect = rect.moved_to(middle_x, middle_y)
        if is_inside(rect, source, angle):
            return rect

    low, high = 0.0, 1.0
    for _ in range(_SHRINK_STEPS):
        middle = (low + high) / 2
        if is_inside(rect.scaled(middle), source, angle):
            low = middle
        else:
            high = middle
    return rect.scaled(low)


def start_crop(source: tuple[int, int], geometry: Geometry) -> Rect:
    """Return the crop a photo should start with for its current shape."""
    turned = turned_size(source, geometry.quarter_turns)
    ratio = aspect_ratio(geometry.aspect, geometry.orientation, turned)
    return largest_rect(turned, geometry.angle, ratio)


# -------------------------------------------------------------------------
# Dragging the crop box (PLAN.md 3.2)
# -------------------------------------------------------------------------


def move(rect: Rect, delta: tuple[float, float], source: tuple[int, int], angle: float) -> Rect:
    """Return *rect* moved by *delta*, never off the picture.

    *rect* is where the box was when the drag began and *delta* is the whole
    movement since, so a movement that would take the box off the picture
    must not be refused: the box has to stop against the edge and stay
    there, or carrying on with the drag would send it back to where it
    started (C-05).
    """
    moved = Rect(rect.x + delta[0], rect.y + delta[1], rect.w, rect.h)
    if is_inside(moved, source, angle):
        return moved
    # Each direction is taken as far as the picture allows, one at a time,
    # so that pushing against the top edge still slides the box sideways.
    against_edge = _slid(rect, delta, 0, source, angle)
    return _slid(against_edge, delta, 1, source, angle)


def _slid(
    rect: Rect, delta: tuple[float, float], axis: int, source: tuple[int, int], angle: float
) -> Rect:
    """Return *rect* moved as far along one axis as stays on the picture."""
    step = delta[axis]
    if step == 0.0:
        return rect

    def at(fraction: float) -> Rect:
        shift = step * fraction
        return Rect(
            rect.x + (shift if axis == 0 else 0.0),
            rect.y + (0.0 if axis == 0 else shift),
            rect.w,
            rect.h,
        )

    if is_inside(at(1.0), source, angle):
        return at(1.0)
    low, high = 0.0, 1.0
    for _ in range(_SHRINK_STEPS):
        middle = (low + high) / 2
        if is_inside(at(middle), source, angle):
            low = middle
        else:
            high = middle
    # is_inside allows half a pixel either way, so the last step can land a
    # hair outside the frame; the box belongs inside it.
    return at(low).sane()



def resize(
    rect: Rect,
    handle: Handle,
    pointer: tuple[float, float],
    source: tuple[int, int],
    angle: float,
    ratio: float | None,
) -> Rect:
    """Return *rect* with *handle* dragged to *pointer*.

    The edge or corner being dragged follows the pointer and the opposite one
    stays where it is. With a fixed ratio the other side follows, which is
    what keeps a 4:5 crop 4:5 however it is dragged (C-02). A result that
    would take in anything beyond the picture is refused rather than
    corrected, so the box simply stops at the edge instead of jumping.
    """
    if handle in (Handle.NONE, Handle.INSIDE, Handle.OUTSIDE):
        return rect

    frame = rotated_size(source, angle)
    # The ratio is of the picture, the rectangle is of the frame.
    frame_ratio = frame[0] / frame[1]
    shape = None if ratio is None else ratio / frame_ratio

    left, top, right, bottom = rect.x, rect.y, rect.right, rect.bottom
    pointer_x = _clamp(pointer[0], 0.0, 1.0)
    pointer_y = _clamp(pointer[1], 0.0, 1.0)

    if handle in (Handle.TOP_LEFT, Handle.LEFT, Handle.BOTTOM_LEFT):
        left = min(pointer_x, right - MINIMUM_SIDE)
    if handle in (Handle.TOP_RIGHT, Handle.RIGHT, Handle.BOTTOM_RIGHT):
        right = max(pointer_x, left + MINIMUM_SIDE)
    if handle in (Handle.TOP_LEFT, Handle.TOP, Handle.TOP_RIGHT):
        top = min(pointer_y, bottom - MINIMUM_SIDE)
    if handle in (Handle.BOTTOM_LEFT, Handle.BOTTOM, Handle.BOTTOM_RIGHT):
        bottom = max(pointer_y, top + MINIMUM_SIDE)

    width = right - left
    height = bottom - top

    if shape is not None:
        if handle.is_edge:
            # One edge was dragged, so the other pair follows around the middle.
            if handle in (Handle.LEFT, Handle.RIGHT):
                height = width / shape
                middle = (top + bottom) / 2
                top, bottom = middle - height / 2, middle + height / 2
            else:
                width = height * shape
                middle = (left + right) / 2
                left, right = middle - width / 2, middle + width / 2
        else:
            # A corner was dragged: the opposite corner is the anchor.
            if width / max(height, _EPSILON) > shape:
                height = width / shape
            else:
                width = height * shape
            if handle in (Handle.TOP_LEFT, Handle.BOTTOM_LEFT):
                left = right - width
            else:
                right = left + width
            if handle in (Handle.TOP_LEFT, Handle.TOP_RIGHT):
                top = bottom - height
            else:
                bottom = top + height

    changed = Rect(left, top, right - left, bottom - top)
    if changed.w < MINIMUM_SIDE or changed.h < MINIMUM_SIDE:
        return rect
    if not is_inside(changed, source, angle):
        return rect
    return changed


def reshape(rect: Rect, source: tuple[int, int], angle: float, ratio: float | None) -> Rect:
    """Return *rect* given a new shape, as large as it can be where it is.

    Used when the aspect buttons are pressed: the crop keeps its middle and
    takes the new shape, growing to whatever the picture allows.
    """
    if ratio is None:
        return rect
    frame = rotated_size(source, angle)
    shape = ratio / (frame[0] / frame[1])
    middle_x, middle_y = rect.centre

    # Start from the largest box of this shape and bring it to where the crop
    # is; then shrink it until it fits, which fit_inside already does.
    largest = largest_rect(source, angle, ratio)
    width, height = largest.w, largest.h
    current_area = rect.w * rect.h
    scale = min(1.0, math.sqrt(current_area / max(width * height, _EPSILON)))
    width *= scale
    height *= scale
    if abs(width / max(height, _EPSILON) - shape) > _EPSILON:
        height = width / shape
    return fit_inside(Rect(middle_x - width / 2, middle_y - height / 2, width, height), source, angle)


# -------------------------------------------------------------------------
# Changing one thing about a geometry (the crop tab's buttons)
# -------------------------------------------------------------------------


def with_angle(geometry: Geometry, source: tuple[int, int], angle: float) -> Geometry:
    """Return *geometry* rotated to *angle*, with the crop brought along."""
    angle = _clamp(angle, ANGLE_MIN, ANGLE_MAX)
    turned = turned_size(source, geometry.quarter_turns)
    crop = rebase(geometry.crop, turned, geometry.angle, angle)
    return replace(geometry, angle=angle, crop=fit_inside(crop, turned, angle))


def with_crop(geometry: Geometry, source: tuple[int, int], crop: Rect) -> Geometry:
    """Return *geometry* with a new crop box, kept on the picture."""
    turned = turned_size(source, geometry.quarter_turns)
    return replace(geometry, crop=fit_inside(crop.sane(), turned, geometry.angle))


def with_aspect(geometry: Geometry, source: tuple[int, int], aspect: str) -> Geometry:
    """Return *geometry* with the crop reshaped to *aspect* (C-02).

    A ratio is first given the way round it is written, so that choosing 4:5
    gives an upright crop and choosing 3:2 a wide one; X turns it over.
    """
    if aspect not in ASPECTS:
        return geometry
    turned = turned_size(source, geometry.quarter_turns)
    orientation = natural_orientation(aspect, turned)
    ratio = aspect_ratio(aspect, orientation, turned)
    if aspect == ASPECT_ORIGINAL:
        # "Original" names the photo's own frame, so choosing it gives the
        # whole of that frame back, rather than a box of the same shape
        # where the last crop happened to be. On a straightened photo that
        # is the largest box of the frame's shape that still holds only
        # picture.
        crop = largest_rect(turned, geometry.angle, ratio)
    else:
        crop = reshape(geometry.crop, turned, geometry.angle, ratio)

    return replace(
        geometry,
        aspect=aspect,
        orientation=orientation,
        crop=fit_inside(crop, turned, geometry.angle),
    )


def with_orientation_flipped(geometry: Geometry, source: tuple[int, int]) -> Geometry:
    """Return *geometry* with its crop stood on its side (X, PLAN.md 3.2).

    Flipping twice has to give back exactly what was there, or pressing X
    again and again walks the box across the picture. So a box that fills
    as much as its shape allows becomes the fullest box of the shape it is
    turned into, and any other box simply has its sides swapped where it
    stands. Both of those are their own opposite.
    """
    flipped = geometry.flipped_orientation()
    turned = turned_size(source, flipped.quarter_turns)
    ratio = aspect_ratio(flipped.aspect, flipped.orientation, turned)
    if ratio is None:
        return flipped

    was = aspect_ratio(geometry.aspect, geometry.orientation, turned)
    if was is not None and _fills_what_it_can(geometry.crop, turned, geometry.angle, was):
        crop = largest_rect(turned, flipped.angle, ratio)
    else:
        crop = _sides_swapped(geometry.crop, rotated_size(turned, flipped.angle))
    return replace(flipped, crop=fit_inside(crop, turned, flipped.angle))


def _sides_swapped(rect: Rect, frame: tuple[int, int]) -> Rect:
    """Return *rect* with its width and height exchanged, about its middle.

    The sides are swapped in pixels rather than in fractions of the frame,
    which is what makes a square crop stay square on a frame that is not.
    """
    middle_x, middle_y = rect.centre
    width = rect.h * frame[1] / max(frame[0], 1)
    height = rect.w * frame[0] / max(frame[1], 1)
    return Rect(middle_x - width / 2, middle_y - height / 2, width, height)


def _fills_what_it_can(
    rect: Rect, source: tuple[int, int], angle: float, ratio: float
) -> bool:
    """Return whether *rect* is as large as its shape allows on the picture."""
    largest = largest_rect(source, angle, ratio)
    return abs(rect.w * rect.h - largest.w * largest.h) < _FULL_ENOUGH



def with_quarter_turn(geometry: Geometry, quarter_turns: int) -> Geometry:
    """Return *geometry* turned by a right angle each way (C-06)."""
    return geometry.turned(quarter_turns)


def with_mirror(geometry: Geometry, vertical: bool = False) -> Geometry:
    """Return *geometry* flipped left to right, or top to bottom (C-08).

    The flip is the one seen on screen: what was on the left of the picture
    as it stands, turned and straightened, comes out on the right, and the
    crop box goes over with it. Mirroring a rotation turns it the other way,
    so the angle changes sign; that is what makes the result exactly the
    mirror image of what was there, and flipping twice gives it all back.

    Top to bottom is left to right and half a turn more, which is how it is
    stored: the mirror stays a single switch whichever way it was made.
    """
    crop = geometry.crop
    turns = geometry.quarter_turns
    if vertical:
        crop = Rect(crop.x, 1.0 - crop.bottom, crop.w, crop.h)
        turns = (turns + 2) % 4
    else:
        crop = Rect(1.0 - crop.right, crop.y, crop.w, crop.h)
    return replace(
        geometry,
        quarter_turns=turns,
        mirrored=not geometry.mirrored,
        # Written out so that a level photo never stores minus zero.
        angle=-geometry.angle if geometry.angle else 0.0,
        crop=crop,
    )


def straighten_angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Return the angle that makes the line from *start* to *end* level (C-04).

    The points are in the frame's own coordinates, where y grows downwards.
    A line that drops towards the right has to be turned back the other way,
    and a line closer to vertical is read as a vertical edge, such as the side
    of a building, which levels the other way round.
    """
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    if abs(delta_x) < _EPSILON and abs(delta_y) < _EPSILON:
        return 0.0
    if abs(delta_x) >= abs(delta_y):
        angle = math.degrees(math.atan2(delta_y, delta_x))
    else:
        # Measured against the vertical instead.
        angle = math.degrees(math.atan2(-delta_x, delta_y))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return _clamp(angle, ANGLE_MIN, ANGLE_MAX)


# -------------------------------------------------------------------------
# Doing it to the photo (PLAN.md section 6 step 2)
# -------------------------------------------------------------------------


def apply(photo: DecodedPhoto, geometry: Geometry | None, crop: bool = True) -> DecodedPhoto:
    """Return *photo* with its lens corrected, turned, mirrored, rotated and cropped.

    The lens comes first (PLAN.md section 6 step 1.5): its profile describes
    the frame as the sensor saw it, before anyone turned or cropped it. The
    spots are mended next (step 1.6), on the frame they were placed on.

    Nothing is copied when there is nothing to do, which is what lets an
    untouched JPEG still be shown exactly as it was decoded (PLAN.md section 6,
    "JPG 原圖直通"). With *crop* false the photo is only turned and rotated,
    which is what the crop tab shows while the box is being dragged
    (PLAN.md 3.2).
    """
    if geometry is not None:
        photo = lens_module.correct(photo, geometry.lens)
        photo = healing.heal(photo, geometry.spots)
    if geometry is None or geometry.is_default:
        return photo
    if (
        not crop
        and geometry.quarter_turns == 0
        and not geometry.mirrored
        and geometry.angle == 0.0
    ):
        return photo

    source_size = photo.size
    image = _turn(photo.image, geometry.quarter_turns, geometry.mirrored)
    clipped = _turn(photo.clipped, geometry.quarter_turns, geometry.mirrored)
    encoded = _turn(photo.encoded, geometry.quarter_turns, geometry.mirrored)
    cut = crop and not geometry.crop.is_whole_frame
    # The faces are moved the way the pixels are: by the same turn, the same
    # rotation matrix and the same cut (PLAN.md 6 step 2.6).
    moved: NDArray[np.float64] | None = None
    if photo.faces:
        moved = np.concatenate([face.in_pixels(source_size) for face in photo.faces])
        moved = _turned_points(moved, source_size, geometry.quarter_turns, geometry.mirrored)

    if geometry.angle != 0.0:
        size = (image.shape[1], image.shape[0])
        matrix, frame = _rotation(size, geometry.angle)
        window = frame
        if cut:
            # Rotate straight into the crop instead of rotating the whole
            # frame and throwing most of it away: the pixels that are kept
            # are the same either way, and there are far fewer of them.
            left, top, right, bottom = geometry.crop.in_pixels(frame)
            matrix[0, 2] -= left
            matrix[1, 2] -= top
            window = (right - left, bottom - top)
            cut = False
        if moved is not None:
            moved = moved @ matrix[:, :2].T + matrix[:, 2]
        # Where the crop is being taken, the edge is repeated so that the
        # interpolation has something to work with and nothing darkens the
        # outermost row of real pixels. Where the whole rotated frame is
        # shown, in the crop tab, the empty corners must look empty rather
        # than smeared (PLAN.md 3.2).
        edge = cv2.BORDER_REPLICATE if crop else cv2.BORDER_CONSTANT
        image = _warp(image, matrix, window, cv2.INTER_CUBIC, edge)
        clipped = _warp(clipped.astype(np.uint8), matrix, window, cv2.INTER_NEAREST, edge) > 0
        if encoded is not None:
            encoded = _warp(encoded, matrix, window, cv2.INTER_CUBIC, edge)

    if cut:
        frame = (image.shape[1], image.shape[0])
        left, top, right, bottom = geometry.crop.in_pixels(frame)
        image = image[top:bottom, left:right]
        clipped = clipped[top:bottom, left:right]
        if encoded is not None:
            encoded = encoded[top:bottom, left:right]
        if moved is not None:
            moved = moved - np.array([left, top], dtype=np.float64)

    faces = photo.faces
    if moved is not None and faces:
        size = (image.shape[1], image.shape[0])
        moved_faces: list[Face] = []
        start = 0
        for face in faces:
            count = len(face.points)
            moved_faces.append(face.moved(moved[start : start + count], size))
            start += count
        faces = tuple(moved_faces)

    return replace(
        photo,
        faces=faces,
        image=np.ascontiguousarray(image),
        clipped=np.ascontiguousarray(clipped),
        encoded=None if encoded is None else np.ascontiguousarray(encoded),
        # A proxy stays a proxy of the same original: what the full size
        # photo would measure once the same geometry had been applied to it.
        shrunk_from=(
            None
            if photo.shrunk_from is None
            else (
                output_size(photo.full_size, geometry)
                if crop
                else frame_size(photo.full_size, geometry)
            )
        ),
    )


def _turned_points(
    points: NDArray[np.float64], size: tuple[int, int], quarter_turns: int, mirrored: bool
) -> NDArray[np.float64]:
    """Return *points* (pixels of a picture of *size*) where :func:`_turn` puts them."""
    width, height = size
    xs = points[:, 0].copy()
    ys = points[:, 1].copy()
    for _ in range(quarter_turns % 4):
        # A quarter turn clockwise: the left column becomes the top row.
        xs, ys = (height - 1) - ys, xs
        width, height = height, width
    if mirrored:
        xs = (width - 1) - xs
    return np.stack((xs, ys), axis=1)


def _turn(
    values: NDArray[Any] | None, quarter_turns: int, mirrored: bool = False
) -> NDArray[Any] | None:
    """Return *values* turned clockwise by right angles, then mirrored if asked."""
    if values is None:
        return values
    if quarter_turns % 4:
        values = np.rot90(values, k=-(quarter_turns % 4))
    if mirrored:
        values = values[:, ::-1]
    return values


def _rotation(
    size: tuple[int, int], angle: float
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """Return the rotation matrix and the size of the box it needs."""
    width, height = size
    frame = rotated_size(size, angle)
    # OpenCV puts pixel centres on whole numbers, so the middle of a photo
    # w wide is (w - 1) / 2. Turning about w / 2 instead would shift every
    # rotated photo by up to half a pixel, and a flipped one the other way.
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    matrix[0, 2] += (frame[0] - width) / 2
    matrix[1, 2] += (frame[1] - height) / 2
    return matrix, frame


def _warp(
    values: NDArray[Any],
    matrix: NDArray[np.float64],
    frame: tuple[int, int],
    how: int,
    edge: int = cv2.BORDER_REPLICATE,
) -> NDArray[Any]:
    """Rotate one array into a frame of its own."""
    return cv2.warpAffine(values, matrix, frame, flags=how, borderMode=edge)


def _clamp(value: float, low: float, high: float) -> float:
    """Return *value* held between *low* and *high*."""
    return max(low, min(high, value))


# -------------------------------------------------------------------------
# Zooming in: one window of the frame (PLAN.md 3.3 and section 4)
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A rectangle of pixels, and how large the frame it was cut from is.

    Zoomed in past what the preview proxy can show, only what is on screen is
    developed, from the full size photo (PLAN.md 3.3 and section 4). Three
    parts of the pipeline are about the frame rather than about the pixel -
    the two blurs of section 6 and the vignette - so a window carries its
    frame with it and the piece comes out exactly as the whole would have.

    The four edges are in pixels of that frame, right and bottom exclusive,
    the way a slice reads.
    """

    #: The width and height of the whole frame.
    frame: tuple[int, int]
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def whole(cls, frame: tuple[int, int]) -> "Window":
        """Return the window that covers all of *frame*."""
        width, height = frame
        return cls(frame=(width, height), left=0, top=0, right=width, bottom=height)

    @classmethod
    def around(
        cls, frame: tuple[int, int], box: tuple[float, float, float, float], margin: int = 0
    ) -> "Window":
        """Return the window covering *box* with *margin* pixels around it.

        The box is in pixels of *frame* and may hang over its edges, which is
        what a photo panned to the edge of the view gives; what comes back is
        always inside the frame and at least one pixel across.
        """
        width, height = frame
        left, top, right, bottom = box
        inside_left = min(max(int(math.floor(left)) - margin, 0), max(width - 1, 0))
        inside_top = min(max(int(math.floor(top)) - margin, 0), max(height - 1, 0))
        return cls(
            frame=(width, height),
            left=inside_left,
            top=inside_top,
            right=max(min(int(math.ceil(right)) + margin, width), inside_left + 1),
            bottom=max(min(int(math.ceil(bottom)) + margin, height), inside_top + 1),
        )

    @property
    def size(self) -> tuple[int, int]:
        """Return the width and height of the window itself."""
        return self.right - self.left, self.bottom - self.top

    @property
    def is_whole(self) -> bool:
        """Return whether the window is the whole frame."""
        return (self.left, self.top) == (0, 0) and (self.right, self.bottom) == self.frame

    def scaled(self, scale: float) -> "Window":
        """Return this window measured in a frame *scale* times the size.

        What the preview does when it develops a window at less than the
        photo's own resolution: the frame and the window shrink together, so
        the window still sits in the same place in the picture.
        """
        width, height = self.frame
        frame = (max(1, round(width * scale)), max(1, round(height * scale)))
        return Window.around(
            frame,
            (
                self.left * scale,
                self.top * scale,
                self.right * scale,
                self.bottom * scale,
            ),
        )

    def grown(self, margin: int) -> "Window":
        """Return this window with *margin* pixels more on every side."""
        if margin <= 0:
            return self
        return Window.around(
            self.frame, (self.left, self.top, self.right, self.bottom), margin
        )

    def inside(self, other: "Window") -> tuple[int, int, int, int]:
        """Return where this window sits inside *other*, in its pixels."""
        return (
            self.left - other.left,
            self.top - other.top,
            self.right - other.left,
            self.bottom - other.top,
        )

    def take(self, values: NDArray[Any]) -> NDArray[Any]:
        """Return the part of a whole frame this window covers, at its size.

        *values* holds the whole frame at whatever resolution it happens to
        have - the photo itself, a mask, or step 7's local brightness - so
        this both cuts and scales.
        """
        height, width = values.shape[:2]
        frame_width, frame_height = self.frame
        if self.is_whole and (width, height) == (frame_width, frame_height):
            return values
        across = width / frame_width
        down = height / frame_height
        left = min(int(math.floor(self.left * across)), width - 1)
        top = min(int(math.floor(self.top * down)), height - 1)
        right = max(min(int(math.ceil(self.right * across)), width), left + 1)
        bottom = max(min(int(math.ceil(self.bottom * down)), height), top + 1)
        part = values[top:bottom, left:right]
        if (right - left, bottom - top) == self.size:
            return part
        if part.dtype == np.bool_:
            return _resized_mask(part, self.size)
        return _resized(part, self.size)

    def trim(self, image: NDArray[Any], inner: "Window") -> NDArray[Any]:
        """Return the part of *image* that *inner* covers.

        *image* is this window's own pixels, margin included; *inner* is the
        window that was asked for. What the margin was developed for is the
        blurs reaching into it, and it is cut off here.
        """
        left, top, right, bottom = inner.inside(self)
        if (left, top) == (0, 0) and (right, bottom) == self.size:
            return image
        return np.ascontiguousarray(image[top:bottom, left:right])


def window_of(photo: DecodedPhoto, window: Window) -> DecodedPhoto:
    """Return the part of *photo* that *window* covers, at the window's size.

    *photo* holds the whole frame at whatever resolution it was decoded at,
    and the window says both where to cut and how many pixels to come back
    with, so a window of the full size photo and a window of the proxy are
    asked for the same way.
    """
    height, width = photo.image.shape[:2]
    if window.is_whole and (width, height) == window.frame:
        return photo

    image = window.take(photo.image)
    clipped = window.take(photo.clipped)
    encoded = None if photo.encoded is None else window.take(photo.encoded)

    # How large the window would be at full resolution, which is what the
    # detail step measures its sizes against (PLAN.md section 6 step 2.5):
    # a window of the screen sized preview is sharpened as finely as the
    # preview, one of the full size photo as finely as the export.
    scale = max(window.frame) / max(max(photo.full_size), 1)
    width, height = window.size
    full = (max(1, round(width / scale)), max(1, round(height / scale)))
    return replace(
        photo,
        image=np.ascontiguousarray(image),
        clipped=np.ascontiguousarray(clipped),
        encoded=None if encoded is None else np.ascontiguousarray(encoded),
        shrunk_from=None if full == (width, height) else full,
    )


def _resized(image: NDArray[Any], size: tuple[int, int]) -> NDArray[Any]:
    """Return *image* at *size*, with the filter that suits the direction."""
    shrinking = size[0] <= image.shape[1] and size[1] <= image.shape[0]
    return cv2.resize(
        image, size, interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    )


def _resized_mask(mask: NDArray[np.bool_], size: tuple[int, int]) -> NDArray[np.bool_]:
    """Return a boolean mask at *size*, keeping a pixel that is mostly covered."""
    covered = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return covered > 0.5


def _as_float(value: Any, default: float) -> float:
    """Return *value* as a float, or *default* when it is not a number."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


__all__ = [
    "ANGLE_MAX",
    "ANGLE_MIN",
    "ANGLE_STEP",
    "ASPECTS",
    "ASPECT_FREE",
    "ASPECT_ORIGINAL",
    "DEFAULT_GEOMETRY",
    "Geometry",
    "Handle",
    "ORIENTATION_LANDSCAPE",
    "ORIENTATION_PORTRAIT",
    "Rect",
    "WHOLE_FRAME",
    "Window",
    "apply",
    "aspect_ratio",
    "fit_inside",
    "frame_size",
    "is_inside",
    "largest_rect",
    "move",
    "rebase",
    "reshape",
    "resize",
    "rotated_size",
    "shown_point",
    "shown_scale",
    "source_point",
    "start_crop",
    "straighten_angle",
    "to_source",
    "natural_orientation",
    "output_size",
    "turned_size",

    "window_of",
    "with_angle",
    "with_aspect",
    "with_crop",
    "with_mirror",
    "with_orientation_flipped",
    "with_quarter_turn",
]
