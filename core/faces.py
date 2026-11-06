"""Where the faces in a photo are, and which of its pixels are skin (PLAN.md 6 step 2.6).

A :class:`Face` is six points - the two eyes, the tip of the nose, the two
corners of the mouth and the middle of the box the detector drew - the size
of that box, and the colour of that person's skin as the photo recorded it. The points are fractions of the frame rather than pixels, so the
same face fits the proxy, the screen sized preview, a zoomed in window and
the full size export alike; :mod:`core.geometry` moves them with the photo
when it is turned, mirrored, rotated or cropped.

From the faces this module draws the skin mask: a soft region shaped like a
face and the neck below it, cut back to the pixels that have the colour of
that person's own cheeks, with the eyes, the brows and the mouth taken out.
Nothing here detects faces; that is :mod:`core.face_detector`, which needs a
model file and the pipeline. This module is plain arithmetic, so the pipeline
can use it without either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------
# The shape of a face, in units of the face itself
# --------------------------------------------------------------------------
#
# S is the distance from between the eyes to the middle of the mouth, which
# stays much the same when a head turns to one side. Directions follow the
# face: "up" from the mouth towards the eyes, "across" at right angles to it.
#
# The face and the neck are placed by the box the detector drew around the
# face, which sits on the face itself even when the head is turned and the
# eyes and mouth have moved to one side of it. Measured on portraits, the
# box runs from the brows to the chin and from cheek to cheek, about 2 S
# across and 2.7 S high.

#: The face, from the hairline to the chin and from cheek to cheek: how far
#: its middle lies above the middle of the box, and its half width and half
#: height, all as shares of the width and height of the box.
FACE_RAISE: Final[float] = 0.08
FACE_HALF_ACROSS: Final[float] = 0.56
FACE_HALF_UP: Final[float] = 0.6
#: The neck, below the chin. Leaving it out would brighten a face above a
#: neck that stays as it was, which shows as a line along the jaw. How far
#: its middle lies below the middle of the box, and its half width and half
#: height, as shares of the box.
NECK_LOWER: Final[float] = 0.85
NECK_HALF_ACROSS: Final[float] = 0.42
NECK_HALF_UP: Final[float] = 0.38
#: How soft the edge of those two regions is: full inside this share of the
#: way out, nothing beyond the second.
REGION_EDGE: Final[tuple[float, float]] = (0.8, 1.15)
#: The middle of the face that is skin whatever its colour, as a share of
#: the face's size.
CORE_SHARE: Final[float] = 0.7

#: The eyes, the brows and the mouth are taken out of the mask. The eyes and
#: brows are sized by the distance between the eyes (D), the mouth by its own
#: width (W): half across and half up for each.
EYE_HALF: Final[tuple[float, float]] = (0.32, 0.19)
BROW_UP: Final[float] = 0.36
BROW_HALF: Final[tuple[float, float]] = (0.38, 0.14)
MOUTH_HALF: Final[tuple[float, float]] = (0.56, 0.3)
#: How soft the edge of what is taken out is, as for REGION_EDGE. The
#: mouth's is narrower: wide, it took in the cheek beside the corners of the
#: mouth and up to the nose, where blemishes are common (measured on 15
#: faces, a wide laugh among them, the lips still all inside).
FEATURE_EDGE: Final[tuple[float, float]] = (0.85, 1.35)
MOUTH_EDGE: Final[tuple[float, float]] = (0.85, 1.2)

#: How far the colour of a pixel may be from the cheeks, in multiples of how
#: much the cheeks themselves vary: fully skin within the first, not at all
#: beyond the second.
COLOUR_REACH: Final[tuple[float, float]] = (1.5, 3.0)
#: The least the cheeks are taken to vary, in stops of red against green and
#: of blue against green, so that a face lit evenly does not leave its own
#: shadowed side out.
COLOUR_LEAST_SPREAD: Final[float] = 0.06
#: The most they are taken to vary. A sample that caught an eyelash or the
#: edge of a nostril would otherwise let hair and background in with it.
COLOUR_MOST_SPREAD: Final[float] = 0.15
#: How much darker and how much brighter than the cheeks skin may be, in
#: stops: fully within the first, not at all beyond the second.
DARKER_REACH: Final[tuple[float, float]] = (2.5, 4.0)
BRIGHTER_REACH: Final[tuple[float, float]] = (0.8, 1.6)
#: Spots on the skin - a blemish, a red mark, a mole - are not the colour of
#: the cheeks either, and would leave holes in the mask exactly where the
#: smoothing is wanted most. Anything unlike skin smaller than this many S
#: across, inside skin, is taken as skin; the hair and the background beside
#: the face are far larger and keep their edge.
COLOUR_HOLE_FILL: Final[float] = 0.12
#: The mask is softened by this many S, so that one stray pixel the colour of
#: skin in a lock of hair does not show as a speck.
MASK_SOFTEN: Final[float] = 0.04

_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True)
class SkinTone:
    """The colour of one person's skin as it was recorded.

    Measured in the photo's own linear colour - the camera's for a RAW file,
    already balanced the way the camera was set - as the ratios of red and of
    blue to green, in stops, and the brightness, in stops. Ratios do not
    change with exposure, and the photo's own colour does not change with any
    slider, so a face is recognised the same however the photo is adjusted.
    """

    red: float
    blue: float
    #: How much the cheeks varied, in stops; at least COLOUR_LEAST_SPREAD.
    spread: float
    brightness: float


@dataclass(frozen=True)
class Face:
    """One face: six points as fractions of the frame, its size and skin tone.

    The points are, in order, the two eyes, the tip of the nose, the two
    corners of the mouth, and the middle of the box the detector drew around
    the face. A fraction is measured from the outer edge of the frame, so 0.5
    is the middle whatever the photo's resolution. The width and height of
    the box are kept as multiples of S, which do not change when the photo
    is turned or scaled.
    """

    points: tuple[tuple[float, float], ...]
    tone: SkinTone
    width: float = 2.0
    height: float = 2.7

    def in_pixels(self, frame: tuple[int, int]) -> NDArray[np.float64]:
        """Return the points in pixels of a frame of *frame* (width, height).

        In OpenCV's convention, where the middle of the first pixel is 0.
        """
        width, height = frame
        return np.array(
            [(u * width - 0.5, v * height - 0.5) for u, v in self.points], dtype=np.float64
        )

    def moved(self, points: NDArray[np.float64], frame: tuple[int, int]) -> "Face":
        """Return this face with its points at *points*, in pixels of *frame*."""
        width, height = frame
        return Face(
            points=tuple(
                (float((x + 0.5) / width), float((y + 0.5) / height)) for x, y in points
            ),
            tone=self.tone,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class Placement:
    """Where one face lies in a photo at hand, in its pixels."""

    face: Face
    #: The six points.
    points: NDArray[np.float64]
    #: The middle of the box the detector drew.
    centre: NDArray[np.float64]
    #: From the mouth towards the eyes, and across, both of length one.
    up: NDArray[np.float64]
    across: NDArray[np.float64]
    #: The size of the face, S (see the top of this module), in pixels.
    size: float
    #: The rectangle the face and neck can reach, as left, top, right and
    #: bottom pixels of the photo at hand, the last two exclusive. Empty when
    #: the face lies wholly outside it.
    box: tuple[int, int, int, int]


def face_size(face: Face, frame: tuple[int, int]) -> float:
    """Return S for *face* in a frame of *frame*, in its pixels.

    The larger of the distance from between the eyes to the mouth and four
    fifths of the distance between the eyes, so that a face seen from above
    or below, which foreshortens the first, still gets its size.
    """
    points = face.in_pixels(frame)
    eyes = (points[0] + points[1]) / 2.0
    mouth = (points[3] + points[4]) / 2.0
    return float(
        max(np.linalg.norm(eyes - mouth), 0.8 * np.linalg.norm(points[0] - points[1]), 1.0)
    )


def place(
    faces: tuple[Face, ...],
    frame: tuple[int, int],
    left: int = 0,
    top: int = 0,
    size: tuple[int, int] | None = None,
) -> list[Placement]:
    """Return where each of *faces* lies in a photo at hand.

    *frame* is the whole frame at the photo's resolution, and *left*, *top*
    and *size* the window of it the photo is, when it is one (PLAN.md 3.3);
    without them the photo is the whole frame.
    """
    width, height = frame if size is None else size
    placed: list[Placement] = []
    for face in faces:
        points = face.in_pixels(frame) - np.array([left, top], dtype=np.float64)
        eyes = (points[0] + points[1]) / 2.0
        mouth = (points[3] + points[4]) / 2.0
        upward = eyes - mouth
        length = float(np.linalg.norm(upward))
        if length < _EPSILON:
            continue
        up = upward / length
        across = np.array([up[1], -up[0]])
        scale = face_size(face, frame)
        centre = points[5]
        # The farthest any part of the face or the neck can reach from the
        # centre, with a margin for softening the mask.
        reach = scale * (
            max(
                (FACE_HALF_UP + FACE_RAISE) * face.height,
                (NECK_HALF_UP + NECK_LOWER) * face.height,
                FACE_HALF_ACROSS * face.width,
            )
            * REGION_EDGE[1]
            + 4.0 * MASK_SOFTEN
        )
        box_left = max(int(math.floor(centre[0] - reach)), 0)
        box_top = max(int(math.floor(centre[1] - reach)), 0)
        box_right = min(int(math.ceil(centre[0] + reach)) + 1, width)
        box_bottom = min(int(math.ceil(centre[1] + reach)) + 1, height)
        if box_right <= box_left or box_bottom <= box_top:
            box = (0, 0, 0, 0)
        else:
            box = (box_left, box_top, box_right, box_bottom)
        placed.append(Placement(face, points, centre, up, across, scale, box))
    return placed


def _smoothstep(values: NDArray[np.float32], low: float, high: float) -> NDArray[np.float32]:
    """Return 0 below *low*, 1 above *high*, and a smooth step in between."""
    position = np.clip((values - np.float32(low)) / np.float32(high - low), 0.0, 1.0)
    return (position * position * (np.float32(3.0) - np.float32(2.0) * position)).astype(
        np.float32
    )


def _ellipse(
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    centre: NDArray[np.float64],
    along: NDArray[np.float64],
    half: tuple[float, float],
    edge: tuple[float, float],
) -> NDArray[np.float32]:
    """Return how far inside an ellipse each pixel is: 1 inside, 0 outside.

    The ellipse's first half axis lies along *along* and its second across
    it; *edge* is the soft rim, as shares of the way out.
    """
    dx = xs - np.float32(centre[0])
    dy = ys - np.float32(centre[1])
    first = (dx * np.float32(along[0]) + dy * np.float32(along[1])) / np.float32(half[0])
    second = (-dx * np.float32(along[1]) + dy * np.float32(along[0])) / np.float32(half[1])
    radius = np.sqrt(first * first + second * second)
    return np.float32(1.0) - _smoothstep(radius, edge[0], edge[1])


def region(placement: Placement, core_share: float = CORE_SHARE) -> NDArray[np.float32]:
    """Return where the face and neck can be, over *placement*'s box.

    Shape alone, before colour: the face and the neck as soft ellipses, with
    the eyes, the brows and the mouth taken out; and, as a second plane, the
    middle of the face, which counts as skin whatever its colour. Every edge in it is soft
    over a good part of S, so it is worked out on a grid REGION_GRID times
    coarser than S and stretched to the box, which is the same to the eye
    and a small part of the work.
    """
    left, top, right, bottom = placement.box
    step = max(1.0, placement.size / REGION_GRID)
    across = max(2, int(math.ceil((right - left) / step)) + 1)
    down = max(2, int(math.ceil((bottom - top) / step)) + 1)
    xs = (left + np.arange(across, dtype=np.float32) * np.float32(step))[None, :]
    ys = (top + np.arange(down, dtype=np.float32) * np.float32(step))[:, None]
    xs, ys = np.broadcast_to(xs, (down, across)), np.broadcast_to(ys, (down, across))
    coarse = _region_at(placement, xs, ys, core_share)
    if step == 1.0:
        return np.ascontiguousarray(coarse[: bottom - top, : right - left])
    # The grid's last node lies at or past the box's far edge, so the box is
    # the part of the stretched grid from its first node on.
    stretched_width = int(round((across - 1) * step)) + 1
    stretched_height = int(round((down - 1) * step)) + 1
    stretched = cv2.resize(
        coarse, (stretched_width, stretched_height), interpolation=cv2.INTER_LINEAR
    )
    return np.ascontiguousarray(stretched[: bottom - top, : right - left])


#: How many grid nodes the region of :func:`region` has across one S.
REGION_GRID: Final[float] = 12.0


def _region_at(
    placement: Placement,
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    core_share: float = CORE_SHARE,
) -> NDArray[np.float32]:
    """Return :func:`region` at the pixel positions *xs*, *ys*."""
    size = placement.size
    up = placement.up
    across = placement.across
    centre = placement.centre

    wide = placement.face.width * size
    high = placement.face.height * size
    face = _ellipse(
        xs,
        ys,
        centre + up * high * FACE_RAISE,
        across,
        (FACE_HALF_ACROSS * wide, FACE_HALF_UP * high),
        REGION_EDGE,
    )
    neck = _ellipse(
        xs,
        ys,
        centre - up * high * NECK_LOWER,
        across,
        (NECK_HALF_ACROSS * wide, NECK_HALF_UP * high),
        REGION_EDGE,
    )
    shape = np.maximum(face, neck)

    points = placement.points
    between = max(float(np.linalg.norm(points[0] - points[1])), _EPSILON)
    eye_line = (points[1] - points[0]) / between
    if float(np.dot(eye_line, across)) < 0.0:
        eye_line = -eye_line
    taken = np.zeros_like(shape)
    for eye in (points[0], points[1]):
        taken = np.maximum(
            taken,
            _ellipse(
                xs, ys, eye, eye_line, (EYE_HALF[0] * between, EYE_HALF[1] * between),
                FEATURE_EDGE,
            ),
        )
        taken = np.maximum(
            taken,
            _ellipse(
                xs,
                ys,
                eye + up * between * BROW_UP,
                eye_line,
                (BROW_HALF[0] * between, BROW_HALF[1] * between),
                FEATURE_EDGE,
            ),
        )
    width = max(float(np.linalg.norm(points[4] - points[3])), 0.5 * between)
    mouth_line = (points[4] - points[3]) / max(float(np.linalg.norm(points[4] - points[3])), _EPSILON)
    if float(np.dot(mouth_line, across)) < 0.0:
        mouth_line = -mouth_line
    taken = np.maximum(
        taken,
        _ellipse(
            xs,
            ys,
            (points[3] + points[4]) / 2.0,
            mouth_line,
            (MOUTH_HALF[0] * width, MOUTH_HALF[1] * width),
            MOUTH_EDGE,
        ),
    )
    kept = np.float32(1.0) - taken
    # The middle of the face is skin whatever its colour: a cheek in shadow,
    # a flush or a freckle must not leave a hole that would show as a patch
    # once the skin is brightened. Colour only decides at the edges, where
    # the face meets hair and background.
    core = _ellipse(
        xs,
        ys,
        centre + up * high * FACE_RAISE,
        across,
        (FACE_HALF_ACROSS * wide * core_share, FACE_HALF_UP * high * core_share),
        REGION_EDGE,
    )
    return np.stack((shape * kept, core * kept), axis=-1).astype(np.float32)


_LN2: Final[float] = math.log(2.0)


def colour_features(image: NDArray[np.float32]) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Return red and blue against green, and the brightness, all in stops."""
    # Floored with NumPy: OpenCV would apply a single number to the first
    # channel only.
    floored = np.maximum(image, np.float32(1e-5), dtype=np.float32)
    logs = cv2.log(floored.reshape(-1, 1, 3) if floored.ndim == 2 else floored)
    logs = logs.reshape(floored.shape)
    scale = np.float32(1.0 / _LN2)
    red = (logs[..., 0] - logs[..., 1]) * scale
    blue = (logs[..., 2] - logs[..., 1]) * scale
    mean = (floored[..., 0] + floored[..., 1] + floored[..., 2]) * np.float32(1.0 / 3.0)
    brightness = cv2.log(mean) * scale
    return red, blue, brightness.reshape(mean.shape)


def likeness(
    image: NDArray[np.float32],
    tone: SkinTone,
    reach: tuple[float, float] = COLOUR_REACH,
) -> NDArray[np.float32]:
    """Return how much each pixel of *image* has the colour of *tone*.

    *reach* is how far the colour may be, as COLOUR_REACH.
    """
    red, blue, brightness = colour_features(image)
    red -= np.float32(tone.red)
    blue -= np.float32(tone.blue)
    distance = cv2.sqrt(red * red + blue * blue) * np.float32(
        1.0 / min(max(tone.spread, COLOUR_LEAST_SPREAD), COLOUR_MOST_SPREAD)
    )
    colour = np.float32(1.0) - _smoothstep(distance.reshape(red.shape), *reach)
    brightness -= np.float32(tone.brightness)
    light = (np.float32(1.0) - _smoothstep(-brightness, *DARKER_REACH)) * (
        np.float32(1.0) - _smoothstep(brightness, *BRIGHTER_REACH)
    )
    return colour * light


def face_mask(
    image: NDArray[np.float32],
    placement: Placement,
    reach: tuple[float, float] = COLOUR_REACH,
    core_share: float = CORE_SHARE,
) -> NDArray[np.float32]:
    """Return how much each pixel of *placement*'s box is this face's skin.

    *image* is the whole photo at hand, in its own linear colour; only the
    part inside the box is looked at. *reach* is how far from the cheeks'
    colour skin may be, as COLOUR_REACH, and *core_share* how much of the
    face counts as skin whatever its colour, as CORE_SHARE.
    """
    left, top, right, bottom = placement.box
    if right <= left or bottom <= top:
        return np.zeros((0, 0), dtype=np.float32)
    width, height = right - left, bottom - top
    part = image[top:bottom, left:right]
    # The mask is softened by MASK_SOFTEN of S in the end, so nothing finer
    # than that survives: a face larger than MASK_DETAIL pixels is worked
    # out on a copy shrunk to that size and stretched back. The proxy, the
    # preview and the export then all draw their mask from much the same
    # small picture, and a large face costs no more than a small one.
    factor = placement.size / MASK_DETAIL
    if factor > 1.5:
        size = (max(2, round(width / factor)), max(2, round(height / factor)))
        part = cv2.resize(part, size, interpolation=cv2.INTER_AREA)
        shapes = cv2.resize(region(placement, core_share), size, interpolation=cv2.INTER_AREA)
    else:
        factor = 1.0
        shapes = region(placement, core_share)
    sigma = MASK_SOFTEN * placement.size / factor
    like = fill_holes(
        likeness(part, placement.face.tone, reach), COLOUR_HOLE_FILL * placement.size / factor
    )
    mask = shapes[..., 0] * np.maximum(like, shapes[..., 1])
    mask = soften(mask, sigma)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    # The blur leaves a trace far beyond the face, too faint to change a
    # pixel; it is let go, so that the steps that follow can skip it.
    mask[mask < MASK_FLOOR] = 0.0
    return np.clip(mask, 0.0, 1.0)


#: Below this a mask's value changes nothing that could be seen.
MASK_FLOOR: Final[float] = 1.0 / 512.0


#: How many pixels across S the mask is worked out at, at most (see
#: :func:`face_mask`).
MASK_DETAIL: Final[float] = 80.0


#: Above this radius a blur is worked out on a smaller copy: the mask holds
#: nothing finer than it.
_SOFTEN_SHRINK_ABOVE: Final[float] = 3.0


def fill_holes(values: NDArray[np.float32], across: float) -> NDArray[np.float32]:
    """Return *values* with every dip narrower than *across* pixels filled in.

    A closing: the highest value within a disc, then the lowest within the
    same disc, which gives back every edge of anything wider than the disc.
    """
    radius = int(round(across / 2.0))
    if radius < 1:
        return values
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(values, cv2.MORPH_CLOSE, disc, borderType=cv2.BORDER_REPLICATE)


def soften(mask: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Blur *mask* by *sigma* pixels, on a smaller copy when that is wide."""
    if sigma < 0.5:
        return mask
    height, width = mask.shape[:2]
    factor = sigma / _SOFTEN_SHRINK_ABOVE
    if factor <= 1.5 or min(height, width) / factor < 8:
        return cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    small = cv2.resize(
        mask,
        (max(2, round(width / factor)), max(2, round(height / factor))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(
        small, (0, 0), _SOFTEN_SHRINK_ABOVE, borderType=cv2.BORDER_REPLICATE
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def skin_mask(
    image: NDArray[np.float32],
    placements: list[Placement],
    reach: tuple[float, float] = COLOUR_REACH,
    core_share: float = CORE_SHARE,
) -> NDArray[np.float32] | None:
    """Return the skin of every face over the whole photo at hand.

    ``None`` when no face reaches into the photo, so that nothing has to be
    carried for it. *reach* and *core_share* are as for :func:`face_mask`.
    """
    height, width = image.shape[:2]
    mask: NDArray[np.float32] | None = None
    for placement in placements:
        left, top, right, bottom = placement.box
        if right <= left or bottom <= top:
            continue
        if mask is None:
            mask = np.zeros((height, width), dtype=np.float32)
        np.maximum(
            mask[top:bottom, left:right],
            face_mask(image, placement, reach, core_share),
            out=mask[top:bottom, left:right],
        )
    return mask


def measure_tone(
    image: NDArray[np.float32], points: NDArray[np.float64]
) -> SkinTone | None:
    """Return the skin tone of the face at *points*, in pixels of *image*.

    Read from both cheeks and from the bridge of the nose, which are skin on
    nearly every face, whatever its make-up, beard or glasses. The median and
    the spread around it are used rather than the mean, so that a stray hair
    or a highlight on the cheek bone does not move them. ``None`` when too
    little of the face is in the photo to be read.
    """
    height, width = image.shape[:2]
    eyes = (points[0] + points[1]) / 2.0
    mouth = (points[3] + points[4]) / 2.0
    size = max(float(np.linalg.norm(eyes - mouth)), 0.8 * float(np.linalg.norm(points[0] - points[1])))
    radius = max(1, int(round(0.14 * size)))
    spots = []
    for eye, corner in ((points[0], points[3]), (points[1], points[4])):
        outward = eye - eyes
        outward = outward / max(float(np.linalg.norm(outward)), _EPSILON)
        spots.append(eye + 0.55 * (corner - eye) + outward * 0.18 * size)
    spots.append(eyes + 0.45 * (points[2] - eyes))
    samples = []
    for x, y in spots:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)
        if x1 > x0 and y1 > y0:
            samples.append(image[y0:y1, x0:x1].reshape(-1, 3))
    if not samples:
        return None
    pixels = np.concatenate(samples).astype(np.float32)
    # Clipped or nearly black pixels say nothing about the colour.
    usable = (pixels.max(axis=1) < 0.97) & (pixels.min(axis=1) > 1e-4)
    if int(usable.sum()) < 8:
        return None
    red, blue, brightness = colour_features(pixels[usable][None, ...])
    red, blue, brightness = red[0], blue[0], brightness[0]
    middle_red = float(np.median(red))
    middle_blue = float(np.median(blue))
    away = np.sqrt((red - middle_red) ** 2 + (blue - middle_blue) ** 2)
    spread = max(1.4826 * float(np.median(away)), COLOUR_LEAST_SPREAD)
    return SkinTone(
        red=middle_red,
        blue=middle_blue,
        spread=spread,
        brightness=float(np.median(brightness)),
    )


# --------------------------------------------------------------------------
# The skin beyond the face: neck, shoulders, chest and arms (PLAN.md 2.2,
# A-30)
# --------------------------------------------------------------------------
#
# Whitening is strong enough that skin left out beside skin brightened shows
# as a seam - a pale face over a darker neck and chest. So it reaches every
# piece of skin joined to a face: the pixels of that person's colour,
# grown out from the face and stopped wherever a clear edge of colour, or a
# step up to something brighter than the face is ever lit, lies between it
# and something else the colour of skin, which is what keeps a wall or a
# sweater the colour of skin out. The edges of shadows on the skin do not
# stop it, and gaps as narrow as a lock of hair are stepped over. It is
# worked out over the whole frame on a small copy, since it is soft anyway,
# and a zoomed in window takes its part of it.

#: The long edge of the copy the skin beyond the face is found on.
BODY_LONG_EDGE: Final[int] = 768
#: How far the colour may be from that person's cheeks, as COLOUR_REACH:
#: wider, as a shoulder or an arm is lit otherwise than the face.
BODY_COLOUR_REACH: Final[tuple[float, float]] = (2.0, 3.5)
#: Where the skin may be told apart from what joins it: an edge is a slope
#: of this many stops of brightness, or of colour, per pixel of the copy,
#: measured after a blur of BODY_EDGE_BLUR pixels so that a soft edge counts
#: as much as a sharp one and noise does not. The colour's is loose enough
#: that the edge of a shadow on skin, which turns it a little bluer, does
#: not count (measured on dappled light under trees).
BODY_EDGE_BRIGHTNESS: Final[float] = 0.08
BODY_EDGE_COLOUR: Final[float] = 0.035
BODY_EDGE_BLUR: Final[float] = 1.5
#: A shadow only ever darkens skin, so an edge of brightness counts only
#: where its bright side is brighter than the face's brightest lit skin (the
#: BODY_LIT_SHARE percentile of it) by more than BODY_LIT_MARGIN stops: that
#: is a wall or a cloth, not the same skin out of a shadow.
BODY_LIT_SHARE: Final[float] = 95.0
BODY_LIT_MARGIN: Final[float] = 0.3
#: Gaps narrower than this many S - a lock of hair over a shoulder, a
#: string, the line of a shadow - do not cut the skin beyond them off.
BODY_BRIDGE: Final[float] = 0.2
#: Within the skin found, dips in the likeness narrower than this many S -
#: a shadow a little too blue, a freckle - are filled, so that it is
#: whitened evenly rather than in patches.
BODY_HOLE_FILL: Final[float] = 0.12
#: The skin beyond the face leaves it below the chin, through the neck:
#: beside and above the face lie hair and background, so nothing may set
#: out from there. This is the head, as a share of the face's region, from
#: the chin up.
BODY_HEAD_GROWN: Final[float] = 1.5
#: The opening starts this far down the face's half height below its middle.
BODY_CHIN: Final[float] = 0.6
#: How far from a face its skin may reach, in S: across, below and above
#: the middle of the face.
BODY_REACH: Final[tuple[float, float, float]] = (6.0, 12.0, 3.0)
#: A photo whose skin has hardly any colour is black and white, or nearly:
#: colour cannot tell skin from anything else there, so only the face is
#: taken. In stops of red and blue against green, from grey.
BODY_LEAST_COLOUR: Final[float] = 0.12
#: How much of the face counts as skin whatever its colour, as CORE_SHARE:
#: more than for smoothing, as only the rim beside hair and background is
#: left to colour.
BODY_CORE_SHARE: Final[float] = 0.85
#: The mask beyond the face is softened by this many S.
BODY_SOFTEN: Final[float] = 0.04


def body_mask(
    image: NDArray[np.float32], faces: tuple[Face, ...]
) -> NDArray[np.float32] | None:
    """Return the skin joined to *faces*, over the whole frame, on a small copy.

    *image* is the whole frame in its own linear colour. The result is at
    most BODY_LONG_EDGE pixels on its long edge, to be stretched to whatever
    size it is wanted at (``Window.take``), and holds the faces' own masks
    as well. ``None`` when no face lies in the frame.
    """
    height, width = image.shape[:2]
    factor = max(1.0, max(height, width) / BODY_LONG_EDGE)
    size = (max(2, round(width / factor)), max(2, round(height / factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if factor > 1.0 else image
    placements = place(faces, size)
    # A face whitened in patches would show every one of them, where the
    # small brightening of the skin slider shows none: the face's own mask
    # here takes colours as far from the cheeks as an arm's may be, and
    # more of the face whatever its colour, which closes the holes a
    # flushed cheek, a shadow or light of two colours left in it.
    mask = skin_mask(small, placements, BODY_COLOUR_REACH, BODY_CORE_SHARE)
    if mask is None:
        return None
    coloured = [
        p for p in placements if math.hypot(p.face.tone.red, p.face.tone.blue) >= BODY_LEAST_COLOUR
    ]
    if not coloured:
        return mask

    red, blue, brightness = colour_features(small)
    # How much each pixel is the colour of any of the faces.
    like = np.zeros(red.shape, dtype=np.float32)
    for placement in coloured:
        tone = placement.face.tone
        spread = min(max(tone.spread, COLOUR_LEAST_SPREAD), COLOUR_MOST_SPREAD)
        distance = cv2.sqrt((red - tone.red) ** 2 + (blue - tone.blue) ** 2) / np.float32(spread)
        colour = np.float32(1.0) - _smoothstep(distance.reshape(red.shape), *BODY_COLOUR_REACH)
        light = brightness - np.float32(tone.brightness)
        lit = (np.float32(1.0) - _smoothstep(-light, *DARKER_REACH)) * (
            np.float32(1.0) - _smoothstep(light, *BRIGHTER_REACH)
        )
        np.maximum(like, colour * lit, out=like)

    # The brightest the faces' own skin is lit: anything brighter beyond an
    # edge is not that skin coming out of a shadow.
    lit_skin = brightness[mask > 0.5]
    lit_level = (
        float(np.percentile(lit_skin, BODY_LIT_SHARE)) if lit_skin.size else float("inf")
    )

    # Where the skin may reach: around each face, mostly below it.
    reach = np.zeros(red.shape, dtype=np.uint8)
    for placement in coloured:
        sideways, below, above = (r * placement.size for r in BODY_REACH)
        cx, cy = placement.centre
        cv2.ellipse(
            reach,
            (int(round(cx)), int(round(cy + (below - above) / 2.0))),
            (int(round(sideways)), int(round((below + above) / 2.0))),
            0.0, 0.0, 360.0, 1, -1,
        )
    # The head is closed but for an opening under the chin as wide as the
    # neck: the skin beyond the face sets out from there and only there.
    rows, columns = np.indices(red.shape, dtype=np.float32)
    closed = np.zeros(red.shape, dtype=bool)
    opening = np.zeros(red.shape, dtype=bool)
    for placement in coloured:
        wide = placement.face.width * placement.size
        high = placement.face.height * placement.size
        middle = placement.centre + placement.up * high * FACE_RAISE
        dx = columns - np.float32(middle[0])
        dy = rows - np.float32(middle[1])
        along = dx * np.float32(placement.across[0]) + dy * np.float32(placement.across[1])
        upward = dx * np.float32(placement.up[0]) + dy * np.float32(placement.up[1])
        half_across = FACE_HALF_ACROSS * wide * REGION_EDGE[1] * BODY_HEAD_GROWN
        half_up = FACE_HALF_UP * high * REGION_EDGE[1] * BODY_HEAD_GROWN
        head = (along / half_across) ** 2 + (upward / half_up) ** 2 <= 1.0
        under = (upward < -BODY_CHIN * FACE_HALF_UP * high) & (
            np.abs(along) < NECK_HALF_ACROSS * wide
        )
        closed |= head & ~under
        opening |= head & under
    ground = (like > 0.5) & (reach > 0) & ~closed

    # Edges: where brightness or colour slope steeply between pixels that
    # are both the colour of skin. Where skin meets hair or a cloth the
    # colour already tells them apart; an edge matters only between skin
    # and something else the colour of skin. So the slopes are measured on
    # a blur that takes in skin coloured pixels alone, and a lock of hair
    # across the skin leaves none behind.
    weight = (like > 0.5).astype(np.float32)
    total = cv2.GaussianBlur(weight, (0, 0), BODY_EDGE_BLUR)
    edge = np.zeros(red.shape, dtype=bool)
    too_bright = np.zeros(red.shape, dtype=bool)
    for plane, step, is_brightness in (
        (brightness, BODY_EDGE_BRIGHTNESS, True),
        (red, BODY_EDGE_COLOUR, False),
        (blue, BODY_EDGE_COLOUR, False),
    ):
        smooth = cv2.GaussianBlur(plane * weight, (0, 0), BODY_EDGE_BLUR) / np.maximum(
            total, np.float32(_EPSILON)
        )
        # OpenCV's 3x3 Sobel counts a slope eight times.
        across = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        down = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        steep = cv2.sqrt(across * across + down * down) > np.float32(8.0 * step)
        if is_brightness:
            too_bright = smooth > np.float32(lit_level + BODY_LIT_MARGIN)
            # The bright side of the edge lies within a couple of pixels.
            steep &= cv2.dilate(too_bright.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        edge |= steep
    barrier = edge & ground
    least = min(p.size for p in coloured)
    radius = max(1, int(round(BODY_BRIDGE * least / 2.0)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    bridged = cv2.morphologyEx(ground.astype(np.uint8), cv2.MORPH_CLOSE, disc) > 0
    # A gap is not bridged next to something brighter than the skin is lit,
    # or the corner between a shoulder and a skin coloured wall would be.
    bright_ground = (ground & too_bright).astype(np.uint8)
    bridged &= ground | (cv2.dilate(bright_ground, disc) == 0)
    candidate = (bridged & ~barrier & (reach > 0) & ~closed).astype(np.uint8)
    seed = (mask > 0.5) & opening & (candidate > 0)
    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    if count <= 1 or not seed.any():
        return mask
    joined = np.zeros(count, dtype=bool)
    joined[np.unique(labels[seed])] = True
    joined[0] = False
    body = joined[labels]
    # Edge pixels lying along the skin belong to it; their colour decides.
    body = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    beyond = np.where(body, fill_holes(like, BODY_HOLE_FILL * least), np.float32(0.0))
    beyond = soften(beyond.astype(np.float32), BODY_SOFTEN * least)
    return np.maximum(mask, beyond)
