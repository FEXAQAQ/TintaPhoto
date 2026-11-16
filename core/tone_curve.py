"""The luminance curve a photo carries (PLAN.md 2.2, A-34).

A curve is a handful of points on a square: how bright a tone is on the way
in, across, and how bright it is to be on the way out, up. Both run from
nought to one in sRGB encoded brightness, the scale the histogram is drawn on,
so that a point put under a peak of the histogram moves the tones of that peak.

The first point always sits at the left edge and the last at the right; they
move up and down only. Between the points the curve is a piecewise cubic whose
slopes are chosen so that it never overshoots: between two points it stays
between their two heights (F. N. Fritsch and J. Butland, "A method for
constructing local monotone piecewise cubic interpolants", 1984). A curve
drawn through rising points therefore rises everywhere, and no tone can be
pushed past the points on either side of it.

How the curve is put on a photo is the pipeline's business (PLAN.md 6 step
9.55); this module only holds the shape.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

#: One point: brightness in, brightness out.
Point = tuple[float, float]
#: The points of a curve, from the darkest tone to the brightest.
Curve = tuple[Point, ...]

#: The curve that changes nothing.
STRAIGHT: Final[Curve] = ((0.0, 0.0), (1.0, 1.0))

#: How many points a curve may hold, the two ends included. More than a
#: dozen is no longer a curve anyone shaped by hand.
MAX_POINTS: Final[int] = 16

#: The least distance across between two neighbouring points: two levels of
#: 255. Closer than that and the curve between them is all but a cliff.
MIN_GAP: Final[float] = 2.0 / 255.0


def _number(value: Any) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _unit(value: float) -> float:
    """Return *value* held between nought and one."""
    return min(max(value, 0.0), 1.0)


def is_straight(curve: Curve) -> bool:
    """Return whether *curve* changes nothing.

    Points that all lie on the diagonal draw the diagonal, however many of
    them there are.
    """
    return all(x == y for x, y in curve)


def normalised(curve: Curve) -> Curve:
    """Return *curve*, or :data:`STRAIGHT` when it draws the same line.

    Two curves that do the same thing then compare equal, which is what lets
    a photo with a point left on the diagonal still count as untouched.
    """
    return STRAIGHT if is_straight(curve) else curve


def cleaned(value: Any) -> Curve:
    """Read a curve out of stored data, whatever state it is in (PLAN.md 7.1).

    What is not a pair of numbers is skipped, every number is brought between
    nought and one, a point that does not lie at least :data:`MIN_GAP` to the
    right of the one before it is skipped, and the two ends are put on the
    edges. Fewer than two points left, or nothing but the diagonal, is the
    straight curve.
    """
    if not isinstance(value, (list, tuple)):
        return STRAIGHT
    points: list[Point] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x, y = _number(item[0]), _number(item[1])
        if x is None or y is None:
            continue
        x, y = _unit(x), _unit(y)
        if points and x < points[-1][0] + MIN_GAP:
            continue
        points.append((x, y))
        if len(points) == MAX_POINTS:
            break
    if len(points) < 2:
        return STRAIGHT
    points[0] = (0.0, points[0][1])
    points[-1] = (1.0, points[-1][1])
    # Putting the ends on the edges can bring them within reach of a
    # neighbour; the neighbour gives way.
    while len(points) > 2 and points[1][0] < MIN_GAP:
        del points[1]
    while len(points) > 2 and points[-2][0] > 1.0 - MIN_GAP:
        del points[-2]
    return normalised(tuple(points))


def can_add(curve: Curve, x: float) -> bool:
    """Return whether a point may be added to *curve* at *x*."""
    if len(curve) >= MAX_POINTS:
        return False
    return all(abs(x - px) >= MIN_GAP for px, _ in curve)


def with_point(curve: Curve, x: float, y: float) -> tuple[Curve, int]:
    """Return *curve* with a point added, and where in it the point went.

    The curve comes back as it was, with -1, when there is no room for it
    (:func:`can_add`).
    """
    x, y = _unit(x), _unit(y)
    if not can_add(curve, x):
        return curve, -1
    index = sum(1 for px, _ in curve if px < x)
    return curve[:index] + ((x, y),) + curve[index:], index


def moved(curve: Curve, index: int, x: float, y: float) -> Curve:
    """Return *curve* with one of its points taken to (*x*, *y*).

    A point stays between its two neighbours, :data:`MIN_GAP` clear of each,
    and the two ends stay on their edges: for them only *y* counts.
    """
    if not 0 <= index < len(curve):
        return curve
    last = len(curve) - 1
    if index == 0:
        x = 0.0
    elif index == last:
        x = 1.0
    else:
        x = min(max(x, curve[index - 1][0] + MIN_GAP), curve[index + 1][0] - MIN_GAP)
    return curve[:index] + ((x, _unit(y)),) + curve[index + 1 :]


def without_point(curve: Curve, index: int) -> Curve:
    """Return *curve* without one of its points; the two ends cannot go."""
    if not 0 < index < len(curve) - 1:
        return curve
    return curve[:index] + curve[index + 1 :]


def slopes(curve: Curve) -> NDArray[np.float64]:
    """Return the slope of the curve at each of its points.

    Where the curve turns - the segments either side of a point lean opposite
    ways, or one of them is flat - the slope is nought, so the point is the
    top or the bottom of the turn. Elsewhere it is the weighted harmonic mean
    of the two segments' slopes, which is never steep enough to carry the
    cubic past either neighbour.
    """
    xs = np.array([x for x, _ in curve], dtype=np.float64)
    ys = np.array([y for _, y in curve], dtype=np.float64)
    widths = np.diff(xs)
    secants = np.diff(ys) / widths
    count = len(curve)
    result = np.zeros(count, dtype=np.float64)
    if count == 2:
        result[:] = secants[0]
        return result
    for k in range(1, count - 1):
        before, after = secants[k - 1], secants[k]
        if before * after <= 0.0:
            continue
        w_before = 2.0 * widths[k] + widths[k - 1]
        w_after = widths[k] + 2.0 * widths[k - 1]
        result[k] = (w_before + w_after) / (w_before / before + w_after / after)
    result[0] = _end_slope(widths[0], widths[1], secants[0], secants[1])
    result[-1] = _end_slope(widths[-1], widths[-2], secants[-1], secants[-2])
    return result


def _end_slope(width: float, next_width: float, secant: float, next_secant: float) -> float:
    """Return the slope at an end of the curve, from its two nearest segments.

    The three point estimate, held back where it would lean against its own
    segment or more than three times as steeply as it.
    """
    slope = ((2.0 * width + next_width) * secant - width * next_secant) / (width + next_width)
    if slope * secant <= 0.0:
        return 0.0
    if secant * next_secant <= 0.0 and abs(slope) > 3.0 * abs(secant):
        return 3.0 * secant
    return float(slope)


def evaluate(curve: Curve, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Read *curve* at every one of *values*.

    Between nought and one the result is the cubic through the points, held
    between nought and one. Past one the curve carries straight on at the
    slope it reached the last point with - never downhill, so that what was
    brighter than white is not turned darker than it - and below nought it
    stays at the height of the first point.
    """
    at = np.asarray(values, dtype=np.float64)
    xs = np.array([x for x, _ in curve], dtype=np.float64)
    ys = np.array([y for _, y in curve], dtype=np.float64)
    tangents = slopes(curve)

    inside = np.clip(at, 0.0, 1.0)
    segment = np.clip(np.searchsorted(xs, inside, side="right") - 1, 0, len(curve) - 2)
    width = xs[segment + 1] - xs[segment]
    t = (inside - xs[segment]) / width
    t2 = t * t
    t3 = t2 * t
    result = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * ys[segment]
        + (t3 - 2.0 * t2 + t) * width * tangents[segment]
        + (-2.0 * t3 + 3.0 * t2) * ys[segment + 1]
        + (t3 - t2) * width * tangents[segment + 1]
    )
    result = np.clip(result, 0.0, 1.0)
    beyond = at > 1.0
    if np.any(beyond):
        result = np.where(beyond, ys[-1] + max(tangents[-1], 0.0) * (at - 1.0), result)
    return result
