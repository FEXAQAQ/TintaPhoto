"""Measure a film look from a photograph a camera developed twice.

A camera that writes a RAW file and a JPEG at the same moment has told us
everything we need: the RAW developed by this program is the starting point,
the JPEG beside it is where the camera's film simulation arrives, and the two
are the same exposure of the same frame, pixel for pixel. Fitting a look to
that pair asks no questions about white balance or metering, because neither
differs between them.

The look is the one in :mod:`core.filters`: three tone curves and three
Fourier series over hue. They are solved by turns - curves against the
target with the hue functions taken back out, then hue functions against
what the curves leave - which converges in a handful of rounds.

Run it with no arguments to fit every look under ``sample/filter/pairs``
and write the result into ``resources/filters``::

    .venv\\Scripts\\python.exe tools\\fit_filter.py

Each folder there is one look. A folder may hold one pair or several: with
several, every scene's samples go into the one fit, which is how a look is
measured over more of the colour circle than a single photograph holds, and
each scene is also left out of the fit in turn and measured against the look
the others made - the honest answer to how it does on a photograph it has
not seen. ``--category`` says which group of the looks tab a new look goes
under; a look already in ``resources/filters`` keeps the one it has.
``--chroma-scale`` makes the written look that much more colourful than the
measurement, for a camera whose rendering is truer than it is pleasing; the
errors reported are those of the look as measured.

``--exposure`` develops every scene at that many stops before measuring. A
camera that renders the same RAW a third of a stop darker than this program
does was metered for its own tone curve, not given a different colour, and
a look measured without the offset would darken every photograph it is put
on. Fit once without it, read the brightness change off the curves the
report prints, and fit again with it; every scene takes the same offset,
because a camera's calibration is the same for every scene, and matching
each scene on its own was measured to fit unseen scenes far worse.

The report, and a picture of what was fitted beside what was aimed at, go to
``sample/filter/_fit``, which is outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import color_management as cm  # noqa: E402
from core import decoder, pipeline  # noqa: E402
from core.decoder import ColorSpace  # noqa: E402
from core.edit_state import EditState  # noqa: E402
from core.filters import (  # noqa: E402
    CATEGORIES,
    DEFAULT_CATEGORY,
    MAX_HUE_TURN,
    MAX_LOG_SCALE,
    Look,
    curve,
    fourier,
    load,
    save,
)

#: RAW formats this tool reads although the program itself does not open
#: them: medium format bodies whose sample files a look can be measured
#: from. LibRaw reads them; adding them here, for this run only, is what
#: lets the decoder take the RAW path for them.
EXTRA_RAW_EXTENSIONS: Final[frozenset[str]] = frozenset({".fff", ".3fr"})
decoder.RAW_EXTENSIONS = decoder.RAW_EXTENSIONS | EXTRA_RAW_EXTENSIONS
decoder.SUPPORTED_EXTENSIONS = decoder.SUPPORTED_EXTENSIONS | EXTRA_RAW_EXTENSIONS

#: The long edge both photographs are brought to before anything is measured.
#: Well under the proxy, because a camera's sharpening, its noise reduction
#: and its demosaic are not part of a look, and shrinking is what removes
#: them from the comparison.
MEASURE_LONG_EDGE: Final[int] = 768

#: Knots in each tone curve, and harmonics in each Fourier series. Three
#: harmonics rather than four: the fourth was measured to buy no accuracy at
#: all - a hundredth of a unit of colour error - and a look that wanders
#: between the hues one scene happened to hold is the thing to avoid.
CURVE_KNOTS: Final[int] = 17
HARMONICS: Final[int] = 3

#: With ``--neutral-curves``, the Oklab chroma at which a sample's say in the
#: tone curves has fallen to a half. The three curves are what a grey goes
#: through, so they are best measured on greys; left to every pixel alike, a
#: set of scenes full of autumn leaves pulled the blue curve down a third of
#: a stop to make the leaves yellower - and every grey yellow with them. The
#: colours themselves are the hue functions' to describe.
NEUTRAL_CHROMA: Final[float] = 0.03

#: How many times the curves and the hue functions are solved by turns.
ROUNDS: Final[int] = 4

#: How much the fit is pulled towards a smooth curve and towards no curve at
#: all, as fractions of the weight the samples carry.
CURVE_SMOOTHNESS: Final[float] = 2e-3
CURVE_IDENTITY: Final[float] = 2e-4
#: How firmly the top of each curve is held at white. A channel at the top
#: of this program's range is clipped, and so is the camera's: both write
#: white there. The samples cannot say so themselves, because clipped pixels
#: are left out of them, and a set of scenes with little near white leaves
#: the top knot to be guessed from the midtones - which once put paper white
#: at a cream grey of 0.89.
CURVE_WHITE: Final[float] = 5e-2
#: The same for the hue functions, where the penalty grows with the harmonic
#: so that the fine detail of a single scene cannot run away with the fit.
SERIES_RIDGE: Final[float] = 4e-3

#: A hue the scene did not hold has nothing to say about the look, and a
#: Fourier series left to itself there will swing wherever fitting the rest
#: of the circle happens to send it. So the whole circle is given a thin
#: layer of "nothing changes here", spread evenly: against a hue the scene
#: is full of it counts for nothing, and in a hue it never held it is the
#: only thing there. This is the fraction of the samples' weight it carries.
#:
#: Half the samples' weight sounds heavy and is not: measured over the three
#: pairs that were fitted, going from a fiftieth to a half costs nine thousandths
#: a unit of colour error where the scene has colour, and takes the furthest
#: hue turn from 36 degrees down to 18. Almost all of what it removes was in
#: hues the scene never held - and a photograph of skin or sky will be made
#: of exactly those.
CIRCLE_PRIOR: Final[float] = 0.5
#: How many places around the circle that prior is applied at.
CIRCLE_PRIOR_POINTS: Final[int] = 72

#: A pixel is too dark or too bright to say anything about a look once a
#: channel is against either end, where the camera's own clipping decides
#: the colour.
SAFE_LOW: Final[float] = 0.02
SAFE_HIGH: Final[float] = 0.98

#: Chroma below which a pixel is treated as neutral and left out of the hue
#: functions, in Oklab.
CHROMA_FLOOR: Final[float] = 0.015

#: Edges are where a pixel's colour depends on exactly how well the two
#: frames line up, so the flattest part of the frame is what gets measured.
GRADIENT_PERCENTILE: Final[float] = 55.0

#: How much of the frame is used, as a fraction of its width and height.
#: Lens shading and distortion are corrected by the camera and not by
#: LibRaw, and both are at their worst in the corners.
CENTRAL_FRACTION: Final[float] = 0.86


# --------------------------------------------------------------------------
# Reading a pair
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One film simulation: a RAW file and the JPEG the camera made of it."""

    identifier: str
    raw_path: Path
    jpeg_path: Path


def find_pairs(root: Path) -> list[Pair]:
    """Return every ``<folder>/<name>.RAF`` with its ``<name>.JPG`` beside it."""
    pairs: list[Pair] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for raw_path in sorted(folder.iterdir()):
            if not decoder.is_raw(raw_path):
                continue
            jpeg_path = raw_path.with_suffix(".JPG")
            if not jpeg_path.exists():
                jpeg_path = raw_path.with_suffix(".jpg")
            if jpeg_path.exists():
                pairs.append(Pair(folder.name.lower(), raw_path, jpeg_path))
    return pairs


#: A look that changes nothing, worn while the source is developed. The
#: pipeline hands a worn look a RAW photo without the base chroma of step 10
#: (``pipeline.base_chroma``), so that is the photo a look has to be measured
#: from: measured from the plain rendering, which has the base chroma, a look
#: put on in the program came out a quarter less colourful than it was fitted.
PLAIN_LOOK: Final[Look] = Look.identity("plain")


def as_a_look_sees_it(photo: decoder.DecodedPhoto, stops: float = 0.0) -> EditState:
    """Return *photo*'s resting state, wearing a look that changes nothing."""
    state = EditState.for_photo(photo.is_raw, *pipeline.shot_white_balance(photo))
    state = state.with_filter(PLAIN_LOOK.identifier)
    return state.with_value("exposure", stops) if stops else state


def our_rendering(path: Path, long_edge: int, stops: float = 0.0) -> NDArray[np.float32]:
    """Return what this program makes of *path* with every slider at rest.

    Every slider but the exposure, which is *stops*. Sixteen bits rather than
    eight: the look is measured from the difference between two pictures,
    and a difference of a step or two matters when the whole of it is only a
    few steps wide.
    """
    photo = decoder.decode_proxy(path, long_edge)
    rendered = pipeline.render(
        photo, as_a_look_sees_it(photo, stops), bits=pipeline.SIXTEEN_BIT, look=PLAIN_LOOK
    )
    return (rendered / 65535.0).astype(np.float32)


def camera_rendering(path: Path, long_edge: int) -> NDArray[np.float32]:
    """Return the camera's own JPEG, upright, as encoded sRGB.

    An sRGB file is taken exactly as it is, untouched by anything here. A
    file in a wider space - a body set to Adobe RGB, say - goes through the
    working space the way the program itself takes it, and the few colours
    that sRGB cannot hold come out beyond nought and one rather than being
    flattened against them; the sampling throws those away.
    """
    photo = decoder.decode_proxy(path, long_edge)
    if photo.color_space in (None, ColorSpace.SRGB) and photo.encoded is not None:
        return (photo.encoded.astype(np.float32) / 255.0).astype(np.float32)
    print(f"  {path.name} is {photo.color_space}, converting it to sRGB")
    return cm.linear_to_srgb(cm.apply_matrix(photo.image, cm.PROPHOTO_TO_SRGB))


def _grey(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a blurred single channel of *image*, for lining two frames up."""
    grey = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, (0, 0), 1.5)


def line_up(
    source: NDArray[np.float32], target: NDArray[np.float32]
) -> tuple[NDArray[np.float32], str]:
    """Return *target* warped onto *source*, and a line saying how far it moved.

    LibRaw hands back a little more of the sensor than the camera puts in its
    JPEG, so the two frames of one exposure are a fraction of a percent apart
    in scale. Left alone that would be read as colour at every edge in the
    picture.
    """
    if target.shape != source.shape:
        target = cv2.resize(
            target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA
        )
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            _grey(source),
            _grey(target),
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
            None,
            3,
        )
    except cv2.error:
        return target, "could not be lined up; left as it was"
    warped = cv2.warpAffine(
        target,
        warp,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    moved = float(np.hypot(warp[0, 2], warp[1, 2]))
    scale = float((warp[0, 0] + warp[1, 1]) / 2.0)
    return warped.astype(np.float32), f"moved {moved:.1f} px, scaled {scale:.4f}"


def shrink(image: NDArray[np.float32], long_edge: int) -> NDArray[np.float32]:
    """Return *image* at *long_edge*, averaging rather than sampling."""
    height, width = image.shape[:2]
    if max(width, height) <= long_edge:
        return image
    scale = long_edge / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def usable(source: NDArray[np.float32], target: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Return the pixels worth measuring: flat, in range and away from the edge."""
    height, width = source.shape[:2]
    inside = np.zeros((height, width), dtype=bool)
    margin_y = int(height * (1.0 - CENTRAL_FRACTION) / 2.0)
    margin_x = int(width * (1.0 - CENTRAL_FRACTION) / 2.0)
    inside[margin_y : height - margin_y, margin_x : width - margin_x] = True

    in_range = (
        (source > SAFE_LOW).all(axis=2)
        & (source < SAFE_HIGH).all(axis=2)
        & (target > SAFE_LOW).all(axis=2)
        & (target < SAFE_HIGH).all(axis=2)
    )

    grey = cv2.cvtColor(np.ascontiguousarray(source), cv2.COLOR_RGB2GRAY)
    gradient = np.hypot(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3),
    )
    flat = gradient <= np.percentile(gradient, GRADIENT_PERCENTILE)
    return inside & in_range & flat


# --------------------------------------------------------------------------
# The two halves of the fit
# --------------------------------------------------------------------------


def to_oklab(encoded: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert encoded sRGB to Oklab, the space the error is measured in."""
    return cm.prophoto_to_oklab(
        cm.apply_matrix(cm.srgb_to_linear(encoded), cm.SRGB_TO_PROPHOTO)
    )


def from_oklab(lab: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Oklab back to encoded sRGB."""
    return cm.linear_to_srgb(cm.apply_matrix(cm.oklab_to_prophoto(lab), cm.PROPHOTO_TO_SRGB))


def even_weight(values: NDArray[np.float32], bins: int, low: float, high: float) -> NDArray[np.float32]:
    """Return weights that give every part of *values* a similar say.

    A photograph is mostly shadow and mostly one or two hues; without this
    the fit would describe the wall behind the subject and guess at the rest.
    The square root is the compromise: it evens the distribution out without
    handing a handful of rare pixels the whole curve.
    """
    index = np.clip(((values - low) / (high - low) * bins).astype(np.int32), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float32)
    counts[counts == 0] = 1.0
    weight = 1.0 / np.sqrt(counts[index])
    return (weight / weight.mean()).astype(np.float32)


def fit_curve(
    source: NDArray[np.float32],
    target: NDArray[np.float32],
    weight: NDArray[np.float32],
    knots: int,
) -> NDArray[np.float64]:
    """Solve one channel's tone curve by weighted least squares.

    The curve is linear in its knot values, so the whole of it is one small
    normal equation however many million pixels went in. Two penalties keep
    it honest where the photograph is thin: one on the second difference, so
    it stays a curve rather than a comb, and a weak one towards the straight
    line, so that a brightness the scene never held comes back unchanged
    instead of wandering.
    """
    straight = np.linspace(0.0, 1.0, knots)
    position = np.clip(source, 0.0, 1.0) * (knots - 1)
    low = np.clip(position.astype(np.int32), 0, knots - 2)
    fraction = (position - low).astype(np.float64)

    normal = np.zeros((knots, knots), dtype=np.float64)
    right = np.zeros(knots, dtype=np.float64)
    share = np.stack((1.0 - fraction, fraction), axis=1)
    index = np.stack((low, low + 1), axis=1)
    for first in range(2):
        for second in range(2):
            np.add.at(
                normal,
                (index[:, first], index[:, second]),
                weight * share[:, first] * share[:, second],
            )
        np.add.at(right, index[:, first], weight * share[:, first] * target)

    total = float(weight.sum())
    second_difference = np.zeros((knots - 2, knots), dtype=np.float64)
    for row in range(knots - 2):
        second_difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    normal += CURVE_SMOOTHNESS * total * (second_difference.T @ second_difference)
    normal += CURVE_IDENTITY * total * np.eye(knots)
    right += CURVE_IDENTITY * total * straight
    normal[-1, -1] += CURVE_WHITE * total
    right[-1] += CURVE_WHITE * total

    values = np.linalg.solve(normal, right)
    # A tone curve that turns back on itself would posterise; nothing in the
    # data asks for it, so a fit that strays there is simply flattened.
    rise = np.maximum(np.diff(values), 0.0)
    return np.concatenate(([values[0]], values[0] + np.cumsum(rise)))


def run_curves(
    source: NDArray[np.float32], curves: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Apply three fitted curves to encoded samples."""
    out = np.empty_like(source)
    for channel in range(3):
        points = np.asarray(curves[channel], dtype=np.float32)
        out[:, channel] = curve(points, source[:, channel])
    return out


def basis(hue: NDArray[np.float32], harmonics: int) -> NDArray[np.float64]:
    """Return the Fourier design matrix: a constant and each harmonic's pair."""
    columns = [np.ones_like(hue, dtype=np.float64)]
    for order in range(1, harmonics + 1):
        columns.append(np.cos(hue * order).astype(np.float64))
        columns.append(np.sin(hue * order).astype(np.float64))
    return np.stack(columns, axis=1)


def fit_series(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    weight: NDArray[np.float32],
    harmonics: int,
) -> NDArray[np.float64]:
    """Solve one Fourier series, penalising the finer harmonics more.

    Two things hold the series down. The ridge keeps it from reaching for
    detail that is not there, more firmly the finer the harmonic; the even
    layer around the circle keeps it at no change wherever the scene held no
    colour of that hue. That layer asks for zero, so it adds nothing to the
    right hand side - only the certainty that the answer there should be
    nothing.
    """
    penalty = [0.0]
    for order in range(1, harmonics + 1):
        penalty.extend([float(order * order), float(order * order)])
    total = float(weight.sum())
    weighted = design * weight[:, None]
    normal = design.T @ weighted + SERIES_RIDGE * total * np.diag(penalty)

    circle = basis(
        np.linspace(-np.pi, np.pi, CIRCLE_PRIOR_POINTS, endpoint=False, dtype=np.float32),
        harmonics,
    )
    normal += (CIRCLE_PRIOR * total / CIRCLE_PRIOR_POINTS) * (circle.T @ circle)
    return np.linalg.solve(normal, weighted.T @ target)


def fit_hue_functions(
    middle: NDArray[np.float32], target: NDArray[np.float32], harmonics: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve how colourful, which way round and how light each hue comes out."""
    lab_middle = to_oklab(middle)
    lab_target = to_oklab(target)
    chroma_middle = np.hypot(lab_middle[:, 1], lab_middle[:, 2])
    chroma_target = np.hypot(lab_target[:, 1], lab_target[:, 2])
    hue_middle = np.arctan2(lab_middle[:, 2], lab_middle[:, 1])
    hue_target = np.arctan2(lab_target[:, 2], lab_target[:, 1])

    coloured = (chroma_middle > CHROMA_FLOOR) & (chroma_target > CHROMA_FLOOR)
    if coloured.sum() < 64:
        flat = np.zeros(2 * harmonics + 1)
        return flat, flat.copy(), flat.copy()

    hue = hue_middle[coloured]
    design = basis(hue, harmonics)
    # Every hue gets a similar say, and a colour speaks louder than one that
    # is nearly grey, where an angle means very little.
    weight = even_weight(hue, 24, -np.pi, np.pi) * np.minimum(chroma_middle[coloured], 0.2)

    chroma = fit_series(
        design,
        np.log(chroma_target[coloured] / chroma_middle[coloured]).astype(np.float64),
        weight,
        harmonics,
    )
    turn = (hue_target[coloured] - hue) + np.pi
    turn = np.mod(turn, 2.0 * np.pi) - np.pi
    rotation = fit_series(design, turn.astype(np.float64), weight, harmonics)

    fade = np.clip(chroma_middle[coloured] / np.float32(0.10), 0.0, 1.0)
    lightness = fit_series(
        design * fade[:, None].astype(np.float64),
        np.log(
            np.maximum(lab_target[coloured, 0], 1e-4) / np.maximum(lab_middle[coloured, 0], 1e-4)
        ).astype(np.float64),
        weight,
        harmonics,
    )
    return chroma, rotation, lightness


def undo_hue_functions(
    target: NDArray[np.float32], hue: NDArray[np.float32], look: Look
) -> NDArray[np.float32]:
    """Return *target* with the look's hue functions taken back out.

    The curves and the hue functions are solved by turns, and each has to be
    solved against what the other leaves. *hue* is the hue the look looked
    up when it went the other way, so that both directions read the same
    place on the three series.
    """
    lab = to_oklab(target).copy()
    turn = -np.clip(
        fourier(np.asarray(look.hue, dtype=np.float32), hue), -0.35, 0.35
    )
    scale = 1.0 / np.exp(
        np.clip(fourier(np.asarray(look.chroma, dtype=np.float32), hue), -0.7, 0.7)
    )
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    fade = np.clip(chroma / np.float32(0.10), 0.0, 1.0)
    lift = 1.0 / np.exp(
        np.clip(fourier(np.asarray(look.lightness, dtype=np.float32), hue) * fade, -0.7, 0.7)
    )
    cosine = np.cos(turn)
    sine = np.sin(turn)
    a = (lab[:, 1] * cosine - lab[:, 2] * sine) * scale
    b = (lab[:, 1] * sine + lab[:, 2] * cosine) * scale
    return from_oklab(
        np.stack((lab[:, 0] * lift, a, b), axis=1).astype(np.float32)
    )


# --------------------------------------------------------------------------
# Error
# --------------------------------------------------------------------------


def difference(left: NDArray[np.float32], right: NDArray[np.float32]) -> dict[str, float]:
    """Return how far apart two sets of encoded colours are.

    In Oklab, times a hundred, so that one unit is near enough the smallest
    difference an eye can find and the numbers read like the colour error
    everyone else quotes. The eighth bit of an sRGB step is about 0.3.
    """
    distance = np.linalg.norm(to_oklab(left) - to_oklab(right), axis=-1) * 100.0
    return {
        "mean": float(distance.mean()),
        "p95": float(np.percentile(distance, 95)),
        "max": float(distance.max()),
    }


def hue_coverage(source: NDArray[np.float32], bins: int = 12) -> list[float]:
    """Return the share of the coloured samples each slice of hue holds.

    A look is only measured where the scene held colour; everywhere else it
    is the prior speaking. This is what says which is which, and it is kept
    with the look so that the answer is still there months later.
    """
    lab = to_oklab(source)
    coloured = np.hypot(lab[:, 1], lab[:, 2]) > CHROMA_FLOOR
    if not coloured.any():
        return [0.0] * bins
    hue = np.arctan2(lab[coloured, 2], lab[coloured, 1])
    index = np.clip(((hue + np.pi) / (2.0 * np.pi) * bins).astype(np.int32), 0, bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    return [round(float(share), 4) for share in counts / counts.sum()]


def reach(look: Look) -> dict[str, float]:
    """Return how far round the circle each hue function goes at its furthest."""
    hue = np.linspace(-np.pi, np.pi, 721, dtype=np.float32)
    return {
        "chroma": round(float(np.abs(fourier(np.asarray(look.chroma, np.float32), hue)).max()), 3),
        "turn_degrees": round(
            float(np.degrees(np.abs(fourier(np.asarray(look.hue, np.float32), hue)).max())), 1
        ),
        "lightness": round(
            float(np.abs(fourier(np.asarray(look.lightness, np.float32), hue)).max()), 3
        ),
    }


def apply_look(look: Look, encoded: NDArray[np.float32]) -> NDArray[np.float32]:
    """Run *encoded* samples through *look*, the way the pipeline will."""
    working = cm.apply_matrix(cm.srgb_to_linear(encoded), cm.SRGB_TO_PROPHOTO)
    return cm.linear_to_srgb(cm.apply_matrix(look.apply(working), cm.PROPHOTO_TO_SRGB))


# --------------------------------------------------------------------------
# Fitting one pair
# --------------------------------------------------------------------------


def fit(
    identifier: str,
    source: NDArray[np.float32],
    target: NDArray[np.float32],
    knots: int = CURVE_KNOTS,
    harmonics: int = HARMONICS,
    rounds: int = ROUNDS,
    neutral_curves: bool = False,
) -> tuple[Look, list[dict[str, float]]]:
    """Fit a look to flat arrays of matching source and target colours.

    With *neutral_curves* the tone curves listen mostly to the near greys,
    see ``NEUTRAL_CHROMA``.
    """
    brightness = cm.channel_max(source)
    weight = even_weight(brightness, 24, 0.0, 1.0)
    if neutral_curves:
        lab = to_oklab(source)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        weight = (weight / (1.0 + (chroma / NEUTRAL_CHROMA) ** 2)).astype(np.float32)
        weight = (weight / weight.mean()).astype(np.float32)
    look = Look.identity(identifier, knots)
    history: list[dict[str, float]] = [difference(source, target)]

    for round_number in range(rounds):
        if round_number == 0:
            curve_target = target
        else:
            middle = run_curves(source, np.asarray(look.curves, dtype=np.float64))
            lab = to_oklab(middle)
            curve_target = undo_hue_functions(
                target, np.arctan2(lab[:, 2], lab[:, 1]).astype(np.float32), look
            )
        curves = np.stack(
            [
                fit_curve(source[:, channel], curve_target[:, channel], weight, knots)
                for channel in range(3)
            ]
        )
        middle = run_curves(source, curves)
        chroma, rotation, lightness = fit_hue_functions(middle, target, harmonics)
        look = Look(
            identifier=identifier,
            curves=(
                tuple(curves[0].tolist()),
                tuple(curves[1].tolist()),
                tuple(curves[2].tolist()),
            ),
            chroma=tuple(chroma.tolist()),
            hue=tuple(rotation.tolist()),
            lightness=tuple(lightness.tolist()),
        )
        history.append(difference(apply_look(look, source), target))
    return look, history


# --------------------------------------------------------------------------
# Pictures of the result
# --------------------------------------------------------------------------


def label(image: NDArray[np.uint8], text: str) -> None:
    """Write *text* into the top left of *image*, in place."""
    cv2.putText(image, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
    cv2.putText(image, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)


def comparison(
    source: NDArray[np.float32],
    fitted: NDArray[np.float32],
    target: NDArray[np.float32],
    path: Path,
) -> None:
    """Write the three pictures side by side: ours, ours with the look, theirs."""
    panels = []
    for image, text in (
        (source, "TintaPhoto"),
        (fitted, "TintaPhoto + filter"),
        (target, "camera JPEG"),
    ):
        panel = np.ascontiguousarray(
            (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        )
        label(panel, text)
        panels.append(panel)
    strip = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def curve_plot(look: Look, path: Path, size: int = 480) -> None:
    """Draw the look's three tone curves and its three functions of hue."""
    canvas = np.full((size, size * 2, 3), 24, dtype=np.uint8)
    for grid in range(1, 4):
        position = size * grid // 4
        cv2.line(canvas, (position, 0), (position, size), (48, 48, 48), 1)
        cv2.line(canvas, (0, position), (size, position), (48, 48, 48), 1)
    steps = np.linspace(0.0, 1.0, size, dtype=np.float32)
    colours = ((230, 90, 90), (90, 220, 120), (110, 140, 240))
    for channel in range(3):
        points = np.asarray(look.curves[channel], dtype=np.float32)
        values = curve(points, steps)
        for x in range(size - 1):
            cv2.line(
                canvas,
                (x, size - 1 - int(np.clip(values[x], 0, 1) * (size - 1))),
                (x + 1, size - 1 - int(np.clip(values[x + 1], 0, 1) * (size - 1))),
                colours[channel],
                2,
            )
    hue = np.linspace(-np.pi, np.pi, size, dtype=np.float32)
    series = (
        (np.exp(fourier(np.asarray(look.chroma, dtype=np.float32), hue)) - 1.0, (230, 90, 90)),
        (fourier(np.asarray(look.hue, dtype=np.float32), hue), (90, 220, 120)),
        (np.exp(fourier(np.asarray(look.lightness, dtype=np.float32), hue)) - 1.0, (110, 140, 240)),
    )
    middle = size // 2
    cv2.line(canvas, (size, middle), (size * 2, middle), (64, 64, 64), 1)
    for values, colour in series:
        scaled = middle - values * (size / 1.2)
        for x in range(size - 1):
            cv2.line(
                canvas,
                (size + x, int(np.clip(scaled[x], 0, size - 1))),
                (size + x + 1, int(np.clip(scaled[x + 1], 0, size - 1))),
                colour,
                2,
            )
    label(canvas, "curves R G B        |        hue: chroma, turn, lightness")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------
# Checking the look on another camera
# --------------------------------------------------------------------------
#
# Fitting a look from a camera's own RAW and JPEG says how well the model can
# describe that camera. It does not say what the look will do to a photograph
# from a different camera, which is the whole point of having it. That is what
# the second camera's frame of the same scene is for: a photograph from one
# body put through the look, measured against the other body's own JPEG of
# the same table.
#
# The two frames are not the same picture - different sensor, lens, framing,
# depth of field and exposure - so they have to be lined up by what is in
# them, and the exposure has to be settled before colour can be compared at
# all. Whatever error is left after that is the honest answer to "will this
# look right on my own photographs".


#: Exposures tried when matching the two cameras' brightness, in stops.
CHECK_EXPOSURES: Final[tuple[float, ...]] = tuple(
    round(value, 2) for value in np.arange(-1.5, 1.51, 0.25)
)


def line_up_scene(
    source: NDArray[np.float32], target: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.bool_], str]:
    """Warp one camera's frame of a scene onto another's, by what is in it."""
    source_grey = (np.clip(_grey(source), 0, 1) * 255).astype(np.uint8)
    target_grey = (np.clip(_grey(target), 0, 1) * 255).astype(np.uint8)
    detector = cv2.SIFT_create()
    source_points, source_description = detector.detectAndCompute(source_grey, None)
    target_points, target_description = detector.detectAndCompute(target_grey, None)
    if source_description is None or target_description is None:
        raise ValueError("neither frame had any features to match")

    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(target_description, source_description, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 16:
        raise ValueError(f"only {len(good)} features matched between the two frames")

    from_points = np.float32([target_points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    to_points = np.float32([source_points[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(from_points, to_points, cv2.RANSAC, 3.0)
    if homography is None:
        raise ValueError("the two frames could not be lined up")

    size = (source.shape[1], source.shape[0])
    warped = cv2.warpPerspective(target, homography, size, flags=cv2.INTER_LINEAR)
    covered = cv2.warpPerspective(
        np.ones(target.shape[:2], dtype=np.uint8), homography, size, flags=cv2.INTER_NEAREST
    )
    # The edge of the warped frame is half one picture and half nothing, so
    # it is pulled in before anything is measured.
    covered = cv2.erode(covered, np.ones((9, 9), np.uint8))
    note = f"{int(inliers.sum())} of {len(good)} features held"
    return warped.astype(np.float32), covered.astype(bool), note


def best_exposure(
    renderings: dict[float, NDArray[np.float32]],
    target: NDArray[np.float32],
    mask: NDArray[np.bool_],
    look: Look | None,
) -> tuple[float, dict[str, float]]:
    """Return the exposure that brings one camera closest to the other.

    A photographer sets the exposure and then chooses a look; comparing the
    two cameras at whatever each metered would measure the difference between
    two light meters, not between two renderings of colour.
    """
    best: tuple[float, dict[str, float]] | None = None
    for stops, rendering in renderings.items():
        samples = rendering[mask]
        if look is not None:
            samples = apply_look(look, samples)
        score = difference(samples, target[mask])
        if best is None or score["mean"] < best[1]["mean"]:
            best = (stops, score)
    assert best is not None  # renderings is never empty
    return best


def cross_camera_check(
    raw_path: Path,
    jpeg_path: Path,
    pairs: list[Pair],
    looks: dict[str, Look],
    report_dir: Path,
    long_edge: int,
) -> list[dict[str, object]]:
    """Measure each look on a second camera's frame of the same scene."""
    photo = decoder.decode_proxy(raw_path, decoder.PROXY_LONG_EDGE)
    shot = pipeline.shot_white_balance(photo)
    # Two sets: the plain photo, as it looks with no filter, and the photo a
    # look is handed, which has no base chroma (see ``PLAIN_LOOK``).
    renderings: dict[float, NDArray[np.float32]] = {}
    look_inputs: dict[float, NDArray[np.float32]] = {}
    full_size: NDArray[np.float32] | None = None
    for stops in CHECK_EXPOSURES:
        state = EditState.for_photo(photo.is_raw, *shot).with_value("exposure", stops)
        rendered = (
            pipeline.render(photo, state, bits=pipeline.SIXTEEN_BIT) / 65535.0
        ).astype(np.float32)
        if stops == 0.0:
            full_size = rendered
        renderings[stops] = shrink(rendered, long_edge)
        handed = pipeline.render(
            photo, as_a_look_sees_it(photo, stops), bits=pipeline.SIXTEEN_BIT, look=PLAIN_LOOK
        )
        look_inputs[stops] = shrink((handed / 65535.0).astype(np.float32), long_edge)
    assert full_size is not None  # nought is one of the exposures tried
    own_jpeg, _ = line_up(full_size, camera_rendering(jpeg_path, decoder.PROXY_LONG_EDGE))
    own_jpeg = shrink(own_jpeg, long_edge)

    results: list[dict[str, object]] = []
    for pair in pairs:
        look = looks[pair.identifier]
        target = shrink(camera_rendering(pair.jpeg_path, decoder.PROXY_LONG_EDGE), long_edge)
        try:
            warped, covered, note = line_up_scene(renderings[0.0], target)
        except (ValueError, cv2.error) as error:
            results.append({"id": pair.identifier, "error": str(error)})
            continue
        mask = usable(renderings[0.0], warped) & covered
        plain_stops, plain = best_exposure(renderings, warped, mask, None)
        look_stops, looked = best_exposure(look_inputs, warped, mask, look)
        results.append(
            {
                "id": pair.identifier,
                "alignment": note,
                "samples": int(mask.sum()),
                "without_filter": {"exposure": plain_stops, **plain},
                "with_filter": {"exposure": look_stops, **looked},
            }
        )
        comparison(
            renderings[look_stops],
            apply_look(look, look_inputs[look_stops]),
            warped,
            report_dir / f"{pair.identifier}-on-the-other-camera.jpg",
        )

    own_mask = usable(renderings[0.0], own_jpeg)
    own_stops, own = best_exposure(renderings, own_jpeg, own_mask, None)
    results.append(
        {
            "id": "our rendering against this camera's own JPEG",
            "samples": int(own_mask.sum()),
            "without_filter": {"exposure": own_stops, **own},
        }
    )
    return results


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """One pair brought to the measuring size, with the pixels worth using."""

    pair: Pair
    source_image: NDArray[np.float32]
    target_image: NDArray[np.float32]
    alignment: str
    source: NDArray[np.float32]
    target: NDArray[np.float32]


#: How many times the shading is measured against a look and taken out.
SHADING_ROUNDS: Final[int] = 3

def radius_squared(height: int, width: int) -> NDArray[np.float32]:
    """Return each pixel's squared distance from the centre, in half diagonals."""
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0)[:, None]
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0)[None, :]
    return ((x * x + y * y) / ((width * width + height * height) / 4.0)).astype(np.float32)


def shade(image: NDArray[np.float32], gain: tuple[float, float]) -> NDArray[np.float32]:
    """Return encoded *image* with a radial gain applied in linear light."""
    r2 = radius_squared(*image.shape[:2])
    factor = np.exp(gain[0] * r2 + gain[1] * r2 * r2)[..., None]
    return cm.linear_to_srgb(cm.srgb_to_linear(image) * factor).astype(np.float32)


def shading_between(
    source: NDArray[np.float32], target: NDArray[np.float32]
) -> tuple[float, float]:
    """Return the radial gain that brings *source*'s lens shading to *target*'s.

    A camera corrects its own lens's vignetting in the JPEG; this program
    does so only when Lensfun knows the lens. Left in, the difference is
    darker corners on one side, which the fit would read as a darker tone
    curve and put into the look. So the brightness the look cannot explain
    is fitted as a smooth function of the distance from the centre - a
    vignette has no other shape - and the source is lit to match before the
    look is measured. Only the change with radius is kept: what is the same
    everywhere belongs to the tone curve.
    """
    r2 = radius_squared(*source.shape[:2])
    in_range = (
        (source > SAFE_LOW).all(axis=2)
        & (source < SAFE_HIGH).all(axis=2)
        & (target > SAFE_LOW).all(axis=2)
        & (target < SAFE_HIGH).all(axis=2)
    )
    total = (0.0, 0.0)
    for _ in range(SHADING_ROUNDS):
        lit = shade(source, total)
        look, _ = fit("shading", lit[in_range], target[in_range])
        fitted = apply_look(look, lit)
        weights = cm.SRGB_TO_XYZ[1].astype(np.float32)
        ours = cm.srgb_to_linear(fitted) @ weights
        theirs = cm.srgb_to_linear(target) @ weights
        keep = in_range & (ours > 1e-3) & (theirs > 1e-3)
        ratio = np.log(theirs[keep] / ours[keep]).astype(np.float64)
        rr = r2[keep].astype(np.float64)
        design = np.stack((np.ones_like(rr), rr, rr * rr), axis=1)
        solved, *_ = np.linalg.lstsq(design, ratio, rcond=None)
        total = (total[0] + float(solved[1]), total[1] + float(solved[2]))
    return total


def measure_scene(pair: Pair, long_edge: int, stops: float = 0.0) -> Scene:
    """Develop *pair*, line its two frames up and pick out the usable pixels."""
    source_image = our_rendering(pair.raw_path, decoder.PROXY_LONG_EDGE, stops)
    target_image = camera_rendering(pair.jpeg_path, decoder.PROXY_LONG_EDGE)
    target_image, alignment = line_up(source_image, target_image)
    gain = shading_between(shrink(source_image, long_edge), shrink(target_image, long_edge))
    source_image = shade(source_image, gain)
    corner = float(np.exp(gain[0] + gain[1]))
    alignment += f"; shading evened out, corners x{corner:.3f}"
    small_source = shrink(source_image, long_edge)
    small_target = shrink(target_image, long_edge)
    mask = usable(small_source, small_target)
    return Scene(
        pair, source_image, target_image, alignment, small_source[mask], small_target[mask]
    )


def pooled(scenes: list[Scene]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return every scene's samples together, no scene with an outsized say.

    A scene of mostly grass would otherwise outvote a small one of flowers
    simply by having more usable pixels, so a scene with more than twice the
    samples of the smaller scenes is thinned to that before they are joined.
    """
    if len(scenes) == 1:
        return scenes[0].source, scenes[0].target
    sizes = sorted(len(scene.source) for scene in scenes)
    cap = sizes[len(sizes) // 4] * 2
    generator = np.random.default_rng(0)
    sources: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []
    for scene in scenes:
        keep = np.arange(len(scene.source))
        if len(keep) > cap:
            keep = np.sort(generator.choice(keep, cap, replace=False))
        sources.append(scene.source[keep])
        targets.append(scene.target[keep])
    return np.concatenate(sources), np.concatenate(targets)


def fit_look(
    identifier: str,
    pairs: list[Pair],
    report_dir: Path,
    long_edge: int,
    stops: float = 0.0,
    neutral_curves: bool = False,
) -> tuple[Look, dict[str, object]]:
    """Fit one look from every pair in its folder and write its pictures."""
    scenes = [measure_scene(pair, long_edge, stops) for pair in pairs]
    source, target = pooled(scenes)
    look, history = fit(identifier, source, target, neutral_curves=neutral_curves)

    per_scene: list[dict[str, object]] = []
    for index, scene in enumerate(scenes):
        entry: dict[str, object] = {
            "raw": scene.pair.raw_path.name,
            "alignment": scene.alignment,
            "samples": int(len(scene.source)),
            "delta_before": difference(scene.source, scene.target),
            "delta_after": difference(apply_look(look, scene.source), scene.target),
        }
        if len(scenes) >= 3:
            others = scenes[:index] + scenes[index + 1 :]
            held_out, _ = fit(identifier, *pooled(others), neutral_curves=neutral_curves)
            entry["delta_left_out"] = difference(
                apply_look(held_out, scene.source), scene.target
            )
        per_scene.append(entry)
        name = identifier if len(scenes) == 1 else f"{identifier}-{index + 1:02d}"
        comparison(
            shrink(scene.source_image, 1200),
            apply_look(look, shrink(scene.source_image, 1200)),
            shrink(scene.target_image, 1200),
            report_dir / f"{name}-comparison.jpg",
        )
    curve_plot(look, report_dir / f"{identifier}-curves.png")

    extent = reach(look)
    summary: dict[str, object] = {
        "id": identifier,
        "raw": scenes[0].pair.raw_path.name if len(scenes) == 1 else f"{len(scenes)} scenes",
        "samples": int(len(source)),
        "delta_before": history[0],
        "delta_after": history[-1],
        "rounds": [round(entry["mean"], 3) for entry in history],
        "reach": extent,
        "midtone_stops": midtone_stops(look),
        "hue_coverage": hue_coverage(source),
        "scenes": per_scene,
    }
    if (
        extent["chroma"] > MAX_LOG_SCALE
        or extent["lightness"] > MAX_LOG_SCALE
        or extent["turn_degrees"] > np.degrees(MAX_HUE_TURN)
    ):
        summary["warning"] = "the fit reaches past what core.filters will let through"
    return look, summary


def midtone_stops(look: Look) -> list[float]:
    """Return how far each curve moves the midtones, in stops of linear light.

    The average over encoded 0.15 to 0.8, the part of a photograph the eye
    reads its brightness from. A look whose three numbers are all well below
    nought darkens the photograph rather than colouring it; see ``--exposure``.
    """
    steps = np.linspace(0.15, 0.8, 27, dtype=np.float32)
    plain = cm.srgb_to_linear(steps)
    return [
        round(
            float(
                np.log2(
                    cm.srgb_to_linear(curve(np.asarray(points, dtype=np.float32), steps)) / plain
                ).mean()
            ),
            3,
        )
        for points in look.curves
    ]


def report_line(summary: dict[str, object]) -> str:
    """Return one line of the printed report."""
    before = summary["delta_before"]
    after = summary["delta_after"]
    extent = summary["reach"]
    assert isinstance(before, dict) and isinstance(after, dict) and isinstance(extent, dict)
    line = (
        f"  {str(summary['id']).upper():<4} {summary['raw']:<14} "
        f"samples {summary['samples']:>7}   "
        f"dE mean {before['mean']:5.2f} -> {after['mean']:5.2f}   "
        f"p95 {before['p95']:5.2f} -> {after['p95']:5.2f}   "
        f"chroma x{np.exp(extent['chroma']):4.2f} turn {extent['turn_degrees']:4.1f} deg"
        f"   midtones {summary['midtone_stops']} EV"
    )
    scenes = summary.get("scenes")
    for scene in scenes if isinstance(scenes, list) else []:
        if "delta_left_out" not in scene:
            continue
        line += (
            f"\n       {scene['raw']:<28} dE {scene['delta_before']['mean']:5.2f} -> "
            f"{scene['delta_after']['mean']:5.2f}, left out of the fit "
            f"{scene['delta_left_out']['mean']:5.2f}"
        )
    if "warning" in summary:
        line += f"\n       {summary['warning']}"
    return line


def check_against_other_camera(
    folder: Path, pairs: list[Pair], looks: dict[str, Look], arguments: argparse.Namespace
) -> list[dict[str, object]]:
    """Run the cross camera check if the second camera's frame is there."""
    if not folder.is_dir():
        return []
    raws = [path for path in sorted(folder.iterdir()) if decoder.is_raw(path)]
    if not raws:
        return []
    jpeg = raws[0].with_suffix(".JPG")
    if not jpeg.exists():
        return []

    print(f"\nchecking the looks on {raws[0].name}, the same scene from another camera")
    checks = cross_camera_check(
        raws[0], jpeg, pairs, looks, arguments.report, arguments.long_edge
    )
    for check in checks:
        if "error" in check:
            print(f"  {str(check['id']).upper():<4} could not be checked: {check['error']}")
            continue
        plain = check["without_filter"]
        assert isinstance(plain, dict)
        if "with_filter" not in check:
            print(
                f"  {check['id']}: dE mean {plain['mean']:5.2f} "
                f"at {plain['exposure']:+.2f} EV"
            )
            continue
        looked = check["with_filter"]
        assert isinstance(looked, dict)
        print(
            f"  {str(check['id']).upper():<4} dE mean {plain['mean']:5.2f} -> "
            f"{looked['mean']:5.2f}   p95 {plain['p95']:5.2f} -> {looked['p95']:5.2f}   "
            f"at {looked['exposure']:+.2f} EV, {check['alignment']}"
        )
    return checks


def main() -> int:
    """Fit every pair under the sample folder and write the looks out."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--samples",
        type=Path,
        default=root / "sample" / "filter" / "pairs",
        help="a folder of folders, each holding one RAW and the JPEG beside it",
    )
    parser.add_argument("--out", type=Path, default=root / "resources" / "filters")
    parser.add_argument("--report", type=Path, default=root / "sample" / "filter" / "_fit")
    parser.add_argument("--long-edge", type=int, default=MEASURE_LONG_EDGE)
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        default=DEFAULT_CATEGORY,
        help="the looks tab group a new look goes under",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.0,
        help="stops every scene is developed at before the look is measured",
    )
    parser.add_argument(
        "--chroma-scale",
        type=float,
        default=1.0,
        help="multiply every hue's colourfulness by this after fitting (a taste choice)",
    )
    parser.add_argument(
        "--neutral-curves",
        action="store_true",
        help="measure the tone curves mostly on near greys",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=root / "sample" / "filter" / "other-camera",
        help="a folder holding another camera's RAW and JPEG of the same scene",
    )
    arguments = parser.parse_args()

    pairs = find_pairs(arguments.samples)
    if not pairs:
        print(f"no RAW and JPEG pairs under {arguments.samples}")
        return 1

    grouped: dict[str, list[Pair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.identifier, []).append(pair)
    print(f"fitting {len(grouped)} looks from {arguments.samples}")
    summaries: list[dict[str, object]] = []
    looks: dict[str, Look] = {}
    for identifier, members in grouped.items():
        look, summary = fit_look(
            identifier,
            members,
            arguments.report,
            arguments.long_edge,
            arguments.exposure,
            arguments.neutral_curves,
        )
        looks[identifier] = look
        if arguments.chroma_scale != 1.0:
            # A choice of taste made after the measurement, not part of it:
            # the camera's own rendering is what the report above describes.
            chroma = list(look.chroma)
            chroma[0] += float(np.log(arguments.chroma_scale))
            look = replace(look, chroma=tuple(chroma))
        existing = arguments.out / f"{identifier}.json"
        category = arguments.category
        if existing.exists():
            category = load(existing).category
        # Deliberately no camera make, model or file name: a look is this
        # project's own, and naming the body it was measured from would
        # invite the reading that the look belongs to that body.
        summary_source = {
            "method": (
                "fitted by least squares: a RAW file developed by this program, "
                "against the JPEG the same camera wrote of the same exposure"
            ),
            "samples": summary["samples"],
            "scenes": len(members),
            "chroma_scale_after_fit": arguments.chroma_scale,
            "delta_e_oklab": summary["delta_after"],
            "hue_coverage": summary["hue_coverage"],
            "fitted_at": datetime.now().isoformat(timespec="seconds"),
        }
        save(
            Look(
                identifier=look.identifier,
                curves=look.curves,
                chroma=look.chroma,
                hue=look.hue,
                lightness=look.lightness,
                source=summary_source,
                category=category,
            ),
            existing,
        )
        summaries.append(summary)
        print(report_line(summary))

    checks = check_against_other_camera(arguments.check, pairs, looks, arguments)

    arguments.report.mkdir(parents=True, exist_ok=True)
    (arguments.report / "report.json").write_text(
        json.dumps({"fit": summaries, "other_camera": checks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"looks written to {arguments.out}")
    print(f"pictures and report written to {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
