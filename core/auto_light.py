"""Automatic light: setting the light group's sliders from the photo itself.

The magic wand on the light group's header (PLAN.md 3.1, section 6). It sets
five sliders - exposure, highlights, shadows, whites and blacks - and leaves
everything else as it is: the contrast the user chose, the white balance,
the colour, the portrait, effects and detail groups, the film look and the
crop.

Nothing here is a formula that predicts what a slider will do. Each slider
is found by trying values on a small copy of the photo, developed through
the very pipeline the preview and the export use, measuring the result and
halving the range until it lands: so what is measured is exactly what will
be seen, whatever the curves of section 6 are tuned to later.

The search starts with the five sliders at zero every time, so pressing the
button twice gives the same answer, and so does pressing it after moving
them by hand.

In order:

1. Exposure takes the median brightness part of the way towards middle grey
   (``AUTO_EXPOSURE_SHARE``), never more than ``AUTO_EXPOSURE_LIMIT_EV``:
   a night stays a night and snow stays bright. (See below for its ceiling.)
2. Highlights come down, only when more than a small share of the photo is
   burnt out, until that share is back to it or the limit is reached.
3. Shadows come up, only when the darkest tenth is darker than the target,
   until it reaches it or the limit is reached.
4. Whites lift the brightest few pixels to just short of white, and never
   pull down a white that is already there (a sun stays a sun).
5. Blacks lower the darkest few pixels to just short of black, which is
   what takes the haze off a flat photo, and never lift a black that is
   already there.

The exposure never rises past the point where the brightest few pixels
reach white, so a small bright subject on a large black ground is not
pushed into burning out by all the black around it.

Every coefficient is a constant at the top of :mod:`core.pipeline`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from core import pipeline
from core.decoder import DecodedPhoto
from core.edit_state import EditState
from core.filters import Look

#: The sliders the wand sets, in the order it sets them.
AUTO_NAMES: Final[tuple[str, ...]] = ("exposure", "highlights", "shadows", "whites", "blacks")

#: How far the exposure is searched for the whole way to middle grey, before
#: the share and the limit are taken; twice the limit, so the share still
#: has something to cut.
_EXPOSURE_SEARCH_EV: Final[float] = 2.0 * pipeline.AUTO_EXPOSURE_LIMIT_EV

_EIGHT_BIT_MAX: Final[float] = 255.0
#: Rec. 709 weights for the brightness of an encoded pixel.
_LUMA: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class _Measured:
    """What one try of the photo looks like, in encoded values from 0 to 1."""

    median: float
    shadow: float
    white: float
    black: float
    clipped: float


def _measure(image: NDArray[np.uint8]) -> _Measured:
    """Measure a developed 8 bit picture."""
    values = image.reshape(-1, 3).astype(np.float32) / np.float32(_EIGHT_BIT_MAX)
    brightness = values @ np.asarray(_LUMA, dtype=np.float32)
    brightest = values.max(axis=1)
    darkest = values.min(axis=1)
    return _Measured(
        median=float(np.median(brightness)),
        shadow=float(np.percentile(brightness, pipeline.AUTO_SHADOW_PERCENTILE)),
        white=float(np.percentile(brightest, pipeline.AUTO_WHITE_PERCENTILE)),
        black=float(np.percentile(darkest, pipeline.AUTO_BLACK_PERCENTILE)),
        clipped=float(np.mean(brightest >= pipeline.AUTO_CLIPPED_LEVEL)),
    )


def _solve(
    reading: Callable[[float], float],
    low: float,
    high: float,
    target: float,
) -> float:
    """Return where between *low* and *high* a rising *reading* meets *target*.

    The reading has to rise with the value, which every slider here does when
    looked at the right way. A target out of reach comes back as the end of
    the range nearest to it.
    """
    for _ in range(pipeline.AUTO_SEARCH_STEPS):
        middle = (low + high) / 2.0
        if reading(middle) > target:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def auto_light(photo: DecodedPhoto, state: EditState, look: Look | None = None) -> EditState:
    """Return *state* with the five light sliders set for *photo*.

    *photo* is the photo as it will be developed - lens corrected, framed and
    cropped - and small: ``AUTO_LONG_EDGE`` is the size it is meant for.
    *look* is the film look *state* wears, if any, as :func:`pipeline.develop`
    needs it. The film grain is left out while measuring, because it is noise
    that would move the percentiles from one try to the next.
    """
    start = state
    for name in AUTO_NAMES:
        start = start.with_value(name, 0.0)
    trial_base = start.with_value("grain", 0.0)
    detailed = pipeline.with_detail(photo, trial_base)

    def measured(trial: EditState) -> _Measured:
        return _measure(pipeline.develop(detailed, trial, look=look))

    def trying(
        name: str, base: EditState, pick: Callable[[_Measured], float]
    ) -> Callable[[float], float]:
        return lambda value: pick(measured(base.with_value(name, value)))

    # 1. Exposure, part of the way to middle grey, and never brighter than
    #    the point where the brightest few pixels reach white: a photo whose
    #    bright parts already touch white is not dark, however much black
    #    surrounds them.
    whole_way = _solve(
        trying("exposure", trial_base, lambda m: m.median),
        -_EXPOSURE_SEARCH_EV,
        _EXPOSURE_SEARCH_EV,
        pipeline.AUTO_MIDTONE_TARGET,
    )
    limit = pipeline.AUTO_EXPOSURE_LIMIT_EV
    stops = float(np.clip(whole_way * pipeline.AUTO_EXPOSURE_SHARE, -limit, limit))
    if stops > 0.0:
        ceiling = _solve(
            trying("exposure", trial_base, lambda m: m.white),
            0.0,
            stops,
            pipeline.AUTO_WHITE_TARGET,
        )
        stops = min(stops, ceiling)
    trial = trial_base.with_value("exposure", stops)

    # 2. Highlights, only when too much of the photo is burnt out. The
    #    burnt out share rises as the slider goes back up towards zero.
    if measured(trial).clipped > pipeline.AUTO_CLIPPED_SHARE:
        highlights = _solve(
            trying("highlights", trial, lambda m: m.clipped),
            -pipeline.AUTO_HIGHLIGHTS_LIMIT,
            0.0,
            pipeline.AUTO_CLIPPED_SHARE,
        )
        trial = trial.with_value("highlights", highlights)

    # 3. Shadows, only when the darkest tenth is too dark.
    if measured(trial).shadow < pipeline.AUTO_SHADOW_TARGET:
        shadows = _solve(
            trying("shadows", trial, lambda m: m.shadow),
            0.0,
            pipeline.AUTO_SHADOWS_LIMIT,
            pipeline.AUTO_SHADOW_TARGET,
        )
        trial = trial.with_value("shadows", shadows)

    # 4. and 5. The white and the black point, each moved one way only:
    #    whites lift the brightest pixels up to white and never pull a white
    #    down, blacks lower the darkest to black and never lift a black.
    whites = _solve(
        trying("whites", trial, lambda m: m.white),
        0.0,
        pipeline.AUTO_WHITES_LIMIT,
        pipeline.AUTO_WHITE_TARGET,
    )
    trial = trial.with_value("whites", whites)
    blacks = _solve(
        trying("blacks", trial, lambda m: m.black),
        -pipeline.AUTO_BLACKS_LIMIT,
        0.0,
        pipeline.AUTO_BLACK_TARGET,
    )
    trial = trial.with_value("blacks", blacks)

    # The sliders show whole steps and the exposure hundredths of a stop, so
    # the answer is rounded to what they can show.
    result = state
    for name in AUTO_NAMES:
        value = getattr(trial, name)
        result = result.with_value(
            name, round(value, 2) if name == "exposure" else float(round(value))
        )
    return result
