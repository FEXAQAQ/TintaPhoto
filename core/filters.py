"""The film looks of the looks tab.

A look is a fixed colour rendering measured from a reference photograph, not
a set of sliders: the same photograph developed twice, once by this program
and once by a camera, says exactly what the camera does to every colour, and
a look is that difference written down.

It is written down in two parts. Three tone curves, one per channel, carry
everything that depends on brightness - the contrast, the toe and the
shoulder, and the casts a film gives the shadows and the highlights. Three
smooth functions of hue then carry what depends on colour: how colourful a
hue comes out, which way it turns, and how light it is. Hue is an angle, so
those three are Fourier series, which are periodic and smooth by
construction; a look therefore has no joins to show in a gradient, and
nothing to blow up where the reference photograph happened to hold no
colour.

The whole of it is about eighty numbers, kept in ``resources/filters`` as
JSON and fitted by ``tools/fit_filter.py``. A grading style no camera ships
- the cinematic looks - has nothing to be measured from, and is written by
``tools/design_filter.py`` from a short recipe into the same form.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from core import color_management as cm

_log = logging.getLogger(__name__)

#: The file format of a look, for the version field of PLAN.md section 7.
FILTER_FILE_VERSION: Final[int] = 1

#: The identifier of "no filter", and the default of the filter parameter.
NO_FILTER: Final[str] = ""

#: The categories the looks tab groups looks under, in the order it shows
#: them. A look names its own; a file that names none, or one this copy does
#: not know, goes under the first.
CATEGORIES: Final[tuple[str, ...]] = ("film", "medium-format", "street", "cinematic")
DEFAULT_CATEGORY: Final[str] = CATEGORIES[0]

#: Chroma at which the hue dependent lightness reaches its full strength.
#: Below it the effect fades out, so that a neutral grey - which has no hue
#: to look up - keeps the lightness the tone curves gave it.
LIGHTNESS_FULL_CHROMA: Final[float] = 0.10

#: How far the two multiplying functions may go, as natural logarithms, and
#: how far a hue may turn, in radians. A fit is free within these and cannot
#: leave them, so a look read from disk can never produce a colour four
#: times too strong or turn green into magenta.
MAX_LOG_SCALE: Final[float] = 0.7
MAX_HUE_TURN: Final[float] = 0.35

#: How many angles the three hue functions are sampled at before a photo is
#: rendered. Four thousand puts a tenth of a degree between neighbours, far
#: finer than three harmonics can move in, so a smooth sky stays smooth.
HUE_TABLE_SIZE: Final[int] = 4096


def fourier(coefficients: NDArray[np.float32], hue: NDArray[np.float32]) -> NDArray[np.float32]:
    """Evaluate ``c0 + sum(a_k cos kh + b_k sin kh)`` at every *hue*.

    The coefficients are flat - the constant first, then each harmonic's
    cosine and sine - because that is the order the fit solves them in.
    """
    total = np.full(hue.shape, np.float32(coefficients[0]), dtype=np.float32)
    for order in range(1, (len(coefficients) + 1) // 2):
        angle = hue * np.float32(order)
        total += np.float32(coefficients[2 * order - 1]) * np.cos(angle)
        total += np.float32(coefficients[2 * order]) * np.sin(angle)
    return total


def curve(points: NDArray[np.float32], values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Read a tone curve at every one of *values*.

    *points* are the curve's knot values, evenly spaced from nought to one.
    Being evenly spaced is worth more than it sounds: which segment a value
    falls in is a multiplication rather than a search through the knots, and
    the fraction along that segment falls out of the same multiplication.

    Outside nought and one that fraction simply runs past its ends, which
    continues the first and last segments straight out. Nothing is clamped:
    the vignette can push a pixel past white and a wide gamut colour can
    take a channel below black, and both have to come back out again rather
    than being flattened against the end of the curve.
    """
    last = len(points) - 1
    position = np.asarray(values, dtype=np.float32) * np.float32(last)
    segment = np.clip(position.astype(np.int32), 0, last - 1)
    along = position - segment
    low = points[segment]
    return low + along * (points[segment + 1] - low)


@dataclass(frozen=True)
class Look:
    """One film look: three tone curves and three functions of hue.

    ``curves`` holds the three channels' knot values, evenly spaced from zero
    to one in the sRGB encoded domain. ``chroma`` and ``lightness`` are
    natural logarithms of a multiplier and ``hue`` is an angle in radians,
    each a Fourier series over the hue circle.
    """

    identifier: str
    curves: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    chroma: tuple[float, ...]
    hue: tuple[float, ...]
    lightness: tuple[float, ...]
    #: What the look was measured from, for the report and the about box.
    source: dict[str, Any] | None = None
    #: Which group of the looks tab the look sits in, one of ``CATEGORIES``.
    category: str = DEFAULT_CATEGORY

    @classmethod
    def identity(cls, identifier: str = NO_FILTER, knots: int = 17) -> "Look":
        """Return a look that changes nothing."""
        straight = tuple(float(v) for v in np.linspace(0.0, 1.0, knots))
        return cls(
            identifier=identifier,
            curves=(straight, straight, straight),
            chroma=(0.0,),
            hue=(0.0,),
            lightness=(0.0,),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the look as the dictionary its JSON file holds."""
        data: dict[str, Any] = {
            "version": FILTER_FILE_VERSION,
            "id": self.identifier,
            "category": self.category,
            "curves": {
                "r": list(self.curves[0]),
                "g": list(self.curves[1]),
                "b": list(self.curves[2]),
            },
            "chroma": list(self.chroma),
            "hue": list(self.hue),
            "lightness": list(self.lightness),
        }
        if self.source is not None:
            data["source"] = self.source
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Look":
        """Build a look from the contents of its JSON file.

        A file with a channel missing, a curve of fewer than two knots or a
        number that is not one is refused rather than quietly repaired: a
        half read look would show as wrong colour, which is worse than a
        filter that does not appear in the list at all.
        """
        curves = data.get("curves")
        if not isinstance(curves, dict):
            raise ValueError("a look needs three tone curves")
        channels: list[tuple[float, ...]] = []
        for name in ("r", "g", "b"):
            values = curves.get(name)
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"the {name} curve needs at least two knots")
            channels.append(tuple(float(value) for value in values))
        if len({len(channel) for channel in channels}) != 1:
            raise ValueError("the three tone curves need the same number of knots")
        source = data.get("source")
        category = data.get("category")
        return cls(
            identifier=str(data.get("id", NO_FILTER)),
            curves=(channels[0], channels[1], channels[2]),
            chroma=_series(data.get("chroma")),
            hue=_series(data.get("hue")),
            lightness=_series(data.get("lightness")),
            source=source if isinstance(source, dict) else None,
            category=category if category in CATEGORIES else DEFAULT_CATEGORY,
        )

    # -- what the pipeline asks -------------------------------------------

    @property
    def is_identity(self) -> bool:
        """Return whether the look leaves every colour exactly where it is."""
        straight = np.linspace(0.0, 1.0, len(self.curves[0]), dtype=np.float64)
        curves_straight = all(
            np.array_equal(np.asarray(curve, dtype=np.float64), straight)
            for curve in self.curves
        )
        series_flat = all(
            value == 0.0
            for series in (self.chroma, self.hue, self.lightness)
            for value in series
        )
        return curves_straight and series_flat

    def apply(self, image: NDArray[np.float32], strength: float = 1.0) -> NDArray[np.float32]:
        """Return *image* rendered through the look.

        *image* is linear ProPhoto RGB, the working space, and so is the
        result. *strength* runs from zero - which returns the image itself -
        to one, and takes every part of the look with it: at a half the
        curves are halfway back to straight and the three hue functions are
        halved, which is a weaker look rather than a blend of two pictures.
        """
        if strength <= 0.0 or self.is_identity:
            return image
        amount = np.float32(min(strength, 1.0))
        encoded = cm.linear_to_srgb(cm.apply_matrix(image, cm.PROPHOTO_TO_SRGB))
        shaped = self._apply_curves(encoded, amount)
        lab = cm.srgb_linear_to_oklab(cm.srgb_to_linear(shaped))
        return cm.oklab_to_prophoto(self._apply_hue_functions(lab, amount))

    def _apply_curves(
        self, encoded: NDArray[np.float32], strength: np.float32
    ) -> NDArray[np.float32]:
        """Run the three tone curves over encoded sRGB."""
        shaped = np.empty_like(encoded)
        for channel, values in enumerate(self.curves):
            plane = encoded[..., channel]
            curved = curve(np.asarray(values, dtype=np.float32), plane)
            if strength >= 1.0:
                shaped[..., channel] = curved
            else:
                shaped[..., channel] = plane + strength * (curved - plane)
        return shaped

    def _hue_tables(
        self, strength: np.float32
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Return the three hue functions sampled around the circle.

        Nothing in them depends on the photo, so they are worked out once
        over a few thousand angles instead of per pixel - three Fourier
        series, two exponentials, a sine and a cosine that would otherwise
        be eighteen trigonometric passes over every pixel in the frame.

        The first two come back as one complex number laid out as a pair:
        turning a colour and making it stronger is multiplying ``a + ib`` by
        ``scale * e**(i * turn)``, so both are carried by its two halves and
        the pixel pays four multiplications for the pair of them.
        """
        hue = np.linspace(-np.pi, np.pi, HUE_TABLE_SIZE, endpoint=False, dtype=np.float32)
        turn = np.clip(
            fourier(np.asarray(self.hue, dtype=np.float32), hue) * strength,
            -MAX_HUE_TURN,
            MAX_HUE_TURN,
        )
        scale = np.exp(
            np.clip(
                fourier(np.asarray(self.chroma, dtype=np.float32), hue) * strength,
                -MAX_LOG_SCALE,
                MAX_LOG_SCALE,
            )
        )
        lift = np.clip(
            fourier(np.asarray(self.lightness, dtype=np.float32), hue) * strength,
            -MAX_LOG_SCALE,
            MAX_LOG_SCALE,
        )
        return (
            (scale * np.cos(turn)).astype(np.float32),
            (scale * np.sin(turn)).astype(np.float32),
            lift.astype(np.float32),
        )

    def _apply_hue_functions(
        self, lab: NDArray[np.float32], strength: np.float32
    ) -> NDArray[np.float32]:
        """Turn, strengthen and lighten each colour by what its hue asks.

        Lightness is the one of the three that cannot be left to look after
        itself near the neutral axis: a grey has no meaningful hue, so its
        share of the lightness function is faded out with chroma. Chroma and
        the turn need no such care, both being multiples of a chroma that is
        already nothing.
        """
        real, imaginary, lift = self._hue_tables(strength)
        lightness = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        hue = np.arctan2(b, a)
        where = np.clip(
            ((hue + np.float32(np.pi)) * np.float32(HUE_TABLE_SIZE / (2.0 * np.pi))).astype(
                np.int32
            ),
            0,
            HUE_TABLE_SIZE - 1,
        )
        multiplier = real[where]
        turner = imaginary[where]
        # Chroma, as a plain square root rather than through hypot, which
        # guards against an overflow that cannot happen in Oklab.
        fade = np.clip(
            np.sqrt(a * a + b * b) / np.float32(LIGHTNESS_FULL_CHROMA), 0.0, 1.0
        )

        return np.stack(
            (
                lightness * np.exp(lift[where] * fade),
                a * multiplier - b * turner,
                a * turner + b * multiplier,
            ),
            axis=-1,
        ).astype(np.float32)


def _series(values: Any) -> tuple[float, ...]:
    """Return a Fourier series read from a file, or a flat zero."""
    if not isinstance(values, list) or not values:
        return (0.0,)
    return tuple(float(value) for value in values)


def load(path: Path) -> Look:
    """Read one look from its JSON file."""
    return Look.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save(look: Look, path: Path) -> None:
    """Write *look* to *path*, through a temporary file (PLAN.md 7.1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(look.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_catalogue(directory: Path) -> tuple[Look, ...]:
    """Read every look in *directory*, in the order their files sort.

    A file that cannot be read is logged and left out, because a filter
    missing from the list is a smaller failure than a program that will not
    start (PLAN.md section 10).
    """
    if not directory.is_dir():
        return ()
    looks: list[Look] = []
    for path in sorted(directory.glob("*.json")):
        try:
            looks.append(load(path))
        except (OSError, ValueError, json.JSONDecodeError):
            _log.exception("could not read the filter %s", path.name)
    return tuple(looks)
