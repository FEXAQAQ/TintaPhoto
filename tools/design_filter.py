"""Write a designed look: one that is described rather than measured.

A measured look (``tools/fit_filter.py``) is what a camera does to a frame.
A designed look is a grading style no camera ships, so there is nothing to
measure it from; instead it is written down as a short, readable recipe -
which way the shadows and the highlights lean, and what happens to the
colours around a few named hues - and this tool turns the recipe into the
same three tone curves and three Fourier series every look is made of.

Nothing here is tuned by hand in the file it writes: change the recipe and
run the tool again. Run it with no arguments to write every design into
``resources/filters``::

    .venv\\Scripts\\python.exe tools\\design_filter.py

``--preview <photo> ...`` also develops each photo with and without the
look, at half and at full strength, side by side, into ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import decoder, pipeline  # noqa: E402
from core.edit_state import EditState  # noqa: E402
from core.filters import (  # noqa: E402
    CATEGORIES,
    MAX_HUE_TURN,
    MAX_LOG_SCALE,
    Look,
    fourier,
    save,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SHIPPED: Final[Path] = PROJECT_ROOT / "resources" / "filters"

#: Knots in each tone curve, the same as a measured look.
CURVE_KNOTS: Final[int] = 17
#: Harmonics in each Fourier series. One more than a measured look: a
#: recipe says exactly what it wants at every hue, so there is no stray
#: detail of a single scene for the extra harmonic to chase, and it lets a
#: change stay around the hue it was asked for.
HARMONICS: Final[int] = 4
#: How firmly the finer harmonics are held down, against the recipe.
SERIES_RIDGE: Final[float] = 1e-3
#: Places around the hue circle the recipe is read at before it is fitted.
CIRCLE_POINTS: Final[int] = 720


@dataclass(frozen=True)
class Tone:
    """What happens to brightness and to the casts of shadows and highlights.

    *shadows* and *highlights* are each channel's push, in encoded sRGB, at
    the place in the tone range where each has its most say; both fade to
    nothing at black and at white, so black stays black and white stays
    white. *contrast* is a gentle S around mid grey: 0.1 makes the middle
    ten per cent steeper and the two ends ten per cent flatter.
    """

    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contrast: float = 0.0


@dataclass(frozen=True)
class Bump:
    """A change to the colours around one hue.

    *hue* is an OkLCh hue in degrees and *width* how far either side of it,
    in degrees, the change has fallen to a half. *chroma* and *lightness*
    are per cent changes at the centre. *pull* draws the hues nearby
    towards *hue*: at 0.5 a colour ten degrees off ends up five degrees off.
    """

    hue: float
    width: float
    chroma: float = 0.0
    lightness: float = 0.0
    pull: float = 0.0


@dataclass(frozen=True)
class Design:
    """A whole designed look: its name, where it goes, and its recipe.

    *saturation* is a per cent change to every colour at once, before the
    bumps. A look worn at full strength takes the place of the RAW base
    chroma (PLAN.md 6 step 10), so a designed look carries some of it here.
    """

    identifier: str
    category: str
    tone: Tone
    bumps: tuple[Bump, ...] = field(default_factory=tuple)
    saturation: float = 0.0


# --------------------------------------------------------------------------
# The designs
# --------------------------------------------------------------------------

#: The skin and the warm colours gather into one orange and come forward;
#: greens turn towards teal and step back; skies and water gather into teal;
#: purples quieten so they do not fight either. The shadows lean teal and
#: the highlights warm, so even a grey scene carries the two colours.
TEAL_ORANGE: Final[Design] = Design(
    identifier="teal-orange",
    category="cinematic",
    tone=Tone(
        shadows=(-0.055, 0.009, 0.0525),
        highlights=(0.055, -0.009, -0.0525),
        contrast=0.14,
    ),
    saturation=10.0,
    bumps=(
        Bump(hue=60.0, width=32.0, chroma=4.0, lightness=3.0, pull=0.4),
        Bump(hue=25.0, width=20.0, chroma=-8.0),
        Bump(hue=140.0, width=40.0, chroma=-45.0, lightness=-6.0),
        Bump(hue=215.0, width=55.0, chroma=18.0, lightness=-6.0, pull=0.45),
        Bump(hue=310.0, width=45.0, chroma=-32.0),
    ),
)

DESIGNS: Final[tuple[Design, ...]] = (TEAL_ORANGE,)


# --------------------------------------------------------------------------
# From a recipe to a look
# --------------------------------------------------------------------------


def _hump(values: NDArray[np.float64], towards_white: bool) -> NDArray[np.float64]:
    """Return a bump over the tone range that is nothing at both ends.

    The shadows' hump peaks about a third of the way up and the highlights'
    two thirds of the way; at mid grey the two meet at the same height, so
    pushes that cancel there leave mid grey neutral. Each peaks at one.
    """
    shape = np.sin(np.pi * values) * (values if towards_white else 1.0 - values)
    reference = np.linspace(0.0, 1.0, 1001)
    peak = np.max(np.sin(np.pi * reference) * reference)
    return shape / peak


def tone_curves(tone: Tone, knots: int = CURVE_KNOTS) -> NDArray[np.float64]:
    """Return the three curves' knot values for *tone*."""
    x = np.linspace(0.0, 1.0, knots)
    s_curve = x - tone.contrast * np.sin(2.0 * np.pi * x) / (2.0 * np.pi)
    shadows = _hump(x, towards_white=False)
    highlights = _hump(x, towards_white=True)
    return np.stack(
        [
            s_curve + tone.shadows[channel] * shadows + tone.highlights[channel] * highlights
            for channel in range(3)
        ]
    )


def _closeness(hue: NDArray[np.float64], bump: Bump) -> NDArray[np.float64]:
    """Return how much of *bump* reaches each hue, one at its centre."""
    centre = np.radians(bump.hue)
    half = np.radians(bump.width)
    sharpness = np.log(2.0) / (1.0 - np.cos(half))
    return np.exp(sharpness * (np.cos(hue - centre) - 1.0))


def hue_targets(
    design: Design, hue: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return what the recipe asks at every *hue*: log chroma, turn, log lightness."""
    chroma = np.full_like(hue, np.log1p(design.saturation / 100.0))
    turn = np.zeros_like(hue)
    lightness = np.zeros_like(hue)
    for bump in design.bumps:
        near = _closeness(hue, bump)
        chroma += near * np.log1p(bump.chroma / 100.0)
        lightness += near * np.log1p(bump.lightness / 100.0)
        offset = np.mod(np.radians(bump.hue) - hue + np.pi, 2.0 * np.pi) - np.pi
        turn += near * bump.pull * offset
    return chroma, turn, lightness


def basis(hue: NDArray[np.float64], harmonics: int) -> NDArray[np.float64]:
    """Return the Fourier design matrix: a constant and each harmonic's pair."""
    columns = [np.ones_like(hue)]
    for order in range(1, harmonics + 1):
        columns.append(np.cos(hue * order))
        columns.append(np.sin(hue * order))
    return np.stack(columns, axis=1)


def fit_series(
    hue: NDArray[np.float64], target: NDArray[np.float64], harmonics: int = HARMONICS
) -> NDArray[np.float64]:
    """Return the Fourier series nearest *target*, finer harmonics held down."""
    design = basis(hue, harmonics)
    penalty = [0.0]
    for order in range(1, harmonics + 1):
        penalty.extend([float(order * order)] * 2)
    normal = design.T @ design + SERIES_RIDGE * len(hue) * np.diag(penalty)
    return np.linalg.solve(normal, design.T @ target)


def build(design: Design) -> Look:
    """Turn *design* into a look, refusing one the program would cut off.

    A look that reaches its limits (``core.filters``) is clipped when it is
    applied, so it would not be what the recipe says; nor would a curve that
    turned back on itself, which would posterise a gradient.
    """
    if design.category not in CATEGORIES:
        raise ValueError(f"{design.identifier}: unknown category {design.category}")
    curves = tone_curves(design.tone)
    if (np.diff(curves, axis=1) < 0.0).any():
        raise ValueError(f"{design.identifier}: a tone curve falls")

    hue = np.linspace(-np.pi, np.pi, CIRCLE_POINTS, endpoint=False)
    chroma, turn, lightness = (fit_series(hue, target) for target in hue_targets(design, hue))
    check = hue.astype(np.float32)
    for name, series, limit in (
        ("chroma", chroma, MAX_LOG_SCALE),
        ("hue", turn, MAX_HUE_TURN),
        ("lightness", lightness, MAX_LOG_SCALE),
    ):
        reach = float(np.abs(fourier(series.astype(np.float32), check)).max())
        if reach > limit:
            raise ValueError(f"{design.identifier}: {name} reaches {reach:.3f}, over {limit}")

    return Look(
        identifier=design.identifier,
        curves=(
            tuple(curves[0].tolist()),
            tuple(curves[1].tolist()),
            tuple(curves[2].tolist()),
        ),
        chroma=tuple(chroma.tolist()),
        hue=tuple(turn.tolist()),
        lightness=tuple(lightness.tolist()),
        source={"method": "designed", "recipe": _recipe(design)},
        category=design.category,
    )


def _recipe(design: Design) -> dict[str, object]:
    """Return the recipe as plain data, to be kept in the look's file."""
    return {
        "shadows": list(design.tone.shadows),
        "highlights": list(design.tone.highlights),
        "contrast": design.tone.contrast,
        "saturation": design.saturation,
        "bumps": [
            {
                "hue": bump.hue,
                "width": bump.width,
                "chroma": bump.chroma,
                "lightness": bump.lightness,
                "pull": bump.pull,
            }
            for bump in design.bumps
        ],
    }


# --------------------------------------------------------------------------
# Looking at it
# --------------------------------------------------------------------------


def preview(look: Look, photo_path: Path, out: Path, long_edge: int = 1200) -> Path:
    """Write *photo_path* without the look, at half and at full strength."""
    import cv2

    photo = decoder.decode_proxy(photo_path, long_edge)
    resting = EditState.for_photo(photo.is_raw, *pipeline.shot_white_balance(photo))
    panels = [pipeline.render(photo, resting)]
    for strength in (50.0, 100.0):
        state = resting.with_filter(look.identifier).with_value("filter_strength", strength)
        panels.append(pipeline.render(photo, state, look=look))
    strip = np.concatenate([np.asarray(panel, dtype=np.uint8) for panel in panels], axis=1)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{look.identifier}_{photo_path.stem}_{photo_path.suffix.lstrip('.')}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main() -> None:
    """Write every design, and preview it on any photos given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "sample" / "filter" / "_design")
    parser.add_argument("--dry-run", action="store_true", help="build without writing")
    arguments = parser.parse_args()

    for design in DESIGNS:
        look = build(design)
        if not arguments.dry_run:
            save(look, SHIPPED / f"{design.identifier}.json")
            print(f"wrote {design.identifier}")
        for photo in arguments.preview:
            print(f"  preview {preview(look, photo, arguments.out)}")


if __name__ == "__main__":
    main()
