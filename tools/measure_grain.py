"""Measure the film grain in photos, against a photo of the same scene without it.

How the grain of PLAN.md section 6 step 11.5 was measured, kept so that it
can be measured again: the camera's own grain, to set the constants, and the
grain this program makes, to check them. Give it the photo without grain
first and then any number with it::

    .venv\\Scripts\\python.exe tools\\measure_grain.py off.jpg weak.jpg strong.jpg

The photos may have been shot by hand: each is lined up with the first by
its whole frame, and then patch by patch. Only flat patches are measured -
a wall, a dark bezel, the face of a plain ball - where what changes from the
first photo to the others is the grain and nothing else; the noise of the
first is taken back out by its variance.

For each photo it reports the grain's spread in 8 bit sRGB levels across the
brightness range, how alike the three channels are, how the grain of
neighbouring pixels is related, and how much it is left after the photo is
shrunk with OpenCV's area average, the way an export of a smaller size is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

#: The edge of the patches the brightness range is measured in.
PATCH: Final[int] = 48
#: The edge of the patches the correlation of neighbours is measured in.
SPECTRUM_PATCH: Final[int] = 128
#: How flat a patch of the first photo has to be, as the spread of its
#: blurred brightness in 8 bit levels.
FLAT: Final[float] = 1.0
#: How far, in pixels, the patch by patch alignment searches.
SEARCH: Final[int] = 12
#: The edges of the brightness bins of the report, in 8 bit levels.
TONE_EDGES: Final[tuple[int, ...]] = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240, 256)
#: The long edges the photos are also measured at.
SHRUNK_LONG_EDGES: Final[tuple[int, ...]] = (2560, 2048, 1080)

_LUMA: Final[NDArray[np.float32]] = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(path: Path) -> NDArray[np.float32]:
    """Return a photo as 8 bit levels in floating point, red green blue."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def shift_between(first: NDArray[np.float32], other: NDArray[np.float32]) -> tuple[int, int]:
    """Return how far *other* sits from *first*, rows and columns, by phase."""
    height, width = first.shape[:2]
    size = (width // 4, height // 4)
    small = [
        cv2.resize(image @ _LUMA, size, interpolation=cv2.INTER_AREA).astype(np.float64)
        for image in (first, other)
    ]
    (across, down), _ = cv2.phaseCorrelate(small[0], small[1])
    return int(round(down * 4)), int(round(across * 4))


def _plane_removed(patch: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return *patch* less a fitted plane, channel by channel."""
    size = patch.shape[0]
    down, across = np.mgrid[0:size, 0:size].astype(np.float32)
    basis = np.stack([np.ones(size * size), down.ravel(), across.ravel()], 1)
    flat = patch.reshape(size * size, -1)
    coefficients, *_ = np.linalg.lstsq(basis, flat, rcond=None)
    return (flat - basis @ coefficients).reshape(patch.shape).astype(np.float32)


def _matched(
    first_blur: NDArray[np.float32],
    other_blur: NDArray[np.float32],
    top: int,
    left: int,
    size: int,
    shift: tuple[int, int],
) -> tuple[int, int] | None:
    """Return where a flat patch of the first photo is in another, or None."""
    down, across = top + shift[0], left + shift[1]
    if down - SEARCH < 0 or across - SEARCH < 0:
        return None
    area = other_blur[
        down - SEARCH : down + size + SEARCH, across - SEARCH : across + size + SEARCH
    ]
    if area.shape != (size + 2 * SEARCH, size + 2 * SEARCH):
        return None
    template = first_blur[top : top + size, left : left + size]
    scores = cv2.matchTemplate(area, template, cv2.TM_SQDIFF)
    _, _, best, _ = cv2.minMaxLoc(scores)
    down, across = down - SEARCH + best[1], across - SEARCH + best[0]
    if other_blur[down : down + size, across : across + size].std() > FLAT * 1.3:
        return None
    return down, across


def measure(
    first: NDArray[np.float32], others: list[NDArray[np.float32]], blur: float = 3.0
) -> list[dict[str, object]]:
    """Return the grain of each of *others*, measured against *first*."""
    shifts = [shift_between(first, other) for other in others]
    first_luma = first @ _LUMA
    first_blur = cv2.GaussianBlur(first_luma, (0, 0), blur)
    other_blurs = [cv2.GaussianBlur(other @ _LUMA, (0, 0), blur) for other in others]
    height, width = first_luma.shape
    margin = max(SEARCH * 2, max(max(abs(a), abs(b)) for a, b in shifts) + SEARCH * 2)

    tones: list[list[tuple[float, float]]] = [[] for _ in others]
    covariances: list[list[NDArray[np.float64]]] = [[] for _ in others]
    spectra = [np.zeros((SPECTRUM_PATCH, SPECTRUM_PATCH)) for _ in others]
    first_spectrum = np.zeros((SPECTRUM_PATCH, SPECTRUM_PATCH))
    counted = 0
    window = np.outer(np.hanning(SPECTRUM_PATCH), np.hanning(SPECTRUM_PATCH))

    for size, step in ((PATCH, PATCH // 2), (SPECTRUM_PATCH, SPECTRUM_PATCH)):
        for top in range(margin, height - margin - size, step):
            for left in range(margin, width - margin - size, step):
                if first_blur[top : top + size, left : left + size].std() > FLAT:
                    continue
                places = [
                    _matched(first_blur, other_blur, top, left, size, shift)
                    for other_blur, shift in zip(other_blurs, shifts)
                ]
                if any(place is None for place in places):
                    continue
                reference = _plane_removed(first[top : top + size, left : left + size])
                level = float((first[top : top + size, left : left + size] @ _LUMA).mean())
                for index, (other, place) in enumerate(zip(others, places)):
                    assert place is not None
                    down, across = place
                    patch = _plane_removed(other[down : down + size, across : across + size])
                    if size == PATCH:
                        grain = (patch @ _LUMA).var() - (reference @ _LUMA).var()
                        tones[index].append((level, float(grain)))
                        covariances[index].append(
                            np.cov(patch.reshape(-1, 3).T) - np.cov(reference.reshape(-1, 3).T)
                        )
                    else:
                        spectra[index] += np.abs(np.fft.fft2((patch @ _LUMA) * window)) ** 2
                if size == SPECTRUM_PATCH:
                    first_spectrum += np.abs(np.fft.fft2((reference @ _LUMA) * window)) ** 2
                    counted += 1

    results = []
    for index in range(len(others)):
        levels = np.array([level for level, _ in tones[index]])
        grains = np.array([grain for _, grain in tones[index]])
        bins = []
        for low, high in zip(TONE_EDGES[:-1], TONE_EDGES[1:]):
            inside = (levels >= low) & (levels < high)
            if inside.sum() >= 4:
                bins.append((low, high - 1, int(inside.sum()), float(np.sqrt(max(np.median(grains[inside]), 0)))))
        midtones = (levels >= 32) & (levels < 224)
        covariance = np.mean(covariances[index], axis=0)
        spread = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        neighbours = None
        if counted:
            power = np.maximum((spectra[index] - first_spectrum) / counted, 0.0)
            correlation = np.fft.fftshift(np.fft.ifft2(power).real)
            correlation /= correlation.max()
            middle = SPECTRUM_PATCH // 2
            neighbours = [float(correlation[middle, middle + lag]) for lag in (1, 2, 3)]
        results.append(
            {
                "shift": shifts[index],
                "patches": int(len(levels)),
                "midtones": float(np.sqrt(max(np.median(grains[midtones]), 0))) if midtones.any() else None,
                "tones": bins,
                "channels": spread.tolist(),
                "channel_correlation": float(
                    (covariance / np.outer(spread, spread))[np.triu_indices(3, 1)].mean()
                ),
                "neighbours": neighbours,
            }
        )
    return results


def shrunk(image: NDArray[np.float32], long_edge: int) -> NDArray[np.float32]:
    """Return *image* shrunk to *long_edge* with OpenCV's area average."""
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def main() -> int:
    """Measure the photos named on the command line and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("without", type=Path, help="the photo without grain")
    parser.add_argument("with_grain", type=Path, nargs="+", help="photos of the same scene with grain")
    parser.add_argument("--no-shrunk", action="store_true", help="skip the smaller sizes")
    arguments = parser.parse_args()

    first = load(arguments.without)
    others = [load(path) for path in arguments.with_grain]
    for path, result in zip(arguments.with_grain, measure(first, others)):
        print(f"\n{path.name}  (shift {result['shift']}, {result['patches']} patches)")
        print(f"  midtones (32-223): {result['midtones']:.2f} levels")
        for low, high, count, value in result["tones"]:  # type: ignore[union-attr]
            print(f"  {low:3d}-{high:3d}: {value:6.2f}  (n={count})")
        channels = ", ".join(f"{value:.2f}" for value in result["channels"])  # type: ignore[union-attr]
        print(f"  channels R, G, B: {channels}; correlation {result['channel_correlation']:.3f}")
        if result["neighbours"] is not None:
            lags = ", ".join(f"{value:+.3f}" for value in result["neighbours"])  # type: ignore[union-attr]
            print(f"  neighbours 1, 2, 3 px apart: {lags}")
    if not arguments.no_shrunk:
        for long_edge in SHRUNK_LONG_EDGES:
            small = [shrunk(image, long_edge) for image in [first, *others]]
            scale = long_edge / max(first.shape[:2])
            values = [result["midtones"] for result in measure(small[0], small[1:], blur=max(3.0 * scale, 1.0))]
            text = ", ".join("-" if value is None else f"{value:.2f}" for value in values)
            print(f"\nshrunk to {long_edge}: midtones {text} levels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
