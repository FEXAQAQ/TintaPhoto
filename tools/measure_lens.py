"""Measure this program's lens correction against the camera's own JPEG.

How the corrections of PLAN.md section 6 step 1.5 were checked, kept so that
they can be checked again, on another body or another lens. Shoot RAW+JPEG
with the camera's own lens corrections switched on, then give the pairs::

    .venv\\Scripts\\python.exe tools\\measure_lens.py A.RAF A.JPG B.RAF B.JPG

For each pair it reports:

* distortion: features of the RAW, developed with the correction off and
  on, are matched to the JPEG and lined up by a shift, a turn and a zoom; how
  far the matches still sit apart is reported by distance from the centre,
  in pixels of the full size photo. What the zoom cannot explain is the
  distortion left over, and it grows towards the corners;
* vignetting: how much the camera brightened each ring of the frame against
  the middle, found by fitting the JPEG's tone curve and a gain per ring
  together, next to the gain this program applies there.

Scenes with detail right into the corners measure best. The vignetting needs
a dozen or so pairs, or a flat wall, to be read to a percent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import decoder, lens, pipeline  # noqa: E402
from core.decoder import DecodedPhoto  # noqa: E402
from core.edit_state import EditState  # noqa: E402
from core.geometry import Geometry  # noqa: E402
from core.lens import LensSettings  # noqa: E402

#: The long edge the photos are compared at.
LONG_EDGE: Final[int] = 2560
#: The rings of the report, as fractions of half the diagonal.
RINGS: Final[tuple[float, ...]] = (0.0, 0.3, 0.6, 0.8, 1.01)
#: The rings the vignetting is fitted in.
VIGNETTING_RINGS: Final[int] = 10
#: The features matched in each photo.
FEATURES: Final[int] = 8000


def develop(photo: DecodedPhoto, settings: LensSettings) -> NDArray[np.uint8]:
    """Return *photo* developed as it opens, with *settings* for the lens."""
    state = EditState.for_photo(True, *pipeline.shot_white_balance(photo))
    return pipeline.render(photo, state, Geometry(lens=settings))


def matches(
    ours: NDArray[np.uint8], theirs: NDArray[np.uint8]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the points of *ours* and *theirs* that show the same thing."""
    sift = cv2.SIFT_create(FEATURES)
    points_ours, described_ours = sift.detectAndCompute(cv2.cvtColor(ours, cv2.COLOR_RGB2GRAY), None)
    points_theirs, described_theirs = sift.detectAndCompute(cv2.cvtColor(theirs, cv2.COLOR_RGB2GRAY), None)
    good = [
        pair[0]
        for pair in cv2.BFMatcher().knnMatch(described_ours, described_theirs, k=2)
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    first = np.float64([points_ours[match.queryIdx].pt for match in good])
    second = np.float64([points_theirs[match.trainIdx].pt for match in good])
    return first, second


def leftover(
    ours: NDArray[np.uint8], theirs: NDArray[np.uint8], to_full: float
) -> tuple[float, list[tuple[float, int]]]:
    """Return the zoom between two developments and the distance left per ring."""
    first, second = matches(ours, theirs)
    matrix, _inliers = cv2.estimateAffinePartial2D(
        first.astype(np.float32), second.astype(np.float32), ransacReprojThreshold=6.0, maxIters=5000
    )
    moved = first @ matrix[:, :2].T + matrix[:, 2]
    error = np.linalg.norm(moved - second, axis=1) * to_full
    height, width = ours.shape[:2]
    radius = np.hypot(first[:, 0] - (width - 1) / 2, first[:, 1] - (height - 1) / 2) / (
        np.hypot(width, height) / 2
    )
    rings = []
    for low, high in zip(RINGS, RINGS[1:]):
        inside = (radius >= low) & (radius < high)
        rings.append((float(np.median(error[inside])) if inside.any() else float("nan"), int(inside.sum())))
    return float(np.hypot(*matrix[0, :2])), rings


def brightening(
    photo: DecodedPhoto, theirs: NDArray[np.uint8], settings: LensSettings
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return, per ring, the camera's gain and this program's, against the middle."""
    straight = lens.correct(photo, LensSettings(enabled=True, distortion=settings.distortion, vignetting=0.0))
    flat = lens.correct(
        DecodedPhoto(
            image=np.ones_like(photo.image),
            is_raw=True,
            clipped=photo.clipped,
            lens=photo.lens,
            shrunk_from=photo.shrunk_from,
        ),
        LensSettings(enabled=True, distortion=settings.distortion, vignetting=settings.vignetting),
    )
    height, width = straight.image.shape[:2]
    ours = develop(photo, LensSettings(enabled=True, distortion=settings.distortion, vignetting=0.0))
    first, second = matches(cv2.resize(ours, (width, height), interpolation=cv2.INTER_AREA), theirs)
    matrix, _inliers = cv2.estimateAffinePartial2D(
        second.astype(np.float32), first.astype(np.float32), ransacReprojThreshold=4.0, maxIters=5000
    )
    size = (theirs.shape[1], theirs.shape[0])
    scene = cv2.warpAffine(straight.image.mean(axis=2), matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    applied = cv2.warpAffine(flat.image[:, :, 1], matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    camera = theirs.astype(np.float32) / 255.0
    camera = np.where(camera <= 0.04045, camera / 12.92, ((camera + 0.055) / 1.055) ** 2.4).mean(axis=2)
    scene = cv2.GaussianBlur(scene, (0, 0), 2.0)
    camera = cv2.GaussianBlur(camera, (0, 0), 2.0)

    rows, columns = np.mgrid[0 : size[1], 0 : size[0]]
    radius = np.hypot(columns - (size[0] - 1) / 2, rows - (size[1] - 1) / 2) / (np.hypot(*size) / 2)
    usable = (scene > 1e-3) & (camera > 0.01) & (camera < 0.95) & (radius < 1.0) & (applied > 0)
    ours_log, camera_log = np.log(scene[usable]), np.log(camera[usable])
    ring = np.minimum((radius[usable] * VIGNETTING_RINGS).astype(int), VIGNETTING_RINGS - 1)

    # The JPEG is a tone curve of the scene times the camera's gain; fit both
    # by turns, the curve as a median per bin and the gain per ring.
    gain = np.zeros(VIGNETTING_RINGS)
    for _round in range(15):
        lifted = ours_log + gain[ring]
        edges = np.quantile(lifted, np.linspace(0.0, 1.0, 61))
        bins = np.clip(np.searchsorted(edges, lifted) - 1, 0, 59)
        curve_in = np.array([np.median(lifted[bins == index]) for index in range(60)])
        curve_out = np.maximum.accumulate(np.array([np.median(camera_log[bins == index]) for index in range(60)]))
        needed = np.interp(camera_log, curve_out, curve_in)
        inside_curve = (camera_log > curve_out[2]) & (camera_log < curve_out[-3])
        for index in range(VIGNETTING_RINGS):
            chosen = (ring == index) & inside_curve
            if chosen.sum() > 100:
                gain[index] = np.median(needed[chosen] - ours_log[chosen])
        gain -= gain[1]
    ours_gain = np.array(
        [
            np.median(applied[usable][ring == index]) if (ring == index).any() else np.nan
            for index in range(VIGNETTING_RINGS)
        ]
    )
    return np.exp(gain), ours_gain / ours_gain[1]


def main() -> int:
    """Measure the pairs named on the command line and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pairs", type=Path, nargs="+", help="RAW and JPEG, RAW and JPEG, ...")
    arguments = parser.parse_args()
    if len(arguments.pairs) % 2:
        parser.error("give the photos in pairs: a RAW, then the camera's JPEG of it")

    for raw_path, jpeg_path in zip(arguments.pairs[0::2], arguments.pairs[1::2]):
        photo = decoder.decode_proxy(raw_path, LONG_EDGE)
        theirs = decoder.decode_proxy(jpeg_path, LONG_EDGE).encoded
        if theirs is None:
            print(f"{jpeg_path.name} is not an 8 bit JPEG")
            continue
        profile = photo.lens
        source = "none" if profile is None else ("camera" if profile.from_camera else "database")
        print(f"\n{raw_path.name}: lens {profile.name if profile else '-'} ({source})")
        to_full = photo.full_size[0] / theirs.shape[1]
        for label, settings in (
            ("off", LensSettings(enabled=False)),
            ("on ", LensSettings(enabled=True)),
        ):
            ours = develop(photo, settings)
            if ours.shape[:2] != theirs.shape[:2]:
                ours = cv2.resize(ours, (theirs.shape[1], theirs.shape[0]), interpolation=cv2.INTER_AREA)
            zoom, rings = leftover(ours, theirs, to_full)
            text = "  ".join(
                f"{low:.1f}-{min(high, 1.0):.1f}: {value:5.2f} px (n={count})"
                for (low, high), (value, count) in zip(zip(RINGS, RINGS[1:]), rings)
            )
            print(f"  distortion {label}  zoom {zoom:.4f}  {text}")
        if profile is not None and profile.vignetting is not None:
            camera_gain, ours_gain = brightening(photo, theirs, LensSettings(enabled=True))
            centres = (np.arange(VIGNETTING_RINGS) + 0.5) / VIGNETTING_RINGS
            print("  vignetting gain against the ring at 0.15 (the middle ring is too small to trust):")
            print("    radius  " + " ".join(f"{value:5.2f}" for value in centres))
            print("    camera  " + " ".join(f"{value:5.3f}" for value in camera_gain))
            print("    ours    " + " ".join(f"{value:5.3f}" for value in ours_gain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
