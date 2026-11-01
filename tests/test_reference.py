"""Reference comparison: the colours must not change by accident.

PLAN.md section 11 asks for this, and CLAUDE.md section 8 asks for it to be
run whenever ``core/pipeline.py`` is touched: fixed photos are developed at
fixed parameters and compared against what they gave before, so that a change
meant to make the pipeline faster or tidier cannot quietly change what it
produces.

What is stored for each case is in ``tests/reference``: the SHA-256 of the
whole developed image, which catches any difference at all, and a small
fingerprint, which says how large the difference is in steps of 255 when the
hash does not match. The photos themselves are not in the repository, so a
case whose photo is missing is skipped.

To take on a change that is intended - new coefficients, say - delete the
stored file or run with ``TINTAPHOTO_WRITE_REFERENCE=1`` and commit the
result, saying in the commit message why the colours moved.
"""

from __future__ import annotations

import hashlib
import json
import os

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import color_management as cm
from core import decoder, pipeline
from core.edit_state import EditState
from core.geometry import Geometry, Rect
from core.lens import LensSettings

SAMPLES = Path(__file__).resolve().parent / "samples"
REFERENCE = Path(__file__).resolve().parent / "reference"

#: Long edge the comparison develops at. Small enough to keep the suite quick,
#: large enough that every step of section 6 runs on real detail.
LONG_EDGE = 1024

#: How wide the stored fingerprint is. It is only there to say how far a
#: difference reaches once the hash has already said there is one.
FINGERPRINT = 16

#: Whether the stored files are being written rather than checked.
WRITING = os.environ.get("TINTAPHOTO_WRITE_REFERENCE") == "1"

#: The photos the comparison runs on: an uncompressed RAW, a compressed RAW
#: and the Adobe RGB JPEG, which between them cover both decoders, both
#: colour paths of section 6 and both of LibRaw's ARW readers. The RAW and
#: the JPEG are the same shot, recorded as a RAW+JPEG pair by the camera.
PHOTOS = ("_DSC2828.ARW", "_DSC2823.ARW", "_DSC2828.JPG")

#: The matrix LibRaw reports for the camera this was written against, and
#: one of the sample photos: enough to put a made up photo through the
#: RAW half of section 6 without needing a file.
CAMERA_RGB_XYZ_MATRIX = np.array(
    [
        [0.7374, -0.2389, -0.0551],
        [-0.5435, 1.3162, 0.2519],
        [-0.1006, 0.1795, 0.6552],
        [0.0000, 0.0000, 0.0000],
    ]
)


def _states(photo: decoder.DecodedPhoto) -> dict[str, tuple[EditState, Geometry]]:
    """Return the cases every photo is developed in."""
    whole = Geometry()
    neutral = EditState.for_photo(photo.is_raw, *pipeline.shot_white_balance(photo))
    return {
        "defaults": (neutral, whole),
        "moved": (
            replace(
                neutral,
                exposure=0.5,
                contrast=30.0,
                highlights=-40.0,
                shadows=40.0,
                whites=15.0,
                blacks=-20.0,
                vibrance=30.0,
                saturation=10.0,
                vignette=-30.0,
                vignette_midpoint=40.0,
            ),
            whole,
        ),
        "extreme": (
            replace(
                neutral,
                temperature=neutral.temperature + (2000.0 if photo.is_raw else 60.0),
                tint=neutral.tint + (30.0 if photo.is_raw else 40.0),
                exposure=-1.5,
                contrast=100.0,
                highlights=-100.0,
                shadows=100.0,
                whites=-100.0,
                blacks=100.0,
                vibrance=100.0,
                saturation=80.0,
                vignette=60.0,
                vignette_midpoint=20.0,
            ),
            whole,
        ),
        # The detail step of section 6 step 2.5, all four sliders at once.
        "detail": (
            replace(
                neutral,
                sharpening=90.0,
                sharpen_masking=30.0,
                luminance_noise=40.0,
                colour_noise=50.0,
            ),
            whole,
        ),
        # Geometry as well, so that step 2 is held to the same promise: the
        # crop and the angle together, which is what rebases the crop box.
        "cropped": (
            replace(neutral, exposure=0.3, saturation=20.0),
            Geometry(quarter_turns=1, angle=4.5, crop=Rect(0.1, 0.15, 0.7, 0.6)),
        ),
        # The lens correction of section 6 step 1.5, as a photo opens with it:
        # on for a RAW whose lens has a profile, off for a JPEG.
        "lens": (neutral, Geometry(lens=LensSettings.for_photo(photo.is_raw))),
    }


def _fingerprint(image: np.ndarray) -> np.ndarray:
    """Return the developed image shrunk to a few pixels, for reporting."""
    return cv2.resize(
        image, (FINGERPRINT, FINGERPRINT), interpolation=cv2.INTER_AREA
    ).astype(np.int16)


def _describe(stored: np.ndarray, now: np.ndarray) -> str:
    """Return how far the colours have moved, in steps of 255."""
    difference = np.abs(now.astype(np.int16) - stored)
    return (
        f"largest difference {difference.max()} of 255, "
        f"average {difference.mean():.3f}, "
        f"{(difference > 0).mean() * 100:.1f}% of the fingerprint moved"
    )


def _case_file(name: str, case: str) -> Path:
    return REFERENCE / f"{name}.{case}.json"


def _cases() -> list[tuple[str, str]]:
    """Return every photo and case there is a stored file for."""
    return [
        (name, case)
        for name in PHOTOS
        for case in ("defaults", "moved", "extreme", "detail", "cropped", "lens")
        if WRITING or _case_file(name, case).is_file()
    ]


_decoded: dict[str, decoder.DecodedPhoto] = {}


def _photo(name: str) -> decoder.DecodedPhoto:
    """Return the sample, decoded once for the whole session."""
    if name not in _decoded:
        _decoded[name] = decoder.decode_proxy(SAMPLES / name, LONG_EDGE)
    return _decoded[name]


@pytest.mark.parametrize(("name", "case"), _cases())
def test_the_colours_are_what_they_were(name: str, case: str) -> None:
    """A photo developed at fixed parameters comes out as it did before."""
    if not (SAMPLES / name).is_file():
        pytest.skip(f"{name} is not in tests/samples")

    photo = _photo(name)
    state, geometry = _states(photo)[case]
    developed = pipeline.render(photo, state, geometry)
    digest = hashlib.sha256(np.ascontiguousarray(developed).tobytes()).hexdigest()

    path = _case_file(name, case)
    if WRITING:
        REFERENCE.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "long_edge": LONG_EDGE,
                    "shape": list(developed.shape),
                    "sha256": digest,
                    "fingerprint": _fingerprint(developed).tolist(),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["long_edge"] == LONG_EDGE, "the stored file is for another size"
    assert list(developed.shape) == stored["shape"], (
        f"{name} {case}: the developed size is now {developed.shape}, "
        f"was {tuple(stored['shape'])}"
    )
    if digest == stored["sha256"]:
        return
    moved = _describe(
        np.asarray(stored["fingerprint"], dtype=np.int16), _fingerprint(developed)
    )
    pytest.fail(
        f"{name} {case}: the colours have changed - {moved}. If the change is "
        f"meant, run the suite once with TINTAPHOTO_WRITE_REFERENCE=1 and say "
        f"in the commit message why."
    )


def test_the_result_does_not_depend_on_the_band_size() -> None:
    """Cutting the photo into bands must not be visible in what comes out.

    The band is sized to fit in cache, which is a number tuned per machine
    and per photo size (``BAND_TARGET_BYTES``). Nothing about the result may
    depend on it: every step inside a band is per pixel, and the two that
    look at a pixel's surroundings are worked out over the whole photo
    before the bands are cut.
    """
    generator = np.random.default_rng(20261022)
    photo = decoder.DecodedPhoto(
        image=generator.random((200, 120, 3), dtype=np.float32),
        is_raw=True,
        clipped=generator.random((200, 120)) > 0.98,
        camera_to_xyz=cm.camera_to_xyz_matrix(CAMERA_RGB_XYZ_MATRIX),
        rgb_xyz_matrix=CAMERA_RGB_XYZ_MATRIX,
        camera_whitebalance=(2.3359, 1.0, 1.6406),
    )
    state = replace(
        EditState.for_photo(True, *pipeline.shot_white_balance(photo)),
        exposure=0.4,
        contrast=25.0,
        highlights=-50.0,
        shadows=60.0,
        whites=20.0,
        blacks=-30.0,
        vibrance=40.0,
        saturation=15.0,
        vignette=-40.0,
    )

    developed = {}
    row_bytes = photo.image.itemsize * photo.image.shape[1] * photo.image.shape[2]
    for size in (4 * 1024, 64 * 1024, 768 * 1024, 64 * 1024 * 1024):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pipeline, "BAND_TARGET_BYTES", size)
            developed[pipeline.band_rows(row_bytes)] = pipeline.develop(photo, state)

    # The sizes above have to give genuinely different cuts, or the test
    # would pass without having compared anything.
    assert len(developed) >= 3, f"only these band heights were tried: {developed.keys()}"
    first, *rest = developed.values()
    for other in rest:
        assert np.array_equal(first, other), (
            "the photo comes out differently depending on how it is cut into "
            "bands, so a step is looking beyond its own band"
        )


def test_there_is_something_to_compare_against() -> None:
    """The stored files must not quietly disappear and leave nothing checked."""
    if not any((SAMPLES / name).is_file() for name in PHOTOS):
        pytest.skip("none of the reference photos is in tests/samples")
    assert _cases(), (
        "no reference file in tests/reference: run the suite once with "
        "TINTAPHOTO_WRITE_REFERENCE=1 to record what the pipeline gives today"
    )
