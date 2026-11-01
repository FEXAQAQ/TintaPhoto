"""Finding the faces in a photo (PLAN.md 6 step 2.6).

With YuNet, the face detector that ships with OpenCV, and its model file in
``resources/models`` (MIT licence). It is run once for each photo, on a copy
of the photo developed the way it opens, and what it finds is handed to every
later step as :class:`core.faces.Face`: the preview, the zoomed in window and
the export all use the same faces, so they cannot disagree on where the skin
is.

Two passes: the whole photo at a small size for the faces that fill much of
it, then the photo at twice that size in overlapping tiles for the small
faces of a group, which the first pass is too coarse to see.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core import lens, pipeline
from core.decoder import PROXY_LONG_EDGE, DecodedPhoto, resize_photo
from core.edit_state import EditState
from core.geometry import Geometry
from core.faces import Face, face_size, measure_tone

logger = logging.getLogger(__name__)

#: The detector's weights, bundled with the program.
MODEL_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent
    / "resources"
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

#: The long edge the whole photo is looked at, in the first pass. YuNet was
#: trained on pictures about this size; looked at much larger, a face that
#: fills the frame is missed.
WHOLE_LONG_EDGE: Final[int] = 640
#: The long edge of the copy the second pass cuts into tiles, and the size
#: of each tile and of their overlap.
TILED_LONG_EDGE: Final[int] = 1280
TILE: Final[int] = 640
TILE_OVERLAP: Final[int] = 160
#: The second pass only keeps faces narrower than this, in pixels of its
#: copy: a larger one was seen by the first pass, and a tile may hold only
#: part of it.
TILE_LARGEST_FACE: Final[int] = 160
#: How sure the detector has to be. Lower, and a knot in wood or a pattern
#: on a shirt now and then passes for a face.
SCORE_THRESHOLD: Final[float] = 0.75
#: Two finds overlapping by more than this are the same face.
OVERLAP_THRESHOLD: Final[float] = 0.3


@lru_cache(maxsize=1)
def _model_bytes() -> bytes:
    """Return the model file, read once.

    Read by Python rather than by OpenCV, which cannot open a path holding
    characters outside the system's code page - a program installed under a
    folder with a Chinese name would otherwise find no faces at all.
    """
    return MODEL_PATH.read_bytes()


def _detector(size: tuple[int, int]) -> cv2.FaceDetectorYN:
    """Return a detector for pictures of *size*; one per call, for thread safety."""
    detector = cv2.FaceDetectorYN.create(
        "onnx",
        np.frombuffer(_model_bytes(), dtype=np.uint8),
        np.zeros(0, dtype=np.uint8),
        size,
        SCORE_THRESHOLD,
        OVERLAP_THRESHOLD,
        5000,
    )
    return detector


def _detect(picture: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Return YuNet's rows for *picture* (BGR): box, five points, score."""
    height, width = picture.shape[:2]
    detector = _detector((width, height))
    _, found = detector.detect(np.ascontiguousarray(picture))
    if found is None:
        return np.zeros((0, 15), dtype=np.float32)
    return np.asarray(found, dtype=np.float32)


def _scaled(rows: NDArray[np.float32], factor: float, dx: float, dy: float) -> NDArray[np.float32]:
    """Return *rows* measured in a picture *factor* times larger, moved by dx, dy."""
    out = rows.copy()
    # Boxes: corner and size. Points: positions. Both scale about the outer
    # edge of the picture, which is where OpenCV's pixel corner lies.
    out[:, 0] = (rows[:, 0] + 0.5) * factor - 0.5 + dx
    out[:, 1] = (rows[:, 1] + 0.5) * factor - 0.5 + dy
    out[:, 2:4] = rows[:, 2:4] * factor
    for index in range(4, 14, 2):
        out[:, index] = (rows[:, index] + 0.5) * factor - 0.5 + dx
        out[:, index + 1] = (rows[:, index + 1] + 0.5) * factor - 0.5 + dy
    return out


def _tile_starts(length: int) -> list[int]:
    """Return where the tiles along one edge of *length* pixels start."""
    if length <= TILE:
        return [0]
    step = TILE - TILE_OVERLAP
    starts = list(range(0, length - TILE, step))
    starts.append(length - TILE)
    return starts


def find_in_picture(picture: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Return every face in *picture* (BGR, about TILED_LONG_EDGE), as YuNet rows."""
    height, width = picture.shape[:2]
    rows: list[NDArray[np.float32]] = []

    factor = max(height, width) / WHOLE_LONG_EDGE
    if factor > 1.0:
        small = cv2.resize(
            picture,
            (max(1, round(width / factor)), max(1, round(height / factor))),
            interpolation=cv2.INTER_AREA,
        )
        rows.append(_scaled(_detect(small), width / small.shape[1], 0.0, 0.0))
        for top in _tile_starts(height):
            for left in _tile_starts(width):
                tile = picture[top : top + TILE, left : left + TILE]
                found = _detect(tile)
                found = found[found[:, 2] < TILE_LARGEST_FACE]
                rows.append(_scaled(found, 1.0, float(left), float(top)))
    else:
        rows.append(_detect(picture))

    every = np.concatenate(rows) if rows else np.zeros((0, 15), dtype=np.float32)
    if len(every) == 0:
        return every
    boxes = [[float(v) for v in row[:4]] for row in every]
    scores = [float(row[14]) for row in every]
    kept = cv2.dnn.NMSBoxes(boxes, scores, SCORE_THRESHOLD, OVERLAP_THRESHOLD)
    kept = np.asarray(kept, dtype=np.int64).reshape(-1)
    # In order across the photo, so that the same photo always lists its
    # faces the same way.
    chosen = every[np.sort(kept)]
    return chosen[np.lexsort((chosen[:, 1], chosen[:, 0]))]


def find(photo: DecodedPhoto) -> tuple[Face, ...]:
    """Return the faces in *photo*, which has had its lens corrected.

    *photo* may be any copy of the photo - the proxy is what the program
    hands over - since it is looked at at a size of its own. Anything that
    goes wrong is logged and treated as no faces, so that a photo whose
    faces cannot be found still opens and develops as usual.
    """
    try:
        view = resize_photo(photo, TILED_LONG_EDGE)
        # Developed the way the photo opens, without the sharpening and the
        # colour noise reduction a RAW rests with: they change nothing the
        # detector can see and are the slowest part of developing it.
        state = replace(
            EditState.for_photo(photo.is_raw, *pipeline.shot_white_balance(photo)),
            sharpening=0.0,
            colour_noise=0.0,
        )
        rgb = pipeline.develop(view, state)
        rows = find_in_picture(np.ascontiguousarray(rgb[..., ::-1]))
    except Exception:  # noqa: BLE001 - a photo without faces is still a photo
        logger.exception("Looking for faces failed")
        return ()

    view_height, view_width = view.image.shape[:2]
    height, width = photo.image.shape[:2]
    faces: list[Face] = []
    for row in rows:
        middle = row[0:2] + row[2:4] / 2.0
        points = np.vstack((row[4:14].reshape(5, 2), middle[None, :])).astype(np.float64)
        on_photo = (points + 0.5) * np.array([width / view_width, height / view_height]) - 0.5
        tone = measure_tone(photo.image, on_photo)
        if tone is None:
            continue
        found = Face(points=(), tone=tone).moved(points, (view_width, view_height))
        size = face_size(found, (view_width, view_height))
        faces.append(replace(found, width=float(row[2]) / size, height=float(row[3]) / size))
    return tuple(faces)


def attach(
    photo: DecodedPhoto,
    state: EditState | None,
    geometry: Geometry | None,
    faces: tuple[Face, ...] | None = None,
) -> DecodedPhoto:
    """Return *photo* carrying its faces, when *state* has a use for them.

    *faces* are the ones already found for this photo with the same lens
    correction, which is how the export uses exactly what the preview did;
    without them they are looked for here, on a proxy sized copy, so that a
    photo exported or put in a collage before anybody looked still gets its
    skin seen to. A photo whose face sliders are at rest comes back as it is.
    """
    if state is None or not state.uses_faces:
        return photo
    if faces is None:
        settings = geometry.lens if geometry is not None else None
        faces = find(lens.correct(resize_photo(photo, PROXY_LONG_EDGE), settings))
    return replace(photo, faces=faces)
