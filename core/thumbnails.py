"""Filmstrip thumbnails (PLAN.md section 4 and F-02).

A thumbnail has to appear quickly above all else, so a RAW file uses the JPEG
preview the camera already put inside it, which takes about a millisecond
instead of a second of demosaicing. Only when a file carries no preview is it
decoded properly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from core.decoder import (
    PROXY_LONG_EDGE,
    DecodeError,
    apply_orientation,
    decode_proxy,
    embedded_preview,
    flattened_pixels,
    is_raw,
)
from core.pipeline import develop

logger = logging.getLogger(__name__)

#: Long edge of a filmstrip thumbnail (PLAN.md section 4).
THUMBNAIL_LONG_EDGE: Final[int] = 256

#: Quality the thumbnails are stored at in the disk cache.
THUMBNAIL_JPEG_QUALITY: Final[int] = 85

_EXIF_ORIENTATION_TAG: Final[int] = 274


def _fit(image: NDArray[np.uint8], long_edge: int) -> NDArray[np.uint8]:
    """Shrink *image* so that its long edge is at most *long_edge*."""
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= long_edge:
        return image
    scale = long_edge / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _encoded_thumbnail(path: Path, long_edge: int) -> NDArray[np.uint8]:
    """Read a JPEG or a PNG at a reduced size, skipping most of the decoding.

    ``draft`` is what a JPEG is read at a fraction of its size by; a PNG has
    no such shortcut and is simply read and shrunk. Either way the pixels are
    only ever an 8 bit thumbnail, so a 16 bit PNG losing its low byte here
    costs nothing that can be seen.
    """
    with Image.open(path) as image:
        orientation = int(image.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        image.draft("RGB", (long_edge, long_edge))
        pixels = flattened_pixels(image)
    return apply_orientation(_fit(pixels, long_edge), orientation)


def make(path: Path, long_edge: int = THUMBNAIL_LONG_EDGE) -> NDArray[np.uint8]:
    """Return a thumbnail of the photo at *path* as 8 bit RGB.

    Raises :class:`DecodeError` if the photo cannot be read at all, so that
    the filmstrip can mark it instead of showing nothing (F-06).
    """
    try:
        if is_raw(path):
            preview = embedded_preview(path)
            if preview is not None:
                return _fit(preview, long_edge)
            logger.info("%s has no embedded preview, decoding it instead", path.name)
            return _fit(develop(decode_proxy(path, PROXY_LONG_EDGE)), long_edge)
        return _encoded_thumbnail(path, long_edge)
    except DecodeError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise DecodeError(f"no thumbnail for {path.name}: {error}") from error


def encode(thumbnail: NDArray[np.uint8]) -> bytes | None:
    """Encode a thumbnail as JPEG for the disk cache."""
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), THUMBNAIL_JPEG_QUALITY],
    )
    return buffer.tobytes() if ok else None


def decode(data: bytes) -> NDArray[np.uint8] | None:
    """Decode a thumbnail that came from the disk cache."""
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
