"""Photo information for the info tab (PLAN.md section 3.9).

Values are returned as numbers and short tokens, never as text for the
screen: the interface turns them into words through ``tr()``, because this
layer knows nothing about languages.

For a JPEG, Pillow reads everything. For a RAW file, LibRaw gives the
exposure, the lens and the timestamp, but not the camera name, the exposure
compensation or the camera's own settings, so those few tags are read
directly out of the file, which is a TIFF underneath. That keeps the project
to the libraries PLAN.md section 4 lists.

Not every RAW file is a TIFF underneath: a RAF begins with a signature of
its own, and the tags are nowhere the TIFF reader looks. The preview the camera stored
in it is a whole JPEG with its own EXIF, though, and LibRaw hands that over,
so the tags come from there instead when the file's own header has none.
"""

from __future__ import annotations

import io
import logging
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import rawpy
from PIL import Image, UnidentifiedImageError

from core.decoder import is_raw, sensor_frame_size

logger = logging.getLogger(__name__)

# TIFF and EXIF tags that LibRaw does not report.
_TAG_MAKE: Final[int] = 271
_TAG_MODEL: Final[int] = 272
_TAG_ORIENTATION: Final[int] = 274
_TAG_EXIF_IFD: Final[int] = 34665
_TAG_EXPOSURE_TIME: Final[int] = 33434
_TAG_F_NUMBER: Final[int] = 33437
_TAG_EXPOSURE_BIAS: Final[int] = 37380
_TAG_METERING_MODE: Final[int] = 37383
_TAG_FLASH: Final[int] = 37385
_TAG_FOCAL_LENGTH: Final[int] = 37386
_TAG_ISO: Final[int] = 34855
_TAG_DATE_TAKEN: Final[int] = 36867
_TAG_WHITE_BALANCE: Final[int] = 41987
_TAG_LENS_MODEL: Final[int] = 42036
_TAG_FOCAL_LENGTH_35MM: Final[int] = 41989

#: Size in bytes of each TIFF field type, indexed by the type number.
_FIELD_SIZES: Final[dict[int, int]] = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}

#: How much of a RAW file to read when looking for its tags.
_HEADER_BYTES: Final[int] = 1024 * 1024
_MAX_IFD_ENTRIES: Final[int] = 512

#: EXIF metering mode numbers, as tokens the interface can translate.
_METERING_MODES: Final[dict[int, str]] = {
    0: "unknown",
    1: "average",
    2: "centre_weighted",
    3: "spot",
    4: "multi_spot",
    5: "pattern",
    6: "partial",
    255: "other",
}

_WHITE_BALANCE_MODES: Final[dict[int, str]] = {0: "auto", 1: "manual"}

_DATE_TAKEN_FORMAT: Final[str] = "%Y:%m:%d %H:%M:%S"

#: EXIF orientations that stand the photo on its side, so that its stored
#: width and height are the other way round from how it is shown.
_TURNED_ORIENTATIONS: Final[frozenset[int]] = frozenset({5, 6, 7, 8})


@dataclass(frozen=True)
class PhotoMetadata:
    """What is known about a photo, in units rather than in words."""

    path: Path
    file_size: int
    file_format: str
    width: int | None = None
    height: int | None = None
    camera: str | None = None
    lens: str | None = None
    aperture: float | None = None
    shutter_seconds: float | None = None
    iso: int | None = None
    focal_length_mm: float | None = None
    exposure_compensation: float | None = None
    #: Tokens for the interface to translate, not text to show.
    white_balance: str | None = None
    metering: str | None = None
    flash_fired: bool | None = None
    captured_at: datetime | None = None

    @property
    def megapixels(self) -> float | None:
        """Return the pixel count in millions, or ``None`` if unknown."""
        if self.width is None or self.height is None:
            return None
        return self.width * self.height / 1_000_000


# --------------------------------------------------------------------------
# A small TIFF reader, enough for the tags LibRaw does not report
# --------------------------------------------------------------------------


def _read_tiff_tags(data: bytes) -> dict[int, Any]:
    """Return the tags of a TIFF header, including its EXIF sub directory."""
    if len(data) < 8:
        return {}
    if data[:2] == b"II":
        byte_order = "<"
    elif data[:2] == b"MM":
        byte_order = ">"
    else:
        return {}

    tags: dict[int, Any] = {}
    try:
        offset = struct.unpack(byte_order + "I", data[4:8])[0]
        _read_ifd(data, offset, byte_order, tags, depth=0)
    except (struct.error, IndexError, UnicodeDecodeError):
        logger.info("A photo's TIFF header could not be read in full")
    return tags


def _read_ifd(
    data: bytes, offset: int, byte_order: str, tags: dict[int, Any], depth: int
) -> None:
    """Read one image file directory into *tags*, following the EXIF pointer."""
    if depth > 2 or offset <= 0 or offset + 2 > len(data):
        return
    count = struct.unpack(byte_order + "H", data[offset : offset + 2])[0]
    if count > _MAX_IFD_ENTRIES:
        return

    for index in range(count):
        entry = offset + 2 + index * 12
        if entry + 12 > len(data):
            return
        tag, field_type, length = struct.unpack(byte_order + "HHI", data[entry : entry + 8])
        size = _FIELD_SIZES.get(field_type, 0) * length
        if size == 0:
            continue
        if size <= 4:
            start = entry + 8
        else:
            start = struct.unpack(byte_order + "I", data[entry + 8 : entry + 12])[0]
        if start + size > len(data):
            continue

        value = _read_value(data[start : start + size], field_type, byte_order)
        if value is None:
            continue
        tags.setdefault(tag, value)
        if tag == _TAG_EXIF_IFD and isinstance(value, int):
            _read_ifd(data, value, byte_order, tags, depth + 1)


def _read_value(raw: bytes, field_type: int, byte_order: str) -> Any:
    """Decode one TIFF field value."""
    if field_type == 2:
        return raw.split(b"\0")[0].decode("ascii", "replace").strip()
    if field_type in (3, 8):
        code = "H" if field_type == 3 else "h"
        return struct.unpack(byte_order + code, raw[:2])[0]
    if field_type in (4, 9):
        code = "I" if field_type == 4 else "i"
        return struct.unpack(byte_order + code, raw[:4])[0]
    if field_type in (5, 10):
        code = "II" if field_type == 5 else "ii"
        numerator, denominator = struct.unpack(byte_order + code, raw[:8])
        return numerator / denominator if denominator else None
    return None


# --------------------------------------------------------------------------
# Reading a photo's information
# --------------------------------------------------------------------------


def _parse_date(value: Any) -> datetime | None:
    """Parse an EXIF date, which is written as ``YYYY:MM:DD HH:MM:SS``."""
    text = _as_text(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text, _DATE_TAKEN_FORMAT)
    except ValueError:
        return None


def _camera_name(make: Any, model: Any) -> str | None:
    """Join the maker and the model without repeating the maker."""
    make = _as_text(make) or ""
    model = _as_text(model) or ""
    if not model:
        return make or None
    first = make.split()[0] if make else ""
    if first and model.upper().startswith(first.upper()):
        return model
    return f"{make} {model}".strip()


def _file_facts(path: Path) -> tuple[int, str]:
    """Return the size in bytes and the extension without its dot."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return size, path.suffix.lstrip(".").upper()


def _preview_tags(raw: rawpy.RawPy) -> dict[int, Any]:
    """Return the EXIF tags of the JPEG preview LibRaw carries, if any.

    This is the way in for a RAW file that is not a TIFF underneath, where
    :func:`_read_tiff_tags` finds nothing: the preview is a whole JPEG and
    carries the camera name and the camera's own settings in its own EXIF.
    """
    try:
        thumb = raw.extract_thumb()
    except (rawpy.LibRawError, rawpy.NotSupportedError) as error:
        logger.info("there is no preview to read tags from: %s", error)
        return {}
    if thumb.format is not rawpy.ThumbFormat.JPEG:
        return {}
    try:
        with Image.open(io.BytesIO(thumb.data)) as preview:
            exif = preview.getexif()
            tags = dict(exif)
            tags.update(dict(exif.get_ifd(_TAG_EXIF_IFD) or {}))
    except (OSError, ValueError, UnidentifiedImageError) as error:
        logger.info("the preview's tags could not be read: %s", error)
        return {}
    return tags


@dataclass(frozen=True)
class CameraFacts:
    """The camera's name and its 35 mm equivalent focal length, for the lens lookup."""

    maker: str = ""
    model: str = ""
    #: Zero when the file does not say.
    focal_length_35mm: float = 0.0


def camera_facts(path: Path, raw: rawpy.RawPy) -> CameraFacts:
    """Return what the lens lookup needs from a RAW file's own tags.

    *raw* is the file already opened by the decoder, whose preview is the
    way in for a file that is not a TIFF underneath. Never raises: what
    cannot be read is simply left empty (PLAN.md 6 step 1.5).
    """
    tags: dict[int, Any] = {}
    try:
        with path.open("rb") as handle:
            tags = _read_tiff_tags(handle.read(_HEADER_BYTES))
    except OSError as error:
        logger.info("%s could not be read for its tags: %s", path.name, error)
    if not tags:
        tags = _preview_tags(raw)
    return CameraFacts(
        maker=_as_text(tags.get(_TAG_MAKE)) or "",
        model=_as_text(tags.get(_TAG_MODEL)) or "",
        focal_length_35mm=_positive(_as_float(tags.get(_TAG_FOCAL_LENGTH_35MM))),
    )


def _positive(value: float | None) -> float:
    """Return *value* if it is a finite positive number, otherwise zero."""
    return value if value is not None and math.isfinite(value) and value > 0 else 0.0


def _raw_metadata(path: Path, tags: dict[int, Any]) -> tuple[dict[str, Any], dict[int, Any]]:
    """Return what LibRaw knows about a RAW file, and the tags it can add.

    The second dictionary is empty unless *tags* is, which is what a file
    whose header is not a TIFF leaves behind: the tags then come from the
    preview, and the caller lets the file's own tags win over them.
    """
    values: dict[str, Any] = {}
    preview: dict[int, Any] = {}
    try:
        with rawpy.imread(str(path)) as raw:
            width, height = sensor_frame_size(raw.sizes)
            turned = raw.sizes.flip in (5, 6)
            values["width"] = height if turned else width
            values["height"] = width if turned else height
            other = raw.other
            values["aperture"] = float(other.aperture) or None
            values["shutter_seconds"] = float(other.shutter_speed) or None
            values["iso"] = int(other.iso_speed) or None
            values["focal_length_mm"] = float(other.focal_length) or None
            values["captured_at"] = other.timestamp
            values["lens"] = _as_text(raw.lens.model)
            if not tags:
                preview = _preview_tags(raw)
    except (OSError, ValueError, rawpy.LibRawError) as error:
        logger.info("%s: LibRaw could not report its details: %s", path.name, error)
    # Anything LibRaw left out comes from the file's own tags.
    values.setdefault("lens", _as_text(tags.get(_TAG_LENS_MODEL)))
    return values, preview


def read(path: Path) -> PhotoMetadata:
    """Return what is known about the photo at *path*.

    Never raises: a photo whose details cannot be read still has a name, a
    size and a format, and the interface simply leaves out the rows it has
    nothing for (PLAN.md section 3.9).
    """
    size, extension = _file_facts(path)
    tags: dict[int, Any] = {}
    values: dict[str, Any] = {}

    if is_raw(path):
        try:
            with path.open("rb") as handle:
                tags = _read_tiff_tags(handle.read(_HEADER_BYTES))
        except OSError as error:
            logger.info("%s could not be read for its tags: %s", path.name, error)
        values, preview = _raw_metadata(path, tags)
        # The file's own tags are the authority; the preview only fills gaps.
        tags = {**preview, **tags}
    else:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                tags = dict(exif)
                tags.update(dict(exif.get_ifd(_TAG_EXIF_IFD) or {}))
                width, height = image.size
                # The size is reported the way the photo is shown, which is
                # upright: the decoder turns it by its EXIF orientation
                # (PLAN.md 6 step 1) and an export is written that way round
                # too, so a stored portrait must not be reported as landscape.
                if tags.get(_TAG_ORIENTATION) in _TURNED_ORIENTATIONS:
                    width, height = height, width
                values["width"], values["height"] = width, height

        except (OSError, ValueError, UnidentifiedImageError) as error:
            logger.info("%s could not be read for its tags: %s", path.name, error)

    flash = tags.get(_TAG_FLASH)
    metering = tags.get(_TAG_METERING_MODE)
    white_balance = tags.get(_TAG_WHITE_BALANCE)

    return PhotoMetadata(
        path=path,
        file_size=size,
        file_format=extension,
        width=values.get("width"),
        height=values.get("height"),
        camera=_camera_name(tags.get(_TAG_MAKE), tags.get(_TAG_MODEL)),
        lens=values.get("lens") or _as_text(tags.get(_TAG_LENS_MODEL)),
        aperture=values.get("aperture") or _as_float(tags.get(_TAG_F_NUMBER)),
        shutter_seconds=values.get("shutter_seconds")
        or _as_float(tags.get(_TAG_EXPOSURE_TIME)),
        iso=values.get("iso") or _as_int(tags.get(_TAG_ISO)),
        focal_length_mm=values.get("focal_length_mm") or _as_float(tags.get(_TAG_FOCAL_LENGTH)),
        exposure_compensation=_as_float(tags.get(_TAG_EXPOSURE_BIAS)),
        white_balance=_WHITE_BALANCE_MODES.get(white_balance)
        if isinstance(white_balance, int)
        else None,
        metering=_METERING_MODES.get(metering) if isinstance(metering, int) else None,
        flash_fired=bool(flash & 1) if isinstance(flash, int) else None,
        captured_at=values.get("captured_at") or _parse_date(tags.get(_TAG_DATE_TAKEN)),
    )


def _as_text(value: Any) -> str | None:
    """Return an EXIF text value without its padding, or ``None`` if empty.

    EXIF text is a C string, and some cameras fill the rest of a fixed-size
    field with NUL characters: some JPEGs pad the lens name to 63
    characters that way. Everything from the first NUL on is not text.
    """
    if not isinstance(value, str):
        return None
    return value.split("\x00", 1)[0].strip() or None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
