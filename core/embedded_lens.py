"""The lens corrections a camera wrote into its own RAW file (PLAN.md 6 step 1.5).

A RAF file carries, for the lens on the camera at the focal length of the
shot, how the lens bent the picture and how much light it lost towards the
corners: the same corrections the camera applies to its own JPEG. They are
measured for that lens on that body, so when a file has them they are used
instead of the Lensfun database, which may only know a similar lens, or know
this one less well.

The format has no public description. What is read here was worked out by
looking at the files and checking the reading against the camera's own JPEGs
(2026-10-31, six RAF+JPG pairs, see PLAN.md 6 step 1.5); no other program's
code was consulted.

* The section of the file that holds the sensor data begins with a TIFF
  header. Its first directory points, with tag ``0xF000``, to a directory of
  the camera's own tags.
* Tag ``0xF00B`` is the distortion and ``0xF010`` the vignetting. Each is a
  list of signed fractions: one value this program does not need, then *n*
  radii, then *n* values. A radius is a fraction of :data:`TABLE_UNIT`; a
  distortion value is how many percent further out than it should the lens
  put that radius of the corrected picture, a vignetting value how many
  percent of the centre's light reached that radius on the sensor.
* Tag ``0xF00F`` holds the chromatic aberration in the same layout, but with
  red and blue it would need two different units to match what the photos
  show, so its meaning is not understood and it is not used.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Final

from core import lens as lens_module
from core.lens import LensProfile

logger = logging.getLogger(__name__)

#: The radius a RAF's tables call 1, as a fraction of half the diagonal of
#: the frame the decoder delivers. Fitted against the camera's JPEGs: with
#: the table applied as it stands, the matched points line up best at 0.980,
#: and 0.003 either side already leaves the corners a pixel further out.
TABLE_UNIT: Final[float] = 0.98

#: Where a RAF's header says the sensor data section starts and how long it
#: is, as two big endian 32 bit numbers.
_SECTION_POINTER: Final[int] = 100

_TAG_CAMERA_DIRECTORY: Final[int] = 0xF000
_TAG_DISTORTION: Final[int] = 0xF00B
_TAG_VIGNETTING: Final[int] = 0xF010

#: The TIFF type of a signed fraction, and its size in bytes.
_SIGNED_RATIONAL: Final[int] = 10
_RATIONAL_BYTES: Final[int] = 8

#: More entries than this is not a directory but something misread.
_MAX_ENTRIES: Final[int] = 512
#: Tables longer than this are not what is expected either.
_MAX_TABLE: Final[int] = 64

#: What a believable table holds. Outside these something has been misread,
#: and correcting with it would do more harm than not correcting.
_MAX_DISTORTION_PERCENT: Final[float] = 30.0
_MIN_LIGHT_PERCENT: Final[float] = 10.0
_MAX_LIGHT_PERCENT: Final[float] = 150.0
_MAX_RADIUS: Final[float] = 2.0


def profile_for(path: Path, lens_name: str) -> LensProfile | None:
    """Return the corrections the camera stored in *path*, or ``None``.

    Only a RAF file stores them in a form this program reads. Never raises:
    a file that cannot be read is simply corrected from the database, or
    not at all.
    """
    if path.suffix.lower() != ".raf":
        return None
    try:
        with path.open("rb") as handle:
            tables = _read_tables(handle)
    except (OSError, struct.error, ValueError) as error:
        logger.info("%s has no lens tables that could be read: %s", path.name, error)
        return None
    if tables is None:
        return None
    distortion, vignetting = tables
    if distortion is None and vignetting is None:
        return None
    return lens_module.camera_profile(lens_name, distortion, vignetting, TABLE_UNIT)


#: A table's radii and the values measured at them.
_Table = tuple[tuple[float, ...], tuple[float, ...]]
#: A directory entry: its type, its count and its four byte value field.
_Entry = tuple[int, int, bytes]
#: Whether a table's values can be believed.
_Check = Callable[[tuple[float, ...]], bool]


def _read_tables(handle: BinaryIO) -> tuple[_Table | None, _Table | None] | None:
    """Return the distortion and vignetting tables of an open RAF file."""
    handle.seek(_SECTION_POINTER)
    start = struct.unpack(">I", _exactly(handle, 4))[0]
    handle.seek(start)
    header = _exactly(handle, 8)
    if header[:4] == b"II*\0":
        order = "<"
    elif header[:4] == b"MM\0*":
        order = ">"
    else:
        return None
    first = struct.unpack(order + "I", header[4:8])[0]
    top = _directory(handle, start, first, order)
    pointer = top.get(_TAG_CAMERA_DIRECTORY)
    if pointer is None:
        return None
    camera = _directory(handle, start, _offset(pointer, order), order)
    return (
        _table(handle, start, camera.get(_TAG_DISTORTION), order, _believable_distortion),
        _table(handle, start, camera.get(_TAG_VIGNETTING), order, _believable_light),
    )



def _directory(handle: BinaryIO, base: int, offset: int, order: str) -> dict[int, _Entry]:
    """Return the entries of the directory at *offset*: type, count, value field."""
    handle.seek(base + offset)
    count = struct.unpack(order + "H", _exactly(handle, 2))[0]
    if count > _MAX_ENTRIES:
        raise ValueError(f"a directory of {count} entries")
    data = _exactly(handle, 12 * count)
    entries: dict[int, _Entry] = {}
    for index in range(count):
        tag, kind, number = struct.unpack(order + "HHI", data[12 * index : 12 * index + 8])
        entries[tag] = (kind, number, data[12 * index + 8 : 12 * index + 12])
    return entries


def _offset(entry: _Entry, order: str) -> int:
    """Return the offset a pointer entry holds."""
    return int(struct.unpack(order + "I", entry[2])[0])


def _table(
    handle: BinaryIO,
    base: int,
    entry: _Entry | None,
    order: str,
    believable: _Check,
) -> _Table | None:
    """Return the radii and values of one table, or ``None`` if it is not usable."""
    if entry is None:
        return None
    kind, number, _field = entry
    size = (number - 1) // 2
    if kind != _SIGNED_RATIONAL or number % 2 == 0 or not 2 <= size <= _MAX_TABLE:
        return None
    handle.seek(base + _offset(entry, order))
    raw = struct.unpack(order + "i" * (2 * number), _exactly(handle, _RATIONAL_BYTES * number))
    values: list[float] = []
    for numerator, denominator in zip(raw[0::2], raw[1::2]):
        if denominator == 0:
            return None
        values.append(numerator / denominator)
    radii = tuple(values[1 : 1 + size])
    measured = tuple(values[1 + size :])
    if not all(math.isfinite(value) for value in (*radii, *measured)):
        return None
    if not all(a < b for a, b in zip((0.0, *radii), (*radii, _MAX_RADIUS))):
        return None
    return (radii, measured) if believable(measured) else None


def _believable_distortion(values: tuple[float, ...]) -> bool:
    """Return whether a distortion table holds a correction worth making."""
    return all(abs(value) <= _MAX_DISTORTION_PERCENT for value in values) and any(values)


def _believable_light(values: tuple[float, ...]) -> bool:
    """Return whether a vignetting table holds a correction worth making."""
    return all(_MIN_LIGHT_PERCENT <= value <= _MAX_LIGHT_PERCENT for value in values) and any(
        value != 100.0 for value in values
    )



def _exactly(handle: BinaryIO, size: int) -> bytes:
    """Return the next *size* bytes, or raise if the file ends first."""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("the file ends early")
    return data
