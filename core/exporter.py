"""Writing the finished photo to a file (PLAN.md section 2.6).

The export and the preview develop the photo with exactly the same code; all
that differs is that the export hands :mod:`core.pipeline` the photo at its
full size and asks for dither (PLAN.md section 6). What is left for this
module is the file: the format, the bit depth, the colour profile, the
exposure details and a name that never overwrites anything.

Three formats, all of them in standard sRGB with the profile embedded
(X-04). The profile is produced by Pillow at runtime rather than shipped
(PLAN.md 13.2). Pillow writes the JPEG and the 8 bit PNG, tifffile the TIFF,
and a 16 bit PNG is encoded by OpenCV and then given its iCCP and eXIf
chunks here, which is what PLAN.md section 6 step 12 asks for and needs no
further package.

Neither EXIF nor a profile can be added by tifffile, so a TIFF is finished
by appending a fresh main directory and an EXIF directory to the end of the
file and pointing the header at them. Writing those directories is also
what gives the JPEG and the PNG their EXIF, so there is one implementation.

The photo the user shot is only ever read. An untouched JPEG asked for at
its own size is copied rather than re-encoded (X-08), and even then the copy
is the only file touched.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import piexif
import tifffile
from numpy.typing import NDArray
from PIL import Image

from core import color_management as cm
from core import decoder, face_detector, metadata, pipeline, watermark
from core import geometry as geometry_module
from core.decoder import DecodedPhoto, DecodeError
from core.edit_state import EditState
from core.faces import Face
from core.filters import Look
from core.geometry import Geometry
from core.metadata import PhotoMetadata
from core.watermark import WatermarkSpec

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# What can be asked for (PLAN.md 2.6 X-02, X-03)
# --------------------------------------------------------------------------

FORMAT_JPEG: Final[str] = "jpg"
FORMAT_PNG: Final[str] = "png"
FORMAT_TIFF: Final[str] = "tiff"
FORMATS: Final[tuple[str, ...]] = (FORMAT_JPEG, FORMAT_PNG, FORMAT_TIFF)

EXTENSIONS: Final[dict[str, str]] = {
    FORMAT_JPEG: ".jpg",
    FORMAT_PNG: ".png",
    FORMAT_TIFF: ".tif",
}

QUALITY_MIN: Final[int] = 1
QUALITY_MAX: Final[int] = 100
QUALITY_DEFAULT: Final[int] = 90

BIT_DEPTHS: Final[tuple[int, int]] = (pipeline.EIGHT_BIT, pipeline.SIXTEEN_BIT)

#: Smallest and largest long edge a resized export may ask for.
LONG_EDGE_MIN: Final[int] = 64
LONG_EDGE_MAX: Final[int] = 20000

#: What is added to the original name (X-05).
DEFAULT_SUFFIX: Final[str] = "_edit"

#: How many names are tried before giving up (X-05: _2, _3 and so on).
_UNIQUE_LIMIT: Final[int] = 1000

#: Above this quality the colour channels are written at full resolution.
#: Below it the usual 4:2:0 subsampling saves a good deal of space for a
#: loss that only shows on hard colour edges, which is where photo editors
#: commonly put the same switch.
CHROMA_FULL_ABOVE_QUALITY: Final[int] = 90

#: PNG compression effort, out of nine. Six is Pillow's own default and the
#: point where more effort stops paying for itself on a photograph.
PNG_COMPRESSION: Final[int] = 6

#: The second byte of each 16 bit sample is near enough random, so it hardly
#: compresses at all: a 16 bit PNG comes to what the same photo costs at
#: 8 bit plus about this many bytes for every pixel. Measured over four of
#: the sample photos, where it predicted the written file to within 2 per
#: cent; only the estimate in the dialog uses it.
PNG_16_BIT_EXTRA_BYTES: Final[float] = 3.4


#: Roughly how much of a TIFF is not pixels, for that same estimate.
TIFF_OVERHEAD_BYTES: Final[int] = 4096

# How far along the export is after each stage, for the progress shown on
# the toolbar button (X-07). The stages are the ones that can be reported
# without reaching into the pipeline: decoding the full size photo,
# developing it, and writing the file.
_PROGRESS_START: Final[float] = 0.05
_PROGRESS_DECODED: Final[float] = 0.45
_PROGRESS_DEVELOPED: Final[float] = 0.85
_PROGRESS_DONE: Final[float] = 1.0


class ExportError(Exception):
    """A photo could not be written, with a reason meant for the log."""


@dataclass(frozen=True)
class ExportOptions:
    """Everything the export dialog decides (PLAN.md 3.11)."""

    format: str = FORMAT_JPEG
    #: JPEG quality, 1 to 100 (X-02).
    quality: int = QUALITY_DEFAULT
    #: Bit depth of a PNG or a TIFF; a JPEG is always 8 bit.
    bits: int = pipeline.EIGHT_BIT
    #: The long edge to scale to, or ``None`` for the photo's own size (X-03).
    long_edge: int | None = None
    #: Keep the details of the shot: time, camera, lens, aperture, shutter
    #: and ISO (X-06).
    keep_metadata: bool = True
    #: Leave the location out, which is the default, for the user's privacy.
    remove_gps: bool = True
    #: The watermark laid over the photo (X-10, PLAN.md 2.9), or ``None``.
    watermark: WatermarkSpec | None = None

    @property
    def extension(self) -> str:
        """Return the file extension this format is written with."""
        return EXTENSIONS[self.format]

    @property
    def depth(self) -> int:
        """Return the bit depth actually used; a JPEG has only the one."""
        return pipeline.EIGHT_BIT if self.format == FORMAT_JPEG else self.bits

    def sane(self) -> "ExportOptions":
        """Return these options with every value inside its range."""
        long_edge = self.long_edge
        if long_edge is not None:
            long_edge = int(min(LONG_EDGE_MAX, max(LONG_EDGE_MIN, long_edge)))
        # A watermark with nothing to draw is no watermark: it must not stop
        # an untouched JPEG from being copied (X-08).
        mark = self.watermark.sane() if self.watermark is not None else None
        return replace(
            self,
            format=self.format if self.format in FORMATS else FORMAT_JPEG,
            quality=int(min(QUALITY_MAX, max(QUALITY_MIN, self.quality))),
            bits=self.bits if self.bits in BIT_DEPTHS else pipeline.EIGHT_BIT,
            long_edge=long_edge,
            watermark=None if mark is None or mark.is_empty else mark,
        )


@dataclass(frozen=True)
class ExportResult:
    """Where the photo was written, and whether it was copied as it stood."""

    path: Path
    #: True when the original file was copied rather than developed again
    #: (X-08), which also means the dialog's quality was not used.
    copied: bool


# --------------------------------------------------------------------------
# Naming (X-05)
# --------------------------------------------------------------------------


def default_name(
    source: Path, suffix: str = DEFAULT_SUFFIX, format: str = FORMAT_JPEG
) -> str:
    """Return the name an export of *source* is offered under (X-05)."""
    extension = EXTENSIONS.get(format, EXTENSIONS[FORMAT_JPEG])
    return f"{source.stem}{suffix}{extension}"


def unique_path(target: Path) -> Path:
    """Return *target*, or the first free ``_2``, ``_3`` ... beside it.

    Nothing is ever overwritten (X-05).
    """
    if not target.exists():
        return target
    for number in range(2, _UNIQUE_LIMIT):
        candidate = target.with_name(f"{target.stem}_{number}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise ExportError(f"no free name beside {target}")


# --------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------


def can_copy_original(
    source: Path,
    state: EditState | None,
    geometry: Geometry | None,
    options: ExportOptions,
    faces: tuple[Face, ...] | None = None,
) -> bool:
    """Return whether the original file can simply be copied (X-08).

    A JPEG nobody has adjusted, asked for as a JPEG at its own size with no
    crop and no rotation, is already the file the user wants: re-encoding it
    would only cost quality (PLAN.md section 6, "JPG passthrough"). Unlike
    the preview, which has to convert anything that is not sRGB before it
    can be shown, this holds whatever colour space the file is in, because a
    copy keeps the file's own profile with it (X-04).
    """
    if options.format != FORMAT_JPEG or options.long_edge is not None:
        return False
    if options.watermark is not None:
        # The watermark changes the pixels, so the file has to be written
        # again (W-12).
        return False
    if source.suffix.lower() not in decoder.JPEG_EXTENSIONS:
        return False
    if geometry is not None and (not geometry.is_default or geometry.spots):
        return False
    if state is None:
        return True
    if faces is not None and not faces:
        # A photo known to hold no faces is untouched by the face sliders.
        state = state.without_faces()
    return state.is_untouched(EditState.for_photo(False))


def export(
    source: Path,
    target: Path,
    state: EditState | None = None,
    geometry: Geometry | None = None,
    options: ExportOptions | None = None,
    progress: Callable[[float], None] | None = None,
    look: Look | None = None,
    faces: tuple[Face, ...] | None = None,
) -> ExportResult:
    """Write the photo at *source* to *target* and return where it landed.

    *target* is the name asked for; if something is there already the file
    is written beside it under a free name (X-05). The original is only read.

    *look* is the film filter *state* names, looked up by the caller the way
    the preview looks it up, so that the exported file is what was on screen.
    *faces* are the faces the preview found, for the same reason; without
    them they are looked for again (PLAN.md 6 step 2.6).
    """
    options = (options or ExportOptions()).sane()
    report = progress if progress is not None else _ignore
    if can_copy_original(source, state, geometry, options, faces):
        # Nothing to develop, so there is no reason to decode it either.
        return export_decoded(None, source, target, state, geometry, options, progress)
    report(_PROGRESS_START)
    try:
        photo = decoder.decode(source)
    except (DecodeError, OSError, ValueError, MemoryError) as error:
        raise ExportError(f"{source.name} could not be exported: {error}") from error
    return export_decoded(photo, source, target, state, geometry, options, progress, look, faces)


def export_decoded(
    photo: DecodedPhoto | None,
    source: Path,
    target: Path,
    state: EditState | None = None,
    geometry: Geometry | None = None,
    options: ExportOptions | None = None,
    progress: Callable[[float], None] | None = None,
    look: Look | None = None,
    faces: tuple[Face, ...] | None = None,
) -> ExportResult:
    """Write *photo*, already decoded from *source*, to *target*.

    The same as :func:`export` for a caller that had to decode the photo
    anyway, such as an export of several photos, which needs the decoded
    photo to read a parameter file (X-09). *photo* may be ``None`` only when
    the original is copied as it is (X-08).
    """
    options = (options or ExportOptions()).sane()
    report = progress if progress is not None else _ignore
    report(_PROGRESS_START)
    try:
        # The folder the user chose normally exists; making it is for the one
        # that has been renamed or unplugged since it was chosen.
        target.parent.mkdir(parents=True, exist_ok=True)
        path = unique_path(target.with_suffix(options.extension))
        if can_copy_original(source, state, geometry, options, faces):
            _copy_original(source, path, options)
            report(_PROGRESS_DONE)
            return ExportResult(path=path, copied=True)
        if photo is None:
            raise ValueError("a photo that is developed must be decoded first")

        photo = face_detector.attach(photo, state, geometry, faces)
        report(_PROGRESS_DECODED)
        image = _develop(photo, state, geometry, options, look)
        report(_PROGRESS_DEVELOPED)
        _write(image, source, path, options)
    except (DecodeError, OSError, ValueError, MemoryError) as error:
        raise ExportError(f"{source.name} could not be exported: {error}") from error
    report(_PROGRESS_DONE)
    return ExportResult(path=path, copied=False)


def export_pixels(
    image: NDArray[Any], target: Path, options: ExportOptions
) -> ExportResult:
    """Write pixels that are already finished, such as a collage (PLAN.md 2.8).

    *image* is sRGB at the depth *options* asks for. Nothing in it is a
    single shot, so no details of one are written; the profile is, as for
    every other export. Nothing is overwritten (X-05).
    """
    options = options.sane()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        path = unique_path(target.with_suffix(options.extension))
        _write_pixels(_watermarked(image, options), path, options, None)
    except (OSError, ValueError, MemoryError) as error:
        raise ExportError(f"{target.name} could not be written: {error}") from error
    return ExportResult(path=path, copied=False)


def _ignore(_fraction: float) -> None:
    """Take a progress report nobody is listening for."""


def _develop(
    photo: DecodedPhoto,
    state: EditState | None,
    geometry: Geometry | None,
    options: ExportOptions,
    look: Look | None = None,
) -> NDArray[Any]:
    """Return the finished pixels: the preview's pipeline at full size.

    Scaling to a smaller long edge happens after the photo is developed
    rather than before: the local adjustment of PLAN.md section 6 step 7 has
    a radius measured against the long edge, so developing a shrunken photo
    would not look like what the preview showed.
    """
    depth = options.depth
    image = pipeline.render(
        photo,
        state,
        geometry,
        dither=depth == pipeline.EIGHT_BIT,
        bits=depth,
        look=look,
    )
    return _watermarked(_scaled(image, options.long_edge), options)


def _watermarked(image: NDArray[Any], options: ExportOptions) -> NDArray[Any]:
    """Return *image* with the watermark *options* ask for (PLAN.md 2.9).

    Laid over the finished pixels at the size they are written, so it is the
    same share of the photo whatever size is asked for; only the pixels under
    it are touched.
    """
    return watermark.apply(
        image, options.watermark, dither=options.depth == pipeline.EIGHT_BIT
    )


def _scaled(image: NDArray[Any], long_edge: int | None) -> NDArray[Any]:
    """Return *image* with its long edge at *long_edge*, never enlarged."""
    height, width = image.shape[:2]
    largest = max(width, height)
    if long_edge is None or long_edge >= largest:
        return image
    scale = long_edge / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Copying an untouched JPEG (X-08)
# --------------------------------------------------------------------------


def _copy_original(source: Path, target: Path, options: ExportOptions) -> None:
    """Copy *source* to *target*, changing only the EXIF that must change.

    The pixels and the compression are untouched, which is the whole point:
    the copy is the original file. Its details are rewritten only when
    something actually has to go, and then with piexif, which replaces the
    EXIF segment and leaves the compressed image alone. When there is
    nothing to remove the copy is identical to the original, byte for byte.
    """
    shutil.copyfile(source, target)
    if options.keep_metadata and not options.remove_gps:
        return
    try:
        if not options.keep_metadata:
            piexif.remove(str(target))
            return
        exif = piexif.load(str(target))
        if not exif.get("GPS"):
            return
        exif["GPS"] = {}
        piexif.insert(piexif.dump(exif), str(target))
    except (OSError, ValueError, struct.error, piexif.InvalidImageDataError):
        logger.info("%s: the copied file's EXIF could not be edited", target.name)


# --------------------------------------------------------------------------
# Writing each format
# --------------------------------------------------------------------------


def _write(
    image: NDArray[Any], source: Path, target: Path, options: ExportOptions
) -> None:
    """Write *image* to *target*, through a part file so nothing is half written."""
    info = metadata.read(source) if options.keep_metadata else None
    _write_pixels(image, target, options, info)


def _write_pixels(
    image: NDArray[Any], target: Path, options: ExportOptions, info: PhotoMetadata | None
) -> None:
    """Write *image* with *info*'s details, through a part file."""
    size = (int(image.shape[1]), int(image.shape[0]))
    exif = exif_block(info, size)
    profile = cm.srgb_profile_bytes()

    part = target.with_name(target.name + ".part")
    try:
        if options.format == FORMAT_JPEG:
            _write_jpeg(image, part, options, profile, exif)
        elif options.format == FORMAT_PNG:
            if options.depth == pipeline.SIXTEEN_BIT:
                _write_png16(image, part, profile, exif)
            else:
                _write_png8(image, part, profile, exif)
        else:
            _write_tiff(image, part, profile, info, size)
        os.replace(part, target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _write_jpeg(
    image: NDArray[np.uint8],
    target: Path | BytesIO,
    options: ExportOptions,
    profile: bytes,
    exif: bytes | None,
) -> None:
    """Write an 8 bit JPEG at the quality asked for (X-02)."""
    Image.fromarray(image).save(
        target,
        "JPEG",
        quality=options.quality,
        subsampling=0 if options.quality >= CHROMA_FULL_ABOVE_QUALITY else 2,
        icc_profile=profile,
        **({"exif": _EXIF_MARKER + exif} if exif else {}),
    )


def _write_png8(
    image: NDArray[np.uint8],
    target: Path | BytesIO,
    profile: bytes,
    exif: bytes | None,
) -> None:
    """Write an 8 bit PNG; Pillow writes both the iCCP and the eXIf chunk."""
    Image.fromarray(image).save(
        target,
        "PNG",
        compress_level=PNG_COMPRESSION,
        icc_profile=profile,
        **({"exif": _EXIF_MARKER + exif} if exif else {}),
    )



def _write_png16(
    image: NDArray[np.uint16], target: Path, profile: bytes, exif: bytes | None
) -> None:
    """Write a 16 bit PNG (PLAN.md 6 step 12).

    Pillow cannot hold three 16 bit channels, so OpenCV encodes the pixels
    and the colour profile and the details of the shot are inserted here as
    iCCP and eXIf chunks, with the standard library.
    """
    written, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    )
    if not written:
        raise ExportError("the 16 bit PNG could not be encoded")
    target.write_bytes(with_png_chunks(bytes(buffer), profile, exif))


def _write_tiff(
    image: NDArray[Any],
    target: Path,
    profile: bytes,
    info: PhotoMetadata | None,
    size: tuple[int, int],
) -> None:
    """Write an uncompressed 8 or 16 bit TIFF with its profile and its EXIF.

    Left uncompressed so that any program can open it: PLAN.md asks nothing
    about the size of a TIFF, and compression is the one thing older readers
    still disagree about.
    """
    tifffile.imwrite(
        target,
        image,
        photometric="rgb",
        metadata=None,
        byteorder=_TIFF_ORDER,
        extratags=[(_TAG_ICC_PROFILE, _TYPE_BYTE, len(profile), profile, True)],
    )
    add_tiff_metadata(target, info, size)


# --------------------------------------------------------------------------
# The estimate shown in the dialog (PLAN.md 3.11)
# --------------------------------------------------------------------------


def estimate_size(
    preview: NDArray[np.uint8], output_pixels: int, options: ExportOptions
) -> int:
    """Return roughly how large the exported file will be, in bytes.

    PLAN.md 3.11: the preview proxy is compressed with the settings asked
    for and the result scaled by the number of pixels, which is why the
    dialog says "about". A TIFF needs no trial at all, since uncompressed
    pixels can be counted exactly.
    """
    options = options.sane()
    if options.format == FORMAT_TIFF:
        return TIFF_OVERHEAD_BYTES + output_pixels * 3 * (options.depth // 8)

    pixels = int(preview.shape[0]) * int(preview.shape[1])
    if pixels == 0 or output_pixels == 0:
        return 0
    buffer = BytesIO()
    profile = cm.srgb_profile_bytes()
    if options.format == FORMAT_JPEG:
        _write_jpeg(preview, buffer, options, profile, None)
    else:
        _write_png8(preview, buffer, profile, None)
    size = buffer.tell() * output_pixels / pixels
    if options.format == FORMAT_PNG and options.depth == pipeline.SIXTEEN_BIT:
        size += output_pixels * PNG_16_BIT_EXTRA_BYTES
    return int(size)



def output_size(
    photo_size: tuple[int, int], geometry: Geometry | None, options: ExportOptions
) -> tuple[int, int]:
    """Return the size the exported file will have, in pixels (PLAN.md 3.11)."""
    width, height = geometry_module.output_size(photo_size, geometry)
    long_edge = options.sane().long_edge
    largest = max(width, height)
    if long_edge is None or largest == 0 or long_edge >= largest:
        return width, height
    scale = long_edge / largest
    return max(1, round(width * scale)), max(1, round(height * scale))


# --------------------------------------------------------------------------
# EXIF, written as a TIFF directory (X-06)
# --------------------------------------------------------------------------

#: Little endian, as every camera this program is meant for writes.
_TIFF_ORDER: Final[str] = "<"

_TYPE_BYTE: Final[int] = 1
_TYPE_ASCII: Final[int] = 2
_TYPE_SHORT: Final[int] = 3
_TYPE_LONG: Final[int] = 4
_TYPE_RATIONAL: Final[int] = 5
_TYPE_UNDEFINED: Final[int] = 7
#: Not a type: an entry copied from a directory that is already written, and
#: so already twelve bytes long.
_TYPE_COPIED: Final[int] = 0

_TYPE_SIZES: Final[dict[int, int]] = {
    _TYPE_BYTE: 1,
    _TYPE_ASCII: 1,
    _TYPE_SHORT: 2,
    _TYPE_LONG: 4,
    _TYPE_RATIONAL: 8,
    _TYPE_UNDEFINED: 1,
}

_TAG_MAKE: Final[int] = 271
_TAG_MODEL: Final[int] = 272
_TAG_ORIENTATION: Final[int] = 274
_TAG_SOFTWARE: Final[int] = 305
_TAG_DATE_TIME: Final[int] = 306
_TAG_EXIF_IFD: Final[int] = 34665
_TAG_ICC_PROFILE: Final[int] = 34675

_TAG_EXPOSURE_TIME: Final[int] = 33434
_TAG_F_NUMBER: Final[int] = 33437
_TAG_ISO: Final[int] = 34855
_TAG_EXIF_VERSION: Final[int] = 36864
_TAG_DATE_TAKEN: Final[int] = 36867
_TAG_FOCAL_LENGTH: Final[int] = 37386
_TAG_COLOR_SPACE: Final[int] = 40961
_TAG_PIXEL_WIDTH: Final[int] = 40962
_TAG_PIXEL_HEIGHT: Final[int] = 40963
_TAG_LENS_MODEL: Final[int] = 42036

#: The photo is written upright, so whatever the camera recorded, the
#: exported file says it needs no turning.
_ORIENTATION_UPRIGHT: Final[int] = 1
#: sRGB, which is what every exported file is in (X-04).
_COLOR_SPACE_SRGB: Final[int] = 1
_EXIF_VERSION: Final[bytes] = b"0231"

_DATE_FORMAT: Final[str] = "%Y:%m:%d %H:%M:%S"

#: What a JPEG's APP1 segment begins with. Pillow writes the bytes it is
#: given as they stand, so the marker is added here; for a PNG it strips the
#: marker off again and writes the rest as the eXIf chunk.
_EXIF_MARKER: Final[bytes] = b"Exif\x00\x00"

#: Denominator for the values a camera writes in tenths.
_TENTHS: Final[int] = 10

#: The largest ISO an EXIF short can hold.
_MAX_SHORT: Final[int] = 65535

_Entry = tuple[int, int, bytes]


def exif_block(info: PhotoMetadata | None, size: tuple[int, int]) -> bytes:
    """Return the details of the shot as a TIFF block (X-06).

    The block is what a JPEG carries in its APP1 segment after the ``Exif``
    marker and what a PNG carries in its eXIf chunk, so both formats get
    their details from the same code.
    """
    main, sub = _exif_entries(info, size)
    header = b"II*\x00" if _TIFF_ORDER == "<" else b"MM\x00*"
    base = len(header) + 4
    placeholder = _ifd([*main, (_TAG_EXIF_IFD, _TYPE_LONG, _long(0))], base)
    sub_offset = base + len(placeholder) + len(placeholder) % 2
    directory = _ifd([*main, (_TAG_EXIF_IFD, _TYPE_LONG, _long(sub_offset))], base)
    return (
        header
        + struct.pack(_TIFF_ORDER + "I", base)
        + directory
        + b"\x00" * (len(directory) % 2)
        + _ifd(sub, sub_offset)
    )


def _exif_entries(
    info: PhotoMetadata | None, size: tuple[int, int]
) -> tuple[list[_Entry], list[_Entry]]:
    """Return the entries of the main and of the EXIF directory (X-06)."""
    width, height = size
    main: list[_Entry] = [
        (_TAG_ORIENTATION, _TYPE_SHORT, _short(_ORIENTATION_UPRIGHT)),
    ]
    sub: list[_Entry] = [
        (_TAG_EXIF_VERSION, _TYPE_UNDEFINED, _EXIF_VERSION),
        (_TAG_COLOR_SPACE, _TYPE_SHORT, _short(_COLOR_SPACE_SRGB)),
        (_TAG_PIXEL_WIDTH, _TYPE_LONG, _long(width)),
        (_TAG_PIXEL_HEIGHT, _TYPE_LONG, _long(height)),
    ]
    if info is None:
        return main, sub

    if info.camera:
        maker, _, rest = info.camera.partition(" ")
        main.append((_TAG_MAKE, _TYPE_ASCII, _ascii(maker)))
        main.append((_TAG_MODEL, _TYPE_ASCII, _ascii(rest or info.camera)))
    taken = _as_date_text(info)
    if taken is not None:
        main.append((_TAG_DATE_TIME, _TYPE_ASCII, _ascii(taken)))
        sub.append((_TAG_DATE_TAKEN, _TYPE_ASCII, _ascii(taken)))
    if info.shutter_seconds:
        sub.append((_TAG_EXPOSURE_TIME, _TYPE_RATIONAL, _rational(info.shutter_seconds)))
    if info.aperture:
        sub.append((_TAG_F_NUMBER, _TYPE_RATIONAL, _rational(info.aperture)))
    if info.iso:
        sub.append((_TAG_ISO, _TYPE_SHORT, _short(min(info.iso, _MAX_SHORT))))
    if info.focal_length_mm:
        sub.append((_TAG_FOCAL_LENGTH, _TYPE_RATIONAL, _rational(info.focal_length_mm)))
    if info.lens:
        sub.append((_TAG_LENS_MODEL, _TYPE_ASCII, _ascii(info.lens)))
    return main, sub


def _as_date_text(info: PhotoMetadata) -> str | None:
    """Return when the photo was taken, in the form EXIF writes it."""
    taken = info.captured_at
    if taken is None:
        return None
    try:
        return taken.strftime(_DATE_FORMAT)
    except (AttributeError, ValueError):
        return None


def _ascii(text: str) -> bytes:
    """Return *text* as an EXIF string, which ends in a zero byte."""
    return text.encode("ascii", "replace") + b"\x00"


def _short(value: int) -> bytes:
    """Return *value* as a 16 bit field."""
    return struct.pack(_TIFF_ORDER + "H", int(value))


def _long(value: int) -> bytes:
    """Return *value* as a 32 bit field."""
    return struct.pack(_TIFF_ORDER + "I", int(value))


def _rational(value: float) -> bytes:
    """Return *value* as the fraction a camera would write.

    A shutter faster than a second becomes 1/n, which is how it is read and
    shown; a longer one, and every other measurement, is written in tenths.
    """
    if 0.0 < value < 1.0:
        return struct.pack(_TIFF_ORDER + "II", 1, max(1, round(1.0 / value)))
    return struct.pack(_TIFF_ORDER + "II", max(0, round(value * _TENTHS)), _TENTHS)


def _ifd(entries: Sequence[_Entry], base: int, next_directory: int = 0) -> bytes:
    """Return one image file directory, to be placed at *base* in the file.

    A value of more than four bytes does not fit in its entry and is written
    after the entries, which is why the directory has to be told where it
    will live. Entries come in order of their tag, as the format requires.
    """
    fields = bytearray()
    values = bytearray()
    head = 2 + 12 * len(entries) + 4
    for tag, kind, payload in sorted(entries, key=lambda entry: entry[0]):
        if kind == _TYPE_COPIED:
            fields += payload
            continue
        count = len(payload) // _TYPE_SIZES[kind]
        if len(payload) <= 4:
            field = payload.ljust(4, b"\x00")
        else:
            if len(values) % 2:
                values += b"\x00"
            field = struct.pack(_TIFF_ORDER + "I", base + head + len(values))
            values += payload
        fields += struct.pack(_TIFF_ORDER + "HHI", tag, kind, count) + field
    return (
        struct.pack(_TIFF_ORDER + "H", len(entries))
        + bytes(fields)
        + struct.pack(_TIFF_ORDER + "I", next_directory)
        + bytes(values)
    )


# --------------------------------------------------------------------------
# Finishing a TIFF
# --------------------------------------------------------------------------


def add_tiff_metadata(
    target: Path, info: PhotoMetadata | None, size: tuple[int, int]
) -> None:
    """Give a written TIFF the details of the shot (X-06).

    tifffile writes the pixels and the colour profile but will not write an
    EXIF directory, so a new main directory is appended to the end of the
    file - the entries it already has, copied as they stand, plus the camera,
    the date and a pointer to an EXIF directory - and the header is pointed
    at it. The directory that was there is left behind, which costs a few
    dozen unused bytes and keeps every offset already in the file correct.
    """
    data = bytearray(target.read_bytes())
    order = "<" if data[:2] == b"II" else ">"
    if order != _TIFF_ORDER:
        raise ExportError("the written TIFF is not in the expected byte order")

    first = struct.unpack_from(order + "I", data, 4)[0]
    count = struct.unpack_from(order + "H", data, first)[0]
    existing: list[_Entry] = []
    for index in range(count):
        at = first + 2 + index * 12
        tag = struct.unpack_from(order + "H", data, at)[0]
        # The directory is being rebuilt, so this is the moment to drop the
        # note tifffile leaves about itself, and anything replaced below.
        if tag in (_TAG_SOFTWARE, _TAG_ORIENTATION, _TAG_EXIF_IFD):
            continue
        existing.append((tag, _TYPE_COPIED, bytes(data[at : at + 12])))
    following = struct.unpack_from(order + "I", data, first + 2 + count * 12)[0]

    main, sub = _exif_entries(info, size)
    if len(data) % 2:
        data += b"\x00"
    base = len(data)
    placeholder = [*existing, *main, (_TAG_EXIF_IFD, _TYPE_LONG, _long(0))]
    sub_offset = base + len(_ifd(placeholder, base, following))
    sub_offset += sub_offset % 2
    entries = [*existing, *main, (_TAG_EXIF_IFD, _TYPE_LONG, _long(sub_offset))]

    data += _ifd(entries, base, following)
    if len(data) % 2:
        data += b"\x00"
    data += _ifd(sub, sub_offset)
    struct.pack_into(order + "I", data, 4, base)
    target.write_bytes(bytes(data))


# --------------------------------------------------------------------------
# Finishing a 16 bit PNG
# --------------------------------------------------------------------------

_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_PNG_ICC_NAME: Final[bytes] = b"ICC Profile"
#: The length, the name and the check of a PNG chunk, around its content.
_PNG_CHUNK_FRAME: Final[int] = 12


def with_png_chunks(data: bytes, profile: bytes, exif: bytes | None) -> bytes:
    """Return the encoded PNG *data* with its iCCP and eXIf chunks added.

    Both go directly after the header chunk, which is where a reader looks
    for the colour profile (PLAN.md 6 step 12).
    """
    if not data.startswith(_PNG_SIGNATURE):
        raise ExportError("what was encoded is not a PNG")
    length = struct.unpack_from(">I", data, len(_PNG_SIGNATURE))[0]
    after_header = len(_PNG_SIGNATURE) + _PNG_CHUNK_FRAME + length

    chunks = _png_chunk(
        b"iCCP", _PNG_ICC_NAME + b"\x00\x00" + zlib.compress(profile, PNG_COMPRESSION)
    )
    if exif:
        chunks += _png_chunk(b"eXIf", exif)
    return data[:after_header] + chunks + data[after_header:]


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    """Return one PNG chunk: its length, its name, its content and a check."""
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    )
