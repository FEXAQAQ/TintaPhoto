"""Reading RAW, JPEG and PNG files (PLAN.md section 2.1 and section 6 step 1).

A decoded photo is returned as linear light in ``float32``, already turned
upright, together with everything the pipeline needs to know about where the
pixels came from: the colour space or camera matrix, the white balance the
camera used, and which pixels were clipped.

Nothing here writes to the photo file. Opening is read only, always.
"""

from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import rawpy
from numpy.typing import NDArray
from PIL import Image, ImageCms, UnidentifiedImageError

from core import color_management as cm
from core import embedded_lens
from core import lens as lens_module
from core.faces import Face
from core.lens import LensProfile, LensShot
from core.skin_range import SkinRange

logger = logging.getLogger(__name__)

#: Long edge of the preview proxy the interface works on (PLAN.md section 4).
PROXY_LONG_EDGE: Final[int] = 2560

#: A channel at or above this fraction of full scale counts as clipped
#: (PLAN.md section 6 step 1).
CLIPPING_THRESHOLD: Final[float] = 0.98

#: RAW extensions. ARW is required; the rest are supported where LibRaw is.
RAW_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".dng"}
)
JPEG_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg"})
PNG_EXTENSIONS: Final[frozenset[str]] = frozenset({".png"})
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = (
    RAW_EXTENSIONS | JPEG_EXTENSIONS | PNG_EXTENSIONS
)

_SRGB_PROFILE_NAME: Final[str] = "sRGB"
_EXIF_ORIENTATION_TAG: Final[int] = 274
#: ColorSpace lives in the Exif sub-IFD, not in the main one.
_EXIF_IFD_TAG: Final[int] = 34665
_EXIF_COLOR_SPACE_TAG: Final[int] = 40961
_EXIF_INTEROP_TAG: Final[int] = 40965
_EXIF_INTEROP_INDEX_TAG: Final[int] = 1
_COLOR_SPACE_SRGB: Final[int] = 1
_COLOR_SPACE_UNCALIBRATED: Final[int] = 65535
_ADOBE_RGB_INTEROP_INDEX: Final[str] = "R03"

#: Where a PNG keeps its bit depth: one byte of the IHDR chunk.
_PNG_BIT_DEPTH_OFFSET: Final[int] = 24

#: The Pillow modes that carry an alpha channel of their own.
_ALPHA_MODES: Final[frozenset[str]] = frozenset({"RGBA", "LA", "PA", "La"})

#: 8 bit full scale, what a JPEG and most PNGs are encoded at.
_EIGHT_BIT: Final[float] = 255.0

#: 16 bit full scale, the depth LibRaw is asked for and a deep PNG carries.
_SIXTEEN_BIT: Final[float] = 65535.0


class DecodeError(Exception):
    """A photo could not be read. The message is for the log, not the user."""


class ColorSpace(Enum):
    """The colour space a decoded JPEG or PNG was encoded in."""

    SRGB = "sRGB"
    ADOBE_RGB = "Adobe RGB"
    DISPLAY_P3 = "Display P3"


#: Matrix from each encoded colour space to the linear ProPhoto working space.
_ENCODED_TO_PROPHOTO: Final[dict[ColorSpace, NDArray[np.float64]]] = {
    ColorSpace.SRGB: cm.SRGB_TO_PROPHOTO,
    ColorSpace.ADOBE_RGB: cm.ADOBE_RGB_TO_PROPHOTO,
    ColorSpace.DISPLAY_P3: cm.DISPLAY_P3_TO_PROPHOTO,
}


@dataclass(frozen=True)
class DecodedPhoto:
    """A decoded photo and what the pipeline needs to know about it.

    ``image`` is linear light, ``float32``, shape ``(height, width, 3)``, and
    already upright. For a RAW photo it is still in the camera's native colour
    space; for a JPEG or a PNG it has been converted to linear ProPhoto RGB.
    """

    image: NDArray[np.float32]
    is_raw: bool
    #: Pixels where at least one channel was clipped (PLAN.md 6 step 4).
    clipped: NDArray[np.bool_]
    #: Camera RGB to XYZ, for RAW only (PLAN.md 6 step 5).
    camera_to_xyz: NDArray[np.float64] | None = None
    #: The camera's own XYZ to RGB matrix as LibRaw reports it, unnormalised.
    #: White balance needs it to work out what a temperature means for this
    #: sensor, which the normalised matrix above can no longer say.
    rgb_xyz_matrix: NDArray[np.float64] | None = None
    #: The white balance multipliers the camera recorded, for RAW only.
    camera_whitebalance: tuple[float, float, float] | None = None
    #: The colour space the file was encoded in, for a JPEG or a PNG only.
    color_space: ColorSpace | None = None
    #: The embedded profile of a JPEG or a PNG, kept so export can pass it on.
    icc_profile: bytes | None = None
    #: The original encoded pixels of an 8 bit JPEG or PNG, upright but
    #: otherwise untouched, so that an unedited file can be shown without
    #: being reprocessed (PLAN.md section 6, "JPG 原圖直通").
    encoded: NDArray[np.uint8] | None = None
    #: The size this photo was shrunk from, when it is a proxy of a larger
    #: one. Zooming in has to know how much detail the proxy is short of
    #: before it can decide to go back to the original (PLAN.md 3.3).
    shrunk_from: tuple[int, int] | None = None
    #: The detail settings ``image`` has already been through (PLAN.md 6 step
    #: 2.5), or ``None`` for none. The step only depends on those four
    #: sliders, so a photo that has been through it is kept and developed
    #: again while any other slider moves (:func:`core.pipeline.with_detail`).
    detail: tuple[float, float, float, float] | None = None
    #: Where this photo's film grain starts from (PLAN.md 6 step 11.5). Taken
    #: from the file's name, so that the grain is the same every time the
    #: photo is shown or exported, moved folder or not, and differs from one
    #: photo to the next.
    grain_seed: int = 0
    #: The lens's corrections at this photo's focal length and aperture, for
    #: a RAW file whose lens the database knows (PLAN.md 6 step 1.5). Gone
    #: from a photo that has been corrected, so it cannot be corrected twice.
    lens: LensProfile | None = None
    #: The faces in the photo (PLAN.md 6 step 2.6), where they lie once the
    #: lens has been corrected, or ``None`` when nobody has looked yet. Found
    #: once by :mod:`core.face_detector` and handed to every copy of the
    #: photo, so that the preview and the export agree on them; the geometry
    #: step moves them with the photo.
    faces: tuple[Face, ...] | None = None
    #: The skin the portrait sliders work on, as the brush left it (PLAN.md
    #: 2.10), in fractions of this photo's own frame. Set by the geometry
    #: step from the photo's geometry and moved with the frame like the
    #: faces; ``None`` is the skin found by itself, with no strokes.
    skin: SkinRange | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Return the width and height in pixels."""
        return self.image.shape[1], self.image.shape[0]

    @property
    def full_size(self) -> tuple[int, int]:
        """Return the size this photo has at full resolution."""
        return self.shrunk_from if self.shrunk_from is not None else self.size


def is_supported(path: Path) -> bool:
    """Return whether *path* has an extension the application can open."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_raw(path: Path) -> bool:
    """Return whether *path* is a RAW file by extension."""
    return path.suffix.lower() in RAW_EXTENSIONS


def is_png(path: Path) -> bool:
    """Return whether *path* is a PNG file by extension."""
    return path.suffix.lower() in PNG_EXTENSIONS


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------

#: LibRaw reports the camera's rotation as 0, 3, 5 or 6; these are the
#: matching EXIF orientation values.
_LIBRAW_FLIP_TO_ORIENTATION: Final[dict[int, int]] = {0: 1, 3: 3, 5: 8, 6: 6}


def libraw_flip_to_orientation(flip: int) -> int:
    """Return the EXIF orientation matching LibRaw's *flip* value."""
    return _LIBRAW_FLIP_TO_ORIENTATION.get(flip, 1)


def apply_orientation(image: NDArray[Any], orientation: int) -> NDArray[Any]:
    """Return *image* turned upright according to an EXIF orientation value.

    The same routine is used for RAW and JPEG so that there is one place where
    a photo is turned upright, and it can be tested without a camera file
    (PLAN.md F-04).
    """
    if orientation == 2:
        return np.ascontiguousarray(image[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(image[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(image[::-1])
    if orientation == 5:
        return np.ascontiguousarray(np.swapaxes(image, 0, 1))
    if orientation == 6:
        return np.ascontiguousarray(np.swapaxes(image, 0, 1)[:, ::-1])
    if orientation == 7:
        return np.ascontiguousarray(np.swapaxes(image, 0, 1)[::-1, ::-1])
    if orientation == 8:
        return np.ascontiguousarray(np.swapaxes(image, 0, 1)[::-1])
    return image


# --------------------------------------------------------------------------
# Proxy
# --------------------------------------------------------------------------


def make_proxy(image: NDArray[np.float32], long_edge: int = PROXY_LONG_EDGE) -> NDArray[np.float32]:
    """Return *image* scaled so that its long edge is at most *long_edge*.

    Area averaging is used, which is the right filter for shrinking and keeps
    noise from turning into aliasing (PLAN.md section 6 step 1).
    """
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= long_edge:
        return image
    scale = long_edge / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)


# --------------------------------------------------------------------------
# RAW
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawData:
    """What one pass of LibRaw gives us, before any conversion to float."""

    pixels: NDArray[np.uint16]
    clipped: NDArray[np.bool_]
    camera_to_xyz: NDArray[np.float64]
    rgb_xyz_matrix: NDArray[np.float64]
    whitebalance: tuple[float, float, float]
    #: The upright size of the photo at full resolution, as (width, height),
    #: which *pixels* is smaller than when it was read at half size.
    full_size: tuple[int, int]
    #: The lens's corrections: the camera's own when the file holds them,
    #: otherwise the database's when it knows the lens.
    lens: LensProfile | None = None


def _read_raw(path: Path, proxy_long_edge: int | None = None) -> _RawData:
    """Run LibRaw and return the upright 16 bit result and its metadata.

    With *proxy_long_edge*, a photo that is still at least that long at half
    size is read at half size: LibRaw then merges each two by two block of
    sensor sites into one pixel instead of demosaicing, which is several
    times faster (a RAF with a 6x6 pattern goes from about 3 seconds to 0.2) and,
    shrunk to the proxy afterwards, gives the same picture. Only the preview
    proxy is read this way; zooming in and exporting read the full sensor.
    """
    with rawpy.imread(str(path)) as raw:
        # The rotation has to be read before processing: asking for user_flip=0
        # makes LibRaw write that choice back into sizes.flip, so afterwards
        # every photo claims to be upright and portrait ones stay on their side.
        orientation = libraw_flip_to_orientation(int(raw.sizes.flip))
        clipped = _clipped_in_sensor_data(raw)
        rgb_xyz_matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64).copy()
        height, width = int(raw.sizes.height), int(raw.sizes.width)
        half_size = (
            proxy_long_edge is not None
            and float(raw.sizes.pixel_aspect) == 1.0
            and max(height, width) // 2 >= proxy_long_edge
        )
        # Read before processing too: at half size LibRaw rewrites the sizes.
        crop = _default_crop(raw.sizes, 2 if half_size else 1)
        # use_camera_wb demosaics with the white balance the camera recorded,
        # which keeps coloured speckle out of the detail; the pipeline then
        # corrects from there. output_color=raw keeps the camera's native
        # primaries, gamma=(1, 1) and no_auto_bright keep the data linear and
        # unscaled, and user_flip=0 leaves rotating to apply_orientation.
        rgb = raw.postprocess(
            half_size=half_size,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            highlight_mode=rawpy.HighlightMode.Blend,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            user_flip=0,
        )
        whitebalance = tuple(float(value) for value in raw.camera_whitebalance[:3])
        shot = _lens_shot(path, raw)
        try:
            camera_to_xyz = cm.camera_to_xyz_matrix(rgb_xyz_matrix)
        except ValueError:
            logger.warning("%s has no usable camera matrix, treating it as sRGB", path.name)
            camera_to_xyz = cm.SRGB_TO_XYZ

    if half_size and clipped is not None:
        clipped = _halve_mask(clipped)
    # The sensor mask is compared against the processed image before either is
    # turned upright, because the rotation changes both shapes alike.
    if clipped is None or clipped.shape != rgb.shape[:2]:
        clipped = _clipped_in_output(rgb)
    if crop is not None:
        rgb = rgb[crop.rows, crop.columns]
        clipped = clipped[crop.rows, crop.columns]
        height, width = crop.full_height, crop.full_width
    pixels = apply_orientation(rgb, orientation)
    clipped = apply_orientation(clipped, orientation)

    if not half_size:
        height, width = rgb.shape[:2]
    full_size = (height, width) if orientation in _SIDEWAYS else (width, height)
    green = whitebalance[1] or 1.0
    return _RawData(
        pixels=pixels,
        clipped=clipped,
        camera_to_xyz=camera_to_xyz,
        rgb_xyz_matrix=rgb_xyz_matrix,
        whitebalance=(whitebalance[0] / green, 1.0, whitebalance[2] / green),
        full_size=full_size,
        lens=_lens_profile(path, shot),
    )


def _lens_profile(path: Path, shot: LensShot | None) -> LensProfile | None:
    """Return the corrections for *path*: the camera's own, or the database's."""
    own = embedded_lens.profile_for(path, shot.lens_model if shot is not None else "")
    return own if own is not None else lens_module.profile_for(shot)


def _c_text(value: object) -> str:
    """Return a LibRaw text field without the NUL padding some cameras leave."""
    return str(value or "").partition(chr(0))[0].strip()


def _lens_shot(path: Path, raw: "rawpy.RawPy") -> LensShot | None:
    """Return what the lens lookup needs to know about how *raw* was taken.

    The lens, focal length and aperture come from LibRaw. The camera's name
    and the 35 mm equivalent focal length, which give the crop factor, come
    from the file's own tags; :mod:`core.metadata` reads those, and is
    imported here rather than at the top because it imports this module.
    """
    from core import metadata

    try:
        lens_model = _c_text(raw.lens.model)
        lens_maker = _c_text(raw.lens.make)
        focal = float(raw.other.focal_length)
        aperture = float(raw.other.aperture)
    except (AttributeError, ValueError, rawpy.LibRawError):
        return None
    if not lens_model:
        return None
    camera = metadata.camera_facts(path, raw)
    return LensShot(
        lens_model=lens_model,
        lens_maker=lens_maker,
        camera_maker=camera.maker,
        camera_model=camera.model,
        focal_length=focal if math.isfinite(focal) else 0.0,
        aperture=aperture if math.isfinite(aperture) else 0.0,
        focal_length_35mm=camera.focal_length_35mm,
    )


@dataclass(frozen=True)
class _Crop:
    """The part of LibRaw's output the camera itself keeps."""

    rows: slice
    columns: slice
    #: The size of the kept part at full resolution, before any rotation.
    full_height: int
    full_width: int


def _default_crop(sizes: Any, step: int) -> _Crop | None:
    """Return the camera's own crop of the sensor, or ``None`` if it has none.

    LibRaw hands over every pixel it can demosaic, which for many bodies is a
    few more than the camera puts in its JPEG: a 24 megapixel ARW comes out
    at 6024 by 4024 where the camera gives 6000 by 4000.
    The extra rim is the edge of the demosaic and belongs to nobody's photo.
    The file says where the camera's frame lies, relative to the whole sensor
    readout, so the masked margin LibRaw has already dropped is taken off
    first. *step* is 2 when the file was read at half size.
    """
    left = int(getattr(sizes, "crop_left_margin", 0)) - int(sizes.left_margin)
    top = int(getattr(sizes, "crop_top_margin", 0)) - int(sizes.top_margin)
    width = int(getattr(sizes, "crop_width", 0))
    height = int(getattr(sizes, "crop_height", 0))
    visible_width, visible_height = int(sizes.width), int(sizes.height)
    if width <= 0 or height <= 0 or left < 0 or top < 0:
        return None
    if left + width > visible_width or top + height > visible_height:
        return None
    if (width, height) == (visible_width, visible_height):
        return None
    return _Crop(
        rows=slice(top // step, (top + height) // step),
        columns=slice(left // step, (left + width) // step),
        full_height=height,
        full_width=width,
    )


def sensor_frame_size(sizes: Any) -> tuple[int, int]:
    """Return the width and height a RAW file decodes to, before rotation.

    *sizes* is LibRaw's ``raw.sizes``. This is the camera's own frame when
    the file names one, so the size reported for a photo is the size it
    opens and exports at.
    """
    crop = _default_crop(sizes, 1)
    if crop is None:
        return int(sizes.width), int(sizes.height)
    return crop.full_width, crop.full_height


#: The EXIF orientations that turn a photo on its side, swapping its width
#: and height.
_SIDEWAYS: Final[frozenset[int]] = frozenset({5, 6, 7, 8})


def _halve_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return *mask* at half size, a pixel set wherever any of its four sites was.

    At half size each pixel is made from one two by two block of sensor
    sites, one per colour, so a pixel is clipped when any of them is: the
    same rule as for a JPEG, where any clipped channel counts.
    """
    height, width = (mask.shape[0] // 2) * 2, (mask.shape[1] // 2) * 2
    blocks = mask[:height, :width].reshape(height // 2, 2, width // 2, 2)
    return blocks.any(axis=(1, 3))


def _clipped_in_sensor_data(raw: "rawpy.RawPy") -> NDArray[np.bool_] | None:
    """Return which sensor sites were saturated, or ``None`` if unavailable.

    Clipping has to be found here rather than in the processed image. By the
    time LibRaw has white balanced and blended the highlights, a sky that was
    fully saturated comes out well below full scale, so looking for white
    pixels in the result finds nothing at all. In the sensor data saturation
    is plain: the value sits at the white level.

    Each sensor site becomes one output pixel, so the mask lines up with the
    processed image without any resampling.
    """
    try:
        sensor = raw.raw_image_visible
        white_level = float(raw.white_level)
    except (AttributeError, ValueError, rawpy.LibRawError):
        return None
    if white_level <= 0 or sensor.ndim != 2:
        return None
    return sensor >= np.uint16(round(white_level * CLIPPING_THRESHOLD))


def _clipped_in_output(pixels: NDArray[np.uint16]) -> NDArray[np.bool_]:
    """Fall back to looking for near white pixels in the processed image."""
    level = np.uint16(round(CLIPPING_THRESHOLD * _SIXTEEN_BIT))
    brightest = np.maximum(np.maximum(pixels[..., 0], pixels[..., 1]), pixels[..., 2])
    return brightest >= level


def _raw_photo(data: _RawData, pixels: NDArray[np.uint16], clipped: NDArray[np.bool_]) -> DecodedPhoto:
    """Build a decoded photo from 16 bit pixels and the file's metadata.

    *pixels* may already have been shrunk to the proxy size, so the size the
    sensor gave is taken from the unshrunk pixels the file was read into.
    """
    sensor = data.full_size
    return DecodedPhoto(
        image=(pixels.astype(np.float32) / np.float32(_SIXTEEN_BIT)),
        is_raw=True,
        clipped=clipped,
        camera_to_xyz=data.camera_to_xyz,
        rgb_xyz_matrix=data.rgb_xyz_matrix,
        camera_whitebalance=data.whitebalance,
        shrunk_from=None if sensor == (pixels.shape[1], pixels.shape[0]) else sensor,
        lens=data.lens,
    )


def _decode_raw(path: Path) -> DecodedPhoto:
    """Decode a RAW file into linear light in the camera's colour space."""
    data = _read_raw(path)
    return _raw_photo(data, data.pixels, data.clipped)


# --------------------------------------------------------------------------
# JPEG and PNG
# --------------------------------------------------------------------------


def _encoded_color_space(image: Image.Image) -> tuple[ColorSpace, bytes | None]:
    """Return the colour space of a JPEG or a PNG and its embedded profile.

    The embedded profile wins. Without one, EXIF decides: ColorSpace 1 means
    sRGB, and ColorSpace 65535 with InteropIndex R03 means Adobe RGB. Anything
    else is treated as sRGB (PLAN.md section 6 step 1), which is also what the
    PNG specification says a file without a profile of its own is.
    """
    profile = image.info.get("icc_profile")
    if profile:
        name = _profile_name(profile)
        if name is not None:
            lowered = name.lower()
            if "adobe rgb" in lowered:
                return ColorSpace.ADOBE_RGB, profile
            if "display p3" in lowered or "dci-p3" in lowered or lowered.startswith("p3"):
                return ColorSpace.DISPLAY_P3, profile
            if _SRGB_PROFILE_NAME.lower() in lowered:
                return ColorSpace.SRGB, profile
            logger.info("%s uses an unrecognised profile, converting it to sRGB", name)
            return ColorSpace.SRGB, profile

    try:
        exif = image.getexif()
    except (OSError, ValueError, AttributeError):
        return ColorSpace.SRGB, None

    color_space = (exif.get_ifd(_EXIF_IFD_TAG) or {}).get(_EXIF_COLOR_SPACE_TAG)
    if color_space == _COLOR_SPACE_UNCALIBRATED:
        interop = exif.get_ifd(_EXIF_INTEROP_TAG) or {}
        index = interop.get(_EXIF_INTEROP_INDEX_TAG, "")
        if isinstance(index, bytes):
            index = index.decode("ascii", "replace")
        if str(index).strip("\x00 ") == _ADOBE_RGB_INTEROP_INDEX:
            return ColorSpace.ADOBE_RGB, None
    return ColorSpace.SRGB, None


def _profile_name(profile: bytes) -> str | None:
    """Return the description of an ICC profile, or ``None`` if unreadable."""
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))).strip()
    except (OSError, ValueError, ImageCms.PyCMSError):
        logger.warning("An embedded colour profile could not be read")
        return None


def _to_srgb_with_profile(image: Image.Image, profile: bytes) -> Image.Image:
    """Convert an image with an unrecognised profile to sRGB."""
    try:
        source = ImageCms.ImageCmsProfile(BytesIO(profile))
        return ImageCms.profileToProfile(
            image, source, ImageCms.createProfile(_SRGB_PROFILE_NAME), outputMode="RGB"
        )
    except (OSError, ValueError, ImageCms.PyCMSError):
        logger.warning("A colour profile could not be applied, using the pixels as sRGB")
        return image


def _needs_srgb_conversion(color_space: ColorSpace, profile: bytes) -> bool:
    """Return whether the file's own profile has to convert it to sRGB.

    A profile that is none of the three this module knows a matrix for is
    read as sRGB, so the pixels themselves have to be taken there first.
    """
    if color_space is not ColorSpace.SRGB:
        return False
    name = _profile_name(profile)
    return name is not None and _SRGB_PROFILE_NAME.lower() not in name.lower()


def _flattened(image: Image.Image) -> Image.Image:
    """Return *image* as RGB, compositing any transparency onto white.

    A photograph saved with an alpha channel is flattened rather than shown
    over whatever happens to be behind it, which is what an editor does when
    it writes such a file out as a photograph.
    """
    if image.mode in _ALPHA_MODES or "transparency" in image.info:
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image.convert("RGBA")).convert("RGB")
    return image if image.mode == "RGB" else image.convert("RGB")


def flattened_pixels(image: Image.Image) -> NDArray[np.uint8]:
    """Return a Pillow *image* as 8 bit RGB, transparency over white.

    The thumbnails use this too, so that a thumbnail and the photo it stands
    for flatten onto the same background.
    """
    return np.asarray(_flattened(image), dtype=np.uint8)


def _onto_white(colour: NDArray[Any], alpha: NDArray[Any]) -> NDArray[Any]:
    """Composite *colour* onto white by *alpha*, keeping the bit depth."""
    full_scale = np.float32(_EIGHT_BIT if colour.dtype == np.uint8 else _SIXTEEN_BIT)
    weight = (alpha.astype(np.float32) / full_scale)[:, :, None]
    mixed = colour.astype(np.float32) * weight + full_scale * (np.float32(1.0) - weight)
    return np.rint(mixed).astype(colour.dtype)


def _png_to_rgb(data: NDArray[Any]) -> NDArray[Any]:
    """Turn what OpenCV read of a PNG into RGB, flattening transparency.

    OpenCV hands back grey as two dimensions, colour as BGR and transparency
    as BGRA, all of it 16 bit by the time it gets here.
    """
    if data.ndim == 2:
        data = data[:, :, None]
    channels = data.shape[2]
    if channels >= 4:
        return np.ascontiguousarray(_onto_white(data[:, :, 2::-1], data[:, :, 3]))
    if channels == 3:
        return np.ascontiguousarray(data[:, :, ::-1])
    return np.ascontiguousarray(np.repeat(data[:, :, :1], 3, axis=2))


def _is_deep_png(path: Path) -> bool:
    """Return whether the PNG at *path* holds 16 bits in each channel.

    The depth is one byte of the IHDR chunk, which the PNG specification puts
    at a fixed place: the signature, the chunk's length and name, and the
    width and height come before it.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(_PNG_BIT_DEPTH_OFFSET + 1)
    except OSError:
        return False
    return len(header) > _PNG_BIT_DEPTH_OFFSET and header[_PNG_BIT_DEPTH_OFFSET] == 16


def _png_pixels(path: Path, opened: Image.Image) -> NDArray[Any]:
    """Return the pixels of the PNG at *path* as RGB, 8 or 16 bit.

    Only a 16 bit file goes through OpenCV, and only because Pillow hands
    back the high byte of each channel and nothing else. Everything else is
    read by Pillow, which is holding the file already and knows what to do
    with a palette or an alpha channel; a second decode would be for nothing.

    OpenCV is handed the file's bytes rather than its name, which is what
    lets a path like ``D:\\Photos\\2026-10 京都`` be read at all.
    """
    if not _is_deep_png(path):
        return flattened_pixels(opened)
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        logger.info("%s was not read by OpenCV, falling back to Pillow", path.name)
        return flattened_pixels(opened)
    return _png_to_rgb(data)


def _to_linear(encoded: NDArray[np.float32], color_space: ColorSpace) -> NDArray[np.float32]:
    """Decode values in 0..1 from *color_space*'s transfer curve to linear light."""
    if color_space is ColorSpace.ADOBE_RGB:
        return cm.adobe_rgb_to_linear(encoded)
    return cm.srgb_to_linear(encoded)


@lru_cache(maxsize=None)
def _linear_table(color_space: ColorSpace) -> NDArray[np.float32]:
    """Return the linear value of each of the 256 codes of an 8 bit photo."""
    codes = np.arange(256, dtype=np.float32) / np.float32(_EIGHT_BIT)
    table = _to_linear(codes, color_space)
    table.flags.writeable = False
    return table


def _encoded_photo(
    pixels: NDArray[Any], color_space: ColorSpace, profile: bytes | None
) -> DecodedPhoto:
    """Return the decoded photo for the encoded *pixels* of a JPEG or a PNG.

    *pixels* are upright RGB, 8 or 16 bit. They are kept for passthrough only
    when they are 8 bit: that is the depth a passthrough preview is shown at,
    and reducing a 16 bit PNG to fit would throw away what it was saved for
    (PLAN.md section 6, "JPG 原圖直通").
    """
    eight_bit = pixels.dtype == np.uint8
    threshold = np.float32(CLIPPING_THRESHOLD)
    if eight_bit:
        # An 8 bit photo has only 256 values per channel, so each is decoded
        # once and looked up: the same numbers, several times faster than
        # decoding tens of millions of pixels one by one. The curve only
        # rises, so a pixel is clipped exactly when its largest channel is.
        table = _linear_table(color_space)
        linear = table[pixels]
        # Pairwise maxima: a reduction along a three-wide axis is over ten
        # times slower in NumPy.
        brightest = np.maximum(np.maximum(pixels[..., 0], pixels[..., 1]), pixels[..., 2])
        clipped = (table >= threshold)[brightest]
    else:
        linear = _to_linear(pixels.astype(np.float32) / np.float32(_SIXTEEN_BIT), color_space)
        clipped = np.any(linear >= threshold, axis=2)
    working = cm.apply_matrix(linear, _ENCODED_TO_PROPHOTO[color_space])

    return DecodedPhoto(
        image=working,
        is_raw=False,
        clipped=clipped,
        color_space=color_space,
        icc_profile=profile,
        encoded=pixels if eight_bit else None,
    )


def _decode_jpeg(path: Path, proxy_long_edge: int | None = None) -> DecodedPhoto:
    """Decode a JPEG into linear light in the ProPhoto working space.

    With *proxy_long_edge*, the JPEG decoder itself scales the photo down
    by a half, a quarter or an eighth, as far as it can while staying at
    least that long: a 24 megapixel file then decodes a quarter of the
    pixels. Only the preview proxy is read this way; the export, and the
    passthrough copy of an untouched photo, use every pixel.
    """
    with Image.open(path) as opened:
        full_width, full_height = opened.size
        if proxy_long_edge is not None:
            scale = proxy_long_edge / max(full_width, full_height)
            if scale < 1:
                opened.draft(
                    opened.mode,
                    (math.ceil(full_width * scale), math.ceil(full_height * scale)),
                )
        opened.load()
        orientation = int(opened.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        color_space, profile = _encoded_color_space(opened)
        converted = opened
        if profile is not None and _needs_srgb_conversion(color_space, profile):
            converted = _to_srgb_with_profile(opened, profile)
        if converted.mode != "RGB":
            converted = converted.convert("RGB")
        pixels = np.asarray(converted, dtype=np.uint8)

    photo = _encoded_photo(apply_orientation(pixels, orientation), color_space, profile)
    full_size = (
        (full_height, full_width) if orientation in _SIDEWAYS else (full_width, full_height)
    )
    if photo.size == full_size:
        return photo
    return replace(photo, shrunk_from=full_size)


def _decode_png(path: Path) -> DecodedPhoto:
    """Decode a PNG into linear light in the ProPhoto working space.

    OpenCV reads the pixels, because Pillow hands back only the high byte of
    each channel of a 16 bit PNG and half of the file would be lost on the way
    in. Pillow still reads the iCCP profile and the eXIf block, and converts
    the rare file whose profile is none of the three known ones, which that
    conversion can only do at 8 bit.
    """
    with Image.open(path) as opened:
        opened.load()
        orientation = int(opened.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        color_space, profile = _encoded_color_space(opened)
        if profile is not None and _needs_srgb_conversion(color_space, profile):
            pixels = flattened_pixels(_to_srgb_with_profile(_flattened(opened), profile))
        else:
            pixels = _png_pixels(path, opened)

    return _encoded_photo(apply_orientation(pixels, orientation), color_space, profile)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def _decode_file(path: Path) -> DecodedPhoto:
    """Decode a supported file at full size, by what kind of file it is."""
    if is_raw(path):
        return _decode_raw(path)
    if is_png(path):
        return _decode_png(path)
    return _decode_jpeg(path)


def grain_seed_for(path: Path) -> int:
    """Return the seed of *path*'s film grain, from its name alone."""
    return zlib.crc32(path.name.encode("utf-8"))


def decode(path: Path) -> DecodedPhoto:
    """Decode the photo at *path*.

    Raises :class:`DecodeError` for anything that cannot be read, so that the
    caller can show one plain sentence instead of letting the error escape
    (PLAN.md F-06).
    """
    if not is_supported(path):
        raise DecodeError(f"{path.suffix} is not a supported format")
    try:
        return replace(_decode_file(path), grain_seed=grain_seed_for(path))
    except DecodeError:
        raise
    except (
        OSError,
        ValueError,
        MemoryError,
        UnidentifiedImageError,
        rawpy.LibRawError,
    ) as error:
        raise DecodeError(f"{path.name} could not be decoded: {error}") from error


def resize_photo(photo: DecodedPhoto, long_edge: int) -> DecodedPhoto:
    """Return *photo* at a smaller size, keeping everything else about it.

    The preview is developed at the size it will be shown at rather than at
    the size it was decoded at, which is several times less work for the same
    picture (PLAN.md section 4).
    """
    if max(photo.image.shape[:2]) <= long_edge:
        return photo
    image = make_proxy(photo.image, long_edge)
    encoded = None
    if photo.encoded is not None:
        encoded = cv2.resize(
            photo.encoded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )
    return replace(
        photo,
        image=image,
        clipped=_shrink_mask(photo.clipped, long_edge),
        encoded=encoded,
        shrunk_from=photo.full_size,
    )


def _shrink_mask(mask: NDArray[np.bool_], long_edge: int) -> NDArray[np.bool_]:
    """Shrink a boolean mask, keeping a pixel that is mostly covered.

    The mask is resized as a single channel image, not as an image with one
    channel: OpenCV drops a trailing axis of one, which would silently turn
    the result into something of the wrong shape.
    """
    height, width = mask.shape[:2]
    largest = max(height, width)
    if largest <= long_edge:
        return mask
    scale = long_edge / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    covered = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return covered > 0.5


def decode_proxy(path: Path, long_edge: int = PROXY_LONG_EDGE) -> DecodedPhoto:
    """Decode *path* straight to the preview proxy.

    The proxy is what the interface works on, so the full size image is never
    built as floating point here: a RAW file is shrunk while it is still 16
    bit integers, which is several times less work than converting twenty four
    million pixels first (PLAN.md section 4).
    """
    if not is_supported(path):
        raise DecodeError(f"{path.suffix} is not a supported format")
    try:
        return replace(_proxy_of(path, long_edge), grain_seed=grain_seed_for(path))
    except DecodeError:
        raise
    except (OSError, ValueError, MemoryError, UnidentifiedImageError, rawpy.LibRawError) as error:
        raise DecodeError(f"{path.name} could not be decoded: {error}") from error


def _proxy_of(path: Path, long_edge: int) -> DecodedPhoto:
    """Decode a supported file straight to the proxy, by what kind it is."""
    if is_png(path):
        return _encoded_proxy(_decode_png(path), long_edge)
    if not is_raw(path):
        return _encoded_proxy(_decode_jpeg(path, long_edge), long_edge)
    data = _read_raw(path, long_edge)
    return _raw_photo(
        data,
        _shrink_pixels(data.pixels, long_edge),
        _shrink_mask(data.clipped, long_edge),
    )


def _shrink_pixels(pixels: NDArray[np.uint16], long_edge: int) -> NDArray[np.uint16]:
    """Shrink 16 bit pixels to the proxy size."""
    height, width = pixels.shape[:2]
    largest = max(height, width)
    if largest <= long_edge:
        return pixels
    scale = long_edge / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)


def _encoded_proxy(full: DecodedPhoto, long_edge: int) -> DecodedPhoto:
    """Return the proxy of an already decoded JPEG or PNG."""
    proxy_image = make_proxy(full.image, long_edge)
    if proxy_image is full.image:
        return full
    encoded = None
    if full.encoded is not None:
        encoded = cv2.resize(
            full.encoded,
            (proxy_image.shape[1], proxy_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return replace(
        full,
        image=proxy_image,
        clipped=_shrink_mask(full.clipped, long_edge),
        encoded=encoded,
        shrunk_from=full.full_size,
    )


def embedded_preview(path: Path) -> NDArray[np.uint8] | None:
    """Return the JPEG preview a RAW file carries, as 8 bit RGB.

    Taking this costs about a millisecond and lets the photo appear while the
    full decode is still running (PLAN.md section 3.7).
    """
    found = embedded_preview_and_size(path)
    return None if found is None else found[0]


def embedded_preview_and_size(
    path: Path,
) -> tuple[NDArray[np.uint8], tuple[int, int]] | None:
    """Return a RAW file's own preview and the size the photo has in full.

    The size is what the decode will report as ``full_size``: the camera's
    own crop of the sensor, turned upright. Read from the same opening of the
    file as the preview, it lets the zoom be measured against the photo
    rather than against the much smaller preview while the photo is still
    being developed (PLAN.md 3.3, 3.7).
    """
    if not is_raw(path):
        return None
    try:
        with rawpy.imread(str(path)) as raw:
            # Read the rotation first, for the same reason as in _read_raw.
            orientation = libraw_flip_to_orientation(int(raw.sizes.flip))
            sensor_landscape = raw.sizes.width >= raw.sizes.height
            crop = _default_crop(raw.sizes, 1)
            if crop is not None:
                width, height = crop.full_width, crop.full_height
            else:
                width, height = int(raw.sizes.width), int(raw.sizes.height)
            thumb = raw.extract_thumb()
    except (OSError, ValueError, rawpy.LibRawError) as error:
        logger.info("%s has no usable embedded preview: %s", path.name, error)
        return None
    full_size = (height, width) if orientation in _SIDEWAYS else (width, height)
    preview = _thumb_pixels(thumb)
    if preview is None:
        return None
    return _orient_preview(preview, orientation, sensor_landscape), full_size


def _thumb_pixels(thumb: Any) -> NDArray[np.uint8] | None:
    """Return the pixels of an embedded preview, or ``None`` if it has none."""

    if thumb.format is rawpy.ThumbFormat.JPEG:
        buffer = np.frombuffer(thumb.data, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if thumb.format is rawpy.ThumbFormat.BITMAP:
        return np.asarray(thumb.data, dtype=np.uint8)
    return None


def _orient_preview(
    preview: NDArray[np.uint8], orientation: int, sensor_landscape: bool
) -> NDArray[np.uint8]:
    """Turn an embedded preview upright, if it is not already.

    Cameras differ: some bodies store the preview of a photo taken sideways
    rotated, while others store it the way the sensor read it. Rotating a
    preview that was already upright is as wrong as not rotating one that was
    not, so the decision is made by looking at the picture: a preview that
    still has the proportions of the sensor is the one that needs turning.

    A photo taken upside down cannot be told apart this way, since turning it
    does not change its proportions. It is left alone, which matches the
    cameras that store the preview already oriented.
    """
    if orientation not in (5, 6, 7, 8):
        return preview
    preview_landscape = preview.shape[1] >= preview.shape[0]
    if preview_landscape != sensor_landscape:
        return preview
    return apply_orientation(preview, orientation)
