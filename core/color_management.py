"""Colour spaces, chromatic adaptation and OkLCh (PLAN.md section 6).

Every colour matrix used by the application lives here, built from the
published standard values, so that the processing pipeline never contains a
matrix of its own. All image functions take and return ``float32`` arrays of
shape ``(height, width, 3)`` holding linear light, and never modify their
input.

The working space is linear ProPhoto RGB with a D50 white point; input is
adapted to it with the Bradford transform, and saturation and vibrance are
applied in OkLCh, which is perceptually uniform.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from io import BytesIO

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# White points (CIE XYZ, normalised to Y = 1)
# --------------------------------------------------------------------------

WHITE_D65: Final[NDArray[np.float64]] = np.array([0.95047, 1.0, 1.08883])
WHITE_D50: Final[NDArray[np.float64]] = np.array([0.96422, 1.0, 0.82521])

# --------------------------------------------------------------------------
# RGB primaries as matrices from linear RGB to CIE XYZ
# --------------------------------------------------------------------------

#: sRGB / IEC 61966-2-1, D65.
SRGB_TO_XYZ: Final[NDArray[np.float64]] = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

#: Adobe RGB (1998), D65.
ADOBE_RGB_TO_XYZ: Final[NDArray[np.float64]] = np.array(
    [
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ]
)

#: Display P3: DCI-P3 primaries with the D65 white point.
DISPLAY_P3_TO_XYZ: Final[NDArray[np.float64]] = np.array(
    [
        [0.4865709, 0.2656677, 0.1982173],
        [0.2289746, 0.6917385, 0.0792869],
        [0.0000000, 0.0451134, 1.0439444],
    ]
)

#: ProPhoto RGB (ROMM), D50. The working space of the pipeline.
PROPHOTO_TO_XYZ: Final[NDArray[np.float64]] = np.array(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100],
    ]
)

XYZ_TO_SRGB: Final[NDArray[np.float64]] = np.linalg.inv(SRGB_TO_XYZ)
XYZ_TO_ADOBE_RGB: Final[NDArray[np.float64]] = np.linalg.inv(ADOBE_RGB_TO_XYZ)
XYZ_TO_DISPLAY_P3: Final[NDArray[np.float64]] = np.linalg.inv(DISPLAY_P3_TO_XYZ)
XYZ_TO_PROPHOTO: Final[NDArray[np.float64]] = np.linalg.inv(PROPHOTO_TO_XYZ)

#: Relative luminance coefficients of the working space, used for the tone
#: adjustments in PLAN.md section 6 step 7.
PROPHOTO_LUMINANCE: Final[NDArray[np.float64]] = PROPHOTO_TO_XYZ[1]

# --------------------------------------------------------------------------
# Chromatic adaptation (Bradford)
# --------------------------------------------------------------------------

BRADFORD: Final[NDArray[np.float64]] = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)


def bradford_adaptation(
    source_white: NDArray[np.float64], destination_white: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the XYZ matrix adapting *source_white* to *destination_white*."""
    source_cone = BRADFORD @ source_white
    destination_cone = BRADFORD @ destination_white
    scale = np.diag(destination_cone / source_cone)
    return np.linalg.inv(BRADFORD) @ scale @ BRADFORD


ADAPT_D65_TO_D50: Final[NDArray[np.float64]] = bradford_adaptation(WHITE_D65, WHITE_D50)
ADAPT_D50_TO_D65: Final[NDArray[np.float64]] = bradford_adaptation(WHITE_D50, WHITE_D65)

# --------------------------------------------------------------------------
# Brightness and colour apart, for the detail step (PLAN.md 6 step 2.5)
# --------------------------------------------------------------------------

#: Splits three channels into one of brightness - their mean - and two of
#: colour: red against blue, and green against the other two. Noise is
#: reduced separately in the two, because colour noise wants a far wider
#: reach than detail in the brightness can stand, and sharpening touches the
#: brightness alone, so that it cannot put a coloured fringe on an edge.
#:
#: A grey has nothing in either colour row, and the brightness row has the
#: channels' own weights, so the inverse spreads a change of brightness over
#: the three channels evenly: what the sharpening adds to it changes no
#: colour.
OPPONENT: Final[NDArray[np.float64]] = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.0, -0.5],
        [-0.25, 0.5, -0.25],
    ]
)
OPPONENT_INVERSE: Final[NDArray[np.float64]] = np.linalg.inv(OPPONENT)

# --------------------------------------------------------------------------
# Transfer functions
# --------------------------------------------------------------------------

_SRGB_LINEAR_THRESHOLD: Final[float] = 0.04045
_SRGB_ENCODED_THRESHOLD: Final[float] = 0.0031308
_SRGB_SLOPE: Final[float] = 12.92
_SRGB_OFFSET: Final[float] = 0.055
_SRGB_EXPONENT: Final[float] = 2.4

#: Adobe RGB (1998) uses a pure gamma of 563/256.
_ADOBE_RGB_GAMMA: Final[float] = 563.0 / 256.0


def power(
    values: NDArray[np.float32], exponent: float, assume_positive: bool = False
) -> NDArray[np.float32]:
    """Raise the magnitude of *values* to *exponent*, keeping their sign.

    OpenCV's power is several times faster than NumPy's on the array sizes the
    preview works with, and it is used throughout for that reason. It works on
    magnitudes, so the sign is restored afterwards unless the caller knows
    there are no negative values, which saves two passes over the image.
    """
    array = np.ascontiguousarray(values, dtype=np.float32)
    magnitude = cv2.pow(array if assume_positive else np.abs(array), float(exponent))
    if magnitude.shape != array.shape:  # OpenCV drops a trailing axis of one
        magnitude = magnitude.reshape(array.shape)
    return magnitude if assume_positive else np.copysign(magnitude, array)


def scale_channels(
    image: NDArray[np.float32], scale: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Multiply every channel of each pixel by the same number.

    Written through OpenCV rather than as ``image * scale[..., None]``: the
    NumPy form broadcasts a column over three channels one pixel at a time on
    a single core, while this one spreads whole rows over every core. On a
    preview sized photo it is two and a half times faster, and the tone steps
    do it three times each.
    """
    plane = np.ascontiguousarray(scale, dtype=np.float32)
    return cv2.multiply(
        np.ascontiguousarray(image, dtype=np.float32), cv2.merge((plane, plane, plane))
    )


def channel_max(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the largest of the three channels of each pixel.

    Written as two elementwise maxima rather than ``image.max(axis=2)``, which
    is several times slower on an image this shape.
    """
    return np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])


def channel_min(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the smallest of the three channels of each pixel."""
    return np.minimum(np.minimum(image[..., 0], image[..., 1]), image[..., 2])


def srgb_to_linear(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Decode sRGB (and Display P3) encoded values to linear light."""
    encoded = np.asarray(values, dtype=np.float32)
    linear = power(
        (np.abs(encoded) + np.float32(_SRGB_OFFSET)) / np.float32(1 + _SRGB_OFFSET),
        _SRGB_EXPONENT,
        assume_positive=True,
    )
    below = encoded <= np.float32(_SRGB_LINEAR_THRESHOLD)
    linear[below] = encoded[below] / np.float32(_SRGB_SLOPE)
    return linear


def linear_to_srgb(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Encode linear light with the sRGB transfer function.

    """
    linear = np.asarray(values, dtype=np.float32)
    # Negative values are replaced by the straight line section below, so
    # the power can skip putting signs back.
    encoded = np.float32(1 + _SRGB_OFFSET) * power(
        np.maximum(linear, np.float32(0.0)), 1.0 / _SRGB_EXPONENT, assume_positive=True
    ) - np.float32(_SRGB_OFFSET)
    below = linear <= np.float32(_SRGB_ENCODED_THRESHOLD)
    encoded[below] = linear[below] * np.float32(_SRGB_SLOPE)
    return encoded


def adobe_rgb_to_linear(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Decode Adobe RGB (1998) encoded values to linear light."""
    return power(np.asarray(values, dtype=np.float32), _ADOBE_RGB_GAMMA)


# --------------------------------------------------------------------------
# Applying matrices to images
# --------------------------------------------------------------------------


def apply_matrix(image: NDArray[np.float32], matrix: NDArray[np.float64]) -> NDArray[np.float32]:
    """Return *image* with the 3 x 3 *matrix* applied to its colour channels."""
    array = np.asarray(image, dtype=np.float32)
    transposed = np.asarray(matrix, dtype=np.float32).T
    return (array.reshape(-1, 3) @ transposed).reshape(array.shape)


def camera_to_xyz_matrix(rgb_xyz_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the matrix from a camera's native RGB to CIE XYZ.

    LibRaw reports ``rgb_xyz_matrix``, which converts XYZ to camera RGB. Its
    rows are scaled here so that camera neutral, meaning equal values in all
    three channels after white balancing, maps exactly onto the D65 white
    point (PLAN.md section 6 step 5), and the result is then inverted.
    """
    xyz_to_camera = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    neutral = xyz_to_camera @ WHITE_D65
    if not np.all(np.isfinite(neutral)) or np.any(np.abs(neutral) < 1e-9):
        raise ValueError("camera colour matrix does not describe a usable neutral")
    normalised = xyz_to_camera / neutral[:, np.newaxis]
    if abs(np.linalg.det(normalised)) < 1e-9:
        raise ValueError("camera colour matrix is singular")
    return np.linalg.inv(normalised)


# --------------------------------------------------------------------------
# Oklab and OkLCh  (Bjorn Ottosson's Oklab, defined against XYZ with D65)
# --------------------------------------------------------------------------

_XYZ_TO_LMS: Final[NDArray[np.float64]] = np.array(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ]
)

_LMS_TO_OKLAB: Final[NDArray[np.float64]] = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)

_LMS_TO_XYZ: Final[NDArray[np.float64]] = np.linalg.inv(_XYZ_TO_LMS)
_OKLAB_TO_LMS: Final[NDArray[np.float64]] = np.linalg.inv(_LMS_TO_OKLAB)

#: Oklab to the cube roots of the cone responses. Exposed because the gamut
#: search works directly in this space: it is the last step that is still
#: linear in Oklab, so a whole line of constant hue can be walked with one
#: multiply and add instead of a full colour conversion per step.
OKLAB_TO_CONE_ROOT: Final[NDArray[np.float64]] = _OKLAB_TO_LMS

#: The cone responses straight to linear sRGB, skipping XYZ and the working
#: space. Both are D65, so no adaptation is needed.
CONE_TO_SRGB_LINEAR: Final[NDArray[np.float64]] = XYZ_TO_SRGB @ _LMS_TO_XYZ

#: And the way back. A film look (:mod:`core.filters`) runs its tone curves
#: in encoded sRGB and its hue functions in Oklab, so it arrives at Oklab
#: from sRGB rather than from the working space; going round by ProPhoto in
#: between would cost a matrix over every pixel for nothing.
SRGB_LINEAR_TO_CONE: Final[NDArray[np.float64]] = _XYZ_TO_LMS @ SRGB_TO_XYZ


def xyz_to_oklab(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert XYZ with a D65 white point to Oklab."""
    cone = apply_matrix(image, _XYZ_TO_LMS)
    return apply_matrix(power(cone, 1.0 / 3.0), _LMS_TO_OKLAB)


def oklab_to_xyz(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Oklab back to XYZ with a D65 white point."""
    cone_root = apply_matrix(image, _OKLAB_TO_LMS)
    return apply_matrix(cone_root * cone_root * cone_root, _LMS_TO_XYZ)


def oklab_to_oklch(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Oklab to OkLCh: lightness, chroma, hue in radians."""
    lab = np.asarray(image, dtype=np.float32)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack((lab[..., 0], chroma, hue), axis=-1).astype(np.float32)


def oklch_to_oklab(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert OkLCh back to Oklab."""
    lch = np.asarray(image, dtype=np.float32)
    a = lch[..., 1] * np.cos(lch[..., 2])
    b = lch[..., 1] * np.sin(lch[..., 2])
    return np.stack((lch[..., 0], a, b), axis=-1).astype(np.float32)


# --------------------------------------------------------------------------
# Convenience conversions between the spaces the pipeline uses
# --------------------------------------------------------------------------

#: Linear ProPhoto (D50) to XYZ with a D65 white point, for the OkLCh steps.
PROPHOTO_TO_XYZ_D65: Final[NDArray[np.float64]] = ADAPT_D50_TO_D65 @ PROPHOTO_TO_XYZ
XYZ_D65_TO_PROPHOTO: Final[NDArray[np.float64]] = XYZ_TO_PROPHOTO @ ADAPT_D65_TO_D50

#: Linear ProPhoto (D50) to linear sRGB (D65), the output transform.
PROPHOTO_TO_SRGB: Final[NDArray[np.float64]] = XYZ_TO_SRGB @ PROPHOTO_TO_XYZ_D65

#: Input spaces to the linear ProPhoto working space.
SRGB_TO_PROPHOTO: Final[NDArray[np.float64]] = XYZ_TO_PROPHOTO @ ADAPT_D65_TO_D50 @ SRGB_TO_XYZ
ADOBE_RGB_TO_PROPHOTO: Final[NDArray[np.float64]] = (
    XYZ_TO_PROPHOTO @ ADAPT_D65_TO_D50 @ ADOBE_RGB_TO_XYZ
)
DISPLAY_P3_TO_PROPHOTO: Final[NDArray[np.float64]] = (
    XYZ_TO_PROPHOTO @ ADAPT_D65_TO_D50 @ DISPLAY_P3_TO_XYZ
)


#: The working space straight to the cone responses, and back. Three
#: matrices in a row are one matrix, and multiplying them once here saves two
#: passes over the photo each way (PLAN.md section 6 step 10).
PROPHOTO_TO_CONE: Final[NDArray[np.float64]] = _XYZ_TO_LMS @ PROPHOTO_TO_XYZ_D65
CONE_TO_PROPHOTO: Final[NDArray[np.float64]] = XYZ_D65_TO_PROPHOTO @ _LMS_TO_XYZ


def prophoto_to_oklab(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear ProPhoto RGB to Oklab."""
    cone = apply_matrix(image, PROPHOTO_TO_CONE)
    return apply_matrix(power(cone, 1.0 / 3.0), _LMS_TO_OKLAB)


def oklab_to_prophoto(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert Oklab back to linear ProPhoto RGB."""
    cone_root = apply_matrix(image, _OKLAB_TO_LMS)
    return apply_matrix(cone_root * cone_root * cone_root, CONE_TO_PROPHOTO)


def oklab_chroma_matrix(factor: float) -> NDArray[np.float64]:
    """Return the map of cone roots that scales Oklab chroma by *factor*.

    Oklab is a matrix away from the cube roots of the cone responses, and
    scaling a and b by one number is a matrix too, so the three are one.
    """
    scale = np.diag([1.0, factor, factor])
    return _OKLAB_TO_LMS @ scale @ _LMS_TO_OKLAB


def scale_oklab_chroma(image: NDArray[np.float32], factor: float) -> NDArray[np.float32]:
    """Scale the Oklab chroma of linear ProPhoto RGB, keeping L and hue.

    The same as going to Oklab, multiplying a and b and coming back, with
    the middle three matrices folded into one: on a preview sized photo that
    is a whole pass over the pixels each way saved.
    """
    cone = apply_matrix(image, PROPHOTO_TO_CONE)
    root = apply_matrix(power(cone, 1.0 / 3.0), oklab_chroma_matrix(factor))
    return apply_matrix(root * root * root, CONE_TO_PROPHOTO)


def srgb_linear_to_oklab(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear sRGB to Oklab without passing through the working space."""
    cone = apply_matrix(image, SRGB_LINEAR_TO_CONE)
    return apply_matrix(power(cone, 1.0 / 3.0), _LMS_TO_OKLAB)


def prophoto_to_oklch(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert linear ProPhoto RGB to OkLCh.

    Going all the way to polar coordinates costs an arc tangent and, coming
    back, a sine and a cosine for every pixel. Anything that only scales
    chroma can skip all three by staying in Oklab, where scaling chroma is
    scaling a and b together, so prefer :func:`prophoto_to_oklab` unless the
    hue angle itself is wanted.
    """
    return oklab_to_oklch(prophoto_to_oklab(image))


def oklch_to_prophoto(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert OkLCh back to linear ProPhoto RGB."""
    return oklab_to_prophoto(oklch_to_oklab(image))


# --------------------------------------------------------------------------
# Colour temperature  (PLAN.md section 6 step 3)
# --------------------------------------------------------------------------
#
# White balance is a point on the Planckian locus plus an offset at right
# angles to it, which is how DNG and every editor describe it: the
# temperature says how far along the locus, the tint how far off it.
#
# The locus is evaluated with Kim, McKay-Allen and Rogers' published cubic
# approximation rather than by integrating Planck's law against the colour
# matching functions, which would need a large table to no better effect
# here. The direction of the tint offset is worked out from the locus itself,
# by taking the perpendicular to it in the 1960 UCS diagram, where the lines
# of equal temperature are perpendicular to the locus by definition.

#: The range the cubic approximation of the locus is defined over.
_LOCUS_MIN_K: Final[float] = 1667.0
_LOCUS_MAX_K: Final[float] = 25000.0

#: Tint units per unit of v in the 1960 UCS diagram. Chosen so that the plus
#: or minus 150 of PLAN.md section 2.2 covers the green to magenta range a
#: camera can record, which is also the scale DNG uses.
TINT_PER_UV: Final[float] = 3000.0

#: Step used to find the direction of the locus by difference, in mired.
_LOCUS_STEP_MIRED: Final[float] = 0.5


def kelvin_to_mired(kelvin: float) -> float:
    """Return the reciprocal temperature, in micro reciprocal degrees.

    Sliders move in mired because equal steps there look like equal changes,
    while equal steps in kelvin do not: 1000 K matters near 3000 K and is
    barely visible near 9000 K (PLAN.md section 6 step 3).
    """
    return 1.0e6 / max(float(kelvin), 1.0)


def mired_to_kelvin(mired: float) -> float:
    """Return the temperature in kelvin for a reciprocal temperature."""
    return 1.0e6 / max(float(mired), 1e-6)


def planckian_xy(kelvin: float) -> tuple[float, float]:
    """Return the CIE xy chromaticity of a black body at *kelvin*."""
    temperature = min(max(float(kelvin), _LOCUS_MIN_K), _LOCUS_MAX_K)
    inverse = 1000.0 / temperature
    if temperature <= 4000.0:
        x = (
            -0.2661239 * inverse**3
            - 0.2343589 * inverse**2
            + 0.8776956 * inverse
            + 0.179910
        )
    else:
        x = (
            -3.0258469 * inverse**3
            + 2.1070379 * inverse**2
            + 0.2226347 * inverse
            + 0.240390
        )
    if temperature <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif temperature <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def xy_to_uv(x: float, y: float) -> tuple[float, float]:
    """Convert CIE xy to the 1960 UCS diagram, where isotherms are straight."""
    denominator = -2.0 * x + 12.0 * y + 3.0
    if abs(denominator) < 1e-12:
        return 0.0, 0.0
    return 4.0 * x / denominator, 6.0 * y / denominator


def uv_to_xy(u: float, v: float) -> tuple[float, float]:
    """Convert a point of the 1960 UCS diagram back to CIE xy."""
    denominator = 2.0 * u - 8.0 * v + 4.0
    if abs(denominator) < 1e-12:
        return 0.0, 0.0
    return 3.0 * u / denominator, 2.0 * v / denominator


def _locus_direction(kelvin: float) -> tuple[float, float]:
    """Return the unit vector along the locus at *kelvin*, in uv."""
    mired = kelvin_to_mired(kelvin)
    before = xy_to_uv(*planckian_xy(mired_to_kelvin(mired - _LOCUS_STEP_MIRED)))
    after = xy_to_uv(*planckian_xy(mired_to_kelvin(mired + _LOCUS_STEP_MIRED)))
    du = after[0] - before[0]
    dv = after[1] - before[1]
    length = (du * du + dv * dv) ** 0.5
    if length < 1e-12:
        return 1.0, 0.0
    return du / length, dv / length


def temperature_to_xy(kelvin: float, tint: float = 0.0) -> tuple[float, float]:
    """Return the white point for a temperature and a tint.

    The tint moves the point at right angles to the locus, along the line of
    equal temperature, so changing it does not change how warm the photo
    looks, only how green or magenta it is.
    """
    u, v = xy_to_uv(*planckian_xy(kelvin))
    along_u, along_v = _locus_direction(kelvin)
    offset = float(tint) / TINT_PER_UV
    return uv_to_xy(u - along_v * offset, v + along_u * offset)


def xy_to_temperature(x: float, y: float) -> tuple[float, float]:
    """Return the temperature and tint of a white point.

    The temperature is the point of the locus nearest to it, and the tint is
    how far it lies off the locus there. The search runs over mired rather
    than kelvin, which keeps its steps even along the locus.
    """
    target_u, target_v = xy_to_uv(x, y)

    def distance(mired: float) -> float:
        u, v = xy_to_uv(*planckian_xy(mired_to_kelvin(mired)))
        return (u - target_u) ** 2 + (v - target_v) ** 2

    # The distance along the locus has a single minimum, so narrowing the
    # bracket from both ends finds it without needing a derivative.
    low = kelvin_to_mired(_LOCUS_MAX_K)
    high = kelvin_to_mired(_LOCUS_MIN_K)
    for _ in range(80):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if distance(first) < distance(second):
            high = second
        else:
            low = first

    kelvin = mired_to_kelvin((low + high) / 2.0)
    u, v = xy_to_uv(*planckian_xy(kelvin))
    along_u, along_v = _locus_direction(kelvin)
    tint = ((target_v - v) * along_u - (target_u - u) * along_v) * TINT_PER_UV
    return kelvin, tint


def camera_neutral_to_temperature(
    neutral: NDArray[np.float64], rgb_xyz_matrix: NDArray[np.float64]
) -> tuple[float, float]:
    """Return the temperature and tint a camera's neutral stands for.

    *neutral* is what the sensor reads for a white subject, which is the
    reciprocal of the white balance multipliers the camera recorded.
    """
    xyz_to_camera = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    try:
        xyz = np.linalg.solve(xyz_to_camera, np.asarray(neutral, dtype=np.float64))
    except np.linalg.LinAlgError:
        return 6500.0, 0.0
    total = float(xyz.sum())
    if abs(total) < 1e-12:
        return 6500.0, 0.0
    return xy_to_temperature(float(xyz[0]) / total, float(xyz[1]) / total)


def temperature_to_camera_neutral(
    kelvin: float, tint: float, rgb_xyz_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return what a camera's sensor reads for white at this temperature."""
    x, y = temperature_to_xy(kelvin, tint)
    if y <= 1e-12:
        return np.ones(3)
    xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
    xyz_to_camera = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    neutral = xyz_to_camera @ xyz
    largest = float(np.max(np.abs(neutral)))
    return neutral / max(largest, 1e-12)


def whitebalance_multipliers(
    kelvin: float, tint: float, rgb_xyz_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the gains that make this white point neutral, with green at one."""
    neutral = temperature_to_camera_neutral(kelvin, tint, rgb_xyz_matrix)
    safe = np.where(np.abs(neutral) < 1e-12, 1e-12, neutral)
    multipliers = 1.0 / safe
    return multipliers / multipliers[1]


# --------------------------------------------------------------------------
# The screen's own profile  (PLAN.md section 6 step 13)
# --------------------------------------------------------------------------
#
# Only the preview goes through this. An exported file is always plain sRGB,
# because it has to look the same on somebody else's screen.


def kelvin_to_relative(kelvin: float, shot: float, low: float, high: float) -> float:
    """Return a RAW temperature as a nudge from the camera's own, -100 to 100.

    Offered as an alternative to kelvin in the settings (PLAN.md 3.10). The
    whole of the kelvin range is still reachable: zero is what the camera
    recorded and each side is measured against its own end of the range, in
    reciprocal temperature so that a step feels the same at both ends.
    """
    mired = kelvin_to_mired(kelvin)
    middle = kelvin_to_mired(shot)
    if mired <= middle:
        span = middle - kelvin_to_mired(high)
        return 0.0 if span <= 0 else min(100.0, 100.0 * (middle - mired) / span)
    span = kelvin_to_mired(low) - middle
    return 0.0 if span <= 0 else max(-100.0, -100.0 * (mired - middle) / span)


def relative_to_kelvin(relative: float, shot: float, low: float, high: float) -> float:
    """Return the kelvin a nudge of *relative* from *shot* stands for."""
    middle = kelvin_to_mired(shot)
    if relative >= 0.0:
        span = middle - kelvin_to_mired(high)
        return mired_to_kelvin(middle - span * min(relative, 100.0) / 100.0)
    span = kelvin_to_mired(low) - middle
    return mired_to_kelvin(middle + span * min(-relative, 100.0) / 100.0)


@lru_cache(maxsize=1)
def srgb_profile_bytes() -> bytes:

    """Return a standard sRGB ICC profile, built here rather than shipped.

    PLAN.md section 13.2 keeps third party profile files out of the project:
    the profile every exported file carries (X-04) is produced by Pillow at
    runtime. It never changes, so it is built once.
    """
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def build_display_transform(profile: bytes | None) -> object | None:

    """Return a transform from sRGB to the screen's own colours.

    Returns ``None`` when there is no profile, when it cannot be read, or
    when it is sRGB already and the transform would do nothing.
    """
    if not profile:
        return None
    try:
        screen = ImageCms.ImageCmsProfile(BytesIO(profile))
        name = ImageCms.getProfileDescription(screen).strip().lower()
        if "srgb" in name:
            return None
        return ImageCms.buildTransform(
            ImageCms.createProfile("sRGB"),
            screen,
            "RGB",
            "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        )
    except (OSError, ValueError, ImageCms.PyCMSError):
        logger.info("The screen's colour profile could not be used")
        return None


def apply_display_transform(
    image: NDArray[np.uint8], transform: object | None
) -> NDArray[np.uint8]:
    """Return *image* converted for the screen it will be shown on."""
    if transform is None:
        return image
    try:
        picture = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
        ImageCms.applyTransform(picture, transform, inPlace=True)
        return np.asarray(picture, dtype=np.uint8)
    except (OSError, ValueError, TypeError, AttributeError, ImageCms.PyCMSError):
        # Showing the photo matters more than showing it in the screen's own
        # colours, and this runs on a background thread, so anything that goes
        # wrong here gives back the photo as it was.
        logger.info("The screen's colour profile could not be applied")
        return image
