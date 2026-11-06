"""The image processing pipeline (PLAN.md section 6).

Every function here is pure: it takes an array and returns a new one, never
modifies its input, never touches a file and keeps no state. The preview and
the export run exactly the same code; only the size of the input differs.

All arrays are ``float32`` linear light of shape ``(height, width, 3)``. The
working space is linear ProPhoto RGB with a D50 white point.

Steps 2.5 to 12, with 2.6, live here. The screen's own profile, step 13, is applied by
whoever shows the result rather than here, because it must never reach an
exported file.

Every coefficient the look depends on is a constant in the block below, so
that tuning never means hunting through the code.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core import color_management as cm
from core import faces as faces_module
from core import geometry as geometry_module
from core import skin_range
from core.decoder import PROXY_LONG_EDGE, ColorSpace, DecodedPhoto
from core.filters import Look
from core.geometry import Geometry, Window
from core.edit_state import EditState

# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------

#: Softening of the clipped highlight mask, as a fraction of the long edge.
#: PLAN.md section 6 step 4 gives 2 px; it is expressed relative to the long
#: edge of the preview proxy so that the preview and the full size export get
#: the same result instead of differing by their scale.
CLIPPED_BLUR_FRACTION: Final[float] = 2.0 / PROXY_LONG_EDGE

#: How far a blur of a given sigma reaches, in sigmas. Beyond four the
#: Gaussian is below a thousandth of its peak and no longer changes an 8 bit
#: result, so this is what a window must be grown by before it is developed
#: on its own (PLAN.md section 4).
BLUR_REACH_SIGMAS: Final[float] = 4.0

#: Base tone curve (PLAN.md section 6 step 9), applied to RAW only.
#: Middle grey in, middle grey out, with the slope through that point given in
#: log-log space. The curve is y = x^k / (x^k + c^k), which has a gentle toe,
#: a smooth shoulder and approaches white without ever reaching it.
#: The slope was chosen by comparing eight sample photos against the camera's
#: own rendering of the same exposures: 1.4 halves the contrast difference
#: that 1.2 left.
TONE_MID_INPUT: Final[float] = 0.18
TONE_MID_OUTPUT: Final[float] = 0.18
TONE_MID_SLOPE: Final[float] = 1.4

#: Gain applied to RAW before the curve, in stops.
#: LibRaw hands over linear data scaled to the point where the sensor
#: saturates, not to a rendered middle grey. A correctly exposed middle grey
#: sits 18% below diffuse white, and diffuse white sits about two and a half
#: stops below saturation, so middle grey arrives at roughly 3% of full scale
#: and has to be lifted by that much to land on TONE_MID_INPUT.
#: 2.4 rather than the 2.5 that argument gives is where eight sample photos
#: sit closest to the camera's own rendering, which is near enough to the
#: derivation to keep it. Still provisional: PLAN.md section 6 asks for these
#: to be settled in stage 2 against a reference editor as well as the camera.
RAW_BASE_EXPOSURE_EV: Final[float] = 2.4

#: Chroma a RAW photo is given with every slider at rest (PLAN.md section 6
#: step 10). The base tone curve scales the three channels together so as not
#: to move the hue, which also means it adds no colour, where a camera's
#: rendering does: measured on 2026-10-28, a RAW developed without this had
#: 66 to 72 per cent of the chroma of a reference editor's default rendering and 67 per
#: cent of the camera's own JPEG of the same exposure. 1.35 brings that to
#: 89 to 97 and 86 per cent. A JPEG already carries the camera's colour and
#: never gets it, and a film filter, which was measured as the whole way from
#: our plain rendering to the camera's, takes its place as it is turned up.
RAW_BASE_CHROMA: Final[float] = 1.35

#: Gamut compression (PLAN.md section 6 step 11).
#: Where the compression starts, as a fraction of the boundary. At 0.90 an
#: untouched sRGB photo put through the pipeline lost colour in 8 per cent of
#: its pixels, a sunset visibly paler than the camera made it; at 0.97 that
#: is 0.4 per cent, while the last three per cent are still rolled off rather
#: than cut.
GAMUT_COMPRESSION_START: Final[float] = 0.97
GAMUT_MAX_CHROMA: Final[float] = 0.40
GAMUT_SEARCH_ITERATIONS: Final[int] = 10
#: How far out the cheap RGB test looks for pixels that may need compressing.
#: Wider than the compression range itself, so that the difference between
#: pushing away from a neutral in RGB and raising chroma in OkLCh cannot let a
#: pixel slip through untreated.
GAMUT_PROBE_MARGIN: Final[float] = 0.80

#: Triangular dither added before quantising to 8 bits, in output steps.
DITHER_AMPLITUDE_STEPS: Final[float] = 0.5

# -- white balance for a JPEG (PLAN.md section 6 step 3) -------------------
#: A JPEG has already been balanced, so its sliders only nudge it. The gains
#: are powers of two so that equal slider steps feel equal.
JPEG_TEMPERATURE_STOPS: Final[float] = 0.6
JPEG_TINT_STOPS: Final[float] = 0.4

# -- the white balance eyedropper (PLAN.md section 6 step 3) --------------
#: How far around the clicked point the eyedropper averages, as a fraction of
#: the long edge: 3 px of the preview proxy, a 7 by 7 patch, which is enough
#: to average the noise out of a grey wall and small enough to stay on it.
PICK_RADIUS_FRACTION: Final[float] = 3.0 / PROXY_LONG_EDGE
#: Below this level in every channel the patch is mostly noise, and the
#: balance worked out from it would be a guess (linear light, before the RAW
#: base exposure: about sRGB 26 of 255 once a RAW is developed).
PICK_DARKEST: Final[float] = 0.002

# -- highlights and shadows (PLAN.md section 6 step 7) --------------------
#: Blur applied to the local brightness, as a fraction of the long edge.
#: Judging a pixel by its surroundings rather than by itself is what keeps a
#: halo from appearing along a high contrast edge.
LOCAL_BRIGHTNESS_BLUR: Final[float] = 0.01
#: How much the sliders can move a region, in stops, at their ends.
TONE_REGION_MAX_EV: Final[float] = 1.5
#: Where each region sits relative to middle grey, in stops, and how quickly
#: its influence falls away.
SHADOW_PIVOT_EV: Final[float] = -2.0
HIGHLIGHT_PIVOT_EV: Final[float] = 1.5
#: How sharply each region gives way to the midtones. The two ends of this
#: pull against each other: a sharper falloff leaves the midtones alone, and
#: also makes the correction change faster across a high contrast edge, which
#: is what a halo is made of. Measured on a black to white edge, 1.5 reaches
#: 1.35 of the 1.5 stops at the extremes while moving middle grey by only a
#: sixth, and its correction still changes four times more gently across the
#: edge than one judging each pixel on its own. Provisional until the
#: comparison with a reference editor that PLAN.md section 9 asks for.
REGION_FALLOFF_PER_EV: Final[float] = 1.5

# -- contrast (PLAN.md section 6 step 8) ----------------------------------
#: Slope added at the ends of the contrast slider, around middle grey.
CONTRAST_SLOPE_RANGE: Final[float] = 0.6

# -- black and white points (PLAN.md section 6 step 9.5) ------------------
#: How far the black point moves at the ends of the blacks slider, in linear
#: light with white at 1.0. At 0.01 the slider's lower end takes everything
#: below a tenth of the encoded range down to pure black, and its upper end
#: lifts pure black to the same place; strong, but that is what the ends of
#: the slider are for. Provisional, like the other coefficients of section 6:
#: to be settled against real photos.
BLACK_POINT_RANGE: Final[float] = 0.01
#: How far the white point moves at the ends of the whites slider, in stops.
WHITE_POINT_MAX_EV: Final[float] = 1.0
#: Where the whites slider takes hold, in stops above middle grey, and how
#: quickly it lets go below that. Both are further out and sharper than the
#: highlights slider of step 7, which is what keeps the two controls apart:
#: at these values the midtones move by three hundredths of a stop while the
#: top of the range moves a whole one, where highlights reaches well down
#: into the midtones. Judged per pixel rather than by its surroundings,
#: because a white point belongs to the photo and not to a region of it.
WHITE_PIVOT_EV: Final[float] = 2.0
WHITE_FALLOFF_PER_EV: Final[float] = 2.5

# -- vignette (PLAN.md section 6 step 9.6) --------------------------------
#: How far the corners move at the ends of the vignette slider, in stops.
#: Two is enough for the darkest corner to read as a deliberate frame rather
#: than as a lens fault, and the same the other way for a bright one.
VIGNETTE_MAX_EV: Final[float] = 2.0
#: How far in the midpoint may bring the falloff. Short of the corner itself,
#: because a transition has to have somewhere to happen.
VIGNETTE_MIDPOINT_LIMIT: Final[float] = 0.99

# -- film grain (PLAN.md section 6 step 11.5) -----------------------------
#: Measured on 2026-10-30 from a camera's own JPEGs, shot one after another
#: with its grain off, weak and strong (small grains, ISO 2500), by taking the
#: grain-off shot away from the others over ten thousand flat patches. The
#: grain is grey - the three channels move together exactly - and the same
#: size at both strengths; strong is twice weak.
#:
#: How much the grain moves a grey of middle brightness with the slider at
#: 100, as the spread (standard deviation) of an 8 bit sRGB value. The
#: camera's strong grain measured 14.3 in the midtones and its weak one 7.2,
#: so 50 is weak; 14.4 is what measures 14.3 once the rolling off near the
#: ends of each channel has taken its share (``tools/measure_grain.py``).
GRAIN_STRENGTH_LEVELS: Final[float] = 14.4
#: How the grain's strength follows the brightness of what it lies on, at
#: these 8 bit sRGB levels, as a share of its strength in the midtones. The
#: camera's grain is even from the shadows to the light tones and fades into
#: pure black and pure white, so that neither is speckled. Set so that the
#: grain made here, measured the same way, is within 5 per cent of the
#: camera's in every band of brightness but the darkest, where at 100 it is
#: a level short.
GRAIN_TONE_LEVELS: Final[tuple[float, ...]] = (0.0, 4.0, 12.0, 24.0, 40.0, 216.0, 247.0, 255.0)
GRAIN_TONE_WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.38, 0.86, 0.97, 1.0, 1.0, 0.57, 0.0)
#: The shape of one grain, as the weight a pixel's noise gives a neighbour
#: this many squared pixels away (0 is the pixel itself, 1 the four beside
#: it, 2 the four corners and so on, out to two pixels). Worked out from the
#: camera's grain so that neighbouring pixels keep the correlation it has:
#: 0.22 beside each other, -0.10 two apart and none from three on, which
#: is a grain some four pixels from one to the next at full size.
GRAIN_KERNEL: Final[tuple[tuple[int, float], ...]] = (
    (0, 0.9588),
    (1, 0.1193),
    (2, -0.0192),
    (4, -0.0565),
    (5, -0.0342),
    (8, -0.006),
)
#: How much of the grain is left in a copy shrunk to this share of the full
#: size, as the photo is shrunk for an export of a smaller size (OpenCV's
#: area average). A smaller copy - the proxy, the preview at the size of the
#: screen - is given that much grain directly rather than making the full
#: size grain and shrinking it, which would cost the full size every time.
#: Simulated, and checked against the camera's own JPEGs shrunk the same
#: way: at 0.41 of the size 0.41 of the grain was left (the camera's 0.43),
#: at 0.33 0.33 and at 0.17 0.16. What is left is fine enough to be treated
#: as one pixel wide.
GRAIN_SHRUNK_SCALES: Final[tuple[float, ...]] = (0.0, 0.5, 0.65, 0.8, 0.99)
GRAIN_SHRUNK_STRENGTHS: Final[tuple[float, ...]] = (0.0, 0.49, 0.596, 0.667, 0.731)
#: From this share of the full size up a photo is its full size, and gets
#: its grain pixel for pixel.
GRAIN_FULL_SIZE_FROM: Final[float] = 0.995

# -- vibrance and saturation (PLAN.md section 6 step 10) ------------------
#: The chroma at which vibrance has stopped doing anything, so that colours
#: which are already strong are left alone.
VIBRANCE_FULL_CHROMA: Final[float] = 0.20
#: Skin sits in this band of OkLCh hue, in degrees, and vibrance is held back
#: there so that faces do not turn orange.
SKIN_HUE_FROM_DEGREES: Final[float] = 35.0
SKIN_HUE_TO_DEGREES: Final[float] = 75.0
SKIN_VIBRANCE_WEIGHT: Final[float] = 0.3

# -- detail: noise reduction and sharpening (PLAN.md section 6 step 2.5) --
#: Every size in this block is in pixels of the full size photo. A photo that
#: is a smaller copy - the proxy, the preview at screen size - has the same
#: sizes scaled down with it, so that what it shows is what the full size
#: photo would look like at that size; a size that falls below
#: DETAIL_FINEST_SIGMA there is finer than a pixel and is left out.
#:
#: The step works on the square root of linear light. Shot noise, which is
#: most of the noise in a photo, grows with the square root of the light, so
#: there it is about as strong in the shadows as in the highlights and one
#: threshold serves the whole range. A RAW photo is first lifted by
#: RAW_BASE_EXPOSURE_EV, so that middle grey sits where it does in a JPEG and
#: the thresholds mean the same for both.
DETAIL_FINEST_SIGMA: Final[float] = 0.5
#: The layers the brightness is split into for noise reduction, by the
#: radius of the blur that separates each from the next.
LUMINANCE_NOISE_SIGMAS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
#: What a layer of brightness has to stand out by to be kept, with the
#: luminance slider at 100, layer by layer. The thresholds are fixed rather
#: than worked out from each photo, so that a window of a zoomed in photo is
#: treated exactly as the whole of it is and a slider means the same on
#: every photo; a noisier photo simply wants it further up. They were set on
#: 2026-10-29 from the noise measured in the flat parts of ARW photos from
#: the main body: at ISO 6400 the four layers held 0.0119, 0.0041, 0.0027 and
#: 0.0014, at ISO 3200 about 0.010, 0.0033, 0.0020 and 0.0010. At 100 the
#: slider lets go of three times the noise of ISO 6400, so 50 cleans ISO
#: 3200 well and 25 takes the edge off it.
LUMINANCE_NOISE_THRESHOLDS: Final[tuple[float, ...]] = (0.036, 0.012, 0.008, 0.004)
#: Colour noise comes in blotches far wider than its grain, so its layers
#: reach much further out.
COLOUR_NOISE_SIGMAS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
#: Measured the same way: the colour layers of ISO 6400 held 0.0065, 0.0045,
#: 0.0037, 0.0020, 0.0009 and 0.0003. At 100 the slider lets go of seven
#: times that. Looked at on a night sky at ISO 6400, four times left
#: coloured blotches at a RAW photo's resting 25 that only went at 50; with
#: seven, 25 is where they go, and 100 takes the colour out of everything
#: smaller than a few pixels, stars included.
COLOUR_NOISE_THRESHOLDS: Final[tuple[float, ...]] = (
    0.047,
    0.032,
    0.027,
    0.014,
    0.0065,
    0.0022,
)
#: From which layer up the colour is guided by the brightness, in pixels of
#: the full size photo. The finest layer of the brightness is mostly its own
#: grain, which would shield the colour's grain from being taken out; from
#: here up it is mostly edges, and a thin painted outline, whose colour the
#: wider layers would otherwise wash away, has one to keep it.
COLOUR_GUIDE_FROM_SIGMA: Final[float] = 2.0
#: Up to which radius, in pixels of the photo at hand, a layer is blurred
#: directly rather than on a smaller copy (see ``_detail_blur``).
DETAIL_DIRECT_BLUR_SIGMA: Final[float] = 8.0
#: Below this share of the full size, colour noise is worked out on a copy
#: of half the size (see ``_reduce_colour_noise``).
COLOUR_HALF_SIZE_BELOW: Final[float] = 0.5
#: The radius of the sharpening, as the sigma of the blur it is taken from.
SHARPEN_SIGMA: Final[float] = 0.8
#: How much of the detail is added back with the slider at 100. Set on
#: 2026-10-29 against two ISO 100 ARW photos exported by a reference editor
#: at its own resting sharpening of 40: with this, our resting 40 has 7 per
#: cent more fine detail than the reference on one photo and 9 per cent less on the
#: other (the share of the brightness finer than one pixel, against that
#: between one and four), where a gain of 1.0 had 10 and 21 per cent less.
SHARPEN_GAIN: Final[float] = 2.5
#: The most an edge may be pushed either way, in the square root domain.
#: Without a limit a hard edge gets a bright and a dark line beside it, which
#: is what a halo is; the change is rolled off towards this smoothly.
SHARPEN_HALO_LIMIT: Final[float] = 0.04
#: How strong an edge has to be before the sharpening reaches it, with the
#: masking slider at 100. Below it, smooth skin and sky are left alone.
SHARPEN_MASK_EDGE: Final[float] = 0.02
#: How far around a pixel the masking looks to decide whether it is an edge,
#: as a multiple of the sharpening radius.
SHARPEN_MASK_REACH: Final[float] = 2.0
#: The radius of the softening that sharpening below zero gives, as the
#: sigma of the blur the brightness is taken towards: at -100 it is that
#: blur. Wide enough to take the crispness off at 100 per cent, narrow
#: enough that the photo is softened rather than out of focus.
SOFTEN_SIGMA: Final[float] = 1.5

# -- skin: smoothing and tone (PLAN.md section 6 steps 2.6 and 10) --------
#: Every size here is a multiple of the face it is applied to - S, the
#: distance from between the eyes to the middle of the mouth (core.faces) -
#: so a face filling the frame and a face in a group are treated alike, and
#: the proxy, the preview and the export alike.
#:
#: The skin is split into layers of detail the way the detail step splits a
#: photo (step 2.5): each layer is what lies between one of these sizes of
#: blur and the next. Finer than the first is the texture of the skin
#: itself, its pores, which is never touched; that is what keeps the skin
#: from looking like plastic. The layers above hold blemishes, spots and
#: fine lines, then the uneven blotches of colour and shade across a cheek;
#: the widest is the shape of the face, and is only touched lightly so that
#: it does not go flat.
SKIN_TEXTURE_SIZE: Final[float] = 0.008
SKIN_LAYER_SIZES: Final[tuple[float, ...]] = (0.016, 0.032, 0.064, 0.128)
#: How much of each layer the slider at 100 takes away, where it is weak.
SKIN_LAYER_DEPTHS: Final[tuple[float, ...]] = (0.6, 0.85, 0.7, 0.3)
#: How strong detail in a layer has to be to be left alone, in the square
#: root of linear light, as the detail step measures. A spot or a line is
#: faint; the rim of a nostril, an eyelid's crease or a strand of hair across
#: the cheek stands out far more and is kept, as an edge is kept by the
#: noise reduction.
SKIN_LAYER_THRESHOLDS: Final[tuple[float, ...]] = (0.05, 0.06, 0.05, 0.04)
#: How much the colour of a layer counts against its brightness when it is
#: judged: the redness of a spot is fainter than its shade, and should still
#: count.
SKIN_COLOUR_WEIGHT: Final[float] = 2.0
#: Before the layers, distinct spots - a blemish, a red mark, a mole - are
#: found and filled in from the skin around them: they stand out too much to
#: be faint detail, and the layers would keep them as they keep an edge. A
#: spot is a low point of SKIN_SPOT_SIGNAL (darker and warmer than around
#: it) in every direction, so a line or the edge of a shadow is not one. It
#: is looked for with blurs of these sizes, in S, and must stand out by
#: SKIN_SPOT_THRESHOLD: the curvature of the signal across its narrower way,
#: times the blur's size squared, which measures a spot of any size alike.
SKIN_SPOT_SIZES: Final[tuple[float, ...]] = (0.01, 0.016, 0.024)
#: The signal, from brightness, warmth and green of the square root of
#: linear light (color_management.OPPONENT).
SKIN_SPOT_SIGNAL: Final[tuple[float, float, float]] = (1.0, -1.0, 0.0)
SKIN_SPOT_THRESHOLD: Final[float] = 0.015
#: A spot must also stand out this many times above how much the signal
#: rises and falls around it, within SKIN_SPOT_NEIGHBOURHOOD of S: a blemish
#: on smooth skin does, a dot of stubble in a beard, a crease among wrinkles
#: or a curl at the hairline does not.
SKIN_SPOT_ISOLATION: Final[float] = 3.0
SKIN_SPOT_NEIGHBOURHOOD: Final[float] = 0.08
#: Spots are only filled where the mask is at least this sure of the skin,
#: away from the soft edges round the eyes, the lips and the hair.
SKIN_SPOT_INSIDE: Final[float] = 0.9
#: The nostrils, which are dark low points too, are left out: an ellipse
#: this far below the tip of the nose, this wide and this high, in S.
SKIN_SPOT_NOSTRILS: Final[tuple[float, float, float]] = (0.06, 0.3, 0.16)
#: Anything larger across than this, in S, is not a spot but a feature - a
#: nostril, the corner of an eye - and is left to the layers.
SKIN_SPOT_LARGEST: Final[float] = 0.08
#: How far around what was found is filled in, and how far around it the
#: skin it is filled from is taken, both in S.
SKIN_SPOT_GROW: Final[float] = 0.012
SKIN_SPOT_FILL: Final[float] = 0.03
#: With the slider at 100: how much lighter the skin is made, as a share of
#: its OkLab lightness; how far its hue is turned from yellow towards red, in
#: degrees; and how much of its chroma is kept. Brighter, a little pinker
#: and a little less yellow - a nudge, measured from the person's own skin
#: rather than towards a set colour, so that every skin tone keeps its own.
SKIN_LIGHTNESS_GAIN: Final[float] = 0.03
SKIN_HUE_TURN_DEGREES: Final[float] = -3.0
SKIN_CHROMA_KEPT: Final[float] = 1.0

# -- whitening (PLAN.md 2.2 A-30, section 6 step 10) -------------------------

#: With the whitening slider at 100, how much lighter the skin is made, as a
#: share of its OkLab lightness, where the lift is greatest. The lift is
#: shaped by L squared times (1 - L), which is greatest at two thirds: lit
#: skin is lifted most, the shadows that model a face far less, so it keeps
#: its shape, and highlights rise little, so they do not burn out.
WHITEN_LIGHTNESS_GAIN: Final[float] = 0.1
#: How much of the yellow (OkLab b, where it is positive) is taken out at 100.
#: That turns the hue towards pink and lowers the chroma a little at once,
#: which is the clear look whitening is after, without turning a skin grey:
#: its red is left as it is.
WHITEN_YELLOW_TAKEN: Final[float] = 0.15

# -- automatic light (PLAN.md section 6, "自動調整光線") -----------------------
#: The long edge the photo is measured at. Each try is one develop of a copy
#: this size, about ten milliseconds, and a search takes some thirty tries.
AUTO_LONG_EDGE: Final[int] = 256
#: How many halvings each slider's search makes: seven leave the exposure
#: within a twentieth of a stop and the other sliders within two steps.
AUTO_SEARCH_STEPS: Final[int] = 7
#: The median brightness the exposure aims for, as an encoded sRGB value:
#: middle grey (18% linear) is 0.46.
AUTO_MIDTONE_TARGET: Final[float] = 0.46
#: How much of the way to that target the exposure goes, and how far it may
#: go at most. Not all of it: a night is meant to be dark and snow to be
#: bright, and taking every photo to middle grey would take that away.
AUTO_EXPOSURE_SHARE: Final[float] = 0.67
AUTO_EXPOSURE_LIMIT_EV: Final[float] = 2.0
#: Where the brightest few pixels are taken: the brightest channel of the
#: pixel at this percentile, as an encoded value, just short of white so
#: that nothing new burns out. The exposure never rises past the point where
#: they reach it - a photo whose bright parts already touch white is not
#: dark, however much black is around them - and the whites slider only
#: ever lifts them to it: a photo that already reaches white, a sun in a
#: sunset say, is left reaching it rather than turned grey.
AUTO_WHITE_PERCENTILE: Final[float] = 99.5
AUTO_WHITE_TARGET: Final[float] = 0.96
AUTO_WHITES_LIMIT: Final[float] = 60.0
#: Where the darkest few are taken by the blacks slider, by their darkest
#: channel: close to black, which is what takes the haze off a flat photo.
#: The slider only ever lowers them: a photo that already holds black keeps
#: it rather than having it lifted to grey.
AUTO_BLACK_PERCENTILE: Final[float] = 0.5
AUTO_BLACK_TARGET: Final[float] = 0.03
AUTO_BLACKS_LIMIT: Final[float] = 60.0
#: A pixel whose brightest channel is at least this is taken as burnt out;
#: when more of the photo than the share below is, the highlights slider
#: brings it down, by no more than the limit. The share is large enough that
#: a light source or a glint, which ought to stay white, does not count.
AUTO_CLIPPED_LEVEL: Final[float] = 0.99
AUTO_CLIPPED_SHARE: Final[float] = 0.02
AUTO_HIGHLIGHTS_LIMIT: Final[float] = 40.0
#: When the darkest tenth of the photo is darker than the target, the
#: shadows slider lifts it, by no more than the limit: enough to open dense
#: shadows a little, not so much that a night or a silhouette turns grey.
AUTO_SHADOW_PERCENTILE: Final[float] = 10.0
AUTO_SHADOW_TARGET: Final[float] = 0.06
AUTO_SHADOWS_LIMIT: Final[float] = 25.0

_EPSILON: Final[float] = 1e-6
_EIGHT_BIT_MAX: Final[float] = 255.0
_SIXTEEN_BIT_MAX: Final[float] = 65535.0
#: What an 8 bit sample is multiplied by to fill the 16 bit range, so that
#: 255 becomes 65535 rather than 65280.
_EIGHT_TO_SIXTEEN: Final[int] = 257

#: The depths an exported file can be written at (PLAN.md 6 step 12).
EIGHT_BIT: Final[int] = 8
SIXTEEN_BIT: Final[int] = 16

#: How large one band of the photo is, in bytes of the photo it is cut from.
#: The photo is far too large to stay in cache, so each step used to read it
#: from memory and write it back; a band that fits stays in cache from one
#: step to the next, which is worth more than the arithmetic itself.
#:
#: What has to fit is a size, not a number of rows, and a row is 7 KB of a
#: 900 px draft and 47 KB of a full size photo. Measured on four sizes of the
#: same photo, the fastest band was 520 to 760 KB at every one of them, so
#: the sixteen rows this used to be was only right at the one width it was
#: measured at. Three quarters of a megabyte is the top of that range, which
#: leaves full size on the sixteen rows it was measured at and gives every
#: smaller size the band it wanted: with the sliders moved, a 2560 px proxy
#: went from 530 to 447 ms, a zoomed in window from 217 to 119, and the draft
#: of a drag from 115 to 45. The result is the same whatever the size, so
#: this is free to tune; ``tests/test_reference.py`` holds it to that.
BAND_TARGET_BYTES: Final[int] = 768 * 1024

#: The fewest and the most rows a band may be, however wide the photo. Too
#: few and the work in a band is lost in the handing out of it; too many and
#: the photo is barely cut up at all.
BAND_ROWS_LEAST: Final[int] = 4
BAND_ROWS_MOST: Final[int] = 256


def band_rows(row_bytes: int) -> int:
    """Return how many rows make up one band of a photo with rows this large."""
    rows = BAND_TARGET_BYTES // max(row_bytes, 1)
    return max(BAND_ROWS_LEAST, min(BAND_ROWS_MOST, rows))

#: How many bands are worked on at once. The steps are per pixel, so bands
#: know nothing of each other.
#:
#: One thread per physical core rather than per logical one. The arithmetic
#: per pixel is cheap and the photo is large, so what the threads are really
#: competing for is memory bandwidth, and the second thread of a core adds
#: nothing to that but contention. Measured on this machine - six cores,
#: twelve threads - with the sliders moved: a 2560 px proxy took 447 ms on
#: twelve threads and 363 on six, a zoomed in window 119 and 121, full size
#: 2434 and 2156. ``os.cpu_count()`` counts logical cores and the standard
#: library cannot say how many are physical, so halving it is the closest
#: that can be had without another package; a machine without hyperthreading
#: is left with half of its cores, which still has most of the bandwidth.
_WORKERS: Final[int] = max(2, min(12, (os.cpu_count() or 4) // 2))

_pool: ThreadPoolExecutor | None = None


def _band_pool() -> ThreadPoolExecutor:
    """Return the shared pool, making it the first time it is wanted."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=_WORKERS, thread_name_prefix="tintaphoto-band"
        )
    return _pool


def in_bands(
    work: Callable[..., NDArray[Any]], *planes: NDArray[Any] | None
) -> NDArray[Any]:
    """Run *work* on matching bands of *planes* and join the results.

    Every plane is cut at the same rows, so a band of the photo is handed to
    *work* together with the band of each plane that goes with it. A plane
    that is ``None`` is passed through as ``None``, which is how the steps
    that only apply to RAW are skipped.
    """
    first = planes[0]
    assert first is not None
    height = first.shape[0]
    # The photo is always the first plane, so how large one of its rows is
    # says how many rows a band of the wanted size holds.
    rows = band_rows(first.itemsize * int(np.prod(first.shape[1:], dtype=np.int64)))
    if height <= rows:
        return work(*planes)

    edges = list(range(0, height, rows))
    pieces: list[tuple[NDArray[Any] | None, ...]] = []
    for index, start in enumerate(edges):
        stop = edges[index + 1] if index + 1 < len(edges) else height
        pieces.append(
            tuple(
                None if plane is None else np.ascontiguousarray(plane[start:stop])
                for plane in planes
            )
        )
    done: Sequence[NDArray[Any]] = list(
        _band_pool().map(lambda parts: work(*parts), pieces)
    )
    return np.concatenate(done, axis=0)

#: A blur this wide carries no detail finer than itself, so it is computed on
#: a smaller copy and stretched back. Above this many pixels of radius the
#: saving is worth the shrinking, and the result is the same to the eye.
_BLUR_SHRINK_ABOVE_SIGMA: Final[float] = 2.0
#: How many pixels of the smaller copy one radius should still cover, so that
#: the blur stays smooth rather than blocky.
_BLUR_SIGMA_AFTER_SHRINK: Final[float] = 2.0


def _wide_blur(values: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Blur *values* by *sigma* pixels, working on a smaller copy when wide.

    A Gaussian of any width throws away everything finer than itself, so for
    the wide blurs this pipeline uses - a fiftieth of the frame - shrinking
    first costs nothing visible and saves most of the work.
    """
    if sigma <= _BLUR_SHRINK_ABOVE_SIGMA:
        return cv2.GaussianBlur(values, (0, 0), sigma)

    height, width = values.shape[:2]
    factor = min(sigma / _BLUR_SIGMA_AFTER_SHRINK, min(height, width) / 8.0)
    if factor <= 1.0:
        return cv2.GaussianBlur(values, (0, 0), sigma)

    small = cv2.resize(
        values,
        (max(2, round(width / factor)), max(2, round(height / factor))),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def _tone_curve_parameters() -> tuple[float, float]:
    """Return the exponent and the half point of the base tone curve.

    Derived from the three constants above: the exponent sets the slope at
    middle grey, and the half point places middle grey on itself.
    """
    exponent = TONE_MID_SLOPE / (1.0 - TONE_MID_OUTPUT)
    half_point = (
        TONE_MID_INPUT**exponent * (1.0 - TONE_MID_OUTPUT) / TONE_MID_OUTPUT
    ) ** (1.0 / exponent)
    return exponent, half_point


TONE_EXPONENT, TONE_HALF_POINT = _tone_curve_parameters()


# --------------------------------------------------------------------------
# Step 2.5: detail - noise reduction and sharpening
# --------------------------------------------------------------------------
#
# The one step that is about the finest detail of the photo rather than about
# its light or its colour. It comes straight after the geometry, before
# anything has lifted the shadows, so that noise is taken out before the
# tone steps make it more visible; and it works on the photo as it was
# decoded, which a RAW has in its camera's colour, already balanced the way
# the camera was set, so that a grey is a grey here as well.


def has_detail(state: EditState) -> bool:
    """Return whether any of the four detail sliders asks for anything."""
    return (
        state.sharpening != 0.0
        or state.luminance_noise != 0.0
        or state.colour_noise != 0.0
    )


def detail_key(state: EditState) -> tuple[float, float, float, float]:
    """Return everything the detail step depends on in *state*."""
    return (
        state.sharpening,
        state.sharpen_masking,
        state.luminance_noise,
        state.colour_noise,
    )


def with_detail(photo: DecodedPhoto, state: EditState) -> DecodedPhoto:
    """Return *photo* after step 2.5, marked as having been through it.

    The step depends on nothing but the four detail sliders and the photo,
    so its result can be kept while every other slider moves; :func:`develop`
    sees the mark and goes on from step 3. A photo already through the step
    is handed back as it is.
    """
    if photo.detail is not None or not has_detail(state):
        return photo
    return replace(
        photo,
        image=apply_detail(photo.image, state, detail_scale(photo), photo.is_raw),
        detail=detail_key(state),
    )


def detail_scale(photo: DecodedPhoto) -> float:
    """Return how many pixels of *photo* one pixel of the full size photo is.

    One for the photo itself, less for the proxy and for the preview at the
    size of the screen; the detail step's sizes are scaled by it, so that a
    smaller copy shows what the full size photo would look like at its size.
    """
    return min(1.0, max(photo.size) / max(max(photo.full_size), 1))


def _detail_levels(
    sigmas: tuple[float, ...], thresholds: tuple[float, ...], scale: float, amount: float
) -> list[tuple[float, float, float]]:
    """Return the layers to split into at *scale*, with their thresholds.

    Each comes as its blur at the size of the photo at hand, its threshold
    and its blur in pixels of the full size photo. A layer finer than a
    pixel of the photo at hand has already been averaged away by shrinking
    it, and is left out.
    """
    strength = amount / 100.0
    return [
        (sigma * scale, threshold * strength, sigma)
        for sigma, threshold in zip(sigmas, thresholds, strict=True)
        if sigma * scale >= DETAIL_FINEST_SIGMA
    ]


def detail_reach(state: EditState, scale: float = 1.0) -> int:
    """Return how many pixels around a pixel the detail step looks at.

    What a window of the photo has to be grown by before it is developed on
    its own (PLAN.md 3.3), at *scale* pixels per pixel of the full size photo.
    """
    sigma = 0.0
    if state.luminance_noise != 0.0:
        sigma = max(sigma, max(LUMINANCE_NOISE_SIGMAS) * scale)
    if state.colour_noise != 0.0:
        sigma = max(sigma, max(COLOUR_NOISE_SIGMAS) * scale)
    if state.sharpening > 0.0:
        sigma = max(sigma, SHARPEN_SIGMA * SHARPEN_MASK_REACH * 2.0 * scale)
    elif state.sharpening < 0.0:
        sigma = max(sigma, SOFTEN_SIGMA * scale)
    if sigma <= 0.0:
        return 0
    return int(np.ceil(BLUR_REACH_SIGMAS * max(1.0, sigma)))


def _by_rows(work: Callable[[slice], None], plane: NDArray[Any]) -> None:
    """Run *work* over bands of rows of *plane*'s height, on the band pool.

    The detail step is made of blurs, which need the whole plane, and of
    arithmetic per pixel in between, which does not. The arithmetic is cheap
    and the planes are large, so done a plane at a time it is held up by
    memory rather than by the processor; done a band at a time on every core,
    the band stays in cache from one operation to the next. *work* writes its
    rows into arrays made beforehand.
    """
    height = plane.shape[0]
    # A band of one plane is a third of a band of the photo it came from.
    row_bytes = plane.itemsize * int(np.prod(plane.shape[1:], dtype=np.int64))
    rows = band_rows(row_bytes * 3)
    if height <= rows:
        work(slice(0, height))
        return
    starts = range(0, height, rows)
    list(_band_pool().map(lambda start: work(slice(start, min(start + rows, height))), starts))


def _detail_blur(values: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Blur one plane of the detail step by *sigma* pixels.

    Up to DETAIL_DIRECT_BLUR_SIGMA the blur is worked out on the plane itself
    rather than on a smaller copy. The detail step lets go of what is under a
    threshold and keeps what is over it, which magnifies any difference in
    the blur, and a smaller copy is cut at different places for a window of
    the photo than for the whole of it; worked out directly, a window of the
    zoomed in preview comes out as the whole photo does (PLAN.md 3.3). The
    widest layers are only of colour, which is smooth that far out, and are
    worth the shortcut.
    """
    if sigma <= DETAIL_DIRECT_BLUR_SIGMA:
        blurred = cv2.GaussianBlur(values, (0, 0), sigma)
    else:
        blurred = _wide_blur(values, sigma)
    # OpenCV drops a trailing axis of one, which a single plane has.
    return blurred.reshape(values.shape)


def shrink_noise(
    stack: NDArray[np.float32],
    levels: list[tuple[float, float, float]],
    guide: bool = False,
) -> NDArray[np.float32]:
    """Take out of *stack* what is too faint to be anything but noise.

    *stack* holds one or more planes as channels. Each is split into layers
    of detail - what each blur in *levels* takes away from the one before -
    and a last, smooth remainder. In every layer a value that stands out from
    its surroundings by much more than its threshold is kept as it is, and
    one that stands out by less is let go, smoothly in between; the layers
    are then added back up. Nothing is lost when every threshold is zero,
    because the layers add up to the photo exactly.

    The planes are judged together, by how far a pixel stands out in all of
    them at once: two planes of colour are shrunk towards grey without their
    hue being turned. The planes are blurred together too, in one pass.

    With *guide* the first plane is a guide: judged with the others in the
    layers from COLOUR_GUIDE_FROM_SIGMA up, but left as it is. The colour is
    guided by the brightness: a thin coloured line along an edge stands out
    in the brightness as well and is kept, where a blotch of colour noise has
    nothing in the brightness to go with it.
    """
    if not levels:
        return stack
    first = 1 if guide else 0
    previous = stack
    kept = np.zeros_like(stack)
    for sigma, threshold, full_sigma in levels:
        blurred = _detail_blur(stack, sigma)
        guided = guide and full_sigma >= COLOUR_GUIDE_FROM_SIGMA
        # Kept above zero so that an empty layer comes out as nothing, not as
        # nothing divided by nothing.
        floor = np.float32(max(threshold**4, 1e-30))

        def shrink(
            rows: slice,
            previous: NDArray[np.float32] = previous,
            blurred: NDArray[np.float32] = blurred,
            guided: bool = guided,
            floor: np.float32 = floor,
        ) -> None:
            """One layer of one band."""
            layer = previous[rows] - blurred[rows]
            squares = layer * layer
            strength = squares[..., first].copy()
            for channel in range(first + 1, squares.shape[-1]):
                strength += squares[..., channel]
            if guided:
                strength += squares[..., 0]
            # A smooth version of a hard threshold: a value twice above it
            # keeps 94 per cent of itself, one at half of it 6 per cent, and
            # nothing jumps. The fourth power rather than the square is what
            # keeps real detail a little above the threshold nearly whole.
            strength *= strength
            factor = strength / (strength + floor)
            if first:
                layer[..., first:] *= factor[..., None]
            else:
                layer *= factor[..., None]
            kept[rows] += layer

        _by_rows(shrink, stack[..., 0])
        previous = blurred
    kept += previous
    return kept


#: Below this radius, in pixels of the photo at hand, the sharpening blur
#: reaches its neighbours by less than four ten thousandths of the pixel
#: itself, and the sharpening changes nothing that could be seen: the screen
#: sized preview of a large photo, which is a fifth of its size. It is not
#: worked out there at all.
SHARPEN_LEAST_SIGMA: Final[float] = 0.25


def sharpen(
    brightness: NDArray[np.float32], amount: float, masking: float, scale: float
) -> NDArray[np.float32]:
    """Sharpen *brightness*, an unsharp mask with its halos held back.

    What a blur of the sharpening radius takes away is the finest detail,
    and adding more of it back is what sharpening is. The change is rolled
    off towards SHARPEN_HALO_LIMIT, so that a hard edge is crisper without a
    glowing line beside it; with *masking* raised only edges are sharpened,
    so that smooth skin and sky keep their grain to themselves. Below zero
    the photo is softened instead (:func:`soften`), and *masking* is not used.
    """
    if amount < 0.0:
        return soften(brightness, -amount, scale)
    sigma = SHARPEN_SIGMA * scale
    if amount == 0.0 or sigma < SHARPEN_LEAST_SIGMA:
        return brightness
    soft = cv2.GaussianBlur(brightness, (0, 0), sigma)
    edge: NDArray[np.float32] | None = None
    if masking > 0.0:
        reach = sigma * SHARPEN_MASK_REACH
        edge = cv2.GaussianBlur(
            cv2.absdiff(brightness, cv2.GaussianBlur(brightness, (0, 0), reach)),
            (0, 0),
            reach,
        )
    limit = np.float32(SHARPEN_HALO_LIMIT)
    push = np.float32(SHARPEN_GAIN * amount / 100.0 / SHARPEN_HALO_LIMIT)
    threshold = np.float32(SHARPEN_MASK_EDGE * max(masking, _EPSILON) / 100.0)
    result = np.empty_like(brightness)

    def push_edges(rows: slice) -> None:
        """Sharpen one band."""
        pushed = (brightness[rows] - soft[rows]) * push
        # limit * tanh(pushed), written as pushed / sqrt(1 + pushed^2), which
        # rolls off the same way and is several times cheaper.
        change = pushed / np.sqrt(pushed * pushed + np.float32(1.0)) * limit
        if edge is not None:
            # Nothing below half of the threshold, everything above one and
            # a half of it, and a smooth step in between.
            position = np.clip(
                (edge[rows] - threshold * np.float32(0.5)) / threshold,
                np.float32(0.0),
                np.float32(1.0),
            )
            change *= position * position * (np.float32(3.0) - np.float32(2.0) * position)
        result[rows] = brightness[rows] + change

    _by_rows(push_edges, brightness)
    return result


def soften(
    brightness: NDArray[np.float32], amount: float, scale: float
) -> NDArray[np.float32]:
    """Soften *brightness* by *amount* per cent, the other way from sharpening.

    The brightness is taken *amount* per cent of the way towards a blur of
    SOFTEN_SIGMA pixels of the full size photo. A blur adds no detail, so
    there is no halo to hold back.
    """
    sigma = SOFTEN_SIGMA * scale
    if amount <= 0.0 or sigma < SHARPEN_LEAST_SIGMA:
        return brightness
    soft = cv2.GaussianBlur(brightness, (0, 0), sigma)
    share = np.float32(min(amount, 100.0) / 100.0)
    result = np.empty_like(brightness)

    def blend(rows: slice) -> None:
        """Soften one band."""
        result[rows] = brightness[rows] + (soft[rows] - brightness[rows]) * share

    _by_rows(blend, brightness)
    return result


def _reduce_colour_noise(
    split: NDArray[np.float32], amount: float, scale: float
) -> NDArray[np.float32]:
    """Take the colour noise out of *split*, brightness first then colour.

    Guided by the brightness as it came, before its own noise is reduced:
    what is an edge is decided by the photo, not by how much of the
    brightness's grain the other slider has taken out.

    On a photo below COLOUR_HALF_SIZE_BELOW of its full size the work is
    done on a copy of half that size, and only the change is stretched back:
    every colour layer left at that size is a blur several of its pixels
    wide already, so the change is smooth, and a quarter of the pixels is
    what keeps a RAW photo, which always rests with some of this, quick to
    open and to crop. The full size photo and the zoomed in preview are
    worked out at their own size.
    """
    height, width = split.shape[:2]
    if scale >= COLOUR_HALF_SIZE_BELOW or min(height, width) < 8:
        return shrink_noise(
            split,
            _detail_levels(COLOUR_NOISE_SIGMAS, COLOUR_NOISE_THRESHOLDS, scale, amount),
            guide=True,
        )
    half = (max(1, round(width / 2)), max(1, round(height / 2)))
    small = cv2.resize(split, half, interpolation=cv2.INTER_AREA)
    factor = half[0] / width
    shrunk = shrink_noise(
        small,
        _detail_levels(COLOUR_NOISE_SIGMAS, COLOUR_NOISE_THRESHOLDS, scale * factor, amount),
        guide=True,
    )
    change = cv2.resize(shrunk - small, (width, height), interpolation=cv2.INTER_LINEAR)
    return split + change


def apply_detail(
    image: NDArray[np.float32], state: EditState, scale: float, is_raw: bool
) -> NDArray[np.float32]:
    """Reduce noise and sharpen (PLAN.md section 6 step 2.5).

    *image* is the photo as it was decoded and framed, and *scale* how many
    of its pixels one pixel of the full size photo is (:func:`detail_scale`).
    The light is taken to its square root and split into brightness and
    colour; the brightness has its noise reduced and is sharpened, the colour
    has its own noise reduced, and the two are put back together.
    """
    if not has_detail(state):
        return image
    lift = np.float32(2.0**RAW_BASE_EXPOSURE_EV if is_raw else 1.0)
    forward = np.asarray(cm.OPPONENT, dtype=np.float32)
    backward = np.asarray(cm.OPPONENT_INVERSE, dtype=np.float32)
    height, width = image.shape[:2]
    # Brightness first, then the two planes of colour.
    split = np.empty((height, width, 3), dtype=np.float32)

    def to_split(rows: slice) -> None:
        """Square roots, then brightness and colour, for one band."""
        roots = cv2.sqrt(cv2.max(cv2.multiply(image[rows], (float(lift),) * 3 + (0.0,)), 0.0))
        split[rows] = cv2.transform(roots, forward)

    _by_rows(to_split, split[..., 0])

    brightness = np.ascontiguousarray(split[..., 0])
    if state.luminance_noise != 0.0:
        brightness = shrink_noise(
            brightness[..., None],
            _detail_levels(
                LUMINANCE_NOISE_SIGMAS, LUMINANCE_NOISE_THRESHOLDS, scale, state.luminance_noise
            ),
        )[..., 0]
    colour = split
    if state.colour_noise != 0.0:
        colour = _reduce_colour_noise(split, state.colour_noise, scale)
    brightness = np.ascontiguousarray(
        sharpen(brightness, state.sharpening, state.sharpen_masking, scale)
    )

    result = np.empty((height, width, 3), dtype=np.float32)

    def join(rows: slice) -> None:
        """Back to three channels and out of the square root, for one band."""
        joined = colour[rows].copy()
        joined[..., 0] = brightness[rows]
        roots = cv2.max(cv2.transform(joined, backward), 0.0)
        result[rows] = cv2.multiply(cv2.multiply(roots, roots), (1.0 / float(lift),) * 3 + (0.0,))

    _by_rows(join, result[..., 0])
    return result


# --------------------------------------------------------------------------
# Step 2.6: skin smoothing
# --------------------------------------------------------------------------
#
# Right after the detail step and on the same footing: the photo as it was
# decoded, in the square root of linear light, split into brightness and two
# planes of colour. Only where the skin mask says so, and only the faint
# detail of the middle sizes: what is taken out is what a spot, a pore that
# has gone red, a fine line or a blotch is made of, and what is left is the
# texture of the skin and every edge that matters. Distinct spots, too
# strong to count as faint, are found first and filled in from the skin
# around them (skin_spots, fill_spots).


def has_skin(state: EditState, photo: DecodedPhoto) -> bool:
    """Return whether the skin slider asks for anything and there is a face for it.

    Smoothing is worked out face by face, so it needs a face even where the
    brush has added skin (PLAN.md 2.10).
    """
    return (
        state.skin != 0.0
        and bool(photo.faces)
        and not skin_range.reaches_nothing(photo.faces, photo.skin)
    )


def has_whiten(state: EditState, photo: DecodedPhoto) -> bool:
    """Return whether the whitening slider asks for anything and there is skin for it.

    Skin found beside a face, or skin the brush added where no face was
    found (PLAN.md 2.10).
    """
    return state.whiten != 0.0 and not skin_range.reaches_nothing(photo.faces, photo.skin)


def needs_body(state: EditState, photo: DecodedPhoto) -> bool:
    """Return whether developing *photo* wants :func:`body_field`.

    Whitening always does; smoothing only when the brush has changed the
    skin it works on.
    """
    if has_whiten(state, photo):
        return True
    return has_skin(state, photo) and photo.skin is not None and not photo.skin.is_default


def body_field(photo: DecodedPhoto) -> NDArray[np.float32] | None:
    """Return the skin of *photo* over its whole frame, as :func:`core.skin_range.field`.

    *photo* is the whole frame, framed and cropped, carrying its faces and
    its skin range. The field is small, and :meth:`Window.take` cuts and
    stretches it for whatever is developed, so a zoomed in window works on
    exactly the skin the whole photo does (PLAN.md 6 step 10, whitening).
    ``None`` when there is no skin to work on.
    """
    if skin_range.reaches_nothing(photo.faces, photo.skin):
        return None
    return skin_range.field(photo.image, photo.faces, photo.skin)


def skin_placements(photo: DecodedPhoto, pane: Window) -> list[faces_module.Placement]:
    """Return where *photo*'s faces lie in it, *photo* being *pane* of its frame."""
    if not photo.faces:
        return []
    return faces_module.place(
        photo.faces, pane.frame, pane.left, pane.top, (pane.size[0], pane.size[1])
    )


def skin_reach(photo: DecodedPhoto, state: EditState, scale: float = 1.0) -> int:
    """Return how far around a pixel the skin steps look, in pixels.

    What a window has to be grown by when the skin slider is moved (PLAN.md
    3.3): the widest layer of the smoothing and the softening of the mask,
    for the largest face, at *scale* pixels per pixel of the full size photo.
    """
    smoothing = has_skin(state, photo)
    if not smoothing and not has_whiten(state, photo):
        return 0
    assert photo.faces is not None
    largest = max(faces_module.face_size(face, photo.full_size) for face in photo.faces)
    # The layers' blurs are taken one after another, which reaches a little
    # further than the widest of them alone. Whitening alone needs only the
    # mask of the face; the skin beyond it comes cut from the whole frame.
    widest = 1.2 * max(SKIN_LAYER_SIZES) if smoothing else 0.0
    sigma = largest * scale * max(
        widest, faces_module.MASK_SOFTEN, faces_module.COLOUR_HOLE_FILL / 2.0
    )
    return int(np.ceil(BLUR_REACH_SIGMAS * max(1.0, sigma)))


def _shaped(
    planes: NDArray[np.float32],
    found: NDArray[np.float32] | None,
    face: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Return the skin to work on: what was found, as the brush left it.

    *planes* is :func:`body_field` cut to the photo at hand, *found* the
    skin found beside the faces at that size and *face* the faces' own mask;
    either may be missing. The strokes act as :func:`core.skin_range.field`
    says: ``max(found × kept, added)``.
    """
    base = planes[..., skin_range.ADDED] * np.float32(0.0)
    if found is not None:
        base = np.maximum(base, found)
    if face is not None:
        base = np.maximum(base, face)
    return np.maximum(
        base * planes[..., skin_range.KEPT], planes[..., skin_range.ADDED]
    ).astype(np.float32)


def smooth_skin(
    image: NDArray[np.float32],
    placements: list[faces_module.Placement],
    mask: NDArray[np.float32],
    amount: float,
    is_raw: bool,
) -> NDArray[np.float32]:
    """Smooth the skin of every face (PLAN.md section 6 step 2.6).

    *mask* is :func:`core.faces.skin_mask` over the whole of *image*; each
    face is worked on inside the part of its box the mask reaches, with the
    layer sizes of its own size. A pixel the mask leaves out comes back
    exactly as it was.

    Only the texture is separated at the photo's own resolution. Everything
    coarser than it is a blur already, so the layers are worked out on a
    copy shrunk until the texture's size is a pixel, and only the change is
    stretched back - the same picture for a small part of the work.
    """
    if amount <= 0.0 or not placements:
        return image
    result = image.copy()
    lift = np.float32(2.0**RAW_BASE_EXPOSURE_EV if is_raw else 1.0)
    forward = np.asarray(cm.OPPONENT, dtype=np.float32)
    backward = np.asarray(cm.OPPONENT_INVERSE, dtype=np.float32)
    strength = np.float32(amount / 100.0)
    weights = np.array([1.0, SKIN_COLOUR_WEIGHT, SKIN_COLOUR_WEIGHT], dtype=np.float32)

    for placement in placements:
        bounds = _skin_bounds(mask, placement)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        weight = mask[top:bottom, left:right] * strength
        part = result[top:bottom, left:right]
        roots = cv2.sqrt(np.maximum(part * lift, np.float32(0.0)))
        split = cv2.transform(roots, forward)

        texture = SKIN_TEXTURE_SIZE * placement.size
        coarse = split if texture < DETAIL_FINEST_SIGMA else _detail_blur(split, texture)
        height, width = split.shape[:2]
        factor = max(1.0, min(texture, min(height, width) / 16.0))
        if factor > 1.5:
            size = (max(2, round(width / factor)), max(2, round(height / factor)))
            previous = cv2.resize(coarse, size, interpolation=cv2.INTER_AREA)
            small_weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        else:
            factor = 1.0
            previous = coarse
            small_weight = weight
        change = np.zeros_like(previous)
        spots = skin_spots(
            previous,
            small_weight / strength,
            placement.size / factor,
            nostrils=_nostrils(placement, left, top, factor),
        )
        if spots is not None:
            healed = fill_spots(previous, spots, small_weight, placement.size / factor)
            change += healed - previous
            previous = healed
        for size_of, depth, threshold in zip(
            SKIN_LAYER_SIZES, SKIN_LAYER_DEPTHS, SKIN_LAYER_THRESHOLDS, strict=True
        ):
            sigma = size_of * placement.size / factor
            if sigma * factor < DETAIL_FINEST_SIGMA:
                continue
            blurred = _detail_blur(previous, sigma)

            def take_faint(
                rows: slice,
                previous: NDArray[np.float32] = previous,
                blurred: NDArray[np.float32] = blurred,
                depth: float = depth,
                floor: np.float32 = np.float32(threshold**4),
            ) -> None:
                """One layer of one band."""
                layer = previous[rows] - blurred[rows]
                judged = cv2.transform(layer * layer, weights[None, :])
                judged = judged.reshape(layer.shape[:2])
                judged *= judged
                # 1 where the layer is faint, 0 where it is an edge, smoothly.
                faint = floor / (judged + floor)
                faint *= small_weight[rows]
                faint *= np.float32(depth)
                change[rows] -= layer * faint[..., None]

            _by_rows(take_faint, previous[..., 0])
            previous = blurred
        if factor > 1.0:
            change = cv2.resize(change, (width, height), interpolation=cv2.INTER_LINEAR)

        def put_back(
            rows: slice,
            split: NDArray[np.float32] = split,
            change: NDArray[np.float32] = change,
            weight: NDArray[np.float32] = weight,
            part: NDArray[np.float32] = part,
        ) -> None:
            """Out of the square root, where the mask reaches, for one band."""
            smoothed = cv2.transform(split[rows] + change[rows], backward)
            smoothed = np.maximum(smoothed, np.float32(0.0))
            smoothed = smoothed * smoothed / lift
            part[rows] = np.where((weight[rows] > 0.0)[..., None], smoothed, part[rows])

        # *part* is a view of *result*, so this writes the smoothed skin in.
        _by_rows(put_back, part[..., 0])
    return result


def _nostrils(
    placement: faces_module.Placement, left: int, top: int, factor: float
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return the nostrils' ellipse, as cv2.ellipse takes it, in a part's pixels.

    The part starts at *left*, *top* of the photo at hand and is shrunk by
    *factor*.
    """
    below, half_across, half_up = SKIN_SPOT_NOSTRILS
    size = placement.size
    centre = placement.points[2] - placement.up * below * size
    angle = float(np.degrees(np.arctan2(placement.across[1], placement.across[0])))
    return (
        ((centre[0] - left) / factor, (centre[1] - top) / factor),
        (2.0 * half_across * size / factor, 2.0 * half_up * size / factor),
        angle,
    )


def skin_spots(
    split: NDArray[np.float32],
    mask: NDArray[np.float32],
    size: float,
    nostrils: tuple[tuple[float, float], tuple[float, float], float] | None = None,
) -> NDArray[np.float32] | None:
    """Return where the spots on the skin are, from 0 to 1, or ``None``.

    *split* is the skin in the square root of linear light, split into
    brightness and colour; *mask* the skin mask over it; *size* the face's
    S in its pixels; *nostrils* an ellipse, as cv2.ellipse takes it, left
    out. A spot is a low point of SKIN_SPOT_SIGNAL in every direction, no
    larger than SKIN_SPOT_LARGEST, standing out from what is around it, well
    inside the mask; it comes back grown by SKIN_SPOT_GROW and softened at
    its edge.
    """
    height, width = split.shape[:2]
    signal = cv2.transform(split, np.array([SKIN_SPOT_SIGNAL], dtype=np.float32))
    signal = signal.reshape(height, width)
    response = np.zeros((height, width), dtype=np.float32)
    for size_of in SKIN_SPOT_SIZES:
        sigma = size_of * size
        if sigma < DETAIL_FINEST_SIGMA:
            continue
        blurred = cv2.GaussianBlur(signal, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        # Second derivatives; OpenCV's 3x3 kernels count each four times.
        across = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        down = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        both = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
        middle = (across + down) * np.float32(0.125)
        apart = (across - down) * np.float32(0.125)
        both *= np.float32(0.25)
        # The lesser curvature: high only where the signal rises every way.
        lesser = middle - cv2.sqrt(apart * apart + both * both)
        np.maximum(response, lesser * np.float32(sigma * sigma), out=response)
    around = max(DETAIL_FINEST_SIGMA, SKIN_SPOT_NEIGHBOURHOOD * size)
    usual = cv2.sqrt(
        cv2.GaussianBlur(response * response, (0, 0), around, borderType=cv2.BORDER_REPLICATE)
    )
    bar = np.maximum(usual * np.float32(SKIN_SPOT_ISOLATION), np.float32(SKIN_SPOT_THRESHOLD))
    found = ((response > bar) & (mask >= SKIN_SPOT_INSIDE)).astype(np.uint8)
    if nostrils is not None:
        cv2.ellipse(found, nostrils, 0, -1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
    if count <= 1:
        return None
    largest = SKIN_SPOT_LARGEST * size
    small = np.maximum(
        stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    ) <= largest
    small[0] = False
    if not small[1:].any():
        return None
    spots = small[labels].astype(np.float32)
    grow = max(1, int(round(SKIN_SPOT_GROW * size)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    spots = cv2.dilate(spots, disc)
    return cv2.GaussianBlur(spots, (0, 0), grow / 2.0, borderType=cv2.BORDER_REPLICATE)


def fill_spots(
    split: NDArray[np.float32],
    spots: NDArray[np.float32],
    weight: NDArray[np.float32],
    size: float,
) -> NDArray[np.float32]:
    """Return *split* with *spots* filled in from the skin around them.

    The skin around is the average of what is not a spot, within
    SKIN_SPOT_FILL of S; the slider and the mask, in *weight*, decide how
    much of a spot is filled.
    """
    sigma = max(DETAIL_FINEST_SIGMA, SKIN_SPOT_FILL * size)
    keep = np.float32(1.0) - spots
    around = cv2.GaussianBlur(split * keep[..., None], (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    share = cv2.GaussianBlur(keep, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    around /= np.maximum(share, np.float32(1e-3))[..., None]
    amount = (spots * weight)[..., None]
    return split + (around - split) * amount


def _skin_bounds(
    mask: NDArray[np.float32], placement: faces_module.Placement
) -> tuple[int, int, int, int] | None:
    """Return the part of *placement*'s box to smooth, or ``None`` for none.

    Where the mask reaches, grown by how far the widest layer's blur looks,
    so that the skin at the edge of the mask is smoothed as it would be in
    the middle of a larger photo.
    """
    left, top, right, bottom = placement.box
    if right <= left or bottom <= top:
        return None
    reached = mask[top:bottom, left:right] > 0.0
    rows = np.flatnonzero(reached.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(reached.any(axis=0))
    margin = int(np.ceil(BLUR_REACH_SIGMAS * 1.2 * max(SKIN_LAYER_SIZES) * placement.size))
    return (
        max(left + int(columns[0]) - margin, left),
        max(top + int(rows[0]) - margin, top),
        min(left + int(columns[-1]) + 1 + margin, right),
        min(top + int(rows[-1]) + 1 + margin, bottom),
    )


# --------------------------------------------------------------------------
# Step 3: white balance (in the camera's own colour space)
# --------------------------------------------------------------------------


def raw_white_balance_gains(
    kelvin: float,
    tint: float,
    shot_multipliers: tuple[float, float, float],
    rgb_xyz_matrix: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Return the gains that move a RAW photo to a new white balance.

    The photo arrives already balanced the way the camera was set, so what is
    applied is the ratio between the wanted gains and those: target over shot,
    with green at one (PLAN.md section 6 step 3).
    """
    target = cm.whitebalance_multipliers(kelvin, tint, rgb_xyz_matrix)
    shot = np.asarray(shot_multipliers, dtype=np.float64)
    shot = np.where(np.abs(shot) < _EPSILON, 1.0, shot)
    gains = target / shot
    return (gains / gains[1]).astype(np.float32)


def jpeg_white_balance_gains(temperature: float, tint: float) -> NDArray[np.float32]:
    """Return the gains for a JPEG's relative white balance sliders.

    A JPEG carries no temperature of its own, so warming it is simply more
    red and less blue, and the three gains are divided by their own average
    so that the photo does not get brighter or darker as well
    (PLAN.md section 6 step 3).
    """
    warm = float(temperature) / 100.0
    green = float(tint) / 100.0
    gains = np.array(
        [
            2.0 ** (JPEG_TEMPERATURE_STOPS * warm),
            2.0 ** (-JPEG_TINT_STOPS * green),
            2.0 ** (-JPEG_TEMPERATURE_STOPS * warm),
        ],
        dtype=np.float64,
    )
    # Weighted by how much each channel contributes to brightness, so that
    # the overall level stays where it was.
    weights = np.asarray(cm.SRGB_TO_XYZ[1], dtype=np.float64)
    return (gains / float(gains @ weights)).astype(np.float32)


def apply_gains(image: NDArray[np.float32], gains: NDArray[np.float32]) -> NDArray[np.float32]:
    """Multiply each colour channel by its own gain."""
    return np.multiply(image, np.asarray(gains, dtype=np.float32), dtype=np.float32)


# --------------------------------------------------------------------------
# Step 4: neutralise clipped highlights (RAW only)
# --------------------------------------------------------------------------


def clipped_weight(
    clipped: NDArray[np.bool_], long_edge: int
) -> NDArray[np.float32] | None:
    """Return the softened mask of step 4, or ``None`` if nothing is clipped.

    Separated from applying it because the softening is a blur, which needs
    the whole photo, while the blend it feeds is per pixel and can be done a
    band at a time.
    """
    if not clipped.any():
        return None
    sigma = max(1.0, long_edge * CLIPPED_BLUR_FRACTION)
    weight = _wide_blur(clipped.astype(np.float32), sigma)
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def blend_to_neutral(
    image: NDArray[np.float32], weight: NDArray[np.float32] | None
) -> NDArray[np.float32]:
    """Move the pixels *weight* points at towards a neutral of their own level."""
    if weight is None:
        return image
    # The brightest channel is the one that still carries the real level, so
    # it is the brightness the neutral is built at.
    neutral = cm.channel_max(image)
    towards_neutral = cv2.subtract(cm.scale_channels(np.ones_like(image), neutral), image)
    return cv2.add(image, cm.scale_channels(towards_neutral, weight))


def neutralise_clipped_highlights(
    image: NDArray[np.float32], clipped: NDArray[np.bool_]
) -> NDArray[np.float32]:
    """Pull clipped pixels towards neutral white (PLAN.md 6 step 4).

    Where a channel has clipped it no longer holds any information, so white
    balancing turns what should be white into magenta or cyan. Those pixels
    are blended towards a neutral of the same brightness, using a softened
    mask as the weight so that the fix fades in instead of showing an edge.
    """
    rows = np.flatnonzero(clipped.any(axis=1))
    if rows.size == 0:
        return image
    columns = np.flatnonzero(clipped.any(axis=0))

    long_edge = max(image.shape[:2])
    sigma = max(1.0, long_edge * CLIPPED_BLUR_FRACTION)
    # Only the part of the photo the softened mask can reach is touched. A
    # clipped sky is a corner of the frame, and the rest of it does not have
    # to be read and written to be left alone.
    reach = int(np.ceil(BLUR_REACH_SIGMAS * sigma)) + 1
    top = max(0, int(rows[0]) - reach)
    bottom = min(image.shape[0], int(rows[-1]) + reach + 1)
    left = max(0, int(columns[0]) - reach)
    right = min(image.shape[1], int(columns[-1]) + reach + 1)

    # The softening and the blend are the two the pipeline itself uses, one a
    # band at a time; they are the same step either way and there is one of
    # them, so what this says about step 4 is what really happens to a photo.
    part = np.ascontiguousarray(image[top:bottom, left:right])
    weight = clipped_weight(
        np.ascontiguousarray(clipped[top:bottom, left:right]), long_edge
    )
    fixed = blend_to_neutral(part, weight)
    if top == 0 and left == 0 and fixed.shape == image.shape:
        return fixed.astype(np.float32)
    result = np.array(image, dtype=np.float32, copy=True)
    result[top:bottom, left:right] = fixed
    return result


# --------------------------------------------------------------------------
# Step 5: camera colour space to the working space
# --------------------------------------------------------------------------


def camera_to_working_space(
    image: NDArray[np.float32], camera_to_xyz: NDArray[np.float64]
) -> NDArray[np.float32]:
    """Convert camera RGB to linear ProPhoto RGB (PLAN.md 6 step 5).

    Camera RGB goes to XYZ with the camera's own matrix, is adapted from D65
    to D50 with Bradford, and lands in the working space.
    """
    matrix = cm.XYZ_TO_PROPHOTO @ cm.ADAPT_D65_TO_D50 @ np.asarray(camera_to_xyz)
    return cm.apply_matrix(image, matrix)


# --------------------------------------------------------------------------
# Steps 6 to 8: exposure, highlights and shadows, contrast
# --------------------------------------------------------------------------


def _luminance(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the brightness of each pixel in the working space."""
    return image @ np.asarray(cm.PROPHOTO_LUMINANCE, dtype=np.float32)


def _scale_evenly(
    image: NDArray[np.float32], scale: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Multiply every channel of each pixel by the same number.

    Scaling the three channels together is what lets the tone adjustments
    change brightness without moving the colour (PLAN.md section 6 steps 7
    and 8).
    """
    return cm.scale_channels(image, scale)


def exposure(image: NDArray[np.float32], stops: float) -> NDArray[np.float32]:
    """Brighten or darken the whole photo (PLAN.md section 6 step 6)."""
    if stops == 0.0:
        return image
    return np.multiply(image, np.float32(2.0**stops), dtype=np.float32)


def highlights_and_shadows(
    image: NDArray[np.float32], highlights: float, shadows: float
) -> NDArray[np.float32]:
    """Lift the shadows and hold back the highlights (PLAN.md 6 step 7).

    Each pixel is judged by how bright its surroundings are, not by its own
    value: the brightness is blurred first, so that the correction varies
    gently across the photo instead of tracing every edge, which is what a
    halo is.
    """
    if highlights == 0.0 and shadows == 0.0:
        return image
    return _scale_evenly(
        image, highlight_shadow_factor(_luminance(image), image.shape, highlights, shadows)
    )


def local_brightness(
    luminance: NDArray[np.float32], shape: tuple[int, ...]
) -> NDArray[np.float32]:
    """Return how bright each pixel's surroundings are, in stops from middle grey.

    This is the one part of step 7 that looks beyond the pixel itself, and
    the only reason the step cannot be done a band at a time; the weights it
    feeds are worked out per pixel in :func:`region_factor`.
    """
    brightness = np.log2(np.maximum(luminance, np.float32(_EPSILON)))
    sigma = max(1.0, max(shape[:2]) * LOCAL_BRIGHTNESS_BLUR)
    local = _wide_blur(brightness, sigma)
    # Measured from middle grey, so that the pivots read as stops either way.
    return (local - np.float32(np.log2(TONE_MID_OUTPUT))).astype(np.float32)


def region_factor(
    relative: NDArray[np.float32], highlights: float, shadows: float
) -> NDArray[np.float32]:
    """Return what each pixel must be multiplied by for step 7."""
    shadow_weight = _falls_off_above(relative, SHADOW_PIVOT_EV)
    highlight_weight = _falls_off_below(relative, HIGHLIGHT_PIVOT_EV)
    change = np.float32(TONE_REGION_MAX_EV) * (
        np.float32(shadows / 100.0) * shadow_weight
        + np.float32(highlights / 100.0) * highlight_weight
    )
    return np.exp2(change).astype(np.float32)


def highlight_shadow_factor(
    luminance: NDArray[np.float32],
    shape: tuple[int, ...],
    highlights: float,
    shadows: float,
) -> NDArray[np.float32]:
    """Return what each pixel must be multiplied by for step 7.

    Separated from applying it so that the three tone steps can be worked out
    on one channel and applied to the photo together.
    """
    return region_factor(local_brightness(luminance, shape), highlights, shadows)


def _falls_off_above(
    relative: NDArray[np.float32],
    pivot: float,
    falloff: float = REGION_FALLOFF_PER_EV,
) -> NDArray[np.float32]:
    """Return a weight that is one in the dark and fades out above *pivot*."""
    return (
        1.0 / (1.0 + np.exp2((relative - np.float32(pivot)) * np.float32(falloff)))
    ).astype(np.float32)


def _falls_off_below(
    relative: NDArray[np.float32],
    pivot: float,
    falloff: float = REGION_FALLOFF_PER_EV,
) -> NDArray[np.float32]:
    """Return a weight that is one in the light and fades out below *pivot*."""
    return (
        1.0 / (1.0 + np.exp2((np.float32(pivot) - relative) * np.float32(falloff)))
    ).astype(np.float32)


def contrast(image: NDArray[np.float32], amount: float) -> NDArray[np.float32]:
    """Stretch or compress the tones around middle grey (PLAN.md 6 step 8).

    The slope is changed in log space with middle grey as the pivot, so that
    what is already middle grey stays exactly where it is while everything
    else moves away from or towards it.
    """
    if amount == 0.0:
        return image
    return _scale_evenly(image, contrast_factor(_luminance(image), amount))


def contrast_factor(
    luminance: NDArray[np.float32], amount: float
) -> NDArray[np.float32]:
    """Return what each pixel must be multiplied by for step 8."""
    slope = np.float32(1.0 + CONTRAST_SLOPE_RANGE * amount / 100.0)
    brightness = np.maximum(luminance, np.float32(_EPSILON))
    middle = np.float32(TONE_MID_OUTPUT)
    # (Y / mid) ** (slope - 1) is the factor that gives the new slope.
    return cm.power(brightness / middle, float(slope) - 1.0, assume_positive=True)


# --------------------------------------------------------------------------
# Step 9: base tone curve (RAW only)
# --------------------------------------------------------------------------


def base_tone_curve(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the base tone curve to a RAW image (PLAN.md 6 step 9).

    Linear RAW data shown as it is looks dark and flat, so a film like curve
    gives the photo its basic shape when every slider is at zero. The curve is
    applied to the brightest channel and the other two are scaled by the same
    factor, which keeps the hue exactly as it was.

    A JPEG already carries the camera's own curve and must not get a second
    one, so this is never applied to one.

    The baseline gain that brings linear sensor data onto the curve's middle
    grey is part of the same step, because the two only make sense together.
    """
    return _scale_evenly(image, base_tone_curve_factor(cm.channel_max(image)))


def base_tone_curve_factor(key: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return what each pixel must be multiplied by for step 9.

    *key* is the brightest channel of the pixel, which is what the curve is
    applied to; the other two follow by the same factor, which is what keeps
    the hue exactly as it was.
    """
    gain = np.float32(2.0**RAW_BASE_EXPOSURE_EV)
    lifted = key * gain
    safe = np.maximum(lifted, np.float32(_EPSILON))
    powered = cm.power(safe, TONE_EXPONENT, assume_positive=True)
    mapped = powered / (powered + np.float32(TONE_HALF_POINT**TONE_EXPONENT))
    return np.where(lifted > _EPSILON, mapped / safe * gain, np.float32(0.0)).astype(
        np.float32
    )


# --------------------------------------------------------------------------
# Step 9.5: black and white points
# --------------------------------------------------------------------------
#
# Where the darkest and the brightest of the photo land. The step sits here,
# after the base tone curve, because this is where both ends of the range
# mean something definite: middle grey is at TONE_MID_OUTPUT and white is at
# 1.0, for a RAW photo and a JPEG alike. Raising the white point sends the
# top of the range above 1.0, which step 11 then brings to white; nothing is
# clipped here but the bottom, where clipping is what a black point is for.
#
# The two ends are not the same kind of operation, because the two ends of a
# range are not alike. A black point is an offset: it is applied to the three
# channels alike, which is what makes a lifted black read as a lifted black
# rather than as a brighter shadow, and it hardly touches middle grey because
# the offset is so small beside it. A white point is a gain, and a gain over
# the whole photo is exposure, so it is weighted to the top of the range and
# applied to the brightest channel with the other two following by the same
# factor, which keeps the hue exactly as it was, as steps 7 to 9 do.


def black_point(blacks: float) -> float:
    """Return where black sits for a blacks slider at *blacks*.

    Below zero the slider raises the black point, so that more of the bottom
    of the range is taken to pure black; above zero it moves the point below
    zero, which lifts pure black to a dark grey.
    """
    return -BLACK_POINT_RANGE * blacks / 100.0


def set_black_point(image: NDArray[np.float32], blacks: float) -> NDArray[np.float32]:
    """Move the black point of *image*, leaving white where it is.

    The offset is taken off every channel and what is left of the range is
    stretched back out, so that 1.0 still comes out as 1.0: white stays white
    however the other end of the range moves.
    """
    if blacks == 0.0:
        return image
    black = np.float32(black_point(blacks))
    scale = np.float32(1.0 / (1.0 - black))
    moved = ((image - black) * scale).astype(np.float32)
    if black <= 0.0:
        # The range has only been lifted, so nothing has crossed zero that
        # was not already below it: the slightly negative channels a camera
        # matrix can produce are left for step 11, as they were before.
        return moved
    return np.maximum(moved, np.float32(0.0))


def white_point_factor(
    key: NDArray[np.float32], whites: float
) -> NDArray[np.float32]:
    """Return what each pixel must be multiplied by for the white point.

    *key* is the brightest channel of the pixel, as in step 9. The weight is
    all but one at the top of the range and has all but let go by middle
    grey, which is what separates this from the exposure of step 6.
    """
    brightness = np.log2(np.maximum(key, np.float32(_EPSILON)))
    relative = brightness - np.float32(np.log2(TONE_MID_OUTPUT))
    weight = _falls_off_below(relative, WHITE_PIVOT_EV, WHITE_FALLOFF_PER_EV)
    change = np.float32(WHITE_POINT_MAX_EV * whites / 100.0) * weight
    return np.exp2(change).astype(np.float32)


def set_white_point(image: NDArray[np.float32], whites: float) -> NDArray[np.float32]:
    """Move the white point of *image*, leaving the midtones where they are."""
    if whites == 0.0:
        return image
    return _scale_evenly(image, white_point_factor(cm.channel_max(image), whites))


def black_and_white_points(
    image: NDArray[np.float32], blacks: float, whites: float
) -> NDArray[np.float32]:
    """Set where the darkest and the brightest of the photo land.

    The black point first and the white point after it, the order a set of
    levels is read in: the black point leaves white at 1.0, so moving it
    first hands the white point the range it expects.
    """
    return set_white_point(set_black_point(image, blacks), whites)


# --------------------------------------------------------------------------
# Step 9.6: vignette
# --------------------------------------------------------------------------
#
# Darkening or lifting the edges of the frame, to put the eye where the photo
# wants it. The step sits after the black and white points for the same
# reason they sit where they do: this is where middle grey is at
# TONE_MID_OUTPUT and white at 1.0 for a RAW photo and a JPEG alike, so that
# "two stops down in the corner" means the same thing for both. It is still
# before step 11, which is what lets a bright vignette push the corners past
# 1.0 and have them brought back to white instead of clipped.
#
# The factor is a gain on the three channels alike, so the hue and the ratio
# between the channels are untouched: nothing but the light changes, which is
# what keeps a vignette from tinting the edges of the frame.
#
# Where a pixel sits is measured out to the corner of the frame rather than
# in pixels, so the preview and the full size export come out the same, and
# so the shape follows the frame: the geometry of step 2 has already cropped
# the photo by the time it arrives here, which makes this a vignette of the
# photo as it is kept, not of the sensor it came from.


def vignette_axes(
    shape: tuple[int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return how far each row and each column lies from the middle, squared.

    Measured from the middle of the frame out to the outermost pixel, so that
    a photo and its smaller preview give the same factor at the same place in
    the frame rather than differing by half a pixel at the edge.

    The two are kept apart rather than worked up into a plane because a band
    of the photo needs only its own rows: the whole factor would be as large
    as a channel of the photo, and this way nothing of that size is built.
    """
    height, width = shape
    across = (2.0 * np.arange(width, dtype=np.float32) / max(width - 1, 1) - 1.0) ** 2
    down = (2.0 * np.arange(height, dtype=np.float32) / max(height - 1, 1) - 1.0) ** 2
    return down.astype(np.float32), across.astype(np.float32)


def vignette_from_axes(
    down: NDArray[np.float32],
    across: NDArray[np.float32],
    amount: float,
    midpoint: float,
) -> NDArray[np.float32]:
    """Return the factor for the rows in *down* and the columns in *across*."""
    # Halved, so that the radius is 1.0 at the corner of any frame.
    radius = np.sqrt((across[None, :] + down[:, None]) * np.float32(0.5))
    start = np.float32(min(max(midpoint, 0.0) / 100.0, VIGNETTE_MIDPOINT_LIMIT))
    reach = np.clip((radius - start) / (np.float32(1.0) - start), 0.0, 1.0)
    weight = reach * reach * (np.float32(3.0) - np.float32(2.0) * reach)
    change = np.float32(VIGNETTE_MAX_EV * amount / 100.0) * weight
    return np.exp2(change).astype(np.float32)


def vignette_factor(
    shape: tuple[int, int], amount: float, midpoint: float
) -> NDArray[np.float32]:
    """Return what each pixel of a photo this shape must be multiplied by.

    *shape* is the height and width of the photo, *amount* runs from -100 for
    the darkest corners to +100 for the brightest, and *midpoint* is how much
    of the frame is left alone, as a percentage of the way out to the corner.

    The falloff from the midpoint to the corner is a smoothstep, which leaves
    no edge at either end of it: the correction arrives and departs with a
    slope of zero, so neither the untouched middle nor the corner shows where
    the vignette begins. A low midpoint therefore spreads the transition over
    most of the frame and a high one keeps it to the corners, which is the
    feathering as well as the size.
    """
    return vignette_from_axes(*vignette_axes(shape), amount, midpoint)


def set_vignette(
    image: NDArray[np.float32], amount: float, midpoint: float
) -> NDArray[np.float32]:
    """Darken or lift the edges of *image* (PLAN.md section 6 step 9.6)."""
    if amount == 0.0:
        return image
    return _scale_evenly(image, vignette_factor(image.shape[:2], amount, midpoint))


# --------------------------------------------------------------------------
# Step 9.7: the film filter
# --------------------------------------------------------------------------
#
# A look is a measured rendering rather than an adjustment: three tone curves
# and three functions of hue fitted to what a camera makes of its own RAW
# file (:mod:`core.filters`). It sits here, after the vignette and before
# vibrance, for the reason that fixed where it was measured. A camera's JPEG
# is its sensor, its white balance and its tone curve, which is steps 1 to
# 9.6 of this pipeline with every slider at rest; a look is the step from
# there to what the camera wrote, so here is the one place it means what it
# was fitted to mean.
#
# Putting it here also settles two things the user would otherwise have to
# think about. The tone sliders come first, so raising the exposure and then
# choosing a look gives the look a brighter photo to render, the way film
# answers to more light. Vibrance, saturation and the gamut compression come
# after, so the colour sliders still work on what is on screen and a look
# that reaches outside sRGB is brought back in by step 11 rather than being
# flattened against the edge of it.
#
# PLAN.md section 6 does not yet number this step; the filter is a feature
# added after it was written (see docs/PROGRESS.md, "濾鏡功能").


def film_filter(
    image: NDArray[np.float32], look: Look | None, strength: float
) -> NDArray[np.float32]:
    """Render the photo through a film look, at *strength* per cent."""
    if look is None or strength <= 0.0:
        return image
    return look.apply(image, strength / 100.0)


# --------------------------------------------------------------------------
# Step 10: vibrance and saturation, in OkLCh
# --------------------------------------------------------------------------


#: The skin hue band of PLAN.md section 6 step 10, as the slopes of its two
#: edges in Oklab. Testing against these costs two multiplications, where
#: working out the hue angle itself would cost an arc tangent per pixel.
_SKIN_SLOPE_FROM: Final[float] = float(np.tan(np.radians(SKIN_HUE_FROM_DEGREES)))
_SKIN_SLOPE_TO: Final[float] = float(np.tan(np.radians(SKIN_HUE_TO_DEGREES)))



def _vibrance_weight(
    chroma: NDArray[np.float32], a: NDArray[np.float32], b: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return how much vibrance applies to each pixel.

    Vibrance is saturation that knows when to stop: it lifts the colours that
    have little of their own and leaves the ones that are already strong,
    and it holds back in the hues skin falls in so that faces do not turn
    orange (PLAN.md section 6 step 10).
    """
    room = np.clip(1.0 - chroma / np.float32(VIBRANCE_FULL_CHROMA), 0.0, 1.0).astype(
        np.float32
    )
    # The skin band lies between two rays out of the neutral axis, so a pixel
    # is inside it when b sits between those two multiples of a.
    in_skin = (
        (a > 0.0) & (b >= a * np.float32(_SKIN_SLOPE_FROM)) & (b <= a * np.float32(_SKIN_SLOPE_TO))
    )
    return np.where(in_skin, room * np.float32(SKIN_VIBRANCE_WEIGHT), room).astype(np.float32)


def base_chroma(is_raw: bool, filter_strength: float | None = None) -> float:
    """Return the chroma a photo is given before its colour sliders.

    A RAW photo gets :data:`RAW_BASE_CHROMA`, a JPEG nothing. A film filter
    was measured from our plain rendering all the way to the camera's, colour
    included, so as it is turned up it takes over from this one: at full
    strength none is left, and at zero the photo is exactly what it would be
    with no filter at all. *filter_strength* is that strength in per cent, or
    ``None`` when no filter is worn.
    """
    if not is_raw:
        return 1.0
    worn = 0.0 if filter_strength is None else min(max(filter_strength / 100.0, 0.0), 1.0)
    return 1.0 + (RAW_BASE_CHROMA - 1.0) * (1.0 - worn)


def vibrance_and_saturation(
    image: NDArray[np.float32], vibrance: float, saturation: float, base: float = 1.0
) -> NDArray[np.float32]:
    """Change how colourful the photo is (PLAN.md section 6 step 10).

    *base* is the chroma the photo starts from, :func:`base_chroma`; the
    sliders then act on the photo as it looks with it, so vibrance judges
    how colourful a pixel already is after it.

    Only chroma moves: lightness and hue are left exactly as they are, which
    is why a sunset gets stronger rather than shifting towards orange. The
    work is done in Oklab rather than OkLCh because scaling chroma there is
    scaling a and b by the same number, which spares an arc tangent and a
    sine and cosine for every pixel.
    """
    uniform = max(base * (1.0 + saturation / 100.0), 0.0)
    if vibrance == 0.0:
        if uniform == 1.0:
            return image
        # The same number for every pixel is a linear map of Oklab, which
        # folds into the matrices on either side of it.
        return cm.scale_oklab_chroma(image, uniform)

    lab = cm.prophoto_to_oklab(image)
    a = lab[..., 1]
    b = lab[..., 2]
    weight = _vibrance_weight(np.hypot(a, b) * np.float32(base), a, b)
    factor = np.float32(uniform) * (1.0 + np.float32(vibrance / 100.0) * weight)
    factor = np.maximum(factor, 0.0)

    adjusted = np.stack((lab[..., 0], a * factor, b * factor), axis=-1).astype(np.float32)
    return cm.oklab_to_prophoto(adjusted)


def skin_tone(
    image: NDArray[np.float32], mask: NDArray[np.float32] | None, amount: float
) -> NDArray[np.float32]:
    """Brighten the skin and take some of the yellow out (PLAN.md 6 step 10).

    In Oklab, as the colour sliders work: the lightness is raised by a share
    of itself, the hue is turned a few degrees from yellow towards red and a
    little chroma is let go, each by as much as the mask and the slider say.
    A pixel the mask leaves out comes back exactly as it was.
    """
    if mask is None or amount <= 0.0:
        return image
    reached = mask > 0.0
    rows = np.flatnonzero(reached.any(axis=1))
    if rows.size == 0:
        return image
    columns = np.flatnonzero(reached.any(axis=0))
    # Only the part the skin reaches is taken through Oklab.
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    if (top, bottom, left, right) != (0, image.shape[0], 0, image.shape[1]):
        result = image.copy()
        result[top:bottom, left:right] = skin_tone(
            image[top:bottom, left:right], mask[top:bottom, left:right], amount
        )
        return result
    weight = mask * np.float32(amount / 100.0)
    touched = weight > 0.0
    lab = cm.prophoto_to_oklab(image)
    angle = np.radians(SKIN_HUE_TURN_DEGREES) * weight
    keep = np.float32(1.0) + np.float32(SKIN_CHROMA_KEPT - 1.0) * weight
    cosine = np.cos(angle).astype(np.float32) * keep
    sine = np.sin(angle).astype(np.float32) * keep
    a = lab[..., 1]
    b = lab[..., 2]
    turned = np.stack(
        (
            lab[..., 0] * (np.float32(1.0) + np.float32(SKIN_LIGHTNESS_GAIN) * weight),
            a * cosine - b * sine,
            a * sine + b * cosine,
        ),
        axis=-1,
    ).astype(np.float32)
    return np.where(touched[..., None], cm.oklab_to_prophoto(turned), image)


def whiten(
    image: NDArray[np.float32], mask: NDArray[np.float32] | None, amount: float
) -> NDArray[np.float32]:
    """Make the skin lighter and less yellow (PLAN.md 6 step 10, A-30).

    In Oklab: the lightness is raised by up to WHITEN_LIGHTNESS_GAIN of
    itself, most for lit skin and least for shadows and highlights, and a
    share of the yellow is taken out, shaped the same way, each by as much
    as the mask and the slider say. A pixel the mask leaves out comes back exactly as it was.
    """
    if mask is None or amount <= 0.0:
        return image
    reached = mask > 0.0
    rows = np.flatnonzero(reached.any(axis=1))
    if rows.size == 0:
        return image
    columns = np.flatnonzero(reached.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    if (top, bottom, left, right) != (0, image.shape[0], 0, image.shape[1]):
        result = image.copy()
        result[top:bottom, left:right] = whiten(
            image[top:bottom, left:right], mask[top:bottom, left:right], amount
        )
        return result
    weight = mask * np.float32(amount / 100.0)
    touched = weight > 0.0
    lab = cm.prophoto_to_oklab(image)
    lightness = np.clip(lab[..., 0], 0.0, 1.0)
    # L squared times (1 - L) is at most 4/27, at two thirds.
    lift = lightness * lightness * (np.float32(1.0) - lightness) * np.float32(27.0 / 4.0)
    yellow = np.maximum(lab[..., 2], np.float32(0.0))
    whitened = np.stack(
        (
            lab[..., 0] * (np.float32(1.0) + np.float32(WHITEN_LIGHTNESS_GAIN) * weight * lift),
            lab[..., 1],
            # As little in shadow as the lift, or a nostril turns purple.
            lab[..., 2] - yellow * np.float32(WHITEN_YELLOW_TAKEN) * weight * lift,
        ),
        axis=-1,
    ).astype(np.float32)
    return np.where(touched[..., None], cm.oklab_to_prophoto(whitened), image)


# --------------------------------------------------------------------------
# Step 11: gamut compression in OkLCh
# --------------------------------------------------------------------------


def _is_in_srgb(cone_root: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Return which colours fit in sRGB, given cube rooted cone responses."""
    srgb = cone_root**3 @ np.asarray(cm.CONE_TO_SRGB_LINEAR, dtype=np.float32).T
    return np.all((srgb >= -_EPSILON) & (srgb <= 1.0 + _EPSILON), axis=-1)


#: Size of the table of where the output gamut ends. The boundary depends
#: only on lightness and hue, so it is worked out once over a grid and looked
#: up per pixel, instead of searching for it again for every pixel of every
#: preview, which is sixteen times the work.
#:
#: At this size no pixel of the sample photos ends up more than four steps of
#: 255 away from what the search itself gives, and the average is a twentieth
#: of one step. Both the preview and the export read the same table, so they
#: still agree with each other exactly, which is what PLAN.md section 6 asks.
_GAMUT_TABLE_LIGHTNESS: Final[int] = 384
_GAMUT_TABLE_HUE: Final[int] = 720

_gamut_table: NDArray[np.float32] | None = None


def gamut_table() -> NDArray[np.float32]:
    """Return the largest chroma sRGB holds, over lightness and hue."""
    global _gamut_table
    if _gamut_table is None:
        lightness = np.linspace(0.0, 1.0, _GAMUT_TABLE_LIGHTNESS, dtype=np.float32)
        hue = np.linspace(-np.pi, np.pi, _GAMUT_TABLE_HUE, dtype=np.float32)
        grid_lightness, grid_hue = np.meshgrid(lightness, hue, indexing="ij")
        _gamut_table = _max_chroma(grid_lightness.ravel(), grid_hue.ravel()).reshape(
            _GAMUT_TABLE_LIGHTNESS, _GAMUT_TABLE_HUE
        )
    return _gamut_table


def _max_chroma_from_table(
    lightness: NDArray[np.float32], hue: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return the largest chroma that fits in sRGB, read from the table.

    The four surrounding entries are blended, which follows the boundary far
    more closely than taking the smallest of them: the boundary turns sharply
    near its widest point for each hue, and reading low there would visibly
    dull colours that had room to spare. Where the blend reads slightly high
    instead, the colour is left a shade outside the gamut and the output
    transform trims it, which at this size is a fraction of one step.
    """
    table = gamut_table()
    rows = np.clip(lightness, 0.0, 1.0) * np.float32(_GAMUT_TABLE_LIGHTNESS - 1)
    columns = (hue + np.float32(np.pi)) / np.float32(2.0 * np.pi) * np.float32(
        _GAMUT_TABLE_HUE - 1
    )

    row_low = np.clip(np.floor(rows), 0, _GAMUT_TABLE_LIGHTNESS - 1).astype(np.intp)
    row_high = np.minimum(row_low + 1, _GAMUT_TABLE_LIGHTNESS - 1)
    column_low = np.clip(np.floor(columns), 0, _GAMUT_TABLE_HUE - 1).astype(np.intp)
    column_high = np.minimum(column_low + 1, _GAMUT_TABLE_HUE - 1)

    row_fraction = (rows - row_low).astype(np.float32)
    column_fraction = (columns - column_low).astype(np.float32)

    top = table[row_low, column_low] * (1.0 - column_fraction) + table[
        row_low, column_high
    ] * column_fraction
    bottom = table[row_high, column_low] * (1.0 - column_fraction) + table[
        row_high, column_high
    ] * column_fraction
    return (top * (1.0 - row_fraction) + bottom * row_fraction).astype(np.float32)


def _max_chroma(
    lightness: NDArray[np.float32], hue: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Return the largest chroma that still fits in sRGB, per colour.

    The sRGB solid is star shaped around the neutral axis in OkLCh, so a
    binary search on chroma finds the boundary; ten steps land within a
    thousandth of the range, far below what a display can show.

    Hue does not change during the search, so the line of constant hue and
    lightness is set up once as a starting point and a direction. Each step is
    then one multiply and add plus one matrix, instead of a whole conversion
    from OkLCh back to RGB.
    """
    basis = np.asarray(cm.OKLAB_TO_CONE_ROOT, dtype=np.float32).T
    zero = np.zeros_like(lightness)
    start = np.stack((lightness, zero, zero), axis=-1) @ basis
    direction = np.stack((zero, np.cos(hue), np.sin(hue)), axis=-1) @ basis

    low = zero
    high = np.full_like(lightness, np.float32(GAMUT_MAX_CHROMA))
    for _ in range(GAMUT_SEARCH_ITERATIONS):
        middle = (low + high) * np.float32(0.5)
        inside = _is_in_srgb(start + middle[..., np.newaxis] * direction)
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return low


def compress_to_gamut(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Fold colours outside sRGB back inside it (PLAN.md 6 step 11).

    Lightness and hue are kept; only chroma is reduced, and the reduction
    starts smoothly just inside the boundary so that a saturated
    sunset turns into a gradient rather than a flat patch of colour. Anything
    brighter than white converges on white at the same time.

    Only the colours that are actually near or past the boundary are searched
    for: a colour needs work exactly when raising its chroma to the start of
    the compression range would take it out of gamut, which is one cheap test.
    In an ordinary photo that leaves a small fraction of the pixels to search.
    """
    candidates = _needs_compression(image)
    if not candidates.any():
        # Nothing to fold, so the photo passes through untouched, as the
        # steps before this one do when their slider has not been moved.
        # Copying it here was a whole read and write of the band for nothing.
        return image

    lch = cm.prophoto_to_oklch(image[candidates])
    lightness = np.clip(lch[..., 0], 0.0, 1.0).astype(np.float32)
    chroma = lch[..., 1].astype(np.float32)
    hue = lch[..., 2].astype(np.float32)

    ceiling = _max_chroma_from_table(lightness, hue)
    knee = ceiling * np.float32(GAMUT_COMPRESSION_START)
    headroom = np.maximum(ceiling - knee, np.float32(_EPSILON))
    compressed = knee + headroom * np.tanh((chroma - knee) / headroom)
    chroma = np.where(chroma > knee, compressed, chroma).astype(np.float32)

    result = np.array(image, dtype=np.float32, copy=True)
    result[candidates] = cm.oklch_to_prophoto(np.stack((lightness, chroma, hue), axis=-1))
    return result


def _needs_compression(image: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Return which pixels are near or past the edge of the output gamut.

    Converting a whole photo to OkLCh and back costs far more than the
    compression itself, which in an ordinary photo touches a fraction of a
    percent of the pixels. This picks the candidates in RGB instead: a pixel
    matters when pushing it away from its own neutral, by the amount the
    compression range covers, would take it outside sRGB.

    The test is deliberately generous. Pushing away from a neutral of equal
    luminance is close to, but not exactly, a chroma increase at constant
    lightness, so the margin is taken a little wider than the compression
    range to make sure no pixel that needs work is missed.
    """
    srgb = cm.apply_matrix(image, cm.PROPHOTO_TO_SRGB)
    luminance = srgb @ np.asarray(cm.SRGB_TO_XYZ[1], dtype=np.float32)
    margin = np.float32(GAMUT_PROBE_MARGIN)
    # The probe is luminance + (channel - luminance) / margin, so it leaves
    # the 0 to 1 range exactly when the brightest or darkest channel does.
    # Comparing those two directly avoids building a whole probe image.
    above = cm.channel_max(srgb) > luminance + margin * (np.float32(1.0) - luminance)
    below = cm.channel_min(srgb) < luminance * (np.float32(1.0) - margin)
    return above | below | (luminance > np.float32(1.0))


# --------------------------------------------------------------------------
# Step 11.5: film grain
# --------------------------------------------------------------------------
#
# Grain is the last thing laid on the photo, over the finished sRGB values
# that step 12 is about to round, because that is where the camera's grain
# was measured to be: the same amount added to all three encoded channels,
# which is what keeps it grey, and equally strong in sRGB levels from the
# shadows to the light tones. Nothing after it sharpens, curves or softens
# it. A channel close to 0 or to 1 has little room to move, so each channel's
# share is rolled off smoothly into the room it has rather than cut - which
# is also why a saturated colour, one of whose channels is at an end, takes
# the grain more quietly, as it does in the camera.
#
# The noise of each pixel is worked out from where the pixel lies in the
# frame and from the photo's seed, not drawn one after another, so a window
# of the photo at full size gets exactly the grain the export has there, and
# the same photo gets the same grain every time.

#: Entries in the table of grain strength over brightness.
_GRAIN_TABLE_SIZE: Final[int] = 1024
#: How far one grain reaches from its centre, in pixels.
_GRAIN_REACH: Final[int] = 2
#: The spread of the sum of two uniform numbers in 0 to 1.
_TRIANGULAR_SPREAD: Final[float] = float(np.sqrt(1.0 / 6.0))
#: The weights an sRGB encoded colour's brightness is read with for the
#: grain's strength, the ones it was measured with (Rec. 709).
_GRAIN_LUMA: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)


def _grain_rings() -> tuple[tuple[float, tuple[tuple[int, int], ...]], ...]:
    """Return the shape of one grain as rings of neighbours and their weights.

    Each ring is every offset at one squared distance, and its weight is
    spread so that the grain's spread comes out as one.
    """
    reach = _GRAIN_REACH
    offsets: dict[int, list[tuple[int, int]]] = {}
    for down in range(-reach, reach + 1):
        for across in range(-reach, reach + 1):
            offsets.setdefault(down * down + across * across, []).append((down, across))
    energy = sum(weight**2 * len(offsets.get(distance, ())) for distance, weight in GRAIN_KERNEL)
    norm = 1.0 / np.sqrt(energy)
    return tuple(
        (weight * norm, tuple(offsets[distance]))
        for distance, weight in GRAIN_KERNEL
        if distance in offsets
    )


def _shape_grains(noise: NDArray[np.float32]) -> NDArray[np.float32]:
    """Shape noise into grains, losing :data:`_GRAIN_REACH` pixels at each edge.

    Added up ring by ring in a fixed order rather than with OpenCV's filter,
    whose sums come out a rounding apart depending on the size of the image:
    a window and the whole frame must get the same grain to the last bit.
    """
    reach = _GRAIN_REACH
    height = noise.shape[0] - 2 * reach
    width = noise.shape[1] - 2 * reach
    shaped = np.zeros((height, width), dtype=np.float32)
    for weight, offsets in GRAIN_RINGS:
        ring = np.zeros((height, width), dtype=np.float32)
        for down, across in offsets:
            ring += noise[reach + down : reach + down + height, reach + across : reach + across + width]
        shaped += ring * np.float32(weight)
    return shaped


def _grain_amplitudes() -> NDArray[np.float32]:
    """Return how far the grain moves an encoded value, over brightness 0 to 1."""
    brightness = np.linspace(0.0, _EIGHT_BIT_MAX, _GRAIN_TABLE_SIZE)
    weight = np.interp(brightness, GRAIN_TONE_LEVELS, GRAIN_TONE_WEIGHTS)
    return (weight * GRAIN_STRENGTH_LEVELS / _EIGHT_BIT_MAX).astype(np.float32)


GRAIN_RINGS: Final[tuple[tuple[float, tuple[tuple[int, int], ...]], ...]] = _grain_rings()
GRAIN_AMPLITUDES: Final[NDArray[np.float32]] = _grain_amplitudes()


def has_grain(state: EditState) -> bool:
    """Return whether the grain slider asks for anything."""
    return state.grain != 0.0


def _hashed_noise(
    top: int, left: int, height: int, width: int, seed: int
) -> NDArray[np.float32]:
    """Return noise of spread one for these rows and columns of a frame.

    Each pixel's number is a hash of where it is and of *seed*, so any part
    of the frame comes out the same whichever window it is asked for in.
    The two halves of the hash are added, which gives a triangular spread:
    once shaped into grains it is as close to the camera's as a bell curve
    would be, and costs one hash rather than two.
    """
    rows = np.arange(top, top + height, dtype=np.int64).astype(np.uint32)
    columns = np.arange(left, left + width, dtype=np.int64).astype(np.uint32)
    value = (columns * np.uint32(0x9E3779B1))[np.newaxis, :] ^ (
        rows * np.uint32(0x85EBCA77) + np.uint32(seed & 0xFFFFFFFF)
    )[:, np.newaxis]
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x2C1B3C6D)
    value ^= value >> np.uint32(12)
    value *= np.uint32(0x297A2D39)
    value ^= value >> np.uint32(15)
    low = (value & np.uint32(0xFFFF)).astype(np.float32)
    high = (value >> np.uint32(16)).astype(np.float32)
    unit = np.float32(1.0 / (_SIXTEEN_BIT_MAX * _TRIANGULAR_SPREAD))
    return ((low + high) * unit - np.float32(1.0 / _TRIANGULAR_SPREAD)).astype(np.float32)


def grain_strength_at(scale: float) -> float:
    """Return how much of the full size grain a copy at *scale* keeps."""
    if scale >= GRAIN_FULL_SIZE_FROM:
        return 1.0
    return float(np.interp(scale, GRAIN_SHRUNK_SCALES, GRAIN_SHRUNK_STRENGTHS))


def grain_field(photo: DecodedPhoto, window: Window | None = None) -> NDArray[np.float32]:
    """Return the grain of *photo*, one number per pixel, before its strength.

    *photo* is the frame, or the *window* of it, at whatever resolution is at
    hand. At full size each pixel's noise is shaped into grains by
    :data:`GRAIN_KERNEL` from the noise around it, which is worked out
    for a two pixel margin and cut off again, so a window has no seam. A
    smaller copy gets the grain that shrinking the full size one would have
    left, which is finer than one of its pixels: plain noise, as strong as
    :func:`grain_strength_at` says.

    The field depends on nothing but the frame, the window and the seed, so
    a caller can keep it while the sliders move.
    """
    pane = Window.whole(photo.size) if window is None else window
    width, height = pane.size
    scale = detail_scale(photo)
    full_size = scale >= GRAIN_FULL_SIZE_FROM
    # A smaller copy is a different grid of pixels from the full size one,
    # so it gets numbers of its own; the same size of copy, the same ones.
    seed = photo.grain_seed if full_size else photo.grain_seed ^ round(scale * _SIXTEEN_BIT_MAX)
    strength = np.float32(grain_strength_at(scale))
    reach = _GRAIN_REACH
    rows = np.arange(pane.top, pane.top + height, dtype=np.int64)[:, np.newaxis]

    def grains(band: NDArray[np.int64]) -> NDArray[np.float32]:
        """The grain of one band of rows."""
        first = int(band[0, 0])
        count = band.shape[0]
        if not full_size:
            return _hashed_noise(first, pane.left, count, width, seed) * strength
        noise = _hashed_noise(
            first - reach, pane.left - reach, count + 2 * reach, width + 2 * reach, seed
        )
        return _shape_grains(noise)

    return in_bands(grains, rows)


def apply_grain(
    encoded: NDArray[np.float32], field: NDArray[np.float32], amount: float
) -> NDArray[np.float32]:
    """Lay film grain over sRGB encoded values in 0 to 1 (PLAN.md 6 step 11.5).

    *field* is :func:`grain_field` for the same pixels and *amount* the
    slider. How far the grain moves a pixel is looked up from its brightness,
    so that it is even in the midtones and fades into black and white; the
    same amount is then given to each channel, rolled off as it nears the end
    of that channel's room so that nothing leaves the 0 to 1 range.
    """
    if amount == 0.0:
        return encoded
    brightness = encoded @ np.asarray(_GRAIN_LUMA, dtype=np.float32)
    index = np.rint(
        np.clip(brightness, 0.0, 1.0) * np.float32(_GRAIN_TABLE_SIZE - 1)
    ).astype(np.intp)
    shift = (GRAIN_AMPLITUDES[index] * field * np.float32(amount / 100.0))[..., np.newaxis]
    # The room each channel has in the direction it is being moved, and the
    # move rolled off into it as s r / (s^4 + r^4)^(1/4): a hair from the
    # identity until the room is nearly used up, which is how the camera's
    # grain behaves until it clips, then bending smoothly onto the end
    # instead of stopping at it. With no room the channel does not move.
    room = np.maximum(np.where(shift > 0.0, np.float32(1.0) - encoded, encoded), 0.0)
    shift_squared = shift * shift
    room_squared = room * room
    reach = np.sqrt(np.sqrt(shift_squared * shift_squared + room_squared * room_squared))
    moved = shift * room / np.maximum(reach, np.float32(1e-12))
    return np.clip(encoded + moved, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Step 12: output transform
# --------------------------------------------------------------------------


def to_srgb(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert the working space to sRGB encoded values in 0 to 1."""
    linear = cm.apply_matrix(image, cm.PROPHOTO_TO_SRGB)
    return cm.linear_to_srgb(np.clip(linear, 0.0, 1.0))


def to_srgb8(image: NDArray[np.float32], dither: bool = False) -> NDArray[np.uint8]:
    """Convert the working space to 8 bit sRGB (PLAN.md 6 step 12)."""
    return quantise_srgb8(to_srgb(image), dither)


def quantise_srgb8(encoded: NDArray[np.float32], dither: bool = False) -> NDArray[np.uint8]:
    """Round sRGB encoded values in 0 to 1 to 8 bit samples.

    With *dither*, triangular noise of plus or minus half a step is added
    before rounding, which keeps a gradual sky from breaking into bands.
    Shared with :mod:`core.collage`, whose canvas is already sRGB.
    """
    scaled = encoded * np.float32(_EIGHT_BIT_MAX)
    if dither:
        generator = np.random.default_rng()
        shape = scaled.shape
        triangular = generator.random(shape, dtype=np.float32) - generator.random(
            shape, dtype=np.float32
        )
        scaled = scaled + triangular * np.float32(DITHER_AMPLITUDE_STEPS)
    # The values are already inside the output range, so the saturating cast
    # only has to round; it is three times faster than doing that in NumPy.
    return cv2.convertScaleAbs(scaled)


def to_srgb16(image: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Convert the working space to 16 bit sRGB (PLAN.md 6 step 12)."""
    return quantise_srgb16(to_srgb(image))


def quantise_srgb16(encoded: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Round sRGB encoded values in 0 to 1 to 16 bit samples.

    Quantised straight, with no dither: 65536 steps are far finer than any
    gradient can band at, which is why PLAN.md asks for dither at 8 bit only.
    """
    return np.rint(encoded * np.float32(_SIXTEEN_BIT_MAX)).astype(np.uint16)


def widen_to_16_bit(image: NDArray[np.uint8]) -> NDArray[np.uint16]:
    """Return 8 bit samples spread over the 16 bit range.

    Used when a JPEG whose pixels pass through untouched is written to a
    16 bit file: there is no more information to be had, so each step is
    simply scaled to the same relative value.
    """
    return image.astype(np.uint16) * np.uint16(_EIGHT_TO_SIXTEEN)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def shot_white_balance(photo: DecodedPhoto) -> tuple[float, float]:
    """Return the temperature and tint the camera recorded for a RAW photo.

    This is where a RAW photo's white balance sliders start, so that leaving
    them alone means leaving the photo as the camera saw it
    (PLAN.md section 6 step 3).
    """
    if not photo.is_raw or photo.rgb_xyz_matrix is None or photo.camera_whitebalance is None:
        return 0.0, 0.0
    multipliers = np.asarray(photo.camera_whitebalance, dtype=np.float64)
    multipliers = np.where(np.abs(multipliers) < _EPSILON, 1.0, multipliers)
    return cm.camera_neutral_to_temperature(1.0 / multipliers, photo.rgb_xyz_matrix)


#: Why a pick could not be used: part of the patch was clipped, so one of
#: its channels no longer says what colour the light was ...
PICK_CLIPPED: Final[str] = "clipped"
#: ... or it was too dark for its colour to be told from noise.
PICK_DARK: Final[str] = "dark"


@dataclass(frozen=True)
class WhiteBalancePick:
    """What the eyedropper found: the balance that turns a spot grey.

    ``problem`` is set, and the two numbers mean nothing, when the spot could
    not be used (``PICK_CLIPPED`` or ``PICK_DARK``).
    """

    temperature: float = 0.0
    tint: float = 0.0
    problem: str | None = None


def pick_white_balance(photo: DecodedPhoto, point: tuple[float, float]) -> WhiteBalancePick:
    """Return the temperature and tint that make the spot at *point* neutral.

    *point* is where the user clicked, in fractions of the width and height
    of *photo*, which is the photo as it is shown: turned, rotated and
    cropped. The median of a small patch around it is taken, so that
    neither noise nor a speck in the patch decides the result.

    The answer is worked out from the decoded colours and not from the
    sliders, so it is the same wherever the sliders are and clicking the same
    spot twice changes nothing; the later steps (exposure, the filter) do not
    move a neutral off neutral, so they do not enter into it. A RAW photo
    arrives balanced the way the camera was set, so its own colours divided
    by the camera's multipliers are what the sensor saw, and that is a
    neutral the temperature can be read from exactly as the shot balance is
    (PLAN.md section 6 step 3). A JPEG's relative sliders are solved for
    directly. Either answer may lie outside a slider's range, which the
    caller's state clamps.
    """
    height, width = photo.image.shape[:2]
    radius = max(1, round(max(width, height) * PICK_RADIUS_FRACTION))
    column = min(width - 1, max(0, int(point[0] * width)))
    row = min(height - 1, max(0, int(point[1] * height)))
    rows = slice(max(0, row - radius), min(height, row + radius + 1))
    columns = slice(max(0, column - radius), min(width, column + radius + 1))
    if photo.clipped[rows, columns].any():
        return WhiteBalancePick(problem=PICK_CLIPPED)
    # The median rather than the mean, so that a speck of dust, a hot pixel
    # or the edge of something else in the patch does not pull the answer.
    level = np.median(photo.image[rows, columns].reshape(-1, 3), axis=0).astype(np.float64)
    if float(level.min()) <= _EPSILON or float(level.max()) < PICK_DARKEST:
        return WhiteBalancePick(problem=PICK_DARK)

    if photo.is_raw:
        if photo.rgb_xyz_matrix is None or photo.camera_whitebalance is None:
            return WhiteBalancePick(*shot_white_balance(photo))
        shot = np.asarray(photo.camera_whitebalance, dtype=np.float64)
        shot = np.where(np.abs(shot) < _EPSILON, 1.0, shot)
        kelvin, tint = cm.camera_neutral_to_temperature(level / shot, photo.rgb_xyz_matrix)
        return WhiteBalancePick(float(round(kelvin)), float(round(tint)))

    red, green, blue = (float(value) for value in level)
    # The temperature gains are reciprocal, so they meet halfway: red and
    # blue end up at their geometric mean, and the tint brings green to it.
    warm = -np.log2(red / blue) / (2.0 * JPEG_TEMPERATURE_STOPS)
    magenta = -np.log2(np.sqrt(red * blue) / green) / JPEG_TINT_STOPS
    return WhiteBalancePick(float(round(100.0 * warm)), float(round(100.0 * magenta)))


def is_passthrough(photo: DecodedPhoto, state: EditState | None = None) -> bool:
    """Return whether *photo* should be shown exactly as it was decoded.

    An sRGB JPEG with nothing changed is already a finished photo, so it is
    shown pixel for pixel instead of being put through the pipeline
    (PLAN.md section 6, "JPG 原圖直通"). A file in another colour space still
    has to be converted, or its colours would be wrong on screen. An 8 bit
    PNG is a finished photo in the same way and is shown the same way; a 16
    bit one keeps no 8 bit copy to show, so it takes the pipeline.
    """
    if photo.is_raw or photo.color_space is not ColorSpace.SRGB or photo.encoded is None:
        return False
    if state is None:
        return True
    if skin_range.reaches_nothing(photo.faces, photo.skin):
        # The face sliders change nothing on a photo without skin for them,
        # so they do not take the photo out of passing through.
        state = state.without_faces()
    return (
        state.is_neutral_colour
        and state.temperature == 0.0
        and state.tint == 0.0
        and not has_detail(state)
    )


def render(
    photo: DecodedPhoto,
    state: EditState | None = None,
    geometry: Geometry | None = None,
    dither: bool = False,
    crop: bool = True,
    bits: int = EIGHT_BIT,
    look: Look | None = None,
) -> NDArray[Any]:
    """Turn a decoded photo into what the screen or a file should show.

    The whole of PLAN.md section 6 in one call: the geometry of step 2 and
    then the colour of steps 3 to 12. The preview and the exported file go
    through here so that they cannot drift apart; what differs between them
    is only the size of the photo handed in.

    Developing is the expensive half and the geometry rarely changes, so a
    caller that is only moving a colour slider should hold on to the result
    of :func:`core.geometry.apply` and call :func:`develop` itself, which is
    the caching PLAN.md section 6 asks for.
    """
    return develop(
        geometry_module.apply(photo, geometry, crop=crop),
        state,
        dither=dither,
        bits=bits,
        look=look,
    )


def blur_margin(
    photo: DecodedPhoto,
    state: EditState,
    frame: tuple[int, int],
    surroundings_apart: bool = False,
    scale: float = 1.0,
) -> int:
    """Return how many pixels around a window have to be developed with it.

    Three things in the chain look beyond the pixel itself: the detail step,
    the softened mask of step 4 and the local brightness of step 7. A window
    is grown by the reach of whichever of them is at work, so that what is
    shown is what the whole photo would have given (PLAN.md 3.3).

    With *surroundings_apart* the local brightness has already been worked out
    over the whole frame by :func:`surroundings_field` and will be handed to
    :func:`develop`, so only the far narrower reaches are left.

    *scale* is how many pixels of the window one pixel of the full size photo
    is, which is what the detail step's sizes are measured in; at its
    default of one the margin is never too small.
    """
    long_edge = max(frame)
    sigma = CLIPPED_BLUR_FRACTION * long_edge if photo.is_raw else 0.0
    if not surroundings_apart and (state.highlights != 0.0 or state.shadows != 0.0):
        sigma = max(sigma, LOCAL_BRIGHTNESS_BLUR * long_edge)
    reach = max(detail_reach(state, scale), skin_reach(photo, state, scale))
    if sigma <= 0.0:
        return reach
    return max(reach, int(np.ceil(BLUR_REACH_SIGMAS * max(1.0, sigma))))


def _working_space_step(
    photo: DecodedPhoto, state: EditState
) -> Callable[[NDArray[np.float32], NDArray[np.float32] | None], NDArray[np.float32]]:
    """Return steps 3 to 6 as one function of a band.

    White balance, the clipped highlights, the camera's colour space and the
    exposure. They are what both :func:`develop` and :func:`surroundings_field`
    have to do before step 7 can judge a pixel by its surroundings, so there
    is one of them rather than two that could drift apart.
    """
    gains = _white_balance_gains(photo, state)

    def to_working_space(
        part: NDArray[np.float32], weight_part: NDArray[np.float32] | None
    ) -> NDArray[np.float32]:
        """Steps 3 to 6, on one band."""
        if gains is not None:
            part = apply_gains(part, gains)
        if photo.is_raw:
            part = blend_to_neutral(part, weight_part)
            part = camera_to_working_space(part, photo.camera_to_xyz)
        return exposure(part, state.exposure)

    return to_working_space


def surroundings_key(state: EditState) -> tuple[float, float, float, bool]:
    """Return everything :func:`surroundings_field` depends on in *state*.

    The field is worked out from steps 3 to 6 and from whether step 7 is
    wanted at all, so a kept one survives every slider after it: contrast,
    the vignette, vibrance and the rest can move without the frame's local
    brightness having to be worked out again.
    """
    return (
        state.temperature,
        state.tint,
        state.exposure,
        state.highlights != 0.0 or state.shadows != 0.0,
    )


def surroundings_field(
    photo: DecodedPhoto, state: EditState
) -> NDArray[np.float32] | None:
    """Return how bright each pixel's surroundings are, over the whole photo.

    Step 7 is the one part of the chain that looks far beyond the pixel, and
    it is what would otherwise force a window of a zoomed in photo to be
    developed with a wide margin around it. Worked out here over the whole
    frame instead, it can be handed to :func:`develop` for any window of that
    frame and no margin is needed (PLAN.md 3.3).

    *photo* is the whole frame at whatever resolution is at hand: the blur is
    a hundredth of the long edge, so it carries nothing finer than that and
    the proxy says the same as the full size photo would. ``None`` comes back
    when neither slider of step 7 has been moved and there is nothing to work
    out.
    """
    if state.highlights == 0.0 and state.shadows == 0.0:
        return None
    if photo.is_raw and photo.camera_to_xyz is None:
        raise ValueError("a RAW photo must carry a camera colour matrix")
    weight = clipped_weight(photo.clipped, max(photo.size)) if photo.is_raw else None
    image = in_bands(_working_space_step(photo, state), photo.image, weight)
    return local_brightness(_luminance(image), image.shape)


def develop(
    photo: DecodedPhoto,
    state: EditState | None = None,
    dither: bool = False,
    bits: int = EIGHT_BIT,
    window: Window | None = None,
    surroundings: NDArray[np.float32] | None = None,
    look: Look | None = None,
    grain: NDArray[np.float32] | None = None,
    body: NDArray[np.float32] | None = None,
) -> NDArray[Any]:
    """Turn a decoded photo into 8 bit sRGB, in the order of PLAN.md section 6.

    Steps 2.5 to 12, in the order the plan fixes and never another: noise
    reduction and sharpening, white balance, clipped highlights, colour
    space, exposure, highlights and shadows, contrast, the base tone curve,
    the black and white points, the vignette, the film filter, vibrance and
    saturation with the skin's tone, gamut compression, film grain, output;
    the skin is smoothed right after the detail step (2.6). The screen's own profile is
    applied afterwards, by whoever is showing the result, because it must not
    reach an exported file.

    *bits* is the depth of the result: 8 for the preview and for a JPEG, 16
    for a 16 bit PNG or TIFF (PLAN.md 6 step 12).

    *window* says that *photo* is one window of a larger frame, which is what
    the preview hands over when it is zoomed in past the proxy (PLAN.md 3.3).
    The three things that are about the frame rather than about the pixel -
    the two blurs and the vignette - are then measured against that frame, so
    the window comes out exactly as the whole photo would have.

    *surroundings* is step 7's local brightness, already worked out over the
    whole frame by :func:`surroundings_field` and cut to this window. Given
    it, the step that reaches furthest beyond the pixel no longer does, so a
    window needs next to no margin and the whole chain runs in one pass.

    *look* is the film filter named by ``state.filter``, which the caller
    looks up for itself: the pipeline keeps no catalogue, having no business
    reading files. Naming a filter and not handing one over is an error
    rather than a photo quietly developed without it.

    *grain* is :func:`grain_field` for this photo and window, which depends
    on nothing a slider moves and so can be kept by the caller; without it
    the field is worked out here when the grain slider asks for one.

    *body* is :func:`body_field` of the whole frame, at whatever resolution
    it was worked out, for the whitening slider. It depends on nothing a
    slider moves either, and a window cannot work it out for itself - the
    skin it holds is the skin joined to a face, which may lie outside the
    window - so a caller developing a window hands it over. Without it the
    field is worked out here from *photo*.
    """
    if bits not in (EIGHT_BIT, SIXTEEN_BIT):
        raise ValueError(f"an output depth of {bits} bits is not supported")
    pane = Window.whole(photo.size) if window is None else window
    if pane.size != photo.size:
        raise ValueError(
            f"a window of {pane.size} was given a photo of {photo.size} to develop"
        )
    if surroundings is not None and surroundings.shape[:2] != photo.image.shape[:2]:
        raise ValueError(
            f"surroundings of {surroundings.shape[:2]} were given for a photo of "
            f"{photo.image.shape[:2]}"
        )
    if state is None:
        state = EditState.for_photo(photo.is_raw, *shot_white_balance(photo))
    if is_passthrough(photo, state):
        assert photo.encoded is not None  # guaranteed by is_passthrough
        return photo.encoded if bits == EIGHT_BIT else widen_to_16_bit(photo.encoded)
    if state.filter and look is None:
        raise ValueError(f"the filter {state.filter!r} was asked for but not given")
    # A look is worn only when the state says so, so that handing one over
    # for a photo that is not wearing it cannot quietly change the photo.
    wearing = look if state.filter else None
    chroma = base_chroma(
        photo.is_raw, state.filter_strength if wearing is not None else None
    )
    if photo.is_raw and photo.camera_to_xyz is None:
        raise ValueError("a RAW photo must carry a camera colour matrix")
    if photo.detail is not None and photo.detail != detail_key(state):
        raise ValueError(
            f"a photo sharpened as {photo.detail} was asked to be developed as "
            f"{detail_key(state)}"
        )
    # Step 2.5 looks at the neighbours of every pixel, so it is done over the
    # whole photo before anything is cut into bands - unless the caller has
    # already done it and kept the result.
    photo = with_detail(photo, state)

    # Step 2.6, over the whole photo for the same reason. The mask is kept
    # for step 10, which brightens the same skin; whitening reaches the skin
    # joined to the faces as well.
    # *skin* is the skin smoothed, which step 10 also brightens a little.
    skin: NDArray[np.float32] | None = None
    whitened: NDArray[np.float32] | None = None
    smoothing = has_skin(state, photo)
    whitening = has_whiten(state, photo)
    if smoothing or whitening:
        planes: NDArray[np.float32] | None = None
        if needs_body(state, photo):
            if body is None:
                body = body_field(photo)
            if body is not None:
                planes = pane.take(body).astype(np.float32, copy=False)
        placements = skin_placements(photo, pane)
        found = photo.skin is None or photo.skin.auto
        # The face's own mask, at the photo's resolution, is sharper than
        # the small field around the eyes and the mouth.
        face = faces_module.skin_mask(photo.image, placements) if found else None
        if whitening and planes is not None:
            whitened = _shaped(planes, planes[..., skin_range.FOUND], face)
        if smoothing:
            skin = face if planes is None else _shaped(planes, None, face)
            if skin is not None:
                photo = replace(
                    photo,
                    image=smooth_skin(photo.image, placements, skin, state.skin, photo.is_raw),
                )

    # Besides the detail step, only two things in the chain look at more
    # than one pixel at a time, and both are blurs: the softened mask of
    # step 4 and the local brightness of step 7. They are worked out over the
    # whole photo, and everything else is done a band at a time, which is
    # what keeps the band in cache from one step to the next.
    weight = clipped_weight(photo.clipped, max(pane.frame)) if photo.is_raw else None
    # How far down the frame each row lies is cut into bands with the photo,
    # and how far across each column lies is the same for every band, so the
    # vignette's factor is worked out a band at a time from the two. Both are
    # measured over the whole frame and then cut to the window, or a window
    # would be darkened as if it were a photo of its own.
    vignette_down, vignette_across = (None, None)
    if state.vignette != 0.0:
        down, across = vignette_axes((pane.frame[1], pane.frame[0]))
        vignette_down = down[pane.top : pane.bottom]
        vignette_across = across[pane.left : pane.right]

    field = None
    if has_grain(state):
        field = grain_field(photo, pane) if grain is None else grain
        if field.shape != photo.image.shape[:2]:
            raise ValueError(
                f"grain of {field.shape} was given for a photo of {photo.image.shape[:2]}"
            )

    to_working_space = _working_space_step(photo, state)
    needs_surroundings = state.highlights != 0.0 or state.shadows != 0.0

    def to_output(
        part: NDArray[np.float32],
        local_part: NDArray[np.float32] | None,
        down_part: NDArray[np.float32] | None,
        grain_part: NDArray[np.float32] | None,
        skin_part: NDArray[np.float32] | None,
        whiten_part: NDArray[np.float32] | None,
    ) -> NDArray[Any]:
        """Steps 7 to 12, on one band."""
        if local_part is not None:
            part = _scale_evenly(
                part, region_factor(local_part, state.highlights, state.shadows)
            )
        part = contrast(part, state.contrast)
        if photo.is_raw:
            part = base_tone_curve(part)
        part = black_and_white_points(part, state.blacks, state.whites)
        if down_part is not None:
            assert vignette_across is not None  # the two are set together
            part = _scale_evenly(
                part,
                vignette_from_axes(
                    down_part, vignette_across, state.vignette, state.vignette_midpoint
                ),
            )
        part = film_filter(part, wearing, state.filter_strength)
        part = vibrance_and_saturation(part, state.vibrance, state.saturation, chroma)
        part = skin_tone(part, skin_part, state.skin)
        part = whiten(part, whiten_part, state.whiten)
        part = compress_to_gamut(part)
        encoded = to_srgb(part)
        if grain_part is not None:
            encoded = apply_grain(encoded, grain_part, state.grain)
        if bits == SIXTEEN_BIT:
            return quantise_srgb16(encoded)
        return quantise_srgb8(encoded, dither=dither)


    if not needs_surroundings or surroundings is not None:
        # Nothing between the two halves is left that needs the whole photo,
        # so the band goes all the way from the sensor's numbers to the
        # finished pixels without the photo being written out in between.
        return in_bands(
            lambda part, weight_part, local_part, down_part, grain_part, skin_part, whiten_part: (
                to_output(
                    to_working_space(part, weight_part),
                    local_part,
                    down_part,
                    grain_part,
                    skin_part,
                    whiten_part,
                )
            ),
            photo.image,
            weight,
            surroundings if needs_surroundings else None,
            vignette_down,
            field,
            skin,
            whitened,
        )

    image = in_bands(to_working_space, photo.image, weight)
    # The blurred brightness is worked out over the whole photo; turning it
    # into a factor is per pixel, so that happens inside the band and never
    # has to be held for the whole photo at once.
    local = local_brightness(_luminance(image), (pane.frame[1], pane.frame[0]))
    return in_bands(to_output, image, local, vignette_down, field, skin, whitened)


def _white_balance_gains(
    photo: DecodedPhoto, state: EditState
) -> NDArray[np.float32] | None:
    """Return the gains of step 3, or ``None`` when there is nothing to do."""
    if photo.is_raw:
        if photo.rgb_xyz_matrix is None or photo.camera_whitebalance is None:
            return None
        return raw_white_balance_gains(
            state.temperature,
            state.tint,
            photo.camera_whitebalance,
            photo.rgb_xyz_matrix,
        )
    if state.temperature == 0.0 and state.tint == 0.0:
        return None
    return jpeg_white_balance_gains(state.temperature, state.tint)
