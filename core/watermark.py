"""A line of text laid over an exported photo (PLAN.md 2.9, section 6).

The watermark is not part of the pipeline: it is laid over the finished,
encoded pixels as the last thing before they are written, so steps 1 to 13
are untouched and a photo exported without one is the same file as before.

Everything is measured against the frame the watermark lands on - the photo
as it is written, or as it is shown - so that it sits in the same place and
at the same proportion at any size:

* its height (the height of one line of text) is a share of the short edge;
* where it sits is one of nine anchors, the corners, edge middles and centre,
  a margin in from the edges, plus an offset from there in short edges, which
  is what a drag leaves behind (W-07, W-08).

The text is drawn by FreeType through Pillow at the size it will be used, so
it is sharp at any size; the blend is done on the sRGB encoded values, where
an opacity reads the way it does in any drawing program, and only the pixels
under the watermark are quantised again, so every other pixel is exactly
what it would have been without it.

Fonts: Inter, bundled, for Latin text; for text with Chinese in it the whole
line is set in the system's Microsoft JhengHei, which has no italic, so its
italic is the upright face slanted. Only reading the font files touches the
disk (:func:`load_fonts`, and the font cache); everything else is pure.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from core import pipeline

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# What a watermark can be (PLAN.md 2.9)
# --------------------------------------------------------------------------

#: The only kind so far; the field is there so that an image watermark can be
#: added later without changing the stored format.
KIND_TEXT: Final[str] = "text"
KINDS: Final[tuple[str, ...]] = (KIND_TEXT,)

COLOUR_WHITE: Final[str] = "white"
COLOUR_BLACK: Final[str] = "black"
COLOURS: Final[tuple[str, ...]] = (COLOUR_WHITE, COLOUR_BLACK)
#: The encoded value each colour is drawn in.
_COLOUR_LEVELS: Final[dict[str, float]] = {COLOUR_WHITE: 1.0, COLOUR_BLACK: 0.0}

ANCHOR_TOP_LEFT: Final[str] = "top-left"
ANCHOR_TOP: Final[str] = "top"
ANCHOR_TOP_RIGHT: Final[str] = "top-right"
ANCHOR_LEFT: Final[str] = "left"
ANCHOR_CENTRE: Final[str] = "centre"
ANCHOR_RIGHT: Final[str] = "right"
ANCHOR_BOTTOM_LEFT: Final[str] = "bottom-left"
ANCHOR_BOTTOM: Final[str] = "bottom"
ANCHOR_BOTTOM_RIGHT: Final[str] = "bottom-right"
#: The nine anchors, row by row from the top left, as the picker lays them out.
ANCHORS: Final[tuple[str, ...]] = (
    ANCHOR_TOP_LEFT,
    ANCHOR_TOP,
    ANCHOR_TOP_RIGHT,
    ANCHOR_LEFT,
    ANCHOR_CENTRE,
    ANCHOR_RIGHT,
    ANCHOR_BOTTOM_LEFT,
    ANCHOR_BOTTOM,
    ANCHOR_BOTTOM_RIGHT,
)
ANCHOR_DEFAULT: Final[str] = ANCHOR_BOTTOM_RIGHT
#: Each anchor's column and row: 0 the left or top, 1 the middle, 2 the right
#: or bottom.
_CELLS: Final[dict[str, tuple[int, int]]] = {
    anchor: (index % 3, index // 3) for index, anchor in enumerate(ANCHORS)
}
_ANCHOR_AT: Final[dict[tuple[int, int], str]] = {cell: name for name, cell in _CELLS.items()}
_MIDDLE: Final[int] = 1
_FAR: Final[int] = 2

TEXT_MAX_LENGTH: Final[int] = 100

#: The height of one line, in per cent of the short edge (W-05).
SIZE_MIN: Final[float] = 1.0
SIZE_MAX: Final[float] = 20.0
SIZE_DEFAULT: Final[float] = 3.0
SIZE_DECIMALS: Final[int] = 1

#: Per cent (W-06).
OPACITY_MIN: Final[int] = 10
OPACITY_MAX: Final[int] = 100
OPACITY_DEFAULT: Final[int] = 70

#: How far in from the edges, in per cent of the short edge (W-07).
MARGIN_MIN: Final[int] = 0
MARGIN_MAX: Final[int] = 20
MARGIN_DEFAULT: Final[int] = 3

#: How far a drag may leave the watermark from its anchor, in short edges.
OFFSET_LIMIT: Final[float] = 1.0

#: A line wider than this share of the photo is set smaller until it fits.
WIDTH_LIMIT: Final[float] = 0.9

#: How far an italic with no italic face of its own leans, in degrees.
SLANT_DEGREES: Final[float] = 12.0
_SLANT: Final[float] = math.tan(math.radians(SLANT_DEGREES))

#: How close, in screen pixels, a dragged watermark has to come to an
#: anchor's line to be pulled onto it (W-07).
SNAP_DISTANCE: Final[float] = 8.0

#: From this code point on (the CJK radicals onwards) a character is taken to
#: be one the bundled Latin font does not have, and the line is set in the
#: system's Chinese font instead.
CJK_START: Final[int] = 0x2E80

#: The bundled faces, by (bold, italic).
FONTS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "resources" / "fonts"
_LATIN_FILES: Final[dict[tuple[bool, bool], str]] = {
    (False, False): "Inter-Regular.ttf",
    (True, False): "Inter-SemiBold.ttf",
    (False, True): "Inter-Italic.ttf",
    (True, True): "Inter-SemiBoldItalic.ttf",
}
#: Microsoft JhengHei, which comes with Windows, by bold.
_CJK_FILES: Final[dict[bool, str]] = {False: "msjh.ttc", True: "msjhbd.ttc"}

#: The size fonts are measured at to learn their line height.
_MEASURE_SIZE: Final[int] = 100
#: A line that still comes out too wide is set again at most this many times.
_FIT_ATTEMPTS: Final[int] = 3
#: Empty pixels around the drawn text, so that antialiasing is never cut off.
_PAD: Final[int] = 1


@dataclass(frozen=True)
class WatermarkSpec:
    """Everything that decides what the watermark looks like and where it is."""

    kind: str = KIND_TEXT
    text: str = ""
    bold: bool = False
    italic: bool = False
    colour: str = COLOUR_WHITE
    opacity: int = OPACITY_DEFAULT
    #: The height of one line, in per cent of the short edge.
    size: float = SIZE_DEFAULT
    anchor: str = ANCHOR_DEFAULT
    #: How far from the anchor a drag left it, in short edges, across and down.
    offset: tuple[float, float] = (0.0, 0.0)
    #: How far in from the edges, in per cent of the short edge.
    margin: int = MARGIN_DEFAULT

    def sane(self) -> "WatermarkSpec":
        """Return this watermark with every value inside its range."""
        across, down = self.offset
        return replace(
            self,
            kind=self.kind if self.kind in KINDS else KIND_TEXT,
            text=" ".join(str(self.text).splitlines())[:TEXT_MAX_LENGTH],
            bold=bool(self.bold),
            italic=bool(self.italic),
            colour=self.colour if self.colour in COLOURS else COLOUR_WHITE,
            opacity=int(_clamp(round(_number(self.opacity, OPACITY_DEFAULT)), OPACITY_MIN, OPACITY_MAX)),
            size=round(_clamp(_number(self.size, SIZE_DEFAULT), SIZE_MIN, SIZE_MAX), SIZE_DECIMALS),
            anchor=self.anchor if self.anchor in ANCHORS else ANCHOR_DEFAULT,
            offset=(
                _clamp(_number(across, 0.0), -OFFSET_LIMIT, OFFSET_LIMIT),
                _clamp(_number(down, 0.0), -OFFSET_LIMIT, OFFSET_LIMIT),
            ),
            margin=int(_clamp(round(_number(self.margin, MARGIN_DEFAULT)), MARGIN_MIN, MARGIN_MAX)),
        )

    @property
    def is_empty(self) -> bool:
        """Return whether there is nothing to draw: no text."""
        return self.kind != KIND_TEXT or not self.text.strip()

    def with_anchor(self, anchor: str) -> "WatermarkSpec":
        """Return the watermark put on *anchor* exactly, as the picker does."""
        return replace(self, anchor=anchor, offset=(0.0, 0.0)).sane()

    def with_style_reset(self) -> "WatermarkSpec":
        """Return the watermark with its text kept and everything else default."""
        return WatermarkSpec(kind=self.kind, text=self.text)

    def to_dict(self) -> dict[str, Any]:
        """Return the watermark as JSON compatible data (PLAN.md 7.3)."""
        return {
            "kind": self.kind,
            "text": self.text,
            "bold": self.bold,
            "italic": self.italic,
            "colour": self.colour,
            "opacity": self.opacity,
            "size": self.size,
            "anchor": self.anchor,
            "offset": [self.offset[0], self.offset[1]],
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "WatermarkSpec":
        """Build a watermark from stored data; anything unusable is the default."""
        if not isinstance(data, dict):
            return cls()
        default = cls()
        offset = data.get("offset")
        if not (isinstance(offset, (list, tuple)) and len(offset) == 2):
            offset = default.offset
        text = data.get("text", "")
        spec = cls(
            kind=str(data.get("kind", KIND_TEXT)),
            text=text if isinstance(text, str) else "",
            bold=data.get("bold") is True,
            italic=data.get("italic") is True,
            colour=str(data.get("colour", default.colour)),
            opacity=data.get("opacity", default.opacity),
            size=data.get("size", default.size),
            anchor=str(data.get("anchor", default.anchor)),
            offset=(offset[0], offset[1]),
            margin=data.get("margin", default.margin),
        )
        if spec.kind not in KINDS:
            # A kind from a later version: nothing this one can draw.
            return replace(cls(), kind=spec.kind)
        return spec.sane()


def _number(value: Any, default: float) -> float:
    """Return *value* as a finite number, or *default*."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fonts:
    """The font files there are to draw with; any of them may be missing."""

    regular: Path | None = None
    bold: Path | None = None
    italic: Path | None = None
    bold_italic: Path | None = None
    cjk_regular: Path | None = None
    cjk_bold: Path | None = None

    @property
    def has_cjk(self) -> bool:
        """Return whether text with Chinese in it can be drawn."""
        return self.cjk_regular is not None or self.cjk_bold is not None


@dataclass(frozen=True)
class FontChoice:
    """The file a line is set in, and whether it has to be slanted by hand."""

    path: Path | None
    slant: bool
    #: The line has characters the chosen font does not have: there is no
    #: Chinese font on this computer.
    missing_glyphs: bool = False


def load_fonts(fonts_dir: Path = FONTS_DIR, system_dir: Path | None = None) -> Fonts:
    """Find the bundled faces and the system's Chinese font."""
    if system_dir is None:
        system_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

    def found(path: Path) -> Path | None:
        return path if path.is_file() else None

    return Fonts(
        regular=found(fonts_dir / _LATIN_FILES[(False, False)]),
        bold=found(fonts_dir / _LATIN_FILES[(True, False)]),
        italic=found(fonts_dir / _LATIN_FILES[(False, True)]),
        bold_italic=found(fonts_dir / _LATIN_FILES[(True, True)]),
        cjk_regular=found(system_dir / _CJK_FILES[False]),
        cjk_bold=found(system_dir / _CJK_FILES[True]),
    )


def needs_cjk(text: str) -> bool:
    """Return whether *text* has characters only the Chinese font has."""
    return any(ord(character) >= CJK_START for character in text)


def choose_font(spec: WatermarkSpec, fonts: Fonts) -> FontChoice:
    """Return the face *spec*'s text is set in (W-04)."""
    if needs_cjk(spec.text):
        if fonts.has_cjk:
            path = (fonts.cjk_bold if spec.bold else fonts.cjk_regular) or (
                fonts.cjk_regular or fonts.cjk_bold
            )
            return FontChoice(path, slant=spec.italic)
        latin = _latin(spec, fonts)
        return replace(latin, missing_glyphs=True)
    return _latin(spec, fonts)


def _latin(spec: WatermarkSpec, fonts: Fonts) -> FontChoice:
    """Return the Inter face for *spec*, slanting the upright one if need be."""
    upright = fonts.bold if spec.bold else fonts.regular
    upright = upright or fonts.regular or fonts.bold
    if not spec.italic:
        return FontChoice(upright, slant=False)
    italic = fonts.bold_italic if spec.bold else fonts.italic
    if italic is not None:
        return FontChoice(italic, slant=False)
    return FontChoice(upright, slant=True)


@lru_cache(maxsize=64)
def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return *path* at *size* pixels; Pillow's own font when there is no file."""
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size, index=0)
        except OSError:
            logger.warning("The font %s could not be read", path)
    return ImageFont.load_default(size)


@lru_cache(maxsize=16)
def _line_ratio(path: Path | None) -> float:
    """Return one line's height per pixel of font size, for *path*."""
    font = _font(path, _MEASURE_SIZE)
    ascent, descent = _metrics(font)
    return max(ascent + descent, 1) / _MEASURE_SIZE


def _metrics(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
    getmetrics = getattr(font, "getmetrics", None)
    if getmetrics is None:
        return _MEASURE_SIZE, 0
    ascent, descent = getmetrics()
    return int(ascent), int(descent)


# --------------------------------------------------------------------------
# Drawing (PLAN.md section 6, "浮水印")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sprite:
    """The watermark drawn, wherever it is going to go.

    ``coverage`` is how much of each pixel the watermark covers, opacity
    included, from 0 to 1; ``level`` is the encoded value it is drawn in.
    """

    coverage: NDArray[np.float32]
    level: float
    #: There was Chinese text and no Chinese font to draw it with.
    missing_glyphs: bool = False

    @property
    def size(self) -> tuple[int, int]:
        """Return the width and height in pixels."""
        height, width = self.coverage.shape[:2]
        return int(width), int(height)


def line_height(spec: WatermarkSpec, frame: tuple[int, int]) -> float:
    """Return the height of one line on *frame*, in its pixels (W-05)."""
    return min(frame) * spec.size / 100.0


def rasterise(
    spec: WatermarkSpec, frame: tuple[int, int], fonts: Fonts
) -> Sprite | None:
    """Draw *spec* at the size it has on *frame*, or return ``None`` for nothing.

    A line wider than WIDTH_LIMIT of the frame is set smaller until it fits.
    """
    spec = spec.sane()
    if spec.is_empty or min(frame) <= 0:
        return None
    choice = choose_font(spec, fonts)
    target = line_height(spec, frame)
    size = max(1, round(target / _line_ratio(choice.path)))
    coverage = _draw(spec.text, _font(choice.path, size), choice.slant)
    widest = frame[0] * WIDTH_LIMIT
    for _attempt in range(_FIT_ATTEMPTS):
        if coverage.shape[1] <= widest or size <= 1:
            break
        size = max(1, math.floor(size * widest / coverage.shape[1]))
        coverage = _draw(spec.text, _font(choice.path, size), choice.slant)
    coverage *= np.float32(spec.opacity / 100.0)
    return Sprite(
        coverage=coverage,
        level=_COLOUR_LEVELS[spec.colour],
        missing_glyphs=choice.missing_glyphs,
    )


def _draw(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, slant: bool
) -> NDArray[np.float32]:
    """Return how much of each pixel *text* covers, in a box just around it.

    The box is the line's own height - ascent to descent - and its advance,
    grown to take in any ink that reaches beyond them, so that a line is the
    same height whatever letters it has.
    """
    ascent, descent = _metrics(font)
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    advance = font.getlength(text)
    x0 = min(0.0, float(left))
    x1 = max(float(advance), float(right))
    y0 = min(-float(ascent), float(top))
    y1 = max(float(descent), float(bottom))
    width = math.ceil(x1 - x0) + 2 * _PAD
    height = math.ceil(y1 - y0) + 2 * _PAD
    canvas = Image.new("L", (max(width, 1), max(height, 1)), 0)
    ImageDraw.Draw(canvas).text(
        (_PAD - x0, _PAD - y0), text, font=font, fill=255, anchor="ls"
    )
    coverage = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    if slant:
        coverage = _slanted(coverage)
    return coverage


def _slanted(coverage: NDArray[np.float32]) -> NDArray[np.float32]:
    """Lean *coverage* right by SLANT_DEGREES, its bottom row staying put."""
    height, width = coverage.shape
    lean = _SLANT * height
    # x' = x + slant * (height - y): the top moves furthest right.
    matrix = np.float32([[1.0, -_SLANT, lean], [0.0, 1.0, 0.0]])
    slanted = cv2.warpAffine(
        coverage,
        matrix,
        (width + math.ceil(lean), height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(slanted, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Where it goes (W-07, W-08)
# --------------------------------------------------------------------------


def _lines(
    spec: WatermarkSpec, frame: tuple[int, int], box: tuple[float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the left edges of the three columns and the tops of the three rows."""
    width, height = frame
    box_width, box_height = box
    margin = min(frame) * spec.margin / 100.0
    columns = (margin, (width - box_width) / 2.0, width - margin - box_width)
    rows = (margin, (height - box_height) / 2.0, height - margin - box_height)
    return columns, rows


def layout(
    spec: WatermarkSpec, frame: tuple[int, int], box: tuple[float, float]
) -> tuple[float, float]:
    """Return where the top left of a *box* sized watermark goes on *frame*.

    Always inside the frame: a watermark is never cut off by the edge.
    """
    spec = spec.sane()
    columns, rows = _lines(spec, frame, box)
    column, row = _CELLS[spec.anchor]
    short = min(frame)
    x = columns[column] + spec.offset[0] * short
    y = rows[row] + spec.offset[1] * short
    return _inside(x, frame[0], box[0]), _inside(y, frame[1], box[1])


def _inside(position: float, length: float, extent: float) -> float:
    """Return *position* moved so that *extent* from it stays within *length*."""
    return _clamp(position, 0.0, max(0.0, length - extent))


@dataclass(frozen=True)
class Drop:
    """Where a dragged watermark was left, and which lines it was pulled onto."""

    spec: WatermarkSpec
    #: Aligned with its column: the left margin, the middle or the right margin.
    snapped_across: bool
    #: Aligned with its row: the top margin, the middle or the bottom margin.
    snapped_down: bool


def from_drop(
    spec: WatermarkSpec,
    frame: tuple[int, int],
    box: tuple[float, float],
    top_left: tuple[float, float],
    snap: float,
) -> Drop:
    """Return the watermark moved so that its top left is at *top_left*.

    It is kept as the nearest anchor plus an offset in short edges (W-08),
    and within *snap* pixels of an anchor's line on either axis it is pulled
    onto that line, its offset on that axis cleared (W-07).
    """
    spec = spec.sane()
    columns, rows = _lines(spec, frame, box)
    x = _inside(top_left[0], frame[0], box[0])
    y = _inside(top_left[1], frame[1], box[1])
    column, across, snapped_across = _nearest(x, columns, snap)
    row, down, snapped_down = _nearest(y, rows, snap)
    short = max(min(frame), 1)
    moved = replace(
        spec, anchor=_ANCHOR_AT[(column, row)], offset=(across / short, down / short)
    ).sane()
    return Drop(moved, snapped_across, snapped_down)


def _nearest(
    position: float, lines: tuple[float, float, float], snap: float
) -> tuple[int, float, bool]:
    """Return the nearest line's index, how far from it, and whether it pulls."""
    index = min(range(len(lines)), key=lambda i: abs(position - lines[i]))
    distance = position - lines[index]
    if abs(distance) <= snap:
        return index, 0.0, True
    return index, distance, False


def guides(
    spec: WatermarkSpec, frame: tuple[int, int], box: tuple[float, float]
) -> tuple[float | None, float | None]:
    """Return the vertical and horizontal line the watermark is aligned with.

    Each is a position on *frame* - the margin line or the middle the
    watermark's edge or centre sits on - or ``None`` when it is off its line
    on that axis. These are the guides drawn while it is dragged (W-07).
    """
    spec = spec.sane()
    column, row = _CELLS[spec.anchor]
    margin = min(frame) * spec.margin / 100.0

    def line(cell: int, length: float, offset: float) -> float | None:
        if offset != 0.0:
            return None
        if cell == _MIDDLE:
            return length / 2.0
        return length - margin if cell == _FAR else margin

    return line(column, frame[0], spec.offset[0]), line(row, frame[1], spec.offset[1])


def anchor_of(spec: WatermarkSpec) -> str | None:
    """Return the anchor the picker marks, or ``None`` once dragged off it."""
    spec = spec.sane()
    return spec.anchor if spec.offset == (0.0, 0.0) else None


# --------------------------------------------------------------------------
# Laying it over a photo
# --------------------------------------------------------------------------


def compose(
    image: NDArray[Any],
    spec: WatermarkSpec,
    sprite: Sprite,
    dither: bool = True,
) -> NDArray[Any]:
    """Return *image* with *sprite* laid over it where *spec* puts it.

    *image* is sRGB encoded, as 8 or 16 bit samples or floats in 0 to 1, and
    the result is the same. The blend is done on the encoded values, and only
    the pixels under the watermark are touched: an 8 bit result is quantised
    there again with *dither* (PLAN.md 6 step 12), everything else is
    returned exactly as it was. *image* itself is not changed.
    """
    height, width = image.shape[:2]
    box = sprite.size
    x, y = layout(spec, (width, height), box)
    left, top = int(round(x)), int(round(y))
    right, bottom = min(width, left + box[0]), min(height, top + box[1])
    result = image.copy()
    if right <= left or bottom <= top:
        return result
    cover = sprite.coverage[: bottom - top, : right - left]
    if image.ndim == 3:
        cover = cover[:, :, np.newaxis]
    region = image[top:bottom, left:right]
    if image.dtype == np.uint8:
        encoded = region.astype(np.float32) / np.float32(255.0)
    elif image.dtype == np.uint16:
        encoded = region.astype(np.float32) / np.float32(65535.0)
    else:
        encoded = region.astype(np.float32)
    blended = encoded * (np.float32(1.0) - cover) + np.float32(sprite.level) * cover
    blended = np.clip(blended, 0.0, 1.0)
    if image.dtype == np.uint8:
        result[top:bottom, left:right] = pipeline.quantise_srgb8(blended, dither)
    elif image.dtype == np.uint16:
        result[top:bottom, left:right] = pipeline.quantise_srgb16(blended)
    else:
        result[top:bottom, left:right] = blended.astype(image.dtype)
    return result


def apply(
    image: NDArray[Any],
    spec: WatermarkSpec | None,
    fonts: Fonts | None = None,
    dither: bool = True,
) -> NDArray[Any]:
    """Return *image* with *spec* drawn at its size and laid over it.

    With no watermark, or nothing to draw, *image* itself is returned.
    """
    if spec is None or spec.is_empty:
        return image
    height, width = image.shape[:2]
    sprite = rasterise(spec, (width, height), fonts if fonts is not None else load_fonts())
    if sprite is None:
        return image
    return compose(image, spec, sprite, dither)
