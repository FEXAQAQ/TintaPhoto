"""Two to four photos side by side or one above the other (PLAN.md 2.8, 3.13).

A collage is a fixed canvas - one of the sizes the social platforms show
best, from 1080 x 1920 standing up to 1080 x 566 lying down - split across or
down into two
to four pieces, with one photo in each piece. The gaps between the pieces are
taken out of the pieces, never added to the canvas, so the file always comes
out at the size the platform wants.

Each photo is placed by three numbers: how far it is zoomed in past the fit
the collage asks for, and which point of the photo sits in the middle of its
piece. Those are fractions of the photo, not pixels, so the same placement
lands in the same place whether the photo is the small copy the screen shows
or the full size one the export develops. The window draws its preview from
:func:`photo_rect` and the export renders with it, so they cannot disagree.

Linked, the photos are one scene shot several times - by night and by day,
say - and all are laid over the whole canvas with one placement, so that each
piece shows its own part of the scene and the seams line up. Because two
shots never quite match, every photo after the first can be shifted and
scaled a little against the first: its alignment.

Everything here is pure: arrays and values in, new arrays out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from core import color_management as cm
from core import pipeline

# --------------------------------------------------------------------------
# What can be chosen
# --------------------------------------------------------------------------

#: The canvases a collage can have, named by their shape.
CANVAS_9X16: Final[str] = "9x16"
CANVAS_3X4: Final[str] = "3x4"
CANVAS_4X5: Final[str] = "4x5"
CANVAS_1X1: Final[str] = "1x1"
CANVAS_16X9: Final[str] = "16x9"
CANVAS_191X100: Final[str] = "191x100"

#: Each canvas in pixels, from the tallest to the widest.
CANVAS_SIZES: Final[dict[str, tuple[int, int]]] = {
    CANVAS_9X16: (1080, 1920),
    CANVAS_3X4: (1080, 1440),
    CANVAS_4X5: (1080, 1350),
    CANVAS_1X1: (1080, 1080),
    CANVAS_16X9: (1920, 1080),
    CANVAS_191X100: (1080, 566),
}
CANVASES: Final[tuple[str, ...]] = tuple(CANVAS_SIZES)

#: The canvases are offered in two groups, one for each kind of social
#: platform, each in the order the user listed them, the best first
#: (PLAN.md 2.8 K-01). Not named after any platform.
PLATFORM_1: Final[str] = "social_1"
PLATFORM_2: Final[str] = "social_2"
PLATFORMS: Final[tuple[str, ...]] = (PLATFORM_1, PLATFORM_2)
PLATFORM_CANVASES: Final[dict[str, tuple[str, ...]]] = {
    PLATFORM_1: (CANVAS_3X4, CANVAS_4X5, CANVAS_1X1, CANVAS_191X100),
    PLATFORM_2: (CANVAS_4X5, CANVAS_9X16, CANVAS_1X1, CANVAS_16X9),
}
#: A first collage is 16:9, as it was before there was a choice of canvas.
PLATFORM_DEFAULT: Final[str] = PLATFORM_2
CANVAS_DEFAULT: Final[str] = CANVAS_16X9

#: Pieces left to right, or top to bottom.
SPLIT_SIDE_BY_SIDE: Final[str] = "side_by_side"
SPLIT_STACKED: Final[str] = "stacked"
SPLITS: Final[tuple[str, ...]] = (SPLIT_SIDE_BY_SIDE, SPLIT_STACKED)

#: Fill the piece and lose the edges, or show the whole photo and leave a
#: border of the background colour.
FIT_COVER: Final[str] = "cover"
FIT_CONTAIN: Final[str] = "contain"
FITS: Final[tuple[str, ...]] = (FIT_COVER, FIT_CONTAIN)

BACKGROUND_WHITE: Final[str] = "white"
BACKGROUND_BLACK: Final[str] = "black"
BACKGROUNDS: Final[tuple[str, ...]] = (BACKGROUND_WHITE, BACKGROUND_BLACK)

#: The colour of the gap and of whatever a contained photo leaves bare, as
#: 8 bit sRGB. Pure white and pure black are the same in every colour space,
#: so they need no conversion on the way to the file or to the screen.
BACKGROUND_SRGB8: Final[dict[str, tuple[int, int, int]]] = {
    BACKGROUND_WHITE: (255, 255, 255),
    BACKGROUND_BLACK: (0, 0, 0),
}

#: How many photos a collage is split into.
COUNT_MIN: Final[int] = 2
COUNT_MAX: Final[int] = 4
COUNTS: Final[tuple[int, ...]] = tuple(range(COUNT_MIN, COUNT_MAX + 1))

#: The gap between each two pieces, in pixels of the canvas.
GAP_MIN: Final[int] = 0
GAP_MAX: Final[int] = 20

#: How far a photo can be zoomed in past its fit; 1 is the fit itself.
ZOOM_MIN: Final[float] = 1.0
ZOOM_MAX: Final[float] = 4.0

#: Where two pieces meet, as a fraction of the canvas along the split. The
#: pieces are even unless a seam is moved, and never so narrow that one all
#: but goes: two pieces keep the seam between 20 and 80 percent, three or
#: four keep every piece at least a tenth of the canvas.
SEAM_DEFAULT: Final[float] = 0.5
SEAM_MIN: Final[float] = 0.2
SEAM_MAX: Final[float] = 0.8
PIECE_MIN: Final[dict[int, float]] = {2: SEAM_MIN, 3: 0.1, 4: 0.1}

#: How far a linked photo can be moved against the first,
#: as a fraction of its own size, and how far scaled. Enough to make up for
#: two shots taken by hand from the same spot, not to recompose one.
ALIGN_SHIFT_MAX: Final[float] = 0.25
ALIGN_SCALE_MIN: Final[float] = 0.8
ALIGN_SCALE_MAX: Final[float] = 1.25

#: What a collage's file name ends with, after the names of its photos.
#: Not translated: a file name should not depend on the interface language.
FILE_SUFFIX: Final[str] = "_collage"

#: What each photo of a carousel is named, after the collage's own name:
#: letters, because a number is what is added when a name is taken (X-05).
SINGLE_SUFFIXES: Final[tuple[str, ...]] = ("_a", "_b", "_c", "_d")

#: How many photos a collage can hold; the layout's count says how many it uses.
SLOTS: Final[int] = COUNT_MAX

#: How far beyond the visible part of a photo is kept when it is cut out
#: before scaling, in source pixels, so that the filter reaching past the
#: edge of the piece still has real pixels to read.
_CUT_MARGIN: Final[int] = 2

_EIGHT_BIT_MAX: Final[float] = 255.0

#: Halvings when looking for the least zoom that keeps linked photos
#: covered: thirty take the answer far below anything that can be seen.
_ZOOM_SEARCH_STEPS: Final[int] = 30


@dataclass(frozen=True)
class Layout:
    """How the canvas is shaped and split, and what fills the gaps."""

    #: The group of canvases chosen from, and the canvas: always one of the group's.
    platform: str = PLATFORM_DEFAULT
    canvas: str = CANVAS_DEFAULT
    split: str = SPLIT_SIDE_BY_SIDE
    fit: str = FIT_COVER
    gap: int = GAP_MIN
    background: str = BACKGROUND_WHITE
    #: All photos laid over the whole canvas with one placement, each piece
    #: showing its own part of the same scene.
    linked: bool = False
    #: How many pieces the canvas is split into.
    count: int = COUNT_MIN
    #: Where each two pieces meet, one fewer than the pieces, in order; any
    #: other number of them means the pieces are even.
    seams: tuple[float, ...] = (SEAM_DEFAULT,)

    def sane(self) -> "Layout":
        """Return this layout with every value one that can be drawn."""
        count = _sane_count(self.count)
        platform = self.platform if self.platform in PLATFORMS else PLATFORM_DEFAULT
        canvases = PLATFORM_CANVASES[platform]
        if self.canvas in canvases:
            canvas = self.canvas
        else:
            # Unusable, the usual canvas if the group has it, or its best one.
            canvas = CANVAS_DEFAULT if CANVAS_DEFAULT in canvases else canvases[0]
        return Layout(
            platform=platform,
            canvas=canvas,
            split=_one_of(self.split, SPLITS),
            fit=_one_of(self.fit, FITS),
            gap=int(min(GAP_MAX, max(GAP_MIN, int(self.gap)))),
            background=_one_of(self.background, BACKGROUNDS),
            linked=bool(self.linked),
            count=count,
            seams=_sane_seams(self.seams, count),
        )

    @property
    def fit_used(self) -> str:
        """Return the fit the photos are placed with; linked, always cover."""
        return FIT_COVER if self.linked else _one_of(self.fit, FITS)

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the width and height of the finished collage."""
        return CANVAS_SIZES[self.sane().canvas]


def with_platform(layout: Layout, platform: str) -> Layout:
    """Return *layout* offered from *platform*'s canvases.

    The canvas stays when that group has it too, and is otherwise the
    group's first, its best one.
    """
    layout = layout.sane()
    if platform not in PLATFORMS:
        return layout
    canvases = PLATFORM_CANVASES[platform]
    canvas = layout.canvas if layout.canvas in canvases else canvases[0]
    return replace(layout, platform=platform, canvas=canvas)


def even_seams(count: int) -> tuple[float, ...]:
    """Return the seams that split the canvas into *count* even pieces."""
    count = _sane_count(count)
    return tuple((index + 1) / count for index in range(count - 1))


def seam_range(layout: Layout, index: int) -> tuple[float, float]:
    """Return how far seam *index* can move without squeezing a piece too far."""
    layout = layout.sane()
    least = PIECE_MIN[layout.count]
    seams = layout.seams
    low = (seams[index - 1] if index > 0 else 0.0) + least
    high = (seams[index + 1] if index + 1 < len(seams) else 1.0) - least
    return low, max(low, high)


def with_seam(layout: Layout, index: int, position: float) -> Layout:
    """Return *layout* with seam *index* moved towards *position*.

    It stops short of its neighbours, so dragging one seam never pushes
    another along.
    """
    layout = layout.sane()
    if not 0 <= index < len(layout.seams):
        return layout
    low, high = seam_range(layout, index)
    seams = list(layout.seams)
    seams[index] = _clamp(position, low, high)
    return replace(layout, seams=tuple(seams)).sane()


def _sane_count(count: Any) -> int:
    """Return *count* as one of :data:`COUNTS`; anything else is two."""
    try:
        number = int(count)
    except (TypeError, ValueError):
        return COUNT_MIN
    return number if number in COUNTS else COUNT_MIN


def _sane_seams(seams: Any, count: int) -> tuple[float, ...]:
    """Return *seams* in order and in range, or even seams if they are unusable."""
    try:
        values = [float(seam) for seam in seams]
    except (TypeError, ValueError):
        return even_seams(count)
    if len(values) != count - 1 or not all(math.isfinite(value) for value in values):
        return even_seams(count)
    least = PIECE_MIN[count]
    kept: list[float] = []
    for index, value in enumerate(values):
        low = (kept[-1] if kept else 0.0) + least
        high = 1.0 - (count - 1 - index) * least
        kept.append(min(high, max(low, value)))
    return tuple(kept)


@dataclass(frozen=True)
class Cell:
    """One piece of the canvas, in whole pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def centre(self) -> tuple[float, float]:
        """Return the middle of the piece."""
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass(frozen=True)
class Placement:
    """Where one photo sits in its piece.

    ``zoom`` is how far past the fit it is enlarged; ``focus_x`` and
    ``focus_y`` are the point of the photo, as fractions of its width and
    height, that sits in the middle of the piece.
    """

    zoom: float = ZOOM_MIN
    focus_x: float = 0.5
    focus_y: float = 0.5


@dataclass(frozen=True)
class Alignment:
    """How a linked photo after the first is moved against the first.

    ``shift_x`` and ``shift_y`` are fractions of the photo's shown width and
    height; ``scale`` enlarges it about its own centre.
    """

    shift_x: float = 0.0
    shift_y: float = 0.0
    scale: float = 1.0

    def sane(self) -> "Alignment":
        """Return this alignment inside the range it may take."""
        return Alignment(
            shift_x=_clamp(self.shift_x, -ALIGN_SHIFT_MAX, ALIGN_SHIFT_MAX),
            shift_y=_clamp(self.shift_y, -ALIGN_SHIFT_MAX, ALIGN_SHIFT_MAX),
            scale=_clamp(self.scale, ALIGN_SCALE_MIN, ALIGN_SCALE_MAX),
        )


#: The alignment that leaves a photo where its placement put it.
UNALIGNED: Final[Alignment] = Alignment()


def cells(layout: Layout) -> tuple[Cell, ...]:
    """Return the pieces of the canvas, first left or top.

    Each two meet at a seam, and the gap is taken out of them either side of
    it. When the gap is odd the later piece gets the pixel left over, which
    is the only way to keep the canvas its size.
    """
    layout = layout.sane()
    width, height = layout.canvas_size
    gap = layout.gap
    length = width if layout.split == SPLIT_SIDE_BY_SIDE else height
    spans: list[tuple[int, int]] = []
    start = 0
    for index, seam in enumerate(layout.seams):
        later = layout.count - 1 - index
        end = math.floor(length * seam - gap / 2.0)
        end = max(start + 1, min(length - later * (gap + 1), end))
        spans.append((start, end - start))
        start = end + gap
    spans.append((start, length - start))
    if layout.split == SPLIT_SIDE_BY_SIDE:
        return tuple(Cell(at, 0, size, height) for at, size in spans)
    return tuple(Cell(0, at, width, size) for at, size in spans)


def seam_at(layout: Layout, index: int) -> float:
    """Return the middle of the gap at seam *index*, in canvas pixels along the split."""
    layout = layout.sane()
    pieces = cells(layout)
    before, after = pieces[index], pieces[index + 1]
    if layout.split == SPLIT_SIDE_BY_SIDE:
        return (before.x + before.width + after.x) / 2.0
    return (before.y + before.height + after.y) / 2.0


def canvas_cell(layout: Layout) -> Cell:
    """Return the whole canvas as one cell."""
    width, height = layout.sane().canvas_size
    return Cell(0, 0, width, height)


def frame(layout: Layout, slot: int) -> Cell:
    """Return what a photo is placed against: its piece, or linked, the canvas."""
    layout = layout.sane()
    return canvas_cell(layout) if layout.linked else cells(layout)[slot]


def aligned(
    rect: tuple[float, float, float, float], alignment: Alignment
) -> tuple[float, float, float, float]:
    """Return *rect* moved and scaled by *alignment*, about its own centre."""
    alignment = alignment.sane()
    left, top, width, height = rect
    centre_x = left + width / 2.0 + alignment.shift_x * width
    centre_y = top + height / 2.0 + alignment.shift_y * height
    width *= alignment.scale
    height *= alignment.scale
    return centre_x - width / 2.0, centre_y - height / 2.0, width, height


def placed_rect(
    layout: Layout,
    slot: int,
    photo_size: tuple[int, int],
    placement: Placement,
    alignment: Alignment = UNALIGNED,
) -> tuple[float, float, float, float]:
    """Return where the photo in *slot* lands on the canvas.

    The one place the preview and the export both ask, so they agree.
    *alignment* only counts when the layout is linked.
    """
    layout = layout.sane()
    rect = photo_rect(placement, photo_size, frame(layout, slot), layout.fit_used)
    return aligned(rect, alignment) if layout.linked else rect


def linked_placement(
    layout: Layout,
    placement: Placement,
    sizes: Sequence[tuple[int, int] | None],
    alignments: Sequence[Alignment] = (),
) -> Placement:
    """Return the shared placement of linked photos, with no background showing.

    The first photo must still cover the canvas, and every later one, moved
    and scaled by its alignment (``alignments[slot]``; the first's is not
    used), must still cover its own piece. Where they cannot all hold at the
    zoom asked for, they are zoomed in together just as far as it takes -
    aligning a photo never leaves a bare strip down one side - and the focus
    is then kept where all hold, as near as can be to the one asked for.
    """
    layout = layout.sane()
    canvas = canvas_cell(layout)
    pieces = cells(layout)
    present = (list(sizes) + [None] * layout.count)[: layout.count]
    reference = next((size for size in present if size is not None), None)
    if reference is None:
        return placement
    base = settled(placement, reference, canvas, FIT_COVER)
    first = present[0]
    if first is None:
        return base
    padded = (list(alignments) + [UNALIGNED] * layout.count)[: layout.count]
    others = [
        (size, padded[slot].sane(), pieces[slot])
        for slot, size in enumerate(present)
        if slot > 0 and size is not None
    ]
    if not others:
        return base

    def bounds(zoom: float, axis: int) -> tuple[float, float]:
        room = (canvas.width, canvas.height)[axis]
        shown_first = first[axis] * fitted_scale(first, canvas, FIT_COVER) * zoom
        margin = room / 2.0 / shown_first
        low, high = margin, 1.0 - margin
        for size, alignment, piece in others:
            shown = size[axis] * fitted_scale(size, canvas, FIT_COVER) * zoom
            shift = (alignment.shift_x, alignment.shift_y)[axis]
            start = (piece.x, piece.y)[axis]
            end = start + (piece.width, piece.height)[axis]
            low = max(low, 0.5 + shift - alignment.scale / 2.0 + (room / 2.0 - start) / shown)
            high = min(high, 0.5 + shift + alignment.scale / 2.0 - (end - room / 2.0) / shown)
        return low, high

    def holds(zoom: float) -> bool:
        return all(low <= high for low, high in (bounds(zoom, 0), bounds(zoom, 1)))

    zoom = base.zoom
    if not holds(zoom):
        if not holds(ZOOM_MAX):
            return base
        low, high = zoom, ZOOM_MAX
        for _ in range(_ZOOM_SEARCH_STEPS):
            middle = (low + high) / 2.0
            if holds(middle):
                high = middle
            else:
                low = middle
        zoom = high
    (low_x, high_x), (low_y, high_y) = bounds(zoom, 0), bounds(zoom, 1)
    return Placement(
        zoom=zoom,
        focus_x=min(high_x, max(low_x, base.focus_x)),
        focus_y=min(high_y, max(low_y, base.focus_y)),
    )


def nudged(
    alignment: Alignment, delta: tuple[float, float], rect: tuple[float, float, float, float]
) -> Alignment:
    """Return the alignment after its photo is moved by *delta* canvas pixels.

    *rect* is where the placement alone puts the photo, before aligning.
    """
    alignment = alignment.sane()
    _left, _top, width, height = rect
    shown_w = width * alignment.scale
    shown_h = height * alignment.scale
    return replace(
        alignment,
        shift_x=alignment.shift_x + (delta[0] / width if shown_w > 0 else 0.0),
        shift_y=alignment.shift_y + (delta[1] / height if shown_h > 0 else 0.0),
    ).sane()


def rescaled(
    alignment: Alignment,
    scale: float,
    anchor: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> Alignment:
    """Return the alignment at *scale*, keeping the point under *anchor* still."""
    alignment = alignment.sane()
    scale = _clamp(scale, ALIGN_SCALE_MIN, ALIGN_SCALE_MAX)
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return replace(alignment, scale=scale)
    now_left, now_top, now_w, now_h = aligned(rect, alignment)
    under_x = (anchor[0] - now_left) / now_w
    under_y = (anchor[1] - now_top) / now_h
    new_w = width * scale
    new_h = height * scale
    centre_x = anchor[0] - under_x * new_w + new_w / 2.0
    centre_y = anchor[1] - under_y * new_h + new_h / 2.0
    return Alignment(
        shift_x=(centre_x - (left + width / 2.0)) / width,
        shift_y=(centre_y - (top + height / 2.0)) / height,
        scale=scale,
    ).sane()


def fitted_scale(photo_size: tuple[int, int], cell: Cell, fit: str) -> float:
    """Return how much the photo is scaled to fill or to fit inside *cell*."""
    width, height = photo_size
    if width <= 0 or height <= 0:
        return 1.0
    across = cell.width / width
    down = cell.height / height
    return max(across, down) if fit == FIT_COVER else min(across, down)


def settled(
    placement: Placement, photo_size: tuple[int, int], cell: Cell, fit: str
) -> Placement:
    """Return *placement* with the zoom in range and the photo kept in place.

    A photo larger than its piece along one direction may slide only as far
    as its edge: it never uncovers background it could have covered. A photo
    smaller than its piece along a direction is centred along it.
    """
    zoom = float(min(ZOOM_MAX, max(ZOOM_MIN, placement.zoom)))
    scale = fitted_scale(photo_size, cell, fit) * zoom
    focus_x = _settled_focus(placement.focus_x, photo_size[0] * scale, cell.width)
    focus_y = _settled_focus(placement.focus_y, photo_size[1] * scale, cell.height)
    return Placement(zoom=zoom, focus_x=focus_x, focus_y=focus_y)


def _settled_focus(focus: float, shown: float, room: float) -> float:
    """Return a focus that keeps a photo *shown* long inside *room*."""
    if shown <= room or shown <= 0.0:
        return 0.5
    margin = room / 2.0 / shown
    return float(min(1.0 - margin, max(margin, focus)))


def photo_rect(
    placement: Placement, photo_size: tuple[int, int], cell: Cell, fit: str
) -> tuple[float, float, float, float]:
    """Return where the whole photo lands on the canvas: x, y, width, height.

    Most of it may lie outside the piece; whatever does is cut away.
    """
    placement = settled(placement, photo_size, cell, fit)
    scale = fitted_scale(photo_size, cell, fit) * placement.zoom
    width = photo_size[0] * scale
    height = photo_size[1] * scale
    centre_x, centre_y = cell.centre
    return (
        centre_x - placement.focus_x * width,
        centre_y - placement.focus_y * height,
        width,
        height,
    )


def panned(
    placement: Placement,
    delta: tuple[float, float],
    photo_size: tuple[int, int],
    cell: Cell,
    fit: str,
) -> Placement:
    """Return the placement after the photo is dragged by *delta* canvas pixels."""
    _x, _y, width, height = photo_rect(placement, photo_size, cell, fit)
    current = settled(placement, photo_size, cell, fit)
    moved = replace(
        current,
        focus_x=current.focus_x - (delta[0] / width if width > 0 else 0.0),
        focus_y=current.focus_y - (delta[1] / height if height > 0 else 0.0),
    )
    return settled(moved, photo_size, cell, fit)


def zoomed(
    placement: Placement,
    zoom: float,
    anchor: tuple[float, float],
    photo_size: tuple[int, int],
    cell: Cell,
    fit: str,
) -> Placement:
    """Return the placement at *zoom*, keeping the point under *anchor* still.

    *anchor* is in canvas pixels; it is where the pointer was when the wheel
    turned, as in the preview (PLAN.md 3.3).
    """
    left, top, width, height = photo_rect(placement, photo_size, cell, fit)
    zoom = float(min(ZOOM_MAX, max(ZOOM_MIN, zoom)))
    if width <= 0 or height <= 0:
        return settled(replace(placement, zoom=zoom), photo_size, cell, fit)
    # The point of the photo under the anchor, as fractions of the photo.
    under_x = (anchor[0] - left) / width
    under_y = (anchor[1] - top) / height
    scale = fitted_scale(photo_size, cell, fit) * zoom
    new_width = photo_size[0] * scale
    new_height = photo_size[1] * scale
    new_left = anchor[0] - under_x * new_width
    new_top = anchor[1] - under_y * new_height
    centre_x, centre_y = cell.centre
    moved = Placement(
        zoom=zoom,
        focus_x=(centre_x - new_left) / new_width,
        focus_y=(centre_y - new_top) / new_height,
    )
    return settled(moved, photo_size, cell, fit)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def background_level(background: str) -> float:
    """Return the background as linear light, the same in every channel."""
    return BACKGROUND_SRGB8[_one_of(background, BACKGROUNDS)][0] / _EIGHT_BIT_MAX


def render_cell(
    image: NDArray[Any],
    cell: Cell,
    rect: tuple[float, float, float, float],
    background: str,
) -> NDArray[np.float32]:
    """Return one piece of the canvas as linear light, *image* laid at *rect*.

    *rect* is where the whole photo lands on the canvas, from
    :func:`placed_rect`; only the part inside *cell* is drawn.

    *image* is a developed photo in sRGB, 8 or 16 bit, at any size: what
    matters is its shape, which says where it goes. Only the part that shows
    is cut out and turned into floating point, so a full size photo costs
    little more memory than the piece it fills.

    Scaling happens in linear light, where averaging pixels averages light.
    A photo shrunk is first reduced with area averaging, which is what keeps
    fine detail from turning into moire; whatever scale is left over, and
    the fraction of a pixel the placement asks for, is done by one affine
    warp whose border is the background, so the photo's edge is smooth.
    """
    level = background_level(background)
    height, width = image.shape[:2]
    left, top, shown_w, shown_h = rect
    scale = shown_w / width
    if scale <= 0.0 or shown_h <= 0.0:
        return np.full((cell.height, cell.width, 3), level, dtype=np.float32)

    # The part of the source that lands inside the piece, with a margin.
    x0 = max(0, math.floor((cell.x - left) / scale) - _CUT_MARGIN)
    y0 = max(0, math.floor((cell.y - top) / scale) - _CUT_MARGIN)
    x1 = min(width, math.ceil((cell.x + cell.width - left) / scale) + _CUT_MARGIN)
    y1 = min(height, math.ceil((cell.y + cell.height - top) / scale) + _CUT_MARGIN)
    if x1 <= x0 or y1 <= y0:
        return np.full((cell.height, cell.width, 3), level, dtype=np.float32)

    cut = _to_linear(image[y0:y1, x0:x1])
    factor_x = factor_y = 1.0
    if scale < 1.0:
        size = (
            max(1, round((x1 - x0) * scale)),
            max(1, round((y1 - y0) * scale)),
        )
        factor_x = size[0] / (x1 - x0)
        factor_y = size[1] / (y1 - y0)
        cut = cv2.resize(cut, size, interpolation=cv2.INTER_AREA)

    # Pixel centres sit at half steps: a source pixel c of the cut lies at
    # x0 + (c + 0.5) / factor in the photo, which lands at left + scale * that
    # on the canvas, and the piece's own pixel i has its centre at i + 0.5.
    step_x = scale / factor_x
    step_y = scale / factor_y
    matrix = np.array(
        [
            [step_x, 0.0, left - cell.x + scale * x0 + 0.5 * step_x - 0.5],
            [0.0, step_y, top - cell.y + scale * y0 + 0.5 * step_y - 0.5],
        ],
        dtype=np.float64,
    )
    # The warp reads past the cut by repeating its edge, and the photo's own
    # edge is drawn below from exactly how much of each pixel it covers, so
    # an enlarged photo does not smear into the background over several
    # pixels and a photo filling its piece does not fade at the sides.
    placed = cv2.warpAffine(
        cut,
        matrix,
        (cell.width, cell.height),
        flags=cv2.INTER_LINEAR if scale < 1.0 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Bicubic enlarging can ring just past black and white.
    np.clip(placed, 0.0, 1.0, out=placed)
    across = _coverage(cell.width, left - cell.x, shown_w)
    down = _coverage(cell.height, top - cell.y, shown_h)
    covered = (down[:, None] * across[None, :])[..., None]
    if np.all(covered == 1.0):
        return placed
    return (placed * covered + np.float32(level) * (1.0 - covered)).astype(np.float32)


def _coverage(count: int, start: float, length: float) -> NDArray[np.float32]:
    """Return how much of each of *count* pixels the span from *start* covers."""
    edges = np.arange(count, dtype=np.float64)
    covered = np.minimum(edges + 1.0, start + length) - np.maximum(edges, start)
    return np.clip(covered, 0.0, 1.0).astype(np.float32)


def compose(layout: Layout, halves: Sequence[NDArray[np.float32] | None]) -> NDArray[np.float32]:
    """Return the canvas in linear light, each piece laid where it belongs.

    A piece with nothing in it is left the background colour, and so are the
    gaps. Pieces past the layout's count are not used.
    """
    layout = layout.sane()
    width, height = layout.canvas_size
    canvas = np.full((height, width, 3), background_level(layout.background), dtype=np.float32)
    for cell, half in zip(cells(layout), halves):
        if half is None:
            continue
        if half.shape[:2] != (cell.height, cell.width):
            raise ValueError(
                f"a piece of {half.shape[1]}x{half.shape[0]} was given for one of "
                f"{cell.width}x{cell.height}"
            )
        canvas[cell.y : cell.y + cell.height, cell.x : cell.x + cell.width] = half
    return canvas


def encode(
    canvas: NDArray[np.float32], bits: int = pipeline.EIGHT_BIT, dither: bool = False
) -> NDArray[Any]:
    """Return the linear canvas as sRGB samples of *bits* bits (PLAN.md 6 step 12)."""
    encoded = cm.linear_to_srgb(np.clip(canvas, 0.0, 1.0))
    if bits == pipeline.SIXTEEN_BIT:
        return pipeline.quantise_srgb16(encoded)
    return pipeline.quantise_srgb8(encoded, dither)


def render_slot(
    layout: Layout,
    slot: int,
    image: NDArray[Any],
    placement: Placement,
    alignment: Alignment = UNALIGNED,
) -> NDArray[np.float32]:
    """Return the piece *slot* fills, *image* placed as the layout says."""
    layout = layout.sane()
    size = (int(image.shape[1]), int(image.shape[0]))
    rect = placed_rect(layout, slot, size, placement, alignment)
    return render_cell(image, cells(layout)[slot], rect, layout.background)


def pieces(layout: Layout, canvas: NDArray[Any]) -> tuple[NDArray[Any], ...]:
    """Return each piece cut out of the finished *canvas*, first left or top.

    The single photos of a carousel (PLAN.md 2.8 K-13) are the collage cut
    apart at its seams: every pixel is the collage's own, so the pieces put
    side by side are the collage again, less the gaps.
    """
    layout = layout.sane()
    width, height = layout.canvas_size
    if canvas.shape[:2] != (height, width):
        raise ValueError(
            f"a canvas of {canvas.shape[1]}x{canvas.shape[0]} was given for one of "
            f"{width}x{height}"
        )
    return tuple(
        canvas[cell.y : cell.y + cell.height, cell.x : cell.x + cell.width].copy()
        for cell in cells(layout)
    )


def render(
    layout: Layout,
    images: Sequence[NDArray[Any] | None],
    placements: Sequence[Placement],
    alignments: Sequence[Alignment] | None = None,
    bits: int = pipeline.EIGHT_BIT,
    dither: bool = False,
) -> NDArray[Any]:
    """Return the finished collage: developed photos in, sRGB samples out.

    Linked, all photos are meant to be given the same placement. Only as
    many photos as the layout's count are used.
    """
    layout = layout.sane()
    alignments = alignments or [UNALIGNED] * len(placements)
    halves = [
        None if image is None else render_slot(layout, slot, image, placement, alignment)
        for slot, (image, placement, alignment) in enumerate(
            zip(images[: layout.count], placements, alignments)
        )
    ]
    return encode(compose(layout, halves), bits, dither)


def _to_linear(image: NDArray[Any]) -> NDArray[np.float32]:
    """Return sRGB samples of either depth as linear light."""
    full_scale = np.iinfo(image.dtype).max if image.dtype.kind == "u" else 1.0
    encoded = image.astype(np.float32) / np.float32(full_scale)
    return cm.srgb_to_linear(encoded)


def _clamp(value: float, low: float, high: float) -> float:
    """Return *value* within *low* and *high*; anything unusable is the middle."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = (low + high) / 2.0
    if not math.isfinite(number):
        number = (low + high) / 2.0
    return min(high, max(low, number))


def _one_of(value: str, choices: tuple[str, ...]) -> str:
    """Return *value* if it is one of *choices*, otherwise the first of them."""
    return value if value in choices else choices[0]
