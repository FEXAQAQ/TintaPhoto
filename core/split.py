"""One photo cut into even pieces (PLAN.md 2.12, 3.17).

The finished photo - adjusted, cropped and at the size it is exported - is
cut side by side or one above the other into two to four pieces, or across
and down into four, and every piece is written as a file of its own. A
100 x 40 photo cut one above the other comes out as two of 50 x 40.

The pieces are always even. When the pixels do not divide evenly the later
piece takes the one left over, so the pieces put back together are the photo
again, pixel for pixel.

Everything here is pure: sizes and arrays in, new values out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from numpy.typing import NDArray

#: How the photo is cut.
SPLIT_SIDE_BY_SIDE: Final[str] = "side_by_side"
SPLIT_STACKED: Final[str] = "stacked"
SPLIT_GRID: Final[str] = "grid"
SPLITS: Final[tuple[str, ...]] = (SPLIT_SIDE_BY_SIDE, SPLIT_STACKED, SPLIT_GRID)

#: Into how many pieces, side by side or stacked; a grid is always two by two.
COUNTS: Final[tuple[int, ...]] = (2, 3, 4)
COUNT_DEFAULT: Final[int] = 2
GRID_COUNT: Final[int] = 4
_GRID_SIDE: Final[int] = 2

#: What each piece's file name ends in, in the order of :func:`rects`: the
#: same letters a collage's carousel uses, since a number is what a name
#: already taken is given (X-05).
LETTERS: Final[str] = "abcd"

#: How many names in a row are tried before giving up on finding a free one.
_UNIQUE_LIMIT: Final[int] = 10_000


@dataclass(frozen=True)
class SplitSpec:
    """How a photo is cut: the way, and into how many pieces."""

    split: str = SPLIT_SIDE_BY_SIDE
    count: int = COUNT_DEFAULT

    def sane(self) -> "SplitSpec":
        """Return this spec with every value one that can be cut."""
        split = self.split if self.split in SPLITS else SPLIT_SIDE_BY_SIDE
        if split == SPLIT_GRID:
            return SplitSpec(split, GRID_COUNT)
        try:
            count = int(self.count)
        except (TypeError, ValueError):
            count = COUNT_DEFAULT
        return SplitSpec(split, count if count in COUNTS else COUNT_DEFAULT)

    @property
    def pieces(self) -> int:
        """Return how many pieces the photo is cut into."""
        return self.sane().count


@dataclass(frozen=True)
class Rect:
    """One piece, in pixels of the photo it is cut from."""

    x: int
    y: int
    width: int
    height: int


def _spans(length: int, count: int) -> list[tuple[int, int]]:
    """Return where each of *count* even pieces along *length* starts, and how long it is.

    Each piece ends where the next begins, at the whole pixel below its even
    share, so the later piece takes what is left over.
    """
    edges = [length * index // count for index in range(count + 1)]
    return [(edges[index], edges[index + 1] - edges[index]) for index in range(count)]


def rects(size: tuple[int, int], spec: SplitSpec) -> tuple[Rect, ...]:
    """Return the pieces of a photo *size* = (width, height) wide and high.

    Left to right, top to bottom; a grid's are read like a page: top left,
    top right, bottom left, bottom right.
    """
    spec = spec.sane()
    width, height = size
    if width < 1 or height < 1:
        return ()
    if spec.split == SPLIT_GRID:
        columns = _spans(width, _GRID_SIDE)
        rows = _spans(height, _GRID_SIDE)
        return tuple(Rect(x, y, w, h) for y, h in rows for x, w in columns)
    if spec.split == SPLIT_SIDE_BY_SIDE:
        return tuple(Rect(x, 0, w, height) for x, w in _spans(width, spec.count))
    return tuple(Rect(0, y, width, h) for y, h in _spans(height, spec.count))


def can_cut(size: tuple[int, int], spec: SplitSpec) -> bool:
    """Return whether every piece of a photo of *size* would hold a pixel."""
    return bool(rects(size, spec)) and all(
        rect.width >= 1 and rect.height >= 1 for rect in rects(size, spec)
    )


def piece_size(size: tuple[int, int], spec: SplitSpec) -> tuple[int, int]:
    """Return the size of the first piece, the one the others are at most a pixel bigger than."""
    pieces = rects(size, spec)
    if not pieces:
        return 0, 0
    return pieces[0].width, pieces[0].height


def cut(image: NDArray[Any], spec: SplitSpec) -> tuple[NDArray[Any], ...]:
    """Return each piece of *image* as an array of its own, in the order of :func:`rects`.

    Every pixel is the photo's own: the pieces put side by side are the
    photo again. Each is a copy, so writing one never touches another.
    """
    height, width = image.shape[:2]
    if not can_cut((width, height), spec):
        raise ValueError(f"a photo of {width}x{height} cannot be cut into {spec.pieces}")
    return tuple(
        image[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width].copy()
        for rect in rects((width, height), spec)
    )


def piece_path(base: Path, index: int) -> Path:
    """Return the file the piece at *index* of an export named *base* is written to."""
    return base.with_name(f"{base.stem}_{LETTERS[index]}{base.suffix}")


def piece_paths(base: Path, count: int) -> tuple[Path, ...]:
    """Return the files the *count* pieces of an export named *base* are written to."""
    return tuple(piece_path(base, index) for index in range(count))


def free_base(base: Path, count: int) -> Path:
    """Return *base*, or the first ``_2``, ``_3`` ... beside it whose pieces are all free.

    The pieces keep their letters together: if any of them is taken, the
    name before the letters moves on, so ``X_edit_a`` and ``X_edit_b`` are
    never split across two exports. Nothing is ever overwritten (X-05).
    """
    if not any(path.exists() for path in piece_paths(base, count)):
        return base
    for number in range(2, _UNIQUE_LIMIT):
        candidate = base.with_name(f"{base.stem}_{number}{base.suffix}")
        if not any(path.exists() for path in piece_paths(candidate, count)):
            return candidate
    raise OSError(f"no free name beside {base}")
