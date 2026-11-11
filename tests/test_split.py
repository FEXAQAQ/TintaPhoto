"""Tests for cutting one photo into even pieces (PLAN.md 2.12).

The pieces must be the photo itself, cut: put back together they are the
same pixels, and the original file is only ever read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image

from core import exporter, metadata, split
from core.edit_state import EditState
from core.exporter import ExportOptions
from core.geometry import Geometry, Rect
from core.split import SplitSpec
from core.watermark import WatermarkSpec


def _fingerprint(path: Path) -> tuple[str, int, int]:
    status = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), status.st_size, status.st_mtime_ns


def _write_png(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Write an 8 bit PNG of random pixels, which passes through untouched."""
    pixels = np.random.default_rng(7).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, "PNG")
    return pixels


def _write_jpeg(path: Path, size: tuple[int, int]) -> Path:
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
    pixels[:, :, 1] = 120
    exif = piexif.dump(
        {"0th": {piexif.ImageIFD.Make: "TINTA"}, "Exif": {piexif.ExifIFD.ISOSpeedRatings: 400}}
    )
    Image.fromarray(pixels).save(path, "JPEG", quality=95, exif=exif)
    return path


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


def test_a_grid_is_always_four() -> None:
    assert SplitSpec(split.SPLIT_GRID, 2).sane() == SplitSpec(split.SPLIT_GRID, 4)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (SplitSpec("diagonal", 3), SplitSpec(split.SPLIT_SIDE_BY_SIDE, 3)),
        (SplitSpec(split.SPLIT_STACKED, 7), SplitSpec(split.SPLIT_STACKED, 2)),
        (SplitSpec(split.SPLIT_STACKED, "x"), SplitSpec(split.SPLIT_STACKED, 2)),  # type: ignore[arg-type]
    ],
)
def test_a_spec_that_cannot_be_cut_is_made_one_that_can(
    spec: SplitSpec, expected: SplitSpec
) -> None:
    assert spec.sane() == expected


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------


def test_stacked_halves_of_a_tall_photo() -> None:
    """The example the feature was asked with: 40 wide, 100 high, cut top and bottom."""
    pieces = split.rects((40, 100), SplitSpec(split.SPLIT_STACKED, 2))
    assert [(r.width, r.height) for r in pieces] == [(40, 50), (40, 50)]
    assert [(r.x, r.y) for r in pieces] == [(0, 0), (0, 50)]


def test_the_later_piece_takes_the_pixel_left_over() -> None:
    pieces = split.rects((40, 101), SplitSpec(split.SPLIT_STACKED, 2))
    assert [r.height for r in pieces] == [50, 51]
    pieces = split.rects((100, 10), SplitSpec(split.SPLIT_SIDE_BY_SIDE, 3))
    assert [r.width for r in pieces] == [33, 33, 34]


def test_a_grid_is_read_like_a_page() -> None:
    pieces = split.rects((11, 7), SplitSpec(split.SPLIT_GRID))
    assert [(r.x, r.y, r.width, r.height) for r in pieces] == [
        (0, 0, 5, 3),
        (5, 0, 6, 3),
        (0, 3, 5, 4),
        (5, 3, 6, 4),
    ]


@pytest.mark.parametrize("spec", [
    SplitSpec(split.SPLIT_SIDE_BY_SIDE, 2),
    SplitSpec(split.SPLIT_SIDE_BY_SIDE, 3),
    SplitSpec(split.SPLIT_STACKED, 4),
    SplitSpec(split.SPLIT_GRID),
])
def test_the_pieces_put_together_are_the_photo(spec: SplitSpec) -> None:
    image = np.random.default_rng(1).integers(0, 65536, (37, 53, 3), dtype=np.uint16)
    pieces = split.cut(image, spec)
    rebuilt = np.zeros_like(image)
    for piece, rect in zip(pieces, split.rects((53, 37), spec), strict=True):
        rebuilt[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] = piece
    assert np.array_equal(rebuilt, image)


def test_each_piece_is_a_copy() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    first = split.cut(image, SplitSpec())[0]
    first[:] = 255
    assert not image.any()


def test_a_photo_too_small_for_its_pieces_is_refused() -> None:
    assert not split.can_cut((3, 10), SplitSpec(split.SPLIT_SIDE_BY_SIDE, 4))
    with pytest.raises(ValueError):
        split.cut(np.zeros((10, 3, 3), dtype=np.uint8), SplitSpec(split.SPLIT_SIDE_BY_SIDE, 4))


def test_the_piece_size_is_the_first_piece() -> None:
    assert split.piece_size((4000, 3000), SplitSpec(split.SPLIT_GRID)) == (2000, 1500)
    assert split.piece_size((0, 0), SplitSpec()) == (0, 0)


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def test_each_piece_is_named_by_its_letter(tmp_path: Path) -> None:
    base = tmp_path / "IMG_0001_edit.jpg"
    assert [path.name for path in split.piece_paths(base, 3)] == [
        "IMG_0001_edit_a.jpg",
        "IMG_0001_edit_b.jpg",
        "IMG_0001_edit_c.jpg",
    ]


def test_a_taken_piece_moves_the_whole_set_on(tmp_path: Path) -> None:
    base = tmp_path / "X_edit.jpg"
    assert split.free_base(base, 2) == base
    (tmp_path / "X_edit_b.jpg").write_bytes(b"taken")
    assert split.free_base(base, 2) == tmp_path / "X_edit_2.jpg"
    # A taken piece past the count asked for is no one's business.
    assert split.free_base(tmp_path / "Y.jpg", 2) == tmp_path / "Y.jpg"


# --------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------


def test_an_untouched_png_comes_out_as_its_own_pixels_cut(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    pixels = _write_png(source, (40, 100))
    before = _fingerprint(source)

    result = exporter.export_split(
        source,
        tmp_path / "photo_edit.png",
        SplitSpec(split.SPLIT_STACKED, 2),
        options=ExportOptions(format=exporter.FORMAT_PNG),
    )

    assert [path.name for path in result.pieces] == ["photo_edit_a.png", "photo_edit_b.png"]
    assert result.path == result.pieces[0]
    assert not result.copied
    for path, expected in zip(result.pieces, (pixels[:50], pixels[50:]), strict=True):
        with Image.open(path) as written:
            assert written.size == (40, 50)
            assert np.array_equal(np.asarray(written.convert("RGB")), expected)
    assert _fingerprint(source) == before


def test_an_untouched_jpeg_is_cut_rather_than_copied(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", (64, 48))
    before = _fingerprint(source)
    result = exporter.export_split(
        source, tmp_path / "photo_edit.jpg", SplitSpec(split.SPLIT_GRID), EditState()
    )
    assert len(result.pieces) == 4
    for path in result.pieces:
        with Image.open(path) as written:
            assert written.size == (32, 24)
    assert _fingerprint(source) == before


def test_the_crop_and_the_long_edge_are_applied_before_cutting(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    _write_png(source, (400, 200))
    frame = Geometry(crop=Rect(0.0, 0.0, 0.5, 1.0))
    result = exporter.export_split(
        source,
        tmp_path / "out.png",
        SplitSpec(split.SPLIT_SIDE_BY_SIDE, 2),
        geometry=frame,
        options=ExportOptions(format=exporter.FORMAT_PNG, long_edge=100),
    )
    for path in result.pieces:
        with Image.open(path) as written:
            assert written.size == (50, 100)


def test_every_piece_keeps_the_details_of_the_shot(tmp_path: Path) -> None:
    source = _write_jpeg(tmp_path / "photo.jpg", (64, 48))
    result = exporter.export_split(
        source, tmp_path / "out.jpg", SplitSpec(split.SPLIT_SIDE_BY_SIDE, 2), EditState()
    )
    for path in result.pieces:
        info = metadata.read(path)
        assert info is not None and info.iso == 400


def test_every_piece_carries_a_watermark_of_its_own(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    pixels = _write_png(source, (400, 300))
    mark = WatermarkSpec(text="© 2026", size=10.0, opacity=100)
    result = exporter.export_split(
        source,
        tmp_path / "out.png",
        SplitSpec(split.SPLIT_SIDE_BY_SIDE, 2),
        options=ExportOptions(format=exporter.FORMAT_PNG, watermark=mark),
    )
    for path, expected in zip(result.pieces, (pixels[:, :200], pixels[:, 200:]), strict=True):
        with Image.open(path) as written:
            assert not np.array_equal(np.asarray(written.convert("RGB")), expected)


def test_a_taken_name_never_overwrites_anything(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    _write_png(source, (20, 20))
    taken = tmp_path / "out_a.png"
    taken.write_bytes(b"mine")
    result = exporter.export_split(
        source,
        tmp_path / "out.png",
        SplitSpec(),
        options=ExportOptions(format=exporter.FORMAT_PNG),
    )
    assert [path.name for path in result.pieces] == ["out_2_a.png", "out_2_b.png"]
    assert taken.read_bytes() == b"mine"


def test_a_failed_export_leaves_no_half_set_behind(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    _write_png(source, (3, 10))
    with pytest.raises(exporter.ExportError):
        exporter.export_split(
            source,
            tmp_path / "out.png",
            SplitSpec(split.SPLIT_SIDE_BY_SIDE, 4),
            options=ExportOptions(format=exporter.FORMAT_PNG),
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo.png"]
