"""Tests for the collage of two to four photos (PLAN.md 2.8, 3.13)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core import collage
from core.collage import Cell, Layout, Placement

LANDSCAPE = collage.CANVAS_16X9
PORTRAIT = collage.CANVAS_4X5
SIDE = collage.SPLIT_SIDE_BY_SIDE
STACKED = collage.SPLIT_STACKED


def _solid(colour: tuple[int, int, int], size: tuple[int, int] = (300, 200)) -> np.ndarray:
    width, height = size
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = colour
    return image


def _gradient(size: tuple[int, int]) -> np.ndarray:
    """A smooth 8 bit picture whose every region is different."""
    width, height = size
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    image = np.stack(
        [xs + 0 * ys, ys + 0 * xs, 0.5 * (xs + ys)], axis=-1
    )
    return np.round(image * 255).astype(np.uint8)


# -- the canvas and its halves (requirement section 4) -----------------------


@pytest.mark.parametrize(
    ("orientation", "split", "half"),
    [
        (LANDSCAPE, SIDE, (960, 1080)),
        (LANDSCAPE, STACKED, (1920, 540)),
        (PORTRAIT, SIDE, (540, 1350)),
        (PORTRAIT, STACKED, (1080, 675)),
    ],
)
def test_without_a_gap_each_half_is_the_size_asked_for(
    orientation: str, split: str, half: tuple[int, int]
) -> None:
    layout = Layout(canvas=orientation, split=split)
    first, second = collage.cells(layout)
    assert (first.width, first.height) == half
    assert (second.width, second.height) == half
    assert layout.canvas_size == collage.CANVAS_SIZES[orientation]


@pytest.mark.parametrize("gap", range(collage.GAP_MIN, collage.GAP_MAX + 1))
@pytest.mark.parametrize("split", [SIDE, STACKED])
@pytest.mark.parametrize("orientation", [LANDSCAPE, PORTRAIT])
def test_the_gap_comes_out_of_the_halves_and_never_grows_the_canvas(
    orientation: str, split: str, gap: int
) -> None:
    layout = Layout(canvas=orientation, split=split, gap=gap)
    width, height = layout.canvas_size
    first, second = collage.cells(layout)
    if split == SIDE:
        assert first.width + gap + second.width == width
        assert second.x == first.x + first.width + gap
        assert first.height == second.height == height
    else:
        assert first.height + gap + second.height == height
        assert second.y == first.y + first.height + gap
        assert first.width == second.width == width
    # An odd gap leaves one pixel over, which goes to the second half.
    assert 0 <= (second.width + second.height) - (first.width + first.height) <= 1


def test_values_out_of_range_are_brought_back() -> None:
    layout = Layout(canvas="sideways", split="diagonal", fit="stretch", gap=99, background="red")
    sane = layout.sane()
    assert sane == Layout(gap=collage.GAP_MAX)
    assert Layout(gap=-5).sane().gap == collage.GAP_MIN


# -- placing a photo -----------------------------------------------------------

CELL = Cell(0, 0, 960, 1080)


@pytest.mark.parametrize("photo", [(6000, 4000), (4000, 6000), (1000, 1000)])
def test_cover_fills_the_half_and_contain_shows_all_of_the_photo(
    photo: tuple[int, int],
) -> None:
    x, y, width, height = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    assert x <= 1e-6 and y <= 1e-6
    assert x + width >= CELL.width - 1e-6 and y + height >= CELL.height - 1e-6
    assert min(width - CELL.width, height - CELL.height) == pytest.approx(0.0, abs=1e-6)

    x, y, width, height = collage.photo_rect(Placement(), photo, CELL, collage.FIT_CONTAIN)
    assert x >= -1e-6 and y >= -1e-6
    assert x + width <= CELL.width + 1e-6 and y + height <= CELL.height + 1e-6
    assert width / height == pytest.approx(photo[0] / photo[1])


def test_a_covering_photo_never_slides_far_enough_to_uncover_the_background() -> None:
    photo = (6000, 4000)
    placement = collage.panned(Placement(), (10_000.0, -10_000.0), photo, CELL, collage.FIT_COVER)
    x, y, width, height = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y <= 1e-6 and y + height >= CELL.height - 1e-6


def test_a_photo_smaller_than_its_half_stays_centred() -> None:
    photo = (6000, 4000)
    placement = collage.panned(Placement(), (300.0, 300.0), photo, CELL, collage.FIT_CONTAIN)
    x, y, width, height = collage.photo_rect(placement, photo, CELL, collage.FIT_CONTAIN)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y + height / 2 == pytest.approx(CELL.height / 2)


def test_panning_moves_the_photo_by_as_much_as_the_pointer() -> None:
    photo = (6000, 4000)
    before = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    placement = collage.panned(Placement(), (100.0, 0.0), photo, CELL, collage.FIT_COVER)
    after = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    assert after[0] - before[0] == pytest.approx(100.0)


def test_zooming_keeps_the_point_under_the_pointer_where_it_was() -> None:
    photo = (6000, 4000)
    anchor = (700.0, 300.0)
    before = collage.photo_rect(Placement(), photo, CELL, collage.FIT_COVER)
    placement = collage.zoomed(Placement(), 2.0, anchor, photo, CELL, collage.FIT_COVER)
    after = collage.photo_rect(placement, photo, CELL, collage.FIT_COVER)
    for axis in (0, 1):
        fraction_before = (anchor[axis] - before[axis]) / before[axis + 2]
        fraction_after = (anchor[axis] - after[axis]) / after[axis + 2]
        assert fraction_after == pytest.approx(fraction_before)
    assert placement.zoom == 2.0


def test_the_zoom_stays_in_range() -> None:
    photo = (6000, 4000)
    assert collage.zoomed(Placement(), 99.0, (0, 0), photo, CELL, collage.FIT_COVER).zoom == (
        collage.ZOOM_MAX
    )
    assert collage.zoomed(Placement(), 0.1, (0, 0), photo, CELL, collage.FIT_COVER).zoom == (
        collage.ZOOM_MIN
    )


# -- rendering -----------------------------------------------------------------


@pytest.mark.parametrize("background", [collage.BACKGROUND_WHITE, collage.BACKGROUND_BLACK])
def test_each_photo_lands_in_its_own_half_and_the_gap_is_the_background(
    background: str,
) -> None:
    layout = Layout(gap=10, background=background)
    red, blue = _solid((200, 30, 30)), _solid((30, 30, 200))
    result = collage.render(layout, [red, blue], [Placement(), Placement()])
    assert result.shape == (1080, 1920, 3) and result.dtype == np.uint8
    first, second = collage.cells(layout)
    assert np.all(result[:, first.x + 2 : first.x + first.width - 2] == (200, 30, 30))
    assert np.all(result[:, second.x + 2 : second.x + second.width - 2] == (30, 30, 200))
    gap = result[:, first.x + first.width : second.x]
    assert gap.shape[1] == 10
    assert np.all(gap == collage.BACKGROUND_SRGB8[background])


def test_an_empty_half_is_left_the_background_colour() -> None:
    layout = Layout(split=STACKED, background=collage.BACKGROUND_BLACK)
    result = collage.render(layout, [_solid((90, 90, 90)), None], [Placement(), Placement()])
    first, second = collage.cells(layout)
    assert np.all(result[second.y + 1 :] == 0)
    assert np.all(result[: first.height - 1] == 90)


def test_contain_leaves_bars_of_the_background_beside_the_photo() -> None:
    layout = Layout(fit=collage.FIT_CONTAIN, background=collage.BACKGROUND_WHITE)
    landscape_photo = _solid((10, 120, 10), (600, 400))
    result = collage.render(layout, [landscape_photo, None], [Placement(), Placement()])
    first = collage.cells(layout)[0]
    # 960 wide makes the photo 640 tall; the rest of 1080 is white.
    bar = (first.height - 640) // 2
    assert np.all(result[: bar - 1, : first.width] == 255)
    assert np.all(result[first.height - bar + 1 :, : first.width] == 255)
    assert np.all(result[bar + 2 : first.height - bar - 2, 2 : first.width - 2] == (10, 120, 10))


def test_the_same_placement_gives_the_same_collage_from_any_size_of_photo() -> None:
    """The preview works on a small copy and the export on the full photo."""
    layout = Layout(canvas=PORTRAIT, split=SIDE)
    placement = Placement(zoom=1.7, focus_x=0.3, focus_y=0.6)
    small = _gradient((900, 600))
    large = _gradient((3600, 2400))
    from_small = collage.render(layout, [small, None], [placement, Placement()])
    from_large = collage.render(layout, [large, None], [placement, Placement()])
    difference = np.abs(from_small.astype(int) - from_large.astype(int))
    assert difference.mean() < 0.6
    assert difference.max() <= 3


def test_sixteen_bit_output_keeps_its_depth() -> None:
    layout = Layout()
    deep = (_gradient((800, 600)).astype(np.uint16) * 257)
    result = collage.render(layout, [deep, deep], [Placement(), Placement()], bits=16)
    assert result.dtype == np.uint16 and result.shape == (1080, 1920, 3)


def test_scaling_happens_in_linear_light() -> None:
    """A fine black and white pattern shrinks to the grey of half the light."""
    pattern = np.zeros((2160, 1920, 3), dtype=np.uint8)
    pattern[::2] = 255
    layout = Layout(split=SIDE)
    result = collage.render(layout, [pattern, None], [Placement(), Placement()])
    first = collage.cells(layout)[0]
    middle = result[first.height // 2, first.width // 2]
    # Half the light is sRGB 188, where averaging the encoded values gives 128.
    assert abs(int(middle[0]) - 188) <= 2


def test_compose_refuses_a_half_of_the_wrong_size() -> None:
    with pytest.raises(ValueError):
        collage.compose(Layout(), [np.zeros((10, 10, 3), dtype=np.float32), None])


def test_rendering_does_not_change_the_photos_given_to_it() -> None:
    photo = _gradient((400, 300))
    kept = photo.copy()
    collage.render(Layout(), [photo, photo], [Placement(zoom=2.0), Placement()])
    assert np.array_equal(photo, kept)


@pytest.mark.parametrize(
    ("format_", "bits", "suffix"),
    [("jpg", 8, ".jpg"), ("png", 8, ".png"), ("png", 16, ".png"), ("tiff", 16, ".tif")],
)
def test_a_finished_collage_is_written_in_every_format(
    tmp_path, format_: str, bits: int, suffix: str
) -> None:
    from core import exporter
    from core.exporter import ExportOptions

    options = ExportOptions(format=format_, bits=bits)
    layout = Layout(canvas=PORTRAIT)
    canvas = collage.render(
        layout, [_gradient((300, 200)), None], [Placement(), Placement()], bits=options.depth
    )
    result = exporter.export_pixels(canvas, tmp_path / "collage.jpg", options)
    assert result.path.suffix == options.extension and result.path.exists()
    assert not result.copied
    if format_ == "tiff":
        import tifffile

        assert tifffile.imread(result.path).shape == (1350, 1080, 3)
    else:
        import cv2

        decoded = cv2.imdecode(np.fromfile(result.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded.shape == (1350, 1080, 3)
        assert decoded.dtype == (np.uint16 if bits == 16 and format_ == "png" else np.uint8)


# -- the seam (PLAN.md 2.8 K-10) ----------------------------------------------


@pytest.mark.parametrize("seam", [0.2, 0.37, 0.5, 0.63, 0.8])
@pytest.mark.parametrize("gap", [0, 1, 7, 20])
def test_the_seam_moves_the_halves_but_never_the_canvas(seam: float, gap: int) -> None:
    layout = Layout(seams=(seam,), gap=gap)
    first, second = collage.cells(layout)
    assert first.width + gap + second.width == 1920
    middle_of_gap = first.width + gap / 2
    assert abs(middle_of_gap - 1920 * seam) <= 1.0


def test_the_seam_stays_in_range() -> None:
    assert Layout(seams=(0.0,)).sane().seams == (collage.SEAM_MIN,)
    assert Layout(seams=(5.0,)).sane().seams == (collage.SEAM_MAX,)
    assert Layout(seams=(float("nan"),)).sane().seams == (collage.SEAM_DEFAULT,)


# -- linked: one scene shot twice ------------------------------------------------


def test_linked_both_photos_are_laid_over_the_whole_canvas() -> None:
    layout = Layout(linked=True, fit=collage.FIT_CONTAIN)
    photo = (2000, 1333)
    placement = Placement(zoom=1.3, focus_x=0.45, focus_y=0.55)
    first = collage.placed_rect(layout, 0, photo, placement)
    second = collage.placed_rect(layout, 1, photo, placement)
    assert first == second
    # Linked always covers, and covers the canvas rather than a half.
    x, y, width, height = collage.placed_rect(layout, 0, photo, Placement())
    assert x <= 0 and y <= 0 and x + width >= 1920 - 1e-6 and y + height >= 1080 - 1e-6


def test_linked_each_half_shows_its_own_part_of_the_scene() -> None:
    """The left half of photo A beside the right half of photo B."""
    a = _gradient((1600, 900))
    b = (255 - _gradient((1600, 900))).astype(np.uint8)
    unlinked_a = collage.render(Layout(), [a, None], [Placement(), Placement()])
    linked = collage.render(Layout(linked=True), [a, b], [Placement(), Placement()])
    whole_a = collage.render(
        Layout(linked=True, seams=(collage.SEAM_MAX,)), [a, None], [Placement(), Placement()]
    )
    whole_b = collage.render(
        Layout(linked=True, seams=(collage.SEAM_MIN,)), [None, b], [Placement(), Placement()]
    )
    left, right = collage.cells(Layout(linked=True))
    assert np.abs(linked[:, 10:900].astype(int) - whole_a[:, 10:900].astype(int)).max() <= 1
    assert np.abs(linked[:, 1000:1900].astype(int) - whole_b[:, 1000:1900].astype(int)).max() <= 1
    # Not what the unlinked collage shows, which centres A in its half.
    assert np.abs(linked[:, 10:900].astype(int) - unlinked_a[:, 10:900].astype(int)).mean() > 10


def test_alignment_only_counts_when_linked() -> None:
    photo = (2000, 1333)
    shifted = collage.Alignment(shift_x=0.1, scale=1.1)
    plain = collage.placed_rect(Layout(), 1, photo, Placement(), shifted)
    assert plain == collage.placed_rect(Layout(), 1, photo, Placement())
    linked = collage.placed_rect(Layout(linked=True), 1, photo, Placement(), shifted)
    base = collage.placed_rect(Layout(linked=True), 1, photo, Placement())
    assert linked[2] == pytest.approx(base[2] * 1.1)
    centre = linked[0] + linked[2] / 2
    assert centre == pytest.approx(base[0] + base[2] / 2 + 0.1 * base[2])


def test_nudging_moves_the_aligned_photo_by_as_much_as_asked() -> None:
    rect = (0.0, 0.0, 2000.0, 1200.0)
    moved = collage.nudged(collage.UNALIGNED, (1.0, -3.0), rect)
    after = collage.aligned(rect, moved)
    assert after[0] == pytest.approx(1.0)
    assert after[1] == pytest.approx(-3.0)


def test_rescaling_keeps_the_point_under_the_pointer_still() -> None:
    rect = (-40.0, -60.0, 2000.0, 1200.0)
    start = collage.Alignment(shift_x=0.02, shift_y=-0.01, scale=1.0)
    anchor = (700.0, 300.0)
    before = collage.aligned(rect, start)
    result = collage.rescaled(start, 1.05, anchor, rect)
    after = collage.aligned(rect, result)
    for axis in (0, 1):
        assert (anchor[axis] - after[axis]) / after[axis + 2] == pytest.approx(
            (anchor[axis] - before[axis]) / before[axis + 2]
        )
    assert result.scale == pytest.approx(1.05)


def test_alignment_stays_in_range() -> None:
    wild = collage.Alignment(shift_x=3.0, shift_y=-3.0, scale=9.0).sane()
    assert wild == collage.Alignment(
        shift_x=collage.ALIGN_SHIFT_MAX, shift_y=-collage.ALIGN_SHIFT_MAX, scale=collage.ALIGN_SCALE_MAX
    )


def test_aligning_never_leaves_a_bare_strip() -> None:
    """Shrinking or moving the second photo zooms the pair in just enough."""
    layout = Layout(linked=True)
    size = (2000, 1333)
    alignment = collage.Alignment(shift_x=-0.03, shift_y=0.01, scale=0.95)
    placement = collage.linked_placement(
        layout, Placement(), [size, size], [collage.UNALIGNED, alignment]
    )
    assert placement.zoom > 1.0
    left, top, width, height = collage.placed_rect(layout, 1, size, placement, alignment)
    half = collage.cells(layout)[1]
    assert left <= half.x + 1e-6 and top <= 1e-6
    assert left + width >= half.x + half.width - 1e-6 and top + height >= 1080 - 1e-6
    first = collage.placed_rect(layout, 0, size, placement)
    assert first[0] <= 1e-6 and first[0] + first[2] >= 1920 - 1e-6
    # Just enough: a hair less zoom would show the background again.
    less = replace(placement, zoom=placement.zoom * 0.99)
    assert not _covers(collage.placed_rect(layout, 1, size, less, alignment), half)

    black = np.zeros((1333, 2000, 3), dtype=np.uint8)
    result = collage.render(
        Layout(linked=True, background=collage.BACKGROUND_WHITE),
        [black, black],
        [placement, placement],
        [collage.UNALIGNED, alignment],
    )
    assert result.max() == 0


def test_a_placement_that_already_covers_is_left_alone() -> None:
    layout = Layout(linked=True)
    size = (2000, 1333)
    asked = Placement(zoom=1.5, focus_x=0.48, focus_y=0.52)
    aligned = [collage.UNALIGNED, collage.Alignment(shift_x=0.01)]
    assert collage.linked_placement(layout, asked, [size, size], aligned) == asked


def _covers(rect, cell) -> bool:
    left, top, width, height = rect
    return (
        left <= cell.x + 1e-6
        and top <= cell.y + 1e-6
        and left + width >= cell.x + cell.width - 1e-6
        and top + height >= cell.y + cell.height - 1e-6
    )


# -- the single photos of a carousel -----------------------------------------


@pytest.mark.parametrize("split", [collage.SPLIT_SIDE_BY_SIDE, collage.SPLIT_STACKED])
@pytest.mark.parametrize("count", [2, 3, 4])
def test_the_pieces_are_the_collage_cut_at_its_seams(split: str, count: int) -> None:
    layout = Layout(split=split, count=count, gap=8)
    width, height = layout.canvas_size
    rng = np.random.default_rng(7)
    canvas = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cut = collage.pieces(layout, canvas)
    assert len(cut) == count
    for piece, cell in zip(cut, collage.cells(layout)):
        assert piece.shape == (cell.height, cell.width, 3)
        assert piece.dtype == canvas.dtype
        assert np.array_equal(
            piece, canvas[cell.y : cell.y + cell.height, cell.x : cell.x + cell.width]
        )


def test_the_same_photo_linked_is_cut_into_different_pieces() -> None:
    layout = Layout(count=3, linked=True)
    size = (1800, 600)
    image = _gradient(size)
    placement = collage.linked_placement(layout, Placement(), [size] * 3)
    canvas = collage.render(layout, [image] * 3, [placement] * 3)
    left, middle, right = collage.pieces(layout, canvas)
    assert left.mean() < middle.mean() < right.mean()


def test_the_pieces_are_their_own_pixels_not_views_of_the_canvas() -> None:
    layout = Layout()
    width, height = layout.canvas_size
    canvas = np.zeros((height, width, 3), dtype=np.uint16)
    first = collage.pieces(layout, canvas)[0]
    first[...] = 1
    assert canvas.max() == 0


def test_pieces_of_a_canvas_of_another_size_are_refused() -> None:
    with pytest.raises(ValueError):
        collage.pieces(Layout(), np.zeros((10, 10, 3), dtype=np.uint8))


def test_the_single_photos_are_not_named_like_a_taken_name() -> None:
    assert len(collage.SINGLE_SUFFIXES) == collage.SLOTS
    for suffix in collage.SINGLE_SUFFIXES:
        assert not suffix.lstrip("_").isdigit()


# -- three or four photos (PLAN.md 2.8 K-02, 2026-11-07) ----------------------


@pytest.mark.parametrize(
    ("orientation", "split", "count", "piece"),
    [
        (LANDSCAPE, SIDE, 3, (640, 1080)),
        (LANDSCAPE, SIDE, 4, (480, 1080)),
        (LANDSCAPE, STACKED, 3, (1920, 360)),
        (LANDSCAPE, STACKED, 4, (1920, 270)),
        (PORTRAIT, SIDE, 3, (360, 1350)),
        (PORTRAIT, SIDE, 4, (270, 1350)),
        (PORTRAIT, STACKED, 3, (1080, 450)),
    ],
)
def test_even_pieces_are_the_size_asked_for(
    orientation: str, split: str, count: int, piece: tuple[int, int]
) -> None:
    layout = Layout(canvas=orientation, split=split, count=count, seams=())
    pieces = collage.cells(layout)
    assert len(pieces) == count
    assert all((cell.width, cell.height) == piece for cell in pieces)


def test_four_portrait_rows_share_the_odd_pixels() -> None:
    pieces = collage.cells(Layout(canvas=PORTRAIT, split=STACKED, count=4, seams=()))
    assert sorted(cell.height for cell in pieces) == [337, 337, 338, 338]
    assert sum(cell.height for cell in pieces) == 1350


@pytest.mark.parametrize("count", collage.COUNTS)
@pytest.mark.parametrize("gap", [0, 1, 7, 20])
@pytest.mark.parametrize("split", [SIDE, STACKED])
@pytest.mark.parametrize("orientation", [LANDSCAPE, PORTRAIT])
def test_the_gaps_never_change_the_canvas(
    orientation: str, split: str, gap: int, count: int
) -> None:
    layout = Layout(canvas=orientation, split=split, gap=gap, count=count, seams=())
    pieces = collage.cells(layout)
    width, height = layout.canvas_size
    if split == SIDE:
        assert sum(cell.width for cell in pieces) + gap * (count - 1) == width
        for before, after in zip(pieces, pieces[1:]):
            assert after.x == before.x + before.width + gap
    else:
        assert sum(cell.height for cell in pieces) + gap * (count - 1) == height
        for before, after in zip(pieces, pieces[1:]):
            assert after.y == before.y + before.height + gap


def test_a_count_that_cannot_be_drawn_is_two() -> None:
    assert Layout(count=1).sane().count == 2
    assert Layout(count=5).sane().count == 2
    assert Layout(count="x").sane().count == 2  # type: ignore[arg-type]


def test_seams_that_do_not_fit_the_count_are_even() -> None:
    assert Layout(count=3, seams=(0.5,)).sane().seams == pytest.approx((1 / 3, 2 / 3))
    assert Layout(count=4, seams=()).sane().seams == pytest.approx((0.25, 0.5, 0.75))
    assert Layout(count=3, seams=("a", 0.6)).sane().seams == pytest.approx((1 / 3, 2 / 3))


def test_seams_stay_in_order_and_leave_every_piece_room() -> None:
    seams = Layout(count=4, seams=(0.05, 0.06, 0.99)).sane().seams
    least = collage.PIECE_MIN[4]
    edges = (0.0, *seams, 1.0)
    for before, after in zip(edges, edges[1:]):
        assert after - before >= least - 1e-9


def test_a_dragged_seam_stops_at_its_neighbour() -> None:
    layout = Layout(count=3, seams=())
    moved = collage.with_seam(layout, 0, 0.9)
    assert moved.seams[0] == pytest.approx(2 / 3 - collage.PIECE_MIN[3])
    assert moved.seams[1] == pytest.approx(2 / 3)
    moved = collage.with_seam(layout, 1, 0.0)
    assert moved.seams == pytest.approx((1 / 3, 1 / 3 + collage.PIECE_MIN[3]))


@pytest.mark.parametrize("count", collage.COUNTS)
def test_seam_at_is_the_middle_of_each_gap(count: int) -> None:
    layout = Layout(count=count, seams=(), gap=8)
    pieces = collage.cells(layout)
    for index in range(count - 1):
        assert collage.seam_at(layout, index) == pieces[index].x + pieces[index].width + 4


def test_four_photos_each_land_in_their_own_piece() -> None:
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    layout = Layout(count=4, seams=(), background=collage.BACKGROUND_BLACK, gap=10)
    result = collage.render(layout, [_solid(colour) for colour in colours], [Placement()] * 4)
    assert result.shape == (1080, 1920, 3)
    for cell, colour in zip(collage.cells(layout), colours):
        centre_x, centre_y = cell.centre
        assert tuple(int(value) for value in result[int(centre_y), int(centre_x)]) == colour
    # The gaps between them are the background.
    first = collage.cells(layout)[0]
    assert result[540, first.x + first.width + 5].max() == 0


def test_photos_past_the_count_are_not_drawn() -> None:
    layout = Layout(count=2, background=collage.BACKGROUND_BLACK)
    images = [_solid((255, 255, 255))] * 2 + [_solid((255, 0, 0))] * 2
    result = collage.render(layout, images, [Placement()] * 4)
    assert result.shape == (1080, 1920, 3)
    assert (result[:, :, 1] == 255).all()


def test_aligning_any_of_four_linked_photos_never_leaves_a_bare_strip() -> None:
    layout = Layout(linked=True, count=4, seams=())
    size = (2000, 1333)
    alignments = [
        collage.UNALIGNED,
        collage.Alignment(shift_x=0.02, scale=0.97),
        collage.Alignment(shift_y=-0.02),
        collage.Alignment(shift_x=-0.03, shift_y=0.01, scale=0.95),
    ]
    placement = collage.linked_placement(layout, Placement(), [size] * 4, alignments)
    assert placement.zoom > 1.0
    for slot, cell in enumerate(collage.cells(layout)):
        rect = collage.placed_rect(layout, slot, size, placement, alignments[slot])
        assert _covers(rect, cell)
    black = np.zeros((1333, 2000, 3), dtype=np.uint8)
    result = collage.render(layout, [black] * 4, [placement] * 4, alignments)
    assert result.max() == 0


def test_linked_pieces_show_their_own_part_of_one_scene() -> None:
    """Three photos of one scene: each piece is that part of its own photo."""
    layout = Layout(linked=True, count=3, seams=())
    a = _gradient((1600, 900))
    b = (255 - a).astype(np.uint8)
    linked = collage.render(layout, [a, b, a], [Placement()] * 3)
    # Each photo alone, over far more of the canvas than its own piece.
    whole_a = collage.render(Layout(linked=True, seams=(0.8,)), [a, None], [Placement()] * 2)
    whole_b = collage.render(Layout(linked=True, seams=(0.2,)), [None, b], [Placement()] * 2)
    last_a = collage.render(Layout(linked=True, seams=(0.2,)), [None, a], [Placement()] * 2)
    for slot, whole in enumerate((whole_a, whole_b, last_a)):
        cell = collage.cells(layout)[slot]
        inner = slice(cell.x + 10, cell.x + cell.width - 10)
        assert np.abs(linked[:, inner].astype(int) - whole[:, inner].astype(int)).max() <= 1


# -- the canvases of each social group (PLAN.md 2.8 K-01, 2026-11-07) ---------


@pytest.mark.parametrize(
    ("canvas", "size"),
    [
        (collage.CANVAS_9X16, (1080, 1920)),
        (collage.CANVAS_3X4, (1080, 1440)),
        (collage.CANVAS_4X5, (1080, 1350)),
        (collage.CANVAS_1X1, (1080, 1080)),
        (collage.CANVAS_16X9, (1920, 1080)),
        (collage.CANVAS_191X100, (1080, 566)),
    ],
)
def test_every_canvas_has_its_size(canvas: str, size: tuple[int, int]) -> None:
    platform = next(p for p, shown in collage.PLATFORM_CANVASES.items() if canvas in shown)
    layout = Layout(platform=platform, canvas=canvas)
    assert layout.sane().canvas == canvas
    assert layout.canvas_size == size


def test_each_group_offers_its_own_canvases_best_first() -> None:
    assert collage.PLATFORM_CANVASES[collage.PLATFORM_1] == (
        collage.CANVAS_3X4, collage.CANVAS_4X5, collage.CANVAS_1X1, collage.CANVAS_191X100
    )
    assert collage.PLATFORM_CANVASES[collage.PLATFORM_2] == (
        collage.CANVAS_4X5, collage.CANVAS_9X16, collage.CANVAS_1X1, collage.CANVAS_16X9
    )
    assert set(collage.CANVASES) == {
        canvas for shown in collage.PLATFORM_CANVASES.values() for canvas in shown
    }


def test_a_first_collage_is_sixteen_by_nine() -> None:
    assert Layout().canvas_size == (1920, 1080)
    assert Layout(platform="nowhere").sane().platform == collage.PLATFORM_DEFAULT


def test_a_canvas_its_group_lacks_becomes_the_group_s_best() -> None:
    layout = Layout(platform=collage.PLATFORM_1, canvas=collage.CANVAS_16X9).sane()
    assert layout.canvas == collage.CANVAS_3X4


def test_another_group_keeps_a_canvas_both_have() -> None:
    square = Layout(platform=collage.PLATFORM_2, canvas=collage.CANVAS_1X1)
    assert collage.with_platform(square, collage.PLATFORM_1).canvas == collage.CANVAS_1X1
    wide = Layout(platform=collage.PLATFORM_2, canvas=collage.CANVAS_16X9)
    moved = collage.with_platform(wide, collage.PLATFORM_1)
    assert (moved.platform, moved.canvas) == (collage.PLATFORM_1, collage.CANVAS_3X4)


@pytest.mark.parametrize("count", collage.COUNTS)
@pytest.mark.parametrize("split", [SIDE, STACKED])
@pytest.mark.parametrize("canvas", collage.CANVASES)
def test_every_canvas_splits_into_every_count(canvas: str, split: str, count: int) -> None:
    platform = next(p for p, shown in collage.PLATFORM_CANVASES.items() if canvas in shown)
    layout = Layout(platform=platform, canvas=canvas, split=split, count=count, seams=(), gap=20)
    black = np.zeros((300, 400, 3), dtype=np.uint8)
    result = collage.render(layout, [black] * count, [Placement()] * count)
    width, height = layout.canvas_size
    assert result.shape == (height, width, 3)
    assert all(cell.width > 0 and cell.height > 0 for cell in collage.cells(layout))
